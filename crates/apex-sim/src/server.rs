//! UDP simulation server running the 14-DOF model in real time.
//!
//! Receives driver inputs on one UDP port and sends vehicle state
//! telemetry on another. The simulation loop targets 1kHz with
//! configurable telemetry output rate.

use std::io;
use std::net::UdpSocket;
use std::time::Instant;

use apex_physics::car_params::GRAVITY;
use apex_physics::{
    AeroModel, CarParams, FourteenDofModel, PacejkaTire, Powertrain, SuspensionSystem,
};

use crate::protocol::{InputPacket, OutputPacket};
use crate::realtime::RealtimeIntegrator;

/// Configuration for the simulation server.
pub struct SimServerConfig {
    /// Address to listen for input packets (e.g., "0.0.0.0:20777").
    pub input_addr: String,
    /// Address to send output packets to (e.g., "127.0.0.1:20778").
    pub output_addr: String,
    /// Telemetry output rate in Hz (default: 60).
    pub telemetry_hz: u32,
    /// Maximum steering angle in radians (default: 0.5, ~28 degrees).
    pub max_steer_angle: f64,
}

impl Default for SimServerConfig {
    fn default() -> Self {
        Self {
            input_addr: "0.0.0.0:20777".to_string(),
            output_addr: "127.0.0.1:20778".to_string(),
            telemetry_hz: 60,
            max_steer_angle: 0.5,
        }
    }
}

impl SimServerConfig {
    /// Number of 1kHz frames between telemetry packets (at least 1).
    pub fn telemetry_interval_frames(&self) -> u32 {
        (1000 / self.telemetry_hz.max(1)).max(1)
    }
}

/// Map driver inputs to physical control values for the 14-DOF model.
///
/// The 14-DOF model expects `control = [steering_angle_rad, drive_torque_Nm,
/// brake_pressure_norm]`. This converts the normalized driver inputs to those
/// physical values:
/// - Steering is scaled by `max_steer_angle`.
/// - Throttle is converted to a wheel drive torque (Nm) by the `powertrain`,
///   which selects the optimal gear for the current `drive_wheel_omega`. The
///   14-DOF model splits this total torque across the driven axle internally.
/// - Brake passes through unchanged (already normalized to [0, 1]).
pub fn map_input_to_control(
    input: &InputPacket,
    config: &SimServerConfig,
    powertrain: &mut Powertrain,
    drive_wheel_omega: f64,
) -> [f64; 3] {
    let steering = input.steering as f64 * config.max_steer_angle;
    let drive = powertrain.drive_torque(input.throttle as f64, drive_wheel_omega);
    let brake = input.brake as f64;
    [steering, drive, brake]
}

/// Running state of the simulation.
pub struct SimState {
    /// Current 14-DOF state vector [24].
    pub state: [f64; 24],
    /// Simulation time (s).
    pub sim_time: f64,
    /// Last received input.
    pub last_input: InputPacket,
    /// Output sequence counter.
    pub out_sequence: u32,
    /// Current lap number.
    pub lap: u32,
    /// Current lap start time.
    pub lap_start_time: f64,
    /// Longitudinal acceleration (g); maintained by the server loop.
    pub accel_long: f64,
    /// Lateral acceleration (g); maintained by the server loop.
    pub accel_lat: f64,
}

impl SimState {
    /// Create a sim state initialized to the static-equilibrium state of `model`
    /// at the given forward `speed` (m/s).
    pub fn new(model: &FourteenDofModel, speed: f64) -> Self {
        Self {
            state: equilibrium_state(model, speed),
            sim_time: 0.0,
            last_input: InputPacket::default(),
            out_sequence: 0,
            lap: 1,
            lap_start_time: 0.0,
            accel_long: 0.0,
            accel_lat: 0.0,
        }
    }

    /// Average angular velocity (rad/s) of the rear (driven) wheels.
    pub fn drive_wheel_omega(&self) -> f64 {
        0.5 * (self.state[14] + self.state[15])
    }
}

/// Build the static-equilibrium state vector for `model` at forward `speed`.
///
/// Sets the chassis height, forward velocity, free-rolling wheel speeds, and
/// equilibrium suspension travel; all other states are zero.
pub fn equilibrium_state(model: &FourteenDofModel, speed: f64) -> [f64; 24] {
    let z = model.equilibrium_travel();
    let r = model.params.wheel_radius;
    let w = speed / r;
    let mut s = [0.0; 24];
    s[2] = model.params.cog_height;
    s[6] = speed;
    s[12] = w;
    s[13] = w;
    s[14] = w;
    s[15] = w;
    s[16] = z[0];
    s[17] = z[1];
    s[18] = z[2];
    s[19] = z[3];
    s
}

/// Build an output packet from the current simulation state.
///
/// Maps the 24-element 14-DOF state vector to the telemetry packet fields.
/// State layout (see [`FourteenDofModel`]): indices 0..6 are position/attitude,
/// 6..12 body velocities and angular rates (yaw rate is index 11), 12..16 wheel
/// spins, 16..20 suspension travel. The acceleration fields are read from the
/// running [`SimState`] (the loop updates them each frame).
pub fn build_output(sim: &SimState) -> OutputPacket {
    let s = &sim.state;
    OutputPacket {
        pos_x: s[0],
        pos_y: s[1],
        pos_z: s[2],
        roll: s[3],
        pitch: s[4],
        yaw: s[5],
        speed: s[6],
        lateral_v: s[7],
        vertical_v: s[8],
        yaw_rate: s[11],
        wheel_fl: s[12],
        wheel_fr: s[13],
        wheel_rl: s[14],
        wheel_rr: s[15],
        susp_fl: s[16],
        susp_fr: s[17],
        susp_rl: s[18],
        susp_rr: s[19],
        accel_long: sim.accel_long,
        accel_lat: sim.accel_lat,
        gear: sim.last_input.gear,
        lap: sim.lap,
        lap_time: sim.sim_time - sim.lap_start_time,
        sim_time: sim.sim_time,
        sequence: sim.out_sequence,
        _pad: 0,
    }
}

/// Advance the simulation by one frame.
///
/// This is the core logic extracted from the server loop for testability.
/// Steps the integrator by one frame, advances the simulation clock, and
/// updates the derived longitudinal/lateral acceleration telemetry (in g).
/// Returns true if a telemetry packet should be sent this frame.
pub fn step_frame(
    sim: &mut SimState,
    integrator: &RealtimeIntegrator,
    model: &FourteenDofModel,
    control: &[f64; 3],
    telemetry_interval_frames: u32,
    frame_counter: &mut u32,
) -> bool {
    let prev_vx = sim.state[6];
    sim.state = integrator.step(model, sim.sim_time, &sim.state, control);
    let dt = integrator.target_dt();
    sim.sim_time += dt;

    // Derived telemetry: longitudinal accel from the change in forward speed,
    // lateral accel from the centripetal term (forward speed × yaw rate).
    sim.accel_long = (sim.state[6] - prev_vx) / dt / GRAVITY;
    sim.accel_lat = sim.state[6] * sim.state[11] / GRAVITY;

    *frame_counter = frame_counter.wrapping_add(1);
    frame_counter.is_multiple_of(telemetry_interval_frames)
}

/// Run the simulation server.
///
/// This function blocks and runs the simulation loop until an error
/// occurs or the process is terminated. It:
/// 1. Binds a UDP socket for receiving input.
/// 2. Sets the input socket to non-blocking mode.
/// 3. Runs the 14-DOF model at 1kHz using [`RealtimeIntegrator`].
/// 4. Polls for input packets each frame (non-blocking recv).
/// 5. Sends output packets at the configured telemetry rate.
///
/// The simulation uses spin-waiting for precise timing. This will
/// use 100% of one CPU core.
pub fn run_server(
    config: SimServerConfig,
    car: &CarParams,
    tire: &PacejkaTire,
    suspension: &SuspensionSystem,
    aero: &AeroModel,
) -> io::Result<()> {
    /// Forward speed (m/s) the model is trimmed for and started at.
    const REFERENCE_SPEED: f64 = 50.0;

    let model = FourteenDofModel::new(car, tire, suspension, aero, REFERENCE_SPEED);
    let integrator = RealtimeIntegrator::new_1khz();
    let mut powertrain = Powertrain::f1_2024();

    let input_socket = UdpSocket::bind(&config.input_addr)?;
    input_socket.set_nonblocking(true)?;
    let output_socket = UdpSocket::bind("0.0.0.0:0")?;

    let mut sim = SimState::new(&model, REFERENCE_SPEED);
    let telemetry_interval = config.telemetry_interval_frames();
    let mut frame_counter: u32 = 0;
    let frame_budget = integrator.target_dt();

    let mut recv_buf = [0u8; InputPacket::SIZE];
    loop {
        let frame_start = Instant::now();

        // (b,c) Poll for the latest input packet (drain the queue).
        loop {
            match input_socket.recv_from(&mut recv_buf) {
                Ok((n, _)) => {
                    if let Some(mut pkt) = InputPacket::from_bytes(&recv_buf[..n]) {
                        pkt.clamp();
                        sim.last_input = pkt;
                    }
                }
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) => return Err(e),
            }
        }

        // (d) Map input to control and (e,f) step the simulation.
        let control = map_input_to_control(
            &sim.last_input,
            &config,
            &mut powertrain,
            sim.drive_wheel_omega(),
        );
        let send_telemetry = step_frame(
            &mut sim,
            &integrator,
            &model,
            &control,
            telemetry_interval,
            &mut frame_counter,
        );

        // (g) Emit telemetry at the configured rate.
        if send_telemetry {
            sim.out_sequence = sim.out_sequence.wrapping_add(1);
            let out = build_output(&sim);
            output_socket.send_to(&out.to_bytes(), &config.output_addr)?;
        }

        // (h) Spin-wait until the frame budget has elapsed.
        while frame_start.elapsed().as_secs_f64() < frame_budget {
            std::hint::spin_loop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rig {
        params: CarParams,
        tire: PacejkaTire,
        susp: SuspensionSystem,
        aero: AeroModel,
    }

    fn rig() -> Rig {
        Rig {
            params: CarParams::default(),
            tire: PacejkaTire::f1_default(),
            susp: SuspensionSystem::f1_default(),
            aero: AeroModel::f1_default(),
        }
    }

    fn model_for<'a>(rg: &'a Rig, speed: f64) -> FourteenDofModel<'a> {
        FourteenDofModel::new(&rg.params, &rg.tire, &rg.susp, &rg.aero, speed)
    }

    #[test]
    fn test_map_input_to_control() {
        let config = SimServerConfig::default();
        let mut powertrain = Powertrain::f1_2024();

        // Full right steering, no throttle, half brake.
        let input = InputPacket {
            steering: 1.0,
            throttle: 0.0,
            brake: 0.5,
            gear: 3,
            sequence: 0,
        };
        let control = map_input_to_control(&input, &config, &mut powertrain, 50.0 / 0.330);
        assert!((control[0] - config.max_steer_angle).abs() < 1e-12);
        assert_eq!(
            control[1], 0.0,
            "zero throttle should give zero drive torque"
        );
        assert!((control[2] - 0.5).abs() < 1e-12);

        // Full left steering -> negative max angle.
        let left = InputPacket {
            steering: -1.0,
            ..Default::default()
        };
        let control = map_input_to_control(&left, &config, &mut powertrain, 50.0 / 0.330);
        assert!((control[0] + config.max_steer_angle).abs() < 1e-12);

        // Full throttle produces positive drive torque.
        let gas = InputPacket {
            throttle: 1.0,
            ..Default::default()
        };
        let control = map_input_to_control(&gas, &config, &mut powertrain, 50.0 / 0.330);
        assert!(
            control[1] > 0.0,
            "full throttle should give positive torque"
        );
    }

    #[test]
    fn test_build_output() {
        let rg = rig();
        let model = model_for(&rg, 50.0);
        let mut sim = SimState::new(&model, 50.0);
        sim.state[0] = 12.0; // pos_x
        sim.state[5] = 0.1; // yaw
        sim.state[11] = 0.25; // yaw rate
        sim.sim_time = 5.0;
        sim.lap_start_time = 2.0;
        sim.lap = 2;
        sim.out_sequence = 42;
        sim.last_input.gear = 5;
        sim.accel_long = -1.1;
        sim.accel_lat = 2.2;

        let out = build_output(&sim);
        assert_eq!(out.pos_x, 12.0);
        assert_eq!(out.yaw, 0.1);
        assert_eq!(out.yaw_rate, 0.25);
        assert_eq!(out.speed, 50.0);
        assert_eq!(out.wheel_fl, sim.state[12]);
        assert_eq!(out.susp_fl, sim.state[16]);
        assert_eq!(out.gear, 5);
        assert_eq!(out.lap, 2);
        assert_eq!(out.sequence, 42);
        assert!((out.lap_time - 3.0).abs() < 1e-12);
        assert_eq!(out.sim_time, 5.0);
        assert_eq!(out.accel_long, -1.1);
        assert_eq!(out.accel_lat, 2.2);
    }

    #[test]
    fn test_step_frame_advances_time() {
        let rg = rig();
        let model = model_for(&rg, 50.0);
        let integ = RealtimeIntegrator::new_1khz();
        let mut sim = SimState::new(&model, 50.0);
        let mut counter = 0;

        let t0 = sim.sim_time;
        step_frame(&mut sim, &integ, &model, &[0.0, 0.0, 0.0], 16, &mut counter);
        assert!((sim.sim_time - t0 - integ.target_dt()).abs() < 1e-15);
    }

    #[test]
    fn test_step_frame_telemetry_interval() {
        let rg = rig();
        let model = model_for(&rg, 50.0);
        let integ = RealtimeIntegrator::new_1khz();
        let mut sim = SimState::new(&model, 50.0);
        let interval: u32 = 16;
        let mut counter = 0;

        for frame in 1u32..=64 {
            let due = step_frame(
                &mut sim,
                &integ,
                &model,
                &[0.0, 0.0, 0.0],
                interval,
                &mut counter,
            );
            assert_eq!(
                due,
                frame.is_multiple_of(interval),
                "frame {frame} telemetry due mismatch"
            );
        }
    }

    #[test]
    fn test_zero_input_stability() {
        let rg = rig();
        let model = model_for(&rg, 50.0);
        let integ = RealtimeIntegrator::new_1khz();
        let mut sim = SimState::new(&model, 50.0);
        let mut counter = 0;

        for _ in 0..1000 {
            step_frame(&mut sim, &integ, &model, &[0.0, 0.0, 0.0], 16, &mut counter);
        }
        assert!(
            sim.state.iter().all(|v| v.is_finite()),
            "state went non-finite over 1000 frames"
        );
    }

    #[test]
    fn test_throttle_increases_speed() {
        let rg = rig();
        let model = model_for(&rg, 30.0);
        let integ = RealtimeIntegrator::new_1khz();
        let config = SimServerConfig::default();
        let mut powertrain = Powertrain::f1_2024();
        let mut sim = SimState::new(&model, 30.0);
        let mut counter = 0;

        let v0 = sim.state[6];
        let input = InputPacket {
            throttle: 1.0,
            ..Default::default()
        };
        for _ in 0..500 {
            let control =
                map_input_to_control(&input, &config, &mut powertrain, sim.drive_wheel_omega());
            step_frame(&mut sim, &integ, &model, &control, 16, &mut counter);
        }
        assert!(
            sim.state.iter().all(|v| v.is_finite()),
            "state went non-finite under throttle"
        );
        assert!(
            sim.state[6] > v0,
            "full throttle should increase speed: {v0} -> {}",
            sim.state[6]
        );
    }

    #[test]
    fn test_steering_produces_yaw() {
        let rg = rig();
        let model = model_for(&rg, 30.0);
        let integ = RealtimeIntegrator::new_1khz();
        let config = SimServerConfig::default();
        let mut powertrain = Powertrain::f1_2024();
        let mut sim = SimState::new(&model, 30.0);
        let mut counter = 0;

        let input = InputPacket {
            steering: 0.5,
            throttle: 0.5,
            ..Default::default()
        };
        for _ in 0..500 {
            let control =
                map_input_to_control(&input, &config, &mut powertrain, sim.drive_wheel_omega());
            step_frame(&mut sim, &integ, &model, &control, 16, &mut counter);
        }
        assert!(
            sim.state.iter().all(|v| v.is_finite()),
            "state went non-finite under steering"
        );
        assert!(
            sim.state[11].abs() > 1e-3,
            "steering should produce a nonzero yaw rate, got {}",
            sim.state[11]
        );
    }
}
