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
use apex_track::Track;

use crate::protocol::{InputPacket, OutputPacket};
use crate::realtime::RealtimeIntegrator;
use crate::shared_mem::SimSharedMem;

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
///   torque is then capped at the traction limit (see below) before being
///   handed to the model, which splits it across the driven axle internally.
/// - Brake passes through unchanged (already normalized to [0, 1]).
///
/// ## Traction limiting
///
/// The powertrain's raw output in a low gear (over 11 kN·m) far exceeds what
/// the tires can transmit. Feeding it directly makes the driven wheels spin up
/// to several times road speed. That stored wheel momentum then keeps the car
/// accelerating for a noticeable window *after* the throttle is released and
/// the brake applied, so braking appears not to work. Capping the drive torque
/// at the per-axle traction limit (`mu * axle_load * wheel_radius`, accounting
/// for the drive split) keeps the wheels tracking road speed, so the brake
/// bites immediately. `speed` is the vehicle's forward speed (m/s), used for
/// the downforce-dependent axle loads.
pub fn map_input_to_control(
    input: &InputPacket,
    config: &SimServerConfig,
    powertrain: &mut Powertrain,
    car: &CarParams,
    grip_mu: f64,
    drive_wheel_omega: f64,
    speed: f64,
) -> [f64; 3] {
    let steering = input.steering as f64 * config.max_steer_angle;
    let raw_drive = powertrain.drive_torque(input.throttle as f64, drive_wheel_omega);
    let drive = raw_drive.min(traction_torque_limit(car, grip_mu, speed));
    let brake = input.brake as f64;
    [steering, drive, brake]
}

/// Fraction of the grip-limited torque the cap allows.
///
/// The driven wheels have small rotational inertia, so with explicit 1kHz
/// integration a torque near the grip peak lets the wheel overshoot the
/// peak-slip point and run away (unbounded wheelspin), which is what defeated
/// braking. Capping at 60% of the grip-limited torque places the steady-state
/// operating point well down the rising (stable) side of the longitudinal slip
/// curve (slip ~0.04 vs a peak near 0.1), so the wheels track road speed and
/// the brake bites immediately while acceleration stays strong.
const TRACTION_MARGIN: f64 = 0.6;

/// Maximum total drive torque (Nm) the driven axle(s) can transmit at `speed`.
///
/// The model distributes the total drive torque to the axles by
/// `drive_distribution` (fraction to the rear). For each driven axle the torque
/// is bounded by `grip_mu * axle_load * wheel_radius` (times [`TRACTION_MARGIN`]);
/// the overall cap is the tightest of those bounds so neither axle is asked to
/// exceed its grip. `grip_mu` is the tire's actual peak longitudinal friction
/// coefficient (see [`tire_peak_longitudinal_mu`]), which is lower than the
/// point-mass `car.tire_mu`.
fn traction_torque_limit(car: &CarParams, grip_mu: f64, speed: f64) -> f64 {
    let (front_load, rear_load) = car.axle_loads(speed, 0.0);
    let r = car.wheel_radius;
    let dd = car.drive_distribution;

    let rear_cap = if dd > 0.0 {
        TRACTION_MARGIN * grip_mu * rear_load * r / dd
    } else {
        f64::INFINITY
    };
    let front_cap = if dd < 1.0 {
        TRACTION_MARGIN * grip_mu * front_load * r / (1.0 - dd)
    } else {
        f64::INFINITY
    };
    rear_cap.min(front_cap)
}

/// Estimate the tire's peak longitudinal friction coefficient (`max fx / Fz`)
/// at the reference load `fz`, by sweeping the slip ratio.
///
/// Computed once at startup; the 14-DOF Pacejka tire peaks below the nominal
/// `CarParams::tire_mu`, so this gives an accurate traction cap.
pub fn tire_peak_longitudinal_mu(tire: &PacejkaTire, fz: f64) -> f64 {
    if fz <= 0.0 {
        return 0.0;
    }
    let mut peak = 0.0_f64;
    for i in 0..=200 {
        let sr = i as f64 * 0.01;
        peak = peak.max(tire.combined_forces_smooth(0.0, sr, fz).fx);
    }
    peak / fz
}

/// Track-relative position state.
#[derive(Debug, Clone, Copy)]
pub struct TrackPosition {
    /// Distance along the track centerline (m), wrapping at `track_length`.
    pub distance: f64,
    /// Lateral offset from centerline (m), positive = right.
    pub lateral_offset: f64,
    /// Track total length (m).
    pub track_length: f64,
    /// Whether the car has crossed the start/finish line this frame.
    pub crossed_start_finish: bool,
    /// Simulation time of the last lap crossing (for cooldown).
    pub last_crossing_time: f64,
    /// Minimum time between lap crossings (s).
    pub min_lap_time: f64,
}

/// Default minimum time between lap crossings (s). No real circuit lap is
/// faster than this, so crossings closer together are projection artifacts.
pub const DEFAULT_MIN_LAP_TIME: f64 = 10.0;

/// Lateral offset (m) beyond which the car is treated as off-track and lap
/// crossings are suppressed (the nearest-point projection is unreliable there).
const OFF_TRACK_OFFSET: f64 = 50.0;

/// Project a world position onto the track centerline.
///
/// Finds the closest point on the track and returns
/// `(distance_along_track, lateral_offset)`, where the offset is positive to the
/// right of the track direction and negative to the left.
///
/// Uses a two-phase approach:
/// 1. Coarse search: scan candidate segments, find the closest one.
/// 2. Fine search: project onto that segment chord for sub-segment precision.
///
/// The `hint_distance` (last known distance) restricts the scan to a window of
/// segments around it for O(1) amortized cost at 1kHz. If the best match in that
/// window is implausibly far (more than [`PROJECT_FALLBACK_DIST`] from the
/// centerline), the search falls back to a full O(N) scan so the tracker can
/// recover if the car teleports or goes far off-track.
pub fn project_onto_track(track: &Track, x: f64, y: f64, hint_distance: f64) -> (f64, f64) {
    let segs = &track.segments;
    let n = segs.len();
    if n < 2 {
        return (0.0, 0.0);
    }
    let last = if track.is_closed { n } else { n - 1 };

    // Coarse window centered on the segment nearest the hint distance.
    let (hint_idx, _) = track.locate(hint_distance);
    let (mut best_s, mut best_d, best_d2) =
        scan_window(track, x, y, hint_idx, PROJECT_WINDOW, last);

    // Fall back to a full scan if the windowed match looks implausibly far.
    if best_d2.sqrt() > PROJECT_FALLBACK_DIST {
        let (s, d, d2) = scan_window(track, x, y, 0, last, last);
        if d2 < best_d2 {
            best_s = s;
            best_d = d;
        }
    }
    (best_s, best_d)
}

/// Half-width (in segments) of the windowed projection scan.
const PROJECT_WINDOW: usize = 50;
/// Lateral distance (m) beyond which a windowed projection is deemed unreliable
/// and a full-track scan is performed instead.
const PROJECT_FALLBACK_DIST: f64 = 50.0;

/// Scan `2 * half + 1` segments centered on `center_idx` (wrapping on closed
/// tracks) and return the `(distance, lateral_offset, squared_distance)` of the
/// closest point on those segment chords to `(x, y)`.
fn scan_window(
    track: &Track,
    x: f64,
    y: f64,
    center_idx: usize,
    half: usize,
    last: usize,
) -> (f64, f64, f64) {
    let segs = &track.segments;
    let n = segs.len();

    let mut best_s = 0.0;
    let mut best_off = 0.0;
    let mut best_d2 = f64::INFINITY;

    // Number of distinct segment chords to consider.
    let count = (2 * half + 1).min(last);
    let start = center_idx as isize - half as isize;
    for k in 0..count {
        let i = (start + k as isize).rem_euclid(last as isize) as usize;
        let a = &segs[i];
        let j = (i + 1) % n;
        let b = &segs[j];
        let ex = b.x - a.x;
        let ey = b.y - a.y;
        let len2 = ex * ex + ey * ey;
        let t = if len2 > 1e-12 {
            (((x - a.x) * ex + (y - a.y) * ey) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let px = a.x + t * ex;
        let py = a.y + t * ey;
        let d2 = (x - px) * (x - px) + (y - py) * (y - py);
        if d2 < best_d2 {
            best_d2 = d2;
            let s_b = if j == 0 { track.total_length } else { b.s };
            best_s = a.s + t * (s_b - a.s);
            // Lateral offset: positive to the right of the (unit) tangent.
            // cross = tangent x car_vec is positive on the left, so negate it.
            let inv_len = if len2 > 1e-12 { 1.0 / len2.sqrt() } else { 0.0 };
            let (tx, ty) = (ex * inv_len, ey * inv_len);
            best_off = (x - px) * ty - (y - py) * tx;
        }
    }
    (best_s, best_off, best_d2)
}

/// Update the car's track position and lap counting from its world position.
///
/// Projects `(x, y)` onto the centerline (using the previous distance as a
/// locality hint), updates `track_pos`, and detects a start/finish crossing
/// when the distance wraps from near the end of the lap back to near the start.
/// On a crossing it increments `sim.lap` and resets `sim.lap_start_time` to the
/// current simulation time so the reported lap time restarts.
///
/// Two guards prevent the projection from spamming lap increments when the car
/// is far off-track (where the nearest-point can jump erratically between
/// distant segments and repeatedly satisfy the wrap condition):
/// - a cooldown that ignores crossings within `min_lap_time` of the last one
///   (no real lap is that fast), and
/// - off-track suppression that ignores crossings while the lateral offset
///   exceeds [`OFF_TRACK_OFFSET`].
fn update_track_position(sim: &mut SimState, track: &Track, x: f64, y: f64) {
    let Some(tp) = sim.track_pos.as_mut() else {
        return;
    };
    let length = tp.track_length;
    let prev = tp.distance;
    let (s, off) = project_onto_track(track, x, y, prev);

    // Start/finish crossing: the distance wrapped from near the end of the lap
    // back to near the start, the car is on-track, and the cooldown has elapsed.
    let time_since_crossing = sim.sim_time - tp.last_crossing_time;
    let crossed = length > 0.0
        && prev > 0.8 * length
        && s < 0.2 * length
        && off.abs() < OFF_TRACK_OFFSET
        && time_since_crossing > tp.min_lap_time;

    // Always update the projected position so the next frame has a good hint,
    // even when a crossing is suppressed.
    tp.distance = s;
    tp.lateral_offset = off;
    tp.crossed_start_finish = crossed;

    if crossed {
        tp.last_crossing_time = sim.sim_time;
        sim.lap = sim.lap.saturating_add(1);
        sim.lap_start_time = sim.sim_time;
    }
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
    /// Track-relative position, when a track is attached. `None` runs the sim
    /// without lap tracking (e.g. in tests).
    pub track_pos: Option<TrackPosition>,
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
            track_pos: None,
        }
    }

    /// Place the car at the track's start line and enable lap tracking.
    ///
    /// Sets the world position and yaw to the track start, and initializes
    /// [`TrackPosition`] so subsequent frames project onto the centerline.
    pub fn attach_track(&mut self, track: &Track) {
        let (x0, y0) = track.position_at(0.0);
        self.state[0] = x0;
        self.state[1] = y0;
        self.state[5] = track.heading_at(0.0);
        self.track_pos = Some(TrackPosition {
            distance: 0.0,
            lateral_offset: 0.0,
            track_length: track.total_length,
            crossed_start_finish: false,
            // Negative infinity so the very first crossing is never on cooldown.
            last_crossing_time: f64::NEG_INFINITY,
            min_lap_time: DEFAULT_MIN_LAP_TIME,
        });
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
        track_distance: sim.track_pos.map(|tp| tp.distance).unwrap_or(0.0),
        track_offset: sim.track_pos.map(|tp| tp.lateral_offset).unwrap_or(0.0),
        _pad: 0,
    }
}

/// Advance the simulation by one frame.
///
/// This is the core logic extracted from the server loop for testability.
/// Steps the integrator by one frame, advances the simulation clock, updates
/// the derived longitudinal/lateral acceleration telemetry (in g), and, when a
/// `track` is supplied, updates the track-relative position and lap counting.
/// Returns true if a telemetry packet should be sent this frame.
pub fn step_frame(
    sim: &mut SimState,
    integrator: &RealtimeIntegrator,
    model: &FourteenDofModel,
    control: &[f64; 3],
    track: Option<&Track>,
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

    // Track-relative position and lap counting (only when a track is attached).
    if let Some(track) = track {
        let (x, y) = (sim.state[0], sim.state[1]);
        update_track_position(sim, track, x, y);
    }

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
///
/// When `shared_mem` is `Some`, driver inputs are also accepted from the
/// shared memory region (whenever its input sequence advances) and every
/// telemetry frame is mirrored into it alongside the UDP send.
///
/// When `track` is `Some`, the car starts at the track's start line and each
/// frame projects its world position onto the centerline to report track
/// distance/offset and count laps. When `None`, lap tracking is skipped.
pub fn run_server(
    config: SimServerConfig,
    car: &CarParams,
    tire: &PacejkaTire,
    suspension: &SuspensionSystem,
    aero: &AeroModel,
    track: Option<&Track>,
    mut shared_mem: Option<SimSharedMem>,
) -> io::Result<()> {
    /// Forward speed (m/s) the model is trimmed for and started at.
    const REFERENCE_SPEED: f64 = 50.0;

    let model = FourteenDofModel::new(car, tire, suspension, aero, REFERENCE_SPEED);
    let integrator = RealtimeIntegrator::new_1khz();
    let mut powertrain = Powertrain::f1_2024();

    // Peak tire grip (computed once) used to traction-limit the drive torque.
    let grip_mu = tire_peak_longitudinal_mu(tire, car.mass * GRAVITY / 4.0);

    let input_socket = UdpSocket::bind(&config.input_addr)?;
    input_socket.set_nonblocking(true)?;
    let output_socket = UdpSocket::bind("0.0.0.0:0")?;

    let mut sim = SimState::new(&model, REFERENCE_SPEED);
    if let Some(track) = track {
        sim.attach_track(track);
    }
    let telemetry_interval = config.telemetry_interval_frames();
    let mut frame_counter: u32 = 0;
    let frame_budget = integrator.target_dt();
    let mut last_shmem_input_seq: u32 = 0;

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

        // Also accept input from shared memory when its sequence advances.
        if let Some(shmem) = &shared_mem {
            let seq = shmem.read_input_sequence();
            if seq != last_shmem_input_seq {
                last_shmem_input_seq = seq;
                if let Some(mut pkt) = shmem.read_input() {
                    pkt.clamp();
                    sim.last_input = pkt;
                }
            }
        }

        // (d) Map input to control and (e,f) step the simulation.
        let control = map_input_to_control(
            &sim.last_input,
            &config,
            &mut powertrain,
            car,
            grip_mu,
            sim.drive_wheel_omega(),
            sim.state[6],
        );
        let send_telemetry = step_frame(
            &mut sim,
            &integrator,
            &model,
            &control,
            track,
            telemetry_interval,
            &mut frame_counter,
        );

        // (g) Emit telemetry at the configured rate.
        if send_telemetry {
            sim.out_sequence = sim.out_sequence.wrapping_add(1);
            let out = build_output(&sim);
            output_socket.send_to(&out.to_bytes(), &config.output_addr)?;
            if let Some(shmem) = &mut shared_mem {
                shmem.write_output(&out);
            }
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
    use apex_track::{build_track, circle_track, TrackPoint};

    /// A simple 100 m straight track from (0,0) to (100,0).
    fn straight_track() -> Track {
        let pts = vec![
            TrackPoint {
                x: 0.0,
                y: 0.0,
                width_left: 5.0,
                width_right: 5.0,
            },
            TrackPoint {
                x: 100.0,
                y: 0.0,
                width_left: 5.0,
                width_right: 5.0,
            },
        ];
        build_track("straight", &pts, false)
    }

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
        let rg = rig();
        let car = &rg.params;
        let grip_mu = tire_peak_longitudinal_mu(&rg.tire, car.mass * 9.81 / 4.0);
        let mut powertrain = Powertrain::f1_2024();
        let omega = 50.0 / 0.330;

        // Full right steering, no throttle, half brake.
        let input = InputPacket {
            steering: 1.0,
            throttle: 0.0,
            brake: 0.5,
            gear: 3,
            sequence: 0,
        };
        let control =
            map_input_to_control(&input, &config, &mut powertrain, car, grip_mu, omega, 50.0);
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
        let control =
            map_input_to_control(&left, &config, &mut powertrain, car, grip_mu, omega, 50.0);
        assert!((control[0] + config.max_steer_angle).abs() < 1e-12);

        // Full throttle produces positive drive torque, capped at the traction
        // limit (never exceeding the powertrain's raw output).
        let gas = InputPacket {
            throttle: 1.0,
            ..Default::default()
        };
        let control =
            map_input_to_control(&gas, &config, &mut powertrain, car, grip_mu, omega, 50.0);
        assert!(
            control[1] > 0.0,
            "full throttle should give positive torque"
        );
        assert!(
            control[1] <= traction_torque_limit(car, grip_mu, 50.0) + 1e-9,
            "drive torque should be capped at the traction limit"
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
        step_frame(
            &mut sim,
            &integ,
            &model,
            &[0.0, 0.0, 0.0],
            None,
            16,
            &mut counter,
        );
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
                None,
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
            step_frame(
                &mut sim,
                &integ,
                &model,
                &[0.0, 0.0, 0.0],
                None,
                16,
                &mut counter,
            );
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

        let grip_mu = tire_peak_longitudinal_mu(&rg.tire, rg.params.mass * 9.81 / 4.0);
        let v0 = sim.state[6];
        let input = InputPacket {
            throttle: 1.0,
            ..Default::default()
        };
        for _ in 0..500 {
            let control = map_input_to_control(
                &input,
                &config,
                &mut powertrain,
                &rg.params,
                grip_mu,
                sim.drive_wheel_omega(),
                sim.state[6],
            );
            step_frame(&mut sim, &integ, &model, &control, None, 16, &mut counter);
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
    fn test_braking_reduces_speed() {
        let rg = rig();
        let model = model_for(&rg, 40.0);
        let integ = RealtimeIntegrator::new_1khz();
        let config = SimServerConfig::default();
        let mut powertrain = Powertrain::f1_2024();
        let mut sim = SimState::new(&model, 40.0);
        let mut counter = 0;
        let grip_mu = tire_peak_longitudinal_mu(&rg.tire, rg.params.mass * 9.81 / 4.0);

        // Phase 1: full throttle to build speed.
        let throttle_input = InputPacket {
            throttle: 1.0,
            ..Default::default()
        };
        for _ in 0..500 {
            let control = map_input_to_control(
                &throttle_input,
                &config,
                &mut powertrain,
                &rg.params,
                grip_mu,
                sim.drive_wheel_omega(),
                sim.state[6],
            );
            step_frame(&mut sim, &integ, &model, &control, None, 16, &mut counter);
        }
        let v_brake_point = sim.state[6];

        // Phase 2: full brake, zero throttle.
        let brake_input = InputPacket {
            throttle: 0.0,
            brake: 1.0,
            ..Default::default()
        };
        for _ in 0..500 {
            let control = map_input_to_control(
                &brake_input,
                &config,
                &mut powertrain,
                &rg.params,
                grip_mu,
                sim.drive_wheel_omega(),
                sim.state[6],
            );
            step_frame(&mut sim, &integ, &model, &control, None, 16, &mut counter);
        }
        let v_after_brake = sim.state[6];

        assert!(
            sim.state.iter().all(|v| v.is_finite()),
            "state went non-finite during throttle/brake"
        );
        // Braking must decelerate the car, and strongly: over 0.5 s of full
        // F1 braking we expect to shed well over 5 m/s, not 1-2 m/s.
        assert!(
            v_after_brake < v_brake_point - 5.0,
            "full brake should strongly reduce speed: {v_brake_point:.1} -> {v_after_brake:.1} m/s"
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

        let grip_mu = tire_peak_longitudinal_mu(&rg.tire, rg.params.mass * 9.81 / 4.0);
        let input = InputPacket {
            steering: 0.5,
            throttle: 0.5,
            ..Default::default()
        };
        for _ in 0..500 {
            let control = map_input_to_control(
                &input,
                &config,
                &mut powertrain,
                &rg.params,
                grip_mu,
                sim.drive_wheel_omega(),
                sim.state[6],
            );
            step_frame(&mut sim, &integ, &model, &control, None, 16, &mut counter);
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

    #[test]
    fn test_project_onto_track_straight() {
        let track = straight_track();
        let (dist, offset) = project_onto_track(&track, 50.0, 5.0, 0.0);
        assert!((dist - 50.0).abs() < 1e-6, "distance {dist}");
        // (50, 5) is to the LEFT of the +X track direction, so offset is negative
        // under the positive-is-right convention.
        assert!((offset - -5.0).abs() < 1e-6, "offset {offset}");
    }

    #[test]
    fn test_project_onto_track_circle() {
        let radius = 100.0;
        let (pts, closed) = circle_track(radius, 10.0, 200);
        let track = build_track("circle", &pts, closed);

        // The centre projects to some point on the loop, a radius away on the
        // inside (left of the CCW direction), so the offset is about -radius.
        let (_dist, offset) = project_onto_track(&track, 0.0, 0.0, 0.0);
        assert!(
            (offset + radius).abs() < 1.0,
            "offset {offset} should be about -{radius}"
        );
    }

    #[test]
    fn test_project_onto_track_hint() {
        let radius = 100.0;
        let (pts, closed) = circle_track(radius, 10.0, 200);
        let track = build_track("circle", &pts, closed);

        // A point on the centreline half-way round the lap.
        let s_true = 0.5 * track.total_length;
        let (x, y) = track.position_at(s_true);

        // A good hint (near the true distance) and a bad hint (start line, far
        // away, forcing the full-scan fallback) must agree.
        let good = project_onto_track(&track, x, y, s_true);
        let bad = project_onto_track(&track, x, y, 0.0);
        assert!((good.0 - bad.0).abs() < 1e-6, "{good:?} vs {bad:?}");
        assert!((good.1 - bad.1).abs() < 1e-6, "{good:?} vs {bad:?}");
        // And the hinted result is the expected arc length (offset ~0).
        assert!((good.0 - s_true).abs() < 1.0, "distance {}", good.0);
        assert!(good.1.abs() < 0.5, "offset {}", good.1);
    }

    #[test]
    fn test_lap_crossing_detection() {
        let radius = 100.0;
        let (pts, closed) = circle_track(radius, 10.0, 200);
        let track = build_track("circle", &pts, closed);
        let l = track.total_length;

        let rg = rig();
        let model = model_for(&rg, 30.0);
        let mut sim = SimState::new(&model, 30.0);
        sim.attach_track(&track);
        assert_eq!(sim.lap, 1);

        // Walk the car forward along the centreline toward the end of the lap.
        for frac in [0.5, 0.8, 0.92, 0.96] {
            let (x, y) = track.position_at(frac * l);
            update_track_position(&mut sim, &track, x, y);
            assert_eq!(sim.lap, 1, "no crossing before wrap (frac {frac})");
        }

        // Wrap back past the start/finish line: the lap count must increment.
        let (x, y) = track.position_at(0.02 * l);
        update_track_position(&mut sim, &track, x, y);
        assert_eq!(sim.lap, 2, "lap should increment on start/finish crossing");
        assert!(
            sim.track_pos.expect("track pos").crossed_start_finish,
            "crossing flag should be set on the crossing frame"
        );
    }

    #[test]
    fn test_lap_time_recorded() {
        let radius = 100.0;
        let (pts, closed) = circle_track(radius, 10.0, 200);
        let track = build_track("circle", &pts, closed);
        let l = track.total_length;

        let rg = rig();
        let model = model_for(&rg, 30.0);
        let mut sim = SimState::new(&model, 30.0);
        sim.attach_track(&track);

        // Approach the line, then cross it at t = 10 s.
        sim.sim_time = 10.0;
        let (x, y) = track.position_at(0.95 * l);
        update_track_position(&mut sim, &track, x, y);
        let (x, y) = track.position_at(0.02 * l);
        update_track_position(&mut sim, &track, x, y);
        assert_eq!(sim.lap, 2);
        assert!((sim.lap_start_time - 10.0).abs() < 1e-9);

        // 60 s later, the reported lap time is the elapsed time on this lap.
        sim.sim_time = 70.0;
        let out = build_output(&sim);
        assert!(
            (out.lap_time - 60.0).abs() < 1e-9,
            "lap_time {}",
            out.lap_time
        );

        // Cross again at t = 130 s: lap time resets and the lap advances.
        let (x, y) = track.position_at(0.95 * l);
        update_track_position(&mut sim, &track, x, y);
        sim.sim_time = 130.0;
        let (x, y) = track.position_at(0.02 * l);
        update_track_position(&mut sim, &track, x, y);
        assert_eq!(sim.lap, 3);
        assert!((sim.lap_start_time - 130.0).abs() < 1e-9);
        let out = build_output(&sim);
        assert!(
            out.lap_time.abs() < 1e-9,
            "lap_time should reset, got {}",
            out.lap_time
        );
    }
}
