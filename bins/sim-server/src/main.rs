#![deny(unsafe_code)]
//! Real-time 14-DOF simulation server.
//!
//! Runs the Apex-14 vehicle dynamics model at 1kHz, receiving driver
//! inputs over UDP and broadcasting vehicle state telemetry.

mod ai_driver;

use std::net::UdpSocket;
use std::path::Path;
use std::time::Instant;

use apex_physics::car_params::GRAVITY;
use apex_physics::{
    AeroModel, CarParams, FourteenDofModel, PacejkaTire, Powertrain, SuspensionSystem,
};
use apex_sim::protocol::InputPacket;
use apex_sim::realtime::RealtimeIntegrator;
use apex_sim::server::{
    build_output, map_input_to_control, run_server, step_frame, tire_peak_longitudinal_mu,
    SimServerConfig, SimState,
};
use apex_sim::shared_mem::SimSharedMem;
use apex_track::{
    build_track, circle_track, load_track_json, monza_circuit, oval_track, silverstone_circuit,
    Track,
};
use clap::Parser;

use ai_driver::AiDriver;

/// Forward speed (m/s) the AI-driven car is trimmed for and started at.
///
/// Matches `apex_rl`'s training environment start speed so the policy begins
/// each run in the same conditions it was trained on.
const START_SPEED: f64 = 20.0;

/// Control decision period (s) for the AI driver.
///
/// The policy is queried at this rate (the training environment's control
/// step, 100 Hz) and the resulting control is held constant across the
/// intervening 1 kHz simulation frames.
const AI_CONTROL_DT: f64 = 0.01;

/// Real-time 14-DOF simulation server for hardware-in-the-loop testing.
#[derive(Parser)]
#[command(name = "sim-server")]
struct Args {
    /// Track to simulate on (built-in name or path to JSON file).
    /// Built-in tracks: silverstone, monza, oval, circle
    #[arg(long, default_value = "silverstone")]
    track: String,

    /// UDP port to listen for input packets.
    #[arg(long, default_value_t = 20777)]
    input_port: u16,

    /// UDP address:port to send telemetry output to.
    #[arg(long, default_value = "127.0.0.1:20778")]
    output_addr: String,

    /// Telemetry output rate in Hz.
    #[arg(long, default_value_t = 60)]
    telemetry_hz: u32,

    /// Use calibrated F1 2024 car parameters instead of defaults.
    #[arg(long)]
    calibrated: bool,

    /// Optional path for shared memory file (for zero-latency local I/O).
    #[arg(long)]
    shared_mem: Option<String>,

    /// Path to trained AI driver policy weights (safetensors file).
    /// When provided, the server drives autonomously instead of reading UDP inputs.
    #[arg(long)]
    ai_driver: Option<String>,
}

/// Resolve the `--track` argument to a [`Track`].
///
/// Matches a built-in circuit name (silverstone, monza, oval, circle); anything
/// else is treated as a path to a track JSON file.
fn load_track(name: &str) -> Result<Track, Box<dyn std::error::Error>> {
    let (points, closed, label) = match name.to_lowercase().as_str() {
        "silverstone" => {
            let (p, c) = silverstone_circuit();
            (p, c, "Silverstone")
        }
        "monza" => {
            let (p, c) = monza_circuit();
            (p, c, "Monza")
        }
        "oval" => {
            let (p, c) = oval_track(1000.0, 100.0, 12.0, 500);
            (p, c, "Oval")
        }
        "circle" => {
            let (p, c) = circle_track(100.0, 12.0, 500);
            (p, c, "Circle")
        }
        _ => return load_track_json(Path::new(name)),
    };
    Ok(build_track(label, &points, closed))
}

/// Run the simulation with an AI driver.
///
/// Similar to [`run_server`] but uses a trained policy for inputs instead of
/// UDP. It still outputs telemetry via UDP (and shared memory if configured) so
/// the HUD can visualize the AI driving.
///
/// The 14-DOF model runs at 1kHz (spin-waited for timing, like [`run_server`]),
/// but the policy is queried at the training control rate ([`AI_CONTROL_DT`],
/// 100 Hz): the computed control is held constant across the intervening frames,
/// mirroring how the training environment applies one action per control step
/// over several integration sub-steps. Each frame projects the car onto the
/// track centerline so the telemetry's `track_distance`/`track_offset` stay
/// valid and laps are counted.
#[allow(clippy::too_many_arguments)]
fn run_ai_server(
    config: SimServerConfig,
    car: &CarParams,
    tire: &PacejkaTire,
    suspension: &SuspensionSystem,
    aero: &AeroModel,
    track: &Track,
    ai_driver: &mut AiDriver,
    mut shared_mem: Option<SimSharedMem>,
) -> std::io::Result<()> {
    let model = FourteenDofModel::new(car, tire, suspension, aero, START_SPEED);
    let integrator = RealtimeIntegrator::new_1khz();
    let mut powertrain = Powertrain::f1_2024();

    // Peak tire grip (computed once) used to traction-limit the drive torque.
    let grip_mu = tire_peak_longitudinal_mu(tire, car.mass * GRAVITY / 4.0);

    let output_socket = UdpSocket::bind("0.0.0.0:0")?;

    let mut sim = SimState::new(&model, START_SPEED);
    sim.attach_track(track);

    let telemetry_interval = config.telemetry_interval_frames();
    let mut frame_counter: u32 = 0;
    let frame_budget = integrator.target_dt();

    // Number of 1kHz frames per policy decision (>= 1).
    let control_interval = ((AI_CONTROL_DT / frame_budget).round() as u32).max(1);
    let mut decision_counter: u32 = 0;
    let mut control = [0.0_f64; 3];

    loop {
        let frame_start = Instant::now();

        // Query the policy at the control rate; hold the control otherwise.
        if decision_counter == 0 {
            let (track_distance, lateral_offset) = sim
                .track_pos
                .map(|tp| (tp.distance, tp.lateral_offset))
                .unwrap_or((0.0, 0.0));
            let action = ai_driver
                .compute_action(
                    &sim.state,
                    track,
                    track_distance,
                    lateral_offset,
                    AI_CONTROL_DT,
                )
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            let input = InputPacket {
                steering: action[0] as f32,
                throttle: action[1] as f32,
                brake: action[2] as f32,
                gear: 3, // auto gear selection happens in the powertrain
                sequence: frame_counter,
            };
            sim.last_input = input;
            control = map_input_to_control(
                &input,
                &config,
                &mut powertrain,
                car,
                grip_mu,
                sim.drive_wheel_omega(),
                sim.state[6],
            );
        }
        decision_counter = (decision_counter + 1) % control_interval;

        // Step the simulation; this also projects onto the track and counts laps.
        let send_telemetry = step_frame(
            &mut sim,
            &integrator,
            &model,
            &control,
            Some(track),
            telemetry_interval,
            &mut frame_counter,
        );

        // Emit telemetry at the configured rate.
        if send_telemetry {
            sim.out_sequence = sim.out_sequence.wrapping_add(1);
            let out = build_output(&sim);
            output_socket.send_to(&out.to_bytes(), &config.output_addr)?;
            if let Some(shmem) = &mut shared_mem {
                shmem.write_output(&out);
            }
        }

        // Spin-wait until the frame budget has elapsed.
        while frame_start.elapsed().as_secs_f64() < frame_budget {
            std::hint::spin_loop();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    env_logger::init();

    // Load or construct the track. The server starts the car at the track's
    // start line and projects each frame onto it for lap tracking.
    let track = load_track(&args.track)?;
    println!(
        "Track: {} ({:.0} m, {})",
        track.name,
        track.total_length,
        if track.is_closed { "closed" } else { "open" }
    );

    // Car and component models.
    let car = if args.calibrated {
        CarParams::f1_2024_calibrated()
    } else {
        CarParams::default()
    };
    let tire = PacejkaTire::f1_default();
    let suspension = SuspensionSystem::f1_default();
    let aero = AeroModel::f1_default();

    // Server configuration from the CLI arguments.
    let config = SimServerConfig {
        input_addr: format!("0.0.0.0:{}", args.input_port),
        output_addr: args.output_addr.clone(),
        telemetry_hz: args.telemetry_hz,
        ..SimServerConfig::default()
    };

    println!("Simulation server starting...");
    println!("  Input:      udp://{}", config.input_addr);
    println!("  Output:     udp://{}", config.output_addr);
    println!("  Telemetry:  {} Hz", config.telemetry_hz);
    println!(
        "  Car:        {}",
        if args.calibrated {
            "F1 2024 calibrated"
        } else {
            "default"
        }
    );

    // Optionally expose a shared memory region for zero-latency local I/O.
    let shared_mem = match &args.shared_mem {
        Some(path) => {
            let mem = SimSharedMem::create(Path::new(path))?;
            println!("  Shared memory: {}", path);
            Some(mem)
        }
        None => None,
    };

    println!("Press Ctrl+C to stop.");

    // With an AI driver the server drives autonomously (no UDP input);
    // otherwise it reads driver inputs over UDP / shared memory.
    if let Some(ai_path) = &args.ai_driver {
        println!("  AI driver:  {}", ai_path);
        let mut ai = AiDriver::load(ai_path, START_SPEED)?;
        run_ai_server(
            config,
            &car,
            &tire,
            &suspension,
            &aero,
            &track,
            &mut ai,
            shared_mem,
        )?;
    } else {
        run_server(
            config,
            &car,
            &tire,
            &suspension,
            &aero,
            Some(&track),
            shared_mem,
        )?;
    }
    Ok(())
}
