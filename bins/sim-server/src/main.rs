#![deny(unsafe_code)]
//! Real-time 14-DOF simulation server.
//!
//! Runs the Apex-14 vehicle dynamics model at 1kHz, receiving driver
//! inputs over UDP and broadcasting vehicle state telemetry.

use std::path::Path;

use apex_physics::{AeroModel, CarParams, PacejkaTire, SuspensionSystem};
use apex_sim::server::{run_server, SimServerConfig};
use apex_sim::shared_mem::SimSharedMem;
use apex_track::{
    build_track, circle_track, load_track_json, monza_circuit, oval_track, silverstone_circuit,
    Track,
};
use clap::Parser;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    env_logger::init();

    // Load or construct the track (used here for reporting; the server trims and
    // runs the model independently of the track geometry).
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

    run_server(config, &car, &tire, &suspension, &aero, shared_mem)?;
    Ok(())
}
