#![deny(unsafe_code)]
//! Apex-14 unified command-line interface.
//!
//! A single `apex-14` entry point with subcommands for lap simulation,
//! trajectory optimization, track import, and car-parameter inspection. The
//! standalone binaries (`simulate`, `optimize`, `compare`, `viewer`, `validate`)
//! remain available; this is an additional front end.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "apex-14")]
#[command(about = "Minimum-time lap simulation and racing line optimization")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run quasi-steady-state lap simulation
    Qss {
        /// Track file (JSON or TUMFTM CSV)
        #[arg(short, long)]
        track: Option<PathBuf>,

        /// Car configuration file (TOML)
        #[arg(short, long)]
        car: Option<PathBuf>,

        /// Use calibrated F1 2024 parameters (default if no --car given)
        #[arg(long, default_value_t = false)]
        calibrated: bool,

        /// Export CSV telemetry to this path
        #[arg(long)]
        csv: Option<PathBuf>,

        /// Export SVG track visualization to this path
        #[arg(long)]
        svg: Option<PathBuf>,
    },

    /// Run trajectory optimization
    Optimize {
        /// Track file (JSON or TUMFTM CSV)
        #[arg(short, long)]
        track: Option<PathBuf>,

        /// Car configuration file (TOML)
        #[arg(short, long)]
        car: Option<PathBuf>,

        /// Number of collocation nodes
        #[arg(short, long, default_value_t = 50)]
        nodes: usize,

        /// Use Hermite-Simpson collocation (default: trapezoidal)
        #[arg(long, default_value_t = false)]
        hermite_simpson: bool,

        /// Use calibrated parameters
        #[arg(long, default_value_t = false)]
        calibrated: bool,
    },

    /// Import a track from TUMFTM CSV format to Apex-14 JSON
    ImportTrack {
        /// Input TUMFTM CSV file
        #[arg(short, long)]
        input: PathBuf,

        /// Output JSON file path
        #[arg(short, long)]
        output: PathBuf,

        /// Track name
        #[arg(short, long)]
        name: String,
    },

    /// List available built-in tracks
    Tracks,

    /// Show car parameters (default, calibrated, or from file)
    CarInfo {
        /// Car configuration file (TOML)
        #[arg(short, long)]
        car: Option<PathBuf>,

        /// Use calibrated parameters as base
        #[arg(long, default_value_t = false)]
        calibrated: bool,

        /// Export parameters to TOML file
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Compute optimal pit stop strategy for a race
    Strategy {
        /// Number of race laps
        #[arg(short, long, default_value_t = 52)]
        laps: usize,

        /// Base lap time in seconds (from QSS or manual input)
        #[arg(short, long)]
        base_time: Option<f64>,

        /// Track file for computing base lap time via QSS
        #[arg(short, long)]
        track: Option<PathBuf>,

        /// Car configuration file
        #[arg(short, long)]
        car: Option<PathBuf>,

        /// Use calibrated parameters
        #[arg(long, default_value_t = false)]
        calibrated: bool,

        /// Maximum number of pit stops to consider
        #[arg(long, default_value_t = 2)]
        max_stops: usize,

        /// Pit stop time loss in seconds
        #[arg(long, default_value_t = 22.0)]
        pit_loss: f64,
    },
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run(cli) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Qss {
            track,
            car,
            calibrated,
            csv,
            svg,
        } => cmd_qss(track, car, calibrated, csv, svg),
        Commands::Optimize {
            track,
            car,
            nodes,
            hermite_simpson,
            calibrated,
        } => cmd_optimize(track, car, nodes, hermite_simpson, calibrated),
        Commands::ImportTrack {
            input,
            output,
            name,
        } => cmd_import_track(input, output, name),
        Commands::Tracks => cmd_tracks(),
        Commands::CarInfo {
            car,
            calibrated,
            export,
        } => cmd_car_info(car, calibrated, export),
        Commands::Strategy {
            laps,
            base_time,
            track,
            car,
            calibrated,
            max_stops,
            pit_loss,
        } => cmd_strategy(laps, base_time, track, car, calibrated, max_stops, pit_loss),
    }
}

// --- shared helpers ---

fn load_track_from_path(
    path: &std::path::Path,
) -> Result<apex_track::Track, Box<dyn std::error::Error>> {
    let path_str = path.to_string_lossy();

    // Detect format from extension
    if path_str.ends_with(".json") {
        apex_track::load_track_json(path)
    } else if path_str.ends_with(".csv") {
        // Assume TUMFTM format
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown");
        apex_track::load_tumftm_csv(path, name)
    } else {
        Err(format!("Unknown track file format: {}. Use .json or .csv", path_str).into())
    }
}

fn load_car_params(
    car_path: Option<PathBuf>,
    calibrated: bool,
) -> Result<apex_physics::CarParams, Box<dyn std::error::Error>> {
    let base = if calibrated {
        apex_physics::CarParams::f1_2024_calibrated()
    } else {
        apex_physics::CarParams::default()
    };

    match car_path {
        Some(path) => apex_physics::load_car_toml(&path, &base),
        None => Ok(base),
    }
}

fn default_track() -> apex_track::Track {
    let (pts, closed) = apex_track::oval_track(500.0, 80.0, 12.0, 300);
    apex_track::build_track("Oval", &pts, closed)
}

// --- subcommands ---

fn cmd_qss(
    track: Option<PathBuf>,
    car: Option<PathBuf>,
    calibrated: bool,
    csv: Option<PathBuf>,
    svg: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let track = match track {
        Some(path) => load_track_from_path(&path)?,
        None => {
            println!("No track specified, using default oval.");
            default_track()
        }
    };
    let params = load_car_params(car, calibrated)?;

    println!("Track: {} ({:.0} m)", track.name, track.total_length);
    println!(
        "Car: mass {:.0} kg, Cl {:.2}, mu {:.2}",
        params.mass, params.lift_coeff, params.tire_mu
    );
    println!();

    let result = apex_physics::qss_lap_sim(&track, &params);

    println!("Lap time:     {:.3} s", result.lap_time);
    println!(
        "Top speed:    {:.1} km/h",
        result.speeds.iter().cloned().fold(f64::MIN, f64::max) * 3.6
    );
    println!(
        "Min speed:    {:.1} km/h",
        result.speeds.iter().cloned().fold(f64::MAX, f64::min) * 3.6
    );
    println!(
        "Max lateral g: {:.2}",
        result.lateral_gs.iter().cloned().fold(f64::MIN, f64::max)
    );

    if let Some(csv_path) = csv {
        apex_telemetry::export_qss_csv(&csv_path, &track, &result)?;
        println!("\nCSV exported to {}", csv_path.display());
    }

    if let Some(svg_path) = svg {
        apex_telemetry::render_track_svg(&svg_path, &track, &result.speeds, &track.name)?;
        println!("SVG exported to {}", svg_path.display());
    }

    Ok(())
}

fn cmd_optimize(
    track: Option<PathBuf>,
    car: Option<PathBuf>,
    nodes: usize,
    hermite_simpson: bool,
    calibrated: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let track = match track {
        Some(path) => load_track_from_path(&path)?,
        None => default_track(),
    };
    let params = load_car_params(car, calibrated)?;

    let method = if hermite_simpson {
        apex_optimizer::collocation::CollocationMethod::HermiteSimpson
    } else {
        apex_optimizer::collocation::CollocationMethod::Trapezoidal
    };

    let config = apex_optimizer::CollocationConfig {
        n_nodes: nodes,
        method,
        ..Default::default()
    };

    println!("Track: {} ({:.0} m)", track.name, track.total_length);
    println!("Nodes: {}, Method: {:?}", nodes, method);
    println!("Optimizing...");

    let optimizer = apex_optimizer::CollocationOptimizer::new(config, &track, &params);
    let solver_config = apex_optimizer::GaussNewtonConfig::default();
    let result = optimizer.optimize_gn(&solver_config);

    println!("\nLap time:  {:.3} s", result.lap_time);
    println!("Converged: {}", result.converged);
    println!("Eq violation: {:.2e}", result.eq_violation);

    Ok(())
}

fn cmd_import_track(
    input: PathBuf,
    output: PathBuf,
    name: String,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Importing track '{}' from {}", name, input.display());
    let track = apex_track::load_tumftm_csv(&input, &name)?;
    println!(
        "Track length: {:.3} km, {} segments",
        track.total_length / 1000.0,
        track.segments.len()
    );

    let json = apex_track::export_track_json(&track)?;
    std::fs::write(&output, json)?;
    println!("Exported to {}", output.display());

    Ok(())
}

fn cmd_tracks() -> Result<(), Box<dyn std::error::Error>> {
    println!("Built-in tracks:");
    println!("  oval        Oval (500m straights, R=80m corners)");
    println!("  circle      Circle (R=100m)");
    println!("  silverstone Silverstone (approximation)");
    println!("  monza       Monza (approximation)");
    println!();
    println!("Load custom tracks with --track <file.json> or --track <file.csv>");
    println!(
        "Import TUMFTM CSV: apex-14 import-track -i Silverstone.csv -o tracks/silverstone.json -n Silverstone"
    );
    Ok(())
}

fn cmd_car_info(
    car: Option<PathBuf>,
    calibrated: bool,
    export: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let params = load_car_params(car, calibrated)?;

    println!("Car Parameters:");
    println!("  Mass:            {:.0} kg", params.mass);
    println!("  Drag coeff:      {:.3}", params.drag_coeff);
    println!("  Lift coeff:      {:.3}", params.lift_coeff);
    println!("  Tire mu:         {:.3}", params.tire_mu);
    println!("  Max drive force: {:.0} N", params.max_drive_force);
    println!("  Max brake force: {:.0} N", params.max_brake_force);
    println!("  Wheelbase:       {:.3} m", params.wheelbase);
    println!("  CoG height:      {:.3} m", params.cog_height);
    println!(
        "  Aero balance:    {:.0}% front",
        params.aero_balance_front * 100.0
    );

    if let Some(export_path) = export {
        let name = if calibrated { "Calibrated" } else { "Default" };
        let toml = apex_physics::export_car_toml(&params, name);
        std::fs::write(&export_path, toml)?;
        println!("\nExported to {}", export_path.display());
    }

    Ok(())
}

fn cmd_strategy(
    laps: usize,
    base_time: Option<f64>,
    track: Option<PathBuf>,
    car: Option<PathBuf>,
    calibrated: bool,
    max_stops: usize,
    pit_loss: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    // Determine base lap time
    let base = match base_time {
        Some(t) => {
            println!("Base lap time: {:.3} s (manual)", t);
            t
        }
        None => {
            let track_data = match track {
                Some(ref path) => load_track_from_path(path)?,
                None => default_track(),
            };
            let params = load_car_params(car, calibrated)?;
            let qss = apex_physics::qss_lap_sim(&track_data, &params);
            println!(
                "Base lap time: {:.3} s (QSS on {})",
                qss.lap_time, track_data.name
            );
            // QSS gives the actual lap time for the track; use it directly as the
            // single-lap baseline for the strategy model.
            qss.lap_time
        }
    };

    println!(
        "Race: {} laps, max {} stops, {:.0}s pit loss",
        laps, max_stops, pit_loss
    );
    println!();

    let mut evaluator = apex_physics::StrategyEvaluator::new(base, laps);
    evaluator.pit_time_loss = pit_loss;

    let results = evaluator.find_optimal(max_stops, 8, true);

    if results.is_empty() {
        println!("No feasible strategies found.");
        return Ok(());
    }

    // Print top 5 strategies
    println!("Top 5 strategies:");
    println!(
        "{:<6} {:<20} {:>12} {:>8}",
        "Rank", "Strategy", "Race Time", "Gap"
    );
    println!("{}", "-".repeat(50));

    let best_time = results[0].total_time;
    for (i, result) in results.iter().take(5).enumerate() {
        let gap = result.total_time - best_time;
        println!(
            "{:<6} {:<20} {:>10.1}s {:>+7.1}s",
            i + 1,
            result.strategy.display(),
            result.total_time,
            gap,
        );
    }

    println!();

    // Print details of the best strategy
    let best = &results[0];
    println!(
        "Optimal: {} ({} stop{})",
        best.strategy.display(),
        best.strategy.num_stops(),
        if best.strategy.num_stops() == 1 {
            ""
        } else {
            "s"
        }
    );
    println!(
        "Total race time: {:.1} s ({:.1} min)",
        best.total_time,
        best.total_time / 60.0
    );
    println!();

    // Stint summary
    println!("Stint breakdown:");
    let mut lap_offset = 0;
    for (i, stint) in best.strategy.stints.iter().enumerate() {
        let first_lap_time = best.lap_times[lap_offset];
        let last_lap_time = best.lap_times[lap_offset + stint.laps - 1];
        println!(
            "  Stint {}: {} {} laps (lap {}-{}) | {:.2}s -> {:.2}s",
            i + 1,
            stint.compound,
            stint.laps,
            lap_offset + 1,
            lap_offset + stint.laps,
            first_lap_time,
            last_lap_time,
        );
        lap_offset += stint.laps;
    }

    // Undercut analysis on the first pit stop
    if best.strategy.num_stops() >= 1 {
        let pit_lap = best.strategy.stints[0].laps;
        let analysis = evaluator.undercut_overcut(
            pit_lap,
            best.strategy.stints[0].compound,
            best.strategy.stints[1].compound,
            0,
        );
        println!();
        println!("Undercut/overcut at lap {}:", pit_lap);
        println!("  Pit 1 lap early: {:+.3}s", analysis.undercut_delta);
        println!("  Pit 1 lap late:  {:+.3}s", analysis.overcut_delta);
        println!("  -> {}", analysis.recommendation);
    }

    Ok(())
}
