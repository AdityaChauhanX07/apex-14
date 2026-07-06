#![deny(unsafe_code)]
//! Apex-14 unified command-line interface.
//!
//! A single `apex-14` entry point with subcommands for lap simulation,
//! trajectory optimization, track import, and car-parameter inspection. The
//! standalone binaries (`simulate`, `optimize`, `compare`, `viewer`, `validate`)
//! remain available; this is an additional front end.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use apex_math::resolve_seed;

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

    /// Align measured telemetry to a track frame and project GPS to (s, n)
    TelemetryAlign {
        /// Input telemetry CSV (standard Apex telemetry format, with x/y)
        #[arg(long)]
        telemetry: PathBuf,

        /// Track file (Apex-14 JSON or TUMFTM CSV) providing the centerline
        #[arg(long)]
        track: PathBuf,

        /// Output aligned CSV path (projected s/s_raw/x/y/lateral_offset)
        #[arg(long)]
        out: PathBuf,
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

    /// Optimize car setup parameters to minimize lap time (CMA-ES + QSS)
    SetupOptimize {
        /// Track to optimize for. Built-in: silverstone, monza, oval, circle;
        /// anything else is treated as a track file path (JSON or CSV).
        #[arg(long, default_value = "silverstone")]
        track: String,

        /// Number of CMA-ES generations.
        #[arg(long, default_value_t = 50)]
        generations: usize,

        /// Initial step size (fraction of parameter range).
        #[arg(long, default_value_t = 0.3)]
        sigma: f64,

        /// Use calibrated F1 parameters as baseline.
        #[arg(long)]
        calibrated: bool,

        /// Random seed for CMA-ES reproducibility. Defaults to 42 when omitted.
        #[arg(long)]
        seed: Option<u64>,

        /// Output TOML file for the optimized setup.
        #[arg(long)]
        output: Option<String>,
    },

    /// Simulate a full Grand Prix with Monte Carlo analysis.
    RaceSim {
        /// Track for the race.
        #[arg(long, default_value = "silverstone")]
        track: String,

        /// Number of race laps.
        #[arg(long, default_value_t = 52)]
        laps: usize,

        /// Number of Monte Carlo simulations.
        #[arg(long, default_value_t = 1000)]
        sims: usize,

        /// Random seed for reproducibility. Defaults to 42 when omitted.
        #[arg(long)]
        seed: Option<u64>,

        /// Optimize strategy for car at this grid position (1-indexed).
        /// If provided, runs strategy optimization for this car.
        #[arg(long)]
        optimize_car: Option<usize>,

        /// Use calibrated car as baseline for lap time computation.
        #[arg(long)]
        calibrated: bool,
    },

    /// Run parameter sensitivity analysis
    Sensitivity {
        /// Track file (JSON or TUMFTM CSV)
        #[arg(short, long)]
        track: Option<PathBuf>,

        /// Car configuration file (TOML)
        #[arg(short, long)]
        car: Option<PathBuf>,

        /// Use calibrated parameters
        #[arg(long, default_value_t = false)]
        calibrated: bool,

        /// Number of samples per parameter for OAT analysis
        #[arg(long, default_value_t = 11)]
        oat_samples: usize,

        /// Number of Monte Carlo samples
        #[arg(long, default_value_t = 1000)]
        mc_samples: usize,

        /// Random seed for reproducibility. Defaults to 42 when omitted.
        #[arg(long)]
        seed: Option<u64>,

        /// Export tornado chart SVG to this path
        #[arg(long)]
        svg: Option<PathBuf>,
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
        Commands::TelemetryAlign {
            telemetry,
            track,
            out,
        } => cmd_telemetry_align(telemetry, track, out),
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
        Commands::SetupOptimize {
            track,
            generations,
            sigma,
            calibrated,
            seed,
            output,
        } => cmd_setup_optimize(track, generations, sigma, calibrated, seed, output),
        Commands::RaceSim {
            track,
            laps,
            sims,
            seed,
            optimize_car,
            calibrated,
        } => cmd_race_sim(track, laps, sims, seed, optimize_car, calibrated),
        Commands::Sensitivity {
            track,
            car,
            calibrated,
            oat_samples,
            mc_samples,
            seed,
            svg,
        } => cmd_sensitivity(track, car, calibrated, oat_samples, mc_samples, seed, svg),
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

/// Resolve a track argument that may be a built-in circuit name or a file path.
///
/// Recognized built-ins: `silverstone`, `monza`, `oval`, `circle`. Anything else
/// is treated as a path to a track file (JSON or TUMFTM CSV).
fn resolve_track(name: &str) -> Result<apex_track::Track, Box<dyn std::error::Error>> {
    let (points, closed, label) = match name.to_lowercase().as_str() {
        "silverstone" => {
            let (p, c) = apex_track::silverstone_circuit();
            (p, c, "Silverstone")
        }
        "monza" => {
            let (p, c) = apex_track::monza_circuit();
            (p, c, "Monza")
        }
        "oval" => {
            let (p, c) = apex_track::oval_track(1000.0, 100.0, 12.0, 500);
            (p, c, "Oval")
        }
        "circle" => {
            let (p, c) = apex_track::circle_track(100.0, 12.0, 500);
            (p, c, "Circle")
        }
        _ => return load_track_from_path(std::path::Path::new(name)),
    };
    Ok(apex_track::build_track(label, &points, closed))
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

    if csv.is_some() || svg.is_some() {
        // QSS has no tunable solver settings and no RNG: settings_hash is the
        // mode label, seed is None.
        let meta = apex_telemetry::RunMetadata::new(
            apex_physics::car_params_hash(&params),
            apex_track::processed_track_hash(&track),
            apex_telemetry::settings_hash_for_mode("qss.grip-circle"),
            None,
        );

        if let Some(csv_path) = csv {
            apex_telemetry::export_qss_csv(&csv_path, &meta, &track, &result)?;
            println!("\nCSV exported to {}", csv_path.display());
        }

        if let Some(svg_path) = svg {
            apex_telemetry::render_track_svg(
                &svg_path,
                &meta,
                &track,
                &result.speeds,
                &track.name,
            )?;
            println!("SVG exported to {}", svg_path.display());
        }
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

fn cmd_telemetry_align(
    telemetry: PathBuf,
    track: PathBuf,
    out: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    use apex_correlate::{
        fit_alignment, import_telemetry, project_to_track, write_telemetry_csv, AlignConfig,
        Mapping,
    };

    println!(
        "Aligning telemetry {} to track {}",
        telemetry.display(),
        track.display()
    );
    // 1. Import measured telemetry (already registry names/units → identity map).
    let tel = import_telemetry(&telemetry, &Mapping::identity())?;
    println!("  telemetry: {} samples", tel.len());

    // 2. Load the centerline.
    let trk = load_track_from_path(&track)?;
    println!(
        "  track: {} ({:.1} m, {} segments)",
        trk.name,
        trk.total_length,
        trk.segments.len()
    );

    // 3. Fit the similarity transform.
    let align = fit_alignment(&tel, &trk, AlignConfig::default())?;
    let theta_deg = align.transform.theta.to_degrees();
    println!("\n=== Alignment fit ===");
    println!("  rotation:   {:.3} deg", theta_deg);
    println!("  scale:      {:.5}", align.transform.scale);
    println!(
        "  translation: ({:.2}, {:.2}) m",
        align.transform.tx, align.transform.ty
    );
    println!("  reflection: {}", align.transform.reflect);
    println!("  direction reversed: {}", align.direction_reversed);
    println!("  s_offset:   {:.2} m", align.s_offset);
    println!("  post-fit RMS: {:.3} m", align.rms);
    println!("  max closest-point distance: {:.3} m", align.max_dist);
    if (align.transform.scale - 1.0).abs() > 0.05 {
        println!(
            "  !! WARNING: scale {:.4} deviates >5% from 1.0 — frames may not both be metres",
            align.transform.scale
        );
    }
    let half_width = 0.5
        * trk
            .segments
            .iter()
            .map(|s| s.width_left + s.width_right)
            .sum::<f64>()
        / trk.segments.len() as f64;
    if align.rms > half_width {
        println!(
            "  !! WARNING: RMS {:.2} m exceeds mean half-width {:.2} m",
            align.rms, half_width
        );
    }

    // 4. Persist the sidecar (flat scalar TOML) next to the telemetry file.
    let sidecar_path = sidecar_for(&telemetry);
    let sidecar = format!(
        "# Fitted FastF1-local → track-frame similarity transform.\n\
         # Derived from measured telemetry; keep local (gitignored).\n\
         scale = {:.8}\n\
         rotation_rad = {:.8}\n\
         rotation_deg = {:.6}\n\
         tx = {:.6}\n\
         ty = {:.6}\n\
         reflection = {}\n\
         direction_reversed = {}\n\
         s_offset = {:.6}\n\
         rms = {:.6}\n\
         max_dist = {:.6}\n\
         telemetry = {:?}\n\
         track = {:?}\n",
        align.transform.scale,
        align.transform.theta,
        theta_deg,
        align.transform.tx,
        align.transform.ty,
        align.transform.reflect,
        align.direction_reversed,
        align.s_offset,
        align.rms,
        align.max_dist,
        telemetry.display().to_string(),
        track.display().to_string(),
    );
    std::fs::write(&sidecar_path, sidecar)?;
    println!("\n  wrote alignment sidecar: {}", sidecar_path.display());

    // 5. Project GPS → (s, n).
    let (mut aligned, stats) = project_to_track(&tel, &trk, &align.transform)?;
    println!("\n=== Projection ===");
    println!(
        "  s_proj span: {:.1} m   (FastF1 raw s span: {:.1} m, diff {:.1} m)",
        stats.s_proj_span,
        stats.s_raw_span,
        stats.s_proj_span - stats.s_raw_span
    );
    println!(
        "  lateral_offset n: min {:.2}  max {:.2}  RMS {:.2} m",
        stats.n_min, stats.n_max, stats.n_rms
    );
    println!(
        "  within track bounds: {:.1}%   max |dist|: {:.2} m   non-monotone samples: {}",
        stats.frac_within_bounds * 100.0,
        stats.max_dist,
        stats.non_monotone
    );

    // 6. Aligned CSV carries descriptive provenance + transform params in the
    //    header (no RunMetadata sim hashes — this is derived-from-measured data).
    aligned
        .metadata
        .push(("aligned_to_track".into(), trk.name.clone()));
    aligned
        .metadata
        .push(("align_rotation_deg".into(), format!("{theta_deg:.4}")));
    aligned.metadata.push((
        "align_scale".into(),
        format!("{:.6}", align.transform.scale),
    ));
    aligned.metadata.push((
        "align_translation_m".into(),
        format!("{:.3},{:.3}", align.transform.tx, align.transform.ty),
    ));
    aligned.metadata.push((
        "align_reflection".into(),
        align.transform.reflect.to_string(),
    ));
    aligned.metadata.push((
        "align_direction_reversed".into(),
        align.direction_reversed.to_string(),
    ));
    aligned
        .metadata
        .push(("align_s_offset_m".into(), format!("{:.3}", align.s_offset)));
    aligned
        .metadata
        .push(("align_rms_m".into(), format!("{:.4}", align.rms)));
    aligned.metadata.push((
        "lateral_offset_sign".into(),
        "positive = left of centerline (direction of travel)".into(),
    ));

    write_telemetry_csv(&out, &aligned)?;
    println!("\n  wrote aligned telemetry: {}", out.display());
    Ok(())
}

/// Derive the alignment sidecar path from a telemetry path:
/// `foo/bar.csv` → `foo/bar.align.toml`.
fn sidecar_for(telemetry: &std::path::Path) -> PathBuf {
    let stem = telemetry
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("telemetry");
    let mut p = telemetry.to_path_buf();
    p.set_file_name(format!("{stem}.align.toml"));
    p
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

fn cmd_sensitivity(
    track: Option<PathBuf>,
    car: Option<PathBuf>,
    calibrated: bool,
    oat_samples: usize,
    mc_samples: usize,
    seed: Option<u64>,
    svg: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let seed = resolve_seed(seed, 42, "sensitivity");
    let track_data = match track {
        Some(ref path) => load_track_from_path(path)?,
        None => default_track(),
    };
    let params = load_car_params(car, calibrated)?;
    let param_set = apex_physics::f1_parameter_set(&params);

    println!(
        "Track: {} ({:.0} m)",
        track_data.name, track_data.total_length
    );
    println!("Parameters: {} variables", param_set.len());
    println!();

    // OAT analysis
    println!("--- One-at-a-Time Sensitivity ---");
    let oat_results = apex_physics::oat_sensitivity(&track_data, &params, &param_set, oat_samples);

    // Sort by absolute sensitivity
    let mut sorted: Vec<&apex_physics::OatResult> = oat_results.iter().collect();
    sorted.sort_by(|a, b| {
        b.sensitivity_pct
            .abs()
            .partial_cmp(&a.sensitivity_pct.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("{:<20} {:>12} {:>10}", "Parameter", "Sensitivity", "% / %");
    println!("{}", "-".repeat(44));
    for result in &sorted {
        println!(
            "{:<20} {:>+10.4} s {:>+9.2}%",
            result.name, result.sensitivity, result.sensitivity_pct
        );
    }
    println!();
    println!("Nominal lap time: {:.3} s", oat_results[0].nominal_lap_time);

    // Monte Carlo
    println!();
    println!(
        "--- Monte Carlo Analysis ({} samples, seed {}) ---",
        mc_samples, seed
    );
    let mc =
        apex_physics::monte_carlo_sensitivity(&track_data, &params, &param_set, mc_samples, seed);
    println!("Mean:    {:.3} s", mc.mean);
    println!(
        "Std dev: {:.3} s ({:.2}%)",
        mc.std_dev,
        mc.std_dev / mc.mean * 100.0
    );
    println!("5th pct: {:.3} s", mc.percentile_5);
    println!("95th pct:{:.3} s", mc.percentile_95);
    println!(
        "Range:   {:.3} s ({:.2}%)",
        mc.percentile_95 - mc.percentile_5,
        (mc.percentile_95 - mc.percentile_5) / mc.mean * 100.0
    );
    println!();
    println!("Top correlations:");
    for (name, corr) in mc.correlations.iter().take(5) {
        let direction = if *corr > 0.0 {
            "more = slower"
        } else {
            "more = faster"
        };
        println!("  {:<20} r = {:+.3}  ({})", name, corr, direction);
    }

    // Export tornado chart
    if let Some(svg_path) = svg {
        // Sensitivity is seeded (Monte Carlo); settings_hash is the mode label.
        let meta = apex_telemetry::RunMetadata::new(
            apex_physics::car_params_hash(&params),
            apex_track::processed_track_hash(&track_data),
            apex_telemetry::settings_hash_for_mode("sensitivity.oat+mc"),
            Some(seed),
        );
        apex_physics::tornado_chart_svg(&oat_results, &svg_path, &meta.svg_metadata_element())?;
        println!();
        println!("Tornado chart exported to {}", svg_path.display());
    }

    Ok(())
}

fn cmd_race_sim(
    track: String,
    laps: usize,
    sims: usize,
    seed: Option<u64>,
    optimize_car: Option<usize>,
    calibrated: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let seed = resolve_seed(seed, 42, "race-sim");
    let track_data = resolve_track(&track)?;
    let params = if calibrated {
        apex_physics::CarParams::f1_2024_calibrated()
    } else {
        apex_physics::CarParams::default()
    };

    // Base lap time from a QSS lap on this track/car drives the grid pace.
    let qss = apex_physics::qss_lap_sim(&track_data, &params);
    let base_lap_time = qss.lap_time;

    println!(
        "Track: {} ({:.0} m)",
        track_data.name, track_data.total_length
    );
    println!(
        "Base lap time: {:.3} s (QSS, {} car)",
        base_lap_time,
        if calibrated { "calibrated" } else { "default" }
    );
    println!(
        "Race: {} laps | {} Monte Carlo sims | seed {}",
        laps, sims, seed
    );
    println!();

    let race_config = apex_race::config::RaceConfig::for_track(track_data.total_length, laps);
    let mut entries = apex_race::config::default_f1_grid(base_lap_time);

    // Optionally optimize the strategy for one car before the headline run.
    let mut opt_result = None;
    if let Some(grid_pos) = optimize_car {
        if grid_pos == 0 || grid_pos > entries.len() {
            return Err(format!(
                "--optimize-car must be between 1 and {} (1-indexed grid position)",
                entries.len()
            )
            .into());
        }
        let car_idx = grid_pos - 1;
        println!(
            "Optimizing strategy for car {} ({})...",
            grid_pos, entries[car_idx].name
        );

        let opt_config = apex_race::strategy_opt::StrategyOptConfig {
            // Keep per-evaluation sims modest so the CMA-ES loop stays tractable.
            n_sims_per_eval: (sims / 10).max(20),
            seed,
            ..Default::default()
        };
        let result = apex_race::strategy_opt::optimize_strategy(
            &race_config,
            &entries,
            car_idx,
            &opt_config,
        );

        // Apply the optimized strategy to the grid for the headline simulation.
        entries[car_idx].strategy = apex_race::config::RaceStrategy {
            start_compound: result
                .compounds
                .first()
                .copied()
                .unwrap_or(apex_race::config::TireCompound::Medium),
            stops: result
                .stop_laps
                .iter()
                .enumerate()
                .map(|(i, &lap)| apex_race::config::PlannedStop {
                    lap,
                    compound: result
                        .compounds
                        .get(i + 1)
                        .copied()
                        .unwrap_or(apex_race::config::TireCompound::Hard),
                })
                .collect(),
        };
        opt_result = Some((grid_pos, result));
        println!();
    }

    let mc = apex_race::monte_carlo::monte_carlo_race(&race_config, &entries, sims, seed);
    print!("{}", apex_race::monte_carlo::format_report(&mc, &entries));

    if let Some((grid_pos, result)) = opt_result {
        let name = &entries[grid_pos - 1].name;
        println!();
        println!(
            "--- Strategy optimization for car {} ({}) ---",
            grid_pos, name
        );
        let stops: Vec<String> = result.stop_laps.iter().map(|l| format!("L{l}")).collect();
        println!("Optimized pit stops: {}", stops.join(", "));
        let compounds: Vec<String> = result.compounds.iter().map(|c| c.to_string()).collect();
        println!("Compound sequence:   {}", compounds.join(" -> "));
        println!("CMA-ES generations:  {}", result.generations);
        println!("Baseline E[pts]:     {:.2}", result.baseline_points);
        println!("Optimized E[pts]:    {:.2}", result.expected_points);
        println!(
            "Improvement:         {:+.2} pts",
            result.expected_points - result.baseline_points
        );
        println!("Optimized E[pos]:    {:.2}", result.expected_position);
        println!("Optimized win prob:  {:.1}%", result.win_prob * 100.0);
    }

    Ok(())
}

fn cmd_setup_optimize(
    track: String,
    generations: usize,
    sigma: f64,
    calibrated: bool,
    seed: Option<u64>,
    output: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let seed = resolve_seed(seed, 42, "setup-optimize");
    let track_data = resolve_track(&track)?;
    let base_car = if calibrated {
        apex_physics::CarParams::f1_2024_calibrated()
    } else {
        apex_physics::CarParams::default()
    };

    println!(
        "Track: {} ({:.0} m)",
        track_data.name, track_data.total_length
    );
    println!(
        "Baseline car: {}",
        if calibrated {
            "F1 2024 calibrated"
        } else {
            "default"
        }
    );
    println!("CMA-ES: {} generations, sigma {:.2}", generations, sigma);

    let track_name = track_data.name.clone();
    let config = apex_optimizer::SetupEvalConfig::new(track_data, base_car);
    let cmaes_config = apex_optimizer::CmaEsConfig {
        max_generations: generations,
        initial_sigma: sigma,
        seed,
        ..Default::default()
    };

    println!("Optimizing...");
    let result = apex_optimizer::optimize_setup(&config, cmaes_config);

    println!();
    println!("Baseline lap time:  {:.3} s", result.baseline_time);
    println!("Optimized lap time: {:.3} s", result.best_time);
    println!(
        "Improvement:        {:.3} s ({:.2}%)",
        result.improvement,
        if result.baseline_time > 0.0 {
            result.improvement / result.baseline_time * 100.0
        } else {
            0.0
        }
    );
    println!("Generations run:    {}", result.generations);
    println!();
    println!("Optimized setup parameters:");
    print!("{}", config.space.format_report(&result.best_params));

    if let Some(out_path) = output {
        let toml = apex_optimizer::export_setup_toml(
            &config.space,
            &config.base_car,
            &result.best_params,
            &track_name,
            result.best_time,
        );
        std::fs::write(&out_path, toml)?;
        println!();
        println!("Optimized setup written to {}", out_path);
    }

    Ok(())
}
