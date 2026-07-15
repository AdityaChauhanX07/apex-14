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

        /// Disable curvature-aware centerline smoothing (on by default)
        #[arg(long, default_value_t = false)]
        no_smooth: bool,

        /// Max deviation (m) a smoothed point may move from its survey point
        #[arg(long, default_value_t = apex_track::DEFAULT_SMOOTH_TOLERANCE_M)]
        smooth_tolerance: f64,

        /// Merge a pre-fetched elevation sidecar (from tools/fetch_elevation.py)
        /// and write a **v2 3D** track JSON. The input must then be the 2D track
        /// JSON the sidecar was computed against; no network calls are made here.
        #[arg(long)]
        elevation: Option<PathBuf>,
    },

    /// Correlate measured telemetry against a QSS sim (deltas, RMSE, corners)
    Correlate {
        /// Aligned telemetry CSV (standard format, with projected `s` + `speed`)
        #[arg(long)]
        telemetry: PathBuf,

        /// Track file (Apex-14 JSON or TUMFTM CSV)
        #[arg(long)]
        track: PathBuf,

        /// Optional v2 **3D** track JSON (elevation). When given, the QSS runs in
        /// 3D (grade / vertical-curvature load / banking) and `--track` is
        /// ignored (the 2D centerline is taken from the 3D file).
        #[arg(long)]
        track_3d: Option<PathBuf>,

        /// Use calibrated F1 2024 parameters
        #[arg(long, default_value_t = false)]
        calibrated: bool,

        /// Car configuration file (TOML), overrides the base preset
        #[arg(long)]
        car: Option<PathBuf>,

        /// Common-grid resample step (m)
        #[arg(long, default_value_t = 10.0)]
        grid_step: f64,

        /// Which line the QSS runs on: `centerline` (default) or `measured`
        /// (the reconstructed driven line).
        #[arg(long, default_value = "centerline")]
        line: String,

        /// Driven-line reconstruction: `direct` (default, from smoothed measured
        /// x/y) or `offset` (centerline + filtered n(s)).
        #[arg(long, default_value = "direct")]
        driven_line: String,

        /// Offset mode: moving-average half-width (m) applied to n(s)
        #[arg(long, default_value_t = apex_correlate::DEFAULT_N_FILTER_WINDOW_M)]
        n_filter: f64,

        /// Direct mode: smoothing deviation budget (m) for measured x/y
        #[arg(long, default_value_t = apex_correlate::DEFAULT_DRIVEN_SMOOTH_TOLERANCE_M)]
        driven_smooth_tolerance: f64,

        /// Output directory for report.md + SVGs
        #[arg(long)]
        out_dir: PathBuf,

        /// Also write the aligned report grid (s, meas_speed, sim_speed) to
        /// this Parquet file.
        #[cfg(feature = "parquet")]
        #[arg(long)]
        parquet: Option<PathBuf>,
    },

    /// Export a standard telemetry CSV to a MoTeC i2 `.ld` log file
    ExportLd {
        /// Input telemetry CSV (standard Apex telemetry format)
        #[arg(long)]
        telemetry: PathBuf,

        /// Output `.ld` file path
        #[arg(long)]
        out: PathBuf,

        /// Output sample rate (Hz). Default: nominal rate inferred from the data.
        #[arg(long)]
        rate: Option<u16>,
    },

    /// Export a standard telemetry CSV to an Apache Parquet file
    #[cfg(feature = "parquet")]
    ExportParquet {
        /// Input telemetry CSV (standard Apex telemetry format)
        #[arg(long)]
        telemetry: PathBuf,

        /// Output `.parquet` file path
        #[arg(long)]
        out: PathBuf,
    },

    /// Identify car parameters by fitting the QSS driven-line sim to measured speed
    Identify {
        /// Aligned telemetry CSV (standard format)
        #[arg(long)]
        telemetry: PathBuf,

        /// Track file (smoothed Apex-14 JSON or TUMFTM CSV)
        #[arg(long)]
        track: PathBuf,

        /// Optional v2 **3D** track JSON. When given, every LM iteration runs the
        /// 3D QSS on the elevated driven line; `--track` is ignored.
        #[arg(long)]
        track_3d: Option<PathBuf>,

        /// Use calibrated F1 2024 parameters as the base
        #[arg(long, default_value_t = false)]
        calibrated: bool,

        /// Base car configuration file (TOML), overrides the preset
        #[arg(long)]
        car: Option<PathBuf>,

        /// Comma-separated free-parameter paths, e.g.
        /// "aero.lift_coeff,aero.drag_coeff,powertrain.power_scale"
        #[arg(long)]
        free: String,

        /// Output fitted-car TOML (overlay) path
        #[arg(long)]
        out: PathBuf,

        /// Driven-line reconstruction: `direct` (default) or `offset`
        #[arg(long, default_value = "direct")]
        driven_line: String,

        /// Offset mode: n(s) moving-average half-width (m)
        #[arg(long, default_value_t = apex_correlate::DEFAULT_N_FILTER_WINDOW_M)]
        n_filter: f64,

        /// Direct mode: smoothing deviation budget (m)
        #[arg(long, default_value_t = apex_correlate::DEFAULT_DRIVEN_SMOOTH_TOLERANCE_M)]
        driven_smooth_tolerance: f64,

        /// Common-grid resample step (m)
        #[arg(long, default_value_t = 10.0)]
        grid_step: f64,
    },

    /// Infer unmeasured channels (accel, loads, grip, power) from measured speed
    Infer {
        /// Aligned telemetry CSV (standard format, with x/y)
        #[arg(long)]
        telemetry: PathBuf,

        /// Track file (smoothed Apex-14 JSON or TUMFTM CSV)
        #[arg(long)]
        track: PathBuf,

        /// Optional v2 **3D** track JSON. When given, the inferred load/grip/power
        /// channels pick up the 3D terms (compression, grade); `--track` ignored.
        #[arg(long)]
        track_3d: Option<PathBuf>,

        /// Fitted car TOML (overlay applied on the calibrated preset)
        #[arg(long)]
        car: PathBuf,

        /// Output CSV path (input channels + inferred channels)
        #[arg(long)]
        out: PathBuf,

        /// Driven-line reconstruction: `direct` (default) or `offset`
        #[arg(long, default_value = "direct")]
        driven_line: String,

        /// Offset mode: n(s) moving-average half-width (m)
        #[arg(long, default_value_t = apex_correlate::DEFAULT_N_FILTER_WINDOW_M)]
        n_filter: f64,

        /// Direct mode: smoothing deviation budget (m)
        #[arg(long, default_value_t = apex_correlate::DEFAULT_DRIVEN_SMOOTH_TOLERANCE_M)]
        driven_smooth_tolerance: f64,
    },

    /// Estimate smoothed dynamic states (slip angles, yaw rate, body slip) with
    /// an RTS single-track Kalman smoother over the measured lap
    Estimate {
        /// Aligned telemetry CSV (standard format, with t/x/y/speed)
        #[arg(long)]
        telemetry: PathBuf,

        /// Track file (smoothed Apex-14 JSON or TUMFTM CSV)
        #[arg(long)]
        track: PathBuf,

        /// Fitted car TOML (overlay applied on the calibrated preset)
        #[arg(long)]
        car: PathBuf,

        /// Output CSV path (measured channels + smoothed states)
        #[arg(long)]
        out: PathBuf,

        /// Position measurement noise sigma (m). Default 3.0 (below the ~4 m
        /// telemetry align RMS).
        #[arg(long, default_value_t = 3.0)]
        pos_sigma: f64,

        /// Disable the course (motion-direction) pseudo-measurement. Not
        /// recommended — the heading diverges without it.
        #[arg(long, default_value_t = false)]
        no_course: bool,
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

    /// Generate the g-g-g performance envelope over (theta, v, g_z) and cache it
    Envelope {
        /// Car configuration file (TOML)
        #[arg(short, long)]
        car: Option<PathBuf>,

        /// Use calibrated F1 2024 parameters
        #[arg(long, default_value_t = false)]
        calibrated: bool,

        /// Speed range (m/s) as MIN:MAX
        #[arg(long, default_value = "5:90")]
        v_range: String,

        /// Vertical-acceleration range (m/s²) as MIN:MAX
        #[arg(long, default_value = "8:14")]
        gz_range: String,

        /// Grid resolution as THETA:V:GZ (angle:speed:g_z samples)
        #[arg(long, default_value = "24:10:6")]
        resolution: String,

        /// Cache directory for the versioned envelope binary
        #[arg(long, default_value = ".apex-cache/envelope")]
        cache_dir: PathBuf,

        /// Export a g-g diagram SVG (boundary polygons at selected slices)
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
            no_smooth,
            smooth_tolerance,
            elevation,
        } => cmd_import_track(input, output, name, !no_smooth, smooth_tolerance, elevation),
        Commands::Correlate {
            telemetry,
            track,
            track_3d,
            calibrated,
            car,
            grid_step,
            line,
            driven_line,
            n_filter,
            driven_smooth_tolerance,
            out_dir,
            #[cfg(feature = "parquet")]
            parquet,
        } => cmd_correlate(
            telemetry,
            track,
            track_3d,
            calibrated,
            car,
            grid_step,
            line,
            driven_line,
            n_filter,
            driven_smooth_tolerance,
            out_dir,
            #[cfg(feature = "parquet")]
            parquet,
        ),
        Commands::ExportLd {
            telemetry,
            out,
            rate,
        } => cmd_export_ld(telemetry, out, rate),
        #[cfg(feature = "parquet")]
        Commands::ExportParquet { telemetry, out } => cmd_export_parquet(telemetry, out),
        Commands::Identify {
            telemetry,
            track,
            track_3d,
            calibrated,
            car,
            free,
            out,
            driven_line,
            n_filter,
            driven_smooth_tolerance,
            grid_step,
        } => cmd_identify(
            telemetry,
            track,
            track_3d,
            calibrated,
            car,
            free,
            out,
            driven_line,
            n_filter,
            driven_smooth_tolerance,
            grid_step,
        ),
        Commands::Infer {
            telemetry,
            track,
            track_3d,
            car,
            out,
            driven_line,
            n_filter,
            driven_smooth_tolerance,
        } => cmd_infer(
            telemetry,
            track,
            track_3d,
            car,
            out,
            driven_line,
            n_filter,
            driven_smooth_tolerance,
        ),
        Commands::Estimate {
            telemetry,
            track,
            car,
            out,
            pos_sigma,
            no_course,
        } => cmd_estimate(telemetry, track, car, out, pos_sigma, !no_course),
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
        Commands::Envelope {
            car,
            calibrated,
            v_range,
            gz_range,
            resolution,
            cache_dir,
            svg,
        } => cmd_envelope(
            car, calibrated, v_range, gz_range, resolution, cache_dir, svg,
        ),
    }
}

/// Parse a `MIN:MAX` float pair.
fn parse_range(s: &str, what: &str) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let (a, b) = s
        .split_once(':')
        .ok_or_else(|| format!("{what} must be MIN:MAX, got '{s}'"))?;
    Ok((a.trim().parse()?, b.trim().parse()?))
}

/// Parse a `THETA:V:GZ` usize triple.
fn parse_resolution(s: &str) -> Result<(usize, usize, usize), Box<dyn std::error::Error>> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        return Err(format!("resolution must be THETA:V:GZ, got '{s}'").into());
    }
    Ok((
        parts[0].trim().parse()?,
        parts[1].trim().parse()?,
        parts[2].trim().parse()?,
    ))
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

#[allow(clippy::too_many_arguments)]
fn cmd_envelope(
    car: Option<PathBuf>,
    calibrated: bool,
    v_range: String,
    gz_range: String,
    resolution: String,
    cache_dir: PathBuf,
    svg: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    use apex_physics::{Envelope, EnvelopeGridSpec};

    let params = load_car_params(car, calibrated)?;
    // Tire / suspension / aero models: the operating-point trim needs all four;
    // the car TOML carries CarParams, the rest use the F1 model defaults.
    let tire = apex_physics::PacejkaTire::f1_default();
    let suspension = apex_physics::SuspensionSystem::f1_default();
    let aero = apex_physics::AeroModel::f1_default();

    let (v_min, v_max) = parse_range(&v_range, "v-range")?;
    let (gz_min, gz_max) = parse_range(&gz_range, "gz-range")?;
    let (theta_res, v_res, gz_res) = parse_resolution(&resolution)?;

    let spec = EnvelopeGridSpec {
        theta_res,
        v_min,
        v_max,
        v_res,
        gz_min,
        gz_max,
        gz_res,
        ..Default::default()
    };
    spec.validate()?;

    println!(
        "Envelope grid: theta={} x v={} x g_z={}  ({} operating points)",
        theta_res,
        v_res,
        gz_res,
        spec.total()
    );
    println!("  v in [{v_min}, {v_max}] m/s,  g_z in [{gz_min}, {gz_max}] m/s^2");

    let t0 = std::time::Instant::now();
    let (env, from_cache) =
        Envelope::generate_cached(&params, &tire, &suspension, &aero, spec, &cache_dir)?;
    let elapsed = t0.elapsed();

    let path = cache_dir.join(format!("{}.apexenv", env.key().to_hex()));
    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    if from_cache {
        println!(
            "Loaded cached envelope in {:.1} ms",
            elapsed.as_secs_f64() * 1e3
        );
    } else {
        println!(
            "Generated envelope in {:.1} ms",
            elapsed.as_secs_f64() * 1e3
        );
    }
    println!("  key:  {}", env.key().to_hex());
    println!("  file: {} ({} bytes)", path.display(), size);

    // Report a couple of sample boundary radii for a quick sanity read.
    let mid_v = 0.5 * (v_min + v_max);
    let gz_ref = if (gz_min..=gz_max).contains(&9.81) {
        9.81
    } else {
        gz_min
    };
    println!(
        "  rho @ v={mid_v:.0}, g_z={gz_ref:.2}:  lateral={:.1}  brake={:.1}  drive={:.1} m/s^2",
        env.rho(std::f64::consts::FRAC_PI_2, mid_v, gz_ref),
        env.rho(std::f64::consts::PI, mid_v, gz_ref),
        env.rho(0.0, mid_v, gz_ref),
    );

    if let Some(svg_path) = svg {
        let meta = apex_telemetry::RunMetadata::new(
            apex_physics::car_params_hash(&params),
            apex_telemetry::settings_hash_for_mode("envelope.no-track"),
            env.key(),
            None,
        );
        // Three speed slices at the reference g_z; sample the boundary finely.
        let speeds = [v_min, mid_v, v_max];
        let slices: Vec<apex_telemetry::EnvelopeSlicePlot> = speeds
            .iter()
            .map(|&v| {
                let n = 128;
                let boundary: Vec<(f64, f64)> = (0..=n)
                    .map(|i| {
                        let theta = i as f64 / n as f64 * std::f64::consts::TAU;
                        let r = env.rho(theta, v, gz_ref);
                        (r * theta.sin(), r * theta.cos()) // (a_y, a_x)
                    })
                    .collect();
                apex_telemetry::EnvelopeSlicePlot {
                    label: format!("v={v:.0} m/s, g_z={gz_ref:.1}"),
                    boundary,
                }
            })
            .collect();
        apex_telemetry::render_envelope_svg(
            &svg_path,
            &meta,
            &slices,
            &format!("g-g envelope ({} kg)", params.mass as i32),
        )?;
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
    smooth: bool,
    smooth_tolerance: f64,
    elevation: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(elev) = elevation {
        return cmd_import_track_elevation(input, elev, output, name);
    }
    println!("Importing track '{}' from {}", name, input.display());
    let raw = apex_track::load_tumftm_csv(&input, &name)?;
    println!(
        "Track length: {:.3} km, {} segments",
        raw.total_length / 1000.0,
        raw.segments.len()
    );

    let (track, meta) = if smooth {
        let (smoothed, report) = apex_track::smooth_track(&raw, smooth_tolerance);
        print_smoothing_diagnostics(&report);
        let meta = apex_track::TrackMetaJson {
            source: Some("TUMFTM racetrack-database".to_string()),
            smoothed: Some(true),
            smooth_tolerance_m: Some(report.tolerance_m),
            smooth_lambda: Some(report.lambda),
            smooth_max_deviation_m: Some(report.max_deviation_m),
        };
        (smoothed, Some(meta))
    } else {
        println!("Smoothing: DISABLED (--no-smooth)");
        let meta = apex_track::TrackMetaJson {
            source: Some("TUMFTM racetrack-database".to_string()),
            smoothed: Some(false),
            ..Default::default()
        };
        (raw, Some(meta))
    };

    let json = apex_track::export_track_json_with_meta(&track, meta)?;
    std::fs::write(&output, json)?;
    println!("Exported to {}", output.display());

    Ok(())
}

/// Minimal view of a `tools/fetch_elevation.py` sidecar (only the fields the
/// Rust merge needs). Network/DEM/smoothing all happen in Python; this side is
/// a deterministic, offline merge.
#[derive(serde::Deserialize)]
struct ElevationSidecar {
    circuit: String,
    dem_dataset: String,
    z: Vec<f64>,
    #[serde(default)]
    banking_deg: f64,
    #[serde(default)]
    elevation_range_m: f64,
}

/// Minimal view of a `tools/georef.py` sidecar (only the quality metrics the
/// Rust merge gates on). Optional: absent if the elevation sidecar wasn't
/// itself derived through the georeferencing step (e.g. a synthetic track).
#[derive(serde::Deserialize)]
struct GeorefSidecar {
    scale: f64,
    coverage_rms_m: f64,
    residual_rms_m: f64,
}

/// Coverage-RMS target used as a soft quality gate (tracks/README.md's
/// "sub-DEM-cell target"): a 25-30 m DEM cell can't usefully resolve
/// georeferencing error above this.
const GEOREF_COVERAGE_RMS_TARGET_M: f64 = 15.0;
/// Acceptable similarity-transform scale range (near-1:1 local-metres ->
/// real-world; large deviations indicate a bad fit, e.g. wrong bbox or units).
const GEOREF_SCALE_RANGE: std::ops::RangeInclusive<f64> = 0.95..=1.05;

/// Load the georef sidecar next to the elevation sidecar (same convention as
/// `fetch_elevation.py`/`georef.py`: `tracks/<circuit>.georef.json`), and
/// **hard-warn** (stderr, non-fatal) if its quality metrics miss the
/// documented targets. Silently does nothing if the file isn't present —
/// georeferencing is optional groundwork, not a hard requirement of the
/// elevation merge.
fn warn_on_georef_quality(elevation_path: &std::path::Path, circuit: &str) {
    let dir = elevation_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let georef_path = dir.join(format!("{circuit}.georef.json"));
    let Ok(contents) = std::fs::read_to_string(&georef_path) else {
        return;
    };
    let sidecar: GeorefSidecar = match serde_json::from_str(&contents) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "WARNING: found '{}' but couldn't parse it ({e}) — skipping georef quality gate",
                georef_path.display()
            );
            return;
        }
    };
    println!(
        "  georef quality: coverage_rms={:.1} m, full_rms={:.1} m, scale={:.5}",
        sidecar.coverage_rms_m, sidecar.residual_rms_m, sidecar.scale
    );
    if sidecar.coverage_rms_m > GEOREF_COVERAGE_RMS_TARGET_M {
        eprintln!(
            "WARNING: georef coverage_rms {:.1} m exceeds the {:.0} m sub-DEM-cell target \
             ({}) — the georeferencing fit may be unreliable for this circuit",
            sidecar.coverage_rms_m,
            GEOREF_COVERAGE_RMS_TARGET_M,
            georef_path.display()
        );
    }
    if !GEOREF_SCALE_RANGE.contains(&sidecar.scale) {
        eprintln!(
            "WARNING: georef scale {:.5} is outside the expected [{:.2}, {:.2}] range \
             ({}) — check the OSM bbox / units",
            sidecar.scale,
            GEOREF_SCALE_RANGE.start(),
            GEOREF_SCALE_RANGE.end(),
            georef_path.display()
        );
    }
}

/// `import-track --elevation`: merge a pre-fetched elevation sidecar into the 2D
/// centerline (loaded from the track JSON it was computed against) and write a
/// **v2 3D** track JSON. Then reload it via `load_ribbon3d_json` and print the
/// geometry-validation report (the load-and-validate path on real data).
fn cmd_import_track_elevation(
    input: PathBuf,
    elevation: PathBuf,
    output: PathBuf,
    name: String,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Merging elevation '{}' into '{}' (2D source {})",
        elevation.display(),
        name,
        input.display()
    );
    // The sidecar's z is aligned 1:1 to the 2D track JSON's points, so load the
    // JSON as-is (no re-smoothing, which would desynchronize the indexing).
    let track = apex_track::load_track_json(&input)?;
    let sidecar: ElevationSidecar = serde_json::from_str(&std::fs::read_to_string(&elevation)?)?;
    println!(
        "  elevation source: {} (DEM {}), range {:.1} m",
        sidecar.circuit, sidecar.dem_dataset, sidecar.elevation_range_m
    );
    warn_on_georef_quality(&elevation, &sidecar.circuit);
    if sidecar.z.len() != track.segments.len() {
        return Err(format!(
            "elevation sidecar has {} z samples but the track has {} points — \
             the sidecar must be computed against this exact 2D centerline",
            sidecar.z.len(),
            track.segments.len()
        )
        .into());
    }

    // Assemble the 3D centerline (x, y, z) and build a Ribbon3d (bank = 0).
    let pts: Vec<[f64; 3]> = track
        .segments
        .iter()
        .zip(&sidecar.z)
        .map(|(seg, &z)| [seg.x, seg.y, z])
        .collect();
    let bank = vec![sidecar.banking_deg.to_radians(); pts.len()];
    let wl: Vec<f64> = track.segments.iter().map(|s| s.width_left).collect();
    let wr: Vec<f64> = track.segments.iter().map(|s| s.width_right).collect();
    let ribbon =
        apex_track::Ribbon3d::from_centerline_3d(&name, &pts, &bank, &wl, &wr, track.is_closed);

    let json = apex_track::export_ribbon3d_json(&ribbon)?;
    std::fs::write(&output, &json)?;
    println!("Exported v2 3D track to {}", output.display());

    // Load-and-validate the written file (round-trip through load_ribbon3d_json).
    let reloaded = apex_track::load_ribbon3d_json(&output)?;
    let v = reloaded.validate();
    let len_2d = track.total_length;
    println!("  validation (reloaded via load_ribbon3d_json):");
    println!("    stations:          {}  closed={}", v.n, v.is_closed);
    println!("    all finite:        {}", v.all_finite);
    println!(
        "    frame ortho error: {:.2e} (0 = perfect orthonormality)",
        v.max_ortho_error
    );
    println!(
        "    length 2D -> 3D:   {:.1} m -> {:.1} m  (+{:.1} m)",
        len_2d,
        v.length_3d,
        v.length_3d - len_2d
    );
    println!("    elevation range:   {:.1} m", v.elevation_range);
    println!(
        "    |Omega_y| p95/max: {:.5} / {:.5} 1/m   |Omega_z| p95: {:.5} 1/m",
        v.omega_y_p95, v.omega_y_max, v.omega_z_p95
    );
    Ok(())
}

/// Print before/after smoothing diagnostics (radii, curvature percentiles,
/// deviation, length).
fn print_smoothing_diagnostics(r: &apex_track::SmoothingReport) {
    let rkm = |k: f64| if k > 1e-9 { 1.0 / k } else { f64::INFINITY };
    println!(
        "Smoothing: ON (tolerance {:.2} m, lambda {:.1}, max deviation {:.3} m)",
        r.tolerance_m, r.lambda, r.max_deviation_m
    );
    println!(
        "  min radius:   {:.1} m -> {:.1} m",
        r.min_radius_before, r.min_radius_after
    );
    println!(
        "  |kappa| p50:  {:.5} (R {:.0} m) -> {:.5} (R {:.0} m)",
        r.kappa_p50_before,
        rkm(r.kappa_p50_before),
        r.kappa_p50_after,
        rkm(r.kappa_p50_after)
    );
    println!(
        "  |kappa| p95:  {:.5} (R {:.0} m) -> {:.5} (R {:.0} m)",
        r.kappa_p95_before,
        rkm(r.kappa_p95_before),
        r.kappa_p95_after,
        rkm(r.kappa_p95_after)
    );
    println!(
        "  |kappa| max:  {:.5} (R {:.1} m) -> {:.5} (R {:.1} m)",
        r.kappa_max_before,
        rkm(r.kappa_max_before),
        r.kappa_max_after,
        rkm(r.kappa_max_after)
    );
    let dlen = r.length_after - r.length_before;
    println!(
        "  length:       {:.1} m -> {:.1} m ({:+.2} m, {:+.3}%)",
        r.length_before,
        r.length_after,
        dlen,
        100.0 * dlen / r.length_before
    );
}

/// Parse the `--driven-line` selector into a [`DrivenLineMode`].
fn driven_line_mode(
    driven_line: &str,
    n_filter: f64,
    smooth_tol: f64,
) -> Result<apex_correlate::DrivenLineMode, Box<dyn std::error::Error>> {
    use apex_correlate::DrivenLineMode;
    match driven_line {
        "direct" => Ok(DrivenLineMode::Direct(smooth_tol)),
        "offset" => Ok(DrivenLineMode::Offset(n_filter)),
        other => Err(format!("--driven-line must be `direct` or `offset`, got `{other}`").into()),
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_correlate(
    telemetry: PathBuf,
    track: PathBuf,
    track_3d: Option<PathBuf>,
    calibrated: bool,
    car: Option<PathBuf>,
    grid_step: f64,
    line: String,
    driven_line: String,
    n_filter: f64,
    driven_smooth_tolerance: f64,
    out_dir: PathBuf,
    #[cfg(feature = "parquet")] parquet: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    use apex_correlate::report::{correlate, write_report, CorrelationConfig, SimTrace};
    use apex_correlate::{driven_sim_trace_3d, import_telemetry, Mapping};

    // 1. Measured telemetry (aligned, standard format → identity mapping).
    let measured = import_telemetry(&telemetry, &Mapping::identity())?;
    // 2. Track (+ optional 3D elevation) + car.
    let ribbon = match &track_3d {
        Some(p) => Some(apex_track::load_ribbon3d_json(p)?),
        None => None,
    };
    if let Some(r) = &ribbon {
        let v = r.validate();
        println!(
            "3D track: {} stations, elevation range {:.1} m, 3D length {:.1} m — QSS runs in 3D",
            v.n, v.elevation_range, v.length_3d
        );
    }
    let trk = match &ribbon {
        Some(r) => r.to_track_2d(),
        None => load_track_from_path(&track)?,
    };
    let elevation = ribbon.as_ref();
    let params = load_car_params(car, calibrated)?;

    // 3. Build the sim trace on the requested line.
    let trace = match line.as_str() {
        "centerline" => {
            let sim = match elevation {
                Some(r) => apex_physics::qss_lap_sim_3d(r, &params),
                None => apex_physics::qss_lap_sim(&trk, &params),
            };
            SimTrace::from_qss(&sim)
        }
        "measured" => {
            let mode = driven_line_mode(&driven_line, n_filter, driven_smooth_tolerance)?;
            let dr = driven_sim_trace_3d(&trk, &measured, &params, mode, elevation)?;
            println!(
                "Driven line: length {:.1} m (centerline {:.1} m, {:+.1} m); {}",
                dr.driven_length,
                dr.centerline_length,
                dr.driven_length - dr.centerline_length,
                dr.detail
            );
            dr.trace
        }
        other => {
            return Err(format!("--line must be `centerline` or `measured`, got `{other}`").into())
        }
    };

    // 4. Sim-side provenance (RunMetadata). QSS has no RNG / tunable solver.
    let meta = apex_telemetry::RunMetadata::new(
        apex_physics::car_params_hash(&params),
        apex_track::processed_track_hash(&trk),
        apex_telemetry::settings_hash_for_mode(&format!("correlate.qss.grip-circle.{line}")),
        None,
    );

    let config = CorrelationConfig {
        grid_step,
        ..CorrelationConfig::default()
    };
    let result = correlate(&measured, &trk, &trace, config)?;
    println!("Line: {}", trace.label);

    // 5. Write report.md + SVGs.
    let report_path = write_report(&out_dir, &result, &measured, &meta, &trk)?;

    // Optional: aligned report grid → Parquet (measured + sim speed side by side).
    #[cfg(feature = "parquet")]
    if let Some(pq_path) = parquet {
        write_correlation_parquet(&pq_path, &result, &meta)?;
        println!("Aligned grid Parquet written to {}", pq_path.display());
    }

    // 6. Console summary.
    println!("{}", result.headline(&trk.name));
    println!();
    println!(
        "Lap: measured {:.3} s, sim {:.3} s, delta {:+.3} s",
        result.measured_lap_from_t, result.lap.sim, result.lap.delta
    );
    if let Some(d) = result.lap_time_mismatch {
        println!(
            "  !! measured header lap-time and t-span disagree by {:.3} s",
            d
        );
    }
    print!("Sectors (equal-arc thirds — NOT official F1): ");
    for i in 0..result.sectors.delta.len() {
        print!("S{} {:+.3}s  ", i + 1, result.sectors.delta[i]);
    }
    println!();
    println!(
        "Speed RMSE {:.3} m/s over {:.0} m; max |Δv| {:.2} m/s at s={:.0} m",
        result.rmse.rmse, result.span, result.rmse.max_abs, result.rmse.s_at_max
    );
    println!(
        "Sim carries most extra speed {:+.2} m/s @ s={:.0} m; most below {:+.2} m/s @ s={:.0} m",
        result.sim_fastest_dv, result.sim_fastest_s, result.sim_slowest_dv, result.sim_slowest_s
    );
    println!("\nCorners detected: {}", result.corners.len());

    // 5 largest apex-speed errors (by |delta|).
    let mut apex = result.apex.clone();
    apex.sort_by(|a, b| b.delta.abs().partial_cmp(&a.delta.abs()).unwrap());
    println!("Top-5 apex-speed errors (sim − measured):");
    for a in apex.iter().take(5) {
        println!(
            "  s={:5.0} m  measured {:5.2}  sim {:5.2}  Δ {:+.2} m/s",
            a.s, a.v_measured, a.v_sim, a.delta
        );
    }

    // 5 largest braking-point offsets (by |offset|).
    let mut brk: Vec<_> = result
        .braking
        .iter()
        .filter(|b| b.offset.is_some())
        .cloned()
        .collect();
    brk.sort_by(|a, b| {
        b.offset
            .unwrap()
            .abs()
            .partial_cmp(&a.offset.unwrap().abs())
            .unwrap()
    });
    println!("Top-5 braking-point offsets (sim − measured, + = sim brakes later):");
    for b in brk.iter().take(5) {
        println!(
            "  corner s={:5.0} m  measured onset {:5.0}  sim onset {:5.0}  offset {:+.0} m",
            b.corner_s,
            b.s_measured.unwrap(),
            b.s_sim.unwrap(),
            b.offset.unwrap()
        );
    }

    println!("\nReport written to {}", report_path.display());
    Ok(())
}

/// Load a standard telemetry CSV and return its grid, ordered columns (axis
/// channel first, then registry order — matching the CSV writer), and the
/// descriptive header metadata.
#[allow(clippy::type_complexity)]
fn load_standard_telemetry(
    telemetry: &std::path::Path,
) -> Result<
    (
        apex_correlate::GridKind,
        Vec<(apex_telemetry::ChannelId, Vec<f64>)>,
        Vec<(String, String)>,
    ),
    Box<dyn std::error::Error>,
> {
    use apex_correlate::{import_telemetry, Mapping};

    let tel = import_telemetry(telemetry, &Mapping::identity())?;
    let axis = tel.grid.axis_channel();
    let mut ordered: Vec<apex_telemetry::ChannelId> = Vec::with_capacity(tel.channels.len());
    if tel.channels.contains_key(&axis) {
        ordered.push(axis);
    }
    for &id in tel.channels.keys() {
        if id != axis {
            ordered.push(id);
        }
    }
    let columns: Vec<(apex_telemetry::ChannelId, Vec<f64>)> = ordered
        .iter()
        .map(|&id| (id, tel.channel(id).unwrap_or(&[]).to_vec()))
        .collect();
    Ok((tel.grid, columns, tel.metadata))
}

fn cmd_export_ld(
    telemetry: PathBuf,
    out: PathBuf,
    rate: Option<u16>,
) -> Result<(), Box<dyn std::error::Error>> {
    use apex_correlate::GridKind;
    use apex_telemetry::motec::{export_ld, Grid, LdOptions};

    let (grid, columns, metadata) = load_standard_telemetry(&telemetry)?;
    let grid = match grid {
        GridKind::S => Grid::Distance,
        GridKind::T => Grid::Time,
    };
    let borrowed: Vec<(apex_telemetry::ChannelId, &[f64])> =
        columns.iter().map(|(id, v)| (*id, v.as_slice())).collect();

    let opts = LdOptions {
        sample_rate_hz: rate,
        timestamp: apex_telemetry::now_rfc3339(),
    };
    let report = export_ld(&out, grid, &borrowed, &metadata, &opts)?;

    println!("Wrote MoTeC .ld: {}", out.display());
    println!(
        "  {} channels, {} samples @ {} Hz ({:.2} s)",
        report.channels.len(),
        report.n_samples,
        report.sample_rate_hz,
        report.duration_s
    );
    if report.gap_filled_samples > 0 {
        println!(
            "  {} sample(s) gap-filled (hold-last); see the `gap_fill` marker channel",
            report.gap_filled_samples
        );
    }
    println!("  channels: {}", channel_summary(&report.channels));
    Ok(())
}

/// Compact `name[unit]` summary of a channel list.
fn channel_summary(channels: &[(String, String)]) -> String {
    channels
        .iter()
        .map(|(n, u)| {
            if u.is_empty() {
                n.clone()
            } else {
                format!("{n}[{u}]")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(feature = "parquet")]
fn cmd_export_parquet(telemetry: PathBuf, out: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    use apex_telemetry::export_channels_parquet;

    let (grid, columns, metadata) = load_standard_telemetry(&telemetry)?;
    let borrowed: Vec<(apex_telemetry::ChannelId, &[f64])> =
        columns.iter().map(|(id, v)| (*id, v.as_slice())).collect();
    // Measured/inferred data: descriptive provenance only, no RunMetadata.
    export_channels_parquet(&out, &borrowed, Some(grid.as_str()), &metadata, None)?;

    let rows = columns.first().map(|(_, v)| v.len()).unwrap_or(0);
    println!("Wrote Parquet: {}", out.display());
    println!(
        "  {} columns, {} rows (grid: {})",
        columns.len(),
        rows,
        grid.as_str()
    );
    Ok(())
}

/// Write the aligned correlation grid (measured + sim speed side by side) to a
/// Parquet file. Columns are explicitly prefixed (`meas_speed` / `sim_speed`)
/// since they mix two sources; provenance is the documented hybrid (sim
/// `RunMetadata` under `run.*`, measured descriptive verbatim).
#[cfg(feature = "parquet")]
fn write_correlation_parquet(
    path: &std::path::Path,
    result: &apex_correlate::CorrelationResult,
    meta: &apex_telemetry::RunMetadata,
) -> Result<(), Box<dyn std::error::Error>> {
    use apex_telemetry::{write_parquet, ParquetColumn};

    let columns = vec![
        ParquetColumn {
            name: "s",
            unit: "m",
            data: &result.grid_s,
        },
        ParquetColumn {
            name: "meas_speed",
            unit: "m/s",
            data: &result.meas_v,
        },
        ParquetColumn {
            name: "sim_speed",
            unit: "m/s",
            data: &result.sim_v,
        },
    ];
    let mut kv: Vec<(String, String)> = vec![
        ("grid".into(), "s".into()),
        ("sim_line".into(), result.sim_label.clone()),
    ];
    // Sim-side reproducible provenance (RunMetadata) under run.* keys.
    kv.push(("run.config_hash".into(), meta.config_hash.to_hex()));
    kv.push(("run.car_hash".into(), meta.car_hash.to_hex()));
    kv.push(("run.track_hash".into(), meta.track_hash.to_hex()));
    kv.push(("run.settings_hash".into(), meta.settings_hash.to_hex()));
    kv.push(("run.git_sha".into(), meta.git_sha.clone()));
    kv.push(("run.apex_version".into(), meta.apex_version.clone()));
    kv.push(("run.timestamp".into(), meta.timestamp.clone()));
    // Measured-side descriptive provenance (verbatim), prefixed to avoid clashes.
    for (k, v) in &result.measured_meta {
        kv.push((format!("measured.{k}"), v.clone()));
    }
    write_parquet(path, &columns, &kv)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_identify(
    telemetry: PathBuf,
    track: PathBuf,
    track_3d: Option<PathBuf>,
    calibrated: bool,
    car: Option<PathBuf>,
    free: String,
    out: PathBuf,
    driven_line: String,
    n_filter: f64,
    driven_smooth_tolerance: f64,
    grid_step: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    use apex_correlate::{identify_3d, import_telemetry, parse_free_param, Mapping};

    let measured = import_telemetry(&telemetry, &Mapping::identity())?;
    let ribbon = match &track_3d {
        Some(p) => Some(apex_track::load_ribbon3d_json(p)?),
        None => None,
    };
    if ribbon.is_some() {
        println!("Fitting in 3D (grade / vertical-curvature load / banking).");
    }
    let trk = match &ribbon {
        Some(r) => r.to_track_2d(),
        None => load_track_from_path(&track)?,
    };
    let elevation = ribbon.as_ref();
    let base = load_car_params(car, calibrated)?;
    let mode = driven_line_mode(&driven_line, n_filter, driven_smooth_tolerance)?;

    // Parse the free-parameter list.
    let paths: Vec<String> = free
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if paths.contains(&"tires.mu".to_string()) {
        println!(
            "note: `tires.mu` is in the headline free set — see the residual analysis; \
             μ is normally kept FIXED so it can't mask the aero deficit."
        );
    }
    let free_params: Vec<_> = paths
        .iter()
        .map(|p| parse_free_param(p, &base))
        .collect::<Result<_, _>>()?;

    // --- Headline fit ---
    println!("=== Identify (headline fit) ===");
    println!("Driven line: {}", mode_label(&driven_line));
    let res = identify_3d(
        &trk,
        &measured,
        &base,
        free_params.clone(),
        mode,
        grid_step,
        elevation,
    )?;
    print_fit_report(&res);
    println!(
        "  runtime: {:.3} s total, {:.4} s/iter ({} iterations, {} grid points)",
        res.total_seconds,
        res.seconds_per_iter,
        res.lm.iterations.len(),
        res.grid_len
    );

    // Guardrail: a bound-pinned fit is not a shippable car.
    if !res.lm.bound_pinned.is_empty() {
        let names: Vec<&str> = res
            .lm
            .bound_pinned
            .iter()
            .map(|&i| res.free[i].path.as_str())
            .collect();
        return Err(format!(
            "STOP: fit pinned to a bound for {names:?} — the fit is unreliable \
             (a parameter hit its physical limit). Not writing {}. Investigate \
             (wrong free set, bad line, or a model gap) before shipping.",
            out.display()
        )
        .into());
    }

    // Write the fitted overlay TOML (our own derived params — committable).
    let toml = fitted_car_toml(&res, &base, &measured, &driven_line);
    std::fs::write(&out, toml)?;
    println!("\nWrote fitted car overlay: {}", out.display());

    // --- Diagnostic: add tires.mu free (evidence for the μ-fixed guardrail) ---
    if !paths.contains(&"tires.mu".to_string()) {
        println!("\n=== Diagnostic fit (μ also free) ===");
        let mut diag = free_params.clone();
        diag.push(parse_free_param("tires.mu", &base)?);
        let dres = identify_3d(&trk, &measured, &base, diag, mode, grid_step, elevation)?;
        print_fit_report(&dres);
        let mu_idx = dres.free.iter().position(|f| f.path == "tires.mu").unwrap();
        let mu_fit = dres.lm.params[mu_idx];
        let mu_se = dres.lm.std_errors[mu_idx];
        println!(
            "  μ moved {:.3} → {:.3} (± {:.3}); headline cost {:.1} vs μ-free cost {:.1}; \
             condition number {:.2e} → {:.2e}",
            base.tire_mu,
            mu_fit,
            mu_se,
            res.lm.cost,
            dres.lm.cost,
            res.lm.condition_number,
            dres.lm.condition_number
        );
        let verdict = if (mu_fit - base.tire_mu).abs() < 0.05
            && dres.lm.condition_number < 10.0 * res.lm.condition_number
        {
            "μ barely moves and identifiability holds — the μ-fixed guardrail is low-cost."
        } else {
            "freeing μ moves it and/or degrades identifiability — the guardrail is justified."
        };
        println!("  verdict: {verdict}");
    }

    Ok(())
}

/// A friendly driven-line label for the report header.
fn mode_label(driven_line: &str) -> &str {
    match driven_line {
        "direct" => "measured (direct, smoothed x/y)",
        "offset" => "measured (offset, centerline + n)",
        other => other,
    }
}

/// Print the initial→final parameters (± std error), cost, and identifiability
/// flags of an identification fit.
fn print_fit_report(res: &apex_correlate::IdentifyResult) {
    println!(
        "  cost: {:.2} → {:.2}  ({} obs, {} iters, converged={})",
        res.lm.initial_cost,
        res.lm.cost,
        res.lm.n_residuals,
        res.lm.iterations.len(),
        res.lm.converged
    );
    println!("  parameters (initial → fitted ± std err):");
    for (i, f) in res.free.iter().enumerate() {
        let fitted = res.lm.params[i];
        let se = res.lm.std_errors[i];
        let weak = if res.lm.weak_params.contains(&i) {
            "  [weakly identifiable: σ > 50% of value]"
        } else {
            ""
        };
        let pinned = if res.lm.bound_pinned.contains(&i) {
            "  [BOUND-PINNED]"
        } else {
            ""
        };
        println!(
            "    {:<24} {:>10.4} → {:>10.4} ± {:<10.4}{}{}",
            f.path, f.initial, fitted, se, weak, pinned
        );
    }
    println!("  condition number (JᵀJ): {:.3e}", res.lm.condition_number);
    if res.lm.weak_pairs.is_empty() {
        println!("  no strongly-correlated parameter pairs (|corr| ≤ 0.95)");
    } else {
        for &(i, j, c) in &res.lm.weak_pairs {
            println!(
                "  !! |corr({}, {})| = {:.3} > 0.95 (jointly weakly identifiable)",
                res.free[i].path, res.free[j].path, c
            );
        }
    }
}

/// Build the fitted-car overlay TOML: only the fitted fields + provenance
/// comments. Loads on top of the calibrated preset.
fn fitted_car_toml(
    res: &apex_correlate::IdentifyResult,
    base: &apex_physics::CarParams,
    measured: &apex_correlate::Telemetry,
    driven_line: &str,
) -> String {
    use apex_correlate::ParamKind;
    let meta = |k: &str| -> String {
        measured
            .metadata
            .iter()
            .find(|(mk, _)| mk == k)
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| "?".to_string())
    };
    let fixed: Vec<&str> = [
        "aero.lift_coeff",
        "aero.drag_coeff",
        "tires.mu",
        "powertrain.power_scale",
    ]
    .into_iter()
    .filter(|p| !res.free.iter().any(|f| f.path == *p))
    .collect();

    let mut s = String::new();
    s.push_str("# Fitted car parameters (OVERLAY on the calibrated preset).\n");
    s.push_str("# These are our own derived parameters (no raw telemetry) — committable.\n");
    s.push_str(&format!(
        "# source: {} {} {} {} driver {} lap {}\n",
        meta("source"),
        meta("year"),
        meta("event"),
        meta("session"),
        meta("driver"),
        meta("lap"),
    ));
    s.push_str(&format!("# driven line: {}\n", mode_label(driven_line)));
    s.push_str(&format!("# fixed params: {}\n", fixed.join(", ")));
    s.push_str(&format!(
        "# fit: cost {:.2} → {:.2}, {} iterations, condition {:.2e}\n",
        res.lm.initial_cost,
        res.lm.cost,
        res.lm.iterations.len(),
        res.lm.condition_number
    ));
    s.push_str(&format!("# date: {}\n", apex_telemetry::now_rfc3339()));
    s.push_str("# NOTE: apply with --car on top of --calibrated.\n\n");

    // Group fitted fields into sections.
    let get = |kind: ParamKind| -> Option<(f64, f64)> {
        res.free
            .iter()
            .position(|f| f.kind == kind)
            .map(|i| (res.lm.params[i], res.lm.std_errors[i]))
    };
    if let (Some((lift, lse)), drag) = (get(ParamKind::LiftCoeff), get(ParamKind::DragCoeff)) {
        s.push_str("[aero]\n");
        s.push_str(&format!("lift_coeff = {lift:.5}  # ± {lse:.5}\n"));
        if let Some((drag, dse)) = drag {
            s.push_str(&format!("drag_coeff = {drag:.5}  # ± {dse:.5}\n"));
        }
        s.push('\n');
    } else if let Some((drag, dse)) = get(ParamKind::DragCoeff) {
        s.push_str("[aero]\n");
        s.push_str(&format!("drag_coeff = {drag:.5}  # ± {dse:.5}\n\n"));
    }
    if let Some((mu, mse)) = get(ParamKind::TireMu) {
        s.push_str("[tires]\n");
        s.push_str(&format!("mu = {mu:.5}  # ± {mse:.5}\n\n"));
    }
    if let Some((scale, sse)) = get(ParamKind::PowerScale) {
        s.push_str("[car]\n");
        s.push_str(&format!(
            "max_drive_force = {:.1}  # power_scale {scale:.5} ± {sse:.5} × base {:.1}\n",
            base.max_drive_force * scale,
            base.max_drive_force
        ));
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn cmd_infer(
    telemetry: PathBuf,
    track: PathBuf,
    track_3d: Option<PathBuf>,
    car: PathBuf,
    out: PathBuf,
    driven_line: String,
    n_filter: f64,
    driven_smooth_tolerance: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    use apex_correlate::{
        import_telemetry, infer_on_driven_3d, write_telemetry_csv, InferConfig, Mapping,
    };
    use apex_telemetry::ChannelId;

    const INFER_VERSION: &str = "1.0.0";

    let measured = import_telemetry(&telemetry, &Mapping::identity())?;
    let ribbon = match &track_3d {
        Some(p) => Some(apex_track::load_ribbon3d_json(p)?),
        None => None,
    };
    if ribbon.is_some() {
        println!("Inferring in 3D (grade / vertical-curvature load / grade-power).");
    }
    let trk = match &ribbon {
        Some(r) => r.to_track_2d(),
        None => load_track_from_path(&track)?,
    };
    // Fitted TOML is an overlay on the calibrated preset.
    let params = load_car_params(Some(car.clone()), true)?;
    let mode = driven_line_mode(&driven_line, n_filter, driven_smooth_tolerance)?;

    let mut res = infer_on_driven_3d(
        &trk,
        &measured,
        &params,
        mode,
        &InferConfig::default(),
        ribbon.as_ref(),
    )?;

    // Descriptive provenance + the effective-parameter caveat (derived-from-
    // measured: no RunMetadata sim block; see docs/telemetry_format.md).
    let car_hash = apex_physics::car_params_hash(&params);
    res.telemetry.metadata.push((
        "inference".into(),
        format!("apex-14 infer v{INFER_VERSION}"),
    ));
    res.telemetry
        .metadata
        .push(("infer_car_file".into(), car.display().to_string()));
    res.telemetry
        .metadata
        .push(("infer_car_hash".into(), car_hash.to_hex()));
    res.telemetry
        .metadata
        .push(("infer_driven_line".into(), driven_line.clone()));
    res.telemetry.metadata.push((
        "inference_caveat".into(),
        "inferred aero/downforce/loads/power derive from the fitted EFFECTIVE \
         coefficients and inherit their absorption of point-mass model limitations \
         — model-consistent estimates, NOT measurements"
            .into(),
    ));

    write_telemetry_csv(&out, &res.telemetry)?;

    // --- sanity report ---
    let ch = |id: ChannelId| res.telemetry.channel(id).unwrap_or(&[]).to_vec();
    let s = ch(ChannelId::S);
    let grip = ch(ChannelId::GripUtil);
    let tp = ch(ChannelId::TractivePower);
    let bp = ch(ChannelId::BrakingPower);
    let long_g = ch(ChannelId::LongitudinalG);

    println!("Inferred {} samples → {}", res.len, out.display());
    println!(
        "  car: {} (hash {})  driven line: {}",
        car.display(),
        car_hash.short(),
        driven_line
    );
    println!("  ⚠ effective-parameter estimates, NOT measurements (see header caveat).");
    let lat_g = ch(ChannelId::LateralG);
    // |lateral_g| percentiles (robust to reconstruction curvature spikes).
    let mut latmag: Vec<f64> = lat_g
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| x.abs())
        .collect();
    latmag.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |v: &[f64], p: f64| {
        if v.is_empty() {
            f64::NAN
        } else {
            v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)]
        }
    };
    println!(
        "  lateral g: p95 {:.2}  p99 {:.2}  max {:.2} @ s={:.0} m",
        pct(&latmag, 0.95),
        pct(&latmag, 0.99),
        res.peak_lat_g,
        res.peak_lat_g_s
    );
    let mut brk: Vec<f64> = long_g
        .iter()
        .filter(|x| x.is_finite() && **x < 0.0)
        .map(|x| -x)
        .collect();
    brk.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let acc_max = long_g
        .iter()
        .cloned()
        .filter(|x| x.is_finite())
        .fold(0.0, f64::max);
    println!(
        "  braking g: p95 {:.2}  max {:.2} @ s={:.0} m    accel g: max {:.2}",
        pct(&brk, 0.95),
        res.peak_brake_g,
        res.peak_brake_g_s,
        acc_max
    );

    // grip-util distribution + over-limit locations.
    let mut g: Vec<f64> = grip.iter().cloned().filter(|x| x.is_finite()).collect();
    g.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !g.is_empty() {
        let pct = |p: f64| g[((p * (g.len() - 1) as f64).round() as usize).min(g.len() - 1)];
        println!(
            "  grip util: p50 {:.2}  p95 {:.2}  max {:.2}",
            pct(0.50),
            pct(0.95),
            pct(1.0)
        );
        let over: Vec<f64> = (0..grip.len())
            .filter(|&i| grip[i].is_finite() && grip[i] > 1.05)
            .map(|i| s[i])
            .collect();
        println!(
            "  grip util > 1.05 at {} samples{}",
            over.len(),
            if over.is_empty() {
                String::new()
            } else {
                format!(
                    " (s ≈ {})",
                    over.iter()
                        .take(6)
                        .map(|x| format!("{x:.0}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        );
    }

    let peak_tp = tp
        .iter()
        .cloned()
        .filter(|x| x.is_finite())
        .fold(0.0, f64::max);
    let peak_bp = bp
        .iter()
        .cloned()
        .filter(|x| x.is_finite())
        .fold(0.0, f64::max);
    println!(
        "  tractive power peak: {:.0} kW   braking power peak: {:.0} kW ({:.1}× tractive)",
        peak_tp / 1000.0,
        peak_bp / 1000.0,
        if peak_tp > 0.0 {
            peak_bp / peak_tp
        } else {
            0.0
        }
    );

    // a_long closure over the lap (should net ~0).
    let mut closure = 0.0;
    for i in 0..s.len() {
        let ds = if i + 1 < s.len() {
            s[i + 1] - s[i]
        } else {
            0.0
        };
        if long_g[i].is_finite() {
            closure += long_g[i] * 9.81 * ds;
        }
    }
    println!(
        "  a_long closure ∫a·ds over lap: {:+.1} m²/s² (≈0 ideal)",
        closure
    );

    Ok(())
}

fn cmd_estimate(
    telemetry: PathBuf,
    track: PathBuf,
    car: PathBuf,
    out: PathBuf,
    pos_sigma: f64,
    use_course: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use apex_correlate::{
        attach_estimated_channels, diagnostics_json, import_telemetry, smooth_states,
        write_telemetry_csv, EstimatorConfig, Mapping,
    };
    use apex_telemetry::ChannelId;

    const EST_VERSION: &str = "1.0.0";

    let measured = import_telemetry(&telemetry, &Mapping::identity())?;
    let trk = load_track_from_path(&track)?;
    // Fitted TOML is an overlay on the calibrated preset.
    let params = load_car_params(Some(car.clone()), true)?;
    // The single-track model needs a tire; the fit does not identify one, so use
    // the representative F1 Pacejka. Slip angles therefore depend on this assumed
    // tire (documented in the output caveat).
    let tire = apex_physics::PacejkaTire::f1_default();

    // The estimator runs in TIME: pull t / x / y / speed from the aligned lap.
    let pull =
        |id: ChannelId, what: &'static str| -> Result<Vec<f64>, Box<dyn std::error::Error>> {
            measured
                .channel(id)
                .map(|c| c.to_vec())
                .ok_or_else(|| format!("telemetry is missing the `{what}` channel").into())
        };
    let t = pull(ChannelId::Time, "t")?;
    let x = pull(ChannelId::X, "x")?;
    let y = pull(ChannelId::Y, "y")?;
    let speed = pull(ChannelId::Speed, "speed")?;

    let cfg = EstimatorConfig {
        pos_sigma,
        use_course,
        ..EstimatorConfig::default()
    };

    let res = smooth_states(&t, &x, &y, &speed, &params, &tire, &cfg)?;
    let mut out_tel = attach_estimated_channels(&measured, &res);

    // Provenance: descriptive (derived-from-measured, no RunMetadata sim block) +
    // car hash + estimator-config hash + the effective-parameter caveat.
    let car_hash = apex_physics::car_params_hash(&params);
    let cfg_hash = apex_telemetry::settings_hash_for_mode(&cfg.settings_label());
    out_tel.metadata.push((
        "estimation".into(),
        format!("apex-14 estimate v{EST_VERSION} (RTS single-track Kalman smoother)"),
    ));
    out_tel
        .metadata
        .push(("estimate_car_file".into(), car.display().to_string()));
    out_tel
        .metadata
        .push(("estimate_car_hash".into(), car_hash.to_hex()));
    out_tel
        .metadata
        .push(("estimate_config_hash".into(), cfg_hash.to_hex()));
    out_tel.metadata.push((
        "estimate_caveat".into(),
        "slip angles / body slip / yaw rate are single-track ESTIMATES that depend \
         on the fitted EFFECTIVE tire+aero parameters and the assumed Pacejka tire \
         model — model-consistent estimates, NOT measurements"
            .into(),
    ));

    write_telemetry_csv(&out, &out_tel)?;

    // Sidecar diagnostics JSON (per-state std devs + NIS + robustness counts).
    let diag_path = out.with_extension("diag.json");
    std::fs::write(&diag_path, diagnostics_json(&res))?;

    // --- console report ---
    let d = &res.diagnostics;
    let deg = |r: f64| r.to_degrees();
    let s = measured.channel(ChannelId::S).map(|c| c.to_vec());
    let s_at = |i: usize| s.as_ref().map(|v| v[i]).unwrap_or(i as f64);

    // Peak slip angles + locations.
    let peak = |v: &[f64]| -> (f64, usize) {
        let mut best = 0.0_f64;
        let mut idx = 0;
        for (i, &val) in v.iter().enumerate() {
            if val.is_finite() && val.abs() > best {
                best = val.abs();
                idx = i;
            }
        }
        (best, idx)
    };
    let (pf, pf_i) = peak(&res.slip_front);
    let (pr, pr_i) = peak(&res.slip_rear);
    let beta_min = res
        .beta
        .iter()
        .cloned()
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min);
    let beta_max = res
        .beta
        .iter()
        .cloned()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let yaw_peak = res
        .state
        .iter()
        .map(|st| st[5].abs())
        .fold(0.0_f64, f64::max);

    println!("Estimated {} samples → {}", res.state.len(), out.display());
    println!(
        "  car: {} (hash {})   course pseudo-measurement: {}",
        car.display(),
        car_hash.short(),
        if use_course { "on" } else { "OFF" }
    );
    println!("  ⚠ single-track estimates, NOT measurements (see header caveat).");
    println!(
        "  peak slip: front {:.2}° @ s={:.0} m   rear {:.2}° @ s={:.0} m",
        deg(pf),
        s_at(pf_i),
        deg(pr),
        s_at(pr_i)
    );
    println!(
        "  body slip beta: [{:+.2}°, {:+.2}°]   yaw-rate peak: {:.3} rad/s",
        deg(beta_min),
        deg(beta_max),
        yaw_peak
    );
    println!(
        "  NIS: mean {:.2} (dof {})   p50 {:.2}  p95 {:.2}  within-95 {:.0}%",
        d.nis_mean,
        d.nis_dof,
        d.nis_p50,
        d.nis_p95,
        100.0 * d.nis_within_95
    );
    println!(
        "  robustness: {} updates, {} gaps, {} soft-gated (down-weighted) updates",
        d.n_updates, d.n_gaps, d.n_rejected
    );

    // Smoother yaw-rate vs kinematic v·kappa (QSS inference sees only the latter).
    // The disagreement IS the transient/dynamic content the QSS misses.
    if let Some(sv) = &s {
        let mut sum_all = 0.0;
        let mut n_all = 0usize;
        let mut sum_corner = 0.0;
        let mut n_corner = 0usize;
        for i in 0..sv.len() {
            let (seg, _) = trk.locate(sv[i]);
            let kappa = trk.segments[seg.min(trk.segments.len() - 1)].curvature;
            let kin = speed[i] * kappa;
            let dyn_yaw = res.state[i][5];
            let diff = (dyn_yaw - kin).abs();
            if diff.is_finite() {
                sum_all += diff;
                n_all += 1;
                if kappa.abs() > 1.0 / 200.0 {
                    sum_corner += diff;
                    n_corner += 1;
                }
            }
        }
        let mean_all = if n_all > 0 {
            sum_all / n_all as f64
        } else {
            0.0
        };
        let mean_corner = if n_corner > 0 {
            sum_corner / n_corner as f64
        } else {
            0.0
        };
        println!(
            "  yaw-rate vs kinematic v·κ: mean |Δ| {:.4} rad/s overall, {:.4} rad/s in corners \
             ({} corner samples) — this gap is the dynamic content QSS inference cannot see",
            mean_all, mean_corner, n_corner
        );
    }
    println!("  diagnostics sidecar: {}", diag_path.display());

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
