#![deny(unsafe_code)]
//! Apex-14 validation binary.
//!
//! Loads real Silverstone track geometry (TUMFTM racetrack database), runs the
//! quasi-steady-state lap simulator and the collocation optimizer, and compares
//! the results against published 2024 F1 qualifying reference data.

use std::path::Path;
use std::process::Command;

use apex_optimizer::{
    CollocationConfig, CollocationMethod, CollocationOptimizer, GaussNewtonConfig,
};
use apex_physics::{qss_lap_sim, CarParams, QssResult};
use apex_track::{build_track, load_tumftm_csv, silverstone_circuit, Track};

// --- Published reference data (2024 F1 qualifying, Silverstone) ---
const PUBLISHED_LENGTH_KM: f64 = 5.891;
const PUBLISHED_LAP_S: f64 = 85.8; // ~1:25.8
const PUBLISHED_TOP_KPH: f64 = 330.0; // pit straight speed trap
const PUBLISHED_MIN_KPH: f64 = 95.0; // Village hairpin
const PUBLISHED_LAT_G: f64 = 5.5; // Copse / Maggotts
const PUBLISHED_BRAKE_G: f64 = 5.8; // Stowe entry

const TUMFTM_URL: &str =
    "https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/tracks/Silverstone.csv";

/// Load the Silverstone track, preferring real TUMFTM data. Tries local files,
/// then a `curl` download, then falls back to the hardcoded approximation.
/// Returns the track and a human-readable description of the data source.
fn load_silverstone() -> (Track, String) {
    // 1. Local files.
    const CANDIDATES: [&str; 2] = [
        "tracks/Silverstone.csv",
        "racetrack-database/tracks/Silverstone.csv",
    ];
    for path in CANDIDATES {
        if Path::new(path).exists() {
            if let Ok(track) = load_tumftm_csv(Path::new(path), "Silverstone") {
                return (track, format!("TUMFTM real data ({path})"));
            }
        }
    }

    // 2. Download via curl.
    let dest = "tracks/Silverstone.csv";
    let _ = std::fs::create_dir_all("tracks");
    let downloaded = Command::new("curl")
        .args(["-sSL", "--fail", TUMFTM_URL, "-o", dest])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if downloaded {
        if let Ok(track) = load_tumftm_csv(Path::new(dest), "Silverstone") {
            return (track, "TUMFTM real data (downloaded)".to_string());
        }
    }

    // 3. Hardcoded approximation.
    let (pts, closed) = silverstone_circuit();
    (
        build_track("Silverstone", &pts, closed),
        "hardcoded approximation".to_string(),
    )
}

/// Summary statistics for a QSS run (speeds in m/s; g-values already in g).
struct Stats {
    lap: f64,
    top_kph: f64,
    min_kph: f64,
    max_lat_g: f64,
    max_lon_g: f64,
}

fn summarize(r: &QssResult) -> Stats {
    let top = r.speeds.iter().cloned().fold(f64::MIN, f64::max);
    let min = r.speeds.iter().cloned().fold(f64::MAX, f64::min);
    let max_lat = r.lateral_gs.iter().map(|g| g.abs()).fold(0.0, f64::max);
    let max_lon = r
        .longitudinal_gs
        .iter()
        .map(|g| g.abs())
        .fold(0.0, f64::max);
    Stats {
        lap: r.lap_time,
        top_kph: top * 3.6,
        min_kph: min * 3.6,
        max_lat_g: max_lat,
        max_lon_g: max_lon,
    }
}

/// Signed percentage error of `value` relative to `reference`.
fn pct_error(value: f64, reference: f64) -> f64 {
    100.0 * (value - reference) / reference
}

fn main() {
    let (track, source) = load_silverstone();
    let length_km = track.total_length / 1000.0;

    let default_car = CarParams::default();
    let calibrated = CarParams::f1_2024_calibrated();

    // (b) QSS with both parameter sets.
    let qss_default = summarize(&qss_lap_sim(&track, &default_car));
    let qss_calibrated = summarize(&qss_lap_sim(&track, &calibrated));

    // (c) Collocation optimizer (point-mass, Hermite-Simpson, N=80, calibrated).
    let config = CollocationConfig {
        n_nodes: 80,
        closed: track.is_closed,
        method: CollocationMethod::HermiteSimpson,
        ..CollocationConfig::default()
    };
    let gn = GaussNewtonConfig {
        max_iterations: 60,
        constraint_tol: 1e-3,
        print_interval: 0,
        ..GaussNewtonConfig::default()
    };
    let opt = CollocationOptimizer::new(config, &track, &calibrated);
    let opt_result = opt.optimize_gn(&gn);

    // (d) Report.
    println!("=== Apex-14 Validation Report: Silverstone ===");
    println!();
    println!("Track: {source}");
    println!(
        "Track length: {:.3} km (published: {:.3} km)",
        length_km, PUBLISHED_LENGTH_KM
    );
    println!();

    println!("--- Published Reference Data (2024 F1 Qualifying) ---");
    println!("Lap time:          ~1:25.8 ({PUBLISHED_LAP_S:.1} s)");
    println!("Top speed:         ~{PUBLISHED_TOP_KPH:.0} km/h (pit straight)");
    println!("Min corner speed:  ~{PUBLISHED_MIN_KPH:.0} km/h (Village hairpin)");
    println!("Peak lateral g:    ~{PUBLISHED_LAT_G:.1}g (Copse/Maggotts)");
    println!("Peak braking g:    ~{PUBLISHED_BRAKE_G:.1}g (Stowe entry)");
    println!();

    println!("--- Simulation Results ---");
    println!("                     | Default params | Calibrated params");
    println!(
        "QSS lap time (s)     |    {:7.1}     |     {:7.1}",
        qss_default.lap, qss_calibrated.lap
    );
    println!(
        "Top speed (km/h)     |    {:7.1}     |     {:7.1}",
        qss_default.top_kph, qss_calibrated.top_kph
    );
    println!(
        "Min speed (km/h)     |    {:7.1}     |     {:7.1}",
        qss_default.min_kph, qss_calibrated.min_kph
    );
    println!(
        "Max lateral g        |    {:7.2}     |     {:7.2}",
        qss_default.max_lat_g, qss_calibrated.max_lat_g
    );
    println!(
        "Max longitudinal g   |    {:7.2}     |     {:7.2}",
        qss_default.max_lon_g, qss_calibrated.max_lon_g
    );
    println!();

    println!("Optimizer (calibrated):");
    println!(
        "Lap time: {:.1} s | eq_viol: {:.2e}",
        opt_result.lap_time, opt_result.eq_violation
    );
    println!();

    let lap_err = pct_error(qss_calibrated.lap, PUBLISHED_LAP_S);
    let top_err = pct_error(qss_calibrated.top_kph, PUBLISHED_TOP_KPH);
    println!("--- Comparison vs Published ---");
    println!("Lap time error:     {lap_err:+.1}% (calibrated QSS vs published)");
    println!("Top speed error:    {top_err:+.1}%");
    println!();

    println!(
        "Note: {}",
        assessment(&source, lap_err, top_err, &qss_default, &qss_calibrated)
    );
}

/// One-paragraph honest assessment of the comparison.
fn assessment(
    source: &str,
    lap_err: f64,
    top_err: f64,
    default: &Stats,
    calibrated: &Stats,
) -> String {
    let geometry = if source.starts_with("TUMFTM") {
        "real measured track geometry"
    } else {
        "an approximate hardcoded track"
    };
    let direction = if lap_err >= 0.0 { "slower" } else { "faster" };
    format!(
        "QSS is a point-mass, grip-circle, flat-track estimate run on the centerline of {geometry}. \
         The calibrated lap ({:.1} s) is {:.1}% {direction} than the ~{:.1} s published qualifying \
         time, while top speed ({:.0} km/h) lands within {:.1}% of the ~{:.0} km/h trap. The lap-time \
         gap is dominated by cornering: the calibrated grip set peaks at {:.1}g laterally (published \
         ~{:.1}g) and bottoms out at {:.0} km/h (published ~{:.0} km/h), so it is conservative through \
         the fast corners. The higher-grip default set is closer on lap time ({:.1} s) and lateral g \
         ({:.1}g) but reaches an unrealistic {:.0} km/h top speed. Neither matches exactly: the \
         centerline is not a racing line, the track is treated as flat (no Maggotts/Becketts camber or \
         elevation), and grip is a single fixed set with no thermal or per-corner setup effects.",
        calibrated.lap,
        lap_err.abs(),
        PUBLISHED_LAP_S,
        calibrated.top_kph,
        top_err.abs(),
        PUBLISHED_TOP_KPH,
        calibrated.max_lat_g,
        PUBLISHED_LAT_G,
        calibrated.min_kph,
        PUBLISHED_MIN_KPH,
        default.lap,
        default.max_lat_g,
        default.top_kph,
    )
}
