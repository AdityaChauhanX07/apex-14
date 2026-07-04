//! Golden-lap regression harness (Phase 0.1, slices 1-3).
//!
//! Guards simulation/optimization output for pinned baseline scenarios on
//! the default oval track. See PHYSICS_CHANGE.md at the repo root for the
//! contract: a failing golden test is a STOP, not something to silence.
//!
//! Baselines:
//! - `golden_oval_qss`: `qss_lap_sim`, calibrated car. Bitwise-deterministic
//!   (no RNG, no rayon, no hashmap-order dependence).
//! - `golden_oval_optimize`: `CollocationOptimizer::optimize_gn` with
//!   Hermite-Simpson collocation, calibrated car, 50 nodes. Also
//!   deterministic — the warmstart is `qss_lap_sim` interpolated onto the
//!   collocation nodes (`initial_guess()`), not an ML/NN warmstart, and
//!   neither `collocation.rs` nor `gauss_newton.rs` touch `rand`.
//!
//! Both compare with tolerance, not bitwise equality, because floating-point
//! results aren't portable across compiler version, OS, or optimization
//! level, and a multi-OS CI matrix is coming in Phase 0.4.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use apex_optimizer::{
    CollocationConfig, CollocationMethod, CollocationOptimizer, GaussNewtonConfig,
};
use apex_physics::{qss_lap_sim, CarParams};
use apex_track::{build_track, oval_track, Track};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Fixture schema
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
struct GoldenFixture {
    metadata: GoldenMetadata,
    payload: GoldenPayload,
}

#[derive(Debug, Serialize, Deserialize)]
struct GoldenMetadata {
    apex_version: String,
    git_sha: String,
    /// Unix epoch seconds. No chrono/time dependency exists anywhere in this
    /// workspace (checked), so we avoid adding one just for a timestamp that
    /// is never asserted on.
    generated_at: i64,
    car_id: String,
    track_id: String,
    mode: String,
    flags: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct GoldenPayload {
    lap_time: f64,
    speed_trace_10m: Vec<SpeedSample>,
    /// Lateral offset trace, only produced by the `optimize` path (QSS has no
    /// notion of lateral offset). `skip_serializing_if` keeps the existing
    /// QSS fixture byte-identical if it's ever regenerated after this field
    /// was added.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    offset_trace_10m: Option<Vec<OffsetSample>>,
    sector_times: Option<Vec<f64>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SpeedSample {
    s: f64,
    v: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct OffsetSample {
    s: f64,
    n: f64,
}

const QSS_FIXTURE_PATH: &str = "tests/fixtures/golden/f1_2024_calibrated__oval_default__qss.json";
const OPTIMIZE_FIXTURE_PATH: &str =
    "tests/fixtures/golden/f1_2024_calibrated__oval_default__optimize_hermite_simpson.json";

// Tolerances (not bitwise) are used deliberately because FP results aren't
// portable across compiler/OS/opt-level, and multi-OS CI is coming in Phase 0.4.
const LAP_TIME_TOL_S: f64 = 0.010;
const SPEED_RMSE_TOL_MS: f64 = 0.1;

/// `optimize --hermite-simpson --calibrated`'s node count is CLI default
/// `50` (`#[arg(short, long, default_value_t = 50)]` on `Commands::Optimize`
/// in `bins/apex-cli/src/main.rs`). Hard-coded here rather than imported so a
/// future change to the CLI default can't silently repoint this golden.
const OPTIMIZE_NODES: usize = 50;

// ---------------------------------------------------------------------------
// Baseline scenario
// ---------------------------------------------------------------------------

fn oval_default_track() -> Track {
    let (points, closed) = oval_track(500.0, 80.0, 12.0, 300);
    build_track("Oval", &points, closed)
}

fn calibrated_car() -> CarParams {
    CarParams::f1_2024_calibrated()
}

// ---------------------------------------------------------------------------
// Shared helpers — single source of truth for both assert and regen paths,
// and shared between the QSS and optimize producers, so the two golden
// scenarios can't drift apart in how they resample or build metadata.
// ---------------------------------------------------------------------------

/// Small, self-contained linear interpolation. Deliberately NOT
/// `apex_optimizer::interp` — this harness must stay independent of the code
/// it's guarding, so a bug or change in the optimizer's helper can never mask
/// (or fake) a regression here. Assumes `xs` is sorted ascending.
fn lerp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let last = xs.len() - 1;
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[last] {
        return ys[last];
    }
    let mut i = 0;
    while i + 1 < xs.len() && xs[i + 1] < x {
        i += 1;
    }
    let (x0, x1, y0, y1) = (xs[i], xs[i + 1], ys[i], ys[i + 1]);
    if (x1 - x0).abs() < 1e-12 {
        return y0;
    }
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// Resamples `(distances, values)` onto a uniform 10 m arc-length grid:
/// `s = 0, 10, 20, ..., floor(L/10)*10`, then one final exact-endpoint
/// sample at `s = L` (`L = *distances.last()`), unless the grid already
/// lands on it. Resampling by arc length (rather than by node/segment index)
/// makes the golden robust to the optimizer's internal mesh or the QSS
/// segment count changing independently of the physical track.
fn resample_to_10m_grid(distances: &[f64], values: &[f64]) -> Vec<(f64, f64)> {
    let l = *distances
        .last()
        .expect("distances must have at least one entry");

    let n_steps = (l / 10.0).floor() as i64;
    let mut samples = Vec::with_capacity(n_steps as usize + 2);
    for i in 0..=n_steps {
        let s = i as f64 * 10.0;
        samples.push((s, lerp(distances, values, s)));
    }
    let last_grid_s = n_steps as f64 * 10.0;
    if (l - last_grid_s).abs() > 1e-9 {
        samples.push((l, lerp(distances, values, l)));
    }
    samples
}

fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn build_metadata(
    car_id: &str,
    track_id: &str,
    mode: &str,
    flags: serde_json::Value,
) -> GoldenMetadata {
    let generated_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_secs() as i64;

    GoldenMetadata {
        apex_version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: git_sha(),
        generated_at,
        car_id: car_id.to_string(),
        track_id: track_id.to_string(),
        mode: mode.to_string(),
        flags,
    }
}

fn fixture_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn load_fixture(path: &PathBuf) -> GoldenFixture {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|_| {
        panic!(
            "golden fixture missing at {}\n\
             Run: REGEN_GOLDEN=1 cargo test -p apex-14 --test golden_lap -- --ignored",
            path.display()
        )
    });
    serde_json::from_str(&raw).expect("golden fixture is not valid JSON")
}

fn write_fixture(path: &PathBuf, fixture: &GoldenFixture) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).expect("failed to create fixture directory");
    }
    let json = serde_json::to_string_pretty(fixture).expect("failed to serialize fixture");
    std::fs::write(path, json).expect("failed to write fixture");
    eprintln!("wrote {}", path.display());
}

/// Asserts `now` matches `fixed` within the golden-lap tolerances: matching
/// resampled grid, then lap-time and speed-RMSE tolerance. Shared by every
/// golden comparison test so both scenarios enforce identical rules.
fn assert_payload_matches(now: &GoldenPayload, fixed: &GoldenPayload) {
    assert_eq!(
        now.speed_trace_10m.len(),
        fixed.speed_trace_10m.len(),
        "sample count differs from fixture ({} vs {}) — track length or sampling grid changed",
        now.speed_trace_10m.len(),
        fixed.speed_trace_10m.len()
    );
    for (i, (a, b)) in now
        .speed_trace_10m
        .iter()
        .zip(fixed.speed_trace_10m.iter())
        .enumerate()
    {
        assert!(
            (a.s - b.s).abs() <= 1e-6,
            "grid point {i} station mismatch: now s={} fixture s={} — track length or sampling grid changed",
            a.s,
            b.s
        );
    }

    assert!(
        (now.lap_time - fixed.lap_time).abs() <= LAP_TIME_TOL_S,
        "lap_time regression: now={:.6}s fixture={:.6}s (tol={:.3}s)",
        now.lap_time,
        fixed.lap_time,
        LAP_TIME_TOL_S
    );

    let sum_sq: f64 = now
        .speed_trace_10m
        .iter()
        .zip(fixed.speed_trace_10m.iter())
        .map(|(a, b)| (a.v - b.v).powi(2))
        .sum();
    let rmse = (sum_sq / now.speed_trace_10m.len() as f64).sqrt();
    assert!(
        rmse < SPEED_RMSE_TOL_MS,
        "speed trace RMSE regression: {rmse:.6} m/s (tol={SPEED_RMSE_TOL_MS:.3} m/s)"
    );
}

// ---------------------------------------------------------------------------
// QSS producer
// ---------------------------------------------------------------------------

fn compute_payload(track: &Track, params: &CarParams) -> GoldenPayload {
    let result = qss_lap_sim(track, params);
    let speed_trace_10m = resample_to_10m_grid(&result.distances, &result.speeds)
        .into_iter()
        .map(|(s, v)| SpeedSample { s, v })
        .collect();

    GoldenPayload {
        lap_time: result.lap_time,
        speed_trace_10m,
        offset_trace_10m: None,
        sector_times: None,
    }
}

// ---------------------------------------------------------------------------
// Optimize (Hermite-Simpson) producer
// ---------------------------------------------------------------------------

/// Runs the collocation optimizer exactly as `cmd_optimize` does for
/// `--hermite-simpson --calibrated` (see `bins/apex-cli/src/main.rs`'s
/// `cmd_optimize`), resamples speeds and lateral offsets onto the 10 m grid,
/// and reports whether the solve converged.
fn compute_optimize_payload(
    track: &Track,
    params: &CarParams,
    nodes: usize,
) -> (GoldenPayload, bool) {
    let config = CollocationConfig {
        n_nodes: nodes,
        method: CollocationMethod::HermiteSimpson,
        ..Default::default()
    };
    let optimizer = CollocationOptimizer::new(config, track, params);
    let result = optimizer.optimize_gn(&GaussNewtonConfig::default());

    let speed_trace_10m = resample_to_10m_grid(&result.stations, &result.speeds)
        .into_iter()
        .map(|(s, v)| SpeedSample { s, v })
        .collect();
    let offset_trace_10m = resample_to_10m_grid(&result.stations, &result.offsets)
        .into_iter()
        .map(|(s, n)| OffsetSample { s, n })
        .collect();

    let payload = GoldenPayload {
        lap_time: result.lap_time,
        speed_trace_10m,
        offset_trace_10m: Some(offset_trace_10m),
        sector_times: None,
    };
    (payload, result.converged)
}

// ---------------------------------------------------------------------------
// Comparison tests
// ---------------------------------------------------------------------------

#[test]
fn golden_oval_qss() {
    let fixture = load_fixture(&fixture_path(QSS_FIXTURE_PATH));

    let track = oval_default_track();
    let params = calibrated_car();
    let now = compute_payload(&track, &params);

    assert_payload_matches(&now, &fixture.payload);
}

#[test]
fn golden_oval_optimize() {
    let fixture = load_fixture(&fixture_path(OPTIMIZE_FIXTURE_PATH));

    let track = oval_default_track();
    let params = calibrated_car();
    let (now, converged) = compute_optimize_payload(&track, &params, OPTIMIZE_NODES);

    assert!(
        converged,
        "optimizer did not converge — golden comparison is meaningless"
    );

    assert_payload_matches(&now, &fixture.payload);

    // offset_trace_10m is captured for forward-compat but not asserted on
    // yet — the default oval is a near-symmetric loop with little lateral
    // freedom, so a meaningful offset tolerance needs a track with a real
    // varied racing line (e.g. Silverstone) to calibrate against. Deferred
    // to the tolerance-hardening slice.
}

// ---------------------------------------------------------------------------
// Regeneration
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn regen_golden_oval() {
    if std::env::var("REGEN_GOLDEN").is_err() {
        eprintln!("skipping: set REGEN_GOLDEN=1 to regenerate the golden fixture");
        return;
    }

    let track = oval_default_track();
    let params = calibrated_car();
    let payload = compute_payload(&track, &params);

    let metadata = build_metadata(
        "f1_2024_calibrated",
        "oval_default",
        "qss",
        // `qss --calibrated` always calls the constant-tire_mu grip-circle
        // model (qss_lap_sim), never the Pacejka load-sensitive
        // qss_lap_sim_tire — there is no fidelity flag on the CLI's `qss`
        // subcommand today, so this records what it actually resolves to.
        serde_json::json!({
            "calibrated": true,
            "fidelity": "grip-circle-constant-mu"
        }),
    );

    write_fixture(
        &fixture_path(QSS_FIXTURE_PATH),
        &GoldenFixture { metadata, payload },
    );
}

#[test]
#[ignore]
fn regen_golden_optimize() {
    if std::env::var("REGEN_GOLDEN").is_err() {
        eprintln!("skipping: set REGEN_GOLDEN=1 to regenerate the golden fixture");
        return;
    }

    let track = oval_default_track();
    let params = calibrated_car();
    let (payload, converged) = compute_optimize_payload(&track, &params, OPTIMIZE_NODES);

    assert!(
        converged,
        "refusing to write golden fixture: optimizer did not converge"
    );

    let metadata = build_metadata(
        "f1_2024_calibrated",
        "oval_default",
        "optimize_hermite_simpson",
        serde_json::json!({
            "calibrated": true,
            "hermite_simpson": true,
            "nodes": OPTIMIZE_NODES
        }),
    );

    write_fixture(
        &fixture_path(OPTIMIZE_FIXTURE_PATH),
        &GoldenFixture { metadata, payload },
    );
}
