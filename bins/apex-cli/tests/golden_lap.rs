//! Golden-lap regression harness (Phase 0.1, slice 1).
//!
//! Guards `qss_lap_sim` output for a single deterministic baseline: the
//! default oval track with the calibrated F1-2024 car. QSS has no RNG,
//! no rayon, and no hashmap-order dependence (see Phase 0 design notes),
//! so any drift here reflects an intentional or accidental change to the
//! physics/track code, not run-to-run noise.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

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
    sector_times: Option<Vec<f64>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SpeedSample {
    s: f64,
    v: f64,
}

const FIXTURE_PATH: &str = "tests/fixtures/golden/f1_2024_calibrated__oval_default__qss.json";

// Tolerances (not bitwise) are used deliberately because FP results aren't
// portable across compiler/OS/opt-level, and multi-OS CI is coming in Phase 0.4.
const LAP_TIME_TOL_S: f64 = 0.010;
const SPEED_RMSE_TOL_MS: f64 = 0.1;

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
// Single source of truth for both the assert path and the regen path.
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

fn compute_payload(track: &Track, params: &CarParams) -> GoldenPayload {
    let result = qss_lap_sim(track, params);
    let l = *result
        .distances
        .last()
        .expect("qss_lap_sim must return at least one segment");

    let n_steps = (l / 10.0).floor() as i64;
    let mut samples = Vec::with_capacity(n_steps as usize + 2);
    for i in 0..=n_steps {
        let s = i as f64 * 10.0;
        samples.push(SpeedSample {
            s,
            v: lerp(&result.distances, &result.speeds, s),
        });
    }
    // Append the exact end-of-array point unless the grid already lands on it.
    let last_grid_s = n_steps as f64 * 10.0;
    if (l - last_grid_s).abs() > 1e-9 {
        samples.push(SpeedSample {
            s: l,
            v: lerp(&result.distances, &result.speeds, l),
        });
    }

    GoldenPayload {
        lap_time: result.lap_time,
        speed_trace_10m: samples,
        sector_times: None,
    }
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_PATH)
}

// ---------------------------------------------------------------------------
// Comparison test
// ---------------------------------------------------------------------------

#[test]
fn golden_oval_qss() {
    let path = fixture_path();
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "golden fixture missing at {}\n\
             Run: REGEN_GOLDEN=1 cargo test -p apex-14 --test golden_lap -- --ignored",
            path.display()
        )
    });
    let fixture: GoldenFixture =
        serde_json::from_str(&raw).expect("golden fixture is not valid JSON");

    let track = oval_default_track();
    let params = calibrated_car();
    let now = compute_payload(&track, &params);

    let fixed = &fixture.payload;

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

    let git_sha = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let generated_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_secs() as i64;

    let fixture = GoldenFixture {
        metadata: GoldenMetadata {
            apex_version: env!("CARGO_PKG_VERSION").to_string(),
            git_sha,
            generated_at,
            car_id: "f1_2024_calibrated".to_string(),
            track_id: "oval_default".to_string(),
            mode: "qss".to_string(),
            // `qss --calibrated` always calls the constant-tire_mu grip-circle
            // model (qss_lap_sim), never the Pacejka load-sensitive
            // qss_lap_sim_tire — there is no fidelity flag on the CLI's `qss`
            // subcommand today, so this records what it actually resolves to.
            flags: serde_json::json!({
                "calibrated": true,
                "fidelity": "grip-circle-constant-mu"
            }),
        },
        payload,
    };

    let path = fixture_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).expect("failed to create fixture directory");
    }
    let json = serde_json::to_string_pretty(&fixture).expect("failed to serialize fixture");
    std::fs::write(&path, json).expect("failed to write fixture");

    eprintln!("wrote {}", path.display());
}
