//! Network integration test: import the real Silverstone track from the TUMFTM
//! racetrack database and run a QSS lap simulation on it.
//!
//! This test downloads data from GitHub at run time, so it is marked
//! `#[ignore]` and excluded from the default `cargo test` run. Execute it
//! explicitly with:
//!
//! ```text
//! cargo test -p apex-physics --test tumftm_silverstone -- --ignored
//! ```
//!
//! It lives in `apex-physics` (rather than `apex-track`) because it exercises
//! both `apex_track::parse_tumftm_csv` and `apex_physics::qss_lap_sim`, and
//! `apex-physics` already depends on `apex-track`. Putting it in `apex-track`
//! would require a dev-dependency back on `apex-physics`, forming a dependency
//! cycle.

use apex_physics::{qss_lap_sim, CarParams};
use apex_track::parse_tumftm_csv;

const SILVERSTONE_URL: &str =
    "https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/tracks/Silverstone.csv";

/// Download a URL to a string by shelling out to `curl` (available on Windows
/// 10+, macOS, and most Linux distros). Returns `None` if the fetch fails for
/// any reason, so the caller can skip gracefully when offline.
fn http_get(url: &str) -> Option<String> {
    let output = std::process::Command::new("curl")
        .args(["-sSL", "--fail", url])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

#[test]
#[ignore = "requires network access to raw.githubusercontent.com"]
fn silverstone_from_tumftm_runs_qss() {
    let csv = match http_get(SILVERSTONE_URL) {
        Some(c) => c,
        None => {
            eprintln!("skipping: could not download {SILVERSTONE_URL} (offline?)");
            return;
        }
    };

    let track = parse_tumftm_csv(&csv, "Silverstone").expect("parse Silverstone CSV");

    // Closed circuit.
    assert!(track.is_closed, "Silverstone should be a closed circuit");

    // Resolution: the TUMFTM centerline is sampled densely.
    assert!(
        track.segments.len() > 100,
        "expected > 100 points, got {}",
        track.segments.len()
    );

    // Track length ~5.89 km (within 5%).
    let expected_m = 5891.0;
    assert!(
        (track.total_length - expected_m).abs() / expected_m < 0.05,
        "length {:.1} m differs from expected {:.1} m by > 5%",
        track.total_length,
        expected_m
    );

    // All curvatures finite, and corners actually exist.
    let mut max_kappa = 0.0_f64;
    for seg in &track.segments {
        assert!(
            seg.curvature.is_finite(),
            "non-finite curvature at s={}",
            seg.s
        );
        max_kappa = max_kappa.max(seg.curvature.abs());
    }
    assert!(
        max_kappa > 0.01,
        "max curvature {max_kappa} too low — corners missing?"
    );

    // A real lap simulation produces a plausible lap time.
    let car = CarParams::default();
    let result = qss_lap_sim(&track, &car);
    assert!(
        result.lap_time > 50.0 && result.lap_time < 120.0,
        "lap time {:.2} s outside plausible [50, 120] range",
        result.lap_time
    );
}
