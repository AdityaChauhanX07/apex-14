//! Smoke integration test for the correlation pipeline, driven entirely by a
//! **synthetic** committed fixture (never the real, gitignored telemetry — CI
//! must be able to run this).

use std::path::PathBuf;

use apex_correlate::report::{correlate, write_report, CorrelationConfig};
use apex_correlate::{import_telemetry, Mapping};
use apex_physics::{qss_lap_sim, CarParams};
use apex_track::{build_track, circle_track};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn correlate_synthetic_lap_end_to_end() {
    // Measured (synthetic) lap: t, s, speed, throttle, brake, gear with two
    // planted speed dips (corners) at s≈200 and s≈450.
    let measured = import_telemetry(fixture("correlate_synth_lap.csv"), &Mapping::identity())
        .expect("import synthetic fixture");

    // A circle track long enough to contain the measured s range (~628 m).
    let (pts, closed) = circle_track(100.0, 12.0, 240);
    let track = build_track("SynthCircle", &pts, closed);
    let sim = qss_lap_sim(&track, &CarParams::default());

    let trace = apex_correlate::report::SimTrace::from_qss(&sim);
    let result =
        correlate(&measured, &track, &trace, CorrelationConfig::default()).expect("correlate");

    // Metrics are finite and the grid spans most of the lap.
    assert!(result.rmse.rmse.is_finite());
    assert!(result.span > 500.0, "span {}", result.span);
    assert_eq!(result.sectors.delta.len(), 3);

    // The two planted dips are detected as corners (no doubles).
    assert_eq!(result.corners.len(), 2, "corners {:?}", result.corners);
    assert_eq!(result.apex.len(), 2);

    // Header lap-time (13.675 s) matches the t-span → no mismatch flagged.
    assert!(result.header_lap_time.is_some());
    assert!(result.lap_time_mismatch.is_none());

    // Headline carries the synthetic session identifiers.
    let head = result.headline(&track.name);
    assert!(
        head.contains("SynthCircle 2024 TEST (SYN)"),
        "headline: {head}"
    );

    // Report + SVGs are produced.
    let dir = std::env::temp_dir().join("apex_correlate_smoke_out");
    let meta = apex_telemetry::RunMetadata::new(
        apex_physics::car_params_hash(&CarParams::default()),
        apex_track::processed_track_hash(&track),
        apex_telemetry::settings_hash_for_mode("correlate.qss.grip-circle"),
        None,
    );
    let report = write_report(&dir, &result, &measured, &meta, &track).expect("write report");
    let md = std::fs::read_to_string(&report).unwrap();
    assert!(md.contains("equal-arc"), "caveat missing");
    assert!(md.contains("## Braking-point offsets"));
    assert!(md.contains("Corners & apex speeds"));
    // Hybrid provenance on the overlay: sim RunMetadata + measured descriptive.
    let svg = std::fs::read_to_string(dir.join("speed_overlay.svg")).unwrap();
    assert!(svg.contains("<apex:config_hash>"));
    assert!(svg.contains("measured-source"));

    std::fs::remove_dir_all(&dir).ok();
}
