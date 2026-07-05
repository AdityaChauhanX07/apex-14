//! Round-trip: import → resample → write → import must reproduce the data
//! (within FP tolerance, NaN gaps preserved).

use std::path::PathBuf;

use apex_correlate::{import_telemetry, write_telemetry_csv, Mapping, Telemetry};
use apex_telemetry::ChannelId;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// NaN-aware approximate equality of two telemetry objects.
fn assert_close(a: &Telemetry, b: &Telemetry, tol: f64) {
    assert_eq!(a.grid, b.grid, "grid differs");
    let a_ids: Vec<ChannelId> = a.channels.keys().copied().collect();
    let b_ids: Vec<ChannelId> = b.channels.keys().copied().collect();
    assert_eq!(a_ids, b_ids, "channel sets differ");
    for id in a_ids {
        let av = a.channel(id).unwrap();
        let bv = b.channel(id).unwrap();
        assert_eq!(av.len(), bv.len(), "length differs for {}", id.name());
        for (x, y) in av.iter().zip(bv) {
            if x.is_nan() || y.is_nan() {
                assert!(
                    x.is_nan() && y.is_nan(),
                    "NaN mismatch for {}: {x} vs {y}",
                    id.name()
                );
            } else {
                assert!(
                    (x - y).abs() <= tol,
                    "value mismatch for {}: {x} vs {y}",
                    id.name()
                );
            }
        }
    }
}

#[test]
fn import_resample_write_import_is_stable() {
    let original = import_telemetry(fixture("synthetic_lap.csv"), &Mapping::identity()).unwrap();
    // Resample onto the same 5 m grid (large max_gap: no synthetic gaps to split).
    let resampled = original.resample_to_s(5.0, 1e9).unwrap();

    let out = std::env::temp_dir().join("apex_correlate_roundtrip.csv");
    write_telemetry_csv(&out, &resampled).unwrap();

    // Re-import the written file with the identity mapping.
    let reimported = import_telemetry(&out, &Mapping::identity()).unwrap();
    assert_close(&resampled, &reimported, 1e-9);

    std::fs::remove_file(&out).ok();
}

#[test]
fn written_file_is_standard_format() {
    let t = import_telemetry(fixture("synthetic_lap.csv"), &Mapping::identity()).unwrap();
    let out = std::env::temp_dir().join("apex_correlate_format.csv");
    write_telemetry_csv(&out, &t).unwrap();
    let text = std::fs::read_to_string(&out).unwrap();

    assert!(text.contains("# grid: s"));
    assert!(text.contains("# columns: s[m], speed[m/s]"));
    // No sim RunMetadata provenance in a measured file.
    assert!(!text.contains("config_hash"));
    // Header row (first non-comment line) is registry names, axis first.
    let header = text.lines().find(|l| !l.starts_with('#')).unwrap();
    assert!(header.starts_with("s,speed"), "header: {header}");
    // NaN gap round-trips as the literal NaN.
    assert!(text.contains("NaN"));

    std::fs::remove_file(&out).ok();
}
