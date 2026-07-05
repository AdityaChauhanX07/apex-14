//! Integration tests for the importer against the standard-format and raw
//! fixtures.

use std::path::PathBuf;

use apex_correlate::{import_telemetry, GridKind, Mapping, UnknownColumns};
use apex_telemetry::ChannelId;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn imports_standard_format_with_identity_mapping() {
    let t = import_telemetry(fixture("synthetic_lap.csv"), &Mapping::identity()).unwrap();
    assert_eq!(t.grid, GridKind::S);
    assert_eq!(t.len(), 5);
    // Registry-named columns mapped to themselves, in registry units.
    assert_eq!(
        t.channel(ChannelId::S).unwrap(),
        &[0.0, 5.0, 10.0, 15.0, 20.0]
    );
    assert_eq!(t.channel(ChannelId::Speed).unwrap()[0], 60.0);
    assert_eq!(t.channel(ChannelId::Rpm).unwrap()[1], 11200.0);

    // The empty row (s=15) is a measured gap → NaN, kept, surfaced via the mask.
    assert_eq!(t.nan_count(ChannelId::Speed), Some(1));
    let mask = t.validity_mask(ChannelId::Throttle).unwrap();
    assert_eq!(mask, vec![true, true, true, false, true]);
    // The axis (s) itself has no gap.
    assert_eq!(t.nan_count(ChannelId::S), Some(0));

    // Reserved header keys are not surfaced as metadata; descriptive ones are.
    assert!(t.metadata.iter().any(|(k, _)| k == "source"));
    assert!(!t
        .metadata
        .iter()
        .any(|(k, _)| k == "grid" || k == "columns"));
}

#[test]
fn maps_raw_source_with_unit_conversion() {
    let mapping = Mapping::from_toml_path(fixture("raw_source_mapping.toml")).unwrap();
    let t = import_telemetry(fixture("raw_source.csv"), &mapping).unwrap();
    assert_eq!(t.grid, GridKind::S);
    assert_eq!(t.len(), 4);

    // 216 km/h → 60 m/s.
    assert!((t.channel(ChannelId::Speed).unwrap()[0] - 60.0).abs() < 1e-9);
    // 100 percent → 1.0 fraction.
    assert!((t.channel(ChannelId::Throttle).unwrap()[0] - 1.0).abs() < 1e-9);
    // Brake 0/1 identity.
    assert_eq!(t.channel(ChannelId::Brake).unwrap()[2], 1.0);
    // -5 deg → radians.
    assert!((t.channel(ChannelId::SteeringAngle).unwrap()[0] + 5.0_f64.to_radians()).abs() < 1e-12);

    // DriverName was unmapped and ignored: no stray channel, no parse error on
    // the non-numeric "VER".
    assert_eq!(t.channels.len(), 5); // s, speed, throttle, brake, steering_angle
}

#[test]
fn unknown_column_error_policy_rejects() {
    // Same raw file, but error on unmapped columns → DriverName trips it.
    let mut mapping = Mapping::from_toml_path(fixture("raw_source_mapping.toml")).unwrap();
    mapping.unknown_columns = UnknownColumns::Error;
    let err = import_telemetry(fixture("raw_source.csv"), &mapping).unwrap_err();
    assert!(err.to_string().contains("DriverName"), "got: {err}");
}

#[test]
fn missing_required_channel_is_listed() {
    let mut mapping = Mapping::identity();
    mapping.required = vec!["lateral_g".to_string()]; // not in the fixture
    let err = import_telemetry(fixture("synthetic_lap.csv"), &mapping).unwrap_err();
    assert!(err.to_string().contains("lateral_g"), "got: {err}");
}
