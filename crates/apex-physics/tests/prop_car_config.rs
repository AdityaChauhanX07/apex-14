//! Property-based tests for the CarParams TOML overlay (apex-physics).
//!
//! The overlay is: `parse_car_toml(toml, base)` clones `base` and replaces only
//! the fields present in `toml`. Properties:
//!  B1 Identity     — an empty overlay yields exactly the base preset.
//!  B2 Idempotence  — applying the same overlay twice equals applying it once.
//!  B3 Precedence   — chosen fields take overlay values; the rest keep base.
//!  B4 Total resolve — any syntactically valid partial config resolves to Ok
//!                     (no panic). NOTE: CarParams has no validation method and
//!                     the task forbids adding one, so "valid resolved config"
//!                     is exactly "resolved without panic / error".
//!
//! Whole-struct equality uses the existing `car_params_hash` content hash
//! (CarParams derives neither PartialEq nor Eq); per-field checks use getters.
//! Determinism: proptest default RNG + on-by-default `proptest-regressions/`.

use apex_physics::{car_params_hash, parse_car_toml, CarParams};
use proptest::prelude::*;

/// Base preset selector — both presets are exercised.
fn base_of(calibrated: bool) -> CarParams {
    if calibrated {
        CarParams::f1_2024_calibrated()
    } else {
        CarParams::default()
    }
}

/// Every scalar field the overlay maps, as `(section, key, getter)`. Kept in
/// lockstep with `apply_config` in car_config.rs.
type Getter = fn(&CarParams) -> f64;
const FIELDS: &[(&str, &str, Getter)] = &[
    ("car", "mass", |p| p.mass),
    ("car", "max_drive_force", |p| p.max_drive_force),
    ("car", "max_brake_force", |p| p.max_brake_force),
    ("aero", "frontal_area", |p| p.frontal_area),
    ("aero", "drag_coeff", |p| p.drag_coeff),
    ("aero", "lift_coeff", |p| p.lift_coeff),
    ("aero", "aero_balance_front", |p| p.aero_balance_front),
    ("tires", "mu", |p| p.tire_mu),
    ("tires", "rolling_resistance", |p| p.rolling_resistance),
    ("geometry", "wheelbase", |p| p.wheelbase),
    ("geometry", "cog_to_front", |p| p.cog_to_front),
    ("geometry", "cog_to_rear", |p| p.cog_to_rear),
    ("geometry", "cog_height", |p| p.cog_height),
    ("geometry", "track_width_front", |p| p.track_width_front),
    ("geometry", "track_width_rear", |p| p.track_width_rear),
    ("geometry", "wheel_radius", |p| p.wheel_radius),
    ("powertrain", "drive_distribution", |p| p.drive_distribution),
    ("powertrain", "brake_bias_front", |p| p.brake_bias_front),
];

/// Distinct TOML section names in a fixed order (matches `FIELDS`).
const SECTIONS: &[&str] = &["car", "aero", "tires", "geometry", "powertrain"];

/// Build a partial TOML string from `(field_index, value)` selections, grouped
/// under the correct `[section]` headers.
fn build_toml(selected: &[(usize, f64)]) -> String {
    let mut out = String::new();
    for &section in SECTIONS {
        let in_section: Vec<&(usize, f64)> = selected
            .iter()
            .filter(|(i, _)| FIELDS[*i].0 == section)
            .collect();
        if in_section.is_empty() {
            continue;
        }
        out.push_str(&format!("[{section}]\n"));
        for (i, v) in in_section {
            out.push_str(&format!("{} = {:?}\n", FIELDS[*i].1, v));
        }
    }
    out
}

/// The f64 a `{:?}`-formatted value round-trips to (what TOML actually stores).
fn stored(v: f64) -> f64 {
    format!("{v:?}").parse().expect("round-trip float")
}

/// Strategy: an include-mask + a value per field, plus a base selector.
fn overlay_inputs() -> impl Strategy<Value = (bool, Vec<bool>, Vec<f64>)> {
    (
        any::<bool>(),
        proptest::collection::vec(any::<bool>(), FIELDS.len()),
        proptest::collection::vec(-1.0e4f64..1.0e4, FIELDS.len()),
    )
}

fn selected_from(mask: &[bool], values: &[f64]) -> Vec<(usize, f64)> {
    (0..FIELDS.len())
        .filter(|&i| mask[i])
        .map(|i| (i, values[i]))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// B1 — an empty (or whitespace/comment-only) overlay reproduces the base
    /// preset exactly, field for field.
    #[test]
    fn empty_overlay_is_identity(
        calibrated in any::<bool>(),
        toml in prop_oneof![
            Just(String::new()),
            Just("   \n\t\n".to_string()),
            Just("# just a comment\n".to_string()),
            Just("\n\n\n".to_string()),
        ],
    ) {
        let base = base_of(calibrated);
        let resolved = parse_car_toml(&toml, &base).expect("empty overlay resolves");
        prop_assert_eq!(car_params_hash(&resolved), car_params_hash(&base));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// B2 — applying the same overlay twice equals applying it once.
    #[test]
    fn overlay_is_idempotent((calibrated, mask, values) in overlay_inputs()) {
        let base = base_of(calibrated);
        let toml = build_toml(&selected_from(&mask, &values));
        let once = parse_car_toml(&toml, &base).expect("first apply");
        let twice = parse_car_toml(&toml, &once).expect("second apply");
        prop_assert_eq!(car_params_hash(&once), car_params_hash(&twice));
    }

    /// B3 — chosen fields resolve to overlay values; unchosen keep base values.
    #[test]
    fn overlay_precedence((calibrated, mask, values) in overlay_inputs()) {
        let base = base_of(calibrated);
        let selected = selected_from(&mask, &values);
        let toml = build_toml(&selected);
        let resolved = parse_car_toml(&toml, &base).expect("overlay resolves");

        for (i, (_, _, getter)) in FIELDS.iter().enumerate() {
            let got = getter(&resolved);
            if mask[i] {
                let expected = stored(values[i]);
                prop_assert_eq!(
                    got, expected,
                    "field {} should take overlay value", FIELDS[i].1
                );
            } else {
                prop_assert_eq!(
                    got, getter(&base),
                    "field {} should keep base value", FIELDS[i].1
                );
            }
        }
    }

    /// B4 — any syntactically valid partial config resolves without panic.
    /// (No CarParams validation exists; success == resolved to Ok.)
    #[test]
    fn any_valid_partial_resolves(
        calibrated in any::<bool>(),
        mask in proptest::collection::vec(any::<bool>(), FIELDS.len()),
        // Wider range incl. large/negative finite values: the overlay applies
        // no validation, so these must still resolve.
        values in proptest::collection::vec(-1.0e9f64..1.0e9, FIELDS.len()),
    ) {
        let base = base_of(calibrated);
        let toml = build_toml(&selected_from(&mask, &values));
        prop_assert!(parse_car_toml(&toml, &base).is_ok());
    }
}
