//! Content-hash correctness + stability tests for `CarParams`.

use apex_physics::car_params::car_params_hash;
use apex_physics::{parse_car_toml, CarParams};

fn hex(c: &CarParams) -> String {
    car_params_hash(c).to_hex()
}

#[test]
fn determinism_and_clone() {
    let c = CarParams::default();
    assert_eq!(hex(&c), hex(&c));
    assert_eq!(hex(&c), hex(&c.clone()));
}

/// Exhaustive per-field sensitivity: perturbing ANY hashed field changes the
/// hash. The list is asserted to cover all 25 fields; combined with the
/// `let CarParams { .. } = self` destructure in `hash_into` (which fails to
/// compile if a field is added and left unhandled), a newly-added field cannot
/// be silently left out of the hash.
#[test]
fn exhaustive_field_sensitivity() {
    let base = CarParams::default();
    let base_h = hex(&base);

    type Mutator = (&'static str, fn(&mut CarParams));
    let mutators: Vec<Mutator> = vec![
        ("mass", |c| c.mass += 1.0),
        ("frontal_area", |c| c.frontal_area += 1.0),
        ("drag_coeff", |c| c.drag_coeff += 1.0),
        ("lift_coeff", |c| c.lift_coeff += 1.0),
        ("air_density", |c| c.air_density += 1.0),
        ("tire_mu", |c| c.tire_mu += 1.0),
        ("rolling_resistance", |c| c.rolling_resistance += 1.0),
        ("max_drive_force", |c| c.max_drive_force += 1.0),
        ("max_brake_force", |c| c.max_brake_force += 1.0),
        ("wheelbase", |c| c.wheelbase += 1.0),
        ("cog_to_front", |c| c.cog_to_front += 1.0),
        ("cog_to_rear", |c| c.cog_to_rear += 1.0),
        ("cog_height", |c| c.cog_height += 1.0),
        ("yaw_inertia", |c| c.yaw_inertia += 1.0),
        ("aero_balance_front", |c| c.aero_balance_front += 1.0),
        ("wheel_radius", |c| c.wheel_radius += 1.0),
        ("wheel_inertia", |c| c.wheel_inertia += 1.0),
        ("track_width_front", |c| c.track_width_front += 1.0),
        ("track_width_rear", |c| c.track_width_rear += 1.0),
        ("brake_bias_front", |c| c.brake_bias_front += 1.0),
        ("drive_distribution", |c| c.drive_distribution += 1.0),
        ("unsprung_mass", |c| c.unsprung_mass += 1.0),
        ("tire_radial_stiffness", |c| c.tire_radial_stiffness += 1.0),
        ("inertia_xx", |c| c.inertia_xx += 1.0),
        ("inertia_yy", |c| c.inertia_yy += 1.0),
    ];
    assert_eq!(
        mutators.len(),
        25,
        "CarParams has 25 hashed fields; update this list when fields change"
    );

    for (name, m) in mutators {
        let mut c = base.clone();
        m(&mut c);
        assert_ne!(hex(&c), base_h, "changing `{name}` did not change the hash");
    }
}

/// Canonical equality: a car resolved via a partial TOML overlay hashes the
/// same as one built directly with the identical resolved values. This proves
/// we hash the resolved `CarParams`, not the (lossy, format-sensitive) TOML.
#[test]
fn toml_overlay_equals_direct() {
    let base = CarParams::default();
    // Overlay overrides two fields; everything else falls back to `base`.
    let via_toml =
        parse_car_toml("[car]\nmass = 800.0\n[aero]\ndrag_coeff = 1.23\n", &base).expect("parse");

    let mut direct = base.clone();
    direct.mass = 800.0;
    direct.drag_coeff = 1.23;

    assert_eq!(
        hex(&via_toml),
        hex(&direct),
        "TOML-resolved and directly-built identical cars must hash identically"
    );
}

/// -0.0 vs +0.0 in a real field must not change the hash.
#[test]
fn signed_zero_field_equal() {
    let pos = CarParams {
        rolling_resistance: 0.0,
        ..Default::default()
    };
    let neg = CarParams {
        rolling_resistance: -0.0,
        ..Default::default()
    };
    assert_eq!(
        hex(&pos),
        hex(&neg),
        "-0.0 and +0.0 field must hash equally"
    );
}

/// FROZEN known-answer vector for `CarParams::default()`. Any accidental
/// encoding / field-order / float-policy change flips this and fails CI.
/// Update only as a deliberate change (and bump `apex_math::HASH_VERSION`).
#[test]
fn frozen_default_vector() {
    assert_eq!(
        hex(&CarParams::default()),
        "5e7045ea4f5efa96c493886b298fad444d2798caa6aefb64e68ed7d1bd1c44f7"
    );
}
