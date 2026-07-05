//! Content-hash correctness + stability tests for `RaceConfig`.

use apex_race::config::{race_config_hash, RaceConfig};

fn hex(c: &RaceConfig) -> String {
    race_config_hash(c).to_hex()
}

#[test]
fn determinism_and_clone() {
    let c = RaceConfig::silverstone_default();
    assert_eq!(hex(&c), hex(&c));
    assert_eq!(hex(&c), hex(&c.clone()));
}

/// Exhaustive per-field sensitivity over all 17 fields. Combined with the
/// destructure in `hash_into` (compile error if a field is added and left
/// unhandled), a new field cannot be silently omitted from the hash.
#[test]
fn exhaustive_field_sensitivity() {
    let base = RaceConfig::silverstone_default();
    let base_h = hex(&base);

    type Mutator = (&'static str, fn(&mut RaceConfig));
    let mutators: Vec<Mutator> = vec![
        ("n_laps", |c| c.n_laps += 1),
        ("pit_loss_time", |c| c.pit_loss_time += 1.0),
        ("pit_stop_time", |c| c.pit_stop_time += 1.0),
        ("track_length", |c| c.track_length += 1.0),
        ("start_fuel_kg", |c| c.start_fuel_kg += 1.0),
        ("fuel_per_lap", |c| c.fuel_per_lap += 1.0),
        ("fuel_time_factor", |c| c.fuel_time_factor += 1.0),
        ("safety_car_prob", |c| c.safety_car_prob += 1.0),
        ("vsc_prob", |c| c.vsc_prob += 1.0),
        ("dnf_prob", |c| c.dnf_prob += 1.0),
        ("rain_prob", |c| c.rain_prob += 1.0),
        ("driver_error_prob", |c| c.driver_error_prob += 1.0),
        ("safety_car_laps", |c| c.safety_car_laps += 1),
        ("vsc_laps", |c| c.vsc_laps += 1),
        ("safety_car_pace", |c| c.safety_car_pace += 1.0),
        ("overtake_gap_threshold", |c| {
            c.overtake_gap_threshold += 1.0
        }),
        ("overtake_base_prob", |c| c.overtake_base_prob += 1.0),
    ];
    assert_eq!(
        mutators.len(),
        17,
        "RaceConfig has 17 hashed fields; update this list when fields change"
    );

    for (name, m) in mutators {
        let mut c = base.clone();
        m(&mut c);
        assert_ne!(hex(&c), base_h, "changing `{name}` did not change the hash");
    }
}

/// FROZEN known-answer vector for `RaceConfig::silverstone_default()`.
#[test]
fn frozen_default_vector() {
    assert_eq!(
        hex(&RaceConfig::silverstone_default()),
        "4395eac7932aced33c227ab7261bddda5038db424e3b8cc1b6b0d7deec4585fc"
    );
}
