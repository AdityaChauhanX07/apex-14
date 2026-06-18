//! Load and save [`CarParams`] from TOML files.
//!
//! A config file is an *overlay*: every field is optional, and only the fields
//! present override the supplied base parameters (typically
//! [`CarParams::default`] or [`CarParams::f1_2024_calibrated`]). This lets users
//! define cars without editing Rust code.

use serde::{Deserialize, Serialize};

use crate::CarParams;

/// Car configuration as represented in a TOML file.
/// All fields are optional - missing fields use the default CarParams values.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CarConfig {
    #[serde(default)]
    pub car: Option<CarSection>,
    #[serde(default)]
    pub aero: Option<AeroSection>,
    #[serde(default)]
    pub tires: Option<TireSection>,
    #[serde(default)]
    pub suspension: Option<SuspensionSection>,
    #[serde(default)]
    pub geometry: Option<GeometrySection>,
    #[serde(default)]
    pub powertrain: Option<PowertrainSection>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CarSection {
    pub name: Option<String>,
    pub mass: Option<f64>,
    pub max_drive_force: Option<f64>,
    pub max_brake_force: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AeroSection {
    pub frontal_area: Option<f64>,
    pub drag_coeff: Option<f64>,
    pub lift_coeff: Option<f64>,
    pub aero_balance_front: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TireSection {
    pub mu: Option<f64>,
    pub rolling_resistance: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SuspensionSection {
    pub front_spring_rate: Option<f64>,
    pub rear_spring_rate: Option<f64>,
    pub front_damping_bump: Option<f64>,
    pub front_damping_rebound: Option<f64>,
    pub rear_damping_bump: Option<f64>,
    pub rear_damping_rebound: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeometrySection {
    pub wheelbase: Option<f64>,
    pub cog_to_front: Option<f64>,
    pub cog_to_rear: Option<f64>,
    pub cog_height: Option<f64>,
    pub track_width_front: Option<f64>,
    pub track_width_rear: Option<f64>,
    pub wheel_radius: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PowertrainSection {
    pub drive_distribution: Option<f64>,
    pub brake_bias_front: Option<f64>,
}

/// Load car parameters from a TOML file.
/// Missing fields fall back to the base parameters (default or calibrated).
pub fn load_car_toml(
    path: &std::path::Path,
    base: &CarParams,
) -> Result<CarParams, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    parse_car_toml(&content, base)
}

/// Parse car parameters from a TOML string.
pub fn parse_car_toml(
    toml_content: &str,
    base: &CarParams,
) -> Result<CarParams, Box<dyn std::error::Error>> {
    let config: CarConfig = toml::from_str(toml_content)?;
    Ok(apply_config(config, base))
}

/// Apply a CarConfig overlay onto base CarParams.
/// Only fields present in the config override the base values.
fn apply_config(config: CarConfig, base: &CarParams) -> CarParams {
    let mut p = base.clone();

    if let Some(car) = config.car {
        if let Some(v) = car.mass {
            p.mass = v;
        }
        if let Some(v) = car.max_drive_force {
            p.max_drive_force = v;
        }
        if let Some(v) = car.max_brake_force {
            p.max_brake_force = v;
        }
    }

    if let Some(aero) = config.aero {
        if let Some(v) = aero.frontal_area {
            p.frontal_area = v;
        }
        if let Some(v) = aero.drag_coeff {
            p.drag_coeff = v;
        }
        if let Some(v) = aero.lift_coeff {
            p.lift_coeff = v;
        }
        if let Some(v) = aero.aero_balance_front {
            p.aero_balance_front = v;
        }
    }

    if let Some(tires) = config.tires {
        if let Some(v) = tires.mu {
            p.tire_mu = v;
        }
        if let Some(v) = tires.rolling_resistance {
            p.rolling_resistance = v;
        }
    }

    if let Some(geo) = config.geometry {
        if let Some(v) = geo.wheelbase {
            p.wheelbase = v;
        }
        if let Some(v) = geo.cog_to_front {
            p.cog_to_front = v;
        }
        if let Some(v) = geo.cog_to_rear {
            p.cog_to_rear = v;
        }
        if let Some(v) = geo.cog_height {
            p.cog_height = v;
        }
        if let Some(v) = geo.track_width_front {
            p.track_width_front = v;
        }
        if let Some(v) = geo.track_width_rear {
            p.track_width_rear = v;
        }
        if let Some(v) = geo.wheel_radius {
            p.wheel_radius = v;
        }
    }

    if let Some(pt) = config.powertrain {
        if let Some(v) = pt.drive_distribution {
            p.drive_distribution = v;
        }
        if let Some(v) = pt.brake_bias_front {
            p.brake_bias_front = v;
        }
    }

    // Suspension is on SuspensionParams, not CarParams, so skip for now
    // (the SuspensionSystem is created separately)

    p
}

/// Export CarParams to a TOML string.
pub fn export_car_toml(params: &CarParams, name: &str) -> String {
    let config = CarConfig {
        car: Some(CarSection {
            name: Some(name.to_string()),
            mass: Some(params.mass),
            max_drive_force: Some(params.max_drive_force),
            max_brake_force: Some(params.max_brake_force),
        }),
        aero: Some(AeroSection {
            frontal_area: Some(params.frontal_area),
            drag_coeff: Some(params.drag_coeff),
            lift_coeff: Some(params.lift_coeff),
            aero_balance_front: Some(params.aero_balance_front),
        }),
        tires: Some(TireSection {
            mu: Some(params.tire_mu),
            rolling_resistance: Some(params.rolling_resistance),
        }),
        geometry: Some(GeometrySection {
            wheelbase: Some(params.wheelbase),
            cog_to_front: Some(params.cog_to_front),
            cog_to_rear: Some(params.cog_to_rear),
            cog_height: Some(params.cog_height),
            track_width_front: Some(params.track_width_front),
            track_width_rear: Some(params.track_width_rear),
            wheel_radius: Some(params.wheel_radius),
        }),
        powertrain: Some(PowertrainSection {
            drive_distribution: Some(params.drive_distribution),
            brake_bias_front: Some(params.brake_bias_front),
        }),
        suspension: None,
    };

    toml::to_string_pretty(&config).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CALIBRATED_TOML: &str = r#"
[car]
name = "F1 2024 Calibrated"
mass = 798.0
max_drive_force = 11000.0
max_brake_force = 25000.0

[aero]
frontal_area = 1.5
drag_coeff = 1.10
lift_coeff = 2.80
aero_balance_front = 0.44

[tires]
mu = 1.55
rolling_resistance = 0.015

[geometry]
wheelbase = 3.60
cog_to_front = 1.67
cog_to_rear = 1.93
cog_height = 0.30
wheel_radius = 0.330
"#;

    const F3_TOML: &str = r#"
[car]
name = "Formula 3 2024"
mass = 690.0
max_drive_force = 6000.0
max_brake_force = 15000.0

[aero]
frontal_area = 1.3
drag_coeff = 0.85
lift_coeff = 1.80
aero_balance_front = 0.43

[tires]
mu = 1.45
rolling_resistance = 0.018

[geometry]
wheelbase = 3.10
cog_to_front = 1.45
cog_to_rear = 1.65
cog_height = 0.28
wheel_radius = 0.310
"#;

    #[test]
    fn parse_full_config() {
        let p = parse_car_toml(CALIBRATED_TOML, &CarParams::default()).unwrap();
        assert_eq!(p.mass, 798.0);
        assert_eq!(p.drag_coeff, 1.10);
        assert_eq!(p.tire_mu, 1.55);
        assert_eq!(p.max_drive_force, 11000.0);
        assert_eq!(p.lift_coeff, 2.80);
    }

    #[test]
    fn partial_config_overlays_base() {
        let base = CarParams::default();
        let p = parse_car_toml("[car]\nmass = 700.0\n", &base).unwrap();
        // Overridden field.
        assert_eq!(p.mass, 700.0);
        // Everything else equals the base.
        assert_eq!(p.drag_coeff, base.drag_coeff);
        assert_eq!(p.tire_mu, base.tire_mu);
        assert_eq!(p.max_drive_force, base.max_drive_force);
        assert_eq!(p.wheelbase, base.wheelbase);
        assert_eq!(p.brake_bias_front, base.brake_bias_front);
    }

    #[test]
    fn empty_config_returns_base() {
        let base = CarParams::default();
        let p = parse_car_toml("", &base).unwrap();
        assert_eq!(p.mass, base.mass);
        assert_eq!(p.drag_coeff, base.drag_coeff);
        assert_eq!(p.tire_mu, base.tire_mu);
        assert_eq!(p.wheelbase, base.wheelbase);
    }

    #[test]
    fn round_trip_preserves_fields() {
        let original = CarParams::f1_2024_calibrated();
        let toml_str = export_car_toml(&original, "F1 2024 Calibrated");
        let parsed = parse_car_toml(&toml_str, &CarParams::default()).unwrap();

        let close = |a: f64, b: f64| (a - b).abs() < 1e-10;
        assert!(close(parsed.mass, original.mass));
        assert!(close(parsed.max_drive_force, original.max_drive_force));
        assert!(close(parsed.max_brake_force, original.max_brake_force));
        assert!(close(parsed.frontal_area, original.frontal_area));
        assert!(close(parsed.drag_coeff, original.drag_coeff));
        assert!(close(parsed.lift_coeff, original.lift_coeff));
        assert!(close(
            parsed.aero_balance_front,
            original.aero_balance_front
        ));
        assert!(close(parsed.tire_mu, original.tire_mu));
        assert!(close(
            parsed.rolling_resistance,
            original.rolling_resistance
        ));
        assert!(close(parsed.wheelbase, original.wheelbase));
        assert!(close(parsed.cog_to_front, original.cog_to_front));
        assert!(close(parsed.cog_to_rear, original.cog_to_rear));
        assert!(close(parsed.cog_height, original.cog_height));
        assert!(close(parsed.track_width_front, original.track_width_front));
        assert!(close(parsed.track_width_rear, original.track_width_rear));
        assert!(close(parsed.wheel_radius, original.wheel_radius));
        assert!(close(
            parsed.drive_distribution,
            original.drive_distribution
        ));
        assert!(close(parsed.brake_bias_front, original.brake_bias_front));
    }

    #[test]
    fn load_from_file() {
        let path = std::env::temp_dir().join(format!("apex_car_test_{}.toml", std::process::id()));
        std::fs::write(&path, CALIBRATED_TOML).unwrap();

        let p = load_car_toml(&path, &CarParams::default()).unwrap();
        assert_eq!(p.mass, 798.0);
        assert_eq!(p.tire_mu, 1.55);
        assert_eq!(p.max_brake_force, 25000.0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn invalid_toml_errors() {
        // Malformed: not valid TOML syntax.
        let result = parse_car_toml("[car\nmass = = 700", &CarParams::default());
        assert!(result.is_err(), "malformed TOML should error, not panic");

        // Wrong type for a numeric field.
        let result = parse_car_toml("[car]\nmass = \"heavy\"\n", &CarParams::default());
        assert!(result.is_err(), "type mismatch should error");
    }

    #[test]
    fn f3_car_differs_from_f1() {
        use apex_track::{build_track, oval_track};

        let f3 = parse_car_toml(F3_TOML, &CarParams::default()).unwrap();
        assert_eq!(f3.mass, 690.0);
        assert_ne!(f3.mass, 798.0);

        let f1 = CarParams::f1_2024_calibrated();

        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 500);
        let oval = build_track("oval", &pts, closed);
        let f3_lap = crate::qss_lap_sim(&oval, &f3).lap_time;
        let f1_lap = crate::qss_lap_sim(&oval, &f1).lap_time;

        assert!(f3_lap.is_finite() && f1_lap.is_finite());
        assert!(
            (f3_lap - f1_lap).abs() > 1e-3,
            "F3 lap {f3_lap} should differ from F1 calibrated {f1_lap}"
        );
    }
}
