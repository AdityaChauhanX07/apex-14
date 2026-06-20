//! Car setup parameter space for optimization.
//!
//! Defines which [`CarParams`] fields are tunable, their bounds, and the
//! mapping between a flat parameter vector and [`CarParams`].
//!
//! Scope note: this space tunes only fields that live on [`CarParams`] — the
//! aerodynamic coefficients/balance, brake bias, CoG height, tire vertical
//! stiffness (a tire-pressure proxy), and longitudinal weight distribution.
//! Spring/damper/anti-roll-bar rates (`SuspensionParams`), ride-height-dependent
//! aero maps (`AeroModel`), and gear/final-drive ratios (`Engine`/`Powertrain`)
//! live on separate structs that are not reachable from [`CarParams`], so they
//! are out of scope for a `CarParams`-based setup space and for the QSS lap
//! objective (`qss_lap_sim`), which itself takes only `&CarParams`.

use apex_physics::CarParams;

/// A single tunable parameter with its name, bounds, and mapping.
#[derive(Debug, Clone)]
pub struct SetupParam {
    /// Human-readable name.
    pub name: String,
    /// Unit string for display.
    pub unit: String,
    /// Minimum allowed value.
    pub min: f64,
    /// Maximum allowed value.
    pub max: f64,
    /// Default/baseline value.
    pub baseline: f64,
}

/// Defines the full space of tunable parameters.
///
/// Maps between a flat `Vec<f64>` parameter vector and the corresponding
/// [`CarParams`] fields. The order of [`SetupParam`]s is the canonical index
/// order used by [`SetupSpace::apply`] and [`SetupSpace::extract`].
pub struct SetupSpace {
    /// Parameter definitions in index order.
    params: Vec<SetupParam>,
}

impl SetupSpace {
    /// Create the standard F1 setup space.
    ///
    /// Includes the aerodynamic, brake, and mass-distribution parameters that
    /// exist on [`CarParams`]. Baselines are read from [`CarParams::default`] so
    /// they stay consistent with the model. Index order (used by [`apply`] and
    /// [`extract`]):
    ///
    /// 0. `drag_coeff`
    /// 1. `lift_coeff`
    /// 2. `aero_balance_front`
    /// 3. `brake_bias_front`
    /// 4. `cog_height`
    /// 5. `tire_radial_stiffness`
    /// 6. front weight distribution (sets `cog_to_front`/`cog_to_rear`)
    ///
    /// [`apply`]: SetupSpace::apply
    /// [`extract`]: SetupSpace::extract
    pub fn f1_standard() -> Self {
        let d = CarParams::default();
        let params = vec![
            SetupParam {
                name: "drag_coeff".to_string(),
                unit: "Cd".to_string(),
                min: 0.70,
                max: 1.30,
                baseline: d.drag_coeff,
            },
            SetupParam {
                name: "lift_coeff".to_string(),
                unit: "Cl".to_string(),
                min: 2.00,
                max: 4.50,
                baseline: d.lift_coeff,
            },
            SetupParam {
                name: "aero_balance_front".to_string(),
                unit: "frac".to_string(),
                min: 0.40,
                max: 0.52,
                baseline: d.aero_balance_front,
            },
            SetupParam {
                name: "brake_bias_front".to_string(),
                unit: "frac".to_string(),
                min: 0.50,
                max: 0.70,
                baseline: d.brake_bias_front,
            },
            SetupParam {
                name: "cog_height".to_string(),
                unit: "m".to_string(),
                min: 0.25,
                max: 0.35,
                baseline: d.cog_height,
            },
            SetupParam {
                name: "tire_radial_stiffness".to_string(),
                unit: "N/m".to_string(),
                min: 180_000.0,
                max: 320_000.0,
                baseline: d.tire_radial_stiffness,
            },
            SetupParam {
                name: "weight_dist_front".to_string(),
                unit: "frac".to_string(),
                min: 0.46,
                max: 0.58,
                // Static front load fraction = cog_to_rear / wheelbase.
                baseline: d.cog_to_rear / d.wheelbase,
            },
        ];
        Self { params }
    }

    /// Number of tunable parameters.
    pub fn dim(&self) -> usize {
        self.params.len()
    }

    /// Get the parameter definitions, in index order.
    pub fn params(&self) -> &[SetupParam] {
        &self.params
    }

    /// Apply a parameter vector to a base [`CarParams`], producing a modified copy.
    ///
    /// Missing entries (when `params` is shorter than [`dim`](SetupSpace::dim))
    /// fall back to the parameter's baseline. Each value is clamped to its bounds
    /// before being written.
    pub fn apply(&self, base: &CarParams, params: &[f64]) -> CarParams {
        let mut car = base.clone();
        for (i, def) in self.params.iter().enumerate() {
            let raw = params.get(i).copied().unwrap_or(def.baseline);
            let v = raw.clamp(def.min, def.max);
            match i {
                0 => car.drag_coeff = v,
                1 => car.lift_coeff = v,
                2 => car.aero_balance_front = v,
                3 => car.brake_bias_front = v,
                4 => car.cog_height = v,
                5 => car.tire_radial_stiffness = v,
                6 => {
                    // Front weight distribution fraction -> longitudinal CoG
                    // position, holding the wheelbase constant. The static front
                    // load fraction equals cog_to_rear / wheelbase.
                    car.cog_to_rear = v * car.wheelbase;
                    car.cog_to_front = car.wheelbase - car.cog_to_rear;
                }
                _ => {}
            }
        }
        car
    }

    /// Extract the parameter vector from a [`CarParams`].
    ///
    /// Inverse of [`apply`](SetupSpace::apply): reads the tunable fields and
    /// returns them as a flat vector in index order.
    pub fn extract(&self, car: &CarParams) -> Vec<f64> {
        (0..self.params.len())
            .map(|i| match i {
                0 => car.drag_coeff,
                1 => car.lift_coeff,
                2 => car.aero_balance_front,
                3 => car.brake_bias_front,
                4 => car.cog_height,
                5 => car.tire_radial_stiffness,
                6 => car.cog_to_rear / car.wheelbase,
                _ => 0.0,
            })
            .collect()
    }

    /// Get the baseline parameter vector (from the default car).
    pub fn baseline_vec(&self) -> Vec<f64> {
        self.params.iter().map(|p| p.baseline).collect()
    }

    /// Get the per-dimension `(min, max)` bounds, in index order.
    pub fn bounds(&self) -> Vec<(f64, f64)> {
        self.params.iter().map(|p| (p.min, p.max)).collect()
    }

    /// Clamp a parameter vector to bounds, in place.
    pub fn clamp(&self, params: &mut [f64]) {
        for (p, def) in params.iter_mut().zip(&self.params) {
            *p = p.clamp(def.min, def.max);
        }
    }

    /// Format a parameter vector as a human-readable report.
    pub fn format_report(&self, params: &[f64]) -> String {
        let mut report = String::new();
        for (i, def) in self.params.iter().enumerate() {
            let val = params.get(i).copied().unwrap_or(def.baseline);
            report.push_str(&format!(
                "  {:<30} {:>12.3} {:<6} (range: {:.3} - {:.3})\n",
                def.name, val, def.unit, def.min, def.max
            ));
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_setup_space_creation() {
        let space = SetupSpace::f1_standard();
        assert!(space.dim() > 0, "setup space should have parameters");
        assert_eq!(space.dim(), space.params().len());
    }

    #[test]
    fn test_baseline_vec_length() {
        let space = SetupSpace::f1_standard();
        assert_eq!(space.baseline_vec().len(), space.dim());
    }

    #[test]
    fn test_apply_extract_roundtrip() {
        let space = SetupSpace::f1_standard();
        let base = CarParams::default();

        // Extract from the default car, apply back, and re-extract: the two
        // parameter vectors must agree (the mapping is bijective per index).
        let v0 = space.extract(&base);
        let car = space.apply(&base, &v0);
        let v1 = space.extract(&car);

        assert_eq!(v0.len(), v1.len());
        for (i, (a, b)) in v0.iter().zip(&v1).enumerate() {
            assert!((a - b).abs() < 1e-9, "param {i} differs: {a} vs {b}");
        }

        // The coupled weight-distribution param must reconstruct the CoG
        // positions of the original car.
        assert!((car.cog_to_front - base.cog_to_front).abs() < 1e-9);
        assert!((car.cog_to_rear - base.cog_to_rear).abs() < 1e-9);
        assert!((car.drag_coeff - base.drag_coeff).abs() < 1e-12);
        assert!((car.tire_radial_stiffness - base.tire_radial_stiffness).abs() < 1e-6);
    }

    #[test]
    fn test_clamp_respects_bounds() {
        let space = SetupSpace::f1_standard();
        // Start well outside every bound (both directions).
        let mut lo: Vec<f64> = space.params().iter().map(|p| p.min - 100.0).collect();
        let mut hi: Vec<f64> = space.params().iter().map(|p| p.max + 100.0).collect();
        space.clamp(&mut lo);
        space.clamp(&mut hi);
        for (i, def) in space.params().iter().enumerate() {
            assert!(
                (def.min..=def.max).contains(&lo[i]),
                "lo[{i}] out of bounds"
            );
            assert!(
                (def.min..=def.max).contains(&hi[i]),
                "hi[{i}] out of bounds"
            );
            assert!(
                (lo[i] - def.min).abs() < 1e-12,
                "lo[{i}] should clamp to min"
            );
            assert!(
                (hi[i] - def.max).abs() < 1e-12,
                "hi[{i}] should clamp to max"
            );
        }
    }

    #[test]
    fn test_apply_clamps_out_of_range() {
        let space = SetupSpace::f1_standard();
        let base = CarParams::default();
        // Drag far above its max should be clamped, not written verbatim.
        let mut v = space.extract(&base);
        v[0] = 99.0;
        let car = space.apply(&base, &v);
        assert!((car.drag_coeff - space.params()[0].max).abs() < 1e-12);
    }

    #[test]
    fn test_format_report() {
        let space = SetupSpace::f1_standard();
        let report = space.format_report(&space.baseline_vec());
        assert!(!report.is_empty());
        // Every parameter name should appear in the report.
        for def in space.params() {
            assert!(
                report.contains(&def.name),
                "report missing parameter {}",
                def.name
            );
        }
    }
}
