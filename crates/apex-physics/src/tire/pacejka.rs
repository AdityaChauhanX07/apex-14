//! Pacejka "Magic Formula" tire model.

use apex_math::Float;

/// Coefficients for the Pacejka Magic Formula.
///
/// The formula computes force as: D · sin(C · arctan(B·x - E·(B·x - arctan(B·x))))
/// where x is the slip quantity (slip angle for lateral, slip ratio for longitudinal).
#[derive(Debug, Clone, Copy)]
pub struct PacejkaCoeffs {
    /// Stiffness factor — controls the slope at the origin.
    pub b: f64,
    /// Shape factor — controls the shape of the curve (typically 1.0–2.0).
    pub c: f64,
    /// Peak factor — D = μ · F_z, the peak force output.
    /// This is computed dynamically from vertical load, so this field stores μ.
    pub mu: f64,
    /// Curvature factor — controls the shape near the peak (typically -2.0 to 1.0).
    pub e: f64,
}

impl PacejkaCoeffs {
    /// Returns representative F1 lateral tire coefficients.
    pub fn f1_lateral() -> Self {
        PacejkaCoeffs {
            b: 12.0,
            c: 1.5,
            mu: 1.75,
            e: -0.5,
        }
    }

    /// Returns representative F1 longitudinal tire coefficients.
    /// Longitudinal curves are typically stiffer (higher B) and sharper (higher C).
    pub fn f1_longitudinal() -> Self {
        PacejkaCoeffs {
            b: 14.0,
            c: 1.65,
            mu: 1.70,
            e: -0.3,
        }
    }
}

/// A single tire instance with lateral and longitudinal Pacejka coefficients,
/// plus load sensitivity parameters.
#[derive(Debug, Clone, Copy)]
pub struct PacejkaTire {
    /// Lateral (cornering) force coefficients.
    pub lateral: PacejkaCoeffs,
    /// Longitudinal (traction/braking) force coefficients.
    pub longitudinal: PacejkaCoeffs,
    /// Load sensitivity coefficient κ_μ — models the decrease in effective μ
    /// as vertical load increases above the nominal value.
    /// μ_eff = μ · (1 - κ_μ · (F_z - F_z_nom) / F_z_nom)
    pub load_sensitivity: f64,
    /// Nominal vertical load (N) — the load at which the base μ applies.
    pub fz_nominal: f64,
}

impl PacejkaTire {
    /// Returns a representative F1 tire with default coefficients.
    pub fn f1_default() -> Self {
        PacejkaTire {
            lateral: PacejkaCoeffs::f1_lateral(),
            longitudinal: PacejkaCoeffs::f1_longitudinal(),
            load_sensitivity: 0.1,
            fz_nominal: 4000.0, // ~quarter of car weight + some downforce
        }
    }

    /// Compute the effective friction coefficient at a given vertical load,
    /// accounting for load sensitivity.
    ///
    /// As F_z increases beyond F_z_nom, the effective μ decreases — this is
    /// a fundamental property of real tires and is why weight transfer costs
    /// total grip.
    pub fn effective_mu(&self, base_mu: f64, fz: f64) -> f64 {
        let ratio = (fz - self.fz_nominal) / self.fz_nominal;
        (base_mu * (1.0 - self.load_sensitivity * ratio)).max(0.0)
    }

    /// Evaluate the Pacejka Magic Formula for a given slip quantity and vertical load.
    ///
    /// This is the core function: F = D · sin(C · arctan(B·x - E·(B·x - arctan(B·x))))
    ///
    /// Arguments:
    /// - coeffs: which set of coefficients to use (lateral or longitudinal)
    /// - slip: the slip quantity (slip angle in radians for lateral, slip ratio dimensionless for longitudinal)
    /// - fz: vertical load on the tire (N, must be positive)
    ///
    /// Returns the force (N). Sign follows the sign of slip.
    pub fn magic_formula(&self, coeffs: &PacejkaCoeffs, slip: f64, fz: f64) -> f64 {
        if fz <= 0.0 {
            return 0.0;
        }

        let mu_eff = self.effective_mu(coeffs.mu, fz);
        let d = mu_eff * fz; // peak force

        let bx = coeffs.b * slip;
        let inner = bx - coeffs.e * (bx - bx.atan());

        d * (coeffs.c * inner.atan()).sin()
    }

    /// Compute pure lateral force from slip angle.
    ///
    /// Slip angle α is in radians. Positive α produces negative F_y (SAE convention),
    /// but this function returns the force magnitude with the sign following the slip
    /// angle sign for simplicity (positive slip → positive force → force opposing the slip).
    pub fn lateral_force(&self, slip_angle: f64, fz: f64) -> f64 {
        self.magic_formula(&self.lateral, slip_angle, fz)
    }

    /// Compute pure longitudinal force from slip ratio.
    ///
    /// Slip ratio κ is dimensionless: κ = (ωR - v_x) / max(|v_x|, v_min).
    /// Positive κ (wheel spinning faster) → positive force (traction).
    /// Negative κ (wheel spinning slower) → negative force (braking).
    pub fn longitudinal_force(&self, slip_ratio: f64, fz: f64) -> f64 {
        self.magic_formula(&self.longitudinal, slip_ratio, fz)
    }

    /// Compute the cornering stiffness — the slope of lateral force vs. slip angle
    /// at zero slip. This is analytically: C_α = B · C · D
    ///
    /// Useful for linear-regime analysis and validation.
    pub fn cornering_stiffness(&self, fz: f64) -> f64 {
        let mu_eff = self.effective_mu(self.lateral.mu, fz);
        let d = mu_eff * fz;
        self.lateral.b * self.lateral.c * d
    }

    /// Generic Pacejka Magic Formula evaluation.
    ///
    /// This is the same computation as `magic_formula`, but generic over any
    /// type implementing `Float` — including `Dual` for automatic differentiation.
    ///
    /// When called with `Dual` arguments, the return value's `.dual` field
    /// contains the derivative of the force with respect to whichever input
    /// was marked as `Dual::variable`.
    pub fn magic_formula_generic<T: Float>(&self, coeffs: &PacejkaCoeffs, slip: T, fz: T) -> T {
        // If fz <= 0, return zero
        if fz.real_value() <= 0.0 {
            return T::zero();
        }

        // Effective mu with load sensitivity
        let fz_nom = T::from_f64(self.fz_nominal);
        let load_sens = T::from_f64(self.load_sensitivity);
        let base_mu = T::from_f64(coeffs.mu);
        let ratio = (fz - fz_nom) / fz_nom;
        let mu_eff = (base_mu * (T::one() - load_sens * ratio)).max(T::zero());

        let d = mu_eff * fz; // peak force

        let b = T::from_f64(coeffs.b);
        let c = T::from_f64(coeffs.c);
        let e = T::from_f64(coeffs.e);

        let bx = b * slip;
        let inner = bx - e * (bx - bx.atan());

        d * (c * inner.atan()).sin()
    }

    /// Generic lateral force computation.
    pub fn lateral_force_generic<T: Float>(&self, slip_angle: T, fz: T) -> T {
        self.magic_formula_generic(&self.lateral, slip_angle, fz)
    }

    /// Generic longitudinal force computation.
    pub fn longitudinal_force_generic<T: Float>(&self, slip_ratio: T, fz: T) -> T {
        self.magic_formula_generic(&self.longitudinal, slip_ratio, fz)
    }

    /// Generic combined forces using the friction circle method.
    /// Returns (fx, fy) as a tuple of generic Float values.
    pub fn combined_forces_generic<T: Float>(
        &self,
        slip_angle: T,
        slip_ratio: T,
        fz: T,
    ) -> (T, T) {
        if fz.real_value() <= 0.0 {
            return (T::zero(), T::zero());
        }

        let fx_pure = self.longitudinal_force_generic(slip_ratio, fz);
        let fy_pure = self.lateral_force_generic(slip_angle, fz);

        // Friction circle limit
        let mu_lat = self.effective_mu(self.lateral.mu, fz.real_value());
        let mu_lon = self.effective_mu(self.longitudinal.mu, fz.real_value());
        let mu_avg = T::from_f64(0.5 * (mu_lat + mu_lon));
        let f_max = mu_avg * fz;

        let f_resultant = (fx_pure * fx_pure + fy_pure * fy_pure).sqrt();

        if f_resultant.real_value() <= f_max.real_value() || f_resultant.real_value() < 1e-6 {
            (fx_pure, fy_pure)
        } else {
            let scale = f_max / f_resultant;
            (fx_pure * scale, fy_pure * scale)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn zero_slip_is_zero_force() {
        let tire = PacejkaTire::f1_default();
        assert_eq!(tire.lateral_force(0.0, 4000.0), 0.0);
        assert_eq!(tire.longitudinal_force(0.0, 4000.0), 0.0);
    }

    #[test]
    fn antisymmetric() {
        let tire = PacejkaTire::f1_default();
        let fp = tire.lateral_force(0.1, 4000.0);
        let fn_ = tire.lateral_force(-0.1, 4000.0);
        assert!(approx(fp, -fn_, 1e-9), "f(0.1)={} f(-0.1)={}", fp, fn_);
        assert!(fp > 0.0);
    }

    #[test]
    fn curve_drops_past_peak() {
        let tire = PacejkaTire::f1_default();
        let f005 = tire.lateral_force(0.05, 4000.0);
        let f010 = tire.lateral_force(0.10, 4000.0);
        let f015 = tire.lateral_force(0.15, 4000.0);
        let f020 = tire.lateral_force(0.20, 4000.0);
        let f030 = tire.lateral_force(0.30, 4000.0);

        // force rises through the low-slip region
        assert!(f010 > f005, "f010 {} should exceed f005 {}", f010, f005);
        // and falls off past the peak — the "magic" of the formula
        assert!(f010 > f030, "f010 {} should exceed f030 {}", f010, f030);
        assert!(f015 > f030, "f015 {} should exceed f030 {}", f015, f030);
        assert!(f020 > f030, "f020 {} should exceed f030 {}", f020, f030);
    }

    #[test]
    fn peak_force_near_mu_fz() {
        let tire = PacejkaTire::f1_default();
        let fz = 4000.0;
        let expected_peak = 1.75 * fz; // D = μ · F_z = 7000 N

        // sweep slip and find the max force
        let mut max_f = 0.0_f64;
        let mut a = 0.0;
        while a <= 0.5 {
            max_f = max_f.max(tire.lateral_force(a, fz));
            a += 0.001;
        }

        assert!(
            (max_f - expected_peak).abs() / expected_peak < 0.05,
            "peak {} vs expected {}",
            max_f,
            expected_peak
        );
    }

    #[test]
    fn load_sensitivity_reduces_relative_grip() {
        let tire = PacejkaTire::f1_default();
        let alpha = 0.1;

        let f_low = tire.lateral_force(alpha, 3000.0);
        let f_high = tire.lateral_force(alpha, 6000.0);

        // more load -> more absolute force
        assert!(f_high > f_low, "f_high {} f_low {}", f_high, f_low);

        // but lower force-per-newton-of-load (effective μ drops with load)
        let ratio_low = f_low / 3000.0;
        let ratio_high = f_high / 6000.0;
        assert!(
            ratio_high < ratio_low,
            "ratio_high {} should be < ratio_low {}",
            ratio_high,
            ratio_low
        );
    }

    #[test]
    fn zero_and_negative_load() {
        let tire = PacejkaTire::f1_default();
        assert_eq!(tire.lateral_force(0.1, 0.0), 0.0);
        assert_eq!(tire.lateral_force(0.1, -100.0), 0.0);
        assert_eq!(tire.longitudinal_force(0.1, -100.0), 0.0);
    }

    #[test]
    fn cornering_stiffness_analytic_and_numeric() {
        let tire = PacejkaTire::f1_default();
        let fz = 4000.0;

        // analytic: C_α = B · C · D = 12 * 1.5 * (1.75 * 4000) = 126_000
        let expected = 12.0 * 1.5 * (1.75 * fz);
        let cs = tire.cornering_stiffness(fz);
        assert!(
            (cs - expected).abs() / expected < 0.05,
            "stiffness {} vs expected {}",
            cs,
            expected
        );

        // numeric: slope at small slip approximates the cornering stiffness
        let small = 0.001;
        let slope = tire.lateral_force(small, fz) / small;
        assert!(
            (slope - cs).abs() / cs < 0.05,
            "numeric slope {} vs analytic {}",
            slope,
            cs
        );
    }

    #[test]
    fn longitudinal_force_sign() {
        let tire = PacejkaTire::f1_default();
        let fz = 4000.0;
        assert_eq!(tire.longitudinal_force(0.0, fz), 0.0);
        assert!(tire.longitudinal_force(0.1, fz) > 0.0, "traction positive");
        assert!(tire.longitudinal_force(-0.1, fz) < 0.0, "braking negative");
    }

    #[test]
    fn f1_tire_sanity() {
        let tire = PacejkaTire::f1_default();
        let fz = 5000.0;

        // peak lateral force in a realistic range
        let mut max_f = 0.0_f64;
        let mut a = 0.0;
        while a <= 0.5 {
            max_f = max_f.max(tire.lateral_force(a, fz));
            a += 0.001;
        }
        assert!(
            (7000.0..=10000.0).contains(&max_f),
            "peak lateral force {} out of realistic range",
            max_f
        );

        // cornering stiffness in a realistic range for F1
        let cs = tire.cornering_stiffness(fz);
        assert!(
            (80_000.0..=200_000.0).contains(&cs),
            "cornering stiffness {} out of realistic range",
            cs
        );
    }

    // --- generic / autodiff tests ---

    use apex_math::Dual;

    #[test]
    fn generic_f64_matches_concrete() {
        let tire = PacejkaTire::f1_default();
        for &alpha in &[0.05, 0.1, 0.2] {
            for &fz in &[3000.0, 4000.0, 5000.0] {
                let concrete = tire.lateral_force(alpha, fz);
                let generic = tire.lateral_force_generic::<f64>(alpha, fz);
                assert!(
                    approx(concrete, generic, 1e-12),
                    "alpha {} fz {}: concrete {} vs generic {}",
                    alpha,
                    fz,
                    concrete,
                    generic
                );
            }
        }
    }

    #[test]
    fn dual_derivative_near_zero_is_cornering_stiffness() {
        let tire = PacejkaTire::f1_default();
        let alpha = Dual::variable(0.001);
        let fz = Dual::constant(4000.0);
        let result = tire.lateral_force_generic(alpha, fz);

        let cs = tire.cornering_stiffness(4000.0);
        assert!(
            (result.dual - cs).abs() / cs < 0.05,
            "dFy/dalpha {} vs cornering stiffness {}",
            result.dual,
            cs
        );
    }

    #[test]
    fn dual_derivative_near_zero_at_peak() {
        let tire = PacejkaTire::f1_default();
        // sweep to find where |dFy/dalpha| is minimized — the peak of the curve
        let mut best_alpha = 0.0;
        let mut best_slope = f64::MAX;
        let mut a = 0.05;
        while a <= 0.15 {
            let r = tire.lateral_force_generic(Dual::variable(a), Dual::constant(4000.0));
            if r.dual.abs() < best_slope {
                best_slope = r.dual.abs();
                best_alpha = a;
            }
            a += 0.001;
        }
        // the peak should sit within the swept window and have a near-flat slope
        assert!(
            (0.05..=0.15).contains(&best_alpha),
            "peak slip {} out of window",
            best_alpha
        );
        // slope at the peak is small relative to the cornering stiffness
        assert!(
            best_slope < 0.02 * tire.cornering_stiffness(4000.0),
            "slope at peak {} not near zero",
            best_slope
        );
    }

    #[test]
    fn dual_derivative_wrt_load() {
        let tire = PacejkaTire::f1_default();
        let alpha = Dual::constant(0.1);
        let fz = Dual::variable(4000.0);
        let result = tire.lateral_force_generic(alpha, fz);

        // dFy/dFz: more load -> more force, but below mu due to load sensitivity
        assert!(result.dual > 0.0, "dFy/dFz {} should be positive", result.dual);
        assert!(
            result.dual < tire.lateral.mu,
            "dFy/dFz {} should be below mu {}",
            result.dual,
            tire.lateral.mu
        );
    }

    #[test]
    fn combined_generic_f64_matches_concrete() {
        let tire = PacejkaTire::f1_default();
        let r = tire.combined_forces(0.1, 0.05, 4000.0);
        let (fx, fy) = tire.combined_forces_generic::<f64>(0.1, 0.05, 4000.0);
        assert!(approx(fx, r.fx, 1e-12), "fx {} vs {}", fx, r.fx);
        assert!(approx(fy, r.fy, 1e-12), "fy {} vs {}", fy, r.fy);
    }

    #[test]
    fn combined_dual_derivative_wrt_slip_angle() {
        let tire = PacejkaTire::f1_default();
        let fz = 4000.0;

        // combined: slip_angle is the variable
        let (_fx, fy) = tire.combined_forces_generic(
            Dual::variable(0.1),
            Dual::constant(0.05),
            Dual::constant(fz),
        );
        assert!(fy.dual.is_finite() && fy.dual != 0.0, "combined dFy/dalpha {}", fy.dual);

        // pure lateral derivative at the same operating point
        let pure = tire.lateral_force_generic(Dual::variable(0.1), Dual::constant(fz));

        // combined slip reduces the force, so its slope magnitude is smaller
        assert!(
            fy.dual.abs() < pure.dual.abs(),
            "combined slope {} should be smaller than pure {}",
            fy.dual,
            pure.dual
        );
    }
}
