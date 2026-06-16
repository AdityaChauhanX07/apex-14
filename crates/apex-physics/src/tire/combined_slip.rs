//! Combined slip tire forces via the friction-circle (similarity) method.
//!
//! Pure-slip lateral and longitudinal forces from the [`PacejkaTire`] model are
//! scaled so the resultant force vector stays within the friction circle. This
//! captures the core behavior that using grip in one direction reduces the grip
//! available in the other.

use super::pacejka::PacejkaTire;

/// Result of a combined slip tire force computation.
#[derive(Debug, Clone, Copy)]
pub struct CombinedSlipResult {
    /// Lateral force after combined slip reduction (N).
    pub fy: f64,
    /// Longitudinal force after combined slip reduction (N).
    pub fx: f64,
    /// Pure lateral force before combined slip (N) — for diagnostics.
    pub fy_pure: f64,
    /// Pure longitudinal force before combined slip (N) — for diagnostics.
    pub fx_pure: f64,
    /// Combined grip utilization (0.0–1.0 within limit, >1.0 if exceeded).
    pub grip_utilization: f64,
}

impl PacejkaTire {
    /// Compute combined lateral and longitudinal tire forces using the
    /// friction ellipse method.
    ///
    /// When a tire generates both lateral and longitudinal force simultaneously,
    /// each force is reduced from its pure-slip value. The total force vector
    /// must stay within the friction ellipse defined by the peak forces.
    ///
    /// # Arguments
    /// - `slip_angle`: tire slip angle α (rad)
    /// - `slip_ratio`: longitudinal slip ratio κ (dimensionless)
    /// - `fz`: vertical load (N)
    ///
    /// # Method
    /// 1. Compute pure forces: `F_x0 = magic_formula(longitudinal, κ, fz)`,
    ///    `F_y0 = magic_formula(lateral, α, fz)`.
    /// 2. Compute the friction circle limit `F_max = μ_eff · fz`, using the
    ///    average of the lateral and longitudinal effective μ.
    /// 3. If `sqrt(F_x0² + F_y0²) <= F_max`, the forces are within the circle and
    ///    are returned unmodified.
    /// 4. Otherwise scale both forces proportionally to bring the resultant back
    ///    onto the circle: `scale = F_max / sqrt(F_x0² + F_y0²)`.
    ///
    /// This is a simplification of the full MF 5.2 combined slip equations, but it
    /// correctly captures the key behavior: using grip in one direction reduces
    /// available grip in the other.
    pub fn combined_forces(
        &self,
        slip_angle: f64,
        slip_ratio: f64,
        fz: f64,
    ) -> CombinedSlipResult {
        if fz <= 0.0 {
            return CombinedSlipResult {
                fy: 0.0,
                fx: 0.0,
                fy_pure: 0.0,
                fx_pure: 0.0,
                grip_utilization: 0.0,
            };
        }

        let fx_pure = self.longitudinal_force(slip_ratio, fz);
        let fy_pure = self.lateral_force(slip_angle, fz);

        // Friction circle limit: use the average of lateral and longitudinal μ
        let mu_avg = 0.5
            * (self.effective_mu(self.lateral.mu, fz)
                + self.effective_mu(self.longitudinal.mu, fz));
        let f_max = mu_avg * fz;

        let f_resultant = (fx_pure * fx_pure + fy_pure * fy_pure).sqrt();
        let grip_utilization = if f_max > 0.0 { f_resultant / f_max } else { 0.0 };

        if f_resultant <= f_max || f_resultant < 1e-6 {
            // Within the friction circle — use pure forces unmodified
            CombinedSlipResult {
                fy: fy_pure,
                fx: fx_pure,
                fy_pure,
                fx_pure,
                grip_utilization,
            }
        } else {
            // Exceeds friction circle — scale proportionally
            let scale = f_max / f_resultant;
            CombinedSlipResult {
                fy: fy_pure * scale,
                fx: fx_pure * scale,
                fy_pure,
                fx_pure,
                grip_utilization,
            }
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
    fn pure_lateral_only() {
        let tire = PacejkaTire::f1_default();
        let fz = 4000.0;
        let r = tire.combined_forces(0.1, 0.0, fz);
        let pure = tire.lateral_force(0.1, fz);

        // no longitudinal slip -> no longitudinal force
        assert!(approx(r.fx, 0.0, 1e-9), "fx {}", r.fx);
        // fy follows the pure lateral force. Near the lateral peak the
        // averaged-μ friction circle (which uses the mean of lateral and
        // longitudinal μ) can clip a pure-axis force by a fraction of a
        // percent, so allow a small tolerance and require fy <= pure.
        assert!(approx(r.fy, pure, 0.02 * pure.abs()), "fy {} vs pure {}", r.fy, pure);
        assert!(r.fy.abs() <= pure.abs() + 1e-9, "fy {} should not exceed pure {}", r.fy, pure);
        // diagnostic always reports the true pure value
        assert!(approx(r.fy_pure, pure, 1e-12));
    }

    #[test]
    fn pure_longitudinal_only() {
        let tire = PacejkaTire::f1_default();
        let r = tire.combined_forces(0.0, 0.1, 4000.0);
        assert!(approx(r.fy, 0.0, 1e-9), "fy {}", r.fy);
        assert!(
            approx(r.fx, tire.longitudinal_force(0.1, 4000.0), 1e-9),
            "fx {}",
            r.fx
        );
        assert!(approx(r.fx, r.fx_pure, 1e-12));
    }

    #[test]
    fn combined_slip_reduces_both_forces() {
        let tire = PacejkaTire::f1_default();
        let fz = 4000.0;
        let r = tire.combined_forces(0.1, 0.1, fz);

        let pure_fy = tire.lateral_force(0.1, fz);
        let pure_fx = tire.longitudinal_force(0.1, fz);

        // combining costs grip in both directions
        assert!(r.fy.abs() < pure_fy.abs(), "fy {} vs pure {}", r.fy, pure_fy);
        assert!(r.fx.abs() < pure_fx.abs(), "fx {} vs pure {}", r.fx, pure_fx);
        assert!(r.grip_utilization > 1.0, "should exceed circle: {}", r.grip_utilization);
    }

    #[test]
    fn friction_circle_scaling() {
        let tire = PacejkaTire::f1_default();
        let fz = 4000.0;
        let r = tire.combined_forces(0.15, 0.15, fz);

        let mu_avg = 0.5
            * (tire.effective_mu(tire.lateral.mu, fz)
                + tire.effective_mu(tire.longitudinal.mu, fz));
        let f_max = mu_avg * fz;

        let resultant = (r.fx * r.fx + r.fy * r.fy).sqrt();
        // scaled back onto (or within) the friction circle
        assert!(resultant <= f_max + 1e-6, "resultant {} vs f_max {}", resultant, f_max);
        // at the limit
        assert!(
            approx(resultant, f_max, 1.0),
            "resultant {} not at f_max {}",
            resultant,
            f_max
        );
        assert!(r.grip_utilization > 1.0, "grip_util {}", r.grip_utilization);
    }

    #[test]
    fn trail_braking_reduces_lateral() {
        let tire = PacejkaTire::f1_default();
        let fz = 4000.0;
        // corner entry: lateral slip plus some braking
        let r = tire.combined_forces(0.08, -0.05, fz);

        let pure_fy = tire.lateral_force(0.08, fz);
        // braking trades lateral grip for deceleration
        assert!(r.fy.abs() < pure_fy.abs(), "fy {} vs pure {}", r.fy, pure_fy);
        // braking produces a negative longitudinal force
        assert!(r.fx < 0.0, "fx {} should be braking (negative)", r.fx);
    }

    #[test]
    fn zero_load_returns_zeros() {
        let tire = PacejkaTire::f1_default();
        let r = tire.combined_forces(0.1, 0.1, 0.0);
        assert_eq!(r.fy, 0.0);
        assert_eq!(r.fx, 0.0);
        assert_eq!(r.fy_pure, 0.0);
        assert_eq!(r.fx_pure, 0.0);
        assert_eq!(r.grip_utilization, 0.0);
    }

    #[test]
    fn symmetry() {
        let tire = PacejkaTire::f1_default();
        let fz = 4000.0;
        let pos = tire.combined_forces(0.1, 0.05, fz);
        let neg = tire.combined_forces(-0.1, -0.05, fz);

        assert!(approx(pos.fy, -neg.fy, 1e-9), "fy {} vs {}", pos.fy, neg.fy);
        assert!(approx(pos.fx, -neg.fx, 1e-9), "fx {} vs {}", pos.fx, neg.fx);
        assert!(
            approx(pos.grip_utilization, neg.grip_utilization, 1e-9),
            "grip {} vs {}",
            pos.grip_utilization,
            neg.grip_utilization
        );
    }

    #[test]
    fn diagnostic_fields_match_pure_outputs() {
        let tire = PacejkaTire::f1_default();
        let fz = 4000.0;

        // within-circle case
        let within = tire.combined_forces(0.02, 0.0, fz);
        assert_eq!(within.fy_pure, tire.lateral_force(0.02, fz));
        assert_eq!(within.fx_pure, tire.longitudinal_force(0.0, fz));

        // scaled case — diagnostics still report the pure values
        let scaled = tire.combined_forces(0.15, 0.15, fz);
        assert_eq!(scaled.fy_pure, tire.lateral_force(0.15, fz));
        assert_eq!(scaled.fx_pure, tire.longitudinal_force(0.15, fz));
    }
}
