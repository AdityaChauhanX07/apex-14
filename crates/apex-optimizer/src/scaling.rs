//! Diagonal variable/constraint scaling for the collocation NLP.
//!
//! Wraps any [`NlpEvaluator`] operating in raw SI units and presents another
//! `NlpEvaluator` operating in near-unit-magnitude scaled space, so a
//! Gauss-Newton (or future interior-point) solver sees a well-conditioned
//! problem without needing to know scaling exists at all — this module
//! attaches at the `NlpEvaluator`/`NlpProblem` boundary, not inside any
//! particular solver. See `docs/design/nlp-scaling.md` for the full design
//! rationale and the per-block reference-scale table.
//!
//! Callers MUST NOT report scaled-space residuals as if they were SI: after
//! solving against a [`ScaledEvaluator`], unscale the solution with
//! [`Scaling::unscale_x`] and re-evaluate feasibility against the original
//! (unscaled) evaluator before reporting `eq_violation`/`ineq_violation` to
//! any caller. `constraint_tol` in `GaussNewtonConfig` is compared inside the
//! solver against scaled residuals, but everything reported *out* of the NLP
//! boundary must be SI.

use apex_math::{CsrBuilder, CsrMatrix};

use crate::nlp::{NlpEvaluator, NlpProblem};

/// Floor applied to any reference scale so scaling never divides by zero
/// (e.g. a track with zero curvature everywhere would otherwise give
/// `curv_scale = 0`).
const SCALE_FLOOR: f64 = 1e-6;

/// Clamps a reference scale away from zero.
pub fn floor_scale(v: f64) -> f64 {
    v.abs().max(SCALE_FLOOR)
}

/// Diagonal (per-component) reference scales: `x_si[i] = x_scale[i] *
/// x_scaled[i]`, and likewise `c_si[i] = c_scale[i] * c_scaled[i]` for each
/// constraint block. All scales are static — computed once from car/track/
/// mesh reference values before a solve starts, never updated during
/// iteration (see `docs/design/nlp-scaling.md`, section A2).
#[derive(Debug, Clone)]
pub struct Scaling {
    /// Length `n_vars`.
    pub x_scale: Vec<f64>,
    /// Length `n_eq`.
    pub c_eq_scale: Vec<f64>,
    /// Length `n_ineq`.
    pub c_ineq_scale: Vec<f64>,
}

impl Scaling {
    /// `x_si -> x_scaled`.
    pub fn scale_x(&self, x_si: &[f64]) -> Vec<f64> {
        x_si.iter()
            .zip(&self.x_scale)
            .map(|(&xi, &s)| xi / s)
            .collect()
    }

    /// `x_scaled -> x_si`. Exact inverse of [`Scaling::scale_x`] to machine
    /// precision (a multiply undoing a divide by the same constant).
    pub fn unscale_x(&self, x_scaled: &[f64]) -> Vec<f64> {
        x_scaled
            .iter()
            .zip(&self.x_scale)
            .map(|(&xi, &s)| xi * s)
            .collect()
    }

    /// Rescales an [`NlpProblem`]'s variable bounds into scaled space.
    /// Infinite bounds are left as infinite (dividing infinity by a finite
    /// positive scale is still infinite, but this avoids relying on that
    /// IEEE-754 behavior implicitly).
    pub fn scale_problem(&self, problem: &NlpProblem) -> NlpProblem {
        let scale_bounds = |bounds: &[f64]| -> Vec<f64> {
            bounds
                .iter()
                .zip(&self.x_scale)
                .map(|(&b, &s)| if b.is_finite() { b / s } else { b })
                .collect()
        };
        NlpProblem {
            n_vars: problem.n_vars,
            n_eq: problem.n_eq,
            n_ineq: problem.n_ineq,
            lower_bounds: scale_bounds(&problem.lower_bounds),
            upper_bounds: scale_bounds(&problem.upper_bounds),
        }
    }

    /// Rescales a constraint Jacobian: `J_scaled[i,j] = J_si[i,j] *
    /// x_scale[j] / row_scale[i]` (chain rule for `x_si = x_scale .*
    /// x_scaled` composed with `c_scaled = c_si / row_scale`).
    fn scale_jacobian(&self, j_si: &CsrMatrix, row_scale: &[f64]) -> CsrMatrix {
        let mut builder = CsrBuilder::new(j_si.nrows(), j_si.ncols());
        for (row, &c) in row_scale.iter().enumerate().take(j_si.nrows()) {
            let (values, cols) = j_si.row_entries(row);
            for (&v, &col) in values.iter().zip(cols.iter()) {
                builder.add(row, col, v * self.x_scale[col] / c);
            }
        }
        builder.build()
    }
}

/// Wraps `inner` (an [`NlpEvaluator`] in raw SI units) to present a
/// scaled-space `NlpEvaluator`. The wrapped solver operates entirely in
/// scaled space and never sees SI units.
pub struct ScaledEvaluator<'a, E: NlpEvaluator> {
    pub inner: &'a E,
    pub scaling: &'a Scaling,
}

impl<E: NlpEvaluator> NlpEvaluator for ScaledEvaluator<'_, E> {
    fn objective(&self, x: &[f64]) -> f64 {
        self.inner.objective(&self.scaling.unscale_x(x))
    }

    fn objective_gradient(&self, x: &[f64]) -> Vec<f64> {
        let x_si = self.scaling.unscale_x(x);
        let grad_si = self.inner.objective_gradient(&x_si);
        grad_si
            .iter()
            .zip(&self.scaling.x_scale)
            .map(|(&g, &s)| g * s)
            .collect()
    }

    fn equality_constraints(&self, x: &[f64]) -> Vec<f64> {
        let x_si = self.scaling.unscale_x(x);
        let c_si = self.inner.equality_constraints(&x_si);
        c_si.iter()
            .zip(&self.scaling.c_eq_scale)
            .map(|(&c, &s)| c / s)
            .collect()
    }

    fn inequality_constraints(&self, x: &[f64]) -> Vec<f64> {
        let x_si = self.scaling.unscale_x(x);
        let c_si = self.inner.inequality_constraints(&x_si);
        c_si.iter()
            .zip(&self.scaling.c_ineq_scale)
            .map(|(&c, &s)| c / s)
            .collect()
    }

    fn equality_jacobian(&self, x: &[f64]) -> CsrMatrix {
        let x_si = self.scaling.unscale_x(x);
        let j_si = self.inner.equality_jacobian(&x_si);
        self.scaling.scale_jacobian(&j_si, &self.scaling.c_eq_scale)
    }

    fn inequality_jacobian(&self, x: &[f64]) -> CsrMatrix {
        let x_si = self.scaling.unscale_x(x);
        let j_si = self.inner.inequality_jacobian(&x_si);
        self.scaling
            .scale_jacobian(&j_si, &self.scaling.c_ineq_scale)
    }

    fn objective_hessian_vec(&self, x: &[f64], v: &[f64]) -> Vec<f64> {
        // f_scaled(x_s) = f(X·x_s), so ∇²f_scaled = X·∇²f·X (X = diag(x_scale)).
        // Hence H_scaled·v = x_scale ⊙ ( H_si · (x_scale ⊙ v) ).
        let x_si = self.scaling.unscale_x(x);
        let xv: Vec<f64> = v
            .iter()
            .zip(&self.scaling.x_scale)
            .map(|(&vi, &s)| vi * s)
            .collect();
        let hv = self.inner.objective_hessian_vec(&x_si, &xv);
        hv.iter()
            .zip(&self.scaling.x_scale)
            .map(|(&h, &s)| h * s)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f(x) = x0 + x1, no constraints. Enough to exercise the wrapper's
    /// plumbing without needing the collocation problem.
    struct Toy;
    impl NlpEvaluator for Toy {
        fn objective(&self, x: &[f64]) -> f64 {
            x[0] + x[1]
        }
        fn objective_gradient(&self, _x: &[f64]) -> Vec<f64> {
            vec![1.0, 1.0]
        }
        fn equality_constraints(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] - 2.0 * x[1]]
        }
        fn inequality_constraints(&self, _x: &[f64]) -> Vec<f64> {
            vec![]
        }
        fn equality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            let mut b = CsrBuilder::new(1, 2);
            b.add(0, 0, 1.0);
            b.add(0, 1, -2.0);
            b.build()
        }
        fn inequality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            CsrMatrix::zeros(0, 2)
        }
    }

    fn toy_scaling() -> Scaling {
        Scaling {
            x_scale: vec![10.0, 0.001],
            c_eq_scale: vec![5.0],
            c_ineq_scale: vec![],
        }
    }

    #[test]
    fn round_trip_unscale_scale_is_identity() {
        let scaling = toy_scaling();
        let x = vec![123.456, -0.0007];
        let round_tripped = scaling.unscale_x(&scaling.scale_x(&x));
        for (a, b) in x.iter().zip(round_tripped.iter()) {
            assert!(
                (a - b).abs() <= 1e-12 * a.abs().max(1.0),
                "round-trip mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn scaled_evaluator_matches_hand_derived_chain_rule() {
        let scaling = toy_scaling();
        let inner = Toy;
        let scaled = ScaledEvaluator {
            inner: &inner,
            scaling: &scaling,
        };

        let x_si = [4.0, 0.002];
        let x_scaled = scaling.scale_x(&x_si);

        // objective is scale-invariant (same physical value, just relabeled input).
        assert!((scaled.objective(&x_scaled) - inner.objective(&x_si)).abs() < 1e-12);

        // equality residual: c_si = x0 - 2*x1 = 4.0 - 0.004 = 3.996; c_scaled = c_si / 5.0.
        let c_scaled = scaled.equality_constraints(&x_scaled);
        let c_si = inner.equality_constraints(&x_si);
        assert!((c_scaled[0] - c_si[0] / 5.0).abs() < 1e-12);

        // Jacobian: dc/dx_scaled_j = J_si[0,j] * x_scale[j] / c_eq_scale[0].
        let j = scaled.equality_jacobian(&x_scaled);
        assert!((j.get(0, 0) - (1.0 * 10.0 / 5.0)).abs() < 1e-12);
        assert!((j.get(0, 1) - (-2.0 * 0.001 / 5.0)).abs() < 1e-12);
    }
}
