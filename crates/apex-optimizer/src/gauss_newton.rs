//! Damped Gauss-Newton solver tailored for the collocation problem.
//!
//! The objective is simple (linear in the time steps) and the difficulty is in
//! satisfying the dynamics-defect equality constraints. A Gauss-Newton step
//! solves the linearized least-squares feasibility problem (with a small
//! objective pull and an inequality penalty), which converges far better than
//! first-order gradient descent on this problem.

use apex_math::CsrMatrix;

use crate::nlp::{NlpEvaluator, NlpProblem};

/// Configuration for the Gauss-Newton collocation solver.
#[derive(Debug, Clone)]
pub struct GaussNewtonConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on constraint violation (max absolute).
    pub constraint_tol: f64,
    /// Convergence tolerance on step size (relative L2 norm).
    pub step_tol: f64,
    /// Damping factor for the Gauss-Newton step (0 < α ≤ 1). Smaller values make
    /// the solver more conservative but more stable.
    pub damping: f64,
    /// Regularization added to the diagonal of `JᵀJ` (Levenberg-Marquardt style).
    pub regularization: f64,
    /// Print progress every N iterations (0 = silent).
    pub print_interval: usize,
}

impl Default for GaussNewtonConfig {
    fn default() -> Self {
        GaussNewtonConfig {
            max_iterations: 100,
            constraint_tol: 1e-4,
            step_tol: 1e-8,
            damping: 0.5,
            regularization: 1e-4,
            print_interval: 0,
        }
    }
}

impl apex_math::ContentHash for GaussNewtonConfig {
    /// Encode the result-determining fields. `print_interval` is EXCLUDED
    /// (cosmetic, bound to `_`). The destructure forces any new field to be
    /// handled here before it compiles.
    fn hash_into(&self, w: &mut apex_math::HashWriter) {
        let GaussNewtonConfig {
            max_iterations,
            constraint_tol,
            step_tol,
            damping,
            regularization,
            print_interval: _, // cosmetic; excluded from content identity
        } = self;
        w.usize(*max_iterations);
        w.f64(*constraint_tol);
        w.f64(*step_tol);
        w.f64(*damping);
        w.f64(*regularization);
    }
}

/// Result of the Gauss-Newton solve.
#[derive(Debug, Clone)]
pub struct GaussNewtonResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Final objective value.
    pub objective: f64,
    /// Maximum equality constraint violation.
    pub eq_violation: f64,
    /// Maximum inequality constraint violation (positive = violated).
    pub ineq_violation: f64,
    /// Iterations performed.
    pub iterations: usize,
    /// Whether the solver converged to within tolerances.
    pub converged: bool,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn project(x: &mut [f64], lower: &[f64], upper: &[f64]) {
    for ((xi, &lb), &ub) in x.iter_mut().zip(lower.iter()).zip(upper.iter()) {
        *xi = xi.max(lb).min(ub);
    }
}

fn eq_violation(c_eq: &[f64]) -> f64 {
    c_eq.iter().fold(0.0_f64, |m, &c| m.max(c.abs()))
}

fn ineq_violation(c_ineq: &[f64]) -> f64 {
    c_ineq.iter().fold(0.0_f64, |m, &c| m.max(c)).max(0.0)
}

/// Solve `(JᵀJ + reg·I)·x = rhs` matrix-free via conjugate gradient.
///
/// The system operator is applied as `Jᵀ(J·v) + reg·v`; `J` is transposed once
/// up front so each iteration is two sparse mat-vec products.
fn conjugate_gradient(
    j_eq: &CsrMatrix,
    rhs: &[f64],
    reg: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let jt = j_eq.transpose();
    let apply = |v: &[f64]| -> Vec<f64> {
        let jv = j_eq.mul_vec(v);
        let mut out = jt.mul_vec(&jv);
        for (o, &vi) in out.iter_mut().zip(v.iter()) {
            *o += reg * vi;
        }
        out
    };

    let n = rhs.len();
    let mut x = vec![0.0; n];
    let mut r = rhs.to_vec(); // r = rhs - A·0
    let mut p = r.clone();
    let rhs_norm = norm(rhs).max(1e-30);
    let mut rs_old = dot(&r, &r);

    for _ in 0..max_iter {
        let ap = apply(&p);
        let alpha = rs_old / (dot(&p, &ap) + 1e-30);
        for (xi, &pi) in x.iter_mut().zip(p.iter()) {
            *xi += alpha * pi;
        }
        for (ri, &api) in r.iter_mut().zip(ap.iter()) {
            *ri -= alpha * api;
        }
        if norm(&r) / rhs_norm < tol {
            break;
        }
        let rs_new = dot(&r, &r);
        let beta = rs_new / (rs_old + 1e-30);
        for (pi, &ri) in p.iter_mut().zip(r.iter()) {
            *pi = ri + beta * *pi;
        }
        rs_old = rs_new;
    }
    x
}

/// Merit function: feasibility (squared) plus a small objective pull.
fn merit(eval: &impl NlpEvaluator, x: &[f64]) -> f64 {
    let c_eq = eval.equality_constraints(x);
    let c_ineq = eval.inequality_constraints(x);
    let eq_sq: f64 = c_eq.iter().map(|&c| c * c).sum();
    let ineq_sq: f64 = c_ineq
        .iter()
        .map(|&c| {
            let v = c.max(0.0);
            v * v
        })
        .sum();
    eq_sq + ineq_sq + 0.01 * eval.objective(x)
}

/// Solve an NLP using a damped Gauss-Newton method with merit line search.
pub fn solve_gauss_newton(
    problem: &NlpProblem,
    evaluator: &impl NlpEvaluator,
    x0: &[f64],
    config: &GaussNewtonConfig,
) -> GaussNewtonResult {
    let lower = &problem.lower_bounds;
    let upper = &problem.upper_bounds;

    let mut x = x0.to_vec();
    project(&mut x, lower, upper);

    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        let c_eq = evaluator.equality_constraints(&x);
        let c_ineq = evaluator.inequality_constraints(&x);
        let j_eq = evaluator.equality_jacobian(&x);
        let obj_grad = evaluator.objective_gradient(&x);

        let ev = eq_violation(&c_eq);
        let iv = ineq_violation(&c_ineq);
        let feasible = ev < config.constraint_tol && iv < config.constraint_tol;

        // --- assemble the normal-equation right-hand side ---
        // rhs = -Jᵀ·c_eq - (obj_weight/2)·∇f - penalty·J_ineqᵀ·max(0, c_ineq)
        let jt = j_eq.transpose();
        let mut rhs = jt.mul_vec(&c_eq);
        for r in rhs.iter_mut() {
            *r = -*r;
        }

        // Objective pull: feasibility-gated so it vanishes while constraints are
        // badly violated (prioritize feasibility) and grows back to the base
        // weight as the problem becomes feasible. Solving through
        // (JᵀJ + reg·I)⁻¹ multiplies a null-space term by ~1/reg, so the base
        // weight is set to `reg` to keep the implied step bounded.
        let c_eq_norm_sq: f64 = c_eq.iter().map(|&c| c * c).sum();
        let obj_weight = config.regularization / (1.0 + c_eq_norm_sq);
        for (r, &g) in rhs.iter_mut().zip(obj_grad.iter()) {
            *r -= 0.5 * obj_weight * g;
        }

        // Inequality penalty: a gentle push to reduce violations. Only applied
        // once the equalities are nearly satisfied — its rhs contribution has a
        // component in the null space of J_eq, which the (JᵀJ+reg·I)⁻¹ solve
        // amplifies by ~1/reg and would otherwise swamp the feasibility step.
        if problem.n_ineq > 0 && ev < 1.0 {
            let j_ineq = evaluator.inequality_jacobian(&x);
            let penalty_w = 1.0;
            let g_pos: Vec<f64> = c_ineq.iter().map(|&c| c.max(0.0)).collect();
            let pen = j_ineq.transpose().mul_vec(&g_pos);
            for (r, &p) in rhs.iter_mut().zip(pen.iter()) {
                *r -= penalty_w * p;
            }
        }

        // --- Gauss-Newton step ---
        let delta = conjugate_gradient(&j_eq, &rhs, config.regularization, 100, 1e-8);
        let step_rel = norm(&delta) / (norm(&x) + 1.0);

        if config.print_interval > 0 && iter % config.print_interval == 0 {
            println!(
                "gn {:4} | obj {:.6} | eq_viol {:.3e} | ineq_viol {:.3e} | step {:.3e}",
                iter,
                evaluator.objective(&x),
                ev,
                iv,
                step_rel
            );
        }

        // converged: feasible and no longer moving
        if feasible && step_rel < config.step_tol {
            converged = true;
            break;
        }

        // --- damped backtracking line search on the merit function ---
        let merit0 = merit(evaluator, &x);
        let mut step = config.damping;
        let mut x_new = x.clone();
        let mut accepted = false;
        loop {
            for (xn, (&xi, &di)) in x_new.iter_mut().zip(x.iter().zip(delta.iter())) {
                *xn = xi + step * di;
            }
            project(&mut x_new, lower, upper);
            if merit(evaluator, &x_new) < merit0 {
                accepted = true;
                break;
            }
            step *= 0.5;
            if step < 1e-12 {
                break;
            }
        }

        if !accepted {
            // stalled — converged iff currently feasible
            converged = feasible;
            break;
        }
        x = x_new;
    }

    let c_eq = evaluator.equality_constraints(&x);
    let c_ineq = evaluator.inequality_constraints(&x);
    let ev = eq_violation(&c_eq);
    let iv = ineq_violation(&c_ineq);
    if !converged {
        converged = ev < config.constraint_tol && iv < config.constraint_tol;
    }

    GaussNewtonResult {
        objective: evaluator.objective(&x),
        eq_violation: ev,
        ineq_violation: iv,
        iterations,
        converged,
        x,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_math::{CsrBuilder, CsrMatrix, Mat3, Vec3};

    #[test]
    fn cg_solves_known_system() {
        // J = [[2,1,0],[0,2,1],[1,0,2]], reg = 0.1
        let mut b = CsrBuilder::new(3, 3);
        b.add(0, 0, 2.0);
        b.add(0, 1, 1.0);
        b.add(1, 1, 2.0);
        b.add(1, 2, 1.0);
        b.add(2, 0, 1.0);
        b.add(2, 2, 2.0);
        let j: CsrMatrix = b.build();
        let reg = 0.1;

        // Reference: A = JᵀJ + reg·I, solve A·x = rhs directly via Mat3.
        let jd = j.to_dense();
        let mut a = [[0.0; 3]; 3];
        for (i, ai) in a.iter_mut().enumerate() {
            for (jj, aij) in ai.iter_mut().enumerate() {
                let mut s = 0.0;
                for row in &jd {
                    s += row[i] * row[jj];
                }
                *aij = s + if i == jj { reg } else { 0.0 };
            }
        }
        let a_mat = Mat3::from_rows(
            Vec3::new(a[0][0], a[0][1], a[0][2]),
            Vec3::new(a[1][0], a[1][1], a[1][2]),
            Vec3::new(a[2][0], a[2][1], a[2][2]),
        );
        let rhs = [1.0, 2.0, 3.0];
        let x_ref = a_mat.inverse().expect("PD") * Vec3::new(rhs[0], rhs[1], rhs[2]);

        let x = conjugate_gradient(&j, &rhs, reg, 100, 1e-12);
        assert!((x[0] - x_ref.x).abs() < 1e-6, "x0 {} vs {}", x[0], x_ref.x);
        assert!((x[1] - x_ref.y).abs() < 1e-6, "x1 {} vs {}", x[1], x_ref.y);
        assert!((x[2] - x_ref.z).abs() < 1e-6, "x2 {} vs {}", x[2], x_ref.z);
    }

    /// f(x) = (x0-3)² + (x1-5)², unconstrained.
    struct Quadratic;
    impl NlpEvaluator for Quadratic {
        fn objective(&self, x: &[f64]) -> f64 {
            (x[0] - 3.0).powi(2) + (x[1] - 5.0).powi(2)
        }
        fn objective_gradient(&self, x: &[f64]) -> Vec<f64> {
            vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] - 5.0)]
        }
        fn equality_constraints(&self, _x: &[f64]) -> Vec<f64> {
            vec![]
        }
        fn inequality_constraints(&self, _x: &[f64]) -> Vec<f64> {
            vec![]
        }
        fn equality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            CsrMatrix::zeros(0, 2)
        }
        fn inequality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            CsrMatrix::zeros(0, 2)
        }
    }

    #[test]
    fn gn_unconstrained_quadratic() {
        let problem = NlpProblem {
            n_vars: 2,
            n_eq: 0,
            n_ineq: 0,
            lower_bounds: vec![f64::NEG_INFINITY; 2],
            upper_bounds: vec![f64::INFINITY; 2],
        };
        let result = solve_gauss_newton(
            &problem,
            &Quadratic,
            &[0.0, 0.0],
            &GaussNewtonConfig::default(),
        );
        assert!((result.x[0] - 3.0).abs() < 1e-2, "x0 {}", result.x[0]);
        assert!((result.x[1] - 5.0).abs() < 1e-2, "x1 {}", result.x[1]);
        assert!(result.converged);
    }

    #[test]
    fn gn_circle_collocation() {
        use crate::collocation::{CollocationConfig, CollocationOptimizer};
        use apex_physics::{qss_lap_sim, CarParams};
        use apex_track::{build_track, circle_track};

        let (pts, closed) = circle_track(100.0, 12.0, 200);
        let track = build_track("circle", &pts, closed);
        let car = CarParams::default();
        let qss_lap = qss_lap_sim(&track, &car).lap_time;

        let config = CollocationConfig {
            n_nodes: 30,
            closed: true,
            ..CollocationConfig::default()
        };
        let opt = CollocationOptimizer::new(config, &track, &car);
        let result = opt.optimize_gn(&GaussNewtonConfig::default());

        assert!(result.converged, "GN should converge on the circle");
        assert!(
            (result.lap_time - qss_lap).abs() / qss_lap < 0.02,
            "lap time {} vs QSS {}",
            result.lap_time,
            qss_lap
        );
    }

    #[test]
    fn gn_oval_collocation() {
        use crate::collocation::{CollocationConfig, CollocationOptimizer};
        use apex_physics::CarParams;
        use apex_track::{build_track, oval_track};

        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 400);
        let track = build_track("oval", &pts, closed);
        let car = CarParams::default();

        let config = CollocationConfig {
            n_nodes: 50,
            closed: true,
            ..CollocationConfig::default()
        };
        let opt = CollocationOptimizer::new(config, &track, &car);

        let gn_config = GaussNewtonConfig {
            max_iterations: 50,
            constraint_tol: 1e-3,
            ..GaussNewtonConfig::default()
        };
        let result = opt.optimize_gn(&gn_config);

        // all outputs finite (eq_violation finiteness is implied by a finite x)
        assert!(result.lap_time.is_finite(), "lap time not finite");
        for &v in &result.speeds {
            assert!(v.is_finite(), "speed not finite");
        }
        for &nn in &result.offsets {
            assert!(nn.is_finite(), "offset not finite");
        }
        for &a in &result.headings {
            assert!(a.is_finite(), "heading not finite");
        }
    }
}
