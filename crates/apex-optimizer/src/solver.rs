//! Augmented Lagrangian NLP solver with projected gradient descent.

use crate::nlp::{NlpEvaluator, NlpProblem};

/// Configuration for the NLP solver.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum number of outer (Augmented Lagrangian) iterations.
    pub max_outer_iter: usize,
    /// Maximum number of inner (gradient descent) iterations per outer step.
    pub max_inner_iter: usize,
    /// Convergence tolerance on constraint violation (L2 norm).
    pub constraint_tol: f64,
    /// Convergence tolerance on objective improvement (relative).
    pub objective_tol: f64,
    /// Initial penalty parameter for constraint violations.
    pub initial_penalty: f64,
    /// Penalty growth factor per outer iteration.
    pub penalty_growth: f64,
    /// Maximum penalty parameter (cap to avoid numerical issues).
    pub max_penalty: f64,
    /// Step size for inner gradient descent (will use backtracking line search).
    pub initial_step_size: f64,
    /// Line search contraction factor (Armijo backtracking).
    pub line_search_beta: f64,
    /// Armijo sufficient decrease parameter.
    pub line_search_c: f64,
    /// Print progress every N outer iterations (0 = silent).
    pub print_interval: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            max_outer_iter: 100,
            max_inner_iter: 500,
            constraint_tol: 1e-4,
            objective_tol: 1e-6,
            initial_penalty: 1.0,
            penalty_growth: 2.0,
            max_penalty: 1e6,
            initial_step_size: 0.01,
            line_search_beta: 0.5,
            line_search_c: 1e-4,
            print_interval: 0,
        }
    }
}

/// Result of the NLP solve.
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Final objective value.
    pub objective: f64,
    /// Maximum equality constraint violation.
    pub eq_violation: f64,
    /// Maximum inequality constraint violation (positive = violated).
    pub ineq_violation: f64,
    /// Number of outer iterations performed.
    pub outer_iterations: usize,
    /// Total number of inner iterations.
    pub total_inner_iterations: usize,
    /// Whether the solver converged to within tolerances.
    pub converged: bool,
}

/// Clamp `x` element-wise to the problem's variable bounds.
fn project(x: &mut [f64], lower: &[f64], upper: &[f64]) {
    for ((xi, &lb), &ub) in x.iter_mut().zip(lower.iter()).zip(upper.iter()) {
        *xi = xi.max(lb).min(ub);
    }
}

/// Value of the augmented Lagrangian at `x`.
fn aug_lag_value(
    eval: &impl NlpEvaluator,
    x: &[f64],
    lambda_eq: &[f64],
    lambda_ineq: &[f64],
    mu: f64,
) -> f64 {
    let mut v = eval.objective(x);

    let c_eq = eval.equality_constraints(x);
    for (&l, &c) in lambda_eq.iter().zip(c_eq.iter()) {
        v += l * c + 0.5 * mu * c * c;
    }

    let c_ineq = eval.inequality_constraints(x);
    for (&l, &c) in lambda_ineq.iter().zip(c_ineq.iter()) {
        let s = (l + mu * c).max(0.0);
        v += (s * s - l * l) / (2.0 * mu);
    }
    v
}

/// Gradient of the augmented Lagrangian at `x`.
fn aug_lag_grad(
    eval: &impl NlpEvaluator,
    x: &[f64],
    lambda_eq: &[f64],
    lambda_ineq: &[f64],
    mu: f64,
) -> Vec<f64> {
    let mut grad = eval.objective_gradient(x);

    // equality contribution: J_eq^T · (λ_eq + μ·c_eq)
    let c_eq = eval.equality_constraints(x);
    let v_eq: Vec<f64> = lambda_eq
        .iter()
        .zip(c_eq.iter())
        .map(|(&l, &c)| l + mu * c)
        .collect();
    let eq_contrib = eval.equality_jacobian(x).transpose().mul_vec(&v_eq);

    // inequality contribution: J_ineq^T · max(0, λ_ineq + μ·c_ineq)
    let c_ineq = eval.inequality_constraints(x);
    let adj: Vec<f64> = lambda_ineq
        .iter()
        .zip(c_ineq.iter())
        .map(|(&l, &c)| (l + mu * c).max(0.0))
        .collect();
    let ineq_contrib = eval.inequality_jacobian(x).transpose().mul_vec(&adj);

    for (g, (&e, &n)) in grad
        .iter_mut()
        .zip(eq_contrib.iter().zip(ineq_contrib.iter()))
    {
        *g += e + n;
    }
    grad
}

/// Maximum absolute equality violation.
fn eq_violation(c_eq: &[f64]) -> f64 {
    c_eq.iter().fold(0.0_f64, |m, &c| m.max(c.abs()))
}

/// Maximum inequality violation (positive part).
fn ineq_violation(c_ineq: &[f64]) -> f64 {
    c_ineq.iter().fold(0.0_f64, |m, &c| m.max(c)).max(0.0)
}

/// Solve an NLP using the Augmented Lagrangian method.
///
/// The Augmented Lagrangian transforms the constrained problem into a sequence
/// of unconstrained (or bound-constrained) subproblems:
///
/// L_A(x, λ, μ) = f(x) + λᵀc(x) + (μ/2)||c(x)||²
///
/// where λ are Lagrange multiplier estimates and μ is the penalty parameter.
/// Each inner solve minimizes L_A for fixed λ, μ using projected gradient
/// descent (projecting onto the variable bounds). After each inner solve,
/// λ is updated and μ is increased.
pub fn solve_nlp(
    problem: &NlpProblem,
    evaluator: &impl NlpEvaluator,
    x0: &[f64],
    config: &SolverConfig,
) -> SolverResult {
    let lower = &problem.lower_bounds;
    let upper = &problem.upper_bounds;

    let mut x = x0.to_vec();
    project(&mut x, lower, upper);

    let mut lambda_eq = vec![0.0; problem.n_eq];
    let mut lambda_ineq = vec![0.0; problem.n_ineq];
    let mut mu = config.initial_penalty;

    let mut total_inner = 0usize;
    let mut outer_done = 0usize;
    let mut converged = false;

    for outer in 0..config.max_outer_iter {
        outer_done = outer + 1;

        // --- inner loop: minimize L_A for fixed (λ, μ) ---
        for _ in 0..config.max_inner_iter {
            let grad = aug_lag_grad(evaluator, &x, &lambda_eq, &lambda_ineq, mu);
            let grad_norm_sq: f64 = grad.iter().map(|g| g * g).sum();
            if grad_norm_sq < 1e-16 {
                break;
            }

            let l0 = aug_lag_value(evaluator, &x, &lambda_eq, &lambda_ineq, mu);

            // Armijo backtracking line search along -grad.
            let mut step = config.initial_step_size;
            let mut accepted = false;
            let mut x_trial = x.clone();
            loop {
                for (xt, (&xi, &gi)) in x_trial.iter_mut().zip(x.iter().zip(grad.iter())) {
                    *xt = xi - step * gi;
                }
                let l_trial = aug_lag_value(evaluator, &x_trial, &lambda_eq, &lambda_ineq, mu);
                if l_trial <= l0 - config.line_search_c * step * grad_norm_sq {
                    accepted = true;
                    break;
                }
                step *= config.line_search_beta;
                if step < 1e-12 {
                    break;
                }
            }

            total_inner += 1;
            if !accepted {
                break;
            }
            // accept the (projected) step
            project(&mut x_trial, lower, upper);
            x = x_trial;
        }

        // --- multiplier and penalty updates ---
        let c_eq = evaluator.equality_constraints(&x);
        let c_ineq = evaluator.inequality_constraints(&x);

        for (l, &c) in lambda_eq.iter_mut().zip(c_eq.iter()) {
            *l += mu * c;
        }
        for (l, &c) in lambda_ineq.iter_mut().zip(c_ineq.iter()) {
            *l = (*l + mu * c).max(0.0);
        }
        mu = (mu * config.penalty_growth).min(config.max_penalty);

        // --- convergence check ---
        let eq_viol = eq_violation(&c_eq);
        let ineq_viol = ineq_violation(&c_ineq);

        if config.print_interval > 0 && outer % config.print_interval == 0 {
            println!(
                "outer {:4} | obj {:.6} | eq_viol {:.3e} | ineq_viol {:.3e} | mu {:.2e}",
                outer,
                evaluator.objective(&x),
                eq_viol,
                ineq_viol,
                mu
            );
        }

        if eq_viol < config.constraint_tol && ineq_viol < config.constraint_tol {
            converged = true;
            break;
        }
    }

    let c_eq = evaluator.equality_constraints(&x);
    let c_ineq = evaluator.inequality_constraints(&x);

    SolverResult {
        objective: evaluator.objective(&x),
        eq_violation: eq_violation(&c_eq),
        ineq_violation: ineq_violation(&c_ineq),
        outer_iterations: outer_done,
        total_inner_iterations: total_inner,
        converged,
        x,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_math::{CsrBuilder, CsrMatrix};

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

    /// f(x) = x0² + x1²  s.t.  x0 + x1 = 1.
    struct EqConstrained;
    impl NlpEvaluator for EqConstrained {
        fn objective(&self, x: &[f64]) -> f64 {
            x[0] * x[0] + x[1] * x[1]
        }
        fn objective_gradient(&self, x: &[f64]) -> Vec<f64> {
            vec![2.0 * x[0], 2.0 * x[1]]
        }
        fn equality_constraints(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] + x[1] - 1.0]
        }
        fn inequality_constraints(&self, _x: &[f64]) -> Vec<f64> {
            vec![]
        }
        fn equality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            let mut b = CsrBuilder::new(1, 2);
            b.add(0, 0, 1.0);
            b.add(0, 1, 1.0);
            b.build()
        }
        fn inequality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            CsrMatrix::zeros(0, 2)
        }
    }

    /// f(x) = (x0-2)² + (x1-2)²  s.t.  x0 + x1 <= 2.
    struct IneqConstrained;
    impl NlpEvaluator for IneqConstrained {
        fn objective(&self, x: &[f64]) -> f64 {
            (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2)
        }
        fn objective_gradient(&self, x: &[f64]) -> Vec<f64> {
            vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 2.0)]
        }
        fn equality_constraints(&self, _x: &[f64]) -> Vec<f64> {
            vec![]
        }
        fn inequality_constraints(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] + x[1] - 2.0]
        }
        fn equality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            CsrMatrix::zeros(0, 2)
        }
        fn inequality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            let mut b = CsrBuilder::new(1, 2);
            b.add(0, 0, 1.0);
            b.add(0, 1, 1.0);
            b.build()
        }
    }

    /// f(x) = x0², single variable, bounds applied externally.
    struct SingleSquare;
    impl NlpEvaluator for SingleSquare {
        fn objective(&self, x: &[f64]) -> f64 {
            x[0] * x[0]
        }
        fn objective_gradient(&self, x: &[f64]) -> Vec<f64> {
            vec![2.0 * x[0]]
        }
        fn equality_constraints(&self, _x: &[f64]) -> Vec<f64> {
            vec![]
        }
        fn inequality_constraints(&self, _x: &[f64]) -> Vec<f64> {
            vec![]
        }
        fn equality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            CsrMatrix::zeros(0, 1)
        }
        fn inequality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            CsrMatrix::zeros(0, 1)
        }
    }

    fn unbounded(n: usize) -> (Vec<f64>, Vec<f64>) {
        (vec![f64::NEG_INFINITY; n], vec![f64::INFINITY; n])
    }

    #[test]
    fn unconstrained_quadratic() {
        let (lb, ub) = unbounded(2);
        let problem = NlpProblem {
            n_vars: 2,
            n_eq: 0,
            n_ineq: 0,
            lower_bounds: lb,
            upper_bounds: ub,
        };
        let result = solve_nlp(&problem, &Quadratic, &[0.0, 0.0], &SolverConfig::default());
        assert!((result.x[0] - 3.0).abs() < 1e-3, "x0 {}", result.x[0]);
        assert!((result.x[1] - 5.0).abs() < 1e-3, "x1 {}", result.x[1]);
        assert!(result.converged);
    }

    #[test]
    fn equality_constrained() {
        let (lb, ub) = unbounded(2);
        let problem = NlpProblem {
            n_vars: 2,
            n_eq: 1,
            n_ineq: 0,
            lower_bounds: lb,
            upper_bounds: ub,
        };
        let result = solve_nlp(&problem, &EqConstrained, &[0.0, 0.0], &SolverConfig::default());
        assert!((result.x[0] - 0.5).abs() < 1e-2, "x0 {}", result.x[0]);
        assert!((result.x[1] - 0.5).abs() < 1e-2, "x1 {}", result.x[1]);
        assert!((result.objective - 0.5).abs() < 1e-2, "obj {}", result.objective);
        assert!(result.converged);
    }

    #[test]
    fn inequality_constrained() {
        let (lb, ub) = unbounded(2);
        let problem = NlpProblem {
            n_vars: 2,
            n_eq: 0,
            n_ineq: 1,
            lower_bounds: lb,
            upper_bounds: ub,
        };
        let result = solve_nlp(&problem, &IneqConstrained, &[0.0, 0.0], &SolverConfig::default());
        assert!((result.x[0] - 1.0).abs() < 1e-2, "x0 {}", result.x[0]);
        assert!((result.x[1] - 1.0).abs() < 1e-2, "x1 {}", result.x[1]);
        assert!(result.converged);
    }

    #[test]
    fn bounded_variable() {
        let problem = NlpProblem {
            n_vars: 1,
            n_eq: 0,
            n_ineq: 0,
            lower_bounds: vec![1.0],
            upper_bounds: vec![5.0],
        };
        let result = solve_nlp(&problem, &SingleSquare, &[3.0], &SolverConfig::default());
        assert!((result.x[0] - 1.0).abs() < 1e-3, "x0 {}", result.x[0]);
        assert!(result.converged);
    }

    #[test]
    fn convergence_flags() {
        // unconstrained converges
        let (lb, ub) = unbounded(2);
        let problem = NlpProblem {
            n_vars: 2,
            n_eq: 0,
            n_ineq: 0,
            lower_bounds: lb,
            upper_bounds: ub,
        };
        let ok = solve_nlp(&problem, &Quadratic, &[0.0, 0.0], &SolverConfig::default());
        assert!(ok.converged);

        // tight tolerance + tiny iteration budget on a constrained problem fails
        let (lb, ub) = unbounded(2);
        let eq_problem = NlpProblem {
            n_vars: 2,
            n_eq: 1,
            n_ineq: 0,
            lower_bounds: lb,
            upper_bounds: ub,
        };
        let config = SolverConfig {
            max_outer_iter: 2,
            constraint_tol: 1e-15,
            ..SolverConfig::default()
        };
        let fail = solve_nlp(&eq_problem, &EqConstrained, &[0.0, 0.0], &config);
        assert!(!fail.converged);
    }
}
