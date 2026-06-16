//! Nonlinear programming (NLP) problem definition.

/// A general-purpose nonlinear programming problem definition.
///
/// Minimize f(x)
/// Subject to:
///   c_eq(x) = 0        (equality constraints)
///   c_ineq(x) <= 0     (inequality constraints)
///   lb <= x <= ub       (variable bounds)
#[derive(Debug, Clone)]
pub struct NlpProblem {
    /// Number of decision variables.
    pub n_vars: usize,
    /// Number of equality constraints.
    pub n_eq: usize,
    /// Number of inequality constraints.
    pub n_ineq: usize,
    /// Lower bounds on decision variables (length n_vars). Use f64::NEG_INFINITY for unbounded.
    pub lower_bounds: Vec<f64>,
    /// Upper bounds on decision variables (length n_vars). Use f64::INFINITY for unbounded.
    pub upper_bounds: Vec<f64>,
}

/// Trait that the user implements to define their specific NLP.
pub trait NlpEvaluator {
    /// Evaluate the objective function f(x).
    fn objective(&self, x: &[f64]) -> f64;

    /// Evaluate the gradient of the objective function ∇f(x).
    fn objective_gradient(&self, x: &[f64]) -> Vec<f64>;

    /// Evaluate all equality constraints c_eq(x). Returns a vector of length n_eq.
    /// The solver will enforce c_eq(x) = 0.
    fn equality_constraints(&self, x: &[f64]) -> Vec<f64>;

    /// Evaluate all inequality constraints c_ineq(x). Returns a vector of length n_ineq.
    /// The solver will enforce c_ineq(x) <= 0.
    fn inequality_constraints(&self, x: &[f64]) -> Vec<f64>;

    /// Evaluate the Jacobian of the equality constraints.
    /// Returns a CsrMatrix of shape (n_eq, n_vars).
    fn equality_jacobian(&self, x: &[f64]) -> apex_math::CsrMatrix;

    /// Evaluate the Jacobian of the inequality constraints.
    /// Returns a CsrMatrix of shape (n_ineq, n_vars).
    fn inequality_jacobian(&self, x: &[f64]) -> apex_math::CsrMatrix;
}
