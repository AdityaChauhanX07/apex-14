//! Nonlinear programming (NLP) problem definition.

use crate::precond::BlockStructure;

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

    /// Hessian-vector product of the **objective** at `x`: returns `H_f(x) · v`.
    ///
    /// Default: the zero vector — appropriate for a linear objective, or when
    /// the objective's curvature is modeled elsewhere (the collocation lap-time
    /// objective is linear; its solver models constraint curvature by a
    /// Gauss-Newton term, not an objective Hessian). A genuinely nonlinear
    /// objective (e.g. a QP `½xᵀQx` or a least-squares residual) should override
    /// this so the interior-point solver ([`crate::ipm`]) has second-order
    /// objective information; the product form keeps it matrix-free (no Hessian
    /// is ever assembled). It is fine to return a Gauss-Newton (positive
    /// semidefinite) approximation instead of the exact Hessian.
    fn objective_hessian_vec(&self, x: &[f64], v: &[f64]) -> Vec<f64> {
        let _ = x;
        vec![0.0; v.len()]
    }

    /// Node-contiguous grouping of the decision variables, when the problem has
    /// collocation structure.
    ///
    /// Default `None` — the problem has no exploitable block structure and any
    /// block-structured preconditioner falls back to Jacobi. A collocation NLP
    /// should return one block per mesh node listing that node's variables (see
    /// [`BlockStructure::strided`] for the common
    /// block-contiguous-by-quantity layout), which lets
    /// [`crate::precond::BlockTridiag`] invert the graph-Laplacian structure the
    /// scalar Jacobi preconditioner cannot see.
    ///
    /// This is **structural metadata only**: it describes an index grouping, not
    /// a reordering. The solver never permutes the decision vector, so returning
    /// a structure cannot change results on the default (Jacobi) path.
    fn block_structure(&self) -> Option<BlockStructure> {
        None
    }
}
