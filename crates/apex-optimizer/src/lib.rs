//! Trajectory optimization for Apex-14: NLP problem definition and an
//! Augmented Lagrangian solver.

pub mod collocation;
pub mod direct_solver;
pub mod gauss_newton;
pub mod nlp;
pub mod solver;

pub use collocation::{CollocationConfig, CollocationOptimizer, OptimizationResult};
pub use direct_solver::{
    solve_direct, CollocationStructure, DirectSolverConfig, DirectSolverResult,
};
pub use gauss_newton::{solve_gauss_newton, GaussNewtonConfig, GaussNewtonResult};
pub use nlp::{NlpEvaluator, NlpProblem};
pub use solver::{solve_nlp, SolverConfig, SolverResult};
