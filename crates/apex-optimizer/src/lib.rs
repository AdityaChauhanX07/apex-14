//! Trajectory optimization for Apex-14: NLP problem definition and an
//! Augmented Lagrangian solver.

pub mod nlp;
pub mod solver;

pub use nlp::{NlpEvaluator, NlpProblem};
pub use solver::{solve_nlp, SolverConfig, SolverResult};
