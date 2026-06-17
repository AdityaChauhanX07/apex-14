#![deny(unsafe_code)]
//! Trajectory optimization for Apex-14: NLP problem definition and an
//! Augmented Lagrangian solver.

pub mod collocation;
pub mod direct_solver;
pub mod forward_sim;
pub mod gauss_newton;
pub mod mesh_refinement;
pub mod nlp;
pub mod solver;

pub use collocation::{
    fourteen_dof_grip_budget, seven_dof_derivatives, tire_limited_forces, CollocationConfig,
    CollocationOptimizer, OptimizationResult,
};
pub use mesh_refinement::{
    optimize_with_refinement, LevelResult, MeshRefinementConfig, RefinedResult,
};
pub use direct_solver::{
    solve_direct, CollocationStructure, DirectSolverConfig, DirectSolverResult,
};
pub use forward_sim::{DetailedTelemetry, ForwardSimulator};
pub use gauss_newton::{solve_gauss_newton, GaussNewtonConfig, GaussNewtonResult};
pub use nlp::{NlpEvaluator, NlpProblem};
pub use solver::{solve_nlp, SolverConfig, SolverResult};
