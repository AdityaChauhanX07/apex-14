#![deny(unsafe_code)]
//! Trajectory optimization for Apex-14: NLP problem definition and an
//! Augmented Lagrangian solver.

pub mod cmaes;
pub mod collocation;
pub mod direct_solver;
pub mod forward_sim;
pub mod gauss_newton;
pub mod layout_optimizer;
pub mod mesh_refinement;
pub mod nlp;
pub mod overtaking;
pub mod racing_quality;
pub mod scaling;
pub mod settings_hash;
pub mod setup;
pub mod setup_eval;
pub mod solver;

pub use cmaes::{CmaEs, CmaEsConfig};
pub use collocation::{
    fourteen_dof_grip_budget, seven_dof_derivatives, tire_limited_forces, CollocationConfig,
    CollocationMethod, CollocationOptimizer, OptimizationResult,
};
pub use direct_solver::{
    solve_direct, CollocationStructure, DirectSolverConfig, DirectSolverResult,
};
pub use forward_sim::{DetailedTelemetry, ForwardSimulator};
pub use gauss_newton::{solve_gauss_newton, GaussNewtonConfig, GaussNewtonResult};
pub use layout_optimizer::{optimize_layout, LayoutOptConfig, LayoutOptResult};
pub use mesh_refinement::{
    optimize_with_refinement, LevelResult, MeshRefinementConfig, RefinedResult,
};
pub use nlp::{NlpEvaluator, NlpProblem};
pub use overtaking::{optimize_overtaking, LeaderTrajectory, OvertakingConfig, OvertakingResult};
pub use racing_quality::{compute_racing_quality, RacingQuality};
pub use scaling::{ScaledEvaluator, Scaling};
pub use settings_hash::{
    al_solver_settings_hash, cmaes_settings_hash, direct_solver_settings_hash,
    gauss_newton_settings_hash, optimize_fourteen_dof_settings_hash, optimize_gn_settings_hash,
};
pub use setup::{SetupParam, SetupSpace};
pub use setup_eval::{
    evaluate_batch, evaluate_setup, export_setup_toml, optimize_setup, GenerationRecord,
    SetupEvalConfig, SetupOptResult,
};
pub use solver::{solve_nlp, SolverConfig, SolverResult};
