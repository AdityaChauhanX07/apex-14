#![deny(unsafe_code)]
//! Vehicle dynamics models for Apex-14, from a 2-DOF point mass upward.

pub mod aero;
pub mod bicycle;
pub mod car_config;
pub mod car_params;
pub mod controller;
pub mod drivetrain;
pub mod fourteen_dof;
pub mod point_mass;
pub mod qss;
pub mod sensitivity;
pub mod seven_dof;
pub mod strategy;
pub mod suspension;
pub mod tire;

pub use aero::{AeroForces, AeroModel};
pub use bicycle::BicycleModel;
pub use car_config::{
    export_car_toml, load_car_toml, parse_car_toml, AeroSection, CarConfig, CarSection,
    GeometrySection, PowertrainSection, SuspensionSection, TireSection,
};
pub use car_params::CarParams;
pub use controller::{solve_care_4x4, LqrController, SpeedController};
pub use drivetrain::{Engine, Gearbox, Powertrain};
pub use fourteen_dof::FourteenDofModel;
pub use point_mass::PointMassModel;
pub use qss::{qss_lap_sim, qss_lap_sim_tire, QssResult};
pub use sensitivity::{
    f1_parameter_set, monte_carlo_sensitivity, oat_sensitivity, tornado_chart_svg,
    MonteCarloResult, OatResult, ParameterDef,
};
pub use seven_dof::SevenDofModel;
pub use strategy::{
    FuelModel, RaceStrategy, Stint, StrategyEvaluator, StrategyResult, TireCompound, UndercutResult,
};
pub use suspension::{AntiRollBar, SuspensionParams, SuspensionSystem};
pub use tire::{smooth_min, CombinedSlipResult, PacejkaCoeffs, PacejkaTire};
