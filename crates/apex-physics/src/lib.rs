#![deny(unsafe_code)]
//! Vehicle dynamics models for Apex-14, from a 2-DOF point mass upward.

pub mod aero;
pub mod bicycle;
pub mod car_config;
pub mod car_params;
pub mod controller;
pub mod drivetrain;
pub mod fourteen_dof;
pub mod grip_map;
pub mod point_mass;
pub mod qss;
pub mod sensitivity;
pub mod seven_dof;
pub mod strategy;
pub mod suspension;
pub mod tire;
pub mod wake;
pub mod weather;

pub use aero::{aero_hash, AeroForces, AeroModel};
pub use bicycle::BicycleModel;
pub use car_config::{
    export_car_toml, load_car_toml, parse_car_toml, AeroSection, CarConfig, CarSection,
    GeometrySection, PowertrainSection, SuspensionSection, TireSection,
};
pub use car_params::{car_params_hash, CarParams};
pub use controller::{solve_care_4x4, LqrController, SpeedController};
pub use drivetrain::{Engine, Gearbox, Powertrain};
pub use fourteen_dof::FourteenDofModel;
pub use grip_map::GripMap;
pub use point_mass::PointMassModel;
pub use qss::{
    qss_lap_sim, qss_lap_sim_3d, qss_lap_sim_3d_with_grip, qss_lap_sim_tire, sector_times,
    sector_times_with_markers, QssResult, DEFAULT_SECTOR_COUNT,
};
pub use sensitivity::{
    f1_parameter_set, monte_carlo_sensitivity, oat_sensitivity, tornado_chart_svg,
    MonteCarloResult, OatResult, ParameterDef,
};
pub use seven_dof::SevenDofModel;
pub use strategy::{
    FuelModel, RaceStrategy, Stint, StrategyEvaluator, StrategyResult, TireCompound, UndercutResult,
};
pub use suspension::{suspension_hash, AntiRollBar, SuspensionParams, SuspensionSystem};
pub use tire::{pacejka_tire_hash, smooth_min, CombinedSlipResult, PacejkaCoeffs, PacejkaTire};
pub use wake::{MultiCarState, OnTrackCar, WakeModel};
pub use weather::{
    analyze_tire_change, effective_grip, TireChangeAnalysis, TireType, WeatherState,
};
