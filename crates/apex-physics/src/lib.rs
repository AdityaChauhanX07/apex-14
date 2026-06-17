#![deny(unsafe_code)]
//! Vehicle dynamics models for Apex-14, from a 2-DOF point mass upward.

pub mod aero;
pub mod bicycle;
pub mod car_params;
pub mod fourteen_dof;
pub mod point_mass;
pub mod qss;
pub mod seven_dof;
pub mod suspension;
pub mod tire;

pub use aero::{AeroForces, AeroModel};
pub use bicycle::BicycleModel;
pub use car_params::CarParams;
pub use fourteen_dof::FourteenDofModel;
pub use point_mass::PointMassModel;
pub use qss::{qss_lap_sim, qss_lap_sim_tire, QssResult};
pub use seven_dof::SevenDofModel;
pub use suspension::{AntiRollBar, SuspensionParams, SuspensionSystem};
pub use tire::{smooth_min, CombinedSlipResult, PacejkaCoeffs, PacejkaTire};
