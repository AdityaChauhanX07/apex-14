#![deny(unsafe_code)]
//! ODE integration methods for Apex-14: fixed-step and adaptive Runge-Kutta
//! solvers.

pub mod rk4;
pub mod rk45;
pub mod traits;

pub use rk4::{rk4_integrate, rk4_integrate_generic, rk4_step, rk4_step_generic};
pub use rk45::{rk45_adaptive_step, rk45_integrate, rk45_step, AdaptiveConfig, AdaptiveStepResult};
pub use traits::{OdeSystem, OdeSystemGeneric};
