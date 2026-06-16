//! The [`OdeSystem`] trait describing a system of ordinary differential
//! equations with `N` state variables and `M` control inputs.

use apex_math::Float;

/// A system of ordinary differential equations.
///
/// The type parameters fix the dimensions: `N` is the number of state
/// variables and `M` the number of control inputs.
pub trait OdeSystem<const N: usize, const M: usize> {
    /// Compute the time derivatives of the state vector.
    ///
    /// * `state` — current state (`N` elements)
    /// * `control` — current control input (`M` elements)
    /// * `t` — current time
    ///
    /// Returns `dstate/dt` (`N` elements).
    fn derivatives(&self, state: &[f64; N], control: &[f64; M], t: f64) -> [f64; N];
}

/// Generic ODE system trait for use with automatic differentiation.
///
/// Same as [`OdeSystem`] but operates on generic [`Float`] types, allowing
/// the same physics code to compute both state derivatives (with `f64`)
/// and Jacobians (with `Dual`).
pub trait OdeSystemGeneric<T: Float, const N: usize, const M: usize> {
    /// Compute the time derivatives of the state vector in a generic numeric type.
    fn derivatives_generic(&self, state: &[T; N], control: &[T; M], t: T) -> [T; N];
}
