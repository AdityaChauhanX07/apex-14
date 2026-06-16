//! The [`OdeSystem`] trait describing a system of ordinary differential
//! equations with `N` state variables and `M` control inputs.

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
