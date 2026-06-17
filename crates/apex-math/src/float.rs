//! The [`Float`] trait: a shared numeric interface over `f64` and [`Dual`].
//!
//! Physics code is written generically over `T: Float` so the same equations
//! drive plain `f64` simulation and [`Dual`]-based automatic differentiation.

use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::dual::Dual;

/// A numeric type supporting the arithmetic and elementary functions used by
/// the physics equations.
///
/// The mixed `f64` operator bounds (`Add<f64, Output = Self>`, etc.) let
/// generic code write expressions like `x + 1.0` and `x * 2.0` directly. The
/// `*_f64` provided methods offer the same thing by name when inference needs a
/// nudge.
pub trait Float:
    Sized
    + Copy
    + Clone
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    /// The additive identity.
    fn zero() -> Self;

    /// The multiplicative identity.
    fn one() -> Self;

    /// Lifts a constant `f64` into this type (zero derivative for [`Dual`]).
    fn from_f64(val: f64) -> Self;

    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, p: f64) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn recip(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;

    /// Returns the underlying real value, for branching in physics code.
    fn real_value(&self) -> f64;

    /// `self + val`, lifting `val` through [`from_f64`](Float::from_f64).
    fn add_f64(self, val: f64) -> Self {
        self + Self::from_f64(val)
    }

    /// `self - val`, lifting `val` through [`from_f64`](Float::from_f64).
    fn sub_f64(self, val: f64) -> Self {
        self - Self::from_f64(val)
    }

    /// `self * val`, lifting `val` through [`from_f64`](Float::from_f64).
    fn mul_f64(self, val: f64) -> Self {
        self * Self::from_f64(val)
    }

    /// `self / val`, lifting `val` through [`from_f64`](Float::from_f64).
    fn div_f64(self, val: f64) -> Self {
        self / Self::from_f64(val)
    }
}

impl Float for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn from_f64(val: f64) -> Self {
        val
    }

    fn sin(self) -> Self {
        f64::sin(self)
    }

    fn cos(self) -> Self {
        f64::cos(self)
    }

    fn tan(self) -> Self {
        f64::tan(self)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    fn abs(self) -> Self {
        f64::abs(self)
    }

    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }

    fn powf(self, p: f64) -> Self {
        f64::powf(self, p)
    }

    fn atan(self) -> Self {
        f64::atan(self)
    }

    fn atan2(self, other: Self) -> Self {
        f64::atan2(self, other)
    }

    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }

    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }

    fn recip(self) -> Self {
        f64::recip(self)
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn ln(self) -> Self {
        f64::ln(self)
    }

    fn real_value(&self) -> f64 {
        *self
    }
}

impl Float for Dual {
    fn zero() -> Self {
        Dual::constant(0.0)
    }

    fn one() -> Self {
        Dual::constant(1.0)
    }

    fn from_f64(val: f64) -> Self {
        Dual::constant(val)
    }

    fn sin(self) -> Self {
        Dual::sin(self)
    }

    fn cos(self) -> Self {
        Dual::cos(self)
    }

    fn tan(self) -> Self {
        Dual::tan(self)
    }

    fn sqrt(self) -> Self {
        Dual::sqrt(self)
    }

    fn abs(self) -> Self {
        Dual::abs(self)
    }

    fn powi(self, n: i32) -> Self {
        Dual::powi(self, n)
    }

    fn powf(self, p: f64) -> Self {
        Dual::powf(self, p)
    }

    fn atan(self) -> Self {
        Dual::atan(self)
    }

    fn atan2(self, other: Self) -> Self {
        Dual::atan2(self, other)
    }

    fn max(self, other: Self) -> Self {
        Dual::max(self, other)
    }

    fn min(self, other: Self) -> Self {
        Dual::min(self, other)
    }

    fn recip(self) -> Self {
        Dual::recip(self)
    }

    fn exp(self) -> Self {
        Dual::exp(self)
    }

    fn ln(self) -> Self {
        Dual::ln(self)
    }

    fn real_value(&self) -> f64 {
        self.real
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn square<T: Float>(x: T) -> T {
        x * x
    }

    // mu * (mass * g + downforce_coeff * v * v)
    fn circle_force<T: Float>(mu: T, mass: T, g: T, v: T, downforce_coeff: T) -> T {
        mu * (mass * g + downforce_coeff * v * v)
    }

    #[test]
    fn square_generic_over_f64() {
        let r: f64 = square(3.0);
        assert_eq!(r, 9.0);
    }

    #[test]
    fn square_generic_over_dual() {
        // d/dx x² = 2x; at x = 3 -> 6
        let x = Dual::variable(3.0);
        let r = square(x);
        assert_eq!(r.real, 9.0);
        assert_eq!(r.dual, 6.0);
    }

    #[test]
    fn from_f64_both_types() {
        assert_eq!(<f64 as Float>::from_f64(2.5), 2.5);
        assert_eq!(<Dual as Float>::from_f64(2.5), Dual::constant(2.5));
        assert_eq!(<Dual as Float>::from_f64(2.5).dual, 0.0);
    }

    #[test]
    fn real_value_both_types() {
        let a: f64 = 4.2;
        assert_eq!(a.real_value(), 4.2);

        let d = Dual::new(4.2, 9.9);
        assert_eq!(d.real_value(), 4.2);
    }

    #[test]
    fn zero_one_both_types() {
        assert_eq!(<f64 as Float>::zero(), 0.0);
        assert_eq!(<f64 as Float>::one(), 1.0);
        assert_eq!(<Dual as Float>::zero(), Dual::constant(0.0));
        assert_eq!(<Dual as Float>::one(), Dual::constant(1.0));
    }

    #[test]
    fn mixed_f64_ops_in_generic_code() {
        fn affine<T: Float>(x: T) -> T {
            x * 2.0 + 1.0
        }
        // f64: 3*2 + 1 = 7
        assert_eq!(affine(3.0_f64), 7.0);
        // Dual: value 7, derivative 2
        let r = affine(Dual::variable(3.0));
        assert_eq!(r.real, 7.0);
        assert_eq!(r.dual, 2.0);
    }

    #[test]
    fn helper_methods() {
        let x = Dual::variable(3.0);
        assert_eq!(x.add_f64(1.0), Dual::new(4.0, 1.0));
        assert_eq!(x.sub_f64(1.0), Dual::new(2.0, 1.0));
        assert_eq!(x.mul_f64(2.0), Dual::new(6.0, 2.0));
        assert_eq!(x.div_f64(2.0), Dual::new(1.5, 0.5));
    }

    #[test]
    fn circle_force_f64() {
        // mu=1.5, mass=800, g=9.81, v=30, downforce=2.0
        // mass*g = 7848; downforce*v*v = 2*900 = 1800
        // mu * (7848 + 1800) = 1.5 * 9648 = 14472
        let f = circle_force(1.5, 800.0, 9.81, 30.0, 2.0);
        assert!((f - 14472.0).abs() < 1e-9);
    }

    #[test]
    fn circle_force_dual_derivative_wrt_v() {
        // f(v) = mu * (mass*g + c*v²); df/dv = mu * 2*c*v
        // at mu=1.5, c=2.0, v=30: 1.5 * 2 * 2 * 30 = 180
        let mu = Dual::constant(1.5);
        let mass = Dual::constant(800.0);
        let g = Dual::constant(9.81);
        let v = Dual::variable(30.0);
        let c = Dual::constant(2.0);

        let f = circle_force(mu, mass, g, v, c);
        assert!((f.real - 14472.0).abs() < 1e-9);
        assert!((f.dual - 180.0).abs() < TOL);
    }
}
