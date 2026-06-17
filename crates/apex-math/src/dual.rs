//! Dual numbers for forward-mode automatic differentiation.
//!
//! A [`Dual`] carries a `real` value and a `dual` (derivative) part.
//! Arithmetic and elementary functions propagate the derivative through the
//! chain rule, so evaluating an expression on a [`Dual::variable`] yields both
//! the function value and its derivative simultaneously.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A dual number `real + dual·ε`, where `ε² = 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    pub real: f64,
    pub dual: f64,
}

impl Dual {
    /// Creates a dual number from explicit real and dual parts.
    pub fn new(real: f64, dual: f64) -> Dual {
        Dual { real, dual }
    }

    /// Creates a constant: a value with a zero derivative.
    pub fn constant(val: f64) -> Dual {
        Dual {
            real: val,
            dual: 0.0,
        }
    }

    /// Creates the independent variable: a value with a unit derivative.
    pub fn variable(val: f64) -> Dual {
        Dual {
            real: val,
            dual: 1.0,
        }
    }

    /// Sine, with derivative `a' · cos(a)`.
    pub fn sin(self) -> Dual {
        Dual {
            real: self.real.sin(),
            dual: self.dual * self.real.cos(),
        }
    }

    /// Cosine, with derivative `-a' · sin(a)`.
    pub fn cos(self) -> Dual {
        Dual {
            real: self.real.cos(),
            dual: -self.dual * self.real.sin(),
        }
    }

    /// Tangent, with derivative `a' / cos²(a)`.
    pub fn tan(self) -> Dual {
        let c = self.real.cos();
        Dual {
            real: self.real.tan(),
            dual: self.dual / (c * c),
        }
    }

    /// Square root, with derivative `a' / (2√a)`.
    pub fn sqrt(self) -> Dual {
        let r = self.real.sqrt();
        Dual {
            real: r,
            dual: self.dual / (2.0 * r),
        }
    }

    /// Absolute value, with derivative `a' · signum(a)` (zero when `a == 0`).
    pub fn abs(self) -> Dual {
        let d = if self.real == 0.0 {
            0.0
        } else {
            self.dual * self.real.signum()
        };
        Dual {
            real: self.real.abs(),
            dual: d,
        }
    }

    /// Integer power, with derivative `a' · n · a^(n-1)`.
    pub fn powi(self, n: i32) -> Dual {
        Dual {
            real: self.real.powi(n),
            dual: self.dual * (n as f64) * self.real.powi(n - 1),
        }
    }

    /// Floating-point power, with derivative `a' · p · a^(p-1)`.
    pub fn powf(self, p: f64) -> Dual {
        Dual {
            real: self.real.powf(p),
            dual: self.dual * p * self.real.powf(p - 1.0),
        }
    }

    /// Arctangent, with derivative `a' / (1 + a²)`.
    pub fn atan(self) -> Dual {
        Dual {
            real: self.real.atan(),
            dual: self.dual / (1.0 + self.real * self.real),
        }
    }

    /// Two-argument arctangent of `self` (y) and `other` (x), with derivative
    /// `(a'·b - a·b') / (a² + b²)`.
    pub fn atan2(self, other: Dual) -> Dual {
        let denom = self.real * self.real + other.real * other.real;
        Dual {
            real: self.real.atan2(other.real),
            dual: (self.dual * other.real - self.real * other.dual) / denom,
        }
    }

    /// Returns `self` if its real part is greater than or equal to `other`'s,
    /// otherwise `other`.
    pub fn max(self, other: Dual) -> Dual {
        if self.real >= other.real {
            self
        } else {
            other
        }
    }

    /// Returns `self` if its real part is less than or equal to `other`'s,
    /// otherwise `other`.
    pub fn min(self, other: Dual) -> Dual {
        if self.real <= other.real {
            self
        } else {
            other
        }
    }

    /// Reciprocal, with derivative `-a' / a²`.
    pub fn recip(self) -> Dual {
        Dual {
            real: 1.0 / self.real,
            dual: -self.dual / (self.real * self.real),
        }
    }

    /// Exponential, with derivative `a' · exp(a)`.
    pub fn exp(self) -> Dual {
        let e = self.real.exp();
        Dual {
            real: e,
            dual: self.dual * e,
        }
    }

    /// Natural logarithm, with derivative `a' / a`.
    pub fn ln(self) -> Dual {
        Dual {
            real: self.real.ln(),
            dual: self.dual / self.real,
        }
    }
}

impl Add for Dual {
    type Output = Dual;

    fn add(self, rhs: Dual) -> Dual {
        Dual {
            real: self.real + rhs.real,
            dual: self.dual + rhs.dual,
        }
    }
}

impl Sub for Dual {
    type Output = Dual;

    fn sub(self, rhs: Dual) -> Dual {
        Dual {
            real: self.real - rhs.real,
            dual: self.dual - rhs.dual,
        }
    }
}

impl Mul for Dual {
    type Output = Dual;

    fn mul(self, rhs: Dual) -> Dual {
        // product rule: (a·b)' = a'·b + a·b'
        Dual {
            real: self.real * rhs.real,
            dual: self.dual * rhs.real + self.real * rhs.dual,
        }
    }
}

impl Div for Dual {
    type Output = Dual;

    fn div(self, rhs: Dual) -> Dual {
        // quotient rule: (a/b)' = (a'·b - a·b') / b²
        Dual {
            real: self.real / rhs.real,
            dual: (self.dual * rhs.real - self.real * rhs.dual) / (rhs.real * rhs.real),
        }
    }
}

impl Neg for Dual {
    type Output = Dual;

    fn neg(self) -> Dual {
        Dual {
            real: -self.real,
            dual: -self.dual,
        }
    }
}

// --- Mixed f64 / Dual overloads ---

impl Add<Dual> for f64 {
    type Output = Dual;

    fn add(self, rhs: Dual) -> Dual {
        Dual::constant(self) + rhs
    }
}

impl Add<f64> for Dual {
    type Output = Dual;

    fn add(self, rhs: f64) -> Dual {
        self + Dual::constant(rhs)
    }
}

impl Sub<Dual> for f64 {
    type Output = Dual;

    fn sub(self, rhs: Dual) -> Dual {
        Dual::constant(self) - rhs
    }
}

impl Sub<f64> for Dual {
    type Output = Dual;

    fn sub(self, rhs: f64) -> Dual {
        self - Dual::constant(rhs)
    }
}

impl Mul<Dual> for f64 {
    type Output = Dual;

    fn mul(self, rhs: Dual) -> Dual {
        Dual::constant(self) * rhs
    }
}

impl Mul<f64> for Dual {
    type Output = Dual;

    fn mul(self, rhs: f64) -> Dual {
        self * Dual::constant(rhs)
    }
}

impl Div<Dual> for f64 {
    type Output = Dual;

    fn div(self, rhs: Dual) -> Dual {
        Dual::constant(self) / rhs
    }
}

impl Div<f64> for Dual {
    type Output = Dual;

    fn div(self, rhs: f64) -> Dual {
        self / Dual::constant(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn approx_eq_dual(a: Dual, b: Dual, tol: f64) -> bool {
        (a.real - b.real).abs() <= tol && (a.dual - b.dual).abs() <= tol
    }

    const TOL: f64 = 1e-12;

    #[test]
    fn constructors() {
        let c = Dual::constant(5.0);
        assert_eq!(c.real, 5.0);
        assert_eq!(c.dual, 0.0);

        let v = Dual::variable(5.0);
        assert_eq!(v.real, 5.0);
        assert_eq!(v.dual, 1.0);

        let g = Dual::new(2.0, 3.0);
        assert_eq!(g.real, 2.0);
        assert_eq!(g.dual, 3.0);
    }

    #[test]
    fn dual_add() {
        let a = Dual::new(1.0, 2.0);
        let b = Dual::new(3.0, 4.0);
        assert_eq!(a + b, Dual::new(4.0, 6.0));
    }

    #[test]
    fn dual_sub() {
        let a = Dual::new(5.0, 4.0);
        let b = Dual::new(3.0, 1.0);
        assert_eq!(a - b, Dual::new(2.0, 3.0));
    }

    #[test]
    fn dual_mul_product_rule() {
        // (3 + 1ε)(4 + 0ε) = 12 + 4ε
        let a = Dual::new(3.0, 1.0);
        let b = Dual::new(4.0, 0.0);
        assert_eq!(a * b, Dual::new(12.0, 4.0));

        // general product rule: (2+3ε)(5+7ε) = 10 + (3*5 + 2*7)ε = 10 + 29ε
        let a = Dual::new(2.0, 3.0);
        let b = Dual::new(5.0, 7.0);
        assert_eq!(a * b, Dual::new(10.0, 29.0));
    }

    #[test]
    fn dual_div_quotient_rule() {
        // (6+1ε)/(2+0ε) = 3 + (1*2 - 6*0)/4 ε = 3 + 0.5ε
        let a = Dual::new(6.0, 1.0);
        let b = Dual::new(2.0, 0.0);
        assert!(approx_eq_dual(a / b, Dual::new(3.0, 0.5), TOL));

        // (6+2ε)/(3+1ε) = 2 + (2*3 - 6*1)/9 ε = 2 + 0ε
        let a = Dual::new(6.0, 2.0);
        let b = Dual::new(3.0, 1.0);
        assert!(approx_eq_dual(a / b, Dual::new(2.0, 0.0), TOL));
    }

    #[test]
    fn dual_neg() {
        assert_eq!(-Dual::new(2.0, -3.0), Dual::new(-2.0, 3.0));
    }

    #[test]
    fn mixed_add() {
        let x = Dual::variable(4.0);
        assert_eq!(x + 1.0, Dual::new(5.0, 1.0));
        assert_eq!(1.0 + x, Dual::new(5.0, 1.0));
    }

    #[test]
    fn mixed_sub() {
        let x = Dual::variable(4.0);
        assert_eq!(x - 1.0, Dual::new(3.0, 1.0));
        // 1 - x has derivative -1
        assert_eq!(1.0 - x, Dual::new(-3.0, -1.0));
    }

    #[test]
    fn mixed_mul() {
        let x = Dual::variable(3.0);
        assert_eq!(2.0 * x, Dual::new(6.0, 2.0));
        assert_eq!(x * 2.0, Dual::new(6.0, 2.0));
    }

    #[test]
    fn mixed_div() {
        let x = Dual::variable(4.0);
        // x / 2 has derivative 1/2
        assert!(approx_eq_dual(x / 2.0, Dual::new(2.0, 0.5), TOL));
        // 8 / x at x=4 = 2, derivative -8/x² = -0.5
        assert!(approx_eq_dual(8.0 / x, Dual::new(2.0, -0.5), TOL));
    }

    #[test]
    fn deriv_sin() {
        // d/dx sin(x) = cos(x); at x = π/4
        let x = Dual::variable(PI / 4.0);
        let r = x.sin();
        assert!(approx_eq_dual(r, Dual::new((PI / 4.0).sin(), (PI / 4.0).cos()), TOL));
    }

    #[test]
    fn deriv_cos() {
        // d/dx cos(x) = -sin(x); at x = π/3
        let x = Dual::variable(PI / 3.0);
        let r = x.cos();
        assert!(approx_eq_dual(
            r,
            Dual::new((PI / 3.0).cos(), -(PI / 3.0).sin()),
            TOL
        ));
    }

    #[test]
    fn deriv_tan() {
        // d/dx tan(x) = 1/cos²(x); at x = π/6
        let x = Dual::variable(PI / 6.0);
        let r = x.tan();
        let expected_dual = 1.0 / (PI / 6.0).cos().powi(2);
        assert!(approx_eq_dual(r, Dual::new((PI / 6.0).tan(), expected_dual), TOL));
    }

    #[test]
    fn deriv_sqrt() {
        // d/dx sqrt(x) = 1/(2√x); at x = 4 -> 0.25
        let x = Dual::variable(4.0);
        let r = x.sqrt();
        assert!(approx_eq_dual(r, Dual::new(2.0, 0.25), TOL));
    }

    #[test]
    fn deriv_abs() {
        // positive branch
        let x = Dual::variable(3.0);
        assert_eq!(x.abs(), Dual::new(3.0, 1.0));
        // negative branch: derivative flips sign
        let x = Dual::variable(-3.0);
        assert_eq!(x.abs(), Dual::new(3.0, -1.0));
        // at zero: dual part is 0.0
        let x = Dual::variable(0.0);
        assert_eq!(x.abs(), Dual::new(0.0, 0.0));
    }

    #[test]
    fn deriv_powi() {
        // f(x) = x²; f'(3) = 6
        let x = Dual::variable(3.0);
        assert!(approx_eq_dual(x.powi(2), Dual::new(9.0, 6.0), TOL));
        // verify x*x agrees with powi(2)
        assert!(approx_eq_dual(x * x, x.powi(2), TOL));
        // f(x) = x³; f'(2) = 3*4 = 12
        let x = Dual::variable(2.0);
        assert!(approx_eq_dual(x.powi(3), Dual::new(8.0, 12.0), TOL));
    }

    #[test]
    fn deriv_powf() {
        // f(x) = x^2.5; f'(x) = 2.5 x^1.5; at x = 4
        let x = Dual::variable(4.0);
        let r = x.powf(2.5);
        let expected_real = 4.0_f64.powf(2.5);
        let expected_dual = 2.5 * 4.0_f64.powf(1.5);
        assert!(approx_eq_dual(r, Dual::new(expected_real, expected_dual), 1e-10));
    }

    #[test]
    fn deriv_atan() {
        // d/dx atan(x) = 1/(1+x²); at x = 1 -> 0.5
        let x = Dual::variable(1.0);
        let r = x.atan();
        assert!(approx_eq_dual(r, Dual::new((1.0_f64).atan(), 0.5), TOL));
    }

    #[test]
    fn deriv_atan2() {
        // atan2(y, x) with y = variable, x = constant.
        // d/dy atan2(y, x) = x / (x² + y²); at y=1, x=1 -> 1/2
        let y = Dual::variable(1.0);
        let x = Dual::constant(1.0);
        let r = y.atan2(x);
        assert!(approx_eq_dual(r, Dual::new((1.0_f64).atan2(1.0), 0.5), TOL));
    }

    #[test]
    fn max_and_min() {
        let a = Dual::new(3.0, 1.0);
        let b = Dual::new(5.0, 0.0);
        assert_eq!(a.max(b), b);
        assert_eq!(a.min(b), a);
        // tie picks self
        let c = Dual::new(3.0, 7.0);
        assert_eq!(a.max(c), a);
        assert_eq!(a.min(c), a);
    }

    #[test]
    fn deriv_recip() {
        // d/dx (1/x) = -1/x²; at x = 2 -> -0.25
        let x = Dual::variable(2.0);
        let r = x.recip();
        assert!(approx_eq_dual(r, Dual::new(0.5, -0.25), TOL));
    }

    #[test]
    fn deriv_exp() {
        // d/dx exp(x) = exp(x); at x = 1 -> e
        let x = Dual::variable(1.0);
        let r = x.exp();
        let e = 1.0_f64.exp();
        assert!(approx_eq_dual(r, Dual::new(e, e), TOL));
    }

    #[test]
    fn deriv_ln() {
        // d/dx ln(x) = 1/x; at x = 2 -> 0.5
        let x = Dual::variable(2.0);
        let r = x.ln();
        assert!(approx_eq_dual(r, Dual::new(2.0_f64.ln(), 0.5), TOL));
    }

    #[test]
    fn chain_rule_sin_of_square() {
        // f(x) = sin(x²); f'(x) = 2x·cos(x²); at x = 1.3
        let x = Dual::variable(1.3);
        let r = x.powi(2).sin();
        let expected_real = (1.3_f64 * 1.3).sin();
        let expected_dual = 2.0 * 1.3 * (1.3_f64 * 1.3).cos();
        assert!(approx_eq_dual(r, Dual::new(expected_real, expected_dual), TOL));

        // same composition via x*x instead of powi
        let r2 = (x * x).sin();
        assert!(approx_eq_dual(r2, r, TOL));
    }

    #[test]
    fn chain_rule_composite_expression() {
        // f(x) = (2x + 1) / sqrt(x); at x = 4
        // f(x) = (2x+1)·x^(-1/2) = 2x^(1/2) + x^(-1/2)
        // f'(x) = x^(-1/2) - 0.5 x^(-3/2); at x=4: 0.5 - 0.5*0.125 = 0.4375
        let x = Dual::variable(4.0);
        let r = (2.0 * x + 1.0) / x.sqrt();
        let expected_real = 9.0 / 2.0; // (8+1)/2
        let expected_dual = 0.4375;
        assert!(approx_eq_dual(r, Dual::new(expected_real, expected_dual), 1e-10));
    }
}
