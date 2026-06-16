//! A 3-dimensional vector of `f64` components.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A 3-dimensional vector with `f64` components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Default for Vec3 {
    fn default() -> Self {
        Vec3::zero()
    }
}

impl Vec3 {
    /// Creates a new vector from its components.
    pub fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    /// Returns the zero vector.
    pub fn zero() -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Returns a vector with all three components set to `v`.
    pub fn splat(v: f64) -> Vec3 {
        Vec3 { x: v, y: v, z: v }
    }

    /// Returns the dot product of `self` and `other`.
    pub fn dot(self, other: Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Returns the cross product of `self` and `other`.
    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Returns the Euclidean length of the vector.
    pub fn magnitude(self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    /// Returns the squared Euclidean length of the vector.
    pub fn magnitude_squared(self) -> f64 {
        self.dot(self)
    }

    /// Returns a unit vector in the same direction as `self`.
    ///
    /// If the magnitude is near zero (below `1e-12`), the zero vector is
    /// returned instead to avoid division by zero.
    pub fn normalized(self) -> Vec3 {
        let mag = self.magnitude();
        if mag < 1e-12 {
            Vec3::zero()
        } else {
            self / mag
        }
    }
}

impl Add for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f64) -> Vec3 {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs * self
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f64) -> Vec3 {
        Vec3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    const TOL: f64 = 1e-12;

    #[test]
    fn construction_and_field_access() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);

        assert_eq!(Vec3::zero(), Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(Vec3::default(), Vec3::zero());
        assert_eq!(Vec3::splat(5.0), Vec3::new(5.0, 5.0, 5.0));
    }

    #[test]
    fn dot_product() {
        // (1,2,3) . (4,-5,6) = 4 - 10 + 18 = 12
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, -5.0, 6.0);
        assert_eq!(a.dot(b), 12.0);

        // orthogonal vectors have zero dot product
        assert_eq!(Vec3::new(1.0, 0.0, 0.0).dot(Vec3::new(0.0, 1.0, 0.0)), 0.0);
    }

    #[test]
    fn cross_product() {
        let i = Vec3::new(1.0, 0.0, 0.0);
        let j = Vec3::new(0.0, 1.0, 0.0);
        let k = Vec3::new(0.0, 0.0, 1.0);

        // i x j = k, j x k = i, k x i = j
        assert_eq!(i.cross(j), k);
        assert_eq!(j.cross(k), i);
        assert_eq!(k.cross(i), j);

        // anticommutative: j x i = -k
        assert_eq!(j.cross(i), -k);

        // a general example: (1,2,3) x (4,5,6) = (-3, 6, -3)
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.cross(b), Vec3::new(-3.0, 6.0, -3.0));
    }

    #[test]
    fn magnitude_and_squared() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(v.magnitude_squared(), 25.0);
        assert_eq!(v.magnitude(), 5.0);

        // (1,2,2) has magnitude 3
        let v = Vec3::new(1.0, 2.0, 2.0);
        assert!(approx_eq(v.magnitude(), 3.0, TOL));
    }

    #[test]
    fn normalized_unit_length() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = v.normalized();
        assert!(approx_eq(n.x, 0.6, TOL));
        assert!(approx_eq(n.y, 0.8, TOL));
        assert!(approx_eq(n.z, 0.0, TOL));
        assert!(approx_eq(n.magnitude(), 1.0, TOL));
    }

    #[test]
    fn normalized_near_zero_returns_zero() {
        // exactly zero
        assert_eq!(Vec3::zero().normalized(), Vec3::zero());

        // below threshold
        let tiny = Vec3::splat(1e-13);
        assert_eq!(tiny.normalized(), Vec3::zero());
    }

    #[test]
    fn op_add() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a + b, Vec3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn op_sub() {
        let a = Vec3::new(4.0, 5.0, 6.0);
        let b = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(a - b, Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn op_mul_scalar() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        assert_eq!(v * 2.0, Vec3::new(2.0, -4.0, 6.0));
    }

    #[test]
    fn op_mul_scalar_reversed() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        assert_eq!(2.0 * v, Vec3::new(2.0, -4.0, 6.0));
        assert_eq!(2.0 * v, v * 2.0);
    }

    #[test]
    fn op_div_scalar() {
        let v = Vec3::new(2.0, -4.0, 6.0);
        assert_eq!(v / 2.0, Vec3::new(1.0, -2.0, 3.0));
    }

    #[test]
    fn op_neg() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        assert_eq!(-v, Vec3::new(-1.0, 2.0, -3.0));
    }
}
