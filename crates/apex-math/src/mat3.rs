//! A 3x3 matrix of `f64` elements stored in row-major order.

use std::ops::{Add, Mul};

use crate::vec3::Vec3;

/// A 3x3 matrix with `f64` elements stored in row-major order.
///
/// Element `(row, col)` is located at `data[row * 3 + col]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub data: [f64; 9],
}

impl Default for Mat3 {
    fn default() -> Self {
        Mat3::identity()
    }
}

impl Mat3 {
    /// Returns the zero matrix (all elements zero).
    pub fn zero() -> Mat3 {
        Mat3 { data: [0.0; 9] }
    }

    /// Returns the 3x3 identity matrix.
    pub fn identity() -> Mat3 {
        Mat3 {
            data: [
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, //
            ],
        }
    }

    /// Constructs a matrix from three row vectors.
    pub fn from_rows(row0: Vec3, row1: Vec3, row2: Vec3) -> Mat3 {
        Mat3 {
            data: [
                row0.x, row0.y, row0.z, //
                row1.x, row1.y, row1.z, //
                row2.x, row2.y, row2.z, //
            ],
        }
    }

    /// Constructs a diagonal matrix with the given diagonal entries; all
    /// off-diagonal elements are zero.
    pub fn from_diagonal(d0: f64, d1: f64, d2: f64) -> Mat3 {
        Mat3 {
            data: [
                d0, 0.0, 0.0, //
                0.0, d1, 0.0, //
                0.0, 0.0, d2, //
            ],
        }
    }

    /// Returns the element at `(row, col)`.
    ///
    /// Performs no bounds checking outside of debug builds.
    pub fn at(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < 3 && col < 3, "Mat3 index out of bounds");
        self.data[row * 3 + col]
    }

    /// Returns a mutable reference to the element at `(row, col)`.
    ///
    /// Performs no bounds checking outside of debug builds.
    pub fn at_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        debug_assert!(row < 3 && col < 3, "Mat3 index out of bounds");
        &mut self.data[row * 3 + col]
    }

    /// Returns row `i` as a [`Vec3`].
    pub fn row(&self, i: usize) -> Vec3 {
        debug_assert!(i < 3, "Mat3 row index out of bounds");
        Vec3::new(self.data[i * 3], self.data[i * 3 + 1], self.data[i * 3 + 2])
    }

    /// Returns column `j` as a [`Vec3`].
    pub fn col(&self, j: usize) -> Vec3 {
        debug_assert!(j < 3, "Mat3 column index out of bounds");
        Vec3::new(self.data[j], self.data[3 + j], self.data[6 + j])
    }

    /// Returns the transpose of the matrix.
    pub fn transpose(self) -> Mat3 {
        let m = &self.data;
        Mat3 {
            data: [
                m[0], m[3], m[6], //
                m[1], m[4], m[7], //
                m[2], m[5], m[8], //
            ],
        }
    }

    /// Returns the determinant of the matrix.
    pub fn determinant(self) -> f64 {
        let m = &self.data;
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6])
    }

    /// Returns the inverse of the matrix, or `None` if it is singular
    /// (determinant below `1e-12` in absolute value).
    ///
    /// Computed via the cofactor/adjugate method.
    pub fn inverse(self) -> Option<Mat3> {
        let det = self.determinant();
        if det.abs() < 1e-12 {
            return None;
        }

        let m = &self.data;
        let inv_det = 1.0 / det;

        // Cofactor matrix; transposed in place to form the adjugate.
        let c00 = m[4] * m[8] - m[5] * m[7];
        let c01 = m[5] * m[6] - m[3] * m[8];
        let c02 = m[3] * m[7] - m[4] * m[6];
        let c10 = m[2] * m[7] - m[1] * m[8];
        let c11 = m[0] * m[8] - m[2] * m[6];
        let c12 = m[1] * m[6] - m[0] * m[7];
        let c20 = m[1] * m[5] - m[2] * m[4];
        let c21 = m[2] * m[3] - m[0] * m[5];
        let c22 = m[0] * m[4] - m[1] * m[3];

        Some(Mat3 {
            data: [
                c00 * inv_det,
                c10 * inv_det,
                c20 * inv_det, //
                c01 * inv_det,
                c11 * inv_det,
                c21 * inv_det, //
                c02 * inv_det,
                c12 * inv_det,
                c22 * inv_det, //
            ],
        })
    }

    /// Returns the trace (sum of the diagonal elements).
    pub fn trace(self) -> f64 {
        self.data[0] + self.data[4] + self.data[8]
    }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3::new(
            self.row(0).dot(rhs),
            self.row(1).dot(rhs),
            self.row(2).dot(rhs),
        )
    }
}

impl Mul<Mat3> for Mat3 {
    type Output = Mat3;

    fn mul(self, rhs: Mat3) -> Mat3 {
        let mut data = [0.0; 9];
        for r in 0..3 {
            for c in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += self.at(r, k) * rhs.at(k, c);
                }
                data[r * 3 + c] = sum;
            }
        }
        Mat3 { data }
    }
}

impl Add for Mat3 {
    type Output = Mat3;

    fn add(self, rhs: Mat3) -> Mat3 {
        let data = std::array::from_fn(|i| self.data[i] + rhs.data[i]);
        Mat3 { data }
    }
}

impl Mul<f64> for Mat3 {
    type Output = Mat3;

    fn mul(self, rhs: f64) -> Mat3 {
        let data = std::array::from_fn(|i| self.data[i] * rhs);
        Mat3 { data }
    }
}

impl Mul<Mat3> for f64 {
    type Output = Mat3;

    fn mul(self, rhs: Mat3) -> Mat3 {
        rhs * self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_mat3(a: Mat3, b: Mat3, tol: f64) -> bool {
        a.data
            .iter()
            .zip(b.data.iter())
            .all(|(x, y)| (x - y).abs() <= tol)
    }

    const TOL: f64 = 1e-10;

    #[test]
    fn identity_and_default() {
        let id = Mat3::identity();
        assert_eq!(id.at(0, 0), 1.0);
        assert_eq!(id.at(1, 1), 1.0);
        assert_eq!(id.at(2, 2), 1.0);
        assert_eq!(id.at(0, 1), 0.0);
        assert_eq!(id.trace(), 3.0);

        // Default is identity, not zero.
        assert_eq!(Mat3::default(), id);
        assert_ne!(Mat3::default(), Mat3::zero());
    }

    #[test]
    fn zero_construction() {
        assert_eq!(Mat3::zero().data, [0.0; 9]);
    }

    #[test]
    fn from_rows_construction() {
        let m = Mat3::from_rows(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );
        assert_eq!(m.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn from_diagonal_construction() {
        let m = Mat3::from_diagonal(2.0, 3.0, 4.0);
        assert_eq!(m.data, [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]);
    }

    #[test]
    fn at_and_at_mut() {
        let mut m = Mat3::from_rows(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );
        assert_eq!(m.at(0, 0), 1.0);
        assert_eq!(m.at(1, 2), 6.0);
        assert_eq!(m.at(2, 1), 8.0);

        *m.at_mut(1, 2) = 60.0;
        assert_eq!(m.at(1, 2), 60.0);
    }

    #[test]
    fn row_and_col_extraction() {
        let m = Mat3::from_rows(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );
        assert_eq!(m.row(0), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(m.row(2), Vec3::new(7.0, 8.0, 9.0));
        assert_eq!(m.col(0), Vec3::new(1.0, 4.0, 7.0));
        assert_eq!(m.col(2), Vec3::new(3.0, 6.0, 9.0));
    }

    #[test]
    fn transpose_works() {
        let m = Mat3::from_rows(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );
        let t = m.transpose();
        assert_eq!(t.row(0), Vec3::new(1.0, 4.0, 7.0));
        assert_eq!(t.row(1), Vec3::new(2.0, 5.0, 8.0));
        assert_eq!(t.row(2), Vec3::new(3.0, 6.0, 9.0));

        // (A^T)^T == A
        assert_eq!(m.transpose().transpose(), m);
    }

    #[test]
    fn determinant_hand_calculated() {
        // | 6  1  1 |
        // | 4 -2  5 |
        // | 2  8  7 |
        // det = 6(-2*7 - 5*8) - 1(4*7 - 5*2) + 1(4*8 - (-2)*2)
        //     = 6(-54) - 1(18) + 1(36) = -324 - 18 + 36 = -306
        let m = Mat3::from_rows(
            Vec3::new(6.0, 1.0, 1.0),
            Vec3::new(4.0, -2.0, 5.0),
            Vec3::new(2.0, 8.0, 7.0),
        );
        assert_eq!(m.determinant(), -306.0);

        // identity determinant is 1, singular matrix is 0
        assert_eq!(Mat3::identity().determinant(), 1.0);
        let singular = Mat3::from_rows(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(2.0, 4.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );
        assert_eq!(singular.determinant(), 0.0);
    }

    #[test]
    fn inverse_times_original_is_identity() {
        let m = Mat3::from_rows(
            Vec3::new(6.0, 1.0, 1.0),
            Vec3::new(4.0, -2.0, 5.0),
            Vec3::new(2.0, 8.0, 7.0),
        );
        let inv = m.inverse().expect("matrix is invertible");
        assert!(approx_eq_mat3(m * inv, Mat3::identity(), TOL));
        assert!(approx_eq_mat3(inv * m, Mat3::identity(), TOL));
    }

    #[test]
    fn inverse_of_singular_is_none() {
        let singular = Mat3::from_rows(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(2.0, 4.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );
        assert!(singular.inverse().is_none());
        assert!(Mat3::zero().inverse().is_none());
    }

    #[test]
    fn trace_works() {
        let m = Mat3::from_rows(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );
        assert_eq!(m.trace(), 15.0);
    }

    #[test]
    fn mat_vec_multiply() {
        // | 1 2 3 |   | 1 |   | 14 |
        // | 4 5 6 | * | 2 | = | 32 |
        // | 7 8 9 |   | 3 |   | 50 |
        let m = Mat3::from_rows(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(m * v, Vec3::new(14.0, 32.0, 50.0));

        // identity leaves the vector unchanged
        assert_eq!(Mat3::identity() * v, v);
    }

    #[test]
    fn mat_mat_multiply() {
        let a = Mat3::from_rows(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );

        // identity * A == A and A * identity == A
        assert_eq!(Mat3::identity() * a, a);
        assert_eq!(a * Mat3::identity(), a);

        // hand-calculated product:
        // | 1 2 |->3   | 9 8 7 |
        // A * B where B =
        // | 9 8 7 |
        // | 6 5 4 |
        // | 3 2 1 |
        let b = Mat3::from_rows(
            Vec3::new(9.0, 8.0, 7.0),
            Vec3::new(6.0, 5.0, 4.0),
            Vec3::new(3.0, 2.0, 1.0),
        );
        // row0: [1*9+2*6+3*3, 1*8+2*5+3*2, 1*7+2*4+3*1] = [30, 24, 18]
        // row1: [4*9+5*6+6*3, 4*8+5*5+6*2, 4*7+5*4+6*1] = [84, 69, 54]
        // row2: [7*9+8*6+9*3, 7*8+8*5+9*2, 7*7+8*4+9*1] = [138, 114, 90]
        let expected = Mat3::from_rows(
            Vec3::new(30.0, 24.0, 18.0),
            Vec3::new(84.0, 69.0, 54.0),
            Vec3::new(138.0, 114.0, 90.0),
        );
        assert_eq!(a * b, expected);
    }

    #[test]
    fn mat_addition() {
        let a = Mat3::from_diagonal(1.0, 2.0, 3.0);
        let b = Mat3::from_diagonal(4.0, 5.0, 6.0);
        assert_eq!(a + b, Mat3::from_diagonal(5.0, 7.0, 9.0));

        let a = Mat3::from_rows(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );
        assert_eq!(a + Mat3::zero(), a);
    }

    #[test]
    fn scalar_multiply() {
        let a = Mat3::from_rows(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );
        let expected = Mat3::from_rows(
            Vec3::new(2.0, 4.0, 6.0),
            Vec3::new(8.0, 10.0, 12.0),
            Vec3::new(14.0, 16.0, 18.0),
        );
        assert_eq!(a * 2.0, expected);
        assert_eq!(2.0 * a, expected);
        assert_eq!(2.0 * a, a * 2.0);
    }
}
