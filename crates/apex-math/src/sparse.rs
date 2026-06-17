//! Compressed Sparse Row (CSR) matrix storage and operations.
//!
//! CSR stores only non-zero entries, row by row, making it efficient for
//! row-wise access and matrix-vector products. It is the standard format for
//! feeding sparse Jacobians (such as the banded one from the collocation
//! optimizer) into optimization solvers.

/// A sparse matrix in Compressed Sparse Row (CSR) format.
///
/// Stores only non-zero elements. Efficient for row-wise access and
/// matrix-vector multiplication. This is the standard format for
/// feeding sparse Jacobians into optimization solvers.
///
/// Storage:
/// - `values`: non-zero entries, row by row.
/// - `col_indices`: column index of each entry in `values`.
/// - `row_ptr`: row_ptr[i] is the index into `values` where row i starts.
///   row_ptr has length nrows+1, and row_ptr[nrows] = nnz.
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    nrows: usize,
    ncols: usize,
    values: Vec<f64>,
    col_indices: Vec<usize>,
    row_ptr: Vec<usize>,
}

/// Builder for constructing a [`CsrMatrix`] incrementally.
///
/// Add entries in any order via `add()`, then call `build()` to produce the
/// compressed CSR format. Duplicate (row, col) entries are summed (standard
/// FEM/optimization convention).
#[derive(Debug)]
pub struct CsrBuilder {
    nrows: usize,
    ncols: usize,
    triplets: Vec<(usize, usize, f64)>, // (row, col, value)
}

impl CsrBuilder {
    /// Create a new builder for a matrix of the given dimensions.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        CsrBuilder {
            nrows,
            ncols,
            triplets: Vec::new(),
        }
    }

    /// Set a single entry. If (row, col) already exists, the value is added
    /// (not replaced) — this matches the triplet assembly convention.
    pub fn add(&mut self, row: usize, col: usize, value: f64) {
        assert!(row < self.nrows, "row {} out of bounds", row);
        assert!(col < self.ncols, "col {} out of bounds", col);
        self.triplets.push((row, col, value));
    }

    /// Build the CSR matrix. Sorts triplets by (row, col), sums duplicates,
    /// and constructs the compressed arrays.
    pub fn build(mut self) -> CsrMatrix {
        // Sort by (row, then col) so entries of each row are contiguous and
        // column-ordered (required for binary-search `get`).
        self.triplets
            .sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut values: Vec<f64> = Vec::new();
        let mut col_indices: Vec<usize> = Vec::new();
        let mut row_ptr: Vec<usize> = vec![0; self.nrows + 1];

        // Sum duplicate (row, col) entries while compressing.
        let mut current_row = 0;
        for (row, col, value) in self.triplets {
            // advance row_ptr boundaries up to this row
            while current_row < row {
                row_ptr[current_row + 1] = values.len();
                current_row += 1;
            }
            // Merge with the previous stored entry when it shares the same
            // (row, col): the last column matches and it belongs to this row.
            let can_merge =
                col_indices.last() == Some(&col) && col_indices.len() > row_ptr[current_row];
            if can_merge {
                if let Some(last_value) = values.last_mut() {
                    *last_value += value;
                }
                continue;
            }
            values.push(value);
            col_indices.push(col);
        }
        // close out remaining rows
        while current_row < self.nrows {
            row_ptr[current_row + 1] = values.len();
            current_row += 1;
        }

        CsrMatrix {
            nrows: self.nrows,
            ncols: self.ncols,
            values,
            col_indices,
            row_ptr,
        }
    }
}

impl CsrMatrix {
    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get the value at (row, col). Returns 0.0 if the entry is not stored.
    /// Uses binary search within the row for O(log nnz_per_row) access.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.nrows || col >= self.ncols {
            return 0.0;
        }
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        match self.col_indices[start..end].binary_search(&col) {
            Ok(offset) => self.values[start + offset],
            Err(_) => 0.0,
        }
    }

    /// Multiply this matrix by a dense vector: y = A * x.
    /// Panics if x.len() != ncols.
    /// Returns a Vec<f64> of length nrows.
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.ncols, "vector length must equal ncols");
        let mut y = vec![0.0; self.nrows];
        for (row, y_i) in y.iter_mut().enumerate() {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            let mut sum = 0.0;
            for k in start..end {
                sum += self.values[k] * x[self.col_indices[k]];
            }
            *y_i = sum;
        }
        y
    }

    /// Compute the transpose of this matrix. Returns a new CsrMatrix.
    pub fn transpose(&self) -> CsrMatrix {
        let mut builder = CsrBuilder::new(self.ncols, self.nrows);
        for row in 0..self.nrows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for k in start..end {
                builder.add(self.col_indices[k], row, self.values[k]);
            }
        }
        builder.build()
    }

    /// Get a slice of the values and column indices for a given row.
    /// Returns (&[f64], &[usize]) — (values, column_indices) for that row.
    pub fn row_entries(&self, row: usize) -> (&[f64], &[usize]) {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        (&self.values[start..end], &self.col_indices[start..end])
    }

    /// Create an identity matrix of the given size.
    pub fn identity(n: usize) -> CsrMatrix {
        let values = vec![1.0; n];
        let col_indices: Vec<usize> = (0..n).collect();
        let row_ptr: Vec<usize> = (0..=n).collect();
        CsrMatrix {
            nrows: n,
            ncols: n,
            values,
            col_indices,
            row_ptr,
        }
    }

    /// Create a zero matrix (no stored entries) of the given dimensions.
    pub fn zeros(nrows: usize, ncols: usize) -> CsrMatrix {
        CsrMatrix {
            nrows,
            ncols,
            values: Vec::new(),
            col_indices: Vec::new(),
            row_ptr: vec![0; nrows + 1],
        }
    }

    /// Scale all values by a scalar.
    pub fn scale(&mut self, scalar: f64) {
        for v in &mut self.values {
            *v *= scalar;
        }
    }

    /// Create a dense representation as Vec<Vec<f64>> (for debugging/testing only).
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.ncols]; self.nrows];
        for (row, drow) in dense.iter_mut().enumerate() {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for k in start..end {
                drow[self.col_indices[k]] = self.values[k];
            }
        }
        dense
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the standard 3x3 test matrix:
    /// [1 0 2]
    /// [0 3 0]
    /// [4 0 5]
    fn build_3x3() -> CsrMatrix {
        let mut b = CsrBuilder::new(3, 3);
        b.add(0, 0, 1.0);
        b.add(0, 2, 2.0);
        b.add(1, 1, 3.0);
        b.add(2, 0, 4.0);
        b.add(2, 2, 5.0);
        b.build()
    }

    #[test]
    fn empty_matrix() {
        let m = CsrMatrix::zeros(3, 4);
        assert_eq!(m.nnz(), 0);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 4);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(m.get(i, j), 0.0);
            }
        }
        assert_eq!(m.mul_vec(&[1.0, 2.0, 3.0, 4.0]), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn identity_matrix() {
        let m = CsrMatrix::identity(4);
        assert_eq!(m.nnz(), 4);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(m.get(i, j), expected);
            }
        }
        assert_eq!(m.mul_vec(&[1.0, 2.0, 3.0, 4.0]), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn manual_construction() {
        let m = build_3x3();
        assert_eq!(m.nnz(), 5);

        let expected = [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(m.get(i, j), expected[i][j], "({}, {})", i, j);
            }
        }

        assert_eq!(m.mul_vec(&[1.0, 1.0, 1.0]), vec![3.0, 3.0, 9.0]);

        let dense = m.to_dense();
        for i in 0..3 {
            assert_eq!(dense[i], expected[i].to_vec());
        }
    }

    #[test]
    fn duplicate_entries_are_summed() {
        let mut b = CsrBuilder::new(2, 2);
        b.add(0, 0, 3.0);
        b.add(0, 0, 7.0);
        let m = b.build();
        assert_eq!(m.get(0, 0), 10.0);
        assert_eq!(m.nnz(), 1);
    }

    #[test]
    fn transpose_works() {
        let m = build_3x3();
        let t = m.transpose();

        // transpose of [1 0 2; 0 3 0; 4 0 5] is [1 0 4; 0 3 0; 2 0 5]
        let expected_t = [[1.0, 0.0, 4.0], [0.0, 3.0, 0.0], [2.0, 0.0, 5.0]];
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(t.get(i, j), expected_t[i][j], "({}, {})", i, j);
            }
        }

        // (A^T)^T == A
        let tt = t.transpose();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(tt.get(i, j), m.get(i, j), "({}, {})", i, j);
            }
        }
    }

    #[test]
    fn row_entries_slices() {
        let m = build_3x3();
        let (v0, c0) = m.row_entries(0);
        assert_eq!(v0, &[1.0, 2.0]);
        assert_eq!(c0, &[0, 2]);

        let (v1, c1) = m.row_entries(1);
        assert_eq!(v1, &[3.0]);
        assert_eq!(c1, &[1]);
    }

    #[test]
    fn scale_doubles_values() {
        let mut m = build_3x3();
        m.scale(2.0);
        assert_eq!(m.get(0, 0), 2.0);
        assert_eq!(m.get(0, 2), 4.0);
        assert_eq!(m.get(1, 1), 6.0);
        assert_eq!(m.get(2, 0), 8.0);
        assert_eq!(m.get(2, 2), 10.0);
        assert_eq!(m.nnz(), 5);
    }

    #[test]
    fn banded_tridiagonal() {
        let n = 100;
        let mut b = CsrBuilder::new(n, n);
        for i in 0..n {
            b.add(i, i, 2.0); // diagonal
            if i > 0 {
                b.add(i, i - 1, -1.0); // sub-diagonal
            }
            if i + 1 < n {
                b.add(i, i + 1, -1.0); // super-diagonal
            }
        }
        let m = b.build();
        assert_eq!(m.nnz(), 100 + 99 + 99);

        // y = A * ones. Interior rows: -1 + 2 - 1 = 0. Ends: 2 - 1 = 1.
        let ones = vec![1.0; n];
        let y = m.mul_vec(&ones);
        assert_eq!(y[0], 1.0);
        assert_eq!(y[n - 1], 1.0);
        assert_eq!(y[50], 0.0);
        assert_eq!(y[1], 0.0);
    }

    #[test]
    fn large_sparse_matrix() {
        let n = 1000;
        let mut b = CsrBuilder::new(n, n);
        for k in 0..5000 {
            let i = (k * 7) % n;
            let j = (k * 13) % n;
            let val = ((i * 7 + j * 13) % 1000) as f64 / 1000.0;
            b.add(i, j, val);
        }
        let m = b.build();
        assert!(m.nnz() <= 5000, "nnz {} exceeds entries added", m.nnz());

        let x: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
        let y = m.mul_vec(&x);
        assert_eq!(y.len(), n);
        for v in y {
            assert!(v.is_finite(), "result not finite: {}", v);
        }
    }
}
