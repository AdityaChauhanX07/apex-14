//! Generic C1 cubic-Hermite interpolation on uniform structured grids.
//!
//! This is the interpolation layer the g-g-g envelope stores its boundary
//! radius on. The design constraint is **C1 smoothness of the evaluated
//! quantity**: the free-trajectory OCP constrains against the envelope, and a
//! gradient-based solver needs the constraint's value *and first derivative* to
//! be continuous everywhere, including across grid-cell boundaries. Plain
//! bilinear interpolation (the only gridded interpolant that existed in the tree
//! before this — `MuScaleGrid`, `GripMap`, the collocation `interp` helper) is
//! only C0: its gradient jumps at every cell face, which shows up as chatter in
//! the OCP's KKT system.
//!
//! # What this provides
//!
//! A [`HermiteGrid`] over `D` axes, each a uniformly spaced [`GridAxis`] (an
//! axis may be **periodic**, for the wrap-around acceleration-plane angle). It
//! evaluates by tensor product of 1-D **cubic Hermite** segments whose node
//! tangents are estimated by finite differences that are **exact for cubics**
//! (a centred 5-point stencil in the interior, one-sided 4-point stencils at the
//! edges). Two properties fall out and are unit-tested:
//!
//! - **Exact reproduction of cubics.** On each cell a cubic Hermite segment is
//!   the unique cubic matching the value and first derivative at both ends. If
//!   the sampled function is a (tensor-product) cubic, the FD tangents are exact,
//!   so the segment reproduces it bit-for-tolerance. See
//!   [`tests::reproduces_cubic_exactly_1d`] / `_3d`.
//! - **C1 continuity.** A node's tangent is a single value computed once from a
//!   fixed stencil, shared by the two cells that meet at it — so value *and*
//!   first derivative match across every cell face. See
//!   [`tests::c1_across_cell_boundaries`].
//!
//! # Derivative path
//!
//! Evaluation is generic over [`Float`], so passing [`Dual`](crate::Dual)
//! coordinates yields the interpolated value **and** its derivative with respect
//! to the seeded input in one pass — the AD path the OCP consumes. The node
//! values stay `f64`; only the query coordinate carries a derivative.
//!
//! # Overshoot
//!
//! Cubic Hermite is not monotone-preserving: between two samples the segment can
//! overshoot beyond the local `[min, max]` of the surrounding nodes when the data
//! has a sharp knee (Runge-like wiggle). This is inherent to any C1-or-higher
//! polynomial interpolant and is documented, not hidden — see
//! [`tests::overshoot_is_bounded_and_local`], which pins the worst-case
//! overshoot on a step-like profile and confirms it stays local (does not
//! propagate past the adjacent cells). The envelope's boundary radius is smooth,
//! so this regime is not exercised in practice.

use crate::float::Float;

/// One axis of a structured grid: `n` uniformly spaced sample coordinates.
///
/// Node `i` (for `i` in `0..n`) sits at `start + i*step`. A **non-periodic**
/// axis spans the closed interval `[start, start + (n-1)*step]`; queries outside
/// it are clamped (constant extrapolation). A **periodic** axis has period
/// `n*step`: node `n` coincides with node `0`, queries wrap, and the tangent
/// stencil wraps too — giving a genuinely C1 closed loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridAxis {
    /// Coordinate of node 0.
    pub start: f64,
    /// Spacing between adjacent nodes (`> 0`).
    pub step: f64,
    /// Number of stored nodes.
    pub n: usize,
    /// Whether the axis wraps (period = `n*step`).
    pub periodic: bool,
}

impl GridAxis {
    /// A non-periodic axis of `n` samples spanning the closed interval
    /// `[start, end]`. Requires `n >= 4` (the cubic stencils need four nodes) and
    /// `end > start`.
    pub fn uniform(start: f64, end: f64, n: usize) -> GridAxis {
        assert!(n >= 4, "cubic Hermite axis needs at least 4 nodes, got {n}");
        assert!(end > start, "axis end {end} must exceed start {start}");
        GridAxis {
            start,
            step: (end - start) / (n - 1) as f64,
            n,
            periodic: false,
        }
    }

    /// A periodic axis of `n` samples covering `[start, start + period)`, with
    /// node `n` wrapping to node `0`. Requires `n >= 5` (so the centred 5-point
    /// stencil references five distinct nodes) and `period > 0`.
    pub fn periodic(start: f64, period: f64, n: usize) -> GridAxis {
        assert!(
            n >= 5,
            "periodic cubic Hermite axis needs at least 5 nodes, got {n}"
        );
        assert!(period > 0.0, "axis period {period} must be positive");
        GridAxis {
            start,
            step: period / n as f64,
            n,
            periodic: true,
        }
    }

    /// Coordinate of node `i`.
    #[inline]
    pub fn coord(&self, i: usize) -> f64 {
        self.start + i as f64 * self.step
    }

    /// Number of nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Always non-empty (`n >= 4`); provided to satisfy the `len`/`is_empty` lint.
    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// The four coefficients (numerator over 6) of the equispaced 4-point cubic
/// first-derivative stencil, indexed by the local position `p` of the node
/// within the window `[w, w+1, w+2, w+3]`. Each row is exact for cubics.
const CUBIC_D4: [[f64; 4]; 4] = [
    [-11.0, 18.0, -9.0, 2.0], // derivative at window node 0
    [-2.0, -3.0, 6.0, -1.0],  // ... at node 1
    [1.0, -6.0, 3.0, 2.0],    // ... at node 2
    [-2.0, 9.0, -18.0, 11.0], // ... at node 3
];

/// The tangent stencil at node `i` of `axis`: `(node_index, coefficient)` pairs
/// giving `df/di` (the derivative with respect to the integer node index, i.e.
/// `step * df/dx`) as a linear combination of node values, exact for cubics.
///
/// Interior nodes of a non-periodic axis use the centred 5-point stencil
/// `(1, -8, 0, 8, -1)/12`; edge nodes use the one-sided 4-point [`CUBIC_D4`]
/// rows. Periodic axes use the wrapped centred 5-point stencil at every node.
/// Every node's stencil is a fixed function of `(axis, i)`, so the two cells
/// meeting at node `i` share the identical tangent — the basis of C1 continuity.
fn tangent_stencil(axis: &GridAxis, i: usize) -> Vec<(usize, f64)> {
    let n = axis.n;
    if axis.periodic {
        // Wrapped centred 5-point: exact for cubics, symmetric.
        let wrap = |k: isize| k.rem_euclid(n as isize) as usize;
        return vec![
            (wrap(i as isize - 2), 1.0 / 12.0),
            (wrap(i as isize - 1), -8.0 / 12.0),
            (wrap(i as isize + 1), 8.0 / 12.0),
            (wrap(i as isize + 2), -1.0 / 12.0),
        ];
    }
    if i >= 2 && i + 2 < n {
        // Centred 5-point on [i-2 .. i+2].
        vec![
            (i - 2, 1.0 / 12.0),
            (i - 1, -8.0 / 12.0),
            (i + 1, 8.0 / 12.0),
            (i + 2, -1.0 / 12.0),
        ]
    } else {
        // One-sided 4-point window nearest-centred on i.
        let w = (i.max(1) - 1).min(n - 4);
        let p = i - w; // local position 0..=3
        (0..4).map(|k| (w + k, CUBIC_D4[p][k] / 6.0)).collect()
    }
}

/// Locate the query on one axis: returns the cell's lower node index `i`, its
/// upper node index `j` (= `i+1`, wrapped for periodic axes), and the local
/// parameter `t in [0, 1]` as a `Float` carrying the query's derivative.
fn locate<T: Float>(axis: &GridAxis, x: T) -> (usize, usize, T) {
    let n = axis.n;
    if axis.periodic {
        let period = axis.step * n as f64;
        let xr = x.real_value();
        // Shift into [start, start+period) without dropping the derivative.
        let k = ((xr - axis.start) / period).floor();
        let x_red = x - k * period;
        let mut i = ((x_red.real_value() - axis.start) / axis.step).floor() as isize;
        i = i.rem_euclid(n as isize);
        let iu = i as usize;
        let t = (x_red - axis.coord(iu)) / axis.step;
        return (iu, (iu + 1) % n, t);
    }
    let end = axis.coord(n - 1);
    let xr = x.real_value();
    if xr <= axis.start {
        // Constant extrapolation below the range.
        return (0, 1, T::zero());
    }
    if xr >= end {
        return (n - 2, n - 1, T::one());
    }
    let i = (((xr - axis.start) / axis.step).floor() as usize).min(n - 2);
    let t = (x - axis.coord(i)) / axis.step;
    (i, i + 1, t)
}

/// The Hermite blend weights over one axis: `(node_index, weight)` pairs whose
/// weighted sum of node values gives the 1-D interpolated value. Repeated
/// indices are permitted (their weights simply add) — the tangent stencils and
/// the cell endpoints may reference the same node, and downstream accumulation
/// is linear, so no merging is needed.
fn axis_weights<T: Float>(axis: &GridAxis, x: T) -> Vec<(usize, T)> {
    let (i, j, t) = locate(axis, x);
    // Cubic Hermite basis on the unit cell (tangents are df/di).
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = t3 * 2.0 - t2 * 3.0 + 1.0;
    let h10 = t3 - t2 * 2.0 + t;
    let h01 = t3 * (-2.0) + t2 * 3.0;
    let h11 = t3 - t2;

    let mut out: Vec<(usize, T)> = Vec::with_capacity(10);
    out.push((i, h00));
    out.push((j, h01));
    for (idx, c) in tangent_stencil(axis, i) {
        out.push((idx, h10 * c));
    }
    for (idx, c) in tangent_stencil(axis, j) {
        out.push((idx, h11 * c));
    }
    out
}

/// A C1 cubic-Hermite interpolant over a `D`-axis uniform structured grid.
///
/// Node values are stored **row-major** over the axes: the flat index of node
/// `(k_0, .., k_{D-1})` is `sum_d k_d * stride_d`, with `stride_{D-1} = 1` and
/// `stride_d = stride_{d+1} * axes[d+1].n`. Construct with [`HermiteGrid::new`],
/// which validates the value-vector length.
#[derive(Debug, Clone)]
pub struct HermiteGrid {
    axes: Vec<GridAxis>,
    strides: Vec<usize>,
    values: Vec<f64>,
}

impl HermiteGrid {
    /// Build a grid from its axes and row-major node values. Panics if `values`
    /// does not have exactly `prod(axis.n)` entries.
    pub fn new(axes: Vec<GridAxis>, values: Vec<f64>) -> HermiteGrid {
        assert!(!axes.is_empty(), "HermiteGrid needs at least one axis");
        let total: usize = axes.iter().map(|a| a.n).product();
        assert_eq!(
            values.len(),
            total,
            "value count {} != product of axis lengths {}",
            values.len(),
            total
        );
        let d = axes.len();
        let mut strides = vec![1usize; d];
        for k in (0..d - 1).rev() {
            strides[k] = strides[k + 1] * axes[k + 1].n;
        }
        HermiteGrid {
            axes,
            strides,
            values,
        }
    }

    /// The grid axes.
    pub fn axes(&self) -> &[GridAxis] {
        &self.axes
    }

    /// Raw row-major node values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Evaluate at `coords` (one coordinate per axis), generic over the numeric
    /// type. Passing [`Dual`](crate::Dual) coordinates returns the value with the
    /// derivative w.r.t. the seeded coordinate. Panics if `coords.len()` differs
    /// from the axis count.
    pub fn eval<T: Float>(&self, coords: &[T]) -> T {
        assert_eq!(
            coords.len(),
            self.axes.len(),
            "expected {} coordinates, got {}",
            self.axes.len(),
            coords.len()
        );
        let per_axis: Vec<Vec<(usize, T)>> = self
            .axes
            .iter()
            .zip(coords)
            .map(|(ax, &x)| axis_weights(ax, x))
            .collect();
        self.accumulate(&per_axis, 0, 0, T::one())
    }

    /// Convenience `f64` evaluation.
    pub fn eval_f64(&self, coords: &[f64]) -> f64 {
        self.eval(coords)
    }

    /// Recursive tensor-product accumulation over the per-axis weight lists.
    fn accumulate<T: Float>(
        &self,
        per_axis: &[Vec<(usize, T)>],
        axis: usize,
        base: usize,
        wacc: T,
    ) -> T {
        if axis == self.axes.len() {
            return wacc * T::from_f64(self.values[base]);
        }
        let stride = self.strides[axis];
        let mut sum = T::zero();
        for &(idx, w) in &per_axis[axis] {
            sum = sum + self.accumulate(per_axis, axis + 1, base + idx * stride, wacc * w);
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dual;

    // --- 1-D exact cubic reproduction ---

    #[test]
    fn reproduces_cubic_exactly_1d() {
        // f(x) = 2x^3 - 3x^2 + 0.5x - 1 sampled on a uniform grid; the Hermite
        // interpolant with cubic-exact tangents must reproduce it to tolerance
        // at off-node points (interior and near both edges).
        let f = |x: f64| 2.0 * x.powi(3) - 3.0 * x.powi(2) + 0.5 * x - 1.0;
        let axis = GridAxis::uniform(-2.0, 3.0, 12);
        let values: Vec<f64> = (0..axis.n).map(|i| f(axis.coord(i))).collect();
        let grid = HermiteGrid::new(vec![axis], values);
        for &x in &[-1.93, -1.5, 0.07, 0.5, 1.234, 2.4, 2.97] {
            let got = grid.eval_f64(&[x]);
            assert!((got - f(x)).abs() < 1e-9, "x={x}: got {got}, want {}", f(x));
        }
    }

    #[test]
    fn reproduces_linear_exactly_and_at_nodes() {
        let f = |x: f64| 3.0 * x + 7.0;
        let axis = GridAxis::uniform(0.0, 10.0, 6);
        let values: Vec<f64> = (0..axis.n).map(|i| f(axis.coord(i))).collect();
        let grid = HermiteGrid::new(vec![axis], values.clone());
        // exact at nodes
        for (i, &val) in values.iter().enumerate() {
            assert!((grid.eval_f64(&[axis.coord(i)]) - val).abs() < 1e-12);
        }
        // exact between nodes
        for &x in &[0.3, 4.4, 7.7, 9.1] {
            assert!((grid.eval_f64(&[x]) - f(x)).abs() < 1e-10);
        }
    }

    // --- derivative path (Dual) ---

    #[test]
    fn dual_derivative_matches_analytic_cubic() {
        // For a reproduced cubic, the Dual derivative equals the analytic one.
        let f = |x: f64| x.powi(3) - 2.0 * x;
        let df = |x: f64| 3.0 * x.powi(2) - 2.0;
        let axis = GridAxis::uniform(-3.0, 3.0, 16);
        let values: Vec<f64> = (0..axis.n).map(|i| f(axis.coord(i))).collect();
        let grid = HermiteGrid::new(vec![axis], values);
        for &x in &[-2.1, -0.4, 0.9, 2.2] {
            let r = grid.eval(&[Dual::variable(x)]);
            assert!((r.real - f(x)).abs() < 1e-9, "value x={x}");
            assert!(
                (r.dual - df(x)).abs() < 1e-7,
                "deriv x={x}: {} vs {}",
                r.dual,
                df(x)
            );
        }
    }

    #[test]
    fn dual_derivative_matches_central_difference() {
        // On a non-polynomial function the Dual derivative must match a central
        // finite difference of the interpolant itself.
        let f = |x: f64| (0.7 * x).sin() + 0.2 * x;
        let axis = GridAxis::uniform(0.0, 6.0, 40);
        let values: Vec<f64> = (0..axis.n).map(|i| f(axis.coord(i))).collect();
        let grid = HermiteGrid::new(vec![axis], values);
        let h = 1e-6;
        for &x in &[0.5, 1.7, 3.3, 5.1] {
            let r = grid.eval(&[Dual::variable(x)]);
            let fd = (grid.eval_f64(&[x + h]) - grid.eval_f64(&[x - h])) / (2.0 * h);
            assert!(
                (r.dual - fd).abs() < 1e-5,
                "x={x}: dual {} vs fd {}",
                r.dual,
                fd
            );
        }
    }

    // --- C1 continuity across cell boundaries ---

    #[test]
    fn c1_across_cell_boundaries() {
        // Value and first derivative must be continuous approaching each interior
        // node from the left and from the right.
        let f = |x: f64| (0.5 * x).cos() + 0.1 * x * x;
        let axis = GridAxis::uniform(0.0, 8.0, 20);
        let values: Vec<f64> = (0..axis.n).map(|i| f(axis.coord(i))).collect();
        let grid = HermiteGrid::new(vec![axis], values);
        for i in 1..axis.n - 1 {
            let xc = axis.coord(i);
            let eps = axis.step * 1e-5;
            let left = grid.eval(&[Dual::variable(xc - eps)]);
            let right = grid.eval(&[Dual::variable(xc + eps)]);
            // Value continuity: the two one-sided values differ only by the true
            // slope over the 2*eps gap (no discontinuity). Scale the tolerance to
            // that finite offset rather than expecting exact equality.
            let slope_bound = 2.0; // |f'| stays well under this on the sample
            assert!(
                (left.real - right.real).abs() < slope_bound * 2.0 * eps + 1e-9,
                "value jump at node {i}: {} vs {}",
                left.real,
                right.real
            );
            // Derivative continuity is the substantive C1 property: the left- and
            // right-cell derivatives at the shared node must agree.
            assert!(
                (left.dual - right.dual).abs() < 1e-3,
                "derivative jump at node {i}: {} vs {}",
                left.dual,
                right.dual
            );
        }
    }

    // --- periodic axis ---

    #[test]
    fn periodic_wraps_c1() {
        use std::f64::consts::PI;
        // A periodic function on [0, 2pi): value & derivative must be continuous
        // across the wrap seam at 0 == 2pi.
        let f = |x: f64| (x).sin() + 0.5 * (2.0 * x).cos();
        let df = |x: f64| (x).cos() - (2.0 * x).sin();
        let n = 64;
        let axis = GridAxis::periodic(0.0, 2.0 * PI, n);
        let values: Vec<f64> = (0..n).map(|i| f(axis.coord(i))).collect();
        let grid = HermiteGrid::new(vec![axis], values);
        // continuity across the seam
        let eps = 1e-5;
        let below = grid.eval(&[Dual::variable(2.0 * PI - eps)]);
        let above = grid.eval(&[Dual::variable(eps)]); // wraps
        assert!((below.real - above.real).abs() < 1e-4, "seam value");
        assert!((below.dual - above.dual).abs() < 1e-3, "seam derivative");
        // accuracy at off-node points, and wrap-invariance
        for &x in &[0.3, 1.9, 4.2, 5.9] {
            let got = grid.eval_f64(&[x]);
            assert!((got - f(x)).abs() < 1e-3, "x={x}: {got} vs {}", f(x));
            // adding a full period gives the identical result
            let wrapped = grid.eval_f64(&[x + 2.0 * PI]);
            assert!((got - wrapped).abs() < 1e-9, "wrap invariance x={x}");
            // derivative sanity
            let r = grid.eval(&[Dual::variable(x)]);
            assert!((r.dual - df(x)).abs() < 5e-2, "x={x} deriv");
        }
    }

    // --- N-D tensor product ---

    #[test]
    fn reproduces_cubic_exactly_3d() {
        // Tensor-product cubic (includes cross terms) must be reproduced exactly.
        let f = |x: f64, y: f64, z: f64| {
            x.powi(3) - 2.0 * x * y + 0.5 * y.powi(2) * z + z.powi(3) - x * y * z
        };
        let ax = GridAxis::uniform(-1.0, 2.0, 8);
        let ay = GridAxis::uniform(0.0, 3.0, 7);
        let az = GridAxis::uniform(-2.0, 1.0, 6);
        let mut values = Vec::with_capacity(ax.n * ay.n * az.n);
        for i in 0..ax.n {
            for j in 0..ay.n {
                for k in 0..az.n {
                    values.push(f(ax.coord(i), ay.coord(j), az.coord(k)));
                }
            }
        }
        let grid = HermiteGrid::new(vec![ax, ay, az], values);
        for &(x, y, z) in &[
            (0.3, 1.1, -0.7),
            (-0.9, 2.4, 0.5),
            (1.7, 0.2, -1.8),
            (1.95, 2.9, 0.95),
        ] {
            let got = grid.eval_f64(&[x, y, z]);
            assert!(
                (got - f(x, y, z)).abs() < 1e-7,
                "({x},{y},{z}): {got} vs {}",
                f(x, y, z)
            );
        }
    }

    #[test]
    fn nd_partial_derivatives_via_dual() {
        // Seeding one axis with a Dual variable yields that partial derivative.
        let f = |x: f64, y: f64| x.powi(3) + x * y.powi(2);
        let dfdx = |x: f64, y: f64| 3.0 * x.powi(2) + y.powi(2);
        let dfdy = |x: f64, y: f64| 2.0 * x * y;
        let ax = GridAxis::uniform(0.0, 3.0, 10);
        let ay = GridAxis::uniform(-2.0, 2.0, 10);
        let mut values = Vec::new();
        for i in 0..ax.n {
            for j in 0..ay.n {
                values.push(f(ax.coord(i), ay.coord(j)));
            }
        }
        let grid = HermiteGrid::new(vec![ax, ay], values);
        let (x, y) = (1.3, 0.6);
        let rx = grid.eval(&[Dual::variable(x), Dual::constant(y)]);
        let ry = grid.eval(&[Dual::constant(x), Dual::variable(y)]);
        assert!((rx.dual - dfdx(x, y)).abs() < 1e-6, "d/dx");
        assert!((ry.dual - dfdy(x, y)).abs() < 1e-6, "d/dy");
    }

    // --- overshoot characterization ---

    #[test]
    fn overshoot_is_bounded_and_local() {
        // A step-like profile is the worst case for a C1 polynomial interpolant.
        // We characterize (do not eliminate) the overshoot: it is modest and
        // confined to the cells adjacent to the step; far from the step the
        // interpolant sits inside the data range.
        let axis = GridAxis::uniform(0.0, 11.0, 12);
        let values: Vec<f64> = (0..12).map(|i| if i < 6 { 0.0 } else { 1.0 }).collect();
        let grid = HermiteGrid::new(vec![axis], values);
        let mut max_over = 0.0_f64;
        let mut max_under = 0.0_f64;
        let mut i = 0.0;
        while i <= 11.0 {
            let y = grid.eval_f64(&[i]);
            max_over = max_over.max(y - 1.0);
            max_under = max_under.min(y);
            i += 0.01;
        }
        // Documented worst-case bounds for this step (Catmull-Rom-class wiggle).
        assert!(max_over < 0.1, "overshoot above 1.0 was {max_over}");
        assert!(max_under > -0.1, "undershoot below 0.0 was {max_under}");
        // Locality: two cells before the step (x <= 3) the value is ~0.
        assert!(
            grid.eval_f64(&[3.0]).abs() < 1e-6,
            "overshoot leaked far from step"
        );
        // two cells after the step (x >= 8) the value is ~1.
        assert!(
            (grid.eval_f64(&[8.0]) - 1.0).abs() < 1e-6,
            "overshoot leaked far from step"
        );
    }

    // --- validation ---

    #[test]
    #[should_panic(expected = "value count")]
    fn rejects_wrong_value_count() {
        let ax = GridAxis::uniform(0.0, 1.0, 4);
        let ay = GridAxis::uniform(0.0, 1.0, 4);
        HermiteGrid::new(vec![ax, ay], vec![0.0; 15]); // should be 16
    }

    #[test]
    #[should_panic(expected = "at least 4 nodes")]
    fn rejects_short_axis() {
        GridAxis::uniform(0.0, 1.0, 3);
    }
}
