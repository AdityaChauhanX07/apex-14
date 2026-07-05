//! Uniform-grid resampling with linear interpolation and gap propagation.

use std::collections::BTreeMap;

use apex_telemetry::ChannelId;

use crate::error::CorrelateError;
use crate::telemetry::{GridKind, Telemetry};

impl Telemetry {
    /// Resample onto a uniform arc-length (`s`) grid of the given `step`
    /// (metres). Requires an `s` channel.
    ///
    /// See [`Telemetry::resample`] for the interpolation and gap semantics.
    pub fn resample_to_s(&self, step: f64, max_gap: f64) -> Result<Telemetry, CorrelateError> {
        self.resample(GridKind::S, step, max_gap)
    }

    /// Resample onto a uniform time (`t`) grid of the given `step` (seconds).
    /// Requires a `t` channel.
    pub fn resample_to_t(&self, step: f64, max_gap: f64) -> Result<Telemetry, CorrelateError> {
        self.resample(GridKind::T, step, max_gap)
    }

    /// Resample every channel onto a uniform grid in `target` coordinates.
    ///
    /// The new grid runs from `axis[0]` in steps of `step` up to `axis[last]`.
    /// Each channel is linearly interpolated onto it. **Gaps stay `NaN`:**
    ///
    /// - If either bracketing source sample is non-finite, the output is `NaN`
    ///   (an input dropout is not interpolated across).
    /// - If the gap between the two bracketing axis samples exceeds `max_gap`,
    ///   the output is `NaN` (no interpolation across a long gap).
    ///
    /// The output axis channel is the exact uniform grid. The source axis must
    /// be finite and strictly increasing.
    pub fn resample(
        &self,
        target: GridKind,
        step: f64,
        max_gap: f64,
    ) -> Result<Telemetry, CorrelateError> {
        if !step.is_finite() || step <= 0.0 {
            return Err(CorrelateError::BadStep(step));
        }
        let axis_id = target.axis_channel();
        let axis = self
            .channel(axis_id)
            .ok_or(CorrelateError::MissingAxis(axis_id.name()))?;

        // Axis must be finite and strictly increasing to bracket-search.
        if axis.len() < 2 || !axis[0].is_finite() {
            return Err(CorrelateError::AxisNotMonotonic(axis_id.name()));
        }
        for w in axis.windows(2) {
            if !w[1].is_finite() || w[1] <= w[0] {
                return Err(CorrelateError::AxisNotMonotonic(axis_id.name()));
            }
        }

        let first = axis[0];
        let last = axis[axis.len() - 1];
        // Number of grid points: inclusive of the endpoint when it lands on the
        // grid. The small epsilon absorbs FP error so an already-uniform axis
        // reproduces exactly (identity).
        let span = last - first;
        let n = (span / step + 1e-9).floor() as usize + 1;
        let grid: Vec<f64> = (0..n).map(|k| first + step * k as f64).collect();

        let mut channels: BTreeMap<ChannelId, Vec<f64>> = BTreeMap::new();
        for (&id, samples) in &self.channels {
            if id == axis_id {
                channels.insert(id, grid.clone());
                continue;
            }
            channels.insert(id, interpolate(axis, samples, &grid, max_gap));
        }
        // The axis channel is always present (grid), even if it wasn't iterated
        // above (it is, since it's a channel) — but guard for safety.
        channels.entry(axis_id).or_insert_with(|| grid.clone());

        Ok(Telemetry {
            grid: target,
            channels,
            metadata: self.metadata.clone(),
        })
    }
}

/// Linear interpolation of `samples` (defined at `axis`) onto `grid`, with gap
/// propagation. `axis` is finite and strictly increasing.
fn interpolate(axis: &[f64], samples: &[f64], grid: &[f64], max_gap: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(grid.len());
    let mut i = 0usize; // lower bracket index, advanced monotonically
    for &x in grid {
        // Advance i so that axis[i] <= x < axis[i+1] (or i is the last usable).
        while i + 1 < axis.len() && axis[i + 1] < x {
            i += 1;
        }
        // Out of range (only possible by FP just past the end): NaN.
        if x < axis[0] || x > axis[axis.len() - 1] {
            out.push(f64::NAN);
            continue;
        }
        let (a_x, b_x) = (axis[i], axis[i + 1]);
        let (a, b) = (samples[i], samples[i + 1]);
        // A grid point that coincides exactly with a source sample takes that
        // sample verbatim — it is not "inside" a gap, so the gap rule and
        // interpolation don't apply (this also makes an already-uniform
        // resample exact).
        if x == a_x {
            out.push(a);
            continue;
        }
        if x == b_x {
            out.push(b);
            continue;
        }
        // Gap too long → do not interpolate across it.
        if b_x - a_x > max_gap {
            out.push(f64::NAN);
            continue;
        }
        // Input dropout on either side → NaN.
        if !a.is_finite() || !b.is_finite() {
            out.push(f64::NAN);
            continue;
        }
        let t = (x - a_x) / (b_x - a_x);
        out.push(a + t * (b - a));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn telem(grid: GridKind, cols: &[(ChannelId, Vec<f64>)]) -> Telemetry {
        Telemetry {
            grid,
            channels: cols.iter().cloned().collect(),
            metadata: Vec::new(),
        }
    }

    #[test]
    fn uniform_signal_is_identity() {
        let s: Vec<f64> = (0..10).map(|k| k as f64 * 2.0).collect();
        let v: Vec<f64> = s.iter().map(|x| 3.0 * x + 1.0).collect();
        let t = telem(
            GridKind::S,
            &[(ChannelId::S, s.clone()), (ChannelId::Speed, v.clone())],
        );
        let r = t.resample_to_s(2.0, 1e9).unwrap();
        let rs = r.channel(ChannelId::S).unwrap();
        let rv = r.channel(ChannelId::Speed).unwrap();
        assert_eq!(rs.len(), s.len());
        for k in 0..s.len() {
            assert!((rs[k] - s[k]).abs() < 1e-9);
            assert!((rv[k] - v[k]).abs() < 1e-9);
        }
    }

    #[test]
    fn midpoint_interpolation() {
        // Resample a step-2 axis onto step-1: midpoints are the averages.
        let s = vec![0.0, 2.0, 4.0];
        let v = vec![0.0, 10.0, 20.0];
        let t = telem(GridKind::S, &[(ChannelId::S, s), (ChannelId::Speed, v)]);
        let r = t.resample_to_s(1.0, 1e9).unwrap();
        let rv = r.channel(ChannelId::Speed).unwrap();
        // grid: 0,1,2,3,4 → 0,5,10,15,20
        assert_eq!(rv.len(), 5);
        let expect = [0.0, 5.0, 10.0, 15.0, 20.0];
        for (got, want) in rv.iter().zip(expect) {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
    }

    #[test]
    fn long_axis_gap_is_not_interpolated() {
        // A big jump in the axis between s=2 and s=20; with max_gap=5 no grid
        // point strictly inside the gap may be interpolated.
        let s = vec![0.0, 1.0, 2.0, 20.0, 21.0];
        let v = vec![0.0, 1.0, 2.0, 20.0, 21.0];
        let t = telem(GridKind::S, &[(ChannelId::S, s), (ChannelId::Speed, v)]);
        let r = t.resample_to_s(1.0, 5.0).unwrap();
        let rs = r.channel(ChannelId::S).unwrap();
        let rv = r.channel(ChannelId::Speed).unwrap();
        for (x, y) in rs.iter().zip(rv) {
            if *x > 2.0 && *x < 20.0 {
                assert!(y.is_nan(), "expected NaN in gap at s={x}, got {y}");
            } else {
                assert!(y.is_finite(), "expected value at s={x}, got {y}");
            }
        }
    }

    #[test]
    fn input_nan_propagates() {
        // A NaN sample at s=2 taints both intervals touching it.
        let s = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let v = vec![0.0, 1.0, f64::NAN, 3.0, 4.0];
        let t = telem(GridKind::S, &[(ChannelId::S, s), (ChannelId::Speed, v)]);
        let r = t.resample_to_s(0.5, 1e9).unwrap();
        let rs = r.channel(ChannelId::S).unwrap();
        let rv = r.channel(ChannelId::Speed).unwrap();
        for (x, y) in rs.iter().zip(rv) {
            // Grid points in (1,3) bracket the NaN on at least one side.
            if *x > 1.0 && *x < 3.0 {
                assert!(y.is_nan(), "expected NaN near s={x}, got {y}");
            }
        }
        // Endpoints away from the NaN interpolate fine.
        assert!((rv[0] - 0.0).abs() < 1e-9);
        assert!((rv.last().unwrap() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_bad_step_and_missing_axis() {
        let s = vec![0.0, 1.0, 2.0];
        let t = telem(GridKind::S, &[(ChannelId::S, s)]);
        assert!(matches!(
            t.resample_to_s(0.0, 1.0),
            Err(CorrelateError::BadStep(_))
        ));
        // No time axis present → resample_to_t fails.
        assert!(matches!(
            t.resample_to_t(0.1, 1.0),
            Err(CorrelateError::MissingAxis(_))
        ));
    }

    #[test]
    fn rejects_nonmonotonic_axis() {
        let s = vec![0.0, 2.0, 1.0]; // decreasing step
        let v = vec![0.0, 1.0, 2.0];
        let t = telem(GridKind::S, &[(ChannelId::S, s), (ChannelId::Speed, v)]);
        assert!(matches!(
            t.resample_to_s(0.5, 1e9),
            Err(CorrelateError::AxisNotMonotonic(_))
        ));
    }
}
