//! Schema-attached `(station, lateral)` grip-multiplier grid — the
//! `mu_scale(s, n)` mechanism.
//!
//! [`MuScaleGrid`] is track **schema** data: it round-trips through the v2
//! JSON file and lives on a [`crate::Ribbon3d`]. This is distinct from
//! `apex_physics::GripMap`, a runtime-only (non-serializable) grid built from
//! a `Track` for weather/rubber-buildup simulation — the two share the same
//! bilinear-interpolation shape but not the data or its lifecycle. A future
//! consolidation could unify them; not attempted here.
//!
//! QSS never reads this grid directly (see `apex_physics::qss::qss_lap_sim_3d_with_grip`) —
//! callers sample [`MuScaleGrid::mu_at`] themselves and hand QSS a plain
//! per-station multiplier vector. This is deliberate: a driven-line QSS run
//! operates on a *reparameterized* ribbon whose own `(s, n)` frame is not the
//! original centerline's, so a QSS-internal lookup on its own ribbon argument
//! would silently sample the wrong grid location.

/// A grip multiplier sampled on a `(station, lateral)` grid, bilinearly
/// interpolated. `1.0` = baseline grip; absent (the common case) means every
/// query returns `1.0` with no grid stored at all.
#[derive(Debug, Clone)]
pub struct MuScaleGrid {
    /// Arc-length station of each grid row (m), strictly increasing,
    /// `stations[0] == 0.0`.
    pub(crate) stations: Vec<f64>,
    /// Lateral offset of each grid column (m), strictly increasing,
    /// left-positive (matches the ribbon frame's `n`, see `docs/math/track3d.md` §1).
    pub(crate) lateral: Vec<f64>,
    /// Grip multiplier at each grid point, row-major:
    /// `values[station_idx * lateral.len() + lateral_idx]`.
    pub(crate) values: Vec<f64>,
}

impl MuScaleGrid {
    /// Build a grid, validating shape and monotonicity.
    ///
    /// Errors if either axis has fewer than 2 samples, either axis is not
    /// strictly increasing, `stations[0] != 0.0`, or `values.len() !=
    /// stations.len() * lateral.len()`.
    pub fn new(stations: Vec<f64>, lateral: Vec<f64>, values: Vec<f64>) -> Result<Self, String> {
        if stations.len() < 2 {
            return Err(format!(
                "mu_scale_grid: need >= 2 stations, got {}",
                stations.len()
            ));
        }
        if lateral.len() < 2 {
            return Err(format!(
                "mu_scale_grid: need >= 2 lateral samples, got {}",
                lateral.len()
            ));
        }
        if stations[0] != 0.0 {
            return Err(format!(
                "mu_scale_grid: stations[0] must be 0.0, got {}",
                stations[0]
            ));
        }
        if !stations.windows(2).all(|w| w[1] > w[0]) {
            return Err("mu_scale_grid: stations must be strictly increasing".into());
        }
        if !lateral.windows(2).all(|w| w[1] > w[0]) {
            return Err("mu_scale_grid: lateral must be strictly increasing".into());
        }
        if values.len() != stations.len() * lateral.len() {
            return Err(format!(
                "mu_scale_grid: values.len() {} != stations.len() {} * lateral.len() {}",
                values.len(),
                stations.len(),
                lateral.len()
            ));
        }
        Ok(MuScaleGrid {
            stations,
            lateral,
            values,
        })
    }

    /// Bilinearly interpolate the grip multiplier at `(s, n)`.
    ///
    /// `s` wraps cyclically on a closed ribbon (`closed = true`, bracketing
    /// across the seam using `total_length` as the implicit upper station);
    /// on an open ribbon it clamps to `[0, total_length]`. `n` clamps to the
    /// grid's lateral extent (no extrapolation past the edge columns).
    pub fn mu_at(&self, s: f64, n: f64, total_length: f64, closed: bool) -> f64 {
        let ns = self.stations.len();
        let nl = self.lateral.len();

        let s = if closed {
            if total_length > 0.0 {
                s.rem_euclid(total_length)
            } else {
                0.0
            }
        } else {
            s.clamp(self.stations[0], self.stations[ns - 1])
        };

        // Bracket the station: [si, si_next) with si_next wrapping to 0 (closed)
        // or clamped to the last index (open).
        let pp = self.stations.partition_point(|&x| x <= s);
        let si = pp.saturating_sub(1).min(ns - 1);
        let (si_next, s_upper) = if si + 1 < ns {
            (si + 1, self.stations[si + 1])
        } else if closed {
            (0, total_length.max(self.stations[si]))
        } else {
            (si, self.stations[si])
        };
        let s_span = s_upper - self.stations[si];
        let s_frac = if s_span > 0.0 {
            ((s - self.stations[si]) / s_span).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Bracket the lateral column, clamped at the edges.
        let n_clamped = n.clamp(self.lateral[0], self.lateral[nl - 1]);
        let pp_n = self.lateral.partition_point(|&x| x <= n_clamped);
        let li = pp_n.saturating_sub(1).min(nl - 2);
        let li_next = li + 1;
        let l_span = self.lateral[li_next] - self.lateral[li];
        let n_frac = if l_span > 0.0 {
            ((n_clamped - self.lateral[li]) / l_span).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let at = |si: usize, li: usize| -> f64 { self.values[si * nl + li] };
        let g00 = at(si, li);
        let g01 = at(si, li_next);
        let g10 = at(si_next, li);
        let g11 = at(si_next, li_next);

        let g0 = g00 + n_frac * (g01 - g00);
        let g1 = g10 + n_frac * (g11 - g10);
        g0 + s_frac * (g1 - g0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corners_grid() -> MuScaleGrid {
        // 2x2 grid: stations [0, 100], lateral [-5, 5].
        // values: (s=0,n=-5)=1.0  (s=0,n=5)=2.0
        //         (s=100,n=-5)=3.0 (s=100,n=5)=4.0
        MuScaleGrid::new(vec![0.0, 100.0], vec![-5.0, 5.0], vec![1.0, 2.0, 3.0, 4.0]).unwrap()
    }

    #[test]
    fn corners_exact() {
        let g = corners_grid();
        assert_eq!(g.mu_at(0.0, -5.0, 100.0, false), 1.0);
        assert_eq!(g.mu_at(0.0, 5.0, 100.0, false), 2.0);
        assert_eq!(g.mu_at(100.0, -5.0, 100.0, false), 3.0);
        assert_eq!(g.mu_at(100.0, 5.0, 100.0, false), 4.0);
    }

    #[test]
    fn midpoint_is_average_of_corners() {
        let g = corners_grid();
        let mid = g.mu_at(50.0, 0.0, 100.0, false);
        assert!((mid - 2.5).abs() < 1e-12, "mid {mid}");
    }

    #[test]
    fn edges_clamp_not_extrapolate() {
        let g = corners_grid();
        // Lateral beyond the grid clamps to the edge column.
        assert_eq!(
            g.mu_at(0.0, -50.0, 100.0, false),
            g.mu_at(0.0, -5.0, 100.0, false)
        );
        assert_eq!(
            g.mu_at(0.0, 50.0, 100.0, false),
            g.mu_at(0.0, 5.0, 100.0, false)
        );
        // Station beyond range clamps on an open ribbon.
        assert_eq!(
            g.mu_at(500.0, 0.0, 100.0, false),
            g.mu_at(100.0, 0.0, 100.0, false)
        );
        assert_eq!(
            g.mu_at(-500.0, 0.0, 100.0, false),
            g.mu_at(0.0, 0.0, 100.0, false)
        );
    }

    #[test]
    fn wraparound_in_s_on_closed_ribbon() {
        // 3-station ring: 0, 40, 80, with total_length 120 (wraps 80->120->0).
        let g = MuScaleGrid::new(
            vec![0.0, 40.0, 80.0],
            vec![0.0, 1.0],
            vec![
                1.0, 1.0, // s=0
                2.0, 2.0, // s=40
                3.0, 3.0, // s=80
            ],
        )
        .unwrap();
        // Halfway between station 80 and the wrapped station 0 (at s=120) is s=100.
        let mid = g.mu_at(100.0, 0.0, 120.0, true);
        assert!((mid - 2.0).abs() < 1e-12, "wrap mid {mid}"); // (3.0+1.0)/2
                                                              // s slightly past total_length wraps back near station 0.
        let wrapped = g.mu_at(125.0, 0.0, 120.0, true);
        let near_zero = g.mu_at(5.0, 0.0, 120.0, true);
        assert!((wrapped - near_zero).abs() < 1e-9);
    }

    #[test]
    fn uniform_grid_returns_exact_constant() {
        // A constant grid must return the EXACT f64 constant everywhere,
        // bit-for-bit — this is what makes an explicit all-1.0 grid bitwise
        // equivalent to no grid at all downstream in QSS.
        let g =
            MuScaleGrid::new(vec![0.0, 10.0, 20.0], vec![-3.0, 0.0, 3.0], vec![1.0; 9]).unwrap();
        for &s in &[0.0, 3.3, 10.0, 15.7, 20.0] {
            for &n in &[-3.0, -1.2, 0.0, 2.5, 3.0] {
                assert_eq!(g.mu_at(s, n, 20.0, false).to_bits(), 1.0_f64.to_bits());
                assert_eq!(g.mu_at(s, n, 20.0, true).to_bits(), 1.0_f64.to_bits());
            }
        }
    }

    #[test]
    fn rejects_bad_shapes() {
        assert!(MuScaleGrid::new(vec![0.0], vec![0.0, 1.0], vec![1.0, 1.0]).is_err());
        assert!(MuScaleGrid::new(vec![0.0, 1.0], vec![0.0], vec![1.0, 1.0]).is_err());
        assert!(MuScaleGrid::new(vec![1.0, 2.0], vec![0.0, 1.0], vec![1.0; 4]).is_err());
        assert!(MuScaleGrid::new(vec![0.0, 1.0, 0.5], vec![0.0, 1.0], vec![1.0; 6]).is_err());
        assert!(MuScaleGrid::new(vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0; 4]).is_err());
        assert!(MuScaleGrid::new(vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0; 3]).is_err());
    }
}
