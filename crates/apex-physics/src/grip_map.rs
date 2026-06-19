//! Spatially varying grip across the track surface (rubbered racing line,
//! off-line marbles), sampled on a (distance, lateral offset) grid.

/// Spatially varying grip multiplier across the track surface.
///
/// The track surface is not uniform: the racing line accumulates rubber
/// (higher grip), while the off-line surface has less rubber and possibly
/// debris (lower grip). This map stores a grip multiplier at each point
/// on a (distance, lateral offset) grid.
#[derive(Debug, Clone)]
pub struct GripMap {
    /// Number of stations along the track.
    n_stations: usize,
    /// Number of lateral samples across the track width.
    n_lateral: usize,
    /// Arc length at each station (m).
    stations: Vec<f64>,
    /// Lateral sample positions relative to centerline (m).
    /// These are the same at every station (uniform grid from -max_width to +max_width).
    lateral_positions: Vec<f64>,
    /// Grip multiplier at each grid point.
    /// Indexed as [station_idx * n_lateral + lateral_idx].
    /// 1.0 = baseline grip, >1.0 = rubbered-in, <1.0 = off-line/contaminated.
    grip: Vec<f64>,
}

impl GripMap {
    /// Create a uniform grip map (multiplier = 1.0 everywhere).
    pub fn uniform(track: &apex_track::Track, n_stations: usize, n_lateral: usize) -> Self {
        let max_width = 8.0; // sample from -8m to +8m across the track
        let stations: Vec<f64> = (0..n_stations)
            .map(|i| track.total_length * i as f64 / n_stations as f64)
            .collect();
        let lateral_positions: Vec<f64> = (0..n_lateral)
            .map(|j| -max_width + 2.0 * max_width * j as f64 / (n_lateral - 1).max(1) as f64)
            .collect();
        let grip = vec![1.0; n_stations * n_lateral];

        GripMap {
            n_stations,
            n_lateral,
            stations,
            lateral_positions,
            grip,
        }
    }

    /// Create a dry-condition grip map with a rubbered racing line.
    ///
    /// The racing line (near centerline, n ~ 0) has elevated grip (~1.05-1.10)
    /// from rubber buildup. Off-line grip drops to ~0.85-0.90. Far off-line
    /// (near track edges) drops further to ~0.70-0.80 due to dust and marbles.
    pub fn dry_rubbered(track: &apex_track::Track, n_stations: usize, n_lateral: usize) -> Self {
        let mut map = Self::uniform(track, n_stations, n_lateral);

        for i in 0..n_stations {
            for j in 0..n_lateral {
                let n = map.lateral_positions[j];
                let abs_n = n.abs();

                // Racing line grip profile (Gaussian centered on centerline)
                let racing_line_bonus = 0.10 * (-0.5 * (abs_n / 2.0).powi(2)).exp();

                // Off-line penalty (increases with distance from centerline)
                let off_line_penalty = if abs_n > 3.0 {
                    0.15 * ((abs_n - 3.0) / 5.0).min(1.0)
                } else {
                    0.0
                };

                // Marble zone near track edges
                let edge_penalty = if abs_n > 5.0 {
                    0.10 * ((abs_n - 5.0) / 3.0).min(1.0)
                } else {
                    0.0
                };

                let idx = i * n_lateral + j;
                map.grip[idx] =
                    (1.0 + racing_line_bonus - off_line_penalty - edge_penalty).clamp(0.3, 1.15);
            }
        }

        map
    }

    /// Interpolate the grip multiplier at an arbitrary (s, n) position.
    ///
    /// Uses bilinear interpolation between the four nearest grid points.
    pub fn grip_at(&self, s: f64, n: f64) -> f64 {
        if self.n_stations == 0 || self.n_lateral == 0 {
            return 1.0;
        }

        // Find bounding station indices
        let s_wrapped = s.rem_euclid(
            self.stations.last().copied().unwrap_or(1.0)
                + self.stations.last().copied().unwrap_or(1.0) / self.n_stations as f64,
        );

        let mut si = 0;
        for i in 0..self.n_stations - 1 {
            if (self.stations[i]..self.stations[i + 1]).contains(&s_wrapped) {
                si = i;
                break;
            }
        }
        let si_next = (si + 1) % self.n_stations;
        let s_frac = if self.stations[si_next] > self.stations[si] {
            (s_wrapped - self.stations[si]) / (self.stations[si_next] - self.stations[si])
        } else {
            0.0
        };

        // Find bounding lateral indices
        let mut li = 0;
        for j in 0..self.n_lateral - 1 {
            if (self.lateral_positions[j]..=self.lateral_positions[j + 1]).contains(&n) {
                li = j;
                break;
            }
        }
        // Clamp to grid
        if n <= self.lateral_positions[0] {
            li = 0;
        }
        if n >= *self.lateral_positions.last().unwrap_or(&0.0) {
            li = self.n_lateral.saturating_sub(2);
        }
        let li_next = (li + 1).min(self.n_lateral - 1);
        let n_frac = if self.lateral_positions[li_next] > self.lateral_positions[li] {
            ((n - self.lateral_positions[li])
                / (self.lateral_positions[li_next] - self.lateral_positions[li]))
                .clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Bilinear interpolation
        let g00 = self.grip[si * self.n_lateral + li];
        let g01 = self.grip[si * self.n_lateral + li_next];
        let g10 = self.grip[si_next * self.n_lateral + li];
        let g11 = self.grip[si_next * self.n_lateral + li_next];

        let g0 = g00 + n_frac * (g01 - g00);
        let g1 = g10 + n_frac * (g11 - g10);
        g0 + s_frac * (g1 - g0)
    }

    /// Evolve the grip map to simulate rubber buildup over a session.
    ///
    /// Cars driving on the racing line deposit rubber, increasing grip there.
    /// Off-line grip slowly degrades as rubber particles ("marbles") accumulate.
    ///
    /// Arguments:
    /// - racing_line_n: lateral offset of the racing line at each station (m).
    ///   If shorter than n_stations, the last value is repeated.
    /// - intensity: how much rubber is deposited per call (0.0-1.0).
    ///   Represents the effect of one session of driving.
    pub fn evolve_rubber(&mut self, racing_line_n: &[f64], intensity: f64) {
        for i in 0..self.n_stations {
            let line_n = if i < racing_line_n.len() {
                racing_line_n[i]
            } else {
                racing_line_n.last().copied().unwrap_or(0.0)
            };

            for j in 0..self.n_lateral {
                let n = self.lateral_positions[j];
                let dist_from_line = (n - line_n).abs();
                let idx = i * self.n_lateral + j;

                // Rubber deposit near the racing line (Gaussian)
                let deposit = intensity * 0.02 * (-0.5 * (dist_from_line / 1.5).powi(2)).exp();

                // Marble accumulation far from the line
                let marbles = if dist_from_line > 3.0 {
                    intensity * 0.005 * ((dist_from_line - 3.0) / 5.0).min(1.0)
                } else {
                    0.0
                };

                self.grip[idx] = (self.grip[idx] + deposit - marbles).clamp(0.3, 1.20);
            }
        }
    }

    /// Get the dimensions of the grip map.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.n_stations, self.n_lateral)
    }

    /// Get a reference to the raw grip data.
    pub fn raw_grip(&self) -> &[f64] {
        &self.grip
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_track::{build_track, oval_track, Track};

    fn oval() -> Track {
        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 600);
        build_track("oval", &pts, closed)
    }

    // (a) Uniform grip map.
    #[test]
    fn uniform_is_unity() {
        let track = oval();
        let map = GripMap::uniform(&track, 50, 33);
        assert!(map.raw_grip().iter().all(|&g| (g - 1.0).abs() < 1e-12));
        assert!((map.grip_at(300.0, 0.0) - 1.0).abs() < 1e-9);
        assert!((map.grip_at(750.0, 4.0) - 1.0).abs() < 1e-9);
    }

    // (b) Dry rubbered grip map.
    #[test]
    fn dry_rubbered_profile() {
        let track = oval();
        let map = GripMap::dry_rubbered(&track, 50, 33);
        let s = 300.0;
        assert!(map.grip_at(s, 0.0) > 1.0, "racing line should be rubbered");
        assert!(map.grip_at(s, 6.0) < 1.0, "off-line should be penalized");
        // Monotonic decrease away from the line.
        assert!(map.grip_at(s, 0.0) > map.grip_at(s, 4.0));
        assert!(map.grip_at(s, 4.0) > map.grip_at(s, 7.0));
    }

    // (c) Rubber evolution.
    #[test]
    fn evolve_builds_rubber_on_line() {
        let track = oval();
        let mut map = GripMap::uniform(&track, 50, 33);
        let (n_stations, _) = map.dimensions();
        let line = vec![0.0; n_stations];

        let before_line = map.grip_at(300.0, 0.0);
        let before_off = map.grip_at(300.0, 6.0);
        map.evolve_rubber(&line, 1.0);

        assert!(
            map.grip_at(300.0, 0.0) > before_line,
            "line grip should increase"
        );
        assert!(
            map.grip_at(300.0, 6.0) < before_off,
            "off-line grip should decrease"
        );
    }
}
