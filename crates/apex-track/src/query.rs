//! Arc-length queries and interpolation on a [`Track`].

use crate::builder::normalize_angle;
use crate::types::Track;

impl Track {
    /// Finds the segment index and interpolation fraction for arc length `s`.
    ///
    /// Returns `(index, fraction)` where `fraction` is in `[0, 1)` for
    /// interpolation between `segments[index]` and the following segment
    /// (wrapping to `segments[0]` on closed tracks). For closed tracks, `s` is
    /// taken modulo `total_length`; for open tracks it is clamped.
    pub fn locate(&self, s: f64) -> (usize, f64) {
        let n = self.segments.len();
        let s = if self.is_closed {
            s.rem_euclid(self.total_length)
        } else {
            s.clamp(0.0, self.total_length)
        };

        // Largest index with segments[index].s <= s.
        let pp = self.segments.partition_point(|seg| seg.s <= s);
        let idx = pp.saturating_sub(1);

        let seg_s = self.segments[idx].s;
        let upper_s = if idx + 1 < n {
            self.segments[idx + 1].s
        } else if self.is_closed {
            self.total_length
        } else {
            // open track at the very end
            return (n - 1, 0.0);
        };

        let span = upper_s - seg_s;
        let frac = if span > 0.0 { (s - seg_s) / span } else { 0.0 };
        (idx, frac)
    }

    /// The index following `idx` for interpolation (wraps on closed tracks).
    fn upper_index(&self, idx: usize) -> usize {
        let n = self.segments.len();
        if self.is_closed {
            (idx + 1) % n
        } else {
            (idx + 1).min(n - 1)
        }
    }

    /// Linearly interpolates curvature at arc length `s`.
    pub fn curvature_at(&self, s: f64) -> f64 {
        let (i, f) = self.locate(s);
        let j = self.upper_index(i);
        let a = self.segments[i].curvature;
        let b = self.segments[j].curvature;
        a + f * (b - a)
    }

    /// Interpolates heading at arc length `s`, accounting for angle wrap.
    pub fn heading_at(&self, s: f64) -> f64 {
        let (i, f) = self.locate(s);
        let j = self.upper_index(i);
        let a = self.segments[i].heading;
        let b = self.segments[j].heading;
        let diff = normalize_angle(b - a);
        normalize_angle(a + f * diff)
    }

    /// Linearly interpolates the world position `(x, y)` at arc length `s`.
    pub fn position_at(&self, s: f64) -> (f64, f64) {
        let (i, f) = self.locate(s);
        let j = self.upper_index(i);
        let a = &self.segments[i];
        let b = &self.segments[j];
        (a.x + f * (b.x - a.x), a.y + f * (b.y - a.y))
    }

    /// Linearly interpolates the half-widths `(width_left, width_right)` at
    /// arc length `s`.
    pub fn width_at(&self, s: f64) -> (f64, f64) {
        let (i, f) = self.locate(s);
        let j = self.upper_index(i);
        let a = &self.segments[i];
        let b = &self.segments[j];
        (
            a.width_left + f * (b.width_left - a.width_left),
            a.width_right + f * (b.width_right - a.width_right),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::builder::build_track;
    use crate::generators::{circle_track, oval_track};
    use std::f64::consts::PI;

    #[test]
    fn circle_track_geometry() {
        let radius = 100.0;
        let (points, closed) = circle_track(radius, 10.0, 200);
        let track = build_track("circle", &points, closed);

        // total length within 1%
        let expected_len = 2.0 * PI * radius;
        assert!(
            (track.total_length - expected_len).abs() / expected_len < 0.01,
            "len {} vs {}",
            track.total_length,
            expected_len
        );

        // every curvature within 5% of 1/radius
        let kappa = 1.0 / radius;
        for seg in &track.segments {
            assert!(
                (seg.curvature - kappa).abs() / kappa < 0.05,
                "curvature {} vs {}",
                seg.curvature,
                kappa
            );
        }

        // curvature_at across several arc lengths
        for frac in [0.0, 0.1, 0.37, 0.5, 0.83] {
            let s = frac * track.total_length;
            let c = track.curvature_at(s);
            assert!((c - kappa).abs() / kappa < 0.05, "curvature_at {}", c);
        }
    }

    #[test]
    fn circle_positions_on_circle() {
        let radius = 100.0;
        let (points, closed) = circle_track(radius, 10.0, 200);
        let track = build_track("circle", &points, closed);

        for frac in [0.05, 0.25, 0.5, 0.625, 0.9] {
            let s = frac * track.total_length;
            let (x, y) = track.position_at(s);
            let dist = (x * x + y * y).sqrt();
            assert!(
                (dist - radius).abs() < 0.5,
                "point at s={} has radius {}",
                s,
                dist
            );
        }
    }

    #[test]
    fn oval_track_geometry() {
        let straight = 1000.0;
        let radius = 100.0;
        let (points, closed) = oval_track(straight, radius, 12.0, 400);
        let track = build_track("oval", &points, closed);

        let expected_len = 2.0 * straight + 2.0 * PI * radius;
        assert!(
            (track.total_length - expected_len).abs() / expected_len < 0.01,
            "len {} vs {}",
            track.total_length,
            expected_len
        );

        // mid-straight: curvature ~ 0
        let c_straight = track.curvature_at(straight / 2.0);
        assert!(c_straight.abs() < 1e-3, "straight curvature {}", c_straight);

        // mid right curve: curvature ~ 1/radius
        let kappa = 1.0 / radius;
        let s_curve = straight + PI * radius / 2.0;
        let c_curve = track.curvature_at(s_curve);
        assert!(
            (c_curve - kappa).abs() / kappa < 0.05,
            "curve curvature {} vs {}",
            c_curve,
            kappa
        );

        // mid top straight: curvature ~ 0
        let c_top = track.curvature_at(straight + PI * radius + straight / 2.0);
        assert!(c_top.abs() < 1e-3, "top straight curvature {}", c_top);
    }

    #[test]
    fn locate_basic_and_wrap() {
        let radius = 100.0;
        let (points, closed) = circle_track(radius, 10.0, 200);
        let track = build_track("circle", &points, closed);

        // locate(0) -> (0, 0.0)
        let (i0, f0) = track.locate(0.0);
        assert_eq!(i0, 0);
        assert!(f0.abs() < 1e-12);

        // landing exactly on a node returns that index with zero fraction
        let node_s = track.segments[50].s;
        let (i_node, f_node) = track.locate(node_s);
        assert_eq!(i_node, 50);
        assert!(f_node.abs() < 1e-9);

        // midpoint -> index near the middle of the segment list
        let (i_mid, _) = track.locate(track.total_length / 2.0);
        assert!((i_mid as i64 - 100).abs() <= 1, "midpoint index {}", i_mid);

        // closed wrap: s beyond total_length maps back
        let a = track.locate(10.0);
        let b = track.locate(track.total_length + 10.0);
        assert_eq!(a.0, b.0);
        assert!((a.1 - b.1).abs() < 1e-9);
    }

    #[test]
    fn width_interpolation() {
        let radius = 100.0;
        let (points, closed) = circle_track(radius, 8.0, 100);
        let track = build_track("circle", &points, closed);
        let (wl, wr) = track.width_at(track.total_length * 0.3);
        assert!((wl - 4.0).abs() < 1e-9);
        assert!((wr - 4.0).abs() < 1e-9);
    }
}
