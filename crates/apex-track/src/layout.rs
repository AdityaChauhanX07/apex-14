//! Track layout definition and conversion.
//!
//! A track layout is defined by a small set of control points.
//! These are interpolated into a smooth closed track via cubic
//! spline fitting, producing a full Track object with computed
//! curvature, headings, and arc-length parameterization.

use crate::builder::build_track;
use crate::track_gen::catmull_rom;
use crate::types::{Track, TrackPoint};

/// Minimum total track width (m) enforced when interpolating or parsing widths.
const MIN_WIDTH: f64 = 8.0;

/// A control point defining part of the track layout.
#[derive(Debug, Clone)]
pub struct ControlPoint {
    /// X position (m).
    pub x: f64,
    /// Y position (m).
    pub y: f64,
    /// Track width at this point (m). Default: 12.0.
    pub width: f64,
}

impl ControlPoint {
    /// Create a control point with default width.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y, width: 12.0 }
    }

    /// Create a control point with specified width.
    pub fn with_width(x: f64, y: f64, width: f64) -> Self {
        Self { x, y, width }
    }
}

/// A track layout defined by control points.
///
/// The control points are interpolated into a smooth closed
/// cubic spline to produce the final track geometry.
#[derive(Debug, Clone)]
pub struct TrackLayout {
    /// Control points defining the track centerline (ordered).
    pub control_points: Vec<ControlPoint>,
    /// Name of the track.
    pub name: String,
    /// Number of output sample points for the interpolated track.
    /// More points = smoother track. Default: 300.
    pub n_samples: usize,
}

impl TrackLayout {
    /// Create a new layout from control points.
    pub fn new(name: &str, points: Vec<ControlPoint>) -> Self {
        Self {
            control_points: points,
            name: name.to_string(),
            n_samples: 300,
        }
    }

    /// Set the number of output sample points.
    pub fn with_samples(mut self, n: usize) -> Self {
        self.n_samples = n;
        self
    }

    /// Number of control points.
    pub fn n_points(&self) -> usize {
        self.control_points.len()
    }

    /// Convert this layout to a [`Track`] by interpolating the control points.
    ///
    /// Uses closed Catmull-Rom spline interpolation to produce a smooth track,
    /// then delegates curvature, heading, and arc-length computation to
    /// [`build_track`].
    ///
    /// Returns `None` if the layout has fewer than 3 control points, or if too
    /// few samples are requested to form a valid track (`n_samples < 3`).
    pub fn to_track(&self) -> Option<Track> {
        let n_points = self.control_points.len();
        if n_points < 3 || self.n_samples < 3 {
            return None;
        }

        let ctrl: Vec<(f64, f64)> = self.control_points.iter().map(|c| (c.x, c.y)).collect();
        let widths: Vec<f64> = self.control_points.iter().map(|c| c.width).collect();

        // Sample the closed Catmull-Rom spline at exactly n_samples points,
        // wrapping the control-point indices to close the loop.
        let mut points = Vec::with_capacity(self.n_samples);
        for k in 0..self.n_samples {
            let u = n_points as f64 * k as f64 / self.n_samples as f64; // in [0, n_points)
            let floor = u.floor();
            let seg = (floor as usize) % n_points;
            let t = u - floor;
            let i0 = (seg + n_points - 1) % n_points;
            let i1 = seg;
            let i2 = (seg + 1) % n_points;
            let i3 = (seg + 2) % n_points;

            let x = catmull_rom(ctrl[i0].0, ctrl[i1].0, ctrl[i2].0, ctrl[i3].0, t);
            let y = catmull_rom(ctrl[i0].1, ctrl[i1].1, ctrl[i2].1, ctrl[i3].1, t);
            // Interpolate the total width and split it evenly across both sides.
            let width =
                catmull_rom(widths[i0], widths[i1], widths[i2], widths[i3], t).max(MIN_WIDTH);
            let half = width / 2.0;

            points.push(TrackPoint {
                x,
                y,
                width_left: half,
                width_right: half,
            });
        }

        Some(build_track(&self.name, &points, true))
    }

    /// Create a layout from a flat parameter vector.
    ///
    /// The vector has 3 values per control point: `[x0, y0, w0, x1, y1, w1, ...]`.
    /// This is the format used by the CMA-ES optimizer.
    pub fn from_params(name: &str, params: &[f64], n_points: usize) -> Option<Self> {
        if params.len() < n_points * 3 {
            return None;
        }
        let points: Vec<ControlPoint> = (0..n_points)
            .map(|i| ControlPoint {
                x: params[i * 3],
                y: params[i * 3 + 1],
                width: params[i * 3 + 2].max(MIN_WIDTH), // minimum width
            })
            .collect();
        Some(Self::new(name, points))
    }

    /// Convert to a flat parameter vector `[x0, y0, w0, x1, y1, w1, ...]`.
    pub fn to_params(&self) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.control_points.len() * 3);
        for cp in &self.control_points {
            params.push(cp.x);
            params.push(cp.y);
            params.push(cp.width);
        }
        params
    }
}

/// Check if the track self-intersects.
///
/// Tests all pairs of non-adjacent segments for intersection.
/// Returns `true` if the track is valid (no self-intersection).
pub fn is_valid_layout(points: &[(f64, f64)]) -> bool {
    let n = points.len();
    if n < 4 {
        return true;
    }
    for i in 0..n {
        let a1 = points[i];
        let a2 = points[(i + 1) % n];
        for j in (i + 2)..n {
            if i == 0 && j == n - 1 {
                continue; // adjacent segments (wrapped)
            }
            let b1 = points[j];
            let b2 = points[(j + 1) % n];
            if segments_intersect(a1, a2, b1, b2) {
                return false;
            }
        }
    }
    true
}

/// Check if two line segments intersect.
fn segments_intersect(a1: (f64, f64), a2: (f64, f64), b1: (f64, f64), b2: (f64, f64)) -> bool {
    // Standard cross-product-based segment intersection test
    let d1 = cross(a1, a2, b1);
    let d2 = cross(a1, a2, b2);
    let d3 = cross(b1, b2, a1);
    let d4 = cross(b1, b2, a2);
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

/// Signed cross product of `(a - o)` and `(b - o)`.
fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Check if a point is inside a polygon (ray casting algorithm).
pub fn point_in_polygon(point: (f64, f64), polygon: &[(f64, f64)]) -> bool {
    let (px, py) = point;
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Check if all points of a track fit within a boundary polygon.
pub fn track_within_boundary(track_points: &[(f64, f64)], boundary: &[(f64, f64)]) -> bool {
    track_points.iter().all(|&p| point_in_polygon(p, boundary))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Control points arranged on a circle of the given radius.
    fn circle_layout(n: usize, radius: f64) -> TrackLayout {
        let points: Vec<ControlPoint> = (0..n)
            .map(|i| {
                let theta = 2.0 * PI * i as f64 / n as f64;
                ControlPoint::new(radius * theta.cos(), radius * theta.sin())
            })
            .collect();
        TrackLayout::new("circle", points)
    }

    #[test]
    fn test_control_point_creation() {
        let cp = ControlPoint::new(100.0, 200.0);
        assert_eq!(cp.x, 100.0);
        assert_eq!(cp.y, 200.0);
        assert_eq!(cp.width, 12.0);

        let cp = ControlPoint::with_width(1.0, 2.0, 15.0);
        assert_eq!(cp.width, 15.0);
    }

    #[test]
    fn test_layout_to_track_circle() {
        let layout = circle_layout(8, 100.0);
        let track = layout.to_track().expect("circle layout should convert");

        assert_eq!(track.segments.len(), 300, "should emit n_samples points");
        assert!(track.is_closed, "track should be closed");

        let expected = 2.0 * PI * 100.0;
        let rel_err = (track.total_length - expected).abs() / expected;
        assert!(
            rel_err < 0.1,
            "length {:.1} should be within 10% of {:.1} (err {:.3})",
            track.total_length,
            expected,
            rel_err
        );
    }

    #[test]
    fn test_layout_to_track_oval() {
        let points = vec![
            ControlPoint::new(-200.0, -80.0),
            ControlPoint::new(0.0, -100.0),
            ControlPoint::new(200.0, -80.0),
            ControlPoint::new(220.0, 0.0),
            ControlPoint::new(200.0, 80.0),
            ControlPoint::new(0.0, 100.0),
            ControlPoint::new(-200.0, 80.0),
            ControlPoint::new(-220.0, 0.0),
        ];
        let layout = TrackLayout::new("oval", points);
        let track = layout.to_track().expect("oval layout should convert");

        assert!(track.total_length > 0.0, "oval should have non-zero length");
        for seg in &track.segments {
            assert!(
                seg.x.is_finite() && seg.y.is_finite(),
                "points must be finite"
            );
        }
    }

    #[test]
    fn test_layout_too_few_points() {
        let layout = TrackLayout::new(
            "tiny",
            vec![ControlPoint::new(0.0, 0.0), ControlPoint::new(10.0, 0.0)],
        );
        assert!(layout.to_track().is_none(), "2 points should yield None");
    }

    #[test]
    fn test_from_params_roundtrip() {
        let points = vec![
            ControlPoint::with_width(10.0, 20.0, 12.0),
            ControlPoint::with_width(30.0, 40.0, 14.0),
            ControlPoint::with_width(50.0, 60.0, 10.0),
        ];
        let layout = TrackLayout::new("rt", points);
        let params = layout.to_params();
        assert_eq!(params.len(), 9);

        let back = TrackLayout::from_params("rt", &params, 3).expect("roundtrip");
        assert_eq!(back.control_points.len(), 3);
        for (a, b) in layout.control_points.iter().zip(&back.control_points) {
            assert_eq!(a.x, b.x);
            assert_eq!(a.y, b.y);
            assert_eq!(a.width, b.width);
        }
    }

    #[test]
    fn test_from_params_too_short() {
        // Need 3*2 = 6 values but only 5 supplied.
        let params = [1.0, 2.0, 12.0, 3.0, 4.0];
        assert!(TrackLayout::from_params("x", &params, 2).is_none());
    }

    #[test]
    fn test_self_intersection_valid() {
        // A convex square does not self-intersect.
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(is_valid_layout(&square));
    }

    #[test]
    fn test_self_intersection_invalid() {
        // A bowtie / figure-8 ordering: edges (0,0)-(2,2) and (2,0)-(0,2) cross.
        let bowtie = [(0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0)];
        assert!(!is_valid_layout(&bowtie), "figure-8 should be invalid");
    }

    #[test]
    fn test_point_in_polygon() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(point_in_polygon((5.0, 5.0), &square), "center is inside");
        assert!(!point_in_polygon((15.0, 5.0), &square), "right is outside");
        assert!(!point_in_polygon((-1.0, 5.0), &square), "left is outside");
    }

    #[test]
    fn test_track_within_boundary() {
        let boundary = [
            (-100.0, -100.0),
            (100.0, -100.0),
            (100.0, 100.0),
            (-100.0, 100.0),
        ];
        let inside = [(0.0, 0.0), (50.0, 50.0), (-50.0, -50.0)];
        assert!(track_within_boundary(&inside, &boundary));

        let one_out = [(0.0, 0.0), (150.0, 0.0)];
        assert!(!track_within_boundary(&one_out, &boundary));
    }

    #[test]
    fn test_layout_curvature_finite() {
        let layout = circle_layout(10, 150.0);
        let track = layout.to_track().expect("layout should convert");
        for seg in &track.segments {
            assert!(
                seg.curvature.is_finite(),
                "curvature must be finite, got {}",
                seg.curvature
            );
            assert!(seg.heading.is_finite(), "heading must be finite");
        }
    }
}
