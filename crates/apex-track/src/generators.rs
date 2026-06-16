//! Parametric track generators useful for tests and demos.

use std::f64::consts::PI;

use crate::types::TrackPoint;

/// Generates a circular track centered at the origin.
///
/// Starts at `(radius, 0)` and proceeds counter-clockwise, so curvature is a
/// constant `+1/radius` everywhere. Returns the points and `closed = true`.
pub fn circle_track(radius: f64, width: f64, num_points: usize) -> (Vec<TrackPoint>, bool) {
    let half = width / 2.0;
    let points = (0..num_points)
        .map(|i| {
            let theta = 2.0 * PI * (i as f64) / (num_points as f64);
            TrackPoint {
                x: radius * theta.cos(),
                y: radius * theta.sin(),
                width_left: half,
                width_right: half,
            }
        })
        .collect();
    (points, true)
}

/// Generates an oval track: two straights of length `straight_length` joined by
/// two semicircles of the given `radius`, centered at the origin.
///
/// Points are distributed evenly by arc length. The path starts at the
/// beginning of the bottom straight and heads in the +X direction
/// (counter-clockwise overall). Returns the points and `closed = true`.
pub fn oval_track(
    straight_length: f64,
    radius: f64,
    width: f64,
    num_points: usize,
) -> (Vec<TrackPoint>, bool) {
    let l = straight_length;
    let r = radius;
    let half = width / 2.0;

    let total = 2.0 * l + 2.0 * PI * r;
    let s1 = l; // end of bottom straight
    let s2 = l + PI * r; // end of right semicircle
    let s3 = 2.0 * l + PI * r; // end of top straight

    let points = (0..num_points)
        .map(|i| {
            let t = total * (i as f64) / (num_points as f64);
            let (x, y) = if t < s1 {
                // bottom straight: from (-l/2, -r) heading +X
                (-l / 2.0 + t, -r)
            } else if t < s2 {
                // right semicircle, center (l/2, 0), start angle -π/2, CCW
                let phi = -PI / 2.0 + (t - s1) / r;
                (l / 2.0 + r * phi.cos(), r * phi.sin())
            } else if t < s3 {
                // top straight: from (l/2, r) heading -X
                (l / 2.0 - (t - s2), r)
            } else {
                // left semicircle, center (-l/2, 0), start angle π/2, CCW
                let phi = PI / 2.0 + (t - s3) / r;
                (-l / 2.0 + r * phi.cos(), r * phi.sin())
            };
            TrackPoint {
                x,
                y,
                width_left: half,
                width_right: half,
            }
        })
        .collect();
    (points, true)
}
