//! Track construction: turning raw [`TrackPoint`]s into a processed [`Track`]
//! with arc length, heading, and curvature at each segment.

use std::f64::consts::PI;

use crate::types::{Track, TrackPoint, TrackSegment};

/// Maps any angle to the range `[-π, π]`.
pub fn normalize_angle(angle: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let mut a = angle % two_pi;
    if a > PI {
        a -= two_pi;
    } else if a < -PI {
        a += two_pi;
    }
    a
}

/// Builds a [`Track`] from raw centerline points.
///
/// Arc length is accumulated from Euclidean distances between consecutive
/// points; heading and curvature are computed via central finite differences
/// (with wrap-around when `closed` is `true`).
pub fn build_track(name: &str, points: &[TrackPoint], closed: bool) -> Track {
    let n = points.len();
    assert!(n >= 2, "a track needs at least two points");

    // Euclidean distance from point `i` to its successor (wrapping for closed).
    let seg_dist = |i: usize| -> f64 {
        let j = if i + 1 < n { i + 1 } else { 0 };
        let dx = points[j].x - points[i].x;
        let dy = points[j].y - points[i].y;
        (dx * dx + dy * dy).sqrt()
    };

    // (a) cumulative arc length
    let mut s = vec![0.0; n];
    for i in 1..n {
        s[i] = s[i - 1] + seg_dist(i - 1);
    }
    let total_length = if closed {
        s[n - 1] + seg_dist(n - 1)
    } else {
        s[n - 1]
    };

    // (b) heading via central finite differences
    let mut heading = vec![0.0; n];
    for (i, h) in heading.iter_mut().enumerate() {
        let (a, b) = neighbor_indices(i, n, closed);
        let dx = points[b].x - points[a].x;
        let dy = points[b].y - points[a].y;
        *h = dy.atan2(dx);
    }

    // (c) curvature = d(heading)/ds, normalizing the heading difference
    let mut curvature = vec![0.0; n];
    for (i, c) in curvature.iter_mut().enumerate() {
        let (a, b) = neighbor_indices(i, n, closed);
        let ds = arc_between(i, a, b, n, closed, &seg_dist);
        let dh = normalize_angle(heading[b] - heading[a]);
        *c = if ds > 0.0 { dh / ds } else { 0.0 };
    }

    // (d) assemble segments
    let segments = (0..n)
        .map(|i| TrackSegment {
            s: s[i],
            x: points[i].x,
            y: points[i].y,
            heading: heading[i],
            curvature: curvature[i],
            width_left: points[i].width_left,
            width_right: points[i].width_right,
        })
        .collect();

    Track {
        name: name.to_string(),
        segments,
        total_length,
        is_closed: closed,
    }
}

/// Returns the `(previous, next)` indices used for the central difference at
/// `i`, matching the boundary handling for both heading and curvature.
fn neighbor_indices(i: usize, n: usize, closed: bool) -> (usize, usize) {
    if closed {
        ((i + n - 1) % n, (i + 1) % n)
    } else if i == 0 {
        (0, 1)
    } else if i == n - 1 {
        (n - 2, n - 1)
    } else {
        (i - 1, i + 1)
    }
}

/// Arc length spanned by the central difference at `i` between neighbors
/// `a` (previous) and `b` (next).
fn arc_between(
    i: usize,
    a: usize,
    b: usize,
    n: usize,
    closed: bool,
    seg_dist: &impl Fn(usize) -> f64,
) -> f64 {
    if closed {
        // a -> i -> b, both consecutive hops
        seg_dist(a) + seg_dist(i)
    } else if i == 0 {
        seg_dist(0)
    } else if i == n - 1 {
        seg_dist(n - 2)
    } else {
        let _ = (a, b);
        seg_dist(i - 1) + seg_dist(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn normalize_angle_known_values() {
        let tol = 1e-9;
        assert!(approx(normalize_angle(0.0), 0.0, tol));
        assert!(approx(normalize_angle(PI), PI, tol));
        assert!(approx(normalize_angle(-PI), -PI, tol));
        assert!(approx(normalize_angle(2.0 * PI), 0.0, tol));
        assert!(approx(normalize_angle(-2.0 * PI), 0.0, tol));
        assert!(approx(normalize_angle(3.0 * PI), PI, tol));

        // just above pi wraps to near -pi
        assert!(approx(normalize_angle(PI + 0.01), -PI + 0.01, tol));
        // just below pi stays
        assert!(approx(normalize_angle(PI - 0.01), PI - 0.01, tol));
        // just below -pi wraps to near +pi
        assert!(approx(normalize_angle(-PI - 0.01), PI - 0.01, tol));
    }

    #[test]
    fn straight_line_has_zero_curvature() {
        let points: Vec<TrackPoint> = (0..5)
            .map(|i| TrackPoint {
                x: i as f64,
                y: 0.0,
                width_left: 1.0,
                width_right: 1.0,
            })
            .collect();
        let track = build_track("line", &points, false);
        assert!(approx(track.total_length, 4.0, 1e-12));
        for seg in &track.segments {
            assert!(approx(seg.curvature, 0.0, 1e-12));
            assert!(approx(seg.heading, 0.0, 1e-12));
        }
    }
}
