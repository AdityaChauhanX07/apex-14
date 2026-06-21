//! Track design constraints and validation.
//!
//! Defines geometric and performance constraints for track layouts
//! and provides functions to check whether a track satisfies them.

use crate::layout::{is_valid_layout, point_in_polygon};
use crate::types::Track;

/// Penalty weight applied per track point lying outside the land boundary.
const BOUNDARY_PENALTY_PER_POINT: f64 = 100.0;
/// Flat penalty applied when the track self-intersects.
const INTERSECTION_PENALTY: f64 = 1000.0;
/// Penalty weight per metre of track-length deviation from the allowed range.
const LENGTH_PENALTY_PER_M: f64 = 1.0;
/// Penalty weight per second of lap-time deviation from the target range.
const LAP_TIME_PENALTY_PER_S: f64 = 10.0;
/// Penalty weight per unit of excess curvature beyond the minimum-radius limit.
const RADIUS_PENALTY_SCALE: f64 = 100.0;
/// Penalty weight per metre that two non-adjacent points fall short of the
/// minimum segment clearance.
const CLEARANCE_PENALTY_SCALE: f64 = 1.0;

/// Constraints for track design optimization.
#[derive(Debug, Clone)]
pub struct TrackConstraints {
    /// Land boundary polygon. The track must fit entirely within this polygon.
    /// Empty means no boundary constraint.
    pub boundary: Vec<(f64, f64)>,
    /// Target lap time range `[min, max]` in seconds.
    /// The optimizer penalizes tracks outside this range.
    pub target_lap_time: (f64, f64),
    /// Minimum corner radius (m). Corners tighter than this are penalized.
    pub min_corner_radius: f64,
    /// Minimum straight length for overtaking (m).
    pub min_straight_length: f64,
    /// Minimum track width (m).
    pub min_width: f64,
    /// Maximum track width (m).
    pub max_width: f64,
    /// Minimum clearance between non-adjacent track segments (m).
    /// Prevents the track from running too close to itself.
    pub min_segment_clearance: f64,
    /// Minimum total track length (m).
    pub min_track_length: f64,
    /// Maximum total track length (m).
    pub max_track_length: f64,
}

impl Default for TrackConstraints {
    fn default() -> Self {
        Self {
            boundary: Vec::new(),
            target_lap_time: (60.0, 120.0),
            min_corner_radius: 15.0,
            min_straight_length: 200.0,
            min_width: 10.0,
            max_width: 20.0,
            min_segment_clearance: 20.0,
            min_track_length: 2000.0,
            max_track_length: 8000.0,
        }
    }
}

/// Result of constraint checking.
#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    /// Total penalty score (0 = all constraints satisfied).
    pub total_penalty: f64,
    /// Boundary violation penalty.
    pub boundary_penalty: f64,
    /// Self-intersection penalty.
    pub intersection_penalty: f64,
    /// Lap time deviation penalty.
    pub lap_time_penalty: f64,
    /// Minimum radius violation penalty.
    pub radius_penalty: f64,
    /// Track length violation penalty.
    pub length_penalty: f64,
    /// Segment clearance violation penalty.
    pub clearance_penalty: f64,
    /// Whether all hard constraints are satisfied (no self-intersection, within
    /// boundary).
    pub feasible: bool,
}

/// Check all constraints on a track layout.
///
/// Returns a violation report with penalty scores. The total penalty is 0 when
/// all constraints are met. `track_points` are the sampled `(x, y)` centerline
/// positions; `lap_time` is the computed lap time, if available (passed in so
/// this module stays free of any physics dependency).
pub fn check_constraints(
    track: &Track,
    track_points: &[(f64, f64)],
    constraints: &TrackConstraints,
    lap_time: Option<f64>,
) -> ConstraintViolation {
    let mut v = ConstraintViolation {
        total_penalty: 0.0,
        boundary_penalty: 0.0,
        intersection_penalty: 0.0,
        lap_time_penalty: 0.0,
        radius_penalty: 0.0,
        length_penalty: 0.0,
        clearance_penalty: 0.0,
        feasible: true,
    };

    // Boundary check.
    if !constraints.boundary.is_empty() {
        let outside_count = track_points
            .iter()
            .filter(|p| !point_in_polygon(**p, &constraints.boundary))
            .count();
        if outside_count > 0 {
            v.boundary_penalty = outside_count as f64 * BOUNDARY_PENALTY_PER_POINT;
            v.feasible = false;
        }
    }

    // Self-intersection.
    if !is_valid_layout(track_points) {
        v.intersection_penalty = INTERSECTION_PENALTY;
        v.feasible = false;
    }

    // Track length.
    let length = track.total_length;
    if length < constraints.min_track_length {
        v.length_penalty = (constraints.min_track_length - length) * LENGTH_PENALTY_PER_M;
    } else if length > constraints.max_track_length {
        v.length_penalty = (length - constraints.max_track_length) * LENGTH_PENALTY_PER_M;
    }

    // Lap time (if computed).
    if let Some(lt) = lap_time {
        if lt < constraints.target_lap_time.0 {
            v.lap_time_penalty = (constraints.target_lap_time.0 - lt) * LAP_TIME_PENALTY_PER_S;
        } else if lt > constraints.target_lap_time.1 {
            v.lap_time_penalty = (lt - constraints.target_lap_time.1) * LAP_TIME_PENALTY_PER_S;
        }
    }

    // Minimum corner radius: curvature magnitude must stay below 1/min_radius.
    if constraints.min_corner_radius > 0.0 {
        let max_curvature = 1.0 / constraints.min_corner_radius;
        let mut radius_penalty = 0.0;
        for seg in &track.segments {
            let excess = seg.curvature.abs() - max_curvature;
            if excess > 0.0 {
                radius_penalty += excess * RADIUS_PENALTY_SCALE;
            }
        }
        v.radius_penalty = radius_penalty;
    }

    // Segment clearance: penalize non-adjacent points running too close. Points
    // near each other in track order are naturally close, so skip a window
    // around each index and only flag genuine self-approaches.
    v.clearance_penalty = clearance_penalty(track_points, constraints.min_segment_clearance);

    v.total_penalty = v.boundary_penalty
        + v.intersection_penalty
        + v.lap_time_penalty
        + v.radius_penalty
        + v.length_penalty
        + v.clearance_penalty;

    v
}

/// Sum the shortfall of every non-adjacent point pair that falls within
/// `min_clearance` of each other. Pairs whose circular index separation is
/// within a small window of the loop are skipped (they are consecutive samples
/// and legitimately close).
fn clearance_penalty(points: &[(f64, f64)], min_clearance: f64) -> f64 {
    let n = points.len();
    if n < 4 || min_clearance <= 0.0 {
        return 0.0;
    }
    // Skip points within ~10% of the loop in either direction.
    let skip = (n / 10).max(2);
    let mut penalty = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let gap = (j - i).min(n - (j - i));
            if gap <= skip {
                continue;
            }
            let dx = points[i].0 - points[j].0;
            let dy = points[i].1 - points[j].1;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < min_clearance {
                penalty += (min_clearance - dist) * CLEARANCE_PENALTY_SCALE;
            }
        }
    }
    penalty
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::build_track;
    use crate::layout::{ControlPoint, TrackLayout};
    use crate::types::TrackPoint;
    use std::f64::consts::PI;

    /// A circular layout with the given control-point count and radius.
    fn circle_layout(n: usize, radius: f64) -> TrackLayout {
        let points: Vec<ControlPoint> = (0..n)
            .map(|i| {
                let theta = 2.0 * PI * i as f64 / n as f64;
                ControlPoint::new(radius * theta.cos(), radius * theta.sin())
            })
            .collect();
        TrackLayout::new("circle", points)
    }

    /// Extract `(x, y)` centerline points from a track.
    fn points_of(track: &Track) -> Vec<(f64, f64)> {
        track.segments.iter().map(|s| (s.x, s.y)).collect()
    }

    #[test]
    fn test_default_constraints() {
        let c = TrackConstraints::default();
        assert!(c.boundary.is_empty());
        assert!(c.target_lap_time.0 < c.target_lap_time.1);
        assert!(c.min_corner_radius > 0.0);
        assert!(c.min_width < c.max_width);
        assert!(c.min_track_length < c.max_track_length);
    }

    #[test]
    fn test_check_constraints_valid_track() {
        // A large, smooth circle satisfies every geometric constraint.
        let layout = circle_layout(8, 350.0);
        let track = layout.to_track().expect("layout converts");
        let points = points_of(&track);
        let v = check_constraints(&track, &points, &TrackConstraints::default(), None);

        assert!(v.feasible, "a clean circle should be feasible");
        assert!(
            v.total_penalty < 1e-6,
            "expected ~zero penalty, got {:.3} (len {:.1})",
            v.total_penalty,
            track.total_length
        );
    }

    #[test]
    fn test_check_constraints_too_short() {
        // A tiny circle is well under the 2000 m minimum length.
        let layout = circle_layout(8, 50.0);
        let track = layout.to_track().expect("layout converts");
        let points = points_of(&track);
        let v = check_constraints(&track, &points, &TrackConstraints::default(), None);

        assert!(
            v.length_penalty > 0.0,
            "short track should incur a length penalty (len {:.1})",
            track.total_length
        );
    }

    #[test]
    fn test_check_constraints_self_intersecting() {
        // A bowtie centerline self-intersects.
        let bowtie = [(0.0, 0.0), (100.0, 100.0), (100.0, 0.0), (0.0, 100.0)];
        let tps: Vec<TrackPoint> = bowtie
            .iter()
            .map(|&(x, y)| TrackPoint {
                x,
                y,
                width_left: 6.0,
                width_right: 6.0,
            })
            .collect();
        let track = build_track("bowtie", &tps, true);
        let v = check_constraints(&track, &bowtie, &TrackConstraints::default(), None);

        assert!(v.intersection_penalty > 0.0, "bowtie should self-intersect");
        assert!(!v.feasible, "self-intersecting track is infeasible");
    }

    #[test]
    fn test_lap_time_penalty() {
        let layout = circle_layout(8, 350.0);
        let track = layout.to_track().expect("layout converts");
        let points = points_of(&track);
        let c = TrackConstraints::default();

        // Lap time far above the target max incurs a penalty.
        let v = check_constraints(&track, &points, &c, Some(300.0));
        assert!(v.lap_time_penalty > 0.0, "slow lap should be penalized");

        // Lap time inside the target range does not.
        let v = check_constraints(&track, &points, &c, Some(90.0));
        assert_eq!(v.lap_time_penalty, 0.0, "in-range lap is unpenalized");
    }
}
