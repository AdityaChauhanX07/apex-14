//! Track racing quality metrics.
//!
//! Scores a track layout based on how good the racing is likely to be,
//! focusing on overtaking opportunities.

use apex_physics::{qss_lap_sim, CarParams};
use apex_track::Track;

/// Total speed drop (m/s) across a deceleration run for it to count as a heavy
/// braking zone (an overtaking opportunity). The calibrated/default car is fast
/// and grippy enough that even tight corners only shed ~15-20 m/s, so this is
/// set well below a true straight-to-corner delta but comfortably above the
/// near-constant speed of a uniform-curvature loop.
const BRAKING_SPEED_DROP: f64 = 10.0;
/// Minimum per-step deceleration (m/s) treated as "still braking"; smaller dips
/// are noise and end the current run.
const DECEL_EPS: f64 = 0.05;
/// Curvature magnitude (1/m) below which a segment is considered a straight.
const STRAIGHT_CURVATURE: f64 = 0.005;

/// Weight contributions for the overall overtaking score.
const W_BRAKING_ZONE: f64 = 10.0;
const W_DRS_STRAIGHT: f64 = 15.0;
const W_MEAN_STRAIGHT: f64 = 0.1;
const W_SPEED_RANGE: f64 = 0.5;

/// Racing quality assessment for a track.
#[derive(Debug, Clone)]
pub struct RacingQuality {
    /// Overall overtaking opportunity score (higher = better).
    pub overtaking_score: f64,
    /// Number of significant braking zones (speed drop > threshold).
    pub braking_zones: usize,
    /// Number of straights long enough for DRS/slipstream overtaking.
    pub drs_straights: usize,
    /// Average straight length (m).
    pub mean_straight_length: f64,
    /// Speed range: max_speed - min_speed (m/s). Wider = more varied.
    pub speed_range: f64,
    /// Lap time (s).
    pub lap_time: f64,
}

/// Compute racing quality metrics for a track.
///
/// Runs QSS to get the speed profile, then analyzes it for overtaking
/// opportunities. The overtaking score rewards heavy braking zones, long
/// straights preceding them (DRS zones), and a wide speed range.
///
/// Returns `None` if QSS fails to produce a finite, positive lap time.
pub fn compute_racing_quality(
    track: &Track,
    car: &CarParams,
    min_straight_length: f64,
) -> Option<RacingQuality> {
    let qss = qss_lap_sim(track, car);
    let lap_time = qss.lap_time;
    if !lap_time.is_finite() || lap_time <= 0.0 {
        return None;
    }

    let speeds = &qss.speeds;
    let n = speeds.len();
    if n < 3 || n != track.segments.len() {
        return None;
    }

    let braking_zones = count_braking_zones(speeds);
    let straights = straight_lengths(track);

    let drs_straights = straights
        .iter()
        .filter(|&&len| len >= min_straight_length)
        .count();
    let mean_straight_length = if straights.is_empty() {
        0.0
    } else {
        straights.iter().sum::<f64>() / straights.len() as f64
    };

    let max_speed = speeds.iter().cloned().fold(f64::MIN, f64::max);
    let min_speed = speeds.iter().cloned().fold(f64::MAX, f64::min);
    let speed_range = (max_speed - min_speed).max(0.0);

    let overtaking_score = braking_zones as f64 * W_BRAKING_ZONE
        + drs_straights as f64 * W_DRS_STRAIGHT
        + mean_straight_length * W_MEAN_STRAIGHT
        + speed_range * W_SPEED_RANGE;

    Some(RacingQuality {
        overtaking_score,
        braking_zones,
        drs_straights,
        mean_straight_length,
        speed_range,
        lap_time,
    })
}

/// Count heavy braking zones in a closed speed profile.
///
/// A braking zone is a maximal run of consecutive decelerating steps whose
/// total speed drop exceeds [`BRAKING_SPEED_DROP`]. The scan is circular: it
/// starts at a non-decelerating step so a braking zone straddling the
/// start/finish line is counted once.
fn count_braking_zones(speeds: &[f64]) -> usize {
    let n = speeds.len();
    if n < 3 {
        return 0;
    }
    let decel = |i: usize| speeds[i] - speeds[(i + 1) % n];

    // Start at a step that is not decelerating, so runs are not split at index 0.
    let start = (0..n).find(|&i| decel(i) <= DECEL_EPS).unwrap_or(0);

    let mut zones = 0;
    let mut run_drop = 0.0;
    let mut in_run = false;
    for k in 0..n {
        let i = (start + k) % n;
        let d = decel(i);
        if d > DECEL_EPS {
            run_drop += d;
            in_run = true;
        } else {
            if in_run && run_drop > BRAKING_SPEED_DROP {
                zones += 1;
            }
            run_drop = 0.0;
            in_run = false;
        }
    }
    if in_run && run_drop > BRAKING_SPEED_DROP {
        zones += 1;
    }
    zones
}

/// Arc lengths (m) of each contiguous straight run on a closed track.
///
/// A segment is "straight" when its curvature magnitude is below
/// [`STRAIGHT_CURVATURE`]. Runs wrap around the start/finish line.
fn straight_lengths(track: &Track) -> Vec<f64> {
    let segs = &track.segments;
    let n = segs.len();
    if n < 3 {
        return Vec::new();
    }

    // Per-step arc length, wrapping the final segment back to the start.
    let ds = |i: usize| -> f64 {
        if i + 1 < n {
            segs[i + 1].s - segs[i].s
        } else {
            track.total_length - segs[i].s
        }
    };
    let is_straight = |i: usize| segs[i].curvature.abs() < STRAIGHT_CURVATURE;

    // Start at a non-straight segment so a straight crossing index 0 is one run.
    let start = (0..n).find(|&i| !is_straight(i)).unwrap_or(0);

    let mut runs = Vec::new();
    let mut current = 0.0;
    let mut in_run = false;
    for k in 0..n {
        let i = (start + k) % n;
        if is_straight(i) {
            current += ds(i);
            in_run = true;
        } else {
            if in_run && current > 0.0 {
                runs.push(current);
            }
            current = 0.0;
            in_run = false;
        }
    }
    if in_run && current > 0.0 {
        runs.push(current);
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_track::{build_track, circle_track, oval_track};

    #[test]
    fn test_racing_quality_oval() {
        let (pts, closed) = oval_track(1000.0, 120.0, 12.0, 400);
        let track = build_track("oval", &pts, closed);
        let car = CarParams::default();

        let q = compute_racing_quality(&track, &car, 200.0).expect("oval has a valid lap");
        assert!(
            q.braking_zones >= 2,
            "an oval has two corners / braking zones, got {}",
            q.braking_zones
        );
        assert!(
            q.overtaking_score > 0.0,
            "oval should have a positive overtaking score, got {:.2}",
            q.overtaking_score
        );
    }

    #[test]
    fn test_racing_quality_circle() {
        let (cpts, cclosed) = circle_track(150.0, 12.0, 400);
        let circle = build_track("circle", &cpts, cclosed);
        let (opts, oclosed) = oval_track(1000.0, 120.0, 12.0, 400);
        let oval = build_track("oval", &opts, oclosed);
        let car = CarParams::default();

        let qc = compute_racing_quality(&circle, &car, 200.0).expect("circle has a valid lap");
        let qo = compute_racing_quality(&oval, &car, 200.0).expect("oval has a valid lap");

        // A constant-curvature circle has near-constant speed: few braking zones.
        assert!(
            qc.braking_zones <= 1,
            "circle should have ~no braking zones, got {}",
            qc.braking_zones
        );
        assert!(
            qc.overtaking_score < qo.overtaking_score,
            "circle ({:.2}) should score below an oval ({:.2})",
            qc.overtaking_score,
            qo.overtaking_score
        );
    }
}
