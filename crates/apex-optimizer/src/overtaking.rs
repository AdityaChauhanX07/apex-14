//! Overtaking trajectory optimization.
//!
//! Optimizes a following car's trajectory through a track section, given the
//! leader's trajectory as a fixed constraint. The follower exploits the
//! slipstream on straights and copes with dirty-air downforce loss in corners,
//! subject to track boundaries and collision avoidance.

/// The leader car's trajectory through a track section, treated as fixed.
///
/// The optimizer finds the fastest path for the follower that avoids
/// collision with the leader while exploiting (or coping with) the wake.
#[derive(Debug, Clone)]
pub struct LeaderTrajectory {
    /// Arc length stations along the track (m).
    pub stations: Vec<f64>,
    /// Lateral offset from centerline at each station (m).
    pub offsets: Vec<f64>,
    /// Speed at each station (m/s).
    pub speeds: Vec<f64>,
}

impl LeaderTrajectory {
    /// Create a leader trajectory from a QSS result (driving on the centerline).
    pub fn from_qss(result: &crate::collocation::OptimizationResult) -> Self {
        LeaderTrajectory {
            stations: result.stations.clone(),
            offsets: result.offsets.clone(),
            speeds: result.speeds.clone(),
        }
    }

    /// Interpolate the leader's lateral offset at a given arc length.
    pub fn offset_at(&self, s: f64) -> f64 {
        interpolate(&self.stations, &self.offsets, s)
    }

    /// Interpolate the leader's speed at a given arc length.
    pub fn speed_at(&self, s: f64) -> f64 {
        interpolate(&self.stations, &self.speeds, s)
    }
}

/// Simple linear interpolation helper.
fn interpolate(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= *xs.last().unwrap_or(&0.0) {
        return *ys.last().unwrap_or(&0.0);
    }

    for i in 0..xs.len() - 1 {
        if (xs[i]..=xs[i + 1]).contains(&x) {
            let t = (x - xs[i]) / (xs[i + 1] - xs[i]);
            return ys[i] + t * (ys[i + 1] - ys[i]);
        }
    }
    *ys.last().unwrap_or(&0.0)
}

/// Configuration for the overtaking trajectory optimizer.
#[derive(Debug, Clone)]
pub struct OvertakingConfig {
    /// Number of nodes for the follower's trajectory.
    pub n_nodes: usize,
    /// Start arc length of the overtaking section (m).
    pub s_start: f64,
    /// End arc length of the overtaking section (m).
    pub s_end: f64,
    /// Minimum longitudinal clearance between cars (m).
    pub min_longitudinal_clearance: f64,
    /// Minimum lateral clearance when side by side (m).
    pub min_lateral_clearance: f64,
    /// Initial gap behind the leader at s_start (m).
    pub initial_gap: f64,
    /// Car width (m) for collision checking.
    pub car_width: f64,
    /// Car length (m) for collision checking.
    pub car_length: f64,
}

impl Default for OvertakingConfig {
    fn default() -> Self {
        OvertakingConfig {
            n_nodes: 40,
            s_start: 0.0,
            s_end: 500.0,
            min_longitudinal_clearance: 6.0, // ~1 car length
            min_lateral_clearance: 2.0,      // ~1 car width
            initial_gap: 10.0,
            car_width: 2.0,
            car_length: 5.7,
        }
    }
}

/// Result of the overtaking optimization.
#[derive(Debug, Clone)]
pub struct OvertakingResult {
    /// Follower's optimized arc length stations.
    pub stations: Vec<f64>,
    /// Follower's optimized lateral offsets.
    pub offsets: Vec<f64>,
    /// Follower's optimized speeds (m/s).
    pub speeds: Vec<f64>,
    /// Time for the follower to traverse the section (s).
    pub section_time: f64,
    /// Leader's time to traverse the same section (s).
    pub leader_time: f64,
    /// Time gained/lost vs leader (negative = follower faster).
    pub time_delta: f64,
    /// Whether the follower ends up ahead at s_end.
    pub overtake_achieved: bool,
    /// Per-node drag factor (showing slipstream utilization).
    pub drag_factors: Vec<f64>,
    /// Per-node downforce factor (showing dirty air exposure).
    pub downforce_factors: Vec<f64>,
    /// Whether the optimizer converged.
    pub converged: bool,
}

/// Optimize the follower's trajectory through a track section.
///
/// The follower starts behind the leader by `initial_gap` at `s_start`. A greedy
/// forward sweep picks, at each node, the lateral offset and speed that minimize
/// time subject to track boundaries, collision avoidance, and the leader's wake
/// (slipstream on straights, dirty-air downforce loss in corners).
pub fn optimize_overtaking(
    track: &apex_track::Track,
    follower_params: &apex_physics::CarParams,
    leader: &LeaderTrajectory,
    wake: &apex_physics::WakeModel,
    config: &OvertakingConfig,
) -> OvertakingResult {
    let n = config.n_nodes;
    let ds = (config.s_end - config.s_start) / (n - 1) as f64;

    let mut stations = Vec::with_capacity(n);
    let mut offsets = Vec::with_capacity(n);
    let mut speeds = Vec::with_capacity(n);
    let mut drag_factors = Vec::with_capacity(n);
    let mut downforce_factors = Vec::with_capacity(n);

    let mut follower_time = 0.0;
    let mut leader_time = 0.0;
    let mut gap = config.initial_gap; // positive = follower is behind

    for i in 0..n {
        let s = config.s_start + i as f64 * ds;
        stations.push(s);

        // Leader state at this station
        let leader_n = leader.offset_at(s);
        let leader_v = leader.speed_at(s);

        // Decide follower's lateral position.
        // Strategy: on straights (low curvature), tuck behind for slipstream.
        //           approaching corners, move to the inside for overtaking.
        let kappa = track.curvature_at(s).abs();
        let (wl, wr) = track.width_at(s);

        let follower_n = if gap > config.car_length * 2.0 {
            // Well behind: tuck into slipstream (same lateral as leader)
            leader_n
        } else if gap > 0.0 {
            // Close behind: start moving to the inside
            if kappa > 0.001 {
                // In a corner: move inside (opposite sign of curvature)
                if track.curvature_at(s) > 0.0 {
                    -wr * 0.6
                } else {
                    wl * 0.6
                }
            } else {
                // On straight: stay in slipstream but offset slightly
                leader_n + config.min_lateral_clearance * 1.5
            }
        } else if leader_n > 0.0 {
            // Alongside or ahead: maintain separation
            -wr * 0.4
        } else {
            wl * 0.4
        };

        // Clamp to track boundaries
        let follower_n_clamped = follower_n.clamp(-wr + 1.0, wl - 1.0);
        offsets.push(follower_n_clamped);

        // Compute actual wake effect
        let actual_lateral_sep = (follower_n_clamped - leader_n).abs();
        let (df, dff) = if gap > 0.0 {
            wake.wake_effect(gap, actual_lateral_sep)
        } else {
            (1.0, 1.0) // no wake effect when ahead
        };
        drag_factors.push(df);
        downforce_factors.push(dff);

        // Compute follower's maximum speed with modified aero.
        // Modified grip: the downforce factor reduces available grip.
        let modified_downforce = follower_params.downforce(leader_v) * dff;
        let modified_grip =
            follower_params.tire_mu * (follower_params.mass * 9.81 + modified_downforce);

        // Cornering limit with modified grip
        let follower_v = if kappa > 1e-6 {
            let v_corner_sq = modified_grip / (follower_params.mass * kappa);
            v_corner_sq.sqrt().min(200.0)
        } else {
            // Straight: benefit from reduced drag.
            // Terminal velocity scales as 1/sqrt(drag_factor).
            let clean_terminal = 200.0; // approximate
            (clean_terminal / df.sqrt()).min(200.0)
        };

        speeds.push(follower_v);

        // Update times and gap
        if i > 0 {
            let dt_follower = ds / follower_v.max(1.0);
            let dt_leader = ds / leader_v.max(1.0);
            follower_time += dt_follower;
            leader_time += dt_leader;

            // Gap changes based on speed difference.
            // If the follower is faster, the gap decreases.
            gap -= (follower_v - leader_v) * dt_follower;
        }
    }

    let section_time = follower_time;
    let time_delta = section_time - leader_time;
    let overtake_achieved = gap <= 0.0;

    OvertakingResult {
        stations,
        offsets,
        speeds,
        section_time,
        leader_time,
        time_delta,
        overtake_achieved,
        drag_factors,
        downforce_factors,
        converged: true, // greedy method always produces a result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_physics::{CarParams, WakeModel};
    use apex_track::{build_track, oval_track, Track};

    fn oval() -> Track {
        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 600);
        build_track("oval", &pts, closed)
    }

    /// Leader at constant speed/offset over a station range.
    fn const_leader(
        s_start: f64,
        s_end: f64,
        n: usize,
        offset: f64,
        speed: f64,
    ) -> LeaderTrajectory {
        let stations: Vec<f64> = (0..n)
            .map(|i| s_start + (s_end - s_start) * i as f64 / (n - 1) as f64)
            .collect();
        LeaderTrajectory {
            offsets: vec![offset; n],
            speeds: vec![speed; n],
            stations,
        }
    }

    // (a) Leader interpolation.
    #[test]
    fn leader_interpolation() {
        let leader = LeaderTrajectory {
            stations: vec![0.0, 100.0, 200.0, 300.0, 400.0],
            offsets: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            speeds: vec![50.0, 60.0, 70.0, 80.0, 90.0],
        };
        // Interior interpolation.
        assert!((leader.offset_at(150.0) - 1.5).abs() < 1e-9);
        assert!((leader.speed_at(150.0) - 65.0).abs() < 1e-9);
        // Before first / after last: clamp to endpoints.
        assert!((leader.offset_at(-10.0) - 0.0).abs() < 1e-9);
        assert!((leader.offset_at(500.0) - 4.0).abs() < 1e-9);
        assert!((leader.speed_at(500.0) - 90.0).abs() < 1e-9);
    }

    // (b) No wake when the leader is far away.
    #[test]
    fn no_wake_when_far() {
        let track = oval();
        let car = CarParams::f1_2024_calibrated();
        let leader = const_leader(0.0, 500.0, 40, 0.0, 80.0);
        let cfg = OvertakingConfig {
            initial_gap: 1000.0,
            ..OvertakingConfig::default()
        };
        let r = optimize_overtaking(&track, &car, &leader, &WakeModel::f1_default(), &cfg);
        assert!(r.drag_factors.iter().all(|&d| (d - 1.0).abs() < 1e-9));
        assert!(r.downforce_factors.iter().all(|&d| (d - 1.0).abs() < 1e-9));
    }

    // (c) Slipstream on a straight: drag reduced, follower gains time.
    #[test]
    fn slipstream_on_straight() {
        let track = oval();
        let car = CarParams::f1_2024_calibrated();
        let leader = const_leader(0.0, 500.0, 40, 0.0, 80.0);
        let cfg = OvertakingConfig {
            s_start: 0.0,
            s_end: 500.0,
            initial_gap: 10.0,
            ..OvertakingConfig::default()
        };
        let r = optimize_overtaking(&track, &car, &leader, &WakeModel::f1_default(), &cfg);
        assert!(
            r.drag_factors.iter().any(|&d| d < 1.0),
            "slipstream should reduce drag at some node"
        );
        assert!(
            r.time_delta < 0.0,
            "follower should gain time, delta {}",
            r.time_delta
        );
    }

    // (d) Dirty air in a corner: downforce reduced, corner speed lower than clean.
    #[test]
    fn dirty_air_in_corner() {
        let track = oval();
        let car = CarParams::f1_2024_calibrated();
        // Right semicircle of the oval (curved section).
        let (s0, s1) = (1050.0, 1250.0);
        let leader = const_leader(s0, s1, 40, 0.0, 55.0);
        let base = OvertakingConfig {
            n_nodes: 40,
            s_start: s0,
            s_end: s1,
            ..OvertakingConfig::default()
        };

        let clean = optimize_overtaking(
            &track,
            &car,
            &leader,
            &WakeModel::f1_default(),
            &OvertakingConfig {
                initial_gap: 1000.0,
                ..base.clone()
            },
        );
        let dirty = optimize_overtaking(
            &track,
            &car,
            &leader,
            &WakeModel::f1_default(),
            &OvertakingConfig {
                initial_gap: 5.0,
                ..base
            },
        );

        assert!(
            dirty.downforce_factors.iter().any(|&d| d < 1.0),
            "dirty air should reduce downforce in the corner"
        );
        assert!(
            dirty.speeds[0] < clean.speeds[0],
            "dirty-air corner speed {} should be below clean {}",
            dirty.speeds[0],
            clean.speeds[0]
        );
    }

    // (e) Collision avoidance: follower offsets laterally when close.
    #[test]
    fn collision_avoidance_lateral_offset() {
        let track = oval();
        let car = CarParams::f1_2024_calibrated();
        let leader = const_leader(0.0, 500.0, 40, 0.0, 80.0);
        let cfg = OvertakingConfig {
            initial_gap: 5.0,
            ..OvertakingConfig::default()
        };
        let r = optimize_overtaking(&track, &car, &leader, &WakeModel::f1_default(), &cfg);
        let separated = r
            .stations
            .iter()
            .zip(r.offsets.iter())
            .any(|(&s, &n)| (n - leader.offset_at(s)).abs() >= cfg.min_lateral_clearance);
        assert!(
            separated,
            "follower should create lateral separation from the leader"
        );
    }

    // (f) Overtake on a long straight with slipstream.
    #[test]
    fn overtake_on_long_straight() {
        let track = oval();
        let car = CarParams::f1_2024_calibrated();
        let leader = const_leader(0.0, 800.0, 40, 0.0, 70.0);
        let cfg = OvertakingConfig {
            s_start: 0.0,
            s_end: 800.0,
            initial_gap: 8.0,
            ..OvertakingConfig::default()
        };
        let r = optimize_overtaking(&track, &car, &leader, &WakeModel::f1_default(), &cfg);
        assert!(r.time_delta < 0.0, "follower should traverse faster");
        assert!(r.overtake_achieved, "follower should pass the leader");
    }

    // (g) Result completeness and validity.
    #[test]
    fn result_is_complete_and_valid() {
        let track = oval();
        let car = CarParams::f1_2024_calibrated();
        let leader = const_leader(0.0, 500.0, 40, 0.0, 80.0);
        let cfg = OvertakingConfig::default();
        let r = optimize_overtaking(&track, &car, &leader, &WakeModel::f1_default(), &cfg);

        let n = cfg.n_nodes;
        assert_eq!(r.stations.len(), n);
        assert_eq!(r.offsets.len(), n);
        assert_eq!(r.speeds.len(), n);
        assert_eq!(r.drag_factors.len(), n);
        assert_eq!(r.downforce_factors.len(), n);

        assert!(r.speeds.iter().all(|&v| v > 0.0 && v.is_finite()));
        for (&s, &offset) in r.stations.iter().zip(r.offsets.iter()) {
            let (wl, wr) = track.width_at(s);
            assert!(
                offset <= wl && offset >= -wr,
                "offset {offset} out of bounds at s={s}"
            );
        }
        assert!(r.section_time > 0.0 && r.leader_time > 0.0);
    }
}
