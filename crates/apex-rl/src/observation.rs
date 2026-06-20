//! Observation extraction and normalization for the RL environment.
//!
//! Extracts a fixed-size observation vector from the vehicle state
//! and track geometry, normalized to approximately [-1, 1].

use std::f64::consts::PI;

use apex_track::{normalize_angle, Track};

/// Number of observation features.
pub const OBS_DIM: usize = 17;

/// Number of action outputs.
pub const ACT_DIM: usize = 3;

/// Number of lookahead points for curvature and width.
pub const N_LOOKAHEAD: usize = 4;

/// Lookahead distances in meters (exponentially spaced).
pub const LOOKAHEAD_DISTANCES: [f64; N_LOOKAHEAD] = [10.0, 30.0, 70.0, 150.0];

/// Normalization constants for the observation space.
#[derive(Debug, Clone)]
pub struct ObsNormalization {
    /// Maximum expected speed (m/s). Default: 100.0.
    pub v_max: f64,
    /// Maximum expected yaw rate (rad/s). Default: 2.0.
    pub max_yaw_rate: f64,
    /// Maximum expected slip angle (rad). Default: 0.3.
    pub max_slip: f64,
    /// Maximum expected g-force. Default: 5.0.
    pub max_g: f64,
    /// Maximum expected curvature (1/m). Default: 0.05.
    pub max_curvature: f64,
    /// Maximum expected track width (m). Default: 20.0.
    pub max_width: f64,
}

impl Default for ObsNormalization {
    fn default() -> Self {
        ObsNormalization {
            v_max: 100.0,
            max_yaw_rate: 2.0,
            max_slip: 0.3,
            max_g: 5.0,
            max_curvature: 0.05,
            max_width: 20.0,
        }
    }
}

/// Clamp a normalized value to the unit interval `[-1, 1]`.
fn clamp_unit(x: f64) -> f64 {
    x.clamp(-1.0, 1.0)
}

/// Extract the observation vector from the current simulation state.
///
/// Returns an array of [`OBS_DIM`] normalized features:
///   - `[0]`     speed / v_max
///   - `[1]`     yaw_rate / max_yaw_rate
///   - `[2]`     lateral_offset / half_width
///   - `[3]`     heading_error / pi
///   - `[4]`     slip_angle / max_slip
///   - `[5-8]`   curvature at 4 lookahead points (normalized)
///   - `[9-12]`  width at 4 lookahead points (normalized)
///   - `[13]`    longitudinal accel / max_g
///   - `[14]`    lateral accel / max_g
///   - `[15]`    previous steering
///   - `[16]`    previous (throttle - brake)
///
/// All features are clamped to `[-1, 1]` after normalization.
///
/// `track_distance`, `lateral_offset` and `track_heading` describe the car's
/// projection onto the centerline (distance along the lap, signed lateral offset
/// in meters with positive to the right, and the track heading at that point).
/// `accel_long_g` and `accel_lat_g` are the longitudinal and lateral
/// accelerations in units of g, supplied by the caller (the environment tracks
/// the speed history needed to compute them). `prev_action` is the previous
/// normalized action `[steering, throttle, brake]`.
#[allow(clippy::too_many_arguments)]
pub fn extract_observation(
    state: &[f64; 24],
    track: &Track,
    track_distance: f64,
    lateral_offset: f64,
    track_heading: f64,
    accel_long_g: f64,
    accel_lat_g: f64,
    prev_action: &[f64; ACT_DIM],
    norm: &ObsNormalization,
) -> [f64; OBS_DIM] {
    // State layout (14-DOF): [5] yaw, [6] forward velocity (vx),
    // [7] lateral velocity (vy), [11] yaw rate.
    let yaw = state[5];
    let forward_v = state[6];
    let lateral_v = state[7];
    let yaw_rate = state[11];

    let slip_angle = lateral_v.atan2(forward_v.max(1.0));
    let heading_error = normalize_angle(yaw - track_heading);

    let (wl0, wr0) = track.width_at(track_distance);
    let half_width = (0.5 * (wl0 + wr0)).max(1e-3);

    let mut obs = [0.0_f64; OBS_DIM];
    obs[0] = clamp_unit(forward_v / norm.v_max);
    obs[1] = clamp_unit(yaw_rate / norm.max_yaw_rate);
    obs[2] = clamp_unit(lateral_offset / half_width);
    obs[3] = clamp_unit(heading_error / PI);
    obs[4] = clamp_unit(slip_angle / norm.max_slip);

    for (k, &d) in LOOKAHEAD_DISTANCES.iter().enumerate() {
        let s = track_distance + d;
        obs[5 + k] = clamp_unit(track.curvature_at(s) / norm.max_curvature);
        let (wl, wr) = track.width_at(s);
        obs[9 + k] = clamp_unit((wl + wr) / norm.max_width);
    }

    obs[13] = clamp_unit(accel_long_g / norm.max_g);
    obs[14] = clamp_unit(accel_lat_g / norm.max_g);
    obs[15] = clamp_unit(prev_action[0]);
    obs[16] = clamp_unit(prev_action[1] - prev_action[2]);
    obs
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_track::{build_track, oval_track};

    /// Build a simple closed oval track for observation tests.
    fn oval() -> Track {
        let (pts, closed) = oval_track(200.0, 50.0, 12.0, 400);
        build_track("oval", &pts, closed)
    }

    #[test]
    fn test_obs_dim() {
        assert_eq!(OBS_DIM, 17);
        assert_eq!(ACT_DIM, 3);
    }

    #[test]
    fn test_extract_observation_shape() {
        let track = oval();
        let mut state = [0.0; 24];
        state[5] = 0.1; // yaw
        state[6] = 30.0; // forward velocity
        state[7] = 2.0; // lateral velocity
        state[11] = 0.5; // yaw rate
        let norm = ObsNormalization::default();
        let prev = [0.2, 0.7, 0.1];

        let obs = extract_observation(&state, &track, 12.0, 1.5, 0.0, -1.2, 2.5, &prev, &norm);
        assert_eq!(obs.len(), OBS_DIM);
        for (i, v) in obs.iter().enumerate() {
            assert!(v.is_finite(), "obs[{i}] = {v} should be finite");
            assert!(
                (-1.0..=1.0).contains(v),
                "obs[{i}] = {v} should be in [-1, 1]"
            );
        }
    }

    #[test]
    fn test_observation_speed_normalized() {
        let track = oval();
        let mut state = [0.0; 24];
        state[6] = 50.0; // forward velocity
        let norm = ObsNormalization::default();
        let prev = [0.0, 0.0, 0.0];

        let obs = extract_observation(&state, &track, 0.0, 0.0, 0.0, 0.0, 0.0, &prev, &norm);
        assert!(
            (obs[0] - 0.5).abs() < 1e-9,
            "speed 50 / v_max 100 should be 0.5, got {}",
            obs[0]
        );
    }
}
