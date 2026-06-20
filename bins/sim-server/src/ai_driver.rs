//! AI driver adapter wrapping a trained PPO policy.
//!
//! Loads a policy network from a safetensors file and computes control
//! inputs from the live simulation state, replicating the observation
//! extraction and action post-processing used by `apex_rl`'s training
//! environment so the driving behaviour matches what was learned.

use std::path::Path;

use apex_physics::car_params::GRAVITY;
use apex_track::Track;
use candle_core::{Device, Tensor};

use apex_rl::observation::{extract_observation, ObsNormalization, ACT_DIM, OBS_DIM};
use apex_rl::policy::{load_agent, postprocess_actions, PolicyNet};

/// AI driver that uses a trained PPO policy to generate control inputs.
///
/// Mirrors `apex_rl::env::RaceEnv`: it tracks the previous (post-processed)
/// action and the previous forward speed so it can reconstruct the exact
/// observation vector the policy was trained on, then queries the policy
/// deterministically (using the distribution mean).
pub struct AiDriver {
    /// Trained policy (actor) network.
    policy: PolicyNet,
    /// Observation normalization constants (must match training).
    obs_norm: ObsNormalization,
    /// Previous post-processed action `[steering, throttle, brake]`.
    prev_action: [f64; ACT_DIM],
    /// Forward speed at the previous decision (m/s), for longitudinal accel.
    prev_speed: f64,
}

impl AiDriver {
    /// Load the AI driver from a safetensors policy file.
    ///
    /// `start_speed` is the forward speed (m/s) the car is launched at, used to
    /// seed the previous-speed history so the first observation reports zero
    /// longitudinal acceleration (matching the environment's reset).
    pub fn load(path: &str, start_speed: f64) -> Result<Self, Box<dyn std::error::Error>> {
        let (policy, _value, _var_map) = load_agent(Path::new(path))?;
        Ok(Self {
            policy,
            obs_norm: ObsNormalization::default(),
            prev_action: [0.0; ACT_DIM],
            prev_speed: start_speed,
        })
    }

    /// Compute control inputs from the current simulation state.
    ///
    /// Extracts the observation (track heading, accelerations, lookahead
    /// curvature/width and previous action), runs the policy network, and
    /// post-processes the raw outputs. Returns `[steering, throttle, brake]`
    /// where steering is in `[-1, 1]` and throttle/brake are in `[0, 1]` — the
    /// same normalized ranges as [`apex_sim::protocol::InputPacket`].
    ///
    /// `track_distance` and `lateral_offset` are the car's projection onto the
    /// centerline; `dt` is the control time step (s) used to derive longitudinal
    /// acceleration from the speed history.
    pub fn compute_action(
        &mut self,
        state: &[f64; 24],
        track: &Track,
        track_distance: f64,
        lateral_offset: f64,
        dt: f64,
    ) -> Result<[f64; 3], Box<dyn std::error::Error>> {
        // Track heading at the current projected position.
        let track_heading = track.heading_at(track_distance);

        // Derived accelerations (g), matching the training environment.
        let speed = state[6];
        let yaw_rate = state[11];
        let accel_long = (speed - self.prev_speed) / dt / GRAVITY;
        let accel_lat = speed * yaw_rate / GRAVITY;
        self.prev_speed = speed;

        // Extract the normalized observation vector.
        let obs = extract_observation(
            state,
            track,
            track_distance,
            lateral_offset,
            track_heading,
            accel_long,
            accel_lat,
            &self.prev_action,
            &self.obs_norm,
        );

        // Convert to a tensor and run the policy deterministically.
        let device = Device::Cpu;
        let obs_f32: Vec<f32> = obs.iter().map(|&v| v as f32).collect();
        let obs_tensor = Tensor::from_vec(obs_f32, (1, OBS_DIM), &device)?;
        let raw_action = self.policy.deterministic_action(&obs_tensor)?;
        let action = postprocess_actions(&raw_action)?;

        // Extract the action as an f64 array.
        let action_vec: Vec<f32> = action.squeeze(0)?.to_vec1()?;
        let result = [
            action_vec[0] as f64, // steering [-1, 1]
            action_vec[1] as f64, // throttle [0, 1]
            action_vec[2] as f64, // brake [0, 1]
        ];

        self.prev_action = result;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use apex_track::{build_track, oval_track};

    /// Forward speed (m/s) used to seed the driver in the tests.
    const TEST_START_SPEED: f64 = 50.0;

    /// Write a fresh, randomly-initialized policy/value pair to `path`.
    ///
    /// Mirrors `apex_rl`'s `test_save_load_roundtrip`: the variable name prefixes
    /// (`policy`/`value`) match what [`AiDriver::load`] expects via `load_agent`.
    fn create_test_weights(path: &Path) {
        let var_map = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(
            &var_map,
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        );
        let _policy = apex_rl::policy::PolicyNet::new(vb.pp("policy")).expect("build policy");
        let _value = apex_rl::policy::ValueNet::new(vb.pp("value")).expect("build value");
        apex_rl::policy::save_agent(&var_map, path).expect("save weights");
    }

    /// A unique temp path for a weights file, namespaced by test tag and PID so
    /// parallel test runs do not collide.
    fn temp_weights_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "apex_ai_driver_{tag}_{}.safetensors",
            std::process::id()
        ))
    }

    /// A simple closed oval track for observation extraction.
    fn test_track() -> Track {
        let (pts, closed) = oval_track(200.0, 50.0, 12.0, 400);
        build_track("oval", &pts, closed)
    }

    /// A minimal-but-valid 14-DOF state at the given forward speed and yaw rate.
    ///
    /// Only the fields read by observation extraction need to be set: `[6]`
    /// forward velocity and `[11]` yaw rate (yaw `[5]` and lateral velocity `[7]`
    /// stay zero).
    fn state_at(speed: f64, yaw_rate: f64) -> [f64; 24] {
        let mut s = [0.0; 24];
        s[6] = speed;
        s[11] = yaw_rate;
        s
    }

    /// Create weights, load a driver from them, then delete the file (the
    /// weights live in memory after loading, so cleanup is immediate and robust
    /// to later assertion failures).
    fn load_test_driver(tag: &str) -> AiDriver {
        let path = temp_weights_path(tag);
        create_test_weights(&path);
        let driver = AiDriver::load(
            path.to_str().expect("temp path is valid UTF-8"),
            TEST_START_SPEED,
        )
        .expect("load should succeed for valid weights");
        let _ = std::fs::remove_file(&path);
        driver
    }

    #[test]
    fn test_load_valid_weights() {
        let path = temp_weights_path("load_valid");
        create_test_weights(&path);
        let result = AiDriver::load(path.to_str().expect("utf-8 path"), TEST_START_SPEED);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_ok(), "loading valid weights should succeed");
    }

    #[test]
    fn test_load_missing_file() {
        let path = temp_weights_path("missing");
        // Make sure the file does not exist.
        let _ = std::fs::remove_file(&path);
        let result = AiDriver::load(path.to_str().expect("utf-8 path"), TEST_START_SPEED);
        assert!(result.is_err(), "loading a missing file should error");
    }

    #[test]
    fn test_compute_action_shape() {
        let mut driver = load_test_driver("shape");
        let track = test_track();
        let state = state_at(30.0, 0.0);
        let action = driver
            .compute_action(&state, &track, 0.0, 0.0, 0.01)
            .expect("compute_action should succeed");
        assert_eq!(action.len(), 3, "action should have 3 elements");
    }

    #[test]
    fn test_compute_action_valid_ranges() {
        let mut driver = load_test_driver("ranges");
        let track = test_track();
        let state = state_at(30.0, 0.0);
        let action = driver
            .compute_action(&state, &track, 0.0, 0.0, 0.01)
            .expect("compute_action should succeed");
        assert!(
            (-1.0..=1.0).contains(&action[0]),
            "steering {} out of [-1, 1]",
            action[0]
        );
        assert!(
            (0.0..=1.0).contains(&action[1]),
            "throttle {} out of [0, 1]",
            action[1]
        );
        assert!(
            (0.0..=1.0).contains(&action[2]),
            "brake {} out of [0, 1]",
            action[2]
        );
    }

    #[test]
    fn test_compute_action_no_nan() {
        let mut driver = load_test_driver("nonan");
        let track = test_track();
        let state = state_at(45.0, 0.2);
        let action = driver
            .compute_action(&state, &track, 10.0, 1.5, 0.01)
            .expect("compute_action should succeed");
        for (i, v) in action.iter().enumerate() {
            assert!(v.is_finite(), "action[{i}] = {v} should be finite");
        }
    }

    #[test]
    fn test_compute_action_updates_prev_state() {
        let mut driver = load_test_driver("prevstate");
        let track = test_track();

        // Initial internal state from the constructor.
        assert!((driver.prev_speed - TEST_START_SPEED).abs() < 1e-12);
        assert_eq!(driver.prev_action, [0.0; ACT_DIM]);

        // First call: prev_speed tracks the state speed, prev_action the result.
        let s1 = state_at(30.0, 0.1);
        let a1 = driver
            .compute_action(&s1, &track, 0.0, 0.0, 0.01)
            .expect("first compute_action");
        assert!(
            (driver.prev_speed - 30.0).abs() < 1e-12,
            "prev_speed should update to the state's speed"
        );
        assert_eq!(
            driver.prev_action, a1,
            "prev_action should track the result"
        );

        // Second call with a different speed: internal state keeps updating.
        let s2 = state_at(40.0, -0.1);
        let a2 = driver
            .compute_action(&s2, &track, 5.0, 1.0, 0.01)
            .expect("second compute_action");
        assert!(
            (driver.prev_speed - 40.0).abs() < 1e-12,
            "prev_speed should update on the second call"
        );
        assert_eq!(driver.prev_action, a2);
    }

    #[test]
    fn test_compute_action_different_states() {
        // Two drivers loaded from the same file share identical weights, so any
        // difference in their output is due to the differing input states.
        let path = temp_weights_path("differ");
        create_test_weights(&path);
        let path_str = path.to_str().expect("utf-8 path");
        let mut driver_a = AiDriver::load(path_str, TEST_START_SPEED).expect("load driver a");
        let mut driver_b = AiDriver::load(path_str, TEST_START_SPEED).expect("load driver b");
        let _ = std::fs::remove_file(&path);

        let track = test_track();
        let action_a = driver_a
            .compute_action(&state_at(15.0, 0.0), &track, 0.0, 0.0, 0.01)
            .expect("compute a");
        let action_b = driver_b
            .compute_action(&state_at(70.0, 0.3), &track, 20.0, 4.0, 0.01)
            .expect("compute b");

        let differs = action_a
            .iter()
            .zip(action_b.iter())
            .any(|(x, y)| (x - y).abs() > 1e-6);
        assert!(
            differs,
            "policy should respond to different states: {action_a:?} vs {action_b:?}"
        );
    }
}
