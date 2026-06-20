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
