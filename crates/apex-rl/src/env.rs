//! Gym-style reinforcement learning environment wrapping the 14-DOF model.
//!
//! Provides reset/step interface for training RL agents to drive
//! race cars on any track.

use apex_integrator::rk4_step;
use apex_physics::car_params::GRAVITY;
use apex_physics::{
    AeroModel, CarParams, FourteenDofModel, PacejkaTire, Powertrain, SuspensionSystem,
};
use apex_track::{normalize_angle, Track};

use crate::observation::{extract_observation, ObsNormalization, ACT_DIM, OBS_DIM};
use crate::reward::{compute_reward, RewardConfig, RewardInput, RewardOutput};

/// Forward speed (m/s) the model is trimmed for and the car is started at.
///
/// Trimming the static equilibrium at the start speed means the car begins each
/// episode already balanced (suspension at rest, tire loads matching gravity and
/// downforce), so there is no launch transient. It is deliberately gentle so the
/// car is forgiving to control in the first corner while staying above the
/// "stuck" speed threshold.
const START_SPEED: f64 = 10.0;

/// Fraction of the grip-limited torque the traction cap allows.
///
/// Mirrors the sim server's input mapping: capping the drive torque below the
/// grip peak keeps the driven wheels tracking road speed (avoiding runaway
/// wheelspin) so braking and acceleration behave correctly under explicit
/// integration.
const TRACTION_MARGIN: f64 = 0.6;

/// Maximum total drive torque (Nm) the driven axle(s) can transmit at `speed`.
///
/// Reimplemented locally from the sim server (the RL crate does not depend on
/// `apex-sim`). For each driven axle the torque is bounded by
/// `grip_mu * axle_load * wheel_radius * TRACTION_MARGIN`, accounting for the
/// drive split; the overall cap is the tightest of those bounds.
fn traction_torque_limit(car: &CarParams, grip_mu: f64, speed: f64) -> f64 {
    let (front_load, rear_load) = car.axle_loads(speed, 0.0);
    let r = car.wheel_radius;
    let dd = car.drive_distribution;

    let rear_cap = if dd > 0.0 {
        TRACTION_MARGIN * grip_mu * rear_load * r / dd
    } else {
        f64::INFINITY
    };
    let front_cap = if dd < 1.0 {
        TRACTION_MARGIN * grip_mu * front_load * r / (1.0 - dd)
    } else {
        f64::INFINITY
    };
    rear_cap.min(front_cap)
}

/// Estimate the tire's peak longitudinal friction coefficient (`max fx / Fz`)
/// at the reference load `fz`, by sweeping the slip ratio.
///
/// Reimplemented locally from the sim server. Computed once at construction.
fn tire_peak_longitudinal_mu(tire: &PacejkaTire, fz: f64) -> f64 {
    if fz <= 0.0 {
        return 0.0;
    }
    let mut peak = 0.0_f64;
    for i in 0..=200 {
        let sr = i as f64 * 0.01;
        peak = peak.max(tire.combined_forces_smooth(0.0, sr, fz).fx);
    }
    peak / fz
}

/// Build the static-equilibrium 14-DOF state vector for `model` at forward
/// `speed`: chassis height, forward velocity, free-rolling wheel speeds, and
/// equilibrium suspension travel; all other states are zero.
fn equilibrium_state(model: &FourteenDofModel, speed: f64) -> [f64; 24] {
    let z = model.equilibrium_travel();
    let r = model.params.wheel_radius;
    let w = speed / r;
    let mut s = [0.0; 24];
    s[2] = model.params.cog_height;
    s[6] = speed;
    s[12] = w;
    s[13] = w;
    s[14] = w;
    s[15] = w;
    s[16] = z[0];
    s[17] = z[1];
    s[18] = z[2];
    s[19] = z[3];
    s
}

/// Project a world position onto the track centerline.
///
/// Returns `(distance_along_track, lateral_offset)`, where the offset is
/// positive to the right of the track direction and negative to the left.
/// Reimplemented locally from the sim server as a full O(N) nearest-point scan
/// over the segment chords (the RL crate does not depend on `apex-sim`).
fn project_onto_track(track: &Track, x: f64, y: f64) -> (f64, f64) {
    let segs = &track.segments;
    let n = segs.len();
    if n < 2 {
        return (0.0, 0.0);
    }
    let last = if track.is_closed { n } else { n - 1 };

    let mut best_s = 0.0;
    let mut best_off = 0.0;
    let mut best_d2 = f64::INFINITY;
    for i in 0..last {
        let a = &segs[i];
        let j = (i + 1) % n;
        let b = &segs[j];
        let ex = b.x - a.x;
        let ey = b.y - a.y;
        let len2 = ex * ex + ey * ey;
        let t = if len2 > 1e-12 {
            (((x - a.x) * ex + (y - a.y) * ey) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let px = a.x + t * ex;
        let py = a.y + t * ey;
        let d2 = (x - px) * (x - px) + (y - py) * (y - py);
        if d2 < best_d2 {
            best_d2 = d2;
            let s_b = if j == 0 { track.total_length } else { b.s };
            best_s = a.s + t * (s_b - a.s);
            // Lateral offset: positive to the right of the (unit) tangent.
            let inv_len = if len2 > 1e-12 { 1.0 / len2.sqrt() } else { 0.0 };
            let (tx, ty) = (ex * inv_len, ey * inv_len);
            best_off = (x - px) * ty - (y - py) * tx;
        }
    }
    (best_s, best_off)
}

/// Configuration for the racing environment.
#[derive(Debug, Clone)]
pub struct EnvConfig {
    /// Simulation time step per RL step (s). Default: 0.01 (100Hz control).
    pub dt: f64,
    /// Number of RK4 sub-steps per RL step. Default: 4.
    pub n_substeps: usize,
    /// Maximum episode length in seconds. Default: 300.0 (5 minutes).
    pub max_episode_time: f64,
    /// Maximum lateral offset before termination (multiples of half-width). Default: 2.0.
    pub max_offset_ratio: f64,
    /// Minimum speed before "stuck" timer starts (m/s). Default: 1.0.
    pub min_speed: f64,
    /// Time below min_speed before termination (s). Default: 5.0.
    pub stuck_timeout: f64,
    /// Number of laps per episode. Default: 1.
    pub n_laps: usize,
    /// Maximum steering angle (rad). Default: 0.5.
    pub max_steer_angle: f64,
    /// Observation normalization constants.
    pub obs_norm: ObsNormalization,
    /// Reward function weights.
    pub reward: RewardConfig,
}

impl Default for EnvConfig {
    fn default() -> Self {
        EnvConfig {
            dt: 0.01,
            n_substeps: 4,
            max_episode_time: 300.0,
            max_offset_ratio: 2.0,
            min_speed: 1.0,
            stuck_timeout: 5.0,
            n_laps: 1,
            max_steer_angle: 0.5,
            obs_norm: ObsNormalization::default(),
            reward: RewardConfig::default(),
        }
    }
}

/// Additional information returned by each step.
#[derive(Debug, Clone)]
pub struct StepInfo {
    /// Current speed (m/s).
    pub speed: f64,
    /// Current lap number.
    pub lap: u32,
    /// Current lap time (s).
    pub lap_time: f64,
    /// Distance along the track (m).
    pub track_distance: f64,
    /// Lateral offset from centerline (m).
    pub lateral_offset: f64,
    /// Whether the episode ended due to going off-track.
    pub off_track: bool,
    /// Whether the episode ended due to being stuck.
    pub stuck: bool,
    /// Whether the episode ended due to timeout.
    pub timeout: bool,
    /// Whether the episode completed successfully (finished n_laps).
    pub completed: bool,
    /// Per-component breakdown of this step's reward, for logging/debugging.
    pub reward_breakdown: RewardOutput,
}

/// Result of an environment step.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Observation vector after the step.
    pub observation: [f64; OBS_DIM],
    /// Reward for this step.
    pub reward: f64,
    /// Whether the episode has ended.
    pub done: bool,
    /// Additional information for logging.
    pub info: StepInfo,
}

/// Racing environment for reinforcement learning.
///
/// Wraps the 14-DOF vehicle dynamics model with a gym-style interface.
/// Each step advances the simulation by `dt` seconds and returns an
/// observation, reward, and done flag.
pub struct RaceEnv {
    // Configuration.
    config: EnvConfig,

    // Physics models (owned, not shared -- each env is independent). The
    // `FourteenDofModel` borrows these, so it is rebuilt on demand rather than
    // stored (which would make `RaceEnv` self-referential).
    car: CarParams,
    tire: PacejkaTire,
    suspension: SuspensionSystem,
    aero: AeroModel,
    powertrain: Powertrain,
    /// Peak tire longitudinal grip, used to traction-limit the drive torque.
    grip_mu: f64,

    // Track.
    track: Track,
    track_length: f64,

    // State.
    state: [f64; 24],
    sim_time: f64,
    track_distance: f64,
    lateral_offset: f64,
    /// Track heading at the current projected position (rad).
    track_heading: f64,
    prev_action: [f64; ACT_DIM],
    /// Forward speed at the end of the previous step (m/s).
    prev_speed: f64,
    /// Longitudinal acceleration at the last step (g).
    accel_long: f64,
    /// Lateral acceleration at the last step (g).
    accel_lat: f64,
    lap: u32,
    lap_start_time: f64,
    stuck_timer: f64,

    // For reward computation.
    prev_track_distance: f64,
}

impl RaceEnv {
    /// Create a new environment for the given track.
    pub fn new(track: Track, config: EnvConfig) -> Self {
        let car = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let suspension = SuspensionSystem::f1_default();
        let aero = AeroModel::f1_default();
        let powertrain = Powertrain::f1_2024();
        let grip_mu = tire_peak_longitudinal_mu(&tire, car.mass * GRAVITY / 4.0);
        let track_length = track.total_length;

        let mut env = RaceEnv {
            config,
            car,
            tire,
            suspension,
            aero,
            powertrain,
            grip_mu,
            track,
            track_length,
            state: [0.0; 24],
            sim_time: 0.0,
            track_distance: 0.0,
            lateral_offset: 0.0,
            track_heading: 0.0,
            prev_action: [0.0; ACT_DIM],
            prev_speed: START_SPEED,
            accel_long: 0.0,
            accel_lat: 0.0,
            lap: 1,
            lap_start_time: 0.0,
            stuck_timer: 0.0,
            prev_track_distance: 0.0,
        };
        env.reset();
        env
    }

    /// Create a new environment with default config.
    pub fn with_defaults(track: Track) -> Self {
        Self::new(track, EnvConfig::default())
    }

    /// Build a 14-DOF model borrowing this environment's physics parameters,
    /// trimmed at [`START_SPEED`].
    fn build_model(&self) -> FourteenDofModel<'_> {
        FourteenDofModel::new(
            &self.car,
            &self.tire,
            &self.suspension,
            &self.aero,
            START_SPEED,
        )
    }

    /// Reset the environment to the start of a new episode.
    ///
    /// Places the car at the start line at [`START_SPEED`] in the track
    /// direction (suspension at equilibrium, wheels matching road speed) and
    /// returns the initial observation.
    pub fn reset(&mut self) -> [f64; OBS_DIM] {
        let (x0, y0) = self.track.position_at(0.0);
        let heading = self.track.heading_at(0.0);

        let mut state = {
            let model = self.build_model();
            equilibrium_state(&model, START_SPEED)
        };
        state[0] = x0;
        state[1] = y0;
        state[5] = heading;

        self.state = state;
        self.sim_time = 0.0;
        self.track_distance = 0.0;
        self.prev_track_distance = 0.0;
        self.lateral_offset = 0.0;
        self.track_heading = heading;
        self.prev_action = [0.0; ACT_DIM];
        self.prev_speed = START_SPEED;
        self.accel_long = 0.0;
        self.accel_lat = 0.0;
        self.lap = 1;
        self.lap_start_time = 0.0;
        self.stuck_timer = 0.0;

        self.observe()
    }

    /// Map a normalized action to the physical 14-DOF control vector
    /// `[steering_angle, drive_torque, brake_pressure]`.
    ///
    /// Mirrors the sim server's input mapping: steering scales by
    /// `max_steer_angle`, the throttle is converted to a traction-limited wheel
    /// drive torque by the powertrain, and the brake passes through unchanged.
    fn map_action_to_control(&mut self, steer_n: f64, throttle: f64, brake: f64) -> [f64; 3] {
        let drive_wheel_omega = 0.5 * (self.state[14] + self.state[15]);
        let raw_drive = self.powertrain.drive_torque(throttle, drive_wheel_omega);
        let drive = raw_drive.min(traction_torque_limit(
            &self.car,
            self.grip_mu,
            self.state[6],
        ));
        [steer_n * self.config.max_steer_angle, drive, brake]
    }

    /// Take one step in the environment.
    ///
    /// Action: `[steering, throttle, brake]` where:
    ///   - steering in `[-1, 1]` (mapped to `[-max_steer_angle, max_steer_angle]`)
    ///   - throttle in `[0, 1]`
    ///   - brake in `[0, 1]`
    ///
    /// Advances the simulation by `dt` seconds (using `n_substeps` RK4 steps),
    /// then returns the new observation, reward, done flag, and info.
    pub fn step(&mut self, action: &[f64; ACT_DIM]) -> StepResult {
        // (1) Clamp the action.
        let steer_n = action[0].clamp(-1.0, 1.0);
        let throttle = action[1].clamp(0.0, 1.0);
        let brake = action[2].clamp(0.0, 1.0);

        // (2) Map to physical controls.
        let control = self.map_action_to_control(steer_n, throttle, brake);

        let prev_speed = self.prev_speed;

        // (3) Integrate with n_substeps RK4 steps.
        {
            let model = self.build_model();
            let n = self.config.n_substeps.max(1);
            let sub_dt = self.config.dt / n as f64;
            let mut state = self.state;
            let mut t = self.sim_time;
            for _ in 0..n {
                state = rk4_step(&model, &state, &control, t, sub_dt);
                t += sub_dt;
            }
            self.state = state;
        }
        self.sim_time += self.config.dt;

        // (4) Project the new position onto the track.
        let (s, off) = project_onto_track(&self.track, self.state[0], self.state[1]);
        self.lateral_offset = off;
        self.track_heading = self.track.heading_at(s);

        // (5) Forward progress this step (m), with start/finish wrapping, plus
        // lap counting on a forward wrap.
        let raw_delta = s - self.prev_track_distance;
        let mut progress = raw_delta;
        let mut crossed = false;
        if raw_delta < -0.5 * self.track_length {
            progress += self.track_length;
            crossed = true;
        } else if raw_delta > 0.5 * self.track_length {
            progress -= self.track_length;
        }
        if crossed {
            self.lap = self.lap.saturating_add(1);
            self.lap_start_time = self.sim_time;
        }
        self.track_distance = s;
        self.prev_track_distance = s;

        // Derived accelerations (g).
        let speed = self.state[6];
        self.accel_long = (speed - prev_speed) / self.config.dt / GRAVITY;
        self.accel_lat = speed * self.state[11] / GRAVITY;
        self.prev_speed = speed;

        // (6) Termination conditions.
        let (wl, wr) = self.track.width_at(s);
        let half_width = (0.5 * (wl + wr)).max(1e-3);
        let off_track = self.lateral_offset.abs() > self.config.max_offset_ratio * half_width;

        if speed < self.config.min_speed {
            self.stuck_timer += self.config.dt;
        } else {
            self.stuck_timer = 0.0;
        }
        let stuck = self.stuck_timer > self.config.stuck_timeout;
        let timeout = self.sim_time > self.config.max_episode_time;
        let completed = self.lap > self.config.n_laps as u32;
        let done = off_track || stuck || timeout || completed;

        // (7) Shaped reward. A crash is an off-track or stuck (spin) termination;
        // running out of time or completing the laps is not penalized.
        let heading_error = normalize_angle(self.state[5] - self.track_heading);
        let reward_input = RewardInput {
            progress,
            track_length: self.track_length,
            speed,
            v_max: self.config.obs_norm.v_max,
            lateral_offset: off,
            half_width,
            heading_error,
            steering_change: steer_n - self.prev_action[0],
            crashed: off_track || stuck,
        };
        let reward_breakdown = compute_reward(&reward_input, &self.config.reward);
        let reward = reward_breakdown.total;

        // (8) Record action, (9) extract observation.
        self.prev_action = [steer_n, throttle, brake];
        let observation = self.observe();

        let info = StepInfo {
            speed,
            lap: self.lap,
            lap_time: self.sim_time - self.lap_start_time,
            track_distance: s,
            lateral_offset: off,
            off_track,
            stuck,
            timeout,
            completed,
            reward_breakdown,
        };
        StepResult {
            observation,
            reward,
            done,
            info,
        }
    }

    /// Get the current observation without stepping.
    pub fn observe(&self) -> [f64; OBS_DIM] {
        extract_observation(
            &self.state,
            &self.track,
            self.track_distance,
            self.lateral_offset,
            self.track_heading,
            self.accel_long,
            self.accel_lat,
            &self.prev_action,
            &self.config.obs_norm,
        )
    }

    /// Get the track length.
    pub fn track_length(&self) -> f64 {
        self.track_length
    }

    /// Get the current simulation time.
    pub fn sim_time(&self) -> f64 {
        self.sim_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_track::{build_track, oval_track};

    /// Build a closed oval track for environment tests.
    fn oval() -> Track {
        let (pts, closed) = oval_track(200.0, 50.0, 12.0, 400);
        build_track("oval", &pts, closed)
    }

    /// A narrower, tighter oval for off-track testing.
    fn tight_oval() -> Track {
        let (pts, closed) = oval_track(100.0, 30.0, 10.0, 400);
        build_track("tight-oval", &pts, closed)
    }

    #[test]
    fn test_env_creation() {
        let _env = RaceEnv::with_defaults(oval());
    }

    #[test]
    fn test_reset_returns_valid_obs() {
        let mut env = RaceEnv::with_defaults(oval());
        let obs = env.reset();
        assert_eq!(obs.len(), OBS_DIM);
        for (i, v) in obs.iter().enumerate() {
            assert!(v.is_finite(), "obs[{i}] = {v} not finite");
        }
    }

    #[test]
    fn test_step_advances_time() {
        let mut env = RaceEnv::with_defaults(oval());
        env.reset();
        let t0 = env.sim_time();
        env.step(&[0.0, 0.0, 0.0]);
        assert!((env.sim_time() - t0 - env.config.dt).abs() < 1e-12);
    }

    #[test]
    fn test_step_returns_valid_result() {
        let mut env = RaceEnv::with_defaults(oval());
        env.reset();
        let result = env.step(&[0.0, 0.5, 0.0]);
        assert!(result.reward.is_finite(), "reward not finite");
        for (i, v) in result.observation.iter().enumerate() {
            assert!(v.is_finite(), "obs[{i}] = {v} not finite");
        }
        assert!(result.info.speed.is_finite());
        assert!(result.info.speed > 0.0, "car should be moving forward");
        assert!(result.info.lap >= 1);
    }

    #[test]
    fn test_throttle_increases_speed() {
        let mut env = RaceEnv::with_defaults(oval());
        env.reset();
        let v0 = env.state[6];
        let mut last = v0;
        for _ in 0..100 {
            let r = env.step(&[0.0, 1.0, 0.0]);
            last = r.info.speed;
        }
        assert!(
            env.state.iter().all(|v| v.is_finite()),
            "state went non-finite under throttle"
        );
        assert!(
            last > v0,
            "full throttle should increase speed: {v0} -> {last}"
        );
    }

    #[test]
    fn test_off_track_terminates() {
        let mut env = RaceEnv::with_defaults(tight_oval());
        env.reset();
        let mut done = false;
        let mut off_track = false;
        for _ in 0..3000 {
            let r = env.step(&[1.0, 0.6, 0.0]); // full-lock steer, moderate throttle
            if r.done {
                done = true;
                off_track = r.info.off_track;
                break;
            }
        }
        assert!(done, "extreme steering should terminate the episode");
        assert!(off_track, "termination should be due to going off-track");
    }

    #[test]
    fn test_progress_reward_positive() {
        let mut env = RaceEnv::with_defaults(oval());
        env.reset();
        let r = env.step(&[0.0, 1.0, 0.0]);
        assert!(
            r.reward > 0.0,
            "forward progress should give positive reward, got {}",
            r.reward
        );
        // The shaped reward's total must match its breakdown, and with the
        // default progress-only config the forward-progress component drives it.
        assert!(
            (r.reward - r.info.reward_breakdown.total).abs() < 1e-12,
            "reward {} should equal breakdown total {}",
            r.reward,
            r.info.reward_breakdown.total
        );
        assert!(
            r.info.reward_breakdown.progress > 0.0,
            "progress component should be positive, got {}",
            r.info.reward_breakdown.progress
        );
        assert_eq!(
            r.info.reward_breakdown.crash, 0.0,
            "no crash on a normal forward step"
        );
    }

    #[test]
    fn test_multiple_resets() {
        let mut env = RaceEnv::with_defaults(oval());
        env.reset();
        for _ in 0..10 {
            env.step(&[0.2, 0.8, 0.0]);
        }
        assert!(env.sim_time() > 0.0);

        let obs = env.reset();
        assert!((env.sim_time()).abs() < 1e-12, "sim_time should reset to 0");
        assert_eq!(env.lap, 1, "lap should reset to 1");
        assert!((env.track_distance).abs() < 1e-12, "distance should reset");
        assert!((env.stuck_timer).abs() < 1e-12, "stuck timer should reset");
        for (i, v) in obs.iter().enumerate() {
            assert!(v.is_finite(), "obs[{i}] = {v} not finite after reset");
        }
    }
}
