//! Configurable reward function for the racing environment.
//!
//! The reward encourages forward progress while penalizing unsafe
//! or inefficient driving. All components are individually weighted
//! for easy tuning.

/// Weights for the reward function components.
///
/// The total reward per step is:
///   `w_progress * progress_speed + w_speed * speed_bonus + w_alive`
///   `- w_offset * offset_penalty - w_heading * heading_penalty`
///   `- w_jerk * jerk_penalty - w_crash * crash_penalty`
///
/// Start with progress-only reward (set all penalties to 0) and add
/// penalties incrementally to fix specific failure modes.
#[derive(Debug, Clone)]
pub struct RewardConfig {
    /// Weight for forward progress along track (primary signal).
    /// The progress term is the forward speed (`progress / dt`) normalized by
    /// `v_max`, i.e. roughly `[0, 1]` per step — on the same scale as the other
    /// terms (the old `progress / track_length` form was orders of magnitude
    /// smaller than the crash penalty, so the agent could never learn from it).
    pub w_progress: f64,
    /// Weight for speed bonus. Bonus = speed / v_max per step.
    pub w_speed: f64,
    /// Weight for lateral offset penalty. Penalty = (offset / half_width)^2.
    pub w_offset: f64,
    /// Weight for heading error penalty. Penalty = (heading_error / pi)^2.
    pub w_heading: f64,
    /// Weight for steering jerk penalty. Penalty = |steering_change|.
    pub w_jerk: f64,
    /// Penalty applied when the episode terminates due to going off-track or spinning.
    pub w_crash: f64,
    /// Weight for centering bonus at high speed. Rewards being near centerline
    /// when going fast (encourages using the full track only when beneficial).
    pub w_centering: f64,
    /// Small per-step reward for being alive (not crashed). Default: 0.01.
    /// Helps the agent learn that surviving is better than crashing early.
    pub w_alive: f64,
}

impl RewardConfig {
    /// Progress-only reward. Start here for initial training.
    /// The simplest reward that can produce lap completion.
    pub fn progress_only() -> Self {
        Self {
            w_progress: 1.0,
            w_speed: 0.0,
            w_offset: 0.0,
            w_heading: 0.0,
            w_jerk: 0.0,
            w_crash: 2.0,
            w_centering: 0.0,
            w_alive: 0.01,
        }
    }

    /// Balanced reward for stable driving. Use after the agent can
    /// complete laps with progress-only.
    pub fn balanced() -> Self {
        Self {
            w_progress: 1.0,
            w_speed: 0.2,
            w_offset: 0.1,
            w_heading: 0.05,
            w_jerk: 0.01,
            w_crash: 10.0,
            w_centering: 0.0,
            w_alive: 0.01,
        }
    }

    /// Racing reward that encourages aggressive, fast driving.
    /// Use only after the agent drives consistently with balanced reward.
    pub fn racing() -> Self {
        Self {
            w_progress: 1.0,
            w_speed: 0.5,
            w_offset: 0.05,
            w_heading: 0.02,
            w_jerk: 0.005,
            w_crash: 5.0,
            w_centering: 0.1,
            w_alive: 0.01,
        }
    }
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self::progress_only()
    }
}

/// Input data for reward computation.
///
/// Collected by the environment during step() and passed to compute_reward().
#[derive(Debug, Clone)]
pub struct RewardInput {
    /// Forward progress along track this step (m). Positive = correct direction.
    pub progress: f64,
    /// Simulation time step for this RL step (s). Used to convert progress (m)
    /// into a velocity for the progress reward.
    pub dt: f64,
    /// Total track length (m).
    pub track_length: f64,
    /// Current speed (m/s).
    pub speed: f64,
    /// Maximum expected speed (m/s).
    pub v_max: f64,
    /// Current lateral offset from centerline (m).
    pub lateral_offset: f64,
    /// Track half-width at current position (m).
    pub half_width: f64,
    /// Heading error relative to track direction (rad).
    pub heading_error: f64,
    /// Change in steering action from previous step (raw, -2 to 2 range).
    pub steering_change: f64,
    /// Whether the episode terminated this step (crash/off-track).
    pub crashed: bool,
}

/// Breakdown of reward components for debugging.
#[derive(Debug, Clone, Default)]
pub struct RewardOutput {
    /// Total reward (sum of all components).
    pub total: f64,
    /// Progress component (positive).
    pub progress: f64,
    /// Speed bonus component (positive).
    pub speed: f64,
    /// Offset penalty component (negative).
    pub offset: f64,
    /// Heading penalty component (negative).
    pub heading: f64,
    /// Jerk penalty component (negative).
    pub jerk: f64,
    /// Crash penalty component (negative).
    pub crash: f64,
    /// Alive bonus component (positive while not crashed).
    pub alive: f64,
}

/// Compute the shaped reward.
///
/// Returns the total reward and a breakdown for logging/debugging.
pub fn compute_reward(input: &RewardInput, config: &RewardConfig) -> RewardOutput {
    let v_max = if input.v_max > 0.0 { input.v_max } else { 1.0 };
    let dt = if input.dt > 0.0 { input.dt } else { 1.0 };

    // Progress reward as a forward-velocity fraction: (progress / dt) / v_max,
    // clamped at zero so backward motion earns no reward. This keeps the term on
    // a ~[0, 1] per-step scale, comparable to the other reward components.
    let progress = config.w_progress * (input.progress / dt).max(0.0) / v_max;
    let speed = config.w_speed * (input.speed / v_max);

    let offset_ratio = (input.lateral_offset / input.half_width.max(1.0)).clamp(-1.0, 1.0);
    let offset = -config.w_offset * offset_ratio * offset_ratio;

    let heading_norm = (input.heading_error / std::f64::consts::PI).clamp(-1.0, 1.0);
    let heading = -config.w_heading * heading_norm * heading_norm;

    let jerk = -config.w_jerk * input.steering_change.abs();

    let crash = if input.crashed { -config.w_crash } else { 0.0 };
    let alive = if input.crashed { 0.0 } else { config.w_alive };

    let total = progress + speed + offset + heading + jerk + crash + alive;

    RewardOutput {
        total,
        progress,
        speed,
        offset,
        heading,
        jerk,
        crash,
        alive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a neutral reward input (no offset, heading, jerk, crash).
    fn neutral_input() -> RewardInput {
        RewardInput {
            progress: 0.0,
            dt: 0.01,
            track_length: 1000.0,
            speed: 0.0,
            v_max: 100.0,
            lateral_offset: 0.0,
            half_width: 6.0,
            heading_error: 0.0,
            steering_change: 0.0,
            crashed: false,
        }
    }

    #[test]
    fn test_progress_only_reward() {
        let config = RewardConfig::progress_only();
        let mut input = neutral_input();
        input.progress = 50.0;

        let out = compute_reward(&input, &config);
        // Progress term = w_progress * (progress / dt) / v_max.
        let exp_progress = config.w_progress * (50.0 / input.dt) / input.v_max;
        let exp_alive = config.w_alive; // not crashed
        assert!(out.total > 0.0, "progress should give positive reward");
        assert!(
            (out.progress - exp_progress).abs() < 1e-9,
            "progress {} vs expected {}",
            out.progress,
            exp_progress
        );
        assert!((out.alive - exp_alive).abs() < 1e-12, "alive {}", out.alive);
        assert!(
            (out.total - (exp_progress + exp_alive)).abs() < 1e-9,
            "total {} vs expected {}",
            out.total,
            exp_progress + exp_alive
        );
        // No penalty component is active with progress-only and a neutral input.
        assert_eq!(out.speed, 0.0);
        assert_eq!(out.offset, 0.0);
        assert_eq!(out.heading, 0.0);
        assert_eq!(out.jerk, 0.0);
        assert_eq!(out.crash, 0.0);
    }

    #[test]
    fn test_crash_penalty() {
        let config = RewardConfig::balanced();
        let mut input = neutral_input();
        input.crashed = true;

        let out = compute_reward(&input, &config);
        assert!(
            (out.crash + config.w_crash).abs() < 1e-12,
            "crash component {} should be -w_crash {}",
            out.crash,
            -config.w_crash
        );
        assert!(
            out.total <= out.crash + 1e-12,
            "crash should dominate total"
        );
    }

    #[test]
    fn test_offset_penalty_quadratic() {
        let config = RewardConfig::balanced();
        let half_width = 6.0;

        // 0% offset -> zero penalty.
        let mut input = neutral_input();
        input.half_width = half_width;
        input.lateral_offset = 0.0;
        assert_eq!(compute_reward(&input, &config).offset, 0.0);

        // 50% offset -> 0.25 * w_offset penalty.
        input.lateral_offset = 0.5 * half_width;
        let out_half = compute_reward(&input, &config);
        assert!(
            (out_half.offset + 0.25 * config.w_offset).abs() < 1e-12,
            "50% offset penalty {} vs {}",
            out_half.offset,
            -0.25 * config.w_offset
        );

        // 100% offset -> 1.0 * w_offset penalty.
        input.lateral_offset = half_width;
        let out_full = compute_reward(&input, &config);
        assert!(
            (out_full.offset + config.w_offset).abs() < 1e-12,
            "100% offset penalty {} vs {}",
            out_full.offset,
            -config.w_offset
        );
    }

    #[test]
    fn test_heading_penalty() {
        let config = RewardConfig::balanced();

        // Zero heading error -> zero penalty.
        let input = neutral_input();
        assert_eq!(compute_reward(&input, &config).heading, 0.0);

        // pi/2 heading error -> 0.25 * w_heading penalty.
        let mut input = neutral_input();
        input.heading_error = std::f64::consts::FRAC_PI_2;
        let out = compute_reward(&input, &config);
        assert!(
            (out.heading + 0.25 * config.w_heading).abs() < 1e-12,
            "pi/2 heading penalty {} vs {}",
            out.heading,
            -0.25 * config.w_heading
        );
    }

    #[test]
    fn test_jerk_penalty_proportional() {
        let config = RewardConfig::balanced();

        // Zero change -> zero penalty.
        let input = neutral_input();
        assert_eq!(compute_reward(&input, &config).jerk, 0.0);

        // 0.5 change -> 0.5 * w_jerk penalty (sign-independent magnitude).
        let mut input = neutral_input();
        input.steering_change = 0.5;
        let out = compute_reward(&input, &config);
        assert!(
            (out.jerk + 0.5 * config.w_jerk).abs() < 1e-12,
            "jerk penalty {} vs {}",
            out.jerk,
            -0.5 * config.w_jerk
        );
    }

    #[test]
    fn test_balanced_reward_all_components() {
        let config = RewardConfig::balanced();
        let input = RewardInput {
            progress: 30.0,
            dt: 0.01,
            track_length: 1000.0,
            speed: 50.0,
            v_max: 100.0,
            lateral_offset: 3.0,
            half_width: 6.0,
            heading_error: std::f64::consts::FRAC_PI_4,
            steering_change: 0.4,
            crashed: true,
        };

        let out = compute_reward(&input, &config);

        let exp_progress = config.w_progress * (30.0 / input.dt) / input.v_max;
        let exp_speed = config.w_speed * 50.0 / 100.0;
        let exp_offset = -config.w_offset * 0.5 * 0.5;
        let heading_norm = std::f64::consts::FRAC_PI_4 / std::f64::consts::PI;
        let exp_heading = -config.w_heading * heading_norm * heading_norm;
        let exp_jerk = -config.w_jerk * 0.4;
        let exp_crash = -config.w_crash;
        let exp_alive = 0.0; // crashed

        assert!((out.progress - exp_progress).abs() < 1e-9);
        assert!((out.speed - exp_speed).abs() < 1e-12);
        assert!((out.offset - exp_offset).abs() < 1e-12);
        assert!((out.heading - exp_heading).abs() < 1e-12);
        assert!((out.jerk - exp_jerk).abs() < 1e-12);
        assert!((out.crash - exp_crash).abs() < 1e-12);
        assert!((out.alive - exp_alive).abs() < 1e-12);

        let expected_total =
            exp_progress + exp_speed + exp_offset + exp_heading + exp_jerk + exp_crash + exp_alive;
        assert!((out.total - expected_total).abs() < 1e-9);
    }

    #[test]
    fn test_reward_output_breakdown_sums() {
        let config = RewardConfig::racing();
        let input = RewardInput {
            progress: 12.0,
            dt: 0.01,
            track_length: 800.0,
            speed: 70.0,
            v_max: 100.0,
            lateral_offset: -2.0,
            half_width: 5.0,
            heading_error: -0.3,
            steering_change: -0.7,
            crashed: false,
        };
        let out = compute_reward(&input, &config);
        let sum =
            out.progress + out.speed + out.offset + out.heading + out.jerk + out.crash + out.alive;
        assert!(
            (out.total - sum).abs() < 1e-12,
            "total {} should equal sum of components {}",
            out.total,
            sum
        );
    }
}
