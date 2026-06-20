//! Training loop orchestration for PPO.
//!
//! Manages the collect-rollouts -> compute-advantages -> PPO-update cycle
//! across multiple environments.

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};

use apex_track::Track;

use crate::env::{EnvConfig, RaceEnv};
use crate::observation::{ACT_DIM, OBS_DIM};
use crate::policy::{postprocess_actions, save_agent, PolicyNet, ValueNet};
use crate::ppo::{compute_gae, ppo_update, PpoConfig, PpoUpdateResult, RolloutBuffer};
use crate::reward::RewardConfig;

/// Configuration for the full training run.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// PPO hyperparameters.
    pub ppo: PpoConfig,
    /// Reward function weights.
    pub reward: RewardConfig,
    /// Environment configuration.
    pub env: EnvConfig,
    /// Total number of environment steps to collect.
    pub total_steps: usize,
    /// Save a checkpoint every N updates. Default: 10.
    pub save_interval: usize,
    /// Log training stats every N updates. Default: 1.
    pub log_interval: usize,
    /// Output path for the final weights.
    pub output_path: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            ppo: PpoConfig::default(),
            reward: RewardConfig::default(),
            env: EnvConfig::default(),
            total_steps: 500_000,
            save_interval: 10,
            log_interval: 1,
            output_path: "driver_policy.safetensors".to_string(),
        }
    }
}

/// Statistics for a single training iteration.
#[derive(Debug, Clone)]
pub struct IterStats {
    /// Iteration number.
    pub iteration: usize,
    /// Total environment steps so far.
    pub total_steps: usize,
    /// Mean episode reward (across completed episodes this iteration).
    pub mean_reward: f64,
    /// Mean episode length in steps.
    pub mean_episode_length: f64,
    /// Mean speed during episodes (m/s).
    pub mean_speed: f64,
    /// Number of episodes completed this iteration.
    pub episodes_completed: usize,
    /// Number of laps completed this iteration.
    pub laps_completed: usize,
    /// PPO update result.
    pub ppo_result: PpoUpdateResult,
}

/// Episode statistics gathered while collecting a rollout.
#[derive(Debug, Clone, Default)]
struct RolloutStats {
    completed_episodes: usize,
    total_reward: f64,
    total_length: usize,
    total_speed: f64,
    laps_completed: usize,
}

/// Convert an observation array to a `[1, OBS_DIM]` f32 tensor.
fn obs_to_tensor(obs: &[f64; OBS_DIM], device: &Device) -> Result<Tensor> {
    let data: Vec<f32> = obs.iter().map(|&v| v as f32).collect();
    Tensor::from_vec(data, (1, OBS_DIM), device)
}

/// Extract a fixed-size action array from a single-row tensor.
fn tensor_row_to_array<const N: usize>(t: &Tensor) -> Result<[f64; N]> {
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let mut out = [0.0; N];
    for (o, s) in out.iter_mut().zip(v.iter()) {
        *o = *s as f64;
    }
    Ok(out)
}

/// Extract the single scalar from a one-element tensor as f64.
fn scalar_f64(t: &Tensor) -> Result<f64> {
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(v.first().copied().unwrap_or(0.0) as f64)
}

/// Collect a rollout from multiple environments.
///
/// Runs each environment for `n_steps`, collecting transitions into the buffer.
/// Uses the policy network to select actions (in training/stochastic mode).
///
/// Environments run sequentially (not parallel) because the policy inference is
/// fast and we need the CPU tensor ops for each step. Transitions are stored in
/// environment-major order (each environment's `n_steps` transitions are
/// contiguous), matching the layout [`compute_gae`] expects.
///
/// The buffer stores raw (pre-squash) actions (matching what
/// [`PolicyNet::log_prob`] expects); the environment receives the post-processed
/// actions from [`postprocess_actions`].
///
/// Returns episode statistics.
fn collect_rollout(
    envs: &mut [RaceEnv],
    policy: &PolicyNet,
    value_net: &ValueNet,
    buffer: &mut RolloutBuffer,
    n_steps: usize,
) -> Result<RolloutStats> {
    let device = Device::Cpu;
    let mut stats = RolloutStats::default();

    for env in envs.iter_mut() {
        // Each environment starts a fresh episode segment at lap 1.
        let mut ep_reward = 0.0;
        let mut ep_len = 0usize;
        let mut prev_lap = 1u32;

        for _ in 0..n_steps {
            let obs = env.observe();
            let obs_t = obs_to_tensor(&obs, &device)?;

            let (raw_action, log_prob) = policy.sample_action(&obs_t)?;
            let value = value_net.value(&obs_t)?;
            let processed = postprocess_actions(&raw_action)?;

            let raw_arr: [f64; ACT_DIM] = tensor_row_to_array(&raw_action)?;
            let proc_arr: [f64; ACT_DIM] = tensor_row_to_array(&processed)?;
            let log_prob_s = scalar_f64(&log_prob)?;
            let value_s = scalar_f64(&value)?;

            let result = env.step(&proc_arr);
            buffer.push(
                obs,
                raw_arr,
                log_prob_s,
                result.reward,
                value_s,
                result.done,
            );

            ep_reward += result.reward;
            ep_len += 1;
            stats.total_speed += result.info.speed;
            if result.info.lap > prev_lap {
                stats.laps_completed += (result.info.lap - prev_lap) as usize;
                prev_lap = result.info.lap;
            }

            if result.done {
                stats.completed_episodes += 1;
                stats.total_reward += ep_reward;
                stats.total_length += ep_len;
                ep_reward = 0.0;
                ep_len = 0;
                env.reset();
                prev_lap = 1;
            }
        }
    }

    Ok(stats)
}

/// Run one full training iteration: collect a rollout, compute advantages, and
/// apply the PPO update. Returns the iteration statistics.
fn run_iteration(
    iteration: usize,
    prev_total_steps: usize,
    envs: &mut [RaceEnv],
    policy: &PolicyNet,
    value_net: &ValueNet,
    var_map: &VarMap,
    ppo: &PpoConfig,
) -> Result<IterStats> {
    let device = Device::Cpu;
    let n_envs = envs.len();
    let n_steps = ppo.n_steps.max(1);

    let mut buffer = RolloutBuffer::with_capacity(n_steps * n_envs);
    let rollout = collect_rollout(envs, policy, value_net, &mut buffer, n_steps)?;

    // Bootstrap value V(s_T) for each environment's final state.
    let mut last_values = Vec::with_capacity(n_envs);
    for env in envs.iter() {
        let obs_t = obs_to_tensor(&env.observe(), &device)?;
        last_values.push(scalar_f64(&value_net.value(&obs_t)?)?);
    }

    compute_gae(&mut buffer, &last_values, n_envs, ppo.gamma, ppo.gae_lambda);
    let ppo_result = ppo_update(&buffer, policy, value_net, var_map, ppo)?;

    let n = buffer.len();
    let episodes = rollout.completed_episodes;
    Ok(IterStats {
        iteration,
        total_steps: prev_total_steps + n,
        mean_reward: if episodes > 0 {
            rollout.total_reward / episodes as f64
        } else {
            0.0
        },
        mean_episode_length: if episodes > 0 {
            rollout.total_length as f64 / episodes as f64
        } else {
            0.0
        },
        mean_speed: rollout.total_speed / n.max(1) as f64,
        episodes_completed: episodes,
        laps_completed: rollout.laps_completed,
        ppo_result,
    })
}

/// Run the full PPO training loop.
///
/// Creates the policy and value networks, builds `ppo.n_envs` environments
/// (cycling through the provided `tracks`), and runs the collect/update cycle
/// until `total_steps` is reached. Checkpoints and the final weights are written
/// to `output_path`.
///
/// The number of iterations is `total_steps / (n_steps * n_envs)` (at least one).
pub fn train(tracks: Vec<Track>, config: TrainConfig) -> Result<()> {
    if tracks.is_empty() {
        return Err(candle_core::Error::msg("train: no tracks provided"));
    }

    let device = Device::Cpu;
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let policy = PolicyNet::new(vb.pp("policy"))?;
    let value_net = ValueNet::new(vb.pp("value"))?;

    let n_envs = config.ppo.n_envs.max(1);
    let mut envs: Vec<RaceEnv> = Vec::with_capacity(n_envs);
    for i in 0..n_envs {
        let track = tracks[i % tracks.len()].clone();
        let mut env_config = config.env.clone();
        env_config.reward = config.reward.clone();
        envs.push(RaceEnv::new(track, env_config));
    }

    let n_steps = config.ppo.n_steps.max(1);
    let transitions_per_iter = (n_steps * n_envs).max(1);
    let n_iters = (config.total_steps / transitions_per_iter).max(1);

    log::info!(
        "training: {} iterations x {} transitions ({} envs x {} steps)",
        n_iters,
        transitions_per_iter,
        n_envs,
        n_steps
    );

    let mut total_steps = 0usize;
    for iteration in 0..n_iters {
        let stats = run_iteration(
            iteration,
            total_steps,
            &mut envs,
            &policy,
            &value_net,
            &var_map,
            &config.ppo,
        )?;
        total_steps = stats.total_steps;

        if config.log_interval > 0 && iteration % config.log_interval == 0 {
            log::info!(
                "iter {:>4} | steps {:>8} | reward {:>8.3} | ep_len {:>6.1} | speed {:>5.1} | eps {:>3} | laps {:>3} | pi_loss {:>7.4} | v_loss {:>7.4} | entropy {:>6.3} | kl {:>7.4} | clip {:>5.3}",
                stats.iteration,
                stats.total_steps,
                stats.mean_reward,
                stats.mean_episode_length,
                stats.mean_speed,
                stats.episodes_completed,
                stats.laps_completed,
                stats.ppo_result.policy_loss,
                stats.ppo_result.value_loss,
                stats.ppo_result.entropy,
                stats.ppo_result.approx_kl,
                stats.ppo_result.clip_fraction,
            );
        }

        if config.save_interval > 0 && iteration > 0 && iteration % config.save_interval == 0 {
            save_agent(&var_map, Path::new(&config.output_path))?;
            log::info!("checkpoint saved to {}", config.output_path);
        }
    }

    save_agent(&var_map, Path::new(&config.output_path))?;
    log::info!("final weights saved to {}", config.output_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_track::{build_track, oval_track};

    fn oval() -> Track {
        let (pts, closed) = oval_track(200.0, 50.0, 12.0, 400);
        build_track("oval", &pts, closed)
    }

    fn make_agent() -> (PolicyNet, ValueNet, VarMap) {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let policy = PolicyNet::new(vb.pp("policy")).expect("policy");
        let value = ValueNet::new(vb.pp("value")).expect("value");
        (policy, value, var_map)
    }

    fn make_envs(n: usize) -> Vec<RaceEnv> {
        (0..n).map(|_| RaceEnv::with_defaults(oval())).collect()
    }

    #[test]
    fn test_collect_rollout() {
        let (policy, value, _vm) = make_agent();
        let mut envs = make_envs(2);
        let mut buffer = RolloutBuffer::new();
        let _stats = collect_rollout(&mut envs, &policy, &value, &mut buffer, 10).expect("rollout");

        assert_eq!(buffer.len(), 20, "2 envs x 10 steps = 20 transitions");
        assert_eq!(buffer.observations.len(), 20);
        assert_eq!(buffer.actions.len(), 20);
        for o in &buffer.observations {
            assert_eq!(o.len(), OBS_DIM);
            assert!(o.iter().all(|v| v.is_finite()), "obs not finite");
        }
        for a in &buffer.actions {
            assert_eq!(a.len(), ACT_DIM);
            assert!(a.iter().all(|v| v.is_finite()), "action not finite");
        }
        assert!(
            buffer.rewards.iter().all(|v| v.is_finite()),
            "reward not finite"
        );
        assert!(
            buffer.values.iter().all(|v| v.is_finite()),
            "value not finite"
        );
        assert!(
            buffer.log_probs.iter().all(|v| v.is_finite()),
            "log_prob not finite"
        );
    }

    #[test]
    fn test_training_iteration() {
        let (policy, value, var_map) = make_agent();
        let mut envs = make_envs(2);
        let ppo = PpoConfig {
            n_envs: 2,
            n_steps: 64,
            k_epochs: 1,
            batch_size: 32,
            ..PpoConfig::default()
        };
        let stats =
            run_iteration(0, 0, &mut envs, &policy, &value, &var_map, &ppo).expect("iteration");

        assert_eq!(stats.total_steps, 128, "2 envs x 64 steps");
        assert!(stats.mean_speed.is_finite());
        assert!(stats.mean_reward.is_finite());
        assert!(stats.mean_episode_length.is_finite());
        assert!(stats.ppo_result.policy_loss.is_finite());
        assert!(stats.ppo_result.value_loss.is_finite());
        assert!(stats.ppo_result.entropy.is_finite());
    }

    #[test]
    fn test_train_short_run() {
        let path = std::env::temp_dir().join("apex_rl_train_short.safetensors");
        let _ = std::fs::remove_file(&path);

        let config = TrainConfig {
            ppo: PpoConfig {
                n_envs: 2,
                n_steps: 64,
                k_epochs: 1,
                batch_size: 32,
                ..PpoConfig::default()
            },
            total_steps: 256,
            save_interval: 0,
            log_interval: 0,
            output_path: path.to_string_lossy().to_string(),
            ..TrainConfig::default()
        };

        train(vec![oval()], config).expect("train short run");
        assert!(path.exists(), "weights file should be saved");
        let _ = std::fs::remove_file(&path);
    }
}
