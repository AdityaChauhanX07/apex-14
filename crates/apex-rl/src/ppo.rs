//! Proximal Policy Optimization (clip variant) for continuous control.
//!
//! Implements the PPO-Clip algorithm with Generalized Advantage Estimation (GAE).
//! Designed for the racing environment with continuous steering/throttle/brake actions.

use candle_core::{Device, Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

use crate::observation::{ACT_DIM, OBS_DIM};
use crate::policy::{PolicyNet, ValueNet};

/// PPO hyperparameters.
#[derive(Debug, Clone)]
pub struct PpoConfig {
    /// Steps to collect per environment per rollout. Default: 2048.
    pub n_steps: usize,
    /// Number of parallel environments. Default: 8.
    pub n_envs: usize,
    /// PPO update epochs per rollout. Default: 4.
    pub k_epochs: usize,
    /// Discount factor. Default: 0.99.
    pub gamma: f64,
    /// GAE lambda. Default: 0.95.
    pub gae_lambda: f64,
    /// PPO clip range. Default: 0.2.
    pub clip_epsilon: f64,
    /// Learning rate. Default: 3e-4.
    pub learning_rate: f64,
    /// Value loss coefficient. Default: 0.5.
    pub value_coef: f64,
    /// Entropy bonus coefficient. Default: 0.01.
    pub entropy_coef: f64,
    /// Maximum gradient norm for clipping. Default: 0.5.
    pub max_grad_norm: f64,
    /// Mini-batch size for updates. Default: 256.
    pub batch_size: usize,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            n_steps: 2048,
            n_envs: 8,
            k_epochs: 4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            learning_rate: 3e-4,
            value_coef: 0.5,
            entropy_coef: 0.01,
            max_grad_norm: 0.5,
            batch_size: 256,
        }
    }
}

/// Buffer storing a complete rollout for PPO training.
///
/// Stores transitions from n_envs environments, each running n_steps.
/// Total transitions = n_envs * n_steps.
#[derive(Debug, Clone, Default)]
pub struct RolloutBuffer {
    /// Observations `[n_total, OBS_DIM]`.
    pub observations: Vec<[f64; OBS_DIM]>,
    /// Actions `[n_total, ACT_DIM]`.
    pub actions: Vec<[f64; ACT_DIM]>,
    /// Log probabilities of actions under the old policy.
    pub log_probs: Vec<f64>,
    /// Rewards received.
    pub rewards: Vec<f64>,
    /// Value estimates V(s) from the critic.
    pub values: Vec<f64>,
    /// Episode done flags.
    pub dones: Vec<bool>,
    /// Computed advantages (filled by [`compute_gae`]).
    pub advantages: Vec<f64>,
    /// Computed returns (filled by [`compute_gae`]).
    pub returns: Vec<f64>,
}

impl RolloutBuffer {
    /// Create an empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a buffer with pre-allocated capacity.
    pub fn with_capacity(n_total: usize) -> Self {
        Self {
            observations: Vec::with_capacity(n_total),
            actions: Vec::with_capacity(n_total),
            log_probs: Vec::with_capacity(n_total),
            rewards: Vec::with_capacity(n_total),
            values: Vec::with_capacity(n_total),
            dones: Vec::with_capacity(n_total),
            advantages: Vec::with_capacity(n_total),
            returns: Vec::with_capacity(n_total),
        }
    }

    /// Add a transition to the buffer.
    pub fn push(
        &mut self,
        obs: [f64; OBS_DIM],
        action: [f64; ACT_DIM],
        log_prob: f64,
        reward: f64,
        value: f64,
        done: bool,
    ) {
        self.observations.push(obs);
        self.actions.push(action);
        self.log_probs.push(log_prob);
        self.rewards.push(reward);
        self.values.push(value);
        self.dones.push(done);
    }

    /// Number of transitions stored.
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Clear all stored data.
    pub fn clear(&mut self) {
        self.observations.clear();
        self.actions.clear();
        self.log_probs.clear();
        self.rewards.clear();
        self.values.clear();
        self.dones.clear();
        self.advantages.clear();
        self.returns.clear();
    }
}

/// Compute Generalized Advantage Estimation.
///
/// Fills the buffer's `advantages` and `returns` fields. Must be called after
/// the rollout is complete, before the PPO update.
///
/// The `last_values` slice contains the value estimates V(s_T) for the state
/// after the last step of each environment (needed for bootstrapping). Length
/// should equal `n_envs` (missing entries are treated as 0).
///
/// Transitions in the buffer are assumed to be stored in environment-major
/// order: `[env0_step0, env0_step1, ..., env0_stepN, env1_step0, ...]`.
///
/// The `returns` are the (un-normalized) value targets `advantage + value`. The
/// `advantages` are normalized to zero mean and unit variance across the whole
/// buffer (subtract mean, divide by `std + 1e-8`), as is standard for PPO.
pub fn compute_gae(
    buffer: &mut RolloutBuffer,
    last_values: &[f64],
    n_envs: usize,
    gamma: f64,
    gae_lambda: f64,
) {
    let n_total = buffer.len();
    buffer.advantages = vec![0.0; n_total];
    buffer.returns = vec![0.0; n_total];
    if n_envs == 0 || n_total == 0 {
        return;
    }
    let t_per_env = n_total / n_envs;
    if t_per_env == 0 {
        return;
    }

    for e in 0..n_envs {
        let base = e * t_per_env;
        let last_value = last_values.get(e).copied().unwrap_or(0.0);
        let mut next_advantage = 0.0;
        for t in (0..t_per_env).rev() {
            let idx = base + t;
            let mask = if buffer.dones[idx] { 0.0 } else { 1.0 };
            let next_value = if t == t_per_env - 1 {
                last_value
            } else {
                buffer.values[idx + 1]
            };
            let delta = buffer.rewards[idx] + gamma * next_value * mask - buffer.values[idx];
            let advantage = delta + gamma * gae_lambda * mask * next_advantage;
            buffer.advantages[idx] = advantage;
            buffer.returns[idx] = advantage + buffer.values[idx];
            next_advantage = advantage;
        }
    }

    normalize_advantages(&mut buffer.advantages);
}

/// Normalize advantages in place to zero mean and unit variance.
fn normalize_advantages(advantages: &mut [f64]) {
    let n = advantages.len();
    if n == 0 {
        return;
    }
    let mean = advantages.iter().sum::<f64>() / n as f64;
    let var = advantages
        .iter()
        .map(|a| (a - mean) * (a - mean))
        .sum::<f64>()
        / n as f64;
    let std = var.sqrt();
    for a in advantages.iter_mut() {
        *a = (*a - mean) / (std + 1e-8);
    }
}

/// Result of a PPO update step.
#[derive(Debug, Clone)]
pub struct PpoUpdateResult {
    /// Mean policy loss across mini-batches.
    pub policy_loss: f64,
    /// Mean value loss across mini-batches.
    pub value_loss: f64,
    /// Mean entropy across mini-batches.
    pub entropy: f64,
    /// Mean KL divergence (approx) between old and new policy.
    pub approx_kl: f64,
    /// Fraction of transitions where the ratio was clipped.
    pub clip_fraction: f64,
}

/// Deterministic Fisher-Yates shuffle of `0..n`, seeded by `seed`.
///
/// Uses a small linear-congruential generator; the mini-batch shuffle does not
/// need to be cryptographically random, only reproducible and well-mixed.
fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = ((state >> 33) as usize) % (i + 1);
        idx.swap(i, j);
    }
    idx
}

/// Build an `[rows, cols]` f32 tensor by gathering rows of `data` at `indices`.
fn gather_rows(data: &[f64], indices: &[usize], cols: usize, device: &Device) -> Result<Tensor> {
    let mut flat = Vec::with_capacity(indices.len() * cols);
    for &i in indices {
        // `data` is laid out row-major; this helper is used for the flat reward /
        // log-prob style vectors via `cols == 1` and for 2-D arrays separately.
        flat.push(data[i] as f32);
    }
    Tensor::from_vec(flat, (indices.len(), cols), device)
}

/// Build an `[rows, width]` f32 tensor by gathering fixed-size rows at `indices`.
fn gather_arrays<const W: usize>(
    data: &[[f64; W]],
    indices: &[usize],
    device: &Device,
) -> Result<Tensor> {
    let mut flat = Vec::with_capacity(indices.len() * W);
    for &i in indices {
        for v in &data[i] {
            flat.push(*v as f32);
        }
    }
    Tensor::from_vec(flat, (indices.len(), W), device)
}

/// Perform the PPO policy and value update.
///
/// Takes a filled rollout buffer (with computed advantages and returns) and
/// updates the policy and value networks using the clipped PPO objective.
///
/// Iterates over the buffer `k_epochs` times, shuffling and splitting into
/// mini-batches each epoch. A single [`AdamW`] optimizer over the shared
/// [`VarMap`] updates both networks from the combined loss.
pub fn ppo_update(
    buffer: &RolloutBuffer,
    policy: &PolicyNet,
    value_net: &ValueNet,
    var_map: &VarMap,
    config: &PpoConfig,
) -> Result<PpoUpdateResult> {
    let device = Device::Cpu;
    let n = buffer.len();

    let mut result = PpoUpdateResult {
        policy_loss: 0.0,
        value_loss: 0.0,
        entropy: 0.0,
        approx_kl: 0.0,
        clip_fraction: 0.0,
    };
    if n == 0 {
        return Ok(result);
    }

    let params = ParamsAdamW {
        lr: config.learning_rate,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let mut optimizer = AdamW::new(var_map.all_vars(), params)?;

    let batch_size = config.batch_size.max(1);
    let mut n_batches = 0usize;

    for epoch in 0..config.k_epochs {
        let order = shuffled_indices(n, epoch as u64 + 1);
        for chunk in order.chunks(batch_size) {
            // Gather the mini-batch tensors.
            let obs = gather_arrays(&buffer.observations, chunk, &device)?;
            let actions = gather_arrays(&buffer.actions, chunk, &device)?;
            let old_log_probs = gather_rows(&buffer.log_probs, chunk, 1, &device)?.squeeze(1)?;
            let advantages = gather_rows(&buffer.advantages, chunk, 1, &device)?.squeeze(1)?;
            let returns = gather_rows(&buffer.returns, chunk, 1, &device)?.squeeze(1)?;

            // Clipped surrogate policy loss.
            let new_log_probs = policy.log_prob(&obs, &actions)?;
            let ratio = (&new_log_probs - &old_log_probs)?.exp()?;
            let surr1 = (&ratio * &advantages)?;
            let clipped = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)?;
            let surr2 = (&clipped * &advantages)?;
            let policy_loss = surr1.minimum(&surr2)?.mean_all()?.neg()?;

            // Value loss (MSE) and entropy bonus.
            let new_values = value_net.value(&obs)?;
            let value_loss = (&new_values - &returns)?.sqr()?.mean_all()?;
            let entropy = policy.entropy(&obs)?.mean_all()?;

            // Diagnostics (read before the tensors are consumed by the loss).
            let pl = policy_loss.to_scalar::<f32>()? as f64;
            let vl = value_loss.to_scalar::<f32>()? as f64;
            let ent = entropy.to_scalar::<f32>()? as f64;
            let (kl, clip_frac) =
                kl_and_clip(&ratio, &new_log_probs, &old_log_probs, config.clip_epsilon)?;

            // total = policy_loss + value_coef * value_loss - entropy_coef * entropy.
            let total_loss = policy_loss
                .add(&(value_loss * config.value_coef)?)?
                .sub(&(entropy * config.entropy_coef)?)?;
            optimizer.backward_step(&total_loss)?;

            result.policy_loss += pl;
            result.value_loss += vl;
            result.entropy += ent;
            result.approx_kl += kl;
            result.clip_fraction += clip_frac;
            n_batches += 1;
        }
    }

    if n_batches > 0 {
        let denom = n_batches as f64;
        result.policy_loss /= denom;
        result.value_loss /= denom;
        result.entropy /= denom;
        result.approx_kl /= denom;
        result.clip_fraction /= denom;
    }
    Ok(result)
}

/// Compute the approximate KL divergence and clip fraction for one mini-batch.
///
/// `approx_kl = mean(old_log_prob - new_log_prob)` and
/// `clip_fraction = mean(|ratio - 1| > epsilon)`. Diagnostics only.
fn kl_and_clip(
    ratio: &Tensor,
    new_log_probs: &Tensor,
    old_log_probs: &Tensor,
    epsilon: f64,
) -> Result<(f64, f64)> {
    let ratios: Vec<f32> = ratio.to_vec1()?;
    let new_lp: Vec<f32> = new_log_probs.to_vec1()?;
    let old_lp: Vec<f32> = old_log_probs.to_vec1()?;
    let n = ratios.len().max(1) as f64;

    let kl: f64 = old_lp
        .iter()
        .zip(new_lp.iter())
        .map(|(o, nw)| (*o - *nw) as f64)
        .sum::<f64>()
        / n;
    let clip: f64 = ratios
        .iter()
        .filter(|r| ((**r - 1.0) as f64).abs() > epsilon)
        .count() as f64
        / n;
    Ok((kl, clip))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{PolicyNet, ValueNet};
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    fn zeros_obs() -> [f64; OBS_DIM] {
        [0.0; OBS_DIM]
    }

    /// Reference (independent) raw GAE for a single environment.
    fn reference_gae(
        rewards: &[f64],
        values: &[f64],
        dones: &[bool],
        last_value: f64,
        gamma: f64,
        lambda: f64,
    ) -> Vec<f64> {
        let t = rewards.len();
        let mut adv = vec![0.0; t];
        let mut next_adv = 0.0;
        for i in (0..t).rev() {
            let mask = if dones[i] { 0.0 } else { 1.0 };
            let next_value = if i == t - 1 {
                last_value
            } else {
                values[i + 1]
            };
            let delta = rewards[i] + gamma * next_value * mask - values[i];
            adv[i] = delta + gamma * lambda * mask * next_adv;
            next_adv = adv[i];
        }
        adv
    }

    fn normalized(mut v: Vec<f64>) -> Vec<f64> {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let var = v.iter().map(|a| (a - mean) * (a - mean)).sum::<f64>() / n;
        let std = var.sqrt();
        for a in v.iter_mut() {
            *a = (*a - mean) / (std + 1e-8);
        }
        v
    }

    #[test]
    fn test_rollout_buffer_push_and_len() {
        let mut buf = RolloutBuffer::new();
        for i in 0..10 {
            buf.push(zeros_obs(), [0.0; ACT_DIM], 0.0, i as f64, 0.0, false);
        }
        assert_eq!(buf.len(), 10);
        assert!(!buf.is_empty());
        buf.clear();
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_compute_gae_simple() {
        let rewards = [1.0, 2.0, 3.0];
        let values = [0.5, 1.5, 2.5];
        let dones = [false, false, false];
        let last_value = 3.0;
        let (gamma, lambda) = (0.99, 0.95);

        let mut buf = RolloutBuffer::new();
        for i in 0..3 {
            buf.push(
                zeros_obs(),
                [0.0; ACT_DIM],
                0.0,
                rewards[i],
                values[i],
                dones[i],
            );
        }
        compute_gae(&mut buf, &[last_value], 1, gamma, lambda);

        let expected = normalized(reference_gae(
            &rewards, &values, &dones, last_value, gamma, lambda,
        ));
        for (got, exp) in buf.advantages.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-6, "advantage {got} vs {exp}");
        }
        // returns = raw advantage + value (un-normalized).
        let raw = reference_gae(&rewards, &values, &dones, last_value, gamma, lambda);
        for i in 0..3 {
            let exp_ret = raw[i] + values[i];
            assert!(
                (buf.returns[i] - exp_ret).abs() < 1e-6,
                "return {} vs {}",
                buf.returns[i],
                exp_ret
            );
        }
    }

    #[test]
    fn test_compute_gae_with_done() {
        let rewards = [1.0, 2.0, 3.0];
        let values = [0.5, 1.5, 2.5];
        let dones = [false, true, false];
        let last_value = 3.0;
        let (gamma, lambda) = (0.99, 0.95);

        let mut buf = RolloutBuffer::new();
        for i in 0..3 {
            buf.push(
                zeros_obs(),
                [0.0; ACT_DIM],
                0.0,
                rewards[i],
                values[i],
                dones[i],
            );
        }
        compute_gae(&mut buf, &[last_value], 1, gamma, lambda);

        let raw = reference_gae(&rewards, &values, &dones, last_value, gamma, lambda);
        let expected = normalized(raw.clone());
        for (got, exp) in buf.advantages.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-6, "advantage {got} vs {exp}");
        }
        // The done at step 1 cuts the value bootstrap: raw adv[1] = reward[1] - value[1].
        assert!(
            (raw[1] - (rewards[1] - values[1])).abs() < 1e-12,
            "done boundary should not bootstrap: {} vs {}",
            raw[1],
            rewards[1] - values[1]
        );
    }

    #[test]
    fn test_compute_gae_normalization() {
        let mut buf = RolloutBuffer::new();
        let rewards = [0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 3.0, 1.0];
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        for i in 0..8 {
            buf.push(
                zeros_obs(),
                [0.0; ACT_DIM],
                0.0,
                rewards[i],
                values[i],
                false,
            );
        }
        compute_gae(&mut buf, &[0.0], 1, 0.99, 0.95);

        let n = buf.advantages.len() as f64;
        let mean = buf.advantages.iter().sum::<f64>() / n;
        let var = buf
            .advantages
            .iter()
            .map(|a| (a - mean) * (a - mean))
            .sum::<f64>()
            / n;
        let std = var.sqrt();
        assert!(mean.abs() < 1e-6, "advantage mean {mean} should be ~0");
        assert!((std - 1.0).abs() < 1e-4, "advantage std {std} should be ~1");
    }

    /// Build a policy + value net sharing one VarMap.
    fn make_agent() -> (PolicyNet, ValueNet, VarMap) {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let policy = PolicyNet::new(vb.pp("policy")).expect("policy");
        let value = ValueNet::new(vb.pp("value")).expect("value");
        (policy, value, var_map)
    }

    /// Build a synthetic buffer of `n` random transitions with GAE filled.
    fn synthetic_buffer(n: usize) -> RolloutBuffer {
        let device = Device::Cpu;
        let mut buf = RolloutBuffer::with_capacity(n);
        for i in 0..n {
            let o = Tensor::rand(0.0f32, 1.0f32, OBS_DIM, &device).expect("rand");
            let ov: Vec<f32> = o.to_vec1().expect("vec");
            let mut obs = [0.0f64; OBS_DIM];
            for (d, s) in obs.iter_mut().zip(ov.iter()) {
                *d = *s as f64;
            }
            let action = [0.1, 0.2, -0.1];
            let reward = ((i % 5) as f64) * 0.5 - 1.0;
            let value = 0.3;
            buf.push(obs, action, -1.0, reward, value, i % 17 == 16);
        }
        compute_gae(&mut buf, &[0.0], 1, 0.99, 0.95);
        buf
    }

    #[test]
    fn test_ppo_update_runs() {
        let (policy, value, var_map) = make_agent();
        let buf = synthetic_buffer(64);
        let config = PpoConfig {
            k_epochs: 1,
            batch_size: 32,
            ..PpoConfig::default()
        };
        let res = ppo_update(&buf, &policy, &value, &var_map, &config).expect("update");
        assert!(res.policy_loss.is_finite(), "policy_loss not finite");
        assert!(res.value_loss.is_finite(), "value_loss not finite");
        assert!(res.entropy.is_finite(), "entropy not finite");
        assert!(res.approx_kl.is_finite(), "approx_kl not finite");
        assert!(res.clip_fraction.is_finite(), "clip_fraction not finite");
        assert!(
            (0.0..=1.0).contains(&res.clip_fraction),
            "clip_fraction {} out of [0,1]",
            res.clip_fraction
        );
    }

    #[test]
    fn test_ppo_update_changes_weights() {
        let (policy, value, var_map) = make_agent();
        let device = Device::Cpu;
        let obs = Tensor::rand(0.0f32, 1.0f32, (4, OBS_DIM), &device).expect("rand");
        let before: Vec<f32> = policy
            .forward(&obs)
            .expect("fwd")
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("vec");

        let buf = synthetic_buffer(64);
        let config = PpoConfig {
            k_epochs: 3,
            batch_size: 32,
            ..PpoConfig::default()
        };
        ppo_update(&buf, &policy, &value, &var_map, &config).expect("update");

        let after: Vec<f32> = policy
            .forward(&obs)
            .expect("fwd")
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("vec");

        let changed = before
            .iter()
            .zip(after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(changed, "PPO update should change policy outputs");
    }

    #[test]
    fn test_ppo_config_default() {
        let c = PpoConfig::default();
        assert_eq!(c.n_steps, 2048);
        assert_eq!(c.n_envs, 8);
        assert_eq!(c.k_epochs, 4);
        assert_eq!(c.gamma, 0.99);
        assert_eq!(c.gae_lambda, 0.95);
        assert_eq!(c.clip_epsilon, 0.2);
        assert_eq!(c.learning_rate, 3e-4);
        assert_eq!(c.value_coef, 0.5);
        assert_eq!(c.entropy_coef, 0.01);
        assert_eq!(c.max_grad_norm, 0.5);
        assert_eq!(c.batch_size, 256);
    }
}
