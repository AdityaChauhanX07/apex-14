//! Policy (actor) and value (critic) networks for PPO.
//!
//! Both are small MLPs using candle. The policy outputs a diagonal
//! Gaussian distribution over continuous actions. The value network
//! outputs a scalar state value estimate.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use crate::observation::{ACT_DIM, OBS_DIM};

/// Width of the first two hidden layers.
const HIDDEN_WIDE: usize = 128;
/// Width of the third hidden layer.
const HIDDEN_NARROW: usize = 64;

/// Half of `log(2*pi)`, the additive constant in a 1-D Gaussian log-density.
fn half_log_2pi() -> f64 {
    0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// Compute log probability of `x` under a diagonal Gaussian
/// `N(mean, exp(log_std)^2)`.
///
/// Returns element-wise log probabilities (same shape as `x`).
fn gaussian_log_prob(x: &Tensor, mean: &Tensor, log_std: &Tensor) -> Result<Tensor> {
    let std = log_std.exp()?;
    let diff = (x - mean)?;
    let z = (diff / std)?;
    // -0.5 * z^2 - log_std - 0.5 * log(2*pi)
    (z.sqr()? * (-0.5))?.sub(log_std)? - half_log_2pi()
}

/// Count the trainable parameters (weights + biases) of a set of linear layers.
fn count_params(layers: &[&Linear]) -> usize {
    let mut total = 0usize;
    for layer in layers {
        total += layer.weight().elem_count();
        if let Some(bias) = layer.bias() {
            total += bias.elem_count();
        }
    }
    total
}

/// Policy network that outputs a diagonal Gaussian distribution.
///
/// Architecture:
///   - Linear(OBS_DIM -> 128) + ReLU
///   - Linear(128 -> 128) + ReLU
///   - Linear(128 -> 64) + ReLU
///   - Linear(64 -> ACT_DIM * 2) -- [mean_0..mean_2, log_std_0..log_std_2]
///
/// The output is split into means and log standard deviations.
/// Actions are sampled from `N(mean, exp(log_std)^2)` during training,
/// or taken as the mean during evaluation.
pub struct PolicyNet {
    l1: Linear,
    l2: Linear,
    l3: Linear,
    l_out: Linear,
}

impl PolicyNet {
    /// Build the policy network.
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let l1 = linear(OBS_DIM, HIDDEN_WIDE, vb.pp("l1"))?;
        let l2 = linear(HIDDEN_WIDE, HIDDEN_WIDE, vb.pp("l2"))?;
        let l3 = linear(HIDDEN_WIDE, HIDDEN_NARROW, vb.pp("l3"))?;
        let l_out = linear(HIDDEN_NARROW, ACT_DIM * 2, vb.pp("l_out"))?;
        Ok(Self { l1, l2, l3, l_out })
    }

    /// Forward pass. Returns raw output `[batch, ACT_DIM * 2]`.
    /// First `ACT_DIM` values are means, last `ACT_DIM` are log_stds.
    pub fn forward(&self, obs: &Tensor) -> Result<Tensor> {
        let x = self.l1.forward(obs)?.relu()?;
        let x = self.l2.forward(&x)?.relu()?;
        let x = self.l3.forward(&x)?.relu()?;
        self.l_out.forward(&x)
    }

    /// Get action distribution parameters from observations.
    ///
    /// Returns `(means, log_stds)` each of shape `[batch, ACT_DIM]`. The log
    /// standard deviations are clamped to `[-5, 2]` to prevent numerical issues.
    pub fn get_distribution(&self, obs: &Tensor) -> Result<(Tensor, Tensor)> {
        let output = self.forward(obs)?;
        let means = output.narrow(1, 0, ACT_DIM)?;
        let log_stds = output.narrow(1, ACT_DIM, ACT_DIM)?;
        let log_stds = log_stds.clamp(-5.0, 2.0)?;
        Ok((means, log_stds))
    }

    /// Sample actions from the policy (for training).
    ///
    /// Returns `(actions, log_probs)` where `actions` of shape `[batch, ACT_DIM]`
    /// are the raw (pre-squash) samples from the diagonal Gaussian and `log_probs`
    /// of shape `[batch]` are their Gaussian log-densities summed over the action
    /// dimensions. Map the raw actions to valid control ranges with
    /// [`postprocess_actions`].
    pub fn sample_action(&self, obs: &Tensor, rng: &mut StdRng) -> Result<(Tensor, Tensor)> {
        let (means, log_stds) = self.get_distribution(obs)?;
        let stds = log_stds.exp()?;

        // z = mean + std * epsilon, epsilon ~ N(0, 1). Candle's CPU `randn_like`
        // draws from an unseedable RNG, so we sample from an explicit seeded
        // `rng` instead to keep training reproducible.
        let n = means.elem_count();
        let noise: Vec<f32> = (0..n)
            .map(|_| {
                let z: f32 = StandardNormal.sample(rng);
                z
            })
            .collect();
        let epsilon = Tensor::from_vec(noise, means.shape(), means.device())?;
        let raw_actions = (&means + (stds * epsilon)?)?;

        let log_probs = gaussian_log_prob(&raw_actions, &means, &log_stds)?;
        let log_probs = log_probs.sum(1)?; // [batch]

        Ok((raw_actions, log_probs))
    }

    /// Get the deterministic action (for evaluation).
    ///
    /// Returns the distribution mean of shape `[batch, ACT_DIM]` (raw, pre-squash;
    /// map to control ranges with [`postprocess_actions`]).
    pub fn deterministic_action(&self, obs: &Tensor) -> Result<Tensor> {
        let (means, _) = self.get_distribution(obs)?;
        Ok(means)
    }

    /// Compute log probability of given actions under the current policy.
    ///
    /// Used during the PPO update to compute the importance ratio. Returns a
    /// tensor of shape `[batch]`.
    pub fn log_prob(&self, obs: &Tensor, actions: &Tensor) -> Result<Tensor> {
        let (means, log_stds) = self.get_distribution(obs)?;
        let log_probs = gaussian_log_prob(actions, &means, &log_stds)?;
        log_probs.sum(1) // [batch]
    }

    /// Compute entropy of the policy distribution (for the entropy bonus).
    ///
    /// Returns a tensor of shape `[batch]`.
    pub fn entropy(&self, obs: &Tensor) -> Result<Tensor> {
        let (_, log_stds) = self.get_distribution(obs)?;
        // Entropy of a diagonal Gaussian = sum_i [ log_std_i + 0.5*(1 + log(2*pi)) ].
        let entropy = (log_stds + 0.5 * (1.0 + (2.0 * std::f64::consts::PI).ln()))?.sum(1)?;
        Ok(entropy)
    }

    /// Count the total number of trainable parameters.
    pub fn param_count(&self) -> usize {
        count_params(&[&self.l1, &self.l2, &self.l3, &self.l_out])
    }
}

/// Value network that estimates state value `V(s)`.
///
/// Architecture:
///   - Linear(OBS_DIM -> 128) + ReLU
///   - Linear(128 -> 128) + ReLU
///   - Linear(128 -> 64) + ReLU
///   - Linear(64 -> 1)
pub struct ValueNet {
    l1: Linear,
    l2: Linear,
    l3: Linear,
    l_out: Linear,
}

impl ValueNet {
    /// Build the value network.
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let l1 = linear(OBS_DIM, HIDDEN_WIDE, vb.pp("l1"))?;
        let l2 = linear(HIDDEN_WIDE, HIDDEN_WIDE, vb.pp("l2"))?;
        let l3 = linear(HIDDEN_WIDE, HIDDEN_NARROW, vb.pp("l3"))?;
        let l_out = linear(HIDDEN_NARROW, 1, vb.pp("l_out"))?;
        Ok(Self { l1, l2, l3, l_out })
    }

    /// Forward pass. Returns value estimates `[batch, 1]`.
    pub fn forward(&self, obs: &Tensor) -> Result<Tensor> {
        let x = self.l1.forward(obs)?.relu()?;
        let x = self.l2.forward(&x)?.relu()?;
        let x = self.l3.forward(&x)?.relu()?;
        self.l_out.forward(&x)
    }

    /// Get scalar value for each observation in the batch. Returns `[batch]`.
    pub fn value(&self, obs: &Tensor) -> Result<Tensor> {
        let v = self.forward(obs)?;
        v.squeeze(1) // [batch]
    }

    /// Count the total number of trainable parameters.
    pub fn param_count(&self) -> usize {
        count_params(&[&self.l1, &self.l2, &self.l3, &self.l_out])
    }
}

/// Convert raw network outputs to valid action ranges.
///
/// - steering: `tanh(raw)` -> `[-1, 1]` (zero-centered, raw 0 = straight)
/// - throttle: `clamp(raw, 0, 1)` -> `[0, 1]` (raw 0 = no throttle)
/// - brake: `clamp(raw, 0, 1)` -> `[0, 1]` (raw 0 = no brake)
///
/// Clamping (rather than `sigmoid`) is used for throttle/brake so the neutral
/// raw action `0` maps to *no* throttle and *no* brake. `sigmoid(0) = 0.5` would
/// instead apply 50% throttle and 50% brake simultaneously at initialization,
/// stalling the car and making the policy very hard to train. Because PPO's
/// log-prob is computed on the raw (pre-squash) actions, the clamp's zero
/// gradient outside `[0, 1]` does not affect the policy-gradient path.
///
/// Takes a tensor whose last dimension is `ACT_DIM` (e.g. `[batch, 3]` or `[3]`)
/// and returns the same shape.
pub fn postprocess_actions(raw: &Tensor) -> Result<Tensor> {
    let last = raw.dims().len() - 1;
    let steering = raw.narrow(last, 0, 1)?.tanh()?;
    let throttle = raw.narrow(last, 1, 1)?.clamp(0.0, 1.0)?;
    let brake = raw.narrow(last, 2, 1)?.clamp(0.0, 1.0)?;
    Tensor::cat(&[&steering, &throttle, &brake], last)
}

/// Overwrite every variable in `var_map` with values from a small
/// deterministic xorshift64 PRNG seeded by `seed`, producing reproducible
/// initial weights.
///
/// candle-core 0.10's CPU backend cannot be seeded (`Device::set_seed` bails
/// with "cannot seed the CPU rng with set_seed") because weight init draws
/// from `rand::rng()`, seeded from OS entropy. Re-initializing the weights
/// ourselves after `VarBuilder` construction sidesteps that by never touching
/// candle's unseedable RNG.
///
/// Variables are sorted by name before the PRNG stream is consumed:
/// [`VarMap`] stores them in a `HashMap` whose iteration order is randomized
/// per process, so consuming the stream in that order would assign different
/// values to different-shaped variables on every run, silently reintroducing
/// nondeterminism. Sorting by name first gives a stable, reproducible
/// assignment.
///
/// This mirrors `apex_ml::train::seed_var_map`; it is duplicated here rather
/// than shared because `apex-rl` does not (and should not) depend on
/// `apex-ml`, and the helper is too small to justify a new crate.
pub fn seed_var_map(var_map: &VarMap, seed: u64) {
    let mut state = seed | 1;
    let mut next_u64 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let map = var_map.data().lock().unwrap();
    let mut named_vars: Vec<(&String, &candle_core::Var)> = map.iter().collect();
    named_vars.sort_by(|a, b| a.0.cmp(b.0));
    for (_, var) in named_vars {
        let dims = var.dims().to_vec();
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|_| {
                let bits = next_u64();
                let unit = (bits >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
                ((unit * 2.0 - 1.0) * 0.1) as f32 // [-0.1, 0.1)
            })
            .collect();
        let t = Tensor::from_vec(data, dims, var.device()).expect("reinit tensor");
        var.set(&t).expect("set var");
    }
}

/// Save policy and value network weights to a safetensors file.
pub fn save_agent(var_map: &VarMap, path: &std::path::Path) -> Result<()> {
    var_map.save(path)
}

/// Load policy and value networks from a safetensors file.
///
/// The networks are rebuilt first (registering their variables in a fresh
/// [`VarMap`]) and then the stored weights are loaded into them; this ordering
/// is required because [`VarMap::load`] only updates variables already present
/// in the map. Returns the loaded policy, value network, and the backing
/// [`VarMap`] (kept alive so the variables remain valid for further training).
pub fn load_agent(path: &std::path::Path) -> Result<(PolicyNet, ValueNet, VarMap)> {
    let device = Device::Cpu;
    let mut var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let policy = PolicyNet::new(vb.pp("policy"))?;
    let value = ValueNet::new(vb.pp("value"))?;
    var_map.load(path)?;
    Ok((policy, value, var_map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    /// Build a fresh policy network with its own VarMap.
    fn make_policy() -> (PolicyNet, VarMap) {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let policy = PolicyNet::new(vb.pp("policy")).expect("policy construction");
        (policy, var_map)
    }

    /// Build a fresh value network with its own VarMap.
    fn make_value() -> (ValueNet, VarMap) {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let value = ValueNet::new(vb.pp("value")).expect("value construction");
        (value, var_map)
    }

    /// A zero observation batch of the given size.
    fn obs_batch(batch: usize) -> Tensor {
        Tensor::zeros((batch, OBS_DIM), DType::F32, &Device::Cpu).expect("zeros")
    }

    #[test]
    fn test_policy_construction() {
        let _ = make_policy();
    }

    #[test]
    fn test_policy_forward_shape() {
        let (policy, _vm) = make_policy();
        let out = policy.forward(&obs_batch(1)).expect("forward");
        assert_eq!(out.dims(), &[1, ACT_DIM * 2]);
    }

    #[test]
    fn test_policy_forward_batch() {
        let (policy, _vm) = make_policy();
        let out = policy.forward(&obs_batch(4)).expect("forward");
        assert_eq!(out.dims(), &[4, ACT_DIM * 2]);
    }

    #[test]
    fn test_policy_sample_action() {
        use rand::SeedableRng;
        let (policy, _vm) = make_policy();
        let mut rng = StdRng::seed_from_u64(0);
        let (actions, log_probs) = policy
            .sample_action(&obs_batch(1), &mut rng)
            .expect("sample");
        assert_eq!(actions.dims(), &[1, ACT_DIM]);
        assert_eq!(log_probs.dims(), &[1]);
        let lp: Vec<f32> = log_probs.to_vec1().expect("to_vec1");
        assert!(lp.iter().all(|v| v.is_finite()), "log_probs not finite");
    }

    #[test]
    fn test_policy_deterministic() {
        let (policy, _vm) = make_policy();
        let action = policy.deterministic_action(&obs_batch(1)).expect("det");
        assert_eq!(action.dims(), &[1, ACT_DIM]);
    }

    #[test]
    fn test_policy_log_prob() {
        let (policy, _vm) = make_policy();
        let obs = obs_batch(2);
        let actions = Tensor::zeros((2, ACT_DIM), DType::F32, &Device::Cpu).expect("zeros");
        let lp = policy.log_prob(&obs, &actions).expect("log_prob");
        assert_eq!(lp.dims(), &[2]);
        let vals: Vec<f32> = lp.to_vec1().expect("to_vec1");
        assert!(vals.iter().all(|v| v.is_finite()), "log_prob not finite");
    }

    #[test]
    fn test_policy_entropy() {
        let (policy, _vm) = make_policy();
        let ent = policy.entropy(&obs_batch(1)).expect("entropy");
        assert_eq!(ent.dims(), &[1]);
        let vals: Vec<f32> = ent.to_vec1().expect("to_vec1");
        assert!(vals[0].is_finite(), "entropy not finite");
        assert!(
            vals[0] > 0.0,
            "Gaussian entropy should be positive, got {}",
            vals[0]
        );
    }

    #[test]
    fn test_value_construction() {
        let _ = make_value();
    }

    #[test]
    fn test_value_forward_shape() {
        let (value, _vm) = make_value();
        let v = value.value(&obs_batch(1)).expect("value");
        assert_eq!(v.dims(), &[1]);
    }

    #[test]
    fn test_value_forward_batch() {
        let (value, _vm) = make_value();
        let v = value.value(&obs_batch(4)).expect("value");
        assert_eq!(v.dims(), &[4]);
    }

    #[test]
    fn test_postprocess_actions() {
        let raw = Tensor::zeros((1, ACT_DIM), DType::F32, &Device::Cpu).expect("zeros");
        let out = postprocess_actions(&raw).expect("postprocess");
        assert_eq!(out.dims(), &[1, ACT_DIM]);
        let vals: Vec<f32> = out.flatten_all().expect("flatten").to_vec1().expect("vec");
        // Neutral raw action [0,0,0] -> tanh(0)=0, clamp(0)=0, clamp(0)=0.
        // No throttle and no brake (the key fix: sigmoid would give 0.5/0.5).
        assert!((vals[0] - 0.0).abs() < 1e-6, "steering {}", vals[0]);
        assert!((vals[1] - 0.0).abs() < 1e-6, "throttle {}", vals[1]);
        assert!((vals[2] - 0.0).abs() < 1e-6, "brake {}", vals[2]);
    }

    #[test]
    fn test_full_throttle_action() {
        // Raw [0, 2, -2] -> [tanh(0)=0, clamp(2)=1, clamp(-2)=0]:
        // zero steering, full throttle, no brake.
        let raw = Tensor::from_vec(vec![0.0f32, 2.0, -2.0], (1, ACT_DIM), &Device::Cpu)
            .expect("from_vec");
        let out = postprocess_actions(&raw).expect("postprocess");
        let vals: Vec<f32> = out.flatten_all().expect("flatten").to_vec1().expect("vec");
        assert!((vals[0] - 0.0).abs() < 1e-6, "steering {}", vals[0]);
        assert!((vals[1] - 1.0).abs() < 1e-6, "throttle {}", vals[1]);
        assert!((vals[2] - 0.0).abs() < 1e-6, "brake {}", vals[2]);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let policy = PolicyNet::new(vb.pp("policy")).expect("policy");
        let value = ValueNet::new(vb.pp("value")).expect("value");

        let obs = Tensor::rand(0.0f32, 1.0f32, (3, OBS_DIM), &device).expect("rand");
        let before_p: Vec<f32> = policy
            .forward(&obs)
            .expect("fwd")
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("vec");
        let before_v: Vec<f32> = value.value(&obs).expect("val").to_vec1().expect("vec");

        let path = std::env::temp_dir().join("apex_rl_agent_roundtrip.safetensors");
        save_agent(&var_map, &path).expect("save");

        let (policy2, value2, _vm2) = load_agent(&path).expect("load");
        let after_p: Vec<f32> = policy2
            .forward(&obs)
            .expect("fwd")
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("vec");
        let after_v: Vec<f32> = value2.value(&obs).expect("val").to_vec1().expect("vec");

        let _ = std::fs::remove_file(&path);

        for (a, b) in before_p.iter().zip(after_p.iter()) {
            assert!((a - b).abs() < 1e-6, "policy output differs: {a} vs {b}");
        }
        for (a, b) in before_v.iter().zip(after_v.iter()) {
            assert!((a - b).abs() < 1e-6, "value output differs: {a} vs {b}");
        }
    }

    #[test]
    fn test_param_count() {
        let (policy, _vp) = make_policy();
        let (value, _vv) = make_value();
        let pc = policy.param_count();
        let vc = value.param_count();
        // Exact: policy = 17*128+128 + 128*128+128 + 128*64+64 + 64*6+6 = 27462.
        //        value  = ... + 64*1+1                               = 27137.
        // (The plan's ~50K estimate is high; see deviation note.)
        assert!(
            (20_000..40_000).contains(&pc),
            "policy param count {pc} out of expected range"
        );
        assert!(
            (20_000..40_000).contains(&vc),
            "value param count {vc} out of expected range"
        );
    }
}
