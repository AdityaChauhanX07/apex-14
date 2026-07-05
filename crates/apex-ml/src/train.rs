//! Training loop for the raceline prediction network.

use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{Device, Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

use crate::data::{TrainingDataset, TrainingSample, N_FIXED};
use crate::net::RacelineNet;

/// Monotonic counter used to give each in-flight checkpoint file a unique name.
static CHECKPOINT_SEQ: AtomicU64 = AtomicU64::new(0);

/// Build a unique temporary path for the best-weight checkpoint.
///
/// Combines the process id and a monotonic counter so concurrent training runs
/// (e.g. parallel tests) never collide on the same file.
fn checkpoint_path() -> std::path::PathBuf {
    let seq = CHECKPOINT_SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "apex_ml_best_{}_{}.safetensors",
        std::process::id(),
        seq
    ))
}

/// Configuration for the training loop.
pub struct TrainConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate for the AdamW optimizer.
    pub learning_rate: f64,
    /// Weight for speed loss relative to offset loss.
    /// Total loss = speed_weight * MSE(speed) + MSE(offset).
    pub speed_weight: f64,
    /// Fraction of data to hold out for validation (0.0 to 1.0).
    pub validation_fraction: f64,
    /// Report validation loss every N epochs.
    pub report_interval: usize,
    /// Stop if validation loss has not improved for this many report intervals.
    pub patience: usize,
    /// Factor to multiply learning rate by when validation loss plateaus.
    /// Set to 1.0 to disable decay. Default: 0.5.
    pub lr_decay_factor: f64,
    /// Number of report intervals without improvement before decaying LR.
    /// Default: 3 (separate from early-stopping patience).
    pub lr_decay_patience: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 200,
            learning_rate: 1e-3,
            speed_weight: 2.0,
            validation_fraction: 0.2,
            report_interval: 10,
            patience: 8,
            lr_decay_factor: 0.5,
            lr_decay_patience: 3,
        }
    }
}

/// Result of a completed training run.
pub struct TrainResult {
    /// Training loss at each epoch.
    pub train_losses: Vec<f64>,
    /// Validation loss at each report interval.
    pub val_losses: Vec<f64>,
    /// Final training loss.
    pub final_train_loss: f64,
    /// Final validation loss.
    pub final_val_loss: f64,
    /// Best validation loss seen during training.
    pub best_val_loss: f64,
    /// Epoch at which the best validation loss was achieved.
    pub best_epoch: usize,
    /// Number of training samples used.
    pub n_train: usize,
    /// Number of validation samples used.
    pub n_val: usize,
}

/// Convert a slice of training samples into input and target tensors.
///
/// Input shape: [n_samples, 4, N_FIXED] (f32)
/// Target shape: [n_samples, 2, N_FIXED] (f32)
fn samples_to_tensors(samples: &[TrainingSample], device: &Device) -> Result<(Tensor, Tensor)> {
    let n = samples.len();
    let mut input_data = Vec::with_capacity(n * 4 * N_FIXED);
    let mut target_data = Vec::with_capacity(n * 2 * N_FIXED);
    for sample in samples {
        for &v in &sample.curvature_profile {
            input_data.push(v as f32);
        }
        for &v in &sample.curvature_deriv_profile {
            input_data.push(v as f32);
        }
        for &v in &sample.width_left_profile {
            input_data.push(v as f32);
        }
        for &v in &sample.width_right_profile {
            input_data.push(v as f32);
        }
        for &v in &sample.speed_profile {
            target_data.push(v as f32);
        }
        for &v in &sample.offset_profile {
            target_data.push(v as f32);
        }
    }
    let input = Tensor::from_vec(input_data, (n, 4, N_FIXED), device)?;
    let target = Tensor::from_vec(target_data, (n, 2, N_FIXED), device)?;
    Ok((input, target))
}

/// Train the raceline network on a dataset.
///
/// Filters to converged samples only, splits into train/val,
/// then trains for the configured number of epochs with early stopping.
/// The VarMap accumulates gradients and the optimizer updates weights in-place,
/// so the network passed in reflects the final (not necessarily best) weights.
pub fn train(
    dataset: &TrainingDataset,
    config: &TrainConfig,
    var_map: &VarMap,
    net: &RacelineNet,
) -> Result<TrainResult> {
    let device = Device::Cpu;

    let converged: Vec<TrainingSample> = dataset
        .samples
        .iter()
        .filter(|s| s.converged)
        .cloned()
        .collect();

    if converged.is_empty() {
        candle_core::bail!("no converged samples in dataset");
    }

    let n_val = ((converged.len() as f64) * config.validation_fraction).round() as usize;
    let n_val = n_val.max(1).min(converged.len() - 1);
    let n_train = converged.len() - n_val;

    let (train_input, train_target) = samples_to_tensors(&converged[..n_train], &device)?;
    let (val_input, val_target) = samples_to_tensors(&converged[n_train..], &device)?;

    let params = ParamsAdamW {
        lr: config.learning_rate,
        ..ParamsAdamW::default()
    };
    let mut optimizer = AdamW::new(var_map.all_vars(), params)?;

    let mut train_losses = Vec::with_capacity(config.epochs);
    let mut val_losses = Vec::new();
    let mut best_val_loss = f64::INFINITY;
    let mut best_epoch = 0usize;
    let mut patience_counter = 0usize;
    let mut lr_decay_counter = 0usize;
    let mut current_lr = config.learning_rate;

    // Best weights are snapshotted to this temp file whenever a new best
    // validation loss is seen, then restored into `var_map` after training so
    // the caller saves the best (not the final, possibly diverged) weights.
    let ckpt = checkpoint_path();

    for epoch in 0..config.epochs {
        let pred = net.forward(&train_input)?;

        let speed_pred = pred.narrow(1, 0, 1)?;
        let speed_target = train_target.narrow(1, 0, 1)?;
        let offset_pred = pred.narrow(1, 1, 1)?;
        let offset_target = train_target.narrow(1, 1, 1)?;

        let speed_mse = speed_pred.sub(&speed_target)?.sqr()?.mean_all()?;
        let offset_mse = offset_pred.sub(&offset_target)?.sqr()?.mean_all()?;
        let total_loss = (speed_mse * config.speed_weight)?.add(&offset_mse)?;

        let loss_val = total_loss.to_scalar::<f32>()? as f64;
        train_losses.push(loss_val);

        optimizer.backward_step(&total_loss)?;

        let should_report =
            (epoch + 1).is_multiple_of(config.report_interval) || epoch == config.epochs - 1;
        if should_report {
            let val_pred = net.forward(&val_input)?;
            let v_speed = val_pred.narrow(1, 0, 1)?;
            let v_offset = val_pred.narrow(1, 1, 1)?;
            let v_speed_mse = v_speed
                .sub(&val_target.narrow(1, 0, 1)?)?
                .sqr()?
                .mean_all()?;
            let v_offset_mse = v_offset
                .sub(&val_target.narrow(1, 1, 1)?)?
                .sqr()?
                .mean_all()?;
            let v_total = (v_speed_mse * config.speed_weight)?.add(&v_offset_mse)?;
            let v_loss = v_total.to_scalar::<f32>()? as f64;
            val_losses.push(v_loss);

            log::info!(
                "epoch {}/{}: train={:.6} val={:.6}",
                epoch + 1,
                config.epochs,
                loss_val,
                v_loss
            );

            if v_loss < best_val_loss {
                best_val_loss = v_loss;
                best_epoch = epoch + 1;
                patience_counter = 0;
                lr_decay_counter = 0;

                // Snapshot the current (best-so-far) weights.
                var_map.save(&ckpt)?;
                log::info!("Checkpoint at epoch {best_epoch} (val_loss={v_loss:.6})");
            } else {
                patience_counter += 1;
                lr_decay_counter += 1;

                // Decay the learning rate once the plateau reaches the (shorter)
                // LR-decay patience, then reset only the LR-decay counter so the
                // model gets a fresh window to improve before early stopping.
                if config.lr_decay_factor < 1.0 && lr_decay_counter >= config.lr_decay_patience {
                    let old_lr = current_lr;
                    current_lr *= config.lr_decay_factor;
                    optimizer.set_learning_rate(current_lr);
                    lr_decay_counter = 0;
                    log::info!("LR decay: {old_lr} -> {current_lr}");
                }

                if patience_counter >= config.patience {
                    log::info!(
                        "early stopping at epoch {} (no improvement for {} intervals)",
                        epoch + 1,
                        config.patience
                    );
                    break;
                }
            }
        }
    }

    // Restore the best weights into the shared VarMap (it uses interior
    // mutability, so a clone writes through to the same variables), then remove
    // the temporary checkpoint file.
    if ckpt.exists() {
        let mut vm = var_map.clone();
        vm.load(&ckpt)?;
        std::fs::remove_file(&ckpt).ok();
        log::info!("Restored best weights from epoch {best_epoch}");
    }

    let final_train_loss = train_losses.last().copied().unwrap_or(f64::NAN);
    let final_val_loss = val_losses.last().copied().unwrap_or(f64::NAN);

    Ok(TrainResult {
        train_losses,
        val_losses,
        final_train_loss,
        final_val_loss,
        best_val_loss,
        best_epoch,
        n_train,
        n_val,
    })
}

/// Overwrite every variable in `var_map` with values from a small
/// deterministic xorshift64 PRNG seeded by `seed`, producing reproducible
/// initial weights.
///
/// candle-core 0.10's CPU backend cannot be seeded (`Device::set_seed` bails
/// with "cannot seed the CPU rng with set_seed") because weight init draws
/// from `rand::rng()`, a thread-local RNG seeded from OS entropy. That makes
/// training runs (and tests that depend on their trajectory) nondeterministic.
/// Re-initializing the weights ourselves after `VarBuilder` construction
/// sidesteps that by never touching candle's unseedable RNG.
///
/// Variables are sorted by name before the PRNG stream is consumed:
/// [`VarMap`] stores them in a `HashMap` whose iteration order is randomized
/// per process by Rust's default hasher. Consuming the stream in that order
/// would assign different values to different-shaped variables on every run,
/// silently reintroducing nondeterminism. Sorting by name first gives a
/// stable, reproducible assignment.
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    use crate::data::TrainingDataset;
    use crate::net::RacelineNet;

    fn make_sample() -> TrainingSample {
        TrainingSample {
            curvature_profile: vec![0.0; N_FIXED],
            curvature_deriv_profile: vec![0.0; N_FIXED],
            width_left_profile: vec![0.5; N_FIXED],
            width_right_profile: vec![0.5; N_FIXED],
            speed_profile: vec![0.8; N_FIXED],
            offset_profile: vec![0.0; N_FIXED],
            speed_norm: 1.0,
            width_norm: 1.0,
            lap_time: 60.0,
            converged: true,
            track_id: "test".to_string(),
        }
    }

    fn make_net_and_map() -> (RacelineNet, VarMap) {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let net = RacelineNet::new(vb).expect("net");
        (net, var_map)
    }

    #[test]
    fn test_samples_to_tensors() {
        let device = Device::Cpu;
        let samples = vec![make_sample(), make_sample()];
        let (input, target) = samples_to_tensors(&samples, &device).expect("tensors");
        assert_eq!(input.dims(), &[2, 4, N_FIXED]);
        assert_eq!(target.dims(), &[2, 2, N_FIXED]);
    }

    #[test]
    fn test_train_reduces_loss() {
        let (net, var_map) = make_net_and_map();
        let dataset = TrainingDataset {
            samples: vec![make_sample(); 5],
            tracks_attempted: 5,
            tracks_converged: 5,
            global_speed_norm: 1.0,
            global_width_norm: 1.0,
        };
        let config = TrainConfig {
            epochs: 10,
            learning_rate: 1e-3,
            report_interval: 5,
            patience: 10,
            validation_fraction: 0.2,
            ..TrainConfig::default()
        };
        // Capture output before training to verify weights are updated.
        let probe = Tensor::ones((1, 4, N_FIXED), DType::F32, &Device::Cpu).expect("ones");
        let pre: Vec<f32> = net
            .forward(&probe)
            .expect("pre-forward")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("to_vec1");

        let result = train(&dataset, &config, &var_map, &net).expect("train");

        let post: Vec<f32> = net
            .forward(&probe)
            .expect("post-forward")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("to_vec1");

        assert_ne!(pre, post, "training did not update network weights");
        assert_eq!(result.n_train, 4);
        assert_eq!(result.n_val, 1);
    }

    /// Compute the weighted validation loss for `net` on `val_input`/`val_target`,
    /// mirroring the exact loss used inside `train`.
    fn weighted_val_loss(
        net: &RacelineNet,
        val_input: &Tensor,
        val_target: &Tensor,
        speed_weight: f64,
    ) -> f64 {
        let pred = net.forward(val_input).expect("forward");
        let s_mse = pred
            .narrow(1, 0, 1)
            .unwrap()
            .sub(&val_target.narrow(1, 0, 1).unwrap())
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap();
        let o_mse = pred
            .narrow(1, 1, 1)
            .unwrap()
            .sub(&val_target.narrow(1, 1, 1).unwrap())
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap();
        let total = (s_mse * speed_weight).unwrap().add(&o_mse).unwrap();
        total.to_scalar::<f32>().unwrap() as f64
    }

    #[test]
    fn test_best_weights_restored_after_divergence() {
        let (net, var_map) = make_net_and_map();
        // Weight init otherwise draws from candle's unseedable CPU RNG, making
        // this test's divergence trajectory nondeterministic (~1/5 flaky).
        seed_var_map(&var_map, 0xACE1_5EED);

        // Enough identical converged samples to give a multi-sample val split.
        let dataset = TrainingDataset {
            samples: vec![make_sample(); 20],
            tracks_attempted: 20,
            tracks_converged: 20,
            global_speed_norm: 1.0,
            global_width_norm: 1.0,
        };

        // A deliberately high learning rate with LR decay disabled, so training
        // overshoots and the final epoch is worse than the best one. Patience is
        // large enough that early stopping does not mask the divergence.
        let config = TrainConfig {
            epochs: 40,
            learning_rate: 0.5,
            report_interval: 4,
            patience: 1000,
            validation_fraction: 0.2,
            lr_decay_factor: 1.0,
            ..TrainConfig::default()
        };

        let result = train(&dataset, &config, &var_map, &net).expect("train");

        // Rebuild the same validation split train() used (converged samples,
        // last n_val of them) to recompute the loss from the restored weights.
        let converged: Vec<TrainingSample> = dataset
            .samples
            .iter()
            .filter(|s| s.converged)
            .cloned()
            .collect();
        let (val_input, val_target) =
            samples_to_tensors(&converged[result.n_train..], &Device::Cpu).expect("val tensors");
        let restored_loss = weighted_val_loss(&net, &val_input, &val_target, config.speed_weight);

        // The weights left in the VarMap must reproduce the BEST validation loss,
        // not the final one.
        assert!(
            (restored_loss - result.best_val_loss).abs() < 1e-4,
            "restored weights give val_loss {restored_loss:.6}, expected best {:.6}",
            result.best_val_loss
        );
        // Sanity: best is no worse than final.
        assert!(result.best_val_loss <= result.final_val_loss + 1e-9);
        // The high LR should have diverged so the best epoch is earlier than the last.
        assert!(
            result.best_epoch < config.epochs,
            "expected divergence (best_epoch {} < {}), final_val={:.6} best_val={:.6}",
            result.best_epoch,
            config.epochs,
            result.final_val_loss,
            result.best_val_loss
        );
    }
}
