//! ML-based warmstart generation for the collocation optimizer.
//!
//! Uses the trained raceline CNN to predict speed and offset profiles
//! from track geometry, denormalizes them into physical units using the
//! constants saved alongside the model, then packs them into the optimizer's
//! decision variable format via
//! `CollocationOptimizer::initial_guess_from_profiles`. Falls back gracefully
//! on any failure.

use apex_optimizer::{CollocationConfig, CollocationOptimizer};
use apex_physics::CarParams;
use apex_track::{extract_features, Track};

use crate::data::{NormConstants, N_FIXED};
use crate::io::{load_norm_constants, meta_path};
use crate::net::{load_network, RacelineNet};

/// Predicted speed and offset profiles from the ML network.
pub struct MlProfiles {
    /// Speed profile resampled to `n_nodes` points (m/s).
    pub speeds: Vec<f64>,
    /// Lateral offset profile resampled to `n_nodes` points (m).
    pub offsets: Vec<f64>,
}

/// Linear interpolation of `ys` (sampled at strictly increasing `xs`) at `x`,
/// clamped to the endpoints.
fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let last = xs.len() - 1;
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[last] {
        return ys[last];
    }
    let mut lo = 0;
    let mut hi = last;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (x - xs[lo]) / (xs[hi] - xs[lo]);
    ys[lo] + t * (ys[hi] - ys[lo])
}

/// Predict speed and offset profiles from track geometry using the ML network.
///
/// Extracts features from the track, runs neural network inference, validates
/// the output, denormalizes it into physical units using `norm`, and resamples
/// from [`N_FIXED`] to `n_nodes` points.
///
/// The network is trained on normalized targets (speed divided by a speed
/// constant, offset divided by a width constant), so the raw output is
/// multiplied back by those constants here.
///
/// Returns `None` if inference fails or the output contains NaN/Inf values.
pub fn ml_predict_profiles(
    net: &RacelineNet,
    track: &Track,
    n_nodes: usize,
    v_min: f64,
    norm: &NormConstants,
) -> Option<MlProfiles> {
    let feat = extract_features(track, N_FIXED);

    let (speed_pred, offset_pred) = net
        .predict(
            &feat.curvature,
            &feat.curvature_deriv,
            &feat.width_left,
            &feat.width_right,
        )
        .ok()?;

    if speed_pred.iter().any(|v| !v.is_finite()) || offset_pred.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // Denormalize the network output back into physical units (m/s and m).
    let speed_pred: Vec<f64> = speed_pred.iter().map(|&v| v * norm.speed_norm).collect();
    let offset_pred: Vec<f64> = offset_pred.iter().map(|&v| v * norm.width_norm).collect();

    let feat_ds = track.total_length / N_FIXED as f64;
    let feat_stations: Vec<f64> = (0..N_FIXED).map(|i| i as f64 * feat_ds).collect();

    let node_stations: Vec<f64> = (0..n_nodes)
        .map(|k| track.total_length * (k as f64) / ((n_nodes - 1) as f64))
        .collect();

    let speeds: Vec<f64> = node_stations
        .iter()
        .map(|&s| interp(&feat_stations, &speed_pred, s).max(v_min))
        .collect();
    let offsets: Vec<f64> = node_stations
        .iter()
        .map(|&s| interp(&feat_stations, &offset_pred, s))
        .collect();

    Some(MlProfiles { speeds, offsets })
}

/// Generate an ML-based initial guess for the collocation optimizer.
///
/// Extracts track features, runs neural network inference, denormalizes the
/// predictions into physical units using `norm`, resamples them from
/// [`N_FIXED`] to `config.n_nodes`, and packs them into the optimizer's
/// decision variable vector using
/// [`CollocationOptimizer::initial_guess_from_profiles`].
///
/// Returns `None` if inference fails or the output contains NaN/Inf values.
/// The caller should fall back to the QSS initial guess in that case.
pub fn ml_initial_guess(
    net: &RacelineNet,
    track: &Track,
    config: &CollocationConfig,
    car: &CarParams,
    norm: &NormConstants,
) -> Option<Vec<f64>> {
    let profiles = ml_predict_profiles(net, track, config.n_nodes, config.v_min, norm)?;
    let optimizer = CollocationOptimizer::new(config.clone(), track, car);
    Some(optimizer.initial_guess_from_profiles(&profiles.speeds, &profiles.offsets))
}

/// Wrapper that holds a loaded network plus its normalization constants and
/// provides warmstart generation.
///
/// Designed to be constructed once and reused across multiple optimizer runs.
pub struct MlWarmstart {
    net: RacelineNet,
    norm: NormConstants,
}

impl MlWarmstart {
    /// Create from an already-loaded network and its normalization constants.
    pub fn from_net(net: RacelineNet, norm: NormConstants) -> Self {
        Self { net, norm }
    }

    /// Load a trained network from a safetensors file together with its
    /// normalization constants from the `.meta.json` sidecar.
    ///
    /// Returns `None` if either file does not exist or cannot be loaded.
    pub fn load(model_path: &std::path::Path) -> Option<Self> {
        if !model_path.exists() {
            return None;
        }
        let net = load_network(model_path).ok()?;
        let norm = load_norm_constants(&meta_path(model_path)).ok()?;
        Some(Self { net, norm })
    }

    /// Predict speed and offset profiles from track geometry.
    ///
    /// Returns `None` if inference fails or the output contains NaN/Inf values.
    pub fn predict_profiles(
        &self,
        track: &Track,
        n_nodes: usize,
        v_min: f64,
    ) -> Option<MlProfiles> {
        ml_predict_profiles(&self.net, track, n_nodes, v_min, &self.norm)
    }

    /// Generate an ML initial guess packed as a decision variable vector.
    ///
    /// Returns `None` if inference fails or the output contains NaN/Inf values.
    pub fn generate(
        &self,
        track: &Track,
        config: &CollocationConfig,
        car: &CarParams,
    ) -> Option<Vec<f64>> {
        ml_initial_guess(&self.net, track, config, car, &self.norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    use apex_track::{build_track, circle_track};

    use crate::io::{meta_path, save_norm_constants};
    use crate::net::save_weights;

    fn make_net() -> (RacelineNet, VarMap) {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let net = RacelineNet::new(vb).expect("network creation failed");
        (net, var_map)
    }

    fn test_norm() -> NormConstants {
        NormConstants {
            speed_norm: 80.0,
            width_norm: 6.0,
        }
    }

    fn make_track() -> Track {
        let (pts, closed) = circle_track(100.0, 12.0, 200);
        build_track("circle", &pts, closed)
    }

    fn make_config(n_nodes: usize) -> CollocationConfig {
        CollocationConfig {
            n_nodes,
            closed: true,
            ..CollocationConfig::default()
        }
    }

    #[test]
    fn test_ml_predict_profiles_shape() {
        let (net, _var_map) = make_net();
        let track = make_track();
        let n_nodes = 50;
        let profiles = ml_predict_profiles(&net, &track, n_nodes, 5.0, &test_norm());
        assert!(profiles.is_some(), "prediction should succeed");
        let p = profiles.unwrap();
        assert_eq!(p.speeds.len(), n_nodes);
        assert_eq!(p.offsets.len(), n_nodes);
    }

    #[test]
    fn test_ml_initial_guess_shape() {
        let (net, _var_map) = make_net();
        let track = make_track();
        let n_nodes = 50;
        let config = make_config(n_nodes);
        let car = CarParams::default();
        let guess = ml_initial_guess(&net, &track, &config, &car, &test_norm());
        assert!(guess.is_some(), "initial guess should succeed");
        let x = guess.unwrap();
        let expected_len = 7 * n_nodes - 1;
        assert_eq!(x.len(), expected_len);
    }

    #[test]
    fn test_ml_initial_guess_no_nan() {
        let (net, _var_map) = make_net();
        let track = make_track();
        let config = make_config(50);
        let car = CarParams::default();
        let x = ml_initial_guess(&net, &track, &config, &car, &test_norm()).expect("guess");
        assert!(
            x.iter().all(|v| v.is_finite()),
            "initial guess contains NaN or Inf"
        );
    }

    #[test]
    fn test_ml_warmstart_load_missing_file() {
        let result = MlWarmstart::load(std::path::Path::new("/nonexistent/model.safetensors"));
        assert!(result.is_none());
    }

    #[test]
    fn test_ml_warmstart_load_missing_meta() {
        let (_net, var_map) = make_net();
        let dir = std::env::temp_dir();
        let path = dir.join("apex_ml_warmstart_no_meta.safetensors");
        save_weights(&var_map, &path).expect("save weights");
        // Remove any stale sidecar so the meta load is guaranteed to fail.
        let _ = std::fs::remove_file(meta_path(&path));

        let warmstart = MlWarmstart::load(&path);
        assert!(
            warmstart.is_none(),
            "load should fail without norm constants sidecar"
        );
    }

    #[test]
    fn test_ml_warmstart_roundtrip() {
        let (_net, var_map) = make_net();
        let track = make_track();
        let config = make_config(50);
        let car = CarParams::default();

        let dir = std::env::temp_dir();
        let path = dir.join("apex_ml_warmstart_test.safetensors");
        save_weights(&var_map, &path).expect("save weights");
        save_norm_constants(&test_norm(), &meta_path(&path)).expect("save norm");

        let warmstart = MlWarmstart::load(&path);
        assert!(warmstart.is_some(), "should load saved weights");

        let ws = warmstart.unwrap();
        let guess = ws.generate(&track, &config, &car);
        assert!(guess.is_some(), "should generate guess from loaded model");

        let x = guess.unwrap();
        assert_eq!(x.len(), 7 * 50 - 1);
        assert!(x.iter().all(|v| v.is_finite()));

        // Verify predict_profiles works too
        let profiles = ws.predict_profiles(&track, 50, 5.0);
        assert!(profiles.is_some());
        let p = profiles.unwrap();
        assert_eq!(p.speeds.len(), 50);
        assert_eq!(p.offsets.len(), 50);
    }

    #[test]
    fn test_ml_initial_guess_with_brake_bias() {
        let (net, _var_map) = make_net();
        let track = make_track();
        let n_nodes = 50;
        let config = CollocationConfig {
            n_nodes,
            closed: true,
            optimize_brake_bias: true,
            ..CollocationConfig::default()
        };
        let car = CarParams::default();
        let guess = ml_initial_guess(&net, &track, &config, &car, &test_norm());
        assert!(guess.is_some());
        let x = guess.unwrap();
        let expected_len = 8 * n_nodes - 1;
        assert_eq!(x.len(), expected_len);
        assert!(x.iter().all(|v| v.is_finite()));
    }
}
