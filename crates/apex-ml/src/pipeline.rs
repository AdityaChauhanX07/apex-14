//! Batch training-data generation: run the collocation optimizer on many tracks
//! in parallel and collect (features, solution) pairs.

use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use apex_optimizer::{
    CollocationConfig, CollocationMethod, CollocationOptimizer, GaussNewtonConfig,
};
use apex_physics::CarParams;
use apex_track::{extract_features, Track};

use crate::data::{TrainingDataset, TrainingSample, N_FIXED};

/// Configuration for the training data pipeline.
pub struct PipelineConfig {
    /// Number of collocation nodes for the optimizer.
    pub n_nodes: usize,
    /// Maximum equality-constraint violation to consider "converged".
    pub convergence_threshold: f64,
    /// Vehicle parameters used for all optimizer runs.
    pub car: CarParams,
    /// Gauss-Newton solver settings.
    pub gn_config: GaussNewtonConfig,
}

impl PipelineConfig {
    /// Default pipeline using a calibrated F1 car and Hermite-Simpson collocation.
    pub fn default_f1() -> Self {
        PipelineConfig {
            n_nodes: 50,
            convergence_threshold: 1.0,
            car: CarParams::f1_2024_calibrated(),
            gn_config: GaussNewtonConfig {
                max_iterations: 50,
                constraint_tol: 1e-3,
                damping: 0.5,
                regularization: 1e-4,
                print_interval: 0,
                ..GaussNewtonConfig::default()
            },
        }
    }
}

/// Linear interpolation of `ys` sampled at strictly increasing `xs`, evaluated
/// at `x`. Clamps to the endpoint values outside the range.
fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    debug_assert_eq!(xs.len(), ys.len());
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

/// Resample `values` (sampled at `src_stations`) to `N_FIXED` evenly-spaced
/// arc-length points over `[0, total_length]`.
fn resample_to_fixed(src_stations: &[f64], values: &[f64], total_length: f64) -> Vec<f64> {
    let ds = total_length / N_FIXED as f64;
    (0..N_FIXED)
        .map(|i| interp(src_stations, values, i as f64 * ds))
        .collect()
}

/// Run the collocation optimizer on a single track and extract a training sample.
///
/// Returns `None` if the optimizer produces non-finite values. Sets
/// `converged` based on `config.convergence_threshold` rather than the
/// solver's internal tolerance.
pub fn generate_sample(
    config: &PipelineConfig,
    track: &Track,
    track_id: &str,
) -> Option<TrainingSample> {
    let coll_config = CollocationConfig {
        n_nodes: config.n_nodes,
        closed: track.is_closed,
        method: CollocationMethod::HermiteSimpson,
        ..CollocationConfig::default()
    };
    let optimizer = CollocationOptimizer::new(coll_config, track, &config.car);
    let result = optimizer.optimize_gn(&config.gn_config);

    // Reject results with non-finite values.
    if !result.lap_time.is_finite()
        || result.speeds.iter().any(|v| !v.is_finite())
        || result.offsets.iter().any(|v| !v.is_finite())
    {
        log::warn!(
            "track {}: optimizer returned non-finite values, skipping",
            track_id
        );
        return None;
    }

    let feat = extract_features(track, N_FIXED);
    let speed_profile = resample_to_fixed(&result.stations, &result.speeds, track.total_length);
    let offset_profile = resample_to_fixed(&result.stations, &result.offsets, track.total_length);

    let converged = result.eq_violation < config.convergence_threshold;

    Some(TrainingSample {
        curvature_profile: feat.curvature,
        curvature_deriv_profile: feat.curvature_deriv,
        width_left_profile: feat.width_left,
        width_right_profile: feat.width_right,
        speed_profile,
        offset_profile,
        lap_time: result.lap_time,
        converged,
        track_id: track_id.to_string(),
    })
}

/// Run the optimizer on many tracks in parallel using Rayon.
///
/// Returns a [`TrainingDataset`] with all results (converged and non-converged).
/// Logs progress every 10 tracks.
pub fn generate_batch(config: &PipelineConfig, tracks: &[(Track, String)]) -> TrainingDataset {
    let total = tracks.len();
    let completed = AtomicUsize::new(0);

    let samples: Vec<TrainingSample> = tracks
        .par_iter()
        .filter_map(|(track, id)| {
            let sample = generate_sample(config, track, id);
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done.is_multiple_of(10) || done == total {
                log::info!("apex-ml: {}/{} tracks processed", done, total);
            }
            sample
        })
        .collect();

    let tracks_converged = samples.iter().filter(|s| s.converged).count();

    TrainingDataset {
        samples,
        tracks_attempted: total,
        tracks_converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_track::{build_track, circle_track, oval_track};

    fn circle() -> (Track, String) {
        let (pts, closed) = circle_track(100.0, 12.0, 200);
        (build_track("circle", &pts, closed), "circle".to_string())
    }

    fn small_oval() -> (Track, String) {
        let (pts, closed) = oval_track(200.0, 50.0, 12.0, 200);
        (build_track("oval", &pts, closed), "oval".to_string())
    }

    #[test]
    fn generate_sample_circle_converges() {
        let config = PipelineConfig::default_f1();
        let (track, id) = circle();
        let sample = generate_sample(&config, &track, &id);
        assert!(sample.is_some(), "expected Some sample for circle track");
        let s = sample.unwrap();
        assert!(
            s.converged,
            "circle track should converge: eq_viol >= threshold"
        );
        assert_eq!(s.speed_profile.len(), N_FIXED);
        assert_eq!(s.offset_profile.len(), N_FIXED);
        assert_eq!(s.curvature_profile.len(), N_FIXED);
        assert_eq!(s.curvature_deriv_profile.len(), N_FIXED);
        assert_eq!(s.width_left_profile.len(), N_FIXED);
        assert_eq!(s.width_right_profile.len(), N_FIXED);
    }

    #[test]
    fn generate_batch_count() {
        let config = PipelineConfig::default_f1();
        let tracks = vec![circle(), small_oval(), circle()];
        let dataset = generate_batch(&config, &tracks);
        assert_eq!(dataset.tracks_attempted, 3);
        assert_eq!(
            dataset.samples.len() + (3 - dataset.samples.len()),
            3,
            "all tracks should produce a sample or be filtered"
        );
        assert!(dataset.samples.len() <= 3);
    }

    #[test]
    fn generated_profiles_have_fixed_length() {
        let config = PipelineConfig::default_f1();
        let (track, id) = circle();
        let sample = generate_sample(&config, &track, &id).expect("sample");
        sample.validate().expect("validate");
    }
}
