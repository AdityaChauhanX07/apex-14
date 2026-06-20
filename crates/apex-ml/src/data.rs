//! Training sample and dataset types for the ML raceline pipeline.

use serde::{Deserialize, Serialize};

/// Number of fixed sample points for ML features and targets.
pub const N_FIXED: usize = 100;

/// A single training sample pairing track features with optimizer output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Track curvature profile resampled to N_FIXED points (normalized, 1/m).
    pub curvature_profile: Vec<f64>,
    /// Curvature derivative profile at N_FIXED points (normalized, 1/m^2).
    pub curvature_deriv_profile: Vec<f64>,
    /// Track width (left boundary distance) at N_FIXED points (normalized, m).
    pub width_left_profile: Vec<f64>,
    /// Track width (right boundary distance) at N_FIXED points (normalized, m).
    pub width_right_profile: Vec<f64>,
    /// Optimized speed profile at N_FIXED points, normalized by [`Self::speed_norm`].
    pub speed_profile: Vec<f64>,
    /// Optimized lateral offset profile at N_FIXED points, normalized by
    /// [`Self::width_norm`].
    pub offset_profile: Vec<f64>,
    /// Speed normalization constant (m/s): multiply [`Self::speed_profile`] by
    /// this to recover physical speeds.
    pub speed_norm: f64,
    /// Width normalization constant (m): multiply [`Self::offset_profile`] by
    /// this to recover physical offsets.
    pub width_norm: f64,
    /// Achieved lap time (s).
    pub lap_time: f64,
    /// Whether the optimizer converged (eq_viol < threshold).
    pub converged: bool,
    /// Track identifier or seed used to generate it.
    pub track_id: String,
}

impl TrainingSample {
    /// Check that all profile vectors have exactly [`N_FIXED`] elements.
    pub fn validate(&self) -> Result<(), String> {
        let check = |name: &str, len: usize| {
            if len == N_FIXED {
                Ok(())
            } else {
                Err(format!(
                    "{} has {} elements, expected {}",
                    name, len, N_FIXED
                ))
            }
        };
        check("curvature_profile", self.curvature_profile.len())?;
        check(
            "curvature_deriv_profile",
            self.curvature_deriv_profile.len(),
        )?;
        check("width_left_profile", self.width_left_profile.len())?;
        check("width_right_profile", self.width_right_profile.len())?;
        check("speed_profile", self.speed_profile.len())?;
        check("offset_profile", self.offset_profile.len())?;
        Ok(())
    }
}

/// A dataset of training samples, serializable to/from JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataset {
    /// All collected samples (including non-converged, for analysis).
    pub samples: Vec<TrainingSample>,
    /// Number of tracks attempted.
    pub tracks_attempted: usize,
    /// Number of tracks where the optimizer converged.
    pub tracks_converged: usize,
    /// Global speed normalization constant (m/s): the max per-sample speed norm.
    /// All sample speed profiles are normalized by this value.
    pub global_speed_norm: f64,
    /// Global width normalization constant (m): the mean per-sample width norm.
    /// All sample offset profiles are normalized by this value.
    pub global_width_norm: f64,
}

/// Normalization constants for the ML target profiles.
///
/// Saved alongside the model weights so that inference can denormalize the
/// network output back into physical units (m/s for speed, m for offset).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormConstants {
    /// Speed normalization constant (m/s).
    pub speed_norm: f64,
    /// Width normalization constant (m).
    pub width_norm: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(n: usize) -> TrainingSample {
        TrainingSample {
            curvature_profile: vec![0.0; n],
            curvature_deriv_profile: vec![0.0; n],
            width_left_profile: vec![0.5; n],
            width_right_profile: vec![0.5; n],
            speed_profile: vec![30.0; n],
            offset_profile: vec![0.0; n],
            speed_norm: 1.0,
            width_norm: 1.0,
            lap_time: 60.0,
            converged: true,
            track_id: "test".to_string(),
        }
    }

    #[test]
    fn validate_passes_for_correct_size() {
        let sample = make_sample(N_FIXED);
        assert!(sample.validate().is_ok());
    }

    #[test]
    fn validate_fails_for_wrong_size() {
        let sample = make_sample(N_FIXED - 1);
        assert!(sample.validate().is_err());

        let sample_big = make_sample(N_FIXED + 1);
        assert!(sample_big.validate().is_err());
    }

    #[test]
    fn dataset_serde_roundtrip() {
        let dataset = TrainingDataset {
            samples: vec![make_sample(N_FIXED)],
            tracks_attempted: 2,
            tracks_converged: 1,
            global_speed_norm: 1.0,
            global_width_norm: 1.0,
        };
        let json = serde_json::to_string(&dataset).expect("serialize");
        let back: TrainingDataset = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.tracks_attempted, dataset.tracks_attempted);
        assert_eq!(back.tracks_converged, dataset.tracks_converged);
        assert_eq!(back.samples.len(), 1);
        assert_eq!(back.samples[0].speed_profile.len(), N_FIXED);
        assert_eq!(back.samples[0].lap_time, 60.0);
        assert!(back.samples[0].converged);
    }
}
