//! Serialization helpers for training datasets.

use std::io::{self, BufReader, BufWriter};
use std::path::Path;

use crate::data::TrainingDataset;

/// Save a training dataset to a JSON file.
pub fn save_dataset(dataset: &TrainingDataset, path: &Path) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, dataset).map_err(io::Error::other)
}

/// Load a training dataset from a JSON file.
pub fn load_dataset(path: &Path) -> io::Result<TrainingDataset> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).map_err(io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{TrainingSample, N_FIXED};

    fn make_dataset() -> TrainingDataset {
        TrainingDataset {
            samples: vec![TrainingSample {
                curvature_profile: vec![0.1; N_FIXED],
                curvature_deriv_profile: vec![0.0; N_FIXED],
                width_left_profile: vec![0.5; N_FIXED],
                width_right_profile: vec![0.5; N_FIXED],
                speed_profile: vec![40.0; N_FIXED],
                offset_profile: vec![0.0; N_FIXED],
                lap_time: 90.0,
                converged: true,
                track_id: "io_test".to_string(),
            }],
            tracks_attempted: 3,
            tracks_converged: 1,
        }
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("apex_ml_test_dataset.json");

        let original = make_dataset();
        save_dataset(&original, &path).expect("save failed");

        let loaded = load_dataset(&path).expect("load failed");
        assert_eq!(loaded.tracks_attempted, original.tracks_attempted);
        assert_eq!(loaded.tracks_converged, original.tracks_converged);
        assert_eq!(loaded.samples.len(), original.samples.len());

        let s0 = &loaded.samples[0];
        let o0 = &original.samples[0];
        assert_eq!(s0.track_id, o0.track_id);
        assert_eq!(s0.lap_time, o0.lap_time);
        assert_eq!(s0.converged, o0.converged);
        assert_eq!(s0.speed_profile, o0.speed_profile);
        assert_eq!(s0.offset_profile, o0.offset_profile);
        assert_eq!(s0.curvature_profile, o0.curvature_profile);
    }
}
