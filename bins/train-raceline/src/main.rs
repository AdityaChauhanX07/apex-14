#![deny(unsafe_code)]
//! Training binary for the ML raceline predictor.
//!
//! Generates a synthetic track dataset (or loads an existing one), trains
//! the 1D circular CNN, and saves the weights to a safetensors file.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;

use apex_ml::{
    data::{NormConstants, TrainingDataset},
    io::{load_dataset, meta_path, save_dataset, save_norm_constants},
    net::{save_weights, RacelineNet},
    pipeline::{generate_batch, PipelineConfig},
    train::{train, TrainConfig},
};
use apex_track::{build_track, random_spline_track};

/// CLI arguments for the training binary.
#[derive(Parser, Debug)]
#[command(author, version, about = "Train the ML raceline predictor")]
struct Args {
    /// Number of random tracks to generate for the training set.
    #[arg(long, default_value_t = 50)]
    n_tracks: usize,

    /// Number of training epochs.
    #[arg(long, default_value_t = 200)]
    epochs: usize,

    /// Initial learning rate for AdamW.
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,

    /// Output path for the trained model weights (safetensors format).
    #[arg(long, default_value = "raceline_model.safetensors")]
    output: PathBuf,

    /// Path to a pre-generated dataset JSON file. If provided, track
    /// generation is skipped.
    #[arg(long)]
    dataset: Option<PathBuf>,
}

fn generate_dataset(n_tracks: usize) -> Result<TrainingDataset, Box<dyn std::error::Error>> {
    log::info!("generating {} random tracks...", n_tracks);

    let mut tracks = Vec::with_capacity(n_tracks);
    for seed in 0..n_tracks {
        match random_spline_track(8, 200.0, 0.3, 0.15, 12.0, seed as u64, 300) {
            Ok((pts, closed)) => {
                let track_id = format!("spline_{seed}");
                let track = build_track(&track_id, &pts, closed);
                tracks.push((track, track_id));
            }
            Err(e) => {
                log::warn!("seed {seed}: track generation failed: {e}");
            }
        }
    }

    log::info!("running optimizer on {} tracks...", tracks.len());
    let pipeline_config = PipelineConfig::default_f1();
    let dataset = generate_batch(&pipeline_config, &tracks);

    log::info!(
        "dataset: {}/{} tracks converged",
        dataset.tracks_converged,
        dataset.tracks_attempted
    );

    save_dataset(&dataset, Path::new("training_data.json"))?;
    log::info!("dataset saved to training_data.json");

    Ok(dataset)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();

    let dataset = match &args.dataset {
        Some(path) => {
            log::info!("loading dataset from {}...", path.display());
            load_dataset(path)?
        }
        None => generate_dataset(args.n_tracks)?,
    };

    log::info!(
        "dataset: {} total samples, {} converged",
        dataset.samples.len(),
        dataset.tracks_converged
    );

    println!(
        "normalization: speed_norm={:.1} m/s, width_norm={:.1} m",
        dataset.global_speed_norm, dataset.global_width_norm
    );

    let device = Device::Cpu;
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let net = RacelineNet::new(vb)?;

    println!("network parameters: {}", net.param_count());

    let train_config = TrainConfig {
        epochs: args.epochs,
        learning_rate: args.lr,
        ..TrainConfig::default()
    };

    println!("training for {} epochs...", train_config.epochs);
    let result = train(&dataset, &train_config, &var_map, &net)?;

    println!(
        "training complete: final_train={:.6}  final_val={:.6}  best_val={:.6} (epoch {})",
        result.final_train_loss, result.final_val_loss, result.best_val_loss, result.best_epoch
    );
    println!(
        "train samples: {}  val samples: {}",
        result.n_train, result.n_val
    );

    save_weights(&var_map, &args.output)?;
    println!("weights saved to {}", args.output.display());

    let norm = NormConstants {
        speed_norm: dataset.global_speed_norm,
        width_norm: dataset.global_width_norm,
    };
    let meta = meta_path(&args.output);
    save_norm_constants(&norm, &meta)?;
    println!("normalization constants saved to {}", meta.display());

    Ok(())
}
