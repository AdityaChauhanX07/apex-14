#![deny(unsafe_code)]
//! Train a PPO agent to drive the 14-DOF car around tracks.

use clap::Parser;

use apex_rl::ppo::PpoConfig;
use apex_rl::reward::RewardConfig;
use apex_rl::train::{train, TrainConfig};
use apex_track::{
    build_track, circle_track, monza_circuit, oval_track, random_spline_track, silverstone_circuit,
    Track,
};

mod seed;
use seed::resolve_seed;

/// CLI arguments for the driver-training binary.
#[derive(Parser, Debug)]
#[command(
    name = "train-driver",
    about = "Train a PPO agent to drive the 14-DOF car"
)]
struct Args {
    /// Track to train on. Built-in: silverstone, monza, oval, circle.
    /// Use "random" to train on randomly generated tracks.
    #[arg(long, default_value = "oval")]
    track: String,

    /// Total environment steps to collect.
    #[arg(long, default_value_t = 500_000)]
    total_steps: usize,

    /// Number of parallel environments.
    #[arg(long, default_value_t = 4)]
    n_envs: usize,

    /// Steps per environment per rollout.
    #[arg(long, default_value_t = 2048)]
    n_steps: usize,

    /// Learning rate.
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,

    /// Reward preset: progress, balanced, racing.
    #[arg(long, default_value = "progress")]
    reward: String,

    /// Output path for trained weights.
    #[arg(long, default_value = "driver_policy.safetensors")]
    output: String,

    /// Base RNG seed for reproducible training (weight init + action sampling).
    /// Defaults to 42 when omitted.
    #[arg(long)]
    seed: Option<u64>,
}

/// Number of random spline tracks to generate for "--track random".
const N_RANDOM_TRACKS: usize = 20;

/// Build the training tracks for the given `--track` argument.
fn build_tracks(track: &str) -> Result<Vec<Track>, Box<dyn std::error::Error>> {
    let lower = track.to_lowercase();
    match lower.as_str() {
        "random" => {
            let mut tracks = Vec::with_capacity(N_RANDOM_TRACKS);
            for seed in 0..N_RANDOM_TRACKS {
                match random_spline_track(8, 200.0, 0.3, 0.15, 12.0, seed as u64, 300) {
                    Ok((pts, closed)) => {
                        tracks.push(build_track(&format!("random_{seed}"), &pts, closed));
                    }
                    Err(e) => log::warn!("seed {seed}: track generation failed: {e}"),
                }
            }
            if tracks.is_empty() {
                return Err("failed to generate any random tracks".into());
            }
            Ok(tracks)
        }
        "silverstone" => {
            let (pts, closed) = silverstone_circuit();
            Ok(vec![build_track("silverstone", &pts, closed)])
        }
        "monza" => {
            let (pts, closed) = monza_circuit();
            Ok(vec![build_track("monza", &pts, closed)])
        }
        "oval" => {
            let (pts, closed) = oval_track(200.0, 50.0, 12.0, 400);
            Ok(vec![build_track("oval", &pts, closed)])
        }
        "circle" => {
            let (pts, closed) = circle_track(100.0, 12.0, 400);
            Ok(vec![build_track("circle", &pts, closed)])
        }
        other => Err(format!(
            "unknown track '{other}' (expected: silverstone, monza, oval, circle, random)"
        )
        .into()),
    }
}

/// Map a reward preset string to a [`RewardConfig`].
fn reward_preset(name: &str) -> Result<RewardConfig, Box<dyn std::error::Error>> {
    match name.to_lowercase().as_str() {
        "progress" => Ok(RewardConfig::progress_only()),
        "balanced" => Ok(RewardConfig::balanced()),
        "racing" => Ok(RewardConfig::racing()),
        other => Err(format!(
            "unknown reward preset '{other}' (expected: progress, balanced, racing)"
        )
        .into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();

    let tracks = build_tracks(&args.track)?;
    let reward = reward_preset(&args.reward)?;
    let seed = resolve_seed(args.seed, 42, "train-driver");

    let config = TrainConfig {
        ppo: PpoConfig {
            n_envs: args.n_envs,
            n_steps: args.n_steps,
            learning_rate: args.lr,
            ..PpoConfig::default()
        },
        reward,
        total_steps: args.total_steps,
        output_path: args.output.clone(),
        seed,
        ..TrainConfig::default()
    };

    println!("== train-driver ==");
    println!("track:        {} ({} loaded)", args.track, tracks.len());
    println!("total_steps:  {}", args.total_steps);
    println!("n_envs:       {}", args.n_envs);
    println!("n_steps:      {}", args.n_steps);
    println!("learning_rate:{}", args.lr);
    println!("reward:       {}", args.reward);
    println!("output:       {}", args.output);

    train(tracks, config)?;

    println!("Training complete, weights saved to {}", args.output);
    Ok(())
}
