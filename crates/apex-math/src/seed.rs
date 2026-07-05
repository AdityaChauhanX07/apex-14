//! Shared RNG-seed resolution + logging for the CLI/training binaries.
//!
//! Every command with a stochastic component resolves its RNG seed through
//! [`resolve_seed`] so the value that will be used is always printed — the
//! roadmap requires the default seed to be logged, never silent, and to be
//! distinguishable from a user-supplied one. Lives in `apex-math` (the zero-dep
//! workspace leaf) so `apex-cli`, `train-driver`, and `train-raceline` share one
//! implementation instead of three drifting copies. Uses only `println!` — no
//! clap/log dependency is pulled in.

/// Resolve the RNG seed for a stochastic command and log the choice.
///
/// Returns `user_seed` when the user passed `--seed`, otherwise `default`. The
/// chosen value is always printed, tagged by `context`, and marked as either
/// user-supplied or the default.
pub fn resolve_seed(user_seed: Option<u64>, default: u64, context: &str) -> u64 {
    match user_seed {
        Some(seed) => {
            println!("[{context}] seed = {seed} (from --seed)");
            seed
        }
        None => {
            println!("[{context}] seed = {default} (default)");
            default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_seed_uses_supplied_value() {
        // Some(_) returns the user's seed verbatim, ignoring the default.
        assert_eq!(resolve_seed(Some(1234), 42, "test"), 1234);
    }

    #[test]
    fn resolve_seed_falls_back_to_default() {
        // None returns the default, preserving pre-flag behavior.
        assert_eq!(resolve_seed(None, 42, "test"), 42);
    }
}
