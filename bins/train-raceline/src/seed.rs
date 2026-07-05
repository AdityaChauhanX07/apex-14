//! Seed resolution and logging for the training binary.
//!
//! A small copy of the CLI's `resolve_seed`: `train-raceline` is a separate
//! binary and cannot reach `apex-cli`'s private module, and this pure
//! CLI-logging concern doesn't belong in a domain library. Duplicating ~10
//! lines is cleaner than introducing a crate solely to share them; the wording
//! is kept identical to the CLI so all commands log seeds the same way.

/// Resolve an RNG seed and log the choice.
///
/// Returns `user_seed` when the user passed the flag, otherwise `default`.
/// Either way the resolved value is printed (with `context` as a prefix) so a
/// run is never silently seeded and a supplied seed is clearly distinguished
/// from the fallback default.
pub fn resolve_seed(user_seed: Option<u64>, default: u64, context: &str) -> u64 {
    match user_seed {
        Some(seed) => {
            println!("[{context}] seed = {seed} (from flag)");
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
        assert_eq!(resolve_seed(Some(1234), 42, "test"), 1234);
    }

    #[test]
    fn resolve_seed_falls_back_to_default() {
        assert_eq!(resolve_seed(None, 42, "test"), 42);
    }
}
