//! Build-time capture of the git revision for provenance metadata.
//!
//! Emits `APEX_GIT_SHA` as a compile-time env var (read via `env!` in the
//! crate): `git rev-parse --short HEAD`, with a `-dirty` suffix when the
//! working tree has uncommitted changes. Falls back to `"unknown"` when git is
//! unavailable or this is not a git checkout (e.g. a crates.io / vendored
//! build). This runs on the HOST regardless of the compile target, so it is
//! inert for the wasm32 build (and apex-telemetry is not in the web-viewer
//! dependency graph anyway).
//!
//! Limitation: cargo caches build-script output, so the `-dirty` suffix is
//! only re-evaluated when a rerun trigger fires. We rerun on `.git/HEAD` and
//! `.git/index` (commit / staging changes); an unstaged edit that does not
//! touch the index may leave a stale suffix until the next rebuild trigger.
//! For byte-identical provenance in tests, pin the timestamp via
//! `APEX_REPRO_TIMESTAMP` — the git sha is stable within a commit.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn main() {
    let sha = git(&["rev-parse", "--short", "HEAD"]);
    let git_sha = match sha {
        Some(sha) => {
            // `git status --porcelain` prints nothing for a clean tree.
            let dirty = git(&["status", "--porcelain"])
                .map(|s| !s.is_empty())
                .unwrap_or(false);
            if dirty {
                format!("{sha}-dirty")
            } else {
                sha
            }
        }
        None => "unknown".to_string(),
    };

    println!("cargo:rustc-env=APEX_GIT_SHA={git_sha}");
    // Re-run when the checked-out commit or the index changes.
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
}
