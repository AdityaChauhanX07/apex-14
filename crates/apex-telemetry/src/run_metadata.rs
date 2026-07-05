//! Provenance metadata embedded in every CSV and SVG telemetry artifact.
//!
//! [`RunMetadata`] is the forced provenance block: every writer in this crate
//! requires one (never an `Option`), so no artifact can be produced without
//! recording what config, car, track, solver settings, code revision, seed,
//! and time produced it. The same ordered key/value list is rendered into CSV
//! comment lines and an SVG `<metadata>` element, so the two formats never
//! drift.
//!
//! # Hash representation
//!
//! All four hashes are rendered as **full 64-character lowercase hex**
//! ([`Hash::to_hex`]), used consistently for every hash field. The full hash
//! (not a 16-char prefix) is embedded deliberately: provenance in a standalone
//! artifact should be complete and independently verifiable — you cannot expand
//! a truncated prefix back to the full digest from the file alone. Short
//! prefixes remain available via [`Hash::short`] for human-facing logs
//! elsewhere; the embedded block uses one representation (full hex) throughout.

use std::fmt::Write as _;

use apex_math::{Hash, HashWriter, HASH_VERSION};

/// Provenance metadata for a single telemetry-producing run.
#[derive(Debug, Clone)]
pub struct RunMetadata {
    /// Composite hash over the car, track, and settings hashes (see
    /// [`RunMetadata::new`] for the exact composition rule).
    pub config_hash: Hash,
    /// Content hash of the resolved car parameters.
    pub car_hash: Hash,
    /// Content hash of the processed track geometry.
    pub track_hash: Hash,
    /// Content hash of the solver/mode settings that determined the result.
    pub settings_hash: Hash,
    /// Git revision captured at build time (`<short-sha>` or `<short-sha>-dirty`,
    /// or `"unknown"`); see `build.rs`.
    pub git_sha: String,
    /// Crate version from `CARGO_PKG_VERSION`.
    pub apex_version: String,
    /// Run seed, or `None` for deterministic (RNG-free) modes.
    pub seed: Option<u64>,
    /// RFC3339 UTC timestamp, or the verbatim `APEX_REPRO_TIMESTAMP` override.
    pub timestamp: String,
}

impl RunMetadata {
    /// Build metadata from the three component hashes and an optional seed.
    ///
    /// # `config_hash` composition rule
    ///
    /// `config_hash` is the BLAKE3 content hash over the canonical encoding
    ///
    /// ```text
    /// HASH_VERSION ‖ "run.config"
    ///   ‖ len(car_hash)      ‖ car_hash.as_bytes()
    ///   ‖ len(track_hash)    ‖ track_hash.as_bytes()
    ///   ‖ len(settings_hash) ‖ settings_hash.as_bytes()
    /// ```
    ///
    /// i.e. the three component hashes' raw 32-byte digests, each length-
    /// prefixed (via [`HashWriter::bytes`]) and concatenated in the fixed order
    /// **car, track, settings**, under the domain tag `"run.config"`. Length
    /// prefixing makes the concatenation unambiguous; the version + domain
    /// prefix matches every other content hash in the workspace.
    ///
    /// `git_sha` and `apex_version` are captured from build-time/compile-time
    /// env; `timestamp` from [`now_rfc3339`].
    pub fn new(car_hash: Hash, track_hash: Hash, settings_hash: Hash, seed: Option<u64>) -> Self {
        let config_hash = compose_config_hash(&car_hash, &track_hash, &settings_hash);
        RunMetadata {
            config_hash,
            car_hash,
            track_hash,
            settings_hash,
            git_sha: env!("APEX_GIT_SHA").to_string(),
            apex_version: env!("CARGO_PKG_VERSION").to_string(),
            seed,
            timestamp: now_rfc3339(),
        }
    }

    /// Ordered `(key, value)` pairs — the single source of truth for both the
    /// CSV comment block and the SVG `<metadata>` element.
    fn fields(&self) -> [(&'static str, String); 8] {
        [
            ("config_hash", self.config_hash.to_hex()),
            ("car_hash", self.car_hash.to_hex()),
            ("track_hash", self.track_hash.to_hex()),
            ("settings_hash", self.settings_hash.to_hex()),
            ("git_sha", self.git_sha.clone()),
            ("apex_version", self.apex_version.clone()),
            (
                "seed",
                match self.seed {
                    Some(s) => s.to_string(),
                    None => "none".to_string(),
                },
            ),
            ("timestamp", self.timestamp.clone()),
        ]
    }

    /// CSV provenance header: one `# key: value` line per field, then one blank
    /// comment line (`#`), ready to be written immediately before the column
    /// header row. Every line begins with `#` so a comment-aware reader skips
    /// the whole block.
    pub fn csv_comment_block(&self) -> String {
        let mut s = String::new();
        for (k, v) in self.fields() {
            let _ = writeln!(s, "# {k}: {v}");
        }
        s.push_str("#\n");
        s
    }

    /// SVG `<metadata>` element carrying the same key/values as a namespaced,
    /// well-formed XML fragment (two-space indented, newline-terminated), ready
    /// to insert just after the `<svg …>` open tag.
    pub fn svg_metadata_element(&self) -> String {
        let mut s = String::new();
        s.push_str("  <metadata>\n");
        s.push_str("    <apex:run xmlns:apex=\"https://apex-14.dev/provenance\">\n");
        for (k, v) in self.fields() {
            let _ = writeln!(s, "      <apex:{k}>{}</apex:{k}>", xml_escape(&v));
        }
        s.push_str("    </apex:run>\n");
        s.push_str("  </metadata>\n");
        s
    }
}

/// Settings hash for a mode that has no tunable solver config (e.g. QSS,
/// sensitivity): hashes just the mode label, so the provenance block still
/// carries a stable, domain-separated `settings_hash`.
pub fn settings_hash_for_mode(mode: &str) -> Hash {
    let mut w = HashWriter::new();
    w.str(HASH_VERSION);
    w.str("run.settings.mode");
    w.str(mode);
    w.finish()
}

/// `config_hash` composition — see [`RunMetadata::new`] for the documented rule.
fn compose_config_hash(car: &Hash, track: &Hash, settings: &Hash) -> Hash {
    let mut w = HashWriter::new();
    w.str(HASH_VERSION);
    w.str("run.config");
    w.bytes(car.as_bytes());
    w.bytes(track.as_bytes());
    w.bytes(settings.as_bytes());
    w.finish()
}

/// Current time as an RFC3339 UTC string (`YYYY-MM-DDTHH:MM:SSZ`).
///
/// Reproducibility escape hatch: if `APEX_REPRO_TIMESTAMP` is set, its value is
/// returned verbatim instead of the wall clock, so two runs with the same seed
/// produce byte-identical artifacts for diffing and golden testing.
pub fn now_rfc3339() -> String {
    if let Ok(pinned) = std::env::var("APEX_REPRO_TIMESTAMP") {
        return pinned;
    }
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format_rfc3339_utc(secs)
}

/// Format Unix epoch seconds as RFC3339 UTC, no external date dependency (the
/// workspace intentionally carries no `chrono`/`time` crate).
fn format_rfc3339_utc(unix_secs: u64) -> String {
    let sec = unix_secs % 60;
    let min = (unix_secs / 60) % 60;
    let hour = (unix_secs / 3600) % 24;
    let days = (unix_secs / 86_400) as i64;
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Days since 1970-01-01 → `(year, month, day)`, Gregorian, via Howard
/// Hinnant's `civil_from_days` (public-domain algorithm).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Minimal XML text escaping for `<metadata>` values.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

/// Serializes the two tests that mutate the process-global
/// `APEX_REPRO_TIMESTAMP` env var, so they cannot interleave a set/remove with
/// each other under parallel test execution. Other tests only *read* the var
/// (via `RunMetadata::new`) and are unaffected.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_meta(seed: Option<u64>) -> RunMetadata {
        RunMetadata::new(
            settings_hash_for_mode("car"),
            settings_hash_for_mode("track"),
            settings_hash_for_mode("settings"),
            seed,
        )
    }

    #[test]
    fn config_hash_is_deterministic_and_order_sensitive() {
        let car = settings_hash_for_mode("a");
        let track = settings_hash_for_mode("b");
        let settings = settings_hash_for_mode("c");
        assert_eq!(
            compose_config_hash(&car, &track, &settings),
            compose_config_hash(&car, &track, &settings)
        );
        // Order matters: swapping car/track changes the composite.
        assert_ne!(
            compose_config_hash(&car, &track, &settings),
            compose_config_hash(&track, &car, &settings)
        );
    }

    #[test]
    fn csv_block_has_all_keys_and_blank_line() {
        let block = sample_meta(Some(7)).csv_comment_block();
        for key in [
            "config_hash",
            "car_hash",
            "track_hash",
            "settings_hash",
            "git_sha",
            "apex_version",
            "seed",
            "timestamp",
        ] {
            assert!(block.contains(&format!("# {key}: ")), "missing {key}");
        }
        assert!(block.lines().all(|l| l.starts_with('#')));
        assert!(block.ends_with("#\n"));
        assert!(block.contains("# seed: 7"));
    }

    #[test]
    fn seed_none_renders_as_none() {
        let block = sample_meta(None).csv_comment_block();
        assert!(block.contains("# seed: none"));
    }

    #[test]
    fn svg_metadata_is_wellformed_fragment() {
        let el = sample_meta(Some(1)).svg_metadata_element();
        assert!(el.contains("<metadata>"));
        assert!(el.contains("</metadata>"));
        assert!(el.contains("xmlns:apex="));
        assert!(el.contains("<apex:config_hash>"));
        assert!(el.contains("</apex:timestamp>"));
    }

    #[test]
    fn repro_timestamp_override_is_verbatim() {
        // Uses a process-global env var; the lock serializes against the other
        // env-mutating test.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("APEX_REPRO_TIMESTAMP", "2020-01-02T03:04:05Z");
        let ts = now_rfc3339();
        std::env::remove_var("APEX_REPRO_TIMESTAMP");
        assert_eq!(ts, "2020-01-02T03:04:05Z");
    }

    #[test]
    fn rfc3339_formats_known_epochs() {
        assert_eq!(format_rfc3339_utc(0), "1970-01-01T00:00:00Z");
        // 2021-01-01T00:00:00Z = 1609459200
        assert_eq!(format_rfc3339_utc(1_609_459_200), "2021-01-01T00:00:00Z");
    }
}
