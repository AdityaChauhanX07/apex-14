//! Channel-mapping configuration (TOML).
//!
//! A mapping tells the importer how to turn an arbitrary source CSV into
//! registry channels: which source column feeds which [`ChannelId`], and what
//! unit the source column is in (for conversion to the registry unit).
//!
//! # Schema
//!
//! ```toml
//! # How to treat source columns that match neither a mapping entry nor a
//! # registry channel name. "ignore" (default) drops them; "error" rejects.
//! unknown_columns = "ignore"
//!
//! # Registry channel names that MUST be present after mapping, else import
//! # errors listing what's absent. Optional; default empty.
//! required = ["s", "speed"]
//!
//! # When true (default), a source column whose name is already a registry
//! # channel name maps to that channel as-is, in the registry unit (identity).
//! # Set false to require an explicit [columns.*] entry for every column.
//! auto_identity = true
//!
//! # Optional override of the file's `# grid:` line ("s" or "t").
//! grid = "s"
//!
//! # One table per source column that needs explicit mapping. The table key is
//! # the exact source-CSV column name (quote it if it contains spaces:
//! # [columns."Position X"]).
//! [columns.Speed]
//! channel = "speed"      # target registry channel name
//! unit = "km/h"          # source unit; converted to the registry unit
//!
//! [columns.Throttle]
//! channel = "throttle"
//! unit = "percent"       # → 0–1 fraction
//! ```
//!
//! An `import_telemetry` of a file already in the standard Apex format needs no
//! config at all: [`Mapping::identity`] (all defaults) maps every
//! registry-named column to itself.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::CorrelateError;

/// Policy for source columns that resolve to no channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum UnknownColumns {
    /// Silently drop the column (default).
    #[default]
    Ignore,
    /// Fail the import, naming the column.
    Error,
}

/// Explicit mapping of one source column to a registry channel.
#[derive(Debug, Clone, Deserialize)]
pub struct ColumnMap {
    /// Target registry channel name (an `apex_telemetry::ChannelId::name()`).
    pub channel: String,
    /// Source unit string (see [`crate::conversion_factor`]); empty = identity.
    #[serde(default)]
    pub unit: String,
}

/// A full channel-mapping configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mapping {
    /// Unknown-column policy.
    #[serde(default)]
    pub unknown_columns: UnknownColumns,
    /// Registry channel names that must be present after mapping.
    #[serde(default)]
    pub required: Vec<String>,
    /// Whether registry-named columns map to themselves without an entry.
    #[serde(default = "default_true")]
    pub auto_identity: bool,
    /// Optional grid override (`"s"` / `"t"`).
    #[serde(default)]
    pub grid: Option<String>,
    /// Explicit per-column mappings, keyed by source column name.
    #[serde(default)]
    pub columns: BTreeMap<String, ColumnMap>,
}

fn default_true() -> bool {
    true
}

impl Default for Mapping {
    fn default() -> Self {
        Mapping {
            unknown_columns: UnknownColumns::Ignore,
            required: Vec::new(),
            auto_identity: true,
            grid: None,
            columns: BTreeMap::new(),
        }
    }
}

impl Mapping {
    /// The identity mapping: no explicit columns, `auto_identity = true`. Import
    /// a standard-format Apex CSV with this — every registry-named column maps
    /// to itself in the registry unit.
    pub fn identity() -> Self {
        Mapping::default()
    }

    /// Parse a mapping from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self, CorrelateError> {
        toml::from_str(s).map_err(CorrelateError::from)
    }

    /// Load and parse a mapping from a TOML file.
    pub fn from_toml_path(path: impl AsRef<Path>) -> Result<Self, CorrelateError> {
        let text = std::fs::read_to_string(path)?;
        Mapping::from_toml_str(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let m = Mapping::identity();
        assert_eq!(m.unknown_columns, UnknownColumns::Ignore);
        assert!(m.auto_identity);
        assert!(m.required.is_empty());
        assert!(m.columns.is_empty());
        assert!(m.grid.is_none());
    }

    #[test]
    fn parses_full_config() {
        let toml = r#"
            unknown_columns = "error"
            required = ["s", "speed"]
            auto_identity = false
            grid = "s"

            [columns.Distance]
            channel = "s"
            unit = "m"

            [columns.Speed]
            channel = "speed"
            unit = "km/h"

            [columns.Throttle]
            channel = "throttle"
            unit = "percent"
        "#;
        let m = Mapping::from_toml_str(toml).unwrap();
        assert_eq!(m.unknown_columns, UnknownColumns::Error);
        assert_eq!(m.required, vec!["s".to_string(), "speed".to_string()]);
        assert!(!m.auto_identity);
        assert_eq!(m.grid.as_deref(), Some("s"));
        assert_eq!(m.columns["Speed"].channel, "speed");
        assert_eq!(m.columns["Speed"].unit, "km/h");
        // unit is optional (defaults to empty = identity).
        assert_eq!(m.columns["Distance"].unit, "m");
    }

    #[test]
    fn unit_defaults_to_empty() {
        let toml = r#"
            [columns.Gear]
            channel = "gear"
        "#;
        let m = Mapping::from_toml_str(toml).unwrap();
        assert_eq!(m.columns["Gear"].unit, "");
    }

    #[test]
    fn rejects_unknown_top_level_field() {
        let toml = "unkown_columns = \"error\"\n"; // typo → deny_unknown_fields
        assert!(Mapping::from_toml_str(toml).is_err());
    }
}
