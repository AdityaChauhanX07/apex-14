//! The crate error type.

use std::fmt;

/// Anything that can go wrong importing, resampling, or writing telemetry.
#[derive(Debug)]
pub enum CorrelateError {
    /// Filesystem I/O failure.
    Io(std::io::Error),
    /// CSV parsing failure (malformed record structure).
    Csv(csv::Error),
    /// Mapping-config TOML failed to parse.
    Toml(toml::de::Error),
    /// A mapping entry names a channel that is not in the registry.
    UnknownChannel(String),
    /// A source column matched neither the mapping nor a registry name, under
    /// the `error` unknown-columns policy.
    UnknownSourceColumn(String),
    /// A mapping entry declared a source unit that cannot be converted to the
    /// channel's registry unit.
    UnknownSourceUnit {
        /// Source column name.
        column: String,
        /// The unrecognized unit string.
        unit: String,
    },
    /// Two source columns resolve to the same registry channel.
    DuplicateTarget(&'static str),
    /// No `# grid:` declaration and no `s`/`t` axis channel to infer one from,
    /// or the declared grid's axis channel is absent from the data.
    MissingGrid,
    /// A `# grid:` line (or mapping `grid`) named something other than `s`/`t`.
    BadGrid(String),
    /// One or more `required` channels were absent after mapping.
    MissingRequired(Vec<String>),
    /// A data field could not be parsed as a finite-or-`NaN` `f64`.
    MalformedValue {
        /// The registry channel the column mapped to.
        channel: &'static str,
        /// Zero-based data-row index.
        row: usize,
        /// The offending raw field text.
        value: String,
    },
    /// The resampling axis (`s` or `t`) was not strictly increasing / finite.
    AxisNotMonotonic(&'static str),
    /// The channel required for the requested resampling grid is missing.
    MissingAxis(&'static str),
    /// Resampling step must be finite and strictly positive.
    BadStep(f64),
}

impl fmt::Display for CorrelateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CorrelateError::Io(e) => write!(f, "I/O error: {e}"),
            CorrelateError::Csv(e) => write!(f, "CSV error: {e}"),
            CorrelateError::Toml(e) => write!(f, "mapping TOML error: {e}"),
            CorrelateError::UnknownChannel(c) => {
                write!(f, "mapping references unknown registry channel `{c}`")
            }
            CorrelateError::UnknownSourceColumn(c) => {
                write!(f, "unknown source column `{c}` (unknown_columns = error)")
            }
            CorrelateError::UnknownSourceUnit { column, unit } => {
                write!(f, "column `{column}`: unrecognized source unit `{unit}`")
            }
            CorrelateError::DuplicateTarget(c) => {
                write!(f, "two source columns both map to channel `{c}`")
            }
            CorrelateError::MissingGrid => write!(
                f,
                "no `# grid:` declaration and no `s`/`t` axis channel to infer one"
            ),
            CorrelateError::BadGrid(g) => write!(f, "invalid grid `{g}` (expected `s` or `t`)"),
            CorrelateError::MissingRequired(names) => {
                write!(f, "missing required channels: {}", names.join(", "))
            }
            CorrelateError::MalformedValue {
                channel,
                row,
                value,
            } => write!(
                f,
                "channel `{channel}`, row {row}: cannot parse `{value}` as a number"
            ),
            CorrelateError::AxisNotMonotonic(a) => {
                write!(
                    f,
                    "resampling axis `{a}` is not strictly increasing / finite"
                )
            }
            CorrelateError::MissingAxis(a) => {
                write!(f, "cannot resample: axis channel `{a}` is absent")
            }
            CorrelateError::BadStep(s) => {
                write!(f, "resampling step must be finite and > 0, got {s}")
            }
        }
    }
}

impl std::error::Error for CorrelateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CorrelateError::Io(e) => Some(e),
            CorrelateError::Csv(e) => Some(e),
            CorrelateError::Toml(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CorrelateError {
    fn from(e: std::io::Error) -> Self {
        CorrelateError::Io(e)
    }
}

impl From<csv::Error> for CorrelateError {
    fn from(e: csv::Error) -> Self {
        CorrelateError::Csv(e)
    }
}

impl From<toml::de::Error> for CorrelateError {
    fn from(e: toml::de::Error) -> Self {
        CorrelateError::Toml(e)
    }
}
