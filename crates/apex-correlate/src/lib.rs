#![deny(unsafe_code)]
//! Telemetry correlation for Apex-14 (Phase 2, task 2.1a).
//!
//! This crate imports **measured** car telemetry — laps from real data sources
//! such as [FastF1](https://docs.fastf1.dev/) — in the standard *Apex telemetry
//! CSV* interchange format (see `docs/telemetry_format.md`), maps every source
//! column through the [channel registry](apex_telemetry::channels), converts
//! source units to the registry-canonical unit, and resamples onto a uniform
//! `s` (distance) or `t` (time) grid.
//!
//! GPS projection and track alignment are explicitly **out of scope** for this
//! task; the imported `x`/`y` (when present) are in the source's own frame.
//!
//! # Pipeline
//!
//! ```text
//! CSV ──import_telemetry(path, &mapping)──▶ Telemetry ──resample_to_s/t──▶ Telemetry
//!                                               │                              │
//!                                               └──────write_telemetry_csv─────┘
//! ```
//!
//! - [`import_telemetry`] parses the file, resolves columns via a [`Mapping`],
//!   and returns a [`Telemetry`]. Measured `NaN`s (real sensor gaps) are kept.
//! - [`Telemetry::resample_to_s`] / [`Telemetry::resample_to_t`] resample with
//!   linear interpolation; gaps longer than a threshold stay `NaN`.
//! - [`write_telemetry_csv`] writes the standard format back out.

mod align;
pub mod driven;
mod error;
mod importer;
mod mapping;
pub mod metrics;
mod project;
pub mod report;
mod resample;
mod telemetry;
mod units;
mod writer;

pub use align::{fit_alignment, AlignConfig, AlignResult, Similarity};
pub use driven::{driven_sim_trace, DrivenResult, DEFAULT_N_FILTER_WINDOW_M};
pub use error::CorrelateError;
pub use importer::import_telemetry;
pub use mapping::{ColumnMap, Mapping, UnknownColumns};
pub use project::{closest_point, project_to_track, ProjectStats, Projection};
pub use report::{correlate, CorrelationConfig, CorrelationResult, SimTrace};
pub use telemetry::{GridKind, Telemetry};
pub use units::conversion_factor;
pub use writer::write_telemetry_csv;
