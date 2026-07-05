#![deny(unsafe_code)]
//! Telemetry export for Apex-14: writing simulation output to disk.
//!
//! Every writer requires a [`RunMetadata`] provenance block (never optional),
//! embedded as CSV comment lines or an SVG `<metadata>` element so no artifact
//! can be produced without recording what produced it.

pub mod csv_export;
pub mod run_metadata;
pub mod svg_track;

pub use csv_export::{export_columns_csv, export_qss_csv};
pub use run_metadata::{now_rfc3339, settings_hash_for_mode, RunMetadata};
pub use svg_track::render_track_svg;
