#![deny(unsafe_code)]
//! Telemetry export for Apex-14: writing simulation output to disk.
//!
//! Every writer requires a [`RunMetadata`] provenance block (never optional),
//! embedded as CSV comment lines or an SVG `<metadata>` element so no artifact
//! can be produced without recording what produced it.

pub mod channels;
pub mod csv_export;
pub mod motec;
#[cfg(feature = "parquet")]
pub mod parquet_export;
pub mod run_metadata;
pub mod svg_envelope;
pub mod svg_track;

pub use channels::{ChannelId, ChannelSpec, Quantity, Unit, CHANNELS};
pub use csv_export::{export_columns_csv, export_qss_csv};
pub use motec::{export_ld, read_ld, Grid, LdOptions, LdReport, MotecError};
#[cfg(feature = "parquet")]
pub use parquet_export::{
    export_channels_parquet, read_parquet, write_parquet, ParquetColumn, ParquetData, ParquetError,
};
pub use run_metadata::{now_rfc3339, settings_hash_for_mode, RunMetadata};
pub use svg_envelope::{render_envelope_svg, EnvelopeSlicePlot};
pub use svg_track::render_track_svg;
