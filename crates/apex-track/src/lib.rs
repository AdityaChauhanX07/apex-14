#![deny(unsafe_code)]
//! Track representation and processing for Apex-14: raw point ingestion,
//! arc-length/heading/curvature computation, parametric generators, and
//! arc-length interpolation queries.

pub mod builder;
pub mod circuits;
pub mod generators;
pub mod parser;
pub mod query;
pub mod types;

pub use builder::{build_track, normalize_angle};
pub use circuits::{monza_circuit, silverstone_circuit};
pub use generators::{circle_track, oval_track};
pub use parser::{
    export_track_json, export_tumftm_csv, load_track_json, load_tumftm_csv, parse_track_json,
    parse_tumftm_csv, TrackFileJson, TrackPointJson,
};
pub use types::{Track, TrackPoint, TrackSegment};
