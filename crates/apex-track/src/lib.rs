#![deny(unsafe_code)]
//! Track representation and processing for Apex-14: raw point ingestion,
//! arc-length/heading/curvature computation, parametric generators, and
//! arc-length interpolation queries.

pub mod builder;
pub mod circuits;
pub mod constraints;
pub mod generators;
pub mod layout;
pub mod parser;
pub mod query;
pub mod track_gen;
pub mod types;

pub use builder::{build_track, normalize_angle};
pub use circuits::{monza_circuit, silverstone_circuit};
pub use constraints::{check_constraints, ConstraintViolation, TrackConstraints};
pub use generators::{circle_track, oval_track};
pub use layout::{
    is_valid_layout, point_in_polygon, track_within_boundary, ControlPoint, TrackLayout,
};
pub use parser::{
    export_track_json, export_tumftm_csv, load_track_json, load_tumftm_csv, parse_track_json,
    parse_tumftm_csv, TrackFileJson, TrackPointJson,
};
pub use track_gen::{extract_features, generate_track_batch, random_spline_track, TrackFeatures};
pub use types::{processed_track_hash, raw_track_hash, Track, TrackPoint, TrackSegment};
