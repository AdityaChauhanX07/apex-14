//! Track representation and processing for Apex-14: raw point ingestion,
//! arc-length/heading/curvature computation, parametric generators, and
//! arc-length interpolation queries.

pub mod builder;
pub mod generators;
pub mod query;
pub mod types;

pub use builder::{build_track, normalize_angle};
pub use generators::{circle_track, oval_track};
pub use types::{Track, TrackPoint, TrackSegment};
