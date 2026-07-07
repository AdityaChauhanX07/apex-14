//! Core track data types: raw points, processed segments, and the assembled
//! [`Track`].

/// A raw input point along the track centerline, with boundary distances.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrackPoint {
    /// World X coordinate (meters).
    pub x: f64,
    /// World Y coordinate (meters).
    pub y: f64,
    /// Distance from the centerline to the left boundary (meters).
    pub width_left: f64,
    /// Distance from the centerline to the right boundary (meters).
    pub width_right: f64,
}

/// A processed track segment: a centerline sample enriched with arc length,
/// heading, and curvature.
#[derive(Debug, Clone)]
pub struct TrackSegment {
    /// Cumulative arc length from the start (meters).
    pub s: f64,
    /// World X coordinate.
    pub x: f64,
    /// World Y coordinate.
    pub y: f64,
    /// Tangent angle (radians, measured from the +X axis).
    pub heading: f64,
    /// Signed curvature κ = 1/R (positive = left turn).
    pub curvature: f64,
    /// Distance to the left boundary.
    pub width_left: f64,
    /// Distance to the right boundary.
    pub width_right: f64,
}

/// A fully processed track: a sequence of segments with geometric metadata.
#[derive(Debug, Clone)]
pub struct Track {
    /// Human-readable track name.
    pub name: String,
    /// Processed segments in order of increasing arc length.
    pub segments: Vec<TrackSegment>,
    /// Total arc length of the track (meters).
    pub total_length: f64,
    /// `true` if the track forms a closed loop.
    pub is_closed: bool,
    /// Optional schema v2 sector-marker stations (m, ascending, sector-start
    /// arc lengths; the first sector implicitly starts at `s = 0`). Absent ⇒
    /// the classic equal-arc-length-thirds split (`apex_physics::DEFAULT_SECTOR_COUNT`).
    /// Intentionally **excluded** from [`processed_track_hash`]: it's
    /// lap-timing metadata, not geometry the solvers integrate over.
    pub sector_markers: Option<Vec<f64>>,
}

impl apex_math::ContentHash for TrackSegment {
    /// Encode all seven geometric fields in declaration order. The destructure
    /// forces any new field to be handled here before it compiles.
    fn hash_into(&self, w: &mut apex_math::HashWriter) {
        let TrackSegment {
            s,
            x,
            y,
            heading,
            curvature,
            width_left,
            width_right,
        } = self;
        w.f64(*s);
        w.f64(*x);
        w.f64(*y);
        w.f64(*heading);
        w.f64(*curvature);
        w.f64(*width_left);
        w.f64(*width_right);
    }
}

impl apex_math::ContentHash for TrackPoint {
    /// Encode all four raw fields in declaration order.
    fn hash_into(&self, w: &mut apex_math::HashWriter) {
        let TrackPoint {
            x,
            y,
            width_left,
            width_right,
        } = self;
        w.f64(*x);
        w.f64(*y);
        w.f64(*width_left);
        w.f64(*width_right);
    }
}

/// Processed-geometry content hash of a [`Track`], under domain
/// `"track.processed"`.
///
/// Hashes the segments (the exact geometry the solvers/QSS consume) plus
/// `is_closed`, and deliberately EXCLUDES `name` (a human label) and
/// `total_length` (derived from the segments). A renamed-but-identical track
/// therefore hashes the same. The segment count is length-prefixed so tracks
/// of different lengths cannot collide.
pub fn processed_track_hash(track: &Track) -> apex_math::Hash {
    use apex_math::ContentHash;
    let mut w = apex_math::HashWriter::new();
    w.str(apex_math::HASH_VERSION);
    w.str("track.processed");
    // name and total_length are intentionally not hashed (see doc comment).
    w.bool(track.is_closed);
    w.u64(track.segments.len() as u64);
    for seg in &track.segments {
        seg.hash_into(&mut w);
    }
    w.finish()
}

/// Raw-input content hash of a track's centerline, under domain `"track.raw"`.
///
/// Hashes the raw [`TrackPoint`] sequence (the source geometry, before
/// processing into segments). This is distinct from and not comparable to
/// [`processed_track_hash`]; the two domains never collide.
pub fn raw_track_hash(points: &[TrackPoint]) -> apex_math::Hash {
    use apex_math::ContentHash;
    let mut w = apex_math::HashWriter::new();
    w.str(apex_math::HASH_VERSION);
    w.str("track.raw");
    w.u64(points.len() as u64);
    for p in points {
        p.hash_into(&mut w);
    }
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_point_serde_roundtrip() {
        let p = TrackPoint {
            x: 1.5,
            y: -2.25,
            width_left: 3.0,
            width_right: 4.0,
        };
        let json = serde_json::to_string(&p).expect("serialize");
        let back: TrackPoint = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.x, p.x);
        assert_eq!(back.y, p.y);
        assert_eq!(back.width_left, p.width_left);
        assert_eq!(back.width_right, p.width_right);
    }
}
