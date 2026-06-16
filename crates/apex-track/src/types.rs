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
