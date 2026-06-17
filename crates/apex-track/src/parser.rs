//! Loading and saving tracks as JSON files.
//!
//! A track file is a JSON object with a `name`, a `closed` flag, an optional
//! uniform `width`, and a list of centerline `points`. Each point may carry its
//! own `width_left`/`width_right`; when absent, the track-level default width
//! (or 12 m) is split evenly to both sides.

use crate::builder::build_track;
use crate::types::{Track, TrackPoint};

/// A track point as represented in a JSON file.
/// Width fields are optional — if absent, the track-level default width is used.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct TrackPointJson {
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub width_left: Option<f64>,
    #[serde(default)]
    pub width_right: Option<f64>,
}

/// A complete track definition as loaded from a JSON file.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct TrackFileJson {
    pub name: String,
    #[serde(default = "default_true")]
    pub closed: bool,
    #[serde(default)]
    pub width: Option<f64>,
    pub points: Vec<TrackPointJson>,
}

fn default_true() -> bool {
    true
}

/// Load a track from a JSON file path.
pub fn load_track_json(path: &std::path::Path) -> Result<Track, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(path)?;
    parse_track_json(&contents)
}

/// Parse a track from a JSON string.
pub fn parse_track_json(json: &str) -> Result<Track, Box<dyn std::error::Error>> {
    let file: TrackFileJson = serde_json::from_str(json)?;

    // Validate
    if file.points.len() < 3 {
        return Err("Track must have at least 3 points".into());
    }

    // Determine default half-width (per side).
    let default_half = file.width.unwrap_or(12.0) / 2.0;

    // Convert to TrackPoints.
    let points: Vec<TrackPoint> = file
        .points
        .iter()
        .map(|p| TrackPoint {
            x: p.x,
            y: p.y,
            width_left: p.width_left.unwrap_or(default_half),
            width_right: p.width_right.unwrap_or(default_half),
        })
        .collect();

    Ok(build_track(&file.name, &points, file.closed))
}

/// Export a track to a JSON string (for saving generated tracks).
pub fn export_track_json(track: &Track) -> Result<String, Box<dyn std::error::Error>> {
    let file = TrackFileJson {
        name: track.name.clone(),
        closed: track.is_closed,
        width: None, // per-point widths are used
        points: track
            .segments
            .iter()
            .map(|seg| TrackPointJson {
                x: seg.x,
                y: seg.y,
                width_left: Some(seg.width_left),
                width_right: Some(seg.width_right),
            })
            .collect(),
    };
    Ok(serde_json::to_string_pretty(&file)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::circle_track;
    use std::f64::consts::PI;

    #[test]
    fn parse_valid_triangle() {
        let json = r#"{
            "name": "Triangle",
            "closed": true,
            "width": 8.0,
            "points": [
                { "x": 0.0, "y": 0.0 },
                { "x": 100.0, "y": 0.0 },
                { "x": 50.0, "y": 80.0 }
            ]
        }"#;
        let track = parse_track_json(json).expect("parse");
        assert_eq!(track.name, "Triangle");
        assert!(track.is_closed);
        assert_eq!(track.segments.len(), 3);
        assert!(track.total_length > 0.0, "length {}", track.total_length);
    }

    #[test]
    fn parse_per_point_widths() {
        let json = r#"{
            "name": "Widths",
            "closed": false,
            "points": [
                { "x": 0.0, "y": 0.0, "width_left": 6.0, "width_right": 4.0 },
                { "x": 10.0, "y": 0.0, "width_left": 7.0, "width_right": 5.0 },
                { "x": 20.0, "y": 0.0, "width_left": 8.0, "width_right": 3.0 }
            ]
        }"#;
        let track = parse_track_json(json).expect("parse");
        assert_eq!(track.segments[0].width_left, 6.0);
        assert_eq!(track.segments[0].width_right, 4.0);
        assert_eq!(track.segments[1].width_left, 7.0);
        assert_eq!(track.segments[2].width_right, 3.0);
    }

    #[test]
    fn parse_uniform_width() {
        let json = r#"{
            "name": "Uniform",
            "closed": false,
            "width": 10.0,
            "points": [
                { "x": 0.0, "y": 0.0 },
                { "x": 10.0, "y": 0.0 },
                { "x": 20.0, "y": 0.0 }
            ]
        }"#;
        let track = parse_track_json(json).expect("parse");
        for seg in &track.segments {
            assert_eq!(seg.width_left, 5.0);
            assert_eq!(seg.width_right, 5.0);
        }
    }

    #[test]
    fn round_trip_circle() {
        let (pts, closed) = circle_track(50.0, 10.0, 48);
        let original = build_track("Circle", &pts, closed);

        let json = export_track_json(&original).expect("export");
        let parsed = parse_track_json(&json).expect("parse");

        assert_eq!(parsed.segments.len(), original.segments.len());
        assert!(
            (parsed.total_length - original.total_length).abs() / original.total_length < 0.01,
            "length {} vs {}",
            parsed.total_length,
            original.total_length
        );

        for frac in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let s = frac * original.total_length;
            let a = original.curvature_at(s);
            let b = parsed.curvature_at(s);
            // curvature ~ 1/50 = 0.02; compare on an absolute scale to avoid
            // dividing by near-zero on any low-curvature sample
            assert!(
                (a - b).abs() < 0.05 * 0.02 + 1e-6,
                "curvature {} vs {} at s={}",
                a,
                b,
                s
            );
        }
    }

    #[test]
    fn error_empty_points() {
        let json = r#"{ "name": "Empty", "closed": true, "points": [] }"#;
        assert!(parse_track_json(json).is_err());
    }

    #[test]
    fn error_two_points() {
        let json = r#"{
            "name": "TooFew",
            "points": [ { "x": 0.0, "y": 0.0 }, { "x": 1.0, "y": 1.0 } ]
        }"#;
        assert!(parse_track_json(json).is_err());
    }

    #[test]
    fn error_invalid_json() {
        assert!(parse_track_json("{ not valid json ]").is_err());
    }

    #[test]
    fn load_from_file() {
        let json = r#"{
            "name": "FileTrack",
            "closed": true,
            "width": 12.0,
            "points": [
                { "x": 0.0, "y": 0.0 },
                { "x": 100.0, "y": 0.0 },
                { "x": 100.0, "y": 100.0 },
                { "x": 0.0, "y": 100.0 }
            ]
        }"#;
        let path = std::env::temp_dir().join("apex14_parser_load_from_file.json");
        std::fs::write(&path, json).expect("write temp");

        let track = load_track_json(&path).expect("load");
        assert_eq!(track.name, "FileTrack");
        assert_eq!(track.segments.len(), 4);
        assert!(track.is_closed);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_committed_test_circle() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tracks/test_circle.json");
        let track = load_track_json(&path).expect("load test_circle.json");

        assert_eq!(track.segments.len(), 36);
        assert!(track.is_closed);

        let expected = 2.0 * PI * 50.0;
        assert!(
            (track.total_length - expected).abs() / expected < 0.05,
            "total_length {} vs expected {}",
            track.total_length,
            expected
        );
    }
}
