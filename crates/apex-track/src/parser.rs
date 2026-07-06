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

/// Provenance/processing metadata recorded in a track JSON file.
///
/// Currently records centerline-smoothing parameters so a re-imported track
/// carries proof of what was done to it (see [`crate::smoothing`]). All fields
/// are optional and skipped when empty, so older files parse unchanged.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct TrackMetaJson {
    /// Free-form source label (e.g. `"TUMFTM racetrack-database"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Whether the centerline was curvature-smoothed on import.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smoothed: Option<bool>,
    /// Deviation tolerance (m) used by smoothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smooth_tolerance_m: Option<f64>,
    /// Regularization weight `λ` chosen by smoothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smooth_lambda: Option<f64>,
    /// Actual maximum point deviation (m) achieved by smoothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smooth_max_deviation_m: Option<f64>,
}

/// A complete track definition as loaded from a JSON file.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct TrackFileJson {
    pub name: String,
    #[serde(default = "default_true")]
    pub closed: bool,
    #[serde(default)]
    pub width: Option<f64>,
    /// Optional processing/provenance metadata (e.g. smoothing parameters).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<TrackMetaJson>,
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
    export_track_json_with_meta(track, None)
}

/// Export a track to a JSON string, embedding optional processing
/// [`TrackMetaJson`] (e.g. smoothing provenance).
pub fn export_track_json_with_meta(
    track: &Track,
    metadata: Option<TrackMetaJson>,
) -> Result<String, Box<dyn std::error::Error>> {
    let file = TrackFileJson {
        name: track.name.clone(),
        closed: track.is_closed,
        width: None, // per-point widths are used
        metadata,
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

/// Import a track from the TUMFTM racetrack database CSV format.
///
/// The TUMFTM format uses 4 columns: `x_m, y_m, w_tr_right_m, w_tr_left_m`.
/// Coordinates are in meters (local Cartesian projection). Tracks are closed
/// circuits — the closure is implicit, so the last point connects back to the
/// first without being repeated.
///
/// Source: <https://github.com/TUMFTM/racetrack-database>
pub fn load_tumftm_csv(
    path: &std::path::Path,
    name: &str,
) -> Result<Track, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(path)?;
    parse_tumftm_csv(&contents, name)
}

/// Parse a track from a TUMFTM-format CSV string.
///
/// A leading header line (`x_m,...`) and `#` comment lines are skipped, so the
/// same parser handles both headered and headerless files. The `w_tr_right_m`
/// and `w_tr_left_m` columns are track half-widths and map directly to a
/// [`TrackPoint`]'s `width_right` / `width_left`.
pub fn parse_tumftm_csv(
    csv_content: &str,
    name: &str,
) -> Result<Track, Box<dyn std::error::Error>> {
    let mut points = Vec::new();

    for line in csv_content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Skip header line if present, plus any comment lines.
        if line.starts_with("x_m") || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 4 {
            return Err(format!("Expected 4 columns, got {}: {}", parts.len(), line).into());
        }

        let x: f64 = parts[0]
            .trim()
            .parse()
            .map_err(|e| format!("Failed to parse x: {} in line: {}", e, line))?;
        let y: f64 = parts[1]
            .trim()
            .parse()
            .map_err(|e| format!("Failed to parse y: {} in line: {}", e, line))?;
        let w_right: f64 = parts[2]
            .trim()
            .parse()
            .map_err(|e| format!("Failed to parse w_right: {} in line: {}", e, line))?;
        let w_left: f64 = parts[3]
            .trim()
            .parse()
            .map_err(|e| format!("Failed to parse w_left: {} in line: {}", e, line))?;

        points.push(TrackPoint {
            x,
            y,
            width_left: w_left,
            width_right: w_right,
        });
    }

    if points.len() < 3 {
        return Err(format!(
            "Track '{}' has only {} points, need at least 3",
            name,
            points.len()
        )
        .into());
    }

    // TUMFTM tracks are always closed circuits.
    Ok(build_track(name, &points, true))
}

/// Convert a [`Track`] to a TUMFTM-format CSV string (for export/compatibility).
///
/// Emits the standard `x_m,y_m,w_tr_right_m,w_tr_left_m` header followed by one
/// row per segment. This is the inverse of [`parse_tumftm_csv`].
pub fn export_tumftm_csv(track: &Track) -> String {
    let mut out = String::from("x_m,y_m,w_tr_right_m,w_tr_left_m\n");
    for seg in &track.segments {
        out.push_str(&format!(
            "{:.6},{:.6},{:.3},{:.3}\n",
            seg.x, seg.y, seg.width_right, seg.width_left
        ));
    }
    out
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

    // ---- TUMFTM CSV import/export ----

    /// A rough square, treated as a closed circuit. Eight points, 10 m spacing,
    /// uniform 5 m half-widths.
    const TEST_TUMFTM_CSV: &str = "\
x_m,y_m,w_tr_right_m,w_tr_left_m
0.000000,0.000000,5.000,5.000
10.000000,0.000000,5.000,5.000
20.000000,0.000000,5.000,5.000
20.000000,10.000000,5.000,5.000
20.000000,20.000000,5.000,5.000
10.000000,20.000000,5.000,5.000
0.000000,20.000000,5.000,5.000
0.000000,10.000000,5.000,5.000
";

    #[test]
    fn tumftm_parse_inline_square() {
        let track = parse_tumftm_csv(TEST_TUMFTM_CSV, "Square").expect("parse");
        assert_eq!(track.name, "Square");
        assert_eq!(track.segments.len(), 8);
        assert!(track.is_closed, "TUMFTM tracks are closed");

        // Perimeter of the implicit closed loop: 8 hops of 10 m each.
        let expected = 80.0;
        assert!(
            (track.total_length - expected).abs() < 1e-9,
            "total_length {} vs expected {}",
            track.total_length,
            expected
        );

        // Half-widths map straight through.
        for seg in &track.segments {
            assert_eq!(seg.width_left, 5.0);
            assert_eq!(seg.width_right, 5.0);
        }
    }

    #[test]
    fn tumftm_width_columns_map_to_correct_side() {
        // w_tr_right_m = 3, w_tr_left_m = 7 — verify they don't get swapped.
        let csv = "\
x_m,y_m,w_tr_right_m,w_tr_left_m
0,0,3.0,7.0
10,0,3.0,7.0
10,10,3.0,7.0
0,10,3.0,7.0
";
        let track = parse_tumftm_csv(csv, "Asym").expect("parse");
        assert_eq!(track.segments[0].width_right, 3.0);
        assert_eq!(track.segments[0].width_left, 7.0);
    }

    #[test]
    fn tumftm_round_trip() {
        let original = parse_tumftm_csv(TEST_TUMFTM_CSV, "Square").expect("parse");

        let exported = export_tumftm_csv(&original);
        let reparsed = parse_tumftm_csv(&exported, "Square").expect("reparse");

        assert_eq!(reparsed.segments.len(), original.segments.len());
        assert!(
            (reparsed.total_length - original.total_length).abs() / original.total_length < 0.01,
            "length {} vs {}",
            reparsed.total_length,
            original.total_length
        );
    }

    #[test]
    fn tumftm_header_optional() {
        // Same data with and without the header line must parse identically.
        let with_header = TEST_TUMFTM_CSV;
        let without_header = "\
0.000000,0.000000,5.000,5.000
10.000000,0.000000,5.000,5.000
20.000000,0.000000,5.000,5.000
20.000000,10.000000,5.000,5.000
20.000000,20.000000,5.000,5.000
10.000000,20.000000,5.000,5.000
0.000000,20.000000,5.000,5.000
0.000000,10.000000,5.000,5.000
";
        let a = parse_tumftm_csv(with_header, "T").expect("with header");
        let b = parse_tumftm_csv(without_header, "T").expect("without header");

        assert_eq!(a.segments.len(), b.segments.len());
        assert_eq!(a.total_length, b.total_length);
        for (sa, sb) in a.segments.iter().zip(b.segments.iter()) {
            assert_eq!(sa.x, sb.x);
            assert_eq!(sa.y, sb.y);
            assert_eq!(sa.curvature, sb.curvature);
        }
    }

    #[test]
    fn tumftm_error_too_few_columns() {
        let csv = "0.0,0.0,5.0\n10.0,0.0,5.0\n20.0,0.0,5.0\n";
        assert!(parse_tumftm_csv(csv, "Bad").is_err());
    }

    #[test]
    fn tumftm_error_non_numeric() {
        let csv = "0.0,0.0,5.0,5.0\nhello,world,5.0,5.0\n20.0,0.0,5.0,5.0\n";
        assert!(parse_tumftm_csv(csv, "Bad").is_err());
    }

    #[test]
    fn tumftm_error_empty() {
        assert!(parse_tumftm_csv("", "Empty").is_err());
        assert!(parse_tumftm_csv("x_m,y_m,w_tr_right_m,w_tr_left_m\n", "HeaderOnly").is_err());
    }
}
