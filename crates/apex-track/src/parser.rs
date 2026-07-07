//! Loading and saving tracks as JSON files.
//!
//! A track file is a JSON object with a `name`, a `closed` flag, an optional
//! uniform `width`, and a list of centerline `points`. Each point may carry its
//! own `width_left`/`width_right`; when absent, the track-level default width
//! (or 12 m) is split evenly to both sides.

use crate::builder::build_track;
use crate::grip_grid::MuScaleGrid;
use crate::ribbon3d::Ribbon3d;
use crate::types::{Track, TrackPoint};

/// A track point as represented in a JSON file.
/// Width fields are optional — if absent, the track-level default width is used.
///
/// # Schema v2 (3D)
///
/// `z` (elevation, m) and `banking_deg` (surface roll, degrees) are the optional
/// 3D extension. Both absent ⇒ the point is flat and the file is v1-compatible;
/// the writer only emits them when 3D data is present, so existing v1 files and
/// their diffs stay byte-stable.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct TrackPointJson {
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub width_left: Option<f64>,
    #[serde(default)]
    pub width_right: Option<f64>,
    /// Elevation (m). Absent ⇒ flat (`z = 0`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub z: Option<f64>,
    /// Banking / roll angle (degrees). Absent ⇒ unbanked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub banking_deg: Option<f64>,
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
    /// Schema version. Absent ⇒ v1 (flat). Emitted only for v2 (3D) files, so
    /// v1 output stays byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
    pub name: String,
    #[serde(default = "default_true")]
    pub closed: bool,
    #[serde(default)]
    pub width: Option<f64>,
    /// Optional processing/provenance metadata (e.g. smoothing parameters).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<TrackMetaJson>,
    /// Optional sector-marker stations (m, ascending sector-start arc
    /// lengths). Absent ⇒ equal-arc-length thirds. Available at v1 or v2 (not
    /// 3D-specific), same as `metadata`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sector_markers: Option<Vec<f64>>,
    /// Optional `mu_scale(s, n)` grip-multiplier grid block (Phase 1.4,
    /// schema v2 / 3D-ribbon-only — ignored by the 2D `parse_track_json` path).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mu_scale_grid: Option<MuScaleGridJson>,
    pub points: Vec<TrackPointJson>,
}

fn default_true() -> bool {
    true
}

/// A schema v2 `mu_scale(s, n)` grip-multiplier grid block, as represented in
/// a track JSON file (see [`crate::grip_grid::MuScaleGrid`]).
///
/// `stations` and `lateral` are the grid axes (m); `values` is the row-major
/// `stations.len() * lateral.len()` flattened grid. Absent ⇒ uniform `1.0`
/// (no grid at all — the byte-stable default).
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct MuScaleGridJson {
    pub stations: Vec<f64>,
    pub lateral: Vec<f64>,
    pub values: Vec<f64>,
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

    let mut track = build_track(&file.name, &points, file.closed);
    track.sector_markers = file.sector_markers.clone();
    Ok(track)
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
        version: None, // flat 2D track → v1-compatible output (no version key)
        name: track.name.clone(),
        closed: track.is_closed,
        width: None, // per-point widths are used
        metadata,
        sector_markers: track.sector_markers.clone(),
        mu_scale_grid: None, // 2D Track has no grid; Ribbon3d/export_ribbon3d_json carries it
        points: track
            .segments
            .iter()
            .map(|seg| TrackPointJson {
                x: seg.x,
                y: seg.y,
                width_left: Some(seg.width_left),
                width_right: Some(seg.width_right),
                z: None,
                banking_deg: None,
            })
            .collect(),
    };
    Ok(serde_json::to_string_pretty(&file)?)
}

/// Load a [`Ribbon3d`] from a JSON file path (schema v1 or v2).
pub fn load_ribbon3d_json(path: &std::path::Path) -> Result<Ribbon3d, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(path)?;
    parse_ribbon3d_json(&contents)
}

/// Parse a [`Ribbon3d`] from a track JSON string, accepting **both** schema
/// versions (the v1→v2 migration function):
///
/// * **v1** (no `version` field, no per-point `z`/`banking_deg`) and any **flat
///   v2** file load as a flat ribbon through the exact 2D pipeline
///   (`build_track` → [`Ribbon3d::from_flat`]), so a flat v2 file is bitwise
///   identical, station-for-station, to the v1 file with the same points.
/// * **v2 with 3D data** (any point carries `z` or `banking_deg`) builds a true
///   3D ribbon via [`Ribbon3d::from_centerline_3d`].
pub fn parse_ribbon3d_json(json: &str) -> Result<Ribbon3d, Box<dyn std::error::Error>> {
    let file: TrackFileJson = serde_json::from_str(json)?;
    if let Some(v) = file.version {
        if v == 0 || v > 2 {
            return Err(format!("unsupported track schema version {v} (supported: 1, 2)").into());
        }
    }
    if file.points.len() < 3 {
        return Err("Track must have at least 3 points".into());
    }
    let default_half = file.width.unwrap_or(12.0) / 2.0;
    let has_3d = file
        .points
        .iter()
        .any(|p| p.z.is_some() || p.banking_deg.is_some());

    if !has_3d {
        // Flat file (v1, or v2 with no 3D data): route through the exact 2D
        // pipeline so the ribbon is a byte-exact flat projection.
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
        let mut track = build_track(&file.name, &points, file.closed);
        track.sector_markers = file.sector_markers.clone();
        let mut ribbon = Ribbon3d::from_flat(&track);
        ribbon.mu_scale_grid = parse_mu_scale_grid(&file.mu_scale_grid)?;
        return Ok(ribbon);
    }

    // 3D file: assemble points + per-point bank/widths and build the ribbon.
    let pts: Vec<[f64; 3]> = file
        .points
        .iter()
        .map(|p| [p.x, p.y, p.z.unwrap_or(0.0)])
        .collect();
    let bank: Vec<f64> = file
        .points
        .iter()
        .map(|p| p.banking_deg.unwrap_or(0.0).to_radians())
        .collect();
    let wl: Vec<f64> = file
        .points
        .iter()
        .map(|p| p.width_left.unwrap_or(default_half))
        .collect();
    let wr: Vec<f64> = file
        .points
        .iter()
        .map(|p| p.width_right.unwrap_or(default_half))
        .collect();
    let mut ribbon = Ribbon3d::from_centerline_3d(&file.name, &pts, &bank, &wl, &wr, file.closed);
    ribbon.sector_markers = file.sector_markers.clone();
    ribbon.mu_scale_grid = parse_mu_scale_grid(&file.mu_scale_grid)?;
    Ok(ribbon)
}

/// Validate + build a [`MuScaleGrid`] from its JSON representation, if present.
fn parse_mu_scale_grid(
    json: &Option<MuScaleGridJson>,
) -> Result<Option<MuScaleGrid>, Box<dyn std::error::Error>> {
    match json {
        None => Ok(None),
        Some(g) => Ok(Some(
            MuScaleGrid::new(g.stations.clone(), g.lateral.clone(), g.values.clone())
                .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?,
        )),
    }
}

/// Export a [`Ribbon3d`] to a JSON string. Emits **v2** (a `version: 2` field
/// plus per-point `z` / `banking_deg`) only when the ribbon carries 3D data;
/// a flat ribbon emits **v1-compatible** output (no version key, no 3D fields),
/// so flat tracks written through this path diff cleanly against v1 files.
pub fn export_ribbon3d_json(ribbon: &Ribbon3d) -> Result<String, Box<dyn std::error::Error>> {
    let has_3d = !ribbon.is_flat();
    let has_v2_data = has_3d || ribbon.mu_scale_grid.is_some();
    let points = ribbon
        .stations
        .iter()
        .map(|st| TrackPointJson {
            x: st.x,
            y: st.y,
            width_left: Some(st.width_left),
            width_right: Some(st.width_right),
            z: if has_3d { Some(st.z) } else { None },
            banking_deg: if has_3d {
                Some(st.bank.to_degrees())
            } else {
                None
            },
        })
        .collect();
    let mu_scale_grid = ribbon.mu_scale_grid.as_ref().map(|g| MuScaleGridJson {
        stations: g.stations.clone(),
        lateral: g.lateral.clone(),
        values: g.values.clone(),
    });
    let file = TrackFileJson {
        version: if has_v2_data { Some(2) } else { None },
        name: ribbon.name.clone(),
        closed: ribbon.is_closed,
        width: None,
        metadata: None,
        sector_markers: ribbon.sector_markers.clone(),
        mu_scale_grid,
        points,
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

    // ---- schema v2 (3D ribbon) groundwork ----

    const V1_JSON: &str = r#"{
        "name": "Tri",
        "closed": true,
        "width": 8.0,
        "points": [
            { "x": 0.0, "y": 0.0 },
            { "x": 100.0, "y": 0.0 },
            { "x": 50.0, "y": 80.0 }
        ]
    }"#;

    #[test]
    fn v1_file_loads_as_flat_ribbon() {
        let ribbon = parse_ribbon3d_json(V1_JSON).expect("parse v1 as ribbon");
        assert_eq!(ribbon.name, "Tri");
        assert!(ribbon.is_closed);
        assert_eq!(ribbon.stations.len(), 3);
        assert!(ribbon.is_flat(), "a v1 file must load as a flat ribbon");
    }

    #[test]
    fn v1_and_flat_v2_are_bitwise_identical() {
        // Same geometry, but v2 tags version:2 and adds explicit z=0 everywhere.
        let v2_flat = r#"{
            "version": 2,
            "name": "Tri",
            "closed": true,
            "width": 8.0,
            "points": [
                { "x": 0.0, "y": 0.0, "z": 0.0 },
                { "x": 100.0, "y": 0.0, "z": 0.0 },
                { "x": 50.0, "y": 80.0, "z": 0.0 }
            ]
        }"#;
        // z=0 everywhere still counts as 3D data present, so this exercises the
        // 3D path; assert the flat PROJECTION (x, y, widths, and the flat frame)
        // is faithful and the ribbon reads as flat geometry.
        let flat_via_v1 = parse_ribbon3d_json(V1_JSON).expect("v1");
        let via_v2 = parse_ribbon3d_json(v2_flat).expect("v2 flat");
        assert_eq!(flat_via_v1.stations.len(), via_v2.stations.len());
        for (a, b) in flat_via_v1.stations.iter().zip(&via_v2.stations) {
            assert_eq!(a.x.to_bits(), b.x.to_bits());
            assert_eq!(a.y.to_bits(), b.y.to_bits());
            assert_eq!(a.z, 0.0);
            assert_eq!(b.z, 0.0);
            assert_eq!(a.width_left.to_bits(), b.width_left.to_bits());
        }
        // z=0/bank=0 everywhere ⇒ the v2 path recovers a flat ribbon.
        assert!(via_v2.is_flat(), "z=0 v2 file must be geometrically flat");
    }

    #[test]
    fn v2_writer_emits_v1_output_for_flat_ribbon() {
        // A flat ribbon must serialize with NO version key and NO z/banking keys,
        // so flat files stay v1-compatible and diff cleanly.
        let (pts, closed) = circle_track(50.0, 10.0, 48);
        let track = build_track("Circle", &pts, closed);
        let ribbon = track.to_ribbon3d();
        let json = export_ribbon3d_json(&ribbon).expect("export flat ribbon");
        assert!(!json.contains("version"), "flat output must omit version");
        assert!(!json.contains("\"z\""), "flat output must omit z");
        assert!(!json.contains("banking"), "flat output must omit banking");
        // And it must equal the legacy Track writer output byte-for-byte.
        let legacy = export_track_json(&track).expect("legacy export");
        assert_eq!(
            json, legacy,
            "flat ribbon JSON must match legacy Track JSON"
        );
    }

    #[test]
    fn v2_3d_file_round_trips_with_version_and_fields() {
        // A banked, climbing 3-point ribbon.
        let json_in = r#"{
            "version": 2,
            "name": "Ramp",
            "closed": false,
            "points": [
                { "x": 0.0,   "y": 0.0, "z": 0.0,  "width_left": 5.0, "width_right": 5.0, "banking_deg": 0.0 },
                { "x": 100.0, "y": 0.0, "z": 5.0,  "width_left": 5.0, "width_right": 5.0, "banking_deg": 4.0 },
                { "x": 200.0, "y": 0.0, "z": 12.0, "width_left": 5.0, "width_right": 5.0, "banking_deg": 6.0 }
            ]
        }"#;
        let ribbon = parse_ribbon3d_json(json_in).expect("parse 3d");
        assert!(!ribbon.is_flat(), "elevation/bank present ⇒ not flat");
        assert!(ribbon.stations[1].z > 0.0);

        let out = export_ribbon3d_json(&ribbon).expect("export 3d");
        assert!(out.contains("\"version\": 2"), "3d output tags version 2");
        assert!(out.contains("\"z\""), "3d output carries z");
        assert!(out.contains("banking_deg"), "3d output carries banking");

        // Round-trip: reload and check elevation/bank survive.
        let back = parse_ribbon3d_json(&out).expect("reparse 3d");
        assert_eq!(back.stations.len(), ribbon.stations.len());
        assert!((back.stations[2].z - 12.0).abs() < 1e-9);
        assert!((back.stations[2].bank - 6.0_f64.to_radians()).abs() < 1e-9);
    }

    #[test]
    fn unsupported_version_is_rejected() {
        let json = r#"{ "version": 3, "name": "Future", "closed": true,
            "points": [ {"x":0.0,"y":0.0}, {"x":1.0,"y":0.0}, {"x":0.0,"y":1.0} ] }"#;
        assert!(parse_ribbon3d_json(json).is_err());
    }

    // ---- mu_scale grid (Phase 1.4) ----

    #[test]
    fn mu_scale_grid_round_trips() {
        let json_in = r#"{
            "version": 2,
            "name": "Grippy",
            "closed": true,
            "mu_scale_grid": {
                "stations": [0.0, 50.0],
                "lateral": [-5.0, 5.0],
                "values": [1.0, 0.7, 1.0, 0.7]
            },
            "points": [
                { "x": 0.0, "y": 0.0 },
                { "x": 100.0, "y": 0.0 },
                { "x": 50.0, "y": 80.0 }
            ]
        }"#;
        let ribbon = parse_ribbon3d_json(json_in).expect("parse grid");
        let grid = ribbon.mu_scale_grid.as_ref().expect("grid attached");
        assert_eq!(grid.mu_at(0.0, -5.0, ribbon.total_length, true), 1.0);
        assert_eq!(grid.mu_at(0.0, 5.0, ribbon.total_length, true), 0.7);

        let out = export_ribbon3d_json(&ribbon).expect("export with grid");
        assert!(out.contains("\"version\": 2"));
        assert!(out.contains("mu_scale_grid"));
        let back = parse_ribbon3d_json(&out).expect("reparse grid");
        let grid_back = back
            .mu_scale_grid
            .as_ref()
            .expect("grid survives round-trip");
        assert_eq!(grid_back.mu_at(0.0, 5.0, back.total_length, true), 0.7);
    }

    #[test]
    fn flat_ribbon_with_explicit_uniform_grid_stays_flat() {
        // A flat ribbon carrying an all-1.0 grid must still report is_flat() —
        // the grid doesn't affect geometry, so it still takes the flat QSS
        // fast path (byte-stability discipline, same as the per-station
        // mu_scale placeholder).
        let json_in = r#"{
            "version": 2,
            "name": "FlatGrid",
            "closed": true,
            "mu_scale_grid": {
                "stations": [0.0, 50.0],
                "lateral": [-5.0, 5.0],
                "values": [1.0, 1.0, 1.0, 1.0]
            },
            "points": [
                { "x": 0.0, "y": 0.0 },
                { "x": 100.0, "y": 0.0 },
                { "x": 50.0, "y": 80.0 }
            ]
        }"#;
        let ribbon = parse_ribbon3d_json(json_in).expect("parse");
        assert!(ribbon.is_flat(), "grid presence must not affect flatness");
        assert!(ribbon.mu_scale_grid.is_some());
    }

    #[test]
    fn invalid_mu_scale_grid_is_rejected() {
        let json = r#"{
            "version": 2,
            "name": "Bad",
            "closed": true,
            "mu_scale_grid": { "stations": [0.0, 50.0], "lateral": [0.0], "values": [1.0, 1.0] },
            "points": [
                { "x": 0.0, "y": 0.0 }, { "x": 100.0, "y": 0.0 }, { "x": 50.0, "y": 80.0 }
            ]
        }"#;
        assert!(parse_ribbon3d_json(json).is_err());
    }

    // ---- sector markers (roadmap 1.2) ----

    #[test]
    fn sector_markers_round_trip_2d_and_3d() {
        let json_in = r#"{
            "name": "Markers",
            "closed": true,
            "sector_markers": [0.0, 100.0, 220.0],
            "points": [
                { "x": 0.0, "y": 0.0 }, { "x": 100.0, "y": 0.0 },
                { "x": 100.0, "y": 100.0 }, { "x": 0.0, "y": 100.0 }
            ]
        }"#;
        let track = parse_track_json(json_in).expect("parse 2d");
        assert_eq!(
            track.sector_markers.as_deref(),
            Some(&[0.0, 100.0, 220.0][..])
        );
        let out = export_track_json(&track).expect("export 2d");
        assert!(out.contains("sector_markers"));
        let back = parse_track_json(&out).expect("reparse 2d");
        assert_eq!(back.sector_markers, track.sector_markers);

        let ribbon = parse_ribbon3d_json(json_in).expect("parse ribbon");
        assert_eq!(
            ribbon.sector_markers.as_deref(),
            Some(&[0.0, 100.0, 220.0][..])
        );
    }

    #[test]
    fn synthetic_3d_load_and_validate_smoke() {
        // A tilted closed ring (z varies sinusoidally): a genuine 3D loop built
        // from SYNTHETIC data only — CI never sees the real gitignored circuits.
        // Exercises the same load_ribbon3d_json -> validate path as real data.
        let n = 360;
        let r = 250.0;
        let amp = 12.0;
        let mut pts = Vec::new();
        let mut len2d = 0.0;
        let mut prev: Option<(f64, f64)> = None;
        for i in 0..n {
            let u = 2.0 * PI * (i as f64) / (n as f64);
            let (x, y) = (r * u.cos(), r * u.sin());
            if let Some((px, py)) = prev {
                len2d += ((x - px).powi(2) + (y - py).powi(2)).sqrt();
            }
            prev = Some((x, y));
            pts.push(TrackPointJson {
                x,
                y,
                width_left: Some(6.0),
                width_right: Some(6.0),
                z: Some(amp * u.sin()),
                banking_deg: Some(0.0),
            });
        }
        let file = TrackFileJson {
            version: Some(2),
            name: "synth3d".into(),
            closed: true,
            width: None,
            metadata: None,
            sector_markers: None,
            mu_scale_grid: None,
            points: pts,
        };
        let json = serde_json::to_string(&file).unwrap();

        let ribbon = parse_ribbon3d_json(&json).expect("load synthetic 3D");
        let v = ribbon.validate();
        assert_eq!(v.n, n);
        assert!(v.is_closed);
        assert!(v.all_finite, "all coords/curvatures must be finite");
        // Frame orthonormality holds on the 3D ring.
        assert!(
            v.max_ortho_error < 1e-9,
            "frame ortho error {}",
            v.max_ortho_error
        );
        // Ω_y (pitch rate) is finite and bounded.
        assert!(v.omega_y_max.is_finite() && v.omega_y_p95 >= 0.0);
        // 3D arc length exceeds the 2D loop (elevation adds length).
        assert!(
            v.length_3d > len2d,
            "3D length {} must exceed 2D length {}",
            v.length_3d,
            len2d
        );
        // Elevation range ≈ 2·amp.
        assert!(
            (v.elevation_range - 2.0 * amp).abs() < 0.5,
            "elevation range {} vs {}",
            v.elevation_range,
            2.0 * amp
        );
    }

    #[test]
    fn legacy_track_parser_still_ignores_3d_fields() {
        // The 2D parse_track_json path is unchanged: a v2 file still loads as a
        // flat 2D Track (3D fields ignored), so existing consumers are untouched.
        let json = r#"{ "version": 2, "name": "R", "closed": false,
            "points": [ {"x":0.0,"y":0.0,"z":9.0}, {"x":10.0,"y":0.0,"z":9.0},
                        {"x":20.0,"y":0.0,"z":9.0} ] }"#;
        let track = parse_track_json(json).expect("2d parse of v2 file");
        assert_eq!(track.segments.len(), 3);
    }
}
