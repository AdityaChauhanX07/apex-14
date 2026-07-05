//! CSV export helpers for QSS simulation output and arbitrary columnar data.
//!
//! Every file opens with a [`RunMetadata`] provenance block: `# key: value`
//! comment lines, then a `# columns: name[unit], ...` line naming each column
//! and its unit (from the [channel registry](crate::channels)), then one blank
//! `#` line, then the column header row. A comment-aware CSV reader
//! (`ReaderBuilder::comment(Some(b'#'))`) skips the whole block transparently.
//!
//! Column headers are the registry [`ChannelId::name`](crate::ChannelId::name)s
//! — not ad-hoc string literals — so the CSV can never drift from the registry.

use std::io::Write as _;
use std::path::Path;

use apex_physics::QssResult;
use apex_track::Track;

use crate::channels::csv_columns_comment;
use crate::{ChannelId, RunMetadata};

/// The QSS CSV columns, in order, as registry channels. Headers and the units
/// line both derive from this single list.
const QSS_COLUMNS: [ChannelId; 8] = [
    ChannelId::S,
    ChannelId::X,
    ChannelId::Y,
    ChannelId::Speed,
    ChannelId::SpeedKph,
    ChannelId::LateralG,
    ChannelId::LongitudinalG,
    ChannelId::Curvature,
];

/// Format a float to 4 decimal places.
fn fmt(v: f64) -> String {
    format!("{:.4}", v)
}

/// Open `path`, write the provenance block (metadata lines + `# columns:` units
/// line + blank `#` separator), and return a CSV writer positioned to emit the
/// column header next. `column_names` is the ordered list of column headers.
fn writer_with_metadata(
    path: &Path,
    meta: &RunMetadata,
    column_names: &[&str],
) -> Result<csv::Writer<std::fs::File>, Box<dyn std::error::Error>> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(meta.csv_comment_lines().as_bytes())?;
    file.write_all(csv_columns_comment(column_names).as_bytes())?;
    file.write_all(b"\n#\n")?;
    Ok(csv::Writer::from_writer(file))
}

/// Export QSS simulation results to a CSV file.
///
/// Columns (from the registry): `s` (m), `x` (m), `y` (m), `speed` (m/s),
/// `speed_kph` (km/h), `lateral_g` (g), `longitudinal_g` (g), `curvature`
/// (1/m). Preceded by the [`RunMetadata`] provenance block.
pub fn export_qss_csv(
    path: &Path,
    meta: &RunMetadata,
    track: &Track,
    result: &QssResult,
) -> Result<(), Box<dyn std::error::Error>> {
    let names: Vec<&str> = QSS_COLUMNS.iter().map(|c| c.name()).collect();
    let mut wtr = writer_with_metadata(path, meta, &names)?;
    wtr.write_record(&names)?;

    for i in 0..track.segments.len() {
        let seg = &track.segments[i];
        wtr.write_record([
            fmt(result.distances[i]),
            fmt(seg.x),
            fmt(seg.y),
            fmt(result.speeds[i]),
            fmt(result.speeds[i] * 3.6),
            fmt(result.lateral_gs[i]),
            fmt(result.longitudinal_gs[i]),
            fmt(seg.curvature),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

/// Export arbitrary named columns to CSV.
///
/// `columns` is a slice of `(column_name, data_slice)` pairs. All data slices
/// must have the same length. Column names should be registry
/// [`ChannelId::name`](crate::ChannelId::name)s so the `# columns:` line carries
/// their units; an unregistered name renders `name[?]`.
pub fn export_columns_csv(
    path: &Path,
    meta: &RunMetadata,
    columns: &[(&str, &[f64])],
) -> Result<(), Box<dyn std::error::Error>> {
    let header: Vec<&str> = columns.iter().map(|(name, _)| *name).collect();
    let mut wtr = writer_with_metadata(path, meta, &header)?;
    wtr.write_record(&header)?;

    let rows = columns.first().map(|(_, data)| data.len()).unwrap_or(0);
    for r in 0..rows {
        let record: Vec<String> = columns.iter().map(|(_, data)| fmt(data[r])).collect();
        wtr.write_record(&record)?;
    }

    wtr.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings_hash_for_mode;
    use apex_physics::qss_lap_sim;
    use apex_physics::CarParams;
    use apex_track::{build_track, circle_track};

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(name)
    }

    fn test_meta(seed: Option<u64>) -> RunMetadata {
        RunMetadata::new(
            settings_hash_for_mode("test-car"),
            settings_hash_for_mode("test-track"),
            settings_hash_for_mode("test-settings"),
            seed,
        )
    }

    /// A comment-aware reader that skips the `#` provenance block.
    fn reader(path: &std::path::Path) -> csv::Reader<std::fs::File> {
        csv::ReaderBuilder::new()
            .comment(Some(b'#'))
            .from_path(path)
            .expect("reader")
    }

    #[test]
    fn qss_csv_export() {
        let params = CarParams::default();
        let (points, closed) = circle_track(100.0, 12.0, 50);
        let track = build_track("circle", &points, closed);
        let result = qss_lap_sim(&track, &params);

        let path = temp_path("apex_test_qss.csv");
        export_qss_csv(&path, &test_meta(None), &track, &result).expect("export");

        let contents = std::fs::read_to_string(&path).expect("read");
        let data: Vec<&str> = contents.lines().filter(|l| !l.starts_with('#')).collect();

        // header (first non-comment line)
        assert_eq!(
            data[0],
            "s,x,y,speed,speed_kph,lateral_g,longitudinal_g,curvature"
        );
        // one data row per segment
        assert_eq!(data.len() - 1, track.segments.len());

        // spot-check a few speeds against the result (rounded to 4 dp)
        for i in [0, 10, 25, 49] {
            let fields: Vec<&str> = data[i + 1].split(',').collect();
            let speed: f64 = fields[3].parse().unwrap();
            assert!(
                (speed - result.speeds[i]).abs() < 1e-3,
                "row {} speed {} vs {}",
                i,
                speed,
                result.speeds[i]
            );
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn csv_contains_all_metadata_keys() {
        let params = CarParams::default();
        let (points, closed) = circle_track(100.0, 12.0, 50);
        let track = build_track("circle", &points, closed);
        let result = qss_lap_sim(&track, &params);

        let path = temp_path("apex_test_qss_meta.csv");
        export_qss_csv(&path, &test_meta(Some(99)), &track, &result).expect("export");
        let contents = std::fs::read_to_string(&path).expect("read");

        for key in [
            "config_hash",
            "car_hash",
            "track_hash",
            "settings_hash",
            "git_sha",
            "apex_version",
            "seed",
            "timestamp",
        ] {
            assert!(
                contents.contains(&format!("# {key}: ")),
                "CSV missing metadata key {key}"
            );
        }
        assert!(contents.contains("# seed: 99"));
        // The provenance block carries the registry-derived units line, and the
        // header row matches the same registry names.
        assert!(
            contents.contains(
                "# columns: s[m], x[m], y[m], speed[m/s], speed_kph[km/h], \
                 lateral_g[g], longitudinal_g[g], curvature[1/m]"
            ),
            "CSV missing/incorrect units line; got:\n{contents}"
        );
        // The provenance block must precede the column header.
        let hdr = contents.find("s,x,y,speed").expect("header present");
        let first_data = contents.find("\n#\n").expect("blank comment line");
        assert!(first_data < hdr, "metadata block must precede the header");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn columns_csv_export() {
        let distance = [0.0, 1.0, 2.0, 3.0, 4.0];
        let speed = [10.0, 20.0, 30.0, 40.0, 50.0];
        let path = temp_path("apex_test_columns.csv");

        export_columns_csv(
            &path,
            &test_meta(None),
            &[("distance", &distance[..]), ("speed", &speed[..])],
        )
        .expect("export");

        let contents = std::fs::read_to_string(&path).expect("read");
        let data: Vec<&str> = contents.lines().filter(|l| !l.starts_with('#')).collect();
        assert_eq!(data[0], "distance,speed");
        assert_eq!(data.len() - 1, 5);

        let row2: Vec<&str> = data[3].split(',').collect(); // distance=2, speed=30
        let d: f64 = row2[0].parse().unwrap();
        let s: f64 = row2[1].parse().unwrap();
        assert!((d - 2.0).abs() < 1e-9);
        assert!((s - 30.0).abs() < 1e-9);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn qss_csv_roundtrip_skips_comments() {
        let params = CarParams::default();
        let (points, closed) = circle_track(100.0, 12.0, 50);
        let track = build_track("circle", &points, closed);
        let result = qss_lap_sim(&track, &params);

        let path = temp_path("apex_test_qss_roundtrip.csv");
        export_qss_csv(&path, &test_meta(Some(3)), &track, &result).expect("export");

        // A comment-aware reader must land on the real header and data, not the
        // provenance block, and every value must survive intact.
        let mut rdr = reader(&path);
        let headers = rdr.headers().expect("headers").clone();
        assert_eq!(&headers[0], "s");
        assert_eq!(&headers[3], "speed");
        assert_eq!(&headers[4], "speed_kph");

        let mut rows = 0;
        for record in rdr.records() {
            let record = record.expect("record");
            let speed: f64 = record[3].parse().unwrap();
            let speed_kph: f64 = record[4].parse().unwrap();
            assert!(
                (speed_kph - speed * 3.6).abs() < 0.01,
                "speed_kph {} vs {}",
                speed_kph,
                speed * 3.6
            );
            rows += 1;
        }
        assert_eq!(rows, track.segments.len());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn byte_identical_with_pinned_timestamp_and_seed() {
        // Serialize env mutation against the other env-touching test.
        let _g = crate::run_metadata::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::set_var("APEX_REPRO_TIMESTAMP", "2026-07-05T00:00:00Z");

        let params = CarParams::default();
        let (points, closed) = circle_track(100.0, 12.0, 50);
        let track = build_track("circle", &points, closed);
        let result = qss_lap_sim(&track, &params);

        // Both metadata blocks are built while the timestamp is pinned.
        let p1 = temp_path("apex_test_repro_a.csv");
        let p2 = temp_path("apex_test_repro_b.csv");
        export_qss_csv(&p1, &test_meta(Some(42)), &track, &result).expect("export a");
        export_qss_csv(&p2, &test_meta(Some(42)), &track, &result).expect("export b");

        std::env::remove_var("APEX_REPRO_TIMESTAMP");

        let a = std::fs::read(&p1).expect("read a");
        let b = std::fs::read(&p2).expect("read b");
        assert_eq!(
            a, b,
            "two writes with pinned timestamp+seed must be byte-identical"
        );

        std::fs::remove_file(&p1).ok();
        std::fs::remove_file(&p2).ok();
    }
}
