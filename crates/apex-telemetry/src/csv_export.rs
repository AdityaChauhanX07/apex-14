//! CSV export helpers for QSS simulation output and arbitrary columnar data.

use std::path::Path;

use apex_physics::QssResult;
use apex_track::Track;

/// Format a float to 4 decimal places.
fn fmt(v: f64) -> String {
    format!("{:.4}", v)
}

/// Export QSS simulation results to a CSV file.
///
/// Columns: `s` (m), `x` (m), `y` (m), `speed` (m/s), `speed_kph` (km/h),
/// `lateral_g`, `longitudinal_g`, `curvature` (1/m).
pub fn export_qss_csv(
    path: &Path,
    track: &Track,
    result: &QssResult,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record([
        "s",
        "x",
        "y",
        "speed",
        "speed_kph",
        "lateral_g",
        "longitudinal_g",
        "curvature",
    ])?;

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
/// must have the same length.
pub fn export_columns_csv(
    path: &Path,
    columns: &[(&str, &[f64])],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)?;

    let header: Vec<&str> = columns.iter().map(|(name, _)| *name).collect();
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
    use apex_physics::qss_lap_sim;
    use apex_physics::CarParams;
    use apex_track::{build_track, circle_track};

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(name)
    }

    #[test]
    fn qss_csv_export() {
        let params = CarParams::default();
        let (points, closed) = circle_track(100.0, 12.0, 50);
        let track = build_track("circle", &points, closed);
        let result = qss_lap_sim(&track, &params);

        let path = temp_path("apex_test_qss.csv");
        export_qss_csv(&path, &track, &result).expect("export");

        let contents = std::fs::read_to_string(&path).expect("read");
        let lines: Vec<&str> = contents.lines().collect();

        // header
        assert_eq!(
            lines[0],
            "s,x,y,speed,speed_kph,lateral_g,longitudinal_g,curvature"
        );
        // one data row per segment
        assert_eq!(lines.len() - 1, track.segments.len());

        // spot-check a few speeds against the result (rounded to 4 dp)
        for i in [0, 10, 25, 49] {
            let fields: Vec<&str> = lines[i + 1].split(',').collect();
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
    fn columns_csv_export() {
        let distance = [0.0, 1.0, 2.0, 3.0, 4.0];
        let speed = [10.0, 20.0, 30.0, 40.0, 50.0];
        let path = temp_path("apex_test_columns.csv");

        export_columns_csv(&path, &[("distance", &distance[..]), ("speed", &speed[..])])
            .expect("export");

        let contents = std::fs::read_to_string(&path).expect("read");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines[0], "distance,speed");
        assert_eq!(lines.len() - 1, 5);

        let row2: Vec<&str> = lines[3].split(',').collect(); // distance=2, speed=30
        let d: f64 = row2[0].parse().unwrap();
        let s: f64 = row2[1].parse().unwrap();
        assert!((d - 2.0).abs() < 1e-9);
        assert!((s - 30.0).abs() < 1e-9);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn qss_csv_roundtrip_with_csv_crate() {
        let params = CarParams::default();
        let (points, closed) = circle_track(100.0, 12.0, 50);
        let track = build_track("circle", &points, closed);
        let result = qss_lap_sim(&track, &params);

        let path = temp_path("apex_test_qss_roundtrip.csv");
        export_qss_csv(&path, &track, &result).expect("export");

        let mut rdr = csv::Reader::from_path(&path).expect("reader");
        let headers = rdr.headers().expect("headers").clone();
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
}
