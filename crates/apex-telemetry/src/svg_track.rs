//! Standalone SVG renderer for a track map with a speed-colored racing line.

use std::fmt::Write as _;
use std::path::Path;

use apex_track::Track;

use crate::RunMetadata;

/// Output image dimensions (pixels).
const IMG_W: f64 = 1200.0;
const IMG_H: f64 = 900.0;

/// Maps a speed to an RGB color using a 5-stop blue→cyan→green→yellow→red
/// gradient. Returns a default green when `max_speed <= min_speed`.
fn speed_to_color(speed: f64, min_speed: f64, max_speed: f64) -> (u8, u8, u8) {
    if max_speed <= min_speed {
        return (0, 255, 0);
    }
    let t = ((speed - min_speed) / (max_speed - min_speed)).clamp(0.0, 1.0);

    // (position, (r, g, b)) stops
    let stops: [(f64, (f64, f64, f64)); 5] = [
        (0.0, (0.0, 0.0, 255.0)),    // blue
        (0.25, (0.0, 255.0, 255.0)), // cyan
        (0.5, (0.0, 255.0, 0.0)),    // green
        (0.75, (255.0, 255.0, 0.0)), // yellow
        (1.0, (255.0, 0.0, 0.0)),    // red
    ];

    for w in stops.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 + 1e-12 {
            let local = if (t1 - t0).abs() > 1e-12 {
                (t - t0) / (t1 - t0)
            } else {
                0.0
            };
            let r = (c0.0 + (c1.0 - c0.0) * local).round() as u8;
            let g = (c0.1 + (c1.1 - c0.1) * local).round() as u8;
            let b = (c0.2 + (c1.2 - c0.2) * local).round() as u8;
            return (r, g, b);
        }
    }
    (255, 0, 0)
}

fn hex(c: (u8, u8, u8)) -> String {
    format!("#{:02x}{:02x}{:02x}", c.0, c.1, c.2)
}

/// Renders a track map to a standalone SVG file with a color-coded speed
/// overlay.
///
/// `speeds` must have the same length as `track.segments`. A [`RunMetadata`]
/// `<metadata>` provenance element is embedded immediately after the root
/// `<svg>` open tag.
pub fn render_track_svg(
    path: &Path,
    meta: &RunMetadata,
    track: &Track,
    speeds: &[f64],
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let segs = &track.segments;
    let n = segs.len().min(speeds.len());
    if n == 0 {
        return Err("cannot render an empty track".into());
    }

    // --- bounding box of centerline ---
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;
    for seg in segs.iter().take(n) {
        min_x = min_x.min(seg.x);
        max_x = max_x.max(seg.x);
        min_y = min_y.min(seg.y);
        max_y = max_y.max(seg.y);
    }
    let world_w = (max_x - min_x).max(1e-6);
    let world_h = (max_y - min_y).max(1e-6);
    let cx = 0.5 * (min_x + max_x);
    let cy = 0.5 * (min_y + max_y);

    // 15% padding on all sides -> usable extent is 1.3x the world span
    let padded_w = world_w * 1.3;
    let padded_h = world_h * 1.3;
    let scale = (IMG_W / padded_w).min(IMG_H / padded_h);

    // world -> pixel, flipping Y so +Y is up
    let px = |x: f64| IMG_W * 0.5 + (x - cx) * scale;
    let py = |y: f64| IMG_H * 0.5 - (y - cy) * scale;

    // stroke widths derived from the physical track width in pixels
    let track_width_px = (segs[0].width_left + segs[0].width_right) * scale;
    let stroke_boundary = (track_width_px * 0.1).max(1.0);
    let stroke_race = stroke_boundary * 2.5;

    let min_speed = speeds[..n].iter().cloned().fold(f64::MAX, f64::min);
    let max_speed = speeds[..n].iter().cloned().fold(f64::MIN, f64::max);

    // --- boundary points ---
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    for seg in segs.iter().take(n) {
        let nx = (seg.heading + std::f64::consts::FRAC_PI_2).cos();
        let ny = (seg.heading + std::f64::consts::FRAC_PI_2).sin();
        left.push((seg.x + seg.width_left * nx, seg.y + seg.width_left * ny));
        right.push((seg.x - seg.width_right * nx, seg.y - seg.width_right * ny));
    }

    // --- build SVG ---
    let mut svg = String::new();
    writeln!(svg, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
    writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">",
        IMG_W as i32, IMG_H as i32, IMG_W as i32, IMG_H as i32
    )?;

    // provenance metadata (valid XML <metadata> element)
    svg.push_str(&meta.svg_metadata_element());

    // background
    writeln!(
        svg,
        "  <rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#1a1a2e\"/>",
        IMG_W as i32, IMG_H as i32
    )?;

    // track surface polygon: left forward, then right backward
    svg.push_str("  <polygon points=\"");
    for &(x, y) in left.iter() {
        write!(svg, "{:.1},{:.1} ", px(x), py(y))?;
    }
    for &(x, y) in right.iter().rev() {
        write!(svg, "{:.1},{:.1} ", px(x), py(y))?;
    }
    writeln!(svg, "\" fill=\"#2d2d44\" stroke=\"none\"/>")?;

    // boundary polylines
    let mut write_boundary = |pts: &[(f64, f64)], close: bool| -> std::fmt::Result {
        svg.push_str("  <polyline points=\"");
        for &(x, y) in pts.iter() {
            write!(svg, "{:.1},{:.1} ", px(x), py(y))?;
        }
        if close {
            let (x, y) = pts[0];
            write!(svg, "{:.1},{:.1} ", px(x), py(y))?;
        }
        writeln!(
            svg,
            "\" fill=\"none\" stroke=\"#555555\" stroke-width=\"{:.2}\"/>",
            stroke_boundary
        )
    };
    write_boundary(&left, track.is_closed)?;
    write_boundary(&right, track.is_closed)?;

    // speed-colored centerline
    let last = if track.is_closed { n } else { n - 1 };
    for i in 0..last {
        let j = (i + 1) % n;
        let color = hex(speed_to_color(speeds[i], min_speed, max_speed));
        writeln!(
            svg,
            "  <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"{:.2}\" stroke-linecap=\"round\"/>",
            px(segs[i].x),
            py(segs[i].y),
            px(segs[j].x),
            py(segs[j].y),
            color,
            stroke_race
        )?;
    }

    // start/finish line (perpendicular at segment 0, spanning the track width)
    writeln!(
        svg,
        "  <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#ffffff\" stroke-width=\"{:.2}\"/>",
        px(left[0].0),
        py(left[0].1),
        px(right[0].0),
        py(right[0].1),
        stroke_boundary
    )?;

    // title
    writeln!(
        svg,
        "  <text x=\"30\" y=\"50\" font-family=\"sans-serif\" font-size=\"34\" fill=\"#ffffff\">{}</text>",
        title
    )?;

    // legend: gradient color bar in the bottom-right
    let bar_w = 260.0;
    let bar_h = 18.0;
    let bar_x = IMG_W - bar_w - 30.0;
    let bar_y = IMG_H - bar_h - 40.0;
    let steps = 60;
    for k in 0..steps {
        let frac = k as f64 / (steps - 1) as f64;
        let sp = min_speed + frac * (max_speed - min_speed);
        let color = hex(speed_to_color(sp, min_speed, max_speed));
        let seg_x = bar_x + frac * bar_w;
        writeln!(
            svg,
            "  <rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.2}\" height=\"{:.1}\" fill=\"{}\"/>",
            seg_x,
            bar_y,
            bar_w / steps as f64 + 1.0,
            bar_h,
            color
        )?;
    }
    writeln!(
        svg,
        "  <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"18\" fill=\"#ffffff\" text-anchor=\"start\">{:.0} km/h</text>",
        bar_x,
        bar_y - 6.0,
        min_speed * 3.6
    )?;
    writeln!(
        svg,
        "  <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"18\" fill=\"#ffffff\" text-anchor=\"end\">{:.0} km/h</text>",
        bar_x + bar_w,
        bar_y - 6.0,
        max_speed * 3.6
    )?;

    writeln!(svg, "</svg>")?;

    std::fs::write(path, svg)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings_hash_for_mode;
    use apex_physics::{qss_lap_sim, CarParams};
    use apex_track::{build_track, circle_track, oval_track};

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(name)
    }

    fn test_meta() -> RunMetadata {
        RunMetadata::new(
            settings_hash_for_mode("test-car"),
            settings_hash_for_mode("test-track"),
            settings_hash_for_mode("test-settings"),
            None,
        )
    }

    #[test]
    fn color_endpoints_and_midpoint() {
        assert_eq!(speed_to_color(10.0, 10.0, 50.0), (0, 0, 255)); // min -> blue
        assert_eq!(speed_to_color(50.0, 10.0, 50.0), (255, 0, 0)); // max -> red
        assert_eq!(speed_to_color(30.0, 10.0, 50.0), (0, 255, 0)); // mid -> green
    }

    #[test]
    fn color_equal_min_max_does_not_panic() {
        let c = speed_to_color(5.0, 5.0, 5.0);
        assert_eq!(c, (0, 255, 0));
    }

    #[test]
    fn oval_svg_is_valid() {
        let params = CarParams::default();
        let (points, closed) = oval_track(500.0, 80.0, 12.0, 300);
        let track = build_track("Oval", &points, closed);
        let result = qss_lap_sim(&track, &params);

        let path = temp_path("apex_test_oval.svg");
        render_track_svg(&path, &test_meta(), &track, &result.speeds, "Apex-14 Oval")
            .expect("render");

        let contents = std::fs::read_to_string(&path).expect("read");
        assert!(contents.starts_with("<?xml") || contents.starts_with("<svg"));
        assert!(contents.contains("</svg>"));
        assert!(contents.contains("Apex-14 Oval"));
        // provenance element present and well-formed
        assert!(contents.contains("<metadata>"));
        assert!(contents.contains("</metadata>"));
        assert!(contents.contains("<apex:config_hash>"));
        assert!(contents.len() > 1000, "svg only {} bytes", contents.len());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn circle_svg_is_valid() {
        let params = CarParams::default();
        let (points, closed) = circle_track(100.0, 12.0, 200);
        let track = build_track("Circle", &points, closed);
        let result = qss_lap_sim(&track, &params);

        let path = temp_path("apex_test_circle.svg");
        render_track_svg(
            &path,
            &test_meta(),
            &track,
            &result.speeds,
            "Apex-14 Circle",
        )
        .expect("render");

        let contents = std::fs::read_to_string(&path).expect("read");
        assert!(contents.starts_with("<?xml") || contents.starts_with("<svg"));
        assert!(contents.contains("</svg>"));
        assert!(contents.contains("Apex-14 Circle"));
        assert!(contents.contains("<metadata>"));
        assert!(contents.len() > 1000, "svg only {} bytes", contents.len());

        std::fs::remove_file(&path).ok();
    }
}
