//! Standalone SVG renderer for a g-g(-g) diagram: the car's feasible-
//! acceleration envelope, one boundary polygon per `(v, g_z)` slice.
//!
//! Mirrors [`render_track_svg`](crate::svg_track::render_track_svg): a
//! self-contained SVG with a [`RunMetadata`] `<metadata>` provenance element
//! embedded right after the root `<svg>` tag. The renderer is data-only — it
//! takes pre-computed boundary polygons in acceleration space (m/s²), so the
//! telemetry crate stays decoupled from the physics envelope types.

use std::fmt::Write as _;
use std::path::Path;

use crate::RunMetadata;

/// One envelope slice to plot: a human label and the closed boundary polygon in
/// acceleration space, as `(a_y, a_x)` pairs (m/s²). `a_y` is the horizontal
/// axis (lateral), `a_x` the vertical axis (longitudinal, +up = drive).
pub struct EnvelopeSlicePlot {
    /// Legend label, e.g. `"v=60 m/s, g_z=9.8"`.
    pub label: String,
    /// Closed boundary polygon as `(a_y, a_x)` points (m/s²).
    pub boundary: Vec<(f64, f64)>,
}

/// Output image dimensions (pixels).
const IMG_W: f64 = 900.0;
const IMG_H: f64 = 900.0;

/// A small qualitative palette for the slice polygons (cycled).
const PALETTE: [&str; 6] = [
    "#4f9dff", "#ff6b6b", "#4fd18b", "#ffd166", "#c792ea", "#ff9f43",
];

/// Render a g-g diagram to a standalone SVG file.
///
/// `slices` are drawn as overlaid boundary polygons on a shared, symmetric
/// acceleration-space axis (auto-scaled to the largest slice). Fails on an empty
/// slice set or an empty polygon.
pub fn render_envelope_svg(
    path: &Path,
    meta: &RunMetadata,
    slices: &[EnvelopeSlicePlot],
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if slices.is_empty() || slices.iter().all(|s| s.boundary.is_empty()) {
        return Err("cannot render an empty envelope".into());
    }

    // Symmetric world extent from the largest coordinate magnitude across all
    // slices, so the friction-circle aspect ratio is preserved (1:1).
    let mut m = 1e-6_f64;
    for s in slices {
        for &(ay, ax) in &s.boundary {
            m = m.max(ay.abs()).max(ax.abs());
        }
    }
    let extent = m * 1.15; // 15% margin
    let scale = (IMG_W.min(IMG_H) * 0.5) / extent;
    let cx = IMG_W * 0.5;
    let cy = IMG_H * 0.5;
    // world (a_y, a_x) -> pixel; +a_x (drive) points up, +a_y (left) points right
    let px = |ay: f64| cx + ay * scale;
    let py = |ax: f64| cy - ax * scale;

    let mut svg = String::new();
    writeln!(svg, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
    writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">",
        IMG_W as i32, IMG_H as i32, IMG_W as i32, IMG_H as i32
    )?;
    svg.push_str(&meta.svg_metadata_element());

    // background
    writeln!(
        svg,
        "  <rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#12121c\"/>",
        IMG_W as i32, IMG_H as i32
    )?;

    // axes + concentric g-rings (every 1 g = 9.81 m/s²) for scale reference
    let g = 9.81;
    let mut ring = g;
    while ring <= extent {
        let r_px = ring * scale;
        writeln!(
            svg,
            "  <circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"{r_px:.1}\" fill=\"none\" stroke=\"#2c2c3c\" stroke-width=\"1\"/>",
        )?;
        ring += g;
    }
    writeln!(
        svg,
        "  <line x1=\"0\" y1=\"{cy:.1}\" x2=\"{IMG_W}\" y2=\"{cy:.1}\" stroke=\"#3a3a4c\" stroke-width=\"1\"/>",
    )?;
    writeln!(
        svg,
        "  <line x1=\"{cx:.1}\" y1=\"0\" x2=\"{cx:.1}\" y2=\"{IMG_H}\" stroke=\"#3a3a4c\" stroke-width=\"1\"/>",
    )?;

    // slice polygons
    for (i, s) in slices.iter().enumerate() {
        if s.boundary.is_empty() {
            continue;
        }
        let color = PALETTE[i % PALETTE.len()];
        svg.push_str("  <polygon points=\"");
        for &(ay, ax) in &s.boundary {
            write!(svg, "{:.1},{:.1} ", px(ay), py(ax))?;
        }
        writeln!(
            svg,
            "\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2\"/>"
        )?;
    }

    // labels
    writeln!(
        svg,
        "  <text x=\"12\" y=\"28\" fill=\"#e0e0e0\" font-family=\"sans-serif\" font-size=\"20\">{}</text>",
        xml_escape(title)
    )?;
    writeln!(
        svg,
        "  <text x=\"{:.0}\" y=\"{:.0}\" fill=\"#8a8a9a\" font-family=\"sans-serif\" font-size=\"13\">+a_x drive (up), +a_y left (right); rings = 1 g</text>",
        12.0,
        IMG_H - 14.0
    )?;
    // legend
    for (i, s) in slices.iter().enumerate() {
        let color = PALETTE[i % PALETTE.len()];
        let y = 52.0 + i as f64 * 20.0;
        writeln!(
            svg,
            "  <rect x=\"12\" y=\"{:.0}\" width=\"14\" height=\"14\" fill=\"{color}\"/>",
            y - 12.0
        )?;
        writeln!(
            svg,
            "  <text x=\"32\" y=\"{y:.0}\" fill=\"#cfcfdf\" font-family=\"sans-serif\" font-size=\"14\">{}</text>",
            xml_escape(&s.label)
        )?;
    }

    writeln!(svg, "</svg>")?;
    std::fs::write(path, svg)?;
    Ok(())
}

/// Minimal XML text escaping for label text.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings_hash_for_mode;

    fn meta() -> RunMetadata {
        RunMetadata::new(
            settings_hash_for_mode("car"),
            settings_hash_for_mode("none"),
            settings_hash_for_mode("envelope"),
            None,
        )
    }

    #[test]
    fn renders_and_embeds_metadata() {
        let dir = std::env::temp_dir().join("apex_env_svg_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gg.svg");
        // a simple friction circle
        let boundary: Vec<(f64, f64)> = (0..=32)
            .map(|i| {
                let t = i as f64 / 32.0 * std::f64::consts::TAU;
                (15.0 * t.sin(), 15.0 * t.cos())
            })
            .collect();
        let slices = vec![EnvelopeSlicePlot {
            label: "v=30, g_z=9.81".to_string(),
            boundary,
        }];
        render_envelope_svg(&path, &meta(), &slices, "g-g diagram").unwrap();
        let s = std::fs::read_to_string(&path).unwrap();
        assert!(s.contains("<svg"));
        assert!(s.contains("<metadata>"));
        assert!(s.contains("<polygon"));
        assert!(s.contains("g-g diagram"));
    }

    #[test]
    fn empty_is_error() {
        let path = std::env::temp_dir().join("apex_env_empty.svg");
        assert!(render_envelope_svg(&path, &meta(), &[], "x").is_err());
    }
}
