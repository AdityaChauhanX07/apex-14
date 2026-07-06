//! Correlation report generation: runs the metrics against a measured lap and a
//! QSS sim, and renders SVG overlays + a deterministic markdown report.
//!
//! Sector splits use the workspace's **equal-arc-length thirds** convention
//! (`apex_physics::sector_times`), which is **NOT** the official F1 sector
//! layout — every output labels this.
//!
//! # Provenance on outputs (hybrid)
//!
//! Sim-derived artifacts carry a full [`RunMetadata`] provenance block (car /
//! track / settings / config hashes). The correlation overlays additionally
//! carry **descriptive measured-source** lines (session identifiers from the
//! measured file header) as an XML comment — a deliberate hybrid: the sim side
//! is reproducible-by-hash, the measured side is descriptive only (no
//! `RunMetadata` hashes for measured data). This is documented in
//! `docs/telemetry_format.md`.

use std::fmt::Write as _;
use std::path::Path;

use apex_physics::QssResult;
use apex_telemetry::{ChannelId, RunMetadata};
use apex_track::Track;

use crate::error::CorrelateError;
use crate::metrics::{
    apex_errors, braking_offsets, detect_corners, measured_sector_times, resample_linear,
    resample_periodic, speed_rmse, ApexError, BrakingOffset, CornerConfig, LapDelta,
    SectorComparison, SpeedRmse,
};
use crate::telemetry::Telemetry;

/// Configuration for a correlation run.
#[derive(Debug, Clone, Copy)]
pub struct CorrelationConfig {
    /// Uniform arc-length grid step for resampling (m).
    pub grid_step: f64,
    /// Corner-detection tuning.
    pub corner: CornerConfig,
    /// Braking-onset deceleration threshold (m/s²).
    pub decel_threshold: f64,
    /// Warn if the measured lap time (t span) and the header lap-time comment
    /// disagree by more than this (s).
    pub lap_time_tol: f64,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        CorrelationConfig {
            grid_step: 10.0,
            corner: CornerConfig::default(),
            decel_threshold: 2.0,
            lap_time_tol: 0.05,
        }
    }
}

/// The full computed correlation between a measured lap and a QSS sim.
#[derive(Debug, Clone)]
pub struct CorrelationResult {
    /// Common resampling grid (arc length, m).
    pub grid_s: Vec<f64>,
    /// Sim speed on the grid (m/s).
    pub sim_v: Vec<f64>,
    /// Measured speed on the grid (m/s).
    pub meas_v: Vec<f64>,
    /// Lap-time delta.
    pub lap: LapDelta,
    /// Per-sector comparison (equal-arc thirds).
    pub sectors: SectorComparison,
    /// Speed-trace RMSE / max error.
    pub rmse: SpeedRmse,
    /// Detected corner apex indices into the grid.
    pub corners: Vec<usize>,
    /// Apex-speed errors, one per corner (grid order).
    pub apex: Vec<ApexError>,
    /// Braking-point offsets, one per corner (grid order).
    pub braking: Vec<BrakingOffset>,
    /// Measured lap time from the `t` channel span (s).
    pub measured_lap_from_t: f64,
    /// Lap-time from the measured header comment, if present (s).
    pub header_lap_time: Option<f64>,
    /// `|t-span − header|` when it exceeds the tolerance (s), else `None`.
    pub lap_time_mismatch: Option<f64>,
    /// Grid `s` where the sim carries the most extra speed vs. measured (m).
    pub sim_fastest_s: f64,
    /// `sim − measured` there (m/s, ≥ 0 side).
    pub sim_fastest_dv: f64,
    /// Grid `s` where the sim is most below measured (m).
    pub sim_slowest_s: f64,
    /// `sim − measured` there (m/s, ≤ 0 side).
    pub sim_slowest_dv: f64,
    /// Total grid span used (m).
    pub span: f64,
    /// Corner-detection speed ceiling used (m/s), for reporting.
    pub corner_ceiling: f64,
    /// Measured-source metadata (descriptive), in file order.
    pub measured_meta: Vec<(String, String)>,
}

/// Fetch a measured-metadata value by key.
fn meta_get<'a>(meta: &'a [(String, String)], key: &str) -> Option<&'a str> {
    meta.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

/// Run the full correlation. `measured` must carry `s` (geometric station),
/// `speed`, and `t`; `sim` is a QSS result on `track`.
pub fn correlate(
    measured: &Telemetry,
    track: &Track,
    sim: &QssResult,
    config: CorrelationConfig,
) -> Result<CorrelationResult, CorrelateError> {
    let meas_s = measured
        .channel(ChannelId::S)
        .ok_or(CorrelateError::MissingAxis("s"))?;
    let meas_v_raw = measured
        .channel(ChannelId::Speed)
        .ok_or(CorrelateError::MissingAxis("speed"))?;
    let meas_t = measured
        .channel(ChannelId::Time)
        .ok_or(CorrelateError::MissingAxis("t"))?;
    if meas_s.len() < 3 {
        return Err(CorrelateError::AlignFailed("too few measured samples"));
    }

    // Common uniform grid in continuous measured s.
    let step = config.grid_step;
    let lo = (meas_s[0] / step).ceil() * step;
    let hi = (meas_s[meas_s.len() - 1] / step).floor() * step;
    let mut grid_s = Vec::new();
    let mut g = lo;
    while g <= hi + 1e-9 {
        grid_s.push(g);
        g += step;
    }

    let meas_v = resample_linear(meas_s, meas_v_raw, &grid_s);
    // Sim speed is periodic in centerline station; grid mod L selects it.
    let sim_v = resample_periodic(&sim.distances, &sim.speeds, track.total_length, &grid_s);

    // Lap time: measured from t span (authoritative), sim from the QSS integral.
    let measured_lap_from_t = meas_t[meas_t.len() - 1] - meas_t[0];
    let lap = LapDelta::new(sim.lap_time, measured_lap_from_t);

    // Header lap-time cross-check.
    let header_lap_time = meta_get(&measured.metadata, "lap_time_s")
        .or_else(|| meta_get(&measured.metadata, "lap_time"))
        .and_then(|v| v.trim().parse::<f64>().ok());
    let lap_time_mismatch = header_lap_time.and_then(|h| {
        let d = (measured_lap_from_t - h).abs();
        (d > config.lap_time_tol).then_some(d)
    });

    // Sectors (equal-arc thirds).
    let n_sectors = apex_physics::DEFAULT_SECTOR_COUNT;
    let dt: Vec<f64> = meas_t.windows(2).map(|w| w[1] - w[0]).collect();
    let meas_sectors = measured_sector_times(meas_s, &dt, track.total_length, n_sectors);
    let sectors = SectorComparison::new(sim.sector_times.clone(), meas_sectors);

    // Speed error + corners.
    let rmse = speed_rmse(&grid_s, &sim_v, &meas_v);
    let corners = detect_corners(&grid_s, &meas_v, config.corner);
    let apex = apex_errors(&grid_s, &sim_v, &meas_v, &corners);
    let braking = braking_offsets(&grid_s, &sim_v, &meas_v, &corners, config.decel_threshold);

    // Where does the sim most over/under-carry speed?
    let (mut fmax, mut fmin) = (f64::NEG_INFINITY, f64::INFINITY);
    let (mut fs, mut ss) = (
        grid_s.first().copied().unwrap_or(0.0),
        grid_s.first().copied().unwrap_or(0.0),
    );
    for i in 0..grid_s.len() {
        if !sim_v[i].is_finite() || !meas_v[i].is_finite() {
            continue;
        }
        let dv = sim_v[i] - meas_v[i];
        if dv > fmax {
            fmax = dv;
            fs = grid_s[i];
        }
        if dv < fmin {
            fmin = dv;
            ss = grid_s[i];
        }
    }

    let span = if grid_s.len() >= 2 {
        grid_s[grid_s.len() - 1] - grid_s[0]
    } else {
        0.0
    };

    Ok(CorrelationResult {
        grid_s,
        sim_v,
        meas_v,
        lap,
        sectors,
        rmse,
        corners,
        apex,
        braking,
        measured_lap_from_t,
        header_lap_time,
        lap_time_mismatch,
        sim_fastest_s: fs,
        sim_fastest_dv: fmax,
        sim_slowest_s: ss,
        sim_slowest_dv: fmin,
        span,
        corner_ceiling: config.corner.ceiling,
        measured_meta: measured.metadata.clone(),
    })
}

impl CorrelationResult {
    /// The one-line headline, e.g.
    /// `Silverstone 2024 Q (RUS): lap delta +1.234 s, speed RMSE 3.21 m/s over 5890 m`.
    pub fn headline(&self, track_name: &str) -> String {
        let year = meta_get(&self.measured_meta, "year").unwrap_or("?");
        let session = meta_get(&self.measured_meta, "session").unwrap_or("?");
        let driver = meta_get(&self.measured_meta, "driver").unwrap_or("?");
        format!(
            "{track_name} {year} {session} ({driver}): lap delta {:+.3} s, speed RMSE {:.2} m/s over {:.0} m",
            self.lap.delta, self.rmse.rmse, self.span
        )
    }
}

// ---------------------------------------------------------------------------
// SVG rendering
// ---------------------------------------------------------------------------

const SVG_W: f64 = 1200.0;

/// A single stacked line-plot panel's data.
struct Panel<'a> {
    title: &'a str,
    y_label: &'a str,
    y_min: f64,
    y_max: f64,
    series: Vec<(&'a str, &'a str, &'a [f64])>, // (label, color, y-values on grid)
    height: f64,
}

/// Map data → pixel and emit a stacked set of panels sharing the x (arc-length)
/// axis, with optional vertical corner markers. Returns the inner SVG body.
fn render_panels(grid_s: &[f64], panels: &[Panel], markers: &[f64]) -> String {
    let margin_l = 70.0;
    let margin_r = 30.0;
    let margin_top = 34.0;
    let panel_gap = 46.0;
    let plot_w = SVG_W - margin_l - margin_r;
    let x0 = grid_s.first().copied().unwrap_or(0.0);
    let x1 = grid_s.last().copied().unwrap_or(1.0);
    let xr = (x1 - x0).max(1e-6);
    let sx = |x: f64| margin_l + (x - x0) / xr * plot_w;

    let mut body = String::new();
    let mut y_cursor = margin_top;
    for panel in panels {
        let top = y_cursor;
        let bot = top + panel.height;
        let yr = (panel.y_max - panel.y_min).max(1e-6);
        let sy = |v: f64| bot - (v - panel.y_min) / yr * panel.height;

        // panel frame
        let _ = writeln!(
            body,
            "  <rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"#12121c\" stroke=\"#333\" stroke-width=\"1\"/>",
            margin_l, top, plot_w, panel.height
        );
        // title + y label
        let _ = writeln!(
            body,
            "  <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"18\" fill=\"#ddd\">{}</text>",
            margin_l, top - 8.0, panel.title
        );
        let _ = writeln!(
            body,
            "  <text x=\"18\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"13\" fill=\"#999\" transform=\"rotate(-90 18 {:.1})\">{}</text>",
            (top + bot) / 2.0, (top + bot) / 2.0, panel.y_label
        );
        // y ticks (min / max)
        for (val, anchor) in [(panel.y_max, "hanging"), (panel.y_min, "auto")] {
            let _ = writeln!(
                body,
                "  <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#888\" text-anchor=\"end\" dominant-baseline=\"{}\">{:.0}</text>",
                margin_l - 6.0,
                sy(val),
                anchor,
                val
            );
        }

        // corner markers
        for &m in markers {
            if m >= x0 && m <= x1 {
                let _ = writeln!(
                    body,
                    "  <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#ff5577\" stroke-width=\"1\" stroke-dasharray=\"4 4\" opacity=\"0.6\"/>",
                    sx(m), top, sx(m), bot
                );
            }
        }

        // series polylines
        for (label, color, ys) in &panel.series {
            body.push_str("  <polyline fill=\"none\" stroke=\"");
            body.push_str(color);
            body.push_str("\" stroke-width=\"2\" points=\"");
            let mut pen_down = false;
            for (i, &yv) in ys.iter().enumerate() {
                if i >= grid_s.len() || !yv.is_finite() {
                    pen_down = false;
                    continue;
                }
                if pen_down {
                    let _ = write!(body, " ");
                }
                let _ = write!(body, "{:.1},{:.1}", sx(grid_s[i]), sy(yv));
                pen_down = true;
            }
            body.push_str("\"/>\n");
            let _ = label; // labels are rendered in the legend below
        }

        // legend
        let mut lx = margin_l + 8.0;
        for (label, color, _) in &panel.series {
            let _ = writeln!(
                body,
                "  <rect x=\"{:.1}\" y=\"{:.1}\" width=\"14\" height=\"4\" fill=\"{}\"/>",
                lx,
                top + 10.0,
                color
            );
            let _ = writeln!(
                body,
                "  <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"13\" fill=\"#ccc\">{}</text>",
                lx + 18.0, top + 14.0, label
            );
            lx += 18.0 + 8.0 * label.len() as f64 + 24.0;
        }

        y_cursor = bot + panel_gap;
    }

    // x axis label + ticks on the last panel bottom
    let axis_y = y_cursor - panel_gap + 22.0;
    for k in 0..=6 {
        let x = x0 + (x1 - x0) * k as f64 / 6.0;
        let _ = writeln!(
            body,
            "  <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#888\" text-anchor=\"middle\">{:.0}</text>",
            sx(x), axis_y, x
        );
    }
    let _ = writeln!(
        body,
        "  <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"14\" fill=\"#aaa\" text-anchor=\"middle\">Distance s [m]</text>",
        margin_l + plot_w / 2.0,
        axis_y + 22.0
    );
    body
}

/// Wrap an SVG body with the root element, a [`RunMetadata`] provenance block,
/// and descriptive measured-source XML comments (the documented hybrid).
fn wrap_svg(
    body: &str,
    height: f64,
    meta: &RunMetadata,
    measured_meta: &[(String, String)],
) -> String {
    let mut svg = String::new();
    let _ = writeln!(svg, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
    let _ = writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">",
        SVG_W as i32, height as i32, SVG_W as i32, height as i32
    );
    // sim-side reproducible provenance
    svg.push_str(&meta.svg_metadata_element());
    // measured-side descriptive provenance (comment, NOT hashes)
    svg.push_str("  <!-- measured-source (descriptive, not a RunMetadata hash):\n");
    for (k, v) in measured_meta {
        // keep comments safe: no "--" sequences
        let safe = v.replace("--", "—");
        let _ = writeln!(svg, "       {k}: {safe}");
    }
    svg.push_str("  -->\n");
    let _ = writeln!(
        svg,
        "  <rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#0d0d16\"/>",
        SVG_W as i32, height as i32
    );
    svg.push_str(body);
    let _ = writeln!(svg, "</svg>");
    svg
}

/// Write the speed-overlay SVG (sim vs measured speed, corner markers).
pub fn write_speed_overlay_svg(
    path: &Path,
    result: &CorrelationResult,
    meta: &RunMetadata,
    title: &str,
) -> Result<(), CorrelateError> {
    let vmax = result
        .sim_v
        .iter()
        .chain(&result.meas_v)
        .cloned()
        .filter(|x| x.is_finite())
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1.0);
    let markers: Vec<f64> = result.corners.iter().map(|&k| result.grid_s[k]).collect();
    let panels = vec![Panel {
        title,
        y_label: "Speed [m/s]",
        y_min: 0.0,
        y_max: (vmax / 10.0).ceil() * 10.0,
        series: vec![
            ("measured (RUS)", "#4fc3f7", &result.meas_v),
            ("QSS sim", "#ffb74d", &result.sim_v),
        ],
        height: 360.0,
    }];
    let body = render_panels(&result.grid_s, &panels, &markers);
    let svg = wrap_svg(&body, 470.0, meta, &result.measured_meta);
    std::fs::write(path, svg)?;
    Ok(())
}

/// Write the driver-inputs panel SVG (throttle/brake, gear, + sim speed ref).
pub fn write_inputs_panel_svg(
    path: &Path,
    result: &CorrelationResult,
    measured: &Telemetry,
    meta: &RunMetadata,
) -> Result<(), CorrelateError> {
    // Resample the input channels onto the same grid.
    let meas_s = measured.channel(ChannelId::S).unwrap();
    let get = |id: ChannelId| -> Vec<f64> {
        match measured.channel(id) {
            Some(c) => resample_linear(meas_s, c, &result.grid_s),
            None => vec![f64::NAN; result.grid_s.len()],
        }
    };
    let throttle = get(ChannelId::Throttle);
    let brake = get(ChannelId::Brake);
    let gear = get(ChannelId::Gear);
    let markers: Vec<f64> = result.corners.iter().map(|&k| result.grid_s[k]).collect();

    let gmax = gear
        .iter()
        .cloned()
        .filter(|x| x.is_finite())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let vmax = result
        .sim_v
        .iter()
        .cloned()
        .filter(|x| x.is_finite())
        .fold(1.0, f64::max);
    let panels = vec![
        Panel {
            title: "Driver inputs (measured)",
            y_label: "0–1",
            y_min: 0.0,
            y_max: 1.05,
            series: vec![
                ("throttle", "#66bb6a", &throttle),
                ("brake", "#ef5350", &brake),
            ],
            height: 180.0,
        },
        Panel {
            title: "Gear (measured)",
            y_label: "gear",
            y_min: 0.0,
            y_max: (gmax + 1.0).ceil(),
            series: vec![("gear", "#ba68c8", &gear)],
            height: 150.0,
        },
        Panel {
            title: "QSS sim speed (reference)",
            y_label: "Speed [m/s]",
            y_min: 0.0,
            y_max: (vmax / 10.0).ceil() * 10.0,
            series: vec![("QSS sim", "#ffb74d", &result.sim_v)],
            height: 180.0,
        },
    ];
    let body = render_panels(&result.grid_s, &panels, &markers);
    let svg = wrap_svg(&body, 660.0, meta, &result.measured_meta);
    std::fs::write(path, svg)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Markdown report
// ---------------------------------------------------------------------------

/// Render the markdown report body (deterministic; `APEX_REPRO_TIMESTAMP` is
/// honored through `meta.timestamp`).
pub fn render_markdown(
    result: &CorrelationResult,
    meta: &RunMetadata,
    track: &Track,
    speed_svg: &str,
    inputs_svg: &str,
) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# Telemetry correlation: {}", track.name);
    let _ = writeln!(md);
    let _ = writeln!(md, "**{}**", result.headline(&track.name));
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "> Sectors are **equal-arc-length thirds** (`apex_physics::sector_times`), \
         NOT official F1 sectors. This is the **unfitted** calibrated preset — no \
         parameter identification has been applied."
    );
    let _ = writeln!(md);

    // Provenance
    let _ = writeln!(md, "## Provenance");
    let _ = writeln!(md);
    let _ = writeln!(md, "**Measured source (descriptive):**");
    let _ = writeln!(md);
    for (k, v) in &result.measured_meta {
        let _ = writeln!(md, "- `{k}`: {v}");
    }
    let _ = writeln!(md);
    let _ = writeln!(md, "**Sim provenance (RunMetadata):**");
    let _ = writeln!(md);
    let _ = writeln!(md, "- `config_hash`: `{}`", meta.config_hash.to_hex());
    let _ = writeln!(md, "- `car_hash`: `{}`", meta.car_hash.to_hex());
    let _ = writeln!(md, "- `track_hash`: `{}`", meta.track_hash.to_hex());
    let _ = writeln!(md, "- `settings_hash`: `{}`", meta.settings_hash.to_hex());
    let _ = writeln!(
        md,
        "- `git_sha`: `{}`  `apex_version`: `{}`",
        meta.git_sha, meta.apex_version
    );
    let _ = writeln!(md, "- `timestamp`: `{}`", meta.timestamp);
    let _ = writeln!(md);

    // Lap time
    let _ = writeln!(md, "## Lap time");
    let _ = writeln!(md);
    let _ = writeln!(md, "| | Time (s) |");
    let _ = writeln!(md, "|---|---|");
    let _ = writeln!(
        md,
        "| Measured (from `t` span) | {:.3} |",
        result.measured_lap_from_t
    );
    if let Some(h) = result.header_lap_time {
        let _ = writeln!(md, "| Measured (header comment) | {h:.3} |");
    }
    let _ = writeln!(md, "| Sim (QSS) | {:.3} |", result.lap.sim);
    let _ = writeln!(
        md,
        "| **Delta (sim − measured)** | **{:+.3}** |",
        result.lap.delta
    );
    let _ = writeln!(md);
    if let Some(d) = result.lap_time_mismatch {
        let _ = writeln!(
            md,
            "> ⚠️ Header lap time and `t`-span disagree by {d:.3} s (> tolerance)."
        );
        let _ = writeln!(md);
    }

    // Sectors
    let _ = writeln!(md, "## Sectors (equal-arc thirds — not official F1)");
    let _ = writeln!(md);
    let _ = writeln!(md, "| Sector | Measured (s) | Sim (s) | Delta (s) |");
    let _ = writeln!(md, "|---|---|---|---|");
    for i in 0..result.sectors.delta.len() {
        let _ = writeln!(
            md,
            "| S{} | {:.3} | {:.3} | {:+.3} |",
            i + 1,
            result.sectors.measured[i],
            result.sectors.sim[i],
            result.sectors.delta[i]
        );
    }
    let _ = writeln!(md);

    // Speed trace
    let _ = writeln!(md, "## Speed trace");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "- **RMSE:** {:.3} m/s over {} grid points ({:.0} m at {:.0} m spacing)",
        result.rmse.rmse,
        result.rmse.n,
        result.span,
        result.grid_s.get(1).copied().unwrap_or(0.0)
            - result.grid_s.first().copied().unwrap_or(0.0)
    );
    let _ = writeln!(
        md,
        "- **Max |Δv|:** {:.2} m/s at s = {:.0} m",
        result.rmse.max_abs, result.rmse.s_at_max
    );
    let _ = writeln!(
        md,
        "- **Sim carries most extra speed:** {:+.2} m/s at s = {:.0} m",
        result.sim_fastest_dv, result.sim_fastest_s
    );
    let _ = writeln!(
        md,
        "- **Sim most below measured:** {:+.2} m/s at s = {:.0} m",
        result.sim_slowest_dv, result.sim_slowest_s
    );
    let _ = writeln!(md);
    let _ = writeln!(md, "![Speed overlay]({speed_svg})");
    let _ = writeln!(md);

    // Corners / apex
    let _ = writeln!(md, "## Corners & apex speeds");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "Detected **{}** corners (measured-speed minima below {:.0} m/s).",
        result.corners.len(),
        result.corner_ceiling
    );
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| s (m) | Measured apex (m/s) | Sim @ s (m/s) | Δ (sim−meas) |"
    );
    let _ = writeln!(md, "|---|---|---|---|");
    for a in &result.apex {
        let _ = writeln!(
            md,
            "| {:.0} | {:.2} | {:.2} | {:+.2} |",
            a.s, a.v_measured, a.v_sim, a.delta
        );
    }
    let _ = writeln!(md);

    // Braking
    let _ = writeln!(md, "## Braking-point offsets");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "Onset = longitudinal decel crossing the threshold on corner approach. \
         Offset = `s_sim − s_measured` (**positive ⇒ sim brakes later**)."
    );
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Corner s (m) | Measured onset (m) | Sim onset (m) | Offset (m) |"
    );
    let _ = writeln!(md, "|---|---|---|---|");
    for b in &result.braking {
        let _ = writeln!(
            md,
            "| {:.0} | {} | {} | {} |",
            b.corner_s,
            opt_m(b.s_measured),
            opt_m(b.s_sim),
            opt_m(b.offset)
        );
    }
    let _ = writeln!(md);

    let _ = writeln!(md, "## Driver inputs");
    let _ = writeln!(md);
    let _ = writeln!(md, "![Inputs panel]({inputs_svg})");
    let _ = writeln!(md);

    md
}

fn opt_m(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.0}"),
        None => "—".to_string(),
    }
}

/// Write `report.md` plus both SVGs into `out_dir`, returning the report path.
#[allow(clippy::too_many_arguments)]
pub fn write_report(
    out_dir: &Path,
    result: &CorrelationResult,
    measured: &Telemetry,
    meta: &RunMetadata,
    track: &Track,
) -> Result<std::path::PathBuf, CorrelateError> {
    std::fs::create_dir_all(out_dir)?;
    let speed_svg = "speed_overlay.svg";
    let inputs_svg = "inputs_panel.svg";
    write_speed_overlay_svg(
        &out_dir.join(speed_svg),
        result,
        meta,
        &format!("{} — sim vs measured speed", track.name),
    )?;
    write_inputs_panel_svg(&out_dir.join(inputs_svg), result, measured, meta)?;

    let md = render_markdown(result, meta, track, speed_svg, inputs_svg);
    let report_path = out_dir.join("report.md");
    std::fs::write(&report_path, md)?;
    Ok(report_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_physics::{qss_lap_sim, CarParams};
    use apex_telemetry::settings_hash_for_mode;
    use apex_track::{build_track, oval_track};
    use std::collections::BTreeMap;

    fn test_meta() -> RunMetadata {
        RunMetadata::new(
            settings_hash_for_mode("car"),
            settings_hash_for_mode("track"),
            settings_hash_for_mode("settings"),
            None,
        )
    }

    #[test]
    fn correlate_on_synthetic_oval() {
        // Build an oval, run QSS, and correlate the sim against ITSELF sampled
        // as "measured" — deltas should be ~0 and RMSE tiny.
        let (pts, closed) = oval_track(500.0, 90.0, 12.0, 400);
        let track = build_track("Oval", &pts, closed);
        let sim = qss_lap_sim(&track, &CarParams::default());

        // Synthesize a measured lap = the sim trace with a time channel.
        let n = sim.distances.len();
        let mut t = vec![0.0; n];
        for i in 1..n {
            let ds = sim.distances[i] - sim.distances[i - 1];
            let v = 0.5 * (sim.speeds[i] + sim.speeds[i - 1]);
            t[i] = t[i - 1] + if v > 0.0 { ds / v } else { 0.0 };
        }
        let mut channels: BTreeMap<ChannelId, Vec<f64>> = BTreeMap::new();
        channels.insert(ChannelId::S, sim.distances.clone());
        channels.insert(ChannelId::Speed, sim.speeds.clone());
        channels.insert(ChannelId::Time, t);
        let measured = Telemetry {
            grid: crate::GridKind::S,
            channels,
            metadata: vec![
                ("session".into(), "Q".into()),
                ("driver".into(), "SIM".into()),
                ("year".into(), "2024".into()),
            ],
        };

        let result = correlate(&measured, &track, &sim, CorrelationConfig::default()).unwrap();
        assert!(
            result.lap.delta.abs() < 0.2,
            "lap delta {}",
            result.lap.delta
        );
        assert!(result.rmse.rmse < 1.0, "rmse {}", result.rmse.rmse);
        assert!(result.span > 1000.0);
        let head = result.headline(&track.name);
        assert!(head.contains("Oval 2024 Q (SIM)"), "headline: {head}");

        // Markdown + SVGs render without panicking and contain key sections.
        let dir = std::env::temp_dir().join("apex_correlate_report_test");
        let path = write_report(&dir, &result, &measured, &test_meta(), &track).unwrap();
        let md = std::fs::read_to_string(&path).unwrap();
        assert!(md.contains("equal-arc"));
        assert!(md.contains("## Lap time"));
        let svg = std::fs::read_to_string(dir.join("speed_overlay.svg")).unwrap();
        assert!(svg.contains("<metadata>"));
        assert!(svg.contains("measured-source"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
