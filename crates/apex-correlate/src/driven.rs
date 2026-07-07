//! Reconstruct the **driven line** from the measured lap and run the QSS sim on
//! it (task 2.3 → 2.4, QSS inference).
//!
//! The centerline QSS is confounded by the racing line: the driver runs wide to
//! open corners. Reconstructing the actual driven path and running the same car
//! on it removes that confound — whatever lap-time / speed gap remains is then
//! genuinely **car-model** error (grip, power, drag), the target of parameter
//! identification (2.4).
//!
//! Two reconstruction modes:
//!
//! - [`DrivenLineMode::Offset`] — offset the smoothed centerline by the measured
//!   signed offset `n(s)` (low-passed with a periodic moving average). Cheap, but
//!   at the very fastest corners it *under-opens* the line (offsetting a curved
//!   centerline by a nearly-constant `n` barely changes the radius).
//! - [`DrivenLineMode::Direct`] — build the driven path **directly** from the
//!   aligned measured `(x, y)` samples, smoothed with the shared
//!   [`apex_track::smoothing`] machinery (periodic, deviation-bounded). This
//!   reproduces the driver's actual radius at speed (e.g. Abbey), so it is the
//!   default.

use apex_physics::{qss_lap_sim, qss_lap_sim_3d, CarParams, DEFAULT_SECTOR_COUNT};
use apex_telemetry::ChannelId;
use apex_track::{build_track, smooth_points, Ribbon3d, Track, TrackPoint};

use crate::error::CorrelateError;
use crate::metrics::measured_sector_times;
use crate::report::SimTrace;
use crate::telemetry::Telemetry;

/// Default half-width (m) of the moving-average filter applied to `n(s)`
/// (offset mode). ±10 m kills sample-to-sample GPS jitter while preserving the
/// corner-scale line shape.
pub const DEFAULT_N_FILTER_WINDOW_M: f64 = 10.0;

/// Default deviation budget (m) for smoothing the measured `(x, y)` in direct
/// mode.
pub const DEFAULT_DRIVEN_SMOOTH_TOLERANCE_M: f64 = 0.75;

/// How the driven line is reconstructed.
#[derive(Debug, Clone, Copy)]
pub enum DrivenLineMode {
    /// Centerline offset by measured `n(s)`; `f64` = moving-average half-width (m).
    Offset(f64),
    /// Driven path built directly from smoothed measured `(x, y)`; `f64` =
    /// smoothing deviation budget (m).
    Direct(f64),
}

/// Result of a driven-line reconstruction + QSS run.
#[derive(Debug, Clone)]
pub struct DrivenResult {
    /// The sim trace (speed by centerline station) for the driven line.
    pub trace: SimTrace,
    /// Arc length of the reconstructed driven path (m).
    pub driven_length: f64,
    /// Centerline arc length (m), for comparison.
    pub centerline_length: f64,
    /// Human-readable summary of the reconstruction settings.
    pub detail: String,
}

/// The reconstructed driven-line **geometry** — car-independent, so it is built
/// once and reused across many QSS runs (e.g. every identify iteration).
pub struct DrivenGeometry {
    /// The driven path (arc length + curvature), a closed track.
    pub track: Track,
    /// The driven path as a **3D ribbon** (z assigned from the centerline's
    /// elevation at each projected station), present only when the pipeline is
    /// run with a 3D centerline. When `Some`, the QSS runs on it in 3D; when
    /// `None`, the flat `track` is used and behavior is unchanged.
    pub driven_ribbon: Option<Ribbon3d>,
    /// Continuous centerline station (m, monotone) of each driven segment, so a
    /// speed profile on this line can be mapped back to centerline coordinates.
    pub station_per_segment: Vec<f64>,
    /// Arc length of the driven path (m).
    pub driven_length: f64,
    /// Centerline arc length (m).
    pub centerline_length: f64,
    /// Human-readable reconstruction summary.
    pub detail: String,
    /// Trace label (`"measured line (direct)"` / `"… (offset)"`).
    pub label: String,
}

/// Reconstruct the driven line per `mode`, run the QSS `car` on it, and return a
/// [`SimTrace`] in centerline-station coordinates. `aligned` must carry the
/// projected `s` station, `lateral_offset` (offset mode), and `x`/`y` (direct
/// mode) channels.
pub fn driven_sim_trace(
    centerline: &Track,
    aligned: &Telemetry,
    car: &CarParams,
    mode: DrivenLineMode,
) -> Result<DrivenResult, CorrelateError> {
    driven_sim_trace_3d(centerline, aligned, car, mode, None)
}

/// [`driven_sim_trace`] with an optional 3D centerline `elevation`. When `Some`,
/// the driven line is elevated (z from the centerline's `elevation_at`) and the
/// QSS runs in 3D; when `None`, behavior is identical to [`driven_sim_trace`].
pub fn driven_sim_trace_3d(
    centerline: &Track,
    aligned: &Telemetry,
    car: &CarParams,
    mode: DrivenLineMode,
    elevation: Option<&Ribbon3d>,
) -> Result<DrivenResult, CorrelateError> {
    let geom = build_driven_geometry_3d(centerline, aligned, mode, elevation)?;
    let sim = run_qss_on_driven(&geom, car);
    let trace = geometry_to_trace(&geom, &sim);
    Ok(DrivenResult {
        trace,
        driven_length: geom.driven_length,
        centerline_length: geom.centerline_length,
        detail: geom.detail,
    })
}

/// Run the QSS on a driven geometry: 3D when the driven ribbon is present
/// (elevated centerline), else the flat 2D QSS on the driven track.
pub fn run_qss_on_driven(geom: &DrivenGeometry, car: &CarParams) -> apex_physics::QssResult {
    match &geom.driven_ribbon {
        Some(ribbon) => qss_lap_sim_3d(ribbon, car),
        None => qss_lap_sim(&geom.track, car),
    }
}

/// Build the car-independent driven-line geometry for `mode` (2D).
pub fn build_driven_geometry(
    centerline: &Track,
    aligned: &Telemetry,
    mode: DrivenLineMode,
) -> Result<DrivenGeometry, CorrelateError> {
    build_driven_geometry_3d(centerline, aligned, mode, None)
}

/// Build the driven-line geometry, optionally assigning elevation from a 3D
/// centerline `elevation` ribbon (z at each projected station; the cross-track
/// elevation difference over a ~14 m track is sub-metre and unresolvable, so the
/// driven line's elevation is taken as the centerline's at the same station).
pub fn build_driven_geometry_3d(
    centerline: &Track,
    aligned: &Telemetry,
    mode: DrivenLineMode,
    elevation: Option<&Ribbon3d>,
) -> Result<DrivenGeometry, CorrelateError> {
    let mut geom = match mode {
        DrivenLineMode::Offset(window) => geom_offset(centerline, aligned, window),
        DrivenLineMode::Direct(tol) => geom_direct(centerline, aligned, tol),
    }?;
    if let Some(elev) = elevation {
        geom.driven_ribbon = Some(elevate_driven(&geom, elev));
    }
    Ok(geom)
}

/// Build the 3D driven ribbon: the driven `(x, y)` path with `z` sampled from
/// the centerline `elevation` at each segment's continuous station.
fn elevate_driven(geom: &DrivenGeometry, elevation: &Ribbon3d) -> Ribbon3d {
    let l = elevation.total_length;
    let pts: Vec<[f64; 3]> = geom
        .track
        .segments
        .iter()
        .zip(&geom.station_per_segment)
        .map(|(seg, &st)| [seg.x, seg.y, elevation.elevation_at(st.rem_euclid(l))])
        .collect();
    let bank = vec![0.0; pts.len()];
    let wl: Vec<f64> = geom.track.segments.iter().map(|s| s.width_left).collect();
    let wr: Vec<f64> = geom.track.segments.iter().map(|s| s.width_right).collect();
    Ribbon3d::from_centerline_3d(
        &geom.track.name,
        &pts,
        &bank,
        &wl,
        &wr,
        geom.track.is_closed,
    )
}

/// Convert a QSS result on a [`DrivenGeometry`] into a centerline-station
/// [`SimTrace`] (station sorted into `[0, L)`, plus equal-thirds sectors).
pub fn geometry_to_trace(geom: &DrivenGeometry, sim: &apex_physics::QssResult) -> SimTrace {
    let l = geom.centerline_length;
    let n = geom.track.segments.len();

    // Station (mod L), sorted + deduped, carrying speed — a strictly increasing
    // trace station for periodic resampling.
    let mut pairs: Vec<(f64, f64)> = (0..n)
        .map(|i| (geom.station_per_segment[i].rem_euclid(l), sim.speeds[i]))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut station = Vec::with_capacity(n);
    let mut speed = Vec::with_capacity(n);
    for (st, sp) in pairs {
        if station.last().map(|&p: &f64| st - p > 1e-6).unwrap_or(true) {
            station.push(st);
            speed.push(sp);
        }
    }

    // Sectors by centerline-station thirds (driven interval times bucketed by
    // the continuous station midpoint).
    let mut interval_dt = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        let ds_driven = if i + 1 < n {
            geom.track.segments[j].s - geom.track.segments[i].s
        } else {
            geom.track.total_length - geom.track.segments[i].s
        };
        let v_avg = 0.5 * (sim.speeds[i] + sim.speeds[j]);
        interval_dt.push(if v_avg > 0.0 { ds_driven / v_avg } else { 0.0 });
    }
    let mut station_closed = geom.station_per_segment.clone();
    station_closed.push(geom.station_per_segment[0] + l);
    let sector_times =
        measured_sector_times(&station_closed, &interval_dt, l, DEFAULT_SECTOR_COUNT);

    SimTrace {
        station,
        speed,
        lap_time: sim.lap_time,
        sector_times,
        label: geom.label.clone(),
    }
}

/// Offset-mode geometry: centerline + filtered `n(s)` along the left normal.
fn geom_offset(
    centerline: &Track,
    aligned: &Telemetry,
    n_filter_window_m: f64,
) -> Result<DrivenGeometry, CorrelateError> {
    let meas_s = aligned
        .channel(ChannelId::S)
        .ok_or(CorrelateError::MissingAxis("s"))?;
    let meas_n = aligned
        .channel(ChannelId::LateralOffset)
        .ok_or(CorrelateError::MissingAxis("lateral_offset"))?;
    if meas_s.len() < 3 {
        return Err(CorrelateError::AlignFailed("too few measured samples"));
    }
    let l = centerline.total_length;
    let n_seg = centerline.segments.len();

    let s0 = meas_s[0];
    let mut n_station: Vec<f64> = Vec::with_capacity(n_seg);
    for seg in &centerline.segments {
        let u = if seg.s >= s0 { seg.s } else { seg.s + l };
        n_station.push(interp_clamped(meas_s, meas_n, u));
    }
    let spacing = l / n_seg as f64;
    let half = ((n_filter_window_m / spacing).round() as usize).max(0);
    let n_filt = moving_average_periodic(&n_station, half);
    let n_peak = n_filt.iter().cloned().fold(0.0_f64, |m, v| m.max(v.abs()));

    let driven_pts: Vec<TrackPoint> = centerline
        .segments
        .iter()
        .zip(&n_filt)
        .map(|(seg, &nv)| {
            let (nx, ny) = (
                (seg.heading + std::f64::consts::FRAC_PI_2).cos(),
                (seg.heading + std::f64::consts::FRAC_PI_2).sin(),
            );
            TrackPoint {
                x: seg.x + nv * nx,
                y: seg.y + nv * ny,
                width_left: seg.width_left,
                width_right: seg.width_right,
            }
        })
        .collect();
    let track = build_track(&format!("{} (driven)", centerline.name), &driven_pts, true);
    let station_per_segment: Vec<f64> = centerline.segments.iter().map(|s| s.s).collect();

    Ok(DrivenGeometry {
        driven_length: track.total_length,
        centerline_length: l,
        detail: format!("offset mode, n filter ±{n_filter_window_m:.0} m, peak |n| {n_peak:.2} m"),
        label: "measured line (offset)".to_string(),
        driven_ribbon: None,
        track,
        station_per_segment,
    })
}

/// Direct-mode geometry: driven path built from smoothed measured `(x, y)`.
fn geom_direct(
    centerline: &Track,
    aligned: &Telemetry,
    smooth_tolerance_m: f64,
) -> Result<DrivenGeometry, CorrelateError> {
    let meas_s = aligned
        .channel(ChannelId::S)
        .ok_or(CorrelateError::MissingAxis("s"))?;
    let x = aligned
        .channel(ChannelId::X)
        .ok_or(CorrelateError::MissingAxis("x"))?;
    let y = aligned
        .channel(ChannelId::Y)
        .ok_or(CorrelateError::MissingAxis("y"))?;
    if meas_s.len() < 5 {
        return Err(CorrelateError::AlignFailed("too few measured samples"));
    }
    let l = centerline.total_length;

    let mut mx = Vec::with_capacity(meas_s.len());
    let mut my = Vec::with_capacity(meas_s.len());
    let mut mst = Vec::with_capacity(meas_s.len());
    for i in 0..meas_s.len() {
        if x[i].is_finite() && y[i].is_finite() && meas_s[i].is_finite() {
            mx.push(x[i]);
            my.push(y[i]);
            mst.push(meas_s[i]);
        }
    }
    if mx.len() < 5 {
        return Err(CorrelateError::AlignFailed(
            "too few finite measured samples",
        ));
    }

    // Trim to exactly ONE lap: the measured lap can overrun start/finish by a
    // few metres; left untrimmed, closing the loop folds that overlap back on
    // itself and manufactures a curvature kink at the seam.
    let s_end = mst[0] + l;
    let keep = mst.iter().position(|&s| s >= s_end).unwrap_or(mst.len());
    if keep >= 5 {
        mx.truncate(keep);
        my.truncate(keep);
        mst.truncate(keep);
    }

    // FastF1 samples are TIME-uniform (dense in slow corners, sparse on
    // straights); `build_track`'s 3-point curvature needs ~uniform arc-length
    // spacing, so resample onto a uniform arc-length grid first.
    let target_spacing = (l / centerline.segments.len() as f64).max(1.0);
    let (ux, uy, stations_cont) = resample_xy_uniform(&mx, &my, &mst, target_spacing);
    let pts: Vec<TrackPoint> = ux
        .iter()
        .zip(&uy)
        .map(|(&px, &py)| TrackPoint {
            x: px,
            y: py,
            width_left: 5.0, // widths are unused by QSS; placeholder
            width_right: 5.0,
        })
        .collect();

    let (smoothed, _lambda, max_dev) = smooth_points(&pts, true, smooth_tolerance_m);
    let track = build_track(
        &format!("{} (driven-direct)", centerline.name),
        &smoothed,
        true,
    );

    Ok(DrivenGeometry {
        driven_length: track.total_length,
        centerline_length: l,
        detail: format!(
            "direct mode, smooth tol {smooth_tolerance_m:.2} m, max deviation {max_dev:.3} m"
        ),
        label: "measured line (direct)".to_string(),
        driven_ribbon: None,
        track,
        station_per_segment: stations_cont,
    })
}

/// Resample a closed measured trace `(mx, my)` with per-sample centerline
/// station `mst` onto a uniform **arc-length** grid of spacing ≈ `spacing`.
/// Returns `(x, y, station)` at the uniform points. The loop is closed (last
/// sample → first) so the driven track has no seam gap.
fn resample_xy_uniform(
    mx: &[f64],
    my: &[f64],
    mst: &[f64],
    spacing: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = mx.len();
    let dist = |i: usize, j: usize| ((mx[i] - mx[j]).powi(2) + (my[i] - my[j]).powi(2)).sqrt();

    // Cumulative chord arc length, plus a wrap point closing back to sample 0.
    let mut arc = Vec::with_capacity(n + 1);
    arc.push(0.0);
    for i in 1..n {
        arc.push(arc[i - 1] + dist(i - 1, i));
    }
    let closing = dist(n - 1, 0);
    let total = arc[n - 1] + closing;

    // Extended arrays with the wrap vertex (position 0 again). Station continues
    // monotonically by the closing chord length so `station` stays increasing.
    let mut ax = mx.to_vec();
    ax.push(mx[0]);
    let mut ay = my.to_vec();
    ay.push(my[0]);
    let mut ast = mst.to_vec();
    ast.push(mst[n - 1] + closing);
    let mut aarc = arc.clone();
    aarc.push(total);

    let m = ((total / spacing).round() as usize).max(4);
    let (mut ux, mut uy, mut ust) = (
        Vec::with_capacity(m),
        Vec::with_capacity(m),
        Vec::with_capacity(m),
    );
    let mut seg = 0usize;
    for k in 0..m {
        let a = k as f64 * total / m as f64;
        while seg + 1 < aarc.len() && aarc[seg + 1] < a {
            seg += 1;
        }
        let (a0, a1) = (aarc[seg], aarc[seg + 1]);
        let t = if a1 > a0 { (a - a0) / (a1 - a0) } else { 0.0 };
        ux.push(ax[seg] + t * (ax[seg + 1] - ax[seg]));
        uy.push(ay[seg] + t * (ay[seg + 1] - ay[seg]));
        ust.push(ast[seg] + t * (ast[seg + 1] - ast[seg]));
    }
    (ux, uy, ust)
}

/// Linear interpolation with endpoint clamping; `xs` strictly increasing.
fn interp_clamped(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    let (mut lo, mut hi) = (0usize, n - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (x - xs[lo]) / (xs[hi] - xs[lo]);
    ys[lo] + t * (ys[hi] - ys[lo])
}

/// Centered periodic moving average with the given half-width (in samples).
fn moving_average_periodic(v: &[f64], half: usize) -> Vec<f64> {
    let n = v.len();
    if half == 0 || n == 0 {
        return v.to_vec();
    }
    let win = 2 * half + 1;
    let mut out = vec![0.0; n];
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for k in 0..win {
            let idx = (i + n + k - half) % n;
            acc += v[idx];
        }
        *o = acc / win as f64;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GridKind;
    use apex_track::oval_track;
    use std::collections::BTreeMap;
    use std::f64::consts::PI;

    fn oval() -> Track {
        let (pts, closed) = oval_track(500.0, 120.0, 12.0, 400);
        build_track("Oval", &pts, closed)
    }

    /// Aligned telemetry running a constant offset, with x/y on the offset line.
    fn aligned_offset(track: &Track, n_val: f64) -> Telemetry {
        let l = track.total_length;
        let m = 400;
        let (mut s, mut nn, mut xs, mut ys) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for k in 0..m {
            let st = k as f64 / (m - 1) as f64 * (l - 1.0);
            let (i, _) = track.locate(st);
            let seg = &track.segments[i];
            let (nx, ny) = (
                (seg.heading + PI / 2.0).cos(),
                (seg.heading + PI / 2.0).sin(),
            );
            s.push(st);
            nn.push(n_val);
            xs.push(seg.x + n_val * nx);
            ys.push(seg.y + n_val * ny);
        }
        let mut channels: BTreeMap<ChannelId, Vec<f64>> = BTreeMap::new();
        channels.insert(ChannelId::S, s);
        channels.insert(ChannelId::LateralOffset, nn);
        channels.insert(ChannelId::X, xs);
        channels.insert(ChannelId::Y, ys);
        Telemetry {
            grid: GridKind::S,
            channels,
            metadata: Vec::new(),
        }
    }

    #[test]
    fn offset_mode_runs() {
        let track = oval();
        let out = driven_sim_trace(
            &track,
            &aligned_offset(&track, 3.0),
            &CarParams::default(),
            DrivenLineMode::Offset(10.0),
        )
        .unwrap();
        assert_eq!(out.trace.station.len(), track.segments.len());
        assert_eq!(out.trace.sector_times.len(), 3);
        assert!(out.trace.lap_time > 0.0);
        assert!(out.detail.contains("offset"));
    }

    #[test]
    fn direct_mode_reproduces_offset_path_length() {
        // With x/y sitting exactly on a constant-offset line, the direct
        // reconstruction's driven length should match the offset line's length
        // (both trace the same physical loop).
        let track = oval();
        let al = aligned_offset(&track, 3.0);
        let car = CarParams::default();
        let direct = driven_sim_trace(&track, &al, &car, DrivenLineMode::Direct(0.75)).unwrap();
        let offset = driven_sim_trace(&track, &al, &car, DrivenLineMode::Offset(2.0)).unwrap();
        assert!(direct.detail.contains("direct"));
        assert_eq!(direct.trace.sector_times.len(), 3);
        assert!(direct.trace.lap_time > 0.0);
        // Same physical line ⇒ lengths agree within a few metres.
        assert!(
            (direct.driven_length - offset.driven_length).abs() < 6.0,
            "direct {} vs offset {}",
            direct.driven_length,
            offset.driven_length
        );
        // Trace station strictly increasing (periodic-resample precondition).
        for w in direct.trace.station.windows(2) {
            assert!(
                w[1] > w[0],
                "station not increasing: {} then {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn moving_average_smooths_periodic() {
        let mut v = vec![0.0; 100];
        v[50] = 10.0;
        let f = moving_average_periodic(&v, 3);
        assert!(f[50] < 2.0, "spike not attenuated: {}", f[50]);
        let mean_in: f64 = v.iter().sum::<f64>() / v.len() as f64;
        let mean_out: f64 = f.iter().sum::<f64>() / f.len() as f64;
        assert!((mean_in - mean_out).abs() < 1e-9);
    }
}
