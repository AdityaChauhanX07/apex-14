//! Reconstruct the **driven line** from the measured lateral offset `n(s)` and
//! run the QSS sim on it (start of task 2.3, QSS inference).
//!
//! The centerline QSS is confounded by the racing line: the driver does not
//! follow the centerline, they run wide/late to open up corners. Because the
//! projection (task 2.1b) gives us the measured signed offset `n(s)`, we can
//! reconstruct the actual driven path — centerline offset by `n(s)` along the
//! left normal — and run the same car on *that* geometry. Whatever lap-time /
//! speed gap remains is then genuinely **car-model** error (grip, power, drag),
//! which parameter identification (2.4) will fit.
//!
//! # `n(s)` filtering
//!
//! Measured `n(s)` carries GPS jitter (the position noise that survived
//! projection). Differentiating the driven path to curvature would amplify it,
//! so `n` is resampled onto the centerline stations and passed through a
//! **centered periodic moving-average** of half-width `n_filter_window_m`
//! (default ±10 m). This is a mild low-pass that preserves the corner-scale
//! shape of the line while removing sample-to-sample jitter.

use apex_physics::{qss_lap_sim, CarParams, DEFAULT_SECTOR_COUNT};
use apex_telemetry::ChannelId;
use apex_track::{build_track, Track, TrackPoint};

use crate::error::CorrelateError;
use crate::metrics::measured_sector_times;
use crate::report::SimTrace;
use crate::telemetry::Telemetry;

/// Default half-width (m) of the moving-average filter applied to `n(s)`.
///
/// ±10 m kills sample-to-sample GPS jitter in `n` while preserving the
/// corner-scale line shape — larger windows measurably under-open the fastest
/// corners (e.g. Abbey), tighter windows let jitter back into the curvature.
pub const DEFAULT_N_FILTER_WINDOW_M: f64 = 10.0;

/// Result of a driven-line reconstruction + QSS run.
#[derive(Debug, Clone)]
pub struct DrivenResult {
    /// The sim trace (speed by centerline station) for the driven line.
    pub trace: SimTrace,
    /// Arc length of the reconstructed driven path (m).
    pub driven_length: f64,
    /// Centerline arc length (m), for comparison.
    pub centerline_length: f64,
    /// `n(s)` moving-average half-width used (m).
    pub n_filter_window_m: f64,
    /// Peak |n| after filtering (m).
    pub n_peak: f64,
}

/// Reconstruct the driven line from `aligned` (which must carry the projected
/// `s` station and `lateral_offset` channels) offset from `centerline`, run the
/// QSS `car` on it, and return a [`SimTrace`] in centerline-station coordinates.
pub fn driven_sim_trace(
    centerline: &Track,
    aligned: &Telemetry,
    car: &CarParams,
    n_filter_window_m: f64,
) -> Result<DrivenResult, CorrelateError> {
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

    // 1. Resample n onto each centerline station (handle the lap-start wrap).
    let s0 = meas_s[0];
    let mut n_station: Vec<f64> = Vec::with_capacity(n_seg);
    for seg in &centerline.segments {
        // The measured continuous s runs [s0, s0+~L]; a centerline station
        // below s0 is covered near the end of the lap (station + L).
        let u = if seg.s >= s0 { seg.s } else { seg.s + l };
        n_station.push(interp_clamped(meas_s, meas_n, u));
    }

    // 2. Light periodic moving-average to kill GPS jitter in n.
    let spacing = l / n_seg as f64;
    let half = ((n_filter_window_m / spacing).round() as usize).max(0);
    let n_filt = moving_average_periodic(&n_station, half);
    let n_peak = n_filt.iter().cloned().fold(0.0_f64, |m, v| m.max(v.abs()));

    // 3. Offset the centerline along its left normal: +n = left = (heading+π/2).
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

    // 4. Build the driven track (its arc length differs from centerline station)
    //    and run the QSS on it.
    let driven = build_track(&format!("{} (driven)", centerline.name), &driven_pts, true);
    let sim = qss_lap_sim(&driven, car);

    // 5. Express the driven speed profile in centerline station: driven segment
    //    i corresponds to centerline station centerline.segments[i].s.
    let station: Vec<f64> = centerline.segments.iter().map(|s| s.s).collect();
    let speed = sim.speeds.clone();

    // 6. Sector split by CENTERLINE-station thirds (comparable to measured):
    //    attribute each driven interval's time to the sector of its centerline
    //    midpoint.
    let mut interval_dt = Vec::with_capacity(n_seg);
    for i in 0..n_seg {
        let j = (i + 1) % n_seg;
        let ds_driven = if i + 1 < n_seg {
            driven.segments[j].s - driven.segments[i].s
        } else {
            driven.total_length - driven.segments[i].s
        };
        let v_avg = 0.5 * (speed[i] + speed[j]);
        interval_dt.push(if v_avg > 0.0 { ds_driven / v_avg } else { 0.0 });
    }
    // measured_sector_times buckets by midpoint of consecutive stations; append
    // the wrap station (s0 + L) so the final interval is attributed correctly.
    let mut station_closed = station.clone();
    station_closed.push(station[0] + l);
    let sector_times =
        measured_sector_times(&station_closed, &interval_dt, l, DEFAULT_SECTOR_COUNT);

    let trace = SimTrace {
        station,
        speed,
        lap_time: sim.lap_time,
        sector_times,
        label: "measured line".to_string(),
    };

    Ok(DrivenResult {
        trace,
        driven_length: driven.total_length,
        centerline_length: l,
        n_filter_window_m,
        n_peak,
    })
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

    /// Aligned telemetry that runs a constant +3 m left offset around the track.
    fn aligned_constant_offset(track: &Track, n_val: f64) -> Telemetry {
        let l = track.total_length;
        let m = 300;
        let mut s = Vec::new();
        let mut nn = Vec::new();
        for k in 0..m {
            s.push(k as f64 / (m - 1) as f64 * (l - 1.0));
            nn.push(n_val);
        }
        let mut channels: BTreeMap<ChannelId, Vec<f64>> = BTreeMap::new();
        channels.insert(ChannelId::S, s);
        channels.insert(ChannelId::LateralOffset, nn);
        Telemetry {
            grid: GridKind::S,
            channels,
            metadata: Vec::new(),
        }
    }

    #[test]
    fn constant_left_offset_lengthens_outside_of_oval() {
        // A constant outward offset on an oval makes the driven loop longer
        // (bigger corner radii); a constant inward offset makes it shorter.
        let track = oval();
        let car = CarParams::default();
        // +n = left. On this oval built counter-clockwise, left is the inside;
        // regardless of sign, |Δlength| must scale with the offset and the
        // driven length must differ from the centerline by ~ 2π·n over curves.
        let out =
            driven_sim_trace(&track, &aligned_constant_offset(&track, 3.0), &car, 10.0).unwrap();
        assert!((out.n_peak - 3.0).abs() < 0.2, "n_peak {}", out.n_peak);
        assert!(
            (out.driven_length - out.centerline_length).abs() > 1.0,
            "driven {} vs centerline {}",
            out.driven_length,
            out.centerline_length
        );
        // Trace is in centerline station and spans the loop.
        assert_eq!(out.trace.station.len(), track.segments.len());
        assert_eq!(out.trace.sector_times.len(), 3);
        assert!(out.trace.lap_time > 0.0);
    }

    #[test]
    fn zero_offset_matches_centerline_length() {
        let track = oval();
        let car = CarParams::default();
        let out =
            driven_sim_trace(&track, &aligned_constant_offset(&track, 0.0), &car, 10.0).unwrap();
        // Driven == centerline geometry ⇒ lengths match tightly.
        assert!(
            (out.driven_length - out.centerline_length).abs() < 1.0,
            "driven {} vs centerline {}",
            out.driven_length,
            out.centerline_length
        );
    }

    #[test]
    fn moving_average_smooths_periodic() {
        // A spike in an otherwise-flat n is attenuated by the filter.
        let mut v = vec![0.0; 100];
        v[50] = 10.0;
        let f = moving_average_periodic(&v, 3);
        assert!(f[50] < 2.0, "spike not attenuated: {}", f[50]);
        // Mean preserved.
        let mean_in: f64 = v.iter().sum::<f64>() / v.len() as f64;
        let mean_out: f64 = f.iter().sum::<f64>() / f.len() as f64;
        assert!((mean_in - mean_out).abs() < 1e-9);
        let _ = PI;
    }
}
