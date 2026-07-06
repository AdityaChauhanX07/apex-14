//! GPS → track-coordinate projection: for each (aligned) telemetry position,
//! find the closest centerline point's arc length `s` and the signed lateral
//! offset `n`.
//!
//! # Sign convention (matches the codebase)
//!
//! The centerline's **left** normal is `(heading + π/2)` = `(-sin h, cos h)`.
//! The viewer builds the left boundary as `center + width_left · normal` and the
//! right as `center − width_right · normal`, and the optimizer's `lateral_offset`
//! is "positive = left". We match that: **positive `n` = LEFT of the centerline**
//! in the direction of travel, negative = right.
//!
//! # Continuity / wraparound
//!
//! Telemetry samples trace the lap once. The closest-point search is seeded from
//! the previous sample's segment (a forward window), so `s` is monotone along
//! the lap and the start/finish wrap is unwrapped into a continuous, increasing
//! arc length.

use apex_telemetry::ChannelId;
use apex_track::Track;

use crate::error::CorrelateError;
use crate::telemetry::Telemetry;
use crate::Similarity;

/// One point's projection onto the centerline.
#[derive(Debug, Clone, Copy)]
pub struct Projection {
    /// Arc length (station, m) of the closest centerline point, in `[0, L)`.
    pub s: f64,
    /// Signed lateral offset (m): positive = left of the centerline.
    pub n: f64,
    /// Distance (m) from the query point to the centerline.
    pub dist: f64,
    /// Index of the centerline segment the closest point lies on.
    pub seg: usize,
}

/// Project `(x, y)` onto the centerline, searching **all** segments (global).
pub fn closest_point(track: &Track, x: f64, y: f64) -> Projection {
    project_over(track, x, y, 0..track.segments.len())
}

/// Project `(x, y)` searching only a window of segments around `hint`
/// (wrapping), for continuity. `back`/`fwd` are segment counts.
fn closest_point_windowed(
    track: &Track,
    x: f64,
    y: f64,
    hint: usize,
    back: usize,
    fwd: usize,
) -> Projection {
    let n = track.segments.len();
    let start = (hint + n - (back % n)) % n;
    let count = (back + fwd + 1).min(n);
    let mut best: Option<Projection> = None;
    for k in 0..count {
        let i = (start + k) % n;
        let p = project_single_segment(track, x, y, i);
        if best.as_ref().map(|b| p.dist < b.dist).unwrap_or(true) {
            best = Some(p);
        }
    }
    best.expect("non-empty window")
}

/// Project over a range of segment indices, returning the closest.
fn project_over(track: &Track, x: f64, y: f64, range: std::ops::Range<usize>) -> Projection {
    let mut best: Option<Projection> = None;
    for i in range {
        let p = project_single_segment(track, x, y, i);
        if best.as_ref().map(|b| p.dist < b.dist).unwrap_or(true) {
            best = Some(p);
        }
    }
    best.expect("non-empty track")
}

/// Project `(x, y)` onto centerline segment `i` (from `seg[i]` to its
/// successor, wrapping on closed tracks). Returns arc length, signed offset,
/// distance, and the segment index.
fn project_single_segment(track: &Track, x: f64, y: f64, i: usize) -> Projection {
    let n = track.segments.len();
    let a = &track.segments[i];
    let (jx, jy, upper_s) = if i + 1 < n {
        let b = &track.segments[i + 1];
        (b.x, b.y, b.s)
    } else if track.is_closed {
        let b = &track.segments[0];
        (b.x, b.y, track.total_length)
    } else {
        // Open track, last node: degenerate segment — project onto the point.
        let dx = x - a.x;
        let dy = y - a.y;
        return Projection {
            s: a.s,
            n: 0.0,
            dist: (dx * dx + dy * dy).sqrt(),
            seg: i,
        };
    };

    let ex = jx - a.x;
    let ey = jy - a.y;
    let len2 = ex * ex + ey * ey;
    // Parameter of the closest point on the segment, clamped to [0, 1].
    let t = if len2 > 0.0 {
        (((x - a.x) * ex + (y - a.y) * ey) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let cx = a.x + t * ex;
    let cy = a.y + t * ey;
    let dx = x - cx;
    let dy = y - cy;
    let dist = (dx * dx + dy * dy).sqrt();

    // Left normal of the segment tangent: (-ey, ex) normalized. Positive n =
    // left of the centerline (direction of travel).
    let seg_len = len2.sqrt();
    let n_signed = if seg_len > 0.0 {
        (dx * (-ey) + dy * ex) / seg_len
    } else {
        0.0
    };

    let s = a.s + t * (upper_s - a.s);
    Projection {
        s,
        n: n_signed,
        dist,
        seg: i,
    }
}

/// Statistics from projecting a lap.
#[derive(Debug, Clone)]
pub struct ProjectStats {
    /// Continuous (unwrapped) `s_proj` span (m).
    pub s_proj_span: f64,
    /// Raw source `s` span (m), for comparison (FastF1 integrates speed).
    pub s_raw_span: f64,
    /// Min / max / RMS of the signed lateral offset `n` (m).
    pub n_min: f64,
    /// See [`ProjectStats::n_min`].
    pub n_max: f64,
    /// See [`ProjectStats::n_min`].
    pub n_rms: f64,
    /// Max closest-point distance (m).
    pub max_dist: f64,
    /// Fraction of samples with `|n|` within the local track half-width `[0,1]`.
    pub frac_within_bounds: f64,
    /// Number of samples whose `s` decreased vs. the previous (non-monotone).
    pub non_monotone: usize,
}

/// Project aligned telemetry onto the track, returning a new [`Telemetry`] with
/// `s` replaced by the projected station, the raw source `s` kept as `s_raw`,
/// `x`/`y` replaced by the track-frame positions, and `lateral_offset` added —
/// plus per-lap [`ProjectStats`].
pub fn project_to_track(
    telemetry: &Telemetry,
    track: &Track,
    transform: &Similarity,
) -> Result<(Telemetry, ProjectStats), CorrelateError> {
    let s_raw = telemetry
        .channel(ChannelId::S)
        .ok_or(CorrelateError::MissingAxis("s"))?
        .to_vec();
    let x = telemetry
        .channel(ChannelId::X)
        .ok_or(CorrelateError::MissingAxis("x"))?;
    let y = telemetry
        .channel(ChannelId::Y)
        .ok_or(CorrelateError::MissingAxis("y"))?;
    let count = s_raw.len();
    if count == 0 {
        return Err(CorrelateError::AlignFailed("empty telemetry"));
    }
    let l = track.total_length;

    let mut s_proj = Vec::with_capacity(count);
    let mut n_vec = Vec::with_capacity(count);
    let mut tx_vec = Vec::with_capacity(count);
    let mut ty_vec = Vec::with_capacity(count);

    let mut hint = 0usize;
    let mut unwrap = 0.0;
    let mut prev_raw = 0.0;
    let mut prev_cont = f64::NEG_INFINITY;
    let mut non_monotone = 0usize;
    let mut n_min = f64::INFINITY;
    let mut n_max = f64::NEG_INFINITY;
    let mut nsq = 0.0;
    let mut maxd: f64 = 0.0;
    let mut within = 0usize;

    for i in 0..count {
        let (px, py) = transform.apply(x[i], y[i]);
        tx_vec.push(px);
        ty_vec.push(py);

        let proj = if i == 0 {
            closest_point(track, px, py)
        } else {
            // Small back window absorbs GPS jitter; forward window must exceed
            // the max sample spacing in arc length.
            closest_point_windowed(track, px, py, hint, 8, 80)
        };
        hint = proj.seg;

        // Unwrap the station into a continuous, monotone arc length.
        if i == 0 {
            unwrap = 0.0;
            prev_raw = proj.s;
        } else {
            let delta = proj.s - prev_raw;
            if delta < -l / 2.0 {
                unwrap += l; // crossed start/finish forward
            } else if delta > l / 2.0 {
                unwrap -= l; // rare backward wrap
            }
            prev_raw = proj.s;
        }
        let cont = proj.s + unwrap;
        if cont < prev_cont {
            non_monotone += 1;
        }
        prev_cont = cont;
        s_proj.push(cont);

        n_vec.push(proj.n);
        n_min = n_min.min(proj.n);
        n_max = n_max.max(proj.n);
        nsq += proj.n * proj.n;
        maxd = maxd.max(proj.dist);

        // Within-bounds check against the local half-width on that side.
        let (wl, wr) = track.width_at(proj.s);
        let half = if proj.n >= 0.0 { wl } else { wr };
        if proj.n.abs() <= half {
            within += 1;
        }
    }

    let stats = ProjectStats {
        s_proj_span: s_proj.last().unwrap() - s_proj.first().unwrap(),
        s_raw_span: s_raw.last().unwrap() - s_raw.first().unwrap(),
        n_min,
        n_max,
        n_rms: (nsq / count as f64).sqrt(),
        max_dist: maxd,
        frac_within_bounds: within as f64 / count as f64,
        non_monotone,
    };

    // Assemble the projected telemetry: keep all channels, remap s/x/y, add
    // s_raw and lateral_offset.
    let mut channels = telemetry.channels.clone();
    channels.insert(ChannelId::SRaw, s_raw);
    channels.insert(ChannelId::S, s_proj);
    channels.insert(ChannelId::X, tx_vec);
    channels.insert(ChannelId::Y, ty_vec);
    channels.insert(ChannelId::LateralOffset, n_vec);

    let out = Telemetry {
        grid: crate::GridKind::S,
        channels,
        metadata: telemetry.metadata.clone(),
    };
    Ok((out, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_track::{build_track, oval_track};
    use std::collections::BTreeMap;
    use std::f64::consts::PI;

    fn oval() -> Track {
        let (pts, closed) = oval_track(600.0, 120.0, 12.0, 600);
        build_track("oval", &pts, closed)
    }

    #[test]
    fn projects_centerline_point_to_zero_offset() {
        let track = oval();
        // A point exactly on the centerline at s≈300 has n≈0.
        let (cx, cy) = track.position_at(300.0);
        let p = closest_point(&track, cx, cy);
        assert!((p.s - 300.0).abs() < 2.0, "s {}", p.s);
        assert!(p.n.abs() < 0.2, "n {}", p.n);
        assert!(p.dist < 0.2, "dist {}", p.dist);
    }

    #[test]
    fn left_offset_is_positive() {
        let track = oval();
        let s = 250.0;
        let (cx, cy) = track.position_at(s);
        let h = track.heading_at(s);
        // Move 3 m along the LEFT normal (h + 90°).
        let (nx, ny) = ((h + PI / 2.0).cos(), (h + PI / 2.0).sin());
        let p = closest_point(&track, cx + 3.0 * nx, cy + 3.0 * ny);
        assert!(p.n > 2.5, "expected +~3, got n={}", p.n);
        // Right side is negative.
        let p2 = closest_point(&track, cx - 3.0 * nx, cy - 3.0 * ny);
        assert!(p2.n < -2.5, "expected -~3, got n={}", p2.n);
    }

    /// Build telemetry that follows the centerline with a known sinusoidal
    /// lateral offset n(s), and project it back.
    #[test]
    fn recovers_sinusoidal_offset_with_wraparound() {
        let track = oval();
        let l = track.total_length;
        let ident = Similarity {
            scale: 1.0,
            theta: 0.0,
            tx: 0.0,
            ty: 0.0,
            reflect: false,
        };
        // Start near the end so the lap crosses the start/finish line (wrap).
        let start = l - 200.0;
        let n_samples = 500;
        let mut s_raw = Vec::new();
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut true_n = Vec::new();
        for j in 0..n_samples {
            let arc = j as f64 / (n_samples - 1) as f64 * (l - 2.0);
            let station = start + arc;
            let (cx, cy) = track.position_at(station);
            let h = track.heading_at(station);
            let nval = 4.0 * (arc / l * 6.0 * PI).sin(); // sinusoid, |n|<=4
            let (nx, ny) = ((h + PI / 2.0).cos(), (h + PI / 2.0).sin());
            s_raw.push(arc);
            xs.push(cx + nval * nx);
            ys.push(cy + nval * ny);
            true_n.push(nval);
        }
        let mut channels: BTreeMap<ChannelId, Vec<f64>> = BTreeMap::new();
        channels.insert(ChannelId::S, s_raw);
        channels.insert(ChannelId::X, xs);
        channels.insert(ChannelId::Y, ys);
        let tel = Telemetry {
            grid: crate::GridKind::S,
            channels,
            metadata: Vec::new(),
        };

        let (out, stats) = project_to_track(&tel, &track, &ident).unwrap();
        let n_out = out.channel(ChannelId::LateralOffset).unwrap();
        let s_out = out.channel(ChannelId::S).unwrap();

        // Recovered n matches the known sinusoid.
        for (got, want) in n_out.iter().zip(&true_n) {
            assert!((got - want).abs() < 0.15, "n {} vs {}", got, want);
        }
        // s is monotone (wrap handled) and spans ~ one lap.
        assert_eq!(stats.non_monotone, 0, "s must be monotone");
        for w in s_out.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-6,
                "s not monotone: {} then {}",
                w[0],
                w[1]
            );
        }
        assert!(
            (stats.s_proj_span - (l - 2.0)).abs() < 5.0,
            "span {} vs {}",
            stats.s_proj_span,
            l - 2.0
        );
        assert!(
            stats.frac_within_bounds > 0.99,
            "within {}",
            stats.frac_within_bounds
        );
    }
}
