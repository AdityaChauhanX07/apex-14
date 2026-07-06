//! Track-frame alignment: fit the 2D similarity transform mapping FastF1-local
//! telemetry XY onto our track (centerline) frame.
//!
//! # Approach (arc-length correspondence + Procrustes)
//!
//! We do **not** know point correspondences a priori, but both curves are
//! arc-length parameterized: telemetry carries `s` (FastF1 Distance) and the
//! centerline carries station (cumulative arc length). Both describe the same
//! closed loop, so normalized arc-length fraction `q ∈ [0,1)` gives a natural
//! correspondence — up to four discrete unknowns:
//!
//! - a **start-line offset** `s_offset ∈ [0, L)` (telemetry `s=0` sits at some
//!   centerline station),
//! - a **direction** (telemetry may run the loop the opposite way),
//! - a **reflection** (FastF1's frame may be left-handed vs. ours),
//!
//! plus the continuous similarity (rotation `θ`, uniform scale `c`, translation
//! `t`). For each of the `2 × 2` (reflection × direction) cases and a candidate
//! `s_offset`, we build corresponded point pairs (telemetry point at fraction
//! `q_j` ↔ centerline point at station `s_offset ± q_j·L`) and solve the
//! closed-form 2D Procrustes/Umeyama similarity (proper rotation, with scale).
//! We grid-search `s_offset`, keep the global least-squares winner, then refine
//! `s_offset` locally. Scale `c` should come out ≈ 1 (both frames are metres).
//!
//! The reported `rms` is the acceptance metric: RMS distance from each
//! transformed telemetry position to its **closest** centerline point (which
//! includes the real lateral offset of the driven line — it is not zero).

use apex_track::Track;
use serde::{Deserialize, Serialize};

use crate::error::CorrelateError;
use crate::project::closest_point;
use crate::telemetry::Telemetry;
use apex_telemetry::ChannelId;

/// A 2D similarity transform (optional reflection, then scale·rotation, then
/// translation) mapping source (telemetry) coordinates into the track frame.
///
/// `apply(x, y)`:
/// 1. if `reflect`, flip `y → -y` (reflection across the source x-axis),
/// 2. rotate by `theta` and scale by `scale`,
/// 3. add `(tx, ty)`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Similarity {
    /// Uniform scale factor (≈ 1 when both frames are metres).
    pub scale: f64,
    /// Rotation angle (radians).
    pub theta: f64,
    /// Translation X (metres, track frame).
    pub tx: f64,
    /// Translation Y (metres, track frame).
    pub ty: f64,
    /// Whether a handedness flip (`y → -y`) is applied before rotation.
    pub reflect: bool,
}

impl Similarity {
    /// Apply the transform to a source point.
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let sy = if self.reflect { -y } else { y };
        let (c, s) = (self.theta.cos(), self.theta.sin());
        let rx = self.scale * (c * x - s * sy);
        let ry = self.scale * (s * x + c * sy);
        (rx + self.tx, ry + self.ty)
    }
}

/// The result of an alignment fit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignResult {
    /// The fitted similarity transform (includes the reflection flag).
    pub transform: Similarity,
    /// Whether telemetry runs the loop opposite to the centerline direction.
    pub direction_reversed: bool,
    /// Centerline station (m) that telemetry `s = s[0]` maps to.
    pub s_offset: f64,
    /// Acceptance RMS: closest-point distance from transformed telemetry to the
    /// centerline (metres). Includes real lateral offset of the driven line.
    pub rms: f64,
    /// Maximum closest-point distance (metres) — an outlier check.
    pub max_dist: f64,
}

/// Tuning for the alignment search.
#[derive(Debug, Clone, Copy)]
pub struct AlignConfig {
    /// Number of corresponded samples used in the coarse search.
    pub n_corr: usize,
    /// Coarse `s_offset` search step (metres).
    pub coarse_step: f64,
}

impl Default for AlignConfig {
    fn default() -> Self {
        AlignConfig {
            n_corr: 240,
            coarse_step: 15.0,
        }
    }
}

/// Fit the similarity transform mapping `telemetry` XY onto `track`'s frame.
///
/// Requires `x`, `y`, and the grid axis (`s`) channels in `telemetry`.
pub fn fit_alignment(
    telemetry: &Telemetry,
    track: &Track,
    config: AlignConfig,
) -> Result<AlignResult, CorrelateError> {
    let s = telemetry
        .channel(ChannelId::S)
        .ok_or(CorrelateError::MissingAxis("s"))?;
    let x = telemetry
        .channel(ChannelId::X)
        .ok_or(CorrelateError::MissingAxis("x"))?;
    let y = telemetry
        .channel(ChannelId::Y)
        .ok_or(CorrelateError::MissingAxis("y"))?;
    if s.len() < 3 || track.segments.len() < 3 {
        return Err(CorrelateError::AlignFailed(
            "need >=3 telemetry samples and track segments",
        ));
    }

    // Keep only samples with finite position and s; require monotone s.
    let mut pts: Vec<(f64, f64, f64)> = Vec::with_capacity(s.len());
    for i in 0..s.len() {
        if s[i].is_finite() && x[i].is_finite() && y[i].is_finite() {
            pts.push((s[i], x[i], y[i]));
        }
    }
    if pts.len() < 3 {
        return Err(CorrelateError::AlignFailed(
            "too few finite telemetry samples",
        ));
    }
    let s0 = pts.first().unwrap().0;
    let s_span = pts.last().unwrap().0 - s0;
    if s_span <= 0.0 {
        return Err(CorrelateError::AlignFailed(
            "telemetry s span is not positive",
        ));
    }
    let l = track.total_length;

    // Corresponded source points at uniform fractions q_j (interpolated from the
    // raw telemetry), fixed across all candidates.
    let m = config.n_corr.max(8);
    let mut src: Vec<(f64, f64)> = Vec::with_capacity(m);
    let mut fracs: Vec<f64> = Vec::with_capacity(m);
    for j in 0..m {
        let q = j as f64 / (m - 1) as f64;
        let s_query = s0 + q * s_span;
        let (px, py) = interp_xy(&pts, s_query);
        src.push((px, py));
        fracs.push(q);
    }

    // Coarse grid search over (reflection, direction, s_offset).
    let mut best: Option<(f64, Candidate)> = None; // (corr_rms, candidate)
    let n_delta = ((l / config.coarse_step).ceil() as usize).max(1);
    for &reflect in &[false, true] {
        // Pre-reflect the source once per reflection case.
        let src_r: Vec<(f64, f64)> = src
            .iter()
            .map(|&(px, py)| if reflect { (px, -py) } else { (px, py) })
            .collect();
        for &dir_rev in &[false, true] {
            for k in 0..n_delta {
                let delta = k as f64 * config.coarse_step;
                let (rms, sim) = fit_candidate(&src_r, &fracs, track, l, delta, dir_rev, reflect);
                let cand = Candidate {
                    sim,
                    dir_rev,
                    delta,
                };
                if best.as_ref().map(|(r, _)| rms < *r).unwrap_or(true) {
                    best = Some((rms, cand));
                }
            }
        }
    }
    let (_, mut winner) = best.expect("at least one candidate");

    // Local refinement of s_offset (two shrinking passes) at the winning
    // reflection/direction.
    let reflect = winner.sim.reflect;
    let src_r: Vec<(f64, f64)> = src
        .iter()
        .map(|&(px, py)| if reflect { (px, -py) } else { (px, py) })
        .collect();
    let mut half = config.coarse_step;
    for _ in 0..2 {
        let steps = 40;
        let lo = winner.delta - half;
        let mut local_best = (f64::INFINITY, winner.clone());
        for k in 0..=steps {
            let delta = lo + (2.0 * half) * (k as f64 / steps as f64);
            let (rms, sim) =
                fit_candidate(&src_r, &fracs, track, l, delta, winner.dir_rev, reflect);
            if rms < local_best.0 {
                local_best = (
                    rms,
                    Candidate {
                        sim,
                        dir_rev: winner.dir_rev,
                        delta,
                    },
                );
            }
        }
        winner = local_best.1;
        half /= (steps as f64) / 4.0; // shrink window
    }

    // Acceptance RMS: closest-point distance over ALL raw samples.
    let mut sumsq = 0.0;
    let mut maxd: f64 = 0.0;
    for &(_, px, py) in &pts {
        let (tx, ty) = winner.sim.apply(px, py);
        let proj = closest_point(track, tx, ty);
        sumsq += proj.dist * proj.dist;
        maxd = maxd.max(proj.dist);
    }
    let rms = (sumsq / pts.len() as f64).sqrt();

    // s_offset normalized to [0, L).
    let s_offset = winner.delta.rem_euclid(l);

    Ok(AlignResult {
        transform: winner.sim,
        direction_reversed: winner.dir_rev,
        s_offset,
        rms,
        max_dist: maxd,
    })
}

#[derive(Debug, Clone)]
struct Candidate {
    sim: Similarity,
    dir_rev: bool,
    delta: f64,
}

/// Fit one candidate: build centerline targets for the given `delta`/direction
/// and run Procrustes on the (already reflected) source. Returns
/// `(corresponded_rms, similarity)`.
fn fit_candidate(
    src_r: &[(f64, f64)],
    fracs: &[f64],
    track: &Track,
    l: f64,
    delta: f64,
    dir_rev: bool,
    reflect: bool,
) -> (f64, Similarity) {
    let dst: Vec<(f64, f64)> = fracs
        .iter()
        .map(|&q| {
            let station = if dir_rev {
                delta - q * l
            } else {
                delta + q * l
            };
            track.position_at(station)
        })
        .collect();
    let (scale, theta, tx, ty, rms) = procrustes(src_r, &dst);
    (
        rms,
        Similarity {
            scale,
            theta,
            tx,
            ty,
            reflect,
        },
    )
}

/// Closed-form 2D similarity (proper rotation + uniform scale + translation)
/// mapping `src → dst` in a least-squares sense (Umeyama). Returns
/// `(scale, theta, tx, ty, rms)`.
fn procrustes(src: &[(f64, f64)], dst: &[(f64, f64)]) -> (f64, f64, f64, f64, f64) {
    let n = src.len() as f64;
    let (mut msx, mut msy, mut mdx, mut mdy) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..src.len() {
        msx += src[i].0;
        msy += src[i].1;
        mdx += dst[i].0;
        mdy += dst[i].1;
    }
    msx /= n;
    msy /= n;
    mdx /= n;
    mdy /= n;

    let (mut a, mut b, mut var_s) = (0.0, 0.0, 0.0);
    for i in 0..src.len() {
        let (sx, sy) = (src[i].0 - msx, src[i].1 - msy);
        let (dx, dy) = (dst[i].0 - mdx, dst[i].1 - mdy);
        a += sx * dx + sy * dy; // dot
        b += sx * dy - sy * dx; // cross
        var_s += sx * sx + sy * sy;
    }
    let theta = b.atan2(a);
    let scale = if var_s > 0.0 {
        (a * a + b * b).sqrt() / var_s
    } else {
        1.0
    };
    let (c, s) = (theta.cos(), theta.sin());
    let tx = mdx - scale * (c * msx - s * msy);
    let ty = mdy - scale * (s * msx + c * msy);

    // RMS of the corresponded residuals.
    let mut sumsq = 0.0;
    for i in 0..src.len() {
        let rx = scale * (c * src[i].0 - s * src[i].1) + tx;
        let ry = scale * (s * src[i].0 + c * src[i].1) + ty;
        let ex = rx - dst[i].0;
        let ey = ry - dst[i].1;
        sumsq += ex * ex + ey * ey;
    }
    (scale, theta, tx, ty, (sumsq / n).sqrt())
}

/// Linear interpolation of telemetry `(x, y)` at arc length `s_query`. `pts` is
/// `(s, x, y)` sorted by strictly increasing `s`; queries are clamped to range.
fn interp_xy(pts: &[(f64, f64, f64)], s_query: f64) -> (f64, f64) {
    if s_query <= pts[0].0 {
        return (pts[0].1, pts[0].2);
    }
    let last = pts.len() - 1;
    if s_query >= pts[last].0 {
        return (pts[last].1, pts[last].2);
    }
    // Binary search for the bracketing interval.
    let (mut lo, mut hi) = (0usize, last);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if pts[mid].0 <= s_query {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (s0, x0, y0) = pts[lo];
    let (s1, x1, y1) = pts[hi];
    let t = (s_query - s0) / (s1 - s0);
    (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_track::{build_track, TrackPoint};
    use std::collections::BTreeMap;

    /// An asymmetric smooth closed loop (no mirror symmetry, no direction
    /// symmetry) so reflection/direction flags are uniquely identifiable —
    /// unlike an oval, where reflect+reverse is equivalent to a rotation.
    fn asymmetric_track() -> Track {
        let n = 400;
        let mut pts = Vec::with_capacity(n);
        for i in 0..n {
            let th = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            let x = 300.0 * th.cos() + 40.0 * (2.0 * th).cos() + 15.0 * (3.0 * th).sin();
            let y = 260.0 * th.sin() + 30.0 * (2.0 * th).sin() - 20.0 * (3.0 * th).cos();
            pts.push(TrackPoint {
                x,
                y,
                width_left: 6.0,
                width_right: 6.0,
            });
        }
        build_track("asym", &pts, true)
    }

    /// Build a telemetry object tracing the centerline (n=0) after a known
    /// similarity transform, for round-trip recovery.
    fn telemetry_on_centerline(
        track: &Track,
        sim: &Similarity,
        dir_rev: bool,
        s_offset: f64,
        n_samples: usize,
    ) -> Telemetry {
        let l = track.total_length;
        let mut s = Vec::new();
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for j in 0..n_samples {
            let q = j as f64 / (n_samples - 1) as f64;
            let arc = q * (l - 1.0); // one lap minus a hair (open telemetry)
            let station = if dir_rev {
                s_offset - arc
            } else {
                s_offset + arc
            };
            let (cx, cy) = track.position_at(station);
            // Invert the similarity to get the source (telemetry-frame) point.
            let (sx, sy) = invert(sim, cx, cy);
            s.push(arc);
            xs.push(sx);
            ys.push(sy);
        }
        let mut channels: BTreeMap<ChannelId, Vec<f64>> = BTreeMap::new();
        channels.insert(ChannelId::S, s);
        channels.insert(ChannelId::X, xs);
        channels.insert(ChannelId::Y, ys);
        Telemetry {
            grid: crate::GridKind::S,
            channels,
            metadata: Vec::new(),
        }
    }

    /// Invert a similarity (track-frame → source-frame), for test data gen.
    fn invert(sim: &Similarity, x: f64, y: f64) -> (f64, f64) {
        let (c, s) = (sim.theta.cos(), sim.theta.sin());
        // undo translation + scale + rotation
        let ux = (x - sim.tx) / sim.scale;
        let uy = (y - sim.ty) / sim.scale;
        let rx = c * ux + s * uy; // R^-1
        let ry = -s * ux + c * uy;
        if sim.reflect {
            (rx, -ry)
        } else {
            (rx, ry)
        }
    }

    #[test]
    fn recovers_pure_rotation_translation() {
        let track = asymmetric_track();
        let sim = Similarity {
            scale: 1.0,
            theta: 0.7,
            tx: 123.0,
            ty: -45.0,
            reflect: false,
        };
        let tel = telemetry_on_centerline(&track, &sim, false, 250.0, 300);
        let r = fit_alignment(&tel, &track, AlignConfig::default()).unwrap();
        assert!(r.rms < 1.0, "rms {}", r.rms);
        assert!(
            (r.transform.scale - 1.0).abs() < 0.02,
            "scale {}",
            r.transform.scale
        );
        assert!(!r.direction_reversed);
        assert!(!r.transform.reflect);
    }

    #[test]
    fn recovers_reflection_and_reverse() {
        let track = asymmetric_track();
        let sim = Similarity {
            scale: 1.0,
            theta: -1.2,
            tx: -300.0,
            ty: 210.0,
            reflect: true,
        };
        let tel = telemetry_on_centerline(&track, &sim, true, 800.0, 300);
        let r = fit_alignment(&tel, &track, AlignConfig::default()).unwrap();
        assert!(r.rms < 1.0, "rms {}", r.rms);
        assert!(r.transform.reflect, "should detect reflection");
        assert!(r.direction_reversed, "should detect reversed direction");
    }

    #[test]
    fn recovers_scale() {
        let track = asymmetric_track();
        let sim = Similarity {
            scale: 1.03,
            theta: 0.2,
            tx: 10.0,
            ty: 20.0,
            reflect: false,
        };
        let tel = telemetry_on_centerline(&track, &sim, false, 100.0, 300);
        let r = fit_alignment(&tel, &track, AlignConfig::default()).unwrap();
        assert!(
            (r.transform.scale - 1.03).abs() < 0.02,
            "scale {}",
            r.transform.scale
        );
    }
}
