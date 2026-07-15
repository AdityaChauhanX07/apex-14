//! Curvature-aware smoothing of imported centerlines.
//!
//! Real survey data (TUMFTM racetrack-database, ~5 m point spacing) carries
//! sub-metre position noise. Position noise is harmless for the *shape* but
//! catastrophic for *curvature* (a second derivative): it manufactures phantom
//! tight corners — e.g. Silverstone reads R ≈ 34–62 m at s ≈ 400 m where the
//! real track is gentle, and an impossible R ≈ 12 m spike near s ≈ 1044 m. The
//! QSS lap simulator reads those radii and slows to a crawl.
//!
//! This module fits a smooth centerline that minimizes curvature roughness
//! subject to a hard cap on how far any point may move from its survey point.
//!
//! Scope note: this is a **pulled-forward slice of the 3D track model work** —
//! it was needed early so the telemetry correlation work could run the QSS on
//! real (noisy) imported centerlines without phantom-curvature slowdowns.
//!
//! # Method (regularized least squares, 2D only)
//!
//! For each coordinate independently we solve
//!
//! ```text
//! minimize_p  Σ_i (p_i − q_i)²  +  λ Σ_i (p_{i−1} − 2 p_i + p_{i+1})²
//! ```
//!
//! i.e. a data term pulling the smoothed points `p` toward the survey points
//! `q`, plus a **second-difference** roughness penalty (the discrete curvature
//! energy). The normal equations are `(I + λ D_2ᵀ D_2) p = q`, a symmetric
//! positive-definite periodic penta-diagonal system (4th-difference stencil
//! `[1, −4, 6, −4, 1]`), solved with conjugate gradients — no external solver,
//! no RNG, no threads, so it stays **wasm-compatible** (apex-track is in the
//! web-viewer graph).
//!
//! Closed tracks use periodic indexing, so there is no kink at the start/finish
//! seam. `λ` is chosen by bisection to be the **largest** (smoothest) value
//! whose maximum point deviation stays within the tolerance — smoothing exactly
//! as hard as the deviation budget allows.
//!
//! This is a pulled-forward minimal slice of the **3D track model work**
//! ("smoothing (regularized least squares) to kill GPS/LiDAR noise"): 2D
//! `(x, y)` only, no
//! elevation, no width smoothing.

use crate::builder::build_track;
use crate::types::{Track, TrackPoint};

/// Default maximum deviation (m) a smoothed point may sit from its survey point.
pub const DEFAULT_SMOOTH_TOLERANCE_M: f64 = 1.0;

/// Before/after diagnostics for a smoothing pass.
#[derive(Debug, Clone)]
pub struct SmoothingReport {
    /// Deviation tolerance requested (m).
    pub tolerance_m: f64,
    /// Regularization weight `λ` that the bisection settled on.
    pub lambda: f64,
    /// Actual maximum point deviation achieved (m).
    pub max_deviation_m: f64,
    /// Total arc length before / after (m) — should change < 0.1%.
    pub length_before: f64,
    /// See [`SmoothingReport::length_before`].
    pub length_after: f64,
    /// Curvature magnitude percentiles before (1/m): p50, p95, max.
    pub kappa_p50_before: f64,
    /// See [`SmoothingReport::kappa_p50_before`].
    pub kappa_p95_before: f64,
    /// See [`SmoothingReport::kappa_p50_before`].
    pub kappa_max_before: f64,
    /// Curvature magnitude percentiles after (1/m): p50, p95, max.
    pub kappa_p50_after: f64,
    /// See [`SmoothingReport::kappa_p50_after`].
    pub kappa_p95_after: f64,
    /// See [`SmoothingReport::kappa_p50_after`].
    pub kappa_max_after: f64,
    /// Tightest radius before (m) = 1 / max|κ|.
    pub min_radius_before: f64,
    /// Tightest radius after (m).
    pub min_radius_after: f64,
}

/// Smooth a processed [`Track`]'s centerline (widths untouched) and return the
/// rebuilt track plus before/after diagnostics.
///
/// The track's `is_closed` flag drives periodic vs. natural boundary handling.
pub fn smooth_track(track: &Track, tolerance_m: f64) -> (Track, SmoothingReport) {
    let points: Vec<TrackPoint> = track
        .segments
        .iter()
        .map(|s| TrackPoint {
            x: s.x,
            y: s.y,
            width_left: s.width_left,
            width_right: s.width_right,
        })
        .collect();

    let (smoothed, lambda, max_dev) = smooth_points(&points, track.is_closed, tolerance_m);
    let out = build_track(&track.name, &smoothed, track.is_closed);

    let (p50_b, p95_b, max_b) = kappa_percentiles(track);
    let (p50_a, p95_a, max_a) = kappa_percentiles(&out);

    let report = SmoothingReport {
        tolerance_m,
        lambda,
        max_deviation_m: max_dev,
        length_before: track.total_length,
        length_after: out.total_length,
        kappa_p50_before: p50_b,
        kappa_p95_before: p95_b,
        kappa_max_before: max_b,
        kappa_p50_after: p50_a,
        kappa_p95_after: p95_a,
        kappa_max_after: max_a,
        min_radius_before: radius_of(max_b),
        min_radius_after: radius_of(max_a),
    };
    (out, report)
}

/// Smooth raw `(x, y)` points, returning `(smoothed_points, lambda, max_dev)`.
///
/// Widths are carried through unchanged. `λ` is chosen so the maximum deviation
/// is as close as possible to (but not exceeding) `tolerance_m`.
pub fn smooth_points(
    points: &[TrackPoint],
    closed: bool,
    tolerance_m: f64,
) -> (Vec<TrackPoint>, f64, f64) {
    let n = points.len();
    if n < 5 || tolerance_m <= 0.0 {
        return (points.to_vec(), 0.0, 0.0);
    }
    let qx: Vec<f64> = points.iter().map(|p| p.x).collect();
    let qy: Vec<f64> = points.iter().map(|p| p.y).collect();

    // Deviation as a function of λ is monotincreasing (λ = 0 ⇒ p = q ⇒ dev 0).
    // Grow an upper bracket until deviation meets/exceeds the tolerance, then
    // bisect for the largest λ within budget.
    let dev_at = |lambda: f64| -> f64 {
        let px = solve(&qx, lambda, closed);
        let py = solve(&qy, lambda, closed);
        max_deviation(&qx, &qy, &px, &py)
    };

    let mut hi = 1.0;
    let cap = 1.0e9;
    while hi < cap && dev_at(hi) < tolerance_m {
        hi *= 8.0;
    }
    // If even the cap keeps us under budget, use it (fully smooth).
    let lambda = if dev_at(hi) < tolerance_m {
        hi
    } else {
        // Bisection on [lo, hi] toward dev = tolerance.
        let mut lo = 0.0;
        let mut hi = hi;
        for _ in 0..48 {
            let mid = 0.5 * (lo + hi);
            if dev_at(mid) > tolerance_m {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        lo // largest λ with deviation ≤ tolerance
    };

    let px = solve(&qx, lambda, closed);
    let py = solve(&qy, lambda, closed);
    let max_dev = max_deviation(&qx, &qy, &px, &py);

    let out = points
        .iter()
        .enumerate()
        .map(|(i, p)| TrackPoint {
            x: px[i],
            y: py[i],
            width_left: p.width_left,
            width_right: p.width_right,
        })
        .collect();
    (out, lambda, max_dev)
}

/// Solve `(I + λ D₂ᵀD₂) p = q` for one coordinate via conjugate gradients.
fn solve(q: &[f64], lambda: f64, closed: bool) -> Vec<f64> {
    let n = q.len();
    if lambda == 0.0 {
        return q.to_vec();
    }
    // x0 = q is a good start (A ≈ I for small λ).
    let mut x = q.to_vec();
    let ax = apply_a(&x, lambda, closed);
    let mut r: Vec<f64> = q.iter().zip(&ax).map(|(qi, a)| qi - a).collect();
    let mut p = r.clone();
    let mut rs = dot(&r, &r);
    let q_norm = dot(q, q).sqrt().max(1e-30);
    let tol = 1e-12 * q_norm;

    let max_iter = 10 * n + 100;
    for _ in 0..max_iter {
        if rs.sqrt() <= tol {
            break;
        }
        let ap = apply_a(&p, lambda, closed);
        let denom = dot(&p, &ap);
        if denom.abs() < 1e-300 {
            break;
        }
        let alpha = rs / denom;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot(&r, &r);
        let beta = rs_new / rs;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs = rs_new;
    }
    x
}

/// `A v = v + λ (D₂ᵀD₂) v`, the 4th-difference roughness operator.
fn apply_a(v: &[f64], lambda: f64, closed: bool) -> Vec<f64> {
    let l = apply_l(v, closed);
    v.iter().zip(&l).map(|(vi, li)| vi + lambda * li).collect()
}

/// `D₂ᵀD₂ v` — the discrete curvature-energy operator. For an interior point
/// the stencil is `v[i−2] − 4 v[i−1] + 6 v[i] − 4 v[i+1] + v[i+2]`. Closed
/// tracks wrap; open tracks omit stencil terms that fall off either end
/// (natural boundary), so the ends are penalty-free.
fn apply_l(v: &[f64], closed: bool) -> Vec<f64> {
    let n = v.len();
    let mut out = vec![0.0; n];
    if closed {
        for i in 0..n {
            let m2 = v[(i + n - 2) % n];
            let m1 = v[(i + n - 1) % n];
            let p1 = v[(i + 1) % n];
            let p2 = v[(i + 2) % n];
            out[i] = m2 - 4.0 * m1 + 6.0 * v[i] - 4.0 * p1 + p2;
        }
    } else {
        // Sum of contributions from each second-difference row D₂ centered at
        // k = 1..n-2: row k touches v[k-1], v[k], v[k+1]. (D₂ᵀD₂ v)_i =
        // Σ_k D₂[k,i] (D₂ v)_k. Build (D₂ v) then apply D₂ᵀ.
        // Second differences at interior nodes; d[0] = d[n-1] = 0 by
        // construction, so the D₂ᵀ application below needs no extra boundary
        // masking (out-of-interior d entries are already zero).
        let mut d = vec![0.0; n];
        for k in 1..n - 1 {
            d[k] = v[k - 1] - 2.0 * v[k] + v[k + 1];
        }
        for i in 0..n {
            let left = if i >= 1 { d[i - 1] } else { 0.0 };
            let right = if i + 1 < n { d[i + 1] } else { 0.0 };
            out[i] = left - 2.0 * d[i] + right;
        }
    }
    out
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn max_deviation(qx: &[f64], qy: &[f64], px: &[f64], py: &[f64]) -> f64 {
    (0..qx.len())
        .map(|i| ((px[i] - qx[i]).powi(2) + (py[i] - qy[i]).powi(2)).sqrt())
        .fold(0.0, f64::max)
}

/// `(p50, p95, max)` of `|κ|` over the track's segments.
fn kappa_percentiles(track: &Track) -> (f64, f64, f64) {
    let mut ks: Vec<f64> = track.segments.iter().map(|s| s.curvature.abs()).collect();
    ks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pct = |p: f64| -> f64 {
        if ks.is_empty() {
            return 0.0;
        }
        let idx = ((p * (ks.len() - 1) as f64).round() as usize).min(ks.len() - 1);
        ks[idx]
    };
    (pct(0.50), pct(0.95), ks.last().copied().unwrap_or(0.0))
}

fn radius_of(kappa_max: f64) -> f64 {
    if kappa_max > 1e-9 {
        1.0 / kappa_max
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// A noisy circle of radius `r` with `n` points and additive radial noise.
    fn noisy_circle(r: f64, n: usize, noise: f64) -> Vec<TrackPoint> {
        // Deterministic pseudo-noise (no RNG dependency).
        (0..n)
            .map(|i| {
                let th = 2.0 * PI * i as f64 / n as f64;
                // A couple of incommensurate sinusoids imitate survey jitter.
                let jitter = noise
                    * ((13.0 * th).sin() * 0.6 + (29.0 * th).cos() * 0.4 + (7.0 * th).sin() * 0.5);
                let rr = r + jitter;
                TrackPoint {
                    x: rr * th.cos(),
                    y: rr * th.sin(),
                    width_left: 5.0,
                    width_right: 5.0,
                }
            })
            .collect()
    }

    #[test]
    fn recovers_circle_radius_within_one_percent() {
        let r = 100.0;
        let n = 400;
        let pts = noisy_circle(r, n, 0.8);
        let raw = build_track("noisy", &pts, true);
        // Raw curvature is badly corrupted by the noise.
        let raw_min_r = radius_of(kappa_percentiles(&raw).2);
        let (smooth, report) = smooth_track(&raw, 1.0);

        // The smoothed tightest radius should be near the true 100 m (the noise
        // no longer manufactures tighter phantom radii).
        let med_kappa = report.kappa_p50_after;
        let med_r = 1.0 / med_kappa;
        assert!(
            (med_r - r).abs() / r < 0.01,
            "median radius {med_r} vs {r} (raw min R was {raw_min_r})"
        );
        // Smoothing strictly reduced the peak curvature.
        assert!(report.kappa_max_after < report.kappa_max_before);
        // Length preserved.
        assert!((smooth.total_length - raw.total_length).abs() / raw.total_length < 0.01);
    }

    #[test]
    fn deviation_constraint_is_honored() {
        let pts = noisy_circle(120.0, 500, 0.9);
        let tol = 0.75;
        let (smoothed, _lambda, max_dev) = smooth_points(&pts, true, tol);
        assert!(max_dev <= tol + 1e-6, "max_dev {max_dev} > tol {tol}");
        // Every individual point within tolerance (the constraint, pointwise).
        for (p, q) in smoothed.iter().zip(&pts) {
            let d = ((p.x - q.x).powi(2) + (p.y - q.y).powi(2)).sqrt();
            assert!(d <= tol + 1e-6, "point deviation {d} > tol {tol}");
        }
    }

    #[test]
    fn periodicity_no_seam_kink() {
        // A smooth ellipse: curvature must be continuous across the seam (index
        // 0), i.e. κ[0] close to the average of its wrapped neighbours.
        let n = 360;
        let pts: Vec<TrackPoint> = (0..n)
            .map(|i| {
                let th = 2.0 * PI * i as f64 / n as f64;
                TrackPoint {
                    x: 200.0 * th.cos(),
                    y: 120.0 * th.sin(),
                    width_left: 5.0,
                    width_right: 5.0,
                }
            })
            .collect();
        let raw = build_track("ellipse", &pts, true);
        let (smooth, _) = smooth_track(&raw, 1.0);
        let k = |i: usize| smooth.segments[i].curvature;
        let seam = k(0);
        let neighbor_avg = 0.5 * (k(n - 1) + k(1));
        assert!(
            (seam - neighbor_avg).abs() < 1e-3,
            "seam curvature {seam} vs neighbour avg {neighbor_avg}"
        );
    }

    #[test]
    fn short_tracks_pass_through() {
        let pts: Vec<TrackPoint> = (0..4)
            .map(|i| TrackPoint {
                x: i as f64,
                y: 0.0,
                width_left: 1.0,
                width_right: 1.0,
            })
            .collect();
        let (out, lambda, dev) = smooth_points(&pts, false, 1.0);
        assert_eq!(out.len(), 4);
        assert_eq!(lambda, 0.0);
        assert_eq!(dev, 0.0);
    }
}
