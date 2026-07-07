//! 3D curved-ribbon track geometry (Phase 1.1 — geometry plumbing only).
//!
//! [`Ribbon3d`] generalizes the flat 2D [`Track`](crate::Track) to a 3D ribbon:
//! a centerline `r(s) = (x, y, z)` carrying an orthonormal moving frame
//! `{t, n, m}` (tangent / left-lateral / surface-normal) and the three
//! **generalized curvatures** (the body-frame Darboux vector)
//! `Ω = (Ω_x, Ω_y, Ω_z)`:
//!
//! * `Ω_z` — horizontal / yaw curvature (the 2D `curvature`),
//! * `Ω_y` — pitch rate (elevation / grade change),
//! * `Ω_x` — roll / banking rate.
//!
//! plus half-widths `w_l(s), w_r(s)` and a `mu_scale(s)` placeholder (defaults
//! `1.0`, unused by physics for now).
//!
//! # Flat degenerate case — byte-exact
//!
//! A **flat** ribbon has `z = 0`, `grade = 0`, `bank = 0`, `Ω_x = Ω_y = 0`, and
//! `Ω_z` equal to the exact 2D `curvature` values the [`Track`](crate::Track)
//! holds today. [`Ribbon3d::from_flat`] copies the segment scalars **verbatim**,
//! and the scalar queries ([`Ribbon3d::position_at`], [`heading_at`], ...) reuse
//! the identical interpolation arithmetic as [`Track`], so a flat ribbon
//! reproduces the 2D queries **bitwise**. This exactness is what keeps the
//! golden fixtures byte-stable while the 3D fields are plumbed but unused.
//!
//! See `docs/math/track3d.md` for the frame construction and the Darboux
//! recovery derivation (completed when the physics lands).

use crate::builder::normalize_angle;
use crate::types::Track;

/// Default (unused) grip scaling — a placeholder for future 3D physics.
pub const DEFAULT_MU_SCALE: f64 = 1.0;

// --- tiny 3-vector helpers (no external dep; apex-track is in the wasm graph) --

type V3 = [f64; 3];

#[inline]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
#[inline]
fn normalize(a: V3) -> V3 {
    let n = norm(a);
    if n > 0.0 {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        a
    }
}

/// The orthonormal moving frame at a station: tangent, left-lateral, and
/// surface-normal unit vectors, right-handed (`t × n = m`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    /// Unit tangent (direction of travel).
    pub t: V3,
    /// Unit left-lateral (across the ribbon, +left).
    pub n: V3,
    /// Unit surface normal (out of the ribbon surface, +up).
    pub m: V3,
}

/// One sampled station of a [`Ribbon3d`]. The flat 2D projection lives in
/// `s, x, y, heading, omega_z, width_left, width_right`; the 3D extension adds
/// `z, grade, bank, omega_y, omega_x`.
#[derive(Debug, Clone, Copy)]
pub struct RibbonStation {
    /// Cumulative arc length from the start (m).
    pub s: f64,
    /// World X (m).
    pub x: f64,
    /// World Y (m).
    pub y: f64,
    /// World Z / elevation (m); `0.0` when flat.
    pub z: f64,
    /// Horizontal heading ψ of the tangent (rad, from +X). Matches the 2D
    /// `TrackSegment::heading`.
    pub heading: f64,
    /// Grade / pitch angle θ of the tangent (rad); `0.0` when flat.
    pub grade: f64,
    /// Bank / roll angle φ of the surface about the tangent (rad); `0.0` when flat.
    pub bank: f64,
    /// Roll-rate generalized curvature Ω_x (1/m); `0.0` when flat.
    pub omega_x: f64,
    /// Pitch-rate generalized curvature Ω_y (1/m); `0.0` when flat.
    pub omega_y: f64,
    /// Yaw / horizontal generalized curvature Ω_z (1/m). Equals the 2D
    /// `TrackSegment::curvature`.
    pub omega_z: f64,
    /// Half-width to the left boundary (m).
    pub width_left: f64,
    /// Half-width to the right boundary (m).
    pub width_right: f64,
    /// Grip-scale placeholder (defaults [`DEFAULT_MU_SCALE`]).
    pub mu_scale: f64,
}

/// A 3D curved ribbon: sampled stations plus loop/length metadata. The station
/// query surface mirrors [`Track`] exactly (`locate`, `position_at`,
/// `heading_at`, `curvature_at`, `width_at`).
#[derive(Debug, Clone)]
pub struct Ribbon3d {
    /// Human-readable name.
    pub name: String,
    /// Stations in order of increasing arc length.
    pub stations: Vec<RibbonStation>,
    /// Total arc length (m).
    pub total_length: f64,
    /// `true` if the ribbon forms a closed loop.
    pub is_closed: bool,
}

/// Geometry-validation report from [`Ribbon3d::validate`].
#[derive(Debug, Clone)]
pub struct RibbonValidation {
    /// Number of stations.
    pub n: usize,
    /// Whether the ribbon is closed.
    pub is_closed: bool,
    /// Total 3D arc length (m).
    pub length_3d: f64,
    /// Elevation range `max z − min z` (m).
    pub elevation_range: f64,
    /// Max orthonormality residual over all frames (0 = perfect).
    pub max_ortho_error: f64,
    /// 95th-percentile `|Ω_y|` (pitch-rate) magnitude (1/m).
    pub omega_y_p95: f64,
    /// Max `|Ω_y|` (1/m).
    pub omega_y_max: f64,
    /// 95th-percentile `|Ω_z|` (yaw-curvature) magnitude (1/m).
    pub omega_z_p95: f64,
    /// Whether every stored coordinate/curvature is finite.
    pub all_finite: bool,
}

/// Build the frame from the three road angles (heading ψ, grade θ, bank φ).
///
/// * `t = (cosθ cosψ, cosθ sinψ, sinθ)` — tangent.
/// * `l0 = (−sinψ, cosψ, 0)` — horizontal left (= `ẑ × t` normalized).
/// * `m0 = t × l0` — surface normal before banking.
/// * bank φ rotates `{l0, m0}` about `t`: `n = cosφ·l0 + sinφ·m0`,
///   `m = −sinφ·l0 + cosφ·m0`.
///
/// Flat (θ = φ = 0) gives `t = (cosψ, sinψ, 0)`, `n = (−sinψ, cosψ, 0)`,
/// `m = (0, 0, 1)` — the 2D left-normal convention.
fn frame_from_angles(heading: f64, grade: f64, bank: f64) -> Frame {
    let (sp, cp) = heading.sin_cos();
    let (st, ct) = grade.sin_cos();
    let t = [ct * cp, ct * sp, st];
    let l0 = [-sp, cp, 0.0];
    let m0 = cross(t, l0);
    let (sb, cb) = bank.sin_cos();
    let n = [
        cb * l0[0] + sb * m0[0],
        cb * l0[1] + sb * m0[1],
        cb * l0[2] + sb * m0[2],
    ];
    let m = [
        -sb * l0[0] + cb * m0[0],
        -sb * l0[1] + cb * m0[1],
        -sb * l0[2] + cb * m0[2],
    ];
    Frame { t, n, m }
}

impl Ribbon3d {
    /// The flat (2D-degenerate) ribbon for `track`: copies every segment scalar
    /// **verbatim** so the scalar queries reproduce the 2D queries bitwise.
    /// `z = grade = bank = Ω_x = Ω_y = 0`, `Ω_z = curvature`, `mu_scale = 1`.
    pub fn from_flat(track: &Track) -> Ribbon3d {
        let stations = track
            .segments
            .iter()
            .map(|seg| RibbonStation {
                s: seg.s,
                x: seg.x,
                y: seg.y,
                z: 0.0,
                heading: seg.heading,
                grade: 0.0,
                bank: 0.0,
                omega_x: 0.0,
                omega_y: 0.0,
                omega_z: seg.curvature,
                width_left: seg.width_left,
                width_right: seg.width_right,
                mu_scale: DEFAULT_MU_SCALE,
            })
            .collect();
        Ribbon3d {
            name: track.name.clone(),
            stations,
            total_length: track.total_length,
            is_closed: track.is_closed,
        }
    }

    /// `true` if this ribbon is geometrically flat: every station has
    /// `z = grade = bank = Ω_x = Ω_y = 0`.
    pub fn is_flat(&self) -> bool {
        self.stations.iter().all(|st| {
            st.z == 0.0
                && st.grade == 0.0
                && st.bank == 0.0
                && st.omega_x == 0.0
                && st.omega_y == 0.0
        })
    }

    /// Number of stations.
    pub fn len(&self) -> usize {
        self.stations.len()
    }

    /// Whether the ribbon has no stations.
    pub fn is_empty(&self) -> bool {
        self.stations.is_empty()
    }

    /// Validate the geometry: frame orthonormality, generalized-curvature
    /// finiteness/smoothness, 3D length and elevation range. Used by the
    /// load-and-validate path (real data + the synthetic smoke test).
    pub fn validate(&self) -> RibbonValidation {
        let n = self.stations.len();
        let mut max_ortho = 0.0_f64;
        let mut all_finite = true;
        let mut oy: Vec<f64> = Vec::with_capacity(n);
        let mut oz: Vec<f64> = Vec::with_capacity(n);
        let (mut zmin, mut zmax) = (f64::INFINITY, f64::NEG_INFINITY);
        for st in &self.stations {
            let f = frame_from_angles(st.heading, st.grade, st.bank);
            let e = [
                (norm(f.t) - 1.0).abs(),
                (norm(f.n) - 1.0).abs(),
                (norm(f.m) - 1.0).abs(),
                dot(f.t, f.n).abs(),
                dot(f.t, f.m).abs(),
                dot(f.n, f.m).abs(),
            ];
            for v in e {
                max_ortho = max_ortho.max(v);
            }
            for v in [st.omega_x, st.omega_y, st.omega_z, st.x, st.y, st.z] {
                all_finite &= v.is_finite();
            }
            oy.push(st.omega_y.abs());
            oz.push(st.omega_z.abs());
            zmin = zmin.min(st.z);
            zmax = zmax.max(st.z);
        }
        let p95 = |mut v: Vec<f64>| -> f64 {
            if v.is_empty() {
                return 0.0;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[((0.95 * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)]
        };
        let omega_y_max = oy.iter().cloned().fold(0.0_f64, f64::max);
        RibbonValidation {
            n,
            is_closed: self.is_closed,
            length_3d: self.total_length,
            elevation_range: if n > 0 { zmax - zmin } else { 0.0 },
            max_ortho_error: max_ortho,
            omega_y_p95: p95(oy),
            omega_y_max,
            omega_z_p95: p95(oz),
            all_finite,
        }
    }

    // --- query surface (mirrors `Track`; flat path is bitwise-identical) ---

    /// Segment index and interpolation fraction for arc length `s`. Identical
    /// semantics and arithmetic to [`Track::locate`].
    pub fn locate(&self, s: f64) -> (usize, f64) {
        let n = self.stations.len();
        let s = if self.is_closed {
            s.rem_euclid(self.total_length)
        } else {
            s.clamp(0.0, self.total_length)
        };
        let pp = self.stations.partition_point(|st| st.s <= s);
        let idx = pp.saturating_sub(1);
        let seg_s = self.stations[idx].s;
        let upper_s = if idx + 1 < n {
            self.stations[idx + 1].s
        } else if self.is_closed {
            self.total_length
        } else {
            return (n - 1, 0.0);
        };
        let span = upper_s - seg_s;
        let frac = if span > 0.0 { (s - seg_s) / span } else { 0.0 };
        (idx, frac)
    }

    /// The index following `idx` (wraps on closed ribbons).
    fn upper_index(&self, idx: usize) -> usize {
        let n = self.stations.len();
        if self.is_closed {
            (idx + 1) % n
        } else {
            (idx + 1).min(n - 1)
        }
    }

    /// Linearly interpolated yaw / horizontal curvature Ω_z. Bitwise-identical
    /// to [`Track::curvature_at`] on a flat ribbon.
    pub fn curvature_at(&self, s: f64) -> f64 {
        let (i, f) = self.locate(s);
        let j = self.upper_index(i);
        let a = self.stations[i].omega_z;
        let b = self.stations[j].omega_z;
        a + f * (b - a)
    }

    /// Alias for [`Ribbon3d::curvature_at`] (Ω_z).
    pub fn omega_z_at(&self, s: f64) -> f64 {
        self.curvature_at(s)
    }

    /// Interpolated horizontal heading ψ (angle-wrap aware). Bitwise-identical
    /// to [`Track::heading_at`] on a flat ribbon.
    pub fn heading_at(&self, s: f64) -> f64 {
        let (i, f) = self.locate(s);
        let j = self.upper_index(i);
        let a = self.stations[i].heading;
        let b = self.stations[j].heading;
        let diff = normalize_angle(b - a);
        normalize_angle(a + f * diff)
    }

    /// Interpolated world `(x, y)`. Bitwise-identical to [`Track::position_at`]
    /// on a flat ribbon.
    pub fn position_at(&self, s: f64) -> (f64, f64) {
        let (i, f) = self.locate(s);
        let j = self.upper_index(i);
        let a = &self.stations[i];
        let b = &self.stations[j];
        (a.x + f * (b.x - a.x), a.y + f * (b.y - a.y))
    }

    /// Interpolated world `(x, y, z)`.
    pub fn position3_at(&self, s: f64) -> (f64, f64, f64) {
        let (i, f) = self.locate(s);
        let j = self.upper_index(i);
        let a = &self.stations[i];
        let b = &self.stations[j];
        (
            a.x + f * (b.x - a.x),
            a.y + f * (b.y - a.y),
            a.z + f * (b.z - a.z),
        )
    }

    /// Interpolated elevation `z`.
    pub fn elevation_at(&self, s: f64) -> f64 {
        let (i, f) = self.locate(s);
        let j = self.upper_index(i);
        let a = self.stations[i].z;
        let b = self.stations[j].z;
        a + f * (b - a)
    }

    /// Interpolated grade / pitch angle θ (rad).
    pub fn grade_at(&self, s: f64) -> f64 {
        let (i, f) = self.locate(s);
        let j = self.upper_index(i);
        let a = self.stations[i].grade;
        let b = self.stations[j].grade;
        a + f * (b - a)
    }

    /// Interpolated bank / roll angle φ (rad).
    pub fn bank_at(&self, s: f64) -> f64 {
        let (i, f) = self.locate(s);
        let j = self.upper_index(i);
        let a = self.stations[i].bank;
        let b = self.stations[j].bank;
        a + f * (b - a)
    }

    /// Interpolated half-widths `(width_left, width_right)`. Bitwise-identical
    /// to [`Track::width_at`] on a flat ribbon.
    pub fn width_at(&self, s: f64) -> (f64, f64) {
        let (i, f) = self.locate(s);
        let j = self.upper_index(i);
        let a = &self.stations[i];
        let b = &self.stations[j];
        (
            a.width_left + f * (b.width_left - a.width_left),
            a.width_right + f * (b.width_right - a.width_right),
        )
    }

    /// Interpolated generalized curvatures `(Ω_x, Ω_y, Ω_z)`.
    pub fn omega_at(&self, s: f64) -> V3 {
        let (i, f) = self.locate(s);
        let j = self.upper_index(i);
        let a = &self.stations[i];
        let b = &self.stations[j];
        [
            a.omega_x + f * (b.omega_x - a.omega_x),
            a.omega_y + f * (b.omega_y - a.omega_y),
            a.omega_z + f * (b.omega_z - a.omega_z),
        ]
    }

    /// The orthonormal moving frame at arc length `s`, built from the
    /// interpolated road angles (ψ, θ, φ).
    pub fn frame_at(&self, s: f64) -> Frame {
        frame_from_angles(self.heading_at(s), self.grade_at(s), self.bank_at(s))
    }

    /// Build a 3D ribbon from an analytic / imported centerline: 3D points,
    /// optional per-point bank angle (rad) and half-widths. Headings, grades,
    /// the moving frame, and the Darboux curvatures are computed by central
    /// finite differences of arc length. This is the general path used for
    /// imported 3D data and the analytic frame tests; the flat fast path is
    /// [`Ribbon3d::from_flat`].
    pub fn from_centerline_3d(
        name: &str,
        pts: &[[f64; 3]],
        bank: &[f64],
        width_left: &[f64],
        width_right: &[f64],
        closed: bool,
    ) -> Ribbon3d {
        let n = pts.len();
        assert!(n >= 3, "ribbon needs at least 3 points");
        assert_eq!(bank.len(), n);
        assert_eq!(width_left.len(), n);
        assert_eq!(width_right.len(), n);

        // Cumulative 3D arc length.
        let mut s = vec![0.0; n];
        for i in 1..n {
            let d = [
                pts[i][0] - pts[i - 1][0],
                pts[i][1] - pts[i - 1][1],
                pts[i][2] - pts[i - 1][2],
            ];
            s[i] = s[i - 1] + norm(d);
        }
        let seam = if closed {
            let d = [
                pts[0][0] - pts[n - 1][0],
                pts[0][1] - pts[n - 1][1],
                pts[0][2] - pts[n - 1][2],
            ];
            norm(d)
        } else {
            0.0
        };
        let total_length = if closed { s[n - 1] + seam } else { s[n - 1] };

        // Neighbour indices for central differences (wrap on closed ribbons).
        let nb = |i: usize| -> (usize, usize) {
            if closed {
                ((i + n - 1) % n, (i + 1) % n)
            } else if i == 0 {
                (0, 1)
            } else if i == n - 1 {
                (n - 2, n - 1)
            } else {
                (i - 1, i + 1)
            }
        };
        // Signed arc-length span between neighbours (handles the closed seam).
        let ds_between = |i: usize, j: usize| -> f64 {
            let mut d = s[j] - s[i];
            if closed {
                if d > total_length * 0.5 {
                    d -= total_length;
                } else if d < -total_length * 0.5 {
                    d += total_length;
                }
            }
            d
        };

        // Pass 1: tangent → heading ψ, grade θ; store the frame.
        let mut heading = vec![0.0; n];
        let mut grade = vec![0.0; n];
        let mut frames = vec![
            Frame {
                t: [1.0, 0.0, 0.0],
                n: [0.0, 1.0, 0.0],
                m: [0.0, 0.0, 1.0],
            };
            n
        ];
        for i in 0..n {
            let (lo, hi) = nb(i);
            let d = [
                pts[hi][0] - pts[lo][0],
                pts[hi][1] - pts[lo][1],
                pts[hi][2] - pts[lo][2],
            ];
            let t = normalize(d);
            let psi = t[1].atan2(t[0]);
            let horiz = (t[0] * t[0] + t[1] * t[1]).sqrt();
            let theta = t[2].atan2(horiz);
            heading[i] = psi;
            grade[i] = theta;
            frames[i] = frame_from_angles(psi, theta, bank[i]);
        }

        // Pass 2: Darboux curvatures from central differences of the frame.
        // Ω_z = t'·n, Ω_y = −t'·m, Ω_x = n'·m.
        let mut stations = Vec::with_capacity(n);
        for i in 0..n {
            let (lo, hi) = nb(i);
            let dds = ds_between(lo, hi);
            let inv = if dds.abs() > 0.0 { 1.0 / dds } else { 0.0 };
            let tp = [
                (frames[hi].t[0] - frames[lo].t[0]) * inv,
                (frames[hi].t[1] - frames[lo].t[1]) * inv,
                (frames[hi].t[2] - frames[lo].t[2]) * inv,
            ];
            let np = [
                (frames[hi].n[0] - frames[lo].n[0]) * inv,
                (frames[hi].n[1] - frames[lo].n[1]) * inv,
                (frames[hi].n[2] - frames[lo].n[2]) * inv,
            ];
            let f = frames[i];
            let omega_z = dot(tp, f.n);
            let omega_y = -dot(tp, f.m);
            let omega_x = dot(np, f.m);
            stations.push(RibbonStation {
                s: s[i],
                x: pts[i][0],
                y: pts[i][1],
                z: pts[i][2],
                heading: heading[i],
                grade: grade[i],
                bank: bank[i],
                omega_x,
                omega_y,
                omega_z,
                width_left: width_left[i],
                width_right: width_right[i],
                mu_scale: DEFAULT_MU_SCALE,
            });
        }

        Ribbon3d {
            name: name.to_string(),
            stations,
            total_length,
            is_closed: closed,
        }
    }
}

impl From<&Track> for Ribbon3d {
    fn from(track: &Track) -> Self {
        Ribbon3d::from_flat(track)
    }
}

impl Track {
    /// View this flat 2D track as a [`Ribbon3d`] (flat degenerate case). The
    /// scalar queries on the result are bitwise-identical to this track's.
    pub fn to_ribbon3d(&self) -> Ribbon3d {
        Ribbon3d::from_flat(self)
    }
}

#[cfg(test)]
mod tests {
    // Frame/vector comparisons index fixed 3-vectors by component, which reads
    // clearer as a range loop than an enumerate() (same rationale as the
    // estimator's dense linear-algebra code).
    #![allow(clippy::needless_range_loop)]

    use super::*;
    use crate::builder::build_track;
    use crate::circuits::silverstone_circuit;
    use crate::generators::{circle_track, oval_track};
    use std::f64::consts::PI;

    // ---- flat-case exactness (bitwise) -----------------------------------

    /// Every flat-ribbon scalar query must equal the legacy 2D query bit-for-bit.
    fn assert_flat_exact(track: &Track) {
        let r = track.to_ribbon3d();
        assert_eq!(r.stations.len(), track.segments.len());
        assert_eq!(r.total_length.to_bits(), track.total_length.to_bits());
        assert_eq!(r.is_closed, track.is_closed);

        // Sample densely across the lap, including exact node stations.
        let mut samples: Vec<f64> = (0..=500)
            .map(|k| k as f64 / 500.0 * track.total_length)
            .collect();
        for seg in &track.segments {
            samples.push(seg.s);
        }
        for &s in &samples {
            let (lx, ly) = track.position_at(s);
            let (rx, ry) = r.position_at(s);
            assert_eq!(lx.to_bits(), rx.to_bits(), "x at s={s}");
            assert_eq!(ly.to_bits(), ry.to_bits(), "y at s={s}");
            assert_eq!(
                track.heading_at(s).to_bits(),
                r.heading_at(s).to_bits(),
                "heading at s={s}"
            );
            assert_eq!(
                track.curvature_at(s).to_bits(),
                r.curvature_at(s).to_bits(),
                "curvature at s={s}"
            );
            let (wl, wr) = track.width_at(s);
            let (rwl, rwr) = r.width_at(s);
            assert_eq!(wl.to_bits(), rwl.to_bits(), "wl at s={s}");
            assert_eq!(wr.to_bits(), rwr.to_bits(), "wr at s={s}");
            assert_eq!(track.locate(s), r.locate(s), "locate at s={s}");
        }
    }

    #[test]
    fn flat_exact_circle() {
        let (pts, closed) = circle_track(100.0, 10.0, 200);
        assert_flat_exact(&build_track("circle", &pts, closed));
    }

    #[test]
    fn flat_exact_oval() {
        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 400);
        assert_flat_exact(&build_track("oval", &pts, closed));
    }

    #[test]
    fn flat_exact_silverstone() {
        let (pts, closed) = silverstone_circuit();
        assert_flat_exact(&build_track("silverstone", &pts, closed));
    }

    #[test]
    fn flat_ribbon_reports_flat_and_zero_omega() {
        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 400);
        let r = build_track("oval", &pts, closed).to_ribbon3d();
        assert!(r.is_flat());
        for &s in &[0.0, 123.0, 900.0, r.total_length * 0.5] {
            let om = r.omega_at(s);
            assert_eq!(om[0], 0.0, "Ω_x");
            assert_eq!(om[1], 0.0, "Ω_y");
            // Ω_z equals the 2D curvature bitwise.
            assert_eq!(om[2].to_bits(), r.curvature_at(s).to_bits());
            assert_eq!(r.elevation_at(s), 0.0);
        }
    }

    // ---- frame orthonormality on analytic 3D ribbons ---------------------

    fn assert_orthonormal(f: Frame, tol: f64) {
        assert!((norm(f.t) - 1.0).abs() < tol, "|t| {}", norm(f.t));
        assert!((norm(f.n) - 1.0).abs() < tol, "|n| {}", norm(f.n));
        assert!((norm(f.m) - 1.0).abs() < tol, "|m| {}", norm(f.m));
        assert!(dot(f.t, f.n).abs() < tol, "t·n {}", dot(f.t, f.n));
        assert!(dot(f.t, f.m).abs() < tol, "t·m {}", dot(f.t, f.m));
        assert!(dot(f.n, f.m).abs() < tol, "n·m {}", dot(f.n, f.m));
        // Right-handed: t × n = m.
        let txn = cross(f.t, f.n);
        for k in 0..3 {
            assert!((txn[k] - f.m[k]).abs() < tol, "t×n≠m comp {k}");
        }
    }

    /// A helix `r(u) = (R cos u, R sin u, h·u)` sampled over `turns`.
    fn helix(r: f64, h: f64, turns: f64, n: usize) -> Vec<[f64; 3]> {
        (0..n)
            .map(|i| {
                let u = turns * 2.0 * PI * i as f64 / (n - 1) as f64;
                [r * u.cos(), r * u.sin(), h * u]
            })
            .collect()
    }

    #[test]
    fn helix_frame_is_orthonormal_and_omega_matches_closed_form() {
        let (r, h) = (100.0, 10.0);
        let n = 4000;
        let pts = helix(r, h, 2.0, n);
        let zeros = vec![0.0; n];
        let w = vec![5.0; n];
        let ribbon = Ribbon3d::from_centerline_3d("helix", &pts, &zeros, &w, &w, false);

        // Closed form (horizontal-transport frame, no bank):
        //   L = √(R²+h²);  Ω_x = h/L²,  Ω_y = 0,  Ω_z = R/L².
        let l2 = r * r + h * h;
        let (ox, oy, oz) = (h / l2, 0.0, r / l2);

        // Skip the one-sided-difference endpoints; the interior is exact-ish.
        for i in 20..(n - 20) {
            let st = ribbon.stations[i];
            assert_orthonormal(frame_from_angles(st.heading, st.grade, st.bank), 1e-12);
            assert!(
                (st.omega_x - ox).abs() < 5e-6,
                "Ω_x {} vs {}",
                st.omega_x,
                ox
            );
            assert!(
                (st.omega_y - oy).abs() < 5e-6,
                "Ω_y {} vs {}",
                st.omega_y,
                oy
            );
            assert!(
                (st.omega_z - oz).abs() < 5e-6,
                "Ω_z {} vs {}",
                st.omega_z,
                oz
            );
        }
    }

    /// A flat circle of radius `R` in the `z = 0` plane with a **constant** bank
    /// angle β. Closed form (rolled horizontal frame): Ω = (0, sinβ/R, cosβ/R).
    #[test]
    fn banked_circle_frame_and_omega() {
        let (radius, beta) = (80.0, 0.15_f64);
        let n = 3000;
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let u = 2.0 * PI * i as f64 / n as f64;
                [radius * u.cos(), radius * u.sin(), 0.0]
            })
            .collect();
        let banks = vec![beta; n];
        let w = vec![6.0; n];
        let ribbon = Ribbon3d::from_centerline_3d("banked", &pts, &banks, &w, &w, true);

        let (ox, oy, oz) = (0.0, beta.sin() / radius, beta.cos() / radius);
        for st in &ribbon.stations {
            assert_orthonormal(frame_from_angles(st.heading, st.grade, st.bank), 1e-12);
            assert!(
                (st.omega_x - ox).abs() < 5e-6,
                "Ω_x {} vs {}",
                st.omega_x,
                ox
            );
            assert!(
                (st.omega_y - oy).abs() < 5e-6,
                "Ω_y {} vs {}",
                st.omega_y,
                oy
            );
            assert!(
                (st.omega_z - oz).abs() < 5e-6,
                "Ω_z {} vs {}",
                st.omega_z,
                oz
            );
        }
        // The stored bank is retrievable and the surface normal is tilted.
        assert!((ribbon.bank_at(ribbon.total_length * 0.3) - beta).abs() < 1e-9);
        let f = ribbon.frame_at(ribbon.total_length * 0.3);
        assert!(f.m[2] < 1.0, "banked normal must tilt off vertical");
    }

    /// C1 continuity of the frame + curvatures across the closed-loop seam.
    #[test]
    fn frame_is_c1_continuous_across_seam() {
        let (radius, beta) = (80.0, 0.15_f64);
        let n = 3000;
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let u = 2.0 * PI * i as f64 / n as f64;
                [radius * u.cos(), radius * u.sin(), 0.0]
            })
            .collect();
        let banks = vec![beta; n];
        let w = vec![6.0; n];
        let ribbon = Ribbon3d::from_centerline_3d("banked", &pts, &banks, &w, &w, true);

        let l = ribbon.total_length;
        let delta = l * 1e-4;
        // C0: position + frame just before and after the seam agree.
        let before = ribbon.frame_at(l - delta);
        let after = ribbon.frame_at(delta);
        for k in 0..3 {
            assert!((before.t[k] - after.t[k]).abs() < 1e-2, "t seam comp {k}");
            assert!((before.n[k] - after.n[k]).abs() < 1e-2, "n seam comp {k}");
            assert!((before.m[k] - after.m[k]).abs() < 1e-2, "m seam comp {k}");
        }
        // C1: curvatures continuous across the seam.
        let ob = ribbon.omega_at(l - delta);
        let oa = ribbon.omega_at(delta);
        for k in 0..3 {
            assert!((ob[k] - oa[k]).abs() < 1e-3, "Ω seam comp {k}");
        }
    }
}
