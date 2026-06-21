//! Random closed-track generation from spline control points, plus ML feature
//! extraction. Useful for producing diverse synthetic training data.

/// Simple deterministic hash for pseudo-random sampling (no rand dependency).
/// Returns a value in [0.0, 1.0).
fn simple_hash(seed: u64, i: u64, j: u64) -> f64 {
    let mut h = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    h = h.wrapping_add(i.wrapping_mul(2862933555777941757));
    h = h.wrapping_add(j.wrapping_mul(3037000493));
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    (h as f64) / (u64::MAX as f64)
}

/// One coordinate of a Catmull-Rom segment at parameter `t` in [0, 1].
pub(crate) fn catmull_rom(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * (2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// Signed area of triangle (a, b, c); >0 = counter-clockwise.
fn orient(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// Whether segments (p1,p2) and (p3,p4) properly cross (excluding endpoint
/// touches and collinear overlap).
fn segments_cross(p1: (f64, f64), p2: (f64, f64), p3: (f64, f64), p4: (f64, f64)) -> bool {
    let d1 = orient(p3, p4, p1);
    let d2 = orient(p3, p4, p2);
    let d3 = orient(p1, p2, p3);
    let d4 = orient(p1, p2, p4);
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

/// Whether the closed control polygon self-intersects (any pair of
/// non-adjacent edges crosses).
fn polygon_self_intersects(pts: &[(f64, f64)]) -> bool {
    let n = pts.len();
    if n < 4 {
        return false;
    }
    for i in 0..n {
        let a1 = pts[i];
        let a2 = pts[(i + 1) % n];
        for j in (i + 2)..n {
            // Skip the wrap-around adjacency between edge 0 and edge n-1.
            if i == 0 && j == n - 1 {
                continue;
            }
            let b1 = pts[j];
            let b2 = pts[(j + 1) % n];
            if segments_cross(a1, a2, b1, b2) {
                return true;
            }
        }
    }
    false
}

/// Generate a random closed track from spline control points.
///
/// Places n_points on a rough circle of given radius, then perturbs each
/// point radially and angularly. A smooth closed cubic spline through the
/// points gives a realistic track shape. Self-intersecting tracks are
/// rejected and regenerated.
///
/// Arguments:
/// - n_points: number of control points (6-12 typical)
/// - base_radius: average distance from center to control points (m)
/// - radial_variance: perturbation range as fraction of base_radius (0.0-0.5)
/// - angular_variance: perturbation range for angular position (rad, 0.0-0.3)
/// - width: track width (m)
/// - seed: deterministic random seed
/// - n_samples: number of output track points (200-500)
///
/// Returns (points, closed=true) or an error if no valid track found in 100 attempts.
#[allow(clippy::too_many_arguments)]
pub fn random_spline_track(
    n_points: usize,
    base_radius: f64,
    radial_variance: f64,
    angular_variance: f64,
    width: f64,
    seed: u64,
    n_samples: usize,
) -> Result<(Vec<super::TrackPoint>, bool), String> {
    let half = width / 2.0;
    let mut s = seed;

    for _ in 0..100 {
        // (a) perturbed-circle control points
        let ctrl: Vec<(f64, f64)> = (0..n_points)
            .map(|i| {
                let base_angle = 2.0 * std::f64::consts::PI * i as f64 / n_points as f64;
                let theta = base_angle + angular_variance * simple_hash(s, i as u64, 0);
                let r = base_radius
                    + base_radius * radial_variance * (2.0 * simple_hash(s, i as u64, 1) - 1.0);
                (r * theta.cos(), r * theta.sin())
            })
            .collect();

        // (c) reject tangled control polygons and retry with the next seed
        if polygon_self_intersects(&ctrl) {
            s = s.wrapping_add(1);
            continue;
        }

        // (b) sample the closed Catmull-Rom spline at exactly n_samples points
        let mut points = Vec::with_capacity(n_samples);
        for k in 0..n_samples {
            let u = n_points as f64 * k as f64 / n_samples as f64; // in [0, n_points)
            let floor = u.floor();
            let seg = (floor as usize) % n_points;
            let t = u - floor;
            let p0 = ctrl[(seg + n_points - 1) % n_points];
            let p1 = ctrl[seg];
            let p2 = ctrl[(seg + 1) % n_points];
            let p3 = ctrl[(seg + 2) % n_points];
            // (d) sampled point with the requested width on each side
            points.push(super::TrackPoint {
                x: catmull_rom(p0.0, p1.0, p2.0, p3.0, t),
                y: catmull_rom(p0.1, p1.1, p2.1, p3.1, t),
                width_left: half,
                width_right: half,
            });
        }

        // (e) closed track
        return Ok((points, true));
    }

    Err(format!(
        "no valid track found in 100 attempts (seed {seed})"
    ))
}

/// Generate a batch of random tracks with varied parameters.
///
/// Sweeps across different sizes, shapes, and complexities to produce
/// a diverse training set.
pub fn generate_track_batch(count: usize, base_seed: u64) -> Vec<(Vec<super::TrackPoint>, bool)> {
    let mut tracks = Vec::with_capacity(count);

    for i in 0..count {
        let seed = base_seed + i as u64;
        // Vary parameters across the batch
        let n_points = 6 + (i % 7); // 6 to 12 control points
        let base_radius = 100.0 + 400.0 * simple_hash(seed, i as u64, 100); // 100-500m
        let radial_var = 0.1 + 0.3 * simple_hash(seed, i as u64, 101); // 0.1-0.4
        let angular_var = 0.05 + 0.2 * simple_hash(seed, i as u64, 102); // 0.05-0.25
        let width = 8.0 + 8.0 * simple_hash(seed, i as u64, 103); // 8-16m

        if let Ok(track) = random_spline_track(
            n_points,
            base_radius,
            radial_var,
            angular_var,
            width,
            seed,
            300,
        ) {
            tracks.push(track);
        }
    }

    tracks
}

/// ML-ready features extracted from a track at evenly-spaced points.
///
/// Features per point: curvature, curvature derivative, width_left, width_right.
/// All values are normalized to approximately [-1, 1] or [0, 1].
pub struct TrackFeatures {
    /// Number of sample points.
    pub n_points: usize,
    /// Curvature at each point (1/m), normalized by max absolute curvature.
    pub curvature: Vec<f64>,
    /// Curvature derivative at each point (1/m^2), normalized.
    pub curvature_deriv: Vec<f64>,
    /// Left width at each point, normalized by max width.
    pub width_left: Vec<f64>,
    /// Right width at each point, normalized by max width.
    pub width_right: Vec<f64>,
    /// Normalization factor for denormalizing curvature.
    pub max_curvature: f64,
    /// Normalization factor for denormalizing widths.
    pub max_width: f64,
    /// Track total length (m).
    pub track_length: f64,
}

/// Extract ML-ready features from a track at `n_points` evenly-spaced points.
pub fn extract_features(track: &super::Track, n_points: usize) -> TrackFeatures {
    let ds = track.total_length / n_points as f64;

    let mut curvature = Vec::with_capacity(n_points);
    let mut curvature_deriv = Vec::with_capacity(n_points);
    let mut width_left = Vec::with_capacity(n_points);
    let mut width_right = Vec::with_capacity(n_points);

    // Sample raw values
    for i in 0..n_points {
        let s = i as f64 * ds;
        curvature.push(track.curvature_at(s));
        let (wl, wr) = track.width_at(s);
        width_left.push(wl);
        width_right.push(wr);
    }

    // Compute curvature derivative via finite differences
    for i in 0..n_points {
        let prev = if i == 0 { n_points - 1 } else { i - 1 };
        let next = if i == n_points - 1 { 0 } else { i + 1 };
        curvature_deriv.push((curvature[next] - curvature[prev]) / (2.0 * ds));
    }

    // Normalize
    let max_curv = curvature
        .iter()
        .map(|c| c.abs())
        .fold(0.0f64, f64::max)
        .max(1e-6);
    let max_curv_deriv = curvature_deriv
        .iter()
        .map(|c| c.abs())
        .fold(0.0f64, f64::max)
        .max(1e-6);
    let max_w = width_left
        .iter()
        .chain(width_right.iter())
        .fold(0.0f64, |a, &b| a.max(b))
        .max(1e-6);

    for c in &mut curvature {
        *c /= max_curv;
    }
    for c in &mut curvature_deriv {
        *c /= max_curv_deriv;
    }
    for w in &mut width_left {
        *w /= max_w;
    }
    for w in &mut width_right {
        *w /= max_w;
    }

    TrackFeatures {
        n_points,
        curvature,
        curvature_deriv,
        width_left,
        width_right,
        max_curvature: max_curv,
        max_width: max_w,
        track_length: track.total_length,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_track, circle_track, oval_track};

    // (a) Single random track generation.
    #[test]
    fn single_track_is_valid() {
        let (points, closed) = random_spline_track(8, 200.0, 0.2, 0.1, 12.0, 42, 300).unwrap();
        assert!(closed);
        assert_eq!(points.len(), 300, "should produce n_samples points");

        let track = build_track("rand", &points, true);
        assert!(track.total_length > 0.0);
        // The perimeter is ~2*pi*base_radius (~6.3x); allow a wide proportional band.
        assert!(
            track.total_length > 2.0 * 200.0 && track.total_length < 12.0 * 200.0,
            "length {} not proportional to base_radius",
            track.total_length
        );
    }

    // (b) Determinism.
    #[test]
    fn deterministic_generation() {
        let a = random_spline_track(8, 200.0, 0.2, 0.1, 12.0, 42, 300).unwrap();
        let b = random_spline_track(8, 200.0, 0.2, 0.1, 12.0, 42, 300).unwrap();
        assert_eq!(a.0.len(), b.0.len());
        assert!(
            a.0.iter()
                .zip(b.0.iter())
                .all(|(p, q)| p.x == q.x && p.y == q.y),
            "same seed must reproduce identical tracks"
        );

        let c = random_spline_track(8, 200.0, 0.2, 0.1, 12.0, 99, 300).unwrap();
        assert!(
            a.0.iter()
                .zip(c.0.iter())
                .any(|(p, q)| p.x != q.x || p.y != q.y),
            "different seeds must differ"
        );
    }

    // (c) Generated tracks are well-formed.
    #[test]
    fn generated_tracks_are_well_formed() {
        for seed in 0..50u64 {
            let n_points = 6 + (seed as usize % 7);
            let (points, _) =
                random_spline_track(n_points, 150.0 + seed as f64, 0.25, 0.15, 12.0, seed, 250)
                    .unwrap();
            let track = build_track("t", &points, true);
            assert!(track.total_length > 0.0, "seed {seed}: zero length");
            for seg in &track.segments {
                assert!(seg.curvature.is_finite(), "seed {seed}: NaN curvature");
            }
        }
    }

    // (d) Batch generation.
    #[test]
    fn batch_generation() {
        let batch = generate_track_batch(20, 1000);
        assert!(
            batch.len() >= 15,
            "batch produced only {} tracks",
            batch.len()
        );

        let lengths: Vec<f64> = batch
            .iter()
            .map(|(pts, closed)| build_track("b", pts, *closed).total_length)
            .collect();
        // Variety: no two tracks share the same length.
        for i in 0..lengths.len() {
            for j in (i + 1)..lengths.len() {
                assert!(
                    (lengths[i] - lengths[j]).abs() > 1e-6,
                    "tracks {i} and {j} have identical length"
                );
            }
        }
    }

    // (e) Feature extraction on an oval.
    #[test]
    fn features_on_oval() {
        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 600);
        let track = build_track("oval", &pts, closed);
        let feat = extract_features(&track, 100);

        assert_eq!(feat.n_points, 100);
        assert_eq!(feat.curvature.len(), 100);
        assert!(feat.curvature.iter().all(|&c| (-1.0..=1.0).contains(&c)));
        assert!(feat.width_left.iter().all(|&w| (0.0..=1.0).contains(&w)));
        assert!(feat.width_right.iter().all(|&w| (0.0..=1.0).contains(&w)));
        // Straight<->corner transitions produce sharp curvature-derivative peaks.
        assert!(
            feat.curvature_deriv.iter().any(|&d| d.abs() > 0.5),
            "expected curvature-derivative peaks at corners"
        );
    }

    // (f) Feature extraction on a circle (constant curvature).
    #[test]
    fn features_on_circle() {
        let (pts, closed) = circle_track(100.0, 12.0, 400);
        let track = build_track("circle", &pts, closed);
        let feat = extract_features(&track, 100);

        // Curvature is constant, so normalized values are all ~equal (all ~+-1).
        assert!(
            feat.curvature.iter().all(|&c| (c.abs() - 1.0).abs() < 0.05),
            "circle curvature should be uniform"
        );
        // Constant curvature -> derivative ~0 (the 1e-6 floor keeps noise from
        // being amplified by normalization).
        assert!(
            feat.curvature_deriv.iter().all(|&d| d.abs() < 0.05),
            "circle curvature derivative should be ~0"
        );
    }

    // (g) Parameter variation across a batch.
    #[test]
    fn batch_has_variety() {
        let batch = generate_track_batch(20, 7000);
        assert!(batch.len() >= 15);

        // All tracks are resampled to the same point count, so variety shows up in
        // length and shape (max curvature), not segment count.
        let lengths: Vec<f64> = batch
            .iter()
            .map(|(pts, closed)| build_track("b", pts, *closed).total_length)
            .collect();
        let max_len = lengths.iter().cloned().fold(f64::MIN, f64::max);
        let min_len = lengths.iter().cloned().fold(f64::MAX, f64::min);
        assert!(max_len > min_len * 1.5, "lengths should vary widely");

        let max_curvs: Vec<f64> = batch
            .iter()
            .map(|(pts, closed)| {
                extract_features(&build_track("b", pts, *closed), 100).max_curvature
            })
            .collect();
        let cmax = max_curvs.iter().cloned().fold(f64::MIN, f64::max);
        let cmin = max_curvs.iter().cloned().fold(f64::MAX, f64::min);
        assert!(cmax > cmin, "corner sharpness should vary across the batch");
    }
}
