//! Hand-crafted real-circuit layouts (Silverstone, Monza).
//!
//! Each circuit is described as a sequence of geometric primitives — straights
//! and constant-radius arcs — that capture the essential corners, straights,
//! and overall lap shape. The primitives are sampled into [`TrackPoint`]s; the
//! corner radii and segment lengths are chosen so the resulting loop has
//! realistic geometry and closes on itself.

use std::f64::consts::PI;

use crate::types::TrackPoint;

/// A single layout primitive.
enum Seg {
    /// A straight of the given length (m).
    Straight(f64),
    /// A constant-radius arc: `(radius_m, signed_angle_deg)`.
    /// Positive angle turns left (CCW), negative turns right (CW).
    Arc(f64, f64),
}

/// Half-width applied to every generated point (12 m total track width).
const HALF_WIDTH: f64 = 6.0;

/// Approximate spacing between sampled points (m).
const SPACING: f64 = 12.0;

fn point(x: f64, y: f64) -> TrackPoint {
    TrackPoint {
        x,
        y,
        width_left: HALF_WIDTH,
        width_right: HALF_WIDTH,
    }
}

/// Walk a sequence of primitives starting at the origin heading north (+Y),
/// sampling points roughly every [`SPACING`] meters.
fn trace(moves: &[Seg]) -> Vec<TrackPoint> {
    let mut x = 0.0_f64;
    let mut y = 0.0_f64;
    let mut h = PI / 2.0; // heading north
    let mut pts = vec![point(x, y)];

    for m in moves {
        match *m {
            Seg::Straight(len) => {
                let steps = (len / SPACING).ceil().max(1.0) as usize;
                let sl = len / steps as f64;
                for _ in 0..steps {
                    x += sl * h.cos();
                    y += sl * h.sin();
                    pts.push(point(x, y));
                }
            }
            Seg::Arc(radius, angle_deg) => {
                let angle = angle_deg * PI / 180.0;
                let len = radius * angle.abs();
                let steps = (len / SPACING).ceil().max(1.0) as usize;
                let sl = len / steps as f64;
                let dth = angle / steps as f64;
                for _ in 0..steps {
                    // midpoint stepping reduces position drift along the arc
                    h += 0.5 * dth;
                    x += sl * h.cos();
                    y += sl * h.sin();
                    h += 0.5 * dth;
                    pts.push(point(x, y));
                }
            }
        }
    }
    pts
}

/// Simulate the move list (no sampling) and return the endpoint plus the
/// heading at the start of each move. Uses the same stepping as [`trace`] so
/// the endpoint matches what `trace` produces.
fn simulate(moves: &[Seg]) -> (f64, f64, Vec<f64>) {
    let mut x = 0.0_f64;
    let mut y = 0.0_f64;
    let mut h = PI / 2.0;
    let mut start_headings = Vec::with_capacity(moves.len());

    for m in moves {
        start_headings.push(h);
        match *m {
            Seg::Straight(len) => {
                x += len * h.cos();
                y += len * h.sin();
            }
            Seg::Arc(radius, angle_deg) => {
                let angle = angle_deg * PI / 180.0;
                let len = radius * angle.abs();
                let steps = (len / SPACING).ceil().max(1.0) as usize;
                let sl = len / steps as f64;
                let dth = angle / steps as f64;
                for _ in 0..steps {
                    h += 0.5 * dth;
                    x += sl * h.cos();
                    y += sl * h.sin();
                    h += 0.5 * dth;
                }
            }
        }
    }
    (x, y, start_headings)
}

/// Close a loop by solving for the lengths of two straights (at indices `ia`
/// and `ib`) such that the path returns exactly to the origin.
///
/// A straight's heading is fixed by the preceding arcs, so the endpoint is an
/// affine function of the two straight lengths — a 2×2 linear solve. The move
/// list's total turning must be a full 360° so the closing headings align.
fn build_closed(mut moves: Vec<Seg>, ia: usize, ib: usize) -> Vec<TrackPoint> {
    // Zero the two adjustable straights, then find the residual endpoint.
    moves[ia] = Seg::Straight(0.0);
    moves[ib] = Seg::Straight(0.0);
    let (ex, ey, headings) = simulate(&moves);
    let ha = headings[ia];
    let hb = headings[ib];

    // Solve a·û_a + b·û_b = -(ex, ey).
    let det = (hb - ha).sin();
    let a = (-ex * hb.sin() + ey * hb.cos()) / det;
    let b = (ex * ha.sin() - ey * ha.cos()) / det;

    moves[ia] = Seg::Straight(a.max(0.0));
    moves[ib] = Seg::Straight(b.max(0.0));
    trace(&moves)
}

/// Total closed-loop length of a point list (consecutive distances plus the
/// wrap from the last point back to the first).
fn closed_length(pts: &[TrackPoint]) -> f64 {
    let n = pts.len();
    let mut total = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        total += ((pts[j].x - pts[i].x).powi(2) + (pts[j].y - pts[i].y).powi(2)).sqrt();
    }
    total
}

/// Uniformly scale all points about the origin so the closed-loop length
/// matches `target` (m). Scaling preserves closure and keeps corner-radius
/// proportions.
fn scale_to_length(mut pts: Vec<TrackPoint>, target: f64) -> Vec<TrackPoint> {
    let current = closed_length(&pts);
    if current <= 0.0 {
        return pts;
    }
    let f = target / current;
    for p in &mut pts {
        p.x *= f;
        p.y *= f;
        // widths are physical and stay fixed regardless of layout scale
    }
    pts
}

/// Generate a simplified Silverstone Circuit layout.
///
/// This captures the essential geometry: the fast sweeps (Maggotts/Becketts),
/// the heavy braking zones (Stowe, Village), the long straights (Hangar, Wellington),
/// and the overall lap shape. The track width is set to 12m throughout.
///
/// Returns (points, closed=true).
/// The Silverstone layout primitives (before closure and scaling).
fn silverstone_moves() -> Vec<Seg> {
    use Seg::*;
    vec![
        Straight(300.0),   // 0  start/finish + National straight
        Arc(200.0, -66.0), // 1  Copse (fast right)
        Straight(90.0),    // 2
        Arc(150.0, 45.0),  // 3  Maggotts (left)
        Arc(120.0, -55.0), // 4  Becketts (right)
        Arc(110.0, 50.0),  // 5  Becketts (left)
        Arc(130.0, -40.0), // 6  Chapel (right)
        Straight(770.0),   // 7  Hangar Straight  (closure knob A)
        Arc(80.0, -120.0), // 8  Stowe (heavy braking right)
        Straight(110.0),   // 9
        Arc(70.0, -70.0),  // 10 Vale (right)
        Arc(90.0, 45.0),   // 11 Club (left)
        Straight(120.0),   // 12
        Arc(180.0, -60.0), // 13 Abbey (fast right)
        Straight(60.0),    // 14
        Arc(150.0, 35.0),  // 15 Farm (left)
        Straight(120.0),   // 16
        Arc(40.0, -130.0), // 17 Village (sharp right hairpin)
        Arc(45.0, 135.0),  // 18 The Loop (tight left)
        Straight(60.0),    // 19
        Arc(90.0, -55.0),  // 20 Aintree (right)
        Straight(560.0),   // 21 Wellington Straight (closure knob B)
        Arc(75.0, 95.0),   // 22 Brooklands (left)
        Arc(50.0, -139.0), // 23 Luffield (slow right)
        Arc(160.0, -30.0), // 24 Woodcote (right kink)
        Straight(80.0),    // 25 onto pit straight
    ]
}

/// Generate a simplified Silverstone Circuit layout.
///
/// Captures the essential geometry of the modern Grand Prix circuit: fast sweeps
/// (Maggotts/Becketts), heavy braking zones (Stowe, Village), and long straights
/// (Hangar, Wellington). Track width is 12m throughout.
///
/// Returns `(points, closed=true)`. Track length is approximately 5.89 km.
pub fn silverstone_circuit() -> (Vec<TrackPoint>, bool) {
    // Close the loop on the Hangar straight and the following connector, then
    // scale to the real lap length (~1.07× — corner radii stay near design).
    let pts = build_closed(silverstone_moves(), 7, 9);
    (scale_to_length(pts, 5891.0), true)
}

// The Monza layout primitives (before closure and scaling).
fn monza_moves() -> Vec<Seg> {
    use Seg::*;
    vec![
        Straight(1100.0),   // 0  Rettifilo (main straight)   (closure knob A)
        Arc(70.0, -70.0),   // 1  Variante del Rettifilo (right)
        Arc(70.0, 70.0),    // 2  and left — first chicane
        Arc(260.0, -70.0),  // 3  Curva Grande (long right)
        Straight(300.0),    // 4
        Arc(70.0, 65.0),    // 5  Variante della Roggia (left)
        Arc(70.0, -65.0),   // 6  and right — second chicane
        Straight(260.0),    // 7
        Arc(150.0, -90.0),  // 8  Lesmo 1 (right)
        Straight(120.0),    // 9
        Arc(140.0, -95.0),  // 10 Lesmo 2 (right)
        Straight(900.0),    // 11 back straight (closure knob B)
        Arc(90.0, 55.0),    // 12 Ascari (left)
        Arc(80.0, -75.0),   // 13 Ascari (right)
        Arc(100.0, 95.0),   // 14 Ascari (left)
        Straight(620.0),    // 15 straight to Parabolica
        Arc(170.0, -180.0), // 16 Parabolica (long right onto main straight)
        Straight(120.0),    // 17
    ]
}

/// Generate a simplified Monza circuit layout.
///
/// A power circuit with long straights, heavy braking chicanes, the two Lesmo
/// corners, Ascari chicane, and Parabolica. Track width is 12m throughout.
///
/// Returns `(points, closed=true)`. Track length is approximately 5.79 km.
pub fn monza_circuit() -> (Vec<TrackPoint>, bool) {
    // Close on the main straight and the post-Lesmo straight, then scale to the
    // real lap length (~0.94× — keeps the chicanes tight).
    let pts = build_closed(monza_moves(), 0, 9);
    (scale_to_length(pts, 5793.0), true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::build_track;

    fn max_abs_curvature(track: &crate::types::Track) -> f64 {
        track
            .segments
            .iter()
            .map(|s| s.curvature.abs())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn silverstone_geometry() {
        let (pts, closed) = silverstone_circuit();
        assert!(closed);
        let track = build_track("Silverstone", &pts, closed);
        assert!(track.is_closed);
        // total length within ~10% of the real 5.89 km
        assert!(
            (5300.0..=6500.0).contains(&track.total_length),
            "length {} out of range",
            track.total_length
        );
        // real corners present
        assert!(
            max_abs_curvature(&track) > 0.005,
            "max curvature {} too low — no corners?",
            max_abs_curvature(&track)
        );
    }

    #[test]
    fn monza_geometry() {
        let (pts, closed) = monza_circuit();
        assert!(closed);
        let track = build_track("Monza", &pts, closed);
        assert!(track.is_closed);
        assert!(
            (5200.0..=6400.0).contains(&track.total_length),
            "length {} out of range",
            track.total_length
        );
        assert!(
            max_abs_curvature(&track) > 0.005,
            "max curvature {} too low — no corners?",
            max_abs_curvature(&track)
        );
    }
}
