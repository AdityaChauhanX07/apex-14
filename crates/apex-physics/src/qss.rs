//! Quasi-Steady-State (QSS) lap simulator.
//!
//! Computes the maximum achievable speed at every track segment, subject to
//! three constraints applied in sequence — the cornering (lateral grip) limit,
//! a forward acceleration pass, and a backward braking pass — then integrates
//! `ds/v` for the lap time.

use apex_track::Track;

use crate::car_params::{CarParams, GRAVITY};
use crate::tire::PacejkaTire;

/// Speed cap used where the track is effectively straight or downforce alone
/// supplies unlimited cornering grip (720 km/h).
const V_CAP: f64 = 200.0;

/// Lower bound applied to any computed speed to avoid singularities (m/s).
const V_FLOOR: f64 = 0.1;

/// Number of sectors a lap is split into when the track carries no explicit
/// sector markers — the classic three-sector split used on real circuits.
/// `Track` has no sector-marker field today, so this is always the split used;
/// [`integrate_lap_and_sectors`] already takes `n_sectors` so honoring
/// per-track markers later is a call-site change, not a rewrite.
pub const DEFAULT_SECTOR_COUNT: usize = 3;

/// Result of a QSS lap simulation.
pub struct QssResult {
    /// Speed at each track segment (m/s).
    pub speeds: Vec<f64>,
    /// Total lap time (seconds).
    pub lap_time: f64,
    /// Arc length `s` at each segment (m).
    pub distances: Vec<f64>,
    /// Lateral acceleration at each segment (m/s²).
    pub lateral_gs: Vec<f64>,
    /// Longitudinal acceleration at each segment (m/s²).
    pub longitudinal_gs: Vec<f64>,
    /// Per-sector times (seconds). With no explicit sector markers the lap is
    /// split into [`DEFAULT_SECTOR_COUNT`] equal-arc-length sectors. Sums to
    /// [`lap_time`](Self::lap_time) up to floating-point reassociation (well
    /// within 1e-9 s), so telemetry/the viewer can display sector splits
    /// without recomputing the lap integral.
    pub sector_times: Vec<f64>,
}

/// Splits per-interval traversal times into equal-arc-length sector totals —
/// the single definition of "how a lap is divided into sectors" for the whole
/// workspace (QSS here, and the collocation-optimize golden, both consume it).
///
/// `stations[i]` is the arc length at node `i`; `interval_times[i]` is the time
/// spent on the interval that *starts* at node `i` (so there are
/// `interval_times.len()` intervals — `stations.len()` for a closed loop whose
/// final interval wraps to the start, or `stations.len() - 1` otherwise). Each
/// interval's time is attributed in full to the sector containing that
/// interval's midpoint station. Because every interval lands in exactly one
/// bucket, the returned times sum to `interval_times.iter().sum()` up to
/// floating-point reassociation, regardless of how many nodes straddle a
/// boundary — attributing whole intervals (rather than splitting the one that
/// crosses a boundary) is what keeps that identity exact, at the cost of
/// placing each boundary at the nearest interval edge.
pub fn sector_times(
    stations: &[f64],
    interval_times: &[f64],
    total_length: f64,
    n_sectors: usize,
) -> Vec<f64> {
    let n = stations.len();
    let sector_len = total_length / n_sectors as f64;
    let mut sectors = vec![0.0; n_sectors];
    for (i, &dt) in interval_times.iter().enumerate() {
        let ds = if i + 1 < n {
            stations[i + 1] - stations[i]
        } else {
            total_length - stations[n - 1]
        };
        let s_mid = stations[i] + 0.5 * ds;
        let idx = ((s_mid / sector_len).floor() as usize).min(n_sectors - 1);
        sectors[idx] += dt;
    }
    sectors
}

/// Integrates lap time and per-sector times from a completed speed profile.
///
/// Builds the same `ds / v_avg` per-interval times as the lap-time integral,
/// then buckets them via [`sector_times`]. `lap_time` is `Σ dt` in interval
/// order (byte-identical to the previous direct accumulation), and the sector
/// split reuses the shared definition — so `sector_times.iter().sum()` equals
/// `lap_time` up to floating-point reassociation, asserted within 1e-9 s by
/// the unit test `sector_times_sum_to_lap_time`.
fn integrate_lap_and_sectors(
    s: &[f64],
    speeds: &[f64],
    total_length: f64,
    closed: bool,
    n_sectors: usize,
) -> (f64, Vec<f64>) {
    let n = s.len();
    let intervals = if closed { n } else { n - 1 };
    let mut dt = Vec::with_capacity(intervals);
    for i in 0..intervals {
        let j = if i + 1 < n { i + 1 } else { 0 };
        let ds = if i + 1 < n {
            s[i + 1] - s[i]
        } else {
            total_length - s[n - 1]
        };
        let v_avg = 0.5 * (speeds[i] + speeds[j]);
        dt.push(if v_avg > 0.0 { ds / v_avg } else { 0.0 });
    }
    let lap_time = dt.iter().sum();
    let sectors = sector_times(s, &dt, total_length, n_sectors);
    (lap_time, sectors)
}

/// Cornering-limited speed at a segment with absolute curvature `kappa`.
fn cornering_speed(params: &CarParams, kappa: f64) -> f64 {
    if kappa < 1e-9 {
        return V_CAP;
    }
    // m·v²·|κ| = μ·(m·g + 0.5·ρ·C_l·A·v²)
    let denom = params.mass * kappa
        - params.tire_mu * 0.5 * params.air_density * params.lift_coeff * params.frontal_area;
    if denom <= 0.0 {
        return V_CAP;
    }
    let v2 = params.tire_mu * params.mass * GRAVITY / denom;
    v2.sqrt().min(V_CAP)
}

/// Maximum speed reachable one step ahead, given the current speed `v` and the
/// curvature `kappa` used for the lateral load there.
fn forward_speed(params: &CarParams, v: f64, kappa: f64, ds: f64) -> f64 {
    let fg = params.max_grip_force(v);
    let fl = params.mass * v * v * kappa;
    let f_lon_max = if fl >= fg {
        0.0
    } else {
        (fg * fg - fl * fl).sqrt()
    };
    let f_accel = f_lon_max.min(params.max_drive_force)
        - params.drag_force(v)
        - params.rolling_resistance_force();
    let a = f_accel / params.mass;
    let v_next_sq = v * v + 2.0 * a * ds;
    v_next_sq.max(V_FLOOR * V_FLOOR).sqrt()
}

/// Maximum speed permissible one step *behind*, such that the car can still
/// brake down to `v` over `ds`. Drag and rolling resistance aid the
/// deceleration.
fn backward_speed(params: &CarParams, v: f64, kappa: f64, ds: f64) -> f64 {
    let fg = params.max_grip_force(v);
    let fl = params.mass * v * v * kappa;
    let f_lon_max = if fl >= fg {
        0.0
    } else {
        (fg * fg - fl * fl).sqrt()
    };
    let a_decel = (f_lon_max.min(params.max_brake_force)
        + params.drag_force(v)
        + params.rolling_resistance_force())
        / params.mass;
    let v_prev_sq = v * v + 2.0 * a_decel * ds;
    v_prev_sq.max(V_FLOOR * V_FLOOR).sqrt()
}

/// Runs the QSS lap simulation for a track and car.
///
/// For closed tracks the forward and backward passes are each run twice so the
/// constraints propagate across the start/finish line. For open tracks the
/// simulation begins from a standing start at the first segment.
pub fn qss_lap_sim(track: &Track, params: &CarParams) -> QssResult {
    let n = track.segments.len();
    let closed = track.is_closed;
    let total_length = track.total_length;

    let s: Vec<f64> = track.segments.iter().map(|seg| seg.s).collect();
    let kappa: Vec<f64> = track
        .segments
        .iter()
        .map(|seg| seg.curvature.abs())
        .collect();

    // Distance from segment `i` to its successor (wrapping for closed tracks).
    let ds_next = |i: usize| -> f64 {
        if i + 1 < n {
            s[i + 1] - s[i]
        } else {
            total_length - s[n - 1]
        }
    };

    // Step 1: cornering-limited speed.
    let mut speeds: Vec<f64> = (0..n).map(|i| cornering_speed(params, kappa[i])).collect();

    // Open tracks start from rest at the first segment.
    if !closed {
        speeds[0] = speeds[0].min(V_FLOOR);
    }

    // Step 2: forward (acceleration) pass.
    let fwd_passes = if closed { 2 } else { 1 };
    for _ in 0..fwd_passes {
        let steps = if closed { n } else { n - 1 };
        for i in 0..steps {
            let j = if i + 1 < n { i + 1 } else { 0 };
            let cand = forward_speed(params, speeds[i], kappa[i], ds_next(i));
            if cand < speeds[j] {
                speeds[j] = cand;
            }
        }
    }

    // Step 3: backward (braking) pass.
    let bwd_passes = if closed { 2 } else { 1 };
    for _ in 0..bwd_passes {
        if closed {
            for i in (0..n).rev() {
                let p = if i == 0 { n - 1 } else { i - 1 };
                let ds = if i == 0 {
                    total_length - s[n - 1]
                } else {
                    s[i] - s[i - 1]
                };
                let cand = backward_speed(params, speeds[i], kappa[i], ds);
                if cand < speeds[p] {
                    speeds[p] = cand;
                }
            }
        } else {
            for i in (1..n).rev() {
                let ds = s[i] - s[i - 1];
                let cand = backward_speed(params, speeds[i], kappa[i], ds);
                if cand < speeds[i - 1] {
                    speeds[i - 1] = cand;
                }
            }
        }
    }

    // Step 4: lap time, per-sector times, and accelerations.
    let (lap_time, sector_times) =
        integrate_lap_and_sectors(&s, &speeds, total_length, closed, DEFAULT_SECTOR_COUNT);

    let lateral_gs: Vec<f64> = (0..n)
        .map(|i| speeds[i] * speeds[i] * kappa[i] / GRAVITY)
        .collect();

    let mut longitudinal_gs = vec![0.0; n];
    for i in 0..n.saturating_sub(1) {
        let ds = s[i + 1] - s[i];
        if ds > 0.0 {
            longitudinal_gs[i] =
                (speeds[i + 1] * speeds[i + 1] - speeds[i] * speeds[i]) / (2.0 * ds) / GRAVITY;
        }
    }

    QssResult {
        speeds,
        lap_time,
        distances: s,
        lateral_gs,
        longitudinal_gs,
        sector_times,
    }
}

// ---------------------------------------------------------------------------
// Tire-aware QSS: Pacejka load-sensitive grip instead of the grip circle.
// ---------------------------------------------------------------------------

/// Total load-sensitive grip from the four tires: `Σ μ_eff(F_z_i)·F_z_i`.
fn tire_available_grip(
    params: &CarParams,
    tire: &PacejkaTire,
    speed: f64,
    longitudinal_accel: f64,
    lateral_accel: f64,
    rsf: f64,
) -> f64 {
    let loads = params.corner_loads(speed, longitudinal_accel, lateral_accel, rsf);
    let mu_blend = 0.5 * (tire.lateral.mu + tire.longitudinal.mu);
    loads
        .iter()
        .map(|&fz| tire.effective_mu(mu_blend, fz) * fz)
        .sum()
}

/// Cornering-limited speed using the tire grip budget, found by bisection.
fn cornering_speed_tire(params: &CarParams, tire: &PacejkaTire, kappa: f64, rsf: f64) -> f64 {
    if kappa < 1e-9 {
        return V_CAP;
    }
    let mut lo = 0.5;
    let mut hi = V_CAP;
    for _ in 0..20 {
        let mid = 0.5 * (lo + hi);
        let lateral_accel = mid * mid * kappa;
        let lat_required = params.mass * lateral_accel;
        let available = tire_available_grip(params, tire, mid, 0.0, lateral_accel, rsf);
        if lat_required <= available {
            lo = mid; // feasible — can go faster
        } else {
            hi = mid;
        }
    }
    lo
}

/// Maximum speed one step ahead, with longitudinal grip from the tire budget.
fn forward_speed_tire(
    params: &CarParams,
    tire: &PacejkaTire,
    v: f64,
    kappa: f64,
    ds: f64,
    rsf: f64,
) -> f64 {
    let lateral_accel = v * v * kappa;
    let available = tire_available_grip(params, tire, v, 0.0, lateral_accel, rsf);
    let f_lat = params.mass * lateral_accel;
    let f_lon_max = (available * available - f_lat * f_lat).max(0.0).sqrt();
    let f_accel = f_lon_max.min(params.max_drive_force)
        - params.drag_force(v)
        - params.rolling_resistance_force();
    let a = f_accel / params.mass;
    let v_next_sq = v * v + 2.0 * a * ds;
    v_next_sq.max(V_FLOOR * V_FLOOR).sqrt()
}

/// Maximum speed one step behind that can still brake to `v`, tire-limited.
fn backward_speed_tire(
    params: &CarParams,
    tire: &PacejkaTire,
    v: f64,
    kappa: f64,
    ds: f64,
    rsf: f64,
) -> f64 {
    let lateral_accel = v * v * kappa;
    let available = tire_available_grip(params, tire, v, 0.0, lateral_accel, rsf);
    let f_lat = params.mass * lateral_accel;
    let f_lon_max = (available * available - f_lat * f_lat).max(0.0).sqrt();
    let a_decel = (f_lon_max.min(params.max_brake_force)
        + params.drag_force(v)
        + params.rolling_resistance_force())
        / params.mass;
    let v_prev_sq = v * v + 2.0 * a_decel * ds;
    v_prev_sq.max(V_FLOOR * V_FLOOR).sqrt()
}

/// Run a QSS lap simulation using the Pacejka tire model with load-sensitive grip.
///
/// This replaces the simple μ·(mg + downforce) grip circle with the actual
/// four-corner load distribution and load-sensitive friction coefficients.
/// The resulting speed profile is conservative (slower) compared to the
/// grip-circle version, but it's a much better warm start for the 7-DOF optimizer.
pub fn qss_lap_sim_tire(
    track: &Track,
    params: &CarParams,
    tire: &PacejkaTire,
    roll_stiffness_front_fraction: f64,
) -> QssResult {
    let n = track.segments.len();
    let closed = track.is_closed;
    let total_length = track.total_length;
    let rsf = roll_stiffness_front_fraction;

    let s: Vec<f64> = track.segments.iter().map(|seg| seg.s).collect();
    let kappa: Vec<f64> = track
        .segments
        .iter()
        .map(|seg| seg.curvature.abs())
        .collect();

    let ds_next = |i: usize| -> f64 {
        if i + 1 < n {
            s[i + 1] - s[i]
        } else {
            total_length - s[n - 1]
        }
    };

    // Step 1: tire-limited cornering speed.
    let mut speeds: Vec<f64> = (0..n)
        .map(|i| cornering_speed_tire(params, tire, kappa[i], rsf))
        .collect();

    if !closed {
        speeds[0] = speeds[0].min(V_FLOOR);
    }

    // Step 2: forward (acceleration) pass.
    let fwd_passes = if closed { 2 } else { 1 };
    for _ in 0..fwd_passes {
        let steps = if closed { n } else { n - 1 };
        for i in 0..steps {
            let j = if i + 1 < n { i + 1 } else { 0 };
            let cand = forward_speed_tire(params, tire, speeds[i], kappa[i], ds_next(i), rsf);
            if cand < speeds[j] {
                speeds[j] = cand;
            }
        }
    }

    // Step 3: backward (braking) pass.
    let bwd_passes = if closed { 2 } else { 1 };
    for _ in 0..bwd_passes {
        if closed {
            for i in (0..n).rev() {
                let p = if i == 0 { n - 1 } else { i - 1 };
                let ds = if i == 0 {
                    total_length - s[n - 1]
                } else {
                    s[i] - s[i - 1]
                };
                let cand = backward_speed_tire(params, tire, speeds[i], kappa[i], ds, rsf);
                if cand < speeds[p] {
                    speeds[p] = cand;
                }
            }
        } else {
            for i in (1..n).rev() {
                let ds = s[i] - s[i - 1];
                let cand = backward_speed_tire(params, tire, speeds[i], kappa[i], ds, rsf);
                if cand < speeds[i - 1] {
                    speeds[i - 1] = cand;
                }
            }
        }
    }

    // Step 4: lap time, per-sector times, and accelerations (unchanged integral).
    let (lap_time, sector_times) =
        integrate_lap_and_sectors(&s, &speeds, total_length, closed, DEFAULT_SECTOR_COUNT);

    let lateral_gs: Vec<f64> = (0..n)
        .map(|i| speeds[i] * speeds[i] * kappa[i] / GRAVITY)
        .collect();

    let mut longitudinal_gs = vec![0.0; n];
    for i in 0..n.saturating_sub(1) {
        let ds = s[i + 1] - s[i];
        if ds > 0.0 {
            longitudinal_gs[i] =
                (speeds[i + 1] * speeds[i + 1] - speeds[i] * speeds[i]) / (2.0 * ds) / GRAVITY;
        }
    }

    QssResult {
        speeds,
        lap_time,
        distances: s,
        lateral_gs,
        longitudinal_gs,
        sector_times,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_track::{build_track, circle_track, oval_track, TrackPoint};

    fn terminal_velocity(params: &CarParams) -> f64 {
        let f_roll = params.rolling_resistance_force();
        let k = 0.5 * params.air_density * params.drag_coeff * params.frontal_area;
        ((params.max_drive_force - f_roll) / k).sqrt()
    }

    #[test]
    fn circle_speed_is_constant() {
        let params = CarParams::default();
        let radius = 100.0;
        let (points, closed) = circle_track(radius, 12.0, 200);
        let track = build_track("circle", &points, closed);
        let result = qss_lap_sim(&track, &params);

        let max = result.speeds.iter().cloned().fold(f64::MIN, f64::max);
        let min = result.speeds.iter().cloned().fold(f64::MAX, f64::min);
        let mean: f64 = result.speeds.iter().sum::<f64>() / result.speeds.len() as f64;

        // within 5% variation around the lap
        assert!(
            (max - min) / mean < 0.05,
            "variation: min {} max {}",
            min,
            max
        );

        // matches the cornering-limit formula for κ = 1/R
        let v_corner = cornering_speed(&params, 1.0 / radius);
        assert!(
            (mean - v_corner).abs() / v_corner < 0.03,
            "mean {} vs corner {}",
            mean,
            v_corner
        );
    }

    #[test]
    fn oval_straights_faster_than_corners() {
        let params = CarParams::default();
        let straight = 1000.0;
        let radius = 100.0;
        let (points, closed) = oval_track(straight, radius, 12.0, 400);
        let track = build_track("oval", &points, closed);
        let result = qss_lap_sim(&track, &params);

        let max = result.speeds.iter().cloned().fold(f64::MIN, f64::max);
        let min = result.speeds.iter().cloned().fold(f64::MAX, f64::min);

        let v_corner = cornering_speed(&params, 1.0 / radius);
        let v_term = terminal_velocity(&params);

        // straights much faster than corners
        assert!(max > min * 1.3, "max {} min {}", max, min);
        // approaches but does not exceed terminal velocity
        assert!(max < v_term, "max {} >= terminal {}", max, v_term);
        assert!(max > 100.0, "max only {}", max);
        // min matches cornering-limited speed for R = 100
        assert!(
            (min - v_corner).abs() / v_corner < 0.05,
            "min {} vs corner {}",
            min,
            v_corner
        );
        // lap time reasonable
        assert!(
            result.lap_time > 20.0 && result.lap_time < 60.0,
            "lap time {}",
            result.lap_time
        );
    }

    #[test]
    fn straight_line_accelerates() {
        let params = CarParams::default();
        let n = 500;
        let length = 1000.0;
        let points: Vec<TrackPoint> = (0..n)
            .map(|i| TrackPoint {
                x: length * (i as f64) / ((n - 1) as f64),
                y: 0.0,
                width_left: 5.0,
                width_right: 5.0,
            })
            .collect();
        let track = build_track("straight", &points, false);
        let result = qss_lap_sim(&track, &params);

        let v_term = terminal_velocity(&params);
        let last = *result.speeds.last().unwrap();
        assert!(last < v_term, "final {} exceeded terminal {}", last, v_term);
        assert!(
            last > 0.85 * v_term,
            "final {} not near terminal {}",
            last,
            v_term
        );

        // monotonically non-decreasing
        for i in 0..result.speeds.len() - 1 {
            assert!(
                result.speeds[i + 1] >= result.speeds[i] - 1e-6,
                "speed dropped at {}: {} -> {}",
                i,
                result.speeds[i],
                result.speeds[i + 1]
            );
        }
    }

    #[test]
    fn sanity_checks() {
        let params = CarParams::default();
        let (points, closed) = oval_track(1000.0, 100.0, 12.0, 400);
        let track = build_track("oval", &points, closed);
        let result = qss_lap_sim(&track, &params);

        let v_corner = cornering_speed(&params, 1.0 / 100.0);
        // straights are faster, so the lap is quicker than driving the whole
        // length at corner speed
        assert!(
            result.lap_time < track.total_length / v_corner,
            "lap {} vs naive {}",
            result.lap_time,
            track.total_length / v_corner
        );

        for &v in &result.speeds {
            assert!(v > 0.0, "non-positive speed {}", v);
            assert!(v < 200.0, "speed too high {}", v);
        }
    }

    #[test]
    fn silverstone_lap_time_reasonable() {
        let params = CarParams::default();
        let (points, closed) = apex_track::silverstone_circuit();
        let track = build_track("Silverstone", &points, closed);
        let result = qss_lap_sim(&track, &params);
        // sanity range (real F1 is ~88 s); this point-mass model with the
        // aggressive default car runs quicker but should land in 60–120 s
        assert!(
            (60.0..=120.0).contains(&result.lap_time),
            "Silverstone lap time {} out of range",
            result.lap_time
        );
    }

    #[test]
    fn sector_times_sum_to_lap_time() {
        let params = CarParams::default();

        // Exercise every producer/topology combination: closed oval, closed
        // synthetic circuit, open straight, and the tire-aware path.
        let (op, oc) = oval_track(1000.0, 100.0, 12.0, 400);
        let oval = build_track("oval", &op, oc);
        let (sp, sc) = apex_track::silverstone_circuit();
        let silverstone = build_track("Silverstone", &sp, sc);
        let straight_pts: Vec<TrackPoint> = (0..300)
            .map(|i| TrackPoint {
                x: 1000.0 * (i as f64) / 299.0,
                y: 0.0,
                width_left: 5.0,
                width_right: 5.0,
            })
            .collect();
        let straight = build_track("straight", &straight_pts, false);
        let tire = PacejkaTire::f1_default();

        let results = [
            qss_lap_sim(&oval, &params),
            qss_lap_sim(&silverstone, &params),
            qss_lap_sim(&straight, &params),
            qss_lap_sim_tire(&oval, &params, &tire, RSF),
        ];

        for r in &results {
            assert_eq!(
                r.sector_times.len(),
                DEFAULT_SECTOR_COUNT,
                "expected {DEFAULT_SECTOR_COUNT} sectors, got {}",
                r.sector_times.len()
            );
            let sum: f64 = r.sector_times.iter().sum();
            assert!(
                (sum - r.lap_time).abs() < 1e-9,
                "sector times {:?} sum to {sum}, lap_time {}",
                r.sector_times,
                r.lap_time
            );
            for (k, &t) in r.sector_times.iter().enumerate() {
                assert!(t > 0.0, "sector {k} time {t} not positive");
            }
        }
    }

    #[test]
    fn monza_faster_than_silverstone() {
        let params = CarParams::default();

        let (sp, sc) = apex_track::silverstone_circuit();
        let silverstone = build_track("Silverstone", &sp, sc);
        let silverstone_lap = qss_lap_sim(&silverstone, &params).lap_time;

        let (mp, mc) = apex_track::monza_circuit();
        let monza = build_track("Monza", &mp, mc);
        let monza_lap = qss_lap_sim(&monza, &params).lap_time;

        // Monza's long straights and fewer corners -> higher average speed
        assert!(
            monza_lap < silverstone_lap,
            "Monza lap {} should be faster than Silverstone {}",
            monza_lap,
            silverstone_lap
        );
    }

    // --- tire-aware QSS tests ---

    use crate::tire::PacejkaTire;

    const RSF: f64 = 0.55;

    #[test]
    fn tire_qss_circle_slightly_slower() {
        let params = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let (points, closed) = circle_track(100.0, 12.0, 200);
        let track = build_track("circle", &points, closed);

        let grip = qss_lap_sim(&track, &params);
        let tire_q = qss_lap_sim_tire(&track, &params, &tire, RSF);

        let grip_mean: f64 = grip.speeds.iter().sum::<f64>() / grip.speeds.len() as f64;
        let tire_mean: f64 = tire_q.speeds.iter().sum::<f64>() / tire_q.speeds.len() as f64;

        // load sensitivity reduces effective grip, so tire-aware is a bit slower
        assert!(
            tire_mean < grip_mean,
            "tire {} should be < grip {}",
            tire_mean,
            grip_mean
        );
        assert!(
            tire_mean > 0.80 * grip_mean,
            "tire {} unreasonably low vs {}",
            tire_mean,
            grip_mean
        );
    }

    #[test]
    fn tire_qss_oval_more_conservative() {
        let params = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let (points, closed) = oval_track(1000.0, 100.0, 12.0, 400);
        let track = build_track("oval", &points, closed);

        let grip = qss_lap_sim(&track, &params);
        let tire_q = qss_lap_sim_tire(&track, &params, &tire, RSF);

        // slower lap, all speeds valid
        assert!(
            tire_q.lap_time > grip.lap_time,
            "tire lap {} should exceed grip {}",
            tire_q.lap_time,
            grip.lap_time
        );
        for &v in &tire_q.speeds {
            assert!(v > 0.0 && v < V_CAP, "speed {} out of range", v);
        }
        // corner (min) speed lower, straight (max) speed similar
        let grip_min = grip.speeds.iter().cloned().fold(f64::MAX, f64::min);
        let tire_min = tire_q.speeds.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            tire_min < grip_min,
            "tire corner {} should be < grip corner {}",
            tire_min,
            grip_min
        );
        let grip_max = grip.speeds.iter().cloned().fold(f64::MIN, f64::max);
        let tire_max = tire_q.speeds.iter().cloned().fold(f64::MIN, f64::max);
        assert!(
            (tire_max - grip_max).abs() / grip_max < 0.05,
            "straight speeds should be similar"
        );
    }

    #[test]
    fn tire_cornering_speed_lower_at_r100() {
        let params = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let kappa = 1.0 / 100.0;
        let grip_v = cornering_speed(&params, kappa);
        let tire_v = cornering_speed_tire(&params, &tire, kappa, RSF);
        assert!(
            tire_v < grip_v,
            "tire {} should be < grip {}",
            tire_v,
            grip_v
        );
        let reduction = (grip_v - tire_v) / grip_v;
        assert!(
            (0.05..=0.20).contains(&reduction),
            "reduction {} should be 5-20%",
            reduction
        );
    }
}
