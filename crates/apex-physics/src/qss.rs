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
// 3D point-mass QSS: grade force, vertical-curvature load, banking.
// See docs/math/track3d.md §5. Flat tracks delegate to `qss_lap_sim` bitwise.
// ---------------------------------------------------------------------------

use apex_track::Ribbon3d;

/// 3D normal (surface-perpendicular) load `N` at a segment (docs §5.1):
/// `N = m(g·cosθ·cosφ + v²·κ·sinφ + v²·κ_v) + F_df(v)`.
fn normal_load_3d(
    params: &CarParams,
    v: f64,
    kappa: f64,
    theta: f64,
    phi: f64,
    kappa_v: f64,
) -> f64 {
    params.mass * (GRAVITY * theta.cos() * phi.cos() + v * v * kappa * phi.sin() + v * v * kappa_v)
        + params.downforce(v)
}

/// 3D in-surface lateral force demand (docs §5.2):
/// `F_lat = m(v²·κ·cosφ − g·sinφ)`.
fn lateral_demand_3d(params: &CarParams, v: f64, kappa: f64, phi: f64) -> f64 {
    params.mass * (v * v * kappa * phi.cos() - GRAVITY * phi.sin())
}

/// Cornering-limited speed with 3D load: largest `v` with `μN ≥ |F_lat|`
/// (bisection; both sides scale with `v²`). Reduces to the flat closed form.
fn cornering_speed_3d(params: &CarParams, kappa: f64, theta: f64, phi: f64, kappa_v: f64) -> f64 {
    if kappa < 1e-9 && phi.abs() < 1e-12 {
        return V_CAP;
    }
    let feasible = |v: f64| -> bool {
        let n = normal_load_3d(params, v, kappa, theta, phi, kappa_v);
        params.tire_mu * n >= lateral_demand_3d(params, v, kappa, phi).abs()
    };
    if feasible(V_CAP) {
        return V_CAP;
    }
    let (mut lo, mut hi) = (0.0, V_CAP);
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        if feasible(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo.max(V_FLOOR)
}

/// Longitudinal grip headroom `F_x,max = sqrt((μN)² − F_lat²)` at `(v, κ, …)`.
fn f_lon_max_3d(params: &CarParams, v: f64, kappa: f64, theta: f64, phi: f64, kappa_v: f64) -> f64 {
    let grip = params.tire_mu * normal_load_3d(params, v, kappa, theta, phi, kappa_v);
    let f_lat = lateral_demand_3d(params, v, kappa, phi).abs();
    if f_lat >= grip {
        0.0
    } else {
        (grip * grip - f_lat * f_lat).sqrt()
    }
}

/// Forward (accel) speed one step ahead, with the grade force `−m·g·sinθ`.
fn forward_speed_3d(
    params: &CarParams,
    v: f64,
    kappa: f64,
    theta: f64,
    phi: f64,
    kappa_v: f64,
    ds: f64,
) -> f64 {
    let f_lon = f_lon_max_3d(params, v, kappa, theta, phi, kappa_v);
    let f_accel = f_lon.min(params.max_drive_force)
        - params.drag_force(v)
        - params.rolling_resistance_force()
        - params.mass * GRAVITY * theta.sin();
    let a = f_accel / params.mass;
    (v * v + 2.0 * a * ds).max(V_FLOOR * V_FLOOR).sqrt()
}

/// Backward (braking) speed one step behind; climbing (`θ>0`) aids the decel.
fn backward_speed_3d(
    params: &CarParams,
    v: f64,
    kappa: f64,
    theta: f64,
    phi: f64,
    kappa_v: f64,
    ds: f64,
) -> f64 {
    let f_lon = f_lon_max_3d(params, v, kappa, theta, phi, kappa_v);
    let a_decel = (f_lon.min(params.max_brake_force)
        + params.drag_force(v)
        + params.rolling_resistance_force()
        + params.mass * GRAVITY * theta.sin())
        / params.mass;
    (v * v + 2.0 * a_decel * ds).max(V_FLOOR * V_FLOOR).sqrt()
}

/// Vertical (pitch) curvature `κ_v = dθ/ds` by central differences (periodic on
/// closed tracks). This is the elevation term, distinct from the raw Darboux
/// `Ω_y` (docs §5).
fn vertical_curvature(s: &[f64], theta: &[f64], closed: bool, total_length: f64) -> Vec<f64> {
    let n = s.len();
    let mut kv = vec![0.0; n];
    for (i, kvi) in kv.iter_mut().enumerate() {
        let (im, ip, ds) = if closed {
            let im = (i + n - 1) % n;
            let ip = (i + 1) % n;
            let mut ds = s[ip] - s[im];
            if ds <= 0.0 {
                ds += total_length;
            }
            (im, ip, ds)
        } else if i == 0 {
            (0, 1, s[1] - s[0])
        } else if i == n - 1 {
            (n - 2, n - 1, s[n - 1] - s[n - 2])
        } else {
            (i - 1, i + 1, s[i + 1] - s[i - 1])
        };
        if ds > 0.0 {
            *kvi = (theta[ip] - theta[im]) / ds;
        }
    }
    kv
}

/// Run the QSS lap simulation on a **3D ribbon** with grade, vertical-curvature
/// load, and banking (docs/math/track3d.md §5). A geometrically flat ribbon
/// short-circuits to the untouched 2D [`qss_lap_sim`] on its projection, so flat
/// tracks are **bitwise-identical** and cost nothing extra.
pub fn qss_lap_sim_3d(ribbon: &Ribbon3d, params: &CarParams) -> QssResult {
    if ribbon.is_flat() {
        return qss_lap_sim(&ribbon.to_track_2d(), params);
    }
    let n = ribbon.stations.len();
    let closed = ribbon.is_closed;
    let total_length = ribbon.total_length;

    let s: Vec<f64> = ribbon.stations.iter().map(|st| st.s).collect();
    let kappa: Vec<f64> = ribbon.stations.iter().map(|st| st.omega_z.abs()).collect();
    let theta: Vec<f64> = ribbon.stations.iter().map(|st| st.grade).collect();
    let phi: Vec<f64> = ribbon.stations.iter().map(|st| st.bank).collect();
    let kappa_v = vertical_curvature(&s, &theta, closed, total_length);

    let ds_next = |i: usize| -> f64 {
        if i + 1 < n {
            s[i + 1] - s[i]
        } else {
            total_length - s[n - 1]
        }
    };

    // Step 1: cornering-limited speed.
    let mut speeds: Vec<f64> = (0..n)
        .map(|i| cornering_speed_3d(params, kappa[i], theta[i], phi[i], kappa_v[i]))
        .collect();
    if !closed {
        speeds[0] = speeds[0].min(V_FLOOR);
    }

    // Step 2: forward pass.
    let fwd_passes = if closed { 2 } else { 1 };
    for _ in 0..fwd_passes {
        let steps = if closed { n } else { n - 1 };
        for i in 0..steps {
            let j = if i + 1 < n { i + 1 } else { 0 };
            let cand = forward_speed_3d(
                params,
                speeds[i],
                kappa[i],
                theta[i],
                phi[i],
                kappa_v[i],
                ds_next(i),
            );
            if cand < speeds[j] {
                speeds[j] = cand;
            }
        }
    }

    // Step 3: backward pass.
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
                let cand = backward_speed_3d(
                    params, speeds[i], kappa[i], theta[i], phi[i], kappa_v[i], ds,
                );
                if cand < speeds[p] {
                    speeds[p] = cand;
                }
            }
        } else {
            for i in (1..n).rev() {
                let ds = s[i] - s[i - 1];
                let cand = backward_speed_3d(
                    params, speeds[i], kappa[i], theta[i], phi[i], kappa_v[i], ds,
                );
                if cand < speeds[i - 1] {
                    speeds[i - 1] = cand;
                }
            }
        }
    }

    // Step 4: lap time, sectors, accelerations (kinematic, unchanged formulas).
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

    // --- 3D QSS: flat byte-invariance + synthetic validation (Phase 1.3) ---

    use apex_track::Ribbon3d;
    use std::f64::consts::PI;

    /// The 3D QSS on a FLAT ribbon must be bitwise-identical to the 2D QSS.
    fn assert_qss_bitwise(track: &apex_track::Track, params: &CarParams) {
        let ribbon = track.to_ribbon3d();
        assert!(ribbon.is_flat());
        let a = qss_lap_sim(track, params);
        let b = qss_lap_sim_3d(&ribbon, params);
        assert_eq!(a.lap_time.to_bits(), b.lap_time.to_bits(), "lap_time");
        assert_eq!(a.speeds.len(), b.speeds.len());
        for i in 0..a.speeds.len() {
            assert_eq!(a.speeds[i].to_bits(), b.speeds[i].to_bits(), "speed[{i}]");
        }
        for i in 0..a.sector_times.len() {
            assert_eq!(
                a.sector_times[i].to_bits(),
                b.sector_times[i].to_bits(),
                "sector[{i}]"
            );
        }
    }

    #[test]
    fn flat_ribbon_qss_bitwise_matches_track() {
        let params = CarParams::f1_2024_calibrated();
        let (op, oc) = oval_track(1000.0, 100.0, 12.0, 400);
        assert_qss_bitwise(&build_track("oval", &op, oc), &params);
        let (cp, cc) = circle_track(100.0, 12.0, 200);
        assert_qss_bitwise(&build_track("circle", &cp, cc), &params);
        let (sp, sc) = apex_track::silverstone_circuit();
        assert_qss_bitwise(&build_track("Silverstone", &sp, sc), &params);
    }

    #[test]
    fn banked_ring_cornering_matches_analytic() {
        // No downforce ⇒ the classic banked-turn closed form applies exactly.
        let params = CarParams {
            lift_coeff: 0.0,
            ..CarParams::default()
        };
        let (r, phi) = (120.0, 0.18_f64);
        let mu = params.tire_mu;
        let kappa = 1.0 / r;
        let v = cornering_speed_3d(&params, kappa, 0.0, phi, 0.0);
        let v2 = GRAVITY * r * (phi.sin() + mu * phi.cos()) / (phi.cos() - mu * phi.sin());
        let v_analytic = v2.sqrt();
        assert!(
            (v - v_analytic).abs() < 1e-4,
            "banked v {v} vs analytic {v_analytic}"
        );
        // Banking raises the limit above the unbanked corner.
        let v_flat = cornering_speed_3d(&params, kappa, 0.0, 0.0, 0.0);
        assert!(v > v_flat, "banked {v} should exceed flat {v_flat}");
    }

    #[test]
    fn vertical_curvature_load_matches_analytic() {
        let params = CarParams::default();
        let v = 65.0;
        let kappa_v = 0.0025; // a dip (compression)
        let n_dip = normal_load_3d(&params, v, 0.0, 0.0, 0.0, kappa_v);
        let n_flat = normal_load_3d(&params, v, 0.0, 0.0, 0.0, 0.0);
        // ΔN = m·v²·κ_v exactly.
        assert!(((n_dip - n_flat) - params.mass * v * v * kappa_v).abs() < 1e-9);
        // Flat load is m·g + downforce.
        assert!((n_flat - (params.mass * GRAVITY + params.downforce(v))).abs() < 1e-9);
        // A crest (κ_v<0) unloads.
        let n_crest = normal_load_3d(&params, v, 0.0, 0.0, 0.0, -kappa_v);
        assert!(n_crest < n_flat);
    }

    /// A straight climbing at constant grade θ.
    fn grade_straight(theta: f64, n: usize, dx: f64) -> Ribbon3d {
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let x = i as f64 * dx;
                [x, 0.0, x * theta.tan()]
            })
            .collect();
        let w = vec![5.0; n];
        let bank = vec![0.0; n];
        Ribbon3d::from_centerline_3d("grade", &pts, &bank, &w, &w, false)
    }

    #[test]
    fn constant_grade_shifts_terminal_speed() {
        let params = CarParams::default();
        let k = 0.5 * params.air_density * params.drag_coeff * params.frontal_area;
        let terminal = |theta: f64| {
            ((params.max_drive_force
                - params.rolling_resistance_force()
                - params.mass * GRAVITY * theta.sin())
                / k)
                .sqrt()
        };
        // Long climb / descent so the QSS reaches terminal.
        let climb = qss_lap_sim_3d(&grade_straight(0.05, 8000, 1.0), &params);
        let descent = qss_lap_sim_3d(&grade_straight(-0.05, 8000, 1.0), &params);
        let vt_climb = terminal(0.05);
        let vt_desc = terminal(-0.05);
        let last = |r: &QssResult| *r.speeds.last().unwrap();
        assert!(
            (last(&climb) - vt_climb).abs() / vt_climb < 0.02,
            "climb terminal {} vs {}",
            last(&climb),
            vt_climb
        );
        assert!(
            (last(&descent) - vt_desc).abs() / vt_desc < 0.02,
            "descent terminal {} vs {}",
            last(&descent),
            vt_desc
        );
        // Descending is faster than climbing (gravity assists).
        assert!(last(&descent) > last(&climb));
    }

    /// A constant-grade straight of length `(n1-1)*dx` running into a tight
    /// hairpin (radius `r`, ~170° turn) held at zero grade through the turn
    /// itself. Used by `braking_pass_grade_matches_analytic_onset` below: the
    /// straight is long/flat enough that grade is the only thing that differs
    /// between the three cases exercised there.
    fn grade_straight_into_hairpin(theta: f64, n1: usize, dx: f64, r: f64, n2: usize) -> Ribbon3d {
        let mut pts: Vec<[f64; 3]> = (0..n1)
            .map(|i| {
                let x = i as f64 * dx;
                [x, 0.0, x * theta.tan()]
            })
            .collect();
        let x_end = (n1 - 1) as f64 * dx;
        let z_end = x_end * theta.tan();
        let turn = 170f64.to_radians();
        for j in 1..=n2 {
            let a = turn * j as f64 / n2 as f64;
            pts.push([x_end + r * a.sin(), r - r * a.cos(), z_end]);
        }
        let n = pts.len();
        let w = vec![5.0; n];
        let bank = vec![0.0; n];
        Ribbon3d::from_centerline_3d("grade_hairpin", &pts, &bank, &w, &w, false)
    }

    /// Direct test that the braking (backward) pass includes the grade force
    /// `+m·g·sinθ` in its longitudinal budget, not just the forward pass.
    ///
    /// Setup: a constant-grade straight (drive- and grip-limited, zero drag/
    /// downforce so both the forward acceleration `a_accel` and the braking
    /// deceleration `a_decel` are speed-independent constants) running into a
    /// tight, effectively-flat hairpin. With a single forward pass building
    /// `v_fwd(s)² = 2·a_accel·s` from the start and a single backward pass
    /// building `v_bwd(s)² = v_apex² + 2·a_decel·(L−s)` from the apex, the
    /// braking-onset point is exactly where the two curves cross:
    ///   `d_onset = a_accel·L/(a_accel+a_decel) − v_apex²/(2·(a_accel+a_decel))`
    /// with `a_accel = μg·cosθ − C_rr·g − g·sinθ` and
    /// `a_decel = μg·cosθ + C_rr·g + g·sinθ` (independently re-derived here,
    /// not by calling the solver's internal helpers). If the braking pass
    /// dropped the grade term, `a_decel` would be identical (`μg·cosθ +
    /// C_rr·g`) across climb/flat/descent and `d_onset` would not separate by
    /// grade at all — this test would fail on the ordering assertion alone.
    #[test]
    fn braking_pass_grade_matches_analytic_onset() {
        let mut params = CarParams::f1_2024_calibrated();
        // Zero drag/downforce so a_accel/a_decel are speed-independent, and
        // drive/brake limits are set high enough that both passes are
        // grip-limited (not power/brake-limited) — required for the closed
        // form below.
        params.drag_coeff = 0.0;
        params.lift_coeff = 0.0;
        params.max_drive_force = 1.0e7;
        params.max_brake_force = 1.0e7;

        let g = GRAVITY;
        let mu = params.tire_mu;
        let croll = params.rolling_resistance;
        let a_accel = |theta: f64| mu * g * theta.cos() - croll * g - g * theta.sin();
        let a_decel = |theta: f64| mu * g * theta.cos() + croll * g + g * theta.sin();

        let (n1, dx, r, n2) = (500usize, 1.0, 5.0, 80usize);
        let l = (n1 - 1) as f64 * dx;
        let v_apex = (mu * g * r).sqrt(); // flat, unbanked circular-corner formula

        let analytic_onset = |theta: f64| -> f64 {
            let (aa, ad) = (a_accel(theta), a_decel(theta));
            aa * l / (aa + ad) - v_apex * v_apex / (2.0 * (aa + ad))
        };

        // Locate the simulated onset: the arg-max of the speed trace on the
        // straight portion (indices < n1) is exactly where forward-limited
        // (still rising) hands off to backward-limited (falling toward the
        // apex).
        let sim_onset = |theta: f64| -> f64 {
            let ribbon = grade_straight_into_hairpin(theta, n1, dx, r, n2);
            let res = qss_lap_sim_3d(&ribbon, &params);
            let (mut peak_i, mut peak_v) = (0usize, 0.0f64);
            for i in 0..n1 {
                if res.speeds[i] > peak_v {
                    peak_v = res.speeds[i];
                    peak_i = i;
                }
            }
            l - res.distances[peak_i]
        };

        let thetas = [-0.05, 0.0, 0.05]; // descent, flat, climb
        let onsets: Vec<f64> = thetas.iter().map(|&t| sim_onset(t)).collect();
        let analytic: Vec<f64> = thetas.iter().map(|&t| analytic_onset(t)).collect();

        for i in 0..3 {
            let rel_err = (onsets[i] - analytic[i]).abs() / analytic[i];
            assert!(
                rel_err < 0.03,
                "theta {}: sim onset {} vs analytic {} (rel err {})",
                thetas[i],
                onsets[i],
                analytic[i],
                rel_err
            );
        }

        // The physical claim under test: downhill braking distance > flat > uphill.
        assert!(
            onsets[0] > onsets[1] && onsets[1] > onsets[2],
            "expected descent {} > flat {} > climb {}",
            onsets[0],
            onsets[1],
            onsets[2]
        );
    }

    #[test]
    fn gravity_work_closes_on_closed_lap() {
        // Tilted closed ring: net elevation change is zero, so gravity does zero
        // net work per lap — Σ m·g·Δz telescopes to machine precision.
        let (n, r, amp) = (720usize, 200.0, 15.0);
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let u = 2.0 * PI * i as f64 / n as f64;
                [r * u.cos(), r * u.sin(), amp * u.sin()]
            })
            .collect();
        let w = vec![6.0; n];
        let bank = vec![0.0; n];
        let ribbon = Ribbon3d::from_centerline_3d("ring", &pts, &bank, &w, &w, true);
        let params = CarParams::default();
        let mut work = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            work += params.mass * GRAVITY * (ribbon.stations[j].z - ribbon.stations[i].z);
        }
        assert!(work.abs() < 1e-6, "net gravity work {work} should be ~0");

        // The 3D QSS produces a valid periodic profile on the elevation-varying ring.
        let res = qss_lap_sim_3d(&ribbon, &params);
        assert!(res.speeds.iter().all(|&v| v > 0.0 && v < V_CAP));
        assert!(res.lap_time > 0.0);
    }
}
