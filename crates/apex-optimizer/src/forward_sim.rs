//! Phase B of the two-phase 14-DOF pipeline: forward-simulate the full 14-DOF
//! model along an optimized racing line.
//!
//! The reduced collocation optimizer (Phase A) works in the curvilinear 4-state
//! frame and cannot represent suspension travel, body roll/pitch, or the
//! ride-height coupling directly. This module replays the optimized speed and
//! curvature profile through the full 14-DOF dynamics (with the adaptive RK45
//! integrator) to recover that detailed chassis behavior.

use apex_integrator::{rk45_adaptive_step, AdaptiveConfig, OdeSystem};
use apex_physics::{AeroModel, CarParams, FourteenDofModel, PacejkaTire, SuspensionSystem};
use apex_track::{normalize_angle, Track};

use crate::collocation::OptimizationResult;

/// Standard gravity (m/s²) for reporting accelerations in g.
const GRAVITY: f64 = 9.81;

/// Forward-simulate the full 14-DOF model along an optimized trajectory.
///
/// Takes the optimized speed profile, steering, and throttle/brake from the
/// collocation optimizer and replays them through the full 14-DOF dynamics
/// to capture suspension travel, chassis roll/pitch, and ride-height effects
/// that the reduced optimizer cannot represent.
pub struct ForwardSimulator<'a> {
    pub params: &'a CarParams,
    pub tire: &'a PacejkaTire,
    pub suspension: &'a SuspensionSystem,
    pub aero: &'a AeroModel,
    pub track: &'a Track,
}

/// Detailed telemetry from the 14-DOF forward simulation.
#[derive(Debug, Clone)]
pub struct DetailedTelemetry {
    /// Time stamps (s).
    pub time: Vec<f64>,
    /// Arc length stations (m).
    pub s: Vec<f64>,
    /// Speed (m/s).
    pub speed: Vec<f64>,
    /// Lateral offset from centerline (m).
    pub lateral_offset: Vec<f64>,
    /// Roll angle (rad).
    pub roll: Vec<f64>,
    /// Pitch angle (rad).
    pub pitch: Vec<f64>,
    /// Suspension displacement per corner [fl, fr, rl, rr] (m).
    pub suspension_fl: Vec<f64>,
    pub suspension_fr: Vec<f64>,
    pub suspension_rl: Vec<f64>,
    pub suspension_rr: Vec<f64>,
    /// Vertical load per tire (N).
    pub fz_fl: Vec<f64>,
    pub fz_fr: Vec<f64>,
    pub fz_rl: Vec<f64>,
    pub fz_rr: Vec<f64>,
    /// Lateral g.
    pub lateral_g: Vec<f64>,
    /// Longitudinal g.
    pub longitudinal_g: Vec<f64>,
    /// Ride height front/rear (m).
    pub ride_height_front: Vec<f64>,
    pub ride_height_rear: Vec<f64>,
    /// Total lap time.
    pub lap_time: f64,
}

impl DetailedTelemetry {
    fn with_capacity(cap: usize) -> Self {
        DetailedTelemetry {
            time: Vec::with_capacity(cap),
            s: Vec::with_capacity(cap),
            speed: Vec::with_capacity(cap),
            lateral_offset: Vec::with_capacity(cap),
            roll: Vec::with_capacity(cap),
            pitch: Vec::with_capacity(cap),
            suspension_fl: Vec::with_capacity(cap),
            suspension_fr: Vec::with_capacity(cap),
            suspension_rl: Vec::with_capacity(cap),
            suspension_rr: Vec::with_capacity(cap),
            fz_fl: Vec::with_capacity(cap),
            fz_fr: Vec::with_capacity(cap),
            fz_rl: Vec::with_capacity(cap),
            fz_rr: Vec::with_capacity(cap),
            lateral_g: Vec::with_capacity(cap),
            longitudinal_g: Vec::with_capacity(cap),
            ride_height_front: Vec::with_capacity(cap),
            ride_height_rear: Vec::with_capacity(cap),
            lap_time: 0.0,
        }
    }
}

/// Linear interpolation of `ys` (sampled at increasing `xs`) at `x`, clamped.
fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let last = xs.len() - 1;
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[last] {
        return ys[last];
    }
    let mut lo = 0;
    let mut hi = last;
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

/// Arc length of the nearest point on the track centerline to `(x, y)`.
///
/// Projects onto each segment chord (clamped) and returns the interpolated arc
/// length of the closest one. Deriving the control reference from the car's true
/// position — rather than dead-reckoning — keeps the tracker from desyncing.
fn project_s(track: &Track, x: f64, y: f64) -> f64 {
    let segs = &track.segments;
    let n = segs.len();
    let last = if track.is_closed { n } else { n - 1 };

    let mut best_s = 0.0;
    let mut best_d2 = f64::INFINITY;
    for i in 0..last {
        let a = &segs[i];
        let j = (i + 1) % n;
        let b = &segs[j];
        let ex = b.x - a.x;
        let ey = b.y - a.y;
        let len2 = ex * ex + ey * ey;
        let t = if len2 > 1e-12 {
            (((x - a.x) * ex + (y - a.y) * ey) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let px = a.x + t * ex;
        let py = a.y + t * ey;
        let d2 = (x - px) * (x - px) + (y - py) * (y - py);
        if d2 < best_d2 {
            best_d2 = d2;
            let s_b = if j == 0 { track.total_length } else { b.s };
            best_s = a.s + t * (s_b - a.s);
        }
    }
    best_s
}

impl ForwardSimulator<'_> {
    /// Simulate the 14-DOF model along the optimized trajectory.
    ///
    /// Uses the speed profile from the optimization result to generate
    /// throttle/brake commands, and the curvature commands to generate steering.
    /// The full 14-DOF model then runs with the RK45 adaptive integrator.
    pub fn simulate(&self, opt_result: &OptimizationResult) -> DetailedTelemetry {
        let params = self.params;
        let total_length = self.track.total_length;

        // Cap cornering to a lateral acceleration the simple tracker can sustain
        // (a higher-fidelity controller could exploit the full grip, but this
        // keeps the forward sim robustly stable on any track).
        let a_lat_max = 17.5; // m/s² (~1.8 g)

        // Start at a speed the controller can actually hold in the first corner,
        // so it does not begin above its sustainable limit and drift wide.
        let v_cap0 = (a_lat_max / self.track.curvature_at(0.0).abs().max(1e-4)).sqrt();
        let v_start = opt_result
            .speeds
            .first()
            .copied()
            .unwrap_or(30.0)
            .min(v_cap0)
            .max(5.0);
        let model =
            FourteenDofModel::new(params, self.tire, self.suspension, self.aero, v_start);

        // --- initial 14-DOF state (24 elements) ---
        let (x0, y0) = self.track.position_at(0.0);
        let psi0 = self.track.heading_at(0.0);
        let z_eq = model.equilibrium_travel();

        let mut state = [0.0f64; 24];
        state[0] = x0;
        state[1] = y0;
        state[2] = self.aero.design_ride_height + params.cog_height;
        state[5] = psi0;
        state[6] = v_start;
        // If we start in a corner, seed the steady cornering yaw rate (v·κ) so
        // there is no turn-in transient. On a (near-)straight start, leave it at
        // zero — a spurious seam-curvature artifact would otherwise yaw the car.
        let kappa_start = self.track.curvature_at(0.0);
        if kappa_start.abs() > 0.002 {
            state[11] = v_start * kappa_start;
        }
        let wheel_omega = v_start / params.wheel_radius;
        for w in state.iter_mut().skip(12).take(4) {
            *w = wheel_omega;
        }
        state[16..20].copy_from_slice(&z_eq);

        let config = AdaptiveConfig {
            atol: 1e-5,
            rtol: 1e-5,
            dt_min: 1e-7,
            dt_max: 0.005,
            ..AdaptiveConfig::default()
        };

        let safety_time = 300.0;
        let record_every = 10usize;

        let mut t = 0.0;
        // `s_progress` is the unwrapped arc length (monotonic, for lap detection);
        // the control reference `s` comes from projecting the car's actual
        // position onto the centerline so it never desyncs from the car.
        let mut s_progress = 0.0;
        let mut dt = config.dt_max;
        let mut accepted = 0usize;
        // Seed the steering at the steady-state feedforward lock for the first
        // corner so the rate limiter does not have to ramp up from zero.
        let kappa0 = self.track.curvature_at(0.0);
        let mut prev_delta =
            kappa0 * params.wheelbase + 0.0012 * (v_start * v_start * kappa0);

        let mut tele = DetailedTelemetry::with_capacity(1024);

        while s_progress < total_length && t < safety_time {
            // --- locate the car on the track by nearest-point projection ---
            let s = project_s(self.track, state[0], state[1]);

            // advance the unwrapped progress by the (wrapped) change in s
            let cur_mod = s_progress.rem_euclid(total_length);
            let mut ds = s - cur_mod;
            if ds > 0.5 * total_length {
                ds -= total_length;
            } else if ds < -0.5 * total_length {
                ds += total_length;
            }
            s_progress += ds;

            // --- controller ---
            let (tx, ty) = self.track.position_at(s);
            let track_heading = self.track.heading_at(s);
            let kappa_track = self.track.curvature_at(s);
            let curv_cmd = interp(&opt_result.stations, &opt_result.curvature_cmds, s);

            let dx = state[0] - tx;
            let dy = state[1] - ty;
            let n_offset = -dx * track_heading.sin() + dy * track_heading.cos();

            // Track the velocity (course) direction, not the body heading, so the
            // steering does not fight the steady-state sideslip during cornering.
            let vx = state[6];
            let vy = state[7];
            let psi = state[5];
            let v_gx = vx * psi.cos() - vy * psi.sin();
            let v_gy = vx * psi.sin() + vy * psi.cos();
            let course = v_gy.atan2(v_gx);
            let heading_error = normalize_angle(course - track_heading);

            let (wl, wr) = self.track.width_at(s);
            let half_width = 0.5 * (wl + wr);
            let off_track = n_offset.abs() > half_width;
            let (k_lat0, k_head0, speed_scale) =
                if off_track { (0.30, 1.1, 0.85) } else { (0.15, 0.8, 1.0) };
            // Speed-schedule the steering feedback: high gains are needed for
            // low-speed tight corners but cause a growing weave on high-speed
            // straights, so scale them down as speed rises.
            let sf = (32.0 / vx.max(1.0)).clamp(0.45, 1.0);
            let k_lat = k_lat0 * sf * sf;
            let k_head = k_head0 * sf;

            // Feedforward: kinematic steering plus an understeer term proportional
            // to lateral acceleration (the tires need a slip angle to corner, so
            // pure kinematic steering runs the car wide at speed).
            let kappa_eff = 0.5 * (curv_cmd + kappa_track);
            let k_understeer = 0.0012;
            let ff = kappa_eff * params.wheelbase + k_understeer * (vx * vx * kappa_eff);

            // Light yaw-rate damper: oppose deviation from the path's steady-state
            // yaw rate (v·κ) to damp the lateral weave. Small gain so it does not
            // dominate the turn-in transient (where omega_z is still building).
            let yaw_rate_error = state[11] - vx * kappa_eff;
            let k_yaw = 0.12 * sf;

            let delta_target = (ff - k_lat * n_offset - k_head * heading_error
                - k_yaw * yaw_rate_error)
                .clamp(-0.5, 0.5);
            // Rate-limit the steering so turn-in cannot snap the car into oversteer.
            let max_rate = 3.0; // rad/s
            let delta = (delta_target - prev_delta)
                .clamp(-max_rate * dt, max_rate * dt)
                + prev_delta;
            prev_delta = delta;

            // Speed target with look-ahead braking: the achievable speed now is
            // limited by the cornering cap at every point within braking distance
            // ahead (a local backward pass), so the car slows *before* corners
            // instead of arriving over the limit and spinning.
            let a_brake = 18.0; // m/s² assumed braking capability for planning
            let mut target = interp(&opt_result.stations, &opt_result.speeds, s);
            let horizon = (vx * vx / (2.0 * a_brake)).clamp(10.0, 400.0);
            let n_look = 8;
            for j in 0..=n_look {
                let ds = horizon * (j as f64) / (n_look as f64);
                let kappa_ahead = self.track.curvature_at(s + ds).abs().max(1e-4);
                let v_corner = (a_lat_max / kappa_ahead).sqrt();
                let v_allowed = (v_corner * v_corner + 2.0 * a_brake * ds).sqrt();
                target = target.min(v_allowed);
            }
            target *= speed_scale;

            // Proportional throttle/brake: hold speed with a drag-offset torque,
            // a little drive when below target, and firm braking when above it.
            let drag_offset = params.drag_force(vx) * params.wheel_radius;
            let speed_err = target - vx;
            let (torque, brake) = if speed_err >= 0.0 {
                ((drag_offset + 400.0 * speed_err).min(3500.0), 0.0)
            } else {
                (0.0, (-speed_err * 0.10).clamp(0.0, 1.0))
            };
            let control = [delta, torque, brake];

            // --- adaptive step ---
            let step = rk45_adaptive_step(&model, &state, &control, t, dt, &config);
            let at_floor = dt <= config.dt_min * (1.0 + 1e-9);
            if step.accepted || at_floor {
                t += dt;
                state = step.state;
                accepted += 1;

                if !state.iter().all(|v| v.is_finite()) {
                    break;
                }

                if accepted.is_multiple_of(record_every) {
                    self.record(&model, &state, &control, t, s_progress, &mut tele);
                }
            }
            dt = step.dt_next;
        }

        // Always record the final state.
        let final_control = [0.0, 0.0, 0.0];
        if state.iter().all(|v| v.is_finite()) {
            self.record(&model, &state, &final_control, t, s_progress, &mut tele);
        }
        tele.lap_time = t;
        tele
    }

    /// Append one telemetry sample for the current state.
    fn record(
        &self,
        model: &FourteenDofModel,
        state: &[f64; 24],
        control: &[f64; 3],
        t: f64,
        s: f64,
        tele: &mut DetailedTelemetry,
    ) {
        let (tx, ty) = self.track.position_at(s);
        let track_heading = self.track.heading_at(s);
        let dx = state[0] - tx;
        let dy = state[1] - ty;
        let n_offset = -dx * track_heading.sin() + dy * track_heading.cos();

        let speed = (state[6] * state[6] + state[7] * state[7]).sqrt();
        let fz = model.tire_loads(state);
        let (rh_f, rh_r) = model.ride_heights_of(state);

        // Accelerations from the dynamics at this instant.
        let d = model.derivatives(state, control, t);
        let a_lon = d[6];
        let a_lat = d[7] + state[6] * state[11];

        tele.time.push(t);
        tele.s.push(s);
        tele.speed.push(speed);
        tele.lateral_offset.push(n_offset);
        tele.roll.push(state[3]);
        tele.pitch.push(state[4]);
        tele.suspension_fl.push(state[16]);
        tele.suspension_fr.push(state[17]);
        tele.suspension_rl.push(state[18]);
        tele.suspension_rr.push(state[19]);
        tele.fz_fl.push(fz[0]);
        tele.fz_fr.push(fz[1]);
        tele.fz_rl.push(fz[2]);
        tele.fz_rr.push(fz[3]);
        tele.lateral_g.push(a_lat / GRAVITY);
        tele.longitudinal_g.push(a_lon / GRAVITY);
        tele.ride_height_front.push(rh_f);
        tele.ride_height_rear.push(rh_r);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collocation::OptimizationResult;
    use apex_track::{build_track, circle_track};

    /// Build a constant-speed optimization result on the given track.
    fn constant_speed_result(track: &Track, n: usize, speed: f64) -> OptimizationResult {
        let stations: Vec<f64> = (0..n)
            .map(|k| track.total_length * (k as f64) / ((n - 1) as f64))
            .collect();
        let curvature_cmds: Vec<f64> =
            stations.iter().map(|&s| track.curvature_at(s)).collect();
        let dt: Vec<f64> = (0..n - 1)
            .map(|k| (stations[k + 1] - stations[k]) / speed)
            .collect();
        let lap_time = dt.iter().sum();
        OptimizationResult {
            speeds: vec![speed; n],
            offsets: vec![0.0; n],
            headings: vec![0.0; n],
            stations,
            drive_forces: vec![0.0; n],
            curvature_cmds,
            time_steps: dt,
            lap_time,
            eq_violation: 0.0,
            converged: true,
        }
    }

    #[test]
    fn forward_sim_produces_valid_telemetry() {
        let (pts, closed) = circle_track(100.0, 12.0, 200);
        let track = build_track("Circle", &pts, closed);
        let params = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let suspension = SuspensionSystem::f1_default();
        let aero = AeroModel::f1_default();

        let opt = constant_speed_result(&track, 60, 40.0);
        let sim = ForwardSimulator {
            params: &params,
            tire: &tire,
            suspension: &suspension,
            aero: &aero,
            track: &track,
        };
        let tele = sim.simulate(&opt);

        // all arrays the same non-zero length
        let len = tele.time.len();
        assert!(len > 5, "expected several samples, got {}", len);
        for arr in [
            &tele.s,
            &tele.speed,
            &tele.lateral_offset,
            &tele.roll,
            &tele.pitch,
            &tele.suspension_fl,
            &tele.suspension_fr,
            &tele.suspension_rl,
            &tele.suspension_rr,
            &tele.fz_fl,
            &tele.fz_fr,
            &tele.fz_rl,
            &tele.fz_rr,
            &tele.lateral_g,
            &tele.longitudinal_g,
            &tele.ride_height_front,
            &tele.ride_height_rear,
        ] {
            assert_eq!(arr.len(), len, "array length mismatch");
            assert!(arr.iter().all(|v| v.is_finite()), "non-finite telemetry");
        }

        // Steady cornering on a circle: roll is small and bounded (a few degrees,
        // not literally zero — the suspension does roll under lateral load).
        let max_roll = tele.roll.iter().cloned().fold(0.0_f64, |m, r| m.max(r.abs()));
        assert!(
            max_roll < 0.06,
            "roll {} rad ({} deg) too large for a gentle circle",
            max_roll,
            max_roll.to_degrees()
        );

        // Suspension displacements in a reasonable range (0–40 mm).
        for z in tele
            .suspension_fl
            .iter()
            .chain(&tele.suspension_fr)
            .chain(&tele.suspension_rl)
            .chain(&tele.suspension_rr)
        {
            assert!(z.abs() < 0.040, "suspension travel {} m out of range", z);
        }
    }
}
