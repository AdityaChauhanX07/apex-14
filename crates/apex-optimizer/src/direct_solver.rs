//! Sequential defect correction — a structure-exploiting collocation solver.
//!
//! Generic NLP solvers (augmented Lagrangian, Gauss-Newton) stall on the
//! trapezoidal dynamics defects because damped merit steps cannot drive every
//! defect to zero simultaneously. Real collocation solvers instead correct the
//! defects directly. This module implements that: a nonlinear block
//! Gauss-Seidel sweep on the dynamics defects (Phase A) followed by a
//! consistency update of controls and timing (Phase B), iterated to feasibility.
//!
//! It is specialized to the point-mass layout `[s|n|v|alpha|f_drive|curv|dt]`
//! and consumes the [`Track`] and [`CarParams`] directly for the correction
//! steps, using the [`NlpEvaluator`] only to measure constraint violations.
//!
//! Characteristics on the point-mass problem: this method is stable and
//! lap-accurate (it anchors the speed profile to the grip-limited optimum) and
//! reliably beats the first-order augmented-Lagrangian solver on feasibility.
//! It does **not** beat the Gauss-Newton solver here — GN solves the true NLP
//! and reaches both a lower defect and a lap time marginally closer to QSS. The
//! residual defect of this heuristic is dominated by the grip-limited braking
//! "kink" at corner entry, which cannot be removed without deviating from the
//! optimal speed profile. It is included as a transparent, structure-exploiting
//! baseline; Gauss-Newton remains the recommended solver.

use apex_physics::car_params::GRAVITY;
use apex_physics::CarParams;
use apex_track::{normalize_angle, Track};

use crate::collocation::point_mass_derivatives;
use crate::nlp::{NlpEvaluator, NlpProblem};

/// Dimensional description of a collocation transcription.
#[derive(Debug, Clone, Copy)]
pub struct CollocationStructure {
    /// Number of collocation nodes.
    pub n_nodes: usize,
    /// Number of state variables per node.
    pub n_states: usize,
    /// Number of control variables per node.
    pub n_controls: usize,
}

/// Configuration for the sequential defect-correction solver.
#[derive(Debug, Clone)]
pub struct DirectSolverConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on the maximum absolute dynamics defect.
    pub constraint_tol: f64,
    /// Defect-correction damping (0.3–0.8 is typical).
    pub damping: f64,
    /// Print progress every N iterations (0 = silent).
    pub print_interval: usize,
}

impl Default for DirectSolverConfig {
    fn default() -> Self {
        DirectSolverConfig {
            max_iterations: 200,
            constraint_tol: 1e-4,
            damping: 0.6,
            print_interval: 0,
        }
    }
}

impl apex_math::ContentHash for DirectSolverConfig {
    /// Encode the result-determining fields. `print_interval` is EXCLUDED
    /// (cosmetic, bound to `_`). The destructure forces any new field to be
    /// handled here before it compiles.
    fn hash_into(&self, w: &mut apex_math::HashWriter) {
        let DirectSolverConfig {
            max_iterations,
            constraint_tol,
            damping,
            print_interval: _, // cosmetic; excluded from content identity
        } = self;
        w.usize(*max_iterations);
        w.f64(*constraint_tol);
        w.f64(*damping);
    }
}

/// Result of the direct solve.
#[derive(Debug, Clone)]
pub struct DirectSolverResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Final objective value (total lap time).
    pub objective: f64,
    /// Maximum equality (dynamics-defect) violation.
    pub eq_violation: f64,
    /// Maximum inequality violation (positive = violated).
    pub ineq_violation: f64,
    /// Iterations performed.
    pub iterations: usize,
    /// Whether the solver converged (max defect below tolerance).
    pub converged: bool,
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &c| m.max(c.abs()))
}

fn max_pos(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &c| m.max(c)).max(0.0)
}

/// Cornering-speed limit at a node from the lateral grip budget.
fn cornering_vmax(car: &CarParams, kappa_abs: f64) -> f64 {
    const V_CAP: f64 = 200.0;
    if kappa_abs < 1e-9 {
        return V_CAP;
    }
    let denom = car.mass * kappa_abs
        - car.tire_mu * 0.5 * car.air_density * car.lift_coeff * car.frontal_area;
    if denom <= 0.0 {
        return V_CAP;
    }
    (car.tire_mu * car.mass * GRAVITY / denom).sqrt().min(V_CAP)
}

/// Solve a collocation NLP by sequential defect correction.
pub fn solve_direct(
    problem: &NlpProblem,
    evaluator: &impl NlpEvaluator,
    x0: &[f64],
    config: &DirectSolverConfig,
    structure: CollocationStructure,
    track: &Track,
    car: &CarParams,
) -> DirectSolverResult {
    let n = structure.n_nodes;
    let ns = structure.n_states;
    let nc = structure.n_controls;

    // Block offsets in the decision vector.
    let s_off = 0;
    let n_off = n;
    let v_off = 2 * n;
    let a_off = 3 * n;
    let fd_off = ns * n; // f_drive (control 0)
    let cv_off = (ns + 1) * n; // curvature_cmd (control 1)
    let dt_off = (ns + nc) * n;
    // Bounds carry v_min / dt limits.
    let v_min = problem.lower_bounds[v_off];
    let dt_min = problem.lower_bounds[dt_off];
    let dt_max = problem.upper_bounds[dt_off];

    let ctx = SweepCtx {
        track,
        car,
        damping: config.damping,
        s_off,
        n_off,
        v_off,
        a_off,
        fd_off,
        cv_off,
        dt_off,
        v_min,
    };

    let mut x = x0.to_vec();
    for ((xi, &lb), &ub) in x
        .iter_mut()
        .zip(problem.lower_bounds.iter())
        .zip(problem.upper_bounds.iter())
    {
        *xi = xi.max(lb).min(ub);
    }

    let mut iterations = 0;
    let mut converged = false;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        let c_eq = evaluator.equality_constraints(&x);
        let eq_viol = max_abs(&c_eq);

        if config.print_interval > 0 && iter % config.print_interval == 0 {
            let obj: f64 = x[dt_off..].iter().sum();
            println!(
                "direct {:4} | obj {:.6} | eq_viol {:.3e}",
                iter, obj, eq_viol
            );
        }

        if eq_viol < config.constraint_tol {
            converged = true;
            break;
        }

        // --- Phase A: defect correction (forward then backward sweep) ---
        for k in 0..n - 1 {
            correct_interval(&mut x, k, k + 1, &ctx);
        }
        for k in (0..n - 1).rev() {
            correct_interval(&mut x, k, k, &ctx);
        }

        // --- Phase B: optimize speed and timing to the grip limit ---
        // Curvature command follows the track (stay near the centerline), and
        // n is clamped within the track boundaries.
        for k in 0..n {
            x[cv_off + k] = track.curvature_at(x[s_off + k]);
            let (wl, wr) = track.width_at(x[s_off + k]);
            x[n_off + k] = x[n_off + k].clamp(-wr, wl);
        }

        // Grip-limited speed profile: cornering limit, then forward acceleration
        // and backward braking passes (QSS). This is the point-mass speed
        // optimum on the fixed centerline and is the stable fixed point.
        grip_limited_speed(&mut x, &ctx, n, track.is_closed);

        // Smooth the speed at sharp transitions (corner entry/exit) to reduce
        // the trapezoidal longitudinal defect, which scales with the curvature
        // of v. Laplacian smoothing preserves linear acceleration ramps and
        // only relaxes kinks; speeds stay capped at the cornering limit.
        for _ in 0..5 {
            let v_prev: Vec<f64> = x[v_off..v_off + n].to_vec();
            for k in 1..n - 1 {
                let smoothed = 0.5 * v_prev[k] + 0.25 * (v_prev[k - 1] + v_prev[k + 1]);
                let cap = cornering_vmax(car, x[cv_off + k].abs());
                x[v_off + k] = smoothed.min(cap).max(v_min);
            }
        }

        // Consistent timing from the speed profile.
        for k in 0..n - 1 {
            let ds = x[s_off + k + 1] - x[s_off + k];
            let v_avg = 0.5 * (x[v_off + k] + x[v_off + k + 1]);
            if v_avg > v_min {
                x[dt_off + k] = (ds / v_avg).clamp(dt_min, dt_max);
            }
        }

        // Drive force from the time-domain node acceleration (central
        // difference of speed, one-sided at the ends), then
        // F = m*a + drag + rolling, clamped to the available grip. This smooth
        // acceleration estimate keeps the trapezoidal longitudinal defect
        // second-order small without the oscillation of an exact recurrence.
        let roll = car.rolling_resistance_force();
        for k in 0..n {
            let v = x[v_off + k];
            let a_k = if k == 0 {
                (x[v_off + 1] - x[v_off]) / x[dt_off]
            } else if k == n - 1 {
                (x[v_off + n - 1] - x[v_off + n - 2]) / x[dt_off + n - 2]
            } else {
                (x[v_off + k + 1] - x[v_off + k - 1]) / (x[dt_off + k - 1] + x[dt_off + k])
            };
            let fd = car.mass * a_k + car.drag_force(v) + roll;
            let f_grip = car.max_grip_force(v);
            let f_lat = car.mass * v * v * x[cv_off + k];
            let lon_avail = (f_grip * f_grip - f_lat * f_lat).max(0.0).sqrt();
            let hi = lon_avail.min(car.max_drive_force);
            let lo = -lon_avail.min(car.max_brake_force);
            x[fd_off + k] = fd.clamp(lo, hi);
        }

        // --- Periodicity for closed tracks ---
        if track.is_closed {
            x[s_off] = 0.0;
            x[s_off + n - 1] = track.total_length;

            let n_avg = 0.5 * (x[n_off] + x[n_off + n - 1]);
            x[n_off] = n_avg;
            x[n_off + n - 1] = n_avg;

            let v_avg = 0.5 * (x[v_off] + x[v_off + n - 1]);
            x[v_off] = v_avg;
            x[v_off + n - 1] = v_avg;

            let a0 = x[a_off];
            let an = x[a_off + n - 1];
            let a_avg = normalize_angle(a0 + 0.5 * normalize_angle(an - a0));
            x[a_off] = a_avg;
            x[a_off + n - 1] = a_avg;
        }
    }

    let c_eq = evaluator.equality_constraints(&x);
    let c_ineq = evaluator.inequality_constraints(&x);
    let eq_viol = max_abs(&c_eq);
    converged = converged || eq_viol < config.constraint_tol;

    DirectSolverResult {
        objective: x[dt_off..].iter().sum(),
        eq_violation: eq_viol,
        ineq_violation: max_pos(&c_ineq),
        iterations,
        converged,
        x,
    }
}

/// Constant context for the defect-correction sweeps: block offsets, bounds,
/// and references to the track and car.
struct SweepCtx<'a> {
    track: &'a Track,
    car: &'a CarParams,
    damping: f64,
    s_off: usize,
    n_off: usize,
    v_off: usize,
    a_off: usize,
    fd_off: usize,
    cv_off: usize,
    dt_off: usize,
    v_min: f64,
}

/// Set the speed profile to the grip-limited optimum: cornering limit, then
/// forward acceleration and backward braking passes (a QSS sweep). This is the
/// point-mass speed optimum on the fixed centerline and the stable fixed point
/// the iteration converges to.
fn grip_limited_speed(x: &mut [f64], c: &SweepCtx, n: usize, closed: bool) {
    let car = c.car;
    let roll = car.rolling_resistance_force();
    let v_min = c.v_min;

    let mut v: Vec<f64> = (0..n)
        .map(|k| cornering_vmax(car, x[c.cv_off + k].abs()).max(v_min))
        .collect();

    let passes = if closed { 2 } else { 1 };

    // forward acceleration passes
    for _ in 0..passes {
        if closed {
            let m = v[0].min(v[n - 1]);
            v[0] = m;
        }
        for k in 0..n - 1 {
            let vk = v[k];
            let f_grip = car.max_grip_force(vk);
            let f_lat = car.mass * vk * vk * x[c.cv_off + k].abs();
            let f_lon = (f_grip * f_grip - f_lat * f_lat)
                .max(0.0)
                .sqrt()
                .min(car.max_drive_force);
            let a = (f_lon - car.drag_force(vk) - roll) / car.mass;
            let ds = x[c.s_off + k + 1] - x[c.s_off + k];
            let vn = (vk * vk + 2.0 * a * ds).max(v_min * v_min).sqrt();
            if vn < v[k + 1] {
                v[k + 1] = vn;
            }
        }
        if closed {
            let m = v[0].min(v[n - 1]);
            v[n - 1] = m;
        }
    }

    // backward braking passes (drag and rolling aid deceleration)
    for _ in 0..passes {
        if closed {
            let m = v[n - 1].min(v[0]);
            v[n - 1] = m;
        }
        for k in (0..n - 1).rev() {
            let vk1 = v[k + 1];
            let f_grip = car.max_grip_force(vk1);
            let f_lat = car.mass * vk1 * vk1 * x[c.cv_off + k + 1].abs();
            let f_lon = (f_grip * f_grip - f_lat * f_lat)
                .max(0.0)
                .sqrt()
                .min(car.max_brake_force);
            let a = (f_lon + car.drag_force(vk1) + roll) / car.mass;
            let ds = x[c.s_off + k + 1] - x[c.s_off + k];
            let vp = (vk1 * vk1 + 2.0 * a * ds).max(v_min * v_min).sqrt();
            if vp < v[k] {
                v[k] = vp;
            }
        }
        if closed {
            let m = v[n - 1].min(v[0]);
            v[0] = m;
        }
    }

    for k in 0..n {
        x[c.v_off + k] = v[k].max(v_min);
    }
}

/// Correct one trapezoidal interval. `target` is the node index whose state is
/// updated: `k+1` for a forward sweep, `k` for a backward sweep.
fn correct_interval(x: &mut [f64], k: usize, target: usize, c: &SweepCtx) {
    let off = [c.s_off, c.n_off, c.v_off, c.a_off];

    let state_k = [
        x[c.s_off + k],
        x[c.n_off + k],
        x[c.v_off + k],
        x[c.a_off + k],
    ];
    let state_k1 = [
        x[c.s_off + k + 1],
        x[c.n_off + k + 1],
        x[c.v_off + k + 1],
        x[c.a_off + k + 1],
    ];
    let ctrl_k = [x[c.fd_off + k], x[c.cv_off + k]];
    let ctrl_k1 = [x[c.fd_off + k + 1], x[c.cv_off + k + 1]];

    let kappa_k = c.track.curvature_at(state_k[0]);
    let kappa_k1 = c.track.curvature_at(state_k1[0]);
    let dk = point_mass_derivatives(c.car, &state_k, &ctrl_k, kappa_k);
    let dk1 = point_mass_derivatives(c.car, &state_k1, &ctrl_k1, kappa_k1);

    let half = 0.5 * x[c.dt_off + k];
    // defect = state_{k+1} - (state_k + (dt/2)(f_k + f_{k+1}))
    let mut defect = [0.0; 4];
    for (j, d) in defect.iter_mut().enumerate() {
        *d = state_k1[j] - (state_k[j] + half * (dk[j] + dk1[j]));
    }

    // Only the lateral states are corrected here: n (j = 1) and alpha (j = 3).
    // Arc length s (j = 0) is a fixed, even-spaced mesh pinned by periodicity,
    // and speed v (j = 2) is set by the grip-limited pass in Phase B. Correcting
    // s or v here creates feedback loops with the dt / speed updates that
    // destabilize the iteration.
    let lateral = [1usize, 3usize];
    if target == k + 1 {
        // forward: pull node k+1 toward the prediction
        for &j in &lateral {
            x[off[j] + k + 1] -= c.damping * defect[j];
        }
    } else {
        // backward: pull node k so the prediction reaches node k+1
        for &j in &lateral {
            x[off[j] + k] += c.damping * defect[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collocation::{CollocationConfig, CollocationOptimizer};
    use crate::solver::SolverConfig;
    use apex_physics::{qss_lap_sim, CarParams};
    use apex_track::{build_track, circle_track, oval_track};

    fn circle_opt() -> (Track, CarParams) {
        let (pts, closed) = circle_track(100.0, 12.0, 200);
        (build_track("circle", &pts, closed), CarParams::default())
    }

    fn oval_opt() -> (Track, CarParams) {
        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 400);
        (build_track("oval", &pts, closed), CarParams::default())
    }

    fn config(n_nodes: usize) -> CollocationConfig {
        CollocationConfig {
            n_nodes,
            closed: true,
            ..CollocationConfig::default()
        }
    }

    #[test]
    fn direct_circle_converges() {
        let (track, car) = circle_opt();
        let opt = CollocationOptimizer::new(config(30), &track, &car);
        let result = opt.optimize_direct(&DirectSolverConfig::default());
        assert!(result.converged, "should converge on circle");
        assert!(
            result.eq_violation < 1e-3,
            "eq_viol {}",
            result.eq_violation
        );
    }

    #[test]
    fn direct_beats_al_on_oval() {
        // Stable feasibility win over the first-order augmented-Lagrangian
        // solver. (The Gauss-Newton solver is genuinely stronger on this
        // point-mass case — see the module/PR notes — so the comparison here is
        // against AL, which the direct sweep reliably beats.)
        let (track, car) = oval_opt();
        let opt = CollocationOptimizer::new(config(50), &track, &car);

        let al = opt.optimize(&SolverConfig {
            max_outer_iter: 15,
            max_inner_iter: 30,
            constraint_tol: 1e-3,
            ..SolverConfig::default()
        });
        let direct = opt.optimize_direct(&DirectSolverConfig {
            max_iterations: 200,
            constraint_tol: 1e-4,
            ..DirectSolverConfig::default()
        });

        assert!(
            direct.eq_violation < al.eq_violation,
            "direct {:.3e} should beat AL {:.3e}",
            direct.eq_violation,
            al.eq_violation
        );
    }

    #[test]
    fn direct_violation_decreases_with_iterations() {
        let (track, car) = oval_opt();
        let opt = CollocationOptimizer::new(config(50), &track, &car);

        let run = |iters| {
            opt.optimize_direct(&DirectSolverConfig {
                max_iterations: iters,
                constraint_tol: 1e-12, // force the full iteration budget
                ..DirectSolverConfig::default()
            })
            .eq_violation
        };
        let v20 = run(20);
        let v50 = run(50);
        let v100 = run(100);

        assert!(v50 <= v20 * 1.001, "v50 {} vs v20 {}", v50, v20);
        assert!(v100 <= v50 * 1.001, "v100 {} vs v50 {}", v100, v50);
    }

    #[test]
    fn direct_oval_lap_time_near_qss() {
        let (track, car) = oval_opt();
        let qss = qss_lap_sim(&track, &car).lap_time;
        let opt = CollocationOptimizer::new(config(50), &track, &car);
        let result = opt.optimize_direct(&DirectSolverConfig::default());

        assert!(
            (result.lap_time - qss).abs() / qss < 0.05,
            "lap {} vs QSS {}",
            result.lap_time,
            qss
        );
    }
}
