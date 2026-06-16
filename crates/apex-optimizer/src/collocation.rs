//! Minimum-lap-time trajectory optimization via direct (trapezoidal)
//! collocation with the 2-DOF point-mass vehicle model.
//!
//! Decision-variable layout (length `7N - 1` for `N` nodes):
//! ```text
//!   [ s_0..s_{N-1} | n_0..n_{N-1} | v_0..v_{N-1} | alpha_0..alpha_{N-1}
//!     | f_drive_0..f_drive_{N-1} | curv_0..curv_{N-1} | dt_0..dt_{N-2} ]
//! ```
//! Block offsets: s=0, n=N, v=2N, alpha=3N, f_drive=4N, curv=5N, dt=6N.

use apex_physics::{qss_lap_sim, CarParams};
use apex_track::Track;

use crate::nlp::{NlpEvaluator, NlpProblem};
use crate::solver::{solve_nlp, SolverConfig, SolverResult};

/// Configuration for the collocation problem.
#[derive(Debug, Clone)]
pub struct CollocationConfig {
    /// Number of collocation nodes.
    pub n_nodes: usize,
    /// Whether the track is closed (periodic boundary conditions).
    pub closed: bool,
    /// Minimum allowed time step (s).
    pub dt_min: f64,
    /// Maximum allowed time step (s).
    pub dt_max: f64,
    /// Minimum speed (m/s) — prevents singularities.
    pub v_min: f64,
}

impl Default for CollocationConfig {
    fn default() -> Self {
        CollocationConfig {
            n_nodes: 100,
            closed: true,
            dt_min: 0.001,
            dt_max: 2.0,
            v_min: 5.0,
        }
    }
}

/// The collocation lap-time optimizer.
///
/// Encapsulates the NLP formulation for minimum-time trajectory optimization
/// using the point-mass vehicle model.
pub struct CollocationOptimizer<'a> {
    /// Problem configuration.
    pub config: CollocationConfig,
    /// Track the trajectory runs on.
    pub track: &'a Track,
    /// Vehicle parameters.
    pub car: &'a CarParams,
}

/// Result of the collocation optimization.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimized speeds at each node (m/s).
    pub speeds: Vec<f64>,
    /// Optimized lateral offsets at each node (m).
    pub offsets: Vec<f64>,
    /// Optimized heading angles at each node (rad).
    pub headings: Vec<f64>,
    /// Arc length stations at each node (m).
    pub stations: Vec<f64>,
    /// Optimized drive forces at each node (N).
    pub drive_forces: Vec<f64>,
    /// Optimized curvature commands at each node (1/m).
    pub curvature_cmds: Vec<f64>,
    /// Time steps between nodes (s).
    pub time_steps: Vec<f64>,
    /// Total optimized lap time (s).
    pub lap_time: f64,
    /// Whether the optimizer converged.
    pub converged: bool,
}

/// Helper struct for unpacked decision variables.
struct UnpackedVars {
    s: Vec<f64>,
    n: Vec<f64>,
    v: Vec<f64>,
    alpha: Vec<f64>,
    f_drive: Vec<f64>,
    curvature_cmd: Vec<f64>,
    dt: Vec<f64>,
}

/// Linear interpolation of `ys` (sampled at strictly increasing `xs`) at `x`,
/// clamped to the endpoints.
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

impl<'a> CollocationOptimizer<'a> {
    /// Create a new optimizer.
    pub fn new(config: CollocationConfig, track: &'a Track, car: &'a CarParams) -> Self {
        CollocationOptimizer { config, track, car }
    }

    /// Number of decision variables (`7N - 1`).
    fn n_vars(&self) -> usize {
        7 * self.config.n_nodes - 1
    }

    /// Number of equality constraints.
    fn n_eq_constraints(&self) -> usize {
        let n = self.config.n_nodes;
        4 * (n - 1) + if self.config.closed { 4 } else { 0 }
    }

    /// Number of inequality constraints.
    fn n_ineq_constraints(&self) -> usize {
        3 * self.config.n_nodes
    }

    /// Create an initial guess from the QSS solution. This warm start is
    /// essential for convergence.
    fn initial_guess(&self) -> Vec<f64> {
        let n = self.config.n_nodes;
        let length = self.track.total_length;
        let qss = qss_lap_sim(self.track, self.car);

        // s: evenly spaced over [0, length]
        let s: Vec<f64> = (0..n)
            .map(|k| length * (k as f64) / ((n - 1) as f64))
            .collect();

        // v: interpolated from the QSS speed profile (floored at v_min)
        let v: Vec<f64> = s
            .iter()
            .map(|&sk| interp(&qss.distances, &qss.speeds, sk).max(self.config.v_min))
            .collect();

        // n, alpha: centerline, aligned
        let nn = vec![0.0; n];
        let alpha = vec![0.0; n];

        // curvature_cmd: track curvature (so the path follows the track)
        let curvature_cmd: Vec<f64> = s.iter().map(|&sk| self.track.curvature_at(sk)).collect();

        // dt: ds / v_avg over each interval, clamped
        let dt: Vec<f64> = (0..n - 1)
            .map(|k| {
                let ds = s[k + 1] - s[k];
                let v_avg = 0.5 * (v[k] + v[k + 1]);
                (ds / v_avg).clamp(self.config.dt_min, self.config.dt_max)
            })
            .collect();

        // f_drive: maintain speed plus accelerate per the QSS speed change
        let drag_roll = self.car.rolling_resistance_force();
        let f_drive: Vec<f64> = (0..n)
            .map(|k| {
                let a = if k + 1 < n {
                    (v[k + 1] - v[k]) / dt[k]
                } else {
                    0.0
                };
                self.car.mass * a + self.car.drag_force(v[k]) + drag_roll
            })
            .collect();

        self.pack(&UnpackedVars {
            s,
            n: nn,
            v,
            alpha,
            f_drive,
            curvature_cmd,
            dt,
        })
    }

    /// Unpack the decision variable vector into individual arrays.
    fn unpack(&self, x: &[f64]) -> UnpackedVars {
        let n = self.config.n_nodes;
        UnpackedVars {
            s: x[0..n].to_vec(),
            n: x[n..2 * n].to_vec(),
            v: x[2 * n..3 * n].to_vec(),
            alpha: x[3 * n..4 * n].to_vec(),
            f_drive: x[4 * n..5 * n].to_vec(),
            curvature_cmd: x[5 * n..6 * n].to_vec(),
            dt: x[6 * n..].to_vec(),
        }
    }

    /// Pack individual arrays back into the decision variable vector.
    fn pack(&self, vars: &UnpackedVars) -> Vec<f64> {
        let mut x = Vec::with_capacity(self.n_vars());
        x.extend_from_slice(&vars.s);
        x.extend_from_slice(&vars.n);
        x.extend_from_slice(&vars.v);
        x.extend_from_slice(&vars.alpha);
        x.extend_from_slice(&vars.f_drive);
        x.extend_from_slice(&vars.curvature_cmd);
        x.extend_from_slice(&vars.dt);
        x
    }

    /// Build the NLP problem definition (dimensions and variable bounds).
    fn build_nlp_problem(&self) -> NlpProblem {
        let n = self.config.n_nodes;
        let n_vars = self.n_vars();

        let mut lower = vec![f64::NEG_INFINITY; n_vars];
        let mut upper = vec![f64::INFINITY; n_vars];

        // speeds: lower bound at v_min
        for lb in lower.iter_mut().take(3 * n).skip(2 * n) {
            *lb = self.config.v_min;
        }
        // drive force: braking/drive limits
        for k in 0..n {
            lower[4 * n + k] = -self.car.max_brake_force;
            upper[4 * n + k] = self.car.max_drive_force;
        }
        // time steps: [dt_min, dt_max]
        for k in 6 * n..n_vars {
            lower[k] = self.config.dt_min;
            upper[k] = self.config.dt_max;
        }

        NlpProblem {
            n_vars,
            n_eq: self.n_eq_constraints(),
            n_ineq: self.n_ineq_constraints(),
            lower_bounds: lower,
            upper_bounds: upper,
        }
    }

    /// Extract a structured result from the raw solver output.
    fn extract_result(&self, solver_result: &SolverResult) -> OptimizationResult {
        let vars = self.unpack(&solver_result.x);
        let lap_time = vars.dt.iter().sum();
        OptimizationResult {
            speeds: vars.v,
            offsets: vars.n,
            headings: vars.alpha,
            stations: vars.s,
            drive_forces: vars.f_drive,
            curvature_cmds: vars.curvature_cmd,
            time_steps: vars.dt,
            lap_time,
            converged: solver_result.converged,
        }
    }

    /// Run the optimization.
    pub fn optimize(&self, solver_config: &SolverConfig) -> OptimizationResult {
        let x0 = self.initial_guess();
        let problem = self.build_nlp_problem();
        let evaluator = CollocationEvaluator { optimizer: self };
        let result = solve_nlp(&problem, &evaluator, &x0, solver_config);
        self.extract_result(&result)
    }
}

/// Implements [`NlpEvaluator`] for the collocation problem.
struct CollocationEvaluator<'a, 'b> {
    optimizer: &'a CollocationOptimizer<'b>,
}

impl NlpEvaluator for CollocationEvaluator<'_, '_> {
    fn objective(&self, x: &[f64]) -> f64 {
        // Sum of all dt_k (total lap time).
        let dt_start = 6 * self.optimizer.config.n_nodes;
        x[dt_start..].iter().sum()
    }

    fn objective_gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut grad = vec![0.0; x.len()];
        let dt_start = 6 * self.optimizer.config.n_nodes;
        for g in &mut grad[dt_start..] {
            *g = 1.0;
        }
        grad
    }

    fn equality_constraints(&self, x: &[f64]) -> Vec<f64> {
        let opt = self.optimizer;
        let n = opt.config.n_nodes;
        let vars = opt.unpack(x);

        let mut constraints = Vec::with_capacity(opt.n_eq_constraints());

        // Trapezoidal dynamics defects over each interval.
        for k in 0..n - 1 {
            let kappa_k = opt.track.curvature_at(vars.s[k]);
            let kappa_k1 = opt.track.curvature_at(vars.s[k + 1]);

            let state_k = [vars.s[k], vars.n[k], vars.v[k], vars.alpha[k]];
            let state_k1 = [vars.s[k + 1], vars.n[k + 1], vars.v[k + 1], vars.alpha[k + 1]];
            let control_k = [vars.f_drive[k], vars.curvature_cmd[k]];
            let control_k1 = [vars.f_drive[k + 1], vars.curvature_cmd[k + 1]];

            let deriv_k = point_mass_derivatives(opt.car, &state_k, &control_k, kappa_k);
            let deriv_k1 = point_mass_derivatives(opt.car, &state_k1, &control_k1, kappa_k1);

            let half_dt = vars.dt[k] / 2.0;
            for j in 0..4 {
                constraints.push(state_k1[j] - state_k[j] - half_dt * (deriv_k[j] + deriv_k1[j]));
            }
        }

        // Periodicity (closed tracks).
        if opt.config.closed {
            constraints.push(vars.s[n - 1] - opt.track.total_length);
            constraints.push(vars.n[n - 1] - vars.n[0]);
            constraints.push(vars.v[n - 1] - vars.v[0]);
            constraints.push(apex_track::normalize_angle(vars.alpha[n - 1] - vars.alpha[0]));
        }

        constraints
    }

    fn inequality_constraints(&self, x: &[f64]) -> Vec<f64> {
        let opt = self.optimizer;
        let n = opt.config.n_nodes;
        let vars = opt.unpack(x);

        let mut constraints = Vec::with_capacity(opt.n_ineq_constraints());

        for k in 0..n {
            // Track boundaries.
            let (wl, wr) = opt.track.width_at(vars.s[k]);
            constraints.push(vars.n[k] - wl); // n - wl <= 0
            constraints.push(-wr - vars.n[k]); // -wr - n <= 0

            // Grip circle.
            let v = vars.v[k];
            let f_grip = opt.car.max_grip_force(v);
            if f_grip > 0.0 {
                let f_lon = vars.f_drive[k];
                let f_lat = opt.car.mass * v * v * vars.curvature_cmd[k];
                constraints.push((f_lon / f_grip).powi(2) + (f_lat / f_grip).powi(2) - 1.0);
            } else {
                constraints.push(-1.0);
            }
        }

        constraints
    }

    fn equality_jacobian(&self, x: &[f64]) -> apex_math::CsrMatrix {
        numerical_jacobian(x, self.optimizer.n_eq_constraints(), |x| {
            self.equality_constraints(x)
        })
    }

    fn inequality_jacobian(&self, x: &[f64]) -> apex_math::CsrMatrix {
        numerical_jacobian(x, self.optimizer.n_ineq_constraints(), |x| {
            self.inequality_constraints(x)
        })
    }
}

/// Evaluate point-mass dynamics without constructing the ODE system struct.
///
/// `state` is `[s, n, v, alpha]`, `control` is `[f_drive, curvature_cmd]`, and
/// `kappa` is the track curvature at the node.
fn point_mass_derivatives(car: &CarParams, state: &[f64; 4], control: &[f64; 2], kappa: f64) -> [f64; 4] {
    let n = state[1];
    let v = state[2];
    let alpha = state[3];
    let f_drive = control[0];
    let curvature_cmd = control[1];

    let f_drag = car.drag_force(v);
    let f_roll = car.rolling_resistance_force();
    let v_safe = v.max(0.1);

    let ds_dt = v_safe * alpha.cos() / (1.0 - n * kappa);
    let dn_dt = v_safe * alpha.sin();
    let dv_dt = (f_drive - f_drag - f_roll) / car.mass;
    let dalpha_dt = curvature_cmd * v_safe - kappa * ds_dt;

    [ds_dt, dn_dt, dv_dt, dalpha_dt]
}

/// Compute a Jacobian numerically using central finite differences.
fn numerical_jacobian(
    x: &[f64],
    n_constraints: usize,
    eval: impl Fn(&[f64]) -> Vec<f64>,
) -> apex_math::CsrMatrix {
    let eps = 1e-7;
    let n_vars = x.len();
    let mut builder = apex_math::CsrBuilder::new(n_constraints, n_vars);

    let mut x_pert = x.to_vec();

    for j in 0..n_vars {
        let x_orig = x_pert[j];

        x_pert[j] = x_orig + eps;
        let f_plus = eval(&x_pert);

        x_pert[j] = x_orig - eps;
        let f_minus = eval(&x_pert);

        x_pert[j] = x_orig;

        for i in 0..n_constraints {
            let deriv = (f_plus[i] - f_minus[i]) / (2.0 * eps);
            if deriv.abs() > 1e-12 {
                builder.add(i, j, deriv);
            }
        }
    }

    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_track::{build_track, circle_track, oval_track};

    fn circle(n_nodes: usize) -> (Track, CarParams, CollocationConfig) {
        let (pts, closed) = circle_track(100.0, 12.0, 200);
        let track = build_track("circle", &pts, closed);
        let car = CarParams::default();
        let config = CollocationConfig {
            n_nodes,
            closed: true,
            ..CollocationConfig::default()
        };
        (track, car, config)
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let (track, car, config) = circle(20);
        let opt = CollocationOptimizer::new(config, &track, &car);
        let x = opt.initial_guess();
        let repacked = opt.pack(&opt.unpack(&x));
        assert_eq!(x.len(), repacked.len());
        for (a, b) in x.iter().zip(repacked.iter()) {
            assert!((a - b).abs() < 1e-12, "{} vs {}", a, b);
        }
    }

    #[test]
    fn initial_guess_is_valid() {
        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 400);
        let track = build_track("oval", &pts, closed);
        let car = CarParams::default();
        let config = CollocationConfig {
            n_nodes: 50,
            ..CollocationConfig::default()
        };
        let opt = CollocationOptimizer::new(config, &track, &car);
        let vars = opt.unpack(&opt.initial_guess());

        assert!(vars.v.iter().all(|&v| v > 0.0), "all speeds positive");
        assert!(vars.dt.iter().all(|&dt| dt > 0.0), "all dt positive");
        assert!(vars.n.iter().all(|&n| n == 0.0), "n on centerline");
        for k in 1..vars.s.len() {
            assert!(vars.s[k] > vars.s[k - 1], "s monotonic at {}", k);
        }
    }

    #[test]
    fn constraints_evaluate() {
        let (track, car, config) = circle(30);
        let opt = CollocationOptimizer::new(config, &track, &car);
        let evaluator = CollocationEvaluator { optimizer: &opt };
        let x = opt.initial_guess();

        let eq = evaluator.equality_constraints(&x);
        let ineq = evaluator.inequality_constraints(&x);
        assert_eq!(eq.len(), opt.n_eq_constraints());
        assert_eq!(ineq.len(), opt.n_ineq_constraints());

        // QSS warm start is approximately dynamically consistent
        let max_defect = eq.iter().fold(0.0_f64, |m, &c| m.max(c.abs()));
        assert!(max_defect.is_finite(), "defect not finite");
        assert!(max_defect < 100.0, "defect {} unexpectedly large", max_defect);
    }

    #[test]
    fn small_optimization_circle() {
        let (track, car, config) = circle(30);
        let opt = CollocationOptimizer::new(config, &track, &car);
        let qss_lap = qss_lap_sim(&track, &car).lap_time;

        let solver_config = SolverConfig {
            max_outer_iter: 10,
            max_inner_iter: 25,
            constraint_tol: 1e-2,
            ..SolverConfig::default()
        };
        let result = opt.optimize(&solver_config);

        assert!(result.lap_time.is_finite(), "lap time not finite");
        assert!(
            (result.lap_time - qss_lap).abs() / qss_lap < 0.20,
            "lap time {} vs QSS {}",
            result.lap_time,
            qss_lap
        );
        assert!(result.speeds.iter().all(|&v| v > 0.0), "speeds positive");
        // track boundaries respected (circle width 6 each side)
        assert!(
            result.offsets.iter().all(|&nn| nn.abs() <= 6.0 + 1e-3),
            "offset within track"
        );
    }

    #[test]
    fn optimization_no_nan_oval() {
        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 400);
        let track = build_track("oval", &pts, closed);
        let car = CarParams::default();
        let config = CollocationConfig {
            n_nodes: 50,
            ..CollocationConfig::default()
        };
        let opt = CollocationOptimizer::new(config, &track, &car);
        let qss_lap = qss_lap_sim(&track, &car).lap_time;

        let solver_config = SolverConfig {
            max_outer_iter: 10,
            max_inner_iter: 15,
            constraint_tol: 1e-2,
            ..SolverConfig::default()
        };
        let result = opt.optimize(&solver_config);

        // no NaNs anywhere
        assert!(result.lap_time.is_finite() && !result.lap_time.is_nan());
        for &v in &result.speeds {
            assert!(v.is_finite(), "speed not finite");
        }
        // optimizer should not make the lap dramatically worse than QSS
        assert!(
            result.lap_time <= qss_lap * 1.10,
            "lap time {} much worse than QSS {}",
            result.lap_time,
            qss_lap
        );
    }
}
