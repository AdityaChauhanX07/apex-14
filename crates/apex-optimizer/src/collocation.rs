//! Minimum-lap-time trajectory optimization via direct (trapezoidal)
//! collocation with the 2-DOF point-mass vehicle model.
//!
//! Decision-variable layout (length `7N - 1` for `N` nodes):
//! ```text
//!   [ s_0..s_{N-1} | n_0..n_{N-1} | v_0..v_{N-1} | alpha_0..alpha_{N-1}
//!     | f_drive_0..f_drive_{N-1} | curv_0..curv_{N-1} | dt_0..dt_{N-2} ]
//! ```
//! Block offsets: s=0, n=N, v=2N, alpha=3N, f_drive=4N, curv=5N, dt=6N.
//!
//! When `optimize_brake_bias` is set, an extra `brake_bias` block is inserted
//! between `curv` and `dt`, giving length `8N - 1` and shifting `dt` to `7N`.

use apex_math::{CsrBuilder, CsrMatrix, Dual, Float};
use apex_physics::car_params::GRAVITY;
use apex_physics::{
    qss_lap_sim, qss_lap_sim_tire, smooth_min, AeroModel, CarParams, PacejkaTire, SuspensionSystem,
};
use apex_track::Track;

/// Front roll-stiffness fraction used for four-corner load transfer in the
/// 7-DOF tire formulation.
const ROLL_STIFFNESS_FRONT: f64 = 0.55;

use crate::nlp::{NlpEvaluator, NlpProblem};
use crate::solver::{solve_nlp, SolverConfig, SolverResult};

/// Collocation discretization scheme for the dynamics defects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CollocationMethod {
    /// Second-order trapezoidal collocation (existing method).
    Trapezoidal,
    /// Fourth-order Hermite-Simpson (separated) collocation.
    /// Adds a midpoint evaluation per interval for higher accuracy.
    HermiteSimpson,
}

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
    /// Collocation discretization scheme for the dynamics defects.
    pub method: CollocationMethod,
    /// Whether to optimize brake bias as an additional control variable.
    /// When true, adds one variable per node (brake_bias_front, 0.5 to 0.8).
    pub optimize_brake_bias: bool,
}

impl Default for CollocationConfig {
    fn default() -> Self {
        CollocationConfig {
            n_nodes: 100,
            closed: true,
            dt_min: 0.001,
            dt_max: 2.0,
            v_min: 5.0,
            // Hermite-Simpson is strictly better when the Jacobian is available
            // (which it is here, via forward-mode autodiff).
            method: CollocationMethod::HermiteSimpson,
            optimize_brake_bias: false,
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
    /// Maximum equality (dynamics-defect) violation at the solution.
    pub eq_violation: f64,
    /// Whether the optimizer converged.
    pub converged: bool,
    /// Optimized brake bias at each node (if optimize_brake_bias was true).
    pub brake_bias: Option<Vec<f64>>,
}

/// Helper struct for unpacked decision variables.
struct UnpackedVars {
    s: Vec<f64>,
    n: Vec<f64>,
    v: Vec<f64>,
    alpha: Vec<f64>,
    f_drive: Vec<f64>,
    curvature_cmd: Vec<f64>,
    /// Per-node brake bias, present only when `optimize_brake_bias` is set.
    brake_bias: Option<Vec<f64>>,
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

    /// Number of decision variables: `7N - 1`, or `8N - 1` when brake bias is
    /// optimized.
    fn n_vars(&self) -> usize {
        self.dt_offset() + (self.config.n_nodes - 1)
    }

    /// Start index of the `dt` block in the decision vector. The optional
    /// `brake_bias` block sits between `curv` and `dt`, so `dt` shifts from `6N`
    /// to `7N` when brake bias is optimized.
    fn dt_offset(&self) -> usize {
        let n = self.config.n_nodes;
        if self.config.optimize_brake_bias {
            7 * n
        } else {
            6 * n
        }
    }

    /// Start index of the optional `brake_bias` block (always `6N`; only present
    /// when `optimize_brake_bias` is set).
    fn brake_bias_offset(&self) -> usize {
        6 * self.config.n_nodes
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

    /// Evenly spaced arc-length stations over `[0, total_length]`.
    fn node_stations(&self) -> Vec<f64> {
        let n = self.config.n_nodes;
        let length = self.track.total_length;
        (0..n)
            .map(|k| length * (k as f64) / ((n - 1) as f64))
            .collect()
    }

    /// Assemble a full decision vector from arc-length stations and a speed
    /// profile: centerline (`n = 0`), aligned (`alpha = 0`), track-following
    /// curvature command, and consistent `dt` / `f_drive`.
    fn guess_from_speeds(&self, s: Vec<f64>, v: Vec<f64>) -> Vec<f64> {
        let n = self.config.n_nodes;
        let nn = vec![0.0; n];
        let alpha = vec![0.0; n];
        let curvature_cmd: Vec<f64> = s.iter().map(|&sk| self.track.curvature_at(sk)).collect();

        let dt: Vec<f64> = (0..n - 1)
            .map(|k| {
                let ds = s[k + 1] - s[k];
                let v_avg = 0.5 * (v[k] + v[k + 1]);
                (ds / v_avg).clamp(self.config.dt_min, self.config.dt_max)
            })
            .collect();

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

        let brake_bias = self.default_brake_bias();

        self.pack(&UnpackedVars {
            s,
            n: nn,
            v,
            alpha,
            f_drive,
            curvature_cmd,
            brake_bias,
            dt,
        })
    }

    /// Seed brake-bias block (the car's static front bias at every node), or
    /// `None` when brake bias is not being optimized.
    fn default_brake_bias(&self) -> Option<Vec<f64>> {
        if self.config.optimize_brake_bias {
            Some(vec![self.car.brake_bias_front; self.config.n_nodes])
        } else {
            None
        }
    }

    /// Create an initial guess from the grip-circle QSS solution. This warm
    /// start is essential for convergence.
    fn initial_guess(&self) -> Vec<f64> {
        let qss = qss_lap_sim(self.track, self.car);
        let s = self.node_stations();
        let v: Vec<f64> = s
            .iter()
            .map(|&sk| interp(&qss.distances, &qss.speeds, sk).max(self.config.v_min))
            .collect();
        self.guess_from_speeds(s, v)
    }

    /// Create an initial guess from the tire-aware (load-sensitive) QSS solution.
    /// Its conservative speed profile is feasible for the 7-DOF tire model, so it
    /// is a much better warm start than the grip-circle guess.
    fn initial_guess_seven_dof(&self, tire: &PacejkaTire) -> Vec<f64> {
        let qss = qss_lap_sim_tire(self.track, self.car, tire, ROLL_STIFFNESS_FRONT);
        let s = self.node_stations();
        let v: Vec<f64> = s
            .iter()
            .map(|&sk| interp(&qss.distances, &qss.speeds, sk).max(self.config.v_min))
            .collect();
        self.guess_from_speeds(s, v)
    }

    /// Create an initial guess from an existing optimization result (for mesh
    /// refinement warm-starting).
    ///
    /// Interpolates the result's per-node fields (speed, lateral offset, heading,
    /// drive force, curvature command) onto this optimizer's mesh — which may
    /// have a different node count — and recomputes `dt` from the new spacing.
    pub fn initial_guess_from_result(&self, result: &OptimizationResult) -> Vec<f64> {
        let n = self.config.n_nodes;
        let s_fine = self.node_stations();
        let src = &result.stations;

        let v: Vec<f64> = s_fine
            .iter()
            .map(|&s| interp(src, &result.speeds, s).max(self.config.v_min))
            .collect();
        let nn: Vec<f64> = s_fine
            .iter()
            .map(|&s| interp(src, &result.offsets, s))
            .collect();
        let alpha: Vec<f64> = s_fine
            .iter()
            .map(|&s| interp(src, &result.headings, s))
            .collect();
        let f_drive: Vec<f64> = s_fine
            .iter()
            .map(|&s| interp(src, &result.drive_forces, s))
            .collect();
        let curvature_cmd: Vec<f64> = s_fine
            .iter()
            .map(|&s| interp(src, &result.curvature_cmds, s))
            .collect();

        let dt: Vec<f64> = (0..n - 1)
            .map(|k| {
                let ds = s_fine[k + 1] - s_fine[k];
                let v_avg = 0.5 * (v[k] + v[k + 1]);
                (ds / v_avg).clamp(self.config.dt_min, self.config.dt_max)
            })
            .collect();

        // Brake bias: interpolate from the source result if it carried one,
        // otherwise fall back to the static seed.
        let brake_bias = if self.config.optimize_brake_bias {
            Some(match &result.brake_bias {
                Some(bb) => s_fine.iter().map(|&s| interp(src, bb, s)).collect(),
                None => vec![self.car.brake_bias_front; n],
            })
        } else {
            None
        };

        self.pack(&UnpackedVars {
            s: s_fine,
            n: nn,
            v,
            alpha,
            f_drive,
            curvature_cmd,
            brake_bias,
            dt,
        })
    }

    /// Run the Gauss-Newton solver from an explicit initial guess `x0` (used for
    /// mesh-refinement warm starts, where `x0` is interpolated from a coarser
    /// solution rather than the QSS warm start).
    pub fn optimize_gn_from(
        &self,
        x0: &[f64],
        config: &crate::gauss_newton::GaussNewtonConfig,
    ) -> OptimizationResult {
        let problem = self.build_nlp_problem();
        let evaluator = CollocationEvaluator { optimizer: self };
        let result = crate::gauss_newton::solve_gauss_newton(&problem, &evaluator, x0, config);
        self.extract_result_gn(&result)
    }

    /// Unpack the decision variable vector into individual arrays.
    fn unpack(&self, x: &[f64]) -> UnpackedVars {
        let n = self.config.n_nodes;
        let dt_start = self.dt_offset();
        let brake_bias = if self.config.optimize_brake_bias {
            let bb = self.brake_bias_offset();
            Some(x[bb..bb + n].to_vec())
        } else {
            None
        };
        UnpackedVars {
            s: x[0..n].to_vec(),
            n: x[n..2 * n].to_vec(),
            v: x[2 * n..3 * n].to_vec(),
            alpha: x[3 * n..4 * n].to_vec(),
            f_drive: x[4 * n..5 * n].to_vec(),
            curvature_cmd: x[5 * n..6 * n].to_vec(),
            brake_bias,
            dt: x[dt_start..].to_vec(),
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
        // Optional brake-bias block sits between curv and dt.
        if let Some(bb) = &vars.brake_bias {
            x.extend_from_slice(bb);
        }
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
        // brake bias (optional): [0.50, 0.80]
        if self.config.optimize_brake_bias {
            let bb = self.brake_bias_offset();
            for k in 0..n {
                lower[bb + k] = 0.50;
                upper[bb + k] = 0.80;
            }
        }
        // time steps: [dt_min, dt_max]
        for k in self.dt_offset()..n_vars {
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
            eq_violation: solver_result.eq_violation,
            converged: solver_result.converged,
            brake_bias: vars.brake_bias,
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

    /// Run optimization using the Gauss-Newton solver.
    pub fn optimize_gn(
        &self,
        config: &crate::gauss_newton::GaussNewtonConfig,
    ) -> OptimizationResult {
        let x0 = self.initial_guess();
        let problem = self.build_nlp_problem();
        let evaluator = CollocationEvaluator { optimizer: self };
        let result = crate::gauss_newton::solve_gauss_newton(&problem, &evaluator, &x0, config);
        self.extract_result_gn(&result)
    }

    /// Optimize with brake bias as an additional control variable.
    ///
    /// Adds one bounded variable per node (`brake_bias_front` in `[0.50, 0.80]`)
    /// and reports the chosen bias in [`OptimizationResult::brake_bias`]. In the
    /// point-mass model the bias does not couple to the dynamics, so the solver
    /// is free to place it anywhere in range; the coupling (and the resulting
    /// front/rear trade-off) appears once a tire model is used.
    pub fn optimize_with_brake_bias(
        &self,
        solver_config: &crate::gauss_newton::GaussNewtonConfig,
    ) -> OptimizationResult {
        let mut cfg = self.config.clone();
        cfg.optimize_brake_bias = true;
        let opt = CollocationOptimizer::new(cfg, self.track, self.car);
        opt.optimize_gn(solver_config)
    }

    /// Grip-circle QSS warm-start decision vector (public for benchmarking and
    /// introspection).
    pub fn warm_start(&self) -> Vec<f64> {
        self.initial_guess()
    }

    /// Equality-constraint residuals at `x` for the point-mass formulation
    /// (public for benchmarking; e.g. to time a numerical Jacobian).
    pub fn equality_residuals(&self, x: &[f64]) -> Vec<f64> {
        CollocationEvaluator { optimizer: self }.equality_constraints(x)
    }

    /// Auto-diff equality Jacobian at `x` for the point-mass formulation
    /// (public for benchmarking).
    pub fn equality_jacobian(&self, x: &[f64]) -> CsrMatrix {
        CollocationEvaluator { optimizer: self }.equality_jacobian(x)
    }

    /// Number of equality constraints (public for benchmarking/introspection).
    pub fn equality_count(&self) -> usize {
        self.n_eq_constraints()
    }

    fn extract_result_gn(
        &self,
        result: &crate::gauss_newton::GaussNewtonResult,
    ) -> OptimizationResult {
        let vars = self.unpack(&result.x);
        OptimizationResult {
            speeds: vars.v.clone(),
            offsets: vars.n.clone(),
            headings: vars.alpha.clone(),
            stations: vars.s.clone(),
            drive_forces: vars.f_drive.clone(),
            curvature_cmds: vars.curvature_cmd.clone(),
            time_steps: vars.dt.clone(),
            lap_time: result.objective,
            eq_violation: result.eq_violation,
            converged: result.converged,
            brake_bias: vars.brake_bias.clone(),
        }
    }

    /// Optimize using the 7-DOF tire model for force limits: Pacejka
    /// combined-slip forces with four-corner load-sensitive grip instead of the
    /// simple grip circle. Solved with the Gauss-Newton solver.
    pub fn optimize_seven_dof(
        &self,
        tire: &PacejkaTire,
        solver_config: &crate::gauss_newton::GaussNewtonConfig,
    ) -> OptimizationResult {
        let x0 = self.initial_guess_seven_dof(tire);
        let problem = self.build_nlp_problem();
        let evaluator = SevenDofEvaluator {
            optimizer: self,
            tire,
        };
        let result =
            crate::gauss_newton::solve_gauss_newton(&problem, &evaluator, &x0, solver_config);
        self.extract_result_gn(&result)
    }

    /// Optimize using the full 14-DOF force model for grip limits.
    ///
    /// The optimizer works in the curvilinear 4-state formulation (s, n, v, alpha)
    /// but computes force limits using:
    /// - Four-corner vertical loads with lateral AND longitudinal weight transfer
    /// - Pacejka combined-slip tire forces with load sensitivity
    /// - Ride-height-sensitive aerodynamic downforce
    /// - Suspension static equilibrium to estimate ride heights from load distribution
    ///
    /// The dynamics defects reuse the autodiffed 7-DOF curvilinear equations (the
    /// longitudinal balance is unchanged by ride-height effects), while the grip
    /// inequality at each node uses the 14-DOF budget, so the binding cornering
    /// limit reflects the ride-height-coupled downforce. After optimization, the
    /// result can be forward-simulated with the full 14-DOF model.
    pub fn optimize_fourteen_dof(
        &self,
        tire: &PacejkaTire,
        suspension: &SuspensionSystem,
        aero: &AeroModel,
        solver_config: &crate::gauss_newton::GaussNewtonConfig,
    ) -> OptimizationResult {
        let x0 = self.initial_guess_seven_dof(tire);
        let problem = self.build_nlp_problem();
        let evaluator = FourteenDofEvaluator {
            optimizer: self,
            tire,
            suspension,
            aero,
        };
        let result =
            crate::gauss_newton::solve_gauss_newton(&problem, &evaluator, &x0, solver_config);
        self.extract_result_gn(&result)
    }

    /// Full 14-DOF optimization pipeline:
    /// 1. Optimize racing line with 14-DOF force model (reduced collocation)
    /// 2. Forward-simulate full 14-DOF model along optimized trajectory
    /// 3. Return both the optimization result and the detailed telemetry
    pub fn optimize_fourteen_dof_full(
        &self,
        tire: &PacejkaTire,
        suspension: &SuspensionSystem,
        aero: &AeroModel,
        solver_config: &crate::gauss_newton::GaussNewtonConfig,
    ) -> (OptimizationResult, crate::forward_sim::DetailedTelemetry) {
        let opt = self.optimize_fourteen_dof(tire, suspension, aero, solver_config);
        let simulator = crate::forward_sim::ForwardSimulator {
            params: self.car,
            tire,
            suspension,
            aero,
            track: self.track,
        };
        let telemetry = simulator.simulate(&opt);
        (opt, telemetry)
    }

    /// Run optimization using the sequential defect-correction (direct) solver.
    pub fn optimize_direct(
        &self,
        config: &crate::direct_solver::DirectSolverConfig,
    ) -> OptimizationResult {
        let x0 = self.initial_guess();
        let problem = self.build_nlp_problem();
        let evaluator = CollocationEvaluator { optimizer: self };
        let structure = crate::direct_solver::CollocationStructure {
            n_nodes: self.config.n_nodes,
            n_states: 4,
            n_controls: 2,
        };
        let result = crate::direct_solver::solve_direct(
            &problem, &evaluator, &x0, config, structure, self.track, self.car,
        );
        self.extract_result_direct(&result)
    }

    fn extract_result_direct(
        &self,
        result: &crate::direct_solver::DirectSolverResult,
    ) -> OptimizationResult {
        let vars = self.unpack(&result.x);
        OptimizationResult {
            speeds: vars.v.clone(),
            offsets: vars.n.clone(),
            headings: vars.alpha.clone(),
            stations: vars.s.clone(),
            drive_forces: vars.f_drive.clone(),
            curvature_cmds: vars.curvature_cmd.clone(),
            time_steps: vars.dt.clone(),
            lap_time: result.objective,
            eq_violation: result.eq_violation,
            converged: result.converged,
            brake_bias: vars.brake_bias.clone(),
        }
    }
}

/// Implements [`NlpEvaluator`] for the collocation problem.
struct CollocationEvaluator<'a, 'b> {
    optimizer: &'a CollocationOptimizer<'b>,
}

impl NlpEvaluator for CollocationEvaluator<'_, '_> {
    fn objective(&self, x: &[f64]) -> f64 {
        // Sum of all dt_k (total lap time).
        let dt_start = self.optimizer.dt_offset();
        x[dt_start..].iter().sum()
    }

    fn objective_gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut grad = vec![0.0; x.len()];
        let dt_start = self.optimizer.dt_offset();
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

        // Dynamics defects over each interval, per the configured method.
        for k in 0..n - 1 {
            let kappa_k = opt.track.curvature_at(vars.s[k]);
            let kappa_k1 = opt.track.curvature_at(vars.s[k + 1]);

            let defects = match opt.config.method {
                CollocationMethod::Trapezoidal => {
                    let state_k = [vars.s[k], vars.n[k], vars.v[k], vars.alpha[k]];
                    let state_k1 = [
                        vars.s[k + 1],
                        vars.n[k + 1],
                        vars.v[k + 1],
                        vars.alpha[k + 1],
                    ];
                    let control_k = [vars.f_drive[k], vars.curvature_cmd[k]];
                    let control_k1 = [vars.f_drive[k + 1], vars.curvature_cmd[k + 1]];

                    let deriv_k = point_mass_derivatives(opt.car, &state_k, &control_k, kappa_k);
                    let deriv_k1 =
                        point_mass_derivatives(opt.car, &state_k1, &control_k1, kappa_k1);

                    let half_dt = vars.dt[k] / 2.0;
                    std::array::from_fn(|j| {
                        state_k1[j] - state_k[j] - half_dt * (deriv_k[j] + deriv_k1[j])
                    })
                }
                CollocationMethod::HermiteSimpson => hermite_simpson_defect(
                    opt.car,
                    vars.s[k],
                    vars.n[k],
                    vars.v[k],
                    vars.alpha[k],
                    vars.f_drive[k],
                    vars.curvature_cmd[k],
                    vars.s[k + 1],
                    vars.n[k + 1],
                    vars.v[k + 1],
                    vars.alpha[k + 1],
                    vars.f_drive[k + 1],
                    vars.curvature_cmd[k + 1],
                    vars.dt[k],
                    kappa_k,
                    kappa_k1,
                ),
            };
            constraints.extend_from_slice(&defects);
        }

        // Periodicity (closed tracks).
        if opt.config.closed {
            constraints.push(vars.s[n - 1] - opt.track.total_length);
            constraints.push(vars.n[n - 1] - vars.n[0]);
            constraints.push(vars.v[n - 1] - vars.v[0]);
            constraints.push(apex_track::normalize_angle(
                vars.alpha[n - 1] - vars.alpha[0],
            ));
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

    fn equality_jacobian(&self, x: &[f64]) -> CsrMatrix {
        self.autodiff_equality_jacobian(x)
    }

    fn inequality_jacobian(&self, x: &[f64]) -> CsrMatrix {
        self.autodiff_inequality_jacobian(x)
    }
}

impl CollocationEvaluator<'_, '_> {
    /// Exact equality-constraint Jacobian via forward-mode autodiff.
    ///
    /// Exploits the banded structure: each interval-`k` defect depends only on
    /// the 13 variables at nodes `k` and `k+1` plus `dt_k`. The track curvature
    /// is held fixed at each node (its dependence on `s` is neglected, exactly
    /// as the trapezoidal defect treats it), so on constant-curvature stretches
    /// this matches finite differences to machine precision.
    fn autodiff_equality_jacobian(&self, x: &[f64]) -> CsrMatrix {
        let opt = self.optimizer;
        let n = opt.config.n_nodes;
        let n_vars = x.len();
        let n_defects = 4 * (n - 1);
        let n_eq = opt.n_eq_constraints();

        let mut builder = CsrBuilder::new(n_eq, n_vars);
        let vars = opt.unpack(x);

        for k in 0..n - 1 {
            let kappa_k = opt.track.curvature_at(vars.s[k]);
            let kappa_k1 = opt.track.curvature_at(vars.s[k + 1]);

            let node_k = [
                vars.s[k],
                vars.n[k],
                vars.v[k],
                vars.alpha[k],
                vars.f_drive[k],
                vars.curvature_cmd[k],
            ];
            let node_k1 = [
                vars.s[k + 1],
                vars.n[k + 1],
                vars.v[k + 1],
                vars.alpha[k + 1],
                vars.f_drive[k + 1],
                vars.curvature_cmd[k + 1],
            ];

            // Global column index of each of the 13 local variables.
            let global_indices = [
                k,
                n + k,
                2 * n + k,
                3 * n + k,
                4 * n + k,
                5 * n + k,
                k + 1,
                n + k + 1,
                2 * n + k + 1,
                3 * n + k + 1,
                4 * n + k + 1,
                5 * n + k + 1,
                opt.dt_offset() + k,
            ];

            let state_k = [node_k[0], node_k[1], node_k[2], node_k[3]];
            let control_k = [node_k[4], node_k[5]];
            let state_k1 = [node_k1[0], node_k1[1], node_k1[2], node_k1[3]];
            let control_k1 = [node_k1[4], node_k1[5]];

            for (wrt, &col) in global_indices.iter().enumerate() {
                let defects = match opt.config.method {
                    CollocationMethod::Trapezoidal => defect_with_dual(
                        opt.car,
                        &node_k,
                        &node_k1,
                        vars.dt[k],
                        [kappa_k, kappa_k1],
                        wrt,
                    ),
                    CollocationMethod::HermiteSimpson => hermite_simpson_defect_with_dual(
                        opt.car,
                        &state_k,
                        &control_k,
                        &state_k1,
                        &control_k1,
                        vars.dt[k],
                        kappa_k,
                        kappa_k1,
                        wrt,
                    ),
                };
                for (j, d) in defects.iter().enumerate() {
                    if d.dual.abs() > 1e-15 {
                        builder.add(4 * k + j, col, d.dual);
                    }
                }
            }

            // Curvature-chain correction for the s-columns: the defects depend
            // on s through κ(s), which the dual-defect routines hold constant.
            // Add (∂defect/∂κ)·(dκ/ds). On constant-curvature stretches dκ/ds ≈ 0
            // so this vanishes; at corner entry/exit it matters and makes the
            // Gauss-Newton step effective.
            let h = 1e-3;
            let dkk_ds = (opt.track.curvature_at(node_k[0] + h)
                - opt.track.curvature_at(node_k[0] - h))
                / (2.0 * h);
            let dkk1_ds = (opt.track.curvature_at(node_k1[0] + h)
                - opt.track.curvature_at(node_k1[0] - h))
                / (2.0 * h);
            let (ddef_dkk, ddef_dkk1): ([f64; 4], [f64; 4]) = match opt.config.method {
                CollocationMethod::Trapezoidal => {
                    // ∂defect/∂κ = -(dt/2)·∂f/∂κ (the trapezoidal weight).
                    let half_dt = vars.dt[k] / 2.0;
                    let dfk = dynamics_dkappa(opt.car, &node_k, kappa_k);
                    let dfk1 = dynamics_dkappa(opt.car, &node_k1, kappa_k1);
                    (
                        std::array::from_fn(|j| -half_dt * dfk[j]),
                        std::array::from_fn(|j| -half_dt * dfk1[j]),
                    )
                }
                CollocationMethod::HermiteSimpson => {
                    // The midpoint couples both nodal curvatures through κ_mid and
                    // x_mid, so differentiate the whole defect w.r.t. each passed
                    // curvature by central differences (skipped where dκ/ds = 0).
                    let hs = |kk: f64, kk1: f64| {
                        hermite_simpson_defect(
                            opt.car, node_k[0], node_k[1], node_k[2], node_k[3], node_k[4],
                            node_k[5], node_k1[0], node_k1[1], node_k1[2], node_k1[3], node_k1[4],
                            node_k1[5], vars.dt[k], kk, kk1,
                        )
                    };
                    let e = 1e-6;
                    let dkk = if dkk_ds != 0.0 {
                        let plus = hs(kappa_k + e, kappa_k1);
                        let minus = hs(kappa_k - e, kappa_k1);
                        std::array::from_fn(|j| (plus[j] - minus[j]) / (2.0 * e))
                    } else {
                        [0.0; 4]
                    };
                    let dkk1 = if dkk1_ds != 0.0 {
                        let plus = hs(kappa_k, kappa_k1 + e);
                        let minus = hs(kappa_k, kappa_k1 - e);
                        std::array::from_fn(|j| (plus[j] - minus[j]) / (2.0 * e))
                    } else {
                        [0.0; 4]
                    };
                    (dkk, dkk1)
                }
            };
            for j in 0..4 {
                let ck = ddef_dkk[j] * dkk_ds;
                if ck.abs() > 1e-15 {
                    builder.add(4 * k + j, k, ck); // s_k column
                }
                let ck1 = ddef_dkk1[j] * dkk1_ds;
                if ck1.abs() > 1e-15 {
                    builder.add(4 * k + j, k + 1, ck1); // s_{k+1} column
                }
            }
        }

        // Periodicity constraints (linear: ±1 entries).
        if opt.config.closed {
            let base = n_defects;
            builder.add(base, n - 1, 1.0); // s[N-1] - L
            builder.add(base + 1, n + n - 1, 1.0); // n[N-1] - n[0]
            builder.add(base + 1, n, -1.0);
            builder.add(base + 2, 2 * n + n - 1, 1.0); // v[N-1] - v[0]
            builder.add(base + 2, 2 * n, -1.0);
            builder.add(base + 3, 3 * n + n - 1, 1.0); // alpha[N-1] - alpha[0]
            builder.add(base + 3, 3 * n, -1.0);
        }

        builder.build()
    }

    /// Exact inequality-constraint Jacobian via autodiff for the grip circle and
    /// analytic ±1 entries for the (constant-width) track boundaries.
    fn autodiff_inequality_jacobian(&self, x: &[f64]) -> CsrMatrix {
        let opt = self.optimizer;
        let n = opt.config.n_nodes;
        let n_vars = x.len();

        let mut builder = CsrBuilder::new(opt.n_ineq_constraints(), n_vars);
        let vars = opt.unpack(x);

        for k in 0..n {
            let row = 3 * k;
            let n_col = n + k;

            // boundary: n_k - wl <= 0  (d/dn = 1); -wr - n_k <= 0  (d/dn = -1)
            builder.add(row, n_col, 1.0);
            builder.add(row + 1, n_col, -1.0);

            // grip circle: derivatives w.r.t. v_k, f_drive_k, curvature_cmd_k
            let v = vars.v[k];
            let fd = vars.f_drive[k];
            let cv = vars.curvature_cmd[k];
            if opt.car.max_grip_force(v) > 0.0 {
                let dv = grip_constraint_generic(
                    opt.car,
                    Dual::variable(v),
                    Dual::constant(fd),
                    Dual::constant(cv),
                )
                .dual;
                let dfd = grip_constraint_generic(
                    opt.car,
                    Dual::constant(v),
                    Dual::variable(fd),
                    Dual::constant(cv),
                )
                .dual;
                let dcv = grip_constraint_generic(
                    opt.car,
                    Dual::constant(v),
                    Dual::constant(fd),
                    Dual::variable(cv),
                )
                .dual;
                if dv.abs() > 1e-15 {
                    builder.add(row + 2, 2 * n + k, dv);
                }
                if dfd.abs() > 1e-15 {
                    builder.add(row + 2, 4 * n + k, dfd);
                }
                if dcv.abs() > 1e-15 {
                    builder.add(row + 2, 5 * n + k, dcv);
                }
            }
        }

        builder.build()
    }
}

/// Evaluate point-mass dynamics without constructing the ODE system struct.
///
/// `state` is `[s, n, v, alpha]`, `control` is `[f_drive, curvature_cmd]`, and
/// `kappa` is the track curvature at the node.
pub(crate) fn point_mass_derivatives(
    car: &CarParams,
    state: &[f64; 4],
    control: &[f64; 2],
    kappa: f64,
) -> [f64; 4] {
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

/// Generic point-mass dynamics, usable with `f64` or [`Dual`] for autodiff.
fn point_mass_derivatives_generic<T: Float>(
    car: &CarParams,
    n: T,
    v: T,
    alpha: T,
    f_drive: T,
    curvature_cmd: T,
    kappa: T,
) -> [T; 4] {
    let f_drag = T::from_f64(0.5 * car.air_density * car.drag_coeff * car.frontal_area) * v * v;
    let f_roll = T::from_f64(car.rolling_resistance_force());
    let v_safe = v.max(T::from_f64(0.1));
    let mass = T::from_f64(car.mass);

    let ds_dt = v_safe * alpha.cos() / (T::one() - n * kappa);
    let dn_dt = v_safe * alpha.sin();
    let dv_dt = (f_drive - f_drag - f_roll) / mass;
    let dalpha_dt = curvature_cmd * v_safe - kappa * ds_dt;

    [ds_dt, dn_dt, dv_dt, dalpha_dt]
}

/// Partial derivative of the point-mass dynamics w.r.t. track curvature `kappa`
/// at a node `[s, n, v, alpha, f_drive, curv]`, via forward-mode autodiff.
fn dynamics_dkappa(car: &CarParams, node: &[f64; 6], kappa: f64) -> [f64; 4] {
    let f = point_mass_derivatives_generic::<Dual>(
        car,
        Dual::constant(node[1]),
        Dual::constant(node[2]),
        Dual::constant(node[3]),
        Dual::constant(node[4]),
        Dual::constant(node[5]),
        Dual::variable(kappa),
    );
    [f[0].dual, f[1].dual, f[2].dual, f[3].dual]
}

/// Generic grip-circle constraint `(f_drive/F_max)² + (m·v²·curv/F_max)² - 1`,
/// matching the `f64` formulation in `inequality_constraints`.
fn grip_constraint_generic<T: Float>(car: &CarParams, v: T, f_drive: T, curvature_cmd: T) -> T {
    let mg = T::from_f64(car.mass * GRAVITY);
    let aero = T::from_f64(0.5 * car.air_density * car.lift_coeff * car.frontal_area);
    let mu = T::from_f64(car.tire_mu);
    let mass = T::from_f64(car.mass);

    let f_max = mu * (mg + aero * v * v);
    let f_lon = f_drive;
    let f_lat = mass * v * v * curvature_cmd;

    (f_lon / f_max) * (f_lon / f_max) + (f_lat / f_max) * (f_lat / f_max) - T::one()
}

/// Evaluate the four trapezoidal defects for interval `k` as [`Dual`] numbers,
/// with local variable `wrt` (0..13) seeded as the differentiation variable.
///
/// Local indices: 0..6 = node-k `[s,n,v,alpha,f_drive,curv]`, 6..12 = node-k+1
/// of the same, 12 = `dt`.
fn defect_with_dual(
    car: &CarParams,
    node_k: &[f64; 6],
    node_k1: &[f64; 6],
    dt: f64,
    kappas: [f64; 2],
    wrt: usize,
) -> [Dual; 4] {
    let mk = |val: f64, idx: usize| {
        if idx == wrt {
            Dual::variable(val)
        } else {
            Dual::constant(val)
        }
    };

    let s_k = mk(node_k[0], 0);
    let n_k = mk(node_k[1], 1);
    let v_k = mk(node_k[2], 2);
    let a_k = mk(node_k[3], 3);
    let fd_k = mk(node_k[4], 4);
    let cv_k = mk(node_k[5], 5);
    let s_k1 = mk(node_k1[0], 6);
    let n_k1 = mk(node_k1[1], 7);
    let v_k1 = mk(node_k1[2], 8);
    let a_k1 = mk(node_k1[3], 9);
    let fd_k1 = mk(node_k1[4], 10);
    let cv_k1 = mk(node_k1[5], 11);
    let dt_d = mk(dt, 12);

    let kk = Dual::constant(kappas[0]);
    let kk1 = Dual::constant(kappas[1]);

    let f_k = point_mass_derivatives_generic::<Dual>(car, n_k, v_k, a_k, fd_k, cv_k, kk);
    let f_k1 = point_mass_derivatives_generic::<Dual>(car, n_k1, v_k1, a_k1, fd_k1, cv_k1, kk1);

    let half = dt_d * Dual::constant(0.5);
    let state_k = [s_k, n_k, v_k, a_k];
    let state_k1 = [s_k1, n_k1, v_k1, a_k1];

    std::array::from_fn(|j| state_k1[j] - state_k[j] - half * (f_k[j] + f_k1[j]))
}

/// Fourth-order Hermite-Simpson (separated) dynamics defect for interval `k`,
/// point-mass model, in `f64`.
///
/// The separated form interpolates a midpoint state (a Hermite cubic through the
/// two nodal states and their derivatives), evaluates the dynamics there, and
/// then applies Simpson's rule to the ODE:
///
/// ```text
///   x_mid   = (x_k + x_{k+1})/2 + (dt/8)·(f_k − f_{k+1})
///   u_mid   = (u_k + u_{k+1})/2
///   f_mid   = f(x_mid, u_mid)
///   defect  = x_{k+1} − x_k − (dt/6)·(f_k + 4·f_mid + f_{k+1})
/// ```
///
/// The midpoint curvature is the linear interpolation `(κ_k + κ_{k+1})/2`,
/// matching how the defect treats curvature as held fixed per node (its
/// `s`-dependence is added back as a chain term in the Jacobian).
#[allow(clippy::too_many_arguments)]
fn hermite_simpson_defect(
    car: &CarParams,
    // Node k
    s_k: f64,
    n_k: f64,
    v_k: f64,
    alpha_k: f64,
    fd_k: f64,
    curv_k: f64,
    // Node k+1
    s_k1: f64,
    n_k1: f64,
    v_k1: f64,
    alpha_k1: f64,
    fd_k1: f64,
    curv_k1: f64,
    // Time step
    dt: f64,
    // Track curvatures at the two nodes
    kappa_k: f64,
    kappa_k1: f64,
) -> [f64; 4] {
    hermite_simpson_defect_generic::<f64>(
        car,
        &[s_k, n_k, v_k, alpha_k],
        &[fd_k, curv_k],
        &[s_k1, n_k1, v_k1, alpha_k1],
        &[fd_k1, curv_k1],
        dt,
        kappa_k,
        kappa_k1,
    )
}

/// [`Dual`] Hermite-Simpson defect for interval `k`, with local variable `wrt`
/// (0..13) seeded as the differentiation variable.
///
/// Local indices: 0..4 = `state_k` `[s,n,v,alpha]`, 4..6 = `control_k`
/// `[f_drive,curv]`, 6..10 = `state_k1`, 10..12 = `control_k1`, 12 = `dt`. The
/// chain rule propagates automatically through the midpoint interpolation and the
/// three dynamics evaluations, so the returned `.dual` parts are exact partials.
#[allow(clippy::too_many_arguments)]
fn hermite_simpson_defect_with_dual(
    car: &CarParams,
    state_k: &[f64; 4],
    control_k: &[f64; 2],
    state_k1: &[f64; 4],
    control_k1: &[f64; 2],
    dt: f64,
    kappa_k: f64,
    kappa_k1: f64,
    wrt: usize,
) -> [Dual; 4] {
    let mk = |val: f64, idx: usize| {
        if idx == wrt {
            Dual::variable(val)
        } else {
            Dual::constant(val)
        }
    };

    let sd_k = [
        mk(state_k[0], 0),
        mk(state_k[1], 1),
        mk(state_k[2], 2),
        mk(state_k[3], 3),
    ];
    let cd_k = [mk(control_k[0], 4), mk(control_k[1], 5)];
    let sd_k1 = [
        mk(state_k1[0], 6),
        mk(state_k1[1], 7),
        mk(state_k1[2], 8),
        mk(state_k1[3], 9),
    ];
    let cd_k1 = [mk(control_k1[0], 10), mk(control_k1[1], 11)];
    let dt_d = mk(dt, 12);

    hermite_simpson_defect_generic::<Dual>(
        car,
        &sd_k,
        &cd_k,
        &sd_k1,
        &cd_k1,
        dt_d,
        Dual::constant(kappa_k),
        Dual::constant(kappa_k1),
    )
}

/// Generic Hermite-Simpson defect, usable with `f64` or [`Dual`]. Curvatures are
/// held fixed (their `s`-dependence is a Jacobian chain term, as for the
/// trapezoidal defect). `s_mid` is interpolated for completeness but the
/// point-mass dynamics do not depend on `s`, so only `n,v,alpha` of the midpoint
/// state feed the midpoint evaluation.
#[allow(clippy::too_many_arguments)]
fn hermite_simpson_defect_generic<T: Float>(
    car: &CarParams,
    state_k: &[T; 4],
    control_k: &[T; 2],
    state_k1: &[T; 4],
    control_k1: &[T; 2],
    dt: T,
    kappa_k: T,
    kappa_k1: T,
) -> [T; 4] {
    // (a,b) nodal dynamics
    let f_k = point_mass_derivatives_generic::<T>(
        car,
        state_k[1],
        state_k[2],
        state_k[3],
        control_k[0],
        control_k[1],
        kappa_k,
    );
    let f_k1 = point_mass_derivatives_generic::<T>(
        car,
        state_k1[1],
        state_k1[2],
        state_k1[3],
        control_k1[0],
        control_k1[1],
        kappa_k1,
    );

    // (c) midpoint state: x_mid = (x_k + x_{k+1})/2 + (dt/8)·(f_k − f_{k+1})
    let eighth = dt * 0.125;
    let x_mid: [T; 4] =
        std::array::from_fn(|j| (state_k[j] + state_k1[j]) * 0.5 + eighth * (f_k[j] - f_k1[j]));

    // (d) midpoint control (linear interpolation)
    let fd_mid = (control_k[0] + control_k1[0]) * 0.5;
    let curv_mid = (control_k[1] + control_k1[1]) * 0.5;
    // (e) midpoint curvature (linear interpolation)
    let kappa_mid = (kappa_k + kappa_k1) * 0.5;

    // (f) midpoint dynamics (point-mass model ignores s, so x_mid[0] is unused)
    let f_mid = point_mass_derivatives_generic::<T>(
        car, x_mid[1], x_mid[2], x_mid[3], fd_mid, curv_mid, kappa_mid,
    );

    // (g) Simpson's-rule defect
    let sixth = dt / 6.0;
    std::array::from_fn(|j| state_k1[j] - state_k[j] - sixth * (f_k[j] + f_mid[j] * 4.0 + f_k1[j]))
}

/// Compute a Jacobian numerically using central finite differences.
/// Retained as a reference/validation tool (used in tests only).
#[cfg(test)]
fn numerical_jacobian_fd(
    x: &[f64],
    n_constraints: usize,
    eval: impl Fn(&[f64]) -> Vec<f64>,
) -> CsrMatrix {
    let eps = 1e-7;
    let n_vars = x.len();
    let mut builder = CsrBuilder::new(n_constraints, n_vars);

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

// ---------------------------------------------------------------------------
// 7-DOF tire-model formulation
//
// A hybrid formulation: it keeps the 4-state curvilinear coordinate frame (so
// the track boundary constraints stay simple) but replaces the point-mass grip
// circle with the Pacejka combined-slip tire model and four-corner, load-
// sensitive vertical loads. Because the wheel-spin and chassis-rotation states
// reach quasi-equilibrium quickly, they are not tracked explicitly; instead the
// realistic tire force *limits* enter the dynamics and the grip constraint.
// ---------------------------------------------------------------------------

/// Generic four-corner load-sensitive grip budget `Σ μ_eff(F_z_i)·F_z_i`.
///
/// Mirrors `CarParams::corner_loads` + `effective_mu` but is `Float`-generic, so
/// it can be autodiffed. All car/tire constants are lifted via `T::from_f64`.
fn available_grip_generic<T: Float>(
    car: &CarParams,
    tire: &PacejkaTire,
    speed: T,
    lateral_accel: T,
    ax: T,
) -> T {
    let m = T::from_f64(car.mass);
    let weight = m * T::from_f64(GRAVITY);
    let df = T::from_f64(0.5 * car.air_density * car.lift_coeff * car.frontal_area) * speed * speed;

    let lf = T::from_f64(car.cog_to_front);
    let lr = T::from_f64(car.cog_to_rear);
    let wb = T::from_f64(car.wheelbase);
    let abf = T::from_f64(car.aero_balance_front);
    let h = T::from_f64(car.cog_height);
    let rsf = T::from_f64(ROLL_STIFFNESS_FRONT);
    let twf = T::from_f64(car.track_width_front);
    let twr = T::from_f64(car.track_width_rear);
    let half = T::from_f64(0.5);

    // longitudinal (axle) loads
    let front_static = weight * lr / wb;
    let rear_static = weight * lf / wb;
    let front_aero = df * abf;
    let rear_aero = df * (T::one() - abf);
    let wt = m * ax * h / wb;
    let front_total = (front_static + front_aero - wt).max(T::zero());
    let rear_total = (rear_static + rear_aero + wt).max(T::zero());

    // lateral transfer split by roll stiffness
    let dfz_front = m * lateral_accel * h * rsf / twf;
    let dfz_rear = m * lateral_accel * h * (T::one() - rsf) / twr;
    let fz_fl = (front_total * half - dfz_front).max(T::zero());
    let fz_fr = (front_total * half + dfz_front).max(T::zero());
    let fz_rl = (rear_total * half - dfz_rear).max(T::zero());
    let fz_rr = (rear_total * half + dfz_rear).max(T::zero());

    let mu_blend = T::from_f64(0.5 * (tire.lateral.mu + tire.longitudinal.mu));
    let fz_nom = T::from_f64(tire.fz_nominal);
    let load_sens = T::from_f64(tire.load_sensitivity);
    let eff =
        |fz: T| (mu_blend * (T::one() - load_sens * (fz - fz_nom) / fz_nom)).max(T::zero()) * fz;
    eff(fz_fl) + eff(fz_fr) + eff(fz_rl) + eff(fz_rr)
}

/// Generic deliverable tire forces with a smooth (C¹) saturation onto the
/// load-sensitive grip budget. Returns `(fx, fy)`.
fn tire_limited_forces_generic<T: Float>(
    car: &CarParams,
    tire: &PacejkaTire,
    speed: T,
    lateral_accel: T,
    longitudinal_force_request: T,
) -> (T, T) {
    let m = T::from_f64(car.mass);
    let ax = longitudinal_force_request / m;
    let available = available_grip_generic(car, tire, speed, lateral_accel, ax);

    let fx_req = longitudinal_force_request;
    let fy_req = m * lateral_accel;
    let r = (fx_req * fx_req + fy_req * fy_req).sqrt();
    if r.real_value() < 1e-9 {
        return (fx_req, fy_req);
    }

    // smooth saturation removes the kink at the grip boundary
    let sharpness = T::from_f64(10.0) / available;
    let r_limited = smooth_min(r, available, sharpness);
    let scale = r_limited / r;
    (fx_req * scale, fy_req * scale)
}

/// Generic curvilinear 7-DOF dynamics: same equations as
/// [`point_mass_derivatives`], with the longitudinal force smoothly capped by
/// the load-sensitive tire grip. `Float`-generic for exact autodiff Jacobians.
fn seven_dof_derivatives_generic<T: Float>(
    car: &CarParams,
    tire: &PacejkaTire,
    state: &[T; 4],
    control: &[T; 2],
    kappa: T,
) -> [T; 4] {
    let n = state[1];
    let v = state[2];
    let alpha = state[3];
    let f_drive = control[0];
    let curvature_cmd = control[1];

    let lateral_accel = v * v * curvature_cmd;
    let (fx_actual, _fy) = tire_limited_forces_generic(car, tire, v, lateral_accel, f_drive);

    let f_drag = T::from_f64(0.5 * car.air_density * car.drag_coeff * car.frontal_area) * v * v;
    let f_roll = T::from_f64(car.rolling_resistance_force());
    let v_safe = v.max(T::from_f64(0.1));
    let m = T::from_f64(car.mass);

    let ds_dt = v_safe * alpha.cos() / (T::one() - n * kappa);
    let dn_dt = v_safe * alpha.sin();
    let dv_dt = (fx_actual - f_drag - f_roll) / m;
    let dalpha_dt = curvature_cmd * v_safe - kappa * ds_dt;

    [ds_dt, dn_dt, dv_dt, dalpha_dt]
}

/// Generic Pacejka grip constraint: `requested / available - 1 ≤ 0`.
fn pacejka_grip_constraint_generic<T: Float>(
    car: &CarParams,
    tire: &PacejkaTire,
    v: T,
    f_drive: T,
    curv: T,
) -> T {
    let m = T::from_f64(car.mass);
    let lateral_accel = v * v * curv;
    let ax = f_drive / m;
    let available = available_grip_generic(car, tire, v, lateral_accel, ax);
    let f_lat = m * v * v * curv;
    let req = (f_drive * f_drive + f_lat * f_lat).sqrt();
    req / available - T::one()
}

/// Compute the longitudinal and lateral force the tires can actually deliver
/// (f64 entry point; uses the smooth saturation).
pub fn tire_limited_forces(
    car: &CarParams,
    tire: &PacejkaTire,
    speed: f64,
    lateral_accel: f64,
    longitudinal_force_request: f64,
) -> (f64, f64) {
    tire_limited_forces_generic::<f64>(car, tire, speed, lateral_accel, longitudinal_force_request)
}

/// Curvilinear 7-DOF dynamics (f64 entry point).
///
/// `state` is `[s, n, v, alpha]`, `control` is `[f_drive, curvature_cmd]`.
pub fn seven_dof_derivatives(
    car: &CarParams,
    tire: &PacejkaTire,
    state: &[f64; 4],
    control: &[f64; 2],
    kappa: f64,
) -> [f64; 4] {
    seven_dof_derivatives_generic::<f64>(car, tire, state, control, kappa)
}

/// Collocation evaluator using the 7-DOF tire and load model for force
/// computation but retaining the curvilinear coordinate frame.
///
/// State at each node: `[s, n, v, alpha]`; control: `[f_drive, curvature_cmd]`
/// (same layout as the point-mass evaluator). The difference is that the
/// dynamics use tire-limited longitudinal force and the grip inequality uses the
/// Pacejka combined-slip / load-sensitive model instead of a simple grip circle.
struct SevenDofEvaluator<'a, 'b> {
    optimizer: &'a CollocationOptimizer<'b>,
    tire: &'a PacejkaTire,
}

impl NlpEvaluator for SevenDofEvaluator<'_, '_> {
    fn objective(&self, x: &[f64]) -> f64 {
        CollocationEvaluator {
            optimizer: self.optimizer,
        }
        .objective(x)
    }

    fn objective_gradient(&self, x: &[f64]) -> Vec<f64> {
        CollocationEvaluator {
            optimizer: self.optimizer,
        }
        .objective_gradient(x)
    }

    fn equality_constraints(&self, x: &[f64]) -> Vec<f64> {
        let opt = self.optimizer;
        let n = opt.config.n_nodes;
        let vars = opt.unpack(x);
        let mut c = Vec::with_capacity(opt.n_eq_constraints());

        for k in 0..n - 1 {
            let kappa_k = opt.track.curvature_at(vars.s[k]);
            let kappa_k1 = opt.track.curvature_at(vars.s[k + 1]);
            let state_k = [vars.s[k], vars.n[k], vars.v[k], vars.alpha[k]];
            let state_k1 = [
                vars.s[k + 1],
                vars.n[k + 1],
                vars.v[k + 1],
                vars.alpha[k + 1],
            ];
            let ctrl_k = [vars.f_drive[k], vars.curvature_cmd[k]];
            let ctrl_k1 = [vars.f_drive[k + 1], vars.curvature_cmd[k + 1]];

            let dk = seven_dof_derivatives(opt.car, self.tire, &state_k, &ctrl_k, kappa_k);
            let dk1 = seven_dof_derivatives(opt.car, self.tire, &state_k1, &ctrl_k1, kappa_k1);

            let half_dt = vars.dt[k] / 2.0;
            for j in 0..4 {
                c.push(state_k1[j] - state_k[j] - half_dt * (dk[j] + dk1[j]));
            }
        }

        if opt.config.closed {
            c.push(vars.s[n - 1] - opt.track.total_length);
            c.push(vars.n[n - 1] - vars.n[0]);
            c.push(vars.v[n - 1] - vars.v[0]);
            c.push(apex_track::normalize_angle(
                vars.alpha[n - 1] - vars.alpha[0],
            ));
        }

        c
    }

    fn inequality_constraints(&self, x: &[f64]) -> Vec<f64> {
        let opt = self.optimizer;
        let n = opt.config.n_nodes;
        let vars = opt.unpack(x);
        let mut c = Vec::with_capacity(opt.n_ineq_constraints());

        for k in 0..n {
            let (wl, wr) = opt.track.width_at(vars.s[k]);
            c.push(vars.n[k] - wl);
            c.push(-wr - vars.n[k]);
            c.push(pacejka_grip_constraint_generic::<f64>(
                opt.car,
                self.tire,
                vars.v[k],
                vars.f_drive[k],
                vars.curvature_cmd[k],
            ));
        }

        c
    }

    fn equality_jacobian(&self, x: &[f64]) -> CsrMatrix {
        self.autodiff_equality_jacobian(x)
    }

    fn inequality_jacobian(&self, x: &[f64]) -> CsrMatrix {
        self.autodiff_inequality_jacobian(x)
    }
}

impl SevenDofEvaluator<'_, '_> {
    /// Exact equality Jacobian for the 7-DOF dynamics via forward-mode autodiff
    /// of the smooth, generic tire dynamics, plus the curvature-chain term in
    /// the s-columns (same banded structure as the point-mass solver).
    fn autodiff_equality_jacobian(&self, x: &[f64]) -> CsrMatrix {
        let opt = self.optimizer;
        let n = opt.config.n_nodes;
        let n_defects = 4 * (n - 1);
        let mut builder = CsrBuilder::new(opt.n_eq_constraints(), x.len());
        let vars = opt.unpack(x);

        for k in 0..n - 1 {
            let kappa_k = opt.track.curvature_at(vars.s[k]);
            let kappa_k1 = opt.track.curvature_at(vars.s[k + 1]);
            let node_k = [
                vars.s[k],
                vars.n[k],
                vars.v[k],
                vars.alpha[k],
                vars.f_drive[k],
                vars.curvature_cmd[k],
            ];
            let node_k1 = [
                vars.s[k + 1],
                vars.n[k + 1],
                vars.v[k + 1],
                vars.alpha[k + 1],
                vars.f_drive[k + 1],
                vars.curvature_cmd[k + 1],
            ];
            let global_indices = [
                k,
                n + k,
                2 * n + k,
                3 * n + k,
                4 * n + k,
                5 * n + k,
                k + 1,
                n + k + 1,
                2 * n + k + 1,
                3 * n + k + 1,
                4 * n + k + 1,
                5 * n + k + 1,
                opt.dt_offset() + k,
            ];

            for (wrt, &col) in global_indices.iter().enumerate() {
                let d = seven_dof_defect_with_dual(
                    opt.car,
                    self.tire,
                    &node_k,
                    &node_k1,
                    vars.dt[k],
                    [kappa_k, kappa_k1],
                    wrt,
                );
                for (j, dj) in d.iter().enumerate() {
                    if dj.dual.abs() > 1e-15 {
                        builder.add(4 * k + j, col, dj.dual);
                    }
                }
            }

            // curvature-chain correction in the s-columns
            let half_dt = vars.dt[k] / 2.0;
            let dfk = seven_dof_dynamics_dkappa(opt.car, self.tire, &node_k, kappa_k);
            let dfk1 = seven_dof_dynamics_dkappa(opt.car, self.tire, &node_k1, kappa_k1);
            let h = 1e-3;
            let dkk = (opt.track.curvature_at(node_k[0] + h)
                - opt.track.curvature_at(node_k[0] - h))
                / (2.0 * h);
            let dkk1 = (opt.track.curvature_at(node_k1[0] + h)
                - opt.track.curvature_at(node_k1[0] - h))
                / (2.0 * h);
            for j in 0..4 {
                let ck = -half_dt * dfk[j] * dkk;
                if ck.abs() > 1e-15 {
                    builder.add(4 * k + j, k, ck);
                }
                let ck1 = -half_dt * dfk1[j] * dkk1;
                if ck1.abs() > 1e-15 {
                    builder.add(4 * k + j, k + 1, ck1);
                }
            }
        }

        if opt.config.closed {
            let base = n_defects;
            builder.add(base, n - 1, 1.0);
            builder.add(base + 1, n + n - 1, 1.0);
            builder.add(base + 1, n, -1.0);
            builder.add(base + 2, 2 * n + n - 1, 1.0);
            builder.add(base + 2, 2 * n, -1.0);
            builder.add(base + 3, 3 * n + n - 1, 1.0);
            builder.add(base + 3, 3 * n, -1.0);
        }

        builder.build()
    }

    /// Exact inequality Jacobian: analytic ±1 for the boundaries and autodiff of
    /// the Pacejka grip constraint w.r.t. (v, f_drive, curvature_cmd).
    fn autodiff_inequality_jacobian(&self, x: &[f64]) -> CsrMatrix {
        let opt = self.optimizer;
        let n = opt.config.n_nodes;
        let mut builder = CsrBuilder::new(opt.n_ineq_constraints(), x.len());
        let vars = opt.unpack(x);

        for k in 0..n {
            let row = 3 * k;
            builder.add(row, n + k, 1.0);
            builder.add(row + 1, n + k, -1.0);

            let v = vars.v[k];
            let fd = vars.f_drive[k];
            let cv = vars.curvature_cmd[k];
            let dv = pacejka_grip_constraint_generic(
                opt.car,
                self.tire,
                Dual::variable(v),
                Dual::constant(fd),
                Dual::constant(cv),
            )
            .dual;
            let dfd = pacejka_grip_constraint_generic(
                opt.car,
                self.tire,
                Dual::constant(v),
                Dual::variable(fd),
                Dual::constant(cv),
            )
            .dual;
            let dcv = pacejka_grip_constraint_generic(
                opt.car,
                self.tire,
                Dual::constant(v),
                Dual::constant(fd),
                Dual::variable(cv),
            )
            .dual;
            if dv.abs() > 1e-15 {
                builder.add(row + 2, 2 * n + k, dv);
            }
            if dfd.abs() > 1e-15 {
                builder.add(row + 2, 4 * n + k, dfd);
            }
            if dcv.abs() > 1e-15 {
                builder.add(row + 2, 5 * n + k, dcv);
            }
        }

        builder.build()
    }
}

/// Partial derivative of the 7-DOF dynamics w.r.t. track curvature `kappa`.
fn seven_dof_dynamics_dkappa(
    car: &CarParams,
    tire: &PacejkaTire,
    node: &[f64; 6],
    kappa: f64,
) -> [f64; 4] {
    let state = [
        Dual::constant(node[0]),
        Dual::constant(node[1]),
        Dual::constant(node[2]),
        Dual::constant(node[3]),
    ];
    let control = [Dual::constant(node[4]), Dual::constant(node[5])];
    let f =
        seven_dof_derivatives_generic::<Dual>(car, tire, &state, &control, Dual::variable(kappa));
    [f[0].dual, f[1].dual, f[2].dual, f[3].dual]
}

/// Evaluate the four 7-DOF trapezoidal defects for interval `k` as duals, with
/// local variable `wrt` (0..13) seeded as the differentiation variable.
fn seven_dof_defect_with_dual(
    car: &CarParams,
    tire: &PacejkaTire,
    node_k: &[f64; 6],
    node_k1: &[f64; 6],
    dt: f64,
    kappas: [f64; 2],
    wrt: usize,
) -> [Dual; 4] {
    let mk = |val: f64, idx: usize| {
        if idx == wrt {
            Dual::variable(val)
        } else {
            Dual::constant(val)
        }
    };

    let state_k = [
        mk(node_k[0], 0),
        mk(node_k[1], 1),
        mk(node_k[2], 2),
        mk(node_k[3], 3),
    ];
    let ctrl_k = [mk(node_k[4], 4), mk(node_k[5], 5)];
    let state_k1 = [
        mk(node_k1[0], 6),
        mk(node_k1[1], 7),
        mk(node_k1[2], 8),
        mk(node_k1[3], 9),
    ];
    let ctrl_k1 = [mk(node_k1[4], 10), mk(node_k1[5], 11)];
    let dt_d = mk(dt, 12);

    let f_k = seven_dof_derivatives_generic::<Dual>(
        car,
        tire,
        &state_k,
        &ctrl_k,
        Dual::constant(kappas[0]),
    );
    let f_k1 = seven_dof_derivatives_generic::<Dual>(
        car,
        tire,
        &state_k1,
        &ctrl_k1,
        Dual::constant(kappas[1]),
    );

    let half = dt_d * Dual::constant(0.5);
    std::array::from_fn(|j| state_k1[j] - state_k[j] - half * (f_k[j] + f_k1[j]))
}

// ---------------------------------------------------------------------------
// 14-DOF force model (two-phase method, Phase A)
//
// Reuses the curvilinear 4-state collocation and the autodiffed 7-DOF dynamics
// for the trajectory defects (the longitudinal balance is unaffected by ride
// height), but replaces the cornering grip *budget* with one that couples
// ride-height-sensitive aero to the suspension load distribution: the four
// corner loads compress the suspension, the resulting ride height sets the
// downforce, and that downforce feeds back into the available tire grip.
// ---------------------------------------------------------------------------

/// Total available grip force (N) from the full 14-DOF quasi-static force model.
///
/// Steps: four-corner loads → suspension static equilibrium → ride heights →
/// ride-height-sensitive aero → aero-adjusted corner loads → load-sensitive
/// effective grip summed over the four corners.
///
/// The front roll-stiffness fraction is the module-wide [`ROLL_STIFFNESS_FRONT`]
/// used throughout the 7-DOF formulation (kept off the signature so the function
/// stays within the clippy argument-count limit without a suppression).
pub fn fourteen_dof_grip_budget(
    car: &CarParams,
    tire: &PacejkaTire,
    suspension: &SuspensionSystem,
    aero: &AeroModel,
    speed: f64,
    lateral_accel: f64,
    longitudinal_accel: f64,
) -> f64 {
    // (a) four-corner vertical loads (already include the simple speed² downforce)
    let mut loads = car.corner_loads(
        speed,
        longitudinal_accel,
        lateral_accel,
        ROLL_STIFFNESS_FRONT,
    );

    // (b) suspension compression that supports those loads
    let z_eq = suspension.static_equilibrium(&loads);

    // (c) ride heights implied by the suspension compression
    let front_rh = aero.design_ride_height - 0.5 * (z_eq[0] + z_eq[1]);
    let rear_rh = aero.design_ride_height - 0.5 * (z_eq[2] + z_eq[3]);

    // (d) ride-height-sensitive aero (quasi-static → zero pitch)
    let aero_f = aero.compute(speed, front_rh, rear_rh, 0.0);

    // (e) swap the simple speed² downforce already baked into `loads` for the
    //     ride-height-sensitive downforce (so it is not double counted)
    let simple_df = car.downforce(speed);
    let simple_front = 0.5 * simple_df * car.aero_balance_front;
    let simple_rear = 0.5 * simple_df * (1.0 - car.aero_balance_front);
    loads[0] += 0.5 * aero_f.downforce_front - simple_front;
    loads[1] += 0.5 * aero_f.downforce_front - simple_front;
    loads[2] += 0.5 * aero_f.downforce_rear - simple_rear;
    loads[3] += 0.5 * aero_f.downforce_rear - simple_rear;

    // (f,g) load-sensitive effective grip summed over the corners
    let base_mu = 0.5 * (tire.lateral.mu + tire.longitudinal.mu);
    loads
        .iter()
        .map(|&fz| {
            let fz = fz.max(0.0);
            tire.effective_mu(base_mu, fz) * fz
        })
        .sum()
}

/// 14-DOF grip inequality at one node: `requested / available − 1 ≤ 0`.
fn fourteen_dof_grip_constraint(
    car: &CarParams,
    tire: &PacejkaTire,
    suspension: &SuspensionSystem,
    aero: &AeroModel,
    v: f64,
    f_drive: f64,
    curv: f64,
) -> f64 {
    let m = car.mass;
    let lateral_accel = v * v * curv;
    let ax = f_drive / m;
    let available =
        fourteen_dof_grip_budget(car, tire, suspension, aero, v, lateral_accel, ax).max(1.0);
    let f_lat = m * v * v * curv;
    let req = (f_drive * f_drive + f_lat * f_lat).sqrt();
    req / available - 1.0
}

/// Collocation evaluator using the 14-DOF ride-height-coupled force model for
/// the grip budget. The trajectory defects, objective, and equality Jacobian are
/// the autodiffed 7-DOF ones; only the grip inequality (and its Jacobian) use the
/// 14-DOF budget, computed by finite differences over its three local variables.
struct FourteenDofEvaluator<'a, 'b> {
    optimizer: &'a CollocationOptimizer<'b>,
    tire: &'a PacejkaTire,
    suspension: &'a SuspensionSystem,
    aero: &'a AeroModel,
}

impl NlpEvaluator for FourteenDofEvaluator<'_, '_> {
    fn objective(&self, x: &[f64]) -> f64 {
        SevenDofEvaluator {
            optimizer: self.optimizer,
            tire: self.tire,
        }
        .objective(x)
    }

    fn objective_gradient(&self, x: &[f64]) -> Vec<f64> {
        SevenDofEvaluator {
            optimizer: self.optimizer,
            tire: self.tire,
        }
        .objective_gradient(x)
    }

    fn equality_constraints(&self, x: &[f64]) -> Vec<f64> {
        SevenDofEvaluator {
            optimizer: self.optimizer,
            tire: self.tire,
        }
        .equality_constraints(x)
    }

    fn equality_jacobian(&self, x: &[f64]) -> CsrMatrix {
        SevenDofEvaluator {
            optimizer: self.optimizer,
            tire: self.tire,
        }
        .equality_jacobian(x)
    }

    fn inequality_constraints(&self, x: &[f64]) -> Vec<f64> {
        let opt = self.optimizer;
        let n = opt.config.n_nodes;
        let vars = opt.unpack(x);
        let mut c = Vec::with_capacity(opt.n_ineq_constraints());

        for k in 0..n {
            let (wl, wr) = opt.track.width_at(vars.s[k]);
            c.push(vars.n[k] - wl);
            c.push(-wr - vars.n[k]);
            c.push(fourteen_dof_grip_constraint(
                opt.car,
                self.tire,
                self.suspension,
                self.aero,
                vars.v[k],
                vars.f_drive[k],
                vars.curvature_cmd[k],
            ));
        }

        c
    }

    fn inequality_jacobian(&self, x: &[f64]) -> CsrMatrix {
        let opt = self.optimizer;
        let n = opt.config.n_nodes;
        let mut builder = CsrBuilder::new(opt.n_ineq_constraints(), x.len());
        let vars = opt.unpack(x);

        let grip = |v: f64, fd: f64, cv: f64| {
            fourteen_dof_grip_constraint(opt.car, self.tire, self.suspension, self.aero, v, fd, cv)
        };

        for k in 0..n {
            let row = 3 * k;
            builder.add(row, n + k, 1.0);
            builder.add(row + 1, n + k, -1.0);

            // Grip depends only on (v_k, f_drive_k, curv_k) — central FD over each.
            let v = vars.v[k];
            let fd = vars.f_drive[k];
            let cv = vars.curvature_cmd[k];
            let rel = 1e-6;
            let hv = (v.abs() * rel).max(1e-6);
            let hf = (fd.abs() * rel).max(1e-3);
            let hc = (cv.abs() * rel).max(1e-7);
            let dv = (grip(v + hv, fd, cv) - grip(v - hv, fd, cv)) / (2.0 * hv);
            let dfd = (grip(v, fd + hf, cv) - grip(v, fd - hf, cv)) / (2.0 * hf);
            let dcv = (grip(v, fd, cv + hc) - grip(v, fd, cv - hc)) / (2.0 * hc);
            if dv.abs() > 1e-15 {
                builder.add(row + 2, 2 * n + k, dv);
            }
            if dfd.abs() > 1e-15 {
                builder.add(row + 2, 4 * n + k, dfd);
            }
            if dcv.abs() > 1e-15 {
                builder.add(row + 2, 5 * n + k, dcv);
            }
        }

        builder.build()
    }
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
        assert!(
            max_defect < 100.0,
            "defect {} unexpectedly large",
            max_defect
        );
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

    // --- auto-diff Jacobian tests ---

    /// First (row, col) where two matrices' dense forms differ by more than `tol`.
    fn first_diff(a: &CsrMatrix, b: &CsrMatrix, tol: f64) -> Option<(usize, usize, f64, f64)> {
        let da = a.to_dense();
        let db = b.to_dense();
        for (i, (ra, rb)) in da.iter().zip(db.iter()).enumerate() {
            for (j, (&va, &vb)) in ra.iter().zip(rb.iter()).enumerate() {
                if (va - vb).abs() > tol {
                    return Some((i, j, va, vb));
                }
            }
        }
        None
    }

    #[test]
    fn autodiff_equality_matches_numerical() {
        let (track, car, config) = circle(20);
        let opt = CollocationOptimizer::new(config, &track, &car);
        let x = opt.initial_guess();
        let evaluator = CollocationEvaluator { optimizer: &opt };

        let ad = evaluator.autodiff_equality_jacobian(&x);
        let fd = numerical_jacobian_fd(&x, opt.n_eq_constraints(), |x| {
            evaluator.equality_constraints(x)
        });

        assert_eq!(ad.nrows(), fd.nrows());
        assert_eq!(ad.ncols(), fd.ncols());
        if let Some((i, j, a, b)) = first_diff(&ad, &fd, 1e-4) {
            panic!(
                "eq jacobian mismatch at ({}, {}): autodiff {} vs fd {}",
                i, j, a, b
            );
        }
    }

    #[test]
    fn autodiff_inequality_matches_numerical() {
        let (track, car, config) = circle(20);
        let opt = CollocationOptimizer::new(config, &track, &car);
        let x = opt.initial_guess();
        let evaluator = CollocationEvaluator { optimizer: &opt };

        let ad = evaluator.autodiff_inequality_jacobian(&x);
        let fd = numerical_jacobian_fd(&x, opt.n_ineq_constraints(), |x| {
            evaluator.inequality_constraints(x)
        });

        assert_eq!(ad.nrows(), fd.nrows());
        assert_eq!(ad.ncols(), fd.ncols());
        if let Some((i, j, a, b)) = first_diff(&ad, &fd, 1e-4) {
            panic!(
                "ineq jacobian mismatch at ({}, {}): autodiff {} vs fd {}",
                i, j, a, b
            );
        }
    }

    #[test]
    fn autodiff_jacobian_valid_large() {
        let (track, car, config) = circle(50);
        let opt = CollocationOptimizer::new(config, &track, &car);
        let x = opt.initial_guess();
        let evaluator = CollocationEvaluator { optimizer: &opt };

        let jac = evaluator.autodiff_equality_jacobian(&x);
        assert_eq!(jac.nrows(), opt.n_eq_constraints());
        assert_eq!(jac.ncols(), x.len());
        assert!(jac.nnz() > 0, "jacobian has no entries");
        for row in jac.to_dense() {
            for v in row {
                assert!(v.is_finite(), "non-finite jacobian entry");
            }
        }
    }

    #[test]
    fn optimizer_works_with_autodiff() {
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

        assert!(result.converged, "should converge on the circle");
        assert!(
            (result.lap_time - qss_lap).abs() / qss_lap < 0.01,
            "lap time {} vs QSS {}",
            result.lap_time,
            qss_lap
        );
    }

    #[test]
    fn gn_beats_al_on_oval() {
        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 400);
        let track = build_track("oval", &pts, closed);
        let car = CarParams::default();
        let config = CollocationConfig {
            n_nodes: 50,
            closed: true,
            ..CollocationConfig::default()
        };
        let opt = CollocationOptimizer::new(config, &track, &car);
        let x0 = opt.initial_guess();
        let problem = opt.build_nlp_problem();
        let evaluator = CollocationEvaluator { optimizer: &opt };

        // Augmented Lagrangian
        let al_cfg = SolverConfig {
            max_outer_iter: 15,
            max_inner_iter: 30,
            constraint_tol: 1e-3,
            ..SolverConfig::default()
        };
        let al = solve_nlp(&problem, &evaluator, &x0, &al_cfg);

        // Gauss-Newton
        let gn_cfg = crate::gauss_newton::GaussNewtonConfig {
            max_iterations: 50,
            constraint_tol: 1e-3,
            ..crate::gauss_newton::GaussNewtonConfig::default()
        };
        let gn = crate::gauss_newton::solve_gauss_newton(&problem, &evaluator, &x0, &gn_cfg);

        assert!(
            gn.eq_violation < al.eq_violation,
            "GN eq_viol {:.3e} should beat AL eq_viol {:.3e}",
            gn.eq_violation,
            al.eq_violation
        );
    }

    // --- 7-DOF tire-model tests ---

    #[test]
    fn tire_limited_forces_straight_and_cornering() {
        let car = CarParams::default();
        let tire = apex_physics::PacejkaTire::f1_default();

        // straight line, modest request -> passes through ~unchanged (the smooth
        // saturation clips a fraction of a percent early, by design)
        let (fx0, fy0) = tire_limited_forces(&car, &tire, 50.0, 0.0, 5000.0);
        assert!((fx0 - 5000.0).abs() / 5000.0 < 0.01, "fx {}", fx0);
        assert!(fy0.abs() < 1e-9, "fy {}", fy0);

        // high lateral acceleration saturates the friction budget, so the
        // deliverable longitudinal force is reduced below the request
        let (fx_corner, _fy) = tire_limited_forces(&car, &tire, 50.0, 50.0, 5000.0);
        assert!(
            fx_corner < 5000.0,
            "fx under cornering {} should be reduced",
            fx_corner
        );
    }

    #[test]
    fn seven_dof_matches_point_mass_on_straight() {
        let car = CarParams::default();
        let tire = apex_physics::PacejkaTire::f1_default();
        let state = [0.0, 0.0, 50.0, 0.0];
        let control = [5000.0, 0.0];

        let sd = seven_dof_derivatives(&car, &tire, &state, &control, 0.0);
        let pm = point_mass_derivatives(&car, &state, &control, 0.0);
        // On a straight the tire limit barely binds; the smooth saturation
        // leaves a fraction-of-a-percent difference in the speed derivative.
        for j in 0..4 {
            assert!(
                (sd[j] - pm[j]).abs() < 0.01,
                "component {}: {} vs {}",
                j,
                sd[j],
                pm[j]
            );
        }
    }

    #[test]
    fn seven_dof_circle_converges() {
        let (track, car, config) = circle(30);
        let tire = apex_physics::PacejkaTire::f1_default();
        let opt = CollocationOptimizer::new(config, &track, &car);
        let gn = crate::gauss_newton::GaussNewtonConfig {
            max_iterations: 40,
            constraint_tol: 1e-3,
            ..crate::gauss_newton::GaussNewtonConfig::default()
        };
        let result = opt.optimize_seven_dof(&tire, &gn);
        // The smooth tire model gives exact (autodiff) Jacobians, but the GN
        // solver still cannot fully repair the QSS warm start — that start uses
        // point-mass grip, which overestimates the load-sensitive 7-DOF grip, so
        // the seed speed is mildly infeasible for the tire model. The result is
        // a sensible near-feasible trajectory rather than a tight solve.
        assert!(result.lap_time.is_finite(), "lap time finite");
        assert!(result.speeds.iter().all(|&v| v > 0.0), "speeds positive");
        assert!(
            result.eq_violation < 1.0,
            "eq_viol {} should be small",
            result.eq_violation
        );
    }

    #[test]
    fn seven_dof_load_sensitivity_changes_lap() {
        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 400);
        let track = build_track("oval", &pts, closed);
        let car = CarParams::default();
        let tire = apex_physics::PacejkaTire::f1_default();
        let config = CollocationConfig {
            n_nodes: 40,
            closed: true,
            ..CollocationConfig::default()
        };
        let opt = CollocationOptimizer::new(config, &track, &car);
        let gn = crate::gauss_newton::GaussNewtonConfig {
            max_iterations: 40,
            constraint_tol: 1e-3,
            ..crate::gauss_newton::GaussNewtonConfig::default()
        };

        let pm = opt.optimize_gn(&gn);
        let sd = opt.optimize_seven_dof(&tire, &gn);

        // load-sensitive tires change the achievable trajectory: the lap times
        // should differ (the tire model is doing something)
        assert!(
            (sd.lap_time - pm.lap_time).abs() / pm.lap_time > 1e-3,
            "7-DOF lap {} should differ from point-mass {}",
            sd.lap_time,
            pm.lap_time
        );
        assert!(sd.lap_time.is_finite() && pm.lap_time.is_finite());
    }

    #[test]
    fn tire_warm_start_reduces_grip_violation() {
        let (track, car, config) = circle(30);
        let tire = apex_physics::PacejkaTire::f1_default();
        let opt = CollocationOptimizer::new(config, &track, &car);
        let evaluator = SevenDofEvaluator {
            optimizer: &opt,
            tire: &tire,
        };

        let g_grip = opt.initial_guess();
        let g_tire = opt.initial_guess_seven_dof(&tire);

        let ineq_grip = evaluator.inequality_constraints(&g_grip);
        let ineq_tire = evaluator.inequality_constraints(&g_tire);

        // the grip constraint is every third inequality (index 3k+2)
        let n = opt.config.n_nodes;
        let max_grip_viol = |c: &[f64]| {
            (0..n)
                .map(|k| c[3 * k + 2].max(0.0))
                .fold(0.0_f64, f64::max)
        };

        let grip_circle = max_grip_viol(&ineq_grip);
        let tire_aware = max_grip_viol(&ineq_tire);
        assert!(
            tire_aware <= grip_circle,
            "tire-aware warm start grip violation {} should not exceed grip-circle {}",
            tire_aware,
            grip_circle
        );
    }

    // --- 14-DOF force-model tests ---

    #[test]
    fn fourteen_dof_grip_budget_behaves() {
        let car = CarParams::default();
        let tire = PacejkaTire::f1_default();
        let susp = SuspensionSystem::f1_default();
        let aero = AeroModel::f1_default();
        let speed = 40.0;

        let g0 = fourteen_dof_grip_budget(&car, &tire, &susp, &aero, speed, 0.0, 0.0);
        let seven = available_grip_generic::<f64>(&car, &tire, speed, 0.0, 0.0);

        // positive and finite
        assert!(
            g0.is_finite() && g0 > 0.0,
            "budget {} must be positive finite",
            g0
        );

        // comparable to the 7-DOF value: ride-height aero shifts the downforce
        // (the equilibrium ride height sits below design, trimming downforce a
        // little) but does not change the budget wildly
        assert!(
            g0 > 0.5 * seven && g0 < 1.2 * seven,
            "14-DOF budget {} should be in band of 7-DOF {}",
            g0,
            seven
        );

        // high lateral acceleration transfers load; load sensitivity then cuts
        // the total available grip
        let g_corner = fourteen_dof_grip_budget(&car, &tire, &susp, &aero, speed, 25.0, 0.0);
        assert!(
            g_corner < g0,
            "cornering grip {} should be below straight-line {}",
            g_corner,
            g0
        );
        assert!(g_corner.is_finite() && g_corner > 0.0);
    }

    #[test]
    fn optimize_fourteen_dof_circle() {
        let (track, car, config) = circle(30);
        let tire = PacejkaTire::f1_default();
        let susp = SuspensionSystem::f1_default();
        let aero = AeroModel::f1_default();
        let opt = CollocationOptimizer::new(config, &track, &car);
        let gn = crate::gauss_newton::GaussNewtonConfig {
            max_iterations: 40,
            constraint_tol: 1e-3,
            ..crate::gauss_newton::GaussNewtonConfig::default()
        };

        let pm = opt.optimize_gn(&gn);
        let sd = opt.optimize_seven_dof(&tire, &gn);
        let fd = opt.optimize_fourteen_dof(&tire, &susp, &aero, &gn);

        assert!(
            fd.lap_time.is_finite() && fd.lap_time > 0.0,
            "14-DOF lap {} finite",
            fd.lap_time
        );
        assert!(fd.speeds.iter().all(|&v| v > 0.0), "14-DOF speeds positive");

        // the ride-height-coupled grip budget yields a different optimum than the
        // simple grip circle and the 7-DOF model
        assert!(
            (fd.lap_time - pm.lap_time).abs() / pm.lap_time > 1e-3,
            "14-DOF lap {} should differ from point-mass {}",
            fd.lap_time,
            pm.lap_time
        );
        assert!(
            (fd.lap_time - sd.lap_time).abs() / sd.lap_time > 1e-4,
            "14-DOF lap {} should differ from 7-DOF {}",
            fd.lap_time,
            sd.lap_time
        );
    }

    #[test]
    fn fourteen_dof_full_pipeline() {
        // A small-radius circle keeps the grip-limited speed (and thus the
        // lateral g) modest, so the forward-sim controller can track it.
        let (pts, closed) = circle_track(30.0, 8.0, 200);
        let track = build_track("small_circle", &pts, closed);
        let car = CarParams::default();
        let config = CollocationConfig {
            n_nodes: 20,
            closed: true,
            ..CollocationConfig::default()
        };
        let tire = PacejkaTire::f1_default();
        let susp = SuspensionSystem::f1_default();
        let aero = AeroModel::f1_default();
        let opt = CollocationOptimizer::new(config, &track, &car);
        let gn = crate::gauss_newton::GaussNewtonConfig {
            max_iterations: 30,
            constraint_tol: 1e-3,
            ..crate::gauss_newton::GaussNewtonConfig::default()
        };

        let (result, tele) = opt.optimize_fourteen_dof_full(&tire, &susp, &aero, &gn);

        assert!(
            result.lap_time.is_finite() && result.lap_time > 0.0,
            "opt lap finite"
        );
        assert!(!tele.time.is_empty(), "telemetry produced");
        assert!(
            tele.lap_time.is_finite() && tele.lap_time > 0.0,
            "telemetry lap finite"
        );

        // the forward-simulated lap should track the optimized lap reasonably
        // (the controller is not a perfect tracker)
        assert!(
            (tele.lap_time - result.lap_time).abs() / result.lap_time < 0.20,
            "forward-sim lap {} should be within 20% of optimized {}",
            tele.lap_time,
            result.lap_time
        );
    }

    // --- Hermite-Simpson collocation tests ---

    fn vec_norm(a: &[f64; 4]) -> f64 {
        a.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Trapezoidal point-mass defect for one interval (for accuracy comparisons).
    #[allow(clippy::too_many_arguments)]
    fn trapezoidal_defect(
        car: &CarParams,
        state_k: &[f64; 4],
        control_k: &[f64; 2],
        state_k1: &[f64; 4],
        control_k1: &[f64; 2],
        dt: f64,
        kappa_k: f64,
        kappa_k1: f64,
    ) -> [f64; 4] {
        let fk = point_mass_derivatives(car, state_k, control_k, kappa_k);
        let fk1 = point_mass_derivatives(car, state_k1, control_k1, kappa_k1);
        std::array::from_fn(|j| state_k1[j] - state_k[j] - 0.5 * dt * (fk[j] + fk1[j]))
    }

    /// High-accuracy RK4 reference endpoint for the point-mass ODE under a
    /// constant control and curvature over `dt` (used as ground truth so that a
    /// scheme's defect measures only its quadrature error).
    fn rk4_reference(
        car: &CarParams,
        state0: &[f64; 4],
        control: &[f64; 2],
        kappa: f64,
        dt: f64,
        nsub: usize,
    ) -> [f64; 4] {
        let h = dt / nsub as f64;
        let mut x = *state0;
        for _ in 0..nsub {
            let k1 = point_mass_derivatives(car, &x, control, kappa);
            let x2: [f64; 4] = std::array::from_fn(|j| x[j] + 0.5 * h * k1[j]);
            let k2 = point_mass_derivatives(car, &x2, control, kappa);
            let x3: [f64; 4] = std::array::from_fn(|j| x[j] + 0.5 * h * k2[j]);
            let k3 = point_mass_derivatives(car, &x3, control, kappa);
            let x4: [f64; 4] = std::array::from_fn(|j| x[j] + h * k3[j]);
            let k4 = point_mass_derivatives(car, &x4, control, kappa);
            for j in 0..4 {
                x[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
            }
        }
        x
    }

    #[test]
    fn hs_defect_near_zero_on_steady_circle() {
        // Constant speed on a constant-curvature circle is a steady state: the
        // dynamics are constant, so the trajectory is exactly representable and
        // both schemes' defects vanish.
        let car = CarParams::default();
        let r = 100.0;
        let kappa = 1.0 / r;
        let v = 50.0;
        let f_hold = car.drag_force(v) + car.rolling_resistance_force();
        let dt = 0.1;
        let s_k = 10.0;
        let s_k1 = s_k + v * dt;
        let state_k = [s_k, 0.0, v, 0.0];
        let state_k1 = [s_k1, 0.0, v, 0.0];
        let ctrl = [f_hold, kappa];

        let hs = hermite_simpson_defect(
            &car, s_k, 0.0, v, 0.0, f_hold, kappa, s_k1, 0.0, v, 0.0, f_hold, kappa, dt, kappa,
            kappa,
        );
        let trap = trapezoidal_defect(&car, &state_k, &ctrl, &state_k1, &ctrl, dt, kappa, kappa);

        assert!(vec_norm(&hs) < 1e-9, "HS defect {hs:?} should be ~0");
        assert!(vec_norm(&trap) < 1e-9, "trap defect {trap:?} should be ~0");
        // HS is never worse than trapezoidal.
        assert!(vec_norm(&hs) <= vec_norm(&trap) + 1e-12);
    }

    #[test]
    fn hs_more_accurate_than_trapezoidal_accelerating() {
        // A car accelerating along a straight: the speed derivative is nonlinear
        // (drag ∝ v²), so the schemes differ. Compared against the exact RK4
        // endpoint, the HS defect (4th order) is far smaller than trapezoidal.
        let car = CarParams::default();
        let v0 = 50.0;
        let f_drive = 9000.0;
        let kappa = 0.0;
        let ctrl = [f_drive, 0.0];
        let dt = 1.8; // ~100 m of travel

        let state0 = [0.0, 0.0, v0, 0.0];
        let state1 = rk4_reference(&car, &state0, &ctrl, kappa, dt, 4000);
        // sanity: it accelerated to roughly 60 m/s
        assert!(state1[2] > 58.0 && state1[2] < 72.0, "v1 = {}", state1[2]);

        let hs = hermite_simpson_defect(
            &car, state0[0], 0.0, v0, 0.0, f_drive, 0.0, state1[0], state1[1], state1[2],
            state1[3], f_drive, 0.0, dt, kappa, kappa,
        );
        let trap = trapezoidal_defect(&car, &state0, &ctrl, &state1, &ctrl, dt, kappa, kappa);

        let hs_n = vec_norm(&hs);
        let trap_n = vec_norm(&trap);
        assert!(
            hs_n < trap_n,
            "HS defect {hs_n} should be smaller than trapezoidal {trap_n}"
        );
        assert!(
            hs_n < 0.2 * trap_n,
            "HS defect {hs_n} should be much smaller than trapezoidal {trap_n}"
        );
    }

    #[test]
    fn hs_autodiff_jacobian_matches_numerical_circle() {
        let (pts, closed) = circle_track(100.0, 12.0, 200);
        let track = build_track("circle", &pts, closed);
        let car = CarParams::default();
        let config = CollocationConfig {
            n_nodes: 20,
            method: CollocationMethod::HermiteSimpson,
            ..CollocationConfig::default()
        };
        let opt = CollocationOptimizer::new(config, &track, &car);
        let x = opt.initial_guess();
        let evaluator = CollocationEvaluator { optimizer: &opt };

        let ad = evaluator.autodiff_equality_jacobian(&x);
        let fd = numerical_jacobian_fd(&x, opt.n_eq_constraints(), |x| {
            evaluator.equality_constraints(x)
        });
        assert_eq!(ad.nrows(), fd.nrows());
        assert_eq!(ad.ncols(), fd.ncols());
        if let Some((i, j, a, b)) = first_diff(&ad, &fd, 1e-4) {
            panic!("HS eq jacobian mismatch at ({i}, {j}): autodiff {a} vs fd {b}");
        }
    }

    #[test]
    fn hs_autodiff_jacobian_matches_numerical_oval() {
        // The oval has curvature transitions, so this exercises the s-column
        // curvature-chain correction (dκ/ds ≠ 0 at corner entry/exit).
        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 400);
        let track = build_track("oval", &pts, closed);
        let car = CarParams::default();
        let config = CollocationConfig {
            n_nodes: 40,
            method: CollocationMethod::HermiteSimpson,
            ..CollocationConfig::default()
        };
        let opt = CollocationOptimizer::new(config, &track, &car);
        let x = opt.initial_guess();
        let evaluator = CollocationEvaluator { optimizer: &opt };

        let ad = evaluator.autodiff_equality_jacobian(&x);
        let fd = numerical_jacobian_fd(&x, opt.n_eq_constraints(), |x| {
            evaluator.equality_constraints(x)
        });
        if let Some((i, j, a, b)) = first_diff(&ad, &fd, 1e-4) {
            panic!("HS oval eq jacobian mismatch at ({i}, {j}): autodiff {a} vs fd {b}");
        }
    }

    #[test]
    fn hs_optimization_circle_converges() {
        let (pts, closed) = circle_track(100.0, 12.0, 200);
        let track = build_track("circle", &pts, closed);
        let car = CarParams::default();
        let gn = crate::gauss_newton::GaussNewtonConfig {
            max_iterations: 40,
            constraint_tol: 1e-3,
            ..crate::gauss_newton::GaussNewtonConfig::default()
        };

        let hs = CollocationOptimizer::new(
            CollocationConfig {
                n_nodes: 30,
                method: CollocationMethod::HermiteSimpson,
                ..CollocationConfig::default()
            },
            &track,
            &car,
        )
        .optimize_gn(&gn);
        let tr = CollocationOptimizer::new(
            CollocationConfig {
                n_nodes: 30,
                method: CollocationMethod::Trapezoidal,
                ..CollocationConfig::default()
            },
            &track,
            &car,
        )
        .optimize_gn(&gn);

        assert!(
            hs.eq_violation < 1e-3,
            "HS eq_viol {} should converge",
            hs.eq_violation
        );
        assert!(
            (hs.lap_time - tr.lap_time).abs() / tr.lap_time < 0.02,
            "HS lap {} should match trapezoidal {} within 2%",
            hs.lap_time,
            tr.lap_time
        );
    }

    #[test]
    fn hs_lower_defect_than_trapezoidal_on_oval() {
        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 400);
        let track = build_track("oval", &pts, closed);
        let car = CarParams::default();
        let gn = crate::gauss_newton::GaussNewtonConfig {
            max_iterations: 40,
            constraint_tol: 1e-4,
            ..crate::gauss_newton::GaussNewtonConfig::default()
        };

        let hs = CollocationOptimizer::new(
            CollocationConfig {
                n_nodes: 50,
                method: CollocationMethod::HermiteSimpson,
                ..CollocationConfig::default()
            },
            &track,
            &car,
        )
        .optimize_gn(&gn);
        let tr = CollocationOptimizer::new(
            CollocationConfig {
                n_nodes: 50,
                method: CollocationMethod::Trapezoidal,
                ..CollocationConfig::default()
            },
            &track,
            &car,
        )
        .optimize_gn(&gn);

        assert!(hs.lap_time.is_finite(), "HS lap finite");
        assert!(hs.speeds.iter().all(|&v| v > 0.0), "HS speeds positive");
        // The key claim: at equal node count the higher-order scheme reaches a
        // more dynamically consistent solution (lower equality violation).
        assert!(
            hs.eq_violation < tr.eq_violation,
            "HS eq_viol {} should be below trapezoidal {}",
            hs.eq_violation,
            tr.eq_violation
        );
    }

    #[test]
    fn hs_high_order_convergence() {
        // Local defect of HS on an accelerating straight, against the exact RK4
        // endpoint, as the step is halved. A 4th-order-or-better method cuts the
        // defect by ≥ ~16× per halving — far steeper than trapezoidal's ~4×.
        let car = CarParams::default();
        let v0 = 50.0;
        let f_drive = 8000.0;
        let kappa = 0.0;
        let ctrl = [f_drive, 0.0];

        let hs_defect_norm = |dt: f64| {
            let state0 = [0.0, 0.0, v0, 0.0];
            let state1 = rk4_reference(&car, &state0, &ctrl, kappa, dt, 8000);
            let d = hermite_simpson_defect(
                &car, 0.0, 0.0, v0, 0.0, f_drive, 0.0, state1[0], state1[1], state1[2], state1[3],
                f_drive, 0.0, dt, kappa, kappa,
            );
            vec_norm(&d)
        };
        let trap_defect_norm = |dt: f64| {
            let state0 = [0.0, 0.0, v0, 0.0];
            let state1 = rk4_reference(&car, &state0, &ctrl, kappa, dt, 8000);
            trapezoidal_defect(&car, &state0, &ctrl, &state1, &ctrl, dt, kappa, kappa)
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
        };

        let d1 = hs_defect_norm(0.4);
        let d2 = hs_defect_norm(0.2);
        let d3 = hs_defect_norm(0.1);
        // monotonic decrease
        assert!(d1 > d2 && d2 > d3, "HS defect must shrink: {d1} {d2} {d3}");
        let r1 = d1 / d2;
        let r2 = d2 / d3;
        assert!(r1 > 12.0, "HS ratio1 {r1} should be ≥ ~16 (high order)");
        assert!(r2 > 12.0, "HS ratio2 {r2} should be ≥ ~16 (high order)");
        // and clearly higher order than trapezoidal (whose ratio is ~4)
        let tr_ratio = trap_defect_norm(0.4) / trap_defect_norm(0.2);
        assert!(
            r2 > 2.0 * tr_ratio,
            "HS ratio {r2} should far exceed trapezoidal ratio {tr_ratio}"
        );
    }

    #[test]
    fn collocation_method_enum_works() {
        let (pts, closed) = circle_track(100.0, 12.0, 200);
        let track = build_track("circle", &pts, closed);
        let car = CarParams::default();
        let gn = crate::gauss_newton::GaussNewtonConfig {
            max_iterations: 40,
            constraint_tol: 1e-3,
            ..crate::gauss_newton::GaussNewtonConfig::default()
        };

        let tr = CollocationOptimizer::new(
            CollocationConfig {
                n_nodes: 30,
                method: CollocationMethod::Trapezoidal,
                ..CollocationConfig::default()
            },
            &track,
            &car,
        )
        .optimize_gn(&gn);
        let hs = CollocationOptimizer::new(
            CollocationConfig {
                n_nodes: 30,
                method: CollocationMethod::HermiteSimpson,
                ..CollocationConfig::default()
            },
            &track,
            &car,
        )
        .optimize_gn(&gn);

        assert!(
            tr.lap_time.is_finite() && tr.lap_time > 0.0,
            "trap lap valid"
        );
        assert!(hs.lap_time.is_finite() && hs.lap_time > 0.0, "HS lap valid");
        // Both valid and close on the easy circle (different schemes, near-equal
        // optima).
        assert!(
            (hs.lap_time - tr.lap_time).abs() / tr.lap_time < 0.02,
            "HS lap {} and trapezoidal lap {} should be close",
            hs.lap_time,
            tr.lap_time
        );
    }

    // --- brake-bias optimization tests ---

    fn gn_cfg() -> crate::gauss_newton::GaussNewtonConfig {
        crate::gauss_newton::GaussNewtonConfig {
            max_iterations: 40,
            constraint_tol: 1e-3,
            ..crate::gauss_newton::GaussNewtonConfig::default()
        }
    }

    #[test]
    fn brake_bias_off_by_default() {
        let (track, car, config) = circle(30);
        assert!(!config.optimize_brake_bias, "default should be off");
        let opt = CollocationOptimizer::new(config, &track, &car);
        let result = opt.optimize_gn(&gn_cfg());
        assert!(result.brake_bias.is_none(), "brake_bias should be None when off");
    }

    #[test]
    fn brake_bias_variable_count() {
        let (track, car, mut config) = circle(30);

        // Off: 7N - 1 = 209 variables.
        config.optimize_brake_bias = false;
        let off = CollocationOptimizer::new(config.clone(), &track, &car);
        assert_eq!(off.n_vars(), 7 * 30 - 1);
        assert_eq!(off.warm_start().len(), 209);

        // On: 8N - 1 = 239 variables.
        config.optimize_brake_bias = true;
        let on = CollocationOptimizer::new(config, &track, &car);
        assert_eq!(on.n_vars(), 8 * 30 - 1);
        assert_eq!(on.warm_start().len(), 239);
    }

    #[test]
    fn brake_bias_on_circle() {
        let (track, car, config) = circle(30);
        let opt = CollocationOptimizer::new(config, &track, &car);

        let baseline = opt.optimize_gn(&gn_cfg());
        let with_bias = opt.optimize_with_brake_bias(&gn_cfg());

        let bias = with_bias
            .brake_bias
            .as_ref()
            .expect("brake_bias should be Some");
        assert_eq!(bias.len(), 30, "one bias per node");
        for &b in bias {
            assert!((0.50..=0.80).contains(&b), "brake bias {b} out of [0.50, 0.80]");
        }

        // Brake bias doesn't couple to the point-mass dynamics, so the lap time
        // should be essentially unchanged.
        assert!(with_bias.lap_time.is_finite());
        assert!(
            (with_bias.lap_time - baseline.lap_time).abs() / baseline.lap_time < 0.02,
            "lap time {} should match baseline {}",
            with_bias.lap_time,
            baseline.lap_time
        );
    }

    #[test]
    fn brake_bias_on_oval() {
        let (pts, closed) = oval_track(1000.0, 100.0, 12.0, 400);
        let track = build_track("oval", &pts, closed);
        let car = CarParams::default();
        let config = CollocationConfig {
            n_nodes: 50,
            closed: true,
            ..CollocationConfig::default()
        };
        let opt = CollocationOptimizer::new(config, &track, &car);

        let baseline = opt.optimize_gn(&gn_cfg());
        let with_bias = opt.optimize_with_brake_bias(&gn_cfg());

        let bias = with_bias
            .brake_bias
            .as_ref()
            .expect("brake_bias should be Some");
        assert_eq!(bias.len(), 50);
        assert!(
            bias.iter().all(|&b| (0.50..=0.80).contains(&b)),
            "all bias values within bounds"
        );

        // No dynamic coupling yet, so the lap time should be close to baseline.
        assert!(with_bias.lap_time.is_finite() && with_bias.lap_time > 0.0);
        assert!(
            (with_bias.lap_time - baseline.lap_time).abs() / baseline.lap_time < 0.05,
            "lap time {} should be close to baseline {}",
            with_bias.lap_time,
            baseline.lap_time
        );
    }
}
