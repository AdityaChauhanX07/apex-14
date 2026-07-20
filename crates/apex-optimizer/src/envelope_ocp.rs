//! Envelope free-trajectory optimal-control problem (OCP).
//!
//! The small `s`-domain OCP that optimizes the racing line against the cached
//! g-g-g envelope, solved by the bound-capable interior-point solver
//! ([`crate::ipm`]). It is the assembly step of the envelope-QSS workstream: the
//! trim solver produced the operating-point map, the envelope generator stored
//! it with C1 interpolation, the IP solver handles the binding track-edge
//! bounds — this module wires them into a minimum-lap-time problem.
//!
//! See `docs/math/envelope_ocp.md` for the kinematics + `a_y` derivation and
//! `docs/design/envelope-qss/free-trajectory-ocp.md` for the design choices and
//! validation.
//!
//! # States, controls, dynamics
//!
//! `s`-domain, arc length along the (possibly 3D) centerline. Per node:
//! states `{n, xi, v}` (lateral offset m, heading error rad, speed m/s),
//! controls `{a_x, kappa_cmd}` (longitudinal accel command m/s², path curvature
//! 1/m). The existing 2D `(1 - n*kappa)` transform (`point_mass.rs`) gives
//!
//! ```text
//! n'  = (1 - n*kappa) * tan(xi)
//! xi' = kappa_cmd * (1 - n*kappa)/cos(xi) - kappa
//! v'  = (a_x - drag(v)/m - roll/m) * (1 - n*kappa)/(v*cos(xi))
//! ```
//!
//! with `kappa = kappa(s)` the centerline curvature. The lateral acceleration
//! fed to the envelope is the vehicle's centripetal accel `a_y = v² * kappa_cmd`
//! (path curvature is the chosen lateral control — it makes `a_y` exact and,
//! crucially, keeps `kappa_cmd`'s influence on the heading dynamics `∂xi'/∂κ ≈ 1`
//! so the dynamics defects are directly controllable; see the math doc).
//!
//! # Objective & constraints
//!
//! Minimize `∫ dt/ds ds = ∫ (1 - n*kappa)/(v*cos(xi)) ds` (plus small
//! control-rate regularization). Subject to:
//! - dynamics defects (equalities, trapezoidal, periodic for a flying lap),
//! - the envelope inequality `|a| <= rho_eff(theta; v, g_z(s))`, `theta =
//!   atan2(a_y, a_x)`, with the **safety margin** `rho_eff = (1 - eps)*rho`
//!   (default `eps = 0.01`, covering the envelope's measured 0.76 %
//!   over-estimation — see the design doc),
//! - track-edge box bounds `-w_right <= n <= +w_left` (the IP solver's home
//!   turf), plus `v >= v_min` and `|xi| <= xi_max`.

use apex_math::{CsrBuilder, CsrMatrix, Dual, Float};
use apex_physics::{qss_lap_sim, CarParams, Envelope, GRAVITY};
use apex_track::Track;

use crate::ipm::{solve_ipm, IpmConfig, IpmStatus};
use crate::nlp::{NlpEvaluator, NlpProblem};

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

/// Configuration for the envelope free-trajectory OCP.
#[derive(Debug, Clone)]
pub struct EnvelopeOcpConfig {
    /// Number of `s`-domain nodes.
    pub n_nodes: usize,
    /// Closed (periodic, flying-lap) boundary. `false` is not yet supported.
    pub closed: bool,
    /// Envelope safety margin: `rho_eff = (1 - eps)*rho`. Default `0.01`.
    pub eps: f64,
    /// Lower speed bound (m/s), guarding the `1/v` in the dynamics.
    pub v_min: f64,
    /// Heading-error bound (rad); keeps `cos(xi) > 0`.
    pub xi_max: f64,
    /// Control-rate regularization weight on `a_x` (per (m/s²)² of node-to-node
    /// change). Small — it only smooths chatter.
    pub rate_weight_ax: f64,
    /// Control-rate regularization weight on `kappa_cmd` (per (1/m)² of
    /// node-to-node change). Small — it only smooths chatter.
    pub rate_weight_kappa: f64,
}

impl Default for EnvelopeOcpConfig {
    fn default() -> Self {
        EnvelopeOcpConfig {
            n_nodes: 60,
            closed: true,
            eps: 0.01,
            v_min: 5.0,
            xi_max: 1.2,
            rate_weight_ax: 1e-4,
            rate_weight_kappa: 1e-1,
        }
    }
}

/// Result of an envelope-OCP solve.
#[derive(Debug, Clone)]
pub struct EnvelopeOcpResult {
    /// Arc-length stations along the centerline (m).
    pub stations: Vec<f64>,
    /// Lateral offsets `n` (m, + = left).
    pub offsets: Vec<f64>,
    /// Heading errors `xi` (rad).
    pub headings: Vec<f64>,
    /// Speeds `v` (m/s).
    pub speeds: Vec<f64>,
    /// Longitudinal accel commands `a_x` (m/s²).
    pub ax: Vec<f64>,
    /// Path-curvature commands `kappa_cmd` (1/m).
    pub kappa_cmd: Vec<f64>,
    /// Total lap time (s).
    pub lap_time: f64,
    /// Max dynamics-defect (equality) violation.
    pub eq_violation: f64,
    /// Max envelope/bound inequality violation.
    pub ineq_violation: f64,
    /// Whether the IP solver converged.
    pub converged: bool,
    /// IP outer iterations.
    pub iterations: usize,
    /// IP terminal status.
    pub status: IpmStatus,
    /// Per-iteration interior-point diagnostics (mu, feasibility, etc.).
    pub log: Vec<crate::ipm::IpmLog>,
}

/// The envelope free-trajectory optimizer.
pub struct EnvelopeOcp<'a> {
    /// Problem configuration.
    pub config: EnvelopeOcpConfig,
    /// Centerline track (2D curvilinear; `kappa` is the horizontal curvature).
    pub track: &'a Track,
    /// Vehicle parameters (used only for drag / rolling resistance and mass).
    pub car: &'a CarParams,
    /// The cached g-g-g envelope providing `rho(theta; v, g_z)`.
    pub envelope: &'a Envelope,
    /// Per-node effective vertical acceleration `g_z(s)` (m/s²). Length
    /// `n_nodes`. Constant `GRAVITY` for a flat track.
    pub gz_profile: Vec<f64>,
}

impl<'a> EnvelopeOcp<'a> {
    /// Create an optimizer with a flat-track `g_z = GRAVITY` profile.
    pub fn new(
        config: EnvelopeOcpConfig,
        track: &'a Track,
        car: &'a CarParams,
        envelope: &'a Envelope,
    ) -> Self {
        let gz_profile = vec![GRAVITY; config.n_nodes];
        EnvelopeOcp {
            config,
            track,
            car,
            envelope,
            gz_profile,
        }
    }

    /// Create an optimizer with an explicit per-node `g_z(s)` profile (from the
    /// 3D QSS machinery). Panics if the profile length differs from `n_nodes`.
    pub fn with_gz_profile(
        config: EnvelopeOcpConfig,
        track: &'a Track,
        car: &'a CarParams,
        envelope: &'a Envelope,
        gz_profile: Vec<f64>,
    ) -> Self {
        assert_eq!(gz_profile.len(), config.n_nodes, "gz profile length");
        EnvelopeOcp {
            config,
            track,
            car,
            envelope,
            gz_profile,
        }
    }

    // --- layout ---

    fn n(&self) -> usize {
        self.config.n_nodes
    }
    fn n_vars(&self) -> usize {
        5 * self.n()
    }
    fn n_eq(&self) -> usize {
        3 * self.n() // periodic: N intervals x 3 states
    }
    fn n_ineq(&self) -> usize {
        self.n() // one envelope constraint per node
    }
    #[inline]
    fn idx_n(&self, k: usize) -> usize {
        k
    }
    #[inline]
    fn idx_xi(&self, k: usize) -> usize {
        self.n() + k
    }
    #[inline]
    fn idx_v(&self, k: usize) -> usize {
        2 * self.n() + k
    }
    #[inline]
    fn idx_ax(&self, k: usize) -> usize {
        3 * self.n() + k
    }
    #[inline]
    fn idx_kap(&self, k: usize) -> usize {
        4 * self.n() + k
    }

    /// Evenly spaced centerline stations over `[0, total_length)` (periodic).
    fn node_stations(&self) -> Vec<f64> {
        let n = self.n();
        let len = self.track.total_length;
        (0..n).map(|k| len * (k as f64) / (n as f64)).collect()
    }

    /// Spacing between adjacent nodes (periodic).
    fn ds(&self) -> f64 {
        self.track.total_length / self.n() as f64
    }

    /// Centerline curvature at every node.
    fn node_curvatures(&self) -> Vec<f64> {
        self.node_stations()
            .iter()
            .map(|&s| self.track.curvature_at(s))
            .collect()
    }

    /// Drag coefficient constant `k` with `drag = k*v²`.
    fn drag_k(&self) -> f64 {
        0.5 * self.car.air_density * self.car.drag_coeff * self.car.frontal_area
    }

    // --- warm start ---

    /// Warm start from the fixed-line (centerline) QSS solution: `n = 0`,
    /// `xi = 0`, `v = QSS speed`, controls consistent with following the
    /// centerline.
    fn warm_start(&self) -> Vec<f64> {
        let n = self.n();
        let s = self.node_stations();
        let ds = self.ds();
        let qss = qss_lap_sim(self.track, self.car);
        let kappa = self.node_curvatures();
        let m = self.car.mass;
        let roll = self.car.rolling_resistance_force();
        let dk = self.drag_k();

        let v: Vec<f64> = s
            .iter()
            .map(|&sk| interp(&qss.distances, &qss.speeds, sk).max(self.config.v_min))
            .collect();

        let mut x = vec![0.0; self.n_vars()];
        for k in 0..n {
            let kn = (k + 1) % n;
            // dv/ds via periodic forward difference; dv/dt = v*dv/ds.
            let dvds = (v[kn] - v[k]) / ds;
            let dvdt = v[k] * dvds;
            // a_x (envelope longitudinal axis) = dv/dt + drag/m + roll/m.
            let ax = dvdt + (dk * v[k] * v[k] + roll) / m;
            x[self.idx_n(k)] = 0.0;
            x[self.idx_xi(k)] = 0.0;
            x[self.idx_v(k)] = v[k];
            x[self.idx_ax(k)] = ax;
            x[self.idx_kap(k)] = kappa[k]; // follow the centerline
        }
        x
    }

    // --- dynamics (generic over Float for dual-number Jacobians) ---

    /// The three state derivatives `[n', xi', v']` at a node, as functions of the
    /// node's `(n, xi, v, a_x, kappa_cmd)` and the (constant) centerline
    /// curvature `kappa`.
    fn dynamics<T: Float>(&self, n: T, xi: T, v: T, ax: T, kap: T, kappa: f64) -> [T; 3] {
        let m = self.car.mass;
        let roll = self.car.rolling_resistance_force();
        let dk = self.drag_k();
        let one_nk = T::one() - n * kappa; // (1 - n*kappa)
        let cosxi = xi.cos();
        let np = one_nk * xi.tan();
        let xip = kap * one_nk / cosxi - T::from_f64(kappa);
        let drag = v * v * dk;
        let dvdt = ax - (drag + T::from_f64(roll)) * (1.0 / m);
        let vp = dvdt * one_nk / (v * cosxi);
        [np, xip, vp]
    }

    /// The 3x5 dynamics Jacobian at a node (rows `[n', xi', v']`, cols
    /// `[n, xi, v, a_x, kappa]`), by forward-mode AD (one dual seed per input).
    fn dynamics_jac(&self, vals: [f64; 5], kappa: f64) -> [[f64; 5]; 3] {
        let [n, xi, v, ax, kap] = vals;
        let mut jac = [[0.0; 5]; 3];
        for (col, _) in vals.iter().enumerate() {
            let seed = |i: usize| {
                if i == col {
                    Dual::variable(vals[i])
                } else {
                    Dual::constant(vals[i])
                }
            };
            let _ = (n, xi, v, ax, kap);
            let d = self.dynamics(seed(0), seed(1), seed(2), seed(3), seed(4), kappa);
            for (row, dr) in d.iter().enumerate() {
                jac[row][col] = dr.dual;
            }
        }
        jac
    }

    /// Objective integrand `g = (1 - n*kappa)/(v*cos(xi))` (`dt/ds`) at a node.
    fn integrand(&self, n: f64, xi: f64, v: f64, kappa: f64) -> f64 {
        (1.0 - n * kappa) / (v * xi.cos())
    }

    // --- envelope inequality ---

    /// Envelope inequality `g_i = |a| - (1-eps)*rho(theta; v, g_z)` at a node,
    /// plus its gradient w.r.t. `(v, a_x, kappa)`. `g_i <= 0` is feasible.
    fn envelope_ineq(&self, v: f64, ax: f64, kap: f64, gz: f64) -> (f64, [f64; 3]) {
        const DELTA: f64 = 1e-3; // smoothing so |a| and theta stay differentiable at 0
        let ay = v * v * kap;
        let r2 = ax * ax + ay * ay;
        let r = (r2 + DELTA * DELTA).sqrt();
        let theta = ay.atan2(ax);
        let (rho, grad) = self.envelope.rho_grad(theta, v, gz);
        let drho_dtheta = grad[0];
        let drho_dv = grad[1];
        let one_eps = 1.0 - self.config.eps;
        let rho_eff = one_eps * rho;

        let g = r - rho_eff;

        // d(ay)/dv, d(ay)/dkap
        let day_dv = 2.0 * v * kap;
        let day_dkap = v * v;
        // d(theta): atan2(ay, ax)
        let denom = r2.max(1e-12);
        let dtheta_dax = -ay / denom;
        let dtheta_day = ax / denom;
        // d|a|
        let dr_dax = ax / r;
        let dr_dv = (ay / r) * day_dv;
        let dr_dkap = (ay / r) * day_dkap;

        let dg_dax = dr_dax - one_eps * (drho_dtheta * dtheta_dax);
        let dg_dv = dr_dv - one_eps * (drho_dv + drho_dtheta * dtheta_day * day_dv);
        let dg_dkap = dr_dkap - one_eps * (drho_dtheta * dtheta_day * day_dkap);

        (g, [dg_dv, dg_dax, dg_dkap])
    }

    // --- problem definition & solve ---

    fn build_problem(&self) -> NlpProblem {
        let n = self.n();
        let nv = self.n_vars();
        let mut lower = vec![f64::NEG_INFINITY; nv];
        let mut upper = vec![f64::INFINITY; nv];
        let s = self.node_stations();
        for k in 0..n {
            let (wl, wr) = self.track.width_at(s[k]);
            // n positive = left; track edges: -w_right <= n <= +w_left.
            lower[self.idx_n(k)] = -wr;
            upper[self.idx_n(k)] = wl;
            lower[self.idx_xi(k)] = -self.config.xi_max;
            upper[self.idx_xi(k)] = self.config.xi_max;
            lower[self.idx_v(k)] = self.config.v_min;
        }
        NlpProblem {
            n_vars: nv,
            n_eq: self.n_eq(),
            n_ineq: self.n_ineq(),
            lower_bounds: lower,
            upper_bounds: upper,
        }
    }

    /// The shared interior-point configuration for the envelope OCP on **real
    /// circuits**. It is the config the mesh/config-robustness study settled on
    /// (`docs/design/envelope-qss/real-track-convergence.md`):
    /// - `rho_growth = 3.0` — a gentle augmented-Lagrangian penalty ramp. The
    ///   solver's default `10.0` drives feasibility so hard that `rho` saturates
    ///   `rho_max` before the *objective* has moved the line to the track edge,
    ///   freezing the iterate near the warm start. `3.0` keeps `rho` moderate
    ///   long enough for the racing line to migrate.
    /// - `mu_reduction = 0.5` — a slow barrier anneal, for the same reason: a
    ///   softer barrier lets `n` travel before the schedule terminates.
    /// - `al_contract = 0.1` — favour Hestenes–Powell multiplier updates over
    ///   penalty growth, so real lap-scale problems reach feasibility at a
    ///   moderate `rho` instead of saturating `rho_max` while still infeasible
    ///   (the Part-A `InfeasibleDetected` failure).
    /// - `rho_max = 3e6` — the penalty ceiling several real circuits genuinely
    ///   need: Monza, Catalunya and Spielberg reach feasibility only once `rho`
    ///   climbs to `≈ 1e5–1e6`.
    /// - `constraint_tol = 1e-4`, `obj_weight = 1.0`.
    ///
    /// **`rho_max` is a problem-scale knob, not a universal constant.** The
    /// mesh-robustness study established that no single `rho_max` serves both
    /// real circuits and the gentle synthetic validation tracks: at `rho_max =
    /// 3e6` the circle's racing line *freezes* near the centerline (the stiff
    /// penalty overwhelms the objective that migrates `n` to the edge), so the
    /// circle validation caps `rho_max` at `3e4`. An online adaptive ceiling was
    /// prototyped and does not bridge the gap — the circle's equality
    /// infeasibility makes large-but-stalling progress at high `rho` that is
    /// indistinguishable from a real circuit's progress-to-feasibility until it
    /// is too late. See the doc for the full account.
    ///
    /// Callers may override any field (e.g. `max_iterations`, a looser
    /// `constraint_tol` for a coarse mesh, or `rho_max` for a gentle track).
    pub fn recommended_ip_config() -> IpmConfig {
        IpmConfig {
            max_iterations: 1000,
            obj_weight: 1.0,
            constraint_tol: 1e-4,
            al_contract: 0.1,
            rho_max: 3e6,
            rho_growth: 3.0,
            mu_reduction: 0.5,
            ..IpmConfig::default()
        }
    }

    /// [`recommended_ip_config`](Self::recommended_ip_config) with the
    /// **block-tridiagonal** preconditioner and the regularization it requires.
    ///
    /// Two changes from the Jacobi config, and the second is not optional:
    ///
    /// - `preconditioner = BlockTridiag` — inverts the node-coupling structure
    ///   exactly (see [`crate::precond`]), which collapses inner CG from a
    ///   saturated 250 iterations to a median of ~2–6 and lets real circuits
    ///   converge past the documented `N >= 44` wall.
    /// - **`reg = 1e-1`, up from `1e-8`.** `Jeq` is `3N x 5N`, so `JeqᵀJeq` is
    ///   rank-deficient by `2N`: the `a_x` and `kappa` control directions carry
    ///   no bounds and hence no barrier term, leaving `reg` as their only
    ///   regularization. Truncated Jacobi-CG never resolved those directions and
    ///   so regularized them *implicitly* (the classical iterative-regularization
    ///   property of CG). An exact solve does resolve them, amplifying them by
    ///   `1/reg`, and the line search collapses. Restoring an explicit `reg` is
    ///   what makes the exact solve usable. Measured: at `N = 96`, `reg = 1e-8`
    ///   gives `LineSearchFailure` at `eq = 26`; `reg = 1e-1` gives `Optimal` at
    ///   `eq = 1.2e-5`.
    ///
    /// `reg = 1e-1` is the measured sweet spot: `1e-2` still stalls at `N = 128`,
    /// while `1e0` converges everywhere but visibly biases the objective
    /// (Silverstone 108 s vs 88 s — over-damped). See
    /// `docs/design/dynamic-ocp/kkt-precond.md` for the sweep.
    pub fn recommended_block_ip_config() -> IpmConfig {
        IpmConfig {
            preconditioner: crate::ipm::Preconditioner::BlockTridiag,
            reg: 1e-1,
            ..Self::recommended_ip_config()
        }
    }

    /// Solve the OCP with the interior-point solver.
    pub fn solve(&self, ip_config: &IpmConfig) -> EnvelopeOcpResult {
        let x0 = self.warm_start();
        let problem = self.build_problem();
        let evaluator = EnvelopeOcpEvaluator { ocp: self };
        let res = solve_ipm(&problem, &evaluator, &x0, ip_config);
        self.extract(&res.x, &res)
    }

    fn extract(&self, x: &[f64], res: &crate::ipm::IpmResult) -> EnvelopeOcpResult {
        let n = self.n();
        let stations = self.node_stations();
        let offsets: Vec<f64> = (0..n).map(|k| x[self.idx_n(k)]).collect();
        let headings: Vec<f64> = (0..n).map(|k| x[self.idx_xi(k)]).collect();
        let speeds: Vec<f64> = (0..n).map(|k| x[self.idx_v(k)]).collect();
        let ax: Vec<f64> = (0..n).map(|k| x[self.idx_ax(k)]).collect();
        let kappa_cmd: Vec<f64> = (0..n).map(|k| x[self.idx_kap(k)]).collect();
        let kappa = self.node_curvatures();
        let ds = self.ds();
        // lap time = ds * sum(g) (trapezoidal on a periodic mesh).
        let lap_time: f64 = (0..n)
            .map(|k| ds * self.integrand(offsets[k], headings[k], speeds[k], kappa[k]))
            .sum();
        EnvelopeOcpResult {
            stations,
            offsets,
            headings,
            speeds,
            ax,
            kappa_cmd,
            lap_time,
            eq_violation: res.eq_violation,
            ineq_violation: res.ineq_violation,
            converged: res.converged,
            iterations: res.iterations,
            status: res.status,
            log: res.history.clone(),
        }
    }
}

/// NLP evaluator wrapping an [`EnvelopeOcp`].
struct EnvelopeOcpEvaluator<'a, 'b> {
    ocp: &'b EnvelopeOcp<'a>,
}

impl EnvelopeOcpEvaluator<'_, '_> {
    fn node_vals(&self, x: &[f64], k: usize) -> [f64; 5] {
        let o = self.ocp;
        [
            x[o.idx_n(k)],
            x[o.idx_xi(k)],
            x[o.idx_v(k)],
            x[o.idx_ax(k)],
            x[o.idx_kap(k)],
        ]
    }
}

impl NlpEvaluator for EnvelopeOcpEvaluator<'_, '_> {
    fn objective(&self, x: &[f64]) -> f64 {
        let o = self.ocp;
        let n = o.n();
        let ds = o.ds();
        let kappa = o.node_curvatures();
        let mut j = 0.0;
        for k in 0..n {
            j += ds * o.integrand(x[o.idx_n(k)], x[o.idx_xi(k)], x[o.idx_v(k)], kappa[k]);
        }
        // control-rate regularization (periodic)
        for k in 0..n {
            let kn = (k + 1) % n;
            let dax = x[o.idx_ax(kn)] - x[o.idx_ax(k)];
            let dkap = x[o.idx_kap(kn)] - x[o.idx_kap(k)];
            j += o.config.rate_weight_ax * dax * dax + o.config.rate_weight_kappa * dkap * dkap;
        }
        j
    }

    fn objective_gradient(&self, x: &[f64]) -> Vec<f64> {
        let o = self.ocp;
        let n = o.n();
        let ds = o.ds();
        let kappa = o.node_curvatures();
        let mut g = vec![0.0; o.n_vars()];
        for k in 0..n {
            let nn = x[o.idx_n(k)];
            let xi = x[o.idx_xi(k)];
            let v = x[o.idx_v(k)];
            let cosxi = xi.cos();
            let one_nk = 1.0 - nn * kappa[k];
            // d/dn, d/dv, d/dxi of g = (1-n*kappa)/(v*cos xi)
            g[o.idx_n(k)] += ds * (-kappa[k] / (v * cosxi));
            g[o.idx_v(k)] += ds * (-one_nk / (v * v * cosxi));
            g[o.idx_xi(k)] += ds * (one_nk * xi.sin() / (v * cosxi * cosxi));
        }
        // rate regularization gradient (periodic): each u_k appears in intervals
        // (k-1 -> k) and (k -> k+1).
        for k in 0..n {
            let kn = (k + 1) % n;
            let kp = (k + n - 1) % n;
            let ax_k = x[o.idx_ax(k)];
            let dax_next = x[o.idx_ax(kn)] - ax_k;
            let dax_prev = ax_k - x[o.idx_ax(kp)];
            g[o.idx_ax(k)] += 2.0 * o.config.rate_weight_ax * (dax_prev - dax_next);
            let kap_k = x[o.idx_kap(k)];
            let dkap_next = x[o.idx_kap(kn)] - kap_k;
            let dkap_prev = kap_k - x[o.idx_kap(kp)];
            g[o.idx_kap(k)] += 2.0 * o.config.rate_weight_kappa * (dkap_prev - dkap_next);
        }
        g
    }

    fn equality_constraints(&self, x: &[f64]) -> Vec<f64> {
        let o = self.ocp;
        let n = o.n();
        let ds = o.ds();
        let kappa = o.node_curvatures();
        let mut c = vec![0.0; o.n_eq()];
        // per node dynamics
        let f: Vec<[f64; 3]> = (0..n)
            .map(|k| {
                let [nn, xi, v, ax, kap] = self.node_vals(x, k);
                o.dynamics(nn, xi, v, ax, kap, kappa[k])
            })
            .collect();
        for i in 0..n {
            let j = (i + 1) % n;
            for comp in 0..3 {
                let zi = match comp {
                    0 => x[o.idx_n(i)],
                    1 => x[o.idx_xi(i)],
                    _ => x[o.idx_v(i)],
                };
                let zj = match comp {
                    0 => x[o.idx_n(j)],
                    1 => x[o.idx_xi(j)],
                    _ => x[o.idx_v(j)],
                };
                c[3 * i + comp] = zj - zi - 0.5 * ds * (f[i][comp] + f[j][comp]);
            }
        }
        c
    }

    fn inequality_constraints(&self, x: &[f64]) -> Vec<f64> {
        let o = self.ocp;
        let n = o.n();
        (0..n)
            .map(|k| {
                let v = x[o.idx_v(k)];
                let ax = x[o.idx_ax(k)];
                let kap = x[o.idx_kap(k)];
                o.envelope_ineq(v, ax, kap, o.gz_profile[k]).0
            })
            .collect()
    }

    fn equality_jacobian(&self, x: &[f64]) -> CsrMatrix {
        let o = self.ocp;
        let n = o.n();
        let ds = o.ds();
        let kappa = o.node_curvatures();
        // per-node 3x5 dynamics Jacobians
        let jf: Vec<[[f64; 5]; 3]> = (0..n)
            .map(|k| o.dynamics_jac(self.node_vals(x, k), kappa[k]))
            .collect();

        let mut b = CsrBuilder::new(o.n_eq(), o.n_vars());
        // column index of local var c at node k
        let col = |k: usize, local: usize| -> usize {
            match local {
                0 => o.idx_n(k),
                1 => o.idx_xi(k),
                2 => o.idx_v(k),
                3 => o.idx_ax(k),
                _ => o.idx_kap(k),
            }
        };
        // `comp` (state row) and `local` (variable column) both index the dense
        // 3x5 node Jacobians and drive the sparse row/col placement, so the
        // range loops are the clear form here.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let j = (i + 1) % n;
            for comp in 0..3 {
                let row = 3 * i + comp;
                // state-difference part: +1 at zj[comp], -1 at zi[comp]
                b.add(row, col(i, comp), -1.0);
                b.add(row, col(j, comp), 1.0);
                // -0.5*ds * dfi/dvar (node i) and dfj/dvar (node j)
                for local in 0..5 {
                    let ci = jf[i][comp][local];
                    if ci != 0.0 {
                        b.add(row, col(i, local), -0.5 * ds * ci);
                    }
                    let cj = jf[j][comp][local];
                    if cj != 0.0 {
                        b.add(row, col(j, local), -0.5 * ds * cj);
                    }
                }
            }
        }
        b.build()
    }

    /// One block per mesh node, holding that node's `{n, xi, v, a_x, kappa}`.
    ///
    /// The layout is block-contiguous by quantity (`idx_n(k) = k`,
    /// `idx_xi(k) = N + k`, …), so node `k`'s variables sit at stride `N` — which
    /// is exactly what [`BlockStructure::strided`] describes. This is index
    /// metadata; nothing is repacked, so the Jacobi path is unaffected.
    ///
    /// The trapezoidal defect for interval `i` touches only nodes `i` and
    /// `(i+1) % N`, and the envelope inequality at node `k` touches only node
    /// `k`, so the condensed operator is block-tridiagonal apart from the
    /// flying-lap wrap.
    fn block_structure(&self) -> Option<crate::precond::BlockStructure> {
        Some(crate::precond::BlockStructure::strided(self.ocp.n(), 5))
    }

    fn inequality_jacobian(&self, x: &[f64]) -> CsrMatrix {
        let o = self.ocp;
        let n = o.n();
        let mut b = CsrBuilder::new(o.n_ineq(), o.n_vars());
        for k in 0..n {
            let v = x[o.idx_v(k)];
            let ax = x[o.idx_ax(k)];
            let kap = x[o.idx_kap(k)];
            let (_, grad) = o.envelope_ineq(v, ax, kap, o.gz_profile[k]);
            // grad = [dg/dv, dg/dax, dg/dkap]
            b.add(k, o.idx_v(k), grad[0]);
            b.add(k, o.idx_ax(k), grad[1]);
            b.add(k, o.idx_kap(k), grad[2]);
        }
        b.build()
    }
}
