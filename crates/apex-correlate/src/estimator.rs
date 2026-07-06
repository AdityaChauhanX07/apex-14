//! RTS smoother (extended Kalman smoother) on the single-track model.
//!
//! The QSS channel inference ([`crate::infer`]) inverts a *point-mass* model and
//! is blind to the vehicle's rotational state: it never sees slip angles, yaw
//! rate as a dynamic quantity, or the transient lag between steering and the
//! car's response. This module recovers those by running the **single-track
//! (bicycle) model** ([`apex_physics::BicycleModel`]) as the process model of an
//! extended Kalman filter over the measured lap, then smoothing with the
//! Rauch–Tung–Striebel (RTS) backward recursion.
//!
//! The controls (steering angle, drive force) are **unknown** — measured
//! telemetry has no calibrated steering channel. We use the standard
//! input-estimation trick: augment the state with slowly-varying `delta` and
//! `f_drive` random-walk states and let the filter infer them from the observed
//! motion. See `docs/math/estimation.md` for the full formulation, the Q/R
//! tuning rationale, NIS consistency definition, and scope limits.
//!
//! ## State (8) and measurements (3–4)
//!
//! State `x = [X, Y, psi, vx, vy, r, delta, f_drive]` (the bicycle's own 6-state
//! ordering, plus the two augmented input states). Measurements are, per epoch,
//! whichever of these are available: aligned GPS position `X, Y` (m), the
//! low-noise `speed` channel (m/s), and a **course** pseudo-measurement — the
//! motion direction `atan2(ΔY, ΔX)` from consecutive GPS points, which observes
//! `psi + beta`. Without the course term the heading is only weakly observable
//! and the EKF diverges within a lap; with it the filter is stable. The course
//! term is derived from the same GPS points as the position update, so its noise
//! is deliberately inflated (`course_sigma`) to avoid over-counting that
//! information — see `docs/math/estimation.md`.

// Dense matrix/vector code throughout this module (and its tests) indexes by
// range loop, which reads more clearly than `enumerate()` for `A[i][j]`-style
// linear algebra.
#![allow(clippy::needless_range_loop)]

use apex_integrator::{rk4_step_generic, OdeSystemGeneric};
use apex_math::{Dual, Float};
use apex_physics::{BicycleModel, CarParams, PacejkaTire};
use apex_telemetry::ChannelId;

use crate::error::CorrelateError;
use crate::telemetry::Telemetry;

/// State-vector dimension (6 vehicle + 2 augmented input states).
pub const N: usize = 8;
/// Measurement-vector dimension: `[X, Y, speed]`.
pub const M: usize = 3;

// State indices.
const IX: usize = 0;
const IY: usize = 1;
const IPSI: usize = 2;
const IVX: usize = 3;
const IVY: usize = 4;
const IR: usize = 5;
const IDELTA: usize = 6;
const IFDRIVE: usize = 7;

type Mat = Vec<Vec<f64>>;

/// Estimator tuning. All noise densities are documented per state/measurement in
/// physical units; defaults are motivated in `docs/math/estimation.md`.
#[derive(Debug, Clone)]
pub struct EstimatorConfig {
    /// Continuous-time process-noise spectral density on the diagonal, one entry
    /// per state in units of `state_unit² / s`. Discretized per step as
    /// `Q_d = q · dt`. Order matches the state vector.
    pub q: [f64; N],
    /// Position measurement noise std (m), for both X and Y. Bounded above by the
    /// telemetry align RMS (~4.1 m at Silverstone); the align residual conflates
    /// true GPS noise with driven-line-vs-centerline offset, so the per-sample
    /// GPS noise is smaller — default 3.0 m.
    pub pos_sigma: f64,
    /// Speed measurement noise std (m/s). The speed channel is derived and
    /// low-noise — default 0.5 m/s.
    pub speed_sigma: f64,
    /// Course (motion-direction) pseudo-measurement noise std (rad). The heading
    /// is otherwise only weakly observable from position+speed and the EKF
    /// diverges without it; a differenced-position course pins `psi + beta`. Its
    /// noise is deliberately **inflated** (default 0.20 rad) because it is derived
    /// from the same GPS points as the position update — see the module doc /
    /// `docs/math/estimation.md`.
    pub course_sigma: f64,
    /// Whether to apply the course pseudo-measurement (default `true`).
    pub use_course: bool,
    /// Minimum inter-sample displacement (m) for a usable course pseudo-measure.
    /// Below this the direction of two noisy points is meaningless, so it is
    /// skipped. Default 2.0 m.
    pub course_min_disp: f64,
    /// Initial-covariance diagonal (variance per state).
    pub p0: [f64; N],
    /// Innovation gate: reject an update whose normalized innovation squared
    /// (NIS, Mahalanobis²) exceeds this. Set generously (default 100): the single-
    /// track model does not perfectly match the real car, so honest model error
    /// routinely pushes NIS above the nominal χ² tail; rejecting those would starve
    /// the filter of the very position/course corrections that keep it closed-loop
    /// and stable. The gate is a guard against GROSS outliers (e.g. a GPS jump),
    /// not a consistency test — that is what the reported NIS statistics are for. A
    /// rejected update becomes a prediction-only step.
    pub nis_gate: f64,
    /// RK4 substep target (s): each measurement interval `dt` is integrated in
    /// `ceil(dt / substep_dt)` equal RK4 steps for propagation accuracy.
    pub substep_dt: f64,
    /// Mean-reversion rate (1/s) of the augmented steering state toward zero. The
    /// steering angle is only observable through the yaw dynamics when the front
    /// tire is loaded; on straights / at small slip it is unobservable and a pure
    /// random walk drifts unboundedly (and locks into tire saturation, corrupting
    /// the front slip angle). Modeling `delta` as an Ornstein–Uhlenbeck process
    /// `d(delta)/dt = −rate·delta` anchors it — steering relaxes to centre absent
    /// evidence, while a corner's yaw dynamics still pull it to the real value.
    /// Default 1.0 (≈1 s relaxation). Set to 0 to recover a pure random walk.
    pub delta_revert: f64,
    /// Maximum EKF predict-step size (s). A measurement interval longer than this
    /// is split into several equal predict sub-steps (each with its own Jacobian
    /// and process-noise increment) before the update. Telemetry epochs can be
    /// long (up to ~0.45 s here) and one linearization over the whole interval is
    /// poor in the tight, low-speed corners where the tire is most nonlinear;
    /// sub-stepping keeps each Jacobian valid. Default 0.1 s.
    pub max_predict_dt: f64,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        EstimatorConfig {
            // X,Y,psi: kinematics integrate velocity almost exactly → tiny slack.
            // vx,vy,r: unmodeled effects (load transfer, combined slip) → moderate.
            // delta,f_drive: the unknown inputs, modeled as random walks. Their
            // densities are kept MODEST, not loose: the augmented inputs are only
            // observable through the dynamics (no direct measurement), and at low
            // speed / small slip they are weakly determined, so an over-loose
            // density lets them run into the nonlinear tire regime and diverge.
            // Per-step (dt≈0.13 s) std: delta ≈ 0.005 rad, f_drive ≈ 0.8 kN — the
            // steering density is intentionally tight (delta is weakly observable);
            // corners still pull it via the state correlations and yaw dynamics.
            // Units: m²/s, m²/s, rad²/s, (m/s)²/s ×2, (rad/s)²/s, rad²/s, N²/s.
            q: [
                0.05,   // X
                0.05,   // Y
                5.0e-4, // psi
                2.0,    // vx
                2.0,    // vy
                0.5,    // r
                2.0e-4, // delta (steering-rate variance density)
                5.0e6,  // f_drive (drive-force-rate variance density)
            ],
            pos_sigma: 3.0,
            speed_sigma: 0.5,
            course_sigma: 0.20,
            use_course: true,
            course_min_disp: 2.0,
            p0: [
                9.0,   // X   (pos_sigma²)
                9.0,   // Y
                0.04,  // psi (0.2 rad)²
                4.0,   // vx  (2 m/s)²
                4.0,   // vy
                0.04,  // r   (0.2 rad/s)²
                0.04,  // delta (0.2 rad)²
                1.0e8, // f_drive (1e4 N)²
            ],
            nis_gate: 100.0,
            substep_dt: 0.02,
            delta_revert: 1.0,
            max_predict_dt: 0.1,
        }
    }
}

impl EstimatorConfig {
    /// A deterministic label encoding the tuning, for provenance (fed to
    /// `apex_telemetry::settings_hash_for_mode`). Two configs that estimate
    /// identically produce the same label.
    pub fn settings_label(&self) -> String {
        let q: Vec<String> = self.q.iter().map(|v| format!("{v:e}")).collect();
        let p0: Vec<String> = self.p0.iter().map(|v| format!("{v:e}")).collect();
        format!(
            "estimate.rts-single-track.q=[{}].pos_sigma={:e}.speed_sigma={:e}.course_sigma={:e}.use_course={}.course_min_disp={:e}.p0=[{}].nis_gate={:e}.substep_dt={:e}.delta_revert={:e}",
            q.join(","),
            self.pos_sigma,
            self.speed_sigma,
            self.course_sigma,
            self.use_course,
            self.course_min_disp,
            p0.join(","),
            self.nis_gate,
            self.substep_dt,
            self.delta_revert,
        ) + &format!(".max_predict_dt={:e}", self.max_predict_dt)
    }
}

/// Augmented single-track process model: the 6-state [`BicycleModel`] with the
/// steering angle and drive force promoted from controls to random-walk states
/// (indices 6, 7). Their time derivative is zero — process noise alone drives
/// them, so the filter treats them as slowly varying unknown inputs.
struct AugBicycle<'a> {
    params: &'a CarParams,
    tire: &'a PacejkaTire,
    /// Steering mean-reversion rate (1/s); see [`EstimatorConfig::delta_revert`].
    delta_revert: f64,
}

impl<T: Float> OdeSystemGeneric<T, N, 0> for AugBicycle<'_> {
    fn derivatives_generic(&self, s: &[T; N], _c: &[T; 0], t: T) -> [T; N] {
        let base = [s[IX], s[IY], s[IPSI], s[IVX], s[IVY], s[IR]];
        let ctrl = [s[IDELTA], s[IFDRIVE]];
        let bike = BicycleModel {
            params: self.params,
            tire: self.tire,
        };
        let d = bike.derivatives_generic(&base, &ctrl, t);
        // Augmented inputs: steering relaxes toward zero (Ornstein–Uhlenbeck),
        // drive force is a pure random walk (it is observable through speed).
        let ddelta = s[IDELTA] * (-self.delta_revert);
        [d[0], d[1], d[2], d[3], d[4], d[5], ddelta, T::zero()]
    }
}

impl AugBicycle<'_> {
    /// Propagate `x` forward by `dt` with `n_sub` equal RK4 substeps, over any
    /// `Float`. Seeded with duals it returns the state and the full-interval
    /// sensitivity in the dual parts.
    fn propagate<T: Float>(&self, x: &[T; N], dt: T, n_sub: usize) -> [T; N] {
        let h = dt / T::from_f64(n_sub as f64);
        let mut s = *x;
        let mut t = T::zero();
        let control: [T; 0] = [];
        for _ in 0..n_sub {
            s = rk4_step_generic(self, &s, &control, t, h);
            t = t + h;
        }
        s
    }

    /// Propagated mean (f64) and the discrete transition Jacobian `F = ∂x_{k+1}/
    /// ∂x_k` (via forward-mode duals, one pass per input state).
    fn propagate_with_jacobian(&self, x: &[f64; N], dt: f64, n_sub: usize) -> ([f64; N], Mat) {
        let mut f = vec![vec![0.0; N]; N];
        let mut mean = [0.0; N];
        for j in 0..N {
            let seed: [Dual; N] = std::array::from_fn(|i| {
                if i == j {
                    Dual::variable(x[i])
                } else {
                    Dual::constant(x[i])
                }
            });
            let out = self.propagate(&seed, Dual::constant(dt), n_sub);
            for i in 0..N {
                f[i][j] = out[i].dual;
                if j == 0 {
                    mean[i] = out[i].real;
                }
            }
        }
        (mean, f)
    }
}

/// Number of RK4 substeps for an interval `dt` under the config.
fn n_substeps(dt: f64, cfg: &EstimatorConfig) -> usize {
    ((dt / cfg.substep_dt).ceil() as usize).max(1)
}

/// Per-lap smoother output. All vectors have one entry per input sample.
#[derive(Debug, Clone)]
pub struct SmootherResult {
    /// Smoothed state means, `[X, Y, psi, vx, vy, r, delta, f_drive]` per sample.
    pub state: Vec<[f64; N]>,
    /// Smoothed marginal standard deviations per state (√diag of P_smooth).
    pub std: Vec<[f64; N]>,
    /// Front-axle slip angle (rad).
    pub slip_front: Vec<f64>,
    /// Rear-axle slip angle (rad).
    pub slip_rear: Vec<f64>,
    /// Body slip angle beta = atan2(vy, vx) (rad).
    pub beta: Vec<f64>,
    /// Per-epoch NIS (Mahalanobis²) at the forward update; NaN at gaps / rejected
    /// updates / the initial epoch.
    pub nis: Vec<f64>,
    /// Consistency + robustness diagnostics.
    pub diagnostics: Diagnostics,
}

/// Filter-consistency and robustness diagnostics.
#[derive(Debug, Clone)]
pub struct Diagnostics {
    /// Total samples.
    pub n_samples: usize,
    /// Epochs with a measurement update applied.
    pub n_updates: usize,
    /// Epochs skipped for a measurement gap (NaN input).
    pub n_gaps: usize,
    /// Updates whose innovation exceeded the gate and were soft-rejected
    /// (measurement noise inflated to bound their influence). High counts flag
    /// model mismatch or outlier-prone regions, not filter failure.
    pub n_rejected: usize,
    /// Measurement degrees of freedom (= [`M`]); the NIS target mean.
    pub nis_dof: usize,
    /// Mean NIS over applied updates (target ≈ `nis_dof`).
    pub nis_mean: f64,
    /// NIS 5th / 50th / 95th percentiles over applied updates.
    pub nis_p05: f64,
    pub nis_p50: f64,
    pub nis_p95: f64,
    /// Fraction of applied updates with NIS ≤ the 3-DOF χ² 95% bound (7.815);
    /// a well-tuned filter sits near 0.95.
    pub nis_within_95: f64,
}

/// Run the EKF forward pass + RTS backward smoother on a time-ordered lap.
///
/// * `t` — measurement epoch times (s), monotone non-decreasing.
/// * `x`, `y` — aligned world position (m); NaN marks a gap.
/// * `speed` — speed magnitude (m/s); NaN marks a gap.
///
/// A sample is a **gap** (prediction-only, no update) if any of x/y/speed is
/// non-finite there. Returns an error only for structurally unusable input
/// (fewer than 2 samples, or no finite sample to initialize from).
pub fn smooth_states(
    t: &[f64],
    x: &[f64],
    y: &[f64],
    speed: &[f64],
    car: &CarParams,
    tire: &PacejkaTire,
    cfg: &EstimatorConfig,
) -> Result<SmootherResult, CorrelateError> {
    let n = t.len();
    if n < 2 || x.len() != n || y.len() != n || speed.len() != n {
        return Err(CorrelateError::EstimatorInput(
            "estimator needs at least 2 aligned t/x/y/speed samples",
        ));
    }
    let model = AugBicycle {
        params: car,
        tire,
        delta_revert: cfg.delta_revert,
    };

    // --- initialize from the first finite sample ---
    let first = (0..n)
        .find(|&i| x[i].is_finite() && y[i].is_finite() && speed[i].is_finite())
        .ok_or(CorrelateError::EstimatorInput(
            "no finite x/y/speed sample to initialize from",
        ))?;
    // Heading from the first finite forward displacement.
    let psi0 = {
        let mut h = 0.0;
        for j in (first + 1)..n {
            if x[j].is_finite() && y[j].is_finite() {
                let dx = x[j] - x[first];
                let dy = y[j] - y[first];
                if dx.hypot(dy) > 1.0 {
                    h = dy.atan2(dx);
                    break;
                }
            }
        }
        h
    };
    let v0 = speed[first].max(1.0);
    let fdrive0 = car.drag_force(v0) + car.rolling_resistance_force();
    let mut x_hat = [x[first], y[first], psi0, v0, 0.0, 0.0, 0.0, fdrive0];
    let mut p = diag(&cfg.p0);

    // Storage for the RTS backward pass.
    let mut filt_x = vec![[0.0; N]; n];
    let mut filt_p: Vec<Mat> = vec![vec![vec![0.0; N]; N]; n];
    let mut pred_x = vec![[0.0; N]; n];
    let mut pred_p: Vec<Mat> = vec![vec![vec![0.0; N]; N]; n];
    let mut fmat: Vec<Mat> = vec![vec![vec![0.0; N]; N]; n];
    let mut nis = vec![f64::NAN; n];
    let mut nis_dof = vec![0usize; n];

    let mut n_updates = 0;
    let mut n_gaps = 0;
    let mut n_rejected = 0;
    // Last finite measured position, for the course pseudo-measurement.
    let mut last_pos: Option<(f64, f64)> = None;

    // The first stored epoch is the initialization (its prior == posterior).
    filt_x[first] = x_hat;
    filt_p[first] = p.clone();
    pred_x[first] = x_hat;
    pred_p[first] = p.clone();
    fmat[first] = identity();
    // Apply an update at the init epoch (position + speed; no course yet).
    let rows0 = build_rows(&x_hat, x[first], y[first], speed[first], None, cfg);
    match apply_update(&x_hat, &p, &rows0, cfg.nis_gate) {
        UpdateOutcome::Applied {
            x_post,
            p_post,
            nis: s,
            dof,
            downweighted,
        } => {
            x_hat = x_post;
            p = p_post;
            filt_x[first] = x_hat;
            filt_p[first] = p.clone();
            nis[first] = s;
            nis_dof[first] = dof;
            n_updates += 1;
            if downweighted {
                n_rejected += 1;
            }
        }
        UpdateOutcome::NoMeasurement => n_gaps += 1,
    }
    if x[first].is_finite() && y[first].is_finite() {
        last_pos = Some((x[first], y[first]));
    }

    let mut prev = first;
    for k in (first + 1)..n {
        let dt = (t[k] - t[prev]).max(1e-3);

        // --- predict prev → k: split long intervals into sub-steps so each
        // EKF Jacobian stays valid.  P⁻ = F P Fᵀ + Q_d per sub-step; the composite
        // transition Jacobian (product over sub-steps) is stored for the RTS pass.
        let n_pred = ((dt / cfg.max_predict_dt).ceil() as usize).max(1);
        let h = dt / n_pred as f64;
        let n_sub = n_substeps(h, cfg);
        let mut f_comp = identity();
        for _ in 0..n_pred {
            let (mean, f_sub) = model.propagate_with_jacobian(&x_hat, h, n_sub);
            let q_d = scaled_diag(&cfg.q, h);
            let mut p_new = mat_add(&mat_mul(&mat_mul(&f_sub, &p), &transpose(&f_sub)), &q_d);
            symmetrize(&mut p_new);
            p = p_new;
            x_hat = mean;
            f_comp = mat_mul(&f_sub, &f_comp);
        }

        pred_x[k] = x_hat;
        pred_p[k] = p.clone();
        fmat[k] = f_comp;

        // --- update at k ---
        let rows = build_rows(&x_hat, x[k], y[k], speed[k], last_pos, cfg);
        match apply_update(&x_hat, &p, &rows, cfg.nis_gate) {
            UpdateOutcome::Applied {
                x_post,
                p_post,
                nis: s,
                dof,
                downweighted,
            } => {
                x_hat = x_post;
                p = p_post;
                nis[k] = s;
                nis_dof[k] = dof;
                n_updates += 1;
                if downweighted {
                    n_rejected += 1;
                }
            }
            UpdateOutcome::NoMeasurement => n_gaps += 1,
        }
        if x[k].is_finite() && y[k].is_finite() {
            last_pos = Some((x[k], y[k]));
        }
        filt_x[k] = x_hat;
        filt_p[k] = p.clone();
        prev = k;
    }

    // --- RTS backward pass ---
    let mut smo_x = filt_x.clone();
    let mut smo_p = filt_p.clone();
    for k in (first..(n - 1)).rev() {
        // Smoother gain C = P_filt[k] Fᵀ[k+1] (P_pred[k+1])⁻¹.
        let a = mat_mul(&filt_p[k], &transpose(&fmat[k + 1])); // 8×8
                                                               // Solve P_pred[k+1] · Cᵀ = Aᵀ  ⇒  columns of Cᵀ from solve_linear.
        let ppred = &pred_p[k + 1];
        let at = transpose(&a);
        let mut ct = vec![vec![0.0; N]; N];
        for col in 0..N {
            let rhs: Vec<f64> = (0..N).map(|row| at[row][col]).collect();
            let solved = apex_math::lm::solve_linear(ppred, &rhs).unwrap_or_else(|| vec![0.0; N]); // singular → zero gain (freeze)
            for row in 0..N {
                ct[row][col] = solved[row];
            }
        }
        let c = transpose(&ct);

        // x_s[k] = x_f[k] + C (x_s[k+1] − x_pred[k+1])
        let dx: Vec<f64> = (0..N).map(|i| smo_x[k + 1][i] - pred_x[k + 1][i]).collect();
        let cdx = mat_vec(&c, &dx);
        for i in 0..N {
            smo_x[k][i] = filt_x[k][i] + cdx[i];
        }
        // P_s[k] = P_f[k] + C (P_s[k+1] − P_pred[k+1]) Cᵀ
        let dp = mat_sub(&smo_p[k + 1], &pred_p[k + 1]);
        let cdpct = mat_mul(&mat_mul(&c, &dp), &transpose(&c));
        smo_p[k] = mat_add(&filt_p[k], &cdpct);
        symmetrize(&mut smo_p[k]);
    }

    // --- assemble outputs ---
    let lf = car.cog_to_front;
    let lr = car.cog_to_rear;
    let mut state = vec![[0.0; N]; n];
    let mut std = vec![[0.0; N]; n];
    let mut slip_front = vec![f64::NAN; n];
    let mut slip_rear = vec![f64::NAN; n];
    let mut beta = vec![f64::NAN; n];
    for k in first..n {
        state[k] = smo_x[k];
        for i in 0..N {
            std[k][i] = smo_p[k][i][i].max(0.0).sqrt();
        }
        let vx = smo_x[k][IVX];
        let vy = smo_x[k][IVY];
        let r = smo_x[k][IR];
        let delta = smo_x[k][IDELTA];
        let vx_safe = vx.max(1.0);
        slip_front[k] = delta - ((vy + r * lf) / vx_safe).atan();
        slip_rear[k] = -((vy - r * lr) / vx_safe).atan();
        beta[k] = vy.atan2(vx_safe);
    }
    // Samples before `first` (if any) copy the first estimate (no data to smooth).
    for k in 0..first {
        state[k] = state[first];
        std[k] = std[first];
        slip_front[k] = slip_front[first];
        slip_rear[k] = slip_rear[first];
        beta[k] = beta[first];
    }

    let diagnostics = build_diagnostics(&nis, &nis_dof, n, n_updates, n_gaps, n_rejected);

    Ok(SmootherResult {
        state,
        std,
        slip_front,
        slip_rear,
        beta,
        nis,
        diagnostics,
    })
}

/// One EKF measurement update at a single epoch. Returns the posterior mean,
/// One measurement row: its `H` row, the scalar innovation `y = z − h(x)`, and
/// the measurement-noise variance `r`.
type Row = ([f64; N], f64, f64);

/// Build the measurement rows available at an epoch from the finite channels.
/// Position (X, Y) and speed rows are added when their samples are finite; the
/// course pseudo-measurement is added when a usable previous position exists and
/// the inter-sample displacement clears `course_min_disp`.
fn build_rows(
    x_pred: &[f64; N],
    zx: f64,
    zy: f64,
    zspeed: f64,
    last_pos: Option<(f64, f64)>,
    cfg: &EstimatorConfig,
) -> Vec<Row> {
    let mut rows: Vec<Row> = Vec::with_capacity(4);
    let pos_var = cfg.pos_sigma * cfg.pos_sigma;
    if zx.is_finite() && zy.is_finite() {
        let mut hx = [0.0; N];
        hx[IX] = 1.0;
        rows.push((hx, zx - x_pred[IX], pos_var));
        let mut hy = [0.0; N];
        hy[IY] = 1.0;
        rows.push((hy, zy - x_pred[IY], pos_var));

        // Course pseudo-measurement: h = psi + atan2(vy, vx) (world velocity
        // direction). Observed course from consecutive GPS points.
        if cfg.use_course {
            if let Some((px, py)) = last_pos {
                let (dx, dy) = (zx - px, zy - py);
                let disp = dx.hypot(dy);
                if disp >= cfg.course_min_disp {
                    let z_course = dy.atan2(dx);
                    let vx = x_pred[IVX];
                    let vy = x_pred[IVY];
                    let sp2 = (vx * vx + vy * vy).max(1e-6);
                    let h = x_pred[IPSI] + vy.atan2(vx);
                    let mut hc = [0.0; N];
                    hc[IPSI] = 1.0;
                    hc[IVX] = -vy / sp2;
                    hc[IVY] = vx / sp2;
                    // Angular dilution of precision: two points `disp` apart with
                    // per-point noise `pos_sigma` give a course uncertainty of
                    // ≈ √2·pos_sigma / disp. Fold that into the floor so the course
                    // is trusted less at low speed (small disp) — exactly where a
                    // fixed course noise otherwise injects phantom lateral motion.
                    let dilution = std::f64::consts::SQRT_2 * cfg.pos_sigma / disp;
                    let var = cfg.course_sigma * cfg.course_sigma + dilution * dilution;
                    rows.push((hc, wrap_angle(z_course - h), var));
                }
            }
        }
    }
    if zspeed.is_finite() {
        let vx = x_pred[IVX];
        let vy = x_pred[IVY];
        let spd = (vx * vx + vy * vy).sqrt().max(1e-6);
        let mut hs = [0.0; N];
        hs[IVX] = vx / spd;
        hs[IVY] = vy / spd;
        rows.push((hs, zspeed - spd, cfg.speed_sigma * cfg.speed_sigma));
    }
    rows
}

/// Outcome of a measurement update at one epoch.
enum UpdateOutcome {
    /// Update applied.
    Applied {
        x_post: [f64; N],
        p_post: Mat,
        nis: f64,
        dof: usize,
        /// True if the innovation exceeded the gate and the measurement noise was
        /// inflated to bound its influence (a soft rejection).
        downweighted: bool,
    },
    /// No usable measurement at this epoch (a gap; prediction-only).
    NoMeasurement,
}

/// Generic EKF measurement update over an arbitrary set of rows. Joseph-form
/// covariance update; NIS is the Mahalanobis² of the stacked innovation.
///
/// The innovation gate is **soft**: rather than dropping a high-innovation
/// update (which would leave the filter open-loop and let a transient divergence
/// lock in — the update it needs to recover is exactly the one being rejected),
/// an update with `NIS > nis_gate` has its measurement noise inflated by
/// `NIS / nis_gate` so the effective NIS is capped at the gate. Gross outliers
/// (e.g. a GPS jump) are thereby heavily down-weighted without ever starving the
/// filter of correction. Such updates are flagged `downweighted`.
fn apply_update(x_pred: &[f64; N], p_pred: &Mat, rows: &[Row], nis_gate: f64) -> UpdateOutcome {
    let m = rows.len();
    if m == 0 {
        return UpdateOutcome::NoMeasurement;
    }
    let hmat: Mat = rows.iter().map(|(h, _, _)| h.to_vec()).collect(); // m×N
    let yv: Vec<f64> = rows.iter().map(|(_, y, _)| *y).collect();
    let rvar0: Vec<f64> = rows.iter().map(|(_, _, r)| *r).collect();

    // Nominal S = H P Hᵀ + R, and the nominal NIS.
    let hp = mat_mul(&hmat, p_pred); // m×N
    let hpht = mat_mul(&hp, &transpose(&hmat)); // m×N · N×m = m×m
    let mut s0 = hpht.clone();
    for i in 0..m {
        s0[i][i] += rvar0[i];
    }
    let Some(s0_inv_y) = apex_math::lm::solve_linear(&s0, &yv) else {
        return UpdateOutcome::NoMeasurement;
    };
    let nis: f64 = (0..m).map(|i| yv[i] * s0_inv_y[i]).sum();

    // Soft gate: inflate R (hence S) so the effective NIS is bounded by the gate.
    let downweighted = nis > nis_gate;
    let scale = if downweighted { nis / nis_gate } else { 1.0 };
    let rvar: Vec<f64> = rvar0.iter().map(|r| r * scale).collect();
    let mut s = hpht;
    for i in 0..m {
        s[i][i] += rvar[i];
    }

    // Kalman gain K = P Hᵀ S⁻¹  (N×m), from  S Kᵀ = (P Hᵀ)ᵀ.
    let pht = mat_mul(p_pred, &transpose(&hmat)); // N×m
    let phtt = transpose(&pht); // m×N
    let mut kt = vec![vec![0.0; N]; m]; // Kᵀ (m×N)
    for col in 0..N {
        let rhs: Vec<f64> = (0..m).map(|row| phtt[row][col]).collect();
        let Some(sol) = apex_math::lm::solve_linear(&s, &rhs) else {
            return UpdateOutcome::NoMeasurement;
        };
        for row in 0..m {
            kt[row][col] = sol[row];
        }
    }
    let k = transpose(&kt); // N×m

    // x⁺ = x⁻ + K y.
    let ky = mat_vec(&k, &yv);
    let mut x_post = *x_pred;
    for i in 0..N {
        x_post[i] += ky[i];
    }

    // Joseph form: P⁺ = (I−KH) P (I−KH)ᵀ + K R Kᵀ.
    let kh = mat_mul(&k, &hmat); // N×N
    let mut ikh = identity();
    for i in 0..N {
        for j in 0..N {
            ikh[i][j] -= kh[i][j];
        }
    }
    let term1 = mat_mul(&mat_mul(&ikh, p_pred), &transpose(&ikh));
    let r_full: Mat = (0..m)
        .map(|i| (0..m).map(|j| if i == j { rvar[i] } else { 0.0 }).collect())
        .collect();
    let krkt = mat_mul(&mat_mul(&k, &r_full), &transpose(&k));
    let mut p_post = mat_add(&term1, &krkt);
    symmetrize(&mut p_post);
    UpdateOutcome::Applied {
        x_post,
        p_post,
        nis,
        dof: m,
        downweighted,
    }
}

/// Wrap an angle to (−π, π].
fn wrap_angle(a: f64) -> f64 {
    a.sin().atan2(a.cos())
}

/// χ² 95% upper bound for small degrees of freedom (used for NIS consistency).
fn chi2_95(dof: usize) -> f64 {
    match dof {
        1 => 3.841,
        2 => 5.991,
        3 => 7.815,
        4 => 9.488,
        _ => 11.070, // dof = 5
    }
}

fn build_diagnostics(
    nis: &[f64],
    nis_dof: &[usize],
    n_samples: usize,
    n_updates: usize,
    n_gaps: usize,
    n_rejected: usize,
) -> Diagnostics {
    let mut vals: Vec<f64> = nis.iter().copied().filter(|v| v.is_finite()).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| -> f64 {
        if vals.is_empty() {
            f64::NAN
        } else {
            vals[((p * (vals.len() - 1) as f64).round() as usize).min(vals.len() - 1)]
        }
    };
    let mean = if vals.is_empty() {
        f64::NAN
    } else {
        vals.iter().sum::<f64>() / vals.len() as f64
    };
    // Modal / typical measurement dof, and the fraction of updates under their
    // own dof's 95% χ² bound (a consistent filter sits near 0.95).
    let (mut within_n, mut within_d) = (0usize, 0usize);
    let mut dof_sum = 0usize;
    for (i, &nv) in nis.iter().enumerate() {
        if nv.is_finite() {
            within_d += 1;
            dof_sum += nis_dof[i];
            if nv <= chi2_95(nis_dof[i]) {
                within_n += 1;
            }
        }
    }
    let within = if within_d == 0 {
        f64::NAN
    } else {
        within_n as f64 / within_d as f64
    };
    let dof = if within_d == 0 {
        M
    } else {
        ((dof_sum as f64 / within_d as f64).round() as usize).max(1)
    };
    Diagnostics {
        n_samples,
        n_updates,
        n_gaps,
        n_rejected,
        nis_dof: dof,
        nis_mean: mean,
        nis_p05: pct(0.05),
        nis_p50: pct(0.50),
        nis_p95: pct(0.95),
        nis_within_95: within,
    }
}

// --- output helpers -------------------------------------------------------

/// The channels this estimator appends to a telemetry set.
pub const OUTPUT_CHANNELS: &[ChannelId] = &[
    ChannelId::SlipAngleFront,
    ChannelId::SlipAngleRear,
    ChannelId::BodySlipAngle,
    ChannelId::YawRate,
    ChannelId::LateralV,
];

/// Attach the smoothed dynamic-state channels to a copy of `aligned` (same grid,
/// same rows). Returns the augmented telemetry ready for CSV writing; provenance
/// metadata is added by the caller.
pub fn attach_estimated_channels(aligned: &Telemetry, res: &SmootherResult) -> Telemetry {
    let mut out = aligned.clone();
    out.channels
        .insert(ChannelId::SlipAngleFront, res.slip_front.clone());
    out.channels
        .insert(ChannelId::SlipAngleRear, res.slip_rear.clone());
    out.channels
        .insert(ChannelId::BodySlipAngle, res.beta.clone());
    out.channels.insert(
        ChannelId::YawRate,
        res.state.iter().map(|s| s[IR]).collect(),
    );
    out.channels.insert(
        ChannelId::LateralV,
        res.state.iter().map(|s| s[IVY]).collect(),
    );
    out
}

/// Serialize the per-state std-dev diagnostics + NIS statistics to a JSON string
/// (the estimator sidecar). Keyed by channel-ish names; hand-rolled to avoid a
/// serde_json dependency in this crate's write path.
pub fn diagnostics_json(res: &SmootherResult) -> String {
    let names = [
        "x",
        "y",
        "psi",
        "vx",
        "vy",
        "yaw_rate",
        "steering_angle",
        "f_drive",
    ];
    let mut s = String::from("{\n  \"state_std\": {\n");
    for (i, name) in names.iter().enumerate() {
        let col: Vec<String> = res.std.iter().map(|v| format!("{:.6}", v[i])).collect();
        s.push_str(&format!("    \"{name}\": [{}]", col.join(",")));
        s.push_str(if i + 1 < names.len() { ",\n" } else { "\n" });
    }
    s.push_str("  },\n");
    let d = &res.diagnostics;
    s.push_str("  \"nis\": {\n");
    s.push_str(&format!("    \"dof\": {},\n", d.nis_dof));
    s.push_str(&format!("    \"mean\": {:.4},\n", d.nis_mean));
    s.push_str(&format!("    \"p05\": {:.4},\n", d.nis_p05));
    s.push_str(&format!("    \"p50\": {:.4},\n", d.nis_p50));
    s.push_str(&format!("    \"p95\": {:.4},\n", d.nis_p95));
    s.push_str(&format!(
        "    \"within_95_bound\": {:.4}\n",
        d.nis_within_95
    ));
    s.push_str("  },\n");
    s.push_str("  \"robustness\": {\n");
    s.push_str(&format!("    \"n_samples\": {},\n", d.n_samples));
    s.push_str(&format!("    \"n_updates\": {},\n", d.n_updates));
    s.push_str(&format!("    \"n_gaps\": {},\n", d.n_gaps));
    s.push_str(&format!("    \"n_rejected\": {}\n", d.n_rejected));
    s.push_str("  }\n}\n");
    s
}

// --- small dense linear algebra ------------------------------------------

fn identity() -> Mat {
    (0..N)
        .map(|i| (0..N).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

fn diag(d: &[f64; N]) -> Mat {
    (0..N)
        .map(|i| (0..N).map(|j| if i == j { d[i] } else { 0.0 }).collect())
        .collect()
}

/// Diagonal `d · scale` as a full N×N matrix (discretized process noise).
fn scaled_diag(d: &[f64; N], scale: f64) -> Mat {
    (0..N)
        .map(|i| {
            (0..N)
                .map(|j| if i == j { d[i] * scale } else { 0.0 })
                .collect()
        })
        .collect()
}

fn transpose(a: &Mat) -> Mat {
    let rows = a.len();
    let cols = a[0].len();
    (0..cols)
        .map(|j| (0..rows).map(|i| a[i][j]).collect())
        .collect()
}

fn mat_mul(a: &Mat, b: &Mat) -> Mat {
    let n = a.len();
    let m = b[0].len();
    let k = b.len();
    let mut out = vec![vec![0.0; m]; n];
    for (i, orow) in out.iter_mut().enumerate() {
        for l in 0..k {
            let ail = a[i][l];
            if ail == 0.0 {
                continue;
            }
            for (j, oj) in orow.iter_mut().enumerate() {
                *oj += ail * b[l][j];
            }
        }
    }
    out
}

fn mat_vec(a: &Mat, v: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(v).map(|(x, y)| x * y).sum())
        .collect()
}

fn mat_add(a: &Mat, b: &Mat) -> Mat {
    a.iter()
        .zip(b)
        .map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| x + y).collect())
        .collect()
}

fn mat_sub(a: &Mat, b: &Mat) -> Mat {
    a.iter()
        .zip(b)
        .map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| x - y).collect())
        .collect()
}

/// Force exact symmetry (average with the transpose) to fight round-off drift.
fn symmetrize(a: &mut Mat) {
    let n = a.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (a[i][j] + a[j][i]);
            a[i][j] = avg;
            a[j][i] = avg;
        }
    }
}

/// Invert a 3×3 matrix via cofactors. `None` if (near-)singular. Retained for
/// the linear-algebra unit tests; the update path uses `solve_linear` directly.
#[cfg(test)]
fn invert3(a: &Mat) -> Option<Mat> {
    let m = a;
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-18 {
        return None;
    }
    let inv_det = 1.0 / det;
    let mut out = vec![vec![0.0; 3]; 3];
    out[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    out[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    out[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    out[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
    out[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    out[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
    Some(out)
}

#[cfg(test)]
mod tests;
