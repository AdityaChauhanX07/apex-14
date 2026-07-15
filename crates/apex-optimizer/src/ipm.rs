//! Primal-dual interior-point solver for bound- and constraint-limited NLPs.
//!
//! Solves `min f(x)  s.t.  c_eq(x) = 0,  c_ineq(x) <= 0,  lb <= x <= ub` behind
//! the existing [`NlpEvaluator`] interface, with **no new linear-algebra
//! dependencies** — the Newton system is condensed to a symmetric
//! positive-definite operator and solved matrix-free by preconditioned
//! conjugate gradient, exactly the machinery the Gauss-Newton solver already
//! uses. It targets the documented projected-Newton **bound deadlock**
//! (`docs/design/gn-solver-bound-deadlock.md`): bounds are handled by a
//! log-barrier with genuine dual multipliers, so an active bound produces the
//! correct implicit multiplier instead of a projection that cancels the step.
//!
//! # Method (see `docs/design/envelope-qss/ip-solver.md`)
//!
//! - **Inequalities & bounds → primal-dual barrier.** Each `c_ineq_j(x) <= 0`
//!   gets a slack `s_I_j > 0` with `c_ineq + s_I = 0` and a multiplier `z_I > 0`;
//!   each finite bound gives an affine slack (`s_L = x - lb`, `s_U = ub - x`)
//!   with multipliers `z_L, z_U > 0`. Perturbed complementarity `s·z = mu`.
//! - **Equalities → augmented Lagrangian.** `c_eq` enters the merit as
//!   `y·c_eq + (rho/2)‖c_eq‖²`; the multiplier `y` is updated between barrier
//!   subproblems so `c_eq → 0` without driving `rho → ∞`. (For the collocation
//!   feasibility problem the objective weight is small and a feasible point
//!   exists, so the least-squares term alone reaches `c_eq ≈ 0`; the multiplier
//!   removes the residual objective bias.)
//! - **Hessian model.** `H = w_f·H_f + rho·JeqᵀJeq + Jineqᵀ Σ_I Jineq +
//!   diag(Σ_L + Σ_U) + reg·I`, where `H_f` is the objective Hessian-vector
//!   product (default zero → Gauss-Newton on the constraints). Every term is
//!   PSD and the barrier + `reg` make `H` SPD, so **CG applies**.
//! - **Condensation.** The bound/inequality multipliers are eliminated
//!   analytically, leaving `H·dx = rhs`; the multiplier steps are recovered by
//!   back-substitution. See the doc for the derivation.
//! - **Globalization.** Fraction-to-the-boundary step caps keep all slacks and
//!   multipliers positive; a backtracking line search on the barrier-augmented
//!   merit (with an ℓ1 penalty on inequality-slack infeasibility) enforces
//!   descent.
//! - **Barrier schedule.** Monotone Fiacco-McCormick: reduce `mu` by a fixed
//!   factor once the barrier subproblem is solved to `kappa_eps·mu`. (Mehrotra
//!   predictor-corrector / adaptive `mu` are noted as future work.)
//!
//! # Scaling
//!
//! Per `docs/design/nlp-scaling.md`, variable scaling is applied **inside** the
//! solver (callers pass unscaled problems): a guarded Jacobi column scaling is
//! measured once at `x0` and the whole solve runs in scaled space, then the
//! result is unscaled and feasibility is recomputed in SI. Column-only scaling
//! keeps `constraint_tol` and reported violations in SI by construction.
//!
//! # Determinism
//!
//! Fixed iteration order, no RNG, deterministic reductions — identical inputs
//! produce a bitwise-identical iterate history ([`IpmResult::history`]).

use apex_math::CsrMatrix;

use crate::nlp::{NlpEvaluator, NlpProblem};
use crate::scaling::{floor_scale, ScaledEvaluator, Scaling};

/// Terminal status of an interior-point solve. No silent failure — every exit
/// maps to one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpmStatus {
    /// Converged: primal/dual infeasibility and complementarity all within
    /// tolerance.
    Optimal,
    /// Hit the iteration cap before converging.
    MaxIter,
    /// The problem appears primal-infeasible (equality residual stalled at a
    /// large value while the penalty grew to its cap).
    InfeasibleDetected,
    /// The line search could not find a merit-decreasing step.
    LineSearchFailure,
}

/// One row of the iteration log (also the determinism-comparison record).
#[derive(Debug, Clone, PartialEq)]
pub struct IpmLog {
    /// Iteration index (0-based).
    pub iter: usize,
    /// Barrier parameter at this iteration.
    pub mu: f64,
    /// Max-abs equality residual (SI).
    pub primal_eq_inf: f64,
    /// Max inequality violation `max(0, c_ineq)` (SI).
    pub primal_ineq_inf: f64,
    /// Max-abs stationarity (dual) residual.
    pub dual_inf: f64,
    /// Max complementarity gap `|s·z - mu|`.
    pub comp: f64,
    /// Primal (x, slack) step length after line search.
    pub alpha_primal: f64,
    /// Dual (multiplier) step length.
    pub alpha_dual: f64,
    /// Inner CG iterations for this step.
    pub cg_iters: usize,
}

/// Result of an interior-point solve. Violations are reported in **SI** (the
/// unscaled problem), matching the Gauss-Newton solver's convention.
#[derive(Debug, Clone)]
pub struct IpmResult {
    /// Solution vector (SI, unscaled).
    pub x: Vec<f64>,
    /// Final objective value `f(x)`.
    pub objective: f64,
    /// Max-abs equality constraint violation (SI).
    pub eq_violation: f64,
    /// Max inequality constraint violation, `max(0, c_ineq)` (SI).
    pub ineq_violation: f64,
    /// Outer iterations performed.
    pub iterations: usize,
    /// Terminal status.
    pub status: IpmStatus,
    /// Whether the solve converged (`status == Optimal`).
    pub converged: bool,
    /// Per-iteration diagnostics (see [`IpmLog`]).
    pub history: Vec<IpmLog>,
}

/// Configuration for the interior-point solver.
#[derive(Debug, Clone)]
pub struct IpmConfig {
    /// Maximum outer iterations.
    pub max_iterations: usize,
    /// Initial barrier parameter.
    pub mu_init: f64,
    /// Barrier floor; below this the barrier is effectively off.
    pub mu_min: f64,
    /// Multiplicative barrier reduction factor `kappa_mu` (`0 < . < 1`).
    pub mu_reduction: f64,
    /// Reduce `mu` when the barrier-subproblem error `E(mu) <= kappa_eps·mu`.
    pub mu_solve_factor: f64,
    /// Fallback: reduce `mu` after this many Newton steps at a fixed `mu` even if
    /// the strict subproblem gate has not been met. Guarantees the barrier keeps
    /// annealing on problems where an active bound keeps the inner stationarity
    /// from vanishing at moderate `mu` (the collocation deadlock).
    pub inner_max_iters: usize,
    /// Convergence tolerance on the SI constraint violations.
    pub constraint_tol: f64,
    /// Convergence tolerance on stationarity + complementarity.
    pub opt_tol: f64,
    /// Diagonal regularization added to the condensed operator.
    pub reg: f64,
    /// Max inner CG iterations per Newton step.
    pub cg_max_iter: usize,
    /// Relative CG residual tolerance.
    pub cg_tol: f64,
    /// Fraction-to-the-boundary floor `tau_min` (`.99` typical).
    pub tau_min: f64,
    /// Weight on the true objective in the merit (small for feasibility-first
    /// collocation solves; `1.0` for genuine optimization / QPs).
    pub obj_weight: f64,
    /// Initial augmented-Lagrangian penalty on the equalities.
    pub rho_eq: f64,
    /// Cap on the augmented-Lagrangian penalty.
    pub rho_max: f64,
    /// Multiplicative growth applied to the AL penalty `rho` each time the
    /// equalities fail to contract by 4x at a schedule advance. The default
    /// `10.0` drives feasibility aggressively (good for pure feasibility /
    /// deadlock solves); optimization problems whose *objective* must reshape
    /// the primal (e.g. a racing line migrating to the track edge) need a
    /// gentler ramp so `rho` does not reach `rho_max` and freeze the iterate
    /// before the objective has done its work.
    pub rho_growth: f64,
    /// Contraction factor gating the augmented-Lagrangian multiplier update: at
    /// a schedule advance the equality multipliers are updated (`y += rho·c`)
    /// only if the equality infeasibility has fallen to `al_contract` of its
    /// value at the last update; otherwise `rho` is grown. The default `0.25`
    /// is a conservative Hestenes–Powell threshold. Problems where feasibility
    /// contracts slowly (large stiff periodic OCPs) converge far better with a
    /// looser value (e.g. `0.9`), which favours multiplier updates over penalty
    /// growth and so reaches feasibility at a moderate `rho` instead of racing
    /// `rho` to `rho_max` and freezing.
    pub al_contract: f64,
    /// How far to push the initial point strictly inside its bounds.
    pub bound_push: f64,
    /// Verbosity: `0` silent, `>0` prints the iteration log.
    pub verbose: usize,
}

impl Default for IpmConfig {
    fn default() -> Self {
        IpmConfig {
            max_iterations: 300,
            mu_init: 1.0,
            mu_min: 1e-9,
            mu_reduction: 0.2,
            mu_solve_factor: 10.0,
            inner_max_iters: 8,
            constraint_tol: 1e-6,
            opt_tol: 1e-6,
            reg: 1e-8,
            cg_max_iter: 250,
            cg_tol: 1e-8,
            tau_min: 0.99,
            obj_weight: 1.0,
            rho_eq: 1.0,
            rho_max: 1e8,
            rho_growth: 10.0,
            al_contract: 0.25,
            bound_push: 1e-2,
            verbose: 0,
        }
    }
}

impl apex_math::ContentHash for IpmConfig {
    fn hash_into(&self, w: &mut apex_math::HashWriter) {
        let IpmConfig {
            max_iterations,
            mu_init,
            mu_min,
            mu_reduction,
            mu_solve_factor,
            inner_max_iters,
            constraint_tol,
            opt_tol,
            reg,
            cg_max_iter,
            cg_tol,
            tau_min,
            obj_weight,
            rho_eq,
            rho_max,
            rho_growth,
            al_contract,
            bound_push,
            verbose: _, // cosmetic
        } = self;
        w.usize(*max_iterations);
        w.f64(*mu_init);
        w.f64(*mu_min);
        w.f64(*mu_reduction);
        w.f64(*mu_solve_factor);
        w.usize(*inner_max_iters);
        w.f64(*constraint_tol);
        w.f64(*opt_tol);
        w.f64(*reg);
        w.usize(*cg_max_iter);
        w.f64(*cg_tol);
        w.f64(*tau_min);
        w.f64(*obj_weight);
        w.f64(*rho_eq);
        w.f64(*rho_max);
        w.f64(*rho_growth);
        w.f64(*al_contract);
        w.f64(*bound_push);
    }
}

// --- small dense vector helpers (sequential -> deterministic) ---

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

fn max_pos(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |m, &x| m.max(x)).max(0.0)
}

/// Preconditioned conjugate gradient for an SPD operator supplied as a closure.
/// `minv` is the diagonal (Jacobi) preconditioner (`1/diag`). Returns the
/// solution and the iteration count.
fn pcg(
    apply: impl Fn(&[f64]) -> Vec<f64>,
    rhs: &[f64],
    minv: &[f64],
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, usize) {
    let n = rhs.len();
    let mut x = vec![0.0; n];
    let mut r = rhs.to_vec();
    let bnorm = norm(rhs).max(1e-30);
    if norm(&r) / bnorm < tol {
        return (x, 0);
    }
    let mut z: Vec<f64> = r.iter().zip(minv).map(|(&ri, &mi)| ri * mi).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut iters = 0;
    for k in 0..max_iter {
        iters = k + 1;
        let ap = apply(&p);
        let pap = dot(&p, &ap) + 1e-30;
        let alpha = rz / pap;
        for (xi, &pi) in x.iter_mut().zip(&p) {
            *xi += alpha * pi;
        }
        for (ri, &api) in r.iter_mut().zip(&ap) {
            *ri -= alpha * api;
        }
        if norm(&r) / bnorm < tol {
            break;
        }
        for (zi, (&ri, &mi)) in z.iter_mut().zip(r.iter().zip(minv)) {
            *zi = ri * mi;
        }
        let rz_new = dot(&r, &z);
        let beta = rz_new / (rz + 1e-30);
        for (pi, &zi) in p.iter_mut().zip(&z) {
            *pi = zi + beta * *pi;
        }
        rz = rz_new;
    }
    (x, iters)
}

/// Build a guarded Jacobi column scaling from the constraint Jacobians at `x0`.
///
/// `x_scale[j] = 1 / ‖[Jeq; Jineq][:, j]‖` when that column has real sensitivity,
/// else `1.0` (a variable touched by no constraint — e.g. a pure-objective QP
/// variable — is left unscaled rather than blown up by the floor). Constraint
/// (row) scales are held at `1.0` so `constraint_tol` and reported violations
/// stay in SI (the `nlp-scaling.md` column-only discipline).
fn build_scaling(evaluator: &impl NlpEvaluator, problem: &NlpProblem, x0: &[f64]) -> Scaling {
    let n = problem.n_vars;
    let mut sumsq = vec![0.0_f64; n];
    let accumulate = |sumsq: &mut [f64], j: &CsrMatrix| {
        for row in 0..j.nrows() {
            let (vals, cols) = j.row_entries(row);
            for (&v, &c) in vals.iter().zip(cols) {
                sumsq[c] += v * v;
            }
        }
    };
    if problem.n_eq > 0 {
        accumulate(&mut sumsq, &evaluator.equality_jacobian(x0));
    }
    if problem.n_ineq > 0 {
        accumulate(&mut sumsq, &evaluator.inequality_jacobian(x0));
    }
    let x_scale: Vec<f64> = sumsq
        .into_iter()
        .map(|s| {
            let nrm = s.sqrt();
            if nrm > floor_scale(0.0) {
                1.0 / nrm
            } else {
                1.0 // untouched by any constraint: leave unscaled
            }
        })
        .collect();
    Scaling {
        x_scale,
        c_eq_scale: vec![1.0; problem.n_eq],
        c_ineq_scale: vec![1.0; problem.n_ineq],
    }
}

/// Solve an NLP with the primal-dual interior-point method.
///
/// Builds variable scaling internally (callers pass an unscaled problem), runs
/// the barrier iteration in scaled space, then unscales the solution and
/// recomputes feasibility in SI.
pub fn solve_ipm(
    problem: &NlpProblem,
    evaluator: &impl NlpEvaluator,
    x0: &[f64],
    config: &IpmConfig,
) -> IpmResult {
    let scaling = build_scaling(evaluator, problem, x0);
    let scaled_eval = ScaledEvaluator {
        inner: evaluator,
        scaling: &scaling,
    };
    let scaled_problem = scaling.scale_problem(problem);
    let x0_scaled = scaling.scale_x(x0);

    let mut result = solve_ipm_core(&scaled_problem, &scaled_eval, &x0_scaled, config);

    // SI boundary: unscale and recompute violations against the raw evaluator.
    result.x = scaling.unscale_x(&result.x);
    result.eq_violation = max_abs(&evaluator.equality_constraints(&result.x));
    result.ineq_violation = max_pos(&evaluator.inequality_constraints(&result.x));
    result.objective = evaluator.objective(&result.x);
    result
}

/// The core barrier iteration, operating entirely in the space it is handed
/// (the caller does any scaling). Kept separate so the scaling wrapper and the
/// numerics are independently testable.
fn solve_ipm_core(
    problem: &NlpProblem,
    eval: &impl NlpEvaluator,
    x0: &[f64],
    config: &IpmConfig,
) -> IpmResult {
    let n = problem.n_vars;
    let n_eq = problem.n_eq;
    let m_i = problem.n_ineq;

    // Finite-bound index sets.
    let lb_idx: Vec<usize> = (0..n)
        .filter(|&i| problem.lower_bounds[i].is_finite())
        .collect();
    let ub_idx: Vec<usize> = (0..n)
        .filter(|&i| problem.upper_bounds[i].is_finite())
        .collect();
    let lb_val: Vec<f64> = lb_idx.iter().map(|&i| problem.lower_bounds[i]).collect();
    let ub_val: Vec<f64> = ub_idx.iter().map(|&i| problem.upper_bounds[i]).collect();

    // --- interior initial point ---
    let mut x = x0.to_vec();
    for (k, &i) in lb_idx.iter().enumerate() {
        let lb = lb_val[k];
        let push = config.bound_push * lb.abs().max(1.0);
        if let Some(pos) = ub_idx.iter().position(|&j| j == i) {
            // both bounds: clamp strictly inside, fall back to midpoint if tight.
            let ub = ub_val[pos];
            let w = ub - lb;
            let lo = lb + config.bound_push * w;
            let hi = ub - config.bound_push * w;
            x[i] = if lo < hi {
                x[i].max(lo).min(hi)
            } else {
                0.5 * (lb + ub)
            };
        } else {
            x[i] = x[i].max(lb + push);
        }
    }
    for (k, &i) in ub_idx.iter().enumerate() {
        // upper-only bounds (both-bounded already handled above).
        if !problem.lower_bounds[i].is_finite() {
            let ub = ub_val[k];
            let push = config.bound_push * ub.abs().max(1.0);
            x[i] = x[i].min(ub - push);
        }
    }

    // Slacks & multipliers.
    let s_min = 1e-8;
    let mut s_i: Vec<f64> = {
        let c_ineq = eval.inequality_constraints(&x);
        c_ineq.iter().map(|&c| (-c).max(s_min)).collect()
    };
    let mut z_i = vec![1.0_f64; m_i];
    let mut z_l = vec![1.0_f64; lb_idx.len()];
    let mut z_u = vec![1.0_f64; ub_idx.len()];
    let mut y = vec![0.0_f64; n_eq]; // equality (AL) multipliers

    let mut mu = config.mu_init;
    let mut rho = config.rho_eq;
    let mut history: Vec<IpmLog> = Vec::new();

    // Reference equality-norm for the augmented-Lagrangian update schedule.
    let mut eq_ref = {
        let c = eval.equality_constraints(&x);
        max_abs(&c).max(1e-12)
    };

    let scatter = |idx: &[usize], vals: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; n];
        for (&i, &v) in idx.iter().zip(vals) {
            out[i] = v;
        }
        out
    };

    let mut status = IpmStatus::MaxIter;
    let mut iterations = 0;
    let mut iters_at_mu = 0usize;
    // Best iterate seen, returned regardless of where the iteration stops. The
    // selection is objective-aware, not merely least-infeasible: among iterates
    // feasible to `constraint_tol` we keep the LOWEST-objective one (so a genuine
    // optimization run returns its optimized point, not the near-warm-start
    // iterate that happens to have the smallest defect); only if no iterate is
    // yet feasible do we fall back to least-infeasible.
    let mut x_best = x.clone();
    let mut best_infeas = f64::INFINITY;
    let mut best_obj = f64::INFINITY;
    let mut have_feasible = false;
    // Relative size of the last accepted primal step, in scaled space. Used to
    // gate the barrier-floor acceptance: the AL path leaves a large residual
    // dual infeasibility that is *not* the acceptance criterion, but we must
    // still not accept while the primal iterate is genuinely travelling toward
    // its optimum (e.g. an OCP whose racing line is migrating to the track
    // edge). Initialised to +inf so the very first iteration cannot accept.
    let mut last_rel_step = f64::INFINITY;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // --- evaluate problem functions ---
        let c_eq = eval.equality_constraints(&x);
        let c_ineq = eval.inequality_constraints(&x);
        let grad_f = eval.objective_gradient(&x);
        let j_eq = eval.equality_jacobian(&x);
        let jt_eq = j_eq.transpose();
        let j_ineq = eval.inequality_jacobian(&x);
        let jt_ineq = j_ineq.transpose();

        // bound slacks
        let s_l: Vec<f64> = lb_idx
            .iter()
            .zip(&lb_val)
            .map(|(&i, &lb)| (x[i] - lb).max(s_min))
            .collect();
        let s_u: Vec<f64> = ub_idx
            .iter()
            .zip(&ub_val)
            .map(|(&i, &ub)| (ub - x[i]).max(s_min))
            .collect();

        // ∇φ = w_f·∇f + Jeqᵀ(y + rho·c_eq)
        let al: Vec<f64> = c_eq.iter().zip(&y).map(|(&c, &yi)| yi + rho * c).collect();
        let jt_al = if n_eq > 0 {
            jt_eq.mul_vec(&al)
        } else {
            vec![0.0; n]
        };
        let grad_phi: Vec<f64> = (0..n)
            .map(|i| config.obj_weight * grad_f[i] + jt_al[i])
            .collect();

        // barrier Σ = z/s
        let sig_i: Vec<f64> = z_i.iter().zip(&s_i).map(|(&z, &s)| z / s).collect();
        let sig_l: Vec<f64> = z_l.iter().zip(&s_l).map(|(&z, &s)| z / s).collect();
        let sig_u: Vec<f64> = z_u.iter().zip(&s_u).map(|(&z, &s)| z / s).collect();
        let sig_l_full = scatter(&lb_idx, &sig_l);
        let sig_u_full = scatter(&ub_idx, &sig_u);

        // primal inequality residual r_ph = c_ineq + s_I
        let r_ph: Vec<f64> = c_ineq.iter().zip(&s_i).map(|(&c, &s)| c + s).collect();

        // --- diagnostics: dual & primal infeasibility, complementarity ---
        // r_d = ∇φ + Jineqᵀ z_I - z_L + z_U
        let mut r_d = grad_phi.clone();
        if m_i > 0 {
            let t = jt_ineq.mul_vec(&z_i);
            for (rd, &ti) in r_d.iter_mut().zip(&t) {
                *rd += ti;
            }
        }
        for (k, &i) in lb_idx.iter().enumerate() {
            r_d[i] -= z_l[k];
        }
        for (k, &i) in ub_idx.iter().enumerate() {
            r_d[i] += z_u[k];
        }
        let dual_inf = max_abs(&r_d);
        let primal_eq_inf = max_abs(&c_eq);
        let primal_ineq_inf = max_pos(&c_ineq);
        let comp = {
            let mut c = 0.0_f64;
            for (s, z) in s_i.iter().zip(&z_i) {
                c = c.max((s * z - mu).abs());
            }
            for (s, z) in s_l.iter().zip(&z_l) {
                c = c.max((s * z - mu).abs());
            }
            for (s, z) in s_u.iter().zip(&z_u) {
                c = c.max((s * z - mu).abs());
            }
            c
        };

        // track the best iterate (objective-aware among feasible points)
        let infeas = primal_eq_inf.max(primal_ineq_inf);
        if infeas <= config.constraint_tol {
            let obj = eval.objective(&x);
            if !have_feasible || obj < best_obj {
                best_obj = obj;
                x_best = x.clone();
                have_feasible = true;
            }
        } else if !have_feasible && infeas < best_infeas {
            best_infeas = infeas;
            x_best = x.clone();
        }

        // --- termination check ---
        // Optimal when the point is primal-feasible AND either (a) it is a full
        // KKT point (stationarity + complementarity within tol — the exact-QP
        // path) or (b) the barrier has annealed to its floor (the feasibility-
        // dominated collocation path, where the near-linear objective leaves
        // residual dual infeasibility that is not the acceptance criterion).
        let feasible =
            primal_eq_inf <= config.constraint_tol && primal_ineq_inf <= config.constraint_tol;
        let kkt_stationary = dual_inf <= config.opt_tol
            && comp <= config.opt_tol.max(mu)
            && mu <= config.mu_min * 10.0;
        // The barrier-floor acceptance is only for the equality-feasibility (AL)
        // path, where the near-linear objective leaves a large residual dual
        // infeasibility that is NOT the acceptance criterion. For pure
        // bound/inequality problems feasibility is trivial, so genuine
        // optimality must come from KKT stationarity.
        let barrier_annealed = n_eq > 0 && mu <= config.mu_min && last_rel_step <= config.opt_tol;
        let optimal = feasible && (kkt_stationary || barrier_annealed);
        if config.verbose > 0 {
            println!(
                "ipm {iter:4} | mu {mu:.2e} | eq {primal_eq_inf:.3e} | ineq {primal_ineq_inf:.3e} \
                 | dual {dual_inf:.3e} | comp {comp:.3e} | rho {rho:.1e}"
            );
        }
        if optimal {
            status = IpmStatus::Optimal;
            history.push(IpmLog {
                iter,
                mu,
                primal_eq_inf,
                primal_ineq_inf,
                dual_inf,
                comp,
                alpha_primal: 0.0,
                alpha_dual: 0.0,
                cg_iters: 0,
            });
            break;
        }

        // --- Fiacco-McCormick outer step ---
        // The *inner* (barrier) subproblem for fixed (mu, y, rho) is measured by
        // stationarity + complementarity + inequality-slack feasibility ONLY
        // (equality feasibility is the outer augmented-Lagrangian loop's job). If
        // it is solved, advance the schedule — reduce mu and update the equality
        // multipliers — and re-linearize on the next pass rather than taking a
        // (near-zero) Newton step. This is what lets an active-but-solved barrier
        // subproblem keep driving the equalities to zero at the mu floor.
        let e_mu = dual_inf.max(comp).max(primal_ineq_inf);
        // Advance the schedule when the inner subproblem is solved OR the inner
        // step budget at this mu is exhausted (the fallback that keeps mu
        // annealing when an active bound holds stationarity away from zero).
        if e_mu <= config.mu_solve_factor * mu || iters_at_mu >= config.inner_max_iters {
            if mu > config.mu_min {
                mu = (config.mu_reduction * mu).max(config.mu_min);
            }
            if n_eq > 0 {
                let eq_now = primal_eq_inf;
                if eq_now <= config.al_contract * eq_ref {
                    for (yi, &c) in y.iter_mut().zip(&c_eq) {
                        *yi += rho * c;
                    }
                    eq_ref = eq_now.max(1e-12);
                } else {
                    rho = (rho * config.rho_growth).min(config.rho_max);
                }
            }
            iters_at_mu = 0;
            history.push(IpmLog {
                iter,
                mu,
                primal_eq_inf,
                primal_ineq_inf,
                dual_inf,
                comp,
                alpha_primal: 0.0,
                alpha_dual: 0.0,
                cg_iters: 0,
            });
            continue;
        }

        // --- condensed RHS ---
        // rhs = -∇φ + [mu/s_L]·e_lb - [mu/s_U]·e_ub - Jineqᵀ(mu/s_I + Σ_I·r_ph)
        let mut rhs: Vec<f64> = grad_phi.iter().map(|&g| -g).collect();
        for (k, &i) in lb_idx.iter().enumerate() {
            rhs[i] += mu / s_l[k];
        }
        for (k, &i) in ub_idx.iter().enumerate() {
            rhs[i] -= mu / s_u[k];
        }
        if m_i > 0 {
            let tmp_i: Vec<f64> = (0..m_i).map(|j| mu / s_i[j] + sig_i[j] * r_ph[j]).collect();
            let contrib = jt_ineq.mul_vec(&tmp_i);
            for (r, &c) in rhs.iter_mut().zip(&contrib) {
                *r -= c;
            }
        }

        // --- condensed SPD operator (matrix-free) ---
        let apply = |v: &[f64]| -> Vec<f64> {
            // objective Hessian (default zero)
            let mut out = if config.obj_weight != 0.0 {
                let hv = eval.objective_hessian_vec(&x, v);
                hv.iter()
                    .map(|&h| config.obj_weight * h)
                    .collect::<Vec<_>>()
            } else {
                vec![0.0; n]
            };
            // rho·JeqᵀJeq·v
            if n_eq > 0 {
                let jv = j_eq.mul_vec(v);
                let jtjv = jt_eq.mul_vec(&jv);
                for (o, &t) in out.iter_mut().zip(&jtjv) {
                    *o += rho * t;
                }
            }
            // Jineqᵀ Σ_I Jineq·v
            if m_i > 0 {
                let jv = j_ineq.mul_vec(v);
                let sv: Vec<f64> = jv.iter().zip(&sig_i).map(|(&a, &s)| a * s).collect();
                let jtsv = jt_ineq.mul_vec(&sv);
                for (o, &t) in out.iter_mut().zip(&jtsv) {
                    *o += t;
                }
            }
            // diag(Σ_L + Σ_U) + reg
            for i in 0..n {
                out[i] += (sig_l_full[i] + sig_u_full[i] + config.reg) * v[i];
            }
            out
        };

        // --- Jacobi preconditioner: 1/diag(M) ---
        let mut diagm = vec![config.reg; n];
        for i in 0..n {
            diagm[i] += sig_l_full[i] + sig_u_full[i];
        }
        if n_eq > 0 {
            for row in 0..j_eq.nrows() {
                let (vals, cols) = j_eq.row_entries(row);
                for (&v, &c) in vals.iter().zip(cols) {
                    diagm[c] += rho * v * v;
                }
            }
        }
        if m_i > 0 {
            for (row, &sig) in sig_i.iter().enumerate() {
                let (vals, cols) = j_ineq.row_entries(row);
                for (&v, &c) in vals.iter().zip(cols) {
                    diagm[c] += sig * v * v;
                }
            }
        }
        let minv: Vec<f64> = diagm.iter().map(|&d| 1.0 / d.max(1e-30)).collect();

        let (dx, cg_iters) = pcg(apply, &rhs, &minv, config.cg_max_iter, config.cg_tol);

        // --- recover slack / multiplier steps ---
        let jineq_dx = if m_i > 0 { j_ineq.mul_vec(&dx) } else { vec![] };
        let ds_i: Vec<f64> = (0..m_i).map(|j| -r_ph[j] - jineq_dx[j]).collect();
        // dz = -z + mu/s - Σ·ds
        let dz_i: Vec<f64> = (0..m_i)
            .map(|j| -z_i[j] + mu / s_i[j] - sig_i[j] * ds_i[j])
            .collect();
        let dz_l: Vec<f64> = (0..lb_idx.len())
            .map(|k| {
                let ds = dx[lb_idx[k]];
                -z_l[k] + mu / s_l[k] - sig_l[k] * ds
            })
            .collect();
        let dz_u: Vec<f64> = (0..ub_idx.len())
            .map(|k| {
                let ds = -dx[ub_idx[k]];
                -z_u[k] + mu / s_u[k] - sig_u[k] * ds
            })
            .collect();

        // --- fraction-to-the-boundary step lengths ---
        let tau = (1.0 - mu).max(config.tau_min).min(0.9995);
        let ftb = |s: &[f64], ds: &[f64]| -> f64 {
            let mut a = 1.0_f64;
            for (&si, &dsi) in s.iter().zip(ds) {
                if dsi < 0.0 {
                    a = a.min(-tau * si / dsi);
                }
            }
            a
        };
        // primal slacks: s_L step = dx[lb], s_U step = -dx[ub], s_I step = ds_i
        let ds_l: Vec<f64> = lb_idx.iter().map(|&i| dx[i]).collect();
        let ds_u: Vec<f64> = ub_idx.iter().map(|&i| -dx[i]).collect();
        let alpha_p = ftb(&s_l, &ds_l).min(ftb(&s_u, &ds_u)).min(ftb(&s_i, &ds_i));
        let alpha_d = ftb(&z_l, &dz_l).min(ftb(&z_u, &dz_u)).min(ftb(&z_i, &dz_i));

        // --- line search on the barrier-augmented merit ---
        let nu = max_abs(&z_i).max(1.0) + 1.0;
        let merit = |xt: &[f64], sit: &[f64]| -> f64 {
            // barrier terms; +inf if any slack non-positive.
            let mut barrier = 0.0;
            for (k, &i) in lb_idx.iter().enumerate() {
                let s = xt[i] - lb_val[k];
                if s <= 0.0 {
                    return f64::INFINITY;
                }
                barrier += s.ln();
            }
            for (k, &i) in ub_idx.iter().enumerate() {
                let s = ub_val[k] - xt[i];
                if s <= 0.0 {
                    return f64::INFINITY;
                }
                barrier += s.ln();
            }
            for &s in sit {
                if s <= 0.0 {
                    return f64::INFINITY;
                }
                barrier += s.ln();
            }
            let ce = eval.equality_constraints(xt);
            let ci = eval.inequality_constraints(xt);
            let phi =
                config.obj_weight * eval.objective(xt) + dot(&y, &ce) + 0.5 * rho * dot(&ce, &ce);
            let feas_pen: f64 = ci.iter().zip(sit).map(|(&c, &s)| (c + s).abs()).sum();
            phi - mu * barrier + nu * feas_pen
        };

        let merit0 = merit(&x, &s_i);
        let mut alpha = alpha_p;
        let mut x_new = x.clone();
        let mut s_i_new = s_i.clone();
        let mut accepted = false;
        for _ in 0..40 {
            for i in 0..n {
                x_new[i] = x[i] + alpha * dx[i];
            }
            for j in 0..m_i {
                s_i_new[j] = s_i[j] + alpha * ds_i[j];
            }
            if merit(&x_new, &s_i_new) < merit0 {
                accepted = true;
                break;
            }
            alpha *= 0.5;
            if alpha < 1e-14 {
                break;
            }
        }

        if !accepted {
            // A genuine stall: the inner subproblem was NOT flagged solved above,
            // yet no merit-decreasing step exists. If we are essentially feasible,
            // accept the current point as optimal; if the equalities are badly
            // violated with the penalty maxed, declare infeasibility; else flag
            // the line-search failure.
            let feasible =
                primal_eq_inf <= config.constraint_tol && primal_ineq_inf <= config.constraint_tol;
            status = if feasible {
                IpmStatus::Optimal
            } else if rho >= config.rho_max {
                IpmStatus::InfeasibleDetected
            } else {
                IpmStatus::LineSearchFailure
            };
            history.push(IpmLog {
                iter,
                mu,
                primal_eq_inf,
                primal_ineq_inf,
                dual_inf,
                comp,
                alpha_primal: 0.0,
                alpha_dual: alpha_d,
                cg_iters,
            });
            break;
        }

        // --- apply steps ---
        iters_at_mu += 1;
        // Record the relative primal step (scaled space) for the barrier-floor
        // acceptance gate above: max |Δx| over max(|x|, 1).
        {
            let mut step_inf = 0.0_f64;
            for &d in &dx {
                step_inf = step_inf.max((alpha * d).abs());
            }
            last_rel_step = step_inf / max_abs(&x).max(1.0);
        }
        x = x_new;
        for j in 0..m_i {
            s_i[j] = s_i_new[j].max(s_min);
        }
        for k in 0..lb_idx.len() {
            z_l[k] = (z_l[k] + alpha_d * dz_l[k]).max(1e-12);
        }
        for k in 0..ub_idx.len() {
            z_u[k] = (z_u[k] + alpha_d * dz_u[k]).max(1e-12);
        }
        for j in 0..m_i {
            z_i[j] = (z_i[j] + alpha_d * dz_i[j]).max(1e-12);
        }

        history.push(IpmLog {
            iter,
            mu,
            primal_eq_inf,
            primal_ineq_inf,
            dual_inf,
            comp,
            alpha_primal: alpha,
            alpha_dual: alpha_d,
            cg_iters,
        });
    }

    // Return the best iterate seen (objective-aware among feasible points; see
    // the `x_best` tracking above). If the stopping iterate is itself feasible
    // and improves the objective, fold it in.
    let final_infeas = {
        let ce = eval.equality_constraints(&x);
        let ci = eval.inequality_constraints(&x);
        max_abs(&ce).max(max_pos(&ci))
    };
    let x = if final_infeas <= config.constraint_tol {
        let obj = eval.objective(&x);
        if !have_feasible || obj < best_obj {
            x
        } else {
            x_best
        }
    } else if have_feasible {
        x_best
    } else if final_infeas <= best_infeas {
        x
    } else {
        x_best
    };

    // Final objective in this (possibly scaled) space; solve_ipm overwrites it
    // with the SI value.
    let objective = eval.objective(&x);
    let c_eq = eval.equality_constraints(&x);
    let c_ineq = eval.inequality_constraints(&x);
    IpmResult {
        x,
        objective,
        eq_violation: max_abs(&c_eq),
        ineq_violation: max_pos(&c_ineq),
        iterations,
        converged: status == IpmStatus::Optimal,
        status,
        history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_math::{CsrBuilder, CsrMatrix};

    // ---------- Analytic bound-constrained QP ----------
    //
    // min ½‖x - a‖²  s.t. lb <= x <= ub. KKT solution is the clamp of `a` into
    // the box, with multipliers on the active bounds. We exercise components on
    // and off their bounds.
    struct BoxQp {
        a: Vec<f64>,
    }
    impl NlpEvaluator for BoxQp {
        fn objective(&self, x: &[f64]) -> f64 {
            0.5 * x
                .iter()
                .zip(&self.a)
                .map(|(&xi, &ai)| (xi - ai).powi(2))
                .sum::<f64>()
        }
        fn objective_gradient(&self, x: &[f64]) -> Vec<f64> {
            x.iter().zip(&self.a).map(|(&xi, &ai)| xi - ai).collect()
        }
        fn equality_constraints(&self, _x: &[f64]) -> Vec<f64> {
            vec![]
        }
        fn inequality_constraints(&self, _x: &[f64]) -> Vec<f64> {
            vec![]
        }
        fn equality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            CsrMatrix::zeros(0, self.a.len())
        }
        fn inequality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            CsrMatrix::zeros(0, self.a.len())
        }
        fn objective_hessian_vec(&self, _x: &[f64], v: &[f64]) -> Vec<f64> {
            v.to_vec() // H = I
        }
    }

    #[test]
    fn qp_box_constrained_active_bounds() {
        // a = [2, -3, 0.5]; box [0,1] on each. Solution = clamp(a) = [1, 0, 0.5].
        let problem = NlpProblem {
            n_vars: 3,
            n_eq: 0,
            n_ineq: 0,
            lower_bounds: vec![0.0; 3],
            upper_bounds: vec![1.0; 3],
        };
        let eval = BoxQp {
            a: vec![2.0, -3.0, 0.5],
        };
        let res = solve_ipm(&problem, &eval, &[0.5, 0.5, 0.5], &IpmConfig::default());
        assert_eq!(
            res.status,
            IpmStatus::Optimal,
            "history: {:?}",
            res.history.last()
        );
        let want = [1.0, 0.0, 0.5];
        for (i, (&got, &w)) in res.x.iter().zip(&want).enumerate() {
            assert!((got - w).abs() < 1e-6, "x[{i}]={got} want {w}");
        }
    }

    // ---------- QP with a linear inequality active at the solution ----------
    //
    // min ½‖x‖²  s.t.  x0 + x1 <= -2  (i.e. c_ineq = x0 + x1 + 2 <= 0).
    // Solution: project origin onto the halfspace -> x = (-1, -1), active.
    struct HalfspaceQp;
    impl NlpEvaluator for HalfspaceQp {
        fn objective(&self, x: &[f64]) -> f64 {
            0.5 * (x[0] * x[0] + x[1] * x[1])
        }
        fn objective_gradient(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0], x[1]]
        }
        fn equality_constraints(&self, _x: &[f64]) -> Vec<f64> {
            vec![]
        }
        fn inequality_constraints(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] + x[1] + 2.0]
        }
        fn equality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            CsrMatrix::zeros(0, 2)
        }
        fn inequality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            let mut b = CsrBuilder::new(1, 2);
            b.add(0, 0, 1.0);
            b.add(0, 1, 1.0);
            b.build()
        }
        fn objective_hessian_vec(&self, _x: &[f64], v: &[f64]) -> Vec<f64> {
            v.to_vec()
        }
    }

    #[test]
    fn qp_active_inequality() {
        let problem = NlpProblem {
            n_vars: 2,
            n_eq: 0,
            n_ineq: 1,
            lower_bounds: vec![f64::NEG_INFINITY; 2],
            upper_bounds: vec![f64::INFINITY; 2],
        };
        let res = solve_ipm(&problem, &HalfspaceQp, &[0.0, 0.0], &IpmConfig::default());
        assert_eq!(res.status, IpmStatus::Optimal);
        assert!((res.x[0] + 1.0).abs() < 1e-5, "x0 {}", res.x[0]);
        assert!((res.x[1] + 1.0).abs() < 1e-5, "x1 {}", res.x[1]);
        assert!(res.ineq_violation < 1e-6);
    }

    // ---------- Equality-constrained QP ----------
    //
    // min ½‖x‖²  s.t.  x0 + x1 = 2. Solution x = (1, 1), y = -1.
    struct EqQp;
    impl NlpEvaluator for EqQp {
        fn objective(&self, x: &[f64]) -> f64 {
            0.5 * (x[0] * x[0] + x[1] * x[1])
        }
        fn objective_gradient(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0], x[1]]
        }
        fn equality_constraints(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] + x[1] - 2.0]
        }
        fn inequality_constraints(&self, _x: &[f64]) -> Vec<f64> {
            vec![]
        }
        fn equality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            let mut b = CsrBuilder::new(1, 2);
            b.add(0, 0, 1.0);
            b.add(0, 1, 1.0);
            b.build()
        }
        fn inequality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            CsrMatrix::zeros(0, 2)
        }
        fn objective_hessian_vec(&self, _x: &[f64], v: &[f64]) -> Vec<f64> {
            v.to_vec()
        }
    }

    #[test]
    fn qp_equality_constrained() {
        let problem = NlpProblem {
            n_vars: 2,
            n_eq: 1,
            n_ineq: 0,
            lower_bounds: vec![f64::NEG_INFINITY; 2],
            upper_bounds: vec![f64::INFINITY; 2],
        };
        let res = solve_ipm(&problem, &EqQp, &[0.0, 0.0], &IpmConfig::default());
        assert!(res.eq_violation < 1e-6, "eq_violation {}", res.eq_violation);
        assert!((res.x[0] - 1.0).abs() < 1e-5, "x0 {}", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 1e-5, "x1 {}", res.x[1]);
    }

    // ---------- Bound-constrained Rosenbrock ----------
    //
    // min 100(x1 - x0²)² + (1 - x0)²  s.t. x <= 0.5 (both). The unconstrained
    // min (1,1) is cut off; the solution sits on the boundary x0 = 0.5.
    struct Rosenbrock;
    impl Rosenbrock {
        // residuals r = [10(x1 - x0²), (1 - x0)]; f = ‖r‖².
        fn resid(x: &[f64]) -> [f64; 2] {
            [10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]]
        }
        fn jac(x: &[f64]) -> [[f64; 2]; 2] {
            [[-20.0 * x[0], 10.0], [-1.0, 0.0]]
        }
    }
    impl NlpEvaluator for Rosenbrock {
        fn objective(&self, x: &[f64]) -> f64 {
            let r = Rosenbrock::resid(x);
            r[0] * r[0] + r[1] * r[1]
        }
        fn objective_gradient(&self, x: &[f64]) -> Vec<f64> {
            let r = Rosenbrock::resid(x);
            let j = Rosenbrock::jac(x);
            // ∇f = 2 Jᵀ r
            vec![
                2.0 * (j[0][0] * r[0] + j[1][0] * r[1]),
                2.0 * (j[0][1] * r[0] + j[1][1] * r[1]),
            ]
        }
        fn equality_constraints(&self, _x: &[f64]) -> Vec<f64> {
            vec![]
        }
        fn inequality_constraints(&self, _x: &[f64]) -> Vec<f64> {
            vec![]
        }
        fn equality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            CsrMatrix::zeros(0, 2)
        }
        fn inequality_jacobian(&self, _x: &[f64]) -> CsrMatrix {
            CsrMatrix::zeros(0, 2)
        }
        fn objective_hessian_vec(&self, x: &[f64], v: &[f64]) -> Vec<f64> {
            // Gauss-Newton Hessian 2 JᵀJ (PSD) — matrix-free product.
            let j = Rosenbrock::jac(x);
            // Jv (2-vector)
            let jv = [
                j[0][0] * v[0] + j[0][1] * v[1],
                j[1][0] * v[0] + j[1][1] * v[1],
            ];
            vec![
                2.0 * (j[0][0] * jv[0] + j[1][0] * jv[1]),
                2.0 * (j[0][1] * jv[0] + j[1][1] * jv[1]),
            ]
        }
    }

    #[test]
    fn rosenbrock_bound_constrained() {
        let problem = NlpProblem {
            n_vars: 2,
            n_eq: 0,
            n_ineq: 0,
            lower_bounds: vec![-2.0, -2.0],
            upper_bounds: vec![0.5, 0.5],
        };
        let cfg = IpmConfig {
            max_iterations: 400,
            ..IpmConfig::default()
        };
        let res = solve_ipm(&problem, &Rosenbrock, &[-1.0, -1.0], &cfg);
        // Known boundary solution: x0 = 0.5, x1 = 0.25 (x1 = x0² makes the first
        // residual zero; x0 pinned at its upper bound by the (1-x0)² term).
        assert!(
            res.x[0] > 0.5 - 1e-4 && res.x[0] <= 0.5 + 1e-9,
            "x0 {}",
            res.x[0]
        );
        assert!((res.x[1] - 0.25).abs() < 1e-3, "x1 {}", res.x[1]);
    }

    // ---------- Determinism ----------
    #[test]
    fn determinism_bitwise_history() {
        let problem = NlpProblem {
            n_vars: 3,
            n_eq: 0,
            n_ineq: 0,
            lower_bounds: vec![0.0; 3],
            upper_bounds: vec![1.0; 3],
        };
        let eval = BoxQp {
            a: vec![2.0, -3.0, 0.5],
        };
        let r1 = solve_ipm(&problem, &eval, &[0.5, 0.5, 0.5], &IpmConfig::default());
        let r2 = solve_ipm(&problem, &eval, &[0.5, 0.5, 0.5], &IpmConfig::default());
        assert_eq!(r1.history.len(), r2.history.len());
        for (a, b) in r1.history.iter().zip(&r2.history) {
            assert_eq!(a.mu.to_bits(), b.mu.to_bits(), "mu differs at {}", a.iter);
            assert_eq!(a.primal_eq_inf.to_bits(), b.primal_eq_inf.to_bits());
            assert_eq!(a.dual_inf.to_bits(), b.dual_inf.to_bits());
            assert_eq!(a.alpha_primal.to_bits(), b.alpha_primal.to_bits());
            assert_eq!(a.cg_iters, b.cg_iters);
        }
        for (a, b) in r1.x.iter().zip(&r2.x) {
            assert_eq!(a.to_bits(), b.to_bits(), "x differs");
        }
    }
}
