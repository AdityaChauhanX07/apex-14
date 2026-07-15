# Bound-capable interior-point NLP solver

**Status: implemented.** A primal-dual interior-point solver in
`apex-optimizer::ipm` for `min f(x)  s.t.  c_eq(x)=0, c_ineq(x)<=0, lb<=x<=ub`,
behind the existing [`NlpEvaluator`] interface. It is the solver the envelope
free-trajectory OCP will run on, and it resolves the documented Gauss-Newton
**bound deadlock** (`docs/design/gn-solver-bound-deadlock.md`). Per the scope
decision (`recon.md` §1.4, §7): pure Rust, **wasm-safe**, **no new
linear-algebra dependencies** — the Newton system is condensed to a symmetric
positive-definite operator and solved matrix-free with the same conjugate-
gradient machinery the Gauss-Newton solver already uses.

IP is **opt-in** (`apex-14 optimize --solver ip`, or
`CollocationOptimizer::optimize_ip`). The default CLI/optimizer paths are
unchanged and the goldens stay on the current default solver.

## 1. Why the deadlock happens, and why a barrier fixes it

`optimize_gn` computes an unconstrained Newton step and enforces variable bounds
by **post-hoc projection**. When the optimum needs a bound to bind — `f_drive`
saturating `max_drive_force` on the oval's straights — the raw step points past
the bound, projection clips it straight back, and the net displacement is
numerically zero: a projected-Newton deadlock (25–28 of ~349 variables pinned).
It floors at `eq_violation ≈ 0.3–0.7`, orders above tolerance.

An interior-point method never projects. Each bound `lb <= x` becomes a slack
`s_L = x - lb > 0` with a **dual multiplier** `z_L > 0` and a perturbed
complementarity `s_L·z_L = mu`. An active bound is simply `s_L → 0`, `z_L → `
(the correct implicit multiplier); the barrier keeps iterates interior and the
`mu → 0` schedule lets them approach the bound smoothly. No step is ever
cancelled.

## 2. Method

Interface: the existing [`NlpEvaluator`] trait, extended with **one** optional
method — `objective_hessian_vec(x, v) -> H_f·v`, defaulting to the zero vector.
The product form keeps everything matrix-free; the default (zero) means a linear
objective, so all existing evaluators (collocation, 7-/14-DOF) compile and behave
unchanged. A genuinely nonlinear objective (a QP, a least-squares residual)
overrides it, optionally with a Gauss-Newton (PSD) approximation.

**Inequalities & bounds → primal-dual barrier.** Each `c_ineq_j <= 0` gets a
slack `s_I_j > 0` with `c_ineq + s_I = 0` and multiplier `z_I > 0`; each finite
bound gives an affine slack `s_L = x - lb` / `s_U = ub - x` with `z_L, z_U > 0`.
Perturbed complementarity `s·z = mu`.

**Equalities → augmented Lagrangian.** `c_eq` enters the merit as
`y·c_eq + (rho/2)‖c_eq‖²`. Between barrier subproblems the multiplier `y` is
updated (`y += rho·c_eq` when `‖c_eq‖` dropped enough, else `rho` is raised),
so `c_eq → 0` without needing `rho → ∞`. This is what nails the dynamics defects
to `1e-6` on the deadlock.

**Hessian model.** The Lagrangian Hessian is modeled as
`H = w_f·H_f + rho·JeqᵀJeq + Jineqᵀ Σ_I Jineq + diag(Σ_L + Σ_U) + reg·I`, with
`Σ = z/s` (the primal-dual barrier diagonals). The `Jeq` term is the
Gauss-Newton curvature of the equality least-squares (the same approximation the
GN solver relies on; no constraint Hessians are required). Every term is PSD and
the barrier + `reg` make `H` SPD.

**Merit / line search.** A backtracking line search on the barrier-augmented
merit `w_f·f + y·c_eq + (rho/2)‖c_eq‖² - mu·Σln(s) + nu·‖c_ineq + s_I‖₁`, accepted
on simple sufficient decrease (the style the GN solver already uses). Chosen over
a filter for simplicity and because the condensed direction is a descent
direction of this merit (the operator is SPD). The fraction-to-the-boundary rule
(`tau = max(tau_min, 1-mu)`) caps the step so all slacks and multipliers stay
strictly positive.

**Barrier schedule.** Monotone Fiacco-McCormick: reduce `mu` by a fixed factor
(`0.2`) once the *inner* subproblem is solved to `kappa_eps·mu`, measured by
stationarity + complementarity + inequality-slack feasibility **only** (equality
feasibility is the outer AL loop's job). A **fallback** reduces `mu` after a
bounded number of inner Newton steps even if that gate is not met — essential
here, because an active bound holds inner stationarity away from zero at moderate
`mu`, and feasibility can only be reached by annealing `mu` down so the bound can
be approached. Mehrotra predictor-corrector / adaptive `mu` are noted as future
work.

## 3. Condensation (the matrix-free SPD system)

Eliminating the slack and bound/inequality multiplier steps from the perturbed
KKT system analytically leaves a single SPD system for the primal step `dx`:

```
[ w_f·H_f + rho·JeqᵀJeq + Jineqᵀ Σ_I Jineq + diag(Σ_L+Σ_U) + reg·I ] dx = rhs
```

solved by **preconditioned conjugate gradient** with a Jacobi (diagonal)
preconditioner — the operator is applied purely as sparse mat-vecs
(`Jeq·v`, `Jeqᵀ·(...)`, `Jineq·v`, `Jineqᵀ·(...)`) plus the objective
Hessian-vector product, exactly the pattern in `gauss_newton::conjugate_gradient`
(now generalized to carry a barrier-weighted diagonal instead of a flat `reg·I`,
which is precisely what `recon.md` §1.4 anticipated). The slack/multiplier steps
`ds_I, dz_I, dz_L, dz_U` are recovered by back-substitution. No factorization, no
indefinite solve, no new dependency.

**CG behavior.** On the analytic QPs CG converges in a handful of iterations
(the operator is tiny). On the collocation deadlock (≈349 vars, N=50), the Jacobi
preconditioner keeps inner CG typically in the low-tens of iterations even as the
barrier diagonal spans many orders of magnitude near active bounds — the
preconditioner captures the large `Σ = z/s` entries, so the conditioning trouble
that a flat `reg·I` would suffer is absorbed. No CG breakdown was observed.

## 4. Scaling

Per `docs/design/nlp-scaling.md`, variable scaling is applied **inside** the
solver so callers pass unscaled problems: a **guarded Jacobi column scaling** is
measured once at `x0` (`x_scale[j] = 1/‖[Jeq;Jineq][:,j]‖`, or `1.0` for a
variable no constraint touches — so a pure-objective QP variable is left
unscaled rather than blown up by the floor). The whole solve runs in scaled space
via `ScaledEvaluator` (which now also chain-rule-scales the objective
Hessian-vector product), then the solution is unscaled and `eq_violation` /
`ineq_violation` are recomputed in SI. Constraint (row) scales are held at `1.0`
(column-only), so `constraint_tol` and reported violations stay in SI by
construction — the same discipline the GN path adopted.

## 5. Determinism & diagnostics

Fixed iteration order, no RNG, sequential reductions, deterministic CG →
bitwise-identical iterate history for identical inputs (unit test
`determinism_bitwise_history`; collocation-scale test
`ip_collocation_determinism_bitwise`). Diagnostics: an [`IpmLog`] per iteration
(`mu`, primal-eq / primal-ineq infeasibility, dual infeasibility, complementarity,
primal/dual step lengths, CG iterations) behind a `verbose` flag, and a terminal
[`IpmStatus`] enum — `Optimal` / `MaxIter` / `InfeasibleDetected` /
`LineSearchFailure` — so there is no silent failure.

## 6. Deadlock result

Configuration: oval (`oval_track(1000, 100, 12, 400)`), Hermite-Simpson, N=50,
`f1_2024_calibrated` car — the documented deadlock (`f_drive` pins at
`max_drive_force` on both straights).

| Solver | Final `eq_violation` | Converged |
|---|---|---|
| Gauss-Newton (100 iters) | `3.40e-1` | no (deadlock) |
| **Interior-point** | **`1.08e-7`** | **yes** |

- Iterations to convergence: ≈110 outer iterations.
- **Wall time: ≈110 ms** (release, dev machine), returning the best feasible
  iterate.

Test: `ip_resolves_gn_bound_deadlock` (asserts `eq_violation <= 1e-6`).
On the circle (where GN already converges), `ip_matches_gn_on_circle` confirms
the IP lap time agrees with GN and QSS.

## 7. Limitations

- **Gauss-Newton objective/constraint curvature.** No constraint Hessians are
  used; the equality curvature is the GN term `rho·JeqᵀJeq`. This is exact for
  feasibility-dominated problems (a feasible point drives the least-squares term
  to zero) and matches how the working GN solver models the problem, but for a
  problem where equalities and a strongly nonlinear objective genuinely trade off,
  the GN model is approximate. A user-supplied constraint Hessian-vector product
  would remove this — future work.
- **Monotone barrier only.** No Mehrotra predictor-corrector / adaptive `mu`;
  convergence is robust but not asymptotically superlinear.
- **Feasibility-first acceptance on the AL path.** For the collocation solve the
  near-linear objective leaves residual dual infeasibility inflated by the large
  AL penalty; the acceptance criterion there is primal feasibility at the barrier
  floor (best feasible iterate returned), not full KKT stationarity of the lap
  time. Pure bound/inequality problems (the QPs) still terminate on genuine KKT
  stationarity.
- **`rho` can ramp high** (up to `1e8`) on badly nonlinear feasibility problems;
  this inflates the reported dual infeasibility (not the acceptance metric) and
  is cosmetic. A gentler penalty schedule is possible future tuning.

## 8. Blast radius

- New `crates/apex-optimizer/src/ipm.rs`; `pub mod ipm;` + re-exports in
  `lib.rs`.
- `nlp.rs`: one **defaulted** trait method (`objective_hessian_vec`) — additive,
  breaks no existing impl.
- `scaling.rs`: `ScaledEvaluator` forwards the new method with chain-rule scaling.
- `collocation.rs`: `optimize_ip` / `optimize_ip_from` / `extract_result_ip` —
  additive; `optimize_gn` and all other paths untouched.
- CLI: `optimize --solver {gn|ip}` (default `gn`).
- **Goldens unchanged** — no default path was modified.
