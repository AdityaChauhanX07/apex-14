# Design note: variable scaling for the collocation NLP

Status: **ADOPTED** (conditioning fix only — see "What this does and does not
fix" below). Implemented in `crates/apex-optimizer/src/scaling.rs` and
`crates/apex-optimizer/src/collocation.rs` (`CollocationOptimizer::optimize_gn`,
`build_scaling`, `jacobi_scale`).

This note originally proposed a fix for the Gauss-Newton convergence failure
characterized in the Phase 0.1 slice-3 diagnosis (oval and
`random_spline_track(seed=42)` fail to converge at N=50; violation *worsens*
50→400 nodes; `circle_track` converges cleanly to `7.9e-6`). Root cause:
raw-SI decision variables span several orders of magnitude in the Jacobian,
fed into a single flat `regularization = 1e-4` in `(JᵀJ + reg·I)`. The
originally-proposed fix (a per-block physical-reference-value heuristic) was
implemented, measured, and disproven — see "Superseded approach" below. What
shipped instead is Jacobi (diagonal) preconditioning, described in A2.

## A1 — Abstraction & placement (unchanged from the original proposal)

The trait/struct boundary (`crates/apex-optimizer/src/nlp.rs`, not modified
by this work):

```rust
pub struct NlpProblem {
    pub n_vars: usize,
    pub n_eq: usize,
    pub n_ineq: usize,
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
}

pub trait NlpEvaluator {
    fn objective(&self, x: &[f64]) -> f64;
    fn objective_gradient(&self, x: &[f64]) -> Vec<f64>;
    fn equality_constraints(&self, x: &[f64]) -> Vec<f64>;
    fn inequality_constraints(&self, x: &[f64]) -> Vec<f64>;
    fn equality_jacobian(&self, x: &[f64]) -> apex_math::CsrMatrix;
    fn inequality_jacobian(&self, x: &[f64]) -> apex_math::CsrMatrix;
}
```

Both `solve_gauss_newton` (`gauss_newton.rs`) and the existing `solve_nlp`
(`solver.rs`) consume nothing but `&impl NlpEvaluator` + `&NlpProblem`.
Neither file was modified. Scaling is a decorator that produces another
`NlpEvaluator`, in the new `crates/apex-optimizer/src/scaling.rs`:

```rust
/// Diagonal (per-component) reference scales. `x_si = x_scale[i] * x_scaled[i]`.
pub struct Scaling {
    pub x_scale: Vec<f64>,      // len n_vars
    pub c_eq_scale: Vec<f64>,   // len n_eq   — held at 1.0 (see A2)
    pub c_ineq_scale: Vec<f64>, // len n_ineq — held at 1.0 (see A2)
}

impl Scaling {
    pub fn scale_x(&self, x_si: &[f64]) -> Vec<f64> { /* x_si[i] / x_scale[i] */ }
    pub fn unscale_x(&self, x_scaled: &[f64]) -> Vec<f64> { /* x_scaled[i] * x_scale[i] */ }
    pub fn scale_problem(&self, p: &NlpProblem) -> NlpProblem { /* divide bounds by x_scale */ }
}

pub struct ScaledEvaluator<'a, E: NlpEvaluator> {
    pub inner: &'a E,
    pub scaling: &'a Scaling,
}
// impl<E: NlpEvaluator> NlpEvaluator for ScaledEvaluator<'_, E> — chain-rule
// scaling of objective/gradient/constraints/Jacobian; unchanged since the
// original proposal.
```

`CollocationOptimizer::optimize_gn` is the only call site: it builds a
`Scaling` from the QSS warmstart via `build_scaling`/`jacobi_scale`, wraps
the evaluator/problem/`x0`, runs `solve_gauss_newton` unchanged, then
unscales the result and **recomputes** `eq_violation`/`ineq_violation`
against the unscaled evaluator before returning `OptimizationResult` (SI at
the boundary — see A6).

## A2 — Reference scales: Jacobi (diagonal) preconditioning, adopted

**Superseded approach.** The original version of this note proposed a static
per-block scale keyed to each variable's raw *physical* magnitude — `s` by
track length (`≈1503` m), `n` by half-width, `v` by warmstart top speed,
`curv` by max track curvature, `f_drive` by max drive/brake force, `dt` by
expected time step, `alpha` left at `1.0`. This was implemented and measured
directly: it drove the `s` column's contribution to `diag(JᵀJ)` from a raw,
already-fine value of `√2 ≈ 1.41` (structural — `s` appears with a ±1
coefficient in exactly two dynamics-defect rows, from the `state_k1[0] -
state_k[0]` term, independent of the track's physical length) up to `2114`
after scaling — a ~1500× *over*-correction, because the heuristic conflated
"how large is this variable's raw value" with "how large is this variable's
actual Jacobian sensitivity," which are unrelated for `s`. This broke 5
previously-passing tests (`gn_circle_collocation`,
`hs_optimization_circle_converges`, `hs_lower_defect_than_trapezoidal_on_oval`,
`mesh_refinement::refinement_on_circle`,
`mesh_refinement::refinement_beats_cold_start_on_oval`) and was replaced,
not patched, by the approach below.

**What shipped: Jacobi/diagonal preconditioning, measured at the warmstart.**
For each decision-variable column `j`, scale by the reciprocal of that
column's actual measured Jacobian norm in the equality Jacobian, evaluated
once at the QSS warmstart `x0`:

```
x_scale[j] = 1 / max(‖J_eq[:, j]‖, floor)      (floor = 1e-6, unchanged from the original proposal)
```

(`crates/apex-optimizer/src/collocation.rs`, `jacobi_scale`). This drives
every scaled column's contribution to `diag(JᵀJ)` to *exactly* `1.0` by
construction (`diag(JᵀJ)_scaled_j = ‖J[:,j]‖² · x_scale[j]² = 1`), rather than
to "roughly comparable" as the physical-heuristic table aimed for. Measured
on the default oval, calibrated car, N=50 — the same case where the old
heuristic produced a `1.40`–`2114` (3.2 orders of magnitude) scaled spread —
Jacobi scaling produces exactly `1.0` for all seven blocks (`s`, `n`, `v`,
`alpha`, `f_drive`, `curv`, `dt`), with none hitting the `1e-6` floor at this
measurement point (every block has genuine, non-degenerate sensitivity — see
the implementation report for the full per-block table).

The reference point is **static**: measured once at the QSS warmstart, a
pure function of `(car, track, n_nodes, method)`, then frozen for the entire
solve — never updated during iteration. An adaptive variant (re-measuring at
a few early iterations before freezing) was also implemented and measured;
its convergence gain over the static warmstart measurement was marginal to
negative on the cases that matter (oval, `random_spline_track`) while adding
a real new knob (a re-measurement schedule) and tripling early-iteration
Jacobian-evaluation cost. The static warmstart measurement was kept on
reproducibility grounds.

**Equality/inequality constraint scales stay at `1.0` (column-only
scaling).** This was already true under the physical-heuristic version's
forced deviation and is unchanged: scaling constraint *values* (not just
variables) would make `constraint_tol` inside `solve_gauss_newton`'s own
feasibility check compare against a residual shrunk by that block's
reference scale, silently changing what `constraint_tol` means for the
solver's own termination decision. Column-only scaling (the standard
variable-scaled Gauss-Newton/Levenberg-Marquardt reformulation — see e.g.
MINPACK's `diag` parameter) avoids this: because constraint values are never
rescaled, `constraint_tol` and any reported `eq_violation`/`ineq_violation`
are *identical* to what the unscaled evaluator would report — provably
equal, not merely close — so the SI-boundary requirement holds by
construction. `Scaling`/`ScaledEvaluator` still support full row scaling
generically, for a future solver where this tradeoff may not apply; the
collocation call site does not use it.

## A3 — Invertibility & neutrality (confirmed, not just argued)

The invertibility argument is unchanged: `unscale_x(scale_x(x)) == x` up to
IEEE-754 rounding (~1e-12 relative), since `x_scale` is a fixed, positive,
finite, iterate-independent vector. This is now backed by passing tests:
`scaling::tests::round_trip_unscale_scale_is_identity` and
`collocation::tests::scaling_round_trip_matches_warmstart_within_1e12` (the
latter using a real oval warmstart vector).

Neutrality was measured directly on `circle_track(100.0, 12.0, 200)`, N=50,
calibrated car (the case that already converged cleanly before any scaling
work): lap time `11.494986` s post-scaling vs. `11.495100` s pre-scaling —
**delta = 1.14e-4 s**. This is roughly 23× tighter than the abandoned
physical-heuristic attempt's delta (`2.59e-3` s) and well within the
golden-lap harness's `0.010` s tolerance. It is not exactly `0` because
pre- and post-scaling runs stop at different (both tiny) feasibility levels
along mathematically-equivalent-but-not-bit-identical iterative paths — the
underlying claim (a change of variables cannot move the physical optimum)
holds; two different stopping iterations landing on identical floats was
always an unrealistic bar.

## A4 — Regularization interaction

`regularization = 1e-4` was **not changed**, per the original recommendation.
Jacobi scaling collapses `diag(JᵀJ)` to exactly `1.0` for every column
(stronger than the "roughly O(1)" the physical-heuristic table aimed for),
which is precisely the regime a single flat regularization constant suits.

`constraint_tol` also required no reinterpretation, unlike the original
proposal's open concern: because column-only scaling never touches
constraint values, `constraint_tol = 1e-4` means exactly the same thing —
"defect satisfied to `1e-4` in SI units" — before and after scaling, for
both the solver's own internal termination check and any reported
violation. This is proven, not just argued, by
`collocation::tests::build_scaling_leaves_constraints_unscaled` (asserts
`c_eq_scale`/`c_ineq_scale` are literally `1.0`) and
`collocation::tests::optimize_gn_reports_si_violation_not_scaled`
(independently recomputes the reported violation via the public API and
asserts equality to `<1e-9`).

## What this does and does not fix

**Fixed (conditioning):**
- `diag(JᵀJ)` column spread: `1.40`–`111.8` raw (broken heuristic made it
  `1.40`–`2114`, ~3.2 orders) → **exactly `1.0` for every block** under
  Jacobi scaling.
- All 5 previously-regressed tests pass again (`gn_circle_collocation`,
  `hs_optimization_circle_converges`, `hs_lower_defect_than_trapezoidal_on_oval`,
  `mesh_refinement::refinement_on_circle`,
  `mesh_refinement::refinement_beats_cold_start_on_oval`).
- Neutrality: circle lap time moves by `1.14e-4` s (23× tighter than the
  broken heuristic's `2.59e-3` s), well inside golden tolerance.

**NOT fixed (convergence on hard tracks):** `optimize --hermite-simpson`
still does **not** converge on the default oval or
`random_spline_track(seed=42)` at N=50 — `eq_violation` stays at `4.08e-1`
(oval) and `3.73e-1` (spline), both *worse in absolute terms* than the
abandoned heuristic's non-converged values (`8.96e-3` and `6.65e-3`
respectively), even though Jacobi scaling is the mathematically correct
conditioning fix. This shows that perfect *local* conditioning at the
warmstart does not guarantee good conditioning along the whole solve
trajectory for a badly nonlinear problem — the frozen Jacobian stops
representing the system well once the iterate moves far from the warmstart,
which happens far more on oval/spline (large speed swings through braking
zones) than on the circle (near-constant speed throughout, converges in a
handful of iterations). **This is a conditioning fix, not a convergence
fix.** Achieving convergence on non-trivial tracks needs warmstart quality /
mesh continuation work — a separate, later slice. The `optimize`
golden (paused since Phase 0.1 slice 3) remains paused until that work
lands.

## A6 — Blast radius (as shipped)

**Changed:**
- New `crates/apex-optimizer/src/scaling.rs` (`Scaling`, `ScaledEvaluator`,
  `floor_scale`).
- `crates/apex-optimizer/src/lib.rs`: `pub mod scaling;` + re-export.
- `crates/apex-optimizer/src/collocation.rs`: `jacobi_scale`, `build_scaling`,
  `optimize_gn` (build/wrap/unscale/recompute-in-SI at the boundary).
  Scope is limited to `optimize_gn`'s point-mass path, as originally scoped.
  `optimize`, `optimize_seven_dof`, `optimize_fourteen_dof`, and
  `optimize_direct` are untouched and out of scope.

**Not changed:**
- `crates/apex-optimizer/src/{nlp.rs, gauss_newton.rs, solver.rs}` — zero
  edits, as designed.
- `apex-physics::qss_lap_sim` / `qss.rs` — untouched. `golden_oval_qss` is
  unaffected by construction and confirmed green.
- `OptimizationResult` — struct definition and field units stay SI.

**Guardrail enforced, not just documented:** `eq_violation`/`ineq_violation`
are recomputed against the unscaled evaluator on the unscaled final `x`
before returning from `optimize_gn` — proven by
`optimize_gn_reports_si_violation_not_scaled`, not left as a comment.
