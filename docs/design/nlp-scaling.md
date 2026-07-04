# Design note: variable/constraint scaling for the collocation NLP

Status: PROPOSED — not implemented. This note designs a fix for the Gauss-Newton
convergence failure characterized in the Phase 0.1 slice-3 diagnosis (oval and
`random_spline_track(seed=42)` fail to converge at N=50; violation *worsens*
50→400 nodes; `circle_track` converges cleanly to 8e-6). Root cause: raw-SI
decision variables and residuals span ~5 orders of magnitude in the Jacobian,
fed into a single flat `regularization = 1e-4` in `(JᵀJ + reg·I)`. This is a
conditioning failure, not a resolution or warmstart failure alone.

## A1 — Abstraction & placement

The current trait/struct boundary (`crates/apex-optimizer/src/nlp.rs`):

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
(`solver.rs`) already consume nothing but `&impl NlpEvaluator` + `&NlpProblem`.
Neither file needs to change. Scaling should be a **decorator that produces
another `NlpEvaluator`**, sitting entirely on the problem-definition side of
that boundary — so a future interior-point solver written against the same
trait gets scaling for free, with zero solver-side code.

Proposed new module, `crates/apex-optimizer/src/scaling.rs`:

```rust
/// Diagonal (per-component) reference scales. `x_si = x_scale[i] * x_scaled[i]`.
pub struct Scaling {
    pub x_scale: Vec<f64>,      // len n_vars
    pub c_eq_scale: Vec<f64>,   // len n_eq
    pub c_ineq_scale: Vec<f64>, // len n_ineq
}

impl Scaling {
    pub fn scale_x(&self, x_si: &[f64]) -> Vec<f64> { /* x_si[i] / x_scale[i] */ }
    pub fn unscale_x(&self, x_scaled: &[f64]) -> Vec<f64> { /* x_scaled[i] * x_scale[i] */ }
    pub fn scale_problem(&self, p: &NlpProblem) -> NlpProblem { /* divide bounds by x_scale */ }
}

/// Wraps any NlpEvaluator to present a scaled-space NlpEvaluator.
/// Solvers never know scaling exists.
pub struct ScaledEvaluator<'a, E: NlpEvaluator> {
    pub inner: &'a E,
    pub scaling: &'a Scaling,
}

impl<E: NlpEvaluator> NlpEvaluator for ScaledEvaluator<'_, E> {
    // objective(x_scaled)            = inner.objective(unscale(x_scaled))
    // objective_gradient(x_scaled)   = inner.objective_gradient(unscale(x)) .* x_scale   (chain rule)
    // equality_constraints(x_scaled) = inner.equality_constraints(unscale(x)) ./ c_eq_scale
    // equality_jacobian(x_scaled)[i,j] = inner_jacobian[i,j] * x_scale[j] / c_eq_scale[i]
    // (inequality_* mirror the equality_* rows)
}
```

This is deliberately the whole abstraction: one data struct (scales), one
generic wrapper. No pluggable strategy trait, no per-solver scaling variants,
no runtime scale selection, no configuration enum. Things I considered and am
explicitly leaving out because they are not needed to fix conditioning:

- **Automatic/adaptive scaling** (e.g. iterative re-scaling from the running
  Jacobian, as some interior-point codes do). The diagnosis shows the
  magnitude spread is a *static* property of the physical units involved
  (mass, track width, curvature), not something that drifts during the solve
  — static scales computed once are sufficient (see A2) and are simpler to
  reason about and test.
- **A general `Scaler` trait with swappable implementations.** There is
  exactly one scaling scheme in this design; a trait would only exist to have
  one implementer, which is speculative generality with no current consumer.
- **Scaling the inequality grip-circle constraint.** It is already
  dimensionless by construction (see A2) — touching it would be scaling
  something that isn't broken.

`CollocationOptimizer::optimize_gn` is the only call site that changes: it
builds a `Scaling` from `self.car`/`self.track`/`self.config` (and the
existing QSS warmstart it already computes), wraps the evaluator, problem,
and `x0`, runs the solver unchanged, then unscales the returned `x` before
calling `extract_result_gn`. Everything downstream of that point is identical
to today.

## A2 — Reference scales per block

All scales below are **static**: computed once from `CarParams`, `Track`, and
`CollocationConfig` (plus the QSS warmstart pass that `initial_guess()`
already runs — reusing its output costs nothing extra), before the GN loop
starts. Nothing depends on the solver's iterates, so scales are reproducible
run-to-run and trivial to unit test in isolation. I see no evidence in the
diagnosis that static scales are insufficient (the magnitude spread is a
fixed property of the problem, not something that changes as `x` moves), so
I'm not proposing iteration-dependent (adaptive) scaling.

| Variable block | Proposed reference scale | Justification |
|---|---|---|
| `s` (station, m) | `track.total_length` | `s ∈ [0, L]` by construction; `s/L ∈ [0,1]`. Purely static from `Track`. |
| `n` (lateral offset, m) | half the track width (e.g. mean or max of `width_at(s)` half-widths over the track) | `n` is physically bounded to roughly this range by the track-boundary inequality; measured warmstart range was `[0,0]` but the *bound*, not the warmstart value, is the right static reference since `n` moves during optimization. |
| `v` (speed, m/s) | `max(qss_warmstart.speeds)` | Already computed for free as part of `initial_guess()`. Ties the scale to *this* car+track pair exactly (measured oval range was `[44.8, 88.0]` m/s), rather than a generic formula that could be wrong for a very different car. |
| `alpha` (heading deviation, rad) | `1.0` (no rescaling) | Heading deviation from the track tangent is already O(0.1–1) rad for normal driving — it is not part of the 5-order-of-magnitude spread the diagnosis identified. Rescaling an already-unit-order quantity would add a moving part for no conditioning benefit — explicitly one of the "don't touch what isn't broken" cases from A1. |
| `f_drive` (drive/brake force, N) | `max(car.max_drive_force, car.max_brake_force)` | A hard physical bound already stored on `CarParams`; static, requires no solve. Puts `f_drive/f_ref` inside roughly `[-1, 0.5]`. |
| `curv` (curvature command, 1/m) | `max(\|track.segments[i].curvature\|)` over the track | Static, computed once from `Track`. For the default oval this is `0.0125` (`1/80`), putting scaled curvature in roughly `[-1, 1]`. |
| `dt` (time step, s) | `qss_warmstart.lap_time / (n_nodes - 1)` | Reuses the QSS pass again; gives the "expected" interval duration at this car/track/mesh-density combination, again for free. |

**Equality-constraint (defect) residual scales**: each dynamics-defect
component is literally a difference of one state variable's values at two
nodes (`state_k1[j] - state_k[j] - ...`), so it has the *same physical units*
as that state component. The minimal, dimensionally-correct choice is to
reuse the matching variable's scale rather than invent an independent
constant:

| Defect component | `c_eq_scale` |
|---|---|
| `ds/dt` residual (index `4k+0`) | `s_scale` (`= total_length`) |
| `dn/dt` residual (index `4k+1`) | `n_scale` |
| `dv/dt` residual (index `4k+2`) | `v_scale` |
| `dalpha/dt` residual (index `4k+3`) | `alpha_scale` (`= 1.0`) |
| periodicity: `s`/`n`/`v`/`alpha` wrap (4 extra, closed tracks) | same four scales, respectively |

**Inequality-constraint scales**: the track-width residual (`n - w_l`,
`-w_r - n`) is a difference of `n`-like quantities → scale by `n_scale`. The
grip-circle residual, `(f_lon/f_grip)^2 + (f_lat/f_grip)^2 - 1`, is **already
dimensionless by construction** — leave its scale at `1.0` (see A1's
explicit-exclusion list).

With this table, every scaled-space quantity is O(0.1–1) instead of spanning
`~1e-3` (force-related) to `~1e2` (curvature/heading-rate-related), which is
exactly the ratio the diagnosis flagged as the conditioning problem.

## A3 — Invertibility & neutrality argument

Scaling is a diagonal linear change of variables: `x_si = diag(x_scale) · x_scaled`.

- **Exact invertibility**: `scale_x` divides elementwise by `x_scale`,
  `unscale_x` multiplies elementwise by the same `x_scale`. Since every entry
  of `x_scale` is a positive, finite, iterate-independent constant (guarded
  with a small floor, e.g. `max(scale, 1e-9)`, for the degenerate edge case
  of a track with zero curvature everywhere, so `curv_scale` is never
  literally zero), `unscale_x(scale_x(x)) == x` up to IEEE-754 floating-point
  rounding — one multiply and one divide by the same constant, which is
  accurate to machine epsilon (~2.2e-16 relative), not exact bit-for-bit in
  general. The round-trip unit test in A5 should therefore assert closeness
  to ~1e-12 relative, not `==`.
- **Physical neutrality**: dividing every decision variable and every
  constraint residual by a fixed positive constant is exactly
  non-dimensionalization — it relabels coordinates and rescales the
  magnitude of "how far from zero" a constraint reads, but the *zero set*
  `{x : c_eq(x) = 0}` and the *objective's ordering* (`f(x_1) < f(x_2)` iff
  `f(unscale(x_1)) < f(unscale(x_2))`, since unscaling the objective is
  identity — the objective itself, lap time, is not rescaled, only its
  gradient picks up a chain-rule factor) are unchanged. The physical optimum
  cannot move; only the numerical path the solver takes to reach it changes.
- **Correctness invariant**: because `circle_track` already converges
  cleanly today (`eq_violation = 7.93e-6`), it is the neutrality control —
  after implementing scaling, it **must** still converge, and its lap time
  must land within a very tight tolerance of today's value (proposed:
  `1e-6` s, far tighter than the golden-lap harness's `0.010` s — this test
  is checking mathematical equivalence of a reparametrized problem, not
  cross-build FP portability, so it should use a much stricter bound than
  the physics golden's tolerance).

## A4 — Regularization interaction

Today, `regularization = 1e-4` is added flatly to every diagonal entry of
`JᵀJ`, but the diagnosis's D5 finding shows `JᵀJ`'s diagonal already spans
~5 orders of magnitude (`~1e-3` for force-related entries to `~1e2` for
curvature/heading-rate entries) *before* regularization is added. A single
flat `reg` cannot be simultaneously right-sized for both ends of that range —
it over-damps the small-magnitude directions and under-damps the
large-magnitude ones.

After scaling, `JᵀJ`'s diagonal collapses to roughly O(1) uniformly (by
construction of the per-block scales in A2), which is precisely the regime a
single flat regularization constant is designed for. **Recommendation:
leave `regularization = 1e-4` unchanged as the starting point** — scaling
should reduce or eliminate the need to touch it, not require a coordinated
change alongside it. This is the minimal move: don't retune two things when
only one was diagnosed as broken. It should be the first thing re-verified
empirically once scaling is implemented (part of A5's efficacy test), rather
than assumed correct or preemptively changed.

One real, first-order side effect that is **not** a regularization question
but is adjacent to it: `constraint_tol = 1e-4` in `GaussNewtonConfig` is
compared against the solver's own internal `eq_violation`/`ineq_violation`
during its termination check (`gauss_newton.rs`'s `feasible = ev <
config.constraint_tol && iv < config.constraint_tol`). If the solver runs
against the *scaled* evaluator (as designed), that internal check now
compares `1e-4` against a scaled residual, not an SI one. Because the
`c_eq_scale` table in A2 reuses each defect's own state-variable scale, a
scaled residual of `1e-4` roughly corresponds to an SI residual of
`1e-4 * (that block's scale)` — e.g. for the `dv/dt` block, `1e-4 * v_scale
≈ 1e-4 * 88 ≈ 8.8e-3` m/s, not `1e-4` m/s. **This is a semantic shift in
what `constraint_tol` means and is flagged here explicitly as an open
decision for the implementer, not silently absorbed** — see A6.

## A5 — Test/verification plan

1. **Neutrality** (falsifiable pass/fail): `circle_track(100.0, 12.0, 200)`,
   same `CollocationConfig` as the existing `gn_circle_collocation` test.
   Assert `converged == true` (unchanged) and
   `|lap_time_post_scaling - lap_time_baseline| <= 1e-6` s. This is the
   single most important test in the suite — if it fails, the scaling
   implementation has a bug (most likely a Jacobian chain-rule error or an
   inconsistent scale/unscale pair), not a research finding.

2. **Efficacy**: oval (default, N=50, calibrated) and
   `random_spline_track(seed=42, 50 nodes)`. Primary target:
   `converged == true`. Fallback success criterion, stated in advance, if
   full convergence isn't reached: `eq_violation` drops by **at least two
   orders of magnitude** relative to the unscaled baseline (oval:
   `6.42e-2 → < 6.42e-4`; spline: `2.90e-2 → < 2.90e-4`), with no regression
   on either track.

3. **Conditioning signature**: reproduce the D4 node-count table exactly
   (N = 50, 100, 200, 400) post-scaling. Success = violation does **not**
   worsen with `N` (flat or improving trend), in contrast to the pre-scaling
   `6.42e-2 → 4.75e-2 → 3.62e-1 → 5.58e-1` trend. This is the most
   diagnostic single check for "did scaling fix the conditioning issue" as
   opposed to "got lucky on one track" — it's the one result that most
   directly targets the root cause rather than a symptom.

4. **Round-trip unit test**: for a handful of representative vectors (the
   QSS warmstart `x0`, the raw variable bounds themselves, and one
   arbitrarily perturbed vector within bounds), assert
   `unscale_x(scale_x(x))` matches `x` elementwise within relative tolerance
   `~1e-12` (per A3's machine-epsilon argument).

**Falsification criterion** (one sentence, stated in advance): if, after
implementing scaling exactly as designed, the oval's and
`random_spline_track(seed=42)`'s `eq_violation` values move by less than
roughly one order of magnitude and/or the N=50→400 violation trend still
worsens rather than flattens or improves, that falsifies "unscaled
conditioning is the dominant cause," and the next investigation should be
warmstart quality / mesh continuation (candidate fix #2 from the prior
diagnosis), not further scaling tuning.

## A6 — Blast radius

**Changes**:
- New file `crates/apex-optimizer/src/scaling.rs` (`Scaling`,
  `ScaledEvaluator`, `scale_x`/`unscale_x`/`scale_problem`).
- `crates/apex-optimizer/src/lib.rs`: `pub mod scaling;` + re-export.
- `crates/apex-optimizer/src/collocation.rs`: `optimize_gn` gains a build
  step (construct `Scaling`, wrap evaluator/problem/x0, unscale the result).
  Scope is deliberately limited to `optimize_gn`'s point-mass path — the
  diagnosis was run against that path specifically. `optimize`,
  `optimize_seven_dof`, `optimize_fourteen_dof`, and `optimize_direct` are
  **not** touched by this design and are explicitly out of scope; if they
  show similar conditioning issues, that's a separate follow-up informed by
  re-running the same diagnosis against them.

**Not changed, and must not change**:
- `crates/apex-optimizer/src/{nlp.rs, gauss_newton.rs, solver.rs}` — zero
  edits. This is the entire point: any current or future solver (including
  the Phase 3 interior-point solver) consumes `NlpEvaluator`/`NlpProblem`
  exactly as today, oblivious to whether scaling sits underneath.
- `apex-physics::qss_lap_sim` / `qss.rs` — completely untouched; scaling only
  touches the collocation NLP path. `golden_oval_qss` is unaffected by
  construction.
- `OptimizationResult` — struct definition and field units stay SI.
  Unscaling happens *before* `extract_result_gn` runs, so nothing downstream
  (CLI printout, telemetry CSV/SVG export, the paused optimize golden from
  slice 3, the viewer) sees anything different in shape or units.

**Explicit risk to guard against**: `eq_violation`/`ineq_violation` in
`OptimizationResult` must be **recomputed against the unscaled inner
evaluator on the unscaled final `x`**, not read off the scaled solver's
`GaussNewtonResult` directly. If that recompute step is skipped, a
scaled-space residual number would leak past the NLP boundary into
`OptimizationResult`, and from there into golden fixtures, telemetry, or CLI
output — silently changing what "eq_violation" means to a human reading it,
even though `lap_time`/`speeds`/`offsets` would still be correct. This must
be enforced by construction (the recompute call is mandatory in
`extract_result_gn`'s call path), not left as a comment or convention.

**Secondary flag**: per A4, `constraint_tol`'s effective meaning shifts once
the solver's internal termination check runs against scaled residuals. The
converged/not-converged decision boundary will move slightly relative to
today even though the *reported* `eq_violation` (recomputed in SI) stays
honest. This is a real behavior change to call out to the maintainer at
implementation time, not something to quietly work around.

---

## Summary

- **File**: this design note lives at `docs/design/nlp-scaling.md`.
- **Per-block scale table**: `s → total_length`, `n → track half-width`,
  `v → max(QSS warmstart speed)`, `alpha → 1.0` (unscaled), `f_drive →
  max(max_drive_force, max_brake_force)`, `curv → max(|track curvature|)`,
  `dt → QSS lap_time / (n_nodes - 1)`; equality-defect scales reuse the
  matching state-variable scale; the grip-circle inequality is left
  unscaled (already dimensionless).
- **Falsification criterion**: if oval/spline `eq_violation` moves less than
  roughly an order of magnitude and the N=50→400 trend still worsens after
  implementing exactly this design, unscaled conditioning is not the
  dominant cause and the investigation should move to warmstart/mesh
  continuation instead.

Nothing in this note has been implemented. Stopping for review.
