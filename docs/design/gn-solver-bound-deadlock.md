# Diagnosis note: why `optimize --hermite-simpson` doesn't converge on non-trivial tracks

Status: **DEFERRED to Phase 3**. This is a diagnosis record, not a design
proposal — nothing here is implemented. See `PHYSICS_CHANGE.md` for the
dated log entry; this note holds the fuller mechanism write-up so it isn't
duplicated at length there.

## Symptom

`CollocationOptimizer::optimize_gn` (Hermite-Simpson, N=50, calibrated car)
converges cleanly on a constant-curvature track (`circle_track`,
`eq_violation → 7.9e-6`) but not on the default oval or
`random_spline_track(seed=42)` — both floor at `eq_violation ≈ 0.68` SI,
roughly 3-4 orders of magnitude above `constraint_tol = 1e-4`, regardless of
iteration budget.

## Root cause

The solver (`crates/apex-optimizer/src/gauss_newton.rs`) computes an
**unconstrained** Newton step from `(JᵀJ + reg·I)Δx = rhs` via CG, and
enforces variable bounds only by **post-hoc projection** (`project()`,
clamping `x` into `[lower, upper]` after each step) — there is no
active-set logic and no Lagrange-multiplier/dual mechanism for bounds.

This deadlocks exactly when the true optimum requires a bound to bind.
Concretely: `f_drive` saturates `max_drive_force` at the nodes on the
oval's two straights (physically correct — a car floors the throttle down a
straight). At the stalled point, the raw Newton direction has `‖Δx‖ ≈ 483`
(large, well-formed), but after damping and projection the net displacement
is `max|x − x_new| ≈ 7e-13` — genuinely zero. **25–28 of ~349 decision
variables sit exactly at a bound** (traced to indices in the `f_drive`
block, two clusters of ~13 consecutive nodes matching the two straights).
The linear system keeps demanding more drive force than physically exists;
projection clips the step back to the identical bound every iteration,
forever — a textbook projected-Newton deadlock, not a tuning problem.

## Ruled out by experiment (do not re-run these)

- **Variable scaling/conditioning** — fixed (Jacobi/diagonal preconditioning,
  see `docs/design/nlp-scaling.md`); confirmed NOT the cause of this
  specific failure (conditioning fix alone did not enable convergence on
  oval/spline).
- **Warmstart quality** — a control-corrected warmstart with 3.2× lower
  initial defect (`8.0 → 2.48`) made the outcome WORSE, not better (stalled
  at a higher final `eq_violation` than the plain QSS warmstart). This rules
  out warmstart quality as the bottleneck.
- **Line-search tuning** — loosening the backtracking give-up threshold by
  8 orders of magnitude (`1e-12 → 1e-20`) had zero effect. The line search
  accepts a step every iteration already (`accepted = true`); it is not
  rejecting anything.
- **Inner-CG precision** — raising the CG iteration cap 20× and tightening
  its tolerance 4 orders of magnitude had no meaningful effect (final
  violation changed in the 6th decimal digit). The direction is being
  solved correctly; it correctly points somewhere the projection then
  cancels.
- **Mesh coarsening** — no coarse N in {10, 15, 20, 25, 30, 40} converges
  from the QSS warmstart either, so mesh continuation has no rung to climb
  from with the current solver.

## Two known fix paths

- **(a) Active-set / bound-multiplier logic in the current GN solver**:
  detect bound-active variables, pin them, and solve the reduced
  free-variable system (classical active-set correction). Cheaper than (b)
  but is new solver infrastructure that would likely be superseded once (b)
  lands.
- **(b) Phase-3 interior-point solver**: handles active bounds natively via
  a log-barrier, which produces the correct implicit multiplier and avoids
  the projection deadlock by construction.

**Decision: deferred to Phase 3 (option b)**, to avoid building bound-handling
solver infrastructure twice.

## Current state

`golden_oval_optimize` (`bins/apex-cli/tests/golden_lap.rs`) remains
`#[ignore]`d; the `optimize` golden fixture is intentionally not generated.
Revisit this note when Phase-3 IP-solver work begins.
