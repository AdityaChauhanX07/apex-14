# Physics Change Log

This file is the audit trail for the golden-lap regression harness
(`bins/apex-cli/tests/golden_lap.rs`). It guards against silent drift in the
simulation output — lap time and the resampled speed trace — for pinned
baseline scenarios. Right now there is exactly one baseline: the default oval
track with the calibrated F1-2024 car, run through `qss_lap_sim`
(`golden_oval_qss`).

QSS is bitwise-deterministic (no RNG, no rayon, no hashmap-order dependence
in the code path it exercises), so any drift in the golden test reflects an
actual change to the physics or track-generation code — not run-to-run noise.
The test still compares with tolerance rather than bitwise equality, because
floating-point results are not portable across compiler version, OS, or
optimization level, and a multi-OS CI matrix is coming in Phase 0.4. The two
tolerances currently in force, defined in `bins/apex-cli/tests/golden_lap.rs`:

- `LAP_TIME_TOL_S = 0.010` (10 ms)
- `SPEED_RMSE_TOL_MS = 0.1` (0.1 m/s RMSE over the resampled speed trace)

When the OS matrix lands, these are exactly what make the gate survive
running on multiple OS/toolchain combinations instead of breaking on
harmless FP rounding differences — the comment beside the constants in
`golden_lap.rs` says so explicitly.

## The rule

**A failing golden test is a STOP, not a thing to silence.** When
`cargo test -p apex-14 --test golden_lap` fails, the maintainer must decide:

- **(a) Unintended regression** — a bug was introduced. Fix the code, not
  the fixture. Do not touch the golden fixture to make the test pass.
- **(b) Intended physics change** — the shift is a deliberate consequence of
  a code change (e.g. a tire model refinement, a car parameter update, a
  track-generation change). In this case:
  1. Add a dated entry to the log below: what changed, why, and the
     observed lap-time delta and speed-RMSE delta.
  2. Regenerate the fixture in the **same commit/PR** as the code change:
     ```
     REGEN_GOLDEN=1 cargo test -p apex-14 --test golden_lap -- --ignored
     ```
  A code change and its golden-fixture update must never be split across
  separate PRs — that defeats the point of the gate.

**Known fidelity note:** `qss --calibrated` currently resolves to the
constant-`tire_mu` grip-circle model (`apex_physics::qss_lap_sim`), not the
Pacejka load-sensitive `qss_lap_sim_tire` path — there is no fidelity
selector on the CLI's `qss` subcommand today. If a future change rewires
`qss --calibrated` to go through `qss_lap_sim_tire` instead, the resulting
golden-fixture shift is **expected** and must be logged here as an intended
change, not chased as a bug.

## Template entry

```
### YYYY-MM-DD — <short title>
- Change: <what code/config changed>
- Rationale: <why>
- Lap-time delta: <old> -> <new> (<+/-Xs>)
- Speed-RMSE delta: <value> m/s
- Fixtures regenerated: <list, e.g. f1_2024_calibrated__oval_default__qss.json>
```

## Log

### 2026-07-03 — Golden-lap harness established
- Change: Added the golden-lap regression harness itself
  (`bins/apex-cli/tests/golden_lap.rs`) and the first pinned baseline.
- Rationale: Golden-lap harness established; oval QSS baseline pinned.
- Lap-time delta: n/a (initial baseline)
- Speed-RMSE delta: n/a (initial baseline)
- Fixtures regenerated: f1_2024_calibrated__oval_default__qss.json (created)

### 2026-07-03 — Jacobi variable scaling adopted in the collocation NLP (conditioning fix only)
- Change: `CollocationOptimizer::optimize_gn` now scales decision variables
  by the reciprocal of their measured equality-Jacobian column norm at the
  QSS warmstart (`1/‖J[:,j]‖`, static, computed once, column-only —
  constraint/residual values stay unscaled). This **supersedes commit
  `44e52e5`** ("scale collocation NLP variables to fix convergence"), which
  introduced a per-block physical-reference-value heuristic (`s` scaled by
  track length, `n` by half-width, etc.). That heuristic was implemented,
  measured, and disproven: it over-scaled the `s` column, whose raw Jacobian
  column norm is a small, purely structural `√2` (from the ±1
  state-difference coefficients, unrelated to physical track length), up to
  a scaled column norm of `2114` — a ~1500× over-correction — and broke 5
  previously-passing tests. This change deletes that heuristic entirely (not
  commented out) and replaces it with the warmstart Jacobian-diagonal
  scaling described above. See `docs/design/nlp-scaling.md` for the full
  history and design.
- Rationale: fixes a real Gauss-Newton conditioning failure (decision
  variables spanned several orders of magnitude in the Jacobian against a
  flat `regularization = 1e-4`), which had broken 5 previously-passing
  optimizer tests. Jacobi scaling restores all 5 to green and drives every
  variable block's scaled Jacobian column norm to exactly `1.0`.
- Lap-time delta: **neutral**. `qss_lap_sim` and the QSS golden
  (`golden_oval_qss`) are completely untouched by this change — this only
  affects `CollocationOptimizer::optimize_gn`, a separate code path. As a
  correctness check, the `circle_track` optimize case (which already
  converged before this change) was re-run: lap time moved by **1.1e-4 s**
  (`11.495100` → `11.494986`), well within any golden tolerance — expected,
  since a change of variables cannot move the physical optimum, only the
  numerical path to it.
- Speed-RMSE delta: n/a (this change does not touch QSS or the speed trace;
  no golden was regenerated).
- Fixtures regenerated: none. The QSS golden is unaffected and stays green.
- **Explicitly NOT a convergence fix**: `optimize --hermite-simpson` still
  does not converge on non-trivial tracks (the default oval, or a random
  spline track) at N=50 — conditioning is fixed, but the solver still
  doesn't reach `constraint_tol` on these cases. The paused `optimize`
  golden (see Phase 0.1 slice 3) remains paused pending a follow-up
  warmstart/mesh-continuation slice; this entry does not unpause it.

### 2026-07-04 — DEFERRAL RECORD: optimize non-convergence root-caused, fix deferred to Phase 3 (no code change)
- This is a deferral record, not a code change — nothing in the solver or
  tests was touched by this entry.
- Symptom: `optimize --hermite-simpson` converges on constant-curvature
  tracks (`circle_track`, `eq_violation → 7.9e-6`) but not on the oval or a
  random spline track — both floor at `eq_violation ≈ 0.68` SI, well above
  `constraint_tol = 1e-4`, regardless of iteration budget.
- Root cause (precise): the projected Gauss-Newton solver
  (`gauss_newton.rs`) computes an unconstrained Newton step and enforces
  variable bounds only by post-hoc projection/clipping — it has no
  active-set or Lagrange-multiplier mechanism. It deadlocks when the
  optimum requires a bound to bind: `f_drive` saturates `max_drive_force`
  across the nodes on the straights (physically correct — a car floors the
  throttle on a straight), the linear system repeatedly demands more force
  than exists, and projection clips the step to ~zero every iteration.
  ~25-28 of 349 variables pinned at bound; net displacement ~7e-13.
- Ruled out by experiment (do not re-run): variable scaling/conditioning
  (fixed separately, wasn't the cause); warmstart quality (a 3.2x-better
  warmstart made it WORSE, not better); line-search tuning (no effect — the
  line search accepts every iteration, isn't the bottleneck); inner-CG
  precision (no effect — the direction is solved correctly, it correctly
  points infeasible); mesh coarsening (no coarse N from 10-40 converges
  either).
- Two known fix paths: (a) add active-set/bound-multiplier logic to the
  current GN solver (pin bound-active variables, solve the reduced
  free-variable system), or (b) the Phase-3 interior-point solver, which
  handles active bounds natively via the log-barrier. **Decision: deferred
  to Phase 3 (option b)**, to avoid building bound-handling solver
  infrastructure twice.
- Status: `golden_oval_optimize` remains `#[ignore]`d; the `optimize` golden
  fixture is intentionally not generated.
- Full mechanism write-up: `docs/design/gn-solver-bound-deadlock.md`.
