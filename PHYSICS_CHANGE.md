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
optimization level, and a multi-OS CI matrix is planned. The two
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

### 2026-07-07 — Grip-map mechanism + sector markers: mu_scale(s,n) grid, marker-aware sectors — **goldens byte-unchanged**
- **What changed.** Added `apex_track::MuScaleGrid`, a bilinearly-interpolated
  `(station, lateral)` grip-multiplier grid, as an optional schema v2 block
  (`mu_scale_grid`, `tracks/README.md`) attached to `Ribbon3d`. The §5.4 grip
  circle (`docs/math/track3d.md` §5.8) gains a `mu_scale(s, n)` factor:
  `μN → μ·mu_scale(s,n)·N`. **QSS never reads a ribbon's `mu_scale_grid` field
  itself** — `qss_lap_sim_3d_with_grip` takes an externally-supplied per-station
  multiplier vector instead, because a driven-line run passes QSS a
  reparameterized ribbon whose own `(s, n)` isn't the original centerline's; a
  QSS-internal lookup on its own ribbon argument would silently sample the
  wrong grid location. Centerline runs bake the vector via the new
  `Ribbon3d::centerline_mu_scale()` (`n = 0`); the driven-line pipeline
  (`apex_correlate::driven`) bakes it from the *original* ribbon at each
  sample's own projected `(s, n)` (`DrivenGeometry::lateral_offset_per_segment`
  / `mu_scale`). `qss_lap_sim_3d` itself is now a thin, grid-oblivious wrapper
  over `qss_lap_sim_3d_with_grip(..., None)`. This mechanism ships empty —
  nothing populates a real grip map yet (a rubbered-line-vs-dirty-line dataset
  is future work).
- **Also landed: sector markers (loose end).** `Track`/`Ribbon3d`
  gained an optional `sector_markers: Option<Vec<f64>>` field (schema v2 JSON,
  `tracks/README.md`); `apex_physics::sector_times_with_markers` buckets by
  explicit boundaries when present, and `qss_lap_sim`/`qss_lap_sim_3d`/
  `qss_lap_sim_tire` honor it automatically, falling back to the existing
  equal-arc-length-thirds split when absent (unchanged default behavior).
  `pit_lane_polyline` is deferred (schema note only, no consumer exists).
- **Flat / grid-absent invariance — no golden change.** Absent grid ⇒ no grid
  is even constructed, and `params.tire_mu * mu_scale` is only ever multiplied
  when a caller opts in explicitly; every existing call site (CLI, viewer,
  golden harness, correlation pipeline) is unchanged and passes no grid, so
  every existing code path executes identical float ops. An explicit all-`1.0`
  grid is *also* bitwise-identical: bilinear interpolation of a constant field
  returns the exact constant (`c + t·(c−c) = c`), and `x * 1.0 == x` is
  IEEE-754 exact — the same "collapses via exact algebra" discipline as the
  flat `cosθ=1.0`/`sinθ=0.0` case (§5.6). Proven by
  `qss::tests::absent_grid_and_explicit_uniform_grid_are_bitwise_equal_to_baseline`.
  Sector markers are similarly additive: absent (`None`, the default for every
  existing track file/fixture) reproduces `sector_times` exactly. **Golden
  lap fixtures untouched — no regeneration needed.**
- **Validation.** `MuScaleGrid::mu_at` unit tests (corners, edge clamping,
  wraparound across a closed-ribbon seam, exact-constant round trip);
  `qss::tests::mu_scale_scales_grip_circle_by_analytic_sqrt_factor` (no
  downforce ⇒ `v ∝ sqrt(mu_scale)` exactly); `qss::tests::low_grip_patch_slows_qss_by_analytic_grip_circle_factor`
  (a synthetic ring with a 0.7 patch slows to within 2% of the analytic
  `sqrt(0.7)` factor); `apex_correlate::driven::tests::driven_line_off_center_hits_low_grip_patch_centerline_does_not`
  (a grid with *lateral* variation only — a driven line at `n=+3` samples the
  low-grip side and slows, while the centerline `n=0` and a direct
  `centerline_mu_scale()` query both read full grip, confirming the
  grid-external design doesn't leak grid effects onto the wrong line);
  `qss::tests::explicit_sector_markers_produce_matching_sector_count_and_sum`.

### 2026-07-07 — 3D point-mass QSS physics: grade, vertical-curvature load, banking — **goldens byte-unchanged**
- **What changed.** Added `qss_lap_sim_3d(&Ribbon3d, &CarParams)` implementing the
  3D point-mass QSS (docs/math/track3d.md §5): the longitudinal grade force
  `−m·g·sinθ`, the 3D normal load `N = m(g·cosθ·cosφ + v²κ·sinφ + v²κ_v) + F_df`
  (compression in dips, unloading over crests, banking support), the grip circle
  on `μN`, and the banked cornering limit. The correlate pipeline
  (`driven`/`identify`/`infer`) gained optional-elevation 3D variants; the tire
  QSS and single-track/four-wheel/14-DOF models are **not** touched.
- **Flat invariance — no golden change.** `qss_lap_sim` (2D) is **untouched**.
  `qss_lap_sim_3d` short-circuits a geometrically flat ribbon (`Ribbon3d::is_flat`)
  straight to `qss_lap_sim` on the 2D projection, so flat tracks execute the
  identical float ops. For `θ=φ=κ_v=0`, `cos=1.0`/`sin=0.0` exactly, so every 3D
  expression collapses to the flat model even on the non-flat code path. Proven by
  `qss::tests::flat_ribbon_qss_bitwise_matches_track` (bitwise on oval / circle /
  Silverstone). **`golden_oval_qss` lap-time Δ = 0.000 s, speed-RMSE Δ = 0.000 —
  byte-identical.** No fixture regenerated.
- **Fidelity deferral.** Higher-fidelity models (single-track / four-wheel /
  14-DOF) get the same 3D terms in a **follow-up task** — the correlation pipeline
  runs on QSS, so QSS is first. Banking is plumbed + unit-tested but `0` in the
  current GLO-30/EU-DEM-derived data (a 25–30 m DEM cannot resolve camber across a
  ~14 m track); `banking_deg` is the manual per-corner override for later.
- **Validation.** Synthetic unit tests pass: banked-ring cornering vs the classic
  closed form (< 1e-4), vertical-curvature load `ΔN = m·v²·κ_v` (< 1e-9), constant-
  grade terminal-speed offset (< 2%), closed-lap gravity work = 0 (< 1e-6), and the
  3D closed-loop inference on a banked ring.
- **Spa result — honest, mixed (acceptance criterion NOT met).** See
  `docs/validation/correlation_spa_2024q.md` (flat-vs-3D section). 3D helps the
  aggregate (preset lap delta +1.841→+1.700 s, RMSE 9.32→8.73; fitted lap delta
  **−4.06→−2.86 s**) but the re-fit `power_scale` did **not** rejoin the pack
  (0.833→**0.802**, slightly worse). Silverstone control (10.7 m range) moved <1%
  on every parameter, as designed. Hypothesis: on a closed lap the grade force is
  conservative (net-zero work) so it cannot shift a lap-wide power multiplier, and
  3D introduces a descent over-carry (max Δv migrates to the Pouhon→Stavelot
  descent, 14→18 m/s) that keeps the point-mass fit de-powered — i.e. Spa's
  de-powering is not primarily an elevation artifact and points to the deferred
  higher-fidelity / energy-management work. **No parameter was tuned to force the
  criterion.**

### 2026-07-05 — Golden-lap harness closeout: converging circle optimize golden + formal interior-point-solver deferral
- **(a) New `golden_circle_optimize` fixture.** Added the first converging
  optimize-mode golden: constant-curvature circle (`circle_track(100.0, 12.0,
  200)`, R=100 m, L≈628.29 m), `f1_2024_calibrated` car, Hermite-Simpson
  transcription, `CIRCLE_OPTIMIZE_NODES = 30`, `optimize_gn` with
  `GaussNewtonConfig::default()`. Verified convergence: `eq_violation ≈ 7.8e-7`,
  far under `constraint_tol = 1e-4`, in well under the 100-iteration budget.
  Fixture (`f1_2024_calibrated__circle__optimize_hermite_simpson.json`):
  `lap_time = 11.494914 s`, `sector_times = [3.963736, 3.567399, 3.963779]`
  (sum = lap_time to 3e-15 s; the 3-way asymmetry is the deterministic
  whole-interval bucketing of ~29 intervals into equal-arc-length thirds, not
  a physics effect), speed trace resampled every 10 m (64 samples). Compared
  under the existing tolerances: lap ±0.010 s, each sector ±0.010 s, speed
  RMSE < 0.1 m/s. The test **fails (not skips)** if the solve does not
  converge — it asserts `converged` before comparing.
- **Determinism.** Before generating the fixture, the exact config was solved
  twice in-process and the two results were **bitwise-identical** on lap time
  and the full speed trace (no RNG, no rayon, no hashmap-order dependence in
  the point-mass `optimize_gn` path; the warmstart is deterministic QSS). A
  permanent guard test `circle_optimize_is_deterministic` asserts this bit-for-
  bit (not a tolerance).
- **Library reuse.** `apex_physics::sector_times(stations, interval_times,
  total_length, n_sectors)` was factored out as the single definition of the
  sector split; QSS's `integrate_lap_and_sectors` now delegates to it, and the
  optimize golden feeds it the optimizer's own node stations and per-interval
  `time_steps`. The QSS goldens are byte-unchanged by this refactor (lap-time
  accumulation order preserved; both QSS fixtures still pass untouched).
- **(b) Formal deferral of the oval / Barcelona optimize goldens to the interior-point solver work.**
  The fixtureless `#[ignore]`d `golden_oval_optimize` test is **removed**
  (replaced by a comment pointing here). Root cause (see
  `docs/design/gn-solver-bound-deadlock.md`): the Gauss-Newton collocation
  solver enforces variable bounds only by post-hoc projection and has no
  active-set / bound-multiplier mechanism, so it deadlocks whenever the optimum
  needs a bound to bind — on the oval and Silverstone, `f_drive` saturates
  `max_drive_force` across the straights, the linear system keeps demanding
  more force than exists, and projection clips the step to ~zero every
  iteration (floors at `eq_violation ≈ 0.2–0.98`, orders of magnitude above
  `constraint_tol`). This is a solver-capability gap, not a tuning issue
  (scaling, warmstart, line-search, and CG precision were all ruled out); the
  fix is the interior-point solver, which handles active bounds
  natively via a log-barrier. No solver numerics were touched here.
- **(c) Scope substitution.** The planned "Barcelona optimize
  Hermite-Simpson golden" is **consciously substituted** by
  `golden_circle_optimize` until the interior-point solver lands. The circle is the one non-trivial
  track the current solver converges on cleanly, so it is what pins
  optimize-mode output today; the oval/Silverstone/Barcelona optimize goldens
  are revisited when the interior-point solver lands (Barcelona additionally
  needs a TUMFTM import — no `tracks/barcelona.*` exists — and `tracks/`
  README forbids committing TUMFTM-derived files).
- Lap-time delta: n/a (new fixture; QSS goldens unchanged). Speed-RMSE delta:
  n/a. Fixtures regenerated:
  `f1_2024_calibrated__circle__optimize_hermite_simpson.json` (created via
  `REGEN_GOLDEN=1 cargo test -p apex-14 --test golden_lap -- --ignored
  regen_golden_circle_optimize`).

### 2026-07-05 — Fixture-schema change: `sector_times` null → computed (NOT a physics change)
- Change: `apex_physics::QssResult` gained a `sector_times: Vec<f64>` field,
  computed by `qss::integrate_lap_and_sectors` for both `qss_lap_sim` and
  `qss_lap_sim_tire`. The lap is split into `DEFAULT_SECTOR_COUNT = 3`
  equal-arc-length sectors (`s ∈ [0, L/3), [L/3, 2L/3), [2L/3, L]`); each
  lap-time interval is attributed in full to the sector containing its
  midpoint station, so the sector times sum to `lap_time` to within
  floating-point reassociation (unit test `sector_times_sum_to_lap_time`
  asserts < 1e-9 s). `Track` has no sector-marker field today, so the equal
  split is always used; the helper already takes `n_sectors` so honoring
  per-track markers later is a call-site change. The golden fixture
  `sector_times` field went from `null` (never computed) to populated, and
  `golden_lap.rs`'s shared comparison now checks each sector within the same
  ±0.010 s tolerance as lap time.
- **Why this is a schema change, not a physics change:** `lap_time` and the
  resampled speed trace are byte-for-byte the intended values as before — the
  lap-time integral accumulates the identical `dt` terms in the identical
  order; sector bucketing is a pure re-attribution of those same terms. No
  simulation output moved. The two QSS goldens still pass their existing
  lap-time/speed-RMSE gates unchanged; only the previously-`null`
  `sector_times` field was populated.
- Lap-time delta: **none** (0.0 s; identical integral, only re-bucketed).
- Speed-RMSE delta: **none** (speed trace untouched).
- Fixtures regenerated (this same commit, via
  `REGEN_GOLDEN=1 cargo test -p apex-14 --test golden_lap -- --ignored`):
  `f1_2024_calibrated__oval_default__qss.json` (sector_times
  `[7.147836, 9.497247, 8.819707]`) and
  `f1_2024_calibrated__silverstone_synthetic__qss.json` (sector_times
  `[22.748393, 26.317347, 35.618236]`). The paused optimize golden is
  unaffected (still `#[ignore]`d, still emits `sector_times: None`).

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
  golden (see the 2026-07-04 deferral record) remains paused pending a follow-up
  warmstart/mesh-continuation slice; this entry does not unpause it.

### 2026-07-04 — DEFERRAL RECORD: optimize non-convergence root-caused, fix deferred to the interior-point solver work (no code change)
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
  free-variable system), or (b) the interior-point solver, which
  handles active bounds natively via the log-barrier. **Decision: deferred
  to the interior-point solver work (option b)**, to avoid building bound-handling solver
  infrastructure twice.
- Status: `golden_oval_optimize` remains `#[ignore]`d; the `optimize` golden
  fixture is intentionally not generated.
- Full mechanism write-up: `docs/design/gn-solver-bound-deadlock.md`.
