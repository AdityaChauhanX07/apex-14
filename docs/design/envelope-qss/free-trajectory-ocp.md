**Status: implemented**

# Free-trajectory OCP — the envelope racing-line optimizer

The assembly step of the envelope-QSS workstream: the small `s`-domain optimal
control problem that optimizes the free racing line against the cached g-g-g
envelope, solved by the bound-capable interior-point solver. The trim solver
produced the operating-point map; the envelope generator stored it with C1
interpolation; the IP solver handles the binding track-edge bounds; this doc
covers wiring them into a minimum-lap-time problem and the validation.

- Formulation + `a_y` derivation: [`docs/math/envelope_ocp.md`](../../math/envelope_ocp.md).
- Solver: [`ip-solver.md`](ip-solver.md). Envelope: [`envelope-generation.md`](envelope-generation.md).
- Code: `crates/apex-optimizer/src/envelope_ocp.rs`, tests
  `crates/apex-optimizer/tests/envelope_ocp.rs`, CLI `apex-14 optimize --envelope`.

## Problem shape

`s`-domain, states `{n, xi, v}`, controls `{a_x, kappa_cmd}`, `5N` variables
laid out `[n | xi | v | a_x | kappa_cmd]`. `3N` trapezoidal dynamics-defect
equalities (periodic flying-lap closure), `N` envelope inequalities
`|a| ≤ (1−eps)·rho(theta; v, g_z)`, track-edge box bounds on `n`, and guards
`v ≥ v_min`, `|xi| ≤ xi_max`. Objective `∫ (1−n·kappa)/(v·cos xi) ds` plus small
control-rate regularization. Warm start from the fixed-line (centerline) QSS.
The full formulation is in the math doc.

## Design decisions

### Control parameterization: `kappa_cmd`, not `a_y`

The lateral control is the **path curvature** `kappa_cmd`, with the envelope
seeing `a_y = v²·kappa_cmd`. The obvious alternative — make `a_y` the control
directly, so the envelope constraint `|a| = sqrt(a_x² + a_y²)` is O(1) and
speed-independent — was implemented and measured, and it **froze**: with `a_y`
direct, `∂xi'/∂a_y ≈ 1/v² ≈ 1/1600`, so the envelope-clamped `a_y` cannot move
the heading defect and the equalities stall at `eq ≈ 2.7` (vs `1e-5` with
`kappa_cmd`). The dynamics coupling dominates the envelope conditioning here.
`kappa_cmd` gives `∂xi'/∂kappa_cmd ≈ 1` and converges; the `v²` factor it puts
in the envelope Jacobian is handled analytically in `envelope_ineq`.

### Safety margin `eps = 0.01`

`rho_eff = (1 − eps)·rho`. Two jobs: (1) keep the optimum strictly interior to
the envelope, where `rho` is C1-smooth (a boundary-exact optimum sits on the
non-smooth hull and hurts the interior-point convergence); (2) cover the
envelope generator's measured ~0.76 % over-estimation on the default
24×10×6 grid (see `envelope-generation.md`). `0.01` comfortably dominates the
0.0076 bias while costing ~1 % of theoretical grip. Exposed as
`EnvelopeOcpConfig::eps` / `--eps`.

### Solver tuning: gentle penalty ramp

The OCP exposed two shortfalls in the interior-point solver's
augmented-Lagrangian schedule, both fixed by new (default-preserving) config
knobs on `IpmConfig`:

- **`rho_growth` (new; default `10.0`, OCP uses `3.0`).** The AL penalty `rho`
  is grown whenever the equalities fail to contract 4× at a schedule advance.
  The stationarity residual `dual_inf` is inflated by the `rho·Jeqᵀc_eq` term,
  so the "inner subproblem solved" gate never fires and the schedule advances
  purely on the iteration-count fallback, ramping `rho ×10` every ~8 iters.
  `rho` then saturated `rho_max` while `eq` was still large, and the stiff
  penalty **froze the trajectory near the warm start** — the racing line never
  reached the track edge. A gentler `×3` ramp keeps `rho` moderate long enough
  for the *objective* to migrate `n` to the edge, reaching feasibility via
  multiplier updates instead of penalty growth.
- **Step-gated barrier-floor acceptance.** The solver previously accepted a
  point as optimal once `mu` hit its floor and the equalities were feasible
  (`barrier_annealed`), *regardless of the objective*. On the OCP that fired at
  iter ~104 with the line barely off centerline. Acceptance is now additionally
  gated on the last primal step being small (`last_rel_step ≤ opt_tol`): a
  frozen iterate is accepted, a still-migrating line is not.
- **Objective-aware best-iterate.** Among iterates feasible to `constraint_tol`
  the solver now returns the **lowest-objective** one (not merely the
  least-infeasible), so a genuine optimization run returns its optimized point.
- **`mu_reduction = 0.5`** (slow barrier anneal) for the same reason: a softer
  barrier lets `n` travel before the schedule ends.

An `al_contract` knob (default `0.25`) was also added and, in this original
point-mass validation, left at its default — loosening it *destabilized* the
dual estimate and made feasibility worse here. **Superseded:** on real circuits
the opposite holds, and the shared config now uses `al_contract = 0.1` (favouring
multiplier updates over penalty growth) — see `real-track-convergence.md` Parts A
and B. `recommended_ip_config()` carries `al_contract = 0.1, rho_max = 3e6`.

These changes leave every existing IP test bit-for-bit unchanged (the deadlock
still resolves to `eq = 1.29e-7`, the QP/Rosenbrock/determinism tests pass) —
verified because the new knobs default to the old behavior.

## Validation (point-mass car; aero/drag/load-sensitivity off)

| Track | Nodes | Status | OCP lap | Baseline | Result |
|-------|------:|--------|--------:|---------:|--------|
| Circle R=100, w=8 | 60 | Optimal (~234 it) | 15.038 s | analytic **15.041 s** | **0.02 % error** |
| Oval L=200 R=80 w=12 | 40 | Optimal (277 it) | 19.449 s | QSS 21.015 s | **−7.45 %**, line reaches **both** edges (`n = ±6`) |
| Silverstone (synthetic) | 60 | MaxIter | 77.54 s | QSS 94.33 s | −17.8 %; see limitation |

*(The wall times originally quoted here — "~9 s / ~10 s / ~52 s" — predated the
`real-track-convergence.md` retuning and no longer hold; the circle now solves sub-second.
Iteration counts and lap times are current. Wall is machine/config-dependent — not a
headline. The circle lap `15.038 s` is re-verified in the close audit.)*

- **Analytic circle.** On a circle the fastest line hugs the inner edge
  (radius `R − w_left`) at the envelope-limited speed; with aero off `rho` is
  speed-independent so `t* = 2π·sqrt((R−w)/rho_eff)`. The OCP matches to
  **0.02 %** and puts `n` on the inner bound. (`circle_matches_closed_form`.)
- **Monotone improvement.** OCP lap < fixed-line QSS on every track that
  converges (circle, oval), asserted in the tests.
- **Corner-cutting.** On the oval the line reaches both track edges (wide
  turn-in, tight apex), `n` within 0.2 m of `±w`. (`oval_corner_cutting_and_monotone`.)
- **Determinism.** Two independent solves are bitwise-identical in `n`, `v`,
  and lap time. (`determinism_bitwise`.)
- **IP-log sanity.** The per-iteration log is populated, `mu` anneals downward,
  and the final logged equality residual is feasible.

### Known limitation — complex circuits at coarse mesh

> **⚠️ Superseded (envelope-analysis task, Part A, 2026-07-15).** The
> curvature-discontinuity explanation below was **tested and falsified** — see
> [`real-track-convergence.md`](real-track-convergence.md). In brief: the
> spline-smoothed, C2-continuous **real** Silverstone does not converge any
> better than the synthetic one (so joints are not the cause); refining the mesh
> makes feasibility **worse**, not better; and the collocation *rate* defect is
> already tiny (~6e-4) at every mesh — the unresolved infeasibility is the
> **envelope inequality**, not the dynamics. The real cause is that
> `recommended_ip_config`'s augmented-Lagrangian schedule (`al_contract = 0.25`,
> `rho_max = 3e4`) was tuned on the synthetic point-mass tracks and saturates
> `rho_max` on real-lap-scale problems. A **config-only** change on existing
> `IpmConfig` knobs — **`al_contract = 0.1`, `rho_max = 3e6`** — reaches
> *machine-tight* feasibility on all five real F1 circuits (regression test
> `silverstone_tuned_reaches_tight`). The surviving limitation is different: tight
> feasibility holds only at a **per-track coarse mesh** (N ≈ 24–40), and the
> lap-time objective is not mesh-converged there — the fix for finer meshes is
> mesh-continuation / predictor-corrector `mu`, not "resolving joints."

The original (falsified) account, kept for the record:
Silverstone (`silverstone_circuit`) **runs and improves ~18 %** but does not
reach tight feasibility (`eq ≈ 0.02`, `ineq ≈ 0.7`) at practical mesh /
iteration budgets. The synthetic layout is built from arc/straight primitives
with **curvature discontinuities** at the joints; at 60–80 nodes (`ds ≈ 75–100`
m) a trapezoidal mesh cannot resolve those transitions, so the collocation
defects there cannot be driven to `1e-4`. This is a discretization limit of the
"small OCP" on a real circuit, not a solver bug — the circle and oval, whose
curvature is piecewise-constant and well-resolved, converge cleanly. The
Silverstone test therefore asserts *runs + improves + finite + uses the track
width*, not a closed-form value. Higher node counts and adaptive `s`-meshing
(refining at curvature jumps) are the natural next step.

## CLI

```
apex-14 optimize --envelope --track <file> [--car <toml> | --calibrated] \
                 [--nodes N] [--eps E] [--csv out.csv] [--svg out.svg]
```

Implies the interior-point solver. Generates (and caches) the envelope for the
car, solves the OCP, prints the OCP lap, the QSS baseline, the improvement, and
the feasibility residuals, and optionally writes the trajectory CSV
(`s, n, xi, v, a_x, kappa, x, y` with a `RunMetadata` provenance block) and a
speed-colored racing-line SVG. The CLI uses a relaxed `constraint_tol = 5e-3`
(physically ~mm on `n`, ~0.005 rad on `xi`) and a larger iteration budget for
complex real tracks.
