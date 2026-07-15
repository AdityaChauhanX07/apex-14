**Status: investigation (envelope-analysis task, Part A).**

# Envelope-OCP convergence on real circuits — the "curvature-discontinuity" explanation is falsified

`free-trajectory-ocp.md` recorded a **Known limitation**: the synthetic Silverstone
`silverstone_circuit` hits `MaxIter` and does not reach tight feasibility, attributed
to **curvature discontinuities at the arc/straight joints** that "a coarse trapezoidal
mesh cannot resolve," with the prescribed fix being *higher node counts and adaptive
`s`-meshing*. Part A of the envelope-analysis task tested that explanation on the
**real, spline-smoothed, C2-continuous** imported Silverstone (`tracks/silverstone.json`,
from the 3D-track import). **The explanation does not survive the test.** Every clause
of it is wrong, and the true cause is a mistuned augmented-Lagrangian schedule that a
**config-only** change fixes.

## 1. The decisive experiment (confound isolation)

The CLI `optimize --envelope --calibrated` changes *two* things at once vs the documented
point-mass validation: the track (synthetic→real) **and** the envelope (the CLI always
builds the full load-sensitive, speed-dependent `AeroModel::f1_default` envelope, plus
`--calibrated` turns on aero). Holding the car fixed at the documented **point-mass**
config and changing **only** the track isolates the curvature hypothesis
(`constraint_tol = 5e-3`, `recommended_ip_config`, N=60):

| run | status | eq viol | ineq viol | tight? |
|---|---|---|---|---|
| **A.** synthetic + point-mass — reproduces the doc baseline (doc: eq≈0.02, ineq≈0.7) | MaxIter | 2.3e-2 | 1.10 | ✗ |
| **B.** REAL C2 Silverstone + point-mass — *only* the track changed | MaxIter | 5.9e-2 | 1.60 | ✗ |
| **C.** REAL + aero-on (= the CLI `--calibrated` path) | InfeasibleDetected | 6.9e-2 | 0.81 | ✗ |

The C2-continuous real track converges **slightly worse**, not better. If the joints were
the cause, the spline-smoothed track (no curvature discontinuities anywhere) would fix it.
It does not. **The curvature-discontinuity explanation is falsified.**

The solver is not broken: on the synthetic **circle** (N=60) and **oval** (N=40) with the
point-mass car it reaches `IpmStatus::Optimal` (the committed `envelope_ocp` tests pass).
The failure is specific to real-*lap-scale* problems.

## 2. Ruling out the task's other candidates

- **Mesh density — ruled out, and backwards.** Real + point-mass, ineq violation at
  N = 40 / 60 / 80 / 100 / 140 = **0.07 / 0.58 / 11.9 / 18.2 / 11.1**. Refining the mesh
  makes feasibility *worse*; the coarsest mesh is the best. The documented prescription
  ("higher node counts and adaptive meshing") points the wrong way.
- **Iteration budget — ruled out.** N=60 real + point-mass with a 7.5× budget (6000 iters)
  plateaus at eq = 0.063, ineq = 0.063 — the same floor as 1500 iters. It *stalls*; it
  does not run out of iterations.
- **High-v aero conditioning — contributory, not the cause.** Point-mass real (row B)
  already fails; turning aero on (row C) only flips the terminal status.

## 3. Where the infeasibility actually lives — the envelope inequality, not the dynamics

Splitting the equality (collocation) residual into its mesh-invariant **rate** form
(`integrated_defect / ds`) versus the envelope **inequality** exposes the real culprit:

| N (real + point-mass) | eq (integrated) | **eq (rate = /ds)** | **envelope ineq** |
|---|---|---|---|
| 40 | 7.8e-2 | **5.3e-4** | 0.07 |
| 60 | 6.2e-2 | **6.3e-4** | 0.58 |
| 100 | 3.7e-2 | **6.3e-4** | 18.2 |

The **dynamics collocation is effectively satisfied at every mesh** — the rate defect is a
tiny, mesh-stable ~6e-4. The CLI's headline "eq_violation ≈ 0.06 > 5e-3" is an artifact of
measuring the **integrated** (h-scaled) defect against a mesh-absolute tolerance; in rate
terms the dynamics are converged. The genuine, unresolved infeasibility is the **g-g-g
envelope inequality** `|a| ≤ (1−eps)ρ`, and it is what explodes with node count.

**h-normalization does not help** (tested): row-scaling the equalities by `1/ds` to make the
integrated defect mesh-invariant *starves* the augmented-Lagrangian penalty (~100× weaker),
so the rate defect blows up to 0.2–0.4 while the envelope ineq stays large. The equalities
were never the problem, so normalizing them cannot fix it.

## 4. Root cause and the config-only fix

`recommended_ip_config()` was tuned on the synthetic **point-mass** validation tracks:
`al_contract = 0.25` (conservative Hestenes–Powell) and `rho_max = 3e4`. On real-lap-scale
problems `rho` saturates `rho_max` while the envelope inequalities are still violated, the
line search then stalls, and the solve terminates `InfeasibleDetected` (that status fires
exactly when *the line search fails AND rho is at rho_max AND the point is infeasible*).
The AL penalty (`rho`) acts on the equalities — which are already fine — so raising the
penalty cannot clean up the inequalities; it just pins `rho` at its ceiling.

Loosening the AL schedule so multiplier updates (not penalty growth) carry feasibility
fixes it. Single-knob changes on real Silverstone (point-mass, N=40):

| config change vs recommended | status | eq | ineq | tight |
|---|---|---|---|---|
| baseline (`al_contract=0.25`, `rho_max=3e4`) | InfeasibleDetected | 7.8e-2 | 7.1e-2 | ✗ |
| **`al_contract = 0.1`** | Optimal (280 it) | **8.2e-6** | **2.0e-6** | ✓✓ |
| `rho_growth = 1.5` | Optimal (271 it) | 8.4e-4 | 4.6e-4 | ✓ |
| `rho_max = 3e6` | Optimal (270 it) | 2.8e-3 | 4.8e-3 | ✓ |

The doc's own note claimed loosening `al_contract` "destabilized the dual estimate and made
feasibility worse" — that was true **for the synthetic point-mass case it was tuned on, and
false for real tracks.** The shared config adopted for the analysis is
**`al_contract = 0.1`, `rho_max = 3e6`** (both existing `IpmConfig` knobs — no code change).

With that config and the **calibrated (aero-on)** car, real F1 circuits reach **machine-tight**
feasibility (eq/ineq ~1e-7): Silverstone, Spa, Monza, Catalunya, Spielberg, and Spa-3D all
converge `Optimal`. This overturns the documented "cannot converge on real circuits."

## 5. The remaining, honest limitation — a coarse-mesh sweet spot, and an unconverged objective

Two caveats survive the fix and are load-bearing for how the Part B numbers may be read:

1. **Per-track coarse-mesh sweet spot; no single N works for all.** Tight feasibility is
   reached only at coarse meshes (**N ≈ 24–40**, `ds ≈ 150–290 m`), and the winning N differs
   per track (Silverstone 40, Monza 30, Catalunya 32, Spielberg 28, Spa 36, Spa-3D 24).
   At **N ≥ 48** the envelope-inequality coordination breaks down again and the solve
   regresses to `MaxIter`/infeasible. So the shared config is one config, but the **node
   count is a per-track knob** — itself the reportable finding the task asked for.
2. **The lap-time objective is NOT mesh-converged at these meshes.** Constraint-tight ≠
   objective-accurate. Synthetic Silverstone, all three tight: N = 24 / 30 / 36 →
   **81.3 / 73.5 / 66.2 s** (a 15 s spread). Real Silverstone: N=30 → 83.9 s, N=40 → 89.8 s.
   At `ds ≈ 200 m` a corner spans one or two nodes, so the "optimal" line over-cuts and the
   lap time is optimistically biased and mesh-dependent. **The tight-feasible QSS-vs-OCP
   deltas are therefore directionally valid (the free line beats the fixed line) but are
   not reliable lap-time magnitudes**, and are not comparable across tracks solved at
   different N.

The real next step (deferred, out of this task's config-only scope) is **not** "more nodes"
but restoring feasibility at *finer* meshes: a mesh-continuation / warm-start ladder
(solve coarse-tight, interpolate, re-solve) or predictor-corrector `mu` so the envelope
inequalities stay coordinated as N grows. Adaptive `s`-refinement is still worthwhile, but
for objective accuracy, not to "resolve joints" — the joints were never the problem.

## 6. Reproduction

All runs above use `EnvelopeOcp` with `EnvelopeOcpConfig::default()` at the stated N and the
IP config `IpmConfig { al_contract: 0.1, rho_max: 3e6, constraint_tol: 5e-3,
max_iterations: 1500, ..EnvelopeOcp::recommended_ip_config() }`. Point-mass runs use the
`envelope_ocp` test's `point_mass_car()` helper; calibrated runs use
`CarParams::f1_2024_calibrated()` with the CLI's `AeroModel::f1_default` envelope. The
committed regression test `silverstone_tuned_reaches_tight` (synthetic circuit, tuned config,
N=36) locks in the config-only fix without depending on the gitignored real track data.
