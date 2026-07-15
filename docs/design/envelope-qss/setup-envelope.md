**Status: integrated (opt-in).** The rank-stability gate passed, so the envelope
free-trajectory OCP is wired as an opt-in inner loop for the CMA-ES setup
optimizer: `setup-optimize --inner envelope [--nodes N]`. The default remains
fixed-line QSS; no existing behavior changes.

# Envelope-OCP as a setup-optimizer inner loop

The envelope-OCP lap time is constraint-tight but **not** mesh-converged — real
circuits swing ~20 % in absolute lap time across node counts `N`
(`real-track-convergence.md` §B.3). A CMA-ES setup search does not need absolute
accuracy, only that setups are **ranked** consistently at a fixed `N`. This
document records the gate that tested that premise (Part 1), and the integration
it unblocked (Part 2).

## Part 1 — the rank-stability gate (decisive)

**Setup.** Real Silverstone (`tracks/silverstone.json`), calibrated car
(`f1_2024_calibrated`). Eight setup variants: the baseline plus seven seeded
(`StdRng` seed `20260715`) uniform draws across the full seven-parameter
`SetupSpace` (drag, downforce, aero balance, brake bias, CoG height, tyre radial
stiffness, front weight distribution). Each variant's **envelope is regenerated**
— the setup changes the `CarParams` content hash, so a stale envelope would be
physically wrong; the cache keys on that hash and correctly misses on every
distinct setup.

**Choosing the mesh pair.** A per-variant sweep over `N ∈ {24,28,32,36,40,44,48}`
showed the tight-feasible set is `N`-dependent (`N=28` and `N=40` each leave
2–3 variants infeasible; `N=44,48` fail widely). **`N=24` and `N=32` are the pair
tight-feasible for all eight variants**, so the gate uses them.

**Result (N=24 vs N=32, all 8 variants tight):**

| variant | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| QSS lap (s) | 107.05 | 105.27 | 103.22 | 112.26 | 102.69 | 115.58 | 111.50 | 112.44 |
| env N=24 (s) | 72.66 | 77.06 | 72.50 | 70.86 | 74.23 | 74.52 | 74.58 | 74.41 |
| env N=32 (s) | 86.19 | 88.57 | 85.56 | 85.22 | 86.84 | 86.86 | 87.59 | 86.98 |

- Ranking (fast→slow) N=24: `[3, 2, 0, 4, 7, 5, 6, 1]`
- Ranking (fast→slow) N=32: `[3, 2, 0, 4, 5, 7, 6, 1]`
- **Spearman ρ(N24, N32) = 0.9762, Kendall τ = 0.9286** — one discordant pair, an
  adjacent swap of variants 5 and 7 (lap times 74.52 vs 74.41 s at N=24, a 0.1 s
  near-tie). A second seed (`777`) gives ρ = 0.9524, also passing.

**The absolute lap times are wildly unconverged and yet the ranking is stable.**
Variant 0 goes 72.7 → 86.2 s between the two meshes (an 18 % swing), and across
the full sweep lap times are even non-monotone in `N` (72.7 → 74.8 → 86.2 → 68.2
at N=24/28/32/36). Absolute accuracy is worthless; the *order* survives.

**GATE: PASS** (ρ ≥ 0.9, one adjacent swap in eight) → proceed to Part 2.

### The QSS↔envelope divergence, and which knobs each model sees

The envelope ranking correlates only weakly with the fixed-line QSS ranking
(Spearman ≈ 0.14–0.69 across seeds). A per-parameter sweep (perturb one knob
`min`↔`max`, hold the rest at baseline; Silverstone, calibrated) explains why:

| setup parameter | Δ QSS lap (s) | Δ envelope lap (s) | seen by |
|---|---:|---:|---|
| drag_coeff | 3.47 | 3.94 | **both** |
| lift_coeff | **14.39** | 0.40 | QSS only |
| cog_height | 0.00 | **0.66** | **envelope only** |
| weight_dist_front | 0.00 | **0.46** | **envelope only** |
| aero_balance_front | 0.00 | 0.00 | neither |
| brake_bias_front | 0.00 | 0.00 | neither |
| tire_radial_stiffness | 0.00 | 0.00 | neither |

Two structural facts fall out, both load-bearing for the integration:

1. **The genuine payoff.** `cog_height` and `weight_dist_front` move the envelope
   lap but are **invisible to the point-mass QSS** (its friction circle has no
   load transfer and no per-axle static split). This is exactly the promised
   physics payoff: the envelope is load-sensitive where the friction circle is
   blind.

2. **Aero-blindness (a real limitation).** `lift_coeff` is QSS's single biggest
   lever (14.4 s) but the envelope barely responds (0.4 s). The reason is
   architectural: the point-mass QSS computes downforce from
   `CarParams::downforce(v)` (i.e. `lift_coeff`), whereas the envelope's downforce
   comes from the **independent `AeroModel`** (`aero.compute(...)`), which the
   sweep holds at `AeroModel::f1_default`. So `lift_coeff` and `aero_balance_front`
   (also an `AeroModel`-side quantity) **do not reach the envelope at all** — only
   `drag_coeff` (through the OCP *dynamics*, not the grip) and the mechanical
   load parameters do. Bridging `CarParams` aero → `AeroModel` is deferred (it is
   a physics-modelling decision, not a wiring change).

Net: of the seven setup knobs, the envelope objective responds to **three**
(drag, CoG height, weight distribution), the QSS objective to **two** (drag,
lift), with only **drag shared**. Three knobs (aero balance, brake bias, tyre
stiffness) move neither model on this track.

## Part 2 — integration

`setup-optimize --inner {qss|envelope} [--nodes N]` (default `qss`, default
`N=32`). The QSS path is byte-for-byte unchanged (`InnerObjective::Qss` is the
default on `SetupEvalConfig`). The envelope path (`InnerObjective::Envelope`):

- **Per candidate:** apply the setup to the base car, **regenerate the g-g-g
  envelope** (content-hash cache under `.apex-cache/envelope` in the CLI; `None`
  in tests, no disk side effects), solve the OCP at the **fixed** `N` with the
  shared real-circuit config (`recommended_ip_config`, `constraint_tol = 5e-3`,
  1500-iter cap). `N` is fixed for the whole run — ranking is mesh-stable only at
  fixed `N`.
- **Non-converged candidates → reject with a penalty**, not the un-converged lap.
  A candidate that does not reach `eq,ineq ≤ 5e-3` returns
  `ENVELOPE_REJECT_PENALTY (1e4) + (eq + ineq)`. **Justification:** the
  un-converged lap is optimistically biased *low* by coarse-mesh over-cutting
  (e.g. 68 s where the tight solve is 86 s), so returning it would make the search
  *prefer* infeasible setups — the opposite of what we want. A large constant puts
  every non-converged candidate below every feasible one; the added residual gives
  CMA-ES a mild gradient back toward feasibility. (Envelope-generation failure
  returns the same penalty.)
- **Determinism.** Seeded CMA-ES + the deterministic envelope inner loop give a
  bitwise-reproducible argmin (`envelope_inner_optimize_is_deterministic`).

### Sanity run — the setup difference (the physics payoff)

Silverstone, calibrated, seed 42, 12 generations, QSS vs envelope (`N=32`):

| parameter | QSS-optimized | envelope-optimized | reads as |
|---|---:|---:|---|
| drag_coeff | **0.700** (min) | **0.700** (min) | both minimise drag |
| lift_coeff | **4.500** (max) | 3.091 (≈ baseline) | QSS maxes downforce; envelope blind |
| cog_height (m) | 0.284 (≈ baseline) | **0.250** (min) | **envelope lowers CoG** |
| weight_dist_front | 0.546 (≈ baseline) | **0.580** (max) | **envelope shifts load forward** |
| aero_balance_front | 0.424 | 0.430 | neither responds (noise) |
| brake_bias_front | 0.578 | 0.530 | neither responds (noise) |
| tire_radial_stiffness | 288 686 | 320 000 | neither responds (noise) |
| **improvement** | **7.007 s (6.55 %)** | **1.749 s (2.03 %)** | |

**The envelope- and QSS-optimized setups differ in exactly the load-sensitive
directions**: the envelope drives CoG height to its **minimum** (a lower CoG cuts
lateral/longitudinal load transfer, so the load-sensitive tyres keep more total
grip) and front weight distribution to its **maximum**, while the friction-circle
QSS leaves both at baseline because it cannot see them. Both agree on minimum
drag. The envelope leaves `lift_coeff` near baseline (it is blind to it), whereas
QSS pins it to maximum downforce — the aero-blindness, made concrete. The smaller
headline improvement (2 % vs 6.5 %) is expected and not the point: the envelope
cannot exploit the big aero lever, so its gains come only from the mechanical
load setup; the **direction** of the setup change is the result, not the
magnitude (which, being an un-converged lap time, is not trustworthy anyway).

### Cost

Per candidate ≈ **1.2 s** wall at `N=32` on real Silverstone (envelope
regeneration dominates; the OCP solve is ~0.7 s, ~270 IP iterations). A 12-generation
run (population 9 for the 7-D space → ~109 candidates) is ≈ **2 minutes**. The QSS
inner loop is ~1 ms/candidate, so the envelope inner loop is ~1000× costlier — the
price of the load-sensitive model. The content-hash cache only helps across exact
setup repeats, which CMA-ES rarely produces; the cost is intrinsic.

## Limitations & future work

- **Aero-blindness.** `lift_coeff` / `aero_balance_front` do not reach the
  envelope. Until a `CarParams`-aero → `AeroModel` bridge exists, the envelope
  inner loop optimises only the mechanical/load setup (plus drag) and lets the
  aero knobs drift as CMA-ES noise. On aero-dominated tracks this is a serious
  gap; the drag/CoG/weight optimisation is still valid.
- **Not mesh-converged.** Absolute lap times (and the improvement magnitudes)
  are not trustworthy — only rankings. The gate is what licenses using it as a
  CMA-ES objective at all.
- **Cost.** ~1000× the QSS inner loop. Fine for a deliberate envelope run;
  the default stays QSS.

## Reproduction

- Gate numbers: build the 8 seeded variants (`SetupSpace::f1_standard`,
  `StdRng::seed_from_u64(20260715)`, uniform over `bounds()`), solve
  `EnvelopeOcp` at `N=24` and `N=32` (`recommended_ip_config`,
  `constraint_tol = 5e-3`, 1500 iters), and feed the two lap-time vectors to
  `apex_optimizer::rank_stability::{spearman, kendall_tau}`.
- The rank-statistics themselves are unit-tested
  (`rank_stability::tests`, including the exact one-adjacent-swap-in-eight case).
- Integration: `apex-14 setup-optimize --track <silverstone> --calibrated
  --inner envelope --nodes 32 --generations 12 --seed 42`; compare to
  `--inner qss`. Tests: `setup_eval::tests::envelope_*`.
