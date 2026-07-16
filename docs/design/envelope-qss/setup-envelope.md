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

2. **Aero-blindness — RESOLVED (2026, aero-bridge task; see §"Aero bridge"
   below).** As originally integrated, `lift_coeff` was QSS's single biggest
   lever (14.4 s) but the envelope barely responded (0.4 s), because the
   point-mass QSS computes downforce from `CarParams::downforce(v)` while the
   envelope's downforce came from the **independent `AeroModel`** held at
   `f1_default`. The [`AeroModel::scaled_for_car`] bridge now scales the
   `AeroModel` to the car's `CarParams` aero, so `lift_coeff` (0.4 → **6.2 s**)
   and `aero_balance_front` (0.0 → **0.35 s**) reach the envelope. The pre-bridge
   table above is kept as the "before"; the post-bridge table is in the aero-
   bridge section.

Net (**pre-bridge**): the envelope objective responded to three knobs (drag, CoG
height, weight distribution), QSS to two (drag, lift), only drag shared. **Post-
bridge** the envelope also responds to lift and aero balance, so envelope and QSS
now share the two big aero levers (drag, lift) while the envelope keeps its
exclusive load-sensitive levers (CoG, weight).

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

## Aero bridge — resolving the aero-blindness (2026, aero-bridge task)

[`AeroModel::scaled_for_car`] (crate `apex-physics`, `aero.rs`) multiplicatively
scales the ride-height `AeroModel` so its **effective coefficients match a car's
`CarParams` aero** (`lift_coeff`, `drag_coeff`, `aero_balance_front`). Envelope
generation now consumes the bridged model at its two committed consumers — the
setup inner loop (`evaluate_setup_envelope`) and the CLI `optimize --envelope`
path — so a `CarParams`-only aero change reaches the envelope.

**Reference condition — the design ride height.** The `AeroModel`'s coefficients
are nominal at `design_ride_height` (where `ride_height_factor == 1.0`); that is
where `cl_front_base`, `cl_rear_base`, `cd_base` are literally defined, so it is
the natural reference. There: `ref_lift = cl_front_base + cl_rear_base`,
`ref_balance = cl_front_base/ref_lift`, `ref_drag = cd_base`. The bridge sets
`lift_scale = car.lift_coeff/ref_lift`, `drag_scale = car.drag_coeff/ref_drag`,
and per-axle balance ratios `bf = aero_balance/ref_balance`,
`br = (1−aero_balance)/(1−ref_balance)` (identity when the balance matches).
`frontal_area`/`air_density` are not bridged — every config here shares
`area = 1.5, ρ = 1.225`, so matching the coefficients matches the force
`q·area·cl` (recorded; fold in the ratio if a car ever diverges).

**Byte-stability — the anchor is the DEFAULT car, not the calibrated car (a
finding).** The task expected the *calibrated* car to bridge to scales exactly
1.0. It does **not**: `f1_2024_calibrated` aero is `lift 2.80, drag 1.10, balance
0.44`, which does **not** equal the `f1_default` reference `3.5 / 0.9 / 0.45`. It
is **`CarParams::default`** whose aero *is* the reference (`1.575 + 1.925 = 3.5`,
`0.9`, and `1.575/3.5 == 0.45` bit-for-bit). So:

- **Default car:** all three scales are exactly `1.0`, every coefficient is
  `x * 1.0 == x` (IEEE-754 exact), and the bridged `AeroModel` — and the whole
  generated envelope — is **byte-identical** to pre-bridge. Locked by
  `aero::bridge_default_car_is_bit_identical` and
  `envelope::bridge_default_car_envelope_is_byte_identical` (key + `to_bytes`).
- **Calibrated car:** the bridge *legitimately changes* its envelope (its true
  `lift 2.80 < 3.5` means less downforce than the old fixed `f1_default` gave it).
  This is the fix working, not a regression. Locked by
  `bridge_calibrated_car_changes_envelope_key`.

This is reported, not forced: making the calibrated car bridge to identity would
require altering its aero (breaking its own QSS goldens) or the reference — both
wrong. The envelope content hash already distinguishes cars by
`car_params_hash`, which **includes** `lift_coeff`/`drag_coeff`/`aero_balance_front`
(verified against the frozen field-sensitivity vector), so no hash change was
needed.

**Post-bridge per-parameter sensitivity** (Silverstone, calibrated, N=32):

| setup parameter | Δ QSS (s) | Δ env pre-bridge (s) | Δ env **post-bridge** (s) |
|---|---:|---:|---:|
| drag_coeff | 3.47 | 3.94 | 3.94 |
| **lift_coeff** | 14.39 | 0.40 | **6.20** |
| **aero_balance_front** | 0.00 | 0.00 | **0.35** |
| cog_height | 0.00 | 0.66 | 0.66 |
| weight_dist_front | 0.00 | 0.46 | 0.46 |
| brake_bias_front | 0.00 | 0.00 | 0.00 |
| tire_radial_stiffness | 0.00 | 0.00 | 0.00 |

`lift_coeff` now moves the envelope by **6.2 s** (was 0.4). It is below QSS's
14.4 s — expected: envelope downforce feeds the **load-sensitive tyre trim**
(diminishing grip at high load) and only re-grips the cornering-limited sections,
whereas the friction circle scales grip with downforce everywhere. `aero_balance`
now bites (0.35 s) via the front/rear split; the load-sensitive levers (CoG,
weight) are unchanged.

**Post-bridge sanity re-run** (Silverstone, calibrated, seed 42, 12 gen, N=32):

| parameter | QSS-optimized | env-opt (pre-bridge) | env-opt (**post-bridge**) |
|---|---:|---:|---:|
| drag_coeff | 0.700 (min) | 0.700 | 0.700 (min) |
| lift_coeff | 4.500 (max) | 3.091 (blind) | **4.500 (max)** |
| cog_height (m) | 0.284 | 0.250 | **0.259** (low) |
| weight_dist_front | 0.546 | 0.580 | **0.580** (max) |
| aero_balance_front | 0.424 | 0.430 | 0.400 (min) |
| improvement | 7.007 s | 1.749 s (2.0 %) | **4.053 s (4.7 %)** |

The envelope now **drives `lift_coeff` to its maximum**, agreeing with QSS on the
big aero levers (lift↑, drag↓) — the aero-blindness is gone. It **still differs
from QSS in the load-sensitive directions** (CoG 0.259 vs 0.284, weight 0.580 vs
0.546), so the load-transfer payoff survives *on top of* correct aero. The
improvement rose 2.0 → 4.7 % because the envelope can finally exploit downforce.

## Limitations & future work

- **Not mesh-converged.** Absolute lap times (and the improvement magnitudes)
  are not trustworthy — only rankings. The gate is what licenses using it as a
  CMA-ES objective at all.
- **Cost.** ~1000× the QSS inner loop. Fine for a deliberate envelope run;
  the default stays QSS.
- **Bridge scope.** The bridge matches the effective (design-ride-height)
  coefficients; it does not re-parameterise the car's aero into a full
  ride-height *map* (front/rear ride-height sensitivity stays at the `f1_default`
  shape). Adequate for the setup levers; a fitted ride-height map is future work.

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
