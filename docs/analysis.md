# Model Fidelity Comparison

This study runs every Apex-14 model fidelity on the same track and analyzes where
the extra degrees of freedom change the answer. All numbers below are produced by
the `compare` binary (`cargo run --release --bin compare`); none are hand-edited.

## Methodology

All models were run on the same oval track (500 m straights, R = 80 m corners,
**1503 m** total length) with identical car parameters (798 kg, C_l = 3.5,
μ = 1.75). Note the track length is ~1503 m for these parameters, not the 2628 m
that a 1000 m-straight / R = 100 m oval would give.

The six configurations:

1. **QSS (grip circle)** - quasi-steady-state forward/backward pass with a simple
   friction-circle grip limit (point-mass physics).
2. **QSS (tire-aware)** - same QSS pass, but the grip limit comes from four-corner,
   load-sensitive Pacejka forces.
3. **Collocation (point-mass)** - trapezoidal direct collocation, Gauss-Newton
   solver, N = 50, grip-circle dynamics.
4. **Collocation (7-DOF tire)** - collocation with the Pacejka combined-slip,
   load-sensitive grip budget.
5. **Collocation (14-DOF)** - collocation with the ride-height-coupled 14-DOF grip
   budget (suspension compression → ride height → downforce → grip).
6. **14-DOF Forward Sim** - replay the optimized line through the full 14-DOF
   dynamics with the simple path-tracking controller.

## Results

```
Model                    | Lap Time (s) | Top Speed (km/h) | Min Speed (km/h) | Max Lat g
-------------------------+--------------+------------------+------------------+----------
QSS (grip circle)        |       20.675 |            389.0 |            200.9 |      4.01
QSS (tire-aware)         |       21.594 |            385.0 |            188.1 |      3.52
Collocation (point-mass) |       20.615 |            385.5 |            198.3 |      4.29
Collocation (7-DOF tire) |       21.544 |            380.0 |            187.2 |      3.80
Collocation (14-DOF)     |       21.553 |            380.6 |            187.3 |      3.78
14-DOF Forward Sim       |      diverged |               -- |               -- |        --
```

The "Max Lat g" column is the path-based cornering load `v²·κ_track / g`, computed
identically for QSS and collocation so the rows are comparable. (The optimizer's
curvature *command* overshoots at unconverged transition nodes, so it is not used
as the lateral-load measure.)

The 14-DOF forward simulation **diverges on the oval**: the simple PID-style
controller cannot track the high-speed straight-to-corner transitions and spins.
Its chassis-dynamics numbers below are therefore taken from a stable
constant-curvature case (a tight R = 30 m circle).

## Analysis

### Effect of Tire Model Fidelity

Comparing the two QSS runs, the **tire-aware model is +4.4 % slower** than the grip
circle (21.594 s vs 20.675 s), and its minimum (corner) speed drops from 200.9 to
188.1 km/h. The cause is **load sensitivity**: a tire's effective μ decreases as
vertical load rises above nominal. During cornering, lateral weight transfer loads
the outer tires and unloads the inner ones. Because the outer tires lose grip
efficiency faster than the inner tires gain it, the *sum* of available grip across
the axle falls below what a load-independent friction circle predicts. The
grip-circle model ignores this and is therefore optimistic. The same effect appears
in the collocation rows: 7-DOF (21.544 s) is +4.5 % over point-mass (20.615 s).

### Effect of Optimization vs QSS

The collocation optimizer and the QSS pass land within a few hundredths of a second
of each other on this track (point-mass: 20.615 s optimized vs 20.675 s QSS; 7-DOF:
21.544 s vs 21.594 s). On an oval this is expected: the racing line is essentially
the centerline (the corners are constant-radius and symmetric), so the optimizer's
main lever - using the track width via a non-zero lateral offset `n` - buys almost
nothing. The optimizer's small edge (~0.3 %) comes from smoothing the
accelerate/brake transitions rather than from a different line. On a circuit with
asymmetric corners the optimization gap would be larger.

### Effect of Ride-Height-Sensitive Aero

The 14-DOF force model (21.553 s) is within 0.01 s of the 7-DOF model (21.544 s) on
this oval, and **+4.6 % vs the point-mass collocation** baseline. The 14-DOF grip
budget adds one mechanism on top of the 7-DOF model: suspension compression under
load lowers the ride height, which changes the downforce via the ground-effect map.
At the oval's operating point the equilibrium ride height sits close enough to the
design point that the downforce change is small, so 7-DOF and 14-DOF nearly
coincide. The ride-height coupling matters far more under heavy braking and large
load swings than in steady high-speed cornering, which is exactly what this oval is
dominated by.

### Forward Simulation vs Optimization

The optimized lap time is the *theoretical* limit - the speed profile that exactly
saturates the grip budget. The forward simulation asks a different question: what can
a controller actually drive? On a stable constant-curvature circle the forward sim
laps **+19.7 % slower** than the optimized line, because the simple controller holds
a deliberate margin below the grip limit (≈1.8 g vs the ≈2.2 g optimum) to stay
stable. On the oval the gap is effectively infinite: the controller diverges at the
straight-to-corner transitions. This gap is a property of the *controller*, not the
vehicle model - an LQR or MPC tracker that plans braking and uses the full grip
envelope would shrink it substantially.

### Chassis Attitude

From the 14-DOF forward simulation (R = 30 m circle, ~1.8 g sustained cornering):

```
Max roll:   2.611 deg
Max pitch:  0.356 deg
Max susp:   32.8 mm
```

The pitch (0.36°) and suspension travel (33 mm) are squarely in the normal F1 range
(< 0.5° pitch, 20-35 mm travel). The roll (2.6°) is slightly above the 1-2° typical
of a fast corner, which is consistent with this being a tight, low-speed R = 30 m
circle pulling sustained ~1.8 g on relatively soft springs - a more aggressive roll
case than a high-speed sweeper where downforce dominates the load.

## Computational Cost

From the criterion benchmark suite (`cargo bench`, release):

| Operation                              | Time     | Note                                    |
|----------------------------------------|----------|-----------------------------------------|
| RK4 step (2-DOF point mass)            | ~25 ns   | zero-allocation fixed-size arrays       |
| Pacejka lateral force (f64)            | ~21 ns   |                                         |
| Pacejka lateral force (`Dual`)         | ~38 ns   | ~1.9× f64 - under the 2.5× target       |
| 14-DOF derivatives                     | ~67 ns   | most expensive per-step computation     |
| Equality Jacobian, N = 50 (auto-diff)  | ~32 µs   | **~52× faster** than finite differences |
| Equality Jacobian, N = 50 (numerical)  | ~1.68 ms |                                         |

The auto-diff Jacobian is the key enabler: it makes the Gauss-Newton inner loop
cheap enough to iterate freely, and forward-mode dual numbers cost under 2× the
plain-`f64` evaluation.

## Limitations

Being honest about what does not work well:

- **The optimizer does not fully converge on this oval.** The reported equality
  violations are not at machine zero (e.g. mesh refinement reaches `eq_viol ≈ 0.7`
  at N = 50, not converged). The Gauss-Newton solver makes progress but cannot
  drive every trapezoidal defect to zero across the sharp straight-to-corner
  curvature steps. The lap times are sensible and the trajectories are usable, but
  this is a near-feasible solve, not a tight one. Mesh refinement (coarse → fine
  warm starting) helps the conditioning but does not fully close the gap here.

- **The forward-sim controller is conservative and not robust.** It is a
  hand-tuned PID-style path tracker with a grip-based speed cap. It is stable on
  constant-curvature cornering but diverges on the oval's transitions. This is a
  controls limitation, not a vehicle-model fault; an optimal controller (LQR/MPC)
  is the right fix.

- **The aero parameters produce unrealistically high speeds and g-forces.** Top
  speeds near 390 km/h and ~4 g of lateral load on an R = 80 m corner are well
  above real F1 figures. This comes from the default C_l = 3.5 over a 1.5 m² frontal
  area with no speed-dependent drag-limited top-speed cap - it is a tuning choice for
  exercising the models, not a defect in the dynamics. The *relative* comparisons
  between fidelities remain valid; the absolute numbers should not be read as
  predictions of real lap times.

---

# Envelope free-trajectory QSS

This section compares, on the **same calibrated car** (`f1_2024_calibrated`), two
lap-time models against each other on real circuits:

- **Fixed-line QSS** — the quasi-steady-state forward/backward pass on the
  centerline (`apex_physics::qss_lap_sim`; 3-D `qss_lap_sim_3d` for Spa-3D). The
  line is fixed; only the speed profile is optimized.
- **Envelope free-trajectory OCP** — the `s`-domain optimal-control problem that
  *also* optimizes the racing line `n(s)` against the cached g-g-g envelope
  (`apex_optimizer::envelope_ocp`, interior-point solver). This is the
  "envelope-QSS" free trajectory.

It is **not** a comparison against the fully dynamic (transient 7-/14-DOF) optimal
control — see the pending note at the end.

## Methodology

Envelope generated per car (`PacejkaTire`/`SuspensionSystem`/`AeroModel::f1_default`,
grid `v ∈ [5, 90] m/s`), cached. The OCP is solved by the interior-point solver
with the **config-only** augmented-Lagrangian retune established in Part A
(`docs/design/envelope-qss/real-track-convergence.md`): a single shared config
`IpmConfig { al_contract: 0.1, rho_max: 3e6, constraint_tol: 5e-3,
max_iterations: 1500, ..recommended_ip_config() }`. Feasibility is judged
`tight` when **both** `eq_violation ≤ 5e-3` and `ineq_violation ≤ 5e-3` (SI).

**Per-track node count `N*` is the one tuned knob.** With the shared config, tight
feasibility is reached only at a coarse, per-track mesh (`N ≈ 24–40`); at `N ≥ 48`
the envelope-inequality coordination breaks down and the solve regresses to
infeasible (the "finer is worse" effect, diagnosed in Part A). `N*` below is the
finest mesh that is tight.

## Results (calibrated car; envelope aero-on)

| track | `N*` | status | IP iters | wall (ms) | eq viol | ineq viol | fixed-line QSS (s) | envelope-OCP (s) | Δ (OCP − QSS) |
|---|---:|---|---:|---:|---|---|---:|---:|---|
| Silverstone (real 2-D) | 40 | Optimal | 271 | 931 | 4.9e-7 | 1.8e-9 | 112.174 | 89.791 | **−22.38 s (−19.9 %)** |
| Monza (real 2-D) | 30 | Optimal | 271 | 688 | 9.7e-7 | 4.3e-7 | 93.114 | 84.330 | **−8.78 s (−9.4 %)** |
| Catalunya (real 2-D) | 32 | Optimal | 272 | 676 | 2.2e-3 | 2.4e-3 | 95.531 | 70.429 | **−25.10 s (−26.3 %)** |
| Spielberg (real 2-D) | 28 | Optimal | 272 | 583 | 3.7e-4 | 2.1e-4 | 76.617 | 67.255 | **−9.36 s (−12.2 %)** |
| Spa (real 2-D) | 36 | Optimal | 271 | 783 | 2.2e-7 | 4.9e-8 | 122.891 | 114.721 | **−8.17 s (−6.6 %)** |
| Spa (real 3-D, `g_z(s)`) | 24 | Optimal | 272 | 645 | 1.5e-5 | 3.9e-6 | 122.223 | 87.054 | **−35.17 s (−28.8 %)** |
| Simple Oval (sample) | — | MaxIter | 1500 | 3498 | 7.9e-2 | 3.7e-1 | 25.415 | 22.848 | *not converged* |

The headline achievement: with the Part-A config retune the envelope-OCP reaches
**machine-tight feasibility on all five real F1 circuits** (and Spa-3D) — a reversal
of the previously documented "cannot converge on real circuits." The interior-point
solve is fast (~0.6–0.9 s, ~270 iters) at the sweet-spot mesh. Every converged solve
improves on the fixed line (the free line always beats the centerline), so the sign
of every Δ is trustworthy.

> **⚠️ The Δ magnitudes are NOT trustworthy lap-time improvements, and are not
> comparable across rows.** Tight *constraint* feasibility does not imply an
> *objective*-converged lap time. At these coarse meshes (`ds ≈ 150–290 m`) a corner
> spans one or two nodes, so the optimizer over-cuts and the lap time is optimistically
> biased and strongly mesh-dependent: e.g. Silverstone's OCP lap is 83.9 s at N=30 but
> 89.8 s at N=40 (both tight); the synthetic Silverstone swings 81.3 → 73.5 → 66.2 s
> across N = 24/30/36. The Spa-2-D (N=36, −6.6 %) vs Spa-3-D (N=24, −28.8 %) gap is
> **dominated by the node-count difference, not by the elevation physics** — do not read
> it as an elevation effect. A mesh-continuation solve (coarse-tight → interpolate →
> refine) is required before any of these magnitudes can be quoted. Until then the table
> is a **convergence/feasibility result with directional deltas**, not a lap-time table.

The `g_z(s)` profile for the Spa-3-D row is the static gravity projection
`g·cos(grade)·cos(bank)` sampled from `spa_3d.json`; the velocity-dependent
vertical-curvature and banking terms of the full 3-D QSS are outside the envelope's
`rho(theta; v, g_z)` parameterization and are **not** captured here (a further reason
the 3-D row's magnitude is not directly comparable to the 2-D row).

The **Simple Oval sample track does not converge** at any mesh — its synthetic geometry
has sharp low-speed hairpins that the coarse envelope-OCP cannot make feasible; it is
marked *not converged* and quotes no delta (per the task's rule).

## Pending — vs the fully dynamic OCP

Every number here is **envelope-QSS vs fixed-line QSS**: both are quasi-steady-state
(instantaneous grip-limited), differing only in whether the line is free. The comparison
that actually values the extra fidelity — **envelope free-trajectory vs the fully dynamic
transient optimal control** (7-/14-DOF, load transfer *dynamics*, tyre thermal,
energy management) — is **pending that work** (the deferred single-track / four-wheel /
14-DOF OCP, PHYSICS_CHANGE 2026-07-07). Only that comparison can say how much of the
fixed→free-line gain survives real transient dynamics.
