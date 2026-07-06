# QSS channel inference

Back-computing the unmeasured channels (accelerations, aero loads, per-axle
vertical loads, friction-circle grip utilization, tractive / braking power) from
a measured speed trace on the reconstructed driven line. Implemented in
`apex-correlate::infer`; run via `apex-14 infer`.

> ⚠️ **Effective-parameter caveat.** Every aero/load/power channel here is
> computed from the **fitted effective car** (`cars/<circuit>_2024q_fitted.toml`).
> Those coefficients absorb the point-mass QSS's limitations (no elevation, no
> transient load transfer, a single fixed aero trim — see
> `docs/validation/correlation_summary.md`). The inferred channels are therefore
> **model-consistent estimates, not physical measurements.** This same line is
> emitted into every inferred CSV's `# inference_caveat:` header.

## Principle — invert the exact QSS model

Inference is the **inverse** of the point-mass QSS grip-circle sim
(`apex_physics::qss::qss_lap_sim`). We deliberately invert *that* model and do
**not** introduce a richer load model than the sim itself uses. When the speed
trace comes from the sim, inference reproduces the sim's own internal channels
(closed-loop test: `a_lat` exact to 1e-9, `a_long` median error < 0.02 g).

Given the driven path's signed curvature `κ(s)` and the measured speed `v(s)`
(both on the driven arc length `s`):

### a. Kinematic accelerations

```
a_lat  = v²·κ                     (signed; + = left, matching lateral_offset)
a_long = d(½v²)/ds = v·dv/ds      (signed; + = accelerating)
```

`a_lat` is exact (a pointwise product). `a_long` requires differentiating a
speed trace sampled at ~7.5 Hz — raw finite differences are unusable, so we use a
**local-linear (Savitzky-Golay-style) derivative**: at each point, a
least-squares straight line through the `(Δs, ½v²)` pairs inside a ±window
(default **±15 m**, `--`-configurable via `InferConfig::deriv_halfwindow_m`); the
slope is `a_long`. Differentiating specific kinetic energy `½v²` (rather than `v`
then multiplying) matches the QSS's own energy integration exactly. The window
wraps for the closed lap.

The driven **curvature** is a second derivative of position, so even a
position-smoothed driven line leaves high-frequency `κ` noise that `v²·κ`
amplifies into phantom high-g spikes. On the real-data path (`infer_on_driven`)
`κ` is first passed through a light periodic moving average (±10 m by default,
`InferConfig::kappa_halfwindow_m`). The core `infer_channels` never smooths `κ`,
so it stays an exact inverter for the closed-loop test.

### b. Aerodynamics

Straight from the car's aero model at `v` (identical to the QSS):

```
downforce  = ½·ρ·C_l·A·v²
aero_drag  = ½·ρ·C_d·A·v²
```

### c. Vertical loads (per axle)

Reported via `CarParams::axle_loads(v, a_long)` — static weight distribution +
aero split + **longitudinal** load transfer, exactly the QSS-parameter-level
split:

```
Fz_front_static = m·g·(cog_to_rear / L)      Fz_rear_static = m·g·(cog_to_front / L)
Fz_*_aero       = downforce · (aero balance)
ΔFz             = m·a_long·h_cog / L         (to the rear under accel)
Fz_front = Fz_front_static + Fz_front_aero − ΔFz    (≥ 0)
Fz_rear  = Fz_rear_static  + Fz_rear_aero  + ΔFz    (≥ 0)
```

**Lateral** transfer (a per-corner split) is **not** applied: the point-mass QSS
has no per-corner grip budget, so splitting grip per corner would invent a model
the sim does not use. Per-axle loads are reported as informative context; they do
not gate grip below.

### d. Grip utilization (total friction circle)

The point-mass QSS bounds the **total** tyre force by the total grip
`F_grip = μ·(m·g + downforce)`. Inference reports the same friction-circle
occupancy:

```
F_lat = m·a_lat
F_x   = m·a_long + aero_drag + rolling        (+ = drive tyre force, − = brake)
grip_util = √(F_lat² + F_x²) / F_grip
```

`F_x` is the *tyre* longitudinal force (what the friction circle limits); it
differs from `m·a_long` by the drag + rolling the tyre must also overcome. At a
corner apex or under heavy braking `grip_util → 1`; on a straight it sits well
below 1. Values **> 1 are reported, never clamped** — they flag either
measurement/reconstruction noise (curvature spikes at the tightest corners) or
model deficiency (e.g. missing elevation load).

### e. Power

```
tractive_power = max(F_x, 0)·v        (accelerating; overcoming drag + inertia)
braking_power  = max(−F_x, 0)·v       (braking; dissipated by the brakes)
```

One is zero at any instant. Braking power for an F1 car peaks well above engine
power (the brakes are the more powerful "actuator").

### f. NaN discipline

Measured gaps propagate: a channel is NaN wherever its inputs are NaN.
`a_lat`/`downforce`/`aero_drag` need only `v`/`κ`; the load / grip / power
channels also need `a_long`, so a NaN inside the derivative window makes only
those NaN.

## Closed-loop validation

Feeding a QSS speed trace back through inference reproduces the sim's own
`lateral_gs`/`longitudinal_gs` (and hence loads/power): on a circle, `a_long ≈ 0`
and `grip_util ≈ 1` (the cornering limit, plus a hair for the drag-overcoming
tyre force); on an oval with straights, `a_lat` is exact and `a_long` matches the
sim's forward difference away from the O(few) constraint-transition kinks, where
a symmetric derivative cannot match a forward difference at a C0 kink in the
piecewise-optimal speed.

## Known distortions

- **Reconstruction curvature spikes.** At the very fastest corners the direct
  driven line under-estimates radius (the same limitation as Abbey in the
  correlation work), inflating `a_lat` and pushing `grip_util > 1` at a handful
  of samples. Reported, not hidden.
- **Missing elevation (the Phase-1 gap).** On hilly circuits (Spa) the flat 2-D
  centerline omits the vertical loading from gradient/compression (Eau
  Rouge/Raidillon, Pouhon). Inferred vertical **loads are under-estimated** there
  (they miss the ~2–3 g compression), the peak `a_lat` at Raidillon is distorted,
  and `grip_util` reads systematically low (the elevation-distorted fit operates
  the effective car below its friction circle). See the campaign summary.
