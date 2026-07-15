# Envelope free-trajectory OCP — formulation

The minimum-lap-time optimal-control problem that optimizes the racing line
against the cached g-g-g envelope. Implemented in
`apex-optimizer::envelope_ocp`, solved by the interior-point solver
(`apex-optimizer::ipm`, see `docs/design/envelope-qss/ip-solver.md`). Design
choices and validation results are in
`docs/design/envelope-qss/free-trajectory-ocp.md`.

## Curvilinear coordinates

We work in the `s`-domain — arc length along the track centerline. The vehicle
state relative to the centerline is

- `n`  — lateral offset (m, `+` = left of the centerline),
- `xi` — heading error (rad) between the vehicle heading and the centerline
  tangent,
- `v`  — speed (m/s).

`kappa(s)` is the centerline curvature (1/m). The controls are

- `a_x`       — longitudinal acceleration command (m/s²),
- `kappa_cmd` — commanded path curvature of the vehicle's actual path (1/m).

## Kinematics: the `(1 - n·kappa)` transform

The standard curvilinear point-mass kinematics (as in `point_mass.rs`) map the
time derivatives to `s`-derivatives through the metric factor
`(1 - n·kappa)`, which is the ratio of centerline arc length to the vehicle's
arc length at offset `n`:

```
ds_vehicle = (1 - n·kappa) / cos(xi) · ... 
```

giving, per unit centerline arc length,

```
n'  = (1 - n·kappa) · tan(xi)
xi' = kappa_cmd · (1 - n·kappa) / cos(xi) − kappa
v'  = ( a_x − drag(v)/m − roll/m ) · (1 - n·kappa) / (v · cos(xi))
```

where `drag(v) = ½·ρ_air·C_d·A·v²` and `roll` is the rolling-resistance force.
`'` denotes `d/ds`. These are exactly the derivatives assembled in
`EnvelopeOcp::dynamics`, generic over the `Float` trait so the 3×5 node
Jacobian comes from one forward-mode dual-number pass (`dynamics_jac`).

## Lateral acceleration `a_y` and the envelope constraint

The acceleration the tires must supply has two components in the vehicle frame:
the longitudinal `a_x` and the **centripetal** (lateral) acceleration required
to follow a path of curvature `kappa_cmd` at speed `v`:

```
a_y = v² · kappa_cmd.
```

Derivation: a path of curvature `kappa_cmd` has instantaneous radius
`R = 1/kappa_cmd`; a point traversing it at speed `v` has centripetal
acceleration `v²/R = v²·kappa_cmd`, directed toward the center of curvature
(the vehicle-frame lateral axis). This is *exact* — no small-angle
approximation — which is why the path curvature (not, say, a steer angle) is
the chosen lateral control. The total planar acceleration magnitude and its
direction (the g-g angle) are

```
|a|   = sqrt(a_x² + a_y²)
theta = atan2(a_y, a_x).
```

The envelope constraint is that this acceleration lie inside the car's g-g-g
boundary at the current speed and vertical load, with a safety margin `eps`:

```
|a| ≤ (1 − eps) · rho(theta; v, g_z(s))          (envelope inequality)
```

where `rho(theta; v, g_z)` is the cached, C1-interpolated boundary radius from
`apex-physics::Envelope` and `g_z(s)` is the per-node effective vertical
acceleration (`GRAVITY` for a flat track; supplied by the 3D QSS machinery
otherwise). The margin `eps` (default `0.01`) both keeps the optimum strictly
interior — where `rho` is smooth — and covers the envelope's measured ~0.76 %
over-estimation (see the design doc). It is exposed as `EnvelopeOcpConfig::eps`.

### Why `kappa_cmd` and not `a_y` as the control

`a_y` is a natural candidate control since it enters the envelope constraint
directly and speed-independently. It was tried and **rejected**: with `a_y` as
the control, the heading dynamics become `xi' = (a_y/v²)·(1-n·kappa)/cos − kappa`,
so `∂xi'/∂a_y = (1-n·kappa)/(v²·cos) ≈ 1/v²` — of order `1/1600` at racing
speed. The (envelope-clamped) `a_y` then has almost no authority over the
heading defect, and the collocation equalities stall far from feasibility.
With `kappa_cmd` as the control `∂xi'/∂kappa_cmd ≈ 1`, so the dynamics defects
are directly controllable and the solver converges. The `v²` factor in
`∂|a|/∂kappa_cmd = (a_y/|a|)·v²` is instead handled inside the envelope
Jacobian analytically. See the design doc's "control parameterization" note.

## Objective

Minimize lap time, written as an integral over centerline arc length:

```
T = ∫ (dt/ds) ds = ∫ (1 − n·kappa) / (v · cos(xi)) ds,
```

the integrand being `dt/ds` — the time to traverse a unit of centerline arc at
offset `n`, heading `xi`, speed `v`. Discretized on the periodic node mesh
with spacing `ds`, `T = ds · Σ_k (1 − n_k·kappa_k)/(v_k·cos(xi_k))`. A small
control-rate regularization `Σ_k [ w_ax·(Δa_x)² + w_κ·(Δkappa_cmd)² ]`
(periodic differences) is added to suppress chatter; the weights are small
(`1e-4`, `1e-1`) and do not materially change the optimum.

## Constraints and boundary

- **Dynamics defects** (equalities): trapezoidal collocation on each interval,
  periodic (flying-lap) closure `k → (k+1) mod N`:
  `z_{k+1} − z_k − ½·ds·(f_k + f_{k+1}) = 0` for each state `z ∈ {n, xi, v}`.
  `3N` equalities.
- **Envelope inequality**: one per node, `|a|_k − (1−eps)·rho_k ≤ 0`, `N` of
  them (smoothed with a `DELTA = 1e-3` floor so `|a|` and `theta` stay
  differentiable at the origin).
- **Track-edge bounds**: `−w_right(s_k) ≤ n_k ≤ +w_left(s_k)` — the box bounds
  the interior-point solver handles natively.
- **Guards**: `v_k ≥ v_min` (keeps `1/v` finite) and `|xi_k| ≤ xi_max` (keeps
  `cos(xi) > 0`).

Variable layout per node `k` (total `5N`): `[n | xi | v | a_x | kappa_cmd]`,
blocked by field. Warm start: the fixed-line (centerline) QSS solution —
`n = 0`, `xi = 0`, `v = v_QSS(s)`, `a_x` from the QSS speed profile,
`kappa_cmd = kappa` (follow the centerline).

## Solver

The problem is handed to the primal-dual interior-point solver with a tuned
configuration (`EnvelopeOcp::recommended_ip_config`): a gentle
augmented-Lagrangian penalty ramp (`rho_growth = 3`) and slow barrier anneal
(`mu_reduction = 0.5`) so the *objective* can migrate the racing line to the
track edge before the schedule terminates — see the design doc. The solve is
bitwise deterministic.
