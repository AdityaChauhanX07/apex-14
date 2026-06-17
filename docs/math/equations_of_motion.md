# Equations of Motion — Derivations

This document records the mathematical models implemented in Apex-14. It is a reference appendix,
not a tutorial: each section states the model, the assumptions, and the derivation of the governing
equations exactly as they appear in the code. Notation is plain Unicode and ASCII math.

Symbols used throughout: `m` vehicle mass, `g` gravitational acceleration, `ρ` air density,
`Cd`/`Cl` drag/downforce coefficients, `A` frontal area, `Crr` rolling-resistance coefficient,
`Iz` yaw inertia, `Iw` wheel inertia, `R` wheel radius, `lf`/`lr` CoG-to-axle distances,
`L = lf + lr` wheelbase.

---

## 1. Point-Mass Model (2-DOF)

### 1.1 Curvilinear Coordinate Frame

The vehicle position is expressed relative to the track centerline using arc length `s` (distance
along the centerline) and lateral offset `n` (signed perpendicular distance, positive to the left),
rather than global Cartesian `(X, Y)`.

This choice is deliberate. The track boundaries are constant functions of `s` in this frame
(`-w_right(s) ≤ n ≤ w_left(s)`), and the track curvature `κ(s)` is a property of the path, not of the
vehicle state. In a global frame those same constraints would be implicit functions of `(X, Y)` that
are expensive to evaluate and differentiate. The curvilinear frame turns the optimizer's boundary
and progress constraints into simple bounds and linear expressions, which keeps the constraint
Jacobian sparse and cheap.

### 1.2 State and Control Vectors

State `x = [s, n, v, α]`:

| Symbol | Meaning                                   | Unit  |
|--------|-------------------------------------------|-------|
| `s`    | arc length along centerline               | m     |
| `n`    | lateral offset from centerline (+left)    | m     |
| `v`    | speed (magnitude of velocity)             | m/s   |
| `α`    | heading of velocity relative to tangent   | rad   |

Control `u = [F_drive, κ_cmd]`:

| Symbol    | Meaning                                  | Unit  |
|-----------|------------------------------------------|-------|
| `F_drive` | net longitudinal force (+drive, −brake)  | N     |
| `κ_cmd`   | commanded path curvature                 | 1/m   |

### 1.3 Equations of Motion

**Progress along the track.** Let the velocity make angle `α` with the local track tangent. The
component along the tangent is `v·cos(α)`. When the vehicle is offset by `n` from the centerline of a
curve with curvature `κ`, the local radius is `(1/κ − n) = (1 − n·κ)/κ`, so a point at offset `n`
traverses arc length more slowly than the centerline by exactly the factor `(1 − n·κ)`. Dividing by
this factor converts tangential ground speed into rate of centerline arc length:

```
ds/dt = v·cos(α) / (1 − n·κ(s))
```

The `(1 − n·κ)` denominator is the path-length scaling between the offset path and the centerline. It
is < 1 on the inside of a corner (offset reduces distance) and > 1 on the outside.

**Lateral motion.** The perpendicular component of velocity moves the vehicle across the track:

```
dn/dt = v·sin(α)
```

**Speed.** Newton's second law along the direction of travel, with aerodynamic drag and rolling
resistance opposing motion:

```
dv/dt = (F_drive − F_drag − F_roll) / m
F_drag = ½·ρ·Cd·A·v²
F_roll = Crr·m·g
```

**Heading.** The vehicle yaws at rate `κ_cmd·v` (commanded path curvature times speed). The track
tangent itself rotates at `κ(s)·(ds/dt)` as the vehicle advances. The heading relative to the tangent
changes at the difference:

```
dα/dt = κ_cmd·v − κ(s)·(ds/dt)
```

At the centerline (`n = 0`) and aligned (`α = 0`), `ds/dt = v` and `dα/dt = (κ_cmd − κ)·v`, so the
path follows the track when `κ_cmd = κ`.

A lower bound `v_safe = max(v, 0.1)` is used in the denominators to avoid singularities at rest.

### 1.4 Grip Circle Constraint

A tire can produce a force vector whose magnitude is bounded by the available grip. The longitudinal
force is `F_lon = F_drive` and the lateral (centripetal) force needed to follow the commanded
curvature is `F_lat = m·v²·κ_cmd`. The combined force must stay inside the friction circle of radius
`F_grip(v)`:

```
(F_lon / F_grip)² + (F_lat / F_grip)² ≤ 1
```

The grip limit is speed-dependent because downforce adds vertical load that grows with `v²`:

```
F_grip(v) = μ·(m·g + ½·ρ·Cl·A·v²)
```

This is why a downforce car can corner far faster at high speed than a static friction estimate
predicts: the normal load — and therefore the friction limit — scales with dynamic pressure.

---

## 2. Bicycle Model (3-DOF)

The single-track model collapses the two wheels of each axle onto the centerline, giving one front
and one rear virtual tire. It captures yaw dynamics and lateral load on each axle while remaining
low-dimensional.

### 2.1 Body-Frame Velocities

Velocity is decomposed into the body frame as longitudinal `vx` and lateral `vy`, with yaw rate `ωz`.
Working in the body frame keeps the tire force directions fixed relative to the chassis (the front
tire is steered by `δ` about the body x-axis), so the force balance does not require rotating every
tire force by the global heading `ψ` at each step. Only the final position update uses `ψ`.

### 2.2 Slip Angle Computation

A tire generates lateral force in response to its slip angle — the angle between the direction the
wheel points and the direction the contact patch actually travels. For each axle the hub velocity in
the body frame combines the chassis translation with the yaw-rate contribution `ω × r`:

- Front hub: longitudinal `vx`, lateral `vy + lf·ωz` (the front is ahead of the CoG by `lf`).
- Rear hub: longitudinal `vx`, lateral `vy − lr·ωz` (the rear is behind by `lr`).

The front wheel is steered by `δ`, so its slip angle subtracts the steer:

```
α_f = δ − arctan((vy + lf·ωz) / vx)
α_r =   − arctan((vy − lr·ωz) / vx)
```

Physically: the tire velocity vector does not align with where the wheel points. The difference is
the slip angle, and the tire responds with a restoring lateral force.

### 2.3 Newton-Euler Equations

Differentiating velocity expressed in a rotating frame introduces Coriolis terms. The acceleration of
the CoG in the body frame is `(dvx/dt − vy·ωz, dvy/dt + vx·ωz)`; the cross terms `vy·ωz` and `vx·ωz`
arise because the body axes themselves rotate at `ωz`. The force and moment balance is:

```
m·(dvx/dt − vy·ωz) = Fx − F_drag − Fy_f·sin(δ)
m·(dvy/dt + vx·ωz) = Fy_f·cos(δ) + Fy_r
Iz·dωz/dt          = lf·Fy_f·cos(δ) − lr·Fy_r
```

`Fy_f` and `Fy_r` are the axle lateral forces from the tire model; the steering angle `δ` projects the
front force between the body longitudinal and lateral axes. The yaw moment is the lever-arm-weighted
difference of the axle lateral forces.

### 2.4 Understeer Gradient

Linearizing the steady-state cornering response (small slip angles, `Fy ≈ Cα·α` with cornering
stiffness `Cα`) yields the understeer gradient:

```
K_us = (m / L)·(lr / Cα_f − lf / Cα_r)
```

`K_us > 0` is understeer: the car needs increasing steer with speed to hold a radius (stable, the
typical road-car setup). `K_us < 0` is oversteer (the rear loses grip first). `K_us = 0` is neutral.
The steady-state yaw rate for steer `δ` at speed `v` follows as `ωz = v·δ / (L·(1 + K_us·v²))`.

---

## 3. Four-Wheel Model (7-DOF)

The 7-DOF model resolves all four tires individually: the 3 chassis DOF of the bicycle model plus the
4 wheel-spin states `[ω_fl, ω_fr, ω_rl, ω_rr]`. This captures longitudinal slip, per-corner load
transfer, and combined-slip tire behavior.

### 3.1 Wheel Contact Patch Velocities

Each wheel hub velocity follows from rigid-body kinematics, `v_hub = v_CoG + ω × r_wheel`, with the
wheel position `r_wheel = (x_off, y_off)` in the body frame:

```
v_hub_x = vx − y_off·ωz
v_hub_y = vy + x_off·ωz
```

For the front wheels (`x_off = +lf`), the hub velocity is rotated into the steered tire frame by `δ`:

```
v_tire_x =  v_hub_x·cos(δ) + v_hub_y·sin(δ)
v_tire_y = −v_hub_x·sin(δ) + v_hub_y·cos(δ)
```

The rear wheels (`x_off = −lr`) are unsteered, so `v_tire = v_hub`. The slip angle at each wheel is
`α = −arctan(v_tire_y / max(|v_tire_x|, 1))`.

### 3.2 Slip Ratio

Longitudinal slip compares the wheel's surface speed `ωR` to the ground speed at the contact patch:

```
κ = (ω·R − v_tire_x) / max(|v_tire_x|, v_min)
```

Positive `κ` means the wheel is spinning faster than the ground (traction/wheelspin); negative means
slower (braking/lockup). The `max(|v_tire_x|, v_min)` regularization is essential: at low speed the
true denominator approaches zero and slip ratio would diverge, making the dynamics stiff and the
integrator unstable. Clamping the denominator bounds the slip ratio and keeps the system
well-conditioned near rest.

### 3.3 Wheel Spin Dynamics

Each wheel is a rotational inertia driven by engine torque and braking, and resisted by the reaction
of the tire's longitudinal force:

```
Iw·dω/dt = T_drive − T_brake − Fx·R
```

`T_drive` is the engine torque routed to that wheel by the drive distribution; `T_brake` is the brake
torque (split by brake bias and always opposing spin); `Fx·R` is the reaction torque from the tire's
longitudinal force at radius `R`.

The force chain is the physical heart of the model: engine torque spins the wheel up → the surface
speed exceeds ground speed → a positive slip ratio develops → the tire model converts slip ratio into
longitudinal force `Fx` → that force accelerates the chassis, and its reaction `−Fx·R` brakes the
wheel spin-up until equilibrium. There is no instantaneous tractive force at zero slip; force only
appears once slip develops, which is why the first instant of applied torque shows wheel spin-up but
near-zero chassis acceleration.

### 3.4 Four-Corner Load Transfer

Static and aerodynamic loads are split front/rear, then lateral acceleration transfers load across
each axle. The transfer per axle is distributed by the roll-stiffness fraction `ε` (front share of
total roll stiffness):

```
ΔFz_front = m·a_y·h_cog·ε       / track_front
ΔFz_rear  = m·a_y·h_cog·(1 − ε) / track_rear
```

with the four loads `Fz = static ± aero ± ΔFz`, each clamped at 0 (a wheel can lift). The roll-stiffness
split is the primary handling-balance lever: a stiffer front anti-roll bar moves more lateral transfer
to the front axle, which — because of load sensitivity (Section 4.2) — costs the front more grip and
pushes the balance toward understeer. The reverse biases toward oversteer. Total grip always falls
under load transfer, so the split sets *where* grip is lost, not whether.

---

## 4. Pacejka Magic Formula

### 4.1 The Formula

The tire force as a function of a slip quantity `x` (slip angle for lateral, slip ratio for
longitudinal):

```
F = D·sin(C·arctan(B·x − E·(B·x − arctan(B·x))))
```

Coefficient roles:

- `B` — stiffness factor: sets the initial slope at the origin (`dF/dx|₀ = B·C·D`, the cornering or
  longitudinal stiffness).
- `C` — shape factor: controls how far the curve bends over and where it asymptotes.
- `D` — peak factor: the maximum force, `D = μ·Fz`.
- `E` — curvature factor: shapes the region near the peak, tuning where and how sharply the force
  rolls off after the maximum.

The `sin(arctan(...))` structure produces the characteristic shape: a near-linear rise, a smooth
peak, and a gentle decline at large slip. This decline — force *dropping* past the peak — is why a
locked or spinning tire produces less force than one at optimal slip.

### 4.2 Load Sensitivity

The effective friction coefficient falls as vertical load rises above a nominal value:

```
μ_eff = μ₀·(1 − κ_μ·(Fz − Fz_nom) / Fz_nom)
```

Real tires are less efficient per newton of load as they are pressed harder: doubling `Fz` less than
doubles the peak force. The direct consequence is that load transfer *always* costs total grip. If a
pair of tires shares load equally they jointly produce more lateral force than the same pair with load
shifted to one side, because the heavily loaded tire gains less than the lightly loaded one loses.
This single property is why managing weight transfer is central to vehicle dynamics.

### 4.3 Combined Slip

A tire generating both lateral and longitudinal force cannot reach its pure-slip peak in either
direction; the resultant is bounded by the friction circle. Apex-14 uses the friction-ellipse
(similarity) method: compute the pure-slip forces `Fx0`, `Fy0` independently, then if their resultant
exceeds the limit `F_max = μ_avg·Fz`, scale both proportionally back onto the circle:

```
scale = F_max / sqrt(Fx0² + Fy0²)        (applied only when the resultant exceeds F_max)
Fx = Fx0·scale,   Fy = Fy0·scale
```

This captures the essential coupling — using grip for braking leaves less for cornering, and vice
versa (the basis of trail braking) — while remaining cheap to evaluate and differentiate.

---

## 5. Automatic Differentiation

### 5.1 Dual Numbers

A dual number carries a value and a derivative as `a + b·ε`, where `ε` is a nilpotent unit with
`ε² = 0`. Substituting into the Taylor expansion of any smooth function `f`:

```
f(a + b·ε) = f(a) + f'(a)·b·ε + ½·f''(a)·(b·ε)² + ...
           = f(a) + b·f'(a)·ε              (all higher terms vanish since ε² = 0)
```

So evaluating `f` on `a + 1·ε` returns `f(a)` in the real part and `f'(a)` in the dual part exactly —
no truncation, no step-size choice. Arithmetic operators propagate the chain rule automatically:
multiplication gives `(a + bε)(c + dε) = ac + (ad + bc)ε` (the product rule), division gives the
quotient rule, and elementary functions are defined by `g(a + bε) = g(a) + b·g'(a)·ε`. Composing these
operations threads the chain rule through arbitrarily complex expressions with machine-precision
derivatives.

### 5.2 The Float Trait

Rather than maintaining separate code for values and derivatives, the physics is written once, generic
over a `Float` trait that abstracts `+ − × ÷`, `sin`, `cos`, `sqrt`, `atan`, and so on. Both `f64` and
the dual-number type `Dual` implement `Float`. Calling the generic dynamics with `f64` arguments
computes forces; calling the identical code with `Dual` arguments computes forces *and* their exact
partial derivatives. This guarantees the derivative code can never drift out of sync with the value
code — there is only one implementation — and it eliminates the truncation error and step-size tuning
of finite differences.

---

## 6. Direct Collocation

### 6.1 Problem Formulation

The minimum-time problem is transcribed into a nonlinear program (NLP) over a discretization of `N`
nodes. The decision variables are the state and control at each node plus the inter-node time steps
`dt_k`. The NLP is:

```
minimize   Σ dt_k
subject to  trapezoidal dynamics defects = 0      (equality)
            track boundaries: −w_right ≤ n ≤ w_left   (inequality)
            grip circle ≤ 1                        (inequality)
            periodicity: x_0 = x_N for closed loops (equality)
            variable bounds (v > 0, dt > 0, force limits)
```

Minimizing total time `Σ dt_k` with the dynamics enforced as constraints is the direct transcription
of optimal control to finite-dimensional optimization.

### 6.2 Trapezoidal Defects

Continuous dynamics `dx/dt = f(x, u)` are enforced between consecutive nodes by the trapezoidal rule.
For each interval `k`:

```
x_{k+1} − x_k − (dt_k/2)·(f_k + f_{k+1}) = 0
```

where `f_k = f(x_k, u_k)`. Trapezoidal collocation is second-order accurate, implicit (better
stability than explicit forward Euler for the stiff slip dynamics), and produces a defect that depends
only on the two endpoint nodes — keeping the constraint Jacobian sparse. A defect vector of zero means
the discrete trajectory is dynamically consistent.

### 6.3 Banded Jacobian Structure

The defect for interval `k` depends only on the variables at nodes `k` and `k+1` (the two states, two
controls, and the time step `dt_k`). Every other partial derivative is structurally zero. The
equality-constraint Jacobian is therefore banded: each block row touches at most 13 columns.

This structure is what makes exact auto-diff cheap. A dense finite-difference Jacobian would require
`2·n_vars` constraint evaluations (perturbing every variable). Exploiting the band, each interval is
differentiated with 13 dual-number evaluations — one per local variable — independent of the total
problem size. The result is an exact, sparse Jacobian assembled directly into CSR form, which gave a
roughly 25× speedup of the optimizer over the finite-difference implementation while removing its
truncation error.

A subtlety: the track curvature `κ(s)` depends on the state `s`. The dual-number sweep holds `κ` fixed
per node, so the `s`-columns receive an explicit correction term `−(dt/2)·(∂f/∂κ)·(dκ/ds)`. On
constant-curvature stretches `dκ/ds ≈ 0` and the correction vanishes; at corner entry and exit it is
significant and is required for the Gauss-Newton step to be accurate.
