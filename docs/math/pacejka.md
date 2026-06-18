# Pacejka Magic Formula - Derivation and Implementation

This document records the tire force model implemented in Apex-14. It is a reference appendix:
each section states the model, its physical meaning, and how it maps to the code. Notation is plain
Unicode and ASCII math, consistent with `equations_of_motion.md`.

Symbols: `Fz` vertical (normal) load, `α` slip angle (rad), `κ` slip ratio (dimensionless),
`μ` friction coefficient, `B`/`C`/`D`/`E` the Magic Formula coefficients.

---

## 1. The Standard Form

The Pacejka Magic Formula maps a single slip quantity `x` (slip angle for lateral force, slip ratio
for longitudinal force) to a force:

```
F(x) = D · sin(C · arctan(B·x − E·(B·x − arctan(B·x))))
```

The four coefficients each shape one aspect of the curve:

- **B - stiffness factor.** Sets the slope at the origin. A larger `B` makes the tire respond more
  sharply to small slip. It is tied to the cornering stiffness through `C_α = B·C·D` (the slope of
  `F` vs `x` at `x = 0`), which the code exposes as `PacejkaTire::cornering_stiffness`.
- **C - shape factor.** Controls the overall character of the curve, in particular how much the
  force falls off after the peak. `C ≈ 1.3` gives a smooth, rounded peak; `C ≈ 1.65` gives a sharper
  peak with more post-peak drop. Lateral tires are typically `C ∈ [1.3, 1.7]`, longitudinal tires
  `C ∈ [1.5, 1.8]`.
- **D - peak factor.** The maximum force the curve can produce: `D = μ · Fz`. Because `sin(·)` is
  bounded by 1, `D` is the absolute ceiling. It scales linearly with vertical load - before load
  sensitivity (Section 3) is applied.
- **E - curvature factor.** Adjusts the shape near and past the peak by warping the argument of the
  outer `arctan`. `E < 0` widens the plateau around the peak (the tire holds near-peak force over a
  larger slip range); `E > 0` sharpens it. Typical range `E ∈ [−2, 1]`.

Apex-14's representative F1 coefficients (`PacejkaCoeffs::f1_lateral` / `f1_longitudinal`):

| Set          | B    | C    | μ    | E    |
|--------------|------|------|------|------|
| Lateral      | 12.0 | 1.50 | 1.75 | −0.5 |
| Longitudinal | 14.0 | 1.65 | 1.70 | −0.3 |

The longitudinal curve is stiffer (higher `B`) and sharper (higher `C`), as real tires are.

---

## 2. The Characteristic Curve Shape

The formula produces the classic three-region force curve:

- **Linear region (small slip).** For small `x`, `arctan(B·x) ≈ B·x`, the inner term collapses to
  `B·x`, and `F ≈ D·C·B·x = C_α·x`. The tire behaves as a linear spring: force is proportional to
  slip, with slope equal to the cornering stiffness.
- **Transition region.** As slip grows the inner `arctan` saturates and the curve bends over toward
  the peak `D`. This is where most of the interesting handling lives - the tire is near, but not yet
  at, its grip limit.
- **Saturation region (large slip).** Past the peak the `sin(C·…)` term turns over and the force
  *decreases* with increasing slip. Physically the contact patch is sliding; demanding more slip
  angle buys less force, not more. `C` and `E` set how steep this falloff is.

This shape is why a vehicle has a well-defined grip limit and why exceeding it (too much steer or
brake) reduces, rather than increases, the force available.

---

## 3. Load Sensitivity

A real tire makes more force under more load, but *less efficiently* - the friction coefficient
itself drops as the contact patch is pressed harder. Apex-14 models this with a linear falloff of
the effective μ about a nominal load `Fz_nom`:

```
μ_eff(Fz) = μ₀ · (1 − κ_μ · (Fz − Fz_nom) / Fz_nom)
```

with `κ_μ` the load-sensitivity coefficient (`PacejkaTire::load_sensitivity`, default `0.1`) and
`Fz_nom` the nominal load (`fz_nominal`, default `4000 N`). The peak force used in the Magic Formula
becomes `D = μ_eff(Fz) · Fz`, so it is no longer exactly linear in load - it bends downward.

The consequence is the single most important fact in this whole model: **weight transfer always
costs total grip.** During cornering, lateral load transfer adds `ΔFz` to the outer tire and removes
`ΔFz` from the inner. Because `μ_eff` decreases with load, the heavily loaded outer tire gains less
grip than the lightly loaded inner tire loses. Summed across the axle, the available lateral force
is below what a load-independent friction circle would predict. This is why the tire-aware QSS lap is
slower than the grip-circle lap (see `docs/analysis.md`), and why a softer-rolling, flatter car
corners better - it keeps the load more evenly split.

---

## 4. Combined Slip

A tire has one contact patch and one friction budget; it cannot independently produce maximum lateral
and maximum longitudinal force at the same time. The constraint is the **friction circle** (or
ellipse): the resultant force vector must stay within a bounded region set by the peak grip.

```
√(Fx² + Fy²) ≤ F_max
```

Using longitudinal grip (braking or traction) therefore reduces the lateral grip still available, and
vice versa - the geometric basis of trail braking and of getting on power before the apex.

Apex-14's implementation (`PacejkaTire::combined_forces`) follows the friction-circle / similarity
approach:

1. Compute the pure lateral force `Fy0(α)` and pure longitudinal force `Fx0(κ)` independently with
   the Magic Formula.
2. Form the resultant `r = √(Fx0² + Fy0²)`.
3. If `r` exceeds the available grip `F_max`, scale both components by `F_max / r` so the vector lands
   on the friction-circle boundary; otherwise leave them unchanged.

This preserves the *direction* of the demanded force while clipping its *magnitude* to the physically
available limit.

---

## 5. The Smooth Saturation

The hard "scale only if `r > F_max`" rule in Section 4 is a piecewise function with a kink at the
boundary. The force is continuous there, but its derivative is not: the scale factor is `1` just
inside the limit and `F_max/r` just outside. A discontinuous Jacobian is poison for a gradient-based
optimizer - the Gauss-Newton step direction jumps as a node crosses the limit, and convergence
stalls or chatters.

`PacejkaTire::combined_forces_smooth` replaces the hard clamp with a smooth minimum. The limited
resultant is `r_lim = smooth_min(r, F_max, k)`, where

```
smooth_min(a, b, k) = −ln(e^(−k·a) + e^(−k·b)) / k
```

This is the log-sum-exp soft-min: it is C∞ (infinitely differentiable) everywhere, approaches the
hard `min(a, b)` as the sharpness `k → ∞`, and rounds the corner over a width ~`1/k` near the
boundary. The code evaluates it via the numerically stable log-sum-exp form (factor out the max
exponent so `exp` never overflows). The forces are then scaled by `r_lim / r`. The cost is a small,
deliberate under-shoot of the limit just below the boundary - a fraction of a percent - in exchange
for a Jacobian that is smooth across the grip limit.

---

## 6. Implementation in Apex-14

The model lives in `crates/apex-physics/src/tire/`:

- `PacejkaCoeffs` holds `{ b, c, mu, e }` for one force channel; `D` is computed at evaluation time
  as `μ_eff·Fz` rather than stored, so a single coefficient set works at any load.
- `PacejkaTire` bundles the lateral and longitudinal coefficient sets plus `load_sensitivity` and
  `fz_nominal`.
- `magic_formula`, `lateral_force`, `longitudinal_force`, and `effective_mu` are the `f64` entry
  points; `combined_forces` / `combined_forces_smooth` add the friction-circle coupling.

Every force function has a `*_generic<T: Float>` twin (`magic_formula_generic`,
`combined_forces_smooth_generic`, …). The `Float` trait is implemented by both `f64` and `Dual`, so
the *same* tire code computes plain forces with `f64` and exact force Jacobians with dual numbers -
no separate, hand-differentiated derivative code to keep in sync. This is what lets the collocation
optimizer differentiate through the tire model exactly (see `docs/math/collocation.md`).
