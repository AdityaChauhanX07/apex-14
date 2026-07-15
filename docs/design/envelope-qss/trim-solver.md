# Operating-point trim solver

**Status: implemented.** The inner evaluation the envelope sweep will call
thousands of times: solve the 14-DOF model's quasi-steady state at an arbitrary
operating point `(v, a_x, a_y, g_z)`. Builds on the g_z pathway
([`gz-pathway.md`](gz-pathway.md)) and the straight-line trim
(`solve_trim_with_gz`). See [`recon.md`](recon.md) §4.1. **No envelope/grid code
here** — this is the point solver only.

## API

`crates/apex-physics/src/trim.rs`, re-exported from the crate root.

```rust
pub struct OperatingPoint { pub v: f64, pub a_x: f64, pub a_y: f64, pub g_z: f64 }
// Default = { v: 0, a_x: 0, a_y: 0, g_z: GRAVITY } — the static reference point.

pub fn solve_operating_point(
    car: &CarParams, tire: &PacejkaTire,
    suspension: &SuspensionSystem, aero: &AeroModel,
    op: OperatingPoint,
) -> Result<TrimResult, OperatingPointError>;

pub struct TrimResult {
    pub loads: [f64; 4],          // tire vertical loads [fl,fr,rl,rr] (N), NOT floored
    pub travel: [f64; 4],         // suspension travel (m, +compression)
    pub ride_heights: (f64, f64), // front, rear (m)
    pub attitude: (f64, f64, f64),// heave (m), pitch (rad), roll (rad)
    pub downforce: f64,           // total aero downforce at trimmed heights (N)
    pub residual: f64,            // final max-abs force-balance residual (N)
    pub iterations: usize,
    pub status: TrimStatus,
}

pub enum TrimStatus { Feasible, Infeasible(InfeasibilityReason), NotConverged { residual } }
pub enum InfeasibilityReason {
    NegativeLoad { corner, load },        // a corner lifted
    GripExceeded { demand, available },   // combined-slip budget exceeded
    PowerLimit   { demand, available },   // traction/brake actuator limit
}
pub enum OperatingPointError { NonPositiveGravity { g_z }, NonFinite }
```

The recon sketched `solve_operating_point(car, op)`; the real signature also
takes `tire`, `suspension`, and `aero`, because the 14-DOF trim needs all four
(the same set `fourteen_dof_grip_budget` takes). `OperatingPoint` is passed by
value (it is `Copy`), so an envelope sweep can build a grid of them cheaply.

## Solver approach

**Did the heave/pitch template extend to roll cleanly? Yes.** The legacy
`solve_trim_with_gz` is a symmetric 2-DOF Newton on `(z_front, z_rear)` balancing
heave + pitch. Because `a_y ≠ 0` breaks left/right symmetry, the general solver
adds roll, giving a **3-DOF Newton on chassis attitude `(heave, pitch, roll)`**.
Corner suspension travels come from a rigid-plane map — which is what makes it
clean:

```
z_i = heave − x_off_i·pitch + y_off_i·roll      // x_off = [lf,lf,−lr,−lr], y_off = [±tf/2, ±tr/2]
```

Using 3 rigid-body DOF (not 4 free corners) **enforces chassis planarity/warp for
free** — there is no 4th (twist) equation to invent. The corner loads reuse the
existing spring + anti-roll-bar model verbatim (`SuspensionSystem::forces`), so
the front/rear **roll-stiffness split emerges from the actual spring + ARB rates**
rather than a hand-set `roll_frac` heuristic. Tire load per corner is
`−(spring+ARB) + m_unsprung·g_z`, exactly the `solve_trim`/`tire_loads`
convention.

Three residuals (verified against closed-form transfer in tests):

```
R_heave = ΣFz − m·g_z − Df_total
R_pitch = lf·(Fz_fl+Fz_fr) − lr·(Fz_rl+Fz_rr) + m·a_x·h      ⇒ ΔFz_front_axle = −m·a_x·h/L
R_roll  = (tf/2)(Fz_fl−Fz_fr) + (tr/2)(Fz_rl−Fz_rr) + m·a_y·h ⇒ total lateral transfer = m·a_y·h
```

Aero enters only through the heave balance and the ride-height feedback (the
loads set the travels → ride heights → `aero.compute` → downforce), matching how
the legacy trim treats aero (no separate aero pitch term). The 3×3 Jacobian is
finite-differenced (`eps = 1e-7`, as legacy) and solved by Cramer's rule
(allocation-free). Newton is seeded from the legacy straight-line compressions
mapped to `(heave, pitch, 0)`, so it starts near the solution and converges in a
few iterations.

**Reuse, not a fork.** At the symmetric point `a_x == 0 && a_y == 0` the solver
**delegates to `solve_trim_with_gz`** and returns a result built from it, so
`solve_operating_point(v, 0, 0, GRAVITY)` is **bit-identical** to the model's
static reference trim (test `straight_line_matches_legacy_trim_bitwise`). Away
from the symmetric point the 3-DOF Newton runs; its `(a_x, a_y) → 0` limit
approaches the legacy solution continuously (to solver tolerance ~1e-9). The only
discontinuity is that measure-zero symmetric column, which gets the canonical
straight-line trim — the physically correct choice — and is `< 1e-9 N` off the
Newton limit. `solve_trim_with_gz` was made `pub(crate)` (no numeric change) to
enable this.

## Feasibility semantics

Feasibility is a **first-class, honest output** — the solver never clamps loads
or accelerations. `TrimStatus` separates three outcomes:

1. **`Feasible`** — converged (`residual ≤ 1e-4 N`), all four tire loads `≥ 0`,
   the demanded planar force `m·√(a_x²+a_y²)` fits the combined-slip
   load-sensitive grip, and `|m·a_x|` is within the actuator limit.
2. **`Infeasible(reason)`** — converged but a constraint is violated. Reason is
   reported in a fixed **priority order**: `NegativeLoad` → `GripExceeded` →
   `PowerLimit`. Negative load is checked first because a lifted corner makes the
   trim unphysical and the grip estimate meaningless; grip is the usual envelope
   boundary; power is the actuator ceiling.
3. **`NotConverged { residual }`** — Newton did not reach tolerance in 50
   iterations (does not occur for physical inputs in the tested range; reported
   rather than hidden).

The grip budget is `Σ_i μ_eff(Fz_i)·Fz_i` with
`μ_eff = ½(μ_lat+μ_lon)·(1 − load_sens·(Fz−Fz_nom)/Fz_nom)` — the same
load-sensitive combined budget as `apex_optimizer::fourteen_dof_grip_budget`.
The feasibility boundary is **monotone** in `a_y` (test
`feasibility_boundary_is_monotone_in_ay`: once feasibility is lost it never
returns — no feasible islands — and the boundary reason is grip/load, never
non-convergence).

## Negative-`g_z` ruling

**Rejected as an input error.** `g_z ≤ 0` returns
`Err(OperatingPointError::NonPositiveGravity)`; non-finite inputs return
`Err(NonFinite)`. The recon deferred the policy to this layer; the ruling:

- At `g_z ≤ 0` the normal-load / grip budget concept inverts (tires would be in
  tension) and grip is undefined — there is no sensible feasibility answer to
  return.
- The envelope sweeps only physically meaningful `g_z > 0` (the effective normal
  acceleration from grade/bank/vertical-curvature is positive on any real
  surface the car stays on).
- The low-level `gz-pathway` functions stay **total** (they never panic on
  negative `g_z`, returning signed loads) by design — the *policy* line is drawn
  here, at the operating-point layer, exactly where the recon said to draw it.
  Keeping the trim total would force every downstream consumer to special-case a
  nonsense regime; a clear error type is cleaner and catches grid-construction
  bugs early.

## `a_y` boundary vs `g_z`: **linear, not sqrt** (task expectation corrected)

The task anticipated the `a_y` feasibility boundary scaling as `√g_z` in the
aero-free low-speed limit. **It scales linearly**, and the test asserts the
linear law (`boundary_ay_scales_linearly_with_gz_in_grip_limited_limit`). The
derivation:

- In the aero-free, load-sensitivity-free limit the grip budget is
  `base_mu · Σ Fz_i = base_mu · m · g_z` (total tire load = `m·g_z` with no
  downforce). The lateral demand is `m·a_y`. The boundary is
  `m·a_y = base_mu·m·g_z ⇒ a_y_boundary = base_mu·g_z` — **first order** in
  `g_z`. (Wheel-lift-limited boundaries scale linearly too:
  `a_y_lift ∝ g_z`.)
- The `√g_z` law from the g_z pathway (`max_corner_speed_scales_as_sqrt_gz`)
  applies to cornering **speed** at a **fixed radius**, where `a_y = v²/R`, so
  `v²/R = μ·g_z ⇒ v ∝ √g_z`. That is a statement about the swept axis being
  *speed*, not lateral acceleration. For an **independent `a_y` axis at fixed
  `v`** — which is how the envelope is parameterized — the g-g diagram scales
  linearly with the friction limit, which scales linearly with `g_z`.

Flagged explicitly because it contradicts the task brief; the physics is
unambiguous (accel-space friction circle radius ∝ `μ·g_z`), so the honest linear
test was written rather than forcing a `√` fit.

## Determinism

Same inputs → **bit-identical** outputs (test `deterministic_bitwise`): pure
`f64` arithmetic, fixed iteration count and order, no RNG, no `HashMap`, no
threads inside a solve. This is a prerequisite for the content-hashed envelope
cache (next task).

## Bench numbers

`crates/apex-physics/benches/trim_bench.rs` (criterion), release, on the dev
machine (Windows, native). Indicative — use for sweep budgeting, not as a gate:

| Bench | Time | Per point |
|---|---|---|
| `single_straight_line` (delegated fast path) | ~135 ns | 135 ns |
| `single_combined` (3-DOF Newton, braking+cornering) | ~464 ns | 464 ns |
| `batch_1000_sequential` | ~479 µs | ~479 ns |
| `batch_1000_rayon` | ~80 µs | ~80 ns |

Rayon gives ~6× on the batch. Budget implication: a dense envelope of, say,
`30 × 30 × 30 × 10 ≈ 2.7×10⁵` operating points costs ~130 ms sequential / ~22 ms
parallel — cheap enough to regenerate freely and to key by content hash. The
straight-line fast path is ~3.4× cheaper than the general solve, so envelopes
weighted toward low-`a_y` slices are cheaper still.
