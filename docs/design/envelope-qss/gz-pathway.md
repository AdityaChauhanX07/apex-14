# g_z pathway — imposed vertical acceleration in the normal-load budget

**Status: implemented.** Groundwork for envelope generation over
`(v, a_x, a_y, g_z)` (see [`recon.md`](recon.md) §3, §7). This change lets every
normal-load / grip-budget site take an **imposed vertical acceleration `g_z`**
(m/s²) in place of the hard-coded `GRAVITY`, defaulting to `GRAVITY` so all
existing behaviour is bit-for-bit unchanged. **No envelope code is added here** —
this is purely the consumption-side plumbing the envelope sweep will drive.

## API style

**Chosen: additive `*_with_gz` variant functions; the pre-existing form is
preserved and delegates with `g_z = GRAVITY`.** One style, used consistently
across both crates. Rationale:

- The recon contract (§3.2) specifies "an optional scalar argument `g_z: f64` …
  keep the existing signatures as thin wrappers." `g_z` is a single scalar, and
  the other operating-point axes (`v`, `a_x`, `a_y`) are already positional
  arguments on these functions, so a lone environment struct would be
  asymmetric. A suffixed variant keeps the delegation trivially auditable: the
  wrapper body is literally `self.foo_with_gz(.., GRAVITY)`.
- `g_z` is **never** a `CarParams` field (it is an environment/operating-point
  input, not a car property). Adding a field would change `car_params_hash` and
  break its frozen known-answer vectors — explicitly forbidden. It is threaded as
  the **last parameter** of each `_with_gz` variant.
- Public API is **additive only**: every original signature is retained and keeps
  working; only new `*_with_gz` entry points are introduced.

One exception to "keep the original": the private `solve_trim` had exactly one
caller (`FourteenDofModel::new`), which now routes through the new
`new_with_gz`. Rather than leave a dead private wrapper (which `-D warnings`
would reject), `solve_trim` was renamed to `solve_trim_with_gz`; the default
path is `FourteenDofModel::new → new_with_gz(.., GRAVITY) → solve_trim_with_gz(..,
GRAVITY)`.

## Sites touched

`g_z` replaces the static/weight `GRAVITY` term only. Aerodynamic downforce
(`½ρ·Cl·A·v²`), the longitudinal-transfer term (`m·a_x·h/L`), and the
lateral-transfer term (`m·a_y·h·…/t`) are **unchanged** — those carry the
vehicle's own accelerations, not gravity.

| Crate / file | Original (preserved) | New variant | g_z substitution |
|---|---|---|---|
| `apex-physics/car_params.rs` | `max_grip_force` | `max_grip_force_with_gz` | `m·g_z` in `μ·(m·g_z + df)` |
| `apex-physics/car_params.rs` | `axle_loads` | `axle_loads_with_gz` | `weight = m·g_z` |
| `apex-physics/car_params.rs` | `axle_loads_generic<T>` | `axle_loads_generic_with_gz<T>` | `weight = m·g_z` (`g_z: f64`) |
| `apex-physics/car_params.rs` | `corner_loads` | `corner_loads_with_gz` | via `axle_loads_with_gz` |
| `apex-physics/fourteen_dof.rs` | `FourteenDofModel::new` | `FourteenDofModel::new_with_gz` | trim `g = g_z` |
| `apex-physics/fourteen_dof.rs` | `solve_trim` → `solve_trim_with_gz` | (renamed) | `let g = g_z` (sole change) |
| `apex-optimizer/collocation.rs` | `grip_constraint_generic<T>` | `grip_constraint_generic_with_gz<T>` | `mg = m·g_z` (point-mass grip circle) |
| `apex-optimizer/collocation.rs` | `available_grip_generic<T>` | `available_grip_generic_with_gz<T>` | `weight = m·g_z` (7-DOF budget) |
| `apex-optimizer/collocation.rs` | `fourteen_dof_grip_budget` | `fourteen_dof_grip_budget_with_gz` | via `corner_loads_with_gz` (14-DOF budget) |

### Site list found vs. the recon list

The recon listed: `max_grip_force`, `axle_loads` (+generic), `corner_loads`,
7-DOF & 14-DOF grip budgets, `solve_trim`. Verified complete against
`grep GRAVITY` over `apex-physics` **and** `apex-optimizer`. The optimizer's
`grip_constraint_generic` (the point-mass grip circle, `mg = m·GRAVITY`) was the
one site not named explicitly in the recon's prose but present in its table
(`collocation.rs:1266`); it **is** a normal-load budget and is included.

Deliberately **excluded** (with reasons):

- **`rolling_resistance_force`** (`car_params.rs`) — `C_rr·m·g` is a longitudinal
  *resistance* force, not a grip/normal-load budget; not in the recon site list.
  (It uses `m·g` as a weight proxy; if a future envelope needs rolling drag to
  track `g_z`, revisit.)
- **14-DOF `derivatives`** (`fourteen_dof.rs`, `let g = GRAVITY`) — this is the
  gravitational body force in the *dynamic* forward-sim equations of motion, not
  a quasi-static grip budget. The envelope trims via `solve_trim_with_gz`; the
  dynamics deliberately still integrate under real `GRAVITY`.
- **`direct_solver.rs`** cornering-speed cap (`tire_mu·m·GRAVITY/denom`) and its
  `max_grip_force` calls — the direct (Gauss-Seidel) solver is not an envelope
  consumer. Its `max_grip_force` calls inherit byte-stable behaviour via the
  preserved wrapper; the inline cornering cap is a QSS mirror left for a later
  unification (see overlap note).
- **`forward_sim.rs`** `a_lat/GRAVITY`, `a_lon/GRAVITY` — unit conversion to "g"
  for telemetry, not a normal-load term.
- **Test helpers** (`car_params::cornering_lat_g`, `bicycle`/`seven_dof` test
  weights) — not production sites; `bicycle`/`seven_dof` production loads flow
  through `axle_loads`/`corner_loads`, already covered.

## Byte-stability argument

The invariant: with `g_z == GRAVITY`, every default code path executes the
**identical float op sequence** as before. This holds by *exact-algebra
substitution*, not by refactor:

- The only edit inside each moved body is the literal token `GRAVITY` → the
  parameter `g_z`. No expression is reassociated, no multiplication reordered, no
  term factored. E.g. `max_grip_force`'s body is `self.tire_mu * (self.mass * g_z
  + self.downforce(speed))`; with `g_z = GRAVITY` this is character-for-character
  the former `self.tire_mu * (self.mass * GRAVITY + self.downforce(speed))`.
- `g_z` enters exactly where `GRAVITY` did in the evaluation graph:
  - `car_params`: `self.mass * g_z` (f64), same operand order as `self.mass *
    GRAVITY`.
  - generic mirrors: `self.mass * g_z` is still computed in **f64** before
    `T::from_f64`, and `m * T::from_f64(g_z)` uses the same two `T::from_f64`
    conversions and the same `T` multiply as `m * T::from_f64(GRAVITY)`.
  - `solve_trim`: a single `let g = g_z;` at the top; every downstream use of the
    local `g` is untouched.
- IEEE-754 makes the substitution exact: `x * g_z` with `g_z` bit-equal to the
  `GRAVITY` constant yields bit-equal results (the same "collapses via exact
  algebra" discipline used for `mu_scale` and the flat 3D-QSS case).

This is enforced by three layers of test, not merely argued:

1. **Delegation/equality (bitwise):** each `*_with_gz(.., GRAVITY)` is asserted
   `to_bits()`-equal to its preserved original (fixed-input tests in
   `car_params.rs`, `collocation.rs`; randomized in `tests/prop_gz_pathway.rs`,
   512 cases).
2. **Frozen snapshots (bitwise):** for the paths **no golden fixture exercises**
   (7-DOF & 14-DOF grip budgets, point-mass grip constraint, `solve_trim`),
   pre-change `to_bits()` values were captured from the prior build and are
   asserted exactly (`solve_trim_gz_default_matches_frozen_snapshot`,
   `optimizer_budgets_gz_default_bitwise_identical`). These catch a mistake that
   an equality-only test could miss (where both old and new share a wrong body).
3. **Golden-lap fixtures:** `golden_oval_qss`, `golden_silverstone_qss` (both hit
   `max_grip_force`), and `golden_circle_optimize` (hits `grip_constraint_generic`)
   pass **unchanged** — the end-to-end anchor. No fixture regenerated; no
   `PHYSICS_CHANGE.md` entry (this change moves no simulation output).

## `qss_lap_sim_3d` overlap note

The 3D QSS (`apex-physics/qss.rs`) computes its own normal load inline —
`N = m(g·cosθ·cosφ + v²κ·sinφ + v²κ_v) + F_df` (§5.1 of `docs/math/track3d.md`) —
and the flat 2D QSS has its own inline cornering limit
(`cornering_speed`, `tire_mu·m·GRAVITY/denom`). **These were deliberately not
touched** (the task scopes them out). The relationship is directional and
intentional:

- **QSS is the *producer* of `g_z`.** Per the scope decision (`recon.md` §7), the
  envelope's `g_z(s)` profile is precomputed by exactly this 3D-QSS machinery:
  the effective normal acceleration `g·cosθ·cosφ + v²κ·sinφ + v²κ_v` **is** the
  `g_z` that the sites above will consume. So `qss_lap_sim_3d` keeps its own
  first-principles form as the reference that *generates* `g_z`.
- **The `car_params` / optimizer budgets are the *consumers*.** They now accept
  that scalar `g_z` without needing any ribbon/3D geometry.

The 2D `cornering_speed` inline form (and the `direct_solver` cornering cap) are
parallel re-implementations of the grip circle that do **not** yet route through
`max_grip_force`. **Should these unify later?** Yes — a future cleanup could route
the QSS 2D cornering limit and the direct-solver cap through
`max_grip_force_with_gz`, giving a single g_z-aware grip-circle definition. That
is out of scope here (it would touch QSS goldens and needs its own byte-stability
pass); recorded so the duplication is a known decision, not drift.

## Low / zero / negative g_z findings

Recorded from the behaviour tests, not papered over:

- **`solve_trim_with_gz` stays finite and convergent** across `g_z ∈ {−g, 0, 1,
  3, g, 2g}` (Newton solve does not diverge). At rest (speed 0, no downforce) the
  total tire load equals `m·g_z` exactly, i.e. it **scales linearly with `g_z`**;
  higher `g_z` compresses the suspension further (verified monotonic).
- **`g_z = 0`** (weightless): at rest the trim's net tire load → 0; the grip
  budget reduces to the pure downforce term (`max_grip_force_with_gz(v, 0) =
  μ·downforce(v)`), which is 0 at `v = 0`. Sensible.
- **`g_z < 0`** (inverted / "upforce" operating point): the linear force balance
  yields **negative (tensile) tire loads** and a negative grip budget at low
  speed (`max_grip_force_with_gz(0, −g) < 0`). The functions remain **total** (no
  panic, no NaN) — `axle_loads`/`corner_loads` floor individual corners at `0.0`
  via their existing `.max(0.0)`, while `solve_trim` reports the signed balance.
  This is a nonphysical regime for a real car but a legitimate math input; the
  envelope will sweep only physically meaningful `g_z ≥ 0` values, so the
  negative branch is documented rather than clamped (clamping would hide input
  errors and is a policy decision better made at the envelope layer).
