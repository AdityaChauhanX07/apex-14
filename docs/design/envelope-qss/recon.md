# Envelope-QSS free-trajectory optimization — reconnaissance

**Status: accepted (recon; no code changed).** This document scopes the work
needed to build **envelope-QSS free-trajectory optimization**: a g-g-g envelope
swept over `(v, a_x, a_y, g_z)` feeding a free-trajectory quasi-steady-state
optimizer. It is sequenced ahead of the **fully dynamic minimum-lap-time OCP**
(future work) as a risk-mitigation step. Every claim below is anchored to a
file:line so the implementation slices can start from ground truth rather than
re-derive it.

Sources read: the internal planning notes, `PHYSICS_CHANGE.md`,
`docs/analysis.md`, `docs/math/{equations_of_motion,track3d,collocation}.md`,
`docs/design/{gn-solver-bound-deadlock,nlp-scaling}.md`, and the
`apex-optimizer` / `apex-math` / `apex-physics` / `apex-track` source.

> **Stale-plan caveat.** The earliest internal planning material framed the next
> major push as a full 14-DOF chassis model and never mentioned a g-g-g envelope.
> Priorities have since shifted: the fully dynamic minimum-lap-time OCP is the
> longer-term target, and this envelope-QSS work is the near-term step that de-risks
> it. That old material is authoritative for *physics* (equations of motion, car
> params, tire) but **stale for scope and deliverable lists.** Treat the dated
> `PHYSICS_CHANGE.md` entries + the deferral notes as the current record.

---

## 1. Solver state

### 1.1 What exists today

There are **two** NLP solvers in `apex-optimizer`, both driven through the same
`NlpEvaluator` / `NlpProblem` interface (`crates/apex-optimizer/src/nlp.rs:11-47`):

| Solver | File | Method | Bound handling | Used by |
|---|---|---|---|---|
| **Gauss-Newton** | `gauss_newton.rs` | Damped GN; normal equations `(JᵀJ + reg·I)Δx = rhs` solved **matrix-free by CG** | **post-hoc `project()` clamp only** | `optimize_gn` (point-mass), `golden_circle_optimize` |
| **Augmented Lagrangian** | `solver.rs` | AL outer loop + **projected gradient descent** inner loop | `project()` inside a *first-order* projected-gradient step | `optimize` (point-mass AL), `optimize_seven_dof`, `optimize_fourteen_dof` |

**Linear algebra actually available** (`apex-math`, `Cargo.toml` dep = `blake3`
only — no `faer`, `nalgebra`, `ndarray`, `sprs`; confirmed absent from
`Cargo.lock`):

- `CsrMatrix` (`sparse.rs`): `mul_vec`, `transpose`, `get`, `to_dense`, `scale`.
  **No factorization of any kind** — no sparse LU / Cholesky / QR / LDLᵀ.
- `lm.rs`: **dense** Gaussian elimination with partial pivoting
  (`solve_linear`, `crates/apex-math/src/lm.rs:330`) and `invert_sym` (`:373`),
  used by Levenberg-Marquardt for *small* dense parameter-fit problems
  (correlate/setup). O(n³) dense — unusable for a 14 000-var collocation KKT.
- `mat3.rs`: 3×3 inverse. `dual.rs` + `float.rs`: forward-mode AD (the Jacobian
  engine).

So the entire large-scale solver stack is **matrix-free CG** today
(`conjugate_gradient`, `gauss_newton.rs:108-152`): it applies `Jᵀ(Jv) + reg·v`
as two sparse mat-vecs per inner iteration and never forms `JᵀJ`.

### 1.2 How bounds/inequalities are handled — exact code paths

**Variable bounds** are stored on `NlpProblem` as `lower_bounds` / `upper_bounds`
(`nlp.rs:19-21`) and enforced **only** by element-wise clamp:

```rust
// gauss_newton.rs:90-94  (identical copy at solver.rs:101-105)
fn project(x: &mut [f64], lower: &[f64], upper: &[f64]) {
    for ((xi, &lb), &ub) in x.iter_mut().zip(lower.iter()).zip(upper.iter()) {
        *xi = xi.max(lb).min(ub);
    }
}
```

In GN it is applied once at start (`gauss_newton.rs:180`) and after every trial
step inside the line search (`:260`). **There is no active-set logic and no
bound-multiplier / dual variable.** The collocation bounds themselves are built
in `collocation.rs:480-517` (`build_nlp_problem`): `v ≥ v_min`,
`-max_brake_force ≤ f_drive ≤ max_drive_force`, `dt ∈ [dt_min, dt_max]`,
brake-bias ∈ `[0.50, 0.80]`.

**Inequality constraints** (track boundary, grip circle) are *not* bounds; they
go through the evaluator (`inequality_constraints` / `inequality_jacobian`) and
are handled differently by each solver:

- GN: a **penalty push** folded into the RHS, gated to fire only once equalities
  are nearly satisfied (`gauss_newton.rs:220-228`).
- AL: proper **multiplier estimates** `lambda_ineq` with a penalty term
  (`solver.rs:150-157, 256-258`).

The critical asymmetry: **AL has multipliers for the general inequalities but
still handles the box bounds by projection**, and its inner solve is only
*first-order* (projected gradient). GN has neither multipliers nor an active set
for bounds.

### 1.3 The Barcelona/oval golden deferral — the deadlock record

Primary record: `docs/design/gn-solver-bound-deadlock.md`. Dated log:
`PHYSICS_CHANGE.md` entries **2026-07-04** (deferral) and **2026-07-05** (golden
substitution). Verbatim failure mode:

> The solver … computes an **unconstrained** Newton step from `(JᵀJ + reg·I)Δx =
> rhs` via CG, and enforces variable bounds only by **post-hoc projection** …
> This deadlocks exactly when the true optimum requires a bound to bind.
> Concretely: `f_drive` saturates `max_drive_force` at the nodes on the oval's
> two straights … the raw Newton direction has `‖Δx‖ ≈ 483` (large, well-formed),
> but after damping and projection the net displacement is `max|x − x_new| ≈
> 7e-13` — genuinely zero. **25–28 of ~349 decision variables sit exactly at a
> bound** … a textbook projected-Newton deadlock, not a tuning problem.

Quantitatively: converges on `circle_track` (`eq_violation → 7.9e-6`) but floors
at `eq_violation ≈ 0.68 / 0.4 / 0.98` SI on oval / `random_spline_track(42)` /
Silverstone — 3–4 orders above `constraint_tol = 1e-4`, regardless of iteration
budget. **Ruled out by experiment (do not re-run):** variable scaling
(`nlp-scaling.md` — Jacobi scaling drove `diag(JᵀJ)` to exactly 1.0 and *still*
did not converge), warmstart quality (a 3.2× better warmstart made it *worse*),
line-search tuning (accepts every step already), inner-CG precision, mesh
coarsening (no N ∈ {10…40} converges).

Consequence: `golden_oval_optimize` was **removed** and the only optimize-mode
golden today is `golden_circle_optimize` (`circle_track(100,12,200)`, N=30,
`lap_time = 11.494914 s`) — the one non-trivial track the current solver
converges on cleanly. Oval / Silverstone / Barcelona optimize goldens are
explicitly parked until the bound-capable solver lands
(`PHYSICS_CHANGE.md:181-207`).

### 1.4 Assessment — minimal bounds fix vs interior-point prototype

**Option (a) — projected-GN / active-set correction (minimal).** Detect
bound-active variables (those clamped by `project`), pin them, and solve the
**reduced free-variable** normal-equation system with the *existing*
`conjugate_gradient`. Mechanically:

- Blast radius: essentially one file (`gauss_newton.rs`) plus a bound-activity
  detector. No new linear algebra — CG already exists and is matrix-free, so the
  reduced system is just CG restricted to free columns (zero the pinned entries
  of `p`/`rhs`, or mask the mat-vec).
- Cost: bound-flip cycling and the drop/add heuristics are the classic active-set
  headaches, but this is *small, self-contained* work.
- The deferral note's own verdict: "Cheaper than (b) but is new solver
  infrastructure that would likely be superseded once (b) lands"
  (`gn-solver-bound-deadlock.md:62-66`).

**Option (b) — primal-dual interior-point prototype.** Handles active bounds
natively via a log-barrier (correct implicit multiplier, no projection
deadlock). Given the current linear algebra, the honest scoping is:

- The IP KKT system is **symmetric indefinite**. There is **no sparse indefinite
  factorization in the tree** and adding `faer`/similar is a **posture conflict**:
  `apex-optimizer` is deliberately kept wasm-compatible (rand `default-features
  = false`, apex-physics pulled `default-features = false` to keep rayon out of
  the wasm graph — see `crates/apex-optimizer/Cargo.toml` comments). A native-only
  LA crate would bifurcate the build.
- The path of least resistance that *fits the existing LA*: form the **condensed /
  normal-equations reduction** of the KKT system (eliminate the slack/dual blocks
  analytically; the barrier contributes a positive-definite diagonal
  `X⁻¹S` to `JᵀJ`) and reuse matrix-free CG — exactly the machinery in
  `conjugate_gradient`, but with a barrier-weighted diagonal instead of a flat
  `reg·I`. This is buildable *without* a factorization.
- Risk of the CG route: as the barrier parameter → 0 the condensed system becomes
  increasingly ill-conditioned near active bounds; CG will need a diagonal
  (Jacobi) preconditioner — the machinery for which **already exists**
  (`jacobi_scale`, `collocation.rs:530`) — plus a fraction-to-boundary step rule
  and a barrier/centrality schedule. Net-new but bounded, and it stays wasm-safe.

**Recommendation:** build option (b), the condensed-KKT primal-dual interior-point
method, "to avoid building bound-handling infrastructure twice." Note that the
*envelope* is a QSS product and does not itself need the collocation solver — see
§2.4 — so the solver upgrade (shared with the dynamic-OCP workstream) can proceed
**in parallel with** the envelope sweep, not as a hard blocker.

---

## 2. Kinematics gap

### 2.1 What actually exists vs what §5.9 defers

The `(1 − n·κ)` curvilinear arc-length relation **already exists** and `n` is
**already a free state** — the "free trajectory" is not new in 2D:

```rust
// crates/apex-physics/src/point_mass.rs:45   (n = state[1], κ = scalar track_curvature)
let ds_dt = v_safe * alpha.cos() / (1.0 - n * kappa);
```

Same factor at **three** collocation RHS sites:

- `collocation.rs:1217` — trapezoidal f64 dynamics
- `collocation.rs:1240` — `Float`-generic dynamics (`T::one() - n * kappa`, the AD path)
- `collocation.rs:1638` — Hermite-Simpson dynamics

The state layout carries `n` as decision block `x[n..2n]` (`unpack`,
`collocation.rs:441-456`), and `optimize_gn` already optimizes it. **The point-mass
free-trajectory OCP with lateral offset is a solved, tested 2D capability** (the
circle golden exercises it).

**What the 3D track model work deferred** (`docs/math/track3d.md` §5.9, verbatim):
the **3D** version.

> Extended `(1 − n·Ω_z)`-style 3D road-coordinate kinematics are not derived
> here. … it consumes `Track`'s scalar `κ`, never `Ribbon3d` … A fully 3D version
> — replacing `κ` with `Ω_z` and accounting for the banked/pitched frame's effect
> on the `(1 − n·Ω_z)` factor and on `dn/dt` — is **not needed by the centerline
> point-mass QSS** … but **is required by the dynamic optimal-control-problem
> formulation**, which does carry `n` as a state.

So the gap is: (i) scalar `κ` → generalized `Ω_z(s)` from `Ribbon3d`; (ii) the
banked/pitched-frame corrections to the `(1 − n·Ω_z)` factor and to `dn/dt`
(a point offset laterally on a banked/graded ribbon does not travel the same
ground path as on a flat one).

### 2.2 Where s-domain kinematics live

- **Point-mass ODE:** `apex-physics/src/point_mass.rs:26-51`. Reads scalar
  `self.track_curvature` set externally.
- **Collocation RHS (×3):** `apex-optimizer/src/collocation.rs:1217, 1240, 1638`.
  Curvature comes from `track.curvature_at(s)` (`collocation.rs:241` etc.). The
  optimizer holds `&'a Track` (`CollocationOptimizer`, `collocation.rs:111`), **not**
  `Ribbon3d`.
- **QSS:** `apex-physics/src/qss.rs` (2D) and `qss_lap_sim_3d` — but QSS has **no
  lateral state** (`n = 0` implicit; see §5.8 note in `track3d.md`), so it has no
  `(1 − n··)` term to integrate at all.
- **Jacobian curvature-correction:** `docs/math/collocation.md` §6.3 — the
  s-columns get `−(dt/2)·(∂f/∂κ)·(dκ/ds)` because the dual sweep holds κ fixed
  per node. A 3D upgrade extends this to `dΩ_z/ds` (and any φ/θ dependence).

### 2.3 Blast radius of adding the curvilinear 3D transform

| Touch point | Change | Risk |
|---|---|---|
| **Track interface** | `CollocationOptimizer` holds `&Track`; needs `Ω_z(s), φ(s), θ(s), κ_v(s)` — these live on `Ribbon3d`, a *parallel* type (`track3d.md` §3, "additive, not a `Track` rewrite"). Either thread a `Ribbon3d` alongside, or add generalized-curvature accessors. | **High** — new data dependency into the optimizer's hot path. |
| **RHS (×3 + point_mass)** | Replace `1.0 - n*kappa` with the banked/graded `(1 − n·Ω_z)` form; add φ/θ terms to `dn/dt`. Must be done in all three collocation copies **and** the `Float`-generic one (AD correctness). | **Medium** — duplicated logic; easy to update one and miss another. |
| **Jacobian correction** | Extend §6.3's s-column term from `dκ/ds` to `dΩ_z/ds`. | **Medium** — silent accuracy loss if missed at corner entry/exit. |
| **Goldens** | `golden_circle_optimize` (point-mass optimize). Byte-stability is preserved **only** if the flat case collapses exactly: on a flat ribbon `Ω_z ≡ κ` bit-for-bit (`ribbon3d::tests::flat_exact_*`, `track3d.md` §3) and φ=θ=0 ⇒ `cos=1.0/sin=0.0` exactly (the §5.6 discipline). | **Medium** — must ship a bitwise flat-collapse regression like `qss::flat_ribbon_qss_bitwise_matches_track`. |

### 2.4 Does this work actually need the 3D transform?

**Only if the envelope free-trajectory OCP runs on 3D (banked/graded) tracks.**
The g-g-g envelope itself (§4) is a QSS/point-mass product with no lateral state.
The "free-trajectory" consumer that needs `n` is the collocation OCP, which today
has the **2D** `(1 − n·κ)` and works. Per the scope decision (§7), the first cut
runs on the **flat 2D** transform and folds 3D road effects in exclusively through
the envelope's `g_z` axis — so the full 3D curvilinear kinematics are **deferred to
the dynamic OCP** and are *not* on this work's critical path.

---

## 3. g_z pathway

### 3.1 Where normal load is computed, per model

All normal-load math hardcodes the vertical term as `mass * GRAVITY`
(`GRAVITY = 9.81`, `car_params.rs:6`) — **no model accepts an imposed vertical
acceleration today.** The 3D normal-load structure (`N = m(g·cosθ·cosφ +
v²κ·sinφ + v²κ_v) + F_df`, `track3d.md` §5.1) exists **only** in
`qss_lap_sim_3d`, and nothing higher-fidelity consumes it (`track3d.md` §5.9,
`PHYSICS_CHANGE.md:132-136`).

| Model | Normal-load site | Signature (inputs) | Vertical term |
|---|---|---|---|
| Point-mass grip circle | `car_params.rs:159` `max_grip_force(v)` | `(speed)` | `mass * GRAVITY` |
| Bicycle / axle | `car_params.rs:172` `axle_loads(v, a_x)` (+ `_generic` `:247`) | `(speed, a_x)` | `mass * GRAVITY` |
| 7-DOF / four-corner | `car_params.rs:202` `corner_loads(v, a_y, a_x, roll_frac)` | `(speed, a_y, a_x)` | `mass * GRAVITY` |
| 7-DOF grip budget (optimizer) | `collocation.rs:1549` `seven_dof_grip_budget` | mirrors `corner_loads` | `m * GRAVITY` |
| 14-DOF grip budget | `collocation.rs:2034` `fourteen_dof_grip_budget` | `(…, speed, a_y, a_x)` | via `corner_loads` + ride-height aero |
| 14-DOF trim | `fourteen_dof.rs:123, 128-129` `solve_trim` | `(speed)` | `g = GRAVITY` |
| 14-DOF derivatives | `fourteen_dof.rs:204` | state | `g = GRAVITY` |

The `mg` constant also appears in the AD dynamics: `collocation.rs:1266`
(`let mg = T::from_f64(car.mass * GRAVITY);`) and `:1549`.

### 3.2 Minimal change to accept imposed g_z WITHOUT ribbon3d consumption

The envelope sweeps `(v, a_x, a_y, g_z)`. `g_z` is an **imposed effective vertical
acceleration** (what §5.1 computes from grade/bank/vertical-curvature), fed in as a
**scalar sweep axis** — the model never needs to know it came from a ribbon.

Minimal, byte-stable plan:

1. Add an **optional scalar argument** `g_z: f64` (semantic: effective normal
   gravity, default `GRAVITY`) to the load functions, replacing the `mass *
   GRAVITY` static term with `mass * g_z`. Keep the existing signatures as thin
   wrappers passing `GRAVITY` so all current call sites are untouched.
2. Because `g_z` defaults to the exact `GRAVITY` literal, `mass * g_z == mass *
   GRAVITY` bit-for-bit (IEEE-754 exact) — the same "collapses via exact algebra"
   discipline proven for `mu_scale` and the 3D-QSS flat case
   (`PHYSICS_CHANGE.md:96-100`, `track3d.md` §5.6). **No golden moves.**
3. **Do NOT add `g_z` as a `CarParams` field.** It is a per-operating-point input,
   not a car property, and adding a field would change `car_params_hash` and break
   the frozen known-answer vectors (`content_hash.rs::exhaustive_field_sensitivity`,
   §5). Thread it as a call argument only.

This threads through: `max_grip_force`, `axle_loads(+generic)`, `corner_loads`,
`seven_dof_grip_budget`, `fourteen_dof_grip_budget`, `solve_trim`. The downforce
term `F_df(v)` is unchanged (it is already a `v`-only aero force, orthogonal to
`g_z`).

### 3.3 Golden / regression tests touching these code paths

- **`golden_oval_qss`**, **`golden_silverstone_synthetic_qss`**
  (`bins/apex-cli/tests/golden_lap.rs`) — both run `qss_lap_sim` → `max_grip_force`.
- **`golden_circle_optimize`** — point-mass `optimize_gn` → grip circle via
  `max_grip_force` (`collocation.rs:934, 1147`).
- **`content_hash.rs::exhaustive_field_sensitivity`** + `toml_overlay_equals_direct`
  (`apex-physics/tests/`) — guard `car_params_hash`; the reason g_z must **not**
  be a CarParams field.
- **Model unit tests:** `bicycle.rs` (`axle_load_*`, understeer), `seven_dof.rs`
  (`corner_loads_symmetry`, `aero_load`, `wheel_spin_equilibrium`),
  `fourteen_dof.rs` (trim/static-equilibrium), and
  `collocation.rs::fourteen_dof_grip_budget_behaves` (`:2625`).
- **Property tests:** `prop_car_config.rs`, `prop_tire.rs` (apex-physics).
- **`docs/analysis.md`** `compare`-binary numbers depend on every grip path.

---

## 4. Envelope prerequisites

### 4.1 Steady-state trim solving

**Partial — one narrow solver exists, the general one does not.**

`solve_trim` (`fourteen_dof.rs:115-174`) is a 2×2 Newton solve (numerical
Jacobian) for the **suspension heave + pitch** equilibrium at a given **speed
only**:

```rust
// residuals: G1 = vertical balance, G2 = pitch balance (moment about CoG)
let g1 = 2.0 * ff + 2.0 * fr - m * g - af.downforce_total;
let g2 = lf * 2.0 * ff - lr * 2.0 * fr;
```

It has **no lateral trim, no longitudinal trim, no yaw-moment balance, and no
`g_z`** — it finds ride height under static + aero load. For a g-g-g envelope
sweep you need a trim at each `(v, a_x, a_y, g_z)` grid point that balances the
four-corner loads and the tire forces. That is **mostly net-new**, but:

- The 2×2 numerical-Jacobian Newton pattern in `solve_trim` is a directly reusable
  template (extend residual/Jacobian dimension).
- For small dense trim systems, `apex-math::levenberg_marquardt` +
  `solve_linear` (`lm.rs:131, 330`) is already available and box-bounded.
- The 7-DOF model has **no** trim solver (it is a forward ODE, `seven_dof.rs:38`);
  a quasi-steady trim (set `dω/dt = 0`, solve wheel-spin + force balance) is
  new work.

### 4.2 Gridded storage with C1 interpolation

**Gap — nothing beyond linear/bilinear (C0) gridded interpolation exists.** The
design calls for **C1** for OCP smoothness.

| Existing interpolant | File | Order |
|---|---|---|
| `MuScaleGrid::mu_at` (station × lateral) | `grip_grid.rs:86` | **bilinear (C0)** |
| `GripMap::grip_at` (station × lateral) | `grip_map.rs:98` | **bilinear (C0)** |
| collocation `interp` helper | `collocation.rs:162` | **linear (C0)** |
| aero `ride_height_factor` | `aero.rs:68-100` | **1D cubic Hermite smoothstep (C1)** — but a hardcoded piecewise scalar map, *not* a general gridded interpolant |

So the only C1 code in the tree is the bespoke 1D aero ride-height curve. A
general **C1 gridded interpolant** (bicubic, or tensor-product C1 Hermite, over
the `(v, a_x, a_y, g_z)` envelope grid) is a **genuine new component**. The aero
Hermite (`t*t*(3-2t)`) is a usable 1D building block but does not generalize to
the multi-axis envelope for free.

### 4.3 Content-hash-keyed caching

**Ready — the key machinery is complete and proven; only a cache *store* is
new.** `apex-math::hash` provides deterministic, versioned
(`apex14.chash.v1`), domain-tagged BLAKE3 content hashing with a bit-exact float
policy (`hash.rs`). Confirmations for keying an envelope cache by car config:

- `CarParams` implements `ContentHash` (`car_params.rs:281-326`, destructure-based
  so a new field is a compile error until hashed) and exposes `car_params_hash`
  under domain `"car"` (`car_params.rs:328-333`).
- `settings_hash.rs` already **composes** multiple configs into one hash under a
  fixed field order (e.g. `optimize_fourteen_dof_settings_hash`,
  `settings_hash.rs:65`) — the exact pattern an envelope cache key needs
  (car ⊕ tire ⊕ aero ⊕ suspension ⊕ envelope-grid-spec).
- `AeroModel` and suspension configs also implement `ContentHash` (`aero.rs:170`).

So an envelope cache can be keyed as `content_hash("envelope.v1", { car_hash,
aero_hash, susp_hash, tire_hash, grid_spec })`. **What does not exist yet:** a
disk/in-memory cache *layer* keyed by that hash — but that is straightforward and
additive; the hard part (canonical, collision-resistant keys) is done.

---

## 5. Risk register

### 5.1 Contradictions with the planning assumptions

| # | Contradiction | Detail |
|---|---|---|
| R1 | **Scope has shifted off the old fidelity ladder.** | The earliest planning material framed the next big push as a full 14-DOF chassis model; this envelope-QSS work reprioritizes toward the free-trajectory optimizer and the dynamic OCP behind it. Anyone reading that old material for scope will be wrong (§ preamble). |
| R2 | **The "external NLP / IPOPT bridge" assumption is dead.** | Early architecture sketches assumed an IPOPT/FFI or external NLP. Reality: all-custom, **no external LA**, and `apex-optimizer` is deliberately **wasm-compatible** (rayon/getrandom kept out — `Cargo.toml`). An IP solver must be from-scratch and wasm-safe; `faer` would break the wasm posture (§1.4). |
| R3 | **3D kinematics is dynamic-OCP groundwork, resolved out of scope here.** | `track3d.md` §5.9 ties `(1 − n·Ω_z)` to the dynamic OCP. The scope decision (§7) keeps this work on the flat 2D transform and routes 3D road effects through the `g_z` axis, so the full 3D curvilinear kinematics defer to the dynamic OCP. **Decided, not a silent gap.** |
| R4 | **g_z must not enter `CarParams`.** | It is per-operating-point, not a car property; a field would break `car_params_hash` frozen vectors (§3.3). Pass as an argument. |
| R5 | **C1 gridded interp genuinely missing.** | The design requires C1; tree has only bilinear/linear grids + one bespoke 1D Hermite (§4.2). This is new work, not a parameter tweak. |
| R6 | **Envelope trim is largely net-new.** | `solve_trim` is speed-only heave/pitch; no lateral/longitudinal/yaw trim, no 7-DOF trim (§4.1). |

### 5.2 Blast radius on the ~556-test suite and byte-stable goldens

> The tree currently carries **3 golden fixtures** (`oval_qss`,
> `silverstone_synthetic_qss`, `circle_optimize_hermite_simpson`) plus a web of
> **bitwise / known-answer** guards in `hash.rs`, `ribbon3d.rs`, `qss.rs`,
> `aero.rs`, `suspension.rs`, `grip_grid.rs`, and the three `content_hash.rs`
> integration suites. (Raw `#[test]` count across crates+bins ≈ 710; the 556
> figure is the `cargo test` pass count — either way the byte-stable subset below is
> the part that *fails loud* on drift.)

| Work item | Blast radius | Mitigation / precedent |
|---|---|---|
| **Solver (active-set or IP)** | **Low on existing goldens.** The deadlock is already quarantined — `golden_oval_optimize` removed, only `golden_circle_optimize` pins optimize-mode, and it converges *today*. New solver = additive; keep `optimize_gn`'s default path bit-identical on the circle. Unpausing oval/Silverstone optimize goldens creates **new** fixtures (log via `PHYSICS_CHANGE.md`). | `gn-solver-bound-deadlock.md`; `PHYSICS_CHANGE.md:181-207` |
| **g_z threading** | **High surface, zero drift if disciplined.** Touches `max_grip_force` — used by *both* QSS goldens *and* the circle optimize golden. Byte-stable **iff** default `g_z ≡ GRAVITY` and exact-algebra collapse. Must ship a regression like `absent_grid_and_explicit_uniform_grid_are_bitwise_equal_to_baseline`. | `PHYSICS_CHANGE.md:96-114`; `track3d.md` §5.6 |
| **3D kinematics** | **Medium** (deferred to the dynamic OCP). Touches `point_mass.rs` + 3 collocation RHS + circle golden. Byte-stable **only** via flat `Ω_z ≡ κ`, φ=θ=0 collapse; needs a bitwise flat-collapse test mirroring `qss::flat_ribbon_qss_bitwise_matches_track`. | `ribbon3d::tests::flat_exact_*`; `track3d.md` §3 |
| **Envelope cache / trim / C1 interp** | **Near-zero on existing tests.** All additive new modules; no existing code path changes. New unit tests only. | `settings_hash.rs` compose pattern |

---

## 6. Recommended work breakdown (dependency-ordered)

Ordered so each slice lands byte-stable and unblocks the next. Items tagged
**[shared]** are shared with the dynamic-OCP workstream (built here, reused there).

- **scope-gate — decision, no code.** Confirm (a) first-cut track dimensionality
  (settled: flat 2D transform + `g_z` axis — see §7) and (b) which fidelities the
  envelope sweeps (point-mass only, or 7-/14-DOF). These answers determine how far
  `trim-solver` and `envelope-generation` reach.

- **gz-pathway — g_z ingestion into the load models.** Thread an optional scalar
  `g_z` (default `GRAVITY`, argument **not** CarParams field) through
  `max_grip_force`, `axle_loads(+generic)`, `corner_loads`,
  `seven_dof_grip_budget`, `fourteen_dof_grip_budget`, `solve_trim`. Ship the
  bitwise `g_z==GRAVITY` regression. *Depends on nothing; unblocks the sweep.*
  (§3)

- **trim-solver — envelope trim solver.** Generalize the `solve_trim` Newton
  pattern (or use `levenberg_marquardt`) to a quasi-steady trim at
  `(v, a_x, a_y, g_z)` for the chosen fidelity; emit the achievable-force envelope
  point. *Depends on gz-pathway.* (§4.1)

- **envelope-generation — grid + C1 interpolation + content-hash cache.** Define
  the `(v, a_x, a_y, g_z)` grid, a **C1** (bicubic / tensor-Hermite) interpolant,
  and a cache layer keyed by
  `content_hash("envelope.v1", car⊕tire⊕aero⊕susp⊕grid)`. Also precompute the
  `g_z(s)` profile along the centerline via the existing 3D QSS machinery (§7).
  *Depends on trim-solver; the cache-key machinery already exists.* (§4.2, §4.3)

- **ip-solver — bound-capable interior-point solver [shared].** Land option (b)
  primal-dual interior-point via condensed-normal-equations + matrix-free CG
  (reusing `conjugate_gradient` + `jacobi_scale`), staying wasm-safe. Keep the
  circle golden bit-identical; unpause the oval/Silverstone optimize goldens as
  **new** fixtures. *Independent of gz-pathway…envelope-generation; can run in
  parallel. Blocks free-trajectory-ocp if the OCP needs bound-binding convergence.*
  (§1.4)

- **free-trajectory-ocp — envelope free-trajectory OCP + validation.** Wire the
  cached g-g-g envelope into the free-trajectory QSS/collocation objective (with
  the precomputed `g_z(s)` profile), validate against the existing fidelities
  (`compare` binary, `docs/analysis.md`), and pin a new envelope golden. *Depends
  on envelope-generation and ip-solver.*

**Critical path:** gz-pathway → trim-solver → envelope-generation →
free-trajectory-ocp, with ip-solver as a parallel [shared] track that
free-trajectory-ocp joins to. Full 3D curvilinear kinematics are **out of scope
here**, deferred to the dynamic OCP (§7).

---

## 7. Scope decisions

Recorded here as the accepted scope for the first cut. These settle R3 (§5.1) and
fix the shape of the `envelope-generation` and `free-trajectory-ocp` slices.

- **Flat-track (2D) curvilinear kinematics for the first cut.** The OCP uses the
  existing `(1 − n·κ)` transform (`point_mass.rs:45`; three collocation RHS sites,
  §2.1). `κ` is taken as the **horizontal curvature of the 3D centerline**
  (`Ω_z`, which equals the 2D signed curvature on a flat ribbon, bit-for-bit —
  `track3d.md` §3). No banked/pitched-frame correction to the `(1 − n·κ)` factor
  or to `dn/dt` in this cut.

- **3D road effects enter exclusively through the envelope's `g_z` axis.** The
  envelope is generated over `(v, a_x, a_y, g_z)` from day one. The OCP consumes a
  precomputed **`g_z(s)` profile along the centerline**, produced by the existing
  3D QSS machinery (`qss_lap_sim_3d`, `track3d.md` §5.1: grade, banking, and
  vertical-curvature all fold into the single effective normal-acceleration term).
  **Flat tracks consume the `g_z = g` slice** of the envelope, which — by the
  exact-algebra collapse (§3.2) — reproduces today's flat results byte-for-byte.

- **Recorded approximation: `g_z` evaluated at the centerline, not at `(s, n)`.**
  The `g_z(s)` profile is sampled along the centerline, so a car running at lateral
  offset `n` sees the centerline's grade/bank rather than its own. This is
  **acceptable at this fidelity** (point-mass envelope QSS; the lateral excursion
  is small relative to the length scale of elevation/bank change). **Revisit under
  the dynamic OCP**, where a true `g_z(s, n)` would pair with the full 3D
  curvilinear kinematics.

- **Full bank/grade-coupled curvilinear kinematics: explicitly deferred to the
  dynamic OCP.** The `(1 − n·Ω_z)` upgrade with banked/pitched-frame corrections
  and a `g_z(s, n)` field (§2.1, §2.3) is not built here. It is dynamic-OCP
  groundwork and is sequenced with that work, not this one.

- **Solver posture: condensed-KKT primal-dual interior-point over the existing
  matrix-free CG.** Build option (b) from §1.4 — a log-barrier primal-dual method
  whose condensed normal-equations system is solved by the existing
  `conjugate_gradient` with a barrier-weighted Jacobi diagonal — with **no external
  linear-algebra dependencies** and **wasm-safe** throughout. **The
  projection-patched Gauss-Newton approach is rejected**: it deadlocks whenever the
  optimum needs a bound to bind (the straights saturate `max_drive_force`), as
  root-caused in `docs/design/gn-solver-bound-deadlock.md` and logged in
  `PHYSICS_CHANGE.md` (2026-07-04 / 2026-07-05). A minimal active-set patch to GN
  is also rejected as throwaway infrastructure that the interior-point method would
  supersede.
