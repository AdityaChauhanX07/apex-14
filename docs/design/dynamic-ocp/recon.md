**Status: reconnaissance (dynamic-ocp workstream, opening survey). No code changed.**

# Fully dynamic minimum-lap-time OCP — reconnaissance

The goal this survey serves: a **fully dynamic** minimum-time optimal control problem —
no QSS assumption, no driver model — solved in the `s`-domain over a whole lap, first on
the 6-state single-track model, later on the 10-state four-wheel model. States
`{n, xi, v_x, v_y, r, delta}`, controls `{steer rate, drive/brake}`, periodic boundary,
track edges + tire feasibility + power + steering limits, objective `∫ dt/ds ds`.

This document inventories what exists, what is new work, and where the load-bearing risks
are. It inherits its solver evidence from the closed envelope-QSS workstream
([`../envelope-qss/CLOSE.md`](../envelope-qss/CLOSE.md),
[`../envelope-qss/real-track-convergence.md`](../envelope-qss/real-track-convergence.md))
and its bound-handling evidence from
[`../gn-solver-bound-deadlock.md`](../gn-solver-bound-deadlock.md).

**Headline conclusions, stated up front:**

1. **The single-track model is already dual-compatible; the four-wheel model is not, and
   not by a small margin.** `BicycleModel` has a `Float`-generic mirror locked by a test.
   `SevenDofModel` is `f64`-only, global-frame, and contains a genuine `signum`
   discontinuity that the `Float` trait cannot even express today.
2. **The critical path is the preconditioner, not the model and not the derivatives.**
   The documented `N²` graph-Laplacian wall stops the current solver at `N ≈ 44`; the
   target is `N = 300–600`. A **node-major block-tridiagonal preconditioner is exactly
   the inverse of the structure that causes the wall**, is implementable with the existing
   `CsrMatrix` + `apex_math::lm::solve_linear`, adds no dependency, and is wasm-safe.
3. **Reverse-mode AD is not needed first, and the roadmap's stated reason for it is
   wrong.** The existing forward-dual Jacobian assembly is `O(stencil width)` per
   interval — *independent of `N`*. The minimal derivative upgrade is a multi-seed dual
   (`DualN<K>`), not a tape.
4. **The Spa business case as written cannot be delivered by this workstream.** A
   minimum-time OCP with no driver model will over-carry the Spa descent *more* than QSS
   does, not less. What it can deliver is a bound that converts the residual from
   "unexplained" to "confirmed driver behaviour". Detailed in §6.3 and §7.1.

---

## 1. Dynamic models as OCP citizens

### 1.1 Where the models live

| Model | File | States | Frame | Controls | `Float`-generic? |
|---|---|---|---|---|---|
| Point-mass | `crates/apex-physics/src/point_mass.rs` | 4 `[s,n,v,alpha]` | **curvilinear** | `[F_drive, kappa_cmd]` | yes (via collocation mirror) |
| Point-mass (OCP copy) | `crates/apex-optimizer/src/collocation.rs:1246` `point_mass_derivatives` / `:1271` `_generic` | 4 | **curvilinear** | same | **yes** |
| "7-DOF" OCP dynamics | `crates/apex-optimizer/src/collocation.rs:1692` `seven_dof_derivatives_generic` | **4** | curvilinear | `[F_drive, kappa_cmd]` | **yes** |
| **Single-track** | `crates/apex-physics/src/bicycle.rs:33` (concrete) / `:95` (generic) | 6 `[X,Y,psi,vx,vy,omega_z]` | **global Cartesian** | `[delta, fx_total]` | **yes** |
| **Four-wheel (7-DOF)** | `crates/apex-physics/src/seven_dof.rs:37` | 10 `[X,Y,psi,vx,vy,omega_z,ω×4]` | **global Cartesian** | `[delta, T_drive, brake]` | **no — `f64` only** |
| 14-DOF | `crates/apex-physics/src/fourteen_dof.rs` | 24 | global | — | no |

> **Naming trap — flag this loudly.** `collocation.rs::seven_dof_derivatives_generic` is
> **not** the 7-DOF model. It is the 4-state curvilinear point-mass RHS with the
> longitudinal force smoothly saturated onto a four-corner load-sensitive grip budget
> (`available_grip_generic_with_gz`, `tire_limited_forces_generic`). It has no wheel-spin
> states, no yaw state, and no lateral velocity. Any plan that reads the roadmap's "reuse
> the 7-DOF dynamics" as applying to this function is planning on a fiction. The real
> 10-state model is `apex-physics/src/seven_dof.rs` and it is not OCP-ready at all.

### 1.2 Genericity verdict

**Single-track: ready.** `impl<T: Float> OdeSystemGeneric<T, 6, 2> for BicycleModel`
(`bicycle.rs:95`) mirrors the concrete impl, and `generic_matches_concrete`
(`bicycle.rs:338`) locks the equivalence at four operating points including one below the
`vx` guard. `dual_jacobian_column_is_finite_and_plausible` (`:371`) exercises the dual
path through the tire force chain. The whole force chain beneath it is generic:
`PacejkaTire::lateral_force_generic` → `magic_formula_generic`
(`tire/pacejka.rs:143`), `CarParams::axle_loads_generic`, `drag_force_generic`.

**Four-wheel: not ready.** `seven_dof.rs` has a single `impl OdeSystem<10, 3>` on `f64`.
There is no `OdeSystemGeneric` mirror. Producing one is not a mechanical transcription —
see the discontinuity table.

### 1.3 Discontinuity / non-smoothness inventory

Every site an NLP would trip on, with whether a C1 path already exists.

| # | Site | File:line | Nature | C1 path today? |
|---|---|---|---|---|
| D1 | Brake torque sign | `seven_dof.rs:124` `t_brake_mag * omega_w[i].signum()` | **Jump discontinuity in the RHS** at `ω = 0`; derivative is a delta | **No.** And `Float` has no `signum` (`apex-math/src/float.rs:34–56` — trait exposes `sin cos tan sqrt abs powi powf atan atan2 max min recip exp ln`). Must be replaced by a smooth surrogate; `tanh` is also absent but constructible from `exp`. |
| D2 | Slip-ratio denominator | `seven_dof.rs:95,102` `v_tire_x.abs().max(1.0)` | Kink from `abs`, second kink from `max` | No; but the kink sits at low speed only. A smooth `sqrt(v² + v_min²)` regularization is the standard fix. |
| D3 | Slip-angle denominator | `seven_dof.rs:94,101`; `bicycle.rs:49,111` `vx.max(1.0)` | Kink at `v_x = 1` | No. Harmless if `v_min` bound in the OCP is above 1 m/s — the envelope OCP already uses `v_min = 5.0` (`envelope_ocp.rs:104`). **Recommend the same, which retires D3 by construction.** |
| D4 | Combined-slip friction clamp (hard) | `tire/combined_slip.rs:98` and `tire/pacejka.rs:196` `combined_forces_generic` | **Kink in the derivative** at the friction boundary — hard `if/else` scale | **Yes, but it is a *different function*.** `combined_forces_smooth_generic` (`combined_slip.rs:176`) uses `smooth_min` (log-sum-exp, C∞) and is locked by `smooth_is_differentiable_at_grip_limit` (`:429`). **The plain `combined_forces_generic` is NOT smooth — verify every call site.** `seven_dof.rs:96,103` calls the **hard** `combined_forces`. |
| D5 | `mu_eff` floor | `tire/pacejka.rs:154` `(base_mu * (1 - load_sens*ratio)).max(zero)` | Kink at very high load | No. Reachable only past `Fz ≈ 11·Fz_nom`; practically inert but should be documented, not silently relied on. |
| D6 | Zero-load early return | `tire/pacejka.rs:146` `if fz.real_value() <= 0.0 { return zero }` | **Value + derivative jump**; also a `real_value()` branch, so the *dual* branches on the *real* part | No. Reachable: `available_grip_generic_with_gz` clamps corner loads at zero (`collocation.rs:1649–1652`), i.e. wheel-lift is modelled. Needs either a `Fz >= Fz_min` inequality constraint in the OCP or a smooth floor. |
| D7 | Corner-load clamps | `collocation.rs:1643–1652` `.max(T::zero())` ×6 | Kinks at wheel-lift | No. Same remedy as D6. |
| D8 | Aero ride-height factor | `aero.rs:192` `((h - h_design)/(h_high - h_design)).min(1.0)` | Piecewise-linear → **kink**, and the segments are only C0 | No. Only reachable via the ride-height path (`solve_operating_point`, 14-DOF), **not** on the single-track OCP path, which uses `CarParams::downforce`/`drag_force` (smooth `v²`). **Not a single-track blocker; is a four-wheel/envelope-regeneration blocker.** |
| D9 | Gear selection | `drivetrain.rs:155` `optimal_gear` — `argmax` over a discrete gear set, plus `if rpm > max_rpm \|\| rpm < idle` rejection | **Integer decision** — genuinely non-smooth, an MINLP if modelled | No, and none is wanted. **Recommendation: do not put the gearbox in the OCP.** Model the powertrain as a smooth `F_drive ≤ P_max/v` power hyperbola plus `F_drive ≤ max_drive_force` (both already `CarParams` fields, `car_params.rs:34,36`). This matches the brief's "power limit" and matches published minimum-time OCP practice. |
| D10 | Engine torque curve | `drivetrain.rs:45,56` piecewise `if rpm <= idle` / segment lookup | C0 piecewise | No. Retired by D9's recommendation (torque curve not used). |
| D11 | Envelope interpolant | `apex-math::interp::HermiteGrid`, consumed via `Envelope::rho_grad` (`envelope.rs:469`) | **C1 by construction** — that was its design purpose | **Yes.** Not on the dynamic-OCP path (the dynamic OCP replaces the envelope with the model), but relevant if a fidelity ladder starts from the envelope OCP. |

**Net:** for the **single-track** OCP the only live items are D3 (retired by `v_min ≥ 5`)
and D4 (choose `combined_forces_smooth_generic`, or — since the single-track model uses
pure lateral forces only, `bicycle.rs:119–120` — nothing at all until longitudinal tire
force is added per-axle). **The single-track model is essentially smooth today.** For the
**four-wheel** OCP, D1/D2/D4/D6/D7 are all live and D1 has no expressible fix without
extending `Float`. That asymmetry is the strongest argument for single-track-first.

### 1.4 How the models are driven today vs what the OCP needs

Today the dynamic models are **integrated forward under a controller**:

- `apex-integrator` (`rk4.rs`, `rk45.rs`) steps `OdeSystem::derivatives(state, control, t)`.
- Controls come from `apex-physics::controller` — `LqrController::compute_steering`
  (`controller.rs:244`, with a CARE solve at `:76`) and `SpeedController` PID (`:344`).
- `apex-optimizer::forward_sim::ForwardSimulator::simulate` (`forward_sim.rs:160`) replays
  an `OptimizationResult` to produce `DetailedTelemetry`.

The OCP needs none of that. It needs a **pure RHS in the `s`-domain**:
`x' = f(x, u, s)` where `'` is `d/ds`, `f` is `Float`-generic, side-effect-free, and
allocation-free per call. Two structural mismatches:

- **Frame.** Both dynamic models are in the **global Cartesian** frame with `X, Y, psi`
  as states. The OCP frame is curvilinear with `n, xi` — `X, Y` disappear entirely and
  `psi` is replaced by `xi = psi - psi_track(s)`. This is not a wrapper; it is a new RHS.
- **Independent variable.** Both are `d/dt`. The OCP is `d/ds`. Every derivative is
  multiplied by `dt/ds` (§2).

The existing `envelope_ocp.rs::dynamics` (`:296`) is the pattern to follow: a small,
`Float`-generic, `self`-borrowing method returning `[T; n_states]`, with the
node-Jacobian taken by looping single dual seeds (`dynamics_jac`, `:312`).

---

## 2. Curvilinear dynamics

### 2.1 What exists

The 2D `(1 - n·kappa)` transform exists in **three** independent copies, all for the
*point-mass* kinematics where the velocity direction is itself a state:

| Site | Form |
|---|---|
| `docs/math/equations_of_motion.md` §1.3 | `ds/dt = v cos α/(1−nκ)`, `dn/dt = v sin α`, `dα/dt = κ_cmd v − κ ds/dt` |
| `collocation.rs:1271` `point_mass_derivatives_generic`, `:1713` (7-DOF variant) | time-domain, `[s,n,v,α]` |
| `envelope_ocp.rs:296` `dynamics` + `docs/math/envelope_ocp.md` | **`s`-domain**, `{n, xi, v}` |

The envelope OCP is the only one already in the `s`-domain, and it is the right template:

```
n'  = (1 - n·kappa)·tan(xi)
xi' = kappa_cmd·(1 - n·kappa)/cos(xi) - kappa
v'  = (a_x - drag/m - roll/m)·(1 - n·kappa)/(v·cos(xi))
```

### 2.2 What is new derivation work

With a chassis that has body-frame `v_x, v_y` and an independent yaw rate `r`, `xi` is now
the **chassis yaw** relative to the tangent, and the velocity direction is `xi + beta`
where `beta = atan2(v_y, v_x)` is the sideslip. Define the single shared factor

```
S(n, xi, v_x, v_y; kappa) = dt/ds = (1 - n·kappa) / (v_x·cos(xi) - v_y·sin(xi))
```

Then the whole `s`-domain RHS is the time-domain RHS scaled by `S`:

```
n'   = (v_x·sin(xi) + v_y·cos(xi))·S
xi'  =  r·S - kappa
v_x' = ( (F_x - F_yf·sin(delta) - F_drag - F_roll)/m + v_y·r )·S
v_y' = ( (F_yf·cos(delta) + F_yr)/m - v_x·r )·S
r'   = ( (l_f·F_yf·cos(delta) - l_r·F_yr)/I_z )·S
delta' = u_delta                       (steer rate defined per metre — see below)
```

objective integrand `dt/ds = S`, so `T = ∫ S ds`.

**Consistency check against the shipped envelope OCP** — this is a real cross-check, not a
restatement. Set `v_y = 0`, `v_x = v`, `r = kappa_cmd·v`. Then
`S = (1−nκ)/(v·cos xi)` and

- `n' = v·sin(xi)·S = (1−nκ)·tan(xi)` ✓ matches
- `xi' = kappa_cmd·v·(1−nκ)/(v·cos xi) − kappa = kappa_cmd(1−nκ)/cos(xi) − kappa` ✓ matches
- `v_x' = (a_x − drag/m − roll/m)·S` ✓ matches

The new transform **reduces exactly to the validated one** in the zero-sideslip,
kinematic-yaw limit. That reduction should be written as a unit test, not just a doc claim.

**New singularity, and it is different from the envelope OCP's.** `S` blows up when
`v_x·cos(xi) = v_y·sin(xi)`, i.e. when the *velocity* (not the chassis) is perpendicular to
the track tangent — `tan(xi) = v_x/v_y`. The envelope OCP guarded this with `v ≥ v_min` and
`|xi| ≤ xi_max = 1.2` (`envelope_ocp.rs:103–104`). Those two are **not sufficient here**:
large `v_y` at small `v_x` still triggers it. Required additions: a sideslip bound
(`|beta| ≤ beta_max`, e.g. 0.3 rad, as a path inequality or via bounds on `v_y`), and
`v_x ≥ v_min`.

**Steer-rate convention.** Defining `u_delta` as `d(delta)/ds` (per metre) rather than
`d(delta)/dt` gives `delta' = u_delta` with no `S` factor — one fewer nonlinear coupling in
the Jacobian, and the physical rate limit becomes `|u_delta| ≤ rate_max·S`, a path
inequality rather than a bound. The alternative (`delta' = u_delta·S` with a plain bound)
is also fine. **Recommend the plain-bound form** (`delta' = u_delta·S`, `|u_delta| ≤
rate_max`) — the bound goes straight into the IP solver's box-constraint machinery, which
is its documented home turf (`ip-solver.md`), rather than adding an inequality row.

**Deliverable: a new `docs/math/dynamic_ocp.md`** carrying the derivation above, the
reduction check, the singularity analysis, and the four-wheel extension when it lands.
`docs/math/equations_of_motion.md` §2 (bicycle) is the time-domain source; it should gain a
forward reference, not be rewritten.

### 2.3 3D — recommendation: **2D-first with `g_z(s)` imposed**

Groundwork for the deferred 3D version is **real and better than expected**.
`apex_track::ribbon3d` already computes and exposes the full Darboux vector per station:
`RibbonStation.omega_x/omega_y/omega_z` (`ribbon3d.rs:106–111`), with accessors
`omega_at(s)` (`:463`), `frame_at(s)` (`:477`), `grade_at`, `bank_at`, `curvature_at`, and
a `validate()` that already checks `omega_y_max` and orthogonality
(`RibbonValidation`, `:150`). `docs/math/track3d.md` §1–2 derives the frame and the
generalized curvatures. So the *geometry* is done.

What is **not** done: `docs/math/track3d.md` states outright that "higher fidelities
(single-track / four-wheel / 14-DOF) do not yet consume any of this — see §5.9 Deferrals".
The 3D *dynamics* — the full `{n, xi, v_x, v_y, r}` transform through `Ω_x, Ω_y, Ω_z`, with
the roll/pitch coupling those imply — is entirely new derivation, roughly triples the
size of the math page, and has **no validation target** (the literature benchmark, §6.2,
is 2D).

**Recommendation: follow the envelope-QSS pattern exactly** — solve 2D with a per-node
`g_z(s)` profile imposed from the 3D machinery. That pattern is already proven end-to-end:
`EnvelopeOcp::with_gz_profile` (`envelope_ocp.rs:177`) takes `gz_profile: Vec<f64>`, the
`*_with_gz` variants exist throughout `apex-physics`, `qss_lap_sim_3d_with_gz`-family
functions exist (`qss.rs:495`), the default `g_z == GRAVITY` path is locked bit-identical
by `prop_gz_pathway` and friends, and Spa-3D solved at `N* = 24` (88.265 s, CLOSE.md §1).
The full `Ω`-based 3D dynamic OCP should be an explicitly named, explicitly deferred item.

---

## 3. Solver scale-up path

### 3.1 The diagnosed failure, restated precisely

From `real-track-convergence.md` §B.2, and it is a **positive-feedback loop**, not a
single bottleneck:

> the periodic first-difference collocation Jacobian gives `JeqᵀJeq` a second-difference
> (graph-Laplacian) structure whose condition number grows like `N²`. Larger `N` → `eq`
> contracts more slowly → the `mu`-coupled ramp grows `rho` → `rho·JeqᵀJeq` is worse
> conditioned → the primal freezes.

with three attached facts: **CG hits `cg_max_iter = 250` on every Newton step at every
`N`, converging and failing runs alike**; raising `cg_max_iter` to 2000 and `cg_tol` to
`1e-12` makes real Silverstone `N ≥ 48` *worse*; and Jacobi preconditioning is exhausted.

### 3.2 (a) A block-structured preconditioner — concrete design

**The key structural observation.** The envelope OCP lays variables out **block-contiguous
by quantity** (`idx_n(k)=k`, `idx_xi(k)=N+k`, `idx_v(k)=2N+k`, …, `envelope_ocp.rs:209–227`),
mirroring `collocation.rs`'s `[s|n|v|α|F|κ|dt]` layout (`docs/math/collocation.md` §3).
In *that* ordering the condensed operator `M = rho·JeqᵀJeq + …` has its couplings at stride
`N` — it looks dense-ish and is invisible to any banded method. **Reorder node-major**
(all `n_states + n_controls` variables of node 0, then node 1, …) and the same operator
becomes **block-tridiagonal with a periodic corner**, because each trapezoidal defect row
touches only nodes `k` and `k+1` (`envelope_ocp.rs:628–646`; `docs/math/collocation.md` §5
states the 13-column stencil for the 4-state case).

That is the whole opportunity, and it is worth stating why it is decisive: **a
block-tridiagonal solve is the exact inverse of a 1D Laplacian.** The documented growth is
`cond ~ N²` precisely because the operator *is* a 1D graph Laplacian. Preconditioning with
its exact block-tridiagonal factorization makes the preconditioned condition number
`O(1)` in `N`. This is not a heuristic improvement over Jacobi; it targets the diagnosed
mechanism directly.

Three implementable rungs, all with **no new dependencies and wasm-safe** (pure `f64`
arithmetic, no threads required, deterministic sequential reductions — matching `ipm.rs`'s
stated determinism contract):

**Rung 1 — block-Jacobi per node.** Assemble the `nb × nb` diagonal block of `M` per node
(`nb = n_states + n_controls`, 8 for single-track), factor each with
`apex_math::lm::solve_linear` (`lm.rs:330`, already public and dense), apply
block-wise in the PCG `z = M⁻¹r` step. Replaces the scalar `minv` at `ipm.rs:735`.
~40 lines. Captures the intra-node tire/aero cross-coupling that scalar Jacobi
structurally cannot see. Cheap insurance, but **does not address the `N²` growth** — the
Laplacian lives in the *inter*-node coupling.

**Rung 2 — block-tridiagonal (the one that matters).** Assemble the three block diagonals
of `M` exactly: iterate the rows of `J_eq` (available as a `CsrMatrix` from
`equality_jacobian`), and because **every row touches exactly two nodes**, its rank-1 outer
product lands entirely within the block-tridiagonal band. So the `rho·JeqᵀJeq` contribution
is captured **exactly**, not approximated. Add `Jineqᵀ Σ_I Jineq` (node-local for the
envelope-style constraints → diagonal blocks only) and `diag(Σ_L + Σ_U) + reg`. Then a
block-Thomas / block-LDLᵀ forward-backward sweep. Cost `O(N · nb³)`: at `N = 600`,
`nb = 8` → `600 × 512 ≈ 3·10⁵` flops per application. Negligible against 250 CG
iterations of matrix-free `J·v`/`Jᵀ·v`.

Handle the **periodic wrap** (the `(N-1, 0)` corner blocks from `j = (i+1) % n`,
`envelope_ocp.rs:572`) one of two ways: (i) **drop it** — the preconditioner need not be
exact, and a rank-`nb` perturbation out of `N·nb` is asymptotically irrelevant; or (ii)
Sherman–Morrison–Woodbury with an `nb × nb` correction solve. **Recommend (i) first**, and
only escalate if measured CG counts say otherwise.

**Rung 3 — periodic block-cyclic reduction.** Exact for the periodic operator. Only if
Rung 2 measurably underperforms.

**What `CsrMatrix` is missing.** Nothing blocking. It has `mul_vec`, `transpose`,
`row_entries`, `get`, `nnz` (`apex-math/src/sparse.rs`). It has **no** sparse–sparse
matmul and **no** triangular solve — but Rung 2 needs neither, because the block assembly
goes directly from `J_eq`'s rows into dense `nb × nb` blocks. `apex_math::lm::solve_linear`
covers the dense block inverses. `mat3.rs` exists for 3×3 but `nb = 8` needs the general
routine.

**Non-negotiable integration constraint.** `ipm.rs` currently has **zero source changes**
since the envelope work and carries `ip_resolves_gn_bound_deadlock` (locked at
`eq = 1.287e-7`) and `determinism_bitwise_history`. A preconditioner **must** be introduced
behind a new `IpmConfig` field defaulting to the present scalar Jacobi, so every existing
number stays bit-identical. See §7.2.

### 3.3 (b) Does mesh continuation plug into the IP solver today?

**Partially — and one documented blocker has been retired.**

*What exists.* `mesh_refinement::optimize_with_refinement` (`mesh_refinement.rs:115`) runs
a coarse→fine ladder: solve at `sequence[0]`, then per level
`opt.initial_guess_from_result(&current)` → `opt.optimize_gn_from(&x0, …)`. It also
switches Trapezoidal→Hermite-Simpson at the finest level (`method_for_level`, `:92`). Four
tests cover it.

*What does not plug in.*

1. **It is hardwired to Gauss-Newton.** `MeshRefinementConfig.solver_configs:
   Vec<GaussNewtonConfig>` (`:24`) and the call is `optimize_gn_from` (`:146`). There is
   no IPM path through it — even though `CollocationOptimizer::optimize_ip_from`
   (`collocation.rs:671`) already exists. This is a ~30-line generalization for the
   4-state collocation problem.
2. **It is hardwired to the 4-state layout.** `initial_guess_from_result` consumes an
   `OptimizationResult` (`speeds/offsets/headings/stations/drive_forces/curvature_cmds/
   time_steps`). Neither `EnvelopeOcpResult` nor a future single-track result fits.
3. **`EnvelopeOcp` has no `solve_from(x0)` at all.** `solve()` (`envelope_ocp.rs:446`)
   calls its private `warm_start()` unconditionally. There is no entry point that accepts
   an external initial guess — so the envelope OCP cannot be laddered *today*, at all.

*The retired blocker.* `gn-solver-bound-deadlock.md` recorded: "**Mesh coarsening** — no
coarse `N` in {10, 15, 20, 25, 30, 40} converges from the QSS warmstart either, so mesh
continuation has no rung to climb from with the current solver." That was the **GN**
solver under the bound deadlock. The IP solver **does** reach machine-tight feasibility at
coarse `N` on every real circuit (`N* = 24–40`, CLOSE.md §1). **The ladder now has a
bottom rung.** This is the single most encouraging piece of inherited evidence for the
continuation strategy and should be cited when the work is scoped.

*Recommended shape.* A small solver-agnostic trait —
`trait MeshLevel { fn n_nodes(&self) -> usize; fn solve_from(&self, x0: &[f64]) -> LevelOutcome; fn lift_from(&self, coarse: &Self, x: &[f64]) -> Vec<f64>; }`
— with the ladder driver generic over it. ~100 lines, reuses the existing structure,
lets the envelope OCP, the 4-state collocation, and the single-track OCP all share one
ladder. Add `EnvelopeOcp::solve_from(x0, cfg)` as a trivially safe refactor of `solve`.

### 3.4 (c) Problem size, and what it implies for the CG budget

Layout assumption: node-major, no `dt` decision variables (the `s`-domain fixes the mesh —
this is a real simplification over `collocation.rs`'s `7N−1` layout, which carries `N−1`
`dt`s). Periodic closure ⇒ `N` intervals for `N` nodes.

| Problem | vars/node | `n_vars` | `n_eq` | `n_ineq` (typ.) |
|---|---:|---:|---:|---:|
| GN collocation, `N=50` (the deadlock reference, `7N−1`) | 7 | **349** | 200 | 150 |
| Envelope OCP, `N=36` (Spa `N*`) | 5 | **180** | 108 | 36 |
| Envelope OCP, `N=60` (config default) | 5 | 300 | 180 | 60 |
| **Single-track, `N=300`** | 8 | **2 400** | **1 800** | ~900 |
| **Single-track, `N=600`** | 8 | **4 800** | **3 600** | ~1 800 |
| Four-wheel, `N=300` | 13 | 3 900 | 3 000 | ~1 500 |
| Four-wheel, `N=600` | 13 | 7 800 | 6 000 | ~3 000 |

(vars/node: single-track `{n, xi, v_x, v_y, r, delta}` + `{u_delta, F_x}` = 8. Four-wheel
adds 4 wheel-spin states and splits drive/brake = 13. `n_ineq` counts tire feasibility
per axle/corner + power; track edges and steer/rate limits are **box bounds**, which the
IP solver handles natively and which do not enter `n_ineq`.)

**The implication for the current CG budget is unambiguous.** Take the envelope OCP's
documented wall at `N ≈ 44` as the reference point. With `cond ~ N²` and unpreconditioned
CG iteration count scaling as `sqrt(cond) ~ N`:

- `N = 300` is **~7× the linear-solve difficulty** of the wall.
- `N = 600` is **~14×**.

And the starting point is not "CG is comfortable" — it is `cg_max_iter = 250` **already
saturating on every Newton step at `N = 32`** (§3.1). So the honest statement is:
**the current CG budget is not marginally short at the target mesh, it is short by more
than an order of magnitude, and raising it is documented to make things worse.** Scaling
`cg_max_iter` to ~3500 is not a plan; it multiplies a 250-iteration inner loop by 14 across
hundreds of outer iterations and, per §B.2's direct experiment, does not even fix the
`N ≥ 48` case.

**This is why `kkt-precond` is the critical path and must land before, not after,
`single-track-ocp` is asked to run at target mesh.** A Rung-2 block-tridiagonal
preconditioner replaces `sqrt(cond) ~ N` with `O(1)`-in-`N` CG counts, which is the
difference between "target mesh is reachable" and "target mesh is not".

---

## 4. Derivatives at scale

### 4.1 Current machinery — inventory

**Forward duals only.** `apex_math::Dual` (`dual.rs`) is a single-seed forward dual;
`apex_math::Float` (`float.rs`) is the generic trait both `f64` and `Dual` implement
(`equations_of_motion.md` §5). There is **no** reverse mode, **no** vector dual, **no**
hyper-dual, and **no** `apex-autodiff` crate.

**Where duals are used:**

| Consumer | Site | Seeding pattern |
|---|---|---|
| Envelope OCP node Jacobian | `envelope_ocp.rs:312` `dynamics_jac` | 5 seeds → dense `3×5` block |
| Envelope inequality | `envelope_ocp.rs:341` `envelope_ineq` | **hand-written analytic gradient** (not AD) |
| Envelope `rho` gradient | `envelope.rs:469` `rho_grad` | 3 single seeds into the C1 `HermiteGrid` |
| Collocation equality Jacobian | `collocation.rs:1009` `autodiff_equality_jacobian` | **13 seeds per interval**, banded CSR assembly |
| Collocation inequality Jacobian | `collocation.rs:1172` | 3 seeds per node for the grip circle; analytic ±1 for edges |
| Estimator (EKF/RTS) | `apex-correlate::estimator` via `BicycleModel::derivatives_generic` | per-state seeds |

**Assembly cost and structure.** `docs/math/collocation.md` §6 measures the autodiff
equality Jacobian at `N = 50` at **~32 µs** versus **~1.7 ms** for the numerical one — a
**52×** speedup — and §5 records the sparsity: at `N = 100`, `≈5 156` structural non-zeros
out of `279 600` dense entries, **1.8 % density**, banded. So sparsity is *already*
exploited, structurally and by construction, in both the assembly loop and the CSR storage.

**Autodiff test coverage** (what the existing tests actually verify):

- `tire/pacejka.rs:400` `generic_f64_matches_concrete`; `:419` derivative-at-zero equals
  cornering stiffness; `:435` derivative vanishes at the curve peak; `:464` `dFy/dFz` sign
  and magnitude; `:485` combined generic matches concrete; `:494` combined slope < pure
  slope.
- `tire/combined_slip.rs:429` `smooth_is_differentiable_at_grip_limit` — autodiff vs
  central finite difference at the grip boundary, `< 1e-3` relative.
- `bicycle.rs:338` `generic_matches_concrete` (value equivalence, incl. below the `vx`
  guard); `:371` `dual_jacobian_column_is_finite_and_plausible` (`∂v̇_y/∂v_y < 0`).
- `collocation.rs` carries a `numerical_jacobian_fd` helper (`:1552`) used to cross-check
  the autodiff Jacobian.

**Gap in coverage:** the tests verify *derivative values at points*. Nothing verifies the
Jacobian's **sparsity pattern** is complete — i.e. that no structurally-nonzero entry is
silently dropped. The `d.dual.abs() > 1e-15` filter (`collocation.rs:1085`) drops entries by
*value*, which is correct for storage but means a genuinely-missing coupling would look
identical to a legitimately-zero one. Worth a whole-Jacobian FD comparison test when the
single-track Jacobian is written.

### 4.2 What the IP solver actually consumes today

This is the decisive question, and the answer narrows the requirement sharply. From
`ipm.rs`:

- `equality_jacobian(x) -> CsrMatrix` and `inequality_jacobian(x) -> CsrMatrix`, used
  **only** for: `mul_vec` / `transpose().mul_vec` (`:680–706`), and `row_entries` to build
  the scalar Jacobi diagonal (`:719–734`) and the column scaling (`:326`).
- `objective_gradient(x) -> Vec<f64>`.
- `objective_hessian_vec(x, v) -> Vec<f64>` — and it is **gated off by default**: the
  Hessian model is `H = w_f·H_f + rho·JeqᵀJeq + Jineqᵀ Σ_I Jineq + diag(Σ) + reg·I`, with
  the header stating "`H_f` is the objective Hessian-vector product (**default zero →
  Gauss-Newton on the constraints**)".

**Therefore: nothing in the solver consumes a second derivative of the constraints, and
nothing consumes an exact Hessian.** The roadmap's "sparsity detection via coloring,
forward-over-reverse Hessians" would produce artifacts the current solver has no input for.

### 4.3 Assessment of the roadmap's `apex-autodiff` plan

The roadmap (`apex14_production_roadmap.md` §3.2) states: *"forward duals don't scale to
thousands of NLP variables"* and prescribes a reverse tape + coloring + forward-over-reverse.

**The premise is wrong as applied to this codebase.** It is true for *naive* AD that seeds
the whole decision vector. It is false here, because the assembly **already** exploits the
band: `autodiff_equality_jacobian` costs **13 seeds per interval** and
`dynamics_jac` costs **5 seeds per node** — both `O(stencil width)`, **independent of
`N`**. Total cost is `O(N × stencil)`, i.e. **linear in problem size**, which is the same
asymptotic a reverse tape would give, without a tape.

Concretely for single-track at `N = 600`: the per-interval stencil is 16 columns
(6 states + 2 controls at node `k`, same at `k+1`; no `dt` variable in the `s`-domain).
That is `600 × 16 = 9 600` RHS evaluations per Jacobian assembly. Each RHS is two Pacejka
calls and some trigonometry. Extrapolating the measured `32 µs @ N=50, 13 seeds` figure
gives single-digit milliseconds per Jacobian — against a Newton step that runs up to 250
matrix-free CG iterations. **Derivative assembly is not, and will not be, the bottleneck.**
The bottleneck is the linear solve (§3).

### 4.4 Recommended staged path

**Stage 1 — nothing. Ship single-track on the existing forward-dual pattern.**
Copy `envelope_ocp.rs::dynamics_jac`'s single-seed loop, widened to the 16-column
interval stencil. Zero new infrastructure. **Then measure**: instrument the fraction of
wall time in `equality_jacobian` versus `pcg`. Evidence, not assumption, decides Stage 2.

**Stage 2 (only if Stage 1 profiling justifies it) — `DualN<K>`, a multi-seed forward
dual.** A `[f64; K]` derivative array instead of a scalar `dual`, implementing the same
`Float` trait. One RHS pass computes `K` columns at once, cutting assembly by ~`K×` by
amortizing the shared real-part arithmetic. For `K = 16` this collapses the per-interval
cost to one pass. It is **additive** to `apex-math` (a new type, `Float` unchanged), so
the bit-identity chain is untouched, and it is a few hundred lines with no tape, no graph,
no coloring. This is the **minimal derivative upgrade**, and the answer to the brief's
question is: **structured forward-mode, not reverse.**

**Stage 3 (deferred, evidence-gated) — reverse tape / exact Hessians.** Justified only if
the solver is first upgraded to *consume* second derivatives — i.e. if the Gauss-Newton
Hessian model is shown, with measurements, to be the limiting factor in outer-iteration
count. Note the ordering dependency: **exact Hessians are useless until `objective_hessian_vec`
is wired to something that wants them.** Since the objective here is
`∫ S ds` — smooth and near-linear in the states — the Gauss-Newton model is a good fit and
this stage may never be needed. Record it as an open question, not a plan.

**Also worth doing in Stage 1 (cheap, high value):** a whole-Jacobian finite-difference
comparison test for the single-track equality Jacobian, closing the sparsity-completeness
gap noted in §4.1. Note the precedent for a subtlety here: the curvature chain rule.
`collocation.rs:1091–1152` adds an explicit `∂defect/∂κ · dκ/ds` correction to the
`s`-columns because the dual sweep holds `κ` fixed. **The `s`-domain OCP does not have this
problem** — `s` is not a decision variable; `kappa(s_k)` is a per-node constant
(`envelope_ocp.rs:242 node_curvatures`). One less thing to get wrong, and worth stating
explicitly so nobody ports the correction across.

---

## 5. Warm-start chain

### 5.1 Today's chain, honestly mapped

| Link | Exists? | Detail |
|---|---|---|
| **ML raceline warmstart** | **exists, produces the right thing, but is not wired into any solver path** | `apex_ml::warmstart::ml_predict_profiles` (`warmstart.rs:61`) returns `MlProfiles { speeds, offsets }`; `ml_initial_guess` (`:116`) packs it via `CollocationOptimizer::initial_guess_from_profiles` (`collocation.rs:382`). `MlWarmstart::load` (`:147`) returns `Option`, designed to fall back. **Grep of `bins/` and `crates/` finds no caller outside `apex-ml` itself and its own tests** — the CLI does not use it. So the link exists as a library capability, not as a wired pipeline. |
| **Fixed-line QSS** | **exists and is the live default** | `qss_lap_sim` (`qss.rs:211`); used by `CollocationOptimizer::initial_guess` (`collocation.rs:289`) and by `EnvelopeOcp::warm_start` (`envelope_ocp.rs:259`). Tire-aware variant `qss_lap_sim_tire` (`qss.rs:722`) seeds the load-sensitive formulations (`collocation.md` §7). 3D variants at `qss.rs:477,495`. |
| **Envelope-OCP trajectories** | **exist as results; not consumable as a warm start** | `EnvelopeOcpResult` carries `{stations, offsets, headings, speeds, ax, kappa_cmd}` (`envelope_ocp.rs:113`). But there is **no** `EnvelopeOcp::solve_from(x0)` and **no** lift function into any other problem. |

**Two concrete papercuts to fix while lifting:**

- **Node-indexing mismatch.** `ml_predict_profiles` and `initial_guess_from_result` use
  open-lap spacing `s_k = L·k/(n−1)` (`warmstart.rs`, `mesh_refinement.rs:169`), while
  `EnvelopeOcp::node_stations` uses **periodic** spacing `s_k = L·k/n`
  (`envelope_ocp.rs:233`). Any lift must resample, not reindex.
- **`EnvelopeOcp::solve` ignores external guesses** (`:446–452`). Adding `solve_from` is a
  prerequisite for both mesh continuation (§3.3) and fidelity continuation.

### 5.2 Lifting `{n, xi, v}` → `{n, xi, v_x, v_y, r, delta}`

The envelope-OCP solution is a *kinematic* trajectory; the single-track guess needs a
*dynamically consistent* one. The mapping is well-determined and every ingredient exists:

| Target | From | Note |
|---|---|---|
| `n` | `offsets[k]` directly | resample periodic→periodic; identical if `N` matches |
| `xi` | `headings[k]` **minus** `beta` | the envelope `xi` is the *velocity* heading error; the single-track `xi` is the *chassis* yaw. They differ by the sideslip. |
| `v_x` | `speeds[k]·cos(beta)` | |
| `v_y` | `speeds[k]·sin(beta)` | `beta` from the trim solver, below |
| `r` | `kappa_cmd[k] · speeds[k]` | yaw rate = path curvature × speed, the kinematic identity the envelope OCP is built on (`envelope_ocp.rs:30` `a_y = v²·kappa_cmd`) |
| `delta` | trim solver | |
| `u_delta` | periodic finite difference of `delta` over `ds` | |
| `F_x` | `m·ax[k]` + drag + rolling | invert `envelope_ocp.rs:280–281`'s warm-start relation |

**`beta` and `delta` come from `apex_physics::trim::solve_operating_point`
(`trim.rs:332`).** That is a 3-DOF Newton trim returning an `OperatingPoint` (`:27`) —
exactly the steady `{delta, beta, r}` for a given `(v, a_x, a_y)`. Cost is **~464 ns per
combined solve** (CLOSE.md §1), so calling it once per node at `N = 600` costs
~0.3 ms. This is a very cheap, physically principled lift, and it reuses shipped,
11-test-covered machinery rather than inventing an approximation.

**Caveat to state in the design doc:** the lift is *steady-state* by construction, so on
transients (the exact thing the dynamic OCP exists to capture) it is only approximately
consistent. That is fine — it is a warm start, not a solution — but the residual defect at
`x0` should be reported so the ladder's starting quality is visible rather than assumed.

### 5.3 Continuation hooks — what exists vs what is scaffolding

| Hook | Status |
|---|---|
| **Mesh continuation** | **Partial scaffolding.** `mesh_refinement.rs` exists but is GN-only and 4-state-only (§3.3). `CollocationOptimizer::optimize_ip_from` exists. Needs the generic ladder. |
| **Fidelity continuation** (point-mass → single-track → four-wheel) | **Nothing.** No shared representation across fidelities, no lift functions. §5.2 is the first rung and must be built. |
| **μ-continuation** (solve at reduced grip, step up) | **Nothing named, but the knobs exist.** `PacejkaTire.lateral.mu` / `.longitudinal.mu` (`pacejka.rs:16`) and `CarParams` μ are plain fields; `MuScaleGrid` (`apex-track::grip_grid`) and `RibbonStation.mu_scale` provide a *spatial* μ scale. A μ-continuation driver is a loop over car/tire clones re-solving from the previous solution — trivial **once `solve_from` exists**. Note the μ-continuation direction: start at *reduced* grip (an easier, less-active-constraint problem) and step up. |
| **Regularization homotopy** (heavily regularized → pure minimum-time) | **Config surface exists.** `EnvelopeOcpConfig.rate_weight_ax` / `rate_weight_kappa` (`envelope_ocp.rs:91–94`) are exactly the control-rate penalties the roadmap's homotopy would anneal. The dynamic OCP should carry equivalents from day one. |

**Common prerequisite for three of the four: `solve_from(x0)`.** It is a small refactor
and it unblocks mesh, fidelity, and μ continuation simultaneously. Do it first.

---

## 6. Validation assets

### 6.1 The QSS steady-corner cross-check

**Machinery that computes the reference: `apex_physics::trim::solve_operating_point`
(`trim.rs:332`).** A 3-DOF Newton trim producing an `OperatingPoint` — the exact steady
`{v, delta, beta, r}` at a given operating condition. Evidence it is trustworthy: 11
tests, the symmetric point is **bit-identical to the legacy trim**
(`straight_line_matches_legacy_trim_bitwise`), and there is a frozen snapshot test
(`solve_trim_gz_default_matches_frozen_snapshot`). Secondary reference: `qss_lap_sim`
(`qss.rs:211`) on a constant-curvature track.

**Track asset:** `apex_track::generators::circle_track(radius, width, n)` — already the
validation vehicle for the envelope OCP, whose `circle_matches_closed_form` test reached
**15.038 s vs 15.041 s analytic (0.02 %)** (CLOSE.md §1).

**What the test should assert.** On a long constant-radius corner, at steady state the
`s`-derivatives of the dynamic states must vanish and the states must match the trim:

- `|v_y'| , |r'| , |v_x'| ≈ 0` in the corner interior (excluding entry/exit transients),
- converged `{v_x, v_y, r, delta}` match `solve_operating_point` at the achieved `a_y`,
- lap time matches `qss_lap_sim` on the same circle to a stated tolerance.

**Inherit one hard-won lesson.** `real-track-convergence.md` Part C documents that
`silverstone_tuned_reaches_tight` was **platform-sensitive** — glibc vs ucrt rounding
through hundreds of inexact CG steps flipped the *terminal status* between `Optimal` and
`MaxIter` without the solution being marginal. The fix was to assert a **quantitative
feasibility bound**, not `assert_eq!(status, Optimal)`, and to pick `N` for the widest
margin. **The dynamic-OCP validation tests must be written that way from the start**, and
should carry a companion negative test (the `silverstone_untuned_still_fails_near_feasibility`
pattern) so the assertion cannot silently stop discriminating.

### 6.2 The Dal Bianco / Lot GP2 Barcelona literature target

**Is the track imported and usable? Yes — but note the naming.** The literature target is
Barcelona; the repo file is **`tracks/catalunya.json`** (Circuit de Barcelona-Catalunya).
It is TUMFTM-derived, **gitignored** (`/tracks/*.json`, redistribution-restricted, see
`tracks/README.md`) but present locally, and it is **already exercised** by the envelope
OCP: `N* = 32`, 71.843 s envelope-OCP vs 95.531 s fixed-line QSS (CLOSE.md §1). A 3D
variant (`catalunya.elevation.json` / `_3d.json`) is **not** present, unlike Silverstone
and Spa — so this target is 2D, which is consistent with the §2.3 recommendation.

**What car config approximates GP2?** GP2 (now F2) sits between F3 and F1: ~690 kg with
driver, ~600–620 hp, markedly less downforce than F1. Available configs
(`cars/`): `f1_2024_default.toml`, `f1_2024_calibrated.toml`, **`f3_car.toml`**
(mass 690, `max_drive_force` 6000 N, `lift_coeff` 1.80, `drag_coeff` 0.85, μ 1.45), plus
five per-circuit fitted F1 cars including `catalunya_2024q_fitted.toml`.

**`f3_car.toml` is the right *basis*, not the right *answer*.** GP2 mass is close, but GP2
power and downforce are meaningfully above F3. The honest approach is a new
`cars/gp2_approx.toml` derived from `f3_car.toml` with power and `lift_coeff` raised
toward the literature's stated vehicle data, **with every deviation from the paper
documented in the validation page**. And the comparison must be framed as the roadmap
itself frames it — *"reproduce trajectory character and sector behavior"* — **not** a
lap-time match. Two reasons to hold that line: (i) the car is an approximation, and
(ii) `real-track-convergence.md` §B.3 establishes that lap-time *magnitudes* from this
solver family are mesh-dependent and not converged, so a lap-time claim would be unearned
on independent grounds.

**Risk to flag now:** this item depends on obtaining the published trajectory/sector data.
That is an external dependency with no in-repo asset behind it. **Mark `validation-gp2`
optional-if-evidence-allows** — the steady-corner check (§6.1) is the *load-bearing*
correctness evidence and is fully self-contained.

### 6.3 The Spa transient / driver residual — and a needed reframing

**Pages that define the acceptance evidence:**
[`../../validation/correlation_spa_envelope.md`](../../validation/correlation_spa_envelope.md)
(the calibration-free null result),
[`../../validation/correlation_spa_2024q.md`](../../validation/correlation_spa_2024q.md)
(the original 3D fit), and
[`../../validation/correlation_summary.md`](../../validation/correlation_summary.md).

**The established facts.** The 3D Spa fit left `power_scale` at a uniquely low **0.802**
and a persistent **descent over-carry** (Pouhon→Stavelot, `s ≈ 3890–4680`): the simulated
lap is **110.303 s vs measured 113.159 s = −2.856 s**. The envelope study then established,
**calibration-free**, that the descent is **not grip-limited** — median grip utilization
**0.45**, only **8 %** of stations above 0.9 — so *no grip law can move it*. Residual
formally isolated to **longitudinal / transient / driver**. Measured median throttle
through the descent: **0.71**.

**The contradiction that must be resolved before this is scoped as a deliverable.**
`CLOSE.md` §3 assigns the Spa residual to the dynamic OCP and calls it *"its business
case."* But the dynamic OCP is specified — in this very brief — as having **no driver
model**, and it is a **minimum-time** solver. The measured driver was at 0.71 median
throttle down a descent where he had over half his grip in reserve; a minimum-time solver
will go **faster** there, not slower. **A minimum-time dynamic OCP therefore cannot
reproduce the measured descent speed, and "the dynamic OCP recovers the descent
over-carry" is not an achievable acceptance criterion as literally worded.**

**What it *can* do, stated as a falsifiable test.** Run the dynamic OCP over the Spa
descent window with realistic longitudinal dynamics — power hyperbola, load transfer,
transient tire response — and compare:

- **Outcome A (expected):** the min-time dynamic OCP is still *at or faster than* the QSS
  sim through the descent. Conclusion: **the residual is confirmed driver behaviour, not
  missing vehicle physics.** The `power_scale = 0.802` outlier is then correctly
  reinterpreted as an *effective* parameter absorbing driver conservatism, and the
  correlation pipeline's model is exonerated. **This is a genuine, publishable result and
  it closes the item.**
- **Outcome B:** the dynamic OCP is meaningfully *slower* than QSS through the descent —
  i.e. transient/load-transfer physics that QSS ignores genuinely costs time there.
  Conclusion: QSS was optimistic, and part of the residual **is** model physics after all.

**Either outcome is a result.** The item should be scoped as *"bound and attribute the Spa
descent residual"*, not *"recover"* it, and the acceptance evidence is the A-vs-B
determination with the descent-window timing table. The measured-throttle and
grip-utilization numbers above are the calibration-free anchors that make the argument
hold regardless of which way it lands.

---

## 7. Risk register + work breakdown

### 7.1 Contradictions with planning assumptions

| # | Contradiction | Evidence | Resolution |
|---|---|---|---|
| **C1** | **The Spa business case is not deliverable as worded.** A min-time OCP with no driver model will over-carry the descent *more* than QSS. | Brief specifies "no driver model"; `correlation_spa_envelope.md` measures 0.45 median grip util and 0.71 median throttle through the descent. | Reframe to "bound and attribute", §6.3. **Do this before the item is committed to.** |
| **C2** | **"Forward duals don't scale to thousands of NLP variables"** — the roadmap's stated reason for `apex-autodiff`. | Existing assembly is `O(stencil)` per interval, `N`-independent (`collocation.rs:1009`, `envelope_ocp.rs:312`); measured 32 µs @ `N=50` vs 1.7 ms FD (`collocation.md` §6). | Reject the premise. Staged forward-mode path, §4.4. Reverse tape deferred and evidence-gated. |
| **C3** | **Roadmap §3.2 prescribes sparse LDLᵀ via a new dependency (`faer`).** | `ipm.rs` header states its design constraint: *"with **no new linear-algebra dependencies**"*. wasm-safety is a standing repo constraint (see the web-viewer dep-graph discipline). | Block-tridiagonal preconditioner + existing CG, §3.2. No dependency, wasm-safe, and it targets the diagnosed mechanism, which a generic sparse factorization would not do any better. |
| **C4** | **"The next step is not more nodes"** (`real-track-convergence.md` §5) vs the roadmap's `N = 300–600` performance target. | Both documents are correct. | They reconcile **only if `kkt-precond` lands first**. This is precisely why the preconditioner is ahead of the model on the critical path. Do not schedule a target-mesh single-track solve before it. |
| **C5** | **The envelope OCP's per-track `N*` knob is a feasibility artifact, not a converged objective.** Lap times swing ~20 % across `N`; `N = 36/44/48` fail outright on real Silverstone. | `real-track-convergence.md` §B.3. | The dynamic OCP must **not** inherit the per-track-`N` pattern. Mesh-convergence of the objective (not just feasibility) is an explicit acceptance criterion, and any lap-time claim before it is met carries the same "directional, not magnitude" caveat CLOSE.md applies. |
| **C6** | **"Reuse the 7-DOF dynamics"** reads as reusable; the 10-state model is `f64`-only, global-frame, and carries a `signum`. | `seven_dof.rs:37,124`; no `OdeSystemGeneric` impl. Compounded by the `seven_dof_derivatives_generic` naming trap (§1.1). | `four-wheel-ocp` is a **from-scratch** RHS, not a port. Size it accordingly and keep it off the critical path. |
| **C7** | **`Float` cannot express the four-wheel model's discontinuities.** No `signum`, no `tanh`. | `float.rs:34–56`. | Extending `Float` is additive but recompiles all of `apex-physics` — the bit-identity chain must be re-verified (§7.2). Another reason to defer four-wheel. |

### 7.2 Blast radius on the 848-test suite and the goldens

Current baseline: **848 `#[test]` attributes** across `crates/` and `bins/` (CLOSE.md
records 841 passing at close; the delta is subsequent work). **Three live goldens**:
`golden_oval_qss`, `golden_silverstone_qss`, `golden_circle_optimize`
(`bins/apex-cli/tests/golden_lap.rs:390,401,423`), plus **three `#[ignore]`d**
(`:489,521,552`) — the optimize goldens paused by the GN bound deadlock, which CLOSE.md §2
says may be **unpaused as new fixtures** when the dynamic OCP needs them.

| Change | Blast radius | Mitigation |
|---|---|---|
| **New `dynamic_ocp` module + new CLI flag** | **Zero.** Purely additive. | — |
| **`ipm.rs` preconditioner** | **High and subtle.** Risks `ip_resolves_gn_bound_deadlock` (locked at `eq = 1.287e-7`), `determinism_bitwise_history`, all 6 `envelope_ocp` tests, every number in `analysis.md`'s envelope table, and — most sharply — the **setup-envelope rank gate, which is already *marginal* post-bridge (Spearman exactly 0.900, 5/8 variants tight)**. A solver change can flip it either way. | **New `IpmConfig` field defaulting to the present scalar Jacobi.** Existing solves must stay **bit-identical**, verified by the determinism test. Then a separate, deliberate decision to switch defaults, with the rank gate re-measured. |
| **Extending `apex_math::Float`** (`signum`/`tanh` for four-wheel) | **Wide recompile, but should be inert.** Adding trait methods with default bodies changes no existing arithmetic. The chain to re-verify: `gz_default_is_bitwise_identical_to_baseline`, `prop_gz_pathway`, `optimizer_budgets_gz_default_bitwise_identical`, `solve_trim_gz_default_matches_frozen_snapshot`, `straight_line_matches_legacy_trim_bitwise`, `bridge_default_car_is_bit_identical`, `bridge_default_car_envelope_is_byte_identical`, plus the frozen content hashes (`tire_hash_frozen_and_field_sensitive`, `aero_hash`, `processed_track_hash`). | Additive-only; run the full byte-stability chain. |
| **`apex-math::DualN<K>`** (Stage 2) | **Low.** New type, `Float` unchanged, `Dual` untouched. | Additive. |
| **Generalizing `mesh_refinement`** | **Moderate.** 4 tests: `interpolation_preserves_constant_speed`, `refinement_on_circle`, `refinement_beats_cold_start_on_oval`, `single_level_matches_plain_gn`. The last two assert cross-solver relationships. | Keep the existing GN entry point as a thin wrapper over the generic ladder so those tests are unchanged. |
| **`EnvelopeOcp::solve_from`** | **Low.** Refactor `solve` to call it with `warm_start()`. | 6 `envelope_ocp` tests must stay bit-identical. |
| **New `cars/gp2_approx.toml`** | **Low**, but note `apex-physics` has **no `CarParams` validation** (recorded in the proptest memory), so a nonsense config fails late and confusingly. | Validate by hand; assert sane derived quantities in the validation test. |
| **Unpausing the ignored optimize goldens** | **Deliberate.** They are paused for a now-retired reason. | Regenerate as **new** fixtures (CLOSE.md §2's explicit guidance), not as a "fix". |

**One repo-hygiene note:** `tracks/*.json` (Catalunya, Silverstone, Spa, Monza, Spielberg)
and `apex14_production_roadmap.md` are **gitignored**. Any validation test must therefore
either skip gracefully when the track file is absent (the `silverstone_tuned_reaches_tight`
precedent: it uses the *synthetic* circuit specifically so it *"[does not] depend on the
gitignored real track data"*) or live outside the committed test suite. **Design the
Catalunya/Spa validation tests around this from the start** — it has already bitten this
codebase once.

### 7.3 Dependency-ordered work breakdown

Feature-named, no phase numbers. **CP** = critical path; **OPT** = optional-if-evidence-allows.

```
curvilinear-dynamics ──┬─> rhs-ocp-interface ──┬─> single-track-ocp ──┬─> validation-steady-corner
                       │                       │        ▲             ├─> validation-gp2      (OPT)
                       │                       │        │             └─> validation-spa      (OPT)
                       │                       │   kkt-precond
                       │                       │        ▲
                       │                       │  mesh-continuation
                       │                       │        ▲
                       │                       └─> warmstart-lift
                       │
                       └─> (deferred) curvilinear-dynamics-3d          (OPT)

sparse-derivatives (OPT, evidence-gated)  ─ ─ ─>  single-track-ocp
four-wheel-ocp (OPT)  <── float-smoothing-ext  <── single-track-ocp
```

| Item | CP/OPT | Depends on | Scope | Note |
|---|---|---|---|---|
| **`curvilinear-dynamics`** | **CP** | — | New `docs/math/dynamic_ocp.md`: full `{n,xi,v_x,v_y,r,delta}` `s`-domain transform, the reduction check to the shipped envelope form, singularity/bounds analysis. **No code.** | Cheapest item on the critical path and everything else reads from it. Do it first. |
| **`rhs-ocp-interface`** | **CP** | curvilinear-dynamics | `Float`-generic curvilinear single-track RHS + node Jacobian, in the `envelope_ocp::dynamics`/`dynamics_jac` shape. Reuses `PacejkaTire::*_generic`, `axle_loads_generic`. Includes the whole-Jacobian FD test (§4.4). | Small — the model beneath it is already generic (§1.2). |
| **`kkt-precond`** | **CP** | (independent; can start immediately) | Node-major reordering + Rung-1 block-Jacobi, then Rung-2 block-tridiagonal, behind an `IpmConfig` flag defaulting to today's Jacobi. Uses `apex_math::lm::solve_linear`. | **The true critical path** (C4). Validate on the *existing* envelope OCP first — a direct, cheap test is whether it pushes real Silverstone past the documented `N ≥ 44` wall. That is a clean, pre-existing benchmark with published numbers to beat. |
| **`mesh-continuation`** | **CP** | kkt-precond (for value), rhs-ocp-interface | Solver-agnostic `MeshLevel` ladder; `EnvelopeOcp::solve_from`; generalize `mesh_refinement`. | Bottom rung now exists (§3.3) — the documented blocker is retired. |
| **`warmstart-lift`** | **CP** | rhs-ocp-interface | `{n,xi,v}` → `{n,xi,v_x,v_y,r,delta}` via `solve_operating_point`. Report residual defect at `x0`. | Cheap (~464 ns/node) and unblocks fidelity continuation. |
| **`single-track-ocp`** | **CP** | rhs-ocp-interface, kkt-precond, warmstart-lift | The OCP itself + `optimize --dynamic --model single-track`. Carry control-rate regularization from day one. | Do **not** schedule a `N = 300–600` run before `kkt-precond` (C4). |
| **`validation-steady-corner`** | **CP** | single-track-ocp | Long constant-radius corner vs `solve_operating_point` + `qss_lap_sim`. Quantitative bounds, not status equality (§6.1). | **The load-bearing correctness evidence.** Fully self-contained — no external data, no gitignored assets. |
| **`sparse-derivatives`** | **OPT** | single-track-ocp profiling | `DualN<K>` multi-seed forward dual. | **Gate on measurement**, not assumption (§4.4). May well never be needed. |
| **`validation-gp2`** | **OPT** | single-track-ocp, mesh-continuation | `cars/gp2_approx.toml` from `f3_car.toml`; Catalunya; character/sector comparison. | External dependency (literature data); gitignored track (§7.2). Optional by construction. |
| **`validation-spa`** | **OPT** | single-track-ocp | The A-vs-B attribution test of §6.3. | **Reframe before committing** (C1). High explanatory value, low risk of a null result — both outcomes are results. |
| **`float-smoothing-ext`** | **OPT** | — | `signum`/`tanh` (or smooth surrogates) on `Float`; smooth wheel-lift and slip-denominator regularizations. | Prerequisite for four-wheel only. Wide recompile — run the byte-stability chain (§7.2). |
| **`four-wheel-ocp`** | **OPT** | float-smoothing-ext, single-track-ocp, kkt-precond | From-scratch 10-state curvilinear generic RHS (C6). | Largest item, weakest evidence, no dependents. Keep it last and keep it optional. |
| **`curvilinear-dynamics-3d`** | **OPT** | curvilinear-dynamics | Full `Ω_x/y/z` transform. | Explicitly deferred (§2.3). 2D + imposed `g_z(s)` is the recommended first cut, following the proven envelope-QSS pattern. |

**Suggested first three moves, in order:** (1) write `docs/math/dynamic_ocp.md`; (2) start
`kkt-precond` in parallel and validate it against the *existing, already-published*
envelope-OCP `N ≥ 44` wall — a benchmark that exists today and needs none of the new model;
(3) `rhs-ocp-interface`. Nothing before those is well-informed, and (2) is the item most
likely to determine whether the target mesh is reachable at all.
