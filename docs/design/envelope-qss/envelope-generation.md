# Envelope generation — grid, C1 interpolation, content-hash cache

**Status: implemented.** The `apex-physics::envelope` module sweeps
[`solve_operating_point`](trim-solver.md) over a `(theta, v, g_z)` grid to
produce the car's g-g-g performance envelope, stores it with C1-continuous
interpolation, and keys it by content hash. This is the surface the
free-trajectory OCP will constrain against; **no OCP code is in this task.**
Builds on the trim solver ([`trim-solver.md`](trim-solver.md)) and the g_z
pathway ([`gz-pathway.md`](gz-pathway.md)); see [`recon.md`](recon.md) §4.

## 1. Representation — boundary radius `rho(theta; v, g_z)`

For each `(v, g_z)` slice the feasible planar-acceleration set is the region
bounded by the grip / load / power limits [`TrimStatus`](trim-solver.md) reports.
We store its boundary as a **radius function** over the acceleration-plane angle:

```
a_x = r*cos(theta),  a_y = r*sin(theta),  feasible  iff  r <= rho(theta; v, g_z)
```

so the envelope is a scalar field `rho` on the 3-axis grid `(theta, v, g_z)`.

**Why this over a signed-distance grid.** The deciding criterion (task §1) is
**C1 smoothness of the resulting OCP constraint**. The radius form gives the OCP
a single scalar inequality per node,

```
c = |a| / rho(atan2(a_y, a_x); v, g_z) <= 1,
```

which is C1 in the decision variables wherever `rho` is C1 and `|a| > 0` (both
`|a|` and `theta` are smooth in `(a_x, a_y)` away from the origin). A
signed-distance field over the full `(a_x, a_y)` plane is C1 too, but the
constraint then reads a 2-D level-set and recovering `d(constraint)/da` is less
direct; the radius form hands the solver `rho` and `d rho/d(theta, v, g_z)`
straight from the interpolant. Two more properties fall out for free:

- **Brake/drive asymmetry is native.** `rho(0)` (pure drive, actuator-limited)
  and `rho(pi)` (pure braking) simply differ along the angle axis — no
  special-casing. (Measured on the F1 default at `v = 48`, `g_z = 9.81`:
  drive `18.8`, braking `26.4`, lateral `25.4` m/s².)
- **Periodicity closes the loop C1.** `theta` is a **periodic** interpolation
  axis, so `rho` and its `theta`-derivative are continuous across `theta = +-pi`.

**Cost / precondition.** `rho` must be single-valued in `theta` — the feasible
region must be **star-shaped about the origin**. It is: the boundary is monotone
along every ray from the origin (grip demand `m*|a|` and the longitudinal
actuator limit both grow monotonically with `r`), which mirrors the
monotone-boundary property proven in [`trim-solver.md`](trim-solver.md). We do
not assume it — we **assert** it on every ray (§2).

## 2. Boundary location — march + bisect, with an island guard

Along each ray we do **not** take the raw grid transition. `boundary_radius`:

1. Evaluates feasibility at `coarse_steps + 1` uniformly spaced radii out to
   `max_accel` (one pass, fixed order).
2. Finds the first infeasible sample, then **asserts monotonicity**: every
   sample at or beyond it must also be infeasible. A feasible sample beyond an
   infeasible one is a non-star-shaped region — a hard `Err(NonMonotoneRay)`
   reporting the exact `(theta, v, g_z, r)` (fail loudly, per task §2).
3. **Bisects** the last-feasible / first-infeasible bracket to `bisect_tol`. The
   iteration count is `ceil(log2(dr/tol))` — a function of the *spec only*, so
   every ray runs the same count (determinism). A trim solve is ~464 ns, so the
   ~40 bisections per ray are affordable.

Edge rulings: a ray still feasible at `max_accel` clamps `rho = max_accel` (does
not occur for physical cars); a ray infeasible at the origin returns `0`.

## 3. C1 interpolation — `apex_math::interp`

Recon confirmed nothing beyond bilinear (C0) existed in the tree
([`recon.md`](recon.md) §4.2). We added a **generic C1 cubic-Hermite
interpolant** in `apex-math` (the leaf crate, next to the dual-number AD types),
`HermiteGrid` over `D` uniform [`GridAxis`]es, N-D by tensor product.

- **Tangents exact for cubics.** Node tangents are finite differences that are
  exact for cubic data: a centred 5-point stencil `(1,-8,0,8,-1)/12` in the
  interior, one-sided 4-point stencils at the edges, and the wrapped centred
  stencil on periodic axes. Because a cubic Hermite segment is the unique cubic
  matching value + derivative at both ends, an exact-tangent segment
  **reproduces any (tensor-product) cubic exactly** — unit-tested in 1-D and 3-D
  (incl. cross terms) to `< 1e-7`.
- **C1 continuity is structural.** A node's tangent is a single value computed
  once from a fixed stencil, shared by the two cells meeting at it, so value
  *and* first derivative match across every cell face. Verified numerically
  (one-sided derivatives across interior nodes agree) and, on the envelope
  itself, by a proptest of directional-derivative continuity across random
  `theta`/`v` cell boundaries.
- **Derivative path.** Evaluation is generic over `Float`, so passing `Dual`
  coordinates yields `rho` **and** `d rho/d(input)` in one pass — the AD path the
  OCP needs. `Envelope::rho_grad` returns `rho` plus `(d/dtheta, d/dv, d/dg_z)`.
- **Overshoot.** Cubic Hermite is not monotone-preserving; on a step-like
  profile it can wiggle. This is inherent to any C1 polynomial interpolant and is
  **documented, not hidden**: a unit test pins the worst-case over/undershoot on
  a unit step at `< 0.1` and confirms it stays local (two cells from the step the
  value is back inside the data range). The envelope's boundary radius is smooth,
  so this regime is not exercised in practice.

**Chosen representation vs the signed-distance alternative:** radius function,
for the reasons in §1 — the C1 smoothness of `c = |a|/rho` is the deciding
factor, and it is delivered directly by `HermiteGrid`.

## 4. Interpolation error (measured)

Interpolated `rho` vs a direct `boundary_radius` solve at off-grid points:

| Grid `(theta x v x g_z)` | Max relative error | Worst-error sign |
|---|---|---|
| `16 x 6 x 4` (deliberately coarse) | **1.31 %** | over-estimate (`+0.0131`) |
| `24 x 10 x 6` (**default**) | **0.76 %** | **over-estimate** (`+0.0076`) |

The error is dominated by the `v` and `g_z` axes where downforce curves the
boundary; the friction-circle limit is reproduced essentially exactly (see the
point-mass test). Both grids' worst case is an **over-estimate** — the
interpolant reports a slightly *larger* feasible radius than the true boundary.

> **OCP safety-margin note (deferred).** The default grid's worst-case
> over-estimate (`+0.76 %`) exceeds `0.5 %`. An over-estimated `rho` means the
> constraint `|a|/rho <= 1` admits accelerations marginally outside the true
> envelope. The free-trajectory OCP should therefore apply a small safety-margin
> factor (`rho_eff = (1 - epsilon)·rho`, `epsilon >~ 0.01`) so the constraint
> stays conservative between grid nodes. **Decision deferred to the OCP task**;
> recorded here so it is not lost.

## 5. Cache format & versioning

**Key.** `envelope_key = hash(HASH_VERSION ‖ "envelope.v1" ‖
ENVELOPE_CODE_VERSION ‖ car ‖ tire ‖ suspension ‖ aero ‖ grid-spec)`, composing
the four existing model content hashes with the grid spec under a fixed field
order — the same compose pattern as `settings_hash.rs`. Any change to a model,
the grid spec, tolerances, or the `ENVELOPE_CODE_VERSION` tag moves the key.

**File format.** A self-describing versioned binary (`.apexenv`): 8-byte magic
`APXENV01`, the 32-byte key, the spec scalars (little-endian `u64` resolutions +
IEEE-754 `f64` bits for the ranges/tolerances), then the row-major `rho` node
values as raw `f64` bits. Little-endian throughout, fixed order — **byte-identical
for identical inputs**. Chosen over Parquet because it is fully deterministic and
dependency-free while staying consistent with the workspace's bit-exact float
policy (`hash.rs`); the telemetry Parquet path is for channel tables, not this
small structured grid. `Envelope::save/load/load_verified` and
`generate_cached` (filename = key hex) provide the disk layer.

**Determinism.** Fixed iteration order; the Rayon sweep (under the existing
`parallel` feature, sequential fallback for wasm) is over embarrassingly parallel
rays `collect`ed back in grid order, so the output is **independent of thread
count** — asserted byte-identical across 1/4/8 worker threads. Per the recon
note: at tens of ms per regen the cache is primarily a **reproducibility
artifact** (hash → identical bytes), not a compute saver.

## 6. Performance (release, dev machine)

| Grid | Operating points | Generation | File size |
|---|---|---|---|
| `24 x 10 x 6` (**default**) | 1 440 | **~11 ms** | 11 640 B |
| `48 x 16 x 8` | 6 144 | ~34 ms | 49 272 B |

Cache hit (load + verify) is ~0.2 ms.

## 7. CLI

```
apex-14 envelope [--car <toml>] [--calibrated]
                 [--v-range MIN:MAX] [--gz-range MIN:MAX]
                 [--resolution THETA:V:GZ]
                 [--cache-dir <dir>] [--svg <path>]
```

Generates (or loads) the cached envelope, prints the key / file / a sanity read
of `rho` at the drive/brake/lateral directions, and optionally writes a **g-g
diagram SVG** — boundary polygons at three speed slices, with 1-g reference
rings, reusing the telemetry SVG conventions (a forced `RunMetadata`
`<metadata>` provenance element; the envelope key sits in the settings slot,
`"envelope.no-track"` in the track slot since there is no track).

## 8. Tests

- **Point-mass limit** (`point_mass_friction_circle`): load-sensitivity + aero
  off, low CoG, generous actuator limits ⇒ `rho = base_mu * g_z` for all `theta`
  (isotropic friction circle), and linear in `g_z` (`rho(2g)/rho(g) = 2`).
- **Aero growth** (`aero_grows_cornering_grip_with_speed`): cornering `rho` at
  `v = 80` exceeds `v = 15` by the downforce contribution.
- **Interpolation error** (`interpolation_matches_direct_solve`): §4, `< 5 %`
  gate, measured 1.31 %.
- **C1** (`envelope_integration.rs` proptests): directional-derivative continuity
  of the interpolated constraint across random `theta`/`v` cell boundaries.
- **Determinism** (`generation_is_deterministic_byte_identical`,
  `byte_identical_across_thread_counts`, `round_trip_bytes`): byte-identical
  regeneration, thread-count independence, serialization round-trip.
- **Key sensitivity** (`key_is_model_and_spec_sensitive`) + interpolant unit
  tests (11) in `apex_math::interp`.

Existing suite: **783 tests + 3 goldens unchanged** (`golden_oval_qss`,
`golden_silverstone_qss`, `golden_circle_optimize` all pass untouched); the 26
new tests bring the total to 809 (point-in-time at this task's landing; the current
workspace count is **841** — see [`CLOSE.md`](CLOSE.md)). All additive — no existing code path changed,
so no golden moves and no `PHYSICS_CHANGE.md` entry (this task produces no
simulation-output drift).
