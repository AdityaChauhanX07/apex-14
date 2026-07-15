# 3D curved-ribbon track geometry

**Status: COMPLETE for the QSS/point-mass fidelity (3D track model work).** This
document defines the 3D ribbon geometry, its moving frame, the 3D point-mass
QSS physics that consumes the generalized curvatures (grade force, banking,
vertical-curvature load — §5), and the `mu_scale(s, n)` grip-scaling grid
(§5.8). Higher fidelities (single-track / four-wheel / 14-DOF) do not yet
consume any of this — see §5.9 Deferrals, and `PHYSICS_CHANGE.md`, for what's
scoped out and why. This file is extended, not "completed," as later
fidelities and the dynamic OCP formulation pick up the deferred items.

Implementation: [`apex_track::ribbon3d`](../../crates/apex-track/src/ribbon3d.rs).

## 1. Centerline and moving frame

A track is a centerline curve `r(s) = (x(s), y(s), z(s))` parameterized by arc
length `s`, carrying an orthonormal right-handed **moving frame** `{t, n, m}`:

- `t(s)` — unit **tangent**, the direction of travel, `t = dr/ds`.
- `n(s)` — unit **left-lateral**, across the ribbon, positive toward track-left.
- `m(s)` — unit **surface normal**, out of the road surface, positive "up".

Right-handed: `t × n = m`.

We build the frame from three **road angles**:

- heading `ψ(s)` — azimuth of the tangent in the horizontal plane,
- grade `θ(s)` — pitch of the tangent above horizontal (elevation gradient),
- bank `φ(s)` — roll of the surface about the tangent (banking/camber).

$$
t = (\cos\theta\cos\psi,\; \cos\theta\sin\psi,\; \sin\theta)
$$

The unbanked lateral and normal come from the horizontal left `l_0 = \hat z \times
t` (normalized) and `m_0 = t \times l_0`:

$$
l_0 = (-\sin\psi,\; \cos\psi,\; 0), \qquad m_0 = t \times l_0
$$

Banking rotates `{l_0, m_0}` about `t` by `φ`:

$$
n = \cos\phi\, l_0 + \sin\phi\, m_0, \qquad
m = -\sin\phi\, l_0 + \cos\phi\, m_0
$$

## 2. Generalized curvatures (the Darboux vector)

The frame rotates along the curve. Its rotation rate, expressed in the **body
frame**, is the Darboux vector `Ω = (Ω_x, Ω_y, Ω_z)`, defined by

$$
\frac{d}{ds}\begin{pmatrix} t \\ n \\ m \end{pmatrix}
=
\begin{pmatrix}
0 & \Omega_z & -\Omega_y \\
-\Omega_z & 0 & \Omega_x \\
\Omega_y & -\Omega_x & 0
\end{pmatrix}
\begin{pmatrix} t \\ n \\ m \end{pmatrix}
$$

i.e. `t' = Ω_z n − Ω_y m`, `n' = −Ω_z t + Ω_x m`, `m' = Ω_y t − Ω_x n`. The three
components are the **generalized curvatures**:

- `Ω_z = t'·n` — **horizontal / yaw curvature** (the classical 2D `curvature`),
- `Ω_y = −t'·m` — **pitch rate** (elevation / grade change),
- `Ω_x = n'·m` — **roll / banking rate**.

Recovery (used by the importer / analytic constructor, by central differences of
the frame):

$$
\Omega_z = t'\cdot n, \qquad \Omega_y = -\,t'\cdot m, \qquad \Omega_x = n'\cdot m
$$

## 3. Flat degenerate case

A flat track has `z = 0`, `θ = 0`, `φ = 0`. Then `t = (\cos\psi, \sin\psi, 0)`,
`n = (-\sin\psi, \cos\psi, 0)`, `m = (0,0,1)`, and

$$
\Omega_x = 0, \qquad \Omega_y = 0, \qquad \Omega_z = \frac{d\psi}{ds} = \kappa
$$

so `Ω_z` **is** the 2D signed curvature `κ = dψ/ds`. The implementation stores
the exact 2D `curvature` values in `Ω_z` and reuses the identical interpolation
arithmetic, so a flat ribbon reproduces the legacy 2D queries **bitwise**
(`ribbon3d::tests::flat_exact_*`). This is what keeps the golden fixtures
byte-stable while the 3D fields are unused.

**Why an additive `Ribbon3d`, not a `Track` rewrite.** Two migration
strategies were available for 3D: (a) extend `Track`/`TrackSegment` in place
with `z`/`grade`/`bank` fields, or (b) add a new, parallel `Ribbon3d` type and
leave `Track` untouched. (b) was chosen, for two concrete reasons, not just
caution:

1. **`TrackSegment` carries a `ContentHash` impl that destructures the struct
   by name** (`apex_math::ContentHash for TrackSegment`, `crates/apex-track/src/types.rs`)
   specifically so that adding a field is a *compile error* until the hash
   function is updated — a forcing function against silently changing what
   `processed_track_hash` covers. Extending `TrackSegment` in place would
   mean every 3D field either joins the geometry hash domain (changing the
   hash of every existing 2D track, a breaking change to anything pinning a
   hash) or is deliberately excluded (a judgment call repeated per field,
   with no structural reason to get it right). A separate `Ribbon3d` sidesteps
   the question entirely: the hash domain stays exactly what it was.
2. **`Track`'s fields are public and read directly across a dozen crates**
   (QSS, the optimizer, the viewer, `apex-correlate`, …) with no accessor
   layer. A facade/rewrite would need every one of those call sites to either
   keep compiling against 2D semantics (defeating the point of a real 3D
   type) or be migrated in lockstep (a large, high-blast-radius change with
   no way to prove each call site was updated correctly). Additive `Ribbon3d`
   means existing call sites are simply unaffected — proven, not assumed, by
   the bitwise `flat_exact_*`/`qss::flat_ribbon_qss_bitwise_matches_track`
   tests above and in §5.6.

Both reasons are really the same shape: (b) makes "did I break anything"
machine-checkable (compile error, or a bitwise test), where (a) would have
made it a matter of careful review.

## 4. Analytic check cases

Two closed-form cases pin the frame and curvature math
(`ribbon3d::tests`):

### Helix (horizontal-transport frame, no bank)

`r(u) = (R\cos u, R\sin u, h u)`, arc-length rate `L = \sqrt{R^2 + h^2}`. The
generalized curvatures are **constant**:

$$
\Omega_x = \frac{h}{L^2}, \qquad \Omega_y = 0, \qquad \Omega_z = \frac{R}{L^2}
$$

### Banked circle (constant bank β, flat plane)

`r(u) = (R\cos u, R\sin u, 0)` with constant bank `φ = β`:

$$
\Omega_x = 0, \qquad \Omega_y = \frac{\sin\beta}{R}, \qquad \Omega_z = \frac{\cos\beta}{R}
$$

(The nonzero `Ω_y` is the Darboux pitch component induced by rolling the frame on
a turning curve; the centerline itself stays flat. This is a body-frame
generalized curvature, not a literal elevation gradient — the two coincide only
in the small-angle / straight regime.)

## 5. 3D point-mass QSS physics

The QSS/point-mass lap model (`apex_physics::qss_lap_sim_3d`) consumes the road
angles `(θ, φ)` and the **vertical (pitch) curvature** `κ_v ≡ dθ/ds` — computed
directly from the grade channel, **not** the raw Darboux `Ω_y` (which conflates
banking-on-a-curve, §4). On the zero-bank real data `κ_v` and `Ω_y` coincide in
magnitude; on a banked flat turn `κ_v = 0` correctly while `Ω_y ≠ 0`. Let
`κ = |Ω_z|` be the horizontal curvature and `sin θ = dz/ds` the grade.

### 5.1 Normal load (surface-normal force balance)

Resolving gravity and the centripetal reaction along the surface normal `m`
(using `ẑ·m = cosθ cosφ` from §1):

$$
N = m\left(g\cos\theta\cos\phi \;+\; v^2\kappa\,\sin\phi \;+\; v^2\kappa_v\right) + F_\text{df}(v)
$$

- `m g cosθ cosφ` — gravity's component onto the surface (grade + bank reduce it),
- `m v^2 κ sinφ` — **banking** support: the horizontal centripetal reaction
  projected onto the tilted normal (raises load in a banked turn),
- `m v^2 κ_v` — **vertical curvature**: compression in dips (`κ_v>0`), unloading
  over crests (`κ_v<0`) — the Eau Rouge / Raidillon term,
- `F_df` — aerodynamic downforce (along the normal).

### 5.2 In-surface lateral demand

$$
F_\text{lat} = m\left(v^2\kappa\cos\phi \;-\; g\sin\phi\right)
$$

the horizontal centripetal projected into the surface plane, less the in-plane
gravity component that **banking** contributes toward the turn.

### 5.3 Tangential (longitudinal) balance — the grade force

$$
m\,a = F_x - F_\text{drag} - F_\text{roll} - \underbrace{m g \sin\theta}_{\text{grade}}
$$

Climbing (`θ>0`) costs drive force; descending adds it. The associated power term
is `m g (dz/ds) v = m g v \sin\theta`.

### 5.4 Grip circle on the 3D normal load

The friction budget is `μN` with `N` from §5.1 (so grip rises under compression,
falls when light), split between lateral and longitudinal:

$$
F_{x,\max} = \sqrt{\max(0,\;(\mu N)^2 - F_\text{lat}^2)}
$$

- **Cornering limit:** the largest `v` with `μN \ge |F_\text{lat}|`. Both sides
  scale with `v^2`, so it is solved by bisection. For `θ=φ=κ_v=0` this reduces to
  the flat closed form `μ(mg+F_\text{df}) = m v^2 κ`.
- **Forward:** `a = (\min(F_{x,\max},F_\text{drive}) - F_\text{drag} - F_\text{roll} - m g\sin\theta)/m`.
- **Backward (braking):** `a_\text{dec} = (\min(F_{x,\max},F_\text{brake}) + F_\text{drag} + F_\text{roll} + m g\sin\theta)/m`
  (climbing helps you slow; descending hurts).

### 5.5 Banked steady-state cornering (analytic check)

With `θ=κ_v=0`, `F_\text{df}=0`, the cornering limit `μN = F_\text{lat}` gives the
classic banked-turn maximum speed, which the synthetic banked-ring test matches:

$$
v_\max^2 = \frac{gR\,(\sin\phi + \mu\cos\phi)}{\cos\phi - \mu\sin\phi}, \qquad R = 1/\kappa
$$

### 5.6 Flat byte-invariance

For `θ=φ=κ_v=0`: `cosθ=cosφ=1.0` and `sinθ=sinφ=0.0` **exactly**, so `N = mg +
F_\text{df}`, `F_\text{lat} = m v^2 κ`, and the grade force is `0` — every 3D
expression collapses to the flat model. To guarantee **bitwise** golden stability,
`qss_lap_sim_3d` short-circuits a fully-flat ribbon (`Ribbon3d::is_flat`) straight
to the untouched `qss_lap_sim` on the 2D projection, so flat tracks execute the
identical float ops (and cost nothing extra). A dedicated test asserts
`qss_lap_sim_3d(flat) == qss_lap_sim` bit-for-bit on oval/circle/Silverstone.

### 5.7 Energy closure

`Σ_i m g \sin\theta_i\,ds_i = m g Σ_i dz_i = m g\,\Delta z_\text{lap} = 0` on a
closed track — the gravity term does zero net work per lap (asserted to machine
precision).

### 5.8 Spatial grip scaling (`mu_scale` grid)

The per-station scalar `mu_scale(s)` placeholder (§5.9, historically `1.0`
and unused) is superseded by a bilinearly-interpolated `(s, n)` grid,
`apex_track::MuScaleGrid` (schema v2's optional `mu_scale_grid` block,
`tracks/README.md`). The §5.4 grip circle becomes

$$
F_{x,\max} = \sqrt{\max\bigl(0,\;(\mu \cdot \text{mu\_scale}(s, n) \cdot N)^2 - F_\text{lat}^2\bigr)},
\qquad \mu N \to \mu \cdot \text{mu\_scale}(s, n) \cdot N \text{ in the §5.4 cornering limit too.}
$$

QSS's point-mass model has no lateral state, so `n` is whatever the *line*
sampling the grid is — the centerline (`n = 0`) for a plain centerline run,
or the driven path's own projected `(s, n)` for a driven-line run
(`apex_correlate::driven`). Critically, **`qss_lap_sim_3d` never reads a
ribbon's `mu_scale_grid` field itself** — a driven-line run passes QSS a
*synthesized*, reparameterized ribbon whose own `(s, n)` is not the
original centerline's, so a QSS-internal lookup on its own ribbon argument
would silently sample the wrong location. Instead `qss_lap_sim_3d_with_grip`
takes an externally-supplied per-station multiplier vector, baked by
whichever code constructs the line.

**Byte stability.** No grid attached ⇒ multiplier `1.0`, and
`params.tire_mu * 1.0 == params.tire_mu` bit-for-bit (IEEE-754 exact) — the
same "exact algebra" collapse as §5.6's `cosθ = 1.0`/`sinθ = 0.0`, not a
branch that skips the multiply. An explicit all-`1.0` grid is bitwise
equivalent too: bilinear interpolation of a constant field returns the
exact constant regardless of the interpolation fraction
(`c + t·(c − c) = c`).

### 5.9 Deferrals

- **Higher fidelities** (single-track / four-wheel / 14-DOF) get the same 3D
  terms in a **follow-up task** — the correlation pipeline runs on QSS, so QSS is
  first. Recorded in `PHYSICS_CHANGE.md`.
- **Banking** is plumbed and unit-tested but `0` in the current GLO-30-derived
  data (a 25–30 m DEM cannot resolve camber across a ~14 m track). The
  `banking_deg` field is the manual per-corner override for later.
- **`mu_scale(s)`** (the per-station scalar) is superseded by the `(s, n)`
  grid mechanism (§5.8); real grip-map *data* (rubbered line vs. dirty line)
  is still future work — §5.8 ships the mechanism, nothing populates a real
  grid yet.
- **Extended `(1 − n·Ω_z)`-style 3D road-coordinate kinematics are not derived
  here.** The classic curvilinear arc-length relation `ds/dt = v·cosα / (1 −
  n·κ)` — a car's true ground speed along the track differs from its
  along-centerline speed by a factor that depends on lateral offset `n` and
  curvature `κ` — already exists in this codebase (`apex_physics::point_mass`,
  `apex_optimizer::collocation`), but only in its flat 2D form: it consumes
  `Track`'s scalar `κ`, never `Ribbon3d`, and has no lateral state at all in
  the 3D point-mass QSS (§5 throughout treats `n = 0` implicitly — see §5.8's
  note on `qss_lap_sim_3d`'s line-supplied grip sampling for the same
  no-lateral-state fact in a different guise). A fully 3D version — replacing
  `κ` with `Ω_z` and accounting for the banked/pitched frame's effect on the
  `(1 − n·Ω_z)` factor and on `dn/dt` — is **not needed by the centerline
  point-mass QSS** (it has no `n` to integrate), but **is required by the
  dynamic optimal-control-problem formulation**, which does carry `n` as
  a state. This is deferred groundwork for that work, not an oversight: deriving
  it now, with no consumer to validate it against, would be unverified math
  sitting idle. Recorded here so the gap is a decision, not a silence.
