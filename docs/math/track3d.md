# 3D curved-ribbon track geometry

**Status: STARTED (Phase 1.1 — geometry plumbing).** This document defines the
3D ribbon geometry and its moving frame. The *physics* that consumes the
generalized curvatures (grade force, banking, load transfer in 3D) lands in a
later Phase 1 slice; until then the fields are computed and plumbed but the
solvers read only the flat projections. This file is completed when that physics
lands.

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

## 5. To be completed when physics lands

- Grade force `−m g \sin\theta` along `t`, and the normal-load modulation by
  `\cos\theta`.
- Banking contribution to lateral grip (component of gravity into the surface).
- `mu_scale(s)` semantics (currently a plumbed placeholder defaulting to `1.0`).
- Whether the vehicle models consume `Ω` directly or the road angles `(ψ, θ, φ)`.
