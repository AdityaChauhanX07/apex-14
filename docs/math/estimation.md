# Dynamic-state estimation — RTS single-track smoother

Recovering the vehicle's **dynamic** state over a measured lap — slip angles, yaw
rate, lateral velocity, body slip, and the transient lag between driver input and
car response — that the quasi-steady-state (QSS) channel inference
(`docs/math/inference.md`) structurally cannot see. Implemented in
`apex-correlate::estimator`; run via `apex-14 estimate`.

> ⚠️ **Effective-parameter caveat.** The process model is the single-track
> (bicycle) model parameterized by the **fitted effective car**
> (`cars/<circuit>_2024q_fitted.toml`) and a representative Pacejka tire
> (`PacejkaTire::f1_default`). The slip angles and body slip therefore inherit
> those parameters' absorption of model limitations, and depend on the assumed
> tire curve. They are **model-consistent estimates, not measurements** — the
> same line is emitted into every output CSV's `# estimate_caveat:` header.

QSS inference inverts a *point-mass* model: it knows `a_lat = v²κ` but has no
rotational state, no slip, and no notion of the car rotating relative to its
velocity. This estimator adds that by running the single-track model as the
**process model** of an extended Kalman filter (EKF) over the lap in *time*, then
smoothing with the Rauch–Tung–Striebel (RTS) backward recursion. The measured
data is time-based (7.5 Hz), so the whole thing runs in the **time domain** — we
do not force the arc-length `s`-domain here.

## 1. State and process model

### State vector (8)

```
x = [ X, Y, ψ, v_x, v_y, r, δ, F_drive ]ᵀ
```

The first six are the single-track model's own state
(`apex_physics::bicycle`): world position `X, Y` (m), heading `ψ` (rad),
body-frame longitudinal / lateral velocity `v_x, v_y` (m/s), and yaw rate `r`
(rad/s). We keep the bicycle's native ordering rather than remap it — reusing the
exact `BicycleModel` dynamics (via its `OdeSystemGeneric` impl) avoids a whole
class of index-remapping bugs and is locked to the concrete model by a test
(`generic_matches_concrete`).

### The unknown-input problem, and why we augment

The controls the single-track model needs — steering angle `δ` and net drive
force `F_drive` — are **not measured**. Real telemetry carries no calibrated
steering channel. Two viable designs:

- **(a) Input augmentation (implemented).** Promote `δ` and `F_drive` to *states*
  that evolve slowly, and let the filter infer them from the observed motion.
  This is the standard input-estimation trick and needs no extra data.
- **(b) Geometric precompute (fallback, not implemented).** Derive `δ` from the
  path curvature (`δ ≈ atan(L·κ)` plus an understeer term) and `F_drive` from the
  longitudinal energy balance, then estimate only the vehicle states. This makes
  the estimator depend on the track geometry and a curvature-to-steer model; we
  note it as the fallback if augmentation proves too weakly observable on a given
  circuit.

Under (a) the two augmented states evolve as:

```
dδ/dt      = −λ_δ · δ + w_δ            (Ornstein–Uhlenbeck, mean-reverting)
dF_drive/dt = w_F                       (random walk)
```

`w_δ, w_F` are white process noises. **`F_drive` is a pure random walk** — it is
strongly observable through the speed channel, so it needs no anchor. **`δ` is
mean-reverting** toward zero with rate `λ_δ` (`delta_revert`, default 1 s⁻¹).
This matters: `δ` is only observable through the yaw dynamics *when the front tire
is loaded*; on a straight or at small slip it is unobservable, and a pure random
walk drifts unboundedly, eventually locking the front tire into Pacejka
saturation (zero gradient) where no measurement can recover it — producing
nonsensical front slip angles. Mean reversion says "steering relaxes to centre
absent evidence", which bounds the drift while a corner's yaw dynamics still pull
`δ` to its real value. With `λ_δ = 0` the scheme degenerates to a pure random
walk (used in the synthetic self-consistency test, where the truth is generated
the same way).

### Continuous dynamics

The vehicle-state derivatives are exactly `BicycleModel::derivatives`
(planar rigid body + Pacejka axle forces + aero drag/downforce/rolling; see
`docs/math/equations_of_motion.md`), evaluated with `δ = x₆`, `F_drive = x₇`.
The augmented model `AugBicycle` wraps this and appends the two input-state
derivatives above.

## 2. Measurements

Per epoch the measurement vector stacks whichever channels are finite:

| row | `h(x)` | source | noise σ |
|-----|--------|--------|---------|
| `X` | `x₀` | aligned GPS X (m) | `pos_sigma` (default **3 m**) |
| `Y` | `x₁` | aligned GPS Y (m) | `pos_sigma` |
| `speed` | `√(v_x²+v_y²)` | speed channel (m/s) | `speed_sigma` (default **0.5 m/s**) |
| `course` | `ψ + atan2(v_y, v_x)` | `atan2(ΔY, ΔX)` of consecutive GPS points | see below |

**Position σ — justified from the GPS analysis.** The telemetry align RMS at
Silverstone is **4.14 m** (`# align_rms_m` in the aligned CSV). That residual
*conflates* true per-sample GPS noise with the driven-line-vs-centerline offset
and small alignment error, so it is an **upper bound** on the GPS noise. We set
`pos_sigma = 3 m`, comfortably below the bound.

**Speed** is a derived, low-noise channel — `speed_sigma = 0.5 m/s`.

**Course is a pseudo-measurement, and it is not optional.** Heading `ψ` is only
*weakly* observable from position + speed (the velocity *direction* is
`ψ + β`, so position alone constrains that sum, not `ψ`), and without a heading
observation the EKF **diverges within a lap** (verified: turning the course term
off makes every downstream state worse). The course
`atan2(ΔY, ΔX)` from two consecutive GPS points observes `ψ + β` directly and
stabilizes the filter. Because it is derived from the *same* GPS points as the
position rows, using it naïvely double-counts that information; two safeguards
keep it honest:

- a **floor** `course_sigma` (default 0.20 rad) that deliberately inflates its
  noise, and
- **angular dilution of precision**: two points `d` apart with per-point noise
  `pos_sigma` give a course uncertainty `≈ √2·pos_sigma / d`, so the effective
  variance is `course_sigma² + (√2·pos_sigma/d)²`. At low speed the points are
  close together and the course is automatically distrusted — exactly where a
  fixed course noise would otherwise inject phantom lateral motion.

The course row is skipped entirely when the inter-sample displacement is below
`course_min_disp` (default 2 m).

The position and speed measurement Jacobians are analytic; the course row is
analytic too (`∂h/∂ψ = 1`, `∂h/∂v_x = −v_y/‖v‖²`, `∂h/∂v_y = v_x/‖v‖²`).

## 3. EKF forward pass

Between measurement epochs `k−1 → k` (interval `Δt = t_k − t_{k−1}`):

- **Propagation.** The mean is advanced with **RK4** (`rk4_step_generic`) at a
  fine substep (`substep_dt`, default 0.02 s). Because telemetry epochs can be
  long (up to ~0.45 s here) and one linearization over a whole interval is poor
  in the tight, low-speed corners where the tire is most nonlinear, a long
  interval is split into `⌈Δt / max_predict_dt⌉` (default `max_predict_dt = 0.1 s`)
  equal EKF **predict sub-steps**. Each sub-step forms its own transition
  Jacobian and adds its own process-noise increment; their product is the
  composite `F` stored for the smoother.
- **Jacobians via dual numbers.** The discrete transition Jacobian
  `F = ∂x_{k}/∂x_{k−1}` is obtained by **forward-mode automatic differentiation**
  (`apex_math::Dual`): the RK4 propagation is run once per input dimension with
  that dimension seeded as the dual variable, and the dual parts of the output
  are the columns of `F`. This differentiates the *entire* multi-substep RK4
  chain exactly — no finite differences.
- **Covariance predict.** `P⁻ = F P Fᵀ + Q_d`, with the discretized process
  noise `Q_d = Q · h` on the diagonal per sub-step.
- **Update (Joseph form).** With innovation `y = z − h(x⁻)`, innovation
  covariance `S = H P⁻ Hᵀ + R`, gain `K = P⁻ Hᵀ S⁻¹`:

  ```
  x⁺ = x⁻ + K y
  P⁺ = (I − K H) P⁻ (I − K H)ᵀ + K R Kᵀ        (Joseph form)
  ```

  The Joseph form is used for numerical symmetry / positive-definiteness; `P` is
  re-symmetrized after every predict and update.

### Innovation monitoring — NIS

The **normalized innovation squared** `NIS = yᵀ S⁻¹ y` is recorded per update.
For a consistent filter `NIS ∼ χ²(m)` with mean `m` (the measurement dimension,
3 or 4). We report the mean, the 5/50/95th percentiles, and the fraction under
the per-dof 95% χ² bound — so consistency is *measurable*, not asserted.

## 4. RTS backward pass

The Rauch–Tung–Striebel recursion runs backward from the last epoch, combining
the forward (filtered) estimate with all *future* measurements:

```
C_k        = P^f_k  Fᵀ_{k+1}  (P⁻_{k+1})⁻¹                    (smoother gain)
x^s_k      = x^f_k + C_k (x^s_{k+1} − x⁻_{k+1})
P^s_k      = P^f_k + C_k (P^s_{k+1} − P⁻_{k+1}) Cᵀ_k
```

`C_k` is obtained by solving `P⁻_{k+1} Cᵀ_k = (P^f_k Fᵀ_{k+1})ᵀ` (Gaussian
elimination, `apex_math::lm::solve_linear`) rather than forming an explicit
inverse. The smoothed marginal standard deviations `√diag(P^s_k)` are the
reported per-state uncertainties.

## 5. Process-noise tuning `Q`

`Q` is a config diagonal with physically-motivated defaults (units and rationale
below). Discretized per sub-step as `Q_d = Q·h`.

| state | `q` default | unit | rationale |
|-------|------------|------|-----------|
| `X`, `Y` | 0.05 | m²/s | kinematics integrate velocity nearly exactly; small slack |
| `ψ` | 5·10⁻⁴ | rad²/s | heading driven by `r`; small direct slack |
| `v_x`, `v_y` | 2.0 | (m/s)²/s | **model error**: unmodeled load transfer, combined slip |
| `r` | 0.5 | (rad/s)²/s | yaw model error |
| `δ` | 2·10⁻⁴ | rad²/s | tight: `δ` is weakly observable; loose values drift into tire saturation |
| `F_drive` | 5·10⁶ | N²/s | observable via speed; moderate so it can track braking/traction |

The single-track model does not perfectly match the real car, so the dominant
`Q` terms are the **vehicle-velocity** ones (`v_x, v_y, r`): they represent honest
model error and keep the filter appropriately uncertain (an over-tight `Q`
collapses `P`, the filter becomes overconfident, and every update trips the
innovation gate — the filter then runs open-loop and the bicycle model spins in
slow corners). `δ` is the opposite: tight, because it is weakly observable and a
loose density lets it wander into garbage.

## 6. Robustness

- **Measurement gaps (NaN).** If a channel is non-finite it contributes no row;
  if no rows remain the epoch is a **prediction-only** step (no update) and its
  covariance grows. Counted as `n_gaps`.
- **Soft innovation gate (divergence guard).** A hard reject of a high-innovation
  update is the *wrong* primitive here: a transient over-estimate makes the
  prediction disagree with the measurement, so a hard gate rejects the very
  correction needed to recover — and the divergence locks in (observed: ~88% of
  updates rejected, filter open-loop, slow corners spinning to >50° slip). Instead
  the gate is **soft**: an update with `NIS > nis_gate` (default 100) has its
  measurement noise inflated by `NIS / nis_gate`, capping the effective NIS at the
  gate. Gross outliers (a GPS jump) are heavily down-weighted without ever
  starving the filter; the filter stays closed-loop everywhere. Such updates are
  counted as `n_rejected` (soft-gated) and reported.

## 7. Outputs

Registry channels appended to the measured telemetry (all append-only, tested):
`slip_angle_front`, `slip_angle_rear`, `body_slip_angle` (β), `yaw_rate`,
`lateral_v`. Slip angles are computed from the smoothed state exactly as the
model defines them:

```
α_f = δ − atan((v_y + r·l_f)/v_x)      α_r = −atan((v_y − r·l_r)/v_x)      β = atan2(v_y, v_x)
```

Per-state standard deviations and the NIS / robustness statistics are written to
a **sidecar JSON** (`<out>.diag.json`) keyed by channel — cleaner than doubling
the channel count with `*_std` columns.

## 8. Validation

### Synthetic calibration (the correctness anchor)

`estimator::tests::synthetic_calibration` generates a ground-truth trajectory by
integrating the single-track model with inputs that are themselves a random walk
**identical to the filter's process model** (`δ` pure random walk, `λ_δ = 0`),
corrupts position (σ = 1 m, below the real GPS bound) and speed (σ = 0.3 m/s) at
7.5 Hz, and asserts:

- **3σ coverage ≥ 95 %** of the truth by the reported covariance over the six
  vehicle states (measured ≈ 0.96–1.00 per state; RMS error ≈ reported σ);
- **NIS consistency** — forward-pass mean NIS in `[1.5, 6]` (target `m`), with
  > 80 % of updates under the χ² 95 % bound;
- **slip-angle recovery** — median error < 0.02 rad (rear), < 0.03 rad (front;
  looser because it carries the weakly-observable steering).

Supporting tests cover near-noise-free recovery, gap handling (covariance grows
through a long gap), the soft gate on a 100 m outlier, and the linear algebra.

### Real data — Silverstone 2024 Q (RUS, lap 25), fitted car

`apex-14 estimate` on `silverstone_2024_Q_aligned.csv` with
`cars/silverstone_2024q_fitted.toml`:

| quantity | p50 | p90 | max |
|----------|-----|-----|-----|
| front slip angle | 1.33° | 3.73° | **5.04°** |
| rear slip angle | 1.03° | 3.33° | **4.47°** |
| body slip β | 0.44° | 1.78° | 2.53° |

Peak **yaw rate 1.15 rad/s**; these slip magnitudes sit squarely in the expected
F1 limit range (~3–7°). NIS mean ≈ 6.0 vs dof 4 (mildly optimistic — honest
model mismatch, *not* inflated); 647/647 epochs updated, **0 gaps, 1 soft-gated**
update. The smoother yaw rate vs the QSS-kinematic `v·κ` disagree by a mean
**0.26 rad/s in corners** — that disagreement *is* the transient/dynamic content
(yaw build-up, body slip, the lag between the geometric path and the car's actual
rotation) that the point-mass QSS inference structurally cannot represent.

## 9. Scope and limitations (honest notes)

- **Why RTS, not MHE.** The original design named moving-horizon estimation (MHE) as the
  alternative. This is an *offline, fixed-interval* problem — the whole lap is
  available at once — so the linear-Gaussian smoother (EKF forward + exact RTS
  backward recursion) is the natural fit: it uses every sample for every estimate
  at a fraction of MHE's per-step optimization cost. MHE's advantages — hard
  inequality constraints and a receding horizon — matter for *online/real-time*
  estimation, which is future work; they buy nothing here.

- **Single-track model.** The estimate has no lateral/longitudinal *load transfer
  in the state* (the four tires are collapsed to two axle tires); load transfer
  enters only through the axle-load model, not as an estimated quantity.
- **Linear-range tire validity.** Slip-angle estimates are most trustworthy at
  small-to-moderate slip. In the slowest, highest-slip corners the Pacejka curve
  is near its peak and the front tire can approach saturation — the regime where
  the single-track assumption is weakest.
- **Fitted-parameter dependence.** Slip / body-slip / force quantities depend on
  the fitted effective aero + the assumed tire; they are model-consistent
  estimates, not measurements (§ caveat).
- **Steering observability.** `δ` (hence front slip) is the least reliable output
  — it is only observable through the loaded-front-tire yaw dynamics and is held
  by mean reversion elsewhere.
- **Four-wheel model is future work.** A higher-fidelity (7- or 14-DOF) process
  model with per-corner loads and estimated load transfer would live behind the
  same estimator interface *if* a `Model` trait emerges naturally from a second
  implementation — we deliberately do **not** force that abstraction from a single
  model today.
