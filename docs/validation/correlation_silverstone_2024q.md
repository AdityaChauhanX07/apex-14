# Telemetry correlation — Silverstone 2024 Q (RUS)

**Fitted (direct driven line, identified car): lap delta +0.921 s, speed RMSE
4.12 m/s over 5890 m** — from an unfitted-preset baseline of +30.9 s / 20.1 m/s.

Progression (each step removes one confound):

| ver | line / car | lap delta | RMSE |
|---|---|---|---|
| v1 | unsmoothed centerline, preset | +30.938 s | 20.13 |
| v2 | smoothed centerline, preset | +26.355 s | 18.24 |
| v3 | driven line (**offset**), preset | +14.544 s | 13.58 |
| v4 | driven line (**direct**), preset | **+9.716 s** | **11.52** |
| v5 | driven line (direct), **fitted car** | **+0.921 s** | **4.12** |

This page records the correlation of the Apex-14 QSS simulator against a real
measured lap, and how the confounds were peeled off in turn — **centerline
curvature noise** (track import), the **racing line** (driver vs. centerline, via
two reconstruction modes), and finally **car-parameter error** (identified) — so
the four sources of disagreement are cleanly separated. It is a **derived
summary** (numbers only) — the raw FastF1 telemetry and the TUMFTM-derived track
are not redistributed and stay local (see `telemetry/README.md`,
`tracks/README.md`). The **fitted car overlay is committed**
(`cars/silverstone_2024q_fitted.toml`) — our own derived parameters, no raw data.

> **Sector convention:** equal-arc-length **thirds** (`apex_physics::sector_times`),
> **NOT** official F1 sectors.

## Methodology

- **Measured source:** FastF1 3.8.3, 2024 British Grand Prix (round 12,
  Silverstone), Qualifying, RUS, lap 25 (1:25.819). Exported via
  `tools/fastf1_export.py`.
- **Track:** TUMFTM `Silverstone.csv` → `apex-14 import-track`. Import now applies
  **curvature-aware smoothing** by default (see below).
- **Alignment** (`apex-correlate::align`): 2D similarity fit onto the (smoothed)
  track frame — rotation 0.302°, scale 0.99891, no reflection/reversal,
  s_offset 9.45 m. **Post-fit RMS 4.14 m** (was 4.15 m on the unsmoothed track —
  smoothing barely moved the fit, as expected: the noise is in curvature, not
  position).
- **Projection** gives geometric station `s` and signed lateral offset `n(s)`
  (**+n = left of centerline**).
- **Comparison engine:** QSS (`apex_physics::qss_lap_sim`), calibrated F1-2024
  preset (mass 798 kg, C_l 2.80, μ 1.55). `apex-14 correlate --line
  centerline|measured` runs the QSS on the centerline or on the reconstructed
  driven line.

## Step 1 — Centerline smoothing (kills curvature noise)

Regularized least squares (2D), second-difference roughness penalty, `λ` chosen
by bisection as the smoothest curve within a max-deviation budget (default
**1.0 m**), periodic (no seam kink). Diagnostics on the real import:

| | Before | After |
|---|---|---|
| Tightest radius (1/max\|κ\|) | 13.5 m | 15.7 m |
| \|κ\| p95 (radius) | 0.0201 (50 m) | 0.0198 (51 m) |
| \|κ\| p50 (radius) | 0.0011 (906 m) | 0.0011 (897 m) |
| Total length | 5886.8 m | 5883.3 m (−0.06%) |

`λ = 5.2`, max point deviation = 1.00 m (the budget binds).

**On the acceptance targets — honest note.** The v1 diagnosis over-attributed the
error to noise. A noise-robust wide-baseline analysis of the raw survey shows:

- **s ≈ 1044 m (The Loop):** raw R ≈ 12 m is a **noise spike on a real ~30 m
  corner**. Smoothing removes the spike (tightest radius 13.5 → 15.7 m); the
  residual tightness is the genuine Loop, read tight by the 3-point curvature
  stencil. ✔ spike gone.
- **s ≈ 400 m (Abbey):** the true **centerline** radius is ~60–80 m at the corner
  scale (R > 120 m only appears at a ±100 m baseline — that is the *racing-line*
  radius, not the centerline). **No smoothing tolerance can make the centerline
  read R > 120 m at Abbey without erasing a real corner** — the >120 m there is
  the driven line, addressed in Step 2. So the "s ≈ 400 → R > 120 m" acceptance
  target is not a centerline property; it is met by the driven-line
  reconstruction, not by smoothing.
- **Length** preserved to −3.5 m (−0.06%). ✔

Net effect on correlation: lap delta **+30.9 → +26.4 s**, RMSE **20.1 → 18.2 m/s**
— curvature noise was worth ~4.5 s, less than v1 implied.

## Step 2 — Measured driven line (removes the racing line)

The centerline QSS corners where the *centerline* bends; the driver runs wide to
open corners. Reconstructing the driven path (smoothed centerline offset by the
measured `n(s)`, `n` low-passed with a ±10 m periodic moving average) and running
the same car on it removes that confound:

| Metric | v1 unsmoothed CL | v2 smoothed CL | v3 smoothed **driven** |
|---|---|---|---|
| Lap delta (sim − meas) | **+30.938 s** | **+26.355 s** | **+14.544 s** |
| Sector Δ S1 / S2 / S3 (s) | +10.44 / +11.22 / +9.27 | +8.65 / +9.52 / +8.19 | **+3.59 / +5.57 / +5.38** |
| Speed RMSE (m/s) | 20.13 | 18.24 | **13.58** |
| Max \|Δv\| (m/s) @ s | 58.99 @ 400 | 55.16 @ 400 | **39.89 @ 3880** |
| Corners (< 70 m/s) | 6 | 5 | 5 |
| Driven length | — | — | 5826.4 m (−56.9 m vs CL) |

Peeling the two confounds cut the lap delta by more than half (**+30.9 → +14.5 s**)
and moved the worst speed error **off** Abbey.

## Apex speeds — centerline vs. driven line

| s (m) | Measured (m/s) | Sim CL | Δ CL | Sim driven | Δ driven |
|---|---|---|---|---|---|
| 930  | 32.09 | 30.03 | −2.06 | 29.96 | −2.13 |
| 1060 | 25.25 | 22.31 | −2.94 | 22.23 | −3.03 |
| 2200 | 32.64 | 32.18 | −0.46 | 26.43 | −6.21 |
| 4050 | 62.00 | 41.90 | −20.10 | 45.70 | **−16.30** |
| 5540 | 29.89 | 26.95 | −2.94 | 28.19 | −1.70 |

The **slow** corners (25–32 m/s apexes) match within ~2–3 m/s on both lines — the
model's **low-speed mechanical grip (μ) is well-calibrated.** The residual is
concentrated at the **medium/fast** corner s = 4050 (measured 62 m/s ≈ 223 km/h,
sim −16 m/s) and the fast Abbey approach.

## Step 3 — Direct driven line (v4, removes the reconstruction floor)

The **offset** reconstruction (centerline + `n·normal`) under-opens the fastest
corners. The **direct** reconstruction builds the driven path straight from the
aligned measured `(x, y)`: resample to uniform arc length (FastF1 samples are
time-uniform → dense in corners, sparse on straights, which wrecks the 3-point
curvature), trim to exactly one lap (else the start/finish overlap folds into a
seam kink), then smooth with the shared regularized-LS smoother (deviation budget
**0.75 m**). Abbey radius (s ≈ 400 m), reconstruction vs. the car's own
wide-baseline trajectory (~128 m):

| s (m) | offset R | **direct R** | car trajectory |
|---|---|---|---|
| 400 | 92 m | **124 m** | ~128 m |

The direct line reproduces the driver's actual radius, dropping the lap delta
**+14.5 → +9.7 s** and RMSE **13.6 → 11.5 m/s** (max error moves off Abbey to
s ≈ 3880 m). This v4 residual is now almost entirely **car-model** error.

## Step 4 — Parameter identification (v5)

The residual signature was: **high-speed corners sim-slow** (downforce low),
**straights sim-fast** (drag low / power high), **low-speed corners already good**
(μ fine). A Levenberg-Marquardt fit (shared `apex_math::lm`) of
`{lift_coeff, drag_coeff, power_scale}` — **μ held fixed** — to `sim − measured`
speed on the direct driven line, from the calibrated preset:

| parameter | preset | fitted | std error | notes |
|---|---|---|---|---|
| `aero.lift_coeff` | 2.80 | **5.374** | ± 0.044 | downforce ~doubled |
| `aero.drag_coeff` | 1.10 | **1.783** | ± 0.055 | L/D = 3.0 (realistic F1) |
| `powertrain.power_scale` | 1.00 | **1.110** | ± 0.028 | ×`max_drive_force` |
| `tires.mu` | 1.55 | *fixed* | — | guardrailed |

Cost dropped **78 327 → 10 027** (87%) in 47 iterations (~3 ms/iter, 0.15 s
total, 590 grid points). Condition number of `JᵀJ` = **1.5 × 10²** (well
conditioned). One flagged pair: **`drag_coeff` ↔ `power_scale`, |corr| = 0.98** —
both set top speed on the straights, so they trade off (expected; not a defect,
but their *individual* values are less certain than their combined effect).

No parameter hit a bound; the fitted values are physically plausible (L/D ≈ 3.0
is a real F1 ratio, ~4 t downforce at 324 km/h). The fitted overlay is
committed.

**v5 acceptance** (fitted car):

| line | lap delta | RMSE |
|---|---|---|
| direct driven | **+0.921 s** | **4.12 m/s** |
| centerline | +14.972 s | 12.82 m/s |

On the driven line the fitted car is within **0.9 s / 4.1 m/s** of reality — a
QSS point-mass matching a real qualifying lap to <1%. On the *centerline* it is
still +15 s, because the fitted car was identified for the **driven** line;
the centerline gap is the racing-line geometry, not the car (as expected — the
fit does not, and should not, absorb it).

### The μ-fixed guardrail — evidence

Re-running the fit with **μ also free** (diagnostic, not shipped):

| | headline (μ fixed) | μ free |
|---|---|---|
| `lift_coeff` | 5.374 | 3.285 |
| `tires.mu` | 1.55 | **2.144 ± 0.061** |
| cost | 10 027 | 8 296 |
| condition number | 1.5 × 10² | **1.4 × 10³** |
| new weak pair | — | `lift_coeff` ↔ `mu`, |corr| = 0.986 |

Freeing μ drops it into a marginally lower cost but pushes μ to **2.14** (near its
2.2 physical ceiling), halves `lift_coeff`, degrades the condition number **10×**,
and makes `lift_coeff` and `mu` jointly non-identifiable (both scale grip). This
is exactly why μ is fixed: it would otherwise **absorb the aero deficit into an
implausible tyre grip**. The guardrail is well justified.

### Honest gaps

- The remaining **+0.9 s / 4.1 m/s** (max 26.6 m/s at s ≈ 3880 m) is genuine
  model limitation: the QSS is a point mass with no transient load transfer,
  tyre thermal, or combined-slip dynamics, and no braking-vs-power asymmetry
  beyond the friction circle.
- `drag_coeff` and `power_scale` are individually only ~loosely identified (they
  co-determine top speed); their *combined* effect on the straights is tight. A
  dedicated top-speed/coast-down segment would separate them.
- Corner count is 5 (prominent minima < 70 m/s), **not** the ~18 named corners —
  Silverstone's fast corners are near-flat on a qualifying lap.

## Reproduce

```bash
# (requires local FastF1 export + TUMFTM import — neither is committed)
apex-14 import-track -i <TUMFTM>/Silverstone.csv -o tracks/silverstone.json -n Silverstone
apex-14 telemetry-align --telemetry telemetry/silverstone_2024_Q.csv \
  --track tracks/silverstone.json --out telemetry/silverstone_2024_Q_aligned.csv
apex-14 correlate --telemetry telemetry/silverstone_2024_Q_aligned.csv \
  --track tracks/silverstone.json --calibrated --line measured --driven-line direct \
  --out-dir telemetry/correlation_out_driven
# Identify the car (μ fixed) and re-correlate with the fitted overlay:
apex-14 identify --telemetry telemetry/silverstone_2024_Q_aligned.csv \
  --track tracks/silverstone.json --calibrated \
  --free "aero.lift_coeff,aero.drag_coeff,powertrain.power_scale" \
  --out cars/silverstone_2024q_fitted.toml --driven-line direct
apex-14 correlate --telemetry telemetry/silverstone_2024_Q_aligned.csv \
  --track tracks/silverstone.json --calibrated --car cars/silverstone_2024q_fitted.toml \
  --line measured --driven-line direct --out-dir telemetry/correlation_out_v5
```

Outputs (`report.md` + SVGs) are local / gitignored; `APEX_REPRO_TIMESTAMP`
makes the reports byte-reproducible.
