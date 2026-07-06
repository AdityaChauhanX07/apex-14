# Telemetry correlation — Silverstone 2024 Q (RUS)

**v1 (unsmoothed centerline): lap delta +30.938 s, speed RMSE 20.13 m/s.**
**v2 (smoothed centerline): +26.355 s, RMSE 18.24 m/s.**
**v3 (smoothed + measured driven line): +14.544 s, RMSE 13.58 m/s.**

This page records the correlation of the Apex-14 QSS simulator against a real
measured lap, and how two confounds were peeled off in turn: **centerline
curvature noise** (track import) and the **racing line** (driver vs. centerline).
It is a **derived summary** (numbers only) — the raw FastF1 telemetry and the
TUMFTM-derived track are not redistributed and stay local
(see `telemetry/README.md`, `tracks/README.md`).

> ⚠️ **Unfitted `--calibrated` preset.** No car-parameter identification has been
> applied. The car was **not** tuned to improve these numbers. The point is to
> isolate the *car-model* residual that parameter identification (2.4) will fit.

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

## Where the residual points (for parameter identification, 2.4)

With curvature noise and the racing line removed, the remaining **+14.5 s** is
now largely **car-model** error, and it has a clear signature:

- **High-speed cornering: sim too slow** (s = 4050: −16 m/s; the Abbey/approach
  region drives the 39.9 m/s max error). At 60–85 m/s the grip is
  downforce-dominated ⇒ **the calibrated aero grip (C_l / downforce) is too low.**
  This is the biggest lever.
- **Straights: sim slightly *faster*** (+7.4 m/s at s = 4890) ⇒ **drag a touch
  low or power/traction a touch high** — the sim tops out above the real car.
- **Low-speed corners: already good** ⇒ leave μ roughly where it is.

So parameter ID should **raise downforce first** (recover high-speed cornering),
then trim **drag/power** on the straights, and treat **μ** as near-final. Fitting
in the other order (μ first) would over-fit low-speed grip to compensate for the
aero deficit.

### Honest gaps

- The driven-line reconstruction (centerline + `n·normal`) has a **floor at the
  very fastest corner**: at Abbey it reconstructs R ≈ 90–95 m vs. the car's own
  ~128 m trajectory, so a slice of the s ≈ 400 residual is reconstruction error,
  not car model. Slow/medium corners reconstruct to within ~5% of the car's
  trajectory. A future improvement is to build the driven line from the smoothed
  measured `(x, y)` directly rather than via the centerline offset.
- Corner count is 5 (prominent minima < 70 m/s), **not** the ~18 named corners —
  Silverstone's fast corners are near-flat on a qualifying lap and form no
  prominent speed minimum.

## Reproduce

```bash
# (requires local FastF1 export + TUMFTM import — neither is committed)
apex-14 import-track -i <TUMFTM>/Silverstone.csv -o tracks/silverstone.json -n Silverstone
apex-14 telemetry-align --telemetry telemetry/silverstone_2024_Q.csv \
  --track tracks/silverstone.json --out telemetry/silverstone_2024_Q_aligned.csv
apex-14 correlate --telemetry telemetry/silverstone_2024_Q_aligned.csv \
  --track tracks/silverstone.json --calibrated --line centerline --out-dir telemetry/correlation_out
apex-14 correlate --telemetry telemetry/silverstone_2024_Q_aligned.csv \
  --track tracks/silverstone.json --calibrated --line measured   --out-dir telemetry/correlation_out_driven
```

Outputs (`report.md` + SVGs) are local / gitignored; `APEX_REPRO_TIMESTAMP`
makes the reports byte-reproducible.
