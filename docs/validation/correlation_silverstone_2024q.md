# Telemetry correlation — Silverstone 2024 Q (RUS)

**Headline: Silverstone 2024 Q (RUS): lap delta +30.938 s, speed RMSE 20.13 m/s over 5890 m.**

This page records the **first end-to-end correlation** of the Apex-14 QSS
simulator against a real measured lap, produced by `apex-14 correlate`. It is a
**derived summary** (numbers only) — the raw FastF1 telemetry is not
redistributed and stays local (see `telemetry/README.md`).

> ⚠️ **This is the UNFITTED `--calibrated` preset.** No parameter identification
> has been applied. The point of this page is to establish a baseline and to
> locate where the model disagrees with reality — that read feeds parameter
> identification (task 2.4). **The car was not tuned to improve these numbers.**

> **Sector convention:** equal-arc-length **thirds** (`apex_physics::sector_times`),
> **NOT** official F1 sectors. Every number below uses that convention.

## Methodology

- **Measured source:** FastF1 3.8.3, 2024 British Grand Prix (round 12,
  Silverstone), Qualifying, driver RUS, lap 25 (1:25.819). Exported to the
  standard Apex telemetry CSV via `tools/fastf1_export.py` (km/h→m/s,
  throttle %→0–1, X/Y decimetres→metres).
- **Track:** TUMFTM racetrack-database `Silverstone.csv` imported to our track
  JSON (`apex-14 import-track`): 5886.8 m centerline, 1178 segments. (LGPL-3.0
  data — imported locally, not committed.)
- **Alignment** (`apex-correlate::align`): 2D similarity fit of the FastF1-local
  XY onto the track frame — rotation **0.283°**, scale **0.99885** (≈1, both
  frames are metres), translation (178.06, −117.57) m, no reflection, no
  direction reversal, start-line offset 9.68 m. **Post-fit RMS 4.15 m**, max
  point distance 10.93 m (< 25 m; no gross outliers).
- **Projection** (`apex-correlate::project`): each sample projected to the
  closest centerline point → geometric station `s` (monotone, wrap handled) and
  signed lateral offset `n` (**positive = left of centerline**). Projected `s`
  span 5895.9 m vs. FastF1 integrated distance 5830.9 m (+65 m, geometric vs.
  speed-integrated); `n ∈ [−10.93, +8.76] m`, RMS 4.15 m; 90% of samples within
  the local track half-width; 0 non-monotone samples.
- **Comparison engine:** QSS lap sim (`apex_physics::qss_lap_sim`) on the same
  centerline with the calibrated F1-2024 preset (mass 798 kg, C_l 2.80,
  μ 1.55). Sim and measured speed are resampled onto a common 10 m arc-length
  grid; corners are detected as prominent measured-speed minima below 70 m/s.

## Lap time

| | Time (s) |
|---|---|
| Measured (`t` span) | 85.819 |
| Measured (header comment) | 85.819 |
| Sim (QSS, calibrated) | 116.757 |
| **Delta (sim − measured)** | **+30.938** |

The header lap-time and the `t`-span agree exactly (cross-check passes).

## Sectors (equal-arc thirds — not official F1)

| Sector | Measured (s) | Sim (s) | Delta (s) |
|---|---|---|---|
| S1 | 29.759 | 40.202 | +10.443 |
| S2 | 27.401 | 38.624 | +11.223 |
| S3 | 28.659 | 37.931 | +9.272 |

The deficit is spread almost evenly across the lap (≈ +10 s per equal-arc
third), i.e. it is not localized to one sector — consistent with a
distributed cause rather than a single bad corner.

## Speed trace

| Metric | Value |
|---|---|
| RMSE | 20.13 m/s over 590 grid points (5890 m @ 10 m) |
| Max \|Δv\| | 58.99 m/s at s = 400 m |
| Sim carries most extra speed | +5.75 m/s at s = 4880 m |
| Sim most below measured | −58.99 m/s at s = 400 m |
| Measured speed range | 25.0 – 90.3 m/s (90 – 325 km/h) |
| Sim speed range | 14.7 – 96.1 m/s (53 – 346 km/h) |

## Corners & apex speeds

Detected **6** corners (prominent measured-speed minima below 70 m/s). Note this
is the count of **prominent braking/apex events on a qualifying lap**, not the
~18 *named* corners — Silverstone's fast corners (Copse, Maggotts–Becketts,
Stowe) are taken with almost no speed loss and do not form prominent minima.

| s (m) | Measured apex (m/s) | Sim @ s (m/s) | Δ (sim − meas) |
|---|---|---|---|
| 930 | 32.12 | 28.27 | −3.84 |
| 1060 | 25.08 | 21.54 | −3.53 |
| 2040 | 47.84 | 34.68 | −13.16 |
| 2200 | 32.39 | 31.10 | −1.29 |
| 4050 | 62.03 | 41.76 | −20.27 |
| 5540 | 30.07 | 25.13 | −4.94 |

The sim is **uniformly slow at every apex** (Δ < 0 everywhere). At the genuinely
slow corners (~25–32 m/s) the error is a few m/s; the large errors are at the
**medium/fast** minima (s = 2040: −13; s = 4050: −20).

## Braking-point offsets

Onset = longitudinal deceleration crossing 2 m/s² on corner approach. Offset =
`s_sim − s_measured` (**positive ⇒ sim brakes later**).

| Corner s (m) | Measured onset (m) | Sim onset (m) | Offset (m) |
|---|---|---|---|
| 930 | 780 | 790 | +10 |
| 1060 | 1000 | 990 | −10 |
| 2040 | 1880 | 1960 | +80 |
| 2200 | 2100 | 2160 | +60 |
| 4050 | 3830 | 3950 | +120 |
| 5540 | 5420 | 5400 | −20 |

At the slow corners the braking points nearly coincide (±10 m). The large
positive offsets (+60…+120 m) are at the medium/fast minima — but they are an
artifact of the sim already being far below the measured speed on the approach
(it never reaches a high enough speed to need to brake as early), so these
offsets should be read together with the apex-speed errors above.

## Where the model disagrees with reality — and why

**Root cause: curvature noise in the imported centerline, not (primarily) car
calibration.** Evidence:

- At s ≈ 400 m the measured car is at **84 m/s (302 km/h)** and its **own
  trajectory** curves gently (R ≈ 130–400 m), yet the imported **centerline**
  reads **R ≈ 34–62 m** at the same place. A 34 m radius forces the QSS to
  ~25 m/s — the −59 m/s max error. This is the single largest disagreement and
  it is a *geometry* artifact.
- The imported centerline contains physically impossible tight spikes
  (e.g. **R ≈ 12 m** near s ≈ 1044 m — no Silverstone corner is that tight).
  These are second-derivative (curvature) noise from the 5 m-spaced survey
  points; they barely move the centerline (alignment RMS is only 4 m) but they
  wreck the local radius the QSS depends on.
- The sim's minimum speed (14.7 m/s / 53 km/h) is far below the measured minimum
  (25.0 m/s / 90 km/h at the Village/Loop complex), i.e. the sim invents slow
  corners *between* the real ones.

**Read for parameter identification (2.4):** car-parameter identification cannot
be run meaningfully against this centerline — the phantom tight corners would
dominate the fit and the identified grip/aero would compensate for geometry
noise. **The centerline must be de-noised (curvature-aware smoothing / spline
refit) before, or jointly with, car-parameter identification.** Only the
genuinely slow corners (25–32 m/s apexes, where sim error is a few m/s) are
currently trustworthy signal; the medium/fast sections are dominated by the
geometry artifact.

## Reproduce

```bash
# (requires local FastF1 export + TUMFTM import — neither is committed)
apex-14 correlate \
  --telemetry telemetry/silverstone_2024_Q_aligned.csv \
  --track tracks/silverstone.json \
  --calibrated \
  --out-dir telemetry/correlation_out
```

Outputs `report.md`, `speed_overlay.svg`, and `inputs_panel.svg` (all local /
gitignored). `APEX_REPRO_TIMESTAMP` makes the report byte-reproducible.
