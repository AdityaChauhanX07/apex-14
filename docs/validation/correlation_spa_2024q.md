# Telemetry correlation — Spa 2024 Q

**Preset +1.841 s / 9.32 m/s → fitted -4.064 s / 4.28 m/s** (direct driven line, μ-fixed identification).

Part of the [5-circuit campaign](correlation_summary.md). Pipeline: FastF1 export →
`import-track` (smoothed) → `telemetry-align` → `correlate` (direct driven line) →
`identify` (μ fixed, free = lift/drag/power_scale) → `correlate` (fitted).
Derived summary only — raw telemetry and the TUMFTM-derived track stay local.

> ⚠️ Fitted parameters are **point-mass-QSS effective values** (they absorb model
> limitations: no elevation, no transient load transfer, single fixed trim), not
> physical measurements. Sector splits (in `correlate` output) are equal-arc
> **thirds**, not official F1 sectors.

## Provenance & hygiene

- **Source:** FastF1, 2024 Belgian Grand Prix, Qualifying, fastest lap.
- **Track:** TUMFTM `Spa.csv` → smoothed import, length 6997 m.
- **Alignment:** scale 1.0019, RMS 2.98 m, s_offset 22.8 m.

| hygiene check | status | detail |
|---|---|---|
| event match | ✅ | resolved='Belgian Grand Prix' req='Belgian Grand Prix' |
| align scale | ✅ | 1.0019 |
| align RMS<8m | ✅ | 2.98 m |
| s_proj within 2% | ✅ | 6997 vs 6997 m (0.01%) |
| corner count sane | ✅ | 7 |

## Correlation

| | lap delta (s) | speed RMSE (m/s) |
|---|---|---|
| Preset (calibrated) | +1.841 | 9.32 |
| **Fitted** | **-4.064** | **4.28** |

Corners detected (prominent < 70 m/s minima): **7**. Preset max |Δv|
at s ≈ 1140 m.

## Fitted parameters (μ fixed at preset)

| parameter | preset | fitted | std error |
|---|---|---|---|
| `aero.lift_coeff` | 2.80 | **4.915** | ± 0.031 |
| `aero.drag_coeff` | 1.10 | **1.239** | ± 0.032 |
| `powertrain.power_scale` | 1.00 | **0.833** | ± 0.016 |
| `tires.mu` | 1.55 | *fixed* | — |

Overlay: `cars/spa_2024q_fitted.toml` (committed).

## Notes

**Spa is the elevation outlier** and the 3D-track-model business case — see the [campaign summary](correlation_summary.md#spa--the-elevation-business-case) for the full story. In short: our flat 2-D QSS ignores gravity, so on the sport's biggest elevation change the fit is forced into a physically wrong **de-powered** regime (`power_scale` 0.833, uniquely low) and produces the **only negative fitted lap delta (−4.06 s)** — the sim ends up faster than the real car because it never pays the climb. Preset residuals peak at **s ≈ 1140 m (top of Raidillon / Kemmel-straight entry)**; after fitting, the worst error migrates to the **Pouhon → Stavelot** descent/climb (s ≈ 3890–4570 m). Every corner apex has the sim carrying too much speed.

## Flat vs 3D elevation physics — an honest, mixed result

With the real 106 m Spa elevation profile (georeferenced + EU-DEM 25 m sampled)
now driving the **3D point-mass QSS** (grade force, vertical-curvature load,
banking), we re-ran the identical pipeline. The **acceptance criterion set in
advance — `power_scale` rejoins the ~1.0–1.16 pack and the negative lap delta is
eliminated — is NOT met.** No parameter was tuned to force it.

| quantity | flat 2-D | 3-D | verdict |
|---|---|---|---|
| preset lap delta | +1.841 s | **+1.700 s** | small improvement |
| preset speed RMSE | 9.32 m/s | **8.73 m/s** | improved |
| **fitted lap delta** | −4.064 s | **−2.856 s** | **improved ~30% (toward 0) but still negative** |
| fitted speed RMSE | 4.28 m/s | 4.31 m/s | ~unchanged |
| **fitted `power_scale`** | 0.833 | **0.802** | **did NOT rejoin the pack (slightly lower)** |
| fitted `lift_coeff` | 4.915 | 4.884 | ~unchanged |
| fitted `drag_coeff` | 1.239 | 1.161 | dropped |
| Eau Rouge grip_util p50 (s 700–1150) | 0.185 | 0.166 | **fell (compression adds *capacity*, not demand)** |

**What 3D got right.** The lap-time sign improved (−4.06 → −2.86 s) and preset
RMSE dropped — elevation *is* a real factor, and the Raidillon climb sits exactly
at the s ≈ 1140 m residual peak the flat model couldn't explain. Silverstone
(10.7 m range) moved < 1 % on every parameter, as a proper control should.

**Why `power_scale` did not recover (hypotheses, not tuning).**
1. **Grade is conservative on a closed lap.** `∮ m g sinθ ds = 0`, so the grade
   force redistributes *where* speed is won/lost but cannot shift a **lap-wide
   power multiplier** — `power_scale` is largely blind to it.
2. **3D introduces a descent over-carry.** The fitted max |Δv| migrates to the
   **Pouhon → Stavelot descent** (s ≈ 4680 m) and *grows* (14.3 → 17.8 m/s): the
   point-mass QSS free-wheels downhill faster than the real, energy-managed car,
   which keeps the fit de-powered.
3. **Eau Rouge is a vertical-load, not a lateral-grip, event.** Its low curvature
   means lateral demand is already small; compression adds normal-load *capacity*,
   so inferred grip_util *falls* rather than rising toward the limit.

**Conclusion.** Spa's de-powering is **not primarily an elevation artifact** — the
elevation-artifact hypothesis is falsified. Correct 3D elevation physics helps the aggregate
but the residual points elsewhere (the low-downforce Spa aero a single fixed car
can't match, tyre thermal, or transient descent/energy-management behaviour) — the
domain of the **deferred single-track / four-wheel / 14-DOF work** (PHYSICS_CHANGE
2026-07-07). The committed `cars/spa_2024q_fitted.toml` is left at the flat fit;
the 3D fit is not "better" and was not promoted.
