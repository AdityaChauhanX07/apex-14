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

**Spa is the elevation outlier** and the Phase-1 business case — see the [campaign summary](correlation_summary.md#spa--the-elevation-business-case) for the full story. In short: our flat 2-D QSS ignores gravity, so on the sport's biggest elevation change the fit is forced into a physically wrong **de-powered** regime (`power_scale` 0.833, uniquely low) and produces the **only negative fitted lap delta (−4.06 s)** — the sim ends up faster than the real car because it never pays the climb. Preset residuals peak at **s ≈ 1140 m (top of Raidillon / Kemmel-straight entry)**; after fitting, the worst error migrates to the **Pouhon → Stavelot** descent/climb (s ≈ 3890–4570 m). Every corner apex has the sim carrying too much speed.
