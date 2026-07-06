# Telemetry correlation — Monza 2024 Q

**Preset +5.362 s / 8.36 m/s → fitted +0.693 s / 3.44 m/s** (direct driven line, μ-fixed identification).

Part of the [5-circuit campaign](correlation_summary.md). Pipeline: FastF1 export →
`import-track` (smoothed) → `telemetry-align` → `correlate` (direct driven line) →
`identify` (μ fixed, free = lift/drag/power_scale) → `correlate` (fitted).
Derived summary only — raw telemetry and the TUMFTM-derived track stay local.

> ⚠️ Fitted parameters are **point-mass-QSS effective values** (they absorb model
> limitations: no elevation, no transient load transfer, single fixed trim), not
> physical measurements. Sector splits (in `correlate` output) are equal-arc
> **thirds**, not official F1 sectors.

## Provenance & hygiene

- **Source:** FastF1, 2024 Italian Grand Prix, Qualifying, fastest lap.
- **Track:** TUMFTM `Monza.csv` → smoothed import, length 5787 m.
- **Alignment:** scale 0.9995, RMS 2.47 m, s_offset 35.2 m.

| hygiene check | status | detail |
|---|---|---|
| event match | ✅ | export cached (event verified on first run) |
| align scale | ✅ | 0.9995 |
| align RMS<8m | ✅ | 2.47 m |
| s_proj within 2% | ✅ | 5770 vs 5787 m (0.29%) |
| corner count sane | ✅ | 6 |

## Correlation

| | lap delta (s) | speed RMSE (m/s) |
|---|---|---|
| Preset (calibrated) | +5.362 | 8.36 |
| **Fitted** | **+0.693** | **3.44** |

Corners detected (prominent < 70 m/s minima): **6**. Preset max |Δv|
at s ≈ 2530 m.

## Fitted parameters (μ fixed at preset)

| parameter | preset | fitted | std error |
|---|---|---|---|
| `aero.lift_coeff` | 2.80 | **5.402** | ± 0.037 |
| `aero.drag_coeff` | 1.10 | **1.353** | ± 0.034 |
| `powertrain.power_scale` | 1.00 | **1.008** | ± 0.020 |
| `tires.mu` | 1.55 | *fixed* | — |

Overlay: `cars/monza_2024q_fitted.toml` (committed).

## Notes

Monza — the 'Temple of Speed' — is the **low-downforce trim** case. The fit lands **low drag (1.353, the 2nd-lowest of the five)** and near-unity `power_scale` (1.008), with `lift_coeff` (5.402) near the pack mean. The skinny-wing package shows up in **drag**, not lift (see the summary page's trim discussion). Cleanest alignment of the campaign (RMS 2.47 m) and a tidy fit (+0.69 s / 3.44 m/s).
