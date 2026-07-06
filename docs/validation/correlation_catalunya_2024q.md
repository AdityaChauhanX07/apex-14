# Telemetry correlation — Catalunya 2024 Q

**Preset +12.331 s / 12.07 m/s → fitted +2.351 s / 6.14 m/s** (direct driven line, μ-fixed identification).

Part of the [5-circuit campaign](correlation_summary.md). Pipeline: FastF1 export →
`import-track` (smoothed) → `telemetry-align` → `correlate` (direct driven line) →
`identify` (μ fixed, free = lift/drag/power_scale) → `correlate` (fitted).
Derived summary only — raw telemetry and the TUMFTM-derived track stay local.

> ⚠️ Fitted parameters are **point-mass-QSS effective values** (they absorb model
> limitations: no elevation, no transient load transfer, single fixed trim), not
> physical measurements. Sector splits (in `correlate` output) are equal-arc
> **thirds**, not official F1 sectors.

## Provenance & hygiene

- **Source:** FastF1, 2024 Spanish Grand Prix, Qualifying, fastest lap.
- **Track:** TUMFTM `Catalunya.csv` → smoothed import, length 4646 m.
- **Alignment:** scale 0.9973, RMS 6.76 m, s_offset 26.3 m.

| hygiene check | status | detail |
|---|---|---|
| event match | ✅ | resolved='Spanish Grand Prix' req='Spanish Grand Prix' |
| align scale | ✅ | 0.9973 |
| align RMS<8m | ✅ | 6.76 m |
| s_proj within 2% | ✅ | 4646 vs 4646 m (0.01%) |
| corner count sane | ✅ | 6 |

## Correlation

| | lap delta (s) | speed RMSE (m/s) |
|---|---|---|
| Preset (calibrated) | +12.331 | 12.07 |
| **Fitted** | **+2.351** | **6.14** |

Corners detected (prominent < 70 m/s minima): **6**. Preset max |Δv|
at s ≈ 2970 m.

## Fitted parameters (μ fixed at preset)

| parameter | preset | fitted | std error |
|---|---|---|---|
| `aero.lift_coeff` | 2.80 | **5.253** | ± 0.047 |
| `aero.drag_coeff` | 1.10 | **1.835** | ± 0.093 |
| `powertrain.power_scale` | 1.00 | **1.157** | ± 0.046 |
| `tires.mu` | 1.55 | *fixed* | — |

Overlay: `cars/catalunya_2024q_fitted.toml` (committed).

## Notes

Catalunya has the campaign's **worst fitted RMSE (6.14 m/s) and highest alignment RMS (6.76 m, still < 8 m)**. Its parameters are unremarkable (lift 5.253, drag 1.835 — high-downforce, as expected for Barcelona), so this is **not** an elevation outlier: the larger residual is alignment / driven-line reconstruction noise across its long medium-speed corners. Trusted less tightly than the others; flagged for a closer look.
