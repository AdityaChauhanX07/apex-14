# Telemetry correlation — Spielberg 2024 Q

**Preset +6.994 s / 10.07 m/s → fitted +0.278 s / 4.33 m/s** (direct driven line, μ-fixed identification).

Part of the [5-circuit campaign](correlation_summary.md). Pipeline: FastF1 export →
`import-track` (smoothed) → `telemetry-align` → `correlate` (direct driven line) →
`identify` (μ fixed, free = lift/drag/power_scale) → `correlate` (fitted).
Derived summary only — raw telemetry and the TUMFTM-derived track stay local.

> ⚠️ Fitted parameters are **point-mass-QSS effective values** (they absorb model
> limitations: no elevation, no transient load transfer, single fixed trim), not
> physical measurements. Sector splits (in `correlate` output) are equal-arc
> **thirds**, not official F1 sectors.

## Provenance & hygiene

- **Source:** FastF1, 2024 Austrian Grand Prix, Qualifying, fastest lap.
- **Track:** TUMFTM `Spielberg.csv` → smoothed import, length 4314 m.
- **Alignment:** scale 0.9974, RMS 3.58 m, s_offset 28.4 m.

| hygiene check | status | detail |
|---|---|---|
| event match | ✅ | resolved='Austrian Grand Prix' req='Austrian Grand Prix' |
| align scale | ✅ | 0.9974 |
| align RMS<8m | ✅ | 3.58 m |
| s_proj within 2% | ✅ | 4309 vs 4314 m (0.11%) |
| corner count sane | ✅ | 6 |

## Correlation

| | lap delta (s) | speed RMSE (m/s) |
|---|---|---|
| Preset (calibrated) | +6.994 | 10.07 |
| **Fitted** | **+0.278** | **4.33** |

Corners detected (prominent < 70 m/s minima): **6**. Preset max |Δv|
at s ≈ 3770 m.

## Fitted parameters (μ fixed at preset)

| parameter | preset | fitted | std error |
|---|---|---|---|
| `aero.lift_coeff` | 2.80 | **5.669** | ± 0.041 |
| `aero.drag_coeff` | 1.10 | **1.515** | ± 0.072 |
| `powertrain.power_scale` | 1.00 | **1.079** | ± 0.033 |
| `tires.mu` | 1.55 | *fixed* | — |

Overlay: `cars/spielberg_2024q_fitted.toml` (committed).

## Notes

Spielberg (Red Bull Ring) is the shortest lap (4.3 km) and gives the **best fitted lap delta of the campaign (+0.278 s)**. It has real elevation too, but far milder than Spa, and the fit absorbs it cleanly with ordinary parameters (lift 5.669 — highest, consistent with its high-load medium-speed corners; power 1.079). No elevation pathology here.
