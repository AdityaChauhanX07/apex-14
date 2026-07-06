# Multi-circuit correlation campaign — 2024 Q, 5 circuits

Apex-14's QSS simulator, correlated against a real 2024 qualifying lap on five
circuits, then per-circuit **parameter identification** (μ fixed, free =
`{lift_coeff, drag_coeff, power_scale}`, direct driven line). This is the M1
"≥5 real circuits" deliverable. Pipeline per circuit:

```
fastf1_export.py → import-track (smoothed) → telemetry-align
→ correlate (preset) → identify → correlate (fitted)
```
driven by `tools/correlate_campaign.py --spec tools/campaign.toml`.

> **Derived summary only.** Raw FastF1 telemetry, the FastF1 cache, and
> TUMFTM-derived track JSON stay local (gitignored). The committed artifacts are
> the fitted-car overlays (`cars/<circuit>_2024q_fitted.toml`) and these docs.

> ⚠️ **The fitted parameters are point-mass-QSS *effective* values, not physical
> measurements.** They absorb the model's limitations (no elevation, no transient
> load transfer, no tyre thermal, a single fixed aero/trim). A parameter that
> varies across circuits beyond its std-error band is absorbing circuit-specific
> model error — that variation is a *diagnostic*, not a measurement.

> **Sector convention** (in the per-circuit pages): equal-arc **thirds**
> (`apex_physics::sector_times`), **NOT** official F1 sectors.

## The 5-circuit table

| circuit | preset Δ (s) | preset RMSE | **fitted Δ (s)** | **fitted RMSE** | lift | drag | power | align RMS | corners |
|---|---|---|---|---|---|---|---|---|---|
| Silverstone | +9.716 | 11.52 | **+0.921** | **4.12** | 5.374 ± 0.044 | 1.783 ± 0.055 | 1.110 ± 0.028 | 4.14 m | 5 |
| Monza | +5.362 | 8.36 | **+0.693** | **3.44** | 5.402 ± 0.037 | 1.353 ± 0.034 | 1.008 ± 0.020 | 2.47 m | 6 |
| Spa | +1.841 | 9.32 | **−4.064** | **4.28** | 4.915 ± 0.031 | 1.239 ± 0.032 | 0.833 ± 0.016 | 2.98 m | 7 |
| Spielberg | +6.994 | 10.07 | **+0.278** | **4.33** | 5.669 ± 0.041 | 1.515 ± 0.072 | 1.079 ± 0.033 | 3.58 m | 6 |
| Catalunya | +12.331 | 12.07 | **+2.351** | **6.14** | 5.253 ± 0.047 | 1.835 ± 0.093 | 1.157 ± 0.046 | 6.76 m | 6 |

Fitting collapses the lap delta from **+0.9…+12.3 s (preset)** to **−4.1…+2.4 s**,
and RMSE from ~8–12 m/s to ~3.4–6.1 m/s. Four of five land within ±2.4 s; **Spa
goes NEGATIVE (−4.06 s)** — the fit makes the sim *faster* than the real car (see
below). All five passed every hygiene check (event match, align scale ∈
[0.95, 1.05], align RMS < 8 m, s_proj within 2% of TUMFTM length, corner count
3–12); none required attention.

## Parameter scatter — the key analysis

| param | values (Sil, Mon, Spa, Spi, Cat) | mean | std | CoV | mean std-err | scatter ÷ std-err |
|---|---|---|---|---|---|---|
| `lift_coeff` | 5.374, 5.402, 4.915, 5.669, 5.253 | 5.323 | 0.274 | **5.1%** | 0.040 | **6.9×** |
| `drag_coeff` | 1.783, 1.353, 1.239, 1.515, 1.835 | 1.545 | 0.261 | **16.9%** | 0.057 | **4.6×** |
| `power_scale` | 1.110, 1.008, 0.833, 1.079, 1.157 | 1.037 | 0.126 | **12.2%** | 0.029 | **4.4×** |

**The cross-circuit scatter is 4.4–6.9× the within-fit std error for every
parameter.** So the spread is *real circuit-to-circuit variation*, not fit noise:
each "effective" parameter is absorbing model error that differs by circuit.

- **`lift_coeff` is the most invariant** (CoV 5.1%). Effective downforce-grip is
  the best-identified, most transferable parameter — the corners need roughly the
  same grip everywhere (μ is fixed, so lift carries the high-speed grip). A
  single value near **5.3** transfers reasonably.
- **`drag_coeff` scatters the most** (CoV 16.9%) and it tracks **real aero trim**:
  the low-downforce tracks fit **low drag** — Monza **1.35**, Spa **1.24** (skinny
  wings) — while the high-downforce tracks fit **high drag** — Catalunya **1.84**,
  Silverstone **1.78**. Our single "effective car" cannot be both Monza-trim and
  Barcelona-trim at once, so drag absorbs the difference. **This is expected and
  is the signal, not a bug.**
- **`power_scale` scatters mostly because of Spa** (0.833). Excluding Spa the
  range is 1.008–1.157 (tight); Spa's low value is an elevation artifact (below).

### Monza — the low-downforce prediction, checked

Predicted: Monza should fit *lower lift + drag*. **Result: drag yes (1.35, the
2nd-lowest), lift no (5.40, near the mean).** The skinny-wing signature lands in
**drag** (which is what a wing level most directly changes); **lift** stays high
because it is here an *effective grip* term the corners demand regardless of
trim, with μ held fixed. So the trim difference is real and visible — it just
shows up in drag, not lift. Honest partial-confirmation.

### Spa — the elevation business case

**Spa is the outlier, and it is the Phase 1 (elevation) business case.** Our QSS
is 2-D: the imported centerline is flat, so gravity's assist on descents and cost
on climbs is invisible. Spa has the sport's biggest elevation change
(Eau Rouge/Raidillon climb, the Pouhon → Stavelot descent/climb). The fingerprint:

- **Only negative fitted lap delta: −4.06 s** (sim faster than the real car). The
  fit minimizes *speed* RMSE, but with no gravity term it cannot also match
  *lap time* — the two objectives disagree exactly when elevation matters.
- **Uniquely de-powered fit: `power_scale` = 0.833** (next lowest 1.008, mean
  1.037 — Spa is 6.5 std-errors below the pack). The optimizer drops engine power
  to fight the sim's tendency to over-carry speed where the real car is fighting
  gravity — a physically wrong "fix" for a missing gradient force.
- **Every corner apex has the sim faster** (Δ = +2.7 … +12.0 m/s, all positive) —
  the model never under-carries, consistent with a missing energy sink (climbs).
- **Residuals concentrate on the elevation features.** The *preset* max |Δv|
  (14–15 m/s) sits at **s ≈ 1140 m — the top of Raidillon / start of the Kemmel
  straight**, i.e. the Eau Rouge/Raidillon climb exit. After the fit de-powers
  the car, the worst point migrates to **s ≈ 3890–4570 m — the Pouhon → Stavelot
  descent/climb** (a ±13–14 m/s swing: sim too slow at 3890, too fast at 4440).
  Both are exactly the gradient-dominated sections a flat model cannot represent.

Spa's *RMSE* (4.28) is mid-pack — the elevation error hides in the **parameters
and the lap-time sign**, not the raw speed RMSE. **This is the quantified case for
adding elevation (Phase 1):** a flat QSS forces Spa's fit into a physically wrong
low-power regime and cannot match its lap time, while the same car+method matches
the flat(ter) circuits to within ±2.4 s.

### Catalunya — noisiest, but not elevation

Catalunya has the **worst fitted RMSE (6.14) and the highest align RMS (6.76 m,
still < 8 m)**. It is not an elevation outlier (params are ordinary: lift 5.25,
drag 1.84, power 1.16). The larger residual is most likely the 2024 layout /
alignment: Catalunya has long medium-speed corners where the driven-line
reconstruction and the flat 2-D centerline accumulate more error, plus the
highest projection RMS. Flagged for a closer look, not trusted as tightly as the
others.

## Reproduce

```bash
# All raw inputs are local; this runs on real data (never in CI).
cargo build --release -p apex-14
python tools/correlate_campaign.py --spec tools/campaign.toml
```
`APEX_REPRO_TIMESTAMP` pins the fitted-TOML date for byte-reproducibility.
Per-circuit detail: `correlation_<circuit>_2024q.md`.
