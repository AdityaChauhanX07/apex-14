# Track3D validation — close-out

This is the honest 3D-track-model story end to end: the geometry, the physics, and
the headline result (Spa), with the numbers that back each claim and
pointers to where they're proven. Companion docs: `docs/math/track3d.md`
(derivations), `tracks/README.md` (schema + elevation workflow),
`docs/validation/correlation_spa_2024q.md` / `correlation_summary.md`
(the correlation campaign this validates against).

## 1. Geometry

**Ribbon3d design.** `apex_track::Ribbon3d` generalizes the flat 2D `Track`
to a moving-frame curve `r(s) = (x, y, z)` with orthonormal frame `{t, n, m}`
and the generalized (Darboux) curvatures `Ω = (Ω_x, Ω_y, Ω_z)` — roll-rate,
pitch-rate, and the familiar 2D yaw-curvature (`docs/math/track3d.md` §1–2).

**Flat-exactness discipline.** A flat ribbon (`Ribbon3d::from_flat`) copies
every 2D segment scalar verbatim, so its scalar queries reproduce the legacy
2D `Track` queries **bit-for-bit** — proven densely (500 samples/lap plus
every exact node station) by `ribbon3d::tests::assert_flat_exact` on a
circle, an oval, and Silverstone (`flat_exact_circle`, `flat_exact_oval`,
`flat_exact_silverstone`). This is what let the entire 3D geometry and
physics land without moving a single golden fixture.

**Analytic frame validation.** The moving-frame construction is checked
against two closed forms with an independent geometric derivation:
- A helix `r(u) = (R cos u, R sin u, hu)`: closed-form curvatures
  `Ω_x = h/L², Ω_y = 0, Ω_z = R/L²` (`L² = R²+h²`) match to **< 5e-6**
  (`helix_frame_is_orthonormal_and_omega_matches_closed_form`).
- A constant-bank flat circle: closed form
  `Ω = (0, sinβ/R, cosβ/R)` matches to **< 5e-6**
  (`banked_circle_frame_and_omega`), plus frame orthonormality to **< 1e-12**
  and C¹ continuity across the closed-loop seam
  (`frame_is_c1_continuous_across_seam`).

**Georeferencing method + accuracy.** TUMFTM centerlines carry no
georeference. `tools/georef.py` fits a 2D similarity transform from the
local centerline onto the OSM `highway=raceway` way (point-to-segment
trimmed ICP). Reported coverage: **Spa ≈ 4.9 m coverage-RMS (98% within
30 m)**, **Silverstone ≈ 4.4 m (92%)** — both well under the 15 m
sub-DEM-cell target (`tracks/README.md` § 3D elevation workflow).

**Elevation source.** The task named Copernicus GLO-30; OpenTopoData's free
tier doesn't serve it, so the pipeline substitutes **EU-DEM 25 m**
(`eudem25m`, also a Copernicus Land Monitoring Service product, finer
posting, Europe-only), falling back to `srtm30m` then Open-Elevation.

**Profile validation.** `tools/fetch_elevation.py` annotates named Spa
landmarks at their georeferenced stations — **Raidillon top (s≈1140 m)**,
Les Combes (s≈2400 m), Pouhon (s≈4400 m), Blanchimont (s≈6000 m) — and
gates the smoothed profile against a public grade sanity anchor
(`GRADE_ANCHOR = {"spa": 0.18, "silverstone": 0.05}`, warns if the smoothed
max grade exceeds 2× the anchor). The imported Spa profile's smoothed grade
peaks around **19.6%**, against the public Raidillon figure of **~18%** —
same order, same location, a reasonable margin for a 25 m DEM crossing a
famously steep, short climb.

## 2. Physics

The 3D point-mass QSS terms (`docs/math/track3d.md` §5) add, on top of the
flat model: the grade force (`−mg sinθ`, §5.3), the 3D normal load
(`N = m(g cosθ cosφ + v²κ sinφ + v²κ_v) + F_df`, §5.1 — banking support and
vertical-curvature compression/unloading), and the grip circle on that load
(§5.4). Synthetic validations, all in `crates/apex-physics/src/qss.rs` tests:

| check | tolerance | test |
|---|---|---|
| Banked steady-state cornering vs. closed form | < 1e-4 | `banked_ring_cornering_matches_analytic` |
| Vertical-curvature load `ΔN = m v² κ_v` | < 1e-9 | `vertical_curvature_load_matches_analytic` |
| Grade-onset ordering (climb slows braking onset vs. descent) | qualitative, signed | `braking_pass_grade_matches_analytic_onset` |
| Energy closure: `Σ mg sinθ ds = mg Δz_lap = 0` on a closed lap | < 1e-6 | `gravity_work_closes_on_closed_lap` |

**Flat byte-stability.** For `θ=φ=κ_v=0`, `cosθ=cosφ=1.0` and
`sinθ=sinφ=0.0` **exactly**, so every 3D expression collapses to the flat
model — and `qss_lap_sim_3d` additionally short-circuits a geometrically
flat ribbon straight to the untouched 2D `qss_lap_sim`, so flat tracks
execute *identical* float ops rather than merely equivalent ones. Proven
bitwise (`.to_bits()` equality on every speed, lap time, and sector time)
across oval/circle/Silverstone by `flat_ribbon_qss_bitwise_matches_track`.
**`golden_oval_qss` moved 0.000 s / 0.000 RMSE** when this physics landed
(`PHYSICS_CHANGE.md`, 2026-07-07 "3D point-mass QSS physics" entry) — no
fixture regeneration was needed.

The grip-map mechanism (mu_scale grid) and the sector-marker
wiring added after this physics both preserve the same discipline —
absent-by-default, bitwise-identical to the pre-existing code paths
(`PHYSICS_CHANGE.md`, 2026-07-07 "Grip-map mechanism + sector markers"
entry).

## 3. The Spa result — a falsification finding

**Pre-registered criterion.** Before re-running Spa on the real 3D track,
the acceptance bar was set in advance: the fitted `power_scale` (uniquely
low at 0.833 on the flat model — see the 5-circuit table and parameter-scatter
analysis in [`correlation_summary.md`](correlation_summary.md#the-5-circuit-table))
should **rejoin the 1.0–1.16 cross-circuit pack**, and the flat model's
uniquely negative fitted lap delta (−4.06 s) should be **eliminated**, once
gravity is correctly modeled.

**Outcome: criterion NOT met.**

| quantity | flat 2-D | 3-D | verdict |
|---|---|---|---|
| preset lap delta | +1.841 s | +1.700 s | small improvement |
| preset speed RMSE | 9.32 m/s | 8.73 m/s | improved |
| **fitted lap delta** | −4.064 s | **−2.856 s** | improved ~30% toward 0, still negative |
| fitted speed RMSE | 4.28 m/s | 4.31 m/s | ~unchanged |
| **fitted `power_scale`** | 0.833 | **0.802** | did **NOT** rejoin the pack (slightly lower) |
| Silverstone control (all params) | — | < 1% movement | control held, as designed |

(Full table and hypotheses: `correlation_spa_2024q.md` § flat vs 3D elevation physics.)

**What improved.** The lap-time sign moved toward zero (−4.06 → −2.86 s)
and preset RMSE dropped — elevation *is* a real, correctly-modeled factor,
and the preset residual peak sits exactly at the Raidillon climb (below).

**The diagnostic chain:**
1. **Grade is conservative on a closed lap — verified analytically, not just
   asserted.** `gravity_work_closes_on_closed_lap` proves
   `Σ mg sinθ ds = 0` to < 1e-6 in the same code that runs the Spa fit. A
   term with zero net work per lap can redistribute *where* speed is
   won/lost but cannot shift a single lap-wide power multiplier — so
   `power_scale` was always a poor lever for an elevation fix.
2. **Residual segmentation.** The *preset* max |Δv| sits at **s ≈ 1140 m**
   (top of Raidillon / Kemmel-straight entry) on both the flat and 3D
   models. After fitting, the worst residual **migrates** to the
   **Pouhon → Stavelot descent** (s ≈ 3890–4570 m on the flat model;
   s ≈ 4680 m on the 3D model) and *grows* — max |Δv| there goes from
   **14.3 → 17.8 m/s**. The point-mass QSS free-wheels downhill faster than
   the energy-managed real car, and that over-carry is what keeps the fit
   de-powered.
3. **Measured-throttle evidence — reproducible via `tools/grade_throttle_stats.py`.**
   The diagnostic bins each aligned telemetry sample by the **local track
   grade at that sample's own projected station** (`dz/ds`, computed by
   central difference at the 3D track's native point spacing — the same
   discretization `Ribbon3d::from_centerline_3d` uses), not by a fixed
   station range — climb/descent/flat sub-segments are otherwise mixed
   together and the result becomes highly sensitive to the exact boundary
   chosen. At a **±2% grade threshold**:

   ```
   python tools/grade_throttle_stats.py telemetry/spa_2024_Q_aligned.csv tracks/spa_3d.json --threshold 0.02
   ```

   | bin | n | throttle p50 | full-throttle frac (`>0.95`) | any-brake frac |
   |---|---|---|---|---|
   | climb (`grade > +2%`) | 307 | **1.00** | 0.723 | 0.075 |
   | descent (`grade < −2%`) | 370 | **0.71** | 0.470 | 0.178 |
   | flat | 167 | 1.00 | 0.587 | 0.216 |

   Stable across nearby thresholds (1.5%–2.5%: descent p50 pinned at exactly
   0.71, full-throttle frac 0.470–0.472; climb p50 pinned at 1.00 throughout).
   The real driver is *managing* the descent (partial throttle almost half
   the time, braking on 18% of descent samples) rather than free-wheeling it
   the way the point-mass QSS does. This is direct, model-external
   confirmation that the over-carry is a genuine driver/energy-management
   gap, not a georeferencing or physics-sign error.

   **Silverstone control** (`--threshold 0.003`, since its elevation range is
   ~20× smaller): climb/descent/flat medians are all **1.00**, full-throttle
   fractions 0.61–0.76 with no climb-vs-descent gap — no throttle-management
   signature, as expected on a track with no meaningful grade to manage.

   *Provenance note:* an earlier adversarial audit pass attempted to
   reproduce this claim using contiguous **station-window** binning (e.g. "s
   in [4400, 4900) = descent") and got wildly threshold-sensitive,
   non-reproducing numbers — that was a **methodology mismatch**, not a
   sign the underlying claim was wrong. Per-sample grade binning (above) is
   the correct reconstruction of the original method and reproduces it
   cleanly.

**Verdict: (b), an intended and correctly-implemented physics change with an
honest negative result** — not a bug to chase. The grade force, 3D normal
load, and grip circle are all independently validated (§2) and the flat
code path is proven byte-identical; the residual is real and belongs to
deferred higher-fidelity / driver-energy-management work, not to this
model.

**Conclusion.** 3D elevation physics is real, correctly implemented, and
**necessary** — it measurably improves both the lap-time sign and preset
RMSE, and it is the only reason the Raidillon-climb residual is explained
at all. But it is **not sufficient** for Spa-class terrain without
dynamic/driver modeling: a point-mass model cannot represent a driver who
chooses not to use all available grip/power on a descent. That gap is
scoped to the deferred single-track / four-wheel / 14-DOF work.

**The Raidillon two-path confirmation (geometry-trust anchor).** Two
*independently derived* pieces of evidence place the same feature at the
same station: `tools/fetch_elevation.py`'s named-landmark annotation (built
from OSM way-matching + the raw elevation profile) marks **"Raidillon top"
at s ≈ 1140 m**; the correlation pipeline's residual analysis (built from
telemetry alignment + the QSS speed trace, with no knowledge of the
elevation import) independently finds the **preset max |Δv| at s ≈ 1140 m**
on *both* the flat and 3D models. These two paths — one geographic, one
telemetric — never share code or data, yet agree on the station to within
noise. That agreement is the trust anchor for the whole georeferencing +
elevation-import pipeline: the DEM-derived profile really is aligned to the
real track, not just plausible-looking.

## 4. Silverstone — the control

Silverstone has negligible elevation range (**10.710 m**, confirmed via
`Ribbon3d::validate`) relative to Spa's ~106 m. Every fitted parameter moved
**< 1%** when switching from the flat to the 3D model — exactly what a
control should do: a track with almost no elevation should be almost
unaffected by elevation physics.

**Demonstrated, not just asserted.** Running `qss_lap_sim` (flat) and
`qss_lap_sim_3d` (3D) side by side on the same Silverstone centerline
(calibrated car), lap time moves **112.0924 s → 112.0989 s** (6.5 ms, 0.006%)
and the three largest, non-adjacent `|Δv|` station-level speed-trace
differences are:

| station `s` | `|Δv|` (flat vs. 3D) |
|---|---|
| ≈ 3596 m | 0.328 m/s |
| ≈ 285 m | 0.275 m/s |
| ≈ 255 m | 0.257 m/s |

All three are **sub-1 m/s**, an order of magnitude below anything the Spa
analysis calls a residual "feature" (14–18 m/s swings at Raidillon and
Pouhon–Stavelot). This is the honest answer to a "speed-trace differences
at known elevation features" ask for Silverstone: **there is no
feature-level signal to tabulate, and this is a demonstrated null result,
not an absence of analysis** — a 10.7 m elevation range simply doesn't
produce corner-scale grade/load effects large enough to separate from
numerical noise in the QSS pass structure. This is the
implementation-correctness evidence that the 3D QSS isn't introducing a
systematic bias independent of terrain; it responds to *how much elevation
there is*, not to being turned on.

## 5. Scope ledger

| item | status | where recorded |
|---|---|---|
| Banking data | **0** for all real tracks — a 25–30 m DEM cannot resolve camber across a ~14 m track width | `tracks/README.md` § 3D elevation workflow; `docs/math/track3d.md` §5.9 |
| Banking *mechanism* | plumbed, unit-tested, manual per-corner override exists | `docs/math/track3d.md` §5.5 (banked closed-form test) |
| Higher-fidelity models (single-track / four-wheel / 14-DOF) | deferred — same 3D terms in a follow-up task | `PHYSICS_CHANGE.md` 2026-07-07 "3D point-mass QSS" entry |
| Real grip-map data (rubbered vs. dirty line) | **mechanism shipped, no data populated** | `PHYSICS_CHANGE.md` 2026-07-07 "Grip-map mechanism" entry; `tracks/README.md` § mu_scale grid |
| Sector markers | wired (schema + QSS), unpopulated for real tracks | `PHYSICS_CHANGE.md` 2026-07-07 "Grip-map mechanism" entry; `tracks/README.md` § Sector markers |
| Pit-lane polyline | deferred — no consumer until race-sim integration | `tracks/README.md` § Sector markers (pit lane note) |
| Multi-lap slicing | deferred | — |
| Golden lap regeneration | **not needed** — byte-stability held throughout the 3D track model work (better-than-planned outcome) | §2 above; `PHYSICS_CHANGE.md` entries |
