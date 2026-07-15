# Spa — does envelope-QSS (load-transfer grip) recover the residual? **No.**

**Part C of the envelope-analysis task.** The 3-D Spa re-fit left `power_scale` at a
uniquely low **0.802** and a persistent **descent over-carry** (the point-mass QSS
free-wheels down Pouhon→Stavelot faster than the real car), with the residual assigned
to dynamic/driver modelling and *suspected partly load-transfer physics*. This page
tests that suspicion: does swapping the point-mass friction-circle grip for the
**load-sensitive g-g-g envelope** — the grip law that *does* carry load transfer —
shrink the descent over-carry?

**Verdict: no, and the physics says it cannot.** The descent over-carry is not a
grip-limited phenomenon, so no grip law can touch it. The residual is further isolated
to **longitudinal / transient / driver** effects — the domain of the deferred dynamic
OCP. Cross-links: [`correlation_spa_2024q.md`](correlation_spa_2024q.md),
[`correlation_summary.md`](correlation_summary.md).

> **No re-identified `power_scale` is quoted here.** This is a *diagnostic* at the
> committed 3-D-fitted car, not a re-identification (see "Why not re-identify" below).
> Per the task's honesty rules, the diagnostic answers the physics question without
> introducing a new fitted number.

## Method (honesty-preserving)

- **Same** telemetry (`spa_2024_Q_aligned.csv`), **same** alignment, **same**
  reconstructed *measured driven line* (direct mode), **same** 3-D track (`spa_3d.json`),
  **same** committed 3-D-fitted car (`lift 4.884, drag 1.161, power_scale 0.8025`,
  μ fixed 1.55) as the original 3-D study. **The only thing changed is the grip law.**
- Fixed-line QSS forward/backward over the driven ribbon, with the grip limit taken from
  the cached g-g-g **envelope** `rho(theta; v, g_z)` (load-sensitive Pacejka + suspension,
  generated per car) instead of the point-mass friction circle `μ·(m·g_z + downforce)`.
- No optimizer, no OCP, no re-identification. The friction-circle sim reproduces the
  study exactly (sim lap **110.303 s** vs measured **113.159 s** = **−2.856 s**, matching
  the study's 3-D fitted lap delta to the millisecond — the harness is faithful).

## The decisive, calibration-free result — the descent is not grip-limited

Grip utilization, inferred from the **measured** speed on the driven line (friction-circle
inversion, `apex-14 infer`), by section:

| section | median grip util | p90 | fraction of stations > 0.9 (grip-limited) |
|---|---:|---:|---:|
| whole lap | 0.27 | 0.78 | 6 % |
| **descent (Pouhon→Stavelot, s 3890–4680)** | **0.45** | 0.83 | **8 %** |
| slow corners (v < 40 m/s) | 0.63 | 1.16 | **21 %** |
| high-speed (v > 75 m/s) | 0.17 | 0.48 | 0 % |
| Eau Rouge/Raidillon (s 700–1150) | 0.17 | 0.66 | 0 % |

Through the descent the real car uses **under half** its grip on median and is near the
limit at only 8 % of stations. **A section that is not grip-limited cannot have its speed
changed by *any* grip law** — friction-circle or load-sensitive envelope alike. The
descent over-carry is a **longitudinal** phenomenon: gravity assists the point-mass model
downhill and it never lifts, whereas the real driver manages the descent (the study's
measured median throttle 0.71). Load transfer modulates *grip*; it does not touch a
longitudinal energy-management gap. Grip-law changes can only bite in the **slow corners**
(21 % grip-limited) — which are not where the over-carry lives.

This is the a-priori physics argument, now confirmed against measured data, and it does
**not** depend on any envelope calibration.

## Why not re-identify (the scope + calibration finding)

Two obstacles make a full envelope-grip re-identification both large *and* ill-posed —
either would independently stop it under the task's "no new degrees of freedom / config-only"
rules:

1. **Per-iteration envelope regeneration.** `lift_coeff` is one of the study's freed
   parameters and it reshapes the grip envelope, so a proper LM fit would have to
   **regenerate the envelope every iteration** (a trim-grid solve). That is a several-hundred-line
   integration, well beyond the task's "config-only" scope.
2. **The fitted aero and the envelope aero are different representations — a calibration
   incompatibility.** The point-mass fit's effective downforce lives entirely in
   `car.lift_coeff = 4.884` (an *effective* value absorbing model error). But the envelope
   derives downforce from a **ride-height `AeroModel` map** (`solve_operating_point` →
   `aero.compute(...)`), which the identification **never touched**. So a naive grip-law
   swap runs the envelope at the *default* aero, not the fitted one: it under-grips by up
   to **~35 m/s at high speed** (measured directly — the friction-circle vs envelope speed
   gap peaks at 34.7 m/s at s≈1130 m, the Kemmel straight, exactly where downforce
   dominates). That is a **calibration artifact, not load-transfer physics**, and it swamps
   the effect under test. Re-identifying would not fix it without **re-parameterizing the
   aero into the envelope's ride-height representation** — new degrees of freedom the
   honesty rules forbid.

The naive swap's aggregate numbers (envelope sim lap 123.6 s, overall RMSE 11.3 m/s,
descent RMSE +90 %) are therefore reported here **only as evidence of the calibration
mismatch**, not as a physics result. They are not a fair envelope-vs-friction comparison.

## Conclusion

- **Escalation criterion (build the full re-identification only if the envelope materially
  moves the descent over-carry): NOT met.** The descent is not grip-limited, so envelope
  grip cannot move it; and the envelope is not even on the fitted car's grip scale.
- **`power_scale` is expected to stay ~0.80.** Its deficit is a stand-in for a missing
  **longitudinal** energy sink on the descent, which a grip-law change does not supply. We
  do not quote a re-identified value — running it would produce a number dominated by the
  aero calibration artifact, and "a wrong Spa number is worse than a delayed one."
- **The residual is formally isolated to transient / driver / energy-management effects** —
  the deferred single-track / four-wheel / 14-DOF **dynamic OCP** (PHYSICS_CHANGE 2026-07-07),
  which models exactly the throttle/brake management and transient load dynamics that a
  quasi-steady grip envelope, however load-sensitive, cannot. This *strengthens* the original
  Spa conclusion rather than overturning it: elevation is real but insufficient, and now
  **grip-side load-transfer is shown insufficient too** — the remaining Spa residual is
  longitudinal and dynamic.

## Reproduction

Diagnostic harness (scratch, not committed): friction-circle QSS via
`apex_correlate::driven::{build_driven_geometry_3d, run_qss_on_driven, geometry_to_trace}`;
envelope-grip QSS mirroring `qss_lap_sim_3d_with_grip` with the grip term replaced by the
cached `Envelope` (`rho`/`constraint`); both scored with the study's `correlate(...)`
pipeline. Grip-utilization table from `telemetry/spa_3d_inferred.csv` (`grip_util` column),
the committed `apex-14 infer` output for the 3-D-fitted car.
