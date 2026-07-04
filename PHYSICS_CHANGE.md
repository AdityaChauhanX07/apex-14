# Physics Change Log

This file is the audit trail for the golden-lap regression harness
(`bins/apex-cli/tests/golden_lap.rs`). It guards against silent drift in the
simulation output — lap time and the resampled speed trace — for pinned
baseline scenarios. Right now there is exactly one baseline: the default oval
track with the calibrated F1-2024 car, run through `qss_lap_sim`
(`golden_oval_qss`).

QSS is bitwise-deterministic (no RNG, no rayon, no hashmap-order dependence
in the code path it exercises), so any drift in the golden test reflects an
actual change to the physics or track-generation code — not run-to-run noise.
The test still compares with tolerance rather than bitwise equality, because
floating-point results are not portable across compiler version, OS, or
optimization level, and a multi-OS CI matrix is coming in Phase 0.4. The two
tolerances currently in force, defined in `bins/apex-cli/tests/golden_lap.rs`:

- `LAP_TIME_TOL_S = 0.010` (10 ms)
- `SPEED_RMSE_TOL_MS = 0.1` (0.1 m/s RMSE over the resampled speed trace)

When the OS matrix lands, these are exactly what make the gate survive
running on multiple OS/toolchain combinations instead of breaking on
harmless FP rounding differences — the comment beside the constants in
`golden_lap.rs` says so explicitly.

## The rule

**A failing golden test is a STOP, not a thing to silence.** When
`cargo test -p apex-14 --test golden_lap` fails, the maintainer must decide:

- **(a) Unintended regression** — a bug was introduced. Fix the code, not
  the fixture. Do not touch the golden fixture to make the test pass.
- **(b) Intended physics change** — the shift is a deliberate consequence of
  a code change (e.g. a tire model refinement, a car parameter update, a
  track-generation change). In this case:
  1. Add a dated entry to the log below: what changed, why, and the
     observed lap-time delta and speed-RMSE delta.
  2. Regenerate the fixture in the **same commit/PR** as the code change:
     ```
     REGEN_GOLDEN=1 cargo test -p apex-14 --test golden_lap -- --ignored
     ```
  A code change and its golden-fixture update must never be split across
  separate PRs — that defeats the point of the gate.

**Known fidelity note:** `qss --calibrated` currently resolves to the
constant-`tire_mu` grip-circle model (`apex_physics::qss_lap_sim`), not the
Pacejka load-sensitive `qss_lap_sim_tire` path — there is no fidelity
selector on the CLI's `qss` subcommand today. If a future change rewires
`qss --calibrated` to go through `qss_lap_sim_tire` instead, the resulting
golden-fixture shift is **expected** and must be logged here as an intended
change, not chased as a bug.

## Template entry

```
### YYYY-MM-DD — <short title>
- Change: <what code/config changed>
- Rationale: <why>
- Lap-time delta: <old> -> <new> (<+/-Xs>)
- Speed-RMSE delta: <value> m/s
- Fixtures regenerated: <list, e.g. f1_2024_calibrated__oval_default__qss.json>
```

## Log

### 2026-07-03 — Golden-lap harness established
- Change: Added the golden-lap regression harness itself
  (`bins/apex-cli/tests/golden_lap.rs`) and the first pinned baseline.
- Rationale: Golden-lap harness established; oval QSS baseline pinned.
- Lap-time delta: n/a (initial baseline)
- Speed-RMSE delta: n/a (initial baseline)
- Fixtures regenerated: f1_2024_calibrated__oval_default__qss.json (created)
