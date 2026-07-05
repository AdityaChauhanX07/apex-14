# `telemetry/` — measured telemetry (local, not committed)

This directory holds **measured** car telemetry imported from external sources
(currently [FastF1](https://docs.fastf1.dev/)) in the standard *Apex telemetry
CSV* format (see [`../docs/telemetry_format.md`](../docs/telemetry_format.md)).
It feeds the `apex-correlate` crate (import → resample → correlate).

## Workflow

1. Export a lap with the FastF1 bridge (requires `pip install fastf1 pandas`):

   ```bash
   python tools/fastf1_export.py \
       --year 2024 --gp "Great Britain" --session Q --driver VER \
       --out telemetry/silverstone_2024_Q_VER.csv
   ```

   - `--driver` is optional; the default is the session's fastest lap.
   - `--lap fastest` (default) or `--lap N` selects the lap.
   - Output is registry-canonical: km/h→m/s, throttle percent→0–1, brake→0/1,
     FastF1 X/Y decimetres→metres. The header carries **descriptive** provenance
     (year/gp/session/driver/lap, `fastf1` version, exporter version) — **no**
     `RunMetadata` sim hashes, because measured data is not a sim artifact.

2. Import it in Rust via `apex_correlate::import_telemetry` (identity mapping —
   the file is already in registry names/units), then `resample_to_s` /
   `resample_to_t`.

## What lives here

- `*.csv` — exported measured laps. **Not committed** (see below).
- `.fastf1_cache/` — FastF1's local HTTP cache
  (`fastf1.Cache.enable_cache`). **Not committed.**

Both are git-ignored (see the repo `.gitignore`). This README is the only
committed file in `telemetry/`.

## Licensing / redistribution stance

FastF1 retrieves official F1 timing/telemetry data (via the F1 live-timing and
Ergast-style APIs) whose redistribution terms belong to FOM/F1, not to us. To
avoid taking a position we haven't deliberately decided, **derived telemetry
CSVs and the FastF1 cache stay local** — they are git-ignored and never pushed.

The `x`/`y` position columns exported here are in FastF1's **own local frame**,
not the Apex track frame; GPS/track alignment is a later task.

Committing curated or aggregated *derived summaries* (as opposed to raw dumps)
may be reconsidered later as a separate, deliberate decision — it is out of
scope for now.
