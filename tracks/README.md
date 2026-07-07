# Track files

Track definitions for Apex-14, stored as JSON and loaded with
`apex_track::load_track_json` (or `parse_track_json` for an in-memory string).

## Format

A track file is a JSON object:

| Field    | Type            | Required | Meaning                                                        |
|----------|-----------------|----------|----------------------------------------------------------------|
| `name`   | string          | yes      | Human-readable track name.                                     |
| `closed` | bool            | no       | Whether the track is a closed loop. Defaults to `true`.        |
| `width`  | number          | no       | Uniform track width (m), split evenly to both sides. Used only where a point omits its own widths. Defaults to `12.0`. |
| `points` | array of points | yes      | Centerline samples, in order. At least 3 are required.         |

Each point is an object:

| Field         | Type   | Required | Meaning                                              |
|---------------|--------|----------|------------------------------------------------------|
| `x`, `y`      | number | yes      | World coordinates (m).                               |
| `width_left`  | number | no       | Distance to the left boundary (m). Falls back to `width / 2`. |
| `width_right` | number | no       | Distance to the right boundary (m). Falls back to `width / 2`. |

Arc length, heading, and curvature are computed from the points by
`build_track` at load time; they are not stored in the file.

## Schema v2 (3D ribbon)

Schema **v2** adds an optional 3D extension. It is fully backward-compatible: a
file with no `version` field and no 3D per-point fields is **v1** (flat), and the
writer emits v2 markers **only** when 3D data is present, so existing files and
their diffs stay byte-stable.

| Field     | Type   | Required | Meaning                                                       |
|-----------|--------|----------|---------------------------------------------------------------|
| `version` | int    | no       | Schema version. Absent ⇒ v1 (flat). Emitted as `2` only for 3D files. Supported: 1, 2. |

Per-point 3D fields:

| Field         | Type   | Required | Meaning                                                   |
|---------------|--------|----------|-----------------------------------------------------------|
| `z`           | number | no       | Elevation (m). Absent ⇒ flat (`z = 0`).                   |
| `banking_deg` | number | no       | Surface banking / roll angle (degrees). Absent ⇒ unbanked.|

Loading:

- `load_track_json` / `parse_track_json` — the **2D** path, unchanged. Any file
  (v1 or v2) loads as a flat 2D `Track`; 3D fields are ignored.
- `load_ribbon3d_json` / `parse_ribbon3d_json` — the **3D** path, returning a
  [`Ribbon3d`]. A v1 or flat-v2 file loads as a byte-exact flat ribbon (through
  the same `build_track` pipeline); a file with any `z`/`banking_deg` builds a 3D
  ribbon (`Ribbon3d::from_centerline_3d`). See `docs/math/track3d.md`.

`export_ribbon3d_json` writes v2 (with `version: 2` and per-point `z` /
`banking_deg`) only when the ribbon is non-flat; a flat ribbon serializes as
v1-compatible output, byte-identical to `export_track_json`.

### 3D example (v2)

```json
{
    "version": 2,
    "name": "Banked ramp",
    "closed": false,
    "points": [
        { "x": 0.0,   "y": 0.0, "z": 0.0,  "banking_deg": 0.0 },
        { "x": 100.0, "y": 0.0, "z": 5.0,  "banking_deg": 4.0 },
        { "x": 200.0, "y": 0.0, "z": 12.0, "banking_deg": 6.0 }
    ]
}
```

### mu_scale grid (Phase 1.4)

An optional `mu_scale_grid` block attaches a `mu_scale(s, n)` grip-multiplier
grid to a v2 ribbon: a bilinearly-interpolated `(station, lateral)` grid,
`1.0` = baseline grip. Absent ⇒ uniform `1.0` (no grid at all) — the
byte-stable default; this ships the **mechanism** only, nothing populates
real grip-map data yet (a real dirty-line-vs-racing-line dataset is future
work).

| Field      | Type            | Required | Meaning                                                             |
|------------|-----------------|----------|----------------------------------------------------------------------|
| `stations` | array of number | yes      | Arc-length row stations (m), strictly increasing, `stations[0] == 0`. |
| `lateral`  | array of number | yes      | Lateral column offsets (m), strictly increasing, left-positive (matches the ribbon frame's `n`, `docs/math/track3d.md` §1). |
| `values`   | array of number | yes      | Grip multiplier at each grid point, row-major (`stations.len() * lateral.len()` entries). |

`apex_track::Ribbon3d::mu_scale_grid` carries the parsed grid;
`apex_track::MuScaleGrid::mu_at(s, n, total_length, closed)` samples it
(`s` wraps cyclically on a closed ribbon; `n` clamps to the grid's lateral
extent). **QSS never reads this field directly** — see
`apex_physics::qss_lap_sim_3d_with_grip`'s docs: a driven-line run must
sample the *original* ribbon's grid at the driven path's own `(s, n)`, not
whatever ribbon QSS happens to be running on, so the multiplier vector is
always baked by the caller (centerline: `Ribbon3d::centerline_mu_scale`;
driven line: `apex_correlate::driven`).

```json
{
    "version": 2,
    "name": "Dirty vs racing line",
    "closed": true,
    "mu_scale_grid": {
        "stations": [0.0, 500.0, 1000.0],
        "lateral": [-6.0, 0.0, 6.0],
        "values": [
            0.85, 1.05, 0.85,
            0.85, 1.05, 0.85,
            0.85, 1.05, 0.85
        ]
    },
    "points": [
        { "x": 0.0,   "y": 0.0 },
        { "x": 500.0, "y": 0.0 },
        { "x": 500.0, "y": 500.0 }
    ]
}
```

### Sector markers (Phase 1.2)

An optional `sector_markers` field (top-level, available at v1 or v2 — same
as `metadata`) lists ascending arc-length stations marking the start of
each sector *after* the first (the first sector always starts at `s = 0`),
so `n_sectors = sector_markers.len() + 1`. Absent ⇒ the classic
equal-arc-length-thirds split (`apex_physics::DEFAULT_SECTOR_COUNT`).
`apex_track::Track`/`Ribbon3d::sector_markers` carry the parsed value;
`qss_lap_sim`/`qss_lap_sim_3d`/`qss_lap_sim_tire` honor it automatically
when present via `apex_physics::sector_times_with_markers`.

```json
{ "name": "...", "closed": true, "sector_markers": [1800.0, 4200.0], "points": [...] }
```

**Pit lane polyline (deferred).** A `pit_lane_polyline` field is a natural
future v2 addition (a separate centerline for the pit-entry/exit path), but
has no consumer today — nothing in the workspace reads a pit lane
independent of the main track. Deferred until race-sim integration; no
field exists yet.

## Minimal example

```json
{
    "name": "Triangle",
    "closed": true,
    "width": 10.0,
    "points": [
        { "x": 0.0,   "y": 0.0 },
        { "x": 100.0, "y": 0.0 },
        { "x": 50.0,  "y": 80.0 }
    ]
}
```

Per-point widths override the uniform default:

```json
{
    "name": "Variable width",
    "closed": false,
    "points": [
        { "x": 0.0,  "y": 0.0, "width_left": 6.0, "width_right": 4.0 },
        { "x": 10.0, "y": 0.0, "width_left": 7.0, "width_right": 5.0 },
        { "x": 20.0, "y": 0.0, "width_left": 8.0, "width_right": 3.0 }
    ]
}
```

## Sample files

- `test_circle.json` - 36 points around a 50 m-radius circle (10° spacing), width 10 m.
- `oval_simple.json` - an oval with 500 m straights and 80 m-radius corners, width 12 m.

## 3D elevation workflow (Phase 1.2)

Real 3D track files are produced from the (georeference-less) TUMFTM centerline
plus external elevation data. **All of it stays local / gitignored** (TUMFTM is
LGPL-3.0; OSM is ODbL; DEM tiles have their own terms). The network lives only in
the Python tools; the Rust side is a deterministic, offline merge.

```bash
# 1. Georeference local metres -> WGS84 by shape-matching the centerline to the
#    OSM raceway outline (Overpass). Writes tracks/<c>.georef.json (+ residual RMS).
python tools/georef.py spa

# 2. Sample a DEM along the georeferenced centerline, smooth z(s), write the
#    elevation sidecar + profile SVG. Caches DEM/OSM responses (zero-cost re-runs).
python tools/fetch_elevation.py spa

# 3. Merge z(s) into the 2D centerline and write a v2 3D track (Rust, no network).
apex-14 import-track --input tracks/spa.json --elevation tracks/spa.elevation.json \
    --output tracks/spa_3d.json --name spa
```

- **Georeferencing.** TUMFTM CSVs carry no georeference (header is only
  `x_m,y_m,w_tr_right_m,w_tr_left_m`). `tools/georef.py` fits a 2D similarity from
  our centerline onto the OSM `highway=raceway` ways (point-to-segment trimmed
  ICP). Reported accuracy: Spa ≈ 4.9 m coverage-RMS (98 % within 30 m),
  Silverstone ≈ 4.4 m (92 %) — both well under the 15 m sub-DEM-cell target.
  **This target is machine-checked, not prose-only:** `georef.py`'s sidecar
  (`tracks/<circuit>.georef.json`) carries `coverage_rms_m`, `residual_rms_m`,
  and `scale`; `import-track --elevation` reads that sidecar (if present next
  to the elevation sidecar) and prints it, **hard-warning to stderr**
  (non-fatal — you may knowingly proceed) if `coverage_rms_m` exceeds 15 m or
  `scale` falls outside `[0.95, 1.05]`. Absent sidecar ⇒ silently skipped (a
  synthetic or hand-authored track has no georeferencing step to gate).
- **DEM.** The task named Copernicus **GLO-30**; OpenTopoData's free tier does not
  serve it, so we use **EU-DEM 25 m** (`eudem25m`, also a Copernicus Land
  Monitoring Service product, finer posting, Europe-only) with `srtm30m` then
  Open-Elevation as fallbacks.
- **Banking** stays `0`: a 25–30 m DEM cannot resolve camber across a ~14 m track
  width. The `banking_deg` field is the manual per-corner override mechanism,
  populated later if a physics slice needs it.
- **Determinism.** With a warm cache the elevation sidecar (and thus the v2 file)
  is byte-reproducible; `APEX_REPRO_TIMESTAMP` pins any timestamp.

## Importing Real Track Data

Apex-14 can import tracks from the
[TUMFTM Racetrack Database](https://github.com/TUMFTM/racetrack-database)
(LGPL-3.0 licensed). Those CSV files are **not** redistributed here - clone the
upstream repository and import them at run time.

### Quick start

```bash
# Clone the racetrack database
git clone https://github.com/TUMFTM/racetrack-database.git /tmp/tracks

# The tracks are in /tmp/tracks/tracks/*.csv
# Apex-14 can load them directly with load_tumftm_csv()
```

```rust
use std::path::Path;
use apex_track::{load_tumftm_csv, export_track_json};

// Import a TUMFTM CSV into the native Track type...
let track = load_tumftm_csv(Path::new("/tmp/tracks/tracks/Silverstone.csv"), "Silverstone")?;

// ...and optionally re-save it as Apex-14 JSON for offline reuse.
std::fs::write("tracks/silverstone.json", export_track_json(&track)?)?;
```

`export_tumftm_csv` performs the reverse conversion (native `Track` →
TUMFTM CSV string).

### Supported formats

- **TUMFTM CSV**: `x_m,y_m,w_tr_right_m,w_tr_left_m` (meters, closed circuit).
  The header line is optional; closure is implicit (the last point is not
  repeated). `w_tr_right_m` / `w_tr_left_m` are half-widths from the centerline.
- **Apex-14 JSON**: native format with optional per-point widths (see above).

### License note

The TUMFTM data is LGPL-3.0. To respect that license, Apex-14 does not commit
any TUMFTM-derived `.csv` or `.json` track files to this repository; import them
locally from the upstream source instead.
