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

- `test_circle.json` — 36 points around a 50 m-radius circle (10° spacing), width 10 m.
- `oval_simple.json` — an oval with 500 m straights and 80 m-radius corners, width 12 m.

## Importing Real Track Data

Apex-14 can import tracks from the
[TUMFTM Racetrack Database](https://github.com/TUMFTM/racetrack-database)
(LGPL-3.0 licensed). Those CSV files are **not** redistributed here — clone the
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
