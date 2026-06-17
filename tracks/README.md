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
