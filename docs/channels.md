# Channel registry

The central telemetry channel registry lives in
`apex_telemetry::channels` (`crates/apex-telemetry/src/channels.rs`). It is the
single source of truth for every channel the workspace produces or consumes:
each has a stable snake_case name, a `Unit`, a coarse `Quantity` (for plot
grouping), a human `display_name`, and a one-line description.

Producers reference the registry at compile time (`ChannelId::name()`,
`display_name()`, `unit().symbol()`) instead of ad-hoc string literals — CSV
headers, viewer/HUD labels, and the UDP `OutputPacket` field docs all resolve to
these entries.

## Extension policy (frozen contract)

- **Append only.** Add new channels to the `define_channels!` table; do not
  reorder existing rows in a way that changes meaning.
- **Names are frozen once released.** A channel's `name` is a wire/format
  identity (CSV header, serde tag). Renaming it breaks stored data.
- **Units never change meaning for an existing name.** If a signal needs a
  different unit, add a *new* channel (e.g. `speed` in m/s and `speed_kph` in
  km/h are two channels of the same `Quantity::Speed`) — never repoint an
  existing name to a new unit.
- **Every future consumer reads the registry.** MoTeC `.ld` export, telemetry
  correlation, and the Python bindings must map through `ChannelId` /
  `ChannelSpec`, not re-declare channel names.

Invariants are enforced by tests in `channels.rs`: unique names, unique ids,
every channel has exactly one spec (the exhaustive `spec()` match makes a
spec-less variant a compile error), and `from_name(name(id)) == id` for all ids.

## Per-source audit (what was found where)

The registry was seeded from every channel that existed in the codebase:

| Source | Channels |
|--------|----------|
| QSS CSV (`apex_telemetry::export_qss_csv`) | `s`, `x`, `y`, `speed`, `speed_kph`, `lateral_g`, `longitudinal_g`, `curvature` |
| optimize detailed CSV (`bins/optimize`, 14-DOF) | `t`, `s`, `speed_kph`, `lateral_offset`, `roll_deg`, `pitch_deg`, `susp_fl…rr`, `fz_fl…rr`, `lateral_g`, `longitudinal_g`, `ride_height_front`, `ride_height_rear` |
| optimize racing-line CSV (`bins/optimize`) | `s`, `n`→`lateral_offset`, `v_kph`→`speed_kph`, `f_drive`, `curvature_cmd` |
| viewer telemetry plots (`apex-viewer`) | `speed_kph` (km/h plot), `lateral_g`, `longitudinal_g`, `curvature`, x-axis `s` |
| viewer HUD (`apex-viewer::hud`) | `speed_kph`, `lateral_g` (`accel_lat`), `longitudinal_g` (`accel_long`), `gear`, `wheel_fl…rr`, `lap`, `lap_time`, `sim_time` |
| UDP `OutputPacket` (`apex-sim`) | `pos_x/y/z`, `roll`, `pitch`, `yaw`, `speed`, `lateral_v`, `vertical_v`, `yaw_rate`, `wheel_fl…rr`, `susp_fl…rr`, `longitudinal_g` (`accel_long`), `lateral_g` (`accel_lat`), `gear`, `lap`, `lap_time`, `sim_time`, `sequence`, `track_distance`, `track_offset` |

### Naming conflicts resolved (registry name wins)

- `v_kph` (optimize) → **`speed_kph`**.
- `n` (optimize racing line) → **`lateral_offset`**.
- UDP `accel_lat` / `accel_long` → **`lateral_g`** / **`longitudinal_g`** (wire
  field names unchanged; only the registry mapping is documented).
- `speed` (m/s) vs `speed_kph` (km/h): kept as two channels sharing
  `Quantity::Speed`; `speed` is canonical SI, `speed_kph`/`roll_deg`/`pitch_deg`
  are display-unit siblings whose descriptions cross-reference the SI channel.

Human display labels live in `display_name` (e.g. `speed`/`speed_kph` both
display "Speed"); the registry `name` is the serialization identity.

## CSV provenance units line

CSV exports add one `# columns:` line to the provenance block, listing each
column and its unit as `name[symbol]` pairs, e.g.:

```
# columns: s[m], x[m], y[m], speed[m/s], speed_kph[km/h], lateral_g[g], longitudinal_g[g], curvature[1/m]
```

It sits between the `# key: value` metadata lines and the blank `#` separator
that precedes the header row, so a comment-aware reader
(`ReaderBuilder::comment(Some(b'#'))`) skips it transparently.
