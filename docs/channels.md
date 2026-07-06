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

## Measured-telemetry channels (Phase 2 correlation)

The telemetry-correlation importer (`apex-correlate`, see
[`docs/telemetry_format.md`](telemetry_format.md)) consumes *measured* car
telemetry (e.g. FastF1 exports). It maps every source column through this
registry, so a handful of driver-input / powertrain channels were appended
(append-only, per the extension policy):

| Channel | Unit | Quantity | Notes |
|---------|------|----------|-------|
| `throttle` | — (dimensionless) | `Dimensionless` | Pedal position, 0…1 (percent sources are converted to a 0–1 fraction on import). |
| `brake` | — (dimensionless) | `Dimensionless` | 0…1 fraction, or a 0/1 on-off flag (FastF1 exposes a boolean). |
| `rpm` | `rpm` | `AngularVelocity` | Engine speed. Modelled as an **angular velocity with its own display unit** (`Unit::Rpm`, symbol `rpm`), not a `Count`: it is a genuine physical rotational speed, and keeping `rpm` as the canonical unit avoids a lossy rpm→rad/s round-trip on the raw ECU value. (Grouped under `AngularVelocity` for coarse plot binning; it is not co-plotted with the rad/s wheel-speed channels.) |
| `steering_angle` | `rad` | `Angle` | Measured steering channel; degree sources convert via `Unit::si_factor()`. |
| `s_raw` | `m` | `Distance` | Raw source arc length (e.g. FastF1 integrated `Distance`) kept alongside the re-projected geometric `s` after track alignment. |

`x`, `y` (world position) and `gear` already existed and are reused as-is for
measured telemetry.

### Track-frame alignment & projection (`apex-correlate::align` / `::project`)

The `apex-14 telemetry-align` command fits a 2D similarity transform mapping the
FastF1-local `x`/`y` onto the Apex track frame, then projects each sample onto
the centerline. The aligned/projected CSV:

- **`s`** ← re-projected **geometric** arc length (station of the closest
  centerline point), unwrapped to be monotone across the start/finish line.
- **`s_raw`** ← the original FastF1 `Distance` (speed-integrated), retained for
  comparison. The two spans differ by ~1% (geometric vs. integrated).
- **`x`, `y`** ← rewritten into the **Apex track frame** (were FastF1-local).
- **`lateral_offset`** ← signed offset `n` of the driven line from the
  centerline. **Sign: positive = LEFT of the centerline in the direction of
  travel** (matches the viewer boundary construction `left = center +
  width_left·normal` with left normal `(heading + π/2)`, and the optimizer's
  `lateral_offset`). Negative = right.

The fitted transform is persisted to a gitignored `*.align.toml` sidecar and
echoed into the aligned CSV's `# align_*` header comments (descriptive
provenance only — measured/derived data carries no `RunMetadata` sim hashes).
