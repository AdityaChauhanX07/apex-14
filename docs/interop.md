# Industry interop: MoTeC i2 `.ld` and Apache Parquet

Apex-14 exports telemetry to two industry-standard formats in addition to its
own [CSV interchange format](telemetry_format.md):

| Format | Module | CLI | Purpose |
|---|---|---|---|
| MoTeC i2 `.ld` | `apex-telemetry::motec` | `apex-14 export-ld` | Open our telemetry in MoTeC i2 (Pro) |
| Apache Parquet | `apex-telemetry::parquet_export` (feature `parquet`) | `apex-14 export-parquet`, `correlate --parquet` | Columnar analytics (pandas / arrow / DuckDB) |

Both are driven by the [channel registry](../crates/apex-telemetry/src/channels.rs):
column names are registry [`ChannelId::name`]s and units come from
[`Unit::symbol`], so neither format can drift from the registry. Provenance
(descriptive source metadata, and a sim [`RunMetadata`] where applicable) is
carried into each format's native metadata.

`apex-telemetry` is **not** in the wasm graph, and the Parquet dependency
(arrow-rs) sits behind a default-on `parquet` feature, so `--no-default-features`
builds stay light and wasm is untouched. Verify:

```
cargo check -p apex-telemetry --no-default-features        # builds without arrow/parquet
cargo tree -p web-viewer --target wasm32-unknown-unknown | grep -E 'apex-telemetry|parquet|arrow'   # → empty
```

---

## MoTeC i2 `.ld`

The `.ld` format is proprietary but well reverse-engineered. Our writer follows
the byte layout of the third-party Python reference
[`ldparser`](https://github.com/gotzl/ldparser) (its `ldData.frompd` / `*.write`
methods) so that both `ldparser` and MoTeC i2 read what we produce. It is **pure
Rust**, no C dependencies.

### File structure

| Block | Size (bytes) | Offset | Contents |
|---|---|---|---|
| `ldHead` | 1762 | 0 | meta/data pointers, channel count, date/time, driver / vehicle / venue / short-comment |
| `ldEvent` | 1154 | 1762 | event name / session / 1024-byte comment (`venue_ptr = 0`, no chained sub-blocks) |
| `ldChan` × N | 124 each | 2916 | per-channel meta, forward/back-linked by absolute file offset |
| data | 4·samples each | after metas | contiguous little-endian `f32`, one channel after another |

### Encoding & precision

Channel data is little-endian **`f32`** (`dtype_a = 0x07`, `dtype = 4`) with the
format's `scale`/`shift`/`mul`/`dec` transform set to identity, so the stored
word is the value verbatim.

**Precision loss:** `f32` carries ~7 significant decimal digits (~1e-7 relative).
Converting our `f64` samples loses precision at that level — e.g. a 300 kW power
rounds to ~0.03 W, a 100 m/s speed to ~1e-5 m/s, a 20 kN load to ~2e-3 N — far
below i2's display resolution. Independent cross-check on the Silverstone file
measured a max relative error of **4.2e-8** on `speed` (see verification below).

### Time base (grid handling)

i2 is time-based: every channel has an integer sample rate (Hz) and timing is
implicit in the sample index. We resample onto a uniform time grid built from
the `t` channel:

- a **t-grid** file already has `t` as its axis — mapped naturally;
- an **s-grid** file (uniform in distance) is resampled against its `t` channel;
- an **s-grid file with no `t` channel is refused** (`MotecError::NoTimeAxis`) —
  there is no way to place it on i2's time base.

The `t` channel is consumed as the axis and is **not** re-emitted as a data
channel (its timing is implicit). Every other channel is emitted. The output
rate defaults to a nominal `round((n-1)/t_span)` Hz (clamped to `[1, 1000]`),
overridable with `export-ld --rate`.

### NaN policy — hold-last + gap marker

i2 has no NaN concept for numeric channels, so gaps are **hold-last** filled:
each `NaN` (a measured dropout or a resample gap) takes the previous finite
value; leading `NaN`s before the first finite sample become `0.0`. A synthetic
unit-less channel **`gap_fill`** (1.0 / 0.0) marks every sample where at least
one channel was filled, so gaps stay visible in i2 rather than silently masked.

### Channel / unit mapping (registry → i2)

Channel names are the registry names verbatim (≤ 32 chars; short-name = first 8
chars). Units map from [`Unit::symbol`] to an ASCII i2 unit string (the `.ld`
unit field is ASCII, ≤ 12 chars):

| Registry unit | Symbol | i2 unit | Note |
|---|---|---|---|
| Meter | `m` | `m` | |
| MeterPerSecond | `m/s` | `m/s` | |
| KilometerPerHour | `km/h` | `km/h` | |
| RadPerSecond | `rad/s` | `rad/s` | |
| G | `g` | `G` | i2 acceleration convention |
| RadPerMeter | `1/m` | `1/m` | |
| Radian | `rad` | `rad` | |
| Degree | `deg` | `deg` | |
| Newton | `N` | `N` | |
| Second | `s` | `s` | |
| Celsius | `°C` | `C` | ASCII-ized (drop the degree glyph) |
| Rpm | `rpm` | `rpm` | |
| Watt | `W` | `W` | |
| None (dimensionless) | (empty) | (empty) | e.g. `throttle`, `grip_util` |

### Provenance packing

Descriptive source metadata (the CSV `# key: value` header) is packed into the
`.ld` string fields so it surfaces in i2's session info:

| i2 field (max chars) | Source metadata keys (first present) |
|---|---|
| `driver` (64) | `driver` |
| `venue` (64) | `venue`, `location`, `circuit`, `track` |
| `vehicleid` (64) | `vehicle`, `car`, `team`, `vehicleid` (default `apex-14`) |
| event `name` (64) | `event`, `gp`, `official_event` |
| event `session` (64) | `session` |
| `short_comment` (64) | `apex-14 export; src <source|exporter>` |
| event `comment` (1024) | **all** metadata pairs joined `key=value; …` |

A sim `RunMetadata` is not applicable to `.ld` (the source is measured/inferred
telemetry); the full descriptive metadata rides in the event comment.

### Verification status: **independent-reader** (real-i2 **pending**)

- **Self round-trip** — `apex-telemetry` writes then reads back with its own
  `read_ld`, asserting names / units / freq / counts / values (unit tests).
- **Independent reader** — `tools/verify_ld.py` parses our output with the
  third-party GPL `ldparser` (fetched to a gitignored cache, never committed) and
  cross-checks values within `f32` precision. Confirmed on the Silverstone file
  (20 channels, 1200 samples @ 14 Hz, `speed` max rel error 4.2e-8).
- **Real i2** — *pending*: the final authority is opening the prepared file
  (below) in MoTeC i2 (Pro), free download.

---

## Apache Parquet

Behind the default-on `parquet` cargo feature (arrow-rs family). One Parquet
file, one **`f64`** column per channel.

### Schema & NaN

- One `Float64`, non-nullable column per channel; column names are registry
  names for single-source files, or explicit prefixed names for mixed files
  (see `correlate --parquet`).
- **`NaN` is preserved natively** — it is a normal `f64` value, not a null. No
  gap-filling; measured dropouts survive unchanged (verified: `NaN` in →
  `NaN` out).

### File metadata schema (footer key/value)

| Key pattern | Value | Written by |
|---|---|---|
| `units:<column>` | unit symbol (empty for dimensionless) | always, one per column |
| `grid` | `s` or `t` (the sample axis) | when known |
| *(descriptive keys)* | every source `# key: value` pair, verbatim | `export-parquet` |
| `run.config_hash`, `run.car_hash`, `run.track_hash`, `run.settings_hash`, `run.git_sha`, `run.apex_version`, `run.seed`, `run.timestamp` | sim [`RunMetadata`] | when the source is a sim artifact |
| `measured.<key>` | measured-source descriptive pairs | `correlate --parquet` (hybrid) |
| `sim_line` | which line the QSS ran on | `correlate --parquet` |

Units are stored as individual `units:<column>` keys (greppable, self-describing)
rather than a JSON blob.

### `correlate --parquet` column naming

The correlation grid mixes a measured trace and a sim trace, so columns are
explicitly **prefixed** to disambiguate (registry names stay pure for
single-source files):

| Column | Unit | Meaning |
|---|---|---|
| `s` | m | common arc-length grid |
| `meas_speed` | m/s | measured speed on the grid |
| `sim_speed` | m/s | QSS sim speed on the grid |

Provenance is the documented [hybrid](telemetry_format.md#hybrid-provenance-on-correlation-artifacts):
sim `RunMetadata` under `run.*`, measured descriptive under `measured.*`.

### Verification status: **independent-reader**

- **Self round-trip** — `apex-telemetry` writes then reads back with its own
  `read_parquet`, asserting columns, values, `NaN` preservation, and metadata
  (unit tests).
- **Independent reader** — `tools/verify_parquet.py` reads our output with
  `pyarrow` and cross-checks a column exactly (Parquet is lossless `f64`).
  Confirmed on the Silverstone file (20 columns, 1166 rows, `speed` max abs diff
  0, all units + provenance present).

---

## CLI

```
# MoTeC .ld from a standard telemetry CSV (t-grid, or s-grid with a t column)
apex-14 export-ld --telemetry <telemetry.csv> --out <file.ld> [--rate <Hz>]

# Straight CSV → Parquet
apex-14 export-parquet --telemetry <telemetry.csv> --out <file.parquet>

# Correlation aligned grid (measured + sim speed) → Parquet
apex-14 correlate --telemetry <csv> --track <trk> --out-dir <dir> --parquet <file.parquet>
```

### Prepared file for MoTeC i2

A ready-to-open `.ld` built from the real Silverstone inferred lap is written to:

```
target/interop/silverstone_2024_Q_inferred.ld
```

Regenerate it with:

```
apex-14 export-ld \
  --telemetry telemetry/silverstone_2024_Q_inferred.csv \
  --out target/interop/silverstone_2024_Q_inferred.ld
```

(`target/` is gitignored — nothing derived from measured telemetry is committed.)
