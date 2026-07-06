//! MoTeC i2 `.ld` log-file writer (and a minimal reader for verification).
//!
//! The `.ld` format is proprietary but well reverse-engineered; this writer
//! follows the byte layout of the Python reference implementation
//! [`ldparser`](https://github.com/gotzl/ldparser) (its `ldData.frompd` /
//! `*.write` methods) so that both `ldparser` **and** MoTeC i2 (Pro) can read
//! what we produce. It is pure Rust — no C dependencies.
//!
//! # What we write
//!
//! - A fixed **header** (`ldHead`, 1762 bytes) carrying the channel-meta and
//!   data pointers, the channel count, a date/time, and the driver / vehicle /
//!   venue / short-comment string fields.
//! - One **event** block (`ldEvent`, 1154 bytes: name / session / 1024-byte
//!   comment) immediately after the header. The event's `venue_ptr` is 0, so no
//!   venue/vehicle sub-blocks are chained (the header already carries the venue
//!   and vehicle strings) — this mirrors `ldparser`'s minimal `frompd` layout.
//! - One **channel-meta** block (`ldChan`, 124 bytes) per channel, forward- and
//!   back-linked by absolute file offsets, then the channel **data** laid out
//!   contiguously in channel order.
//!
//! # Encoding & precision
//!
//! Channel data is written as little-endian **`f32`** (`dtype_a = 0x07`,
//! `dtype = 4`), with the format's `scale`/`shift`/`mul`/`dec` transform set to
//! identity (`scale = mul = 1`, `shift = dec = 0`), so the stored word is the
//! value verbatim. `f32` gives ~7 significant decimal digits; converting our
//! `f64` samples loses precision at roughly the 1e-7 relative level (e.g. a
//! 300 kW power stores to ~0.03 W, a 100 m/s speed to ~1e-5 m/s) — far below i2's
//! display resolution. This is documented in `docs/interop.md`.
//!
//! # Time base
//!
//! i2 is **time-based**: every channel has an integer sample rate (Hz) and
//! timing is implicit in the sample index. We therefore resample onto a uniform
//! time grid built from the telemetry's `t` channel:
//!
//! - a **t-grid** file already has `t` as its axis — natural;
//! - an **s-grid** file (uniform in distance) must still carry a `t` channel; we
//!   resample every channel against `t`. Without a `t` channel we **refuse**
//!   ([`MotecError::NoTimeAxis`]).
//!
//! The `t` channel is consumed as the axis and is **not** re-emitted as a data
//! channel (its timing is implicit); every other channel is emitted.
//!
//! # NaN policy
//!
//! i2 has no NaN concept for numeric channels, so gaps are **hold-last**
//! filled: each `NaN` (a real measured dropout, or a resample gap) takes the
//! previous finite value; leading `NaN`s before the first finite sample become
//! `0.0`. An extra synthetic channel **`gap_fill`** (unit-less, 1.0 / 0.0)
//! marks every sample where at least one channel was gap-filled, so the fill is
//! visible in i2 rather than silently masking missing data.

use std::io::Write as _;
use std::path::Path;

use crate::channels::{ChannelId, Unit};

// --- fixed block sizes (bytes), verified against ldparser struct layouts ---
const LD_HEAD_SIZE: usize = 1762;
const LD_EVENT_SIZE: usize = 1154;
const LD_CHAN_SIZE: usize = 124;
/// Absolute offset of the event block (right after the header).
const EVENT_PTR: u32 = LD_HEAD_SIZE as u32;
/// Absolute offset of the first channel-meta block (after header + event).
const META_PTR: u32 = (LD_HEAD_SIZE + LD_EVENT_SIZE) as u32;

/// Which coordinate the incoming columns are uniform in (used only for a clear
/// refusal message; the algorithm always resamples against the `t` channel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grid {
    /// Uniform in time (`t` is the axis).
    Time,
    /// Uniform in arc length (`s` is the axis; a `t` channel is still required).
    Distance,
}

impl Grid {
    fn as_str(self) -> &'static str {
        match self {
            Grid::Time => "t",
            Grid::Distance => "s",
        }
    }
}

/// Options controlling the `.ld` encoding.
#[derive(Debug, Clone)]
pub struct LdOptions {
    /// Output sample rate (Hz). `None` picks a nominal rate from the data
    /// (`round((n-1) / t_span)`, clamped to `[1, 1000]`).
    pub sample_rate_hz: Option<u16>,
    /// RFC3339 timestamp for the header date/time (and for deterministic
    /// output). Typically `run_metadata::now_rfc3339()`.
    pub timestamp: String,
}

impl Default for LdOptions {
    fn default() -> Self {
        LdOptions {
            sample_rate_hz: None,
            timestamp: crate::run_metadata::now_rfc3339(),
        }
    }
}

/// Summary of a written `.ld` file.
#[derive(Debug, Clone)]
pub struct LdReport {
    /// The uniform sample rate written (Hz).
    pub sample_rate_hz: u16,
    /// Samples per channel.
    pub n_samples: usize,
    /// Emitted channels, as `(name, unit)` pairs in file order (includes the
    /// synthetic `gap_fill` marker).
    pub channels: Vec<(String, String)>,
    /// Number of samples that were gap-filled in at least one channel.
    pub gap_filled_samples: usize,
    /// Time span covered (s).
    pub duration_s: f64,
}

/// Anything that can go wrong writing (or reading) a `.ld` file.
#[derive(Debug)]
pub enum MotecError {
    /// Filesystem I/O failure.
    Io(std::io::Error),
    /// No `t` channel present, so the data cannot be placed on i2's time base
    /// (carries the source grid kind for a clearer message).
    NoTimeAxis(Grid),
    /// Fewer than two samples, or no non-time channels to emit.
    TooFewSamples,
    /// The `t` axis is not finite and strictly increasing.
    NonMonotonicTime,
    /// A read-back file was malformed (verification reader only).
    Malformed(&'static str),
}

impl std::fmt::Display for MotecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MotecError::Io(e) => write!(f, "I/O error: {e}"),
            MotecError::NoTimeAxis(g) => write!(
                f,
                "cannot export {} telemetry to .ld: no `t` channel to build i2's \
                 time base from (a `t` column is required; i2 is time-based)",
                g.as_str()
            ),
            MotecError::TooFewSamples => {
                write!(f, "too few samples or no non-time channels to write")
            }
            MotecError::NonMonotonicTime => {
                write!(f, "`t` axis is not finite and strictly increasing")
            }
            MotecError::Malformed(m) => write!(f, "malformed .ld file: {m}"),
        }
    }
}

impl std::error::Error for MotecError {}

impl From<std::io::Error> for MotecError {
    fn from(e: std::io::Error) -> Self {
        MotecError::Io(e)
    }
}

/// Map a registry [`Unit`] to an ASCII unit string i2 accepts (≤ 12 chars).
///
/// Mostly the registry symbol; the two that differ are `°C` → `C` (the `.ld`
/// unit field is ASCII) and `g`-force → `G` (i2's acceleration convention).
pub fn i2_unit(unit: Unit) -> &'static str {
    match unit {
        Unit::Meter => "m",
        Unit::MeterPerSecond => "m/s",
        Unit::KilometerPerHour => "km/h",
        Unit::RadPerSecond => "rad/s",
        Unit::G => "G",
        Unit::RadPerMeter => "1/m",
        Unit::Radian => "rad",
        Unit::Degree => "deg",
        Unit::Newton => "N",
        Unit::Second => "s",
        Unit::Celsius => "C",
        Unit::Rpm => "rpm",
        Unit::Watt => "W",
        Unit::None => "",
    }
}

/// Write `columns` to `path` as a MoTeC i2 `.ld` file. See the module docs for
/// the time base, encoding, and NaN policy.
///
/// `columns` is `(channel, samples)` in the desired channel order; all sample
/// vectors must be the same length and one of them must be [`ChannelId::Time`].
/// `metadata` is descriptive provenance (the source CSV's `# key: value`
/// header), packed into the `.ld` string fields (see [`pack_provenance`]).
pub fn export_ld(
    path: &Path,
    grid: Grid,
    columns: &[(ChannelId, &[f64])],
    metadata: &[(String, String)],
    opts: &LdOptions,
) -> Result<LdReport, MotecError> {
    // --- locate the time axis ---
    let time = columns
        .iter()
        .find(|(id, _)| *id == ChannelId::Time)
        .map(|(_, v)| *v)
        .ok_or(MotecError::NoTimeAxis(grid))?;
    if time.len() < 2 {
        return Err(MotecError::TooFewSamples);
    }
    // The axis must be finite and strictly increasing to bracket-interpolate.
    if !time[0].is_finite() {
        return Err(MotecError::NonMonotonicTime);
    }
    for w in time.windows(2) {
        if !w[1].is_finite() || w[1] <= w[0] {
            return Err(MotecError::NonMonotonicTime);
        }
    }

    let t0 = time[0];
    let t_last = time[time.len() - 1];
    let span = t_last - t0;

    // --- choose the uniform rate + build the target time grid ---
    let rate = match opts.sample_rate_hz {
        Some(r) if r >= 1 => r,
        _ => {
            let nominal = ((time.len() - 1) as f64 / span).round();
            nominal.clamp(1.0, 1000.0) as u16
        }
    };
    let period = 1.0 / rate as f64;
    let n_out = (span / period + 1e-9).floor() as usize + 1;
    let grid_t: Vec<f64> = (0..n_out).map(|k| t0 + period * k as f64).collect();

    // --- resample every non-time channel onto the grid, then hold-last fill ---
    let emit: Vec<(ChannelId, &[f64])> = columns
        .iter()
        .filter(|(id, _)| *id != ChannelId::Time)
        .copied()
        .collect();
    if emit.is_empty() {
        return Err(MotecError::TooFewSamples);
    }

    let mut gap_any = vec![false; n_out];
    let mut filled: Vec<Vec<f32>> = Vec::with_capacity(emit.len());
    for (_, samples) in &emit {
        let interp = resample_linear(time, samples, &grid_t);
        let mut out = vec![0.0f32; n_out];
        let mut last = 0.0f64;
        let mut have_last = false;
        for (k, &v) in interp.iter().enumerate() {
            if v.is_finite() {
                last = v;
                have_last = true;
                out[k] = v as f32;
            } else {
                // Gap: hold last finite value (or 0.0 before the first).
                out[k] = if have_last { last as f32 } else { 0.0 };
                gap_any[k] = true;
            }
        }
        filled.push(out);
    }
    let gap_filled_samples = gap_any.iter().filter(|&&g| g).count();

    // --- assemble the emitted channel table (+ synthetic gap marker) ---
    struct Chan {
        name: String,
        short: String,
        unit: String,
        data: Vec<f32>,
    }
    let mut chans: Vec<Chan> = Vec::with_capacity(emit.len() + 1);
    for (i, (id, _)) in emit.iter().enumerate() {
        chans.push(Chan {
            name: id.name().to_string(),
            short: short_name(id.name()),
            unit: i2_unit(id.unit()).to_string(),
            data: std::mem::take(&mut filled[i]),
        });
    }
    chans.push(Chan {
        name: "gap_fill".to_string(),
        short: "gap_fill".to_string(),
        unit: String::new(),
        data: gap_any.iter().map(|&g| if g { 1.0 } else { 0.0 }).collect(),
    });

    let n = chans.len();
    let data_ptr0 = META_PTR + (n as u32) * (LD_CHAN_SIZE as u32);
    let data_bytes_per_chan = (n_out * 4) as u32;

    // --- build the byte image ---
    let total = data_ptr0 as usize + n * n_out * 4;
    let mut buf = vec![0u8; total];

    write_header(&mut buf, n as u32, data_ptr0, metadata, &opts.timestamp);
    write_event(&mut buf, metadata);

    for (k, c) in chans.iter().enumerate() {
        let base = META_PTR as usize + k * LD_CHAN_SIZE;
        let prev = if k == 0 {
            0
        } else {
            META_PTR + ((k - 1) as u32) * LD_CHAN_SIZE as u32
        };
        let next = if k + 1 == n {
            0
        } else {
            META_PTR + ((k + 1) as u32) * LD_CHAN_SIZE as u32
        };
        let data_ptr = data_ptr0 + (k as u32) * data_bytes_per_chan;
        write_chan_meta(
            &mut buf,
            base,
            prev,
            next,
            data_ptr,
            n_out as u32,
            rate,
            k as u16,
            &c.name,
            &c.short,
            &c.unit,
        );
        // channel data (little-endian f32, contiguous)
        let mut off = data_ptr as usize;
        for &v in &c.data {
            buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
            off += 4;
        }
    }

    let mut file = std::fs::File::create(path)?;
    file.write_all(&buf)?;
    file.flush()?;

    Ok(LdReport {
        sample_rate_hz: rate,
        n_samples: n_out,
        channels: chans.into_iter().map(|c| (c.name, c.unit)).collect(),
        gap_filled_samples,
        duration_s: span,
    })
}

/// Linear interpolation of `samples` (defined at strictly-increasing finite
/// `axis`) onto `grid`. A grid point outside `[axis[0], axis[last]]`, or one
/// whose bracketing source samples are non-finite, yields `NaN` (the caller
/// hold-last fills these).
fn resample_linear(axis: &[f64], samples: &[f64], grid: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(grid.len());
    let mut i = 0usize;
    let last = axis.len() - 1;
    for &x in grid {
        while i + 1 < axis.len() && axis[i + 1] < x {
            i += 1;
        }
        if x < axis[0] || x > axis[last] {
            out.push(f64::NAN);
            continue;
        }
        let (ax, bx) = (axis[i], axis[i + 1]);
        let (a, b) = (samples[i], samples[i + 1]);
        if x == ax {
            out.push(a);
        } else if x == bx {
            out.push(b);
        } else if !a.is_finite() || !b.is_finite() {
            out.push(f64::NAN);
        } else {
            let f = (x - ax) / (bx - ax);
            out.push(a + f * (b - a));
        }
    }
    out
}

/// An 8-char short name (the `.ld` `short_name` field), from the registry name.
fn short_name(name: &str) -> String {
    name.chars().take(8).collect()
}

// ---------------------------------------------------------------------------
// Provenance packing
// ---------------------------------------------------------------------------

/// Pick the first present value among `keys` (case-sensitive) from `metadata`.
fn pick<'a>(metadata: &'a [(String, String)], keys: &[&str]) -> Option<&'a str> {
    for k in keys {
        if let Some((_, v)) = metadata.iter().find(|(mk, _)| mk == k) {
            return Some(v.as_str());
        }
    }
    None
}

/// The provenance strings packed into the `.ld` header/event fields, derived
/// from the descriptive source metadata.
///
/// Recognized keys map to dedicated i2 fields so they surface in the session
/// info; the full metadata list is additionally joined into the 1024-byte event
/// comment so nothing is lost.
pub struct Provenance {
    /// Driver (header `driver`, 64 chars).
    pub driver: String,
    /// Vehicle id (header `vehicleid`, 64 chars).
    pub vehicle: String,
    /// Venue / circuit (header `venue`, 64 chars).
    pub venue: String,
    /// Event name (event `name`, 64 chars).
    pub event: String,
    /// Session (event `session`, 64 chars).
    pub session: String,
    /// Short comment (header `short_comment`, 64 chars).
    pub short_comment: String,
    /// Full comment (event `comment`, 1024 chars) — all metadata joined.
    pub comment: String,
}

/// Build [`Provenance`] from descriptive `metadata`, using a documented
/// precedence for each dedicated field and joining every pair into the comment.
pub fn pack_provenance(metadata: &[(String, String)]) -> Provenance {
    let driver = pick(metadata, &["driver"]).unwrap_or("").to_string();
    let venue = pick(metadata, &["venue", "location", "circuit", "track"])
        .unwrap_or("")
        .to_string();
    let vehicle = pick(metadata, &["vehicle", "car", "team", "vehicleid"])
        .unwrap_or("apex-14")
        .to_string();
    let event = pick(metadata, &["event", "gp", "official_event"])
        .unwrap_or("")
        .to_string();
    let session = pick(metadata, &["session"]).unwrap_or("").to_string();
    let source = pick(metadata, &["source", "exporter"]).unwrap_or("apex-14");
    let short_comment = format!("apex-14 export; src {source}");
    // Full provenance: every metadata pair, joined; truncation to 1024 bytes is
    // handled at write time.
    let comment = metadata
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("; ");
    Provenance {
        driver,
        vehicle,
        venue,
        event,
        session,
        short_comment,
        comment,
    }
}

// ---------------------------------------------------------------------------
// Low-level byte writers
// ---------------------------------------------------------------------------

fn put_u32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_u16(buf: &mut [u8], off: usize, v: u16) {
    buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
}
fn put_i16(buf: &mut [u8], off: usize, v: i16) {
    buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
}

/// Write a fixed-width, NUL-padded ASCII string. Non-ASCII bytes become `?`;
/// the string is truncated to `len`. The remaining bytes are left as the
/// buffer's existing zeros.
fn put_str(buf: &mut [u8], off: usize, len: usize, s: &str) {
    for (i, ch) in s.chars().take(len).enumerate() {
        buf[off + i] = if ch.is_ascii() { ch as u8 } else { b'?' };
    }
}

/// Write the 1762-byte `ldHead` at offset 0.
fn write_header(
    buf: &mut [u8],
    n_channs: u32,
    data_ptr0: u32,
    metadata: &[(String, String)],
    timestamp: &str,
) {
    let p = pack_provenance(metadata);
    let (date, time) = date_time_from_rfc3339(timestamp);

    put_u32(buf, 0, 0x40); // ldmarker
    put_u32(buf, 8, META_PTR); // chann_meta_ptr
    put_u32(buf, 12, data_ptr0); // chann_data_ptr
    put_u32(buf, 36, EVENT_PTR); // event_ptr
                                 // static/magic numbers (mirroring ldparser's frompd writer)
    put_u16(buf, 64, 1);
    put_u16(buf, 66, 0x4240);
    put_u16(buf, 68, 0x000f);
    put_u32(buf, 70, 0x1f44); // device serial
    put_str(buf, 74, 8, "ADL"); // device type
    put_u16(buf, 82, 420); // device version
    put_u16(buf, 84, 0xadb0);
    put_u32(buf, 86, n_channs);
    put_str(buf, 94, 16, &date);
    put_str(buf, 126, 16, &time);
    put_str(buf, 158, 64, &p.driver);
    put_str(buf, 222, 64, &p.vehicle);
    put_str(buf, 350, 64, &p.venue);
    put_u32(buf, 1502, 0xc81a4); // "pro logging" magic
    put_str(buf, 1572, 64, &p.short_comment);
}

/// Write the 1154-byte `ldEvent` at [`EVENT_PTR`]. `venue_ptr` is 0 (no chained
/// venue/vehicle sub-blocks).
fn write_event(buf: &mut [u8], metadata: &[(String, String)]) {
    let p = pack_provenance(metadata);
    let base = EVENT_PTR as usize;
    put_str(buf, base, 64, &p.event);
    put_str(buf, base + 64, 64, &p.session);
    put_str(buf, base + 128, 1024, &p.comment);
    put_u16(buf, base + 1152, 0); // venue_ptr = 0
}

/// Write one 124-byte `ldChan` meta block at `base` (float32 encoding).
#[allow(clippy::too_many_arguments)]
fn write_chan_meta(
    buf: &mut [u8],
    base: usize,
    prev: u32,
    next: u32,
    data_ptr: u32,
    data_len: u32,
    freq: u16,
    index: u16,
    name: &str,
    short: &str,
    unit: &str,
) {
    put_u32(buf, base, prev);
    put_u32(buf, base + 4, next);
    put_u32(buf, base + 8, data_ptr);
    put_u32(buf, base + 12, data_len);
    put_u16(buf, base + 16, 0x2ee1u16.wrapping_add(index)); // counter
    put_u16(buf, base + 18, 0x07); // dtype_a = float
    put_u16(buf, base + 20, 4); // dtype = 4-byte (f32)
    put_u16(buf, base + 22, freq);
    put_i16(buf, base + 24, 0); // shift
    put_i16(buf, base + 26, 1); // mul
    put_i16(buf, base + 28, 1); // scale
    put_i16(buf, base + 30, 0); // dec
    put_str(buf, base + 32, 32, name);
    put_str(buf, base + 64, 8, short);
    put_str(buf, base + 72, 12, unit);
}

/// Convert an RFC3339 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`) to i2's
/// `("DD/MM/YYYY", "HH:MM:SS")` header strings. Unparseable input falls back to
/// the Unix epoch so output stays deterministic and well-formed.
fn date_time_from_rfc3339(ts: &str) -> (String, String) {
    // Expect at least "YYYY-MM-DDTHH:MM:SS".
    let bytes = ts.as_bytes();
    let ok = ts.len() >= 19
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && (bytes[10] == b'T' || bytes[10] == b' ')
        && bytes[13] == b':'
        && bytes[16] == b':'
        && ts[0..4].chars().all(|c| c.is_ascii_digit());
    if ok {
        let y = &ts[0..4];
        let mo = &ts[5..7];
        let d = &ts[8..10];
        let hms = &ts[11..19];
        (format!("{d}/{mo}/{y}"), hms.to_string())
    } else {
        ("01/01/1970".to_string(), "00:00:00".to_string())
    }
}

// ---------------------------------------------------------------------------
// Minimal reader (verification / round-trip only)
// ---------------------------------------------------------------------------

/// One channel parsed back from a `.ld` file.
#[derive(Debug, Clone)]
pub struct LdChannel {
    /// Channel name.
    pub name: String,
    /// Unit string.
    pub unit: String,
    /// Sample rate (Hz).
    pub freq: u16,
    /// Decoded samples (`f32` widened to `f64`).
    pub data: Vec<f64>,
}

/// A `.ld` file parsed back into its header strings and channels.
#[derive(Debug, Clone)]
pub struct LdData {
    /// Header driver string.
    pub driver: String,
    /// Header vehicle-id string.
    pub vehicle: String,
    /// Header venue string.
    pub venue: String,
    /// Channels, in file (meta-link) order.
    pub channels: Vec<LdChannel>,
}

fn get_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}
fn get_u16(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(buf[off..off + 2].try_into().unwrap())
}
fn get_str(buf: &[u8], off: usize, len: usize) -> String {
    let raw = &buf[off..off + len];
    let end = raw.iter().position(|&b| b == 0).unwrap_or(len);
    String::from_utf8_lossy(&raw[..end]).trim().to_string()
}

/// Read back a `.ld` file written by [`export_ld`] (only the float32 subset we
/// write is supported). Used for round-trip verification.
pub fn read_ld(path: &Path) -> Result<LdData, MotecError> {
    let buf = std::fs::read(path)?;
    if buf.len() < LD_HEAD_SIZE {
        return Err(MotecError::Malformed("shorter than header"));
    }
    if get_u32(&buf, 0) != 0x40 {
        return Err(MotecError::Malformed("bad ldmarker"));
    }
    let mut meta_ptr = get_u32(&buf, 8);
    let driver = get_str(&buf, 158, 64);
    let vehicle = get_str(&buf, 222, 64);
    let venue = get_str(&buf, 350, 64);

    let mut channels = Vec::new();
    while meta_ptr != 0 {
        let base = meta_ptr as usize;
        if base + LD_CHAN_SIZE > buf.len() {
            return Err(MotecError::Malformed("channel meta out of range"));
        }
        let next = get_u32(&buf, base + 4);
        let data_ptr = get_u32(&buf, base + 8) as usize;
        let data_len = get_u32(&buf, base + 12) as usize;
        let dtype_a = get_u16(&buf, base + 18);
        let dtype = get_u16(&buf, base + 20);
        let freq = get_u16(&buf, base + 22);
        let name = get_str(&buf, base + 32, 32);
        let unit = get_str(&buf, base + 72, 12);
        if dtype_a != 0x07 || dtype != 4 {
            return Err(MotecError::Malformed("unsupported channel dtype"));
        }
        if data_ptr + data_len * 4 > buf.len() {
            return Err(MotecError::Malformed("channel data out of range"));
        }
        let mut data = Vec::with_capacity(data_len);
        for k in 0..data_len {
            let off = data_ptr + k * 4;
            let v = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
            data.push(v as f64);
        }
        channels.push(LdChannel {
            name,
            unit,
            freq,
            data,
        });
        meta_ptr = next;
    }

    Ok(LdData {
        driver,
        vehicle,
        venue,
        channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(name)
    }

    fn opts() -> LdOptions {
        LdOptions {
            sample_rate_hz: Some(10),
            timestamp: "2026-07-06T12:34:56Z".to_string(),
        }
    }

    #[test]
    fn block_sizes_match_ldparser() {
        assert_eq!(LD_HEAD_SIZE, 1762);
        assert_eq!(LD_EVENT_SIZE, 1154);
        assert_eq!(LD_CHAN_SIZE, 124);
        assert_eq!(META_PTR, 2916);
    }

    #[test]
    fn t_grid_round_trips() {
        // 0..1s at 10 Hz already-uniform t; speed ramps 0..100.
        let t: Vec<f64> = (0..11).map(|k| k as f64 * 0.1).collect();
        let speed: Vec<f64> = (0..11).map(|k| k as f64 * 10.0).collect();
        let cols: Vec<(ChannelId, &[f64])> =
            vec![(ChannelId::Time, &t[..]), (ChannelId::Speed, &speed[..])];
        let meta = vec![
            ("driver".to_string(), "RUS".to_string()),
            ("location".to_string(), "Silverstone".to_string()),
        ];
        let path = temp("apex_ld_tgrid.ld");
        let report = export_ld(&path, Grid::Time, &cols, &meta, &opts()).unwrap();
        assert_eq!(report.sample_rate_hz, 10);
        assert_eq!(report.n_samples, 11);
        // t is consumed as the axis (not emitted); speed + gap_fill remain.
        assert_eq!(report.channels.len(), 2);

        let back = read_ld(&path).unwrap();
        assert_eq!(back.driver, "RUS");
        assert_eq!(back.venue, "Silverstone");
        let sp = back
            .channels
            .iter()
            .find(|c| c.name == "speed")
            .expect("speed channel");
        assert_eq!(sp.unit, "m/s");
        assert_eq!(sp.freq, 10);
        assert_eq!(sp.data.len(), 11);
        for (got, want) in sp.data.iter().zip(&speed) {
            assert!((got - want).abs() < 1e-3, "{got} vs {want}");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn s_grid_resamples_via_t() {
        // Uniform in s (0,10,..,50) but t is nonlinear (car accelerates), so the
        // .ld output must be uniform in t, not s.
        let s: Vec<f64> = (0..6).map(|k| k as f64 * 10.0).collect();
        let t = [0.0, 0.2, 0.35, 0.45, 0.52, 0.58];
        let speed = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let cols: Vec<(ChannelId, &[f64])> = vec![
            (ChannelId::S, &s[..]),
            (ChannelId::Time, &t[..]),
            (ChannelId::Speed, &speed[..]),
        ];
        let path = temp("apex_ld_sgrid.ld");
        let report = export_ld(&path, Grid::Distance, &cols, &[], &opts()).unwrap();
        // 0..0.58 s at 10 Hz → floor(0.58*10)+1 = 6 samples.
        assert_eq!(report.n_samples, 6);
        // s is emitted (kept), t is the axis (dropped); + gap_fill.
        let names: Vec<&str> = report.channels.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"s"));
        assert!(names.contains(&"speed"));
        assert!(names.contains(&"gap_fill"));
        assert!(!names.contains(&"t"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn s_grid_without_t_is_refused() {
        let s = [0.0, 10.0, 20.0];
        let speed = [10.0, 20.0, 30.0];
        let cols: Vec<(ChannelId, &[f64])> =
            vec![(ChannelId::S, &s[..]), (ChannelId::Speed, &speed[..])];
        let path = temp("apex_ld_no_t.ld");
        let err = export_ld(&path, Grid::Distance, &cols, &[], &opts()).unwrap_err();
        assert!(matches!(err, MotecError::NoTimeAxis(Grid::Distance)));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn nan_is_hold_last_filled_with_marker() {
        let t: Vec<f64> = (0..11).map(|k| k as f64 * 0.1).collect();
        // A NaN gap at the source samples around index 5.
        let mut speed: Vec<f64> = (0..11).map(|k| k as f64 * 10.0).collect();
        speed[5] = f64::NAN;
        let cols: Vec<(ChannelId, &[f64])> =
            vec![(ChannelId::Time, &t[..]), (ChannelId::Speed, &speed[..])];
        let path = temp("apex_ld_nan.ld");
        let report = export_ld(&path, Grid::Time, &cols, &[], &opts()).unwrap();
        assert!(report.gap_filled_samples > 0);

        let back = read_ld(&path).unwrap();
        let sp = back.channels.iter().find(|c| c.name == "speed").unwrap();
        let gap = back.channels.iter().find(|c| c.name == "gap_fill").unwrap();
        // No NaN survives into the file; every value is finite.
        assert!(sp.data.iter().all(|v| v.is_finite()));
        // The marker flags at least one sample.
        assert!(gap.data.iter().any(|&v| v > 0.5));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn leading_nan_becomes_zero() {
        let t: Vec<f64> = (0..5).map(|k| k as f64 * 0.1).collect();
        let speed = [f64::NAN, f64::NAN, 30.0, 40.0, 50.0];
        let cols: Vec<(ChannelId, &[f64])> =
            vec![(ChannelId::Time, &t[..]), (ChannelId::Speed, &speed[..])];
        let path = temp("apex_ld_lead_nan.ld");
        export_ld(&path, Grid::Time, &cols, &[], &opts()).unwrap();
        let back = read_ld(&path).unwrap();
        let sp = back.channels.iter().find(|c| c.name == "speed").unwrap();
        assert_eq!(sp.data[0], 0.0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn non_monotonic_time_is_rejected() {
        let t = [0.0, 0.1, 0.1, 0.2]; // not strictly increasing
        let speed = [1.0, 2.0, 3.0, 4.0];
        let cols: Vec<(ChannelId, &[f64])> =
            vec![(ChannelId::Time, &t[..]), (ChannelId::Speed, &speed[..])];
        let path = temp("apex_ld_bad_t.ld");
        let err = export_ld(&path, Grid::Time, &cols, &[], &opts()).unwrap_err();
        assert!(matches!(err, MotecError::NonMonotonicTime));
    }

    #[test]
    fn i2_units_are_ascii() {
        for &id in crate::channels::CHANNELS {
            let u = i2_unit(id.unit());
            assert!(u.is_ascii(), "unit for {id:?} is not ascii: {u}");
            assert!(u.len() <= 12);
        }
        assert_eq!(i2_unit(Unit::Celsius), "C");
        assert_eq!(i2_unit(Unit::G), "G");
    }

    #[test]
    fn date_time_parse() {
        assert_eq!(
            date_time_from_rfc3339("2026-07-06T12:34:56Z"),
            ("06/07/2026".to_string(), "12:34:56".to_string())
        );
        assert_eq!(
            date_time_from_rfc3339("garbage"),
            ("01/01/1970".to_string(), "00:00:00".to_string())
        );
    }

    #[test]
    fn provenance_maps_fields() {
        let meta = vec![
            ("driver".to_string(), "RUS".to_string()),
            ("session".to_string(), "Q".to_string()),
            ("event".to_string(), "British Grand Prix".to_string()),
            ("location".to_string(), "Silverstone".to_string()),
        ];
        let p = pack_provenance(&meta);
        assert_eq!(p.driver, "RUS");
        assert_eq!(p.session, "Q");
        assert_eq!(p.event, "British Grand Prix");
        assert_eq!(p.venue, "Silverstone");
        assert!(p.comment.contains("driver=RUS"));
        assert!(p.comment.contains("location=Silverstone"));
    }
}
