//! Writer for the standard Apex telemetry CSV format.

use std::io::Write as _;
use std::path::Path;

use apex_telemetry::channels::csv_columns_comment;
use apex_telemetry::ChannelId;

use crate::error::CorrelateError;
use crate::telemetry::Telemetry;

/// Write `telem` to `path` in the standard Apex telemetry CSV format
/// (see `docs/telemetry_format.md`):
///
/// ```text
/// # <metadata key>: <value>      (from telem.metadata, in order)
/// # grid: s|t
/// # columns: name[unit], ...     (registry units, via csv_columns_comment)
/// #
/// <header row of channel names>
/// <data rows>
/// ```
///
/// Columns are ordered with the grid axis first, then the remaining channels in
/// registry declaration order. Values use the shortest round-tripping `f64`
/// representation; non-finite samples are written as `NaN` (kept as gaps).
///
/// This is **measured** data: no `RunMetadata` provenance block is written (see
/// the format doc's "measured data vs. sim artifacts").
pub fn write_telemetry_csv(
    path: impl AsRef<Path>,
    telem: &Telemetry,
) -> Result<(), CorrelateError> {
    // Column order: axis first, then the rest in ChannelId (registry) order.
    let axis = telem.grid.axis_channel();
    let mut ordered: Vec<ChannelId> = Vec::with_capacity(telem.channels.len());
    if telem.channels.contains_key(&axis) {
        ordered.push(axis);
    }
    for &id in telem.channels.keys() {
        if id != axis {
            ordered.push(id);
        }
    }
    let names: Vec<&str> = ordered.iter().map(|c| c.name()).collect();

    let mut file = std::fs::File::create(path)?;
    // Metadata comment lines (descriptive provenance only).
    for (k, v) in &telem.metadata {
        writeln!(file, "# {k}: {v}")?;
    }
    // Reserved grid + columns lines, then the blank separator.
    writeln!(file, "# grid: {}", telem.grid.as_str())?;
    file.write_all(csv_columns_comment(&names).as_bytes())?;
    file.write_all(b"\n#\n")?;

    let mut wtr = csv::Writer::from_writer(file);
    wtr.write_record(&names)?;

    let rows = telem.len();
    let cols: Vec<&[f64]> = ordered
        .iter()
        .map(|id| telem.channel(*id).unwrap_or(&[]))
        .collect();
    for r in 0..rows {
        let record: Vec<String> = cols.iter().map(|c| fmt_val(c[r])).collect();
        wtr.write_record(&record)?;
    }
    wtr.flush()?;
    Ok(())
}

/// Shortest round-tripping representation; non-finite → `NaN`/`inf`/`-inf`
/// (Rust's `Display` already emits these, and `f64::parse` reads them back).
fn fmt_val(v: f64) -> String {
    format!("{v}")
}
