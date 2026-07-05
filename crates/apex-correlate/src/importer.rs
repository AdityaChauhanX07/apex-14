//! The generic telemetry importer.

use std::collections::BTreeMap;
use std::path::Path;

use apex_telemetry::ChannelId;

use crate::error::CorrelateError;
use crate::mapping::{Mapping, UnknownColumns};
use crate::telemetry::{GridKind, Telemetry};
use crate::units::conversion_factor;

/// One resolved source column: where it is in the record, which channel it
/// feeds, and the source→registry unit factor.
struct ColumnPlan {
    idx: usize,
    channel: ChannelId,
    factor: f64,
}

/// Import a telemetry CSV into a [`Telemetry`], resolving source columns through
/// `mapping` and converting each to its registry unit.
///
/// The file is the standard Apex telemetry CSV (see `docs/telemetry_format.md`):
/// `# key: value` comment header (with reserved `# grid:` and `# columns:`
/// lines), then a header row of column names, then data rows. For a file
/// already in registry names/units, pass [`Mapping::identity`].
///
/// Non-finite measured values (empty fields, `NaN`, `inf`) are kept as-is; only
/// a genuinely unparseable field is an error.
pub fn import_telemetry(
    path: impl AsRef<Path>,
    mapping: &Mapping,
) -> Result<Telemetry, CorrelateError> {
    let text = std::fs::read_to_string(path)?;

    // --- pass 1: comment header (metadata + reserved grid line) ---
    let mut metadata: Vec<(String, String)> = Vec::new();
    let mut grid_decl: Option<GridKind> = None;
    for raw in text.lines() {
        let line = raw.trim_start();
        let Some(body) = line.strip_prefix('#') else {
            continue;
        };
        let body = body.trim();
        if body.is_empty() {
            continue;
        }
        let Some((key, val)) = body.split_once(':') else {
            continue;
        };
        let (key, val) = (key.trim(), val.trim());
        match key {
            "grid" => {
                grid_decl = Some(
                    GridKind::parse(val).ok_or_else(|| CorrelateError::BadGrid(val.to_string()))?,
                );
            }
            // The units line is informational (units come from the registry /
            // mapping); skip it as a metadata pair.
            "columns" => {}
            _ => metadata.push((key.to_string(), val.to_string())),
        }
    }

    // --- pass 2: CSV body (comment-aware) ---
    let mut rdr = csv::ReaderBuilder::new()
        .comment(Some(b'#'))
        .trim(csv::Trim::All)
        .from_reader(text.as_bytes());
    let header = rdr.headers()?.clone();

    // Resolve each source column to a plan (or skip / error).
    let mut plan: Vec<ColumnPlan> = Vec::new();
    for (idx, name) in header.iter().enumerate() {
        let resolved: Option<(ChannelId, f64)> = if let Some(cm) = mapping.columns.get(name) {
            let channel = ChannelId::from_name(&cm.channel)
                .ok_or_else(|| CorrelateError::UnknownChannel(cm.channel.clone()))?;
            let factor = conversion_factor(&cm.unit, channel.unit()).ok_or_else(|| {
                CorrelateError::UnknownSourceUnit {
                    column: name.to_string(),
                    unit: cm.unit.clone(),
                }
            })?;
            Some((channel, factor))
        } else if mapping.auto_identity {
            ChannelId::from_name(name).map(|c| (c, 1.0))
        } else {
            None
        };

        let (channel, factor) = match resolved {
            Some(x) => x,
            None => match mapping.unknown_columns {
                UnknownColumns::Ignore => continue,
                UnknownColumns::Error => {
                    return Err(CorrelateError::UnknownSourceColumn(name.to_string()))
                }
            },
        };

        if plan.iter().any(|p| p.channel == channel) {
            return Err(CorrelateError::DuplicateTarget(channel.name()));
        }
        plan.push(ColumnPlan {
            idx,
            channel,
            factor,
        });
    }

    // Read data rows into per-column vectors.
    let mut data: Vec<Vec<f64>> = vec![Vec::new(); plan.len()];
    for (row, rec) in rdr.records().enumerate() {
        let rec = rec?;
        for (ci, p) in plan.iter().enumerate() {
            let raw = rec.get(p.idx).unwrap_or("").trim();
            let value = parse_measured(raw).ok_or_else(|| CorrelateError::MalformedValue {
                channel: p.channel.name(),
                row,
                value: raw.to_string(),
            })?;
            data[ci].push(value * p.factor);
        }
    }

    let mut channels: BTreeMap<ChannelId, Vec<f64>> = BTreeMap::new();
    for (ci, p) in plan.iter().enumerate() {
        channels.insert(p.channel, std::mem::take(&mut data[ci]));
    }

    // Grid: explicit mapping override, then file declaration, then inference.
    let grid = if let Some(g) = mapping.grid.as_deref() {
        GridKind::parse(g).ok_or_else(|| CorrelateError::BadGrid(g.to_string()))?
    } else if let Some(g) = grid_decl {
        g
    } else if channels.contains_key(&ChannelId::S) {
        GridKind::S
    } else if channels.contains_key(&ChannelId::Time) {
        GridKind::T
    } else {
        return Err(CorrelateError::MissingGrid);
    };
    if !channels.contains_key(&grid.axis_channel()) {
        return Err(CorrelateError::MissingGrid);
    }

    // Required channels.
    let mut absent = Vec::new();
    for r in &mapping.required {
        let ch =
            ChannelId::from_name(r).ok_or_else(|| CorrelateError::UnknownChannel(r.clone()))?;
        if !channels.contains_key(&ch) {
            absent.push(r.clone());
        }
    }
    if !absent.is_empty() {
        return Err(CorrelateError::MissingRequired(absent));
    }

    Ok(Telemetry {
        grid,
        channels,
        metadata,
    })
}

/// Parse one measured field. Empty → `NaN` (a real gap). Otherwise parse as
/// `f64` (which also accepts `NaN`/`inf`); a non-numeric field is `None`
/// (→ a `MalformedValue` error).
fn parse_measured(raw: &str) -> Option<f64> {
    if raw.is_empty() {
        return Some(f64::NAN);
    }
    raw.parse::<f64>().ok()
}
