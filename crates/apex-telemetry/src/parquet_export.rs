//! Apache Parquet output for telemetry (feature `parquet`, default-on).
//!
//! One Parquet file, one `f64` column per channel, column names = channel
//! registry names (or explicit strings for multi-source files). `NaN` is a
//! normal `f64` value and is preserved natively (no null masking). Provenance
//! and per-column units ride in the **file-level key/value metadata** (the
//! Parquet footer):
//!
//! - `units:<column>` = the unit symbol (empty for dimensionless), one key per
//!   column — greppable and self-describing;
//! - `grid` = `s` | `t` (the sample axis), when known;
//! - every descriptive `# key: value` provenance pair verbatim;
//! - a full [`RunMetadata`](crate::RunMetadata) block (`config_hash`, …,
//!   `timestamp`) under `run.*` keys when the source is a sim artifact.
//!
//! This module is only compiled with the `parquet` feature; `--no-default-
//! features` builds omit it and the arrow/parquet dependency entirely.

use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

use crate::channels::ChannelId;
use crate::RunMetadata;

/// One Parquet column: a name, a unit symbol, and the `f64` samples.
pub struct ParquetColumn<'a> {
    /// Column name (a registry channel name, or an explicit label such as
    /// `meas_speed`).
    pub name: &'a str,
    /// Unit symbol for the `units:<name>` metadata key (may be empty).
    pub unit: &'a str,
    /// Samples (`NaN` preserved).
    pub data: &'a [f64],
}

/// Anything that can go wrong writing/reading Parquet.
#[derive(Debug)]
pub enum ParquetError {
    /// Filesystem I/O failure.
    Io(std::io::Error),
    /// Underlying arrow/parquet failure.
    Parquet(parquet::errors::ParquetError),
    /// Columns had mismatched lengths.
    RaggedColumns,
}

impl std::fmt::Display for ParquetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParquetError::Io(e) => write!(f, "I/O error: {e}"),
            ParquetError::Parquet(e) => write!(f, "parquet error: {e}"),
            ParquetError::RaggedColumns => write!(f, "columns have mismatched lengths"),
        }
    }
}

impl std::error::Error for ParquetError {}

impl From<std::io::Error> for ParquetError {
    fn from(e: std::io::Error) -> Self {
        ParquetError::Io(e)
    }
}
impl From<parquet::errors::ParquetError> for ParquetError {
    fn from(e: parquet::errors::ParquetError) -> Self {
        ParquetError::Parquet(e)
    }
}
impl From<arrow_schema::ArrowError> for ParquetError {
    fn from(e: arrow_schema::ArrowError) -> Self {
        ParquetError::Parquet(parquet::errors::ParquetError::ArrowError(e.to_string()))
    }
}

/// Write `columns` to a Parquet file at `path`, with `kv` extra file-level
/// key/value metadata. A `units:<name>` key is added automatically for every
/// column. All columns must have the same length.
pub fn write_parquet(
    path: &Path,
    columns: &[ParquetColumn],
    kv: &[(String, String)],
) -> Result<(), ParquetError> {
    let rows = columns.first().map(|c| c.data.len()).unwrap_or(0);
    if columns.iter().any(|c| c.data.len() != rows) {
        return Err(ParquetError::RaggedColumns);
    }

    let fields: Vec<Field> = columns
        .iter()
        .map(|c| Field::new(c.name, DataType::Float64, false))
        .collect();
    let schema = Arc::new(Schema::new(fields));

    let arrays: Vec<ArrayRef> = columns
        .iter()
        .map(|c| Arc::new(Float64Array::from(c.data.to_vec())) as ArrayRef)
        .collect();
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;

    // File-level key/value metadata: units per column, then caller-supplied.
    let mut meta: Vec<KeyValue> = Vec::new();
    for c in columns {
        meta.push(KeyValue::new(
            format!("units:{}", c.name),
            c.unit.to_string(),
        ));
    }
    for (k, v) in kv {
        meta.push(KeyValue::new(k.clone(), v.clone()));
    }

    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(meta))
        .build();

    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Descriptive + sim provenance packed into Parquet file metadata.
///
/// `grid` (if `Some`) becomes the `grid` key; `metadata` pairs are copied
/// verbatim; a `RunMetadata`, if present, is written under `run.*` keys.
fn provenance_kv(
    grid: Option<&str>,
    metadata: &[(String, String)],
    run: Option<&RunMetadata>,
) -> Vec<(String, String)> {
    let mut kv: Vec<(String, String)> = Vec::new();
    if let Some(g) = grid {
        kv.push(("grid".to_string(), g.to_string()));
    }
    for (k, v) in metadata {
        kv.push((k.clone(), v.clone()));
    }
    if let Some(m) = run {
        kv.push(("run.config_hash".into(), m.config_hash.to_hex()));
        kv.push(("run.car_hash".into(), m.car_hash.to_hex()));
        kv.push(("run.track_hash".into(), m.track_hash.to_hex()));
        kv.push(("run.settings_hash".into(), m.settings_hash.to_hex()));
        kv.push(("run.git_sha".into(), m.git_sha.clone()));
        kv.push(("run.apex_version".into(), m.apex_version.clone()));
        kv.push((
            "run.seed".into(),
            m.seed
                .map(|s| s.to_string())
                .unwrap_or_else(|| "none".into()),
        ));
        kv.push(("run.timestamp".into(), m.timestamp.clone()));
    }
    kv
}

/// Convenience: write registry `columns` (name + unit from the registry) to a
/// Parquet file, packing `grid`, descriptive `metadata`, and an optional
/// `RunMetadata` into the footer. Used by the straight CSV→Parquet converter.
pub fn export_channels_parquet(
    path: &Path,
    columns: &[(ChannelId, &[f64])],
    grid: Option<&str>,
    metadata: &[(String, String)],
    run: Option<&RunMetadata>,
) -> Result<(), ParquetError> {
    let pcols: Vec<ParquetColumn> = columns
        .iter()
        .map(|(id, data)| ParquetColumn {
            name: id.name(),
            unit: id.unit().symbol(),
            data,
        })
        .collect();
    let kv = provenance_kv(grid, metadata, run);
    write_parquet(path, &pcols, &kv)
}

// ---------------------------------------------------------------------------
// Reader (verification / round-trip only)
// ---------------------------------------------------------------------------

/// A Parquet file read back: columns (name → `f64` data) and file KV metadata.
#[derive(Debug, Clone)]
pub struct ParquetData {
    /// `(column_name, samples)` in file order.
    pub columns: Vec<(String, Vec<f64>)>,
    /// File-level key/value metadata.
    pub metadata: Vec<(String, String)>,
}

impl ParquetData {
    /// Look up a metadata value by key.
    pub fn meta(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Look up a column's samples by name.
    pub fn column(&self, name: &str) -> Option<&[f64]> {
        self.columns
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.as_slice())
    }
}

/// Read a Parquet file written by this module back into memory (all columns are
/// expected to be `Float64`). Used for round-trip verification.
pub fn read_parquet(path: &Path) -> Result<ParquetData, ParquetError> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

    // File-level KV metadata from the footer.
    let mut metadata = Vec::new();
    if let Some(kvs) = builder.metadata().file_metadata().key_value_metadata() {
        for kv in kvs {
            metadata.push((kv.key.clone(), kv.value.clone().unwrap_or_default()));
        }
    }

    let schema = builder.schema().clone();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let mut columns: Vec<(String, Vec<f64>)> =
        names.iter().map(|n| (n.clone(), Vec::new())).collect();

    let reader = builder.build()?;
    for batch in reader {
        let batch = batch?;
        for (ci, col) in columns.iter_mut().enumerate() {
            let arr = batch
                .column(ci)
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or(ParquetError::Parquet(
                    parquet::errors::ParquetError::General("expected Float64 column".to_string()),
                ))?;
            col.1.extend(arr.values().iter().copied());
        }
    }

    Ok(ParquetData { columns, metadata })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings_hash_for_mode;

    fn temp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(name)
    }

    #[test]
    fn channels_round_trip_with_units_and_nan() {
        let s = [0.0, 10.0, 20.0, 30.0];
        let speed = [72.3, f64::NAN, 74.1, 75.0]; // NaN preserved natively
        let cols: Vec<(ChannelId, &[f64])> =
            vec![(ChannelId::S, &s[..]), (ChannelId::Speed, &speed[..])];
        let meta = vec![
            ("driver".to_string(), "RUS".to_string()),
            ("year".to_string(), "2024".to_string()),
        ];
        let path = temp("apex_pq_channels.parquet");
        export_channels_parquet(&path, &cols, Some("s"), &meta, None).unwrap();

        let back = read_parquet(&path).unwrap();
        // Columns + values (NaN preserved).
        let sp = back.column("speed").unwrap();
        assert_eq!(sp.len(), 4);
        assert!(sp[1].is_nan(), "NaN must survive round-trip");
        assert_eq!(sp[0], 72.3);
        // Units metadata.
        assert_eq!(back.meta("units:speed"), Some("m/s"));
        assert_eq!(back.meta("units:s"), Some("m"));
        // Grid + descriptive provenance.
        assert_eq!(back.meta("grid"), Some("s"));
        assert_eq!(back.meta("driver"), Some("RUS"));
        assert_eq!(back.meta("year"), Some("2024"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn run_metadata_is_embedded() {
        let s = [0.0, 1.0];
        let cols: Vec<(ChannelId, &[f64])> = vec![(ChannelId::S, &s[..])];
        let run = RunMetadata::new(
            settings_hash_for_mode("car"),
            settings_hash_for_mode("track"),
            settings_hash_for_mode("settings"),
            Some(7),
        );
        let path = temp("apex_pq_run.parquet");
        export_channels_parquet(&path, &cols, Some("s"), &[], Some(&run)).unwrap();
        let back = read_parquet(&path).unwrap();
        assert_eq!(
            back.meta("run.config_hash"),
            Some(run.config_hash.to_hex().as_str())
        );
        assert_eq!(back.meta("run.seed"), Some("7"));
        assert_eq!(
            back.meta("run.apex_version"),
            Some(run.apex_version.as_str())
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn explicit_prefixed_columns() {
        // The correlate use case: mixed measured/sim columns, explicit names.
        let s = [0.0, 10.0];
        let meas = [70.0, 71.0];
        let sim = [72.0, 73.0];
        let cols = vec![
            ParquetColumn {
                name: "s",
                unit: "m",
                data: &s,
            },
            ParquetColumn {
                name: "meas_speed",
                unit: "m/s",
                data: &meas,
            },
            ParquetColumn {
                name: "sim_speed",
                unit: "m/s",
                data: &sim,
            },
        ];
        let path = temp("apex_pq_prefixed.parquet");
        write_parquet(&path, &cols, &[("grid".into(), "s".into())]).unwrap();
        let back = read_parquet(&path).unwrap();
        assert_eq!(back.column("meas_speed").unwrap(), &[70.0, 71.0]);
        assert_eq!(back.column("sim_speed").unwrap(), &[72.0, 73.0]);
        assert_eq!(back.meta("units:sim_speed"), Some("m/s"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn ragged_columns_rejected() {
        let a = [0.0, 1.0];
        let b = [0.0];
        let cols = vec![
            ParquetColumn {
                name: "a",
                unit: "",
                data: &a,
            },
            ParquetColumn {
                name: "b",
                unit: "",
                data: &b,
            },
        ];
        let path = temp("apex_pq_ragged.parquet");
        assert!(matches!(
            write_parquet(&path, &cols, &[]),
            Err(ParquetError::RaggedColumns)
        ));
    }
}
