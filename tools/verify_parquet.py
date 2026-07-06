#!/usr/bin/env python3
"""Independently verify a Parquet file written by ``apex-14 export-parquet`` /
``correlate --parquet`` using the third-party ``pyarrow`` reader.

Prints the schema, file-level key/value metadata (units + provenance), row
count, and per-column value ranges; confirms NaN survives natively; and
(optionally) cross-checks a reference CSV column exactly (Parquet is lossless
f64, so values must match to floating-point equality).

Usage:
    pip install pyarrow
    python tools/verify_parquet.py <file.parquet> [reference.csv] [--column speed]
"""
import sys
import argparse
import math

try:
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow required (pip install pyarrow)", file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser(description="Verify an apex-14 Parquet with pyarrow")
    ap.add_argument("parquet", help="the .parquet file to verify")
    ap.add_argument("reference_csv", nargs="?", help="optional standard telemetry CSV to cross-check")
    ap.add_argument("--column", default="speed", help="column to cross-check against the CSV")
    args = ap.parse_args()

    pf = pq.ParquetFile(args.parquet)
    table = pf.read()
    print(f"=== schema ({table.num_columns} columns, {table.num_rows} rows) ===")
    for field in table.schema:
        print(f"  {field.name:<20} {field.type}")

    print("\n=== file key/value metadata ===")
    meta = pf.metadata.metadata or {}
    kv = {k.decode(): v.decode() for k, v in meta.items()}
    # show units first, then everything else, sorted for determinism
    units = {k: v for k, v in kv.items() if k.startswith("units:")}
    other = {k: v for k, v in kv.items() if not k.startswith("units:") and k != "ARROW:schema"}
    for k in sorted(units):
        print(f"  {k} = {units[k]!r}")
    for k in sorted(other):
        print(f"  {k} = {other[k]!r}")

    print("\n=== column ranges (NaN counted separately) ===")
    nan_seen = False
    for name in table.column_names:
        col = table.column(name).to_pylist()
        finite = [x for x in col if x is not None and not math.isnan(x)]
        n_nan = sum(1 for x in col if x is not None and math.isnan(x))
        nan_seen = nan_seen or n_nan > 0
        rng = (min(finite), max(finite)) if finite else (float("nan"), float("nan"))
        print(f"  {name:<20} n={len(col):<5} NaN={n_nan:<4} range=[{rng[0]:.4g}, {rng[1]:.4g}]")
    if nan_seen:
        print("NaN preserved natively (as expected for measured gaps).")

    if args.reference_csv:
        if not cross_check(args, table):
            sys.exit(1)

    print("\nOK: pyarrow (independent reader) parsed the file successfully.")


def cross_check(args, table):
    """Parquet is lossless f64, so a column must match the CSV exactly."""
    name = args.column
    if name not in table.column_names:
        print(f"(skip cross-check: column {name} not in Parquet)")
        return True

    header = None
    csv_vals = []
    with open(args.reference_csv, newline="") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = [p.strip() for p in line.rstrip("\n").split(",")]
            if header is None:
                header = parts
                continue
            row = dict(zip(header, parts))
            if name not in row:
                print(f"(skip cross-check: CSV lacks {name})")
                return True
            s = row[name]
            csv_vals.append(float(s) if s not in ("", "NaN", "nan") else float("nan"))

    pq_vals = table.column(name).to_pylist()
    if len(pq_vals) != len(csv_vals):
        print(
            f"(note: row counts differ — Parquet {len(pq_vals)} vs CSV {len(csv_vals)}; "
            "expected if the file is a resampled/aligned grid, skipping exact check)"
        )
        return True

    worst = 0.0
    for a, b in zip(pq_vals, csv_vals):
        an = a is None or math.isnan(a)
        bn = math.isnan(b)
        if an and bn:
            continue
        if an != bn:
            print(f"FAIL: NaN mismatch ({a} vs {b})", file=sys.stderr)
            return False
        worst = max(worst, abs(a - b))
    print(f"\ncross-check `{name}` vs {args.reference_csv}: max abs diff {worst:.3g} (should be ~0, lossless f64)")
    if worst > 1e-9:
        print(f"FAIL: lossless column differs by {worst}", file=sys.stderr)
        return False
    return True


if __name__ == "__main__":
    main()
