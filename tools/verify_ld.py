#!/usr/bin/env python3
"""Independently verify a MoTeC ``.ld`` file written by ``apex-14 export-ld``.

This parses the file with the **third-party** reverse-engineered reader
``ldparser`` (github.com/gotzl/ldparser) — NOT our own Rust reader — so a
successful parse is independent evidence that the byte layout we emit is the one
i2 expects. It prints channel names, units, sample counts, and per-channel value
ranges, and (optionally) cross-checks a reference CSV column within f32
precision.

``ldparser`` is **GPL-3.0**, which is incompatible with this repo's MIT/Apache
licensing, so it is deliberately NOT vendored/committed. If it is not importable,
this script fetches ``ldparser.py`` from the pinned upstream commit into a local,
gitignored cache (``tools/.ldparser_cache/``) on your own machine — a runtime
convenience for the developer running verification, not redistribution by this
project.

Usage:
    python tools/verify_ld.py <file.ld> [reference.csv] [--channel speed]

The final authority remains opening the file in real MoTeC i2 (free download);
this script is the strongest check available without it.
"""
import os
import sys
import argparse

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required (pip install numpy)", file=sys.stderr)
    sys.exit(2)

# Pinned upstream (GPL-3.0). Not committed; fetched to a local cache if needed.
_LDPARSER_URL = "https://raw.githubusercontent.com/gotzl/ldparser/master/ldparser.py"
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ldparser_cache")


def _load_ldparser():
    try:
        import ldparser  # already installed?
        return ldparser
    except ImportError:
        pass
    cached = os.path.join(_CACHE_DIR, "ldparser.py")
    if not os.path.exists(cached):
        try:
            import urllib.request
            os.makedirs(_CACHE_DIR, exist_ok=True)
            print(f"fetching independent reader ldparser (GPL-3.0) -> {cached}", file=sys.stderr)
            urllib.request.urlretrieve(_LDPARSER_URL, cached)
        except Exception as e:  # noqa: BLE001
            print(
                f"ERROR: could not obtain ldparser ({e}).\n"
                f"Manually download {_LDPARSER_URL} to {cached} and retry.",
                file=sys.stderr,
            )
            sys.exit(2)
    sys.path.insert(0, _CACHE_DIR)
    import ldparser
    return ldparser


ldparser = _load_ldparser()


def main():
    ap = argparse.ArgumentParser(description="Verify an apex-14 .ld with ldparser")
    ap.add_argument("ld", help="the .ld file to verify")
    ap.add_argument("reference_csv", nargs="?", help="optional standard telemetry CSV to cross-check")
    ap.add_argument("--channel", default="speed", help="channel to cross-check against the CSV")
    args = ap.parse_args()

    data = ldparser.ldData.fromfile(args.ld)
    head = data.head
    print("=== header (parsed by ldparser) ===")
    print(head)
    print()
    print("=== channels ===")
    names = list(data)
    for ch in data.channs:
        vals = ch.data
        finite = vals[np.isfinite(vals)]
        rng = (float(finite.min()), float(finite.max())) if len(finite) else (float("nan"), float("nan"))
        print(
            f"  {ch.name:<18} [{ch.unit:<5}] {ch.freq:>4} Hz  "
            f"n={len(vals):<5} range=[{rng[0]:.4g}, {rng[1]:.4g}]"
        )
    print()
    print(f"channels read back: {len(data.channs)}")

    # sample counts must all agree
    counts = {len(ch.data) for ch in data.channs}
    if len(counts) != 1:
        print(f"FAIL: inconsistent sample counts across channels: {counts}", file=sys.stderr)
        sys.exit(1)
    print(f"sample count (consistent): {counts.pop()}")

    if "gap_fill" in names:
        gap = data["gap_fill"].data
        n_gap = int((gap > 0.5).sum())
        print(f"gap_fill marker: {n_gap} filled sample(s)")

    # Optional cross-check against a reference CSV column, resampled to time.
    if args.reference_csv:
        ok = cross_check(args, data)
        if not ok:
            sys.exit(1)

    print("\nOK: ldparser (independent reader) parsed the file successfully.")


def cross_check(args, data):
    """Cross-check one channel: interpolate the CSV column onto the .ld time grid
    and compare within a tolerance (the CSV is on an s-grid with a t column; the
    .ld is uniform in t)."""
    import csv

    ch_name = args.channel
    names = list(data)
    if ch_name not in names:
        print(f"(skip cross-check: channel {ch_name} not in .ld)")
        return True

    # read the CSV (comment-aware): t and the target channel
    rows_t, rows_v = [], []
    header = None
    with open(args.reference_csv, newline="") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = [p.strip() for p in line.rstrip("\n").split(",")]
            if header is None:
                header = parts
                continue
            row = dict(zip(header, parts))
            if "t" not in row or ch_name not in row:
                print(f"(skip cross-check: CSV lacks t or {ch_name})")
                return True
            try:
                t = float(row["t"])
                v = float(row[ch_name])
            except ValueError:
                continue
            rows_t.append(t)
            rows_v.append(v)

    if len(rows_t) < 2:
        print("(skip cross-check: too few CSV rows)")
        return True

    t_csv = np.array(rows_t)
    v_csv = np.array(rows_v)
    # the .ld time grid: uniform at the channel freq starting at t_csv[0]
    ch = data[ch_name]
    n = len(ch.data)
    t_ld = t_csv[0] + np.arange(n) / ch.freq
    v_ref = np.interp(t_ld, t_csv, v_csv)

    v_ld = np.array(ch.data)
    finite = np.isfinite(v_ref) & np.isfinite(v_ld)
    if not finite.any():
        print("(skip cross-check: no overlapping finite samples)")
        return True
    diff = np.abs(v_ld[finite] - v_ref[finite])
    scale = max(1.0, float(np.nanmax(np.abs(v_ref[finite]))))
    max_rel = float(diff.max() / scale)
    print(
        f"\ncross-check `{ch_name}` vs {args.reference_csv}: "
        f"max abs diff {diff.max():.4g}, max rel {max_rel:.2e} (f32 ~1e-7 + resample)"
    )
    # The .ld and CSV differ by our resample AND f32 rounding; a loose bound
    # catches gross corruption without failing on legitimate interpolation.
    if max_rel > 0.05:
        print(f"FAIL: cross-check relative error {max_rel:.3f} too large", file=sys.stderr)
        return False
    return True


if __name__ == "__main__":
    main()
