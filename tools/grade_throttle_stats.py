#!/usr/bin/env python3
"""Per-sample grade-binned throttle/brake statistics from real telemetry.

Reconstructs the measured-throttle diagnostic cited in
`docs/validation/track3d.md` (the Spa 3D-QSS falsification finding): does the
real driver manage descents (partial throttle, deliberate braking) rather
than free-wheeling them the way a point-mass QSS does?

Method: load an aligned telemetry CSV (`s[m], throttle[], brake[], ...`,
produced by `tools/fastf1_export.py` + `apex-cli telemetry-align`) and a
schema-v2 3D track JSON (`z` per point, `apex-cli import-track --elevation`
output). For each *track* point compute the local grade `dz/ds` by central
difference over the native point spacing (mirrors
`apex_track::Ribbon3d::from_centerline_3d`'s grade computation — same
discretization, not a hand-picked window). Interpolate that per-point grade
onto each telemetry sample's own projected station `s`, bin samples into
descent (`grade < -threshold`), climb (`grade > +threshold`), and flat
(between), and report per bin: sample count, throttle p50/mean, the
full-throttle fraction (`throttle > 0.95`), and the any-brake fraction
(`brake > 0`).

This is PER-SAMPLE grade binning, not contiguous station windows: a sample at
a given `s` is classified by the track's local grade there, regardless of
which named "climb" or "descent" region of the lap it happens to sit in —
this was the actual prior methodology and is materially different from
binning by fixed station ranges (e.g. "s in [4400, 4900)"), which mixes
climb/flat/descent sub-segments together and is far more sensitive to the
exact boundary chosen.

Determinism: pure function of the two input files; no RNG, no network, no
wall-clock. Only reads local (gitignored) telemetry/track data — never
committed, matching the rest of this pipeline's local-data discipline.

Usage:
    python tools/grade_throttle_stats.py telemetry/spa_2024_Q_aligned.csv tracks/spa_3d.json
    python tools/grade_throttle_stats.py telemetry/silverstone_2024_Q_aligned.csv tracks/silverstone_3d.json --threshold 0.003
"""

import argparse
import csv
import io
import json
import sys


def log(*a):
    print("grade_throttle_stats:", *a, file=sys.stderr)


def load_track_points(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    pts = d["points"]
    closed = d.get("closed", True)
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    zs = [p.get("z", 0.0) for p in pts]
    return xs, ys, zs, closed


def track_stations_and_grade(xs, ys, zs, closed):
    """Cumulative planar arc length `s` at each track point (matching the 2D
    station axis telemetry is aligned to), and the local grade `dz/ds` at
    each point via central difference over the native point spacing —
    the same discretization `Ribbon3d::from_centerline_3d` uses for `grade`,
    just computed here directly from the track JSON rather than via the Rust
    type, since this is a standalone Python diagnostic.
    """
    n = len(xs)
    s = [0.0] * n
    for i in range(1, n):
        s[i] = s[i - 1] + ((xs[i] - xs[i - 1]) ** 2 + (ys[i] - ys[i - 1]) ** 2) ** 0.5
    total_length = s[-1]
    if closed:
        total_length += ((xs[0] - xs[-1]) ** 2 + (ys[0] - ys[-1]) ** 2) ** 0.5

    def neighbors(i):
        if closed:
            return (i - 1) % n, (i + 1) % n
        if i == 0:
            return 0, 1
        if i == n - 1:
            return n - 2, n - 1
        return i - 1, i + 1

    def ds_between(i, j):
        d = s[j] - s[i]
        if closed:
            if d > total_length * 0.5:
                d -= total_length
            elif d < -total_length * 0.5:
                d += total_length
        return d

    grade = [0.0] * n
    for i in range(n):
        lo, hi = neighbors(i)
        dds = ds_between(lo, hi)
        if dds != 0.0:
            grade[i] = (zs[hi] - zs[lo]) / dds

    return s, total_length, grade


def interp_periodic(station, values, total_length, s_query):
    """Linear interpolation of `values` (sampled at ascending `station`,
    wrapping over `total_length`) at `s_query`."""
    n = len(station)
    sq = s_query % total_length if total_length > 0 else 0.0
    # binary search for the bracketing index
    lo, hi = 0, n - 1
    if sq <= station[0] or sq >= station[-1]:
        # wrap segment: station[-1] -> station[0] + total_length
        s0, s1 = station[-1], station[0] + total_length
        v0, v1 = values[-1], values[0]
        span = s1 - s0
        sq_eff = sq if sq >= station[0] else sq + total_length
        t = 0.0 if span <= 0 else (sq_eff - s0) / span
        return v0 + t * (v1 - v0)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if station[mid] <= sq:
            lo = mid
        else:
            hi = mid
    span = station[hi] - station[lo]
    t = 0.0 if span <= 0 else (sq - station[lo]) / span
    return values[lo] + t * (values[hi] - values[lo])


def load_telemetry(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line)
    r = csv.DictReader(io.StringIO("".join(rows)))
    out = []
    for row in r:
        out.append(
            {
                "s": float(row["s"]),
                "throttle": float(row["throttle"]),
                "brake": float(row["brake"]),
            }
        )
    return out


def percentile(xs, p):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    idx = min(n - 1, int(round(p * (n - 1))))
    return xs[idx]


def bin_stats(samples):
    n = len(samples)
    if n == 0:
        return {"n": 0}
    throttle = [s["throttle"] for s in samples]
    return {
        "n": n,
        "throttle_p50": round(percentile(throttle, 0.5), 3),
        "throttle_mean": round(sum(throttle) / n, 3),
        "full_throttle_frac": round(sum(1 for t in throttle if t > 0.95) / n, 3),
        "any_brake_frac": round(sum(1 for s in samples if s["brake"] > 0) / n, 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("telemetry_csv")
    ap.add_argument("track_json")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="grade magnitude (fraction, e.g. 0.02 = 2%%) separating climb/descent from flat",
    )
    args = ap.parse_args()

    xs, ys, zs, closed = load_track_points(args.track_json)
    station, total_length, grade = track_stations_and_grade(xs, ys, zs, closed)
    telemetry = load_telemetry(args.telemetry_csv)

    bins = {"climb": [], "descent": [], "flat": []}
    for row in telemetry:
        g = interp_periodic(station, grade, total_length, row["s"])
        if g > args.threshold:
            bins["climb"].append(row)
        elif g < -args.threshold:
            bins["descent"].append(row)
        else:
            bins["flat"].append(row)

    result = {
        "telemetry_csv": args.telemetry_csv,
        "track_json": args.track_json,
        "threshold": args.threshold,
        "total_samples": len(telemetry),
        "bins": {k: bin_stats(v) for k, v in bins.items()},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
