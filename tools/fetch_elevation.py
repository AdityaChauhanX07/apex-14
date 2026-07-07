#!/usr/bin/env python3
"""Fetch and smooth a track elevation profile z(s) from a DEM, and write a
gitignored elevation sidecar the Rust importer merges into a v2 3D track file.

Pipeline (network stays here; Rust never calls the network):

  tracks/<c>.json (2D centerline)  +  tracks/<c>.georef.json (local->WGS84)
        │  map each centerline (x,y) -> (lat,lon)
        ▼
  DEM query  (OpenTopoData primary, Open-Elevation fallback)
        │  batch <=100 pts/req, <=1 req/s, cached per (lat,lon) -> zero re-run cost
        ▼
  raw z(s)  ──periodic regularized smoothing (deviation budget)──▶  smooth z(s)
        │
        ▼
  tracks/<c>.elevation.json   (z per centerline point + roughness stats)
  tracks/<c>.elevation.svg    (profile plot)

DEM note: the task named Copernicus **GLO-30**; OpenTopoData's free tier does not
serve GLO-30 (`cop30` is absent). We use **EU-DEM 25m** (`eudem25m`) — also a
Copernicus Land Monitoring Service product, finer 25 m posting, Europe-only which
covers both circuits — with `srtm30m` then Open-Elevation as fallbacks. Banking
is left 0 (a 25-30 m DEM cannot resolve camber across a ~14 m track width; a
per-corner manual override is the future mechanism).

Usage:
    python tools/fetch_elevation.py spa
    python tools/fetch_elevation.py silverstone
"""

import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

CACHE_DIR = os.path.join("tracks", ".elevation_cache")
OTD = "https://api.opentopodata.org/v1"
PRIMARY = "eudem25m"      # Copernicus EU-DEM 25 m (GLO-30 unavailable on OTD free tier)
FALLBACK_OTD = "srtm30m"  # global 30 m
R_EARTH = 6378137.0

# Public grade anchors for a sanity gate (max |dz/ds|).
GRADE_ANCHOR = {"spa": 0.18, "silverstone": 0.05}


def log(*a):
    print("elevation:", *a, file=sys.stderr)


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# georeference: local (x, y) -> WGS84
# --------------------------------------------------------------------------- #
def make_mapper(g):
    scale, th, tx, ty = g["scale"], g["theta_rad"], g["tx"], g["ty"]
    reflect, lat0, lon0 = g["reflect"], g["lat0"], g["lon0"]
    c, s = math.cos(th), math.sin(th)

    def to_wgs84(x, y):
        xr = -x if reflect else x
        ex = scale * (c * xr - s * y) + tx
        ey = scale * (s * xr + c * y) + ty
        lat = lat0 + math.degrees(ey / R_EARTH)
        lon = lon0 + math.degrees(ex / (R_EARTH * math.cos(math.radians(lat0))))
        return lat, lon

    return to_wgs84


# --------------------------------------------------------------------------- #
# DEM query with an on-disk (lat,lon) cache
# --------------------------------------------------------------------------- #
def _cache_path(dataset):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{dataset}.json")


def _key(lat, lon):
    return f"{lat:.6f},{lon:.6f}"


def _http_json(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "apex-14-elevation/1.0 (research)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def query_dem(latlon, stats):
    """Elevations for a list of (lat,lon), cached. Returns list aligned to input."""
    cache = _cache_path(PRIMARY)
    store = load_json(cache) if os.path.exists(cache) else {}
    out = [None] * len(latlon)
    missing = []
    for i, (la, lo) in enumerate(latlon):
        k = _key(la, lo)
        if k in store:
            out[i] = store[k]
            stats["cache_hits"] += 1
        else:
            missing.append(i)

    # Batch the misses: <=100 locations/request, <=1 req/s.
    B = 100
    for j in range(0, len(missing), B):
        idxs = missing[j:j + B]
        locs = "|".join(f"{latlon[i][0]:.6f},{latlon[i][1]:.6f}" for i in idxs)
        elev = _query_batch(locs, len(idxs), stats)
        for i, e in zip(idxs, elev):
            out[i] = e
            store[_key(*latlon[i])] = e
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(store, f)
        if j + B < len(missing):
            time.sleep(1.05)
    return out


def _query_batch(locs, n, stats):
    """One batch against OpenTopoData primary, then fallbacks. Returns n elevs."""
    for ds in (PRIMARY, FALLBACK_OTD):
        try:
            url = f"{OTD}/{ds}?locations={urllib.parse.quote(locs)}"
            d = _http_json(url)
            if d.get("status") == "OK":
                stats["requests"] += 1
                stats["by_dataset"][ds] = stats["by_dataset"].get(ds, 0) + n
                vals = [r["elevation"] for r in d["results"]]
                if all(v is not None for v in vals):
                    return vals
        except Exception as e:  # noqa: BLE001
            log(f"  {ds} batch failed: {e}")
        time.sleep(1.05)
    # Last resort: Open-Elevation.
    try:
        url = "https://api.open-elevation.com/api/v1/lookup?locations=" + urllib.parse.quote(locs)
        d = _http_json(url)
        stats["requests"] += 1
        stats["by_dataset"]["open-elevation"] = stats["by_dataset"].get("open-elevation", 0) + n
        return [r["elevation"] for r in d["results"]]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"all DEM providers failed for a batch: {e}")


# --------------------------------------------------------------------------- #
# periodic regularized smoothing of z(s)  (deviation-budget, curvature penalty)
# --------------------------------------------------------------------------- #
def smooth_periodic(s, z_raw, closed, budget_m=2.5):
    """Minimize Σ(z-z_raw)² + λ Σ(z''(s))², periodic if closed. λ auto-picked so
    the max |z-z_raw| deviation is ~budget_m (the DEM vertical noise scale)."""
    n = len(z_raw)
    z_raw = np.asarray(z_raw, float)
    # Second-difference operator D (periodic wrap if closed).
    D = np.zeros((n, n))
    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            continue
        D[i, (i - 1) % n] += 1.0
        D[i, i] += -2.0
        D[i, (i + 1) % n] += 1.0
    DtD = D.T @ D
    ident = np.eye(n)

    def solve(lam):
        return np.linalg.solve(ident + lam * DtD, z_raw)

    # Bisect λ so max deviation ≈ budget.
    lo, hi = 1e-1, 1e9
    z = solve(hi)
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        z = solve(mid)
        dev = float(np.max(np.abs(z - z_raw)))
        if dev > budget_m:
            hi = mid
        else:
            lo = mid
    return z, float(lo * hi) ** 0.5


def roughness(s, z, closed):
    """grade dz/ds via central differences; p95 |grade| and max |grade|."""
    n = len(z)
    g = np.zeros(n)
    for i in range(n):
        if closed:
            im, ip = (i - 1) % n, (i + 1) % n
            ds = (s[ip] - s[im])
            if ds <= 0:
                ds += s[-1] - s[0]
        else:
            im, ip = max(0, i - 1), min(n - 1, i + 1)
            ds = s[ip] - s[im]
        g[i] = (z[ip] - z[im]) / ds if ds > 0 else 0.0
    ag = np.abs(g)
    return {"p95_grade": float(np.percentile(ag, 95)), "max_grade": float(np.max(ag))}


def arc_length(pts, closed):
    s = [0.0]
    for a, b in zip(pts, pts[1:]):
        s.append(s[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    return s


# --------------------------------------------------------------------------- #
# SVG elevation profile
# --------------------------------------------------------------------------- #
def write_svg(path, s, z, name, features):
    W, H, ml, mr, mt, mb = 1200, 340, 60, 20, 30, 40
    pw, ph = W - ml - mr, H - mt - mb
    smax = s[-1]
    zmin, zmax = min(z), max(z)
    zr = max(1e-6, zmax - zmin)
    sx = lambda v: ml + v / smax * pw
    sy = lambda v: mt + ph - (v - zmin) / zr * ph
    pts = " ".join(f"{sx(si):.1f},{sy(zi):.1f}" for si, zi in zip(s, z))
    out = [f'<?xml version="1.0" encoding="UTF-8"?>',
           f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           f'<rect width="{W}" height="{H}" fill="#0d0d16"/>',
           f'<text x="{ml}" y="20" fill="#ddd" font-family="sans-serif" font-size="16">{name} — elevation z(s), range {zr:.0f} m</text>',
           f'<polyline fill="none" stroke="#4fc3f7" stroke-width="2" points="{pts}"/>']
    for st, lbl in features:
        x = sx(st)
        out.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" stroke="#ff5577" stroke-width="1" stroke-dasharray="4 4" opacity="0.6"/>')
        out.append(f'<text x="{x+3:.1f}" y="{mt+12}" fill="#ff99aa" font-family="sans-serif" font-size="11">{lbl}</text>')
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<text x="{sx(frac*smax):.1f}" y="{H-12}" fill="#888" font-family="sans-serif" font-size="11" text-anchor="middle">{frac*smax:.0f}</text>')
    out.append(f'<text x="{ml-6}" y="{sy(zmax):.1f}" fill="#888" font-size="11" text-anchor="end">{zmax:.0f}</text>')
    out.append(f'<text x="{ml-6}" y="{sy(zmin):.1f}" fill="#888" font-size="11" text-anchor="end">{zmin:.0f}</text>')
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


FEATURES = {
    "spa": [(1140, "Raidillon top"), (2400, "Les Combes"), (4400, "Pouhon"), (6000, "Blanchimont")],
    "silverstone": [(1500, "Village"), (3500, "Maggotts"), (4200, "Stowe")],
}


def main():
    if len(sys.argv) < 2:
        print("usage: python tools/fetch_elevation.py <spa|silverstone>", file=sys.stderr)
        return 2
    name = sys.argv[1]
    track = load_json(f"tracks/{name}.json")
    g = load_json(f"tracks/{name}.georef.json")
    pts = [(p["x"], p["y"]) for p in track["points"]]
    closed = track.get("closed", True)
    s = arc_length(pts, closed)
    log(f"{name}: {len(pts)} points, 2D length {s[-1]:.1f} m, closed={closed}")

    to_wgs84 = make_mapper(g)
    latlon = [to_wgs84(x, y) for x, y in pts]

    stats = {"requests": 0, "cache_hits": 0, "by_dataset": {}}
    z_raw = query_dem(latlon, stats)
    log(f"DEM: {stats['requests']} requests, {stats['cache_hits']} cache hits, {stats['by_dataset']}")

    rough_raw = roughness(s, np.asarray(z_raw), closed)
    z_sm, lam = smooth_periodic(s, z_raw, closed)
    rough_sm = roughness(s, z_sm, closed)
    log(f"roughness raw:    p95 grade {rough_raw['p95_grade']*100:.1f}%  max {rough_raw['max_grade']*100:.1f}%")
    log(f"roughness smooth: p95 grade {rough_sm['p95_grade']*100:.1f}%  max {rough_sm['max_grade']*100:.1f}%  (lambda={lam:.1f})")
    zr = float(np.max(z_sm) - np.min(z_sm))
    log(f"elevation range: {zr:.1f} m   [{float(np.min(z_sm)):.1f} .. {float(np.max(z_sm)):.1f}]")

    anchor = GRADE_ANCHOR.get(name)
    if anchor and rough_sm["max_grade"] > anchor * 2.0:
        log(f"WARNING: smoothed max grade {rough_sm['max_grade']*100:.1f}% >> anchor ~{anchor*100:.0f}% — investigate before shipping")

    # Deterministic timestamp (APEX_REPRO_TIMESTAMP where present).
    ts = os.environ.get("APEX_REPRO_TIMESTAMP", "unset")
    sidecar = {
        "circuit": name,
        "dem_dataset": PRIMARY,
        "dem_note": "Copernicus EU-DEM 25m via OpenTopoData (GLO-30 unavailable on free tier)",
        "georef_rms_m": g.get("residual_rms_m"),
        "n_points": len(pts),
        "length_2d_m": round(s[-1], 3),
        "z": [round(float(v), 3) for v in z_sm],
        "elevation_range_m": round(zr, 2),
        "roughness_raw": rough_raw,
        "roughness_smoothed": rough_sm,
        "smoothing_lambda": lam,
        "banking_deg": 0.0,
        "banking_note": "DEM cannot resolve camber across track width; per-corner override is future work",
        "generated_at": ts,
    }
    out = f"tracks/{name}.elevation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    log(f"wrote {out}")
    write_svg(f"tracks/{name}.elevation.svg", s, z_sm, track.get("name", name), FEATURES.get(name, []))
    log(f"wrote tracks/{name}.elevation.svg")

    print(json.dumps({
        "circuit": name, "dem": PRIMARY,
        "requests": stats["requests"], "cache_hits": stats["cache_hits"],
        "elevation_range_m": round(zr, 1),
        "max_grade_pct": round(rough_sm["max_grade"] * 100, 1),
        "p95_grade_pct": round(rough_sm["p95_grade"] * 100, 1),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
