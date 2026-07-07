#!/usr/bin/env python3
"""Georeference a TUMFTM-frame track centerline to WGS84 by shape-matching to
OpenStreetMap.

The TUMFTM racetrack-database CSVs (and our derived `tracks/<circuit>.json`) are
in a **local metric frame with an arbitrary origin** — the file header is only
`x_m,y_m,w_tr_right_m,w_tr_left_m`, there is no georeference. To sample an
elevation DEM we need `local -> WGS84`.

Method (correspondence-free): fetch the circuit's `highway=raceway` ways from the
Overpass API (excluding kart/pit/moto ways by name), project their lat/lon nodes
to a local ENU metric frame, then fit a 2D **similarity** (scale, rotation,
translation, optional reflection) from our centerline onto that OSM point cloud
with a global rotation search + trimmed ICP refinement. The residual RMS
(centerline-to-OSM nearest neighbour) is the reported positional accuracy.

Network stays here (Python/tools); the Rust side never calls the network. The
output is a gitignored `tracks/<circuit>.georef.json` sidecar with the transform
and quality metrics. Overpass responses are cached under a gitignored dir so
re-runs cost zero API calls.

Usage:
    python tools/georef.py spa
    python tools/georef.py silverstone
"""

import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

R_EARTH = 6378137.0
CACHE_DIR = os.path.join("tracks", ".georef_cache")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Per-circuit config: OSM bbox (S,W,N,E), a regex of way names to EXCLUDE from
# the racing surface, our centerline json, and public anchor landmarks
# (approximate lat/lon) for a sanity spot-check.
CIRCUITS = {
    "spa": {
        "bbox": (50.425, 5.955, 50.445, 5.985),
        "exclude": ("kart", "moto", "pit", "support", "campus"),
        "track_json": "tracks/spa.json",
        # Recognizable corners to spot-check, taken from OSM's own named ways
        # (authoritative positions of the famous points).
        "anchor_ways": ["Eau Rouge", "Raidillon", "Les Combes", "Malmedy", "Blanchimont"],
    },
    "silverstone": {
        "bbox": (52.060, -1.040, 52.090, -1.005),
        # The Silverstone bbox contains several overlapping layouts (GP, Stowe
        # Circuit, National) all tagged highway=raceway. Whitelist the current
        # Grand Prix "Arena" layout by its named corners/straights; drop anything
        # containing "circuit" (the separate Stowe Circuit ways).
        "include_names": [
            "Abbey", "Farm Curve", "Village", "The Loop", "Aintree", "Wellington",
            "Brooklands", "Luffield", "Woodcote", "Copse", "Maggotts", "Becketts",
            "Chapel", "Hangar", "Stowe", "Vale", "Club", "Hamilton", "National Pit",
        ],
        "exclude": ("circuit",),
        "track_json": "tracks/silverstone.json",
        "anchor_ways": ["Copse", "Maggotts", "Becketts", "Stowe", "Club"],
    },
}


def log(*a):
    print("georef:", *a, file=sys.stderr)


# --------------------------------------------------------------------------- #
# Overpass fetch (cached)
# --------------------------------------------------------------------------- #
def fetch_raceways(name, bbox):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{name}_raceways.json")
    if os.path.exists(cache):
        log(f"cache hit {cache}")
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    s, w, n, e = bbox
    query = f'[out:json][timeout:60];way["highway"="raceway"]({s},{w},{n},{e});out geom;'
    log(f"fetching Overpass raceways for {name} ...")
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        OVERPASS_URL,
        data=body,
        headers={"User-Agent": "apex-14-georef/1.0 (research; contact via repo)"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(data, f)
    time.sleep(1.0)  # be a good citizen
    return data


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def enu(lat, lon, lat0, lon0):
    """Equirectangular projection to local ENU metres about (lat0, lon0)."""
    x = math.radians(lon - lon0) * math.cos(math.radians(lat0)) * R_EARTH
    y = math.radians(lat - lat0) * R_EARTH
    return x, y


def inv_enu(x, y, lat0, lon0):
    lat = lat0 + math.degrees(y / R_EARTH)
    lon = lon0 + math.degrees(x / (R_EARTH * math.cos(math.radians(lat0))))
    return lat, lon


def load_centerline(path):
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    pts = [(p["x"], p["y"]) for p in j["points"]]
    return pts, j.get("closed", True), j.get("name", "track")


def _way_kept(tags, exclude, include):
    """Whether an OSM way belongs to the target racing surface."""
    nm = (tags.get("name") or "").lower()
    if any(x in nm for x in exclude):
        return False
    if include is not None:
        return any(nm.startswith(w.lower()) for w in include)
    return True


def osm_cloud(data, exclude, include=None):
    """Collect lat/lon nodes of the target racing surface."""
    pts = []
    for w in data.get("elements", []):
        if not _way_kept(w.get("tags", {}), exclude, include):
            continue
        for g in w.get("geometry", []):
            pts.append((g["lat"], g["lon"]))
    return pts


def osm_named_way_centroids(data, lat0, lon0):
    """ENU centroid of each named racing way (authoritative corner anchors)."""
    out = {}
    for w in data.get("elements", []):
        nm = w.get("tags", {}).get("name")
        g = w.get("geometry", [])
        if not nm or not g:
            continue
        e = [enu(p["lat"], p["lon"], lat0, lon0) for p in g]
        cx = sum(p[0] for p in e) / len(e)
        cy = sum(p[1] for p in e) / len(e)
        out[nm] = (cx, cy)
    return out


def osm_segments(data, exclude, lat0, lon0, include=None):
    """Polyline segments (ENU endpoint pairs) of the target racing surface."""
    segs = []
    for w in data.get("elements", []):
        if not _way_kept(w.get("tags", {}), exclude, include):
            continue
        g = w.get("geometry", [])
        e = [enu(p["lat"], p["lon"], lat0, lon0) for p in g]
        for a, b in zip(e, e[1:]):
            segs.append((a, b))
    return segs


def nearest_on_segments(px, py, segs):
    """Nearest point on the segment set to (px, py); returns (qx, qy, dist2)."""
    best = (px, py, 1e18)
    for (ax, ay), (bx, by) in segs:
        ex, ey = bx - ax, by - ay
        ll = ex * ex + ey * ey
        t = 0.0 if ll <= 0 else ((px - ax) * ex + (py - ay) * ey) / ll
        t = max(0.0, min(1.0, t))
        qx, qy = ax + t * ex, ay + t * ey
        d = (px - qx) ** 2 + (py - qy) ** 2
        if d < best[2]:
            best = (qx, qy, d)
    return best


def centroid(pts):
    n = len(pts)
    return sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n


def nn_rms(src, dst_grid, trim=1.0):
    """Trimmed RMS of nearest-neighbour distances src -> dst (brute force)."""
    ds = []
    for sx, sy in src:
        best = 1e18
        for dx, dy in dst_grid:
            d = (sx - dx) ** 2 + (sy - dy) ** 2
            if d < best:
                best = d
        ds.append(best)
    ds.sort()
    k = max(1, int(len(ds) * trim))
    kept = ds[:k]
    return math.sqrt(sum(kept) / len(kept))


def apply_sim(pts, scale, theta, tx, ty, reflect):
    c, s = math.cos(theta), math.sin(theta)
    out = []
    for x, y in pts:
        xr = -x if reflect else x
        out.append((scale * (c * xr - s * y) + tx, scale * (s * xr + c * y) + ty))
    return out


def umeyama(src, dst):
    """Least-squares similarity (scale, R, t) mapping src -> dst (2D)."""
    n = len(src)
    msx = sum(p[0] for p in src) / n
    msy = sum(p[1] for p in src) / n
    mdx = sum(p[0] for p in dst) / n
    mdy = sum(p[1] for p in dst) / n
    sxx = sxy = syx = syy = var = 0.0
    for (sx, sy), (dx, dy) in zip(src, dst):
        ax, ay = sx - msx, sy - msy
        bx, by = dx - mdx, dy - mdy
        sxx += bx * ax
        sxy += bx * ay
        syx += by * ax
        syy += by * ay
        var += ax * ax + ay * ay
    # Optimal rotation from the 2x2 cross-covariance [[sxx,sxy],[syx,syy]].
    theta = math.atan2(syx - sxy, sxx + syy)
    c, s = math.cos(theta), math.sin(theta)
    scale = (c * (sxx + syy) + s * (syx - sxy)) / var
    tx = mdx - scale * (c * msx - s * msy)
    ty = mdy - scale * (s * msx + c * msy)
    return scale, theta, tx, ty


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def seg_rms(moved, segs, trim=1.0):
    """Trimmed RMS of point-to-polyline distances."""
    ds = [nearest_on_segments(px, py, segs)[2] for px, py in moved]
    ds.sort()
    k = max(1, int(len(ds) * trim))
    return math.sqrt(sum(ds[:k]) / k)


def fit(centerline, osm_nodes, segs):
    """Global rotation search (vs node cloud) + point-to-segment trimmed ICP."""
    cc = centroid(centerline)
    oc = centroid(osm_nodes)
    src0 = [(x - cc[0], y - cc[1]) for x, y in centerline]

    # Coarse: rotation search against the node cloud (cheap NN), 1-degree steps.
    best = None
    sub = src0[::4]
    for reflect in (False, True):
        for deg in range(0, 360):
            th = math.radians(deg)
            moved = apply_sim(sub, 1.0, th, oc[0], oc[1], reflect)
            r = nn_rms(moved, osm_nodes, trim=0.7)
            if best is None or r < best[0]:
                best = (r, th, reflect)
    _, theta, reflect = best
    scale = 1.0
    c, s = math.cos(theta), math.sin(theta)
    px = -cc[0] if reflect else cc[0]
    tx2 = oc[0] - scale * (c * px - s * cc[1])
    ty2 = oc[1] - scale * (s * px + c * cc[1])

    # Refine: point-to-SEGMENT trimmed ICP (robust to OSM node-density on straights).
    for _ in range(25):
        moved = apply_sim(centerline, scale, theta, tx2, ty2, reflect)
        corr, res = [], []
        for mx, my in moved:
            qx, qy, d = nearest_on_segments(mx, my, segs)
            corr.append((qx, qy))
            res.append(d)
        thresh = sorted(res)[max(1, int(len(res) * 0.85)) - 1]
        idx = [i for i in range(len(res)) if res[i] <= thresh]
        pre = [(-centerline[i][0], centerline[i][1]) if reflect else centerline[i] for i in idx]
        dst_k = [corr[i] for i in idx]
        scale, theta, tx2, ty2 = umeyama(pre, dst_k)

    moved = apply_sim(centerline, scale, theta, tx2, ty2, reflect)
    # Coverage-aware accuracy: RMS over centerline points that actually have an
    # OSM reference within `gate` metres (the rest are OSM coverage gaps on
    # unnamed connector ways, not fit error).
    gate = 30.0
    covered = [nearest_on_segments(px, py, segs)[2] for px, py in moved]
    within = [d for d in covered if d <= gate * gate]
    coverage_frac = len(within) / len(covered)
    coverage_rms = math.sqrt(sum(within) / len(within)) if within else float("nan")
    return {
        "scale": scale,
        "theta": theta,
        "tx": tx2,
        "ty": ty2,
        "reflect": reflect,
        "rms": seg_rms(moved, segs, 1.0),
        "trimmed_rms": seg_rms(moved, segs, 0.9),
        "coverage_rms": coverage_rms,
        "coverage_frac": coverage_frac,
    }


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CIRCUITS:
        print("usage: python tools/georef.py <spa|silverstone>", file=sys.stderr)
        return 2
    name = sys.argv[1]
    cfg = CIRCUITS[name]

    centerline, closed, tname = load_centerline(cfg["track_json"])
    log(f"{tname}: {len(centerline)} centerline points, closed={closed}")

    include = cfg.get("include_names")
    data = fetch_raceways(name, cfg["bbox"])
    cloud_ll = osm_cloud(data, cfg["exclude"], include)
    log(f"OSM racing-surface nodes (filtered): {len(cloud_ll)}")
    if len(cloud_ll) < 20:
        log("too few OSM nodes; aborting")
        return 1

    lat0, lon0 = centroid(cloud_ll)
    osm_e = [enu(la, lo, lat0, lon0) for la, lo in cloud_ll]
    segs = osm_segments(data, cfg["exclude"], lat0, lon0, include)
    log(f"OSM racing-surface segments: {len(segs)}")

    t = fit(centerline, osm_e, segs)
    t["lat0"] = lat0
    t["lon0"] = lon0
    log(
        f"fit: scale={t['scale']:.5f} theta={math.degrees(t['theta']):.2f}deg "
        f"reflect={t['reflect']} rms={t['rms']:.2f}m trimmed_rms={t['trimmed_rms']:.2f}m "
        f"coverage_rms={t['coverage_rms']:.2f}m ({t['coverage_frac']*100:.0f}% within 30m)"
    )

    # --- WGS84 mapper for a local (x, y) ---
    def local_to_wgs84(x, y):
        mx, my = apply_sim([(x, y)], t["scale"], t["theta"], t["tx"], t["ty"], t["reflect"])[0]
        return inv_enu(mx, my, lat0, lon0)

    # --- anchor spot-checks: distance from the transformed centerline to each
    #     recognizable OSM-named corner (authoritative reference position) ---
    moved = apply_sim(centerline, t["scale"], t["theta"], t["tx"], t["ty"], t["reflect"])
    named = osm_named_way_centroids(data, lat0, lon0)
    anchors_out = {}
    for aname in cfg["anchor_ways"]:
        # match the first OSM way whose name starts with the anchor label
        target = next((named[k] for k in named if k.lower().startswith(aname.lower())), None)
        if target is None:
            log(f"anchor {aname}: no OSM named way found")
            continue
        ax, ay = target
        best_d = min(math.hypot(mx - ax, my - ay) for mx, my in moved)
        # also report the arc-length station of the nearest point (for the profile)
        best_i = min(range(len(moved)), key=lambda i: math.hypot(moved[i][0] - ax, moved[i][1] - ay))
        anchors_out[aname] = {"offset_m": round(best_d, 1), "station_frac": round(best_i / len(moved), 3)}
        log(f"anchor {aname}: centerline passes within {best_d:.1f} m (station ~{best_i/len(moved)*100:.0f}%)")

    sidecar = {
        "circuit": name,
        "method": "osm-raceway-shape-match (Overpass highway=raceway) + trimmed ICP similarity",
        "note": "local(TUMFTM) -> ENU(lat0,lon0) via similarity, then ENU -> WGS84 (equirectangular)",
        "lat0": lat0,
        "lon0": lon0,
        "scale": t["scale"],
        "theta_rad": t["theta"],
        "tx": t["tx"],
        "ty": t["ty"],
        "reflect": t["reflect"],
        "residual_rms_m": t["rms"],
        "residual_trimmed_rms_m": t["trimmed_rms"],
        "coverage_rms_m": t["coverage_rms"],
        "coverage_frac": t["coverage_frac"],
        "osm_nodes": len(cloud_ll),
        "anchor_offsets_m": anchors_out,
    }
    out = f"tracks/{name}.georef.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    log(f"wrote {out}")
    print(json.dumps({"circuit": name, "rms_m": round(t["rms"], 2),
                      "trimmed_rms_m": round(t["trimmed_rms"], 2),
                      "scale": round(t["scale"], 5), "reflect": t["reflect"],
                      "anchors": anchors_out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
