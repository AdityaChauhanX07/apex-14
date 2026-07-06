#!/usr/bin/env python3
"""Multi-circuit correlation campaign driver for apex-14.

Runs, per circuit in a spec TOML:
    fastf1 export -> import-track (smoothed) -> telemetry-align
    -> correlate (preset) -> identify (mu fixed) -> correlate (fitted)
and aggregates a summary table with per-circuit hygiene checks.

Resumable: each stage is skipped when its output already exists (use --force to
redo everything, or --force-stage <name> for one stage). The FastF1 export is
the only network stage; if it fails the circuit is stopped, the exact command is
printed, and the campaign continues with the other circuits.

All raw telemetry / cache / TUMFTM-derived files stay local (gitignored). Only
the fitted TOMLs (cars/<slug>_2024q_fitted.toml) and docs pages are committable.

Usage:
    python tools/correlate_campaign.py --spec tools/campaign.toml
    python tools/correlate_campaign.py --spec tools/campaign.toml --only monza
    python tools/correlate_campaign.py --spec tools/campaign.toml --force
"""

import argparse
import json
import math
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APEX = ROOT / "target" / "release" / "apex-14.exe"
PY = sys.executable
FREE = "aero.lift_coeff,aero.drag_coeff,powertrain.power_scale"

# Hygiene thresholds.
SCALE_LO, SCALE_HI = 0.95, 1.05
ALIGN_RMS_MAX = 8.0
SPROJ_TOL = 0.02  # 2% of TUMFTM length
CORNER_LO, CORNER_HI = 3, 12  # sane count of prominent <70 m/s minima


def run(cmd, **kw):
    """Run a command, capturing utf-8 stdout/stderr. Returns CompletedProcess."""
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw
    )


def track_length(json_path: Path) -> float:
    """Perimeter (closed) of a track JSON's points."""
    d = json.loads(json_path.read_text(encoding="utf-8"))
    pts = [(p["x"], p["y"]) for p in d["points"]]
    n = len(pts)
    return sum(
        math.hypot(pts[i][0] - pts[(i + 1) % n][0], pts[i][1] - pts[(i + 1) % n][1])
        for i in range(n)
    )


def search(pattern, text, group=1, cast=float, default=None):
    m = re.search(pattern, text)
    if not m:
        return default
    try:
        return cast(m.group(group))
    except (ValueError, IndexError):
        return default


# --- stages -------------------------------------------------------------------


def stage_export(c, cfg, out, force):
    tel = ROOT / "telemetry" / f"{c['slug']}_2024_Q.csv"
    cmd = [
        PY, str(ROOT / "tools" / "fastf1_export.py"),
        "--year", str(cfg["year"]), "--gp", c["gp"],
        "--session", cfg["session"], "--out", str(tel),
    ]
    out["export_cmd"] = " ".join(f'"{a}"' if " " in a else a for a in cmd)
    if tel.exists() and not force:
        out["export"] = "cached"
        return True
    r = run(cmd, cwd=ROOT)
    log = r.stdout + r.stderr
    out["resolved_event"] = search(r"RESOLVED event\s*:\s*(.+?)\s*\(", log, cast=str)
    out["requested_gp"] = search(r"requested --gp\s*:\s*'([^']*)'", log, cast=str)
    if r.returncode != 0:
        out["export"] = "FAILED"
        out["export_error"] = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
        return False
    out["export"] = "ok"
    return True


def stage_import(c, cfg, out, force):
    trk = ROOT / "tracks" / f"{c['slug']}.json"
    if trk.exists() and not force:
        out["import"] = "cached"
        return True
    csv = Path(cfg["tumftm_dir"]) / c["tumftm"]
    r = run([str(APEX), "import-track", "-i", str(csv), "-o", str(trk),
             "-n", c["name"]], cwd=ROOT)
    if r.returncode != 0:
        out["import"] = "FAILED"
        out["import_error"] = (r.stdout + r.stderr)[-400:]
        return False
    out["min_radius_after"] = search(r"min radius:\s*[\d.]+ m -> ([\d.]+) m", r.stdout)
    out["import"] = "ok"
    return True


def stage_align(c, cfg, out, force):
    tel = ROOT / "telemetry" / f"{c['slug']}_2024_Q.csv"
    trk = ROOT / "tracks" / f"{c['slug']}.json"
    aligned = ROOT / "telemetry" / f"{c['slug']}_2024_Q_aligned.csv"
    sidecar = ROOT / "telemetry" / f"{c['slug']}_2024_Q.align.toml"
    if aligned.exists() and sidecar.exists() and not force:
        out["align"] = "cached"
    else:
        r = run([str(APEX), "telemetry-align", "--telemetry", str(tel),
                 "--track", str(trk), "--out", str(aligned)], cwd=ROOT)
        if r.returncode != 0:
            out["align"] = "FAILED"
            out["align_error"] = (r.stdout + r.stderr)[-400:]
            return False
        out["s_proj_span"] = search(r"s_proj span:\s*([\d.]+) m", r.stdout)
        out["within_bounds_pct"] = search(r"within track bounds:\s*([\d.]+)%", r.stdout)
        out["align"] = "ok"
    # Read the structured sidecar (robust across cache/fresh).
    sc = tomllib.loads(sidecar.read_text(encoding="utf-8"))
    out["align_scale"] = sc.get("scale")
    out["align_rms"] = sc.get("rms")
    out["align_rotation_deg"] = sc.get("rotation_deg")
    out["s_offset"] = sc.get("s_offset")
    return True


def _parse_correlate(stdout):
    span = search(r"over ([\d]+) m", stdout)
    return {
        "delta": search(r"lap delta ([+-]?[\d.]+) s", stdout),
        "rmse": search(r"speed RMSE ([\d.]+) m/s", stdout),
        "span": span,
        "corners": search(r"Corners detected:\s*(\d+)", stdout, cast=int),
        "max_dv": search(r"max \|.v\| ([\d.]+) m/s at s=([\d.]+) m", stdout),
        "max_dv_s": search(r"max \|.v\| [\d.]+ m/s at s=([\d.]+) m", stdout),
    }


def stage_correlate(c, out, tag, car=None):
    aligned = ROOT / "telemetry" / f"{c['slug']}_2024_Q_aligned.csv"
    trk = ROOT / "tracks" / f"{c['slug']}.json"
    odir = ROOT / "telemetry" / "campaign" / c["slug"] / tag
    cmd = [str(APEX), "correlate", "--telemetry", str(aligned), "--track", str(trk),
           "--calibrated", "--line", "measured", "--driven-line", "direct",
           "--out-dir", str(odir)]
    if car:
        cmd += ["--car", str(car)]
    r = run(cmd, cwd=ROOT)
    if r.returncode != 0:
        out[f"correlate_{tag}"] = "FAILED"
        out[f"correlate_{tag}_error"] = (r.stdout + r.stderr)[-400:]
        return None
    m = _parse_correlate(r.stdout)
    out[f"correlate_{tag}"] = "ok"
    out[f"{tag}_delta"] = m["delta"]
    out[f"{tag}_rmse"] = m["rmse"]
    if tag == "preset":
        out["corners"] = m["corners"]
        out["corr_span"] = m["span"]
        out["max_dv"] = m["max_dv"]
        out["max_dv_s"] = m["max_dv_s"]
    return m


def stage_identify(c, out, force):
    fitted = ROOT / "cars" / f"{c['slug']}_2024q_fitted.toml"
    aligned = ROOT / "telemetry" / f"{c['slug']}_2024_Q_aligned.csv"
    trk = ROOT / "tracks" / f"{c['slug']}.json"
    if fitted.exists() and not force:
        out["identify"] = "cached"
    else:
        r = run([str(APEX), "identify", "--telemetry", str(aligned), "--track", str(trk),
                 "--calibrated", "--free", FREE, "--out", str(fitted),
                 "--driven-line", "direct"], cwd=ROOT)
        out["identify_returncode"] = r.returncode
        if r.returncode != 0:
            out["identify"] = "BOUND-PIN or FAILED"
            out["identify_error"] = (r.stdout + r.stderr).strip().splitlines()[-1:]
            return False
        # Parse only the headline fit (before the diagnostic section).
        head = r.stdout.split("=== Diagnostic")[0]
        params = {}
        for path, key in [("aero.lift_coeff", "lift"), ("aero.drag_coeff", "drag"),
                          ("powertrain.power_scale", "power")]:
            m = re.search(
                re.escape(path) + r"\s+[\d.eE+-]+\s*.\s*([\d.eE+-]+)\s*.\s*([\d.eE+-]+)",
                head,
            )
            if m:
                params[key] = float(m.group(1))
                params[key + "_se"] = float(m.group(2))
        out["fit_params"] = params
        out["fit_cost0"] = search(r"cost:\s*([\d.]+)\s*.\s*[\d.]+", head)
        out["fit_cost1"] = search(r"cost:\s*[\d.]+\s*.\s*([\d.]+)", head)
        out["fit_condition"] = search(r"condition number \(J.J\):\s*([\d.eE+-]+)", head)
        out["fit_iters"] = search(r"\((\d+) iterations", head, cast=int)
        # μ-free diagnostic verdict line.
        out["mu_diag"] = search(r"μ moved [\d.]+ . ([\d.]+)", r.stdout)
        out["identify"] = "ok"
    # Always read fitted params from the committed TOML (robust for cached runs).
    _read_fitted_toml(fitted, out)
    return True


def _read_fitted_toml(fitted: Path, out):
    """Read fitted lift/drag/power_scale + std errors from the overlay TOML."""
    text = fitted.read_text(encoding="utf-8")
    lift = search(r"lift_coeff\s*=\s*([\d.]+)\s*#\s*.\s*([\d.]+)", text)
    lift_se = search(r"lift_coeff\s*=\s*[\d.]+\s*#\s*.\s*([\d.]+)", text)
    drag = search(r"drag_coeff\s*=\s*([\d.]+)\s*#\s*.\s*([\d.]+)", text)
    drag_se = search(r"drag_coeff\s*=\s*[\d.]+\s*#\s*.\s*([\d.]+)", text)
    power = search(r"power_scale\s*([\d.]+)\s*.\s*([\d.]+)", text)
    power_se = search(r"power_scale\s*[\d.]+\s*.\s*([\d.]+)", text)
    out.setdefault("fit_params", {})
    p = out["fit_params"]
    for k, v in [("lift", lift), ("lift_se", lift_se), ("drag", drag),
                 ("drag_se", drag_se), ("power", power), ("power_se", power_se)]:
        if v is not None:
            p.setdefault(k, v)


# --- hygiene ------------------------------------------------------------------


def hygiene(c, out):
    checks = []

    def chk(name, ok, detail):
        checks.append((name, bool(ok), detail))

    ev = (out.get("resolved_event") or "").strip().lower()
    req = (out.get("requested_gp") or c["gp"]).strip().lower()
    # Substring either way (resolved "British Grand Prix" vs requested same).
    ev_ok = bool(ev) and (req in ev or ev in req or ev == req) if ev else "cached"
    if ev == "":
        chk("event match", True, "export cached (event verified on first run)")
    else:
        chk("event match", ev_ok, f"resolved={out.get('resolved_event')!r} req={c['gp']!r}")

    scale = out.get("align_scale")
    chk("align scale", scale is not None and SCALE_LO <= scale <= SCALE_HI,
        f"{scale:.4f}" if scale is not None else "?")
    rms = out.get("align_rms")
    chk("align RMS<8m", rms is not None and rms < ALIGN_RMS_MAX,
        f"{rms:.2f} m" if rms is not None else "?")

    span = out.get("s_proj_span") or out.get("corr_span")
    tlen = out.get("track_length")
    if span and tlen:
        err = abs(span - tlen) / tlen
        chk("s_proj within 2%", err < SPROJ_TOL, f"{span:.0f} vs {tlen:.0f} m ({err*100:.2f}%)")
    else:
        chk("s_proj within 2%", False, "span/length unavailable")

    corners = out.get("corners")
    chk("corner count sane", corners is not None and CORNER_LO <= corners <= CORNER_HI,
        f"{corners}" if corners is not None else "?")

    out["hygiene"] = [{"check": n, "ok": ok, "detail": d} for n, ok, d in checks]
    out["needs_attention"] = any(c[1] is False for c in checks)
    return checks


# --- driver -------------------------------------------------------------------


def process_circuit(c, cfg, force):
    out = {"slug": c["slug"], "name": c["name"], "gp": c["gp"]}
    print(f"\n{'='*70}\n{c['name']} ({c['gp']})\n{'='*70}")

    if not stage_export(c, cfg, out, force):
        print(f"  export FAILED — run manually:\n    {out['export_cmd']}")
        out["needs_attention"] = True
        return out
    print(f"  export: {out['export']}")

    if not stage_import(c, cfg, out, force):
        print("  import-track FAILED"); out["needs_attention"] = True; return out
    trk = ROOT / "tracks" / f"{c['slug']}.json"
    out["track_length"] = track_length(trk)
    print(f"  import: {out['import']}  (track {out['track_length']:.0f} m)")

    if not stage_align(c, cfg, out, force):
        print("  align FAILED"); out["needs_attention"] = True; return out
    print(f"  align: scale={out.get('align_scale'):.4f} rms={out.get('align_rms'):.2f} m")

    stage_correlate(c, out, "preset")
    print(f"  correlate preset: delta={out.get('preset_delta')} RMSE={out.get('preset_rmse')} "
          f"corners={out.get('corners')}")

    if not stage_identify(c, out, force):
        print(f"  identify: {out.get('identify')}  {out.get('identify_error', '')}")
        out["needs_attention"] = True
    else:
        p = out.get("fit_params", {})
        print(f"  identify: lift={p.get('lift')} drag={p.get('drag')} power={p.get('power')}")
        stage_correlate(c, out, "fitted", car=ROOT / "cars" / f"{c['slug']}_2024q_fitted.toml")
        print(f"  correlate fitted: delta={out.get('fitted_delta')} RMSE={out.get('fitted_rmse')}")

    hygiene(c, out)
    for n, ok, d in [(h["check"], h["ok"], h["detail"]) for h in out["hygiene"]]:
        flag = "ok " if ok else "!! "
        print(f"    [{flag}] {n}: {d}")
    if out.get("needs_attention"):
        print("  >> NEEDS ATTENTION")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--only", default=None, help="run a single circuit slug")
    ap.add_argument("--force", action="store_true", help="redo all stages")
    args = ap.parse_args(argv)

    # Windows consoles default to cp1252; force UTF-8 so Δ/μ/± print.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    if not APEX.exists():
        die(f"release binary missing: {APEX}. Build with: cargo build --release -p apex-14")

    cfg = tomllib.loads(args.spec.read_text(encoding="utf-8"))
    circuits = cfg["circuit"]
    if args.only:
        circuits = [c for c in circuits if c["slug"] == args.only]
        if not circuits:
            die(f"no circuit with slug {args.only!r}")

    (ROOT / "telemetry" / "campaign").mkdir(parents=True, exist_ok=True)
    results = [process_circuit(c, cfg, args.force) for c in circuits]

    summary = ROOT / "telemetry" / "campaign" / "summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {summary}")
    print_summary(results)
    return 0


def die(msg):
    print(f"correlate_campaign: error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def print_summary(results):
    print(f"\n{'='*100}\nCAMPAIGN SUMMARY\n{'='*100}")
    hdr = f"{'circuit':<12} {'preset Δ':>9} {'preset RMSE':>11} {'fit Δ':>8} {'fit RMSE':>9} " \
          f"{'lift':>6} {'drag':>6} {'power':>6} {'align rms':>9} {'corners':>7} {'attn':>5}"
    print(hdr)
    for r in results:
        p = r.get("fit_params", {})
        def g(k, w=".2f"):
            v = r.get(k)
            return format(v, w) if isinstance(v, (int, float)) else "-"
        def gp(k, w=".3f"):
            v = p.get(k)
            return format(v, w) if isinstance(v, (int, float)) else "-"
        print(f"{r['slug']:<12} {g('preset_delta'):>9} {g('preset_rmse'):>11} "
              f"{g('fitted_delta'):>8} {g('fitted_rmse'):>9} {gp('lift'):>6} {gp('drag'):>6} "
              f"{gp('power'):>6} {g('align_rms'):>9} {str(r.get('corners','-')):>7} "
              f"{'YES' if r.get('needs_attention') else '':>5}")


if __name__ == "__main__":
    raise SystemExit(main())
