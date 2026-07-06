#!/usr/bin/env python3
"""Export a single lap of FastF1 telemetry to the standard Apex telemetry CSV.

The output conforms to ``docs/telemetry_format.md``: a ``# key: value`` comment
header (data-source identifiers only — NO RunMetadata sim hashes, because this
is *measured* data, not a simulation artifact), a ``# grid:`` line, a
``# columns:`` units line, then registry-named columns in registry-canonical
units.

Units are converted **here** so the CSV is registry-canonical on disk:

    * speed:    km/h -> m/s   (FastF1 'Speed' is km/h)
    * throttle: percent -> 0-1 fraction
    * brake:    bool -> 0/1
    * x, y:     FastF1 position is in *decimetres* -> metres  (÷10)

The x/y frame is FastF1-local (its own coordinate system), NOT the Apex track
frame. Aligning it to the track is a later task; the columns are exported as-is
for that future work.

Dependencies: Python stdlib + ``fastf1`` + ``pandas``. If ``fastf1`` is not
installed, or the requested session/lap is not found, the script exits with a
clear message and a non-zero status.

Example::

    python tools/fastf1_export.py --year 2024 --gp "Great Britain" \\
        --session Q --driver VER --out telemetry/silverstone_2024_Q_VER.csv
"""

import argparse
import sys
from pathlib import Path

# Script version — bumped when the output format/semantics change. Recorded in
# the CSV header for provenance.
SCRIPT_VERSION = "1.1.0"

# Cache directory for FastF1's HTTP cache (kept local; see telemetry/README.md).
CACHE_DIR = Path("telemetry/.fastf1_cache")

# Output column order: registry channel names in registry-canonical units.
# (t, s from FastF1 Time/Distance; the rest are measured channels.)
COLUMNS = ["t", "s", "speed", "throttle", "brake", "gear", "rpm", "x", "y"]

# Registry unit symbols for the `# columns:` line (must match the registry:
# apex_telemetry::channels). Empty string => dimensionless (renders `[]`).
UNIT_SYMBOLS = {
    "t": "s",
    "s": "m",
    "speed": "m/s",
    "throttle": "",
    "brake": "",
    "gear": "",
    "rpm": "rpm",
    "x": "m",
    "y": "m",
}


def die(msg: str, code: int = 1) -> "NoReturn":  # type: ignore[name-defined]
    """Print an error to stderr and exit non-zero."""
    print(f"fastf1_export: error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export one lap of FastF1 telemetry as an Apex telemetry CSV."
    )
    p.add_argument("--year", type=int, required=True, help="Season year, e.g. 2024")
    p.add_argument(
        "--gp",
        required=True,
        help='Grand Prix name or round, e.g. "British Grand Prix", "Silverstone", or "12"',
    )
    p.add_argument(
        "--session",
        required=True,
        help="Session identifier, e.g. Q, R, FP1, S (sprint)",
    )
    p.add_argument(
        "--driver",
        default=None,
        help="Driver code (e.g. VER). Default: the session's fastest lap.",
    )
    p.add_argument(
        "--lap",
        default="fastest",
        help='Which lap: "fastest" (default) or a 1-based lap number.',
    )
    p.add_argument(
        "--out",
        default=None,
        type=Path,
        help=(
            "Output CSV path. Default: telemetry/<location>_<year>_<session>_<driver>.csv "
            "derived from the RESOLVED session metadata (never from --gp)."
        ),
    )
    p.add_argument(
        "--allow-fuzzy",
        action="store_true",
        help=(
            "Permit FastF1's fuzzy event matcher to resolve --gp to a materially "
            "different event name. Without this, a fuzzy mismatch is a hard error "
            "(guards against e.g. 'Great Britain' silently resolving to Austria)."
        ),
    )
    return p.parse_args(argv)


def _norm(s) -> str:
    """Lowercase, keep only alphanumerics — for tolerant event-name comparison."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def event_matches(requested: str, name, location, official, round_number) -> bool:
    """True if `requested` plausibly names the resolved event.

    Accepts an exact round number, or a normalized (case/punctuation-insensitive)
    match / substring against the resolved event name, circuit location, or
    official name. A materially different string (e.g. 'Great Britain' vs
    'British Grand Prix'/'Silverstone') does NOT match — that must be an explicit
    --allow-fuzzy decision.
    """
    req_raw = str(requested).strip()
    if req_raw.isdigit():
        try:
            return int(req_raw) == int(round_number)
        except (TypeError, ValueError):
            return False
    r = _norm(requested)
    if not r:
        return False
    for cand in (name, location, official):
        c = _norm(cand)
        if c and (r == c or r in c or c in r):
            return True
    return False


def _slug(s) -> str:
    """A filename-safe lowercase slug of a resolved metadata string."""
    return "".join(ch if ch.isalnum() else "_" for ch in str(s).strip().lower()).strip("_")


def load_fastf1():
    """Import fastf1/pandas, or exit with an actionable message."""
    try:
        import fastf1  # noqa: F401
        import pandas  # noqa: F401
    except ImportError as e:
        die(
            f"missing dependency ({e.name}). Install with: "
            "pip install fastf1 pandas"
        )
    import fastf1

    return fastf1


def select_lap(session, driver, lap_spec):
    """Return the chosen lap (a fastf1 Lap), or exit with a clear message."""
    laps = session.laps
    if driver is not None:
        laps = laps.pick_drivers(driver)
        if laps is None or len(laps) == 0:
            die(f"no laps for driver {driver!r} in this session")

    if str(lap_spec).lower() == "fastest":
        lap = laps.pick_fastest()
        if lap is None or (hasattr(lap, "empty") and lap.empty):
            die("no fastest lap available (session may have no lap data)")
        return lap

    # Numeric lap number.
    try:
        lap_number = int(lap_spec)
    except ValueError:
        die(f"--lap must be 'fastest' or an integer, got {lap_spec!r}")

    match = laps[laps["LapNumber"] == lap_number]
    if len(match) == 0:
        die(f"lap number {lap_number} not found for the selection")
    return match.iloc[0]


def build_frame(lap):
    """Get car telemetry with a Distance channel added; convert units.

    Returns a pandas DataFrame with the Apex output columns.
    """
    import pandas as pd

    # add_distance() computes the 'Distance' channel; get_car_data() has
    # Speed/Throttle/Brake/nGear/RPM; get_pos_data() has X/Y. Merge on time.
    try:
        tel = lap.get_telemetry()  # merged car + pos + Distance
    except Exception as e:  # noqa: BLE001 - surface any FastF1 assembly failure
        die(f"failed to assemble telemetry: {e}")

    tel = tel.add_distance() if "Distance" not in tel else tel

    def col(name):
        if name not in tel:
            die(f"expected FastF1 channel {name!r} missing from telemetry")
        return tel[name]

    # Time: seconds from lap start. FastF1 'Time' is a timedelta.
    t = col("Time").dt.total_seconds()

    out = pd.DataFrame(
        {
            "t": t,
            "s": col("Distance").astype(float),  # metres
            "speed": col("Speed").astype(float) / 3.6,  # km/h -> m/s
            "throttle": col("Throttle").astype(float) / 100.0,  # percent -> 0-1
            "brake": col("Brake").astype(float),  # bool/0-1 -> 0/1
            "gear": col("nGear").astype(float),
            "rpm": col("RPM").astype(float),
            "x": col("X").astype(float) / 10.0,  # decimetres -> metres
            "y": col("Y").astype(float) / 10.0,  # decimetres -> metres
        }
    )
    return out[COLUMNS]


def columns_comment() -> str:
    parts = [f"{name}[{UNIT_SYMBOLS[name]}]" for name in COLUMNS]
    return "# columns: " + ", ".join(parts)


def write_csv(path: Path, frame, header_meta: dict) -> None:
    """Write the standard Apex telemetry CSV (grid: s)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        # Descriptive provenance only — NO RunMetadata sim hashes (measured data).
        for k, v in header_meta.items():
            f.write(f"# {k}: {v}\n")
        f.write("# grid: s\n")
        f.write(columns_comment() + "\n")
        f.write("#\n")
        f.write(",".join(COLUMNS) + "\n")
        for row in frame.itertuples(index=False):
            f.write(",".join(_fmt(v) for v in row) + "\n")


def _fmt(v) -> str:
    """Format a value; non-finite -> 'NaN' (kept as a measured gap)."""
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "NaN"
    if fv != fv:  # NaN
        return "NaN"
    return repr(fv)


def main(argv=None) -> int:
    args = parse_args(argv)
    fastf1 = load_fastf1()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    try:
        session = fastf1.get_session(args.year, args.gp, args.session)
    except Exception as e:  # noqa: BLE001
        die(f"could not resolve session ({args.year} {args.gp!r} {args.session!r}): {e}")

    try:
        session.load(telemetry=True, laps=True, weather=False)
    except Exception as e:  # noqa: BLE001
        die(f"failed to load session data: {e}")

    # --- Resolve the ACTUAL event from the loaded session, and guard against
    # FastF1's fuzzy matcher having silently corrected --gp to another event.
    ev = _event_fields(session, args.year)
    print("=" * 64)
    print(f"  requested --gp : {args.gp!r}")
    print(f"  RESOLVED event : {ev['name']}  (round {ev['round']}, {ev['location']})")
    print(f"  session date   : {ev['date']}   year {ev['year']}   session {args.session}")
    print("=" * 64)

    if not event_matches(args.gp, ev["name"], ev["location"], ev["official"], ev["round"]):
        if not args.allow_fuzzy:
            die(
                f"requested --gp {args.gp!r} resolved to a DIFFERENT event: "
                f"{ev['name']!r} at {ev['location']!r} (round {ev['round']}). "
                "FastF1's fuzzy matcher may have corrected the name. If this is "
                "intended, re-run with --allow-fuzzy; otherwise fix --gp."
            )
        print(
            "fastf1_export: warning: --gp did not match the resolved event; "
            "proceeding because --allow-fuzzy was given.",
            file=sys.stderr,
        )

    lap = select_lap(session, args.driver, args.lap)
    frame = build_frame(lap)

    driver = args.driver if args.driver is not None else _lap_driver(lap)
    lap_number = _lap_number(lap)
    lap_time = _lap_time(lap)

    # Header + default output filename derive ONLY from resolved metadata.
    header_meta = {
        "source": "fastf1",
        "year": ev["year"],
        "event": ev["name"],
        "official_event": ev["official"],
        "location": ev["location"],
        "round": ev["round"],
        "session": args.session,
        "session_date": ev["date"],
        "driver": driver,
        "lap": lap_number,
        "lap_time_s": lap_time,
        "requested_gp": args.gp,  # recorded for traceability, not authoritative
        "fastf1_version": getattr(fastf1, "__version__", "unknown"),
        "exporter": "tools/fastf1_export.py",
        "exporter_version": SCRIPT_VERSION,
    }

    out_path = args.out
    if out_path is None:
        loc = _slug(ev["location"]) or _slug(ev["name"]) or "session"
        out_path = Path("telemetry") / f"{loc}_{ev['year']}_{args.session}_{driver}.csv"

    write_csv(out_path, frame, header_meta)
    print(
        f"wrote {len(frame)} samples -> {out_path} "
        f"({ev['name']} {args.session} {driver} lap {lap_number}, "
        f"lap time {lap_time}s)"
    )
    return 0


def _event_fields(session, fallback_year) -> dict:
    """Read resolved event metadata defensively from the loaded session."""
    ev = getattr(session, "event", None)

    def field(*keys, default="unknown"):
        for k in keys:
            try:
                val = ev[k]
            except Exception:  # noqa: BLE001
                continue
            if val is not None and str(val) != "nan":
                return val
        return default

    name = field("EventName")
    official = field("OfficialEventName", default=name)
    location = field("Location", "Country")
    round_number = field("RoundNumber", default="unknown")

    # Year and date from the event date if available, else the session date.
    year = fallback_year
    date_str = "unknown"
    try:
        ed = ev["EventDate"]
        year = int(ed.year)
        date_str = ed.date().isoformat()
    except Exception:  # noqa: BLE001
        sd = getattr(session, "date", None)
        if sd is not None:
            try:
                year = int(sd.year)
                date_str = sd.date().isoformat()
            except Exception:  # noqa: BLE001
                pass

    return {
        "name": name,
        "official": official,
        "location": location,
        "round": round_number,
        "year": year,
        "date": date_str,
    }


def _lap_time(lap):
    """Lap time in seconds (float) if available, else 'unknown'."""
    try:
        lt = lap["LapTime"]
        secs = lt.total_seconds()
        return round(secs, 3) if secs == secs else "unknown"  # guard NaT/NaN
    except Exception:  # noqa: BLE001
        return "unknown"


def _lap_driver(lap):
    try:
        return lap["Driver"]
    except Exception:  # noqa: BLE001
        return "unknown"


def _lap_number(lap):
    try:
        n = lap["LapNumber"]
        return int(n) if n == n else "unknown"  # guard NaN
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
