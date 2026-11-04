"""STIB trams and buses from the GTFS timetable -> viewer/public/transit/<roi>.json.

Per weekday (the feed's dates are mapped to a normal week, REF_WEEK), every trip that stops at least
twice inside the region, with its stops in the region and their times. The metro runs underground and
is left out. Positions in voxel coordinates.

Usage: uv run python transit.py [roi-name]
Data: STIB-MIVB open data (GTFS), via data/raw/gtfs/stib.zip.
"""
import csv
import io
import json
import sys
import tomllib
import zipfile
from datetime import date
from pathlib import Path

from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
CFG = tomllib.loads((ROOT / "config.toml").read_text())
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
ROI_NAME = ARGS[0] if ARGS else CFG["default"]
ROI = CFG["roi"][ROI_NAME]
ZIP = ROOT / "data/raw/gtfs/stib.zip"
OUT = ROOT / "viewer/public/transit" / f"{ROI_NAME}.json"
VS = CFG["voxel"]["size"]

# a normal week inside the feed's validity: Monday .. Sunday
REF_WEEK = [date(2026, 3, 16 + k) for k in range(7)]
MODES = {"0": 0, "3": 3}            # GTFS route_type: 0 tram, 3 bus (1 metro: underground, skipped)

to_l72 = Transformer.from_crs(4326, 31370, always_xy=True)


def rows(z, name):
    return csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8-sig"))


def secs(t):
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def main():
    z = zipfile.ZipFile(ZIP)
    # stops in the region
    stops, stop_ix, out_stops = {}, {}, []
    for r in rows(z, "stops.txt"):
        x, y = to_l72.transform(float(r["stop_lon"]), float(r["stop_lat"]))
        inside = ROI["xmin"] <= x < ROI["xmax"] and ROI["ymin"] <= y < ROI["ymax"]
        if inside:
            stop_ix[r["stop_id"]] = len(out_stops)
            out_stops.append([round((x - ROI["xmin"]) / VS, 1), round((y - ROI["ymin"]) / VS, 1), r["stop_name"].title()])
    # routes
    routes, route_ix, out_routes = {}, {}, []
    for r in rows(z, "routes.txt"):
        if r["route_type"] in MODES:
            routes[r["route_id"]] = r
    # services per reference day
    cal = {r["service_id"]: r for r in rows(z, "calendar.txt")}
    extra = {}
    for r in rows(z, "calendar_dates.txt"):
        extra.setdefault(r["date"], {})[r["service_id"]] = r["exception_type"]
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    active = []
    for k, d in enumerate(REF_WEEK):
        ds = d.strftime("%Y%m%d")
        s = {sid for sid, c in cal.items() if c[days[k]] == "1" and c["start_date"] <= ds <= c["end_date"]}
        for sid, ex in extra.get(ds, {}).items():
            (s.add if ex == "1" else s.discard)(sid)
        active.append(s)
    trips = {}
    for r in rows(z, "trips.txt"):
        if r["route_id"] in routes:
            trips[r["trip_id"]] = r
    # stop times of those trips, in the region only
    times = {}
    for r in rows(z, "stop_times.txt"):
        t = r["trip_id"]
        if t in trips and r["stop_id"] in stop_ix:
            times.setdefault(t, []).append((int(r["stop_sequence"]), stop_ix[r["stop_id"]],
                                            secs(r["arrival_time"]), secs(r["departure_time"])))
    out_days = [[] for _ in range(7)]
    for t, st in times.items():
        if len(st) < 2:
            continue
        st.sort()
        tr = trips[t]
        rid = tr["route_id"]
        if rid not in route_ix:
            r = routes[rid]
            route_ix[rid] = len(out_routes)
            out_routes.append([r["route_short_name"], MODES[r["route_type"]], r["route_color"]])
        entry = [route_ix[rid], tr["trip_headsign"].title(), [[s, a, b] for _, s, a, b in st]]
        for k in range(7):
            if tr["service_id"] in active[k]:
                out_days[k].append(entry)
    for d in out_days:
        d.sort(key=lambda e: e[2][0][1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"routes": out_routes, "stops": out_stops, "days": out_days,
                               "fields": {"routes": ["name", "mode (0 tram, 3 bus)", "colour"],
                                          "trips": ["route", "headsign", "[[stop, arrival s, departure s]]"],
                                          "days": "Monday .. Sunday"}}, separators=(",", ":"), ensure_ascii=False))
    print(f"{len(out_stops)} stops, {len(out_routes)} lines ({', '.join(r[0] for r in out_routes)}); "
          f"trips per day {[len(d) for d in out_days]} -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
