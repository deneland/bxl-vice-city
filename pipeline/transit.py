"""STIB trams and buses from the GTFS timetable -> viewer/public/transit/<roi>.json.

Per weekday (the feed's dates are mapped to a normal week: the one in the feed with the most service,
i.e. not a holiday week), every trip that stops at least
twice inside the region, with its stops in the region and their times. The metro runs underground and
is left out. Positions in voxel coordinates.

Usage: uv run python transit.py [roi-name]
Data: STIB-MIVB open data (GTFS), via data/raw/gtfs/stib.zip.
"""
import csv
import io
import json
import zipfile
from datetime import date, timedelta

from common import ROOT, VS, region, to_l72

ROI_NAME, ROI = region()
ZIP = ROOT / "data/raw/gtfs/stib.zip"
OUT = ROOT / "viewer/public/transit" / f"{ROI_NAME}.json"
MODES = {"0": 0, "3": 3}            # GTFS route_type: 0 tram, 3 bus (1 metro: underground, skipped)



def rows(z, name):
    return csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8-sig"))


def secs(t):
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def main():
    z = zipfile.ZipFile(ZIP)
    # stops in the region
    stop_ix, out_stops = {}, []
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

    def services(d):
        ds, wd = d.strftime("%Y%m%d"), days[d.weekday()]
        s = {sid for sid, c in cal.items() if c[wd] == "1" and c["start_date"] <= ds <= c["end_date"]}
        for sid, ex in extra.get(ds, {}).items():
            (s.add if ex == "1" else s.discard)(sid)
        return s
    # the reference week: of the weeks the feed covers, the one whose thinnest day has the most services
    parse = lambda v: date(int(v[:4]), int(v[4:6]), int(v[6:]))
    first, last = min(parse(c["start_date"]) for c in cal.values()), max(parse(c["end_date"]) for c in cal.values())
    monday = first - timedelta(days=first.weekday())
    weeks = []
    while monday + timedelta(days=6) <= last:
        week = [monday + timedelta(days=k) for k in range(7)]
        weeks.append((min(len(services(d)) for d in week), week))
        monday += timedelta(days=7)
    if not weeks:
        raise SystemExit("the GTFS feed does not cover a whole week")
    ref_week = max(weeks)[1]
    active = [services(d) for d in ref_week]
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
    if not all(out_days):
        raise SystemExit(f"no trips on {[days[k] for k, d in enumerate(out_days) if not d]} of the week of {ref_week[0]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"routes": out_routes, "stops": out_stops, "days": out_days,
                               "fields": {"routes": ["name", "mode (0 tram, 3 bus)", "colour"],
                                          "trips": ["route", "headsign", "[[stop, arrival s, departure s]]"],
                                          "days": "Monday .. Sunday"}}, separators=(",", ":"), ensure_ascii=False))
    print(f"week of {ref_week[0]}: {len(out_stops)} stops, {len(out_routes)} lines ({', '.join(r[0] for r in out_routes)}); "
          f"trips per day {[len(d) for d in out_days]} -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
