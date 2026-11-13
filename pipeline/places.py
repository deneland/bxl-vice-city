"""Places for the traffic agents, from OpenStreetMap: homes (buildings with an address, weighted by
their number of floors, estimated from the LiDAR height) and destinations (schools, shops, food,
health, offices, leisure, services) -> viewer/public/places/<roi>.json.

Homes on EXCLUDE_STREETS are left out (privacy): by their address, and buildings without an address
by distance to the street (its geometry from roads.py's OSM download). Places are in voxel coordinates.

Usage: uv run python places.py [roi-name] [--refetch]
Map data © OpenStreetMap contributors, ODbL.
"""
import json
import sys

import numpy as np

from common import C_BUILDING, ROOT, bbox, osm_nodes, overpass, region, to_l72
from model import Model

ROI_NAME, ROI = region()
RAW = ROOT / "data/raw" / f"osm_places_{ROI_NAME}.json"
OUT = ROOT / "viewer/public/places" / f"{ROI_NAME}.json"

EXCLUDE_STREETS = ["Grote Bosstraat"]          # no simulated homes here
EXCLUDE_NEAR = 20.0                            # m: buildings without an address this close to them too

# destination categories (order = index in the output)
CATS = ["school", "kindergarten", "shop", "supermarket", "food", "health", "office", "leisure", "services"]
AMENITY = {
    "school": "school", "college": "school", "university": "school",
    "kindergarten": "kindergarten", "childcare": "kindergarten",
    "restaurant": "food", "cafe": "food", "fast_food": "food", "bar": "food", "pub": "food",
    "pharmacy": "health", "clinic": "health", "doctors": "health", "dentist": "health", "hospital": "health",
    "theatre": "leisure", "cinema": "leisure", "library": "leisure", "community_centre": "leisure",
    "place_of_worship": "leisure", "arts_centre": "leisure",
    "bank": "services", "post_office": "services", "townhall": "services", "police": "services",
}
LEISURE = {"sports_centre", "fitness_centre", "swimming_pool", "pitch", "sports_hall"}
NOT_HOMES = {"school", "industrial", "retail", "garages", "garage", "shed", "roof", "church", "commercial",
             "office", "warehouse", "kindergarten", "hospital", "public", "civic", "train_station", "service"}

# opening hours: a week as 7 day masks (Monday first), bit h = open for most of hour h. From OSM's
# opening_hours where it parses (the common forms: "Mo-Fr 09:00-18:00; Sa 10:00-17:00", "24/7", "off",
# lunch breaks, past midnight); else a typical week for the category
DAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
def hours(a, b):
    return sum(1 << h for h in range(24) if min(b, (h + 1) * 60) - max(a, h * 60) >= 30)
WEEK = lambda days, a, b: [hours(a * 60, b * 60) if d in days else 0 for d in range(7)]
DEFAULT_HOURS = {
    "school": WEEK(range(5), 7, 18), "kindergarten": WEEK(range(5), 7, 18),
    "shop": WEEK(range(6), 10, 19), "supermarket": WEEK(range(6), 8, 20),
    "food": WEEK(range(7), 11, 23), "health": WEEK(range(5), 8, 19), "office": WEEK(range(5), 7, 19),
    "leisure": WEEK(range(7), 8, 22), "services": WEEK(range(5), 9, 17),
}


def parse_hours(v):
    """OSM opening_hours -> 7 day masks, or None when not understood"""
    try:
        return _parse_hours(v.strip())
    except (ValueError, IndexError):
        return None


def _parse_hours(v):
    if v == "24/7":
        return [(1 << 24) - 1] * 7
    week = [0] * 7
    ok = False
    for rule in v.split(";"):
        rule = rule.strip()
        if not rule or rule.startswith("PH") or rule.startswith("SH"):
            continue
        parts = rule.split(" ", 1)
        if parts[0][:2] in DAYS:
            days = set()
            for span in parts[0].split(","):
                if "-" in span:
                    a, b = span.split("-")
                    if a not in DAYS or b not in DAYS:
                        return None
                    i, j = DAYS.index(a), DAYS.index(b)
                    days |= set(range(i, j + 1)) if i <= j else set(range(i, 7)) | set(range(0, j + 1))
                elif span in DAYS:
                    days.add(DAYS.index(span))
                elif span == "PH":
                    continue
                else:
                    return None
            times = parts[1].strip() if len(parts) > 1 else "00:00-24:00"
        else:
            days, times = set(range(7)), rule
        if times in ("off", "closed"):
            for d in days:
                week[d] = 0
            ok = True
            continue
        masks = [0] * 7                      # (a rule replaces earlier rules for its days)
        for span in times.split(","):
            try:
                a, b = span.strip().split("-")
                ha, ma = map(int, a.split(":")); hb, mb = map(int, b.rstrip("+").split(":"))
            except ValueError:
                return None
            t0, t1 = ha * 60 + ma, hb * 60 + mb
            for d in days:
                if t1 > t0:
                    masks[d] |= hours(t0, min(t1, 1440))
                else:                        # past midnight: the rest of this day, then the next morning
                    masks[d] |= hours(t0, 1440)
                    masks[(d + 1) % 7] |= hours(0, t1)
        for d in days:
            week[d] = masks[d]
        for d in range(7):
            if d not in days and masks[d]:
                week[d] |= masks[d]
        ok = True
    return week if ok else None




def fetch():
    bb = bbox(ROI)
    overpass(f'[out:json][timeout:120];(way["building"]({bb});nwr["shop"]({bb});nwr["amenity"]({bb});'
             f'nwr["office"]({bb});nwr["leisure"]({bb}););out tags center;', RAW)


def category(t):
    if t.get("shop"):
        return "supermarket" if t["shop"] in ("supermarket", "convenience") else "shop"
    if t.get("amenity") in AMENITY:
        return AMENITY[t["amenity"]]
    if t.get("office") or t.get("building") in ("office", "commercial"):
        return "office"
    if t.get("leisure") in LEISURE:
        return "leisure"
    return None


def excluded_street_lines():
    """polylines (L72 m) of the EXCLUDE_STREETS, from the road network's OSM download"""
    raw = ROOT / "data/raw" / f"osm_roads_{ROI_NAME}.json"
    if not raw.exists():
        raise SystemExit(f"{raw.relative_to(ROOT)} missing: run roads.py first (needed to keep homes off EXCLUDE_STREETS)")
    els = json.loads(raw.read_text())["elements"]
    nodes = osm_nodes(els)
    return [np.array([nodes[n] for n in e["nodes"] if n in nodes]) for e in els
            if e["type"] == "way" and any(x in e.get("tags", {}).get("name", "") for x in EXCLUDE_STREETS)]


def near_lines(p, lines, d):
    """is point p (L72 m) within d m of any of the polylines"""
    for q in lines:
        a, b = q[:-1], q[1:]
        ab = b - a
        t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-9), 0, 1)
        if (np.hypot(*(a + t[:, None] * ab - p).T) < d).any():
            return True
    return False


def build():
    els = json.loads(RAW.read_text())["elements"]
    m = Model(ROI_NAME)
    X0, Y0 = m.man["origin"][:2]
    vs = m.vs

    def vox(e):
        c = e.get("center") or {"lon": e.get("lon"), "lat": e.get("lat")}
        if c["lon"] is None:
            return None
        x, y = to_l72.transform(c["lon"], c["lat"])
        vx, vy = (x - X0) / vs, (y - Y0) / vs
        if not (0 <= vx < m.size[0] and 0 <= vy < m.size[1]):
            return None
        return vx, vy

    def floors(vx, vy):
        """floors from the LiDAR: building height (top building voxel above the lowest) / 3 m"""
        best = 0
        for dx in (-4, 0, 4):
            for dy in (-4, 0, 4):
                c = m.column(int(vx) + dx, int(vy) + dy)
                if c is None:
                    continue
                b = np.flatnonzero(c == C_BUILDING)
                if len(b):
                    best = max(best, (b.max() - np.flatnonzero(c).min()) * vs)
        return int(np.clip(round(best / 3), 1, 12))

    streets, names = {}, {}
    sid = lambda s: streets.setdefault(s, len(streets))
    nid = lambda s: names.setdefault(s, len(names))
    homes, pois, skipped = [], [], 0
    weeks, week_ix, parsed, tagged = [], {}, 0, 0
    def week_of(cat, t):
        nonlocal parsed, tagged
        w = None
        if "opening_hours" in t:
            tagged += 1
            w = parse_hours(t["opening_hours"])
            parsed += w is not None
        key = tuple(w or DEFAULT_HOURS[cat])
        if key not in week_ix:
            week_ix[key] = len(weeks)
            weeks.append(list(key))
        return week_ix[key]
    # spatial order: the LiDAR tiles behind floors() are read through a bounded cache
    def where(e):
        c = e.get("center") or {"lon": e.get("lon") or 0, "lat": e.get("lat") or 0}
        return (round(c["lat"] * 300), c["lon"])
    els.sort(key=where)
    excluded = excluded_street_lines()
    for e in els:
        t = e.get("tags", {})
        p = vox(e)
        if p is None:
            continue
        cat = category(t)
        if cat:
            pois.append([round(p[0], 1), round(p[1], 1), CATS.index(cat), nid(t.get("name", "")), week_of(cat, t)])
            continue
        if "building" in t and t["building"] not in NOT_HOMES and ("addr:housenumber" in t or t["building"] in
                                                                  ("apartments", "house", "residential", "terrace")):
            street = t.get("addr:street", "")
            if any(x in street for x in EXCLUDE_STREETS) or (
                    not street and near_lines(np.array(p) * vs + [X0, Y0], excluded, EXCLUDE_NEAR)):
                skipped += 1
                continue
            lv = t.get("building:levels", "")
            homes.append([round(p[0], 1), round(p[1], 1), int(lv) if lv.isdigit() else floors(*p), sid(street)])
    # homes: [x, y, floors, street]; pois: [x, y, cat, name, opening week]; weeks: 7 day masks, Monday first
    # (bit h: open in hour h)
    out = {"cats": CATS, "streets": list(streets), "names": list(names), "homes": homes, "pois": pois, "weeks": weeks}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    cnt = {c: sum(1 for q in pois if q[2] == i) for i, c in enumerate(CATS)}
    print(f"{len(homes)} homes ({sum(h[2] for h in homes)} floors, {skipped} on excluded streets left out), "
          f"{len(pois)} destinations {cnt}, opening hours from OSM for {parsed} ({tagged} tagged), {len(weeks)} distinct weeks -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    if "--refetch" in sys.argv or not RAW.exists():
        fetch()
    build()
