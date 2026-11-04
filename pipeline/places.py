"""Places for the traffic agents, from OpenStreetMap: homes (buildings with an address, weighted by
their number of floors, estimated from the LiDAR height) and destinations (schools, shops, food,
health, offices, leisure, services) -> viewer/public/places/<roi>.json.

Homes on EXCLUDE_STREETS are left out (privacy). Places are in voxel coordinates.

Usage: uv run python places.py [roi-name] [--refetch]
Map data © OpenStreetMap contributors, ODbL.
"""
import json
import sys
import time
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from pyproj import Transformer

from model import Model

ROOT = Path(__file__).resolve().parent.parent
CFG = tomllib.loads((ROOT / "config.toml").read_text())
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
ROI_NAME = ARGS[0] if ARGS else CFG["default"]
ROI = CFG["roi"][ROI_NAME]
RAW = ROOT / "data/raw" / f"osm_places_{ROI_NAME}.json"
OUT = ROOT / "viewer/public/places" / f"{ROI_NAME}.json"

EXCLUDE_STREETS = ["Grote Bosstraat"]          # no simulated homes here
BUILDING, GROUND = 6, 2

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

to_wgs = Transformer.from_crs(31370, 4326, always_xy=True)
to_l72 = Transformer.from_crs(4326, 31370, always_xy=True)


def fetch():
    corners = [to_wgs.transform(x, y) for x in (ROI["xmin"], ROI["xmax"]) for y in (ROI["ymin"], ROI["ymax"])]
    bb = (f"{min(c[1] for c in corners):.6f},{min(c[0] for c in corners):.6f},"
          f"{max(c[1] for c in corners):.6f},{max(c[0] for c in corners):.6f}")
    q = (f'[out:json][timeout:120];(way["building"]({bb});nwr["shop"]({bb});nwr["amenity"]({bb});'
         f'nwr["office"]({bb});nwr["leisure"]({bb}););out tags center;')
    req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                                 data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "bxl-vice-voxel/0.1"})
    for i in range(5):            # the public Overpass server is often briefly overloaded
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                RAW.write_bytes(r.read())
            return
        except Exception as e:
            print(f"overpass: {e}; retrying")
            time.sleep(10 * (i + 1))
    raise SystemExit("overpass unavailable")


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
                b = np.flatnonzero(c == BUILDING)
                if len(b):
                    best = max(best, (b.max() - np.flatnonzero(c).min()) * vs)
        return int(np.clip(round(best / 3), 1, 12))

    streets, names = {}, {}
    sid = lambda s: streets.setdefault(s, len(streets))
    nid = lambda s: names.setdefault(s, len(names))
    homes, pois, skipped = [], [], 0
    for e in els:
        t = e.get("tags", {})
        p = vox(e)
        if p is None:
            continue
        cat = category(t)
        if cat:
            pois.append([round(p[0], 1), round(p[1], 1), CATS.index(cat), nid(t.get("name", ""))])
            continue
        if "building" in t and t["building"] not in NOT_HOMES and ("addr:housenumber" in t or t["building"] in
                                                                  ("apartments", "house", "residential", "terrace")):
            street = t.get("addr:street", "")
            if any(x in street for x in EXCLUDE_STREETS):
                skipped += 1
                continue
            lv = t.get("building:levels", "")
            homes.append([round(p[0], 1), round(p[1], 1), int(lv) if lv.isdigit() else floors(*p), sid(street)])
    out = {"build": m.man["build"], "cats": CATS, "streets": list(streets), "names": list(names),
           "homes": homes, "pois": pois,
           "fields": {"homes": ["x", "y", "floors", "street"], "pois": ["x", "y", "cat", "name"]}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    cnt = {c: sum(1 for q in pois if q[2] == i) for i, c in enumerate(CATS)}
    print(f"{len(homes)} homes ({sum(h[2] for h in homes)} floors, {skipped} on excluded streets left out), "
          f"{len(pois)} destinations {cnt} -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    if "--refetch" in sys.argv or not RAW.exists():
        fetch()
    build()
