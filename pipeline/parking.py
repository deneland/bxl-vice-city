"""Parking places along the kerbs, where the LiDAR saw parked cars in 2021 -> viewer/public/parking/<roi>.json.

Real parking follows the street: a row of cars on a line parallel to the kerb, nose to tail, facing the
driving direction of that side. So places are laid out along the road network (roads.json), and the
LiDAR vehicle footprints (voxelize.py) decide on which sides people park, at what distance from the
centreline, and which stretches stay free. (Feel over accuracy: the scan only labelled part of the
parked cars, so gaps up to GAP between seen cars are parking too.)

  per road edge and side: sample the edge every voxel; across it, count footprint columns at each
  distance from the centreline (beyond the driving lane, before the pavement) -> the parking line is
  the distance with the most footprint. Along that line, stretches with footprint (gaps between cars
  closed) become rows of evenly spaced places, kept clear of junctions. Rows whose footprint is deep
  across the kerb (> 3.6 m) are perpendicular parking: places 2.6 m apart, facing the kerb.

Car parks (OpenStreetMap, data/raw/osm_lots_<roi>.json): inside each mapped surface car park, the
LiDAR footprints give the stalls: a car-sized blob of footprint is a car (along its long axis); bigger
blobs (cars touching) a row of cars side by side (wide: 2.5 m apart) or nose to tail (narrow). Public underground / multi-storey
garages are places too, at their entrances, never drawn: cars drive in and are gone.

Places are [x, y (voxels), heading (rad, x east / y north), kind (0 kerb, 1 car park, 2 garage)].

Usage: uv run python parking.py [roi-name] [--refetch]
"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from common import (E_CROSSWALKS, E_FROM, E_KIND, E_ONEWAY, E_PARKING, E_PAVEMENT, E_TO, E_XYZ, KIND_ROAD, ROOT,
                    VS, bbox, overpass, region, to_l72)
from model import Model

ROI_NAME, ROI = region()
FOOT = ROOT / "data/cache/vehicles" / f"{ROI_NAME}.npy"
ROADS = ROOT / "viewer/public/roads" / f"{ROI_NAME}.json"
OUT = ROOT / "viewer/public/parking" / f"{ROI_NAME}.json"

PARALLEL = 5.4      # m of kerb per parked car (car + gap)
PERP = 2.6          # m per perpendicular place
JUNCTION = 5.0      # m beyond the crossing street's kerb line kept clear (Belgian highway code: 5 m)
CROSSWALK = 6.0     # m kept clear either side of a mapped crosswalk (5 m before it, plus its width)
STOP_CLEAR = 15.0   # m kept clear around bus / tram stops (highway code: 15 m either side of the sign)
TRANSIT = ROOT / "viewer/public/transit" / f"{ROI_NAME}.json"
GAP = 25.0          # m: shorter stretches without a car between two cars are parking too (the 2021 scan
                    # labelled only part of the parked cars as vehicles); longer ones stay free (bus stops,
                    # driveways, schools)
REACH = 20.0        # m: a row ending this close to the junction margin is extended to it
MIN_ROW = 4.5       # m: shorter stretches are not a parking place
LANE_CLEAR = {0: 2.6, 1: 2.0}   # m from the centreline the parking line must be (two-way / one-way)
HALF = 1.0          # m: footprint within this of the parking line counts for the row
LOTS = ROOT / "data/raw" / f"osm_lots_{ROI_NAME}.json"
KERB, LOT, GARAGE = 0, 1, 2
GARAGE_CAP = (40, 150)  # places per public garage: when OSM has no capacity, at most


def resample(p, step):
    """polyline (n, 2) -> points every `step` (same units) with unit directions"""
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] < step:
        return None, None, cum[-1]
    s = np.arange(0, cum[-1], step)
    x, y = np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])
    pts = np.stack([x, y], 1)
    d = np.gradient(pts, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    return pts, d, cum[-1]


def runs(mask):
    """[start, end) index pairs of True runs"""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def fetch_lots():
    bb = bbox(ROI, 20.0)
    q = (f'[out:json][timeout:120];(way["amenity"="parking"]({bb});nwr["amenity"="parking_entrance"]({bb});'
         f'nwr["amenity"="parking"]["parking"~"underground|multi-storey"]({bb}););out tags geom;')
    overpass(q, LOTS)


def lot_places(grid, x0, y0, origin, kerb):
    """stalls inside OSM's surface car parks where the LiDAR saw cars, and public garages"""
    els = json.loads(LOTS.read_text())["elements"]
    h, w = grid.shape
    vox = lambda lon, lat: ((np.array(to_l72.transform(lon, lat)) - origin) / VS)
    mask = Image.new("1", (w, h), 0)
    d = ImageDraw.Draw(mask)
    lots = 0
    for e in els:
        t = e.get("tags", {})
        if e["type"] != "way" or t.get("amenity") != "parking" or t.get("parking", "surface") not in ("surface", "yes"):
            continue                                       # (street-side / lane parking: the kerb rows)
        pts = [tuple(vox(g["lon"], g["lat"]) - [x0, y0]) for g in e.get("geometry", [])]
        if len(pts) >= 3:
            d.polygon(pts, fill=1)
            lots += 1
    inside = grid & np.array(mask, bool)
    lab, n = ndimage.label(inside, structure=np.ones((3, 3)))
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        yy, xx = np.nonzero(lab[sl] == k)
        if len(yy) < 12:
            continue
        P = np.c_[xx + sl[1].start + x0, yy + sl[0].start + y0].astype(float)
        c = P.mean(0)
        ev, V = np.linalg.eigh(np.cov((P - c).T))
        u, v = V[:, 1], V[:, 0]                            # major, minor axis
        pu, pv = (P - c) @ u, (P - c) @ v
        lu = (np.percentile(pu, 98) - np.percentile(pu, 2)) * VS + VS
        lv = (np.percentile(pv, 98) - np.percentile(pv, 2)) * VS + VS
        if lu < 2.5:
            continue
        if lu <= 6.5 and lv <= 3.0:                        # one car: along its own long axis
            q = c + u * (np.percentile(pu, 98) + np.percentile(pu, 2)) / 2
            out.append([q[0], q[1], np.arctan2(u[1], u[0]) + (np.pi if len(out) % 2 else 0)])
            continue
        mu, mv = (np.percentile(pu, 98) + np.percentile(pu, 2)) / 2, (np.percentile(pv, 98) + np.percentile(pv, 2)) / 2
        if lv >= 3.2:                                      # cars side by side, in one or more rows
            rows = max(1, int(round(lv / 5.0)))
            n = max(1, int(round(lu / 2.5)))
            for r in range(rows):
                ov = mv + ((r + 0.5) / rows - 0.5) * lv / VS
                hd = v if r % 2 == 0 else -v               # (back to back rows face apart)
                for i in range(n):
                    ou = mu + ((i + 0.5) / n - 0.5) * lu / VS
                    q = c + u * ou + v * ov
                    out.append([q[0], q[1], np.arctan2(hd[1], hd[0])])
        else:                                              # nose to tail
            n = max(1, int(round(lu / 5.4)))
            for i in range(n):
                ou = mu + ((i + 0.5) / n - 0.5) * lu / VS
                q = c + u * ou + v * mv
                out.append([q[0], q[1], np.arctan2(u[1], u[0])])
    if out and len(kerb):                                  # (not on top of a kerb place)
        dk, _ = cKDTree(np.array(kerb)[:, :2]).query(np.array(out)[:, :2])
        out = [o for o, dd in zip(out, dk) if dd * VS > 2.5]
    spots = [[round(float(x), 1), round(float(y), 1), round(float(hd), 3), LOT] for x, y, hd in out]
    # public garages: places at their entrances (or the garage itself where no entrance is mapped)
    pub = lambda t: t.get("access", "yes") not in ("private", "no", "permit")
    at = lambda e: (e["lon"], e["lat"]) if "lon" in e else (e["bounds"]["minlon"] / 2 + e["bounds"]["maxlon"] / 2,
                                                           e["bounds"]["minlat"] / 2 + e["bounds"]["maxlat"] / 2)
    garages = [(vox(*at(e)), e["tags"]) for e in els if e.get("tags", {}).get("parking") in ("underground", "multi-storey")
               and e["tags"].get("amenity") == "parking" and pub(e["tags"])]
    entrances = [vox(*at(e)) for e in els if e.get("tags", {}).get("amenity") == "parking_entrance" and pub(e["tags"])
                 and e["tags"].get("parking", "underground") in ("underground", "multi-storey")]
    ng = 0
    for g, t in garages:
        doors = [p for p in entrances if np.hypot(*(p - g)) * VS < 150] or [g]
        cap = int(t["capacity"]) if t.get("capacity", "").isdigit() else GARAGE_CAP[0]
        cap = min(cap, GARAGE_CAP[1])
        if not (0 <= g[0] - x0 < w and 0 <= g[1] - y0 < h):
            continue
        ng += 1
        for i in range(cap):
            p = doors[i % len(doors)]
            spots.append([round(float(p[0]), 1), round(float(p[1]), 1), 0.0, GARAGE])
    print(f"{lots} surface car parks -> {sum(1 for s in spots if s[3] == LOT)} stalls where the LiDAR saw cars; "
          f"{ng} public garages -> {sum(1 for s in spots if s[3] == GARAGE)} places")
    return spots


def main():
    f = np.load(FOOT).astype(np.int64)
    x0, y0 = f.min(0) - 2
    w, h = f.max(0) - f.min(0) + 5
    grid = np.zeros((h, w), bool)
    grid[f[:, 1] - y0, f[:, 0] - x0] = True

    def occ(q):   # q: (..., 2) voxel coordinates -> footprint there
        ix, iy = np.rint(q[..., 0]).astype(np.int64) - x0, np.rint(q[..., 1]).astype(np.int64) - y0
        ok = (ix >= 0) & (iy >= 0) & (ix < w) & (iy < h)
        out = np.zeros(ix.shape, bool)
        out[ok] = grid[iy[ok], ix[ok]]
        return out

    roads = json.loads(ROADS.read_text())
    edges = roads["edges"]
    # how far each junction's crossing streets reach: the kerb is ~1 m inside the pavement walking line
    # (pavement offsets from the LiDAR facades, roads.py); degree-2 nodes are bends, not junctions
    deg, reach = {}, {}
    for e in edges:
        if e[E_KIND] != KIND_ROAD:
            continue
        half = max(e[E_PAVEMENT]) - 1.0 if max(e[E_PAVEMENT]) > 0 else 5.0
        for n in (e[E_FROM], e[E_TO]):
            deg[n] = deg.get(n, 0) + 1
            reach.setdefault(n, []).append(half)
    def margin(node, own_half):
        if deg.get(node, 0) <= 2:
            return 2.0
        others = sorted(reach[node], reverse=True)
        cross = others[0] if others[0] != own_half or len(others) == 1 else others[1]
        return cross + JUNCTION
    stops = np.array([[x, y] for x, y, *_ in json.loads(TRANSIT.read_text())["stops"]], float) if TRANSIT.exists() else np.zeros((0, 2))
    spots, rows, used = [], 0, 0
    offs = np.arange(1.5, 12.01, 0.5)                    # m from the centreline
    for e in edges:
        if e[E_KIND] != KIND_ROAD:
            continue
        oneway, pave = e[E_ONEWAY], e[E_PAVEMENT]
        osm = e[E_PARKING]
        p = np.array(e[E_XYZ], float).reshape(-1, 3)[:, :2]
        pts, d, L = resample(p, 1.0)                      # every voxel (0.5 m)
        own = max(pave) - 1.0 if max(pave) > 0 else 5.0
        m0, m1 = margin(e[E_FROM], own), margin(e[E_TO], own)
        if pts is None or L * VS < m0 + m1 + MIN_ROW:
            continue
        left = np.stack([-d[:, 1], d[:, 0]], 1)           # x east, y north
        s_m = np.arange(len(pts)) * VS
        inner = (s_m > m0) & (s_m < L * VS - m1)
        for m, _ in e[E_CROSSWALKS]:                      # mapped crosswalks
            inner &= np.abs(s_m - m) > CROSSWALK
        if len(stops):                                    # bus / tram stops
            near = np.min(np.hypot(*(pts[:, None, :] - stops[None, :, :]).transpose(2, 0, 1)), axis=1) * VS
            inner &= near > STOP_CLEAR
        for side, sgn in ((0, 1.0), (1, -1.0)):
            if osm[side] == 1:                            # OSM: no parking on this side
                continue
            lim = pave[side] if pave[side] > 0 else 9.0   # the pavement (or ~9 m) bounds the parking side
            cand = offs[(offs >= LANE_CLEAR[1 if oneway else 0]) & (offs <= lim)]
            if not len(cand):
                continue
            # footprint across the edge at each candidate distance: (along, across)
            q = pts[:, None, :] + sgn * left[:, None, :] * (cand[None, :, None] / VS)
            o = occ(q) & inner[:, None]
            prof = o.sum(0)
            tagged = osm[side] >= 2                       # OSM: there is parking on this side
            if prof.max() < MIN_ROW / VS and not tagged:  # nobody parks on this side
                continue
            if prof.max() >= MIN_ROW / VS:
                k = int(np.argmax(prof))
                line = cand[k]
            else:                                         # tagged, no LiDAR evidence: ~2.2 m inside the pavement line
                line = float(np.clip((pave[side] if pave[side] > 0 else 6.0) - 2.2, cand[0], cand[-1]))
            near = np.abs(cand - line) <= HALF
            along = o[:, near].any(1) if not tagged else inner.copy()
            # close gaps between parked cars; extend rows that end near the junction margins
            along = ndimage.binary_closing(np.pad(along, int(GAP / VS)), structure=np.ones(int(GAP / VS) + 1, bool))
            along = along[int(GAP / VS):-int(GAP / VS)] & inner
            idx = np.nonzero(inner)[0]
            if along.any() and len(idx):
                lo, hi = idx[0], idx[-1]
                on = np.nonzero(along)[0]
                if on[0] - lo < REACH / VS: along[lo:on[0]] = True
                if hi - on[-1] < REACH / VS: along[on[-1]:hi + 1] = True
            along &= inner                                 # (crosswalks and stops stay clear)
            deep = o.sum(1)                                # how far across the kerb the footprint reaches
            for a, b in runs(along):
                length = (b - a) * VS
                if length < MIN_ROW:
                    continue
                perp = np.median(deep[a:b][deep[a:b] > 0]) * VS > 3.6 if (deep[a:b] > 0).any() else False
                if osm[side] >= 3:
                    perp = True                           # OSM: diagonal / perpendicular (drawn perpendicular)
                elif osm[side] == 2:
                    perp = False
                pitch = PERP if perp else PARALLEL
                n = max(1, int(round(length / pitch)))
                rows += 1
                for i in range(n):
                    t = a + (i + 0.5) * (b - a) / n        # sample index (fractional)
                    j = min(len(pts) - 1, int(round(t)))
                    c = pts[j] + sgn * left[j] * (line / VS)
                    dx, dy = d[j]
                    if perp:
                        hd = np.arctan2(sgn * left[j][1], sgn * left[j][0])   # nose towards the kerb
                    else:
                        # driving direction of this side: right side = edge direction; left side faces
                        # the other way on two-way streets, the same way on one-way streets
                        forward = side == 1 or oneway
                        hd = np.arctan2(dy, dx) if forward else np.arctan2(-dy, -dx)
                    spots.append([round(float(c[0]), 1), round(float(c[1]), 1), round(float(hd), 3), KERB])
                used += b - a
    if "--refetch" in sys.argv or not LOTS.exists():
        fetch_lots()
    spots += lot_places(grid, x0, y0, np.array(Model(ROI_NAME).man["origin"][:2]), spots)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"spots": spots, "fields": ["x", "y", "heading (rad, x east / y north)", "kind (0 kerb, 1 car park, 2 garage)"]},
                              separators=(",", ":")))
    print(f"{len(f):,} footprint columns -> {rows:,} kerbside rows, {used * VS / 1000:.1f} km of kerb, "
          f"{len(spots):,} parking places -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
