"""Street lights: Mapillary's map features (lamps triangulated from street-level images), snapped to
the voxel model -> viewer/public/lights/<roi>.json.

  fetch    all object--street-light map features in the region (cached in data/raw/)
  snap     shift by the global Mapillary->Lambert 72 offset (streetview.py calibration), merge
           duplicates, find the ground under each lamp and the top of its pole in the L0 voxels

Usage: uv run python lights.py [roi-name] [--refetch]
Needs MAPILLARY_TOKEN in ../.env. Map data © Mapillary contributors, CC-BY-SA.
"""
import json
import sys
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
CFG = tomllib.loads((ROOT / "config.toml").read_text())
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
ROI_NAME = ARGS[0] if ARGS else CFG["default"]
ROI = CFG["roi"][ROI_NAME]
RAW = ROOT / "data/raw" / f"mapillary_lights_{ROI_NAME}.json"
CAL = ROOT / "data/cache/streetview" / ROI_NAME / "calibration.json"
TILES = ROOT / "viewer/public/tiles" / ROI_NAME
OUT = ROOT / "viewer/public/lights" / f"{ROI_NAME}.json"

QUERY = 250.0      # m per API request
MERGE = 4.0        # m; detections closer than this are one lamp
SEARCH = 1.5       # m; radius around a lamp in which its pole is looked for
HEAD_MIN, HEAD_MAX = 3.0, 12.0   # m above ground for a lamp head
HEAD_DEFAULT = 5.0               # m; wall-mounted lamps and poles the LiDAR missed
MIN_SCORE = 0.5    # same as streetview.py

to_wgs = Transformer.from_crs(31370, 4326, always_xy=True)
to_l72 = Transformer.from_crs(4326, 31370, always_xy=True)


def token():
    return next(l.split("=", 1)[1].strip().strip("'\"") for l in (ROOT / ".env").read_text().splitlines()
                if l.startswith("MAPILLARY_TOKEN"))


def fetch():
    tok = token()
    x0, y0, x1, y1 = ROI["xmin"], ROI["ymin"], ROI["xmax"], ROI["ymax"]
    cells = [(x, y) for x in np.arange(x0, x1, QUERY) for y in np.arange(y0, y1, QUERY)]

    def query(c):
        a = to_wgs.transform(c[0], c[1])
        b = to_wgs.transform(min(c[0] + QUERY, x1), min(c[1] + QUERY, y1))
        u = (f"https://graph.mapillary.com/map_features?bbox={a[0]:.6f},{a[1]:.6f},{b[0]:.6f},{b[1]:.6f}"
             "&object_values=object--street-light&fields=id,geometry,last_seen_at&limit=2000")
        req = urllib.request.Request(u, headers={"Authorization": f"OAuth {tok}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["data"]

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(query, cells))
    if any(len(r) >= 2000 for r in res):
        print("warning: a query hit the 2000 cap; lower QUERY")
    feats = {d["id"]: d for r in res for d in r}
    RAW.write_text(json.dumps(list(feats.values())))
    print(f"{len(feats)} street-light features")


class Model:
    """L0 voxel classes, tile by tile on demand."""
    def __init__(self):
        self.man = json.loads((TILES / "manifest.json").read_text())
        L0 = self.man["levels"][0]
        self.vs, self.size = L0["voxelSize"], L0["size"]
        self.files = {(t["tx"], t["ty"]): t["file"] for t in L0["tiles"]}
        self.cache = {}

    def tile(self, tx, ty):
        if (tx, ty) not in self.cache:
            occ = np.zeros((64, 64, self.size[2]), np.uint8)
            f = self.files.get((tx, ty))
            if f:
                v = np.frombuffer((TILES / f).read_bytes(), np.uint8).reshape(-1, 8)
                v = v[(v[:, 7] & 2) == 0]
                occ[v[:, 0] - 1, v[:, 1] - 1, v[:, 2]] = np.maximum(v[:, 6], 1)
            self.cache[(tx, ty)] = occ
        return self.cache[(tx, ty)]

    def column(self, x, y):
        if not (0 <= x < self.size[0] and 0 <= y < self.size[1]):
            return None
        return self.tile(x // 64, y // 64)[x % 64, y % 64]


def snap():
    feats = json.loads(RAW.read_text())
    cal = json.loads(CAL.read_text())
    off = np.median([c["offset"] for c in cal.values() if c["score"] >= MIN_SCORE], 0)
    xy = np.array([to_l72.transform(*f["geometry"]["coordinates"]) for f in feats]) + off[:2]

    # merge duplicate detections (greedy, around the earliest-listed point)
    used = np.zeros(len(xy), bool)
    lamps = []
    for i in range(len(xy)):
        if used[i]:
            continue
        near = ~used & (np.hypot(*(xy - xy[i]).T) < MERGE)
        used |= near
        lamps.append(xy[near].mean(0))

    m = Model()
    X0, Y0 = m.man["origin"][:2]
    vs, r = m.vs, int(round(SEARCH / m.vs))
    out, on_pole = [], 0
    for lx, ly in lamps:
        cx, cy = int((lx - X0) / vs), int((ly - Y0) / vs)
        # ground: lowest ground voxel near the lamp
        g = None
        for R in (2, 6, 12):
            zs = [np.flatnonzero(c == 2) for dx in range(-R, R + 1) for dy in range(-R, R + 1)
                  if (c := m.column(cx + dx, cy + dy)) is not None]
            zs = [z.max() for z in zs if len(z)]
            if zs:
                g = int(np.median(zs))
                break
        if g is None:
            continue
        # pole: a run of unclassified voxels that starts near the ground and goes up without
        # gaps longer than one voxel; the lamp head is its top (the tallest pole in the search radius)
        best = None
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                c = m.column(cx + dx, cy + dy)
                if c is None or dx * dx + dy * dy > r * r:
                    continue
                z, top, gap = g + 1, None, 0
                while z < g + int(HEAD_MAX / vs) + 2 and z < len(c):
                    if c[z] == 1:
                        top, gap = z, 0
                    elif top is None and z > g + int(1.5 / vs):
                        break               # no pole starting near the ground
                    else:
                        gap += 1
                        if gap > 1:
                            break
                    z += 1
                if top is not None and HEAD_MIN / vs <= top - g <= HEAD_MAX / vs and (best is None or top > best[2]):
                    best = (cx + dx, cy + dy, int(top))
        if best:
            on_pole += 1
            hx, hy, hz = best
        else:
            hx, hy, hz = cx, cy, g + int(HEAD_DEFAULT / vs)
            # wall-mounted lamps are often placed inside the facade: move to the nearest open voxel
            c = m.column(hx, hy)
            if c is not None and c[hz]:
                free = [(dx * dx + dy * dy, dx, dy) for dx in range(-6, 7) for dy in range(-6, 7)
                        if (c2 := m.column(hx + dx, hy + dy)) is not None and not c2[hz]]
                if free:
                    _, dx, dy = min(free)
                    hx, hy = hx + dx, hy + dy
        out.append([hx, hy, g + 1, hz])       # voxel coords: column, ground surface, head voxel
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"build": m.man["build"], "lamps": out}, separators=(",", ":")))
    print(f"{len(feats)} detections -> {len(lamps)} lamps ({len(out)} on the model, {on_pole} on a LiDAR pole); "
          f"offset {off[:2].round(2).tolist()} -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    if "--refetch" in sys.argv or not RAW.exists():
        fetch()
    snap()
