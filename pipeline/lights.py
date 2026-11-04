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
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import ROOT, mapillary_token, region, to_l72, to_wgs
from model import Model

ROI_NAME, ROI = region()
RAW = ROOT / "data/raw" / f"mapillary_lights_{ROI_NAME}.json"
CAL = ROOT / "data/cache/streetview" / ROI_NAME / "calibration.json"
OUT = ROOT / "viewer/public/lights" / f"{ROI_NAME}.json"

QUERY = 250.0      # m per API request
MERGE = 4.0        # m; detections closer than this are one lamp
SEARCH = 1.5       # m; radius around a lamp in which its pole is looked for
HEAD_MIN, HEAD_MAX = 3.0, 12.0   # m above ground for a lamp head
HEAD_DEFAULT = 5.0               # m; wall-mounted lamps and poles the LiDAR missed
MIN_SCORE = 0.5    # same as streetview.py

def fetch():
    tok = mapillary_token()
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


def snap():
    feats = json.loads(RAW.read_text())
    # the global Mapillary offset (streetview.py calibrate); another region's is fine: it is a systematic bias
    cal_f = CAL if CAL.exists() else next(iter(sorted(CAL.parent.parent.glob("*/calibration.json"))), None)
    cal = json.loads(cal_f.read_text()) if cal_f else {}
    good = [c["offset"] for c in cal.values() if c["score"] >= MIN_SCORE]
    off = np.median(good, 0) if good else np.zeros(3)
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

    lamps.sort(key=lambda q: (round(q[1] / 250), q[0]))   # spatial order (bounded tile cache)
    m = Model(ROI_NAME)
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
