"""Movement network from OpenStreetMap: drivable roads, tram and rail tracks as a graph
(intersections + edges), with heights from the voxel ground -> viewer/public/roads/<roi>.json.

Edges are split at every node shared by two or more ways. Per edge: start/end node, kind, one-way,
speed (km/h), lanes, the centreline as voxel coordinates (x east, y north, z = ground surface),
resampled every SAMPLE m, the road rank (priority) and traffic controls along it. One-way edges point
in the driving direction. Nodes: x, y, z, 1 if the intersection itself carries traffic signals, and 1
if it is where an edge leaves the region (entry / exit for through traffic). Edges are clipped to the
region. Double tram tracks get a direction (trams keep right); single track stays two-way.

Usage: uv run python roads.py [roi-name] [--refetch]
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
RAW = ROOT / "data/raw" / f"osm_roads_{ROI_NAME}.json"
OUT = ROOT / "viewer/public/roads" / f"{ROI_NAME}.json"

ROADS = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street",
         "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"]
KIND = {"road": 0, "tram": 1, "rail": 2}
# road class -> rank: who has priority at an unsigned intersection (Belgium: otherwise from the right)
RANK = {"motorway": 6, "trunk": 6, "primary": 5, "secondary": 4, "tertiary": 3, "unclassified": 2,
        "residential": 2, "living_street": 1}
# traffic controls on nodes: code
SIGNALS, GIVE_WAY, STOP = 1, 2, 3
DEFAULT_SPEED = {"living_street": 20, "primary": 50, "trunk": 50, "motorway": 90}   # else 30 (Brussels zone 30)
SAMPLE = 2.0      # m between height samples
MERGE = 15.0      # m; road intersections joined by a shorter edge are one intersection
MARGIN = 20.0     # m around the region

to_wgs = Transformer.from_crs(31370, 4326, always_xy=True)
to_l72 = Transformer.from_crs(4326, 31370, always_xy=True)


def fetch():
    corners = [to_wgs.transform(x, y) for x in (ROI["xmin"] - MARGIN, ROI["xmax"] + MARGIN)
               for y in (ROI["ymin"] - MARGIN, ROI["ymax"] + MARGIN)]
    lo = (min(c[1] for c in corners), min(c[0] for c in corners))
    hi = (max(c[1] for c in corners), max(c[0] for c in corners))
    bb = f"{lo[0]:.6f},{lo[1]:.6f},{hi[0]:.6f},{hi[1]:.6f}"
    q = (f'[out:json][timeout:120];(way["highway"~"^({"|".join(ROADS)})$"]({bb});'
         f'way["railway"~"^(tram|rail|light_rail)$"]({bb}););out body;>;out body qt;')
    req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                                 data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "bxl-vice-voxel/0.1"})
    for i in range(5):            # the public Overpass server is often briefly overloaded (502/504)
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                RAW.write_bytes(r.read())
            return
        except Exception as e:
            print(f"overpass: {e}; retrying")
            time.sleep(10 * (i + 1))
    raise SystemExit("overpass unavailable")


def speed(t):
    s = t.get("maxspeed", "")
    if s.isdigit():
        return int(s)
    if t.get("railway") == "tram":
        return 30
    if t.get("railway"):
        return 60
    return DEFAULT_SPEED.get(t.get("highway", "").replace("_link", ""), 30)


def consolidate(edges, node_xy, node_sig, vs):
    """Merge intersections joined by short road edges (several OSM nodes in one real crossing) into one
    node at their centroid; drop the short edges; snap the other edges' ends to the merged node."""
    parent = list(range(len(node_xy)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def length(e):
        p = np.array(e[6]).reshape(-1, 3)[:, :2] * vs
        return float(np.hypot(*np.diff(p, axis=0).T).sum())

    for e in edges:
        if e[2] == KIND["road"] and e[0] != e[1] and length(e) < MERGE:
            parent[find(e[0])] = find(e[1])
    groups = {}
    for i in range(len(node_xy)):
        groups.setdefault(find(i), []).append(i)
    new_ix, xy, sig = {}, [], []
    for members in groups.values():
        k = len(xy)
        xy.append(list(np.mean([node_xy[i] for i in members], axis=0)))
        sig.append(max(node_sig[i] for i in members))
        for i in members:
            new_ix[i] = k
    out = []
    for e in edges:
        a, b = new_ix[e[0]], new_ix[e[1]]
        if a == b and e[2] == KIND["road"] and length(e) < MERGE:
            continue                                         # inside a merged intersection
        pts = list(e[6])
        pts[0:3] = [round(float(c), 1) for c in xy[a]]
        pts[-3:] = [round(float(c), 1) for c in xy[b]]
        out.append([a, b, *e[2:6], pts, *e[7:]])
    print(f"consolidated {len(node_xy)} -> {len(xy)} nodes, {len(edges)} -> {len(out)} edges")
    return out, xy, sig


def edge_length(pts, vs):
    p = np.array(pts).reshape(-1, 3)[:, :2] * vs
    return np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]


def clip(edges, node_xy, node_sig, size, vs):
    """Keep only the parts of edges inside the region; where an edge leaves it, a boundary node."""
    boundary = [0] * len(node_xy)
    out = []
    for e in edges:
        p = np.array(e[6]).reshape(-1, 3)
        inside = (p[:, 0] >= 1) & (p[:, 1] >= 1) & (p[:, 0] < size[0] - 1) & (p[:, 1] < size[1] - 1)
        if inside.all():
            out.append(e)
            continue
        cum = edge_length(e[6], vs)
        i = 0
        while i < len(p):
            if not inside[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(p) and inside[j + 1]:
                j += 1
            if j > i:
                ends = []
                for k, orig in ((i, e[0]), (j, e[1])):
                    if k in (0, len(p) - 1):
                        ends.append(orig)
                    else:                                    # new boundary node
                        node_xy.append([round(float(c), 1) for c in p[k]])
                        node_sig.append(0)
                        boundary.append(1)
                        ends.append(len(node_xy) - 1)
                ctl = [[round(m - cum[i], 1), c, d] for m, c, d in e[8] if cum[i] <= m <= cum[j]]
                out.append([ends[0], ends[1], *e[2:6], [round(float(c), 1) for c in p[i:j + 1].ravel()], e[7], ctl])
            i = j + 1
    print(f"clipped to the region: {len(edges)} -> {len(out)} edges, {sum(boundary)} boundary nodes")
    return out, node_xy, node_sig, boundary


def tram_directions(edges, vs):
    """OSM maps both tracks of a tram line but not their direction: on double track, trams keep right
    (the other track is on their left). Single track stays two-way."""
    trams = [e for e in edges if e[2] == KIND["tram"] and not e[3]]
    pts = {id(e): np.array(e[6]).reshape(-1, 3)[:, :2] for e in trams}
    n = 0
    for e in trams:
        p = pts[id(e)]
        if len(p) < 3:
            continue
        k = len(p) // 2
        P, D = p[k], p[k + 1] - p[k - 1]
        D = D / (np.linalg.norm(D) or 1)
        best = None
        for o in trams:
            if o is e:
                continue
            q = pts[id(o)]
            d = np.hypot(*(q - P).T)
            i = int(np.argmin(d))
            if not (1.5 / vs <= d[i] <= 5.0 / vs) or len(q) < 2:
                continue
            Do = q[min(i + 1, len(q) - 1)] - q[max(i - 1, 0)]
            if abs(np.dot(Do, D)) / (np.linalg.norm(Do) or 1) < 0.8:
                continue
            if best is None or d[i] < best[0]:
                best = (d[i], q[i])
        if best is None:
            continue
        rel = best[1] - P
        left = D[0] * rel[1] - D[1] * rel[0] > 0           # other track on the left (x east, y north)
        if not left:                                       # reverse: travel the other way
            L = edge_length(e[6], vs)[-1]
            e[0], e[1] = e[1], e[0]
            e[6] = [c for i in range(len(e[6]) - 3, -1, -3) for c in e[6][i:i + 3]]
            e[8] = sorted([[round(L - m, 1), c, -d] for m, c, d in e[8]])
        e[3] = 1
        n += 1
    print(f"tram tracks: {n} of {len(trams)} given a direction (double track)")


def build():
    els = json.loads(RAW.read_text())["elements"]
    nodes = {e["id"]: to_l72.transform(e["lon"], e["lat"]) for e in els if e["type"] == "node"}
    ntags = {e["id"]: e.get("tags", {}) for e in els if e["type"] == "node"}

    def control(nid):
        t = ntags.get(nid, {})
        if t.get("highway") == "traffic_signals" or t.get("crossing") == "traffic_signals":
            return SIGNALS
        return {"give_way": GIVE_WAY, "stop": STOP}.get(t.get("highway"), 0)
    ways = [e for e in els if e["type"] == "way" and e["tags"].get("tunnel") not in ("yes", "building_passage")
            and e["tags"].get("access") not in ("no", "private") and e["tags"].get("service") is None]
    use = {}
    for w in ways:
        for n in w["nodes"]:
            use[n] = use.get(n, 0) + 1

    m = Model(ROI_NAME)
    X0, Y0 = m.man["origin"][:2]
    vs = m.vs

    def ground(px, py, bridge):
        """ground surface (voxels) at a point: median over a small patch of the top ground voxel per column
        (bridges: top of anything solid)"""
        cx, cy = int(px), int(py)
        zs = []
        for dx in (-2, 0, 2):
            for dy in (-2, 0, 2):
                c = m.column(cx + dx, cy + dy)
                if c is None:
                    continue
                k = np.flatnonzero(c != 0 if bridge else c == 2)
                if len(k):
                    zs.append(k.max() + 1)
        return float(np.median(zs)) if zs else np.nan

    node_ix, node_xy, node_sig = {}, [], []
    edges, stats = [], {"road": 0, "tram": 0, "rail": 0}
    for w in ways:
        t = w["tags"]
        kind = "road" if "highway" in t else ("tram" if t["railway"] == "tram" else "rail")
        ow = t.get("oneway")
        oneway = 1 if ow in ("yes", "1", "true", "-1") or t.get("junction") == "roundabout" else 0
        ids = w["nodes"][::-1] if ow == "-1" else w["nodes"]
        bridge = t.get("bridge") not in (None, "no") or t.get("layer", "0") not in ("0", "-1", "-2")
        lanes = int(t["lanes"]) if t.get("lanes", "").isdigit() else (1 if oneway else 2)
        # split at shared nodes
        cuts = [0] + [i for i in range(1, len(ids) - 1) if use[ids[i]] > 1] + [len(ids) - 1]
        for a, b in zip(cuts, cuts[1:]):
            seg = ids[a:b + 1]
            xy = np.array([nodes[n] for n in seg])
            v = (xy - [X0, Y0]) / vs                       # voxel coordinates
            inside = (v[:, 0] >= 0) & (v[:, 1] >= 0) & (v[:, 0] < m.size[0]) & (v[:, 1] < m.size[1])
            if not inside.any():
                continue
            # resample every SAMPLE m
            d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            n = max(1, int(np.ceil(d[-1] / SAMPLE)))
            s = np.unique(np.r_[np.linspace(0, d[-1], n + 1), d])
            p = np.c_[np.interp(s, d, v[:, 0]), np.interp(s, d, v[:, 1])]
            z = np.array([ground(px, py, bridge) for px, py in p])
            if np.isnan(z).all():
                continue
            ok = ~np.isnan(z)
            z = np.interp(np.arange(len(z)), np.flatnonzero(ok), z[ok])
            z = np.convolve(np.pad(z, 2, mode="edge"), np.ones(5) / 5, mode="valid")    # smooth kerb steps
            for nid in (seg[0], seg[-1]):
                if nid not in node_ix:
                    node_ix[nid] = len(node_xy)
                    node_xy.append(None)
                    node_sig.append(1 if control(nid) == SIGNALS else 0)
            # controls on the nodes between the ends: [m from the edge start, code, direction]
            # (direction: 1 along the edge, -1 against it, 0 both; OSM's direction tag is relative to the way)
            ctl = []
            for i in range(1, len(seg) - 1):
                c = control(seg[i])
                if c:
                    dtag = ntags[seg[i]].get("direction", "")
                    dr = {"forward": 1, "backward": -1}.get(dtag, 0) * (-1 if ow == "-1" else 1)
                    ctl.append([round(float(d[i]), 1), c, dr])
            node_xy[node_ix[seg[0]]] = [*p[0], z[0]]
            node_xy[node_ix[seg[-1]]] = [*p[-1], z[-1]]
            rank = RANK.get(t.get("highway", "").replace("_link", ""), 0)
            edges.append([node_ix[seg[0]], node_ix[seg[-1]], KIND[kind], oneway, speed(t), lanes,
                          [round(float(c), 1) for c in np.c_[p, z].ravel()], rank, ctl])
            stats[kind] += d[-1]
    edges, node_xy, node_sig = consolidate(edges, node_xy, node_sig, vs)
    edges, node_xy, node_sig, boundary = clip(edges, node_xy, node_sig, m.size, vs)
    tram_directions(edges, vs)
    out = {"build": m.man["build"],
           "nodes": [[*(round(float(c), 1) for c in xyz), sig, b] for xyz, sig, b in zip(node_xy, node_sig, boundary)],
           "edges": edges, "fields": ["from", "to", "kind(0 road,1 tram,2 rail)", "oneway", "kmh", "lanes", "xyz...",
                                      "rank", "controls [m, 1 signals|2 give way|3 stop, dir]"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    nctl = sum(len(e[8]) for e in edges)
    print(f"{len(node_xy)} nodes ({sum(node_sig)} signalled), {len(edges)} edges ({nctl} controls); road {stats['road'] / 1000:.1f} km, tram {stats['tram'] / 1000:.1f} km, "
          f"rail {stats['rail'] / 1000:.1f} km -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    if "--refetch" in sys.argv or not RAW.exists():
        fetch()
    build()
