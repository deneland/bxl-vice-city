"""Movement network from OpenStreetMap: drivable roads, tram and rail tracks as a graph
(intersections + edges), with heights from the voxel ground -> viewer/public/roads/<roi>.json.

Edges are split at every node shared by two or more ways. Per edge: start/end node, kind, one-way,
speed (km/h), lanes, the centreline as voxel coordinates (x east, y north, z = ground surface),
resampled every SAMPLE m, the road rank (priority) and traffic controls along it. One-way edges point
in the driving direction. Nodes: x, y, z, 1 if the junction is signalled (assign_signals), and 1
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
RAW_PATHS = ROOT / "data/raw" / f"osm_paths_{ROI_NAME}.json"
OUT = ROOT / "viewer/public/roads" / f"{ROI_NAME}.json"

ROADS = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street",
         "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"]
KIND = {"road": 0, "tram": 1, "rail": 2}
BUILDING = 6
# road class -> rank: who has priority at an unsigned intersection (Belgium: otherwise from the right)
RANK = {"motorway": 6, "trunk": 6, "primary": 5, "secondary": 4, "tertiary": 3, "unclassified": 2,
        "residential": 2, "living_street": 1}
# traffic controls on nodes: code
SIGNALS, GIVE_WAY, STOP = 1, 2, 3
CALMING = 5                     # speed bump / cushion / raised table (cars slow down over it)
CALM_KINDS = {"cushion", "table", "hump", "bump", "yes", "dip"}
DEFAULT_SPEED = {"living_street": 20, "primary": 50, "trunk": 50, "motorway": 90}   # else 30 (Brussels zone 30)
SAMPLE = 2.0      # m between height samples
MERGE = 15.0      # m; road intersections joined by a shorter edge are one intersection
CLUSTER = 20.0    # m; ...unless the merged junction would be wider than this
MARGIN = 20.0     # m around the region

to_wgs = Transformer.from_crs(31370, 4326, always_xy=True)
to_l72 = Transformer.from_crs(4326, 31370, always_xy=True)


def overpass(q, dest):
    req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                                 data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "bxl-vice-voxel/0.1"})
    for i in range(5):            # the public Overpass server is often briefly overloaded (502/504)
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                dest.write_bytes(r.read())
            return
        except Exception as e:
            print(f"overpass: {e}; retrying")
            time.sleep(10 * (i + 1))
    raise SystemExit("overpass unavailable")


def bbox():
    corners = [to_wgs.transform(x, y) for x in (ROI["xmin"] - MARGIN, ROI["xmax"] + MARGIN)
               for y in (ROI["ymin"] - MARGIN, ROI["ymax"] + MARGIN)]
    return (f"{min(c[1] for c in corners):.6f},{min(c[0] for c in corners):.6f},"
            f"{max(c[1] for c in corners):.6f},{max(c[0] for c in corners):.6f}")


def fetch_paths():
    """footways, paths, pedestrian streets, steps (parks, squares); pavements and crossings are generated"""
    overpass(f'[out:json][timeout:120];way["highway"~"^(footway|path|pedestrian|steps)$"]({bbox()});'
             'out body;>;out skel qt;', RAW_PATHS)


def walk_paths(m, X0, Y0, vs):
    """Footpaths as polylines with ground heights, split at shared nodes: [[node a, node b], xyz...]."""
    els = json.loads(RAW_PATHS.read_text())["elements"]
    nodes = {e["id"]: to_l72.transform(e["lon"], e["lat"]) for e in els if e["type"] == "node"}
    ways = [e for e in els if e["type"] == "way" and e["tags"].get("footway") not in ("sidewalk", "crossing")
            and e["tags"].get("access") not in ("no", "private")]
    use = {}
    for w in ways:
        for n in w["nodes"]:
            use[n] = use.get(n, 0) + 1
    ids, out = {}, []
    def ground(px, py):
        zs = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                c = m.column(int(px) + dx, int(py) + dy)
                if c is not None:
                    k = np.flatnonzero(c == 2)
                    if len(k):
                        zs.append(k.max() + 1)
        return float(np.median(zs)) if zs else np.nan
    for w in ways:
        seq = w["nodes"]
        cuts = [0] + [i for i in range(1, len(seq) - 1) if use[seq[i]] > 1] + [len(seq) - 1]
        for a, b in zip(cuts, cuts[1:]):
            seg = seq[a:b + 1]
            xy = np.array([nodes[n] for n in seg])
            v = (xy - [X0, Y0]) / vs
            if not ((v[:, 0] >= 0) & (v[:, 1] >= 0) & (v[:, 0] < m.size[0]) & (v[:, 1] < m.size[1])).all():
                continue
            d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            n = max(1, int(np.ceil(d[-1] / SAMPLE)))
            s = np.unique(np.r_[np.linspace(0, d[-1], n + 1), d])
            p = np.c_[np.interp(s, d, v[:, 0]), np.interp(s, d, v[:, 1])]
            z = np.array([ground(px, py) for px, py in p])
            if np.isnan(z).all():
                continue
            ok = ~np.isnan(z)
            z = np.interp(np.arange(len(z)), np.flatnonzero(ok), z[ok])
            for nid in (seg[0], seg[-1]):
                ids.setdefault(nid, len(ids))
            out.append([[ids[seg[0]], ids[seg[-1]]], [round(float(c), 1) for c in np.c_[p, z].ravel()]])
    print(f"{len(out)} footpath segments, {sum(edge_length(x[1], vs)[-1] for x in out) / 1000:.1f} km")
    return out


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
    node at their centroid; drop the short edges. The other edges keep their own end points."""
    parent = list(range(len(node_xy)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def length(e):
        p = np.array(e[6]).reshape(-1, 3)[:, :2] * vs
        return float(np.hypot(*np.diff(p, axis=0).T).sum())

    # tracks get their own copy of nodes shared with roads (cars and trams do not interact yet), so road
    # intersections can be merged without bending the tracks
    roadn = {n for e in edges if e[2] == KIND["road"] for n in (e[0], e[1])}
    copy = {}
    for e in edges:
        if e[2] == KIND["road"]:
            continue
        for k in (0, 1):
            if e[k] in roadn:
                if e[k] not in copy:
                    copy[e[k]] = len(node_xy)
                    node_xy.append(list(node_xy[e[k]]))
                    node_sig.append(0)
                    parent.append(len(parent))
                e[k] = copy[e[k]]
    # (never at a roundabout: its pieces between entries are often < MERGE, and merging them collapsed every
    # small roundabout into one point where all its roads met; a short approach piece merged into a ring
    # node pulled the ring into a spike)
    ring_nodes = {n for e in edges if e[2] == KIND["road"] and e[11] for n in (e[0], e[1])}
    # shortest connections first, and never into a cluster wider than CLUSTER: chains of short edges used to
    # merge whole squares into one junction with curves between every pair of its roads
    members = {i: [i] for i in range(len(node_xy))}
    def width_ok(ra, rb):
        pa = np.array([node_xy[i][:2] for i in members[ra]]) * vs
        pb = np.array([node_xy[i][:2] for i in members[rb]]) * vs
        return float(np.max(np.hypot(*(pa[:, None, :] - pb[None, :, :]).transpose(2, 0, 1)))) <= CLUSTER
    short = sorted((length(e), k) for k, e in enumerate(edges)
                   if e[2] == KIND["road"] and e[0] != e[1] and not e[11] and e[0] not in ring_nodes and e[1] not in ring_nodes)
    for L, k in short:
        if L >= MERGE:
            break
        ra, rb = find(edges[k][0]), find(edges[k][1])
        if ra != rb and width_ok(ra, rb):
            parent[ra] = rb
            members[rb] += members.pop(ra)
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
    out, pieces = [], []
    for e in edges:
        a, b = new_ix[e[0]], new_ix[e[1]]
        if a == b and e[2] == KIND["road"]:
            # inside a merged intersection (or a loop back into it): not driven as an edge (the junction's
            # curves join the roads around it), but kept as a junction piece: it is still road surface
            # (drawn, a street zone; it used to vanish, e.g. a side street across a boulevard's median)
            pieces.append([a, b, *e[2:6], list(e[6]), *e[7:]])
            continue
        # the edges keep their own OSM ends: one junction for the traffic, but no road is stretched to the
        # cluster's centroid (that drew straight spikes across dual carriageways and squares); the junction
        # curves join the real ends
        out.append([a, b, *e[2:6], list(e[6]), *e[7:]])
    print(f"consolidated {len(node_xy)} -> {len(xy)} nodes, {len(edges)} -> {len(out)} edges ({len(pieces)} junction pieces)")
    return out, xy, sig, pieces


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
                xw = [[round(m - cum[i], 1), c] for m, c in e[10] if cum[i] <= m <= cum[j]]   # (index 10 before pavements())
                out.append([ends[0], ends[1], *e[2:6], [round(float(c), 1) for c in p[i:j + 1].ravel()], e[7], ctl, e[9], xw, *e[11:]])
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
            # single track: follow a one-way road running alongside (roundabouts, one-way streets)
            rd = road_along(e, edges, vs)
            if rd is None:
                continue
            if rd < 0:
                reverse(e, vs)
            e[3] = 1
            n += 1
            continue
        rel = best[1] - P
        left = D[0] * rel[1] - D[1] * rel[0] > 0           # other track on the left (x east, y north)
        if not left:                                       # travel the other way
            reverse(e, vs)
        e[3] = 1
        n += 1
    # chains: a track continuing a one-way track through a node (and nothing else there) continues its direction
    tram_all = [e for e in edges if e[2] == KIND["tram"]]
    changed = True
    while changed:
        changed = False
        deg = {}
        for e in tram_all:
            for k in (e[0], e[1]):
                deg[k] = deg.get(k, 0) + 1
        for e in tram_all:
            if e[3]:
                continue
            for node, at_start in ((e[0], True), (e[1], False)):
                if deg.get(node) != 2:
                    continue
                f = next(f for f in tram_all if f is not e and node in (f[0], f[1]))
                if not f[3]:
                    continue
                into = f[1] == node                         # f arrives at the node
                if into != at_start:                        # e must leave where f arrives (and vice versa)
                    reverse(e, vs)
                e[3] = 1
                n += 1
                changed = True
                break
    print(f"tram tracks: {n} of {len(trams)} given a direction")


def pavements(edges, m, vs):
    """Where the pavement is, per side of each road edge: just in front of the facades. From the
    centreline, look sideways every 10 m for the first building voxel 1-6 m above the road; the
    pavement runs 1.5 m in front of it (median over the edge). Appended to the edge as
    [left, right] in metres (left / right of the edge direction); 0: no facade found."""
    for e in edges:
        if e[2] != KIND["road"]:
            e.insert(10, [0, 0])   # (index 10: crosswalks and the roundabout flag follow)
            continue
        p = np.array(e[6]).reshape(-1, 3)
        found = [[], []]
        for k in range(1, len(p) - 1, 5):
            d = p[k + 1, :2] - p[k - 1, :2]
            d = d / (np.linalg.norm(d) or 1)
            left = np.array([-d[1], d[0]])                   # x east, y north: left of travel
            for side, sgn in ((0, 1), (1, -1)):
                for r in range(4, 50):                       # 2..25 m, voxel steps
                    q = p[k, :2] + sgn * left * r
                    c = m.column(int(q[0]), int(q[1]))
                    if c is None:
                        break
                    z0 = int(p[k, 2])
                    if (c[z0 + 2:z0 + 12] == BUILDING).any():
                        found[side].append(r * vs)
                        break
        e.insert(10, [round(float(np.median(f)) - 1.5, 1) if len(f) >= 2 else 0 for f in found])


SIG_REACH = 35.0   # m: an OSM light this close to a junction (along the road, through plain bends) is that junction's
CROSSING_LIGHT = 4  # control code: a light on its own (a pedestrian crossing away from junctions)


def assign_signals(edges, node_sig, pieces, vs):
    """Which junctions have traffic lights, and where each incoming road stops for them.

    OSM maps lights inconsistently: on the junction node, at each stop line (with or without
    traffic_signals:direction), on the crosswalks around it, sometimes only on some of its roads. So OSM
    lights are evidence, not the model:
      - a junction is signalled if a light is on its node, inside it (a merged junction's pieces), or within
        SIG_REACH of it on one of its roads (through plain bends);
      - every road into a signalled junction gets a stop line (the viewer runs the phases): where OSM has a
        light for that direction on that road, else at the junction's mouth. Lights on a junction's exits
        (cars just past it used to stop again there) are part of its cycle: dropped;
      - lights away from junctions are pedestrian-crossing lights (CROSSING_LIGHT) for the directions that
        reach them.
    Rewrites the controls of road edges: signals become [m, SIGNALS, dir towards the junction] (stop lines,
    always for the junction the edge leads to in that direction) or [m, CROSSING_LIGHT, dir]."""
    roads = [e for e in edges if e[2] == KIND["road"]]
    deg, at = {}, {}
    for e in roads:
        for n in (e[0], e[1]):
            deg[n] = deg.get(n, 0) + 1
            at.setdefault(n, []).append(e)
    length = {id(e): edge_length(e[6], vs)[-1] for e in roads}
    is_junction = lambda n: deg.get(n, 0) >= 3

    def reach(e, end, d):
        """the junction reached leaving edge e through its end `end` (0 start, 1 end) within SIG_REACH
        (d: m already to that end), passing plain bends; or None"""
        n, seen = e[end], {id(e)}
        while d <= SIG_REACH:
            if is_junction(n):
                return n
            nxt = [o for o in at.get(n, []) if id(o) not in seen]
            if deg.get(n, 0) != 2 or len(nxt) != 1:
                return None                                 # dead end / region edge
            o = nxt[0]
            seen.add(id(o))
            d += length[id(o)]
            n = o[1] if o[0] == n else o[0]
        return None

    signalled = {n for n, s in enumerate(node_sig) if s and is_junction(n)}
    signalled |= {e[0] for e in pieces if any(c == SIGNALS for _, c, _ in e[8])}
    stats = {"stop line (OSM direction)": 0, "stop line (nearest junction)": 0, "on an exit: in the junction's cycle": 0,
             "for a junction beyond a bend": 0, "crossing light": 0}
    plan = []                                               # (edge, m, kind, dir, junction)
    for e in roads:
        L = length[id(e)]
        for m, c, dr in e[8]:
            if c != SIGNALS:
                continue
            ends = {0: reach(e, 0, m), 1: reach(e, 1, L - m)}   # junction behind each end (or None)
            dirs = [dr] if dr else ([1] if e[3] else [1, -1])   # directions driving through this light
            # each direction that drives through it heads for the junction ahead (end 1 for +1, end 0 for -1)
            ahead = {d: ends[1 if d > 0 else 0] for d in dirs}
            if dr == 0 and not e[3] and ahead[1] is not None and ahead[-1] is not None:
                # (two-way, no direction, junctions both ways: the nearer one's)
                ahead = {1: ahead[1]} if L - m <= m else {-1: ahead[-1]}
            got = {d: j for d, j in ahead.items() if j is not None}
            if got:
                for d, j in got.items():
                    signalled.add(j)
                    direct = e[1 if d > 0 else 0] == j
                    plan.append((e, m, "stop" if direct else "beyond", d, j))
                    stats["stop line (OSM direction)" if dr and direct else "stop line (nearest junction)" if direct
                          else "for a junction beyond a bend"] += 1
            elif any(j is not None for j in ends.values()):
                # only behind it: a light on a junction's exit (cars come from the junction)
                for j in ends.values():
                    if j is not None:
                        signalled.add(j)
                plan.append((e, m, "exit", 0, None))
                stats["on an exit: in the junction's cycle"] += 1
            else:
                plan.append((e, m, "crossing", dr if dr else (1 if e[3] else 0), None))
                stats["crossing light"] += 1
    # lights on plain bends (OSM ways split there): crossing lights for the traffic reaching them, unless a
    # junction is close (then they are its)
    for n, s in enumerate(node_sig):
        if not s or is_junction(n) or deg.get(n, 0) == 0:
            continue
        near = [j for e in at[n] for j in [reach(e, 1 if e[0] == n else 0, length[id(e)])] if j is not None and
                length[id(e)] <= SIG_REACH]
        if near:
            signalled.update(near)
            continue
        for e in at[n]:
            to_n = 1 if e[1] == n else -1
            if e[3] and to_n < 0:
                continue
            plan.append((e, length[id(e)] - 0.5 if to_n > 0 else 0.5, "crossing", to_n, None))
            stats["crossing light"] += 1
    for e in roads:
        e[8] = [x for x in e[8] if x[1] != SIGNALS]
    for e, m, kind, d, j in plan:
        if kind == "stop":
            e[8].append([m, SIGNALS, d])
        elif kind == "crossing":
            e[8].append([m, CROSSING_LIGHT, d])
    for e in roads:
        e[8].sort()
    for n in range(len(node_sig)):
        node_sig[n] = 1 if n in signalled else 0
    print(f"traffic lights: {len(signalled)} signalled junctions; OSM lights -> " + ", ".join(f"{v} {k}" for k, v in stats.items()))


def reverse(e, vs):
    L = edge_length(e[6], vs)[-1]
    e[0], e[1] = e[1], e[0]
    e[6] = [c for i in range(len(e[6]) - 3, -1, -3) for c in e[6][i:i + 3]]
    e[8] = sorted([[round(L - m, 1), c, -d] for m, c, d in e[8]])


def road_along(e, edges, vs):
    """+1 / -1 if a one-way road runs alongside tram edge e (same / opposite direction), else None"""
    p = np.array(e[6]).reshape(-1, 3)[:, :2]
    if len(p) < 3:
        return None
    votes = 0
    for k in range(1, len(p) - 1, max(1, len(p) // 6)):
        P, D = p[k], p[k + 1] - p[k - 1]
        D = D / (np.linalg.norm(D) or 1)
        best = None
        for o in edges:
            if o[2] != KIND["road"] or not o[3]:
                continue
            q = np.array(o[6]).reshape(-1, 3)[:, :2]
            d = np.hypot(*(q - P).T)
            i = int(np.argmin(d))
            if d[i] > 8.0 / vs or len(q) < 2:
                continue
            Do = q[min(i + 1, len(q) - 1)] - q[max(i - 1, 0)]
            c = float(np.dot(Do, D)) / (np.linalg.norm(Do) or 1)
            if abs(c) > 0.7 and (best is None or d[i] < best[0]):
                best = (d[i], c)
        if best:
            votes += 1 if best[1] > 0 else -1
    return None if votes == 0 else (1 if votes > 0 else -1)


PARK_NO, PARK_PARALLEL, PARK_DIAGONAL, PARK_PERP = 1, 2, 3, 4


def parking_sides(t, reversed_way):
    """OSM kerbside parking per side [left, right] of the way's drawing direction (flipped for oneway=-1, whose
    edge runs against it): 0 unknown, 1 none, 2 parallel, 3 diagonal, 4 perpendicular. Both tagging schemes:
    parking:<side>=lane|street_side|on_kerb|half_on_kerb|no + parking:<side>:orientation, and the older
    parking:lane:<side>=parallel|diagonal|perpendicular|no_parking|no_stopping."""
    def side(sd):
        out = 0
        for key in (f"parking:{sd}", "parking:both"):
            v = t.get(key)
            if v is None:
                continue
            if v in ("no", "no_parking", "no_stopping", "separate"):
                return PARK_NO
            o = t.get(f"{key}:orientation", "parallel")
            return {"diagonal": PARK_DIAGONAL, "perpendicular": PARK_PERP}.get(o, PARK_PARALLEL)
        for key in (f"parking:lane:{sd}", "parking:lane:both"):
            v = t.get(key)
            if v is None:
                continue
            if v in ("no", "no_parking", "no_stopping", "fire_lane", "separate"):
                return PARK_NO
            return {"diagonal": PARK_DIAGONAL, "perpendicular": PARK_PERP}.get(v, PARK_PARALLEL)
        return out
    l, r = side("left"), side("right")
    return [r, l] if reversed_way else [l, r]


def build():
    els = json.loads(RAW.read_text())["elements"]
    nodes = {e["id"]: to_l72.transform(e["lon"], e["lat"]) for e in els if e["type"] == "node"}
    ntags = {e["id"]: e.get("tags", {}) for e in els if e["type"] == "node"}

    def control(nid):
        t = ntags.get(nid, {})
        if t.get("highway") == "traffic_signals" or t.get("crossing") == "traffic_signals":
            return SIGNALS
        c = {"give_way": GIVE_WAY, "stop": STOP}.get(t.get("highway"), 0)
        return c or (CALMING if t.get("traffic_calming") in CALM_KINDS else 0)
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

    # spatial order: the LiDAR tiles behind ground() are read through a bounded cache
    ways.sort(key=lambda w: (round(nodes[w["nodes"][0]][1] / 250), nodes[w["nodes"][0]][0]))
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
                    # (traffic lights carry their direction as traffic_signals:direction; signs as direction)
                    dtag = ntags[seg[i]].get("traffic_signals:direction") or ntags[seg[i]].get("direction", "")
                    dr = {"forward": 1, "backward": -1}.get(dtag, 0) * (-1 if ow == "-1" else 1)
                    ctl.append([round(float(d[i]), 1), c, dr])
            if kind == "road" and t.get("traffic_calming") in CALM_KINDS:   # (a raised table mapped as its own way)
                ctl.append([round(float(d[-1]) / 2, 1), CALMING, 0])
            node_xy[node_ix[seg[0]]] = [*p[0], z[0]]
            node_xy[node_ix[seg[-1]]] = [*p[-1], z[-1]]
            rank = RANK.get(t.get("highway", "").replace("_link", ""), 0)
            cyc = " ".join(t.get(k, "") for k in ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both"))
            contra = 1 if oneway and kind == "road" and (t.get("oneway:bicycle") == "no" or "opposite" in cyc) else 0
            # mapped crosswalks along the edge: [m from the start, 1 signals | 2 marked (zebra) | 3 unmarked]
            xw = []
            if kind == "road":
                for i in range(1, len(seg) - 1):
                    nt = ntags.get(seg[i], {})
                    if nt.get("highway") == "crossing" or "crossing" in nt:
                        cr = nt.get("crossing", "")
                        typ = 1 if cr == "traffic_signals" else 3 if cr in ("unmarked", "informal") or nt.get("crossing:markings") == "no" else 2
                        xw.append([round(float(d[i]), 1), typ])
            ring = 1 if t.get("junction") in ("roundabout", "circular") else 0
            pk = parking_sides(t, ow == "-1") if kind == "road" else [0, 0]
            edges.append([node_ix[seg[0]], node_ix[seg[-1]], KIND[kind], oneway, speed(t), lanes,
                          [round(float(c), 1) for c in np.c_[p, z].ravel()], rank, ctl, contra, xw, ring, pk])
            stats[kind] += d[-1]
    edges, node_xy, node_sig, pieces = consolidate(edges, node_xy, node_sig, vs)
    pieces = [e for e in pieces if all(1 <= c < lim for c, lim in zip(e[6][0:2], m.size[:2]))]
    edges, node_xy, node_sig, boundary = clip(edges, node_xy, node_sig, m.size, vs)
    tram_directions(edges, vs)
    assign_signals(edges, node_sig, pieces, vs)
    pavements(edges, m, vs)
    pavements(pieces, m, vs)
    if "--refetch" in sys.argv or not RAW_PATHS.exists():
        fetch_paths()
    paths = walk_paths(m, X0, Y0, vs)
    out = {"build": m.man["build"], "paths": paths,
           "nodes": [[*(round(float(c), 1) for c in xyz), sig, b] for xyz, sig, b in zip(node_xy, node_sig, boundary)],
           "edges": edges, "pieces": pieces, "fields": ["from", "to", "kind(0 road,1 tram,2 rail)", "oneway", "kmh", "lanes", "xyz...",
                                      "rank", "controls [m, 1 stop line of the signalled junction ahead|2 give way|3 stop|4 crossing light|5 speed bump, dir]",
                                      "bikes both ways (one-way street open to cycling against the traffic)",
                                      "pavement [left, right] m from the centreline (0: none found)",
                                      "crosswalks [[m, 1 signals|2 marked|3 unmarked]]", "roundabout",
                                      "OSM parking [left, right]: 0 unknown, 1 none, 2 parallel, 3 diagonal, 4 perpendicular"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    nctl = sum(len(e[8]) for e in edges)
    print(f"{len(node_xy)} nodes ({sum(node_sig)} signalled), {len(edges)} edges ({nctl} controls); road {stats['road'] / 1000:.1f} km, tram {stats['tram'] / 1000:.1f} km, "
          f"rail {stats['rail'] / 1000:.1f} km -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    if "--refetch" in sys.argv or not RAW.exists():
        fetch()
    build()
