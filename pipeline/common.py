"""Shared by the pipeline scripts: paths and config, the region named on the command line, coordinate
transforms, the voxel tile format, OpenStreetMap (Overpass) and Mapillary access, logging, CPU count."""
import functools
import gzip
import json
import os
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
CFG = tomllib.loads((ROOT / "config.toml").read_text())
VS = CFG["voxel"]["size"]                 # m per voxel (L0)
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
USER_AGENT = "bxl-vice-voxel/0.1"


def region(pos: int = 0):
    """(name, region config with its name) from the pos-th positional argument, else config.toml's default"""
    name = ARGS[pos] if len(ARGS) > pos else CFG["default"]
    return name, {"name": name, **CFG["roi"][name]}


# Lambert 72 (EPSG:31370, metres) <-> WGS84 (lon, lat)
to_wgs = Transformer.from_crs(31370, 4326, always_xy=True)
to_l72 = Transformer.from_crs(4326, 31370, always_xy=True)

# --- voxel tiles (voxelize.py writes them; the viewer (tileFormat.ts), model.py and streetview.py read them) ---
# record: 8 bytes x+1, y+1, z, r, g, b, class, flags (x / y tile-local with a 1-voxel apron)
# On disk (.vxt, manifest version 3) a tile is stored compactly, then gzipped (see encode_tile); the
# readers turn it back into these records. (Version 2 builds: raw records in .bin files.)
C_UNCLASSIFIED, C_GROUND, C_LOWVEG, C_HIGHVEG, C_BUILDING = 1, 2, 3, 5, 6    # LiDAR (ASPRS) classes
C_VEHICLE = 64            # derived by voxelize.py: parked vehicles found in unclassified geometry
FLAG_FILL = 1             # geometry synthesised (wall / terrain fill), not measured
FLAG_APRON = 2            # belongs to the neighbouring tile, used only for culling
FLAG_PTCOLOR = 4          # synthetic voxel whose colour comes from nearby measured points
FLAG_PHOTO = 8            # colour from street-level photos (streetview.py layer)
LABEL_SHIFT = 4           # flags bits 4-7: semantic label (streetview.L_NAMES)

TILE_MAGIC = b"VXT\x03"
TILE_COLOURS = 256        # palette size per tile


def encode_tile(rec: np.ndarray, width: int) -> bytes:
    """Records (n x 8) -> the compact tile file. Before gzip:
         magic "VXT\\x03", width (tile size + 2, u8), 0, palette size P (u16 le), voxel count n (u32 le),
         palette (P x rgb),
         voxel positions in ascending order of key = (x * width + y) * 256 + z (x, y as in the records):
           the difference to the previous key (first: to 0) as a LEB128 varint,
         then per voxel, in that order: palette index (n bytes), class (n bytes), flags (n bytes).
       Colours: at most TILE_COLOURS per tile (maximum coverage, so rare colours such as a red sign keep an entry, then refined by k-means; no dithering;
       exact when the tile has fewer)."""
    from PIL import Image   # (only needed when writing)
    key = (rec[:, 0].astype(np.int64) * width + rec[:, 1]) * 256 + rec[:, 2]
    order = np.argsort(key, kind="stable")
    rec, key = rec[order], key[order]
    rgb = np.ascontiguousarray(rec[:, 3:6])
    colours, idx = np.unique(rgb, axis=0, return_inverse=True)
    if len(colours) > TILE_COLOURS:
        q = Image.fromarray(rgb.reshape(1, -1, 3)).quantize(TILE_COLOURS, Image.Quantize.MAXCOVERAGE, kmeans=4, dither=Image.Dither.NONE)
        idx = np.asarray(q).ravel()
        colours = np.array(q.getpalette()[: 3 * TILE_COLOURS], np.uint8).reshape(-1, 3)[: int(idx.max()) + 1]
    # varints of the key differences (vectorised: 7 bits per byte, high bit = more follows)
    d = np.diff(key, prepend=0).astype(np.uint64)
    nbytes = np.ones(len(d), np.int64)
    for k in range(1, 5):
        nbytes += d >= (1 << (7 * k))
    out = np.zeros(int(nbytes.sum()), np.uint8)
    pos = np.concatenate([[0], np.cumsum(nbytes)[:-1]])
    for k in range(5):
        m = nbytes > k
        out[pos[m] + k] = ((d[m] >> np.uint64(7 * k)) & np.uint64(127)).astype(np.uint8) | np.where(nbytes[m] > k + 1, 128, 0).astype(np.uint8)
    head = TILE_MAGIC + bytes([width, 0]) + len(colours).to_bytes(2, "little") + len(rec).to_bytes(4, "little")
    body = head + colours.astype(np.uint8).tobytes() + out.tobytes() + idx.astype(np.uint8).tobytes() \
        + rec[:, 6].tobytes() + rec[:, 7].tobytes()
    return gzip.compress(body, 9, mtime=0)


def decode_tile(data: bytes) -> np.ndarray:
    """a tile (encode_tile) -> records (n x 8, uint8)"""
    data = gzip.decompress(data)
    if data[:4] != TILE_MAGIC:
        raise ValueError("not a voxel tile")
    width, P, n = data[4], int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:12], "little")
    o = 12
    pal = np.frombuffer(data, np.uint8, 3 * P, o).reshape(-1, 3)
    o += 3 * P
    b = np.frombuffer(data, np.uint8, len(data) - o, o)
    ends = np.flatnonzero(b < 128)[:n]           # last byte of each varint
    starts = np.concatenate([[0], ends[:-1] + 1])
    d = np.zeros(n, np.int64)
    for k in range(5):
        m = starts + k <= ends
        d[m] |= (b[starts[m] + k].astype(np.int64) & 127) << (7 * k)
    key = np.cumsum(d)
    o += int(ends[-1]) + 1 if n else 0
    rest = np.frombuffer(data, np.uint8, 3 * n, o).reshape(3, n)
    rec = np.empty((n, 8), np.uint8)
    rec[:, 2] = key % 256
    rec[:, 1] = (key // 256) % width
    rec[:, 0] = key // 256 // width
    rec[:, 3:6] = pal[rest[0]]
    rec[:, 6], rec[:, 7] = rest[1], rest[2]
    return rec


# --- manifest (version 5) and tile archives ---
# Each level's tiles are listed as columns, file names implied:
#   tiles: {"at": [ty * W + tx of each tile, ascending, as differences (first: from 0)],
#           "zmin": [...], "zh": [zmax - zmin], "len": [bytes of the tile in its archive]}   W = tiles across
# The tiles are stored in archives of ARCHIVE x ARCHIVE tiles (L{level}/a_{tx // ARCHIVE}_{ty // ARCHIVE}.vxa):
# the tile files concatenated in list order, so a tile's offset is the sum of the lengths before it in the
# same archive. The viewer reads one tile with an HTTP range request: a few hundred files instead of tens of
# thousands (faster uploads, fewer requests to object storage).
MANIFEST_VERSION = 5
ARCHIVE = 16


def tile_file(level: int, tx: int, ty: int) -> str:
    """a single tile (voxelize.py writes these first; pack_archives puts them into archives)"""
    return f"L{level}/t_{tx}_{ty}.vxt"


def archive_file(level: int, tx: int, ty: int) -> str:
    return f"L{level}/a_{tx // ARCHIVE}_{ty // ARCHIVE}.vxa"


def pack_level_tiles(tiles: list, level: dict, tile_size: int) -> dict:
    W = -(-level["size"][0] // tile_size)
    ts = sorted(tiles, key=lambda t: (t["ty"], t["tx"]))
    at = [t["ty"] * W + t["tx"] for t in ts]
    return {"at": [a - b for a, b in zip(at, [0] + at[:-1])], "zmin": [t["zmin"] for t in ts],
            "zh": [t["zmax"] - t["zmin"] for t in ts], "len": [t["len"] for t in ts]}


def level_tiles(level: dict, tile_size: int) -> list:
    """a manifest level's tiles as [{file (its archive), off, len, tx, ty, zmin, zmax}]"""
    t = level["tiles"]
    W, at, out, ends = -(-level["size"][0] // tile_size), 0, [], {}
    for d, z0, zh, n in zip(t["at"], t["zmin"], t["zh"], t["len"]):
        at += d
        tx, ty = at % W, at // W
        f = archive_file(level["level"], tx, ty)
        out.append({"file": f, "off": ends.get(f, 0), "len": n, "tx": tx, "ty": ty, "zmin": z0, "zmax": z0 + zh})
        ends[f] = ends.get(f, 0) + n
    return out


def read_tile_at(base: Path, t: dict) -> np.ndarray:
    """records of a manifest tile (level_tiles entry), read from its archive"""
    with open(base / t["file"], "rb") as f:
        f.seek(t["off"])
        return decode_tile(f.read(t["len"]))


def pack_archives(base: Path, levels: list, tile_size: int):
    """the single tile files (tile_file) of each level (its "tiles": [{tx, ty, zmin, zmax}]) -> archives;
    replaces "tiles" with the packed list (pack_level_tiles); deletes the single files. Returns bytes per level."""
    sizes = []
    for lv in levels:
        ts = sorted(lv["tiles"], key=lambda t: (t["ty"], t["tx"]))
        groups = {}
        for t in ts:
            groups.setdefault(archive_file(lv["level"], t["tx"], t["ty"]), []).append(t)
        total = 0
        for name, group in groups.items():
            with open(base / name, "wb") as out:
                for t in group:
                    data = (base / tile_file(lv["level"], t["tx"], t["ty"])).read_bytes()
                    out.write(data)
                    t["len"] = len(data)
                    total += len(data)
        for t in ts:
            (base / tile_file(lv["level"], t["tx"], t["ty"])).unlink()
        lv["tiles"] = pack_level_tiles(ts, lv, tile_size)
        sizes.append(total)
    return sizes


# --- road network (roads.py -> viewer/public/roads/<roi>.json): each edge is a list in this order ---
#   from, to (nodes), kind (0 road, 1 tram, 2 rail), oneway, km/h, lanes, xyz (polyline, voxels), rank (road class),
#   controls [[m, 1 stop line of the signalled junction ahead | 2 give way | 3 stop | 4 crossing light | 5 speed bump, dir]],
#   contraflow (one-way street open to cycling against the traffic), pavement [left, right] (m from the centreline,
#   0: none found), crosswalks [[m, 1 signals | 2 marked | 3 unmarked]], roundabout,
#   OSM parking [left, right] (0 unknown, 1 none, 2 parallel, 3 diagonal, 4 perpendicular)
(E_FROM, E_TO, E_KIND, E_ONEWAY, E_KMH, E_LANES, E_XYZ, E_RANK, E_CONTROLS, E_CONTRA, E_PAVEMENT, E_CROSSWALKS,
 E_RING, E_PARKING) = range(14)
KIND_ROAD, KIND_TRAM, KIND_RAIL = 0, 1, 2
# Polylines (edge and piece xyz, footpath points) are stored as whole tenths of a voxel, each point as the
# difference to the previous one ("xyz": "delta10"); read_roads gives them back as voxel floats.
ROADS_XYZ = "delta10"


def pack_line(xyz: list) -> list:
    q = [round(v * 10) for v in xyz]
    return q[:3] + [q[i] - q[i - 3] for i in range(3, len(q))]


def unpack_line(q: list) -> list:
    out = list(q[:3])
    for i in range(3, len(q)):
        out.append(out[i - 3] + q[i])
    return [v / 10 for v in out]


def _map_lines(net: dict, fn) -> dict:
    net = dict(net)
    for k in ("edges", "pieces"):
        net[k] = [[*e[:E_XYZ], fn(e[E_XYZ]), *e[E_XYZ + 1:]] for e in net.get(k, [])]
    net["paths"] = [[ends, fn(p)] for ends, p in net.get("paths", [])]
    return net


def write_roads(path: Path, net: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**_map_lines(net, pack_line), "xyz": ROADS_XYZ}, separators=(",", ":")))


def read_roads(path: Path) -> dict:
    net = json.loads(path.read_text())
    return _map_lines(net, unpack_line)


# --- timetable (transit.py -> viewer/public/transit/<roi>.json) ---
# Trips repeat: the same stops in the same order (a pattern) and the same times between them (a timing).
#   patterns: [[route, headsign, [stop, ...]]]   timings: [[arrival - first arrival, departure - arrival] per stop, flat]
#   schedules: [[[pattern, timing, first arrival s], ...]]   (distinct days)   week: [schedule of Monday .. Sunday]
def pack_timetable(days: list) -> dict:
    patterns, timings, schedules, week, seen = {}, {}, [], [], {}
    for day in days:
        trips = []
        for route, headsign, st in day:
            t0 = st[0][1]
            p = patterns.setdefault((route, headsign, tuple(s[0] for s in st)), len(patterns))
            q = timings.setdefault(tuple(x for s in st for x in (s[1] - t0, s[2] - s[1])), len(timings))
            trips.append([p, q, t0])
        key = json.dumps(trips)
        if key not in seen:
            seen[key] = len(schedules)
            schedules.append(trips)
        week.append(seen[key])
    return {"patterns": [[r, h, list(st)] for r, h, st in patterns], "timings": [list(t) for t in timings],
            "schedules": schedules, "week": week}


def log_clock():
    """log(msg): print with the seconds since this call"""
    t0 = time.time()
    return lambda msg: print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def jobs() -> int:
    """worker processes: --jobs=N, else the CPUs we may use (container quota, not the host's cores)"""
    a = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--jobs=")), None)
    if a:
        return max(1, int(a))
    try:
        q, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q != "max":
            return max(1, int(q) // int(period))
    except (OSError, ValueError):
        pass
    return max(1, (os.cpu_count() or 2) - 1)


# --- OpenStreetMap ---
def bbox(roi, margin: float = 0.0) -> str:
    """the region (plus margin m) as Overpass's "south,west,north,east" """
    corners = [to_wgs.transform(x, y) for x in (roi["xmin"] - margin, roi["xmax"] + margin)
               for y in (roi["ymin"] - margin, roi["ymax"] + margin)]
    return (f"{min(c[1] for c in corners):.6f},{min(c[0] for c in corners):.6f},"
            f"{max(c[1] for c in corners):.6f},{max(c[0] for c in corners):.6f}")


def overpass(query: str, dest: Path):
    """run an Overpass query and save the answer to dest. The public server is often briefly overloaded
    (429 / 5xx: retried); an answer cut short by a timeout or memory limit still comes as HTTP 200 with a
    "remark" and partial elements: never saved (it would be reused as if complete)."""
    req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                                 data=urllib.parse.urlencode({"data": query}).encode(),
                                 headers={"User-Agent": USER_AGENT})
    for i in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read()
            remark = json.loads(body).get("remark", "")
            if "error" in remark.lower():
                raise RuntimeError(f"incomplete answer: {remark}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            stamp = json.loads(body).get("osm3s", {}).get("timestamp_osm_base", "?")
            print(f"overpass: {dest.name} (OSM data of {stamp})")
            return
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise SystemExit(f"overpass: HTTP {e.code} (the query is wrong?)") from e
            print(f"overpass: HTTP {e.code}; retrying")
        except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
            print(f"overpass: {e}; retrying")
        time.sleep(10 * (i + 1))
    raise SystemExit("overpass unavailable")


# --- Mapillary ---
MAPILLARY_MIN_SCORE = 0.5   # calibrated segments (streetview.py calibrate) matching the voxels less than this are not used


@functools.cache
def mapillary_token() -> str:
    """MAPILLARY_TOKEN from the repository's .env (gitignored; the only place it is kept)"""
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("MAPILLARY_TOKEN"):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("MAPILLARY_TOKEN missing in .env")


def mapillary_get(url: str, auth: bool = True) -> bytes:
    """a Mapillary API request (auth: with the token; image files do not need it), retried a few times"""
    for i in range(4):
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"OAuth {mapillary_token()}"} if auth else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if i == 3:
                raise
            time.sleep(2 ** i)


def mapillary_grid(x0: float, y0: float, x1: float, y1: float, url, cell: float = 250.0) -> list:
    """the "data" of one API query per cell x cell m of the L72 box (each query stays below the 2000-result
    cap); url(bbox) gives the query for a "lon0,lat0,lon1,lat1" bbox"""
    from concurrent.futures import ThreadPoolExecutor

    def query(c):
        a = to_wgs.transform(c[0], c[1])
        b = to_wgs.transform(min(c[0] + cell, x1), min(c[1] + cell, y1))
        return json.loads(mapillary_get(url(f"{a[0]:.6f},{a[1]:.6f},{b[0]:.6f},{b[1]:.6f}")))["data"]

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(query, [(x, y) for x in np.arange(x0, x1, cell) for y in np.arange(y0, y1, cell)]))
    if any(len(r) >= 2000 for r in res):
        print("warning: a Mapillary query hit the 2000-result cap: use smaller cells")
    return [d for r in res for d in r]


def mapillary_offset(cal: dict) -> np.ndarray:
    """the systematic Mapillary -> Lambert 72 offset (east, north, up): the median over the well-matching
    segments of a streetview.py calibration"""
    good = [c["offset"] for c in cal.values() if c["score"] >= MAPILLARY_MIN_SCORE]
    return np.median(good, 0) if good else np.zeros(3)
