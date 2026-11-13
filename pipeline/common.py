"""Shared by the pipeline scripts: paths and config, the region named on the command line, coordinate
transforms, the voxel tile format, OpenStreetMap (Overpass) and Mapillary access, logging, CPU count."""
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
    """A tile file (compact .vxt, or a version 2 .bin of raw records) -> records (n x 8, uint8)."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    if data[:4] != TILE_MAGIC:
        return np.frombuffer(data, np.uint8).reshape(-1, 8)
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


def read_tile(path: Path) -> np.ndarray:
    return decode_tile(path.read_bytes())


# --- road network (roads.py -> viewer/public/roads/<roi>.json): each edge is a list in this order ---
(E_FROM, E_TO, E_KIND, E_ONEWAY, E_KMH, E_LANES, E_XYZ, E_RANK, E_CONTROLS, E_CONTRA, E_PAVEMENT, E_CROSSWALKS,
 E_RING, E_PARKING) = range(14)
KIND_ROAD, KIND_TRAM, KIND_RAIL = 0, 1, 2


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
def mapillary_token() -> str:
    """MAPILLARY_TOKEN from the repository's .env (gitignored; the only place it is kept)"""
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("MAPILLARY_TOKEN"):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("MAPILLARY_TOKEN missing in .env")
