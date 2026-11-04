"""Shared by the pipeline scripts: paths and config, the region named on the command line, coordinate
transforms, the voxel tile format, OpenStreetMap (Overpass) and Mapillary access, logging, CPU count."""
import json
import os
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

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

# --- voxel tiles (voxelize.py writes them; viewer/src/voxels.ts and model.py read them) ---
# record: 8 bytes x+1, y+1, z, r, g, b, class, flags (x / y tile-local with a 1-voxel apron)
C_UNCLASSIFIED, C_GROUND, C_LOWVEG, C_HIGHVEG, C_BUILDING = 1, 2, 3, 5, 6    # LiDAR (ASPRS) classes
C_VEHICLE = 64            # derived by voxelize.py: parked vehicles found in unclassified geometry
FLAG_FILL = 1             # geometry synthesised (wall / terrain fill), not measured
FLAG_APRON = 2            # belongs to the neighbouring tile, used only for culling
FLAG_PTCOLOR = 4          # synthetic voxel whose colour comes from nearby measured points
FLAG_PHOTO = 8            # colour from street-level photos (streetview.py layer)
LABEL_SHIFT = 4           # flags bits 4-7: semantic label (streetview.L_NAMES)


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
