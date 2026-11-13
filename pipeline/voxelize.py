"""LiDAR point cloud -> sparse coloured voxels -> LOD pyramid of quadtree tiles.

Single-source build: only the Brussels 2021 LiDAR (CC-BY Paradigm/UrbIS).

Works in 256 m blocks so memory does not grow with the region:
  pass 1  stream the LAS files once and bucket points per block (+32 m margin)
          into data/cache/points/<roi>/ (16 bytes per point, reused next run)
  pass 2  per block:
            a. bin points into VS-metre voxels: mean RGB, majority class (drop 1-point voxels)
            b. per column: ground height (class 2, holes filled from nearest) and building top
            c. buildings and terrain made solid, only the visible shell is kept
            d. wall voxels coloured from nearby facade points; loose facade voxels absorbed
            e. parked vehicles found in the geometry: removed (their footprints and colours kept for
               parking.py / vehicles.py), the road under them coloured like the open road around
            f. tidy: low loose clutter dropped; in the street zones (roads.py network) also what is left
               of 2021's traffic on the carriageway and clutter on the pavements
            g. isolated specks (clusters backed by < SPECK_POINTS points) dropped
            then levels L0..L3 (0.5..4 m) built and written as 64x64-column tiles; the
               margin supplies each tile's 1-voxel apron across block seams
  pass 3  levels above L3 (one tile spans several blocks) built from the collected L3 voxels

Tile (tx,ty) at level L has children (2tx..2tx+1, 2ty..2ty+1) at L-1.

Usage: uv run python voxelize.py [roi-name] [--jobs=N]   (default region from config.toml)
  merges data/cache/streetview/<roi>/layer.npz (streetview.py) when present

Builds into <name>.building and swaps it in at the end, so a running viewer never
sees a half-written build; the manifest carries a build id for cache busting.
"""

import json
import multiprocessing as mp
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import laspy
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from common import (CFG, MANIFEST_VERSION, encode_tile, pack_archives, read_roads, tile_file, C_BUILDING, E_KIND, E_LANES, E_ONEWAY, E_PAVEMENT, E_XYZ, KIND_ROAD, C_GROUND, C_HIGHVEG, C_LOWVEG, C_UNCLASSIFIED, C_VEHICLE, FLAG_APRON, FLAG_FILL,
                    FLAG_PHOTO, FLAG_PTCOLOR, LABEL_SHIFT, ROOT, VS, jobs as cpu_jobs, log_clock, region)

ROI_NAME, ROI = region()
LEVELS = ROI.get("levels", 4)
CH = 64            # tile size in columns (every level)
NZ = 256           # max voxel layers at L0 (z stored as uint8)
BLOCK = 512        # block size in L0 columns (= one L3 tile = 256 m)
MARGIN = 64        # extra L0 columns read around each block (32 m)
OUT = ROOT / "viewer" / "public" / "tiles" / ROI["name"]
CACHE = ROOT / "data" / "cache" / "points" / ROI["name"]
DEST = OUT  # where tiles are written (a staging folder during a build)
LAYER_F = ROOT / "data" / "cache" / "streetview" / ROI["name"] / "layer.npz"
LAYER = {}  # street-level colour layer, loaded in main() (shared with forked workers)

FOOTPRINTS = ROOT / "data" / "cache" / "vehicles"
# tidy: unclassified voxels low above the ground are kept only when they belong to something: a
# structure rising above TIDY_H (posts, signs, shelters), a building (attached to it), or something the
# street photos recognised (building, sign, pole, fence, street furniture). Everything higher is kept.
TIDY_H = 5                 # voxels (2.5 m) above the ground
# street tidy: in the carriageway (OSM roads, measured kerbs) everything
# unclassified up to ROAD_H that is not part of a taller structure goes (cars of 2021 the vehicle finder missed:
# vans, buses, moving cars smeared by the scan; fragments); on pavements also next to the facades
ROAD_H = 8                 # voxels (4 m)
FLOAT_VOX = 20             # small clusters floating more than FLOAT_H above the ground go
FLOAT_H = 2
SEM_KEEP = [1, 2, 6, 7, 8, 11]
LABEL_MIN_SESSIONS = 2
# semantic classes 1-12 as in streetview.L_NAMES (12 = sky is a photo/model mismatch, never a voxel class)
SEM_BUILDING, SEM_VEG, SEM_GROUND, SEM_SKY, SEM_WATER, SEM_RAIL, SEM_OTHER = 1, 3, 9, 12, 13, 14, 15
SEM_FROM_LIDAR = np.full(256, SEM_OTHER, np.uint8)
SEM_FROM_LIDAR[[C_GROUND, C_LOWVEG, C_HIGHVEG, C_BUILDING, 9, 10, C_VEHICLE]] = [
    SEM_GROUND, SEM_VEG, SEM_VEG, SEM_BUILDING, SEM_WATER, SEM_RAIL, 4]
SEM_SIGN = 2
SEM_TRANSIENT = [4, 5, 10]   # vehicle, bike, person: they move, so photos taken later cannot label LiDAR geometry

PT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("c", "u1")])

def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


NX = int(round((ROI["xmax"] - ROI["xmin"]) / VS))
NY = int(round((ROI["ymax"] - ROI["ymin"]) / VS))
NBX, NBY = ceil_div(NX, BLOCK), ceil_div(NY, BLOCK)

log = log_clock()


# --- pass 1: bucket points per block -----------------------------------------
def bucket_tile(tile: str):
    """Pass 1 for one LAS file (runs in a worker): points in the region, bucketed per block (+margin)
    into CACHE/b{block}_{tile}.bin. Returns (tile, low ground percentiles, points)."""
    path = next((ROOT / "data/raw/lidar").glob(f"*{tile}*.las"))
    files, ground_lo, total = {}, [], 0
    with laspy.open(path) as f:
        for ch in f.chunk_iterator(5_000_000):
            x = np.asarray(ch.x) - ROI["xmin"]
            y = np.asarray(ch.y) - ROI["ymin"]
            c = np.asarray(ch.classification)
            m = (x >= 0) & (x < NX * VS) & (y >= 0) & (y < NY * VS) & (c != 7) & (c != 18)
            if not m.any():
                continue
            rec = np.empty(m.sum(), PT)
            rec["x"], rec["y"], rec["z"] = x[m], y[m], np.asarray(ch.z)[m]
            rec["r"], rec["g"], rec["b"] = (np.asarray(ch.red)[m] >> 8), (np.asarray(ch.green)[m] >> 8), (np.asarray(ch.blue)[m] >> 8)
            rec["c"] = c[m]
            g = rec["z"][rec["c"] == C_GROUND]
            if len(g):
                ground_lo.append(float(np.percentile(g, 0.5)))   # robust: tunnels / canal edges are below
            total += len(rec)
            cx, cy = rec["x"] / VS, rec["y"] / VS
            bx0 = np.floor((cx - MARGIN) / BLOCK).clip(0, NBX - 1).astype(int)
            bx1 = np.floor((cx + MARGIN) / BLOCK).clip(0, NBX - 1).astype(int)
            by0 = np.floor((cy - MARGIN) / BLOCK).clip(0, NBY - 1).astype(int)
            by1 = np.floor((cy + MARGIN) / BLOCK).clip(0, NBY - 1).astype(int)
            for dx in (0, 1):
                for dy in (0, 1):
                    bx, by = bx0 + dx, by0 + dy
                    sel = (bx <= bx1) & (by <= by1)
                    if dx and not sel.any():
                        continue
                    bid = (by * NBX + bx)[sel]
                    r = rec[sel]
                    o = np.argsort(bid, kind="stable")
                    bid, r = bid[o], r[o]
                    cuts = np.flatnonzero(np.diff(bid)) + 1
                    for part, b in zip(np.split(r, cuts), bid[np.r_[0, cuts]] if len(bid) else []):
                        fh = files.get(b) or files.setdefault(b, open(CACHE / f"b{b}_{tile}.bin", "ab"))
                        fh.write(part.tobytes())
    for fh in files.values():
        fh.close()
    return tile, ground_lo, total


def bucket_points(jobs: int) -> float:
    """Pass 1: the LAS files in parallel (one worker per file; reading them is the slow part)."""
    key = {"roi": {k: ROI[k] for k in ("xmin", "ymin", "xmax", "ymax", "tiles")}, "block": BLOCK,
           "margin": MARGIN, "vs": VS, "format": 3}
    meta_f = CACHE / "meta.json"
    if meta_f.exists():
        meta = json.loads(meta_f.read_text())
        if meta["key"] == key:
            log(f"pass 1: reusing point cache ({meta['points']:,} points)")
            return meta["z0"]
    CACHE.mkdir(parents=True, exist_ok=True)
    for f in CACHE.glob("*"):
        f.unlink()
    ground_lo, total = [], 0
    # (each reader holds one 5M-point chunk at a time; a few in parallel saturate the disk)
    with ProcessPoolExecutor(max_workers=min(jobs, 6, len(ROI["tiles"])), mp_context=mp.get_context("fork")) as ex:
        for tile, gl, n in ex.map(bucket_tile, ROI["tiles"]):
            ground_lo += gl
            total += n
            log(f"pass 1: {tile} read ({n:,} points; {total:,} in region so far)")
    z0 = float(np.floor(min(ground_lo))) - 1.0
    meta_f.write_text(json.dumps({"key": key, "z0": z0, "points": total}))
    return z0


# --- helpers ----------------------------------------------------------------
def fill_nearest(grid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.all():
        return grid
    if not valid.any():
        return grid
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return grid[tuple(idx)]


def open_colour(gcol, open_m, size=9):
    """Smoothed colour of open ground, continued (nearest) into the rest: for ground that the aerial
    photo shows covered (tree crowns, cars and their shadows)."""
    w = ndimage.uniform_filter(open_m.astype(np.float32), size=size)
    sm = np.stack([ndimage.uniform_filter(np.where(open_m, gcol[..., c], 0).astype(np.float32), size=size)
                   for c in range(3)], -1) / np.maximum(w, 1e-6)[..., None]
    return fill_nearest(sm.round().clip(0, 255).astype(np.uint8), open_m & (w > 0.05))


def aggregate(key, rgb, cls, flags=None, weight=None):
    """Group by key: (weighted) mean colour, majority class, OR of flags, count."""
    ukey, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    w = np.ones(len(key)) if weight is None else weight
    wsum = np.bincount(inv, weights=w, minlength=len(ukey))
    col = np.stack([np.bincount(inv, weights=rgb[:, i] * w, minlength=len(ukey)) for i in range(3)], 1)
    col = (col / wsum[:, None]).round().clip(0, 255).astype(np.uint8)
    k2, c2 = np.unique(inv * 32 + cls, return_counts=True)
    order = np.lexsort((c2, k2 // 32))
    k2 = k2[order]
    last = np.r_[k2[1:] // 32 != k2[:-1] // 32, True]
    ucls = np.empty(len(ukey), np.uint8)
    ucls[k2[last] // 32] = (k2[last] % 32).astype(np.uint8)
    uflag = None
    if flags is not None:
        # low bits: OR of flags; high bits: majority street-level label among labelled members
        uflag = np.zeros(len(ukey), np.uint8)
        np.bitwise_or.at(uflag, inv, flags & ((1 << LABEL_SHIFT) - 1))
        lab = flags >> LABEL_SHIFT
        m = lab > 0
        if m.any():
            k3, c3 = np.unique(inv[m] * 16 + lab[m], return_counts=True)
            o = np.lexsort((c3, k3 // 16))
            k3 = k3[o]
            last = np.r_[k3[1:] // 16 != k3[:-1] // 16, True]
            uflag[k3[last] // 16] |= ((k3[last] % 16) << LABEL_SHIFT).astype(np.uint8)
    return ukey, col, ucls, uflag, cnt


class Vox:
    """Voxels of one level in global level coordinates."""
    def __init__(self, x, y, z, col, cls, flag):
        self.x, self.y, self.z, self.col, self.cls, self.flag = x, y, z, col, cls, flag

    def __len__(self):
        return len(self.x)

    def coarser(self) -> "Vox":
        W = int(self.x.max()) // 2 + 2 if len(self) else 1
        H = int(self.y.max()) // 2 + 2 if len(self) else 1
        k = ((self.z // 2) * H + self.y // 2) * W + self.x // 2
        # colour from the children seen from above (nothing directly on top), so a coarse
        # voxel's top face shows roof/ground/canopy and not the walls or fill beneath
        W0, H0 = int(self.x.max()) + 2, int(self.y.max()) + 2
        own = np.sort((self.z * H0 + self.y) * W0 + self.x)
        up = (self.z * H0 + self.y) * W0 + self.x + W0 * H0
        exposed = own[np.searchsorted(own, up).clip(0, len(own) - 1)] != up
        ukey, col, cls, flag, _ = aggregate(k, self.col, self.cls, self.flag, np.where(exposed, 50.0, 1.0))
        return Vox(ukey % W, (ukey // W) % H, ukey // (W * H), col, cls, flag)

    def select(self, x0, x1, y0, y1) -> "Vox":
        m = (self.x >= x0) & (self.x < x1) & (self.y >= y0) & (self.y < y1)
        return Vox(self.x[m], self.y[m], self.z[m], self.col[m], self.cls[m], self.flag[m])

    @staticmethod
    def concat(parts) -> "Vox":
        return Vox(*[np.concatenate([getattr(p, a) for p in parts]) for a in ("x", "y", "z", "col", "cls", "flag")])


def write_tiles(L, v: Vox, tx_range, ty_range):
    tiles = []
    for ty in ty_range:
        ym = (v.y >= ty * CH - 1) & (v.y <= ty * CH + CH)
        for tx in tx_range:
            m = ym & (v.x >= tx * CH - 1) & (v.x <= tx * CH + CH)
            if not m.any():
                continue
            lx, ly = v.x[m] - tx * CH, v.y[m] - ty * CH
            apron = (lx < 0) | (lx >= CH) | (ly < 0) | (ly >= CH)
            if apron.all():
                continue
            buf = np.empty((m.sum(), 8), np.uint8)
            buf[:, 0], buf[:, 1], buf[:, 2] = lx + 1, ly + 1, v.z[m]
            buf[:, 3:6] = v.col[m]
            buf[:, 6] = v.cls[m]
            buf[:, 7] = v.flag[m] | np.where(apron, FLAG_APRON, 0)
            name = tile_file(L, tx, ty)
            (DEST / name).write_bytes(encode_tile(buf, CH + 2))
            zr = v.z[m][~apron]
            tiles.append({"tx": tx, "ty": ty, "zmin": int(zr.min()), "zmax": int(zr.max())})
    return tiles


# --- pass 2: one block ---------------------------------------------------------
SPECK_POINTS = 10   # isolated clusters with fewer LiDAR points are dropped


def specks(key, cls, cnt, ground, GW, GH, st) -> np.ndarray:
    """26-connected clusters of non-ground voxels; returns a mask of voxels in clusters with < SPECK_POINTS points."""
    vx, vy, vz = key % GW, (key // GW) % GH, key // (GW * GH)
    m = cls != C_GROUND
    if not m.any():
        return np.zeros(len(key), bool)
    z0, z1 = vz[m].min(), vz[m].max()
    occ = np.zeros((z1 - z0 + 1, GH, GW), bool)
    occ[vz[m] - z0, vy[m], vx[m]] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3), bool))
    del occ
    vl = np.zeros(len(key), np.int64)
    vl[m] = lab[vz[m] - z0, vy[m], vx[m]]
    del lab
    pts = np.bincount(vl[m], weights=np.minimum(cnt[m], 1 << 20), minlength=n + 1)
    drop = m & (pts[vl] < SPECK_POINTS)
    st["specks"] += int(drop.sum())
    # small clusters floating in the air (birds, cable bits, scanner noise)
    nvox = np.bincount(vl[m], minlength=n + 1)
    h = vz - ground[vy, vx]
    hmin = np.full(n + 1, 1 << 30)
    np.minimum.at(hmin, vl[m], h[m])
    fl = (nvox < FLOAT_VOX) & (hmin > FLOAT_H)
    fl[0] = False
    fd = m & fl[vl] & ~drop & (cls != C_BUILDING)
    st["floating"] = st.get("floating", 0) + int(fd.sum())
    drop |= fd
    return drop


def tidy_candidates(akey, acls, aflag, ground, is_bld, gx0, gy0, GW, GH, st):
    """voxels of a block to drop unless the street photos recognise them:
      - low (<= TIDY_H) unclassified voxels not part of a taller structure nor attached to a building
      - (street tidy) in the carriageway: unclassified up to ROAD_H unless part of something reaching above it
        (trees, lamp posts, poles); on pavements: low clutter also next to the facades (bikes, bins)"""
    ax, ay, az = akey % GW, (akey // GW) % GH, akey // (GW * GH)
    u = (acls == C_UNCLASSIFIED) & (aflag & FLAG_FILL == 0)
    h = az - ground[ay, ax]
    low = u & (h <= TIDY_H)
    labels = None

    def reaching_above(sel, hmin):
        """of the voxels in sel: part of a 26-connected unclassified structure reaching above hmin"""
        nonlocal labels
        if labels is None:   # (one labelling per block for both questions)
            uz0 = int(az[u].min())
            grid = np.zeros((int(az[u].max()) - uz0 + 1, GH, GW), bool)
            grid[az[u] - uz0, ay[u], ax[u]] = True
            labels = (*ndimage.label(grid, structure=np.ones((3, 3, 3))), uz0)
        lab, n, uz0 = labels
        tall = np.zeros(n + 1, bool)
        hi = u & (h > hmin)
        tall[lab[az[hi] - uz0, ay[hi], ax[hi]]] = True
        tall[0] = False
        return tall[lab[az[sel] - uz0, ay[sel], ax[sel]]]

    part_of_tall = np.zeros(len(akey), bool)
    if low.any():
        part_of_tall[low] = reaching_above(low, TIDY_H)
        cand = low & ~part_of_tall & ~ndimage.binary_dilation(is_bld, iterations=2)[ay, ax]
    else:
        cand = low
    zb = zones_block(gx0, gy0, GW, GH)
    if zb is not None and u.any():
        zone = zb[ay, ax]
        rz = u & (zone == Z_ROAD) & (h <= ROAD_H)
        if rz.any():
            road_c = np.zeros(len(akey), bool)
            road_c[np.flatnonzero(rz)[~reaching_above(rz, ROAD_H + 1)]] = True
            st["street_road"] = st.get("street_road", 0) + int(road_c.sum())
            cand = cand | road_c
        if low.any():
            pv = low & ~part_of_tall & (zone == Z_PAVE) & ~cand
            st["street_pave"] = st.get("street_pave", 0) + int(pv.sum())
            cand = cand | pv
    return cand


def voxelize_block(p: np.ndarray, gx0: int, gy0: int, GW: int, GH: int, z0: float, st: dict) -> Vox:
    """Steps a-g on one block (+margin). Returns L0 voxels in global column coords."""
    x, y, z = p["x"].astype(np.float64), p["y"].astype(np.float64), p["z"].astype(np.float64)
    rgb = np.stack([p["r"], p["g"], p["b"]], 1)
    cls = p["c"]
    ix = (x / VS).astype(np.int64) - gx0
    iy = (y / VS).astype(np.int64) - gy0
    iz = np.floor((z - z0) / VS).astype(np.int64)
    ok = (iz >= 0) & (iz < NZ) & (ix >= 0) & (ix < GW) & (iy >= 0) & (iy < GH)
    st["dropped_z"] += int(((iz < 0) | (iz >= NZ)).sum())
    x, y, z, ix, iy, iz, rgb, cls = x[ok], y[ok], z[ok], ix[ok], iy[ok], iz[ok], rgb[ok], cls[ok]

    # a. points -> voxels
    vkey, col, vcls, _, cnt = aggregate((iz * GH + iy) * GW + ix, rgb, cls)
    keep = cnt >= 2
    vkey, col, vcls, cnt = vkey[keep], col[keep], vcls[keep], cnt[keep]
    vx, vy, vz = vkey % GW, (vkey // GW) % GH, vkey // (GW * GH)

    # b. columns: ground and roofs
    BIG = np.iinfo(np.int64).max
    ground = np.full((GH, GW), BIG, np.int64)
    g = vcls == C_GROUND
    np.minimum.at(ground, (vy[g], vx[g]), vz[g])
    has_ground = ground != BIG
    gcol = np.zeros((GH, GW, 3), np.uint8)
    gcol[vy[g], vx[g]] = col[g]
    ground = ndimage.median_filter(fill_nearest(ground, has_ground), size=3)
    ground = np.where(ground == BIG, 0, ground)
    # LiDAR RGB comes from a top-down aerial photo: ground under a tree crown gets the crown's
    # colour. Distrust it and continue the surrounding open ground (smoothed) under the canopy.
    hv = (vcls == C_HIGHVEG) & (vz > ground[vy, vx] + 4)
    canopy = np.zeros((GH, GW), bool)
    canopy[vy[hv], vx[hv]] = True
    canopy = ndimage.binary_dilation(canopy, iterations=1)
    open_g = has_ground & ~canopy
    if open_g.any() and canopy.any():
        w = ndimage.uniform_filter(open_g.astype(np.float32), size=5)
        sm = np.stack([ndimage.uniform_filter(np.where(open_g, gcol[..., c], 0).astype(np.float32), size=5)
                       for c in range(3)], -1) / np.maximum(w, 1e-6)[..., None]
        gcol = np.where(open_g[..., None], gcol, fill_nearest(sm.round().clip(0, 255).astype(np.uint8), open_g))
        under = g & canopy[vy, vx]
        col[under] = gcol[vy[under], vx[under]]
        st["under_canopy"] += int(under.sum())
    gcol = fill_nearest(gcol, has_ground | canopy)
    btop = np.full((GH, GW), -1, np.int64)
    b = vcls == C_BUILDING
    np.maximum.at(btop, (vy[b], vx[b]), vz[b])
    bcol = np.zeros((GH, GW, 3), np.uint8)
    bt = b & (vz == btop[vy, vx])
    bcol[vy[bt], vx[bt]] = col[bt]
    is_bld = btop > ground

    # c. solid shell
    top = np.where(is_bld, btop, ground)
    pad = np.pad(top, 1, mode="edge")
    nmin = np.minimum.reduce([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
    lo = np.maximum(np.minimum(top, nmin + 1), 0)
    n = (top - lo + 1).ravel()
    fy, fx = np.divmod(np.repeat(np.arange(GW * GH), n), GW)
    starts = np.repeat(np.cumsum(n) - n, n)
    fz = np.repeat(lo.ravel(), n) + (np.arange(n.sum()) - starts)
    fb = is_bld[fy, fx] & (fz > ground[fy, fx])
    wall = np.array([196, 170, 140], np.float32)
    fcol = np.where(fb[:, None], (0.45 * bcol[fy, fx] + 0.55 * wall),
                    gcol[fy, fx] * np.where(fz < ground[fy, fx], 0.8, 1.0)[:, None]).astype(np.uint8)
    fcls = np.where(fb, C_BUILDING, C_GROUND).astype(np.uint8)
    fflag = np.full(len(fz), FLAG_FILL, np.uint8)
    fkey = (fz * GH + fy) * GW + fx

    # d. facade colours from points
    isb2 = is_bld & ((btop - ground) * VS > 2.5)
    band = ndimage.binary_dilation(isb2, iterations=2) & ~ndimage.binary_erosion(isb2, iterations=2)
    roof_m = z0 + (ndimage.maximum_filter(np.where(isb2, btop, -1), size=5) + 1) * VS
    ground_m = z0 + (ground + 1) * VS
    fac = (band[iy, ix] & (z - ground_m[iy, ix] > 0.5) & (z < roof_m[iy, ix] - 0.3)
           & np.isin(cls, [C_UNCLASSIFIED, C_BUILDING]))
    wall_vox = np.nonzero(fb & (fz <= btop[fy, fx]))[0]
    st["wall"] += len(wall_vox)
    if fac.sum() >= 8 and len(wall_vox):
        tree = cKDTree(np.stack([x[fac], y[fac], z[fac]], 1))
        frgb = rgb[fac].astype(np.float32)
        centres = np.stack([(gx0 + fx[wall_vox] + 0.5) * VS, (gy0 + fy[wall_vox] + 0.5) * VS,
                            z0 + (fz[wall_vox] + 0.5) * VS], 1)
        coloured = np.zeros(len(wall_vox), bool)
        for radius in (0.9, 2.0):
            todo = np.nonzero(~coloured)[0]
            if not len(todo):
                break
            dist, idx = tree.query(centres[todo], k=8, distance_upper_bound=radius, workers=1)
            valid = np.isfinite(dist)
            w = np.where(valid, 1.0 / (dist + 0.15), 0.0)
            wsum = w.sum(1)
            hit = wsum > 0
            idx = np.where(valid, idx, 0)
            c = (frgb[idx] * w[:, :, None]).sum(1) / np.maximum(wsum, 1e-9)[:, None]
            sel = todo[hit]
            fcol[wall_vox[sel]] = c[hit].round().clip(0, 255).astype(np.uint8)
            fflag[wall_vox[sel]] |= FLAG_PTCOLOR
            coloured[sel] = True
        st["wall_coloured"] += int(coloured.sum())

    vis = (vz >= lo[vy, vx]) | (vz > top[vy, vx])
    vis &= vz >= ground[vy, vx] - 1
    near_bld = ndimage.binary_dilation(isb2, iterations=1) & ~is_bld
    eaves = ndimage.maximum_filter(np.where(isb2, btop, -1), size=3)
    loose = (near_bld[vy, vx] & np.isin(vcls, [C_UNCLASSIFIED, C_BUILDING])
             & (vz > ground[vy, vx] + 1) & (vz < eaves[vy, vx]))
    st["absorbed"] += int((vis & loose).sum())
    vis &= ~loose
    vkey, col, vcls, cnt = vkey[vis], col[vis], vcls[vis], cnt[vis]

    # e. parked vehicles (the LiDAR has no class for them): blobs of unclassified voxels standing
    #    0.5-3 m on the ground, car-width (rows of parked cars are long but never wide), nothing tall above
    vx, vy, vz = vkey % GW, (vkey // GW) % GH, vkey // (GW * GH)
    h = vz - ground[vy, vx]
    u = vcls == C_UNCLASSIFIED
    low = u & (h >= 1) & (h <= 6)
    foot = np.zeros((GH, GW), bool)
    foot[vy[low], vx[low]] = True
    tall = np.zeros((GH, GW), bool)
    tall[vy[u & (h > 6)], vx[u & (h > 6)]] = True
    foot &= ~is_bld & ~ndimage.binary_dilation(tall, iterations=1)
    foot = ndimage.binary_opening(foot)                     # drop bollards, fence lines, specks
    lab, n = ndimage.label(foot)
    if n:
        idx = np.arange(1, n + 1)
        area = ndimage.sum(foot, lab, idx) * VS * VS
        half = ndimage.maximum(ndimage.distance_transform_edt(foot), lab, idx) * VS   # half width, m
        ok = np.r_[False, (area >= 2.5) & (half >= 0.6) & (half <= 1.6)]
        veh = low & ok[lab[vy, vx]]
        vcls[veh] = C_VEHICLE
        st["vehicle"] += int(veh.sum())
        if veh.any():
            # 2021's parked cars leave the geometry (the ground under them is in the fill below); their
            # footprint columns are kept: where cars stood is where people park (simulated parking)
            fp = np.zeros((GH, GW), bool)
            fp[vy[veh], vx[veh]] = True
            st["footprint"] = np.flatnonzero(fp.ravel())
            st["vehicle_rgb"] = np.asarray(col[veh][::4]).astype(np.uint8)   # (a sample: car paint palette, vehicles.py)
            vkey, col, vcls, cnt = vkey[~veh], col[~veh], vcls[~veh], cnt[~veh]
            # the ground where the cars stood took the colour of the nearest measured ground: their shadow
            # in the aerial photo. Continue the open road around them instead (as under tree crowns).
            fpd = ndimage.binary_dilation(fp, iterations=3)
            open_r = has_ground & ~canopy & ~fpd
            if open_r.any():
                road = open_colour(gcol, open_r)
                fsel = fpd[fy, fx] & (fcls == C_GROUND)
                fcol[fsel] = (road[fy[fsel], fx[fsel]] * np.where(fz[fsel] < ground[fy[fsel], fx[fsel]], 0.8, 1.0)[:, None]).astype(np.uint8)
                vy2, vx2 = (vkey // GW) % GH, vkey % GW
                gsel = (vcls == C_GROUND) & fpd[vy2, vx2]
                col[gsel] = road[vy2[gsel], vx2[gsel]]
                st["under_car"] = st.get("under_car", 0) + int(fsel.sum() + gsel.sum())
    new = ~np.isin(fkey, vkey, assume_unique=True)
    akey = np.concatenate([vkey, fkey[new]])
    acol = np.concatenate([col, fcol[new]])
    acls = np.concatenate([vcls, fcls[new]])
    aflag = np.concatenate([np.zeros(len(vkey), np.uint8), fflag[new]])
    acnt = np.concatenate([cnt, np.full(int(new.sum()), 1 << 20)])   # fill voxels: solid by construction

    # f. tidy: low, loose unclassified voxels are candidates to drop (unless the photos recognise them)
    cand = tidy_candidates(akey, acls, aflag, ground, is_bld, gx0, gy0, GW, GH, st)
    if cand.any():   # the ground under the clutter: continue the open ground (not its shadow)
        ax, ay = akey % GW, (akey // GW) % GH
        cc = np.zeros((GH, GW), bool)
        cc[ay[cand], ax[cand]] = True
        ccd = ndimage.binary_dilation(cc, iterations=2)
        open_r = has_ground & ~canopy & ~ccd
        if open_r.any():
            road = open_colour(gcol, open_r)
            gs = (acls == C_GROUND) & ccd[ay, ax]
            acol[gs] = road[ay[gs], ax[gs]]

    # g. isolated specks: small clusters (not counting the ground they stand on) backed by few points
    drop = specks(akey, acls, acnt, ground, GW, GH, st)
    akey, acol, acls, aflag = akey[~drop], acol[~drop], acls[~drop], aflag[~drop]
    if cand is not None:
        st["tidy_cand"] = cand[~drop]
    return Vox(akey % GW + gx0, (akey // GW) % GH + gy0, akey // (GW * GH), acol, acls, aflag)


# --- street zones (from the road network of an earlier build: roads.py, OSM + measured pavement lines) ---
ZONES_F = ROOT / "data/cache/zones" / f"{ROI['name']}.npy"
ROADS_F = ROOT / "viewer/public/roads" / f"{ROI['name']}.json"
Z_ROAD, Z_PAVE = 1, 2
_zones = None


def build_zones() -> bool:
    """Per L0 column: 1 carriageway, 2 pavement (kerb to just short of the facade), 0 elsewhere. Kerb and
    facade from the pavement walking lines roads.py measured (facade - 1.5 m; kerb ~1 m inside it), widths
    capped to plausible streets (squares and parks are not road)."""
    if not ROADS_F.exists():
        return False
    from PIL import Image, ImageDraw
    net = read_roads(ROADS_F)
    img = Image.new("L", (NX, NY), 0)
    dr = ImageDraw.Draw(img)
    road_polys, pave_polys = [], []
    for e in net["edges"] + net.get("pieces", []):   # (pieces: road inside merged junctions)
        if e[E_KIND] != KIND_ROAD:
            continue
        p = np.array(e[E_XYZ], float).reshape(-1, 3)[:, :2]
        if len(p) < 2:
            continue
        d = np.gradient(p, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        left = np.stack([-d[:, 1], d[:, 0]], 1)
        lanes = e[E_LANES] if e[E_LANES] else (1 if e[E_ONEWAY] else 2)
        cap = (5.0 if e[E_ONEWAY] else 6.5) + max(0, lanes - 2) * 1.6
        k, f = [], []
        for side, sgn in ((0, 1.0), (1, -1.0)):
            pav = e[E_PAVEMENT][side]
            kerb = min(pav - 1.0, cap) if pav > 0 else min(4.0, cap)
            face = min(pav + 0.9, kerb + 4.0) if pav > 0 else kerb
            k.append(p + sgn * left * (kerb / VS))
            f.append(p + sgn * left * (face / VS))
        road_polys.append(np.vstack([k[0], k[1][::-1]]))
        for side in (0, 1):
            pave_polys.append(np.vstack([k[side], f[side][::-1]]))
    for poly in pave_polys:
        dr.polygon([tuple(q) for q in poly], fill=Z_PAVE)
    for poly in road_polys:
        dr.polygon([tuple(q) for q in poly], fill=Z_ROAD)
    ZONES_F.parent.mkdir(parents=True, exist_ok=True)
    np.save(ZONES_F, np.asarray(img, np.uint8))   # [y, x]
    return True


def zones_block(gx0, gy0, GW, GH):
    global _zones
    if _zones is None:
        _zones = np.load(ZONES_F, mmap_mode="r") if ZONES_F.exists() else False
    if _zones is False:
        return None
    return np.asarray(_zones[gy0:gy0 + GH, gx0:gx0 + GW])


def process_block(bx: int, by: int, z0: float):
    """Pass 2 for one block (runs in a worker process). Writes L0..L3 tiles into DEST."""
    st = {"dropped_z": 0, "wall": 0, "wall_coloured": 0, "absorbed": 0, "under_canopy": 0, "photo": 0, "labelled": 0, "vehicle": 0, "specks": 0, "tidy": 0}
    levels = [{"tiles": [], "voxels": 0} for _ in range(LEVELS)]
    parts = sorted(CACHE.glob(f"b{by * NBX + bx}_*.bin"))
    if not parts:
        return levels, st, None, None
    p = np.concatenate([np.fromfile(f, PT) for f in parts])
    cx0, cy0 = bx * BLOCK, by * BLOCK
    cx1, cy1 = min(cx0 + BLOCK, NX), min(cy0 + BLOCK, NY)
    gx0, gy0 = max(cx0 - MARGIN, 0), max(cy0 - MARGIN, 0)
    gx1, gy1 = min(cx1 + MARGIN, NX), min(cy1 + MARGIN, NY)
    v = voxelize_block(p, gx0, gy0, gx1 - gx0, gy1 - gy0, z0, st)
    del p
    # vehicle footprint columns inside this block (global column coordinates)
    fp = st.pop("footprint", None)
    foot = None
    if fp is not None:
        fx, fy = fp % (gx1 - gx0) + gx0, fp // (gx1 - gx0) + gy0
        inb = (fx >= cx0) & (fx < cx1) & (fy >= cy0) & (fy < cy1)
        foot = np.stack([fx[inb], fy[inb]], 1).astype(np.uint16)
    if LAYER:
        # street-level photo colours (streetview.py), keyed by L0 voxel; coarser levels inherit them
        k = (v.z.astype(np.int64) * NY + v.y) * NX + v.x
        pos = np.searchsorted(LAYER["key"], k).clip(0, len(LAYER["key"]) - 1)
        m = LAYER["key"][pos] == k
        v.col[m] = LAYER["rgb"][pos[m]]
        v.flag[m] |= FLAG_PHOTO
        st["photo"] += int(m.sum())
    # semantic class: from the LiDAR class, refined by street-level labels where sessions agree
    sem = SEM_FROM_LIDAR[v.cls]
    if LAYER:
        pos = np.searchsorted(LAYER["lkey"], k).clip(0, len(LAYER["lkey"]) - 1)
        m = LAYER["lkey"][pos] == k
        lab = LAYER["label"][pos[m]]
        cls = v.cls[m]
        # photos only fill in what the LiDAR left unclassified (and shop signs on buildings);
        # never sky (a mismatch, not a class) and never things that move
        use = (((cls == C_UNCLASSIFIED) | ((cls == C_BUILDING) & (lab == SEM_SIGN)))
               & (lab != SEM_SKY) & ~np.isin(lab, SEM_TRANSIENT))
        idx = np.flatnonzero(m)[use]
        sem[idx] = lab[use]
        st["labelled"] += len(idx)
    v.flag = (v.flag & ((1 << LABEL_SHIFT) - 1)) | (sem << LABEL_SHIFT).astype(np.uint8)
    cand = st.pop("tidy_cand", None)
    if cand is not None:   # --tidy: drop the loose low clutter the photos did not recognise
        drop = cand & ~np.isin(sem, SEM_KEEP)
        st["tidy"] += int(drop.sum())
        keep = ~drop
        v = Vox(v.x[keep], v.y[keep], v.z[keep], v.col[keep], v.cls[keep], v.flag[keep])
    carry = None
    BL = min(LEVELS, 4)
    for L in range(BL):
        if L:
            v = v.coarser()
        s = 1 << L
        x0, x1, y0, y1 = cx0 // s, ceil_div(cx1, s), cy0 // s, ceil_div(cy1, s)
        levels[L]["tiles"] += write_tiles(L, v, range(x0 // CH, ceil_div(x1, CH)), range(y0 // CH, ceil_div(y1, CH)))
        levels[L]["voxels"] += int(((v.x >= x0) & (v.x < x1) & (v.y >= y0) & (v.y < y1)).sum())
        if L == BL - 1:
            carry = v.select(x0, x1, y0, y1)
    return levels, st, carry, foot


def main() -> None:
    global DEST
    jobs = cpu_jobs()
    log(f"region '{ROI['name']}': {NX}x{NY} columns @ {VS} m, {NBX}x{NBY} blocks of {BLOCK * VS:.0f} m, "
        f"{LEVELS} levels, {jobs} parallel jobs")
    z0 = bucket_points(jobs)
    if LAYER_F.exists():
        lf = np.load(LAYER_F)
        if tuple(lf["size"]) == (NX, NY):
            ok = lf["lnses"] >= LABEL_MIN_SESSIONS
            LAYER.update(key=lf["key"], rgb=lf["rgb"], lkey=lf["lkey"][ok], label=lf["label"][ok])
            log(f"street-level layer: {len(LAYER['key']):,} coloured, {ok.sum():,} labelled voxels")
        else:
            log("street-level layer is for another region size; ignored")
    log(f"z0 = {z0} m (height range {z0}–{z0 + NZ * VS} m)")
    log("street zones from the road network: " + ("built" if build_zones() else "no roads.json yet (run roads.py; then build again)"))

    # build into a staging folder; the viewer keeps serving the old build until the swap
    DEST = OUT.with_name(OUT.name + ".building")
    shutil.rmtree(DEST, ignore_errors=True)
    for L in range(LEVELS):
        (DEST / f"L{L}").mkdir(parents=True)

    levels = [{"tiles": [], "voxels": 0} for _ in range(LEVELS)]
    st = {"dropped_z": 0, "wall": 0, "wall_coloured": 0, "absorbed": 0, "under_canopy": 0, "photo": 0, "labelled": 0, "vehicle": 0, "specks": 0, "tidy": 0}
    carry, feet, paint = [], [], []
    blocks = [(bx, by) for by in range(NBY) for bx in range(NBX)]
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("fork")) as ex:
        futs = [ex.submit(process_block, bx, by, z0) for bx, by in blocks]
        for i, fut in enumerate(as_completed(futs), 1):
            lvs, bst, c, foot = fut.result()
            if foot is not None:
                feet.append(foot)
            if "vehicle_rgb" in bst:
                paint.append(bst.pop("vehicle_rgb"))
            for L, lv in enumerate(lvs):
                levels[L]["voxels"] += lv["voxels"]
                levels[L]["tiles"] += lv["tiles"]
            for k, v in bst.items():
                if isinstance(v, (int, float)):
                    st[k] = st.get(k, 0) + v
            if c is not None:
                carry.append(c)
            if i % max(1, len(blocks) // 6) == 0 or i == len(blocks):
                log(f"blocks {i}/{len(blocks)}: {levels[0]['voxels']:,} L0 voxels")

    if feet:   # where 2021's parked cars stood (columns), for parking spots
        FOOTPRINTS.mkdir(parents=True, exist_ok=True)
        np.save(FOOTPRINTS / f"{ROI['name']}.npy", np.concatenate(feet))
        if paint:
            np.save(FOOTPRINTS / f"{ROI['name']}_rgb.npy", np.concatenate(paint))
        log(f"vehicles removed from the geometry; {sum(len(f) for f in feet):,} footprint columns kept")
    # pass 3: levels spanning several blocks
    BL = min(LEVELS, 4)
    v = Vox.concat(carry) if carry else None
    for L in range(BL, LEVELS):
        v = v.coarser()
        s = 1 << L
        levels[L].update(tiles=write_tiles(L, v, range(ceil_div(ceil_div(NX, s), CH)), range(ceil_div(ceil_div(NY, s), CH))), voxels=len(v))

    log(f"walls: {st['wall_coloured']:,}/{st['wall']:,} voxels coloured from facade points "
        f"({st['wall_coloured'] / max(st['wall'], 1):.1%}); absorbed {st['absorbed']:,} loose facade voxels; "
        f"{st['under_canopy']:,} ground voxels under tree crowns recoloured from open ground; "
        f"{st['photo']:,} voxels coloured and {st['labelled']:,} classified from street-level photos; "
        f"{st['vehicle']:,} vehicle voxels found in unclassified geometry; {st['specks']:,} voxels in isolated specks dropped; {st['tidy']:,} loose low voxels tidied ({st.get('street_road', 0):,} candidates in the carriageway, {st.get('street_pave', 0):,} on pavements); {st.get('floating', 0):,} floating voxels dropped; "
        f"{st['dropped_z']:,} points outside height range")
    out_levels = [{"level": L, "voxelSize": VS * (1 << L), "size": [ceil_div(NX, 1 << L), ceil_div(NY, 1 << L), NZ >> L],
                   "tiles": lv["tiles"]} for L, lv in enumerate(levels)]
    sizes = pack_archives(DEST, out_levels, CH)   # (the single tile files -> archives read by range requests)
    for L, lv in enumerate(levels):
        log(f"L{L}: {VS * (1 << L)} m, {lv['voxels']:,} voxels, {len(lv['tiles'])} tiles, {sizes[L] / 1e6:.1f} MB")
    manifest = {
        "version": MANIFEST_VERSION, "build": time.strftime("%Y%m%d-%H%M%S"), "name": ROI["name"], "crs": "EPSG:31370",
        "origin": [ROI["xmin"], ROI["ymin"], z0], "tileSize": CH,
        "home": [CFG.get("view", {}).get("x", (ROI["xmin"] + ROI["xmax"]) / 2),
                 CFG.get("view", {}).get("y", (ROI["ymin"] + ROI["ymax"]) / 2)],
        "levels": out_levels,
        "attribution": "LiDAR © Paradigm / UrbIS – datastore.brussels (CC-BY 4.0)",
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=1))
    # swap: two renames, so the served folder is complete before and after
    old = OUT.with_name(OUT.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if OUT.exists():
        OUT.rename(old)
    DEST.rename(OUT)
    shutil.rmtree(old, ignore_errors=True)
    log(f"build {manifest['build']} live at {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
