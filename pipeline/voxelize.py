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
            e. levels L0..L3 (0.5..4 m) built and written as 64x64-column tiles; the
               margin supplies each tile's 1-voxel apron across block seams
  pass 3  levels above L3 (one tile spans several blocks) built from the collected L3 voxels

Tile (tx,ty) at level L has children (2tx..2tx+1, 2ty..2ty+1) at L-1.

Usage: uv run python voxelize.py [roi-name] [--jobs=N]      (default region from config.toml)

Builds into <name>.building and swaps it in at the end, so a running viewer never
sees a half-written build; the manifest carries a build id for cache busting.
"""

import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import laspy
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
CFG = tomllib.loads((ROOT / "config.toml").read_text())
ROI_NAME = next((a for a in sys.argv[1:] if not a.startswith("-") and not a.endswith(".las")), CFG["default"])
ROI = {"name": ROI_NAME, **CFG["roi"][ROI_NAME]}
VS = CFG.get("voxel", {}).get("size", 0.5)
LEVELS = ROI.get("levels", 4)
CH = 64            # tile size in columns (every level)
NZ = 256           # max voxel layers at L0 (z stored as uint8)
BLOCK = 512        # block size in L0 columns (= one L3 tile = 256 m)
MARGIN = 64        # extra L0 columns read around each block (32 m)
OUT = ROOT / "viewer" / "public" / "tiles" / ROI["name"]
CACHE = ROOT / "data" / "cache" / "points" / ROI["name"]
DEST = OUT  # where tiles are written (a staging folder during a build)

C_UNCLASSIFIED, C_GROUND, C_LOWVEG, C_HIGHVEG, C_BUILDING = 1, 2, 3, 5, 6
FLAG_FILL = 1      # geometry synthesised (wall / terrain fill), not measured
FLAG_APRON = 2     # belongs to the neighbouring tile, used only for culling
FLAG_PTCOLOR = 4   # synthetic voxel whose colour comes from nearby measured points

PT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("c", "u1")])

NX = int(round((ROI["xmax"] - ROI["xmin"]) / VS))
NY = int(round((ROI["ymax"] - ROI["ymin"]) / VS))
NBX, NBY = -(-NX // BLOCK), -(-NY // BLOCK)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# --- pass 1: bucket points per block -----------------------------------------
def bucket_points() -> float:
    key = {"roi": {k: ROI[k] for k in ("xmin", "ymin", "xmax", "ymax", "tiles")}, "block": BLOCK,
           "margin": MARGIN, "vs": VS, "format": 1}
    meta_f = CACHE / "meta.json"
    if meta_f.exists():
        meta = json.loads(meta_f.read_text())
        if meta["key"] == key:
            log(f"pass 1: reusing point cache ({meta['points']:,} points)")
            return meta["z0"]
    CACHE.mkdir(parents=True, exist_ok=True)
    for f in CACHE.glob("*"):
        f.unlink()
    files = {}
    ground_lo, total = [], 0
    for tile in ROI["tiles"]:
        path = next((ROOT / "data/raw/lidar").glob(f"*{tile}*.las"))
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
                    ground_lo.append(np.percentile(g, 0.01))
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
                            fh = files.get(b) or files.setdefault(b, open(CACHE / f"b{b}.bin", "ab"))
                            fh.write(part.tobytes())
        log(f"pass 1: read {path.name} ({total:,} points in region so far)")
    for fh in files.values():
        fh.close()
    z0 = float(np.floor(min(ground_lo))) - 1.0
    meta_f.write_text(json.dumps({"key": key, "z0": z0, "points": total}))
    return z0


# --- helpers ----------------------------------------------------------------
def cpu_quota() -> int:
    """CPUs we may actually use (container quota), not the host core count."""
    try:
        q, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q != "max":
            return max(1, int(q) // int(period))
    except (OSError, ValueError):
        pass
    return max(1, (os.cpu_count() or 2) - 1)


def fill_nearest(grid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.all():
        return grid
    if not valid.any():
        return grid
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return grid[tuple(idx)]


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
        uflag = np.zeros(len(ukey), np.uint8)
        np.bitwise_or.at(uflag, inv, flags)
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


def count_faces(v: Vox, x0, x1, y0, y1) -> int:
    """Visible faces of the voxels inside [x0,x1)x[y0,y1), using all of v as neighbours."""
    W = int(v.x.max()) + 2
    H = int(v.y.max()) + 2
    key = (v.z * H + v.y) * W + v.x
    skey = np.sort(key)
    inside = (v.x >= x0) & (v.x < x1) & (v.y >= y0) & (v.y < y1)
    faces = 0
    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        nx, ny, nz = v.x + dx, v.y + dy, v.z + dz
        nk = (nz * H + ny) * W + nx
        pos = np.searchsorted(skey, nk).clip(0, len(skey) - 1)
        occ = (skey[pos] == nk) & (nx >= 0) & (ny >= 0) & (nz >= 0)
        if dz == -1:
            occ |= (v.flag & FLAG_FILL) > 0
        faces += int((~occ & inside).sum())
    return faces


def write_tiles(L, v: Vox, tx_range, ty_range):
    tiles, nbytes = [], 0
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
            name = f"L{L}/t_{tx}_{ty}.bin"
            (DEST / name).write_bytes(buf.tobytes())
            nbytes += buf.nbytes
            zr = v.z[m][~apron]
            tiles.append({"file": name, "tx": tx, "ty": ty, "count": int((~apron).sum()),
                          "zmin": int(zr.min()), "zmax": int(zr.max())})
    return tiles, nbytes


# --- pass 2: one block ---------------------------------------------------------
def voxelize_block(p: np.ndarray, gx0: int, gy0: int, GW: int, GH: int, z0: float, st: dict) -> Vox:
    """Steps a-d on one block (+margin). Returns L0 voxels in global column coords."""
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
    vkey, col, vcls = vkey[keep], col[keep], vcls[keep]
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
    vkey, col, vcls = vkey[vis], col[vis], vcls[vis]
    new = ~np.isin(fkey, vkey, assume_unique=True)
    akey = np.concatenate([vkey, fkey[new]])
    acol = np.concatenate([col, fcol[new]])
    acls = np.concatenate([vcls, fcls[new]])
    aflag = np.concatenate([np.zeros(len(vkey), np.uint8), fflag[new]])
    return Vox(akey % GW + gx0, (akey // GW) % GH + gy0, akey // (GW * GH), acol, acls, aflag)


def process_block(bx: int, by: int, z0: float):
    """Pass 2 for one block (runs in a worker process). Writes L0..L3 tiles into DEST."""
    st = {"dropped_z": 0, "wall": 0, "wall_coloured": 0, "absorbed": 0, "under_canopy": 0}
    levels = [{"tiles": [], "voxels": 0, "faces": 0, "bytes": 0} for _ in range(LEVELS)]
    f = CACHE / f"b{by * NBX + bx}.bin"
    if not f.exists():
        return levels, st, None
    p = np.fromfile(f, PT)
    cx0, cy0 = bx * BLOCK, by * BLOCK
    cx1, cy1 = min(cx0 + BLOCK, NX), min(cy0 + BLOCK, NY)
    gx0, gy0 = max(cx0 - MARGIN, 0), max(cy0 - MARGIN, 0)
    gx1, gy1 = min(cx1 + MARGIN, NX), min(cy1 + MARGIN, NY)
    v = voxelize_block(p, gx0, gy0, gx1 - gx0, gy1 - gy0, z0, st)
    del p
    carry = None
    BL = min(LEVELS, 4)
    for L in range(BL):
        if L:
            v = v.coarser()
        s = 1 << L
        x0, x1, y0, y1 = cx0 // s, -(-cx1 // s), cy0 // s, -(-cy1 // s)
        tiles, nb = write_tiles(L, v, range(x0 // CH, -(-x1 // CH)), range(y0 // CH, -(-y1 // CH)))
        lv = levels[L]
        lv["tiles"] += tiles
        lv["bytes"] += nb
        lv["voxels"] += int(((v.x >= x0) & (v.x < x1) & (v.y >= y0) & (v.y < y1)).sum())
        lv["faces"] += count_faces(v, x0, x1, y0, y1)
        if L == BL - 1:
            carry = v.select(x0, x1, y0, y1)
    return levels, st, carry


def main() -> None:
    global DEST
    jobs = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--jobs=")), cpu_quota()))
    log(f"region '{ROI['name']}': {NX}x{NY} columns @ {VS} m, {NBX}x{NBY} blocks of {BLOCK * VS:.0f} m, "
        f"{LEVELS} levels, {jobs} parallel jobs")
    z0 = bucket_points()
    log(f"z0 = {z0} m (height range {z0}–{z0 + NZ * VS} m)")

    # build into a staging folder; the viewer keeps serving the old build until the swap
    DEST = OUT.with_name(OUT.name + ".building")
    shutil.rmtree(DEST, ignore_errors=True)
    for L in range(LEVELS):
        (DEST / f"L{L}").mkdir(parents=True)

    levels = [{"tiles": [], "voxels": 0, "faces": 0, "bytes": 0} for _ in range(LEVELS)]
    st = {"dropped_z": 0, "wall": 0, "wall_coloured": 0, "absorbed": 0, "under_canopy": 0}
    carry = []
    blocks = [(bx, by) for by in range(NBY) for bx in range(NBX)]
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("fork")) as ex:
        futs = [ex.submit(process_block, bx, by, z0) for bx, by in blocks]
        for i, fut in enumerate(as_completed(futs), 1):
            lvs, bst, c = fut.result()
            for L, lv in enumerate(lvs):
                for k in ("voxels", "faces", "bytes"):
                    levels[L][k] += lv[k]
                levels[L]["tiles"] += lv["tiles"]
            for k in st:
                st[k] += bst[k]
            if c is not None:
                carry.append(c)
            if i % max(1, len(blocks) // 6) == 0 or i == len(blocks):
                log(f"blocks {i}/{len(blocks)}: {levels[0]['voxels']:,} L0 voxels")

    # pass 3: levels spanning several blocks
    BL = min(LEVELS, 4)
    v = Vox.concat(carry) if carry else None
    for L in range(BL, LEVELS):
        v = v.coarser()
        s = 1 << L
        tiles, nb = write_tiles(L, v, range(0, -(-(-(-NX // s)) // CH)), range(0, -(-(-(-NY // s)) // CH)))
        levels[L].update(tiles=tiles, bytes=nb, voxels=len(v), faces=count_faces(v, 0, 1 << 30, 0, 1 << 30))

    log(f"walls: {st['wall_coloured']:,}/{st['wall']:,} voxels coloured from facade points "
        f"({st['wall_coloured'] / max(st['wall'], 1):.1%}); absorbed {st['absorbed']:,} loose facade voxels; "
        f"{st['under_canopy']:,} ground voxels under tree crowns recoloured from open ground; "
        f"{st['dropped_z']:,} points outside height range")
    out_levels = []
    for L, lv in enumerate(levels):
        lv["tiles"].sort(key=lambda t: (t["ty"], t["tx"]))
        log(f"L{L}: {VS * (1 << L)} m, {lv['voxels']:,} voxels, ~{lv['faces']:,} faces, "
            f"{len(lv['tiles'])} tiles, {lv['bytes'] / 1e6:.1f} MB")
        out_levels.append({"level": L, "voxelSize": VS * (1 << L), "size": [-(-NX // (1 << L)), -(-NY // (1 << L)), NZ >> L],
                           "voxels": lv["voxels"], "faces": lv["faces"], "tiles": lv["tiles"]})
    manifest = {
        "version": 2, "build": time.strftime("%Y%m%d-%H%M%S"), "name": ROI["name"], "crs": "EPSG:31370",
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
