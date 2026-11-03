"""LiDAR point cloud -> sparse coloured voxels -> LOD pyramid of quadtree tiles.

Single-source build: only the Brussels 2021 LiDAR (CC-BY Paradigm/UrbIS).

Steps
  1. read points inside the ROI, bin them into VS-metre voxels
  2. per voxel: mean RGB, majority class, point count (drop 1-point voxels = noise)
  3. per column: ground height (class 2, holes filled by nearest neighbour)
     and building top (class 6)
  4. make buildings and terrain solid, but only emit the voxels that can be
     seen from the side (down to the lowest neighbouring column top)
  5. colour synthetic wall voxels from nearby facade points; absorb the loose
     facade-point voxels that sat in front of the walls
  6. build coarser levels (2x per level, solid if any child is solid)
  7. write each level as 64x64-column tiles with a 1-voxel apron
     (tile (tx,ty) at level L has children (2tx..2tx+1, 2ty..2ty+1) at L-1)

Usage: uv run python voxelize.py
"""

import json
import time
import tomllib
from pathlib import Path

import laspy
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
CFG = tomllib.loads((ROOT / "config.toml").read_text())
ROI = CFG["roi"]
VS = CFG.get("voxel", {}).get("size", 0.5)
LEVELS = CFG.get("voxel", {}).get("levels", 4)
CH = 64            # tile size in columns (every level)
NZ = 256           # max voxel layers at L0 (z stored as uint8)
OUT = ROOT / "viewer" / "public" / "tiles" / ROI["name"]

C_GROUND, C_LOWVEG, C_HIGHVEG, C_BUILDING, C_WATER = 2, 3, 5, 6, 9
C_UNCLASSIFIED = 1
FLAG_FILL = 1      # geometry synthesised (wall / terrain fill), not measured
FLAG_APRON = 2     # belongs to the neighbouring tile, used only for culling
FLAG_PTCOLOR = 4   # synthetic voxel whose colour comes from nearby measured points

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def read_points():
    xs, ys, zs, rgb, cls = [], [], [], [], []
    for tile in CFG["lidar"]["tiles"]:
        path = next((ROOT / "data/raw/lidar").glob(f"*{tile}*.las"))
        with laspy.open(path) as f:
            for ch in f.chunk_iterator(5_000_000):
                x, y = np.asarray(ch.x), np.asarray(ch.y)
                m = (x >= ROI["xmin"]) & (x < ROI["xmax"]) & (y >= ROI["ymin"]) & (y < ROI["ymax"])
                c = np.asarray(ch.classification)[m]
                keep = (c != 7) & (c != 18)  # noise classes
                xs.append(x[m][keep]); ys.append(y[m][keep]); zs.append(np.asarray(ch.z)[m][keep])
                cls.append(c[keep])
                rgb.append(np.stack([np.asarray(ch.red)[m][keep], np.asarray(ch.green)[m][keep],
                                     np.asarray(ch.blue)[m][keep]], 1) >> 8)
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs),
            np.concatenate(rgb).astype(np.uint8), np.concatenate(cls).astype(np.uint8))


def fill_nearest(grid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return grid[tuple(idx)]


def aggregate(key, rgb, cls, flags=None):
    """Group by key: mean colour, majority class, OR of flags, count."""
    ukey, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    col = np.stack([np.bincount(inv, weights=rgb[:, i], minlength=len(ukey)) for i in range(3)], 1)
    col = (col / cnt[:, None]).round().astype(np.uint8)
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


def count_faces(ax, ay, az, aflag, NX, NY):
    key = (az * NY + ay) * NX + ax
    o = np.argsort(key)
    skey = key[o]
    faces = 0
    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        nx_, ny_, nz_ = ax + dx, ay + dy, az + dz
        inside = (nx_ >= 0) & (nx_ < NX) & (ny_ >= 0) & (ny_ < NY) & (nz_ >= 0)
        nk = (nz_ * NY + ny_) * NX + nx_
        pos = np.searchsorted(skey, nk).clip(0, len(skey) - 1)
        occ = inside & (skey[pos] == nk)
        if dz == -1:  # down faces of solid voxels are never seen
            occ |= (aflag & FLAG_FILL) > 0
        faces += int((~occ).sum())
    return faces


def write_level(L, ax, ay, az, acol, acls, aflag, NXL, NYL):
    d = OUT / f"L{L}"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.bin"):
        old.unlink()
    ntx, nty = -(-NXL // CH), -(-NYL // CH)
    tiles, total = [], 0
    for ty in range(nty):
        ym = (ay >= ty * CH - 1) & (ay <= ty * CH + CH)
        for tx in range(ntx):
            m = ym & (ax >= tx * CH - 1) & (ax <= tx * CH + CH)
            if not m.any():
                continue
            lx, ly = ax[m] - tx * CH, ay[m] - ty * CH
            apron = (lx < 0) | (lx >= CH) | (ly < 0) | (ly >= CH)
            if apron.all():
                continue
            buf = np.empty((m.sum(), 8), np.uint8)
            buf[:, 0], buf[:, 1], buf[:, 2] = lx + 1, ly + 1, az[m]
            buf[:, 3:6] = acol[m]
            buf[:, 6] = acls[m]
            buf[:, 7] = aflag[m] | np.where(apron, FLAG_APRON, 0)
            name = f"L{L}/t_{tx}_{ty}.bin"
            (OUT / name).write_bytes(buf.tobytes())
            total += buf.nbytes
            zr = az[m][~apron]
            tiles.append({"file": name, "tx": tx, "ty": ty, "count": int((~apron).sum()),
                          "zmin": int(zr.min()), "zmax": int(zr.max())})
    return tiles, total


def main() -> None:
    x, y, z, rgb, cls = read_points()
    log(f"{len(x):,} points in ROI")

    NX = int(round((ROI["xmax"] - ROI["xmin"]) / VS))
    NY = int(round((ROI["ymax"] - ROI["ymin"]) / VS))
    z0 = float(np.floor(np.percentile(z[cls == C_GROUND], 0.01))) - 1.0
    ix = ((x - ROI["xmin"]) / VS).astype(np.int64)
    iy = ((y - ROI["ymin"]) / VS).astype(np.int64)
    iz = np.floor((z - z0) / VS).astype(np.int64)
    ok = (iz >= 0) & (iz < NZ)
    x, y, z, ix, iy, iz, rgb, cls = x[ok], y[ok], z[ok], ix[ok], iy[ok], iz[ok], rgb[ok], cls[ok]
    log(f"grid {NX}x{NY}x{NZ} @ {VS} m, z0 = {z0} m; dropped {(~ok).sum():,} out-of-range points")

    # --- 2. aggregate points into voxels -----------------------------------
    vkey, col, vcls, _, cnt = aggregate((iz * NY + iy) * NX + ix, rgb, cls)
    keep = cnt >= 2
    vkey, col, vcls = vkey[keep], col[keep], vcls[keep]
    vx, vy, vz = vkey % NX, (vkey // NX) % NY, vkey // (NX * NY)
    log(f"{len(vkey):,} measured voxels (dropped {(~keep).sum():,} single-point voxels)")

    # --- 3. column analysis -------------------------------------------------
    BIG = np.iinfo(np.int64).max
    ground = np.full((NY, NX), BIG, np.int64)
    g = vcls == C_GROUND
    np.minimum.at(ground, (vy[g], vx[g]), vz[g])
    has_ground = ground != BIG
    gcol = np.zeros((NY, NX, 3), np.uint8)
    gcol[vy[g], vx[g]] = col[g]
    ground = ndimage.median_filter(fill_nearest(ground, has_ground), size=3)
    gcol = fill_nearest(gcol, has_ground)
    log(f"ground measured in {has_ground.mean():.1%} of columns, rest filled by nearest")

    btop = np.full((NY, NX), -1, np.int64)
    b = vcls == C_BUILDING
    np.maximum.at(btop, (vy[b], vx[b]), vz[b])
    bcol = np.zeros((NY, NX, 3), np.uint8)
    bt = b & (vz == btop[vy, vx])
    bcol[vy[bt], vx[bt]] = col[bt]
    is_bld = btop > ground
    log(f"building columns: {is_bld.mean():.1%}")

    # --- 4. solid fill, culled to what is visible from the side -------------
    top = np.where(is_bld, btop, ground)
    pad = np.pad(top, 1, mode="edge")
    nmin = np.minimum.reduce([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
    lo = np.maximum(np.minimum(top, nmin + 1), 0)
    n = (top - lo + 1).ravel()
    fy, fx = np.divmod(np.repeat(np.arange(NX * NY), n), NX)
    starts = np.repeat(np.cumsum(n) - n, n)
    fz = np.repeat(lo.ravel(), n) + (np.arange(n.sum()) - starts)
    fb = is_bld[fy, fx] & (fz > ground[fy, fx])
    wall = np.array([196, 170, 140], np.float32)
    fcol = np.where(fb[:, None], (0.45 * bcol[fy, fx] + 0.55 * wall),
                    gcol[fy, fx] * np.where(fz < ground[fy, fx], 0.8, 1.0)[:, None]).astype(np.uint8)
    fcls = np.where(fb, C_BUILDING, C_GROUND).astype(np.uint8)
    fflag = np.full(len(fz), FLAG_FILL, np.uint8)
    fkey = (fz * NY + fy) * NX + fx
    log(f"{len(fkey):,} fill voxels (visible shell only)")

    # --- 5. facade colours from points -------------------------------------
    # facade points: near a footprint edge, above ground, below the local eaves, not vegetation/ground
    isb2 = is_bld & ((btop - ground) * VS > 2.5)
    band = ndimage.binary_dilation(isb2, iterations=2) & ~ndimage.binary_erosion(isb2, iterations=2)
    roof_m = z0 + (ndimage.maximum_filter(np.where(isb2, btop, -1), size=5) + 1) * VS
    ground_m = z0 + (ground + 1) * VS
    h = z - ground_m[iy, ix]
    fac = band[iy, ix] & (h > 0.5) & (z < roof_m[iy, ix] - 0.3) & np.isin(cls, [C_UNCLASSIFIED, C_BUILDING])
    tree = cKDTree(np.stack([x[fac], y[fac], z[fac]], 1))
    frgb = rgb[fac].astype(np.float32)
    log(f"{fac.sum():,} facade points")

    wall_vox = np.nonzero(fb & (fz <= btop[fy, fx]))[0]
    centres = np.stack([ROI["xmin"] + (fx[wall_vox] + 0.5) * VS, ROI["ymin"] + (fy[wall_vox] + 0.5) * VS,
                        z0 + (fz[wall_vox] + 0.5) * VS], 1)
    coloured = np.zeros(len(wall_vox), bool)
    for radius in (0.9, 2.0):
        todo = np.nonzero(~coloured)[0]
        if not len(todo):
            break
        dist, idx = tree.query(centres[todo], k=8, distance_upper_bound=radius, workers=-1)
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
        log(f"  wall voxels coloured within {radius} m: {hit.sum():,}")
    log(f"wall voxels with point colour: {coloured.sum():,} / {len(wall_vox):,} ({coloured.mean():.1%})")

    # measured voxels inside the solid (below lo) are hidden -> drop; below ground -> noise
    vis = (vz >= lo[vy, vx]) | (vz > top[vy, vx])
    vis &= vz >= ground[vy, vx] - 1
    # loose facade hits: non-building column right next to a building, between ground and eaves.
    # Their colour now lives in the wall, so drop them.
    near_bld = ndimage.binary_dilation(isb2, iterations=1) & ~is_bld
    eaves = ndimage.maximum_filter(np.where(isb2, btop, -1), size=3)
    loose = (near_bld[vy, vx] & np.isin(vcls, [C_UNCLASSIFIED, C_BUILDING])
             & (vz > ground[vy, vx] + 1) & (vz < eaves[vy, vx]))
    log(f"absorbed {(vis & loose).sum():,} loose facade voxels into walls")
    vis &= ~loose
    vkey, col, vcls = vkey[vis], col[vis], vcls[vis]
    # merge: measured voxels win over fill
    new = ~np.isin(fkey, vkey, assume_unique=True)
    akey = np.concatenate([vkey, fkey[new]])
    acol = np.concatenate([col, fcol[new]])
    acls = np.concatenate([vcls, fcls[new]])
    aflag = np.concatenate([np.zeros(len(vkey), np.uint8), fflag[new]])
    o = np.argsort(akey)
    akey, acol, acls, aflag = akey[o], acol[o], acls[o], aflag[o]
    ax, ay, az = akey % NX, (akey // NX) % NY, akey // (NX * NY)
    log(f"{len(akey):,} voxels at L0")

    # --- 6 + 7. LOD pyramid, written as tiles -------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.bin"):  # v1 flat chunks
        old.unlink()
    levels = []
    for L in range(LEVELS):
        if L == 0:
            lx, ly, lz, lcol, lcls, lflag = ax, ay, az, acol, acls, aflag
        else:
            NXp, NYp = -(-NX // (1 << (L - 1))), -(-NY // (1 << (L - 1)))
            NXL_, NYL_ = -(-NXp // 2), -(-NYp // 2)
            k = ((pz // 2) * NYL_ + py // 2) * NXL_ + px // 2
            ukey, lcol, lcls, lflag, _ = aggregate(k, pcol, pcls, pflag)
            lx, ly, lz = ukey % NXL_, (ukey // NXL_) % NYL_, ukey // (NXL_ * NYL_)
        NXL, NYL = -(-NX // (1 << L)), -(-NY // (1 << L))
        tiles, nbytes = write_level(L, lx, ly, lz, lcol, lcls, lflag, NXL, NYL)
        faces = count_faces(lx, ly, lz, lflag, NXL, NYL)
        levels.append({"level": L, "voxelSize": VS * (1 << L), "size": [NXL, NYL, NZ >> L],
                       "voxels": int(len(lx)), "faces": faces, "tiles": tiles})
        log(f"L{L}: {VS * (1 << L)} m, {len(lx):,} voxels, ~{faces:,} faces, {len(tiles)} tiles, {nbytes / 1e6:.1f} MB")
        px, py, pz, pcol, pcls, pflag = lx, ly, lz, lcol, lcls, lflag

    manifest = {
        "version": 2,
        "name": ROI["name"], "crs": "EPSG:31370",
        "origin": [ROI["xmin"], ROI["ymin"], z0], "tileSize": CH,
        "home": [CFG.get("view", {}).get("x", (ROI["xmin"] + ROI["xmax"]) / 2),
                 CFG.get("view", {}).get("y", (ROI["ymin"] + ROI["ymax"]) / 2)],
        "levels": levels,
        "attribution": "LiDAR © Paradigm / UrbIS – datastore.brussels (CC-BY 4.0)",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"wrote manifest v2 -> {(OUT / 'manifest.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
