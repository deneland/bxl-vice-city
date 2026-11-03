"""LiDAR point cloud -> sparse coloured voxels -> binary chunks for the viewer.

Single-source build: only the Brussels 2021 LiDAR (CC-BY Paradigm/UrbIS).

Steps
  1. read points inside the ROI, bin them into VS-metre voxels
  2. per voxel: mean RGB, majority class, point count (drop 1-point voxels = noise)
  3. per column: ground height (class 2, holes filled by nearest neighbour)
     and building top (class 6)
  4. make buildings and terrain solid, but only emit the voxels that can be
     seen from the side (down to the lowest neighbouring column top)
  5. write 64x64-column chunks with a 1-voxel apron so the mesher can cull
     faces across chunk borders

Usage: uv run python voxelize.py
"""

import json
import time
import tomllib
from pathlib import Path

import laspy
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
CFG = tomllib.loads((ROOT / "config.toml").read_text())
ROI = CFG["roi"]
VS = CFG.get("voxel", {}).get("size", 0.5)
CH = 64            # chunk size in columns
NZ = 256           # max voxel layers (z stored as uint8)
OUT = ROOT / "viewer" / "public" / "tiles" / ROI["name"]

C_GROUND, C_LOWVEG, C_HIGHVEG, C_BUILDING, C_WATER = 2, 3, 5, 6, 9
FLAG_FILL = 1      # voxel synthesised (wall / terrain fill), not measured
FLAG_APRON = 2     # belongs to the neighbouring chunk, used only for culling

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
    ix, iy, iz, rgb, cls = ix[ok], iy[ok], iz[ok], rgb[ok], cls[ok]
    log(f"grid {NX}x{NY}x{NZ} @ {VS} m, z0 = {z0} m; dropped {(~ok).sum():,} out-of-range points")

    # --- 2. aggregate points into voxels -----------------------------------
    key = (iz * NY + iy) * NX + ix
    vkey, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    col = np.stack([np.bincount(inv, weights=rgb[:, i], minlength=len(vkey)) for i in range(3)], 1)
    col = (col / cnt[:, None]).round().astype(np.uint8)
    # majority class: sort (voxel, class-count) and keep the last entry per voxel
    k2, c2 = np.unique(inv * 32 + cls, return_counts=True)
    order = np.lexsort((c2, k2 // 32))
    k2 = k2[order]
    last = np.r_[k2[1:] // 32 != k2[:-1] // 32, True]
    vcls = np.empty(len(vkey), np.uint8)
    vcls[k2[last] // 32] = (k2[last] % 32).astype(np.uint8)

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
    # roof colour per column = colour of the top building voxel
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
    # walls: roof colour blended towards a warm brick/stone tone; terrain: ground colour
    wall = np.array([196, 170, 140], np.float32)
    fcol = np.where(fb[:, None], (0.45 * bcol[fy, fx] + 0.55 * wall),
                    gcol[fy, fx] * np.where(fz < ground[fy, fx], 0.8, 1.0)[:, None]).astype(np.uint8)
    fcls = np.where(fb, C_BUILDING, C_GROUND).astype(np.uint8)
    fkey = (fz * NY + fy) * NX + fx
    log(f"{len(fkey):,} fill voxels (visible shell only)")

    # measured voxels inside the solid (below lo) are hidden -> drop; below ground -> noise
    vis = (vz >= lo[vy, vx]) | (vz > top[vy, vx])
    vis &= vz >= ground[vy, vx] - 1
    vkey, col, vcls = vkey[vis], col[vis], vcls[vis]
    # merge: measured voxels win over fill
    new = ~np.isin(fkey, vkey, assume_unique=True)
    akey = np.concatenate([vkey, fkey[new]])
    acol = np.concatenate([col, fcol[new]])
    acls = np.concatenate([vcls, fcls[new]])
    aflag = np.concatenate([np.zeros(len(vkey), np.uint8), np.full(new.sum(), FLAG_FILL, np.uint8)])
    o = np.argsort(akey)
    akey, acol, acls, aflag = akey[o], acol[o], acls[o], aflag[o]
    ax, ay, az = akey % NX, (akey // NX) % NY, akey // (NX * NY)
    log(f"{len(akey):,} voxels total")

    # visible face estimate
    faces = 0
    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        nx_, ny_, nz_ = ax + dx, ay + dy, az + dz
        inside = (nx_ >= 0) & (nx_ < NX) & (ny_ >= 0) & (ny_ < NY) & (nz_ >= 0)
        nk = (nz_ * NY + ny_) * NX + nx_
        pos = np.searchsorted(akey, nk).clip(0, len(akey) - 1)
        occ = inside & (akey[pos] == nk)
        if dz == -1:  # down faces of solid voxels are never seen
            occ |= aflag == FLAG_FILL
        faces += int((~occ).sum())
    log(f"~{faces:,} visible faces")

    # --- 5. write chunks ----------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.bin"):
        old.unlink()
    ncx, ncy = -(-NX // CH), -(-NY // CH)
    chunks = []
    total_bytes = 0
    for cy in range(ncy):
        ym = (ay >= cy * CH - 1) & (ay <= cy * CH + CH)
        for cx in range(ncx):
            m = ym & (ax >= cx * CH - 1) & (ax <= cx * CH + CH)
            if not m.any():
                continue
            lx, ly = ax[m] - cx * CH, ay[m] - cy * CH
            apron = (lx < 0) | (lx >= CH) | (ly < 0) | (ly >= CH)
            buf = np.empty((m.sum(), 8), np.uint8)
            buf[:, 0], buf[:, 1], buf[:, 2] = lx + 1, ly + 1, az[m]
            buf[:, 3:6] = acol[m]
            buf[:, 6] = acls[m]
            buf[:, 7] = aflag[m] | np.where(apron, FLAG_APRON, 0)
            name = f"c_{cx}_{cy}.bin"
            (OUT / name).write_bytes(buf.tobytes())
            total_bytes += buf.nbytes
            zr = az[m][~apron]
            chunks.append({"file": name, "cx": cx, "cy": cy, "count": int((~apron).sum()),
                           "zmin": int(zr.min()) if len(zr) else 0,
                           "zmax": int(zr.max()) if len(zr) else 0})
    manifest = {
        "name": ROI["name"], "crs": "EPSG:31370",
        "origin": [ROI["xmin"], ROI["ymin"], z0], "voxelSize": VS,
        "size": [NX, NY, NZ], "chunkSize": CH,
        "home": [CFG.get("view", {}).get("x", (ROI["xmin"] + ROI["xmax"]) / 2),
                 CFG.get("view", {}).get("y", (ROI["ymin"] + ROI["ymax"]) / 2)],
        "chunks": chunks,
        "attribution": "LiDAR © Paradigm / UrbIS – datastore.brussels (CC-BY 4.0)",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"wrote {len(chunks)} chunks, {total_bytes / 1e6:.1f} MB -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
