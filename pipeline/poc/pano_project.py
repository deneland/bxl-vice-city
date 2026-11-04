"""POC: colour voxels from pose-corrected Panoramax 360° images and patch the live L0 tiles.

For every pixel of every panorama (above the vehicle), cast a ray from the corrected camera pose,
take the first voxel hit within 25 m (skipping grazing angles), and record the pixel colour.
Each voxel gets the per-channel median of its samples (cars/people passing differ between images).

Usage: uv run python poc/pano_project.py [--apply]
  without --apply: statistics + data/cache/poc/pano_colours.npz only
  --apply: also rewrite colours in viewer/public/tiles/schaerbeek/L0 (a rebuild undoes it)
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from pano_align import OUT, PANO, TILES, VS, X0, Y0, Z0, CH, man, load_voxels  # noqa: E402
from pano_pose import dirs_world, rot  # noqa: E402

PW, PH = 1024, 512       # sampling resolution of each panorama
EL_MIN = -25.0           # below this the vehicle / rider fills the image
MAX_D, STEP = 25.0, 0.25
MIN_COS = 0.26           # skip rays hitting a face at more than ~75° from its normal
NX = man["levels"][0]["size"][0]
NY = man["levels"][0]["size"][1]

AREA = {}


def init(area):
    AREA.update(area)


def project(job):
    meta, pose = job
    occ, c0 = AREA["occ"], AREA["c0"]
    n0, n1 = occ.shape[0], occ.shape[1]
    im = np.asarray(Image.open(PANO / "sd" / f"{meta['id']}.jpg").convert("RGB").resize((PW, PH), Image.BILINEAR))
    Dc = dirs_world(PW, PH)
    el = np.degrees(np.arcsin(Dc[..., 2]))
    keep = el > EL_MIN
    d = Dc[keep] @ rot(pose["yaw"], pose["pitch"], pose["roll"]).T
    pix = im[keep]
    p0 = np.array([(pose["pos"][0] - X0) / VS - c0[0], (pose["pos"][1] - Y0) / VS - c0[1], (pose["pos"][2] - Z0) / VS])
    alive = np.arange(len(d))
    prev = np.floor(p0 + 0 * d).astype(int)
    keys, cols = [], []
    for t in np.arange(STEP, MAX_D, STEP):
        pf = p0 + d[alive] * (t / VS)
        p = np.floor(pf).astype(int)
        ok = (p[:, 0] >= 0) & (p[:, 0] < n0) & (p[:, 1] >= 0) & (p[:, 1] < n1) & (p[:, 2] >= 0) & (p[:, 2] < 256)
        h = np.zeros(len(alive), bool)
        h[ok] = occ[p[ok, 0], p[ok, 1], p[ok, 2]] > 0
        if h.any():
            idx = alive[h]
            ph, pv = p[h], prev[idx]
            # which face was crossed: the axis that changed (first one if several)
            axis = np.argmax(ph != pv, axis=1)
            cosang = np.abs(d[idx, axis])
            good = cosang > MIN_COS
            g = ph[good]
            keys.append((g[:, 2].astype(np.int64) * NY + (g[:, 1] + c0[1])) * NX + (g[:, 0] + c0[0]))
            cols.append(pix[idx[good]])
        prev[alive] = p
        alive = alive[~h & ok]
        if not len(alive):
            break
    k = np.concatenate(keys) if keys else np.zeros(0, np.int64)
    c = np.concatenate(cols) if cols else np.zeros((0, 3), np.uint8)
    return k, c


def median_per_key(keys, cols):
    o = np.argsort(keys, kind="stable")
    keys, cols = keys[o], cols[o]
    ukey, start, cnt = np.unique(keys, return_index=True, return_counts=True)
    med = np.empty((len(ukey), 3), np.uint8)
    for ch in range(3):
        # sort values inside each group, take the middle element
        o2 = np.lexsort((cols[:, ch], keys))
        med[:, ch] = cols[o2, ch][start + cnt // 2]
    return ukey, med, cnt


def main():
    t0 = time.time()
    poses = {p["id"]: p for p in json.loads((OUT / "poses.json").read_text())}
    metas = [m for m in json.loads((PANO / "images.json").read_text()) if m["id"] in poses]
    xs = np.array([poses[m["id"]]["pos"][0] for m in metas])
    ys = np.array([poses[m["id"]]["pos"][1] for m in metas])
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    radius = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + MAX_D + 5
    occ, _, origin = load_voxels(cx, cy, radius)
    c0 = (round((origin[0] - X0) / VS), round((origin[1] - Y0) / VS))
    print(f"{len(metas)} panoramas, area {2 * radius:.0f} m around {cx:.0f},{cy:.0f}, grid {occ.shape}")
    with ProcessPoolExecutor(11, initializer=init, initargs=({"occ": occ, "c0": c0},)) as ex:
        parts = list(ex.map(project, [(m, poses[m["id"]]) for m in metas]))
    keys = np.concatenate([k for k, _ in parts])
    cols = np.concatenate([c for _, c in parts])
    ukey, med, cnt = median_per_key(keys, cols)
    sel = cnt >= 3
    ukey, med, cnt = ukey[sel], med[sel], cnt[sel]
    np.savez(OUT / "pano_colours.npz", key=ukey, rgb=med, n=cnt)
    print(f"{len(keys):,} samples -> {len(ukey):,} voxels with >=3 samples (median {np.median(cnt):.0f} each) "
          f"in {time.time() - t0:.0f} s")
    if "--apply" in sys.argv:
        apply(ukey, med)


def apply(ukey, med):
    changed = tiles = 0
    for t in man["levels"][0]["tiles"]:
        f = TILES / t["file"]
        v = np.frombuffer(f.read_bytes(), np.uint8).reshape(-1, 8).copy()
        gx = v[:, 0].astype(np.int64) - 1 + t["tx"] * CH
        gy = v[:, 1].astype(np.int64) - 1 + t["ty"] * CH
        k = (v[:, 2].astype(np.int64) * NY + gy) * NX + gx
        pos = np.searchsorted(ukey, k).clip(0, len(ukey) - 1)
        m = ukey[pos] == k
        if m.any():
            v[m, 3:6] = med[pos[m]]
            f.write_bytes(v.tobytes())
            changed += int(m.sum())
            tiles += 1
    mf = TILES / "manifest.json"
    mj = json.loads(mf.read_text())
    mj["build"] = mj["build"] + "-pano"  # cache-bust the viewer
    mf.write_text(json.dumps(mj, indent=1))
    print(f"patched {changed:,} voxels in {tiles} L0 tiles (build {mj['build']})")


if __name__ == "__main__":
    main()
