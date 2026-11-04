"""POC: correct Panoramax 360° camera poses against the voxel model.

For each panorama: search position (x, y around GPS) and rotation (yaw/pitch/roll around the
recorded heading) so that our model's silhouette edges fall on strong edges in the photo and
clear blue sky in the photo is sky in the model. Equirectangular images: no lens model needed.

Usage: uv run python poc/pano_pose.py [id-prefix ...]   (default: all images)
Writes data/cache/poc/poses.json and before/after overlays.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from pano_align import OUT, PANO, VS, Z0, ground_z, load_voxels, to_l72  # noqa: E402

W, H = 512, 256          # matching resolution (equirectangular)
EL_MIN = -20.0           # ignore the bottom of the panorama (vehicle, rider)


def dirs_world(W, H):
    u = (np.arange(W) + 0.5) / W
    v = (np.arange(H) + 0.5) / H
    az = np.radians((u - 0.5) * 360.0)
    el = np.radians(90.0 - v * 180.0)
    AZ, EL = np.meshgrid(az, el)
    return np.stack([np.sin(AZ) * np.cos(EL), np.cos(AZ) * np.cos(EL), np.sin(EL)], -1)


def render_depth(occ, origin, cam, D, max_d=60.0, step=0.25):
    """Depth (m) along world directions D (H,W,3); -1 = nothing within max_d."""
    d = D.reshape(-1, 3)
    p0 = np.array([(cam[0] - origin[0]) / VS, (cam[1] - origin[1]) / VS, (cam[2] - Z0) / VS])
    hit = np.full(len(d), -1.0)
    alive = np.arange(len(d))
    n = occ.shape[0]
    for t in np.arange(step, max_d, step):
        p = (p0 + d[alive] * (t / VS)).astype(int)
        ok = (p[:, 0] >= 0) & (p[:, 0] < n) & (p[:, 1] >= 0) & (p[:, 1] < n) & (p[:, 2] >= 0) & (p[:, 2] < 256)
        h = np.zeros(len(alive), bool)
        h[ok] = occ[p[ok, 0], p[ok, 1], p[ok, 2]] > 0
        hit[alive[h]] = t
        alive = alive[~h & ok]
        if not len(alive):
            break
    return hit.reshape(D.shape[:2])


def silhouette_edges(depth):
    dz = np.log(np.where(depth < 0, 200.0, depth))
    e = (np.abs(np.diff(dz, axis=0, prepend=dz[:1])) > 0.3) | (np.abs(np.diff(dz, axis=1, prepend=dz[:, :1])) > 0.3)
    return e.astype(np.float32)


def rot(yaw, pitch, roll):
    """Camera -> world. Camera frame: +y forward, +x right, +z up. Degrees."""
    y, p, r = np.radians([yaw, pitch, roll])
    Rz = np.array([[np.cos(y), np.sin(y), 0], [-np.sin(y), np.cos(y), 0], [0, 0, 1]])  # clockwise from north
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])
    return Rz @ Rx @ Ry


class Photo:
    def __init__(self, path):
        im = np.asarray(Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)).astype(np.float32)
        g = im.mean(2)
        grad = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))
        grad = ndimage.gaussian_filter(grad, 1.5)
        self.grad = grad / (np.percentile(grad, 99) + 1e-6)
        r, gg, b = im[..., 0], im[..., 1], im[..., 2]
        mx = im.max(2)
        self.blue_sky = (b > r + 18) & (b > gg + 4) & (mx > 120)
        D = dirs_world(W, H)
        el = np.degrees(np.arcsin(D[..., 2]))
        self.mask = el > EL_MIN
        self.dirs = D  # camera-frame directions of each photo pixel


def score(photo, edges_w, sky_w, R):
    """Sample world maps at the photo pixels' world directions under rotation R."""
    d = photo.dirs[photo.mask] @ R.T
    az = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
    el = np.degrees(np.arcsin(np.clip(d[:, 2], -1, 1)))
    u = ((az / 360.0 + 0.5) * W).astype(int) % W
    v = np.clip(((90.0 - el) / 180.0 * H).astype(int), 0, H - 1)
    e = edges_w[v, u]
    s_edge = (photo.grad[photo.mask] * e).sum() / max(e.sum(), 1)
    bs = photo.blue_sky[photo.mask]
    s_sky = sky_w[v, u][bs].mean() if bs.sum() > 50 else 0.5
    return s_edge + 0.5 * s_sky


def solve(meta):
    t0 = time.time()
    x, y = to_l72.transform(meta["lon"], meta["lat"])
    occ, _, origin = load_voxels(x, y, 70)
    gz = ground_z(occ, origin, (x, y))
    photo = Photo(PANO / "sd" / f"{meta['id']}.jpg")
    D = dirs_world(W, H)
    edges_cache = {}

    def world_maps(dx, dy, dz):
        k = (round(dx, 2), round(dy, 2), round(dz, 2))
        if k not in edges_cache:
            depth = render_depth(occ, origin, (x + dx, y + dy, gz + 2.0 + dz), D)
            e = ndimage.maximum_filter(silhouette_edges(depth), size=2)
            edges_cache[k] = (e, (depth < 0).astype(np.float32))
        return edges_cache[k]

    def best_rotation(maps, yaw0, around=(0, 0, 0), span=(8, 10, 6), step=(2, 2, 2)):
        e, s = maps
        best = (-1, None)
        for dyaw in np.arange(-span[0], span[0] + 1e-6, step[0]):
            for p in np.arange(-span[1], span[1] + 1e-6, step[1]):
                for r in np.arange(-span[2], span[2] + 1e-6, step[2]):
                    rr = (yaw0 + around[0] + dyaw, around[1] + p, around[2] + r)
                    sc = score(photo, e, s, rot(*rr))
                    if sc > best[0]:
                        best = (sc, rr)
        return best

    yaw0 = meta["heading"]
    before = score(photo, *world_maps(0, 0, 0), rot(yaw0, 0, 0))
    # 1) rotation at GPS position, 2) position grid with that rotation, 3) refine both
    sc, rr = best_rotation(world_maps(0, 0, 0), yaw0)
    best = (sc, (0.0, 0.0, 0.0), rr)
    for step_m in (1.0, 0.5):
        cx, cy, cz = best[1]
        for dx in (-2 * step_m, -step_m, 0, step_m, 2 * step_m):
            for dy in (-2 * step_m, -step_m, 0, step_m, 2 * step_m):
                pos = (cx + dx, cy + dy, cz)
                sc = score(photo, *world_maps(*pos), rot(*best[2]))
                if sc > best[0]:
                    best = (sc, pos, best[2])
        sc, rr = best_rotation(world_maps(*best[1]), 0, around=best[2], span=(3, 3, 3), step=(1, 1, 1))
        if sc > best[0]:
            best = (sc, best[1], rr)
    for dz in (-0.5, 0.5):
        pos = (best[1][0], best[1][1], best[1][2] + dz)
        sc = score(photo, *world_maps(*pos), rot(*best[2]))
        if sc > best[0]:
            best = (sc, pos, best[2])
    sc, pos, (yaw, pitch, roll) = best
    res = {"id": meta["id"], "date": meta["date"], "gps": [x, y], "ground": gz,
           "pos": [x + pos[0], y + pos[1], gz + 2.0 + pos[2]], "shift": list(pos),
           "yaw": yaw, "pitch": pitch, "roll": roll, "dyaw": yaw - yaw0,
           "score_before": before, "score_after": sc, "renders": len(edges_cache), "secs": time.time() - t0}
    overlay(meta, photo, world_maps(0, 0, 0)[0], rot(yaw0, 0, 0), world_maps(*pos)[0], rot(yaw, pitch, roll))
    return res


def overlay(meta, photo, e0, R0, e1, R1):
    im = np.asarray(Image.open(PANO / "sd" / f"{meta['id']}.jpg").convert("RGB").resize((W * 2, H * 2)))
    rows = []
    for e, R, colr in ((e0, R0, (255, 0, 255)), (e1, R1, (0, 255, 0))):
        Dp = dirs_world(W * 2, H * 2).reshape(-1, 3) @ R.T
        az = np.degrees(np.arctan2(Dp[:, 0], Dp[:, 1]))
        el = np.degrees(np.arcsin(np.clip(Dp[:, 2], -1, 1)))
        u = ((az / 360.0 + 0.5) * W).astype(int) % W
        v = np.clip(((90.0 - el) / 180.0 * H).astype(int), 0, H - 1)
        m = e[v, u].reshape(H * 2, W * 2) > 0
        o = im.copy()
        o[m] = colr
        rows.append(o)
    Image.fromarray(np.concatenate(rows, 0)).save(OUT / f"pose_{meta['id'][:8]}.png")


def main():
    metas = json.loads((PANO / "images.json").read_text())
    pref = [a for a in sys.argv[1:] if not a.startswith("-")]
    if pref:
        metas = [m for m in metas if any(m["id"].startswith(p) for p in pref)]
    jobs = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--jobs=")), 10))
    with ProcessPoolExecutor(jobs) as ex:
        results = list(ex.map(solve, metas))
    f = OUT / "poses.json"
    old = {r["id"]: r for r in json.loads(f.read_text())} if f.exists() else {}
    old.update({r["id"]: r for r in results})
    f.write_text(json.dumps(list(old.values()), indent=1, default=float))
    for r in results:
        print(f"{r['id'][:8]} {r['date'][:10]} score {r['score_before']:.3f} -> {r['score_after']:.3f}  "
              f"shift {r['shift'][0]:+.1f},{r['shift'][1]:+.1f},{r['shift'][2]:+.1f} m  "
              f"yaw {r['dyaw']:+.0f}° pitch {r['pitch']:+.0f}° roll {r['roll']:+.0f}°  ({r['renders']} renders, {r['secs']:.0f} s)")


if __name__ == "__main__":
    main()
