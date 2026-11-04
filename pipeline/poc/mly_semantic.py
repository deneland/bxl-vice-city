"""POC: validate and refine Mapillary 360° poses with semantic agreement.

Mapillary's own segmentation labels each pixel (building, vegetation, sky, road, ...). Cast a ray per
pixel from the camera into the voxel model and check the voxel class it hits:
  building pixel -> building voxel, sky -> nothing, road/sidewalk -> ground, vegetation -> tree.
Score = mean per-class agreement (balanced, so sky/road do not dominate).

Refinement: coordinate descent over position (east, north, height) and rotation (yaw, pitch, roll)
around Mapillary's pose, maximising the score. Writes data/cache/poc/mly_poses.json.

Usage: uv run python poc/mly_semantic.py [--refine] [--jobs=11]
"""
import base64
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mapbox_vector_tile
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from mly_align import RAW, rotvec_to_matrix, world_dirs  # noqa: E402
from pano_align import OUT, VS, X0, Y0, load_voxels, to_l72  # noqa: E402

SW, SH = 256, 128          # semantic image size used for scoring
LAT_MIN = -25.0
MAX_D, STEP = 40.0, 0.25
DET = RAW / "det"
# label id: 1 building, 2 vegetation, 3 sky, 4 ground; 9 = ignore (vehicles, people, ego vehicle)
LABELS = [  # drawn in order, later wins
    (4, ("construction--flat--", "construction--barrier--curb", "nature--terrain", "marking--", "void--ground")),
    (1, ("construction--structure--building", "construction--barrier--wall", "object--sign--store")),
    (2, ("nature--vegetation",)),
    (3, ("nature--sky",)),
    (9, ("object--vehicle--", "human--", "void--ego-vehicle", "void--dynamic")),
]
NAMES = {1: "building", 2: "vegetation", 3: "sky", 4: "ground"}
AREA = {}


def label_image(image_id):
    im = Image.new("L", (SW, SH), 0)
    draw = ImageDraw.Draw(im)
    dets = json.loads((DET / f"{image_id}.json").read_text()).get("data", [])
    for lab, prefixes in LABELS:
        for det in dets:
            if not det["value"].startswith(prefixes):
                continue
            tile = mapbox_vector_tile.decode(base64.b64decode(det["geometry"]), default_options={"y_coord_down": True})
            for layer in tile.values():
                ext = layer.get("extent", 4096)
                for feat in layer["features"]:
                    g = feat["geometry"]
                    for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
                        ring = [(x * SW / ext, y * SH / ext) for x, y in poly[0]]
                        if len(ring) >= 3:
                            draw.polygon(ring, fill=lab)
    lab = np.asarray(im).copy()
    rows = 90.0 - (np.arange(SH) + 0.5) / SH * 180.0
    lab[rows <= LAT_MIN] = 0
    return lab


def small_rot(dyaw, dpitch, droll):
    """World-frame rotation (east, north, up) for small heading/tilt corrections, degrees."""
    y, p, r = np.radians([dyaw, dpitch, droll])
    Rz = np.array([[np.cos(y), np.sin(y), 0], [-np.sin(y), np.cos(y), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])
    return Rz @ Rx @ Ry


def voxel_labels(cam, dirs):
    """Ray-cast each direction; return label of first voxel hit (3 = sky if none)."""
    occ, c0 = AREA["occ"], AREA["c0"]
    n0, n1 = occ.shape[:2]
    p0 = np.array([(cam[0] - X0) / VS - c0[0], (cam[1] - Y0) / VS - c0[1], cam[2]])
    out = np.full(len(dirs), 3, np.uint8)
    alive = np.arange(len(dirs))
    for t in np.arange(STEP, MAX_D, STEP):
        p = np.floor(p0 + dirs[alive] * (t / VS)).astype(int)
        ok = (p[:, 0] >= 0) & (p[:, 0] < n0) & (p[:, 1] >= 0) & (p[:, 1] < n1) & (p[:, 2] >= 0) & (p[:, 2] < 256)
        c = np.zeros(len(alive), np.uint8)
        c[ok] = occ[p[ok, 0], p[ok, 1], p[ok, 2]]
        h = c > 0
        if h.any():
            cl = c[h]
            out[alive[h]] = np.select([cl == 6, (cl == 3) | (cl == 5), cl == 2], [1, 2, 4], 0)
        alive = alive[~h & ok]
        if not len(alive):
            break
    return out


def score(lab, vox):
    per = {}
    for k in NAMES:
        m = lab == k
        if m.sum() > 30:
            per[k] = float((vox[m] == k).mean())
    weights = {1: 2.0, 3: 1.5, 4: 1.0, 2: 0.5}  # buildings/sky most reliable, trees change
    tot = sum(weights[k] for k in per)
    return (sum(per[k] * weights[k] for k in per) / tot if tot else 0.0), per


def solve(d):
    lab = label_image(d["id"]).ravel()
    sel = lab > 0
    sel &= lab != 9
    R = rotvec_to_matrix(d["computed_rotation"])
    D0 = world_dirs(R, SW, SH).reshape(-1, 3)[sel]
    labs = lab[sel]
    x, y = to_l72.transform(*d["computed_geometry"]["coordinates"])
    occ, c0 = AREA["occ"], AREA["c0"]
    gx, gy = int((x - X0) / VS) - c0[0], int((y - Y0) / VS) - c0[1]
    col = occ[gx - 1:gx + 2, gy - 1:gy + 2].reshape(-1, occ.shape[2])
    lows = [np.nonzero(c)[0].min() for c in col if c.any()]
    z_ground = (np.median(lows) + 1) if lows else 0.0   # in voxel units

    g = AREA.get('offset', (0.0, 0.0))
    if AREA.get('offset_grid'):
        # global-offset mode: score this image at each candidate shared offset
        return {'id': d['id'], 'grid': [score(labs, voxel_labels((x + ox, y + oy, z_ground + 2.0 / VS), D0))[0]
                                     for ox, oy in AREA['offset_grid']]}

    def evaluate(p):
        dx, dy, dz, dyaw, dpitch, droll = p
        dx, dy = dx + g[0], dy + g[1]
        dirs = D0 @ small_rot(dyaw, dpitch, droll).T
        cam = (x + dx, y + dy, z_ground + (2.0 + dz) / VS)
        return score(labs, voxel_labels(cam, dirs))

    p = np.zeros(6)
    s0, per0 = evaluate(p)
    best, per = s0, per0
    if AREA.get("refine"):
        steps = np.array([0.6, 0.6, 0.5, 2.0, 2.0, 2.0])
        limits = np.array([1.5, 1.5, 1.5, 8.0, 8.0, 6.0])  # residual around the global offset
        for _ in range(3):
            improved = True
            while improved:
                improved = False
                for i in range(6):
                    for sgn in (1, -1):
                        q = p.copy()
                        q[i] += sgn * steps[i]
                        if abs(q[i]) > limits[i]:
                            continue
                        s, pr = evaluate(q)
                        if s > best + 1e-4:
                            best, per, p, improved = s, pr, q, True
                            break
            steps /= 2
    return {"id": d["id"], "sequence": d["sequence"], "score_before": s0, "score_after": best,
            "per_before": per0, "per_after": per, "delta": p.tolist(), "offset": list(g)}


def init(area):
    AREA.update(area)


def main():
    t0 = time.time()
    data = json.loads((RAW / "street_360.json").read_text())
    data = [d for d in data if (DET / f"{d['id']}.json").exists()]
    refine = "--refine" in sys.argv
    jobs = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--jobs=")), 11))
    xy = np.array([to_l72.transform(*d["computed_geometry"]["coordinates"]) for d in data])
    cx, cy = xy.mean(0)
    occ, _, origin = load_voxels(cx, cy, np.abs(xy - [cx, cy]).max() + MAX_D + 5)
    c0 = (round((origin[0] - X0) / VS), round((origin[1] - Y0) / VS))
    offset = (0.0, 0.0)
    if "--global" in sys.argv:
        grid = [(ox, oy) for ox in np.arange(-4, 4.01, 0.5) for oy in np.arange(-4, 4.01, 0.5)]
        sample = data[::4]
        with ProcessPoolExecutor(jobs, initializer=init, initargs=({"occ": occ, "c0": c0, "offset_grid": grid},)) as ex:
            G = np.array([r["grid"] for r in ex.map(solve, sample, chunksize=2)])
        mean = G.mean(0)
        k = int(mean.argmax())
        offset = grid[k]
        zero = grid.index((0.0, 0.0))
        print(f"global offset over {len(sample)} images: east {offset[0]:+.1f} m, north {offset[1]:+.1f} m "
              f"(agreement {mean[zero]:.3f} -> {mean[k]:.3f}) in {time.time() - t0:.0f} s", flush=True)
    res = []
    with ProcessPoolExecutor(jobs, initializer=init, initargs=({"occ": occ, "c0": c0, "refine": refine, "offset": offset},)) as ex:
        for i, r in enumerate(ex.map(solve, data, chunksize=2), 1):
            res.append(r)
            if i % 20 == 0 or i == len(data):
                print(f"  {i}/{len(data)} images, {time.time() - t0:.0f} s", flush=True)
    (OUT / "mly_poses.json").write_text(json.dumps(res, indent=1))
    b = np.array([r["score_before"] for r in res])
    a = np.array([r["score_after"] for r in res])
    print(f"\nsemantic agreement over {len(res)} images: {b.mean():.3f} -> {a.mean():.3f} (median {np.median(b):.3f} -> {np.median(a):.3f})")
    for k, nm in NAMES.items():
        vb = [r["per_before"][str(k)] if str(k) in r["per_before"] else r["per_before"].get(k) for r in res]
        va = [r["per_after"][str(k)] if str(k) in r["per_after"] else r["per_after"].get(k) for r in res]
        vb = [v for v in vb if v is not None]
        va = [v for v in va if v is not None]
        print(f"  {nm:10s} {np.mean(vb):.2f} -> {np.mean(va):.2f}")
    if refine:
        D = np.array([r["delta"] for r in res])
        print("corrections (median |x|): east %.2f m, north %.2f m, height %.2f m, yaw %.1f°, pitch %.1f°, roll %.1f°" % tuple(np.median(np.abs(D), 0)))
        by = {}
        for r in res:
            by.setdefault(r["sequence"], []).append(r)
        for s, rs in by.items():
            dd = np.array([r["delta"] for r in rs])
            print(f"  session {s[:8]} n={len(rs):3d}  score {np.mean([r['score_before'] for r in rs]):.2f} -> {np.mean([r['score_after'] for r in rs]):.2f}  "
                  f"height {np.median(dd[:, 2]):+.2f} m (sd {dd[:, 2].std():.2f})  pitch {np.median(dd[:, 4]):+.1f}° (sd {dd[:, 4].std():.1f})")
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
