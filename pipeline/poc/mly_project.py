"""POC: colour voxels from Mapillary 360° images (SfM poses) and patch the live L0 tiles.

Usage: uv run python poc/mly_project.py [--apply] [--mask] [--vote-by-session]
  --mask             drop pixels Mapillary detected as vehicles, people, riders, the camera
                     vehicle or sky (its own segmentation, fetched per image and cached)
  --semantic         a building voxel only accepts pixels Mapillary labels building/wall/store
                     sign, and none inside vegetation, sky, vehicle, person, pole, sign or fence
  --vote-by-session  median per capture session first, then median across sessions,
                     so one long session cannot outvote the others
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import base64

import mapbox_vector_tile
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from mly_align import RAW, get, rotvec_to_matrix, world_dirs  # noqa: E402
from mly_semantic import small_rot  # noqa: E402
from pano_align import OUT, VS, X0, Y0, Z0, ground_z, load_voxels, to_l72  # noqa: E402
import pano_project as pp  # noqa: E402

PW, PH = 768, 384
LAT_MIN = -25.0  # camera-frame elevation below which the vehicle fills the image
AREA = {}
EXCLUDE = ("object--vehicle--", "human--", "void--ego-vehicle", "nature--sky")
DET = RAW / "det"


def fetch_detections(d):
    f = DET / f"{d['id']}.json"
    if not f.exists():
        f.write_bytes(get(f"https://graph.mapillary.com/{d['id']}/detections?fields=value,geometry"))
    return f


FACADE = ("construction--structure--building", "construction--barrier--wall", "object--sign--store",
          "object--sign--advertisement")
BLOCK = EXCLUDE + ("nature--vegetation", "object--support--", "object--traffic-sign--", "object--traffic-light",
                   "construction--barrier--fence", "void--dynamic")


def exclusion_mask(image_id, prefixes=EXCLUDE):
    """Pixels (PH, PW) covered by detections whose label starts with one of prefixes."""
    m = Image.new("L", (PW, PH), 0)
    draw = ImageDraw.Draw(m)
    f = DET / f"{image_id}.json"
    if not f.exists():
        return np.zeros((PH, PW), bool)
    for det in json.loads(f.read_text()).get("data", []):
        if not det["value"].startswith(prefixes):
            continue
        tile = mapbox_vector_tile.decode(base64.b64decode(det["geometry"]), default_options={"y_coord_down": True})
        for layer in tile.values():
            ext = layer.get("extent", 4096)
            for feat in layer["features"]:
                g = feat["geometry"]
                polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
                for poly in polys:
                    ring = [(x * PW / ext, y * PH / ext) for x, y in poly[0]]
                    if len(ring) >= 3:
                        draw.polygon(ring, fill=255)
    return np.asarray(m) > 0


def init(area):
    AREA.update(area)


def project(d):
    occ, c0 = AREA["occ"], AREA["c0"]
    n0, n1 = occ.shape[:2]
    im = np.asarray(Image.open(RAW / f"{d['id']}.jpg").convert("RGB").resize((PW, PH), Image.BILINEAR))
    rows = 90.0 - (np.arange(PH) + 0.5) / PH * 180.0
    keep = np.repeat((rows > LAT_MIN)[:, None], PW, 1)
    if AREA.get('semantic'):
        keep &= exclusion_mask(d['id'], FACADE) & ~exclusion_mask(d['id'], BLOCK)
    elif AREA.get('mask'):
        keep &= ~exclusion_mask(d['id'])
    dw = world_dirs(rotvec_to_matrix(d["computed_rotation"]), PW, PH)[keep]
    pix = im[keep]
    x, y = to_l72.transform(*d["computed_geometry"]["coordinates"])
    cal = AREA.get("poses", {}).get(d["id"])   # calibrated pose from mly_semantic.py (--calibrated)
    dz = 0.0
    if cal:
        ddx, ddy, dz, dyaw, dpitch, droll = cal["delta"]
        dw = dw @ small_rot(dyaw, dpitch, droll).T
        x, y = x + cal["offset"][0] + ddx, y + cal["offset"][1] + ddy
    gx, gy = (x - X0) / VS - c0[0], (y - Y0) / VS - c0[1]
    # camera height: 2 m above the ground under it (car roof / backpack mounts)
    gzc = int(gx), int(gy)
    col = occ[gzc[0] - 1:gzc[0] + 2, gzc[1] - 1:gzc[1] + 2].reshape(-1, occ.shape[2])
    lows = [np.nonzero(c)[0].min() for c in col if c.any()]
    zcam = (np.median(lows) + 1) + (2.0 + dz) / VS if lows else 0
    p0 = np.array([gx, gy, zcam])
    alive = np.arange(len(dw))
    prev = np.floor(np.repeat(p0[None], len(dw), 0)).astype(int)
    keys, cols = [], []
    for t in np.arange(pp.STEP, pp.MAX_D, pp.STEP):
        p = np.floor(p0 + dw[alive] * (t / VS)).astype(int)
        ok = (p[:, 0] >= 0) & (p[:, 0] < n0) & (p[:, 1] >= 0) & (p[:, 1] < n1) & (p[:, 2] >= 0) & (p[:, 2] < 256)
        h = np.zeros(len(alive), bool)
        h[ok] = occ[p[ok, 0], p[ok, 1], p[ok, 2]] > 0
        if h.any():
            idx = alive[h]
            ph = p[h]
            axis = np.argmax(ph != prev[idx], axis=1)
            good = np.abs(dw[idx, axis]) > pp.MIN_COS
            cls = occ[ph[:, 0], ph[:, 1], ph[:, 2]]
            # only walls/shopfronts: a side face (crossed along x or y), never roofs, ground or trees
            good &= (axis < 2) & (cls != 3) & (cls != 5)
            if AREA.get('semantic'):
                good &= cls == 6  # building voxels only
            g = ph[good]
            keys.append((g[:, 2].astype(np.int64) * pp.NY + (g[:, 1] + c0[1])) * pp.NX + (g[:, 0] + c0[0]))
            cols.append(pix[idx[good]])
        prev[alive] = p
        alive = alive[~h & ok]
        if not len(alive):
            break
    return (np.concatenate(keys) if keys else np.zeros(0, np.int64),
            np.concatenate(cols) if cols else np.zeros((0, 3), np.uint8))


def main():
    t0 = time.time()
    data = json.loads((RAW / "street_360.json").read_text())
    todo = [d for d in data if not (RAW / f"{d['id']}.jpg").exists()]
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(lambda d: (RAW / f"{d['id']}.jpg").write_bytes(get(d["thumb_2048_url"])), todo))
    print(f"{len(data)} panoramas ({len(todo)} downloaded) in {time.time() - t0:.0f} s", flush=True)
    xy = np.array([to_l72.transform(*d["computed_geometry"]["coordinates"]) for d in data])
    cx, cy = xy.mean(0)
    radius = np.abs(xy - [cx, cy]).max() + pp.MAX_D + 5
    occ, _, origin = load_voxels(cx, cy, radius)
    c0 = (round((origin[0] - X0) / VS), round((origin[1] - Y0) / VS))
    semantic = "--semantic" in sys.argv
    mask = "--mask" in sys.argv or semantic
    if mask:
        DET.mkdir(exist_ok=True)
        with ThreadPoolExecutor(12) as ex:
            list(ex.map(fetch_detections, data))
        print(f"detections cached for {len(data)} images", flush=True)
    with ProcessPoolExecutor(11, initializer=init, initargs=({"occ": occ, "c0": c0, "mask": mask, "semantic": semantic,
                                                        "poses": {r["id"]: r for r in json.loads((OUT / "mly_poses.json").read_text())} if "--calibrated" in sys.argv else {}},)) as ex:
        parts = list(ex.map(project, data, chunksize=4))
    keys = np.concatenate([k for k, _ in parts])
    cols = np.concatenate([c for _, c in parts])
    if "--vote-by-session" in sys.argv:
        seqs = sorted({d["sequence"] for d in data})
        sid = np.concatenate([np.full(len(k), seqs.index(d["sequence"]), np.int64) for (k, _), d in zip(parts, data)])
        k2, m2, c2 = pp.median_per_key(keys * 64 + sid, cols)      # one vote per (voxel, session)
        good = c2 >= 2                                               # a session needs 2+ samples to vote
        ukey, med, cnt = pp.median_per_key(k2[good] // 64, m2[good])  # cnt = number of sessions
        sel = (cnt >= 2) | np.isin(ukey, k2[good & (c2 >= 4)] // 64)
        ukey, med, cnt = ukey[sel], med[sel], cnt[sel]
        np.savez(OUT / "mly_colours.npz", key=ukey, rgb=med, n=cnt)
        print(f"{len(keys):,} samples -> {len(ukey):,} voxels coloured by session vote "
              f"(median {np.median(cnt):.0f} sessions each) in {time.time() - t0:.0f} s", flush=True)
        if "--apply" in sys.argv:
            pp.apply(ukey, med)
        return
    ukey, med, cnt = pp.median_per_key(keys, cols)
    sel = cnt >= 3
    ukey, med, cnt = ukey[sel], med[sel], cnt[sel]
    np.savez(OUT / "mly_colours.npz", key=ukey, rgb=med, n=cnt)
    print(f"{len(keys):,} samples -> {len(ukey):,} voxels coloured (median {np.median(cnt):.0f} samples each) "
          f"in {time.time() - t0:.0f} s", flush=True)
    if "--apply" in sys.argv:
        pp.apply(ukey, med)


if __name__ == "__main__":
    main()
