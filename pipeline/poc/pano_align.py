"""POC: render the voxel model as an equirectangular panorama from a Panoramax image's pose
and overlay it on the photo, to judge whether camera poses are good enough for projection.

Usage: uv run python poc/pano_align.py <image-id-prefix> [dx dy dz dyaw]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
TILES = ROOT / "viewer/public/tiles/schaerbeek"
PANO = ROOT / "data/raw/panoramax"
OUT = ROOT / "data/cache/poc"
OUT.mkdir(parents=True, exist_ok=True)
man = json.loads((TILES / "manifest.json").read_text())
X0, Y0, Z0 = man["origin"]
VS, CH = 0.5, man["tileSize"]
to_l72 = Transformer.from_crs(4326, 31370, always_xy=True)


def load_voxels(cx, cy, radius):
    """Dense occupancy + colour grid of L0 voxels within radius (m) of (cx, cy) in L72."""
    c0 = int((cx - radius - X0) / VS), int((cy - radius - Y0) / VS)
    n = int(2 * radius / VS)
    occ = np.zeros((n, n, 256), np.uint8)       # x, y, z ; 0 empty, else class
    col = np.zeros((n, n, 256, 3), np.uint8)
    for t in man["levels"][0]["tiles"]:
        if not (c0[0] - CH <= t["tx"] * CH <= c0[0] + n and c0[1] - CH <= t["ty"] * CH <= c0[1] + n):
            continue
        v = np.frombuffer((TILES / t["file"]).read_bytes(), np.uint8).reshape(-1, 8)
        v = v[(v[:, 7] & 2) == 0]
        gx = v[:, 0].astype(int) - 1 + t["tx"] * CH - c0[0]
        gy = v[:, 1].astype(int) - 1 + t["ty"] * CH - c0[1]
        m = (gx >= 0) & (gx < n) & (gy >= 0) & (gy < n)
        occ[gx[m], gy[m], v[m, 2]] = np.maximum(v[m, 6], 1)
        col[gx[m], gy[m], v[m, 2]] = v[m, 3:6]
    return occ, col, (X0 + c0[0] * VS, Y0 + c0[1] * VS)


def render(occ, col, origin, cam, yaw_deg, W=1024, H=512, max_d=70.0, step=0.2):
    """Ray-march an equirectangular panorama. Column 0.5*W looks along yaw (0 = north, 90 = east)."""
    u = (np.arange(W) + 0.5) / W
    v = (np.arange(H) + 0.5) / H
    az = np.radians(yaw_deg + (u - 0.5) * 360.0)
    el = np.radians(90.0 - v * 180.0)
    AZ, EL = np.meshgrid(az, el)
    d = np.stack([np.sin(AZ) * np.cos(EL), np.cos(AZ) * np.cos(EL), np.sin(EL)], -1).reshape(-1, 3)
    p0 = np.array([(cam[0] - origin[0]) / VS, (cam[1] - origin[1]) / VS, (cam[2] - Z0) / VS])
    hit = np.full(len(d), -1.0)
    rgb = np.zeros((len(d), 3), np.uint8)
    alive = np.arange(len(d))
    n = occ.shape[0]
    for t in np.arange(step, max_d, step):
        p = (p0 + d[alive] * (t / VS)).astype(int)
        ok = (p[:, 0] >= 0) & (p[:, 0] < n) & (p[:, 1] >= 0) & (p[:, 1] < n) & (p[:, 2] >= 0) & (p[:, 2] < 256)
        h = np.zeros(len(alive), bool)
        h[ok] = occ[p[ok, 0], p[ok, 1], p[ok, 2]] > 0
        if h.any():
            idx = alive[h]
            hit[idx] = t
            rgb[idx] = col[p[h, 0], p[h, 1], p[h, 2]]
        alive = alive[~h & ok]
        if not len(alive):
            break
    return hit.reshape(H, W), rgb.reshape(H, W, 3)


def ground_z(occ, origin, cam_xy):
    gx, gy = int((cam_xy[0] - origin[0]) / VS), int((cam_xy[1] - origin[1]) / VS)
    col = occ[gx - 2:gx + 3, gy - 2:gy + 3]
    zs = [np.nonzero(c)[0] for c in col.reshape(-1, 256)]
    lows = [z.min() for z in zs if len(z)]
    return Z0 + (np.median(lows) + 1) * VS  # top of the lowest (ground) voxel


def main():
    pid = sys.argv[1]
    dx, dy, dz, dyaw = (list(map(float, sys.argv[2:6])) + [0, 0, 0, 0])[:4]
    meta = next(m for m in json.loads((PANO / "images.json").read_text()) if m["id"].startswith(pid))
    x, y = to_l72.transform(meta["lon"], meta["lat"])
    occ, col, origin = load_voxels(x, y, 80)
    gz = ground_z(occ, origin, (x, y))
    cam = (x + dx, y + dy, gz + 2.0 + dz)
    yaw = meta["heading"] + dyaw
    depth, syn = render(occ, col, origin, cam, yaw)
    photo = Image.open(PANO / "sd" / f"{meta['id']}.jpg").convert("RGB").resize((1024, 512))
    # overlay: photo with the synthetic depth edges (silhouettes, skyline) in magenta
    dz_ = np.where(depth < 0, 200.0, depth)
    edge = (np.abs(np.diff(np.log(dz_), axis=0, prepend=np.log(dz_[:1]))) > 0.25) | \
           (np.abs(np.diff(np.log(dz_), axis=1, prepend=np.log(dz_[:, :1]))) > 0.25)
    ov = np.asarray(photo).copy()
    ov[edge] = [255, 0, 255]
    stack = np.concatenate([ov, np.where(depth[..., None] < 0, 220, syn)], 0)
    out = OUT / f"align_{meta['id'][:8]}.png"
    Image.fromarray(stack.astype(np.uint8)).save(out)
    print(f"{meta['id'][:8]} {meta['date'][:10]} L72 {x:.1f},{y:.1f} ground {gz:.1f} m heading {yaw:.0f}° -> {out}")


if __name__ == "__main__":
    main()
