"""CPU preview of the voxel chunks (no browser needed): top-down and oblique views.

Usage: uv run python preview.py [radius_m]   -> data/cache/preview_*.png
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
man = json.loads((ROOT / "viewer/public/tiles/home/manifest.json").read_text())
VS, CH = man["voxelSize"], man["chunkSize"]
OUT = ROOT / "data/cache"
OUT.mkdir(parents=True, exist_ok=True)

parts = []
for c in man["chunks"]:
    v = np.frombuffer((ROOT / "viewer/public/tiles/home" / c["file"]).read_bytes(), np.uint8).reshape(-1, 8)
    v = v[(v[:, 7] & 2) == 0]
    g = np.empty((len(v), 7), np.int32)
    g[:, 0] = v[:, 0].astype(np.int32) - 1 + c["cx"] * CH
    g[:, 1] = v[:, 1].astype(np.int32) - 1 + c["cy"] * CH
    g[:, 2] = v[:, 2]
    g[:, 3:6] = v[:, 3:6]
    g[:, 6] = v[:, 6]
    parts.append(g)
V = np.concatenate(parts)
x, y, z, rgb = V[:, 0], V[:, 1], V[:, 2], V[:, 3:6].astype(np.float32)
NX, NY = man["size"][0], man["size"][1]
hx = int((man["home"][0] - man["origin"][0]) / VS)
hy = int((man["home"][1] - man["origin"][1]) / VS)

# top-down: highest voxel per column
o = np.argsort(z)
img = np.full((NY, NX, 3), 255, np.uint8)
img[NY - 1 - y[o], x[o]] = rgb[o].astype(np.uint8)
Image.fromarray(img).save(OUT / "preview_top.png")

# oblique: camera from the south-west, 35° elevation, painter's algorithm
r = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
rv = int(r / VS)
m = (abs(x - hx) < rv) & (abs(y - hy) < rv)
x, y, z, rgb = x[m] - hx, y[m] - hy, z[m] - z[m].min(), rgb[m]
key = set()
occ = np.zeros((2 * rv + 2, 2 * rv + 2, 256), bool)
occ[x + rv, y + rv, z] = True
top_open = ~occ[x + rv, y + rv, np.minimum(z + 1, 255)]
yaw = np.radians(45)
u = x * np.cos(yaw) - y * np.sin(yaw)
w = x * np.sin(yaw) + y * np.cos(yaw)       # depth axis (towards north-east)
el = np.radians(35)
sx = u
sy = -(w * np.sin(el) + z * np.cos(el))
depth = w * np.cos(el) - z * np.sin(el)
S = 3
px = ((sx - sx.min()) * S).astype(int)
py = ((sy - sy.min()) * S).astype(int)
H, W = py.max() + S + 1, px.max() + S + 1
shade = np.where(top_open, 1.0, 0.68)[:, None]
col = np.clip(rgb * shade, 0, 255).astype(np.uint8)
order = np.argsort(-depth)
im = np.full((H, W, 3), (188, 211, 230), np.uint8)
for dy in range(S):
    for dx in range(S):
        im[py[order] + dy, px[order] + dx] = col[order]
Image.fromarray(im).save(OUT / "preview_oblique.png")
print(f"{len(V):,} voxels; wrote {OUT / 'preview_top.png'} and {OUT / 'preview_oblique.png'} ({W}x{H})")
