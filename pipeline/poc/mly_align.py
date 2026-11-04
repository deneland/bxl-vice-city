"""POC: check Mapillary's SfM poses for 360° images by overlaying voxel-model edges.

Usage: uv run python poc/mly_align.py [n_images]
Uses MAPILLARY_TOKEN from ../.env. Mapillary/OpenSfM conventions: computed_rotation is an
axis-angle world->camera rotation in a local east-north-up frame; camera x right, y down,
z forward; equirectangular bearing = (cos(lat) sin(lon), -sin(lat), cos(lat) cos(lon)).
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from pano_align import OUT, ROOT, ground_z, load_voxels, to_l72  # noqa: E402
from pano_pose import render_depth, silhouette_edges  # noqa: E402

RAW = ROOT / "data/raw/mapillary"
RAW.mkdir(parents=True, exist_ok=True)
TOKEN = next(l.split("=", 1)[1].strip().strip("'\"") for l in (ROOT / ".env").read_text().splitlines() if l.startswith("MAPILLARY_TOKEN"))
FIELDS = "id,captured_at,is_pano,camera_type,sequence,computed_geometry,computed_rotation,computed_altitude,altitude,thumb_2048_url"


def get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"OAuth {TOKEN}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def rotvec_to_matrix(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def world_dirs(R, W, H):
    """World (east, north, up) direction of every equirectangular pixel."""
    x = (np.arange(W) + 0.5) / W - 0.5
    y = (np.arange(H) + 0.5) / H - 0.5
    lon, lat = np.meshgrid(x * 2 * np.pi, -y * np.pi)
    b = np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
    return b @ R  # R^T b for each pixel (R maps world -> camera)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    f = RAW / "street_360.json"
    if not f.exists():
        data = json.loads(get(f"https://graph.mapillary.com/images?bbox=4.3760,50.8671,4.3788,50.8684&limit=2000&fields={FIELDS}"))["data"]
        data = [d for d in data if d.get("is_pano") and d.get("computed_rotation") and d.get("computed_geometry")]
        f.write_text(json.dumps(data, indent=1))
    data = json.loads(f.read_text())
    # spread the sample over capture sessions
    by = {}
    for d in data:
        by.setdefault(d["sequence"], []).append(d)
    pick = [seq[len(seq) // 2] for seq in sorted(by.values(), key=len, reverse=True)][:n]
    W, H = 1024, 512
    for d in pick:
        img = RAW / f"{d['id']}.jpg"
        if not img.exists():
            img.write_bytes(get(d["thumb_2048_url"]))
        lon, lat = d["computed_geometry"]["coordinates"]
        x, y = to_l72.transform(lon, lat)
        occ, _, origin = load_voxels(x, y, 70)
        gz = ground_z(occ, origin, (x, y))
        R = rotvec_to_matrix(d["computed_rotation"])
        D = world_dirs(R, W, H)
        depth = render_depth(occ, origin, (x, y, gz + 2.0), D)
        e = silhouette_edges(depth) > 0
        photo = np.asarray(Image.open(img).convert("RGB").resize((W, H)))
        o = photo.copy()
        o[e] = [0, 255, 0]
        Image.fromarray(np.concatenate([photo, o], 0)).save(OUT / f"mly_{d['id']}.png")
        date = np.datetime64(d["captured_at"], "ms").astype(str)[:10]
        print(f"{d['id']} {date} seq {d['sequence'][:8]} ({len(by[d['sequence']])} imgs) -> mly_{d['id']}.png")
    print(f"{len(data)} usable 360° images in {len(by)} sequences")


if __name__ == "__main__":
    main()
