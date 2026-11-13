"""Car paint colours as seen in Schaerbeek: k-means over the colours of the parked cars voxelize.py found in
the LiDAR (and removed from the geometry; aerial-photo colours, i.e. mostly roofs), sampled into
data/cache/vehicles/<roi>_rgb.npy -> viewer/public/vehicles/<roi>.json, used by the traffic models.

Usage: uv run python vehicles.py [roi-name]
"""
import json

import numpy as np

from common import ROOT, region

ROI_NAME, _ = region()
SRC = ROOT / "data/cache/vehicles" / f"{ROI_NAME}_rgb.npy"
OUT = ROOT / "viewer/public/vehicles" / f"{ROI_NAME}.json"
K = 12


def main():
    if not SRC.exists():
        raise SystemExit(f"{SRC.relative_to(ROOT)} missing: build the voxels first (voxelize.py)")
    c = np.load(SRC).astype(np.float32)
    if len(c) < K:
        raise SystemExit(f"only {len(c)} vehicle colours in {SRC.relative_to(ROOT)}")
    rng = np.random.default_rng(1)
    c = c[rng.choice(len(c), min(len(c), 200000), replace=False)]
    # k-means (Lloyd) in sRGB
    cent = c[rng.choice(len(c), K, replace=False)]
    for _ in range(30):
        lab = np.argmin(((c[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        cent = np.array([c[lab == k].mean(0) if (lab == k).any() else cent[k] for k in range(K)])
    w = np.bincount(lab, minlength=K) / len(c)
    order = np.argsort(-w)
    palette = [{"rgb": [int(x) for x in cent[k]], "share": round(float(w[k]), 3)} for k in order]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"palette": palette}, indent=1))
    print(f"{len(c)} vehicle voxels sampled -> {OUT.relative_to(ROOT)}")
    for p in palette:
        print("  #%02x%02x%02x %.1f%%" % (*p["rgb"], 100 * p["share"]))


if __name__ == "__main__":
    main()
