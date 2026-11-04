"""Car paint colours as seen in Schaerbeek: k-means over the colours of voxels labelled vehicle
(parked cars in the LiDAR; aerial-photo colours, i.e. mostly roofs) -> viewer/public/vehicles/<roi>.json,
used by the traffic models.

Usage: uv run python vehicles.py [roi-name]
"""
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CFG = tomllib.loads((ROOT / "config.toml").read_text())
ROI_NAME = next((a for a in sys.argv[1:] if not a.startswith("-")), CFG["default"])
TILES = ROOT / "viewer/public/tiles" / ROI_NAME
OUT = ROOT / "viewer/public/vehicles" / f"{ROI_NAME}.json"
VEHICLE, APRON = 4, 2
K = 12


def main():
    man = json.loads((TILES / "manifest.json").read_text())
    cols = []
    for t in man["levels"][0]["tiles"]:
        v = np.frombuffer((TILES / t["file"]).read_bytes(), np.uint8).reshape(-1, 8)
        m = ((v[:, 7] >> 4) == VEHICLE) & ((v[:, 7] & APRON) == 0)
        cols.append(v[m, 3:6])
    c = np.concatenate(cols).astype(np.float32)
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
    OUT.write_text(json.dumps({"build": man["build"], "palette": palette}, indent=1))
    print(f"{len(c)} vehicle voxels sampled -> {OUT.relative_to(ROOT)}")
    for p in palette:
        print("  #%02x%02x%02x %.1f%%" % (*p["rgb"], 100 * p["share"]))


if __name__ == "__main__":
    main()
