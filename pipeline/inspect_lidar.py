"""Print a quick quality report for the LiDAR tile(s) covering the ROI.

Usage: uv run python inspect_lidar.py ../data/raw/lidar/<file>.las
"""

import sys
import tomllib
from pathlib import Path

import laspy
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CFG = tomllib.loads((ROOT / "config.toml").read_text())
ROI = CFG["roi"]

# ASPRS standard class names (LAS 1.4)
CLASS_NAMES = {
    0: "never classified", 1: "unclassified", 2: "ground", 3: "low vegetation",
    4: "medium vegetation", 5: "high vegetation", 6: "building", 7: "low noise",
    8: "reserved/model key", 9: "water", 10: "rail", 11: "road surface",
    12: "reserved/overlap", 13: "wire guard", 14: "wire conductor",
    15: "transmission tower", 17: "bridge deck", 18: "high noise",
}


def report(path: Path) -> None:
    with laspy.open(path) as f:
        h = f.header
        dims = list(h.point_format.dimension_names)
        print(f"file            {path.name}  ({path.stat().st_size / 1e9:.2f} GB)")
        print(f"LAS version     {h.version}   point format {h.point_format.id}")
        print(f"points          {h.point_count:,}")
        print(f"bounds x        {h.mins[0]:.1f} – {h.maxs[0]:.1f}")
        print(f"bounds y        {h.mins[1]:.1f} – {h.maxs[1]:.1f}")
        print(f"bounds z        {h.mins[2]:.1f} – {h.maxs[2]:.1f}")
        print(f"scale/offset    {h.scales} / {h.offsets}")
        crs = h.parse_crs()
        print(f"CRS             {crs.to_string() if crs else 'not in header'}")
        print(f"dimensions      {', '.join(dims)}")
        has_rgb = "red" in dims

        cls_counts = np.zeros(256, dtype=np.int64)
        roi_cls = np.zeros(256, dtype=np.int64)
        returns = np.zeros(16, dtype=np.int64)
        rgb_nonzero = 0
        rgb_max = 0
        roi_z = []
        n_roi = 0
        for chunk in f.chunk_iterator(5_000_000):
            x, y, z = chunk.x, chunk.y, chunk.z
            c = np.asarray(chunk.classification, dtype=np.uint8)
            cls_counts += np.bincount(c, minlength=256)
            returns += np.bincount(np.asarray(chunk.number_of_returns), minlength=16)[:16]
            m = (x >= ROI["xmin"]) & (x < ROI["xmax"]) & (y >= ROI["ymin"]) & (y < ROI["ymax"])
            n_roi += int(m.sum())
            roi_cls += np.bincount(c[m], minlength=256)
            if m.any():
                zs = np.asarray(z[m])
                roi_z.append(zs[:: max(1, len(zs) // 200_000)])
            if has_rgb:
                r = np.asarray(chunk.red)
                rgb_nonzero += int((r > 0).sum())
                rgb_max = max(rgb_max, int(r.max()), int(np.asarray(chunk.green).max()))

    total = cls_counts.sum()
    print("\nclassification (whole tile | ROI):")
    for k in np.nonzero(cls_counts)[0]:
        print(f"  {k:3d} {CLASS_NAMES.get(int(k), '?'):20s} {cls_counts[k] / total:6.1%} | "
              f"{roi_cls[k] / max(n_roi, 1):6.1%}")
    print("\nnumber_of_returns distribution:",
          {int(i): f"{returns[i] / returns.sum():.1%}" for i in np.nonzero(returns)[0]})
    if has_rgb:
        print(f"\nRGB             present; non-zero red on {rgb_nonzero / total:.1%} of points, "
              f"max value {rgb_max} ({'16-bit' if rgb_max > 255 else '8-bit'} range)")
    else:
        print("\nRGB             not present")

    area = (ROI["xmax"] - ROI["xmin"]) * (ROI["ymax"] - ROI["ymin"])
    print(f"\nROI '{ROI['name']}' {ROI['xmax'] - ROI['xmin']}×{ROI['ymax'] - ROI['ymin']} m: "
          f"{n_roi:,} points, {n_roi / area:.1f} pts/m²")
    if roi_z:
        z = np.concatenate(roi_z)
        p = np.percentile(z, [0.1, 1, 50, 99, 99.9])
        print(f"ROI z percentiles 0.1/1/50/99/99.9: {' / '.join(f'{v:.1f}' for v in p)}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        report(Path(arg))
