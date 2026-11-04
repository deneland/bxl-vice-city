"""Read access to the built L0 voxel tiles (viewer/public/tiles/<roi>), tile by tile on demand."""
import json
from collections import OrderedDict

import numpy as np

from common import FLAG_APRON, ROOT



class Model:
    """L0 voxel classes, tile by tile on demand (LRU: a whole commune would not fit in memory)."""
    MAX_TILES = 1500   # ~1 MB each

    def __init__(self, roi: str):
        self.dir = ROOT / "viewer/public/tiles" / roi
        self.man = json.loads((self.dir / "manifest.json").read_text())
        L0 = self.man["levels"][0]
        self.vs, self.size = L0["voxelSize"], L0["size"]
        self.files = {(t["tx"], t["ty"]): t["file"] for t in L0["tiles"]}
        self.cache = OrderedDict()

    def tile(self, tx, ty):
        if (tx, ty) in self.cache:
            self.cache.move_to_end((tx, ty))
        else:
            occ = np.zeros((64, 64, self.size[2]), np.uint8)
            f = self.files.get((tx, ty))
            if f:
                v = np.frombuffer((self.dir / f).read_bytes(), np.uint8).reshape(-1, 8)
                v = v[(v[:, 7] & FLAG_APRON) == 0]
                occ[v[:, 0] - 1, v[:, 1] - 1, v[:, 2]] = np.maximum(v[:, 6], 1)
            self.cache[(tx, ty)] = occ
            if len(self.cache) > self.MAX_TILES:
                self.cache.popitem(last=False)
        return self.cache[(tx, ty)]

    def column(self, x, y):
        if not (0 <= x < self.size[0] and 0 <= y < self.size[1]):
            return None
        return self.tile(x // 64, y // 64)[x % 64, y % 64]
