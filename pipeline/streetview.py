"""Street-level imagery (Mapillary 360°) -> per-voxel colours for the voxel pipeline.

Stages (each cached, rerun only what changed):
  fetch      list all 360° images with SfM poses in the region, keep a subset (one image per
             SPACING m per capture session, newest SESSIONS sessions per CELL m cell), download
             1024 px thumbnails and Mapillary's segmentation for those
  calibrate  per capture session, find the position offset (east, north, height) that best
             matches Mapillary's segmentation (building / sky / ground / vegetation) with the
             voxel classes seen along the same rays (GPS bias differs per session)
  project    cast a ray per pixel into the voxel model (geometry from the built L0 tiles) and
             vote per voxel, one vote per session: colour (facade pixels on building walls only)
             and label (car, pole, sign, ...). Writes layer.npz, which voxelize.py merges.

Usage: uv run python streetview.py fetch|calibrate|project [roi-name] [--jobs=N]
Needs MAPILLARY_TOKEN in ../.env. Image data © Mapillary contributors, CC-BY-SA.
"""
import base64
import json
import multiprocessing as mp
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import cache

import mapbox_vector_tile
import numpy as np
from numba import njit
from PIL import Image, ImageDraw

from common import ARGS, ROOT, VS, jobs as njobs, log_clock, mapillary_token, region, to_l72, to_wgs

ROI_NAME, ROI = region(1)   # (the first argument is the stage)
RAW = ROOT / "data/raw/mapillary"
IMG, DET = RAW / "img", RAW / "det"
WORK = ROOT / "data/cache/streetview" / ROI_NAME
FIELDS = "id,captured_at,sequence,computed_geometry,computed_rotation,thumb_1024_url"

SPACING = 5.0     # m between kept images along one capture session
CELL = 25.0       # m; per cell keep only the newest SESSIONS sessions
SESSIONS = 4
QUERY = 250.0     # m; API bbox size per request (stays below the 2000-result cap)
MARGIN = 30.0     # m; images just outside the region still see facades inside it

log = log_clock()
token = cache(mapillary_token)   # (read when first needed: calibrate / project work offline)


def get(url, auth=True, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"OAuth {token()}"} if auth else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def list_images():
    x0, y0 = ROI["xmin"] - MARGIN, ROI["ymin"] - MARGIN
    x1, y1 = ROI["xmax"] + MARGIN, ROI["ymax"] + MARGIN
    cells = [(x, y) for x in np.arange(x0, x1, QUERY) for y in np.arange(y0, y1, QUERY)]

    def query(c):
        a = to_wgs.transform(c[0], c[1])
        b = to_wgs.transform(min(c[0] + QUERY, x1), min(c[1] + QUERY, y1))
        bb = f"{a[0]:.6f},{a[1]:.6f},{b[0]:.6f},{b[1]:.6f}"
        d = json.loads(get(f"https://graph.mapillary.com/images?bbox={bb}&is_pano=true&limit=2000&fields={FIELDS}"))
        return d["data"]

    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(query, cells))
    if any(len(r) >= 2000 for r in res):
        log("warning: a query hit the 2000-image cap; lower QUERY")
    seen, out = set(), []
    for d in (d for r in res for d in r):
        if d["id"] in seen or not d.get("computed_rotation") or not d.get("computed_geometry"):
            continue
        seen.add(d["id"])
        x, y = to_l72.transform(*d["computed_geometry"]["coordinates"])
        if x0 <= x < x1 and y0 <= y < y1:
            out.append({**d, "x": x, "y": y})
    return out


def select(images):
    """Thin along each session, then keep the newest SESSIONS sessions per CELL."""
    by_seq = {}
    for d in images:
        by_seq.setdefault(d["sequence"], []).append(d)
    thin = []
    for seq in by_seq.values():
        seq.sort(key=lambda d: d["captured_at"])
        last = None
        for d in seq:
            if last is None or np.hypot(d["x"] - last["x"], d["y"] - last["y"]) >= SPACING:
                thin.append(d)
                last = d
    newest = {}   # (cell, sequence) -> capture time of that session
    for d in thin:
        k = (int(d["x"] // CELL), int(d["y"] // CELL))
        newest.setdefault(k, {})[d["sequence"]] = max(newest.get(k, {}).get(d["sequence"], 0), d["captured_at"])
    keep_seq = {k: set(sorted(v, key=v.get, reverse=True)[:SESSIONS]) for k, v in newest.items()}
    return [d for d in thin if d["sequence"] in keep_seq[(int(d["x"] // CELL), int(d["y"] // CELL))]]


def download(d):
    n = 0
    f = IMG / f"{d['id']}.jpg"
    if not f.exists():
        f.write_bytes(get(d["thumb_1024_url"], auth=False))
        n += 1
    f = DET / f"{d['id']}.json"
    if not f.exists():
        f.write_bytes(get(f"https://graph.mapillary.com/{d['id']}/detections?fields=value,geometry"))
        n += 1
    return n


def fetch():
    WORK.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    DET.mkdir(parents=True, exist_ok=True)
    images = list_images()
    log(f"{len(images):,} 360° images with SfM poses in {len({d['sequence'] for d in images})} sessions")
    sel = select(images)
    log(f"selected {len(sel):,} images ({len({d['sequence'] for d in sel})} sessions): one per {SPACING:.0f} m "
        f"per session, newest {SESSIONS} sessions per {CELL:.0f} m cell")
    fails, done = 0, 0
    with ThreadPoolExecutor(16) as ex:
        futs = [ex.submit(download, d) for d in sel]
        for i, fu in enumerate(futs, 1):
            try:
                done += fu.result()
            except Exception:
                fails += 1
            if i % 1000 == 0 or i == len(futs):
                log(f"  {i:,}/{len(futs):,} images ready ({done:,} files downloaded, {fails} failed)")
    ok = [d for d in sel if (IMG / f"{d['id']}.jpg").exists() and (DET / f"{d['id']}.json").exists()]
    for d in ok:
        d.pop("thumb_1024_url", None)
    (WORK / "images.json").write_text(json.dumps(ok))
    log(f"{len(ok):,} images with thumbnail + segmentation -> {WORK / 'images.json'}")


# --- shared: voxel model, labels, rays ----------------------------------------------------
TILES = ROOT / "viewer/public/tiles" / ROI_NAME
PAD = 120            # voxels of empty border around the region (images in the margin)
CAM_H = 2.0          # m above ground: car roof / backpack mounts
MAX_D = 25.0         # m; farther hits are too blurry to be worth a vote
MIN_COS = 0.26       # skip faces hit at > ~75° from their normal
EL_MIN = -25.0       # camera-frame elevation below which the vehicle / rider fills the image
PW, PH = 768, 384    # projection resolution
G = {}               # per-process globals (filled before forking)

# label codes of the label layer; drawn in this order, later wins
L_NAMES = {1: "building", 2: "shop sign", 3: "vegetation", 4: "vehicle", 5: "bike", 6: "pole",
           7: "traffic sign", 8: "fence", 9: "ground", 10: "person", 11: "street furniture", 12: "sky"}
L_RULES = [
    (9, ("construction--flat--", "construction--barrier--curb", "nature--terrain", "marking", "void--ground",
         "object--manhole", "object--catch-basin")),
    (1, ("construction--structure--building", "construction--barrier--wall")),
    (8, ("construction--barrier--fence", "construction--barrier--guard-rail", "construction--barrier--other")),
    (3, ("nature--vegetation",)),
    (2, ("object--sign--store", "object--sign--advertisement", "object--banner")),
    (4, ("object--vehicle--car", "object--vehicle--truck", "object--vehicle--bus", "object--vehicle--other-vehicle",
         "object--vehicle--trailer", "object--vehicle--caravan", "object--vehicle--on-rails",
         "object--vehicle--wheeled-slow")),
    (5, ("object--vehicle--bicycle", "object--vehicle--motorcycle", "human--rider")),
    (6, ("object--support--", "object--street-light")),
    (7, ("object--traffic-sign", "object--traffic-light", "regulatory--", "warning--", "information--",
         "complementary--")),
    (11, ("object--trash-can", "object--bench", "object--bike-rack", "object--junction-box", "object--parking-meter",
          "object--fire-hydrant", "object--mailbox", "object--phone-booth", "object--traffic-cone")),
    (10, ("human--person",)),
    (12, ("nature--sky",)),
    (0, ("void--ego-vehicle",)),
]


def label_image(image_id, W=PW, H=PH):
    """Mapillary's segmentation rasterised to label codes (H, W); 0 = unknown / ignore."""
    im = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(im)
    dets = json.loads((DET / f"{image_id}.json").read_text()).get("data", [])
    for code, prefixes in L_RULES:
        for det in dets:
            if not det["value"].startswith(prefixes):
                continue
            tile = mapbox_vector_tile.decode(base64.b64decode(det["geometry"]), default_options={"y_coord_down": True})
            for layer in tile.values():
                ext = layer.get("extent", 4096)
                for feat in layer["features"]:
                    g = feat["geometry"]
                    for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
                        ring = [(x * W / ext, y * H / ext) for x, y in poly[0]]
                        if len(ring) >= 3:
                            draw.polygon(ring, fill=code)
    lab = np.asarray(im).copy()
    lat = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    lab[lat <= EL_MIN] = 0
    return lab


def rotvec_to_matrix(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def world_dirs(rotvec, W, H):
    """World (east, north, up) direction of every equirectangular pixel, (H, W, 3).
    Mapillary/OpenSfM: computed_rotation is axis-angle world->camera, camera x right, y down, z forward."""
    x = (np.arange(W) + 0.5) / W - 0.5
    y = (np.arange(H) + 0.5) / H - 0.5
    lon, lat = np.meshgrid(x * 2 * np.pi, -y * np.pi)
    b = np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
    return (b @ rotvec_to_matrix(rotvec)).astype(np.float32)


def load_model():
    """L0 occupancy (voxel class, 0 = empty) of the whole region plus PAD, from the built tiles.
    Kept as a memory-mapped file next to the other caches (a whole commune is ~18 GB dense): built once
    per voxel build, shared by the forked workers through the page cache; a camera's rays only touch
    its surroundings, so the part in use stays small."""
    man = json.loads((TILES / "manifest.json").read_text())
    L0 = man["levels"][0]
    nx, ny = L0["size"][0] + 2 * PAD, L0["size"][1] + 2 * PAD
    WORK.mkdir(parents=True, exist_ok=True)
    f = WORK / f"occ-{man.get('build', 'x')}.u8"
    if not f.exists():
        for old in WORK.glob("occ-*.u8"):
            old.unlink()
        tmp = f.with_suffix(".tmp")
        occ = np.memmap(tmp, np.uint8, "w+", shape=(nx, ny, 256))
        for t in L0["tiles"]:
            v = np.frombuffer((TILES / t["file"]).read_bytes(), np.uint8).reshape(-1, 8)
            v = v[(v[:, 7] & 2) == 0]
            occ[v[:, 0].astype(np.int64) - 1 + t["tx"] * 64 + PAD, v[:, 1].astype(np.int64) - 1 + t["ty"] * 64 + PAD,
                v[:, 2]] = np.maximum(v[:, 6], 1)
        occ.flush()
        del occ
        tmp.rename(f)
        log(f"occupancy grid built ({nx}x{ny}x256, {f.stat().st_size / 1e9:.1f} GB on disk)")
    return np.memmap(f, np.uint8, "r", shape=(nx, ny, 256)), man


@njit(cache=True, fastmath=True, nogil=True)
def cast(occ, p0, dirs, maxd, hit, axis, dist):
    """Exact voxel traversal (Amanatides & Woo) from p0 along each dir, in voxel units.
    hit[i] = first occupied voxel, axis[i] = axis of the face entered (-1: nothing within maxd)."""
    nx, ny, nz = occ.shape
    for i in range(dirs.shape[0]):
        axis[i] = -1
        x, y, z = int(np.floor(p0[0])), int(np.floor(p0[1])), int(np.floor(p0[2]))
        d0, d1, d2 = dirs[i, 0], dirs[i, 1], dirs[i, 2]
        sx = 1 if d0 > 0 else -1
        sy = 1 if d1 > 0 else -1
        sz = 1 if d2 > 0 else -1
        tdx = abs(1.0 / d0) if d0 != 0 else 1e30
        tdy = abs(1.0 / d1) if d1 != 0 else 1e30
        tdz = abs(1.0 / d2) if d2 != 0 else 1e30
        tmx = ((x + 1 - p0[0]) if d0 > 0 else (p0[0] - x)) * tdx
        tmy = ((y + 1 - p0[1]) if d1 > 0 else (p0[1] - y)) * tdy
        tmz = ((z + 1 - p0[2]) if d2 > 0 else (p0[2] - z)) * tdz
        while True:
            if tmx < tmy and tmx < tmz:
                x += sx
                t = tmx
                tmx += tdx
                a = 0
            elif tmy < tmz:
                y += sy
                t = tmy
                tmy += tdy
                a = 1
            else:
                z += sz
                t = tmz
                tmz += tdz
                a = 2
            if t > maxd or x < 0 or y < 0 or z < 0 or x >= nx or y >= ny or z >= nz:
                break
            if occ[x, y, z] != 0:
                hit[i, 0], hit[i, 1], hit[i, 2] = x, y, z
                axis[i] = a
                dist[i] = t
                break


def camera(d, off=(0.0, 0.0, 0.0)):
    """Camera position in voxel units of the padded model: SfM position + offset, CAM_H above ground."""
    occ, X0, Y0 = G["occ"], G["X0"], G["Y0"]
    gx = (d["x"] + off[0] - X0) / VS + PAD
    gy = (d["y"] + off[1] - Y0) / VS + PAD
    ix, iy = int(gx), int(gy)
    lows = []
    for c in occ[ix - 1:ix + 2, iy - 1:iy + 2].reshape(-1, occ.shape[2]):
        nz = np.flatnonzero(c)
        if len(nz):
            lows.append(nz[0])
    zg = (np.median(lows) + 1) if lows else 0.0
    return np.array([gx, gy, zg + (CAM_H + off[2]) / VS])


def rays(d, off, dirs):
    occ = G["occ"]
    n = len(dirs)
    hit = np.zeros((n, 3), np.int32)
    axis = np.zeros(n, np.int8)
    dist = np.zeros(n, np.float32)
    cast(occ, camera(d, off), dirs, MAX_D / VS if G.get("maxd") is None else G["maxd"], hit, axis, dist)
    return hit, axis, dist


# --- calibrate --------------------------------------------------------------------------
CW, CH_ = 192, 96    # calibration label resolution
SEGMENT = 10         # images per calibrated segment (~50 m at 5 m spacing): SfM drifts along long sessions
CAL_IMAGES = 5       # images per segment used for scoring
MIN_SCORE = 0.5      # segments agreeing less than this after calibration do not vote
SCORE_W = {1: 2.0, 12: 1.5, 9: 1.0, 3: 0.5}


def segments(data):
    """Each session in time order, cut into runs of SEGMENT images: [(segment id, session, images)]."""
    by = {}
    for d in data:
        by.setdefault(d["sequence"], []).append(d)
    out = []
    for seq, ds in by.items():
        ds.sort(key=lambda d: d["captured_at"])
        n = max(1, round(len(ds) / SEGMENT))
        for i, part in enumerate(np.array_split(np.arange(len(ds)), n)):
            out.append((f"{seq}/{i}", seq, [ds[j] for j in part]))
    return out


def score_imgs(imgs, off):
    """Mean balanced agreement between Mapillary labels and voxel classes along the rays."""
    occ, tot = G["occ"], []
    for d, lab, dirs in imgs:
        hit, axis, _ = rays(d, off, dirs)
        cls = np.where(axis >= 0, occ[hit[:, 0], hit[:, 1], hit[:, 2]], 0)
        got = np.select([cls == 6, (cls == 3) | (cls == 5), cls == 2, axis < 0], [1, 3, 9, 12], 0)
        s = w = 0.0
        for k, wk in SCORE_W.items():
            m = lab == k
            if m.sum() > 30:
                s += wk * (got[m] == k).mean()
                w += wk
        if w:
            tot.append(s / w)
    return float(np.mean(tot)) if tot else 0.0


def calibrate_segment(args):
    sid, seq, ds = args
    pick = [ds[i] for i in np.unique(np.linspace(0, len(ds) - 1, min(CAL_IMAGES, len(ds))).round().astype(int))]
    imgs = []
    for d in pick:
        lab = label_image(d["id"], CW, CH_).ravel()
        dirs = world_dirs(d["computed_rotation"], CW, CH_).reshape(-1, 3)
        m = np.isin(lab, list(SCORE_W))
        imgs.append((d, lab[m], np.ascontiguousarray(dirs[m])))
    G["maxd"] = 40.0 / VS
    s0 = score_imgs(imgs, (0.0, 0.0, 0.0))
    best = (s0, (0.0, 0.0, 0.0))
    cands = [(x, y, 0.0) for x in np.arange(-6, 6.01, 1.5) for y in np.arange(-6, 6.01, 1.5)]
    for step in (None, 0.75, 0.375):
        if step:
            c = best[1]
            cands = [(c[0] + i * step, c[1] + j * step, c[2]) for i in (-1, 0, 1) for j in (-1, 0, 1)]
        for off in cands:
            sc = score_imgs(imgs, off)
            if sc > best[0]:
                best = (sc, off)
    for dz in (-0.5, 0.5):
        off = (best[1][0], best[1][1], best[1][2] + dz)
        sc = score_imgs(imgs, off)
        if sc > best[0]:
            best = (sc, off)
    return sid, {"offset": [float(v) for v in best[1]], "score": best[0], "score_raw": s0, "n": len(ds),
                 "images": [d["id"] for d in ds]}


def calibrate():
    jobs = njobs()
    data = json.loads((WORK / "images.json").read_text())
    G["occ"], man = load_model()
    G["X0"], G["Y0"] = man["origin"][0], man["origin"][1]
    segs = segments(data)
    log(f"calibrating {len(segs)} segments of {len({s[1] for s in segs})} sessions on {jobs} jobs")
    res = {}
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
        for i, (sid, r) in enumerate(ex.map(calibrate_segment, segs, chunksize=4), 1):
            res[sid] = r
            if i % 200 == 0 or i == len(segs):
                log(f"  {i}/{len(segs)} segments")
    (WORK / "calibration.json").write_text(json.dumps(res))
    sc = np.array([r["score"] for r in res.values()])
    raw = np.array([r["score_raw"] for r in res.values()])
    off = np.array([r["offset"] for r in res.values()])
    edge = (np.abs(off[:, :2]) >= 5.9).any(1)
    log(f"agreement {raw.mean():.3f} -> {sc.mean():.3f}; offset median east {np.median(off[:, 0]):+.2f} north "
        f"{np.median(off[:, 1]):+.2f} height {np.median(off[:, 2]):+.2f} m; {edge.sum()} segments at the search edge; "
        f"{(sc < MIN_SCORE).sum()} segments ({sum(r['n'] for r in res.values() if r['score'] < MIN_SCORE)} images) "
        f"below {MIN_SCORE} will not vote")


# --- project ----------------------------------------------------------------------------
def project_session(args):
    """All images of one session -> per voxel: median colour (facade pixels on walls) and
    majority label. One session = one vote later, so long sessions cannot dominate."""
    seq, jobs = args                 # jobs: [(image, offset)]
    occ, nx, ny = G["occ"], G["NX"], G["NY"]
    G["maxd"] = None
    ks, cs, lk = [], [], []
    for d, off in jobs:
        lab = label_image(d["id"]).ravel()
        keep = lab > 0
        dirs = np.ascontiguousarray(world_dirs(d["computed_rotation"], PW, PH).reshape(-1, 3)[keep])
        lab = lab[keep]
        im = np.asarray(Image.open(IMG / f"{d['id']}.jpg").convert("RGB").resize((PW, PH), Image.BILINEAR))
        pix = im.reshape(-1, 3)[keep]
        hit, axis, _ = rays(d, off, dirs)
        h = axis >= 0
        x, y, z = hit[h, 0] - PAD, hit[h, 1] - PAD, hit[h, 2]
        a, lb, px, dr = axis[h], lab[h], pix[h], dirs[h]
        inside = (x >= 0) & (y >= 0) & (x < nx) & (y < ny)
        cls = occ[hit[h, 0], hit[h, 1], hit[h, 2]]
        cosang = np.abs(dr[np.arange(len(a)), a])
        key = (z.astype(np.int64) * ny + y) * nx + x
        # colour: facade / shop-sign pixels on side faces of building voxels
        c = inside & (a < 2) & (cls == 6) & ((lb == 1) | (lb == 2)) & (cosang > MIN_COS)
        ks.append(key[c])
        cs.append(px[c])
        lk.append(key[inside] * 16 + lb[inside])
    key = np.concatenate(ks)
    col = np.concatenate(cs)
    ck, crgb, cn = median_per_key(key, col)
    ok = cn >= 2                                        # a session needs 2+ samples to vote a colour
    lk = np.concatenate(lk)
    u, n = np.unique(lk, return_counts=True)            # (voxel, label) sample counts in this session
    vk, lb = u // 16, (u % 16).astype(np.uint8)
    o = np.lexsort((-n, vk))                            # per voxel: most frequent label first
    vk, lb, n = vk[o], lb[o], n[o]
    first = np.r_[True, vk[1:] != vk[:-1]]
    return ck[ok], crgb[ok], cn[ok], vk[first], lb[first], n[first]


def median_per_key(keys, cols):
    """Per unique key: per-channel median of cols and sample count."""
    if not len(keys):
        return keys, np.zeros((0, 3), np.uint8), np.zeros(0, np.int64)
    o = np.argsort(keys, kind="stable")
    keys, cols = keys[o], cols[o]
    ukey, start, cnt = np.unique(keys, return_index=True, return_counts=True)
    med = np.empty((len(ukey), 3), np.uint8)
    for ch in range(3):
        o = np.lexsort((cols[:, ch], keys))
        med[:, ch] = cols[o, ch][start + cnt // 2]
    return ukey, med, cnt


def project():
    jobs = njobs()
    data = json.loads((WORK / "images.json").read_text())
    calf = WORK / "calibration.json"
    cal = json.loads(calf.read_text()) if calf.exists() else {}
    G["occ"], man = load_model()
    G["X0"], G["Y0"] = man["origin"][0], man["origin"][1]
    G["NX"], G["NY"] = man["levels"][0]["size"][:2]
    if not cal:
        log("warning: no calibration.json, using Mapillary positions as is")
    by, skipped = {}, 0
    glob = None
    if "--global" in sys.argv and cal:   # one shared offset: median over well-matching segments
        good = [c["offset"] for c in cal.values() if c["score"] >= MIN_SCORE]
        glob = np.median(good, 0).tolist()
        log(f"global offset {glob}")
    for sid, seq, ds in segments(data):
        c = cal.get(sid)
        if c and (c["score"] < MIN_SCORE or max(abs(c["offset"][0]), abs(c["offset"][1])) >= 5.9):  # no real optimum
            skipped += len(ds)
            continue
        off = glob or (c["offset"] if c else [0.0, 0.0, 0.0])
        by.setdefault(seq, []).extend((d, off) for d in ds)
    tasks = sorted(by.items(), key=lambda kv: -len(kv[1]))
    log(f"projecting {sum(len(t[1]) for t in tasks):,} images in {len(tasks)} sessions on {jobs} jobs "
        f"({skipped:,} images in badly matching or unresolved segments skipped)")
    C, L = [], []
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
        for i, (ck, crgb, cn, vk, lb, _) in enumerate(ex.map(project_session, tasks), 1):
            C.append((ck, crgb, cn))
            L.append((vk, lb))
            if i % 50 == 0 or i == len(tasks):
                log(f"  {i}/{len(tasks)} sessions")
    # colour: median across sessions; keep if 2+ sessions agree, or one session saw it 4+ times
    ck = np.concatenate([c[0] for c in C])
    crgb = np.concatenate([c[1] for c in C])
    strong = np.unique(np.concatenate([c[0][c[2] >= 4] for c in C]))
    key, rgb, nses = median_per_key(ck, crgb)
    sel = (nses >= 2) | np.isin(key, strong)
    key, rgb, nses = key[sel], rgb[sel], nses[sel]
    # label: the label most sessions voted for (ties: lowest code)
    vk = np.concatenate([l[0] for l in L])
    lb = np.concatenate([l[1] for l in L])
    u, n = np.unique(vk * 16 + lb, return_counts=True)
    lv, ll = u // 16, (u % 16).astype(np.uint8)
    o = np.lexsort((-n, lv))
    lv, ll, n = lv[o], ll[o], n[o]
    first = np.r_[True, lv[1:] != lv[:-1]]
    lkey, lab, lnses = lv[first], ll[first], n[first].astype(np.uint16)
    np.savez(WORK / "layer.npz", key=key, rgb=rgb, nses=nses.astype(np.uint16),
             lkey=lkey, label=lab, lnses=lnses, size=np.array([G["NX"], G["NY"]]), build=man.get("build", ""))
    log(f"colour layer: {len(key):,} voxels (median {np.median(nses):.0f} sessions each)")
    cnt = np.bincount(lab[lnses >= 2], minlength=16)
    log("label layer (2+ sessions agree): " + ", ".join(f"{L_NAMES[k]} {cnt[k]:,}" for k in L_NAMES if cnt[k]))


if __name__ == "__main__":
    {"fetch": fetch, "calibrate": calibrate, "project": project}[ARGS[0] if ARGS else "fetch"]()
