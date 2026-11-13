"""Bring a built region up to the compact formats in place, without rebuilding from the source data:
  tiles     raw records (.bin) -> compact .vxt (common.encode_tile)
  manifest  tile lists -> columns (version 4, common.pack_level_tiles)
  roads     polylines -> delta-coded tenths of a voxel (common.write_roads)
  transit   trips -> patterns, timings and distinct days (common.pack_timetable)
Each step is skipped when already done. New runs of voxelize.py, roads.py and transit.py write these directly.

Usage: uv run python repack.py [name] [--jobs=N]   (a folder in viewer/public/tiles/)
"""
import json
import multiprocessing as mp
import time

from common import (ARGS, CFG, MANIFEST_VERSION, ROADS_XYZ, ROOT, decode_tile, encode_tile, jobs, level_tiles,
                    pack_level_tiles, pack_timetable, read_roads, tile_file, write_roads)

NAME = ARGS[0] if ARGS else CFG["default"]   # (the built folder's name: also regions no longer in config.toml)
PUBLIC = ROOT / "viewer/public"
DIR = PUBLIC / "tiles" / NAME


def convert(args):
    src, dst, width = args
    data = encode_tile(decode_tile((DIR / src).read_bytes()), width)
    (DIR / dst).write_bytes(data)
    return (DIR / src).stat().st_size, len(data)


def tiles():
    man_path = DIR / "manifest.json"
    man = json.loads(man_path.read_text())
    if man.get("version", 2) >= MANIFEST_VERSION:
        print(f"{NAME}: tiles already compact (manifest version {man['version']})")
        return
    ch = man["tileSize"]
    todo, old = [], []
    for lv in man["levels"]:
        for t in level_tiles(lv, ch):
            dst = tile_file(lv["level"], t["tx"], t["ty"])
            if t["file"] != dst:
                todo.append((t["file"], dst, ch + 2))
                old.append(t["file"])
    t0, before, after = time.time(), 0, 0
    if todo:
        with mp.Pool(jobs()) as pool:
            for i, (a, b) in enumerate(pool.imap_unordered(convert, todo, chunksize=16)):
                before += a
                after += b
                if (i + 1) % 4000 == 0:
                    print(f"  {i + 1}/{len(todo)} tiles", flush=True)
    for lv in man["levels"]:
        ts = [{**t, "file": tile_file(lv["level"], t["tx"], t["ty"])} for t in level_tiles(lv, ch)]
        lv["bytes"] = sum((DIR / t["file"]).stat().st_size for t in ts)
        lv["tiles"] = pack_level_tiles(ts, lv, ch)
    man["version"] = MANIFEST_VERSION
    man["build"] += "-c"   # (cache busting)
    # the manifest switches the viewer over; the old files go after it
    tmp = man_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(man, separators=(",", ":")))
    tmp.replace(man_path)
    for f in old:
        (DIR / f).unlink()
    if todo:
        print(f"{NAME}: {len(todo)} tiles, {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB ({after / before:.1%}) in {time.time() - t0:.0f} s")
    print(f"{NAME}: manifest -> version {MANIFEST_VERSION}, {man_path.stat().st_size / 1e3:.0f} kB")


def roads():
    f = PUBLIC / "roads" / f"{NAME}.json"
    if not f.exists() or json.loads(f.read_text()).get("xyz") == ROADS_XYZ:
        return
    before = f.stat().st_size
    write_roads(f, read_roads(f))
    print(f"{NAME}: roads {before / 1e6:.1f} -> {f.stat().st_size / 1e6:.1f} MB")


def transit():
    f = PUBLIC / "transit" / f"{NAME}.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    if "days" not in d:
        return
    before = f.stat().st_size
    out = {"routes": d["routes"], "stops": d["stops"], **pack_timetable(d["days"]),
           "fields": {**d.get("fields", {}), "trips": "see common.pack_timetable"}}
    out["fields"].pop("days", None)
    f.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    print(f"{NAME}: timetable {before / 1e6:.1f} -> {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    tiles()
    roads()
    transit()
