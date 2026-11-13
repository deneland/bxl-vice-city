"""Convert a built region's tiles from raw records (.bin, manifest version 2) to the compact format
(.vxt, version 3; common.encode_tile), in place, without rebuilding from the LiDAR. New builds of
voxelize.py write the compact format directly.

Usage: uv run python repack.py [name] [--jobs=N]   (a folder in viewer/public/tiles/)
"""
import json
import multiprocessing as mp
import time

from common import ARGS, CFG, ROOT, decode_tile, encode_tile, jobs

NAME = ARGS[0] if ARGS else CFG["default"]   # (the built folder's name: also regions no longer in config.toml)
DIR = ROOT / "viewer/public/tiles" / NAME


def convert(args):
    src, width = args
    path = DIR / src
    data = encode_tile(decode_tile(path.read_bytes()), width)
    dst = path.with_suffix(".vxt")
    dst.write_bytes(data)
    return src, dst.relative_to(DIR).as_posix(), path.stat().st_size, len(data)


def main():
    man_path = DIR / "manifest.json"
    man = json.loads(man_path.read_text())
    if man.get("version", 2) >= 3:
        print(f"{NAME}: already compact (manifest version {man['version']})")
        return
    width = man["tileSize"] + 2
    todo = [(t["file"], width) for lv in man["levels"] for t in lv["tiles"]]
    t0, before, after, names = time.time(), 0, 0, {}
    with mp.Pool(jobs()) as pool:
        for i, (src, dst, a, b) in enumerate(pool.imap_unordered(convert, todo, chunksize=16)):
            names[src] = dst
            before += a
            after += b
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(todo)} tiles", flush=True)
    for lv in man["levels"]:
        for t in lv["tiles"]:
            t["file"] = names[t["file"]]
        lv["bytes"] = sum((DIR / t["file"]).stat().st_size for t in lv["tiles"])
    man["version"] = 3
    man["build"] += "-c"   # (cache busting)
    # the manifest switches the viewer over; the old files go after it
    tmp = man_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(man, indent=1))
    tmp.replace(man_path)
    for src in names:
        (DIR / src).unlink()
    print(f"{NAME}: {len(todo)} tiles, {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB ({after / before:.1%}) "
          f"in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
