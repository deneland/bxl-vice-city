"""Contact sheet of junctions from viewer/bench/junctions.mjs: the worst-scoring ones and a random sample by
type (3-way, 4-way, 5+ way, signalled), each drawn 80 m wide: incoming lane ends (blue), outgoing lane
starts (green), the curves joining them (orange; red when flagged), the junction centre (black), road
centrelines (grey) and junction pieces (purple: road surface inside a merged junction). Lights: stop
lines as squares coloured by signal group (red / cyan), crossing lights as magenta crosses, crosswalks as
small circles (filled: run with the lights; open: not).

Usage: uv run python qa_junctions.py [junctions.json]   (default: the newest run)
"""
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "data/metrics/junctions").glob("*/junctions.json"))[-1]
js = json.loads(src.read_text())
roads = json.loads((ROOT / "viewer/public/roads/schaerbeek.json").read_text())
VS = 0.5
# road centrelines and junction pieces in the viewer's frame (x east, z south: x * vs, -y * vs)
lines = [("edge", [(e[6][i] * VS, -e[6][i + 1] * VS) for i in range(0, len(e[6]), 3)]) for e in roads["edges"] if e[2] == 0]
lines += [("piece", [(e[6][i] * VS, -e[6][i + 1] * VS) for i in range(0, len(e[6]), 3)]) for e in roads.get("pieces", []) if e[2] == 0]
random.seed(1)

# junctions the user pointed out (viewer frame: x east, z south, metres from the region origin)
WATCH = {"Chazal/Rogier": (2298, -1462), "Place Bremer": (1879, -1384), "Chazal roundabout": (2409, -1763)}
watch = [min(js, key=lambda j: (j["centre"][0] - x) ** 2 + (j["centre"][1] - z) ** 2) for x, z in WATCH.values()]
worst = sorted(js, key=lambda j: -j["score"])[:12]
pick = watch + [j for j in worst if j not in watch][:9]
for label, sel in (("3-way", lambda j: j["degree"] == 3), ("4-way", lambda j: j["degree"] == 4),
                   ("5+", lambda j: j["degree"] >= 5), ("signals", lambda j: j["signalled"])):
    pool = [j for j in js if sel(j) and j not in pick]
    pick += random.sample(pool, min(6 if label == "signals" else 3, len(pool)))

T, M = 320, 80.0                  # tile px, metres across
cols = 6
pick = pick[:30]
rows = (len(pick) + cols - 1) // cols
sheet = Image.new("RGB", (cols * T, rows * T), (250, 250, 250))
for k, j in enumerate(pick):
    tile = Image.new("RGB", (T, T), (250, 250, 250))   # (own image: long lines must not run into the next tile)
    d = ImageDraw.Draw(tile)
    ox, oy = 0, 0
    cx, cz = j["centre"]
    tx = lambda q: (ox + T / 2 + (q[0] - cx) * T / M, oy + T / 2 + (q[1] - cz) * T / M)
    for kind, ln in lines:   # centrelines (grey) and junction pieces (purple) around it
        if min(abs(q[0] - cx) + abs(q[1] - cz) for q in ln) < M:
            d.line([tx(q) for q in ln], fill=(150, 60, 200) if kind == "piece" else (190, 190, 190), width=1)
    for a in j["ins"]:
        if len(a["line"]) > 1:
            d.line([tx(q) for q in a["line"]], fill=(40, 80, 220), width=2)
    for b in j["outs"]:
        if len(b["line"]) > 1:
            d.line([tx(q) for q in b["line"]], fill=(40, 170, 60), width=2)
    for c in j["curves"]:
        bad = c["chord"] > 3 and c["len"] > 2 * c["chord"]
        if len(c["pts"]) > 1:
            d.line([tx(q) for q in c["pts"]], fill=(220, 30, 30) if bad else (240, 150, 30), width=1)
    GROUP = [(220, 30, 30), (0, 170, 200)]
    for a in j["ins"]:
        if a.get("light"):
            x, y = tx(a["light"]["at"])
            col = GROUP[a["light"]["group"] % 2]
            d.rectangle([x - 4, y - 4, x + 4, y + 4], fill=col if a["light"]["osm"] else None, outline=col, width=2)
    for x_ in [c for a in j["ins"] + j["outs"] for c in a.get("crossings", [])]:
        x, y = tx(x_)
        d.line([x - 5, y - 5, x + 5, y + 5], fill=(220, 0, 220), width=2); d.line([x - 5, y + 5, x + 5, y - 5], fill=(220, 0, 220), width=2)
    for z in [z for a in j["ins"] + j["outs"] for z in a.get("zebras", [])]:
        x, y = tx(z["at"])
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(60, 60, 60) if z["walk"] >= 0 else None, outline=(60, 60, 60))
    x, y = tx(j["centre"])
    d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(0, 0, 0))
    flags = " ".join(f"{f}{v}" for f, v in j["flags"].items() if v)
    d.text((ox + 4, oy + 4), f"node {j['node']} deg {j['degree']}{' sig' if j['signalled'] else ''} score {j['score']}", fill=(0, 0, 0))
    d.text((ox + 4, oy + 16), flags, fill=(200, 0, 0))
    d.rectangle([ox, oy, ox + T - 1, oy + T - 1], outline=(200, 200, 200))
    sheet.paste(tile, ((k % cols) * T, (k // cols) * T))
out = src.parent / "sheet.png"
sheet.save(out)
print(f"{len(pick)} junctions ({len(worst)} worst, the rest sampled by type) -> {out}")
