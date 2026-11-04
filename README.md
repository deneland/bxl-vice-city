# bxl vice voxel

A voxel city of Schaerbeek (Brussels) built from real data: the 2021 Brussels LiDAR, OpenStreetMap,
Mapillary street-level imagery and the STIB timetable. The browser viewer draws the city in 0.5 m voxels
and simulates traffic, trams, buses, parked cars and people walking. It should feel real; it is not an
accurate simulator.

```
pipeline/   Python: raw data -> voxel tiles + JSON for the viewer (viewer/public/, gitignored)
viewer/     TypeScript + three.js (WebGPU renderer, WebGL2 fallback), Vite
config.toml regions (Lambert 72 boxes), voxel size, the start view
data/       raw downloads, caches and metrics (gitignored)
```

## Setup

- Python 3.12+ with [uv](https://docs.astral.sh/uv/): `cd pipeline && uv sync`
- Node 22+: `cd viewer && npm install`
- `.env` in the repository root with `MAPILLARY_TOKEN=...` (needed by `streetview.py` and `lights.py`
  only). Never commit it.
- Manual downloads into `data/raw/`:
  - `lidar/`: the Brussels 2021 LiDAR LAS tiles listed for the region in `config.toml`
    (datastore.brussels, CC-BY Paradigm / UrbIS)
  - `gtfs/stib.zip`: the STIB-MIVB GTFS feed (STIB open data)
  - OpenStreetMap and Mapillary data are fetched by the scripts themselves and cached in `data/raw/`
    (`--refetch` to fetch again)

## Building a region

Run from `pipeline/`, with the region name from `config.toml` (default `schaerbeek`). Order matters:
later steps read earlier outputs.

| step | script | reads | writes |
|---|---|---|---|
| 1 | `voxelize.py` | LiDAR | voxel tiles (`viewer/public/tiles/`), parked-car footprints and colours |
| 2 | `streetview.py fetch / calibrate / project` | Mapillary, tiles | photo colours and labels layer |
| 3 | `roads.py` | OSM, tiles (ground heights) | road / tram / footpath network |
| 4 | `voxelize.py` again | LiDAR, photo layer, road network | final tiles: photo colours, streets tidied |
| 5 | `places.py`, `transit.py`, `lights.py`, `vehicles.py` | OSM, GTFS, Mapillary, tiles, car colours | homes and destinations, timetable, street lamps, car paint palette |
| 6 | `parking.py` | footprints, roads, transit, OSM | kerbside, car-park and garage places |

`voxelize.py` streams the LAS files once into `data/cache/points/` and reuses that cache; a full
Schaerbeek build takes ~7 minutes. The viewer serves the newest build (`manifest.json`). Shared code
(config, region argument, coordinate transforms, OSM / Mapillary access, the tile and road formats) is in
`pipeline/common.py`. Lint: `uv run ruff check`.

## Viewer

```
cd viewer
npm run dev          # http://127.0.0.1:5173
npm run check        # type check
npm run build        # production build
```

URL parameters (tests, screenshots, tuning) are all listed in `viewer/src/config.ts`, e.g. `time=8.5`,
`date=2026-11-04`, `weather=clear`, `roads=1` (draw the road network), `telemetry=0`.

## Measuring

Performance comes first: an average laptop (Intel iGPU, Firefox) must run it smoothly.

- `npm run bench` (`-- --laptop`: CPU throttled, GPU timings scaled to a laptop iGPU): scripted
  camera scenarios in headless Chromium, compared with `bench/baseline.json`
- `npm run sessions`: summary of real sessions; the viewer posts telemetry to the dev server
  (`data/metrics/sessions.jsonl`)
- `npm run junctions`, then `uv run python qa_junctions.py` in `pipeline/`: checks every road junction
  as the simulation builds it (geometry and traffic lights) and draws a contact sheet

Measure on finished builds only.

## Data licences

LiDAR © Paradigm / UrbIS (CC-BY 4.0) · Map data © OpenStreetMap contributors (ODbL) · Street-level
imagery © Mapillary contributors (CC-BY-SA) · Timetable STIB-MIVB open data.
