# bxl vice city

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

Everything the viewer downloads is kept small (`pipeline/common.py`), since it is streamed to every visitor:
tiles are compact (positions delta-coded, a 256-colour palette per tile, gzipped; ~0.4 GB for Schaerbeek)
and packed into archives of 16 x 16 tiles (~100 files, read with HTTP range requests); the manifest lists
tiles as columns, road polylines are delta-coded, and the timetable stores each trip as a stop pattern, a
timing and a start time. A first visit loads about 13 MB, mostly the tiles in view.

## Viewer

```
cd viewer
npm run dev          # http://127.0.0.1:5173
npm run check        # type check
npm run build        # production build
```

Keys: `?` shows the controls, `D` frame and tile statistics. URL parameters (tests, screenshots, tuning)
are all listed in `viewer/src/config.ts`, e.g. `time=8.5`, `date=2026-11-04`, `weather=clear`, `roads=1`
(draw the road network), `telemetry=0`, `tier=low` (phone budgets), `cache=0`.

Built to hold up on any device: fetched tiles are kept per build in the browser's Cache Storage (a repeat
visit downloads no tiles); phones and weak machines get smaller GPU memory budgets (`src/tier.ts`) and the
level of detail coarsens when a view needs more than the budget; a lost GPU context reloads at the same view.

The data files (`viewer/public/`: tiles, roads, timetable, ...) can live on another host than the app:
build with `VITE_DATA_URL=https://.../` or open the page with `?data=https://.../` (the host must allow
cross-origin requests).

## Publishing

The site is static: the app and the data files, no server code. `npm run deploy` (in `viewer/`) builds it and
puts it on the `gh-pages` branch as a single commit replacing the previous one; `git push -f origin gh-pages`
publishes it (GitHub Pages: deploy from the `gh-pages` branch). The tiles are read with HTTP range requests,
so any static host that supports them works; the data can also live on another host (`VITE_DATA_URL`).

## Measuring

Performance comes first: an average laptop (Intel iGPU, Firefox) must run it smoothly.

- `npm run bench` (`-- --laptop`: CPU throttled, GPU timings scaled to a laptop iGPU): scripted
  camera scenarios in headless Chromium, compared with `bench/baseline.json`
- `npm run sessions`: summary of real sessions; the viewer posts telemetry to the dev server
  (`data/metrics/sessions.jsonl`)
- `npm run noisecal`: runs the simulation headless at a few hours of the day and writes the noise levels
  per street type that the noise map starts from (`public/noise/<roi>.json`)

Measure on finished builds only.

## Licence

The code in this repository (pipeline and viewer) is released under the [MIT licence](LICENSE).

The data the pipeline downloads is **not** part of this repository and is not covered by that licence: it
stays under its own terms, listed below. If you build a region and publish the result (tiles, road and
place files, screenshots, a hosted viewer), keep these credits with it. The viewer shows them in its
bottom-right corner.

## Data sources and licences

| source | used for | licence | credit |
|---|---|---|---|
| [Brussels airborne LiDAR point cloud 2021](https://geobru-geonetwork.irisnet.be/geonetwork/srv/api/records/ff1124e1-424e-11ee-b156-00090ffe0001) (Paradigm / urban.brussels, via [datastore.brussels](https://datastore.brussels)) | the voxel city, ground heights, parked cars | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | LiDAR © Paradigm / UrbIS, datastore.brussels, CC BY 4.0 (voxelised and modified) |
| [OpenStreetMap](https://www.openstreetmap.org) (via the [Overpass API](https://overpass-api.de)) | roads, tram tracks, footpaths, traffic lights, crossings, speed bumps, buildings with addresses, shops and other places, car parks, the Schaerbeek boundary | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/) | © OpenStreetMap contributors, [openstreetmap.org/copyright](https://www.openstreetmap.org/copyright) |
| [Mapillary](https://www.mapillary.com) (API) | 360° street-level images (façade colours, semantic labels), street-lamp map features | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) ([terms](https://www.mapillary.com/terms)) | imagery © Mapillary contributors, CC BY-SA |
| [STIB-MIVB open data](https://data.stib-mivb.brussels) (GTFS) | tram and bus lines, stops and timetable | STIB-MIVB open data licence ([terms](https://data.stib-mivb.brussels/terms/terms-and-conditions/)) | timetable: STIB-MIVB open data (used in modified form: lines clipped to the region, trips simulated) |
| [Open-Meteo](https://open-meteo.com) (API, fetched by the viewer) | live weather (clouds, rain) | data [CC BY 4.0](https://open-meteo.com/en/licence); the free API is for non-commercial use | weather data by Open-Meteo.com |

What follows from these licences when publishing a built city (not the code):

- **Share-alike.** Tiles coloured from Mapillary imagery are an adaptation of CC BY-SA material, and the
  road, place and parking files are derived from OpenStreetMap (ODbL). Publish those under CC BY-SA 4.0 /
  ODbL respectively, with the credits above.
- **Changes.** CC BY 4.0 (LiDAR, Open-Meteo) and the STIB licence ask you to say that the data was
  modified: the LiDAR is voxelised, filtered and recoloured; the timetable is clipped and simulated.
- **Weather.** A commercial deployment needs an Open-Meteo API subscription; otherwise switch live
  weather off (`weather=clear`).
- `viewer/public/noise/schaerbeek.json` (the only data file in the repository) is the simulation's own
  output: traffic noise per street class, derived from the OpenStreetMap road network (ODbL).
