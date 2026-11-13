// Where people live and go (pipeline places.py) and the tram / bus stops (transit.py), indexed for the
// questions the day plans ask: a home weighted by its floors, a destination of some kinds (anywhere, or
// one of the nearest), is it open, the nearest stop. Positions are converted once, at load, to scene metres
// in the traffic group's frame (x east, z south), like everything the simulation works with.
import type { WalkSpot } from './pedestrians';
import { H } from './clock';

/** places.json */
interface PlacesData {
  cats: string[]; streets: string[]; names: string[];
  homes: [number, number, number, number][];   // x, y (voxels), floors, street
  pois: [number, number, number, number, number?][];    // x, y (voxels), category, name, opening week
  weeks?: number[][];                          // 7 day masks (Monday first): bit h = open in hour h
}

/** a place in a day plan */
export type Where = { kind: 'home' } | { kind: 'poi'; i: number } | { kind: 'edge' } | { kind: 'house'; i: number }
  | { kind: 'park'; spot: WalkSpot } | { kind: 'stop'; i: number };

const CELL = 200;   // m: grid cell for destination searches

export class PlaceIndex {
  readonly homeX: Float32Array; readonly homeZ: Float32Array;
  readonly poiX: Float32Array; readonly poiZ: Float32Array;
  private stopXZ: Float32Array = new Float32Array(0);
  private stopNames: string[] = [];
  private homeCum: Float64Array;
  private byCat: number[][];
  private pools = new Map<string, number[]>();
  private grids = new Map<string, Map<number, number[]>>();
  private nearCache = new Map<string, number[]>();
  private stopCache = new Map<number, number>();

  constructor(readonly data: PlacesData, vs: number) {
    const { homes, pois } = data;
    this.homeX = Float32Array.from(homes, (h) => h[0] * vs); this.homeZ = Float32Array.from(homes, (h) => -h[1] * vs);
    this.poiX = Float32Array.from(pois, (p) => p[0] * vs); this.poiZ = Float32Array.from(pois, (p) => -p[1] * vs);
    let t = 0;
    this.homeCum = Float64Array.from(homes, (h) => (t += h[2]));
    this.byCat = data.cats.map((_, c) => pois.flatMap((p, i) => (p[2] === c ? [i] : [])));
  }

  get homeCount() { return this.data.homes.length; }

  /** tram / bus stops (transit.json: x, y in voxels, name) */
  setStops(stops: [number, number, string][], vs: number) {
    this.stopXZ = Float32Array.from(stops.flatMap(([x, y]) => [x * vs, -y * vs]));
    this.stopNames = stops.map((s) => s[2]);
    this.stopCache.clear();
  }

  /** a home, weighted by its floors */
  pickHome(r: () => number) {
    const c = this.homeCum, x = r() * c[c.length - 1];
    let lo = 0, hi = c.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (c[mid] >= x) hi = mid; else lo = mid + 1; }
    return lo;
  }

  /** a destination of these categories anywhere in the area (none: the region edge) */
  poi(r: () => number, ...cats: string[]): Where {
    const pool = this.poolOf(cats);
    return pool.length ? { kind: 'poi', i: pool[Math.floor(r() * pool.length)] } : { kind: 'edge' };
  }

  /** a destination of these categories within max m of a home: one of the nearest few */
  near(home: number, r: () => number, max: number, ...cats: string[]): Where | null {
    // the nearest few per 25 m cell of homes and category set, cached (neighbours share their shops and
    // schools; every later day is cheap)
    const hx = Math.floor(this.homeX[home] / 25), hz = Math.floor(this.homeZ[home] / 25);
    const key = `${hx},${hz}|${max}|${cats.join(',')}`;
    let pool = this.nearCache.get(key);
    if (!pool) {
      const x = hx * 25 + 12.5, z = hz * 25 + 12.5;
      // rings of grid cells outward; stop once the 6 nearest are closer than anything further out can be
      const grid = this.gridOf(cats), R = Math.ceil(max / CELL);
      const cx = Math.floor(x / CELL), cz = Math.floor(z / CELL), found: { i: number; d: number }[] = [];
      for (let ring = 0; ring <= R; ring++) {
        for (let gx = cx - ring; gx <= cx + ring; gx++)
          for (let gz = cz - ring; gz <= cz + ring; gz++) {
            if (Math.max(Math.abs(gx - cx), Math.abs(gz - cz)) !== ring) continue;
            for (const i of grid.get(gx * 65536 + gz) ?? []) {
              const d = Math.hypot(this.poiX[i] - x, this.poiZ[i] - z);
              if (d < max) found.push({ i, d });
            }
          }
        if (found.length >= 6) {
          found.sort((a, b) => a.d - b.d);
          if (found[5].d <= ring * CELL) break;   // ring + 1 is at least `ring` cells away
        }
      }
      pool = found.sort((a, b) => a.d - b.d).slice(0, 6).map((f) => f.i);
      this.nearCache.set(key, pool);
    }
    return pool.length ? { kind: 'poi', i: pool[Math.floor(r() * pool.length)] } : null;
  }

  /** is destination i open at t (seconds of the day; may run past midnight) on weekday wd (0 Monday) */
  isOpen(i: number, wd: number, t: number) {
    const w = this.data.weeks?.[this.data.pois[i][4] ?? -1];
    if (!w) return true;
    const day = (wd + Math.floor(t / (24 * H))) % 7, h = Math.floor((t % (24 * H)) / H);
    return ((w[day] >> h) & 1) === 1;
  }

  /** the nearest destination of the same kind as i, within reach m, open at t (-1: none) */
  openNear(i: number, wd: number, t: number, reach: number) {
    const grid = this.gridOf([this.data.cats[this.data.pois[i][2]]]), R = Math.ceil(reach / CELL);
    const x = this.poiX[i], z = this.poiZ[i], cx = Math.floor(x / CELL), cz = Math.floor(z / CELL);
    let best = -1, bd = Infinity;
    for (let gx = cx - R; gx <= cx + R; gx++)
      for (let gz = cz - R; gz <= cz + R; gz++)
        for (const j of grid.get(gx * 65536 + gz) ?? []) {
          const d = Math.hypot(this.poiX[j] - x, this.poiZ[j] - z);
          if (d < bd && d < reach && this.isOpen(j, wd, t)) { bd = d; best = j; }
        }
    return best;
  }

  /** another home 200-500 m from this one (a walk around the block), or -1 */
  houseNear(home: number, r: () => number) {
    for (let t = 0; t < 30; t++) {
      const i = Math.floor(r() * this.homeCount);
      const d = Math.hypot(this.homeX[i] - this.homeX[home], this.homeZ[i] - this.homeZ[home]);
      if (d > 200 && d < 500) return i;
    }
    return -1;
  }

  /** nearest stop to a home within 700 m (-1: none) */
  stopNear(home: number) {
    let best = this.stopCache.get(home);
    if (best === undefined) {
      best = -1;
      let bd = 700 ** 2;
      for (let i = 0; i < this.stopNames.length; i++) {
        const d = (this.stopXZ[i * 2] - this.homeX[home]) ** 2 + (this.stopXZ[i * 2 + 1] - this.homeZ[home]) ** 2;
        if (d < bd) { bd = d; best = i; }
      }
      this.stopCache.set(home, best);
    }
    return best;
  }

  /** scene position (x, z) of a home / destination / stop */
  homePos(i: number): [number, number] { return [this.homeX[i], this.homeZ[i]]; }
  poiPos(i: number): [number, number] { return [this.poiX[i], this.poiZ[i]]; }
  stopPos(i: number): [number, number] { return [this.stopXZ[i * 2], this.stopXZ[i * 2 + 1]]; }

  stopName(i: number) { return this.stopNames[i]; }
  /** a destination's name, else its kind */
  poiLabel(i: number) { const p = this.data.pois[i]; return this.data.names[p[3]] || this.data.cats[p[2]]; }
  /** street of a home (bilingual names: the French half) */
  homeStreet(i: number) { return (this.data.streets[this.data.homes[i][3]] || 'a street nearby').split(' - ')[0]; }

  /** all destinations of these categories (cached per category set) */
  private poolOf(cats: string[]) {
    const key = cats.join(',');
    let p = this.pools.get(key);
    if (!p) { p = cats.flatMap((c) => this.byCat[this.data.cats.indexOf(c)] ?? []); this.pools.set(key, p); }
    return p;
  }
  /** destinations of these categories by grid cell (key gx * 65536 + gz) */
  private gridOf(cats: string[]) {
    const key = cats.join(',');
    let g = this.grids.get(key);
    if (!g) {
      g = new Map();
      for (const i of this.poolOf(cats)) {
        const k = Math.floor(this.poiX[i] / CELL) * 65536 + Math.floor(this.poiZ[i] / CELL);
        let l = g.get(k);
        if (!l) g.set(k, (l = []));
        l.push(i);
      }
      this.grids.set(key, g);
    }
    return g;
  }
}
