// Parking places (pipeline parking.py: kerbs and car parks where the LiDAR saw parked cars in 2021, public
// garages from OSM) and who stands in them. Garage places are never drawn: cars drive in and are gone.
// Positions in the traffic group's frame (x east, z south); heading in that frame.

import { rng } from './rng';

export interface ParkingData { spots: [number, number, number, number?][] }   // x, y (voxels), heading (rad, x east / y north), kind
export const GARAGE = 2;   // kind: 0 kerb, 1 car park, 2 garage

export class Parking {
  readonly x: Float32Array; readonly z: Float32Array; readonly heading: Float32Array;
  readonly hidden: Uint8Array;   // 1: in a garage (not drawn)
  private owner: (unknown | null)[];
  private grid = new Map<number, number[]>();   // cell (i, j) -> places; key (i + 32768) * 65536 + (j + 32768)
  private static CELL = 40;
  private static key(i: number, j: number) { return (i + 32768) * 65536 + (j + 32768); }

  constructor(data: ParkingData, vs: number) {
    const n = data.spots.length;
    this.x = new Float32Array(n); this.z = new Float32Array(n); this.heading = new Float32Array(n);
    this.owner = new Array(n).fill(null);
    this.hidden = new Uint8Array(n);
    data.spots.forEach(([x, y, h, kind], i) => {
      this.hidden[i] = kind === GARAGE ? 1 : 0;
      this.x[i] = x * vs; this.z[i] = -y * vs;
      this.heading[i] = h;   // (x east / y north) = scene rotation about y for a model pointing +x
      const key = Parking.key(Math.floor(this.x[i] / Parking.CELL), Math.floor(this.z[i] / Parking.CELL));
      if (!this.grid.has(key)) this.grid.set(key, []);
      this.grid.get(key)!.push(i);
    });
  }

  get size() { return this.x.length; }
  /** cars of nobody we simulate (second cars, visitors, people from outside): a fixed random share of the
   *  places is taken by them, so the kerbs are about as full as real ones; they never leave */
  static readonly ANON = { anon: true };
  fillAnon(share: number, seed = 7) {
    const r = rng(seed);
    for (let i = 0; i < this.size; i++) if (r() < share && this.owner[i] === null) this.owner[i] = Parking.ANON;
  }

  /** nearest free place within max metres of (x, z), or -1. Rings of cells outward from the nearest: stops
   *  as soon as no cell further out can hold a nearer place (kerbs are full: most places are taken) */
  nearestFree(x: number, z: number, max = 150) {
    const C = Parking.CELL, R = Math.ceil(max / C), cx = Math.floor(x / C), cz = Math.floor(z / C);
    let best = -1, bd = max * max;
    for (let r = 0; r <= R; r++) {
      // every place in ring r is at least (r - 1) cells away
      const near = Math.max(0, r - 1) * C;
      if (near * near >= bd) break;
      for (let i = cx - r; i <= cx + r; i++)
        for (let j = cz - r; j <= cz + r; j++) {
          if (Math.max(Math.abs(i - cx), Math.abs(j - cz)) !== r) continue;   // (ring only)
          const cell = this.grid.get(Parking.key(i, j));
          if (!cell) continue;
          for (const k of cell) {
            if (this.owner[k] !== null) continue;
            const d = (this.x[k] - x) ** 2 + (this.z[k] - z) ** 2;
            if (d < bd) { bd = d; best = k; }
          }
        }
    }
    return best;
  }
  /** every place within r metres of (x, z), with who stands there (null: free) */
  /** places to draw within r of (x, z) (not the ones in garages) */
  forEachNear(x: number, z: number, r: number, fn: (k: number, owner: unknown | null) => void) {
    const C = Parking.CELL, n = Math.ceil(r / C), cx = Math.floor(x / C), cz = Math.floor(z / C), r2 = r * r;
    for (let i = cx - n; i <= cx + n; i++)
      for (let j = cz - n; j <= cz + n; j++)
        for (const k of this.grid.get(Parking.key(i, j)) ?? [])
          if (!this.hidden[k] && (this.x[k] - x) ** 2 + (this.z[k] - z) ** 2 < r2) fn(k, this.owner[k]);
  }
  take(i: number, who: unknown) { this.owner[i] = who; }
  free(i: number) { if (i >= 0) this.owner[i] = null; }
  clear() { for (let i = 0; i < this.size; i++) if (this.owner[i] !== Parking.ANON) this.owner[i] = null; }
}
