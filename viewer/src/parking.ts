// Parking places (pipeline parking.py: kerbs and car parks where the LiDAR saw parked cars in 2021, public
// garages from OSM) and who stands in them. Garage places are never drawn: cars drive in and are gone.
// Positions in the traffic group's frame (x east, z south); heading in that frame.

import { rng } from './rng';
import { PointGrid } from './grid';
import type { Agent } from './agents';

interface ParkingData { spots: [number, number, number, number?][] }   // x, y (voxels), heading (rad, x east / y north), kind
const GARAGE = 2;   // kind: 0 kerb, 1 car park, 2 garage

export class Parking {
  readonly x: Float32Array; readonly z: Float32Array; readonly heading: Float32Array;
  readonly hidden: Uint8Array;   // 1: in a garage (not drawn)
  private owner: (Agent | typeof Parking.ANON | null)[];
  private grid = new PointGrid<number>(40);   // places
  private drawn = new PointGrid<number>(40);  // places that are drawn (not in garages)

  constructor(data: ParkingData, vs: number) {
    const n = data.spots.length;
    this.x = new Float32Array(n); this.z = new Float32Array(n); this.heading = new Float32Array(n);
    this.owner = new Array(n).fill(null);
    this.hidden = new Uint8Array(n);
    data.spots.forEach(([x, y, h, kind], i) => {
      this.hidden[i] = kind === GARAGE ? 1 : 0;
      this.x[i] = x * vs; this.z[i] = -y * vs;
      this.heading[i] = h;   // (x east / y north) = scene rotation about y for a model pointing +x
      this.grid.add(this.x[i], this.z[i], i);
      if (!this.hidden[i]) this.drawn.add(this.x[i], this.z[i], i);
    });
  }

  get size() { return this.x.length; }
  /** cars of nobody we simulate (second cars, visitors, people from outside): a fixed random share of the
   *  places is taken by them, so the kerbs are about as full as real ones; they never leave */
  static readonly ANON: unique symbol = Symbol('anonymous car');
  fillAnon(share: number, seed = 7) {
    const r = rng(seed);
    for (let i = 0; i < this.size; i++) if (r() < share && this.owner[i] === null) this.owner[i] = Parking.ANON;
  }

  /** nearest free place within max metres of (x, z), or -1 */
  nearestFree(x: number, z: number, max = 150) {
    return this.grid.nearest(x, z, max, (k) => this.owner[k] === null) ?? -1;
  }
  /** places to draw within r of (x, z) (not the ones in garages), with who stands there (null: free) */
  forEachNear(x: number, z: number, r: number, fn: (k: number, owner: Agent | typeof Parking.ANON | null) => void) {
    this.drawn.forEachWithin(x, z, r, (k) => fn(k, this.owner[k]));
  }
  take(i: number, who: Agent | typeof Parking.ANON) { this.owner[i] = who; }
  free(i: number) { if (i >= 0) this.owner[i] = null; }
  clear() { for (let i = 0; i < this.size; i++) if (this.owner[i] !== Parking.ANON) this.owner[i] = null; }
}
