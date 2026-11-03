// CPU-side access to the full-resolution (L0) voxels: occupancy for walking, records for picking.
// Tiles are fetched on demand (the browser cache usually has them from rendering) and kept in a
// small LRU. Voxel space: x east, y north, z up, in L0 voxels; see toVoxel() for the scene mapping.
import * as THREE from 'three/webgpu';
import type { LevelInfo, TileInfo } from './lod';

const FLAG_APRON = 2, FLAG_PHOTO = 8;
const MAX_TILES = 48;   // ~1 MB occupancy each

export interface VoxelRecord { x: number; y: number; z: number; r: number; g: number; b: number; cls: number; flags: number }

interface Tile { occ: Uint8Array; photo: Uint8Array; raw: Uint8Array; index: Map<number, number> | null; used: number }

export class Voxels {
  readonly vs: number;
  readonly nx: number; readonly ny: number; readonly nz: number;
  private infos = new Map<number, TileInfo>();
  private tiles = new Map<number, Tile>();
  private pending = new Map<number, Promise<Tile | null>>();
  private clock = 0;

  constructor(private base: string, private ch: number, level: LevelInfo, private world: THREE.Object3D, private query = '') {
    this.vs = level.voxelSize;
    [this.nx, this.ny, this.nz] = level.size;
    for (const t of level.tiles) this.infos.set(this.key(t.tx, t.ty), t);
  }

  private key(tx: number, ty: number) { return ty * 4096 + tx; }

  /** scene point -> continuous voxel coordinates */
  toVoxel(p: THREE.Vector3, out = new THREE.Vector3()) {
    const w = this.world.position;
    return out.set((p.x - w.x) / this.vs, -(p.z - w.z) / this.vs, (p.y - w.y) / this.vs);
  }
  /** continuous voxel coordinates -> scene point */
  toScene(v: THREE.Vector3, out = new THREE.Vector3()) {
    const w = this.world.position;
    return out.set(v.x * this.vs + w.x, v.z * this.vs + w.y, -v.y * this.vs + w.z);
  }

  /** true once the tile containing voxel column (x, y) is available (or does not exist) */
  has(x: number, y: number) {
    const k = this.key(Math.floor(x / this.ch), Math.floor(y / this.ch));
    return !this.infos.has(k) || this.tiles.has(k);
  }

  /** fetch the tiles within r voxels of (x, y) */
  preload(x: number, y: number, r: number): Promise<unknown> {
    const out: Promise<unknown>[] = [];
    for (let ty = Math.floor((y - r) / this.ch); ty <= Math.floor((y + r) / this.ch); ty++)
      for (let tx = Math.floor((x - r) / this.ch); tx <= Math.floor((x + r) / this.ch); tx++) out.push(this.load(tx, ty));
    return Promise.all(out);
  }

  load(tx: number, ty: number): Promise<Tile | null> {
    const k = this.key(tx, ty);
    const t = this.tiles.get(k);
    if (t) { t.used = ++this.clock; return Promise.resolve(t); }
    const info = this.infos.get(k);
    if (!info) return Promise.resolve(null);
    let p = this.pending.get(k);
    if (!p) {
      p = fetch(`${this.base}/${info.file}${this.query}`).then((r) => r.arrayBuffer()).then((buf) => {
        const raw = new Uint8Array(buf);
        const ch = this.ch, occ = new Uint8Array(ch * ch * this.nz), photo = new Uint8Array(ch * ch);
        for (let o = 0; o < raw.length; o += 8) {
          if (raw[o + 7] & FLAG_APRON) continue;
          const col = (raw[o + 1] - 1) * ch + raw[o] - 1;
          occ[raw[o + 2] * ch * ch + col] = raw[o + 6] || 255; // LiDAR class, 0 never used
          if (raw[o + 7] & FLAG_PHOTO) photo[col] = 1;
        }
        const tile: Tile = { occ, photo, raw, index: null, used: ++this.clock };
        this.tiles.set(k, tile);
        this.pending.delete(k);
        if (this.tiles.size > MAX_TILES) {
          const old = [...this.tiles.entries()].sort((a, b) => a[1].used - b[1].used)[0];
          this.tiles.delete(old[0]);
        }
        return tile;
      });
      this.pending.set(k, p);
    }
    return p;
  }

  /** LiDAR class of the voxel (0 = empty or tile not loaded) */
  cls(x: number, y: number, z: number): number {
    if (z < 0 || z >= this.nz || x < 0 || y < 0) return 0;
    const tx = Math.floor(x / this.ch), ty = Math.floor(y / this.ch);
    const t = this.tiles.get(this.key(tx, ty));
    if (!t) return 0;
    return t.occ[(z * this.ch + (y - ty * this.ch)) * this.ch + (x - tx * this.ch)];
  }

  /** does the column hold a voxel coloured from street-level photos (i.e. seen from a street)? */
  photo(x: number, y: number): boolean {
    const tx = Math.floor(x / this.ch), ty = Math.floor(y / this.ch);
    const t = this.tiles.get(this.key(tx, ty));
    return !!t && t.photo[(y - ty * this.ch) * this.ch + (x - tx * this.ch)] === 1;
  }

  /** full record of a voxel in a loaded tile */
  record(x: number, y: number, z: number): VoxelRecord | null {
    const tx = Math.floor(x / this.ch), ty = Math.floor(y / this.ch);
    const t = this.tiles.get(this.key(tx, ty));
    if (!t) return null;
    if (!t.index) {
      t.index = new Map();
      for (let o = 0; o < t.raw.length; o += 8)
        if (!(t.raw[o + 7] & FLAG_APRON)) t.index.set((t.raw[o + 2] * 128 + t.raw[o + 1]) * 128 + t.raw[o], o);
    }
    const o = t.index.get((z * 128 + y - ty * this.ch + 1) * 128 + x - tx * this.ch + 1);
    if (o === undefined) return null;
    const r = t.raw;
    return { x, y, z, r: r[o + 3], g: r[o + 4], b: r[o + 5], cls: r[o + 6], flags: r[o + 7] };
  }

  /** First solid voxel along a scene-space ray (loads tiles on the way). Returns the voxel and the face normal. */
  async raycast(origin: THREE.Vector3, dir: THREE.Vector3, maxDist = 4000, skip?: (cls: number) => boolean) {
    const o = this.toVoxel(origin);
    const d = new THREE.Vector3(dir.x, -dir.z, dir.y).normalize();
    // clip the ray to the voxel grid
    let t0 = 0, t1 = maxDist / this.vs;
    const lo = [0, 0, 0], hi = [this.nx, this.ny, this.nz], oa = [o.x, o.y, o.z], da = [d.x, d.y, d.z];
    for (let i = 0; i < 3; i++) {
      if (Math.abs(da[i]) < 1e-12) { if (oa[i] < lo[i] || oa[i] >= hi[i]) return null; continue; }
      let a = (lo[i] - oa[i]) / da[i], b = (hi[i] - oa[i]) / da[i];
      if (a > b) [a, b] = [b, a];
      t0 = Math.max(t0, a); t1 = Math.min(t1, b);
    }
    if (t0 >= t1) return null;
    const p = [0, 1, 2].map((i) => Math.min(hi[i] - 1, Math.max(0, Math.floor(oa[i] + da[i] * (t0 + 1e-6)))));
    const step = da.map((v) => (v > 0 ? 1 : -1));
    const tDelta = da.map((v) => (v === 0 ? Infinity : Math.abs(1 / v)));
    const tMax = da.map((v, i) => (v === 0 ? Infinity : t0 + ((v > 0 ? p[i] + 1 : p[i]) - (oa[i] + v * t0)) / v));
    let axis = -1;
    for (let n = 0; n < 20000; n++) {
      if (!this.has(p[0], p[1])) await this.load(Math.floor(p[0] / this.ch), Math.floor(p[1] / this.ch));
      const c = this.cls(p[0], p[1], p[2]);
      if (c && !skip?.(c)) {
        const normal = [0, 0, 0];
        if (axis >= 0) normal[axis] = -step[axis];
        return { voxel: new THREE.Vector3(p[0], p[1], p[2]), normal: new THREE.Vector3(...normal), cls: c };
      }
      axis = tMax[0] < tMax[1] ? (tMax[0] < tMax[2] ? 0 : 2) : (tMax[1] < tMax[2] ? 1 : 2);
      if (tMax[axis] > t1) return null;
      p[axis] += step[axis];
      tMax[axis] += tDelta[axis];
    }
    return null;
  }
}
