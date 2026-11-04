// Basic traffic: vehicles drive along the lanes at the speed limit and pick a random next lane at
// every intersection (no U-turns unless it is a dead end). No interaction: they pass through each other.
// Cars on roads, trams on tram tracks, trains on rail. All drawn as boxes in one instanced mesh.
import * as THREE from 'three/webgpu';
import { KIND, type Lane } from './roads';

interface Path { pts: Float32Array; cum: Float32Array; length: number }

interface Vehicle {
  lane: Lane;          // current lane, or the lane being entered while on a junction curve
  path: Path;          // what it is driving on now: a lane or a junction curve
  prev: Path | null;   // previous path, for the carriages still on it
  s: number;           // distance of the front along path
  f: number;           // fraction of the speed limit this driver keeps
  seg: number;         // number of boxes (carriages)
  segLen: number; gap: number;
  size: [number, number]; // height, width
  first: number;       // first instance index
}

// length, height, width (m), boxes, gap between boxes
const TYPES = {
  car: { len: 4.3, h: 1.5, w: 1.8, seg: 1, gap: 0 },
  van: { len: 5.6, h: 2.3, w: 2.0, seg: 1, gap: 0 },
  tram: { len: 10.5, h: 3.3, w: 2.3, seg: 3, gap: 0.4 },
  train: { len: 24, h: 4.0, w: 2.9, seg: 4, gap: 0.8 },
};
const CAR_COLORS = [0xf2f2f2, 0x1b1b1d, 0x7d8288, 0xb9bdc2, 0x22324f, 0x8c1c1c, 0x3a3f45, 0xe8e2d4];

/** point (and tangent) at distance s along a path */
function sample(p: Path, s: number, out: THREE.Vector3, dir?: THREE.Vector3) {
  const { pts, cum } = p;
  s = Math.min(Math.max(s, 0), p.length);
  let lo = 0, hi = cum.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cum[m] <= s) lo = m; else hi = m; }
  const seg = cum[hi] - cum[lo], t = seg > 0 ? (s - cum[lo]) / seg : 0;
  out.set(pts[lo * 3] + (pts[hi * 3] - pts[lo * 3]) * t, pts[lo * 3 + 1] + (pts[hi * 3 + 1] - pts[lo * 3 + 1]) * t,
    pts[lo * 3 + 2] + (pts[hi * 3 + 2] - pts[lo * 3 + 2]) * t);
  if (dir) dir.set(pts[hi * 3] - pts[lo * 3], pts[hi * 3 + 1] - pts[lo * 3 + 1], pts[hi * 3 + 2] - pts[lo * 3 + 2]).normalize();
  return out;
}

/** smooth curve from the end of one lane to the start of the next (quadratic Bézier) */
function junction(a: Lane, b: Lane): Path | null {
  const n = a.pts.length / 3;
  const p0 = new THREE.Vector3(a.pts[n * 3 - 3], a.pts[n * 3 - 2], a.pts[n * 3 - 1]);
  const p2 = new THREE.Vector3(b.pts[0], b.pts[1], b.pts[2]);
  const d = p0.distanceTo(p2);
  if (d < 0.3) return null;
  const da = new THREE.Vector3(), db = new THREE.Vector3(), tmp = new THREE.Vector3();
  sample(a, a.length - 0.1, tmp, da);
  sample(b, 0.1, tmp, db);
  const c = p0.clone().addScaledVector(da, d / 2).add(p2.clone().addScaledVector(db, -d / 2)).multiplyScalar(0.5);
  const N = 8, pts = new Float32Array((N + 1) * 3), cum = new Float32Array(N + 1);
  const q = new THREE.Vector3(), last = p0.clone();
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    q.set(0, 0, 0).addScaledVector(p0, (1 - t) ** 2).addScaledVector(c, 2 * t * (1 - t)).addScaledVector(p2, t * t);
    pts.set([q.x, q.y, q.z], i * 3);
    if (i) cum[i] = cum[i - 1] + q.distanceTo(last);
    last.copy(q);
  }
  return { pts, cum, length: cum[N] };
}

export class Traffic {
  readonly mesh: THREE.InstancedMesh;
  private vehicles: Vehicle[] = [];
  private out = new Map<number, Lane[]>();    // node -> lanes leaving it

  /** material: lit like the city (deferred.ts litMaterial); instance colours are the paint */
  constructor(private lanes: Lane[], counts = { car: 400, tram: 12, train: 3 },
    material: THREE.Material = new THREE.MeshStandardNodeMaterial({ roughness: 0.55, metalness: 0.1 })) {
    for (const l of lanes) {
      if (!this.out.has(l.from)) this.out.set(l.from, []);
      this.out.get(l.from)!.push(l);
    }
    const byKind = (k: number) => lanes.filter((l) => l.kind === k && l.length > 5);
    const plan: [keyof typeof TYPES, Lane[], number][] = [
      ['car', byKind(KIND.road), counts.car], ['tram', byKind(KIND.tram), counts.tram], ['train', byKind(KIND.rail), counts.train]];
    let instances = 0;
    for (const [type, pool, n] of plan) {
      if (!pool.length) continue;
      const total = pool.reduce((s, l) => s + l.length, 0);
      for (let i = 0; i < n; i++) {
        // random lane, weighted by length, random spot on it
        let r = Math.random() * total, lane = pool[0];
        for (const l of pool) { r -= l.length; if (r <= 0) { lane = l; break; } }
        const t = type === 'car' && Math.random() < 0.12 ? TYPES.van : TYPES[type];
        this.vehicles.push({ lane, path: lane, prev: null, s: Math.random() * lane.length, f: 0.75 + 0.25 * Math.random(),
          seg: t.seg, segLen: t.len, gap: t.gap, size: [t.h, t.w], first: instances });
        instances += t.seg;
      }
    }
    this.mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), material, Math.max(1, instances));
    this.mesh.frustumCulled = false;
    const c = new THREE.Color();
    for (const v of this.vehicles) {
      const kind = v.lane.kind;
      c.set(kind === KIND.road ? CAR_COLORS[Math.floor(Math.random() * CAR_COLORS.length)] : kind === KIND.tram ? 0xd9dad4 : 0x4a5a8c);
      for (let k = 0; k < v.seg; k++) this.mesh.setColorAt(v.first + k, c);
    }
    this.update(0);
  }

  get count() { return this.vehicles.length; }

  private nextLane(l: Lane): Lane | null {
    const opts = this.out.get(l.to)?.filter((n) => n.kind === l.kind) ?? [];
    const fwd = opts.filter((n) => n.edge !== l.edge);
    const pick = fwd.length ? fwd : opts;    // dead end: turn around
    return pick.length ? pick[Math.floor(Math.random() * pick.length)] : null;
  }

  private pos = new THREE.Vector3(); private dir = new THREE.Vector3();
  private m4 = new THREE.Matrix4(); private q = new THREE.Quaternion(); private e = new THREE.Euler(0, 0, 0, 'YZX');
  private scl = new THREE.Vector3();

  update(dt: number) {
    for (const v of this.vehicles) {
      v.s += v.lane.speed * v.f * dt;
      while (v.s > v.path.length) {
        v.s -= v.path.length;
        if (v.path !== v.lane) { v.prev = v.path; v.path = v.lane; continue; }   // junction curve done
        const next = this.nextLane(v.lane);
        if (!next) {   // one-way out of the region: start again somewhere else on the same kind of track
          const pool = this.lanes.filter((l) => l.kind === v.lane.kind && l.length > 5);
          v.lane = v.path = pool[Math.floor(Math.random() * pool.length)];
          v.prev = null; v.s = 0;
          break;
        }
        const j = junction(v.lane, next);
        v.prev = v.lane;
        v.lane = next;
        v.path = j ?? next;
      }
      // boxes: carriage k's centre trails the front by (k + 0.5) lengths (+ gaps)
      for (let k = 0; k < v.seg; k++) {
        let s = v.s - (k + 0.5) * v.segLen - k * v.gap, p: Path = v.path;
        if (s < 0 && v.prev) { s += v.prev.length; p = v.prev; }
        sample(p, s, this.pos, this.dir);
        this.e.set(0, Math.atan2(-this.dir.z, this.dir.x), Math.asin(Math.max(-1, Math.min(1, this.dir.y))));
        this.q.setFromEuler(this.e);
        this.pos.y += v.size[0] / 2;
        this.m4.compose(this.pos, this.q, this.scl.set(v.segLen, v.size[0], v.size[1]));
        this.mesh.setMatrixAt(v.first + k, this.m4);
      }
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}
