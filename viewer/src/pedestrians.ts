// Pedestrians: walk along pavements (both sides of every street, where the LiDAR facades are: pipeline
// roads.py), round corners and across streets at intersections. They wait for a gap before crossing;
// while someone crosses, cars may not enter that arm of the intersection (traffic.ts). Drawn as small
// voxel people (0.125 m voxels) whose legs and arms swing in the vertex shader.
import * as THREE from 'three/webgpu';
import { crosswalkKey, KIND, sceneLine, type RoadNetwork } from './roads';
import type { Traffic } from './traffic';
import type { Agent } from './agents';
import { cumulative, pointAt } from './polyline';
import { MinHeap } from './heap';
import { PointGrid } from './grid';

const DEFAULT_PAVEMENT = [6, 6, 6, 7, 8.5, 9.5, 10];   // m from the centreline, by road rank (no facade found)
const RED_PATIENCE = 120;   // s at a red light before walking anyway (when no car is near)
const CROSS_PATIENCE = 25;   // s waiting for a gap before crossing anyway (cars then stop)

interface Seg { pts: Float32Array; cum: Float32Array; len: number; a: number; b: number; cross: number | null; park?: boolean }   // cross: crosswalk key (roads.ts crosswalkKey)
interface Walker {
  agent: Agent; segs: { seg: Seg; fwd: boolean }[]; k: number; s: number; v: number; dist: number;
  wait: number; crossing: number | null; idx: number; end: number;   // end: m to walk on the last segment
}
export interface WalkSpot { seg: Seg; s: number }

function polyline(pts: number[]) {
  const cum = cumulative(pts, true);   // (walking distance: horizontal)
  return { pts: new Float32Array(pts), cum, len: cum[cum.length - 1] };
}
const at = (seg: Seg, s: number, out: THREE.Vector3) => pointAt(seg.pts, seg.cum, s, out);

// --- voxel person (0.125 m), x forward, y up, z across; limb: 0 body, 1/2 legs, 3/4 arms -----------------
const PV = 0.125;
function personGeometry() {
  // [x0, x1, y0, y1, z0, z1, material, limb] in voxels; materials: 1 shirt, 2 skin, 3 trousers, 4 hair/shoes
  const boxes: number[][] = [
    [0, 2, 0, 1, 0, 1, 4, 1], [0, 2, 0, 1, 2, 3, 4, 2],         // shoes
    [0, 2, 1, 7, 0, 1, 3, 1], [0, 2, 1, 7, 2, 3, 3, 2],         // legs
    [0, 2, 7, 12, 0, 3, 1, 0],                                  // torso
    [0, 2, 7, 12, -1, 0, 1, 3], [0, 2, 7, 12, 3, 4, 1, 4],      // arms
    [0, 2, 12, 14, 0, 3, 2, 0], [0, 2, 14, 15, 0, 3, 4, 0],     // head, hair
  ];
  const pos: number[] = [], nrm: number[] = [], mat: number[] = [], limb: number[] = [], idx: number[] = [];
  const F: [number[], number[], number[], number[]][] = [   // normal, then 4 corners as (x,y,z) selectors 0/1
    [[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1]], [[-1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 0]], [[0, -1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1]],
    [[0, 0, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1]], [[0, 0, -1], [0, 0, 0], [0, 1, 0], [1, 0, 0]],
  ];
  for (const [x0, x1, y0, y1, z0, z1, m, l] of boxes) {
    const X = [x0, x1], Y = [y0, y1], Z = [z0, z1];
    for (const [n, c0, c1, c2] of F) {
      const base = pos.length / 3;
      const c3 = [c1[0] + c2[0] - c0[0], c1[1] + c2[1] - c0[1], c1[2] + c2[2] - c0[2]];
      for (const c of [c0, c1, c2, c3]) {
        pos.push((X[c[0]] - 1) * PV, Y[c[1]] * PV, (Z[c[2]] - 1.5) * PV);
        nrm.push(...n); mat.push(m); limb.push(l);
      }
      idx.push(base, base + 1, base + 2, base + 2, base + 1, base + 3);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  // body part (material) and limb in one attribute: WebGPU allows only 8 vertex buffers per draw
  g.setAttribute('part', new THREE.Float32BufferAttribute(mat.flatMap((m, i) => [m, limb[i]]), 2));
  g.setIndex(idx);
  return g;
}

const SHIRTS = [0x2b3a55, 0x7a1f1f, 0xd8d4c8, 0x3d5a3a, 0x1c1c1e, 0x6b6f75, 0xb0743a, 0x4a6fa5, 0x8e8a7a, 0xc9a227];
const SKIN = [0xf1c8a8, 0xd9a47e, 0xa8714f, 0x7a4a30, 0x5a3624, 0xe6b894];
const TROUSERS = [0x1f2a3a, 0x2a2a2a, 0x4a4a48, 0x3a3226, 0x6a6a70];

export class Pedestrians {
  readonly mesh: THREE.InstancedMesh;
  private segs: Seg[] = [];
  private adj = new Map<number, { seg: Seg; fwd: boolean }[]>();   // graph vertex -> segments leaving it
  private walkers: Walker[] = [];
  private byAgent = new Map<Agent, Walker>();
  private free: number[] = [];
  private busy = new Map<number, number>();     // crosswalk key -> people on it
  private grid = new PointGrid<WalkSpot>(40);   // pavement / path points, for spotNear
  private walk: THREE.InstancedBufferAttribute;   // (walk phase, heading) per person
  onArrive: ((agent: Agent) => void) | null = null;

  constructor(net: RoadNetwork, vs: number, material: THREE.Material, private traffic: Traffic | null, capacity = 1500) {
    this.net = net; this.vs = vs;
    // --- mesh: instanced people with per-instance colours and walk phase ---
    const g = personGeometry();
    // per person: shirt, skin, trousers colours (one interleaved buffer, static) and walk phase + heading
    // (one buffer, updated every frame); WebGPU allows only 8 vertex buffers per draw
    const look = new Float32Array(capacity * 9);
    const c = new THREE.Color();
    for (let i = 0; i < capacity; i++) {
      c.set(SHIRTS[i % SHIRTS.length]); look.set([c.r, c.g, c.b], i * 9);
      c.set(SKIN[(i * 7) % SKIN.length]); look.set([c.r, c.g, c.b], i * 9 + 3);
      c.set(TROUSERS[(i * 3) % TROUSERS.length]); look.set([c.r, c.g, c.b], i * 9 + 6);
    }
    const lb = new THREE.InstancedInterleavedBuffer(look, 9);
    g.setAttribute('shirt', new THREE.InterleavedBufferAttribute(lb, 3, 0));
    g.setAttribute('skin', new THREE.InterleavedBufferAttribute(lb, 3, 3));
    g.setAttribute('pants', new THREE.InterleavedBufferAttribute(lb, 3, 6));
    this.walk = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 2), 2);
    this.walk.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('walk', this.walk);
    this.mesh = new THREE.InstancedMesh(g, material, capacity);
    this.mesh.frustumCulled = false;
    const zero = new THREE.Matrix4().makeScale(0, 0, 0);
    for (let i = capacity - 1; i >= 0; i--) { this.mesh.setMatrixAt(i, zero); this.free.push(i); }
  }

  private net: RoadNetwork; private vs: number;
  /** build the walking network (pavements, corners, crossings, footpaths) in steps, a frame in between each
   *  (on a slow CPU the whole commune takes ~0.5 s: one task that long is a visible freeze) */
  async build(nextFrame: () => Promise<void>) {
    for (const _ of this.graph()) await nextFrame();
  }

  private *graph(): Generator<void> {
    const net = this.net, vs = this.vs;
    // --- pavements: both sides of every road edge, trimmed back from the intersections ---
    let vid = 0;
    const end = new Map<string, number>();        // `${edge}:${side}:${0 start|1 end}` -> graph vertex
    const vtx = (k: string) => { if (!end.has(k)) end.set(k, vid++); return end.get(k)!; };
    const sides = new Map<string, Seg>();
    for (let ei = 0; ei < net.edges.length; ei++) {
      const e = net.edges[ei];
      if (ei % 300 === 299) yield;
      if (e[2] !== KIND.road) continue;
      const raw = e[6], n = raw.length / 3;
      if (n < 2) continue;
      const flat = sceneLine(raw, vs), p = Array.from({ length: n }, (_, i) => flat.slice(i * 3, i * 3 + 3));
      const pav = e[10] ?? [0, 0];
      for (const side of [0, 1]) {   // 0 left, 1 right of the edge direction
        const w = pav[side] || DEFAULT_PAVEMENT[e[7]] || 6;
        const pts: number[] = [];
        for (let i = 0; i < n; i++) {
          const a = p[Math.max(0, i - 1)], b = p[Math.min(n - 1, i + 1)];
          let dx = b[0] - a[0], dz = b[2] - a[2];
          const l = Math.hypot(dx, dz) || 1; dx /= l; dz /= l;
          // right of travel (y up): (-dz, dx); left: (dz, -dx)
          const sx = side === 0 ? dz : -dz, sz = side === 0 ? -dx : dx;
          pts.push(p[i][0] + sx * w, p[i][1] + 0.05, p[i][2] + sz * w);
        }
        const full = polyline(pts);
        const trim = Math.min(w + 1, full.len * 0.4);
        const kept: number[] = [];
        const tmp = new THREE.Vector3();
        const seg0 = { ...full, a: 0, b: 0, cross: null } as Seg;
        for (let s = trim; s <= full.len - trim + 0.01; s += Math.max(1, (full.len - 2 * trim) / Math.ceil((full.len - 2 * trim) / 3))) {
          at(seg0, s, tmp); kept.push(tmp.x, tmp.y, tmp.z);
        }
        if (kept.length < 6) continue;
        const seg: Seg = { ...polyline(kept), a: vtx(`${ei}:${side}:0`), b: vtx(`${ei}:${side}:1`), cross: null };
        this.segs.push(seg);
        sides.set(`${ei}:${side}`, seg);
      }
    }
    yield;
    // --- intersections: corner links between neighbouring streets (round the block corner) ---
    const inc = new Map<number, { ei: number; atStart: boolean; ang: number }[]>();
    net.edges.forEach((e, ei) => {
      if (e[2] !== KIND.road) return;
      const raw = e[6], n = raw.length / 3;
      for (const atStart of [true, false]) {
        const node = atStart ? e[0] : e[1];
        const i0 = atStart ? 0 : n - 1, i1 = atStart ? Math.min(n - 1, 2) : Math.max(0, n - 3);
        const ang = Math.atan2(raw[i1 * 3 + 1] - raw[i0 * 3 + 1], raw[i1 * 3] - raw[i0 * 3]);   // leaving the node (x east, y north)
        if (!inc.has(node)) inc.set(node, []);
        inc.get(node)!.push({ ei, atStart, ang });
      }
    });
    const link = (va: number, vb: number, pa: THREE.Vector3, pb: THREE.Vector3, cross: number | null) => {
      const seg: Seg = { ...polyline([pa.x, pa.y, pa.z, pb.x, pb.y, pb.z]), a: va, b: vb, cross };
      this.segs.push(seg);
    };
    const endPoint = (ei: number, side: number, atStart: boolean) => {
      const seg = sides.get(`${ei}:${side}`);
      if (!seg) return null;
      return { v: atStart ? seg.a : seg.b, p: at(seg, atStart ? 0 : seg.len, new THREE.Vector3()) };
    };
    for (const [, list] of inc) {
      list.sort((x, y) => x.ang - y.ang);   // counter-clockwise
      for (let k = 0; k < list.length; k++) {
        const A = list[k], B = list[(k + 1) % list.length];
        // leaving the node: the left side is the edge's left (0) if it starts here, else its right (1)
        const leftOf = (x: typeof A) => (x.atStart ? 0 : 1), rightOf = (x: typeof A) => (x.atStart ? 1 : 0);
        if (list.length > 1) {   // the corner between A and the next street B (counter-clockwise)
          const pa = endPoint(A.ei, leftOf(A), A.atStart), pb = endPoint(B.ei, rightOf(B), B.atStart);
          if (pa && pb && pa.p.distanceTo(pb.p) < 40) link(pa.v, pb.v, pa.p, pb.p, null);
        }
      }
    }
    yield;
    // pavements are split where crosswalks and footpaths join them (one pass, below)
    const splits = new Map<Seg, { s: number; v: number }[]>();
    const tmp = new THREE.Vector3();
    const cut = (seg: Seg, s: number) => {   // a vertex on seg at s (its end, or a new one)
      if (s < 0.3) return seg.a;
      if (s > seg.len - 0.3) return seg.b;
      const v = vid++;
      if (!splits.has(seg)) splits.set(seg, []);
      splits.get(seg)!.push({ s, v });
      return v;
    };
    // --- crosswalks where OpenStreetMap has them (zebras, signalled, unmarked): across the street from one
    // pavement to the other. (No crossing at every corner any more: people walk to the mapped ones.)
    const nearestOn = (seg: Seg, q: THREE.Vector3) => {
      let bs = 0, bd = Infinity;
      for (let t = 0; t <= seg.len; t += 0.5) { at(seg, t, tmp); const d = (tmp.x - q.x) ** 2 + (tmp.z - q.z) ** 2; if (d < bd) { bd = d; bs = t; } }
      return { s: bs, d: Math.sqrt(bd) };
    };
    net.edges.forEach((e, ei) => {
      const xw = e[11];
      if (e[2] !== KIND.road || !xw?.length) return;
      const line = sceneLine(e[6], vs), cum = cumulative(line, true);   // (crosswalks: m along the edge, horizontally)
      xw.forEach(([m], i) => {
        const L = sides.get(`${ei}:0`), R = sides.get(`${ei}:1`);
        if (!L || !R) return;
        const q = pointAt(line, cum, m, new THREE.Vector3());
        const l = nearestOn(L, q), r = nearestOn(R, q);
        if (l.d > 25 || r.d > 25) return;   // (no pavement near: a crosswalk to nowhere)
        const vl = cut(L, l.s), vr = cut(R, r.s);
        link(vl, vr, at(L, l.s, new THREE.Vector3()), at(R, r.s, new THREE.Vector3()), crosswalkKey(ei, i));
      });
    });
    yield;
    // --- footpaths (parks, squares): their own network, linked to the pavements where they end near one ---
    if (net.paths?.length) {
      const fv = new Map<number, number>();
      const fvtx = (id: number) => { if (!fv.has(id)) fv.set(id, vid++); return fv.get(id)!; };
      const deg = new Map<number, number>();
      const paths: Seg[] = [];
      for (const [[na, nb], raw] of net.paths) {
        const pts = sceneLine(raw, vs, 0.05);
        if (pts.length < 6) continue;
        const seg: Seg = { ...polyline(pts), a: fvtx(na), b: fvtx(nb), cross: null, park: true };
        paths.push(seg);
        deg.set(seg.a, (deg.get(seg.a) ?? 0) + 1); deg.set(seg.b, (deg.get(seg.b) ?? 0) + 1);
      }
      // nearest pavement points (sampled) for linking path ends
      const paveGrid = new PointGrid<{ seg: Seg; s: number; x: number; y: number; z: number }>(15);   // (a commune: ~300k points)
      let nY = 0;
      for (const seg of this.segs) if (seg.cross === null && seg.pts.length > 6) {
        if (++nY % 1500 === 0) yield;
        for (let s = 0.5; s < seg.len - 0.5; s += 2) {
          at(seg, s, tmp);
          paveGrid.add(tmp.x, tmp.z, { seg, s, x: tmp.x, y: tmp.y, z: tmp.z });
        }
      }
      for (const seg of paths) {
        if (++nY % 1500 === 0) yield;
        for (const [v, s] of [[seg.a, 0], [seg.b, seg.len]] as const) {
          if ((deg.get(v) ?? 0) > 1) continue;   // only dangling ends
          at(seg, s, tmp);
          const best = paveGrid.nearest(tmp.x, tmp.z, 15);
          if (!best) continue;
          const nv = cut(best.seg, best.s);
          link(v, nv, tmp.clone(), new THREE.Vector3(best.x, best.y, best.z), null);
        }
        this.segs.push(seg);
      }
    }
    // split pavements where crosswalks and paths join them
    {
      for (const [seg, cuts] of splits) {
        cuts.sort((x, y) => x.s - y.s);
        this.segs.splice(this.segs.indexOf(seg), 1);
        let a = seg.a, s0 = 0;
        for (const c of [...cuts, { s: seg.len, v: seg.b }]) {
          if (c.s - s0 > 0.2) {
            const pts: number[] = [];
            for (let s = s0; s < c.s; s += 2) { at(seg, s, tmp); pts.push(tmp.x, tmp.y, tmp.z); }
            at(seg, c.s, tmp); pts.push(tmp.x, tmp.y, tmp.z);
            if (pts.length >= 6) this.segs.push({ ...polyline(pts), a, b: c.v, cross: null });
          }
          a = c.v; s0 = c.s;
        }
      }
    }
    yield;
    let nA = 0;
    for (const seg of this.segs) {
      if (++nA % 3000 === 0) yield;
      for (const [v, fwd] of [[seg.a, true], [seg.b, false]] as const) {
        if (!this.adj.has(v)) this.adj.set(v, []);
        this.adj.get(v)!.push({ seg, fwd });
      }
      if (seg.cross !== null) continue;
      const tmp = new THREE.Vector3();
      for (let s = 1; s < seg.len - 1; s += 4) {
        at(seg, s, tmp);
        this.grid.add(tmp.x, tmp.z, { seg, s });
      }
    }
    this.traffic?.setZebras((key) => (this.busy.get(key) ?? 0) > 0);

  }

  get walking() { return this.walkers.length; }

  // --- people standing still (waiting at a stop) ---
  private standing = new Map<Agent, number>();
  /** show someone standing at a spot (facing a random way) */
  stand(agent: Agent, sp: WalkSpot) {
    if (this.standing.has(agent) || !this.free.length) return;
    const i = this.free.pop()!;
    const p = at(sp.seg, sp.s, new THREE.Vector3());
    // a little scatter, so a crowd does not stand in one voxel
    p.x += (Math.random() - 0.5) * 2.5; p.z += (Math.random() - 0.5) * 2.5;
    const yaw = Math.random() * Math.PI * 2;
    this.mesh.setMatrixAt(i, this.m4.compose(p, this.q.setFromEuler(this.e.set(0, yaw, 0)), this.one));
    this.walk.setXY(i, 0, yaw);
    this.mesh.instanceMatrix.needsUpdate = true;
    this.standing.set(agent, i);
  }
  unstand(agent: Agent) {
    const i = this.standing.get(agent);
    if (i === undefined) return;
    this.mesh.setMatrixAt(i, new THREE.Matrix4().makeScale(0, 0, 0));
    this.mesh.instanceMatrix.needsUpdate = true;
    this.free.push(i);
    this.standing.delete(agent);
  }

  /** debug overlay (roads view, R): pavements green, footpaths blue, crossings white, corner links grey */
  overlay() {
    const pos: number[] = [], col: number[] = [];
    const c = new THREE.Color();
    for (const seg of this.segs) {
      c.set(seg.cross !== null ? 0xffffff : seg.park ? 0x2fb9ff : seg.len < 40 && seg.pts.length === 6 ? 0x888888 : 0x3ddc84);
      for (let i = 0; i + 5 < seg.pts.length; i += 3) {
        pos.push(seg.pts[i], seg.pts[i + 1] + 0.6, seg.pts[i + 2], seg.pts[i + 3], seg.pts[i + 4] + 0.6, seg.pts[i + 5]);
        col.push(c.r, c.g, c.b, c.r, c.g, c.b);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    const lines = new THREE.LineSegments(g, new THREE.LineBasicNodeMaterial({ vertexColors: true, fog: false }));
    const ghost = new THREE.LineSegments(g, new THREE.LineBasicNodeMaterial({
      vertexColors: true, fog: false, depthTest: false, transparent: true, opacity: 0.3 }));
    ghost.renderOrder = 6;
    const group = new THREE.Group();
    group.add(lines, ghost);
    return group;
  }

  private parks: Seg[] | null = null;
  /** a point on a footpath (park, square) between min and max metres from (x, z), if any */
  parkSpot(x: number, z: number, min: number, max: number, r: () => number): WalkSpot | null {
    const parks = this.parks ??= this.segs.filter((sg) => sg.park);
    const tmp = new THREE.Vector3();
    for (let t = 0; t < 40; t++) {
      const seg = parks[Math.floor(r() * parks.length)];
      if (!seg) return null;
      const s = r() * seg.len;
      at(seg, s, tmp);
      const d = Math.hypot(tmp.x - x, tmp.z - z);
      if (d >= min && d <= max) return { seg, s };
    }
    return null;
  }

  /** nearest pavement point to a scene position (this group's frame) */
  spotNear(x: number, z: number): WalkSpot | null {
    return this.grid.nearest(x, z, 160);
  }

  /** shortest walk (A* over pavements, corners and crossings): the segments with their direction */
  private route(from: WalkSpot, to: WalkSpot) {
    if (from.seg === to.seg) return [{ seg: from.seg, fwd: to.s >= from.s }];
    const goal = at(to.seg, to.s, new THREE.Vector3()), p = new THREE.Vector3();
    const start = at(from.seg, from.s, new THREE.Vector3());
    const maxCost = start.distanceTo(goal) * 2.5 + 400;   // m: generous detours
    type Node = { v: number; cost: number; prev: Node | null; via: { seg: Seg; fwd: boolean } | null };
    const best = new Map<number, number>();
    const heap = new MinHeap<Node>();   // by estimated total cost
    const push = (n: Node, s: Seg, fwd: boolean) => {
      at(s, fwd ? s.len : 0, p);
      heap.push(n.cost + p.distanceTo(goal), n);
    };
    // start at both ends of the first segment
    for (const fwd of [true, false]) {
      const n: Node = { v: fwd ? from.seg.b : from.seg.a, cost: fwd ? from.seg.len - from.s : from.s, prev: null, via: { seg: from.seg, fwd } };
      best.set(n.v, n.cost);
      push(n, from.seg, fwd);
    }
    let guard = 0;
    while (heap.size && guard++ < 60000) {
      const n = heap.pop()!;
      if (n.cost > (best.get(n.v) ?? Infinity)) continue;
      for (const e of this.adj.get(n.v) ?? []) {
        if (e.seg === to.seg) {   // reached the goal segment
          const out: { seg: Seg; fwd: boolean }[] = [{ seg: to.seg, fwd: e.fwd }];
          for (let x: Node | null = n; x; x = x.prev) if (x.via) out.unshift(x.via);
          return out;
        }
        const nv = e.fwd ? e.seg.b : e.seg.a;
        const c = n.cost + e.seg.len + (e.seg.cross !== null ? 4 : 0);
        if (c > maxCost) continue;   // (an unreachable goal gives up early instead of searching everywhere)
        if (c < (best.get(nv) ?? Infinity)) {
          best.set(nv, c);
          const nn: Node = { v: nv, cost: c, prev: n, via: e };
          push(nn, e.seg, e.fwd);
        }
      }
    }
    return null;
  }

  /** start a walk; progress (0-1): already that far (after a jump in time). Returns its length (0: none) */
  spawn(agent: Agent, from: WalkSpot, to: WalkSpot, progress = 0) {
    if (!this.free.length) return 0;
    const segs = this.route(from, to);
    if (!segs) return 0;
    const lenOf = (k: number) => {
      const { seg, fwd } = segs[k];
      const a = k === 0 ? (fwd ? from.s : seg.len - from.s) : 0;
      const b = k === segs.length - 1 ? (fwd ? to.s : seg.len - to.s) : seg.len;
      return Math.max(0, b - a);
    };
    let total = 0;
    for (let k = 0; k < segs.length; k++) total += lenOf(k);
    const w: Walker = { agent, segs, k: 0, s: segs[0].fwd ? from.s : segs[0].seg.len - from.s, v: 1.1 + Math.random() * 0.5,
      dist: 0, wait: 0, crossing: null, idx: this.free.pop()!, end: segs[segs.length - 1].fwd ? to.s : segs[segs.length - 1].seg.len - to.s };
    // jump ahead
    let left = progress * total;
    while (left > 0 && w.k < segs.length) {
      const l = (w.k === segs.length - 1 ? w.end : segs[w.k].seg.len) - w.s;
      if (left < l) { w.s += left; break; }
      left -= l; w.k++; w.s = 0;
    }
    if (w.k >= segs.length) { this.free.push(w.idx); return 0; }
    this.walkers.push(w);
    this.byAgent.set(w.agent, w);
    return total;
  }

  remove(agent: Agent) {
    const w = this.byAgent.get(agent);
    if (w) this.release(w);
  }
  clear() { for (const w of this.walkers.slice()) this.release(w); for (const a of [...this.standing.keys()]) this.unstand(a); }
  private release(w: Walker) {
    if (w.crossing !== null) this.busy.set(w.crossing, (this.busy.get(w.crossing) ?? 1) - 1);
    this.mesh.setMatrixAt(w.idx, new THREE.Matrix4().makeScale(0, 0, 0));
    this.free.push(w.idx);
    this.walkers.splice(this.walkers.indexOf(w), 1);
    this.byAgent.delete(w.agent);
  }

  /** scene position of a pavement / footpath spot */
  spotPosition(sp: WalkSpot) { return at(sp.seg, sp.s, new THREE.Vector3()); }

  /** the instance of an agent (for picking) */
  agentAt(i: number) { return this.walkers.find((w) => w.idx === i)?.agent ?? [...this.standing].find(([, k]) => k === i)?.[0] ?? null; }
  positionOf(agent: Agent, out = new THREE.Vector3()) {
    const w = this.byAgent.get(agent);
    if (!w) {
      const i = this.standing.get(agent);
      if (i === undefined) return null;
      this.mesh.getMatrixAt(i, this.m4);
      return out.setFromMatrixPosition(this.m4);
    }
    const { seg, fwd } = w.segs[w.k];
    return at(seg, fwd ? w.s : seg.len - w.s, out);
  }
  /** what is left of a walk: points (this group's frame; points = false: none) and length */
  routeAhead(agent: Agent, points = true) {
    const w = this.byAgent.get(agent);
    const pts: number[] = [];
    let len = 0;
    if (!w) return { pts, len };
    const p = new THREE.Vector3();
    for (let k = w.k; k < w.segs.length; k++) {
      const { seg, fwd } = w.segs[k];
      const s0 = k === w.k ? w.s : 0, s1 = k === w.segs.length - 1 ? w.end : seg.len;
      if (points) for (let s = s0; s <= s1; s += 2) { at(seg, fwd ? s : seg.len - s, p); pts.push(p.x, p.y + 0.3, p.z); }
      len += Math.max(0, s1 - s0);
    }
    return { pts, len };
  }

  private m4 = new THREE.Matrix4(); private q = new THREE.Quaternion(); private e = new THREE.Euler();
  private pa = new THREE.Vector3(); private pb = new THREE.Vector3(); private one = new THREE.Vector3(1, 1, 1);

  update(dt: number) {
    for (const w of this.walkers.slice()) {
      let step = w.v * dt;
      while (step > 0 && w.k < w.segs.length) {
        const { seg } = w.segs[w.k];
        const stop = w.k === w.segs.length - 1 ? w.end : seg.len;
        // about to cross a street: wait for a gap (or until patience runs out; then cars wait)
        if (seg.cross !== null && w.crossing !== seg.cross) {
          // under a light: wait for the cars' red (a rare one gives up after a long wait); else for a gap
          const walk = this.traffic?.walkSignal(seg.cross) ?? null;
          const clear = !this.traffic || this.traffic.zebraClear(seg.cross);   // (no car that could not stop)
          const go = walk === false ? clear && w.wait >= RED_PATIENCE : clear;
          if (!go && (walk === false || w.wait < CROSS_PATIENCE)) { w.wait += dt; step = 0; break; }
          w.wait = 0; w.crossing = seg.cross;
          this.busy.set(seg.cross, (this.busy.get(seg.cross) ?? 0) + 1);
        }
        const room = stop - w.s;
        if (step < room) { w.s += step; w.dist += step; step = 0; break; }
        w.s = stop; w.dist += room; step -= room;
        if (w.crossing !== null) { this.busy.set(w.crossing, (this.busy.get(w.crossing) ?? 1) - 1); w.crossing = null; }
        w.k++; w.s = 0;
      }
      if (w.k >= w.segs.length) { const a = w.agent; this.release(w); this.onArrive?.(a); continue; }
      // pose: facing the walking direction, legs swing with the distance walked
      const { seg, fwd } = w.segs[w.k];
      const s = fwd ? w.s : seg.len - w.s, d = fwd ? 0.5 : -0.5;
      at(seg, s, this.pa); at(seg, s + d, this.pb);
      const ang = Math.atan2(-(this.pb.z - this.pa.z), this.pb.x - this.pa.x);
      this.q.setFromEuler(this.e.set(0, ang, 0));
      this.m4.compose(this.pa, this.q, this.one);
      this.mesh.setMatrixAt(w.idx, this.m4);
      this.walk.setXY(w.idx, step === 0 && w.wait > 0 ? 0 : w.dist * 2.4, ang);
    }
    // draw only up to the highest slot in use (slots are handed out from the low end)
    let hi = -1;
    for (const w of this.walkers) if (w.idx > hi) hi = w.idx;
    for (const i of this.standing.values()) if (i > hi) hi = i;
    this.mesh.count = hi + 1;
    this.mesh.instanceMatrix.clearUpdateRanges(); this.mesh.instanceMatrix.addUpdateRange(0, (hi + 1) * 16);
    this.mesh.instanceMatrix.needsUpdate = true;
    this.walk.clearUpdateRanges(); this.walk.addUpdateRange(0, (hi + 1) * 2);
    this.walk.needsUpdate = true;
  }
}
