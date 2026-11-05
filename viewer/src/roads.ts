// Movement network (pipeline roads.py, from OpenStreetMap): the graph vehicles drive on.
// Lanes are derived here: each direction of travel follows the centreline shifted to the right.
// Debug overlay: R key / ?roads=1.
import * as THREE from 'three/webgpu';
import { cumulative } from './polyline';

export interface RoadNetwork {
  nodes: [number, number, number, number, number][];    // voxel coords: x east, y north, z ground; signals; region edge
  // from, to, kind, oneway, km/h, lanes, xyz..., rank, controls [m from start, code, direction],
  // bikes both ways (contraflow cycling), pavement [left, right] m from the centreline (0: none found),
  // mapped crosswalks [[m from start, 1 signals | 2 marked | 3 unmarked]], roundabout
  edges: [number, number, number, number, number, number, number[], number, [number, number, number][], number, [number, number],
    [number, number][]?, number?][];
  /** road inside merged junctions (pipeline consolidate): same fields as edges, not driven as edges */
  pieces?: RoadNetwork['edges'];
  /** footpaths (parks, squares, pedestrian streets): [[node a, node b], xyz...] */
  paths?: [[number, number], number[]][];
}

/** traffic control codes on edges (pipeline roads.py) */
export const CONTROL = { signals: 1, giveWay: 2, stop: 3, crossingLight: 4, calming: 5 };   // (roads.py: 1 = a stop line of the signalled junction ahead)

export const KIND = { road: 0, tram: 1, rail: 2 };
/** key of a mapped crosswalk: its edge and index along it (shared by pedestrians.ts and traffic.ts) */
export const crosswalkKey = (edge: number, i: number) => edge * 1000 + i;
const COLORS = [0xffd400, 0xff3dd0, 0x33d6ff];   // road, tram, rail
const ONEWAY = 0xff7a00;
const LANE_COLOR = 0xffffff;
const LIFT = 0.6;                                // m above the ground, so the lines are not hidden in it
const HALF_LANE = 1.5;                           // m from the centreline to a two-way road's lane

/** one direction of travel along one edge, in scene coordinates local to the world group */
export interface Lane {
  edge: number;
  from: number; to: number;    // node indices
  kind: number;
  speed: number;               // m/s
  pts: Float32Array;           // x, y, z per point
  cum: Float32Array;           // distance along the lane at each point
  length: number;
  rank: number;                // road class: priority at unsigned intersections
  bikeOnly?: boolean;          // contraflow cycle lane in a one-way street
  controls: { s: number; code: number }[];   // signals / give way / stop along the lane, in driving direction
  crosswalks: { s: number; key: number }[];  // mapped crosswalks along the lane (cars give way to people on them)
  toSignal: boolean;           // the junction it leads to is signalled (roads.py assign_signals)
}

export async function loadRoads(url: string): Promise<RoadNetwork | null> {
  const r = await fetch(url);
  return r.ok ? r.json() : null;
}

// roads.json is in voxels (x east, y north, z up); the scene in metres (x east, y up, z south)
const toScene = (vs: number, x: number, y: number, z: number) => [x * vs, z * vs, -y * vs];
/** a roads.json polyline (voxels: x, y, z per point) as a flat scene polyline (metres: x, y up, z) */
export function sceneLine(raw: ArrayLike<number>, vs: number, lift = 0): number[] {
  const out: number[] = [];
  for (let i = 0; i < raw.length; i += 3) out.push(raw[i] * vs, raw[i + 2] * vs + lift, -raw[i + 1] * vs);
  return out;
}

/** polyline shifted sideways by `off` metres (positive: to the right of the direction of travel) */
function offsetLine(p: number[], off: number) {
  const n = p.length / 3, out = new Float32Array(p.length);
  for (let i = 0; i < n; i++) {
    // average direction of the adjacent segments (mitred corner)
    const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
    let dx = p[b * 3] - p[a * 3], dz = p[b * 3 + 2] - p[a * 3 + 2];
    const l = Math.hypot(dx, dz) || 1;
    dx /= l; dz /= l;
    // miter: at a bend the corner point sits off / cos(half the bend) out, so both neighbouring segments
    // stay `off` from the centreline (without it, lanes folded at sharp corners); capped at 2x (no spikes)
    // (the bend measured over at least 1 m each side: resampled lines have near-duplicate points whose
    // tiny segments have no meaningful direction)
    let k = 1;
    if (i > 0 && i < n - 1) {
      let j = i - 1, m = i + 1;
      while (j > 0 && Math.hypot(p[i * 3] - p[j * 3], p[i * 3 + 2] - p[j * 3 + 2]) < 1) j--;
      while (m < n - 1 && Math.hypot(p[m * 3] - p[i * 3], p[m * 3 + 2] - p[i * 3 + 2]) < 1) m++;
      let ux = p[i * 3] - p[j * 3], uz = p[i * 3 + 2] - p[j * 3 + 2], vx = p[m * 3] - p[i * 3], vz = p[m * 3 + 2] - p[i * 3 + 2];
      const ul = Math.hypot(ux, uz), vl = Math.hypot(vx, vz);
      if (ul > 0.3 && vl > 0.3) {
        ux /= ul; uz /= ul; vx /= vl; vz /= vl;
        const cosHalf = Math.sqrt(Math.max(0, (1 + ux * vx + uz * vz) / 2));
        k = 1 / Math.max(0.5, cosHalf);
      }
    }
    // right of travel direction d (y up): d × up = (-dz, 0, dx)
    out[i * 3] = p[i * 3] - dz * off * k;
    out[i * 3 + 1] = p[i * 3 + 1];
    out[i * 3 + 2] = p[i * 3 + 2] + dx * off * k;
  }
  return out;
}

function reversed(p: number[]) {
  const out: number[] = [];
  for (let i = p.length - 3; i >= 0; i -= 3) out.push(p[i], p[i + 1], p[i + 2]);
  return out;
}

/** all lanes: forward along each edge, plus backward on two-way edges; roads keep right */
export function buildLanes(net: RoadNetwork, vs: number): Lane[] {
  const lanes: Lane[] = [];
  net.edges.forEach(([from, to, kind, oneway, kmh, , raw, rank, ctl, contra, , xw], edge) => {
    const p = sceneLine(raw, vs);
    let len = 0;
    for (let i = 3; i < p.length; i += 3) len += Math.hypot(p[i] - p[i - 3], p[i + 2] - p[i - 1]);
    const off = kind === KIND.road && !oneway ? HALF_LANE : 0;   // tracks are mapped one per line: no shift
    const dirs: [number, number, number[], number][] = [[from, to, p, 1]];
    if (!oneway || (kind === KIND.road && contra)) dirs.push([to, from, reversed(p), -1]);
    for (const [a, b, line, sign] of dirs) {
      const bikeOnly = !!oneway && sign < 0;   // cycling against the traffic, at the kerb
      const pts = offsetLine(line, bikeOnly ? HALF_LANE + 1 : off);
      const cum = cumulative(pts);
      const length = cum[cum.length - 1];
      const controls = ctl.filter(([, , dr]) => dr === 0 || dr === sign)
        .map(([m, code]) => ({ s: (sign > 0 ? m : len - m) * (length / (len || 1)), code })).sort((x, y) => x.s - y.s);
      const crosswalks = (xw ?? []).map(([m], i) => ({ s: (sign > 0 ? m : len - m) * (length / (len || 1)), key: crosswalkKey(edge, i) }))
        .sort((x, y) => x.s - y.s);
      lanes.push({ edge, from: a, to: b, kind, speed: kmh / 3.6, pts, cum, length, rank, controls, crosswalks, bikeOnly,
        toSignal: net.nodes[b]?.[3] === 1 });
    }
  });
  return lanes;
}

/** centrelines (one-way roads orange), the right-hand lanes (white) and a dot per intersection */
export function roadOverlay(net: RoadNetwork, lanes: Lane[], vs: number) {
  const pos: number[] = [], col: number[] = [];
  const c = new THREE.Color();
  // (junction pieces: road surface inside merged junctions, driven through the junction's curves)
  for (const [, , kind, oneway, , , p] of [...net.edges, ...(net.pieces ?? [])]) {
    c.set(kind === KIND.road && oneway ? ONEWAY : COLORS[kind]);
    for (let i = 0; i + 5 < p.length; i += 3) {
      const a = toScene(vs, p[i], p[i + 1], p[i + 2]), b = toScene(vs, p[i + 3], p[i + 4], p[i + 5]);
      pos.push(a[0], a[1] + LIFT, a[2], b[0], b[1] + LIFT, b[2]);
      col.push(c.r, c.g, c.b, c.r, c.g, c.b);
    }
  }
  for (const l of lanes) {
    c.set(l.bikeOnly ? 0xc080ff : LANE_COLOR);                       // contraflow cycle lanes purple
    if (l.kind !== KIND.road || (net.edges[l.edge][3] && !l.bikeOnly)) continue;   // one-way lanes: the centreline
    const p = l.pts;
    for (let i = 0; i + 5 < p.length; i += 3) {
      pos.push(p[i], p[i + 1] + LIFT, p[i + 2], p[i + 3], p[i + 4] + LIFT, p[i + 5]);
      col.push(c.r, c.g, c.b, c.r, c.g, c.b);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  const lines = new THREE.LineSegments(g, new THREE.LineBasicNodeMaterial({ vertexColors: true, fog: false }));
  // faint copy drawn through everything, so hidden stretches (under trees, in coarse distant tiles) still show
  const ghost = new THREE.LineSegments(g, new THREE.LineBasicNodeMaterial({
    vertexColors: true, fog: false, depthTest: false, transparent: true, opacity: 0.35 }));
  ghost.renderOrder = 6;

  const np = new Float32Array(net.nodes.length * 3);
  net.nodes.forEach(([x, y, z], i) => { const s = toScene(vs, x, y, z); np.set([s[0], s[1] + LIFT, s[2]], i * 3); });
  const pg = new THREE.BufferGeometry();
  pg.setAttribute('position', new THREE.BufferAttribute(np, 3));
  const dots = new THREE.Points(pg, new THREE.PointsNodeMaterial({ color: 0xffffff, size: 1, fog: false }));

  const group = new THREE.Group();
  group.add(lines, ghost, dots);
  return group;
}
