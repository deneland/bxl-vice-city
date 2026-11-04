// Movement network (pipeline roads.py, from OpenStreetMap): the graph vehicles drive on.
// Lanes are derived here: each direction of travel follows the centreline shifted to the right.
// Debug overlay: R key / ?roads=1.
import * as THREE from 'three/webgpu';

export interface RoadNetwork {
  nodes: [number, number, number, number, number][];    // voxel coords: x east, y north, z ground; signals; region edge
  // from, to, kind, oneway, km/h, lanes, xyz..., rank, controls [m from start, code, direction],
  // bikes both ways (contraflow cycling), pavement [left, right] m from the centreline (0: none found)
  edges: [number, number, number, number, number, number, number[], number, [number, number, number][], number, [number, number]][];
  /** footpaths (parks, squares, pedestrian streets): [[node a, node b], xyz...] */
  paths?: [[number, number], number[]][];
}

/** traffic control codes on edges (pipeline roads.py) */
export const CONTROL = { signals: 1, giveWay: 2, stop: 3 };

export const KIND = { road: 0, tram: 1, rail: 2 };
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
}

export async function loadRoads(url: string): Promise<RoadNetwork | null> {
  const r = await fetch(url);
  return r.ok ? r.json() : null;
}

const toScene = (vs: number, x: number, y: number, z: number) => [x * vs, z * vs, -y * vs];

/** polyline shifted sideways by `off` metres (positive: to the right of the direction of travel) */
function offsetLine(p: number[], off: number) {
  const n = p.length / 3, out = new Float32Array(p.length);
  for (let i = 0; i < n; i++) {
    // average direction of the adjacent segments (mitred corner)
    const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
    let dx = p[b * 3] - p[a * 3], dz = p[b * 3 + 2] - p[a * 3 + 2];
    const l = Math.hypot(dx, dz) || 1;
    dx /= l; dz /= l;
    // right of travel direction d (y up): d × up = (-dz, 0, dx)
    out[i * 3] = p[i * 3] - dz * off;
    out[i * 3 + 1] = p[i * 3 + 1];
    out[i * 3 + 2] = p[i * 3 + 2] + dx * off;
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
  net.edges.forEach(([from, to, kind, oneway, kmh, , raw, rank, ctl, contra], edge) => {
    const p: number[] = [];
    for (let i = 0; i < raw.length; i += 3) p.push(...toScene(vs, raw[i], raw[i + 1], raw[i + 2]));
    let len = 0;
    for (let i = 3; i < p.length; i += 3) len += Math.hypot(p[i] - p[i - 3], p[i + 2] - p[i - 1]);
    const off = kind === KIND.road && !oneway ? HALF_LANE : 0;   // tracks are mapped one per line: no shift
    const dirs: [number, number, number[], number][] = [[from, to, p, 1]];
    if (!oneway || (kind === KIND.road && contra)) dirs.push([to, from, reversed(p), -1]);
    for (const [a, b, line, sign] of dirs) {
      const bikeOnly = !!oneway && sign < 0;   // cycling against the traffic, at the kerb
      const pts = offsetLine(line, bikeOnly ? HALF_LANE + 1 : off);
      const cum = new Float32Array(pts.length / 3);
      for (let i = 1; i < cum.length; i++)
        cum[i] = cum[i - 1] + Math.hypot(pts[i * 3] - pts[i * 3 - 3], pts[i * 3 + 1] - pts[i * 3 - 2], pts[i * 3 + 2] - pts[i * 3 - 1]);
      const length = cum[cum.length - 1];
      const controls = ctl.filter(([, , dr]) => dr === 0 || dr === sign)
        .map(([m, code]) => ({ s: (sign > 0 ? m : len - m) * (length / (len || 1)), code })).sort((x, y) => x.s - y.s);
      lanes.push({ edge, from: a, to: b, kind, speed: kmh / 3.6, pts, cum, length, rank, controls, bikeOnly });
    }
  });
  return lanes;
}

/** centrelines (one-way roads orange), the right-hand lanes (white) and a dot per intersection */
export function roadOverlay(net: RoadNetwork, lanes: Lane[], vs: number) {
  const pos: number[] = [], col: number[] = [];
  const c = new THREE.Color();
  for (const [, , kind, oneway, , , p] of net.edges) {
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
