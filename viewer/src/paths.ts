// The lane graph vehicles drive on: paths (lanes between junction trims, and the curves through
// junctions), their geometry and speed limits from curvature, and the records the simulation keeps on them.
import * as THREE from 'three/webgpu';
import { KIND, type Lane } from './roads';
import { cumulative, pointAt } from './polyline';
import type { Kind } from './vehicleTypes';
import type { Agent } from './agents';

export const A_LAT = 2.6;           // m/s², lateral acceleration in curves (× the driver's aggressiveness)
const MIN_RADIUS = 5.5;             // m, car turning circle (path of the car's centre)

export const GREEN = 0, AMBER = 1, RED = 2;
export const LIGHT_COLORS = [new THREE.Color(0x30ff60), new THREE.Color(0xffb000), new THREE.Color(0xff2a1a)];

/** a stretch vehicles drive along: a lane (between its junction trims) or a junction curve */
export interface Path {
  pts: Float32Array; cum: Float32Array; length: number;
  vmax: Float32Array;          // curve speed limit at each point
  speed: number; kind: number;
  start: number; end: number;  // driven part: lanes from after the junction behind to the stop line
  occ: Vehicle[];              // vehicles whose front is on this path, sorted by position
  // lanes
  lane: Lane | null;
  next: Map<Path, Path>;       // next lane -> junction curve
  approach: Approach | null;   // how this lane enters the junction at its end
  crossings: { s: number; sig: Signal }[];   // pedestrian-crossing lights along the lane
  bumps: number[];                           // speed bumps / cushions / raised tables along the lane (s)
  zebras: { s: number; key: number }[];      // mapped crosswalks along the lane (give way to people on them)
  // junction curves
  junction: Junction | null; from: Path | null; to: Path | null; turn: number;   // turn: heading change, rad
}

/** a kerbside spot on a lane: where a trip starts or ends */
export interface Spot { p: Path; s: number }

/** a timetabled run of a STIB line through the region */
export interface TransitRun {
  line: string; headsign: string; color: string; mode: number;
  stops: { p: Path; s: number; dep: number; name: string; id: number; arrived: number; served: boolean }[];
  next: number;   // next stop to serve
}

export interface Approach { junction: Junction; yields: boolean; rank: number; dir: THREE.Vector2; signal: Signal | null; lightS: number }   // lightS: where its light stands
export interface Junction { occ: Set<Vehicle>; conflicts: Map<Path, Map<Path, boolean>>; signalled: boolean }
export interface Signal { cycle: number; offset: number; phases: [number, number][]; state: number }   // phases: [from, to) green

export interface Vehicle {
  t: Kind; kind: number;
  route: Path[];               // route[0]: the path the front bumper is on; then planned paths
  hist: Path[];                // paths behind (for the axles and carriages still on them)
  s: number; v: number; f: number;   // front bumper position on route[0], speed, share of the speed limit
  ag: number;                  // aggressiveness: scales acceleration, braking, cornering; shortens headway
  grant: Path | null;          // junction curve it may enter
  inJunction: Path | null;     // junction curve it occupies (until its rear has left it)
  wait: number;                // s spent waiting for a junction
  trip: Path[];                // cars: lanes still to drive after the route planned so far
  dest: { p: Path; s: number } | null;   // cars: where to pull over (null: leave the region)
  arrive: number;              // when it started approaching its next junction (first come, first served)
  inst: number[];             // per section: its body in the fleet (fleet.ts draws it)
  agent: Agent | null;         // cars: who drives (agents.ts); null for trams / trains
  transit?: TransitRun;        // STIB trams and buses: line and stops (transit.ts)
  stuck: number;               // s without moving
  block?: string | null; blockBy?: Vehicle | null;   // debugging: what it brakes for
  why?: string;                // debugging: why it may not enter its next junction
}

const tmp = new THREE.Vector3(), tmp2 = new THREE.Vector3();

export const sample = (p: Path, s: number, out: THREE.Vector3) => pointAt(p.pts, p.cum, s, out);

/** speed limit from curvature alone (100 m/s on straights): heading change over a ±4 m window */
export function curveLimits(pts: Float32Array, cum: Float32Array) {
  const n = cum.length, out = new Float32Array(n).fill(100);
  const dir = (i: number, j: number) => Math.atan2(pts[j * 3 + 2] - pts[i * 3 + 2], pts[j * 3] - pts[i * 3]);
  for (let i = 0; i < n; i++) {
    let a = i, b = i;
    while (a > 0 && cum[i] - cum[a] < 4) a--;
    while (b < n - 1 && cum[b] - cum[i] < 4) b++;
    if (a === i || b === i) continue;
    let d = Math.abs(dir(i, b) - dir(a, i));
    if (d > Math.PI) d = 2 * Math.PI - d;
    const k = d / Math.max(1, cum[b] - cum[a]);              // 1/m
    if (k > 1e-4) out[i] = Math.min(100, Math.sqrt(A_LAT / k));
  }
  return out;
}

export function makePath(pts: Float32Array, speed: number, kind: number, cum = cumulative(pts)): Path {
  const length = cum[cum.length - 1];
  return { pts, cum, length, vmax: curveLimits(pts, cum), speed, kind, start: 0, end: length, occ: [],
    lane: null, next: new Map(), approach: null, crossings: [], bumps: [], zebras: [], junction: null, from: null, to: null, turn: 0 };
}

/** travel direction (x, z) at s */
export function heading(p: Path, s: number) {
  sample(p, Math.max(0, s - 1), tmp); sample(p, Math.min(p.length, s + 1), tmp2);
  return new THREE.Vector2(tmp2.x - tmp.x, tmp2.z - tmp.z).normalize();
}

/** travel direction (x, z) between s and s + d along the path (d < 0: looking back) */
export function longHeading(p: Path, s: number, d: number) {
  const a = Math.min(s, s + d), b = Math.max(s, s + d);
  sample(p, Math.max(0, a), tmp); sample(p, Math.min(p.length, b), tmp2);
  return new THREE.Vector2(tmp2.x - tmp.x, tmp2.z - tmp.z).normalize();
}

/** junction curve: cubic Bézier from a's stop line to b's start, tangent to both */
export function makeCurve(a: Path, b: Path): Path {
  const p0 = sample(a, a.end, new THREE.Vector3()), p3 = sample(b, b.start, new THREE.Vector3());
  const da = heading(a, a.end), db = heading(b, b.start);
  const chord = p0.distanceTo(p3);
  // a turn: where the two lane lines meet (x), the curve approximates a circular arc tangent to both
  // (Bézier handle = 4/3 tan(θ/4) R, tangent length = R tan(θ/2)); straight on / U-turn: plain handles
  const theta = Math.acos(Math.max(-1, Math.min(1, da.dot(db))));
  const den = da.x * db.y - da.y * db.x;
  let k0 = Math.max(chord * 0.45, a.kind === KIND.road ? MIN_RADIUS * 0.6 : 1), k3 = k0;
  if (Math.abs(den) > 0.05 && theta > 0.15 && theta < 2.6) {
    const rx = p3.x - p0.x, rz = p3.z - p0.z;
    const ta = (rx * db.y - rz * db.x) / den, tb = (rx * da.y - rz * da.x) / den;   // p0 + ta·da = p3 + tb·db
    if (ta > 0 && tb < 0 && ta < 3 * chord && -tb < 3 * chord) {
      const h = (4 / 3) * Math.tan(theta / 4) / Math.tan(theta / 2);
      k0 = ta * h; k3 = -tb * h;
    }
  }
  const p1 = p0.clone().add(new THREE.Vector3(da.x, 0, da.y).multiplyScalar(k0));
  const p2 = p3.clone().sub(new THREE.Vector3(db.x, 0, db.y).multiplyScalar(k3));
  const N = 12, pts = new Float32Array((N + 1) * 3), q = new THREE.Vector3();
  for (let i = 0; i <= N; i++) {
    const t = i / N, u = 1 - t;
    q.set(0, 0, 0).addScaledVector(p0, u * u * u).addScaledVector(p1, 3 * u * u * t).addScaledVector(p2, 3 * u * t * t)
      .addScaledVector(p3, t * t * t);
    pts.set([q.x, q.y, q.z], i * 3);
  }
  const c = makePath(pts, Math.min(a.speed, b.speed), a.kind);
  c.from = a; c.to = b;
  c.turn = Math.atan2(da.x * db.y - da.y * db.x, da.dot(db));   // signed heading change
  return c;
}

/** do two junction curves cross or come closer than a car width? */
export function conflict(a: Path, b: Path) {
  if (a.from === b.from) return false;          // same approach: they follow each other
  if (a.to === b.to) return true;               // merging into the same lane
  for (let i = 0; i < a.cum.length; i++)
    for (let j = 0; j < b.cum.length; j++)
      if (Math.hypot(a.pts[i * 3] - b.pts[j * 3], a.pts[i * 3 + 2] - b.pts[j * 3 + 2]) < 2.2) return true;
  return false;
}

/** a signal's colour at time t (s of the day) */
export function signalState(sig: Signal, t: number) {
  const tt = (t + sig.offset) % sig.cycle;
  for (const [g0, g1] of sig.phases) {
    if (tt >= g0 && tt < g1) return GREEN;
    if (tt >= g1 && tt < g1 + 3) return AMBER;
  }
  return RED;
}
