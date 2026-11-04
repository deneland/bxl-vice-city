// Traffic: vehicles as simple agents on the lane graph (roads.ts), one lane per direction.
//
//  bodies      two axles on the path (rear axle trails the front by the wheelbase), so a car turns
//              like a car; junction curves are kept wider than the turning circle
//  speed       Intelligent Driver Model (IDM): accelerate towards the desired speed, keep distance to
//              whatever is ahead (vehicle, red light, a junction you may not enter yet); the desired
//              speed drops before curves (comfortable lateral acceleration)
//  junctions   a vehicle needs a grant to enter a junction curve: no conflicting curve may be in use,
//              and conflicting vehicles with precedence go first (signals; give way / stop signs; road
//              rank; straight before turning; otherwise from the right, Belgian priorité de droite);
//              after waiting long enough a vehicle goes anyway (no gridlock)
//  lights      OSM traffic signals: intersections cycle two approach groups; pedestrian-crossing
//              signals turn red now and then; drawn as small coloured boxes at the stop line
//
//  trips       each car drives a trip: from the kerb (or the region edge: through traffic) to a destination
//              (kerb or region edge), routed by A* on the directed lane graph (one-ways built in; left
//              turns and U-turns cost extra); at the destination it pulls over and a new trip starts
//
// Trams and trains use the same model on their own tracks (they ignore road signals and do not
// interact with cars yet) and choose their next track at random.
import * as THREE from 'three/webgpu';
import { CONTROL, KIND, type Lane } from './roads';
import { buildModels } from './vehicleModels';

const A_LAT = 2.6;           // m/s², lateral acceleration in curves (× the driver's aggressiveness)
const MIN_RADIUS = 5.5;      // m, car turning circle (path of the car's centre)
const PLAN = 120;            // m of route planned ahead
const LOOK = 90;             // m looked ahead for obstacles and curves
const APPROACH = 30;         // m: signals / signs this close to a lane's end belong to the junction
const PATIENCE = 6;          // s waiting at a junction before ignoring priority (if nothing is in the way)
const NO_ROOM_PATIENCE = 8;  // s before entering even without room after the junction
const FORCE = 20;            // s waiting at a junction (not for a light): go anyway, even through a conflict
const STUCK = 90;            // s without moving (not queueing for a light): taken off, new trip (last resort)
const THROUGH = 0.3;         // share of trips that start / end at the region edge
const SWITCH_COS = Math.cos(50 * Math.PI / 180);   // trams / trains: sharpest track connection
const ETA = 4;               // s: give way only to conflicting vehicles arriving within this time
const QUEUE_COST = 6;        // s of routing cost per vehicle already on a lane
const TRIM = 9;              // m: lanes end this far before the intersection; curves turn in that space

/** one body on two axles: model, length, front overhang (bumper to front axle), wheelbase */
interface Section { model: string; len: number; h: number; over: number; wb: number }
/** vehicle type: its sections (carriages, front first), the gap between them, driving parameters */
interface Kind { name: string; secs: Section[]; gap: number; total: number; amax: number; b: number; T: number; s0: number }

const MODELS = buildModels();
const sec = (model: string, over: number, wb: number): Section => ({ model, len: MODELS[model].len, h: MODELS[model].h, over, wb });
const kind = (name: string, secs: Section[], gap: number, amax: number, b: number, T: number, s0: number): Kind =>
  ({ name, secs, gap, total: secs.reduce((a, x) => a + x.len, 0) + gap * (secs.length - 1), amax, b, T, s0 });
const CAR = [2.4, 3.2, 1.2, 2.0] as const;
const TYPES = {
  hatch: kind('hatch', [sec('hatch', 0.75, 2.5)], 0, ...CAR),
  sedan: kind('sedan', [sec('sedan', 0.9, 2.75)], 0, ...CAR),
  estate: kind('estate', [sec('estate', 0.9, 2.8)], 0, ...CAR),
  suv: kind('suv', [sec('suv', 0.9, 2.75)], 0, ...CAR),
  van: kind('van', [sec('van', 0.9, 3.6)], 0, 1.7, 2.6, 1.4, 2.0),
  bus: kind('bus', [sec('bus', 2.7, 6.0)], 0, 1.1, 1.8, 1.8, 2.5),
  // STIB Flexity: T3000 ~32 m in 5 sections, T4000 ~43 m in 7
  t3000: kind('t3000', [sec('tramCab', 0.8, 4.8), sec('tramMid', 0.8, 4.4), sec('tramPanto', 0.8, 4.4),
    sec('tramMid', 0.8, 4.4), sec('tramRear', 0.8, 4.8)], 0.15, 1.0, 1.3, 2.0, 3.0),
  t4000: kind('t4000', [sec('tramCab', 0.8, 4.8), sec('tramMid', 0.8, 4.4), sec('tramPanto', 0.8, 4.4), sec('tramMid', 0.8, 4.4),
    sec('tramPanto', 0.8, 4.4), sec('tramMid', 0.8, 4.4), sec('tramRear', 0.8, 4.8)], 0.15, 1.0, 1.3, 2.0, 3.0),
  train: kind('train', [0, 1, 2, 3].map(() => sec('carriage', 3, 18)), 0.8, 0.6, 0.8, 3.0, 5.0),
};
// road vehicle mix
const ROAD_MIX: [Kind, number][] = [[TYPES.hatch, 0.27], [TYPES.sedan, 0.22], [TYPES.estate, 0.12], [TYPES.suv, 0.17],
  [TYPES.van, 0.14], [TYPES.bus, 0.05]];
const pickKind = () => { let r = Math.random(); for (const [k, w] of ROAD_MIX) { r -= w; if (r <= 0) return k; } return TYPES.hatch; };
const VAN_WHITE = 0xeceeee, BUS_COLORS = [0xe9e9e4, 0xd0d3d6, 0x2f5597, 0xb22a2a], TRAM_SILVER = 0xb9bdc0, TRAIN = 0xcfd2d4;
const ACCENTS = [0x8c1c1c, 0x1f3f7a, 0x3c6fb0, 0xc9b99a];   // a few real colours the aerial palette misses
const LIGHT_COLORS = [new THREE.Color(0x30ff60), new THREE.Color(0xffb000), new THREE.Color(0xff2a1a)];
const GREEN = 0, AMBER = 1, RED = 2;

/** a stretch vehicles drive along: a lane (between its junction trims) or a junction curve */
interface Path {
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
  // junction curves
  junction: Junction | null; from: Path | null; to: Path | null; turn: number;   // turn: heading change, rad
}

interface Approach { junction: Junction; yields: boolean; rank: number; dir: THREE.Vector2; signal: Signal | null }
interface Junction { occ: Set<Vehicle>; conflicts: Map<Path, Map<Path, boolean>>; signalled: boolean }
interface Signal { cycle: number; offset: number; phases: [number, number][]; state: number }   // phases: [from, to) green

interface Vehicle {
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
  inst: { mesh: THREE.InstancedMesh; i: number }[];   // per section: its model's instanced mesh, index
  stuck: number;               // s without moving
  block?: string | null; blockBy?: Vehicle | null;   // debugging: what it brakes for
  why?: string;                // debugging: why it may not enter its next junction
}

const tmp = new THREE.Vector3(), tmp2 = new THREE.Vector3();

function sample(p: Path, s: number, out: THREE.Vector3) {
  const { pts, cum } = p;
  s = Math.min(Math.max(s, 0), p.length);
  let lo = 0, hi = cum.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cum[m] <= s) lo = m; else hi = m; }
  const seg = cum[hi] - cum[lo], k = seg > 0 ? (s - cum[lo]) / seg : 0;
  return out.set(pts[lo * 3] + (pts[hi * 3] - pts[lo * 3]) * k, pts[lo * 3 + 1] + (pts[hi * 3 + 1] - pts[lo * 3 + 1]) * k,
    pts[lo * 3 + 2] + (pts[hi * 3 + 2] - pts[lo * 3 + 2]) * k);
}

/** speed limit from curvature alone (100 m/s on straights): heading change over a ±4 m window */
function curveLimits(pts: Float32Array, cum: Float32Array) {
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

function makePath(pts: Float32Array, speed: number, kind: number): Path {
  const cum = new Float32Array(pts.length / 3);
  for (let i = 1; i < cum.length; i++)
    cum[i] = cum[i - 1] + Math.hypot(pts[i * 3] - pts[i * 3 - 3], pts[i * 3 + 1] - pts[i * 3 - 2], pts[i * 3 + 2] - pts[i * 3 - 1]);
  const length = cum[cum.length - 1];
  return { pts, cum, length, vmax: curveLimits(pts, cum), speed, kind, start: 0, end: length, occ: [],
    lane: null, next: new Map(), approach: null, crossings: [], junction: null, from: null, to: null, turn: 0 };
}

/** travel direction (x, z) at s */
function heading(p: Path, s: number) {
  sample(p, Math.max(0, s - 1), tmp); sample(p, Math.min(p.length, s + 1), tmp2);
  return new THREE.Vector2(tmp2.x - tmp.x, tmp2.z - tmp.z).normalize();
}

/** junction curve: cubic Bézier from a's stop line to b's start, tangent to both */
function makeCurve(a: Path, b: Path): Path {
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
function conflict(a: Path, b: Path) {
  if (a.from === b.from) return false;          // same approach: they follow each other
  if (a.to === b.to) return true;               // merging into the same lane
  for (let i = 0; i < a.cum.length; i++)
    for (let j = 0; j < b.cum.length; j++)
      if (Math.hypot(a.pts[i * 3] - b.pts[j * 3], a.pts[i * 3 + 2] - b.pts[j * 3 + 2]) < 2.2) return true;
  return false;
}

/** binary min-heap (A* open list) */
class Heap<T> {
  private k: number[] = []; private v: T[] = [];
  get size() { return this.k.length; }
  push(key: number, val: T) {
    const { k, v } = this;
    let i = k.length;
    k.push(key); v.push(val);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (k[p] <= key) break;
      k[i] = k[p]; v[i] = v[p]; i = p;
    }
    k[i] = key; v[i] = val;
  }
  pop(): T | undefined {
    const { k, v } = this;
    if (!k.length) return undefined;
    const top = v[0], lk = k.pop()!, lv = v.pop()!;
    if (k.length) {
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= k.length) break;
        if (c + 1 < k.length && k[c + 1] < k[c]) c++;
        if (k[c] >= lk) break;
        k[i] = k[c]; v[i] = v[c]; i = c;
      }
      k[i] = lk; v[i] = lv;
    }
    return top;
  }
}

function signalState(sig: Signal, t: number) {
  const tt = (t + sig.offset) % sig.cycle;
  for (const [g0, g1] of sig.phases) {
    if (tt >= g0 && tt < g1) return GREEN;
    if (tt >= g1 && tt < g1 + 3) return AMBER;
  }
  return RED;
}

export class Traffic {
  readonly lights: THREE.InstancedMesh;
  readonly vehicles: Vehicle[] = [];
  private paths: Path[] = [];
  private byNode = new Map<number, Path[]>();     // node -> lanes leaving it
  private signals: Signal[] = [];
  private lightAt: { sig: Signal; i: number; state: number }[] = [];
  private time = Math.random() * 1000;

  /** material: lit like the city (deferred.ts litMaterial); instance colours are the paint */
  readonly group = new THREE.Group();

  /** material: reads the model attributes vcol / vmat and the per-instance paint (deferred.ts vehicleMaterial);
   *  boundary: is this node where the network leaves the region (entry / exit for through traffic)?;
   *  palette: car paint colours (sRGB 0-255) with their shares (pipeline vehicles.py) */
  constructor(lanes: Lane[], counts = { car: 400, tram: 12, train: 3 }, material: THREE.Material,
    private boundary: (node: number) => boolean = () => false, palette: { rgb: number[]; share: number }[] = []) {
    // --- paths: lanes trimmed back from the junctions, so the curves have room to turn ---
    const incoming = new Map<number, Path[]>();
    for (const l of lanes) {
      const p = makePath(l.pts, l.speed, l.kind);
      p.lane = l;
      const trim = Math.min(l.kind === KIND.road ? TRIM : 3, p.length * 0.35);
      p.start = trim; p.end = p.length - trim;
      this.paths.push(p);
      if (!this.byNode.has(l.from)) this.byNode.set(l.from, []);
      this.byNode.get(l.from)!.push(p);
      if (!incoming.has(l.to)) incoming.set(l.to, []);
      incoming.get(l.to)!.push(p);
    }
    // --- junctions, approaches, signals ---
    for (const ins of incoming.values()) {
      const j: Junction = { occ: new Set(), conflicts: new Map(), signalled: false };
      for (const p of ins) {
        const near = p.lane!.controls.filter((c) => c.s > p.length - APPROACH);
        if (p.kind === KIND.road && near.some((c) => c.code === CONTROL.signals)) j.signalled = true;
        p.approach = { junction: j, rank: p.lane!.rank, dir: heading(p, p.end), signal: null,
          yields: near.some((c) => c.code === CONTROL.giveWay || c.code === CONTROL.stop) };
        // signals further from the junction: pedestrian crossings
        if (p.kind === KIND.road)
          for (const c of p.lane!.controls)
            if (c.code === CONTROL.signals && c.s <= p.length - APPROACH && c.s > p.start + 5) {
              const g = 35 + Math.random() * 35;
              const sig: Signal = { cycle: g + 15, offset: Math.random() * 100, phases: [[0, g]], state: GREEN };
              this.signals.push(sig);
              p.crossings.push({ s: c.s, sig });
            }
      }
      if (!j.signalled) continue;
      // two groups of approaches by axis: green for one, then the other (3 s amber, 2 s all red)
      const roads = ins.filter((p) => p.kind === KIND.road);
      const ref = roads[0].approach!.dir;
      const g0 = 18 + Math.random() * 8, g1 = 15 + Math.random() * 8, cycle = g0 + g1 + 10, offset = Math.random() * 60;
      const groups: Signal[] = [
        { cycle, offset, phases: [[0, g0]], state: GREEN },
        { cycle, offset, phases: [[g0 + 5, g0 + 5 + g1]], state: RED },
      ];
      this.signals.push(...groups);
      for (const p of roads) p.approach!.signal = groups[Math.abs(p.approach!.dir.dot(ref)) > 0.6 ? 0 : 1];
    }
    // --- trip ends: kerbside (weighted by length) and region-edge entries / exits ---
    const roads = this.paths.filter((p) => p.kind === KIND.road && p.end - p.start > 30);
    this.kerb = roads;
    this.kerbTotal = roads.reduce((a, p) => a + p.end - p.start, 0);
    this.entries = this.paths.filter((p) => p.kind === KIND.road && this.boundary(p.lane!.from) && p.end - p.start > 2);
    this.exits = this.paths.filter((p) => p.kind === KIND.road && this.boundary(p.lane!.to));
    // --- vehicles ---
    const byKind = (k: number) => this.paths.filter((p) => p.kind === k && p.end - p.start > 10);
    const plan: [string, Path[], number][] = [
      ['car', byKind(KIND.road), counts.car], ['tram', byKind(KIND.tram), counts.tram], ['train', byKind(KIND.rail), counts.train]];
    const paints: number[][] = [];   // per vehicle, linear rgb
    const carPaint = this.paintPicker(palette);
    const counts2 = new Map<string, number>();
    for (const [type, pool, n] of plan) {
      if (!pool.length) continue;
      for (let i = 0; i < n; i++) {
        const t = type === 'car' ? pickKind() : type === 'tram' ? (Math.random() < 0.3 ? TYPES.t4000 : TYPES.t3000) : TYPES.train;
        const calm = t === TYPES.bus || type !== 'car';
        // drivers differ: most moderate, some fast (up to ~50 % over the limit, hard on the pedals)
        const ag = calm ? 1 : 0.75 + 0.65 * Math.random() ** 1.5;
        const v: Vehicle = { t, kind: pool[0].kind, route: [], hist: [], s: 0, v: 0, ag,
          f: calm ? 0.95 : 0.95 + 0.5 * (ag - 0.75) / 0.65 + 0.1 * Math.random(),
          grant: null, inJunction: null, wait: 0, arrive: 0, inst: [], stuck: 0, trip: [], dest: null };
        if (type === 'car') this.newTrip(v, true); else this.place(v, pool);
        this.vehicles.push(v);
        for (const x of t.secs) counts2.set(x.model, (counts2.get(x.model) ?? 0) + 1);
        const hex = t === TYPES.van ? (Math.random() < 0.65 ? VAN_WHITE : carPaint())
          : t === TYPES.bus ? BUS_COLORS[Math.floor(Math.random() * BUS_COLORS.length)]
            : t === TYPES.t3000 || t === TYPES.t4000 ? TRAM_SILVER : t === TYPES.train ? TRAIN : carPaint();
        const c = new THREE.Color(hex);
        paints.push([c.r, c.g, c.b]);
      }
    }
    // one instanced mesh per model; per-instance paint
    const meshes = new Map<string, THREE.InstancedMesh>();
    for (const [model, n] of counts2) {
      const geo = MODELS[model].geo.clone();
      geo.setAttribute('paint', new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3));
      const m = new THREE.InstancedMesh(geo, material, n);
      m.frustumCulled = false;
      m.count = 0;
      meshes.set(model, m);
      this.group.add(m);
    }
    this.vehicles.forEach((v, k) => {
      for (const x of v.t.secs) {
        const mesh = meshes.get(x.model)!, i = mesh.count++;
        (mesh.geometry.getAttribute('paint') as THREE.BufferAttribute).setXYZ(i, paints[k][0], paints[k][1], paints[k][2]);
        v.inst.push({ mesh, i });
      }
    });
    this.meshes = [...meshes.values()];
    // --- signal heads: a small box at the right of each stop line ---
    const heads: { pos: THREE.Vector3; sig: Signal }[] = [];
    const right = (p: Path, s: number, d: number) => {
      const h = heading(p, s);
      return sample(p, s, new THREE.Vector3()).add(new THREE.Vector3(-h.y * d, 2.6, h.x * d));
    };
    for (const p of this.paths) {
      if (p.approach?.signal) heads.push({ pos: right(p, p.end, 2.2), sig: p.approach.signal });
      for (const x of p.crossings) heads.push({ pos: right(p, x.s, 2.2), sig: x.sig });
    }
    this.lights = new THREE.InstancedMesh(new THREE.BoxGeometry(0.35, 0.9, 0.35), new THREE.MeshBasicNodeMaterial(),
      Math.max(1, heads.length));
    this.lights.frustumCulled = false;
    heads.forEach((h, i) => {
      this.lights.setMatrixAt(i, new THREE.Matrix4().makeTranslation(h.pos.x, h.pos.y, h.pos.z));
      this.lights.setColorAt(i, LIGHT_COLORS[GREEN]);
      this.lightAt.push({ sig: h.sig, i, state: GREEN });
    });
    this.lights.count = heads.length;
    this.update(0);
  }

  get count() { return this.vehicles.length; }
  get signalCount() { return this.lightAt.length; }

  /** put a vehicle somewhere on a random lane of its kind */
  private place(v: Vehicle, pool?: Path[]) {
    // trams / trains come in at the region edge (as if from the rest of the line); else anywhere
    if (!pool) {
      const edge = this.paths.filter((p) => p.kind === v.kind && this.boundary(p.lane!.from));
      pool = edge.length ? edge : this.paths.filter((p) => p.kind === v.kind && p.end - p.start > Math.min(v.t.total, 60));
    }
    const p = pool[Math.floor(Math.random() * pool.length)];
    // the whole vehicle on the lane where possible (carriages behind the start would fold up)
    const room = p.end - p.start - v.t.total;
    const edge = this.boundary(p.lane!.from);
    v.route = [p]; v.hist = []; v.v = p.speed * 0.5;
    v.s = edge ? p.start : room > 0 ? p.start + v.t.total + Math.random() * room : p.end;
    v.grant = null; v.wait = 0;
    if (v.inJunction) v.inJunction.junction?.occ.delete(v);
    v.inJunction = null;
    this.plan(v);
  }

  private meshes: THREE.InstancedMesh[] = [];

  /** car paint: the measured palette (lifted a little: aerial colours are shaded roofs) plus a few accents */
  private paintPicker(palette: { rgb: number[]; share: number }[]) {
    const cols = palette.map((p) => ({ hex: new THREE.Color().setRGB(
      Math.min(1, p.rgb[0] / 255 * 1.25), Math.min(1, p.rgb[1] / 255 * 1.25), Math.min(1, p.rgb[2] / 255 * 1.25),
      THREE.SRGBColorSpace).getHex(), w: p.share * 0.9 }));
    for (const a of ACCENTS) cols.push({ hex: a, w: (palette.length ? 0.1 : 1) / ACCENTS.length });
    const total = cols.reduce((a, c) => a + c.w, 0);
    return () => { let r = Math.random() * total; for (const c of cols) { r -= c.w; if (r <= 0) return c.hex; } return cols[0].hex; };
  }

  /** is this vehicle waiting for a light (itself, or at the back of a queue whose head is)? */
  private atLight(v: Vehicle) {
    let h = v;
    for (let n = 0; n < 40 && h.block === 'leader' && h.blockBy; n++) h = h.blockBy;
    return h.block === 'junction:light' || h.block === 'crossing';
  }

  /** for the shadow pass: let every vehicle mesh render in another layer too */
  enableLayer(layer: number) { for (const m of this.meshes) m.layers.enable(layer); }

  private kerb: Path[] = []; private kerbTotal = 0;
  private entries: Path[] = []; private exits: Path[] = [];
  trips = 0; failed = 0;

  private kerbSpot(): { p: Path; s: number } {
    let r = Math.random() * this.kerbTotal;
    for (const p of this.kerb) {
      r -= p.end - p.start;
      // keep away from the intersections at both ends
      if (r <= 0) return { p, s: Math.min(p.end - 15, Math.max(p.start + 8, p.end + r)) };
    }
    const p = this.kerb[0];
    return { p, s: (p.start + p.end) / 2 };
  }

  /** start a new trip: from the kerb or the region edge, to the kerb or the region edge */
  private newTrip(v: Vehicle, initial = false) {
    if (v.kind !== KIND.road) { this.place(v); return; }
    for (let attempt = 0; attempt < 8; attempt++) {
      const fromEdge = !initial && this.entries.length && Math.random() < THROUGH;
      const o = fromEdge ? { p: this.entries[Math.floor(Math.random() * this.entries.length)], s: 0 } : this.kerbSpot();
      if (fromEdge) o.s = o.p.start;
      // keep clear of vehicles already there
      if (o.p.occ.some((y) => y !== v && Math.abs(y.s - o.s) < 10)) continue;
      const toEdge = this.exits.length && Math.random() < THROUGH;
      const dest = toEdge ? null : this.kerbSpot();
      const goal = toEdge ? this.exits[Math.floor(Math.random() * this.exits.length)] : dest!.p;
      if (goal === o.p && (!dest || dest.s <= o.s + 20)) continue;
      const lanes = this.route(o.p, goal);
      if (!lanes) { this.failed++; continue; }
      if (v.inJunction) v.inJunction.junction?.occ.delete(v);
      v.inJunction = null; v.grant = null; v.wait = 0; v.stuck = 0;
      v.route = [o.p]; v.hist = []; v.s = o.s; v.v = fromEdge || initial ? o.p.speed * 0.6 : 0;   // from the kerb: pull out
      v.trip = lanes.slice(1); v.dest = dest;
      this.trips++;
      this.plan(v);
      return;
    }
    this.place(v);       // nothing found: anywhere, no destination (rare)
    v.trip = []; v.dest = null;
  }

  /** A* over lanes (directed: one-ways included), cost = travel time + turn penalties */
  private route(from: Path, to: Path): Path[] | null {
    const goal = sample(to, to.end, new THREE.Vector3());
    const cost = new Map<Path, number>([[from, 0]]), prev = new Map<Path, Path>();
    const open = new Heap<Path>();
    open.push(0, from);
    const pos = new THREE.Vector3(), done = new Set<Path>();
    while (open.size) {
      const a = open.pop()!;
      if (done.has(a)) continue;
      done.add(a);
      if (a === to) {
        const out = [a];
        for (let p = prev.get(a); p; p = prev.get(p)) out.unshift(p);
        return out;
      }
      const da = heading(a, a.end);
      for (const b of this.byNode.get(a.lane!.to) ?? []) {
        if (b.kind !== a.kind || b.end - b.start < 0.5) continue;
        const db = heading(b, b.start);
        const turn = Math.atan2(da.x * db.y - da.y * db.x, da.dot(db));   // > 0: right (x east, z south)
        const pen = b.lane!.edge === a.lane!.edge ? 60 : turn < -0.5 ? 4 : turn > 0.5 ? 1.5 : 0;
        // congestion: every vehicle already on the lane costs time (drivers avoid queues)
        const c = cost.get(a)! + (b.end - b.start + 10) / b.speed + pen + b.occ.length * QUEUE_COST;
        if (c < (cost.get(b) ?? Infinity)) {
          cost.set(b, c); prev.set(b, a);
          sample(b, b.end, pos);
          open.push(c + pos.distanceTo(goal) / 14, b);
        }
      }
    }
    return null;
  }

  /** extend the route (cars: along their trip; trams, trains: random) with the curves joining lanes */
  private plan(v: Vehicle) {
    let ahead = v.route[0].end - v.s;
    for (let i = 1; i < v.route.length; i++) ahead += v.route[i].end - v.route[i].start;
    while (ahead < PLAN) {
      const last = v.route[v.route.length - 1];
      if (!last.lane) return;   // (always ends on a lane)
      let next: Path | undefined;
      if (v.kind === KIND.road) {
        next = v.trip.shift();
        if (!next) return;                          // the trip ends on this lane
      } else {
        // rails: only connections a switch allows (no sharp bends, no reversing); none: end of the line
        const dir = heading(last, last.end);
        const pick = (this.byNode.get(last.lane.to) ?? []).filter((n) => n.kind === last.kind && n.end - n.start > 1
          && n.lane!.edge !== last.lane!.edge && dir.dot(heading(n, n.start)) > SWITCH_COS);
        if (!pick.length) return;                   // leaves the network: handled when it gets there
        next = pick[Math.floor(Math.random() * pick.length)];
      }
      let curve = last.next.get(next);
      if (!curve) {
        curve = makeCurve(last, next);
        curve.junction = last.approach?.junction ?? null;
        last.next.set(next, curve);
      }
      v.route.push(curve, next);
      ahead += curve.length + next.end - next.start;
    }
  }

  private conflicts(j: Junction, a: Path, b: Path) {
    let m = j.conflicts.get(a);
    if (!m) j.conflicts.set(a, m = new Map());
    let c = m.get(b);
    if (c === undefined) { c = conflict(a, b); m.set(b, c); }
    return c;
  }

  /** does vehicle y (approaching curve cy) go before x (curve cx)? */
  private before(y: Vehicle, cy: Path, x: Vehicle, cx: Path) {
    const ay = cy.from!.approach!, ax = cx.from!.approach!;
    if (!ax.junction.signalled) {
      if (ay.yields !== ax.yields) return ax.yields;           // give way / stop signs
      if (ay.rank !== ax.rank) return ay.rank > ax.rank;       // main road first
    }
    const dot = ax.dir.dot(ay.dir);
    if (dot < -0.7) {                                          // oncoming: straight before turning
      const ty = Math.abs(cy.turn), tx = Math.abs(cx.turn);
      if (Math.abs(ty - tx) > 0.3) return ty < tx;
    } else if (!ax.junction.signalled && dot < 0.7) {
      // crossing paths: y comes from x's right if y travels towards x's left
      return ay.dir.x * -ax.dir.y + ay.dir.y * ax.dir.x < 0;
    }
    return y.arrive < x.arrive;
  }

  /** is there room on the lane after junction curve c for vehicle v (counting vehicles still in curves into it)? */
  private room(v: Vehicle, c: Path) {
    const out = c.to!, need = v.t.total + v.t.s0;
    let free = out.end - out.start;
    const last = out.occ[0];                       // rearmost vehicle on the exit lane
    if (last) free = last.s - last.t.total - out.start;
    for (const y of c.junction!.occ) if (y !== v && y.inJunction?.to === out) free -= y.t.total + y.t.s0;
    return free > need || (last !== undefined && last.v > 2) || (out.end - out.start < need && !last);   // a moving queue makes room
  }

  private pos = new THREE.Vector3(); private front = new THREE.Vector3(); private rear = new THREE.Vector3();
  private m4 = new THREE.Matrix4(); private q = new THREE.Quaternion(); private e = new THREE.Euler(0, 0, 0, 'YZX');
  private scl = new THREE.Vector3();

  /** distance from the front bumper to the start of route[i] */
  private distTo(v: Vehicle, i: number) {
    let d = v.route[0].end - v.s;
    for (let k = 1; k < i; k++) d += v.route[k].end - v.route[k].start;
    return d;
  }

  update(dt: number) {
    this.time += dt;
    for (const sig of this.signals) sig.state = signalState(sig, this.time);
    // occupancy per path, sorted by position
    const used = new Set<Path>();
    for (const v of this.vehicles) { const p = v.route[0]; if (!used.has(p)) { used.add(p); p.occ.length = 0; } p.occ.push(v); }
    for (const p of used) if (p.occ.length > 1) p.occ.sort((a, b) => a.s - b.s);
    this.lastUsed.forEach((p) => { if (!used.has(p)) p.occ.length = 0; });
    this.lastUsed = used;

    // junction grants: in order of arrival, a vehicle near its next junction curve may enter if nothing
    // conflicting is in there and no conflicting vehicle with precedence is about to enter
    const cands: { v: Vehicle; c: Path; d: number; free: boolean }[] = [];
    const committed: { v: Vehicle; c: Path }[] = [];
    for (const v of this.vehicles) {
      const i = v.route.findIndex((p, k) => k > 0 && p.junction);
      if (i < 0) continue;
      const d = this.distTo(v, i);
      // a grant is final once the vehicle can no longer stop before the junction
      if (v.grant === v.route[i] && d < (v.v * v.v) / (2 * v.t.b * 1.5) + 0.5) { committed.push({ v, c: v.route[i] }); continue; }
      v.grant = null;
      if (d < (v.v * v.v) / (2 * v.t.b) + 15) cands.push({ v, c: v.route[i], d, free: true });
      else { v.arrive = this.time; v.wait = 0; }
    }
    // pass 1: hard constraints (light, something conflicting inside, no room after the junction)
    for (const x of cands) {
      if (x.v.kind !== KIND.road) continue;
      const j = x.c.junction!, sig = x.c.from!.approach!.signal;
      const canStop = x.d > (x.v.v * x.v.v) / (2 * x.v.t.b * 1.5);
      x.v.why = '';
      if (sig && !(sig.state === GREEN || (sig.state === AMBER && !canStop))) x.v.why = 'light';
      else if ([...j.occ].some((y) => y !== x.v && y.inJunction && this.conflicts(j, x.c, y.inJunction))) x.v.why = 'occupied';
      else if (committed.some((y) => y.v !== x.v && y.c.junction === j && this.conflicts(j, x.c, y.c))) x.v.why = 'committed';
      else if (x.v.wait < NO_ROOM_PATIENCE && !this.room(x.v, x.c)) x.v.why = 'no room';   // don't block the box
      x.free = !x.v.why;
    }
    // pass 2: among the vehicles that could go, precedence (in order of arrival)
    cands.sort((a, b) => a.v.arrive - b.v.arrive);
    for (const x of cands) {
      // deadlock breaker: waited long enough for something other than a light -> go
      if (x.v.kind === KIND.road && x.v.wait > FORCE && x.v.why !== 'light') { x.v.grant = x.c; continue; }
      if (!x.free) continue;
      if (x.v.kind !== KIND.road) { x.v.grant = x.c; continue; }   // trams / trains: their own signals, not modelled
      const j = x.c.junction!, signalled = !!x.c.from!.approach!.signal;
      let ok = true;
      for (const y of cands) {
        if (y === x || !y.free || y.c.junction !== j || !this.conflicts(j, x.c, y.c)) continue;
        // only vehicles that will actually be there soon matter (not one far back or standing still)
        if (y.d / Math.max(y.v.v, 1.5) > ETA) continue;
        if (y.v.grant === y.c) { ok = false; x.v.why = 'granted other'; break; }   // allowed in before x
        const patient = !signalled && x.v.wait > PATIENCE && x.v.wait >= y.v.wait;   // breaks priority cycles
        if (!patient && this.before(y.v, y.c, x.v, x.c)) { ok = false; x.v.why = 'yield'; break; }
      }
      if (ok) x.v.grant = x.c;
    }

    // IDM per vehicle
    for (const v of this.vehicles) {
      const t = v.t;
      // desired speed: the speed limit, lowered ahead of curves so that braking is comfortable
      let v0 = v.route[0].speed * v.f, gap = Infinity, vLead = 0;
      v.block = null; v.blockBy = null;
      let d = -v.s;   // distance from the front bumper to s = 0 of route[i]
      for (let i = 0; i < v.route.length && d < LOOK; i++) {
        const p = v.route[i];
        const from = i === 0 ? v.s : p.start;
        for (let k = 0; k < p.cum.length; k++) {             // curves ahead
          const s = p.cum[k];
          if (s < from || s > p.end) continue;
          const dd = d + s;
          if (dd > LOOK) break;
          // curve limits scale with sqrt(lateral acceleration); brake for them late and firmly
          const vm = Math.min(p.vmax[k] * Math.sqrt(v.ag), p.speed * v.f);
          v0 = Math.min(v0, Math.sqrt(vm * vm + 2 * t.b * v.ag * 0.8 * Math.max(0, dd - t.total / 2)));
        }
        if (gap === Infinity) {                              // vehicle ahead on this path
          for (const o of p.occ) {
            if (o === v || o.s <= from) continue;
            gap = d + o.s - o.t.total; vLead = o.v; v.block = 'leader'; v.blockBy = o;
            break;
          }
        }
        // stop lines: a junction curve we may not enter yet, crossing lights
        if (i > 0 && p.junction && v.grant !== p && d + p.start < gap) { gap = d + p.start; vLead = 0; v.block = 'junction:' + (v.why || '?'); v.blockBy = null; }
        if (v.kind === KIND.road)
          for (const x of p.crossings)
            if (x.s > from && x.sig.state !== GREEN && d + x.s < gap
              && (x.sig.state === RED || d + x.s > (v.v * v.v) / (2 * t.b * 1.5))) { gap = d + x.s; vLead = 0; v.block = 'crossing'; v.blockBy = null; }
        // destination: pull over at the kerb spot
        if (v.dest && p === v.dest.p && i === v.route.length - 1 && d + v.dest.s < gap) { gap = d + v.dest.s + t.s0; vLead = 0; }
        d += p.end - (v.route[i + 1]?.start ?? 0);
      }
      // IDM acceleration
      const vv = v.v, dv = vv - vLead, amax = t.amax * v.ag, b = t.b * v.ag;
      const sStar = t.s0 + vv * t.T / v.ag + (vv * dv) / (2 * Math.sqrt(amax * b));
      let a = amax * (1 - Math.pow(vv / Math.max(v0, 0.1), 4));
      if (gap < Infinity) a -= amax * Math.pow(Math.max(sStar, 0) / Math.max(gap - 0.5, 0.1), 2);
      v.v = Math.max(0, vv + Math.max(a, -9) * dt);
      v.wait = v.v < 0.5 && v.route.some((p, k) => k > 0 && p.junction && v.grant !== p) ? v.wait + dt : 0;
      v.stuck = v.v < 0.2 && !this.atLight(v) ? v.stuck + dt : 0;
      if (v.stuck > STUCK) { this.newTrip(v); continue; }                  // gridlock safety valve
      if (v.dest && v.route.length === 1 && v.route[0] === v.dest.p && v.s > v.dest.s - 2 && v.v < 1) {
        this.newTrip(v);                                                    // arrived: park, next trip
        continue;
      }
      v.s += v.v * dt;
      this.advance(v);
    }

    // boxes: two axles on the path per box
    for (const v of this.vehicles) {
      const t = v.t;
      let fb = v.s;                                         // front bumper of the section
      t.secs.forEach((x, k) => {
        this.at(v, fb - x.over, this.front);
        this.at(v, fb - x.over - x.wb, this.rear);
        const dir = tmp.subVectors(this.front, this.rear);
        dir.divideScalar(dir.length() || 1);
        this.pos.copy(this.front).addScaledVector(dir, x.over - x.len / 2);   // body centre, along the axle line
        this.e.set(0, Math.atan2(-dir.z, dir.x), Math.asin(Math.max(-1, Math.min(1, dir.y))));
        this.q.setFromEuler(this.e);
        this.pos.y += x.h / 2;
        this.m4.compose(this.pos, this.q, this.scl.set(1, 1, 1));
        v.inst[k].mesh.setMatrixAt(v.inst[k].i, this.m4);
        fb -= x.len + t.gap;
      });
    }
    for (const m of this.meshes) m.instanceMatrix.needsUpdate = true;
    // signal heads
    let changed = false;
    for (const l of this.lightAt)
      if (l.state !== l.sig.state) { l.state = l.sig.state; this.lights.setColorAt(l.i, LIGHT_COLORS[l.state]); changed = true; }
    if (changed && this.lights.instanceColor) this.lights.instanceColor.needsUpdate = true;
  }
  private lastUsed = new Set<Path>();

  /** move on to the next path when the front passes the end of the current one */
  private advance(v: Vehicle) {
    this.plan(v);   // keep the planned route PLAN m ahead (long lanes: it has to grow before the lane ends)
    while (v.s > v.route[0].end) {
      const over = v.s - v.route[0].end;
      v.hist.unshift(v.route.shift()!);
      if (v.hist.length > 6) v.hist.pop();
      if (!v.route.length) { this.newTrip(v); return; }    // left the region (or the network): next trip
      const p = v.route[0];
      v.s = p.start + over;
      if (p.junction) {                                     // entering a junction curve
        if (v.inJunction) v.inJunction.junction?.occ.delete(v);
        v.inJunction = p; p.junction.occ.add(v); v.grant = null;
      }
      this.plan(v);
    }
    // leave the junction once the rear is out of the curve
    if (v.inJunction && v.route[0] !== v.inJunction) {
      const behind = v.route[0] === v.inJunction.to ? v.s - v.route[0].start : Infinity;
      if (behind > v.t.total) { v.inJunction.junction?.occ.delete(v); v.inJunction = null; }
    }
  }

  /** point at distance s along the vehicle's route (s may be behind the current path: history) */
  private at(v: Vehicle, s: number, out: THREE.Vector3) {
    let p = v.route[0];
    let h = 0;
    while (s < p.start && h < v.hist.length) {
      const prev = v.hist[h++];
      s = prev.end - (p.start - s);
      p = prev;
    }
    if (s < 0) {   // before the start of what we know (just entered the region): continue straight back
      sample(p, 0, out);
      const d = sample(p, Math.min(2, p.length), tmp2).sub(out).normalize();
      return out.addScaledVector(d, s);
    }
    return sample(p, s, out);
  }
}
