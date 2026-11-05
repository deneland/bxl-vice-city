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
//  lights      signalled junctions and stop lines from roads.py (OSM lights as evidence): junctions cycle
//              two approach groups; pedestrian-crossing lights turn red now and then; crosswalks under
//              lights run with them; drawn as small coloured boxes at the stop line
//
//  trips       each car drives a trip: from the kerb (or the region edge: through traffic) to a destination
//              (kerb or region edge), routed by A* on the directed lane graph (one-ways built in; left
//              turns and U-turns cost extra); at the destination it pulls over and a new trip starts
//
// Trams and trains use the same model on their own tracks (they ignore road signals and do not interact
// with cars yet): STIB trams and buses run their timetable (transit.ts), trains choose tracks at random.
//
// The paths and their geometry are in paths.ts, the vehicle types in vehicleTypes.ts.
import * as THREE from 'three/webgpu';
import { CONTROL, KIND, type Lane } from './roads';
import { MinHeap } from './heap';
import { PointGrid } from './grid';
import { Fleet } from './fleet';
import type { Agent } from './agents';
import { AMBER, GREEN, LIGHT_COLORS, RED, conflict, heading, longHeading, makeCurve, makePath, sample, signalState,
  type Junction, type Path, type Signal, type Spot, type Vehicle } from './paths';
import { ACCENTS, BUS_COLORS, MODELS, STIB_BUS, TRAIN, TRAM_SILVER, TYPES, VAN_WHITE, pickKind, type Kind } from './vehicleTypes';

const PLAN = 120;            // m of route planned ahead
const LOOK = 90;             // m looked ahead for obstacles and curves
const BUMP = 20 / 3.6;       // m/s over a speed bump / raised table
const APPROACH = 30;         // m: signals / signs this close to a lane's end belong to the junction
const PATIENCE = 6;          // s waiting at a junction before ignoring priority (if nothing is in the way)
const NO_ROOM_PATIENCE = 8;  // s before entering even without room after the junction
const FORCE = 20;            // s waiting at a junction (not for a light): go anyway, even through a conflict
const STUCK = 90;            // s without moving (not queueing for a light): taken off, new trip (last resort)
const SWITCH_COS = Math.cos(50 * Math.PI / 180);   // trams / trains: sharpest track connection
const ETA = 4;               // s: give way only to conflicting vehicles arriving within this time
const QUEUE_COST = 6;        // s of routing cost per vehicle already on a lane
const KERB_REACH = 160;      // m: farthest a place may be from the kerb spot it is reached from
const TRIM = 9;              // m: lanes end this far before the intersection; curves turn in that space


const byS = (a: { s: number }, b: { s: number }) => a.s - b.s;
/** index of the next junction curve on a route after the current path, or -1 */
function nextJunction(route: Path[]) { for (let k = 1; k < route.length; k++) if (route[k].junction) return k; return -1; }
/** a junction curve ahead that the vehicle may not enter yet */
function waitingAtJunction(v: Vehicle) {
  for (let k = 1; k < v.route.length; k++) if (v.route[k].junction && v.grant !== v.route[k]) return true;
  return false;
}
const BLOCK = new Map<string, string>();
/** 'junction:<reason>' (cached: built per vehicle per step otherwise) */
function junctionBlock(why: string | undefined) {
  const k = why || '?';
  let b = BLOCK.get(k);
  if (!b) { b = 'junction:' + k; BLOCK.set(k, b); }
  return b;
}
/** distance a vehicle needs to stop, braking firmly (1.5 x its comfortable deceleration) */
const stopDistance = (v: Vehicle) => (v.v * v.v) / (2 * v.t.b * 1.5);
const STEP = 1 / 20;   // s: simulation step (drawing interpolates in between)

const tmp = new THREE.Vector3(), tmp2 = new THREE.Vector3();

export class Traffic {
  readonly lights: THREE.InstancedMesh;
  readonly vehicles: Vehicle[] = [];
  private paths: Path[] = [];
  private byNode = new Map<number, Path[]>();     // node -> lanes leaving it
  private signals: Signal[] = [];
  private lightAt: { sig: Signal; i: number; state: number }[] = [];
  private time = Math.random() * 1000;

  /** material: lit like the city; instance colours are the paint */
  readonly group = new THREE.Group();

  /** material: reads the model attributes vcol / vmat and the per-instance paint (deferred.ts vehicleMaterial);
   *  boundary: is this node where the network leaves the region (entry / exit for through traffic)?;
   *  palette: car paint colours (sRGB 0-255) with their shares (pipeline vehicles.py) */
  constructor(lanes: Lane[], counts = { car: 400, tram: 12, train: 3 }, material: THREE.Material,
    private boundary: (node: number) => boolean = () => false, palette: { rgb: number[]; share: number }[] = []) {
    // --- paths: lanes trimmed back from the junctions, so the curves have room to turn ---
    const incoming = new Map<number, Path[]>();
    for (const l of lanes) {
      const p = makePath(l.pts, l.speed, l.kind, l.cum);
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
    // Which junctions have lights and where each road stops for them comes from roads.py (assign_signals:
    // OSM's lights as evidence): a signalled node, stop lines (CONTROL.signals) on the lanes into it; every
    // road into a signalled junction runs with its phases (stop line at its mouth where OSM has none).
    const phasesAt = new Map<number, { ref: THREE.Vector2; groups: Signal[] }>();   // signalled junction node ->
    for (const ins of incoming.values()) {
      const j: Junction = { occ: new Set(), conflicts: new Map(), signalled: false };
      for (const p of ins) {
        const near = p.lane!.controls.filter((c) => c.s > p.length - APPROACH);
        if (p.kind === KIND.road && p.lane!.toSignal) j.signalled = true;
        // (the first stop line the car meets: OSM sometimes maps one at the stop line and one at the crosswalk)
        const light = p.lane!.controls.find((c) => c.code === CONTROL.signals);
        p.approach = { junction: j, rank: p.lane!.rank, dir: heading(p, p.end), signal: null,
          yields: near.some((c) => c.code === CONTROL.giveWay || c.code === CONTROL.stop),
          lightS: light ? Math.min(Math.max(light.s, p.start), p.end) : p.end };
      }
      if (!j.signalled) continue;
      // two groups of approaches by axis: green for one, then the other (3 s amber, 2 s all red)
      const roads = ins.filter((p) => p.kind === KIND.road);
      if (!roads.length) continue;
      const ref = roads[0].approach!.dir;
      const g0 = 18 + Math.random() * 8, g1 = 15 + Math.random() * 8, cycle = g0 + g1 + 10, offset = Math.random() * 60;
      const groups: Signal[] = [
        { cycle, offset, phases: [[0, g0]], state: GREEN },
        { cycle, offset, phases: [[g0 + 5, g0 + 5 + g1]], state: RED },
      ];
      this.signals.push(...groups);
      for (const p of roads) p.approach!.signal = groups[Math.abs(p.approach!.dir.dot(ref)) > 0.6 ? 0 : 1];
      phasesAt.set(ins[0].lane!.to, { ref, groups });
    }
    // pedestrian-crossing lights away from junctions (roads.py; one signal per place, both directions)
    const crossingSig: { q: THREE.Vector3; sig: Signal }[] = [];
    for (const p of this.paths) {
      if (p.kind !== KIND.road) continue;
      for (const c of p.lane!.controls)
        if (c.code === CONTROL.crossingLight) {
          const s = Math.min(Math.max(c.s, p.start), p.end), q = sample(p, s, new THREE.Vector3());
          let sig = crossingSig.find((x) => x.q.distanceTo(q) < 8)?.sig;
          if (!sig) {
            const g = 35 + Math.random() * 35;
            sig = { cycle: g + 15, offset: Math.random() * 100, phases: [[0, g]], state: GREEN };
            this.signals.push(sig); crossingSig.push({ q, sig });
          }
          p.crossings.push({ s, sig });
        }
      // mapped crosswalks: cars give way to people on them (those in the junction's trimmed ends: at its edge)
      for (const z of p.lane!.crosswalks) {
        const zs = Math.min(Math.max(z.s, p.start + 0.5), p.end - 0.5);
        p.zebras.push({ s: zs, key: z.key });
        if (!this.zebraAt.has(z.key)) this.zebraAt.set(z.key, []);
        this.zebraAt.get(z.key)!.push({ p, s: zs });
      }
    }
    for (const p of this.paths)
      if (p.kind === KIND.road) p.bumps = p.lane!.controls.filter((c) => c.code === CONTROL.calming).map((c) => c.s);
    // crosswalks under lights: people walk when the cars over them have red. At a signalled junction: the
    // phase of the road it crosses (on an approach, between its stop line and the junction; on an exit, the
    // group of that road's axis); away from junctions: the crossing light there
    for (const [key, list] of this.zebraAt) {
      let sig: Signal | null = null;
      for (const { p, s } of list) {
        if (p.approach?.signal && s >= p.approach.lightS - 3) { sig = p.approach.signal; break; }
        const ph = phasesAt.get(p.lane!.from);
        if (ph && s < p.start + APPROACH) { sig = ph.groups[Math.abs(heading(p, p.start).dot(ph.ref)) > 0.6 ? 0 : 1]; break; }
        const x = p.crossings.find((c) => Math.abs(c.s - s) < 12);
        if (x) { sig = x.sig; break; }
      }
      if (sig) this.walkSig.set(key, sig);
    }
    // --- trip ends: kerbside and region-edge entries / exits ---
    this.kerb = this.paths.filter((p) => p.kind === KIND.road && p.end - p.start > 30 && !p.lane!.bikeOnly);
    this.entries = this.paths.filter((p) => p.kind === KIND.road && this.boundary(p.lane!.from) && p.end - p.start > 2 && !p.lane!.bikeOnly);
    this.exits = this.paths.filter((p) => p.kind === KIND.road && this.boundary(p.lane!.to) && !p.lane!.bikeOnly);
    // --- vehicle meshes: one instanced mesh per model, a pool of instances (cars come and go) ---
    this.fleet = new Fleet(MODELS, material);
    this.group.add(this.fleet.group);
    this.carPaint = this.paintPicker(palette);
    // --- trams and trains: always on the network ---
    const byKind = (k: number) => this.paths.filter((p) => p.kind === k && p.end - p.start > 10);
    for (const [kindNo, n] of [[KIND.tram, counts.tram], [KIND.rail, counts.train]] as const) {
      const pool = byKind(kindNo);
      if (!pool.length) continue;
      for (let i = 0; i < n; i++) {
        const t = kindNo === KIND.tram ? (Math.random() < 0.3 ? TYPES.t4000 : TYPES.t3000) : TYPES.train;
        const v = this.make(t, kindNo, kindNo === KIND.tram ? TRAM_SILVER : TRAIN, 1, 0.95, null);
        this.place(v, pool);
      }
    }
    // --- signal heads: a small box at the right of each stop line ---
    const heads: { pos: THREE.Vector3; sig: Signal }[] = [];
    const right = (p: Path, s: number, d: number) => {
      const h = heading(p, s);
      return sample(p, s, new THREE.Vector3()).add(new THREE.Vector3(-h.y * d, 2.6, h.x * d));
    };
    for (const p of this.paths) {
      if (p.approach?.signal) heads.push({ pos: right(p, p.approach.lightS, 2.2), sig: p.approach.signal });   // (where OSM has it)
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
    this.step(0); this.pose(0);
  }

  get count() { return this.vehicles.length; }

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

  /** draws the vehicles: what the camera sees, at a level of detail for the distance (fleet.ts) */
  private fleet!: Fleet;
  private carPaint!: () => number;
  /** a STIB vehicle has stopped at stop `id` (transit.ts stop index): people board / alight */
  onDwell: ((id: number) => void) | null = null;
  /** called when a car's trip ends (arrived, left the region, or given up when stuck) */
  onArrive: ((agent: Agent, arrived: boolean) => void) | null = null;

  /** a new vehicle with instances for its sections */
  private make(t: Kind, kind: number, paint: number, ag: number, f: number, agent: Agent | null): Vehicle {
    const c = new THREE.Color(paint);
    const v: Vehicle = { t, kind, route: [], hist: [], s: 0, v: 0, ag, f, grant: null, inJunction: null, wait: 0,
      arrive: 0, inst: [], stuck: 0, trip: [], dest: null, agent };
    for (const x of t.secs) v.inst.push(this.fleet.add(x.model, c, 1));
    this.vehicles.push(v);
    if (agent) this.byAgent.set(agent, v);
    return v;
  }
  private byAgent = new Map<Agent, Vehicle>();   // agent -> the vehicle they drive

  /** take a vehicle off the road: free its instances and junction */
  private release(v: Vehicle) {
    for (const h of v.inst) this.fleet.remove(h);
    v.inst = [];
    if (v.inJunction) v.inJunction.junction?.occ.delete(v);
    v.inJunction = null;
    if (v.agent && this.byAgent.get(v.agent) === v) this.byAgent.delete(v.agent);
    const k = this.vehicles.indexOf(v);
    if (k >= 0) this.vehicles.splice(k, 1);
  }

  /** a car's trip ends */
  private finish(v: Vehicle, arrived: boolean) {
    const a = v.agent;
    this.release(v);
    if (a) this.onArrive?.(a, arrived);
  }

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

  /** (development) every road junction with all its possible curves and the lane ends around it, to check
   *  the geometry the pipeline produced (bench/junctions.mjs) */
  junctionGeometry() {
    const byTo = new Map<number, Path[]>();
    for (const p of this.paths)
      if (p.lane && p.kind === KIND.road && !p.lane.bikeOnly) {
        if (!byTo.has(p.lane.to)) byTo.set(p.lane.to, []);
        byTo.get(p.lane.to)!.push(p);
      }
    const pt = (p: Path, s: number) => { const v = sample(p, s, new THREE.Vector3()); return [+v.x.toFixed(2), +v.z.toFixed(2)]; };
    const stretch = (p: Path, s0: number, s1: number) => {
      const o: number[][] = [];
      for (let s = Math.max(0, s0); s <= Math.min(p.length, s1); s += 2) o.push(pt(p, s));
      return o;
    };
    const out = [];
    for (const [node, ins] of byTo) {
      const outs = (this.byNode.get(node) ?? []).filter((b) => b.kind === KIND.road && !b.lane!.bikeOnly);
      const curves = [];
      for (const a of ins)
        for (const b of outs) {
          if (b.lane!.edge === a.lane!.edge) continue;   // (U-turns)
          const c = makeCurve(a, b);
          const p0 = pt(a, a.end), p1 = pt(b, b.start);
          curves.push({ from: a.lane!.edge, to: b.lane!.edge, len: +c.length.toFixed(2), chord: +Math.hypot(p1[0] - p0[0], p1[1] - p0[1]).toFixed(2),
            turn: +c.turn.toFixed(3), pts: stretch(c, 0, c.length) });
        }
      // lights: each approach's stop line (OSM's or the mouth), lights just past the junction, and its crosswalks
      const sigId = (g: Signal | null | undefined) => (g ? this.signals.indexOf(g) : -1);
      const zebras = (p: Path, s0: number, s1: number) => p.zebras.filter((z) => z.s >= s0 && z.s <= s1)
        .map((z) => ({ at: pt(p, z.s), walk: sigId(this.walkSig.get(z.key)) }));
      out.push({ node, signalled: !!ins[0].lane!.toSignal,
        ins: ins.map((a) => ({ edge: a.lane!.edge, end: pt(a, a.end), line: stretch(a, a.end - 30, a.end),
          light: a.approach?.signal ? { at: pt(a, a.approach.lightS), back: +(a.end - a.approach.lightS).toFixed(1), group: sigId(a.approach.signal),
            osm: a.lane!.controls.some((c) => c.code === CONTROL.signals) } : null,
          crossings: a.crossings.filter((c) => c.s > a.end - 30).map((c) => pt(a, c.s)),
          zebras: zebras(a, a.end - 30, a.end) })),
        outs: outs.map((b) => ({ edge: b.lane!.edge, start: pt(b, b.start), line: stretch(b, b.start, b.start + 30),
          crossings: b.crossings.filter((c) => c.s < b.start + 25).map((c) => pt(b, c.s)),
          zebras: zebras(b, b.start, b.start + 30) })), curves });
    }
    return out;
  }

  /** for the shadow pass: let every vehicle mesh render in another layer too */
  enableLayer(layer: number) { this.fleet.enableLayer(layer); }
  /** fill the vehicle instance buffers for this camera (once per frame, before rendering) */
  draw(camera: THREE.Camera) { this.fleet.update(camera); }
  private kerb: Path[] = [];   // lanes cars park along (spotNear)
  private entries: Path[] = []; private exits: Path[] = [];

  /** nearest kerb spot to a scene position (in this group's frame), away from intersections */
  spotNear(x: number, z: number, kind: number = KIND.road): Spot | null {
    if (kind !== KIND.road) return this.trackSpot(x, z, kind);
    if (!this.kerbGrid) {   // (built on first use)
      this.kerbGrid = new PointGrid<Spot>(40);
      for (const p of this.kerb)
        for (let s = p.start + 8; s < p.end - 15; s += 6) { sample(p, s, tmp); this.kerbGrid.add(tmp.x, tmp.z, { p, s }); }
    }
    return this.kerbGrid.nearest(x, z, KERB_REACH);
  }
  private kerbGrid: PointGrid<Spot> | null = null;

  /** candidate points on tracks / lanes of this kind near a position, nearest first, one per lane */
  spotsNear(x: number, z: number, kind: number, max = 25): Spot[] {
    const best = new Map<Path, { s: number; d: number }>();
    for (const p of this.paths) {
      if (p.kind !== kind || p.end - p.start < 4) continue;
      for (let s = p.start + 1; s < p.end - 1; s += 2) {
        sample(p, s, tmp);
        const d = (tmp.x - x) ** 2 + (tmp.z - z) ** 2;
        if (d < max * max && d < (best.get(p)?.d ?? Infinity)) best.set(p, { s, d });
      }
    }
    return [...best.entries()].sort((a, b) => a[1].d - b[1].d).slice(0, 4).map(([p, b]) => ({ p, s: b.s }));
  }
  routable(a: Path, b: Path) { return a === b || !!this.stopRoute(a, b); }

  /** shortest way in from the region edge to `p` / out from `p` to the edge (cached per lane) */
  private edgeWay(p: Path, inbound: boolean): Path[] | null {
    const cache = inbound ? this.inCache : this.outCache;
    if (cache.has(p)) return cache.get(p)!;
    // one search for all region-edge lanes at once (inbound: from all of them to p; outbound: from p to the
    // nearest one), instead of a search per edge lane
    const G = this.routeGraph(), ip = G.id.get(p);
    let best: Path[] | null = null;
    if (ip !== undefined) {
      const edge = (i: number) => G.lanes[i].kind === p.kind && (inbound ? this.boundary(G.lanes[i].lane!.from) : this.boundary(G.lanes[i].lane!.to));
      if (inbound) {
        const src: number[] = [];
        for (let i = 0; i < G.lanes.length; i++) if (edge(i)) src.push(i);
        best = src.length ? this.search(src, (i) => i === ip, G.ex[ip], G.ez[ip]) : null;
      } else best = this.search([ip], edge, NaN, NaN);
    }
    cache.set(p, best);
    return best;
  }
  private inCache = new Map<Path, Path[] | null>();
  private outCache = new Map<Path, Path[] | null>();
  /** seconds to drive in from the edge to a stop (for starting runs early) */
  leadIn(sp: Spot, mode: number) {
    const r = this.edgeWay(sp.p, true);
    if (!r) return 0;
    return (r.slice(0, -1).reduce((a, x) => a + x.end - x.start, 0) + sp.s) / (mode === 0 ? 7 : 8) + 10;
  }

  /** nearest point on a track of this kind (tram stops) */
  private trackSpot(x: number, z: number, kind: number): Spot | null {
    let best: { p: Path; s: number; d: number } | null = null;
    for (const p of this.paths) {
      if (p.kind !== kind || p.end - p.start < 4) continue;
      for (let s = p.start + 1; s < p.end - 1; s += 2) {
        sample(p, s, tmp);
        const d = (tmp.x - x) ** 2 + (tmp.z - z) ** 2;
        if (!best || d < best.d) best = { p, s, d };
      }
    }
    return best && best.d < 30 * 30 ? { p: best.p, s: best.s } : null;
  }

  /** Put a STIB tram or bus on the road for its run: it drives from stop to stop and waits at each
   *  until the timetabled departure; it leaves after the last stop. `from`: index of the first stop
   *  to start at (after a jump in time). */
  spawnTransit(info: { line: string; headsign: string; color: string; mode: number },
    stops: { spot: Spot; dep: number; name: string; id: number }[], from = 0): Vehicle | null {
    const st = stops.slice(from);
    if (st.length < 2) return null;
    // in from the region edge (when starting at the first stop), stop to stop, out to the edge
    const inWay = from === 0 ? this.edgeWay(st[0].spot.p, true) : null;
    const lanes: Path[] = inWay ? [...inWay] : [st[0].spot.p];
    for (let i = 0; i + 1 < st.length; i++) {
      const r = this.stopRoute(st[i].spot.p, st[i + 1].spot.p);
      if (!r) return null;
      lanes.push(...r.slice(1));
    }
    const outWay = this.edgeWay(st[st.length - 1].spot.p, false);
    if (outWay) lanes.push(...outWay.slice(1));
    const tram = info.mode === 0;
    const t = tram ? (info.line === '7' ? TYPES.t4000 : TYPES.t3000) : TYPES.bus;
    const v = this.make(t, tram ? KIND.tram : KIND.road, tram ? TRAM_SILVER : STIB_BUS, 1, 1, null);
    v.transit = { ...info, next: 0, stops: st.map((x) => ({ p: x.spot.p, s: x.spot.s, dep: x.dep, name: x.name, id: x.id, arrived: -1, served: false })) };
    v.route = [lanes[0]]; v.trip = lanes.slice(1);
    v.s = inWay ? lanes[0].start : st[0].spot.s; v.v = inWay ? lanes[0].speed * 0.6 : 0;
    this.plan(v);
    return v;
  }

  /** remove all STIB vehicles (after a jump in time) */
  clearTransit() { for (const v of this.vehicles.filter((x) => x.transit)) this.release(v); }

  /** where a spot is in the scene (this group's frame) */
  spotPosition(sp: Spot, out = new THREE.Vector3()) { return sample(sp.p, sp.s, out); }


  /** Put a car on the road for a trip, from a kerb spot or the region edge to a kerb spot or the
   *  region edge. progress (0-1): already that far along (after a jump in time). Returns the route
   *  length in metres, or 0 if there is no route. */
  spawn(agent: Agent, from: Spot | 'edge', to: Spot | 'edge', car: { kind: string; paint: number; ag: number; f: number },
    progress = 0): number {
    for (let attempt = 0; attempt < 4; attempt++) {
      if ((from === 'edge' && !this.entries.length) || (to === 'edge' && !this.exits.length)) return 0;
      const o = from === 'edge' ? (() => { const p = this.entries[Math.floor(Math.random() * this.entries.length)]; return { p, s: p.start }; })() : from;
      const goal = to === 'edge' ? this.exits[Math.floor(Math.random() * this.exits.length)] : to.p;
      if (goal === o.p && to !== 'edge' && to.s <= o.s + 5) return 0;
      // bounded: generous for any real trip (detours, queues), but an unreachable goal gives up early
      // instead of searching the whole network (a frame spike)
      const G = this.routeGraph(), ia = G.id.get(o.p), ib = G.id.get(goal);
      const crow = ia !== undefined && ib !== undefined ? Math.hypot(G.ex[ia] - G.ex[ib], G.ez[ia] - G.ez[ib]) : 0;
      const lanes = this.route(o.p, goal, car.kind === 'bike', crow / 3 + 600);
      if (!lanes) { if (from !== 'edge' && to !== 'edge') return 0; continue; }
      // where along the route (progress > 0: somewhere on the way)
      let k = 0, s = o.s;
      const len = lanes.reduce((a, p, i) => a + (i === 0 ? p.end - o.s : p.end - p.start), 0);
      if (progress > 0) {
        let left = progress * len;
        for (k = 0; k < lanes.length; k++) {
          const from0 = k === 0 ? o.s : lanes[k].start, l = lanes[k].end - from0;
          if (left < l || k === lanes.length - 1) { s = from0 + Math.min(left, l); break; }
          left -= l;
        }
      }
      const t = (TYPES as Record<string, Kind>)[car.kind] ?? TYPES.hatch;
      const v = this.make(t, KIND.road, car.paint, car.ag, car.f, agent);
      v.route = [lanes[k]]; v.s = s; v.v = from === 'edge' || progress > 0 ? lanes[k].speed * 0.6 : 0;   // from the kerb: pull out
      v.trip = lanes.slice(k + 1); v.dest = to === 'edge' ? null : to;
      this.plan(v);
      return len;
    }
    return 0;
  }

  /** someone is on (or stepping onto) the mapped crosswalk with this key (pedestrians.ts) */
  private zebraBusy: ((key: number) => boolean) | null = null;
  setZebras(busy: (key: number) => boolean) { this.zebraBusy = busy; }
  private zebraAt = new Map<number, { p: Path; s: number }[]>();   // crosswalk key -> where it is on each lane
  private walkSig = new Map<number, Signal>();   // crosswalk key -> the car signal it runs against
  /** crosswalk under a light: may people start crossing (the cars over it have red); null: no light there */
  walkSignal(key: number): boolean | null { const s = this.walkSig.get(key); return s ? s.state === RED : null; }
  /** may a pedestrian step onto this crosswalk: no car about to pass over it that could not stop */
  zebraClear(key: number) {
    for (const { p, s } of this.zebraAt.get(key) ?? [])
      for (const v of p.occ) {
        const d = s - v.s;   // front bumper to the crosswalk
        if (d > -v.t.total && d < 3 + (v.v * v.v) / (2 * v.t.b) + v.v * 0.8) return false;
      }
    return true;
  }

  /** the vehicle drawn as instance i of mesh (for picking) */
  vehicleAt(mesh: THREE.Object3D, i: number) {
    const h = this.fleet.handleAt(mesh, i);
    return h < 0 ? null : this.vehicles.find((v) => v.inst.includes(h)) ?? null;
  }
  /** the car an agent drives now, if any */
  vehicleOf(agent: Agent) { return this.byAgent.get(agent) ?? null; }
  /** a vehicle's front bumper position (this group's frame) */
  positionOf(v: Vehicle, out = new THREE.Vector3()) { return this.at(v, v.s, out); }
  /** points along what is left of a vehicle's trip (this group's frame; points = false: none), and its length (m) */
  routeAhead(v: Vehicle, points = true) {
    const pts: number[] = [];
    let len = 0;
    const lanes = [...v.route, ...v.trip.flatMap((l: Path, k: number) => {
      const prev = k === 0 ? v.route[v.route.length - 1] : v.trip[k - 1];
      const c = prev.next.get(l);
      return c ? [c, l] : [l];
    })];
    lanes.forEach((p, k) => {
      const from = k === 0 ? v.s : p.start;
      const to = v.dest && p === v.dest.p && k === lanes.length - 1 ? v.dest.s : p.end;
      if (points) for (let s = from; s <= to; s += 2) { sample(p, s, tmp); pts.push(tmp.x, tmp.y + 0.3, tmp.z); }
      len += Math.max(0, to - from);
    });
    return { pts, len };
  }
  /** a short description of a vehicle that has no agent (trams, trains) */
  describe(v: Vehicle) {
    if (v.transit) return `${v.transit.mode === 0 ? 'Tram' : 'Bus'} ${v.transit.line} → ${v.transit.headsign}`;
    return v.t.name === 't3000' ? 'STIB tram (T3000-style)' : v.t.name === 't4000' ? 'STIB tram (T4000-style)' : 'train'; }
  pickable() { return this.fleet.meshes(); }
  speedOf(v: Vehicle) { return v.v; }

  /** position of the vehicle an agent drives (this group's frame), if any */
  positionOfAgent(agent: Agent) { const v = this.vehicleOf(agent); return v ? this.positionOf(v) : null; }
  /** metres left of an agent's trip */
  routeLeft(agent: Agent) { const v = this.vehicleOf(agent); return v ? this.routeAhead(v, false).len : 0; }

  // --- parked cars: static instances, no simulation ---
  private parked = new Map<object, number[]>();   // parked car (its owner, or a key for an anonymous one) -> bodies
  /** show an agent's car parked at (x, z) with a heading (rad about y), on the ground of the nearest lane */
  park(key: object, car: { kind: string; paint: number }, x: number, z: number, heading: number) {
    if (this.parked.has(key)) return;
    const t = (TYPES as Record<string, Kind>)[car.kind] ?? TYPES.hatch;
    const sp = this.spotNear(x, z);
    const y = sp ? sample(sp.p, sp.s, tmp).y : 0;
    const c = new THREE.Color(car.paint), list: number[] = [];
    for (const x0 of t.secs) {
      const h = this.fleet.add(x0.model, c, 0);   // lights off
      this.m4.compose(tmp2.set(x, y + x0.h / 2, z), this.q.setFromEuler(this.e.set(0, heading, 0)), this.scl.set(1, 1, 1));
      this.fleet.setPose(h, this.m4);
      list.push(h);
    }
    this.parked.set(key, list);
  }
  unpark(key: object) {
    const list = this.parked.get(key);
    if (!list) return;
    for (const h of list) this.fleet.remove(h);
    this.parked.delete(key);
  }
  clearParked() { for (const a of [...this.parked.keys()]) this.unpark(a); }

  /** take an agent's car off the road (e.g. after a jump in time) */
  remove(agent: Agent) {
    const v = this.byAgent.get(agent);
    if (v) this.release(v);
  }

  /** take all cars off the road */
  clearCars() { for (const v of this.vehicles.filter((x) => x.kind === KIND.road && !x.transit)) this.release(v); }

  /** default car model and paint for a new agent */
  randomCar() {
    const t = pickKind();
    const paint = t === TYPES.van ? (Math.random() < 0.65 ? VAN_WHITE : this.carPaint())
      : t === TYPES.bus ? BUS_COLORS[Math.floor(Math.random() * BUS_COLORS.length)] : this.carPaint();
    const calm = t === TYPES.bus;
    // drivers differ: most moderate, some fast (up to ~50 % over the limit, hard on the pedals)
    const ag = calm ? 1 : 0.75 + 0.65 * Math.random() ** 1.5;
    return { kind: t.name, paint, ag, f: calm ? 0.95 : 0.95 + 0.5 * (ag - 0.75) / 0.65 + 0.1 * Math.random() };
  }

  // --- routing graph: lanes as nodes (directed: one-ways included), built once; per search only typed arrays
  // (no allocations); cost = travel time + turn penalties ---
  private rg: {
    id: Map<Path, number>; lanes: Path[]; ex: Float32Array; ez: Float32Array;   // lane end (stop line) positions
    off: Int32Array; to: Int32Array; base: Float32Array;   // successors (CSR) and their static cost (time + turn penalty)
    g: Float64Array; prev: Int32Array; seen: Uint32Array; closed: Uint32Array; stamp: number; heap: MinHeap<number>;
  } | null = null;
  private routeGraph() {
    if (this.rg) return this.rg;
    const lanes = this.paths.filter((p) => p.lane), id = new Map(lanes.map((p, i) => [p, i] as [Path, number]));
    const n = lanes.length, ex = new Float32Array(n), ez = new Float32Array(n), q = new THREE.Vector3();
    lanes.forEach((p, i) => { sample(p, p.end, q); ex[i] = q.x; ez[i] = q.z; });
    const off = new Int32Array(n + 1), to: number[] = [], base: number[] = [];
    lanes.forEach((a, i) => {
      off[i] = to.length;
      const da = heading(a, a.end);
      for (const b of this.byNode.get(a.lane!.to) ?? []) {
        if (b.kind !== a.kind || b.end - b.start < 0.5) continue;
        const db = heading(b, b.start);
        const turn = Math.atan2(da.x * db.y - da.y * db.x, da.dot(db));   // > 0: right (x east, z south)
        const pen = b.lane!.edge === a.lane!.edge ? 60
          : a.kind !== KIND.road ? (da.dot(db) < SWITCH_COS ? 500 : 0)   // rails: only switch-like connections
            : turn < -0.5 ? 4 : turn > 0.5 ? 1.5 : 0;
        to.push(id.get(b)!); base.push((b.end - b.start + 10) / b.speed + pen);
      }
    });
    off[n] = to.length;
    this.rg = { id, lanes, ex, ez, off, to: Int32Array.from(to), base: Float32Array.from(base),
      g: new Float64Array(n), prev: new Int32Array(n), seen: new Uint32Array(n), closed: new Uint32Array(n), stamp: 0, heap: new MinHeap<number>() };
    return this.rg;
  }

  /** A* over the lanes (time + turn penalties + queues); null if `to` cannot be reached */
  private route(from: Path, to: Path, bike = false, maxCost = Infinity): Path[] | null {
    const G = this.routeGraph();
    const s = G.id.get(from), t = G.id.get(to);
    if (s === undefined || t === undefined) return null;
    return this.search([s], (i) => i === t, G.ex[t], G.ez[t], bike, maxCost);
  }

  /** best-first search from any of `sources` to the first lane where `goal` holds; (gx, gz): A* target, or
   *  NaN for none (several goals: plain Dijkstra) */
  private search(sources: number[], goal: (i: number) => boolean, gx: number, gz: number, bike = false, maxCost = Infinity) {
    const G = this.routeGraph();
    const st = ++G.stamp, { g, prev, seen, closed, ex, ez, off, base, lanes, heap } = G;
    const astar = !Number.isNaN(gx);
    heap.clear();
    for (const s of sources) { g[s] = 0; prev[s] = -1; seen[s] = st; heap.push(0, s); }
    while (heap.size) {
      const a = heap.pop()!;
      if (closed[a] === st) continue;
      closed[a] = st;
      if (goal(a)) {
        const out: Path[] = [];
        for (let k = a; k >= 0; k = prev[k]) out.push(lanes[k]);
        return out.reverse();
      }
      for (let e = off[a]; e < off[a + 1]; e++) {
        const b = G.to[e], lb = lanes[b];
        if (lb.lane!.bikeOnly && !bike) continue;
        // congestion: every vehicle already on the lane costs time (drivers avoid queues)
        const c = g[a] + base[e] + lb.occ.length * QUEUE_COST;
        if (c > maxCost) continue;   // (bounded searches give up instead of exploring the whole network)
        if (seen[b] !== st || c < g[b]) {
          g[b] = c; prev[b] = a; seen[b] = st;
          // weighted A* (straight line at 10 m/s, ~1.4x an admissible estimate): far fewer lanes searched
          // on long trips, routes at most slightly longer than the best (drivers are not perfect either)
          heap.push(astar ? c + Math.hypot(ex[b] - gx, ez[b] - gz) / 10 : c, b);
        }
      }
    }
    return null;
  }

  /** route for timetabled vehicles between two stops: the same pairs come back all day, so cached
   *  (without queues: buses and trams keep to their line anyway) */
  private stopRoutes = new Map<string, Path[] | null>();
  /** is the stop-to-stop route between a and b known already (no search needed) */
  hasStopRoute(a: Path, b: Path) {
    const G = this.routeGraph();
    return a === b || this.stopRoutes.has(`${G.id.get(a)}>${G.id.get(b)}`);
  }
  stopRoute(a: Path, b: Path): Path[] | null {
    const G = this.routeGraph();
    const key = `${G.id.get(a)}>${G.id.get(b)}`;
    let r = this.stopRoutes.get(key);
    // bounded: consecutive stops are a few hundred metres apart; a wrong-direction track candidate would
    // otherwise search the whole network before failing
    if (r === undefined) {
      const d = Math.hypot(G.ex[G.id.get(a)!] - G.ex[G.id.get(b)!], G.ez[G.id.get(a)!] - G.ez[G.id.get(b)!]);
      r = this.route(a, b, false, d / 4 + 240);
      this.stopRoutes.set(key, r);
    }
    return r;
  }

  /** extend the route (cars: along their trip; trams, trains: random) with the curves joining lanes */
  private plan(v: Vehicle) {
    let ahead = v.route[0].end - v.s;
    for (let i = 1; i < v.route.length; i++) ahead += v.route[i].end - v.route[i].start;
    while (ahead < PLAN) {
      const last = v.route[v.route.length - 1];
      if (!last.lane) return;   // (always ends on a lane)
      let next: Path | undefined;
      if (v.kind === KIND.road || v.transit) {
        next = v.trip.shift();
        if (!next) return;                          // the trip ends on this lane
      } else {
        // rails: only connections a switch allows (no sharp bends, no reversing); none: end of the line
        // (directions over 8 m, so small kinks in the OSM geometry do not count; if no connection
        // passes, the gentlest one: trams only leave at real ends of the network)
        const dir = longHeading(last, last.end, -8);
        const opts = (this.byNode.get(last.lane.to) ?? []).filter((n) => n.kind === last.kind && n.end - n.start > 1
          && n.lane!.edge !== last.lane!.edge).map((n) => ({ n, c: dir.dot(longHeading(n, n.start, 8)) }));
        if (!opts.length) return;                   // leaves the network: handled when it gets there
        const ok = opts.filter((o) => o.c > SWITCH_COS);
        next = ok.length ? ok[Math.floor(Math.random() * ok.length)].n : opts.sort((a, b) => b.c - a.c)[0].n;
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

  private now = 0;   // seconds of the day (timetables)

  // The simulation (signals, junction grants, car following, routes) runs at a fixed STEP; drawing
  // happens every frame, each vehicle extrapolated by its speed for the time since the last step
  // (at most one STEP, so visually exact). Fewer, steadier simulation steps; same smooth motion.
  private acc = 0;
  update(dt: number, now = 0) {
    this.now = now;
    this.acc += dt;
    let n = 0;
    while (this.acc >= STEP && n < 4) { this.step(STEP); this.acc -= STEP; n++; }
    if (n === 4) this.acc = Math.min(this.acc, STEP);   // fell behind (a long frame): drop time, never spiral
    this.pose(this.acc);
    this.lastSteps = n;
  }
  /** simulation steps run in the last update (telemetry) */
  lastSteps = 0;

  private step(dt: number) {
    this.time += dt;
    for (const sig of this.signals) sig.state = signalState(sig, this.time);
    // occupancy per path, sorted by position
    // (two sets used in turn: no allocation per step)
    const used = this.usedB; used.clear();
    for (const v of this.vehicles) { const p = v.route[0]; if (!used.has(p)) { used.add(p); p.occ.length = 0; } p.occ.push(v); }
    for (const p of used) if (p.occ.length > 1) p.occ.sort(byS);
    for (const p of this.lastUsed) if (!used.has(p)) p.occ.length = 0;
    this.usedB = this.lastUsed;
    this.lastUsed = used;

    // junction grants: in order of arrival, a vehicle near its next junction curve may enter if nothing
    // conflicting is in there and no conflicting vehicle with precedence is about to enter
    // (candidate records are reused from step to step)
    const cands = this.cands, committed = this.committed;
    cands.length = 0; committed.length = 0;
    let nRec = 0;
    const rec = (v: Vehicle, c: Path, d: number) => {
      const r = this.recPool[nRec] ??= { v, c, d, free: true };
      nRec++;
      r.v = v; r.c = c; r.d = d; r.free = true;
      return r;
    };
    for (const v of this.vehicles) {
      const i = nextJunction(v.route);
      if (i < 0) continue;
      const d = this.distTo(v, i);
      // a grant is final once the vehicle can no longer stop before the junction
      if (v.grant === v.route[i] && d < stopDistance(v) + 0.5) { committed.push(rec(v, v.route[i], d)); continue; }
      v.grant = null;
      if (d < (v.v * v.v) / (2 * v.t.b) + 15) cands.push(rec(v, v.route[i], d));
      else { v.arrive = this.time; v.wait = 0; }
    }
    // pass 1: hard constraints (light, something conflicting inside, no room after the junction)
    for (const x of cands) {
      if (x.v.kind !== KIND.road) continue;
      const j = x.c.junction!, sig = x.c.from!.approach!.signal;
      const canStop = x.d > stopDistance(x.v);
      x.v.why = '';
      if (sig && !(sig.state === GREEN || (sig.state === AMBER && !canStop))) x.v.why = 'light';
      else if (this.occupied(j, x.v, x.c)) x.v.why = 'occupied';
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

    // IDM per vehicle (a copy: cars that finish their trip leave the list)
    const list = this.stepList;   // (a copy: cars that finish their trip leave this.vehicles)
    list.length = 0;
    for (const v of this.vehicles) list.push(v);
    for (const v of list) {
      const t = v.t;
      // desired speed: the speed limit, lowered ahead of curves so that braking is comfortable
      let v0 = Math.min(v.route[0].speed * v.f, t.maxV ?? Infinity), gap = Infinity, vLead = 0;
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
        // speed bumps (OSM traffic calming): over them at BUMP, braking comfortably before; until the rear is over
        if (t !== TYPES.bike)
          for (const sb of p.bumps) {
            const dd = d + sb;
            if (dd < -t.total || dd > LOOK) continue;
            v0 = Math.min(v0, dd <= 0 ? BUMP : Math.sqrt(BUMP * BUMP + 2 * t.b * 0.6 * dd));
          }
        if (gap === Infinity) {                              // vehicle ahead on this path
          for (const o of p.occ) {
            if (o === v || o.s <= from) continue;
            if (o.t === TYPES.bike && t !== TYPES.bike) continue;   // cars overtake cyclists
            gap = d + o.s - o.t.total; vLead = o.v; v.block = 'leader'; v.blockBy = o;
            break;
          }
        }
        // stop lines: a junction curve we may not enter yet, crossing lights
        if (i > 0 && p.junction && v.grant !== p && d + p.start < gap) { gap = d + p.start; vLead = 0; v.block = junctionBlock(v.why); v.blockBy = null; }
        if (v.kind === KIND.road)
          for (const x of p.crossings)
            if (x.s > from && x.sig.state !== GREEN && d + x.s < gap
              && (x.sig.state === RED || d + x.s > stopDistance(v))) { gap = d + x.s; vLead = 0; v.block = 'crossing'; v.blockBy = null; }
        // mapped crosswalks with someone on them: stop before (unless too close to stop: drive on)
        if (v.kind === KIND.road && this.zebraBusy)
          for (const z of p.zebras)
            if (z.s > from && d + z.s - 1 < gap && d + z.s > stopDistance(v) + 1 && this.zebraBusy(z.key)) {
              gap = d + z.s - 1; vLead = 0; v.block = 'crosswalk'; v.blockBy = null;
            }
        // STIB: the next stop (wait there until the departure time)
        if (v.transit) {
          const st = v.transit.stops[v.transit.next];
          if (st && st.p === p && st.s >= from - 2 && d + st.s < gap) { gap = d + st.s + t.s0; vLead = 0; }
        }
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
      v.wait = v.v < 0.5 && waitingAtJunction(v) ? v.wait + dt : 0;
      v.stuck = v.v < 0.2 && !this.atLight(v) ? v.stuck + dt : 0;
      if (v.transit) {   // at a stop: dwell, then go at the timetabled departure (or 15 s, if late)
        const R = v.transit, st = R.stops[R.next];
        if (st && v.route[0] === st.p && v.s > st.s - 2.5) {
          if (v.s > st.s + 3) st.served = true;                            // drove past it
          else if (v.v < 0.5) {
            if (st.arrived < 0) { st.arrived = this.now; this.onDwell?.(st.id); }   // people get on / off
            const late = this.now > st.dep + 30;   // running late: shorter stops
            if (this.now >= st.dep && this.now - st.arrived >= (late ? 6 : 12)) st.served = true;
          }
          if (st.served) {
            R.next++;
            // (after the last stop in the region it drives on to the edge and leaves there)
          }
        }
        v.stuck = 0;
      }
      if (v.stuck > STUCK) { this.finish(v, false); continue; }            // gridlock safety valve
      if (v.dest && v.route.length === 1 && v.route[0] === v.dest.p && v.s > v.dest.s - 2 && v.v < 1) {
        this.finish(v, true);                                               // arrived: parks
        continue;
      }
      v.s += v.v * dt;
      this.advance(v);
    }

    // signal heads
    let changed = false;
    for (const l of this.lightAt)
      if (l.state !== l.sig.state) { l.state = l.sig.state; this.lights.setColorAt(l.i, LIGHT_COLORS[l.state]); changed = true; }
    if (changed && this.lights.instanceColor) this.lights.instanceColor.needsUpdate = true;
  }

  /** vehicle bodies: two axles on the path per body, `ahead` seconds after the last step */
  private pose(ahead: number) {
    for (const v of this.vehicles) {
      const t = v.t;
      let fb = v.s + v.v * ahead;                          // front bumper of the section
      for (let k = 0; k < t.secs.length; k++) {
        const x = t.secs[k];
        this.at(v, fb - x.over, this.front);
        this.at(v, fb - x.over - x.wb, this.rear);
        const dir = tmp.subVectors(this.front, this.rear);
        dir.divideScalar(dir.length() || 1);
        this.pos.copy(this.front).addScaledVector(dir, x.over - x.len / 2);   // body centre, along the axle line
        if (t === TYPES.bike && !v.route[0].lane?.bikeOnly) { this.pos.x -= dir.z * 1.1; this.pos.z += dir.x * 1.1; }   // at the kerb
        this.e.set(0, Math.atan2(-dir.z, dir.x), Math.asin(Math.max(-1, Math.min(1, dir.y))));
        this.q.setFromEuler(this.e);
        this.pos.y += x.h / 2;
        this.m4.compose(this.pos, this.q, this.scl.set(1, 1, 1));
        this.fleet.setPose(v.inst[k], this.m4);
        fb -= x.len + t.gap;
      }
    }
  }
  private lastUsed = new Set<Path>();
  private usedB = new Set<Path>();
  private cands: { v: Vehicle; c: Path; d: number; free: boolean }[] = [];
  private committed: { v: Vehicle; c: Path; d: number; free: boolean }[] = [];
  private recPool: { v: Vehicle; c: Path; d: number; free: boolean }[] = [];
  private stepList: Vehicle[] = [];
  /** something conflicting with curve c inside junction j (other than v) */
  private occupied(j: Junction, v: Vehicle, c: Path) {
    for (const y of j.occ) if (y !== v && y.inJunction && this.conflicts(j, c, y.inJunction)) return true;
    return false;
  }

  /** move on to the next path when the front passes the end of the current one */
  private advance(v: Vehicle) {
    this.plan(v);   // keep the planned route PLAN m ahead (long lanes: it has to grow before the lane ends)
    while (v.s > v.route[0].end) {
      const over = v.s - v.route[0].end;
      v.hist.unshift(v.route.shift()!);
      if (v.hist.length > 6) v.hist.pop();
      if (!v.route.length) {                              // left the region (or the network)
        if (v.transit) this.release(v);
        else if (v.kind === KIND.road) this.finish(v, true); else this.place(v);
        return;
      }
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
    for (let i = 1; s > p.end && i < v.route.length; i++) {   // ahead of the current path (drawn between steps)
      s = v.route[i].start + (s - p.end);
      p = v.route[i];
    }
    if (s < 0) {   // before the start of what we know (just entered the region): continue straight back
      sample(p, 0, out);
      const d = sample(p, Math.min(2, p.length), tmp2).sub(out).normalize();
      return out.addScaledVector(d, s);
    }
    return sample(p, s, out);
  }
}
