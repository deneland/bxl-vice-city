// People who drive: a fixed, seeded population with a home, a car and a daily plan (activities with
// times) that depends on the day of the week. Each departure becomes a car trip in the traffic
// simulation; arriving ends it. The plans are made to feel right, not to be a calibrated model:
// school runs around 8:15 and 15:30 (Wednesday noon), commuters, shoppers, evenings out (busier on
// Friday and Saturday), delivery vans, taxis, and through traffic crossing the area.
//
// Agents are kept apart from vehicles: an agent exists all day, its car only while it drives. That is
// what will let the simulation run in detail only near the camera when the region grows.
import type { Spot, Traffic } from './traffic';
import type { Pedestrians, WalkSpot } from './pedestrians';
import { Parking } from './parking';
import { prof } from './prof';

export interface Places {
  cats: string[]; streets: string[]; names: string[];
  homes: [number, number, number, number][];   // x, y (voxels), floors, street
  pois: [number, number, number, number, number?][];    // x, y (voxels), category, name, opening week
  weeks?: number[][];                          // 7 day masks (Monday first): bit h = open in hour h (places.py)
}

type Where = { kind: 'home' } | { kind: 'poi'; i: number } | { kind: 'edge' } | { kind: 'house'; i: number }
  | { kind: 'park'; spot: WalkSpot } | { kind: 'stop'; i: number };
export interface Activity { where: Where; purpose: string; until: number }   // until: seconds of the day

export interface Agent {
  id: number; name: string; type: string;
  mode: 'car' | 'walk' | 'bike';
  home: number;                       // index in places.homes (-1: lives outside the area)
  car: { kind: string; paint: number; ag: number; f: number };
  plan: Activity[];
  i: number;                          // current activity (at plan[i], or driving to plan[i + 1])
  driving: boolean;                   // on the way with a detailed vehicle / walker (near the camera)
  departed: number;                   // when the current trip started (s of day)
  departedFrom: number;               // parking place the current trip started from (-1: none)
  travel: Travel | null;              // on the way, away from the camera: a timeline only
  ver: number;                        // invalidates queued events
  parked: number;                     // parking place of the car while at a place (-1: none)
  reserve: number;                    // parking place taken for the trip's destination (-1: none)
  shown: boolean;                     // the parked car is on screen
  ride: number;                       // public transport: 0 -, 1 waiting at the stop, 2 riding, 3 about to get off
  retry: number;                      // no detailed trip before this time (the last attempt found no route)
  rideT: number;                      // since when (s of day)
}

/** a trip far from the camera: no vehicle, just where from / to and when */
interface Travel { t0: number; t1: number; pa: [number, number] | null; pb: [number, number] | null }

/** min-heap of timed events (departures, arrivals of timeline trips) */
class Events {
  private t: number[] = []; private a: Agent[] = []; private v: number[] = [];
  get size() { return this.t.length; }
  clear() { this.t.length = this.a.length = this.v.length = 0; }
  push(t: number, a: Agent) {
    let i = this.t.length;
    this.t.push(t); this.a.push(a); this.v.push(a.ver);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.t[p] <= t) break;
      this.t[i] = this.t[p]; this.a[i] = this.a[p]; this.v[i] = this.v[p]; i = p;
    }
    this.t[i] = t; this.a[i] = a; this.v[i] = a.ver;
  }
  peek() { return this.t.length ? this.t[0] : Infinity; }
  pop(): Agent | null {
    const { t, a, v } = this;
    const top = a[0], tv = v[0];
    const lt = t.pop()!, la = a.pop()!, lv = v.pop()!;
    if (t.length) {
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= t.length) break;
        if (c + 1 < t.length && t[c + 1] < t[c]) c++;
        if (t[c] >= lt) break;
        t[i] = t[c]; a[i] = a[c]; v[i] = v[c]; i = c;
      }
      t[i] = lt; a[i] = la; v[i] = lv;
    }
    return tv === top.ver ? top : null;   // stale entries are skipped
  }
}

// --- seeded randomness ---------------------------------------------------------------
function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const H = 3600, M = 60;
const VISITS = new Set(['shop', 'dinner', 'errand', 'pickup-food']);   // purposes that need the place open
const VISIT_REACH = 1500;   // m: how far a closed place may be swapped for an open one

// a Brussels mix of first and last names (generated people; nobody real)
const FIRST = ['Lucas', 'Emma', 'Mohamed', 'Louise', 'Adam', 'Olivia', 'Noah', 'Lina', 'Arthur', 'Nour', 'Louis', 'Sarah',
  'Ayoub', 'Alice', 'Jules', 'Inès', 'Victor', 'Yasmine', 'Gabriel', 'Camille', 'Rayan', 'Julie', 'Elias', 'Fatima',
  'Thomas', 'Marie', 'Ismaël', 'Léa', 'Hugo', 'Amira', 'Maxime', 'Clara', 'Youssef', 'Chloé', 'Mehdi', 'Zoé', 'Tom',
  'Manon', 'Ilyas', 'Anna', 'Piotr', 'Ana', 'Dimitri', 'Maria', 'Karim', 'Sofia', 'Jan', 'Eva', 'Pieter', 'Ilse',
  'Mamadou', 'Aminata', 'Ahmet', 'Elif', 'Andrei', 'Ioana', 'Marco', 'Giulia', 'Diego', 'Lucía'];
const LAST = ['Peeters', 'Janssens', 'Maes', 'Dubois', 'Lambert', 'Martin', 'Dupont', 'El Amrani', 'Benali', 'Diallo',
  'Claes', 'Wouters', 'Goossens', 'Leroy', 'Simon', 'Laurent', 'Mertens', 'Willems', 'Jacobs', 'Ben Ali', 'Bah',
  'Nowak', 'Kowalski', 'Popescu', 'Ionescu', 'Rossi', 'García', 'Fernandes', 'Yilmaz', 'Demir', 'Kaya', 'Haddad',
  'El Idrissi', 'Vermeulen', 'De Smet', 'Van den Broeck', 'Michel', 'Renard', 'Lejeune', 'Collard', 'Bouzid', 'Camara'];

// walkers (share of the walking population): kids walking to school with a parent, shoppers,
// dog walkers, errands, evening strolls
const WALK_TYPES: [string, number][] = [['kid', 0.18], ['walkShop', 0.24], ['dog', 0.14], ['walkErrand', 0.12], ['stroll', 0.12],
  ['transit', 0.2]];   // transit: walks to a tram / bus stop, rides away, comes back

// cyclists and delivery couriers (counts set in the constructor)
const BIKE_COLORS = [0x1b1b1d, 0x8c1c1c, 0x2f5597, 0xd8d4c8, 0x3d5a3a, 0xc9a227];
const COURIER_BOXES = [0x00ccbc, 0xff8000, 0x06c167];

const OCCUPANCY = 0.88;   // share of kerb parking places taken (simulated + anonymous cars)
const JUMP_MIN = 200;   // agents re-planned per frame after a jump in time, at least (then by time budget)

// share of each type in the population
const TYPES: [string, number][] = [['parent', 0.14], ['commuterOut', 0.18], ['commuterIn', 0.12], ['shopper', 0.18],
  ['evening', 0.1], ['errands', 0.08], ['delivery', 0.06], ['taxi', 0.04], ['through', 0.1]];

export const PURPOSE: Record<string, string> = {
  home: 'at home', school: 'dropping the kids at school', pickup: 'picking up the kids', work: 'at work',
  out: 'working outside the area', shop: 'shopping', dinner: 'out for dinner', errand: 'running an errand',
  delivery: 'delivering', fare: 'with a passenger', away: 'outside the area', passing: 'passing through',
  dog: 'walking the dog', stroll: 'out for a walk', wait: 'taking the tram / bus', 'pickup-food': 'collecting an order',
};

export class Agents {
  readonly list: Agent[] = [];
  private spots = new Map<string, Spot | null>();
  private byCat: number[][];
  private homeW: number[];
  private homeTotal: number;
  private day = '';
  private last = -1;
  /** called when something changed that the info panel shows */
  onChange: ((a: Agent) => void) | null = null;
  /** agents still to re-plan after a jump in time (0: done) */
  get replanning() { return this.list.length - Math.min(this.pending, this.list.length); }
  /** counts for the benchmark: agents simulated in detail, as timelines, parked cars shown */
  stats() {
    let detailed = 0, timeline = 0, shown = 0;
    for (const a of this.list) { if (a.driving) detailed++; else if (a.travel) timeline++; if (a.shown) shown++; }
    return { agents: this.list.length, detailed, timeline, parkedShown: shown, replanning: this.list.length - Math.min(this.pending, this.list.length), jumps: this.jumps.length,
      promoted: this.promoStats.ok, failed: this.promoStats.fail, promoMs: Math.round(this.promoStats.ms) };
  }

  constructor(readonly places: Places, private traffic: Traffic, private vs: number, count = 3000,
    private peds: Pedestrians | null = null, walkers = 0, cyclists = 0, private parking: Parking | null = null) {
    this.byCat = places.cats.map((_, c) => places.pois.flatMap((p, i) => (p[2] === c ? [i] : [])));
    this.homeW = places.homes.map((h) => h[2]);
    this.homeTotal = this.homeW.reduce((a, b) => a + b, 0);
    const r = rng(1);
    for (let id = 0; id < count; id++) {
      let x = r(), type = TYPES[0][0];
      for (const [t, w] of TYPES) { x -= w; if (x <= 0) { type = t; break; } }
      const outside = type === 'commuterIn' || type === 'through' || type === 'delivery';
      let car = traffic.randomCar();
      while (car.kind === 'bus') car = traffic.randomCar();
      if (type === 'delivery') car = { ...car, kind: 'van', paint: r() < 0.65 ? 0xeceeee : car.paint, ag: 1.1, f: 1.1 };
      if (type === 'taxi') car = { ...car, kind: 'sedan', paint: 0x1b1b1d };
      this.list.push({ id, type, mode: 'car', name: `${FIRST[Math.floor(r() * FIRST.length)]} ${LAST[Math.floor(r() * LAST.length)]}`,
        home: outside ? -1 : this.pickHome(r), car, plan: [], i: 0, driving: false, departed: 0, travel: null, ver: 0, parked: -1, reserve: -1, shown: false, departedFrom: -1, ride: 0, rideT: 0, retry: 0 });
    }
    for (let k = 0; k < (peds ? walkers : 0); k++) {
      let x = r(), type = WALK_TYPES[0][0];
      for (const [t, w] of WALK_TYPES) { x -= w; if (x <= 0) { type = t; break; } }
      this.list.push({ id: count + k, type, mode: 'walk', name: `${FIRST[Math.floor(r() * FIRST.length)]} ${LAST[Math.floor(r() * LAST.length)]}`,
        home: this.pickHome(r), car: traffic.randomCar(), plan: [], i: 0, driving: false, departed: 0, travel: null, ver: 0, parked: -1, reserve: -1, shown: false, departedFrom: -1, ride: 0, rideT: 0, retry: 0 });
    }
    for (let k = 0; k < cyclists; k++) {
      const courier = k < cyclists * 0.06;
      this.list.push({ id: count + walkers + k, type: courier ? 'courier' : 'cyclist', mode: courier ? 'car' : 'bike',
        name: `${FIRST[Math.floor(r() * FIRST.length)]} ${LAST[Math.floor(r() * LAST.length)]}`, home: courier ? -1 : this.pickHome(r),
        car: courier ? { kind: 'scooter', paint: COURIER_BOXES[Math.floor(r() * 3)], ag: 1.3, f: 1.15 }
          : { kind: 'bike', paint: BIKE_COLORS[Math.floor(r() * BIKE_COLORS.length)], ag: 1, f: 1 },
        plan: [], i: 0, driving: false, departed: 0, travel: null, ver: 0, parked: -1, reserve: -1, shown: false, departedFrom: -1, ride: 0, rideT: 0, retry: 0 });
    }
    traffic.onArrive = (a, arrived) => this.arrive(a as Agent, arrived);
    if (peds) peds.onArrive = (a) => this.arrive(a as Agent, true);
    // kerbs about as full as real ones (~88 %): what our drivers do not fill, anonymous cars do
    if (parking) {
      const own = this.list.filter((a) => this.parks(a)).length;
      parking.fillAnon(Math.max(0, OCCUPANCY - own / Math.max(1, parking.size)));
    }
  }

  private homeCum: Float64Array | null = null;
  /** a home, weighted by its floors (binary search in the cumulative weights: same pick as a linear scan) */
  private pickHome(r: () => number) {
    if (!this.homeCum) { let t = 0; this.homeCum = Float64Array.from(this.homeW, (w) => (t += w)); }
    const c = this.homeCum, x = r() * this.homeTotal;
    let lo = 0, hi = c.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (c[mid] >= x) hi = mid; else lo = mid + 1; }
    return lo;
  }
  private pools = new Map<string, number[]>();
  /** all places of these categories (cached per category set) */
  private poolOf(cats: string[]) {
    const key = cats.join(',');
    let p = this.pools.get(key);
    if (!p) { p = cats.flatMap((c) => this.byCat[this.places.cats.indexOf(c)] ?? []); this.pools.set(key, p); }
    return p;
  }
  private poi(r: () => number, ...cats: string[]): Where {
    const pool = this.poolOf(cats);
    return pool.length ? { kind: 'poi', i: pool[Math.floor(r() * pool.length)] } : { kind: 'edge' };
  }

  /** a destination of these categories near home (walking distance): one of the nearest few */
  private near(a: Agent, r: () => number, max: number, ...cats: string[]): Where | null {
    // the nearest few per 25 m cell of homes and category set, cached (neighbours share their shops and
    // schools; every later day is cheap)
    const h0 = this.places.homes[a.home], hc = [Math.floor(h0[0] / 50), Math.floor(h0[1] / 50)];
    const key = `${hc[0]},${hc[1]}|${max}|${cats.join(',')}`;
    let pool = this.nearCache.get(key);
    if (!pool) {
      const h = [hc[0] * 50 + 25, hc[1] * 50 + 25];
      // candidates from the cells around (a grid per category set) instead of every place
      // rings of cells outward; stop once the 6 nearest are closer than anything further out can be
      const grid = this.gridOf(cats), C = Agents.POI_CELL, Cm = C * this.vs, R = Math.ceil(max / Cm);
      const cx = Math.floor(h[0] / C), cy = Math.floor(h[1] / C), found: { i: number; d: number }[] = [];
      for (let r = 0; r <= R; r++) {
        for (let gx = cx - r; gx <= cx + r; gx++)
          for (let gy = cy - r; gy <= cy + r; gy++) {
            if (Math.max(Math.abs(gx - cx), Math.abs(gy - cy)) !== r) continue;
            for (const i of grid.get(gx * 65536 + gy) ?? []) {
              const d = Math.hypot(this.places.pois[i][0] - h[0], this.places.pois[i][1] - h[1]) * this.vs;
              if (d < max) found.push({ i, d });
            }
          }
        if (found.length >= 6) {
          found.sort((x, y) => x.d - y.d);
          if (found[5].d <= r * Cm) break;   // ring r + 1 is at least r cells away
        }
      }
      pool = found.sort((x, y) => x.d - y.d).slice(0, 6).map((x) => x.i);
      this.nearCache.set(key, pool);
    }
    return pool.length ? { kind: 'poi', i: pool[Math.floor(r() * pool.length)] } : null;
  }
  private nearCache = new Map<string, number[]>();

  /** is place i open at t (seconds of the day; may run past midnight) on weekday wd (0 Sunday)? */
  isOpen(i: number, wd: number, t: number) {
    const w = this.places.weeks?.[this.places.pois[i][4] ?? -1];
    if (!w) return true;
    const day = ((wd + 6) % 7 + Math.floor(t / (24 * H))) % 7, h = Math.floor((t % (24 * H)) / H);
    return ((w[day] >> h) & 1) === 1;
  }
  /** visitors (not staff, pupils or parents dropping off) go somewhere open when they arrive: a closed
   * place is swapped for the nearest open one of its kind within VISIT_REACH (else they go anyway) */
  private openFor(w: Where, purpose: string, wd: number, leave: number): Where {
    if (w.kind !== 'poi' || !VISITS.has(purpose)) return w;
    const t = leave + 10 * M;   // (about when they get there)
    if (this.isOpen(w.i, wd, t)) { this.openStats.open++; return w; }
    const o = this.places.pois[w.i], grid = this.gridOf([this.places.cats[o[2]]]), C = Agents.POI_CELL;
    const cx = Math.floor(o[0] / C), cy = Math.floor(o[1] / C), R = Math.ceil(VISIT_REACH / (C * this.vs));
    let best = -1, bd = Infinity;
    for (let gx = cx - R; gx <= cx + R; gx++)
      for (let gy = cy - R; gy <= cy + R; gy++)
        for (const i of grid.get(gx * 65536 + gy) ?? []) {
          const q = this.places.pois[i], d = Math.hypot(q[0] - o[0], q[1] - o[1]) * this.vs;
          if (d < bd && d < VISIT_REACH && this.isOpen(i, wd, t)) { bd = d; best = i; }
        }
    this.openStats[best >= 0 ? 'swapped' : 'closed']++;
    return best >= 0 ? { kind: 'poi', i: best } : w;
  }
  /** visits planned to an open place / swapped to an open one / to a closed one (nothing open near) */
  openStats = { open: 0, swapped: 0, closed: 0 };
  private static POI_CELL = 400;   // voxels (200 m)
  private grids = new Map<string, Map<number, number[]>>();
  /** places of these categories by grid cell (key gx * 65536 + gy) */
  private gridOf(cats: string[]) {
    const key = cats.join(',');
    let g = this.grids.get(key);
    if (!g) {
      g = new Map();
      for (const i of this.poolOf(cats)) {
        const p = this.places.pois[i], k = Math.floor(p[0] / Agents.POI_CELL) * 65536 + Math.floor(p[1] / Agents.POI_CELL);
        let l = g.get(k);
        if (!l) g.set(k, (l = []));
        l.push(i);
      }
      this.grids.set(key, g);
    }
    return g;
  }
  // --- public transport stops (transit.ts data) ---
  private stops: [number, number, string][] = [];
  setStops(stops: [number, number, string][]) { this.stops = stops; }
  private stopCache = new Map<number, number>();
  /** nearest stop to the agent's home within 700 m (-1: none) */
  private stopNear(a: Agent) {
    if (!this.stopCache.has(a.home)) {
      const h = this.places.homes[a.home];
      let best = -1, bd = (700 / this.vs) ** 2;
      this.stops.forEach(([x, y], i) => { const d = (x - h[0]) ** 2 + (y - h[1]) ** 2; if (d < bd) { bd = d; best = i; } });
      this.stopCache.set(a.home, best);
    }
    return this.stopCache.get(a.home)!;
  }
  private waiting = new Map<number, Set<Agent>>();    // stop -> people waiting to board
  private alighting = new Map<number, Set<Agent>>();  // stop -> people about to get off there
  private stopOf(a: Agent) { const w = a.plan[a.i]?.where; return w?.kind === 'stop' ? w.i : -1; }
  private showWaiting(a: Agent) {
    const st = this.stopOf(a);
    const sp = st >= 0 ? this.spot(a, a.plan[a.i].where) : null;
    const p = st >= 0 ? [this.stops[st][0] * this.vs, -this.stops[st][1] * this.vs] as [number, number] : null;
    if (a.ride === 1 && sp && this.inZone(p, this.radius.walk)) this.peds?.stand(a, sp as WalkSpot);
    else this.peds?.unstand(a);
  }
  /** a tram / bus stopped at stop `id`: the people waiting get on, the ones riding to it get off */
  onDwell(id: number) {
    for (const a of this.waiting.get(id) ?? []) { this.peds?.unstand(a); a.ride = 2; a.rideT = this.last; a.ver++; this.onChange?.(a); }
    this.waiting.get(id)?.clear();
    for (const a of this.alighting.get(id) ?? []) { a.ride = 0; this.depart(a, this.last); }
    this.alighting.get(id)?.clear();
  }

  /** a walk: preferably to a park path or square 250-900 m away, else around the block */
  private around(a: Agent, r: () => number): Where {
    const h = this.places.homes[a.home];
    const sp = this.peds?.parkSpot(h[0] * this.vs, -h[1] * this.vs, 250, 900, r);
    if (sp && r() < 0.75) return { kind: 'park', spot: sp };
    return this.block(a, r);
  }
  /** another building 200-500 m from home (a walk around the block) */
  private block(a: Agent, r: () => number): Where {
    const h = this.places.homes[a.home];
    for (let t = 0; t < 30; t++) {
      const i = Math.floor(r() * this.places.homes.length), o = this.places.homes[i];
      const d = Math.hypot(o[0] - h[0], o[1] - h[1]) * this.vs;
      if (d > 200 && d < 500) return { kind: 'house', i };
    }
    return { kind: 'home' };
  }

  /** a walker's day */
  private walkPlan(a: Agent, r: () => number, wd: number): Activity[] {
    const u = (a0: number, b0: number) => a0 + r() * (b0 - a0);
    const weekday = wd >= 1 && wd <= 5, wed = wd === 3;
    const home: Where = { kind: 'home' };
    const p: Activity[] = [];
    const act = (where: Where, purpose: string, until: number) =>
      p.push({ where: this.openFor(where, purpose, wd, p.length ? p[p.length - 1].until : 0), purpose, until });
    const outAndBack = (where: Where | null, purpose: string, t: number, stay: number) => {
      if (!where) return;
      act(home, 'home', t); act(where, purpose, t + stay);
    };
    switch (a.type) {
      case 'kid': {
        const school = weekday ? this.near(a, r, 1200, 'school', 'kindergarten') : null;
        if (!school) { outAndBack(this.near(a, r, 900, 'shop', 'supermarket', 'food'), 'shop', u(10 * H, 17 * H), u(20 * M, 60 * M)); break; }
        act(home, 'home', u(7 * H + 45 * M, 8 * H + 15 * M));
        act(school, 'school', wed ? u(11 * H + 55 * M, 12 * H + 10 * M) : u(15 * H + 20 * M, 15 * H + 45 * M));
        break;
      }
      case 'walkShop':
        outAndBack(this.near(a, r, 900, 'supermarket', 'shop'), 'shop', u(9 * H, 19 * H), u(10 * M, 35 * M));
        if (r() < 0.6) outAndBack(this.near(a, r, 900, 'shop', 'food', 'health'), 'shop', u(p.length ? p[p.length - 1].until + 30 * M : 10 * H, 19 * H), u(10 * M, 30 * M));
        break;
      case 'dog':
        for (const t of [u(7 * H, 8 * H + 30 * M), u(12 * H, 14 * H), u(19 * H, 22 * H)].slice(0, r() < 0.5 ? 2 : 3))
          outAndBack(this.around(a, r), 'dog', t, u(1 * M, 4 * M));
        break;
      case 'walkErrand':
        if (wd !== 0) outAndBack(this.near(a, r, 1000, 'health', 'services', 'leisure', 'food'), 'errand', u(9 * H, 18 * H), u(15 * M, 50 * M));
        break;
      case 'transit': {   // to work / town by tram or bus; weekends sometimes into town
        const stop = this.stopNear(a);
        if (stop < 0) break;
        const go = weekday ? u(7 * H, 9 * H) : r() < 0.5 ? u(10 * H, 16 * H) : -1;
        if (go < 0) break;
        act(home, 'home', go);
        act({ kind: 'stop', i: stop }, 'wait', weekday ? u(16 * H + 30 * M, 19 * H + 30 * M) : go + u(2 * H, 4 * H));
        break;
      }
      case 'stroll':
        if (r() < 0.5) outAndBack(this.near(a, r, 1000, 'food', 'leisure'), 'dinner', u(17 * H, 21 * H + 30 * M), u(30 * M, 120 * M));
        else outAndBack(this.around(a, r), 'stroll', u(wd === 0 || wd === 6 ? 10 * H : 16 * H, 20 * H), u(15 * M, 60 * M));
        break;
    }
    act(home, 'home', 24 * H);
    for (let k = 1; k < p.length; k++) p[k].until = Math.max(p[k].until, p[k - 1].until + 2 * M);
    return p;
  }

  /** the day's plan for an agent (seeded by agent and date: the same every time for that day) */
  private makePlan(a: Agent, date: string): Activity[] {
    const [y, m, d] = date.split('-').map(Number);
    const wd = new Date(Date.UTC(y, m - 1, d)).getUTCDay();   // 0 Sunday … 3 Wednesday … 6 Saturday
    const weekday = wd >= 1 && wd <= 5, wed = wd === 3;
    const r = rng(a.id * 7919 + y * 372 + m * 31 + d);
    if (a.mode === 'walk') return this.walkPlan(a, r, wd);
    const u = (a0: number, b0: number) => a0 + r() * (b0 - a0);
    const home: Where = a.home >= 0 ? { kind: 'home' } : { kind: 'edge' };
    const p: Activity[] = [];
    const act = (where: Where, purpose: string, until: number) =>
      p.push({ where: this.openFor(where, purpose, wd, p.length ? p[p.length - 1].until : 0), purpose, until });
    const shopping = () => {
      if (weekday || wd === 6 || r() < 0.3) {   // Sundays: most shops closed
        act(home, 'home', u(9 * H, 19 * H));
        act(this.poi(r, 'supermarket', 'shop'), 'shop', 0);
        p[p.length - 1].until = p[p.length - 2].until + u(15 * M, 50 * M);
        if (r() < 0.3) act(this.poi(r, 'shop', 'supermarket'), 'shop', p[p.length - 1].until + u(15 * M, 40 * M));
      }
      act(home, 'home', 24 * H);
    };
    switch (a.type) {
      case 'cyclist': {   // to work / school or an errand by bike, mostly weekdays
        if (weekday && r() < 0.6) {
          act(home, 'home', u(7 * H + 15 * M, 9 * H));
          act(this.poi(r, 'office', 'school', 'shop', 'health', 'services'), 'work', u(16 * H + 30 * M, 18 * H + 45 * M));
        } else if (r() < 0.7) {
          act(home, 'home', u(9 * H, 18 * H));
          act(this.poi(r, 'shop', 'supermarket', 'food', 'leisure'), 'shop', p[0].until + u(20 * M, 90 * M));
        }
        act(home, 'home', 24 * H);
        break;
      }
      case 'courier': {   // meal deliveries in the evening (and some lunchtimes): restaurant -> customer -> ...
        const slots = [[18 * H, 22 * H + 30 * M]];
        if (r() < 0.4) slots.unshift([11 * H + 45 * M, 13 * H + 45 * M]);
        let t = slots[0][0] + u(-20 * M, 20 * M);
        act({ kind: 'edge' }, 'away', t);
        for (const [s0, s1] of slots) {
          t = Math.max(t, s0);
          while (t < s1) {
            act(this.poi(r, 'food'), 'pickup-food', t + u(3 * M, 8 * M));
            t = p[p.length - 1].until;
            act({ kind: 'house', i: Math.floor(r() * this.places.homes.length) }, 'delivery', t + u(4 * M, 9 * M));
            t = p[p.length - 1].until + u(1 * M, 6 * M);
          }
        }
        act({ kind: 'edge' }, 'away', 24 * H);
        break;
      }
      case 'parent': {
        if (!weekday) return (shopping(), p);
        const school = this.poi(r, 'school', 'kindergarten');
        const drop = u(7 * H + 40 * M, 8 * H + 15 * M);
        const pick = wed ? u(11 * H + 50 * M, 12 * H + 20 * M) : u(15 * H + 15 * M, 15 * H + 45 * M);
        act(home, 'home', drop);
        act(school, 'school', drop + u(4 * M, 9 * M));
        const x = r();
        if (x < 0.4) act(this.poi(r, 'office', 'shop', 'health', 'school'), 'work', pick - u(10 * M, 20 * M));
        else if (x < 0.7) act({ kind: 'edge' }, 'out', pick - u(15 * M, 25 * M));
        else act(home, 'home', pick - u(8 * M, 15 * M));
        act(school, 'pickup', pick + u(5 * M, 10 * M));
        act(home, 'home', 24 * H);
        break;
      }
      case 'commuterOut':
        if (!weekday) return (shopping(), p);
        act(home, 'home', u(7 * H, 9 * H + 15 * M));
        act({ kind: 'edge' }, 'out', u(16 * H + 30 * M, 19 * H + 30 * M));
        act(home, 'home', 24 * H);
        break;
      case 'commuterIn':
        if (!weekday) { act(home, 'away', 24 * H); break; }
        act(home, 'away', u(7 * H + 15 * M, 9 * H + 15 * M));
        act(this.poi(r, 'office', 'shop', 'supermarket', 'health', 'school', 'services'), 'work', u(16 * H, 18 * H + 30 * M));
        act(home, 'away', 24 * H);
        break;
      case 'shopper': shopping(); break;
      case 'evening': {
        const go = wd === 5 || wd === 6 ? 0.85 : 0.4;
        if (r() > go) { act(home, 'home', 24 * H); break; }
        const t = u(18 * H + 30 * M, 20 * H + 45 * M);
        act(home, 'home', t);
        act(this.poi(r, 'food', 'leisure'), 'dinner', t + u(80 * M, 160 * M));
        act(home, 'home', 24 * H);
        break;
      }
      case 'errands':
        if (!weekday && wd !== 6) { act(home, 'home', 24 * H); break; }
        act(home, 'home', u(9 * H, 17 * H));
        act(this.poi(r, 'health', 'services', 'leisure', 'shop'), 'errand', p[0].until + u(20 * M, 60 * M));
        act(home, 'home', 24 * H);
        break;
      case 'delivery': {
        if (wd === 0) { act(home, 'away', 24 * H); break; }
        let t = u(8 * H, 10 * H);
        act(home, 'away', t);
        const stops = 5 + Math.floor(r() * 5);
        for (let k = 0; k < stops; k++) {
          t += u(10 * M, 35 * M);
          act(r() < 0.6 ? this.poi(r, 'shop', 'supermarket', 'food', 'office') : { kind: 'home' }, 'delivery', t + u(4 * M, 9 * M));
          t = p[p.length - 1].until;
        }
        act(home, 'away', 24 * H);
        break;
      }
      case 'taxi': {
        let t = u(6 * H, 9 * H);
        act({ kind: 'edge' }, 'away', t);
        while (t < 25 * H) {
          t += u(8 * M, 30 * M);
          act(r() < 0.5 ? { kind: 'home' } : this.poi(r, 'food', 'leisure', 'health'), 'fare', t);
        }
        p[p.length - 1].until = 24 * H;
        break;
      }
      case 'through': {
        // edge to edge, a few times a day, following the daily traffic curve
        let t = 0;
        const n = weekday ? 3 : 2;
        const times = Array.from({ length: n }, () => {
          // rush hours more likely
          const pk = r();
          return pk < 0.35 ? u(7 * H, 9 * H + 30 * M) : pk < 0.7 ? u(16 * H, 19 * H) : pk < 0.95 ? u(6 * H, 23 * H)
            : u(0, 6 * H);   // a little night traffic
        }).sort((x, y) => x - y);
        for (const tt of times) {
          act({ kind: 'edge' }, 'away', Math.max(tt, t + 10 * M));
          act({ kind: 'edge' }, 'passing', 0);   // (arrives at the other edge; leaves 'immediately')
          t = p[p.length - 2].until;
          p[p.length - 1].until = t + 1;
        }
        act({ kind: 'edge' }, 'away', 24 * H);
        break;
      }
    }
    // make the times increasing (short stays are fine)
    for (let k = 1; k < p.length; k++) p[k].until = Math.max(p[k].until, p[k - 1].until + 2 * M);
    return p;
  }

  /** kerbside spot of a place (cached) */
  private spot(a: Agent, w: Where): Spot | WalkSpot | 'edge' | null {
    if (w.kind === 'edge') return 'edge';
    if (w.kind === 'park') return w.spot;
    if (w.kind === 'stop') {
      const key = `s${w.i}`;
      if (!this.spots.has(key)) { const [x, y] = this.stops[w.i]; this.spots.set(key, this.peds?.spotNear(x * this.vs, -y * this.vs) as never ?? null); }
      return this.spots.get(key)!;
    }
    const walk = a.mode === 'walk';
    const key = (walk ? 'w' : 'c') + (w.kind === 'home' ? `h${a.home}` : w.kind === 'house' ? `h${w.i}` : `p${w.i}`);
    if (!this.spots.has(key)) {
      const xy = w.kind === 'home' ? this.places.homes[a.home] : w.kind === 'house' ? this.places.homes[w.i] : this.places.pois[w.i];
      const x = xy ? xy[0] * this.vs : 0, z = xy ? -xy[1] * this.vs : 0;
      this.spots.set(key, !xy ? null : walk ? (this.peds?.spotNear(x, z) ?? null) as never : this.traffic.spotNear(x, z));
    }
    return this.spots.get(key)!;
  }

  /** label of a place, for the info panel */
  label(a: Agent, w: Where) {
    if (w.kind === 'edge') return 'outside the area';
    if (w.kind === 'home') return a.home >= 0 ? `home, ${this.street(this.places.homes[a.home][3])}` : 'outside the area';
    if (w.kind === 'house') return this.street(this.places.homes[w.i][3]);
    if (w.kind === 'park') return 'a park / square';
    if (w.kind === 'stop') return `stop ${this.stops[w.i][2]}`;
    const p = this.places.pois[w.i];
    return this.places.names[p[3]] || this.places.cats[p[2]];
  }
  street(i: number) {
    const s = this.places.streets[i] || 'a street nearby';
    return s.split(' - ')[0];   // bilingual names: the French half
  }

  /** a rough trip duration before the route is known (for jumps in time) */
  private guess(a: Agent, from: Where, to: Where) {
    const pos = (w: Where) => w.kind === 'home' && a.home >= 0 ? this.places.homes[a.home]
      : w.kind === 'poi' ? this.places.pois[w.i] : w.kind === 'house' ? this.places.homes[w.i] : null;
    if (from.kind === 'park' || to.kind === 'park') return 700 / 1.3 + 20;
    if (from.kind === 'stop' || to.kind === 'stop') return 400 / 1.3 + 20;
    const pa = pos(from), pb = pos(to);
    const d = pa && pb ? Math.hypot(pa[0] - pb[0], pa[1] - pb[1]) * this.vs * 1.4 : 900;
    return a.mode === 'walk' ? d / 1.3 + 20 : d / 6 + 60;
  }

  // --- near the camera in detail, elsewhere as timelines -------------------------------------
  private focus: [number, number] | null = null;
  /** where you look (scene position in the traffic group's frame) */
  setFocus(x: number, z: number) { if (this.focus) { this.focus[0] = x; this.focus[1] = z; } else this.focus = [x, z]; }
  /** detail radius (m): vehicles, walkers, parked cars */
  radius = { drive: 900, walk: 550, park: 450 };

  /** does this agent's car stand somewhere while they are at a place (not taxis, couriers, through traffic) */
  private parks(a: Agent) {
    return !!this.parking && a.mode === 'car' && a.type !== 'taxi' && a.type !== 'courier' && a.type !== 'through';
  }
  /** take a parking place near a place (-1: none / outside the area) */
  private parkNear(a: Agent, w: Where) {
    const p = this.posOf(a, w);
    if (!p || !this.parking) return -1;
    const k = this.parking.nearestFree(p[0], p[1], 180);
    if (k >= 0) this.parking.take(k, a);
    return k;
  }
  private leaveParking(a: Agent) {
    if (a.parked >= 0) this.parking!.free(a.parked);
    if (a.shown) { this.traffic.unpark(a); a.shown = false; this.shownSet.delete(a); }
    a.parked = -1;
  }
  /** show / hide a parked car depending on the distance to where you look */
  private showParked(a: Agent) {
    const P = this.parking!;
    const want = a.parked >= 0 && !P.hidden[a.parked] && this.inZone([P.x[a.parked], P.z[a.parked]], this.radius.park);
    if (want && !a.shown) { this.traffic.park(a, a.car, P.x[a.parked], P.z[a.parked], P.heading[a.parked]); a.shown = true; this.shownSet.add(a); }
    else if (!want && a.shown) { this.traffic.unpark(a); a.shown = false; this.shownSet.delete(a); }
  }
  /** the kerb spot next to a parking place */
  private parkSpot(k: number) { return this.traffic.spotNear(this.parking!.x[k], this.parking!.z[k]); }
  private travelling = new Set<Agent>();
  private pending = Infinity;   // agents re-planned so far after a jump
  private events = new Events();
  private tick = -1;

  private inZone(p: [number, number] | null, r: number) {
    if (!this.focus) return true;
    if (!p) return false;
    const dx = p[0] - this.focus[0], dz = p[1] - this.focus[1];
    return dx * dx + dz * dz < r * r;
  }
  /** is a timeline trip within r of the focus now (no allocation: checked for every trip every frame) */
  private travelInZone(t: Travel, now: number, r: number) {
    if (!this.focus) return true;
    const k = Math.min(1, Math.max(0, (now - t.t0) / Math.max(1, t.t1 - t.t0)));
    let x: number, z: number;
    if (t.pa && t.pb) { x = t.pa[0] + (t.pb[0] - t.pa[0]) * k; z = t.pa[1] + (t.pb[1] - t.pa[1]) * k; }
    else { const p = k < 0.5 ? t.pa : t.pb; if (!p) return false; x = p[0]; z = p[1]; }
    const dx = x - this.focus[0], dz = z - this.focus[1];
    return dx * dx + dz * dz < r * r;
  }
  /** scene position (x, z) of a place: home / poi / house / park; null for outside the area */
  private posOf(a: Agent, w: Where): [number, number] | null {
    const xy = w.kind === 'home' ? (a.home >= 0 ? this.places.homes[a.home] : null) : w.kind === 'house' ? this.places.homes[w.i]
      : w.kind === 'poi' ? this.places.pois[w.i] : null;
    if (xy) return [xy[0] * this.vs, -xy[1] * this.vs];
    if (w.kind === 'park') { const p = this.peds?.spotPosition(w.spot); return p ? [p.x, p.z] : null; }
    if (w.kind === 'stop') { const s0 = this.stops[w.i]; return [s0[0] * this.vs, -s0[1] * this.vs]; }
    return null;
  }
  /** a timeline trip's position now (straight line; outside the area: the far end) */
  private travelPos(t: Travel, now: number): [number, number] | null {
    const k = Math.min(1, Math.max(0, (now - t.t0) / Math.max(1, t.t1 - t.t0)));
    if (t.pa && t.pb) return [t.pa[0] + (t.pb[0] - t.pa[0]) * k, t.pa[1] + (t.pb[1] - t.pa[1]) * k];
    return k < 0.5 ? t.pa : t.pb;
  }

  /** put the agent's trip on the road / pavement in detail (progress: share already done) */
  private detail(a: Agent, progress: number) {
    const from = a.plan[a.i], to = a.plan[a.i + 1];
    let sa = this.spot(a, from.where), sb = this.spot(a, to.where);
    if (a.mode === 'car' && a.departedFrom >= 0) sa = this.parkSpot(a.departedFrom) ?? sa;   // pull out of the parking place
    if (a.mode === 'car' && a.reserve >= 0) sb = this.parkSpot(a.reserve) ?? sb;             // and drive to the one taken there
    if (!sa || !sb) return false;
    if (a.mode === 'walk')
      return sa !== 'edge' && sb !== 'edge' && (this.peds?.spawn(a, sa as WalkSpot, sb as WalkSpot, progress) ?? 0) > 0;
    if (sa === 'edge' && sb === 'edge' && to.purpose !== 'passing') return false;   // outside -> outside: no drive
    return this.traffic.spawn(a, sa as Spot | 'edge', sb as Spot | 'edge', a.car, progress) > 0;
  }

  private depart(a: Agent, now: number, progress = 0) {
    const from = a.plan[a.i], to = a.plan[a.i + 1];
    if (!to) return;
    a.departed = now;
    if (this.parks(a)) {   // leave the parking place; take one at the destination
      a.departedFrom = a.parked;
      this.leaveParking(a);
      a.reserve = this.parkNear(a, to.where);
    }
    const dur = this.guess(a, from.where, to.where);
    if (from.where.kind === 'edge' && to.where.kind === 'edge' && to.purpose !== 'passing') { this.arrive(a, true); return; }
    const t: Travel = { t0: now - progress * dur, t1: now - progress * dur + dur, pa: this.posOf(a, from.where), pb: this.posOf(a, to.where) };
    const r = a.mode === 'walk' ? this.radius.walk : this.radius.drive;
    if (this.inZone(this.travelPos(t, now), r) && this.spare() && this.detail(a, progress)) {
      a.driving = true; this.drivingSet.add(a); a.ver++;
    } else {   // away from the camera (or no route): a timeline
      a.travel = t; a.ver++;
      this.travelling.add(a);
      this.events.push(t.t1, a);
    }
    this.onChange?.(a);
  }

  private arrive(a: Agent, _arrived: boolean) {
    a.driving = false;
    this.drivingSet.delete(a);
    a.travel = null;
    this.travelling.delete(a);
    a.i = Math.min(a.i + 1, a.plan.length - 1);
    if (a.mode === 'walk' && a.plan[a.i].purpose === 'wait') {   // at the stop: wait for a tram / bus
      const st = this.stopOf(a);
      a.ride = 1; a.rideT = this.last;
      if (!this.waiting.has(st)) this.waiting.set(st, new Set());
      this.waiting.get(st)!.add(a);
      this.showWaiting(a);
    }
    if (this.parks(a)) {   // park in the place taken at departure (or a free one nearby)
      a.parked = a.reserve >= 0 ? a.reserve : this.parkNear(a, a.plan[a.i].where);
      a.reserve = -1; a.departedFrom = -1;
      this.showParked(a);
    }
    a.ver++;
    if (a.i + 1 < a.plan.length) this.events.push(a.plan[a.i].until, a);
    this.onChange?.(a);
  }

  // --- frame budget: starting detailed trips (route searches) and re-planning after a jump are spread
  // over frames, so the frame rate never dips; what does not fit waits as a timeline for the next frame
  /** ms per frame for starting trips in detail / re-planning after a jump */
  budgetMs = 2; jumpBudgetMs = 6;
  private deadline = 0;
  jumps: { at: number; from: number; to: number; day: string }[] = [];   // (diagnostics)
  private last0 = -1;
  private spare() { return performance.now() < this.deadline; }
  private drivingSet = new Set<Agent>();   // agents on the way in detail (vehicle / walker)
  private shownSet = new Set<Agent>();     // agents whose parked car is on screen
  private anonShown = new Map<number, object>();   // anonymous parked cars on screen: place -> key
  private anonCars = new Map<number, { kind: string; paint: number; ag: number; f: number }>();
  private parkedAt: [number, number] | null = null;   // focus at the last parked-car refresh
  private parkedT = 0; private parkedMore = false;
  promoStats = { ok: 0, fail: 0, ms: 0 };   // (diagnostics)
  private promoteAt = 0; private promoteBacklog = true;

  /** timeline trips near the focus get a vehicle / walker (as far as the frame budget allows) */
  private promote(now: number) {
    // every frame while trips wait for the budget, else 4x a second (the scan covers every timeline trip)
    const t = performance.now();
    if (!this.promoteBacklog && t - this.promoteAt < 250) return;
    this.promoteAt = t;
    this.promoteBacklog = false;
    for (const a of this.travelling) {
      const r = a.mode === 'walk' ? this.radius.walk : this.radius.drive;
      const t = a.travel!;
      if (now < a.retry) continue;
      const k = (now - t.t0) / Math.max(1, t.t1 - t.t0);
      if (k >= 1) continue;
      if (!this.travelInZone(t, now, r - 60)) continue;
      if (!this.spare()) { this.promoteBacklog = true; return; }   // (the clock only checked for real candidates)
      const t0 = performance.now();
      const ok = this.detail(a, Math.max(0, k));
      this.promoStats.ms += performance.now() - t0;
      if (ok) {
        this.promoStats.ok++;
        this.travelling.delete(a); a.travel = null; a.driving = true; this.drivingSet.add(a); a.ver++;
      } else { this.promoStats.fail++; a.retry = now + 20; }   // no route / no room: not again for a while
    }
  }

  /** parked cars near the focus on screen, the others off: only when the focus moved (spatial lookup) */
  private refreshParked() {
    const P = this.parking, f = this.focus;
    if (!P || !f) return;
    if (this.parkedAt && !this.parkedMore && Math.hypot(f[0] - this.parkedAt[0], f[1] - this.parkedAt[1]) < 25) return;
    const t = performance.now();
    if (t - this.parkedT < 200) return;   // (fast flights: at most 5 refreshes a second)
    this.parkedT = t;
    this.parkedAt = [f[0], f[1]];
    let budget = 300;                     // new parked cars per refresh; the rest next time
    this.parkedMore = false;
    for (const a of this.shownSet) this.showParked(a);            // hides the ones now too far
    const r2 = this.radius.park ** 2;
    for (const [k, key] of this.anonShown)                         // anonymous cars: same
      if ((P.x[k] - f[0]) ** 2 + (P.z[k] - f[1]) ** 2 > r2) { this.traffic.unpark(key); this.anonShown.delete(k); }
    P.forEachNear(f[0], f[1], this.radius.park, (k, owner) => {   // shows the ones now near
      if (budget <= 0) { this.parkedMore = true; return; }
      if (owner === Parking.ANON) {
        if (!this.anonShown.has(k)) {
          budget--;
          const key = { anonPlace: k };
          let car = this.anonCars.get(k);
          if (!car) {   // (once per place: the same car stands there whenever you come back)
            do car = this.traffic.randomCar(); while (car.kind === 'bus');
            this.anonCars.set(k, car);
          }
          this.traffic.park(key, car, P.x[k], P.z[k], P.heading[k]);
          this.anonShown.set(k, key);
        }
        return;
      }
      const a = owner as Agent;
      if (a && !a.shown && a.parked === k) { budget--; this.showParked(a); }
    });
  }

  /** once a second: vehicles far away become timelines; people stop waiting at stops after 15 min */
  private rebalance(now: number) {
    for (const [, set] of this.waiting) for (const a of set) {
      if (now - a.rideT > 900) { set.delete(a); this.peds?.unstand(a); a.ride = 2; a.rideT = now; }
      else this.showWaiting(a);
    }
    for (const [, set] of this.alighting) for (const a of set)
      if (now - a.rideT > 900) { set.delete(a); a.ride = 0; this.depart(a, now); }
    if (!this.focus) return;
    let demote = 80;   // (after a jump of the view, hundreds are far away at once: spread over seconds)
    for (const a of this.drivingSet) {
      if (demote <= 0) break;
      const r = a.mode === 'walk' ? this.radius.walk : this.radius.drive;
      const p = a.mode === 'walk' ? this.peds?.positionOf(a) : this.traffic.positionOfAgent(a);
      if (!p || this.inZone([p.x, p.z], r + 150)) continue;
      const left = a.mode === 'walk' ? this.peds?.routeAhead(a).len ?? 0 : this.traffic.routeLeft(a);
      demote--;
      if (a.mode === 'walk') this.peds?.remove(a); else this.traffic.remove(a);
      const from = a.plan[a.i], to = a.plan[a.i + 1];
      const t: Travel = { t0: a.departed, t1: now + left / (a.mode === 'walk' ? 1.3 : 7) + 5,
        pa: this.posOf(a, from.where), pb: this.posOf(a, to.where) };
      a.driving = false; a.travel = t; a.ver++;
      this.drivingSet.delete(a);
      this.travelling.add(a);
      this.events.push(t.t1, a);
    }
  }

  /** advance to `now` (seconds of the day) on `date` (YYYY-MM-DD); a jump re-places everyone */
  update(now: number, date: string) {
    // a jump: new day, first call, or the clock moved more than 2 minutes; small steps back
    // (clock re-syncs) are ignored: time does not go back
    const jump = date !== this.day || this.last < 0 || Math.abs(now - this.last) > 120;
    this.last0 = this.last;
    if (!jump && now < this.last) now = this.last;
    this.last = now;
    this.deadline = performance.now() + this.budgetMs;
    if (jump) {
      this.jumps.push({ at: Math.round(performance.now()), from: Math.round(this.last0), to: Math.round(now), day: date });
      this.day = date;
      this.drivingSet.clear(); this.shownSet.clear(); this.anonShown.clear(); this.parkedAt = null;
      this.traffic.clearCars();
      this.peds?.clear();
      this.events.clear();
      this.travelling.clear();
      this.traffic.clearParked();
      this.parking?.clear();
      for (const a of this.list) { a.parked = a.reserve = a.departedFrom = -1; a.shown = false; a.ride = 0; a.retry = 0; }
      this.waiting.clear(); this.alighting.clear();
      this.pending = 0;   // re-plan everyone, a batch per frame (a whole commune takes a few frames)
    }
    if (this.pending < this.list.length) {
      // re-plan in time-boxed batches (a whole commune takes a second or so, spread over frames)
      const stop = performance.now() + this.jumpBudgetMs;
      let n = this.pending;
      for (; n < this.list.length && (n - this.pending < JUMP_MIN || performance.now() < stop); n++) {
        const a = this.list[n];
        a.plan = this.makePlan(a, date);
        a.driving = false; a.travel = null; a.ver++;
        this.drivingSet.delete(a);
        // where in the plan at `now`: at a place, or on the way (then put mid-trip)
        let i = 0;
        while (i + 1 < a.plan.length && a.plan[i].until <= now) i++;
        a.i = Math.max(0, i - 1);
        const prev = a.plan[a.i];
        if (i > 0 && prev.until <= now) {
          const dur = this.guess(a, prev.where, a.plan[i].where);
          if (now < prev.until + dur) { this.depart(a, prev.until, (now - prev.until) / dur); continue; }
          a.i = i;
        }
        if (a.i + 1 < a.plan.length) this.events.push(a.plan[a.i].until, a);
        if (this.parks(a)) { a.parked = this.parkNear(a, a.plan[a.i].where); this.showParked(a); }
        if (a.plan[a.i].purpose === 'wait') { a.ride = 2; a.rideT = now; }   // away on the tram / bus
      }
      this.pending = n;
      this.tick = now;
      if (n < this.list.length) return;
      this.deadline = performance.now() + this.budgetMs;   // (the re-plan used this frame's time)
    }
    // due events: departures (at a place) and arrivals of timeline trips
    prof.begin('ag:events');
    while (this.events.peek() <= now) {
      const a = this.events.pop();
      if (!a || a.driving) continue;
      if (a.travel) this.arrive(a, true);
      else if (a.ride === 2 || a.ride === 1) {   // time to come back: get off the next tram / bus at the stop
        const st = this.stopOf(a);
        this.waiting.get(st)?.delete(a); this.peds?.unstand(a);
        a.ride = 3; a.rideT = now;
        if (!this.alighting.has(st)) this.alighting.set(st, new Set());
        this.alighting.get(st)!.add(a);
      } else if (a.i + 1 < a.plan.length) this.depart(a, now);
    }
    prof.end('ag:events');
    prof.begin('ag:rebalance');
    if (now - this.tick >= 1 || now < this.tick) { this.tick = now; this.rebalance(now); }
    prof.end('ag:rebalance');
    prof.begin('ag:promote');
    this.promote(now);   // every frame, within the budget
    prof.end('ag:promote');
    prof.begin('ag:parked');
    this.refreshParked();
    prof.end('ag:parked');
  }

  /** agents on the way: in detail, as timelines */
  get driving() { return this.list.filter((a) => a.driving).length; }
  get onTheWay() { return this.travelling.size; }
}
