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

export interface Places {
  cats: string[]; streets: string[]; names: string[];
  homes: [number, number, number, number][];   // x, y (voxels), floors, street
  pois: [number, number, number, number][];    // x, y (voxels), category, name
}

type Where = { kind: 'home' } | { kind: 'poi'; i: number } | { kind: 'edge' } | { kind: 'house'; i: number }
  | { kind: 'park'; spot: WalkSpot };
export interface Activity { where: Where; purpose: string; until: number }   // until: seconds of the day

export interface Agent {
  id: number; name: string; type: string;
  mode: 'car' | 'walk' | 'bike';
  home: number;                       // index in places.homes (-1: lives outside the area)
  car: { kind: string; paint: number; ag: number; f: number };
  plan: Activity[];
  i: number;                          // current activity (at plan[i], or driving to plan[i + 1])
  driving: boolean;
  departed: number;                   // when the current trip started (s of day)
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
const WALK_TYPES: [string, number][] = [['kid', 0.22], ['walkShop', 0.3], ['dog', 0.18], ['walkErrand', 0.15], ['stroll', 0.15]];

// cyclists and delivery couriers (counts set in the constructor)
const BIKE_COLORS = [0x1b1b1d, 0x8c1c1c, 0x2f5597, 0xd8d4c8, 0x3d5a3a, 0xc9a227];
const COURIER_BOXES = [0x00ccbc, 0xff8000, 0x06c167];

// share of each type in the population
const TYPES: [string, number][] = [['parent', 0.14], ['commuterOut', 0.18], ['commuterIn', 0.12], ['shopper', 0.18],
  ['evening', 0.1], ['errands', 0.08], ['delivery', 0.06], ['taxi', 0.04], ['through', 0.1]];

export const PURPOSE: Record<string, string> = {
  home: 'at home', school: 'dropping the kids at school', pickup: 'picking up the kids', work: 'at work',
  out: 'working outside the area', shop: 'shopping', dinner: 'out for dinner', errand: 'running an errand',
  delivery: 'delivering', fare: 'with a passenger', away: 'outside the area', passing: 'passing through',
  dog: 'walking the dog', stroll: 'out for a walk', 'pickup-food': 'collecting an order',
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

  constructor(readonly places: Places, private traffic: Traffic, private vs: number, count = 3000,
    private peds: Pedestrians | null = null, walkers = 0, cyclists = 0) {
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
        home: outside ? -1 : this.pickHome(r), car, plan: [], i: 0, driving: false, departed: 0 });
    }
    for (let k = 0; k < (peds ? walkers : 0); k++) {
      let x = r(), type = WALK_TYPES[0][0];
      for (const [t, w] of WALK_TYPES) { x -= w; if (x <= 0) { type = t; break; } }
      this.list.push({ id: count + k, type, mode: 'walk', name: `${FIRST[Math.floor(r() * FIRST.length)]} ${LAST[Math.floor(r() * LAST.length)]}`,
        home: this.pickHome(r), car: traffic.randomCar(), plan: [], i: 0, driving: false, departed: 0 });
    }
    for (let k = 0; k < cyclists; k++) {
      const courier = k < cyclists * 0.06;
      this.list.push({ id: count + walkers + k, type: courier ? 'courier' : 'cyclist', mode: courier ? 'car' : 'bike',
        name: `${FIRST[Math.floor(r() * FIRST.length)]} ${LAST[Math.floor(r() * LAST.length)]}`, home: courier ? -1 : this.pickHome(r),
        car: courier ? { kind: 'scooter', paint: COURIER_BOXES[Math.floor(r() * 3)], ag: 1.3, f: 1.15 }
          : { kind: 'bike', paint: BIKE_COLORS[Math.floor(r() * BIKE_COLORS.length)], ag: 1, f: 1 },
        plan: [], i: 0, driving: false, departed: 0 });
    }
    traffic.onArrive = (a, arrived) => this.arrive(a as Agent, arrived);
    if (peds) peds.onArrive = (a) => this.arrive(a as Agent, true);
  }

  private pickHome(r: () => number) {
    let x = r() * this.homeTotal;
    for (let i = 0; i < this.homeW.length; i++) { x -= this.homeW[i]; if (x <= 0) return i; }
    return 0;
  }
  private poi(r: () => number, ...cats: string[]): Where {
    const pool = cats.flatMap((c) => this.byCat[this.places.cats.indexOf(c)] ?? []);
    return pool.length ? { kind: 'poi', i: pool[Math.floor(r() * pool.length)] } : { kind: 'edge' };
  }

  /** a destination of these categories near home (walking distance): one of the nearest few */
  private near(a: Agent, r: () => number, max: number, ...cats: string[]): Where | null {
    const h = this.places.homes[a.home];
    const pool = cats.flatMap((c) => this.byCat[this.places.cats.indexOf(c)] ?? [])
      .map((i) => ({ i, d: Math.hypot(this.places.pois[i][0] - h[0], this.places.pois[i][1] - h[1]) * this.vs }))
      .filter((x) => x.d < max).sort((x, y) => x.d - y.d).slice(0, 6);
    return pool.length ? { kind: 'poi', i: pool[Math.floor(r() * pool.length)].i } : null;
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
    const act = (where: Where, purpose: string, until: number) => p.push({ where, purpose, until });
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
    const act = (where: Where, purpose: string, until: number) => p.push({ where, purpose, until });
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
    const pa = pos(from), pb = pos(to);
    const d = pa && pb ? Math.hypot(pa[0] - pb[0], pa[1] - pb[1]) * this.vs * 1.4 : 900;
    return a.mode === 'walk' ? d / 1.3 + 20 : d / 6 + 60;
  }

  private depart(a: Agent, now: number, progress = 0) {
    const from = a.plan[a.i], to = a.plan[a.i + 1];
    const sa = this.spot(a, from.where), sb = this.spot(a, to.where);
    a.departed = now;
    const go = a.mode === 'walk'
      ? sa && sb && sa !== 'edge' && sb !== 'edge' && (this.peds?.spawn(a, sa as WalkSpot, sb as WalkSpot, progress) ?? 0) > 0
      : sa && sb && !(sa === 'edge' && sb === 'edge' && to.purpose !== 'passing')   // outside -> outside: no drive
        && this.traffic.spawn(a, sa as Spot | 'edge', sb as Spot | 'edge', a.car, progress) > 0;
    if (go) {
      a.driving = true;
    } else {
      a.i++;   // no route (or nothing to drive): just be there
    }
    this.onChange?.(a);
  }

  private arrive(a: Agent, _arrived: boolean) {
    a.driving = false;
    a.i = Math.min(a.i + 1, a.plan.length - 1);
    this.onChange?.(a);
  }

  /** advance to `now` (seconds of the day) on `date` (YYYY-MM-DD); a jump re-places everyone */
  update(now: number, date: string) {
    // a jump: new day, first call, or the clock moved more than 2 minutes; small steps back
    // (clock re-syncs) are ignored: time does not go back
    const jump = date !== this.day || this.last < 0 || Math.abs(now - this.last) > 120;
    if (!jump && now < this.last) now = this.last;
    this.last = now;
    if (jump) {
      this.day = date;
      this.traffic.clearCars();
      this.peds?.clear();
      for (const a of this.list) {
        a.plan = this.makePlan(a, date);
        a.driving = false;
        // where in the plan at `now`: at a place, or on the way (then the car is put mid-route)
        let i = 0;
        while (i + 1 < a.plan.length && a.plan[i].until <= now) i++;
        a.i = Math.max(0, i - 1);
        const prev = a.plan[a.i];
        if (i > 0 && prev.until <= now) {
          const dur = this.guess(a, prev.where, a.plan[i].where);
          if (now < prev.until + dur) { this.depart(a, prev.until, (now - prev.until) / dur); continue; }
          a.i = i;
        }
      }
      return;
    }
    for (const a of this.list)
      if (!a.driving && a.i + 1 < a.plan.length && now >= a.plan[a.i].until) this.depart(a, now);
  }

  /** the agent driving a vehicle, if any */
  get driving() { return this.list.filter((a) => a.driving).length; }
}
