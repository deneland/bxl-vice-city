// The people: a fixed, seeded population (drivers, walkers, cyclists) with a home, a vehicle and a day plan
// (plans.ts). Each departure becomes a trip: in detail near the camera (a car in traffic.ts, a walker in
// pedestrians.ts), elsewhere a timeline (from, to, when); arriving ends it. Also their parked cars, and
// riding trams and buses (waiting at stops, getting on and off).
//
// Agents are kept apart from vehicles: an agent exists all day, its car only while it drives. That is
// what will let the simulation run in detail only near the camera when the region grows.
import { MinHeap } from './heap';
import { dayPlan, type Activity, type PlanContext } from './plans';
import type { PlaceIndex, Where } from './places';
import { rng } from './rng';
import type { Traffic } from './traffic';
import type { Spot } from './paths';
import type { Pedestrians, WalkSpot } from './pedestrians';
import { Parking } from './parking';
import { prof } from './prof';

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

/** timed events (departures, arrivals of timeline trips); an agent re-planned since (a.ver) skips its old ones */
class Events {
  private h = new MinHeap<Agent>();
  get size() { return this.h.size; }
  clear() { this.h.clear(); }
  push(t: number, a: Agent) { this.h.push(t, a, a.ver); }
  peek() { return this.h.peek(); }
  pop(): Agent | null { const a = this.h.pop(); return a && this.h.poppedTag === a.ver ? a : null; }
}

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
const WALK_SPEED = 1.3;   // m/s: average walking speed (estimates for trips away from the camera)
const JUMP_MIN = 200;   // agents re-planned per frame after a jump in time, at least (then by time budget)

// share of each type in the population
const TYPES: [string, number][] = [['parent', 0.14], ['commuterOut', 0.18], ['commuterIn', 0.12], ['shopper', 0.18],
  ['evening', 0.1], ['errands', 0.08], ['delivery', 0.06], ['taxi', 0.04], ['through', 0.1]];


export class Agents {
  readonly list: Agent[] = [];
  private spots = new Map<string, Spot | WalkSpot | null>();   // kerb (cars) / pavement (walkers) spot of a place
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

  private plans: PlanContext;
  /** visits planned to an open place / swapped to an open one / to a closed one (diagnostics) */
  get openStats() { return this.plans.openStats; }

  constructor(readonly places: PlaceIndex, private traffic: Traffic, count = 3000,
    private peds: Pedestrians | null = null, walkers = 0, cyclists = 0, private parking: Parking | null = null) {
    this.plans = { places, parkSpot: peds ? (x, z, min, max, r) => peds.parkSpot(x, z, min, max, r) : null,
      openStats: { open: 0, swapped: 0, closed: 0 } };
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
        home: outside ? -1 : places.pickHome(r), car, plan: [], i: 0, driving: false, departed: 0, travel: null, ver: 0, parked: -1, reserve: -1, shown: false, departedFrom: -1, ride: 0, rideT: 0, retry: 0 });
    }
    for (let k = 0; k < (peds ? walkers : 0); k++) {
      let x = r(), type = WALK_TYPES[0][0];
      for (const [t, w] of WALK_TYPES) { x -= w; if (x <= 0) { type = t; break; } }
      this.list.push({ id: count + k, type, mode: 'walk', name: `${FIRST[Math.floor(r() * FIRST.length)]} ${LAST[Math.floor(r() * LAST.length)]}`,
        home: places.pickHome(r), car: traffic.randomCar(), plan: [], i: 0, driving: false, departed: 0, travel: null, ver: 0, parked: -1, reserve: -1, shown: false, departedFrom: -1, ride: 0, rideT: 0, retry: 0 });
    }
    for (let k = 0; k < cyclists; k++) {
      const courier = k < cyclists * 0.06;
      this.list.push({ id: count + walkers + k, type: courier ? 'courier' : 'cyclist', mode: courier ? 'car' : 'bike',
        name: `${FIRST[Math.floor(r() * FIRST.length)]} ${LAST[Math.floor(r() * LAST.length)]}`, home: courier ? -1 : places.pickHome(r),
        car: courier ? { kind: 'scooter', paint: COURIER_BOXES[Math.floor(r() * 3)], ag: 1.3, f: 1.15 }
          : { kind: 'bike', paint: BIKE_COLORS[Math.floor(r() * BIKE_COLORS.length)], ag: 1, f: 1 },
        plan: [], i: 0, driving: false, departed: 0, travel: null, ver: 0, parked: -1, reserve: -1, shown: false, departedFrom: -1, ride: 0, rideT: 0, retry: 0 });
    }
    traffic.onArrive = (a, arrived) => this.arrive(a, arrived);
    if (peds) peds.onArrive = (a) => this.arrive(a, true);
    // kerbs about as full as real ones (~88 %): what our drivers do not fill, anonymous cars do
    if (parking) {
      const own = this.list.filter((a) => this.parks(a)).length;
      parking.fillAnon(Math.max(0, OCCUPANCY - own / Math.max(1, parking.size)));
    }
  }

  private waiting = new Map<number, Set<Agent>>();    // stop -> people waiting to board
  private alighting = new Map<number, Set<Agent>>();  // stop -> people about to get off there
  private stopOf(a: Agent) { const w = a.plan[a.i]?.where; return w?.kind === 'stop' ? w.i : -1; }
  private showWaiting(a: Agent) {
    const st = this.stopOf(a);
    const sp = st >= 0 ? this.spot(a, a.plan[a.i].where) : null;
    const p = st >= 0 ? this.places.stopPos(st) : null;
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

  /** kerbside spot of a place (cached) */
  private spot(a: Agent, w: Where): Spot | WalkSpot | 'edge' | null {
    if (w.kind === 'edge') return 'edge';
    if (w.kind === 'park') return w.spot;
    if (w.kind === 'stop') {
      const key = `s${w.i}`;
      if (!this.spots.has(key)) { const [x, z] = this.places.stopPos(w.i); this.spots.set(key, this.peds?.spotNear(x, z) ?? null); }
      return this.spots.get(key)!;
    }
    const walk = a.mode === 'walk';
    const key = (walk ? 'w' : 'c') + (w.kind === 'home' ? `h${a.home}` : w.kind === 'house' ? `h${w.i}` : `p${w.i}`);
    if (!this.spots.has(key)) {
      const xz = this.posOf(a, w);
      this.spots.set(key, !xz ? null : walk ? this.peds?.spotNear(xz[0], xz[1]) ?? null : this.traffic.spotNear(xz[0], xz[1]));
    }
    return this.spots.get(key)!;
  }

  /** label of a place, for the info panel */
  label(a: Agent, w: Where) {
    if (w.kind === 'edge') return 'outside the area';
    if (w.kind === 'home') return a.home >= 0 ? `home, ${this.places.homeStreet(a.home)}` : 'outside the area';
    if (w.kind === 'house') return this.places.homeStreet(w.i);
    if (w.kind === 'park') return 'a park / square';
    if (w.kind === 'stop') return `stop ${this.places.stopName(w.i)}`;
    return this.places.poiLabel(w.i);
  }

  /** a rough trip duration before the route is known (for jumps in time) */
  private guess(a: Agent, from: Where, to: Where) {
    if (from.kind === 'park' || to.kind === 'park') return 700 / WALK_SPEED + 20;
    if (from.kind === 'stop' || to.kind === 'stop') return 400 / WALK_SPEED + 20;
    const pa = this.posOf(a, from), pb = this.posOf(a, to);
    const d = pa && pb ? Math.hypot(pa[0] - pb[0], pa[1] - pb[1]) * 1.4 : 900;
    return a.mode === 'walk' ? d / WALK_SPEED + 20 : d / 6 + 60;
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
    switch (w.kind) {
      case 'home': return a.home >= 0 ? this.places.homePos(a.home) : null;
      case 'house': return this.places.homePos(w.i);
      case 'poi': return this.places.poiPos(w.i);
      case 'stop': return this.places.stopPos(w.i);
      case 'park': { const p = this.peds?.spotPosition(w.spot); return p ? [p.x, p.z] : null; }
      default: return null;
    }
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
      const a = owner;
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
      const left = a.mode === 'walk' ? this.peds?.routeAhead(a, false).len ?? 0 : this.traffic.routeLeft(a);
      demote--;
      if (a.mode === 'walk') this.peds?.remove(a); else this.traffic.remove(a);
      const from = a.plan[a.i], to = a.plan[a.i + 1];
      const t: Travel = { t0: a.departed, t1: now + left / (a.mode === 'walk' ? WALK_SPEED : 7) + 5,
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
        a.plan = dayPlan(a, date, this.plans);
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

}
