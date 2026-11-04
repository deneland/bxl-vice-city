// STIB trams and buses on their real lines, to the timetable (pipeline transit.py, from the GTFS feed).
// A run appears at its first stop in the region at its time, drives stop to stop (traffic.ts waits at
// each until the departure time) and leaves after its last stop in the region.
import { KIND } from './roads';
import type { Spot, Traffic } from './traffic';

export interface TransitData {
  routes: [string, number, string][];               // name, mode (0 tram, 3 bus), colour
  stops: [number, number, string][];                // x, y (voxels), name
  days: [number, string, [number, number, number][]][][];   // Monday..Sunday: [route, headsign, [[stop, arr, dep]]]
}

export class Transit {
  private spots = new Map<string, Spot | null>();
  private day = -1;
  private next = 0;          // index of the next run to start today
  private last = -1;

  constructor(private data: TransitData, private traffic: Traffic, private vs: number) {
    this.warm();
  }

  /** compute the routes between consecutive stops of every run ahead of time, in idle time, so starting a
   *  run later costs no route searches (each pair is searched once; traffic.ts caches them) */
  private warm() {
    const seen = new Set<string>(), todo: [number, number, number][] = [];
    for (const day of this.data.days) for (const [ri, , st] of day) {
      const mode = this.data.routes[ri][1];
      for (let i = 0; i + 1 < st.length; i++) {
        const key = `${st[i][0]}>${st[i + 1][0]}:${mode}`;
        if (!seen.has(key)) { seen.add(key); todo.push([st[i][0], st[i + 1][0], mode]); }
      }
    }
    const idle = (cb: (d: { timeRemaining(): number }) => void) =>
      'requestIdleCallback' in window ? requestIdleCallback(cb, { timeout: 500 }) : setTimeout(() => cb({ timeRemaining: () => 4 }), 16);
    const step = (d: { timeRemaining(): number }) => {
      const end = performance.now() + Math.min(4, Math.max(1, d.timeRemaining()));
      while (todo.length && performance.now() < end) {
        const [a, b, mode] = todo.pop()!;
        for (const ca of this.cands(a, mode)) for (const cb of this.cands(b, mode)) this.traffic.routable(ca.p, cb.p);
      }
      if (todo.length) idle(step);
    };
    idle(step);
  }

  private spot(stop: number, mode: number) {
    const key = `${stop}:${mode}`;
    if (!this.spots.has(key)) {
      const [x, y] = this.data.stops[stop];
      this.spots.set(key, this.traffic.spotNear(x * this.vs, -y * this.vs, mode === 0 ? KIND.tram : KIND.road));
    }
    return this.spots.get(key)!;
  }

  /** candidate track / kerb positions of a stop (trams: several tracks nearby, e.g. both directions) */
  private cands(stop: number, mode: number) {
    const key = `${stop}:${mode}:c`;
    if (!this.candCache.has(key)) {
      const [x, y] = this.data.stops[stop];
      const c = mode === 0 ? this.traffic.spotsNear(x * this.vs, -y * this.vs, KIND.tram) : [this.spot(stop, mode)].filter(Boolean) as Spot[];
      this.candCache.set(key, c);
    }
    return this.candCache.get(key)!;
  }
  private candCache = new Map<string, Spot[]>();

  private focus: [number, number] | null = null;
  /** only runs with a stop within this distance of the focus get a vehicle (m) */
  radius = 1500;
  setFocus(x: number, z: number) { this.focus = [x, z]; }
  /** ms per frame for starting runs (route searches between their stops) */
  budgetMs = 1.5;
  private under: [number, string, [number, number, number][]][] = [];

  private start(run: [number, string, [number, number, number][]], from: number) {
    const [ri, headsign, st] = run;
    if (this.focus) {
      const f = this.focus, r2 = this.radius * this.radius;
      if (!st.some(([stop]) => {
        const [x, y] = this.data.stops[stop];
        return (x * this.vs - f[0]) ** 2 + (-y * this.vs - f[1]) ** 2 < r2;
      })) return;   // far from where you look
    }
    const [line, mode, color] = this.data.routes[ri];
    // per stop, the nearest candidate reachable from the previous stop's choice (trams: right track)
    const stops: { spot: Spot; dep: number; name: string; id: number }[] = [];
    for (const [stop, , dep] of st) {
      const cs = this.cands(stop, mode);
      const prev = stops[stops.length - 1]?.spot;
      const spot = prev ? cs.find((c) => this.traffic.routable(prev.p, c.p)) : cs[0];
      if (spot) stops.push({ spot, dep, name: this.data.stops[stop][2], id: stop });
    }
    this.traffic.spawnTransit({ line, headsign, color, mode }, stops, Math.min(from, Math.max(0, stops.length - 2)));
  }

  /** `now`: seconds of the day; `weekday`: 0 Monday .. 6 Sunday */
  update(now: number, weekday: number) {
    const runs = this.data.days[weekday] ?? [];
    const jump = weekday !== this.day || this.last < 0 || Math.abs(now - this.last) > 120;
    if (!jump && now < this.last) now = this.last;
    this.last = now;
    const stop = performance.now() + this.budgetMs;
    if (jump) {
      this.day = weekday;
      this.traffic.clearTransit();
      this.next = 0;
      // runs under way: start them at the last stop they passed (a few per frame, see below)
      this.under = [];
      for (; this.next < runs.length && runs[this.next][2][0][1] <= now; this.next++) {
        const st = runs[this.next][2];
        if (st[st.length - 1][2] > now) this.under.push(runs[this.next]);
      }
    }
    // (runs under way after a jump: as many per frame as the budget allows; they start at the stop
    // they would be at by now)
    while (this.under.length && performance.now() < stop) {
      const run = this.under.pop()!, st = run[2];
      if (st[st.length - 1][2] <= now) continue;
      let k = 0;
      while (k + 1 < st.length && st[k + 1][1] <= now) k++;
      this.start(run, k);
    }
    // start runs early enough to drive in from the region edge to their first stop
    while (this.next < runs.length) {
      const run = runs[this.next], [ri, , st] = run, mode = this.data.routes[ri][1];
      const first = this.cands(st[0][0], mode)[0];
      const lead = first ? this.traffic.leadIn(first, mode) : 0;
      if (st[0][1] - lead > now || performance.now() > stop) break;
      this.start(run, 0);
      this.next++;
    }
  }
}
