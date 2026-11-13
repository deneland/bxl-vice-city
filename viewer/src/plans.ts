// Day plans: what each person does on a given day (activities with times and places), by type and
// weekday. Made to feel right, not a calibrated model: school runs around 8:15 and 15:30 (Wednesday noon),
// commuters, shoppers, evenings out (busier on Friday and Saturday), delivery vans, taxis, through traffic;
// at night: night and early shifts, night taxis, people coming home late, early deliveries;
// walkers: kids to school, shopping, dog walks, errands, strolls, tram / bus rides. Seeded per person and
// date: the same plan every time for that day.
//
// A plan covers one calendar day, 0:00-24:00. Night activities cross midnight: a day that ends somewhere
// (at work on a night shift, in a taxi, out late) is followed by a day that starts there, so the drive
// home after midnight is in the next day's plan. Both days decide it the same way (by weekday, or by a
// draw seeded per person and night: lateNight).
import { H, M, weekday as weekdayOf } from './clock';
import { rng } from './rng';
import type { PlaceIndex, Where } from './places';
import type { WalkSpot } from './pedestrians';
import type { Agent } from './agents';

export interface Activity { where: Where; purpose: string; until: number }   // until: seconds of the day

/** what plans need: the places, park paths for walks (pedestrians.ts) */
export interface PlanContext {
  places: PlaceIndex;
  parkSpot: ((x: number, z: number, min: number, max: number, r: () => number) => WalkSpot | null) | null;
}

export const PURPOSE: Record<string, string> = {
  home: 'at home', school: 'dropping the kids at school', pickup: 'picking up the kids', work: 'at work',
  out: 'working outside the area', shop: 'shopping', dinner: 'out for dinner', errand: 'running an errand',
  delivery: 'delivering', fare: 'with a passenger', away: 'outside the area', passing: 'passing through',
  dog: 'walking the dog', stroll: 'out for a walk', wait: 'taking the tram / bus', 'pickup-food': 'collecting an order',
  late: 'out for the night',
};

/** does this person (type 'evening') stay out past midnight on the night that starts on `date`? (the same
 *  answer for the plan of that day and of the next) */
function lateNight(a: Agent, date: string) {
  const [y, m, d] = date.split('-').map(Number), wd = (weekdayOf(date) + 1) % 7;
  const p = wd === 5 || wd === 6 ? 0.25 : 0.12;   // (more on Friday and Saturday nights)
  return rng(a.id * 104729 + y * 372 + m * 31 + d)() < p;
}
/** where someone is at midnight of the night that starts on `date` (the same for both days' plans) */
function midnightPlace(a: Agent, date: string, P: PlaceIndex): Where {
  const [y, m, d] = date.split('-').map(Number);
  return P.poi(rng(a.id * 7907 + y * 372 + m * 31 + d), 'food', 'leisure');
}
/** the date before (YYYY-MM-DD) */
function dayBefore(date: string) {
  const t = new Date(`${date}T12:00:00Z`);
  t.setUTCDate(t.getUTCDate() - 1);
  return t.toISOString().slice(0, 10);
}

const VISITS = new Set(['shop', 'dinner', 'errand', 'pickup-food']);   // purposes that need the place open
const VISIT_REACH = 1500;   // m: how far a closed place may be swapped for an open one

/** visitors (not staff, pupils or parents dropping off) go somewhere open when they arrive: a closed place is
 * swapped for the nearest open one of its kind within VISIT_REACH (else they go anyway). wd: 0 Sunday */
function openFor(ctx: PlanContext, w: Where, purpose: string, wd: number, leave: number): Where {
  if (w.kind !== 'poi' || !VISITS.has(purpose)) return w;
  const t = leave + 10 * M, monday0 = (wd + 6) % 7;   // (about when they get there)
  if (ctx.places.isOpen(w.i, monday0, t)) return w;
  const best = ctx.places.openNear(w.i, monday0, t, VISIT_REACH);
  return best >= 0 ? { kind: 'poi', i: best } : w;
}

/** a walk: preferably to a park path or square 250-900 m away, else around the block */
function around(a: Agent, r: () => number, ctx: PlanContext): Where {
  const [x, z] = ctx.places.homePos(a.home);
  const sp = ctx.parkSpot?.(x, z, 250, 900, r);
  if (sp && r() < 0.75) return { kind: 'park', spot: sp };
  const i = ctx.places.houseNear(a.home, r);
  return i >= 0 ? { kind: 'house', i } : { kind: 'home' };
}

/** what every plan is built with: the seeded draws, the day, home, the activities so far and how to add one */
interface Day {
  r: () => number;
  u: (a: number, b: number) => number;   // a time (or anything) uniformly between a and b
  wd: number; weekday: boolean; wed: boolean;   // wd: 0 Sunday … 6 Saturday
  home: Where;
  p: Activity[];
  act: (where: Where, purpose: string, until: number) => void;
}

/** the times increasing (short stays are fine) */
function settle(p: Activity[]) {
  for (let k = 1; k < p.length; k++) p[k].until = Math.max(p[k].until, p[k - 1].until + 2 * M);
  return p;
}

/** a walker's day */
function walkPlan(a: Agent, D: Day, ctx: PlanContext): Activity[] {
  const P = ctx.places, { r, u, wd, weekday, wed, home, p, act } = D;
  const outAndBack = (where: Where | null, purpose: string, t: number, stay: number) => {
    if (!where) return;
    act(home, 'home', t); act(where, purpose, t + stay);
  };
  switch (a.type) {
    case 'kid': {
      const school = weekday ? P.near(a.home, r, 1200, 'school', 'kindergarten') : null;
      if (!school) { outAndBack(P.near(a.home, r, 900, 'shop', 'supermarket', 'food'), 'shop', u(10 * H, 17 * H), u(20 * M, 60 * M)); break; }
      act(home, 'home', u(7 * H + 45 * M, 8 * H + 15 * M));
      act(school, 'school', wed ? u(11 * H + 55 * M, 12 * H + 10 * M) : u(15 * H + 20 * M, 15 * H + 45 * M));
      break;
    }
    case 'walkShop':
      outAndBack(P.near(a.home, r, 900, 'supermarket', 'shop'), 'shop', u(9 * H, 19 * H), u(10 * M, 35 * M));
      if (r() < 0.6) outAndBack(P.near(a.home, r, 900, 'shop', 'food', 'health'), 'shop', u(p.length ? p[p.length - 1].until + 30 * M : 10 * H, 19 * H), u(10 * M, 30 * M));
      break;
    case 'dog':
      for (const t of [u(7 * H, 8 * H + 30 * M), u(12 * H, 14 * H), u(19 * H, 22 * H)].slice(0, r() < 0.5 ? 2 : 3))
        outAndBack(around(a, r, ctx), 'dog', t, u(1 * M, 4 * M));
      break;
    case 'walkErrand':
      if (wd !== 0) outAndBack(P.near(a.home, r, 1000, 'health', 'services', 'leisure', 'food'), 'errand', u(9 * H, 18 * H), u(15 * M, 50 * M));
      break;
    case 'transit': {   // to work / town by tram or bus; weekends sometimes into town
      const stop = P.stopNear(a.home);
      if (stop < 0) break;
      const go = weekday ? u(7 * H, 9 * H) : r() < 0.5 ? u(10 * H, 16 * H) : -1;
      if (go < 0) break;
      act(home, 'home', go);
      act({ kind: 'stop', i: stop }, 'wait', weekday ? u(16 * H + 30 * M, 19 * H + 30 * M) : go + u(2 * H, 4 * H));
      break;
    }
    case 'stroll':
      if (r() < 0.5) outAndBack(P.near(a.home, r, 1000, 'food', 'leisure'), 'dinner', u(17 * H, 21 * H + 30 * M), u(30 * M, 120 * M));
      else outAndBack(around(a, r, ctx), 'stroll', u(wd === 0 || wd === 6 ? 10 * H : 16 * H, 20 * H), u(15 * M, 60 * M));
      break;
  }
  act(home, 'home', 24 * H);
  return settle(p);
}

/** the day's plan for an agent (seeded by agent and date: the same every time for that day) */
export function dayPlan(a: Agent, date: string, ctx: PlanContext): Activity[] {
  const P = ctx.places;
  const [y, m, d] = date.split('-').map(Number);
  const wd = (weekdayOf(date) + 1) % 7;   // (the plans count from Sunday: 0 Sunday … 3 Wednesday … 6 Saturday)
  const r = rng(a.id * 7919 + y * 372 + m * 31 + d), p: Activity[] = [];
  const D: Day = { r, u: (a0, b0) => a0 + r() * (b0 - a0), wd, weekday: wd >= 1 && wd <= 5, wed: wd === 3,
    home: a.home >= 0 ? { kind: 'home' } : { kind: 'edge' }, p,
    act: (where, purpose, until) => p.push({ where: openFor(ctx, where, purpose, wd, p.length ? p[p.length - 1].until : 0), purpose, until }) };
  if (a.mode === 'walk') return walkPlan(a, D, ctx);
  const { u, weekday, wed, home, act } = D;
  const shopping = () => {
    if (weekday || wd === 6 || r() < 0.3) {   // Sundays: most shops closed
      act(home, 'home', u(9 * H, 19 * H));
      act(P.poi(r, 'supermarket', 'shop'), 'shop', 0);
      p[p.length - 1].until = p[p.length - 2].until + u(15 * M, 50 * M);
      if (r() < 0.3) act(P.poi(r, 'shop', 'supermarket'), 'shop', p[p.length - 1].until + u(15 * M, 40 * M));
    }
    act(home, 'home', 24 * H);
  };
  switch (a.type) {
    case 'cyclist': {   // to work / school or an errand by bike, mostly weekdays
      if (weekday && r() < 0.6) {
        act(home, 'home', u(7 * H + 15 * M, 9 * H));
        act(P.poi(r, 'office', 'school', 'shop', 'health', 'services'), 'work', u(16 * H + 30 * M, 18 * H + 45 * M));
      } else if (r() < 0.7) {
        act(home, 'home', u(9 * H, 18 * H));
        act(P.poi(r, 'shop', 'supermarket', 'food', 'leisure'), 'shop', p[0].until + u(20 * M, 90 * M));
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
          act(P.poi(r, 'food'), 'pickup-food', t + u(3 * M, 8 * M));
          t = p[p.length - 1].until;
          act({ kind: 'house', i: Math.floor(r() * P.homeCount) }, 'delivery', t + u(4 * M, 9 * M));
          t = p[p.length - 1].until + u(1 * M, 6 * M);
        }
      }
      act({ kind: 'edge' }, 'away', 24 * H);
      break;
    }
    case 'parent': {
      if (!weekday) return (shopping(), p);
      const school = P.poi(r, 'school', 'kindergarten');
      const drop = u(7 * H + 40 * M, 8 * H + 15 * M);
      const pick = wed ? u(11 * H + 50 * M, 12 * H + 20 * M) : u(15 * H + 15 * M, 15 * H + 45 * M);
      act(home, 'home', drop);
      act(school, 'school', drop + u(4 * M, 9 * M));
      const x = r();
      if (x < 0.4) act(P.poi(r, 'office', 'shop', 'health', 'school'), 'work', pick - u(10 * M, 20 * M));
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
      act(P.poi(r, 'office', 'shop', 'supermarket', 'health', 'school', 'services'), 'work', u(16 * H, 18 * H + 30 * M));
      act(home, 'away', 24 * H);
      break;
    case 'shopper': shopping(); break;
    case 'evening': {
      // out late last night: the day starts there, home between 0:30 and 3:00
      if (lateNight(a, dayBefore(date))) act(midnightPlace(a, dayBefore(date), P), 'late', u(30 * M, 3 * H));
      const late = lateNight(a, date), go = late ? 1 : wd === 5 || wd === 6 ? 0.85 : 0.4;
      if (r() > go) { act(home, 'home', 24 * H); break; }
      const t = u(18 * H + 30 * M, 20 * H + 45 * M);
      act(home, 'home', t);
      if (late) { act(midnightPlace(a, date, P), 'late', 24 * H); break; }   // (still out at midnight)
      act(P.poi(r, 'food', 'leisure'), 'dinner', t + u(80 * M, 160 * M));
      act(home, 'home', 24 * H);
      break;
    }
    case 'nightShift': {   // 22:00-06:00, five nights a week (Sunday to Thursday nights): hospitals, cleaning, logistics
      const work: Where = a.id % 2 ? P.poi(rng(a.id), 'health', 'office', 'services', 'supermarket') : { kind: 'edge' };
      const purpose = work.kind === 'edge' ? 'out' : 'work';
      if (wd >= 1 && wd <= 5) act(work, purpose, u(5 * H + 50 * M, 6 * H + 30 * M));   // the end of last night's shift
      if (wd <= 4) { act(home, 'home', u(21 * H + 10 * M, 21 * H + 50 * M)); act(work, purpose, 24 * H); }
      else act(home, 'home', 24 * H);
      break;
    }
    case 'earlyShift': {   // 05:00-13:30 on weekdays and most Saturdays: construction, bakeries, markets, cleaning
      if (wd === 0 || (wd === 6 && r() < 0.4)) { act(home, 'home', 24 * H); break; }
      const work: Where = a.id % 3 ? { kind: 'edge' } : P.poi(rng(a.id), 'supermarket', 'shop', 'food', 'office');
      act(home, 'home', u(4 * H + 30 * M, 5 * H + 30 * M));
      act(work, work.kind === 'edge' ? 'out' : 'work', u(13 * H, 14 * H));
      act(home, 'home', 24 * H);
      break;
    }
    case 'errands':
      if (!weekday && wd !== 6) { act(home, 'home', 24 * H); break; }
      act(home, 'home', u(9 * H, 17 * H));
      act(P.poi(r, 'health', 'services', 'leisure', 'shop'), 'errand', p[0].until + u(20 * M, 60 * M));
      act(home, 'home', 24 * H);
      break;
    case 'delivery': {
      if (wd === 0) { act(home, 'away', 24 * H); break; }
      let t = r() < 0.3 ? u(5 * H, 6 * H + 30 * M) : u(8 * H, 10 * H);   // (some restock shops and markets early)
      act(home, 'away', t);
      const stops = 5 + Math.floor(r() * 5);
      for (let k = 0; k < stops; k++) {
        t += u(10 * M, 35 * M);
        act(r() < 0.6 ? P.poi(r, 'shop', 'supermarket', 'food', 'office') : { kind: 'home' }, 'delivery', t + u(4 * M, 9 * M));
        t = p[p.length - 1].until;
      }
      act(home, 'away', 24 * H);
      break;
    }
    case 'taxi': {
      // a fare every 8-30 min; day shift 6-9h to 19-22h, or (3 in 10 taxis) the night shift from 18-20h to 4-5:30h
      // (fewer fares in the small hours, except after Friday and Saturday nights)
      const fares = (t: number, end: number, gap: () => number) => {
        while ((t += gap()) < end) act(r() < 0.5 ? { kind: 'home' } : P.poi(r, 'food', 'leisure', 'health'), 'fare', t);
      };
      if (a.id % 10 < 3) {
        const busyNight = wd === 6 || wd === 0;   // (this morning ends a Friday or Saturday night)
        act(midnightPlace(a, dayBefore(date), P), 'fare', u(5 * M, 20 * M));   // (the fare of midnight)
        fares(p[0].until, u(4 * H, 5 * H + 30 * M), () => (busyNight ? u(10 * M, 30 * M) : u(15 * M, 45 * M)));
        act({ kind: 'edge' }, 'away', u(18 * H, 20 * H));
        fares(p[p.length - 1].until, 24 * H, () => u(8 * M, 30 * M));
        act(midnightPlace(a, date, P), 'fare', 24 * H);   // (on a fare at midnight)
      } else {
        const t = u(6 * H, 9 * H);
        act({ kind: 'edge' }, 'away', t);
        fares(t, u(19 * H, 22 * H), () => u(8 * M, 30 * M));
        act({ kind: 'edge' }, 'away', 24 * H);
      }
      break;
    }
    case 'through': {
      // edge to edge, a few times a day, following the daily traffic curve
      let t = 0;
      const n = weekday ? 3 : 2;
      const times = Array.from({ length: n }, () => {
        // rush hours more likely
        const pk = r();
        return pk < 0.28 ? u(7 * H, 9 * H + 30 * M) : pk < 0.55 ? u(16 * H, 19 * H) : pk < 0.75 ? u(6 * H, 23 * H)
          : u(0, 6 * H);   // night traffic (lorries, long-distance)
      }).sort((x, y) => x - y);
      for (const tt of times) {
        act({ kind: 'edge' }, 'away', Math.max(tt, t + 30 * M));
        act({ kind: 'edge' }, 'passing', 0);   // (arrives at the other edge and goes on)
        t = p[p.length - 2].until;
        p[p.length - 1].until = t + 20 * M;    // (about the crossing; arriving later, it goes on at once)
      }
      act({ kind: 'edge' }, 'away', 24 * H);
      break;
    }
  }
  return settle(p);
}
