// The city's sounds from the simulation: what each vehicle emits (for the ear; sirens also for the noise map), and the
// events (a horn when stuck behind something that is not a red light, a tram's bell when it pulls away and
// now and then to warn, a bus's air hiss when it kneels at a stop). Runs while traffic runs; the noise map
// always gets the energy, the audio engine (audio.ts) only exists once sound is switched on.
import { AudioEngine, REF_DIST, Voice, lwGain } from './audio';
import { NoiseMap, vehicleLw } from './noisemap';
import { KIND } from './roads';
import type { Traffic } from './traffic';
import type { Vehicle } from './paths';
import * as THREE from 'three/webgpu';

const TICK = 0.1;            // s between updates of sources and events
const VOICES = 10;           // vehicles placed individually
// horns are rare: only after a long wait that is not a red light, and never in quick succession
const HONK_AFTER = 6;        // s stuck (not at a light) before a horn becomes possible
const HONK_RATE = 0.04;      // horns per second per stuck car after that (× aggressiveness)
const HONK_COOLDOWN = 120;   // s between a car's horns
const HONK_GAP = 15;         // s between any two horns in the city
const WARN_RATE = 1 / 400;   // tram warning bells per second per moving tram
const C = 343;               // m/s, speed of sound
const CITY_FLOOR = 0.06;     // bed amplitude of the unsimulated city around (same scale as voices)
const LW = { horn: 112, bell: 106, hiss: 100 };   // dB(A) sound power of the events
const DUR = { horn: 0.5, bell: 1.2, hiss: 1 };    // s (energy for the noise map)

export class CitySound {
  audio: AudioEngine | null = null;
  private voices: Voice[] = [];
  private acc = 0;
  private time = 0;
  private nextHonk = new WeakMap<Vehicle, number>();
  private lastHonk = -Infinity;
  private pos = new THREE.Vector3();
  /** listener in the traffic group's frame, its view direction and up */
  readonly ear = new THREE.Vector3();
  readonly fwd = new THREE.Vector3(0, 0, -1);
  readonly up = new THREE.Vector3(0, 1, 0);

  constructor(private traffic: Traffic, private noise: NoiseMap) {
    traffic.onTransit = (v, e) => {
      const p = traffic.positionOf(v, this.pos);
      if (v.kind === KIND.tram && e === 'depart') this.event('bell', p, this.audio?.snd.bell, 1);
      if (v.kind === KIND.road && e === 'arrive') this.event('hiss', p, this.audio?.snd.hiss, 0.9 + Math.random() * 0.2);
    };
  }

  /** switch the sound on (creates the audio engine: must be called from a click / key press) or off */
  setSound(on: boolean) {
    if (on && !this.audio) {
      this.audio = new AudioEngine();
      for (let i = 0; i < VOICES; i++) this.voices.push(new Voice(this.audio.ctx, this.audio.out, this.audio.snd));
      this.setActive(this.active);
    }
    this.audio?.setEnabled(on);
  }
  /** traffic shown / hidden: the city falls silent while it is hidden (nothing updates the voices then) */
  setActive(on: boolean) { this.active = on; this.audio?.out.gain.setTargetAtTime(on ? 0.6 : 0, this.audio.ctx.currentTime, 0.1); }
  private active = true;

  /** a one-shot: energy into the noise map; heard if sound is on */
  private event(kind: keyof typeof LW, p: THREE.Vector3, buf: AudioBuffer | undefined, rate: number) {
    this.noise.emit(p.x, p.z, LW[kind], DUR[kind]);
    if (!buf || !this.audio?.enabled) return;
    const g = lwGain(LW[kind]);
    if (g * REF_DIST / Math.max(REF_DIST, p.distanceTo(this.ear)) < 0.002) return;   // (inaudible)
    this.audio.play(buf, p.x, p.y + 1.2, p.z, g, rate);
  }

  update(dt: number) {
    this.acc += dt;
    if (this.acc < TICK) return;
    const step = this.acc;
    this.acc = 0;
    this.time += step;
    const a = this.audio?.enabled ? this.audio : null;
    const ex = this.ear.x, ey = this.ear.y, ez = this.ear.z;
    // loudest sources at the ear; everything else goes into the bed
    const best: { v: Vehicle; amp: number; lw: number; x: number; y: number; z: number; d: number }[] = [];
    let bed = 0, bedBright = 0;
    const p = this.pos;
    for (const v of this.traffic.vehicles) {
      this.traffic.positionOf(v, p);
      const lw = vehicleLw(v.t.name, v.v, !!v.siren);
      if (v.siren) this.noise.emit(p.x, p.z, lw, step);   // (the map has the regular traffic per street type)
      // events
      if (v.kind === KIND.road && !v.transit && !v.siren && v.t.name !== 'bike' && v.stuck > HONK_AFTER
        && this.time - this.lastHonk > HONK_GAP && Math.random() < HONK_RATE * v.ag * step
        && (this.nextHonk.get(v) ?? 0) < this.time) {
        this.nextHonk.set(v, this.time + HONK_COOLDOWN);
        this.lastHonk = this.time;
        const horns = a?.snd.horns;
        this.event('horn', p, horns?.[Math.floor(Math.random() * horns.length)], 0.92 + Math.random() * 0.18);
      }
      if (v.kind === KIND.tram && v.v > 2 && Math.random() < WARN_RATE * step) this.event('bell', p, a?.snd.warn, 1);
      if (!a || lw <= 0) continue;
      const d = Math.max(REF_DIST, Math.hypot(p.x - ex, p.y + 1 - ey, p.z - ez));
      const amp = lwGain(lw) * REF_DIST / d;
      if (amp < 1e-3) continue;
      if (best.length < VOICES + 4 || amp > best[best.length - 1].amp) {
        best.push({ v, amp, lw, x: p.x, y: p.y + 1, z: p.z, d });
        best.sort((u, w) => w.amp - u.amp);
        if (best.length > VOICES + 4) { const o = best.pop()!; bed += o.amp * o.amp; bedBright += o.amp * o.amp * Math.min(1, o.v.v / 14); }
      } else { bed += amp * amp; bedBright += amp * amp * Math.min(1, v.v / 14); }
    }
    if (!a) return;
    // voices: keep the ones whose vehicle is still among the loudest (hysteresis: VOICES + 4), free the
    // others; give free voices (faded out for a moment) to the loudest unvoiced vehicles
    const now = this.time;
    const want = new Map(best.map((b) => [b.v as object, b]));
    for (const vc of this.voices) if (vc.owner && !want.has(vc.owner)) vc.release(now);
    const voiced = new Set(this.voices.map((vc) => vc.owner));
    for (const b of best) {
      if (voiced.has(b.v)) continue;
      const vc = this.voices.find((x) => !x.owner && now - x.freeSince > 0.25);
      if (!vc) break;
      vc.assign(b.v, !!b.v.siren);
      voiced.add(b.v);
    }
    for (const b of best) if (!voiced.has(b.v)) { bed += b.amp * b.amp; bedBright += b.amp * b.amp * Math.min(1, b.v.v / 14); }
    for (const vc of this.voices) {
      const b = vc.owner ? want.get(vc.owner) : null;
      if (!b) continue;
      const v = b.v, tram = v.kind === KIND.tram;
      // Doppler: change of distance since the last tick (receding: lower)
      const vr = vc.lastDist < 0 ? 0 : (b.d - vc.lastDist) / step;
      vc.lastDist = b.d;
      const dop = C / (C + Math.max(-60, Math.min(60, vr)));
      const g = lwGain(v.siren ? vehicleLw(v.t.name, v.v, false) : b.lw);
      const speed = Math.min(1, v.v / 14);
      vc.set(b.x, b.y, b.z, g * (tram ? 0.7 : 1), 220 + 900 * speed + (tram ? 0 : 150), v.siren ? lwGain(118) * 0.05 : 0,
        120 + v.v * 45, tram && v.v > 0.5 ? g * 0.05 * speed : 0, dop);
    }
    // bed: the energy sum of the rest (as an amplitude), plus the city beyond the simulation (far fewer
    // vehicles are simulated than drive in Brussels: without it, quiet streets fall silent)
    a.setBed(Math.sqrt(bed + CITY_FLOOR ** 2), bed > 0 ? bedBright / bed : 0);
    a.setListener(ex, ey, ez, this.fwd.x, this.fwd.y, this.fwd.z, this.up.x, this.up.y, this.up.z);
  }
}
