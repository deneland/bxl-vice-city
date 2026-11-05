// Noise map: the averaged sound level (Leq, dB(A)) the simulated traffic produces at every point of the
// ground plan, shown as a colour layer over the city (N). Vehicles emit sound power by type and speed
// (rough CNOSSOS-like figures), horns / bells / sirens add theirs; the worker (noise.worker.ts) spreads the
// energy over the map and divides by how long each area was simulated. The layer is a texture sampled in
// the resolve pass (deferred.ts), so switching it on costs no city redraw.
import * as THREE from 'three/webgpu';
import { mix, select, smoothstep, texture, uniform, vec2, vec3 } from 'three/tsl';
import { gridOrigin } from './nightlights';

const CELL = 4;                       // m per map cell
const DB = [40, 85] as const;         // byte 1..255 of the map covers this range
const OBSERVED_RADIUS = 700;          // m around the focus where traffic is simulated in full (agents.ts: 900)
const SEND = 0.2;                     // s between batches to the worker
const COMPUTE = 2.5;                  // s between map updates while the layer is on

/** 1 while the layer is shown */
export const noiseOn = uniform(0);
const noiseTex = texture(new THREE.DataTexture(new Uint8Array(1), 1, 1, THREE.RedFormat));
const noiseSize = uniform(new THREE.Vector2(1, 1));

/** map value (0: not observed, else 1/255 .. 1) at a scene position */
export function noiseLevelAt(world: any) {
  const rel = world.sub(gridOrigin);
  return noiseTex.sample(vec2(rel.x, rel.z.negate()).div(noiseSize)).r;
}

/** colour scale of noise maps (END-style): green below 50, yellow, orange, red, purple, dark blue above 75 */
const STOPS: [number, number][] = [[45, 0x2e9e45], [52, 0xb9d63a], [58, 0xf2d62a], [64, 0xf0901f], [69, 0xd9261c], [74, 0x9a1b6e], [80, 0x3a2a8c]];
const lin = (h: number) => { const c = new THREE.Color(h); return vec3(c.r, c.g, c.b); };
export function noiseRamp(db: any) {
  let c: any = lin(STOPS[0][1]);
  for (let i = 1; i < STOPS.length; i++) c = mix(c, lin(STOPS[i][1]), smoothstep(STOPS[i - 1][0], STOPS[i][0], db));
  return c;
}

/** the lit city colour tinted by the map value (keeps the shading, so streets and walls stay readable) */
export function noiseOverlay(col: any, v: any) {
  const db = v.mul(255).sub(1).div(254).mul(DB[1] - DB[0]).add(DB[0]);
  const lum = col.dot(vec3(0.299, 0.587, 0.114));
  const tinted = noiseRamp(db).mul(lum.mul(1.5).add(0.06));
  const seen = smoothstep(0.5 / 255, 2 / 255, v);
  return select(noiseOn.greaterThan(0.5), mix(col, tinted, seen.mul(0.85)), col);
}

/** sound power (dB(A) re 1 pW) of a vehicle by type and speed (m/s): propulsion + rolling noise */
export function vehicleLw(kind: string, v: number, siren: boolean): number {
  const sp = Math.max(v, 1) / 13.9;   // relative to 50 km/h
  const sum = (...l: number[]) => 10 * Math.log10(l.reduce((a, x) => a + 10 ** (x / 10), 0));
  let lw: number;
  switch (kind) {
    case 'bike': return 0;
    case 'scooter': lw = v > 0.5 ? sum(88, 86 + 30 * Math.log10(sp)) : 0; break;
    case 'bus': lw = sum(90, 99 + 30 * Math.log10(sp)); break;
    case 'van': lw = sum(82, 97 + 33 * Math.log10(sp)); break;
    case 't3000': case 't4000': lw = v > 0.3 ? sum(84, 100 + 30 * Math.log10(sp)) : 75; break;
    case 'train': lw = v > 0.3 ? sum(92, 104 + 30 * Math.log10(sp)) : 85; break;
    default: lw = sum(80, 95.5 + 33 * Math.log10(sp));   // cars (and police, ambulance)
  }
  return siren ? sum(lw, 118) : lw;
}
export const dbToWatt = (db: number) => 10 ** (db / 10) * 1e-12;

export class NoiseMap {
  private worker: Worker;
  private batch: number[] = [];
  private sinceSend = 0;
  private sinceCompute = COMPUTE;
  private busy = false;
  private tex: THREE.DataTexture;
  private focus: [number, number] = [0, 0];

  /** region size (m, east × north); onUpdate: a new map arrived (redraw) */
  constructor(size: [number, number], private onUpdate: () => void) {
    const w = Math.ceil(size[0] / CELL), h = Math.ceil(size[1] / CELL);
    this.tex = new THREE.DataTexture(new Uint8Array(w * h), w, h, THREE.RedFormat, THREE.UnsignedByteType);
    this.tex.magFilter = this.tex.minFilter = THREE.LinearFilter;
    this.tex.unpackAlignment = 1;
    this.tex.needsUpdate = true;
    noiseTex.value = this.tex;
    noiseSize.value.set(w * CELL, h * CELL);
    this.worker = new Worker(new URL('./noise.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e) => {
      if (e.data.type !== 'map') return;
      this.busy = false;
      this.tex.image.data = e.data.data;
      this.tex.needsUpdate = true;
      this.onUpdate();
    };
    this.worker.postMessage({ type: 'init', w, h, cell: CELL, db: DB });
  }

  get visible() { return noiseOn.value > 0.5; }
  set visible(on: boolean) { noiseOn.value = on ? 1 : 0; if (on) this.sinceCompute = COMPUTE; }

  /** sound energy emitted at a scene position (group frame: x east, z south) over dt seconds */
  emit(x: number, z: number, lw: number, dt: number) {
    if (lw > 0) this.batch.push(x, -z, dbToWatt(lw) * dt);
  }
  setFocus(x: number, z: number) { this.focus[0] = x; this.focus[1] = -z; }

  /** once per frame: send the batch now and then; recompute the map while it is shown */
  update(dt: number) {
    this.sinceSend += dt;
    if (this.sinceSend >= SEND) {
      const src = new Float32Array(this.batch);
      this.worker.postMessage({ type: 'add', src, focus: this.focus, radius: OBSERVED_RADIUS, dt: this.sinceSend }, [src.buffer]);
      this.batch.length = 0;
      this.sinceSend = 0;
    }
    this.sinceCompute += dt;
    if (this.visible && !this.busy && this.sinceCompute >= COMPUTE) {
      this.busy = true;
      this.sinceCompute = 0;
      this.worker.postMessage({ type: 'compute' });
    }
  }

  /** (tests) the last map: share of cells observed, and the level histogram of those (dB) */
  stats() {
    const d = this.tex.image.data as Uint8Array;
    let seen = 0, sum = 0, max = 0;
    for (const v of d) if (v) { seen++; const db = DB[0] + ((v - 1) / 254) * (DB[1] - DB[0]); sum += db; max = Math.max(max, db); }
    return { seen: seen / d.length, mean: seen ? sum / seen : 0, max };
  }

  reset() { this.worker.postMessage({ type: 'reset' }); }

  /** legend: the colour scale as HTML */
  static legend() {
    const hex = (h: number) => '#' + h.toString(16).padStart(6, '0');
    return '<b>noise</b> Leq dB(A), traffic since you started watching<br>' +
      STOPS.map(([db, h], i) => `<span style="background:${hex(h)}"></span>${i === 0 ? '< ' : ''}${db}${i === STOPS.length - 1 ? '+' : ''}`).join('<br>');
  }
}
