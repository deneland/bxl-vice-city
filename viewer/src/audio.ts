// Sound engine (Web Audio): sounds generated at start-up (no recordings), placed in 3D around the listener.
//
//  voices     a few continuous sources that follow vehicles (engine / tyre roar, a tram's motor whine, a
//             siren), each through a panner: level by distance (inverse: 6 dB per doubling), Doppler shift
//             from the vehicle's speed towards the listener
//  one-shots  horn, tram bell, bus air hiss: played once at a position
//  bed        the rest of the traffic, too many or too far to place one by one, as one filtered rumble
//             whose level is the sum of their contributions
//
// Levels follow the sources' sound power (dB(A)): gain = 10^((Lw − REF) / 20) at REF_DIST metres.
const REF = 94;           // dB(A) sound power that plays at gain 1 ...
export const REF_DIST = 5;   // ... at this distance (m)

export const lwGain = (lw: number) => 10 ** ((lw - REF) / 20);


/** sounds generated once (mono) */
function synth(ctx: BaseAudioContext) {
  const sr = ctx.sampleRate;
  const make = (sec: number, f: (t: number, i: number) => number) => {
    const n = Math.round(sec * sr), b = ctx.createBuffer(1, n, sr), d = b.getChannelData(0);
    for (let i = 0; i < n; i++) d[i] = f(i / sr, i);
    return b;
  };
  const norm = (b: AudioBuffer, peak = 0.9) => {
    const d = b.getChannelData(0);
    let m = 0;
    for (let i = 0; i < d.length; i++) m = Math.max(m, Math.abs(d[i]));
    if (m > 0) for (let i = 0; i < d.length; i++) d[i] *= peak / m;
    return b;
  };
  // a loop with no seam: the last 50 ms crossfaded into the start
  const seamless = (b: AudioBuffer) => {
    const d = b.getChannelData(0), n = Math.round(0.05 * sr), L = d.length - n;
    for (let i = 0; i < n; i++) { const k = i / n; d[i] = d[i] * k + d[L + i] * (1 - k); }
    const out = ctx.createBuffer(1, L, sr);
    out.copyToChannel(d.subarray(0, L), 0);
    return out;
  };
  // brown-ish noise (road / engine roar), then a gentle low-pass
  const brown = (sec: number, leak: number) => {
    let y = 0, z = 0;
    return seamless(norm(make(sec, () => { y = y * leak + (Math.random() * 2 - 1) * 0.1; z += (y - z) * 0.2; return z; })));
  };
  // bell: inharmonic partials with their own decays, a short strike click; `strikes` seconds
  const bell = (f0: number, strikes: number[], ring: number) => {
    const P = [[1, 1, 1], [2.0, 0.5, 0.7], [2.76, 0.45, 0.55], [4.07, 0.22, 0.35], [5.4, 0.14, 0.25], [8.93, 0.06, 0.15]];
    return norm(make(strikes[strikes.length - 1] + ring, (t) => {
      let v = 0;
      for (const s of strikes) {
        const u = t - s;
        if (u < 0) continue;
        for (const [r, a, d] of P) v += a * Math.exp(-u / (d * ring)) * Math.sin(2 * Math.PI * f0 * r * u + r);
        if (u < 0.004) v += (Math.random() * 2 - 1) * 0.6 * (1 - u / 0.004);   // strike
      }
      return v;
    }), 0.8);
  };
  // horn: two tones a third apart, rich in harmonics, soft-clipped, short attack / release
  const horn = (dur: number[], f1: number, f2: number) => {
    const end = dur[dur.length - 1];
    return norm(make(end + 0.05, (t) => {
      let on = 0;
      for (let k = 0; k + 1 < dur.length; k += 2)
        if (t >= dur[k] && t < dur[k + 1] + 0.04) on = Math.min(1, (t - dur[k]) / 0.012, (dur[k + 1] + 0.04 - t) / 0.04);
      if (on <= 0) return 0;
      let v = 0;
      for (let h = 1; h <= 9; h++) v += (Math.sin(2 * Math.PI * f1 * h * t) + Math.sin(2 * Math.PI * f2 * h * t)) / h;
      return Math.tanh(v * 0.8) * on;
    }), 0.7);
  };
  // European two-tone siren (high / low, 0.55 s each), square-ish, as a seamless loop
  const siren = () => {
    const T = 0.55, fa = Math.round(587 * T) / T, fb = Math.round(440 * T) / T;
    let ph = 0;
    return norm(make(2 * T, (t) => {
      const f = t < T ? fa : fb;
      ph += f / sr;
      let v = 0;
      for (let h = 1; h <= 11; h += 2) v += Math.sin(2 * Math.PI * h * ph) / h;
      return Math.tanh(v * 1.5);
    }), 0.8);
  };
  // air hiss (bus kneeling / doors): high-passed noise, fast attack, decay
  const hiss = (sec: number) => {
    let lp = 0;
    return norm(make(sec, (t) => {
      const w = Math.random() * 2 - 1;
      lp += (w - lp) * 0.15;
      return (w - lp) * Math.min(1, t / 0.03) * Math.exp(-t / (sec * 0.35));
    }), 0.6);
  };
  return {
    roar: brown(4, 0.995),
    rumble: brown(4, 0.999),
    // STIB Flexity bell: a bright double "ding-ding" when pulling away; a quick series to warn
    bell: bell(1180, [0, 0.27], 0.9),
    warn: bell(1180, [0, 0.16, 0.32, 0.48, 0.64], 0.7),
    horns: [horn([0, 0.18], 420, 525), horn([0, 0.12, 0.22, 0.34], 405, 505), horn([0, 0.75], 440, 550), horn([0, 0.3], 350, 440)],
    siren: siren(),
    hiss: hiss(1.1),
  };
}
type Sounds = ReturnType<typeof synth>;

/** a continuous source following something: base loop (and optional extra loop, e.g. a siren) → panner */
export class Voice {
  readonly panner: PannerNode;
  private gain: GainNode;
  private filter: BiquadFilterNode;
  private src: AudioBufferSourceNode;
  private extra: AudioBufferSourceNode | null = null;
  private extraGain: GainNode;
  private whine: OscillatorNode;
  private whineGain: GainNode;
  /** what it follows (null: free, fading out) */
  owner: object | null = null;
  freeSince = 0;
  lastDist = -1;

  constructor(private ctx: AudioContext, out: AudioNode, private snd: Sounds) {
    this.panner = new PannerNode(ctx, { panningModel: 'equalpower', distanceModel: 'inverse', refDistance: REF_DIST,
      rolloffFactor: 1, maxDistance: 10000 });
    this.panner.connect(out);
    this.gain = new GainNode(ctx, { gain: 0 });
    this.gain.connect(this.panner);
    this.filter = new BiquadFilterNode(ctx, { type: 'lowpass', frequency: 600, Q: 0.5 });
    this.filter.connect(this.gain);
    this.src = new AudioBufferSourceNode(ctx, { buffer: snd.roar, loop: true });
    this.src.connect(this.filter);
    this.src.start(ctx.currentTime, Math.random() * snd.roar.duration);
    this.extraGain = new GainNode(ctx, { gain: 0 });
    this.extraGain.connect(this.panner);
    // electric motor whine (trams), silent otherwise
    this.whine = new OscillatorNode(ctx, { type: 'triangle', frequency: 200 });
    this.whineGain = new GainNode(ctx, { gain: 0 });
    this.whine.connect(this.whineGain).connect(this.panner);
    this.whine.start();
  }

  /** follow `owner` (a new one: starts silent and fades in) */
  assign(owner: object, siren: boolean) {
    this.owner = owner;
    this.lastDist = -1;
    if (siren && !this.extra) {
      this.extra = new AudioBufferSourceNode(this.ctx, { buffer: this.snd.siren, loop: true });
      this.extra.connect(this.extraGain);
      this.extra.start(this.ctx.currentTime, Math.random() * this.snd.siren.duration);
    } else if (!siren && this.extra) { this.extra.stop(); this.extra.disconnect(); this.extra = null; }
  }
  release(now: number) {
    this.owner = null;
    this.freeSince = now;
    const t = this.ctx.currentTime;
    this.gain.gain.setTargetAtTime(0, t, 0.08);
    this.extraGain.gain.setTargetAtTime(0, t, 0.08);
    this.whineGain.gain.setTargetAtTime(0, t, 0.08);
  }

  /** position (listener frame, m), roar level (gain), low-pass cutoff (Hz), siren level, tram whine (Hz, level),
   *  Doppler factor */
  set(x: number, y: number, z: number, roar: number, cutoff: number, siren: number, whineHz: number, whine: number, doppler: number) {
    const t = this.ctx.currentTime, k = 0.06;
    const p = this.panner;
    if (this.gain.gain.value < 1e-4) { p.positionX.value = x; p.positionY.value = y; p.positionZ.value = z; }   // (silent: jump)
    else { p.positionX.setTargetAtTime(x, t, k); p.positionY.setTargetAtTime(y, t, k); p.positionZ.setTargetAtTime(z, t, k); }
    this.gain.gain.setTargetAtTime(roar, t, 0.12);
    this.filter.frequency.setTargetAtTime(cutoff, t, 0.1);
    this.src.playbackRate.setTargetAtTime(doppler, t, 0.1);
    this.extraGain.gain.setTargetAtTime(siren, t, 0.1);
    if (this.extra) this.extra.playbackRate.setTargetAtTime(doppler, t, 0.1);
    this.whineGain.gain.setTargetAtTime(whine, t, 0.15);
    this.whine.frequency.setTargetAtTime(whineHz * doppler, t, 0.1);
  }
}

export class AudioEngine {
  readonly ctx: AudioContext;
  readonly snd: Sounds;
  readonly out: GainNode;
  private bedGain: GainNode;
  private bedFilter: BiquadFilterNode;
  private oneShots = 0;

  constructor() {
    this.ctx = new AudioContext();
    this.snd = synth(this.ctx);
    // master: gain, then a compressor so a siren next to you does not clip
    const comp = new DynamicsCompressorNode(this.ctx, { threshold: -14, knee: 10, ratio: 6, attack: 0.005, release: 0.25 });
    comp.connect(this.ctx.destination);
    this.out = new GainNode(this.ctx, { gain: 0.6 });
    this.out.connect(comp);
    // bed: low rumble of distant traffic, not placed
    this.bedFilter = new BiquadFilterNode(this.ctx, { type: 'lowpass', frequency: 350, Q: 0.3 });
    this.bedGain = new GainNode(this.ctx, { gain: 0 });
    const bed = new AudioBufferSourceNode(this.ctx, { buffer: this.snd.rumble, loop: true });
    bed.connect(this.bedFilter).connect(this.bedGain).connect(this.out);
    bed.start();
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) this.ctx.suspend(); else if (this.enabled) this.ctx.resume();
    });
  }

  enabled = true;
  setEnabled(on: boolean) {
    this.enabled = on;
    if (on) this.ctx.resume(); else this.ctx.suspend();
  }

  /** listener at (x, y, z) looking along f with up u (same frame as the sources) */
  setListener(x: number, y: number, z: number, fx: number, fy: number, fz: number, ux: number, uy: number, uz: number) {
    const l = this.ctx.listener, t = this.ctx.currentTime, k = 0.05;
    if (l.positionX) {
      l.positionX.setTargetAtTime(x, t, k); l.positionY.setTargetAtTime(y, t, k); l.positionZ.setTargetAtTime(z, t, k);
      l.forwardX.setTargetAtTime(fx, t, k); l.forwardY.setTargetAtTime(fy, t, k); l.forwardZ.setTargetAtTime(fz, t, k);
      l.upX.setTargetAtTime(ux, t, k); l.upY.setTargetAtTime(uy, t, k); l.upZ.setTargetAtTime(uz, t, k);
    } else {   // (older Firefox / Safari)
      l.setPosition(x, y, z);
      l.setOrientation(fx, fy, fz, ux, uy, uz);
    }
  }

  /** background traffic: amplitude (same scale as a voice at its distance), brightness 0..1 */
  setBed(amp: number, bright: number) {
    const t = this.ctx.currentTime;
    this.bedGain.gain.setTargetAtTime(amp, t, 0.4);
    this.bedFilter.frequency.setTargetAtTime(250 + 500 * bright, t, 0.4);
  }

  /** play a sound once at a position; gain from the source's power (lwGain); rate: pitch / speed */
  play(buf: AudioBuffer, x: number, y: number, z: number, gain: number, rate = 1) {
    if (this.oneShots >= 8) return;   // (a crowd of horns: enough)
    const ctx = this.ctx;
    const p = new PannerNode(ctx, { panningModel: 'equalpower', distanceModel: 'inverse', refDistance: REF_DIST,
      rolloffFactor: 1, maxDistance: 10000, positionX: x, positionY: y, positionZ: z });
    const g = new GainNode(ctx, { gain });
    const s = new AudioBufferSourceNode(ctx, { buffer: buf, playbackRate: rate });
    s.connect(g).connect(p).connect(this.out);
    this.oneShots++;
    s.onended = () => { this.oneShots--; s.disconnect(); g.disconnect(); p.disconnect(); };
    s.start();
  }
}
