// Per-frame CPU profiler: named sections of the main loop (simulation, LOD, render submit, ...).
// Cheap enough to leave on: two performance.now() calls per section, results in fixed ring buffers.
//
//   prof.begin('traffic'); ...; prof.end('traffic');
//   prof.endFrame();                       once per animation tick
//   prof.report()                          { section: { avg, p95, max } } over the last ~10 s
//
// Hitches: prof.longTasks counts main-thread tasks over 50 ms (Chromium's longtask observer), which also
// catches work outside the loop (tile uploads, fetch callbacks, garbage collection inside them).

const N = 600;
const SPIKE = 25;   // ms

class Ring {
  readonly v = new Float32Array(N);
  i = 0; n = 0;
  push(x: number) { this.v[this.i] = x; this.i = (this.i + 1) % N; if (this.n < N) this.n++; }
  stats() {
    const s = Array.from(this.v.subarray(0, this.n)).sort((a, b) => a - b);
    const r = (x: number) => Math.round(x * 100) / 100;
    if (!s.length) return { avg: 0, p95: 0, max: 0 };
    return { avg: r(s.reduce((a, b) => a + b, 0) / s.length), p95: r(s[Math.floor(s.length * 0.95)]), max: r(s[s.length - 1]) };
  }
}

class Prof {
  private start = new Map<string, number>();
  private acc = new Map<string, number>();       // this frame
  private rings = new Map<string, Ring>();
  longTasks: { t: number; ms: number }[] = [];

  constructor() {
    try {
      new PerformanceObserver((l) => {
        for (const e of l.getEntries()) {
          this.longTasks.push({ t: Math.round(e.startTime), ms: Math.round(e.duration) });
          if (this.longTasks.length > 200) this.longTasks.shift();
        }
      }).observe({ type: 'longtask', buffered: true });
    } catch { /* not Chromium */ }
  }

  begin(name: string) { this.start.set(name, performance.now()); }
  end(name: string) {
    const t = this.start.get(name);
    if (t === undefined) return;
    this.acc.set(name, (this.acc.get(name) ?? 0) + performance.now() - t);
    this.start.delete(name);
  }
  /** close the frame: every known section gets a sample (0 if it did not run) */
  /** the slowest frames (> SPIKE ms of loop work) with their sections: what causes a hitch */
  spikes: { t: number; total: number; parts: Record<string, number> }[] = [];
  endFrame() {
    let total = 0;
    for (const v of this.acc.values()) total += v;
    if (total > SPIKE) {
      const parts: Record<string, number> = {};
      for (const [k, v] of this.acc) if (v > 1) parts[k] = Math.round(v * 10) / 10;
      this.spikes.push({ t: Math.round(performance.now()), total: Math.round(total), parts });
      if (this.spikes.length > 50) this.spikes.shift();
    }
    for (const k of this.acc.keys()) if (!this.rings.has(k)) this.rings.set(k, new Ring());
    for (const [k, r] of this.rings) r.push(this.acc.get(k) ?? 0);
    this.acc.clear();
  }
  report() {
    const out: Record<string, { avg: number; p95: number; max: number }> = {};
    for (const [k, r] of this.rings) out[k] = r.stats();
    return out;
  }
  // GPU time per render pass (from the renderer's timestamp queries, arrives a few frames late)
  private gpu = new Map<string, Ring>();
  gpuSample(pass: string, ms: number) {
    let r = this.gpu.get(pass);
    if (!r) this.gpu.set(pass, (r = new Ring()));
    r.push(ms);
  }
  gpuReport() {
    const out: Record<string, { avg: number; p95: number; max: number }> = {};
    for (const [k, r] of this.gpu) out[k] = r.stats();
    return out;
  }
  reset() { this.rings.clear(); this.gpu.clear(); this.longTasks = []; this.spikes = []; }
  /** one line for the HUD: the sections with the largest average */
  line(top = 6) {
    return Object.entries(this.report()).sort((a, b) => b[1].avg - a[1].avg).slice(0, top)
      .map(([k, s]) => `${k} ${s.avg.toFixed(1)}/${s.max.toFixed(0)}`).join(' · ');
  }
}

export const prof = new Prof();
