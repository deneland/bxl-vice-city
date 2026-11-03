// Dev-only session telemetry. Posts summaries, errors and snapshots to the
// Vite dev server (POST /__metrics, see vite.config.ts), which appends them to
// data/metrics/. Nothing is sent in production builds.

export const TELEMETRY = import.meta.env.DEV && new URLSearchParams(location.search).get('telemetry') !== '0';

const ENDPOINT = '/__metrics';
const FLUSH_MS = 5000;
const KEEP_RECENT = 600; // frames kept for snapshots

export interface FrameInfo {
  interval: number;   // ms since previous frame (what the user feels)
  cpu: number;        // ms spent in our update + render submit
  moving: boolean;    // camera changed this frame
  drawCalls: number;
  triangles: number;
}

const pct = (sorted: number[], p: number) =>
  sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))] : 0;

function stats(v: number[]) {
  if (!v.length) return null;
  const s = [...v].sort((a, b) => a - b);
  const r = (x: number) => Math.round(x * 100) / 100;
  return { n: s.length, avg: r(s.reduce((a, b) => a + b, 0) / s.length), p50: r(pct(s, 0.5)),
    p95: r(pct(s, 0.95)), p99: r(pct(s, 0.99)), max: r(s[s.length - 1]) };
}

function post(body: unknown, beacon = false) {
  const json = JSON.stringify(body);
  if (beacon && navigator.sendBeacon) { navigator.sendBeacon(ENDPOINT, json); return; }
  fetch(ENDPOINT, { method: 'POST', body: json, headers: { 'content-type': 'application/json' } }).catch(() => {});
}

class Telemetry {
  readonly session = `${new Date().toISOString().replace(/[:.]/g, '-')}-${Math.random().toString(36).slice(2, 6)}`;
  private t0 = performance.now();
  private seq = 0;
  private frames: FrameInfo[] = [];
  private gpu: number[] = [];
  private idle = 0;
  private recent: FrameInfo[] = [];
  private env: Record<string, unknown> = {};
  private state: () => Record<string, unknown> = () => ({});
  private timer = 0;

  start() {
    if (!TELEMETRY) return;
    this.env = {
      url: location.pathname + location.search,
      ua: navigator.userAgent,
      screen: { w: innerWidth, h: innerHeight, dpr: devicePixelRatio },
      cores: navigator.hardwareConcurrency,
    };
    addEventListener('error', (e) => this.error('error', `${e.message} @ ${e.filename}:${e.lineno}`));
    addEventListener('unhandledrejection', (e) => this.error('unhandledrejection', String(e.reason?.stack ?? e.reason)));
    for (const level of ['error', 'warn'] as const) {
      const orig = console[level].bind(console);
      console[level] = (...args: unknown[]) => {
        orig(...args);
        this.error(`console.${level}`, args.map((a) => (a instanceof Error ? a.stack : String(a))).join(' '));
      };
    }
    addEventListener('pagehide', () => this.flush('pagehide', true));
    document.addEventListener('visibilitychange', () => { if (document.hidden) this.flush('hidden', true); });
    this.timer = window.setInterval(() => this.flush('interval'), FLUSH_MS);
    post({ type: 'start', session: this.session, env: this.env });
  }

  setEnv(extra: Record<string, unknown>) { Object.assign(this.env, extra); if (TELEMETRY) post({ type: 'env', session: this.session, env: this.env }); }
  setState(fn: () => Record<string, unknown>) { this.state = fn; }

  /** One-off events: load milestones, settings changes, … */
  event(name: string, data: Record<string, unknown> = {}) {
    if (TELEMETRY) post({ type: 'event', session: this.session, t: this.now(), name, ...data });
  }

  error(kind: string, message: string) {
    if (!TELEMETRY) return;
    if (message.includes('[vite]')) return;
    post({ type: 'error', session: this.session, t: this.now(), kind, message: message.slice(0, 4000) });
  }

  frame(f: FrameInfo) {
    if (!TELEMETRY) return;
    this.frames.push(f);
    this.recent.push(f);
    if (this.recent.length > KEEP_RECENT) this.recent.shift();
  }

  /** an animation tick where nothing changed and nothing was drawn */
  idleTick() { if (TELEMETRY) this.idle++; }

  gpuTime(ms: number) { if (TELEMETRY && ms > 0) this.gpu.push(ms); }

  flush(reason: string, beacon = false) {
    if (!TELEMETRY || (!this.frames.length && !this.idle && reason === 'interval')) return;
    const f = this.frames;
    const moving = f.filter((x) => x.moving);
    const intervals = f.map((x) => x.interval);
    post({
      type: 'summary', session: this.session, seq: this.seq++, t: this.now(), reason,
      visible: !document.hidden,
      frames: f.length,
      idleTicks: this.idle,
      movingFrames: moving.length,
      interval: stats(intervals),
      intervalMoving: stats(moving.map((x) => x.interval)),
      cpu: stats(f.map((x) => x.cpu)),
      gpu: stats(this.gpu),
      longFrames: { over20: intervals.filter((x) => x > 20).length, over33: intervals.filter((x) => x > 33).length,
        over50: intervals.filter((x) => x > 50).length },
      drawCalls: stats(f.map((x) => x.drawCalls))?.avg,
      triangles: stats(f.map((x) => x.triangles))?.avg,
      memMB: (performance as any).memory ? Math.round((performance as any).memory.usedJSHeapSize / 1e6) : undefined,
      ...this.state(),
    }, beacon);
    this.frames = [];
    this.gpu = [];
    this.idle = 0;
  }

  snapshot(canvas: HTMLCanvasElement, note = '') {
    if (!TELEMETRY) return;
    const id = `${this.session}-snap${this.seq++}`;
    const image = canvas.toDataURL('image/png');
    post({ type: 'snapshot', session: this.session, id, t: this.now(), note, recent: this.recent.slice(),
      ...this.state(), image });
    return id;
  }

  stop() { clearInterval(this.timer); }
  private now() { return Math.round(performance.now() - this.t0); }
}

export const telemetry = new Telemetry();
