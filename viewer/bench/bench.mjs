// Repeatable performance benchmark: headless Chromium on the local GPU,
// uncapped frame rate, fixed camera views and paths.
//
//   npm run bench                       run, print table, compare to bench/baseline.json
//   npm run bench -- --save-baseline    also store this run as the new baseline
//   npm run bench -- --only street,orbit --query shadows=1 --size 1920x1080
//   npm run bench -- --strict          abort if the GPU is busy with something else
//
// Needs the dev server running (npm run dev). Results + screenshots go to
// data/metrics/bench/<timestamp>/.
import { chromium } from 'playwright-core';
import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '../..');
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(`--${name}`); return i >= 0 ? args[i + 1] : def; };
const flag = (name) => args.includes(`--${name}`);

const URL_BASE = opt('url', 'http://127.0.0.1:5173/');
const [W, H] = opt('size', '1600x1000').split('x').map(Number);
const EXTRA_QUERY = opt('query', '');
const ONLY = opt('only', '')?.split(',').filter(Boolean);
const CHROME = opt('chrome', process.env.CHROME ?? '/usr/bin/chromium');
const cfg = JSON.parse(readFileSync(resolve(HERE, 'scenarios.json'), 'utf8'));
const scenarios = cfg.scenarios.filter((s) => !ONLY.length || ONLY.includes(s.name));

const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const OUT = resolve(ROOT, 'data/metrics/bench', stamp);
mkdirSync(OUT, { recursive: true });

const pct = (s, p) => s.length ? s[Math.min(s.length - 1, Math.floor(s.length * p))] : null;
const r2 = (x) => (x == null ? null : Math.round(x * 100) / 100);
function stats(v) {
  if (!v.length) return null;
  const s = [...v].sort((a, b) => a - b);
  return { avg: r2(s.reduce((a, b) => a + b, 0) / s.length), p50: r2(pct(s, 0.5)), p95: r2(pct(s, 0.95)),
    p99: r2(pct(s, 0.99)), max: r2(s.at(-1)) };
}

// Background GPU load (e.g. a viewer tab open in your browser) skews results: sample it first.
function gpuLoad(samples = 6) {
  try {
    const v = [];
    for (let i = 0; i < samples; i++) {
      v.push(Number(execFileSync('nvidia-smi', ['--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']).toString().trim().split('\n')[0]));
      execFileSync('sleep', ['0.3']);
    }
    return Math.round(v.reduce((a, b) => a + b, 0) / v.length);
  } catch { return null; }
}
const idleBefore = gpuLoad();
if (idleBefore != null && idleBefore > 10) {
  console.log(`⚠ GPU is ${idleBefore}% busy before the run (another app or an open viewer tab). Results will be skewed.`);
  if (flag('strict')) { console.log('--strict: aborting'); process.exit(2); }
}

const browser = await chromium.launch({
  executablePath: CHROME, headless: true,
  args: ['--ignore-gpu-blocklist', '--enable-features=Vulkan', '--use-angle=vulkan', '--disable-vulkan-surface',
    '--enable-unsafe-webgpu', '--disable-gpu-vsync', '--disable-frame-rate-limit'],
});
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });

const query = ['telemetry=0', 'gpu=1', 'continuous=1', EXTRA_QUERY].filter(Boolean).join('&');
const tNav = Date.now();
await page.goto(`${URL_BASE}?${query}`);
await page.waitForFunction(() => window.__viewer, null, { timeout: 30000 });
const load = await page.evaluate(async () => { await window.__viewer.ready; return { readyMs: Math.round(performance.now()), ...window.__viewer.info() }; });
load.wallMs = Date.now() - tNav;
const env = await page.evaluate(() => {
  const be = window.__viewer.renderer.backend;
  const gl = be.gl;
  const ext = gl?.getExtension('WEBGL_debug_renderer_info');
  return { backend: be.isWebGPUBackend ? 'WebGPU' : 'WebGL2', gpu: gl ? gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER) : 'webgpu',
    ua: navigator.userAgent, mem: performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null };
});

// runs N frames in the page, optionally moving the camera along a path each frame
const measure = (s) => page.evaluate(({ s, frames, warmup }) => new Promise((done) => {
  const v = window.__viewer;
  v.setView(s.view);
  const P = s.view.pos, T = s.view.target;
  const at = (t) => {
    if (s.path === 'orbit') {
      const dx = P[0] - T[0], dz = P[2] - T[2], a = t * Math.PI * 2;
      return { pos: [T[0] + dx * Math.cos(a) - dz * Math.sin(a), P[1], T[2] + dx * Math.sin(a) + dz * Math.cos(a)], target: T };
    }
    if (s.path === 'pan') {
      const d = 400 * t;
      return { pos: [P[0] + d, P[1], P[2]], target: [T[0] + d, T[1], T[2]] };
    }
    if (s.path === 'dolly') {
      const k = 1 - 0.95 * t; // fly towards the target
      return { pos: [T[0] + (P[0] - T[0]) * k, T[1] + (P[1] - T[1]) * k + 2, T[2] + (P[2] - T[2]) * k], target: T };
    }
    return null;
  };
  const rec = [], gpu = [];
  let i = 0;
  const gh = (ms) => { if (i > warmup) gpu.push(ms); };
  const fh = (f) => {
    i++;
    if (i > warmup) rec.push(f);
    const t = Math.max(0, (i - warmup) / frames);
    const view = at(t);
    if (view) { v.controls.enableDamping = false; v.setView(view); }
    if (i >= warmup + frames) {
      v.frameHooks.delete(fh); v.gpuHooks.delete(gh); v.controls.enableDamping = true;
      done({ rec, gpu });
    }
  };
  v.gpuHooks.add(gh);
  v.frameHooks.add(fh);
}), { s, frames: cfg.frames, warmup: cfg.warmupFrames });

const results = [];
for (const s of scenarios) {
  // settle: move the camera, let LOD tiles for this view finish loading (paths load while moving)
  await page.evaluate(async (view) => {
    const v = window.__viewer;
    v.setView(view);
    for (let i = 0; i < 3; i++) await new Promise((r) => requestAnimationFrame(r));
    await v.whenIdle?.();
  }, s.view);
  const { rec, gpu } = await measure(s);
  const lod = await page.evaluate(() => window.__viewer.lod?.stats ?? null);
  if (!s.path) await page.screenshot({ path: resolve(OUT, `${s.name}.png`) });
  results.push({
    name: s.name, path: s.path ?? null,
    interval: stats(rec.map((f) => f.interval)),
    cpu: stats(rec.map((f) => f.cpu)),
    gpu: stats(gpu),
    drawCalls: Math.round(stats(rec.map((f) => f.drawCalls)).avg),
    trianglesM: r2(stats(rec.map((f) => f.triangles)).avg / 1e6),
    lod,
  });
  process.stdout.write('.');
}
process.stdout.write('\n');
await browser.close();

const idleAfter = gpuLoad();
const run = { gpuBusyBefore: idleBefore, gpuBusyAfter: idleAfter, stamp, url: `${URL_BASE}?${query}`, size: [W, H], env, load, results, errors: [...new Set(errors)] };
writeFileSync(resolve(OUT, 'results.json'), JSON.stringify(run, null, 1));

// --- report ------------------------------------------------------------------
const basePath = resolve(HERE, 'baseline.json');
const base = existsSync(basePath) ? JSON.parse(readFileSync(basePath, 'utf8')) : null;
const delta = (now, old) => {
  if (now == null || old == null || old === 0) return '';
  const d = ((now - old) / old) * 100;
  return Math.abs(d) < 3 ? ' (=)' : ` (${d > 0 ? '+' : ''}${d.toFixed(0)}%)`;
};
const fmt = (x) => (x == null ? '–' : x.toFixed(2));
console.log(`GPU busy before/after run: ${idleBefore ?? '?'}% / ${idleAfter ?? '?'}%${(idleBefore > 10 || idleAfter > 10) ? '  ⚠ not idle, compare with care' : ''}`);
console.log(`${env.backend} · ${env.gpu}\n${W}x${H} · query "${query}" · first view ${load.readyMs} ms (${load.ready ?? load.chunks}/${load.total ?? load.chunks} tiles loaded) · heap ${env.mem} MB`);
console.log('\nscenario     gpu avg ms      gpu p95 ms      frame p50  frame p99 ms     cpu p99  draws  tris(M)  tiles L0/L1/L2/L3 (end of run)');
for (const r of results) {
  const b = base?.results.find((x) => x.name === r.name);
  console.log(
    r.name.padEnd(12),
    (fmt(r.gpu?.avg) + delta(r.gpu?.avg, b?.gpu?.avg)).padEnd(15),
    (fmt(r.gpu?.p95) + delta(r.gpu?.p95, b?.gpu?.p95)).padEnd(15),
    fmt(r.interval.p50).padEnd(10),
    (fmt(r.interval.p99) + delta(r.interval.p99, b?.interval?.p99)).padEnd(16),
    fmt(r.cpu.p99).padEnd(9),
    String(r.drawCalls).padEnd(6),
    fmt(r.trianglesM).padEnd(8),
    r.lod ? r.lod.perLevel.join('/') : '',
  );
}
if (base) console.log(`\n(compared with baseline ${base.stamp}; % = change, (=) within 3%)`);
if (run.errors.length) console.log('\nerrors:\n  ' + run.errors.join('\n  '));
console.log(`\nresults + screenshots: ${OUT}`);
if (flag('save-baseline')) { writeFileSync(basePath, JSON.stringify(run, null, 1)); console.log('saved as baseline'); }
