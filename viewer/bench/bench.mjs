// Repeatable performance benchmark: headless Chromium on the local GPU,
// uncapped frame rate, fixed camera views and paths.
//
//   npm run bench                       run, print table, compare to bench/baseline.json
//   npm run bench -- --save-baseline    also store this run as the new baseline
//   npm run bench -- --only street,orbit --query shadows=1 --size 1920x1080
//   npm run bench -- --strict          abort if the GPU is busy with something else
//   npm run bench -- --laptop          average-laptop profile: 1920x1080, CPU throttled 3x, and the GPU
//                                      time scaled to an integrated laptop GPU (--gpu-factor, default 14)
//   npm run bench -- --backend webgpu  measure the WebGPU path (default: WebGL2)
//   npm run bench -- --profile street  also record a JS CPU profile in that scenario; prints the top functions
//
// The simulation runs at a fixed date / time / weather (--time 8.5 = 08:30, a Wednesday) so runs compare.
// Per scenario: frame interval, CPU per loop section (src/prof.ts), GPU time, long tasks (> 50 ms).
// 'jump' scenarios teleport the camera and measure the hitches while tiles and agents catch up.
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
const LAPTOP = flag('laptop');
const [W, H] = opt('size', LAPTOP ? '1920x1080' : '1600x1000').split('x').map(Number);
const CPU_THROTTLE = Number(opt('cpu-throttle', LAPTOP ? 3 : 1));
// 3090 at full clocks vs an integrated laptop GPU (Iris Xe 96 EU / Radeon 680M): ~12-16x less compute and
// bandwidth. The 3090 rarely reaches full clocks here: bursty frames keep it in its idle state (P8, ~210 MHz
// core / 405 MHz memory, which is itself about Iris Xe class), so the factor is scaled by the clocks seen
// during each scenario: 14 x sqrt(core share x memory share).
const GPU_FACTOR_FULL = Number(opt('gpu-factor', LAPTOP ? 14 : 1));
const gpuFactor = (c) => (!c || GPU_FACTOR_FULL === 1) ? GPU_FACTOR_FULL
  : GPU_FACTOR_FULL * Math.sqrt((c.gr / c.grMax) * (c.mem / c.memMax));
const GPU_FACTOR = GPU_FACTOR_FULL;
// samples the GPU clocks in the background while a scenario runs
function clockSampler() {
  const v = [];
  const t = setInterval(() => {
    try {
      const [gr, mem, grMax, memMax] = execFileSync('nvidia-smi', ['--query-gpu=clocks.gr,clocks.mem,clocks.max.gr,clocks.max.mem', '--format=csv,noheader,nounits'])
        .toString().trim().split('\n')[0].split(',').map(Number);
      v.push({ gr, mem, grMax, memMax });
    } catch { /* no nvidia-smi */ }
  }, 400);
  return () => {
    clearInterval(t);
    if (!v.length) return null;
    const avg = (k) => Math.round(v.reduce((a, x) => a + x[k], 0) / v.length);
    return { gr: avg('gr'), mem: avg('mem'), grMax: v[0].grMax, memMax: v[0].memMax,
      grMin: Math.min(...v.map((x) => x.gr)), grMaxSeen: Math.max(...v.map((x) => x.gr)) };
  };
}
const PROFILE = opt('profile', '');
const TIME = opt('time', '8.5');
const BACKEND = opt('backend', 'webgl');   // webgl (Firefox/Safari path) or webgpu (Chrome on Windows/macOS)
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

// (not ?backend=webgl: forcing WebGL loses the device headless; the bench reloads until it has the backend)
const query = ['telemetry=0', 'gpu=1', 'continuous=1', 'dynres=0', `time=${TIME}`, 'date=2026-11-04', 'weather=clear', EXTRA_QUERY]
  .filter(Boolean).join('&');
if (!/^telemetry=0&gpu=1&continuous=1&dynres=0&time=/.test(query)) throw new Error(`bench query malformed: ${query}`);
// headless Chromium hands out WebGPU only some of the time: reload until we have the backend asked for
let load, tNav;
for (let attempt = 0; attempt < 8; attempt++) {
  tNav = Date.now();
  await page.goto(`${URL_BASE}?${query}`);
  await page.waitForFunction(() => window.__viewer, null, { timeout: 30000 });
  const be = await page.evaluate(() => (window.__viewer.renderer.backend.isWebGPUBackend ? 'WebGPU' : 'WebGL2'));
  if ((be === 'WebGPU') === (BACKEND === 'webgpu')) break;
}
load = await page.evaluate(async () => { await window.__viewer.ready; return { readyMs: Math.round(performance.now()), ...window.__viewer.info() }; });
load.wallMs = Date.now() - tNav;
if ((load.backend === 'WebGPU') !== (BACKEND === 'webgpu')) {
  console.log(`✗ wanted ${BACKEND}, the page got ${load.backend}: results would not compare. Aborting.`);
  await browser.close(); process.exit(3);
}
// let the simulation populate (first plans, parked cars, riders) before measuring
await page.evaluate(() => new Promise((r) => setTimeout(r, 6000)));
const cdp = await page.context().newCDPSession(page);

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
  if (s.jump) { v.controls.enableDamping = false; v.setView(s.jump); }   // teleport on the first measured frame
  const at = (t) => {
    if (s.path === 'orbit') {
      const dx = P[0] - T[0], dz = P[2] - T[2], a = t * Math.PI * 2;
      return { pos: [T[0] + dx * Math.cos(a) - dz * Math.sin(a), P[1], T[2] + dx * Math.sin(a) + dz * Math.cos(a)], target: T };
    }
    if (s.path === 'pan') {
      const [mx, mz] = s.move ?? [400, 0];   // metres over the run
      return { pos: [P[0] + mx * t, P[1], P[2] + mz * t], target: [T[0] + mx * t, T[1], T[2] + mz * t] };
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
// --laptop: CPU numbers come from a pass with the CPU throttled, GPU numbers from an unthrottled pass
// (GPU timer queries include the gaps while a slowed-down CPU is still sending commands)
const passes = CPU_THROTTLE > 1 ? [{ throttle: 1, use: 'gpu' }, { throttle: CPU_THROTTLE, use: 'cpu' }] : [{ throttle: 1, use: 'both' }];
const gpuPass = new Map();
for (const pass of passes) {
await cdp.send('Emulation.setCPUThrottlingRate', { rate: pass.throttle });
// aggregate a CPU profile: self time per function, top n
function topFunctions(profile, n = 25) {
  const self = new Map(), byId = new Map(profile.nodes.map((x) => [x.id, x]));
  const dt = profile.timeDeltas, total = dt.reduce((a, b) => a + b, 0) / 1000;
  profile.samples.forEach((id, i) => {
    const f = byId.get(id).callFrame;
    const key = `${f.functionName || '(anon)'} ${f.url.split('/').pop().split('?')[0]}:${f.lineNumber + 1}`;
    self.set(key, (self.get(key) ?? 0) + (dt[i] ?? 0) / 1000);
  });
  return [...self.entries()].sort((a, b) => b[1] - a[1]).slice(0, n)
    .map(([k, ms]) => ({ fn: k, ms: Math.round(ms), share: +(100 * ms / total).toFixed(1) }));
}
for (const s of scenarios) {
  // settle: move the camera, let LOD tiles for this view finish loading (paths load while moving)
  await page.evaluate(async (view) => {
    const v = window.__viewer;
    v.setView(view);
    for (let i = 0; i < 3; i++) await new Promise((r) => requestAnimationFrame(r));
    await v.whenIdle?.();
    await new Promise((r) => setTimeout(r, 1500));   // agents near the new focus
    v.prof.reset();
  }, s.view);
  if (PROFILE === s.name) { await cdp.send('Profiler.enable'); await cdp.send('Profiler.setSamplingInterval', { interval: 200 }); await cdp.send('Profiler.start'); }
  const stopClocks = clockSampler();
  const { rec, gpu } = await measure(s);
  const clocks = stopClocks();
  let top = null;
  if (PROFILE === s.name) {
    const { profile } = await cdp.send('Profiler.stop');
    writeFileSync(resolve(OUT, `${s.name}.cpuprofile`), JSON.stringify(profile));
    top = topFunctions(profile);
  }
  const extra = await page.evaluate(() => {
    const v = window.__viewer;
    return { lod: v.lod?.stats ?? null, sections: v.prof.report(), gpuPasses: v.prof.gpuReport(), spikes: v.prof.spikes.slice(), longTasks: v.prof.longTasks.slice(),
      heapMB: performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null,
      sim: v.agents ? { ...v.agents.stats(), vehicles: v.traffic?.count ?? null, walking: v.peds?.walking ?? null, jumpLog: v.agents.jumps, clock: Math.round(v.agentTime()) } : null };
  });
  if (!s.path && !s.jump) await page.screenshot({ path: resolve(OUT, `${s.name}.png`) });
  const intervals = rec.map((f) => f.interval);
  results.push({
    name: s.name, path: s.path ?? (s.jump ? 'jump' : null),
    interval: stats(intervals),
    hitches: { over33: intervals.filter((x) => x > 33).length, over50: intervals.filter((x) => x > 50).length,
      over100: intervals.filter((x) => x > 100).length, longTasks: extra.longTasks.length,
      longestTask: Math.max(0, ...extra.longTasks.map((t) => t.ms)) },
    cpu: stats(rec.map((f) => f.cpu)),
    gpu: stats(gpu),
    drawCalls: Math.round(stats(rec.map((f) => f.drawCalls)).avg),
    trianglesM: r2(stats(rec.map((f) => f.triangles)).avg / 1e6),
    clocks, lod: extra.lod, sections: extra.sections, gpuPasses: extra.gpuPasses, spikes: extra.spikes, heapMB: extra.heapMB, sim: extra.sim, top,
  });
  process.stdout.write('.');
}
if (pass.use === 'gpu') {   // keep the GPU side, measure the CPU side next
  for (const r of results) gpuPass.set(r.name, { gpu: r.gpu, gpuPasses: r.gpuPasses, clocks: r.clocks, drawCalls: r.drawCalls, trianglesM: r.trianglesM });
  results.length = 0;
}
}
for (const r of results) if (gpuPass.has(r.name)) Object.assign(r, gpuPass.get(r.name));
process.stdout.write('\n');
await browser.close();

const idleAfter = gpuLoad();
if (results.some((r) => r.drawCalls === 0)) console.log('\n⚠ some scenarios drew nothing (device lost?): their numbers are meaningless');
if (results.some((r) => r.sim?.jumpLog?.length > 1)) console.log('\n⚠ the agents\' clock jumped during the run (re-planning): see sim.jumpLog');
const run = { laptop: LAPTOP, cpuThrottle: CPU_THROTTLE, gpuFactor: GPU_FACTOR, gpuBusyBefore: idleBefore, gpuBusyAfter: idleAfter, stamp, url: `${URL_BASE}?${query}`, size: [W, H], env, load, results, errors: [...new Set(errors)] };
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
console.log('\nCPU per loop section, ms avg / p95 (top 7):');
for (const r of results) console.log(r.name.padEnd(12), Object.entries(r.sections ?? {}).sort((a, b) => b[1].avg - a[1].avg).slice(0, 7)
  .map(([k, x]) => `${k} ${x.avg.toFixed(2)}/${x.p95.toFixed(1)}`).join('  '));
console.log('\nGPU per render pass, ms avg (scene>target):');
for (const r of results) console.log(r.name.padEnd(12), Object.entries(r.gpuPasses ?? {}).sort((a, b) => b[1].avg - a[1].avg).slice(0, 8)
  .map(([k, x]) => `${k} ${x.avg.toFixed(2)}`).join('  '));
console.log('\nhitches: frames > 33 / 50 / 100 ms, long tasks (longest ms), heap MB');
for (const r of results) console.log(r.name.padEnd(12), `${r.hitches.over33} / ${r.hitches.over50} / ${r.hitches.over100}`.padEnd(14),
  `${r.hitches.longTasks} (${r.hitches.longestTask})`.padEnd(12), r.heapMB ?? '');
console.log('\nspike frames (> 25 ms of loop work): count, and which sections caused them (ms summed over spikes)');
for (const r of results) {
  const sum = {};
  for (const sp of r.spikes ?? []) for (const [k, v] of Object.entries(sp.parts)) sum[k] = (sum[k] ?? 0) + v;
  console.log(r.name.padEnd(12), String(r.spikes?.length ?? 0).padEnd(4), Object.entries(sum).sort((a, b) => b[1] - a[1]).slice(0, 5)
    .map(([k, v]) => `${k} ${Math.round(v)}`).join('  '));
}
if (GPU_FACTOR > 1 || CPU_THROTTLE > 1) {
  // verdict for an average laptop: CPU is measured throttled; GPU is the 3090 time scaled by GPU_FACTOR.
  // The frame costs roughly max(cpu, gpu) when the two overlap; 16.7 ms = 60 fps, 33 ms = 30 fps.
  console.log(`\naverage-laptop estimate (CPU throttled ${CPU_THROTTLE}x; GPU time x${GPU_FACTOR} at full clocks, scaled by the clocks seen):`);
  for (const r of results) {
    // GPU: the average (headless Chromium composites with a GPU readback each frame; its stalls land in
    // random passes' timer queries, so GPU p95 is noise here). CPU: p95.
    const k = gpuFactor(r.clocks), g = (r.gpu?.avg ?? 0) * k, c = r.cpu.p95, f = Math.max(g, c);
    console.log(r.name.padEnd(12), `cpu p95 ${c.toFixed(1)}  gpu avg ~${g.toFixed(1)} (x${k.toFixed(1)}, ${r.clocks?.gr ?? '?'}/${r.clocks?.mem ?? '?'} MHz)  -> ~${(1000 / Math.max(f, 1e-3)).toFixed(0)} fps`,
      f <= 16.7 ? 'smooth' : f <= 33 ? 'playable' : 'TOO SLOW');
  }
}
for (const r of results) if (r.top) {
  console.log(`\nCPU profile '${r.name}' (self time, top functions):`);
  for (const t of r.top) console.log(`  ${String(t.ms).padStart(6)} ms ${t.share.toFixed(1).padStart(5)}%  ${t.fn}`);
}
if (base) console.log(`\n(compared with baseline ${base.stamp}; % = change, (=) within 3%)`);
if (run.errors.length) console.log('\nerrors:\n  ' + run.errors.join('\n  '));
console.log(`\nresults + screenshots: ${OUT}`);
if (flag('save-baseline')) { writeFileSync(basePath, JSON.stringify(run, null, 1)); console.log('saved as baseline'); }
