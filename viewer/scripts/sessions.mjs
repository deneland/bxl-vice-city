// Summarise real viewing sessions recorded by the dev-server telemetry sink.
//   npm run sessions              last 5 sessions
//   npm run sessions -- --last 20 / --session <id> / --since 2026-11-03T10:00
import { readFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const FILE = resolve(dirname(fileURLToPath(import.meta.url)), '../../data/metrics/sessions.jsonl');
const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(`--${n}`); return i >= 0 ? args[i + 1] : d; };
if (!existsSync(FILE)) { console.log('no sessions recorded yet'); process.exit(0); }

const msgs = readFileSync(FILE, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
const by = new Map();
for (const m of msgs) {
  if (!by.has(m.session)) by.set(m.session, { id: m.session, env: {}, events: [], errors: [], sums: [], snaps: [] });
  const s = by.get(m.session);
  if (m.type === 'start' || m.type === 'env') Object.assign(s.env, m.env);
  else if (m.type === 'event') s.events.push(m);
  else if (m.type === 'error') s.errors.push(m);
  else if (m.type === 'summary') s.sums.push(m);
  else if (m.type === 'snapshot') s.snaps.push(m);
  s.last = m.received;
}
let sessions = [...by.values()];
if (opt('session')) sessions = sessions.filter((s) => s.id.includes(opt('session')));
if (opt('since')) sessions = sessions.filter((s) => s.last >= opt('since'));
sessions = sessions.slice(-Number(opt('last', 5)));

const browser = (ua = '') => (ua.match(/(HeadlessChrome|Firefox|Edg|Chrome|Safari)\/(\d+)/) ?? []).slice(1).join(' ') || ua;
// combine per-flush percentiles weighted by frame count (approximation: max of p99s, weighted avg of avgs)
const merge = (sums, key) => {
  const xs = sums.map((s) => ({ n: s[key]?.n ?? s.frames, v: s[key] })).filter((x) => x.v);
  if (!xs.length) return null;
  const n = xs.reduce((a, x) => a + x.n, 0);
  return { n, avg: xs.reduce((a, x) => a + x.v.avg * x.n, 0) / n, p95: Math.max(...xs.map((x) => x.v.p95)),
    p99: Math.max(...xs.map((x) => x.v.p99)), max: Math.max(...xs.map((x) => x.v.max)) };
};
const f = (x) => (x == null ? '–' : x.toFixed(1));

for (const s of sessions) {
  const e = s.env;
  const visible = s.sums.filter((x) => x.visible && x.frames > 0);
  const all = merge(visible, 'interval'), mov = merge(visible, 'intervalMoving'), gpu = merge(visible, 'gpu');
  const long = visible.reduce((a, x) => ({ o20: a.o20 + x.longFrames.over20, o33: a.o33 + x.longFrames.over33, o50: a.o50 + x.longFrames.over50 }), { o20: 0, o33: 0, o50: 0 });
  const load = s.events.find((x) => x.name === 'initial-load');
  const lastSum = visible.at(-1);
  console.log(`\n■ ${s.id}  (last seen ${s.last})`);
  console.log(`  ${browser(e.ua)} · ${e.backend ?? '?'} · ${e.gpu ?? '?'} · ${e.screen?.w}x${e.screen?.h}@${e.screen?.dpr} · url ${e.url}`);
  if (load) console.log(`  load: first view in ${load.ms} ms (${load.ready}/${load.total} tiles)`);
  if (all) {
    console.log(`  frames ${all.n} (${mov?.n ?? 0} moving) · interval avg ${f(all.avg)} p95 ${f(all.p95)} p99 ${f(all.p99)} max ${f(all.max)} ms`);
    if (mov) console.log(`  while moving: avg ${f(mov.avg)} p95 ${f(mov.p95)} p99 ${f(mov.p99)} ms`);
    console.log(`  long frames: >20ms ${long.o20} · >33ms ${long.o33} · >50ms ${long.o50}` +
      (gpu ? ` · gpu avg ${f(gpu.avg)} p95 ${f(gpu.p95)} ms` : ' · gpu timing unavailable'));
    if (lastSum) console.log(`  draws ~${lastSum.drawCalls} · tris ~${((lastSum.triangles ?? 0) / 1e6).toFixed(1)}M · heap ${lastSum.memMB ?? '–'} MB · view ${JSON.stringify(lastSum.view)}`);
  }
  const uniq = [...new Map(s.errors.filter((x) => !x.message.includes('WebGPU is not available')).map((x) => [x.message.split('\n')[0], x])).values()];
  for (const x of uniq.slice(0, 5)) console.log(`  ! ${x.kind}: ${x.message.split('\n')[0].slice(0, 160)}`);
  for (const x of s.snaps) console.log(`  📷 ${x.image}  view ${JSON.stringify(x.view)}`);
}
