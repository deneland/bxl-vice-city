// Junction geometry check: every road junction as the simulation builds it (lanes trimmed at the junction,
// the curves joining them), with automatic flags for implausible geometry. Run after the pipeline changes
// roads (roads.py), then look at the contact sheet:
//
//   npm run junctions                 metrics + data/metrics/junctions/<stamp>/junctions.json
//   (cd ../pipeline && uv run python qa_junctions.py)   contact sheet of the worst and a random sample
//
// Flags per junction:
//   loop     a curve much longer than the gap it bridges (len > 2 x chord): a detour or knot
//   far      a curve point more than 35 m from the junction's centre: cars swing wide through it
//   spread   lane ends more than 30 m from the centre: a junction that is really several
//   kink     a sharp bend (> 60 degrees within 2 m) in the last / first 30 m of a lane: a spike
//   dead     an incoming lane with no curve out although the junction has exits
import { chromium } from 'playwright-core';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '../..');
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const OUT = resolve(ROOT, 'data/metrics/junctions', stamp);
mkdirSync(OUT, { recursive: true });

const b = await chromium.launch({ executablePath: process.env.CHROME ?? '/usr/bin/chromium', headless: true,
  args: ['--ignore-gpu-blocklist', '--enable-features=Vulkan', '--use-angle=vulkan', '--disable-vulkan-surface'] });
const p = await b.newPage({ viewport: { width: 800, height: 500 } });
await p.goto('http://127.0.0.1:5173/?telemetry=0&time=3&date=2026-11-04&weather=clear');
await p.waitForFunction(() => window.__viewer?.traffic, null, { timeout: 90000 });
const js = await p.evaluate(() => window.__viewer.traffic.junctionGeometry());
await b.close();

const d = (a, c) => Math.hypot(a[0] - c[0], a[1] - c[1]);
const angle = (a, b, c) => {
  const u = [b[0] - a[0], b[1] - a[1]], v = [c[0] - b[0], c[1] - b[1]];
  const l = Math.hypot(...u) * Math.hypot(...v);
  return l ? Math.acos(Math.max(-1, Math.min(1, (u[0] * v[0] + u[1] * v[1]) / l))) * 180 / Math.PI : 0;
};
const kinks = (line) => { let k = 0; for (let i = 1; i + 1 < line.length; i++) if (angle(line[i - 1], line[i], line[i + 1]) > 60) k++; return k; };
const counts = { loop: 0, far: 0, spread: 0, kink: 0, dead: 0 };
for (const j of js) {
  const ends = [...j.ins.map((x) => x.end), ...j.outs.map((x) => x.start)];
  j.centre = [ends.reduce((a, x) => a + x[0], 0) / ends.length, ends.reduce((a, x) => a + x[1], 0) / ends.length];
  j.flags = {
    loop: j.curves.filter((c) => c.chord > 3 && c.len > 2 * c.chord).length,
    far: j.curves.filter((c) => c.pts.some((q) => d(q, j.centre) > 35)).length,
    spread: +Math.max(0, ...ends.map((e) => d(e, j.centre))).toFixed(1) > 30 ? 1 : 0,
    kink: j.ins.reduce((a, x) => a + kinks(x.line), 0) + j.outs.reduce((a, x) => a + kinks(x.line), 0),
    // (dead ends, a single street: turning back there is right)
    dead: j.outs.length && new Set([...j.ins.map((x) => x.edge), ...j.outs.map((x) => x.edge)]).size > 1 ? j.ins.filter((a) => !j.curves.some((c) => c.from === a.edge)).length : 0,
  };
  j.score = 3 * j.flags.loop + 2 * j.flags.far + 3 * j.flags.spread + 2 * j.flags.kink + 2 * j.flags.dead;
  j.degree = new Set([...j.ins.map((x) => x.edge), ...j.outs.map((x) => x.edge)]).size;
  for (const k of Object.keys(counts)) if (j.flags[k]) counts[k]++;
}
writeFileSync(resolve(OUT, 'junctions.json'), JSON.stringify(js));
const n = js.length, bad = js.filter((j) => j.score > 0).length;
console.log(`${n} road junctions; ${bad} flagged (${(100 * bad / n).toFixed(1)} %)`);
for (const [k, v] of Object.entries(counts)) console.log(`  ${k.padEnd(7)} ${v}`);
console.log(`worst: ${js.sort((a, b) => b.score - a.score).slice(0, 8).map((j) => `node ${j.node} (${j.score})`).join(', ')}`);
console.log(`-> ${OUT}/junctions.json`);
