// Noise map calibration: how much sound each street type emits in the simulation, at a few hours of the day.
// Runs the viewer headless (one hour at a time; the simulation runs in real time, ~13 min per hour), moves the
// focus over a grid of points across the region (after each move, traffic fills in before measuring), then
// reads the energy and observed time per kind × road class from the noise worker and writes
// public/noise/<roi>.json, the street-type levels the noise map starts from (noisemap.ts loadPrior).
//
//   node bench/noisecal.mjs [--roi schaerbeek] [--hours 3,8.5,13,18.5] [--grid 3] [--warm 25] [--measure 50]
//                           [--date 2026-11-04]   (a Wednesday)
// The dev server must be running (npm run dev).
import { chromium } from 'playwright-core';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(`--${name}`); return i >= 0 ? args[i + 1] : def; };
const ROI = opt('roi', 'schaerbeek');
const HOURS = opt('hours', '3,8.5,13,18.5').split(',').map(Number);
const GRID = Number(opt('grid', 3));
const WARM = Number(opt('warm', 25)) * 1000;
const MEASURE = Number(opt('measure', 50)) * 1000;
const DATE = opt('date', '2026-11-04');
const URL0 = opt('url', 'http://127.0.0.1:5173/');
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

const manifest = JSON.parse(readFileSync(`${ROOT}/public/tiles/${ROI}/manifest.json`, 'utf8'));
const L0 = manifest.levels[0], VS = L0.voxelSize;
const size = [L0.size[0] * VS, L0.size[1] * VS];   // m, east × north
// grid points (east, north), inset so the 700 m observed disc stays mostly inside the region
const points = [];
for (let j = 0; j < GRID; j++) for (let i = 0; i < GRID; i++)
  points.push([size[0] * (i + 0.5) / GRID, size[1] * (j + 0.5) / GRID]);

async function runHour(hour) {
  let browser = null;
  try {
    browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true,
      args: ['--ignore-gpu-blocklist', '--enable-features=Vulkan', '--use-angle=vulkan'] });
    const page = await browser.newPage({ viewport: { width: 640, height: 360 } });
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(`${URL0}?roi=${ROI}&backend=webgl&sirens=0&telemetry=0&shadows=0&time=${hour}&date=${DATE}&weather=clear`);
    await page.waitForFunction(() => window.__viewer?.noise, null, { timeout: 120000 });
    await page.waitForTimeout(30000);   // (people and vehicles are still being set up after the first view)
    for (const [k, [e, n]] of points.entries()) {
      // look at the point from above (the focus is the orbit target), drop what happens while traffic fills in
      const cars = await page.evaluate(([e, n]) => {
        const v = window.__viewer, o = v.traffic.group.parent.position;
        v.noise.recording = false;
        v.setView({ pos: [e + o.x - 60, 220, -n + o.z + 80], target: [e + o.x, 0, -n + o.z] });
        return v.traffic.vehicles.length;
      }, [e, n]);
      await page.waitForTimeout(WARM);
      await page.evaluate(() => { window.__viewer.noise.recording = true; });
      await page.waitForTimeout(MEASURE);
      const after = await page.evaluate(() => window.__viewer.traffic.vehicles.length);
      console.log(`  ${hour} h: point ${k + 1}/${points.length} (${e.toFixed(0)}, ${n.toFixed(0)}): vehicles ${cars} -> ${after}`);
    }
    const r = await page.evaluate(async () => {
      const d = await window.__viewer.noise.rates();
      return { e: [...d.e], t: [...d.t], kinds: d.kinds, ranks: d.ranks, cell: d.cell };
    });
    if (errors.length) console.log(`  ${hour} h: page errors`, errors.slice(0, 5));
    return r;
  } finally {
    await browser?.close().catch(() => {});
  }
}

console.log(`noise calibration: ${ROI}, hours ${HOURS.join(', ')}, ${points.length} points × (${WARM / 1000} s warm-up + ${MEASURE / 1000} s)`);
// One hour at a time (several browsers at once crashed the GPU process), each merged into the table and
// saved as soon as it is done: a later failure loses only that hour. Each try has a hard time limit.
const KIND_NAMES = ['road', 'tram', 'rail'];
const FILE = `${ROOT}/public/noise/${ROI}.json`;
const LIMIT = 120000 + 60000 + points.length * (WARM + MEASURE + 15000);
const timed = (p) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error('time limit')), LIMIT))]);
let table = null;
try { table = JSON.parse(readFileSync(FILE, 'utf8')); if (!table.seconds) table = null; } catch { /* new table */ }
for (const h of HOURS) {
  let r = null;
  for (let attempt = 1; attempt <= 3 && !r; attempt++) {
    try { r = await timed(runHour(h)); } catch (e) { console.log(`  ${h} h: attempt ${attempt} failed (${e.message.split('\n')[0]})`); }
  }
  if (!r) { console.log(`  ${h} h: given up`); continue; }
  table ??= { roi: ROI, date: DATE, cell: r.cell, kinds: r.kinds, ranks: r.ranks,
    note: 'W per map cell by kind (road, tram, rail) x road class, measured in the simulation (bench/noisecal.mjs); null: never observed',
    hours: [], rates: [], seconds: [] };
  // replace this hour if the table has it, keep the hours sorted
  const rows = table.hours.map((x, i) => ({ h: x, rate: table.rates[i], sec: table.seconds[i] })).filter((x) => x.h !== h);
  rows.push({ h, rate: r.e.map((e, x) => (r.t[x] > 0 ? +(e / r.t[x]).toPrecision(4) : null)), sec: r.t.map((t) => Math.round(t)) });
  rows.sort((a, b) => a.h - b.h);
  Object.assign(table, { hours: rows.map((x) => x.h), rates: rows.map((x) => x.rate), seconds: rows.map((x) => x.sec) });
  mkdirSync(dirname(FILE), { recursive: true });
  writeFileSync(FILE, JSON.stringify(table));
  // summary: equivalent sound power level per cell, by kind and class
  const line = [];
  for (let k = 0; k < r.kinds; k++)
    for (let c = 0; c < r.ranks; c++) {
      const x = k * r.ranks + c;
      if (r.t[x] > 0 && r.e[x] > 0) line.push(`${KIND_NAMES[k]}${c}:${(10 * Math.log10(r.e[x] / r.t[x] / 1e-12)).toFixed(1)}dB`);
    }
  console.log(`${h} h  ${line.join('  ')}  -> saved`);
}
