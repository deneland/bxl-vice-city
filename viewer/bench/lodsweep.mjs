// LOD threshold sweep: cost (triangles drawn) vs visual impact (difference from
// the full-detail render) for each fixed view in scenarios.json.
//
//   npm run lodsweep                         default thresholds 2,3,4,6,8,12 px
//   npm run lodsweep -- --errs 3,6,10 --only overview,whole-roi
//
// Visual impact per view, against ?lod=0 (always finest level):
//   changed  share of pixels whose colour differs by more than 8% after a 3x3 blur
//            (blur ignores single-pixel aliasing shimmer)
//   at ¼     the same at quarter resolution (ignores 1-2 px edge shifts; closer to what you notice)
//   psnr     peak signal-to-noise ratio in dB (higher = closer; >35 dB is hard to see)
// Images: data/metrics/lodsweep/<stamp>/<view>_err<N>.png plus *_diff.png heatmaps.
import { chromium } from 'playwright-core';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '../..');
const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(`--${n}`); return i >= 0 ? args[i + 1] : d; };
const URL_BASE = opt('url', 'http://127.0.0.1:5173/');
const ERRS = opt('errs', '2,3,4,6,8,12').split(',').map(Number);
const ONLY = opt('only', '').split(',').filter(Boolean);
const [W, H] = opt('size', '1600x1000').split('x').map(Number);
const cfg = JSON.parse(readFileSync(resolve(HERE, 'scenarios.json'), 'utf8'));
const views = cfg.scenarios.filter((s) => !s.path && (!ONLY.length || ONLY.includes(s.name)));
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const OUT = resolve(ROOT, 'data/metrics/lodsweep', stamp);
mkdirSync(OUT, { recursive: true });

// a fresh browser per capture: later pages in one headless browser came out blank
const launch = () => chromium.launch({
  executablePath: opt('chrome', '/usr/bin/chromium'), headless: true,
  args: ['--ignore-gpu-blocklist', '--enable-features=Vulkan', '--use-angle=vulkan',
    '--disable-gpu-vsync', '--disable-frame-rate-limit', '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'],
});

async function capture(query) {
  const browser = await launch();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  await page.bringToFront();
  await page.goto(`${URL_BASE}?telemetry=0&gpu=0&continuous=1&${query}`);
  await page.addStyleTag({ content: '#hud,#help,#attr{display:none!important}' });
  await page.waitForFunction(() => window.__viewer, null, { timeout: 30000 });
  await page.evaluate(() => window.__viewer.ready);
  const out = {};
  for (const v of views) {
    const info = await page.evaluate(async (view) => {
      const V = window.__viewer;
      V.controls.enableDamping = false;
      V.setView(view);
      for (let i = 0; i < 3; i++) await new Promise((r) => requestAnimationFrame(r));
      await V.whenIdle();
      for (let i = 0; i < 3; i++) await new Promise((r) => requestAnimationFrame(r));
      return { tris: V.renderer.info.render.triangles, draws: V.renderer.info.render.drawCalls, ...V.lod.stats };
    }, v.view);
    await page.bringToFront();
    out[v.name] = { ...info, png: await page.screenshot() };
  }
  await browser.close();
  return out;
}

// image comparison runs inside a browser page (canvas), no image libs needed
const cmpBrowser = await launch();
const cmpPage = await cmpBrowser.newPage();
async function compare(a, b) {
  return cmpPage.evaluate(async ([a, b]) => {
    const load = (b64) => new Promise((res) => { const i = new Image(); i.onload = () => res(i); i.src = 'data:image/png;base64,' + b64; });
    const [ia, ib] = await Promise.all([load(a), load(b)]);
    const w = ia.width, h = ia.height;
    const px = (img, blur) => {
      const c = new OffscreenCanvas(w, h), x = c.getContext('2d');
      if (blur) x.filter = 'blur(1px)';
      x.drawImage(img, 0, 0);
      return x.getImageData(0, 0, w, h).data;
    };
    const A = px(ia, false), B = px(ib, false), Ab = px(ia, true), Bb = px(ib, true);
    let se = 0, changed = 0;
    const heat = new ImageData(w, h);
    for (let i = 0; i < A.length; i += 4) {
      for (let k = 0; k < 3; k++) { const d = A[i + k] - B[i + k]; se += d * d; }
      const db = Math.max(Math.abs(Ab[i] - Bb[i]), Math.abs(Ab[i + 1] - Bb[i + 1]), Math.abs(Ab[i + 2] - Bb[i + 2]));
      const on = db > 20;
      if (on) changed++;
      const g = (A[i] + A[i + 1] + A[i + 2]) / 9;
      heat.data[i] = on ? 255 : g; heat.data[i + 1] = on ? 40 : g; heat.data[i + 2] = on ? 40 : g; heat.data[i + 3] = 255;
    }
    const mse = se / (w * h * 3);
    // same comparison at quarter resolution: roughly what you notice without pixel peeping
    const q = (img) => { const c = new OffscreenCanvas(w >> 2, h >> 2), x = c.getContext('2d');
      x.imageSmoothingQuality = 'high'; x.drawImage(img, 0, 0, w >> 2, h >> 2); return x.getImageData(0, 0, w >> 2, h >> 2).data; };
    const Aq = q(ia), Bq = q(ib);
    let changedQ = 0;
    for (let i = 0; i < Aq.length; i += 4)
      if (Math.max(Math.abs(Aq[i] - Bq[i]), Math.abs(Aq[i + 1] - Bq[i + 1]), Math.abs(Aq[i + 2] - Bq[i + 2])) > 20) changedQ++;
    const c = new OffscreenCanvas(w, h); c.getContext('2d').putImageData(heat, 0, 0);
    const blob = await c.convertToBlob({ type: 'image/png' });
    const buf = new Uint8Array(await blob.arrayBuffer());
    let s = ''; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode(...buf.subarray(i, i + 0x8000));
    return { changedQ: changedQ / ((w >> 2) * (h >> 2)), changed: changed / (w * h), psnr: mse ? 10 * Math.log10(255 * 255 / mse) : 99, heat: btoa(s) };
  }, [a.toString('base64'), b.toString('base64')]);
}

const ref = await capture('lod=0&budget=200');
for (const [n, r] of Object.entries(ref)) writeFileSync(resolve(OUT, `${n}_ref.png`), r.png);
const rows = [];
for (const err of ERRS) {
  const run = await capture(`err=${err}`);
  for (const v of views) {
    const r = run[v.name], f = ref[v.name];
    const c = await compare(f.png, r.png);
    writeFileSync(resolve(OUT, `${v.name}_err${err}.png`), r.png);
    writeFileSync(resolve(OUT, `${v.name}_err${err}_diff.png`), Buffer.from(c.heat, 'base64'));
    rows.push({ err, view: v.name, tris: r.tris, refTris: f.tris, faces: r.faces, perLevel: r.perLevel,
      changed: c.changed, changedQ: c.changedQ, psnr: c.psnr });
  }
  process.stdout.write('.');
}
process.stdout.write('\n');
await cmpBrowser.close();
writeFileSync(resolve(OUT, 'sweep.json'), JSON.stringify(rows, null, 1));

const pad = (s, n) => String(s).padEnd(n);
console.log(`\n${pad('err px', 8)}${pad('view', 12)}${pad('tris', 10)}${pad('vs full', 9)}${pad('changed', 9)}${pad('at ¼', 8)}${pad('psnr dB', 9)}tiles L0/L1/L2/L3…`);
for (const r of rows) {
  console.log(pad(r.err, 8) + pad(r.view, 12) + pad((r.tris / 1e6).toFixed(2) + 'M', 10) +
    pad(Math.round(100 * r.tris / r.refTris) + '%', 9) + pad((100 * r.changed).toFixed(2) + '%', 9) + pad((100 * r.changedQ).toFixed(2) + '%', 8) +
    pad(r.psnr.toFixed(1), 9) + r.perLevel.join('/'));
}
console.log('\nsummary (all views):');
for (const err of ERRS) {
  const rs = rows.filter((r) => r.err === err);
  const cost = rs.reduce((a, r) => a + r.tris, 0) / rs.reduce((a, r) => a + r.refTris, 0);
  const worst = rs.reduce((a, r) => (r.changedQ > a.changedQ ? r : a));
  console.log(`  err ${pad(err, 4)} triangles ${pad(Math.round(100 * cost) + '%', 5)} of full · worst view ${pad(worst.view, 10)} ${(100 * worst.changedQ).toFixed(2)}% changed at ¼ res (${(100 * worst.changed).toFixed(2)}% full res)`);
}
console.log(`\nimages: ${OUT}`);
