// Noise map worker: accumulates the sound energy the simulated vehicles emit (per 4 m cell of the ground
// plan) and turns it into an equivalent continuous level (Leq, dB(A)) at every cell.
//
//  sources     energy (J) emitted in each cell since the map started (main thread sends batches)
//  observed    seconds each area was inside the simulated zone around the focus (32 m cells): vehicles
//              only exist near where you look, so a level is energy received / time observed there
//  spreading   point sources in the open, received intensity E / (2π r²) (hemisphere over the ground):
//              the 1/r² kernel as a sum of Gaussians one octave apart (a pyramid: sum 2×2 down, blur,
//              add back up), scaled once against the exact kernel. No screening by buildings yet.
//
// in:  { type: 'init', w, h, cell, db: [min, max] }   grid size in cells, cell size (m), range of the bytes
//      { type: 'add', src: Float32Array [east, north, joules]..., focus: [east, north], radius, dt }
//      { type: 'compute' }  ->  { type: 'map', data: Uint8Array (w × h; 0: not observed, else dB) }
//      { type: 'reset' }

let DB_MIN = 40, DB_MAX = 85;   // byte 1..255 covers this range (set by noisemap.ts)
const OBS = 32;          // m: cells of the observation-time grid
const MIN_OBS = 20;      // s observed before a level is shown

let W = 0, H = 0, CELL = 4;
let src = new Float64Array(0);
let obs = new Float32Array(0), OW = 0, OH = 0;
let calib = 1;           // pyramid output × calib = intensity per joule (exact at the calibration distances)

/** received energy density (J/m² before calibration) at every cell from the energy emitted per cell */
function spread(s: Float64Array, w: number, h: number, cell: number): Float64Array {
  // levels: sum 2×2 down until cells are ~256 m
  const lv: { a: Float64Array; w: number; h: number; c: number }[] = [{ a: s, w, h, c: cell }];
  while (lv[lv.length - 1].c < 256 && lv[lv.length - 1].w > 2 && lv[lv.length - 1].h > 2) {
    const p = lv[lv.length - 1], nw = Math.ceil(p.w / 2), nh = Math.ceil(p.h / 2), a = new Float64Array(nw * nh);
    for (let j = 0; j < p.h; j++) for (let i = 0; i < p.w; i++) a[(j >> 1) * nw + (i >> 1)] += p.a[j * p.w + i];
    lv.push({ a, w: nw, h: nh, c: p.c * 2 });
  }
  // each level: binomial blur (σ ≈ 1 cell), as density per m²; then from the coarsest, upsample and add
  let acc: Float64Array | null = null, aw = 0, ah = 0;
  for (let k = lv.length - 1; k >= 0; k--) {
    const { a, w: lw, h: lh, c } = lv[k];
    const d = blur(a, lw, lh, 1 / (c * c));
    if (acc) {
      // bilinear: fine cell centre (i + 0.5) -> coarse coordinate (i + 0.5) / 2 - 0.5
      for (let j = 0; j < lh; j++) {
        const y = Math.min(Math.max((j + 0.5) / 2 - 0.5, 0), ah - 1), y0 = Math.floor(y), y1 = Math.min(y0 + 1, ah - 1), fy = y - y0;
        for (let i = 0; i < lw; i++) {
          const x = Math.min(Math.max((i + 0.5) / 2 - 0.5, 0), aw - 1), x0 = Math.floor(x), x1 = Math.min(x0 + 1, aw - 1), fx = x - x0;
          d[j * lw + i] += (acc[y0 * aw + x0] * (1 - fx) + acc[y0 * aw + x1] * fx) * (1 - fy)
            + (acc[y1 * aw + x0] * (1 - fx) + acc[y1 * aw + x1] * fx) * fy;
        }
      }
    }
    acc = d; aw = lw; ah = lh;
  }
  return acc!;
}

const K = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16];
function blur(a: Float64Array, w: number, h: number, scale: number) {
  const t = new Float64Array(w * h), o = new Float64Array(w * h);
  for (let j = 0; j < h; j++) {
    const r = j * w;
    for (let i = 0; i < w; i++) {
      let v = 0;
      for (let k = -2; k <= 2; k++) { const x = i + k; if (x >= 0 && x < w) v += a[r + x] * K[k + 2]; }
      t[r + i] = v;
    }
  }
  for (let j = 0; j < h; j++)
    for (let i = 0; i < w; i++) {
      let v = 0;
      for (let k = -2; k <= 2; k++) { const y = j + k; if (y >= 0 && y < h) v += t[y * w + i] * K[k + 2]; }
      o[j * w + i] = v * scale;
    }
  return o;
}

/** calibrate: one joule in the middle of a test grid, compared with 1 / (2π r²) at 16..256 m (log mean) */
function calibrate() {
  const n = 512, s = new Float64Array(n * n), c = n / 2;
  s[c * n + c] = 1;
  const r = spread(s, n, n, CELL);
  let sum = 0, cnt = 0;
  for (const m of [16, 32, 64, 128, 256]) {
    const d = m / CELL, exact = 1 / (2 * Math.PI * m * m);
    for (const [dx, dy] of [[d, 0], [0, d], [d * 0.7071, d * 0.7071]]) {
      const got = r[Math.round(c + dy) * n + Math.round(c + dx)];
      sum += Math.log(exact / got); cnt++;
    }
  }
  return Math.exp(sum / cnt);
}

self.onmessage = (e: MessageEvent) => {
  const m = e.data;
  if (m.type === 'init') {
    W = m.w; H = m.h; CELL = m.cell; [DB_MIN, DB_MAX] = m.db;
    src = new Float64Array(W * H);
    OW = Math.ceil((W * CELL) / OBS); OH = Math.ceil((H * CELL) / OBS);
    obs = new Float32Array(OW * OH);
    calib = calibrate();
  } else if (m.type === 'reset') {
    src.fill(0); obs.fill(0);
  } else if (m.type === 'add') {
    const a: Float32Array = m.src;
    for (let k = 0; k + 2 < a.length; k += 3) {
      const i = Math.floor(a[k] / CELL), j = Math.floor(a[k + 1] / CELL);
      if (i >= 0 && j >= 0 && i < W && j < H) src[j * W + i] += a[k + 2];
    }
    // observed: every 32 m cell whose centre is within the radius of the focus
    const [fx, fy] = m.focus as [number, number], R = m.radius as number;
    const i0 = Math.max(0, Math.floor((fx - R) / OBS)), i1 = Math.min(OW - 1, Math.floor((fx + R) / OBS));
    const j0 = Math.max(0, Math.floor((fy - R) / OBS)), j1 = Math.min(OH - 1, Math.floor((fy + R) / OBS));
    for (let j = j0; j <= j1; j++)
      for (let i = i0; i <= i1; i++) {
        const dx = (i + 0.5) * OBS - fx, dy = (j + 0.5) * OBS - fy;
        if (dx * dx + dy * dy < R * R) obs[j * OW + i] += m.dt;
      }
  } else if (m.type === 'compute') {
    const r = spread(src, W, H, CELL), out = new Uint8Array(W * H);
    const k = calib / 1e-12;   // intensity re 1 pW/m²
    for (let j = 0; j < H; j++) {
      const oj = Math.min(OH - 1, Math.floor((j * CELL) / OBS)) * OW;
      for (let i = 0; i < W; i++) {
        const t = obs[oj + Math.min(OW - 1, Math.floor((i * CELL) / OBS))];
        if (t < MIN_OBS) continue;
        const db = 10 * Math.log10(Math.max((r[j * W + i] * k) / t, 1e-3));
        out[j * W + i] = Math.max(1, Math.min(255, Math.round(1 + ((db - DB_MIN) / (DB_MAX - DB_MIN)) * 254)));
      }
    }
    (self as unknown as Worker).postMessage({ type: 'map', data: out }, [out.buffer]);
  }
};
