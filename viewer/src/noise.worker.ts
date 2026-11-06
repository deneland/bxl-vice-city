// Noise map worker: the equivalent continuous level (Leq, dB(A)) of the traffic at every 4 m cell of the
// ground plan, over the whole region.
//
//  sources     energy (J) the simulated vehicles emitted in each cell, per kind (road / tram / rail)
//  observed    seconds each area was inside the simulated zone around the focus (32 m cells): vehicles
//              only exist near where you look
//  estimate    elsewhere, streets of the same kind emit like the observed ones: per kind and road class
//              (lanes.json rank), the mean power per cell of the observed cells. A cell's power is its
//              measured energy plus that estimate for a prior time T0, divided by (observed time + T0): an
//              unobserved street gets its class mean, a long-watched one what was measured there (the way
//              official noise maps are made: typical flows per street type, measured where available)
//  spreading   point sources in the open, received intensity P / (2π r²) (hemisphere over the ground):
//              the 1/r² kernel as a sum of Gaussians one octave apart (a pyramid: sum 2×2 down, blur,
//              add back up), scaled once against the exact kernel. No screening by buildings yet.
//
// in:  { type: 'init', w, h, cell, db: [min, max] }   grid size in cells, cell size (m), range of the bytes
//      { type: 'lanes', pts: Float32Array [east, north, kind, rank]... }   points every ~2 m along the lanes
//      { type: 'add', src: Float32Array [east, north, joules, kind]..., focus: [east, north], radius, dt }
//      { type: 'compute', prior: Float64Array | null }  ->  { type: 'map', data: Uint8Array (w × h; 0: no level, else dB) }
//            prior: W per cell by kind × class from a calibration run (bench/noisecal.mjs), NaN where unknown
//      { type: 'rates' }  ->  { type: 'rates', e, t }   energy and observed time per kind × class (calibration)
//      { type: 'reset' }

let DB_MIN = 40, DB_MAX = 85;   // byte 1..255 covers this range (set by noisemap.ts)
const OBS = 32;          // m: cells of the observation-time grid
const T0 = 60;           // s: prior weight of the street-type estimate in observed cells
const KINDS = 3;         // road, tram, rail (roads.ts KIND)
const RANKS = 16;        // road classes per kind (lane rank, clamped)

let W = 0, H = 0, CELL = 4;
let src: Float64Array[] = [];                 // per kind: J per cell
let cls = new Int8Array(0);                   // per kind × cell: lane class (rank) there, -1: no lane of that kind
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

const obsAt = (i: number, j: number) =>
  obs[Math.min(OH - 1, Math.floor((j * CELL) / OBS)) * OW + Math.min(OW - 1, Math.floor((i * CELL) / OBS))];

/** energy (J) and observed cell-seconds by kind and class */
function classSums() {
  const e = new Float64Array(KINDS * RANKS), t = new Float64Array(KINDS * RANKS);
  for (let k = 0; k < KINDS; k++) {
    const s = src[k], c = k * W * H;
    for (let j = 0; j < H; j++)
      for (let i = 0; i < W; i++) {
        const r = cls[c + j * W + i];
        if (r < 0) continue;
        const o = obsAt(i, j);
        if (o <= 0) continue;
        e[k * RANKS + r] += s[j * W + i]; t[k * RANKS + r] += o;
      }
  }
  return { e, t };
}

/** mean power per cell (W) by kind and class: the calibrated prior where there is one, else from the cells
 *  observed so far (NaN: unknown) */
function classRates(prior: Float64Array | null) {
  const { e, t } = classSums();
  const rate = new Float64Array(KINDS * RANKS).fill(NaN);
  for (let x = 0; x < rate.length; x++) rate[x] = prior && !Number.isNaN(prior[x]) ? prior[x] : t[x] > 0 ? e[x] / t[x] : NaN;
  // a class never observed: the nearest observed class of the same kind
  for (let k = 0; k < KINDS; k++)
    for (let r = 0; r < RANKS; r++) {
      if (!Number.isNaN(rate[k * RANKS + r])) continue;
      for (let d = 1; d < RANKS; d++) {
        const a = rate[k * RANKS + r - d], b = rate[k * RANKS + r + d];
        if (r - d >= 0 && !Number.isNaN(a)) { rate[k * RANKS + r] = a; break; }
        if (r + d < RANKS && !Number.isNaN(b)) { rate[k * RANKS + r] = b; break; }
      }
    }
  return rate;
}

self.onmessage = (e: MessageEvent) => {
  const m = e.data;
  if (m.type === 'init') {
    W = m.w; H = m.h; CELL = m.cell; [DB_MIN, DB_MAX] = m.db;
    src = Array.from({ length: KINDS }, () => new Float64Array(W * H));
    cls = new Int8Array(KINDS * W * H).fill(-1);
    OW = Math.ceil((W * CELL) / OBS); OH = Math.ceil((H * CELL) / OBS);
    obs = new Float32Array(OW * OH);
    calib = calibrate();
  } else if (m.type === 'lanes') {
    const a: Float32Array = m.pts;
    for (let k = 0; k + 3 < a.length; k += 4) {
      const i = Math.floor(a[k] / CELL), j = Math.floor(a[k + 1] / CELL), kind = a[k + 2];
      if (i < 0 || j < 0 || i >= W || j >= H || kind < 0 || kind >= KINDS) continue;
      const x = kind * W * H + j * W + i, r = Math.max(0, Math.min(RANKS - 1, a[k + 3]));
      if (r > cls[x]) cls[x] = r;   // (several lanes in a cell: the highest class)
    }
  } else if (m.type === 'rates') {
    const { e, t } = classSums();
    (self as unknown as Worker).postMessage({ type: 'rates', e, t, kinds: KINDS, ranks: RANKS, cell: CELL });
  } else if (m.type === 'reset') {
    for (const s of src) s.fill(0);
    obs.fill(0);
  } else if (m.type === 'add') {
    const a: Float32Array = m.src;
    for (let k = 0; k + 3 < a.length; k += 4) {
      const i = Math.floor(a[k] / CELL), j = Math.floor(a[k + 1] / CELL), kind = a[k + 3];
      if (i >= 0 && j >= 0 && i < W && j < H && kind >= 0 && kind < KINDS) src[kind][j * W + i] += a[k + 2];
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
    // power per cell: measured where observed, the street-type estimate elsewhere (blended by T0)
    const rate = classRates(m.prior ?? null), P = new Float64Array(W * H);
    for (let j = 0; j < H; j++)
      for (let i = 0; i < W; i++) {
        const o = obsAt(i, j), x = j * W + i;
        let p = 0;
        for (let k = 0; k < KINDS; k++) {
          const r = cls[k * W * H + x], est = r >= 0 ? rate[k * RANKS + r] : NaN;
          p += Number.isNaN(est) ? (o > 0 ? src[k][x] / (o + T0) : 0) : (src[k][x] + est * T0) / (o + T0);
        }
        P[x] = p;
      }
    const r = spread(P, W, H, CELL), out = new Uint8Array(W * H);
    const k = calib / 1e-12;   // intensity re 1 pW/m²
    let any = false;
    for (let x = 0; x < W * H; x++) {
      const v = r[x] * k;
      if (v <= 0) continue;
      any = true;
      const db = 10 * Math.log10(v);
      out[x] = Math.max(1, Math.min(255, Math.round(1 + ((db - DB_MIN) / (DB_MAX - DB_MIN)) * 254)));
    }
    (self as unknown as Worker).postMessage({ type: 'map', data: out, any }, [out.buffer]);
  }
};
