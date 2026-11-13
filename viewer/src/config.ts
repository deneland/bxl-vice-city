// Settings from the page URL (tests, benchmarks, tuning): every query parameter the viewer reads, in one place.
const q = new URLSearchParams(location.search);
const num = (k: string, fallback: number) => (q.has(k) ? Number(q.get(k)) : fallback);
const opt = (k: string) => (q.has(k) ? Number(q.get(k)) : null);

export const config = {
  roi: q.get('roi') ?? 'schaerbeek',
  /** keep fetched tiles in Cache Storage for the next visit (?cache=0: off, e.g. to measure downloads) */
  tileCache: q.get('cache') !== '0',
  /** ?tier=low|high: force the device tier (tier.ts; default: detected) */
  tier: q.get('tier'),
  /** where the data files are (tiles, roads, timetable, ...): ?data=<url>, else VITE_DATA_URL set at build
   *  time, else next to the page. Lets the data live on another host than the app (a bigger region). */
  data: (q.get('data') ?? import.meta.env.VITE_DATA_URL ?? '').replace(/([^/])$/, '$1/'),
  /** ?backend=webgl: force the WebGL2 backend */
  forceWebGL: q.get('backend') === 'webgl',
  shadows: q.get('shadows') !== '0',
  /** ?continuous=1: render every animation frame (benchmarks) */
  continuous: q.get('continuous') === '1',
  /** per-section CPU times in the HUD */
  prof: q.get('prof') === '1',
  /** GPU timer queries: ?gpu=1 on, ?gpu=0 off (default: with telemetry) */
  gpu: q.get('gpu'),
  /** telemetry to the dev server (?telemetry=0: off) */
  telemetry: q.get('telemetry') !== '0',
  /** render resolution budget, megapixels, and dynamic resolution (?dynres=0: off) */
  megapixels: opt('mp'),   // (default by device tier, tier.ts)
  dynres: q.get('dynres') !== '0',
  /** ?cam=px,py,pz,tx,ty,tz: start view */
  cam: q.get('cam')?.split(',').map(Number) ?? null,
  roads: q.get('roads') === '1',
  traffic: q.get('traffic') !== '0',
  /** emergency vehicles (siren): mean seconds between calls near where you look (?sirens=0: none) */
  sirens: num('sirens', 240),
  agents: opt('agents'), walkers: opt('walkers'), cyclists: opt('cyclists'),
  /** ?date=YYYY-MM-DD, ?time=hours (not live), ?weather=clear|... */
  date: q.get('date'),
  time: q.get('time'),
  weather: q.get('weather'),
  /** level of detail: max on-screen voxel size (px), resident faces (millions) */
  err: num('err', 8),
  budget: opt('budget'),   // resident faces, millions (default by device tier, tier.ts)
  labels: q.get('mode') === 'labels',
  tilt: q.get('tilt') === '1',
} as const;

/** a data file's address (path relative to the data root, e.g. `roads/schaerbeek.json`) */
export const dataUrl = (path: string) => config.data + path;
