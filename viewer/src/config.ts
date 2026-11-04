// Settings from the page URL (tests, benchmarks, tuning): every query parameter the viewer reads, in one place.
const q = new URLSearchParams(location.search);
const num = (k: string, fallback: number) => (q.has(k) ? Number(q.get(k)) : fallback);
const opt = (k: string) => (q.has(k) ? Number(q.get(k)) : null);

export const config = {
  roi: q.get('roi') ?? 'schaerbeek',
  /** ?backend=webgl: force the WebGL2 backend */
  forceWebGL: q.get('backend') === 'webgl',
  shadows: q.get('shadows') !== '0',
  /** ?continuous=1: render every animation frame (benchmarks); =city: also redraw the cached city layer */
  continuous: q.get('continuous') === '1' || q.get('continuous') === 'city',
  cityAlways: q.get('continuous') === 'city',
  /** per-section CPU times in the HUD */
  prof: q.get('prof') === '1',
  /** GPU timer queries: ?gpu=1 on, ?gpu=0 off (default: with telemetry) */
  gpu: q.get('gpu'),
  /** telemetry to the dev server (?telemetry=0: off) */
  telemetry: q.get('telemetry') !== '0',
  /** render resolution budget, megapixels, and dynamic resolution (?dynres=0: off) */
  megapixels: num('mp', 3.5),
  dynres: q.get('dynres') !== '0',
  /** ?cam=px,py,pz,tx,ty,tz: start view */
  cam: q.get('cam')?.split(',').map(Number) ?? null,
  roads: q.get('roads') === '1',
  traffic: q.get('traffic') !== '0',
  cars: num('cars', 400),
  agents: opt('agents'), walkers: opt('walkers'), cyclists: opt('cyclists'),
  /** ?date=YYYY-MM-DD, ?time=hours (not live), ?weather=clear|... */
  date: q.get('date'),
  time: q.get('time'),
  weather: q.get('weather'),
  /** level of detail: max on-screen voxel size (px), ?lod=0 off, resident faces (millions) */
  err: num('err', 8),
  lod: q.get('lod') !== '0',
  budget: num('budget', 16),
  labels: q.get('mode') === 'labels',
  tilt: q.get('tilt') === '1',
  bokeh: num('bokeh', 2.5),
  /** ?dbg=shadow: shadow maps as colours; ?dbg=holes: magenta where nothing was drawn */
  debug: q.get('dbg'),
} as const;
