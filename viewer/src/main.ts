import { config, dataUrl } from './config';
import { levelTiles, tileCache } from './tileFormat';
import { deviceTier } from './tier';
import { H, brusselsInstant, brusselsNow, weekday } from './clock';
import { DynamicResolution } from './dynres';
import { PlaceIndex } from './places';
import * as THREE from 'three/webgpu';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { TileManager, type LevelInfo } from './lod';
import { TELEMETRY, telemetry, type FrameInfo } from './telemetry';
import { ATLAS_W, LABELS, viewMode } from './voxelMaterial';
import { buildLanes, loadRoads, roadOverlay } from './roads';
import { Traffic } from './traffic';
import { Agents } from './agents';
import { AgentView } from './agentView';
import { Pedestrians } from './pedestrians';
import { Transit, unpackTimetable } from './transit';
import { Parking } from './parking';
import { baseVoxel, gridOrigin, litShare, loadLamps, night, nightFor, windowsLit } from './nightlights';
import { Voxels } from './voxels';
import { Walk } from './walk';
import { Inspector } from './inspect';
import { lighting, sunDirection, sunPosition } from './sun';
import { applyWeather, describe, fetchWeather, Precipitation, presetWeather, type Weather } from './weather';
import { blurPx, DEPTH_RATIO, focalLength, focusDistance } from './tiltshift';
import { Deferred, DYNAMIC_SHADOW_LAYER, beaconTime } from './deferred';
import { NoiseMap } from './noisemap';
import { CitySound } from './citySound';
import { prof } from './prof';

telemetry.start();

interface Manifest {
  build?: string; origin: [number, number, number]; tileSize: number;
  home: [number, number]; levels: LevelInfo[]; attribution: string;
}
type Vec3 = [number, number, number];
interface View { pos: Vec3; target: Vec3 }

const ROI = config.roi;
const SHADOWS = config.shadows;
// render every animation frame (benchmarks); default: only when something changed
const CONTINUOUS = config.continuous;
const PROF = config.prof;   // per-section CPU times in the HUD
const GPU_TIMING = config.gpu !== '0' && (TELEMETRY || config.gpu === '1');
const SKY = new THREE.Color(0xbcd3e6);
const LAT = 50.866, LON = 4.375; // Schaerbeek

const hud = document.getElementById('hud')!;
document.getElementById('attr')!.textContent = 'loading…';

const round = (v: number) => Math.round(v * 100) / 100;

async function gpuName(renderer: THREE.WebGPURenderer): Promise<string> {
  const be = renderer.backend as any;
  if (be.isWebGPUBackend) {
    const a = await navigator.gpu?.requestAdapter();
    return a?.info ? `${a.info.vendor} ${a.info.architecture} ${a.info.description}`.trim() : 'unknown';
  }
  const gl: WebGL2RenderingContext | undefined = be.gl;
  const ext = gl?.getExtension('WEBGL_debug_renderer_info');
  return gl ? String(gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER)) : 'unknown';
}

const matricesClose = (a: THREE.Matrix4, b: THREE.Matrix4, eps: number) =>
  a.elements.every((v, i) => Math.abs(v - b.elements[i]) <= eps);

async function main() {
  const tStart = performance.now();
  const manifest: Manifest = await (await fetch(dataUrl(`tiles/${ROI}/manifest.json`), { cache: 'no-store' })).json();
  for (const lv of manifest.levels) lv.tiles = levelTiles(lv, manifest.tileSize);
  const L0 = manifest.levels[0];
  const VS = L0.voxelSize, CH = manifest.tileSize;
  // data credits, short as on web maps (the full wording in each tooltip and in the README, Data sources and licences)
  const attr = document.getElementById('attr')!;
  attr.textContent = '© ';
  for (const [i, [text, href, title]] of ([
    ['Paradigm / UrbIS', 'https://datastore.brussels', manifest.attribution + ', voxelised'],
    ['OpenStreetMap', 'https://www.openstreetmap.org/copyright', '© OpenStreetMap contributors, ODbL'],
    ['Mapillary', 'https://www.mapillary.com', 'street-level imagery © Mapillary contributors, CC BY-SA 4.0'],
    ['STIB-MIVB', 'https://data.stib-mivb.brussels', 'timetable: STIB-MIVB open data, modified (clipped, simulated)'],
    ['Open-Meteo', 'https://open-meteo.com', 'weather data by Open-Meteo.com, CC BY 4.0'],
  ] as const).entries()) {
    if (i) attr.append(' · ');
    const el = document.createElement('a');
    el.href = href; el.target = '_blank'; el.rel = 'noopener'; el.textContent = text; el.title = title;
    attr.append(el);
  }
  const repo = document.createElement('a');
  repo.href = 'https://github.com/deneland/bxl-vice-city'; repo.target = '_blank'; repo.rel = 'noopener';
  repo.textContent = 'source on GitHub'; repo.title = 'code (MIT) and how the city is built';
  attr.prepend(repo, ' · ');

  // scene frame: origin at the home point, ground ≈ y 0
  const homeVX = (manifest.home[0] - manifest.origin[0]) / VS;
  const homeVY = (manifest.home[1] - manifest.origin[1]) / VS;
  const groundGuess = 20; // voxels above z0, refined once chunks arrive

  // ?backend=webgl: force WebGL2 (what Firefox and Safari get); default: WebGPU where available
  const renderer = new THREE.WebGPURenderer({ antialias: false, trackTimestamp: GPU_TIMING, forceWebGL: config.forceWebGL });  // FXAA in deferred.ts
  // Dynamic resolution: render at most MP megapixels (high-DPI screens would otherwise draw 4-8 MP every
  // frame), and lower the internal resolution while moving if frames get slow; raise it again when smooth.
  const tier = deviceTier();
  const MP = tier.megapixels * 1e6;
  const res = new DynamicResolution(() => Math.min(devicePixelRatio, 2, Math.sqrt(MP / (innerWidth * innerHeight))),
    () => resize(), config.dynres);
  renderer.setPixelRatio(res.ratio);
  renderer.setSize(innerWidth, innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  document.body.appendChild(renderer.domElement);
  await renderer.init();
  const backend = (renderer.backend as any).isWebGPUBackend ? 'WebGPU' : 'WebGL2';
  telemetry.setEnv({ backend, gpu: await gpuName(renderer), shadows: SHADOWS, gpuTiming: GPU_TIMING, tier: `${tier.name} (${tier.why})`,
    pixelRatio: renderer.getPixelRatio() });

  renderer.info.autoReset = false;
  // GPU time per pass: label every render call (frame, call index) so the timestamp queries can be named
  const passLabels = new Map<number, string[]>();
  if (GPU_TIMING) {
    const render = renderer.render.bind(renderer);
    renderer.render = ((sc: THREE.Object3D, cam: THREE.Camera) => {
      const rt = renderer.getRenderTarget() as any;
      const f = renderer.info.frame;
      if (!passLabels.has(f)) { passLabels.set(f, []); if (passLabels.size > 120) passLabels.delete(passLabels.keys().next().value!); }
      passLabels.get(f)![renderer.info.render.frameCalls] = `${sc.name || sc.type}>${rt?.label ?? (rt ? 'rt' : 'screen')}`;
      return render(sc, cam);
    }) as typeof renderer.render;
  }   // several render calls per frame; reset once per frame below
  // two scenes: the static city (cached G-buffer) and the forward layer (lights, fog, moving and
  // transparent things); see deferred.ts
  const cityScene = new THREE.Scene();
  cityScene.name = 'city';
  const scene = new THREE.Scene();
  scene.name = 'forward';
  const sky = SKY.clone();
  scene.fog = new THREE.Fog(SKY.clone(), 700, 2600);
  // shadows (on by default, ?shadows=0 off): our own two-layer shadow maps in deferred.ts; the box covers the region from any sun direction
  const deferred = new Deferred(renderer, SHADOWS ? { half: (Math.hypot(L0.size[0], L0.size[1]) * VS) / 2 + 60 } : undefined);
  // what the next tick must redraw (set from input, settings and loading anywhere below)
  let cityDirty = true;       // the city G-buffer (view mode, size)
  let dirty = true;           // something outside camera / tiles changed: draw the next tick
  let dynShadowDirty = true;  // the vehicle shadows (traffic switched on / off)
  let clickTimer = 0;         // a click waits briefly: a double-click sets the orbit point instead

  const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.5, 6000);
  camera.position.set(-110, 120, 150);

  const controls = new MapControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 4;
  controls.maxDistance = 2200;
  controls.screenSpacePanning = false;
  controls.zoomToCursor = true;   // the orbit point follows what you zoom into
  controls.listenToKeyEvents(window);
  controls.keyPanSpeed = 25;
  const home: View = { pos: camera.position.toArray() as Vec3, target: [0, 0, 0] };

  const setView = (v: View) => {
    camera.position.fromArray(v.pos);
    controls.target.fromArray(v.target);
    controls.update();
  };
  const getView = (): View => ({
    pos: camera.position.toArray().map(round) as Vec3,
    target: controls.target.toArray().map(round) as Vec3,
  });
  // the address bar follows the view, the chosen time and the open overlays (a link to share; see config.ts, at):
  // written half a second after things settle (browsers limit how often a page may rewrite its URL)
  let shareTimer = 0;
  function shareSoon() {
    clearTimeout(shareTimer);
    shareTimer = setTimeout(() => {
      try { history.replaceState(history.state, '', shareUrl()); } catch { /* (rate-limited: the next change writes it) */ }
    }, 500);
  }
  function shareUrl() {
    const u = new URL(location.href), q = u.searchParams;
    const set = (k: string, v: string | null) => (v === null ? q.delete(k) : q.set(k, v));
    if (!walk.active) {   // (walking: the link keeps the last map view)
      const v = getView(), atHome = [...v.pos, ...v.target].every((n, i) => Math.abs(n - [...home.pos, ...home.target][i]) < 0.05);
      set('at', atHome ? null : [...toL72(v.pos), ...toL72(v.target)].map((n) => n.toFixed(1)).join(','));
      q.delete('cam');
    }
    set('time', live ? null : timeInput.value);
    set('date', live ? null : day);
    set('mode', viewMode.value ? 'labels' : null);
    set('tilt', tiltOn ? '1' : null);
    set('roads', roadsOn ? '1' : null);
    set('noise', noiseWanted ? '1' : null);
    set('traffic', trafficWanted ? null : '0');
    u.search = q.toString().replace(/%2C/g, ',');   // (readable: commas are fine in a query)
    return u;
  }
  const hemi = new THREE.HemisphereLight(0xdfeeff, 0x5a5040, 1.6);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xfff1dc, 3.2);
  scene.add(sun, sun.target);

  const world = new THREE.Group();
  // put the home tile's lowest surface at y≈0 (known from the manifest, before loading)
  const homeTile = L0.tiles.find((t) => t.tx === Math.floor(homeVX / CH) && t.ty === Math.floor(homeVY / CH));
  world.position.set(-homeVX * VS, -(homeTile?.zmin ?? groundGuess) * VS, homeVY * VS);
  cityScene.add(world);
  // shareable links give the view in Lambert 72 and metres above sea level, not scene metres: a link stays
  // valid when the data is rebuilt or the home point moves
  const zBase = manifest.origin[2] + (homeTile?.zmin ?? groundGuess) * VS;
  const toL72 = ([x, y, z]: Vec3): Vec3 => [x + manifest.home[0], manifest.home[1] - z, y + zBase];
  const fromL72 = ([X, Y, Z]: number[]): Vec3 => [X - manifest.home[0], Z - zBase, manifest.home[1] - Y];
  const inRegion = ([x, , z]: Vec3) => x >= world.position.x && x <= world.position.x + L0.size[0] * VS &&
    z <= world.position.z && z >= world.position.z - L0.size[1] * VS;
  const valid = (v: number[] | null): v is number[] => v?.length === 6 && v.every(Number.isFinite);
  if (valid(config.cam)) setView({ pos: config.cam.slice(0, 3) as Vec3, target: config.cam.slice(3) as Vec3 });
  else if (valid(config.at) && inRegion(fromL72(config.at.slice(3)))) {
    setView({ pos: fromL72(config.at.slice(0, 3)), target: fromL72(config.at.slice(3)) });
  }
  // the city never moves: no per-render matrix update of every resident tile (lod.ts updates a tile's
  // world matrix once, when it is added)
  world.updateMatrixWorld(true);
  cityScene.matrixWorldAutoUpdate = false;
  // same frame for the forward layer (lamps, roads overlay, traffic)
  const fx = new THREE.Group();
  fx.position.copy(world.position);
  scene.add(fx);
  const glowScene = new THREE.Scene();
  glowScene.name = 'glow';   // additive light: lamp glows
  const glowFx = new THREE.Group();
  glowFx.position.copy(world.position);
  glowScene.add(glowFx);
  // region centre in the scene: sun target, so the shadow box covers everything
  sun.target.position.set((L0.size[0] / 2) * VS + world.position.x, 0, -(L0.size[1] / 2) * VS + world.position.z);
  gridOrigin.value.copy(world.position);
  baseVoxel.value = VS;
  let lamps: THREE.Object3D | null = null;
  loadLamps(dataUrl(`lights/${ROI}.json`), glowFx, VS, [L0.size[0], L0.size[1]]).then((m) => {
    lamps = m;
    if (m) m.visible = night.value > 0.01;
    dirty = true;
  }).catch((e) => console.warn('street lights unavailable', e));
  // movement network (OSM): debug overlay (R) and traffic (T)
  const network = loadRoads(dataUrl(`roads/${ROI}.json`)).then((net) => net && { net, lanes: buildLanes(net, VS) })
    .catch((e) => { console.warn('road network unavailable', e); return null; });
  let roads: THREE.Object3D | null = null;
  let roadsOn = config.roads;
  const setRoads = async (on: boolean) => {
    if (on && !roads) {
      const n = await network;
      if (!n) return;
      roads = roadOverlay(n.net, n.lanes, VS);
      if (peds) roads.add(peds.overlay());   // pavements and crossings
      fx.add(roads);
    }
    if (roads) roads.visible = on;
    roadsOn = on;
    shareSoon();
    dirty = true;
  };
  if (config.roads) setRoads(true);
  let traffic: Traffic | null = null;
  let agents: Agents | null = null;
  let peds: Pedestrians | null = null;
  let transit: Transit | null = null;
  // sound and the noise map (both need traffic: built with it)
  let city: CitySound | null = null;
  let noise: NoiseMap | null = null;
  let soundWanted = false, noiseWanted = config.noise;
  let nextCall = performance.now() + config.sirens * (0.2 + 0.5 * Math.random()) * 1000;   // the first emergency call
  // agent clock: seconds of the day; runs on from the chosen time (live: the Brussels clock)
  let clockBase = 0, clockAt = performance.now();
  const agentTime = () => Math.min(24 * H - 1, clockBase + (performance.now() - clockAt) / 1000);
  let trafficOn = false;
  let trafficWanted = config.traffic || config.noise;   // (the noise map comes from the traffic)
  const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => setTimeout(r, 0)));
  let buildingTraffic = false;   // (set-up awaits frames: a second T press must not start another)
  const setTraffic = async (on: boolean) => {
    trafficOn = trafficWanted = on;
    shareSoon();
    if (on && !traffic && !buildingTraffic) {
      buildingTraffic = true;
      try {
        // all data files at once
        const json = (u: string) => fetch(dataUrl(u)).then((r) => (r.ok ? r.json() : null)).catch(() => null);
        const [n, palette, tt, places, pk] = await Promise.all([network,
          json(`vehicles/${ROI}.json`),    // car paint colours measured on the parked cars in the LiDAR (vehicles.py)
          json(`transit/${ROI}.json`),     // STIB lines from the timetable (transit.py); without it trams wander
          json(`places/${ROI}.json`),      // homes and destinations of the people (places.py)
          json(`parking/${ROI}.json`)]);   // kerbside parking places (parking.py)
        if (!n) { trafficOn = false; return; }
        // (a frame between the big set-up steps: none of them blocks for long on a slow CPU)
        await nextFrame();
        traffic = new Traffic(n.lanes, deferred.vehicleMaterial(),
          (node) => n.net.nodes[node]?.[4] === 1,                             // region edge: through traffic
          palette?.palette ?? []);
        if (tt) transit = new Transit(unpackTimetable(tt), traffic, VS);
        traffic.enableLayer(DYNAMIC_SHADOW_LAYER);
        fx.add(traffic.group, traffic.lights);
        await nextFrame();
        // pedestrians on the pavements (both sides of every street), crossing at intersections
        peds = new Pedestrians(n.net, VS, deferred.pedestrianMaterial(), traffic);
        peds.mesh.layers.enable(DYNAMIC_SHADOW_LAYER);
        await peds.build(nextFrame);
        fx.add(peds.mesh);
        if (roads) roads.add(peds.overlay());
        // population in proportion to the homes (calibrated on a 1.5 km area: 5224 homes, 8000 drivers)
        const k = places ? places.homes.length / 5224 : 1;
        await nextFrame();
        const parking = pk ? new Parking(pk, VS) : null;
        const index = places ? new PlaceIndex(places, VS) : null;
        if (index && tt) index.setStops(tt.stops, VS);
        if (index) agents = new Agents(index, traffic, config.agents ?? Math.round(8000 * k), peds,
          config.walkers ?? Math.round(10000 * k), config.cyclists ?? Math.round(1500 * k), parking);
        if (agents && tt) {   // tram / bus riders: who gets on / off when a vehicle stops
          const ag = agents;
          traffic.onDwell = (id) => ag.onDwell(id);
        }
        noise = new NoiseMap([L0.size[0] * VS, L0.size[1] * VS], () => { dirty = true; });
        noise.visible = noiseWanted;
        noise.setLanes(n.lanes);
        city = new CitySound(traffic, noise);
        if (soundWanted) city.setSound(true);
      } catch (e) {   // (whatever failed: traffic stays off, and T can try again)
        console.error('traffic set-up failed', e);
        trafficOn = false;
      } finally {
        buildingTraffic = false;
      }
    }
    if (traffic) traffic.group.visible = traffic.lights.visible = trafficOn;
    if (peds) peds.mesh.visible = trafficOn;
    city?.setActive(trafficOn);
    dynShadowDirty = true;   // redraw (or clear) the vehicle shadows once
    dirty = true;
  };

  // --- time of day + weather: live over Schaerbeek, or chosen with the slider ---------
  const timeInput = document.getElementById('time') as HTMLInputElement;
  const timeLabel = document.getElementById('time-label')!;
  const liveBtn = document.getElementById('live') as HTMLButtonElement;
  let day = config.date ?? brusselsNow()[0];
  let live = config.time === null;
  let weather: Weather | null = null;   // null: unknown (not loaded or service down) -> clear-sky look
  const precip = new Precipitation();
  scene.add(precip.mesh);
  let precipOn = false;
  const setTime = (hours: number) => {
    const { az, alt } = sunPosition(brusselsInstant(day, hours), LAT, LON);
    clockBase = hours * H; clockAt = performance.now();
    const l = lighting(alt);
    const fog = applyWeather(weather ?? presetWeather('clear'), l);
    (scene.fog as THREE.Fog).near = fog.fogNear;
    (scene.fog as THREE.Fog).far = fog.fogFar;
    precipOn = weather ? precip.set(weather) : false;
    sunDirection(az, Math.max(alt, 0.02), sun.position).multiplyScalar(2000).add(sun.target.position);
    sun.color.copy(l.sunColor);
    sun.intensity = l.sunIntensity;
    hemi.color.copy(l.hemiColor);
    hemi.intensity = l.hemiIntensity;
    night.value = nightFor(alt * 180 / Math.PI);
    windowsLit.value = litShare(hours);
    if (lamps) lamps.visible = night.value > 0.01;
    sky.copy(l.sky);
    scene.fog!.color.copy(l.sky);
    document.body.style.background = '#' + l.sky.getHexString();
    const hh = Math.floor(hours), mm = Math.floor((hours - hh) * 60);
    timeLabel.textContent = `${live ? 'live · ' : ''}${day} ${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}` +
      ` · ${weather ? `${describe(weather)}, ${Math.round(weather.temp)}°C` : 'weather unavailable'} · sun ${(alt * 180 / Math.PI).toFixed(0)}°`;
    liveBtn.style.display = live ? 'none' : '';
    shareSoon();
    dirty = true;
  };
  const tick = () => { // live: follow the clock
    if (!live) return;
    const [d, h] = brusselsNow();
    day = d;
    timeInput.value = String(h);
    setTime(h);
  };
  timeInput.value = config.time ?? String(brusselsNow()[1]);
  timeInput.addEventListener('input', () => { live = false; setTime(Number(timeInput.value)); });
  liveBtn.addEventListener('click', () => { live = true; tick(); });
  setInterval(tick, 30e3);

  // GPU context lost (the browser or OS took the GPU back, e.g. a phone under memory pressure): everything on
  // the GPU is gone. Rebuilding it in place is not worth the complexity: reload at the same view and time.
  // Twice within a minute: stop and say so (a reload loop helps nobody).
  renderer.onDeviceLost = (info: { message?: string }) => {
    console.warn('GPU context lost', info?.message ?? '');
    const KEY = 'bxl-context-lost', last = Number(sessionStorage.getItem(KEY) ?? 0);
    const msg = document.getElementById('lost')!;
    msg.style.display = 'block';
    if (Date.now() - last < 60_000) {
      msg.innerHTML = 'The graphics card stopped drawing this page (twice in a minute).<br>' +
        'Close other tabs or apps, then <a href="#" onclick="location.reload();return false">reload</a>.';
      return;
    }
    try { sessionStorage.setItem(KEY, String(Date.now())); } catch { /* (private mode: no loop guard) */ }
    const u = shareUrl();
    msg.textContent = 'The graphics card stopped drawing this page; reloading…';
    setTimeout(() => location.replace(u.href), 800);
  };
  const wPreset = config.weather;
  const updateWeather = async () => {
    try {
      weather = wPreset ? presetWeather(wPreset) : await fetchWeather(LAT, LON);
      setTime(Number(timeInput.value));
      telemetry.event('weather', { ...weather! });
    } catch (e) { console.warn('weather unavailable', e); }
  };
  if (!wPreset) setInterval(updateWeather, 10 * 60e3);

  // --- level-of-detail tiles ------------------------------------------------
  const lod = new TileManager(dataUrl(`tiles/${ROI}`), CH, manifest.levels, world, config.err, renderer,
    tier.inflight, tier.workers, tier.atlasRows);
  lod.query = manifest.build ? `?v=${manifest.build}` : '';
  lod.cache = config.tileCache ? tileCache(ROI, manifest.build) : null;
  lod.errScale = tier.detail;
  lod.budgetFaces = Math.min(tier.budgetFaces, (tier.atlasRows * ATLAS_W) / 2);  // resident faces (the atlas holds twice that: slack)
  const t0 = performance.now();
  let loadMs = 0;
  const ready = new Promise<void>((resolve) => {
    // first full, settled view: all tiles the initial camera needs are meshed
    const check = () => lod.whenIdle().then(() => {
      loadMs = performance.now() - t0;
      telemetry.event('initial-load', { ms: Math.round(performance.now() - tStart), ...lod.stats });
      resolve();
    });
    setTimeout(check, 50);
  });
  // traffic, people and their day plans after the first full view of the city: the city comes first
  // (they competed for the main thread), people and cars appear a moment later
  if (trafficWanted) ready.then(() => setTraffic(true));

  // --- CPU voxel access, walking, picking ---------------------------------
  const vox = new Voxels(dataUrl(`tiles/${ROI}`), CH, L0, world, lod.query, lod.cache);
  const crosshair = document.getElementById('crosshair')!;
  const walkHint = document.getElementById('walk-hint')!;
  const walk = new Walk(camera, renderer.domElement, vox, (active) => {
    dirty = true;
    controls.enabled = !active;
    crosshair.style.display = active && walk.locked ? 'block' : 'none';
    walkHint.style.display = active && !walk.locked ? 'block' : 'none';
  });
  const toggleWalk = () => {
    if (walk.active) {
      walk.exit();
      // back to the map, looking at where you stood
      const p = camera.position.clone();
      const back = camera.getWorldDirection(new THREE.Vector3()).setY(0).normalize().multiplyScalar(-60);
      setView({ pos: [p.x + back.x, p.y + 50, p.z + back.z], target: [p.x, p.y - 1.7, p.z] });
      telemetry.event('walk', { on: false });
    } else {
      walk.enter(controls.target.clone()).then(() => telemetry.event('walk', { on: walk.active }));
    }
  };
  const inspector = new Inspector(vox, scene, manifest.origin[2], () => { dirty = true; });
  const agentView = new AgentView(() => traffic, () => agents, () => peds, fx, controls, camera, () => { dirty = true; });
  const raycaster = new THREE.Raycaster();
  let down: { x: number; y: number; t: number } | null = null;
  renderer.domElement.addEventListener('pointerdown', (e) => { if (e.button === 0) down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (!down || e.button !== 0) return;
    const click = Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5 && performance.now() - down.t < 400;
    down = null;
    if (!click) return;
    if (walk.active && !walk.locked) return; // this click re-locks the mouse
    const ndc = walk.active ? new THREE.Vector2(0, 0)
      : new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    // a vehicle first (who drives it; hit-tested now, it moves on), else the voxel under the cursor
    raycaster.setFromCamera(ndc, camera);
    const found = trafficOn ? agentView.find(raycaster.ray) : null;
    // wait briefly: a double-click sets the orbit point instead
    clearTimeout(clickTimer);
    clickTimer = setTimeout(() => {
      raycaster.setFromCamera(ndc, camera);
      if (agentView.select(found)) { inspector.clear(); telemetry.event('inspect-vehicle', {}); return; }
      agentView.clear();
      inspector.pick(raycaster.ray).then(() => telemetry.event('inspect', {}));
    }, walk.active ? 0 : 250);
  });

  // double-click: orbit around / focus on that voxel (smoothly turn towards it)
  let targetAnim: { from: THREE.Vector3; to: THREE.Vector3; t0: number } | null = null;
  renderer.domElement.addEventListener('dblclick', async (e) => {
    clearTimeout(clickTimer);
    if (walk.active) return;
    raycaster.setFromCamera(new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1), camera);
    const hit = await vox.raycast(raycaster.ray.origin, raycaster.ray.direction);
    if (!hit) return;
    const to = vox.toScene(hit.voxel.clone().addScalar(0.5));
    targetAnim = { from: controls.target.clone(), to, t0: performance.now() };
    telemetry.event('set-target', {});
  });

  // --- tilt-shift ----------------------------------------------------------
  let tiltOn = false;
  const setTilt = (on: boolean) => {
    tiltOn = on;
    deferred.setTilt(on);
    lod.errAt = on ? blurPx : null;   // blurred areas need less detail
    shareSoon();
    dirty = true;
  };

  // --- view mode: colours or semantic classes ----------------
  const legend = document.getElementById('legend')!;
  legend.innerHTML = '<b>classes</b> (LiDAR + street-level photos)<br>' + LABELS.map(([, name, h]) =>
    `<span style="background:#${h.toString(16).padStart(6, '0')}"></span>${name}`).join('<br>');
  const setMode = (labels: boolean) => {
    viewMode.value = labels ? 1 : 0;
    legend.style.display = labels ? 'block' : 'none';
    cityDirty = true;
    shareSoon();
    telemetry.event('view-mode', { mode: labels ? 'classes' : 'colour' });
  };
  setMode(config.labels);

  // --- sound and noise map -------------------------------------------------
  const soundBtn = document.getElementById('sound') as HTMLButtonElement;
  const noiseLegend = document.getElementById('noise-legend')!;
  noiseLegend.innerHTML = NoiseMap.legend();
  const setSound = (on: boolean) => {
    soundWanted = on;
    city?.setSound(on);   // (the audio engine is created here: in a click, as browsers require)
    soundBtn.textContent = on ? '🔊 sound on' : '🔇 sound off';
    telemetry.event('sound', { on });
  };
  soundBtn.addEventListener('click', () => setSound(!soundWanted));

  // --- share: a link to this view (phones: the share sheet; else copied). Only the view's own parameters,
  // not this browser's tuning (tier, data, telemetry, ...)
  const shareBtn = document.getElementById('share') as HTMLButtonElement;
  let shareReset = 0;
  shareBtn.addEventListener('click', async () => {
    const all = shareUrl(), u = new URL(all.origin + all.pathname);
    for (const k of ['roi', 'at', 'time', 'date', 'mode', 'tilt', 'roads', 'noise', 'traffic']) {
      const v = all.searchParams.get(k);
      if (v !== null) u.searchParams.set(k, v);
    }
    const url = u.href.replace(/%2C/g, ',');
    const done = (text: string) => {
      shareBtn.textContent = text;
      clearTimeout(shareReset);
      shareReset = setTimeout(() => { shareBtn.textContent = '🔗'; }, 1500);
    };
    try {
      if (navigator.share && matchMedia('(pointer: coarse)').matches) await navigator.share({ title: document.title, url });
      else { await navigator.clipboard.writeText(url); done('copied ✓'); }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') prompt('link to this view', url);   // (no clipboard access: copy it by hand)
    }
    telemetry.event('share', {});
  });
  const setNoise = (on: boolean) => {
    noiseWanted = on;
    if (noise) noise.visible = on;
    noiseLegend.style.display = on ? 'block' : 'none';
    if (on && !trafficOn) setTraffic(true);   // (the map comes from the traffic)
    shareSoon();
    dirty = true;
    telemetry.event('noise-map', { on });
  };

  // --- input ---------------------------------------------------------------
  let snapshotRequested = false;
  // the default view stays clean: frame and tile statistics behind D, the controls behind ? (or the button)
  const toggleHelp = () => document.body.classList.toggle('help');
  document.getElementById('help-btn')!.addEventListener('click', toggleHelp);
  window.addEventListener('keydown', (e: KeyboardEvent) => {
    switch (e.key.toLowerCase()) {
      case 'd': if (!walk.active) document.body.classList.toggle('debug'); break;   // (in walk mode D steps right)
      case '?': toggleHelp(); break;
      case 'h': if (!walk.active) setView(home); break;
      case 'l': setMode(viewMode.value === 0); break;
      case 'p': if (TELEMETRY) snapshotRequested = true; break;
      case 'v': toggleWalk(); break;
      case 'm': setTilt(!tiltOn); break;
      case 'r': setRoads(!roads?.visible); break;
      case 't': setTraffic(!trafficOn); break;
      case 'n': if (!walk.active) setNoise(!noiseWanted); break;
      case 'escape': inspector.clear(); agentView.clear(); break;
      case 'f': agentView.setFollow(!agentView.isFollowing); break;
    }
    dirty = true;
  });
  setTime(Number(timeInput.value));
  updateWeather();
  setTilt(config.tilt);
  noiseLegend.style.display = noiseWanted ? 'block' : 'none';   // (?noise=1: the map itself comes with the traffic)
  const resize = () => {
    dirty = true;
    cityDirty = true;
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setPixelRatio(res.ratio);
    renderer.setSize(innerWidth, innerHeight);
    const s = renderer.getDrawingBufferSize(new THREE.Vector2());
    deferred.setSize(s.x, s.y);
  };
  window.addEventListener('resize', () => { res.clamp(); resize(); });
  telemetry.setState(() => ({
    view: getView(), lod: lod.stats, errPx: lod.errPx, pixelRatio: +res.ratio.toFixed(2),
    screen: { w: innerWidth, h: innerHeight, dpr: devicePixelRatio },
  }));

  // --- loop + HUD ----------------------------------------------------------
  const frameTimes: number[] = [];
  const frameHooks = new Set<(f: FrameInfo) => void>();
  const gpuHooks = new Set<(ms: number) => void>();
  const lastCam = new THREE.Matrix4();
  let last = performance.now(), hudAt = 0, gpuMs = 0, gpuInFlight = false;
  let flashUntil = 0;
  let lastTickRendered = false, rendered = 0, skipped = 0, cityRenders = 0, shadowRenders = 0;
  const shadowSun = new THREE.Vector3();   // sun direction of the current city shadow map
  let shadowTiles = true;
  let wasMoving = false, lastCpu = 0;
  let prevTags: Record<string, boolean> = {}, prevOwn: Record<string, boolean> = {}, shadowStepped = false, dynShadowDrawn = false;
  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const interval = now - last;
    last = now;

    if (targetAnim) {
      const k = Math.min(1, (now - targetAnim.t0) / 400), e = k * k * (3 - 2 * k);
      controls.target.lerpVectors(targetAnim.from, targetAnim.to, e);
      if (k >= 1) targetAnim = null;
    }
    if (!walk.active) agentView.update(agentTime());   // (follow moves camera and orbit point)
    if (walk.active) walk.update(interval / 1000);
    else controls.update();
    if (trafficOn && traffic) {
      const now = agentTime();
      // detail near where you look (walking: where you are)
      const f = walk.active ? camera.position : controls.target;
      agents?.setFocus(f.x - fx.position.x, f.z - fx.position.z);
      transit?.setFocus(f.x - fx.position.x, f.z - fx.position.z);
      if (agents) agents.budgetMs = wasMoving ? 1 : 2;   // (moving: the frame needs its time for the city)
      prof.begin('agents'); agents?.update(now, day); prof.end('agents');
      prof.begin('transit'); transit?.update(now, weekday(day)); prof.end('transit');
      prof.begin('traffic'); traffic.update(Math.min(interval, 100) / 1000, now); prof.end('traffic');
      prof.begin('peds'); peds?.update(Math.min(interval, 100) / 1000); prof.end('peds');
      // emergency calls: now and then a police car or ambulance passes near where you look
      if (config.sirens > 0 && performance.now() > nextCall) {
        if (!traffic.vehicles.some((v) => v.siren)) traffic.spawnEmergency(f.x - fx.position.x, f.z - fx.position.z);
        nextCall = performance.now() + config.sirens * (0.5 + Math.random()) * 1000;
      }
      if (city && noise) {
        prof.begin('sound');
        // the ear: walking, where you stand; on the map, a quarter of the way from the orbit point to the
        // camera (zoomed out, the city goes quiet)
        const ear = walk.active ? camera.position.clone() : controls.target.clone().lerp(camera.position, 0.25);
        city.ear.copy(ear).sub(fx.position);
        camera.getWorldDirection(city.fwd);
        city.up.set(0, 1, 0).applyQuaternion(camera.quaternion);
        city.update(Math.min(interval, 100) / 1000);
        noise.setFocus(f.x - fx.position.x, f.z - fx.position.z);
        noise.hour = now / H;
        noise.update(Math.min(interval, 100) / 1000);
        prof.end('sound');
      }
    }
    beaconTime.value = now / 1000;
    camera.updateMatrixWorld();
    // damping keeps nudging the camera by invisible amounts forever; ignore changes below 1e-4
    const moving = !matricesClose(lastCam, camera.matrixWorld, 1e-4);
    wasMoving = moving;
    if (moving) { lastCam.copy(camera.matrixWorld); shareSoon(); }
    if (tiltOn) {
      // focus on the orbit target (map) or 20 m ahead (walking), measured along the view direction
      const d = walk.active ? 20 : Math.max(5, controls.target.clone().sub(camera.position).dot(camera.getWorldDirection(new THREE.Vector3())));
      focusDistance.value = d;
      focalLength.value = d * DEPTH_RATIO;
    }
    prof.begin('lod');
    const tilesChanged = moving || dirty || lod.needsUpdate ? lod.update(camera, renderer.domElement.height) : false;
    prof.end('lod');
    if (lastTickRendered) res.frame(interval, moving, now, lastCpu, lod.stats.loading > 0 || (agents?.replanning ?? 0) > 0);
    const cityChanged = moving || tilesChanged || cityDirty;
    // (a city shadow rebuild in slices needs frames to finish, also when nothing else changes)
    if (!(CONTINUOUS || cityChanged || precipOn || trafficOn || dirty || snapshotRequested || deferred.shadowBusy)) {
      // nothing changed: draw nothing, the GPU stays idle
      lastTickRendered = false;
      skipped++;
      prof.endFrame();
      telemetry.idleTick();
      updateHud(now);
      return;
    }
    dirty = false;
    renderer.info.reset();
    shadowStepped = dynShadowDrawn = false;
    if (traffic && trafficOn) { prof.begin('fleet'); traffic.draw(camera); prof.end('fleet'); }   // vehicles in view
    if (cityChanged) {   // the expensive part: only when the city on screen changed
      prof.begin('city');
      deferred.renderCity(cityScene, camera);
      prof.end('city');
      cityDirty = false;
      cityRenders++;
    }
    deferred.sync(camera, sun, hemi, scene.fog as THREE.Fog, sky);
    if (deferred.shadows) {
      // city shadows: when the sun moved noticeably, or other tiles were loaded (once loading settles)
      if (tilesChanged) shadowTiles = true;
      const sunMoved = deferred.sunDir.value.angleTo(shadowSun) > 0.005;
      if (sunMoved) {   // (time slider: redraw at once, it has to follow the sun)
        prof.begin('shadowCity');
        deferred.renderStaticShadow(cityScene, sun.target.position);
        prof.end('shadowCity');
        shadowSun.copy(deferred.sunDir.value);
        shadowRenders++;
      } else if (shadowTiles && !moving && !lod.stats.loading && !deferred.shadowBusy) {
        deferred.beginStaticShadow(cityScene, sun.target.position);   // other tiles: redraw in slices
        shadowTiles = false;
      }
      shadowStepped = sunMoved;
      if (deferred.shadowBusy && !sunMoved) {
        shadowStepped = true;
        prof.begin('shadowCity');
        if (deferred.stepStaticShadow(cityScene)) shadowRenders++;
        prof.end('shadowCity');
      }
      // vehicles, around what you look at: every other frame is enough (a fast car's shadow lags < 0.25 m;
      // the map and its transform always match)
      // (while the camera moves the city is redrawn every frame too: a real Intel laptop dropped exactly the
      // frames that had both, so then only every 8th frame; a moving view hides the lag)
      dynShadowDrawn = dynShadowDirty || (trafficOn && (rendered & (moving ? 7 : 1)) === 0);
      if (dynShadowDrawn) {
        prof.begin('shadowDyn');
        deferred.renderDynamicShadow(scene, walk.active ? camera.position : controls.target);
        prof.end('shadowDyn');
        dynShadowDirty = false;   // (toggling traffic sets it: redraw or clear once)
      }
    }
    prof.begin('present');
    deferred.present(scene, glowScene, camera);
    prof.end('present');
    rendered++;
    const cpu = performance.now() - now;
    lastCpu = cpu;
    prof.endFrame();
    // a frame interval only means something if the previous tick also rendered
    if (lastTickRendered) {
      frameTimes.push(interval);
      if (frameTimes.length > 240) frameTimes.shift();
      const frame = { interval, cpu, moving, drawCalls: renderer.info.render.drawCalls,
        triangles: renderer.info.render.triangles };
      telemetry.frame(frame);
      telemetry.tags(interval, prevTags);
      for (const h of frameHooks) h(frame);
    }
    lastTickRendered = true;
    // what this frame did (its cost shows in the next interval)
    {
      const t: Record<string, boolean> = { moving, city: cityChanged, cityShadow: shadowStepped, carShadow: dynShadowDrawn,
        simStep: (traffic?.lastSteps ?? 0) > 0, tiles: lod.lastArrived > 0, cpu12: cpu > 12 };
      // (GPU work can land in the next interval: also the frame before, as p:<tag>)
      for (const [k, v] of Object.entries(prevOwn)) t['p:' + k] = v;
      prevOwn = { city: cityChanged, carShadow: dynShadowDrawn, cityShadow: shadowStepped, tiles: lod.lastArrived > 0 };
      prevTags = t;
    }

    if (GPU_TIMING && !gpuInFlight) {
      gpuInFlight = true;
      renderer.resolveTimestampsAsync(THREE.TimestampQuery.RENDER).then((ms) => {
        if (typeof ms === 'number' && ms > 0) {
          const pool = (renderer.backend as any).timestampQueryPool?.render;
          const byFrame = new Map<number, Map<string, number>>();
          for (const [uid, d] of pool?.timestamps ?? []) {
            const m = /^r:(\d+):.*:f(\d+)$/.exec(uid);
            if (!m) continue;
            const f = Number(m[2]), name = passLabels.get(f)?.[Number(m[1]) - 1] ?? `call${m[1]}`;   // (three counts calls from 1)
            if (!byFrame.has(f)) byFrame.set(f, new Map());
            const fm = byFrame.get(f)!;
            fm.set(name, (fm.get(name) ?? 0) + d);
          }
          for (const fm of byFrame.values()) for (const [k, d] of fm) prof.gpuSample(k, d);
          gpuMs = ms;
          telemetry.gpuTime(ms);
          for (const h of gpuHooks) h(ms);
        }
        gpuInFlight = false;
      });
    }
    if (snapshotRequested) {
      snapshotRequested = false;
      telemetry.snapshot(renderer.domElement); // same task as render -> canvas still has the frame
      flashUntil = now + 1200;
    }

    updateHud(now);
  });

  function updateHud(now: number) {
    if (now - hudAt < 500) return;
    hudAt = now;
    const sorted = [...frameTimes].sort((x, y) => x - y);
    const avg = sorted.length ? sorted.reduce((s, v) => s + v, 0) / sorted.length : 0;
    const p99 = sorted[Math.floor(sorted.length * 0.99)] ?? 0;
    hud.textContent =
      (lastTickRendered && avg ? `${backend} · ${(1000 / avg).toFixed(0)} fps · avg ${avg.toFixed(1)} ms · p99 ${p99.toFixed(1)} ms` : `${backend} · idle, nothing to draw`) +
      ` · ${rendered} frames drawn (city ${cityRenders}), ${skipped} skipped` +
      (GPU_TIMING ? ` · gpu ${gpuMs.toFixed(2)} ms` : '') + '\n' +
      `tiles ${lod.stats.perLevel.map((n, l) => `L${l}:${n}`).join(' ')} · ${(lod.stats.faces / 1e6).toFixed(2)}M faces (${(lod.stats.quads / 1e6).toFixed(2)}M quads) · ` +
      `${renderer.info.render.drawCalls} draws · ${lod.stats.loading ? `loading ${lod.stats.loading}` : `${lod.stats.ready}/${lod.stats.total} tiles in memory`}` +
      ` (${(lod.stats.residentFaces / 1e6).toFixed(1)}M faces)` +
      `${loadMs ? ` · first view ${(loadMs / 1000).toFixed(1)} s` : ''} · err ${lod.errPx}px` +
      ` · ${renderer.domElement.width}×${renderer.domElement.height} (×${res.ratio.toFixed(2)})` +
      ` · shadows ${SHADOWS ? 'on' : 'off'}${tiltOn ? ' · tilt-shift' : ''}${walk.active ? ' · walking' : ''}` +
      (TELEMETRY ? ' · telemetry on' : '') + (PROF ? `\ncpu ms avg/max: ${prof.line(8)}` : '') + (now < flashUntil ? '\nsnapshot sent ✓' : '');
  }

  // test hook for the benchmark harness and debugging
  (window as any).__viewer = {
    renderer, scene, cityScene, deferred, camera, controls, ready, setView, getView, home, frameHooks, gpuHooks, lod, vox, walk, inspector,
    setTime, setTilt, toggleWalk, setTraffic, get traffic() { return traffic; }, get agents() { return agents; }, get peds() { return peds; }, get transit() { return transit; }, get city() { return city; }, get noise() { return noise; }, setNoise, setSound, agentTime,
    whenIdle: () => lod.whenIdle(), prof,
    info: () => ({ backend, tier: tier.name, budget: lod.budgetFaces, atlasRows: lod.atlas.rows, errScale: +lod.errScale.toFixed(2), ...lod.stats, loadMs, rendered, cityRenders, shadowRenders, gpuMs,
      drawCalls: renderer.info.render.drawCalls, triangles: renderer.info.render.triangles }),
  };
}

main().catch((e) => {
  const msg = `error: ${e?.message ?? e}`;
  hud.textContent = msg + '\n(WebGL/WebGPU may be disabled in this browser)';
  hud.classList.add('error');
  telemetry.error('main', String(e?.stack ?? e));
  console.error(e);
});
