import * as THREE from 'three/webgpu';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { TileManager, type LevelInfo } from './lod';
import { TELEMETRY, telemetry } from './telemetry';
import { LABELS, viewMode } from './voxelMaterial';
import { buildLanes, loadRoads, roadOverlay } from './roads';
import { Traffic } from './traffic';
import { baseVoxel, gridOrigin, lampLightAt, litShare, loadLamps, night, nightFor, windowsLit } from './nightlights';
import { Voxels } from './voxels';
import { Walk } from './walk';
import { Inspector } from './inspect';
import { lighting, sunDirection, sunPosition } from './sun';
import { applyWeather, describe, fetchWeather, Precipitation, presetWeather, type Weather } from './weather';
import { blurPx, DEPTH_RATIO, focalLength, focusDistance } from './tiltshift';
import { Deferred, DYNAMIC_SHADOW_LAYER } from './deferred';

telemetry.start();

interface Manifest {
  version: number; build?: string; name: string; origin: [number, number, number]; tileSize: number;
  home: [number, number]; levels: LevelInfo[]; attribution: string;
}
type Vec3 = [number, number, number];
export interface View { pos: Vec3; target: Vec3 }

const params = new URLSearchParams(location.search);
const ROI = params.get('roi') ?? 'schaerbeek';
const SHADOWS = params.get('shadows') !== '0';   // ?shadows=0 to turn off
// render every animation frame (benchmarks); default: only when something changed
const CONTINUOUS = params.get('continuous') === '1';
const GPU_TIMING = params.get('gpu') !== '0' && (TELEMETRY || params.get('gpu') === '1');
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
  const manifest: Manifest = await (await fetch(`tiles/${ROI}/manifest.json`, { cache: 'no-store' })).json();
  const L0 = manifest.levels[0];
  const VS = L0.voxelSize, CH = manifest.tileSize, NZ = L0.size[2];
  document.getElementById('attr')!.textContent = manifest.attribution;

  // scene frame: origin at the home point, ground ≈ y 0
  const homeVX = (manifest.home[0] - manifest.origin[0]) / VS;
  const homeVY = (manifest.home[1] - manifest.origin[1]) / VS;
  const groundGuess = 20; // voxels above z0, refined once chunks arrive

  const renderer = new THREE.WebGPURenderer({ antialias: false, trackTimestamp: GPU_TIMING });  // FXAA in deferred.ts
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  document.body.appendChild(renderer.domElement);
  await renderer.init();
  const backend = (renderer.backend as any).isWebGPUBackend ? 'WebGPU' : 'WebGL2';
  telemetry.setEnv({ backend, gpu: await gpuName(renderer), shadows: SHADOWS, gpuTiming: GPU_TIMING,
    pixelRatio: renderer.getPixelRatio() });

  renderer.info.autoReset = false;   // several render calls per frame; reset once per frame below
  // two scenes: the static city (cached G-buffer) and the forward layer (lights, fog, moving and
  // transparent things); see deferred.ts
  const cityScene = new THREE.Scene();
  const scene = new THREE.Scene();
  const sky = SKY.clone();
  scene.fog = new THREE.Fog(SKY.clone(), 700, 2600);
  // shadows (on by default, ?shadows=0 off): our own two-layer shadow maps in deferred.ts; the box covers the region from any sun direction
  const deferred = new Deferred(renderer, SHADOWS ? { half: (Math.hypot(L0.size[0], L0.size[1]) * VS) / 2 + 60 } : undefined);
  let cityDirty = true; // the city G-buffer must be redrawn (view mode, size)

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
  controls.target.set(0, 0, 0);
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
  const cam = params.get('cam')?.split(',').map(Number);
  if (cam?.length === 6 && cam.every(Number.isFinite)) setView({ pos: cam.slice(0, 3) as Vec3, target: cam.slice(3) as Vec3 });

  const hemi = new THREE.HemisphereLight(0xdfeeff, 0x5a5040, 1.6);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xfff1dc, 3.2);
  scene.add(sun, sun.target);

  const world = new THREE.Group();
  // put the home tile's lowest surface at y≈0 (known from the manifest, before loading)
  const homeTile = L0.tiles.find((t) => t.tx === Math.floor(homeVX / CH) && t.ty === Math.floor(homeVY / CH));
  world.position.set(-homeVX * VS, -(homeTile?.zmin ?? groundGuess) * VS, homeVY * VS);
  cityScene.add(world);
  // same frame for the forward layer (lamps, roads overlay, traffic)
  const fx = new THREE.Group();
  fx.position.copy(world.position);
  scene.add(fx);
  const glowScene = new THREE.Scene();   // additive light: lamp glows
  const glowFx = new THREE.Group();
  glowFx.position.copy(world.position);
  glowScene.add(glowFx);
  // region centre in the scene: sun target, so the shadow box covers everything
  world.updateMatrixWorld();
  sun.target.position.set((L0.size[0] / 2) * VS + world.position.x, 0, -(L0.size[1] / 2) * VS + world.position.z);
  gridOrigin.value.copy(world.position);
  baseVoxel.value = VS;
  let lamps: THREE.Object3D | null = null;
  loadLamps(`lights/${ROI}.json`, glowFx, VS, [L0.size[0], L0.size[1]]).then((m) => {
    lamps = m;
    if (m) m.visible = night.value > 0.01;
    dirty = true;
  }).catch((e) => console.warn('street lights unavailable', e));
  // movement network (OSM): debug overlay (R) and traffic (T)
  const network = loadRoads(`roads/${ROI}.json`).then((net) => net && { net, lanes: buildLanes(net, VS) })
    .catch((e) => { console.warn('road network unavailable', e); return null; });
  let roads: THREE.Object3D | null = null;
  const setRoads = async (on: boolean) => {
    if (on && !roads) {
      const n = await network;
      if (!n) return;
      roads = roadOverlay(n.net, n.lanes, VS);
      fx.add(roads);
    }
    if (roads) roads.visible = on;
    dirty = true;
  };
  if (params.get('roads') === '1') setRoads(true);
  let traffic: Traffic | null = null;
  let trafficOn = false;
  const setTraffic = async (on: boolean) => {
    trafficOn = on;
    if (on && !traffic) {
      const n = await network;
      if (!n) { trafficOn = false; return; }
      traffic = new Traffic(n.lanes, { car: Number(params.get('cars') ?? 400), tram: 12, train: 3 },
        deferred.litMaterial((w) => lampLightAt(w, 0.8).mul(night)),   // lit like the city, lamps at night
        (node) => n.net.nodes[node]?.[4] === 1);                        // region edge: through traffic
      traffic.mesh.layers.enable(DYNAMIC_SHADOW_LAYER);
      fx.add(traffic.mesh, traffic.lights);
    }
    if (traffic) traffic.mesh.visible = traffic.lights.visible = trafficOn;
    dynShadowDirty = true;   // redraw (or clear) the vehicle shadows once
    dirty = true;
  };
  if (params.get('traffic') !== '0') setTraffic(true);

  // --- time of day + weather: live over Schaerbeek, or chosen with the slider ---------
  const timeInput = document.getElementById('time') as HTMLInputElement;
  const timeLabel = document.getElementById('time-label')!;
  const liveBtn = document.getElementById('live') as HTMLButtonElement;
  const brusselsNow = () => {  // [YYYY-MM-DD, fractional hours] on the Brussels clock
    const p = Object.fromEntries(new Intl.DateTimeFormat('en-CA', { timeZone: 'Europe/Brussels', hourCycle: 'h23',
      year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).formatToParts(new Date())
      .map((x) => [x.type, x.value]));
    return [`${p.year}-${p.month}-${p.day}`, Number(p.hour) + Number(p.minute) / 60] as const;
  };
  let day = params.get('date') ?? brusselsNow()[0];
  let live = !params.has('time');
  let weather: Weather | null = null;   // null: unknown (not loaded or service down) -> clear-sky look
  const precip = new Precipitation();
  scene.add(precip.mesh);
  let precipOn = false;
  const brusselsOffset = (d: Date) => {  // minutes ahead of UTC (CET/CEST)
    const tz = new Intl.DateTimeFormat('en', { timeZone: 'Europe/Brussels', timeZoneName: 'shortOffset' })
      .formatToParts(d).find((p) => p.type === 'timeZoneName')?.value ?? 'GMT+1';
    const m = tz.match(/GMT([+-]\d+)(?::(\d+))?/);
    return m ? Number(m[1]) * 60 + Math.sign(Number(m[1])) * Number(m[2] ?? 0) : 60;
  };
  const setTime = (hours: number) => {
    const [y, mo, d] = day.split('-').map(Number);
    let t = new Date(Date.UTC(y, mo - 1, d) + hours * 3600e3);
    t = new Date(t.getTime() - brusselsOffset(t) * 60e3);
    const { az, alt } = sunPosition(t, LAT, LON);
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
    const hh = Math.floor(hours), mm = Math.round((hours - hh) * 60);
    timeLabel.textContent = `${live ? 'live · ' : ''}${day} ${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}` +
      ` · ${weather ? `${describe(weather)}, ${Math.round(weather.temp)}°C` : 'weather unavailable'} · sun ${(alt * 180 / Math.PI).toFixed(0)}°`;
    liveBtn.style.display = live ? 'none' : '';
    dirty = true;
  };
  const tick = () => { // live: follow the clock
    if (!live) return;
    const [d, h] = brusselsNow();
    day = d;
    timeInput.value = String(h);
    setTime(h);
  };
  timeInput.value = params.get('time') ?? String(brusselsNow()[1]);
  timeInput.addEventListener('input', () => { live = false; setTime(Number(timeInput.value)); });
  liveBtn.addEventListener('click', () => { live = true; tick(); });
  setInterval(tick, 30e3);
  const wPreset = params.get('weather');
  const updateWeather = async () => {
    try {
      weather = wPreset ? presetWeather(wPreset) : await fetchWeather(LAT, LON);
      setTime(Number(timeInput.value));
      telemetry.event('weather', { ...weather! });
    } catch (e) { console.warn('weather unavailable', e); }
  };
  if (!wPreset) setInterval(updateWeather, 10 * 60e3);

  // --- level-of-detail tiles ------------------------------------------------
  const ERR = Number(params.get('err') ?? 16);          // max on-screen size of one voxel, px
  const lod = new TileManager(`tiles/${ROI}`, CH, manifest.levels, world, SHADOWS, params.get('lod') === '0' ? 0 : ERR);
  lod.query = manifest.build ? `?v=${manifest.build}` : '';
  lod.budgetFaces = Number(params.get('budget') ?? 16) * 1e6;  // resident faces, millions
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

  // --- CPU voxel access, walking, picking ---------------------------------
  const vox = new Voxels(`tiles/${ROI}`, CH, L0, world, lod.query);
  const crosshair = document.getElementById('crosshair')!;
  const walkHint = document.getElementById('walk-hint')!;
  let mapView: View | null = null;
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
      mapView = null;
      telemetry.event('walk', { on: false });
    } else {
      mapView = getView();
      walk.enter(controls.target.clone()).then(() => telemetry.event('walk', { on: walk.active }));
    }
  };
  const inspector = new Inspector(vox, scene, manifest.origin[2], () => { dirty = true; });
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
    // wait briefly: a double-click sets the orbit point instead
    clearTimeout(clickTimer);
    clickTimer = setTimeout(() => {
      raycaster.setFromCamera(ndc, camera);
      inspector.pick(raycaster.ray).then(() => telemetry.event('inspect', {}));
    }, walk.active ? 0 : 250);
  });
  let clickTimer = 0;

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
    telemetry.event('view-mode', { mode: labels ? 'classes' : 'colour' });
  };
  setMode(params.get('mode') === 'labels');

  // --- input ---------------------------------------------------------------
  let snapshotRequested = false;
  window.addEventListener('keydown', (e: KeyboardEvent) => {
    if ((e.key === 'h' || e.key === 'H') && !walk.active) setView(home);
    if (e.key === 'l' || e.key === 'L') setMode(viewMode.value === 0);
    if ((e.key === 'p' || e.key === 'P') && TELEMETRY) snapshotRequested = true;
    if (e.key === 'v' || e.key === 'V') toggleWalk();
    if (e.key === 'm' || e.key === 'M') setTilt(!tiltOn);
    if (e.key === 'r' || e.key === 'R') setRoads(!roads?.visible);
    if (e.key === 't' || e.key === 'T') setTraffic(!trafficOn);
    if (e.key === 'Escape') inspector.clear();
    dirty = true;
  });
  let dirty = true; // something outside camera/tiles changed: draw the next tick
  setTime(Number(timeInput.value));
  updateWeather();
  setTilt(params.get('tilt') === '1');
  window.addEventListener('resize', () => {
    dirty = true;
    cityDirty = true;
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
    const s = renderer.getDrawingBufferSize(new THREE.Vector2());
    deferred.setSize(s.x, s.y);
  });

  telemetry.setState(() => ({
    view: getView(), lod: lod.stats, errPx: lod.errPx,
    screen: { w: innerWidth, h: innerHeight, dpr: devicePixelRatio },
  }));

  // --- loop + HUD ----------------------------------------------------------
  const frameTimes: number[] = [];
  const frameHooks = new Set<(f: { interval: number; cpu: number; moving: boolean; drawCalls: number; triangles: number }) => void>();
  const gpuHooks = new Set<(ms: number) => void>();
  const lastCam = new THREE.Matrix4();
  let last = performance.now(), hudAt = 0, gpuMs = 0, gpuInFlight = false;
  let flashUntil = 0;
  let lastTickRendered = false, rendered = 0, skipped = 0, cityRenders = 0, shadowRenders = 0;
  const shadowSun = new THREE.Vector3();   // sun direction of the current city shadow map
  let shadowTiles = true, dynShadowDirty = true;
  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const interval = now - last;
    last = now;

    if (targetAnim) {
      const k = Math.min(1, (now - targetAnim.t0) / 400), e = k * k * (3 - 2 * k);
      controls.target.lerpVectors(targetAnim.from, targetAnim.to, e);
      if (k >= 1) targetAnim = null;
    }
    if (walk.active) walk.update(interval / 1000);
    else controls.update();
    if (trafficOn && traffic) traffic.update(Math.min(interval, 100) / 1000);
    camera.updateMatrixWorld();
    // damping keeps nudging the camera by invisible amounts forever; ignore changes below 1e-4
    const moving = !matricesClose(lastCam, camera.matrixWorld, 1e-4);
    if (moving) lastCam.copy(camera.matrixWorld);
    if (tiltOn) {
      // focus on the orbit target (map) or 20 m ahead (walking), measured along the view direction
      const d = walk.active ? 20 : Math.max(5, controls.target.clone().sub(camera.position).dot(camera.getWorldDirection(new THREE.Vector3())));
      focusDistance.value = d;
      focalLength.value = d * DEPTH_RATIO;
    }
    const tilesChanged = moving || dirty || lod.needsUpdate ? lod.update(camera, renderer.domElement.height) : false;
    const cityChanged = CONTINUOUS || moving || tilesChanged || cityDirty;
    if (!(cityChanged || precipOn || trafficOn || dirty || snapshotRequested)) {
      // nothing changed: draw nothing, the GPU stays idle
      lastTickRendered = false;
      skipped++;
      telemetry.idleTick();
      updateHud(now);
      return;
    }
    dirty = false;
    renderer.info.reset();
    if (cityChanged) {   // the expensive part: only when the city on screen changed
      deferred.renderCity(cityScene, camera);
      cityDirty = false;
      cityRenders++;
    }
    deferred.sync(camera, sun, hemi, scene.fog as THREE.Fog, sky);
    if (deferred.shadows) {
      // city shadows: when the sun moved noticeably, or other tiles were loaded (once loading settles)
      if (tilesChanged) shadowTiles = true;
      const sunMoved = deferred.sunDir.value.angleTo(shadowSun) > 0.005;
      if (sunMoved || (shadowTiles && !moving && !lod.stats.loading)) {
        deferred.renderStaticShadow(cityScene, sun.target.position);
        shadowSun.copy(deferred.sunDir.value);
        shadowTiles = false;
        shadowRenders++;
      }
      if (trafficOn || dynShadowDirty) {   // vehicles, around what you look at
        deferred.renderDynamicShadow(scene, walk.active ? camera.position : controls.target);
        dynShadowDirty = trafficOn;
      }
    }
    deferred.present(scene, glowScene, camera);
    rendered++;
    const cpu = performance.now() - now;
    // a frame interval only means something if the previous tick also rendered
    if (lastTickRendered) {
      frameTimes.push(interval);
      if (frameTimes.length > 240) frameTimes.shift();
      const frame = { interval, cpu, moving, drawCalls: renderer.info.render.drawCalls,
        triangles: renderer.info.render.triangles };
      telemetry.frame(frame);
      for (const h of frameHooks) h(frame);
    }
    lastTickRendered = true;

    if (GPU_TIMING && !gpuInFlight) {
      gpuInFlight = true;
      renderer.resolveTimestampsAsync(THREE.TimestampQuery.RENDER).then((ms) => {
        if (typeof ms === 'number' && ms > 0) {
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
      `tiles ${lod.stats.perLevel.map((n, l) => `L${l}:${n}`).join(' ')} · ${(lod.stats.faces / 1e6).toFixed(2)}M faces · ` +
      `${renderer.info.render.drawCalls} draws · ${lod.stats.loading ? `loading ${lod.stats.loading}` : `${lod.stats.ready}/${lod.stats.total} tiles in memory`}` +
      ` (${(lod.stats.residentFaces / 1e6).toFixed(1)}M faces)` +
      `${loadMs ? ` · first view ${(loadMs / 1000).toFixed(1)} s` : ''} · err ${lod.errPx}px` +
      ` · shadows ${SHADOWS ? 'on' : 'off'}${tiltOn ? ' · tilt-shift' : ''}${walk.active ? ' · walking' : ''}` +
      (TELEMETRY ? ' · telemetry on' : '') + (now < flashUntil ? '\nsnapshot sent ✓' : '');
  }

  // test hook for the benchmark harness and debugging
  (window as any).__viewer = {
    renderer, scene, cityScene, deferred, camera, controls, ready, setView, getView, home, frameHooks, gpuHooks, lod, vox, walk, inspector,
    setTime, setTilt, toggleWalk, setTraffic, get traffic() { return traffic; },
    whenIdle: () => lod.whenIdle(),
    info: () => ({ backend, ...lod.stats, loadMs, rendered, cityRenders, shadowRenders, gpuMs,
      drawCalls: renderer.info.render.drawCalls, triangles: renderer.info.render.triangles }),
  };
}

main().catch((e) => {
  const msg = `error: ${e?.message ?? e}`;
  hud.textContent = msg + '\n(WebGL/WebGPU may be disabled in this browser)';
  telemetry.error('main', String(e?.stack ?? e));
  console.error(e);
});
