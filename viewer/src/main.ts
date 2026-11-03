import * as THREE from 'three/webgpu';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { TileManager, type LevelInfo } from './lod';
import { TELEMETRY, telemetry } from './telemetry';
import { LABELS, viewMode } from './voxelMaterial';

telemetry.start();

interface Manifest {
  version: number; build?: string; name: string; origin: [number, number, number]; tileSize: number;
  home: [number, number]; levels: LevelInfo[]; attribution: string;
}
type Vec3 = [number, number, number];
export interface View { pos: Vec3; target: Vec3 }

const params = new URLSearchParams(location.search);
const ROI = params.get('roi') ?? 'schaerbeek';
const SHADOWS = params.get('shadows') === '1';
// render every animation frame (benchmarks); default: only when something changed
const CONTINUOUS = params.get('continuous') === '1';
const GPU_TIMING = params.get('gpu') !== '0' && (TELEMETRY || params.get('gpu') === '1');
const SKY = new THREE.Color(0xbcd3e6);

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

  const renderer = new THREE.WebGPURenderer({ antialias: true, trackTimestamp: GPU_TIMING });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = SHADOWS;
  document.body.appendChild(renderer.domElement);
  await renderer.init();
  const backend = (renderer.backend as any).isWebGPUBackend ? 'WebGPU' : 'WebGL2';
  telemetry.setEnv({ backend, gpu: await gpuName(renderer), shadows: SHADOWS, gpuTiming: GPU_TIMING,
    pixelRatio: renderer.getPixelRatio() });

  const scene = new THREE.Scene();
  scene.background = SKY;
  scene.fog = new THREE.Fog(SKY, 700, 2600);

  const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.5, 6000);
  camera.position.set(-110, 120, 150);

  const controls = new MapControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 4;
  controls.maxDistance = 2200;
  controls.screenSpacePanning = false;
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

  scene.add(new THREE.HemisphereLight(0xdfeeff, 0x5a5040, 1.6));
  const sun = new THREE.DirectionalLight(0xfff1dc, 3.2);
  sun.position.set(-180, 260, 120); // from the south-west-ish, afternoon
  if (SHADOWS) {
    sun.castShadow = true;
    const half = (L0.size[0] * VS) / 2 + 60;
    Object.assign(sun.shadow.camera, { left: -half, right: half, top: half, bottom: -half, near: 1, far: 1200 });
    sun.shadow.mapSize.set(4096, 4096);
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.3;
  }
  scene.add(sun, sun.target);

  const world = new THREE.Group();
  // put the home tile's lowest surface at y≈0 (known from the manifest, before loading)
  const homeTile = L0.tiles.find((t) => t.tx === Math.floor(homeVX / CH) && t.ty === Math.floor(homeVY / CH));
  world.position.set(-homeVX * VS, -(homeTile?.zmin ?? groundGuess) * VS, homeVY * VS);
  scene.add(world);

  // --- level-of-detail tiles ------------------------------------------------
  const ERR = Number(params.get('err') ?? 8);           // max on-screen size of one voxel, px
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

  // --- view mode: colours or semantic classes ----------------
  const legend = document.getElementById('legend')!;
  legend.innerHTML = '<b>classes</b> (LiDAR + street-level photos)<br>' + LABELS.map(([, name, h]) =>
    `<span style="background:#${h.toString(16).padStart(6, '0')}"></span>${name}`).join('<br>');
  const setMode = (labels: boolean) => {
    viewMode.value = labels ? 1 : 0;
    legend.style.display = labels ? 'block' : 'none';
    telemetry.event('view-mode', { mode: labels ? 'classes' : 'colour' });
  };
  setMode(params.get('mode') === 'labels');

  // --- input ---------------------------------------------------------------
  let snapshotRequested = false;
  window.addEventListener('keydown', (e: KeyboardEvent) => {
    if (e.key === 'h' || e.key === 'H') setView(home);
    if (e.key === 'l' || e.key === 'L') setMode(viewMode.value === 0);
    if ((e.key === 'p' || e.key === 'P') && TELEMETRY) snapshotRequested = true;
    dirty = true;
  });
  let dirty = true; // something outside camera/tiles changed: draw the next tick
  window.addEventListener('resize', () => {
    dirty = true;
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
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
  let lastTickRendered = false, rendered = 0, skipped = 0;
  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const interval = now - last;
    last = now;

    controls.update();
    camera.updateMatrixWorld();
    // damping keeps nudging the camera by invisible amounts forever; ignore changes below 1e-4
    const moving = !matricesClose(lastCam, camera.matrixWorld, 1e-4);
    if (moving) lastCam.copy(camera.matrixWorld);
    const tilesChanged = moving || dirty || lod.needsUpdate ? lod.update(camera, renderer.domElement.height) : false;
    if (!(CONTINUOUS || moving || tilesChanged || dirty || snapshotRequested)) {
      // nothing changed: draw nothing, the GPU stays idle
      lastTickRendered = false;
      skipped++;
      telemetry.idleTick();
      updateHud(now);
      return;
    }
    dirty = false;
    renderer.render(scene, camera);
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
      ` · ${rendered} frames drawn, ${skipped} skipped` +
      (GPU_TIMING ? ` · gpu ${gpuMs.toFixed(2)} ms` : '') + '\n' +
      `tiles ${lod.stats.perLevel.map((n, l) => `L${l}:${n}`).join(' ')} · ${(lod.stats.faces / 1e6).toFixed(2)}M faces · ` +
      `${renderer.info.render.drawCalls} draws · ${lod.stats.loading ? `loading ${lod.stats.loading}` : `${lod.stats.ready}/${lod.stats.total} tiles in memory`}` +
      ` (${(lod.stats.residentFaces / 1e6).toFixed(1)}M faces)` +
      `${loadMs ? ` · first view ${(loadMs / 1000).toFixed(1)} s` : ''} · err ${lod.errPx}px` +
      ` · shadows ${SHADOWS ? 'on' : 'off'}` +
      (TELEMETRY ? ' · telemetry on' : '') + (now < flashUntil ? '\nsnapshot sent ✓' : '');
  }

  // test hook for the benchmark harness and debugging
  (window as any).__viewer = {
    renderer, scene, camera, controls, ready, setView, getView, home, frameHooks, gpuHooks, lod,
    whenIdle: () => lod.whenIdle(),
    info: () => ({ backend, ...lod.stats, loadMs,
      drawCalls: renderer.info.render.drawCalls, triangles: renderer.info.render.triangles }),
  };
}

main().catch((e) => {
  const msg = `error: ${e?.message ?? e}`;
  hud.textContent = msg + '\n(WebGL/WebGPU may be disabled in this browser)';
  telemetry.error('main', String(e?.stack ?? e));
  console.error(e);
});
