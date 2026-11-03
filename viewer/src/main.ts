import * as THREE from 'three/webgpu';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { createVoxelMaterial, voxelGeometryBase } from './voxelMaterial';
import type { MeshRequest, MeshResult } from './mesher.worker';

interface ChunkInfo { file: string; cx: number; cy: number; count: number; zmin: number; zmax: number }
interface Manifest {
  name: string; origin: [number, number, number]; voxelSize: number;
  size: [number, number, number]; chunkSize: number; home: [number, number];
  chunks: ChunkInfo[]; attribution: string;
}

const params = new URLSearchParams(location.search);
const ROI = params.get('roi') ?? 'small';
const SKY = new THREE.Color(0xbcd3e6);

const hud = document.getElementById('hud')!;
document.getElementById('attr')!.textContent = 'loading…';

async function main() {
  const manifest: Manifest = await (await fetch(`tiles/${ROI}/manifest.json`)).json();
  const VS = manifest.voxelSize, CH = manifest.chunkSize, NZ = manifest.size[2];
  document.getElementById('attr')!.textContent = manifest.attribution;

  // scene frame: origin at the home point, ground ≈ y 0
  const homeVX = (manifest.home[0] - manifest.origin[0]) / VS;
  const homeVY = (manifest.home[1] - manifest.origin[1]) / VS;
  const groundGuess = 20; // voxels above z0, refined once chunks arrive

  const renderer = new THREE.WebGPURenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = params.get('shadows') !== '0';
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  document.body.appendChild(renderer.domElement);
  await renderer.init();
  const backend = (renderer.backend as any).isWebGPUBackend ? 'WebGPU' : 'WebGL2';

  const scene = new THREE.Scene();
  scene.background = SKY;
  scene.fog = new THREE.Fog(SKY, 350, 1300);

  const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.5, 4000);
  camera.position.set(-110, 120, 150);

  const controls = new MapControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 4;
  controls.maxDistance = 900;
  controls.screenSpacePanning = false;
  controls.listenToKeyEvents(window);
  controls.keyPanSpeed = 25;
  controls.target.set(0, 0, 0);
  const home = { pos: camera.position.clone(), target: controls.target.clone() };

  scene.add(new THREE.HemisphereLight(0xdfeeff, 0x5a5040, 1.6));
  const sun = new THREE.DirectionalLight(0xfff1dc, 3.2);
  sun.position.set(-180, 260, 120); // from the south-west-ish, afternoon
  sun.castShadow = renderer.shadowMap.enabled;
  const half = (manifest.size[0] * VS) / 2 + 60;
  Object.assign(sun.shadow.camera, { left: -half, right: half, top: half, bottom: -half, near: 1, far: 1200 });
  sun.shadow.mapSize.set(4096, 4096);
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.3;
  scene.add(sun, sun.target);

  const material = createVoxelMaterial(VS);
  const world = new THREE.Group();
  world.position.set(-homeVX * VS, -groundGuess * VS, homeVY * VS);
  scene.add(world);

  // --- worker pool -------------------------------------------------------
  const nWorkers = Math.max(2, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));
  const workers = Array.from({ length: nWorkers }, () =>
    new Worker(new URL('./mesher.worker.ts', import.meta.url), { type: 'module' }));
  const pending = new Map<number, ChunkInfo>();
  let totalFaces = 0, loaded = 0, groundSet = false;
  const t0 = performance.now();
  let loadMs = 0;

  const onMesh = (ev: MessageEvent<MeshResult>) => {
    const { id, a, c, faces } = ev.data;
    const info = pending.get(id)!;
    pending.delete(id);
    loaded++;
    if (loaded === manifest.chunks.length) loadMs = performance.now() - t0;
    if (!faces) return;
    const g = new THREE.InstancedBufferGeometry();
    g.index = voxelGeometryBase.index;
    g.setAttribute('position', voxelGeometryBase.getAttribute('position'));
    g.setAttribute('packed', new THREE.InstancedBufferAttribute(a, 2));
    g.setAttribute('rgba', new THREE.InstancedBufferAttribute(c, 4, true));
    g.instanceCount = faces;
    // bounds in chunk-local three coords (positionNode is invisible to three)
    const s = CH * VS;
    g.boundingBox = new THREE.Box3(
      new THREE.Vector3(0, info.zmin * VS, -s), new THREE.Vector3(s, (info.zmax + 1) * VS, 0));
    g.boundingSphere = g.boundingBox.getBoundingSphere(new THREE.Sphere());
    const mesh = new THREE.Mesh(g, material);
    mesh.position.set(info.cx * s, 0, -info.cy * s);
    mesh.castShadow = mesh.receiveShadow = renderer.shadowMap.enabled;
    mesh.matrixAutoUpdate = false;
    mesh.updateMatrix();
    world.add(mesh);
    totalFaces += faces;
    if (!groundSet && info.cx === Math.floor(homeVX / CH) && info.cy === Math.floor(homeVY / CH)) {
      // put the home chunk's lowest surface at y≈0
      world.position.y = -info.zmin * VS;
      groundSet = true;
    }
  };
  workers.forEach((w) => (w.onmessage = onMesh));

  // load nearest chunks first
  const order = [...manifest.chunks].sort((p, q) =>
    Math.hypot((p.cx + 0.5) * CH - homeVX, (p.cy + 0.5) * CH - homeVY) -
    Math.hypot((q.cx + 0.5) * CH - homeVX, (q.cy + 0.5) * CH - homeVY));
  let next = 0;
  order.forEach((info, i) => {
    fetch(`tiles/${ROI}/${info.file}`).then((r) => r.arrayBuffer()).then((buf) => {
      pending.set(i, info);
      const req: MeshRequest = { id: i, buf, ch: CH, nz: NZ };
      workers[next++ % nWorkers].postMessage(req, [buf]);
    });
  });

  // --- input ---------------------------------------------------------------
  window.addEventListener('keydown', (e: KeyboardEvent) => {
    if (e.key === 'h' || e.key === 'H') {
      camera.position.copy(home.pos);
      controls.target.copy(home.target);
    }
  });
  window.addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  // --- loop + HUD ----------------------------------------------------------
  const frameTimes: number[] = [];
  let last = performance.now(), hudAt = 0;
  renderer.setAnimationLoop(() => {
    const now = performance.now();
    frameTimes.push(now - last);
    if (frameTimes.length > 240) frameTimes.shift();
    last = now;
    controls.update();
    renderer.render(scene, camera);
    if (now - hudAt > 500) {
      hudAt = now;
      const sorted = [...frameTimes].sort((x, y) => x - y);
      const avg = sorted.reduce((s, v) => s + v, 0) / sorted.length;
      const p99 = sorted[Math.floor(sorted.length * 0.99)] ?? 0;
      hud.textContent =
        `${backend} · ${(1000 / avg).toFixed(0)} fps · avg ${avg.toFixed(1)} ms · p99 ${p99.toFixed(1)} ms\n` +
        `chunks ${loaded}/${manifest.chunks.length}${loadMs ? ` in ${(loadMs / 1000).toFixed(1)} s` : ''}` +
        ` · ${(totalFaces / 1e6).toFixed(2)}M faces · shadows ${renderer.shadowMap.enabled ? 'on' : 'off'}`;
    }
  });
  (window as any).__viewer = { renderer, scene, camera, controls, stats: () => ({ loaded, totalFaces, frameTimes }) };
}

main().catch((e) => {
  hud.textContent = `error: ${e?.message ?? e}`;
  console.error(e);
});
