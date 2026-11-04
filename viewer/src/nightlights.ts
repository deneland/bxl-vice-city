// City lights after dark: street lamps (real positions, pipeline lights.py) light pools on the ground
// via a 1 m/texel light map sampled in the voxel shader, plus a glowing billboard per lamp head.
// Lit windows and shop signs are emissive terms in voxelMaterial.ts driven by the uniforms here.
import * as THREE from 'three/webgpu';
import {
  cameraProjectionMatrix, exp, float, modelViewMatrix, positionGeometry, texture, uniform, vec3, vec4, attribute,
} from 'three/tsl';

/** 0 by day, 1 at night: lamps and windows fade in around sunset */
export const night = uniform(0);
/** fraction of window cells with the lights on (depends on the hour) */
export const windowsLit = uniform(0.3);
/** scene position of voxel (0,0,0) (the world group's offset) */
export const gridOrigin = uniform(new THREE.Vector3());
/** region size in metres (east, north) */
export const gridSize = uniform(new THREE.Vector2(1, 1));
/** light map: r = lamp light on the ground (0..1), g = ground height of the nearest lamps (voxels / 255) */
export const lampMap = texture(new THREE.DataTexture(new Uint8Array(4), 1, 1));
export const LAMP_COLOR = new THREE.Color(1.0, 0.72, 0.42);   // warm white (linear)

const POOL = 14;   // m; radius of a lamp's light on the ground

/** night factor from the sun altitude (degrees) */
export const nightFor = (deg: number) => { const t = Math.min(1, Math.max(0, (3 - deg) / 7)); return t * t * (3 - 2 * t); };

/** share of lit windows by local hour: evening peak, few at night, some in the early morning */
export function litShare(h: number) {
  const pts: [number, number][] = [[0, 0.12], [2, 0.05], [5, 0.05], [7, 0.3], [9, 0.12], [16, 0.15], [19, 0.45],
    [22, 0.35], [24, 0.12]];
  for (let i = 1; i < pts.length; i++)
    if (h <= pts[i][0]) { const [a, b] = [pts[i - 1], pts[i]]; return a[1] + (b[1] - a[1]) * (h - a[0]) / (b[0] - a[0]); }
  return 0.12;
}

/** Load the lamps, build the light map and the lamp-head billboards (added to `world`). */
export async function loadLamps(url: string, world: THREE.Object3D, vs: number, size: [number, number]) {
  const r = await fetch(url);
  if (!r.ok) return null;
  const lamps: [number, number, number, number][] = (await r.json()).lamps;
  const W = Math.round(size[0] * vs), H = Math.round(size[1] * vs);   // 1 m texels
  const light = new Float32Array(W * H), gsum = new Float32Array(W * H);
  for (const [x, y, zg] of lamps) {
    const cx = x * vs, cy = y * vs;
    for (let j = Math.max(0, Math.floor(cy - POOL)); j <= Math.min(H - 1, Math.ceil(cy + POOL)); j++)
      for (let i = Math.max(0, Math.floor(cx - POOL)); i <= Math.min(W - 1, Math.ceil(cx + POOL)); i++) {
        const d2 = ((i + 0.5 - cx) ** 2 + (j + 0.5 - cy) ** 2) / (POOL * POOL);
        if (d2 >= 1) continue;
        const w = (1 - d2) ** 2;
        light[j * W + i] += w;
        gsum[j * W + i] += w * zg;
      }
  }
  const data = new Uint8Array(W * H * 4);
  for (let k = 0; k < W * H; k++) {
    data[k * 4] = Math.min(255, light[k] * 255);
    data[k * 4 + 1] = light[k] > 0 ? Math.round(gsum[k] / light[k]) : 0;
  }
  const tex = new THREE.DataTexture(data, W, H);
  tex.magFilter = tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  lampMap.value = tex;
  gridSize.value.set(W, H);

  // lamp heads: one camera-facing quad per lamp, additive glow
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, -0.5, 0.5, 0, 0.5, 0.5, 0], 3));
  g.setIndex([0, 1, 2, 2, 1, 3]);
  const centers = new Float32Array(lamps.length * 3);
  lamps.forEach(([x, y, , zh], i) => centers.set([(x + 0.5) * vs, (zh + 0.5) * vs, -(y + 0.5) * vs], i * 3));
  g.setAttribute('center', new THREE.InstancedBufferAttribute(centers, 3));
  g.instanceCount = lamps.length;
  const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  const SIZE = 3;   // m
  const view = modelViewMatrix.mul(vec4(attribute('center', 'vec3'), 1));
  m.vertexNode = cameraProjectionMatrix.mul(view.add(vec4(positionGeometry.xy.mul(SIZE), 0, 0)));
  const r2 = positionGeometry.xy.mul(2).dot(positionGeometry.xy.mul(2));
  const glow = exp(r2.mul(-40)).mul(3).add(exp(r2.mul(-6)).mul(0.35));    // bright core + soft halo
  m.colorNode = vec4(vec3(LAMP_COLOR.r, LAMP_COLOR.g, LAMP_COLOR.b).mul(glow).mul(night), float(1));
  m.fog = false;
  const mesh = new THREE.Mesh(g, m);
  mesh.frustumCulled = false;
  mesh.renderOrder = 10;
  world.add(mesh);
  return mesh;
}
