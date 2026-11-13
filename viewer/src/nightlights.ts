// City lights after dark: street lamps (real positions, pipeline lights.py) light pools on the ground
// via a 1 m/texel light map sampled in the voxel shader, plus a glowing billboard per lamp head.
// Lit windows and shop signs are emissive terms in voxelMaterial.ts driven by the uniforms here.
import * as THREE from 'three/webgpu';
import {
  cameraProjectionMatrix, exp, float, modelViewMatrix, positionGeometry, smoothstep, texture, uniform, vec2, vec3, vec4,
  attribute,
} from 'three/tsl';

/** 0 by day, 1 at night: lamps and windows fade in around sunset */
export const night = uniform(0);
/** fraction of window cells with the lights on (depends on the hour) */
export const windowsLit = uniform(0.3);
/** scene position of voxel (0,0,0) (the world group's offset) */
export const gridOrigin = uniform(new THREE.Vector3());
/** region size in metres (east, north) */
const gridSize = uniform(new THREE.Vector2(1, 1));
/** light map: r = lamp light on the ground (0..1), g = ground height of the nearest lamps (voxels / 255) */
const lampMap = texture(new THREE.DataTexture(new Uint8Array(4), 1, 1));
const LAMP_COLOR = new THREE.Color(1.0, 0.72, 0.42);   // warm white (linear)
/** L0 voxel size (m), for the ground heights in the light map */
export const baseVoxel = uniform(0.5);

/** street-lamp light (at full night strength, multiply by albedo) at a scene position; `up` 1 for
 *  upward-facing surfaces, less for walls */
export function lampLightAt(world: any, up: any) {
  const rel = world.sub(gridOrigin);                                       // metres from voxel (0,0,0)
  const lm = lampMap.sample(vec2(rel.x, rel.z.negate()).div(gridSize));
  const above = rel.y.sub(lm.g.mul(baseVoxel.mul(255)));                   // height above the lamps' ground
  const fall = float(1).sub(smoothstep(2.5, 9, above)).mul(smoothstep(-1, 0, above));
  return vec3(LAMP_COLOR.r, LAMP_COLOR.g, LAMP_COLOR.b).mul(lm.r.mul(fall).mul(up).mul(1.6));
}

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
  // light map: 2 m texels (a pool is 14 m in radius and filtered: 1 m looked the same at 4x the texels and
  // ~300 MB of temporary arrays for a whole commune); two channels (light, ground height)
  const T = 2, W = Math.ceil((size[0] * vs) / T), H = Math.ceil((size[1] * vs) / T);
  const light = new Float32Array(W * H), gsum = new Float32Array(W * H);
  const R = POOL / T;
  for (const [x, y, zg] of lamps) {
    const cx = (x * vs) / T, cy = (y * vs) / T;
    const j0 = Math.max(0, Math.floor(cy - R)), j1 = Math.min(H - 1, Math.ceil(cy + R));
    const i0 = Math.max(0, Math.floor(cx - R)), i1 = Math.min(W - 1, Math.ceil(cx + R));
    for (let j = j0; j <= j1; j++) {
      const dy = (j + 0.5 - cy) / R, row = j * W;
      for (let i = i0; i <= i1; i++) {
        const dx = (i + 0.5 - cx) / R, d2 = dx * dx + dy * dy;
        if (d2 >= 1) continue;
        const w = (1 - d2) * (1 - d2);
        light[row + i] += w;
        gsum[row + i] += w * zg;
      }
    }
  }
  const data = new Uint8Array(W * H * 2);
  for (let k = 0; k < W * H; k++) {
    const l = light[k];
    data[k * 2] = l >= 1 ? 255 : l * 255;
    data[k * 2 + 1] = l > 0 ? Math.round(gsum[k] / l) : 0;
  }
  const tex = new THREE.DataTexture(data, W, H, THREE.RGFormat, THREE.UnsignedByteType);
  tex.magFilter = tex.minFilter = THREE.LinearFilter;
  tex.unpackAlignment = 2;
  tex.needsUpdate = true;
  lampMap.value = tex;
  gridSize.value.set(W * T, H * T);   // metres covered

  // lamp heads: one camera-facing quad per lamp, additive glow
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, -0.5, 0.5, 0, 0.5, 0.5, 0], 3));
  g.setIndex([0, 1, 2, 2, 1, 3]);
  const centers = new Float32Array(lamps.length * 3);
  lamps.forEach(([x, y, , zh], i) => centers.set([(x + 0.5) * vs, (zh + 0.5) * vs, -(y + 0.5) * vs], i * 3));
  g.setAttribute('center', new THREE.InstancedBufferAttribute(centers, 3));
  g.instanceCount = lamps.length;
  // drawn in the additive glow layer (deferred.ts)
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
