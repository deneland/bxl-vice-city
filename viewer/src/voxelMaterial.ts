// One material per level for all tiles. Each instance is a rectangle of voxel faces (mesher.worker.ts); the
// vertex shader expands a unit quad over it, the fragment shader reads each face's colour from the face atlas.
import * as THREE from 'three/webgpu';
import {
  attribute, float, floor, hash, int, ivec2, mod, mix, mrt, positionGeometry, positionWorld, pow, select, smoothstep, textureLoad,
  uniform, varying, vec3, vec4,
} from 'three/tsl';
import { gridOrigin, lampLightAt } from './nightlights';

// semantic class per voxel (pipeline voxelize.py SEM_*: LiDAR class refined by street-level labels)
export const LABELS: [number, string, number][] = [
  [1, 'building', 0xd9c7a7], [2, 'shop sign', 0xff3d9a], [3, 'vegetation', 0x3fae49], [4, 'vehicle', 0x2f6bff],
  [5, 'bike', 0x00d5ff], [6, 'pole', 0xffd400], [7, 'traffic sign', 0xff2a2a], [8, 'fence', 0xa65cff],
  [9, 'ground', 0x6b6b6b], [10, 'person', 0xff8a00], [11, 'street furniture', 0x9cff3c], [13, 'water', 0x1f8fd6],
  [14, 'rail', 0x7a4a2a], [15, 'other', 0xb8b8b8],
];

/** 0: photo/LiDAR colours, 1: semantic classes */
export const viewMode = uniform(0);

export const voxelGeometryBase = (() => {
  // unit quad corners (q.x, q.y); triangles (0,1,2) (2,1,3) are CCW for u×v
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0], 3));
  g.setIndex([0, 1, 2, 2, 1, 3]);
  return g;
})();

// bit unpack for integers stored exactly in float32
const bits = (v: any, shift: number, count: number) => mod(floor(v.div(2 ** shift)), 2 ** count);

const axisVec = (i: any) =>
  vec3(select(i.equal(0), 1, 0), select(i.equal(1), 1, 0), select(i.equal(2), 1, 0));

export function createVoxelMaterial(voxelSize: number, baseSize: number, atlas: THREE.Texture) {
  // unlit: writes the G-buffer (deferred.ts lights it); `output` is the albedo
  const m = new THREE.MeshBasicNodeMaterial({ fog: false });

  // one instance per merged run of faces (mesher.worker.ts): first face, direction, occlusion, length,
  // and where its faces' colours start in the face atlas (lod.ts)
  const quad = attribute('quad', 'vec3');
  const q = positionGeometry;

  const p0 = quad.x;
  const dir = bits(p0, 20, 3);
  const axis = floor(dir.div(2));
  const positive = float(1).sub(mod(dir, 2));
  const lu = bits(quad.y, 0, 8), lv = bits(quad.y, 8, 8);   // rectangle: faces along the run axis, across
  // negative faces swap u/v to flip winding so the front side faces outwards
  const a = select(positive.equal(1), q.x, q.y);
  const b = select(positive.equal(1), q.y, q.x);
  // runs go up (z) for side faces, along x for top / bottom: that is the u axis (a) for axis 1 and 2,
  // the v axis (b) for axis 0
  const runOnA = select(axis.equal(0), float(0), float(1));
  const A = a.mul(mix(lv, lu, runOnA)), B = b.mul(mix(lu, lv, runOnA));
  const pv = vec3(bits(p0, 0, 6), bits(p0, 6, 6), bits(p0, 12, 8))
    .add(axisVec(axis).mul(positive))
    .add(axisVec(mod(axis.add(1), 3)).mul(A))
    .add(axisVec(mod(axis.add(2), 3)).mul(B));
  // voxel space (x east, y north, z up) -> three (x, y up, -z north)
  m.positionNode = vec3(pv.x, pv.z, pv.y.negate()).mul(voxelSize);

  // --- per pixel: which face of the run, its texel in the atlas, occlusion within the face ---
  const flat = <T,>(n: T, name: string) => (varying(n as any, name) as any).setInterpolation('flat');
  const vStart = flat(quad.z, 'vStart'), vLu = flat(lu, 'vLu'), vLv = flat(lv, 'vLv');
  const vRunOnA = flat(runOnA, 'vRunOnA');
  const run = varying(mix(B, A, runOnA), 'vRun');          // 0 .. lu along the run axis
  const cross = varying(mix(A, B, runOnA), 'vCross');      // 0 .. lv across
  const i = floor(run).min(vLu.sub(1)), jv = floor(cross).min(vLv.sub(1));
  const idx = vStart.add(jv.mul(vLu)).add(i);               // texels row by row
  const W = 2048;   // (FaceAtlas.W)
  const texel = textureLoad(atlas, ivec2(int(mod(idx, W)), int(floor(idx.div(W)))));
  // unpack the face's 32 bits (mesher.worker.ts): colour 6+6+6, occlusion 8, label 4, photo 1
  const by = (c: any) => floor(c.mul(255).add(0.5));
  const b0 = by(texel.r), b1 = by(texel.g), b2 = by(texel.b), b3 = by(texel.a);
  const r6 = mod(b0, 64), g6 = floor(b0.div(64)).add(mod(b1, 16).mul(4)), b6 = floor(b1.div(16)).add(mod(b2, 4).mul(16));
  const vAo = floor(b2.div(4)).add(mod(b3, 4).mul(64));
  const label = mod(floor(b3.div(4)), 16), photo = mod(floor(b3.div(64)), 2);
  // ambient occlusion: the face's 4 corner levels, blended across it (corner (a, b) at bits 2(a + 2b))
  const lr = run.sub(i), lx = cross.sub(jv), la = mix(lx, lr, vRunOnA), lb = mix(lr, lx, vRunOnA);
  const corner = (ca: number, cb: number) => mod(floor(vAo.div(4 ** (ca + 2 * cb))), 4);
  const lvl = mix(mix(corner(0, 0), corner(1, 0), la), mix(corner(0, 1), corner(1, 1), la), lb);
  const ao = mix(float(0.42), float(1.0), lvl.div(3));

  // LiDAR colours are sRGB -> linear, with a light saturation lift
  const srgb = vec3(r6, g6, b6).div(63);
  const lum = srgb.dot(vec3(0.299, 0.587, 0.114));
  const col = pow(mix(vec3(lum), srgb, 1.25).clamp(0, 1), vec3(2.2));

  // class view: palette colour per semantic class
  const hex = (h: number) => { const c = new THREE.Color(h); return vec3(c.r, c.g, c.b); }; // sRGB hex -> linear
  let lc: any = hex(0xff00ff); // unknown code: magenta, should never show
  for (const [code, , h] of LABELS) lc = select(label.equal(code), hex(h), lc);

  m.colorNode = vec4(mix(col, lc, viewMode).mul(ao), 1);

  // --- night lights (see nightlights.ts) ---
  // street lamps: light map over the ground plan, fading out above lamp height
  const rel = positionWorld.sub(gridOrigin);                             // metres from voxel (0,0,0)
  const up = select(dir.equal(4), float(1), float(0.6));
  const lampLight = lampLightAt(positionWorld, varying(up, 'vUp'));
  // lit windows: dark (glass) photo-coloured facade voxels, per window cell (1.5 m wide, 3 m floor) on or off
  const vSide = flat(select(dir.lessThan(4), float(1), float(0)), 'vSide');
  const isFacade = select(label.equal(1).and(photo.equal(1)).and(vSide.equal(1)), 1, 0);
  const xAxis = varying(select(axis.equal(0), 1, 0), 'vXAxis');                // face normal along east-west
  const v = rel.div(baseSize).mul(vec3(1, 1, -1));                            // L0 voxels: x east, y up, z north
  const along = mix(v.x, v.z, xAxis), across = mix(v.z, v.x, xAxis);
  const cell = floor(along.div(3)).add(floor(v.y.sub(0.01).div(6)).mul(101)).add(floor(across.add(0.5).div(2)).mul(7919));
  const glass = float(1).sub(smoothstep(0.18, 0.32, lum));                    // photo colour is dark: window
  // shop signs glow
  const sign = select(label.equal(2), 1, 0);
  const colourView = float(1).sub(viewMode);
  // face normal in scene space (dir: 0 +x, 1 -x, 2 north = -z, 3 south = +z, 4 up, 5 down)
  const normal = varying(vec3(
    select(dir.equal(0), 1, select(dir.equal(1), -1, 0)),
    select(dir.equal(4), 1, select(dir.equal(5), -1, 0)),
    select(dir.equal(2), -1, select(dir.equal(3), 1, 0))), 'vNormal');
  // G-buffer (deferred.ts): emissive at full night strength + the window cell's random code;
  // normal + how much the voxel is a window. Night factor and lit share are applied when resolving.
  m.mrtNode = mrt({
    emissive: vec4(col.mul(lampLight.add(sign.mul(1.5))).mul(ao).mul(colourView), hash(cell)),
    normal: vec4(normal, glass.mul(isFacade).mul(colourView)),
  });
  return m;
}
