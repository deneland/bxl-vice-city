// One material for all chunks. Each instance is one voxel face; the vertex
// shader expands a unit quad into the right face from packed attributes.
import * as THREE from 'three/webgpu';
import {
  attribute, float, floor, hash, mod, mix, positionGeometry, positionWorld, pow, select, smoothstep, uniform, varying,
  vec2, vec3, vec4,
} from 'three/tsl';
import { LAMP_COLOR, gridOrigin, gridSize, lampMap, night, windowsLit } from './nightlights';

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

export function createVoxelMaterial(voxelSize: number, baseSize = voxelSize) {
  const m = new THREE.MeshStandardNodeMaterial({ roughness: 0.92, metalness: 0 });
  m.flatShading = true;

  const packed = attribute('packed', 'vec2');
  const rgba = attribute('rgba', 'vec4');
  const q = positionGeometry;

  const p0 = packed.x;
  const dir = bits(p0, 20, 3);
  const axis = floor(dir.div(2));
  const positive = float(1).sub(mod(dir, 2));
  // negative faces swap u/v to flip winding so the front side faces outwards
  const a = select(positive.equal(1), q.x, q.y);
  const b = select(positive.equal(1), q.y, q.x);
  const pv = vec3(bits(p0, 0, 6), bits(p0, 6, 6), bits(p0, 12, 8))
    .add(axisVec(axis).mul(positive))
    .add(axisVec(mod(axis.add(1), 3)).mul(a))
    .add(axisVec(mod(axis.add(2), 3)).mul(b));
  // voxel space (x east, y north, z up) -> three (x, y up, -z north)
  m.positionNode = vec3(pv.x, pv.z, pv.y.negate()).mul(voxelSize);

  // per-corner ambient occlusion level 0..3, interpolated across the face
  const lvl = mod(floor(bits(packed.y, 0, 8).div(pow(4, a.add(b.mul(2))))), 4);
  const ao = varying(mix(float(0.42), float(1.0), lvl.div(3)), 'vAO');

  // LiDAR colours are sRGB -> linear, with a light saturation lift
  const srgb = rgba.xyz;
  const lum = srgb.dot(vec3(0.299, 0.587, 0.114));
  const col = varying(pow(mix(vec3(lum), srgb, 1.25).clamp(0, 1), vec3(2.2)), 'vColor');

  // class view: palette colour per semantic class
  const label = bits(packed.y, 17, 4);
  const hex = (h: number) => { const c = new THREE.Color(h); return vec3(c.r, c.g, c.b); }; // sRGB hex -> linear
  let lc: any = hex(0xff00ff); // unknown code: magenta, should never show
  for (const [code, , h] of LABELS) lc = select(label.equal(code), hex(h), lc);
  const labelCol: any = varying(lc, 'vLabel');

  m.colorNode = vec4(mix(col, labelCol, viewMode).mul(ao), 1);

  // --- night lights (see nightlights.ts) ---
  // street lamps: light map over the ground plan, fading out above lamp height
  const rel = positionWorld.sub(gridOrigin);                             // metres from voxel (0,0,0)
  const lm = lampMap.sample(vec2(rel.x, rel.z.negate()).div(gridSize));
  const above = rel.y.sub(lm.g.mul(255 * baseSize));
  const fall = float(1).sub(smoothstep(2.5, 9, above)).mul(smoothstep(-1, 0, above));
  const up = select(dir.equal(4), float(1), float(0.6));
  const lampLight = varying(up, 'vUp').mul(lm.r).mul(fall).mul(1.6);
  // lit windows: dark (glass) photo-coloured facade voxels, per window cell (1.5 m wide, 3 m floor) on or off
  const isFacade = varying(select(label.equal(1).and(bits(packed.y, 21, 1).equal(1)).and(dir.lessThan(4)), 1, 0), 'vFacade');
  const xAxis = varying(select(axis.equal(0), 1, 0), 'vXAxis');                // face normal along east-west
  const v = rel.div(baseSize).mul(vec3(1, 1, -1));                            // L0 voxels: x east, y up, z north
  const along = mix(v.x, v.z, xAxis), across = mix(v.z, v.x, xAxis);
  const cell = floor(along.div(3)).add(floor(v.y.sub(0.01).div(6)).mul(101)).add(floor(across.add(0.5).div(2)).mul(7919));
  const lit = select(hash(cell).lessThan(windowsLit), 1, 0);
  const glass = float(1).sub(smoothstep(0.18, 0.32, lum));                    // photo colour is dark: window
  const warm = select(hash(cell.add(17)).lessThan(0.8), vec3(1.0, 0.62, 0.3), vec3(0.55, 0.7, 1.0));
  const windowLight = warm.mul(lit).mul(glass).mul(isFacade).mul(1.4);
  // shop signs glow
  const sign = varying(select(label.equal(2), 1, 0), 'vSign');
  const lampCol = vec3(LAMP_COLOR.r, LAMP_COLOR.g, LAMP_COLOR.b);
  m.emissiveNode = col.mul(lampCol.mul(lampLight).add(sign.mul(1.5))).mul(ao).add(windowLight).mul(night)
    .mul(float(1).sub(viewMode));
  return m;
}
