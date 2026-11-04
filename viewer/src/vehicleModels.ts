// Voxel models for traffic, generated at load time on a 0.25 m grid (twice the city's detail: wheels,
// window bands and lights are visible, but they are still clearly voxels). Each voxel has a material:
// paint (the vehicle's own colour, per instance), a fixed colour (glass, tyres, trim, livery), or a light.
// Models are centred on their body (x forward, y up, z across), sizes in metres.
import * as THREE from 'three/webgpu';

export const VOX = 0.25;

/** voxel materials: 1 paint (per instance); fixed colours; lights (emissive at night) */
export const M = { paint: 1, glass: 2, tyre: 3, trim: 4, head: 5, tail: 6, roof: 7, stripe: 8, cab: 9, skin: 10, jacket: 11, jeans: 12 };
const FIXED: Record<number, number> = {
  [M.glass]: 0x1c2630, [M.tyre]: 0x141414, [M.trim]: 0x2b2c2e, [M.head]: 0xe8e4d8, [M.tail]: 0x7a1410,
  [M.roof]: 0xd9dad6, [M.stripe]: 0xa47a3c, [M.cab]: 0x3a3d42, [M.skin]: 0xd9a47e, [M.jacket]: 0x2a3140, [M.jeans]: 0x334a6b,
};

class Grid {
  readonly nx: number; readonly ny: number; readonly nz: number;
  readonly v: Uint8Array;
  constructor(len: number, h: number, w: number, readonly vox = VOX) {
    this.nx = Math.round(len / vox); this.ny = Math.round(h / vox); this.nz = Math.round(w / vox);
    this.v = new Uint8Array(this.nx * this.ny * this.nz);
  }
  in(x: number, y: number, z: number) { return x >= 0 && y >= 0 && z >= 0 && x < this.nx && y < this.ny && z < this.nz; }
  get(x: number, y: number, z: number) { return this.in(x, y, z) ? this.v[(y * this.nz + z) * this.nx + x] : 0; }
  set(x: number, y: number, z: number, m: number) { if (this.in(x, y, z)) this.v[(y * this.nz + z) * this.nx + x] = m; }
}

const vox = (m: number) => Math.round(m / VOX);

/** wheels: tyre voxels below the body and in the lower body's outer columns at both axles */
function wheels(g: Grid, axles: number[], clear: number) {
  for (const ax of axles)
    for (let x = ax - 1; x <= ax + 1; x++)
      for (let y = 0; y <= clear; y++)
        for (const z of [0, 1, g.nz - 2, g.nz - 1])
          if (y < clear || z === 0 || z === g.nz - 1) g.set(x, y, z, M.tyre);
}

interface CarSpec { len: number; w: number; h: number; belt: number; hood: number; trunk: number;
  slopeF: number; slopeR: number; over: number; wb: number }

/** car: lower body to the belt line, a glass cabin with a painted roof, bumpers, wheels, lights */
function car(s: CarSpec) {
  const g = new Grid(s.len, s.h, s.w);
  const clear = 1, belt = vox(s.belt), top = g.ny;
  const front = g.nx - 1;
  const cabF = Math.round(g.nx * (1 - s.hood)), cabR = Math.round(g.nx * s.trunk);
  for (let x = 0; x < g.nx; x++)
    for (let z = 0; z < g.nz; z++)
      for (let y = clear; y < belt; y++) {
        const end = x === 0 || x === front, side = z === 0 || z === g.nz - 1;
        if (end && side) continue;                                   // rounded corners
        if (end && y === belt - 1 && (z < 2 || z > g.nz - 3)) continue;
        g.set(x, y, z, end && y === clear ? M.trim : M.paint);
      }
  // cabin: sloped windscreen / rear window, glass sides, painted roof and pillars
  for (let y = belt; y < top; y++) {
    const k = y - belt;
    const xf = cabF - Math.round(k * s.slopeF), xr = cabR + Math.round(k * s.slopeR);
    const mid = Math.round((xf + xr) / 2);
    for (let x = xr; x <= xf; x++)
      for (let z = 1; z < g.nz - 1; z++) {
        const roof = y === top - 1, pillar = x === mid || x === xr || x === xf;
        g.set(x, y, z, roof || (pillar && (z === 1 || z === g.nz - 2) && x !== xf) ? M.paint : M.glass);
      }
  }
  wheels(g, [front - vox(s.over), front - vox(s.over + s.wb)], clear);
  for (const z of [1, 2, g.nz - 3, g.nz - 2]) { g.set(front, belt - 2, z, M.head); g.set(0, belt - 2, z, M.tail); }
  return g;
}

/** van: short bonnet, tall box with a windscreen and cab side windows */
function van() {
  const len = 5.3, w = 2.0, h = 2.4, g = new Grid(len, h, w);
  const clear = 1, belt = vox(1.0), front = g.nx - 1, bonnet = vox(0.9);
  for (let x = 0; x < g.nx; x++)
    for (let z = 0; z < g.nz; z++)
      for (let y = clear; y < g.ny; y++) {
        const end = x === 0 || x === front, side = z === 0 || z === g.nz - 1;
        if (end && side) continue;
        if (y >= belt && x > front - bonnet + Math.round((y - belt) * 0.5)) continue;   // sloped front
        const cabWin = y >= belt + 1 && y < g.ny - 1 && x > front - bonnet - vox(1.2);
        const windscreen = y >= belt && y < g.ny - 1 && x === front - bonnet + Math.round((y - belt) * 0.5) && !side;
        g.set(x, y, z, windscreen || (cabWin && side) ? M.glass : end && y === clear ? M.trim : M.paint);
      }
  wheels(g, [front - vox(0.9), front - vox(0.9 + 3.6)], clear);
  for (const z of [1, 2, g.nz - 3, g.nz - 2]) { g.set(front, belt - 2, z, M.head); g.set(0, belt - 1, z, M.tail); }
  return g;
}

/** city bus: low skirt, long window band, big windscreen, wheels */
function bus() {
  const len = 12, w = 2.5, h = 3.1, g = new Grid(len, h, w);
  const clear = 1, front = g.nx - 1, winLo = vox(1.0), winHi = vox(2.6);
  for (let x = 0; x < g.nx; x++)
    for (let z = 0; z < g.nz; z++)
      for (let y = clear; y < g.ny; y++) {
        const end = x === 0 || x === front, side = z === 0 || z === g.nz - 1;
        if (end && side) continue;
        const band = y >= winLo && y < winHi && !(side && x % 9 === 0);   // window posts
        const door = side && z === g.nz - 1 && y < winHi && (Math.abs(x - (front - 6)) < 4 || Math.abs(x - g.nx / 2) < 4);
        const screen = x === front && y >= winLo - 1 && y < g.ny - 1;
        g.set(x, y, z, y === g.ny - 1 ? M.roof : band || door || screen ? M.glass : M.paint);
      }
  wheels(g, [front - vox(2.7), front - vox(2.7 + 6.0)], clear);
  for (const z of [1, 2, g.nz - 3, g.nz - 2]) { g.set(front, 2, z, M.head); g.set(0, 3, z, M.tail); }
  return g;
}

/** STIB-style low-floor tram section (Bombardier Flexity look): silver body, dark window band,
 *  gold stripe under a light roof; cab ends with a dark sloped nose and a large windscreen */
function tramSection(len: number, cab: 'front' | 'rear' | null, pantograph: boolean) {
  const w = 2.3, h = 3.75, g = new Grid(len, h, w);   // (top row: pantograph)
  const floor = 1, winLo = vox(1.05), winHi = vox(2.45), stripe = vox(3.0), roof = vox(3.25);
  const nose = cab === 'front' ? g.nx - 1 : 0, dir = cab === 'front' ? -1 : 1;
  for (let x = 0; x < g.nx; x++)
    for (let z = 0; z < g.nz; z++)
      for (let y = floor; y < roof; y++) {
        const side = z === 0 || z === g.nz - 1;
        const fromNose = cab ? (x - nose) * dir : 99;                  // voxels behind the nose
        if (cab && fromNose < 3 && y >= winLo && fromNose < Math.round((y - winLo) * 0.4)) continue;   // sloped nose
        if (cab && fromNose === 0 && side) continue;
        let m = M.paint;
        if (y >= winLo && y < winHi) m = M.glass;
        if (y === stripe) m = M.stripe;                                // thin gold line under the roof edge
        if (side && y >= floor && y < winHi && (x % 22 === 11)) m = M.glass;   // doors
        if (cab && fromNose < 5) m = y >= winLo && y < stripe ? M.glass : M.cab;
        g.set(x, y, z, m);
      }
  for (let x = 0; x < g.nx; x++) for (let z = 1; z < g.nz - 1; z++) g.set(x, roof, z, M.roof);
  if (pantograph) for (let x = g.nx / 2 - 3; x < g.nx / 2 + 3; x++) for (let z = 3; z < g.nz - 3; z++) g.set(x, roof + 1, z, M.trim);
  if (cab) for (const z of [2, 3, g.nz - 4, g.nz - 3]) g.set(nose, 3, z, cab === 'front' ? M.head : M.tail);
  return g;
}

/** train carriage: painted body, window band, grey roof */
function carriage(len: number) {
  const w = 2.9, h = 4.0, g = new Grid(len, h, w);
  for (let x = 0; x < g.nx; x++)
    for (let z = 0; z < g.nz; z++)
      for (let y = 2; y < g.ny; y++) {
        const band = y >= vox(1.6) && y < vox(2.6) && x % 10 !== 0;
        g.set(x, y, z, y === g.ny - 1 ? M.roof : band ? M.glass : M.paint);
      }
  wheels(g, [vox(3), g.nx - 1 - vox(3)], 2);
  return g;
}

/** fill a box of voxels [x0, x1) × [y0, y1) × [z0, z1) */
function box(g: Grid, x0: number, x1: number, y0: number, y1: number, z0: number, z1: number, m: number) {
  for (let x = x0; x < x1; x++) for (let y = y0; y < y1; y++) for (let z = z0; z < z1; z++) g.set(x, y, z, m);
}

/** bicycle with rider, 0.125 m voxels (x forward): frame in paint */
function bicycle() {
  const g = new Grid(1.75, 1.75, 0.5, 0.125);
  box(g, 0, 4, 0, 4, 1, 3, M.tyre); box(g, 10, 14, 0, 4, 1, 3, M.tyre);   // wheels
  box(g, 2, 12, 3, 4, 2, 2 + 1, M.paint); box(g, 5, 6, 3, 7, 2, 3, M.paint); box(g, 10, 11, 4, 9, 2, 3, M.paint);   // frame, seat post, fork
  box(g, 9, 11, 9, 10, 0, 4, M.trim);                                     // handlebar
  box(g, 5, 7, 4, 8, 1, 3, M.jeans);                                      // legs
  box(g, 5, 8, 8, 12, 1, 3, M.jacket); box(g, 8, 10, 9, 11, 1, 3, M.jacket);   // body, arms
  box(g, 6, 8, 12, 14, 1, 3, M.skin);                                     // head
  g.set(13, 5, 2, M.head); g.set(0, 5, 2, M.tail);
  return g;
}

/** moped with a delivery box (in paint) and a rider with a helmet, 0.125 m voxels */
function moped() {
  const g = new Grid(1.9, 1.75, 0.625, 0.125);
  box(g, 0, 3, 0, 3, 2, 3, M.tyre); box(g, 12, 15, 0, 3, 2, 3, M.tyre);
  box(g, 2, 13, 2, 4, 1, 4, M.trim); box(g, 11, 13, 4, 8, 1, 4, M.trim);   // body, front
  box(g, 4, 9, 4, 5, 1, 4, M.cab);                                         // seat
  box(g, 0, 4, 5, 9, 0, 5, M.paint);                                       // delivery box
  box(g, 6, 8, 5, 9, 1, 4, M.jeans); box(g, 5, 9, 9, 12, 1, 4, M.jacket); box(g, 9, 11, 10, 11, 1, 4, M.jacket);
  box(g, 6, 9, 12, 14, 1, 4, M.trim);                                      // helmet
  g.set(14, 5, 2, M.head); g.set(0, 3, 2, M.tail);
  return g;
}

/** voxel grid -> mesh of its outer faces; attributes: position, normal, vcol (fixed colour), vmat */
function mesh(g: Grid) {
  const pos: number[] = [], nrm: number[] = [], col: number[] = [], mat: number[] = [], idx: number[] = [];
  const VOX = g.vox, ox = (g.nx * VOX) / 2, oy = (g.ny * VOX) / 2, oz = (g.nz * VOX) / 2;
  const c = new THREE.Color();
  const dirs = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  for (let y = 0; y < g.ny; y++)
    for (let z = 0; z < g.nz; z++)
      for (let x = 0; x < g.nx; x++) {
        const m = g.get(x, y, z);
        if (!m) continue;
        c.set(FIXED[m] ?? 0xffffff);
        const shade = 1;
        for (const [dx, dy, dz] of dirs) {
          if (g.get(x + dx, y + dy, z + dz)) continue;
          // face corners: the two axes other than the normal
          const a = dx ? [0, 1, 0] : [1, 0, 0], b = dz ? [0, 1, 0] : dx ? [0, 0, 1] : [0, 0, 1];
          const cx = x + (dx > 0 ? 1 : 0), cy = y + (dy > 0 ? 1 : 0), cz = z + (dz > 0 ? 1 : 0);
          const base = pos.length / 3;
          for (const [u, v] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
            pos.push((cx + a[0] * u + b[0] * v) * VOX - ox, (cy + a[1] * u + b[1] * v) * VOX - oy, (cz + a[2] * u + b[2] * v) * VOX - oz);
            nrm.push(dx, dy, dz);
            col.push(c.r * shade, c.g * shade, c.b * shade);
            mat.push(m === M.paint ? 1 : m === M.head ? 2 : m === M.tail ? 3 : 0);
          }
          // winding: front side along the normal
          const n = new THREE.Vector3(dx, dy, dz), e1 = new THREE.Vector3(...a), e2 = new THREE.Vector3(...b);
          if (e1.cross(e2).dot(n) > 0) idx.push(base, base + 1, base + 2, base + 2, base + 1, base + 3);
          else idx.push(base, base + 2, base + 1, base + 2, base + 3, base + 1);
        }
      }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  geo.setAttribute('vcol', new THREE.Float32BufferAttribute(col, 3));
  geo.setAttribute('vmat', new THREE.Float32BufferAttribute(mat, 1));
  geo.setIndex(idx);
  return geo;
}

/** all models by name, with their size (length, height, width) */
export function buildModels() {
  const grids: Record<string, Grid> = {
    hatch: car({ len: 3.9, w: 1.75, h: 1.5, belt: 0.9, hood: 0.22, trunk: 0.1, slopeF: 1.2, slopeR: 0.3, over: 0.75, wb: 2.5 }),
    sedan: car({ len: 4.6, w: 1.8, h: 1.45, belt: 0.85, hood: 0.27, trunk: 0.2, slopeF: 1.3, slopeR: 1.1, over: 0.9, wb: 2.75 }),
    estate: car({ len: 4.7, w: 1.8, h: 1.5, belt: 0.9, hood: 0.25, trunk: 0.06, slopeF: 1.3, slopeR: 0.3, over: 0.9, wb: 2.8 }),
    suv: car({ len: 4.6, w: 1.9, h: 1.7, belt: 1.05, hood: 0.24, trunk: 0.08, slopeF: 1.0, slopeR: 0.4, over: 0.9, wb: 2.75 }),
    van: van(),
    bus: bus(),
    tramCab: tramSection(6.4, 'front', false), tramRear: tramSection(6.4, 'rear', false),
    tramMid: tramSection(6.0, null, false), tramPanto: tramSection(6.0, null, true),
    carriage: carriage(24),
    bike: bicycle(), moped: moped(),
  };
  const out: Record<string, { geo: THREE.BufferGeometry; len: number; h: number; w: number }> = {};
  for (const [k, g] of Object.entries(grids)) out[k] = { geo: mesh(g), len: g.nx * g.vox, h: g.ny * g.vox, w: g.nz * g.vox };
  return out;
}
