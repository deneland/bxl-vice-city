// Draws the vehicles (moving and parked): the simulation owns poses, this owns the GPU side.
//
// Every vehicle body (a car, one tram section, ...) is a handle with a model, a paint colour, a lights
// flag and a pose. Each frame, only what is in the camera's view goes into the instance buffers, packed
// (InstancedMesh.count = what is drawn), and with the model at a level of detail for its distance:
// near: the full 0.25 m voxel model; far: the same model at 0.5 m voxels (a voxel under ~2 px there).
// Off-screen and very distant vehicles cost nothing on the GPU. Only near vehicles cast shadows.
import * as THREE from 'three/webgpu';

export interface FleetModel { geo: THREE.BufferGeometry; far: THREE.BufferGeometry; len: number; h: number; w: number }

const NEAR = 150;     // m: full model closer than this, coarse model beyond
const FAR = 1100;     // m: not drawn beyond
const GROW = 1024;    // handles added at a time

interface Lod { mesh: THREE.InstancedMesh; paint: THREE.InstancedBufferAttribute; lights: THREE.InstancedBufferAttribute; who: Int32Array }

export class Fleet {
  readonly group = new THREE.Group();
  private names: string[];
  private radius: Float32Array;                    // bounding sphere radius per model
  private lods: Lod[][] = [];                      // [model][0 near, 1 far]
  // per handle
  private model = new Int16Array(0);
  private mat = new Float32Array(0);               // 16 per handle (column-major, group frame)
  private col = new Float32Array(0);               // paint rgb
  private lit = new Float32Array(0);               // lights 1 / 0 (parked)
  private alive = new Uint8Array(0);
  private free: number[] = [];
  private top = 0;
  private frustum = new THREE.Frustum();
  private m4 = new THREE.Matrix4();
  private cam = new THREE.Vector3();
  private pick = new Int8Array(0);
  private counts!: Int32Array;   // per (model, lod): drawn this frame
  private at!: Int32Array;

  constructor(models: Record<string, FleetModel>, private material: THREE.Material) {
    this.names = Object.keys(models);
    this.radius = Float32Array.from(this.names.map((n) => Math.hypot(models[n].len, models[n].h, models[n].w) / 2));
    // sized for a full view of parked and moving cars, so the buffers rarely have to grow (a regrow
    // rebuilds the mesh: a spike)
    const cars = new Set(['hatch', 'sedan', 'estate', 'suv', 'van']);
    this.lods = this.names.map((n) => [this.makeLod(models[n].geo, cars.has(n) ? 2048 : 256), this.makeLod(models[n].far, cars.has(n) ? 2048 : 256)]);
    this.grow(GROW);
    this.counts = new Int32Array(this.names.length * 2);
    this.at = new Int32Array(this.names.length * 2);
  }

  private makeLod(base: THREE.BufferGeometry, cap: number): Lod {
    const geo = base.clone();
    const paint = new THREE.InstancedBufferAttribute(new Float32Array(cap * 3), 3);
    const lights = new THREE.InstancedBufferAttribute(new Float32Array(cap), 1);
    paint.setUsage(THREE.DynamicDrawUsage); lights.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('paint', paint);
    geo.setAttribute('lights', lights);
    const mesh = new THREE.InstancedMesh(geo, this.material, cap);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;   // (culled here, per instance)
    mesh.count = 0;
    mesh.visible = false;
    this.group.add(mesh);
    return { mesh, paint, lights, who: new Int32Array(cap) };
  }

  /** an instanced mesh holds at most its capacity: replace it with a bigger one */
  private ensure(m: number, l: number, n: number) {
    const old = this.lods[m][l];
    if (n <= old.mesh.instanceMatrix.count) return old;
    let cap = old.mesh.instanceMatrix.count;
    while (cap < n) cap *= 2;
    const base = new THREE.BufferGeometry();
    for (const k of ['position', 'normal', 'vcol', 'vmat']) base.setAttribute(k, old.mesh.geometry.getAttribute(k));
    base.setIndex(old.mesh.geometry.getIndex());
    const lod = this.makeLod(base, cap);
    lod.mesh.layers.mask = old.mesh.layers.mask;
    this.group.remove(old.mesh);
    old.mesh.dispose();
    this.lods[m][l] = lod;
    return lod;
  }

  private grow(n: number) {
    const cap = this.model.length + n;
    const g = <T extends Float32Array | Int16Array | Uint8Array>(a: T, k: number, C: new (n: number) => T) => { const b = new C(cap * k); b.set(a); return b; };
    this.model = g(this.model, 1, Int16Array); this.mat = g(this.mat, 16, Float32Array);
    this.col = g(this.col, 3, Float32Array); this.lit = g(this.lit, 1, Float32Array); this.alive = g(this.alive, 1, Uint8Array);
  }

  /** a new body of `model`; returns its handle */
  add(model: string, paint: THREE.Color, lights: number) {
    const m = this.names.indexOf(model);
    if (m < 0) throw new Error(`no model ${model}`);
    let h = this.free.pop();
    if (h === undefined) { if (this.top >= this.model.length) this.grow(GROW); h = this.top++; }
    this.model[h] = m; this.alive[h] = 1;
    this.col[h * 3] = paint.r; this.col[h * 3 + 1] = paint.g; this.col[h * 3 + 2] = paint.b;
    this.lit[h] = lights;
    this.mat.fill(0, h * 16, h * 16 + 16);   // (not drawn until it has a pose)
    return h;
  }
  remove(h: number) { if (this.alive[h]) { this.alive[h] = 0; this.free.push(h); } }
  setPose(h: number, m: THREE.Matrix4) { m.toArray(this.mat, h * 16); }

  /** handle drawn as instance i of `mesh` in the last frame (picking), or -1 */
  handleAt(mesh: THREE.Object3D, i: number) {
    for (const ls of this.lods) for (const l of ls) if (l.mesh === mesh) return i < l.mesh.count ? l.who[i] : -1;
    return -1;
  }
  /** meshes to ray-test (picking) */
  meshes() { return this.lods.flat().map((l) => l.mesh).filter((m) => m.visible); }

  /** near models cast shadows into the vehicle shadow map (far ones are outside its box anyway) */
  enableLayer(layer: number) {
    for (const ls of this.lods) ls[0].mesh.layers.enable(layer);
  }

  /** fill the instance buffers with what `camera` sees */
  update(camera: THREE.Camera) {
    this.group.updateWorldMatrix(true, false);
    // camera frustum and position in this group's frame
    this.m4.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).multiply(this.group.matrixWorld);
    this.frustum.setFromProjectionMatrix(this.m4, camera.coordinateSystem);
    this.cam.setFromMatrixPosition(camera.matrixWorld).applyMatrix4(this.m4.copy(this.group.matrixWorld).invert());
    const counts = this.counts;
    counts.fill(0);
    const planes = this.frustum.planes, { mat, model, alive, radius } = this;
    const cx = this.cam.x, cy = this.cam.y, cz = this.cam.z;
    // pass 1: count per (model, lod), so buffers are big enough before writing
    if (this.pick.length < this.top) this.pick = new Int8Array(this.model.length);
    const pick = this.pick;
    for (let h = 0; h < this.top; h++) {
      if (!alive[h] || mat[h * 16 + 15] === 0) { pick[h] = -1; continue; }
      const o = h * 16, x = mat[o + 12], y = mat[o + 13], z = mat[o + 14], r = radius[model[h]];
      const dx = x - cx, dy = y - cy, dz = z - cz, d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      let vis = d < FAR;
      for (let p = 0; vis && p < 6; p++) {
        const n = planes[p].normal;
        if (n.x * x + n.y * y + n.z * z + planes[p].constant < -r) vis = false;
      }
      pick[h] = vis ? (d < NEAR ? 0 : 1) : -1;
      if (vis) counts[model[h] * 2 + pick[h]]++;
    }
    // pass 2: write packed
    for (let m = 0; m < this.lods.length; m++) { this.ensure(m, 0, counts[m * 2]); this.ensure(m, 1, counts[m * 2 + 1]); }
    const at = this.at;
    at.fill(0);
    const col = this.col, lit = this.lit;
    for (let h = 0; h < this.top; h++) {
      const l = pick[h];
      if (l < 0) continue;
      const m = model[h], lod = this.lods[m][l], i = at[m * 2 + l]++;
      const dst = lod.mesh.instanceMatrix.array as Float32Array, o = h * 16, d = i * 16;
      for (let k = 0; k < 16; k++) dst[d + k] = mat[o + k];
      const pa = lod.paint.array as Float32Array;
      pa[i * 3] = col[h * 3]; pa[i * 3 + 1] = col[h * 3 + 1]; pa[i * 3 + 2] = col[h * 3 + 2];
      (lod.lights.array as Float32Array)[i] = lit[h];
      lod.who[i] = h;
    }
    for (let m = 0; m < this.lods.length; m++)
      for (let l = 0; l < 2; l++) {
        const lod = this.lods[m][l], n = at[m * 2 + l];
        lod.mesh.count = n;
        lod.mesh.visible = n > 0;
        if (!n) continue;
        lod.mesh.instanceMatrix.clearUpdateRanges(); lod.mesh.instanceMatrix.addUpdateRange(0, n * 16); lod.mesh.instanceMatrix.needsUpdate = true;
        lod.paint.clearUpdateRanges(); lod.paint.addUpdateRange(0, n * 3); lod.paint.needsUpdate = true;
        lod.lights.clearUpdateRanges(); lod.lights.addUpdateRange(0, n); lod.lights.needsUpdate = true;
      }
  }

  /** bodies drawn last frame (near, far) */
  get drawn() {
    let n = 0, f = 0;
    for (const [a, b] of this.lods) { n += a.mesh.count; f += b.mesh.count; }
    return { near: n, far: f };
  }
}
