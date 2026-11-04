// Deferred rendering with a cached city layer.
//
//  city      the voxel tiles, drawn into a G-buffer only when the camera, the tiles or the view mode
//            change (millions of faces): albedo (with AO), emissive + window code, normal + glass, depth
//  forward   everything that moves or is transparent (vehicles, rain, overlays), drawn every frame on top
//            of a copy of the city depth, so buildings hide them; three.js lights them
//  glow      additive light (lamp glows), depth-tested against city + forward, added when resolving
//  resolve   one full-screen pass per frame lights the city from the G-buffer (sun, sky, fog, lamps,
//            windows, shadows), composites the forward layer, then miniature DoF / tone mapping / FXAA
//  shadows   (optional) two sun shadow maps: the city over the whole region, redrawn only when the sun
//            moves or other tiles are loaded; and the vehicles in a small box around the view, every frame
//
// A frame with only moving vehicles costs their own geometry plus a few full-screen passes.
import * as THREE from 'three/webgpu';
import {
  abs, attribute, dot, float, fract, hash, instanceIndex, normalGeometry, getViewPosition, max, min, mix, mrt, output, perspectiveDepthToViewZ, renderOutput, saturation,
  cos, normalWorld, positionGeometry, positionLocal, positionWorld, select, sin, smoothstep, step, texture, uniform, uv, vec2, vec3, vec4,
} from 'three/tsl';
import { fxaa } from 'three/addons/tsl/display/FXAANode.js';
import { dof } from 'three/addons/tsl/display/DepthOfFieldNode.js';
import { night, windowsLit } from './nightlights';
import { BOKEH, focalLength, focusDistance } from './tiltshift';

/** Less clean: the city's voxels carry photo noise and grime, so models should too. Per-voxel brightness
 *  variation (voxel cell from the model-space position, nudged inside along the normal), darker towards
 *  the bottom (road dirt), and a per-instance amount of dirt. vox: model voxel size; grime: strength. */
function weathered(vox: number, grime: number) {
  const cell = positionGeometry.sub(normalGeometry.mul(vox * 0.25)).div(vox).floor();
  const n = hash(cell.x.add(cell.y.mul(57)).add(cell.z.mul(113)).add(instanceIndex.toFloat().mul(1.7)));
  const dirt = hash(instanceIndex.toFloat().mul(7.13).add(0.5));                 // this vehicle's dirt
  const low = float(1).sub(smoothstep(-0.95, -0.1, positionGeometry.y));       // 1 at the bottom
  return float(0.78).add(n.mul(0.34)).mul(float(1).sub(low.mul(grime * 0.5).mul(dirt.mul(0.8).add(0.4))));
}

/** per-voxel G-buffer outputs written by the voxel material (see voxelMaterial.ts mrtNode) */
export const GBUF = ['output', 'emissive', 'normal'] as const;

/** layer of objects that cast shadows into the per-frame (vehicle) shadow map */
export const DYNAMIC_SHADOW_LAYER = 1;
const DEBUG_SHADOW = location.search.includes('dbg=shadow');
const DEPTH_BLIT = new URLSearchParams(location.search).get('depthcopy') !== 'shader';   // (?depthcopy=shader: the old copy, for A/B)
const STATIC_SIZE = 4096, DYN_SIZE = 1024, DYN_HALF = 120;   // texels; m half-width of the vehicle shadow box

interface ShadowMap { rt: THREE.RenderTarget; cam: THREE.OrthographicCamera; mat: any }

export class Deferred {
  readonly city: THREE.RenderTarget;
  readonly fwd: THREE.RenderTarget;
  readonly glow: THREE.RenderTarget;
  private cityMRT = mrt({ output, emissive: vec4(0), normal: vec4(0, 1, 0, 0) });
  private restore: THREE.QuadMesh;
  private pipeline: THREE.RenderPipeline;
  private composite: any;
  private viewZ: any;
  private stat: ShadowMap | null = null;
  private dyn: ShadowMap | null = null;
  private outputMRT = mrt({ output });
  private depthOnly = new THREE.MeshBasicNodeMaterial();
  private camWorld = uniform(new THREE.Matrix4());
  private projInv = uniform(new THREE.Matrix4());

  // lighting, copied from the three.js lights / fog every frame
  readonly sunDir = uniform(new THREE.Vector3(0, 1, 0));
  readonly sunCol = uniform(new THREE.Color());
  readonly skyCol = uniform(new THREE.Color());       // hemisphere light, sky side (× intensity)
  readonly groundCol = uniform(new THREE.Color());    // hemisphere light, ground side (× intensity)
  readonly fogCol = uniform(new THREE.Color());
  readonly fogNear = uniform(700);
  readonly fogFar = uniform(2600);
  readonly background = uniform(new THREE.Color());
  readonly near = uniform(0.5);
  readonly far = uniform(6000);

  /** shadows: half-width (m) of the box the city shadow map must cover */
  constructor(private renderer: THREE.WebGPURenderer, shadows?: { half: number }) {
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    const opts = { type: THREE.HalfFloatType, depthBuffer: true };
    this.city = new THREE.RenderTarget(size.x, size.y, { ...opts, count: GBUF.length });
    GBUF.forEach((n, i) => { this.city.textures[i].name = n; });
    (this.city as any).label = 'gbuffer';   // (names for the GPU pass timings, main.ts)
    this.city.depthTexture = new THREE.DepthTexture(size.x, size.y, THREE.FloatType);   // (32-bit: copyable)
    this.fwd = new THREE.RenderTarget(size.x, size.y, opts);
    this.fwd.depthTexture = new THREE.DepthTexture(size.x, size.y, THREE.FloatType);
    this.glow = new THREE.RenderTarget(size.x, size.y, opts);
    (this.fwd as any).label = 'fwd'; (this.glow as any).label = 'glow';
    this.glow.depthTexture = this.fwd.depthTexture;   // shared: glows are hidden by city and vehicles

    // copies the city depth into the forward target, so the forward layer is depth-tested against it
    const m = new THREE.MeshBasicNodeMaterial({ colorWrite: false, depthTest: true, depthWrite: true });
    m.depthFunc = THREE.AlwaysDepth;
    m.depthNode = texture(this.city.depthTexture, uv()).x;
    this.restore = new THREE.QuadMesh(m);

    if (shadows) {
      const mk = (size: number, half: number): ShadowMap => {
        const rt = new THREE.RenderTarget(size, size, { type: THREE.UnsignedByteType, depthBuffer: true });
        rt.texture.name = 'output';
        rt.depthTexture = new THREE.DepthTexture(size, size, THREE.FloatType);   // (32-bit: copyable on WebGPU)
        const cam = new THREE.OrthographicCamera(-half, half, half, -half, 1, 5000);
        cam.coordinateSystem = renderer.coordinateSystem;
        cam.updateProjectionMatrix();
        return { rt, cam, mat: uniform(new THREE.Matrix4()) };
      };
      this.stat = mk(STATIC_SIZE, shadows.half);
      this.dyn = mk(DYN_SIZE, DYN_HALF);
      (this.stat.rt as any).label = 'shadowCity'; (this.dyn.rt as any).label = 'shadowDyn';
      this.dyn.cam.layers.set(DYNAMIC_SHADOW_LAYER);
    }

    // --- resolve: light the city ---
    const alb = texture(this.city.textures[0], uv());
    const emi = texture(this.city.textures[1], uv());
    const nrm = texture(this.city.textures[2], uv());
    const depth = texture(this.city.depthTexture, uv()).x;
    const dist = perspectiveDepthToViewZ(depth, this.near, this.far).negate();
    const world = this.stat ? this.camWorld.mul(vec4(getViewPosition(uv(), depth, this.projInv), 1)).xyz : null;
    const lit = DEBUG_SHADOW ? this.light(nrm.xyz, world).mul(1 / Math.PI) : alb.rgb.mul(this.light(nrm.xyz, world));
    // lit windows: emissive.a is the window cell's random code, normal.a how much the voxel looks like glass
    const code = emi.a;
    const warm = select(fract(code.mul(13.7)).lessThan(0.8), vec3(1.0, 0.62, 0.3), vec3(0.55, 0.7, 1.0));
    const win = warm.mul(select(code.lessThan(windowsLit), 1, 0)).mul(nrm.w).mul(1.4);
    const glow = emi.rgb.add(win).mul(night);
    const fogged = mix(lit.add(glow), this.fogCol, smoothstep(this.fogNear, this.fogFar, dist));
    // (?dbg=holes: magenta where no geometry was drawn)
    const bg: any = location.search.includes('dbg=holes') ? vec3(1, 0, 1) : vec3(this.background as any);
    const cityCol = select(depth.greaterThanEqual(1), bg, fogged);
    // forward layer is premultiplied (cleared to transparent black)
    const f = texture(this.fwd.texture, uv());
    const g = texture(this.glow.texture, uv());
    this.composite = vec4(cityCol.mul(float(1).sub(f.a)).add(f.rgb).add(g.rgb), 1);
    this.viewZ = perspectiveDepthToViewZ(texture(this.fwd.depthTexture, uv()).x, this.near, this.far);

    this.pipeline = new THREE.RenderPipeline(renderer);
    this.pipeline.outputColorTransform = false;
    this.setTilt(false);
  }

  /** Light arriving at a surface (multiply by albedo): sun (with shadows) and sky. Shared by the city
   *  (resolve pass) and forward-lit objects such as vehicles, so both look the same. */
  light(n: any, world: any, vehicleShadows = true, taps = 4): any {
    // three.js physical lights on a rough dielectric: Lambert diffuse (albedo / π) from sun and hemisphere
    const hemi = mix(this.groundCol, this.skyCol, n.y.mul(0.5).add(0.5));
    let sun: any = this.sunCol.mul(max(dot(n, this.sunDir), 0));
    if (this.stat && this.dyn && world) {
      const p = world.add(n.mul(0.35));   // pushed off the surface a little against shadow acne
      const webgl = this.renderer.coordinateSystem === THREE.WebGLCoordinateSystem;
      const lookup = (sm: ShadowMap, size: number, bias: number) => {
        const c = sm.mat.mul(vec4(p, 1));
        const ndc = c.xyz.div(c.w);
        const suv = vec2(ndc.x.mul(0.5).add(0.5), ndc.y.mul(0.5).add(0.5).oneMinus());
        const z = (webgl ? ndc.z.mul(0.5).add(0.5) : ndc.z).sub(bias);
        let lit: any = float(0);
        const offs = taps === 1 ? [[0, 0]] : [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]];   // 2×2 PCF (or 1 tap)
        for (const [dx, dy] of offs)
          lit = lit.add(step(z, texture(sm.rt.depthTexture!, suv.add(vec2(dx / size, dy / size))).x));
        const inside = suv.x.greaterThan(0).and(suv.x.lessThan(1)).and(suv.y.greaterThan(0)).and(suv.y.lessThan(1));
        return select(inside, lit.mul(1 / offs.length), float(1));
      };
      // depth range 5000 m: 1e-4 ≈ 0.5 m
      // (vehicles skip the vehicle shadow map: at its resolution they would shadow themselves)
      const st = lookup(this.stat, STATIC_SIZE, 1e-4);
      const sh = vehicleShadows ? min(st, lookup(this.dyn, DYN_SIZE, 5e-5)) : st;
      if (DEBUG_SHADOW) return vec3(lookup(this.stat, STATIC_SIZE, 1e-4), lookup(this.dyn, DYN_SIZE, 5e-5), 0).mul(Math.PI);
      sun = sun.mul(sh);
    }
    // (× 1.1: stands in for the faint specular of the rough material we used before)
    return hemi.add(sun).mul(1.1 / Math.PI);
  }

  /** material for opaque forward objects (vehicles), lit like the city; instance colours give the albedo */
  litMaterial(extra?: (world: any) => any) {
    const m = new THREE.MeshBasicNodeMaterial();
    let l = this.light(normalWorld, positionWorld);
    if (extra) l = l.add(extra(positionWorld));
    m.colorNode = vec4(l, 1);
    return m;
  }

  /** material for the voxel vehicle models (vehicleModels.ts): albedo from the model's fixed colours
   *  (vcol) or the instance's paint (vmat 1); head / tail lights (vmat 2 / 3) glow, mostly at night */
  vehicleMaterial(extra?: (world: any) => any) {
    const m = new THREE.MeshBasicNodeMaterial();
    const kind = attribute('vmat', 'float').add(0.5).floor();   // (interpolated: not exactly an integer)
    // paint a little muted, like the LiDAR colours of the city around it
    const paint = attribute('paint', 'vec3'), pl = dot(paint, vec3(0.299, 0.587, 0.114));
    const alb = select(kind.equal(1), mix(paint, vec3(pl), 0.25).mul(0.85), attribute('vcol', 'vec3'));
    let l = this.light(normalWorld, positionWorld, false, 1);   // (small objects: one shadow tap is enough)
    l = l.mul(weathered(0.25, 0.9));
    if (extra) l = l.add(extra(positionWorld));
    const lamp = select(kind.equal(2), vec3(4, 3.6, 3), select(kind.equal(3), vec3(2.6, 0.15, 0.1), vec3(0)));
    m.colorNode = vec4(alb.mul(l).add(lamp.mul(night.mul(0.9).add(0.1)).mul(attribute('lights', 'float'))), 1);   // parked: off
    return m;
  }

  /** material for pedestrians (pedestrians.ts): shirt / skin / trousers per instance, legs and arms
   *  swing with the walk phase (vertex shader) */
  pedestrianMaterial(extra?: (world: any) => any) {
    const m = new THREE.MeshBasicNodeMaterial();
    const part = attribute('part', 'vec2'), walk = attribute('walk', 'vec2');
    const limb = part.y.add(0.5).floor(), mat = part.x.add(0.5).floor();
    const ph = walk.x, pg = positionGeometry;
    const legSign = select(limb.equal(2), float(-1), float(1)), armSign = select(limb.equal(4), float(-1), float(1));
    const leg = select(limb.lessThan(3).and(limb.greaterThan(0)), sin(ph).mul(legSign).mul(0.2).mul(float(0.875).sub(pg.y).div(0.875)), float(0));
    const arm = select(limb.greaterThan(2), sin(ph).mul(armSign).mul(-0.14).mul(float(1.5).sub(pg.y).div(0.65)), float(0));
    // (positionNode replaces the already instanced position: add the swing to positionLocal, turned by
    // the person's heading `yaw`, instead of starting from the raw geometry)
    const dx = leg.add(arm), yaw = walk.y;
    m.positionNode = positionLocal.add(vec3(cos(yaw).mul(dx), abs(sin(ph)).mul(0.025), sin(yaw).negate().mul(dx)));
    const alb = select(mat.equal(1), attribute('shirt', 'vec3'), select(mat.equal(2), attribute('skin', 'vec3'),
      select(mat.equal(3), attribute('pants', 'vec3'), vec3(0.04, 0.035, 0.03))));
    let l = this.light(normalWorld, positionWorld, false);
    l = l.mul(weathered(0.125, 0.3));
    if (extra) l = l.add(extra(positionWorld));
    m.colorNode = vec4(alb.mul(l), 1);
    return m;
  }

  /** miniature look: shallow depth of field on the composited image (city + forward layer) */
  setTilt(on: boolean) {
    let c = this.composite;
    if (on) {
      const blurred: any = dof(c, this.viewZ, focusDistance, focalLength, BOKEH);
      c = vec4(saturation(blurred.rgb, 1.3), 1);
    }
    this.pipeline.outputNode = fxaa(renderOutput(c));
    this.pipeline.needsUpdate = true;
  }

  private fwdReady = false;
  setSize(w: number, h: number) {   // (shadow maps keep their size)
    this.fwdReady = false;
    this.city.setSize(w, h);
    this.fwd.setSize(w, h);
    this.glow.setSize(w, h);
  }

  /** copy light / fog / camera state into the resolve uniforms */
  sync(camera: THREE.PerspectiveCamera, sunLight: THREE.DirectionalLight, hemi: THREE.HemisphereLight,
    fog: THREE.Fog, background: THREE.Color) {
    this.sunDir.value.subVectors(sunLight.position, sunLight.target.position).normalize();
    this.sunCol.value.copy(sunLight.color).multiplyScalar(sunLight.intensity);
    this.skyCol.value.copy(hemi.color).multiplyScalar(hemi.intensity);
    this.groundCol.value.copy(hemi.groundColor).multiplyScalar(hemi.intensity);
    this.fogCol.value.copy(fog.color);
    this.fogNear.value = fog.near;
    this.fogFar.value = fog.far;
    this.background.value.copy(background);
    this.near.value = camera.near;
    this.far.value = camera.far;
    this.camWorld.value.copy(camera.matrixWorld);
    this.projInv.value.copy(camera.projectionMatrixInverse);
  }

  get shadows() { return this.stat !== null; }

  private aim(sm: ShadowMap, center: THREE.Vector3) {
    sm.cam.position.copy(center).addScaledVector(this.sunDir.value, 2500);
    sm.cam.lookAt(center);
    sm.cam.updateMatrixWorld();
    sm.mat.value.multiplyMatrices(sm.cam.projectionMatrix, sm.cam.matrixWorldInverse);
  }

  /** city shadow map over the whole region (expensive: when the sun moved or other tiles were loaded) */
  renderStaticShadow(cityScene: THREE.Scene, center: THREE.Vector3) {
    if (!this.stat) return;
    this.aim(this.stat, center);
    const r = this.renderer;
    r.setRenderTarget(this.stat.rt);
    r.setMRT(this.outputMRT);
    r.clear();
    r.render(cityScene, this.stat.cam);
    r.setMRT(null);
    r.setRenderTarget(null);
  }

  // --- city shadows in slices: after tiles load, the new map is drawn into a back buffer a group of
  // tiles per frame (depth testing merges the groups exactly as one draw would) and copied to the front
  // when complete, so there is no 20-30 ms frame. (The sun moving still redraws at once: see above.)
  private statBack: ShadowMap | null = null;
  private job: { groups: THREE.Object3D[][]; k: number } | null = null;
  /** quads per slice frame */
  sliceQuads = 600_000;
  get shadowBusy() { return this.job !== null; }

  beginStaticShadow(cityScene: THREE.Scene, center: THREE.Vector3) {
    if (!this.stat) return;
    if (!this.statBack) {
      const rt = new THREE.RenderTarget(STATIC_SIZE, STATIC_SIZE, { type: THREE.UnsignedByteType, depthBuffer: true });
      rt.texture.name = 'output';
      rt.depthTexture = new THREE.DepthTexture(STATIC_SIZE, STATIC_SIZE, THREE.FloatType);   // (copyable on WebGPU)
      (rt as any).label = 'shadowCity';
      const cam = this.stat.cam.clone();
      this.statBack = { rt, cam, mat: uniform(new THREE.Matrix4()) };
    }
    this.statBack.cam.copy(this.stat.cam);
    this.aim(this.statBack, center);
    // the tiles drawn now, in groups of about sliceQuads
    const groups: THREE.Object3D[][] = [[]];
    let n = 0;
    cityScene.traverseVisible((o) => {
      if (!(o as THREE.Mesh).isMesh) return;
      const q = ((o as THREE.Mesh).geometry as THREE.InstancedBufferGeometry).instanceCount ?? 0;
      if (n > 0 && n + q > this.sliceQuads) { groups.push([]); n = 0; }
      groups[groups.length - 1].push(o); n += q;
    });
    this.job = { groups, k: 0 };
  }

  /** draw the next group; returns true when the new map is complete and in use */
  stepStaticShadow(cityScene: THREE.Scene) {
    const job = this.job, back = this.statBack;
    if (!job || !back || !this.stat) return true;
    const group = new Set(job.groups[job.k]);
    const hidden: THREE.Object3D[] = [];
    cityScene.traverseVisible((o) => { if ((o as THREE.Mesh).isMesh && !group.has(o)) hidden.push(o); });
    for (const o of hidden) o.visible = false;
    const off = job.groups[job.k].filter((o) => !o.visible && o.parent);   // (hidden by a LOD change since)
    for (const o of off) o.visible = true;
    const r = this.renderer, autoClear = r.autoClear;
    r.setRenderTarget(back.rt);
    r.setMRT(this.outputMRT);
    if (job.k === 0) r.clear();
    r.autoClear = false;
    r.render(cityScene, back.cam);
    r.autoClear = autoClear;
    r.setMRT(null);
    r.setRenderTarget(null);
    for (const o of hidden) o.visible = true;
    for (const o of off) o.visible = false;
    if (++job.k < job.groups.length) return false;
    // complete: the back map becomes the front one
    r.copyTextureToTexture(back.rt.depthTexture!, this.stat.rt.depthTexture!);
    this.stat.mat.value.copy(back.mat.value);
    this.job = null;
    return true;
  }
  /** a full redraw supersedes a sliced one in progress */
  cancelStaticShadow() { this.job = null; }

  /** vehicle shadow map around `focus` (every frame while they move) */
  renderDynamicShadow(fwdScene: THREE.Scene, focus: THREE.Vector3) {
    if (!this.dyn) return;
    // snap to whole texels so shadows do not shimmer when the view pans
    const t = (2 * DYN_HALF) / DYN_SIZE;
    const f = focus.clone().applyMatrix4(this.dyn.cam.matrixWorldInverse);
    f.x = Math.round(f.x / t) * t; f.y = Math.round(f.y / t) * t;
    this.aim(this.dyn, f.applyMatrix4(this.dyn.cam.matrixWorld));
    const r = this.renderer;
    r.setRenderTarget(this.dyn.rt);
    r.clear();
    // plain material: the lit one samples this very shadow map (a feedback loop)
    const override = fwdScene.overrideMaterial;
    fwdScene.overrideMaterial = this.depthOnly;
    r.render(fwdScene, this.dyn.cam);
    fwdScene.overrideMaterial = override;
    r.setRenderTarget(null);
  }

  /** draw the city G-buffer (expensive; only when it changed) */
  renderCity(cityScene: THREE.Scene, camera: THREE.Camera) {
    const r = this.renderer;
    r.setRenderTarget(this.city);
    r.setMRT(this.cityMRT);
    r.setClearColor(0x000000, 0);
    r.clear();
    r.render(cityScene, camera);
    r.setMRT(null);
    r.setRenderTarget(null);
  }

  /** draw the forward layer on a copy of the city depth and the glow layer, then resolve and present */
  present(fwdScene: THREE.Scene, glowScene: THREE.Scene, camera: THREE.Camera) {
    const r = this.renderer;
    const autoClear = r.autoClear;
    if (DEPTH_BLIT) {
      // the city depth copied in by the GPU (a shader writing depth would disable early depth rejection
      // for the vehicles drawn after it: hidden parked cars would all be shaded)
      if (!this.fwdReady) {   // (the copy needs the target's framebuffer: it exists once it was drawn into)
        r.setRenderTarget(this.fwd); r.clear(); r.setRenderTarget(null);
        this.fwdReady = true;
      }
      r.copyTextureToTexture(this.city.depthTexture!, this.fwd.depthTexture!);
      r.setRenderTarget(this.fwd);
      r.setClearColor(0x000000, 0);
      r.clear(true, false, false);
      r.autoClear = false;
    } else {
      r.setRenderTarget(this.fwd);
      r.setClearColor(0x000000, 0);
      r.clear();
      r.autoClear = false;
      this.restore.render(r);
    }
    r.render(fwdScene, camera);
    r.setRenderTarget(this.glow);
    r.clear(true, false, false);
    r.render(glowScene, camera);
    r.autoClear = autoClear;
    r.setRenderTarget(null);
    this.pipeline.render();
  }
}
