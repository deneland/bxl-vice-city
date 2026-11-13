// Level-of-detail tile tree.
//
// Each level L has 64x64-column tiles of voxel size VS*2^L; tile (tx,ty) at L
// has children (2tx..2tx+1, 2ty..2ty+1) at L-1. Every frame we walk down from
// the coarsest tiles and split a tile while one of its voxels would cover more
// than `errPx` pixels. A parent is only replaced once all of its children are
// meshed, so there are never holes; coarse tiles load first.
import * as THREE from 'three/webgpu';
import { ATLAS_W, createVoxelMaterial, voxelGeometryBase } from './voxelMaterial';
import type { MeshRequest, MeshResult } from './mesher.worker';
import type { TileEntry } from './tileFormat';

export type TileInfo = TileEntry;
export interface LevelInfo { level: number; voxelSize: number; size: [number, number, number]; voxels: number; faces: number; tiles: TileInfo[] }

/** Face data (colour, class) of all resident tiles in one texture: each tile owns a range of rows. Merged
 *  quads (mesher.worker.ts) index it by face; new tiles are copied in on the GPU. */
export class FaceAtlas {
  static readonly W = ATLAS_W;   // x 8192 rows (4096 on low-tier devices, tier.ts) = 33.5M faces: twice the face budget, it never runs full
  readonly tex: THREE.Texture;
  private target: THREE.RenderTarget | null = null;
  private free: [number, number][];
  constructor(webgpu: boolean, readonly rows = 8192) {
    // allocated on the GPU without CPU data. WebGPU: a render target's texture (a DataTexture without
    // data makes its upload throw); WebGL: a DataTexture (copies into render-target textures do not
    // arrive there; three warns once about the empty upload, harmless)
    if (webgpu) {
      this.target = new THREE.RenderTarget(FaceAtlas.W, rows, { type: THREE.UnsignedByteType, format: THREE.RGBAFormat,
        depthBuffer: false, generateMipmaps: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
      this.tex = this.target.texture;
    } else {
      const t = new THREE.DataTexture(null, FaceAtlas.W, rows, THREE.RGBAFormat, THREE.UnsignedByteType);
      t.minFilter = t.magFilter = THREE.NearestFilter;
      t.generateMipmaps = false;
      t.needsUpdate = true;   // (allocates it; without this three binds a shared placeholder)
      this.tex = t;
    }
    this.free = [[0, (rows * FaceAtlas.W) / FaceAtlas.BLOCK]];   // [first block, blocks], sorted
  }
  private ready = false;
  // allocation in blocks of BLOCK texels (a tile's range may span rows: the shader indexes texels linearly)
  static readonly BLOCK = 256;
  private blocks(n: number) { return Math.max(1, Math.ceil(n / FaceAtlas.BLOCK)); }
  /** first texel of a free range for n texels (first fit), or -1 */
  alloc(n: number) {
    const need = this.blocks(n);
    for (let i = 0; i < this.free.length; i++) {
      const [b, c] = this.free[i];
      if (c < need) continue;
      if (c === need) this.free.splice(i, 1); else this.free[i] = [b + need, c - need];
      return b * FaceAtlas.BLOCK;
    }
    return -1;
  }
  release(start: number, n: number) {
    const b0 = start / FaceAtlas.BLOCK, nb = this.blocks(n);
    let i = 0;
    while (i < this.free.length && this.free[i][0] < b0) i++;
    this.free.splice(i, 0, [b0, nb]);
    if (i + 1 < this.free.length && this.free[i][0] + this.free[i][1] === this.free[i + 1][0]) {
      this.free[i][1] += this.free[i + 1][1]; this.free.splice(i + 1, 1);
    }
    if (i > 0 && this.free[i - 1][0] + this.free[i - 1][1] === this.free[i][0]) {
      this.free[i - 1][1] += this.free[i][1]; this.free.splice(i, 1);
    }
  }
  /** copy n texels (4 bytes each) to the linear range from `start`: a partial first row, whole rows, a
   *  partial last row */
  upload(renderer: THREE.WebGPURenderer, start: number, data: Uint8Array) {
    const be = renderer.backend as any;
    if (!this.ready) {
      if (this.target) renderer.initRenderTarget(this.target); else renderer.initTexture(this.tex);
      this.ready = true;
    }
    // written straight into the atlas (no temporary texture: WebGL would upload the data twice)
    const d = be.get(this.tex);
    const put = (from: number, w: number, h: number, x: number, y: number) => {
      const bytes = data.subarray(from * 4, (from + w * h) * 4);
      if (be.isWebGPUBackend) {
        be.device.queue.writeTexture({ texture: d.texture, origin: { x, y } }, bytes, { bytesPerRow: w * 4 }, { width: w, height: h });
      } else {
        const gl = be.gl as WebGL2RenderingContext, st = be.state;
        st.bindTexture(gl.TEXTURE_2D, d.textureGPU);
        st.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        st.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
        st.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
        st.pixelStorei(gl.UNPACK_ROW_LENGTH, 0);
        st.pixelStorei(gl.UNPACK_SKIP_PIXELS, 0);
        st.pixelStorei(gl.UNPACK_SKIP_ROWS, 0);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, x, y, w, h, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
      }
    };
    const W = FaceAtlas.W, n = data.length / 4;
    let done = 0, pos = start;
    if (pos % W) {                                   // head: rest of the first row
      const w = Math.min(n, W - (pos % W));
      put(0, w, 1, pos % W, Math.floor(pos / W));
      done += w; pos += w;
    }
    const rows = Math.floor((n - done) / W);
    if (rows) { put(done, W, rows, 0, pos / W); done += rows * W; pos += rows * W; }
    if (n - done) put(done, n - done, 1, 0, pos / W);   // tail
  }
}

interface Node {
  id: number; L: number; info: TileInfo; vs: number;
  children: Node[];
  state: 'idle' | 'queued' | 'loading' | 'ready';
  mesh: THREE.Mesh | null; faces: number; quads: number; start: number;   // start: first atlas texel (-1: none)
  box: THREE.Box3;      // world space
  diag: number;         // box diagonal
  prio: number;
  used: number;         // frame number when last visited
}

export interface LodStats { perLevel: number[]; faces: number; quads: number; loading: number; ready: number; total: number; residentFaces: number; evicted: number }

export class TileManager {
  private nodes: Node[] = [];
  private roots: Node[] = [];
  private queue: Node[] = [];
  private inflight = 0;
  private shown = new Set<Node>();
  private arrived: MeshResult[] = [];
  /** ms per frame for integrating new tiles */
  arriveMs = 3;
  arriveQuads = 300_000;
  /** tiles integrated in the last update (telemetry) */
  lastArrived = 0;
  private showB = new Set<Node>();
  private readyCount = 0;
  private workers: Worker[];
  private nextWorker = 0;
  private materials: THREE.Material[];
  private idleWaiters: (() => void)[] = [];
  private tmp = new THREE.Vector3();
  stats: LodStats = { perLevel: [], faces: 0, quads: 0, loading: 0, ready: 0, total: 0, residentFaces: 0, evicted: 0 };
  private frame = 0;
  private resident = 0;   // faces held in memory
  readonly atlas: FaceAtlas;
  private evicted = 0;
  /** keep at most this many faces meshed (4 bytes each in the atlas + 12 per merged quad); least recently used go first */
  budgetFaces = 16_000_000;
  /** detail scale (>= 1): when the view needs more faces than the budget (everything resident is in use), the
   *  allowed on-screen voxel size grows until it fits; it eases back when there is room. Without this a view
   *  larger than the budget (a phone, tier.ts) loads and evicts the same tiles forever. */
  errScale = 1;
  errPx: number;
  /** optional per-tile error limit (px) from its distance range, e.g. larger where depth of field blurs */
  errAt: ((dmin: number, dmax: number) => number) | null = null;
  /** appended to tile URLs (build id), so a browser never mixes tiles from two builds */
  query = '';
  /** Cache Storage cache for this build's tiles (tileFormat.ts fetchTile; null: none) */
  cache: string | null = null;

  constructor(
    private base: string, private tileSize: number, private levels: LevelInfo[],
    private group: THREE.Group, private shadows: boolean, errPx: number, private renderer: THREE.WebGPURenderer,
    maxInflight = 16, workers = 0, atlasRows = 8192,
  ) {
    this.atlas = new FaceAtlas(!!(renderer.backend as any).isWebGPUBackend, atlasRows);
    this.errPx = errPx;
    this.maxInflight = maxInflight;
    group.updateMatrixWorld(true);
    this.materials = levels.map((l) => createVoxelMaterial(l.voxelSize, levels[0].voxelSize, this.atlas.tex));
    const byKey = new Map<string, Node>();
    for (const lv of levels) {
      for (const t of lv.tiles) {
        const s = tileSize * lv.voxelSize;
        const box = new THREE.Box3(
          new THREE.Vector3(t.tx * s, t.zmin * lv.voxelSize, -(t.ty + 1) * s),
          new THREE.Vector3((t.tx + 1) * s, (t.zmax + 1) * lv.voxelSize, -t.ty * s),
        ).applyMatrix4(group.matrixWorld);
        const n: Node = { id: this.nodes.length, L: lv.level, info: t, vs: lv.voxelSize, children: [],
          state: 'idle', mesh: null, faces: 0, quads: 0, start: -1, box, diag: box.getSize(new THREE.Vector3()).length(), prio: 0, used: 0 };
        this.nodes.push(n);
        byKey.set(`${lv.level}/${t.tx}/${t.ty}`, n);
      }
    }
    const top = levels.length - 1;
    for (const n of this.nodes) {
      if (n.L === top) { this.roots.push(n); continue; }
      byKey.get(`${n.L + 1}/${n.info.tx >> 1}/${n.info.ty >> 1}`)?.children.push(n);
    }
    const nWorkers = workers || Math.max(2, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));
    this.workers = Array.from({ length: nWorkers }, () => {
      const w = new Worker(new URL('./mesher.worker.ts', import.meta.url), { type: 'module' });
      w.onmessage = (ev: MessageEvent<MeshResult>) => { this.arrived.push(ev.data); };   // (integrated in update, within a budget)
      return w;
    });
    for (const r of this.roots) this.request(r, 1e9);
    this.stats.total = this.nodes.length;
  }
  private maxInflight: number;

  /** Resolves once nothing is queued or loading (after the next update). */
  /** tiles still queued or loading */
  get busy() { return this.inflight > 0 || this.queue.length > 0 || this.arrived.length > 0 || this.roots.some((r) => r.state !== 'ready'); }
  private meshArrived = false;
  /** true while tiles load, or when a mesh arrived since the last update (it must be shown) */
  get needsUpdate() { return this.busy || this.meshArrived || this.idleWaiters.length > 0; }

  whenIdle(): Promise<void> {
    if (!this.busy && !this.meshArrived) return Promise.resolve();
    return new Promise((r) => this.idleWaiters.push(r));
  }

  /** Picks the tiles to show. Returns true if what is on screen changed (a render is needed). */
  update(camera: THREE.PerspectiveCamera, viewportPx: number): boolean {
    const focal = viewportPx / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    const cam = camera.getWorldPosition(this.tmp);
    const show = this.showB; show.clear();   // (two sets used in turn)
    const perLevel = this.levels.map(() => 0);
    let faces = 0, quads = 0;

    this.meshArrived = false;
    const frame = ++this.frame;
    // new tiles from the workers: as many as fit in a few ms (each one is an upload and a new mesh)
    const until = performance.now() + this.arriveMs;
    let k = 0;
    let q = 0;   // (and at most arriveQuads new quads: their first render uploads them)
    while (k < this.arrived.length && (k === 0 || (performance.now() < until && q < this.arriveQuads))) {
      q += this.arrived[k].quads;
      this.onMesh(this.arrived[k++]);
    }
    if (k) this.arrived.splice(0, k);
    this.lastArrived = k;
    const visit = (n: Node) => {
      n.used = frame;
      const d = n.box.distanceToPoint(cam);
      const px = d > 0 ? (n.vs * focal) / d : Infinity;
      const err = (this.errAt ? Math.max(this.errPx, this.errAt(d, d + n.diag)) : this.errPx) * this.errScale;
      if (n.L > 0 && px > err && n.children.length) {
        let allReady = true;
        for (const k of n.children) {
          k.used = frame; // wanted: never evict it while its siblings load
          if (k.state !== 'ready') {
            allReady = false;
            if (k.state === 'idle' || k.state === 'queued') this.request(k, k.L * 1e6 - k.box.distanceToPoint(cam));
          }
        }
        if (allReady) { for (const k of n.children) visit(k); return; }
      }
      show.add(n);
      perLevel[n.L]++;
      faces += n.faces; quads += n.quads;
    };
    for (const r of this.roots) if (r.state === 'ready') visit(r);

    let changed = show.size !== this.shown.size;
    for (const n of this.shown) if (!show.has(n)) { changed = true; if (n.mesh) n.mesh.visible = false; }
    for (const n of show) if (n.mesh && !n.mesh.visible) { changed = true; n.mesh.visible = true; }
    this.showB = this.shown;
    this.shown = show;
    this.evict();
    // over budget after eviction: all of it is in use (shown or wanted) -> coarser; on screen well under -> finer
    if (this.resident > this.budgetFaces) this.errScale = Math.min(8, this.errScale * 1.02);
    else if (faces < this.budgetFaces * 0.5 && this.errScale > 1) this.errScale = Math.max(1, this.errScale / 1.005);
    this.pump();

    this.stats = { perLevel, faces, quads, loading: this.inflight + this.queue.length,
      ready: this.readyCount, total: this.nodes.length,
      residentFaces: this.resident, evicted: this.evicted };
    if (!this.inflight && !this.queue.length && this.roots.every((r) => r.state === 'ready') && this.idleWaiters.length) {
      const w = this.idleWaiters; this.idleWaiters = [];
      w.forEach((f) => f());
    }
    return changed;
  }

  /** free a tile's mesh and atlas range */
  private drop(n: Node) {
    if (n.mesh) { this.group.remove(n.mesh); n.mesh.geometry.dispose(); n.mesh = null; }
    if (n.start >= 0) { this.atlas.release(n.start, n.faces); n.start = -1; }
    this.resident -= n.faces;
    n.faces = 0; n.quads = 0;
    n.state = 'idle';
    this.readyCount--;
    this.evicted++;
  }

  private evict() {
    const target = this.budgetFaces;
    if (this.resident <= target) return;
    const cand = this.nodes.filter((n) => n.state === 'ready' && n.L < this.levels.length - 1 && !this.shown.has(n) && n.used < this.frame)
      .sort((a, b) => a.used - b.used);
    for (const n of cand) {
      if (this.resident <= target * 0.85) break;
      this.drop(n);
    }
  }

  /** the atlas has no room for n texels: free least recently used tiles until it has; first texel or -1 */
  private lru: Node[] = []; private lruFrame = -1;
  private freeFor(n: number) {
    if (this.lruFrame !== this.frame) {   // (eviction candidates, least recently used first: once per frame)
      this.lru = this.nodes.filter((x) => x.state === 'ready' && x.start >= 0 && x.L < this.levels.length - 1 && !this.shown.has(x) && x.used < this.frame)
        .sort((a, b) => b.used - a.used);
      this.lruFrame = this.frame;
    }
    while (this.lru.length) {
      const x = this.lru.pop()!;
      if (x.state !== 'ready' || x.start < 0) continue;
      this.drop(x);
      const s = this.atlas.alloc(n);
      if (s >= 0) return s;
    }
    return -1;
  }

  private request(n: Node, prio: number) {
    n.prio = prio;
    if (n.state === 'idle') { n.state = 'queued'; this.queue.push(n); }
  }

  private pump() {
    if (!this.queue.length || this.inflight >= this.maxInflight) return;
    this.queue.sort((a, b) => b.prio - a.prio);
    while (this.queue.length && this.inflight < this.maxInflight) {
      const n = this.queue.shift()!;
      n.state = 'loading';
      this.inflight++;
      // the mesher worker fetches the tile itself (keeps network callbacks and copies off the main thread)
      const url = new URL(`${this.base}/${n.info.file}${this.query}`, location.href).href;
      const req: MeshRequest = { id: n.id, tile: { url, off: n.info.off, len: n.info.len, cache: this.cache }, ch: this.tileSize, nz: this.levels[n.L].size[2] };
      this.workers[this.nextWorker++ % this.workers.length].postMessage(req);
    }
  }

  private onMesh({ id, q, t, quads, faces, failed }: MeshResult) {
    const n = this.nodes[id];
    this.inflight--;
    if (failed) { n.state = 'idle'; return; }   // (network error: asked for again later)
    if (faces) {
      // room in the face atlas: free least recently used tiles if needed
      let start = this.atlas.alloc(faces);
      if (start < 0) start = this.freeFor(faces);
      if (start < 0) { n.state = 'idle'; return; }   // (still full: asked for again later)
      this.atlas.upload(this.renderer, start, t);
      n.start = start;
      const base = start;
      for (let i = 2; i < q.length; i += 3) q[i] += base;
    }
    n.state = 'ready';
    this.readyCount++;
    this.meshArrived = true;
    n.faces = faces;
    n.quads = quads;
    this.resident += faces;
    if (!faces) return;
    const g = new THREE.InstancedBufferGeometry();
    g.index = voxelGeometryBase.index;
    g.setAttribute('position', voxelGeometryBase.getAttribute('position'));
    g.setAttribute('quad', new THREE.InstancedBufferAttribute(q, 3));
    g.instanceCount = quads;
    // bounds in tile-local coords (the shader's positionNode is invisible to three)
    const s = this.tileSize * n.vs;
    g.boundingBox = new THREE.Box3(
      new THREE.Vector3(0, n.info.zmin * n.vs, -s), new THREE.Vector3(s, (n.info.zmax + 1) * n.vs, 0));
    g.boundingSphere = g.boundingBox.getBoundingSphere(new THREE.Sphere());
    const mesh = new THREE.Mesh(g, this.materials[n.L]);
    mesh.position.set(n.info.tx * s, 0, -n.info.ty * s);
    mesh.castShadow = mesh.receiveShadow = this.shadows;
    mesh.matrixAutoUpdate = false;
    mesh.updateMatrix();
    mesh.matrixWorld.multiplyMatrices(this.group.matrixWorld, mesh.matrix);   // (the city scene does not update matrices)
    mesh.visible = false; // the next update decides
    this.group.add(mesh);
    n.mesh = mesh;
  }
}
