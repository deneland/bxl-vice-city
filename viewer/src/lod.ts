// Level-of-detail tile tree.
//
// Each level L has 64x64-column tiles of voxel size VS*2^L; tile (tx,ty) at L
// has children (2tx..2tx+1, 2ty..2ty+1) at L-1. Every frame we walk down from
// the coarsest tiles and split a tile while one of its voxels would cover more
// than `errPx` pixels. A parent is only replaced once all of its children are
// meshed, so there are never holes; coarse tiles load first.
import * as THREE from 'three/webgpu';
import { createVoxelMaterial, voxelGeometryBase } from './voxelMaterial';
import type { MeshRequest, MeshResult } from './mesher.worker';

export interface TileInfo { file: string; tx: number; ty: number; count: number; zmin: number; zmax: number }
export interface LevelInfo { level: number; voxelSize: number; size: [number, number, number]; voxels: number; faces: number; tiles: TileInfo[] }

interface Node {
  id: number; L: number; info: TileInfo; vs: number;
  children: Node[];
  state: 'idle' | 'queued' | 'loading' | 'ready';
  mesh: THREE.Mesh | null; faces: number;
  box: THREE.Box3;      // world space
  prio: number;
}

export interface LodStats { perLevel: number[]; faces: number; loading: number; ready: number; total: number }

export class TileManager {
  private nodes: Node[] = [];
  private roots: Node[] = [];
  private queue: Node[] = [];
  private inflight = 0;
  private shown = new Set<Node>();
  private workers: Worker[];
  private nextWorker = 0;
  private materials: THREE.Material[];
  private idleWaiters: (() => void)[] = [];
  private tmp = new THREE.Vector3();
  stats: LodStats = { perLevel: [], faces: 0, loading: 0, ready: 0, total: 0 };
  errPx: number;

  constructor(
    private base: string, private tileSize: number, private levels: LevelInfo[],
    private group: THREE.Group, private shadows: boolean, errPx: number, maxInflight = 6,
  ) {
    this.errPx = errPx;
    this.maxInflight = maxInflight;
    group.updateMatrixWorld(true);
    this.materials = levels.map((l) => createVoxelMaterial(l.voxelSize));
    const byKey = new Map<string, Node>();
    for (const lv of levels) {
      for (const t of lv.tiles) {
        const s = tileSize * lv.voxelSize;
        const box = new THREE.Box3(
          new THREE.Vector3(t.tx * s, t.zmin * lv.voxelSize, -(t.ty + 1) * s),
          new THREE.Vector3((t.tx + 1) * s, (t.zmax + 1) * lv.voxelSize, -t.ty * s),
        ).applyMatrix4(group.matrixWorld);
        const n: Node = { id: this.nodes.length, L: lv.level, info: t, vs: lv.voxelSize, children: [],
          state: 'idle', mesh: null, faces: 0, box, prio: 0 };
        this.nodes.push(n);
        byKey.set(`${lv.level}/${t.tx}/${t.ty}`, n);
      }
    }
    const top = levels.length - 1;
    for (const n of this.nodes) {
      if (n.L === top) { this.roots.push(n); continue; }
      byKey.get(`${n.L + 1}/${n.info.tx >> 1}/${n.info.ty >> 1}`)?.children.push(n);
    }
    const nWorkers = Math.max(2, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));
    this.workers = Array.from({ length: nWorkers }, () => {
      const w = new Worker(new URL('./mesher.worker.ts', import.meta.url), { type: 'module' });
      w.onmessage = (ev: MessageEvent<MeshResult>) => this.onMesh(ev.data);
      return w;
    });
    for (const r of this.roots) this.request(r, 1e9);
    this.stats.total = this.nodes.length;
  }
  private maxInflight: number;

  /** Resolves once nothing is queued or loading (after the next update). */
  whenIdle(): Promise<void> { return new Promise((r) => this.idleWaiters.push(r)); }

  update(camera: THREE.PerspectiveCamera, viewportPx: number) {
    const focal = viewportPx / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    const cam = camera.getWorldPosition(this.tmp);
    const show = new Set<Node>();
    const perLevel = this.levels.map(() => 0);
    let faces = 0;

    const visit = (n: Node) => {
      const d = n.box.distanceToPoint(cam);
      const px = d > 0 ? (n.vs * focal) / d : Infinity;
      if (n.L > 0 && px > this.errPx && n.children.length) {
        let allReady = true;
        for (const k of n.children) {
          if (k.state !== 'ready') {
            allReady = false;
            if (k.state === 'idle' || k.state === 'queued') this.request(k, k.L * 1e6 - k.box.distanceToPoint(cam));
          }
        }
        if (allReady) { for (const k of n.children) visit(k); return; }
      }
      show.add(n);
      perLevel[n.L]++;
      faces += n.faces;
    };
    for (const r of this.roots) if (r.state === 'ready') visit(r);

    for (const n of this.shown) if (!show.has(n) && n.mesh) n.mesh.visible = false;
    for (const n of show) if (n.mesh) n.mesh.visible = true;
    this.shown = show;
    this.pump();

    this.stats = { perLevel, faces, loading: this.inflight + this.queue.length,
      ready: this.nodes.filter((n) => n.state === 'ready').length, total: this.nodes.length };
    if (!this.inflight && !this.queue.length && this.roots.every((r) => r.state === 'ready') && this.idleWaiters.length) {
      const w = this.idleWaiters; this.idleWaiters = [];
      w.forEach((f) => f());
    }
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
      fetch(`${this.base}/${n.info.file}`).then((r) => r.arrayBuffer()).then((buf) => {
        const req: MeshRequest = { id: n.id, buf, ch: this.tileSize, nz: this.levels[n.L].size[2] };
        this.workers[this.nextWorker++ % this.workers.length].postMessage(req, [buf]);
      });
    }
  }

  private onMesh({ id, a, c, faces }: MeshResult) {
    const n = this.nodes[id];
    this.inflight--;
    n.state = 'ready';
    n.faces = faces;
    if (!faces) return;
    const g = new THREE.InstancedBufferGeometry();
    g.index = voxelGeometryBase.index;
    g.setAttribute('position', voxelGeometryBase.getAttribute('position'));
    g.setAttribute('packed', new THREE.InstancedBufferAttribute(a, 2));
    g.setAttribute('rgba', new THREE.InstancedBufferAttribute(c, 4, true));
    g.instanceCount = faces;
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
    mesh.visible = false; // the next update decides
    this.group.add(mesh);
    n.mesh = mesh;
  }
}
