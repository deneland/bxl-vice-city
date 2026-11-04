// First-person walking at street level: mouse look (pointer lock), WASD / arrows, shift to run.
// Collision against L0 voxels: walls stop you, kerbs and steps up to STEP are climbed, gravity
// pulls you down. Vegetation is passable (the LiDAR canopy would otherwise trap you under trees).
import { C_GROUND } from './tileFormat';
import * as THREE from 'three/webgpu';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import type { Voxels } from './voxels';

const EYE = 1.7, HEIGHT = 1.8, RADIUS = 0.25, STEP = 0.6;   // metres
const WALK = 3, RUN = 10, GRAVITY = 9.8;
const PASSABLE = new Set([3, 4, 5]);               // LiDAR veg classes

export class Walk {
  active = false;
  private look: PointerLockControls;
  private feet = new THREE.Vector3();      // voxel space, continuous
  private vz = 0;                          // vertical speed, m/s
  private keys = new Set<string>();
  private loading = false;

  constructor(private camera: THREE.PerspectiveCamera, dom: HTMLElement, private vox: Voxels,
    private onChange: (active: boolean) => void) {
    this.look = new PointerLockControls(camera, dom);
    addEventListener('keydown', (e: KeyboardEvent) => { if (this.active) this.keys.add(e.code); });
    addEventListener('keyup', (e: KeyboardEvent) => this.keys.delete(e.code));
    addEventListener('blur', () => this.keys.clear());
    dom.addEventListener('click', () => { if (this.active && !this.look.isLocked) this.look.lock(); });
    for (const ev of ['change', 'lock', 'unlock'] as const) this.look.addEventListener(ev, () => this.onChange(this.active));
  }

  get locked() { return this.look.isLocked; }

  private solid(x: number, y: number, z: number) {
    const c = this.vox.cls(x, y, z);
    return c !== 0 && !PASSABLE.has(c);
  }

  /** is the body free with feet at (x, y, z)? (voxel units) */
  private free(x: number, y: number, z: number) {
    const r = RADIUS / this.vox.vs;
    const z0 = Math.floor(z + 1e-3), z1 = Math.floor(z + HEIGHT / this.vox.vs - 1e-3);
    for (let yy = Math.floor(y - r); yy <= Math.floor(y + r); yy++)
      for (let xx = Math.floor(x - r); xx <= Math.floor(x + r); xx++)
        for (let zz = z0; zz <= z1; zz++) if (this.solid(xx, yy, zz)) return false;
    return true;
  }

  /** top of the highest solid under the footprint at or below z + maxUp, or -1 */
  private support(x: number, y: number, z: number, maxUp: number) {
    const r = RADIUS / this.vox.vs;
    let best = -1;
    for (let yy = Math.floor(y - r); yy <= Math.floor(y + r); yy++)
      for (let xx = Math.floor(x - r); xx <= Math.floor(x + r); xx++)
        for (let zz = Math.min(this.vox.nz - 1, Math.floor(z + maxUp)); zz >= Math.max(0, Math.floor(z) - 40); zz--)
          if (this.solid(xx, yy, zz)) { best = Math.max(best, zz + 1); break; }
    return best;
  }

  /** start walking near a scene point: on the nearest open ground (not on a roof) */
  async enter(at: THREE.Vector3) {
    if (this.loading) return;
    this.loading = true;
    const v = this.vox.toVoxel(at);
    await this.vox.preload(v.x, v.y, 80);
    this.loading = false;
    const spot = this.findGround(Math.floor(v.x), Math.floor(v.y));
    if (!spot) return;
    this.feet.copy(spot);
    this.vz = 0;
    this.active = true;
    this.place();
    this.camera.lookAt(this.camera.position.clone().add(this.bestView()));
    this.camera.near = 0.1;
    this.camera.updateProjectionMatrix();
    this.look.lock();
    this.onChange(true);
  }

  exit() {
    this.active = false;
    this.keys.clear();
    if (this.look.isLocked) this.look.unlock();
    this.camera.near = 0.5;
    this.camera.updateProjectionMatrix();
    this.onChange(false);
  }

  /** spiral out from (x, y) to a column with ground and room to stand */
  private findGround(x0: number, y0: number) {
    for (let r = 0; r < 120; r++) {
      for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
        const x = x0 + dx, y = y0 + dy;
        for (let z = 0; z < this.vox.nz; z++) {
          if (this.vox.cls(x, y, z) !== C_GROUND) continue;
          // open space: room to stand here and 2 m around (a street, not a gap between houses)
          const ok = [[0, 0], [4, 0], [-4, 0], [0, 4], [0, -4], [3, 3], [-3, 3], [3, -3], [-3, -3]]
            .every(([ox, oy]) => this.free(x + 0.5 + ox, y + 0.5 + oy, z + 1));
          if (ok && this.nearStreetFacade(x, y)) return new THREE.Vector3(x + 0.5, y + 0.5, z + 1);
          break;
        }
      }
    }
    return null;
  }

  /** look down the street: the most open horizontal direction, leaning towards the map camera's heading */
  private bestView() {
    const cam = this.camera.getWorldDirection(new THREE.Vector3()).setY(0).normalize();
    let best = cam, bestScore = -1;
    for (let i = 0; i < 32; i++) {
      const a = (i / 32) * 2 * Math.PI, dx = Math.cos(a), dy = Math.sin(a);   // voxel space
      let d = 0;
      while (d < 80 / this.vox.vs && this.free(this.feet.x + dx * d, this.feet.y + dy * d, this.feet.z + 1)) d += 1;
      const dir = new THREE.Vector3(dx, 0, -dy);
      const score = d * (1.3 + 0.3 * dir.dot(cam));
      if (score > bestScore) { bestScore = score; best = dir; }
    }
    return best;
  }

  /** facades coloured from street-level photos within ~10 m: we are on a street, not in a back garden */
  private nearStreetFacade(x: number, y: number) {
    const r = Math.round(10 / this.vox.vs);
    for (let dy = -r; dy <= r; dy += 2) for (let dx = -r; dx <= r; dx += 2) if (this.vox.photo(x + dx, y + dy)) return true;
    return false;
  }

  private place() {
    const eye = this.feet.clone();
    eye.z += EYE / this.vox.vs;
    this.vox.toScene(eye, this.camera.position);
  }

  /** advance by dt seconds; returns true if the camera moved */
  update(dt: number): boolean {
    if (!this.active) return false;
    dt = Math.min(dt, 0.1);
    const vs = this.vox.vs;
    // keep the neighbourhood loaded; stand still until it is
    if (!this.vox.has(this.feet.x, this.feet.y)) { this.vox.preload(this.feet.x, this.feet.y, 80); return false; }
    this.vox.preload(this.feet.x, this.feet.y, 80);

    const k = this.keys;
    const fwd = +(k.has('KeyW') || k.has('ArrowUp')) - +(k.has('KeyS') || k.has('ArrowDown'));
    const side = +(k.has('KeyD') || k.has('ArrowRight')) - +(k.has('KeyA') || k.has('ArrowLeft'));
    const speed = (k.has('ShiftLeft') || k.has('ShiftRight') ? RUN : WALK) / vs;
    const before = this.feet.clone();

    if (fwd || side) {
      const dir = this.camera.getWorldDirection(new THREE.Vector3());
      const f = new THREE.Vector2(dir.x, -dir.z).normalize();          // voxel space forward
      const move = new THREE.Vector2(f.x * fwd + f.y * side, f.y * fwd - f.x * side).normalize().multiplyScalar(speed * dt);
      // per axis so you slide along walls; climb small steps
      for (const [mx, my] of [[move.x, 0], [0, move.y]]) {
        const nx = this.feet.x + mx, ny = this.feet.y + my;
        for (let up = 0; up <= STEP / vs; up += 0.5) {
          if (this.free(nx, ny, this.feet.z + up)) { this.feet.set(nx, ny, this.feet.z + up); break; }
        }
      }
    }
    // gravity / settle onto the ground
    const top = this.support(this.feet.x, this.feet.y, this.feet.z, 0);
    if (top >= 0 && this.feet.z - top < 0.05) {
      this.feet.z = top; this.vz = 0;
    } else {
      this.vz -= GRAVITY * dt;
      const nz = this.feet.z + (this.vz * dt) / vs;
      this.feet.z = top >= 0 ? Math.max(nz, top) : Math.max(nz, 0);
      if (this.feet.z === top) this.vz = 0;
    }
    if (this.feet.equals(before)) return false;
    this.place();
    return true;
  }
}
