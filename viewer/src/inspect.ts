// Click a voxel: outline it and show what the pipeline knows about it.
import * as THREE from 'three/webgpu';
import { LABELS } from './voxelMaterial';
import type { Voxels } from './voxels';

const FLAG_FILL = 1, FLAG_PHOTO = 8;
const LIDAR: Record<number, string> = {
  0: 'never classified', 1: 'unclassified', 2: 'ground', 3: 'low vegetation', 4: 'medium vegetation',
  5: 'high vegetation', 6: 'building', 9: 'water', 10: 'rail', 64: 'vehicle (from geometry)',
};
const LABEL = new Map(LABELS.map(([c, n, h]) => [c, { n, h }]));
const hex = (h: number) => '#' + h.toString(16).padStart(6, '0');

export class Inspector {
  private box: THREE.LineSegments;
  private panel = document.getElementById('inspect')!;
  private seq = 0;

  constructor(private vox: Voxels, scene: THREE.Scene, private zOrigin: number, private onChange: () => void) {
    const vs = vox.vs * 1.04;
    this.box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(vs, vs, vs)),
      new THREE.LineBasicMaterial({ color: 0xff2bd6, depthTest: false }));
    this.box.renderOrder = 10;
    this.box.visible = false;
    scene.add(this.box);
  }

  clear() {
    this.box.visible = false;
    this.panel.style.display = 'none';
    this.onChange();
  }

  async pick(ray: THREE.Ray) {
    const seq = ++this.seq;
    const hit = await this.vox.raycast(ray.origin, ray.direction);
    if (seq !== this.seq) return;  // a newer click won
    if (!hit) { this.clear(); return; }
    const { x, y, z } = hit.voxel;
    const rec = this.vox.record(x, y, z);
    if (!rec) { this.clear(); return; }
    const vs = this.vox.vs;
    this.vox.toScene(new THREE.Vector3(x + 0.5, y + 0.5, z + 0.5), this.box.position);
    this.box.visible = true;

    // ground: lowest ground voxel in this column or close by
    let ground = -1;
    for (let r = 0; r < 12 && ground < 0; r++)
      for (let dy = -r; dy <= r && ground < 0; dy++) for (let dx = -r; dx <= r && ground < 0; dx++)
        for (let zz = 0; zz < this.vox.nz; zz++) if (this.vox.cls(x + dx, y + dy, zz) === 2) { ground = zz + 1; break; }

    const lab = LABEL.get(rec.flags >> 4);
    const rgb = `rgb(${rec.r},${rec.g},${rec.b})`;
    const rows: [string, string][] = [
      ['class', lab ? `<span class="sw" style="background:${hex(lab.h)}"></span>${lab.n}` : `code ${rec.flags >> 4}`],
      ['LiDAR class', LIDAR[rec.cls] ?? `${rec.cls}`],
      ['colour', `<span class="sw" style="background:${rgb}"></span>${rec.flags & FLAG_PHOTO ? 'street-level photos' : 'aerial photo (LiDAR)'}`],
      ['height', `${(this.zOrigin + (z + 1) * vs).toFixed(1)} m above sea level` +
        (ground >= 0 ? ` · ${((z + 1 - ground) * vs).toFixed(1)} m above ground` : '')],
    ];
    if (rec.flags & FLAG_FILL) rows.push(['', 'interior fill (not measured)']);
    this.panel.innerHTML = rows.map(([k, v]) => `<div><b>${k}</b> ${v}</div>`).join('') +
      '<div class="dim">click empty sky or Esc to close</div>';
    this.panel.style.display = 'block';
    this.onChange();
  }
}
