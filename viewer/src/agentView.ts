// Click a vehicle: who drives it and their day (agents.ts); Follow keeps the camera on them, and the
// rest of their trip is drawn on the street. Trams and trains just say what they are.
import * as THREE from 'three/webgpu';
import { PURPOSE, type Agent, type Agents } from './agents';
import type { Traffic, Vehicle } from './traffic';

const hhmm = (t: number) => `${String(Math.floor(t / 3600) % 24).padStart(2, '0')}:${String(Math.floor(t / 60) % 60).padStart(2, '0')}`;
const TYPE_LABEL: Record<string, string> = {
  parent: 'parent, school run', commuterOut: 'works outside the area', commuterIn: 'comes in to work',
  shopper: 'running errands, shopping', evening: 'going out tonight', errands: 'errands', delivery: 'delivery driver',
  taxi: 'taxi driver', through: 'passing through',
};
const CAR_LABEL: Record<string, string> = { hatch: 'hatchback', sedan: 'saloon', estate: 'estate car', suv: 'SUV', van: 'van' };

export class AgentView {
  private el = document.getElementById('agent')!;
  private agent: Agent | null = null;
  private vehicle: Vehicle | null = null;      // trams / trains (no agent)
  private following = false;
  private line: THREE.Line;
  private lastPos = new THREE.Vector3();
  private hasPos = false;
  private tPanel = 0;
  private ray = new THREE.Raycaster();

  constructor(private traffic: () => Traffic | null, private agents: () => Agents | null, fx: THREE.Object3D,
    private controls: { target: THREE.Vector3 }, private camera: THREE.Camera, private onChange: () => void) {
    this.line = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicNodeMaterial({ color: 0x40e0ff, depthTest: false,
      transparent: true, opacity: 0.85, fog: false }));
    this.line.frustumCulled = false;
    this.line.renderOrder = 7;
    this.line.visible = false;
    fx.add(this.line);
    this.el.addEventListener('click', (e) => {
      const b = (e.target as HTMLElement).closest('button');
      if (b?.dataset.act === 'follow') this.setFollow(!this.following);
      if (b?.dataset.act === 'close') this.clear();
    });
  }

  get active() { return !!(this.agent || this.vehicle); }
  get isFollowing() { return this.following; }

  /** try to pick a vehicle along a ray; true if one was hit */
  pick(ray: THREE.Ray) {
    const t = this.traffic();
    if (!t) return false;
    const meshes = t.pickable();
    for (const m of meshes) m.computeBoundingSphere();   // instances move: refresh before casting
    this.ray.ray.copy(ray);
    const hit = this.ray.intersectObjects(meshes, false)[0];
    if (!hit || hit.instanceId === undefined) return false;
    const v = t.vehicleAt(hit.object, hit.instanceId);
    if (!v) return false;
    this.agent = (v.agent as Agent) ?? null;
    this.vehicle = v.agent ? null : v;
    this.hasPos = false;
    this.render(true);
    this.onChange();
    return true;
  }

  clear() {
    this.agent = null; this.vehicle = null; this.following = false;
    this.el.style.display = 'none';
    this.line.visible = false;
    this.onChange();
  }

  setFollow(on: boolean) {
    this.following = on && this.active;
    this.hasPos = false;
    this.render(true);
  }

  /** per frame: keep the camera on the vehicle (follow), refresh the route line and the panel */
  update(now: number) {
    if (!this.active) return;
    const t = this.traffic();
    const v = this.agent ? t?.vehicleOf(this.agent) ?? null : this.vehicle;
    const pos = new THREE.Vector3();
    if (v && t) {
      t.positionOf(v, pos);
      pos.applyMatrix4(t.group.matrixWorld);
      if (this.following) {
        // move camera and orbit point together, so the view keeps its angle
        if (this.hasPos) {
          const d = pos.clone().sub(this.lastPos);
          this.controls.target.add(d);
          this.camera.position.add(d);
        } else {
          const d = pos.clone().sub(this.controls.target);
          this.controls.target.add(d);
          this.camera.position.add(d);
        }
        this.lastPos.copy(pos); this.hasPos = true;
      }
    } else this.hasPos = false;
    if (now - this.tPanel > 0.5 || now < this.tPanel) {
      this.tPanel = now;
      if (v && t && this.agent) {
        const r = t.routeAhead(v);
        const g = this.line.geometry as THREE.BufferGeometry;
        g.setAttribute('position', new THREE.Float32BufferAttribute(r.pts, 3));
        this.line.visible = r.pts.length > 3;
      } else this.line.visible = false;
      this.render(false, now, v);
      this.onChange();
    }
  }

  private render(force: boolean, now = 0, v: Vehicle | null = null) {
    const a = this.agent, A = this.agents(), t = this.traffic();
    let html = '';
    const btn = `<button data-act="follow">${this.following ? 'stop following' : 'follow'}</button> <button data-act="close">×</button>`;
    if (a && A) {
      const at = a.plan[a.i], next = a.plan[a.i + 1];
      const car = `${CAR_LABEL[a.car.kind] ?? a.car.kind}`;
      let doing: string;
      if (a.driving && next) {
        const left = v && t ? t.routeAhead(v).len : 0;
        const eta = v && t ? left / Math.max(4, t.speedOf(v)) : 0;
        doing = `driving to ${A.label(a, next.where)}<br><span class="dim">${PURPOSE[next.purpose] ?? next.purpose}` +
          (left ? ` · ${Math.round(left)} m, about ${Math.max(1, Math.round(eta / 60))} min` : '') + '</span>';
      } else {
        doing = `${PURPOSE[at.purpose] ?? at.purpose} (${A.label(a, at.where)})` +
          (next ? `<br><span class="dim">leaves at ${hhmm(at.until)}</span>` : '');
      }
      const plan = a.plan.map((p, k) => {
        const when = k === 0 ? '' : hhmm(a.plan[k - 1].until);
        const cur = k === a.i + (a.driving ? 1 : 0);
        return `<div class="${cur ? 'cur' : k < a.i ? 'past' : ''}"><span class="t">${when}</span>${PURPOSE[p.purpose] ?? p.purpose} · ${A.label(a, p.where)}</div>`;
      }).join('');
      html = `<div class="hd"><b>${a.name}</b> ${btn}</div>` +
        `<div class="dim">${TYPE_LABEL[a.type] ?? a.type} · ${a.home >= 0 ? `lives on ${A.street(A.places.homes[a.home][3])}` : 'lives outside the area'} · ${car}</div>` +
        `<div class="now">${doing}</div><div class="plan">${plan}</div>`;
    } else if (this.vehicle && t) {
      html = `<div class="hd"><b>${t.describe(this.vehicle)}</b> ${btn}</div><div class="dim">random lines for now; real STIB lines come later</div>`;
    }
    if (force || html) { this.el.innerHTML = html; this.el.style.display = html ? 'block' : 'none'; }
  }
}
