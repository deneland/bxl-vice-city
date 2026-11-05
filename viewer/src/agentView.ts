// Click a vehicle: who drives it and their day (agents.ts); Follow keeps the camera on them, and the
// rest of their trip is drawn on the street. Trams and trains just say what they are.
import { hhmm } from './clock';
import * as THREE from 'three/webgpu';
import type { Agent, Agents } from './agents';
import { PURPOSE } from './plans';
import type { Traffic } from './traffic';
import type { Vehicle } from './paths';
import type { Pedestrians } from './pedestrians';

const TYPE_LABEL: Record<string, string> = {
  parent: 'parent, school run', commuterOut: 'works outside the area', commuterIn: 'comes in to work',
  shopper: 'running errands, shopping', evening: 'going out tonight', errands: 'errands', delivery: 'delivery driver',
  taxi: 'taxi driver', through: 'passing through',
  transit: 'takes the tram / bus', cyclist: 'cyclist', courier: 'meal delivery courier', kid: 'walks the kids to school', walkShop: 'shops on foot', dog: 'dog owner', walkErrand: 'errands on foot', stroll: 'evening walk',
};
const CAR_LABEL: Record<string, string> = { hatch: 'hatchback', sedan: 'saloon', estate: 'estate car', suv: 'SUV', van: 'van',
  bike: 'bicycle', scooter: 'moped' };

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

  constructor(private traffic: () => Traffic | null, private agents: () => Agents | null,
    private peds: () => Pedestrians | null, fx: THREE.Object3D,
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
    const P = this.peds();
    const meshes: THREE.Object3D[] = [...t.pickable(), ...(P ? [P.mesh] : [])];
    for (const m of meshes) (m as THREE.InstancedMesh).computeBoundingSphere();   // instances move: refresh first
    this.ray.ray.copy(ray);
    const hit = this.ray.intersectObjects(meshes, false)[0];
    if (!hit || hit.instanceId === undefined) return false;
    if (P && hit.object === P.mesh) {
      const a = P.agentAt(hit.instanceId);
      if (!a) return false;
      this.agent = a as Agent; this.vehicle = null; this.hasPos = false;
      this.render(true); this.onChange();
      return true;
    }
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
    const t = this.traffic(), P = this.peds();
    const walking = this.agent?.mode === 'walk';
    const v = this.agent && !walking ? t?.vehicleOf(this.agent) ?? null : this.vehicle;
    const pos = new THREE.Vector3();
    const has = walking ? !!(P && P.positionOf(this.agent, pos)) : !!(v && t && t.positionOf(v, pos));
    if (has) {
      pos.applyMatrix4((walking ? P!.mesh : t!.group).matrixWorld);
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
      if (this.agent && (walking ? has : v && t)) {
        const r = walking ? P!.routeAhead(this.agent) : t!.routeAhead(v!);
        const g = this.line.geometry as THREE.BufferGeometry;
        g.setAttribute('position', new THREE.Float32BufferAttribute(r.pts, 3));
        this.line.visible = r.pts.length > 3;
      } else this.line.visible = false;
      this.render(false, v, walking ? P!.routeAhead(this.agent, false).len : -1);
      this.onChange();
    }
  }

  private render(force: boolean, v: Vehicle | null = null, walkLeft = -1) {
    const a = this.agent, A = this.agents(), t = this.traffic();
    let html = '';
    const btn = `<button data-act="follow">${this.following ? 'stop following' : 'follow'}</button> <button data-act="close">×</button>`;
    if (a && A) {
      const at = a.plan[a.i], next = a.plan[a.i + 1];
      const car = a.mode === 'walk' ? 'on foot' : `${CAR_LABEL[a.car.kind] ?? a.car.kind}`;
      let doing: string;
      if (a.ride === 1 || a.ride === 2 || a.ride === 3) {
        doing = a.ride === 1 ? `waiting at ${A.label(a, at.where)}` : a.ride === 2 ? 'on a tram / bus, away'
          : `coming back: gets off at ${A.label(a, at.where)}`;
      } else if ((a.driving || a.travel) && next) {
        const left = walkLeft >= 0 ? walkLeft : v && t ? t.routeAhead(v, false).len : 0;
        const eta = walkLeft >= 0 ? walkLeft / 1.3 : v && t ? left / Math.max(4, t.speedOf(v)) : 0;
        doing = `${a.mode === 'walk' ? 'walking' : 'driving'} to ${A.label(a, next.where)}<br><span class="dim">${PURPOSE[next.purpose] ?? next.purpose}` +
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
        `<div class="dim">${TYPE_LABEL[a.type] ?? a.type} · ${a.home >= 0 ? `lives on ${A.places.homeStreet(a.home)}` : 'lives outside the area'} · ${car}</div>` +
        `<div class="now">${doing}</div><div class="plan">${plan}</div>`;
    } else if (this.vehicle && t) {
      const R = this.vehicle.transit;
      if (R) {
        const st = R.stops[R.next];
        const stops = R.stops.map((x, k) => `<div class="${k < R.next ? 'past' : k === R.next ? 'cur' : ''}"><span class="t">${hhmm(x.dep)}</span>${x.name}</div>`).join('');
        html = `<div class="hd"><b><span style="background:#${R.color};padding:0 4px;border-radius:2px">${R.line}</span> → ${R.headsign}</b> ${btn}</div>` +
          `<div class="dim">STIB ${R.mode === 0 ? 'tram' : 'bus'}, to the timetable</div>` +
          (st ? `<div class="now">next stop: ${st.name}, departs ${hhmm(st.dep)}</div>` : '') + `<div class="plan">${stops}</div>`;
      } else html = `<div class="hd"><b>${t.describe(this.vehicle)}</b> ${btn}</div>`;
    }
    if (force || html) { this.el.innerHTML = html; this.el.style.display = html ? 'block' : 'none'; }
  }
}
