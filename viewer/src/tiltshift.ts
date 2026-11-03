// Miniature look: very shallow depth of field focused on what you look at, plus a little extra
// saturation. A real city photographed with the depth of field of a macro lens reads as a scale model.
import * as THREE from 'three/webgpu';
import { pass, saturation, uniform, vec4 } from 'three/tsl';
import { dof } from 'three/addons/tsl/display/DepthOfFieldNode.js';

/** distance from the camera to the sharp plane, metres (set every frame) */
export const focusDistance = uniform(100);
/** how far from the sharp plane things are fully blurred, metres (scales with focus distance) */
export const focalLength = uniform(30);
/** depth of the sharp zone relative to the focus distance: smaller = stronger miniature effect */
export const DEPTH_RATIO = 0.3;
export const BOKEH = Number(new URLSearchParams(location.search).get('bokeh') ?? 2.5);  // ?bokeh= to tune

/** Diameter (px) of the blur at depths [dmin, dmax]: detail smaller than this is invisible, so the LOD
 *  may use coarser voxels there. DoF node: CoC = smoothstep(0, focalLength, |depth - focus|); bokeh disc
 *  radius BOKEH*CoC px, then a same-size max filter, so about 4*BOKEH*CoC px across. */
export function blurPx(dmin: number, dmax: number) {
  const f = focusDistance.value, fl = focalLength.value;
  if (dmin <= f && f <= dmax) return 0;
  const gap = Math.min(Math.abs(dmin - f), Math.abs(dmax - f));
  const t = Math.min(1, gap / fl), coc = t * t * (3 - 2 * t);
  return 4 * BOKEH * coc;
}

export function createTiltShift(renderer: THREE.WebGPURenderer, scene: THREE.Scene, camera: THREE.Camera) {
  const scenePass = pass(scene, camera, { samples: 4 });
  const col = scenePass.getTextureNode('output');
  const blurred: any = dof(col, scenePass.getViewZNode(), focusDistance, focalLength, BOKEH);
  const pipeline = new THREE.RenderPipeline(renderer);
  pipeline.outputNode = vec4(saturation(blurred.rgb, 1.3), 1);
  return pipeline;
}
