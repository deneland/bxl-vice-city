// Miniature look: very shallow depth of field focused on what you look at, plus a little extra
// saturation (applied in deferred.ts). A real city photographed with the depth of field of a macro lens reads as a scale model.
import { uniform } from 'three/tsl';
import { config } from './config';

/** distance from the camera to the sharp plane, metres (set every frame) */
export const focusDistance = uniform(100);
/** how far from the sharp plane things are fully blurred, metres (scales with focus distance) */
export const focalLength = uniform(30);
/** depth of the sharp zone relative to the focus distance: smaller = stronger miniature effect */
export const DEPTH_RATIO = 0.3;
export const BOKEH = config.bokeh;

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
