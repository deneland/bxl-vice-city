// Device tier: phones and weak machines get smaller memory budgets, so the page does not run the GPU out of
// memory (a lost context or a killed tab). Decided once at start-up; ?tier=low|high overrides it.
//
//   low   a touch-first device (coarse pointer), at most 4 GB of memory (navigator.deviceMemory, Chromium only),
//         or a GPU whose largest texture is under 8192 (the full face atlas would not fit)
import { config } from './config';

export interface Tier {
  name: 'low' | 'high';
  why: string;
  atlasRows: number;     // face atlas rows (x 4096 faces; 4 bytes each): 8192 = 134 MB, 4096 = 67 MB
  budgetFaces: number;   // resident faces (at most half the atlas)
  megapixels: number;    // render resolution cap
  workers: number;       // mesher workers
  inflight: number;      // tiles fetched / meshed at once
}

/** the largest texture this device's WebGL2 supports (a throwaway context; 0: no WebGL2) */
function maxTextureSize(): number {
  try {
    const gl = document.createElement('canvas').getContext('webgl2');
    if (!gl) return 0;
    const n = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    gl.getExtension('WEBGL_lose_context')?.loseContext();   // (free it now: browsers cap live contexts)
    return n;
  } catch { return 0; }
}

export function deviceTier(): Tier {
  const cores = navigator.hardwareConcurrency || 4;
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  const touch = matchMedia('(pointer: coarse)').matches && !matchMedia('(pointer: fine)').matches;
  const tex = maxTextureSize();
  const reasons = [touch && 'touch screen', mem !== undefined && mem <= 4 && `${mem} GB memory`,
    tex > 0 && tex < 8192 && `max texture ${tex}`].filter(Boolean) as string[];
  const forced = config.tier === 'low' || config.tier === 'high' ? config.tier : null;
  const low = forced ? forced === 'low' : reasons.length > 0;
  const why = forced ? `?tier=${forced}` : reasons.join(', ') || 'desktop';
  return low
    ? { name: 'low', why, atlasRows: 4096, budgetFaces: (config.budget ?? 5) * 1e6, megapixels: config.megapixels ?? 1.5,
        workers: Math.max(2, Math.min(3, cores - 1)), inflight: 8 }
    : { name: 'high', why, atlasRows: 8192, budgetFaces: (config.budget ?? 16) * 1e6, megapixels: config.megapixels ?? 3.5,
        workers: Math.max(2, Math.min(8, cores - 1)), inflight: 16 };
}
