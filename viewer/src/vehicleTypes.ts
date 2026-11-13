// Vehicle types: the voxel models (vehicleModels.ts) as bodies on axles, with their driving parameters,
// the road vehicle mix and fleet colours.
import { buildModels } from './vehicleModels';

/** one body on two axles: model, length, front overhang (bumper to front axle), wheelbase */
export interface Section { model: string; len: number; h: number; over: number; wb: number }
/** vehicle type: its sections (carriages, front first), the gap between them, driving parameters */
export interface Kind { name: string; secs: Section[]; gap: number; total: number; amax: number; b: number; T: number; s0: number;
  maxV?: number }   // own top speed (bicycles), m/s

export const MODELS = buildModels();
const sec = (model: string, over: number, wb: number): Section => ({ model, len: MODELS[model].len, h: MODELS[model].h, over, wb });
const kind = (name: string, secs: Section[], gap: number, amax: number, b: number, T: number, s0: number): Kind =>
  ({ name, secs, gap, total: secs.reduce((a, x) => a + x.len, 0) + gap * (secs.length - 1), amax, b, T, s0 });
const CAR = [2.4, 3.2, 1.2, 2.0] as const;
export const TYPES = {
  hatch: kind('hatch', [sec('hatch', 0.75, 2.5)], 0, ...CAR),
  sedan: kind('sedan', [sec('sedan', 0.9, 2.75)], 0, ...CAR),
  estate: kind('estate', [sec('estate', 0.9, 2.8)], 0, ...CAR),
  suv: kind('suv', [sec('suv', 0.9, 2.75)], 0, ...CAR),
  van: kind('van', [sec('van', 0.9, 3.6)], 0, 1.7, 2.6, 1.4, 2.0),
  bus: kind('bus', [sec('bus', 2.7, 6.0)], 0, 1.1, 1.8, 1.8, 2.5),
  // STIB Flexity: T3000 ~32 m in 5 sections, T4000 ~43 m in 7
  t3000: kind('t3000', [sec('tramCab', 0.8, 4.8), sec('tramMid', 0.8, 4.4), sec('tramPanto', 0.8, 4.4),
    sec('tramMid', 0.8, 4.4), sec('tramRear', 0.8, 4.8)], 0.15, 1.0, 1.3, 2.0, 3.0),
  t4000: kind('t4000', [sec('tramCab', 0.8, 4.8), sec('tramMid', 0.8, 4.4), sec('tramPanto', 0.8, 4.4), sec('tramMid', 0.8, 4.4),
    sec('tramPanto', 0.8, 4.4), sec('tramMid', 0.8, 4.4), sec('tramRear', 0.8, 4.8)], 0.15, 1.0, 1.3, 2.0, 3.0),
  bike: { ...kind('bike', [sec('bike', 0.25, 1.05)], 0, 1.0, 2.2, 0.9, 1.0), maxV: 6 },        // ~22 km/h
  scooter: { ...kind('scooter', [sec('moped', 0.2, 1.3)], 0, 3.0, 3.5, 1.0, 1.5), maxV: 14 },
  // emergency vehicles (rare, siren on): quick off the mark
  police: kind('police', [sec('police', 0.9, 2.8)], 0, 3.0, 3.5, 1.0, 2.0),
  ambulance: kind('ambulance', [sec('ambulance', 0.9, 3.6)], 0, 2.2, 3.0, 1.2, 2.0),
  train: kind('train', [0, 1, 2, 3].map(() => sec('carriage', 3, 18)), 0.8, 0.6, 0.8, 3.0, 5.0),
};
// road vehicle mix
const ROAD_MIX: [Kind, number][] = [[TYPES.hatch, 0.27], [TYPES.sedan, 0.22], [TYPES.estate, 0.12], [TYPES.suv, 0.17], [TYPES.van, 0.14]];
const MIX_TOTAL = ROAD_MIX.reduce((a, [, w]) => a + w, 0);
export const pickKind = () => { let r = Math.random() * MIX_TOTAL; for (const [k, w] of ROAD_MIX) { r -= w; if (r <= 0) return k; } return TYPES.hatch; };
export const STIB_BUS = 0xc9ccd0;
export const POLICE_WHITE = 0xf1f2ef, AMBULANCE_YELLOW = 0xf0d21a;
export const VAN_WHITE = 0xeceeee, TRAM_SILVER = 0xb9bdc0, TRAIN = 0xcfd2d4;
export const ACCENTS = [0x8c1c1c, 0x1f3f7a, 0x3c6fb0, 0xc9b99a];   // a few real colours the aerial palette misses
