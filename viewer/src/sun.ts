// Sun position for a place and time (NOAA simplified equations, ~0.1° accuracy is plenty here)
// and the matching light/sky settings.
import * as THREE from 'three/webgpu';

const RAD = Math.PI / 180;

/** azimuth (radians, clockwise from north) and altitude (radians) of the sun */
export function sunPosition(date: Date, lat: number, lon: number) {
  const day = (date.getTime() - Date.UTC(date.getUTCFullYear(), 0, 1)) / 86400000; // fractional day of year, UTC
  const g = (2 * Math.PI / 365) * (day - 0.5);
  const eqTime = 229.18 * (0.000075 + 0.001868 * Math.cos(g) - 0.032077 * Math.sin(g)
    - 0.014615 * Math.cos(2 * g) - 0.040849 * Math.sin(2 * g));                     // minutes
  const decl = 0.006918 - 0.399912 * Math.cos(g) + 0.070257 * Math.sin(g) - 0.006758 * Math.cos(2 * g)
    + 0.000907 * Math.sin(2 * g) - 0.002697 * Math.cos(3 * g) + 0.00148 * Math.sin(3 * g);
  const utcMin = date.getUTCHours() * 60 + date.getUTCMinutes() + date.getUTCSeconds() / 60;
  const ha = ((utcMin + eqTime + 4 * lon) / 4 - 180) * RAD;                         // hour angle
  const phi = lat * RAD;
  const sinAlt = Math.sin(phi) * Math.sin(decl) + Math.cos(phi) * Math.cos(decl) * Math.cos(ha);
  const alt = Math.asin(sinAlt);
  const az = Math.atan2(Math.sin(ha), Math.cos(ha) * Math.sin(phi) - Math.tan(decl) * Math.cos(phi)) + Math.PI;
  return { az, alt };
}

/** unit vector towards the sun in scene space (x east, y up, -z north) */
export function sunDirection(az: number, alt: number, out = new THREE.Vector3()) {
  return out.set(Math.sin(az) * Math.cos(alt), Math.sin(alt), -Math.cos(az) * Math.cos(alt));
}

const smooth = (a: number, b: number, x: number) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

const SKY_DAY = new THREE.Color(0xbcd3e6), SKY_GOLD = new THREE.Color(0xe9b98f), SKY_NIGHT = new THREE.Color(0x0c1424);
const SUN_WHITE = new THREE.Color(0xfff1dc), SUN_LOW = new THREE.Color(0xff9a4a);
const HEMI_DAY = new THREE.Color(0xdfeeff), HEMI_NIGHT = new THREE.Color(0x7083b8);

/** light and sky for a sun altitude (radians) */
export function lighting(alt: number) {
  const deg = alt / RAD;
  const day = smooth(-6, 8, deg);       // civil twilight -> full day
  const low = 1 - smooth(2, 20, deg);   // golden hour
  const sky = SKY_NIGHT.clone().lerp(SKY_DAY, day).lerp(SKY_GOLD, low * day * 0.55);
  return {
    sky,
    sunColor: SUN_WHITE.clone().lerp(SUN_LOW, low),
    sunIntensity: 3.2 * smooth(-1, 6, deg),
    hemiColor: HEMI_NIGHT.clone().lerp(HEMI_DAY, day),
    hemiIntensity: 0.8 + 0.8 * day,    // night: moonlight-ish, the city stays readable
  };
}
