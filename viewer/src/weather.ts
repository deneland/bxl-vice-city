// Live weather over Schaerbeek (Open-Meteo, free, no key) and its look: overcast dims the sun and greys
// the sky, low visibility pulls the fog in, rain/snow are streaks around the camera.
import * as THREE from 'three/webgpu';
import { attribute, cameraPosition, fract, time, uniform, vec4 } from 'three/tsl';

export interface Weather {
  code: number;         // WMO weather code
  cloud: number;        // 0..1
  rain: number;         // mm/h (rain + showers)
  snow: number;         // cm/h
  visibility: number;   // m
  temp: number;         // °C
  windSpeed: number;    // m/s
  windDir: number;      // degrees, direction the wind comes FROM
}

const WMO: Record<number, string> = {
  0: 'clear', 1: 'mostly clear', 2: 'partly cloudy', 3: 'overcast', 45: 'fog', 48: 'fog',
  51: 'light drizzle', 53: 'drizzle', 55: 'heavy drizzle', 56: 'freezing drizzle', 57: 'freezing drizzle',
  61: 'light rain', 63: 'rain', 65: 'heavy rain', 66: 'freezing rain', 67: 'freezing rain',
  71: 'light snow', 73: 'snow', 75: 'heavy snow', 77: 'snow grains',
  80: 'rain showers', 81: 'rain showers', 82: 'violent showers', 85: 'snow showers', 86: 'snow showers',
  95: 'thunderstorm', 96: 'thunderstorm, hail', 99: 'thunderstorm, hail',
};
export const describe = (w: Weather) => WMO[w.code] ?? `code ${w.code}`;

export async function fetchWeather(lat: number, lon: number): Promise<Weather> {
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
    '&current=temperature_2m,weather_code,cloud_cover,rain,showers,snowfall,visibility,wind_speed_10m,wind_direction_10m';
  const r = await fetch(url);
  if (!r.ok) throw new Error(`open-meteo ${r.status}`);
  const c = (await r.json()).current;
  return { code: c.weather_code, cloud: c.cloud_cover / 100, rain: c.rain + c.showers, snow: c.snowfall,
    visibility: c.visibility, temp: c.temperature_2m, windSpeed: c.wind_speed_10m / 3.6, windDir: c.wind_direction_10m };
}

/** fixed weather for testing: ?weather=clear|overcast|fog|rain|storm|snow */
export function presetWeather(name: string): Weather {
  const base: Weather = { code: 0, cloud: 0, rain: 0, snow: 0, visibility: 50000, temp: 12, windSpeed: 3, windDir: 240 };
  const p: Record<string, Partial<Weather>> = {
    clear: {}, overcast: { code: 3, cloud: 1 }, fog: { code: 45, cloud: 1, visibility: 300 },
    rain: { code: 63, cloud: 1, rain: 2.5, visibility: 8000 }, storm: { code: 82, cloud: 1, rain: 12, visibility: 3000, windSpeed: 12 },
    snow: { code: 73, cloud: 1, snow: 1.5, visibility: 2000, temp: -1 },
  };
  return { ...base, ...(p[name] ?? {}) };
}

const GREY = new THREE.Color(0x9aa3ad);

/** adjust the clear-sky lighting for clouds and visibility */
export function applyWeather(w: Weather, l: { sky: THREE.Color; sunIntensity: number; hemiIntensity: number }) {
  const overcast = Math.min(1, w.cloud);
  l.sunIntensity *= 1 - 0.85 * overcast * overcast;       // broken clouds still let a lot of sun through
  const lum = l.sky.r * 0.3 + l.sky.g * 0.59 + l.sky.b * 0.11;
  l.sky.lerp(GREY.clone().multiplyScalar(lum / 0.6), 0.75 * overcast);
  l.hemiIntensity *= 1 + 0.25 * overcast;                  // diffuse light makes up for part of the sun
  // fog: visibility is where contrast drops to ~5%
  const far = Math.min(2600, Math.max(150, w.visibility));
  return { fogNear: far * 0.15, fogFar: far };
}

// --- precipitation: streaks in a box that follows the camera -------------------------
const N = 20000, BOX = 60;   // streaks, box size (m)

export class Precipitation {
  readonly mesh: THREE.LineSegments;
  private vel = uniform(new THREE.Vector3(0, -8, 0));
  private len = uniform(0.035);                            // streak length in seconds of fall
  private col = uniform(new THREE.Color(0xaab4c0));
  private alpha = uniform(0.35);

  constructor() {
    const seed = new Float32Array(N * 2 * 3), tip = new Float32Array(N * 2);
    for (let i = 0; i < N; i++) {
      const s = [Math.random(), Math.random(), Math.random()];
      seed.set(s, i * 6); seed.set(s, i * 6 + 3);
      tip[i * 2 + 1] = 1;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 2 * 3), 3));
    g.setAttribute('seed', new THREE.BufferAttribute(seed, 3));
    g.setAttribute('tip', new THREE.BufferAttribute(tip, 1));
    const m = new THREE.LineBasicNodeMaterial({ transparent: true, depthWrite: false });
    // position wraps inside a box centred on the camera, so the streaks never run out
    const s = attribute('seed', 'vec3').mul(BOX);
    const moved = s.add(this.vel.mul(time)).sub(cameraPosition);
    const p = fract(moved.div(BOX)).mul(BOX).sub(BOX / 2).add(cameraPosition);  // fract: same on WebGL and WebGPU for negatives
    m.positionNode = p.sub(this.vel.mul(this.len).mul(attribute('tip', 'float')));
    m.colorNode = vec4(this.col, this.alpha);
    m.fog = true;
    this.mesh = new THREE.LineSegments(g, m);
    this.mesh.frustumCulled = false;
    this.mesh.visible = false;
  }

  /** returns true while something falls (the scene must then be redrawn every frame) */
  set(w: Weather): boolean {
    const snow = w.snow > 0.05, rain = w.rain > 0.05;
    this.mesh.visible = snow || rain;
    if (!this.mesh.visible) return false;
    // wind blows towards (dir + 180°); scene x east, -z north
    const a = (w.windDir + 180) * Math.PI / 180, wind = Math.min(w.windSpeed, 15) * (snow ? 0.6 : 0.25);
    const fall = snow ? 1.2 : 8;
    this.vel.value.set(Math.sin(a) * wind, -fall, -Math.cos(a) * wind);
    this.len.value = snow ? 0.08 : 0.09;
    this.col.value.set(snow ? 0xffffff : 0xd0dae4);
    const amount = Math.min(1, (snow ? w.snow * 2 : w.rain) / 4);
    this.alpha.value = (snow ? 0.75 : 0.4) + 0.4 * amount;
    this.mesh.geometry.setDrawRange(0, 2 * Math.round(N * (0.25 + 0.75 * amount)));
    return true;
  }
}
