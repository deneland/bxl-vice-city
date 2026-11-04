// Polylines as flat xyz arrays with cumulative lengths: the lanes, curves and footpaths everything moves on.

/** cumulative length at each point of a flat xyz polyline (flat: horizontal distance only) */
export function cumulative(pts: ArrayLike<number>, flat = false): Float32Array {
  const n = pts.length / 3, cum = new Float32Array(n);
  for (let i = 1; i < n; i++) {
    const dx = pts[i * 3] - pts[i * 3 - 3], dy = flat ? 0 : pts[i * 3 + 1] - pts[i * 3 - 2], dz = pts[i * 3 + 2] - pts[i * 3 - 1];
    cum[i] = cum[i - 1] + Math.hypot(dx, dy, dz);
  }
  return cum;
}

/** the point at distance s along the polyline (clamped to its ends) */
export function pointAt<V extends { set(x: number, y: number, z: number): V }>(pts: ArrayLike<number>, cum: Float32Array, s: number, out: V): V {
  s = Math.min(Math.max(s, 0), cum[cum.length - 1]);
  let lo = 0, hi = cum.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cum[m] <= s) lo = m; else hi = m; }
  const d = cum[hi] - cum[lo], k = d > 0 ? (s - cum[lo]) / d : 0;
  return out.set(pts[lo * 3] + (pts[hi * 3] - pts[lo * 3]) * k, pts[lo * 3 + 1] + (pts[hi * 3 + 1] - pts[lo * 3 + 1]) * k,
    pts[lo * 3 + 2] + (pts[hi * 3 + 2] - pts[lo * 3 + 2]) * k);
}
