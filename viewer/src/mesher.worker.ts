// Turns one chunk of sparse voxels into merged visible faces (rectangles).
//
// Input voxel record (8 bytes, tileFormat.ts decodeTile): x+1, y+1, z, r, g, b, class, flags
//   x/y are chunk-local with a 1-voxel apron (0 and CH+1 are neighbours' voxels).
//
// Every visible face is found as before (one per exposed voxel side, skirts at tile edges, ambient
// occlusion per corner). Then faces of the same direction and plane are merged into rectangles (greedy):
// a wall 20 voxels high and 10 wide is one quad instead of 200. Each face keeps its own colour and
// occlusion: the quad points to its faces' texels (row by row) that the fragment shader reads
// (voxelMaterial.ts), so it looks the same.
//
// Output:
//   q: Float32 x3 per quad  [ x | y<<6 | z<<12 | dir<<20 ,  lu | lv<<8 ,  first texel (tile-local) ]
//      (first face; a rectangle of lu faces along u by lv along v; all < 2^24 so exact in float32)
//   t: 32 bits per face, v outer / u inner: colour r, g, b 6 bits each (sRGB), ambient occlusion 8 bits
//      (2 per corner), label 4 bits, photo 1 bit; as bytes:
//        [ r | g<<6 ,  g>>2 | b<<4 ,  b>>4 | ao<<2 ,  ao>>6 | label<<2 | photo<<6 ]
// dir: 0 +x, 1 -x, 2 +y, 3 -y, 4 +z (up), 5 -z (down)   (voxel space, z up)
// u (run axis): z for side faces, x for dir 4/5; v: y for dir 0/1, x for dir 2/3, y for dir 4/5.

export interface MeshRequest { id: number; url: string; ch: number; nz: number }   // (the worker fetches the tile itself)
export interface MeshResult { id: number; q: Float32Array; t: Uint8Array; quads: number; faces: number; failed?: boolean }

import { FLAG_APRON, FLAG_FILL, fetchTile } from './tileFormat';

const SKIRT = 8;   // voxels below a column's top that get side faces along the tile edge
const MAXRUN = 255;

// unit vectors per axis
const E = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

self.onmessage = async (ev: MessageEvent<MeshRequest>) => {
  const { id, url, ch, nz } = ev.data;
  let v: Uint8Array;
  try {
    v = await fetchTile(url);
  }
  catch { (self as unknown as Worker).postMessage({ id, q: new Float32Array(0), t: new Uint8Array(0), quads: 0, faces: 0, failed: true }); return; }
  const n = v.length / 8;
  const W = ch + 2;
  // occupancy: 0 empty, 1 measured, 2 fill (solid)
  const occ = new Uint8Array(W * W * nz);
  const at = (x: number, y: number, z: number) =>
    x < 0 || y < 0 || x >= W || y >= W || z < 0 || z >= nz ? 0 : occ[(z * W + y) * W + x];
  for (let i = 0; i < n; i++) {
    const o = i * 8;
    occ[(v[o + 2] * W + v[o + 1]) * W + v[o]] = v[o + 7] & FLAG_FILL ? 2 : 1;
  }

  // top of each column, for the skirts below
  const top = new Int16Array(W * W).fill(-1);
  for (let i = 0; i < n; i++) {
    const o = i * 8, k = v[o + 1] * W + v[o];
    if (v[o + 2] > top[k]) top[k] = v[o + 2];
  }

  // --- 1. all visible faces ---
  const cap = n * 6;
  const key = new Float64Array(cap);     // sort key: (direction, plane), then v, then u
  const groupKey = new Int32Array(cap), fu = new Uint16Array(cap), fv = new Uint16Array(cap);
  const fp = new Uint32Array(cap);       // x | y<<6 | z<<12 | dir<<20
  const ft = new Uint8Array(cap * 4);
  let f = 0;
  const p = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const o = i * 8;
    const flags = v[o + 7];
    if (flags & FLAG_APRON) continue;
    const x = v[o], y = v[o + 1], z = v[o + 2];
    for (let dir = 0; dir < 6; dir++) {
      const axis = dir >> 1;
      const s = dir & 1 ? -1 : 1;
      const e = E[axis];
      const nx = x + e[0] * s, ny = y + e[1] * s, nzz = z + e[2] * s;
      // Neighbouring tiles may be drawn at another level of detail, so a face hidden by the
      // neighbour's voxels (apron) can open a crack: near the top of the column, keep it (skirt).
      const apron = nx === 0 || ny === 0 || nx === W - 1 || ny === W - 1;
      if (at(nx, ny, nzz) && !(apron && axis < 2 && z >= top[y * W + x] - SKIRT)) continue;
      if (dir === 5 && (flags & FLAG_FILL || z === 0)) continue; // undersides of solids are never seen
      // ambient occlusion per corner, sampled in the layer in front of the face
      const eu = E[(axis + 1) % 3], ev2 = E[(axis + 2) % 3];
      let ao = 0;
      for (let b = 0; b < 2; b++) {
        for (let aa = 0; aa < 2; aa++) {
          const du = aa ? 1 : -1, dv = b ? 1 : -1;
          p[0] = nx + eu[0] * du; p[1] = ny + eu[1] * du; p[2] = nzz + eu[2] * du;
          const s1 = at(p[0], p[1], p[2]) ? 1 : 0;
          const s2 = at(nx + ev2[0] * dv, ny + ev2[1] * dv, nzz + ev2[2] * dv) ? 1 : 0;
          const cc = at(p[0] + ev2[0] * dv, p[1] + ev2[1] * dv, p[2] + ev2[2] * dv) ? 1 : 0;
          const lvl = s1 && s2 ? 0 : 3 - (s1 + s2 + cc);
          ao |= lvl << ((aa + 2 * b) * 2);
        }
      }
      const X = x - 1, Y = y - 1;
      fp[f] = X | (Y << 6) | (z << 12) | (dir << 20);
      // plane and in-plane coordinates (u: run axis, v: the other one); sort key: plane, then v, then u
      const plane = dir < 2 ? X : dir < 4 ? Y : z;
      const u = dir < 4 ? z : X, vv = dir < 2 ? Y : dir < 4 ? X : Y;
      groupKey[f] = dir * 512 + plane;
      fu[f] = u; fv[f] = vv;
      key[f] = (groupKey[f] * 256 + vv) * 256 + u;
      // texel: colour 6+6+6 bits, occlusion 8, label 4, photo 1 (see the header)
      const r6 = v[o + 3] >> 2, g6 = v[o + 4] >> 2, b6 = v[o + 5] >> 2, lab = flags >> 4, photo = (flags >> 3) & 1;
      ft[f * 4] = r6 | ((g6 & 3) << 6);
      ft[f * 4 + 1] = (g6 >> 2) | ((b6 & 15) << 4);
      ft[f * 4 + 2] = (b6 >> 4) | ((ao & 63) << 2);
      ft[f * 4 + 3] = (ao >> 6) | (lab << 2) | (photo << 6);
      f++;
    }
  }

  // --- 2. merge into rectangles, per direction and plane (greedy): u = the run axis (z for side faces,
  // x for top / bottom), v = the other axis in the plane (y for dir 0/1, x for dir 2/3, y for dir 4/5) ---
  const order = new Uint32Array(f);
  for (let i = 0; i < f; i++) order[i] = i;
  order.sort((a, b) => key[a] - key[b]);   // groups each (dir, plane), then v, then u
  const q = new Float32Array(f * 3);
  const t = new Uint8Array(f * 4);
  let quads = 0, texel = 0;
  const G = 256;                            // grid side (u, v < 256)
  const grid = new Int32Array(G * G).fill(-1);
  for (let k = 0; k < f;) {
    // one (dir, plane) group
    const g0 = k, gk = groupKey[order[k]];
    while (k < f && groupKey[order[k]] === gk) k++;
    for (let j = g0; j < k; j++) { const i = order[j]; grid[fu[i] + fv[i] * G] = i; }
    for (let j = g0; j < k; j++) {
      const i0 = order[j];
      const u0 = fu[i0], v0 = fv[i0];
      if (grid[u0 + v0 * G] !== i0) continue;           // already in a rectangle
      let lu = 1;
      while (u0 + lu < G && lu < MAXRUN && grid[u0 + lu + v0 * G] >= 0) lu++;
      let lv = 1;
      grow: while (v0 + lv < G && lv < MAXRUN) {
        for (let du = 0; du < lu; du++) if (grid[u0 + du + (v0 + lv) * G] < 0) break grow;
        lv++;
      }
      // texels row by row (v outer, u inner); the cells leave the grid
      for (let dv = 0; dv < lv; dv++)
        for (let du = 0; du < lu; du++) {
          const c = u0 + du + (v0 + dv) * G, fi = grid[c];
          t.set(ft.subarray(fi * 4, fi * 4 + 4), (texel + dv * lu + du) * 4);
          grid[c] = -1;
        }
      q[quads * 3] = fp[i0];
      q[quads * 3 + 1] = lu | (lv << 8);
      q[quads * 3 + 2] = texel;
      texel += lu * lv;
      quads++;
    }
  }
  const out: MeshResult = { id, q: q.slice(0, quads * 3), t: t.slice(0, f * 4), quads, faces: f };
  (self as unknown as Worker).postMessage(out, [out.q.buffer, out.t.buffer]);
};
