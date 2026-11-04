// Turns one chunk of sparse voxels into merged visible faces ("runs").
//
// Input voxel record (8 bytes): x+1, y+1, z, r, g, b, class, flags
//   x/y are chunk-local with a 1-voxel apron (0 and CH+1 are neighbours' voxels).
//
// Every visible face is found as before (one per exposed voxel side, skirts at tile edges, ambient
// occlusion per corner). Then consecutive faces of the same direction are merged into one quad: side faces in vertical runs (walls), top / bottom faces in runs along x. A wall of
// 20 voxels is one quad instead of 20. Each face keeps its own colour and occlusion: the quad points to a
// run of texels (one per face) that the fragment shader reads (voxelMaterial.ts), so it looks the same.
//
// Output:
//   q: Float32 x3 per quad  [ x | y<<6 | z<<12 | dir<<20 ,  len<<8 ,  first texel (tile-local) ]
//      (first face of the run; all < 2^24 so exact in float32)
//   t: 32 bits per face, in run order: colour r, g, b 6 bits each (sRGB), ambient occlusion 8 bits (2 per
//      corner), label 4 bits, photo 1 bit; as bytes:
//        [ r | g<<6 ,  g>>2 | b<<4 ,  b>>4 | ao<<2 ,  ao>>6 | label<<2 | photo<<6 ]
// dir: 0 +x, 1 -x, 2 +y, 3 -y, 4 +z (up), 5 -z (down)   (voxel space, z up)
// Runs go along z for side faces (dir 0-3) and along x for dir 4/5, at most 255 faces.

export interface MeshRequest { id: number; buf: ArrayBuffer; ch: number; nz: number }
export interface MeshResult { id: number; q: Float32Array; t: Uint8Array; quads: number; faces: number }

const FLAG_FILL = 1;
const FLAG_APRON = 2;
const SKIRT = 8;   // voxels below a column's top that get side faces along the tile edge
const MAXRUN = 255;

// unit vectors per axis
const E = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

self.onmessage = (ev: MessageEvent<MeshRequest>) => {
  const { id, buf, ch, nz } = ev.data;
  const v = new Uint8Array(buf);
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
  const key = new Float64Array(cap);     // sort key: direction, then the cross coordinates, then the run coordinate
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
      // runs: side faces along z, top / bottom faces along x; consecutive faces have consecutive keys
      key[f] = dir < 4 ? ((dir * 64 + X) * 64 + Y) * 512 + z : ((dir * 512 + z) * 64 + Y) * 128 + X;   // (gaps: runs never wrap into the next row)
      // texel: colour 6+6+6 bits, occlusion 8, label 4, photo 1 (see the header)
      const r6 = v[o + 3] >> 2, g6 = v[o + 4] >> 2, b6 = v[o + 5] >> 2, lab = flags >> 4, photo = (flags >> 3) & 1;
      ft[f * 4] = r6 | ((g6 & 3) << 6);
      ft[f * 4 + 1] = (g6 >> 2) | ((b6 & 15) << 4);
      ft[f * 4 + 2] = (b6 >> 4) | ((ao & 63) << 2);
      ft[f * 4 + 3] = (ao >> 6) | (lab << 2) | (photo << 6);
      f++;
    }
  }

  // --- 2. merge runs ---
  const order = new Uint32Array(f);
  for (let i = 0; i < f; i++) order[i] = i;
  order.sort((a, b) => key[a] - key[b]);
  const q = new Float32Array(f * 3);
  const t = new Uint8Array(f * 4);
  let quads = 0;
  for (let k = 0; k < f;) {
    const first = order[k];
    let len = 1;
    while (k + len < f && len < MAXRUN) {
      const nxt = order[k + len];
      if (key[nxt] !== key[first] + len) break;
      len++;
    }
    q[quads * 3] = fp[first];
    q[quads * 3 + 1] = len << 8;
    q[quads * 3 + 2] = k;   // its faces' texels are k .. k+len-1 (written below in sorted order)
    for (let j = 0; j < len; j++) t.set(ft.subarray(order[k + j] * 4, order[k + j] * 4 + 4), (k + j) * 4);
    quads++;
    k += len;
  }
  const out: MeshResult = { id, q: q.slice(0, quads * 3), t: t.slice(0, f * 4), quads, faces: f };
  (self as unknown as Worker).postMessage(out, [out.q.buffer, out.t.buffer]);
};
