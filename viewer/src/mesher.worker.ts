// Turns one chunk of sparse voxels into packed visible faces.
//
// Input voxel record (8 bytes): x+1, y+1, z, r, g, b, class, flags
//   x/y are chunk-local with a 1-voxel apron (0 and CH+1 are neighbours' voxels).
// Output per face:
//   a: Float32 x2  [ x | y<<6 | z<<12 | dir<<20 ,  ao | class<<8 | fill<<16 | label<<17 ]
//      | photo<<21   (all < 2^24 so they are exact in float32)
//   c: Uint8 x4    [ r, g, b, 255 ]
// dir: 0 +x, 1 -x, 2 +y, 3 -y, 4 +z (up), 5 -z (down)   (voxel space, z up)

export interface MeshRequest { id: number; buf: ArrayBuffer; ch: number; nz: number }
export interface MeshResult { id: number; a: Float32Array; c: Uint8Array; faces: number }

const FLAG_FILL = 1;
const FLAG_APRON = 2;
const SKIRT = 8;   // voxels below a column's top that get side faces along the tile edge

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

  const a = new Float32Array(n * 6 * 2);
  const c = new Uint8Array(n * 6 * 4);
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
      a[f * 2] = (x - 1) | ((y - 1) << 6) | (z << 12) | (dir << 20);
      a[f * 2 + 1] = ao | (v[o + 6] << 8) | ((flags & FLAG_FILL) << 16) | ((flags >> 4) << 17) | (((flags >> 3) & 1) << 21);
      c[f * 4] = v[o + 3];
      c[f * 4 + 1] = v[o + 4];
      c[f * 4 + 2] = v[o + 5];
      c[f * 4 + 3] = 255;
      f++;
    }
  }
  const out: MeshResult = { id, a: a.slice(0, f * 2), c: c.slice(0, f * 4), faces: f };
  (self as unknown as Worker).postMessage(out, [out.a.buffer, out.c.buffer]);
};
