// The voxel tile records written by pipeline/voxelize.py (its constants are in pipeline/common.py):
// 8 bytes per voxel: x+1, y+1, z, r, g, b, LiDAR class, flags (x / y tile-local with a 1-voxel apron).
// On disk tiles are compact and gzipped (.vxt, common.encode_tile); decodeTile turns them back into records.
export const FLAG_FILL = 1;      // geometry synthesised (wall / terrain fill), not measured
export const FLAG_APRON = 2;     // belongs to the neighbouring tile (culling only)
export const FLAG_PHOTO = 8;     // colour from street-level photos
export const C_GROUND = 2;       // LiDAR (ASPRS) ground class

const MAGIC = [0x56, 0x58, 0x54, 0x03];

/** a manifest level's tile list: version 4 stores it as columns with the file names implied
 *  (pipeline common.pack_level_tiles); older manifests as a list of objects. Returns the objects. */
export function levelTiles(level: { level: number; size: number[]; tiles: unknown }, tileSize: number) {
  const t = level.tiles as { at: number[]; zmin: number[]; zh: number[] } | { file: string; tx: number; ty: number; zmin: number; zmax: number }[];
  if (Array.isArray(t)) return t;
  const W = Math.ceil(level.size[0] / tileSize), out = [];
  let at = 0;
  for (let i = 0; i < t.at.length; i++) {
    at += t.at[i];
    const tx = at % W, ty = (at - tx) / W;
    out.push({ file: `L${level.level}/t_${tx}_${ty}.vxt`, tx, ty, zmin: t.zmin[i], zmax: t.zmin[i] + t.zh[i] });
  }
  return out;
}   // "VXT\x03"

/** fetch a tile and decode it (a page instead of a tile, e.g. a server's fallback for a missing file, is an error) */
export async function fetchTile(url: string): Promise<Uint8Array> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`tile ${url}: HTTP ${r.status}`);
  if (r.headers.get('content-type')?.startsWith('text/html')) throw new Error(`tile ${url}: got a web page`);
  return decodeTile(await r.arrayBuffer());
}

/** a fetched tile file -> records (8 bytes per voxel). Accepts the compact format gzipped (or already
 *  unzipped, if a server sent it with Content-Encoding: gzip) and the raw records of version 2 builds. */
export async function decodeTile(buf: ArrayBuffer): Promise<Uint8Array> {
  let b = new Uint8Array(buf);
  if (b[0] === 0x1f && b[1] === 0x8b)   // (gzip: a raw record never starts so, x + 1 <= 65)
    b = new Uint8Array(await new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  if (!MAGIC.every((m, i) => b[i] === m)) return b;
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const width = b[4], P = dv.getUint16(6, true), n = dv.getUint32(8, true);
  const pal = 12;
  let o = pal + 3 * P;
  const rec = new Uint8Array(n * 8);
  // positions: varint differences of key = (x * width + y) * 256 + z (fits a double exactly)
  let key = 0;
  for (let i = 0; i < n; i++) {
    let d = 0, shift = 1, c: number;
    do { c = b[o++]; d += (c & 127) * shift; shift *= 128; } while (c & 128);
    key += d;
    const z = key % 256, xy = (key - z) / 256, y = xy % width;
    rec[i * 8] = (xy - y) / width; rec[i * 8 + 1] = y; rec[i * 8 + 2] = z;
  }
  const idx = o, cls = o + n, flags = o + 2 * n;
  for (let i = 0; i < n; i++) {
    const r = i * 8, p = pal + 3 * b[idx + i];
    rec[r + 3] = b[p]; rec[r + 4] = b[p + 1]; rec[r + 5] = b[p + 2];
    rec[r + 6] = b[cls + i]; rec[r + 7] = b[flags + i];
  }
  return rec;
}
