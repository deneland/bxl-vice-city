// Voxel tiles as pipeline/voxelize.py writes them (format: pipeline/common.py encode_tile, pack_archives).
// Decoded, a tile is 8 bytes per voxel: x+1, y+1, z, r, g, b, LiDAR class, flags (x / y tile-local with a
// 1-voxel apron). On disk each tile is compact and gzipped, and tiles sit in archives of ARCHIVE x ARCHIVE,
// read one tile at a time with an HTTP range request.
export const FLAG_FILL = 1;      // geometry synthesised (wall / terrain fill), not measured
export const FLAG_APRON = 2;     // belongs to the neighbouring tile (culling only)
export const FLAG_PHOTO = 8;     // colour from street-level photos
export const C_GROUND = 2;       // LiDAR (ASPRS) ground class

const MAGIC = [0x56, 0x58, 0x54, 0x03];   // "VXT\x03"
const ARCHIVE = 16;                        // tiles per archive side (common.ARCHIVE)

export interface TileEntry { file: string; off: number; len: number; tx: number; ty: number; zmin: number; zmax: number }

/** a manifest level's tile list (common.pack_level_tiles: columns, archive names implied; a tile's offset is
 *  the sum of the lengths before it in its archive) -> one entry per tile */
export function levelTiles(level: { level: number; size: number[]; tiles: unknown }, tileSize: number): TileEntry[] {
  const t = level.tiles as { at: number[]; zmin: number[]; zh: number[]; len: number[] };
  const W = Math.ceil(level.size[0] / tileSize), out: TileEntry[] = [], ends = new Map<string, number>();
  let at = 0;
  for (let i = 0; i < t.at.length; i++) {
    at += t.at[i];
    const tx = at % W, ty = (at - tx) / W;
    const file = `L${level.level}/a_${Math.floor(tx / ARCHIVE)}_${Math.floor(ty / ARCHIVE)}.vxa`, off = ends.get(file) ?? 0;
    out.push({ file, off, len: t.len[i], tx, ty, zmin: t.zmin[i], zmax: t.zmin[i] + t.zh[i] });
    ends.set(file, off + t.len[i]);
  }
  return out;
}

/** a tile: its archive's URL, its byte range there, and the Cache Storage cache keeping this build's tiles
 *  (null: no caching) */
export interface TileRef { url: string; off: number; len: number; cache: string | null }

/** fetch a tile and decode it: from the build's cache if it holds it, else with a range request into its
 *  archive, then kept in the cache for the next visit (a static host like GitHub Pages lets the browser keep
 *  files for minutes only). A page instead of a tile (a server's fallback for a missing file) is an error. */
export async function fetchTile(ref: TileRef): Promise<Uint8Array> {
  const key = `${ref.url}${ref.url.includes('?') ? '&' : '?'}at=${ref.off}`;
  let cache: Cache | null = null;
  if (ref.cache && typeof caches !== 'undefined') {
    try {
      cache = await caches.open(ref.cache);
      const hit = await cache.match(key);
      if (hit) return decodeTile(await hit.arrayBuffer());
    } catch { cache = null; }   // (no Cache Storage here, e.g. not a secure context: network only)
  }
  const r = await fetch(ref.url, { headers: { Range: `bytes=${ref.off}-${ref.off + ref.len - 1}` } });
  if (!r.ok) throw new Error(`tile ${ref.url}: HTTP ${r.status}`);
  if (r.headers.get('content-type')?.startsWith('text/html')) throw new Error(`tile ${ref.url}: got a web page`);
  let buf = await r.arrayBuffer();
  if (r.status === 200) buf = buf.slice(ref.off, ref.off + ref.len);   // (a server without range requests)
  if (cache) cache.put(key, new Response(buf.slice(0))).catch(() => {});
  return decodeTile(buf);
}

/** the Cache Storage cache for a build's tiles; caches of other builds of the region are deleted */
export function tileCache(region: string, build: string | undefined): string | null {
  if (!build || typeof caches === 'undefined') return null;
  const name = `bxl-tiles-${region}-${build}`;
  caches.keys().then((keys) => {
    for (const k of keys) if (k.startsWith(`bxl-tiles-${region}-`) && k !== name) caches.delete(k);
  }).catch(() => {});
  return name;
}

/** a tile (common.encode_tile, gzipped) -> records */
async function decodeTile(buf: ArrayBuffer): Promise<Uint8Array> {
  let b = new Uint8Array(buf);
  if (b[0] === 0x1f && b[1] === 0x8b)   // (not yet unzipped by the browser: no Content-Encoding from the server)
    b = new Uint8Array(await new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  if (!MAGIC.every((m, i) => b[i] === m)) throw new Error('not a voxel tile');
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
