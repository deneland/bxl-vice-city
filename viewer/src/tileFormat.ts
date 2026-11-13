// The voxel tile records written by pipeline/voxelize.py (its constants are in pipeline/common.py):
// 8 bytes per voxel: x+1, y+1, z, r, g, b, LiDAR class, flags (x / y tile-local with a 1-voxel apron).
// On disk tiles are compact and gzipped (.vxt, common.encode_tile); decodeTile turns them back into records.
export const FLAG_FILL = 1;      // geometry synthesised (wall / terrain fill), not measured
export const FLAG_APRON = 2;     // belongs to the neighbouring tile (culling only)
export const FLAG_PHOTO = 8;     // colour from street-level photos
export const C_GROUND = 2;       // LiDAR (ASPRS) ground class

const MAGIC = [0x56, 0x58, 0x54, 0x03];

const ARCHIVE = 16;   // tiles per archive side (pipeline common.ARCHIVE)

export interface TileEntry { file: string; tx: number; ty: number; zmin: number; zmax: number; off?: number; len?: number }

/** a manifest level's tile list. Version 5 (pipeline common.pack_level_tiles): columns, file names implied,
 *  tiles inside archives of ARCHIVE x ARCHIVE (each tile's offset = the lengths before it in its archive);
 *  version 4: the same without archives; older: a list of objects. */
export function levelTiles(level: { level: number; size: number[]; tiles: unknown }, tileSize: number): TileEntry[] {
  const t = level.tiles as { at: number[]; zmin: number[]; zh: number[]; len?: number[] } | TileEntry[];
  if (Array.isArray(t)) return t;
  const W = Math.ceil(level.size[0] / tileSize), out: TileEntry[] = [], ends = new Map<string, number>();
  let at = 0;
  for (let i = 0; i < t.at.length; i++) {
    at += t.at[i];
    const tx = at % W, ty = (at - tx) / W, e: TileEntry = { file: '', tx, ty, zmin: t.zmin[i], zmax: t.zmin[i] + t.zh[i] };
    if (t.len) {
      e.file = `L${level.level}/a_${Math.floor(tx / ARCHIVE)}_${Math.floor(ty / ARCHIVE)}.vxa`;
      e.off = ends.get(e.file) ?? 0; e.len = t.len[i];
      ends.set(e.file, e.off + e.len);
    } else e.file = `L${level.level}/t_${tx}_${ty}.vxt`;
    out.push(e);
  }
  return out;
}

/** where a tile is: its file (or archive) URL, the byte range in an archive, and the Cache Storage cache
 *  that keeps fetched tiles for this build (null: no caching) */
export interface TileRef { url: string; off?: number; len?: number; cache: string | null }   // "VXT\x03"

/** fetch a tile and decode it: from the build's cache if it holds it, else from the network (a range request
 *  into its archive), kept in the cache for the next visit. A static host like GitHub Pages only lets the
 *  browser keep files for minutes; the cache keeps them until the build changes. A page instead of a tile
 *  (a server's fallback for a missing file) is an error. */
export async function fetchTile(ref: TileRef): Promise<Uint8Array> {
  const ranged = ref.len !== undefined && ref.off !== undefined;
  const key = ranged ? `${ref.url}${ref.url.includes('?') ? '&' : '?'}at=${ref.off}` : ref.url;
  let cache: Cache | null = null;
  if (ref.cache && typeof caches !== 'undefined') {
    try {
      cache = await caches.open(ref.cache);
      const hit = await cache.match(key);
      if (hit) return decodeTile(await hit.arrayBuffer());
    } catch { cache = null; }   // (no Cache Storage here, e.g. not a secure context: network only)
  }
  const r = await fetch(ref.url, ranged ? { headers: { Range: `bytes=${ref.off}-${ref.off! + ref.len! - 1}` } } : undefined);
  if (!r.ok) throw new Error(`tile ${ref.url}: HTTP ${r.status}`);
  if (r.headers.get('content-type')?.startsWith('text/html')) throw new Error(`tile ${ref.url}: got a web page`);
  let buf = await r.arrayBuffer();
  if (ranged && r.status === 200) buf = buf.slice(ref.off!, ref.off! + ref.len!);   // (the server ignored the range)
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
