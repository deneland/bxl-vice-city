/** Points (x, z, in metres) in square cells, for "nearest" and "everything within r" questions: kerb and
 *  pavement spots, parking places. Numeric cell keys, no allocation per query. */
export class PointGrid<T> {
  private cells = new Map<number, number[]>();   // cell -> indices into items
  private xs: number[] = []; private zs: number[] = [];
  readonly items: T[] = [];

  constructor(readonly cell: number) {}

  private static key(i: number, j: number) { return (i + 32768) * 65536 + (j + 32768); }

  add(x: number, z: number, item: T) {
    const k = PointGrid.key(Math.floor(x / this.cell), Math.floor(z / this.cell));
    let c = this.cells.get(k);
    if (!c) this.cells.set(k, (c = []));
    c.push(this.items.length);
    this.items.push(item); this.xs.push(x); this.zs.push(z);
  }

  /** the nearest item within max m that `accept` takes (all if none given), or null. Rings of cells outward
   *  from the nearest; stops as soon as no cell further out can hold a nearer item. */
  nearest(x: number, z: number, max: number, accept?: (item: T) => boolean): T | null {
    const C = this.cell, R = Math.ceil(max / C), cx = Math.floor(x / C), cz = Math.floor(z / C);
    let best = -1, bd = max * max;
    for (let r = 0; r <= R; r++) {
      const near = Math.max(0, r - 1) * C;   // every item in ring r is at least this far
      if (near * near >= bd) break;
      for (let i = cx - r; i <= cx + r; i++)
        for (let j = cz - r; j <= cz + r; j++) {
          if (Math.max(Math.abs(i - cx), Math.abs(j - cz)) !== r) continue;   // (ring only)
          const c = this.cells.get(PointGrid.key(i, j));
          if (!c) continue;
          for (const k of c) {
            const d = (this.xs[k] - x) ** 2 + (this.zs[k] - z) ** 2;
            if (d < bd && (!accept || accept(this.items[k]))) { bd = d; best = k; }
          }
        }
    }
    return best >= 0 ? this.items[best] : null;
  }

  /** every item within r m of (x, z) */
  forEachWithin(x: number, z: number, r: number, fn: (item: T) => void) {
    const C = this.cell, n = Math.ceil(r / C), cx = Math.floor(x / C), cz = Math.floor(z / C), r2 = r * r;
    for (let i = cx - n; i <= cx + n; i++)
      for (let j = cz - n; j <= cz + n; j++)
        for (const k of this.cells.get(PointGrid.key(i, j)) ?? [])
          if ((this.xs[k] - x) ** 2 + (this.zs[k] - z) ** 2 < r2) fn(this.items[k]);
  }
}
