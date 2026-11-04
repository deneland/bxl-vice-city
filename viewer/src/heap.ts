/** binary min-heap of values by numeric key, in parallel arrays (no allocation per entry). An optional numeric
 *  tag travels with each value (e.g. a version, to recognise stale entries): poppedTag after pop(). */
export class MinHeap<T> {
  private k: number[] = []; private v: T[] = []; private g: number[] = [];
  /** tag of the value pop() returned last */
  poppedTag = 0;
  get size() { return this.k.length; }
  clear() { this.k.length = this.v.length = this.g.length = 0; }
  peek() { return this.k.length ? this.k[0] : Infinity; }
  push(key: number, val: T, tag = 0) {
    const { k, v, g } = this;
    let i = k.length;
    k.push(key); v.push(val); g.push(tag);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (k[p] <= key) break;
      k[i] = k[p]; v[i] = v[p]; g[i] = g[p]; i = p;
    }
    k[i] = key; v[i] = val; g[i] = tag;
  }
  pop(): T | undefined {
    const { k, v, g } = this;
    if (!k.length) return undefined;
    const top = v[0];
    this.poppedTag = g[0];
    const lk = k.pop()!, lv = v.pop()!, lg = g.pop()!;
    if (k.length) {
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= k.length) break;
        if (c + 1 < k.length && k[c + 1] < k[c]) c++;
        if (k[c] >= lk) break;
        k[i] = k[c]; v[i] = v[c]; g[i] = g[c]; i = c;
      }
      k[i] = lk; v[i] = lv; g[i] = lg;
    }
    return top;
  }
}
