// Dynamic resolution: fewer pixels when the GPU cannot keep up while the camera moves, more again once
// it is smooth. Judged on missed frames, and only kept if it helps.
import { MISSED, telemetry } from './telemetry';

export class DynamicResolution {
  /** device pixels per CSS pixel to render at */
  ratio: number;
  private acc = 0; private n = 0; private cpu = 0; private slow = 0; private good = 0; private at = performance.now();
  private lowered: { from: number; slow: number } | null = null;
  private noHelpUntil = 0;

  /** maxRatio: the most pixels wanted (screen and budget); apply: re-size to `ratio` */
  constructor(private maxRatio: () => number, private apply: () => void, private enabled = true) {
    this.ratio = maxRatio();
  }

  /** the window changed size: never more than the new maximum */
  clamp() { this.ratio = Math.min(this.ratio, this.maxRatio()); }

  /** a rendered frame: its interval and main-thread time (ms); busy: loading / re-planning (not judged then)
   *
   * Averages ~1 s of moving frames. Only lowered when the GPU is what is slow: if the main thread's own work
   * takes most of the frame, fewer pixels would not help (a real laptop lost half its resolution to the
   * start-up work and kept it). */
  frame(interval: number, moving: boolean, now: number, cpu: number, busy: boolean) {
    if (!this.enabled) return;
    if (moving) { this.acc += interval; this.cpu += cpu; this.n++; if (interval > MISSED) this.slow++; }
    if (now - this.at < 1000) return;
    this.at = now;
    const n = this.n;
    if (n < 20) { this.acc = this.cpu = this.n = this.slow = 0; return; }   // not enough moving frames to judge
    const avg = this.acc / n, cpuAvg = this.cpu / n, slow = this.slow / n;
    this.acc = this.cpu = this.n = this.slow = 0;
    const max = this.maxRatio(), min = Math.max(0.5, max * 0.6);
    const set = (r: number, reverted = false) => {
      this.ratio = r;
      this.apply();
      telemetry.event('resolution', { ratio: +r.toFixed(2), avg: +avg.toFixed(1), slow: +slow.toFixed(2), ...(reverted ? { reverted } : {}) });
    };
    // did the last step down help? If the slow share did not drop by a quarter, the GPU is not limited by
    // pixels (a real Intel laptop: geometry-bound, 25 % slow at 0.6 as at 1.0): back up, and leave it
    if (this.lowered) {
      const helped = slow < this.lowered.slow * 0.75;
      if (!helped) { set(this.lowered.from, true); this.noHelpUntil = now + 120_000; }
      this.lowered = null;
      if (!helped) return;
    }
    // What you notice is missed frames (a 60 Hz display shows one as a 33 ms stutter), not the average.
    // (Frames of 20-25 ms followed by shorter ones are callback jitter, not misses: Firefox on the Intel test
    // laptop shows ~25 % of those while missing only ~2 %.) More than 8 % missed while moving, CPU not the
    // bottleneck -> 10 % fewer pixels per side.
    if ((slow > 0.08 || avg > 22) && this.ratio > min && !busy && cpuAvg < avg * 0.6 && now > this.noHelpUntil) {
      this.lowered = { from: this.ratio, slow };
      this.good = 0;
      set(Math.max(min, this.ratio * 0.9));
    } else if (slow < 0.02 && avg < 17.5 && this.ratio < max) {   // smooth for a while: try more pixels
      if (++this.good >= 3) { this.good = 0; set(Math.min(max, this.ratio * 1.1)); }
    } else this.good = 0;
  }
}
