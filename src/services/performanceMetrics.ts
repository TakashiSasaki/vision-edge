// Successful-event span estimator, deliberately independent of React and RAF.
export const METRIC_WINDOW_MS = 5000;
export const MIN_RATE_SAMPLES = 2;
export const MIN_DURATION_SAMPLES = 1;
export const MIN_RATE_SPAN_MS = 250;
export const MAX_METRIC_SAMPLES = 300;
export const NO_INPUT_TIMEOUT_MS = 2000;
export const HUD_INTERVAL_MS = 250;
export interface PerformanceSnapshot {
  inferenceFps: number | null;
  drawFps: number | null;
  processingMs: number | null;
  stalled: boolean;
}
export class PerformanceMetrics {
  private inference: number[] = [];
  private draws: number[] = [];
  private durations: Array<{ at: number; ms: number }> = [];
  private lastClock: number | null = null;
  private lastInput: number | null = null;
  private started: number | null = null;
  constructor(private clock: () => number = () => performance.now()) { this.now(); }
  private now(): number | null {
    const now = this.clock();
    if (!Number.isFinite(now) || now < 0) return null;
    if (this.lastClock !== null && now < this.lastClock) this.reset();
    this.started ??= now;
    this.lastClock = now;
    this.inference = this.inference.filter(t => now - t <= METRIC_WINDOW_MS);
    this.draws = this.draws.filter(t => now - t <= METRIC_WINDOW_MS);
    this.durations = this.durations.filter(d => now - d.at <= METRIC_WINDOW_MS);
    return now;
  }
  reset(): void {
    this.inference = []; this.draws = []; this.durations = [];
    this.lastInput = null; this.lastClock = null; this.started = null;
  }
  input(): void { const now = this.now(); if (now !== null) this.lastInput = now; }
  private append(events: number[], now: number): void {
    if (events.length && now <= events[events.length - 1]) return;
    events.push(now);
    if (events.length > MAX_METRIC_SAMPLES) events.shift();
  }
  inferenceCompleted(start: number, end: number): void {
    const now = this.now();
    if (now === null || !Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end < start || end > now) return;
    this.append(this.inference, now);
    if (end > start) this.durations.push({ at: now, ms: end - start });
    if (this.durations.length > MAX_METRIC_SAMPLES) this.durations.shift();
  }
  drawCompleted(): void { const now = this.now(); if (now !== null) this.append(this.draws, now); }
  snapshot(): PerformanceSnapshot {
    const now = this.now();
    const rate = (events: number[]) => {
      if (now === null || events.length < MIN_RATE_SAMPLES || now - events[events.length - 1] >= NO_INPUT_TIMEOUT_MS) return null;
      const span = events[events.length - 1] - events[0];
      return span >= MIN_RATE_SPAN_MS ? 1000 * (events.length - 1) / span : null;
    };
    return {
      inferenceFps: rate(this.inference), drawFps: rate(this.draws),
      processingMs: now !== null && this.durations.length >= MIN_DURATION_SAMPLES && now - this.durations[this.durations.length - 1].at < NO_INPUT_TIMEOUT_MS
        ? this.durations.reduce((mean, d) => mean + d.ms / this.durations.length, 0) : null,
      stalled: now !== null && (this.lastInput ?? this.started) !== null && now - (this.lastInput ?? this.started!) >= NO_INPUT_TIMEOUT_MS,
    };
  }
}
