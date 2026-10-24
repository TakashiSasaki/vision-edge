// JS wall intervals only. No CPU/GPU utilization or physical presentation claim.
export const PIPELINE_WINDOW_MS = 5000;
export const PIPELINE_EVENT_CAP = 2048;
export const PIPELINE_QUANTILE_CAP = 300;
export const PIPELINE_JOB_CAP = 64;
export const PIPELINE_STAGES = ['capture', 'comparison', 'queue', 'job', 'sdk', 'postprocess', 'cleanup', 'engine', 'draw', 'latency'] as const;
export type PipelineStage = typeof PIPELINE_STAGES[number];
export type AttemptOutcome = 'success' | 'failed' | 'cancelled' | 'stale';
export type ResultOutcome = 'first-draw' | 'overwritten' | 'cancelled' | 'hidden' | 'draw-failed' | 'expired' | 'capacity' | 'unbound';
export interface PipelineEvent { generation?: symbol; stage: PipelineStage; outcome: AttemptOutcome; at: number; ms: number | null }
export interface PipelineObserver { generation?: symbol; now: () => number; record: (event: PipelineEvent) => void }
interface Stamp { at: number; epoch: number }
// The only observer boundary used by product code. A disabled probe is undefined:
// callers bypass clock reads, token/event allocation, and aggregation entirely.
export class PipelineProbe {
  private last = -1;
  private epoch = 0;
  readonly generation: symbol | undefined;
  constructor(private observer: PipelineObserver) {
    try { this.generation = observer.generation; } catch { this.generation = undefined; }
  }
  start(): Stamp | undefined {
    try {
      const at = this.observer.now();
      if (!Number.isFinite(at) || at < 0 || at < this.last) { this.epoch++; return; }
      this.last = at;
      return { at, epoch: this.epoch };
    } catch { this.epoch++; return; }
  }
  elapsed(start: Stamp | undefined): { at: number; ms: number | null } | undefined {
    const end = this.start();
    if (!end) return;
    return { at: end.at, ms: start && start.epoch === end.epoch && end.at >= start.at ? end.at - start.at : null };
  }
  duration(stage: PipelineStage, ms: number | null, outcome: AttemptOutcome): void {
    const end = this.start();
    try { this.observer.record({ generation: this.generation, stage, outcome, at: end?.at ?? NaN, ms }); } catch { /* Missing telemetry only. */ }
  }
  end(stage: PipelineStage, start: Stamp | undefined, outcome: AttemptOutcome = 'success'): number | undefined {
    const end = this.elapsed(start);
    try { this.observer.record({ generation: this.generation, stage, outcome, at: end?.at ?? NaN, ms: end?.ms ?? null }); } catch { /* Telemetry cannot fail a product operation. */ }
    return end?.ms == null ? undefined : end.at;
  }
}
export interface InferenceMeasurement {
  generation: symbol;
  jobId: number;
  captureId: number;
  captureStart: number | null;
  captureEnd: number | null;
  immutable: boolean;
}
interface Timed { at: number; ms: number | null }
// Fixed capacity circular store. Overflow coverage is explicit until the last
// discarded event leaves the inclusive window; no bucket boundary approximation.
class WindowStore<T extends { at: number }> {
  private events: Array<T | undefined> = new Array(PIPELINE_EVENT_CAP);
  private next = 0;
  private size = 0;
  private droppedAt = -Infinity;
  add(event: T): void {
    if (this.size === PIPELINE_EVENT_CAP) this.droppedAt = this.events[this.next]!.at;
    this.events[this.next] = event;
    this.next = (this.next + 1) % PIPELINE_EVENT_CAP;
    this.size = Math.min(PIPELINE_EVENT_CAP, this.size + 1);
  }
  window(now: number): { events: T[]; truncated: boolean } {
    const events: T[] = [];
    for (let i = 0; i < this.size; i++) {
      const event = this.events[(this.next - this.size + i + PIPELINE_EVENT_CAP) % PIPELINE_EVENT_CAP]!;
      if (event.at >= now - PIPELINE_WINDOW_MS && event.at <= now) events.push(event);
    }
    return { events, truncated: this.droppedAt >= now - PIPELINE_WINDOW_MS };
  }
}
export interface StageStatistics {
  count: number; measured: number; missing: number; totalMs: number | null;
  mean: number | null; median: number | null; p95: number | null; max: number | null;
  quantileCount: number; truncated: boolean;
}
export interface PipelineSnapshot {
  stages: Record<PipelineStage, Record<AttemptOutcome, StageStatistics>>;
  results: Partial<Record<ResultOutcome, number>>;
  resultsTruncated: boolean;
  pendingResults: number;
  unclocked: number;
  captureDutyPercent: number | null;
}
const OUTCOMES: AttemptOutcome[] = ['success', 'failed', 'cancelled', 'stale'];
export class PipelineMetrics implements PipelineObserver {
  generation = Symbol('pipeline-generation');
  private last = -1;
  private unclocked = 0;
  private stores = new Map<string, WindowStore<Timed>>();
  private outcomes = new WindowStore<{ at: number; outcome: ResultOutcome }>();
  private jobs = new Map<number, { identity: InferenceMeasurement; accepted: number }>();
  constructor(private clock: () => number = () => performance.now()) {}
  now = (): number => this.clock();
  private time(): number | null {
    try {
      const now = this.clock();
      if (!Number.isFinite(now) || now < 0 || now < this.last) return null;
      this.last = now; return now;
    } catch { return null; }
  }
  reset(): void { this.generation = Symbol('pipeline-generation'); this.stores.clear(); this.jobs.clear(); this.outcomes = new WindowStore(); this.last = -1; this.unclocked = 0; }
  record = (event: PipelineEvent): void => {
    if (event.generation !== undefined && event.generation !== this.generation) return;
    if (!PIPELINE_STAGES.includes(event.stage) || !OUTCOMES.includes(event.outcome)) return;
    if (!Number.isFinite(event.at) || event.at < 0 || event.at < this.last) { this.unclocked = Math.min(Number.MAX_SAFE_INTEGER, this.unclocked + 1); return; }
    this.last = event.at;
    const key = `${event.stage}:${event.outcome}`;
    let store = this.stores.get(key);
    if (!store) { store = new WindowStore(); this.stores.set(key, store); }
    const ms = event.ms !== null && Number.isFinite(event.ms) && event.ms >= 0 ? event.ms : null;
    store.add({ at: event.at, ms });
  };
  private outcome(outcome: ResultOutcome, at: number): void { this.outcomes.add({ at, outcome }); }
  private expire(now: number): void {
    for (const [id, job] of this.jobs) if (now - job.accepted > PIPELINE_WINDOW_MS) {
      this.jobs.delete(id); this.outcome('expired', now);
    }
  }
  accept(identity: InferenceMeasurement): void {
    if (identity.generation !== this.generation) return;
    const now = this.time(); if (now === null) { this.unclocked = Math.min(Number.MAX_SAFE_INTEGER, this.unclocked + 1); return; }
    this.expire(now);
    if (this.jobs.size >= PIPELINE_JOB_CAP) {
      this.jobs.delete(this.jobs.keys().next().value!); this.outcome('capacity', now);
    }
    this.jobs.set(identity.jobId, { identity, accepted: now });
  }
  discard(id: number, outcome: Exclude<ResultOutcome, 'first-draw' | 'unbound'>): void {
    const now = this.time();
    if (now !== null) this.expire(now);
    if (!this.jobs.delete(id)) return;
    if (now !== null) this.outcome(outcome, now);
    else this.unclocked = Math.min(Number.MAX_SAFE_INTEGER, this.unclocked + 1);
  }
  discardAll(outcome: 'cancelled' | 'hidden' | 'draw-failed'): void {
    for (const id of this.jobs.keys()) this.discard(id, outcome);
  }
  firstDraw(identity: InferenceMeasurement): void {
    if (identity.generation !== this.generation) return;
    const now = this.time();
    if (now === null) {
      const job = this.jobs.get(identity.jobId);
      if (job?.identity === identity) { this.jobs.delete(identity.jobId); this.unclocked = Math.min(Number.MAX_SAFE_INTEGER, this.unclocked + 1); }
      return;
    }
    this.expire(now);
    const job = this.jobs.get(identity.jobId);
    // Reference identity prevents an unrelated/replacement job from matching.
    if (!job || job.identity !== identity) return;
    this.jobs.delete(identity.jobId);
    this.outcome('first-draw', now);
    if (!identity.immutable || identity.captureStart === null || identity.captureEnd === null || identity.captureEnd > now || identity.captureStart < 0 || identity.captureEnd < identity.captureStart) {
      this.outcome('unbound', now); return;
    }
    this.record({ stage: 'latency', outcome: 'success', at: now, ms: now - identity.captureStart });
  }
  snapshot(): PipelineSnapshot | null {
    const now = this.time(); if (now === null) return null;
    this.expire(now);
    const stages = {} as PipelineSnapshot['stages'];
    for (const stage of PIPELINE_STAGES) {
      stages[stage] = {} as Record<AttemptOutcome, StageStatistics>;
      for (const outcome of OUTCOMES) {
        const window = this.stores.get(`${stage}:${outcome}`)?.window(now);
        const events = window?.events ?? [];
        const valid = events.flatMap(e => e.ms === null ? [] : [e.ms]);
        const values = valid.slice(-PIPELINE_QUANTILE_CAP).sort((a, b) => a - b);
        const n = values.length;
        const total = valid.reduce((sum, ms) => sum + ms, 0);
        stages[stage][outcome] = {
          count: events.length, measured: valid.length, missing: events.length - valid.length,
          totalMs: valid.length && Number.isFinite(total) ? total : null, mean: valid.length && Number.isFinite(total) ? total / valid.length : null,
          median: n ? values[Math.floor((n - 1) / 2)] / 2 + values[Math.floor(n / 2)] / 2 : null,
          p95: n ? values[Math.ceil(n * 0.95) - 1] : null,
          max: valid.length ? Math.max(...valid) : null,
          quantileCount: n, truncated: window?.truncated ?? false,
        };
      }
    }
    const results: PipelineSnapshot['results'] = {};
    const resultWindow = this.outcomes.window(now);
    for (const e of resultWindow.events) results[e.outcome] = (results[e.outcome] ?? 0) + 1;
    const captures = OUTCOMES.map(outcome => stages.capture[outcome]);
    // CaptureInput-only completed-interval occupancy / fixed 5000 ms window.
    // Comparison outside captureInput is shown separately, never silently included.
    const duty = this.unclocked > 0 || captures.some(s => s.truncated || s.missing || (s.measured > 0 && s.totalMs === null)) || !captures.some(s => s.measured)
      ? null : captures.reduce((sum, s) => sum + (s.totalMs ?? 0), 0) / PIPELINE_WINDOW_MS * 100;
    const captureDutyPercent = duty !== null && Number.isFinite(duty) ? duty : null;
    return { stages, results, unclocked: this.unclocked, resultsTruncated: resultWindow.truncated, pendingResults: this.jobs.size, captureDutyPercent };
  }
}
// Guard statistics consumers as well as observers. Used outside all product catches.
export function telemetry<T>(action: () => T): T | undefined { try { return action(); } catch { return; } }
