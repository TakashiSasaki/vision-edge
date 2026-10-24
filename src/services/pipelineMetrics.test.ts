import { expect, it } from 'vitest';
import { PipelineMetrics, PipelineProbe, PIPELINE_EVENT_CAP, PIPELINE_JOB_CAP, telemetry, type InferenceMeasurement } from './pipelineMetrics';
let generation = Symbol('test');
const metricsWithGeneration = (clock: () => number) => { const metrics = new PipelineMetrics(clock); generation = metrics.generation; return metrics; };
const identity = (jobId: number, captureStart = 0, immutable = true): InferenceMeasurement => ({ generation, jobId, captureId: jobId + 10, captureStart, captureEnd: captureStart, immutable });
it('keeps actual zero, even medians, nearest-rank p95, same-timestamp events, and inclusive boundaries', () => {
  let now = 0; const metrics = metricsWithGeneration(() => now);
  for (const ms of [0, 10, 20, 30]) metrics.record({ at: 0, ms, stage: 'capture', outcome: 'success' });
  expect(metrics.snapshot()!.stages.capture.success).toMatchObject({ count: 4, measured: 4, mean: 15, median: 15, p95: 30, max: 30, totalMs: 60 });
  now = 5000; expect(metrics.snapshot()!.stages.capture.success.count).toBe(4);
  now = 5000.001; expect(metrics.snapshot()!.stages.capture.success).toMatchObject({ count: 0, mean: null, median: null, totalMs: null });
});
it('separates full retained event totals from latest 300 quantiles and exposes overflow coverage', () => {
  let now = 0; const metrics = metricsWithGeneration(() => now);
  for (let i = 0; i < 1000; i++) metrics.record({ at: 0, ms: i, stage: 'comparison', outcome: 'success' });
  expect(metrics.snapshot()!.stages.comparison.success).toMatchObject({ count: 1000, measured: 1000, totalMs: 499500, mean: 499.5, median: 849.5, p95: 984, max: 999, quantileCount: 300, truncated: false });
  for (let i = 0; i < 4000; i++) metrics.record({ at: 0, ms: 1, stage: 'comparison', outcome: 'success' });
  expect(metrics.snapshot()!.stages.comparison.success).toMatchObject({ count: PIPELINE_EVENT_CAP, totalMs: PIPELINE_EVENT_CAP, truncated: true });
  now = 5000; expect(metrics.snapshot()!.stages.comparison.success.truncated).toBe(true);
  now++; expect(metrics.snapshot()!.stages.comparison.success.truncated).toBe(false);
});
it.each([NaN, Infinity, -1, null])('represents missing/invalid duration %s without zero substitution', ms => {
  const metrics = metricsWithGeneration(() => 0);
  metrics.record({ at: 0, ms, stage: 'sdk', outcome: 'success' });
  expect(metrics.snapshot()!.stages.sdk.success).toMatchObject({ count: 1, measured: 0, missing: 1, mean: null, p95: null, totalMs: null });
  expect(metrics.snapshot()!.stages.postprocess.success.count).toBe(0);
});
it.each([NaN, Infinity, -1])('quarantines broken clock %s and recovers with missing interval, not invented zero', invalid => {
  let now = 10; const metrics = metricsWithGeneration(() => now); const probe = new PipelineProbe(metrics);
  const start = probe.start(); now = invalid; probe.end('capture', start);
  expect(metrics.snapshot()).toBeNull();
  now = 11; probe.end('capture', start);
  expect(metrics.snapshot()!.stages.capture.success).toMatchObject({ measured: 0, missing: 1 });
  expect(metrics.snapshot()!.unclocked).toBe(1);
});
it('rejects backwards clocks until catch-up and invalidates crossing spans; equal clocks are measured zero', () => {
  let now = 10; const metrics = metricsWithGeneration(() => now); const probe = new PipelineProbe(metrics);
  const start = probe.start(); metrics.snapshot(); now = 9; probe.end('sdk', start); expect(metrics.snapshot()).toBeNull();
  now = 10; probe.end('sdk', start);
  const next = probe.start(); probe.end('sdk', next);
  expect(metrics.snapshot()!.stages.sdk.success).toMatchObject({ measured: 1, missing: 1, mean: 0 });
});
it('isolates throwing clocks, observers and statistics consumers', () => {
  const probe = new PipelineProbe({ now: () => { throw Error('clock'); }, record: () => { throw Error('observer'); } });
  expect(() => probe.end('sdk', probe.start())).not.toThrow();
  expect(() => new PipelineProbe({ now: () => 1, record: () => { throw Error('observer'); } }).end('engine', undefined)).not.toThrow();
  expect(telemetry(() => { throw Error('statistics'); })).toBeUndefined();
});
it('separates success/failed/cancelled/stale durations, capture-only duty, and resets generations', () => {
  const metrics = metricsWithGeneration(() => 100);
  for (const outcome of ['success', 'failed', 'cancelled', 'stale'] as const) metrics.record({ at: 100, ms: 10, stage: 'capture', outcome });
  metrics.record({ at: 100, ms: 200, stage: 'comparison', outcome: 'success' });
  const snapshot = metrics.snapshot()!;
  for (const outcome of ['success', 'failed', 'cancelled', 'stale'] as const) expect(snapshot.stages.capture[outcome]).toMatchObject({ count: 1, totalMs: 10 });
  expect(snapshot.captureDutyPercent).toBe(0.8);
  metrics.reset(); expect(metrics.snapshot()!.stages.capture.success.count).toBe(0);
});
it('withholds capture duty on incomplete or missing coverage', () => {
  const metrics = metricsWithGeneration(() => 0);
  expect(metrics.snapshot()!.captureDutyPercent).toBeNull();
  for (let i = 0; i <= PIPELINE_EVENT_CAP; i++) metrics.record({ at: 0, ms: 1, stage: 'capture', outcome: 'success' });
  expect(metrics.snapshot()!.captureDutyPercent).toBeNull();
  metrics.reset(); metrics.record({ at: 0, ms: null, stage: 'capture', outcome: 'success' });
  expect(metrics.snapshot()!.captureDutyPercent).toBeNull();
});
it('binds one first draw to the exact accepted identity; empty results need no special exception', () => {
  let now = 20; const metrics = metricsWithGeneration(() => now); const job = identity(1, 10);
  metrics.accept(job); metrics.firstDraw({ ...job }); // Different identity cannot claim it.
  expect(metrics.snapshot()!.pendingResults).toBe(1);
  now = 30; metrics.firstDraw(job); now = 50; metrics.firstDraw(job);
  expect(metrics.snapshot()!.stages.latency.success).toMatchObject({ count: 1, mean: 20 });
  expect(metrics.snapshot()!.results).toEqual({ 'first-draw': 1 });
  const live = identity(2, 50, false); metrics.accept(live); metrics.firstDraw(live);
  expect(metrics.snapshot()!.results.unbound).toBe(1);
  expect(metrics.snapshot()!.stages.latency.success.count).toBe(1);
});
it.each(['overwritten', 'cancelled', 'hidden', 'draw-failed'] as const)('records %s as never drawn, never zero latency or a later recovery sample', outcome => {
  const metrics = metricsWithGeneration(() => 10); const job = identity(1);
  metrics.accept(job); metrics.discard(job.jobId, outcome); metrics.firstDraw(job);
  expect(metrics.snapshot()!.results).toEqual({ [outcome]: 1 });
  expect(metrics.snapshot()!.stages.latency.success.mean).toBeNull();
});
it('bounds result retention by count and age, and rejects old identities after generation reset', () => {
  let now = 0; const metrics = metricsWithGeneration(() => now);
  for (let i = 0; i < 1000; i++) metrics.accept(identity(i));
  expect(metrics.snapshot()!.pendingResults).toBe(PIPELINE_JOB_CAP);
  expect(metrics.snapshot()!.results.capacity).toBe(1000 - PIPELINE_JOB_CAP);
  now = 5000; expect(metrics.snapshot()!.pendingResults).toBe(PIPELINE_JOB_CAP);
  now++; expect(metrics.snapshot()!.pendingResults).toBe(0);
  expect(metrics.snapshot()!.results.expired).toBe(PIPELINE_JOB_CAP);
  const old = identity(1, now); metrics.accept(old); metrics.reset();
  generation = metrics.generation; metrics.accept(identity(1, now)); metrics.firstDraw(old);
  expect(metrics.snapshot()!.stages.latency.success.count).toBe(0);
});
it('bounds outcome retention and labels discarded coverage', () => {
  const metrics = metricsWithGeneration(() => 0);
  for (let i = 0; i < 3000; i++) { metrics.accept(identity(i)); metrics.discard(i, 'hidden'); }
  expect(metrics.snapshot()!.results).toEqual({ hidden: PIPELINE_EVENT_CAP });
  expect(metrics.snapshot()!.resultsTruncated).toBe(true);
});

it('rejects old probe emissions after reset and does not put them in the replacement generation', () => {
  const metrics = metricsWithGeneration(() => 0); const oldProbe = new PipelineProbe(metrics); const start = oldProbe.start();
  metrics.reset(); oldProbe.end('capture', start);
  expect(metrics.snapshot()!.stages.capture.success.count).toBe(0);
});
it('a first draw with an invalid clock becomes a missing measurement and is not later expired or drawn twice', () => {
  let now = 0; const metrics = metricsWithGeneration(() => now); const job = identity(1);
  metrics.accept(job); now = NaN; metrics.firstDraw(job); now = 10;
  metrics.firstDraw(job); expect(metrics.snapshot()!.unclocked).toBe(1);
  expect(metrics.snapshot()!.pendingResults).toBe(0); expect(metrics.snapshot()!.stages.latency.success.mean).toBeNull();
});
it('does not publish overflowing totals or medians as finite observations', () => {
  const metrics = metricsWithGeneration(() => 0);
  for (let i = 0; i < 2; i++) metrics.record({ stage: 'capture', outcome: 'success', at: 0, ms: Number.MAX_VALUE });
  const snapshot = metrics.snapshot()!;
  expect(snapshot.stages.capture.success.totalMs).toBeNull();
  expect(snapshot.stages.capture.success.median).toBe(Number.MAX_VALUE);
  expect(snapshot.captureDutyPercent).toBeNull();
});
