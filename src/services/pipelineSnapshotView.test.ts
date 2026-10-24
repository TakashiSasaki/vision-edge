import { expect, it } from 'vitest';
import { PipelineMetrics, PIPELINE_EVENT_CAP, PIPELINE_QUANTILE_CAP, type PipelineSnapshot } from './pipelineMetrics';
import { readPipelineSnapshot } from './pipelineSnapshotView';
function snapshot(): PipelineSnapshot {
  const metrics = new PipelineMetrics(() => 0);
  metrics.record({ stage: 'capture', outcome: 'success', at: 0, ms: 0 });
  metrics.record({ stage: 'sdk', outcome: 'success', at: 0, ms: 7 });
  return metrics.snapshot()!;
}
it.each([null, undefined, false, 1, 'snapshot', [], Symbol('snapshot'), 1n, () => 0])('makes non-record snapshot %s unavailable', value => {
  expect(readPipelineSnapshot(value)).toBeNull();
});
it.each([null, undefined, [], 7, 'stages'])('contains unavailable stages %s without discarding other valid sections', stages => {
  const source = snapshot(); const view = readPipelineSnapshot({ ...source, stages })!;
  expect(view.stages.capture.success).toBeNull(); expect(view.results).toEqual({});
  expect(view.pendingResults).toBe(0); expect(view.captureDutyPercent).toBeNull();
});
it('preserves actual zero, normal null/no-execution data and missing-duration partial data', () => {
  const source = snapshot(); const view = readPipelineSnapshot(source)!;
  expect(view.stages).toEqual(source.stages); expect(view.captureDutyPercent).toBe(0);
  const metrics = new PipelineMetrics(() => 0);
  metrics.record({ stage: 'sdk', outcome: 'success', at: 0, ms: null });
  metrics.record({ stage: 'sdk', outcome: 'success', at: 0, ms: 0 });
  expect(readPipelineSnapshot(metrics.snapshot())!.stages.sdk.success).toMatchObject({ count: 2, measured: 1, missing: 1, mean: 0 });
});
it('retains >300 quantile coverage, truncated windows and valid overflowed sum/mean', () => {
  const metrics = new PipelineMetrics(() => 0);
  for (let i = 0; i <= PIPELINE_EVENT_CAP; i++) metrics.record({ stage: 'comparison', outcome: 'success', at: 0, ms: i });
  const source = metrics.snapshot()!; const view = readPipelineSnapshot(source)!;
  expect(view.stages.comparison.success).toEqual(source.stages.comparison.success);
  expect(view.stages.comparison.success).toMatchObject({ count: PIPELINE_EVENT_CAP, quantileCount: PIPELINE_QUANTILE_CAP, truncated: true });
  for (let i = 0; i < 2; i++) metrics.record({ stage: 'capture', outcome: 'success', at: 0, ms: Number.MAX_VALUE });
  const overflow = metrics.snapshot()!;
  expect(readPipelineSnapshot(overflow)!.stages.capture.success).toEqual(overflow.stages.capture.success);
  expect(readPipelineSnapshot(overflow)!.captureDutyPercent).toBeNull();
});
it.each([null, undefined, [], 'outcomes'])('isolates invalid nested stage %s to its own cells', capture => {
  const source = snapshot(); const view = readPipelineSnapshot({ ...source, stages: { ...source.stages, capture } })!;
  expect(view.stages.capture.success).toBeNull(); expect(view.stages.sdk.success!.mean).toBe(7);
});
it.each([
  { count: -1 }, { count: NaN }, { count: Infinity }, { count: 1.5 }, { count: PIPELINE_EVENT_CAP + 1 },
  { measured: 2 }, { missing: 1 }, { quantileCount: 2 }, { quantileCount: 301 }, { truncated: 'false' },
])('makes inconsistent count/coverage row %j unavailable', change => {
  const source = snapshot(); Object.assign(source.stages.capture.success, change);
  const view = readPipelineSnapshot(source)!;
  expect(view.stages.capture.success).toBeNull(); expect(view.stages.sdk.success!.mean).toBe(7);
  expect(view.captureDutyPercent).toBeNull();
});
it.each([NaN, Infinity, -1, '0', {}, 0n, Symbol('duration'), undefined])('makes malformed duration %s missing while preserving valid columns', value => {
  const source = snapshot(); const row = source.stages.sdk.success;
  Object.assign(row, { mean: value }); const view = readPipelineSnapshot(source)!;
  expect(view.stages.sdk.success).toMatchObject({ count: 1, mean: null, median: 7, p95: 7, max: 7, totalMs: 7 });
});
it('does not turn malformed times on an unexecuted stage into measured zero', () => {
  const source = snapshot(); Object.assign(source.stages.draw.success, { mean: 0, median: 0, p95: 0, max: 0, totalMs: 0 });
  expect(readPipelineSnapshot(source)!.stages.draw.success).toMatchObject({ count: 0, measured: 0, mean: null, median: null, p95: null, max: null, totalMs: null });
});
it.each([null, undefined, [], { hidden: NaN }, { hidden: -1 }, { hidden: 0.5 }, { hidden: '1' }, { hidden: {} }, { hidden: 2048, expired: 1 }])('makes invalid result counters %j unavailable, with no invented no-outcome state', results => {
  const source = snapshot(); const view = readPipelineSnapshot({ ...source, results })!;
  expect(view.results).toBeNull(); expect(view.stages.sdk.success!.mean).toBe(7);
});
it('copies only the bounded known result fields and preserves partial outcome dictionaries', () => {
  const source = snapshot(); const results = { hidden: 2, unbound: 1, unknown: { toString: () => { throw Error('coercion'); } } };
  expect(readPipelineSnapshot({ ...source, results })!.results).toEqual({ hidden: 2, unbound: 1 });
});
it('validates counters/flags and withholds duty on missing capture coverage without clamping valid >100 percent', () => {
  const source = snapshot();
  const view = readPipelineSnapshot({ ...source, pendingResults: 65, unclocked: {}, resultsTruncated: 'false', captureDutyPercent: 9 })!;
  expect(view).toMatchObject({ pendingResults: null, unclocked: null, resultsTruncated: null, captureDutyPercent: null });
  expect(readPipelineSnapshot({ ...source, captureDutyPercent: Infinity })!.captureDutyPercent).toBeNull();
  expect(readPipelineSnapshot({ ...source, captureDutyPercent: 120 })!.captureDutyPercent).toBe(120);
});
it('contains throwing getters/proxies and never rereads validated counters or exposes mutable raw objects', () => {
  const throwing = new Proxy({}, { get() { throw Error('statistics access'); } });
  expect(readPipelineSnapshot(throwing)).toBeNull();
  const source = snapshot(); let reads = 0;
  Object.defineProperty(source, 'pendingResults', { get: () => ++reads === 1 ? 2 : { invalid: true } });
  const view = readPipelineSnapshot(source)!;
  expect(view.pendingResults).toBe(2); expect(reads).toBe(1);
  source.stages.sdk.success.mean = NaN;
  expect(view.stages.sdk.success!.mean).toBe(7);
  Object.defineProperty(source.stages.capture.success, 'mean', { get() { throw Error('row'); } });
  const partial = readPipelineSnapshot(source)!;
  expect(partial.stages.capture.success).toBeNull(); expect(partial.stages.sdk.success!.median).toBe(7);
});
