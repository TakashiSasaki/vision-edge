import { expect, it } from 'vitest';
import { MAX_METRIC_SAMPLES, METRIC_WINDOW_MS, PerformanceMetrics } from './performanceMetrics';

it('measures successful event spans and complete processing duration with exact clock values', () => {
  let now = 0; const metrics = new PerformanceMetrics(() => now);
  expect(metrics.snapshot()).toEqual({ inferenceFps: null, drawFps: null, processingMs: null, stalled: false });
  now = 10; metrics.input(); metrics.inferenceCompleted(2, 10);
  expect(metrics.snapshot().inferenceFps).toBeNull();
  now = 510; metrics.input(); metrics.inferenceCompleted(498, 510); metrics.drawCompleted();
  now = 1010; metrics.input(); metrics.drawCompleted();
  expect(metrics.snapshot()).toEqual({ inferenceFps: 2, drawFps: 2, processingMs: 10, stalled: false });
  now = 3010;
  expect(metrics.snapshot()).toEqual({ inferenceFps: null, drawFps: null, processingMs: null, stalled: true });
  now += METRIC_WINDOW_MS;
  expect(metrics.snapshot().inferenceFps).toBeNull();
});
it('rejects invalid measurements and duplicate timestamps; resets after clock reversal', () => {
  let now = 1000; const m = new PerformanceMetrics(() => now);
  for (const [start, end] of [[NaN, 1], [-1, 1], [2, 2], [4, 3], [0, Infinity], [0, 1001]]) m.inferenceCompleted(start, end);
  expect(m.snapshot().processingMs).toBeNull();
  m.inferenceCompleted(990, 1000); m.inferenceCompleted(990, 1000); m.drawCompleted(); m.drawCompleted();
  expect(m.snapshot().inferenceFps).toBeNull();
  now = NaN; expect(m.snapshot().drawFps).toBeNull();
  now = -1; expect(m.snapshot().processingMs).toBeNull();
  now = 10; expect(m.snapshot().processingMs).toBeNull();
});
it('handles low-frequency input, long background pauses and bounded retention without clamping', () => {
  let now = 0; const m = new PerformanceMetrics(() => now);
  for (let i = 0; i < MAX_METRIC_SAMPLES * 10; i++) { now += 1; m.drawCompleted(); }
  expect(m.snapshot().drawFps).toBe(1000);
  expect((m as unknown as { draws: number[] }).draws).toHaveLength(MAX_METRIC_SAMPLES);
  now += 60000; expect(m.snapshot().drawFps).toBeNull();
  m.input(); m.inferenceCompleted(now - 1, now);
  now += 1500; m.input(); m.inferenceCompleted(now - 1, now);
  expect(m.snapshot().inferenceFps).toBeCloseTo(2 / 3);
  m.reset(); expect(m.snapshot().processingMs).toBeNull();
});
it('reports stalled input even if no first frame ever arrives; zero-duration work has no duration estimate', () => {
  let now = 0; const m = new PerformanceMetrics(() => now);
  now = 2500; expect(m.snapshot().stalled).toBe(true);
  m.inferenceCompleted(now, now); now += 500; m.inferenceCompleted(now, now);
  expect(m.snapshot().inferenceFps).toBe(2); expect(m.snapshot().processingMs).toBeNull();
});
