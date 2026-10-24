import { expect, it } from 'vitest';
import { PerformanceMetrics } from '../../../../src/services/performanceMetrics';
// Independent array oracle: deliberately no production constants or helpers.
function oracle(events, now) {
  const retained = events.filter(t => now - t <= 5000).slice(-300);
  if (retained.length < 2 || now - retained.at(-1) >= 2000) return null;
  const elapsedSeconds = (retained.at(-1) - retained[0]) / 1000;
  return elapsedSeconds < 0.25 ? null : (retained.length - 1) / elapsedSeconds;
}
it('independent irregular-event arithmetic, retention boundaries, idle aging and resume', () => {
  let now = 0; const m = new PerformanceMetrics(() => now); const events = [];
  for (const time of [10, 400, 1700, 2800, 5000, 5010, 5011, 6300, 12000, 12500]) {
    now = time; m.input(); m.inferenceCompleted(now - 7, now); m.drawCompleted(); events.push(now);
    const result = m.snapshot();
    const expected = oracle(events, now);
    if (expected === null) {
      expect(result.inferenceFps).toBeNull(); expect(result.drawFps).toBeNull();
    } else {
      expect(result.inferenceFps).toBeCloseTo(expected, 12);
      expect(result.drawFps).toBeCloseTo(expected, 12);
    }
    expect(result.processingMs).toBeCloseTo(7);
  }
  now = 14499; expect(m.snapshot().inferenceFps).toBe(oracle(events, now));
  now = 14500; expect(m.snapshot().inferenceFps).toBeNull();
  now = 90000; expect(m.snapshot().inferenceFps).toBeNull();
  m.input(); m.inferenceCompleted(now - 7, now); expect(m.snapshot().inferenceFps).toBeNull();
  now += 1500; m.input(); m.inferenceCompleted(now - 7, now);
  expect(m.snapshot().inferenceFps).toBeCloseTo(1 / 1.5);
});
it('separate count/span oracle over >300 events, with exact 5-second boundary', () => {
  let now = 0; const m = new PerformanceMetrics(() => now), events = [];
  for (let i = 0; i < 1000; i++) { now += 17; m.drawCompleted(); events.push(now); }
  expect(m.snapshot().drawFps).toBeCloseTo(1000 / 17);
  expect(m.snapshot().drawFps).toBe(oracle(events, now));
  now = 22000; expect(m.snapshot().drawFps).toBeNull();
  expect(m.draws).toEqual([17000]); // Inclusive boundary; cannot supply a fabricated rate.
  now++; m.snapshot(); expect(m.draws).toEqual([]);
});
it('full duration mean, empty successful events, invalid/refused input, clocks and zero versus unknown', () => {
  let now = 100; const m = new PerformanceMetrics(() => now);
  m.inferenceCompleted(70, 100);
  now = 600; m.inferenceCompleted(590, 600);
  expect(m.snapshot().processingMs).toBe(20); expect(m.snapshot().inferenceFps).toBe(2);
  for (const pair of [[NaN, 3], [1, Infinity], [-1, 3], [4, 3], [601, 700]]) m.inferenceCompleted(...pair);
  expect(m.snapshot().processingMs).toBe(20);
  now = 2600; expect(m.snapshot().processingMs).toBeNull(); expect(m.snapshot().inferenceFps).toBeNull();
  now = 10; expect(m.snapshot().inferenceFps).toBeNull();
  m.inferenceCompleted(10, 10); now = 510; m.inferenceCompleted(510, 510);
  expect(m.snapshot().inferenceFps).toBe(2); expect(m.snapshot().processingMs).toBeNull();
  now = NaN; expect(m.snapshot().inferenceFps).toBeNull();
});
