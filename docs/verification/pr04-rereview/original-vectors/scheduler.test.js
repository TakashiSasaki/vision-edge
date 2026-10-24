import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { JobLock, startFrameScheduler } from '../../../../src/services/frameScheduler';
let callbacks, serial;
beforeEach(() => {
  callbacks = new Map(); serial = 0; vi.useFakeTimers();
  vi.stubGlobal('requestAnimationFrame', fn => { callbacks.set(++serial, fn); return serial; });
  vi.stubGlobal('cancelAnimationFrame', id => callbacks.delete(id));
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });
function harness(native, lock = new JobLock()) {
  const video = { readyState: 2, videoWidth: 1, videoHeight: 1, currentTime: 0 };
  if (native) {
    video.requestVideoFrameCallback = fn => { callbacks.set(++serial, fn); return serial; };
    video.cancelVideoFrameCallback = id => callbacks.delete(id);
  }
  const work = vi.fn(() => video.currentTime);
  const adopt = vi.fn(), error = vi.fn(), input = vi.fn();
  let alive = true;
  const stop = startFrameScheduler({ video, current: () => alive, work, adopt, error, input, lock });
  function deliver(time, count = time) {
    video.currentTime = time;
    const [id, callback] = callbacks.entries().next().value;
    callbacks.delete(id);
    callback(0, { mediaTime: time, presentedFrames: count });
  }
  return { video, work, adopt, error, input, stop, deliver, invalidate: () => { alive = false; } };
}
it.each([false, true])('F01: deferral must not infer the same live frame twice (%s)', native => {
  const s = harness(native);
  s.deliver(1);
  // Decoder advances before queued execution; no JS arrival callback yet.
  s.video.currentTime = 2;
  vi.runOnlyPendingTimers(); // SDK boundary consumes live frame 2.
  s.deliver(2);
  vi.runOnlyPendingTimers(); // Same live frame 2 is consumed again.
  console.log('F01', { native, consumedFrames: s.work.mock.results.map(r => r.value) });
  s.stop();
  expect(s.work.mock.results.map(r => r.value)).toEqual([2]);
});
it.each([false, true])('first zero, equal new-session time, coarse timestamps and cancellation (%s)', native => {
  const a = harness(native); a.deliver(0, 0); vi.runOnlyPendingTimers();
  a.deliver(0, 0); vi.runOnlyPendingTimers(); expect(a.work).toHaveBeenCalledTimes(1);
  a.stop(); expect(callbacks.size).toBe(0);
  const b = harness(native); b.deliver(0, 0); vi.runOnlyPendingTimers();
  expect(b.work).toHaveBeenCalledTimes(1);
  if (native) { b.deliver(0, 1); vi.runOnlyPendingTimers(); expect(b.work).toHaveBeenCalledTimes(2); }
  b.stop();
});
it('F02: consumes an already ready first frame without waiting for a second native arrival', () => {
  const s = harness(true);
  vi.advanceTimersByTime(400);
  s.stop();
  expect(s.work).toHaveBeenCalledTimes(1);
});
it.each([false, true])('gate immediately before work, then accept same input after readiness (%s)', native => {
  const s = harness(native); s.deliver(0); s.video.readyState = 1; vi.runOnlyPendingTimers();
  expect(s.work).not.toHaveBeenCalled();
  s.video.readyState = 2; s.deliver(0); vi.runOnlyPendingTimers();
  expect(s.work).toHaveBeenCalledTimes(1); s.stop();
});
it.each([false, true])('throwing work releases ownership and never retries unchanged input (%s)', native => {
  const s = harness(native); s.work.mockImplementationOnce(() => { throw Error('SDK'); });
  s.deliver(1); vi.runOnlyPendingTimers(); s.deliver(1); vi.runOnlyPendingTimers();
  expect(s.error).toHaveBeenCalledTimes(1); expect(s.work).toHaveBeenCalledTimes(1);
  s.deliver(2); vi.runOnlyPendingTimers(); expect(s.adopt).toHaveBeenCalledWith(2); s.stop();
});
it('async stale completion cannot adopt; foreign releases cannot unlock; latest replaces 100 arrivals', async () => {
  const lock = new JobLock(); const s = harness(true, lock);
  let resolve; s.work.mockImplementationOnce(() => new Promise(yes => { resolve = yes; }));
  s.deliver(0); vi.runOnlyPendingTimers();
  for (let i = 1; i <= 100; i++) s.deliver(i);
  lock.release(Symbol('foreign')); expect(lock.acquire()).toBeNull();
  resolve(0); await vi.runAllTimersAsync();
  expect(s.work).toHaveBeenCalledTimes(2); expect(s.adopt).toHaveBeenLastCalledWith(100);
  s.stop();
  const old = harness(true, lock); old.work.mockImplementationOnce(() => new Promise(yes => { resolve = yes; }));
  old.deliver(0); vi.runOnlyPendingTimers(); old.stop();
  const replacement = harness(true, lock); replacement.deliver(0); vi.runOnlyPendingTimers();
  expect(replacement.work).not.toHaveBeenCalled();
  resolve(9); await vi.runAllTimersAsync();
  expect(old.adopt).not.toHaveBeenCalled();
  // Documented asynchronous gap: without another rVFC arrival, pending stays asleep.
  console.log('R01 replacement pending after old lock release:', replacement.work.mock.calls.length);
  replacement.deliver(1); vi.runOnlyPendingTimers(); expect(replacement.work).toHaveBeenCalledTimes(1);
  replacement.stop();
});
