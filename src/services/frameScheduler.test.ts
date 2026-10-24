import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { JobLock, startFrameScheduler } from './frameScheduler';
let raf: Map<number, FrameRequestCallback>;
let rvfc: Map<number, VideoFrameRequestCallback>;
let id: number;
beforeEach(() => {
  vi.useFakeTimers(); raf = new Map(); rvfc = new Map(); id = 0;
  vi.stubGlobal('requestAnimationFrame', vi.fn((fn: FrameRequestCallback) => { raf.set(++id, fn); return id; }));
  vi.stubGlobal('cancelAnimationFrame', vi.fn((key: number) => raf.delete(key)));
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });
function setup(native = false, lock = new JobLock()) {
  let current = true;
  const video = Object.defineProperties(document.createElement('video'), {
    readyState: { configurable: true, value: 2 }, videoWidth: { value: 640 }, videoHeight: { value: 480 },
  });
  if (native) {
    video.requestVideoFrameCallback = vi.fn(fn => { rvfc.set(++id, fn); return id; });
    video.cancelVideoFrameCallback = vi.fn(key => { rvfc.delete(key); });
  }
  const work = vi.fn<() => number | Promise<number>>(() => 1);
  const adopt = vi.fn(); const error = vi.fn(); const input = vi.fn();
  const stop = startFrameScheduler({ video, current: () => current, work, adopt, error, input, lock });
  const frame = (time: number, count = time) => {
    if (!native) video.currentTime = time;
    const callbacks = native ? rvfc : raf;
    const entry = callbacks.entries().next().value!; callbacks.delete(entry[0]);
    if (native) (entry[1] as VideoFrameRequestCallback)(0, { mediaTime: time, presentedFrames: count } as VideoFrameCallbackMetadata);
    else (entry[1] as FrameRequestCallback)(0);
  };
  return { video, frame, work, adopt, error, input, stop, invalidate: () => { current = false; } };
}
it.each([false, true])('deduplicates %s frames, processes first zero timestamp and cancels paired callbacks', native => {
  const f = setup(native); f.frame(0); vi.runOnlyPendingTimers();
  expect(f.work).toHaveBeenCalledTimes(1);
  for (let i = 0; i < 10; i++) f.frame(0);
  vi.runOnlyPendingTimers(); expect(f.work).toHaveBeenCalledTimes(1);
  f.frame(1); vi.runOnlyPendingTimers(); expect(f.work).toHaveBeenCalledTimes(2);
  f.stop(); expect(raf.size + rvfc.size).toBe(0);
  const next = setup(native); next.frame(0); vi.runOnlyPendingTimers(); expect(next.work).toHaveBeenCalledTimes(1); next.stop();
});
it('checks readiness at reservation and immediately before work, and current generations at adoption', async () => {
  const f = setup(); Object.defineProperty(f.video, 'readyState', { value: 1, configurable: true });
  f.frame(0); vi.runOnlyPendingTimers(); expect(f.work).not.toHaveBeenCalled();
  Object.defineProperty(f.video, 'readyState', { value: 2 });
  f.frame(0); f.invalidate(); vi.runOnlyPendingTimers(); expect(f.work).not.toHaveBeenCalled(); f.stop();
  const g = setup(); let resolve!: (value: number) => void;
  g.work.mockImplementation(() => new Promise<number>(yes => { resolve = yes; }));
  g.frame(0); vi.runOnlyPendingTimers(); g.invalidate(); resolve(1); await Promise.resolve();
  expect(g.adopt).not.toHaveBeenCalled(); g.stop();
});
it('keeps one latest pending candidate during slow work, releasing only the owning job', async () => {
  const lock = new JobLock(); const f = setup(false, lock);
  let resolve!: (value: number) => void;
  f.work.mockImplementationOnce(() => new Promise<number>(yes => { resolve = yes; }));
  f.frame(0); vi.runOnlyPendingTimers();
  for (let i = 1; i < 100; i++) f.frame(i);
  expect(f.work).toHaveBeenCalledTimes(1);
  resolve(1); await vi.runAllTimersAsync(); expect(f.work).toHaveBeenCalledTimes(2); f.stop();
  const token = lock.acquire()!; lock.release(Symbol('old')); expect(lock.acquire()).toBeNull(); lock.release(token);
  expect(lock.acquire()).not.toBeNull();
});
it('old cleanup and old completion cannot unlock a running job or adopt into replacement', async () => {
  const lock = new JobLock(); const f = setup(false, lock);
  let resolve!: (value: number) => void;
  f.work.mockImplementationOnce(() => new Promise<number>(yes => { resolve = yes; }));
  f.frame(0); vi.runOnlyPendingTimers(); f.stop();
  const g = setup(false, lock); g.frame(0); vi.runOnlyPendingTimers(); expect(g.work).not.toHaveBeenCalled();
  resolve(1); await vi.runAllTimersAsync(); expect(f.adopt).not.toHaveBeenCalled();
  g.frame(1); vi.runOnlyPendingTimers(); expect(g.work).toHaveBeenCalledTimes(1); g.stop();
});
it('does not retry failed unchanged frames and recovers on subsequent input', () => {
  const f = setup(); f.work.mockImplementationOnce(() => { throw new Error('SDK'); });
  f.frame(0); vi.runOnlyPendingTimers(); expect(f.error).toHaveBeenCalledTimes(1);
  f.frame(0); vi.runOnlyPendingTimers(); expect(f.work).toHaveBeenCalledTimes(1);
  f.frame(1); vi.runOnlyPendingTimers(); expect(f.adopt).toHaveBeenCalledWith(1); f.stop();
});
it('prefers decoded frame counts over advancing fallback playback time', () => {
  const f = setup(); let frames = 1;
  f.video.getVideoPlaybackQuality = () => ({ totalVideoFrames: frames } as VideoPlaybackQuality);
  f.frame(0); vi.runOnlyPendingTimers(); f.frame(1); vi.runOnlyPendingTimers();
  expect(f.work).toHaveBeenCalledTimes(1);
  frames++; f.frame(2); vi.runOnlyPendingTimers(); expect(f.work).toHaveBeenCalledTimes(2); f.stop();
});
it('uses presented-frame identity with coarse media timestamps and ignores invalid metadata', () => {
  const f = setup(true);
  f.frame(NaN, 1); vi.runOnlyPendingTimers(); expect(f.work).not.toHaveBeenCalled();
  f.frame(0, 1); vi.runOnlyPendingTimers();
  f.frame(0, 2); vi.runOnlyPendingTimers(); expect(f.work).toHaveBeenCalledTimes(2);
  f.frame(1, 2); vi.runOnlyPendingTimers(); expect(f.work).toHaveBeenCalledTimes(2); f.stop();
});
