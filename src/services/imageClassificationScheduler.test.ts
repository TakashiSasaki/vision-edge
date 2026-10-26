import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { ImageClassificationEngine, type ImageClassificationTask } from './imageClassificationEngine';
import { startImageClassificationScheduler } from './imageClassificationScheduler';
import { JobLock } from './frameScheduler';
import type { WholeImageClassificationFrame } from '../types/classification';
let raf: Map<number, FrameRequestCallback>, native: Map<number, VideoFrameRequestCallback>, next: number;
let frames: { close: ReturnType<typeof vi.fn> }[];
let stops: (() => void)[], engines: ImageClassificationEngine[];
beforeEach(() => {
  vi.useFakeTimers(); raf = new Map(); native = new Map(); next = 0; frames = []; stops = []; engines = [];
  vi.stubGlobal('requestAnimationFrame', vi.fn((cb: FrameRequestCallback) => { raf.set(++next, cb); return next; }));
  vi.stubGlobal('cancelAnimationFrame', vi.fn((id: number) => raf.delete(id)));
  vi.stubGlobal('VideoFrame', class { displayWidth = 640; displayHeight = 480; timestamp: number; close = vi.fn(); constructor(video: HTMLVideoElement) { this.timestamp = video.currentTime * 1e6; frames.push(this); } });
});
afterEach(() => { stops.forEach(stop => stop()); engines.forEach(e => e.dispose()); vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });
async function fixture(useNative = false) {
  let current = true;
  const task = { sdkDelegate: 'GPU' as const, classifyForVideo: vi.fn<ImageClassificationTask['classifyForVideo']>(() => ({ classifications: [{ headIndex: 0, headName: '', categories: [{ index: 7, categoryName: 'raw', displayName: '', score: .8 }] }] })), close: vi.fn<ImageClassificationTask['close']>() };
  const e = new ImageClassificationEngine(async () => task); engines.push(e); await e.initialize();
  const video = Object.defineProperties(document.createElement('video'), { readyState: { value: 2 }, videoWidth: { value: 640 }, videoHeight: { value: 480 } });
  if (useNative) { video.requestVideoFrameCallback = vi.fn(cb => { native.set(++next, cb); return next; }); video.cancelVideoFrameCallback = vi.fn(id => { native.delete(id); }); }
  const adopt = vi.fn<(frame: WholeImageClassificationFrame) => void>(), error = vi.fn(), lock = new JobLock();
  const start = () => { const stop = startImageClassificationScheduler({ engine: e, video, current: () => current, adopt, error, input: vi.fn(), lock }); stops.push(stop); return stop; };
  const tick = (time: number) => { video.currentTime = time; const callbacks = useNative ? native : raf, [id, cb] = callbacks.entries().next().value!; callbacks.delete(id); if (useNative) (cb as VideoFrameRequestCallback)(0, { mediaTime: time, presentedFrames: time + 1 } as VideoFrameCallbackMetadata); else (cb as FrameRequestCallback)(0); vi.runOnlyPendingTimers(); };
  return { e, task, video, adopt, error, lock, start, tick, invalidate() { current = false; } };
}
it.each([false, true])('whole-frame scheduler bootstrap/dedup/progress/stop and cleanup %s', async useNative => {
  const f = await fixture(useNative), stop = f.start(); vi.runOnlyPendingTimers(); expect(f.adopt).toHaveBeenCalledTimes(1);
  f.tick(0); expect(f.task.classifyForVideo).toHaveBeenCalledTimes(1); f.tick(1); expect(f.adopt).toHaveBeenCalledTimes(2);
  const accepted = f.adopt.mock.calls[1][0]; expect(accepted.kind).toBe('whole-image-classification'); expect(f.e.isCurrentFrame(accepted)).toBe(true);
  f.invalidate(); stop(); stop(); expect(f.e.isCurrentFrame(accepted)).toBe(false); expect(raf.size + native.size).toBe(0);
  expect(frames.length).toBeGreaterThan(0); for (const frame of frames) expect(frame.close).toHaveBeenCalledTimes(1);
  const token = f.lock.acquire(); expect(token).not.toBeNull(); if (token) f.lock.release(token);
});
it('pending old scheduler cannot adopt after variant switch or clear output belonging to replacement', async () => {
  const f = await fixture(), stop = f.start(); await f.e.initialize('lite2'); const newFrame = f.e.classifyVideoFrame(Object.assign(document.createElement('canvas'), { width: 640, height: 480 }));
  vi.runOnlyPendingTimers(); expect(f.adopt).not.toHaveBeenCalled(); stop(); if (newFrame.status === 'ready') expect(f.e.isCurrentFrame(newFrame)).toBe(true);
});
it('work reentry invalidation suppresses adoption and releases captured frame and lock', async () => {
  const f = await fixture(); f.task.classifyForVideo.mockImplementationOnce(() => { f.e.clearResults(); return { classifications: [] }; }); f.start(); vi.runOnlyPendingTimers(); expect(f.adopt).not.toHaveBeenCalled();
  for (const frame of frames) expect(frame.close).toHaveBeenCalledTimes(1); const token = f.lock.acquire(); expect(token).not.toBeNull(); if (token) f.lock.release(token);
});
it('actual SDK failure reaches error, never adoption, and resources still close', async () => {
  const f = await fixture(); f.task.classifyForVideo.mockImplementationOnce(() => { throw Error('SCHEDULER_SDK_FAILURE'); }); f.start(); vi.runOnlyPendingTimers(); expect(f.error).toHaveBeenCalledWith(expect.objectContaining({ stage: 'run' })); expect(f.adopt).not.toHaveBeenCalled(); for (const frame of frames) expect(frame.close).toHaveBeenCalledTimes(1);
});
it('starting without accepted initialization is an explicit error, not zero-call success', () => {
  const e = new ImageClassificationEngine(); engines.push(e); expect(() => startImageClassificationScheduler({ engine: e, video: document.createElement('video'), current: () => true, adopt: vi.fn(), error: vi.fn(), input: vi.fn(), lock: new JobLock() })).toThrow('requires a ready model');
});

it('native live-video fallback stays live video with no invented immutable-input binding', async () => {
  vi.stubGlobal('VideoFrame', undefined); const f = await fixture();
  f.video.getVideoPlaybackQuality = vi.fn(() => ({ totalVideoFrames: 1 }) as VideoPlaybackQuality);
  f.start(); vi.runOnlyPendingTimers(); expect(f.adopt).toHaveBeenCalledTimes(1);
  expect(f.task.classifyForVideo.mock.calls[0][0]).toBe(f.video);
});
it('owned Canvas fallback passes its full snapshot and releases only the owned input after work', async () => {
  vi.stubGlobal('VideoFrame', undefined);
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function(this: HTMLCanvasElement) {
    return { drawImage: vi.fn(), getImageData: () => ({ width: this.width, height: this.height, data: new Uint8ClampedArray(this.width * this.height * 4) }), putImageData: vi.fn() } as unknown as CanvasRenderingContext2D;
  });
  const f = await fixture(); let input: HTMLCanvasElement | undefined;
  f.task.classifyForVideo.mockImplementationOnce(source => { expect(source).toBeInstanceOf(HTMLCanvasElement); input = source as HTMLCanvasElement; expect([input.width, input.height]).toEqual([640,480]); return { classifications: [] }; });
  f.start(); vi.runOnlyPendingTimers(); expect(f.adopt).toHaveBeenCalledTimes(1); expect(input).toBeDefined(); expect([input!.width, input!.height]).toEqual([0,0]);
});
