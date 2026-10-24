import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { JobLock, startFrameScheduler } from './frameScheduler';
let raf: Map<number, FrameRequestCallback>;
let rvfc: Map<number, VideoFrameRequestCallback>;
let id: number;
let pixelIds: WeakMap<HTMLVideoElement, number>;
beforeEach(() => {
  pixelIds = new WeakMap();
  vi.useFakeTimers(); raf = new Map(); rvfc = new Map(); id = 0;
  const contexts = new WeakMap<HTMLCanvasElement, CanvasRenderingContext2D>();
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function(this: HTMLCanvasElement) {
    const previous = contexts.get(this); if (previous) return previous;
    let image = 0;
    const context = { putImageData: (pixels: ImageData) => { image = new DataView(pixels.data.buffer).getFloat64(0, true); }, drawImage: (video: HTMLVideoElement) => { image = pixelIds.get(video) ?? video.currentTime; },
      getImageData: (_x: number, _y: number, width: number, height: number) => {
        const data = new Uint8ClampedArray(width * height * 4); new DataView(data.buffer).setFloat64(0, image, true);
        return { data, width, height };
      } } as unknown as CanvasRenderingContext2D;
    contexts.set(this, context); return context;
  });
  vi.stubGlobal('requestAnimationFrame', vi.fn((fn: FrameRequestCallback) => { raf.set(++id, fn); return id; }));
  vi.stubGlobal('cancelAnimationFrame', vi.fn((key: number) => raf.delete(key)));
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });
function setup(native = false, lock = new JobLock(), dimensions = { width: 640, height: 480 }) {
  let current = true;
  const video = Object.defineProperties(document.createElement('video'), {
    readyState: { configurable: true, value: 2 }, videoWidth: { value: dimensions.width, configurable: true }, videoHeight: { value: dimensions.height, configurable: true },
  });
  if (native) {
    video.requestVideoFrameCallback = vi.fn(fn => { rvfc.set(++id, fn); return id; });
    video.cancelVideoFrameCallback = vi.fn(key => { rvfc.delete(key); });
  }
  const work = vi.fn<() => number | Promise<number>>(() => 1);
  const adopt = vi.fn(); const error = vi.fn(); const input = vi.fn();
  const stop = startFrameScheduler({ video, current: () => current, work, adopt, error, input, lock });
  const frame = (time: number, count = time) => {
    if (Number.isFinite(time)) video.currentTime = time;
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
  vi.runOnlyPendingTimers(); expect(f.work).toHaveBeenCalledTimes(1); // Bootstrap the already-ready image.
  f.frame(NaN, 1); vi.runOnlyPendingTimers(); expect(f.work).toHaveBeenCalledTimes(1);
  f.frame(0, 1); vi.runOnlyPendingTimers();
  pixelIds.set(f.video, 2); // A genuinely different image at the same coarse media time.
  f.frame(0, 2); vi.runOnlyPendingTimers(); expect(f.work).toHaveBeenCalledTimes(2);
  f.frame(1, 2); vi.runOnlyPendingTimers(); expect(f.work).toHaveBeenCalledTimes(2); f.stop();
});
it.each([false, true])('F01: samples live input immediately before execution and reconciles its later arrival (%s)', native => {
  const f = setup(native);
  f.work.mockImplementation(() => f.video.currentTime);
  f.frame(1);
  f.video.currentTime = 2; // Decoder progresses without delivering B's callback.
  vi.runOnlyPendingTimers();
  f.frame(2); vi.runOnlyPendingTimers();
  expect(f.work.mock.results.map(result => result.value)).toEqual([2]);
  expect(f.adopt).toHaveBeenCalledTimes(1);
  f.frame(3); vi.runOnlyPendingTimers();
  expect(f.work.mock.results.map(result => result.value)).toEqual([2, 3]);
  f.stop();
});
it.each([false, true])('F01: uses execution-time playback quality, never the unrelated arrival-counter domain (%s)', native => {
  const f = setup(native); let frames = 4;
  f.video.getVideoPlaybackQuality = () => ({ totalVideoFrames: frames } as VideoPlaybackQuality);
  f.work.mockImplementation(() => frames);
  f.frame(0.05, 2);
  frames = 12; f.video.currentTime = 0.385963;
  vi.runOnlyPendingTimers();
  // Reproduce independent native evidence: quality 12, presentation 8, and
  // a continuously advancing playback time all describe the same consumed image.
  f.frame(0.386066, 8); vi.runOnlyPendingTimers();
  expect(f.work.mock.results.map(result => result.value)).toEqual([12]);
  frames = 13; f.frame(0.45, 9); vi.runOnlyPendingTimers();
  expect(f.work.mock.results.map(result => result.value)).toEqual([12, 13]); f.stop();
});
it.each([false, true])('F02: bootstraps a pre-presented image once, including a new session at the same zero time (%s)', native => {
  for (let session = 0; session < 2; session++) {
    const f = setup(native); vi.advanceTimersByTime(400);
    expect(f.work).toHaveBeenCalledTimes(1);
    f.frame(0, 1); vi.runOnlyPendingTimers();
    expect(f.work).toHaveBeenCalledTimes(1); // Late callback reconciles bootstrap.
    f.stop(); expect(raf.size + rvfc.size).toBe(0);
  }
});
it.each([false, true])('bootstrap and arrival share latest-only work when the live frame advances before execution (%s)', native => {
  const f = setup(native); let frames = 1;
  f.video.getVideoPlaybackQuality = () => ({ totalVideoFrames: frames } as VideoPlaybackQuality);
  f.work.mockImplementation(() => frames);
  frames = 2; f.frame(0.02, 2); vi.runOnlyPendingTimers();
  f.frame(0.02, 2); vi.runOnlyPendingTimers();
  expect(f.work.mock.results.map(result => result.value)).toEqual([2]);
  f.stop();
});
it.each([false, true])('a failed live B is not retried when its delayed callback follows A (%s)', native => {
  const f = setup(native); f.work.mockImplementationOnce(() => { throw new Error('SDK'); });
  f.frame(1); f.video.currentTime = 2; vi.runOnlyPendingTimers();
  f.frame(2); vi.runOnlyPendingTimers();
  expect(f.work).toHaveBeenCalledTimes(1); expect(f.error).toHaveBeenCalledTimes(1);
  f.frame(3); vi.runOnlyPendingTimers(); expect(f.adopt).toHaveBeenCalledTimes(1); f.stop();
});
function installFrameCapture(videoTime: () => number) {
  const frames: Array<{ timestamp: number; close: ReturnType<typeof vi.fn> }> = [];
  class CapturedFrame {
    timestamp = videoTime() * 1000000;
    displayWidth = 640; displayHeight = 480;
    close = vi.fn();
    constructor() { frames.push(this); }
  }
  vi.stubGlobal('VideoFrame', CapturedFrame);
  return frames;
}
it.each([false, true])('immutable input binds the consumed image even while playhead/quality/arrival identities diverge (%s)', native => {
  let imageTime = 1; const frames = installFrameCapture(() => imageTime); const f = setup(native);
  f.video.getVideoPlaybackQuality = () => ({ totalVideoFrames: 7 } as VideoPlaybackQuality);
  const images: number[] = [];
  f.work.mockImplementation((input?: HTMLVideoElement | VideoFrame) => {
    images.push((input as VideoFrame).timestamp);
    imageTime = 3; // Live source advances after capture, but SDK still owns B.
    expect((input as VideoFrame).timestamp).toBe(images.at(-1));
    return images.length;
  });
  f.frame(0.05, 2); imageTime = 2; vi.runOnlyPendingTimers();
  expect(images).toEqual([2000000]);
  imageTime = 2; f.frame(0.386066, 8); vi.runOnlyPendingTimers(); expect(images).toEqual([2000000]);
  imageTime = 3; f.frame(0.40, 9); vi.runOnlyPendingTimers(); expect(images).toEqual([2000000, 3000000]);
  f.stop(); expect(frames.every(frame => frame.close.mock.calls.length === 1)).toBe(true);
});
it('captured input survives an old async job until completion; cleanup never releases a foreign image or lock', async () => {
  let imageTime = 0; const frames = installFrameCapture(() => imageTime); const lock = new JobLock(); const f = setup(true, lock);
  let resolve!: (value: number) => void;
  f.work.mockImplementationOnce(() => new Promise<number>(yes => { resolve = yes; }));
  vi.runOnlyPendingTimers();
  const owned = frames.find(frame => frame.close.mock.calls.length === 0)!;
  expect(owned).toBeDefined(); f.stop(); expect(owned.close).not.toHaveBeenCalled();
  imageTime = 1; const g = setup(true, lock); vi.runOnlyPendingTimers(); expect(g.work).not.toHaveBeenCalled();
  resolve(1); await vi.runAllTimersAsync(); expect(owned.close).toHaveBeenCalledTimes(1); expect(f.adopt).not.toHaveBeenCalled();
  g.frame(1, 1); vi.runOnlyPendingTimers(); expect(g.work).toHaveBeenCalledTimes(1); g.stop();
  expect(frames.every(frame => frame.close.mock.calls.length === 1)).toBe(true);
});
it('failed immutable input is closed once, deduplicated across late arrival and recovers on the next image', () => {
  let imageTime = 0; const frames = installFrameCapture(() => imageTime); const f = setup(true);
  f.work.mockImplementationOnce(() => { throw new Error('SDK'); });
  vi.runOnlyPendingTimers(); f.frame(0.2, 1); vi.runOnlyPendingTimers(); expect(f.work).toHaveBeenCalledTimes(1);
  imageTime = 1; f.frame(1, 2); vi.runOnlyPendingTimers(); expect(f.adopt).toHaveBeenCalledTimes(1); f.stop();
  expect(frames.every(frame => frame.close.mock.calls.length === 1)).toBe(true);
});
it('capture exceptions stop acquisition deliberately without leaking callbacks or locks; a fresh retry can run', () => {
  const error = vi.fn(); vi.stubGlobal('VideoFrame', class { constructor() { throw new Error('capture'); } });
  const video = Object.defineProperties(document.createElement('video'), { readyState: { value: 2 }, videoWidth: { value: 640 }, videoHeight: { value: 480 } });
  const lock = new JobLock(); const work = vi.fn();
  startFrameScheduler({ video, lock, current: () => true, work, adopt: vi.fn(), error, input: vi.fn() });
  expect(error).toHaveBeenCalledTimes(1); expect(raf.size + rvfc.size).toBe(0); expect(work).not.toHaveBeenCalled();
  vi.stubGlobal('VideoFrame', undefined); const f = setup(false, lock); vi.runOnlyPendingTimers(); expect(f.work).toHaveBeenCalledTimes(1); f.stop();
});
it.each([NaN, -1, Infinity])('rejects invalid captured-image timestamp %s and closes every temporary handle', timestamp => {
  const frames = installFrameCapture(() => timestamp); const f = setup(true);
  vi.runOnlyPendingTimers(); f.frame(0, 1); vi.runOnlyPendingTimers();
  expect(f.work).not.toHaveBeenCalled(); f.stop();
  expect(frames.every(frame => frame.close.mock.calls.length === 1)).toBe(true);
});
it('capture reentry invalidates the job before SDK work and still closes its owned image', () => {
  let cancel = () => {}; const closes: Array<ReturnType<typeof vi.fn>> = [];
  vi.stubGlobal('VideoFrame', class {
    timestamp = 0; displayWidth = 640; displayHeight = 480; close = vi.fn();
    constructor() { closes.push(this.close); if (closes.length === 2) cancel(); }
  });
  const f = setup(true); cancel = f.invalidate; vi.runOnlyPendingTimers(); f.stop();
  expect(f.work).not.toHaveBeenCalled(); expect(closes.every(close => close.mock.calls.length === 1)).toBe(true);
});
it('capture failure after acquiring the job token releases that token and permits a fresh retry', () => {
  const lock = new JobLock(); let captures = 0; const close = vi.fn();
  vi.stubGlobal('VideoFrame', class {
    timestamp = 0; displayWidth = 640; displayHeight = 480; close = close;
    constructor() { if (++captures === 2) throw new Error('execution capture'); }
  });
  const f = setup(true, lock); vi.runOnlyPendingTimers();
  expect(f.work).not.toHaveBeenCalled(); expect(f.error).toHaveBeenCalledTimes(1); expect(close).toHaveBeenCalledTimes(1);
  expect(raf.size + rvfc.size).toBe(0); f.stop();
  vi.stubGlobal('VideoFrame', undefined); const next = setup(false, lock); vi.runOnlyPendingTimers(); expect(next.work).toHaveBeenCalledTimes(1); next.stop();
});
it.each([false, true])('pixel-only input ignores playhead/metadata drift, accepts changed images and releases owned canvases (%s)', native => {
  const f = setup(native); pixelIds.set(f.video, 0);
  const images: number[] = []; const owned: HTMLCanvasElement[] = [];
  f.work.mockImplementation((input?: HTMLVideoElement | VideoFrame | HTMLCanvasElement) => {
    const canvas = input as HTMLCanvasElement; owned.push(canvas);
    images.push(new DataView(canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data.buffer).getFloat64(0, true));
    return images.length;
  });
  vi.runOnlyPendingTimers();
  for (let i = 1; i <= 60; i++) { f.frame(i / 10, i); vi.advanceTimersByTime(50); }
  expect(images).toEqual([0]); expect(f.input).toHaveBeenCalledTimes(1);
  pixelIds.set(f.video, 7); f.frame(6, 60); vi.runOnlyPendingTimers();
  expect(images).toEqual([0, 7]); expect(f.input).toHaveBeenCalledTimes(2);
  f.stop(); expect(owned.every(canvas => canvas.width === 0 && canvas.height === 0)).toBe(true);
});
it.each([false, true])('pixel snapshot binds late B exactly and never retries a failed unchanged B (%s)', native => {
  const f = setup(native); const pixels: number[] = [];
  f.work.mockImplementation((input?: HTMLVideoElement | VideoFrame | HTMLCanvasElement) => {
    const canvas = input as HTMLCanvasElement;
    pixels.push(new DataView(canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data.buffer).getFloat64(0, true));
    throw new Error('SDK');
  });
  pixelIds.set(f.video, 1); f.frame(1, 1);
  pixelIds.set(f.video, 2); f.video.currentTime = 2; vi.runOnlyPendingTimers();
  f.frame(2.1, 2); vi.runOnlyPendingTimers(); f.frame(3, 3); vi.runOnlyPendingTimers();
  expect(pixels).toEqual([2]); expect(f.error).toHaveBeenCalledTimes(1);
  f.work.mockReturnValue(7); pixelIds.set(f.video, 3); f.frame(3, 3); vi.runOnlyPendingTimers();
  expect(f.adopt).toHaveBeenCalledWith(7); expect(f.work).toHaveBeenCalledTimes(2); f.stop();
});
it('pixel comparison checks every byte and dimensions rather than a sparse hash or clock', () => {
  let current = new Uint8ClampedArray([255, 0, 0, 255, 0, 0, 255, 255]);
  vi.mocked(HTMLCanvasElement.prototype.getContext).mockImplementation(() => {
    let image = current.slice();
    return { drawImage() { image = current.slice(); }, putImageData(pixels: ImageData) { image = new Uint8ClampedArray(pixels.data); },
      getImageData() { return { data: image.slice() }; } } as unknown as CanvasRenderingContext2D;
  });
  const f = setup(false, new JobLock(), { width: 2, height: 1 }); vi.runOnlyPendingTimers();
  current = new Uint8ClampedArray([0, 0, 255, 255, 255, 0, 0, 255]); // Same sums/histogram, different image.
  f.frame(0); vi.runOnlyPendingTimers(); expect(f.work).toHaveBeenCalledTimes(2);
  Object.defineProperties(f.video, { videoWidth: { value: 1 }, videoHeight: { value: 2 } });
  f.frame(0); vi.runOnlyPendingTimers(); expect(f.work).toHaveBeenCalledTimes(3); f.stop();
});
it('pixel capture/readback failure stops callbacks and releases its job lock for explicit recovery', () => {
  const lock = new JobLock(); const f = setup(false, lock);
  // Acquisition succeeded; make the owned execution snapshot fail after locking.
  vi.mocked(HTMLCanvasElement.prototype.getContext).mockReturnValue(null);
  vi.runOnlyPendingTimers(); expect(f.work).not.toHaveBeenCalled(); expect(f.error).toHaveBeenCalledTimes(1);
  expect(raf.size + rvfc.size).toBe(0); const token = lock.acquire(); expect(token).not.toBeNull(); lock.release(token!); f.stop();
});
