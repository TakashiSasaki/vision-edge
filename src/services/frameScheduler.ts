import type { InferenceInput } from '../types/vision';
import { PipelineProbe, type PipelineObserver, type InferenceMeasurement, type AttemptOutcome } from './pipelineMetrics';

// An execution owns its token across cancellation and replacement sessions.
export class JobLock {
  private owner: symbol | null = null;
  acquire(): symbol | null { if (this.owner) return null; return this.owner = Symbol('inference'); }
  release(token: symbol): void { if (this.owner === token) this.owner = null; }
}
interface SchedulerOptions<T> {
  video: HTMLVideoElement;
  bootstrap?: boolean;
  current: () => boolean;
  work: (input: InferenceInput, measurement?: InferenceMeasurement) => T | Promise<T>;
  adopt: (result: T, measurement?: InferenceMeasurement) => void;
  instrumentation?: PipelineObserver;
  error: (cause: unknown) => void;
  captureError?: (cause: unknown) => void;
  input: () => void;
  lock: JobLock;
}
// No configured rate cap. One latest candidate; never a FIFO of video frames.
// The timer yields to the browser; synchronous MediaPipe still runs on main.
export function startFrameScheduler<T>(options: SchedulerOptions<T>): () => void {
  const { video, current, work, adopt, error, input, lock } = options;
  const probe = options.instrumentation ? new PipelineProbe(options.instrumentation) : undefined;
  const measurementGeneration = probe?.generation ?? (probe ? Symbol('scheduler-generation') : undefined);
  let captureId = 0;
  let jobId = 0;
  let stopped = false;
  let callback: number | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let reserved: ReturnType<PipelineProbe['start']>;
  let pending = false;
  let latestArrival: { key: string; mediaTime: number } | null = null;
  type Pixels = { width: number; height: number; data: Uint8ClampedArray };
  type Progress = { time: number; qualityFrames: number | null; frameTime: number | null; pixels: Pixels | null };
  const samePixels = (a: Pixels | null, b: Pixels | null) => {
    const start = probe?.start();
    try { return a !== null && b !== null
      && a.width === b.width && a.height === b.height && a.data.length === b.data.length
      && a.data.every((value, index) => value === b.data[index]); }
    finally { probe?.end('comparison', start); }
  };
  let observedPixels: Pixels | null = null;
  let pixelCanvas: HTMLCanvasElement | null = null;
  let pixelContext: CanvasRenderingContext2D | null = null;
  let consumed: { progress: Progress; arrivalKey: string | null } | null = null;
  let lastObserved: string | null = null;
  let running = false;
  const useVideoCallback = typeof video.requestVideoFrameCallback === 'function' && typeof video.cancelVideoFrameCallback === 'function';
  const valid = () => !stopped && current() && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0;
  const stop = () => {
    stopped = true; pending = false; consumed = null; observedPixels = null;
    if (pixelCanvas) { pixelCanvas.width = 0; pixelCanvas.height = 0; }
    pixelCanvas = null; pixelContext = null;
    if (timer !== null) { clearTimeout(timer); probe?.end('queue', reserved, 'cancelled'); timer = null; }
    if (callback !== null) {
      if (useVideoCallback) video.cancelVideoFrameCallback(callback);
      else cancelAnimationFrame(callback);
    }
    // Only a running job releases its own lock and captured image.
  };
  const captureFailed = (cause: unknown) => {
    if (!valid()) return;
    stop();
    (options.captureError ?? error)(cause);
  };
  // A VideoFrame carries the source image's own timestamp, unlike currentTime
  // (playhead) and playback-quality totals (which can include decoded-ahead data).
  // Keep this immutable image alive until its owning job finishes.
  const captureRaw = (ownInput = false): { source: InferenceInput; progress: Progress; release: () => void } | null => {
    const time = video.currentTime;
    if (!Number.isFinite(time) || time < 0) return null;
    if (typeof VideoFrame === 'function') {
      const frame = new VideoFrame(video);
      if (!Number.isFinite(frame.timestamp) || frame.timestamp < 0) { frame.close(); return null; }
      return { source: frame, progress: { time, qualityFrames: null, frameTime: frame.timestamp, pixels: null }, release: () => frame.close() };
    }
    const frames = video.getVideoPlaybackQuality?.()?.totalVideoFrames;
    if (Number.isFinite(frames) && frames! > 0)
      return { source: video, progress: { time, qualityFrames: frames!, frameTime: null, pixels: null }, release() {} };
    // A running playhead is not source progress. When no image timestamp or
    // usable counter exists, bind a private full-resolution pixel snapshot to
    // its exact bytes. Identical content can coalesce; changed pixels cannot.
    const canvas = pixelCanvas ??= document.createElement('canvas');
    if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
    if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
    const context = pixelContext ??= canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('Pixel capture context unavailable');
    context.drawImage(video, 0, 0);
    const image = context.getImageData(0, 0, canvas.width, canvas.height);
    if (image.data.length !== canvas.width * canvas.height * 4) throw new Error('Incomplete pixel capture');
    const pixels = { width: canvas.width, height: canvas.height, data: image.data };
    const progress = { time, qualityFrames: null, frameTime: null, pixels };
    if (!ownInput) return { source: video, progress, release() {} };
    const owned = document.createElement('canvas'); owned.width = canvas.width; owned.height = canvas.height;
    try {
      const copy = owned.getContext('2d');
      if (!copy) throw new Error('Owned pixel snapshot unavailable');
      copy.putImageData(image, 0, 0);
      return { source: owned, progress, release() { owned.width = 0; owned.height = 0; } };
    } catch (cause) { owned.width = 0; owned.height = 0; throw cause; }
  };
  const captureInput = (ownInput = false): (NonNullable<ReturnType<typeof captureRaw>> & { measurement?: InferenceMeasurement }) | null => {
    const start = probe?.start();
    const id = probe ? ++captureId : 0;
    let outcome: AttemptOutcome = 'failed';
    let captured: ReturnType<typeof captureRaw>;
    let end: number | undefined;
    try { captured = captureRaw(ownInput); outcome = captured ? 'success' : 'cancelled'; }
    finally { end = probe?.end('capture', start, outcome); }
    if (!captured) return null;
    if (!probe) return captured;
    return { ...captured, measurement: {
      generation: measurementGeneration!, jobId: 0, captureId: id, captureStart: start?.at ?? null, captureEnd: end ?? null,
      immutable: captured.source !== video,
    } satisfies InferenceMeasurement };
  };
  const release = (captured: ReturnType<typeof captureInput>) => {
    if (!captured) return;
    const start = probe?.start();
    let outcome: AttemptOutcome = 'failed';
    try { captured.release(); outcome = 'success'; }
    finally { probe?.end('cleanup', start, outcome); }
  };
  const arrivalKey = (progress: Progress) => latestArrival?.mediaTime === progress.time ? latestArrival.key : null;
  const alreadyConsumed = (progress: Progress): boolean => {
    if (!consumed) return false;
    const previous = consumed.progress;
    if (progress.pixels !== null) return samePixels(progress.pixels, previous.pixels);
    if (progress.frameTime !== null && previous.frameTime !== null) return progress.frameTime === previous.frameTime;
    if (progress.qualityFrames !== null && previous.qualityFrames !== null)
      return progress.qualityFrames === previous.qualityFrames;
    const key = arrivalKey(progress);
    if (progress.qualityFrames === null && previous.qualityFrames === null && key !== null && key === consumed.arrivalKey)
      return true;
    if (progress.time !== previous.time) return false;
    // With only currentTime available, reconcile a delayed rVFC notification
    // with an image consumed ahead of that notification (including bootstrap).
    if (!useVideoCallback || key === null) return true;
    if (consumed.arrivalKey === null) { consumed.arrivalKey = key; return true; }
    // A new presented frame can share a coarse currentTime value.
    return consumed.arrivalKey === key;
  };
  const observe = (progress: Progress) => {
    if (progress.pixels !== null) {
      if (!samePixels(progress.pixels, observedPixels)) { observedPixels = progress.pixels; input(); }
      return;
    }
    const key = progress.frameTime !== null ? `image:${progress.frameTime}` : progress.qualityFrames !== null ? `quality:${progress.qualityFrames}`
      : `time:${progress.time}:${useVideoCallback ? arrivalKey(progress) ?? '' : ''}`;
    if (key !== lastObserved) { lastObserved = key; input(); }
  };
  const reserve = () => {
    if (!pending || running || timer !== null || !valid()) return;
    reserved = probe?.start();
    timer = setTimeout(() => {
      timer = null;
      if (!pending || !valid()) { probe?.end('queue', reserved, 'cancelled'); return; }
      const token = lock.acquire();
      probe?.end('queue', reserved, token ? 'success' : 'cancelled');
      if (!token) return; // Next acquisition tick retries only the latest candidate.
      pending = false; running = true;
      const jobStart = probe?.start();
      let jobOutcome: AttemptOutcome = 'cancelled';
      let jobMeasured = false;
      let captured: ReturnType<typeof captureInput> = null;
      const finish = () => {
        const owned = captured; captured = null;
        try { release(owned); }
        finally {
          if (!jobMeasured) { probe?.end('job', jobStart, stopped && jobOutcome !== 'failed' ? 'cancelled' : jobOutcome); jobMeasured = true; }
          running = false; lock.release(token); if (!stopped) reserve();
        }
      };
      const failed = (cause: unknown) => { jobOutcome = 'failed'; if (valid()) error(cause); };
      const completed = (value: T, measurement?: InferenceMeasurement) => {
        if (valid()) {
          if (measurement) adopt(value, measurement); else adopt(value);
          jobOutcome = 'success';
        } else jobOutcome = stopped ? 'cancelled' : 'stale';
      };
      try {
        if (!valid()) { finish(); return; }
        try { captured = captureInput(true); }
        catch (cause) { jobOutcome = 'failed'; captureFailed(cause); finish(); return; }
        if (!captured || !valid() || alreadyConsumed(captured.progress)) { finish(); return; }
        const { progress, source } = captured;
        observe(progress);
        if (!valid()) { finish(); return; }
        // Attempts, including failed work, consume this live input. Mark before
        // the call so reentry or a later callback cannot submit it again.
        consumed = { progress, arrivalKey: arrivalKey(progress) };
        const measurement = captured.measurement;
        if (measurement) measurement.jobId = ++jobId;
        const result = measurement ? work(source, measurement) : work(source);
        if (result instanceof Promise) {
          void result.then(value => completed(value, measurement), failed).catch(failed).finally(finish);
        } else {
          completed(result, measurement);
          finish();
        }
      } catch (cause) { try { failed(cause); } finally { finish(); } }
    }, 0);
  };
  const arrive = () => {
    if (!valid()) return;
    let sampled: ReturnType<typeof captureInput> = null;
    try {
      sampled = captureInput();
      if (!sampled || !valid() || alreadyConsumed(sampled.progress)) return;
      pending = true;
      observe(sampled.progress);
      reserve();
    } catch (cause) { captureFailed(cause); }
    finally { release(sampled); }
  };
  const next = () => {
    if (stopped) return;
    if (useVideoCallback) callback = video.requestVideoFrameCallback((_now, metadata) => {
      callback = null;
      // presentedFrames distinguishes frames even if mediaTime has coarse precision.
      if (Number.isFinite(metadata.mediaTime) && metadata.mediaTime >= 0 && Number.isFinite(metadata.presentedFrames) && metadata.presentedFrames >= 0) {
        latestArrival = { mediaTime: metadata.mediaTime, key: metadata.presentedFrames > 0 ? `frame:${metadata.presentedFrames}` : `time:${metadata.mediaTime}` };
        arrive();
      }
      next();
    });
    else callback = requestAnimationFrame(() => {
      callback = null;
      arrive();
      next();
    });
  };
  next();
  // A frame may have been presented while the model was loading. rVFC only
  // reports future presentations, so bootstrap the current usable image once.
  // It shares the same consumed-input check as arrivals, preventing races.
  if (options.bootstrap !== false) arrive();
  return stop;
}
