import type { InferenceInput } from '../types/vision';

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
  work: (input: InferenceInput) => T | Promise<T>;
  adopt: (result: T) => void;
  error: (cause: unknown) => void;
  captureError?: (cause: unknown) => void;
  input: () => void;
  lock: JobLock;
}
// No configured rate cap. One latest candidate; never a FIFO of video frames.
// The timer yields to the browser; synchronous MediaPipe still runs on main.
export function startFrameScheduler<T>(options: SchedulerOptions<T>): () => void {
  const { video, current, work, adopt, error, input, lock } = options;
  let stopped = false;
  let callback: number | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pending = false;
  let latestArrival: { key: string; mediaTime: number } | null = null;
  type Progress = { time: number; qualityFrames: number | null; frameTime: number | null };
  let consumed: { progress: Progress; arrivalKey: string | null } | null = null;
  let lastObserved: string | null = null;
  let running = false;
  const useVideoCallback = typeof video.requestVideoFrameCallback === 'function' && typeof video.cancelVideoFrameCallback === 'function';
  const valid = () => !stopped && current() && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0;
  const stop = () => {
    stopped = true; pending = false;
    if (timer !== null) clearTimeout(timer);
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
  const captureInput = (): { source: InferenceInput; progress: Progress; release: () => void } | null => {
    const time = video.currentTime;
    if (!Number.isFinite(time) || time < 0) return null;
    if (typeof VideoFrame === 'function') {
      const frame = new VideoFrame(video);
      if (!Number.isFinite(frame.timestamp) || frame.timestamp < 0) { frame.close(); return null; }
      return { source: frame, progress: { time, qualityFrames: null, frameTime: frame.timestamp }, release: () => frame.close() };
    }
    const frames = video.getVideoPlaybackQuality?.().totalVideoFrames;
    return { source: video, progress: { time, qualityFrames: Number.isFinite(frames) && frames! > 0 ? frames! : null, frameTime: null }, release() {} };
  };
  const arrivalKey = (progress: Progress) => latestArrival?.mediaTime === progress.time ? latestArrival.key : null;
  const alreadyConsumed = (progress: Progress): boolean => {
    if (!consumed) return false;
    const previous = consumed.progress;
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
    const key = progress.frameTime !== null ? `image:${progress.frameTime}` : progress.qualityFrames !== null ? `quality:${progress.qualityFrames}`
      : `time:${progress.time}:${useVideoCallback ? arrivalKey(progress) ?? '' : ''}`;
    if (key !== lastObserved) { lastObserved = key; input(); }
  };
  const reserve = () => {
    if (!pending || running || timer !== null || !valid()) return;
    timer = setTimeout(() => {
      timer = null;
      if (!pending || !valid()) return;
      const token = lock.acquire();
      if (!token) return; // Next acquisition tick retries only the latest candidate.
      pending = false; running = true;
      let captured: ReturnType<typeof captureInput> = null;
      const finish = () => {
        const owned = captured; captured = null;
        try { owned?.release(); }
        finally { running = false; lock.release(token); if (!stopped) reserve(); }
      };
      const failed = (cause: unknown) => { if (valid()) error(cause); };
      try {
        if (!valid()) { finish(); return; }
        try { captured = captureInput(); }
        catch (cause) { captureFailed(cause); finish(); return; }
        if (!captured || !valid() || alreadyConsumed(captured.progress)) { finish(); return; }
        const { progress, source } = captured;
        observe(progress);
        if (!valid()) { finish(); return; }
        // Attempts, including failed work, consume this live input. Mark before
        // the call so reentry or a later callback cannot submit it again.
        consumed = { progress, arrivalKey: arrivalKey(progress) };
        const result = work(source);
        if (result instanceof Promise) {
          void result.then(value => { if (valid()) adopt(value); }, failed).catch(failed).finally(finish);
        } else {
          if (valid()) adopt(result);
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
    finally { sampled?.release(); }
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
