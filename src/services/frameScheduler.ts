// An execution owns its token across cancellation and replacement sessions.
export class JobLock {
  private owner: symbol | null = null;
  acquire(): symbol | null { if (this.owner) return null; return this.owner = Symbol('inference'); }
  release(token: symbol): void { if (this.owner === token) this.owner = null; }
}
interface SchedulerOptions<T> {
  video: HTMLVideoElement;
  current: () => boolean;
  work: () => T | Promise<T>;
  adopt: (result: T) => void;
  error: (cause: unknown) => void;
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
  let lastKey: string | null = null;
  let running = false;
  const useVideoCallback = typeof video.requestVideoFrameCallback === 'function' && typeof video.cancelVideoFrameCallback === 'function';
  const valid = () => !stopped && current() && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0;
  const reserve = () => {
    if (!pending || running || timer !== null || !valid()) return;
    timer = setTimeout(() => {
      timer = null;
      if (!pending || !valid()) return;
      const token = lock.acquire();
      if (!token) return; // Next acquisition tick retries only the latest candidate.
      pending = false; running = true;
      const finish = () => { running = false; lock.release(token); if (!stopped) reserve(); };
      const failed = (cause: unknown) => { if (valid()) error(cause); };
      try {
        if (!valid()) { finish(); return; }
        const result = work();
        if (result instanceof Promise) {
          void result.then(value => { if (valid()) adopt(value); }, failed).catch(failed).finally(finish);
        } else {
          if (valid()) adopt(result);
          finish();
        }
      } catch (cause) { try { failed(cause); } finally { finish(); } }
    }, 0);
  };
  const arrive = (key: string) => {
    if (!valid()) return;
    if (key !== lastKey) { lastKey = key; pending = true; input(); }
    reserve();
  };
  const next = () => {
    if (stopped) return;
    if (useVideoCallback) callback = video.requestVideoFrameCallback((_now, metadata) => {
      callback = null;
      // presentedFrames distinguishes frames even if mediaTime has coarse precision.
      if (Number.isFinite(metadata.mediaTime) && metadata.mediaTime >= 0 && Number.isFinite(metadata.presentedFrames) && metadata.presentedFrames >= 0)
        arrive(metadata.presentedFrames > 0 ? `frame:${metadata.presentedFrames}` : `time:${metadata.mediaTime}`);
      next();
    });
    else callback = requestAnimationFrame(() => {
      callback = null;
      const time = video.currentTime;
      const frames = video.getVideoPlaybackQuality?.().totalVideoFrames;
      if (Number.isFinite(time) && time >= 0) arrive(Number.isFinite(frames) && frames! > 0 ? `frame:${frames}` : `time:${time}`);
      next();
    });
  };
  next();
  return () => {
    stopped = true; pending = false;
    if (timer !== null) clearTimeout(timer);
    if (callback !== null) {
      if (useVideoCallback) video.cancelVideoFrameCallback(callback);
      else cancelAnimationFrame(callback);
    }
    // A running job alone releases its lock. Cleanup cannot unlock its successor.
  };
}
