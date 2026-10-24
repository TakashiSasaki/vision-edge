// Synthetic matched-work experiment. Uses real browser Canvas/VideoFrame,
// production scheduler/engine/renderer/statistics/details, deterministic SDKs,
// a manual rVFC driver and a canvas standing in for the camera element.
import '../../../../src/index.css';
import { createRoot } from 'react-dom/client';
import { flushSync } from 'react-dom';
import { PipelineDetails } from '../../../../src/components/PipelineDetails';
import { JobLock, startFrameScheduler } from '../../../../src/services/frameScheduler';
import { VisionEngine, type VisionTaskFactory } from '../../../../src/services/visionEngine';
import { drawCanvasFrame } from '../../../../src/services/canvasRenderer';
import { PipelineMetrics, PipelineProbe, type InferenceMeasurement } from '../../../../src/services/pipelineMetrics';
import type { FrameResult, RunnableModelType } from '../../../../src/types/vision';
import type { MPMask } from '@mediapipe/tasks-vision';
interface TrialOptions { width: number; height: number; path: 'rgba' | 'video-frame'; model: RunnableModelType; enabled: boolean; iterations: number }
const NativeFrame = window.VideoFrame;
const root = createRoot(document.getElementById('hud')!);
const current = () => true;
const stats = (values: number[]) => {
  const sorted = values.slice().sort((a, b) => a - b); const n = sorted.length;
  const mean = values.reduce((sum, v) => sum + v, 0) / n;
  return { count: n, mean, median: (sorted[Math.floor((n - 1) / 2)] + sorted[Math.floor(n / 2)]) / 2,
    p95: sorted[Math.ceil(n * 0.95) - 1], max: sorted[n - 1], variance: values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / n };
};
async function trial(options: TrialOptions) {
  const { width, height, path, model, enabled, iterations } = options;
  const source = document.createElement('canvas'); source.width = width; source.height = height;
  const context = source.getContext('2d')!;
  let scene = 0; let sourceTime = 0; let callback: VideoFrameRequestCallback | undefined;
  const counts = { captured: 0, compared: 0, processed: 0, drawn: 0, sdk: 0, released: 0, snapshot: 0, metricClock: 0, maskRead: 0, maskClosed: 0 };
  Object.defineProperties(source, {
    currentTime: { get: () => sourceTime }, readyState: { value: 2 }, videoWidth: { value: width }, videoHeight: { value: height },
    requestVideoFrameCallback: { value: (fn: VideoFrameRequestCallback) => { callback = fn; return 1; } },
    cancelVideoFrameCallback: { value: () => { callback = undefined; } },
  });
  // Count actual calls at their product boundaries in BOTH modes. The wrappers
  // are identical across modes and do no extra timing. Comparison calls have no
  // external API; every() counts actual full-array scans in both modes. Nullable
  // calls are additionally checked against the deterministic script in enabled runs.
  const proto = HTMLCanvasElement.prototype;
  const originalGetContext = proto.getContext;
  const originalEvery = Uint8ClampedArray.prototype.every;
  Uint8ClampedArray.prototype.every = function(callback, thisArg) { counts.compared++; return originalEvery.call(this, callback, thisArg); };
  const originalCreate = document.createElement.bind(document);
  const contexts = new WeakMap<object, unknown>();
  const patchContext = (canvas: HTMLCanvasElement) => {
    const ctx = originalGetContext.call(canvas, '2d') as CanvasRenderingContext2D;
    if (!ctx || contexts.has(ctx)) return ctx;
    contexts.set(ctx, ctx);
    const read = ctx.getImageData.bind(ctx);
    ctx.getImageData = (...args: Parameters<typeof ctx.getImageData>) => { counts.captured++; return read(...args); };
    return ctx;
  };
  document.createElement = ((tag: string, elementOptions?: ElementCreationOptions) => {
    const element = originalCreate(tag, elementOptions);
    if (tag === 'canvas') patchContext(element as HTMLCanvasElement);
    return element;
  }) as typeof document.createElement;
  class MeasuredFrame {
    constructor() {
      counts.captured++;
      const frame = new NativeFrame(source, { timestamp: sourceTime * 1e6 });
      const close = frame.close.bind(frame);
      frame.close = () => { counts.released++; close(); };
      return frame;
    }
  }
  Object.defineProperty(window, 'VideoFrame', { configurable: true, value: path === 'rgba' ? undefined : MeasuredFrame });
  const floats = new Float32Array(width * height); floats.fill(0.9);
  const mask = { width, height, getAsFloat32Array: () => { counts.maskRead++; return floats; } } as unknown as MPMask;
  const factory: VisionTaskFactory = {
    load: async () => ({ wasmLoaderPath: '', wasmBinaryPath: '' }),
    detector: async () => ({ close() {}, detectForVideo: () => {
      counts.sdk++;
      return { detections: [{ categories: [{ categoryName: 'dog', displayName: '', score: 0.9, index: 0 }],
        boundingBox: { originX: width / 4, originY: height / 4, width: width / 2, height: height / 2, angle: 0 }, keypoints: [] }] };
    } }),
    segmenter: async () => ({ close() {}, segmentForVideo: () => {
      counts.sdk++; return { confidenceMasks: [mask], categoryMask: undefined, close() { counts.maskClosed++; } };
    } }),
  };
  const engine = new VisionEngine(factory); await engine.initialize(model);
  const originalTimeout = window.setTimeout;
  const jobs: Array<() => void> = [];
  // Run exactly the timer callbacks scheduled by production. Avoid timer clamp
  // noise when comparing instrumentation cost, but retain true queue timestamps.
  window.setTimeout = ((handler: TimerHandler) => {
    if (typeof handler !== 'function') throw Error('Unexpected string timer');
    jobs.push(handler as () => void); return jobs.length;
  }) as typeof setTimeout;
  const pipeline = enabled ? new PipelineMetrics(() => { counts.metricClock++; return performance.now(); }) : undefined;
  const probe = pipeline ? new PipelineProbe(pipeline) : undefined;
  const canvas = document.createElement('canvas'); const buffer = { current: null as HTMLCanvasElement | null };
  let result: FrameResult | undefined; let identity: InferenceMeasurement | undefined;
  const engineTimes: number[] = []; const drawTimes: number[] = []; const totalTimes: number[] = []; const hudTimes: number[] = [];
  const overlays = (ctx: CanvasRenderingContext2D, w: number, h: number, objects: Extract<FrameResult, { status: 'ready' }>['objects']) => {
    ctx.clearRect(0, 0, w, h);
    for (const object of objects) {
      if (object.maskCanvas) ctx.drawImage(object.maskCanvas, 0, 0, w, h);
      const box = object.bbox; ctx.strokeRect(box.x * w, box.y * h, box.width * w, box.height * h);
    }
  };
  flushSync(() => root.render(<PipelineDetails snapshot={null} />));
  const start = performance.now();
  const stop = startFrameScheduler({
    video: source as unknown as HTMLVideoElement, bootstrap: false, current, lock: new JobLock(), instrumentation: pipeline,
    input() {}, error: cause => { throw cause; },
    work(input) {
      counts.processed++; const before = performance.now();
      const value = engine.detectVideoFrame(input, model, 0.3, pipeline);
      engineTimes.push(performance.now() - before); return value;
    },
    adopt(value, measurement) { result = value; identity = measurement; if (pipeline && identity) pipeline.accept(identity); },
  });
  try {
    for (let i = 0; i < iterations; i++) {
      const before = performance.now();
      scene++; sourceTime = scene / 30;
      // A single final pixel changes: equal polls exercise the complete identity scan.
      if (i === 0) { context.fillStyle = '#123456'; context.fillRect(0, 0, width, height); }
      context.fillStyle = scene % 2 ? '#abcdef' : '#fedcba'; context.fillRect(width - 1, height - 1, 1, 1);
      const arrival = () => { const fn = callback!; callback = undefined; fn(0, { mediaTime: sourceTime, presentedFrames: scene } as VideoFrameCallbackMetadata); };
      arrival();
      const job = jobs.shift(); if (!job) throw Error('Missing queued work'); job();
      if (result?.status !== 'ready') throw Error('Missing successful result');
      const drawBefore = performance.now(); const stamp = probe?.start();
      const drawn = drawCanvasFrame(canvas, source as unknown as HTMLVideoElement, result, buffer, overlays, current);
      probe?.end('draw', stamp); if (!drawn) throw Error('Missing Canvas copy'); counts.drawn++;
      if (pipeline && identity) pipeline.firstDraw(identity);
      drawTimes.push(performance.now() - drawBefore);
      // Duplicate poll must capture/compare, but must never infer again.
      arrival(); if (jobs.length) throw Error('Duplicate unexpectedly queued');

      // Match the existing publication cadence by synthetic source time: one
      // snapshot + real React/DOM details preparation per eight 30fps frames.
      if ((i + 1) % 8 === 0) {
        const hudBefore = performance.now();
        if (pipeline) {
          counts.snapshot++;
          const snapshot = pipeline.snapshot();
          flushSync(() => root.render(<PipelineDetails snapshot={snapshot} />));
        }
        hudTimes.push(performance.now() - hudBefore);
      }
      totalTimes.push(performance.now() - before);
    }
    if (counts.processed !== iterations || counts.sdk !== iterations || counts.drawn !== iterations || counts.captured !== iterations * 3) throw Error(`Unmatched work: ${JSON.stringify(counts)}`);
    if (counts.compared !== (path === 'rgba' ? iterations * 5 - 3 : 0)) throw Error('Unmatched full-array comparisons');
    if (!enabled && (counts.metricClock || counts.snapshot)) throw Error('Disabled instrumentation executed');
    // Window snapshots are not lifetime counters: expensive trials may exceed
    // five seconds. Actual matched-work counts above are independent of retention.
    return { ...options, counts, elapsedMs: performance.now() - start, total: stats(totalTimes), engine: stats(engineTimes), draw: stats(drawTimes), hud: stats(hudTimes), checksum: scene, samples: { engine: engineTimes, draw: drawTimes, total: totalTimes, hud: hudTimes } };
  } finally {
    stop(); engine.dispose(); document.createElement = originalCreate;
    window.setTimeout = originalTimeout; Object.defineProperty(window, 'VideoFrame', { configurable: true, value: NativeFrame });
    Uint8ClampedArray.prototype.every = originalEvery;
  }
}
window.runPipelineTrial = trial;
declare global { interface Window { runPipelineTrial: typeof trial } }
