import { ObjectDetector, ImageSegmenter, type Detection, type MPMask } from '@mediapipe/tasks-vision';
import { VisionEngine } from '../src/services/visionEngine';
import { PipelineMetrics, type PipelineSnapshot } from '../src/services/pipelineMetrics';
import { inputStamp } from './input-provenance';
import { checkNegativeScene, type SceneBoundary, type Stamp } from './acceptance';
import type { DetectionFrame, FrameResult } from '../src/types/vision';

type RawDetection = { categories: Array<{ label: string; score: number }>; bbox: Detection['boundingBox'] };
type ProcessedObject = { label: string; confidence: number; bbox: { x: number; y: number; width: number; height: number }; mask: { width: number; height: number } | null };
type SdkResult = { detections?: Detection[]; confidenceMasks?: MPMask[]; close: () => void };
type Raw = { sdkSucceeded: boolean; stamp: Stamp | null; requestId: number | null; scene: string; id: number; model: string; input: string; ms: number; detections?: RawDetection[]; masks?: ReturnType<typeof summarizeMask>[]; error?: string };
export const probe = {
  marker: 'PR07_REAL_SDK_OBSERVATION',
  creates: [] as Array<{ model: string; delegate: string; at: number; ms?: number; error?: string }>,
  observationVersion: 2, closes: 0, resultCloses: 0, calls: 0, sdkSuccesses: 0, rawDropped: 0,
  raw: [] as Raw[],
  processed: [] as Array<{ id: number; rawId: number; status: string; model: string; input: string; stamp: Stamp | null; requestId: number; objects: ProcessedObject[] }>,
  draws: [] as Array<{ id: number; alphaPixels: number; width: number; height: number; maskDraws: number }>,
  adoptions: [] as Array<{ id: number }>,
  negativeCheck: (boundary: SceneBoundary, expected: string) => checkNegativeScene(probe, boundary, expected),
  snapshots: [] as PipelineSnapshot[],
  invalidateAfterNextSdk: false,
  stop: () => {},
};
declare global { interface Window { __realSdkProbe: typeof probe } }
window.__realSdkProbe = probe;
let activeEngine: VisionEngine | undefined;
let resultId = 0;
const identities = new WeakMap<object, number>();
const drawn = new WeakSet<object>();
function summarizeMask(mask: { width: number; height: number; getAsFloat32Array(): Float32Array }) {
  const data = mask.getAsFloat32Array();
  let finite = true; let min = Infinity; let max = -Infinity; let foreground = 0;
  for (const value of data) { finite &&= Number.isFinite(value); min = Math.min(min, value); max = Math.max(max, value); if (value >= 0.35) foreground++; }
  const region = (x0: number, y0: number, x1: number, y1: number) => {
    let sum = 0; let n = 0;
    for (let y = Math.floor(y0 * mask.height); y < y1 * mask.height; y++)
      for (let x = Math.floor(x0 * mask.width); x < x1 * mask.width; x++) { sum += data[y * mask.width + x]; n++; }
    return sum / n;
  };
  return { width: mask.width, height: mask.height, length: data.length, finite, min, max, foregroundFraction: foreground / data.length,
    personRegion: region(0.4, 0.35, 0.65, 0.75), backgroundRegion: region(0, 0, 0.12, 0.12) };
}
// Every wrapper invokes the original with identical receiver/arguments and returns
// its exact result. No factory replacement, delegate rewrite or result fabrication.
for (const [model, Task, method] of [
  ['mediapipe', ObjectDetector, 'detectForVideo'],
  ['selfie-segmenter', ImageSegmenter, 'segmentForVideo'],
] as const) {
  Task.createFromOptions = new Proxy(Task.createFromOptions, {
    async apply(target, receiver, args) {
      const row = { model, delegate: String(args[1]?.baseOptions?.delegate), at: performance.now(), ms: undefined as number | undefined, error: undefined as string | undefined };
      probe.creates.push(row);
      try { return await Reflect.apply(target, receiver, args); }
      catch (error) { row.error = String(error); throw error; }
      finally { row.ms = performance.now() - row.at; }
    },
  });
  const prototype = Task.prototype;
  const original = Reflect.get(prototype, method) as (...args: unknown[]) => SdkResult;
  Object.defineProperty(prototype, method, { configurable: true, value: new Proxy(original, {
    apply(target, receiver, args) {
      const row: Raw = { sdkSucceeded: false, stamp: inputStamp(args[0]), requestId: activeEngine?.getState().requestId ?? null, scene: window.__realCamera?.scene ?? 'unknown', id: ++probe.calls, model, input: 'timestamp' in args[0] ? 'VideoFrame' : 'getContext' in args[0] ? 'Canvas' : 'HTMLVideoElement', ms: 0 };
      const at = performance.now();
      try {
        const result = Reflect.apply(target, receiver, args);
        row.sdkSucceeded = true; probe.sdkSuccesses++;
        row.ms = performance.now() - at; // Original SDK call only, excludes observations below.
        if (model === 'mediapipe') row.detections = result.detections!.map(det => ({ categories: det.categories.map(cat => ({ label: cat.categoryName, score: cat.score })), bbox: det.boundingBox }));
        else {
          row.masks = result.confidenceMasks?.map(summarizeMask);
          result.close = new Proxy(result.close, { apply(close, owner, closeArgs) { probe.resultCloses++; return Reflect.apply(close, owner, closeArgs); } });
        }
        if (probe.invalidateAfterNextSdk) { probe.invalidateAfterNextSdk = false; activeEngine?.clearDetections(); }
        return result;
      } catch (error) { if (!row.sdkSucceeded) row.ms = performance.now() - at; row.error = String(error); throw error; }
      finally { if (probe.raw.length === 80) { probe.raw.shift(); probe.rawDropped++; } probe.raw.push(row); }
    },
  }) });
  prototype.close = new Proxy(prototype.close, { apply(target, receiver, args) { probe.closes++; return Reflect.apply(target, receiver, args); } });
}
VisionEngine.prototype.initialize = new Proxy(VisionEngine.prototype.initialize, {
  apply(target, receiver, args) { activeEngine = receiver; return Reflect.apply(target, receiver, args); },
});
VisionEngine.prototype.detectVideoFrame = new Proxy(VisionEngine.prototype.detectVideoFrame, {
  apply(target, receiver, args) {
    activeEngine = receiver;
    const result: FrameResult = Reflect.apply(target, receiver, args);
    const id = ++resultId; identities.set(result, id);
    const raw = probe.raw.at(-1);
    if (result.status === 'ready') for (const object of result.objects) if (object.maskCanvas) masks.add(object.maskCanvas);
    probe.processed.push({ id, rawId: probe.calls, status: result.status, model: result.model, input: raw?.input ?? 'unknown', stamp: raw?.stamp ?? null, requestId: result.requestId,
      objects: result.status === 'ready' ? result.objects.map(obj => ({ label: obj.label, confidence: obj.confidence, bbox: { ...obj.bbox }, mask: obj.maskCanvas ? { width: obj.maskCanvas.width, height: obj.maskCanvas.height } : null })) : [] });
    if (probe.processed.length > 80) probe.processed.shift();
    return result;
  },
});
PipelineMetrics.prototype.snapshot = new Proxy(PipelineMetrics.prototype.snapshot, {
  apply(target, receiver, args) {
    const result = Reflect.apply(target, receiver, args);
    if (result) { probe.snapshots.push(structuredClone(result)); if (probe.snapshots.length > 24) probe.snapshots.shift(); }
    return result;
  },
});
const masks = new WeakSet<HTMLCanvasElement>();
let maskDraws = 0;
CanvasRenderingContext2D.prototype.drawImage = new Proxy(CanvasRenderingContext2D.prototype.drawImage, {
  apply(target, receiver, args) {
    // Count draw calls using the exact postprocessed maskCanvas identity.
    if (masks.has(args[0])) maskDraws++;
    return Reflect.apply(target, receiver, args);
  },
});
export function observeAdoption(frame: DetectionFrame) {
  probe.adoptions.push({ id: identities.get(frame) ?? -1 });
  if (probe.adoptions.length > 80) probe.adoptions.shift();
}
export function observeDraw(frame: DetectionFrame, canvas: HTMLCanvasElement) {
  if (drawn.has(frame)) return;
  drawn.add(frame);
  const data = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data;
  let alphaPixels = 0;
  for (let i = 3; i < data.length; i += 4) if (data[i]) alphaPixels++;
  probe.draws.push({ id: identities.get(frame) ?? -1, alphaPixels, width: canvas.width, height: canvas.height, maskDraws });
  if (probe.draws.length > 80) probe.draws.shift();
}
// Only unload teardown uses this; camera stop/restart is exercised through UI.
probe.stop = () => activeEngine?.dispose();
