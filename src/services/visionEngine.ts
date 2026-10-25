import { FilesetResolver, ObjectDetector, ImageSegmenter, type ImageSegmenterResult } from '@mediapipe/tasks-vision';
import type { DetectedObject, RunnableModelType, ModelState, FrameResult, InferenceInput } from '../types/vision';
import { PipelineProbe, type PipelineObserver, type AttemptOutcome } from './pipelineMetrics';
import { requireRunnableModel } from './modelRegistry';
import { getCategoryColor, getLabelInfo } from './dictionary';
import { createYoloSegTask, type YoloSegTask, type YoloSegTaskOptions } from './yoloSegService';
import { createSegFormerTask, type SegFormerTask, type SegFormerTaskOptions } from './segformerService';
import { unknownOnnxBackend } from './onnxSession';
import { resolveMediaPipeWasmUrls, resolveModelAssetUrls, recordResolvedSource } from './modelAssetResolver';

type Fileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;
export type DetectorTask = Pick<ObjectDetector, 'detectForVideo' | 'close'>;
export type SegmenterTask = Pick<ImageSegmenter, 'segmentForVideo' | 'close'>;
export interface VisionTaskFactory {
  load: () => Promise<Fileset>;
  detector: (fileset: Fileset, delegate: 'GPU') => Promise<DetectorTask>;
  segmenter: (fileset: Fileset, delegate: 'GPU') => Promise<SegmenterTask>;
  yoloSeg?: (options?: YoloSegTaskOptions) => Promise<YoloSegTask>;
  segformer?: (options?: SegFormerTaskOptions) => Promise<SegFormerTask>;
}
const DELEGATE = 'GPU' as const;
const defaultFactory: VisionTaskFactory = {
  load: async () => {
    const wasm = await resolveMediaPipeWasmUrls();
    try {
      const fileset = await FilesetResolver.forVisionTasks(wasm.primaryUrl);
      recordResolvedSource('wasm-mediapipe', wasm.source);
      return fileset;
    } catch (primaryError) {
      if (wasm.fallbackUrl) {
        const fileset = await FilesetResolver.forVisionTasks(wasm.fallbackUrl);
        recordResolvedSource('wasm-mediapipe', 'external-fallback');
        return fileset;
      }
      throw primaryError;
    }
  },
  detector: async (fileset, delegate) => {
    const asset = await resolveModelAssetUrls('mediapipe');
    try {
      const task = await ObjectDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: asset.primaryUrl, delegate },
        scoreThreshold: 0.25, runningMode: 'VIDEO',
      });
      recordResolvedSource('mediapipe', asset.source);
      return task;
    } catch (primaryError) {
      if (asset.fallbackUrl) {
        const task = await ObjectDetector.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: asset.fallbackUrl, delegate },
          scoreThreshold: 0.25, runningMode: 'VIDEO',
        });
        recordResolvedSource('mediapipe', 'external-fallback');
        return task;
      }
      throw primaryError;
    }
  },
  segmenter: async (fileset, delegate) => {
    const asset = await resolveModelAssetUrls('selfie-segmenter');
    try {
      const task = await ImageSegmenter.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: asset.primaryUrl, delegate },
        runningMode: 'VIDEO', outputConfidenceMasks: true, outputCategoryMask: false,
      });
      recordResolvedSource('selfie-segmenter', asset.source);
      return task;
    } catch (primaryError) {
      if (asset.fallbackUrl) {
        const task = await ImageSegmenter.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: asset.fallbackUrl, delegate },
          runningMode: 'VIDEO', outputConfidenceMasks: true, outputCategoryMask: false,
        });
        recordResolvedSource('selfie-segmenter', 'external-fallback');
        return task;
      }
      throw primaryError;
    }
  },
  yoloSeg: (options) => createYoloSegTask(options),
  segformer: (options) => createSegFormerTask(options),
};
type ActiveTask =
  | { model: 'mediapipe'; task: DetectorTask }
  | { model: 'selfie-segmenter'; task: SegmenterTask }
  | { model: 'yolo-seg'; task: YoloSegTask }
  | { model: 'segformer'; task: SegFormerTask };
type InitializationResult = Extract<ModelState, { status: 'ready' | 'error' }> | { status: 'cancelled'; model: RunnableModelType; requestId: number };

// One owner, one active task, no task cache. Pending tasks cannot be aborted by
// the SDK; obsolete completions are closed before they can become active.
export class VisionEngine {
  private activeTask: ActiveTask | null = null;
  private state: ModelState = { status: 'idle', model: null, requestId: 0 };
  private generation = 0;
  private pending: Promise<InitializationResult> | null = null;
  private listeners = new Set<() => void>();
  private offscreenMaskCanvas: HTMLCanvasElement | null = null;
  private maskImageData: ImageData | null = null;
  private lastStableObjects: DetectedObject[] = [];
  private emptyFrameBufferCount = 0;
  private outputGeneration = 0;
  private acceptedInspector: {task:SegFormerTask;objects:DetectedObject[];generation:number;requestId:number}|null=null;

  constructor(private readonly factory: VisionTaskFactory = defaultFactory) {}

  getState = (): ModelState => this.state;
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };
  private publish(state: ModelState): void {
    this.state = state;
    this.listeners.forEach(listener => listener());
  }
  private resetOutputs(): void {
    this.outputGeneration++;
    this.lastStableObjects = [];
    this.emptyFrameBufferCount = 0;
    this.offscreenMaskCanvas = null;
    this.maskImageData = null;
    this.acceptedInspector=null;
    if(this.activeTask?.model==='segformer')this.activeTask.task.clearResult?.();
  }
  // A new camera session must not inherit temporal detections from another scene.
  // Keep the ready model task and its initialization generation intact.
  clearDetections(): void { this.resetOutputs(); }

  private closeTask(active: ActiveTask | null): void {
    if (!active) return;
    try { const closing=active.task.close();if(closing)Promise.resolve(closing).catch(cause=>console.warn('Vision task cleanup failed',{model:active.model,cause})); }
    catch (cause) { console.warn('Vision task cleanup failed', { model: active.model, cause }); }
  }

  initialize(model: RunnableModelType): Promise<InitializationResult> {
    requireRunnableModel(model);
    if (this.state.model === model) {
      if (this.state.status === 'loading' && this.pending) return this.pending;
      if (this.state.status === 'ready') return Promise.resolve(this.state);
    }
    const requestId = ++this.generation;
    this.closeTask(this.activeTask);
    this.activeTask = null;
    this.resetOutputs();
    const cancelled = (): InitializationResult => ({ status: 'cancelled', model, requestId });
    const current = () => requestId === this.generation;

    // Establish ownership before synchronous subscribers can reenter initialize
    // or dispose. Return this request's Promise, never a replacement's pending.
    let resolve!: (result: InitializationResult) => void;
    let reject!: (cause: unknown) => void;
    const pending = new Promise<InitializationResult>((yes, no) => { resolve = yes; reject = no; });
    this.pending = pending;
    this.publish({ status: 'loading', model, requestId });
    void (async (): Promise<InitializationResult> => {
      let stage: 'wasm' | 'task' = 'wasm';
      try {
        if (!current()) return cancelled();
        let active: ActiveTask;
        if (model === 'yolo-seg') {
          stage = 'task';
          const yoloCreator = this.factory.yoloSeg ?? createYoloSegTask;
          const task = await yoloCreator();
          if (!task) throw new Error('Model task was not created');
          active = { model, task };
        } else if (model === 'segformer') {
          stage = 'task';
          const segformerCreator = this.factory.segformer ?? createSegFormerTask;
          const task = await segformerCreator();
          if (!task) throw new Error('Model task was not created');
          active = { model, task };
        } else {
          const fileset = await this.factory.load();
          if (!current()) return cancelled();
          stage = 'task';
          active = model === 'mediapipe'
            ? { model, task: await this.factory.detector(fileset, DELEGATE) }
            : { model, task: await this.factory.segmenter(fileset, DELEGATE) };
          if (!active.task) throw new Error('Model task was not created');
        }
        if (!current()) { this.closeTask(active); return cancelled(); }
        this.activeTask = active;
        const ready = { status: 'ready' as const, model, requestId, ...(active.model==='yolo-seg'||active.model==='segformer'?{backend:active.task.backend??unknownOnnxBackend()}:{delegate:DELEGATE}) };
        this.publish(ready);
        return ready;
      } catch (cause) {
        if (!current()) return cancelled();
        const error = { status: 'error' as const, model, requestId, stage, cause };
        console.warn('Vision initialization failed', { model, stage, cause });
        this.publish(error);
        return error;
      } finally {
        if (current()) this.pending = null;
      }
    })().then(resolve, reject);
    return pending;
  }

  // Only the owning App calls this on selection cancellation/unmount. An old
  // initialization's finally block never closes the current generation's task.
  dispose(): void {
    const requestId = ++this.generation;
    this.pending = null;
    this.closeTask(this.activeTask);
    this.activeTask = null;
    this.resetOutputs();
    this.publish({ status: 'idle', model: null, requestId });
  }

  public detectVideoFrame(
    videoElement: InferenceInput,
    model: 'selfie-segmenter' | 'mediapipe',
    confidenceThreshold: number,
    instrumentation?: PipelineObserver
  ): FrameResult;
  public detectVideoFrame(
    videoElement: InferenceInput,
    model: 'yolo-seg' | 'segformer',
    confidenceThreshold: number,
    instrumentation?: PipelineObserver
  ): Promise<FrameResult>;
  public detectVideoFrame(
    videoElement: InferenceInput,
    model: RunnableModelType,
    confidenceThreshold: number,
    instrumentation?: PipelineObserver
  ): FrameResult | Promise<FrameResult>;
  public detectVideoFrame(
    videoElement: InferenceInput,
    model: RunnableModelType,
    confidenceThreshold: number,
    instrumentation?: PipelineObserver
  ): FrameResult | Promise<FrameResult> {
    requireRunnableModel(model);
    const state = this.state;
    const active = this.activeTask;
    const outputGeneration = this.outputGeneration;
    const videoReady = 'readyState' in videoElement ? videoElement.readyState >= 2 : 'displayWidth' in videoElement
      ? videoElement.displayWidth > 0 && videoElement.displayHeight > 0 : videoElement.width > 0 && videoElement.height > 0;
    if (state.status !== 'ready' || state.model !== model || active?.model !== model || !videoReady) {
      return { status: 'not-ready', model, requestId: state.requestId };
    }
    if (active.model === 'yolo-seg') {
      return this.detectYoloVideoFrame(videoElement, confidenceThreshold, instrumentation, state, active, outputGeneration);
    }
    if (active.model === 'segformer') {
      return this.detectSegFormerVideoFrame(videoElement, confidenceThreshold, instrumentation, state, active, outputGeneration);
    }
    const probe = instrumentation ? new PipelineProbe(instrumentation) : undefined;
    const engineStart = probe?.start();
    let engineOutcome: AttemptOutcome = 'failed';
    let postStart: ReturnType<PipelineProbe['start']>;
    let postStarted = false;
    let postActive = false;
    let postMs: number | null = 0;
    try {
      const frameTimestamp = performance.now();
      let rawObjects: DetectedObject[] = [];

      const vW = ('videoWidth' in videoElement ? videoElement.videoWidth : 'displayWidth' in videoElement ? videoElement.displayWidth : videoElement.width) || 640;
      const vH = ('videoHeight' in videoElement ? videoElement.videoHeight : 'displayHeight' in videoElement ? videoElement.displayHeight : videoElement.height) || 480;

      // 1. SELFIE SEGMENTER: Real pixel-accurate segmentation mask + tight bounding box
      if (active.model === 'selfie-segmenter') {
        let segResult: ImageSegmenterResult | null = null;
        try {
          const sdkStart = probe?.start();
          let sdkOutcome: AttemptOutcome = 'failed';
          try { segResult = active.task.segmentForVideo(videoElement, frameTimestamp); sdkOutcome = 'success'; }
          finally { probe?.end('sdk', sdkStart, sdkOutcome); }
          postStart = probe?.start();
          postStarted = true; postActive = true;
          if (this.state !== state || this.activeTask !== active || this.outputGeneration !== outputGeneration) {
            engineOutcome = 'stale';
            return { status: 'not-ready', model, requestId: state.requestId };
          }
          const confMask = segResult.confidenceMasks?.[0];

          if (confMask) {
            const mW = confMask.width;
            const mH = confMask.height;
            const maskFloats = confMask.getAsFloat32Array();

            if (!this.offscreenMaskCanvas) {
              this.offscreenMaskCanvas = document.createElement('canvas');
            }
            if (this.offscreenMaskCanvas.width !== mW || this.offscreenMaskCanvas.height !== mH) {
              this.offscreenMaskCanvas.width = mW;
              this.offscreenMaskCanvas.height = mH;
              this.maskImageData = null;
            }

            const maskCtx = this.offscreenMaskCanvas.getContext('2d');
            if (maskCtx) {
              if (!this.maskImageData) {
                this.maskImageData = maskCtx.createImageData(mW, mH);
              }
              const data = this.maskImageData.data;
              const threshold = Math.max(0.25, confidenceThreshold);

              let minX = mW;
              let maxX = 0;
              let minY = mH;
              let maxY = 0;
              let count = 0;
              let sumScore = 0;

              for (let i = 0; i < maskFloats.length; i++) {
                const conf = maskFloats[i];
                const pIdx = i * 4;
                if (conf >= threshold) {
                  const px = i % mW;
                  const py = Math.floor(i / mW);
                  if (px < minX) minX = px;
                  if (px > maxX) maxX = px;
                  if (py < minY) minY = py;
                  if (py > maxY) maxY = py;
                  count++;
                  sumScore += conf;

                  // High-visibility cyan/teal neon glow with dynamic alpha
                  data[pIdx] = 20;       // R
                  data[pIdx + 1] = 184;   // G (vivid teal/emerald)
                  data[pIdx + 2] = 166;   // B
                  data[pIdx + 3] = Math.min(255, Math.floor(conf * 210)); // Alpha
                } else {
                  data[pIdx + 3] = 0;
                }
              }

              // If a person is present in the frame (> 0.5% of total pixels)
              if (count > (mW * mH * 0.005)) {
                maskCtx.putImageData(this.maskImageData, 0, 0);

                const normX = Math.max(0, minX / mW);
                const normY = Math.max(0, minY / mH);
                const normW = Math.min(1, Math.max(0.04, (maxX - minX) / mW));
                const normH = Math.min(1, Math.max(0.04, (maxY - minY) / mH));
                const avgConf = sumScore / count;

                rawObjects = [{
                  id: 'det-selfie-person',
                  label: 'person',
                  labelJa: '人物 (セルフィー)',
                  category: '人物',
                  confidence: Math.min(0.99, avgConf),
                  bbox: {
                    x: normX,
                    y: normY,
                    width: normW,
                    height: normH
                  },
                  color: '#10b981',
                  maskCanvas: this.offscreenMaskCanvas
                }];
              }
            }
          }
        } finally {
          if (postStarted && probe) postMs = probe.elapsed(postStart)?.ms ?? null;
          postStart = undefined; postActive = false;
          if (segResult) {
            const cleanupStart = probe?.start();
            let cleanupOutcome: AttemptOutcome = 'failed';
            try {
              segResult.close();
              cleanupOutcome = 'success';
            } catch (cause) {
              console.warn('Vision result cleanup failed', { cause });
            } finally { probe?.end('cleanup', cleanupStart, cleanupOutcome); }
          }
        }
      }

      // EfficientDet-Lite0 produces boxes only; no substitute segmentation.
      if (active.model === 'mediapipe') {
        const sdkStart = probe?.start();
        let sdkOutcome: AttemptOutcome = 'failed';
        let detectionResult;
        try { detectionResult = active.task.detectForVideo(videoElement, frameTimestamp); sdkOutcome = 'success'; }
        finally { probe?.end('sdk', sdkStart, sdkOutcome); }
        postStart = probe?.start();
        postStarted = true; postActive = true;

        if (detectionResult.detections && detectionResult.detections.length > 0) {
          rawObjects = detectionResult.detections.map((det, idx) => {
            const cat = det.categories[0];
            const label = cat ? cat.categoryName : 'object';
            const score = cat ? cat.score : 0.8;
            const bbox = det.boundingBox || { originX: 0, originY: 0, width: 0, height: 0 };
            const labelInfo = getLabelInfo(label);

            const normX = Math.max(0, bbox.originX / vW);
            const normY = Math.max(0, bbox.originY / vH);
            const normW = Math.min(1, bbox.width / vW);
            const normH = Math.min(1, bbox.height / vH);

            return {
              id: `det-${idx}-${label}`,
              label,
              labelJa: labelInfo.ja,
              category: labelInfo.category,
              confidence: score,
              bbox: {
                x: normX,
                y: normY,
                width: normW,
                height: normH
              },
              color: getCategoryColor(label)
            };
          });
        }
      }

      if (active.model === 'selfie-segmenter') { postStart = probe?.start(); postActive = true; }

      if (this.state !== state || this.activeTask !== active || this.outputGeneration !== outputGeneration) {
        engineOutcome = 'stale';
        return { status: 'not-ready', model, requestId: state.requestId };
      }

      // Filter by user confidence threshold (only for object detector; segmenter already filtered)
      const filteredObjects = model === 'selfie-segmenter'
        ? rawObjects
        : rawObjects.filter(obj => obj.confidence >= confidenceThreshold);

      // Existing within-model smoothing; cross-model state is reset on initialization.
      let finalObjects: DetectedObject[];

      if (filteredObjects.length > 0) {
        if (model === 'selfie-segmenter') {
          // Immediate 1:1 real-time tracking for selfie segmenter (0 latency)
          finalObjects = filteredObjects;
        } else {
          // Exponential Moving Average (EMA) smoothing for bounding boxes to avoid jitter
          finalObjects = filteredObjects.map(obj => {
            const prev = this.lastStableObjects.find(p => p.label === obj.label);
            if (prev) {
              return {
                ...obj,
                bbox: {
                  x: prev.bbox.x * 0.2 + obj.bbox.x * 0.8,
                  y: prev.bbox.y * 0.2 + obj.bbox.y * 0.8,
                  width: prev.bbox.width * 0.2 + obj.bbox.width * 0.8,
                  height: prev.bbox.height * 0.2 + obj.bbox.height * 0.8
                }
              };
            }
            return obj;
          });
        }
        this.lastStableObjects = finalObjects;
        this.emptyFrameBufferCount = 0;
      } else {
        this.emptyFrameBufferCount++;
        // Never freeze old positions on screen! Limit stale retention to max 2 frames
        if (this.emptyFrameBufferCount <= 2 && this.lastStableObjects.length > 0) {
          finalObjects = this.lastStableObjects.map(obj => ({
            ...obj,
            confidence: obj.confidence * 0.85
          }));
        } else {
          this.lastStableObjects = [];
          finalObjects = [];
        }
      }

      // Synchronous SDK doubles/subscribers can reenter lifecycle methods too.
      if (this.state !== state || this.activeTask !== active || this.outputGeneration !== outputGeneration) {
        engineOutcome = 'stale';
        return { status: 'not-ready', model, requestId: state.requestId };
      }
      engineOutcome = 'success';
      return { status: 'ready', model, requestId: state.requestId, objects: finalObjects };
    } catch (cause) {
      if (this.state === state && this.activeTask === active && this.outputGeneration === outputGeneration) this.resetOutputs();
      throw cause;
    } finally {
      if (postStarted && probe) {
        // Segmenter transformations precede result.close; filtering/smoothing follow it.
        // Publish one summed exclusive postprocess sample per attempted transform.
        const tail = postActive ? probe.elapsed(postStart)?.ms ?? null : 0;
        probe.duration('postprocess', postMs === null || tail === null ? null : postMs + tail, engineOutcome);
      }
      probe?.end('engine', engineStart, engineOutcome);
    }
  }

  private async detectYoloVideoFrame(
    videoElement: InferenceInput,
    confidenceThreshold: number,
    instrumentation: PipelineObserver | undefined,
    state: Extract<ModelState, { status: 'ready' }>,
    active: { model: 'yolo-seg'; task: YoloSegTask },
    outputGeneration: number
  ): Promise<FrameResult> {
    const probe = instrumentation ? new PipelineProbe(instrumentation) : undefined;
    const engineStart = probe?.start();
    let engineOutcome: AttemptOutcome = 'failed';
    let postStart: ReturnType<PipelineProbe['start']>;
    let postStarted = false;
    let postActive = false;
    const postMs: number | null = 0;
    try {
      const sdkStart = probe?.start();
      let sdkOutcome: AttemptOutcome = 'failed';
      let rawObjects: DetectedObject[] = [];
      try {
        const detectResult = active.task.detectForVideo(videoElement, confidenceThreshold);
        rawObjects = detectResult instanceof Promise ? await detectResult : detectResult;
        sdkOutcome = 'success';
      } finally {
        probe?.end('sdk', sdkStart, sdkOutcome);
      }
      postStart = probe?.start();
      postStarted = true;
      postActive = true;
      if (this.state !== state || this.activeTask !== active || this.outputGeneration !== outputGeneration) {
        engineOutcome = 'stale';
        return { status: 'not-ready', model: 'yolo-seg', requestId: state.requestId };
      }

      const filteredObjects = rawObjects.filter(obj => obj.confidence >= confidenceThreshold);
      let finalObjects: DetectedObject[];

      if (filteredObjects.length > 0) {
        finalObjects = filteredObjects.map(obj => {
          const prev = this.lastStableObjects.find(p => p.label === obj.label);
          if (prev && !obj.maskCanvas) {
            return {
              ...obj,
              bbox: {
                x: prev.bbox.x * 0.2 + obj.bbox.x * 0.8,
                y: prev.bbox.y * 0.2 + obj.bbox.y * 0.8,
                width: prev.bbox.width * 0.2 + obj.bbox.width * 0.8,
                height: prev.bbox.height * 0.2 + obj.bbox.height * 0.8
              }
            };
          }
          return obj;
        });
        this.lastStableObjects = finalObjects;
        this.emptyFrameBufferCount = 0;
      } else {
        this.emptyFrameBufferCount++;
        if (this.emptyFrameBufferCount <= 2 && this.lastStableObjects.length > 0) {
          finalObjects = this.lastStableObjects.map(obj => ({
            ...obj,
            confidence: obj.confidence * 0.85
          }));
        } else {
          this.lastStableObjects = [];
          finalObjects = [];
        }
      }

      if (this.state !== state || this.activeTask !== active || this.outputGeneration !== outputGeneration) {
        engineOutcome = 'stale';
        return { status: 'not-ready', model: 'yolo-seg', requestId: state.requestId };
      }
      engineOutcome = 'success';
      return { status: 'ready', model: 'yolo-seg', requestId: state.requestId, objects: finalObjects };
    } catch (cause) {
      if (this.state === state && this.activeTask === active && this.outputGeneration === outputGeneration) {this.resetOutputs();throw cause;}
      engineOutcome='stale';return {status:'not-ready',model:'yolo-seg',requestId:state.requestId};
    } finally {
      if (postStarted && probe) {
        const tail = postActive ? probe.elapsed(postStart)?.ms ?? null : 0;
        probe.duration('postprocess', postMs === null || tail === null ? null : postMs + tail, engineOutcome);
      }
      probe?.end('engine', engineStart, engineOutcome);
    }
  }

  private async detectSegFormerVideoFrame(
    videoElement: InferenceInput,
    confidenceThreshold: number,
    instrumentation: PipelineObserver | undefined,
    state: Extract<ModelState, { status: 'ready' }>,
    active: { model: 'segformer'; task: SegFormerTask },
    outputGeneration: number
  ): Promise<FrameResult> {
    const probe = instrumentation ? new PipelineProbe(instrumentation) : undefined;
    const engineStart = probe?.start();
    let engineOutcome: AttemptOutcome = 'failed';
    let postStart: ReturnType<PipelineProbe['start']>;
    let postStarted = false;
    let postActive = false;
    const postMs: number | null = 0;
    try {
      const sdkStart = probe?.start();
      let sdkOutcome: AttemptOutcome = 'failed';
      let rawObjects: DetectedObject[] = [];
      try {
        const detectResult = active.task.detectForVideo(videoElement, confidenceThreshold);
        rawObjects = detectResult instanceof Promise ? await detectResult : detectResult;
        sdkOutcome = 'success';
      } finally {
        probe?.end('sdk', sdkStart, sdkOutcome);
      }
      postStart = probe?.start();
      postStarted = true;
      postActive = true;
      if (this.state !== state || this.activeTask !== active || this.outputGeneration !== outputGeneration) {
        engineOutcome = 'stale';
        return { status: 'not-ready', model: 'segformer', requestId: state.requestId };
      }

      this.lastStableObjects = rawObjects;
      this.emptyFrameBufferCount = 0;
      const semantic=active.task.acceptResult?.(rawObjects);

      if (this.state !== state || this.activeTask !== active || this.outputGeneration !== outputGeneration) {
        engineOutcome = 'stale';
        return { status: 'not-ready', model: 'segformer', requestId: state.requestId };
      }
      this.acceptedInspector={task:active.task,objects:rawObjects,generation:outputGeneration,requestId:state.requestId};
      engineOutcome = 'success';
      return { status: 'ready', model: 'segformer', requestId: state.requestId, objects: rawObjects, ...(semantic?{semanticMask:semantic.maskCanvas}:{}) };
    } catch (cause) {
      if (this.state === state && this.activeTask === active && this.outputGeneration === outputGeneration) {this.resetOutputs();throw cause;}
      engineOutcome='stale';return {status:'not-ready',model:'segformer',requestId:state.requestId};
    } finally {
      if (postStarted && probe) {
        const tail = postActive ? probe.elapsed(postStart)?.ms ?? null : 0;
        probe.duration('postprocess', postMs === null || tail === null ? null : postMs + tail, engineOutcome);
      }
      probe?.end('engine', engineStart, engineOutcome);
    }
  }

  public findObjectAtPoint(
    normX: number,
    normY: number,
    objects: DetectedObject[]
  ): DetectedObject | null {
    if (this.activeTask?.model === 'segformer' && this.activeTask.task.getPixelClass) {
      const accepted=this.acceptedInspector;
      if(!accepted||accepted.task!==this.activeTask.task||accepted.objects!==objects||accepted.generation!==this.outputGeneration||accepted.requestId!==this.state.requestId)return null;
      const pixelInfo = this.activeTask.task.getPixelClass(normX, normY);
      if (pixelInfo) {
        const matched = objects.find(o => o.id === `segformer-${pixelInfo.id}`);
        if (matched) return pixelInfo.confidence===undefined?matched:{...matched,confidence:pixelInfo.confidence,confidenceKind:'semantic-softmax-pixel'};
        if(pixelInfo.confidence===undefined||!Number.isFinite(pixelInfo.confidence))return null;
        return {
          id: `segformer-${pixelInfo.id}`,
          label: pixelInfo.en,
          labelJa: pixelInfo.ja,
          category: pixelInfo.category,
          confidence: pixelInfo.confidence, confidenceKind:'semantic-softmax-pixel', areaFraction:pixelInfo.areaFraction,
          bbox: pixelInfo.bbox??{x:normX,y:normY,width:0,height:0},
          color: pixelInfo.color,
        };
      }
      return null;
    }

    const sorted = [...objects].sort((a, b) => {
      const areaA = a.bbox.width * a.bbox.height;
      const areaB = b.bbox.width * b.bbox.height;
      return areaA - areaB;
    });

    for (const obj of sorted) {
      // Pixel-precise hit test for real mask canvas
      if (obj.maskCanvas) {
        const mx = Math.floor(normX * obj.maskCanvas.width);
        const my = Math.floor(normY * obj.maskCanvas.height);
        if (mx >= 0 && mx < obj.maskCanvas.width && my >= 0 && my < obj.maskCanvas.height) {
          const mCtx = obj.maskCanvas.getContext('2d');
          if (mCtx) {
            const pixel = mCtx.getImageData(mx, my, 1, 1).data;
            if (pixel[3] > 30) {
              return obj;
            }
          }
        }
      }

      // Polygon test
      if (obj.maskPolygon && obj.maskPolygon.length > 2) {
        if (this.isPointInPolygon(normX, normY, obj.maskPolygon)) {
          return obj;
        }
      }

      // Bounding box test
      const { x, y, width, height } = obj.bbox;
      if (normX >= x && normX <= x + width && normY >= y && normY <= y + height) {
        return obj;
      }
    }

    return null;
  }

  private isPointInPolygon(
    x: number,
    y: number,
    polygon: Array<{ x: number; y: number }>
  ): boolean {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i].x;
      const yi = polygon[i].y;
      const xj = polygon[j].x;
      const yj = polygon[j].y;

      const intersect =
        yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  }
}
