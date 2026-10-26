import { FilesetResolver, ImageClassifier, type ImageClassifierResult } from '@mediapipe/tasks-vision';
import type { InferenceInput } from '../types/vision';
import type { ClassificationDelegate, ClassificationFrameResult, ClassificationOptions, ClassificationState, ImageClassifierVariantId, WholeImageClassificationFrame } from '../types/classification';
import { classifierAsset, resolveClassifierAsset } from './imageClassifierAssets';
import { recordResolvedSources, resolveMediaPipeWasmUrls, type AssetDeliverySource, type AssetSourceRecorder } from './modelAssetResolver';
import { PipelineProbe, type AttemptOutcome, type PipelineObserver } from './pipelineMetrics';

export interface ImageClassificationTask {
  readonly sdkDelegate?: ClassificationDelegate;
  classifyForVideo: (input: InferenceInput, timestampMs: number) => ImageClassifierResult;
  close: () => void | Promise<void>;
}
export type ImageClassificationFactory = (variantId: ImageClassifierVariantId, delegate: ClassificationDelegate, recordSource: AssetSourceRecorder, current?: () => boolean) => Promise<ImageClassificationTask>;
export class ClassificationInitializationError extends Error {
  constructor(public readonly stage: 'asset' | 'wasm' | 'task', cause: unknown) {
    super(`Image classification ${stage}: ${cause instanceof Error ? cause.message : String(cause)}`, { cause });
    this.name = 'ClassificationInitializationError';
  }
}
export class ClassificationFrameError extends Error {
  constructor(public readonly stage: 'input' | 'run' | 'output', cause: unknown) {
    super(`Image classification ${stage}: ${cause instanceof Error ? cause.message : String(cause)}`, { cause });
    this.name = 'ClassificationFrameError';
  }
}
export const createImageClassificationTask: ImageClassificationFactory = async (variantId, delegate, recordSource, current = () => true) => {
  const entry = classifierAsset(variantId);
  let asset;
  try {
    asset = await resolveClassifierAsset(variantId);
    if (asset.modelId !== entry.modelId || asset.variantId !== variantId || asset.modelAssetPath !== entry.upstreamUrl || asset.source !== 'external') throw new Error('Classifier asset identity mismatch');
  } catch (cause) { throw new ClassificationInitializationError('asset', cause); }
  if (!current()) throw new Error('Classifier initialization cancelled before WASM');
  let fileset;
  const wasm = await resolveMediaPipeWasmUrls();
  if (!current()) throw new Error('Classifier initialization cancelled before WASM');
  try {
    try { fileset = await FilesetResolver.forVisionTasks(wasm.primaryUrl); recordSource('wasm-mediapipe', wasm.source); }
    catch (cause) {
      if (!wasm.fallbackUrl) throw cause;
      fileset = await FilesetResolver.forVisionTasks(wasm.fallbackUrl); recordSource('wasm-mediapipe', 'external-fallback');
    }
  } catch (cause) { throw new ClassificationInitializationError('wasm', cause); }
  if (!current()) throw new Error('Classifier initialization cancelled before task');
  try {
    const task = await ImageClassifier.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: asset.modelAssetPath, delegate }, runningMode: 'VIDEO', maxResults: -1,
    });
    recordSource('image-classifier', asset.source); recordSource(`image-classifier:${variantId}`, asset.source);
    return { sdkDelegate: delegate, classifyForVideo: (input, timestamp) => task.classifyForVideo(input, timestamp), close: () => task.close() };
  } catch (cause) { throw new ClassificationInitializationError('task', cause); }
};
interface Owner { task: ImageClassificationTask; users: number; closed: boolean; released: boolean; lastTimestamp: number }
function readyInput(input: InferenceInput): boolean {
  if ('readyState' in input) return input.readyState >= 2 && input.videoWidth > 0 && input.videoHeight > 0;
  return 'displayWidth' in input ? input.displayWidth > 0 && input.displayHeight > 0 : input.width > 0 && input.height > 0;
}
function normalizedResult(result: ImageClassifierResult, threshold: number, max: number) {
  if (!result || !Array.isArray(result.classifications)) throw new Error('Missing SDK classifications');
  const sdkTimestampMs = result.timestampMs;
  if (sdkTimestampMs !== undefined && !Number.isFinite(sdkTimestampMs)) throw new Error('Invalid SDK timestamp');
  const classifications = result.classifications.map(head => {
    if (!head || !Number.isInteger(head.headIndex) || typeof head.headName !== 'string' || !Array.isArray(head.categories)) throw new Error('Invalid classification head');
    const categories = head.categories.map(category => {
      if (!category || !Number.isInteger(category.index) || typeof category.categoryName !== 'string' || typeof category.displayName !== 'string' || !Number.isFinite(category.score)) throw new Error('Invalid SDK category');
      return { index: category.index, categoryName: category.categoryName, displayName: category.displayName, score: category.score };
    }).filter(category => category.score >= threshold).sort((a, b) => b.score - a.score).slice(0, max);
    return { headIndex: head.headIndex, headName: head.headName, categories };
  });
  return { classifications, sdkTimestampMs };
}
export class ImageClassificationEngine {
  private state: ClassificationState = { status: 'idle', model: null, requestId: 0 };
  private generation = 0;
  private outputGeneration = 0;
  private frameSequence = 0;
  private latest: WholeImageClassificationFrame | null = null;
  private active: Owner | null = null;
  private pending: Promise<ClassificationInitializationResult> | null = null;
  private listeners = new Set<() => void>();
  constructor(private readonly factory: ImageClassificationFactory = createImageClassificationTask) {}
  getState = (): ClassificationState => this.state;
  subscribe = (listener: () => void): (() => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; };
  private publish(state: ClassificationState, sources?: ReadonlyMap<string, AssetDeliverySource>) {
    this.state = state;
    if (sources) recordResolvedSources(sources);
    this.listeners.forEach(listener => listener());
  }
  clearResults(): void { this.outputGeneration++; this.latest = null; }
  private release(owner: Owner): void {
    owner.closed = true;
    if (owner.users || owner.released) return;
    owner.released = true;
    try { const result = owner.task.close(); if (result) Promise.resolve(result).catch(cause => console.warn('Classifier cleanup failed', cause)); }
    catch (cause) { console.warn('Classifier cleanup failed', cause); }
  }
  initialize(variant: ImageClassifierVariantId = 'lite0', delegate: ClassificationDelegate = 'GPU'): Promise<ClassificationInitializationResult> {
    const entry = classifierAsset(variant);
    if (delegate !== 'CPU' && delegate !== 'GPU') throw new RangeError('Unsupported classification delegate');
    if (this.state.model === 'image-classifier' && this.state.variantId === entry.variantId && this.state.requestedDelegate === delegate) {
      if (this.state.status === 'loading' && this.pending) return this.pending;
      if (this.state.status === 'ready') return Promise.resolve(this.state);
    }
    const requestId = ++this.generation;
    const previous = this.active; this.active = null; this.clearResults();
    const sources = new Map<string, AssetDeliverySource>();
    const record: AssetSourceRecorder = (key, source) => { sources.set(key, source); };
    const current = () => requestId === this.generation;
    const cancelled = (): ClassificationInitializationResult => ({ status: 'cancelled', model: 'image-classifier', requestId });
    let resolve!: (value: ClassificationInitializationResult) => void;
    let reject!: (cause: unknown) => void;
    const pending = new Promise<ClassificationInitializationResult>((yes, no) => { resolve = yes; reject = no; });
    this.pending = pending;
    this.publish({ status: 'loading', model: 'image-classifier', variantId: entry.variantId, requestedDelegate: delegate, requestId });
    // Reentrant cleanup sees this request's established state/promise. It may
    // supersede it, but cannot be overwritten by a later obsolete publication.
    if (previous) this.release(previous);
    void (async (): Promise<ClassificationInitializationResult> => {
      try {
        if (!current()) return cancelled();
        const task = await this.factory(entry.variantId, delegate, record, current);
        if (!task || typeof task.classifyForVideo !== 'function' || typeof task.close !== 'function') {
          if (task && typeof task.close === 'function') this.release({ task, users: 0, closed: false, released: false, lastTimestamp: -1 });
          throw new Error('Classifier task was not created');
        }
        if (task.sdkDelegate !== undefined && task.sdkDelegate !== delegate) { this.release({ task, users: 0, closed: false, released: false, lastTimestamp: -1 }); throw new Error('Classifier delegate configuration mismatch'); }
        const owner: Owner = { task, users: 0, closed: false, released: false, lastTimestamp: -1 };
        if (!current()) { this.release(owner); return cancelled(); }
        this.active = owner;
        const ready = { status: 'ready' as const, model: 'image-classifier' as const, variantId: entry.variantId, requestedDelegate: delegate, sdkDelegate: task.sdkDelegate ?? ('unreported' as const), actualBackend: 'unmeasured' as const, requestId };
        this.publish(ready, sources); return ready;
      } catch (cause) {
        if (!current()) return cancelled();
        const error = { status: 'error' as const, model: 'image-classifier' as const, variantId: entry.variantId, requestedDelegate: delegate, requestId, stage: cause instanceof ClassificationInitializationError ? cause.stage : 'task' as const, cause };
        this.publish(error); return error;
      } finally { if (current()) this.pending = null; }
    })().then(resolve, reject);
    return pending;
  }
  dispose(): void {
    const requestId = ++this.generation; this.pending = null;
    const previous = this.active; this.active = null; this.clearResults();
    this.publish({ status: 'idle', model: null, requestId });
    if (previous) this.release(previous);
  }
  isCurrentFrame(frame: WholeImageClassificationFrame): boolean {
    return this.latest === frame && this.state.status === 'ready' && this.state.requestId === frame.requestId && this.state.variantId === frame.variantId && this.outputGeneration === frame.outputGeneration;
  }
  classifyVideoFrame(input: InferenceInput, options: ClassificationOptions = {}, instrumentation?: PipelineObserver): ClassificationFrameResult {
    const state = this.state, owner = this.active, generation = this.outputGeneration;
    const notReady = (): ClassificationFrameResult => ({ status: 'not-ready', model: 'image-classifier', requestId: state.requestId });
    if (state.status !== 'ready' || !owner || owner.closed || !readyInput(input)) return notReady();
    const valid = () => this.active === owner && !owner.closed && this.state.requestId === state.requestId && this.outputGeneration === generation;
    const probe = instrumentation ? new PipelineProbe(instrumentation) : undefined, start = probe?.start();
    let outcome: AttemptOutcome = 'failed', stage: 'input' | 'run' | 'output' = 'input';
    owner.users++;
    try {
      const threshold = options.scoreThreshold ?? 0, max = options.maxResults ?? 5;
      if (!Number.isFinite(threshold) || !Number.isSafeInteger(max) || max < 1 || max > 10000) throw new Error('Invalid classification threshold/maxResults');
      const now = performance.now(); if (!Number.isFinite(now) || now < 0) throw new Error('Invalid inference clock');
      const timestamp = Math.max(now, owner.lastTimestamp + 1); owner.lastTimestamp = timestamp;
      stage = 'run'; const sdkStart = probe?.start(); let raw: ImageClassifierResult;
      if (!valid()) { outcome = 'cancelled'; return notReady(); }
      try { raw = owner.task.classifyForVideo(input, timestamp); probe?.end('sdk', sdkStart, 'success'); }
      catch (cause) { probe?.end('sdk', sdkStart, 'failed'); throw cause; }
      if (!valid()) { outcome = 'cancelled'; return notReady(); }
      stage = 'output'; const postStart = probe?.start(); let normalized;
      try { normalized = normalizedResult(raw, threshold, max); probe?.end('postprocess', postStart, 'success'); }
      catch (cause) { probe?.end('postprocess', postStart, 'failed'); throw cause; }
      if (!valid()) { outcome = 'cancelled'; return notReady(); }
      const frame: WholeImageClassificationFrame = { status: 'ready', kind: 'whole-image-classification', model: 'image-classifier', variantId: state.variantId, requestId: state.requestId, frameId: ++this.frameSequence, outputGeneration: generation, timestampMs: timestamp, ...(normalized.sdkTimestampMs === undefined ? {} : { sdkTimestampMs: normalized.sdkTimestampMs }), classifications: normalized.classifications };
      this.latest = frame; outcome = 'success'; return frame;
    } catch (cause) {
      if (!valid()) { outcome = 'cancelled'; return notReady(); }
      this.clearResults();
      throw new ClassificationFrameError(stage, cause);
    } finally { owner.users--; if (owner.closed) this.release(owner); probe?.end('engine', start, outcome); }
  }
}
export type ClassificationInitializationResult = Extract<ClassificationState, { status: 'ready' | 'error' }> | { status: 'cancelled'; model: 'image-classifier'; requestId: number };
