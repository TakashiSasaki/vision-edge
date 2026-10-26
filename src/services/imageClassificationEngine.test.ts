// Controlled SDK boundaries only: no real models, inference or acquisition.
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { FilesetResolver, ImageClassifier, type ImageClassifierResult } from '@mediapipe/tasks-vision';
import { ImageClassificationEngine, ClassificationFrameError, ClassificationInitializationError, type ImageClassificationTask, type ImageClassificationFactory } from './imageClassificationEngine';
import * as assets from './imageClassifierAssets';
import { ALL_MODEL_REGISTRY, MODEL_REGISTRY, IMAGE_CLASSIFIER_MODEL, isRunnableModel } from './modelRegistry';
import { getResolvedSource, recordResolvedSource, setContainerModelCacheEnabledForTest, subscribeAssetResolution } from './modelAssetResolver';
import { VisionEngine } from './visionEngine';
import type { InferenceInput, RunnableModelType } from '../types/vision';
import type { ImageClassifierVariantId } from '../types/classification';
import { PipelineMetrics } from './pipelineMetrics';
const category = (index = 0, score = .8, categoryName = 'raw-label', displayName = '') => ({ index, score, categoryName, displayName });
const result = (categories = [category()]): ImageClassifierResult => ({ classifications: [{ headIndex: 0, headName: '', categories }] });
const canvas = () => Object.assign(document.createElement('canvas'), { width: 640, height: 480 });
const task = () => ({ sdkDelegate: 'GPU' as const, classifyForVideo: vi.fn<ImageClassificationTask['classifyForVideo']>(() => result()), close: vi.fn<ImageClassificationTask['close']>() });
function deferred<T>() { let resolve!: (value: T) => void, reject!: (cause: Error) => void; const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; }
let engines: ImageClassificationEngine[];
function engine(factory?: ImageClassificationFactory) { const e = new ImageClassificationEngine(factory); engines.push(e); return e; }
beforeEach(() => { engines = []; setContainerModelCacheEnabledForTest(false); vi.spyOn(console, 'warn').mockImplementation(() => {}); vi.spyOn(globalThis, 'fetch').mockRejectedValue(Error('UNEXPECTED_NETWORK')); });
afterEach(() => { engines.forEach(e => e.dispose()); setContainerModelCacheEnabledForTest(null); vi.restoreAllMocks(); vi.unstubAllGlobals(); });
function assetBoundary(variant: ImageClassifierVariantId) { const entry = assets.classifierAsset(variant); return vi.spyOn(assets, 'resolveClassifierAsset').mockResolvedValue({ modelId: 'image-classifier', variantId: variant, modelAssetPath: entry.upstreamUrl, source: 'external' }); }
function sdkBoundary() { const t = task(); const load = vi.spyOn(FilesetResolver, 'forVisionTasks').mockResolvedValue({ wasmLoaderPath: 'controlled-loader', wasmBinaryPath: 'controlled-wasm' }); const create = vi.spyOn(ImageClassifier, 'createFromOptions').mockResolvedValue(t as unknown as ImageClassifier); return { t, load, create }; }
it('registers a fifth classification UI family while keeping it outside the detection engine', () => {
  expect(ALL_MODEL_REGISTRY).toHaveLength(5); expect(MODEL_REGISTRY).toHaveLength(4);
  expect(IMAGE_CLASSIFIER_MODEL).toMatchObject({ id: 'image-classifier', availableInUI: true, outputType: 'whole-image-classification', assetVerification: 'blocked' });
  expect(IMAGE_CLASSIFIER_MODEL.variants.map(v => v.id)).toEqual(['lite0', 'lite2']);
  expect(isRunnableModel('image-classifier')).toBe(false);
  expect(() => new VisionEngine().initialize('image-classifier' as RunnableModelType)).toThrow('Unsupported');
});
it.each(['lite0', 'lite2'] as const)('default %s assets block before WASM, SDK or any fetch', async variant => {
  const sdk = sdkBoundary(), e = engine(); const state = await e.initialize(variant);
  expect(state).toMatchObject({ status: 'error', stage: 'asset', variantId: variant });
  expect(state.status === 'error' && state.cause).toBeInstanceOf(ClassificationInitializationError);
  expect(sdk.load).not.toHaveBeenCalled(); expect(sdk.create).not.toHaveBeenCalled(); expect(fetch).not.toHaveBeenCalled();
  expect(assets.classifierAssetStatuses()[variant]).toMatchObject({ status: 'blocked', sha256: null, license: null, acquisitionEnabled: false, localUrl: null });
});
for (const variant of ['lite0', 'lite2'] as const) it.each(['GPU', 'CPU'] as const)('%s delegate is explicitly passed to pinned0.10.14 VIDEO API for '+variant, async delegate => {
  assetBoundary(variant); const sdk = sdkBoundary(), e = engine();
  expect(await e.initialize(variant, delegate)).toMatchObject({ status: 'ready', variantId: variant, requestedDelegate: delegate, sdkDelegate: delegate, actualBackend: 'unmeasured' });
  expect(sdk.create).toHaveBeenCalledWith(expect.anything(), { baseOptions: { modelAssetPath: assets.classifierAsset(variant).upstreamUrl, delegate }, runningMode: 'VIDEO', maxResults: -1 });
  const source = canvas(), frame = e.classifyVideoFrame(source);
  expect(frame).toMatchObject({ status: 'ready', kind: 'whole-image-classification', variantId: variant });
  expect(sdk.t.classifyForVideo).toHaveBeenCalledWith(source, expect.any(Number));
  expect(sdk.t.classifyForVideo.mock.calls[0]).toHaveLength(2); // no ROI/crop/resizing options
  expect(getResolvedSource('image-classifier:'+variant)).toBe('external');
});
it.each(['unknown', '', null, 0, ['lite0'], {}])('rejects unsupported/non-string variant %j before replacing current ownership', async variant => {
  const t = task(), e = engine(async () => t); await e.initialize('lite0'); const state = e.getState();
  expect(() => e.initialize(variant as ImageClassifierVariantId)).toThrow(); expect(e.getState()).toBe(state); expect(t.close).not.toHaveBeenCalled();
});
it('rejects wrong model/variant/URL resolution before SDK creation', async () => {
  const asset = assetBoundary('lite0'), sdk = sdkBoundary(); asset.mockResolvedValue({ modelId: 'image-classifier', variantId: 'lite2', modelAssetPath: assets.classifierAsset('lite2').upstreamUrl, source: 'external' });
  expect(await engine().initialize('lite0')).toMatchObject({ status: 'error', stage: 'asset' }); expect(sdk.create).not.toHaveBeenCalled();
});
it('GPU failure is an error; explicit CPU retry is separate with no implicit delegate fallback', async () => {
  assetBoundary('lite0'); const sdk = sdkBoundary(), e = engine(); sdk.create.mockRejectedValueOnce(Error('GPU_REJECTED'));
  expect(await e.initialize('lite0')).toMatchObject({ status: 'error', stage: 'task', requestedDelegate: 'GPU' }); expect(sdk.create).toHaveBeenCalledTimes(1);
  expect(await e.initialize('lite0', 'CPU')).toMatchObject({ status: 'ready', sdkDelegate: 'CPU' }); expect(sdk.create).toHaveBeenCalledTimes(2);
});
it('failed WASM initializes no task and explicit retry recovers', async () => {
  assetBoundary('lite0'); const sdk = sdkBoundary(), e = engine(); sdk.load.mockRejectedValueOnce(Error('WASM_REJECTED'));
  expect(await e.initialize()).toMatchObject({ status: 'error', stage: 'wasm' }); expect(sdk.create).not.toHaveBeenCalled();
  expect((await e.initialize()).status).toBe('ready');
});
it('same-model loading notification shares exact pending promise; disposal prevents startup', async () => {
  const t = task(), factory = vi.fn(async () => t), e = engine(factory); let nested: ReturnType<typeof e.initialize> | undefined;
  const stop = e.subscribe(() => { if (e.getState().status === 'loading' && !nested) nested = e.initialize(); });
  const p = e.initialize(); expect(nested).toBe(p); expect((await p).status).toBe('ready'); stop(); expect(factory).toHaveBeenCalledTimes(1);
  const f = engine(factory); const off = f.subscribe(() => { if (f.getState().status === 'loading') f.dispose(); });
  expect((await f.initialize()).status).toBe('cancelled'); off(); expect(factory).toHaveBeenCalledTimes(1);
});
it.each(['success', 'failure'] as const)('stale ABA %s cannot overwrite current state/provenance', async outcome => {
  const gate = deferred<ImageClassificationTask>(), old = task(), current = task(); let count = 0;
  const factory: ImageClassificationFactory = async (variant, _delegate, record) => { const n = ++count; if (n === 1) { const value = await gate.promise; record('image-classifier', 'container-cache'); record('image-classifier:'+variant, 'container-cache'); return value; } record('image-classifier', 'external'); record('image-classifier:'+variant, 'external'); return current; };
  const e = engine(factory), p = e.initialize('lite0'); await e.initialize('lite2'); await e.initialize('lite0'); const state = e.getState();
  if (outcome === 'success') gate.resolve(old); else gate.reject(Error('OBSOLETE_REJECT'));
  expect((await p).status).toBe('cancelled'); expect(e.getState()).toBe(state); expect(getResolvedSource('image-classifier')).toBe('external'); expect(getResolvedSource('image-classifier:lite0')).toBe('external');
  expect(old.close).toHaveBeenCalledTimes(outcome === 'success' ? 1 : 0);
});
it('repeated dispose/restart cancels every pending task and closes each exactly once', async () => {
  const a = deferred<ImageClassificationTask>(), b = deferred<ImageClassificationTask>(), t1 = task(), t2 = task(), t3 = task();
  const factory = vi.fn<ImageClassificationFactory>().mockReturnValueOnce(a.promise).mockReturnValueOnce(b.promise).mockResolvedValue(t3), e = engine(factory);
  const p1 = e.initialize(); e.dispose(); const p2 = e.initialize(); e.dispose(); await e.initialize();
  b.resolve(t2); a.resolve(t1); expect((await p1).status).toBe('cancelled'); expect((await p2).status).toBe('cancelled');
  expect(t1.close).toHaveBeenCalledTimes(1); expect(t2.close).toHaveBeenCalledTimes(1); expect(t3.close).not.toHaveBeenCalled(); e.dispose(); e.dispose(); expect(t3.close).toHaveBeenCalledTimes(1);
});
it('failed initialization discards even successful buffered WASM observations', async () => {
  recordResolvedSource('wasm-mediapipe', 'external'); const e = engine(async (_v, _d, record) => { record('wasm-mediapipe', 'container-cache'); throw Error('TASK_FAILURE'); });
  expect((await e.initialize()).status).toBe('error'); expect(getResolvedSource('wasm-mediapipe')).toBe('external');
});
it('accepted source batch is visible only with ready ownership, including synchronous disposal', async () => {
  const e = engine(async (v, _d, record) => { record('image-classifier', 'external'); record('image-classifier:'+v, 'external'); return task(); }); let observed = 0;
  const stop = subscribeAssetResolution(() => { if (getResolvedSource('image-classifier') === 'external') { observed++; expect(e.getState().status).toBe('ready'); expect(getResolvedSource('image-classifier:lite0')).toBe('external'); e.dispose(); } });
  try { await e.initialize(); } finally { stop(); }
  expect(observed).toBe(1); expect(e.getState().status).toBe('idle');
});
it('unreported injected delegate is not claimed as observed; mismatched configuration is rejected and released', async () => {
  const t = { classifyForVideo: vi.fn(() => result()), close: vi.fn() }; expect(await engine(async () => t).initialize()).toMatchObject({ status: 'ready', requestedDelegate: 'GPU', sdkDelegate: 'unreported' });
  const wrong = { ...task(), sdkDelegate: 'CPU' as const }; expect(await engine(async () => wrong).initialize()).toMatchObject({ status: 'error', stage: 'task' }); expect(wrong.close).toHaveBeenCalledTimes(1);
});
it('preserves actual indices/names/scores/heads, sorts stable topK and thresholds without fabricating boxes', async () => {
  const categories = [category(1000, .5, 'English A'), category(0, .9, ''), category(7, .5, 'English B'), category(-1, .1, 'raw-negative-index')];
  const raw: ImageClassifierResult = { classifications: [{ headIndex: 3, headName: 'raw-head', categories }, { headIndex: 8, headName: '', categories: [] }], timestampMs: 123 };
  const t = task(); t.classifyForVideo.mockReturnValue(raw); const e = engine(async () => t); await e.initialize();
  const frame = e.classifyVideoFrame(canvas(), { scoreThreshold: .5, maxResults: 3 }); expect(frame.status).toBe('ready');
  if (frame.status !== 'ready') throw Error('not ready');
  expect(frame.classifications).toEqual([{ headIndex: 3, headName: 'raw-head', categories: [categories[1], categories[0], categories[2]] }, { headIndex: 8, headName: '', categories: [] }]);
  expect(frame.sdkTimestampMs).toBe(123); expect(raw.classifications[0].categories).toEqual(categories); expect(frame).not.toHaveProperty('objects'); expect(frame).not.toHaveProperty('bbox'); expect(frame).not.toHaveProperty('semanticMask');
  raw.classifications[0].categories[0].score = .01; expect(frame.classifications[0].categories[1].score).toBe(.5);
});
it('empty SDK classifications are a legitimate empty frame, not an inference-acceptance claim', async () => {
  const t = task(); t.classifyForVideo.mockReturnValue({ classifications: [] }); const e = engine(async () => t); await e.initialize(); expect(e.classifyVideoFrame(canvas())).toMatchObject({ status: 'ready', classifications: [] });
});
it.each([NaN, Infinity, -Infinity])('rejects nonfinite threshold %s before SDK', async value => {
  const t = task(), e = engine(async () => t); await e.initialize(); expect(() => e.classifyVideoFrame(canvas(), { scoreThreshold: value })).toThrow(ClassificationFrameError); expect(t.classifyForVideo).not.toHaveBeenCalled();
});
it.each([0, -1, 1.5, 10001, NaN])('rejects invalid maxResults %s before SDK', async value => {
  const t = task(), e = engine(async () => t); await e.initialize(); expect(() => e.classifyVideoFrame(canvas(), { maxResults: value })).toThrow(ClassificationFrameError); expect(t.classifyForVideo).not.toHaveBeenCalled();
});
for (const malformed of [null, {}, { classifications: null }, { classifications: [{ headIndex: 0, headName: '', categories: null }] }, result([category(0, NaN)]), result([{ ...category(), index: 1.5 }]), result([{ ...category(), categoryName: null } as unknown as ReturnType<typeof category>]), { ...result(), timestampMs: NaN }]) it('rejects malformed SDK output '+JSON.stringify(malformed), async () => {
  const t = task(), e = engine(async () => t); await e.initialize(); const previous = e.classifyVideoFrame(canvas()); t.classifyForVideo.mockReturnValue(malformed as unknown as ImageClassifierResult);
  expect(() => e.classifyVideoFrame(canvas())).toThrow(ClassificationFrameError); if (previous.status === 'ready') expect(e.isCurrentFrame(previous)).toBe(false); expect(t.close).not.toHaveBeenCalled();
});
it('run failure propagates, invalidates previous output, and never records engine success for that attempt', async () => {
  const t = task(), e = engine(async () => t); await e.initialize(); const old = e.classifyVideoFrame(canvas()); const metrics = new PipelineMetrics();
  t.classifyForVideo.mockImplementationOnce(() => { throw Error('SDK_RUN_FAILURE'); }); expect(() => e.classifyVideoFrame(canvas(), {}, metrics)).toThrow('SDK_RUN_FAILURE');
  if (old.status === 'ready') expect(e.isCurrentFrame(old)).toBe(false); expect(e.getState().status).toBe('ready'); expect(t.close).not.toHaveBeenCalled();
  const stats = metrics.snapshot(); if (!stats) throw Error('Missing metric snapshot'); expect(stats.stages.sdk.failed.count).toBe(1); expect(stats.stages.engine.failed.count).toBe(1); expect(stats.stages.engine.success.count).toBe(0);
});
it.each(['clear', 'dispose', 'switch', 'throw-after-dispose'] as const)('sync in-flight %s cannot publish stale results or release its task before run returns', async mode => {
  const t = task(), e = engine(async () => t); await e.initialize(); let next: ReturnType<typeof e.initialize> | undefined;
  t.classifyForVideo.mockImplementationOnce(() => { if (mode === 'clear') e.clearResults(); else if (mode === 'switch') next = e.initialize('lite2'); else e.dispose(); expect(t.close).not.toHaveBeenCalled(); if (mode === 'throw-after-dispose') throw Error('STALE_ERROR'); return result(); });
  expect(e.classifyVideoFrame(canvas())).toMatchObject({ status: 'not-ready' }); await next; expect(t.close).toHaveBeenCalledTimes(mode === 'clear' ? 0 : 1);
});
it('clear, next frame and model switch invalidate exact old frame ownership', async () => {
  const e = engine(async () => task()); await e.initialize(); const first = e.classifyVideoFrame(canvas()), second = e.classifyVideoFrame(canvas());
  if (first.status !== 'ready' || second.status !== 'ready') throw Error('not ready'); expect(e.isCurrentFrame(first)).toBe(false); expect(e.isCurrentFrame(second)).toBe(true);
  e.clearResults(); expect(e.isCurrentFrame(second)).toBe(false); const third = e.classifyVideoFrame(canvas()); await e.initialize('lite2'); if (third.status === 'ready') expect(e.isCurrentFrame(third)).toBe(false);
});
it.each(['canvas', 'frame', 'video'] as const)('passes the actual full-frame %s unchanged and uses strictly increasing timestamps', async kind => {
  const t = task(), e = engine(async () => t); await e.initialize(); const source: InferenceInput = kind === 'canvas' ? canvas() : kind === 'frame' ? { displayWidth: 640, displayHeight: 480 } as VideoFrame : Object.defineProperties(document.createElement('video'), { readyState: { value: 2 }, videoWidth: { value: 640 }, videoHeight: { value: 480 } });
  vi.spyOn(performance, 'now').mockReturnValue(1); e.classifyVideoFrame(source); e.classifyVideoFrame(source); expect(t.classifyForVideo.mock.calls.map(a => a[0])).toEqual([source, source]); expect(t.classifyForVideo.mock.calls[1][1]).toBeGreaterThan(t.classifyForVideo.mock.calls[0][1]);
});
it('unready input and idle state execute zero SDK calls', async () => {
  const t = task(), e = engine(async () => t); expect(e.classifyVideoFrame(canvas()).status).toBe('not-ready'); await e.initialize(); const c = canvas(); c.width = 0; expect(e.classifyVideoFrame(c).status).toBe('not-ready'); expect(t.classifyForVideo).not.toHaveBeenCalled();
});
it('handles sync and async close failures without unhandled rejection, duplicate release or replacement corruption', async () => {
  for (const asyncFailure of [false, true]) { const t = task(); t.close.mockImplementation(() => { if (asyncFailure) return Promise.reject(Error('CLOSE_REJECT')); throw Error('CLOSE_THROW'); }); const e = engine(async () => t); await e.initialize(); e.dispose(); e.dispose(); await Promise.resolve(); await e.initialize('lite2'); expect(e.getState().status).toBe('ready'); expect(t.close).toHaveBeenCalledTimes(1); }
  expect(console.warn).toHaveBeenCalledWith('Classifier cleanup failed', expect.any(Error));
});

it('default factory cancels an old WASM completion before SDK/model creation and cannot overwrite the accepted source', async () => {
  setContainerModelCacheEnabledForTest(true);
  vi.mocked(fetch).mockImplementation(async () => new Response(JSON.stringify({ enabled: true, models: {}, wasm: { mediapipe: { available: true, localUrl: '/api/model-cache/wasm/mediapipe', upstreamUrl: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm' } } }), { headers: { 'Content-Type': 'application/json' } }));
  vi.spyOn(assets, 'resolveClassifierAsset').mockImplementation(async variant => ({ modelId: 'image-classifier', variantId: variant, modelAssetPath: assets.classifierAsset(variant).upstreamUrl, source: 'external' }));
  const sdk = sdkBoundary(), gate = deferred<{ wasmLoaderPath: string; wasmBinaryPath: string }>(), entered = deferred<void>();
  sdk.load.mockImplementationOnce(() => { entered.resolve(); return gate.promise; }).mockRejectedValueOnce(Error('NEW_LOCAL_WASM_FAILURE'));
  const e = engine(), obsolete = e.initialize('lite0'); await entered.promise;
  expect((await e.initialize('lite2')).status).toBe('ready'); expect(getResolvedSource('wasm-mediapipe')).toBe('external-fallback');
  gate.resolve({ wasmLoaderPath: 'old-loader', wasmBinaryPath: 'old-wasm' }); expect((await obsolete).status).toBe('cancelled');
  expect(sdk.create).toHaveBeenCalledTimes(1); expect(getResolvedSource('wasm-mediapipe')).toBe('external-fallback');
});
it('captures validated options before SDK work can change the caller object', async () => {
  const options = { scoreThreshold: .5, maxResults: 1 }, t = task(), e = engine(async () => t); await e.initialize();
  t.classifyForVideo.mockImplementationOnce(() => { options.maxResults = 0; options.scoreThreshold = NaN; return result([category(7,.8), category(8,.7)]); });
  const frame = e.classifyVideoFrame(canvas(), options); if (frame.status !== 'ready') throw Error('not ready'); expect(frame.classifications[0].categories).toEqual([category(7,.8)]);
});
it('close reentry cannot overwrite a replacement initialization promise or loading state', async () => {
  const old = task(), replacement = task(), factory = vi.fn<ImageClassificationFactory>().mockResolvedValueOnce(old).mockResolvedValue(replacement), e = engine(factory); await e.initialize();
  let nested: ReturnType<typeof e.initialize> | undefined; old.close.mockImplementationOnce(() => { nested = e.initialize('lite2'); });
  const outer = e.initialize('lite0','CPU'); expect(e.getState()).toMatchObject({ status: 'loading', variantId: 'lite2' }); expect(e.initialize('lite2')).toBe(nested);
  expect((await outer).status).toBe('cancelled'); expect((await nested)?.status).toBe('ready'); expect(e.getState()).toMatchObject({ status:'ready', variantId:'lite2' }); expect(factory).toHaveBeenCalledTimes(2); expect(old.close).toHaveBeenCalledTimes(1);
});
it('dispose close reentry preserves a newly initialized request instead of publishing obsolete idle', async () => {
  const old = task(), e = engine(vi.fn<ImageClassificationFactory>().mockResolvedValueOnce(old).mockResolvedValue(task())); await e.initialize();
  let nested: ReturnType<typeof e.initialize> | undefined; old.close.mockImplementationOnce(() => { nested = e.initialize('lite2'); });
  e.dispose(); expect(e.getState()).toMatchObject({status:'loading',variantId:'lite2'}); expect(e.initialize('lite2')).toBe(nested); expect((await nested)?.status).toBe('ready');
});
