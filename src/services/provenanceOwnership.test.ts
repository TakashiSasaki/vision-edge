// Model-free S2-F1 races through the actual engine, default factories and resolver.
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { FilesetResolver, ObjectDetector, ImageSegmenter } from '@mediapipe/tasks-vision';
import { VisionEngine } from './visionEngine';
import { MODEL_VARIANT_CATALOG, WASM_ASSET_CATALOG, getResolvedSource, recordResolvedSource, setContainerModelCacheEnabledForTest, subscribeAssetResolution } from './modelAssetResolver';
import type { RunnableModelType } from '../types/vision';
const ortBoundary = vi.hoisted(() => ({ create: vi.fn() }));
vi.mock('onnxruntime-web', () => ({ env: { wasm: {}, logLevel: 'error' }, InferenceSession: { create: ortBoundary.create } }));
const variants = Object.values(MODEL_VARIANT_CATALOG);
const defaults = variants.filter(e => e.isDefault);
function deferred<T>() { let resolve!: (value: T) => void; let reject!: (reason: Error) => void; const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; }
function handle(model: RunnableModelType) { return { model, close: vi.fn(), release: vi.fn(async () => {}), detectForVideo: vi.fn(), segmentForVideo: vi.fn(), run: vi.fn(), inputNames: [], outputNames: [] }; }
type Handle = ReturnType<typeof handle>;
let plans: Map<string, (() => Promise<Handle>)[]>;
let engines: VisionEngine[];
function plan(url: string, action: () => Promise<Handle>) { plans.set(url, [...(plans.get(url) ?? []), action]); }
function hold(url: string) { const entered = deferred<void>(), done = deferred<Handle>(); plan(url, () => { entered.resolve(); return done.promise; }); return { entered: entered.promise, ...done }; }
function rejectNext(url: string) { plan(url, async () => { throw Error('CONTROLLED_SESSION_FAILURE'); }); }
function engine() { const e = new VisionEngine(); engines.push(e); return e; }
function snapshot() { return Object.fromEntries([...variants.map(e => `${e.modelId}:${e.variantId}`), ...defaults.map(e => e.modelId), 'wasm-mediapipe', 'wasm-ort'].map(key => [key, getResolvedSource(key)])); }
beforeEach(() => {
  plans = new Map(); engines = []; setContainerModelCacheEnabledForTest(true);
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  const row = (e: typeof variants[number]) => ({ id: e.modelId, variantId: e.variantId, file: e.file, localUrl: e.localUrl, upstreamUrl: e.upstreamUrl, license: e.license, status: 'cached', error: null, size: 2048, cachedAt: '2026-10-26T00:00:00Z' });
  const status = { enabled: true, models: Object.fromEntries(defaults.map(e => [e.modelId, row(e)])), variants: Object.fromEntries(variants.map(e => [`${e.modelId}:${e.variantId}`, row(e)])), wasm: Object.fromEntries(Object.entries(WASM_ASSET_CATALOG).map(([k, e]) => [k, { ...e, available: true }])) };
  vi.spyOn(globalThis, 'fetch').mockImplementation(async url => { expect(url).toBe('/api/model-cache/status'); return new Response(JSON.stringify(status), { headers: { 'Content-Type': 'application/json' } }); });
  const create = async (url: string) => { const action = plans.get(url)?.shift(); if (action) return action(); const entry = variants.find(e => e.localUrl === url || e.upstreamUrl === url); if (!entry) throw Error(`Unexpected model URL ${url}`); return handle(entry.modelId); };
  vi.spyOn(FilesetResolver, 'forVisionTasks').mockResolvedValue({ wasmLoaderPath: 'controlled-loader', wasmBinaryPath: 'controlled-wasm' });
  vi.spyOn(ObjectDetector, 'createFromOptions').mockImplementation((_fileset, options) => create(String(options.baseOptions?.modelAssetPath)) as unknown as Promise<ObjectDetector>);
  vi.spyOn(ImageSegmenter, 'createFromOptions').mockImplementation((_fileset, options) => create(String(options.baseOptions?.modelAssetPath)) as unknown as Promise<ImageSegmenter>);
  ortBoundary.create.mockReset().mockImplementation((url: string) => create(url));
});
afterEach(() => { engines.forEach(e => e.dispose()); setContainerModelCacheEnabledForTest(null); vi.restoreAllMocks(); });
for (const finish of ['success', 'failure'] as const) it.each(defaults)(`S2-F1 ABA stale ${finish} preserves accepted source: $modelId`, async entry => {
  const e = engine(), old = hold(entry.localUrl), obsolete = e.initialize(entry.modelId, entry.variantId);
  await old.entered;
  const other = entry.modelId === 'mediapipe' ? 'segformer' : 'mediapipe';
  expect((await e.initialize(other)).status).toBe('ready');
  rejectNext(entry.localUrl);
  expect(await e.initialize(entry.modelId, entry.variantId)).toMatchObject({ status: 'ready', variantId: entry.variantId });
  expect(getResolvedSource(`${entry.modelId}:${entry.variantId}`)).toBe('external-fallback');
  const accepted = snapshot(), state = e.getState(), obsoleteHandle = handle(entry.modelId);
  if (finish === 'success') old.resolve(obsoleteHandle);
  else { rejectNext(entry.upstreamUrl); old.reject(Error('OBSOLETE_PRIMARY_FAILURE')); }
  expect((await obsolete).status).toBe('cancelled');
  expect(e.getState()).toBe(state); expect(snapshot()).toEqual(accepted);
  expect(obsoleteHandle.close).toHaveBeenCalledTimes(finish === 'success' && !['segformer', 'yolo-seg'].includes(entry.modelId) ? 1 : 0);
  expect(obsoleteHandle.release).toHaveBeenCalledTimes(finish === 'success' && ['segformer', 'yolo-seg'].includes(entry.modelId) ? 1 : 0);
});
it.each(defaults)('S2-F1 cross-variant stale completion preserves current model/UI source: $modelId', async entry => {
  const other = variants.find(v => v.modelId === entry.modelId && v.variantId !== entry.variantId)!;
  const e = engine(), old = hold(entry.localUrl), obsolete = e.initialize(entry.modelId, entry.variantId);
  await old.entered; rejectNext(other.localUrl);
  expect(await e.initialize(other.modelId, other.variantId)).toMatchObject({ status: 'ready', variantId: other.variantId });
  expect(getResolvedSource(other.modelId)).toBe('external-fallback');
  const accepted = snapshot(); old.resolve(handle(entry.modelId));
  expect((await obsolete).status).toBe('cancelled'); expect(snapshot()).toEqual(accepted);
});
it.each(defaults)('S2-F1 repeated ABA/dispose closes every obsolete success without publication: $modelId', async entry => {
  const e = engine(), first = hold(entry.localUrl), p1 = e.initialize(entry.modelId, entry.variantId); await first.entered;
  e.dispose();
  const second = hold(entry.localUrl), p2 = e.initialize(entry.modelId, entry.variantId); await second.entered;
  e.dispose(); rejectNext(entry.localUrl);
  expect((await e.initialize(entry.modelId, entry.variantId)).status).toBe('ready');
  const accepted = snapshot(), a = handle(entry.modelId), b = handle(entry.modelId);
  second.resolve(b); expect((await p2).status).toBe('cancelled');
  first.resolve(a); expect((await p1).status).toBe('cancelled');
  expect(snapshot()).toEqual(accepted);
  for (const h of [a, b]) expect(['yolo-seg', 'segformer'].includes(entry.modelId) ? h.release : h.close).toHaveBeenCalledTimes(1);
});
it('S2-F1 obsolete MediaPipe WASM completion cannot overwrite accepted WASM provenance', async () => {
  const gate = deferred<{ wasmLoaderPath: string; wasmBinaryPath: string }>(), entered = deferred<void>();
  vi.mocked(FilesetResolver.forVisionTasks).mockImplementationOnce(() => { entered.resolve(); return gate.promise; });
  const e = engine(), obsolete = e.initialize('mediapipe'); await entered.promise;
  vi.mocked(FilesetResolver.forVisionTasks).mockRejectedValueOnce(Error('CURRENT_LOCAL_WASM_FAILURE'));
  expect((await e.initialize('selfie-segmenter')).status).toBe('ready');
  expect(getResolvedSource('wasm-mediapipe')).toBe('external-fallback'); const accepted = snapshot();
  gate.resolve({ wasmLoaderPath: 'old-loader', wasmBinaryPath: 'old-wasm' });
  expect((await obsolete).status).toBe('cancelled'); expect(snapshot()).toEqual(accepted);
  expect(ObjectDetector.createFromOptions).not.toHaveBeenCalled();
});
it.each(defaults)('S2-F1 no successful asset publication from a failed current task: $modelId', async entry => {
  // Seed a distinguishable previous observation, rather than treating the
  // getResolvedSource default as evidence that a source was actually published.
  for (const key of Object.keys(snapshot())) recordResolvedSource(key, 'external');
  const before = snapshot(); rejectNext(entry.localUrl); rejectNext(entry.upstreamUrl);
  expect(await engine().initialize(entry.modelId, entry.variantId)).toMatchObject({ status: 'error', stage: 'task' });
  expect(snapshot()).toEqual(before);
});
it('S2-F1 observers see accepted model/variant/WASM sources together with ready ownership', async () => {
  const entry = defaults.find(e => e.modelId === 'mediapipe')!, e = engine(); rejectNext(entry.localUrl);
  const observed: { state: string; model: string; variant: string }[] = [];
  const stop = subscribeAssetResolution(() => { if (getResolvedSource(entry.modelId) === 'external-fallback') observed.push({ state: e.getState().status, model: getResolvedSource(entry.modelId), variant: getResolvedSource(`${entry.modelId}:${entry.variantId}`) }); });
  try { expect((await e.initialize(entry.modelId)).status).toBe('ready'); } finally { stop(); }
  expect(observed).toEqual([{ state: 'ready', model: 'external-fallback', variant: 'external-fallback' }]);
});

it.each(defaults)('S2-F1 stale primary failure followed by successful fallback is still cancelled: $modelId', async entry => {
  const e = engine(), old = hold(entry.localUrl), obsolete = e.initialize(entry.modelId, entry.variantId);
  await old.entered; e.dispose();
  expect((await e.initialize(entry.modelId, entry.variantId)).status).toBe('ready');
  expect(getResolvedSource(entry.modelId)).toBe('container-cache');
  const accepted = snapshot(), lateFallback = handle(entry.modelId);
  plan(entry.upstreamUrl, async () => lateFallback); old.reject(Error('OBSOLETE_LOCAL_FAILURE'));
  expect((await obsolete).status).toBe('cancelled'); expect(snapshot()).toEqual(accepted);
  expect(['yolo-seg', 'segformer'].includes(entry.modelId) ? lateFallback.release : lateFallback.close).toHaveBeenCalledTimes(1);
});
it('S2-F1 obsolete MediaPipe WASM fallback success is not an accepted observation', async () => {
  const gate = deferred<{ wasmLoaderPath: string; wasmBinaryPath: string }>(), entered = deferred<void>();
  vi.mocked(FilesetResolver.forVisionTasks).mockImplementationOnce(() => { entered.resolve(); return gate.promise; });
  const e = engine(), obsolete = e.initialize('mediapipe'); await entered.promise;
  expect((await e.initialize('selfie-segmenter')).status).toBe('ready');
  const accepted = snapshot(); gate.reject(Error('OBSOLETE_WASM_PRIMARY_FAILURE'));
  expect((await obsolete).status).toBe('cancelled'); expect(snapshot()).toEqual(accepted);
  expect(ObjectDetector.createFromOptions).not.toHaveBeenCalled();
});
it('S2-F1 synchronous accepted-source observer can switch without a partial or late old publication', async () => {
  const e = engine(); let replacement: ReturnType<VisionEngine['initialize']> | undefined;
  const stop = subscribeAssetResolution(() => {
    const state = e.getState();
    if (state.status === 'ready' && state.variantId === 'lite0' && !replacement) {
      expect(getResolvedSource('mediapipe')).toBe('external-fallback');
      expect(getResolvedSource('mediapipe:lite0')).toBe('external-fallback');
      replacement = e.initialize('mediapipe', 'lite2');
    }
  });
  try {
    rejectNext(MODEL_VARIANT_CATALOG['mediapipe:lite0'].localUrl);
    await e.initialize('mediapipe', 'lite0'); await replacement;
    expect(replacement).toBeDefined();
    expect(e.getState()).toMatchObject({ status: 'ready', variantId: 'lite2' });
    expect(getResolvedSource('mediapipe')).toBe('container-cache');
    expect(getResolvedSource('mediapipe:lite2')).toBe('container-cache');
    expect(getResolvedSource('mediapipe:lite0')).toBe('external-fallback');
  } finally { stop(); }
});
