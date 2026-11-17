// Model-free routing controls: actual resolver/engine & controlled SDK.
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { FilesetResolver, ImageSegmenter, ObjectDetector } from '@mediapipe/tasks-vision';
import {
  MODEL_VARIANT_CATALOG,
  STEREO_VARIANT_CATALOG,
  getResolvedSource,
  getVariantCatalogEntry,
  resetResolvedSourcesForTest,
  resolveModelAssetUrls,
} from './modelAssetResolver';
import { STEREO_CORS_PROXY_PREFIX } from './stereoCorsProxyServer';
import { VisionEngine } from './visionEngine';

const boundary = vi.hoisted(() => ({ create: vi.fn() }));
vi.mock('onnxruntime-web', () => ({
  env: { wasm: {}, logLevel: 'error' },
  InferenceSession: { create: boundary.create },
}));

const entries = Object.entries(MODEL_VARIANT_CATALOG);

beforeEach(() => {
  resetResolvedSourcesForTest();
  boundary.create.mockReset();
});

afterEach(() => {
  resetResolvedSourcesForTest();
  vi.restoreAllMocks();
});

it.each(entries)('resolves %s directly to its exact upstream HTTPS CDN URL without container cache', async (_key, e) => {
  const fetchSpy = vi.spyOn(globalThis, 'fetch');
  const resolved = await resolveModelAssetUrls(e.modelId, e.variantId);
  expect(resolved).toEqual({
    primaryUrl: e.upstreamUrl,
    fallbackUrl: null,
    source: 'external',
  });
  expect(e.localUrl).toBe(e.upstreamUrl);
  expect(fetchSpy).not.toHaveBeenCalled();
});

it('routes non-CORS stereo variants via stateless CORS proxy and keeps CDN stereo variants direct', async () => {
  expect(await resolveModelAssetUrls('stereo-depth', 'lightstereo-s')).toEqual({
    primaryUrl: STEREO_VARIANT_CATALOG['stereo-depth:lightstereo-s'].upstreamUrl,
    fallbackUrl: null,
    source: 'external',
  });
  for (const id of [
    'mobilestereonet-2d',
    'hitnet-sf',
    'crestereo-init-iter2',
    'crestereo-combined-iter5',
  ] as const) {
    const resolved = await resolveModelAssetUrls('stereo-depth', id);
    expect(resolved.primaryUrl.startsWith(STEREO_CORS_PROXY_PREFIX)).toBe(true);
    expect(resolved.fallbackUrl).toBeNull();
    expect(resolved.source).toBe('cors-proxy');
  }
});

it('unknown explicit variant cannot be relabelled as a default asset', () => {
  expect(() => getVariantCatalogEntry('mediapipe', 'unknown')).toThrow();
  const engine = new VisionEngine();
  expect(() => engine.initialize('mediapipe', 'unknown')).toThrow();
  expect(engine.getState().status).toBe('idle');
  engine.dispose();
});

async function sdkBoundary() {
  vi.spyOn(FilesetResolver, 'forVisionTasks').mockResolvedValue({
    wasmLoaderPath: 'mock-loader',
    wasmBinaryPath: 'mock-wasm',
  });
  const detector = vi.spyOn(ObjectDetector, 'createFromOptions').mockResolvedValue({
    detectForVideo: vi.fn(),
    close: vi.fn(),
  } as unknown as ObjectDetector);
  const segmenter = vi.spyOn(ImageSegmenter, 'createFromOptions').mockResolvedValue({
    segmentForVideo: vi.fn(),
    close: vi.fn(),
  } as unknown as ImageSegmenter);
  boundary.create.mockResolvedValue({
    inputNames: [],
    outputNames: [],
    run: vi.fn(),
    release: vi.fn(async () => {}),
  });
  return { detector, segmenter };
}

it.each(entries)('actual initialization binds exact variant upstream URL and ready identity: %s', async (_key, e) => {
  const sdk = await sdkBoundary();
  const engine = new VisionEngine();
  const state = await engine.initialize(e.modelId, e.variantId);
  expect(state).toMatchObject({ status: 'ready', model: e.modelId, variantId: e.variantId });
  const urls =
    e.modelId === 'mediapipe'
      ? sdk.detector.mock.calls.map((c) => c[1].baseOptions?.modelAssetPath)
      : e.modelId === 'selfie-segmenter'
        ? sdk.segmenter.mock.calls.map((c) => c[1].baseOptions?.modelAssetPath)
        : boundary.create.mock.calls.map((c) => c[0]);
  expect(urls).toEqual([e.upstreamUrl]);
  expect(getResolvedSource(`${e.modelId}:${e.variantId}`)).toBe('external');
  engine.dispose();
  await Promise.resolve();
});

it.each(entries)('session creation failure cannot yield ready identity: %s', async (_key, e) => {
  const sdk = await sdkBoundary();
  const error = new Error('CONTROLLED_MODEL_FAILURE');
  if (e.modelId === 'mediapipe') sdk.detector.mockRejectedValue(error);
  else if (e.modelId === 'selfie-segmenter') sdk.segmenter.mockRejectedValue(error);
  else boundary.create.mockRejectedValue(error);

  const engine = new VisionEngine();
  expect(await engine.initialize(e.modelId, e.variantId)).toMatchObject({
    status: 'error',
    model: e.modelId,
    variantId: e.variantId,
  });
  expect(engine.getState().status).not.toBe('ready');
  engine.dispose();
});

it('invalid request leaves an already-ready valid variant accurately labelled', async () => {
  const sdk = await sdkBoundary();
  const engine = new VisionEngine();
  const ready = await engine.initialize('mediapipe', 'lite2');
  expect(() => engine.initialize('mediapipe', 'not-a-variant')).toThrow(RangeError);
  expect(engine.getState()).toBe(ready);
  expect(sdk.detector).toHaveBeenCalledTimes(1);
  engine.dispose();
});

it('late initialization cannot replace a newer variant or retain its task', async () => {
  const sdk = await sdkBoundary();
  let resolve!: (value: ObjectDetector) => void;
  const pending = new Promise<ObjectDetector>((yes) => {
    resolve = yes;
  });
  sdk.detector.mockImplementationOnce(() => pending);

  const engine = new VisionEngine();
  const older = engine.initialize('mediapipe', 'lite2');
  await vi.waitFor(() => expect(sdk.detector).toHaveBeenCalledTimes(1));
  expect(await engine.initialize('mediapipe', 'lite0')).toMatchObject({
    status: 'ready',
    variantId: 'lite0',
  });
  const close = vi.fn();
  resolve({ close, detectForVideo: vi.fn() } as unknown as ObjectDetector);
  expect(await older).toMatchObject({ status: 'cancelled' });
  expect(engine.getState()).toMatchObject({ status: 'ready', variantId: 'lite0' });
  expect(close).toHaveBeenCalledTimes(1);
  engine.dispose();
});
