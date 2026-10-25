import { Buffer } from 'node:buffer';
import { mkdtemp, rm } from 'node:fs/promises';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FilesetResolver, ImageSegmenter, ObjectDetector } from '@mediapipe/tasks-vision';
import { ContainerModelCacheManager } from './containerModelCacheServer';
import {
  getResolvedSource,
  LOCAL_MEDIAPIPE_WASM_URL,
  MODEL_ASSET_CATALOG,
  resolveMediaPipeWasmUrls,
  resolveModelAssetUrls,
  resolveOrtWasmUrls,
  setContainerModelCacheEnabledForTest,
  UPSTREAM_MEDIAPIPE_WASM_URL,
  UPSTREAM_ORT_WASM_URL,
} from './modelAssetResolver';
import { MODEL_REGISTRY } from './modelRegistry';
import { VisionEngine } from './visionEngine';

function invokeMiddleware(
  middleware: ReturnType<ContainerModelCacheManager['createMiddleware']>,
  url: string,
  method = 'GET'
): Promise<{
  statusCode: number;
  headers: Record<string, string>;
  body: Buffer;
}> {
  return new Promise((resolve, reject) => {
    const headers: Record<string, string> = {};
    const chunks: Buffer[] = [];
    const req = { url, method } as IncomingMessage;
    const mockRes = {
      statusCode: 200,
      setHeader(name: string, value: string) {
        headers[name.toLowerCase()] = String(value);
      },
      end(chunk?: Buffer | string) {
        if (chunk !== undefined) {
          chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
        }
        resolve({
          statusCode: mockRes.statusCode,
          headers,
          body: Buffer.concat(chunks),
        });
      },
    };
    const res = mockRes as unknown as ServerResponse;

    middleware(req, res, (err?: unknown) => {
      if (err) reject(err);
      else resolve({ statusCode: 404, headers, body: Buffer.alloc(0) });
    }).catch(reject);
  });
}

describe('Container Model & WASM Cache Server and Client Fallback', () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(tmpdir(), 'container-model-cache-'));
    setContainerModelCacheEnabledForTest(null);
  });

  afterEach(async () => {
    setContainerModelCacheEnabledForTest(null);
    vi.restoreAllMocks();
    await rm(tempDir, { recursive: true, force: true });
  });

  it('exposes upstream URLs, local cache paths, and verified licenses for all 4 models in MODEL_REGISTRY', () => {
    expect(MODEL_REGISTRY).toHaveLength(4);
    expect(MODEL_ASSET_CATALOG['selfie-segmenter'].license).toBe('Apache-2.0');
    expect(MODEL_ASSET_CATALOG.mediapipe.license).toBe('Apache-2.0');
    expect(MODEL_ASSET_CATALOG['yolo-seg'].license).toBe('AGPL-3.0');
    expect(MODEL_ASSET_CATALOG.segformer.license).toBe(
      'NVIDIA Source Code License (Non-Commercial)'
    );

    for (const model of MODEL_REGISTRY) {
      expect(model.license).toBeTruthy();
      expect(model.upstreamUrl).toMatch(/^https:\/\//);
      expect(model.localCachePath).toMatch(/^\/api\/model-cache\/models\//);
    }
  });

  it('prefetches all 4 models into container cache on startup and serves models and WASM locally', async () => {
    const fakePayload = Buffer.alloc(2048, 0x42);
    const mockFetch = vi.fn(async () => new Response(fakePayload, { status: 200 }));

    const manager = new ContainerModelCacheManager({
      cacheDir: tempDir,
      fetchImpl: mockFetch as unknown as typeof fetch,
    });

    const snapshot = await manager.prefetchAllModels();
    expect(mockFetch).toHaveBeenCalledTimes(9);
    for (const id of ['selfie-segmenter', 'mediapipe', 'yolo-seg', 'segformer'] as const) {
      expect(snapshot[id].status).toBe('cached');
      expect(snapshot[id].size).toBe(2048);
    }

    const middleware = manager.createMiddleware();

    const statusRes = await invokeMiddleware(middleware, '/api/model-cache/status');
    expect(statusRes.statusCode).toBe(200);
    const statusJson = JSON.parse(statusRes.body.toString('utf8'));
    expect(statusJson.enabled).toBe(true);
    expect(statusJson.wasm.mediapipe.available).toBe(true);
    expect(statusJson.wasm.ort.available).toBe(true);
    expect(statusJson.models['yolo-seg'].status).toBe('cached');

    const modelRes = await invokeMiddleware(
      middleware,
      '/api/model-cache/models/yolov8n-seg.onnx'
    );
    expect(modelRes.statusCode).toBe(200);
    expect(modelRes.headers['x-model-source']).toBe('container-cache');
    expect(modelRes.headers['x-model-license']).toBe('AGPL-3.0');
    expect(modelRes.body.byteLength).toBe(2048);

    const wasmRes = await invokeMiddleware(
      middleware,
      '/api/model-cache/wasm/mediapipe/vision_wasm_internal.js'
    );
    expect(wasmRes.statusCode).toBe(200);
    expect(wasmRes.headers['x-wasm-source']).toBe('container-cache');
    expect(wasmRes.headers['content-type']).toContain('text/javascript');
    expect(wasmRes.body.byteLength).toBeGreaterThan(1000);
  });

  it('resolves container cache URLs when enabled and falls back to external URLs if a model fails to download or load', async () => {
    setContainerModelCacheEnabledForTest(true);

    const statusPayload = {
      enabled: true,
      wasm: {
        mediapipe: {
          available: true,
          localUrl: LOCAL_MEDIAPIPE_WASM_URL,
          upstreamUrl: UPSTREAM_MEDIAPIPE_WASM_URL,
          license: 'Apache-2.0',
        },
        ort: {
          available: true,
          localUrl: '/api/model-cache/wasm/ort/',
          upstreamUrl: UPSTREAM_ORT_WASM_URL,
          license: 'MIT',
        },
      },
      models: {
        'selfie-segmenter': {
          ...MODEL_ASSET_CATALOG['selfie-segmenter'],
          status: 'cached',
          size: 4096,
          cachedAt: new Date().toISOString(),
          error: null,
        },
        mediapipe: {
          ...MODEL_ASSET_CATALOG.mediapipe,
          status: 'unavailable',
          size: null,
          cachedAt: null,
          error: 'HTTP 403',
        },
        'yolo-seg': {
          ...MODEL_ASSET_CATALOG['yolo-seg'],
          status: 'cached',
          size: 4096,
          cachedAt: new Date().toISOString(),
          error: null,
        },
        segformer: {
          ...MODEL_ASSET_CATALOG.segformer,
          status: 'cached',
          size: 4096,
          cachedAt: new Date().toISOString(),
          error: null,
        },
      },
    };

    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(statusPayload), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    );

    const wasmResolved = await resolveMediaPipeWasmUrls();
    expect(wasmResolved.primaryUrl).toBe(LOCAL_MEDIAPIPE_WASM_URL);
    expect(wasmResolved.fallbackUrl).toBe(UPSTREAM_MEDIAPIPE_WASM_URL);
    expect(wasmResolved.source).toBe('container-cache');

    const ortWasmResolved = await resolveOrtWasmUrls();
    expect(ortWasmResolved.primaryUrl).toContain('/api/model-cache/wasm/ort/');
    expect(ortWasmResolved.fallbackUrl).toBe(UPSTREAM_ORT_WASM_URL);

    const cachedSelfie = await resolveModelAssetUrls('selfie-segmenter');
    expect(cachedSelfie.primaryUrl).toBe(
      '/api/model-cache/models/selfie_segmenter.tflite'
    );
    expect(cachedSelfie.fallbackUrl).toBe(
      MODEL_ASSET_CATALOG['selfie-segmenter'].upstreamUrl
    );
    expect(cachedSelfie.source).toBe('container-cache');

    const unavailableMediapipe = await resolveModelAssetUrls('mediapipe');
    expect(unavailableMediapipe.primaryUrl).toBe(
      MODEL_ASSET_CATALOG.mediapipe.upstreamUrl
    );
    expect(unavailableMediapipe.fallbackUrl).toBeNull();
    expect(unavailableMediapipe.source).toBe('external-fallback');

    // Verify VisionEngine defaultFactory falls back to external URL when local URL fails
    const forVisionTasksSpy = vi
      .spyOn(FilesetResolver, 'forVisionTasks')
      .mockRejectedValueOnce(new Error('local WASM failed'))
      .mockResolvedValueOnce({ wasmLoaderPath: 'loader', wasmBinaryPath: 'wasm' });

    const mockSegmenter = {
      segmentForVideo: vi.fn(),
      close: vi.fn(),
    } as unknown as ImageSegmenter;
    const createSegmenterSpy = vi
      .spyOn(ImageSegmenter, 'createFromOptions')
      .mockRejectedValueOnce(new Error('local model failed'))
      .mockResolvedValueOnce(mockSegmenter);

    const mockDetector = {
      detectForVideo: vi.fn(),
      close: vi.fn(),
    } as unknown as ObjectDetector;
    vi.spyOn(ObjectDetector, 'createFromOptions').mockResolvedValue(mockDetector);

    const engine = new VisionEngine();
    const result = await engine.initialize('selfie-segmenter');
    expect(result.status).toBe('ready');
    expect(forVisionTasksSpy).toHaveBeenNthCalledWith(1, LOCAL_MEDIAPIPE_WASM_URL);
    expect(forVisionTasksSpy).toHaveBeenNthCalledWith(2, UPSTREAM_MEDIAPIPE_WASM_URL);
    expect(createSegmenterSpy).toHaveBeenNthCalledWith(
      1,
      expect.anything(),
      expect.objectContaining({
        baseOptions: expect.objectContaining({
          modelAssetPath: '/api/model-cache/models/selfie_segmenter.tflite',
        }),
      })
    );
    expect(createSegmenterSpy).toHaveBeenNthCalledWith(
      2,
      expect.anything(),
      expect.objectContaining({
        baseOptions: expect.objectContaining({
          modelAssetPath: MODEL_ASSET_CATALOG['selfie-segmenter'].upstreamUrl,
        }),
      })
    );
    expect(getResolvedSource('wasm-mediapipe')).toBe('external-fallback');
    expect(getResolvedSource('selfie-segmenter')).toBe('external-fallback');
    engine.dispose();
  });
});
