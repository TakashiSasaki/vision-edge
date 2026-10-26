import { Buffer } from 'node:buffer';
import { mkdtemp, rm } from 'node:fs/promises';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FilesetResolver, ImageSegmenter, ObjectDetector } from '@mediapipe/tasks-vision';
import { ContainerModelCacheManager, containerModelCachePlugin } from './containerModelCacheServer';
import { prefetchModels } from '../../scripts/prefetch-models';
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
    expect(mockFetch).toHaveBeenCalledTimes(11);
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

  it('triggers startup model prefetching in dev and preview servers via containerModelCachePlugin', async () => {
    const fakePayload = Buffer.alloc(2048, 0x42);
    const mockFetch = vi.fn(async () => new Response(fakePayload, { status: 200 }));
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    const plugin = containerModelCachePlugin({
      cacheDir: tempDir,
      fetchImpl: mockFetch as unknown as typeof fetch,
      enableInPreview: true,
    });

    type MockServer = { middlewares: { use: (fn: unknown) => void } };
    const mockDevServer: MockServer = { middlewares: { use: vi.fn() } };
    if (typeof plugin.configureServer === 'function') {
      (plugin.configureServer as (server: unknown) => void)(mockDevServer);
    }
    expect(mockDevServer.middlewares.use).toHaveBeenCalledTimes(1);
    expect(logSpy).toHaveBeenCalledWith(expect.stringContaining('[container-model-cache] Container dev server started'));

    const mockPreviewServer: MockServer = { middlewares: { use: vi.fn() } };
    const savedTestLogDir = process.env.TEST_LOG_DIR;
    const savedCi = process.env.CI;
    try {
      delete process.env.TEST_LOG_DIR;
      delete process.env.CI;
      if (typeof plugin.configurePreviewServer === 'function') {
        (plugin.configurePreviewServer as (server: unknown) => void)(mockPreviewServer);
      }
      expect(mockPreviewServer.middlewares.use).toHaveBeenCalledTimes(1);
      expect(logSpy).toHaveBeenCalledWith(expect.stringContaining('[container-model-cache] Container preview server started'));
      await vi.waitFor(() =>
        expect(logSpy).toHaveBeenCalledWith(
          expect.stringContaining('[container-model-cache] Startup download completed')
        )
      );
    } finally {
      if (savedTestLogDir !== undefined) process.env.TEST_LOG_DIR = savedTestLogDir;
      if (savedCi !== undefined) process.env.CI = savedCi;
    }
  });

  it('runs prefetchModels script function and returns cache counts for all 11 variants', async () => {
    const fakePayload = Buffer.alloc(2048, 0x42);
    const mockFetch = vi.fn(async () => new Response(fakePayload, { status: 200 }));
    const manager = new ContainerModelCacheManager({
      cacheDir: tempDir,
      fetchImpl: mockFetch as unknown as typeof fetch,
    });

    const result = await prefetchModels(manager);
    expect(result.totalCount).toBe(11);
    // 9 legacy variants accept synthetic bytes; 2 pinned image classifiers strictly require their pinned SHA256/size
    expect(result.cachedCount).toBe(9);
    expect(result.unavailableCount).toBe(2);
  });

  it('waits for container download to complete before responding when an uncached model file is accessed', async () => {
    let releaseDownload!: () => void;
    const downloadGate = new Promise<void>(resolve => {
      releaseDownload = resolve;
    });
    const fakePayload = Buffer.alloc(4096, 0x5a);
    let downloadCompleted = false;
    const mockFetch = vi.fn(async () => {
      await downloadGate;
      downloadCompleted = true;
      return new Response(fakePayload, { status: 200 });
    });

    const manager = new ContainerModelCacheManager({
      cacheDir: tempDir,
      fetchImpl: mockFetch as unknown as typeof fetch,
    });
    const middleware = manager.createMiddleware();

    let responseResolved = false;
    const responsePromise = invokeMiddleware(
      middleware,
      '/api/model-cache/models/yolov8n-seg.onnx'
    ).then(res => {
      responseResolved = true;
      return res;
    });

    await vi.waitFor(() => expect(mockFetch).toHaveBeenCalledTimes(1));
    expect(downloadCompleted).toBe(false);
    expect(responseResolved).toBe(false);

    releaseDownload();
    const res = await responsePromise;
    expect(downloadCompleted).toBe(true);
    expect(res.statusCode).toBe(200);
    expect(res.headers['x-model-source']).toBe('container-cache');
    expect(res.body).toEqual(fakePayload);

    // Even if the cached file is removed from the container later, accessing the model URL re-downloads and waits
    await rm(manager.getVariantFilePath('yolo-seg:yolov8n-seg'), { force: true });
    await rm(`${manager.getVariantFilePath('yolo-seg:yolov8n-seg')}.identity.json`, { force: true });
    const redownloadRes = await invokeMiddleware(
      middleware,
      '/api/model-cache/models/yolov8n-seg.onnx'
    );
    expect(redownloadRes.statusCode).toBe(200);
    expect(redownloadRes.body).toEqual(fakePayload);
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it('serves /model-cache.html with readable model file URLs for non-JS user agents without triggering downloads', async () => {
    const mockFetch = vi.fn(async () => new Response(Buffer.alloc(2048, 0x42), { status: 200 }));
    const manager = new ContainerModelCacheManager({
      cacheDir: tempDir,
      fetchImpl: mockFetch as unknown as typeof fetch,
    });
    const middleware = manager.createMiddleware();

    const headers: Record<string, string> = {};
    let body = '';
    const req = {
      url: '/model-cache.html',
      method: 'GET',
      headers: {
        host: 'example.run.app',
        'x-forwarded-proto': 'https',
      },
    } as unknown as IncomingMessage;
    const res = {
      statusCode: 200,
      setHeader(name: string, value: string) {
        headers[name.toLowerCase()] = String(value);
      },
      end(chunk?: Buffer | string) {
        body = chunk ? String(chunk) : '';
      },
    } as unknown as ServerResponse;

    await middleware(req, res, () => {});
    expect(res.statusCode).toBe(200);
    expect(headers['content-type']).toContain('text/html');
    expect(body).not.toContain('<noscript>');
    expect(body).toContain('https://example.run.app/model-cache.html');
    expect(body).toContain('https://example.run.app/api/model-cache/models/yolov8n-seg.onnx');
    expect(body).toContain('https://example.run.app/api/model-cache/models/efficientnet_lite0.tflite?sha256=');
    expect(mockFetch).not.toHaveBeenCalled();
  });
});
