import { Buffer } from 'node:buffer';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import type { IncomingMessage, ServerResponse } from 'node:http';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  MODEL_CACHE_NAME,
  clearBrowserModelCache,
  deleteBrowserCachedModel,
  fetchModelBytesWithBrowserCache,
  isLikelyHtmlPayload,
  listBrowserCachedModels,
} from './browserModelCache';
import {
  MODEL_ASSET_CATALOG,
  UPSTREAM_MEDIAPIPE_WASM_URL,
  UPSTREAM_ORT_WASM_URL,
  isContainerModelCacheEnabled,
  resolveMediaPipeWasmUrls,
  resolveModelAssetUrls,
  resolveOrtWasmUrls,
} from './modelAssetResolver';
import { MODEL_REGISTRY } from './modelRegistry';
import {
  STEREO_CORS_PROXY_PREFIX,
  STEREO_CORS_PROXY_UPSTREAMS,
  createModelCorsProxyMiddleware,
  modelCorsProxyPlugin,
} from './stereoCorsProxyServer';

function invokeMiddleware(
  middleware: ReturnType<typeof createModelCorsProxyMiddleware>,
  url: string,
  method = 'GET'
): Promise<{
  statusCode: number;
  headers: Record<string, string>;
  body: Buffer;
  passedThrough: boolean;
}> {
  return new Promise((resolve, reject) => {
    const headers: Record<string, string> = {};
    const chunks: Buffer[] = [];
    const req = { url, method } as IncomingMessage;
    const mockRes = {
      statusCode: 200,
      destroyed: false,
      writableEnded: false,
      setHeader(name: string, value: string) {
        headers[name.toLowerCase()] = String(value);
      },
      write(chunk: Buffer | Uint8Array) {
        chunks.push(Buffer.from(chunk));
        return true;
      },
      end(chunk?: Buffer | string) {
        if (chunk !== undefined) {
          chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
        }
        resolve({
          statusCode: mockRes.statusCode,
          headers,
          body: Buffer.concat(chunks),
          passedThrough: false,
        });
      },
      once() {},
      removeListener() {},
    };

    middleware(req, mockRes as unknown as ServerResponse, (err?: unknown) => {
      if (err) reject(err);
      else resolve({ statusCode: 404, headers, body: Buffer.alloc(0), passedThrough: true });
    }).catch(reject);
  });
}

describe('Browser Model Cache, Service Worker Rules & Stateless Stereo CORS Proxy', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('disables server-side container cache and routes all core models + WASM directly to CORS upstream URLs', async () => {
    expect(isContainerModelCacheEnabled()).toBe(false);
    expect(MODEL_REGISTRY).toHaveLength(4);
    expect(MODEL_ASSET_CATALOG['selfie-segmenter'].license).toBe('Apache-2.0');
    expect(MODEL_ASSET_CATALOG.mediapipe.license).toBe('Apache-2.0');
    expect(MODEL_ASSET_CATALOG['yolo-seg'].license).toBe('AGPL-3.0');
    expect(MODEL_ASSET_CATALOG.segformer.license).toBe(
      'NVIDIA Source Code License (Non-Commercial)'
    );

    for (const model of MODEL_REGISTRY) {
      expect(model.upstreamUrl).toMatch(/^https:\/\//);
      expect(model.localCachePath).toBe(model.upstreamUrl);
      const resolved = await resolveModelAssetUrls(model.id);
      expect(resolved).toEqual({
        primaryUrl: model.upstreamUrl,
        fallbackUrl: null,
        source: 'external',
      });
    }

    expect(await resolveMediaPipeWasmUrls()).toEqual({
      primaryUrl: UPSTREAM_MEDIAPIPE_WASM_URL,
      fallbackUrl: null,
      source: 'external',
    });
    expect(await resolveOrtWasmUrls()).toEqual({
      primaryUrl: UPSTREAM_ORT_WASM_URL,
      fallbackUrl: null,
      source: 'external',
    });
  });

  it('vite.config.ts mounts stateless modelCorsProxyPlugin instead of containerModelCachePlugin', async () => {
    const viteConfig = await readFile(path.resolve(process.cwd(), 'vite.config.ts'), 'utf8');
    expect(viteConfig).toContain('modelCorsProxyPlugin');
    expect(viteConfig).not.toContain('containerModelCachePlugin');
    expect(viteConfig).not.toContain('__CONTAINER_MODEL_CACHE_ENABLED__');
  });

  it('stateless stereo CORS proxy streams allowlisted GitHub Releases binaries without writing to container disk', async () => {
    const { mkdtemp, rm } = await import('node:fs/promises');
    const { tmpdir } = await import('node:os');
    const tempRoot = await mkdtemp(path.join(tmpdir(), 'stateless-proxy-'));
    try {
      const sampleOnnx = Buffer.alloc(2048, 0x33);
      const mockFetch = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
        expect(String(url)).toBe(STEREO_CORS_PROXY_UPSTREAMS['hitnet_sceneflow_240x320.onnx']);
        return new Response(init?.method === 'HEAD' ? null : sampleOnnx, {
          status: 200,
          headers: {
            'Content-Type': 'application/octet-stream',
            'Content-Length': String(sampleOnnx.byteLength),
          },
        });
      });

      const middleware = createModelCorsProxyMiddleware({
        rootDir: tempRoot,
        fetchImpl: mockFetch as unknown as typeof fetch,
      });

      // Zero disk directory creation inside .cache/container-models
      const getRes = await invokeMiddleware(
        middleware,
        `${STEREO_CORS_PROXY_PREFIX}hitnet_sceneflow_240x320.onnx`
      );
      expect(getRes.statusCode).toBe(200);
      expect(getRes.headers['cache-control']).toBe('public, max-age=86400');
      expect(getRes.headers['x-proxy-cache']).toBe('stateless-stream');
      expect(getRes.headers['x-upstream-host']).toBe('github.com');
      expect(getRes.body).toEqual(sampleOnnx);
      expect(mockFetch).toHaveBeenCalledTimes(1);

      // Second request streams directly again (no server disk state); browser SW handles client cache
      const headRes = await invokeMiddleware(
        middleware,
        `${STEREO_CORS_PROXY_PREFIX}hitnet_sceneflow_240x320.onnx`,
        'HEAD'
      );
      expect(headRes.statusCode).toBe(200);
      expect(headRes.body.byteLength).toBe(0);
      expect(mockFetch).toHaveBeenCalledTimes(2);
      expect(existsSync(path.resolve(tempRoot, '.cache', 'container-models'))).toBe(false);

      // Reject unknown filenames (404), unsafe paths (400), and unrelated routes pass through
      expect((await invokeMiddleware(middleware, `${STEREO_CORS_PROXY_PREFIX}unknown.onnx`)).statusCode).toBe(404);
      expect((await invokeMiddleware(middleware, `${STEREO_CORS_PROXY_PREFIX}..%2Fsecret.onnx`)).statusCode).toBe(400);
      expect((await invokeMiddleware(middleware, '/api/model-cache/status')).passedThrough).toBe(true);
      expect(mockFetch).toHaveBeenCalledTimes(2);
    } finally {
      await rm(tempRoot, { recursive: true, force: true });
    }
  });

  it('stateless stereo CORS proxy rejects HTML error pages with 502 and mounts on dev + preview', async () => {
    const htmlFetch = vi.fn(async () =>
      new Response('<!doctype html><html>rate limit</html>', {
        status: 200,
        headers: { 'Content-Type': 'text/html; charset=utf-8' },
      })
    );
    const middleware = createModelCorsProxyMiddleware({
      fetchImpl: htmlFetch as unknown as typeof fetch,
    });

    const res = await invokeMiddleware(
      middleware,
      `${STEREO_CORS_PROXY_PREFIX}crestereo_init_iter2_180x320.onnx`
    );
    expect(res.statusCode).toBe(502);
    expect(JSON.parse(res.body.toString('utf8')).error).toContain('HTML');

    const plugin = modelCorsProxyPlugin();
    const useDev = vi.fn();
    const usePreview = vi.fn();
    (plugin.configureServer as (s: unknown) => void)({ middlewares: { use: useDev } });
    (plugin.configurePreviewServer as (s: unknown) => void)({ middlewares: { use: usePreview } });
    expect(useDev).toHaveBeenCalledTimes(1);
    expect(usePreview).toHaveBeenCalledTimes(1);
  });

  it('browserModelCache caches valid binaries in Cache Storage, rejects HTML/corrupt payloads, and supports listing & deletion', async () => {
    const store = new Map<string, Response>();
    const cacheObj = {
      match: vi.fn(async (k: Request | string) => store.get(typeof k === 'string' ? k : k.url)?.clone()),
      put: vi.fn(async (k: Request | string, v: Response) => {
        store.set(typeof k === 'string' ? k : k.url, v);
      }),
      delete: vi.fn(async (k: Request | string) => store.delete(typeof k === 'string' ? k : k.url)),
      keys: vi.fn(async () => Array.from(store.keys()).map((url) => ({ url } as Request))),
    };
    vi.stubGlobal('caches', {
      open: vi.fn(async (name: string) => {
        expect(name).toBe(MODEL_CACHE_NAME);
        return cacheObj;
      }),
      delete: vi.fn(async () => {
        store.clear();
        return true;
      }),
    });

    expect(isLikelyHtmlPayload(new TextEncoder().encode('<!DOCTYPE html>'))).toBe(true);
    expect(isLikelyHtmlPayload(new Uint8Array(2048).fill(0x12))).toBe(false);

    const payload = new Uint8Array(2048).fill(0x42);
    const mockFetch = vi.fn(async () => new Response(payload, { status: 200 }));
    const url = 'https://cdn.jsdelivr.net/gh/zz990099/lightstereo_cpp@master/models/lightstereo_s_sceneflow_general_opt_256_512.onnx';

    const first = await fetchModelBytesWithBrowserCache(url, { fetchImpl: mockFetch as unknown as typeof fetch });
    expect(first.fromCache).toBe(false);
    expect(first.bytes).toEqual(payload);
    expect(mockFetch).toHaveBeenCalledTimes(1);

    const second = await fetchModelBytesWithBrowserCache(url, { fetchImpl: mockFetch as unknown as typeof fetch });
    expect(second.fromCache).toBe(true);
    expect(second.bytes).toEqual(payload);
    expect(mockFetch).toHaveBeenCalledTimes(1);

    const list = await listBrowserCachedModels();
    expect(list).toHaveLength(1);
    expect(list[0]).toMatchObject({
      url,
      host: 'cdn.jsdelivr.net',
      fileName: 'lightstereo_s_sceneflow_general_opt_256_512.onnx',
      size: 2048,
    });

    expect(await deleteBrowserCachedModel(url)).toBe(true);
    expect(await listBrowserCachedModels()).toEqual([]);

    await fetchModelBytesWithBrowserCache(url, { fetchImpl: mockFetch as unknown as typeof fetch });
    expect(await listBrowserCachedModels()).toHaveLength(1);
    expect(await clearBrowserModelCache()).toBe(true);
    expect(await listBrowserCachedModels()).toEqual([]);
  });

  it('public/sw.js and public/model-cache.html enforce browser-only caching and expose readable catalog URLs', async () => {
    const swCode = await readFile(path.resolve(process.cwd(), 'public', 'sw.js'), 'utf8');
    expect(swCode).toContain("MODEL_CACHE_NAME = 'vision-studio-models-v1'");
    expect(swCode).toContain('/api/cors-proxy/stereo/');
    expect(swCode).toContain('CLEAR_MODEL_CACHE');
    expect(swCode).toContain('DELETE_MODEL_CACHE_ENTRY');
    expect(swCode).toContain("event.request.headers.has('range')");

    const html = await readFile(path.resolve(process.cwd(), 'public', 'model-cache.html'), 'utf8');
    expect(html).not.toContain('/api/model-cache/');
    expect(html).toContain('https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite');
    expect(html).toContain('https://huggingface.co/unileon-robotics/YOLO8-ONNX/resolve/main/yolov8n-seg.onnx');
    expect(html).toContain('/api/cors-proxy/stereo/hitnet_sceneflow_240x320.onnx');
  });
});
