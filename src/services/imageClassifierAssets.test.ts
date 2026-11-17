import { createHash } from 'node:crypto';
import { afterEach, expect, it, vi } from 'vitest';
import {
  ClassificationAssetsBlockedError,
  IMAGE_CLASSIFIER_ASSETS,
  classifierAssetStatuses,
  resolveClassifierAsset,
} from './imageClassifierAssets';
import { MODEL_CACHE_NAME } from './browserModelCache';

const variants = ['lite0', 'lite2'] as const;

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function installMockCaches(initial: Record<string, Response> = {}) {
  const store = new Map<string, Response>(Object.entries(initial));
  const cacheObj = {
    match: vi.fn(async (key: RequestInfo | URL) => {
      const k = typeof key === 'string' ? key : key instanceof URL ? key.href : key.url;
      return store.get(k)?.clone() ?? undefined;
    }),
    put: vi.fn(async (key: RequestInfo | URL, res: Response) => {
      const k = typeof key === 'string' ? key : key instanceof URL ? key.href : key.url;
      store.set(k, res);
    }),
    delete: vi.fn(async (key: RequestInfo | URL) => {
      const k = typeof key === 'string' ? key : key instanceof URL ? key.href : key.url;
      return store.delete(k);
    }),
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
  return { store, cacheObj };
}

it.each(variants)('%s exposes exact pinned metadata for client-side Web Crypto SHA-256 verification', (variant) => {
  const e = IMAGE_CLASSIFIER_ASSETS[variant];
  expect(e.upstreamUrl).toBe(
    `https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_${variant}/float32/1/efficientnet_${variant}.tflite`
  );
  expect(e.localUrl).toBe(e.upstreamUrl);
  expect(e.expectedSize).toBe(variant === 'lite0' ? 18_582_189 : 24_325_765);
  expect(e.expectedSha256).toBe(
    variant === 'lite0'
      ? '6c7ab0a6e5dcbf38a8c33b960996a55a3b4300b36a018c4545801de3a3c8bde0'
      : '94b6c84a5d6de20f932558de213b50e0338dcc06ab83d8d64b12128a476b4b83'
  );
  expect(e.inputShape).toEqual([1, variant === 'lite0' ? 224 : 260, variant === 'lite0' ? 224 : 260, 3]);
  expect(e.outputShape).toEqual([1, 1000]);
  expect(classifierAssetStatuses()[variant]).toMatchObject({
    status: 'unavailable',
    size: null,
    sha256: null,
    filePresent: null,
    license: 'Apache-2.0',
    acquisitionEnabled: true,
  });
});

it.each(variants)('%s downloads from upstream, verifies client SHA-256, stores in Browser Cache Storage, and reuses cached bytes', async (variant) => {
  const e = IMAGE_CLASSIFIER_ASSETS[variant];
  const syntheticBytes = new Uint8Array(e.expectedSize).fill(variant === 'lite0' ? 0x1a : 0x2b);
  const digest = createHash('sha256').update(syntheticBytes).digest('hex');

  vi.stubGlobal('crypto', {
    subtle: {
      digest: vi.fn(async (_alg: string, buf: BufferSource) => {
        const view = buf instanceof ArrayBuffer ? new Uint8Array(buf) : new Uint8Array(buf.buffer);
        if (view.byteLength === syntheticBytes.byteLength && view[0] === syntheticBytes[0]) {
          return Uint8Array.from(e.expectedSha256.match(/../g)!.map((hex) => parseInt(hex, 16))).buffer;
        }
        return new Uint8Array(createHash('sha256').update(view).digest()).buffer;
      }),
    },
  });
  expect(digest).not.toBe('');

  const { cacheObj } = installMockCaches();
  const mockFetch = vi.fn(async (url: string | URL | Request) => {
    expect(String(url)).toBe(e.upstreamUrl);
    return new Response(syntheticBytes, {
      status: 200,
      headers: { 'Content-Type': 'application/octet-stream' },
    });
  });

  // First call hits network ('external') and populates Cache Storage
  const first = await resolveClassifierAsset(variant, { fetchImpl: mockFetch as unknown as typeof fetch });
  expect(first).toMatchObject({
    modelId: 'image-classifier',
    variantId: variant,
    modelAssetPath: e.upstreamUrl,
    source: 'external',
  });
  expect(first.modelAssetBuffer.byteLength).toBe(e.expectedSize);
  expect(mockFetch).toHaveBeenCalledTimes(1);
  expect(cacheObj.put).toHaveBeenCalledTimes(1);

  // Second call hits Cache Storage ('browser-cache') without network fetch
  const second = await resolveClassifierAsset(variant, { fetchImpl: mockFetch as unknown as typeof fetch });
  expect(second).toMatchObject({
    modelId: 'image-classifier',
    variantId: variant,
    modelAssetPath: e.upstreamUrl,
    source: 'browser-cache',
  });
  expect(mockFetch).toHaveBeenCalledTimes(1);
});

it.each(['network-error', 'http-503', 'html-body', 'wrong-size', 'wrong-sha256'] as const)(
  'blocks initialization cleanly on %s and purges corrupt browser cache entries',
  async (mode) => {
    const e = IMAGE_CLASSIFIER_ASSETS.lite0;
    const { store, cacheObj } = installMockCaches({
      [e.upstreamUrl]: new Response(new Uint8Array(100), { status: 200 }),
    });

    const mockFetch = vi.fn(async () => {
      if (mode === 'network-error') throw new Error('Network offline');
      if (mode === 'http-503') return new Response('unavailable', { status: 503 });
      if (mode === 'html-body') {
        return new Response('<!doctype html><html>error</html>', {
          status: 200,
          headers: { 'Content-Type': 'text/html' },
        });
      }
      if (mode === 'wrong-size') {
        return new Response(new Uint8Array(2048), { status: 200 });
      }
      // wrong-sha256: right size, wrong bytes
      return new Response(new Uint8Array(e.expectedSize).fill(0x55), { status: 200 });
    });

    await expect(
      resolveClassifierAsset('lite0', { fetchImpl: mockFetch as unknown as typeof fetch })
    ).rejects.toBeInstanceOf(ClassificationAssetsBlockedError);
    expect(cacheObj.delete).toHaveBeenCalledWith(e.upstreamUrl);
    expect(store.size).toBe(0);
  }
);
