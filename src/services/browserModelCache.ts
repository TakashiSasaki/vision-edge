export const MODEL_CACHE_NAME = 'vision-studio-models-v1';
export const STEREO_CORS_PROXY_PREFIX = '/api/cors-proxy/stereo/';
export const MIRAGE_FIXTURE_PREFIX = '/api/mirage/fixtures/';

export const STEREO_CORS_PROXY_UPSTREAMS: Readonly<Record<string, string>> = Object.freeze({
  'model_528_240_float32.onnx':
    'https://drive.usercontent.google.com/download?id=1Dkyrg5Fu554gqxfclkHC6zJBCYrdzOO0&export=download&confirm=t',
  'hitnet_sceneflow_240x320.onnx':
    'https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_sceneflow_240x320.onnx',
  'crestereo_init_iter2_180x320.onnx':
    'https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter2_180x320.onnx',
  'crestereo_combined_iter5_240x320.onnx':
    'https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter5_240x320.onnx',
});

const MIN_VALID_MODEL_BYTES = 1024;

export interface FetchModelWithCacheOptions {
  expectedSize?: number;
  expectedSha256?: string;
  fetchImpl?: typeof fetch;
  signal?: AbortSignal;
}

export interface BrowserModelFetchResult {
  bytes: Uint8Array;
  fromCache: boolean;
  sha256: string | null;
}

export interface BrowserCachedModelEntry {
  url: string;
  host: string;
  fileName: string;
  size: number | null;
  contentType: string;
  cachedAt: string | null;
}

function toAbsoluteUrl(url: string): string {
  if (url.startsWith('http://') || url.startsWith('https://')) {
    return url;
  }
  if (typeof window !== 'undefined' && window.location?.origin) {
    return new URL(url, window.location.origin).href;
  }
  return url;
}

export function isLikelyHtmlPayload(bytes: Uint8Array): boolean {
  if (bytes.byteLength === 0) return true;
  const head = String.fromCharCode(...bytes.subarray(0, Math.min(32, bytes.byteLength)))
    .trimStart()
    .toLowerCase();
  return head.startsWith('<!doctype') || head.startsWith('<html');
}

export async function computeSha256Hex(bytes: Uint8Array): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) {
    throw new Error('Web Crypto subtle.digest unavailable for SHA-256 verification');
  }
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const digest = await subtle.digest('SHA-256', copy.buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

export async function verifyModelBytes(
  bytes: Uint8Array,
  expectedSize?: number,
  expectedSha256?: string
): Promise<string | null> {
  if (bytes.byteLength < MIN_VALID_MODEL_BYTES || isLikelyHtmlPayload(bytes)) {
    throw new Error(`Invalid model binary payload (${bytes.byteLength} bytes)`);
  }
  if (expectedSize !== undefined && bytes.byteLength !== expectedSize) {
    throw new Error(`Model size mismatch: got ${bytes.byteLength}, expected ${expectedSize}`);
  }
  if (expectedSha256 !== undefined) {
    const sha256 = await computeSha256Hex(bytes);
    if (sha256 !== expectedSha256.toLowerCase()) {
      throw new Error(`Model SHA-256 mismatch: got ${sha256}, expected ${expectedSha256}`);
    }
    return sha256;
  }
  return null;
}

async function openModelCache(): Promise<Cache | null> {
  if (typeof globalThis.caches?.open !== 'function') {
    return null;
  }
  try {
    return await globalThis.caches.open(MODEL_CACHE_NAME);
  } catch {
    return null;
  }
}

export async function fetchModelBytesWithBrowserCache(
  url: string,
  options: FetchModelWithCacheOptions = {}
): Promise<BrowserModelFetchResult> {
  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  if (typeof fetchImpl !== 'function') {
    throw new Error('Fetch API is unavailable');
  }

  const cacheKey = toAbsoluteUrl(url);
  const cache = await openModelCache();

  if (cache) {
    try {
      const cached = await cache.match(cacheKey);
      if (cached && cached.ok && cached.status === 200 && cached.type !== 'opaque') {
        const contentType = cached.headers.get('content-type')?.toLowerCase() ?? '';
        if (!contentType.startsWith('text/html')) {
          const cachedBytes = new Uint8Array(await cached.arrayBuffer());
          const sha256 = await verifyModelBytes(
            cachedBytes,
            options.expectedSize,
            options.expectedSha256
          );
          return { bytes: cachedBytes, fromCache: true, sha256 };
        }
      }
      if (cached) {
        await cache.delete(cacheKey).catch(() => {});
      }
    } catch {
      await cache.delete(cacheKey).catch(() => {});
    }
  }

  const init: RequestInit = options.signal
    ? { method: 'GET', redirect: 'follow', signal: options.signal }
    : { method: 'GET', redirect: 'follow' };
  const response = await fetchImpl(url, init);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const contentType = response.headers.get('content-type')?.toLowerCase() ?? '';
  if (contentType.startsWith('text/html')) {
    throw new Error(`Unexpected HTML response from ${url}`);
  }

  const bytes = new Uint8Array(await response.arrayBuffer());
  const sha256 = await verifyModelBytes(bytes, options.expectedSize, options.expectedSha256);

  if (cache) {
    try {
      const copy = new Uint8Array(bytes.byteLength);
      copy.set(bytes);
      const headers = new Headers({
        'Content-Type': contentType || 'application/octet-stream',
        'Content-Length': String(copy.byteLength),
        'X-Vision-Cached-At': new Date().toISOString(),
        ...(sha256 ? { 'X-Vision-Sha256': sha256 } : {}),
      });
      await cache.put(cacheKey, new Response(copy.buffer, { status: 200, headers }));
    } catch {
      // Cache write failure (e.g. quota exceeded / private mode) must not block inference
    }
  }

  return { bytes, fromCache: false, sha256 };
}

export async function listBrowserCachedModels(): Promise<BrowserCachedModelEntry[]> {
  const cache = await openModelCache();
  if (!cache) return [];
  const requests = await cache.keys();
  const entries: BrowserCachedModelEntry[] = [];
  for (const req of requests) {
    const res = await cache.match(req);
    if (!res) continue;
    let parsed: URL | null;
    try {
      parsed = new URL(req.url);
    } catch {
      parsed = null;
    }
    const headerLen = res.headers.get('content-length');
    let size: number | null = headerLen && /^\d+$/.test(headerLen) ? Number(headerLen) : null;
    if (size === null) {
      try {
        size = (await res.clone().arrayBuffer()).byteLength;
      } catch {
        size = null;
      }
    }
    const pathParts = parsed?.pathname.split('/').filter(Boolean) ?? [];
    const fileName = pathParts.at(-1) || req.url;
    entries.push({
      url: req.url,
      host: parsed?.host ?? 'unknown',
      fileName,
      size,
      contentType: res.headers.get('content-type') || 'application/octet-stream',
      cachedAt: res.headers.get('x-vision-cached-at'),
    });
  }
  return entries.sort((a, b) => a.fileName.localeCompare(b.fileName));
}

export async function deleteBrowserCachedModel(url: string): Promise<boolean> {
  const cache = await openModelCache();
  if (!cache) return false;
  return cache.delete(url);
}

export async function clearBrowserModelCache(): Promise<boolean> {
  if (typeof globalThis.caches?.delete !== 'function') {
    return false;
  }
  return globalThis.caches.delete(MODEL_CACHE_NAME);
}
