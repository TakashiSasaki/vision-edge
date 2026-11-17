const APP_CACHE_NAME = 'vision-studio-app-v3';
const MODEL_CACHE_NAME = 'vision-studio-models-v1';

const PRECACHE_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icon.svg',
  '/pwa-192x192.png',
  '/pwa-512x512.png',
  '/pwa-maskable-512x512.png',
  '/apple-touch-icon.png',
  '/favicon.ico',
];

const MODEL_BINARY_EXTENSIONS = ['.onnx', '.tflite', '.cact', '.wasm', '.data'];

const ALLOWED_EXTERNAL_CACHE_HOSTS = new Set([
  'storage.googleapis.com',
  'huggingface.co',
  'cdn-lfs.huggingface.co',
  'cdn-lfs-us-1.huggingface.co',
  'cdn-lfs-eu-1.huggingface.co',
  'cdn.jsdelivr.net',
  'raw.githubusercontent.com',
  'drive.usercontent.google.com',
]);

function isAllowedExternalHost(hostname) {
  return (
    ALLOWED_EXTERNAL_CACHE_HOSTS.has(hostname) ||
    hostname.endsWith('.hf.co') ||
    hostname.endsWith('.huggingface.co')
  );
}

function hasExtension(pathname, extensions) {
  const lower = pathname.toLowerCase();
  return extensions.some((ext) => lower.endsWith(ext));
}

function isModelOrWasmAssetRequest(url) {
  if (url.origin === self.location.origin) {
    if (url.pathname.startsWith('/api/cors-proxy/stereo/')) return true;
    if (url.pathname.startsWith('/wasm/cactus/')) {
      return hasExtension(url.pathname, ['.wasm', '.js', '.mjs']);
    }
    return hasExtension(url.pathname, MODEL_BINARY_EXTENSIONS);
  }

  if (!isAllowedExternalHost(url.hostname)) return false;
  if (hasExtension(url.pathname, MODEL_BINARY_EXTENSIONS)) return true;
  if (url.hostname === 'cdn.jsdelivr.net' && hasExtension(url.pathname, ['.js', '.mjs'])) {
    return true;
  }
  if (url.hostname === 'drive.usercontent.google.com' && url.pathname === '/download') {
    return true;
  }
  return false;
}

function isCacheableModelResponse(response) {
  if (!response || !response.ok || response.status !== 200 || response.type === 'opaque') {
    return false;
  }
  const contentType = (response.headers.get('content-type') || '').toLowerCase();
  if (contentType.startsWith('text/html')) {
    return false;
  }
  return true;
}

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(APP_CACHE_NAME)
      .then(async (cache) => {
        for (const asset of PRECACHE_ASSETS) {
          try {
            await cache.add(asset);
          } catch {
            // Non-blocking: individual cache failure must not fail SW install
          }
        }
      })
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => {
        return Promise.all(
          keys
            .filter((key) => key !== APP_CACHE_NAME && key !== MODEL_CACHE_NAME)
            .map((key) => caches.delete(key))
        );
      })
      .then(() => self.clients.claim())
  );
});

self.addEventListener('message', (event) => {
  const data = event.data;
  if (!data || typeof data.type !== 'string') return;
  if (data.type === 'SKIP_WAITING') {
    void self.skipWaiting();
    return;
  }
  if (data.type === 'CLEAR_MODEL_CACHE') {
    event.waitUntil(caches.delete(MODEL_CACHE_NAME));
    return;
  }
  if (data.type === 'DELETE_MODEL_CACHE_ENTRY' && typeof data.url === 'string') {
    event.waitUntil(
      caches.open(MODEL_CACHE_NAME).then((cache) => cache.delete(data.url))
    );
  }
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;

  // Never intercept Range requests (e.g. chunked >28MB downloads or video seeking)
  if (event.request.headers.has('range')) return;

  let url;
  try {
    url = new URL(event.request.url);
  } catch {
    return;
  }

  // Skip Vite dev/HMR internals and Mirage non-model fixture APIs
  if (
    url.origin === self.location.origin &&
    (url.pathname.startsWith('/src/') ||
      url.pathname.startsWith('/@') ||
      url.pathname.startsWith('/node_modules/') ||
      url.pathname.startsWith('/api/mirage/fixtures/') ||
      url.searchParams.has('t') ||
      url.searchParams.has('v'))
  ) {
    return;
  }

  // Cache-first browser caching for models, WASM binaries, CDN runtimes, and stateless stereo CORS proxy
  if (isModelOrWasmAssetRequest(url)) {
    event.respondWith(
      caches.open(MODEL_CACHE_NAME).then(async (cache) => {
        const cached = await cache.match(event.request);
        if (cached && isCacheableModelResponse(cached)) {
          return cached;
        }
        const response = await fetch(event.request);
        if (isCacheableModelResponse(response)) {
          const cloned = response.clone();
          const headers = new Headers(cloned.headers);
          if (!headers.has('x-vision-cached-at')) {
            headers.set('x-vision-cached-at', new Date().toISOString());
          }
          const cacheBody = await cloned.arrayBuffer();
          await cache
            .put(
              event.request,
              new Response(cacheBody, {
                status: cloned.status,
                statusText: cloned.statusText,
                headers,
              })
            )
            .catch(() => {});
        }
        return response;
      })
    );
    return;
  }

  // Never intercept unrecognized cross-origin requests or remaining /api/ endpoints
  if (url.origin !== self.location.origin || url.pathname.startsWith('/api/')) {
    return;
  }

  // Navigation requests: Network first, fall back to cached shell
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).catch(async () => {
        const cached = (await caches.match('/index.html')) || (await caches.match('/'));
        if (cached) return cached;
        return new Response('Offline', { status: 503, headers: { 'Content-Type': 'text/plain' } });
      })
    );
    return;
  }

  // Static app assets: Cache first with network fallback
  event.respondWith(
    caches
      .match(event.request)
      .then((cached) => {
        if (cached) {
          return cached;
        }
        return fetch(event.request).then((response) => {
          if (!response || response.status !== 200 || response.type !== 'basic') {
            return response;
          }
          const cloned = response.clone();
          caches
            .open(APP_CACHE_NAME)
            .then((cache) => cache.put(event.request, cloned))
            .catch(() => {});
          return response;
        });
      })
      .catch(() => fetch(event.request))
  );
});
