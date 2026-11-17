import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  MODEL_CACHE_NAME,
  formatBytes,
  mountInventory,
  readBrowserCacheState,
  summarizeBrowserCache,
} from '../../public/model-cache.mjs';

function makeFakeCaches(initialEntries: Record<string, { body: Uint8Array; headers?: Record<string, string> }> = {}) {
  const store = new Map<string, Response>();
  for (const [url, { body, headers }] of Object.entries(initialEntries)) {
    const mergedHeaders = new Headers({
      'Content-Type': 'application/octet-stream',
      'Content-Length': String(body.byteLength),
      'X-Vision-Cached-At': '2026-11-16T00:00:00.000Z',
      ...(headers ?? {}),
    });
    store.set(url, new Response(body.slice().buffer, { status: 200, headers: mergedHeaders }));
  }

  const cacheObj = {
    keys: vi.fn(async () => Array.from(store.keys()).map((url) => ({ url } as Request))),
    match: vi.fn(async (req: Request | string) => {
      const key = typeof req === 'string' ? req : req.url;
      return store.get(key)?.clone() ?? undefined;
    }),
    delete: vi.fn(async (req: Request | string) => {
      const key = typeof req === 'string' ? req : req.url;
      return store.delete(key);
    }),
  };

  return {
    store,
    cacheObj,
    cachesApi: {
      open: vi.fn(async (name: string) => {
        expect(name).toBe(MODEL_CACHE_NAME);
        return cacheObj;
      }),
      delete: vi.fn(async (name: string) => {
        expect(name).toBe(MODEL_CACHE_NAME);
        store.clear();
        return true;
      }),
    },
  };
}

function setupDom() {
  for (const id of ['refresh', 'clear-all']) {
    const btn = document.createElement('button');
    btn.id = id;
    document.body.append(btn);
  }
  for (const id of ['sw-status', 'notice', 'observed', 'cached']) {
    const div = document.createElement('div');
    div.id = id;
    document.body.append(div);
  }
}

describe('Browser model-cache inventory UI', () => {
  afterEach(() => {
    document.body.replaceChildren();
    vi.restoreAllMocks();
  });

  it('formats byte sizes cleanly and summarizes valid browser cache records while excluding invalid URLs', () => {
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(2048)).toContain('2.0 KB');
    expect(formatBytes(18_582_189)).toContain('17.7 MB');
    expect(formatBytes(null)).toBe('—');
    expect(() => summarizeBrowserCache(null as unknown as [])).toThrow();

    const summary = summarizeBrowserCache([
      {
        url: 'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite',
        size: 18_582_189,
        contentType: 'application/octet-stream',
        cachedAt: '2026-11-16T01:00:00.000Z',
        sha256: 'a'.repeat(64),
      },
      {
        url: 'javascript:alert(1)',
        size: 100,
      },
    ]);

    expect(summary.invalid).toBe(1);
    expect(summary.totalBytes).toBe(18_582_189);
    expect(summary.entries).toHaveLength(1);
    expect(summary.entries[0]).toMatchObject({
      fileName: 'efficientnet_lite0.tflite',
      host: 'storage.googleapis.com',
      sha256: 'aaaaaaaaaaaa...',
    });
  });

  it('renders browser Cache Storage entries safely, deletes individual items, and clears all with SW notifications', async () => {
    setupDom();
    const fake = makeFakeCaches({
      'https://huggingface.co/Xenova/segformer-b0/model_quantized.onnx': {
        body: new Uint8Array(4096),
      },
      'http://localhost/api/cors-proxy/stereo/hitnet_sceneflow_240x320.onnx': {
        body: new Uint8Array(2048),
        headers: { 'X-Vision-Sha256': 'b'.repeat(64) },
      },
    });
    const postMessage = vi.fn();
    const swContainer = {
      controller: { postMessage },
      getRegistration: vi.fn(async () => ({ active: { state: 'activated' } })),
    };

    mountInventory(document, {
      cachesApi: fake.cachesApi as unknown as CacheStorage,
      serviceWorkerContainer: swContainer as unknown as ServiceWorkerContainer,
    });

    await vi.waitFor(() => {
      expect(document.getElementById('notice')!.textContent).toContain('ブラウザ保存済み 2件');
    });
    expect(document.getElementById('sw-status')!.textContent).toContain('activated・ページ制御中');
    expect(document.querySelectorAll('#cached tbody tr')).toHaveLength(2);

    // Delete single item (first sorted alphabetically is hitnet_sceneflow_240x320.onnx)
    const delBtn = document.querySelector('#cached tbody tr button.danger-button') as HTMLButtonElement;
    delBtn.click();
    await vi.waitFor(() => {
      expect(document.getElementById('notice')!.textContent).toContain('ブラウザ保存済み 1件');
    });
    expect(postMessage).toHaveBeenCalledWith({
      type: 'DELETE_MODEL_CACHE_ENTRY',
      url: 'http://localhost/api/cors-proxy/stereo/hitnet_sceneflow_240x320.onnx',
    });

    // Clear all remaining items
    (document.getElementById('clear-all') as HTMLButtonElement).click();
    await vi.waitFor(() => {
      expect(document.getElementById('notice')!.textContent).toContain('ブラウザ保存済み 0件');
    });
    expect(postMessage).toHaveBeenCalledWith({ type: 'CLEAR_MODEL_CACHE' });
    expect((document.getElementById('clear-all') as HTMLButtonElement).disabled).toBe(true);
  });

  it('reports unavailable gracefully when Cache Storage API is absent', async () => {
    setupDom();
    const state = await readBrowserCacheState(null as unknown as CacheStorage, undefined);
    expect(state.available).toBe(false);

    mountInventory(document, { cachesApi: null as unknown as CacheStorage });
    await vi.waitFor(() => {
      expect(document.getElementById('notice')!.textContent).toContain('取得不能');
    });
  });

  it('displays the current page URL and copies it to clipboard on button click', async () => {
    setupDom();
    const pageUrl = document.createElement('code');
    pageUrl.id = 'page-url';
    const copyBtn = document.createElement('button');
    copyBtn.id = 'copy-page-url';
    const copyNotice = document.createElement('span');
    copyNotice.id = 'copy-notice';
    document.body.append(pageUrl, copyBtn, copyNotice);

    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    const fake = makeFakeCaches();

    mountInventory(document, { cachesApi: fake.cachesApi as unknown as CacheStorage });
    expect(pageUrl.textContent).toBeTruthy();
    copyBtn.click();
    await vi.waitFor(() => expect(copyNotice.textContent).toContain('コピーしました'));
    expect(writeText).toHaveBeenCalledWith(pageUrl.textContent);
  });
});
