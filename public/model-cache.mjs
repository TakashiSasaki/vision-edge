export const MODEL_CACHE_NAME = 'vision-studio-models-v1';

export function formatBytes(bytes) {
  if (typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes < 0) return '—';
  if (bytes < 1024) return `${bytes.toLocaleString('ja-JP')} B`;
  const mb = bytes / (1024 * 1024);
  if (mb >= 1) {
    return `${mb.toFixed(1)} MB (${bytes.toLocaleString('ja-JP')} bytes)`;
  }
  return `${(bytes / 1024).toFixed(1)} KB (${bytes.toLocaleString('ja-JP')} bytes)`;
}

function sanitizeSafeUrl(rawUrl) {
  if (typeof rawUrl !== 'string' || !rawUrl.length) return null;
  try {
    const parsed = new URL(rawUrl, typeof window !== 'undefined' ? window.location.origin : 'https://localhost');
    if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') return null;
    return parsed;
  } catch {
    return null;
  }
}

export function summarizeBrowserCache(records) {
  if (!Array.isArray(records)) {
    throw new TypeError('Browser cache records must be an array');
  }
  const entries = [];
  let totalBytes = 0;
  let invalid = 0;

  for (const record of records) {
    const parsed = sanitizeSafeUrl(record?.url);
    const size = typeof record?.size === 'number' && Number.isSafeInteger(record.size) && record.size >= 0
      ? record.size
      : null;
    if (!parsed) {
      invalid++;
      continue;
    }
    const parts = parsed.pathname.split('/').filter(Boolean);
    const rawName = parts.at(-1) || parsed.pathname || parsed.host;
    const safeFileName = rawName.replace(/[\r\n<>]/g, '').slice(0, 160) || 'cached-model';
    if (size !== null) totalBytes += size;
    entries.push({
      url: parsed.href,
      host: parsed.host,
      fileName: safeFileName,
      size,
      sizeLabel: formatBytes(size),
      contentType: typeof record?.contentType === 'string' ? record.contentType.slice(0, 100) : 'application/octet-stream',
      cachedAt: typeof record?.cachedAt === 'string' && Number.isFinite(Date.parse(record.cachedAt))
        ? new Date(record.cachedAt).toLocaleString('ja-JP')
        : '—',
      sha256: typeof record?.sha256 === 'string' && /^[a-f0-9]{64}$/i.test(record.sha256)
        ? `${record.sha256.slice(0, 12)}...`
        : '—',
    });
  }

  entries.sort((a, b) => a.fileName.localeCompare(b.fileName));
  return { entries, totalBytes, invalid };
}

export async function readBrowserCacheState(
  cachesApi = typeof globalThis.caches !== 'undefined' ? globalThis.caches : undefined,
  swContainer = typeof navigator !== 'undefined' ? navigator.serviceWorker : undefined
) {
  let swStatus = {
    supported: Boolean(swContainer),
    registered: false,
    activeState: null,
    controllerActive: Boolean(swContainer?.controller),
  };

  if (swContainer && typeof swContainer.getRegistration === 'function') {
    try {
      const reg = await swContainer.getRegistration('/');
      if (reg) {
        swStatus = {
          supported: true,
          registered: true,
          activeState: reg.active?.state ?? reg.waiting?.state ?? reg.installing?.state ?? null,
          controllerActive: Boolean(swContainer.controller),
        };
      }
    } catch {
      // Service Worker inspection failure stays non-fatal
    }
  }

  if (!cachesApi || typeof cachesApi.open !== 'function') {
    return { available: false, swStatus, entries: [], totalBytes: 0, invalid: 0 };
  }

  const cache = await cachesApi.open(MODEL_CACHE_NAME);
  const requests = await cache.keys();
  const rawRecords = [];

  for (const req of requests) {
    const res = await cache.match(req);
    if (!res) continue;
    const lenHeader = res.headers.get('content-length');
    let size = lenHeader && /^\d+$/.test(lenHeader) ? Number(lenHeader) : null;
    if (size === null) {
      try {
        size = (await res.clone().arrayBuffer()).byteLength;
      } catch {
        size = null;
      }
    }
    rawRecords.push({
      url: req.url,
      size,
      contentType: res.headers.get('content-type') || 'application/octet-stream',
      cachedAt: res.headers.get('x-vision-cached-at'),
      sha256: res.headers.get('x-vision-sha256'),
    });
  }

  const summary = summarizeBrowserCache(rawRecords);
  return { available: true, swStatus, ...summary };
}

function renderCacheTable(document, host, entries, onDelete) {
  host.replaceChildren();
  if (!entries.length) {
    const p = document.createElement('p');
    p.textContent = 'このブラウザの Cache Storage にはまだモデル・WASMが保存されていません（モデルを実行すると自動保存されます）。';
    host.append(p);
    return;
  }

  const headers = ['ファイル名 / URL', '配信元', 'サイズ', 'SHA-256確認', '保存日時', '操作'];
  const table = document.createElement('table');
  const head = table.createTHead();
  const hrow = head.insertRow();
  for (const label of headers) {
    const th = document.createElement('th');
    th.scope = 'col';
    th.textContent = label;
    hrow.append(th);
  }
  const body = table.createTBody();

  for (const entry of entries) {
    const tr = body.insertRow();

    const tdFile = tr.insertCell();
    tdFile.dataset.label = headers[0];
    const strong = document.createElement('strong');
    strong.textContent = entry.fileName;
    const pathDiv = document.createElement('div');
    pathDiv.className = 'cache-path';
    pathDiv.textContent = entry.url;
    tdFile.append(strong, pathDiv);

    const tdHost = tr.insertCell();
    tdHost.dataset.label = headers[1];
    tdHost.textContent = entry.host;

    const tdSize = tr.insertCell();
    tdSize.dataset.label = headers[2];
    tdSize.textContent = entry.sizeLabel;

    const tdSha = tr.insertCell();
    tdSha.dataset.label = headers[3];
    tdSha.textContent = entry.sha256;

    const tdTime = tr.insertCell();
    tdTime.dataset.label = headers[4];
    tdTime.textContent = entry.cachedAt;

    const tdAction = tr.insertCell();
    tdAction.dataset.label = headers[5];
    const delBtn = document.createElement('button');
    delBtn.type = 'button';
    delBtn.className = 'danger-button small-button';
    delBtn.textContent = '削除';
    delBtn.setAttribute('aria-label', `${entry.fileName} をキャッシュから削除`);
    delBtn.addEventListener('click', () => {
      void onDelete(entry.url);
    });
    tdAction.append(delBtn);
  }

  host.append(table);
}

export function mountInventory(document, env = {}) {
  const cachesApi = env.cachesApi ?? (typeof globalThis.caches !== 'undefined' ? globalThis.caches : undefined);
  const swContainer = env.serviceWorkerContainer ?? (typeof navigator !== 'undefined' ? navigator.serviceWorker : undefined);

  const pageUrlEl = document.getElementById('page-url');
  const copyBtn = document.getElementById('copy-page-url');
  const copyNotice = document.getElementById('copy-notice');
  if (pageUrlEl) {
    const currentUrl =
      typeof window !== 'undefined' && window.location?.href
        ? window.location.href
        : document.URL || '/model-cache.html';
    pageUrlEl.textContent = currentUrl;
    if (copyBtn) {
      copyBtn.addEventListener('click', async () => {
        try {
          if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(currentUrl);
          } else {
            const ta = document.createElement('textarea');
            ta.value = currentUrl;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.append(ta);
            ta.focus();
            ta.select();
            document.execCommand('copy');
            ta.remove();
          }
          if (copyNotice) copyNotice.textContent = 'URLをコピーしました！';
          setTimeout(() => {
            if (copyNotice) copyNotice.textContent = '';
          }, 3000);
        } catch {
          if (copyNotice) copyNotice.textContent = 'コピーに失敗しました。';
        }
      });
    }
  }

  const refreshBtn = document.getElementById('refresh');
  const clearAllBtn = document.getElementById('clear-all');
  const swStatusEl = document.getElementById('sw-status');
  const notice = document.getElementById('notice');
  const observed = document.getElementById('observed');
  const cachedHost = document.getElementById('cached');

  let pending = false;

  async function deleteEntry(url) {
    if (!cachesApi || typeof cachesApi.open !== 'function') return;
    try {
      const cache = await cachesApi.open(MODEL_CACHE_NAME);
      await cache.delete(url);
      swContainer?.controller?.postMessage({ type: 'DELETE_MODEL_CACHE_ENTRY', url });
    } finally {
      await refresh();
    }
  }

  async function clearAll() {
    if (!cachesApi || typeof cachesApi.delete !== 'function') return;
    if (clearAllBtn) clearAllBtn.disabled = true;
    try {
      await cachesApi.delete(MODEL_CACHE_NAME);
      swContainer?.controller?.postMessage({ type: 'CLEAR_MODEL_CACHE' });
    } finally {
      if (clearAllBtn) clearAllBtn.disabled = false;
      await refresh();
    }
  }

  async function refresh() {
    if (pending) return;
    pending = true;
    if (refreshBtn) refreshBtn.disabled = true;
    if (notice) notice.textContent = 'ブラウザキャッシュを確認中…';

    try {
      const state = await readBrowserCacheState(cachesApi, swContainer);
      if (swStatusEl) {
        if (!state.swStatus.supported) {
          swStatusEl.textContent = 'Service Worker: このブラウザでは非対応です（Cache Storage 直接参照のみ使用）';
        } else if (state.swStatus.registered) {
          swStatusEl.textContent = `Service Worker: 登録済み (${state.swStatus.activeState ?? 'pending'}${state.swStatus.controllerActive ? '・ページ制御中' : ''})`;
        } else {
          swStatusEl.textContent = 'Service Worker: 未登録（HTTPSまたはメインスタジオ起動時に登録されます）';
        }
      }

      if (!state.available) {
        if (notice) notice.textContent = '取得不能：この環境では Cache Storage API を参照できません。';
        if (observed) observed.textContent = `確認時刻: ${new Date().toLocaleString('ja-JP')}`;
        cachedHost?.replaceChildren();
        return;
      }

      if (notice) {
        notice.textContent = state.invalid
          ? `ブラウザ保存済み ${state.entries.length}件（合計 ${formatBytes(state.totalBytes)} · 無効レコード ${state.invalid}件を除外）`
          : `ブラウザ保存済み ${state.entries.length}件（合計 ${formatBytes(state.totalBytes)}）`;
      }
      if (observed) {
        observed.textContent = `確認時刻: ${new Date().toLocaleString('ja-JP')} ／ ストア名: ${MODEL_CACHE_NAME}`;
      }
      if (clearAllBtn) {
        clearAllBtn.disabled = state.entries.length === 0;
      }
      if (cachedHost) {
        renderCacheTable(document, cachedHost, state.entries, deleteEntry);
      }
    } catch {
      if (notice) notice.textContent = '取得不能：ブラウザキャッシュの読み取りに失敗しました。';
      if (observed) observed.textContent = `確認試行時刻: ${new Date().toLocaleString('ja-JP')}`;
      cachedHost?.replaceChildren();
    } finally {
      pending = false;
      if (refreshBtn) refreshBtn.disabled = false;
    }
  }

  refreshBtn?.addEventListener('click', refresh);
  clearAllBtn?.addEventListener('click', clearAll);
  void refresh();
  return refresh;
}
