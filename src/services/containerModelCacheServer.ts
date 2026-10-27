import { IMAGE_CLASSIFIER_ASSETS } from './classifierAssetManifest.ts';
import { classifierAssetStatuses } from './imageClassifierAssets.ts';
import { Buffer } from 'node:buffer';
import { existsSync, createReadStream } from 'node:fs';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, open, readFile, rename, rm, lstat, type FileHandle, writeFile } from 'node:fs/promises';
import type { IncomingMessage, ServerResponse } from 'node:http';
import path from 'node:path';
import process from 'node:process';
import type { Plugin } from 'vite';
import type { RunnableModelType } from '../types/vision.ts';
import {
  LOCAL_MEDIAPIPE_WASM_URL,
  LOCAL_ORT_WASM_URL,
  MODEL_ASSET_CATALOG,
  MODEL_VARIANT_CATALOG,
  UPSTREAM_MEDIAPIPE_WASM_URL,
  UPSTREAM_ORT_WASM_URL,
  type ContainerCacheStatusResponse,
  type CacheAssetModelId,
  type ModelVariantCatalogEntry,
  type ContainerModelStatusEntry,
  getVariantCatalogEntry,
} from './modelAssetResolver.ts';

type CacheCatalogEntry = Omit<ModelVariantCatalogEntry, 'modelId'> & {
  modelId: CacheAssetModelId; expectedSize?: number; expectedSha256?: string;
};
const CACHE_CATALOG: Record<string, CacheCatalogEntry> = {
  ...MODEL_VARIANT_CATALOG,
  ...Object.fromEntries(Object.values(IMAGE_CLASSIFIER_ASSETS).map(e => [`${e.modelId}:${e.variantId}`, e])),
};

const STARTUP_PRIORITY_KEYS: readonly string[] = [
  'selfie-segmenter:general',
  'mediapipe:lite0',
  'segformer:b0',
  'yolo-seg:yolov8n-seg',
  'image-classifier:lite0',
  'selfie-segmenter:landscape',
  'mediapipe:lite2',
  'segformer:b1',
  'image-classifier:lite2',
  'segformer:b2',
  'yolo-seg:yolov8s-seg',
];

function getOrderedCatalogKeys(prioritizeDefaults = false): string[] {
  const allKeys = Object.keys(CACHE_CATALOG);
  if (!prioritizeDefaults) return allKeys;
  const prioritized = STARTUP_PRIORITY_KEYS.filter(key => Object.hasOwn(CACHE_CATALOG, key));
  const remaining = allKeys.filter(key => !prioritized.includes(key));
  return [...prioritized, ...remaining];
}

// Public API errors are bounded diagnostics, never raw filesystem paths or
// arbitrary upstream messages (which may contain credentials).
function publicCacheError(cause: unknown): string {
  const message = cause instanceof Error ? cause.message : '';
  if (/^(HTTP \d{3}|Model cache acquisition timeout|Model exceeds cache size limit|Empty model response body|Downloaded payload too small \(\d+ bytes\)|Pinned model (size|SHA256) mismatch|invalid cache (file\/size|metadata file)|cache (identity metadata mismatch|byte hash mismatch|file exceeds size limit))$/.test(message)) return message;
  return 'Cache acquisition or validation failed';
}

const MIN_VALID_MODEL_BYTES = 1024;
const PREFETCH_TIMEOUT_MS = 45_000;
const MAX_MODEL_BYTES = 512 * 1024 * 1024;

export interface ContainerModelCacheOptions {
  rootDir?: string;
  cacheDir?: string;
  fetchImpl?: typeof fetch;
  autoPrefetchOnStart?: boolean;
  enableInPreview?: boolean;
  startupDelayMs?: number;
  startupConcurrency?: number;
  startupYieldMs?: number;
}

export interface PrefetchAllModelsOptions {
  concurrency?: number;
  yieldMs?: number;
  signal?: AbortSignal;
  prioritizeDefaults?: boolean;
}

function contentTypeForFile(filename: string): string {
  if (filename.endsWith('.wasm')) return 'application/wasm';
  if (filename.endsWith('.js') || filename.endsWith('.mjs')) {
    return 'text/javascript; charset=utf-8';
  }
  if (filename.endsWith('.json')) return 'application/json; charset=utf-8';
  return 'application/octet-stream';
}

function isSafeFilename(name: string): boolean {
  return (
    Boolean(name) &&
    !name.includes('/') &&
    !name.includes('\\') &&
    !name.includes('..') &&
    /^[a-zA-Z0-9._-]+$/.test(name)
  );
}

function getRequestOrigin(req: IncomingMessage): string | null {
  const forwardedHost = req.headers?.['x-forwarded-host'];
  const rawHost =
    (Array.isArray(forwardedHost) ? forwardedHost[0] : forwardedHost) ||
    req.headers?.host;
  if (!rawHost) return null;
  const host = rawHost.split(',')[0].trim();
  if (!host || !/^[a-zA-Z0-9.:-]+$/.test(host)) return null;
  const forwardedProto = req.headers?.['x-forwarded-proto'];
  const rawProto = (
    Array.isArray(forwardedProto) ? forwardedProto[0] : forwardedProto
  )
    ?.split(',')[0]
    ?.trim();
  const proto =
    rawProto === 'https' || rawProto === 'http'
      ? rawProto
      : host.endsWith('.run.app') || host.endsWith('.ai.studio')
        ? 'https'
        : 'http';
  return `${proto}://${host}`;
}

export class ContainerModelCacheManager {
  public readonly rootDir: string;
  public readonly cacheDir: string;
  private readonly fetchImpl: typeof fetch;
  private readonly variantStates = new Map<string, ContainerModelStatusEntry>();
  private readonly variantInflight = new Map<string, Promise<ContainerModelStatusEntry>>();

  constructor(options: ContainerModelCacheOptions = {}) {
    this.rootDir = options.rootDir ?? process.cwd();
    this.cacheDir =
      options.cacheDir ?? path.join(this.rootDir, '.cache', 'container-models');
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch;

    for (const key of Object.keys(CACHE_CATALOG)) {
      const entry = CACHE_CATALOG[key];
      const statusEntry: ContainerModelStatusEntry = {
        id: entry.modelId,
        variantId: entry.variantId,
        file: entry.file,
        status: 'unavailable',
        size: null,
        cachedAt: null,
        error: null,
        localUrl: entry.localUrl,
        upstreamUrl: entry.upstreamUrl,
        license: entry.license,
        filePresent: null, integrity: 'unknown', sha256: null,
        expectedSize: entry.expectedSize ?? null, expectedSha256: entry.expectedSha256 ?? null,
      };
      this.variantStates.set(key, statusEntry);
    }
  }

  public getModelFilePath(modelId: RunnableModelType): string {
    const entry = getVariantCatalogEntry(modelId);
    return this.getVariantFilePath(`${modelId}:${entry.variantId}`);
  }

  public getVariantFilePath(key: string): string {
    const entry = this.requireVariant(key);
    return path.join(this.cacheDir, entry.file);
  }

  public findModelByFilename(filename: string): CacheAssetModelId | null {
    if (!isSafeFilename(filename)) return null;
    for (const key of Object.keys(CACHE_CATALOG)) {
      if (CACHE_CATALOG[key].file === filename) {
        return CACHE_CATALOG[key].modelId;
      }
    }
    for (const modelId of Object.keys(MODEL_ASSET_CATALOG) as RunnableModelType[]) {
      if (MODEL_ASSET_CATALOG[modelId].file === filename) {
        return modelId;
      }
    }
    return null;
  }

  public findVariantKeyByFilename(filename: string): string | null {
    if (!isSafeFilename(filename)) return null;
    for (const key of Object.keys(CACHE_CATALOG)) {
      if (CACHE_CATALOG[key].file === filename) {
        return key;
      }
    }
    return null;
  }

  public getWasmDir(kind: 'mediapipe' | 'ort'): string {
    return kind === 'mediapipe'
      ? path.join(this.rootDir, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm')
      : path.join(this.rootDir, 'node_modules', 'onnxruntime-web', 'dist');
  }

  public isWasmAvailable(kind: 'mediapipe' | 'ort'): boolean {
    const dir = this.getWasmDir(kind);
    const probeFile =
      kind === 'mediapipe'
        ? path.join(dir, 'vision_wasm_internal.js')
        : path.join(dir, 'ort-wasm-simd-threaded.mjs');
    return existsSync(dir) && (existsSync(probeFile) || existsSync(path.join(dir, 'ort.min.js')));
  }

  private requireVariant(key: string) {
    if (!Object.hasOwn(CACHE_CATALOG, key)) throw new RangeError(`Unknown model variant: ${key}`);
    return CACHE_CATALOG[key];
  }

  public async syncDiskState(modelId: RunnableModelType): Promise<ContainerModelStatusEntry> {
    const entry = getVariantCatalogEntry(modelId);
    return this.syncVariantDiskState(`${modelId}:${entry.variantId}`);
  }

  private async readIdentity(key: string, size: number) {
    const entry = this.requireVariant(key), filename = `${this.getVariantFilePath(key)}.identity.json`;
    const info = await lstat(filename);
    if (!info.isFile() || info.isSymbolicLink() || info.size > 4096) throw new Error('invalid cache metadata file');
    const record = JSON.parse(await readFile(filename, 'utf8'));
    if (!record || record.schemaVersion !== 1 || record.modelId !== entry.modelId || record.variantId !== entry.variantId || record.file !== entry.file || record.localUrl !== entry.localUrl || record.upstreamUrl !== entry.upstreamUrl || record.size !== size || typeof record.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(record.sha256)) throw new Error('cache identity metadata mismatch');
    return record as { sha256: string };
  }

  private async readVerifiedCache(key: string, collect = false) {
    const entry = this.requireVariant(key);
    const filename = this.getVariantFilePath(key), info = await lstat(filename);
    if (!info.isFile() || info.isSymbolicLink() || info.size < MIN_VALID_MODEL_BYTES || info.size > MAX_MODEL_BYTES) throw new Error('invalid cache file/size');
    if (entry.expectedSize !== undefined && info.size !== entry.expectedSize) throw new Error('Pinned model size mismatch');
    const record = await this.readIdentity(key, info.size);
    const hash = createHash('sha256'), chunks: Buffer[] = []; let size = 0;
    for await (const chunk of createReadStream(filename)) {
      size += chunk.length; if (size > (entry.expectedSize ?? MAX_MODEL_BYTES)) throw new Error('cache file exceeds size limit');
      hash.update(chunk); if (collect) chunks.push(chunk);
    }
    const sha256 = hash.digest('hex');
    if (size !== info.size || sha256 !== record.sha256) throw new Error('cache byte hash mismatch');
    if (entry.expectedSha256 !== undefined && sha256 !== entry.expectedSha256) throw new Error('Pinned model SHA256 mismatch');
    return { size, sha256, cachedAt: info.mtime.toISOString(), bytes: collect ? Buffer.concat(chunks) : null };
  }

  public async syncVariantDiskState(key: string): Promise<ContainerModelStatusEntry> {
    this.requireVariant(key);
    const current = this.variantStates.get(key)!;
    if (this.variantInflight.has(key)) return current;
    const file = this.getVariantFilePath(key);
    let updated: ContainerModelStatusEntry;
    try {
      const data = await this.readVerifiedCache(key);
      updated = { ...current, status: 'cached', size: data.size, sha256: data.sha256, filePresent: true, integrity: current.expectedSha256 ? 'pinned' : 'self-recorded', cachedAt: data.cachedAt, error: null };
    } catch (cause) {
      const empty = !existsSync(file) && !existsSync(`${file}.identity.json`);
      updated = { ...current, status: 'unavailable', size: null, sha256: null, filePresent: existsSync(file), integrity: empty ? 'not-present' : 'invalid', cachedAt: null, error: empty ? current.error : `Unverified cache: ${publicCacheError(cause)}` };
    }
    // A validation started before a download must not overwrite newer state.
    if (this.variantStates.get(key) !== current) return this.variantStates.get(key)!;
    this.variantStates.set(key, updated);
    return updated;
  }

  public async prefetchVariant(key: string): Promise<ContainerModelStatusEntry> {
    const entry = this.requireVariant(key);
    const existingInflight = this.variantInflight.get(key);
    if (existingInflight) {
      return existingInflight;
    }

    const diskEntry = await this.syncVariantDiskState(key);
    if (diskEntry.status === 'cached') {
      return diskEntry;
    }

    const existingPromise = this.variantInflight.get(key);
    if (existingPromise) {
      return existingPromise;
    }

    const current = this.variantStates.get(key)!;
    const downloadingState: ContainerModelStatusEntry = {
      ...current,
      status: 'downloading',
      size: null, cachedAt: null, error: null, sha256: null, filePresent: null, integrity: 'pending',
    };
    this.variantStates.set(key, downloadingState);

    const task = (async (): Promise<ContainerModelStatusEntry> => {
      const controller = new AbortController();
      let timer: ReturnType<typeof setTimeout> | undefined;
      let rejectDeadline!: (err: Error) => void;
      const deadline = new Promise<never>((_, reject) => {
        rejectDeadline = reject;
      });
      const resetTimer = () => {
        if (timer !== undefined) clearTimeout(timer);
        timer = setTimeout(() => {
          controller.abort();
          rejectDeadline(new Error('Model cache acquisition timeout'));
        }, PREFETCH_TIMEOUT_MS);
      };
      resetTimer();
      const targetPath = this.getVariantFilePath(key);
      const tempPath = `${targetPath}.part-${randomUUID()}`;
      const metadataPath = `${targetPath}.identity.json`, metadataTemp = `${tempPath}.identity.json`;
      try {
        await mkdir(this.cacheDir, { recursive: true });
        const response = await Promise.race([this.fetchImpl(entry.upstreamUrl, {
          method: 'GET', redirect: 'follow', signal: controller.signal,
        }), deadline]);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const advertised = Number(response.headers.get('Content-Length'));
        if (Number.isFinite(advertised) && advertised > MAX_MODEL_BYTES) throw new Error('Model exceeds cache size limit');
        if (response.headers.has('Content-Length') && entry.expectedSize !== undefined && advertised !== entry.expectedSize) throw new Error('Pinned model size mismatch');
        const reader = response.body?.getReader();
        if (!reader) throw new Error('Empty model response body');
        const hash = createHash('sha256');
        let headChunks: Uint8Array[] = [];
        let fh: FileHandle | undefined;
        let size = 0;
        try {
          while (true) {
            const { done, value } = await Promise.race([reader.read(), deadline]); if (done) break;
            resetTimer();
            size += value.byteLength;
            if (size > (entry.expectedSize ?? MAX_MODEL_BYTES)) { controller.abort(); void reader.cancel().catch(() => {}); throw new Error('Model exceeds cache size limit'); }
            hash.update(value);
            if (!fh) {
              headChunks.push(value);
              if (size >= MIN_VALID_MODEL_BYTES) {
                fh = await open(tempPath, 'w');
                for (const c of headChunks) await fh.write(c);
                headChunks = [];
              }
            } else {
              await fh.write(value);
            }
          }
          if (size < MIN_VALID_MODEL_BYTES) {
            throw new Error(`Downloaded payload too small (${size} bytes)`);
          }
          if (entry.expectedSize !== undefined && size !== entry.expectedSize) throw new Error('Pinned model size mismatch');
          const sha256 = hash.digest('hex');
          if (entry.expectedSha256 !== undefined && sha256 !== entry.expectedSha256) throw new Error('Pinned model SHA256 mismatch');
          if (fh) {
            await fh.close();
            fh = undefined;
          } else {
            await writeFile(tempPath, Buffer.concat(headChunks));
          }
          await writeFile(metadataTemp, JSON.stringify({ schemaVersion: 1, modelId: entry.modelId, variantId: entry.variantId, file: entry.file, localUrl: entry.localUrl, upstreamUrl: entry.upstreamUrl, size, sha256 }));
          await rename(tempPath, targetPath);
          await rename(metadataTemp, metadataPath);

          const cachedState: ContainerModelStatusEntry = {
            ...current,
            status: 'cached',
            size, sha256, filePresent: true, integrity: entry.expectedSha256 ? 'pinned' : 'self-recorded',
            cachedAt: new Date().toISOString(),
            error: null,
          };
          this.variantStates.set(key, cachedState);
          return cachedState;
        } finally {
          if (fh) {
            await fh.close().catch(() => {});
            fh = undefined;
          }
          void reader.cancel().catch(() => {});
          reader.releaseLock();
        }
      } catch (cause) {
        controller.abort();
        await Promise.allSettled([rm(tempPath, { force: true }), rm(metadataTemp, { force: true })]);
        const unavailableState: ContainerModelStatusEntry = {
          ...current,
          status: 'unavailable',
          size: null, cachedAt: null, sha256: null, filePresent: existsSync(targetPath), integrity: existsSync(targetPath) ? 'invalid' : 'not-present',
          error: publicCacheError(cause),
        };
        this.variantStates.set(key, unavailableState);
        return unavailableState;
      } finally {
        if (timer !== undefined) clearTimeout(timer);
        this.variantInflight.delete(key);
      }
    })();

    this.variantInflight.set(key, task);
    return task;
  }

  public async prefetchModel(modelId: RunnableModelType): Promise<ContainerModelStatusEntry> {
    const entry = getVariantCatalogEntry(modelId);
    return this.prefetchVariant(`${modelId}:${entry.variantId}`);
  }

  public async prefetchAllModels(
    options: PrefetchAllModelsOptions = {}
  ): Promise<Record<RunnableModelType, ContainerModelStatusEntry>> {
    const keys = getOrderedCatalogKeys(Boolean(options.prioritizeDefaults));
    const concurrency = options.concurrency;
    if (
      concurrency === undefined &&
      !options.yieldMs &&
      !options.signal &&
      !options.prioritizeDefaults
    ) {
      await Promise.allSettled(keys.map(key => this.prefetchVariant(key)));
      return this.getModelsSnapshot();
    }

    const maxConcurrency = Math.max(1, Math.min(keys.length || 1, concurrency ?? keys.length));
    const yieldMs = Math.max(0, options.yieldMs ?? 0);
    const signal = options.signal;
    let nextIndex = 0;

    const waitYield = (): Promise<void> =>
      new Promise<void>(resolve => {
        if (signal?.aborted) {
          resolve();
          return;
        }
        const onAbort = () => {
          clearTimeout(timer);
          resolve();
        };
        const timer = setTimeout(() => {
          signal?.removeEventListener('abort', onAbort);
          resolve();
        }, yieldMs);
        timer.unref?.();
        signal?.addEventListener('abort', onAbort, { once: true });
      });

    const worker = async (): Promise<void> => {
      while (!signal?.aborted) {
        const currentIndex = nextIndex++;
        if (currentIndex >= keys.length) break;
        const key = keys[currentIndex];
        try {
          await this.prefetchVariant(key);
        } catch {
          // Individual variant errors are recorded inside variantStates
        }
        if (signal?.aborted || nextIndex >= keys.length) break;
        if (yieldMs > 0) {
          await waitYield();
        } else {
          await Promise.resolve();
        }
      }
    };

    await Promise.allSettled(Array.from({ length: maxConcurrency }, () => worker()));
    return this.getModelsSnapshot();
  }

  public getVariantsSnapshot(): Record<string, ContainerModelStatusEntry> {
    return Object.fromEntries([...this.variantStates].map(([key, value]) => [key, { ...value }]));
  }

  public getModelsSnapshot(): Record<RunnableModelType, ContainerModelStatusEntry> {
    return Object.fromEntries(Object.keys(MODEL_ASSET_CATALOG).map(id => {
      const entry = getVariantCatalogEntry(id as RunnableModelType);
      return [id, { ...this.variantStates.get(`${id}:${entry.variantId}`)! }];
    })) as Record<RunnableModelType, ContainerModelStatusEntry>;
  }

  public async getStatusResponse(): Promise<ContainerCacheStatusResponse> {
    await Promise.all(Object.keys(CACHE_CATALOG).map(key => this.syncVariantDiskState(key)));
    return {
      enabled: true, schemaVersion: 2, observedAt: new Date().toISOString(), catalogCount: Object.keys(CACHE_CATALOG).length,
      wasm: {
        mediapipe: { available: this.isWasmAvailable('mediapipe'), localUrl: LOCAL_MEDIAPIPE_WASM_URL, upstreamUrl: UPSTREAM_MEDIAPIPE_WASM_URL, license: 'Apache-2.0' },
        ort: { available: this.isWasmAvailable('ort'), localUrl: LOCAL_ORT_WASM_URL, upstreamUrl: UPSTREAM_ORT_WASM_URL, license: 'MIT' },
      },
      models: this.getModelsSnapshot(), variants: this.getVariantsSnapshot(),
      classification: classifierAssetStatuses(this.getVariantsSnapshot()),
    };
  }

  public createMiddleware() {
    return async (
      req: IncomingMessage,
      res: ServerResponse,
      next: (err?: unknown) => void
    ): Promise<void> => {
      try {
        const rawUrl = req.url ?? '/';
        const pathname = new URL(rawUrl, 'http://localhost').pathname;

        if (
          pathname === '/model-cache.html' &&
          (req.method === 'GET' || req.method === 'HEAD')
        ) {
          const htmlPath = path.join(this.rootDir, 'public', 'model-cache.html');
          if (existsSync(htmlPath)) {
            let html = await readFile(htmlPath, 'utf8');
            const origin = getRequestOrigin(req);
            const status = await this.getStatusResponse();
            const families: Record<string, string> = {
              'selfie-segmenter': 'Selfie Segmenter',
              mediapipe: 'EfficientDet',
              'yolo-seg': 'YOLO',
              segformer: 'SegFormer',
              'image-classifier': 'EfficientNet（画像分類）',
            };
            const stateLabels: Record<string, string> = {
              cached: '配信準備済み',
              downloading: '取得中',
              unavailable: '未準備',
            };
            const integrityLabels: Record<string, string> = {
              pinned: '固定ハッシュ一致',
              'self-recorded': '取得時ハッシュ一致',
              invalid: '検証失敗',
              'not-present': '未保存',
              pending: '検証待ち',
              unknown: '不明',
            };
            const escapeHtml = (str: string) =>
              str
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;');
            const rows = Object.values(status.variants ?? {}).map(v => {
              const cached =
                v.status === 'cached' &&
                v.filePresent === true &&
                (v.integrity === 'pinned' || v.integrity === 'self-recorded') &&
                typeof v.size === 'number' &&
                v.size > 0 &&
                v.error === null;
              const relUrl = v.localUrl;
              const fullUrl = origin ? `${origin}${relUrl}` : relUrl;
              return {
                family: families[v.id] ?? v.id,
                variant: v.variantId ?? '',
                file: v.file,
                size: cached && v.size ? `${v.size.toLocaleString('ja-JP')} bytes` : '—',
                presence: v.filePresent === true ? 'あり' : v.filePresent === false ? 'なし' : '不明',
                readiness: stateLabels[v.status] ?? '不明',
                integrity: integrityLabels[v.integrity ?? 'unknown'] ?? '不明',
                error: v.error ? '取得・検証エラーあり（詳細は管理ログ）' : '—',
                cached,
                relUrl,
                fullUrl,
              };
            });
            const renderTable = (items: typeof rows, emptyText: string, isCachedTable: boolean) => {
              if (!items.length) return `<p>${escapeHtml(emptyText)}</p>`;
              const headers = ['モデル', 'バリアント', 'ファイル名', '検証済みサイズ', '実ファイル', '準備状態', '整合性', '診断'];
              const ths = headers.map(h => `<th scope="col">${escapeHtml(h)}</th>`).join('');
              const trs = items
                .map(r => {
                  const fileCell = isCachedTable
                    ? `<a href="${escapeHtml(r.relUrl)}" class="model-link" download="${escapeHtml(r.file)}">${escapeHtml(r.file)}</a><div class="cache-path"><a href="${escapeHtml(r.fullUrl)}"><code>${escapeHtml(r.fullUrl)}</code></a></div>`
                    : `${escapeHtml(r.file)}<div class="cache-path"><a class="catalog-model-url" href="${escapeHtml(r.fullUrl)}" download="${escapeHtml(r.file)}"><code>${escapeHtml(r.fullUrl)}</code></a></div>`;
                  return `<tr><td data-label="モデル">${escapeHtml(r.family)}</td><td data-label="バリアント">${escapeHtml(r.variant)}</td><td data-label="ファイル名">${fileCell}</td><td data-label="検証済みサイズ">${escapeHtml(r.size)}</td><td data-label="実ファイル">${escapeHtml(r.presence)}</td><td data-label="準備状態">${escapeHtml(r.readiness)}</td><td data-label="整合性">${escapeHtml(r.integrity)}</td><td data-label="診断">${escapeHtml(r.error)}</td></tr>`;
                })
                .join('');
              return `<table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;
            };
            const readyRows = rows.filter(r => r.cached);
            const configuredRows = rows.filter(r => !r.cached);
            html = html.replace(
              '<p id="notice" role="status" aria-live="polite">状態はまだ不明です。</p>',
              `<p id="notice" role="status" aria-live="polite">保存・検証済み ${readyRows.length}件 ／ 設定済み ${rows.length}件</p>`
            );
            html = html.replace(
              '<p id="observed"></p>',
              `<p id="observed">サーバー観測時刻: ${escapeHtml(status.observedAt ?? '')}</p>`
            );
            html = html.replace(
              '<div id="cached"></div>',
              `<div id="cached">${renderTable(readyRows, 'この観測では保存・検証済みファイルはありません。', true)}</div>`
            );
            html = html.replace(
              '<div id="configured"></div>',
              `<div id="configured">${renderTable(configuredRows, '未準備の登録ファイルはありません。', false)}</div>`
            );
            if (origin) {
              html = html.replace(
                '<code id="page-url" class="page-url-text">/model-cache.html</code>',
                `<code id="page-url" class="page-url-text">${escapeHtml(origin)}/model-cache.html</code>`
              );
              html = html.replace(
                /<a class="catalog-model-url" href="(\/api\/model-cache\/models\/[^"]+)" download="([^"]+)"><code>\1<\/code><\/a>/g,
                `<a class="catalog-model-url" href="${escapeHtml(origin)}$1" download="$2"><code>${escapeHtml(origin)}$1</code></a> <span class="cache-path">(パス: <a href="$1"><code>$1</code></a>)</span>`
              );
            }
            res.statusCode = 200;
            res.setHeader('Content-Type', 'text/html; charset=utf-8');
            res.setHeader('Cache-Control', 'no-store');
            res.end(req.method === 'HEAD' ? undefined : html);
            return;
          }
        }

        if (!pathname.startsWith('/api/model-cache')) {
          next();
          return;
        }

        if (req.method !== 'GET' && req.method !== 'HEAD') {
          res.statusCode = 405;
          res.setHeader('Content-Type', 'application/json; charset=utf-8');
          res.end(JSON.stringify({ error: 'Method Not Allowed' }));
          return;
        }

        if (pathname === '/api/model-cache/status') {
          const status = await this.getStatusResponse();
          const body = JSON.stringify(status);
          res.statusCode = 200;
          res.setHeader('Content-Type', 'application/json; charset=utf-8');
          res.setHeader('Cache-Control', 'no-store');
          res.end(req.method === 'HEAD' ? undefined : body);
          return;
        }

        if (pathname.startsWith('/api/model-cache/models/')) {
          const filename = decodeURIComponent(
            pathname.slice('/api/model-cache/models/'.length)
          );
          const variantKey = this.findVariantKeyByFilename(filename);
          if (!variantKey) {
            res.statusCode = 404; res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end(JSON.stringify({ error: 'Unknown model file' })); return;
          }
          const entry = this.requireVariant(variantKey);
          const modelId = entry.modelId;
          if (entry.expectedSha256 && new URL(rawUrl, 'http://localhost').search !== `?sha256=${entry.expectedSha256}`) {
            res.statusCode = 400; res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end(JSON.stringify({ error: 'Pinned asset URL required' })); return;
          }
          const inflight = this.variantInflight.get(variantKey);
          let state = inflight
            ? await inflight
            : await this.syncVariantDiskState(variantKey);
          if (this.variantInflight.has(variantKey)) {
            state = await this.variantInflight.get(variantKey)!;
          } else if (state.status !== 'cached' && state.error === null) {
            state = await this.prefetchVariant(variantKey);
          }

          if (state.status !== 'cached') {
            res.statusCode = 503;
            res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end(
              JSON.stringify({
                error: entry.expectedSha256 ? 'Pinned model unavailable; no upstream fallback permitted' : 'Model not available in container cache; use external fallback',
                model: modelId,
                variantId: entry.variantId,
                upstreamUrl: state.upstreamUrl,
                reason: state.error,
              })
            );
            return;
          }

          let bytes: Buffer;
          try { bytes = (await this.readVerifiedCache(variantKey, true)).bytes!; }
          catch (cause) {
            this.variantStates.set(variantKey, { ...state, status: 'unavailable', size: null, sha256: null, filePresent: existsSync(this.getVariantFilePath(variantKey)), integrity: 'invalid', cachedAt: null, error: `Unverified cache: ${publicCacheError(cause)}` });
            res.statusCode = 503; res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end(JSON.stringify({ error: entry.expectedSha256 ? 'Pinned cached model invalid; no upstream fallback permitted' : 'Cached model identity changed; use requested upstream', model: entry.modelId, variantId: entry.variantId, upstreamUrl: entry.upstreamUrl })); return;
          }
          res.statusCode = 200;
          res.setHeader('Content-Type', 'application/octet-stream');
          res.setHeader('Content-Length', String(bytes.byteLength));
          res.setHeader('Cache-Control', entry.expectedSha256 ? 'no-store' : 'public, max-age=3600');
          if (entry.expectedSha256) res.setHeader('X-Model-SHA256', entry.expectedSha256);
          res.setHeader('X-Model-Source', 'container-cache');
          res.setHeader('X-Model-Id', entry.modelId);
          res.setHeader('X-Model-Variant', entry.variantId);
          res.setHeader('X-Model-License', state.license);
          if (entry.expectedSha256) res.setHeader('Link', '</licenses/Apache-2.0.txt>; rel="license", </licenses/classifier-models.txt>; rel="describedby"');
          res.end(req.method === 'HEAD' ? undefined : bytes);
          return;
        }

        if (
          pathname.startsWith('/api/model-cache/wasm/mediapipe/') ||
          pathname.startsWith('/api/model-cache/wasm/ort/')
        ) {
          const isMediaPipe = pathname.startsWith('/api/model-cache/wasm/mediapipe/');
          const prefix = isMediaPipe
            ? '/api/model-cache/wasm/mediapipe/'
            : '/api/model-cache/wasm/ort/';
          const filename = decodeURIComponent(pathname.slice(prefix.length));
          if (!isSafeFilename(filename)) {
            res.statusCode = 400;
            res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end(JSON.stringify({ error: 'Invalid WASM filename' }));
            return;
          }

          const baseDir = this.getWasmDir(isMediaPipe ? 'mediapipe' : 'ort');
          const filePath = path.resolve(baseDir, filename);
          if (!filePath.startsWith(`${path.resolve(baseDir)}${path.sep}`)) {
            res.statusCode = 403;
            res.end();
            return;
          }

          try {
            const bytes = await readFile(filePath);
            res.statusCode = 200;
            res.setHeader('Content-Type', contentTypeForFile(filename));
            res.setHeader('Content-Length', String(bytes.byteLength));
            res.setHeader('Cache-Control', 'public, max-age=3600');
            res.setHeader('X-Wasm-Source', 'container-cache');
            res.end(req.method === 'HEAD' ? undefined : bytes);
            return;
          } catch {
            res.statusCode = 404;
            res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end(JSON.stringify({ error: 'WASM asset not found in container' }));
            return;
          }
        }

        res.statusCode = 404;
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.end(JSON.stringify({ error: 'Not Found' }));
      } catch (error) {
        next(error);
      }
    };
  }
}

type MinimalHttpServer = {
  listening?: boolean;
  once?: (event: string, listener: () => void) => unknown;
};

export function containerModelCachePlugin(
  options: ContainerModelCacheOptions = {}
): Plugin {
  const manager = new ContainerModelCacheManager(options);
  const defaultDelayMs = options.fetchImpl ? 0 : 5_000;
  const envDelayRaw = process.env.MODEL_CACHE_STARTUP_DELAY_MS;
  const envDelayMs =
    envDelayRaw !== undefined && envDelayRaw.trim() !== ''
      ? Number(envDelayRaw)
      : undefined;
  const startupDelayMs = Math.max(
    0,
    options.startupDelayMs ??
      (envDelayMs !== undefined && Number.isFinite(envDelayMs)
        ? envDelayMs
        : defaultDelayMs)
  );
  const startupConcurrency = Math.max(1, options.startupConcurrency ?? 1);
  const startupYieldMs = Math.max(
    0,
    options.startupYieldMs ?? (options.fetchImpl ? 0 : 200)
  );

  const triggerStartupPrefetch = (
    serverType: string,
    httpServer?: MinimalHttpServer | null
  ) => {
    if (options.autoPrefetchOnStart === false) return;

    console.log(
      `[container-model-cache] Container ${serverType} server started: initiating startup download for all model files...`
    );

    const controller = new AbortController();
    let delayTimer: ReturnType<typeof setTimeout> | undefined;

    const runPrefetch = () => {
      if (controller.signal.aborted) return;
      void manager
        .prefetchAllModels({
          concurrency: startupConcurrency,
          yieldMs: startupYieldMs,
          signal: controller.signal,
          prioritizeDefaults: true,
        })
        .then(() => {
          if (controller.signal.aborted) return;
          const variants = manager.getVariantsSnapshot();
          const cachedCount = Object.values(variants).filter(
            v => v.status === 'cached'
          ).length;
          const totalCount = Object.keys(variants).length;
          console.log(
            `[container-model-cache] Startup download completed: ${cachedCount}/${totalCount} variants cached.`
          );
        })
        .catch(err => {
          if (controller.signal.aborted) return;
          console.warn(
            '[container-model-cache] Startup download encountered an error:',
            err instanceof Error ? err.message : String(err)
          );
        });
    };

    const scheduleAfterListen = () => {
      if (controller.signal.aborted) return;
      if (startupDelayMs <= 0) {
        queueMicrotask(runPrefetch);
        return;
      }
      delayTimer = setTimeout(runPrefetch, startupDelayMs);
      delayTimer.unref?.();
    };

    if (httpServer && typeof httpServer.once === 'function') {
      httpServer.once('close', () => {
        controller.abort();
        if (delayTimer !== undefined) clearTimeout(delayTimer);
      });
      if ('listening' in httpServer && httpServer.listening === false) {
        httpServer.once('listening', scheduleAfterListen);
        return;
      }
    }

    scheduleAfterListen();
  };

  return {
    name: 'container-model-cache-server',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use(manager.createMiddleware());
      triggerStartupPrefetch(
        'dev',
        (server as { httpServer?: MinimalHttpServer | null }).httpServer
      );
    },
    configurePreviewServer(server) {
      const isTest = process.env.TEST_LOG_DIR !== undefined || process.env.CI !== undefined;
      const enabled = process.env.CONTAINER_MODEL_CACHE === 'true' || (!isTest && options.enableInPreview !== false);
      if (enabled) {
        server.middlewares.use(manager.createMiddleware());
        triggerStartupPrefetch(
          'preview',
          (server as { httpServer?: MinimalHttpServer | null }).httpServer
        );
      }
    },
    transformIndexHtml(html) {
      const injection =
        '<script>window.__CONTAINER_MODEL_CACHE_ENABLED__ = true;</script>';
      if (html.includes('</head>')) {
        return html.replace('</head>', `  ${injection}\n</head>`);
      }
      return `${injection}\n${html}`;
    },
  };
}
