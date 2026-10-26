import { Buffer } from 'node:buffer';
import { existsSync, createReadStream } from 'node:fs';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rm, lstat, writeFile } from 'node:fs/promises';
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
  type ContainerModelStatusEntry,
  getVariantCatalogEntry,
} from './modelAssetResolver.ts';

const MIN_VALID_MODEL_BYTES = 1024;
const PREFETCH_TIMEOUT_MS = 45_000;
const MAX_MODEL_BYTES = 512 * 1024 * 1024;

export interface ContainerModelCacheOptions {
  rootDir?: string;
  cacheDir?: string;
  fetchImpl?: typeof fetch;
  autoPrefetchOnStart?: boolean;
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

    for (const key of Object.keys(MODEL_VARIANT_CATALOG)) {
      const entry = MODEL_VARIANT_CATALOG[key];
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

  public findModelByFilename(filename: string): RunnableModelType | null {
    if (!isSafeFilename(filename)) return null;
    for (const key of Object.keys(MODEL_VARIANT_CATALOG)) {
      if (MODEL_VARIANT_CATALOG[key].file === filename) {
        return MODEL_VARIANT_CATALOG[key].modelId;
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
    for (const key of Object.keys(MODEL_VARIANT_CATALOG)) {
      if (MODEL_VARIANT_CATALOG[key].file === filename) {
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
    if (!Object.hasOwn(MODEL_VARIANT_CATALOG, key)) throw new RangeError(`Unknown model variant: ${key}`);
    return MODEL_VARIANT_CATALOG[key];
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
    const filename = this.getVariantFilePath(key), info = await lstat(filename);
    if (!info.isFile() || info.isSymbolicLink() || info.size < MIN_VALID_MODEL_BYTES || info.size > MAX_MODEL_BYTES) throw new Error('invalid cache file/size');
    const record = await this.readIdentity(key, info.size);
    const hash = createHash('sha256'), chunks: Buffer[] = []; let size = 0;
    for await (const chunk of createReadStream(filename)) {
      size += chunk.length; if (size > MAX_MODEL_BYTES) throw new Error('cache file exceeds size limit');
      hash.update(chunk); if (collect) chunks.push(chunk);
    }
    if (size !== info.size || hash.digest('hex') !== record.sha256) throw new Error('cache byte hash mismatch');
    return { size, cachedAt: info.mtime.toISOString(), bytes: collect ? Buffer.concat(chunks) : null };
  }

  public async syncVariantDiskState(key: string): Promise<ContainerModelStatusEntry> {
    this.requireVariant(key);
    const current = this.variantStates.get(key)!;
    if (this.variantInflight.has(key)) return current;
    const file = this.getVariantFilePath(key);
    let updated: ContainerModelStatusEntry;
    try {
      const data = await this.readVerifiedCache(key);
      updated = { ...current, status: 'cached', size: data.size, cachedAt: data.cachedAt, error: null };
    } catch (cause) {
      const empty = !existsSync(file) && !existsSync(`${file}.identity.json`);
      updated = { ...current, status: 'unavailable', size: null, cachedAt: null, error: empty ? current.error ?? (current.status === 'cached' ? 'Cached files disappeared' : null) : `Unverified cache: ${cause instanceof Error ? cause.message : String(cause)}` };
    }
    // A validation started before a download must not overwrite newer state.
    if (this.variantStates.get(key) !== current) return this.variantStates.get(key)!;
    this.variantStates.set(key, updated);
    return updated;
  }

  public async prefetchVariant(key: string): Promise<ContainerModelStatusEntry> {
    const entry = this.requireVariant(key);
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
      size: null, cachedAt: null, error: null,
    };
    this.variantStates.set(key, downloadingState);

    const task = (async (): Promise<ContainerModelStatusEntry> => {
      const controller = new AbortController();
      let timer: ReturnType<typeof setTimeout>;
      const deadline = new Promise<never>((_, reject) => { timer = setTimeout(() => { controller.abort(); reject(new Error('Model cache acquisition timeout')); }, PREFETCH_TIMEOUT_MS); });
      const targetPath = this.getVariantFilePath(key);
      const tempPath = `${targetPath}.part-${randomUUID()}`;
      const metadataPath = `${targetPath}.identity.json`, metadataTemp = `${tempPath}.identity.json`;
      try {
        await mkdir(this.cacheDir, { recursive: true });
        const response = await Promise.race([this.fetchImpl(entry.upstreamUrl, {
          method: 'GET', redirect: 'follow', signal: controller.signal,
        }), deadline]);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status} from ${entry.upstreamUrl}`);
        }
        const advertised = Number(response.headers.get('Content-Length'));
        if (Number.isFinite(advertised) && advertised > MAX_MODEL_BYTES) throw new Error('Model exceeds cache size limit');
        const reader = response.body?.getReader();
        if (!reader) throw new Error('Empty model response body');
        const chunks: Uint8Array[] = []; let size = 0;
        try {
          while (true) {
            const { done, value } = await Promise.race([reader.read(), deadline]); if (done) break;
            size += value.byteLength;
            if (size > MAX_MODEL_BYTES) { controller.abort(); void reader.cancel().catch(() => {}); throw new Error('Model exceeds cache size limit'); }
            chunks.push(value);
          }
        } finally { void reader.cancel().catch(() => {}); reader.releaseLock(); }
        const bytes = Buffer.concat(chunks);
        if (bytes.byteLength < MIN_VALID_MODEL_BYTES) {
          throw new Error(`Downloaded payload too small (${bytes.byteLength} bytes)`);
        }
        await writeFile(tempPath, bytes);
        await writeFile(metadataTemp, JSON.stringify({ schemaVersion: 1, modelId: entry.modelId, variantId: entry.variantId, file: entry.file, localUrl: entry.localUrl, upstreamUrl: entry.upstreamUrl, size: bytes.byteLength, sha256: createHash('sha256').update(bytes).digest('hex') }));
        await rename(tempPath, targetPath);
        await rename(metadataTemp, metadataPath);

        const cachedState: ContainerModelStatusEntry = {
          ...current,
          status: 'cached',
          size: bytes.byteLength,
          cachedAt: new Date().toISOString(),
          error: null,
        };
        this.variantStates.set(key, cachedState);
        return cachedState;
      } catch (cause) {
        await Promise.allSettled([rm(tempPath, { force: true }), rm(metadataTemp, { force: true })]);
        const unavailableState: ContainerModelStatusEntry = {
          ...current,
          status: 'unavailable',
          size: null, cachedAt: null,
          error: cause instanceof Error ? cause.message : String(cause),
        };
        this.variantStates.set(key, unavailableState);
        return unavailableState;
      } finally {
        clearTimeout(timer!);
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

  public async prefetchAllModels(): Promise<Record<RunnableModelType, ContainerModelStatusEntry>> {
    await Promise.allSettled(Object.keys(MODEL_VARIANT_CATALOG).map(key => this.prefetchVariant(key)));
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
    await Promise.all(Object.keys(MODEL_VARIANT_CATALOG).map(key => this.syncVariantDiskState(key)));
    return {
      enabled: true,
      wasm: {
        mediapipe: { available: this.isWasmAvailable('mediapipe'), localUrl: LOCAL_MEDIAPIPE_WASM_URL, upstreamUrl: UPSTREAM_MEDIAPIPE_WASM_URL, license: 'Apache-2.0' },
        ort: { available: this.isWasmAvailable('ort'), localUrl: LOCAL_ORT_WASM_URL, upstreamUrl: UPSTREAM_ORT_WASM_URL, license: 'MIT' },
      },
      models: this.getModelsSnapshot(), variants: this.getVariantsSnapshot(),
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
          let state = await this.syncVariantDiskState(variantKey);
          if (this.variantInflight.has(variantKey)) state = await this.variantInflight.get(variantKey)!;
          else if (state.status !== 'cached' && state.error === null) state = await this.prefetchVariant(variantKey);

          if (state.status !== 'cached') {
            res.statusCode = 503;
            res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end(
              JSON.stringify({
                error: 'Model not available in container cache; use external fallback',
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
            this.variantStates.set(variantKey, { ...state, status: 'unavailable', size: null, cachedAt: null, error: `Unverified cache: ${String(cause)}` });
            res.statusCode = 503; res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end(JSON.stringify({ error: 'Cached model identity changed; use requested upstream', model: entry.modelId, variantId: entry.variantId, upstreamUrl: entry.upstreamUrl })); return;
          }
          res.statusCode = 200;
          res.setHeader('Content-Type', 'application/octet-stream');
          res.setHeader('Content-Length', String(bytes.byteLength));
          res.setHeader('Cache-Control', 'public, max-age=3600');
          res.setHeader('X-Model-Source', 'container-cache');
          res.setHeader('X-Model-Id', entry.modelId);
          res.setHeader('X-Model-Variant', entry.variantId);
          res.setHeader('X-Model-License', state.license);
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

export function containerModelCachePlugin(
  options: ContainerModelCacheOptions = {}
): Plugin {
  const manager = new ContainerModelCacheManager(options);
  return {
    name: 'container-model-cache-server',
    apply: 'serve',
    configureServer(server) {
      if (options.autoPrefetchOnStart !== false) {
        void manager.prefetchAllModels();
      }
      server.middlewares.use(manager.createMiddleware());
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
