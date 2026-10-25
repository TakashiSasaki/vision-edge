import { Buffer } from 'node:buffer';
import { existsSync } from 'node:fs';
import { mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import type { IncomingMessage, ServerResponse } from 'node:http';
import path from 'node:path';
import process from 'node:process';
import type { Plugin } from 'vite';
import type { RunnableModelType } from '../types/vision.ts';
import {
  LOCAL_MEDIAPIPE_WASM_URL,
  LOCAL_ORT_WASM_URL,
  MODEL_ASSET_CATALOG,
  UPSTREAM_MEDIAPIPE_WASM_URL,
  UPSTREAM_ORT_WASM_URL,
  type ContainerCacheStatusResponse,
  type ContainerModelStatusEntry,
} from './modelAssetResolver.ts';

const MIN_VALID_MODEL_BYTES = 1024;
const PREFETCH_TIMEOUT_MS = 45_000;

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
  private readonly states = new Map<RunnableModelType, ContainerModelStatusEntry>();
  private readonly inflight = new Map<RunnableModelType, Promise<ContainerModelStatusEntry>>();

  constructor(options: ContainerModelCacheOptions = {}) {
    this.rootDir = options.rootDir ?? process.cwd();
    this.cacheDir =
      options.cacheDir ?? path.join(this.rootDir, '.cache', 'container-models');
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch;

    for (const modelId of Object.keys(MODEL_ASSET_CATALOG) as RunnableModelType[]) {
      const entry = MODEL_ASSET_CATALOG[modelId];
      this.states.set(modelId, {
        id: modelId,
        file: entry.file,
        status: 'unavailable',
        size: null,
        cachedAt: null,
        error: null,
        localUrl: entry.localUrl,
        upstreamUrl: entry.upstreamUrl,
        license: entry.license,
      });
    }
  }

  public getModelFilePath(modelId: RunnableModelType): string {
    const entry = MODEL_ASSET_CATALOG[modelId];
    return path.join(this.cacheDir, entry.file);
  }

  public findModelByFilename(filename: string): RunnableModelType | null {
    if (!isSafeFilename(filename)) return null;
    for (const modelId of Object.keys(MODEL_ASSET_CATALOG) as RunnableModelType[]) {
      if (MODEL_ASSET_CATALOG[modelId].file === filename) {
        return modelId;
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

  public async syncDiskState(modelId: RunnableModelType): Promise<ContainerModelStatusEntry> {
    const current = this.states.get(modelId)!;
    const filePath = this.getModelFilePath(modelId);
    try {
      const info = await stat(filePath);
      if (info.isFile() && info.size >= MIN_VALID_MODEL_BYTES) {
        const updated: ContainerModelStatusEntry = {
          ...current,
          status: 'cached',
          size: info.size,
          cachedAt: info.mtime.toISOString(),
          error: null,
        };
        this.states.set(modelId, updated);
        return updated;
      }
    } catch {
      // File not on disk yet
    }
    return current;
  }

  public async prefetchModel(modelId: RunnableModelType): Promise<ContainerModelStatusEntry> {
    const diskEntry = await this.syncDiskState(modelId);
    if (diskEntry.status === 'cached') {
      return diskEntry;
    }

    const existingPromise = this.inflight.get(modelId);
    if (existingPromise) {
      return existingPromise;
    }

    const entry = MODEL_ASSET_CATALOG[modelId];
    const current = this.states.get(modelId)!;
    const downloadingState: ContainerModelStatusEntry = {
      ...current,
      status: 'downloading',
      error: null,
    };
    this.states.set(modelId, downloadingState);

    const task = (async (): Promise<ContainerModelStatusEntry> => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), PREFETCH_TIMEOUT_MS);
      const targetPath = this.getModelFilePath(modelId);
      const tempPath = `${targetPath}.part-${process.pid}-${Date.now()}`;
      try {
        await mkdir(this.cacheDir, { recursive: true });
        const response = await this.fetchImpl(entry.upstreamUrl, {
          method: 'GET',
          redirect: 'follow',
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status} from ${entry.upstreamUrl}`);
        }
        const arrayBuffer = await response.arrayBuffer();
        const bytes = Buffer.from(arrayBuffer);
        if (bytes.byteLength < MIN_VALID_MODEL_BYTES) {
          throw new Error(`Downloaded payload too small (${bytes.byteLength} bytes)`);
        }
        await writeFile(tempPath, bytes);
        await rename(tempPath, targetPath);

        const cachedState: ContainerModelStatusEntry = {
          ...current,
          status: 'cached',
          size: bytes.byteLength,
          cachedAt: new Date().toISOString(),
          error: null,
        };
        this.states.set(modelId, cachedState);
        return cachedState;
      } catch (cause) {
        await rm(tempPath, { force: true }).catch(() => {});
        const unavailableState: ContainerModelStatusEntry = {
          ...current,
          status: 'unavailable',
          size: null,
          error: cause instanceof Error ? cause.message : String(cause),
        };
        this.states.set(modelId, unavailableState);
        return unavailableState;
      } finally {
        clearTimeout(timer);
        this.inflight.delete(modelId);
      }
    })();

    this.inflight.set(modelId, task);
    return task;
  }

  public async prefetchAllModels(): Promise<Record<RunnableModelType, ContainerModelStatusEntry>> {
    const modelIds = Object.keys(MODEL_ASSET_CATALOG) as RunnableModelType[];
    await Promise.allSettled(modelIds.map(id => this.prefetchModel(id)));
    return this.getModelsSnapshot();
  }

  public getModelsSnapshot(): Record<RunnableModelType, ContainerModelStatusEntry> {
    return {
      'selfie-segmenter': { ...this.states.get('selfie-segmenter')! },
      mediapipe: { ...this.states.get('mediapipe')! },
      'yolo-seg': { ...this.states.get('yolo-seg')! },
      segformer: { ...this.states.get('segformer')! },
    };
  }

  public async getStatusResponse(): Promise<ContainerCacheStatusResponse> {
    const modelIds = Object.keys(MODEL_ASSET_CATALOG) as RunnableModelType[];
    await Promise.all(
      modelIds.map(async id => {
        if (!this.inflight.has(id)) {
          await this.syncDiskState(id);
        }
      })
    );
    return {
      enabled: true,
      wasm: {
        mediapipe: {
          available: this.isWasmAvailable('mediapipe'),
          localUrl: LOCAL_MEDIAPIPE_WASM_URL,
          upstreamUrl: UPSTREAM_MEDIAPIPE_WASM_URL,
          license: 'Apache-2.0',
        },
        ort: {
          available: this.isWasmAvailable('ort'),
          localUrl: LOCAL_ORT_WASM_URL,
          upstreamUrl: UPSTREAM_ORT_WASM_URL,
          license: 'MIT',
        },
      },
      models: this.getModelsSnapshot(),
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
          const modelId = this.findModelByFilename(filename);
          if (!modelId) {
            res.statusCode = 404;
            res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end(JSON.stringify({ error: 'Unknown model file' }));
            return;
          }

          let state = await this.syncDiskState(modelId);
          if (this.inflight.has(modelId)) {
            state = await this.inflight.get(modelId)!;
          } else if (state.status !== 'cached' && state.error === null) {
            state = await this.prefetchModel(modelId);
          }

          if (state.status !== 'cached') {
            res.statusCode = 503;
            res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end(
              JSON.stringify({
                error: 'Model not available in container cache; use external fallback',
                model: modelId,
                upstreamUrl: state.upstreamUrl,
                reason: state.error,
              })
            );
            return;
          }

          const filePath = this.getModelFilePath(modelId);
          const bytes = await readFile(filePath);
          res.statusCode = 200;
          res.setHeader('Content-Type', 'application/octet-stream');
          res.setHeader('Content-Length', String(bytes.byteLength));
          res.setHeader('Cache-Control', 'public, max-age=3600');
          res.setHeader('X-Model-Source', 'container-cache');
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
