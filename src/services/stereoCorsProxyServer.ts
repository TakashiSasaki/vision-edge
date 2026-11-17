import type { IncomingMessage, ServerResponse } from 'node:http';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import type { Plugin } from 'vite';
import {
  MIRAGE_FIXTURE_PREFIX,
  STEREO_CORS_PROXY_PREFIX,
  STEREO_CORS_PROXY_UPSTREAMS,
} from './browserModelCache.ts';

export { MIRAGE_FIXTURE_PREFIX, STEREO_CORS_PROXY_PREFIX, STEREO_CORS_PROXY_UPSTREAMS };

const UPSTREAM_TIMEOUT_MS = 45_000;

export interface ModelCorsProxyOptions {
  rootDir?: string;
  fetchImpl?: typeof fetch;
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

function writeJson(
  res: ServerResponse,
  statusCode: number,
  payload: Record<string, unknown>
): void {
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.end(JSON.stringify(payload));
}

function contentTypeForFixture(filename: string): string {
  if (filename.endsWith('.png')) return 'image/png';
  if (filename.endsWith('.json')) return 'application/json; charset=utf-8';
  return 'application/octet-stream';
}

export function createModelCorsProxyMiddleware(options: ModelCorsProxyOptions = {}) {
  const rootDir = options.rootDir ?? process.cwd();
  const fetchImpl = options.fetchImpl ?? globalThis.fetch;

  return async (
    req: IncomingMessage,
    res: ServerResponse,
    next: (err?: unknown) => void
  ): Promise<void> => {
    const rawUrl = req.url;
    if (!rawUrl) {
      next();
      return;
    }

    let pathname: string;
    try {
      pathname = new URL(rawUrl, 'http://localhost').pathname;
    } catch {
      next();
      return;
    }

    const isStereoProxy = pathname.startsWith(STEREO_CORS_PROXY_PREFIX);
    const isMirageFixture = pathname.startsWith(MIRAGE_FIXTURE_PREFIX);
    if (!isStereoProxy && !isMirageFixture) {
      next();
      return;
    }

    if (req.method !== 'GET' && req.method !== 'HEAD') {
      writeJson(res, 405, { error: 'Method Not Allowed' });
      return;
    }

    if (isMirageFixture) {
      let filename: string;
      try {
        filename = decodeURIComponent(pathname.slice(MIRAGE_FIXTURE_PREFIX.length));
      } catch {
        writeJson(res, 400, { error: 'Invalid fixture filename' });
        return;
      }
      if (!isSafeFilename(filename)) {
        writeJson(res, 400, { error: 'Invalid fixture filename' });
        return;
      }

      const candidateDirs = [
        path.resolve(rootDir, 'src', 'assets', 'mirage-reconstruction', 'presets'),
        path.resolve(rootDir, 'scripts', 'mirage', 'presets'),
      ];
      let targetPath: string | null = null;
      for (const dir of candidateDirs) {
        const candidate = path.resolve(dir, filename);
        if (!candidate.startsWith(`${dir}${path.sep}`)) {
          writeJson(res, 403, { error: 'Forbidden fixture path' });
          return;
        }
        if (existsSync(candidate)) {
          targetPath = candidate;
          break;
        }
      }

      if (!targetPath) {
        writeJson(res, 404, { error: 'Fixture not found' });
        return;
      }

      try {
        const bytes = await readFile(targetPath);
        res.statusCode = 200;
        res.setHeader('Content-Type', contentTypeForFixture(filename));
        res.setHeader('Content-Length', String(bytes.byteLength));
        res.setHeader('Cache-Control', 'no-store');
        res.end(req.method === 'HEAD' ? undefined : bytes);
      } catch {
        writeJson(res, 404, { error: 'Fixture not found' });
      }
      return;
    }

    // Stateless CORS pass-through proxy for allowlisted GitHub Releases stereo ONNX models
    let filename: string;
    try {
      filename = decodeURIComponent(pathname.slice(STEREO_CORS_PROXY_PREFIX.length));
    } catch {
      writeJson(res, 400, { error: 'Invalid stereo model filename' });
      return;
    }

    if (!isSafeFilename(filename)) {
      writeJson(res, 400, { error: 'Invalid stereo model filename' });
      return;
    }

    const upstreamUrl = Object.hasOwn(STEREO_CORS_PROXY_UPSTREAMS, filename)
      ? STEREO_CORS_PROXY_UPSTREAMS[filename]
      : undefined;
    if (!upstreamUrl) {
      writeJson(res, 404, { error: 'Unknown stereo model file' });
      return;
    }

    const upstreamHost = new URL(upstreamUrl).hostname;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
    const onDisconnect = () => controller.abort();
    req.once?.('aborted', onDisconnect);
    res.once?.('close', onDisconnect);

    try {
      const upstreamRes = await fetchImpl(upstreamUrl, {
        method: req.method,
        redirect: 'follow',
        signal: controller.signal,
      });

      if (!upstreamRes.ok) {
        await upstreamRes.body?.cancel?.().catch(() => {});
        writeJson(res, 502, {
          error: 'Upstream stereo model fetch failed',
          upstreamStatus: upstreamRes.status,
          upstreamHost,
        });
        return;
      }

      const contentType = upstreamRes.headers.get('content-type')?.toLowerCase() ?? '';
      if (contentType.startsWith('text/html')) {
        await upstreamRes.body?.cancel?.().catch(() => {});
        writeJson(res, 502, {
          error: 'Upstream returned unexpected HTML payload instead of ONNX binary',
          upstreamHost,
        });
        return;
      }

      res.statusCode = 200;
      res.setHeader('Content-Type', 'application/octet-stream');
      res.setHeader('Cache-Control', 'public, max-age=86400');
      res.setHeader('X-Proxy-Cache', 'stateless-stream');
      res.setHeader('X-Upstream-Host', upstreamHost);

      const upstreamLen = upstreamRes.headers.get('content-length');
      if (upstreamLen && /^\d+$/.test(upstreamLen)) {
        res.setHeader('Content-Length', upstreamLen);
      }

      if (req.method === 'HEAD') {
        await upstreamRes.body?.cancel?.().catch(() => {});
        res.end();
        return;
      }

      if (!upstreamRes.body || typeof upstreamRes.body.getReader !== 'function') {
        const bytes = Buffer.from(await upstreamRes.arrayBuffer());
        res.setHeader('Content-Length', String(bytes.byteLength));
        res.end(bytes);
        return;
      }

      const reader = upstreamRes.body.getReader();
      try {
        while (true) {
          if (req.destroyed || res.destroyed || res.writableEnded) {
            await reader.cancel().catch(() => {});
            break;
          }
          const { done, value } = await reader.read();
          if (done) break;
          if (value && value.byteLength > 0) {
            const canContinue = (
              res as ServerResponse & { write?: (chunk: Uint8Array) => boolean }
            ).write?.(value);
            if (canContinue === false) {
              await new Promise<void>((resolve) => {
                res.once?.('drain', resolve);
                res.once?.('close', resolve);
              });
            }
          }
        }
        if (!res.destroyed && !res.writableEnded) {
          res.end();
        }
      } finally {
        reader.releaseLock?.();
      }
    } catch (err) {
      if (!res.headersSent && !res.destroyed && !res.writableEnded) {
        writeJson(res, 502, {
          error: 'Stateless stereo CORS proxy failed',
          reason: err instanceof Error ? err.message : 'Upstream fetch error',
        });
      }
    } finally {
      clearTimeout(timeout);
      req.removeListener?.('aborted', onDisconnect);
      res.removeListener?.('close', onDisconnect);
    }
  };
}

export function modelCorsProxyPlugin(options: ModelCorsProxyOptions = {}): Plugin {
  const middleware = createModelCorsProxyMiddleware(options);
  return {
    name: 'stereo-model-cors-proxy',
    configureServer(server) {
      server.middlewares.use(middleware);
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware);
    },
  };
}
