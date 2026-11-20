import { createReadStream, existsSync } from 'node:fs';
import { stat } from 'node:fs/promises';
import http, { type IncomingMessage, type ServerResponse } from 'node:http';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  createModelCorsProxyMiddleware,
  type ModelCorsProxyOptions,
} from './src/services/stereoCorsProxyServer.ts';

const MIME_TYPES: Readonly<Record<string, string>> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.wasm': 'application/wasm',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.xml': 'application/xml; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.webmanifest': 'application/manifest+json; charset=utf-8',
};

export interface ProductionServerOptions extends ModelCorsProxyOptions {
  distDir?: string;
}

function getContentType(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase();
  return MIME_TYPES[ext] ?? 'application/octet-stream';
}

function getCacheControl(urlPath: string): string {
  if (urlPath.startsWith('/assets/')) {
    return 'public, max-age=31536000, immutable';
  }
  if (urlPath === '/' || urlPath.endsWith('.html') || urlPath === '/sw.js') {
    return 'no-cache';
  }
  return 'public, max-age=3600';
}

async function sendStaticFile(
  req: IncomingMessage,
  res: ServerResponse,
  filePath: string,
  urlPath: string
): Promise<void> {
  const fileStat = await stat(filePath);
  const totalSize = fileStat.size;
  const contentType = getContentType(filePath);
  const cacheControl = getCacheControl(urlPath);

  res.setHeader('Content-Type', contentType);
  res.setHeader('Cache-Control', cacheControl);
  res.setHeader('Accept-Ranges', 'bytes');

  const rangeHeader = req.headers.range;
  if (rangeHeader && /^bytes=\d*-\d*$/.test(rangeHeader)) {
    const [startStr, endStr] = rangeHeader.slice('bytes='.length).split('-');
    let start = startStr ? Number.parseInt(startStr, 10) : 0;
    let end = endStr ? Number.parseInt(endStr, 10) : totalSize - 1;
    if (!startStr && endStr) {
      const suffixLen = Number.parseInt(endStr, 10);
      start = Math.max(0, totalSize - suffixLen);
      end = totalSize - 1;
    }
    if (
      Number.isNaN(start) ||
      Number.isNaN(end) ||
      start < 0 ||
      end < start ||
      start >= totalSize
    ) {
      res.statusCode = 416;
      res.setHeader('Content-Range', `bytes */${totalSize}`);
      res.end();
      return;
    }
    end = Math.min(end, totalSize - 1);
    const chunkSize = end - start + 1;
    res.statusCode = 206;
    res.setHeader('Content-Range', `bytes ${start}-${end}/${totalSize}`);
    res.setHeader('Content-Length', String(chunkSize));
    if (req.method === 'HEAD') {
      res.end();
      return;
    }
    await new Promise<void>((resolve, reject) => {
      const stream = createReadStream(filePath, { start, end });
      stream.on('error', reject);
      stream.on('end', resolve);
      res.on('close', () => {
        stream.destroy();
        resolve();
      });
      stream.pipe(res);
    });
    return;
  }

  res.statusCode = 200;
  res.setHeader('Content-Length', String(totalSize));
  if (req.method === 'HEAD') {
    res.end();
    return;
  }

  await new Promise<void>((resolve, reject) => {
    const stream = createReadStream(filePath);
    stream.on('error', reject);
    stream.on('end', resolve);
    res.on('close', () => {
      stream.destroy();
      resolve();
    });
    stream.pipe(res);
  });
}

export function createProductionServer(options: ProductionServerOptions = {}): http.Server {
  const rootDir = options.rootDir ?? path.dirname(fileURLToPath(import.meta.url));
  const distDir = path.resolve(options.distDir ?? path.join(rootDir, 'dist'));
  const proxyMiddleware = createModelCorsProxyMiddleware({
    rootDir,
    fetchImpl: options.fetchImpl,
  });

  return http.createServer((req, res) => {
    void (async () => {
      let passedThrough = false;
      await proxyMiddleware(req, res, () => {
        passedThrough = true;
      });
      if (!passedThrough || res.writableEnded || res.destroyed) {
        return;
      }

      if (req.method !== 'GET' && req.method !== 'HEAD') {
        res.statusCode = 405;
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        res.end('Method Not Allowed');
        return;
      }

      let pathname: string;
      try {
        pathname = decodeURIComponent(new URL(req.url ?? '/', 'http://localhost').pathname);
      } catch {
        res.statusCode = 400;
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        res.end('Bad Request');
        return;
      }

      if (pathname.startsWith('/api/')) {
        res.statusCode = 404;
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.end(JSON.stringify({ error: 'Not Found' }));
        return;
      }

      const relPath = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '');
      const candidate = path.resolve(distDir, relPath);
      if (candidate !== distDir && !candidate.startsWith(`${distDir}${path.sep}`)) {
        res.statusCode = 403;
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        res.end('Forbidden');
        return;
      }

      let targetFile: string | null = null;
      if (existsSync(candidate)) {
        const st = await stat(candidate);
        if (st.isFile()) {
          targetFile = candidate;
        } else if (st.isDirectory()) {
          const dirIndex = path.join(candidate, 'index.html');
          if (existsSync(dirIndex)) {
            targetFile = dirIndex;
          }
        }
      }

      // SPA fallback to dist/index.html for client-side routes without file extensions
      if (!targetFile && !path.extname(pathname)) {
        const spaIndex = path.join(distDir, 'index.html');
        if (existsSync(spaIndex)) {
          targetFile = spaIndex;
        }
      }

      if (!targetFile) {
        res.statusCode = 404;
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        res.end('Not Found');
        return;
      }

      await sendStaticFile(req, res, targetFile, pathname);
    })().catch(() => {
      if (!res.headersSent && !res.destroyed && !res.writableEnded) {
        res.statusCode = 500;
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        res.end('Internal Server Error');
      }
    });
  });
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  const port = Number(process.env.PORT) || 3000;
  const host = process.env.HOST || '0.0.0.0';
  const server = createProductionServer();
  server.listen(port, host, () => {
    console.log(`VisionStudio production server listening on http://${host}:${port}`);
  });
}
