// Model-free controls for Mirage .cact archive delivery and container cache quota.
import { Buffer } from 'node:buffer';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ContainerModelCacheManager } from './containerModelCacheServer';
import {
  MIRAGE_LOCAL_URL,
  MIRAGE_MODEL_ASSET_ENTRY,
  MIRAGE_UPSTREAM_URL,
  MIRAGE_VARIANT_CATALOG,
  getVariantCatalogEntry,
  resolveModelAssetUrls,
  setContainerModelCacheEnabledForTest,
} from './modelAssetResolver';
import * as mirageReconstructor from './mirageOnnxReconstructor';

async function request(
  manager: ContainerModelCacheManager,
  method: 'GET' | 'HEAD',
  url = MIRAGE_LOCAL_URL,
) {
  const headers: Record<string, string> = {};
  let body: string | Buffer = '';
  const response = {
    statusCode: 200,
    setHeader(name: string, value: string) { headers[name.toLowerCase()] = String(value); },
    end(chunk: string | Buffer = '') { body = chunk; },
  };
  await manager.createMiddleware()(
    { url, method } as IncomingMessage,
    response as unknown as ServerResponse,
    (error?: unknown) => { throw error ?? new Error('Mirage route was not handled'); },
  );
  return { status: response.statusCode, headers, body: String(body) };
}

describe('Mirage asset delivery and cache integrity', () => {
  let directory: string;

  beforeEach(async () => {
    directory = await mkdtemp(path.join(tmpdir(), 'mirage-artifact-gate-'));
  });

  afterEach(async () => {
    setContainerModelCacheEnabledForTest(null);
    vi.restoreAllMocks();
    await rm(directory, { recursive: true, force: true });
  });

  it('serves the cached original archive without fetching a replacement', async () => {
    const archive = Buffer.from('controlled native archive delivery fixture');
    const archiveSha256 = createHash('sha256').update(archive).digest('hex');
    vi.spyOn(mirageReconstructor, 'getMirageCactExpectedPin').mockReturnValue({
      size: archive.byteLength,
      sha256: archiveSha256,
    });
    await writeFile(path.join(directory, 'mirage.cact'), archive);
    const fetch = vi.fn<typeof globalThis.fetch>();
    const manager = new ContainerModelCacheManager({ cacheDir: directory, fetchImpl: fetch });

    const result = await request(manager, 'GET', MIRAGE_LOCAL_URL);

    expect(result.status).toBe(200);
    expect(result.headers['x-model-variant']).toBe('mirage-51m');
    expect(result.headers['x-model-sha256']).toBe(archiveSha256);
    expect(result.body).toBe(archive.toString());
    expect(fetch).not.toHaveBeenCalled();
  });

  it('replaces corrupted mirage.cact when upstream supplies a verified archive and rejects bad downloads without leaving temp files', async () => {
    const validArchive = Buffer.alloc(2048, 0x37);
    const validSha256 = createHash('sha256').update(validArchive).digest('hex');
    vi.spyOn(mirageReconstructor, 'getMirageCactExpectedPin').mockReturnValue({
      size: validArchive.byteLength,
      sha256: validSha256,
    });

    const targetPath = path.join(directory, 'mirage.cact');
    await writeFile(targetPath, Buffer.from('corrupted truncated cact bytes'));

    // First attempt: upstream returns wrong SHA-256 -> must fail with 503 and leave no .part-* files
    const badFetch = vi.fn<typeof globalThis.fetch>(async () =>
      new Response(Buffer.alloc(2048, 0x99), { status: 200 })
    );
    const managerBad = new ContainerModelCacheManager({
      cacheDir: directory,
      fetchImpl: badFetch,
    });
    const badRes = await request(managerBad, 'GET', MIRAGE_LOCAL_URL);
    expect(badRes.status).toBe(503);
    expect(JSON.parse(badRes.body)).toMatchObject({
      reason: 'Pinned model SHA256 mismatch',
    });
    const remainingFiles = await readdir(directory);
    expect(remainingFiles.filter(f => f.includes('.part-'))).toEqual([]);

    // Second attempt: upstream returns verified bytes -> atomically replaces corrupted file and serves 200
    const goodFetch = vi.fn<typeof globalThis.fetch>(async () =>
      new Response(validArchive, {
        status: 200,
        headers: { 'Content-Length': String(validArchive.byteLength) },
      })
    );
    const managerGood = new ContainerModelCacheManager({
      cacheDir: directory,
      fetchImpl: goodFetch,
    });
    const goodRes = await request(managerGood, 'GET', MIRAGE_LOCAL_URL);
    expect(goodRes.status).toBe(200);
    expect(Buffer.from(goodRes.body)).toEqual(validArchive);
    expect(await readFile(targetPath)).toEqual(validArchive);
  });

  it('deduplicates concurrent mirage.cact acquisitions and isolates caller cancellation', async () => {
    const validArchive = Buffer.alloc(2048, 0x52);
    const validSha256 = createHash('sha256').update(validArchive).digest('hex');
    vi.spyOn(mirageReconstructor, 'getMirageCactExpectedPin').mockReturnValue({
      size: validArchive.byteLength,
      sha256: validSha256,
    });

    let releaseUpstream!: () => void;
    const gate = new Promise<void>(r => { releaseUpstream = r; });
    const fetch = vi.fn<typeof globalThis.fetch>(async () => {
      await gate;
      return new Response(validArchive, { status: 200 });
    });

    const manager = new ContainerModelCacheManager({ cacheDir: directory, fetchImpl: fetch });
    const mw = manager.createMiddleware();

    // Caller 1 disconnects before upstream completes
    const caller1Listeners: Record<string, (() => void)[]> = {};
    const caller1Req = {
      url: MIRAGE_LOCAL_URL,
      method: 'GET',
      destroyed: false,
      once(ev: string, fn: () => void) { (caller1Listeners[ev] ??= []).push(fn); },
      removeListener() {},
    } as unknown as IncomingMessage;
    const caller1Res = {
      statusCode: 200,
      destroyed: false,
      writableEnded: false,
      once() {},
      removeListener() {},
      setHeader() {},
      end() {},
    } as unknown as ServerResponse;

    const p1 = mw(caller1Req, caller1Res, () => {});
    const p2 = request(manager, 'GET', MIRAGE_LOCAL_URL);

    await vi.waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));

    // Disconnect caller 1 while shared download is in flight
    (caller1Req as { destroyed: boolean }).destroyed = true;
    for (const fn of caller1Listeners['close'] ?? []) fn();
    await p1;

    // Release upstream: caller 2 still receives the completed verified download
    releaseUpstream();
    const res2 = await p2;
    expect(res2.status).toBe(200);
    expect(Buffer.from(res2.body)).toEqual(validArchive);
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('evicts idle cached files to stay within maxCacheBytes while never deleting files currently in use', async () => {
    const fileA = path.join(directory, 'yolov8n-seg.onnx');
    const fileB = path.join(directory, 'segformer_b0.onnx');
    await writeFile(fileA, Buffer.alloc(2048, 0x11));
    await writeFile(fileB, Buffer.alloc(2048, 0x22));

    const manager = new ContainerModelCacheManager({
      cacheDir: directory,
      maxCacheBytes: 3000,
    });

    // Mark fileA (oldest) as actively being read/streamed
    manager.retainCacheFile(fileA);
    const evictedWhileAInUse = await manager.enforceCacheQuota(0);
    // fileA is protected because it is in use; idle fileB is evicted instead
    expect(evictedWhileAInUse).toEqual(['segformer_b0.onnx']);
    expect(await readdir(directory)).toEqual(['yolov8n-seg.onnx']);

    // Once fileA is released, incoming 2048-byte allocation can evict fileA
    manager.releaseCacheFile(fileA);
    const evictedAfterRelease = await manager.enforceCacheQuota(2048);
    expect(evictedAfterRelease).toEqual(['yolov8n-seg.onnx']);
    expect(await readdir(directory)).toEqual([]);
  });

  it.each([false, true])('keeps the original .cact routing as the default (cache=%s)', async (cacheEnabled) => {
    setContainerModelCacheEnabledForTest(cacheEnabled);
    const fetch = vi.spyOn(globalThis, 'fetch');
    const expected = cacheEnabled
      ? { primaryUrl: MIRAGE_LOCAL_URL, fallbackUrl: MIRAGE_UPSTREAM_URL, source: 'container-cache' }
      : { primaryUrl: MIRAGE_UPSTREAM_URL, fallbackUrl: null, source: 'external' };

    expect(await resolveModelAssetUrls('mirage')).toEqual(expected);
    expect(await resolveModelAssetUrls('mirage', 'mirage-51m')).toEqual(expected);
    expect(getVariantCatalogEntry('mirage')).toMatchObject({ variantId: 'mirage-51m', isDefault: true });
    expect(MIRAGE_MODEL_ASSET_ENTRY.file).toBe('mirage.cact');
    expect(MIRAGE_VARIANT_CATALOG).not.toHaveProperty('mirage:mirage-webgpu');
    expect(fetch).not.toHaveBeenCalled();
  });
});
