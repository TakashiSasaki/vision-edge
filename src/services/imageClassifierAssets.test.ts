import { afterEach, expect, it, vi } from 'vitest';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { ContainerModelCacheManager } from './containerModelCacheServer';
import { IMAGE_CLASSIFIER_ASSETS, classifierAssetStatuses, resolveClassifierAsset, ClassificationAssetsBlockedError } from './imageClassifierAssets';
const directories: string[] = [];
afterEach(async () => { await Promise.all(directories.splice(0).map(d => rm(d, { recursive: true, force: true }))); vi.restoreAllMocks(); });
async function manager() { const dir = await mkdtemp(path.join(tmpdir(), 'classifier-assets-')); directories.push(dir); const fetch = vi.fn<typeof globalThis.fetch>(async () => new Response(Buffer.alloc(2048, 5))); return { m: new ContainerModelCacheManager({ cacheDir: dir, fetchImpl: fetch }), fetch, dir }; }
async function request(m: ContainerModelCacheManager, url: string) { let body = '', statusCode = 200; const response = { get statusCode() { return statusCode; }, set statusCode(s: number) { statusCode = s; }, setHeader: vi.fn(), end: (chunk: string) => { body = chunk; } }; await m.createMiddleware()({ url, method: 'GET' } as IncomingMessage, response as unknown as ServerResponse, error => { if (error) throw error; }); return { statusCode, body }; }
it.each(['lite0', 'lite2'] as const)('%s official pinned reference has no invented bytes/license/hash or I/O permission', async variant => {
  const fetch = vi.spyOn(globalThis, 'fetch'); const entry = IMAGE_CLASSIFIER_ASSETS[variant];
  expect(entry.upstreamUrl).toBe(`https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_${variant}/float32/1/efficientnet_${variant}.tflite`);
  expect(classifierAssetStatuses()[variant]).toMatchObject({ status: 'blocked', sha256: null, size: null, license: null, localUrl: null, acquisitionEnabled: false });
  await expect(resolveClassifierAsset(variant)).rejects.toBeInstanceOf(ClassificationAssetsBlockedError); expect(fetch).not.toHaveBeenCalled();
});
it('server status publishes the two blocked candidates without changing nine existing variant identities', async () => {
  const { m, fetch } = await manager(), status = await m.getStatusResponse();
  expect(Object.keys(status.variants!)).toHaveLength(9); expect(Object.keys(status.models)).toHaveLength(4);
  expect(status.classification).toEqual(classifierAssetStatuses()); expect(fetch).not.toHaveBeenCalled();
  const http = await request(m, '/api/model-cache/status'); expect(http.statusCode).toBe(200); expect(JSON.parse(http.body).classification).toEqual(status.classification);
});
it('startup prefetch still acquires only the old nine generated fixture assets, never a classifier', async () => {
  const { m, fetch } = await manager(); await m.prefetchAllModels(); expect(fetch).toHaveBeenCalledTimes(9);
  for (const args of fetch.mock.calls) expect(String(args[0])).not.toContain('image_classifier');
  expect((await m.getStatusResponse()).classification).toEqual(classifierAssetStatuses());
});
it.each(['lite0', 'lite2'] as const)('disk bytes cannot enable %s status, direct model serving or prefetch', async variant => {
  const { m, dir, fetch } = await manager(); await writeFile(path.join(dir, IMAGE_CLASSIFIER_ASSETS[variant].file), Buffer.alloc(2048, 99));
  const http = await request(m, '/api/model-cache/models/'+IMAGE_CLASSIFIER_ASSETS[variant].file); expect(http.statusCode).toBe(404);
  await expect(m.prefetchVariant('image-classifier:'+variant)).rejects.toThrow('Unknown model variant');
  expect((await m.getStatusResponse()).classification![variant].status).toBe('blocked'); expect(fetch).not.toHaveBeenCalled();
});
