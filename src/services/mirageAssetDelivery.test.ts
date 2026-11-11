// Model-free controls for blocked Mirage ONNX generation and delivery.
import { Buffer } from 'node:buffer';
import { spawnSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { generateMirageWebGpuOnnxFile } from '../../scripts/mirage/convertMirageToOnnx.mjs';
import { ContainerModelCacheManager } from './containerModelCacheServer';
import {
  MIRAGE_LOCAL_URL,
  MIRAGE_MODEL_ASSET_ENTRY,
  MIRAGE_UPSTREAM_URL,
  MIRAGE_VARIANT_CATALOG,
  MIRAGE_WEBGPU_LOCAL_URL,
  getVariantCatalogEntry,
  resolveModelAssetUrls,
  setContainerModelCacheEnabledForTest,
} from './modelAssetResolver';
import {
  MIRAGE_WEBGPU_UNAVAILABLE_CODE,
  MIRAGE_WEBGPU_UNAVAILABLE_REASON,
} from './mirageAvailability';
import { buildMirageWebGpuOnnxBytes } from './mirageOnnxConverter';

const unavailable = {
  code: MIRAGE_WEBGPU_UNAVAILABLE_CODE,
  message: MIRAGE_WEBGPU_UNAVAILABLE_REASON,
};

async function request(
  manager: ContainerModelCacheManager,
  method: 'GET' | 'HEAD',
  url = `${MIRAGE_WEBGPU_LOCAL_URL}?sha256=untrusted`,
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
    (error?: unknown) => { throw error ?? new Error('Mirage ONNX route was not handled'); },
  );
  return { status: response.statusCode, headers, body: String(body) };
}

describe('Mirage ONNX fidelity gate at artifact boundaries', () => {
  let directory: string;

  beforeEach(async () => {
    directory = await mkdtemp(path.join(tmpdir(), 'mirage-artifact-gate-'));
  });

  afterEach(async () => {
    setContainerModelCacheEnabledForTest(null);
    vi.restoreAllMocks();
    await rm(directory, { recursive: true, force: true });
  });

  it('cannot generate a replacement graph with no weights or supplied partial weights', () => {
    const fetch = vi.spyOn(globalThis, 'fetch');
    for (const args of [[], [{ stemW0: new Float32Array(3 * 3 * 3 * 32) }]]) {
      expect(() => Reflect.apply(buildMirageWebGpuOnnxBytes, undefined, args)).toThrowError(
        MIRAGE_WEBGPU_UNAVAILABLE_REASON,
      );
    }
    expect(fetch).not.toHaveBeenCalled();
  });

  it('rejects the conversion API without fetching or overwriting an existing artifact', async () => {
    const target = path.join(directory, 'mirage-webgpu.onnx');
    const preserved = Buffer.from('unvalidated artifact retained for inspection');
    await writeFile(target, preserved);
    const fetch = vi.spyOn(globalThis, 'fetch');

    await expect(Reflect.apply(generateMirageWebGpuOnnxFile, undefined, [target])).rejects.toMatchObject(unavailable);

    expect(fetch).not.toHaveBeenCalled();
    expect(await readFile(target)).toEqual(preserved);
    expect(await readdir(directory)).toEqual(['mirage-webgpu.onnx']);
  });

  it.each([false, true])('CLI exits unavailable without creating or replacing an artifact (existing=%s)', async (existing) => {
    const outputDirectory = path.join(directory, '.cache', 'container-models');
    const target = path.join(outputDirectory, 'mirage-webgpu.onnx');
    const preserved = Buffer.from('unvalidated CLI artifact retained for inspection');
    if (existing) {
      await mkdir(outputDirectory, { recursive: true });
      await writeFile(target, preserved);
    }
    const script = path.resolve('scripts/mirage/convertMirageToOnnx.mjs');
    const guard = 'globalThis.fetch = () => { process.stderr.write("UNEXPECTED_MODEL_FETCH"); throw new Error("No network allowed"); };';
    const result = spawnSync(process.execPath, ['--import', `data:text/javascript,${encodeURIComponent(guard)}`, script], {
      cwd: directory,
      encoding: 'utf8',
      timeout: 5000,
    });

    expect(result.error).toBeUndefined();
    expect(result.status).toBe(1);
    expect(result.signal).toBeNull();
    expect(result.stdout).toBe('');
    expect(JSON.parse(result.stderr.trim())).toEqual({
      status: 'unavailable',
      code: MIRAGE_WEBGPU_UNAVAILABLE_CODE,
      error: MIRAGE_WEBGPU_UNAVAILABLE_REASON,
    });
    if (existing) {
      expect(await readFile(target)).toEqual(preserved);
      expect(await readdir(outputDirectory)).toEqual(['mirage-webgpu.onnx']);
    } else {
      expect(await readdir(directory)).toEqual([]);
    }
  });

  for (const existing of [false, true]) {
    it.each(['GET', 'HEAD'] as const)(`HTTP %s cannot serve or generate ONNX (existing=${existing})`, async (method) => {
      const target = path.join(directory, 'mirage-webgpu.onnx');
      const preserved = Buffer.from('unvalidated server artifact retained for inspection');
      if (existing) await writeFile(target, preserved);
      const fetch = vi.fn<typeof globalThis.fetch>();
      const manager = new ContainerModelCacheManager({ cacheDir: directory, fetchImpl: fetch });

      const result = await request(manager, method);

      expect(result.status).toBe(503);
      expect(result.headers['cache-control']).toBe('no-store');
      expect(result.headers['content-type']).toBe('application/json; charset=utf-8');
      if (method === 'HEAD') {
        expect(result.body).toBe('');
      } else {
        expect(JSON.parse(result.body)).toEqual({
          status: 'unavailable',
          code: MIRAGE_WEBGPU_UNAVAILABLE_CODE,
          error: MIRAGE_WEBGPU_UNAVAILABLE_REASON,
          model: 'mirage',
          variantId: 'mirage-webgpu',
        });
      }
      expect(fetch).not.toHaveBeenCalled();
      if (existing) expect(await readFile(target)).toEqual(preserved);
      expect(await readdir(directory)).toEqual(existing ? ['mirage-webgpu.onnx'] : []);
    });
  }

  it.each([false, true])('resolver blocks ONNX before any network lookup (cache=%s)', async (cacheEnabled) => {
    setContainerModelCacheEnabledForTest(cacheEnabled);
    const fetch = vi.spyOn(globalThis, 'fetch');

    await expect(resolveModelAssetUrls('mirage', 'mirage-webgpu')).rejects.toMatchObject(unavailable);

    expect(fetch).not.toHaveBeenCalled();
  });

  it('still serves the cached original archive without fetching a replacement', async () => {
    const archive = Buffer.from('controlled native archive delivery fixture');
    await writeFile(path.join(directory, 'mirage.cact'), archive);
    const fetch = vi.fn<typeof globalThis.fetch>();
    const manager = new ContainerModelCacheManager({ cacheDir: directory, fetchImpl: fetch });

    const result = await request(manager, 'GET', MIRAGE_LOCAL_URL);

    expect(result.status).toBe(200);
    expect(result.headers['x-model-variant']).toBe('mirage-51m');
    expect(result.body).toBe(archive.toString());
    expect(fetch).not.toHaveBeenCalled();
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
