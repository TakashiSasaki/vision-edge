import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { describe, expect, it, vi } from 'vitest';
import * as ort from 'onnxruntime-web';
import {
  captureMirageSourceFrameSnapshot,
  compareMirageTensorParity,
  computeBoxIou,
  computeMirageCeil32Metadata,
  convertProposalsToDetectedObjects,
  createMirageExperimentalOnnxTask,
  decodeMirageExperimentalOutputs,
  MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256,
  MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE,
  MIRAGE_EXPERIMENTAL_ONNX_FILE,
  MIRAGE_EXPERIMENTAL_ONNX_FINAL_OUTPUTS,
  MirageExperimentalOnnxEngine,
  fetchModelBuffer,
  packResizedPixelsToMirageTensor,
  probeWebGpuAdapterDiagnostics,
  sha256HexSync,
  verifyMirageExperimentalOnnxBytes,
  type MirageCeil32PreprocessMetadata,
  type MirageSourceFrameSnapshot,
} from './mirageExperimentalOnnxService';
import {
  ensureReconstructedMirageOnnxFile,
  hasMirageReconstructionAssets,
  MIRAGE_CACT_EXPECTED_SHA256,
  MIRAGE_CACT_EXPECTED_SIZE,
  MIRAGE_PACKAGE_ROOT,
  MIRAGE_RECONSTRUCTION_ASSETS_PIN,
  resolveMirageReconstructionAssetPaths,
} from './mirageOnnxReconstructor';
import { MirageWasmEngine } from './mirageWasmService';
import { ContainerModelCacheManager } from './containerModelCacheServer';
import {
  getVariantCatalogEntry,
  MIRAGE_EXPERIMENTAL_ONNX_LOCAL_URL,
  MODEL_VARIANT_CATALOG,
} from './modelAssetResolver';
import { backendDescription, MIRAGE_MODEL } from './modelRegistry';
import { VisionEngine } from './visionEngine';

const FIXTURES_DIR = path.resolve('.cache/mirage-browser-fixtures');
const MANIFEST_PATH = path.join(FIXTURES_DIR, 'manifest.json');

interface BrowserFixtureProposal {
  cell: [number, number];
  score: number;
  bbox: [number, number, number, number];
  mask_active_pixels?: number;
  mask_sha256?: string;
  mask_packed_base64?: string;
}

interface BrowserFixtureCase {
  name: string;
  group: string;
  valid_hw: [number, number];
  metadata: MirageCeil32PreprocessMetadata;
  heads: Record<string, { file: string; shape: number[]; sha256: string }>;
  decoded_default: {
    object_count: number;
    text_count: number;
    objects: BrowserFixtureProposal[];
    text: BrowserFixtureProposal[];
  };
  decoded_nms_0p50: {
    object_count: number;
    text_count: number;
    objects: BrowserFixtureProposal[];
    text: BrowserFixtureProposal[];
  };
}

interface BrowserFixtureManifest {
  schema_version: number;
  onnx: { size: number; sha256: string };
  cases: Record<string, BrowserFixtureCase>;
}

function invokeMiddleware(
  middleware: ReturnType<ContainerModelCacheManager['createMiddleware']>,
  url: string,
  method = 'GET'
): Promise<{ status: number; headers: Record<string, string>; body: Buffer }> {
  return new Promise((resolve, reject) => {
    const headers: Record<string, string> = {};
    const chunks: Buffer[] = [];
    const req = { url, method, headers: { host: 'localhost:3000' } } as unknown as IncomingMessage;
    const rawRes = {
      statusCode: 200,
      setHeader(name: string, value: string) {
        headers[name.toLowerCase()] = String(value);
      },
      end(chunk?: Buffer | string) {
        if (chunk !== undefined) {
          chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk)));
        }
        resolve({
          status: rawRes.statusCode,
          headers,
          body: Buffer.concat(chunks),
        });
      },
    };
    const res = rawRes as unknown as ServerResponse;
    middleware(req, res, (err) => {
      if (err) reject(err);
      else resolve({ status: 404, headers, body: Buffer.alloc(0) });
    });
  });
}

describe('Experimental Partial Mirage ONNX Integration', () => {
  it('preserves existing MODEL_VARIANT_CATALOG count while registering experimental Mirage ONNX variants', () => {
    // Existing 10 detection/segmentation variants in MODEL_VARIANT_CATALOG remain untouched
    expect(Object.keys(MODEL_VARIANT_CATALOG)).toHaveLength(10);

    const wasmEntry = getVariantCatalogEntry('mirage', 'mirage-onnx-exp-wasm');
    expect(wasmEntry.file).toBe(MIRAGE_EXPERIMENTAL_ONNX_FILE);
    expect(wasmEntry.expectedSize).toBe(MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE);
    expect(wasmEntry.expectedSha256).toBe(MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256);
    expect(wasmEntry.localUrl).toBe(MIRAGE_EXPERIMENTAL_ONNX_LOCAL_URL);

    const webgpuEntry = getVariantCatalogEntry('mirage', 'mirage-onnx-exp-webgpu');
    expect(webgpuEntry.file).toBe(MIRAGE_EXPERIMENTAL_ONNX_FILE);
    expect(webgpuEntry.expectedSize).toBe(MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE);
    expect(webgpuEntry.expectedSha256).toBe(MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256);

    // Default Mirage variant remains mirage-51m (.cact WASM)
    expect(MIRAGE_MODEL.defaultVariantId).toBe('mirage-51m');
    expect(getVariantCatalogEntry('mirage').variantId).toBe('mirage-51m');

    // mirage-webgpu remains blocked with unvalidated error
    expect(() => getVariantCatalogEntry('mirage', 'mirage-webgpu')).toThrow(
      /Mirage ONNX is unavailable|not been completed/i
    );

    // Backend descriptions distinguish official WASM, blocked mirage-webgpu, and experimental partial ONNX
    expect(
      backendDescription({
        status: 'ready',
        model: 'mirage',
        variantId: 'mirage-onnx-exp-wasm',
        requestId: 1,
        backend: {
          runtime: 'onnxruntime-web',
          requestedProvider: 'wasm',
          configuredProviders: ['wasm'],
          fallbackReason: null,
          actualBackend: 'unmeasured',
        },
      })
    ).toContain('実験用部分ONNX 要求: WASM／設定: WASM');
  });

  it('reproduces ceil32 preprocessing metadata, top-left placement, and zero padding', () => {
    const meta = computeMirageCeil32Metadata(640, 427);
    expect(meta).toEqual({
      W: 640,
      H: 427,
      w: 512,
      h: 352,
      r: 0.8,
      coordinate_scale: [1.0, 512 / 352, 1.0, 512 / 352],
      resize_rule: 'ceil32',
    });

    const rgba = new Uint8ClampedArray(meta.w * meta.h * 4);
    // Set top-left pixel (0, 0) to RGB (255, 127, 0)
    rgba[0] = 255;
    rgba[1] = 127;
    rgba[2] = 0;
    rgba[3] = 255;

    const packed = packResizedPixelsToMirageTensor(rgba, 4, meta);
    expect(packed.valid_hw).toEqual([352, 512]);
    expect(packed.image).toHaveLength(3 * 512 * 512);
    const plane = 512 * 512;
    expect(packed.image[0]).toBeCloseTo(255 / 127.5 - 1.0, 5);
    expect(packed.image[plane]).toBeCloseTo(127 / 127.5 - 1.0, 5);
    expect(packed.image[2 * plane]).toBeCloseTo(-1.0, 5);

    // Padded rows >= 352 must be strictly 0.0
    for (let c = 0; c < 3; c++) {
      expect(packed.image[c * plane + 352 * 512]).toBe(0.0);
      expect(packed.image[c * plane + 511 * 512 + 511]).toBe(0.0);
    }
  });

  it.skipIf(process.env.REQUIRE_MIRAGE_REAL_MODEL !== '1' && !existsSync(MANIFEST_PATH))(
    'matches Python perception_diagnostics.py decode and corpus_postprocessing.py decode_policy across all 8 verified fixtures',
    () => {
      expect(existsSync(MANIFEST_PATH)).toBe(true);
      const manifest = JSON.parse(
        readFileSync(MANIFEST_PATH, 'utf8')
      ) as BrowserFixtureManifest;

      expect(manifest.onnx.size).toBe(MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE);
      expect(manifest.onnx.sha256).toBe(MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256);

      const caseNames = Object.keys(manifest.cases);
      expect(caseNames.length).toBeGreaterThanOrEqual(8);

      for (const name of caseNames) {
        const entry = manifest.cases[name];
        const tsMeta = computeMirageCeil32Metadata(
          entry.metadata.W,
          entry.metadata.H
        );
        expect(tsMeta).toEqual(entry.metadata);
        expect([tsMeta.h, tsMeta.w]).toEqual(entry.valid_hw);

        const loadFloat32 = (file: string) => {
          const buf = readFileSync(path.join(FIXTURES_DIR, file));
          return new Float32Array(
            buf.buffer,
            buf.byteOffset,
            buf.byteLength / 4
          );
        };

        const rawHeads = {
          detections: loadFloat32(entry.heads.detections.file),
          prototypes: loadFloat32(entry.heads.prototypes.file),
          coefficients: loadFloat32(entry.heads.coefficients.file),
        };

        // 1. Default policy: confidence 0.3, 3x3 local peaks, max 64 objects, max 32 text, no NMS
        const decodedDefault = decodeMirageExperimentalOutputs(
          rawHeads,
          tsMeta,
          { confidence: 0.3, nmsIou: null }
        );
        expect(decodedDefault.objects).toHaveLength(
          entry.decoded_default.object_count
        );
        expect(decodedDefault.text).toHaveLength(
          entry.decoded_default.text_count
        );

        for (let i = 0; i < entry.decoded_default.objects.length; i++) {
          const expected = entry.decoded_default.objects[i];
          const actual = decodedDefault.objects[i];
          expect(actual.cell).toEqual(expected.cell);
          expect(actual.score).toBeCloseTo(expected.score, 6);
          expect(actual.bbox).toEqual(expected.bbox);
          expect(actual.mask).toBeDefined();
          expect(actual.maskActivePixels).toBe(expected.mask_active_pixels);
          const actualMaskSha = createHash('sha256')
            .update(Buffer.from(actual.mask!))
            .digest('hex');
          expect(actualMaskSha).toBe(expected.mask_sha256);
        }

        for (let i = 0; i < entry.decoded_default.text.length; i++) {
          const expected = entry.decoded_default.text[i];
          const actual = decodedDefault.text[i];
          expect(actual.cell).toEqual(expected.cell);
          expect(actual.score).toBeCloseTo(expected.score, 6);
          expect(actual.bbox).toEqual(expected.bbox);
        }

        // 2. Prespecified 0.5 greedy IoU NMS policy
        const decodedNms = decodeMirageExperimentalOutputs(rawHeads, tsMeta, {
          confidence: 0.3,
          nmsIou: 0.5,
        });
        expect(decodedNms.objects).toHaveLength(
          entry.decoded_nms_0p50.object_count
        );
        expect(decodedNms.text).toHaveLength(entry.decoded_nms_0p50.text_count);

        for (let i = 0; i < entry.decoded_nms_0p50.objects.length; i++) {
          const expected = entry.decoded_nms_0p50.objects[i];
          const actual = decodedNms.objects[i];
          expect(actual.cell).toEqual(expected.cell);
          expect(actual.score).toBeCloseTo(expected.score, 6);
          expect(actual.bbox).toEqual(expected.bbox);
          expect(actual.maskActivePixels).toBe(expected.mask_active_pixels);
        }
      }
    }
  );

  it('never fabricates object names or OCR text strings and separates native WASM overlay sources', async () => {
    const detections = new Float32Array(64 * 64 * 10).fill(-10);
    const prototypes = new Float32Array(32 * 128 * 128).fill(0);
    const coefficients = new Float32Array(32 * 64 * 64).fill(0);

    // Put 1 object peak at (16, 20) and 1 text peak at (8, 12)
    const objIdx = (16 * 64 + 20) * 10;
    detections[objIdx + 0] = 2.5; // object logit -> sigmoid ~ 0.924
    detections[objIdx + 2] = 16;  // width
    detections[objIdx + 3] = 20;  // height
    detections[objIdx + 6] = 0.5; // dx
    detections[objIdx + 7] = 0.5; // dy

    const txtIdx = (8 * 64 + 12) * 10;
    detections[txtIdx + 1] = 1.8; // text logit -> sigmoid ~ 0.858
    detections[txtIdx + 4] = 24;  // width
    detections[txtIdx + 5] = 6;   // height
    detections[txtIdx + 8] = 0.25;
    detections[txtIdx + 9] = 0.25;

    coefficients[0 * 4096 + 16 * 64 + 20] = 2.0;
    prototypes.fill(1.0, 0, 128 * 128);

    const meta = computeMirageCeil32Metadata(512, 512);
    const decoded = decodeMirageExperimentalOutputs(
      { detections, prototypes, coefficients },
      meta
    );

    const converted = convertProposalsToDetectedObjects(decoded, {
      sourceAttributionPrefix: true,
    });
    expect(converted).toHaveLength(2);
    expect(converted[0].label).toBe('object_region_candidate_1');
    expect(converted[0].labelJa).toBe('[ONNX候補] 物体領域候補 #1 (16,20)');
    expect(converted[0].category).toBe('物体領域候補 (部分ONNX)');
    expect(converted[1].label).toBe('text_region_candidate_1');
    expect(converted[1].labelJa).toBe('[ONNX候補] テキスト領域候補 #1 (8,12)');
    expect(converted[1].category).toBe('テキスト領域候補 (部分ONNX)');

    // Verify MirageExperimentalOnnxEngine requests only the 3 terminal heads, disposes tensors, and tags native overlay
    const runSpy = vi.fn(async (_feeds: Record<string, ort.Tensor>, fetches?: string[]) => {
      expect(fetches).toEqual([...MIRAGE_EXPERIMENTAL_ONNX_FINAL_OUTPUTS]);
      return {
        detections: new ort.Tensor('float32', detections, [1, 64, 64, 10]),
        prototypes: new ort.Tensor('float32', prototypes, [1, 32, 128, 128]),
        coefficients: new ort.Tensor('float32', coefficients, [1, 32, 64, 64]),
      };
    });
    const releaseSpy = vi.fn(async () => {});
    const fakeSession = {
      inputNames: ['image', 'valid_hw'],
      outputNames: [...MIRAGE_EXPERIMENTAL_ONNX_FINAL_OUTPUTS],
      run: runSpy,
      release: releaseSpy,
    } as unknown as ort.InferenceSession;

    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      imageSmoothingEnabled: true,
      imageSmoothingQuality: 'high',
      clearRect: vi.fn(),
      drawImage: vi.fn(),
      getImageData: (_x: number, _y: number, w: number, h: number) => ({
        data: new Uint8ClampedArray(w * h * 4).fill(128),
      }),
      createImageData: (w: number, h: number) => ({
        data: new Uint8ClampedArray(w * h * 4),
      }),
      putImageData: vi.fn(),
    } as unknown as CanvasRenderingContext2D);

    const nativeOverlayClose = vi.fn();
    const engine = new MirageExperimentalOnnxEngine(
      fakeSession,
      'mirage-onnx-exp-wasm',
      undefined,
      async () => ({
        detectForVideo: async () => [
          {
            id: 'native-1',
            label: 'person',
            labelJa: '人物（中央・中）',
            category: '人物',
            confidence: 0.91,
            bbox: { x: 0.2, y: 0.2, width: 0.3, height: 0.4 },
            color: '#10b981',
          },
        ],
        close: nativeOverlayClose,
      })
    );

    const inputCanvas = document.createElement('canvas');
    inputCanvas.width = 640;
    inputCanvas.height = 480;

    // Default: pure ONNX proposals only
    const pureResult = await engine.detectForVideo(inputCanvas, 0.3);
    expect(pureResult).toHaveLength(2);
    expect(pureResult.every((o) => o.labelJa.startsWith('[ONNX候補]'))).toBe(true);

    // With overlayNativeWasm: true, native results are clearly tagged with [公式WASM]
    const mixedResult = await engine.detectForVideo(inputCanvas, 0.3, {
      overlayNativeWasm: true,
    });
    expect(mixedResult).toHaveLength(3);
    expect(mixedResult[2].labelJa).toBe('[公式WASM] 人物（中央・中）');
    expect(mixedResult[2].category).toBe('人物 (公式WASM)');

    const diag = engine.getLastDiagnostics();
    expect(diag?.validHw).toEqual([384, 512]);
    expect(diag?.objectProposalCount).toBe(1);
    expect(diag?.textProposalCount).toBe(1);
    expect(diag?.nativeOverlayCount).toBe(1);

    await engine.close();
    expect(releaseSpy).toHaveBeenCalledTimes(1);
    expect(nativeOverlayClose).toHaveBeenCalledTimes(1);
    vi.restoreAllMocks();
  });

  it('enforces pinned SHA-256 and byte length on /api/model-cache/mirage/cact-dequantized-reference-order.onnx', async () => {
    const tempRoot = mkdtempSync(path.join(tmpdir(), 'mirage-exp-cache-test-'));
    try {
      const manager = new ContainerModelCacheManager({
        rootDir: tempRoot,
        cacheDir: path.join(tempRoot, '.cache', 'container-models'),
        autoPrefetchOnStart: false,
      });
      const mw = manager.createMiddleware();

      // 1. Missing ?sha256= query parameter -> 400
      const noPinRes = await invokeMiddleware(
        mw,
        `/api/model-cache/mirage/${MIRAGE_EXPERIMENTAL_ONNX_FILE}`
      );
      expect(noPinRes.status).toBe(400);

      // 2. Missing file in .cache -> 503 MIRAGE_EXPERIMENTAL_ONNX_MISSING
      const missingRes = await invokeMiddleware(
        mw,
        MIRAGE_EXPERIMENTAL_ONNX_LOCAL_URL
      );
      expect(missingRes.status).toBe(503);
      expect(JSON.parse(missingRes.body.toString('utf8'))).toMatchObject({
        code: 'MIRAGE_EXPERIMENTAL_ONNX_MISSING',
        expectedSize: MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE,
        expectedSha256: MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256,
      });

      // 3. Corrupted file in .cache -> 503 MIRAGE_EXPERIMENTAL_ONNX_INTEGRITY_FAILED
      const cacheDir = path.join(tempRoot, '.cache', 'container-models');
      mkdirSync(cacheDir, { recursive: true });
      const corruptedPath = path.join(cacheDir, MIRAGE_EXPERIMENTAL_ONNX_FILE);
      writeFileSync(corruptedPath, Buffer.alloc(2048, 0x42));
      const corruptedRes = await invokeMiddleware(
        mw,
        MIRAGE_EXPERIMENTAL_ONNX_LOCAL_URL
      );
      expect(corruptedRes.status).toBe(503);
      expect(JSON.parse(corruptedRes.body.toString('utf8'))).toMatchObject({
        code: 'MIRAGE_EXPERIMENTAL_ONNX_INTEGRITY_FAILED',
      });

      // 4. mirage-webgpu.onnx remains 503 MIRAGE_ONNX_NOT_VALIDATED
      const webgpuRes = await invokeMiddleware(
        mw,
        '/api/model-cache/mirage/mirage-webgpu.onnx'
      );
      expect(webgpuRes.status).toBe(503);
      expect(JSON.parse(webgpuRes.body.toString('utf8'))).toMatchObject({
        code: 'MIRAGE_ONNX_NOT_VALIDATED',
      });
    } finally {
      rmSync(tempRoot, { recursive: true, force: true });
    }
  });

  it('serves the verified 34,281,503-byte ONNX artifact and verifies SHA-256 when present in workspace .cache', async () => {
    const verifiedPath = path.resolve(
      '.cache/mirage-refinement/numerical-cact-dequantized-pixel-0-r3/cact-dequantized-reference-order.onnx'
    );
    if (!existsSync(verifiedPath)) return;

    const manager = new ContainerModelCacheManager({
      rootDir: process.cwd(),
      autoPrefetchOnStart: false,
    });
    const mw = manager.createMiddleware();
    const headRes = await invokeMiddleware(
      mw,
      MIRAGE_EXPERIMENTAL_ONNX_LOCAL_URL,
      'HEAD'
    );
    expect(headRes.status).toBe(200);
    expect(headRes.headers['content-length']).toBe(
      String(MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE)
    );
    expect(headRes.headers['x-model-sha256']).toBe(
      MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256
    );

    const fileBytes = readFileSync(verifiedPath);
    const verified = await verifyMirageExperimentalOnnxBytes(fileBytes);
    expect(verified).toEqual({
      size: MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE,
      sha256: MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256,
    });
  });

  it('prevents stale experimental Mirage results after VisionEngine switches models', async () => {
    let resolveRun!: (v: Record<string, ort.Tensor>) => void;
    const slowSession = {
      inputNames: ['image', 'valid_hw'],
      outputNames: [...MIRAGE_EXPERIMENTAL_ONNX_FINAL_OUTPUTS],
      run: vi.fn(
        () =>
          new Promise<Record<string, ort.Tensor>>((r) => {
            resolveRun = r;
          })
      ),
      release: vi.fn(async () => {}),
    } as unknown as ort.InferenceSession;

    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      imageSmoothingEnabled: true,
      imageSmoothingQuality: 'high',
      clearRect: vi.fn(),
      drawImage: vi.fn(),
      getImageData: (_x: number, _y: number, w: number, h: number) => ({
        data: new Uint8ClampedArray(w * h * 4),
      }),
    } as unknown as CanvasRenderingContext2D);

    const expTask = new MirageExperimentalOnnxEngine(
      slowSession,
      'mirage-onnx-exp-wasm'
    );
    const visionEngine = new VisionEngine({
      load: async () => ({ wasmLoaderPath: '', wasmBinaryPath: '' }),
      detector: async () => ({
        detectForVideo: () => ({ detections: [] }),
        close: vi.fn(),
      }),
      segmenter: async () => ({
        segmentForVideo: () => ({
          confidenceMasks: [],
          categoryMask: undefined,
          close: vi.fn(),
        }),
        close: vi.fn(),
      }),
      mirage: async () => expTask,
    });

    await visionEngine.initialize('mirage', 'mirage-onnx-exp-wasm');
    const c = document.createElement('canvas');
    c.width = 512;
    c.height = 512;

    const pendingFrame = visionEngine.detectVideoFrame(c, 'mirage', 0.3);
    // Switch to mediapipe while experimental ONNX inference is still in-flight
    await visionEngine.initialize('mediapipe', 'lite0');
    expect(slowSession.release).not.toHaveBeenCalled();

    resolveRun({
      detections: new ort.Tensor(
        'float32',
        new Float32Array(64 * 64 * 10),
        [1, 64, 64, 10]
      ),
      prototypes: new ort.Tensor(
        'float32',
        new Float32Array(32 * 128 * 128),
        [1, 32, 128, 128]
      ),
      coefficients: new ort.Tensor(
        'float32',
        new Float32Array(32 * 64 * 64),
        [1, 32, 64, 64]
      ),
    });

    const settled = await pendingFrame;
    expect(settled).toMatchObject({ status: 'not-ready', model: 'mirage' });
    expect(slowSession.release).toHaveBeenCalledTimes(1);
    expect(computeBoxIou([0, 0, 10, 10], [0, 0, 10, 10])).toBe(1);
    visionEngine.dispose();
    vi.restoreAllMocks();
  });

  describe('fetchModelBuffer', () => {
    it('fetches large models in chunks using HTTP Range requests', async () => {
      const mockTotalSize = 34_281_503;
      const CHUNK_SIZE = 16 * 1024 * 1024;
      const c1Size = CHUNK_SIZE;
      const c2Size = CHUNK_SIZE;
      const c3Size = mockTotalSize - 2 * CHUNK_SIZE;

      const c1Data = new Uint8Array(c1Size).fill(1);
      const c2Data = new Uint8Array(c2Size).fill(2);
      const c3Data = new Uint8Array(c3Size).fill(3);

      const requestedRanges: string[] = [];
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, init) => {
        const range = (init?.headers as Record<string, string>)?.['Range'] ?? '';
        requestedRanges.push(range);

        let body: Uint8Array;
        if (range === `bytes=0-${c1Size - 1}`) {
          body = c1Data;
        } else if (range === `bytes=${c1Size}-${c1Size + c2Size - 1}`) {
          body = c2Data;
        } else {
          body = c3Data;
        }

        return new Response(body as unknown as BodyInit, {
          status: 206,
          headers: {
            'Content-Range': `${range}/${mockTotalSize}`,
            'Content-Length': String(body.byteLength),
          },
        });
      });

      const onChunkSpy = vi.fn();
      const buffer = await fetchModelBuffer('https://mock.invalid/model.onnx', {
        expectedSize: mockTotalSize,
        onChunk: onChunkSpy,
      });

      expect(fetchSpy).toHaveBeenCalledTimes(3);
      expect(requestedRanges).toEqual([
        `bytes=0-${c1Size - 1}`,
        `bytes=${c1Size}-${c1Size + c2Size - 1}`,
        `bytes=${c1Size + c2Size}-${mockTotalSize - 1}`,
      ]);
      expect(onChunkSpy).toHaveBeenCalledTimes(3);
      expect(buffer.byteLength).toBe(mockTotalSize);

      const uint8 = new Uint8Array(buffer);
      expect(uint8[0]).toBe(1);
      expect(uint8[c1Size]).toBe(2);
      expect(uint8[c1Size + c2Size]).toBe(3);

      fetchSpy.mockRestore();
    });

    it('falls back to single payload when server returns 200 on first chunk', async () => {
      const mockSize = 34_281_503;
      const fullData = new Uint8Array(mockSize).fill(42);

      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => {
        return new Response(fullData as unknown as BodyInit, {
          status: 200,
          headers: { 'Content-Length': String(mockSize) },
        });
      });

      const buffer = await fetchModelBuffer('https://mock.invalid/model.onnx', {
        expectedSize: mockSize,
      });

      expect(fetchSpy).toHaveBeenCalledTimes(1);
      expect(buffer.byteLength).toBe(mockSize);

      fetchSpy.mockRestore();
    });
  });

  describe('R1/R0 Runtime Reconstruction & Staged Packaging', () => {
    it('resolves runtime reconstruction assets independent of process.cwd() and without docs/ or scripts/mirage/', async () => {
      const originalCwd = process.cwd();
      const isolatedStageDir = mkdtempSync(path.join(tmpdir(), 'mirage-staged-runtime-'));
      const foreignCwdDir = mkdtempSync(path.join(tmpdir(), 'mirage-foreign-cwd-'));
      try {
        // 1. Default MIRAGE_PACKAGE_ROOT resolution works even when process.cwd() is a foreign empty directory
        process.chdir(foreignCwdDir);
        expect(hasMirageReconstructionAssets()).toBe(true);
        const defaultPaths = resolveMirageReconstructionAssetPaths();
        expect(defaultPaths.assetsDir.startsWith(MIRAGE_PACKAGE_ROOT)).toBe(true);
        expect(existsSync(defaultPaths.mappingPath)).toBe(true);
        expect(existsSync(defaultPaths.skeletonGzPath)).toBe(true);
        expect(existsSync(defaultPaths.placementsPath)).toBe(true);

        // 2. Build a clean staged directory containing ONLY src/assets/mirage-reconstruction (no docs/, no scripts/mirage/, no .cache/)
        const stagedAssetsDir = path.join(isolatedStageDir, 'src', 'assets', 'mirage-reconstruction');
        const stagedPresetsDir = path.join(stagedAssetsDir, 'presets');
        mkdirSync(stagedPresetsDir, { recursive: true });
        for (const pin of Object.values(MIRAGE_RECONSTRUCTION_ASSETS_PIN)) {
          const srcFile = path.join(defaultPaths.assetsDir, pin.file);
          const bytes = readFileSync(srcFile);
          expect(bytes.byteLength).toBe(pin.size);
          expect(createHash('sha256').update(bytes).digest('hex')).toBe(pin.sha256);
          writeFileSync(path.join(stagedAssetsDir, pin.file), bytes);
        }
        for (const presetName of ['coco_000000458054.png', 'overlap_shapes.png', 'text_regions.png']) {
          writeFileSync(
            path.join(stagedPresetsDir, presetName),
            readFileSync(path.join(defaultPaths.presetsDir, presetName))
          );
        }
        expect(existsSync(path.join(isolatedStageDir, 'docs'))).toBe(false);
        expect(existsSync(path.join(isolatedStageDir, 'scripts'))).toBe(false);
        expect(hasMirageReconstructionAssets(isolatedStageDir)).toBe(true);

        // 3. Presets are served from the staged runtime directory without .cache/mirage-browser-fixtures
        const stagedCacheDir = path.join(isolatedStageDir, 'writable-cache');
        const stagedManager = new ContainerModelCacheManager({
          rootDir: isolatedStageDir,
          cacheDir: stagedCacheDir,
          autoPrefetchOnStart: false,
        });
        const mw = stagedManager.createMiddleware();
        const presetRes = await invokeMiddleware(
          mw,
          '/api/model-cache/mirage/fixtures/overlap_shapes.png'
        );
        expect(presetRes.status).toBe(200);
        expect(presetRes.headers['content-type']).toBe('image/png');
        expect(presetRes.body.byteLength).toBe(2620);

        // 4. If verified mirage.cact is available locally, verify full reconstruction and corrupted ONNX replacement in the isolated stage
        const localCactPath = path.join(originalCwd, '.cache', 'container-models', 'mirage.cact');
        if (existsSync(localCactPath)) {
          const cactBytes = readFileSync(localCactPath);
          expect(cactBytes.byteLength).toBe(MIRAGE_CACT_EXPECTED_SIZE);
          expect(createHash('sha256').update(cactBytes).digest('hex')).toBe(MIRAGE_CACT_EXPECTED_SHA256);

          mkdirSync(stagedCacheDir, { recursive: true });
          writeFileSync(path.join(stagedCacheDir, 'mirage.cact'), cactBytes);
          // Seed a corrupted ONNX file to verify automatic replacement
          const targetOnnxPath = path.join(stagedCacheDir, MIRAGE_EXPERIMENTAL_ONNX_FILE);
          writeFileSync(targetOnnxPath, Buffer.from('corrupted-onnx-payload'));

          const headRes = await invokeMiddleware(
            mw,
            MIRAGE_EXPERIMENTAL_ONNX_LOCAL_URL,
            'HEAD'
          );
          expect(headRes.status).toBe(200);
          expect(headRes.headers['content-length']).toBe(
            String(MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE)
          );
          expect(headRes.headers['x-model-sha256']).toBe(
            MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256
          );

          const reconstructedBytes = readFileSync(targetOnnxPath);
          expect(reconstructedBytes.byteLength).toBe(MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE);
          expect(createHash('sha256').update(reconstructedBytes).digest('hex')).toBe(
            MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256
          );
        }
      } finally {
        process.chdir(originalCwd);
        rmSync(isolatedStageDir, { recursive: true, force: true });
        rmSync(foreignCwdDir, { recursive: true, force: true });
      }
    });

    it('rejects tampered reconstruction assets or tampered mirage.cact before writing target ONNX', () => {
      const tempRoot = mkdtempSync(path.join(tmpdir(), 'mirage-tampered-reconstruct-'));
      try {
        const targetOnnx = path.join(tempRoot, MIRAGE_EXPERIMENTAL_ONNX_FILE);
        expect(() =>
          ensureReconstructedMirageOnnxFile(targetOnnx, Buffer.alloc(1024, 0xaa))
        ).toThrow(/mirage\.cact byte length mismatch/);
        expect(existsSync(targetOnnx)).toBe(false);
      } finally {
        rmSync(tempRoot, { recursive: true, force: true });
      }
    });
  });

  describe('R2 Tensor Parity Metrics & Strict WebGPU Execution Mode', () => {
    it('computes maxAbsDiff, meanAbsDiff, rmsDiff, maxRelDiff (floor 1e-6), withinAtolRtol(2e-4), cosineSimilarity, and finite checks', () => {
      const expected = new Float32Array([0.0, 1e-8, 0.5, -1.25]);
      const actualPass = new Float32Array([1e-5, 1e-8 + 5e-5, 0.5 + 1e-4, -1.25 - 1e-4]);
      const passMetrics = compareMirageTensorParity(actualPass, expected, 2e-4, 2e-4, 1e-6);
      expect(passMetrics.withinAtolRtol).toBe(true);
      expect(passMetrics.allcloseAt2e4).toBe(true);
      expect(passMetrics.violationsAt2e4).toBe(0);
      expect(passMetrics.actualFinite).toBe(true);
      expect(passMetrics.expectedFinite).toBe(true);
      expect(passMetrics.rmsDiff).toBeGreaterThan(0);
      expect(passMetrics.cosineSimilarity).toBeGreaterThan(0.999999);

      const actualFail = new Float32Array([0.0, 1e-8, 0.5 + 1e-3, -1.25]);
      const failMetrics = compareMirageTensorParity(actualFail, expected, 2e-4, 2e-4, 1e-6);
      expect(failMetrics.withinAtolRtol).toBe(false);
      expect(failMetrics.violationsAt2e4).toBe(1);

      const actualNan = new Float32Array([Number.NaN, 0, 0, 0]);
      const nanMetrics = compareMirageTensorParity(actualNan, expected, 2e-4, 2e-4, 1e-6);
      expect(nanMetrics.actualFinite).toBe(false);
      expect(nanMetrics.withinAtolRtol).toBe(false);
    });

    it('enforces strict WebGPU execution without silent WASM fallback when allowWasmFallback is false', async () => {
      const dummyOnnxBytes = new Uint8Array(64);
      vi.spyOn(globalThis, 'fetch').mockImplementation(async () => {
        return new Response(dummyOnnxBytes, {
          status: 200,
          headers: { 'Content-Length': String(dummyOnnxBytes.byteLength) },
        });
      });

      const adapterDiag = await probeWebGpuAdapterDiagnostics();
      expect(typeof adapterDiag.navigatorGpuPresent).toBe('boolean');

      await expect(
        createMirageExperimentalOnnxTask({
          variantId: 'mirage-onnx-exp-webgpu',
          allowWasmFallback: false,
        })
      ).rejects.toThrow();

      vi.restoreAllMocks();
    });
  });

  describe('R3 Same-Frame Snapshot & Lifecycle Ownership', () => {
    it('captures an immutable source frame before await so ONNX and native WASM overlay process the exact same frame even when live video advances', async () => {
      const detections = new Float32Array(64 * 64 * 10).fill(-10);
      const prototypes = new Float32Array(32 * 128 * 128).fill(0);
      const coefficients = new Float32Array(32 * 64 * 64).fill(0);
      detections[(10 * 64 + 10) * 10 + 0] = 3.0;
      detections[(10 * 64 + 10) * 10 + 2] = 16;
      detections[(10 * 64 + 10) * 10 + 3] = 16;

      let currentLivePixelR = 17; // Frame 1 pixel R=17
      const canvasPixelMap = new WeakMap<HTMLCanvasElement, Uint8ClampedArray>();

      vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (
        this: HTMLCanvasElement
      ) {
        return {
          imageSmoothingEnabled: true,
          imageSmoothingQuality: 'high',
          clearRect: vi.fn(),
          drawImage: vi.fn((source: CanvasImageSource) => {
            const w = this.width || 64;
            const h = this.height || 64;
            const srcBuf = canvasPixelMap.get(source as HTMLCanvasElement);
            const dstBuf = new Uint8ClampedArray(w * h * 4);
            const rVal = srcBuf ? srcBuf[0] : currentLivePixelR;
            for (let i = 0; i < dstBuf.length; i += 4) {
              dstBuf[i] = rVal;
              dstBuf[i + 1] = 40;
              dstBuf[i + 2] = 80;
              dstBuf[i + 3] = 255;
            }
            canvasPixelMap.set(this, dstBuf);
          }),
          putImageData: vi.fn((imgData: { data: Uint8ClampedArray }) => {
            canvasPixelMap.set(this, new Uint8ClampedArray(imgData.data));
          }),
          getImageData: (_x: number, _y: number, w: number, h: number) => {
            const stored = canvasPixelMap.get(this);
            if (stored && stored.length === w * h * 4) {
              return { data: new Uint8ClampedArray(stored) };
            }
            const data = new Uint8ClampedArray(w * h * 4);
            for (let i = 0; i < data.length; i += 4) {
              data[i] = currentLivePixelR;
              data[i + 1] = 40;
              data[i + 2] = 80;
              data[i + 3] = 255;
            }
            return { data };
          },
          createImageData: (w: number, h: number) => ({
            data: new Uint8ClampedArray(w * h * 4),
          }),
        } as unknown as CanvasRenderingContext2D;
      });

      let resolveOnnxRun!: (v: Record<string, ort.Tensor>) => void;
      const slowSession = {
        inputNames: ['image', 'valid_hw'],
        outputNames: [...MIRAGE_EXPERIMENTAL_ONNX_FINAL_OUTPUTS],
        run: vi.fn(
          () =>
            new Promise<Record<string, ort.Tensor>>((r) => {
              resolveOnnxRun = r;
            })
        ),
        release: vi.fn(async () => {}),
      } as unknown as ort.InferenceSession;

      let capturedNativeSnapshot: MirageSourceFrameSnapshot | undefined;
      let capturedNativeGeneration: number | undefined;
      const mockCactusDescribe = vi.fn((rgbBytes: Uint8Array) => {
        // Verify MirageWasmEngine resampled from Frame 1 (R=17), NOT Frame 2 (R=222)
        expect(rgbBytes[0]).toBe(17);
      });

      const mockWorker = {
        onmessage: null as ((event: MessageEvent) => void) | null,
        onerror: null as ((event: ErrorEvent) => void) | null,
        postMessage: vi.fn(
          (msg: { type: string; id: number; rgbBuffer: ArrayBuffer }) => {
            if (msg.type === 'describe') {
              const rgbBytes = new Uint8Array(msg.rgbBuffer);
              mockCactusDescribe(rgbBytes);
              queueMicrotask(() => {
                mockWorker.onmessage?.({
                  data: {
                    type: 'describe-result',
                    id: msg.id,
                    ok: true,
                    observation: {
                      objects: [
                        {
                          label: 'chair',
                          score: 0.88,
                          bbox: [100, 100, 300, 300],
                        },
                      ],
                      text: [],
                    },
                  },
                } as MessageEvent);
              });
            }
          }
        ),
        terminate: vi.fn(),
      };

      const realNativeEngine = new MirageWasmEngine(mockWorker);
      const origDetectForVideo =
        realNativeEngine.detectForVideo.bind(realNativeEngine);
      const nativeDetectSpy = vi
        .spyOn(realNativeEngine, 'detectForVideo')
        .mockImplementation(async (input, threshold, extra) => {
          capturedNativeSnapshot = extra?.sourceSnapshot as
            | MirageSourceFrameSnapshot
            | undefined;
          capturedNativeGeneration = extra?.sourceGeneration;
          return origDetectForVideo(input, threshold, extra);
        });

      const engine = new MirageExperimentalOnnxEngine(
        slowSession,
        'mirage-onnx-exp-wasm',
        undefined,
        async () => realNativeEngine
      );

      const liveVideoCanvas = document.createElement('canvas');
      liveVideoCanvas.width = 64;
      liveVideoCanvas.height = 64;

      // Start inference on Frame 1 (R=17)
      const pendingInference = engine.detectForVideo(liveVideoCanvas, 0.3, {
        overlayNativeWasm: true,
        sourceGeneration: 7,
      });

      // Advance live video to Frame 2 (R=222) while ONNX is still awaiting session.run()
      currentLivePixelR = 222;
      await vi.waitFor(() => expect(slowSession.run).toHaveBeenCalledTimes(1));

      resolveOnnxRun({
        detections: new ort.Tensor('float32', detections, [1, 64, 64, 10]),
        prototypes: new ort.Tensor('float32', prototypes, [1, 32, 128, 128]),
        coefficients: new ort.Tensor('float32', coefficients, [1, 32, 64, 64]),
      });

      const results = await pendingInference;
      expect(results).toHaveLength(2);
      expect(nativeDetectSpy).toHaveBeenCalledTimes(1);
      expect(mockCactusDescribe).toHaveBeenCalledTimes(1);
      expect(capturedNativeSnapshot).toBeDefined();
      expect(capturedNativeSnapshot!.rgba[0]).toBe(17);
      expect(capturedNativeGeneration).toBe(7);
      // Snapshot must be released after both ONNX and native overlay finish
      expect(capturedNativeSnapshot!.refCount).toBe(0);
      expect(capturedNativeSnapshot!.disposed).toBe(true);

      const diag = engine.getLastDiagnostics();
      expect(diag?.frameAlignment).toBeDefined();
      expect(diag?.frameAlignment?.sameSourceFrame).toBe(true);
      expect(diag?.frameAlignment?.onnxFrameId).toBe(diag?.frameAlignment?.nativeFrameId);
      expect(diag?.frameAlignment?.onnxRgbaSha256).toBe(diag?.frameAlignment?.nativeRgbaSha256);
      expect(diag?.frameAlignment?.onnxSourceGeneration).toBe(7);
      expect(diag?.frameAlignment?.nativeSourceGeneration).toBe(7);

      await engine.close();
      vi.restoreAllMocks();
    });

    it('isolates overlapping snapshots and prevents stale native overlay execution after source change or engine close', async () => {
      let currentPixelR = 50;
      vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (
        this: HTMLCanvasElement
      ) {
        return {
          imageSmoothingEnabled: true,
          imageSmoothingQuality: 'high',
          clearRect: vi.fn(),
          drawImage: vi.fn(),
          putImageData: vi.fn(),
          getImageData: (_x: number, _y: number, w: number, h: number) => {
            const data = new Uint8ClampedArray(w * h * 4);
            for (let i = 0; i < data.length; i += 4) {
              data[i] = currentPixelR;
              data[i + 3] = 255;
            }
            return { data };
          },
          createImageData: (w: number, h: number) => ({
            data: new Uint8ClampedArray(w * h * 4),
          }),
        } as unknown as CanvasRenderingContext2D;
      });

      const liveCanvas = document.createElement('canvas');
      liveCanvas.width = 32;
      liveCanvas.height = 32;

      // Capture two overlapping snapshots on different frames
      const snap1 = captureMirageSourceFrameSnapshot(liveCanvas, 1, 1);
      currentPixelR = 99;
      const snap2 = captureMirageSourceFrameSnapshot(liveCanvas, 2, 1);

      expect(snap1.frameId).toBe(1);
      expect(snap2.frameId).toBe(2);
      expect(snap1.rgba[0]).toBe(50);
      expect(snap2.rgba[0]).toBe(99);
      expect(snap1.rgbaSha256).not.toBe(snap2.rgbaSha256);
      expect(snap1.rgbaSha256).toBe(sha256HexSync(snap1.rgba));

      snap1.retain();
      expect(snap1.refCount).toBe(2);
      snap1.release();
      expect(snap1.disposed).toBe(false);
      snap1.release();
      expect(snap1.disposed).toBe(true);
      // Idempotent second release on zero refCount must not throw or go negative
      snap1.release();
      expect(snap1.refCount).toBe(0);
      expect(snap2.disposed).toBe(false);
      snap2.release();
      expect(snap2.disposed).toBe(true);

      // Verify notifySourceChanged cancels in-flight inference before native overlay runs
      let resolveOnnx!: (v: Record<string, ort.Tensor>) => void;
      const slowSession = {
        inputNames: ['image', 'valid_hw'],
        outputNames: [...MIRAGE_EXPERIMENTAL_ONNX_FINAL_OUTPUTS],
        run: vi.fn(
          () =>
            new Promise<Record<string, ort.Tensor>>((r) => {
              resolveOnnx = r;
            })
        ),
        release: vi.fn(async () => {}),
      } as unknown as ort.InferenceSession;

      const nativeDetect = vi.fn(async () => []);
      const nativeClose = vi.fn();
      const engine = new MirageExperimentalOnnxEngine(
        slowSession,
        'mirage-onnx-exp-wasm',
        undefined,
        async () => ({
          detectForVideo: nativeDetect,
          close: nativeClose,
        })
      );

      const inFlight = engine.detectForVideo(liveCanvas, 0.3, {
        overlayNativeWasm: true,
      });
      // Switch camera / source while ONNX is still running
      await vi.waitFor(() => expect(slowSession.run).toHaveBeenCalledTimes(1));
      engine.notifySourceChanged();

      resolveOnnx({
        detections: new ort.Tensor('float32', new Float32Array(64 * 64 * 10), [1, 64, 64, 10]),
        prototypes: new ort.Tensor('float32', new Float32Array(32 * 128 * 128), [1, 32, 128, 128]),
        coefficients: new ort.Tensor('float32', new Float32Array(32 * 64 * 64), [1, 32, 64, 64]),
      });

      const cancelledResult = await inFlight;
      expect(cancelledResult).toEqual([]);
      expect(nativeDetect).not.toHaveBeenCalled();

      await engine.close();
      vi.restoreAllMocks();
    });
  });
});
