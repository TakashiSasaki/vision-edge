import { expect, test, type BrowserContext } from '@playwright/test';
import { existsSync, writeFileSync } from 'node:fs';
import { IncomingMessage, ServerResponse } from 'node:http';
import path from 'node:path';
import { ContainerModelCacheManager } from '../src/services/containerModelCacheServer';
import {
  MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256,
  MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE,
  type MirageTensorParityMetrics,
} from '../src/services/mirageExperimentalOnnxService';
import type { MirageInferExtraOptions } from '../src/services/mirageWasmService';
import type { OnnxBackendInfo } from '../src/types/vision';

interface IdenticalInputCaseReport {
  sourceDimensions: [number, number];
  validHw: [number, number];
  coordinateScale: [number, number, number, number];
  identicalInputParity: {
    inferenceMs: number;
    detections: MirageTensorParityMetrics;
    prototypes: MirageTensorParityMetrics;
    coefficients: MirageTensorParityMetrics;
    allThreeHeadsAllcloseAt2e4: boolean;
    exactProposalParityNone: boolean;
    exactProposalParityNms050: boolean;
    objectCountNoNms: number;
    textCountNoNms: number;
    objectCountNms050: number;
    textCountNms050: number;
  };
}

const WORKSPACE_ROOT = process.cwd();
const ONNX_ARTIFACT_PATH = path.resolve(
  WORKSPACE_ROOT,
  '.cache/mirage-refinement/numerical-cact-dequantized-pixel-0-r3/cact-dequantized-reference-order.onnx'
);
const FIXTURE_MANIFEST_PATH = path.resolve(
  WORKSPACE_ROOT,
  '.cache/mirage-browser-fixtures/manifest.json'
);

function runCacheMiddleware(
  middleware: ReturnType<ContainerModelCacheManager['createMiddleware']>,
  urlPath: string,
  method: string,
  requestHeaders: Record<string, string> = {}
): Promise<{
  handled: boolean;
  status: number;
  headers: Record<string, string>;
  body: Buffer;
}> {
  return new Promise((resolve, reject) => {
    const headers: Record<string, string> = {};
    const chunks: Buffer[] = [];
    const req = {
      url: urlPath,
      method,
      headers: {
        host: '127.0.0.1:4173',
        ...(requestHeaders.range ? { range: requestHeaders.range } : {}),
      },
    } as unknown as IncomingMessage;
    const rawRes = {
      statusCode: 200,
      setHeader(name: string, value: string) {
        headers[name.toLowerCase()] = String(value);
      },
      end(chunk?: Buffer | string) {
        if (chunk !== undefined) {
          chunks.push(
            Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk))
          );
        }
        resolve({
          handled: true,
          status: rawRes.statusCode,
          headers,
          body: Buffer.concat(chunks),
        });
      },
    };
    const res = rawRes as unknown as ServerResponse;
    middleware(req, res, (err) => {
      if (err) reject(err);
      else
        resolve({
          handled: false,
          status: 404,
          headers,
          body: Buffer.alloc(0),
        });
    });
  });
}

async function routeLocalModelCache(context: BrowserContext) {
  const manager = new ContainerModelCacheManager({
    rootDir: WORKSPACE_ROOT,
    autoPrefetchOnStart: false,
  });
  const middleware = manager.createMiddleware();

  await context.route('**/*', async (route) => {
    const reqUrl = new URL(route.request().url());
    if (reqUrl.origin !== 'http://127.0.0.1:4173') {
      return route.abort('blockedbyclient');
    }
    if (reqUrl.pathname.startsWith('/api/model-cache/')) {
      const result = await runCacheMiddleware(
        middleware,
        reqUrl.pathname + reqUrl.search,
        route.request().method(),
        route.request().headers()
      );
      if (result.handled) {
        return route.fulfill({
          status: result.status,
          headers: result.headers,
          body: result.body,
        });
      }
    }
    return route.continue();
  });

  await context.addInitScript(() => {
    (
      window as unknown as { __CONTAINER_MODEL_CACHE_ENABLED__?: boolean }
    ).__CONTAINER_MODEL_CACHE_ENABLED__ = true;
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: {
        getUserMedia: () =>
          Promise.reject(
            new DOMException(
              'Still-image verification mode: physical camera disabled',
              'NotAllowedError'
            )
          ),
      },
    });
  });
}

test.describe('Experimental Partial Mirage ONNX Browser Verification', () => {
  test.describe.configure({ mode: 'serial' });
  test.setTimeout(240_000);

  let part1Report: {
    schemaVersion: number;
    recordedAt: string;
    userAgent: string;
    artifact: Record<string, unknown>;
    executionProviders: {
      wasm: {
        variantId: string;
        sessionCreateMs: number;
        backend: OnnxBackendInfo;
      };
      webgpu: {
        backend: OnnxBackendInfo;
        applicationWarmupOutputLengths: number[];
        [key: string]: unknown;
      };
    };
    cases: Record<string, IdenticalInputCaseReport>;
  } | null = null;

  test('verifies SHA-256, WASM vs WebGPU execution providers, and identical-input tensor parity (atol=rtol=2e-4) across 8 cases', async ({
    context,
    page,
  }, testInfo) => {
    test.setTimeout(360_000);
    if (process.env.REQUIRE_MIRAGE_REAL_MODEL === '1') {
      expect(
        existsSync(ONNX_ARTIFACT_PATH),
        `Expected reconstructed ONNX at ${ONNX_ARTIFACT_PATH}`
      ).toBe(true);
      expect(
        existsSync(FIXTURE_MANIFEST_PATH),
        `Expected fixture manifest at ${FIXTURE_MANIFEST_PATH}`
      ).toBe(true);
    } else {
      test.skip(
        !existsSync(ONNX_ARTIFACT_PATH) || !existsSync(FIXTURE_MANIFEST_PATH),
        'Requires local verified cact-dequantized-reference-order.onnx and .cache/mirage-browser-fixtures/manifest.json'
      );
    }

    await routeLocalModelCache(context);
    await page.goto('/');
    await expect(page.getByRole('banner').getByText('VisionStudio AI')).toBeVisible();

    const browserReport = await page.evaluate(
      async ({ expectedSize, expectedSha256 }) => {
        const bridge = window.__mirageExperimentalBridge;
        if (!bridge) {
          throw new Error('window.__mirageExperimentalBridge is not registered');
        }
        const activeBridge = bridge;
        activeBridge.cancelInFlightMirageWasmInit?.();

        // 1. Fetch and verify the ONNX artifact in browser WebCrypto
        const onnxUrl = `/api/model-cache/mirage/cact-dequantized-reference-order.onnx?sha256=${expectedSha256}`;
        const onnxFetchStart = performance.now();
        const onnxRes = await fetch(onnxUrl);
        if (!onnxRes.ok) {
          throw new Error(`Failed to fetch ONNX artifact: HTTP ${onnxRes.status}`);
        }
        const onnxBuffer = await onnxRes.arrayBuffer();
        const onnxFetchMs = performance.now() - onnxFetchStart;
        const verified = await bridge.verifyMirageExperimentalOnnxBytes(onnxBuffer);

        // 2. Load the 8-case Python reference manifest
        const manifestRes = await fetch(
          '/api/model-cache/mirage/fixtures/manifest.json'
        );
        if (!manifestRes.ok) {
          throw new Error(
            `Failed to fetch fixture manifest: HTTP ${manifestRes.status}`
          );
        }
        const manifest = await manifestRes.json();

        // 3. Probe WebGPU availability & direct WebGPU EP behavior on the 64-output graph
        const adapterDiagnostics = await bridge.probeWebGpuAdapterDiagnostics();
        const webgpuAdapterAvailable = adapterDiagnostics.adapterAvailable;
        const webgpuDirectSessionOutcome: {
          attempted: boolean;
          sessionCreated: boolean;
          runSucceeded: boolean;
          createMs: number | null;
          runMs: number | null;
          error: string | null;
        } = {
          attempted: false,
          sessionCreated: false,
          runSucceeded: false,
          createMs: null,
          runMs: null,
          error: null,
        };

        // Ensure ort.env.wasm.wasmPaths points to the local container cache BEFORE any session creation
        bridge.ort.env.logLevel = 'error';
        bridge.ort.env.wasm.numThreads = 1;
        bridge.ort.env.wasm.wasmPaths = `${window.location.origin}/api/model-cache/wasm/ort/`;

        function compareFloat32Tensors(
          actual: Float32Array,
          expected: Float32Array,
          atol = 2e-4,
          rtol = 2e-4
        ) {
          return activeBridge.compareMirageTensorParity(
            actual,
            expected,
            atol,
            rtol,
            1e-6
          );
        }

        async function fetchFloat32(filename: string): Promise<Float32Array> {
          const r = await fetch(`/api/model-cache/mirage/fixtures/${filename}`);
          if (!r.ok) {
            throw new Error(`Failed to fetch ${filename}: HTTP ${r.status}`);
          }
          return new Float32Array(await r.arrayBuffer());
        }

        // Test direct ['webgpu'] session creation (no automatic WASM fallback)
        webgpuDirectSessionOutcome.attempted = true;
        const directStart = performance.now();
        try {
          const directSession = await bridge.ort.InferenceSession.create(
            new Uint8Array(onnxBuffer),
            {
              executionProviders: ['webgpu'],
              graphOptimizationLevel: 'all',
              logSeverityLevel: 3,
            }
          );
          webgpuDirectSessionOutcome.sessionCreated = true;
          webgpuDirectSessionOutcome.createMs = performance.now() - directStart;
          await directSession.release();
        } catch (createErr) {
          webgpuDirectSessionOutcome.createMs = performance.now() - directStart;
          webgpuDirectSessionOutcome.error = `WebGPU session create failed: ${
            createErr instanceof Error ? createErr.message : String(createErr)
          }`;
        }

        // Test strict task creation (allowWasmFallback: false)
        const strictWebGpuTaskOutcome: {
          status: 'PASSED' | 'PASSED_SOFTWARE_ADAPTER' | 'BLOCKED';
          error: string | null;
        } = {
          status: 'BLOCKED',
          error: null,
        };
        try {
          const strictTask = await bridge.createMirageExperimentalOnnxTask({
            variantId: 'mirage-onnx-exp-webgpu',
            allowWasmFallback: false,
          });
          strictWebGpuTaskOutcome.status = adapterDiagnostics.isSoftwareAdapter
            ? 'PASSED_SOFTWARE_ADAPTER'
            : 'PASSED';
          await strictTask.close();
        } catch (strictErr) {
          strictWebGpuTaskOutcome.status = 'BLOCKED';
          strictWebGpuTaskOutcome.error =
            strictErr instanceof Error ? strictErr.message : String(strictErr);
        }

        // 4. Test mirage-onnx-exp-webgpu via application task factory and verify full WebGPU Encoder parity on text_regions
        const webgpuTaskStart = performance.now();
        const webgpuTask = await bridge.createMirageExperimentalOnnxTask({
          variantId: 'mirage-onnx-exp-webgpu',
        });
        const webgpuTaskInitMs = performance.now() - webgpuTaskStart;
        const webgpuBackendInfo = webgpuTask.backend;

        const gpuCaseEntry = manifest.cases.text_regions;
        const [gpuImage, gpuRefDet, gpuRefProto, gpuRefCoeff] = await Promise.all([
          fetchFloat32(gpuCaseEntry.input_tensor_file),
          fetchFloat32(gpuCaseEntry.heads.detections.file),
          fetchFloat32(gpuCaseEntry.heads.prototypes.file),
          fetchFloat32(gpuCaseEntry.heads.coefficients.file),
        ]);
        const gpuMeta = bridge.computeMirageCeil32Metadata(
          gpuCaseEntry.metadata.W,
          gpuCaseEntry.metadata.H
        );
        const webgpuWarmupStart = performance.now();
        const webgpuBlankHeads = await webgpuTask.runRawHeads({
          image: gpuImage,
          valid_hw: [gpuCaseEntry.valid_hw[0], gpuCaseEntry.valid_hw[1]],
          metadata: gpuMeta,
        });
        const webgpuWarmupMs = performance.now() - webgpuWarmupStart;
        webgpuDirectSessionOutcome.runSucceeded =
          webgpuBlankHeads.detections.length === 40960 &&
          webgpuBlankHeads.prototypes.length === 524288 &&
          webgpuBlankHeads.coefficients.length === 131072;
        webgpuDirectSessionOutcome.runMs = Number(webgpuWarmupMs.toFixed(2));
        const webgpuDetDiff = compareFloat32Tensors(webgpuBlankHeads.detections, gpuRefDet);
        const webgpuProtoDiff = compareFloat32Tensors(webgpuBlankHeads.prototypes, gpuRefProto);
        const webgpuCoeffDiff = compareFloat32Tensors(webgpuBlankHeads.coefficients, gpuRefCoeff);
        const webgpuTextRegionsParity = {
          caseName: 'text_regions',
          detectionsAllcloseAt2e4: webgpuDetDiff.allcloseAt2e4,
          prototypesAllcloseAt2e4: webgpuProtoDiff.allcloseAt2e4,
          coefficientsAllcloseAt2e4: webgpuCoeffDiff.allcloseAt2e4,
          allThreeHeadsAllcloseAt2e4:
            webgpuDetDiff.allcloseAt2e4 &&
            webgpuProtoDiff.allcloseAt2e4 &&
            webgpuCoeffDiff.allcloseAt2e4,
          minCosineSimilarity: Math.min(
            webgpuDetDiff.cosineSimilarity,
            webgpuProtoDiff.cosineSimilarity,
            webgpuCoeffDiff.cosineSimilarity
          ),
          maxAbsDiff: Math.max(
            webgpuDetDiff.maxAbsDiff,
            webgpuProtoDiff.maxAbsDiff,
            webgpuCoeffDiff.maxAbsDiff
          ),
        };
        await webgpuTask.close();

        // 5. Initialize mirage-onnx-exp-wasm and run all 8 identical-input benchmark cases
        const wasmTaskStart = performance.now();
        const wasmTask = await bridge.createMirageExperimentalOnnxTask({
          variantId: 'mirage-onnx-exp-wasm',
        });
        const wasmTaskInitMs = performance.now() - wasmTaskStart;
        const wasmBackendInfo = wasmTask.backend;

        const caseReports: Record<string, IdenticalInputCaseReport> = {};
        const caseNames = Object.keys(manifest.cases);

        for (const caseName of caseNames) {
          const entry = manifest.cases[caseName];
          const [refImage, refDetections, refPrototypes, refCoefficients] =
            await Promise.all([
              fetchFloat32(`${caseName}.image.f32.bin`),
              fetchFloat32(entry.heads.detections.file),
              fetchFloat32(entry.heads.prototypes.file),
              fetchFloat32(entry.heads.coefficients.file),
            ]);

          const metadata = bridge.computeMirageCeil32Metadata(
            entry.metadata.W,
            entry.metadata.H
          );

          // Part A: Identical Input Tensor (Python Pillow float32 tensor -> Browser ORT WASM)
          const infStart = performance.now();
          const browserHeads = await wasmTask.runRawHeads({
            image: refImage,
            valid_hw: [entry.valid_hw[0], entry.valid_hw[1]],
            metadata,
          });
          const identicalInputInferenceMs = performance.now() - infStart;

          const detectionsDiff = compareFloat32Tensors(
            browserHeads.detections,
            refDetections
          );
          const prototypesDiff = compareFloat32Tensors(
            browserHeads.prototypes,
            refPrototypes
          );
          const coefficientsDiff = compareFloat32Tensors(
            browserHeads.coefficients,
            refCoefficients
          );

          const decodedNone = bridge.decodeMirageExperimentalOutputs(
            browserHeads,
            metadata,
            { confidence: 0.3, nmsIou: null, includeMasks: true }
          );
          const decodedNms050 = bridge.decodeMirageExperimentalOutputs(
            browserHeads,
            metadata,
            { confidence: 0.3, nmsIou: 0.5, includeMasks: false }
          );

          async function sha256HexOfUint8(arr?: Uint8Array): Promise<string | null> {
            if (!arr) return null;
            const copy = new Uint8Array(arr.byteLength);
            copy.set(arr);
            const buf = await crypto.subtle.digest('SHA-256', copy.buffer);
            return Array.from(new Uint8Array(buf))
              .map((b) => b.toString(16).padStart(2, '0'))
              .join('');
          }

          const browserMaskHashes = await Promise.all(
            decodedNone.objects.map((o) => sha256HexOfUint8(o.mask))
          );

          const exactProposalParityNone =
            decodedNone.objects.length === entry.decoded_default.object_count &&
            decodedNone.text.length === entry.decoded_default.text_count &&
            decodedNone.objects.every(
              (o, idx) =>
                o.cell[0] === entry.decoded_default.objects[idx].cell[0] &&
                o.cell[1] === entry.decoded_default.objects[idx].cell[1] &&
                o.bbox.every(
                  (v, bi) => v === entry.decoded_default.objects[idx].bbox[bi]
                ) &&
                Math.abs(o.score - entry.decoded_default.objects[idx].score) <
                  1e-5 &&
                o.maskActivePixels ===
                  entry.decoded_default.objects[idx].mask_active_pixels &&
                (!entry.decoded_default.objects[idx].mask_sha256 ||
                  browserMaskHashes[idx] ===
                    entry.decoded_default.objects[idx].mask_sha256)
            ) &&
            decodedNone.text.every(
              (t, idx) =>
                t.cell[0] === entry.decoded_default.text[idx].cell[0] &&
                t.cell[1] === entry.decoded_default.text[idx].cell[1] &&
                t.bbox.every(
                  (v, bi) => v === entry.decoded_default.text[idx].bbox[bi]
                ) &&
                Math.abs(t.score - entry.decoded_default.text[idx].score) < 1e-5
            );

          const exactProposalParityNms050 =
            decodedNms050.objects.length === entry.decoded_nms_0p50.object_count &&
            decodedNms050.text.length === entry.decoded_nms_0p50.text_count &&
            decodedNms050.objects.every(
              (o, idx) =>
                o.cell[0] === entry.decoded_nms_0p50.objects[idx].cell[0] &&
                o.cell[1] === entry.decoded_nms_0p50.objects[idx].cell[1] &&
                o.bbox.every(
                  (v, bi) => v === entry.decoded_nms_0p50.objects[idx].bbox[bi]
                )
            );

          caseReports[caseName] = {
            sourceDimensions: [entry.metadata.W, entry.metadata.H],
            validHw: entry.valid_hw,
            coordinateScale: metadata.coordinate_scale,
            identicalInputParity: {
              inferenceMs: Number(identicalInputInferenceMs.toFixed(2)),
              detections: detectionsDiff,
              prototypes: prototypesDiff,
              coefficients: coefficientsDiff,
              allThreeHeadsAllcloseAt2e4:
                detectionsDiff.allcloseAt2e4 &&
                prototypesDiff.allcloseAt2e4 &&
                coefficientsDiff.allcloseAt2e4,
              exactProposalParityNone,
              exactProposalParityNms050,
              objectCountNoNms: decodedNone.objects.length,
              textCountNoNms: decodedNone.text.length,
              objectCountNms050: decodedNms050.objects.length,
              textCountNms050: decodedNms050.text.length,
            },
          };
        }

        await wasmTask.close();

        return {
          schemaVersion: 1,
          recordedAt: new Date().toISOString(),
          userAgent: navigator.userAgent,
          artifact: {
            url: onnxUrl,
            size: verified.size,
            sha256: verified.sha256,
            fetchMs: Number(onnxFetchMs.toFixed(2)),
            matchesExpectedSize: verified.size === expectedSize,
            matchesExpectedSha256: verified.sha256 === expectedSha256,
          },
          executionProviders: {
            wasm: {
              variantId: 'mirage-onnx-exp-wasm',
              sessionCreateMs: Number(wasmTaskInitMs.toFixed(2)),
              backend: wasmBackendInfo,
            },
            webgpu: {
              variantId: 'mirage-onnx-exp-webgpu',
              navigatorGpuPresent: adapterDiagnostics.navigatorGpuPresent,
              webgpuAdapterAvailable,
              adapterDiagnostics,
              directWebGpuSession: webgpuDirectSessionOutcome,
              strictWebGpuTask: strictWebGpuTaskOutcome,
              applicationTaskInitMs: Number(webgpuTaskInitMs.toFixed(2)),
              applicationWarmupInferenceMs: Number(webgpuWarmupMs.toFixed(2)),
              applicationWarmupOutputLengths: [
                webgpuBlankHeads.detections.length,
                webgpuBlankHeads.prototypes.length,
                webgpuBlankHeads.coefficients.length,
              ],
              webgpuTextRegionsParity,
              backend: webgpuBackendInfo,
            },
          },
          cases: caseReports,
        };
      },
      {
        expectedSize: MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE,
        expectedSha256: MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256,
      }
    );

    await page.goto('about:blank');

    part1Report = browserReport;
    writeFileSync(
      testInfo.outputPath('mirage-browser-onnx-identical-results.json'),
      JSON.stringify(browserReport, null, 2) + '\n',
      'utf8'
    );

    // Assertions on artifact verification
    expect(browserReport.artifact.matchesExpectedSize).toBe(true);
    expect(browserReport.artifact.matchesExpectedSha256).toBe(true);
    expect(browserReport.artifact.size).toBe(34_281_503);
    expect(browserReport.artifact.sha256).toBe(
      '52d52d2918160404cd806702a270736819712025cf3c6aa8e907a06bc3cbdd07'
    );

    // Assertions on WASM and WebGPU EP reporting
    expect(
      browserReport.executionProviders.wasm.backend.requestedProvider
    ).toBe('wasm');
    expect(
      browserReport.executionProviders.wasm.backend.configuredProviders
    ).toEqual(['wasm']);
    expect(
      browserReport.executionProviders.webgpu.backend.requestedProvider
    ).toBe('webgpu');
    expect(
      browserReport.executionProviders.webgpu.webgpuAdapterAvailable
    ).toBe(true);
    expect(
      browserReport.executionProviders.webgpu.directWebGpuSession.sessionCreated
    ).toBe(true);
    expect(
      browserReport.executionProviders.webgpu.directWebGpuSession.runSucceeded
    ).toBe(true);
    expect(['PASSED', 'PASSED_SOFTWARE_ADAPTER']).toContain(
      browserReport.executionProviders.webgpu.strictWebGpuTask.status
    );
    expect(
      browserReport.executionProviders.webgpu.backend.configuredProviders
    ).toContain('webgpu');
    expect(
      browserReport.executionProviders.webgpu.applicationWarmupOutputLengths
    ).toEqual([64 * 64 * 10, 32 * 128 * 128, 32 * 64 * 64]);
    expect(
      browserReport.executionProviders.webgpu.webgpuTextRegionsParity
        .allThreeHeadsAllcloseAt2e4
    ).toBe(true);

    // Assertions on all 8 benchmark cases in real browser WASM
    const cases = browserReport.cases;
    expect(Object.keys(cases)).toHaveLength(8);
    for (const [caseName, report] of Object.entries(cases)) {
      expect(
        report.identicalInputParity.allThreeHeadsAllcloseAt2e4,
        `${caseName}: browser WASM raw heads allclose(atol=rtol=2e-4) to Python ORT CPU`
      ).toBe(true);
      expect(
        report.identicalInputParity.exactProposalParityNone,
        `${caseName}: browser decoded proposals (nms=null) match Python 100%`
      ).toBe(true);
      expect(
        report.identicalInputParity.exactProposalParityNms050,
        `${caseName}: browser decoded proposals (nms=0.5) match Python 100%`
      ).toBe(true);
    }
  });

  test('verifies Canvas 2D vs Pillow ceil32 preprocessing, zero padding, and browser WASM inference parity across 8 cases', async ({
    context,
    page,
  }, testInfo) => {
    if (process.env.REQUIRE_MIRAGE_REAL_MODEL === '1') {
      expect(
        existsSync(ONNX_ARTIFACT_PATH),
        `Expected reconstructed ONNX at ${ONNX_ARTIFACT_PATH}`
      ).toBe(true);
      expect(
        existsSync(FIXTURE_MANIFEST_PATH),
        `Expected fixture manifest at ${FIXTURE_MANIFEST_PATH}`
      ).toBe(true);
    } else {
      test.skip(
        !existsSync(ONNX_ARTIFACT_PATH) || !existsSync(FIXTURE_MANIFEST_PATH),
        'Requires local verified cact-dequantized-reference-order.onnx and .cache/mirage-browser-fixtures/manifest.json'
      );
    }

    await routeLocalModelCache(context);
    await page.goto('/');
    await expect(page.getByRole('banner').getByText('VisionStudio AI')).toBeVisible();

    const preprocessReport = await page.evaluate(async () => {
      const bridge = window.__mirageExperimentalBridge;
      if (!bridge) {
        throw new Error('window.__mirageExperimentalBridge is not registered');
      }
      bridge.cancelInFlightMirageWasmInit?.();

      bridge.ort.env.logLevel = 'error';
      bridge.ort.env.wasm.numThreads = 1;
      bridge.ort.env.wasm.wasmPaths = `${window.location.origin}/api/model-cache/wasm/ort/`;

      const manifestRes = await fetch(
        '/api/model-cache/mirage/fixtures/manifest.json'
      );
      if (!manifestRes.ok) {
        throw new Error(
          `Failed to fetch fixture manifest: HTTP ${manifestRes.status}`
        );
      }
      const manifest = await manifestRes.json();

      const wasmTask = await bridge.createMirageExperimentalOnnxTask({
        variantId: 'mirage-onnx-exp-wasm',
      });

      function boxIou(a: number[], b: number[]): number {
        const ix1 = Math.max(a[0], b[0]);
        const iy1 = Math.max(a[1], b[1]);
        const ix2 = Math.min(a[2], b[2]);
        const iy2 = Math.min(a[3], b[3]);
        const iw = Math.max(0, ix2 - ix1);
        const ih = Math.max(0, iy2 - iy1);
        const inter = iw * ih;
        const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
        const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
        return inter / Math.max(areaA + areaB - inter, 1e-9);
      }

      async function fetchFloat32(filename: string): Promise<Float32Array> {
        const r = await fetch(`/api/model-cache/mirage/fixtures/${filename}`);
        if (!r.ok) {
          throw new Error(`Failed to fetch ${filename}: HTTP ${r.status}`);
        }
        return new Float32Array(await r.arrayBuffer());
      }

      async function loadPngElement(filename: string): Promise<HTMLImageElement> {
        return new Promise((resolve, reject) => {
          const img = new Image();
          img.onload = () => resolve(img);
          img.onerror = () => reject(new Error(`Failed to load ${filename}`));
          img.src = `/api/model-cache/mirage/fixtures/${filename}`;
        });
      }

      const caseReports: Record<string, {
        sourceDimensions: [number, number];
        validHw: [number, number];
        coordinateScale: [number, number, number, number];
        canvasVsPillowPreprocess: {
          preprocessMs: number;
          inferenceMs: number;
          resizeOccurred: boolean;
          validRegionMaxAbsDiff: number;
          validRegionMeanAbsDiff: number;
          validRegionDiffRatio: number;
          paddingMaxAbsValue: number;
          canvasObjectCountNoNms: number;
          canvasTextCountNoNms: number;
          canvasObjectCountNms050: number;
          canvasTextCountNms050: number;
          matchedPillowObjectsAtIoU085: number;
          totalPillowObjectsNoNms: number;
        };
      }> = {};

      for (const caseName of Object.keys(manifest.cases)) {
        const entry = manifest.cases[caseName];
        const refImage = await fetchFloat32(`${caseName}.image.f32.bin`);
        const metadata = bridge.computeMirageCeil32Metadata(
          entry.metadata.W,
          entry.metadata.H
        );

        const imgEl = await loadPngElement(`${caseName}.png`);
        const preStart = performance.now();
        const canvasPreprocessed = wasmTask.preprocessInput(imgEl);
        const canvasPreprocessMs = performance.now() - preStart;

        const validH = metadata.h;
        const validW = metadata.w;
        let validRegionMaxAbsDiff = 0;
        let validRegionSumAbsDiff = 0;
        let validRegionDiffCountOver1e4 = 0;
        let validElements = 0;
        let paddingMaxAbsValue = 0;

        for (let c = 0; c < 3; c++) {
          const planeOffset = c * 512 * 512;
          for (let y = 0; y < 512; y++) {
            const rowOffset = planeOffset + y * 512;
            for (let x = 0; x < 512; x++) {
              const idx = rowOffset + x;
              const canvasVal = canvasPreprocessed.image[idx];
              const pillowVal = refImage[idx];
              if (y < validH && x < validW) {
                validElements++;
                const diff = Math.abs(canvasVal - pillowVal);
                validRegionSumAbsDiff += diff;
                if (diff > validRegionMaxAbsDiff) validRegionMaxAbsDiff = diff;
                if (diff > 1e-4) validRegionDiffCountOver1e4++;
              } else {
                const absPad = Math.abs(canvasVal);
                if (absPad > paddingMaxAbsValue) paddingMaxAbsValue = absPad;
              }
            }
          }
        }

        const canvasInfStart = performance.now();
        const canvasHeads = await wasmTask.runRawHeads(canvasPreprocessed);
        const canvasInferenceMs = performance.now() - canvasInfStart;
        const canvasDecodedNone = bridge.decodeMirageExperimentalOutputs(
          canvasHeads,
          metadata,
          { confidence: 0.3, nmsIou: null, includeMasks: false }
        );
        const canvasDecodedNms050 = bridge.decodeMirageExperimentalOutputs(
          canvasHeads,
          metadata,
          { confidence: 0.3, nmsIou: 0.5, includeMasks: false }
        );

        let matchedObjectsIoU085 = 0;
        for (const refObj of entry.decoded_default.objects as Array<{ bbox: number[] }>) {
          const bestIou = canvasDecodedNone.objects.reduce(
            (best, cand) => Math.max(best, boxIou(refObj.bbox, cand.bbox)),
            0
          );
          if (bestIou >= 0.85) matchedObjectsIoU085++;
        }

        caseReports[caseName] = {
          sourceDimensions: [entry.metadata.W, entry.metadata.H],
          validHw: entry.valid_hw,
          coordinateScale: metadata.coordinate_scale,
          canvasVsPillowPreprocess: {
            preprocessMs: Number(canvasPreprocessMs.toFixed(2)),
            inferenceMs: Number(canvasInferenceMs.toFixed(2)),
            resizeOccurred:
              entry.metadata.W !== metadata.w || entry.metadata.H !== metadata.h,
            validRegionMaxAbsDiff: Number(validRegionMaxAbsDiff.toFixed(6)),
            validRegionMeanAbsDiff: Number(
              (validRegionSumAbsDiff / Math.max(1, validElements)).toFixed(6)
            ),
            validRegionDiffRatio: Number(
              (
                validRegionDiffCountOver1e4 / Math.max(1, validElements)
              ).toFixed(4)
            ),
            paddingMaxAbsValue,
            canvasObjectCountNoNms: canvasDecodedNone.objects.length,
            canvasTextCountNoNms: canvasDecodedNone.text.length,
            canvasObjectCountNms050: canvasDecodedNms050.objects.length,
            canvasTextCountNms050: canvasDecodedNms050.text.length,
            matchedPillowObjectsAtIoU085: matchedObjectsIoU085,
            totalPillowObjectsNoNms: entry.decoded_default.object_count,
          },
        };
      }

      await wasmTask.close();
      return caseReports;
    });

    await page.goto('about:blank');

    const mergedCases: Record<string, unknown> = {};
    for (const [caseName, preEntry] of Object.entries(preprocessReport)) {
      const part1Case = part1Report?.cases?.[caseName];
      mergedCases[caseName] = {
        ...(part1Case ?? preEntry),
        canvasVsPillowPreprocess: preEntry.canvasVsPillowPreprocess,
      };
    }

    const mergedReport = part1Report
      ? { ...part1Report, cases: mergedCases }
      : { schemaVersion: 1, recordedAt: new Date().toISOString(), cases: mergedCases };

    const outputReportPath = testInfo.outputPath(
      'mirage-browser-onnx-results.json'
    );
    writeFileSync(
      outputReportPath,
      JSON.stringify(mergedReport, null, 2) + '\n',
      'utf8'
    );
    if (process.env.WRITE_MIRAGE_BROWSER_REPORT === '1') {
      const docsReportPath = path.resolve(
        WORKSPACE_ROOT,
        'docs/verification/mirage-browser-onnx-results.json'
      );
      writeFileSync(
        docsReportPath,
        JSON.stringify(mergedReport, null, 2) + '\n',
        'utf8'
      );
    }

    expect(Object.keys(preprocessReport)).toHaveLength(8);
    for (const [caseName, report] of Object.entries(preprocessReport)) {
      expect(
        report.canvasVsPillowPreprocess.paddingMaxAbsValue,
        `${caseName}: zero padding outside valid_hw`
      ).toBe(0);
    }

    // For 512x512 native inputs (no resize), Canvas 2D and Pillow RGB pixels must be identical
    expect(
      preprocessReport.overlap_shapes.canvasVsPillowPreprocess.resizeOccurred
    ).toBe(false);
    expect(
      preprocessReport.overlap_shapes.canvasVsPillowPreprocess.validRegionMaxAbsDiff
    ).toBeLessThanOrEqual(1e-6);
  });

  test('renders experimental partial Mirage ONNX proposals, masks, NMS toggle, and optional native WASM overlay in the VisionStudio AI UI', async ({
    context,
    page,
  }, testInfo) => {
    test.setTimeout(360_000);
    if (process.env.REQUIRE_MIRAGE_REAL_MODEL === '1') {
      expect(
        existsSync(ONNX_ARTIFACT_PATH),
        `Expected reconstructed ONNX at ${ONNX_ARTIFACT_PATH}`
      ).toBe(true);
      expect(
        existsSync(FIXTURE_MANIFEST_PATH),
        `Expected fixture manifest at ${FIXTURE_MANIFEST_PATH}`
      ).toBe(true);
    } else {
      test.skip(
        !existsSync(ONNX_ARTIFACT_PATH) || !existsSync(FIXTURE_MANIFEST_PATH),
        'Requires local verified cact-dequantized-reference-order.onnx and .cache/mirage-browser-fixtures/manifest.json'
      );
    }

    await routeLocalModelCache(context);
    await page.goto('/');

    const header = page.getByRole('banner');
    const modelSelect = header.getByRole('combobox', {
      name: 'モデル',
      exact: true,
    });
    await modelSelect.selectOption('mirage');

    // Default variant remains mirage-51m; mirage-webgpu remains disabled
    const variantSelect = header.getByRole('combobox', {
      name: 'バリアント',
      exact: true,
    });
    await expect(variantSelect).toHaveValue('mirage-51m');
    await expect(
      variantSelect.locator('option[value="mirage-webgpu"]')
    ).toBeDisabled();

    // Select experimental WASM variant
    await variantSelect.selectOption('mirage-onnx-exp-wasm');
    await expect(variantSelect).toHaveValue('mirage-onnx-exp-wasm');

    // Experimental limitation banner is visible in ControlPanel
    await expect(
      page.getByText('【実験用機能】部分Mirage ONNX 領域候補推論')
    ).toBeVisible();
    await expect(
      page.getByText('452/583 tensors · opset 17')
    ).toBeVisible();

    // Wait for session initialization
    await expect(page.getByRole('status')).toContainText('モデル準備完了', {
      timeout: 60_000,
    });
    await expect(header.getByLabel('バックエンド状態')).toContainText(
      '実験用部分ONNX 要求: WASM'
    );

    // 1. Run preset: overlap_shapes.png
    await page.getByRole('button', { name: '検証: 重なり図形' }).click();
    await expect(
      page.getByText('[ONNX候補] 物体領域候補 #1').first()
    ).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText('認識結果 (4件)')).toBeVisible();

    // Switch to モデル調整 tab to inspect live diagnostics and test NMS policy buttons
    await page.getByRole('button', { name: 'モデル調整' }).click();
    const diagBox = page.getByLabel('実験用部分ONNX実行診断');
    await expect(diagBox).toContainText('要求EP: wasm ／ 設定EP: wasm');
    await expect(diagBox).toContainText('抽出候補: 物体 4件 ／ テキスト 0件');

    // Toggle NMS to Greedy NMS (IoU 0.5) -> 4 overlapping candidates reduce to 3
    await page.getByRole('button', { name: /Greedy NMS \(IoU 0\.5\)/ }).click();
    await expect(diagBox).toContainText('抽出候補: 物体 3件 ／ テキスト 0件', {
      timeout: 30_000,
    });
    await page.getByRole('button', { name: /NMSなし（既定）/ }).click();
    await expect(diagBox).toContainText('抽出候補: 物体 4件 ／ テキスト 0件', {
      timeout: 30_000,
    });

    // 2. Run preset: text_regions.png
    await page.getByRole('button', { name: '検証: テキスト領域' }).click();
    await expect(diagBox).toContainText(
      '入力 valid_hw: [256, 512] (元画像 512×256 · ceil32)',
      { timeout: 30_000 }
    );
    await expect(diagBox).toContainText('抽出候補: 物体 2件 ／ テキスト 0件', {
      timeout: 30_000,
    });
    await page.getByRole('button', { name: /認識物体/ }).click();
    await expect(
      page.getByText('[ONNX候補] 物体領域候補 #1').first()
    ).toBeVisible({ timeout: 30_000 });

    // 3. Run preset: coco_000000458054.png (640x426 -> valid_hw [352, 512])
    await page.getByRole('button', { name: '検証: COCO 458054' }).click();
    await page.getByRole('button', { name: 'モデル調整' }).click();
    await expect(diagBox).toContainText(
      '入力 valid_hw: [352, 512] (元画像 640×426 · ceil32)',
      { timeout: 30_000 }
    );
    await expect(diagBox).toContainText('抽出候補: 物体 17件 ／ テキスト 0件');

    await page.screenshot({
      path: testInfo.outputPath('mirage-experimental-onnx-ui-coco.png'),
      fullPage: true,
    });

    // 4. Switch to experimental WebGPU variant and verify initialization + EP diagnostics
    await page.getByRole('button', { name: 'カメラ映像' }).click();
    await variantSelect.selectOption('mirage-onnx-exp-webgpu');
    await expect(page.getByRole('status')).toContainText('モデル準備完了', {
      timeout: 60_000,
    });
    await expect(diagBox).toContainText('要求EP: webgpu ／ 設定EP: webgpu', {
      timeout: 30_000,
    });
    await page.goto('about:blank');
  });

  test('verifies R3 same-frame source snapshot and live stream isolation in browser', async ({
    context,
    page,
  }) => {
    if (process.env.REQUIRE_MIRAGE_REAL_MODEL === '1') {
      expect(
        existsSync(ONNX_ARTIFACT_PATH),
        `Expected reconstructed ONNX at ${ONNX_ARTIFACT_PATH}`
      ).toBe(true);
    } else {
      test.skip(
        !existsSync(ONNX_ARTIFACT_PATH),
        'Requires local verified cact-dequantized-reference-order.onnx'
      );
    }

    await routeLocalModelCache(context);
    await page.goto('/');
    await expect(page.getByRole('banner').getByText('VisionStudio AI')).toBeVisible();

    const sameFrameAlignment = await page.evaluate(async () => {
      const bridge = window.__mirageExperimentalBridge;
      if (!bridge) {
        throw new Error('window.__mirageExperimentalBridge is not registered');
      }
      bridge.cancelInFlightMirageWasmInit?.();
      bridge.ort.env.logLevel = 'error';
      bridge.ort.env.wasm.numThreads = 1;
      bridge.ort.env.wasm.wasmPaths = `${window.location.origin}/api/model-cache/wasm/ort/`;

      let sameFrameNativeReceivedSha256: string | null = null;
      const alignmentTask = await bridge.createMirageExperimentalOnnxTask({
        variantId: 'mirage-onnx-exp-wasm',
        nativeOverlayTaskFactory: async () => ({
          detectForVideo: async (
            _input: unknown,
            _threshold: number,
            extra?: MirageInferExtraOptions
          ) => {
            sameFrameNativeReceivedSha256 =
              extra?.sourceSnapshot?.rgbaSha256 ?? null;
            return [
              {
                id: 'native-align-1',
                label: 'overlap',
                labelJa: '重なり図形',
                category: '図形',
                confidence: 0.9,
                bbox: { x: 0.1, y: 0.1, width: 0.5, height: 0.5 },
                color: '#10b981',
              },
            ];
          },
          close: () => {},
        }),
      });
      const liveStreamCanvas = document.createElement('canvas');
      liveStreamCanvas.width = 128;
      liveStreamCanvas.height = 128;
      const liveStreamCtx = liveStreamCanvas.getContext('2d')!;
      liveStreamCtx.fillStyle = 'rgb(25, 100, 200)';
      liveStreamCtx.fillRect(0, 0, 128, 128);
      const frame1Rgba = liveStreamCtx.getImageData(0, 0, 128, 128).data;
      const frame1ExpectedSha256 = bridge.sha256HexSync(frame1Rgba);

      const alignmentPromise = alignmentTask.detectForVideo(
        liveStreamCanvas,
        0.3,
        { overlayNativeWasm: true, sourceGeneration: 19 }
      );
      // Immediately mutate the live canvas to Frame 2 while ONNX inference is running
      liveStreamCtx.fillStyle = 'rgb(250, 10, 10)';
      liveStreamCtx.fillRect(0, 0, 128, 128);
      const frame2Rgba = liveStreamCtx.getImageData(0, 0, 128, 128).data;
      const frame2Sha256 = bridge.sha256HexSync(frame2Rgba);

      await alignmentPromise;
      const diagnostics =
        alignmentTask.getLastDiagnostics()?.frameAlignment ?? null;
      await alignmentTask.close();

      return {
        frame1ExpectedSha256,
        frame2Sha256,
        nativeReceivedSha256: sameFrameNativeReceivedSha256,
        diagnostics,
      };
    });

    await page.goto('about:blank');

    expect(sameFrameAlignment.frame1ExpectedSha256).not.toBe(
      sameFrameAlignment.frame2Sha256
    );
    expect(sameFrameAlignment.nativeReceivedSha256).toBe(
      sameFrameAlignment.frame1ExpectedSha256
    );
    expect(sameFrameAlignment.diagnostics).toMatchObject({
      onnxSourceGeneration: 19,
      nativeSourceGeneration: 19,
      onnxRgbaSha256: sameFrameAlignment.frame1ExpectedSha256,
      nativeRgbaSha256: sameFrameAlignment.frame1ExpectedSha256,
      sameSourceFrame: true,
    });
  });

  test('verifies full 4-graph Mirage browser WASM parity (Reader OCR, CrossKV, step0 logits, autoregressive tokens, assembly) and strict WebGPU probe', async ({
    context,
    page,
  }, testInfo) => {
    test.setTimeout(360_000);
    const fullRefPath = path.resolve(
      WORKSPACE_ROOT,
      '.cache/mirage-browser-fixtures/full_pipeline_reference.json'
    );
    const cactPath = path.resolve(
      WORKSPACE_ROOT,
      '.cache/container-models/mirage.cact'
    );
    const hasAllArtifacts =
      existsSync(ONNX_ARTIFACT_PATH) &&
      existsSync(FIXTURE_MANIFEST_PATH) &&
      existsSync(fullRefPath) &&
      existsSync(cactPath);

    if (process.env.REQUIRE_MIRAGE_REAL_MODEL === '1') {
      expect(
        hasAllArtifacts,
        `Expected ONNX, fixtures, full_pipeline_reference.json, and mirage.cact under ${WORKSPACE_ROOT}/.cache`
      ).toBe(true);
    } else if (!hasAllArtifacts) {
      test.skip(true, 'Real Mirage ONNX/CACT/full-pipeline reference artifacts not present locally');
    }

    await routeLocalModelCache(context);
    await page.goto('/');

    interface RefLangEntry {
      generatedTokenIds: number[];
      decodedText: string;
      step0LogitsFile: string;
      step0LogitsSha256: string;
    }

    interface RefCaseEntry {
      confidence: number;
      nmsIou: number;
      objectCount: number;
      textCount: number;
      ocrLines: string[];
      ckFile: string;
      cvFile: string;
      languages: Record<'en' | 'fr' | 'es', RefLangEntry>;
    }

    interface FullRefManifest {
      schemaVersion: number;
      cases: Record<string, RefCaseEntry>;
    }

    const report = await page.evaluate(async () => {
      const bridge = window.__mirageExperimentalBridge;
      if (!bridge) {
        throw new Error('Mirage experimental bridge missing on window');
      }
      bridge.cancelInFlightMirageWasmInit?.();
      bridge.ort.env.logLevel = 'error';
      bridge.ort.env.wasm.numThreads = 1;
      bridge.ort.env.wasm.wasmPaths = `${window.location.origin}/api/model-cache/wasm/ort/`;

      const [manifestRes, fullRefRes] = await Promise.all([
        fetch('/api/model-cache/mirage/fixtures/manifest.json'),
        fetch('/api/model-cache/mirage/fixtures/full_pipeline_reference.json'),
      ]);
      if (!manifestRes.ok || !fullRefRes.ok) {
        throw new Error(
          `Failed to load fixture manifests: ${manifestRes.status}, ${fullRefRes.status}`
        );
      }
      const manifest = (await manifestRes.json()) as {
        cases: Record<
          string,
          {
            png_file: string;
            input_tensor_file: string;
            valid_hw: [number, number];
            metadata: { W: number; H: number };
          }
        >;
      };
      const fullRef = (await fullRefRes.json()) as FullRefManifest;

      async function fetchFloat32(filename: string): Promise<Float32Array> {
        const res = await fetch(`/api/model-cache/mirage/fixtures/${filename}`);
        if (!res.ok) {
          throw new Error(`Failed to fetch ${filename}: HTTP ${res.status}`);
        }
        return new Float32Array(await res.arrayBuffer());
      }

      async function loadPngElement(filename: string): Promise<HTMLImageElement> {
        return new Promise((resolve, reject) => {
          const img = new Image();
          img.onload = () => resolve(img);
          img.onerror = () => reject(new Error(`Failed to load ${filename}`));
          img.src = `/api/model-cache/mirage/fixtures/${filename}`;
        });
      }

      function extractNativeRgba(img: HTMLImageElement, w: number, h: number): Uint8ClampedArray {
        const c = document.createElement('canvas');
        c.width = w;
        c.height = h;
        const ctx = c.getContext('2d');
        if (!ctx) throw new Error('2D context unavailable');
        ctx.drawImage(img, 0, 0, w, h);
        return ctx.getImageData(0, 0, w, h).data;
      }

      const fullAcquireStart = performance.now();
      const fullAssets = await bridge.acquireMirageFullPipelineAssets({
        variantId: 'mirage-onnx-exp-wasm',
        delegate: 'WASM',
        allowWasmFallback: true,
      });
      const fullAcquireMs = performance.now() - fullAcquireStart;

      const wasmTask = await bridge.createMirageExperimentalOnnxTask({
        variantId: 'mirage-onnx-exp-wasm',
        fullPipeline: fullAssets,
      });

      const fullDecodeMatrix: Record<string, ReadonlyArray<'en' | 'fr' | 'es'>> = {
        text_regions: ['en', 'fr', 'es'],
        overlap_shapes: ['fr'],
        coco_000000331352: ['en', 'fr'],
        coco_000000458054: ['en'],
      };

      interface LangReport {
        step0LogitsAllcloseAt2e4: boolean;
        step0LogitsMaxAbsDiff: number;
        step0LogitsViolationsAt2e4: number;
        step0LogitsCosineSimilarity: number;
        fullSequenceTested: boolean;
        tokenIdsExactMatch: boolean | null;
        decodedTextExactMatch: boolean | null;
        generatedTokenCount: number;
        stepsExecuted: number;
        decodeMs: number;
        assembledObjectCount?: number;
        assembledRegionCount?: number;
        assembledTextCount?: number;
        assembledRelationCount?: number;
      }

      const caseReports: Record<
        string,
        {
          objectCount: number;
          textCount: number;
          ocrLinesActual: string[];
          ocrLinesExpected: string[];
          ocrExactMatch: boolean;
          ckAllcloseAt2e4: boolean;
          cvAllcloseAt2e4: boolean;
          ckMaxAbsDiff: number;
          cvMaxAbsDiff: number;
          cvViolationsAt2e4: number;
          cvLength: number;
          cvViolationRate: number;
          cvCosineSimilarity: number;
          languages: Record<string, LangReport>;
        }
      > = {};

      const cachedInputs: Record<
        string,
        {
          memory: Float32Array;
          dense_features: Float32Array;
          metadata: ReturnType<typeof bridge.computeMirageCeil32Metadata>;
          decoded: ReturnType<typeof bridge.decodeMirageExperimentalOutputs>;
          nativeRgba: Uint8ClampedArray;
          W: number;
          H: number;
        }
      > = {};

      for (const caseName of Object.keys(fullRef.cases)) {
        const refCase = fullRef.cases[caseName];
        const fixtEntry = manifest.cases[caseName];
        const metadata = bridge.computeMirageCeil32Metadata(
          fixtEntry.metadata.W,
          fixtEntry.metadata.H
        );
        const image = await fetchFloat32(fixtEntry.input_tensor_file);
        const rawHeads = await wasmTask.runRawHeads(
          { image, valid_hw: fixtEntry.valid_hw, metadata },
          { includeDecoderFeatures: true }
        );
        if (!rawHeads.memory || !rawHeads.dense_features) {
          throw new Error(`Encoder omitted memory/dense_features for ${caseName}`);
        }

        const decoded = bridge.decodeMirageExperimentalOutputs(rawHeads, metadata, {
          confidence: 0.3,
          nmsIou: 0.5,
          maxObjects: 20,
          maxText: 20,
          includeMasks: true,
        });

        const imgEl = await loadPngElement(fixtEntry.png_file);
        const nativeRgba = extractNativeRgba(imgEl, fixtEntry.metadata.W, fixtEntry.metadata.H);
        if (caseName === 'text_regions' || caseName === 'coco_000000331352') {
          cachedInputs[caseName] = {
            memory: rawHeads.memory,
            dense_features: rawHeads.dense_features,
            metadata,
            decoded,
            nativeRgba,
            W: fixtEntry.metadata.W,
            H: fixtEntry.metadata.H,
          };
        }
        const ocrLinesActual = await bridge.readMirageTextLines(
          fullAssets.readerSession,
          nativeRgba,
          fixtEntry.metadata.W,
          fixtEntry.metadata.H,
          decoded.text,
          fullAssets.tableReader.charset
        );

        const crossKv = await bridge.runMirageCrossKv(
          fullAssets.crossKvSession,
          rawHeads.memory,
          rawHeads.dense_features,
          metadata,
          decoded.objects,
          decoded.text,
          fullAssets.tableReader
        );
        const [refCk, refCv] = await Promise.all([
          fetchFloat32(refCase.ckFile),
          fetchFloat32(refCase.cvFile),
        ]);
        const ckParity = bridge.compareMirageTensorParity(crossKv.crossK, refCk, 2e-4, 2e-4, 1e-4);
        const cvParity = bridge.compareMirageTensorParity(crossKv.crossV, refCv, 2e-4, 2e-4, 1e-4);

        const fullLangs = fullDecodeMatrix[caseName] ?? [];
        const langReports: Record<string, LangReport> = {};

        for (const lang of ['en', 'fr', 'es'] as const) {
          const refLang = refCase.languages[lang];
          const runFull = fullLangs.includes(lang);
          const decodeStart = performance.now();
          const decRes = await bridge.runMirageGreedyDecode(
            fullAssets.decoderStepSession,
            crossKv,
            fullAssets.tableReader,
            {
              language: lang,
              maxTokens: runFull ? 128 : 0,
              captureStep0Logits: true,
            }
          );
          const decodeMs = performance.now() - decodeStart;
          if (!decRes.step0Logits) {
            throw new Error(`Missing step0Logits on ${caseName}/${lang}`);
          }
          const refStep0 = await fetchFloat32(refLang.step0LogitsFile);
          const step0Parity = bridge.compareMirageTensorParity(
            decRes.step0Logits,
            refStep0,
            2e-4,
            2e-4,
            1e-4
          );

          let tokenIdsExactMatch: boolean | null = null;
          let decodedTextExactMatch: boolean | null = null;
          let assembledCounts:
            | {
                assembledObjectCount: number;
                assembledRegionCount: number;
                assembledTextCount: number;
                assembledRelationCount: number;
              }
            | undefined;

          if (runFull) {
            tokenIdsExactMatch =
              decRes.generatedTokenIds.length === refLang.generatedTokenIds.length &&
              decRes.generatedTokenIds.every((id, idx) => id === refLang.generatedTokenIds[idx]);
            decodedTextExactMatch = decRes.decodedText === refLang.decodedText;
            const obs = bridge.assembleMirageObservation({
              decodedText: decRes.decodedText,
              generatedTokenIds: decRes.generatedTokenIds,
              language: lang,
              objects: decoded.objects,
              texts: decoded.text,
              ocrStrings: ocrLinesActual,
              includeMasks: true,
            });
            assembledCounts = {
              assembledObjectCount: obs.objects.length,
              assembledRegionCount: obs.regions.length,
              assembledTextCount: obs.text.length,
              assembledRelationCount: obs.relations.length,
            };
          }

          langReports[lang] = {
            step0LogitsAllcloseAt2e4: step0Parity.allcloseAt2e4,
            step0LogitsMaxAbsDiff: Number(step0Parity.maxAbsDiff.toExponential(3)),
            step0LogitsViolationsAt2e4: step0Parity.violationsAt2e4,
            step0LogitsCosineSimilarity: step0Parity.cosineSimilarity,
            fullSequenceTested: runFull,
            tokenIdsExactMatch,
            decodedTextExactMatch,
            generatedTokenCount: decRes.generatedTokenIds.length,
            stepsExecuted: decRes.stepsExecuted,
            decodeMs: Number(decodeMs.toFixed(1)),
            ...assembledCounts,
          };
        }

        caseReports[caseName] = {
          objectCount: decoded.objects.length,
          textCount: decoded.text.length,
          ocrLinesActual,
          ocrLinesExpected: refCase.ocrLines,
          ocrExactMatch:
            ocrLinesActual.length === refCase.ocrLines.length &&
            ocrLinesActual.every((s, idx) => s === refCase.ocrLines[idx]),
          ckAllcloseAt2e4: ckParity.allcloseAt2e4,
          cvAllcloseAt2e4: cvParity.allcloseAt2e4,
          ckMaxAbsDiff: Number(ckParity.maxAbsDiff.toExponential(3)),
          cvMaxAbsDiff: Number(cvParity.maxAbsDiff.toExponential(3)),
          cvViolationsAt2e4: cvParity.violationsAt2e4,
          cvLength: cvParity.length,
          cvViolationRate: Number((cvParity.violationsAt2e4 / cvParity.length).toExponential(3)),
          cvCosineSimilarity: cvParity.cosineSimilarity,
          languages: langReports,
        };
      }

      // End-to-end detectForVideo in 'full' analysisMode with Canvas2D image input
      const img331352 = await loadPngElement(manifest.cases.coco_000000331352.png_file);
      const canvas331352 = document.createElement('canvas');
      canvas331352.width = manifest.cases.coco_000000331352.metadata.W;
      canvas331352.height = manifest.cases.coco_000000331352.metadata.H;
      const ctx331352 = canvas331352.getContext('2d');
      if (!ctx331352) throw new Error('Failed to create 2D context for canvas331352');
      ctx331352.drawImage(img331352, 0, 0, canvas331352.width, canvas331352.height);
      const e2eDetections = await wasmTask.detectForVideo(canvas331352, 0.3, {
        analysisMode: 'full',
        experimentalNmsIou: 0.5,
        mirageLanguage: 'fr',
        maxTokens: 48,
      });
      const e2eDiag = wasmTask.getLastDiagnostics();
      await wasmTask.close();

      // Strict 4-session WebGPU probe (no fallback to WASM)
      const webgpuAdapter = await bridge.probeWebGpuAdapterDiagnostics();
      let strictWebGpuStatus: string;
      let strictWebGpuError: string | null = null;
      let strictWebGpuEncoderCreated = false;
      let strictWebGpuFullAssetsCreated = false;
      let strictWebGpuFullPipelineRunSucceeded = false;
      let strictWebGpuOcrExactMatch = false;
      let strictWebGpuCkAllcloseAt2e4: boolean | null = null;
      let strictWebGpuCkCosineSimilarity: number | null = null;
      let strictWebGpuCvCosineSimilarity: number | null = null;
      let strictWebGpuStep0LogitsCosineSimilarity: number | null = null;
      let strictWebGpuStep0LogitsAllcloseAt2e4: boolean | null = null;
      let strictWebGpuTokenPrefixExactMatch: boolean | null = null;
      let strictWebGpuDecodeMs: number | null = null;
      try {
        const gpuTask = await bridge.createMirageExperimentalOnnxTask({
          variantId: 'mirage-onnx-exp-webgpu',
          allowWasmFallback: false,
          fullPipeline: false,
        });
        strictWebGpuEncoderCreated = true;
        try {
          const gpuFull = await bridge.acquireMirageFullPipelineAssets({
            variantId: 'mirage-onnx-exp-webgpu',
            delegate: 'WebGPU',
            allowWasmFallback: false,
          });
          strictWebGpuFullAssetsCreated = true;
          try {
            const cachedText = cachedInputs.text_regions;
            const cachedCoco = cachedInputs.coco_000000331352;
            if (!cachedText || !cachedCoco) {
              throw new Error('Missing cached inputs for WebGPU 4-session probe');
            }

            // 1. Run CrossKV + 8-layer Decoder (4-token prefix) on WebGPU using cached text_regions features
            const refTextCase = fullRef.cases.text_regions;
            const gpuCrossKv = await bridge.runMirageCrossKv(
              gpuFull.crossKvSession,
              cachedText.memory,
              cachedText.dense_features,
              cachedText.metadata,
              cachedText.decoded.objects,
              cachedText.decoded.text,
              gpuFull.tableReader
            );
            const [refGpuCk, refGpuCv] = await Promise.all([
              fetchFloat32(refTextCase.ckFile),
              fetchFloat32(refTextCase.cvFile),
            ]);
            const gpuCkParity = bridge.compareMirageTensorParity(
              gpuCrossKv.crossK,
              refGpuCk,
              2e-4,
              2e-4,
              1e-4
            );
            strictWebGpuCkAllcloseAt2e4 = gpuCkParity.allcloseAt2e4;
            strictWebGpuCkCosineSimilarity = gpuCkParity.cosineSimilarity;
            strictWebGpuCvCosineSimilarity = bridge.compareMirageTensorParity(
              gpuCrossKv.crossV,
              refGpuCv,
              2e-4,
              2e-4,
              1e-4
            ).cosineSimilarity;

            const gpuDecodeStart = performance.now();
            const gpuDecRes = await bridge.runMirageGreedyDecode(
              gpuFull.decoderStepSession,
              gpuCrossKv,
              gpuFull.tableReader,
              {
                language: 'en',
                maxTokens: 4,
                captureStep0Logits: true,
              }
            );
            strictWebGpuDecodeMs = Number((performance.now() - gpuDecodeStart).toFixed(1));
            if (!gpuDecRes.step0Logits) {
              throw new Error('WebGPU greedy decode omitted step0Logits on text_regions');
            }
            const refGpuStep0 = await fetchFloat32(refTextCase.languages.en.step0LogitsFile);
            const gpuStep0Parity = bridge.compareMirageTensorParity(
              gpuDecRes.step0Logits,
              refGpuStep0,
              2e-4,
              2e-4,
              1e-4
            );
            strictWebGpuStep0LogitsCosineSimilarity = gpuStep0Parity.cosineSimilarity;
            strictWebGpuStep0LogitsAllcloseAt2e4 = gpuStep0Parity.allcloseAt2e4;
            const expectedPrefix = refTextCase.languages.en.generatedTokenIds.slice(0, 4);
            strictWebGpuTokenPrefixExactMatch =
              gpuDecRes.generatedTokenIds.length === 4 &&
              gpuDecRes.generatedTokenIds.every((id, idx) => id === expectedPrefix[idx]);

            // 2. Run Reader OCR on WebGPU for cached coco_000000331352 proposals
            const refCocoCase = fullRef.cases.coco_000000331352;
            const gpuOcrLinesActual = await bridge.readMirageTextLines(
              gpuFull.readerSession,
              cachedCoco.nativeRgba,
              cachedCoco.W,
              cachedCoco.H,
              cachedCoco.decoded.text,
              gpuFull.tableReader.charset
            );
            strictWebGpuOcrExactMatch =
              gpuOcrLinesActual.length === refCocoCase.ocrLines.length &&
              gpuOcrLinesActual.every((s, idx) => s === refCocoCase.ocrLines[idx]);

            strictWebGpuFullPipelineRunSucceeded = true;
            strictWebGpuStatus = webgpuAdapter.isSoftwareAdapter
              ? 'PASSED_SOFTWARE_ADAPTER'
              : 'PASSED_HARDWARE_WEBGPU';
          } finally {
            await gpuFull.readerSession.release?.().catch(() => undefined);
            await gpuFull.crossKvSession.release?.().catch(() => undefined);
            await gpuFull.decoderStepSession.release?.().catch(() => undefined);
          }
        } finally {
          await gpuTask.close();
        }
      } catch (err) {
        strictWebGpuError = err instanceof Error ? err.message : String(err);
        strictWebGpuStatus =
          strictWebGpuEncoderCreated && !strictWebGpuFullAssetsCreated
            ? 'BLOCKED_FULL_PIPELINE_SESSIONS'
            : 'BLOCKED';
      }

      return {
        fullAcquireMs: Number(fullAcquireMs.toFixed(1)),
        vocabSize: fullAssets.tableReader.vocab.size,
        charsetLength: fullAssets.tableReader.charset.length,
        backends: {
          reader: fullAssets.readerBackend,
          crossKv: fullAssets.crossKvBackend,
          decoderStep: fullAssets.decoderStepBackend,
        },
        cases: caseReports,
        detectForVideoE2e: {
          detectionCount: e2eDetections.length,
          firstLabel: e2eDetections[0]?.label ?? '',
          hasNativeAttribution: e2eDetections.some((d) => d.label.includes('[公式WASM]')),
          pipelineMode: e2eDiag?.pipelineMode ?? null,
          fullPipelineAssetsLoaded: e2eDiag?.fullPipelineAssetsLoaded ?? false,
          decoderLanguage: e2eDiag?.decoderLanguage ?? null,
          generatedTokenCount: e2eDiag?.generatedTokenCount ?? 0,
          ocrLineCount: e2eDiag?.ocrLineCount ?? 0,
          hasAssembledObservation: Boolean(e2eDiag?.assembledObservation),
        },
        strictWebGpu4SessionProbe: {
          status: strictWebGpuStatus,
          hardwareWebGpuStatus:
            webgpuAdapter.adapterAvailable && !webgpuAdapter.isSoftwareAdapter
              ? strictWebGpuStatus
              : webgpuAdapter.isSoftwareAdapter
                ? 'BLOCKED_SOFTWARE_ADAPTER_ONLY'
                : 'BLOCKED',
          adapter: webgpuAdapter,
          encoderCreated: strictWebGpuEncoderCreated,
          fullAssetsCreated: strictWebGpuFullAssetsCreated,
          fullPipelineRunSucceeded: strictWebGpuFullPipelineRunSucceeded,
          ocrExactMatch: strictWebGpuOcrExactMatch,
          ckAllcloseAt2e4: strictWebGpuCkAllcloseAt2e4,
          ckCosineSimilarity: strictWebGpuCkCosineSimilarity,
          cvCosineSimilarity: strictWebGpuCvCosineSimilarity,
          step0LogitsCosineSimilarity: strictWebGpuStep0LogitsCosineSimilarity,
          step0LogitsAllcloseAt2e4: strictWebGpuStep0LogitsAllcloseAt2e4,
          tokenPrefixExactMatch: strictWebGpuTokenPrefixExactMatch,
          decodeMs: strictWebGpuDecodeMs,
          error: strictWebGpuError,
        },
      };
    });

    writeFileSync(
      testInfo.outputPath('mirage-full-pipeline-browser.json'),
      JSON.stringify(report, null, 2) + '\n',
      'utf8'
    );
    await testInfo.attach('mirage-full-pipeline-browser.json', {
      path: testInfo.outputPath('mirage-full-pipeline-browser.json'),
      contentType: 'application/json',
    });

    for (const [caseName, cReport] of Object.entries(report.cases)) {
      expect(cReport.ckAllcloseAt2e4, `${caseName} ck parity`).toBe(true);
      // cross_attn/v_proj is unnormalized (values up to ~280); WASM vs CPU matmul and encoder deltas leave <0.1% elements above 2e-4
      expect(cReport.cvCosineSimilarity, `${caseName} cv cosine similarity`).toBeGreaterThanOrEqual(0.999999);
      expect(cReport.cvViolationRate, `${caseName} cv 2e-4 violation rate`).toBeLessThanOrEqual(0.001);
      expect(cReport.cvMaxAbsDiff, `${caseName} cv maxAbsDiff`).toBeLessThanOrEqual(0.1);
      expect(cReport.ocrExactMatch, `${caseName} Reader OCR match`).toBe(true);
      for (const [lang, lReport] of Object.entries(cReport.languages)) {
        expect(
          lReport.step0LogitsAllcloseAt2e4,
          `${caseName}/${lang} step0 logits parity (maxAbsDiff=${lReport.step0LogitsMaxAbsDiff}, viol=${lReport.step0LogitsViolationsAt2e4})`
        ).toBe(true);
        if (lReport.fullSequenceTested) {
          expect(lReport.tokenIdsExactMatch, `${caseName}/${lang} token IDs`).toBe(true);
          expect(lReport.decodedTextExactMatch, `${caseName}/${lang} decodedText`).toBe(true);
        }
      }
    }

    expect(report.detectForVideoE2e.pipelineMode).toBe('full_pipeline');
    expect(report.detectForVideoE2e.fullPipelineAssetsLoaded).toBe(true);
    expect(report.detectForVideoE2e.decoderLanguage).toBe('fr');
    expect(report.detectForVideoE2e.generatedTokenCount).toBe(48);
    expect(report.detectForVideoE2e.ocrLineCount).toBeGreaterThanOrEqual(1);
    expect(report.detectForVideoE2e.hasNativeAttribution).toBe(false);
    expect(report.detectForVideoE2e.detectionCount).toBeGreaterThan(0);
    expect(report.detectForVideoE2e.hasAssembledObservation).toBe(true);

    expect(['PASSED_SOFTWARE_ADAPTER', 'PASSED_HARDWARE_WEBGPU']).toContain(
      report.strictWebGpu4SessionProbe.status
    );
    expect(report.strictWebGpu4SessionProbe.hardwareWebGpuStatus).toBe(
      report.strictWebGpu4SessionProbe.adapter.isSoftwareAdapter
        ? 'BLOCKED_SOFTWARE_ADAPTER_ONLY'
        : 'PASSED_HARDWARE_WEBGPU'
    );
    expect(report.strictWebGpu4SessionProbe.encoderCreated).toBe(true);
    expect(report.strictWebGpu4SessionProbe.fullAssetsCreated).toBe(true);
    expect(report.strictWebGpu4SessionProbe.fullPipelineRunSucceeded).toBe(true);
    expect(report.strictWebGpu4SessionProbe.ocrExactMatch).toBe(true);
    expect(report.strictWebGpu4SessionProbe.ckAllcloseAt2e4).toBe(true);
    expect(
      report.strictWebGpu4SessionProbe.cvCosineSimilarity
    ).toBeGreaterThanOrEqual(0.999999);
    expect(report.strictWebGpu4SessionProbe.step0LogitsAllcloseAt2e4).toBe(true);
    expect(
      report.strictWebGpu4SessionProbe.step0LogitsCosineSimilarity
    ).toBeGreaterThanOrEqual(0.999999);
    expect(report.strictWebGpu4SessionProbe.tokenPrefixExactMatch).toBe(true);
  });

  test('switches experimental Mirage UI from fast proposal mode to full 4-graph inference and changes decoder language', async ({
    context,
    page,
  }) => {
    test.setTimeout(300_000);
    const fullRefPath = path.resolve(
      WORKSPACE_ROOT,
      '.cache/mirage-browser-fixtures/full_pipeline_reference.json'
    );
    const cactPath = path.resolve(
      WORKSPACE_ROOT,
      '.cache/container-models/mirage.cact'
    );
    const hasAllArtifacts =
      existsSync(ONNX_ARTIFACT_PATH) &&
      existsSync(FIXTURE_MANIFEST_PATH) &&
      existsSync(fullRefPath) &&
      existsSync(cactPath);

    if (process.env.REQUIRE_MIRAGE_REAL_MODEL === '1') {
      expect(hasAllArtifacts).toBe(true);
    } else if (!hasAllArtifacts) {
      test.skip(true, 'Real Mirage full-pipeline artifacts not present locally');
    }

    await routeLocalModelCache(context);
    await page.goto('/');

    const header = page.getByRole('banner');
    await header.getByRole('combobox', { name: 'モデル', exact: true }).selectOption('mirage');
    const variantSelect = header.getByRole('combobox', { name: 'バリアント', exact: true });
    await variantSelect.selectOption('mirage-onnx-exp-wasm');

    await expect(page.getByRole('status')).toContainText('モデル準備完了', {
      timeout: 60_000,
    });

    await page.getByRole('button', { name: '検証: テキスト領域' }).click();
    await expect(
      page.getByText('[ONNX候補] 物体領域候補 #1').first()
    ).toBeVisible({ timeout: 30_000 });

    await page.getByRole('button', { name: 'モデル調整' }).click();
    const pipelineGroup = page.getByRole('group', { name: 'ONNX パイプライン構成' });
    await pipelineGroup.getByRole('button', { name: /フル推論（OCR・命名）/ }).click();

    const diagBox = page.getByLabel('実験用部分ONNX実行診断');
    await expect(diagBox).toContainText('実験用 Mirage ONNX フル推論 診断', {
      timeout: 180_000,
    });
    await expect(diagBox).toContainText('full_pipeline (Encoder+Reader+CrossKV+Decoder)');
    await expect(diagBox).toContainText(/デコーダ: 言語 en · [1-9]\d*トークン/, {
      timeout: 60_000,
    });

    const langSelect = page.getByLabel('Mirage 出力言語');
    await langSelect.selectOption('fr');
    await expect(diagBox).toContainText(/デコーダ: 言語 fr · [1-9]\d*トークン/, {
      timeout: 180_000,
    });

    await page.getByRole('button', { name: /^認識物体/ }).click();
    await expect(page.getByText(/認識結果 \(\d+件\)/)).toBeVisible();
    await expect(page.getByText('[公式WASM]')).toHaveCount(0);
  });
});

