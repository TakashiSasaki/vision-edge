import { expect, test, type BrowserContext } from '@playwright/test';
import { existsSync, writeFileSync } from 'node:fs';
import { IncomingMessage, ServerResponse } from 'node:http';
import path from 'node:path';
import { ContainerModelCacheManager } from '../src/services/containerModelCacheServer';
import {
  MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256,
  MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE,
} from '../src/services/mirageExperimentalOnnxService';

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
  method: string
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
      headers: { host: '127.0.0.1:4173' },
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
        route.request().method()
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
  test.setTimeout(240_000);

  test('verifies SHA-256, WASM vs WebGPU execution providers, identical-input tensor parity (atol=rtol=2e-4), and Canvas vs Pillow preprocessing across 8 cases', async ({
    context,
    page,
  }, testInfo) => {
    test.skip(
      !existsSync(ONNX_ARTIFACT_PATH) || !existsSync(FIXTURE_MANIFEST_PATH),
      'Requires local verified cact-dequantized-reference-order.onnx and .cache/mirage-browser-fixtures/manifest.json'
    );

    await routeLocalModelCache(context);
    await page.goto('/');
    await expect(page.getByRole('banner').getByText('VisionStudio AI')).toBeVisible();

    const browserReport = await page.evaluate(
      async ({ expectedSize, expectedSha256 }) => {
        const bridge = window.__mirageExperimentalBridge;
        if (!bridge) {
          throw new Error('window.__mirageExperimentalBridge is not registered');
        }

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
        const navGpu = (navigator as unknown as { gpu?: { requestAdapter?: () => Promise<unknown> } }).gpu;
        let webgpuAdapterAvailable = false;
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

        if (navGpu && typeof navGpu.requestAdapter === 'function') {
          try {
            const adapter = await navGpu.requestAdapter();
            webgpuAdapterAvailable = !!adapter;
          } catch {
            webgpuAdapterAvailable = false;
          }
        }

        // Ensure ort.env.wasm.wasmPaths points to the local container cache BEFORE any session creation
        bridge.ort.env.logLevel = 'error';
        bridge.ort.env.wasm.numThreads = 1;
        bridge.ort.env.wasm.wasmPaths = `${window.location.origin}/api/model-cache/wasm/ort/`;

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
          try {
            const runStart = performance.now();
            const dummyImage = new bridge.ort.Tensor(
              'float32',
              new Float32Array(1 * 3 * 512 * 512),
              [1, 3, 512, 512]
            );
            const dummyHw = new bridge.ort.Tensor(
              'int64',
              new BigInt64Array([BigInt(512), BigInt(512)]),
              [2]
            );
            const out = await directSession.run(
              { image: dummyImage, valid_hw: dummyHw },
              ['detections', 'prototypes', 'coefficients']
            );
            webgpuDirectSessionOutcome.runSucceeded =
              !!out.detections && !!out.prototypes && !!out.coefficients;
            webgpuDirectSessionOutcome.runMs = performance.now() - runStart;
            dummyImage.dispose?.();
            dummyHw.dispose?.();
            for (const v of Object.values(out)) {
              (v as { dispose?: () => void })?.dispose?.();
            }
          } catch (runErr) {
            webgpuDirectSessionOutcome.error = `WebGPU run failed: ${
              runErr instanceof Error ? runErr.message : String(runErr)
            }`;
          } finally {
            await directSession.release();
          }
        } catch (createErr) {
          webgpuDirectSessionOutcome.createMs = performance.now() - directStart;
          webgpuDirectSessionOutcome.error = `WebGPU session create failed: ${
            createErr instanceof Error ? createErr.message : String(createErr)
          }`;
        }

        // 4. Test mirage-onnx-exp-webgpu via application task factory (with configured fallback observation)
        const webgpuTaskStart = performance.now();
        const webgpuTask = await bridge.createMirageExperimentalOnnxTask({
          variantId: 'mirage-onnx-exp-webgpu',
        });
        const webgpuTaskInitMs = performance.now() - webgpuTaskStart;
        const webgpuBackendInfo = webgpuTask.backend;

        // Run one inference on square_blank using webgpuTask to confirm end-to-end execution
        const webgpuWarmupStart = performance.now();
        const webgpuBlankHeads = await webgpuTask.runRawHeads({
          image: new Float32Array(1 * 3 * 512 * 512),
          valid_hw: [512, 512],
          metadata: bridge.computeMirageCeil32Metadata(512, 512),
        });
        const webgpuWarmupMs = performance.now() - webgpuWarmupStart;
        await webgpuTask.close();

        // 5. Initialize mirage-onnx-exp-wasm and run all 8 benchmark cases
        const wasmTaskStart = performance.now();
        const wasmTask = await bridge.createMirageExperimentalOnnxTask({
          variantId: 'mirage-onnx-exp-wasm',
        });
        const wasmTaskInitMs = performance.now() - wasmTaskStart;
        const wasmBackendInfo = wasmTask.backend;

        function compareFloat32Tensors(
          actual: Float32Array,
          expected: Float32Array,
          atol = 2e-4,
          rtol = 2e-4
        ) {
          if (actual.length !== expected.length) {
            throw new Error(
              `Tensor length mismatch: ${actual.length} vs ${expected.length}`
            );
          }
          let maxAbsDiff = 0;
          let sumAbsDiff = 0;
          let maxRelDiff = 0;
          let violationsAt2e4 = 0;
          for (let i = 0; i < actual.length; i++) {
            const a = actual[i];
            const b = expected[i];
            const diff = Math.abs(a - b);
            sumAbsDiff += diff;
            if (diff > maxAbsDiff) maxAbsDiff = diff;
            const denom = Math.max(Math.abs(b), 1e-12);
            const rel = diff / denom;
            if (rel > maxRelDiff) maxRelDiff = rel;
            if (diff > atol + rtol * Math.abs(b)) {
              violationsAt2e4++;
            }
          }
          return {
            length: actual.length,
            maxAbsDiff,
            meanAbsDiff: sumAbsDiff / actual.length,
            maxRelDiff,
            violationsAt2e4,
            allcloseAt2e4: violationsAt2e4 === 0,
          };
        }

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

        const caseReports: Record<string, unknown> = {};
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
            { confidenceThreshold: 0.3, nmsIou: null, includeMasks: true }
          );
          const decodedNms050 = bridge.decodeMirageExperimentalOutputs(
            browserHeads,
            metadata,
            { confidenceThreshold: 0.3, nmsIou: 0.5, includeMasks: false }
          );

          const exactProposalParityNone =
            decodedNone.objectCount === entry.decoded_default.object_count &&
            decodedNone.textCount === entry.decoded_default.text_count &&
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
                  entry.decoded_default.objects[idx].mask_active_pixels
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
            decodedNms050.objectCount === entry.decoded_nms_0p50.object_count &&
            decodedNms050.textCount === entry.decoded_nms_0p50.text_count &&
            decodedNms050.objects.every(
              (o, idx) =>
                o.cell[0] === entry.decoded_nms_0p50.objects[idx].cell[0] &&
                o.cell[1] === entry.decoded_nms_0p50.objects[idx].cell[1] &&
                o.bbox.every(
                  (v, bi) => v === entry.decoded_nms_0p50.objects[idx].bbox[bi]
                )
            );

          // Part B: Browser Canvas 2D Bilinear Preprocessing from PNG vs Python Pillow Bilinear
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
            { confidenceThreshold: 0.3, nmsIou: null, includeMasks: false }
          );
          const canvasDecodedNms050 = bridge.decodeMirageExperimentalOutputs(
            canvasHeads,
            metadata,
            { confidenceThreshold: 0.3, nmsIou: 0.5, includeMasks: false }
          );

          // Compare Canvas-preprocessed proposals vs Pillow-preprocessed proposals
          let matchedObjectsIoU085 = 0;
          for (const refObj of decodedNone.objects) {
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
              objectCountNoNms: decodedNone.objectCount,
              textCountNoNms: decodedNone.textCount,
              objectCountNms050: decodedNms050.objectCount,
              textCountNms050: decodedNms050.textCount,
            },
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
              canvasObjectCountNoNms: canvasDecodedNone.objectCount,
              canvasTextCountNoNms: canvasDecodedNone.textCount,
              canvasObjectCountNms050: canvasDecodedNms050.objectCount,
              canvasTextCountNms050: canvasDecodedNms050.textCount,
              matchedPillowObjectsAtIoU085: matchedObjectsIoU085,
              totalPillowObjectsNoNms: decodedNone.objectCount,
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
              navigatorGpuPresent: !!navGpu,
              webgpuAdapterAvailable,
              directWebGpuSession: webgpuDirectSessionOutcome,
              applicationTaskInitMs: Number(webgpuTaskInitMs.toFixed(2)),
              applicationWarmupInferenceMs: Number(webgpuWarmupMs.toFixed(2)),
              applicationWarmupOutputLengths: [
                webgpuBlankHeads.detections.length,
                webgpuBlankHeads.prototypes.length,
                webgpuBlankHeads.coefficients.length,
              ],
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

    // Save structured JSON report to test output directory
    const outputReportPath = testInfo.outputPath(
      'mirage-browser-onnx-results.json'
    );
    writeFileSync(
      outputReportPath,
      JSON.stringify(browserReport, null, 2) + '\n',
      'utf8'
    );
    if (process.env.WRITE_MIRAGE_BROWSER_REPORT === '1') {
      const docsReportPath = path.resolve(
        WORKSPACE_ROOT,
        'docs/verification/mirage-browser-onnx-results.json'
      );
      writeFileSync(
        docsReportPath,
        JSON.stringify(browserReport, null, 2) + '\n',
        'utf8'
      );
    }

    // Assertions on artifact verification
    expect(browserReport.artifact.matchesExpectedSize).toBe(true);
    expect(browserReport.artifact.matchesExpectedSha256).toBe(true);
    expect(browserReport.artifact.size).toBe(34_281_503);
    expect(browserReport.artifact.sha256).toBe(
      '52d52d2918160404cd806702a270736819712025cf3c6aa8e907a06bc3cbdd07'
    );

    // Assertions on WASM and WebGPU EP reporting
    expect(browserReport.executionProviders.wasm.backend.requestedProvider).toBe(
      'wasm'
    );
    expect(
      browserReport.executionProviders.wasm.backend.configuredProviders
    ).toEqual(['wasm']);
    expect(
      browserReport.executionProviders.webgpu.backend.requestedProvider
    ).toBe('webgpu');
    expect(
      browserReport.executionProviders.webgpu.applicationWarmupOutputLengths
    ).toEqual([64 * 64 * 10, 32 * 128 * 128, 32 * 64 * 64]);

    // Assertions on all 8 benchmark cases in real browser WASM
    const cases = browserReport.cases as Record<
      string,
      {
        identicalInputParity: {
          allThreeHeadsAllcloseAt2e4: boolean;
          exactProposalParityNone: boolean;
          exactProposalParityNms050: boolean;
        };
        canvasVsPillowPreprocess: {
          resizeOccurred: boolean;
          validRegionMaxAbsDiff: number;
          paddingMaxAbsValue: number;
        };
      }
    >;
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
      expect(
        report.canvasVsPillowPreprocess.paddingMaxAbsValue,
        `${caseName}: zero padding outside valid_hw`
      ).toBe(0);
    }

    // For 512x512 native inputs (no resize), Canvas 2D and Pillow RGB pixels must be identical
    expect(cases.overlap_shapes.canvasVsPillowPreprocess.resizeOccurred).toBe(
      false
    );
    expect(
      cases.overlap_shapes.canvasVsPillowPreprocess.validRegionMaxAbsDiff
    ).toBeLessThanOrEqual(1e-6);
  });

  test('renders experimental partial Mirage ONNX proposals, masks, NMS toggle, and optional native WASM overlay in the VisionStudio AI UI', async ({
    context,
    page,
  }, testInfo) => {
    test.skip(
      !existsSync(ONNX_ARTIFACT_PATH) || !existsSync(FIXTURE_MANIFEST_PATH),
      'Requires local verified cact-dequantized-reference-order.onnx and .cache/mirage-browser-fixtures/manifest.json'
    );

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
    await expect(page.getByText('認識結果 (6件)')).toBeVisible();

    // Switch to モデル調整 tab to inspect live diagnostics and test NMS policy buttons
    await page.getByRole('button', { name: 'モデル調整' }).click();
    const diagBox = page.getByLabel('実験用部分ONNX実行診断');
    await expect(diagBox).toContainText('要求EP: wasm ／ 設定EP: wasm');
    await expect(diagBox).toContainText('抽出候補: 物体 6件 ／ テキスト 0件');

    // Toggle NMS to Greedy NMS (IoU 0.5) -> 6 overlapping candidates reduce to 2
    await page.getByRole('button', { name: /Greedy NMS \(IoU 0\.5\)/ }).click();
    await expect(diagBox).toContainText('抽出候補: 物体 2件 ／ テキスト 0件', {
      timeout: 30_000,
    });
    await page.getByRole('button', { name: /NMSなし（既定）/ }).click();
    await expect(diagBox).toContainText('抽出候補: 物体 6件 ／ テキスト 0件', {
      timeout: 30_000,
    });

    // 2. Run preset: text_regions.png
    await page.getByRole('button', { name: '検証: テキスト領域' }).click();
    await expect(diagBox).toContainText('抽出候補: 物体 3件 ／ テキスト 1件', {
      timeout: 30_000,
    });
    await page.getByRole('button', { name: /認識物体/ }).click();
    await expect(
      page.getByText('[ONNX候補] テキスト領域候補 #1').first()
    ).toBeVisible({ timeout: 30_000 });

    // 3. Run preset: coco_000000458054.png
    await page.getByRole('button', { name: '検証: COCO 458054' }).click();
    await page.getByRole('button', { name: 'モデル調整' }).click();
    await expect(diagBox).toContainText(
      '入力 valid_hw: [384, 512] (元画像 640×427 · ceil32)',
      { timeout: 30_000 }
    );
    await expect(diagBox).toContainText('抽出候補: 物体 20件 ／ テキスト 15件');

    await page.screenshot({
      path: testInfo.outputPath('mirage-experimental-onnx-ui-coco.png'),
      fullPage: true,
    });

    // 4. Switch to experimental WebGPU variant and verify initialization + inference
    await variantSelect.selectOption('mirage-onnx-exp-webgpu');
    await expect(page.getByRole('status')).toContainText('モデル準備完了', {
      timeout: 60_000,
    });
    await expect(diagBox).toContainText('要求EP: webgpu', {
      timeout: 30_000,
    });
  });
});
