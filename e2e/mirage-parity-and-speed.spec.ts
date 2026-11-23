import { expect, test, type BrowserContext } from '@playwright/test';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { IncomingMessage, ServerResponse } from 'node:http';
import path from 'node:path';
import { ContainerModelCacheManager } from '../src/services/containerModelCacheServer';
import type { MirageDecodedPerceptionResult } from '../src/services/mirageExperimentalOnnxService';

const WORKSPACE_ROOT = process.cwd();
const CACT_PATH = path.resolve(WORKSPACE_ROOT, '.cache/container-models/mirage.cact');
const NATIVE_GEOMETRY_PATH = path.resolve(
  WORKSPACE_ROOT,
  'docs/verification/mirage-refinement-native-geometry.json'
);
const NATIVE_CORPUS_PATH = path.resolve(
  WORKSPACE_ROOT,
  'docs/verification/mirage-native-corpus-results.json'
);
const OUTPUT_DOCS_JSON_PATH = path.resolve(
  WORKSPACE_ROOT,
  'docs/mirage-onnx-parity-and-speed.json'
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

test.describe('Mirage ONNX (WASM & WebGPU) vs Official Cactus Parity & Speed Benchmark', () => {
  test.setTimeout(360_000);

  test('verifies output parity and measures processing speed across mirage-onnx-exp-wasm, mirage-onnx-exp-webgpu, and official Cactus runtime', async ({
    context,
    page,
  }, testInfo) => {
    if (process.env.REQUIRE_MIRAGE_REAL_MODEL === '1') {
      expect(existsSync(CACT_PATH), `Expected mirage.cact at ${CACT_PATH}`).toBe(true);
    } else {
      test.skip(!existsSync(CACT_PATH), 'Requires local .cache/container-models/mirage.cact');
    }

    const nativeGeometryRaw = JSON.parse(readFileSync(NATIVE_GEOMETRY_PATH, 'utf8'));
    const nativeCorpusRaw = JSON.parse(readFileSync(NATIVE_CORPUS_PATH, 'utf8'));

    await routeLocalModelCache(context);
    await page.goto('/');
    await expect(page.getByRole('banner').getByText('VisionStudio AI')).toBeVisible();

    const benchmarkReport = await page.evaluate(
      async ({ nativeGeometry, nativeCorpus }) => {
        const bridge = window.__mirageExperimentalBridge;
        if (!bridge) {
          throw new Error('window.__mirageExperimentalBridge is not registered');
        }
        bridge.cancelInFlightMirageWasmInit?.();
        bridge.ort.env.logLevel = 'error';
        bridge.ort.env.wasm.numThreads = 1;
        bridge.ort.env.wasm.wasmPaths = `${window.location.origin}/api/model-cache/wasm/ort/`;

        const round = (val: number, digits = 3) =>
          Number.isFinite(val) ? Number(val.toFixed(digits)) : val;

        const summarizeTimes = (samples: number[]) => {
          const sorted = [...samples].sort((a, b) => a - b);
          const mean = samples.reduce((s, v) => s + v, 0) / Math.max(1, samples.length);
          const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
          const min = sorted[0] ?? 0;
          const max = sorted[sorted.length - 1] ?? 0;
          const variance =
            samples.reduce((s, v) => s + (v - mean) ** 2, 0) /
            Math.max(1, samples.length);
          return {
            runs: samples.length,
            samplesMs: samples.map((v) => round(v, 2)),
            meanMs: round(mean, 2),
            medianMs: round(median, 2),
            minMs: round(min, 2),
            maxMs: round(max, 2),
            stdDevMs: round(Math.sqrt(variance), 2),
          };
        };

        const boxIoU = (
          a: [number, number, number, number],
          b: [number, number, number, number]
        ): number => {
          const ix1 = Math.max(a[0], b[0]);
          const iy1 = Math.max(a[1], b[1]);
          const ix2 = Math.min(a[2], b[2]);
          const iy2 = Math.min(a[3], b[3]);
          const iw = Math.max(0, ix2 - ix1);
          const ih = Math.max(0, iy2 - iy1);
          const inter = iw * ih;
          if (inter <= 0) return 0;
          const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
          const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
          const union = areaA + areaB - inter;
          return union > 0 ? inter / union : 0;
        };

        const evaluateBoxMatching = (
          candidateBoxes: [number, number, number, number][],
          referenceBoxes: [number, number, number, number][],
          iouThreshold = 0.5
        ) => {
          const pairs: { cIdx: number; rIdx: number; iou: number }[] = [];
          for (let cIdx = 0; cIdx < candidateBoxes.length; cIdx++) {
            for (let rIdx = 0; rIdx < referenceBoxes.length; rIdx++) {
              const iou = boxIoU(candidateBoxes[cIdx], referenceBoxes[rIdx]);
              if (iou >= iouThreshold) {
                pairs.push({ cIdx, rIdx, iou });
              }
            }
          }
          pairs.sort((a, b) => b.iou - a.iou);
          const usedC = new Set<number>();
          const usedR = new Set<number>();
          const matched: {
            candidateIndex: number;
            referenceIndex: number;
            iou: number;
            candidateBox: [number, number, number, number];
            referenceBox: [number, number, number, number];
          }[] = [];
          for (const p of pairs) {
            if (!usedC.has(p.cIdx) && !usedR.has(p.rIdx)) {
              usedC.add(p.cIdx);
              usedR.add(p.rIdx);
              matched.push({
                candidateIndex: p.cIdx,
                referenceIndex: p.rIdx,
                iou: round(p.iou, 4),
                candidateBox: candidateBoxes[p.cIdx],
                referenceBox: referenceBoxes[p.rIdx],
              });
            }
          }
          const tp = matched.length;
          const fp = candidateBoxes.length - tp;
          const fn = referenceBoxes.length - tp;
          const precision = candidateBoxes.length > 0 ? tp / candidateBoxes.length : 1;
          const recall = referenceBoxes.length > 0 ? tp / referenceBoxes.length : 1;
          const f1 =
            precision + recall > 0
              ? (2 * precision * recall) / (precision + recall)
              : candidateBoxes.length === 0 && referenceBoxes.length === 0
                ? 1
                : 0;
          const meanMatchedIoU =
            tp > 0
              ? matched.reduce((acc, m) => acc + m.iou, 0) / tp
              : candidateBoxes.length === 0 && referenceBoxes.length === 0
                ? 1
                : 0;
          return {
            candidateCount: candidateBoxes.length,
            referenceCount: referenceBoxes.length,
            matchedCount: tp,
            falsePositiveCount: fp,
            falseNegativeCount: fn,
            precision: round(precision, 4),
            recall: round(recall, 4),
            geometricF1: round(f1, 4),
            meanMatchedIoU: round(meanMatchedIoU, 4),
            matchedPairs: matched,
          };
        };

        const compareProposals = (
          actual: MirageDecodedPerceptionResult,
          expected: MirageDecodedPerceptionResult
        ) => {
          const objectCountMatch =
            actual.objects.length === expected.objects.length;
          const textCountMatch = actual.text.length === expected.text.length;
          let maxScoreDiff = 0;
          let maxBoxCoordDiff = 0;
          let cellsExactMatch = objectCountMatch && textCountMatch;
          let masksExactMatch = objectCountMatch;

          if (objectCountMatch) {
            for (let i = 0; i < actual.objects.length; i++) {
              const a = actual.objects[i];
              const b = expected.objects[i];
              if (a.cell[0] !== b.cell[0] || a.cell[1] !== b.cell[1]) {
                cellsExactMatch = false;
              }
              maxScoreDiff = Math.max(maxScoreDiff, Math.abs(a.score - b.score));
              for (let k = 0; k < 4; k++) {
                maxBoxCoordDiff = Math.max(
                  maxBoxCoordDiff,
                  Math.abs(a.box[k] - b.box[k])
                );
              }
              if (a.maskActivePixels !== b.maskActivePixels) {
                masksExactMatch = false;
              }
            }
          }
          if (textCountMatch) {
            for (let i = 0; i < actual.text.length; i++) {
              const a = actual.text[i];
              const b = expected.text[i];
              if (a.cell[0] !== b.cell[0] || a.cell[1] !== b.cell[1]) {
                cellsExactMatch = false;
              }
              maxScoreDiff = Math.max(maxScoreDiff, Math.abs(a.score - b.score));
              for (let k = 0; k < 4; k++) {
                maxBoxCoordDiff = Math.max(
                  maxBoxCoordDiff,
                  Math.abs(a.box[k] - b.box[k])
                );
              }
            }
          }

          const exactMatch =
            objectCountMatch &&
            textCountMatch &&
            cellsExactMatch &&
            masksExactMatch &&
            maxScoreDiff < 1e-4 &&
            maxBoxCoordDiff <= 1;

          return {
            exactMatch,
            objectCountActual: actual.objects.length,
            objectCountExpected: expected.objects.length,
            textCountActual: actual.text.length,
            textCountExpected: expected.text.length,
            cellsExactMatch,
            masksExactMatch,
            maxScoreDiff: round(maxScoreDiff, 6),
            maxBoxCoordDiff: round(maxBoxCoordDiff, 4),
          };
        };

        const loadPresetImage = async (name: string) => {
          const url = `/api/model-cache/mirage/fixtures/${name}.png`;
          const img = new Image();
          img.crossOrigin = 'anonymous';
          await new Promise<void>((resolve, reject) => {
            img.onload = () => resolve();
            img.onerror = () => reject(new Error(`Failed to load ${url}`));
            img.src = url;
          });
          const width = img.naturalWidth;
          const height = img.naturalHeight;
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          if (!ctx) throw new Error('2D context unavailable');
          ctx.drawImage(img, 0, 0);
          const imageData = ctx.getImageData(0, 0, width, height);
          const rgba = new Uint8Array(
            imageData.data.buffer,
            imageData.data.byteOffset,
            imageData.data.byteLength
          );

          const metadata = bridge.computeMirageCeil32Metadata(width, height);
          const resizeCanvas = document.createElement('canvas');
          resizeCanvas.width = metadata.w;
          resizeCanvas.height = metadata.h;
          const resizeCtx = resizeCanvas.getContext('2d');
          if (!resizeCtx) throw new Error('Resize 2D context unavailable');
          resizeCtx.drawImage(canvas, 0, 0, metadata.w, metadata.h);
          const resizedPixels = resizeCtx.getImageData(
            0,
            0,
            metadata.w,
            metadata.h
          ).data;
          const prep = bridge.packResizedPixelsToMirageTensor(
            resizedPixels,
            4,
            metadata
          );

          return {
            name,
            width,
            height,
            canvas,
            rgba,
            metadata,
            prep,
          };
        };

        // 1. Probe WebGPU Adapter Diagnostics
        const webgpuAdapter = await bridge.probeWebGpuAdapterDiagnostics();

        // 2. Load 3 Preset Images
        const presets = await Promise.all([
          loadPresetImage('overlap_shapes'),
          loadPresetImage('text_regions'),
          loadPresetImage('coco_000000458054'),
        ]);

        // 3. Initialize all 3 runtimes and measure session creation latency
        const wasmInitStart = performance.now();
        const wasmTask = await bridge.createMirageExperimentalOnnxTask({
          variantId: 'mirage-onnx-exp-wasm',
          fullPipeline: false,
        });
        const wasmEncoderInitMs = round(performance.now() - wasmInitStart, 2);

        const wasmFullInitStart = performance.now();
        const wasmFullAssets = await bridge.acquireMirageFullPipelineAssets({
          variantId: 'mirage-onnx-exp-wasm',
          delegate: 'WASM',
          allowWasmFallback: false,
        });
        const wasmFullPipelineInitMs = round(
          performance.now() - wasmFullInitStart,
          2
        );

        const webgpuInitStart = performance.now();
        const webgpuTask = await bridge.createMirageExperimentalOnnxTask({
          variantId: 'mirage-onnx-exp-webgpu',
          allowWasmFallback: false,
          fullPipeline: false,
        });
        const webgpuEncoderInitMs = round(
          performance.now() - webgpuInitStart,
          2
        );

        const webgpuFullInitStart = performance.now();
        const webgpuFullAssets = await bridge.acquireMirageFullPipelineAssets({
          variantId: 'mirage-onnx-exp-webgpu',
          delegate: 'WebGPU',
          allowWasmFallback: false,
        });
        const webgpuFullPipelineInitMs = round(
          performance.now() - webgpuFullInitStart,
          2
        );

        const cactusInitStart = performance.now();
        const cactusTask = await bridge.createMirageWasmTask({
          variantId: 'mirage-51m',
        });
        const cactusWasmInitMs = round(performance.now() - cactusInitStart, 2);

        try {
          // 4. FAST MODE BENCHMARK & PARITY (Encoder 3-head + Proposal Decoder vs Official Cactus)
          const fastModeCases: Record<string, unknown> = {};
          const BENCHMARK_RUNS = 5;

          for (const preset of presets) {
            // Warmup runs
            await wasmTask.runRawHeads(preset.prep);
            await webgpuTask.runRawHeads(preset.prep);
            await cactusTask.detectForVideo(preset.canvas, 0.3, {
              analysisMode: 'fast',
            });

            // Timed runs: ONNX WASM
            const wasmEncoderTimes: number[] = [];
            const wasmTotalTimes: number[] = [];
            let lastWasmHeads = await wasmTask.runRawHeads(preset.prep);
            let lastWasmDecodedNone = bridge.decodeMirageExperimentalOutputs(
              lastWasmHeads,
              preset.metadata,
              { confidence: 0.3, nmsIou: null, includeMasks: true }
            );
            let lastWasmDecodedNms050 = bridge.decodeMirageExperimentalOutputs(
              lastWasmHeads,
              preset.metadata,
              { confidence: 0.3, nmsIou: 0.5, includeMasks: true }
            );

            for (let i = 0; i < BENCHMARK_RUNS; i++) {
              const t0 = performance.now();
              const heads = await wasmTask.runRawHeads(preset.prep);
              const t1 = performance.now();
              lastWasmDecodedNone = bridge.decodeMirageExperimentalOutputs(
                heads,
                preset.metadata,
                { confidence: 0.3, nmsIou: null, includeMasks: true }
              );
              lastWasmDecodedNms050 = bridge.decodeMirageExperimentalOutputs(
                heads,
                preset.metadata,
                { confidence: 0.3, nmsIou: 0.5, includeMasks: true }
              );
              const t2 = performance.now();
              wasmEncoderTimes.push(t1 - t0);
              wasmTotalTimes.push(t2 - t0);
              lastWasmHeads = heads;
            }

            // Timed runs: ONNX WebGPU
            const webgpuEncoderTimes: number[] = [];
            const webgpuTotalTimes: number[] = [];
            let lastWebgpuHeads = await webgpuTask.runRawHeads(preset.prep);
            let lastWebgpuDecodedNone = bridge.decodeMirageExperimentalOutputs(
              lastWebgpuHeads,
              preset.metadata,
              { confidence: 0.3, nmsIou: null, includeMasks: true }
            );
            let lastWebgpuDecodedNms050 = bridge.decodeMirageExperimentalOutputs(
              lastWebgpuHeads,
              preset.metadata,
              { confidence: 0.3, nmsIou: 0.5, includeMasks: true }
            );

            for (let i = 0; i < BENCHMARK_RUNS; i++) {
              const t0 = performance.now();
              const heads = await webgpuTask.runRawHeads(preset.prep);
              const t1 = performance.now();
              lastWebgpuDecodedNone = bridge.decodeMirageExperimentalOutputs(
                heads,
                preset.metadata,
                { confidence: 0.3, nmsIou: null, includeMasks: true }
              );
              lastWebgpuDecodedNms050 = bridge.decodeMirageExperimentalOutputs(
                heads,
                preset.metadata,
                { confidence: 0.3, nmsIou: 0.5, includeMasks: true }
              );
              const t2 = performance.now();
              webgpuEncoderTimes.push(t1 - t0);
              webgpuTotalTimes.push(t2 - t0);
              lastWebgpuHeads = heads;
            }

            // Timed runs: Official Cactus WASM (fast mode)
            const cactusFastTimes: number[] = [];
            for (let i = 0; i < BENCHMARK_RUNS; i++) {
              const t0 = performance.now();
              await cactusTask.detectForVideo(preset.canvas, 0.3, {
                analysisMode: 'fast',
              });
              const t1 = performance.now();
              cactusFastTimes.push(t1 - t0);
            }
            const cactusFastRawObs = cactusTask.getLastRawObservation?.();
            if (!cactusFastRawObs) throw new Error('Cactus fast raw observation unavailable');

            // Parity: ONNX WebGPU vs ONNX WASM
            const detectionsParity = bridge.compareMirageTensorParity(
              lastWebgpuHeads.detections,
              lastWasmHeads.detections,
              2e-4,
              2e-4,
              1e-6
            );
            const prototypesParity = bridge.compareMirageTensorParity(
              lastWebgpuHeads.prototypes,
              lastWasmHeads.prototypes,
              2e-4,
              2e-4,
              1e-6
            );
            const coefficientsParity = bridge.compareMirageTensorParity(
              lastWebgpuHeads.coefficients,
              lastWasmHeads.coefficients,
              2e-4,
              2e-4,
              1e-6
            );
            const proposalParityNone = compareProposals(
              lastWebgpuDecodedNone,
              lastWasmDecodedNone
            );
            const proposalParityNms050 = compareProposals(
              lastWebgpuDecodedNms050,
              lastWasmDecodedNms050
            );

            // Parity vs Official Cactus Native (libcactus.so) & Official Cactus WASM (cactus-wasm.wasm)
            type Box = [number, number, number, number];
            const normalizeNativeBox = ([y0, x0, y1, x1]: Box): Box =>
              [x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000];
            const nativeFastObs = (
              nativeGeometry?.cases?.find(
                (candidate: { name: string }) => candidate.name === preset.name
              )?.output ?? nativeCorpus?.native_reserved?.cases?.find(
                (candidate: { name: string }) => candidate.name === preset.name
              )?.requests?.['geometry-only']?.output
            ) as { objects?: { bbox: Box }[]; text?: { bbox: Box }[] } | undefined;
            const nativeObjectBoxes = nativeFastObs?.objects?.map((o) => normalizeNativeBox(o.bbox));
            const nativeTextBoxes = nativeFastObs?.text?.map((t) => normalizeNativeBox(t.bbox));
            const cactusWasmObjectBoxes = (cactusFastRawObs.objects ?? []).map(
              (o) => normalizeNativeBox(o.bbox)
            );
            const cactusWasmTextBoxes = (cactusFastRawObs.text ?? []).map(
              (t) => normalizeNativeBox(t.bbox)
            );
            const matchNativeGeometry = (objects: Box[], text: Box[]) =>
              nativeObjectBoxes && nativeTextBoxes
                ? {
                    objects: evaluateBoxMatching(objects, nativeObjectBoxes),
                    text: evaluateBoxMatching(text, nativeTextBoxes),
                  }
                : null;

            const onnxNoneObjectBoxes = lastWasmDecodedNone.objects.map(
              (o) => o.box
            );
            const onnxNoneTextBoxes = lastWasmDecodedNone.text.map((t) => t.box);
            const onnxNmsObjectBoxes = lastWasmDecodedNms050.objects.map(
              (o) => o.box
            );
            const onnxNmsTextBoxes = lastWasmDecodedNms050.text.map((t) => t.box);

            const wasmTiming = summarizeTimes(wasmTotalTimes);
            const webgpuTiming = summarizeTimes(webgpuTotalTimes);
            const cactusTiming = summarizeTimes(cactusFastTimes);

            fastModeCases[preset.name] = {
              dimensions: [preset.width, preset.height],
              validHw: preset.prep.valid_hw,
              speed: {
                onnxWasmEncoderMs: summarizeTimes(wasmEncoderTimes),
                onnxWasmTotalMs: wasmTiming,
                onnxWebgpuEncoderMs: summarizeTimes(webgpuEncoderTimes),
                onnxWebgpuTotalMs: webgpuTiming,
                officialCactusWasmMs: cactusTiming,
                fasterOnnxProvider:
                  wasmTiming.medianMs <= webgpuTiming.medianMs
                    ? 'mirage-onnx-exp-wasm'
                    : 'mirage-onnx-exp-webgpu',
                wasmVsWebgpuSpeedupRatio: round(
                  webgpuTiming.medianMs / Math.max(0.01, wasmTiming.medianMs),
                  2
                ),
                onnxWasmVsOfficialCactusRatio: round(
                  wasmTiming.medianMs / Math.max(0.01, cactusTiming.medianMs),
                  2
                ),
              },
              webgpuVsWasmParity: {
                detections: detectionsParity,
                prototypes: prototypesParity,
                coefficients: coefficientsParity,
                allThreeHeadsAllcloseAt2e4:
                  detectionsParity.allcloseAt2e4 &&
                  prototypesParity.allcloseAt2e4 &&
                  coefficientsParity.allcloseAt2e4,
                exactProposalParityNoNms: proposalParityNone,
                exactProposalParityNms050: proposalParityNms050,
              },
              vsOfficialCactusRuntime: {
                nativeVsCactusWasmAgreement: matchNativeGeometry(cactusWasmObjectBoxes, cactusWasmTextBoxes),
                onnxNoNmsVsOfficialNative: matchNativeGeometry(onnxNoneObjectBoxes, onnxNoneTextBoxes),
                onnxNms050VsOfficialNative: matchNativeGeometry(onnxNmsObjectBoxes, onnxNmsTextBoxes),
                onnxNms050VsOfficialCactusWasm: {
                  objects: evaluateBoxMatching(
                    onnxNmsObjectBoxes,
                    cactusWasmObjectBoxes
                  ),
                  text: evaluateBoxMatching(
                    onnxNmsTextBoxes,
                    cactusWasmTextBoxes
                  ),
                },
              },
            };
          }

          // 5. FULL 4-GRAPH PIPELINE BENCHMARK & PARITY (Encoder + Reader CTC OCR + CrossKV + 8-Layer Autoregressive Decoder + Assembly)
          const fullPipelineCases: Record<string, unknown> = {};
          const MAX_TOKENS = 32;

          for (const preset of presets) {
            // Run ONNX WASM Full 4-Graph Pipeline
            const wasmT0 = performance.now();
            const wasmEncRes = await wasmTask.runRawHeads(preset.prep, {
              includeDecoderFeatures: true,
            });
            const wasmT1 = performance.now();
            if (!wasmEncRes.memory || !wasmEncRes.dense_features) {
              throw new Error('WASM encoder missing decoder features');
            }
            const wasmDecodedProps = bridge.decodeMirageExperimentalOutputs(
              wasmEncRes,
              preset.metadata,
              {
                confidence: 0.3,
                nmsIou: 0.5,
                maxObjects: 20,
                maxText: 20,
                includeMasks: true,
              }
            );
            const wasmOcrStart = performance.now();
            const wasmOcrLines = await bridge.readMirageTextLines(
              wasmFullAssets.readerSession,
              preset.rgba,
              preset.width,
              preset.height,
              wasmDecodedProps.text,
              wasmFullAssets.tableReader.charset
            );
            const wasmOcrEnd = performance.now();

            const wasmCkStart = performance.now();
            const wasmCrossKv = await bridge.runMirageCrossKv(
              wasmFullAssets.crossKvSession,
              wasmEncRes.memory,
              wasmEncRes.dense_features,
              preset.metadata,
              wasmDecodedProps.objects,
              wasmDecodedProps.text,
              wasmFullAssets.tableReader
            );
            const wasmCkEnd = performance.now();

            const wasmDecStart = performance.now();
            const wasmDecodeRes = await bridge.runMirageGreedyDecode(
              wasmFullAssets.decoderStepSession,
              wasmCrossKv,
              wasmFullAssets.tableReader,
              {
                language: 'en',
                maxTokens: MAX_TOKENS,
                captureStep0Logits: true,
              }
            );
            const wasmDecEnd = performance.now();

            const wasmAsmStart = performance.now();
            const wasmObservation = bridge.assembleMirageObservation({
              decodedText: wasmDecodeRes.decodedText,
              generatedTokenIds: wasmDecodeRes.generatedTokenIds,
              language: 'en',
              objects: wasmDecodedProps.objects,
              texts: wasmDecodedProps.text,
              ocrStrings: wasmOcrLines,
              includeMasks: true,
            });
            const wasmAsmEnd = performance.now();

            // Run ONNX WebGPU Full 4-Graph Pipeline
            const webgpuT0 = performance.now();
            const webgpuEncRes = await webgpuTask.runRawHeads(preset.prep, {
              includeDecoderFeatures: true,
            });
            const webgpuT1 = performance.now();
            if (!webgpuEncRes.memory || !webgpuEncRes.dense_features) {
              throw new Error('WebGPU encoder missing decoder features');
            }
            const webgpuDecodedProps = bridge.decodeMirageExperimentalOutputs(
              webgpuEncRes,
              preset.metadata,
              {
                confidence: 0.3,
                nmsIou: 0.5,
                maxObjects: 20,
                maxText: 20,
                includeMasks: true,
              }
            );
            const webgpuOcrStart = performance.now();
            const webgpuOcrLines = await bridge.readMirageTextLines(
              webgpuFullAssets.readerSession,
              preset.rgba,
              preset.width,
              preset.height,
              webgpuDecodedProps.text,
              webgpuFullAssets.tableReader.charset
            );
            const webgpuOcrEnd = performance.now();

            const webgpuCkStart = performance.now();
            const webgpuCrossKv = await bridge.runMirageCrossKv(
              webgpuFullAssets.crossKvSession,
              webgpuEncRes.memory,
              webgpuEncRes.dense_features,
              preset.metadata,
              webgpuDecodedProps.objects,
              webgpuDecodedProps.text,
              webgpuFullAssets.tableReader
            );
            const webgpuCkEnd = performance.now();

            const webgpuDecStart = performance.now();
            const webgpuDecodeRes = await bridge.runMirageGreedyDecode(
              webgpuFullAssets.decoderStepSession,
              webgpuCrossKv,
              webgpuFullAssets.tableReader,
              {
                language: 'en',
                maxTokens: MAX_TOKENS,
                captureStep0Logits: true,
              }
            );
            const webgpuDecEnd = performance.now();

            const webgpuAsmStart = performance.now();
            const webgpuObservation = bridge.assembleMirageObservation({
              decodedText: webgpuDecodeRes.decodedText,
              generatedTokenIds: webgpuDecodeRes.generatedTokenIds,
              language: 'en',
              objects: webgpuDecodedProps.objects,
              texts: webgpuDecodedProps.text,
              ocrStrings: webgpuOcrLines,
              includeMasks: true,
            });
            const webgpuAsmEnd = performance.now();

            // Run Official Cactus WASM Full Mode
            const cactusFullStart = performance.now();
            await cactusTask.detectForVideo(preset.canvas, 0.3, {
              analysisMode: 'full',
            });
            const cactusFullEnd = performance.now();
            const cactusFullObs = cactusTask.getLastRawObservation?.();
            if (!cactusFullObs) throw new Error('Cactus full raw observation unavailable');

            // Check Native Corpus Reference if available
            const nativeFullCase = nativeCorpus?.native_reserved?.cases?.find(
              (candidate: { name: string }) => candidate.name === preset.name
            )?.requests?.['full-score']?.output;

            // Stage-by-stage parity between WebGPU and WASM
            const memoryParity = bridge.compareMirageTensorParity(
              webgpuEncRes.memory,
              wasmEncRes.memory,
              2e-4,
              2e-4,
              1e-4
            );
            const denseFeaturesParity = bridge.compareMirageTensorParity(
              webgpuEncRes.dense_features,
              wasmEncRes.dense_features,
              2e-4,
              2e-4,
              1e-4
            );
            const crossKParity = bridge.compareMirageTensorParity(
              webgpuCrossKv.crossK,
              wasmCrossKv.crossK,
              2e-4,
              2e-4,
              1e-4
            );
            const crossVParity = bridge.compareMirageTensorParity(
              webgpuCrossKv.crossV,
              wasmCrossKv.crossV,
              2e-4,
              2e-4,
              1e-4
            );
            const step0LogitsParity =
              webgpuDecodeRes.step0Logits && wasmDecodeRes.step0Logits
                ? bridge.compareMirageTensorParity(
                    webgpuDecodeRes.step0Logits,
                    wasmDecodeRes.step0Logits,
                    2e-4,
                    2e-4,
                    1e-4
                  )
                : null;

            const ocrExactMatch =
              webgpuOcrLines.length === wasmOcrLines.length &&
              webgpuOcrLines.every((line, idx) => line === wasmOcrLines[idx]);
            const tokenIdsExactMatch =
              webgpuDecodeRes.generatedTokenIds.length ===
                wasmDecodeRes.generatedTokenIds.length &&
              webgpuDecodeRes.generatedTokenIds.every(
                (id, idx) => id === wasmDecodeRes.generatedTokenIds[idx]
              );
            const decodedTextExactMatch =
              webgpuDecodeRes.decodedText === wasmDecodeRes.decodedText;

            const wasmTotalFullMs = round(
              wasmT1 -
                wasmT0 +
                (wasmOcrEnd - wasmOcrStart) +
                (wasmCkEnd - wasmCkStart) +
                (wasmDecEnd - wasmDecStart) +
                (wasmAsmEnd - wasmAsmStart),
              2
            );
            const webgpuTotalFullMs = round(
              webgpuT1 -
                webgpuT0 +
                (webgpuOcrEnd - webgpuOcrStart) +
                (webgpuCkEnd - webgpuCkStart) +
                (webgpuDecEnd - webgpuDecStart) +
                (webgpuAsmEnd - webgpuAsmStart),
              2
            );
            const cactusFullMs = round(cactusFullEnd - cactusFullStart, 2);

            fullPipelineCases[preset.name] = {
              speedBreakdownMs: {
                onnxWasm: {
                  encoderMs: round(wasmT1 - wasmT0, 2),
                  readerOcrMs: round(wasmOcrEnd - wasmOcrStart, 2),
                  crossKvMs: round(wasmCkEnd - wasmCkStart, 2),
                  decoderTotalMs: round(wasmDecEnd - wasmDecStart, 2),
                  decoderSteps: wasmDecodeRes.generatedTokenIds.length,
                  decoderPerStepMs: round(
                    (wasmDecEnd - wasmDecStart) /
                      Math.max(1, wasmDecodeRes.generatedTokenIds.length),
                    2
                  ),
                  assemblyMs: round(wasmAsmEnd - wasmAsmStart, 2),
                  totalMs: wasmTotalFullMs,
                },
                onnxWebgpu: {
                  encoderMs: round(webgpuT1 - webgpuT0, 2),
                  readerOcrMs: round(webgpuOcrEnd - webgpuOcrStart, 2),
                  crossKvMs: round(webgpuCkEnd - webgpuCkStart, 2),
                  decoderTotalMs: round(webgpuDecEnd - webgpuDecStart, 2),
                  decoderSteps: webgpuDecodeRes.generatedTokenIds.length,
                  decoderPerStepMs: round(
                    (webgpuDecEnd - webgpuDecStart) /
                      Math.max(1, webgpuDecodeRes.generatedTokenIds.length),
                    2
                  ),
                  assemblyMs: round(webgpuAsmEnd - webgpuAsmStart, 2),
                  totalMs: webgpuTotalFullMs,
                },
                officialCactusWasm: {
                  totalMs: cactusFullMs,
                },
                fasterOnnxProvider:
                  wasmTotalFullMs <= webgpuTotalFullMs
                    ? 'mirage-onnx-exp-wasm'
                    : 'mirage-onnx-exp-webgpu',
                wasmVsWebgpuSpeedupRatio: round(
                  webgpuTotalFullMs / Math.max(0.01, wasmTotalFullMs),
                  2
                ),
              },
              webgpuVsWasmParity: {
                memory: memoryParity,
                denseFeatures: denseFeaturesParity,
                crossK: crossKParity,
                crossV: crossVParity,
                step0Logits: step0LogitsParity,
                ocrExactMatch,
                tokenIdsExactMatch,
                decodedTextExactMatch,
                wasmOcrLines,
                webgpuOcrLines,
                wasmGeneratedTokenIds: wasmDecodeRes.generatedTokenIds,
                webgpuGeneratedTokenIds: webgpuDecodeRes.generatedTokenIds,
                wasmDecodedText: wasmDecodeRes.decodedText,
                webgpuDecodedText: webgpuDecodeRes.decodedText,
              },
              vsOfficialCactusFullObservation: {
                onnxWasmAssembledSummary: {
                  objectCount: wasmObservation.objects.length,
                  objectLabels: wasmObservation.objects.map(
                    (o) => o.score === undefined ? o.name : `${o.name} (${round(o.score, 2)})`
                  ),
                  regionCount: wasmObservation.regions.length,
                  regionLabels: wasmObservation.regions.map(
                    (r) => r.name
                  ),
                  textCount: wasmObservation.text.length,
                  textItems: wasmObservation.text.map(
                    (t) => `${t.string} (${round(t.score, 2)})`
                  ),
                  relationCount: wasmObservation.relations.length,
                },
                onnxWebgpuAssembledSummary: {
                  objectCount: webgpuObservation.objects.length,
                  objectLabels: webgpuObservation.objects.map(
                    (o) => o.score === undefined ? o.name : `${o.name} (${round(o.score, 2)})`
                  ),
                  regionCount: webgpuObservation.regions.length,
                  regionLabels: webgpuObservation.regions.map(
                    (r) => r.name
                  ),
                  textCount: webgpuObservation.text.length,
                  textItems: webgpuObservation.text.map(
                    (t) => `${t.string} (${round(t.score, 2)})`
                  ),
                  relationCount: webgpuObservation.relations.length,
                },
                officialCactusWasmSummary: {
                  objectCount: cactusFullObs.objects?.length ?? 0,
                  objectLabels: (cactusFullObs.objects ?? []).map(
                    (o) => o.score === undefined ? o.name : `${o.name} (${round(o.score, 2)})`
                  ),
                  regionCount: cactusFullObs.regions?.length ?? 0,
                  regionLabels: (cactusFullObs.regions ?? []).map(
                    (r) => `${r.name} (${round(r.score, 2)})`
                  ),
                  textCount: cactusFullObs.text?.length ?? 0,
                  textItems: (cactusFullObs.text ?? []).map(
                    (t) => `${t.string} (${round(t.score, 2)})`
                  ),
                  relationCount: cactusFullObs.relations?.length ?? 0,
                },
                officialNativeSummary: nativeFullCase
                  ? {
                      objectCount: (nativeFullCase.objects ?? []).length,
                      objectLabels: (nativeFullCase.objects ?? []).map(
                        (o: { name: string; score?: number }) => o.score === undefined
                          ? o.name : `${o.name} (${round(o.score, 2)})`
                      ),
                      regionCount: (nativeFullCase.regions ?? []).length,
                      regionLabels: (nativeFullCase.regions ?? []).map(
                        (r: { name: string; score?: number }) => r.score === undefined
                          ? r.name : `${r.name} (${round(r.score, 2)})`
                      ),
                      textCount: (nativeFullCase.text ?? []).length,
                      textItems: (nativeFullCase.text ?? []).map(
                        (t: { string: string; score?: number }) => t.score === undefined
                          ? t.string : `${t.string} (${round(t.score, 2)})`
                      ),
                      relationCount: (nativeFullCase.relations ?? []).length,
                    }
                  : null,
              },
            };
          }

          return {
            schemaVersion: 1,
            recordedAt: new Date().toISOString(),
            environment: {
              userAgent: navigator.userAgent,
              hardwareConcurrency: navigator.hardwareConcurrency ?? null,
              webgpuAdapter,
            },
            sessionInitializationMs: {
              onnxWasmEncoderMs: wasmEncoderInitMs,
              onnxWasmFullPipelineAssetsMs: wasmFullPipelineInitMs,
              onnxWasmTotalInitMs: round(
                wasmEncoderInitMs + wasmFullPipelineInitMs,
                2
              ),
              onnxWebgpuEncoderMs: webgpuEncoderInitMs,
              onnxWebgpuFullPipelineAssetsMs: webgpuFullPipelineInitMs,
              onnxWebgpuTotalInitMs: round(
                webgpuEncoderInitMs + webgpuFullPipelineInitMs,
                2
              ),
              officialCactusWasmInitMs: cactusWasmInitMs,
              backends: {
                onnxWasmEncoder: wasmTask.backend,
                onnxWasmReader: wasmFullAssets.readerBackend,
                onnxWasmCrossKv: wasmFullAssets.crossKvBackend,
                onnxWasmDecoderStep: wasmFullAssets.decoderStepBackend,
                onnxWebgpuEncoder: webgpuTask.backend,
                onnxWebgpuReader: webgpuFullAssets.readerBackend,
                onnxWebgpuCrossKv: webgpuFullAssets.crossKvBackend,
                onnxWebgpuDecoderStep: webgpuFullAssets.decoderStepBackend,
              },
            },
            fastProposalMode: {
              benchmarkRunsPerImage: BENCHMARK_RUNS,
              cases: fastModeCases,
            },
            full4GraphPipelineMode: {
              maxTokens: MAX_TOKENS,
              language: 'en',
              cases: fullPipelineCases,
            },
          };
        } finally {
          await wasmFullAssets.readerSession.release?.().catch(() => undefined);
          await wasmFullAssets.crossKvSession.release?.().catch(() => undefined);
          await wasmFullAssets.decoderStepSession
            .release?.()
            .catch(() => undefined);
          await webgpuFullAssets.readerSession
            .release?.()
            .catch(() => undefined);
          await webgpuFullAssets.crossKvSession
            .release?.()
            .catch(() => undefined);
          await webgpuFullAssets.decoderStepSession
            .release?.()
            .catch(() => undefined);
          await wasmTask.close();
          await webgpuTask.close();
          cactusTask.close();
        }
      },
      { nativeGeometry: nativeGeometryRaw, nativeCorpus: nativeCorpusRaw }
    );

    const outPath = testInfo.outputPath('mirage-onnx-parity-and-speed.json');
    const serialized = JSON.stringify(benchmarkReport, null, 2) + '\n';
    writeFileSync(outPath, serialized, 'utf8');
    await testInfo.attach('mirage-onnx-parity-and-speed.json', {
      path: outPath,
      contentType: 'application/json',
    });

    if (process.env.WRITE_MIRAGE_PARITY_REPORT === '1') {
      writeFileSync(OUTPUT_DOCS_JSON_PATH, serialized, 'utf8');
    }

    // Assertions:
    // 1. Verify strict session configuration, not measured kernel placement.
    const backends = benchmarkReport.sessionInitializationMs.backends;
    for (const [backend, provider] of [
      [backends.onnxWasmEncoder, 'wasm'],
      [backends.onnxWebgpuEncoder, 'webgpu'],
      [backends.onnxWebgpuDecoderStep, 'webgpu'],
    ] as const) {
      expect(backend).toMatchObject({
        requestedProvider: provider,
        configuredProviders: [provider],
        fallbackReason: null,
        actualBackend: 'unmeasured',
      });
    }

    // 2. Fast Proposal Mode: WebGPU vs WASM parity holds across all 3 cases
    for (const [caseName, caseData] of Object.entries(
      benchmarkReport.fastProposalMode.cases as Record<
        string,
        {
          webgpuVsWasmParity: {
            allThreeHeadsAllcloseAt2e4: boolean;
            exactProposalParityNoNms: { exactMatch: boolean };
            exactProposalParityNms050: { exactMatch: boolean };
          };
        }
      >
    )) {
      expect(
        caseData.webgpuVsWasmParity.allThreeHeadsAllcloseAt2e4,
        `${caseName}: WebGPU vs WASM 3-head allcloseAt2e4`
      ).toBe(true);
      expect(
        caseData.webgpuVsWasmParity.exactProposalParityNoNms.exactMatch,
        `${caseName}: WebGPU vs WASM exact decoded proposal match (no NMS)`
      ).toBe(true);
      expect(
        caseData.webgpuVsWasmParity.exactProposalParityNms050.exactMatch,
        `${caseName}: WebGPU vs WASM exact decoded proposal match (NMS=0.5)`
      ).toBe(true);
    }

    // 3. Full 4-Graph Pipeline Mode: WebGPU vs WASM parity holds across all 3 cases
    for (const [caseName, caseData] of Object.entries(
      benchmarkReport.full4GraphPipelineMode.cases as Record<
        string,
        {
          webgpuVsWasmParity: {
            memory: { allcloseAt2e4: boolean };
            crossK: { allcloseAt2e4: boolean };
            crossV: { cosineSimilarity: number };
            step0Logits: { allcloseAt2e4: boolean } | null;
            ocrExactMatch: boolean;
            tokenIdsExactMatch: boolean;
            decodedTextExactMatch: boolean;
          };
        }
      >
    )) {
      expect(
        caseData.webgpuVsWasmParity.memory.allcloseAt2e4,
        `${caseName}: WebGPU vs WASM encoder memory allcloseAt2e4`
      ).toBe(true);
      expect(
        caseData.webgpuVsWasmParity.crossK.allcloseAt2e4,
        `${caseName}: WebGPU vs WASM crossK allcloseAt2e4`
      ).toBe(true);
      expect(
        caseData.webgpuVsWasmParity.crossV.cosineSimilarity,
        `${caseName}: WebGPU vs WASM crossV cosineSimilarity`
      ).toBeGreaterThanOrEqual(0.999999);
      expect(
        caseData.webgpuVsWasmParity.step0Logits?.allcloseAt2e4,
        `${caseName}: WebGPU vs WASM step0Logits allcloseAt2e4`
      ).toBe(true);
      expect(
        caseData.webgpuVsWasmParity.ocrExactMatch,
        `${caseName}: WebGPU vs WASM Reader CTC OCR exact match`
      ).toBe(true);
      expect(
        caseData.webgpuVsWasmParity.tokenIdsExactMatch,
        `${caseName}: WebGPU vs WASM 32-step greedy token IDs exact match`
      ).toBe(true);
      expect(
        caseData.webgpuVsWasmParity.decodedTextExactMatch,
        `${caseName}: WebGPU vs WASM decoded text exact match`
      ).toBe(true);
    }
  });
});
