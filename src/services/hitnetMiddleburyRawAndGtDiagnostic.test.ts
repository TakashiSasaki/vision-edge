// @vitest-environment node
import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  HITNET_MIDDLEBURY_PIXEL_COUNT,
  loadAndVerifyMiddleburySceneAssets,
  type MiddleburyEval3SceneId,
} from './hitnetMiddleburyEval3Benchmark';
import {
  analyzeMiddleburySourceGtConsistency,
  decodePngRgbTopToBottomBitExact,
  EXPECTED_RAW_HITNET_BYTE_LENGTH,
  EXPECTED_RAW_HITNET_PROXY_PATH,
  runHitnetMiddleburyRawAndGtDiagnostic,
  verifyRawHitnetProvenanceAndContract,
} from './hitnetMiddleburyRawAndGtDiagnostic';
import { type ResidualDiagnosticWorkerRunner } from './hitnetResidualLocalization';
import { HITNET_DEV_NON_JSEP_FIXTURE_SPEC } from './stereoDepthWebGpuDevFixture';
import { HITNET_M1_FULL_GRAPH_SHA256, HITNET_RAW_MODEL_SHA256 } from './stereoDepthWorkerCore';

function nodeSha256Hex(bytes: Uint8Array): string {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function createDiskPlusProxyFetch(rawOnnxBytes: Uint8Array | null, rawStatus = 200): typeof fetch {
  const publicDir = path.resolve(process.cwd(), 'public');
  return (async (input: string | URL | Request) => {
    const urlStr = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    if (urlStr === EXPECTED_RAW_HITNET_PROXY_PATH) {
      if (rawStatus !== 200 || !rawOnnxBytes) {
        return new Response('Not found', { status: rawStatus });
      }
      return new Response(new Uint8Array(rawOnnxBytes), {
        status: 200,
        headers: {
          'content-type': 'application/octet-stream',
          'x-proxy-cache': 'stateless-stream',
          'x-upstream-host': 'github.com',
        },
      });
    }
    const rel = urlStr.replace(/^\/+/, '');
    const fullPath = path.resolve(publicDir, rel);
    try {
      const bytes = await fsp.readFile(fullPath);
      return new Response(new Uint8Array(bytes), { status: 200 });
    } catch {
      return new Response('<!doctype html><html>404</html>', {
        status: 404,
        headers: { 'content-type': 'text/html; charset=utf-8' },
      });
    }
  }) as unknown as typeof fetch;
}

describe('hitnetMiddleburyRawAndGtDiagnostic', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('verifies real MiddEval3 source-native GT consistency across Motorcycle (full + upper-left trouble ROI vs control ROI) and Teddy', async () => {
    const fetchDisk = createDiskPlusProxyFetch(null, 404);
    const mcycle = await loadAndVerifyMiddleburySceneAssets('Motorcycle', fetchDisk);
    const teddy = await loadAndVerifyMiddleburySceneAssets('Teddy', fetchDisk);

    const im0Mcycle = await decodePngRgbTopToBottomBitExact(mcycle.im0Bytes);
    expect(im0Mcycle.width).toBe(741);
    expect(im0Mcycle.height).toBe(497);
    expect(im0Mcycle.rgbTopToBottom.length).toBe(741 * 497 * 3);

    const mcycleGt = await analyzeMiddleburySourceGtConsistency({
      sceneAssets: mcycle,
      fetchImpl: fetchDisk,
    });
    expect(mcycleGt.yGtMapStatus.available).toBe(false);
    expect(mcycleGt.yGtMapStatus.calibDyavgPx).toBeCloseTo(0.23, 5);
    expect(mcycleGt.yGtMapStatus.calibDymaxPx).toBeCloseTo(0.379, 5);
    expect(mcycleGt.rois).toHaveLength(6); // 3 ROIs x 2 eyes

    for (const r of mcycleGt.rois) {
      expect(r.evaluatedRecipCount).toBeGreaterThan(25000);
      expect(r.oppositeNonoccHitRate).toBeGreaterThan(0.98);
      expect(r.nearestRecipAbsMeanPx).toBeLessThan(0.05);
      expect(r.nearestRecipLe1PxRate).toBeGreaterThan(0.99);
      expect(r.bilinearRecipAbsMeanPx).toBeLessThan(0.10);
      expect(r.zncc5x5TexturedMean).toBeGreaterThan(0.85);
      expect(r.signAndUnitCheck.confirmedUnitAndSign).toBe(true);
      expect(r.signAndUnitCheck.positive1xZncc5x5).toBeGreaterThan(
        r.signAndUnitCheck.negatedSignZncc5x5 + 0.5
      );
      expect(r.signAndUnitCheck.positive1xZncc5x5).toBeGreaterThan(
        r.signAndUnitCheck.halfScale05xZncc5x5 + 0.3
      );
    }

    const teddyGt = await analyzeMiddleburySourceGtConsistency({
      sceneAssets: teddy,
      fetchImpl: fetchDisk,
    });
    expect(teddyGt.yGtMapStatus.calibDyavgPx).toBe(0);
    expect(teddyGt.yGtMapStatus.calibDymaxPx).toBe(0);
    expect(teddyGt.rois).toHaveLength(2);
    for (const r of teddyGt.rois) {
      expect(r.evaluatedRecipCount).toBeGreaterThan(100000);
      expect(r.nearestRecipAbsMeanPx).toBeLessThan(0.05);
      expect(r.zncc5x5TexturedMean).toBeGreaterThan(0.85);
      expect(r.signAndUnitCheck.confirmedUnitAndSign).toBe(true);
    }
  }, 30000);

  it('aborts RAW comparison without weight substitution when stereodemo raw provenance/SHA fails', async () => {
    const fakeWrongBytes = new Uint8Array(EXPECTED_RAW_HITNET_BYTE_LENGTH).fill(0x11);
    const fetchWrong = createDiskPlusProxyFetch(fakeWrongBytes, 200);

    const prov = await verifyRawHitnetProvenanceAndContract(fetchWrong);
    expect(prov.check.verified).toBe(false);
    expect(prov.check.failureReason).toMatch(/Raw HITNet release binary mismatch/);
    expect(prov.rawBytes).toBeNull();

    const workerCalls: number[] = [];
    const shouldNotBeCalledRunner: ResidualDiagnosticWorkerRunner = async () => {
      workerCalls.push(1);
      throw new Error('Should not run inference when raw provenance fails');
    };

    const report = await runHitnetMiddleburyRawAndGtDiagnostic({
      isDev: true,
      fetchImpl: fetchWrong,
      runWorker: shouldNotBeCalledRunner,
    });

    expect(report.rawProvenance.verified).toBe(false);
    expect(report.rawComparisonExecuted).toBe(false);
    expect(report.rawComparisonAbortedReason).toMatch(/Raw HITNet release binary mismatch/);
    expect(workerCalls).toHaveLength(0);
    // Source-GT consistency check still completes independently
    expect(report.sourceGtConsistencyByScene.Motorcycle.rois).toHaveLength(6);
    expect(report.sourceGtConsistencyByScene.Teddy.rois).toHaveLength(2);
  }, 30000);

  it('verifies stereodemo raw bytes and runs raw vs canonical M1 G0 RGB/BGR WASM parity & GT accuracy when raw binary is present', async () => {
    const rawPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (!fs.existsSync(rawPath)) {
      return;
    }
    const rawOnnx = new Uint8Array(fs.readFileSync(rawPath));
    expect(rawOnnx.byteLength).toBe(EXPECTED_RAW_HITNET_BYTE_LENGTH);
    expect(nodeSha256Hex(rawOnnx)).toBe(HITNET_RAW_MODEL_SHA256);

    const fetchValid = createDiskPlusProxyFetch(rawOnnx, 200);
    const prov = await verifyRawHitnetProvenanceAndContract(fetchValid);
    expect(prov.check.verified).toBe(true);
    expect(prov.check.contractMatches).toBe(true);
    expect(prov.check.rawSha256).toBe(HITNET_RAW_MODEL_SHA256);
    expect(prov.check.rawNodeCount).toBe(686);

    const calls: Array<{ modelSha: string; feedSha: string; provider: string; opt: string }> = [];
    const mockWasmRunner: ResidualDiagnosticWorkerRunner = async (req) => {
      const modelSha = nodeSha256Hex(new Uint8Array(req.modelBuffer));
      const feedSha = nodeSha256Hex(new Uint8Array(req.feedBuffer));
      calls.push({
        modelSha,
        feedSha,
        provider: req.provider,
        opt: req.graphOptimizationLevel,
      });

      const isM1 = modelSha === HITNET_M1_FULL_GRAPH_SHA256;
      const refOut = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT).fill(isM1 ? 10.00001 : 10.0);
      const secOut = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT).fill(10.0);

      return {
        id: 1,
        ok: true,
        pinnedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        provider: req.provider,
        graphOptimizationLevel: req.graphOptimizationLevel,
        fallbackBehavior: 'no-fallback-explicit-error',
        configuredProviders: [req.provider],
        entryModuleUrl: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs',
        wasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
        ortVersions: { common: '1.30.0', web: '1.30.0' },
        loadedOrtResources: [
          'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs',
          'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.asyncify.wasm',
        ],
        foreignOrtResources: [],
        loadedAssetProvenance: [
          {
            role: 'wasm',
            url: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.asyncify.wasm',
            byteLength: 1000,
            sha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedWasmSha256,
            via: 'WebAssembly.instantiateStreaming',
            referenceLocalDistSha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedWasmSha256,
            matchesLocalDistSha256: true,
          },
        ],
        sessionOptionsApplied: {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'basic',
          enableCpuMemArena: false,
          enableMemPattern: false,
          numThreads: 1,
          proxy: false,
          wasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
          disableCpuEpFallback: false,
          fetchMode: 'all-session-outputs',
          profilingRequested: false,
        },
        feedTensorName: 'input',
        feedTensorShape: [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        sessionInputNames: ['input'],
        sessionOutputNames: ['reference_output_disparity', 'secondary_output_disparity'],
        workerGpuAdapter: null,
        gpuValidationErrors: null,
        gpuKernelProfileSummary: null,
        modelByteLength: req.modelBuffer.byteLength,
        modelSha256: modelSha,
        feedSha256: feedSha,
        createMs: 15,
        coldRunMs: 10,
        warmRunMs: [8],
        outputs: [
          {
            name: 'reference_output_disparity',
            dims: [1, 240, 320, 1],
            dataType: 'float32',
            buffer: refOut.buffer,
          },
          {
            name: 'secondary_output_disparity',
            dims: [1, 240, 320, 1],
            dataType: 'float32',
            buffer: secOut.buffer,
          },
        ],
      };
    };

    const report = await runHitnetMiddleburyRawAndGtDiagnostic({
      isDev: true,
      fetchImpl: fetchValid,
      runWorker: mockWasmRunner,
      loadSceneAssets: (id: MiddleburyEval3SceneId) => loadAndVerifyMiddleburySceneAssets(id, fetchValid),
      buildGeometryRgba: async ({ sceneAssets }) => {
        const left = new Uint8ClampedArray(HITNET_MIDDLEBURY_PIXEL_COUNT * 4);
        const right = new Uint8ClampedArray(HITNET_MIDDLEBURY_PIXEL_COUNT * 4);
        left[0] = sceneAssets.im0Bytes[40];
        right[0] = sceneAssets.im1Bytes[40];
        return {
          G0: { leftRgba: left, rightRgba: right },
          G1: { leftRgba: left, rightRgba: right },
        };
      },
    });

    expect(report.rawComparisonExecuted).toBe(true);
    expect(report.m1Sha256).toBe(HITNET_M1_FULL_GRAPH_SHA256);
    expect(report.n1OffProof?.verified).toBe(true);
    // 2 scenes x 2 feeds (RGB/BGR) x 2 models (raw/M1) = 8 WASM calls
    expect(calls).toHaveLength(8);
    expect(calls.every((c) => c.provider === 'wasm' && c.opt === 'basic')).toBe(true);
    expect(calls.filter((c) => c.modelSha === HITNET_RAW_MODEL_SHA256)).toHaveLength(4);
    expect(calls.filter((c) => c.modelSha === HITNET_M1_FULL_GRAPH_SHA256)).toHaveLength(4);

    const mcycleRgb = report.rawVsM1ByScene.Motorcycle!.byFeed.C0_RGB;
    expect(mcycleRgb.rawArm.modelSha256).toBe(HITNET_RAW_MODEL_SHA256);
    expect(mcycleRgb.m1Arm.modelSha256).toBe(HITNET_M1_FULL_GRAPH_SHA256);
    expect(mcycleRgb.rawArm.feedSha256).toBe(mcycleRgb.m1Arm.feedSha256);
    expect(mcycleRgb.fullTensorParityRawVsM1.reference_output_disparity.maxAbsDiff).toBeCloseTo(1e-5, 6);
    expect(mcycleRgb.fullTensorParityRawVsM1.secondary_output_disparity.maxAbsDiff).toBe(0);
    expect(report.diagnosisSummary.rawVsM1Verdict).toMatch(/^RAW_VS_M1_MATCHED:/);
    expect(report.diagnosisSummary.sourceGtConsistencyVerdict).toMatch(/^SOURCE_GT_CONSISTENT:/);
  }, 30000);
});
