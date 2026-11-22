// @vitest-environment node
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  HITNET_MIDDLEBURY_MODEL_HEIGHT,
  HITNET_MIDDLEBURY_MODEL_WIDTH,
  HITNET_MIDDLEBURY_PIXEL_COUNT,
  loadAndVerifyMiddleburySceneAssets,
  type MiddleburyEval3SceneId,
} from './hitnetMiddleburyEval3Benchmark';
import {
  buildMiddleburyGeometryTransforms,
  buildSharedPhysicalSampleSetForEye,
  evaluateFactorialArmEye,
  FACTORIAL_ARM_SPECS,
  packRgbaStereoEyesWithChannelOrder,
  runHitnetMiddleburyFactorialDiagnosis,
} from './hitnetMiddleburyFactorialDiagnosis';
import { type ResidualDiagnosticWorkerRunner } from './hitnetResidualLocalization';
import { HITNET_DEV_NON_JSEP_FIXTURE_SPEC } from './stereoDepthWebGpuDevFixture';
import { HITNET_M1_FULL_GRAPH_SHA256 } from './stereoDepthWorkerCore';

const textEncoder = new TextEncoder();

function nodeSha256Hex(bytes: Uint8Array): string {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function hexToArrayBuffer(hex: string): ArrayBuffer {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return bytes.buffer;
}

function encodeVarint(value: number): Uint8Array {
  const bytes: number[] = [];
  let v = value >>> 0;
  while (v >= 0x80) {
    bytes.push((v & 0x7f) | 0x80);
    v >>>= 7;
  }
  bytes.push(v);
  return new Uint8Array(bytes);
}

function concatBytes(parts: readonly Uint8Array[]): Uint8Array {
  let total = 0;
  for (const p of parts) total += p.byteLength;
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.byteLength;
  }
  return out;
}

function encodeLenField(fn: number, payload: Uint8Array): Uint8Array {
  return concatBytes([encodeVarint((fn << 3) | 2), encodeVarint(payload.byteLength), payload]);
}

function encodeVarintField(fn: number, val: number): Uint8Array {
  return concatBytes([encodeVarint((fn << 3) | 0), encodeVarint(val)]);
}

function encodeStrField(fn: number, value: string): Uint8Array {
  return encodeLenField(fn, textEncoder.encode(value));
}

function encodeNode(
  opType: string,
  name: string,
  inputs: readonly string[],
  outputs: readonly string[]
): Uint8Array {
  return encodeLenField(
    1,
    concatBytes([
      ...inputs.map((inp) => encodeStrField(1, inp)),
      ...outputs.map((out) => encodeStrField(2, out)),
      encodeStrField(3, name),
      encodeStrField(4, opType),
    ])
  );
}

function encodeValueInfoFloat(name: string, dims: readonly number[]): Uint8Array {
  const dimFields = dims.map((d) => encodeLenField(1, encodeVarintField(1, d)));
  const shapeProto = encodeLenField(2, concatBytes(dimFields));
  const tensorTypeProto = encodeLenField(1, concatBytes([encodeVarintField(1, 1), shapeProto]));
  return concatBytes([encodeStrField(1, name), encodeLenField(2, tensorTypeProto)]);
}

function encodeInt64Initializer(name: string, vals: readonly number[]): Uint8Array {
  const raw = new Uint8Array(vals.length * 8);
  const dv = new DataView(raw.buffer);
  for (let i = 0; i < vals.length; i++) dv.setBigInt64(i * 8, BigInt(vals[i]), true);
  return encodeLenField(
    5,
    concatBytes([
      encodeVarintField(1, vals.length),
      encodeVarintField(2, 7),
      encodeStrField(8, name),
      encodeLenField(9, raw),
    ])
  );
}

function buildMinimalCanonicalM1Stub(): Uint8Array {
  const nodes = [
    encodeNode('Reshape', 'Transpose__908', ['input', 'slot0_shape'], ['Transpose__908:0']),
    encodeNode('Reshape', 'Transpose__1006', ['input', 'slot0_shape'], ['Transpose__1006:0']),
    encodeNode('Reshape', 'prim/head', ['input', 'out_shape'], ['reference_output_disparity']),
    encodeNode('Reshape', 'sec/head', ['input', 'out_shape'], ['secondary_output_disparity']),
  ];
  const graphBody = concatBytes([
    ...nodes,
    encodeStrField(2, 'middlebury_factorial_stub'),
    encodeInt64Initializer('slot0_shape', [1, 6, 240, 320]),
    encodeInt64Initializer('out_shape', [1, 240, 320, 1]),
    encodeLenField(11, encodeValueInfoFloat('input', [1, 6, 240, 320])),
    encodeLenField(12, encodeValueInfoFloat('reference_output_disparity', [1, 240, 320, 1])),
    encodeLenField(12, encodeValueInfoFloat('secondary_output_disparity', [1, 240, 320, 1])),
  ]);
  const opsetImport = encodeLenField(8, concatBytes([encodeStrField(1, ''), encodeVarintField(2, 12)]));
  const baseModel = concatBytes([
    encodeVarintField(1, 7),
    encodeLenField(7, graphBody),
    opsetImport,
  ]);
  const targetLen = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength;
  const padKey = encodeStrField(1, 'pad');
  let valLen = targetLen - (baseModel.byteLength + padKey.byteLength + 6);
  while (true) {
    const valField = encodeLenField(2, new Uint8Array(Math.max(0, valLen)).fill(0x61));
    const entry = encodeLenField(14, concatBytes([padKey, valField]));
    const total = baseModel.byteLength + entry.byteLength;
    if (total === targetLen) return concatBytes([baseModel, entry]);
    valLen += targetLen - total;
  }
}

function createDiskFixtureFetch(): typeof fetch {
  const publicDir = path.resolve(process.cwd(), 'public');
  return (async (input: string | URL | Request) => {
    const urlStr = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    const rel = urlStr.replace(/^\/+/, '');
    const fullPath = path.resolve(publicDir, rel);
    const bytes = await fs.readFile(fullPath);
    return new Response(new Uint8Array(bytes), { status: 200 });
  }) as unknown as typeof fetch;
}

describe('hitnetMiddleburyFactorialDiagnosis', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('computes exact fractional isotropic G1 transforms (scaleX=scaleY) with pad/crop exclusions for Motorcycle & Teddy', () => {
    const mcycle = buildMiddleburyGeometryTransforms('Motorcycle', 741, 497);
    expect(mcycle.g0.modeKind).toBe('direct-anisotropic');
    expect(mcycle.g0.scaleX).toBeCloseTo(320 / 741, 10);
    expect(mcycle.g0.scaleY).toBeCloseTo(240 / 497, 10);
    expect(mcycle.g1.modeKind).toBe('centered-vertical-letterbox');
    expect(mcycle.g1.scaleX).toBeCloseTo(320 / 741, 10);
    expect(mcycle.g1.scaleY).toBeCloseTo(mcycle.g1.scaleX, 12);
    expect(mcycle.g1.scaleYOverScaleX).toBe(1.0);
    expect(mcycle.g1.dstRect.y).toBeCloseTo(12.685560054, 6);
    expect(mcycle.g1.dstRect.height).toBeCloseTo(214.62887989, 6);
    expect(mcycle.g1.activeFullModelRowStart).toBe(13);
    expect(mcycle.g1.activeFullModelRowEndInclusive).toBe(226);
    expect(mcycle.g1.excludedBoundaryPadModelRows).toEqual([
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    ]);

    const teddy = buildMiddleburyGeometryTransforms('Teddy', 450, 375);
    expect(teddy.g1.modeKind).toBe('centered-vertical-crop');
    expect(teddy.g1.scaleX).toBeCloseTo(320 / 450, 10);
    expect(teddy.g1.scaleYOverScaleX).toBe(1.0);
    expect(teddy.g1.srcRect).toEqual({ x: 0, y: 18.75, width: 450, height: 337.5 });
    expect(teddy.g1.dstRect).toEqual({ x: 0, y: 0, width: 320, height: 240 });
    expect(teddy.g1.activeFullModelRowStart).toBe(0);
    expect(teddy.g1.activeFullModelRowEndInclusive).toBe(239);
    expect(teddy.g1.excludedBoundaryPadModelRows).toEqual([]);
  });

  it('swaps strictly R and B in both eyes for C1_BGR while preserving Green and L/R eye separation', () => {
    const leftRgba = new Uint8ClampedArray(HITNET_MIDDLEBURY_PIXEL_COUNT * 4);
    const rightRgba = new Uint8ClampedArray(HITNET_MIDDLEBURY_PIXEL_COUNT * 4);
    leftRgba[0] = 51;
    leftRgba[1] = 102;
    leftRgba[2] = 204;
    leftRgba[3] = 255;

    const rIdx = 777 * 4;
    rightRgba[rIdx] = 25;
    rightRgba[rIdx + 1] = 150;
    rightRgba[rIdx + 2] = 225;
    rightRgba[rIdx + 3] = 255;

    const feedRgb = packRgbaStereoEyesWithChannelOrder(leftRgba, rightRgba, 'C0_RGB');
    const feedBgr = packRgbaStereoEyesWithChannelOrder(leftRgba, rightRgba, 'C1_BGR');
    const px = HITNET_MIDDLEBURY_PIXEL_COUNT;

    expect(feedBgr[0]).toBeCloseTo(feedRgb[px * 2], 6); // Left B -> ch0
    expect(feedBgr[px]).toBeCloseTo(feedRgb[px], 6); // Left G preserved
    expect(feedBgr[px * 2]).toBeCloseTo(feedRgb[0], 6); // Left R -> ch2
    expect(feedBgr[px * 3 + 777]).toBeCloseTo(feedRgb[px * 5 + 777], 6); // Right B -> ch3
    expect(feedBgr[px * 4 + 777]).toBeCloseTo(feedRgb[px * 4 + 777], 6); // Right G preserved
    expect(feedBgr[px * 5 + 777]).toBeCloseTo(feedRgb[px * 3 + 777], 6); // Right R -> ch5
  });

  it('builds strictly unique intersected physical source samples and consistent stratified evaluation buckets', async () => {
    const fetchDisk = createDiskFixtureFetch();
    const mcycle = await loadAndVerifyMiddleburySceneAssets('Motorcycle', fetchDisk);
    const { g0, g1 } = buildMiddleburyGeometryTransforms('Motorcycle', mcycle.calib.width, mcycle.calib.height);

    const leftSet = buildSharedPhysicalSampleSetForEye({
      eye: 'left',
      sceneId: 'Motorcycle',
      gtPfm: mcycle.disp0GtPfm,
      maskNocc: mcycle.mask0Nocc,
      calib: mcycle.calib,
      g0,
      g1,
    });
    const rightSet = buildSharedPhysicalSampleSetForEye({
      eye: 'right',
      sceneId: 'Motorcycle',
      gtPfm: mcycle.disp1GtPfm,
      maskNocc: mcycle.mask1Nocc,
      calib: mcycle.calib,
      g0,
      g1,
    });

    // Excluded letterbox rows (26 rows * 320 = 8320) shrink shared support strictly below G0 baseline
    expect(leftSet.support.excludedByG1ActiveOrEdgeCount).toBe(26 * 320);
    expect(leftSet.support.sharedNonoccCount).toBeLessThan(leftSet.support.priorG0BaselineNonoccCount);
    expect(leftSet.support.sharedNonoccCount).toBeGreaterThan(40000);

    // Every sample point must map to a unique native source pixel, stay in correspondence bounds, and satisfy my1 inside [13..226]
    const seen = new Set<number>();
    let outOfBoundsCorr = 0;
    let padRowHit = 0;
    let dupSrc = 0;
    for (const pt of leftSet.sharedNonoccPoints) {
      const srcKey = pt.iy * mcycle.calib.width + pt.ix;
      if (seen.has(srcKey)) dupSrc++;
      seen.add(srcKey);
      if (pt.myG1 < 13 || pt.myG1 > 226) padRowHit++;
      const xMatch = pt.ix - pt.gtNativePx;
      if (xMatch < 0 || xMatch > mcycle.calib.width - 1) outOfBoundsCorr++;
    }
    expect(dupSrc).toBe(0);
    expect(padRowHit).toBe(0);
    expect(outOfBoundsCorr).toBe(0);

    // Evaluate a synthetic prediction with fixed +1.5px bias & 1 negative value under G1
    const predG1 = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT).fill(0);
    for (const pt of leftSet.sharedAllValidPoints) {
      predG1[pt.idxG1] = pt.gtModelPx + 1.5;
    }
    predG1[leftSet.sharedNonoccPoints[0].idxG1] = -0.5;
    const feed = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT * 6);
    const evalRes = evaluateFactorialArmEye({
      pred: predG1,
      geometry: 'G1_isotropic_same_scaleX',
      sampleSet: leftSet,
      oppositeSampleSet: rightSet,
      feedFloat32: feed,
      calib: mcycle.calib,
      rectificationCaveat: mcycle.spec.rectificationCaveat,
    });

    expect(evalRes.nonocc.evaluatedCount).toBe(leftSet.support.sharedNonoccCount);
    expect(evalRes.nonocc.invalidPredCount).toBe(1);
    expect(evalRes.nonocc.negativePredCount).toBe(1);
    expect(evalRes.nonocc.signedBiasModelPx).toBeCloseTo(1.5, 2);
    expect(evalRes.nonocc.olsSlope).toBeCloseTo(1.0, 2);
    expect(evalRes.nonocc.bad1Rate).toBe(1.0);
    expect(evalRes.nonocc.bad3Rate).toBeLessThan(0.01);

    const sumBands = evalRes.disparityBands.reduce((s, b) => s + b.count, 0);
    const sumEdges = evalRes.edgeCategories.reduce((s, b) => s + b.count, 0);
    const sumTex = evalRes.textureCategories.reduce((s, b) => s + b.count, 0);
    expect(sumBands).toBe(evalRes.nonocc.evaluatedCount);
    expect(sumEdges).toBe(evalRes.nonocc.evaluatedCount);
    expect(sumTex).toBe(evalRes.nonocc.evaluatedCount);
    expect(evalRes.worstClustersTop5).toHaveLength(5);
    expect(evalRes.worstClustersTop5[0].allGtWithinNdisp).toBe(true);
    expect(evalRes.worstClustersTop5[0].yResidualStatus).toMatch(/disp\*GTy\.pfm absent/);
  });

  it('orchestrates 2x2 WASM + bounded WebGPU factorial diagnosis and handles both NO_PREPROCESSING_FIX_FOUND and IMPROVEMENT_OBSERVED verdicts', async () => {
    const fetchDisk = createDiskFixtureFetch();
    const m1Stub = buildMinimalCanonicalM1Stub();
    const m1Sha = HITNET_M1_FULL_GRAPH_SHA256;
    vi.spyOn(crypto.webcrypto.subtle, 'digest').mockImplementation(async (_alg, data) => {
      const u8 =
        data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(
              (data as ArrayBufferView).buffer,
              (data as ArrayBufferView).byteOffset,
              (data as ArrayBufferView).byteLength
            );
      if (u8.byteLength === m1Stub.byteLength) {
        return hexToArrayBuffer(HITNET_M1_FULL_GRAPH_SHA256);
      }
      return hexToArrayBuffer(nodeSha256Hex(u8));
    });

    const buildSyntheticRgba: Parameters<typeof runHitnetMiddleburyFactorialDiagnosis>[0]['buildGeometryRgba'] =
      async ({ sceneAssets, g0Transform }) => {
        const makePair = (salt: number) => {
          const left = new Uint8ClampedArray(HITNET_MIDDLEBURY_PIXEL_COUNT * 4);
          const right = new Uint8ClampedArray(HITNET_MIDDLEBURY_PIXEL_COUNT * 4);
          left[0] = (sceneAssets.im0Bytes[50] + salt) & 0xff;
          left[1] = 100;
          left[2] = 200;
          left[3] = 255;
          right[0] = (sceneAssets.im1Bytes[50] + salt) & 0xff;
          right[1] = 90;
          right[2] = 180;
          right[3] = 255;
          return { leftRgba: left, rightRgba: right };
        };
        return {
          G0: makePair(g0Transform.srcWidth & 0x1f),
          G1: makePair(99),
        };
      };

    // Case 1: Constant identical outputs across all 4 arms -> NO_PREPROCESSING_FIX_FOUND
    const callsConstant: Array<{ provider: string; prof: boolean; feedSha: string }> = [];
    const runnerNoFix: ResidualDiagnosticWorkerRunner = async (req) => {
      const feedSha = nodeSha256Hex(new Uint8Array(req.feedBuffer));
      callsConstant.push({ provider: req.provider, prof: Boolean(req.enableWebGpuProfiling), feedSha });
      const pOut = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT).fill(10.0);
      const sOut = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT).fill(10.0);
      if (req.provider === 'webgpu') {
        pOut[1234] = 10.0005;
      }
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
        ],
        foreignOrtResources: [],
        loadedAssetProvenance: [],
        sessionOptionsApplied: {
          executionProviders: [req.provider],
          graphOptimizationLevel: req.graphOptimizationLevel,
          enableCpuMemArena: true,
          enableMemPattern: true,
          numThreads: 1,
          proxy: false,
          wasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
          disableCpuEpFallback: Boolean(req.disableCpuEpFallback),
          fetchMode: 'all-session-outputs',
          profilingRequested: Boolean(req.enableWebGpuProfiling),
        },
        feedTensorName: 'input',
        feedTensorShape: [1, 6, HITNET_MIDDLEBURY_MODEL_HEIGHT, HITNET_MIDDLEBURY_MODEL_WIDTH],
        feedTensorDataType: 'float32',
        sessionInputNames: ['input'],
        sessionOutputNames: ['reference_output_disparity', 'secondary_output_disparity'],
        workerGpuAdapter: req.provider === 'webgpu' ? 'Mock GPU' : null,
        gpuValidationErrors: [],
        gpuKernelProfileSummary: req.enableWebGpuProfiling
          ? {
              profilingRequested: true,
              ortProfilingKernelTypes: { Conv: 10 },
              ortProfilingSamplePrograms: ['Conv'],
              webgpuComputePassCount: 8,
              webgpuDispatchWorkgroupsCount: 16,
              webgpuDispatchIndirectCount: 0,
              webgpuTimestampWritePassCount: 0,
              webgpuQueueSubmitCount: 1,
              webgpuCommandBufferCount: 1,
              webgpuQuerySetCount: 0,
              webgpuSamplePassLabels: ['Conv'],
              deviceObserverAttached: true,
              ortProfilingEventCount: 10,
              timestampQuerySupported: true,
              note: 'mock dispatch',
            }
          : null,
        modelByteLength: req.modelBuffer.byteLength,
        modelSha256: m1Sha,
        feedSha256: feedSha,
        createMs: 20,
        coldRunMs: 12,
        warmRunMs: [6],
        outputs: [
          {
            name: 'reference_output_disparity',
            dims: [1, 240, 320, 1],
            dataType: 'float32',
            buffer: pOut.buffer,
          },
          {
            name: 'secondary_output_disparity',
            dims: [1, 240, 320, 1],
            dataType: 'float32',
            buffer: sOut.buffer,
          },
        ],
      };
    };

    const reportNoFix = await runHitnetMiddleburyFactorialDiagnosis({
      m1Bytes: m1Stub,
      isDev: true,
      runWorker: runnerNoFix,
      loadSceneAssets: (id: MiddleburyEval3SceneId) => loadAndVerifyMiddleburySceneAssets(id, fetchDisk),
      buildGeometryRgba: buildSyntheticRgba,
    });

    expect(reportNoFix.ok).toBe(true);
    expect(reportNoFix.m1Sha256).toBe(m1Sha);
    expect(reportNoFix.n1OffProof.verified).toBe(true);
    // Per scene: 4 WASM arms + 2 GPU unprofiled + 2 GPU profiled = 8 calls -> 16 total
    expect(callsConstant).toHaveLength(16);
    expect(Object.keys(reportNoFix.scenes.Motorcycle.wasmArms)).toEqual(
      FACTORIAL_ARM_SPECS.map((s) => s.armId)
    );
    expect(reportNoFix.scenes.Motorcycle.gpuConfirmation.baselineArm.dispatchObserved).toBe(true);
    expect(reportNoFix.scenes.Motorcycle.gpuConfirmation.candidateArm?.armId).toBe(
      'G1_C0_isotropic_RGB'
    );
    expect(reportNoFix.verdict.anyMeaningfulImprovement).toBe(false);
    expect(reportNoFix.verdict.summaryVerdict).toMatch(/^NO_PREPROCESSING_FIX_FOUND:/);
    expect(reportNoFix.verdict.smallestSubsequentFollowUpIfNoImprovement).toHaveLength(2);

    // Case 2: BGR arm cuts EPE sharply on Teddy (when feed[0] > 0.7 -> BGR pixel ch0 == 200/255)
    const runnerBgrHelps: ResidualDiagnosticWorkerRunner = async (req) => {
      const f32 = new Float32Array(req.feedBuffer);
      const isBgr = f32[0] > 0.75; // 200/255 = 0.784 vs <= 82/255 = 0.32
      const val = isBgr ? 14.0 : 0.0;
      const baseRes = await runnerNoFix(req);
      if (!baseRes.ok) return baseRes;
      return {
        ...baseRes,
        outputs: baseRes.outputs.map((o) => ({
          ...o,
          buffer: new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT).fill(val).buffer,
        })),
      };
    };

    const reportImproved = await runHitnetMiddleburyFactorialDiagnosis({
      m1Bytes: m1Stub,
      sceneIds: ['Teddy'],
      isDev: true,
      runWorker: runnerBgrHelps,
      loadSceneAssets: (id: MiddleburyEval3SceneId) => loadAndVerifyMiddleburySceneAssets(id, fetchDisk),
      buildGeometryRgba: buildSyntheticRgba,
    });
    expect(reportImproved.verdict.anyMeaningfulImprovement).toBe(true);
    expect(reportImproved.verdict.summaryVerdict).toMatch(/^IMPROVEMENT_OBSERVED:/);
    expect(reportImproved.scenes.Teddy.gpuConfirmation.candidateArm?.channelOrder).toBe('C1_BGR');
  }, 30000);
});
