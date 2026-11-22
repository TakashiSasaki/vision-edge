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
  packRgbaStereoEyesToHitnetFeed,
} from './hitnetMiddleburyEval3Benchmark';
import {
  decodePngRgbTopToBottomBitExact,
  type DecodedRgbImageGrid,
} from './hitnetMiddleburyRawAndGtDiagnostic';
import {
  buildNoPadLadderSharedPointsForEye,
  buildPredeclaredNoPadScaleLadderPlacements,
  buildShiftedCropContextControlSupportForEye,
  PREDECLARED_SHIFTED_CONTEXT_CONTROL_CROP_A,
  runHitnetMiddleburyNoPadScaleLadderDiagnostic,
  type NoPadCropFeedBuilder,
} from './hitnetMiddleburyNoPadScaleLadderDiagnostic';
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
    encodeStrField(2, 'middlebury_no_pad_ladder_stub'),
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

function bilinearSampleFullSourceTo320x240Rgba(
  img: DecodedRgbImageGrid,
  srcRect: { x: number; y: number; width: number; height: number }
): Uint8ClampedArray {
  const mw = HITNET_MIDDLEBURY_MODEL_WIDTH;
  const mh = HITNET_MIDDLEBURY_MODEL_HEIGHT;
  const out = new Uint8ClampedArray(mw * mh * 4);
  for (let my = 0; my < mh; my++) {
    const sy = Math.max(
      0,
      Math.min(img.height - 1, srcRect.y + ((my + 0.5) * srcRect.height) / mh - 0.5)
    );
    const iy0 = Math.floor(sy);
    const iy1 = Math.min(img.height - 1, iy0 + 1);
    const wy = sy - iy0;

    for (let mx = 0; mx < mw; mx++) {
      const sx = Math.max(
        0,
        Math.min(img.width - 1, srcRect.x + ((mx + 0.5) * srcRect.width) / mw - 0.5)
      );
      const ix0 = Math.floor(sx);
      const ix1 = Math.min(img.width - 1, ix0 + 1);
      const wx = sx - ix0;

      const off = (my * mw + mx) * 4;
      for (let c = 0; c < 3; c++) {
        const p00 = img.rgbTopToBottom[(iy0 * img.width + ix0) * 3 + c];
        const p10 = img.rgbTopToBottom[(iy0 * img.width + ix1) * 3 + c];
        const p01 = img.rgbTopToBottom[(iy1 * img.width + ix0) * 3 + c];
        const p11 = img.rgbTopToBottom[(iy1 * img.width + ix1) * 3 + c];
        out[off + c] = Math.round(
          (p00 * (1 - wx) + p10 * wx) * (1 - wy) + (p01 * (1 - wx) + p11 * wx) * wy
        );
      }
      out[off + 3] = 255;
    }
  }
  return out;
}

const nodeBuildNoPadCropFeed: NoPadCropFeedBuilder = async ({ im0Decoded, im1Decoded, srcRect }) => {
  const lRgba = bilinearSampleFullSourceTo320x240Rgba(im0Decoded, srcRect);
  const rRgba = bilinearSampleFullSourceTo320x240Rgba(im1Decoded, srcRect);
  const feedFloat32 = packRgbaStereoEyesToHitnetFeed(lRgba, rRgba);
  return {
    feedFloat32,
    feedSha256: nodeSha256Hex(new Uint8Array(feedFloat32.buffer)),
  };
};

describe('hitnetMiddleburyNoPadScaleLadderDiagnostic', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('predeclares nested in-bounds fractional windows (320x240, 426.667x320, 640x480) and builds shared physical support on Motorcycle Q', async () => {
    const placements = buildPredeclaredNoPadScaleLadderPlacements(741, 497);

    // Check exact dimensions & isotropy
    for (const loc of ['upper_left', 'center'] as const) {
      expect(placements[loc].scale_1_00.srcRect.width).toBe(320);
      expect(placements[loc].scale_1_00.srcRect.height).toBe(240);
      expect(placements[loc].scale_0_75.srcRect.width).toBeCloseTo(1280 / 3, 9);
      expect(placements[loc].scale_0_75.srcRect.height).toBe(320);
      expect(placements[loc].scale_0_75.scaleX).toBeCloseTo(0.75, 12);
      expect(placements[loc].scale_0_75.scaleY).toBeCloseTo(0.75, 12);
      expect(placements[loc].scale_0_50.srcRect.width).toBe(640);
      expect(placements[loc].scale_0_50.srcRect.height).toBe(480);
      expect(placements[loc].scale_0_50.scaleX).toBe(0.5);
      expect(placements[loc].scale_0_50.scaleY).toBe(0.5);

      // Strictly in bounds and nested: 1.00x ⊂ 0.75x ⊂ 0.50x
      const r1 = placements[loc].scale_1_00.srcRect;
      const r75 = placements[loc].scale_0_75.srcRect;
      const r50 = placements[loc].scale_0_50.srcRect;
      expect(r50.x).toBeGreaterThanOrEqual(0);
      expect(r50.y).toBeGreaterThanOrEqual(0);
      expect(r50.x + r50.width).toBeLessThanOrEqual(741);
      expect(r50.y + r50.height).toBeLessThanOrEqual(497);

      expect(r75.x).toBeGreaterThanOrEqual(r50.x);
      expect(r75.y).toBeGreaterThanOrEqual(r50.y);
      expect(r75.x + r75.width).toBeLessThanOrEqual(r50.x + r50.width + 1e-9);
      expect(r75.y + r75.height).toBeLessThanOrEqual(r50.y + r50.height + 1e-9);

      expect(r1.x).toBeGreaterThanOrEqual(r75.x);
      expect(r1.y).toBeGreaterThanOrEqual(r75.y);
      expect(r1.x + r1.width).toBeLessThanOrEqual(r75.x + r75.width + 1e-9);
      expect(r1.y + r1.height).toBeLessThanOrEqual(r75.y + r75.height + 1e-9);
    }

    const assets = await loadAndVerifyMiddleburySceneAssets('Motorcycle', createDiskFixtureFetch());
    const ulLeft = await buildNoPadLadderSharedPointsForEye({
      eye: 'left',
      locationId: 'upper_left',
      placements: placements.upper_left,
      gtPfm: assets.disp0GtPfm,
      maskNocc: assets.mask0Nocc,
      calib: assets.calib,
    });
    const ctRight = await buildNoPadLadderSharedPointsForEye({
      eye: 'right',
      locationId: 'center',
      placements: placements.center,
      gtPfm: assets.disp1GtPfm,
      maskNocc: assets.mask1Nocc,
      calib: assets.calib,
    });

    expect(ulLeft.counts.sharedNonoccCount).toBeGreaterThan(25000);
    expect(ctRight.counts.sharedNonoccCount).toBeGreaterThan(25000);

    // Shifted-crop 1.0x fixed intersection
    const shiftLeft = await buildShiftedCropContextControlSupportForEye({
      eye: 'left',
      cropA: PREDECLARED_SHIFTED_CONTEXT_CONTROL_CROP_A.srcRect,
      cropB: { x: 210, y: 128, width: 320, height: 240 },
      gtPfm: assets.disp0GtPfm,
      maskNocc: assets.mask0Nocc,
      calib: assets.calib,
    });
    expect(shiftLeft.intersectionBoxNativeQ.xMin).toBe(212);
    expect(shiftLeft.intersectionBoxNativeQ.xMax).toBe(437);
    expect(shiftLeft.intersectionBoxNativeQ.yMin).toBe(130);
    expect(shiftLeft.intersectionBoxNativeQ.yMax).toBe(317);
    expect(shiftLeft.sharedNonoccPoints.length).toBeGreaterThan(10000);
  });

  it('orchestrates 7 distinct WASM runs + bounded paired WebGPU dispatch check without duplicating center 1.0x', async () => {
    const m1Bytes = buildMinimalCanonicalM1Stub();
    const realDigest = crypto.webcrypto.subtle.digest.bind(crypto.webcrypto.subtle);
    vi.spyOn(crypto.webcrypto.subtle, 'digest').mockImplementation(async (alg, data) => {
      const view =
        data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      if (view.byteLength === m1Bytes.byteLength && view[0] === m1Bytes[0] && view[1] === m1Bytes[1]) {
        return hexToArrayBuffer(HITNET_M1_FULL_GRAPH_SHA256);
      }
      return realDigest(alg, data);
    });

    const assets = await loadAndVerifyMiddleburySceneAssets('Motorcycle', createDiskFixtureFetch());
    const im0 = await decodePngRgbTopToBottomBitExact(assets.im0Bytes);
    const im1 = await decodePngRgbTopToBottomBitExact(assets.im1Bytes);
    const placements = buildPredeclaredNoPadScaleLadderPlacements(741, 497);
    const ul050Feed = await nodeBuildNoPadCropFeed({
      im0Decoded: im0,
      im1Decoded: im1,
      srcRect: placements.upper_left.scale_0_50.srcRect,
    });

    const workerCalls: Array<{ provider: string; feedSha256: string; profiling: boolean }> = [];
    const seenWasmFeeds = new Set<string>();

    const mockWorker: ResidualDiagnosticWorkerRunner = async (req) => {
      const feedSha256 = nodeSha256Hex(new Uint8Array(req.feedBuffer));
      const profilingRequested = Boolean(req.enableWebGpuProfiling);
      workerCalls.push({ provider: req.provider, feedSha256, profiling: profilingRequested });

      if (req.provider === 'wasm') {
        expect(seenWasmFeeds.has(feedSha256)).toBe(false);
        seenWasmFeeds.add(feedSha256);
      }

      // Predict 18Q (9.0 model px * 2) for upper_left 0.50x vs ~1-2Q baseline (1.0 model px) so upper_left 0.50x has largest EPE improvement
      const isUl050 = feedSha256 === ul050Feed.feedSha256;
      const predModelVal = isUl050 ? 9.0 : 1.0;
      const pOut = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT).fill(predModelVal);
      const sOut = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT).fill(predModelVal - 0.25);
      const isGpu = req.provider === 'webgpu';

      return {
        id: 1,
        ok: true,
        pinnedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        entryModuleUrl:
          'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs',
        wasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
        ortVersions: { common: '1.30.0', web: '1.30.0' },
        loadedOrtResources: [
          'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs',
          'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.asyncify.wasm',
        ],
        foreignOrtResources: [],
        loadedAssetProvenance: [],
        provider: req.provider,
        configuredProviders: [req.provider],
        fallbackBehavior: 'no-fallback-explicit-error',
        graphOptimizationLevel: 'basic',
        sessionOptionsApplied: {
          executionProviders: [req.provider],
          graphOptimizationLevel: 'basic',
          enableCpuMemArena: false,
          enableMemPattern: false,
          numThreads: 1,
          proxy: false,
          wasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
          disableCpuEpFallback: Boolean(req.disableCpuEpFallback),
          fetchMode: 'all-session-outputs',
          profilingRequested,
        },
        modelByteLength: m1Bytes.byteLength,
        modelSha256: HITNET_M1_FULL_GRAPH_SHA256,
        feedSha256,
        feedTensorName: 'input',
        feedTensorShape: [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        sessionInputNames: ['input'],
        sessionOutputNames: ['reference_output_disparity', 'secondary_output_disparity'],
        workerGpuAdapter: isGpu ? 'mock-gpu' : null,
        gpuValidationErrors: [],
        gpuKernelProfileSummary: profilingRequested
          ? {
              profilingRequested: true,
              timestampQuerySupported: true,
              ortProfilingEventCount: 112,
              ortProfilingKernelTypes: { Conv: 10 },
              ortProfilingSamplePrograms: ['Conv'],
              webgpuComputePassCount: 44,
              webgpuDispatchWorkgroupsCount: 88,
              webgpuDispatchIndirectCount: 0,
              webgpuTimestampWritePassCount: 44,
              webgpuQueueSubmitCount: 2,
              webgpuCommandBufferCount: 2,
              webgpuQuerySetCount: 2,
              webgpuSamplePassLabels: ['Conv'],
              deviceObserverAttached: true,
              note: 'mock dispatch',
            }
          : null,
        createMs: 20,
        coldRunMs: isGpu ? 15 : 58,
        warmRunMs: [isGpu ? 7 : 42],
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

    const report = await runHitnetMiddleburyNoPadScaleLadderDiagnostic({
      m1Bytes,
      isDev: true,
      includeWebGpuCheck: true,
      disableCpuEpFallback: true,
      runWorker: mockWorker,
      loadMotorcycleAssets: () => loadAndVerifyMiddleburySceneAssets('Motorcycle', createDiskFixtureFetch()),
      buildCropFeed: nodeBuildNoPadCropFeed,
    });

    expect(report.ok).toBe(true);
    expect(report.m1Sha256).toBe(HITNET_M1_FULL_GRAPH_SHA256);
    expect(report.n1OffProof.n1ExplicitlyOff).toBe(true);
    expect(report.n1OffProof.n1ReplacementCount).toBe(0);

    // 3 upper_left + 3 center + 1 shifted_left_top = 7 WASM runs (center 1.0x reused for shifted control!)
    // + 2 GPU runs + 2 GPU dispatch profiling checks = 4 WebGPU calls
    expect(workerCalls.filter((c) => c.provider === 'wasm')).toHaveLength(7);
    expect(seenWasmFeeds.size).toBe(7);
    expect(workerCalls.filter((c) => c.provider === 'webgpu')).toHaveLength(4);

    // Explicit context & distinction reporting
    expect(report.distinctionFromPriorDiagnostics.priorNativeAndDownscalePlusPadSummary).toContain(
      '138x102'
    );
    expect(report.distinctionFromPriorDiagnostics.epistemicRule).toContain('NO pure-scale');
    expect(report.officialMatchedDyNote).toContain('dyavg=0.23');

    // Bounded WebGPU dispatch check picked upper_left scale_0_50 vs scale_1_00 with real dispatch
    expect(report.boundedGpuDispatchCheck.selectedLocationId).toBe('upper_left');
    expect(report.boundedGpuDispatchCheck.selectedComparisonScaleId).toBe('scale_0_50');
    expect(report.boundedGpuDispatchCheck.realDispatchObservedBothArms).toBe(true);

      // Verify shifted 1.0x control reused center scale_1_00 runId cleanly
      expect(report.sameScaleContextOnlyControl.cropBWasmReused.run.runId).toBe(
        'center_scale_1_00_wasm'
      );
    },
    20000
  );
});
