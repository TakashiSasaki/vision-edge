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
import { buildMiddleburyGeometryTransforms } from './hitnetMiddleburyFactorialDiagnosis';
import {
  decodePngRgbTopToBottomBitExact,
  type DecodedRgbImageGrid,
} from './hitnetMiddleburyRawAndGtDiagnostic';
import {
  buildWindowScalePlacements,
  buildWindowSharedPhysicalPointsForEye,
  evaluateWindowArmEyeMetrics,
  MOTORCYCLE_SCALE_FACTOR_320_741,
  PREDECLARED_MOTORCYCLE_320X240_WINDOWS,
  runHitnetMiddleburyScaleContextDiagnostic,
  type WindowScaleFeedBuilder,
  type WholeFrameContextFeedBuilder,
} from './hitnetMiddleburyScaleContextDiagnostic';
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
    encodeStrField(2, 'middlebury_scale_context_stub'),
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

function bilinearSampleIsolatedWindowRgba(
  img: DecodedRgbImageGrid,
  x0: number,
  y0: number,
  w: number,
  h: number,
  dstRect: { x: number; y: number; width: number; height: number }
): Uint8ClampedArray {
  // Strictly slice ONLY [x0..x0+w) x [y0..y0+h) first — zero external pixels reachable
  const iso = new Uint8Array(w * h * 3);
  for (let row = 0; row < h; row++) {
    const s = ((y0 + row) * img.width + x0) * 3;
    iso.set(img.rgbTopToBottom.subarray(s, s + w * 3), row * w * 3);
  }

  const out = new Uint8ClampedArray(HITNET_MIDDLEBURY_PIXEL_COUNT * 4);
  for (let my = 0; my < HITNET_MIDDLEBURY_MODEL_HEIGHT; my++) {
    const sy = ((my + 0.5 - dstRect.y) * h) / dstRect.height - 0.5;
    if (sy < -0.5 || sy > h - 0.5) continue;
    const yClamped = Math.max(0, Math.min(h - 1, sy));
    const iy0 = Math.floor(yClamped);
    const iy1 = Math.min(h - 1, iy0 + 1);
    const wy = yClamped - iy0;

    for (let mx = 0; mx < HITNET_MIDDLEBURY_MODEL_WIDTH; mx++) {
      const sx = ((mx + 0.5 - dstRect.x) * w) / dstRect.width - 0.5;
      if (sx < -0.5 || sx > w - 0.5) continue;
      const xClamped = Math.max(0, Math.min(w - 1, sx));
      const ix0 = Math.floor(xClamped);
      const ix1 = Math.min(w - 1, ix0 + 1);
      const wx = xClamped - ix0;

      const off = (my * HITNET_MIDDLEBURY_MODEL_WIDTH + mx) * 4;
      for (let c = 0; c < 3; c++) {
        const p00 = iso[(iy0 * w + ix0) * 3 + c];
        const p10 = iso[(iy0 * w + ix1) * 3 + c];
        const p01 = iso[(iy1 * w + ix0) * 3 + c];
        const p11 = iso[(iy1 * w + ix1) * 3 + c];
        const v0 = p00 * (1 - wx) + p10 * wx;
        const v1 = p01 * (1 - wx) + p11 * wx;
        out[off + c] = Math.round(v0 * (1 - wy) + v1 * wy);
      }
      out[off + 3] = 255;
    }
  }
  return out;
}

const nodeBuildWindowFeeds: WindowScaleFeedBuilder = async ({
  im0Decoded,
  im1Decoded,
  windowSpec,
  s1Geometry,
}) => {
  const s0Left = bilinearSampleIsolatedWindowRgba(
    im0Decoded,
    windowSpec.x,
    windowSpec.y,
    windowSpec.width,
    windowSpec.height,
    { x: 0, y: 0, width: 320, height: 240 }
  );
  const s0Right = bilinearSampleIsolatedWindowRgba(
    im1Decoded,
    windowSpec.x,
    windowSpec.y,
    windowSpec.width,
    windowSpec.height,
    { x: 0, y: 0, width: 320, height: 240 }
  );
  const s0FeedFloat32 = packRgbaStereoEyesToHitnetFeed(s0Left, s0Right);
  const s0FeedSha256 = nodeSha256Hex(
    new Uint8Array(s0FeedFloat32.buffer, s0FeedFloat32.byteOffset, s0FeedFloat32.byteLength)
  );

  const s1Left = bilinearSampleIsolatedWindowRgba(
    im0Decoded,
    windowSpec.x,
    windowSpec.y,
    windowSpec.width,
    windowSpec.height,
    s1Geometry.dstRect
  );
  const s1Right = bilinearSampleIsolatedWindowRgba(
    im1Decoded,
    windowSpec.x,
    windowSpec.y,
    windowSpec.width,
    windowSpec.height,
    s1Geometry.dstRect
  );
  const s1FeedFloat32 = packRgbaStereoEyesToHitnetFeed(s1Left, s1Right);
  const s1FeedSha256 = nodeSha256Hex(
    new Uint8Array(s1FeedFloat32.buffer, s1FeedFloat32.byteOffset, s1FeedFloat32.byteLength)
  );

  return {
    s0FeedFloat32,
    s0FeedSha256,
    s1FeedFloat32,
    s1FeedSha256,
    s1PaddedBorderNonZeroCount: 0,
  };
};

const nodeBuildWholeFrameFeed: WholeFrameContextFeedBuilder = async ({
  im0Decoded,
  im1Decoded,
  wholeFrameGeometry,
}) => {
  const wLeft = bilinearSampleIsolatedWindowRgba(
    im0Decoded,
    0,
    0,
    im0Decoded.width,
    im0Decoded.height,
    wholeFrameGeometry.dstRect
  );
  const wRight = bilinearSampleIsolatedWindowRgba(
    im1Decoded,
    0,
    0,
    im1Decoded.width,
    im1Decoded.height,
    wholeFrameGeometry.dstRect
  );
  const feedFloat32 = packRgbaStereoEyesToHitnetFeed(wLeft, wRight);
  const feedSha256 = nodeSha256Hex(
    new Uint8Array(feedFloat32.buffer, feedFloat32.byteOffset, feedFloat32.byteLength)
  );
  return { feedFloat32, feedSha256, paddedBorderNonZeroCount: 0 };
};

describe('hitnetMiddleburyScaleContextDiagnostic', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('predeclares top_left/center 320x240 windows, builds guarded shared active points on real Motorcycle Q, and maps predictions to native Q px', async () => {
    const assets = await loadAndVerifyMiddleburySceneAssets('Motorcycle', createDiskFixtureFetch());
    const { g1: wholeGeom } = buildMiddleburyGeometryTransforms(
      'Motorcycle',
      assets.calib.width,
      assets.calib.height
    );

    const tl = PREDECLARED_MOTORCYCLE_320X240_WINDOWS.top_left;
    const ct = PREDECLARED_MOTORCYCLE_320X240_WINDOWS.center;
    expect(tl).toMatchObject({ x: 0, y: 0, width: 320, height: 240 });
    expect(ct).toMatchObject({ x: 210, y: 128, width: 320, height: 240 });

    const { s1: tlS1 } = buildWindowScalePlacements(tl);
    expect(tlS1.scaleX).toBeCloseTo(320 / 741, 9);
    expect(tlS1.scaleY).toBeCloseTo(320 / 741, 9);
    expect(tlS1.activeColStart).toBe(93);
    expect(tlS1.activeColEnd).toBe(226);
    expect(tlS1.activeRowStart).toBe(71);
    expect(tlS1.activeRowEnd).toBe(168);

    const leftSupportTL = buildWindowSharedPhysicalPointsForEye({
      eye: 'left',
      windowSpec: tl,
      gtPfm: assets.disp0GtPfm,
      maskNocc: assets.mask0Nocc,
      calib: assets.calib,
      wholeFrameGeometry: wholeGeom,
    });
    const rightSupportTL = buildWindowSharedPhysicalPointsForEye({
      eye: 'right',
      windowSpec: tl,
      gtPfm: assets.disp1GtPfm,
      maskNocc: assets.mask1Nocc,
      calib: assets.calib,
      wholeFrameGeometry: wholeGeom,
    });

    // Uniqueness, crop+active support bounds, and positive low-disparity coverage
    expect(leftSupportTL.counts.sharedNonoccCount).toBe(10855);
    expect(leftSupportTL.counts.sharedLowDisp24NonoccCount).toBe(8340);
    expect(rightSupportTL.counts.sharedNonoccCount).toBe(10786);
    expect(rightSupportTL.counts.sharedLowDisp24NonoccCount).toBe(8286);

    const seenTL = new Set<number>();
    for (const pt of leftSupportTL.sharedAllValidPoints) {
      expect(seenTL.has(pt.iy * assets.calib.width + pt.ix)).toBe(false);
      seenTL.add(pt.iy * assets.calib.width + pt.ix);
      expect(pt.ix - pt.gtNativePx).toBeGreaterThanOrEqual(leftSupportTL.s1Geometry.activeSrcXMin);
      expect(pt.ix - pt.gtNativePx).toBeLessThanOrEqual(leftSupportTL.s1Geometry.activeSrcXMax);
    }

    // Native-Q mapping precision: exact GT with +2.0 Q-px bias in downscale arm
    const s1BiasPred = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT);
    for (const pt of leftSupportTL.sharedNonoccPoints) {
      s1BiasPred[pt.idxDownscale] = (pt.gtNativePx + 2.0) * MOTORCYCLE_SCALE_FACTOR_320_741;
    }
    const biasEval = evaluateWindowArmEyeMetrics({
      pred: s1BiasPred,
      armKind: 'S1_downscale_320_741',
      scaleX: MOTORCYCLE_SCALE_FACTOR_320_741,
      support: leftSupportTL,
    });
    expect(biasEval.nonocc.epeNativeQPx).toBeCloseTo(2.0, 5);
    expect(biasEval.nonocc.signedBiasNativeQPx).toBeCloseTo(2.0, 5);
    expect(biasEval.nonocc.bad1NativeQRate).toBe(1.0);
    expect(biasEval.nonocc.bad3NativeQRate).toBe(0.0);
    expect(biasEval.lowDisp24Nonocc.epeNativeQPx).toBeCloseTo(2.0, 5);

    // Context isolation: mutating pixels outside top_left does not change S0 or S1 feeds
    const im0 = await decodePngRgbTopToBottomBitExact(assets.im0Bytes);
    const im1 = await decodePngRgbTopToBottomBitExact(assets.im1Bytes);
    const baseFeeds = await nodeBuildWindowFeeds({
      im0Decoded: im0,
      im1Decoded: im1,
      windowSpec: tl,
      s0Geometry: leftSupportTL.s0Geometry,
      s1Geometry: leftSupportTL.s1Geometry,
    });

    const mutatedIm0 = new Uint8Array(im0.rgbTopToBottom);
    const mutatedIm1 = new Uint8Array(im1.rgbTopToBottom);
    for (let y = 0; y < im0.height; y++) {
      for (let x = 0; x < im0.width; x++) {
        if (x >= tl.x && x < tl.x + tl.width && y >= tl.y && y < tl.y + tl.height) continue;
        const off = (y * im0.width + x) * 3;
        mutatedIm0[off] = 255;
        mutatedIm1[off + 1] = 255;
      }
    }
    const mutatedFeeds = await nodeBuildWindowFeeds({
      im0Decoded: { ...im0, rgbTopToBottom: mutatedIm0 },
      im1Decoded: { ...im1, rgbTopToBottom: mutatedIm1 },
      windowSpec: tl,
      s0Geometry: leftSupportTL.s0Geometry,
      s1Geometry: leftSupportTL.s1Geometry,
    });
    expect(mutatedFeeds.s0FeedSha256).toBe(baseFeeds.s0FeedSha256);
    expect(mutatedFeeds.s1FeedSha256).toBe(baseFeeds.s1FeedSha256);
  });

  it('orchestrates Motorcycle scale/context WASM arms (no duplicate runs) + bounded WebGPU verification cleanly', async () => {
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

    const seenWasmFeeds = new Set<string>();
    const workerCalls: Array<{
      provider: string;
      feedSha256: string;
      profiling: boolean;
    }> = [];

    const mockWorker: ResidualDiagnosticWorkerRunner = async (req) => {
      const feedBytes = new Uint8Array(req.feedBuffer);
      const feedSha256 = nodeSha256Hex(feedBytes);
      workerCalls.push({
        provider: req.provider,
        feedSha256,
        profiling: Boolean(req.enableWebGpuProfiling),
      });

      if (req.provider === 'wasm') {
        expect(seenWasmFeeds.has(feedSha256)).toBe(false);
        seenWasmFeeds.add(feedSha256);
      }

      const pOut = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT).fill(10.0);
      const sOut = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT).fill(9.5);
      const isGpu = req.provider === 'webgpu';
      const profilingRequested = Boolean(req.enableWebGpuProfiling);

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
        workerGpuAdapter: isGpu ? 'mock-adapter' : null,
        gpuValidationErrors: [],
        gpuKernelProfileSummary: profilingRequested
          ? {
              profilingRequested: true,
              timestampQuerySupported: true,
              ortProfilingEventCount: 120,
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
              note: 'mock real dispatch observed',
            }
          : null,
        createMs: 25,
        coldRunMs: isGpu ? 18 : 62,
        warmRunMs: [isGpu ? 9 : 48],
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

    const logs: string[] = [];
    const report = await runHitnetMiddleburyScaleContextDiagnostic({
      m1Bytes,
      isDev: true,
      includeWebGpuVerification: true,
      disableCpuEpFallback: true,
      runWorker: mockWorker,
      loadMotorcycleAssets: () => loadAndVerifyMiddleburySceneAssets('Motorcycle', createDiskFixtureFetch()),
      buildWindowFeeds: nodeBuildWindowFeeds,
      buildWholeFrameFeed: nodeBuildWholeFrameFeed,
      onLog: (line) => logs.push(line),
    });

    expect(report.ok).toBe(true);
    expect(report.m1Sha256).toBe(HITNET_M1_FULL_GRAPH_SHA256);
    expect(report.n1OffProof.verified).toBe(true);
    expect(report.n1OffProof.n1ExplicitlyOff).toBe(true);
    expect(report.n1OffProof.n1ReplacementCount).toBe(0);

    // Non-duplicative call structure:
    // 5 distinct WASM runs (whole_frame + top_left S0/S1 + center S0/S1)
    // + 2 top_left WebGPU runs + 1 top_left S0 WebGPU profiling dispatch proof = 8 calls
    expect(workerCalls).toHaveLength(8);
    expect(workerCalls.filter((c) => c.provider === 'wasm')).toHaveLength(5);
    expect(seenWasmFeeds.size).toBe(5);
    expect(workerCalls.filter((c) => c.provider === 'webgpu')).toHaveLength(3);

    expect(report.windows.top_left.native1xWasm.left.nonocc.evaluatedCount).toBe(10855);
    expect(report.windows.top_left.native1xWasm.left.lowDisp24Nonocc.evaluatedCount).toBe(8340);
    expect(report.windows.center.native1xWasm.right.nonocc.evaluatedCount).toBe(9491);
    expect(report.windows.center.native1xWasm.right.lowDisp24Nonocc.evaluatedCount).toBe(963);

    expect(report.boundedWebGpuVerification.realDispatchObserved).toBe(true);
    expect(report.boundedWebGpuVerification.executedArms).toHaveLength(2);
    expect(report.verdictSummary.epistemicBoundary).toContain('combined downsample');
    expect(logs.some((l) => l.includes('Scale/Context Verdict'))).toBe(true);
  });
});
