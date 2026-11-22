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
  assertActiveSupportByteIdentityBeforeInference,
  computeStrictActiveInteriorBox,
  extractEyeActiveFloat32Bytes,
  fillPaddingOutsideActiveBoxRgba,
  runHitnetMiddleburyPaddingAblationDiagnostic,
  type PaddingAblationFeedBundleBuilder,
} from './hitnetMiddleburyPaddingAblationDiagnostic';
import {
  buildWindowScalePlacements,
  PREDECLARED_MOTORCYCLE_320X240_WINDOWS,
  type HitnetMotorcycleScaleContextReport,
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
    encodeStrField(2, 'middlebury_padding_ablation_stub'),
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

function bilinearDownscaleWindowBlackRgba(
  img: DecodedRgbImageGrid,
  x0: number,
  y0: number,
  w: number,
  h: number,
  dstRect: { x: number; y: number; width: number; height: number }
): Uint8ClampedArray {
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
        out[off + c] = Math.round(
          (p00 * (1 - wx) + p10 * wx) * (1 - wy) + (p01 * (1 - wx) + p11 * wx) * wy
        );
      }
      out[off + 3] = 255;
    }
  }
  return out;
}

const nodeBuildPaddingFeeds: PaddingAblationFeedBundleBuilder = async ({
  im0Decoded,
  im1Decoded,
  windowSpec,
  s1Geometry,
  activeInteriorBox,
}) => {
  const blackLeftRgba = bilinearDownscaleWindowBlackRgba(
    im0Decoded,
    windowSpec.x,
    windowSpec.y,
    windowSpec.width,
    windowSpec.height,
    s1Geometry.dstRect
  );
  const blackRightRgba = bilinearDownscaleWindowBlackRgba(
    im1Decoded,
    windowSpec.x,
    windowSpec.y,
    windowSpec.width,
    windowSpec.height,
    s1Geometry.dstRect
  );
  const repLeftRgba = fillPaddingOutsideActiveBoxRgba(
    blackLeftRgba,
    activeInteriorBox,
    'pad_replicate'
  );
  const repRightRgba = fillPaddingOutsideActiveBoxRgba(
    blackRightRgba,
    activeInteriorBox,
    'pad_replicate'
  );
  const refLeftRgba = fillPaddingOutsideActiveBoxRgba(
    blackLeftRgba,
    activeInteriorBox,
    'pad_reflect'
  );
  const refRightRgba = fillPaddingOutsideActiveBoxRgba(
    blackRightRgba,
    activeInteriorBox,
    'pad_reflect'
  );

  const blackFeed = packRgbaStereoEyesToHitnetFeed(blackLeftRgba, blackRightRgba);
  const repFeed = packRgbaStereoEyesToHitnetFeed(repLeftRgba, repRightRgba);
  const refFeed = packRgbaStereoEyesToHitnetFeed(refLeftRgba, refRightRgba);

  const activeL = extractEyeActiveFloat32Bytes(blackFeed, 'left', activeInteriorBox);
  const activeR = extractEyeActiveFloat32Bytes(blackFeed, 'right', activeInteriorBox);
  const activeBoth = new Uint8Array(activeL.byteLength + activeR.byteLength);
  activeBoth.set(activeL, 0);
  activeBoth.set(activeR, activeL.byteLength);

  return {
    windowId: windowSpec.windowId,
    activeInteriorBox,
    leftActiveFloat32Sha256: nodeSha256Hex(activeL),
    rightActiveFloat32Sha256: nodeSha256Hex(activeR),
    bothEyesActiveFloat32Sha256: nodeSha256Hex(activeBoth),
    feeds: {
      pad_black: {
        paddingMode: 'pad_black',
        feedFloat32: blackFeed,
        feedSha256: nodeSha256Hex(new Uint8Array(blackFeed.buffer)),
      },
      pad_replicate: {
        paddingMode: 'pad_replicate',
        feedFloat32: repFeed,
        feedSha256: nodeSha256Hex(new Uint8Array(repFeed.buffer)),
      },
      pad_reflect: {
        paddingMode: 'pad_reflect',
        feedFloat32: refFeed,
        feedSha256: nodeSha256Hex(new Uint8Array(refFeed.buffer)),
      },
    },
  };
};

describe('hitnetMiddleburyPaddingAblationDiagnostic', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('keeps active interior [91..228]x[69..170] 100% byte-identical across black/replicate/reflect, verifies edge fill, and fails array guard if perturbed', async () => {
    const assets = await loadAndVerifyMiddleburySceneAssets('Motorcycle', createDiskFixtureFetch());
    const im0 = await decodePngRgbTopToBottomBitExact(assets.im0Bytes);
    const im1 = await decodePngRgbTopToBottomBitExact(assets.im1Bytes);

    const tlSpec = PREDECLARED_MOTORCYCLE_320X240_WINDOWS.top_left;
    const { s1 } = buildWindowScalePlacements(tlSpec);
    const box = computeStrictActiveInteriorBox(s1);
    expect(box.colStart).toBe(91);
    expect(box.colEnd).toBe(228);
    expect(box.rowStart).toBe(69);
    expect(box.rowEnd).toBe(170);
    expect(box.pixelCountPerEye).toBe(138 * 102);

    const bundle = await nodeBuildPaddingFeeds({
      im0Decoded: im0,
      im1Decoded: im1,
      windowSpec: tlSpec,
      s1Geometry: s1,
      activeInteriorBox: box,
    });

    // All 3 full feed hashes must differ cleanly while active-interior SHAs pass
    expect(bundle.feeds.pad_black.feedSha256).not.toBe(bundle.feeds.pad_replicate.feedSha256);
    expect(bundle.feeds.pad_black.feedSha256).not.toBe(bundle.feeds.pad_reflect.feedSha256);
    expect(bundle.feeds.pad_replicate.feedSha256).not.toBe(bundle.feeds.pad_reflect.feedSha256);

    const verified = await assertActiveSupportByteIdentityBeforeInference({
      windowId: 'top_left',
      bundle,
    });
    expect(verified.bothEyesActiveFloat32Sha256).toBe(bundle.bothEyesActiveFloat32Sha256);

    // Check replicate & reflect border semantics on ch0 at rowStart+5:
    // colStart - 1 must match colStart (replicate) and colStart + 1 (reflect101)
    const testRow = box.rowStart + 5;
    const repCh0 = bundle.feeds.pad_replicate.feedFloat32;
    const refCh0 = bundle.feeds.pad_reflect.feedFloat32;
    expect(repCh0[testRow * 320 + (box.colStart - 1)]).toBe(
      repCh0[testRow * 320 + box.colStart]
    );
    expect(refCh0[testRow * 320 + (box.colStart - 1)]).toBe(
      refCh0[testRow * 320 + (box.colStart + 1)]
    );

    // Array guard MUST fail loudly before inference if an active interior pixel changes
    bundle.feeds.pad_replicate.feedFloat32[box.rowStart * 320 + box.colStart] += 0.01;
    await expect(
      assertActiveSupportByteIdentityBeforeInference({ windowId: 'top_left', bundle })
    ).rejects.toThrow(/Array guard failed/);
  });

  it('orchestrates 6-WASM padding arms + bounded promising-arm paired WebGPU check without repeating prior diagnostic', async () => {
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

    // Prebuild exact expected pad_black baseline feeds to verify prior-report feed SHA match
    const assets = await loadAndVerifyMiddleburySceneAssets('Motorcycle', createDiskFixtureFetch());
    const im0 = await decodePngRgbTopToBottomBitExact(assets.im0Bytes);
    const im1 = await decodePngRgbTopToBottomBitExact(assets.im1Bytes);
    const tlBundle = await nodeBuildPaddingFeeds({
      im0Decoded: im0,
      im1Decoded: im1,
      windowSpec: PREDECLARED_MOTORCYCLE_320X240_WINDOWS.top_left,
      s1Geometry: buildWindowScalePlacements(PREDECLARED_MOTORCYCLE_320X240_WINDOWS.top_left).s1,
      activeInteriorBox: computeStrictActiveInteriorBox(
        buildWindowScalePlacements(PREDECLARED_MOTORCYCLE_320X240_WINDOWS.top_left).s1
      ),
    });
    const ctBundle = await nodeBuildPaddingFeeds({
      im0Decoded: im0,
      im1Decoded: im1,
      windowSpec: PREDECLARED_MOTORCYCLE_320X240_WINDOWS.center,
      s1Geometry: buildWindowScalePlacements(PREDECLARED_MOTORCYCLE_320X240_WINDOWS.center).s1,
      activeInteriorBox: computeStrictActiveInteriorBox(
        buildWindowScalePlacements(PREDECLARED_MOTORCYCLE_320X240_WINDOWS.center).s1
      ),
    });

    const priorMockReport = {
      ok: true,
      sceneId: 'Motorcycle',
      m1Sha256: HITNET_M1_FULL_GRAPH_SHA256,
      windows: {
        top_left: {
          native1xWasm: {
            left: { nonocc: { epeNativeQPx: 1.4 } },
            right: { nonocc: { epeNativeQPx: 1.5 } },
          },
          downscale320_741Wasm: {
            run: { feedSha256: tlBundle.feeds.pad_black.feedSha256 },
          },
          wholeFrameContextRefOnWindowPoints: {
            left: { nonocc: { epeNativeQPx: 2.2 } },
            right: { nonocc: { epeNativeQPx: 2.4 } },
          },
        },
        center: {
          native1xWasm: {
            left: { nonocc: { epeNativeQPx: 1.8 } },
            right: { nonocc: { epeNativeQPx: 1.9 } },
          },
          downscale320_741Wasm: {
            run: { feedSha256: ctBundle.feeds.pad_black.feedSha256 },
          },
          wholeFrameContextRefOnWindowPoints: {
            left: { nonocc: { epeNativeQPx: 3.1 } },
            right: { nonocc: { epeNativeQPx: 3.2 } },
          },
        },
      },
    } as unknown as HitnetMotorcycleScaleContextReport;

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

      // Make top_left + pad_reflect genuinely lower EPE (6.0 model px ~ 13.9Q vs 12.0 model px ~ 27.8Q) so reflect is chosen
      const isTlReflect = feedSha256 === tlBundle.feeds.pad_reflect.feedSha256;
      const predVal = isTlReflect ? 6.0 : 12.0;
      const pOut = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT).fill(predVal);
      const sOut = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT).fill(predVal - 0.5);
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
              ortProfilingEventCount: 110,
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
        createMs: 22,
        coldRunMs: isGpu ? 16 : 60,
        warmRunMs: [isGpu ? 8 : 44],
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
    const report = await runHitnetMiddleburyPaddingAblationDiagnostic({
      m1Bytes,
      isDev: true,
      includeWebGpuCheck: true,
      disableCpuEpFallback: true,
      priorScaleContextReport: priorMockReport,
      runWorker: mockWorker,
      loadMotorcycleAssets: () => loadAndVerifyMiddleburySceneAssets('Motorcycle', createDiskFixtureFetch()),
      buildFeeds: nodeBuildPaddingFeeds,
      onLog: (line) => logs.push(line),
    });

    expect(report.ok).toBe(true);
    expect(report.m1Sha256).toBe(HITNET_M1_FULL_GRAPH_SHA256);
    expect(report.n1OffProof.n1ExplicitlyOff).toBe(true);
    expect(report.n1OffProof.n1ReplacementCount).toBe(0);

    // Exactly 6 WASM runs (top_left black/replicate/reflect + center black/replicate/reflect)
    // + 2 paired WebGPU output/timing runs + 2 WebGPU dispatch profiling checks = 10 worker calls
    expect(workerCalls.filter((c) => c.provider === 'wasm')).toHaveLength(6);
    expect(seenWasmFeeds.size).toBe(6);
    expect(workerCalls.filter((c) => c.provider === 'webgpu')).toHaveLength(4);

    // Prior black feed hashes + non-causal references verified without repeating S0/whole-frame
    expect(report.windows.top_left.activeByteIdentityProof.priorBlackBaselineFeedShaMatched).toBe(true);
    expect(report.windows.center.activeByteIdentityProof.priorBlackBaselineFeedShaMatched).toBe(true);
    expect(report.windows.top_left.priorReferencesNonCausal.reusedFromPriorReport).toBe(true);
    expect(report.windows.top_left.priorReferencesNonCausal.native1xLeftNonoccEpeQ).toBe(1.4);

    // Bounded promising-arm paired WebGPU check picked top_left + pad_reflect with real dispatch
    expect(report.boundedPromisingGpuPairedCheck.selectedWindowId).toBe('top_left');
    expect(report.boundedPromisingGpuPairedCheck.selectedCandidatePadding).toBe('pad_reflect');
    expect(report.boundedPromisingGpuPairedCheck.realDispatchObservedBothArms).toBe(true);

    // Conclusion emits strongest next controlled test when native gap remains
    expect(report.conclusion.nativeGapRemains).toBe(true);
    expect(report.conclusion.strongestNextControlledTest.length).toBeGreaterThan(40);
    expect(logs.some((l) => l.includes('Padding Ablation Conclusion'))).toBe(true);
  });
});
