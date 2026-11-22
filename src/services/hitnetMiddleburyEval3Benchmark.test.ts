// @vitest-environment node
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  evaluateEyeSignSanity,
  evaluateMiddleburyDisparityMask,
  HITNET_MIDDLEBURY_PIXEL_COUNT,
  loadAndVerifyMiddleburySceneAssets,
  MIDDLEBURY_EVAL3_Q_SCENES,
  packRgbaStereoEyesToHitnetFeed,
  parseMiddleburyCalibTxt,
  parsePfmSingleBand,
  resampleMiddleburyEyeGtNearestGrid,
  runHitnetMiddleburyEval3Benchmark,
  type MiddleburyCalibData,
  type MiddleburyEval3SceneId,
} from './hitnetMiddleburyEval3Benchmark';
import { type ResidualDiagnosticWorkerRunner } from './hitnetResidualLocalization';
import { HITNET_M1_FULL_GRAPH_SHA256 } from './stereoDepthWorkerCore';
import { HITNET_DEV_NON_JSEP_FIXTURE_SPEC } from './stereoDepthWebGpuDevFixture';

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
    encodeStrField(2, 'middlebury_eval3_stub'),
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

function buildTestPfmBytes(
  width: number,
  height: number,
  topToBottomRows: number[][],
  littleEndian: boolean
): Uint8Array {
  const scaleStr = littleEndian ? '-1.0' : '1.0';
  const header = textEncoder.encode(`Pf\n${width} ${height}\n${scaleStr}\n`);
  const payload = new Uint8Array(width * height * 4);
  const dv = new DataView(payload.buffer);
  for (let yFile = 0; yFile < height; yFile++) {
    const yImg = height - 1 - yFile;
    for (let x = 0; x < width; x++) {
      dv.setFloat32((yFile * width + x) * 4, topToBottomRows[yImg][x], littleEndian);
    }
  }
  return concatBytes([header, payload]);
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

describe('hitnetMiddleburyEval3Benchmark', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('loads and SHA-verifies both official MiddEval3 trainingQ scenes (Motorcycle & Teddy) with bit-exact PFM-mask alignment', async () => {
    const fetchDisk = createDiskFixtureFetch();

    for (const sceneId of ['Motorcycle', 'Teddy'] as MiddleburyEval3SceneId[]) {
      const loaded = await loadAndVerifyMiddleburySceneAssets(sceneId, fetchDisk);
      const spec = MIDDLEBURY_EVAL3_Q_SCENES[sceneId];
      expect(loaded.calib.width).toBe(spec.expectedWidth);
      expect(loaded.calib.height).toBe(spec.expectedHeight);
      expect(loaded.calib.ndisp).toBe(spec.expectedNdisp);
      expect(loaded.calib.dyavg).toBeCloseTo(spec.expectedDyavg, 6);
      expect(loaded.calib.dymax).toBeCloseTo(spec.expectedDymax, 6);

      // Verify bottom-to-top PFM flip aligns 100% with mask==0 unknown/infinity holes on BOTH eyes
      const checkMaskAlignment = (pfm: Float32Array, mask: Uint8Array) => {
        let maskZero = 0;
        let infAtZero = 0;
        let nonZeroAtInf = 0;
        let unexpectedMaskValCount = 0;
        for (let i = 0; i < mask.length; i++) {
          const isInf = !Number.isFinite(pfm[i]);
          const mv = mask[i];
          if (mv === 0) {
            maskZero++;
            if (isInf) infAtZero++;
          } else if (isInf) {
            nonZeroAtInf++;
          }
          if (mv !== 0 && mv !== 128 && mv !== 255) {
            unexpectedMaskValCount++;
          }
        }
        expect(unexpectedMaskValCount).toBe(0);
        expect(maskZero).toBeGreaterThan(0);
        expect(infAtZero).toBe(maskZero);
        expect(nonZeroAtInf).toBe(0);
      };

      checkMaskAlignment(loaded.disp0GtPfm.dataTopToBottom, loaded.mask0Nocc.grayTopToBottom);
      checkMaskAlignment(loaded.disp1GtPfm.dataTopToBottom, loaded.mask1Nocc.grayTopToBottom);
    }
  });

  it('parses both little-endian and big-endian PFMs with bottom-to-top vertical flip', () => {
    const rows = [
      [10.5, 20.25, Infinity],
      [1.0, 2.0, 3.0],
    ];
    for (const littleEndian of [true, false]) {
      const pfm = parsePfmSingleBand(buildTestPfmBytes(3, 2, rows, littleEndian));
      expect(pfm.width).toBe(3);
      expect(pfm.height).toBe(2);
      expect(pfm.littleEndian).toBe(littleEndian);
      // Top row (y=0) should be [10.5, 20.25, Infinity], bottom row (y=1) [1.0, 2.0, 3.0]
      expect(pfm.dataTopToBottom[0]).toBeCloseTo(10.5, 5);
      expect(pfm.dataTopToBottom[1]).toBeCloseTo(20.25, 5);
      expect(pfm.dataTopToBottom[2]).toBe(Infinity);
      expect(pfm.dataTopToBottom[3]).toBeCloseTo(1.0, 5);
      expect(pfm.dataTopToBottom[4]).toBeCloseTo(2.0, 5);
      expect(pfm.dataTopToBottom[5]).toBeCloseTo(3.0, 5);
    }
  });

  it('parses Middlebury calib.txt accurately and rejects invalid fields', () => {
    const parsed = parseMiddleburyCalibTxt(
      'cam0=[999.421 0 294.182; 0 999.421 252.932; 0 0 1]\n' +
        'cam1=[999.421 0 326.96; 0 999.421 252.932; 0 0 1]\n' +
        'doffs=32.778\nbaseline=193.001\nwidth=741\nheight=497\nndisp=70\nisint=0\nvmin=7\nvmax=65\ndyavg=0.23\ndymax=0.379\n'
    );
    expect(parsed.width).toBe(741);
    expect(parsed.height).toBe(497);
    expect(parsed.ndisp).toBe(70);
    expect(parsed.dymax).toBeCloseTo(0.379, 5);
    expect(() => parseMiddleburyCalibTxt('width=741\n')).toThrow(/calib.txt/);
  });

  it('nearest-grid GT resampling avoids interpolation across edges/invalids and rejects <=0, >ndisp, and crop-lost correspondence pre-prediction', () => {
    const srcW = 640;
    const srcH = 480;
    const calib: MiddleburyCalibData = {
      cam0: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      cam1: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      doffs: 0,
      baseline: 100,
      width: srcW,
      height: srcH,
      ndisp: 80,
      isint: 0,
      vmin: 5,
      vmax: 70,
      dyavg: 0,
      dymax: 0,
    };

    const gtData = new Float32Array(srcW * srcH).fill(30);
    const maskData = new Uint8Array(srcW * srcH).fill(255);
    // Row 1 (iy=1, hit by my=0 since round((0+0.5)*2 - 0.5) = 1) contains <=0, Infinity, >ndisp, and sharp edge 60
    const row1 = srcW * 1;
    for (let x = 0; x < 40; x++) {
      gtData[row1 + x] = 0; // <= 0 invalid
    }
    gtData[row1 + 101] = Infinity; // Infinity hole (mx=50 hits ix=round(100.5)=101)
    gtData[row1 + 103] = 95; // > ndisp (mx=51 hits ix=103)
    gtData[row1 + 201] = 60; // left eye at x=201 -> xMatch = 201 - 60 = 141 (< 160 crop start)
    maskData[row1 + 201] = 255;

    const gtPfm = {
      magic: 'Pf' as const,
      width: srcW,
      height: srcH,
      scaleFactor: -1,
      littleEndian: true,
      dataTopToBottom: gtData,
    };
    const maskNocc = { width: srcW, height: srcH, colorType: 0, grayTopToBottom: maskData };

    // With cropRect starting at x=160 (width=480), scaleX = 320/480 = 2/3.
    const resampledLeft = resampleMiddleburyEyeGtNearestGrid({
      eye: 'left',
      gtPfm,
      maskNocc,
      calib,
      cropRect: { x: 160, y: 0, width: 480, height: 480 },
    });
    expect(resampledLeft.scaleX).toBeCloseTo(320 / 480, 6);
    expect(resampledLeft.coverage.cropLostCorrespondenceCount).toBeGreaterThan(0);
    // For x in [160, 189], x_match = ix - 30 < 160 -> lost to crop BEFORE any prediction runs!
    expect(resampledLeft.nonoccMask[0]).toBe(0);

    // Full-frame left eye at ix <= 29 also has x_match = ix - 30 < 0 -> rejected by correspondence bounds!
    const resampledFull = resampleMiddleburyEyeGtNearestGrid({
      eye: 'left',
      gtPfm,
      maskNocc,
      calib,
    });
    expect(resampledFull.scaleX).toBe(0.5);
    expect(resampledFull.coverage.gtInvalidOrNonPositiveCount).toBeGreaterThan(0);
    expect(resampledFull.coverage.gtOverNdispCount).toBeGreaterThan(0);
    expect(resampledFull.coverage.cropLostCorrespondenceCount).toBeGreaterThan(0);
    // Sharp edge value (60 * 0.5 = 30.0) without averaging with neighbors!
    const mxEdge = Math.floor((200 + 0.5) * (320 / srcW) - 0.5);
    const valAtEdge = resampledFull.gtDisparityModelPx[mxEdge];
    // Nearest-neighbor hits exact 30.0 (from 60 * 0.5) or exact neighbor 15.0 (from 30 * 0.5), never blurred midpoint!
    expect(valAtEdge === 30 || valAtEdge === 15).toBe(true);
  });

  it('never drops invalid predictions (NaN / negative) via prediction-dependent exclusions and detects sign orientation', async () => {
    const fetchDisk = createDiskFixtureFetch();
    const mcycle = await loadAndVerifyMiddleburySceneAssets('Motorcycle', fetchDisk);
    const leftGt = resampleMiddleburyEyeGtNearestGrid({
      eye: 'left',
      gtPfm: mcycle.disp0GtPfm,
      maskNocc: mcycle.mask0Nocc,
      calib: mcycle.calib,
    });
    const rightGt = resampleMiddleburyEyeGtNearestGrid({
      eye: 'right',
      gtPfm: mcycle.disp1GtPfm,
      maskNocc: mcycle.mask1Nocc,
      calib: mcycle.calib,
    });

    // Build exact GT prediction with 1 negative value and 1 NaN inside nonocc
    const predExact = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT);
    let firstNonocc = -1;
    let secondNonocc = -1;
    for (let i = 0; i < HITNET_MIDDLEBURY_PIXEL_COUNT; i++) {
      if (leftGt.nonoccMask[i] === 1) {
        if (firstNonocc === -1) firstNonocc = i;
        else if (secondNonocc === -1) secondNonocc = i;
        predExact[i] = leftGt.gtDisparityModelPx[i];
      }
    }
    expect(firstNonocc).toBeGreaterThanOrEqual(0);
    expect(secondNonocc).toBeGreaterThan(firstNonocc);

    // Positive-sign GT sanity check on perfect pred
    const sanity = evaluateEyeSignSanity(predExact, leftGt, rightGt);
    expect(sanity.correctEyePosSignEpeModelPx).toBeCloseTo(0, 6);
    expect(sanity.negatedGtEpeModelPx).toBeGreaterThan(20);
    expect(sanity.positiveSignConfirmed).toBe(true);

    // Inject negative prediction -> counted as invalidPred AND contributes full error to EPE/Bad3
    predExact[firstNonocc] = -2.0;
    const negMetrics = evaluateMiddleburyDisparityMask(predExact, leftGt, 'nonocc');
    expect(negMetrics.invalidPredCount).toBe(1);
    expect(negMetrics.negativePredCount).toBe(1);
    expect(negMetrics.nonFinitePredCount).toBe(0);
    expect(negMetrics.epeModelPx).toBeGreaterThan(0);
    expect(negMetrics.bad3Count).toBe(1);

    // Inject NaN -> nonFinitePredCount=1, epeModelPx=Infinity, bad1/bad3 incremented
    predExact[secondNonocc] = Number.NaN;
    const nanMetrics = evaluateMiddleburyDisparityMask(predExact, leftGt, 'nonocc');
    expect(nanMetrics.nonFinitePredCount).toBe(1);
    expect(nanMetrics.invalidPredCount).toBe(2);
    expect(nanMetrics.epeModelPx).toBe(Infinity);
    expect(nanMetrics.bad1Count).toBe(2);
  });

  it('orchestrates both Motorcycle and Teddy scenes with verified M1 pre-session SHAs, separate WebGPU dispatch profiling, matched warm runs >= 2, and go/no-go gate report', async () => {
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

    const gpuNonProfCalls: Array<{ warmRuns: number; feedSha: string }> = [];
    const gpuProfCalls: number[] = [];
    const wasmCalls: Array<{ warmRuns: number; feedSha: string }> = [];

    const mockRunner: ResidualDiagnosticWorkerRunner = async (req) => {
      const feedSha = nodeSha256Hex(new Uint8Array(req.feedBuffer));
      const warmRuns = req.warmRuns ?? 0;
      const pOut = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT).fill(12.0);
      const sOut = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT).fill(12.0);
      if (req.provider === 'wasm') {
        sOut[10] = 12.002;
        wasmCalls.push({ warmRuns, feedSha });
      } else if (req.enableWebGpuProfiling) {
        gpuProfCalls.push(warmRuns);
      } else {
        gpuNonProfCalls.push({ warmRuns, feedSha });
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
          'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.asyncify.wasm',
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
        feedTensorShape: [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        sessionInputNames: ['input'],
        sessionOutputNames: ['reference_output_disparity', 'secondary_output_disparity'],
        workerGpuAdapter: req.provider === 'webgpu' ? 'Mock WebGPU Adapter' : null,
        gpuValidationErrors: [],
        gpuKernelProfileSummary: req.enableWebGpuProfiling
          ? {
              profilingRequested: true,
              ortProfilingKernelTypes: { Conv: 42 },
              ortProfilingSamplePrograms: ['Conv'],
              webgpuComputePassCount: 12,
              webgpuDispatchWorkgroupsCount: 36,
              webgpuDispatchIndirectCount: 0,
              webgpuTimestampWritePassCount: 0,
              webgpuQueueSubmitCount: 2,
              webgpuCommandBufferCount: 2,
              webgpuQuerySetCount: 0,
              webgpuSamplePassLabels: ['Conv'],
              deviceObserverAttached: true,
              ortProfilingEventCount: 42,
              timestampQuerySupported: true,
              note: 'mock dispatch',
            }
          : null,
        modelByteLength: req.modelBuffer.byteLength,
        modelSha256: m1Sha,
        feedSha256: feedSha,
        createMs: 25,
        coldRunMs: 18,
        warmRunMs: Array.from({ length: warmRuns }, (_, i) => 7.5 + i * 0.2),
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

    const report = await runHitnetMiddleburyEval3Benchmark({
      m1Bytes: m1Stub,
      warmRuns: 3,
      isDev: true,
      runWorker: mockRunner,
      loadSceneAssets: (sceneId) => loadAndVerifyMiddleburySceneAssets(sceneId, fetchDisk),
      buildFeed: async ({ im0Bytes, im1Bytes }) => {
        const leftRgba = new Uint8ClampedArray(HITNET_MIDDLEBURY_PIXEL_COUNT * 4);
        const rightRgba = new Uint8ClampedArray(HITNET_MIDDLEBURY_PIXEL_COUNT * 4);
        leftRgba[0] = im0Bytes[100];
        rightRgba[0] = im1Bytes[100];
        const feedFloat32 = packRgbaStereoEyesToHitnetFeed(leftRgba, rightRgba);
        const feedSha256 = nodeSha256Hex(
          new Uint8Array(feedFloat32.buffer, feedFloat32.byteOffset, feedFloat32.byteLength)
        );
        return { feedFloat32, feedSha256 };
      },
    });

    expect(report.ok).toBe(true);
    expect(report.benchmarkMode).toBe('local-public-fixture-320x240-not-official-leaderboard');
    expect(report.m1Sha256).toBe(m1Sha);
    expect(report.n1OffProof.verified).toBe(true);
    expect(gpuNonProfCalls).toHaveLength(2);
    expect(wasmCalls).toHaveLength(2);
    expect(gpuProfCalls).toEqual([0, 0]);
    expect(gpuNonProfCalls.map((c) => c.feedSha)).toEqual(wasmCalls.map((c) => c.feedSha));
    expect(report.scenes.Motorcycle.webgpu.profilingOffTimings.warmRunMs).toHaveLength(3);
    expect(report.scenes.Teddy.webgpu.dispatchCheckProfilingOn?.dispatchObserved).toBe(true);
    expect(report.scenes.Motorcycle.leftGtCoverage.nonoccValidGtCount).toBeGreaterThan(50000);
    expect(report.scenes.Teddy.leftGtCoverage.nonoccValidGtCount).toBeGreaterThan(50000);
    expect(report.productionGoNoGoEvaluation.explicitLimits.length).toBeGreaterThan(3);

    // Verify canonical mismatch aborts pre-session
    await expect(
      runHitnetMiddleburyEval3Benchmark({
        m1Bytes: new Uint8Array(100),
        isDev: true,
        runWorker: mockRunner,
      })
    ).rejects.toThrow(/Exact canonical M1 verification failed/);
  });
});
