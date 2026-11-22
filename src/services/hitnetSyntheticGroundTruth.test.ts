// @vitest-environment node
import crypto from 'node:crypto';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  computeSyntheticGtRois,
  evaluateDisparityRoiMetricsPx,
  generateSyntheticStereoGroundTruthPair,
  runHitnetSyntheticGroundTruthTest,
} from './hitnetSyntheticGroundTruth';
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
    encodeStrField(2, 'synthetic_gt_stub'),
    encodeInt64Initializer('slot0_shape', [1, 6, 240, 320]),
    encodeInt64Initializer('out_shape', [1, 240, 320, 1]),
    encodeLenField(11, encodeValueInfoFloat('input', [1, 6, 240, 320])),
    encodeLenField(12, encodeValueInfoFloat('reference_output_disparity', [1, 240, 320, 1])),
    encodeLenField(12, encodeValueInfoFloat('secondary_output_disparity', [1, 240, 320, 1])),
  ]);
  const opsetImport = encodeLenField(8, concatBytes([encodeStrField(1, ''), encodeVarintField(2, 12)]));
  const baseModel = concatBytes([
    encodeVarintField(1, 7),
    encodeStrField(2, 'synthetic'),
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

describe('hitnetSyntheticGroundTruth', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('generates deterministic D16 & D32 feeds with verified stereo shift correspondence and exact feed SHA-256 hashes', async () => {
    const d16a = await generateSyntheticStereoGroundTruthPair(16);
    const d16b = await generateSyntheticStereoGroundTruthPair(16);
    const d32 = await generateSyntheticStereoGroundTruthPair(32);

    expect(d16a.feedSha256).toBe(d16b.feedSha256);
    expect(d16a.feedSha256).not.toBe(d32.feedSha256);
    expect(d16a.feedDims).toEqual([1, 6, 240, 320]);
    expect(d16a.feedFloat32.length).toBe(6 * 240 * 320);

    // Check exact shift correspondence across all 3 RGB channels: I_R(y, x) == I_L(y, x + d_true)
    const plane = 320 * 240;
    for (let y = 16; y < 224; y += 31) {
      for (let x = 20; x < 240; x += 29) {
        for (let c = 0; c < 3; c++) {
          const iR = d16a.feedFloat32[(3 + c) * plane + y * 320 + x];
          const iLShifted = d16a.feedFloat32[c * plane + y * 320 + (x + 16)];
          expect(iR).toBe(iLShifted);

          const iR32 = d32.feedFloat32[(3 + c) * plane + y * 320 + x];
          const iLShifted32 = d32.feedFloat32[c * plane + y * 320 + (x + 32)];
          expect(iR32).toBe(iLShifted32);
        }
      }
    }
  });

  it('computes analytic occlusion/valid masks and identical 53,248-pixel common cropped ROI across D16 and D32', () => {
    const r16 = computeSyntheticGtRois(16);
    const r32 = computeSyntheticGtRois(32);

    expect(r16.analyticValidMask.primaryLeft).toEqual({
      xStart: 32,
      xEndExclusive: 304,
      yStart: 16,
      yEndExclusive: 224,
      pixelCount: 208 * 272,
    });
    expect(r16.analyticValidMask.secondaryRight).toEqual({
      xStart: 16,
      xEndExclusive: 288,
      yStart: 16,
      yEndExclusive: 224,
      pixelCount: 208 * 272,
    });

    // Common ROI is strictly equal across both D16 & D32: 208 x 256 = 53,248 pixels each
    expect(r16.commonCroppedRoi).toEqual(r32.commonCroppedRoi);
    expect(r32.commonCroppedRoi.primaryLeft.pixelCount).toBe(53248);
    expect(r32.commonCroppedRoi.secondaryRight.pixelCount).toBe(53248);
  });

  it('never drops negative or erroneous predictions via prediction-dependent masking when computing EPE/p95/Bad1/Bad3/invalidRate', () => {
    const rois = computeSyntheticGtRois(16);
    const roi = rois.commonCroppedRoi.primaryLeft;
    const pred = new Float32Array(320 * 240).fill(16.0);

    // Inject 1,000 negative predictions (-5.0px => |err|=21.0px) inside common ROI!
    let injected = 0;
    for (let y = roi.yStart; y < roi.yEndExclusive && injected < 1000; y++) {
      for (let x = roi.xStart; x < roi.xEndExclusive && injected < 1000; x++) {
        pred[y * 320 + x] = -5.0;
        injected++;
      }
    }

    const m = evaluateDisparityRoiMetricsPx(pred, 'primary-common-roi', roi, 16);
    expect(m.evaluatedCount).toBe(53248);
    expect(m.invalidCount).toBe(1000);
    expect(m.invalidRate).toBeCloseTo(1000 / 53248, 8);
    // All 1000 negative errors are included in EPE, Bad1, Bad3 (no excluding errors!)
    expect(m.bad1Count).toBe(1000);
    expect(m.bad3Count).toBe(1000);
    expect(m.epePx).toBeCloseTo((1000 * 21.0) / 53248, 6);
    expect(m.maxAbsErrorPx).toBe(21.0);
  });

  it('executes D16 and D32 full-graph untapped M1 runs with OFF-profile timing and separate profiling-ON GPU dispatch check', async () => {
    const modelBytes = buildMinimalCanonicalM1Stub();
    vi.spyOn(crypto.webcrypto.subtle, 'digest').mockImplementation(async (_alg, data) => {
      const u8 =
        data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(
              (data as ArrayBufferView).buffer,
              (data as ArrayBufferView).byteOffset,
              (data as ArrayBufferView).byteLength
            );
      if (u8.byteLength === modelBytes.byteLength) {
        return hexToArrayBuffer(HITNET_M1_FULL_GRAPH_SHA256);
      }
      return hexToArrayBuffer(nodeSha256Hex(u8));
    });

    const d16Pair = await generateSyntheticStereoGroundTruthPair(16);
    const profCalls: Array<{ profiling: boolean; provider: string }> = [];

    const runner: ResidualDiagnosticWorkerRunner = async (req) => {
      const f32 = new Float32Array(req.feedBuffer);
      const isD16 = nodeSha256Hex(new Uint8Array(f32.buffer)) === d16Pair.feedSha256;
      const dispVal = isD16 ? 16.0 : 32.0;
      const isGpu = req.provider === 'webgpu';
      profCalls.push({ profiling: Boolean(req.enableWebGpuProfiling), provider: req.provider });

      if (req.enableWebGpuProfiling && isGpu) {
        return {
          id: 1,
          ok: true,
          pinnedOrtVersion: '1.30.0',
          runtimeAssetFamily: 'asyncify',
          entryModuleUrl: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs',
          wasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
          ortVersions: { common: '1.30.0', web: '1.30.0' },
          loadedOrtResources: [],
          foreignOrtResources: [],
          loadedAssetProvenance: [],
          provider: 'webgpu',
          configuredProviders: ['webgpu'],
          fallbackBehavior: 'no-fallback-explicit-error',
          graphOptimizationLevel: req.graphOptimizationLevel,
          sessionOptionsApplied: {
            executionProviders: ['webgpu'],
            graphOptimizationLevel: req.graphOptimizationLevel,
            enableCpuMemArena: true,
            enableMemPattern: true,
            numThreads: 1,
            proxy: false,
            wasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
            disableCpuEpFallback: false,
            fetchMode: 'all-session-outputs',
            profilingRequested: true,
          },
          workerGpuAdapter: 'amd rdna3 mock',
          modelByteLength: modelBytes.byteLength,
          modelSha256: HITNET_M1_FULL_GRAPH_SHA256,
          feedSha256: nodeSha256Hex(new Uint8Array(f32.buffer)),
          feedTensorName: 'input',
          feedTensorShape: [1, 6, 240, 320],
          feedTensorDataType: 'float32',
          sessionInputNames: ['input'],
          sessionOutputNames: ['reference_output_disparity', 'secondary_output_disparity'],
          gpuValidationErrors: [],
          gpuKernelProfileSummary: {
            profilingRequested: true,
            timestampQuerySupported: true,
            webgpuComputePassCount: 8,
            webgpuDispatchWorkgroupsCount: 120,
            webgpuDispatchIndirectCount: 0,
            webgpuTimestampWritePassCount: 0,
            webgpuQueueSubmitCount: 2,
            webgpuCommandBufferCount: 2,
            webgpuQuerySetCount: 0,
            webgpuSamplePassLabels: ['Conv'],
            ortProfilingEventCount: 8,
            ortProfilingKernelTypes: { Conv: 4 },
            ortProfilingSamplePrograms: ['Conv'],
            deviceObserverAttached: true,
            note: 'mock dispatch',
          },
          createMs: 25,
          coldRunMs: 40,
          warmRunMs: [],
          outputs: [],
        };
      }

      const pOut = new Float32Array(76800).fill(isGpu ? dispVal + 0.1 : dispVal);
      const sOut = new Float32Array(76800).fill(isGpu ? dispVal + 0.2 : dispVal);
      return {
        id: 1,
        ok: true,
        pinnedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        entryModuleUrl: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs',
        wasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
        ortVersions: { common: '1.30.0', web: '1.30.0' },
        loadedOrtResources: [],
        foreignOrtResources: [],
        loadedAssetProvenance: [],
        provider: req.provider,
        configuredProviders: [req.provider],
        fallbackBehavior: 'no-fallback-explicit-error',
        graphOptimizationLevel: req.graphOptimizationLevel,
        sessionOptionsApplied: {
          executionProviders: [req.provider],
          graphOptimizationLevel: req.graphOptimizationLevel,
          enableCpuMemArena: isGpu,
          enableMemPattern: isGpu,
          numThreads: 1,
          proxy: false,
          wasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
          disableCpuEpFallback: false,
          fetchMode: 'all-session-outputs',
          profilingRequested: false,
        },
        workerGpuAdapter: isGpu ? 'amd rdna3 mock' : null,
        modelByteLength: modelBytes.byteLength,
        modelSha256: HITNET_M1_FULL_GRAPH_SHA256,
        feedSha256: nodeSha256Hex(new Uint8Array(f32.buffer)),
        feedTensorName: 'input',
        feedTensorShape: [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        sessionInputNames: ['input'],
        sessionOutputNames: ['reference_output_disparity', 'secondary_output_disparity'],
        gpuValidationErrors: isGpu ? [] : null,
        gpuKernelProfileSummary: null,
        createMs: 12,
        coldRunMs: 15,
        warmRunMs: [9, 11],
        outputs: [
          { name: 'reference_output_disparity', dims: [1, 240, 320, 1], dataType: 'float32', buffer: pOut.buffer },
          { name: 'secondary_output_disparity', dims: [1, 240, 320, 1], dataType: 'float32', buffer: sOut.buffer },
        ],
      };
    };

    const rep = await runHitnetSyntheticGroundTruthTest({
      m1Bytes: modelBytes,
      graphOptimizationLevel: 'basic',
      warmRuns: 2,
      runWorker: runner,
      isDev: true,
    });

    expect(rep.ok).toBe(true);
    expect(rep.cases.D16.webgpu.profilingOffTimings.warmMedianMs).toBe(10);
    expect(rep.cases.D16.webgpu.dispatchCheckProfilingOn?.dispatchObserved).toBe(true);
    expect(rep.cases.D16.webgpu.primary.commonRoi.epePx).toBeCloseTo(0.1, 5);
    expect(rep.cases.D16.wasm.primary.commonRoi.epePx).toBe(0);
    expect(rep.cases.D32.webgpu.secondary.commonRoi.epePx).toBeCloseTo(0.2, 5);
    expect(profCalls.filter((c) => !c.profiling).length).toBe(4);
    expect(profCalls.filter((c) => c.profiling).length).toBe(2);
  });
});
