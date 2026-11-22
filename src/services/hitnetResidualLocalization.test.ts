// @vitest-environment node
import crypto from 'node:crypto';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  checkFloat32BitIdentity,
  getImmediateParentTapGroups,
  getOrderedIntervalNodesForBisection,
  HITNET_RESIDUAL_PARITY_GATES,
  HITNET_RESIDUAL_SOURCE_CALIBRATION_DEFAULT,
  inspectHitnetM1ResidualLadderTopology,
  passesStrictRawGate,
  runHitnetResidualLocalizationDiagnostic,
  summarizeDisparityResidualAndSourceDepthImpact,
  validateHitnetFinalOutputs,
  type ResidualDiagnosticWorkerRunner,
} from './hitnetResidualLocalization';
import { HITNET_DEV_NON_JSEP_FIXTURE_SPEC } from './stereoDepthWebGpuDevFixture';
import { HITNET_M1_FULL_GRAPH_SHA256 } from './stereoDepthWorkerCore';
import type { OrtVersionDiagnosticResponse } from './hitnetOrtComparisonHelper';

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

function encodeSignedVarint64(val: number): Uint8Array {
  let v = BigInt.asUintN(64, BigInt(val));
  const out: number[] = [];
  while (v >= 0x80n) {
    out.push(Number(v & 0x7fn) | 0x80);
    v >>= 7n;
  }
  out.push(Number(v));
  return new Uint8Array(out);
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

function encodeAttrInts(name: string, vals: readonly number[]): Uint8Array {
  return encodeLenField(
    5,
    concatBytes([
      encodeStrField(1, name),
      ...vals.map((v) => encodeVarintField(8, v)),
      encodeVarintField(20, 7),
    ])
  );
}

function encodeNode(
  opType: string,
  name: string,
  inputs: readonly string[],
  outputs: readonly string[],
  attrs: readonly Uint8Array[] = []
): Uint8Array {
  return encodeLenField(
    1,
    concatBytes([
      ...inputs.map((inp) => encodeStrField(1, inp)),
      ...outputs.map((out) => encodeStrField(2, out)),
      encodeStrField(3, name),
      encodeStrField(4, opType),
      ...attrs,
    ])
  );
}

function encodeValueInfoFloat(name: string, dims: readonly number[]): Uint8Array {
  const dimFields = dims.map((d) => encodeLenField(1, encodeVarintField(1, d)));
  const shapeProto = encodeLenField(2, concatBytes(dimFields));
  const tensorTypeProto = encodeLenField(1, concatBytes([encodeVarintField(1, 1), shapeProto]));
  return concatBytes([encodeStrField(1, name), encodeLenField(2, tensorTypeProto)]);
}

function encodeFloatInitializer(name: string, dims: readonly number[]): Uint8Array {
  return encodeLenField(
    5,
    concatBytes([
      ...dims.map((d) => encodeVarintField(1, d)),
      encodeVarintField(2, 1),
      encodeStrField(8, name),
    ])
  );
}

function encodeInt64Initializer(name: string, vals: readonly number[]): Uint8Array {
  return encodeLenField(
    5,
    concatBytes([
      encodeVarintField(1, vals.length),
      encodeVarintField(2, 7), // INT64
      ...vals.map((v) => concatBytes([encodeVarint((7 << 3) | 0), encodeSignedVarint64(v)])),
      encodeStrField(8, name),
    ])
  );
}

function buildSyntheticMultiBranchHitnetOnnx(targetByteLength?: number): Uint8Array {
  const nodes = [
    encodeNode('Slice', 'fe_shared/slice0', ['input', 's0', 'e1', 'a1'], ['stem_s']),
    encodeNode('Transpose', 'fe_shared/stem_trans', ['stem_s'], ['stem_0'], [encodeAttrInts('perm', [0, 2, 3, 1])]),
    encodeNode('Relu', 'fe_shared/stem_r1', ['stem_0'], ['stem_1']),
    encodeNode('Add', 'fe_shared/stem_add2', ['stem_1', 'zero_init'], ['stem_2']),
    encodeNode('Relu', 'fe_shared/stem_r3', ['stem_2'], ['stem_3']),
    encodeNode('Relu', 'branch_left/fe3', ['stem_3'], ['fe_shared_3']),
    encodeNode('Relu', 'branch_right/fe31', ['stem_3'], ['fe_shared_31']),
    encodeNode('Add', 'prim/add_a', ['fe_shared_3', 'zero_init'], ['prim_a']),
    encodeNode('Mul', 'prim/mul_b', ['prim_a', 'fe_shared_3'], ['prim_b']),
    encodeNode('Relu', 'prim/head', ['prim_b'], ['reference_output_disparity']),
    encodeNode('Add', 'sec/head', ['fe_shared_31', 'zero_init'], ['secondary_output_disparity']),
  ];

  const graphBody = concatBytes([
    ...nodes,
    encodeStrField(2, 'synthetic_hitnet_m1_graph'),
    encodeInt64Initializer('s0', [0]),
    encodeInt64Initializer('e1', [1]),
    encodeInt64Initializer('a1', [1]),
    encodeFloatInitializer('zero_init', [1, 240, 320, 1]),
    encodeLenField(11, encodeValueInfoFloat('input', [1, 6, 240, 320])),
    encodeLenField(12, encodeValueInfoFloat('reference_output_disparity', [1, 240, 320, 1])),
    encodeLenField(12, encodeValueInfoFloat('secondary_output_disparity', [1, 240, 320, 1])),
  ]);

  const opsetImport = encodeLenField(8, concatBytes([encodeStrField(1, ''), encodeVarintField(2, 13)]));
  const baseParts = [
    encodeVarintField(1, 7),
    encodeStrField(2, 'synthetic'),
    encodeLenField(7, graphBody),
    opsetImport,
  ];
  const baseModel = concatBytes(baseParts);

  if (!targetByteLength) return baseModel;
  if (targetByteLength <= baseModel.byteLength + 16) {
    throw new Error(`targetByteLength too small: ${targetByteLength}`);
  }

  // Pad ModelProto field 14 (metadata_props: StringStringEntryProto) with a dummy value to exact byteLength
  const padKey = encodeStrField(1, 'pad');
  const dummyValLen = 1000;
  const approxPrefix = baseModel.byteLength + padKey.byteLength + 6;
  let valLen = targetByteLength - approxPrefix;
  while (true) {
    const valField = encodeLenField(2, new Uint8Array(valLen).fill(0x61));
    const entry = encodeLenField(14, concatBytes([padKey, valField]));
    const total = baseModel.byteLength + entry.byteLength;
    if (total === targetByteLength) {
      return concatBytes([baseModel, entry]);
    }
    valLen += targetByteLength - total;
    if (valLen < dummyValLen - 1000 && valLen < 0) {
      throw new Error('Failed to converge padding length');
    }
  }
}

function readGraphOutputNamesFromOnnx(bytes: Uint8Array): string[] {
  const topology = inspectHitnetM1ResidualLadderTopology(bytes);
  return [...topology.graphOutputNames];
}

function makeFilledFloat32(length: number, baseVal: number, perIndexDelta = 0): Float32Array {
  const arr = new Float32Array(new ArrayBuffer(length * 4));
  for (let i = 0; i < length; i++) {
    arr[i] = baseVal + (i % 97) * 1e-3 + perIndexDelta;
  }
  return arr;
}

function makeFilledFloat32Buffer(length: number, baseVal: number, perIndexDelta = 0): ArrayBuffer {
  return makeFilledFloat32(length, baseVal, perIndexDelta).buffer as ArrayBuffer;
}

function makeMockSuccessBase(req: {
  provider: 'webgpu' | 'wasm';
  graphOptimizationLevel: 'disabled' | 'basic';
}) {
  return {
    id: 1,
    ok: true as const,
    pinnedOrtVersion: '1.30.0' as const,
    runtimeAssetFamily: 'asyncify' as const,
    entryModuleUrl: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs',
    wasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
    ortVersions: { common: '1.30.0', web: '1.30.0' },
    sessionOptionsApplied: {
      executionProviders: [req.provider],
      graphOptimizationLevel: req.graphOptimizationLevel,
      enableCpuMemArena: req.provider === 'webgpu',
      enableMemPattern: req.provider === 'webgpu',
      numThreads: 1,
      proxy: false,
      wasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
      disableCpuEpFallback: false,
      fetchMode: 'all-session-outputs' as const,
      profilingRequested: false,
    },
    sessionInputNames: ['input'],
    sessionOutputNames: ['reference_output_disparity', 'secondary_output_disparity'],
    workerGpuAdapter: req.provider === 'webgpu' ? 'amd rdna3 mock' : null,
    loadedAssetProvenance: [],
    provider: req.provider,
    configuredProviders: [req.provider],
    fallbackBehavior: 'no-fallback-explicit-error' as const,
    graphOptimizationLevel: req.graphOptimizationLevel,
  };
}

describe('hitnetResidualLocalization (DEV-only non-JSEP residual localization)', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('keeps strict raw thresholds unchanged and enforces DEV-only guard', () => {
    expect(HITNET_RESIDUAL_PARITY_GATES).toEqual({
      expectedLength: 76800,
      expectedDims: [1, 240, 320, 1],
      expectedDataType: 'float32',
      maxAbsDiffMax: 1e-3,
      p99AbsDiffMax: 1e-4,
      meanAbsDiffMax: 1e-5,
    });
    expect(HITNET_RESIDUAL_SOURCE_CALIBRATION_DEFAULT.validDepthMinDisparityPx).toBe(0.05);
    expect(() =>
      validateHitnetFinalOutputs([
        {
          name: 'secondary_output_disparity',
          dims: [1, 240, 320, 1],
          dataType: 'float32',
          buffer: new Float32Array(76800).buffer,
        },
        {
          name: 'reference_output_disparity',
          dims: [1, 240, 320, 1],
          dataType: 'float32',
          buffer: new Float32Array(76800).buffer,
        },
      ])
    ).not.toThrow();

    const wrongOrder = validateHitnetFinalOutputs([
      {
        name: 'secondary_output_disparity',
        dims: [1, 240, 320, 1],
        dataType: 'float32',
        buffer: new Float32Array(76800).buffer,
      },
      {
        name: 'reference_output_disparity',
        dims: [1, 240, 320, 1],
        dataType: 'float32',
        buffer: new Float32Array(76800).buffer,
      },
    ]);
    expect(wrongOrder.ok).toBe(false);
    expect(wrongOrder.error).toContain('Ordered output[0] mismatch');

    const withNaN = new Float32Array(76800);
    withNaN[123] = Number.NaN;
    const nanCheck = validateHitnetFinalOutputs([
      {
        name: 'reference_output_disparity',
        dims: [1, 240, 320, 1],
        dataType: 'float32',
        buffer: withNaN.buffer,
      },
      {
        name: 'secondary_output_disparity',
        dims: [1, 240, 320, 1],
        dataType: 'float32',
        buffer: new Float32Array(76800).buffer,
      },
    ]);
    expect(nanCheck.ok).toBe(false);
    expect(nanCheck.error).toContain('non-finite');

    const okCheck = validateHitnetFinalOutputs([
      {
        name: 'reference_output_disparity',
        dims: [1, 240, 320, 1],
        dataType: 'float32',
        buffer: new Float32Array(76800).buffer,
      },
      {
        name: 'secondary_output_disparity',
        dims: [1, 240, 320, 1],
        dataType: 'float32',
        buffer: new Float32Array(76800).buffer,
      },
    ]);
    expect(okCheck.ok).toBe(true);
    expect(passesStrictRawGate({
      finiteCount: 76800,
      nonFiniteCount: 0,
      length: 76800,
      min: 0,
      max: 10,
      mean: 5,
      referenceMean: 5,
      maxAbsDiff: 9.9e-4,
      p99AbsDiff: 9.9e-5,
      meanAbsDiff: 9.9e-6,
      rmsDiff: 1e-5,
      nonFiniteMismatchCount: 0,
    })).toBe(true);
  });

  it('rejects non-DEV execution fail-closed', async () => {
    await expect(
      runHitnetResidualLocalizationDiagnostic({
        m1Bytes: new Uint8Array(8),
        f0102FeedFloat32: new Float32Array(1 * 6 * 240 * 320),
        isDev: false,
      })
    ).rejects.toThrow(/DEV-only/i);
  });

  it('verifies bitwise Float32 invariance and source/depth residual summary', () => {
    const a = new Float32Array([0, 1.5, -2.25, 3.0]);
    const b = new Float32Array([0, 1.5, -2.25, 3.0]);
    expect(checkFloat32BitIdentity(a, b).bitIdentical).toBe(true);

    const c = new Float32Array([-0, 1.5, -2.25, 3.0]);
    const negZeroCheck = checkFloat32BitIdentity(a, c);
    expect(negZeroCheck.bitIdentical).toBe(false);
    expect(negZeroCheck.mismatchCount).toBe(1);
    expect(negZeroCheck.firstMismatchIndex).toBe(0);

    // Primary disparity residual summary using real extractDisparityFromOrtTensor & disparityToDepthMeters
    const n = 76800;
    const refPrimary = new Float32Array(n).fill(5.0);
    const candPrimary = new Float32Array(n).fill(5.0);
    // Pixel 10 flips from valid (>0) to negative (masked out as -1 by extractDisparityFromOrtTensor)
    candPrimary[10] = -0.02;
    // Pixel 20: ref=0.04 (0.06 src px > 0.05 cutoff -> valid depth), cand=0.02 (0.03 src px <= 0.05 -> null depth)
    refPrimary[20] = 0.04;
    candPrimary[20] = 0.02;
    // Pixel 325 (y=1, x=5): +0.012 raw disparity diff -> ~32500 / 260.. depth shift
    candPrimary[325] = 5.012;

    const impact = summarizeDisparityResidualAndSourceDepthImpact(
      'F0102(opt=disabled)',
      candPrimary,
      refPrimary,
      new Float32Array(n).fill(2.0),
      new Float32Array(n).fill(2.0)
    );
    expect(impact.scaleToSource).toBe(1.5);
    expect(impact.primarySignBranch.sameBranch).toBe(true);
    expect(impact.primarySourceMaskChanges.refValidToCandInvalid).toBe(1);
    expect(impact.primaryDepthThresholdChanges.refValidToCandNull).toBe(2); // pixel 10 (-1) & pixel 20 (0.03 <= 0.05)
    expect(impact.topPrimaryLocations[0].y).toBe(0);
    expect(impact.topPrimaryLocations[0].x).toBe(10);
    expect(impact.topPrimaryLocations[0].rawAbsDiff).toBeCloseTo(5.02, 4);
    const coord325 = impact.topPrimaryLocations.find((t) => t.y === 1 && t.x === 5);
    expect(coord325).toBeDefined();
    expect(coord325!.rawAbsDiff).toBeCloseTo(0.012, 5);
    expect(coord325!.srcAbsDiffPx).toBeCloseTo(0.018, 5);
    expect(coord325!.depthDiffM).toBeLessThan(0);
  });

  it('parses synthetic multi-branch ONNX topology and extracts interval + immediate non-initializer parents', () => {
    const model = buildSyntheticMultiBranchHitnetOnnx();
    const topology = inspectHitnetM1ResidualLadderTopology(model);

    expect(topology.stemCommonNodes.map((n) => n.outputs[0])).toEqual([
      'stem_s',
      'stem_0',
      'stem_1',
      'stem_2',
      'stem_3',
    ]);
    expect(topology.stemMidTensor).toBe('stem_1');
    expect(topology.stemPreSplitTensor).toBe('stem_3');
    expect(topology.feSplitTensors).toEqual(['fe_shared_3', 'fe_shared_31']);

    const primInterval = getOrderedIntervalNodesForBisection(topology, ['fe_shared_3'], 'reference_output_disparity');
    expect(primInterval.map((n) => n.name)).toEqual(['prim/add_a', 'prim/mul_b']);

    const mulNode = topology.producerByOutput.get('prim_b')!;
    const mulParents = getImmediateParentTapGroups(topology, mulNode);
    expect(mulParents).toEqual([['prim_a', 'fe_shared_3']]);

    // Initializer 'zero_init' is excluded from add_a immediate parents
    const addNode = topology.producerByOutput.get('prim_a')!;
    expect(getImmediateParentTapGroups(topology, addNode)).toEqual([['fe_shared_3']]);
  });

  it('runs 2x2 optimizer controls, separates non-timing profiling, bisects to first diverging operator, and verifies bit-identical finals', async () => {
    const m1Bytes = buildSyntheticMultiBranchHitnetOnnx(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength);
    const f0102Feed = new Float32Array(1 * 6 * 240 * 320).fill(0.25);
    const f0126Feed = new Float32Array(1 * 6 * 240 * 320).fill(0.5);

    const origDigest = crypto.subtle.digest.bind(crypto.subtle);
    vi.spyOn(crypto.subtle, 'digest').mockImplementation(async (algo, data) => {
      const view =
        data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array((data as ArrayBufferView).buffer, (data as ArrayBufferView).byteOffset, (data as ArrayBufferView).byteLength);
      if (view.byteLength === m1Bytes.byteLength && Buffer.compare(view, m1Bytes) === 0) {
        return hexToArrayBuffer(HITNET_M1_FULL_GRAPH_SHA256);
      }
      if (
        view.byteLength === f0102Feed.byteLength &&
        view[0] === new Uint8Array(f0102Feed.buffer)[0] &&
        new Float32Array(view.buffer, view.byteOffset, 1)[0] === 0.25
      ) {
        return hexToArrayBuffer(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256);
      }
      if (
        view.byteLength === f0126Feed.byteLength &&
        new Float32Array(view.buffer, view.byteOffset, 1)[0] === 0.5
      ) {
        return hexToArrayBuffer(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256);
      }
      return origDigest(algo, data);
    });

    const seenRequests: Array<{
      provider: string;
      opt: string;
      profiling: boolean;
      warmRuns: number;
      outputs: string[];
      modelSha: string;
    }> = [];

    const mockWorker: ResidualDiagnosticWorkerRunner = async (req) => {
      expect(req.runtimeAssetFamily).toBe('asyncify');
      const mBytes = new Uint8Array(req.modelBuffer);
      const modelSha =
        mBytes.byteLength === m1Bytes.byteLength && Buffer.compare(mBytes, m1Bytes) === 0
          ? HITNET_M1_FULL_GRAPH_SHA256
          : nodeSha256Hex(mBytes);
      const fVal = new Float32Array(req.feedBuffer)[0];
      const feedSha =
        fVal === 0.25
          ? HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256
          : HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256;
      const requestedOutputs = readGraphOutputNamesFromOnnx(mBytes);
      seenRequests.push({
        provider: req.provider,
        opt: req.graphOptimizationLevel,
        profiling: Boolean(req.enableWebGpuProfiling),
        warmRuns: req.warmRuns ?? 0,
        outputs: requestedOutputs,
        modelSha,
      });

      const isGpu = req.provider === 'webgpu';
      // Introduce divergence at `prim_b` (Mul) and downstream `reference_output_disparity` ONLY on WebGPU,
      // while upstream tensors (`stem_1`, `stem_3`, `fe_shared_3`, `prim_a`) have either 0 or sub-gate (5e-6) diff.
      const buildOutputBuffer = (name: string): { buffer: ArrayBuffer; dims: number[] } => {
        if (name === 'stem_1') {
          return { buffer: makeFilledFloat32Buffer(76800, 1.0, 0), dims: [1, 240, 320, 1] };
        }
        if (name === 'stem_3') {
          // Earliest tiny non-zero sub-gate difference (5e-6 < 1e-5 mean, < 1e-4 p99, < 1e-3 max)
          return { buffer: makeFilledFloat32Buffer(76800, 1.5, isGpu ? 5e-6 : 0), dims: [1, 240, 320, 1] };
        }
        if (name === 'fe_shared_3' || name === 'fe_shared_31' || name === 'prim_a') {
          return { buffer: makeFilledFloat32Buffer(76800, 2.0, isGpu ? 5e-6 : 0), dims: [1, 240, 320, 1] };
        }
        if (name === 'prim_b') {
          // Divergence starts here (> 1e-3 gate)
          return { buffer: makeFilledFloat32Buffer(76800, 4.0, isGpu ? 0.008 : 0), dims: [1, 240, 320, 1] };
        }
        if (name === 'reference_output_disparity') {
          return { buffer: makeFilledFloat32Buffer(76800, 6.0, isGpu ? 0.008 : 0), dims: [1, 240, 320, 1] };
        }
        return { buffer: makeFilledFloat32Buffer(76800, 3.0, isGpu ? 5e-6 : 0), dims: [1, 240, 320, 1] };
      };

      const outputs = requestedOutputs.map((name) => {
        const built = buildOutputBuffer(name);
        return {
          name,
          dataType: 'float32',
          dims: built.dims,
          buffer: built.buffer,
        };
      });

      const resp: OrtVersionDiagnosticResponse = {
        ...makeMockSuccessBase(req),
        loadedOrtResources: [
          'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs',
          'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.asyncify.wasm',
        ],
        foreignOrtResources: [],
        modelByteLength: mBytes.byteLength,
        modelSha256: modelSha,
        feedSha256: feedSha,
        feedTensorName: 'input',
        feedTensorShape: [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        createMs: 12,
        coldRunMs: 15,
        warmRunMs: req.warmRuns ? [10] : [],
        gpuValidationErrors: req.provider === 'webgpu' ? [] : null,
        gpuKernelProfileSummary: req.enableWebGpuProfiling
          ? {
              profilingRequested: true,
              timestampQuerySupported: false,
              ortProfilingEventCount: 100,
              ortProfilingKernelTypes: { Mul: 2 },
              ortProfilingSamplePrograms: ['Mul'],
              webgpuComputePassCount: 100,
              webgpuDispatchWorkgroupsCount: 100,
              webgpuDispatchIndirectCount: 0,
              webgpuTimestampWritePassCount: 0,
              webgpuQueueSubmitCount: 10,
              webgpuCommandBufferCount: 10,
              webgpuQuerySetCount: 0,
              webgpuSamplePassLabels: ['Mul'],
              deviceObserverAttached: true,
              note: 'mock profile',
            }
          : null,
        outputs,
      };
      return resp;
    };

    const logs: string[] = [];
    const report = await runHitnetResidualLocalizationDiagnostic({
      m1Bytes,
      f0102FeedFloat32: f0102Feed,
      f0126FeedFloat32: f0126Feed,
      runWorker: mockWorker,
      isDev: true,
      onLog: (l) => logs.push(l),
    });

    expect(report.ok).toBe(true);
    expect(report.abortReason).toBeNull();
    expect(report.frames.F0102).not.toBeNull();
    expect(report.frames.F0102!.comparisons.webgpuBasicVsWasmBasic.passesBothOutputs).toBe(false);
    expect(report.frames.F0102!.comparisons.webgpuDisabledVsWasmDisabled.passesBothOutputs).toBe(false);
    expect(report.frames.F0102!.comparisons.webgpuBasicVsWasmBasic.disclosedCrossEpDifferences.length).toBeGreaterThanOrEqual(4);
    expect(report.frames.F0126).not.toBeNull();

    // Ensure timing runs had profiling=false and profile runs had warmRuns=0 & profiling=true
    const f0102ProfRequests = seenRequests.filter((r) => r.profiling);
    expect(f0102ProfRequests).toHaveLength(4); // F0102 (basic, disabled) + F0126 (basic, disabled)
    expect(f0102ProfRequests.every((r) => r.warmRuns === 0 && r.provider === 'webgpu')).toBe(true);

    expect(report.localization?.executed).toBe(true);
    expect(report.localization?.selectedOptLevel).toBe('disabled');
    expect(report.localization?.abortedByFinalPerturbation).toBe(false);
    // Earliest non-zero (2e-5 at stem_3) separated from first gate-breaching operator (`prim/mul_b` -> `prim_b`, 0.008)
    expect(report.localization?.earliestNonZeroTap?.tensorName).toBe('stem_3');
    expect(report.localization?.firstGateBreachingTap?.tensorName).toBe('prim_b');
    expect(report.localization?.suspectedOperator?.opType).toBe('Mul');
    expect(report.localization?.suspectedOperator?.nodeName).toBe('prim/mul_b');
    expect(report.localization?.suspectedOperator?.immediateParentNames).toEqual(['prim_a', 'fe_shared_3']);

    // Verify all tap steps used identical tapped SHA across providers != M1 base SHA and retained both finals
    for (const rec of [
      ...(report.localization?.coarseLadderRecords ?? []),
      ...(report.localization?.bisectionRecords ?? []),
    ]) {
      expect(rec.tappedModelSha256).not.toBe(report.m1Sha256);
      expect(rec.gpuModelSha256).toBe(rec.tappedModelSha256);
      expect(rec.wasmModelSha256).toBe(rec.tappedModelSha256);
      expect(rec.finalInvariance.gpuPrimaryBitIdentical.bitIdentical).toBe(true);
      expect(rec.finalInvariance.wasmPrimaryBitIdentical.bitIdentical).toBe(true);
    }

    const joinedLogs = logs.join('\n');
    expect(joinedLogs).toContain('ORT Web does not expose per-node EP placement in JS');
    expect(joinedLogs).toContain('parentComparisons=');
  });

  it('stops immediately when tapping changes either final output by 1 bit and skips ladder when gates pass', async () => {
    const m1Bytes = buildSyntheticMultiBranchHitnetOnnx(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength);
    const f0102Feed = new Float32Array(1 * 6 * 240 * 320).fill(0.25);

    const origDigest = crypto.subtle.digest.bind(crypto.subtle);
    vi.spyOn(crypto.subtle, 'digest').mockImplementation(async (algo, data) => {
      const view =
        data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array((data as ArrayBufferView).buffer, (data as ArrayBufferView).byteOffset, (data as ArrayBufferView).byteLength);
      if (view.byteLength === m1Bytes.byteLength && Buffer.compare(view, m1Bytes) === 0) {
        return hexToArrayBuffer(HITNET_M1_FULL_GRAPH_SHA256);
      }
      if (view.byteLength === f0102Feed.byteLength) {
        return hexToArrayBuffer(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256);
      }
      return origDigest(algo, data);
    });

    // Case A: Passing untapped -> ladder skipped
    const passWorker: ResidualDiagnosticWorkerRunner = async (req) => {
      const outputs = readGraphOutputNamesFromOnnx(new Uint8Array(req.modelBuffer)).map((name) => ({
        name,
        dataType: 'float32',
        dims: [1, 240, 320, 1],
        buffer: makeFilledFloat32Buffer(76800, 1.0, 0),
      }));
      return {
        ...makeMockSuccessBase(req),
        loadedOrtResources: [],
        foreignOrtResources: [],
        modelByteLength: m1Bytes.byteLength,
        modelSha256: HITNET_M1_FULL_GRAPH_SHA256,
        feedSha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256,
        feedTensorName: 'input',
        feedTensorShape: [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        createMs: 1,
        coldRunMs: 1,
        warmRunMs: [],
        gpuValidationErrors: [],
        gpuKernelProfileSummary: null,
        outputs,
      };
    };

    const passReport = await runHitnetResidualLocalizationDiagnostic({
      m1Bytes,
      f0102FeedFloat32: f0102Feed,
      f0126FeedFloat32: null,
      runWorker: passWorker,
      isDev: true,
    });
    expect(passReport.localization?.executed).toBe(false);
    expect(passReport.localization?.reasonSkipped).toContain('passed strict raw parity gates');

    // Case B: Tapping alters final output -> strict immediate abort
    const perturbWorker: ResidualDiagnosticWorkerRunner = async (req) => {
      const mBytes = new Uint8Array(req.modelBuffer);
      const isTapped = mBytes.byteLength !== m1Bytes.byteLength;
      const modelSha = isTapped ? nodeSha256Hex(mBytes) : HITNET_M1_FULL_GRAPH_SHA256;
      const outNames = readGraphOutputNamesFromOnnx(mBytes);
      const outputs = outNames.map((name) => {
        const isGpu = req.provider === 'webgpu';
        // Perturb secondary final output when tapped on WebGPU
        const delta =
          isGpu && name === 'reference_output_disparity'
            ? 0.02
            : isGpu && isTapped && name === 'secondary_output_disparity'
              ? 1e-4
              : 0;
        return {
          name,
          dataType: 'float32',
          dims: [1, 240, 320, 1],
          buffer: makeFilledFloat32Buffer(76800, 1.0, delta),
        };
      });
      return {
        ...makeMockSuccessBase(req),
        loadedOrtResources: [],
        foreignOrtResources: [],
        modelByteLength: mBytes.byteLength,
        modelSha256: modelSha,
        feedSha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256,
        feedTensorName: 'input',
        feedTensorShape: [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        createMs: 1,
        coldRunMs: 1,
        warmRunMs: [],
        gpuValidationErrors: [],
        gpuKernelProfileSummary: null,
        outputs,
      };
    };

    const stopReport = await runHitnetResidualLocalizationDiagnostic({
      m1Bytes,
      f0102FeedFloat32: f0102Feed,
      f0126FeedFloat32: null,
      runWorker: perturbWorker,
      isDev: true,
    });
    expect(stopReport.localization?.executed).toBe(true);
    expect(stopReport.localization?.abortedByFinalPerturbation).toBe(true);
    expect(stopReport.abortReason).toContain('changed final outputs relative to same-provider untapped control');
    expect(stopReport.localization?.coarseLadderRecords).toHaveLength(1);
    expect(stopReport.localization?.bisectionRecords).toHaveLength(0);
    expect(stopReport.localization?.suspectedOperator).toBeNull();
  });

  it('rejects any unexpected JSEP or foreign ORT resource contamination fail-closed', async () => {
    const m1Bytes = buildSyntheticMultiBranchHitnetOnnx(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength);
    const f0102Feed = new Float32Array(1 * 6 * 240 * 320).fill(0.25);

    const origDigest = crypto.subtle.digest.bind(crypto.subtle);
    vi.spyOn(crypto.subtle, 'digest').mockImplementation(async (algo, data) => {
      const view =
        data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array((data as ArrayBufferView).buffer, (data as ArrayBufferView).byteOffset, (data as ArrayBufferView).byteLength);
      if (view.byteLength === m1Bytes.byteLength && Buffer.compare(view, m1Bytes) === 0) {
        return hexToArrayBuffer(HITNET_M1_FULL_GRAPH_SHA256);
      }
      if (view.byteLength === f0102Feed.byteLength) {
        return hexToArrayBuffer(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256);
      }
      return origDigest(algo, data);
    });

    const jsepContaminatedWorker: ResidualDiagnosticWorkerRunner = async (req) => ({
      ...makeMockSuccessBase(req),
      entryModuleUrl: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.bundle.min.mjs',
      loadedOrtResources: ['https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.jsep.wasm'],
      foreignOrtResources: ['https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.jsep.wasm'],
      modelByteLength: m1Bytes.byteLength,
      modelSha256: HITNET_M1_FULL_GRAPH_SHA256,
      feedSha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256,
      feedTensorName: 'input',
      feedTensorShape: [1, 6, 240, 320],
      feedTensorDataType: 'float32',
      createMs: 1,
      coldRunMs: 1,
      warmRunMs: [],
      gpuValidationErrors: [],
      gpuKernelProfileSummary: null,
      outputs: [],
    });

    await expect(
      runHitnetResidualLocalizationDiagnostic({
        m1Bytes,
        f0102FeedFloat32: f0102Feed,
        f0126FeedFloat32: null,
        runWorker: jsepContaminatedWorker,
        isDev: true,
      })
    ).rejects.toThrow(/non-JSEP resource guard/i);
  });
});
