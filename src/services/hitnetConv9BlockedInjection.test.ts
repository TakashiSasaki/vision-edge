// @vitest-environment node
import crypto from 'node:crypto';
import fs from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  buildDeterministicFixedSampleIndices,
  buildVerifiedPrefixThroughNode,
  computeConv2dCandidateFullTensor,
  computeFloat32UlpDistance,
  computeSingleConvElementVariant,
  evaluateConvReductionCandidatesOnFixedIndices,
  runHitnetConv9BlockedInjectionDiagnostic,
  selectAndValidatePredeclaredCandidate,
} from './hitnetConv9BlockedInjection';
import { buildVerifiedSuffixAfterConv7 } from './hitnetConv7ActivationSubstitution';
import { prepareHitnetM1FullGraphBytes } from './hitnetGatherNdDiagnostic';
import { type OrtVersionDiagnosticOutput, type OrtVersionDiagnosticSuccess } from './hitnetOrtComparisonHelper';
import {
  resolveVerifiedPostConv7ChainSpec,
} from './hitnetPostConv7TapReplay';
import {
  inspectHitnetM1ResidualLadderTopology,
  type ResidualDiagnosticWorkerRunner,
} from './hitnetResidualLocalization';
import { parseM1ModelProtoWithRawSlices } from './hitnetRgbStemFirstNonZero';
import { HITNET_DEV_NON_JSEP_FIXTURE_SPEC } from './stereoDepthWebGpuDevFixture';
import {
  HITNET_M1_FULL_GRAPH_SHA256,
  HITNET_RAW_MODEL_SHA256,
} from './stereoDepthWorkerCore';

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

function encodeAttrInt(name: string, val: number): Uint8Array {
  return encodeLenField(
    5,
    concatBytes([
      encodeStrField(1, name),
      concatBytes([encodeVarint((3 << 3) | 0), encodeSignedVarint64(val)]),
      encodeVarintField(20, 2),
    ])
  );
}

function encodeAttrFloat(name: string, val: number): Uint8Array {
  const raw = new Uint8Array(4);
  new DataView(raw.buffer).setFloat32(0, val, true);
  return encodeLenField(
    5,
    concatBytes([
      encodeStrField(1, name),
      concatBytes([encodeVarint((2 << 3) | 5), raw]),
      encodeVarintField(20, 1),
    ])
  );
}

function encodeAttrInts(name: string, vals: readonly number[]): Uint8Array {
  return encodeLenField(
    5,
    concatBytes([
      encodeStrField(1, name),
      ...vals.map((v) => concatBytes([encodeVarint((8 << 3) | 0), encodeSignedVarint64(v)])),
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

function encodeFloatInitializer(name: string, dims: readonly number[], values: readonly number[]): Uint8Array {
  const raw = new Uint8Array(values.length * 4);
  const dv = new DataView(raw.buffer);
  for (let i = 0; i < values.length; i++) dv.setFloat32(i * 4, values[i], true);
  return encodeLenField(
    5,
    concatBytes([
      ...dims.map((d) => encodeVarintField(1, d)),
      encodeVarintField(2, 1),
      encodeStrField(8, name),
      encodeLenField(9, raw),
    ])
  );
}

function encodeInt32Initializer(name: string, dims: readonly number[], values: readonly number[]): Uint8Array {
  const raw = new Uint8Array(values.length * 4);
  const dv = new DataView(raw.buffer);
  for (let i = 0; i < values.length; i++) dv.setInt32(i * 4, values[i], true);
  return encodeLenField(
    5,
    concatBytes([
      ...dims.map((d) => encodeVarintField(1, d)),
      encodeVarintField(2, 6),
      encodeStrField(8, name),
      encodeLenField(9, raw),
    ])
  );
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

function buildSyntheticM1ChainOnnx(opts?: {
  targetByteLength?: number;
  replaceSlot0WithN1Transpose?: boolean;
}): Uint8Array {
  const convW7 = Array.from({ length: 32 * 3 * 3 * 3 }, (_, i) => ((i % 7) - 3) * 0.05);
  const convB7 = Array.from({ length: 32 }, (_, i) => i * 0.01);
  const convW9 = Array.from({ length: 40 * 32 * 4 * 4 }, (_, i) => ((i % 11) - 5) * 0.031);
  const convB9 = Array.from({ length: 40 }, (_, i) => (i % 5) * 0.017 - 0.03);
  const convW11 = Array.from({ length: 40 * 40 * 3 * 3 }, (_, i) => ((i % 3) - 1) * 0.01);
  const convB11 = Array.from({ length: 40 }, () => 0.005);

  const slot0Op = opts?.replaceSlot0WithN1Transpose ? 'Transpose' : 'Reshape';
  const slot0Inputs = opts?.replaceSlot0WithN1Transpose ? ['cast_slot0_a'] : ['cast_slot0_a', 'slot0_shape'];
  const slot0Attrs = opts?.replaceSlot0WithN1Transpose ? [encodeAttrInts('perm', [0, 3, 1, 2])] : [];
  const slot1Inputs = opts?.replaceSlot0WithN1Transpose ? ['cast_slot0_b'] : ['cast_slot0_b', 'slot0_shape'];

  const nodes = [
    encodeNode('Mul', 'mul', ['input', 'mul/x'], ['mul']),
    encodeNode('Sub', 'sub', ['mul', 'sub_c'], ['sub']),
    encodeNode('Tile', 'level0_1/level_init/Tile_2', ['tile_src_0', 'tile_rep'], ['level0_1/level_init/Tile_2']),
    encodeNode('Tile', 'level0/level_init/Tile_2', ['tile_src_1', 'tile_rep'], ['level0/level_init/Tile_2']),
    encodeNode('Slice', 'hitnet_single_frame/strided_slice_1', ['sub', 's3', 'e6', 'a1'], ['slice_r']),
    encodeNode('Slice', 'hitnet_single_frame/strided_slice', ['sub', 's0', 's3', 'a1'], ['slice_l']),
    encodeNode('Concat', 'concat', ['slice_l', 'slice_r'], ['concat'], [encodeAttrInt('axis', 0)]),
    encodeNode(
      'Conv',
      'fe_shared/conv_down_1_0/BiasAdd;fe_shared/conv_down_1_0/Conv2D',
      ['concat', 'w_down_1_0', 'b_down_1_0'],
      ['fe_shared/conv_down_1_0/BiasAdd;fe_shared/conv_down_1_0/Conv2D:0'],
      [
        encodeAttrInts('strides', [1, 1]),
        encodeAttrInts('dilations', [1, 1]),
        encodeAttrInts('kernel_shape', [3, 3]),
        encodeAttrInts('pads', [1, 1, 1, 1]),
        encodeAttrInt('group', 1),
      ]
    ),
    encodeNode(
      'LeakyRelu',
      'fe_shared/LeakyRelu',
      ['fe_shared/conv_down_1_0/BiasAdd;fe_shared/conv_down_1_0/Conv2D:0'],
      ['fe_shared/LeakyRelu:0'],
      [encodeAttrFloat('alpha', 0.20000000298023224)]
    ),
    encodeNode(
      'Conv',
      'fe_shared/conv_down_0/BiasAdd;fe_shared/conv_down_0/Conv2D;fe_shared/conv_down_0/bias1',
      ['fe_shared/LeakyRelu:0', 'const_fold_opt__1272', 'b_down_0'],
      ['fe_shared/conv_down_0/BiasAdd:0'],
      [
        encodeAttrInts('strides', [2, 2]),
        encodeAttrInts('dilations', [1, 1]),
        encodeAttrInts('kernel_shape', [4, 4]),
        encodeAttrInts('pads', [1, 1, 1, 1]),
        encodeAttrInt('group', 1),
      ]
    ),
    encodeNode(
      'LeakyRelu',
      'fe_shared/LeakyRelu_1',
      ['fe_shared/conv_down_0/BiasAdd:0'],
      ['fe_shared/LeakyRelu_1:0'],
      [encodeAttrFloat('alpha', 0.20000000298023224)]
    ),
    encodeNode(
      'Conv',
      'fe_shared/conv_down_1_1/BiasAdd;fe_shared/conv_down_1_1/Conv2D;fe_shared/conv_down_1_1/bias1',
      ['fe_shared/LeakyRelu_1:0', 'const_fold_opt__1366', 'b_down_1_1'],
      ['fe_shared/conv_down_1_1/BiasAdd:0'],
      [
        encodeAttrInts('strides', [1, 1]),
        encodeAttrInts('dilations', [1, 1]),
        encodeAttrInts('kernel_shape', [3, 3]),
        encodeAttrInts('pads', [1, 1, 1, 1]),
        encodeAttrInt('group', 1),
      ]
    ),
    encodeNode('Cast', 'level0/level_init/Cast', ['level0/level_init/Tile_2'], ['cast_slot0_a'], [encodeAttrInt('to', 1)]),
    encodeNode('Cast', 'level0_1/level_init/Cast', ['level0_1/level_init/Tile_2'], ['cast_slot0_b'], [encodeAttrInt('to', 1)]),
    encodeNode(slot0Op, 'Transpose__908', slot0Inputs, ['Transpose__908:0'], slot0Attrs),
    encodeNode(slot0Op, 'Transpose__1006', slot1Inputs, ['Transpose__1006:0'], slot0Attrs),
    encodeNode('Slice', 'prim/head_slice', ['fe_shared/conv_down_1_1/BiasAdd:0', 's0', 'e1', 'a0'], ['head_slice_0']),
    encodeNode('Reshape', 'prim/head', ['head_slice_0', 'out_shape'], ['reference_output_disparity']),
    encodeNode('Reshape', 'sec/head', ['head_slice_0', 'out_shape'], ['secondary_output_disparity']),
  ];

  const graphBody = concatBytes([
    ...nodes,
    encodeStrField(2, 'synthetic_hitnet_m1_conv9_chain'),
    encodeFloatInitializer('mul/x', [], [2.0]),
    encodeFloatInitializer('sub_c', [], [1.0]),
    encodeInt32Initializer('tile_src_0', [1, 1, 4], [1, 2, 3, 4]),
    encodeInt32Initializer('tile_src_1', [1, 1, 4], [5, 6, 7, 8]),
    encodeInt64Initializer('tile_rep', [1, 60, 6400]),
    encodeInt64Initializer('slot0_shape', [1, 1, 60, 80]),
    encodeInt64Initializer('s0', [0]),
    encodeInt64Initializer('s3', [3]),
    encodeInt64Initializer('e6', [6]),
    encodeInt64Initializer('e1', [1]),
    encodeInt64Initializer('a0', [0]),
    encodeInt64Initializer('a1', [1]),
    encodeInt64Initializer('out_shape', [1, 240, 320, 1]),
    encodeFloatInitializer('w_down_1_0', [32, 3, 3, 3], convW7),
    encodeFloatInitializer('b_down_1_0', [32], convB7),
    encodeFloatInitializer('const_fold_opt__1272', [40, 32, 4, 4], convW9),
    encodeFloatInitializer('b_down_0', [40], convB9),
    encodeFloatInitializer('const_fold_opt__1366', [40, 40, 3, 3], convW11),
    encodeFloatInitializer('b_down_1_1', [40], convB11),
    encodeLenField(11, encodeValueInfoFloat('input', [1, 6, 240, 320])),
    encodeLenField(12, encodeValueInfoFloat('reference_output_disparity', [1, 240, 320, 1])),
    encodeLenField(12, encodeValueInfoFloat('secondary_output_disparity', [1, 240, 320, 1])),
  ]);

  const opsetImport = encodeLenField(8, concatBytes([encodeStrField(1, ''), encodeVarintField(2, 13)]));
  const baseModel = concatBytes([
    encodeVarintField(1, 7),
    encodeStrField(2, 'synthetic'),
    encodeLenField(7, graphBody),
    opsetImport,
  ]);

  if (!opts?.targetByteLength) return baseModel;
  const padKey = encodeStrField(1, 'pad');
  let valLen = opts.targetByteLength - (baseModel.byteLength + padKey.byteLength + 6);
  while (true) {
    const valField = encodeLenField(2, new Uint8Array(Math.max(0, valLen)).fill(0x61));
    const entry = encodeLenField(14, concatBytes([padKey, valField]));
    const total = baseModel.byteLength + entry.byteLength;
    if (total === opts.targetByteLength) return concatBytes([baseModel, entry]);
    valLen += opts.targetByteLength - total;
  }
}

function makeMockSuccessBase(req: {
  provider: 'webgpu' | 'wasm';
  graphOptimizationLevel: 'disabled' | 'basic';
}): Omit<OrtVersionDiagnosticSuccess, 'modelByteLength' | 'modelSha256' | 'feedSha256' | 'feedTensorName' | 'feedTensorShape' | 'feedTensorDataType' | 'outputs'> {
  return {
    id: 1,
    ok: true as const,
    pinnedOrtVersion: '1.30.0' as const,
    runtimeAssetFamily: 'asyncify' as const,
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
        url: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.asyncify.wasm',
        sha256: 'deadbeef',
        byteLength: 12345,
        role: 'wasm',
        via: 'WebAssembly.instantiate',
        referenceLocalDistSha256: null,
        matchesLocalDistSha256: null,
        note: 'synthetic mock',
      },
    ],
    provider: req.provider,
    configuredProviders: [req.provider],
    fallbackBehavior: 'no-fallback-explicit-error' as const,
    graphOptimizationLevel: req.graphOptimizationLevel,
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
    workerGpuAdapter: req.provider === 'webgpu' ? 'amd rdna3 mock' : null,
    sessionInputNames: ['input'],
    sessionOutputNames: ['reference_output_disparity', 'secondary_output_disparity'],
    gpuValidationErrors: req.provider === 'webgpu' ? [] : null,
    gpuKernelProfileSummary: null,
    createMs: 12,
    coldRunMs: req.provider === 'webgpu' ? 18 : 220,
    warmRunMs: req.provider === 'webgpu' ? [9] : [],
  };
}

function mockShaSubtleForSyntheticM1(
  modelBytes: Uint8Array,
  f0102Feed: Float32Array,
  f0126Feed: Float32Array
) {
  const f0102Sha = nodeSha256Hex(new Uint8Array(f0102Feed.buffer));
  const f0126Sha = nodeSha256Hex(new Uint8Array(f0126Feed.buffer));
  const exactBytesLen = modelBytes.byteLength;
  vi.spyOn(crypto.webcrypto.subtle, 'digest').mockImplementation(async (_alg, data) => {
    const u8 =
      data instanceof ArrayBuffer
        ? new Uint8Array(data)
        : new Uint8Array(
            (data as ArrayBufferView).buffer,
            (data as ArrayBufferView).byteOffset,
            (data as ArrayBufferView).byteLength
          );
    if (
      u8.byteLength === exactBytesLen &&
      u8[0] === modelBytes[0] &&
      u8[exactBytesLen - 1] === modelBytes[exactBytesLen - 1]
    ) {
      return hexToArrayBuffer(HITNET_M1_FULL_GRAPH_SHA256);
    }
    const h = nodeSha256Hex(u8);
    if (u8.byteLength === f0102Feed.byteLength && h === f0102Sha) {
      return hexToArrayBuffer(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256);
    }
    if (u8.byteLength === f0126Feed.byteLength && h === f0126Sha) {
      return hexToArrayBuffer(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256);
    }
    return hexToArrayBuffer(h);
  });
}

describe('hitnetConv9BlockedInjection', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('matches computeConv2dCandidateFullTensor 0-bit to computeSingleConvElementVariant across flat/K=64/128/256 and K=27 short control', async () => {
    const m1 = buildSyntheticM1ChainOnnx();
    const parsed = await parseM1ModelProtoWithRawSlices(m1);

    // Deterministic indices are strictly data-independent & sorted
    const idxA = buildDeterministicFixedSampleIndices(2 * 40 * 120 * 160, 512);
    const idxB = buildDeterministicFixedSampleIndices(2 * 40 * 120 * 160, 512);
    expect(idxA).toEqual(idxB);
    expect(idxA.length).toBe(512);
    for (let i = 1; i < idxA.length; i++) {
      expect(idxA[i]).toBeGreaterThan(idxA[i - 1]);
    }

    // ULP distance sanity checks (0 for +0/-0, 1 for adjacent f32)
    expect(computeFloat32UlpDistance(0.0, -0.0)).toBe(0);
    expect(computeFloat32UlpDistance(1.0, Math.fround(1.0 + Math.pow(2, -23)))).toBe(1);

    // Non-trivial float32 activations on small Conv shape verified element-by-element via fast & scalar full-tensor
    const x9 = new Float32Array(2 * 32 * 8 * 10);
    for (let i = 0; i < x9.length; i++) {
      x9[i] = Math.fround(Math.sin(i * 0.37) * 1.7 + Math.cos(i * 0.13) * 0.4);
    }
    const variants = [
      'flat_ic_kh_kw_bias_last',
      'blocked_k64_ic_kh_kw_bias_last',
      'blocked_k128_ic_kh_kw_bias_last',
      'blocked_k256_ic_kh_kw_bias_last',
    ] as const;

    for (const vId of variants) {
      const fullOut = computeConv2dCandidateFullTensor({
        parsed,
        convNodeIdx: 9,
        xData: x9,
        inputDims: [2, 32, 8, 10],
        outDims: [2, 40, 4, 5],
        candidateId: vId,
      });
      const sampleIdx = buildDeterministicFixedSampleIndices(fullOut.length, 64);
      for (const fi of sampleIdx) {
        const single = computeSingleConvElementVariant({
          xData: x9,
          wData: parsed.initializersByName.get('const_fold_opt__1272')!.float32Values!,
          bData: parsed.initializersByName.get('b_down_0')!.float32Values!,
          N: 2,
          C: 32,
          H: 8,
          W: 10,
          M: 40,
          oH: 4,
          oW: 5,
          CPerG: 32,
          kH: 4,
          kW: 4,
          group: 1,
          strideH: 2,
          strideW: 2,
          dilH: 1,
          dilW: 1,
          padTop: 1,
          padLeft: 1,
          flatIdx: fi,
          variantId: vId,
        });
        expect(single).toBe(fullOut[fi]);
      }
    }

    // #7 short control (K=27 < 64): blocked variants reduce cleanly to exact same result as flat
    const x7 = new Float32Array(2 * 3 * 6 * 8);
    for (let i = 0; i < x7.length; i++) {
      x7[i] = Math.fround(((i % 19) - 9) * 0.11);
    }
    const flat7 = computeConv2dCandidateFullTensor({
      parsed,
      convNodeIdx: 7,
      xData: x7,
      inputDims: [2, 3, 6, 8],
      outDims: [2, 32, 6, 8],
      candidateId: 'flat_ic_kh_kw_bias_last',
    });
    const k128_7 = computeConv2dCandidateFullTensor({
      parsed,
      convNodeIdx: 7,
      xData: x7,
      inputDims: [2, 3, 6, 8],
      outDims: [2, 32, 6, 8],
      candidateId: 'blocked_k128_ic_kh_kw_bias_last',
    });
    expect(k128_7).toEqual(flat7);
  });

  it('predeclared selection advances for non-0-ULP promising approximations on F0102+F0126 and stops cleanly when held-out F0126 fails or none qualify', async () => {
    const m1 = buildSyntheticM1ChainOnnx();
    const parsed = await parseM1ModelProtoWithRawSlices(m1);
    const x9 = new Float32Array(2 * 32 * 8 * 10);
    for (let i = 0; i < x9.length; i++) {
      x9[i] = Math.fround(Math.sin(i * 0.29) * 2.1);
    }
    const k128 = computeConv2dCandidateFullTensor({
      parsed,
      convNodeIdx: 9,
      xData: x9,
      inputDims: [2, 32, 8, 10],
      outDims: [2, 40, 4, 5],
      candidateId: 'blocked_k128_ic_kh_kw_bias_last',
    });
    // GPU output follows KH->KW->IC
    const gpu9 = new Float32Array(k128.length);
    for (let i = 0; i < gpu9.length; i++) {
      gpu9[i] = Math.fround(k128[i] + ((i % 7) - 3) * 1e-4);
    }
    // Simulated WASM is close (within ~1-2 ULPs / 2e-6) to K=128 (NOT 0-ULP bit-exact!)
    const wasmApproxPass = new Float32Array(k128.length);
    for (let i = 0; i < wasmApproxPass.length; i++) {
      wasmApproxPass[i] = Math.fround(k128[i] + ((i % 3) - 1) * 1e-6);
    }

    const eval102Pass = await evaluateConvReductionCandidatesOnFixedIndices({
      parsed,
      convNodeIdx: 9,
      inputF32: x9,
      inputDims: [2, 32, 8, 10],
      gpuOutF32: gpu9,
      wasmOutF32: wasmApproxPass,
      outDims: [2, 40, 4, 5],
      sampleCount: 128,
    });
    const eval126Pass = await evaluateConvReductionCandidatesOnFixedIndices({
      parsed,
      convNodeIdx: 9,
      inputF32: x9,
      inputDims: [2, 32, 8, 10],
      gpuOutF32: gpu9,
      wasmOutF32: wasmApproxPass,
      outDims: [2, 40, 4, 5],
      sampleCount: 128,
    });

    const passSel = selectAndValidatePredeclaredCandidate(eval102Pass, eval126Pass);
    expect(passSel.f0102Promising).toBe(true);
    expect(passSel.f0126HeldOutValidated).toBe(true);
    expect(passSel.advancedToInjection).toBe(true);
    expect(passSel.selectedCandidateId).toBe('blocked_k128_ic_kh_kw_bias_last');
    expect(passSel.stopReason).toBeNull();

    // Case B: F0102 succeeds, held-out F0126 diverges away from K=128 -> MUST stop without advancing
    const wasmBad126 = new Float32Array(k128.length);
    for (let i = 0; i < wasmBad126.length; i++) {
      wasmBad126[i] = Math.fround(gpu9[i] + 5e-4);
    }
    const eval126Fail = await evaluateConvReductionCandidatesOnFixedIndices({
      parsed,
      convNodeIdx: 9,
      inputF32: x9,
      inputDims: [2, 32, 8, 10],
      gpuOutF32: gpu9,
      wasmOutF32: wasmBad126,
      outDims: [2, 40, 4, 5],
      sampleCount: 128,
    });
    const failHeldOutSel = selectAndValidatePredeclaredCandidate(eval102Pass, eval126Fail);
    expect(failHeldOutSel.f0102Promising).toBe(true);
    expect(failHeldOutSel.f0126HeldOutValidated).toBe(false);
    expect(failHeldOutSel.advancedToInjection).toBe(false);
    expect(failHeldOutSel.stopReason).toContain('failed held-out F0126');

    // Case C: None qualify on F0102 -> stop immediately
    const noneSel = selectAndValidatePredeclaredCandidate(eval126Fail, eval126Fail);
    expect(noneSel.f0102Promising).toBe(false);
    expect(noneSel.advancedToInjection).toBe(false);
    expect(noneSel.selectedCandidateId).toBeNull();
  });

  it('runs Stage 1 -> Stage 2 (split controls + practical replacement + causal common-WASM-#7 controls) -> Stage 3 (D16/D32 fixed ROI)', async () => {
    const modelBytes = buildSyntheticM1ChainOnnx({
      targetByteLength: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength,
    });
    const f0102 = new Float32Array(1 * 6 * 240 * 320);
    const f0126 = new Float32Array(1 * 6 * 240 * 320);
    for (let i = 0; i < f0102.length; i++) {
      f0102[i] = ((i * 17) % 251) / 255;
      f0126[i] = ((i * 31) % 241) / 255;
    }
    mockShaSubtleForSyntheticM1(modelBytes, f0102, f0126);

    const parsed = await parseM1ModelProtoWithRawSlices(modelBytes);
    const chain = await resolveVerifiedPostConv7ChainSpec(parsed);

    const f0102RawSha = nodeSha256Hex(new Uint8Array(f0102.buffer));
    const f0126RawSha = nodeSha256Hex(new Uint8Array(f0126.buffer));
    const fixedIdx9 = buildDeterministicFixedSampleIndices(2 * 40 * 120 * 160, 512);
    const fixedIdx7 = buildDeterministicFixedSampleIndices(2 * 32 * 240 * 320, 512);

    const gpuPrefixNode8 = new Float32Array(2 * 32 * 240 * 320).fill(0.4);
    gpuPrefixNode8[12] = -0.35;
    const wasmPrefixNode8 = new Float32Array(2 * 32 * 240 * 320).fill(0.4);
    wasmPrefixNode8[12] = -0.35;
    wasmPrefixNode8[19] = 0.400001; // different #7/#8 from GPU prefix

    // Seed predictable differences ONLY at fixed sample indices on node #9 so we don't compute full 3.84M elements inside mock
    const synthOut9 = (x8: Float32Array, isGpu: boolean) => {
      const baseVal = x8[19] > 0.4000005 ? 1.2 : 1.0;
      const out = new Float32Array(2 * 40 * 120 * 160).fill(baseVal);
      for (const fi of fixedIdx9) {
        const k128 = computeSingleConvElementVariant({
          xData: x8,
          wData: parsed.initializersByName.get('const_fold_opt__1272')!.float32Values!,
          bData: parsed.initializersByName.get('b_down_0')!.float32Values!,
          N: 2,
          C: 32,
          H: 240,
          W: 320,
          M: 40,
          oH: 120,
          oW: 160,
          CPerG: 32,
          kH: 4,
          kW: 4,
          group: 1,
          strideH: 2,
          strideW: 2,
          dilH: 1,
          dilW: 1,
          padTop: 1,
          padLeft: 1,
          flatIdx: fi,
          variantId: 'blocked_k128_ic_kh_kw_bias_last',
        });
        out[fi] = isGpu
          ? Math.fround(k128 + ((fi % 5) - 2) * 2e-4)
          : k128; // matches K=128!
      }
      return out;
    };

    const synthOut7 = (xConcat: Float32Array, isGpu: boolean) => {
      const out = new Float32Array(2 * 32 * 240 * 320).fill(0.4);
      for (const fi of fixedIdx7) {
        const flat = computeSingleConvElementVariant({
          xData: xConcat,
          wData: parsed.initializersByName.get('w_down_1_0')!.float32Values!,
          bData: parsed.initializersByName.get('b_down_1_0')!.float32Values!,
          N: 2,
          C: 6,
          H: 240,
          W: 320,
          M: 32,
          oH: 240,
          oW: 320,
          CPerG: 3,
          kH: 3,
          kW: 3,
          group: 2,
          strideH: 1,
          strideW: 1,
          dilH: 1,
          dilW: 1,
          padTop: 1,
          padLeft: 1,
          flatIdx: fi,
          variantId: 'flat_ic_kh_kw_bias_last',
        });
        out[fi] = isGpu ? Math.fround(flat + 1e-6) : flat;
      }
      return out;
    };

    const concatMock = new Float32Array(2 * 6 * 240 * 320).fill(0.25);
    const gpuOut7Mock = synthOut7(concatMock, true);
    const wasmOut7Mock = synthOut7(concatMock, false);
    const gpuOut9OnGpu8 = synthOut9(gpuPrefixNode8, true);
    const gpuOut9OnWasm8 = synthOut9(wasmPrefixNode8, true);
    const wasmOut9OnWasm8 = synthOut9(wasmPrefixNode8, false);

    // Note: in practical replaced-#9 pipeline, Stage 2 evaluates K=128 on practNode8F32 -> feed[0] tells suffix what #9 input came in
    const runner: ResidualDiagnosticWorkerRunner = async (req) => {
      const mBytes = new Uint8Array(req.modelBuffer);
      const isGpu = req.provider === 'webgpu';
      const isFullUntapped = mBytes.byteLength === modelBytes.byteLength;
      const topo = inspectHitnetM1ResidualLadderTopology(mBytes);
      const outNames = Array.from(topo.graphOutputNames);

      const fShaRaw = nodeSha256Hex(new Uint8Array(req.feedBuffer));
      const resolvedFeedSha =
        isFullUntapped || req.feedName === 'input'
          ? fShaRaw === f0102RawSha
            ? HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256
            : fShaRaw === f0126RawSha
              ? HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256
              : fShaRaw
          : fShaRaw;

      if (req.enableWebGpuProfiling && isGpu) {
        return {
          ...makeMockSuccessBase({ provider: 'webgpu', graphOptimizationLevel: req.graphOptimizationLevel }),
          modelByteLength: mBytes.byteLength,
          modelSha256: isFullUntapped ? HITNET_M1_FULL_GRAPH_SHA256 : nodeSha256Hex(mBytes),
          feedSha256: resolvedFeedSha,
          feedTensorName: req.feedName ?? 'input',
          feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
          feedTensorDataType: 'float32',
          outputs: [],
          gpuKernelProfileSummary: {
            profilingRequested: true,
            timestampQuerySupported: true,
            webgpuComputePassCount: 4,
            webgpuDispatchWorkgroupsCount: 32,
            webgpuDispatchIndirectCount: 0,
            webgpuTimestampWritePassCount: 0,
            webgpuQueueSubmitCount: 1,
            webgpuCommandBufferCount: 1,
            webgpuQuerySetCount: 0,
            webgpuSamplePassLabels: ['Conv'],
            ortProfilingEventCount: 4,
            ortProfilingKernelTypes: { Conv: 1 },
            ortProfilingSamplePrograms: ['Conv'],
            deviceObserverAttached: true,
            note: 'mock dispatch',
          },
        };
      }

      if (req.allowArbitraryOutputs && req.feedName === chain.node8LeakyRelu.outputTensor) {
        const xIn = new Float32Array(req.feedBuffer);
        const out = xIn[19] > 0.4000005 ? (isGpu ? gpuOut9OnWasm8 : wasmOut9OnWasm8) : isGpu ? gpuOut9OnGpu8 : wasmOut9OnWasm8;
        return {
          ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: req.graphOptimizationLevel }),
          modelByteLength: mBytes.byteLength,
          modelSha256: nodeSha256Hex(mBytes),
          feedSha256: resolvedFeedSha,
          feedTensorName: chain.node8LeakyRelu.outputTensor,
          feedTensorShape: [2, 32, 240, 320],
          feedTensorDataType: 'float32',
          outputs: [{ name: chain.node9ConvDown0.outputTensor, dims: [2, 40, 120, 160], dataType: 'float32', buffer: out.slice().buffer }],
        };
      }

      const isPrefixModel = outNames.includes(chain.node8LeakyRelu.outputTensor) && !outNames.includes('reference_output_disparity');
      if (isPrefixModel) {
        const outputs: OrtVersionDiagnosticOutput[] = outNames.map((oName) => {
          if (oName === chain.node8LeakyRelu.outputTensor) {
            return {
              name: oName,
              dims: [2, 32, 240, 320],
              dataType: 'float32',
              buffer: (isGpu ? gpuPrefixNode8 : wasmPrefixNode8).slice().buffer,
            };
          }
          return {
            name: oName,
            dims: [1, 60, 25600],
            dataType: 'int32',
            buffer: new Int32Array(1 * 60 * 25600).fill(7).buffer,
          };
        });
        return {
          ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: req.graphOptimizationLevel }),
          modelByteLength: mBytes.byteLength,
          modelSha256: nodeSha256Hex(mBytes),
          feedSha256: resolvedFeedSha,
          feedTensorName: 'input',
          feedTensorShape: [1, 6, 240, 320],
          feedTensorDataType: 'float32',
          outputs,
        };
      }

      // Determine synthetic disparity depending on whether feed came from synthetic D16/D32, unmodified GPU/WASM, or practical #9
      let dispBase = isGpu ? (req.graphOptimizationLevel === 'basic' ? 16.08 : 16.09) : 16.0;
      if (req.feedName === chain.node9ConvDown0.outputTensor) {
        const fIn = new Float32Array(req.feedBuffer);
        // If fed K=128 output instead of raw GPU #9 output, disparity tracks closely to WASM (16.002)
        const isGpuRaw9 = Math.abs(fIn[fixedIdx9[0]] - gpuOut9OnGpu8[fixedIdx9[0]]) < 1e-9;
        dispBase = isGpuRaw9 ? 16.09 : isGpu ? 16.002 : 16.0;
      }

      const outputs: OrtVersionDiagnosticOutput[] = outNames.map((oName) => {
        if (oName === 'reference_output_disparity' || oName === 'secondary_output_disparity') {
          return {
            name: oName,
            dims: [1, 240, 320, 1],
            dataType: 'float32',
            buffer: new Float32Array(76800).fill(dispBase).buffer,
          };
        }
        if (oName.includes('Tile_2')) {
          return {
            name: oName,
            dims: [1, 60, 25600],
            dataType: 'int32',
            buffer: new Int32Array(1 * 60 * 25600).fill(7).buffer,
          };
        }
        if (oName === 'concat') {
          return { name: oName, dims: [2, 6, 240, 320], dataType: 'float32', buffer: concatMock.slice().buffer };
        }
        if (oName === chain.conv7OutputTensor) {
          return { name: oName, dims: [2, 32, 240, 320], dataType: 'float32', buffer: (isGpu ? gpuOut7Mock : wasmOut7Mock).slice().buffer };
        }
        if (oName === chain.node8LeakyRelu.outputTensor) {
          return { name: oName, dims: [2, 32, 240, 320], dataType: 'float32', buffer: wasmPrefixNode8.slice().buffer };
        }
        if (oName === chain.node9ConvDown0.outputTensor) {
          return { name: oName, dims: [2, 40, 120, 160], dataType: 'float32', buffer: (isGpu ? gpuOut9OnWasm8 : wasmOut9OnWasm8).slice().buffer };
        }
        return { name: oName, dims: [2, 40, 120, 160], dataType: 'float32', buffer: new Float32Array(2 * 40 * 120 * 160).buffer };
      });

      return {
        ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: req.graphOptimizationLevel }),
        modelByteLength: mBytes.byteLength,
        modelSha256: isFullUntapped ? HITNET_M1_FULL_GRAPH_SHA256 : nodeSha256Hex(mBytes),
        feedSha256: resolvedFeedSha,
        feedTensorName: req.feedName ?? 'input',
        feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        outputs,
      };
    };

    const rep = await runHitnetConv9BlockedInjectionDiagnostic({
      m1Bytes: modelBytes,
      f0102FeedFloat32: f0102,
      f0126FeedFloat32: f0126,
      normalizeSlot0ReshapeToTranspose: false,
      warmRunsAppBaseline: 1,
      runWorker: runner,
      isDev: true,
    });

    expect(rep.ok).toBe(true);
    expect(rep.aborted).toBe(false);
    expect(rep.n1OffProof.verified).toBe(true);
    expect(rep.n1OffProof.n1ReplacementCount).toBe(0);
    expect(rep.interventionGraphOptimizationLevel).toBe('disabled');
    expect(rep.appBaselineGraphOptimizationLevel).toBe('basic');

    // Stage 1: short #7 control + predeclared F0102 selection + held-out F0126
    expect(rep.stage1.f0102.conv7ShortControlEval.dims.K).toBe(27);
    expect(rep.stage1.f0102.conv9FullSuffixVsOneOpBitExactGpu).toBe(true);
    expect(rep.stage1.f0102.conv9FullSuffixVsOneOpBitExactWasm).toBe(true);
    expect(rep.stage1.selection.selectedCandidateId).toBe('blocked_k128_ic_kh_kw_bias_last');
    expect(rep.stage1.selection.advancedToInjection).toBe(true);
    expect(rep.stage1.f0102.provenances.fullM1GpuOptDisabled.separateProfilingPass.dispatchObserved).toBe(true);

    // Stage 2: Split controls bit-exact, practical replaced-#9 + separate causal common-WASM-#7 control + honest wall timing
    expect(rep.stage2RealFrames.executed).toBe(true);
    const s2 = rep.stage2RealFrames.f0102!;
    expect(s2.splitControls.allPassed).toBe(true);
    expect(s2.practicalReplaced9VsBaselines.vsUnmodifiedWasmFullOptDisabled.primary.stats.maxAbsDiff).toBeCloseTo(0.002, 4);
    expect(s2.practicalReplaced9VsBaselines.baselineGpuAppBasicVsWasmOptDisabled.primary.stats.maxAbsDiff).toBeCloseTo(0.08, 4);
    expect(s2.causalCommonWasm7Controls.controlScopeNote).toContain('NOT comparable');
    expect(s2.endToEndWallTiming.practicalReplaced9PipelineOptDisabled.endToEndExecuteWallMs).toBeGreaterThan(
      s2.endToEndWallTiming.practicalReplaced9PipelineOptDisabled.hostCandidate9ComputeMs
    );
    expect(s2.provenances.practicalPrefix0to8Gpu.separateProfilingPass.dispatchObserved).toBe(true);
    expect(s2.provenances.practicalSuffix10toEndGpu.separateProfilingPass.dispatchObserved).toBe(true);

    // Stage 3: D16 & D32 fixed ROI metrics
    expect(rep.stage3SyntheticD16D32.executed).toBe(true);
    expect(rep.stage3SyntheticD16D32.D16?.commonCroppedRoiPixelCount).toBe(53248);
    expect(rep.stage3SyntheticD16D32.D32?.commonCroppedRoiPixelCount).toBe(53248);
  }, 60000);

  it('aborts cleanly if per-provider split control fails bit-exactness against untapped run, and skips Stage 3', async () => {
    const modelBytes = buildSyntheticM1ChainOnnx({
      targetByteLength: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength,
    });
    const f0102 = new Float32Array(1 * 6 * 240 * 320).fill(0.2);
    const f0126 = new Float32Array(1 * 6 * 240 * 320).fill(0.7);
    mockShaSubtleForSyntheticM1(modelBytes, f0102, f0126);

    const parsed = await parseM1ModelProtoWithRawSlices(modelBytes);
    const chain = await resolveVerifiedPostConv7ChainSpec(parsed);
    const fixedIdx9 = buildDeterministicFixedSampleIndices(2 * 40 * 120 * 160, 512);
    const node8 = new Float32Array(2 * 32 * 240 * 320).fill(0.25);
    const wasm9 = new Float32Array(2 * 40 * 120 * 160).fill(0.9);
    const gpu9 = new Float32Array(2 * 40 * 120 * 160).fill(0.9);
    for (const fi of fixedIdx9) {
      const k128 = computeSingleConvElementVariant({
        xData: node8,
        wData: parsed.initializersByName.get('const_fold_opt__1272')!.float32Values!,
        bData: parsed.initializersByName.get('b_down_0')!.float32Values!,
        N: 2, C: 32, H: 240, W: 320, M: 40, oH: 120, oW: 160, CPerG: 32, kH: 4, kW: 4, group: 1,
        strideH: 2, strideW: 2, dilH: 1, dilW: 1, padTop: 1, padLeft: 1, flatIdx: fi,
        variantId: 'blocked_k128_ic_kh_kw_bias_last',
      });
      wasm9[fi] = k128;
      gpu9[fi] = Math.fround(k128 + 3e-4);
    }

    const runner: ResidualDiagnosticWorkerRunner = async (req) => {
      const mBytes = new Uint8Array(req.modelBuffer);
      const isGpu = req.provider === 'webgpu';
      const isFullUntapped = mBytes.byteLength === modelBytes.byteLength;
      const topo = inspectHitnetM1ResidualLadderTopology(mBytes);
      const outNames = Array.from(topo.graphOutputNames);
      const fShaRaw = nodeSha256Hex(new Uint8Array(req.feedBuffer));
      const f0102Sha = nodeSha256Hex(new Uint8Array(f0102.buffer));
      const f0126Sha = nodeSha256Hex(new Uint8Array(f0126.buffer));
      const resolvedFeedSha =
        req.feedName === 'input'
          ? fShaRaw === f0102Sha
            ? HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256
            : fShaRaw === f0126Sha
              ? HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256
              : fShaRaw
          : fShaRaw;

      if (req.enableWebGpuProfiling && isGpu) {
        return {
          ...makeMockSuccessBase({ provider: 'webgpu', graphOptimizationLevel: req.graphOptimizationLevel }),
          modelByteLength: mBytes.byteLength,
          modelSha256: isFullUntapped ? HITNET_M1_FULL_GRAPH_SHA256 : nodeSha256Hex(mBytes),
          feedSha256: resolvedFeedSha,
          feedTensorName: req.feedName ?? 'input',
          feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
          feedTensorDataType: 'float32',
          outputs: [],
          gpuKernelProfileSummary: {
            profilingRequested: true, timestampQuerySupported: true, webgpuComputePassCount: 2,
            webgpuDispatchWorkgroupsCount: 16, webgpuDispatchIndirectCount: 0, webgpuTimestampWritePassCount: 0,
            webgpuQueueSubmitCount: 1, webgpuCommandBufferCount: 1, webgpuQuerySetCount: 0,
            webgpuSamplePassLabels: ['Conv'], ortProfilingEventCount: 2, ortProfilingKernelTypes: { Conv: 1 },
            ortProfilingSamplePrograms: ['Conv'], deviceObserverAttached: true, note: 'mock',
          },
        };
      }
      if (req.allowArbitraryOutputs && req.feedName === chain.node8LeakyRelu.outputTensor) {
        return {
          ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: req.graphOptimizationLevel }),
          modelByteLength: mBytes.byteLength,
          modelSha256: nodeSha256Hex(mBytes),
          feedSha256: resolvedFeedSha,
          feedTensorName: chain.node8LeakyRelu.outputTensor,
          feedTensorShape: [2, 32, 240, 320],
          feedTensorDataType: 'float32',
          outputs: [{ name: chain.node9ConvDown0.outputTensor, dims: [2, 40, 120, 160], dataType: 'float32', buffer: (isGpu ? gpu9 : wasm9).slice().buffer }],
        };
      }
      if (outNames.includes(chain.node8LeakyRelu.outputTensor) && !outNames.includes('reference_output_disparity')) {
        return {
          ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: req.graphOptimizationLevel }),
          modelByteLength: mBytes.byteLength,
          modelSha256: nodeSha256Hex(mBytes),
          feedSha256: resolvedFeedSha,
          feedTensorName: 'input',
          feedTensorShape: [1, 6, 240, 320],
          feedTensorDataType: 'float32',
          outputs: outNames.map((oName) =>
            oName === chain.node8LeakyRelu.outputTensor
              ? { name: oName, dims: [2, 32, 240, 320], dataType: 'float32', buffer: node8.slice().buffer }
              : { name: oName, dims: [1, 60, 25600], dataType: 'int32', buffer: new Int32Array(1 * 60 * 25600).fill(7).buffer }
          ),
        };
      }

      const outputs: OrtVersionDiagnosticOutput[] = outNames.map((oName) => {
        if (oName === 'reference_output_disparity' || oName === 'secondary_output_disparity') {
          // Break GPU split control bit-identity when feed is suffix9 (#10..end)
          const val = req.feedName === chain.node9ConvDown0.outputTensor && isGpu ? 17.5 : isGpu ? 16.2 : 16.0;
          return { name: oName, dims: [1, 240, 320, 1], dataType: 'float32', buffer: new Float32Array(76800).fill(val).buffer };
        }
        if (oName.includes('Tile_2')) {
          return { name: oName, dims: [1, 60, 25600], dataType: 'int32', buffer: new Int32Array(1 * 60 * 25600).fill(7).buffer };
        }
        if (oName === chain.node9ConvDown0.outputTensor) {
          return { name: oName, dims: [2, 40, 120, 160], dataType: 'float32', buffer: (isGpu ? gpu9 : wasm9).slice().buffer };
        }
        const dims = parsed.staticMeta.tensorMeta[oName]?.dims ?? [2, 32, 240, 320];
        return { name: oName, dims: [...dims], dataType: 'float32', buffer: new Float32Array(dims.reduce((a, b) => a * b, 1)).fill(0.25).buffer };
      });

      return {
        ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: req.graphOptimizationLevel }),
        modelByteLength: mBytes.byteLength,
        modelSha256: isFullUntapped ? HITNET_M1_FULL_GRAPH_SHA256 : nodeSha256Hex(mBytes),
        feedSha256: resolvedFeedSha,
        feedTensorName: req.feedName ?? 'input',
        feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        outputs,
      };
    };

    const rep = await runHitnetConv9BlockedInjectionDiagnostic({
      m1Bytes: modelBytes,
      f0102FeedFloat32: f0102,
      f0126FeedFloat32: f0126,
      normalizeSlot0ReshapeToTranspose: false,
      runWorker: runner,
      isDev: true,
    });

    expect(rep.ok).toBe(false);
    expect(rep.aborted).toBe(true);
    expect(rep.abortReason).toContain('Per-provider split control failed bit-exactness');
    expect(rep.stage3SyntheticD16D32.executed).toBe(false);
  }, 25000);

  if (fs.existsSync('/tmp/hitnet_sceneflow_240x320.onnx')) {
    it('constructs verified #0..#8 prefix and #10..#725 suffix with preserved bypass edges on real canonical M1', async () => {
      const raw = new Uint8Array(fs.readFileSync('/tmp/hitnet_sceneflow_240x320.onnx'));
      if (nodeSha256Hex(raw) === HITNET_RAW_MODEL_SHA256) {
        const m1 = prepareHitnetM1FullGraphBytes(raw, { rawModelSha256: HITNET_RAW_MODEL_SHA256 });
        expect(nodeSha256Hex(m1.bytes)).toBe(HITNET_M1_FULL_GRAPH_SHA256);

        const parsed = await parseM1ModelProtoWithRawSlices(m1.bytes);
        const chain = await resolveVerifiedPostConv7ChainSpec(parsed);
        expect(chain.node8LeakyRelu.nodeIdx).toBe(8);
        expect(chain.node9ConvDown0.nodeIdx).toBe(9);

        const suffix9 = await buildVerifiedSuffixAfterConv7(parsed, { cutNodeIdx: 9 });
        expect(suffix9.suffixNodeRange).toEqual({ firstOriginalIdx: 10, lastOriginalIdx: 725 });
        expect(suffix9.primaryCutInput.name).toBe(chain.node9ConvDown0.outputTensor);
        expect(suffix9.bypassCutInputs.map((b) => b.name)).toContain(chain.node8LeakyRelu.outputTensor);

        const prefix0to8 = await buildVerifiedPrefixThroughNode(parsed, 8, [
          chain.node8LeakyRelu.outputTensor,
          ...suffix9.bypassCutInputs.map((b) => b.name),
        ]);
        expect(prefix0to8.preservedPrefixVerified).toBe(true);
        expect(prefix0to8.prefixNodeCount).toBe(9);
        expect(prefix0to8.outputSpecs.map((o) => o.name)).toEqual([
          'fe_shared/LeakyRelu',
          'level0_1/level_init/Tile_2',
          'level0/level_init/Tile_2',
        ]);
      }
    }, 20000);
  }
});
