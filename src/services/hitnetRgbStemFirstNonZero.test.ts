// @vitest-environment node
import crypto from 'node:crypto';
import fs from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  executeOrtDiagnosticWorkerRun,
  type OrtVersionDiagnosticOutput,
  type OrtVersionDiagnosticSuccess,
} from './hitnetOrtComparisonHelper';
import { prepareHitnetM1FullGraphBytes } from './hitnetGatherNdDiagnostic';
import {
  buildSingleOpCapturedInputModel,
  compareFloat32TensorsDetailed,
  computeSequentialFloat64Reference,
  parseM1ModelProtoWithRawSlices,
  resolveRgbStemSeedAndAncestorClosure,
  runHitnetRgbStemFirstNonZeroDiagnostic,
} from './hitnetRgbStemFirstNonZero';
import {
  inspectHitnetM1ResidualLadderTopology,
  type ResidualDiagnosticWorkerRunner,
} from './hitnetResidualLocalization';
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

function encodeFloatInitializer(
  name: string,
  dims: readonly number[],
  values: readonly number[],
  opts?: { externalLocation?: string }
): Uint8Array {
  if (opts?.externalLocation) {
    const extEntry = encodeLenField(
      13,
      concatBytes([encodeStrField(1, 'location'), encodeStrField(2, opts.externalLocation)])
    );
    return encodeLenField(
      5,
      concatBytes([
        ...dims.map((d) => encodeVarintField(1, d)),
        encodeVarintField(2, 1),
        encodeStrField(8, name),
        extEntry,
        encodeVarintField(14, 1),
      ])
    );
  }
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

function buildSyntheticRgbStemHitnetOnnx(opts?: {
  targetByteLength?: number;
  externalConvWeight?: boolean;
}): Uint8Array {
  // Mirrors real M1 RGB-stem structure with interleaved unrelated nodes (#2, #3)
  const convW1 = Array.from({ length: 32 * 3 * 3 * 3 }, (_, i) => ((i % 7) - 3) * 0.05);
  const convB1 = Array.from({ length: 32 }, (_, i) => i * 0.01);
  const convW2 = Array.from({ length: 40 * 32 * 4 * 4 }, (_, i) => ((i % 5) - 2) * 0.02);
  const convB2 = Array.from({ length: 40 }, () => 0.0);
  const convW3 = Array.from({ length: 40 * 40 * 3 * 3 }, (_, i) => ((i % 3) - 1) * 0.01);
  const convB3 = Array.from({ length: 40 }, () => 0.0);

  const nodes = [
    // #0 Mul
    encodeNode('Mul', 'mul', ['input', 'mul/x'], ['mul']),
    // #1 Sub
    encodeNode('Sub', 'sub', ['mul', 'sub_c'], ['sub']),
    // #2, #3 interleaved unrelated nodes (NOT in #11 closure!)
    encodeNode('Identity', 'level0_1/level_init/Tile_2', ['mul/x'], ['unrelated_2']),
    encodeNode('Identity', 'level0/level_init/Tile_2', ['sub_c'], ['unrelated_3']),
    // #4, #5 Slice
    encodeNode('Slice', 'hitnet_single_frame/strided_slice_1', ['sub', 's3', 'e6', 'a1'], ['slice_r']),
    encodeNode('Slice', 'hitnet_single_frame/strided_slice', ['sub', 's0', 's3', 'a1'], ['slice_l']),
    // #6 Concat
    encodeNode('Concat', 'concat', ['slice_l', 'slice_r'], ['concat'], [encodeAttrInt('axis', 0)]),
    // #7 Conv (conv_down_1_0)
    encodeNode(
      'Conv',
      'fe_shared/conv_down_1_0/BiasAdd;fe_shared/conv_down_1_0/Conv2D',
      ['concat', 'w_down_1_0', 'b_down_1_0'],
      ['conv_1_0_out'],
      [
        encodeAttrInts('strides', [1, 1]),
        encodeAttrInts('dilations', [1, 1]),
        encodeAttrInts('kernel_shape', [3, 3]),
        encodeAttrInts('pads', [1, 1, 1, 1]),
        encodeAttrInt('group', 1),
      ]
    ),
    // #8 LeakyRelu
    encodeNode('LeakyRelu', 'fe_shared/LeakyRelu', ['conv_1_0_out'], ['leaky_0'], [encodeAttrFloat('alpha', 0.2)]),
    // #9 Conv (conv_down_0)
    encodeNode(
      'Conv',
      'fe_shared/conv_down_0/BiasAdd;fe_shared/conv_down_0/Conv2D',
      ['leaky_0', 'w_down_0', 'b_down_0'],
      ['conv_0_out'],
      [
        encodeAttrInts('strides', [2, 2]),
        encodeAttrInts('dilations', [1, 1]),
        encodeAttrInts('kernel_shape', [4, 4]),
        encodeAttrInts('pads', [1, 1, 1, 1]),
        encodeAttrInt('group', 1),
      ]
    ),
    // #10 LeakyRelu_1
    encodeNode('LeakyRelu', 'fe_shared/LeakyRelu_1', ['conv_0_out'], ['leaky_1'], [encodeAttrFloat('alpha', 0.2)]),
    // #11 Conv (conv_down_1_1) -> Seed node
    encodeNode(
      'Conv',
      'fe_shared/conv_down_1_1/BiasAdd;fe_shared/conv_down_1_1/Conv2D',
      ['leaky_1', 'w_down_1_1', 'b_down_1_1'],
      ['fe_shared/conv_down_1_1/BiasAdd;fe_shared/conv_down_1_1/Conv2D:0'],
      [
        encodeAttrInts('strides', [1, 1]),
        encodeAttrInts('dilations', [1, 1]),
        encodeAttrInts('kernel_shape', [3, 3]),
        encodeAttrInts('pads', [1, 1, 1, 1]),
        encodeAttrInt('group', 1),
      ]
    ),
    // #12, #13 Final heads
    encodeNode(
      'Slice',
      'prim/head_slice',
      ['fe_shared/conv_down_1_1/BiasAdd;fe_shared/conv_down_1_1/Conv2D:0', 's0', 'e1', 'a0'],
      ['head_p_slice']
    ),
    encodeNode('Reshape', 'prim/head', ['head_p_slice', 'out_shape'], ['reference_output_disparity']),
    encodeNode('Reshape', 'sec/head', ['head_p_slice', 'out_shape'], ['secondary_output_disparity']),
  ];

  const graphBody = concatBytes([
    ...nodes,
    encodeStrField(2, 'synthetic_hitnet_rgb_stem_graph'),
    encodeFloatInitializer('mul/x', [], [2.0]),
    encodeFloatInitializer('sub_c', [], [1.0]),
    encodeInt64Initializer('s0', [0]),
    encodeInt64Initializer('s3', [3]),
    encodeInt64Initializer('e6', [6]),
    encodeInt64Initializer('e1', [1]),
    encodeInt64Initializer('a0', [0]),
    encodeInt64Initializer('a1', [1]),
    encodeInt64Initializer('out_shape', [1, 240, 320, 1]),
    encodeFloatInitializer(
      'w_down_1_0',
      [32, 3, 3, 3],
      convW1,
      opts?.externalConvWeight ? { externalLocation: 'weights.bin' } : undefined
    ),
    encodeFloatInitializer('b_down_1_0', [32], convB1),
    encodeFloatInitializer('w_down_0', [40, 32, 4, 4], convW2),
    encodeFloatInitializer('b_down_0', [40], convB2),
    encodeFloatInitializer('w_down_1_1', [40, 40, 3, 3], convW3),
    encodeFloatInitializer('b_down_1_1', [40], convB3),
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
}) {
  return {
    id: 1,
    ok: true as const,
    pinnedOrtVersion: '1.30.0' as const,
    runtimeAssetFamily: 'asyncify' as const,
    entryModuleUrl: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs',
    wasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
    ortVersions: { common: '1.30.0', web: '1.30.0' },
    loadedOrtResources: ['https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs'],
    foreignOrtResources: [],
    loadedAssetProvenance: [],
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
    coldRunMs: 8,
    warmRunMs: [],
  };
}

describe('hitnetRgbStemFirstNonZero', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('enumerates and verifies the exact 10-node deduplicated topologically ordered ancestor closure of conv_down_1_1 (omitting interleaved nodes)', async () => {
    const model = buildSyntheticRgbStemHitnetOnnx();
    const parsed = await parseM1ModelProtoWithRawSlices(model);
    const closure = resolveRgbStemSeedAndAncestorClosure(parsed);

    expect(closure.seedNodeIdx).toBe(11);
    expect(closure.topologicalOrderVerified).toBe(true);
    expect(closure.ancestryVerified).toBe(true);
    expect(closure.missingProducers).toEqual([]);
    expect(closure.closureNodeIndices).toEqual([0, 1, 4, 5, 6, 7, 8, 9, 10, 11]);
    expect(closure.omittedInterleavedIndices).toEqual([2, 3]);
    expect(closure.tapBatches).toHaveLength(5);
    expect(closure.tapBatches.every((b) => b.tensorNames.length <= 2)).toBe(true);
  });

  it('distinguishes BIT_IDENTICAL, SIGNED_ZERO_ONLY_BIT_DIFF (+0 vs -0), NONZERO_NUMERIC_SUBGATE, and STRICT_GATE_BREACH cleanly', () => {
    const dims = [1, 2, 2, 2];

    const a0 = new Float32Array([0.0, 1.5, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    const b0 = new Float32Array(a0);
    const cBitId = compareFloat32TensorsDetailed(a0, b0, dims);
    expect(cBitId.classification).toBe('BIT_IDENTICAL');
    expect(cBitId.bitDiffCount).toBe(0);
    expect(cBitId.signedZeroOnlyCount).toBe(0);
    expect(cBitId.nonZeroNumericDiffCount).toBe(0);

    const aNegZ = new Float32Array(a0);
    aNegZ[0] = -0.0;
    const cSignedZero = compareFloat32TensorsDetailed(aNegZ, b0, dims);
    expect(cSignedZero.classification).toBe('SIGNED_ZERO_ONLY_BIT_DIFF');
    expect(cSignedZero.bitDiffCount).toBe(1);
    expect(cSignedZero.signedZeroOnlyCount).toBe(1);
    expect(cSignedZero.nonZeroNumericDiffCount).toBe(0);
    expect(cSignedZero.firstBitDiff?.signedZeroPair).toBe(true);
    expect(cSignedZero.firstBitDiff?.candHex).toBe('0x80000000');
    expect(cSignedZero.firstBitDiff?.refHex).toBe('0x00000000');
    expect(cSignedZero.firstNonZeroNumericDiff).toBeNull();

    const aSubGate = new Float32Array(a0);
    aSubGate[5] = 5.0 + 1e-6;
    const cSub = compareFloat32TensorsDetailed(aSubGate, b0, dims);
    expect(cSub.classification).toBe('NONZERO_NUMERIC_SUBGATE');
    expect(cSub.nonZeroNumericDiffCount).toBe(1);
    expect(cSub.passesStrictGate).toBe(true);
    expect(cSub.firstNonZeroNumericDiff?.coord).toEqual([0, 1, 0, 1]);

    const aBreach = new Float32Array(a0);
    aBreach[7] = 7.01;
    const cBreach = compareFloat32TensorsDetailed(aBreach, b0, dims);
    expect(cBreach.classification).toBe('STRICT_GATE_BREACH');
    expect(cBreach.passesStrictGate).toBe(false);
  });

  it('builds a valid single-op Conv submodel, flags external initializers, and computes exact sequential float64 reference', async () => {
    const cleanModel = buildSyntheticRgbStemHitnetOnnx();
    const parsedClean = await parseM1ModelProtoWithRawSlices(cleanModel);
    const convNode = parsedClean.nodes[7];
    const sub = buildSingleOpCapturedInputModel(parsedClean, convNode);

    expect(sub.dynamicInputSpecs).toEqual([
      { name: 'concat', dims: [2, 3, 240, 320], elemType: 1 },
    ]);
    expect(sub.outputSpec.name).toBe('conv_1_0_out');
    expect(sub.outputSpec.dims).toEqual([2, 32, 240, 320]);
    expect(sub.initializerSpecs.map((x) => x.name)).toEqual(['w_down_1_0', 'b_down_1_0']);

    // Validate sequential float64 reference on a Mul op (fast full-tensor check)
    const mulNode = parsedClean.nodes[0];
    const dynIn = new Float32Array(1 * 6 * 240 * 320);
    dynIn[0] = 0.25;
    dynIn[1] = -0.5;
    const expectedMul = new Float32Array(dynIn.length);
    for (let i = 0; i < expectedMul.length; i++) expectedMul[i] = dynIn[i] * 2.0;
    const f64Mul = computeSequentialFloat64Reference(
      parsedClean,
      mulNode,
      { input: { dims: [1, 6, 240, 320], data: dynIn } },
      [1, 6, 240, 320],
      expectedMul,
      expectedMul,
      [0, 1]
    );
    expect(f64Mul.available).toBe(true);
    expect(f64Mul.mode).toBe('full-tensor');
    expect(f64Mul.maxAbsGpuVsF64).toBe(0);

    // Check external initializer blocker detection
    const extModel = buildSyntheticRgbStemHitnetOnnx({ externalConvWeight: true });
    const parsedExt = await parseM1ModelProtoWithRawSlices(extModel);
    expect(parsedExt.initializersByName.get('w_down_1_0')?.isExternal).toBe(true);
    expect(() => buildSingleOpCapturedInputModel(parsedExt, parsedExt.nodes[7])).toThrow(
      /external initializer/i
    );
  });

  it('supports allowArbitraryOutputs and extraFeeds in executeOrtDiagnosticWorkerRun and disposes extra tensors', async () => {
    const disposeCalls: number[] = [];
    const seenFeeds: string[] = [];
    const outF32 = new Float32Array([1.25, -2.5, 3.75, 4.0]);

    const res = await executeOrtDiagnosticWorkerRun(
      '1.30.0',
      {
        id: 77,
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: new Uint8Array([1, 2, 3, 4]).buffer,
        feedBuffer: new Float32Array([1, 2]).buffer,
        feedName: 'slice_l',
        feedDims: [1, 2],
        extraFeeds: [
          { name: 'slice_r', dims: [1, 2], buffer: new Float32Array([3, 4]).buffer },
        ],
        allowArbitraryOutputs: true,
        provider: 'wasm',
        graphOptimizationLevel: 'disabled',
        warmRuns: 0,
      },
      {
        getResourceEntries: () => [],
        loadOrtModule: async () => ({
          env: { versions: { common: '1.30.0', web: '1.30.0' }, wasm: {} },
          Tensor: class {
            dims: readonly number[];
            type: string;
            data: Float32Array;
            constructor(type: string, data: Float32Array, dims: number[]) {
              this.type = type;
              this.data = data;
              this.dims = dims;
            }
            dispose() {
              disposeCalls.push(1);
            }
          },
          InferenceSession: {
            create: async () => ({
              inputNames: ['slice_l', 'slice_r'],
              outputNames: ['concat'],
              run: async (feeds) => {
                seenFeeds.push(...Object.keys(feeds));
                return {
                  concat: { dims: [2, 2], type: 'float32', data: outF32 },
                };
              },
              release: async () => {},
            }),
          },
        }),
      }
    );

    expect(res.response.ok).toBe(true);
    expect(seenFeeds).toEqual(['slice_l', 'slice_r']);
    expect(disposeCalls).toHaveLength(2);
    if (res.response.ok) {
      expect(res.response.outputs).toHaveLength(1);
      expect(res.response.outputs[0].name).toBe('concat');
    }
  });

  it('runs complete closure sweep (#0..#6 bit-identical negative controls, #7 Conv first nonzero, single-op replay + F0126 confirmation)', async () => {
    const modelBytes = buildSyntheticRgbStemHitnetOnnx({
      targetByteLength: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength,
    });
    const fakeM1Sha = HITNET_M1_FULL_GRAPH_SHA256;
    const f0102Feed = new Float32Array(1 * 6 * 240 * 320).fill(0.5);
    const f0126Feed = new Float32Array(1 * 6 * 240 * 320).fill(0.25);
    const f0102Sha = nodeSha256Hex(new Uint8Array(f0102Feed.buffer));
    const f0126Sha = nodeSha256Hex(new Uint8Array(f0126Feed.buffer));
    const exactBytesLen = modelBytes.byteLength;

    vi.spyOn(crypto.webcrypto.subtle, 'digest').mockImplementation(async (_alg, data) => {
      const u8 =
        data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array((data as ArrayBufferView).buffer, (data as ArrayBufferView).byteOffset, (data as ArrayBufferView).byteLength);
      if (u8.byteLength === exactBytesLen && u8[0] === modelBytes[0]) {
        return hexToArrayBuffer(fakeM1Sha);
      }
      if (u8.byteLength === f0102Feed.byteLength) {
        const hex = nodeSha256Hex(u8);
        if (hex === f0102Sha) return hexToArrayBuffer(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256);
        if (hex === f0126Sha) return hexToArrayBuffer(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256);
      }
      return hexToArrayBuffer(nodeSha256Hex(u8));
    });

    const parsed = await parseM1ModelProtoWithRawSlices(modelBytes);
    const seenRunnerCalls: Array<{
      provider: string;
      outputs: string[];
      arbitrary: boolean;
      profiling: boolean;
    }> = [];

    const mockRunner: ResidualDiagnosticWorkerRunner = async (req) => {
      const mBytes = new Uint8Array(req.modelBuffer);
      const isUntappedExactM1 = mBytes.byteLength === exactBytesLen;
      const mSha = isUntappedExactM1 ? fakeM1Sha : nodeSha256Hex(mBytes);
      const feedU8 = new Uint8Array(req.feedBuffer);
      const rawFSha = nodeSha256Hex(feedU8);
      const fSha =
        rawFSha === f0102Sha
          ? HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256
          : rawFSha === f0126Sha
            ? HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256
            : rawFSha;

      const outNames = Array.from(inspectHitnetM1ResidualLadderTopology(mBytes).graphOutputNames);
      seenRunnerCalls.push({
        provider: req.provider,
        outputs: outNames,
        arbitrary: Boolean(req.allowArbitraryOutputs),
        profiling: Boolean(req.enableWebGpuProfiling),
      });

      const isGpu = req.provider === 'webgpu';
      const outputs: OrtVersionDiagnosticOutput[] = [];
      for (const oName of outNames) {
        if (oName === 'reference_output_disparity' || oName === 'secondary_output_disparity') {
          const arr = new Float32Array(76800).fill(1.0);
          if (isGpu) {
            arr[0] = 1.003;
            arr[10] = 1.0005;
          }
          outputs.push({ name: oName, dims: [1, 240, 320, 1], dataType: 'float32', buffer: arr.buffer });
          continue;
        }
        const meta = parsed.staticMeta.tensorMeta[oName];
        const dims = meta?.dims ?? [2, 32, 240, 320];
        const len = dims.reduce((a, b) => a * b, 1);
        const arr = new Float32Array(len).fill(0.5);
        // Negative controls #0..#6 bit-identical; #7 Conv (conv_1_0_out) & later nodes diverge!
        if (isGpu && (oName.includes('conv_1_0_out') || oName.includes('leaky') || oName.includes('conv_0_out') || oName.includes('conv_down_1_1'))) {
          arr[123] = 0.5 + 3.5e-5;
        }
        outputs.push({ name: oName, dims: [...dims], dataType: 'float32', buffer: arr.buffer });
      }

      const successRes: OrtVersionDiagnosticSuccess = {
        ...makeMockSuccessBase({
          provider: req.provider,
          graphOptimizationLevel: req.graphOptimizationLevel,
        }),
        modelByteLength: mBytes.byteLength,
        modelSha256: mSha,
        feedSha256: fSha,
        feedTensorName: req.feedName ?? 'input',
        feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        outputs,
        gpuKernelProfileSummary: req.enableWebGpuProfiling
          ? {
              profilingRequested: true,
              timestampQuerySupported: true,
              ortProfilingEventCount: 3,
              ortProfilingKernelTypes: { Conv: 1 },
              ortProfilingSamplePrograms: ['Conv2DNaive'],
              webgpuComputePassCount: 2,
              webgpuDispatchWorkgroupsCount: 2,
              webgpuDispatchIndirectCount: 0,
              webgpuTimestampWritePassCount: 0,
              webgpuQueueSubmitCount: 1,
              webgpuCommandBufferCount: 1,
              webgpuQuerySetCount: 0,
              webgpuSamplePassLabels: ['Conv'],
              deviceObserverAttached: true,
              note: 'mock dispatch observed',
            }
          : null,
      };
      return successRes;
    };

    const report = await runHitnetRgbStemFirstNonZeroDiagnostic({
      m1Bytes: modelBytes,
      f0102FeedFloat32: f0102Feed,
      f0126FeedFloat32: f0126Feed,
      confirmF0126Candidate: true,
      runWorker: mockRunner,
      isDev: true,
    });

    expect(report.ok).toBe(true);
    expect(report.abortReason).toBeNull();
    expect(report.closure.closureNodeIndices).toEqual([0, 1, 4, 5, 6, 7, 8, 9, 10, 11]);
    expect(report.negativeControlsBitIdentical.map((c) => c.nodeIdx)).toEqual([0, 1, 4, 5, 6]);
    expect(report.earliestBitDiffNode?.nodeIdx).toBe(7);
    expect(report.earliestSignedZeroOnlyNode).toBeNull();
    expect(report.earliestNonZeroNumericNode?.nodeIdx).toBe(7);
    expect(report.earliestNonZeroNumericNode?.allImmediateInputsByteEqual).toBe(true);
    expect(report.earliestNonZeroNumericNode?.comparison.stats.maxAbsDiff).toBeCloseTo(3.5e-5, 7);

    // Single-op replay verified
    expect(report.singleOpReplay?.executed).toBe(true);
    expect(report.singleOpReplay?.targetNodeIdx).toBe(7);
    expect(report.singleOpReplay?.webgpuKernelDispatchObserved).toBe(true);
    expect(report.singleOpReplay?.reproducesFullM1GpuBits).toBe(true);
    expect(report.singleOpReplay?.reproducesFullM1WasmBits).toBe(true);
    expect(report.singleOpReplay?.reproducesFullM1Diff).toBe(true);
    expect(report.singleOpReplay?.float64Reference?.available).toBe(true);

    // F0126 confirmation executed
    expect(report.f0126Confirmation?.executed).toBe(true);
    expect(report.f0126Confirmation?.candidateTap?.bitIdenticalFinals).toBe(true);
    expect(report.f0126Confirmation?.candidateTap?.parentComparison?.classification).toBe('BIT_IDENTICAL');
    expect(report.f0126Confirmation?.candidateTap?.targetComparison?.classification).toBe('NONZERO_NUMERIC_SUBGATE');
    expect(seenRunnerCalls.some((c) => c.arbitrary && c.profiling)).toBe(true);
  }, 15000);

  it('aborts cleanly if any tap perturbs final outputs', async () => {
    const modelBytes = buildSyntheticRgbStemHitnetOnnx({
      targetByteLength: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength,
    });
    const fakeM1Sha = HITNET_M1_FULL_GRAPH_SHA256;
    const f0102Feed = new Float32Array(1 * 6 * 240 * 320).fill(0.5);
    const f0102Sha = nodeSha256Hex(new Uint8Array(f0102Feed.buffer));
    const exactBytesLen = modelBytes.byteLength;

    vi.spyOn(crypto.webcrypto.subtle, 'digest').mockImplementation(async (_alg, data) => {
      const u8 =
        data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array((data as ArrayBufferView).buffer, (data as ArrayBufferView).byteOffset, (data as ArrayBufferView).byteLength);
      if (u8.byteLength === exactBytesLen && u8[0] === modelBytes[0]) {
        return hexToArrayBuffer(fakeM1Sha);
      }
      if (u8.byteLength === f0102Feed.byteLength && nodeSha256Hex(u8) === f0102Sha) {
        return hexToArrayBuffer(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256);
      }
      return hexToArrayBuffer(nodeSha256Hex(u8));
    });

    const parsed = await parseM1ModelProtoWithRawSlices(modelBytes);
    const mockPerturbedRunner: ResidualDiagnosticWorkerRunner = async (req) => {
      const mBytes = new Uint8Array(req.modelBuffer);
      const isUntapped = mBytes.byteLength === exactBytesLen;
      const outNames = Array.from(inspectHitnetM1ResidualLadderTopology(mBytes).graphOutputNames);
      const isGpu = req.provider === 'webgpu';

      const outputs = outNames.map((oName) => {
        if (oName === 'reference_output_disparity' || oName === 'secondary_output_disparity') {
          const arr = new Float32Array(76800).fill(1.0);
          if (!isUntapped && isGpu) arr[5] = 1.0001;
          return { name: oName, dims: [1, 240, 320, 1], dataType: 'float32', buffer: arr.buffer };
        }
        const dims = parsed.staticMeta.tensorMeta[oName]?.dims ?? [1, 6, 240, 320];
        const len = dims.reduce((a, b) => a * b, 1);
        return { name: oName, dims: [...dims], dataType: 'float32', buffer: new Float32Array(len).buffer };
      });

      const successRes: OrtVersionDiagnosticSuccess = {
        ...makeMockSuccessBase({
          provider: req.provider,
          graphOptimizationLevel: req.graphOptimizationLevel,
        }),
        modelByteLength: mBytes.byteLength,
        modelSha256: isUntapped ? fakeM1Sha : nodeSha256Hex(mBytes),
        feedSha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256,
        feedTensorName: 'input',
        feedTensorShape: [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        outputs,
      };
      return successRes;
    };

    const aborted = await runHitnetRgbStemFirstNonZeroDiagnostic({
      m1Bytes: modelBytes,
      f0102FeedFloat32: f0102Feed,
      runWorker: mockPerturbedRunner,
      isDev: true,
    });

    expect(aborted.ok).toBe(true);
    expect(aborted.abortReason).toContain('perturbed final outputs');
    expect(aborted.tapBatchSummaries).toHaveLength(1);
    expect(aborted.nodeObservations).toHaveLength(0);
    expect(aborted.singleOpReplay).toBeNull();
  });

  it('distinguishes earlier SIGNED_ZERO_ONLY_BIT_DIFF from later nonzero divergence and discloses parent diff on replay', async () => {
    const modelBytes = buildSyntheticRgbStemHitnetOnnx({
      targetByteLength: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength,
    });
    const fakeM1Sha = HITNET_M1_FULL_GRAPH_SHA256;
    const f0102Feed = new Float32Array(1 * 6 * 240 * 320).fill(0.5);
    const f0102Sha = nodeSha256Hex(new Uint8Array(f0102Feed.buffer));
    const exactBytesLen = modelBytes.byteLength;

    vi.spyOn(crypto.webcrypto.subtle, 'digest').mockImplementation(async (_alg, data) => {
      const u8 =
        data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array((data as ArrayBufferView).buffer, (data as ArrayBufferView).byteOffset, (data as ArrayBufferView).byteLength);
      if (u8.byteLength === exactBytesLen && u8[0] === modelBytes[0]) {
        return hexToArrayBuffer(fakeM1Sha);
      }
      if (u8.byteLength === f0102Feed.byteLength && nodeSha256Hex(u8) === f0102Sha) {
        return hexToArrayBuffer(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256);
      }
      return hexToArrayBuffer(nodeSha256Hex(u8));
    });

    const parsed = await parseM1ModelProtoWithRawSlices(modelBytes);
    const mockSignedZeroRunner: ResidualDiagnosticWorkerRunner = async (req) => {
      const mBytes = new Uint8Array(req.modelBuffer);
      const isUntapped = mBytes.byteLength === exactBytesLen;
      const outNames = Array.from(inspectHitnetM1ResidualLadderTopology(mBytes).graphOutputNames);
      const isGpu = req.provider === 'webgpu';

      const outputs: OrtVersionDiagnosticOutput[] = outNames.map((oName) => {
        if (oName === 'reference_output_disparity' || oName === 'secondary_output_disparity') {
          const arr = new Float32Array(76800).fill(1.0);
          if (isGpu) arr[0] = 1.003;
          return { name: oName, dims: [1, 240, 320, 1], dataType: 'float32', buffer: arr.buffer };
        }
        const dims = parsed.staticMeta.tensorMeta[oName]?.dims ?? [1, 6, 240, 320];
        const len = dims.reduce((a, b) => a * b, 1);
        const arr = new Float32Array(len).fill(0.0);
        if (isGpu && oName === 'mul') {
          // Signed zero bit diff at #0 Mul (+0 vs -0)
          arr[2] = -0.0;
        } else if (isGpu && (oName === 'sub' || oName.startsWith('slice') || oName === 'concat')) {
          arr[2] = 0.0;
        } else if (isGpu && oName === 'sub_nonexistent') {
          arr[2] = 0.0;
        }
        if (oName === 'sub') {
          // Make #1 Sub numerically diverge right after #0 Mul's signed-zero bit diff
          arr.fill(0.25);
          if (isGpu) arr[2] = 0.25001;
        }
        return { name: oName, dims: [...dims], dataType: 'float32', buffer: arr.buffer };
      });

      return {
        ...makeMockSuccessBase({
          provider: req.provider,
          graphOptimizationLevel: req.graphOptimizationLevel,
        }),
        modelByteLength: mBytes.byteLength,
        modelSha256: isUntapped ? fakeM1Sha : nodeSha256Hex(mBytes),
        feedSha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256,
        feedTensorName: 'input',
        feedTensorShape: [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        outputs,
      };
    };

    const rep = await runHitnetRgbStemFirstNonZeroDiagnostic({
      m1Bytes: modelBytes,
      f0102FeedFloat32: f0102Feed,
      runWorker: mockSignedZeroRunner,
      isDev: true,
    });

    expect(rep.earliestBitDiffNode?.nodeIdx).toBe(0);
    expect(rep.earliestSignedZeroOnlyNode?.nodeIdx).toBe(0);
    expect(rep.earliestNonZeroNumericNode?.nodeIdx).toBe(1);
    expect(rep.earliestNonZeroNumericNode?.allImmediateInputsByteEqual).toBe(false);
    expect(rep.singleOpReplay?.executed).toBe(false);
    expect(rep.singleOpReplay?.blockerReason).toContain('SIGNED_ZERO_ONLY_BIT_DIFF');
  });

  if (fs.existsSync('/tmp/hitnet_sceneflow_240x320.onnx')) {
    it('verifies exact 10-node #0..#11 ancestor closure directly against local raw HITNet M1 weights', async () => {
      const raw = new Uint8Array(fs.readFileSync('/tmp/hitnet_sceneflow_240x320.onnx'));
      if (nodeSha256Hex(raw) === HITNET_RAW_MODEL_SHA256) {
        const m1 = prepareHitnetM1FullGraphBytes(raw, { rawModelSha256: HITNET_RAW_MODEL_SHA256 });
        expect(nodeSha256Hex(m1.bytes)).toBe(HITNET_M1_FULL_GRAPH_SHA256);
        const parsed = await parseM1ModelProtoWithRawSlices(m1.bytes);
        const closure = resolveRgbStemSeedAndAncestorClosure(parsed);
        expect(closure.closureNodeIndices).toEqual([0, 1, 4, 5, 6, 7, 8, 9, 10, 11]);
        expect(closure.omittedInterleavedIndices).toEqual([2, 3]);
        expect(closure.topologicalOrderVerified).toBe(true);
        expect(closure.ancestryVerified).toBe(true);
      }
    });
  }
});
