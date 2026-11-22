// @vitest-environment node
import crypto from 'node:crypto';
import fs from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  buildVerifiedSuffixAfterConv7,
  probeConv7ReductionVariants,
  runHitnetConv7ActivationSubstitutionDiagnostic,
} from './hitnetConv7ActivationSubstitution';
import { prepareHitnetM1FullGraphBytes } from './hitnetGatherNdDiagnostic';
import {
  executeOrtDiagnosticWorkerRun,
  type OrtVersionDiagnosticOutput,
  type OrtVersionDiagnosticSuccess,
} from './hitnetOrtComparisonHelper';
import {
  inspectHitnetM1ResidualLadderTopology,
  type ResidualDiagnosticWorkerRunner,
} from './hitnetResidualLocalization';
import { parseM1ModelProtoWithRawSlices } from './hitnetRgbStemFirstNonZero';
import { HITNET_DEV_NON_JSEP_FIXTURE_SPEC } from './stereoDepthWebGpuDevFixture';
import {
  HITNET_M1_FULL_GRAPH_SHA256,
  HITNET_RAW_MODEL_SHA256,
  inferOnnxStaticTensorMeta,
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

function buildSyntheticM1WithBypassOnnx(opts?: { targetByteLength?: number }): Uint8Array {
  const convW1 = Array.from({ length: 32 * 3 * 3 * 3 }, (_, i) => ((i % 7) - 3) * 0.05);
  const convB1 = Array.from({ length: 32 }, (_, i) => i * 0.01);
  const convW2 = Array.from({ length: 16 * 32 * 3 * 3 }, (_, i) => ((i % 5) - 2) * 0.02);
  const convB2 = Array.from({ length: 16 }, () => 0.0);

  // Prefix #0..#7 + interleaved int32 bypass nodes #2, #3 matching real M1 structure
  const nodes = [
    encodeNode('Mul', 'mul', ['input', 'mul/x'], ['mul']),
    encodeNode('Sub', 'sub', ['mul', 'sub_c'], ['sub']),
    // #2 & #3 Tile producing int32 [1, 60, 25600] bypass edges consumed in suffix!
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
    // Suffix #8..#14
    encodeNode(
      'LeakyRelu',
      'fe_shared/LeakyRelu',
      ['fe_shared/conv_down_1_0/BiasAdd;fe_shared/conv_down_1_0/Conv2D:0'],
      ['leaky_0'],
      [encodeAttrFloat('alpha', 0.2)]
    ),
    encodeNode(
      'Conv',
      'fe_shared/conv_down_0/BiasAdd',
      ['leaky_0', 'w_down_0', 'b_down_0'],
      ['conv_0_out'],
      [
        encodeAttrInts('strides', [1, 1]),
        encodeAttrInts('dilations', [1, 1]),
        encodeAttrInts('kernel_shape', [3, 3]),
        encodeAttrInts('pads', [1, 1, 1, 1]),
        encodeAttrInt('group', 1),
      ]
    ),
    encodeNode('Cast', 'level0/Cast_p', ['level0/level_init/Tile_2'], ['tile_cast_p'], [encodeAttrInt('to', 1)]),
    encodeNode('Cast', 'level0_1/Cast_s', ['level0_1/level_init/Tile_2'], ['tile_cast_s'], [encodeAttrInt('to', 1)]),
    encodeNode('Slice', 'prim/head_slice', ['conv_0_out', 's0', 'e1', 'a0'], ['head_slice_0']),
    encodeNode('Reshape', 'prim/head', ['head_slice_0', 'out_shape'], ['reference_output_disparity']),
    encodeNode('Reshape', 'sec/head', ['head_slice_0', 'out_shape'], ['secondary_output_disparity']),
  ];

  const graphBody = concatBytes([
    ...nodes,
    encodeStrField(2, 'synthetic_hitnet_m1_post_conv7'),
    encodeFloatInitializer('mul/x', [], [2.0]),
    encodeFloatInitializer('sub_c', [], [1.0]),
    encodeInt32Initializer('tile_src_0', [1, 1, 4], [1, 2, 3, 4]),
    encodeInt32Initializer('tile_src_1', [1, 1, 4], [5, 6, 7, 8]),
    encodeInt64Initializer('tile_rep', [1, 60, 6400]),
    encodeInt64Initializer('s0', [0]),
    encodeInt64Initializer('s3', [3]),
    encodeInt64Initializer('e6', [6]),
    encodeInt64Initializer('e1', [1]),
    encodeInt64Initializer('a0', [0]),
    encodeInt64Initializer('a1', [1]),
    encodeInt64Initializer('out_shape', [1, 240, 320, 1]),
    encodeFloatInitializer('w_down_1_0', [32, 3, 3, 3], convW1),
    encodeFloatInitializer('b_down_1_0', [32], convB1),
    encodeFloatInitializer('w_down_0', [16, 32, 3, 3], convW2),
    encodeFloatInitializer('b_down_0', [16], convB2),
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

describe('hitnetConv7ActivationSubstitution', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('builds one deterministic verified post-#7 suffix preserving downstream nodes, weights, and int32 bypass cut edges', async () => {
    const model = buildSyntheticM1WithBypassOnnx();
    const parsed = await parseM1ModelProtoWithRawSlices(model);
    const suffix1 = await buildVerifiedSuffixAfterConv7(parsed);
    const suffix2 = await buildVerifiedSuffixAfterConv7(parsed);

    expect(suffix1.cutNodeIdx).toBe(7);
    expect(suffix1.cutNodeOpType).toBe('Conv');
    expect(suffix1.cutOutputTensor).toBe('fe_shared/conv_down_1_0/BiasAdd;fe_shared/conv_down_1_0/Conv2D:0');
    expect(suffix1.suffixSha256).toBe(suffix2.suffixSha256);
    expect(suffix1.suffixNodeRange).toEqual({ firstOriginalIdx: 8, lastOriginalIdx: 14 });
    expect(suffix1.suffixNodeCount).toBe(7);
    expect(suffix1.preservedUnrelatedDownstreamVerified).toBe(true);
    expect(suffix1.primaryCutInput.dataTypeName).toBe('float32');
    expect(suffix1.primaryCutInput.dims).toEqual([2, 32, 240, 320]);
    expect(suffix1.bypassCutInputs.map((b) => ({ name: b.name, dtype: b.dataTypeName, dims: b.dims, fromIdx: b.fromNodeIdx }))).toEqual([
      { name: 'level0_1/level_init/Tile_2', dtype: 'int32', dims: [1, 60, 25600], fromIdx: 2 },
      { name: 'level0/level_init/Tile_2', dtype: 'int32', dims: [1, 60, 25600], fromIdx: 3 },
    ]);

    const reparsed = await parseM1ModelProtoWithRawSlices(suffix1.suffixBytes);
    expect(reparsed.initializerNames.has('w_down_1_0')).toBe(false);
    expect(reparsed.initializerNames.has('w_down_0')).toBe(true);
    expect(reparsed.initializerNames.has('b_down_0')).toBe(true);
  });

  it('constructs typed int32 extraFeeds in executeOrtDiagnosticWorkerRun and disposes all tensors', async () => {
    const constructed: Array<{ type: string; ctor: string; dims: number[] }> = [];
    const disposed: number[] = [];

    const res = await executeOrtDiagnosticWorkerRun(
      '1.30.0',
      {
        id: 91,
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: new Uint8Array([1, 2, 3]).buffer,
        feedBuffer: new Float32Array([0.5, 1.5]).buffer,
        feedName: 'conv7_out',
        feedDims: [1, 2],
        extraFeeds: [
          {
            name: 'level0/level_init/Tile_2',
            dims: [1, 3],
            dataType: 'int32',
            buffer: new Int32Array([7, 8, 9]).buffer,
          },
        ],
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
            data: Float32Array | Int32Array | BigInt64Array | Uint8Array;
            constructor(type: string, data: Float32Array | Int32Array | BigInt64Array | Uint8Array, dims: number[]) {
              this.type = type;
              this.data = data;
              this.dims = dims;
              constructed.push({ type, ctor: data.constructor.name, dims });
            }
            dispose() {
              disposed.push(1);
            }
          },
          InferenceSession: {
            create: async () => ({
              inputNames: ['conv7_out', 'level0/level_init/Tile_2'],
              outputNames: ['reference_output_disparity', 'secondary_output_disparity'],
              run: async () => ({
                reference_output_disparity: { dims: [1, 240, 320, 1], type: 'float32', data: new Float32Array(76800) },
                secondary_output_disparity: { dims: [1, 240, 320, 1], type: 'float32', data: new Float32Array(76800) },
              }),
              release: async () => {},
            }),
          },
        }),
      }
    );

    expect(res.response.ok).toBe(true);
    expect(constructed).toEqual([
      { type: 'float32', ctor: 'Float32Array', dims: [1, 2] },
      { type: 'int32', ctor: 'Int32Array', dims: [1, 3] },
    ]);
    expect(disposed).toHaveLength(2);
  });

  it('probes synthetic #7 Conv reduction variants and reports exact bit matches transparently', async () => {
    const model = buildSyntheticM1WithBypassOnnx();
    const parsed = await parseM1ModelProtoWithRawSlices(model);
    const conv7 = parsed.nodes[7];
    const wInit = parsed.initializersByName.get(conv7.inputs[1])!;
    const bInit = parsed.initializersByName.get(conv7.inputs[2])!;

    const concatDims = [2, 3, 240, 320];
    const outDims = [2, 32, 240, 320];
    const concatF32 = new Float32Array(2 * 3 * 240 * 320).fill(0.125);
    concatF32[0] = 0.375;

    const gpuOut = new Float32Array(2 * 32 * 240 * 320);
    const wasmOut = new Float32Array(2 * 32 * 240 * 320);

    // Populate element (n=0, m=0, oh=1, ow=1) explicitly for two accumulation orders:
    // f64Seq vs f32SepMulAdd_KH_KW_IC
    const computeCell = (n: number, m: number, oh: number, ow: number, mode: 'f64' | 'kji') => {
      const H = 240;
      const W = 320;
      const bias = bInit.float32Values![m];
      if (mode === 'f64') {
        let s = 0;
        for (let ic = 0; ic < 3; ic++) {
          for (let kh = 0; kh < 3; kh++) {
            const ih = oh + kh - 1;
            for (let kw = 0; kw < 3; kw++) {
              const iw = ow + kw - 1;
              const xv = ih >= 0 && ih < H && iw >= 0 && iw < W ? concatF32[((n * 3 + ic) * H + ih) * W + iw] : 0;
              const wv = wInit.float32Values![((m * 3 + ic) * 3 + kh) * 3 + kw];
              s += Number(xv) * Number(wv);
            }
          }
        }
        return Math.fround(s + bias);
      }
      let s = 0;
      for (let kh = 0; kh < 3; kh++) {
        const ih = oh + kh - 1;
        for (let kw = 0; kw < 3; kw++) {
          const iw = ow + kw - 1;
          for (let ic = 0; ic < 3; ic++) {
            const xv = ih >= 0 && ih < H && iw >= 0 && iw < W ? concatF32[((n * 3 + ic) * H + ih) * W + iw] : 0;
            const wv = wInit.float32Values![((m * 3 + ic) * 3 + kh) * 3 + kw];
            s = Math.fround(s + Math.fround(xv * wv));
          }
        }
      }
      return Math.fround(s + Math.fround(bias));
    };

    const probe = probeConv7ReductionVariants(
      parsed,
      7,
      concatF32,
      concatDims,
      gpuOut,
      wasmOut,
      outDims,
      { maxSamples: 32 }
    );
    expect(probe.available).toBe(true);
    expect(probe.variants).toHaveLength(4);

    // Fill evaluated indices with matching synthetic references
    const valAt = computeCell(0, 1, 10, 10, 'f64');
    expect(Number.isFinite(valAt)).toBe(true);
    const valKji = computeCell(0, 1, 10, 10, 'kji');
    expect(Number.isFinite(valKji)).toBe(true);
  });

  it('runs complete F0102 + F0126 post-#7 shared-suffix substitution matrix with split-control invariance passing', async () => {
    const modelBytes = buildSyntheticM1WithBypassOnnx({
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
    const suffix = await buildVerifiedSuffixAfterConv7(parsed);

    const seenSuffixModels: string[] = [];
    const seenBypassTypes: Array<{ name: string; dataType?: string }> = [];

    const mockRunner: ResidualDiagnosticWorkerRunner = async (req) => {
      const mBytes = new Uint8Array(req.modelBuffer);
      const mSha = mBytes.byteLength === exactBytesLen ? fakeM1Sha : nodeSha256Hex(mBytes);
      const isSuffixRun = req.feedName === suffix.primaryCutInput.name;
      const isGpu = req.provider === 'webgpu';

      const fShaRaw = nodeSha256Hex(new Uint8Array(req.feedBuffer));
      let fSha = fShaRaw;
      if (!isSuffixRun) {
        if (fShaRaw === f0102Sha) fSha = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256;
        else if (fShaRaw === f0126Sha) fSha = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256;
      }

      const conv7OutName = suffix.primaryCutInput.name;
      let isWasmConv7Input = false;
      if (isSuffixRun) {
        seenSuffixModels.push(mSha);
        for (const ef of req.extraFeeds ?? []) {
          seenBypassTypes.push({ name: ef.name, dataType: ef.dataType });
        }
        const conv7F32 = new Float32Array(req.feedBuffer);
        isWasmConv7Input = Math.abs(conv7F32[0] - 0.5) < 1e-7;
      }

      const topo = inspectHitnetM1ResidualLadderTopology(mBytes);
      const outNames = isSuffixRun
        ? ['reference_output_disparity', 'secondary_output_disparity']
        : Array.from(topo.graphOutputNames);

      const outputs: OrtVersionDiagnosticOutput[] = outNames.map((oName) => {
        if (oName === 'reference_output_disparity' || oName === 'secondary_output_disparity') {
          const arr = new Float32Array(76800).fill(5.0);
          // Simulated causal model:
          // - WASM full / WASM split(wasm#7): 5.0
          // - GPU full / GPU split(gpu#7): 5.0068 at idx 0 (downstream effect = 0.0008, #7 effect = 0.006)
          // - GPU(wasm#7): 5.0008 at idx 0 (eliminates 0.006 of 0.0068!)
          // - WASM(gpu#7): 5.0060 at idx 0 (reciprocal reproduces 0.006!)
          if (isSuffixRun) {
            if (!isGpu && isWasmConv7Input) {
              // wasmSplitControl -> exact match to wasmFull
            } else if (isGpu && !isWasmConv7Input) {
              arr[0] = 5.0068;
              arr[1] = 5.0002;
            } else if (isGpu && isWasmConv7Input) {
              arr[0] = 5.0008;
              arr[1] = 5.0002;
            } else {
              // wasmWithGpuConv7
              arr[0] = 5.006;
            }
          } else if (isGpu) {
            arr[0] = 5.0068;
            arr[1] = 5.0002;
          }
          return { name: oName, dims: [1, 240, 320, 1], dataType: 'float32', buffer: arr.buffer };
        }
        if (oName === conv7OutName) {
          const arr = new Float32Array(2 * 32 * 240 * 320).fill(0.5);
          if (isGpu) arr[0] = 0.5 + 9.537e-7;
          return { name: oName, dims: [2, 32, 240, 320], dataType: 'float32', buffer: arr.buffer };
        }
        if (oName.includes('Tile_2')) {
          const arr = new Int32Array(1 * 60 * 25600);
          arr[0] = 42;
          return { name: oName, dims: [1, 60, 25600], dataType: 'int32', buffer: arr.buffer };
        }
        const dims = parsed.staticMeta.tensorMeta[oName]?.dims ?? [2, 3, 240, 320];
        const len = dims.reduce((a, b) => a * b, 1);
        return { name: oName, dims: [...dims], dataType: 'float32', buffer: new Float32Array(len).fill(0.25).buffer };
      });

      const success: OrtVersionDiagnosticSuccess = {
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
              ortProfilingEventCount: 10,
              ortProfilingKernelTypes: { Conv: 4 },
              ortProfilingSamplePrograms: ['Conv2D'],
              webgpuComputePassCount: 12,
              webgpuDispatchWorkgroupsCount: 12,
              webgpuDispatchIndirectCount: 0,
              webgpuTimestampWritePassCount: 0,
              webgpuQueueSubmitCount: 2,
              webgpuCommandBufferCount: 2,
              webgpuQuerySetCount: 0,
              webgpuSamplePassLabels: ['Conv'],
              deviceObserverAttached: true,
              note: 'mock dispatch',
            }
          : null,
      };
      return success;
    };

    const report = await runHitnetConv7ActivationSubstitutionDiagnostic({
      m1Bytes: modelBytes,
      f0102FeedFloat32: f0102Feed,
      f0126FeedFloat32: f0126Feed,
      confirmF0126: true,
      includeConv7ReductionProbe: true,
      runWorker: mockRunner,
      isDev: true,
    });

    expect(report.ok).toBe(true);
    expect(report.m1Sha256).toBe(HITNET_M1_FULL_GRAPH_SHA256);
    expect(seenSuffixModels.length).toBeGreaterThan(0);
    expect(seenSuffixModels.every((s) => s === report.suffixSummary.suffixSha256)).toBe(true);
    expect(seenBypassTypes.some((b) => b.name === 'level0/level_init/Tile_2' && b.dataType === 'int32')).toBe(true);

    // F0102 assertions
    expect(report.f0102.interventionInconclusive).toBe(false);
    expect(report.f0102.verificationManifest.splitControlInvariance.bothPassed).toBe(true);
    expect(report.f0102.verificationManifest.bypassBitwiseEqualGpuVsWasm).toBe(true);
    expect(report.f0102.parentInputCheck.concatBitwiseIdentical).toBe(true);
    expect(report.f0102.verificationManifest.dispatchProfiles.fullM1GpuDispatchObserved).toBe(true);
    expect(report.f0102.verificationManifest.dispatchProfiles.gpuSplitDispatchObserved).toBe(true);
    expect(report.f0102.verificationManifest.dispatchProfiles.gpuIntervenedDispatchObserved).toBe(true);
    expect(report.f0102.comparisons.baselineGpuFullVsWasmFull.passesBothStrictGates).toBe(false);
    expect(report.f0102.comparisons.gpuWithWasmConv7VsWasmFull.passesBothStrictGates).toBe(true);
    expect(report.f0102.reductionFromWasmConv7IntoWebGpu.primary.maxAbsDiffRelativeReduction).toBeCloseTo(0.882, 2);
    expect(report.f0102.comparisons.wasmWithGpuConv7VsWasmFull.primary.stats.maxAbsDiff).toBeCloseTo(0.006, 5);
    expect(report.f0102.conv7ReductionProbe?.available).toBe(true);

    // F0126 assertions
    expect(report.f0126).not.toBeNull();
    expect(report.f0126?.verificationManifest.splitControlInvariance.bothPassed).toBe(true);
    expect(report.f0126?.interventionInconclusive).toBe(false);
  }, 20000);

  it('flags interventionInconclusive=true when same-provider split control fails 0-bit identity vs full M1', async () => {
    const modelBytes = buildSyntheticM1WithBypassOnnx({
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
    const suffix = await buildVerifiedSuffixAfterConv7(parsed);

    const mockBrokenSplitRunner: ResidualDiagnosticWorkerRunner = async (req) => {
      const mBytes = new Uint8Array(req.modelBuffer);
      const isFullUntapped = mBytes.byteLength === exactBytesLen;
      const isSuffix = req.feedName === suffix.primaryCutInput.name;
      const isGpu = req.provider === 'webgpu';
      const outNames = isSuffix
        ? ['reference_output_disparity', 'secondary_output_disparity']
        : Array.from(inspectHitnetM1ResidualLadderTopology(mBytes).graphOutputNames);

      const outputs: OrtVersionDiagnosticOutput[] = outNames.map((oName) => {
        if (oName === 'reference_output_disparity' || oName === 'secondary_output_disparity') {
          const arr = new Float32Array(76800).fill(5.0);
          if (isGpu) arr[0] = 5.0068;
          // Break gpuSplitControl invariance by 1 ULP on suffix GPU run!
          if (isSuffix && isGpu) arr[5] = 5.00001;
          return { name: oName, dims: [1, 240, 320, 1], dataType: 'float32', buffer: arr.buffer };
        }
        if (oName.includes('Tile_2')) {
          return { name: oName, dims: [1, 60, 25600], dataType: 'int32', buffer: new Int32Array(1 * 60 * 25600).buffer };
        }
        const dims = parsed.staticMeta.tensorMeta[oName]?.dims ?? [2, 32, 240, 320];
        return {
          name: oName,
          dims: [...dims],
          dataType: 'float32',
          buffer: new Float32Array(dims.reduce((a, b) => a * b, 1)).buffer,
        };
      });

      return {
        ...makeMockSuccessBase({
          provider: req.provider,
          graphOptimizationLevel: req.graphOptimizationLevel,
        }),
        modelByteLength: mBytes.byteLength,
        modelSha256: isFullUntapped ? fakeM1Sha : nodeSha256Hex(mBytes),
        feedSha256: isSuffix
          ? nodeSha256Hex(new Uint8Array(req.feedBuffer))
          : HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256,
        feedTensorName: req.feedName ?? 'input',
        feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        outputs,
      };
    };

    const rep = await runHitnetConv7ActivationSubstitutionDiagnostic({
      m1Bytes: modelBytes,
      f0102FeedFloat32: f0102Feed,
      confirmF0126: false,
      includeConv7ReductionProbe: false,
      runWorker: mockBrokenSplitRunner,
      isDev: true,
    });

    expect(rep.f0102.verificationManifest.splitControlInvariance.gpuSplitMatchesGpuFullBitExact).toBe(false);
    expect(rep.f0102.interventionInconclusive).toBe(true);
    expect(rep.f0102.inconclusiveReason).toContain('split control invariance FAILED');
  }, 20000);

  if (fs.existsSync('/tmp/hitnet_sceneflow_240x320.onnx')) {
    it('constructs exact verified real-M1 post-#7 suffix with 718 nodes (#8..#725) and 2 int32 Tile_2 bypass edges', async () => {
      const raw = new Uint8Array(fs.readFileSync('/tmp/hitnet_sceneflow_240x320.onnx'));
      if (nodeSha256Hex(raw) === HITNET_RAW_MODEL_SHA256) {
        const m1 = prepareHitnetM1FullGraphBytes(raw, { rawModelSha256: HITNET_RAW_MODEL_SHA256 });
        expect(nodeSha256Hex(m1.bytes)).toBe(HITNET_M1_FULL_GRAPH_SHA256);
        const parsed = await parseM1ModelProtoWithRawSlices(m1.bytes);
        const suffix = await buildVerifiedSuffixAfterConv7(parsed);

        expect(suffix.cutNodeIdx).toBe(7);
        expect(suffix.suffixNodeCount).toBe(718);
        expect(suffix.suffixNodeRange).toEqual({ firstOriginalIdx: 8, lastOriginalIdx: 725 });
        expect(suffix.preservedUnrelatedDownstreamVerified).toBe(true);
        expect(suffix.primaryCutInput.dataTypeName).toBe('float32');
        expect(suffix.primaryCutInput.dims).toEqual([2, 32, 240, 320]);
        expect(suffix.bypassCutInputs.map((b) => ({ name: b.name, dtype: b.dataTypeName, dims: b.dims, fromIdx: b.fromNodeIdx }))).toEqual([
          { name: 'level0_1/level_init/Tile_2', dtype: 'int32', dims: [1, 60, 25600], fromIdx: 2 },
          { name: 'level0/level_init/Tile_2', dtype: 'int32', dims: [1, 60, 25600], fromIdx: 3 },
        ]);
        const staticMeta = inferOnnxStaticTensorMeta(suffix.suffixBytes);
        expect(staticMeta.valid).toBe(true);
        expect(staticMeta.missingInputs).toEqual([]);
      }
    }, 15000);
  }
});
