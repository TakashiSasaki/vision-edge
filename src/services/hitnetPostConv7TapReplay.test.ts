// @vitest-environment node
import crypto from 'node:crypto';
import fs from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { prepareHitnetM1FullGraphBytes } from './hitnetGatherNdDiagnostic';
import {
  type OrtVersionDiagnosticOutput,
} from './hitnetOrtComparisonHelper';
import {
  probeConvOpReductionVariantsExtended,
  probeLeakyReluNumericVariants,
  resolveVerifiedPostConv7ChainSpec,
  runHitnetPostConv7TapReplayDiagnostic,
  verifyCanonicalM1NoN1Replacements,
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
  const convW1 = Array.from({ length: 32 * 3 * 3 * 3 }, (_, i) => ((i % 7) - 3) * 0.05);
  const convB1 = Array.from({ length: 32 }, (_, i) => i * 0.01);
  // Keep stride-2 4x4 kernel & 3x3 conv_down_1_1 lightweight
  const convW9 = Array.from({ length: 40 * 32 * 4 * 4 }, (_, i) => ((i % 5) - 2) * 0.015);
  const convB9 = Array.from({ length: 40 }, (_, i) => (i % 4) * 0.02 - 0.03);
  const convW11 = Array.from({ length: 40 * 40 * 3 * 3 }, (_, i) => ((i % 3) - 1) * 0.01);
  const convB11 = Array.from({ length: 40 }, () => 0.005);

  const slot0Op = opts?.replaceSlot0WithN1Transpose ? 'Transpose' : 'Reshape';
  const slot0Inputs = opts?.replaceSlot0WithN1Transpose ? ['cast_slot0_a'] : ['cast_slot0_a', 'slot0_shape'];
  const slot0Attrs = opts?.replaceSlot0WithN1Transpose ? [encodeAttrInts('perm', [0, 3, 1, 2])] : [];
  const slot1Inputs = opts?.replaceSlot0WithN1Transpose ? ['cast_slot0_b'] : ['cast_slot0_b', 'slot0_shape'];

  const nodes = [
    encodeNode('Mul', 'mul', ['input', 'mul/x'], ['mul']),
    encodeNode('Sub', 'sub', ['mul', 'sub_c'], ['sub']),
    // #2 & #3 Tile producing int32 [1, 60, 25600] bypass edges consumed in suffix
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
    // #8 LeakyRelu (fe_shared/LeakyRelu) -> [2, 32, 240, 320]
    encodeNode(
      'LeakyRelu',
      'fe_shared/LeakyRelu',
      ['fe_shared/conv_down_1_0/BiasAdd;fe_shared/conv_down_1_0/Conv2D:0'],
      ['fe_shared/LeakyRelu:0'],
      [encodeAttrFloat('alpha', 0.20000000298023224)]
    ),
    // #9 stride-2 Conv (fe_shared/conv_down_0/.../bias1) -> [2, 40, 120, 160]
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
    // #10 LeakyRelu_1 -> [2, 40, 120, 160]
    encodeNode(
      'LeakyRelu',
      'fe_shared/LeakyRelu_1',
      ['fe_shared/conv_down_0/BiasAdd:0'],
      ['fe_shared/LeakyRelu_1:0'],
      [encodeAttrFloat('alpha', 0.20000000298023224)]
    ),
    // #11 Conv conv_down_1_1 -> [2, 40, 120, 160]
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
    // Original M1 Reshape nodes that N1 would replace if enabled (kept as Reshape in canonical M1)
    encodeNode('Cast', 'level0/level_init/Cast', ['level0/level_init/Tile_2'], ['cast_slot0_a'], [encodeAttrInt('to', 1)]),
    encodeNode('Cast', 'level0_1/level_init/Cast', ['level0_1/level_init/Tile_2'], ['cast_slot0_b'], [encodeAttrInt('to', 1)]),
    encodeNode(slot0Op, 'Transpose__908', slot0Inputs, ['Transpose__908:0'], slot0Attrs),
    encodeNode(slot0Op, 'Transpose__1006', slot1Inputs, ['Transpose__1006:0'], slot0Attrs),
    // Final head nodes producing [1, 240, 320, 1]
    encodeNode('Slice', 'prim/head_slice', ['fe_shared/conv_down_1_1/BiasAdd:0', 's0', 'e1', 'a0'], ['head_slice_0']),
    encodeNode('Reshape', 'prim/head', ['head_slice_0', 'out_shape'], ['reference_output_disparity']),
    encodeNode('Reshape', 'sec/head', ['head_slice_0', 'out_shape'], ['secondary_output_disparity']),
  ];

  const graphBody = concatBytes([
    ...nodes,
    encodeStrField(2, 'synthetic_hitnet_m1_post_conv7_chain'),
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
    encodeFloatInitializer('w_down_1_0', [32, 3, 3, 3], convW1),
    encodeFloatInitializer('b_down_1_0', [32], convB1),
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
    createMs: 10,
    coldRunMs: 6,
    warmRunMs: [],
  };
}

function mockShaSubtleForSyntheticM1(
  modelBytes: Uint8Array,
  f0102Feed: Float32Array,
  f0126Feed?: Float32Array
) {
  const f0102Sha = nodeSha256Hex(new Uint8Array(f0102Feed.buffer));
  const f0126Sha = f0126Feed ? nodeSha256Hex(new Uint8Array(f0126Feed.buffer)) : null;
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
    if (u8.byteLength === exactBytesLen && u8[0] === modelBytes[0] && u8[exactBytesLen - 1] === modelBytes[exactBytesLen - 1]) {
      return hexToArrayBuffer(HITNET_M1_FULL_GRAPH_SHA256);
    }
    const h = nodeSha256Hex(u8);
    if (u8.byteLength === f0102Feed.byteLength && h === f0102Sha) {
      return hexToArrayBuffer(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256);
    }
    if (f0126Feed && f0126Sha && u8.byteLength === f0126Feed.byteLength && h === f0126Sha) {
      return hexToArrayBuffer(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256);
    }
    return hexToArrayBuffer(h);
  });
}

describe('hitnetPostConv7TapReplay', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('rejects N1 enabled explicitly or detected N1 Transpose replacements inside canonical M1', async () => {
    const m1 = buildSyntheticM1ChainOnnx({
      targetByteLength: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength,
      replaceSlot0WithN1Transpose: false,
    });
    const parsedM1 = await parseM1ModelProtoWithRawSlices(m1);
    const proofClean = verifyCanonicalM1NoN1Replacements(parsedM1, m1, HITNET_M1_FULL_GRAPH_SHA256, false);
    expect(proofClean.verified).toBe(true);
    expect(proofClean.n1ExplicitlyOff).toBe(true);
    expect(proofClean.n1ReplacementCount).toBe(0);
    expect(proofClean.verifiedOriginalSlot0ReshapeNodes).toEqual(['Transpose__908@#14', 'Transpose__1006@#15']);

    // If slot0 nodes were replaced with N1 Transpose, N1 guard fails closed
    const n1Model = buildSyntheticM1ChainOnnx({
      targetByteLength: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength,
      replaceSlot0WithN1Transpose: true,
    });
    const parsedN1 = await parseM1ModelProtoWithRawSlices(n1Model);
    const proofBad = verifyCanonicalM1NoN1Replacements(parsedN1, n1Model, HITNET_M1_FULL_GRAPH_SHA256, false);
    expect(proofBad.verified).toBe(false);
    expect(proofBad.n1ReplacementCount).toBe(2);
    expect(proofBad.detectedN1TransposeNodes).toEqual([
      'Transpose__908@#14(Transpose)',
      'Transpose__1006@#15(Transpose)',
    ]);

    const feed = new Float32Array(1 * 6 * 240 * 320).fill(0.25);
    await expect(
      runHitnetPostConv7TapReplayDiagnostic({
        m1Bytes: m1,
        f0102FeedFloat32: feed,
        // @ts-expect-error testing runtime rejection if caller attempts to turn N1 on
        normalizeSlot0ReshapeToTranspose: true,
        runWorker: vi.fn(),
        isDev: true,
      })
    ).rejects.toThrow(/MUST be explicitly OFF/i);
  });

  it('localizes #8 LeakyRelu and stops BEFORE #10/#11 when #8 differs, verifying 1-op replay & LeakyRelu probes', async () => {
    const modelBytes = buildSyntheticM1ChainOnnx({
      targetByteLength: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength,
    });
    const f0102 = new Float32Array(1 * 6 * 240 * 320).fill(0.35);
    mockShaSubtleForSyntheticM1(modelBytes, f0102);

    const parsed = await parseM1ModelProtoWithRawSlices(modelBytes);
    const chain = await resolveVerifiedPostConv7ChainSpec(parsed);

    const seenTap1011 = { executed: false };
    const runner: ResidualDiagnosticWorkerRunner = async (req) => {
      const mBytes = new Uint8Array(req.modelBuffer);
      const isGpu = req.provider === 'webgpu';
      const isFullUntapped = mBytes.byteLength === modelBytes.byteLength;
      const isSuffix = req.feedName === chain.conv7OutputTensor;
      const isOneOp = req.allowArbitraryOutputs === true;
      const topo = inspectHitnetM1ResidualLadderTopology(mBytes);
      const outNames = Array.from(topo.graphOutputNames);

      if (outNames.includes(chain.node10LeakyRelu1!.outputTensor)) {
        seenTap1011.executed = true;
      }

      if (req.enableWebGpuProfiling && isGpu) {
        return {
          ...makeMockSuccessBase({ provider: 'webgpu', graphOptimizationLevel: 'disabled' }),
          modelByteLength: mBytes.byteLength,
          modelSha256: nodeSha256Hex(mBytes),
          feedSha256: nodeSha256Hex(new Uint8Array(req.feedBuffer)),
          feedTensorName: req.feedName ?? 'input',
          feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
          feedTensorDataType: 'float32',
          outputs: [],
          gpuKernelProfileSummary: {
            profilingRequested: true,
            timestampQuerySupported: true,
            webgpuComputePassCount: 3,
            webgpuDispatchWorkgroupsCount: 16,
            webgpuDispatchIndirectCount: 0,
            webgpuTimestampWritePassCount: 0,
            webgpuQueueSubmitCount: 1,
            webgpuCommandBufferCount: 1,
            webgpuQuerySetCount: 0,
            webgpuSamplePassLabels: ['LeakyRelu'],
            ortProfilingEventCount: 3,
            ortProfilingKernelTypes: { LeakyRelu: 1 },
            ortProfilingSamplePrograms: ['LeakyRelu'],
            deviceObserverAttached: true,
            note: 'mock dispatch',
          },
        };
      }

      if (isOneOp) {
        const inp = new Float32Array(req.feedBuffer);
        const out = new Float32Array(inp.length);
        const alphaF32 = Math.fround(0.20000000298023224);
        for (let i = 0; i < inp.length; i++) {
          const x = inp[i];
          if (isGpu) {
            out[i] = x >= 0 ? Math.fround(x) : Math.fround(x * alphaF32) + (i === 0 ? 1e-4 : 0);
          } else {
            out[i] = x >= 0 ? Math.fround(x) : Math.fround(Number(x) * 0.20000000298023224);
          }
        }
        return {
          ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: 'disabled' }),
          modelByteLength: mBytes.byteLength,
          modelSha256: nodeSha256Hex(mBytes),
          feedSha256: nodeSha256Hex(new Uint8Array(req.feedBuffer)),
          feedTensorName: req.feedName ?? chain.conv7OutputTensor,
          feedTensorShape: req.feedDims ?? [2, 32, 240, 320],
          feedTensorDataType: 'float32',
          outputs: [{ name: chain.node8LeakyRelu.outputTensor, dims: [2, 32, 240, 320], dataType: 'float32', buffer: out.buffer }],
        };
      }

      const outputs: OrtVersionDiagnosticOutput[] = outNames.map((oName) => {
        if (oName === 'reference_output_disparity' || oName === 'secondary_output_disparity') {
          const arr = new Float32Array(76800).fill(isGpu ? 5.01 : 5.0);
          return { name: oName, dims: [1, 240, 320, 1], dataType: 'float32', buffer: arr.buffer };
        }
        if (oName.includes('Tile_2')) {
          return { name: oName, dims: [1, 60, 25600], dataType: 'int32', buffer: new Int32Array(1 * 60 * 25600).fill(7).buffer };
        }
        if (oName === chain.conv7OutputTensor) {
          const arr = new Float32Array(2 * 32 * 240 * 320).fill(-0.5);
          if (isGpu) arr[0] = -0.500001;
          return { name: oName, dims: [2, 32, 240, 320], dataType: 'float32', buffer: arr.buffer };
        }
        if (oName === chain.node8LeakyRelu.outputTensor) {
          const arr = new Float32Array(2 * 32 * 240 * 320).fill(Math.fround(-0.5 * 0.20000000298023224));
          if (isGpu) arr[0] = Math.fround(-0.5 * Math.fround(0.20000000298023224)) + 1e-4;
          return { name: oName, dims: [2, 32, 240, 320], dataType: 'float32', buffer: arr.buffer };
        }
        const dims = parsed.staticMeta.tensorMeta[oName]?.dims ?? [2, 40, 120, 160];
        return {
          name: oName,
          dims: [...dims],
          dataType: 'float32',
          buffer: new Float32Array(dims.reduce((a, b) => a * b, 1)).fill(1.0).buffer,
        };
      });

      return {
        ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: 'disabled' }),
        modelByteLength: mBytes.byteLength,
        modelSha256: isFullUntapped ? HITNET_M1_FULL_GRAPH_SHA256 : nodeSha256Hex(mBytes),
        feedSha256: isSuffix
          ? nodeSha256Hex(new Uint8Array(req.feedBuffer))
          : HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256,
        feedTensorName: req.feedName ?? 'input',
        feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        outputs,
      };
    };

    const rep = await runHitnetPostConv7TapReplayDiagnostic({
      m1Bytes: modelBytes,
      f0102FeedFloat32: f0102,
      confirmF0126: false,
      normalizeSlot0ReshapeToTranspose: false,
      runWorker: runner,
      isDev: true,
    });

    expect(rep.ok).toBe(true);
    expect(rep.n1OffProof.verified).toBe(true);
    expect(rep.n1OffProof.f0102FullGpuPreSessionModelSha256).toBe(HITNET_M1_FULL_GRAPH_SHA256);
    expect(rep.n1OffProof.f0102FullWasmPreSessionModelSha256).toBe(HITNET_M1_FULL_GRAPH_SHA256);
    expect(rep.f0102.localizedFirstDifferingOp.stage).toBe('node8-leakyrelu');
    expect(seenTap1011.executed).toBe(false);
    expect(rep.f0102.paired1011Tap.executed).toBe(false);
    expect(rep.f0102.isolatedReplay?.executed).toBe(true);
    expect(rep.f0102.isolatedReplay?.webgpuDispatchObserved).toBe(true);
    expect(rep.f0102.isolatedReplay?.reproducesFullSuffixDiff).toBe(true);
    expect(rep.f0102.isolatedReplay?.leakyReluVariants?.available).toBe(true);
  }, 25000);

  it('localizes #9 stride-2 Conv when #8 is exact, stops BEFORE #10/#11, verifies 1-op replay & bounded F0126 confirmation', async () => {
    const modelBytes = buildSyntheticM1ChainOnnx({
      targetByteLength: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength,
    });
    const f0102 = new Float32Array(1 * 6 * 240 * 320).fill(0.35);
    const f0126 = new Float32Array(1 * 6 * 240 * 320).fill(0.65);
    mockShaSubtleForSyntheticM1(modelBytes, f0102, f0126);

    const parsed = await parseM1ModelProtoWithRawSlices(modelBytes);
    const chain = await resolveVerifiedPostConv7ChainSpec(parsed);

    const seenTap1011 = { executed: false };
    const runner: ResidualDiagnosticWorkerRunner = async (req) => {
      const mBytes = new Uint8Array(req.modelBuffer);
      const isGpu = req.provider === 'webgpu';
      const isFullUntapped = mBytes.byteLength === modelBytes.byteLength;
      const isSuffix = req.feedName === chain.conv7OutputTensor;
      const isOneOp = req.allowArbitraryOutputs === true;
      const topo = inspectHitnetM1ResidualLadderTopology(mBytes);
      const outNames = Array.from(topo.graphOutputNames);

      if (outNames.includes(chain.node10LeakyRelu1!.outputTensor)) {
        seenTap1011.executed = true;
      }

      if (req.enableWebGpuProfiling && isGpu) {
        return {
          ...makeMockSuccessBase({ provider: 'webgpu', graphOptimizationLevel: 'disabled' }),
          modelByteLength: mBytes.byteLength,
          modelSha256: nodeSha256Hex(mBytes),
          feedSha256: nodeSha256Hex(new Uint8Array(req.feedBuffer)),
          feedTensorName: req.feedName ?? 'input',
          feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
          feedTensorDataType: 'float32',
          outputs: [],
          gpuKernelProfileSummary: {
            profilingRequested: true,
            timestampQuerySupported: true,
            webgpuComputePassCount: 4,
            webgpuDispatchWorkgroupsCount: 24,
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

      if (isOneOp) {
        const outLen = 2 * 40 * 120 * 160;
        const out = new Float32Array(outLen).fill(1.25);
        if (isGpu) out[3] = 1.25002;
        return {
          ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: 'disabled' }),
          modelByteLength: mBytes.byteLength,
          modelSha256: nodeSha256Hex(mBytes),
          feedSha256: nodeSha256Hex(new Uint8Array(req.feedBuffer)),
          feedTensorName: req.feedName ?? chain.node8LeakyRelu.outputTensor,
          feedTensorShape: req.feedDims ?? [2, 32, 240, 320],
          feedTensorDataType: 'float32',
          outputs: [{ name: chain.node9ConvDown0.outputTensor, dims: [2, 40, 120, 160], dataType: 'float32', buffer: out.buffer }],
        };
      }

      const feedShaNow = nodeSha256Hex(new Uint8Array(req.feedBuffer));
      const f0102RawSha = nodeSha256Hex(new Uint8Array(f0102.buffer));
      const resolvedFeedSha = isSuffix
        ? feedShaNow
        : feedShaNow === f0102RawSha
          ? HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256
          : HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256;

      const outputs: OrtVersionDiagnosticOutput[] = outNames.map((oName) => {
        if (oName === 'reference_output_disparity' || oName === 'secondary_output_disparity') {
          const arr = new Float32Array(76800).fill(isGpu ? 5.01 : 5.0);
          return { name: oName, dims: [1, 240, 320, 1], dataType: 'float32', buffer: arr.buffer };
        }
        if (oName.includes('Tile_2')) {
          return { name: oName, dims: [1, 60, 25600], dataType: 'int32', buffer: new Int32Array(1 * 60 * 25600).fill(7).buffer };
        }
        if (oName === chain.conv7OutputTensor) {
          return { name: oName, dims: [2, 32, 240, 320], dataType: 'float32', buffer: new Float32Array(2 * 32 * 240 * 320).fill(0.5).buffer };
        }
        if (oName === chain.node8LeakyRelu.outputTensor) {
          // #8 is 0-bit exact between GPU & WASM
          return { name: oName, dims: [2, 32, 240, 320], dataType: 'float32', buffer: new Float32Array(2 * 32 * 240 * 320).fill(0.5).buffer };
        }
        if (oName === chain.node9ConvDown0.outputTensor) {
          const arr = new Float32Array(2 * 40 * 120 * 160).fill(1.25);
          if (isGpu) arr[3] = 1.25002;
          return { name: oName, dims: [2, 40, 120, 160], dataType: 'float32', buffer: arr.buffer };
        }
        const dims = parsed.staticMeta.tensorMeta[oName]?.dims ?? [2, 40, 120, 160];
        return {
          name: oName,
          dims: [...dims],
          dataType: 'float32',
          buffer: new Float32Array(dims.reduce((a, b) => a * b, 1)).fill(1.0).buffer,
        };
      });

      return {
        ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: 'disabled' }),
        modelByteLength: mBytes.byteLength,
        modelSha256: isFullUntapped ? HITNET_M1_FULL_GRAPH_SHA256 : nodeSha256Hex(mBytes),
        feedSha256: resolvedFeedSha,
        feedTensorName: req.feedName ?? 'input',
        feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        outputs,
      };
    };

    const rep = await runHitnetPostConv7TapReplayDiagnostic({
      m1Bytes: modelBytes,
      f0102FeedFloat32: f0102,
      f0126FeedFloat32: f0126,
      confirmF0126: true,
      normalizeSlot0ReshapeToTranspose: false,
      runWorker: runner,
      isDev: true,
    });

    expect(rep.ok).toBe(true);
    expect(seenTap1011.executed).toBe(false);
    expect(rep.f0102.localizedFirstDifferingOp.stage).toBe('node9-conv');
    expect(rep.f0102.paired89Tap.node8?.comparison.bitDiffCount).toBe(0);
    expect(rep.f0102.paired89Tap.node9?.comparison.bitDiffCount).toBe(1);
    expect(rep.f0102.paired1011Tap.executed).toBe(false);
    expect(rep.f0102.isolatedReplay?.reproducesFullSuffixDiff).toBe(true);
    expect(rep.f0102.isolatedReplay?.convReductionVariants?.available).toBe(true);
    expect(rep.f0126).not.toBeNull();
    expect(rep.f0126?.localizedFirstDifferingOp.stage).toBe('node9-conv');
  }, 25000);

  it('executes #10/#11 paired tap IF AND ONLY IF both #8 AND #9 are 0-bit-exact, localizing #11 Conv', async () => {
    const modelBytes = buildSyntheticM1ChainOnnx({
      targetByteLength: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength,
    });
    const f0102 = new Float32Array(1 * 6 * 240 * 320).fill(0.45);
    mockShaSubtleForSyntheticM1(modelBytes, f0102);

    const parsed = await parseM1ModelProtoWithRawSlices(modelBytes);
    const chain = await resolveVerifiedPostConv7ChainSpec(parsed);

    const runner: ResidualDiagnosticWorkerRunner = async (req) => {
      const mBytes = new Uint8Array(req.modelBuffer);
      const isGpu = req.provider === 'webgpu';
      const isFullUntapped = mBytes.byteLength === modelBytes.byteLength;
      const isSuffix = req.feedName === chain.conv7OutputTensor;
      const isOneOp = req.allowArbitraryOutputs === true;
      const topo = inspectHitnetM1ResidualLadderTopology(mBytes);
      const outNames = Array.from(topo.graphOutputNames);

      if (req.enableWebGpuProfiling && isGpu) {
        return {
          ...makeMockSuccessBase({ provider: 'webgpu', graphOptimizationLevel: 'disabled' }),
          modelByteLength: mBytes.byteLength,
          modelSha256: nodeSha256Hex(mBytes),
          feedSha256: nodeSha256Hex(new Uint8Array(req.feedBuffer)),
          feedTensorName: req.feedName ?? 'input',
          feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
          feedTensorDataType: 'float32',
          outputs: [],
          gpuKernelProfileSummary: {
            profilingRequested: true,
            timestampQuerySupported: true,
            webgpuComputePassCount: 2,
            webgpuDispatchWorkgroupsCount: 8,
            webgpuDispatchIndirectCount: 0,
            webgpuTimestampWritePassCount: 0,
            webgpuQueueSubmitCount: 1,
            webgpuCommandBufferCount: 1,
            webgpuQuerySetCount: 0,
            webgpuSamplePassLabels: ['Conv'],
            ortProfilingEventCount: 2,
            ortProfilingKernelTypes: { Conv: 1 },
            ortProfilingSamplePrograms: ['Conv'],
            deviceObserverAttached: true,
            note: 'mock dispatch',
          },
        };
      }

      if (isOneOp) {
        const out = new Float32Array(2 * 40 * 120 * 160).fill(2.5);
        if (isGpu) out[7] = 2.5001;
        return {
          ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: 'disabled' }),
          modelByteLength: mBytes.byteLength,
          modelSha256: nodeSha256Hex(mBytes),
          feedSha256: nodeSha256Hex(new Uint8Array(req.feedBuffer)),
          feedTensorName: req.feedName ?? chain.node10LeakyRelu1!.outputTensor,
          feedTensorShape: req.feedDims ?? [2, 40, 120, 160],
          feedTensorDataType: 'float32',
          outputs: [{ name: chain.node11ConvDown11!.outputTensor, dims: [2, 40, 120, 160], dataType: 'float32', buffer: out.buffer }],
        };
      }

      const outputs: OrtVersionDiagnosticOutput[] = outNames.map((oName) => {
        if (oName === 'reference_output_disparity' || oName === 'secondary_output_disparity') {
          const arr = new Float32Array(76800).fill(isGpu ? 5.01 : 5.0);
          return { name: oName, dims: [1, 240, 320, 1], dataType: 'float32', buffer: arr.buffer };
        }
        if (oName.includes('Tile_2')) {
          return { name: oName, dims: [1, 60, 25600], dataType: 'int32', buffer: new Int32Array(1 * 60 * 25600).fill(7).buffer };
        }
        if (oName === chain.node11ConvDown11!.outputTensor) {
          const arr = new Float32Array(2 * 40 * 120 * 160).fill(2.5);
          if (isGpu) arr[7] = 2.5001;
          return { name: oName, dims: [2, 40, 120, 160], dataType: 'float32', buffer: arr.buffer };
        }
        const dims = parsed.staticMeta.tensorMeta[oName]?.dims ?? [2, 32, 240, 320];
        return {
          name: oName,
          dims: [...dims],
          dataType: 'float32',
          buffer: new Float32Array(dims.reduce((a, b) => a * b, 1)).fill(0.5).buffer,
        };
      });

      return {
        ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: 'disabled' }),
        modelByteLength: mBytes.byteLength,
        modelSha256: isFullUntapped ? HITNET_M1_FULL_GRAPH_SHA256 : nodeSha256Hex(mBytes),
        feedSha256: isSuffix
          ? nodeSha256Hex(new Uint8Array(req.feedBuffer))
          : HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256,
        feedTensorName: req.feedName ?? 'input',
        feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        outputs,
      };
    };

    const rep = await runHitnetPostConv7TapReplayDiagnostic({
      m1Bytes: modelBytes,
      f0102FeedFloat32: f0102,
      confirmF0126: false,
      normalizeSlot0ReshapeToTranspose: false,
      runWorker: runner,
      isDev: true,
    });

    expect(rep.ok).toBe(true);
    expect(rep.f0102.paired89Tap.node8?.comparison.bitDiffCount).toBe(0);
    expect(rep.f0102.paired89Tap.node9?.comparison.bitDiffCount).toBe(0);
    expect(rep.f0102.paired1011Tap.executed).toBe(true);
    expect(rep.f0102.paired1011Tap.node10?.comparison.bitDiffCount).toBe(0);
    expect(rep.f0102.paired1011Tap.node11?.comparison.bitDiffCount).toBe(1);
    expect(rep.f0102.localizedFirstDifferingOp.stage).toBe('node11-conv11');
    expect(rep.f0102.isolatedReplay?.reproducesFullSuffixDiff).toBe(true);
  }, 25000);

  it('aborts cleanly if paired #8/#9 tap perturbs suffix final disparity outputs relative to untapped suffix', async () => {
    const modelBytes = buildSyntheticM1ChainOnnx({
      targetByteLength: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength,
    });
    const f0102 = new Float32Array(1 * 6 * 240 * 320).fill(0.45);
    mockShaSubtleForSyntheticM1(modelBytes, f0102);

    const parsed = await parseM1ModelProtoWithRawSlices(modelBytes);
    const chain = await resolveVerifiedPostConv7ChainSpec(parsed);

    const runner: ResidualDiagnosticWorkerRunner = async (req) => {
      const mBytes = new Uint8Array(req.modelBuffer);
      const isGpu = req.provider === 'webgpu';
      const isFullUntapped = mBytes.byteLength === modelBytes.byteLength;
      const isSuffix = req.feedName === chain.conv7OutputTensor;
      const outNames = Array.from(inspectHitnetM1ResidualLadderTopology(mBytes).graphOutputNames);
      const isTap89 = isSuffix && outNames.includes(chain.node8LeakyRelu.outputTensor);

      const outputs: OrtVersionDiagnosticOutput[] = outNames.map((oName) => {
        if (oName === 'reference_output_disparity' || oName === 'secondary_output_disparity') {
          const arr = new Float32Array(76800).fill(isGpu ? 5.01 : 5.0);
          if (isTap89 && isGpu) arr[11] = 5.02; // Break invariance on GPU #8/#9 tap!
          return { name: oName, dims: [1, 240, 320, 1], dataType: 'float32', buffer: arr.buffer };
        }
        if (oName.includes('Tile_2')) {
          return { name: oName, dims: [1, 60, 25600], dataType: 'int32', buffer: new Int32Array(1 * 60 * 25600).fill(7).buffer };
        }
        const dims = parsed.staticMeta.tensorMeta[oName]?.dims ?? [2, 32, 240, 320];
        return {
          name: oName,
          dims: [...dims],
          dataType: 'float32',
          buffer: new Float32Array(dims.reduce((a, b) => a * b, 1)).fill(0.5).buffer,
        };
      });

      return {
        ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: 'disabled' }),
        modelByteLength: mBytes.byteLength,
        modelSha256: isFullUntapped ? HITNET_M1_FULL_GRAPH_SHA256 : nodeSha256Hex(mBytes),
        feedSha256: isSuffix
          ? nodeSha256Hex(new Uint8Array(req.feedBuffer))
          : HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256,
        feedTensorName: req.feedName ?? 'input',
        feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        outputs,
      };
    };

    const rep = await runHitnetPostConv7TapReplayDiagnostic({
      m1Bytes: modelBytes,
      f0102FeedFloat32: f0102,
      confirmF0126: false,
      normalizeSlot0ReshapeToTranspose: false,
      runWorker: runner,
      isDev: true,
    });

    expect(rep.ok).toBe(false);
    expect(rep.aborted).toBe(true);
    expect(rep.f0102.aborted).toBe(true);
    expect(rep.f0102.abortReason).toContain('Paired #8/#9 tap perturbed suffix final outputs');
    expect(rep.f0102.isolatedReplay).toBeNull();
  }, 20000);

  it('evaluates LeakyRelu and Conv extended reduction variants consistently', async () => {
    const modelBytes = buildSyntheticM1ChainOnnx();
    const parsed = await parseM1ModelProtoWithRawSlices(modelBytes);
    const chain = await resolveVerifiedPostConv7ChainSpec(parsed);

    const node8 = parsed.nodes[chain.node8LeakyRelu.nodeIdx];
    const in8 = new Float32Array([-0.5, -1.25, 0.0, 2.0]);
    const gpu8 = new Float32Array(in8.map((x) => (x >= 0 ? Math.fround(x) : Math.fround(x * Math.fround(0.20000000298023224)))));
    const wasm8 = new Float32Array(in8.map((x) => (x >= 0 ? Math.fround(x) : Math.fround(Number(x) * 0.20000000298023224))));
    const leakyRep = probeLeakyReluNumericVariants(node8, in8, gpu8, wasm8, { maxSamples: 16 });
    expect(leakyRep.available).toBe(true);
    expect(leakyRep.evaluatedCount).toBe(4);

    // Quick Conv probe check on small synthetic 1x1-valid shapes? Use real node9 dimensions sliced cleanly on subset (1 elem array mock)
    const convIn = new Float32Array(2 * 32 * 240 * 320).fill(0.1);
    const convGpu = new Float32Array(2 * 40 * 120 * 160).fill(0.0);
    const convWasm = new Float32Array(2 * 40 * 120 * 160).fill(0.0);
    const convRep = probeConvOpReductionVariantsExtended(
      parsed,
      chain.node9ConvDown0.nodeIdx,
      convIn,
      [2, 32, 240, 320],
      convGpu,
      convWasm,
      [2, 40, 120, 160],
      { maxSamples: 16 }
    );
    expect(convRep.available).toBe(true);
    expect(convRep.extraVariants.map((v) => v.variantId)).toEqual([
      'f32Vec4IC_KH_KW',
      'f32Fma_IC_KH_KW',
      'f32Fma_KH_KW_IC',
    ]);
  });

  if (fs.existsSync('/tmp/hitnet_sceneflow_240x320.onnx')) {
    it('verifies exact real canonical M1 N1=OFF and post-#7 #8/#9 -> #10/#11 graph chain on disk', async () => {
      const raw = new Uint8Array(fs.readFileSync('/tmp/hitnet_sceneflow_240x320.onnx'));
      if (nodeSha256Hex(raw) === HITNET_RAW_MODEL_SHA256) {
        const m1 = prepareHitnetM1FullGraphBytes(raw, { rawModelSha256: HITNET_RAW_MODEL_SHA256 });
        const m1Sha = nodeSha256Hex(m1.bytes);
        expect(m1Sha).toBe(HITNET_M1_FULL_GRAPH_SHA256);

        const parsed = await parseM1ModelProtoWithRawSlices(m1.bytes);
        const n1Proof = verifyCanonicalM1NoN1Replacements(parsed, m1.bytes, m1Sha, false);
        expect(n1Proof.verified).toBe(true);
        expect(n1Proof.n1ReplacementCount).toBe(0);

        const chain = await resolveVerifiedPostConv7ChainSpec(parsed);
        expect(chain.conv7Idx).toBe(7);
        expect(chain.node8LeakyRelu.nodeIdx).toBe(8);
        expect(chain.node8LeakyRelu.opType).toBe('LeakyRelu');
        expect(chain.node8LeakyRelu.nodeName).toBe('fe_shared/LeakyRelu');
        expect(chain.node9ConvDown0.nodeIdx).toBe(9);
        expect(chain.node9ConvDown0.opType).toBe('Conv');
        expect(chain.node9ConvDown0.outputDims).toEqual([2, 40, 120, 160]);
        expect(chain.node10LeakyRelu1?.nodeIdx).toBe(10);
        expect(chain.node11ConvDown11?.nodeIdx).toBe(11);
      }
    }, 15000);
  }
});
