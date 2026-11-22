// @vitest-environment node
import crypto from 'node:crypto';
import fs from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { computeConv2dCandidateFullTensor } from './hitnetConv9BlockedInjection';
import { buildVerifiedSuffixAfterConv7 } from './hitnetConv7ActivationSubstitution';
import { prepareHitnetM1FullGraphBytes } from './hitnetGatherNdDiagnostic';
import { type OrtVersionDiagnosticOutput } from './hitnetOrtComparisonHelper';
import { resolveVerifiedPostConv7ChainSpec } from './hitnetPostConv7TapReplay';
import {
  inspectHitnetM1ResidualLadderTopology,
  type ResidualDiagnosticWorkerRunner,
} from './hitnetResidualLocalization';
import { parseM1ModelProtoWithRawSlices } from './hitnetRgbStemFirstNonZero';
import {
  populateFreshOwnedHostConv9Buffer,
  resolveSecondaryOutlierAncestrySpec,
  runHitnetF0126SecondaryOutlierDiagnostic,
} from './hitnetF0126SecondaryOutlierDiagnostic';
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

function encodeValueInfo(name: string, elemType: number, dims: readonly number[]): Uint8Array {
  const dimFields = dims.map((d) => encodeLenField(1, encodeVarintField(1, d)));
  const shapeProto = encodeLenField(2, concatBytes(dimFields));
  const tensorTypeProto = encodeLenField(1, concatBytes([encodeVarintField(1, elemType), shapeProto]));
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

/**
 * Builds a synthetic canonical-compatible ONNX graph with #0..#9 RGB-stem, bypass Tile_2 + LeakyRelu #8,
 * shared primary head, and an explicit secondary-only ancestry chain:
 * coarse Cost -> ArgMin(axis=3, select_last_index=0) -> Cast -> init_to_prop/ExpandDims
 * -> prop_warp -> prop0/add -> prop0/ArgMin(singleton axis=3) -> postProp0 Squeeze
 * -> refinement_l0 -> refinement_l1 -> refinement_l2/add -> refinement_l2/add_3 -> secondary_output_disparity.
 */
function buildSyntheticM1SecondaryChainOnnx(targetByteLength?: number): Uint8Array {
  const convW7 = Array.from({ length: 32 * 3 * 3 * 3 }, (_, i) => ((i % 7) - 3) * 0.05);
  const convB7 = Array.from({ length: 32 }, (_, i) => i * 0.01);
  const convW9 = Array.from({ length: 40 * 32 * 4 * 4 }, (_, i) => ((i % 11) - 5) * 0.031);
  const convB9 = Array.from({ length: 40 }, (_, i) => (i % 5) * 0.017 - 0.03);
  const convW11 = Array.from({ length: 40 * 40 * 3 * 3 }, (_, i) => ((i % 3) - 1) * 0.01);
  const convB11 = Array.from({ length: 40 }, () => 0.005);

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
    // Shared M1 slot-0 Reshapes (no N1 Transpose)
    encodeNode('Cast', 'level0/level_init/Cast_slot0', ['level0/level_init/Tile_2'], ['cast_slot0_a'], [encodeAttrInt('to', 1)]),
    encodeNode('Cast', 'level0_1/level_init/Cast_slot0', ['level0_1/level_init/Tile_2'], ['cast_slot0_b'], [encodeAttrInt('to', 1)]),
    encodeNode('Reshape', 'Transpose__908', ['cast_slot0_a', 'slot0_shape'], ['Transpose__908:0']),
    encodeNode('Reshape', 'Transpose__1006', ['cast_slot0_b', 'slot0_shape'], ['Transpose__1006:0']),
    // Bypass consumer of #8 (fe_shared/LeakyRelu:0) so suffix9 retains #8 as bypass cut edge
    encodeNode('Identity', 'fe_shared/conv_up_0/bypass_ref', ['fe_shared/LeakyRelu:0'], ['bypass8_copy']),
    // Primary path
    encodeNode('Add', 'prim/combine_8_11', ['fe_shared/conv_down_1_1/BiasAdd:0', 'prim/bias_pad'], ['prim_feat']),
    encodeNode('Slice', 'prim/head_slice', ['prim_feat', 's0', 'e1', 'a0'], ['head_slice_0']),
    encodeNode('Reshape', 'prim/head', ['head_slice_0', 'out_shape'], ['reference_output_disparity']),
    // Secondary-only ancestry chain (diverges from shared conv_down_1_1 & Tile_2 & bypass8)
    encodeNode('Reshape', 'level0_1/level_init/Reshape_2', ['fe_shared/conv_down_1_1/BiasAdd:0', 'sec_cost_shape'], ['level0_1/level_init/Reshape_2:0']),
    encodeNode(
      'ArgMin',
      'level0_1/level_init/ArgMin',
      ['level0_1/level_init/Reshape_2:0'],
      ['level0_1/level_init/ArgMin:0'],
      [encodeAttrInt('axis', 3), encodeAttrInt('keepdims', 0), encodeAttrInt('select_last_index', 0)]
    ),
    encodeNode(
      'Cast',
      'level0_1/level_init/Cast',
      ['level0_1/level_init/ArgMin:0'],
      ['level0_1/level_init/Cast:0'],
      [encodeAttrInt('to', 1)]
    ),
    encodeNode(
      'Reshape',
      'level0_1/level_init/init_to_prop/ExpandDims',
      ['level0_1/level_init/Cast:0', 'init_prop_shape'],
      ['level0_1/level_init/init_to_prop/ExpandDims:0']
    ),
    encodeNode(
      'Reshape',
      'level0_1/shared/prop0/prop_warp/Reshape_1',
      ['bypass8_copy', 'prop_warp_shape'],
      ['level0_1/shared/prop0/prop_warp/Reshape_1:0']
    ),
    encodeNode(
      'Add',
      'level0_1/shared/prop0/add',
      ['level0_1/level_init/init_to_prop/ExpandDims:0', 'level0_1/shared/prop0/prop_warp/Reshape_1:0'],
      ['level0_1/shared/prop0/add:0']
    ),
    encodeNode(
      'ArgMin',
      'level0_1/shared/prop0/ArgMin',
      ['level0_1/shared/prop0/add:0'],
      ['level0_1/shared/prop0/ArgMin:0'],
      [encodeAttrInt('axis', 3), encodeAttrInt('keepdims', 0), encodeAttrInt('select_last_index', 0)]
    ),
    encodeNode(
      'Reshape',
      'shared_1/strided_slice__391',
      ['level0_1/shared/prop0/add:0', 'post_prop0_shape'],
      ['shared_1/strided_slice__391:0']
    ),
    encodeNode(
      'Reshape',
      'shared_1/refinement_l0/strided_slice_17__462',
      ['shared_1/strided_slice__391:0', 'ref_l0_shape'],
      ['shared_1/refinement_l0/strided_slice_17__462:0']
    ),
    encodeNode(
      'Reshape',
      'shared_1/refinement_l1/strided_slice_173__533',
      ['shared_1/refinement_l0/strided_slice_17__462:0', 'ref_l1_shape'],
      ['shared_1/refinement_l1/strided_slice_173__533:0']
    ),
    encodeNode(
      'Reshape',
      'shared_1/refinement_l2/pre_add_reshape',
      ['shared_1/refinement_l1/strided_slice_173__533:0', 'ref_l2_add_shape'],
      ['shared_1/refinement_l2/pre_add_reshape:0']
    ),
    encodeNode(
      'Add',
      'shared_1/refinement_l2/add',
      ['shared_1/refinement_l2/pre_add_reshape:0', 'sec_zero_16'],
      ['shared_1/refinement_l2/add:0']
    ),
    encodeNode(
      'Slice',
      'shared_1/refinement_l2/slice_1',
      ['shared_1/refinement_l2/add:0', 's0', 'e1', 'a1'],
      ['shared_1/refinement_l2/slice_1:0']
    ),
    encodeNode(
      'Add',
      'shared_1/refinement_l2/add_3',
      ['shared_1/refinement_l2/slice_1:0', 'sec_zero_1'],
      ['shared_1/refinement_l2/add_3:0']
    ),
    encodeNode('Reshape', 'sec/head', ['shared_1/refinement_l2/add_3:0', 'out_shape'], ['secondary_output_disparity']),
  ];

  const graphBody = concatBytes([
    ...nodes,
    encodeStrField(2, 'synthetic_hitnet_m1_secondary_outlier_chain'),
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
    encodeInt64Initializer('sec_cost_shape', [1, 60, 80, 320]),
    encodeInt64Initializer('init_prop_shape', [1, 60, 80, 1, 16]),
    encodeInt64Initializer('prop_warp_shape', [1, 60, 80, 1, 16]),
    encodeInt64Initializer('post_prop0_shape', [1, 60, 80, 16]),
    encodeInt64Initializer('ref_l0_shape', [1, 120, 160, 16]),
    encodeInt64Initializer('ref_l1_shape', [1, 240, 320, 16]),
    encodeInt64Initializer('ref_l2_add_shape', [1, 16, 240, 320]),
    encodeFloatInitializer('prim/bias_pad', [], [0.0]),
    encodeFloatInitializer('sec_zero_16', [], [0.0]),
    encodeFloatInitializer('sec_zero_1', [], [0.0]),
    encodeFloatInitializer('w_down_1_0', [32, 3, 3, 3], convW7),
    encodeFloatInitializer('b_down_1_0', [32], convB7),
    encodeFloatInitializer('const_fold_opt__1272', [40, 32, 4, 4], convW9),
    encodeFloatInitializer('b_down_0', [40], convB9),
    encodeFloatInitializer('const_fold_opt__1366', [40, 40, 3, 3], convW11),
    encodeFloatInitializer('b_down_1_1', [40], convB11),
    encodeLenField(11, encodeValueInfo('input', 1, [1, 6, 240, 320])),
    encodeLenField(13, encodeValueInfo('fe_shared/LeakyRelu:0', 1, [2, 32, 240, 320])),
    encodeLenField(13, encodeValueInfo('fe_shared/conv_down_0/BiasAdd:0', 1, [2, 40, 120, 160])),
    encodeLenField(13, encodeValueInfo('level0_1/level_init/Tile_2', 6, [1, 60, 25600])),
    encodeLenField(13, encodeValueInfo('level0/level_init/Tile_2', 6, [1, 60, 25600])),
    encodeLenField(13, encodeValueInfo('level0_1/level_init/Reshape_2:0', 1, [1, 60, 80, 320])),
    encodeLenField(13, encodeValueInfo('level0_1/level_init/ArgMin:0', 7, [1, 60, 80])),
    encodeLenField(13, encodeValueInfo('level0_1/level_init/Cast:0', 1, [1, 60, 80, 1])),
    encodeLenField(13, encodeValueInfo('level0_1/level_init/init_to_prop/ExpandDims:0', 1, [1, 60, 80, 1, 16])),
    encodeLenField(13, encodeValueInfo('level0_1/shared/prop0/prop_warp/Reshape_1:0', 1, [1, 60, 80, 1, 16])),
    encodeLenField(13, encodeValueInfo('level0_1/shared/prop0/add:0', 1, [1, 60, 80, 1, 16])),
    encodeLenField(13, encodeValueInfo('level0_1/shared/prop0/ArgMin:0', 7, [1, 60, 80])),
    encodeLenField(13, encodeValueInfo('shared_1/strided_slice__391:0', 1, [1, 60, 80, 16])),
    encodeLenField(13, encodeValueInfo('shared_1/refinement_l0/strided_slice_17__462:0', 1, [1, 120, 160, 16])),
    encodeLenField(13, encodeValueInfo('shared_1/refinement_l1/strided_slice_173__533:0', 1, [1, 240, 320, 16])),
    encodeLenField(13, encodeValueInfo('shared_1/refinement_l2/add:0', 1, [1, 16, 240, 320])),
    encodeLenField(13, encodeValueInfo('shared_1/refinement_l2/add_3:0', 1, [1, 1, 240, 320])),
    encodeLenField(12, encodeValueInfo('reference_output_disparity', 1, [1, 240, 320, 1])),
    encodeLenField(12, encodeValueInfo('secondary_output_disparity', 1, [1, 240, 320, 1])),
  ]);

  const opsetImport = encodeLenField(8, concatBytes([encodeStrField(1, ''), encodeVarintField(2, 13)]));
  const baseModel = concatBytes([
    encodeVarintField(1, 7),
    encodeStrField(2, 'synthetic'),
    encodeLenField(7, graphBody),
    opsetImport,
  ]);

  if (!targetByteLength) return baseModel;
  const padKey = encodeStrField(1, 'pad');
  let valLen = targetByteLength - (baseModel.byteLength + padKey.byteLength + 6);
  while (true) {
    const valField = encodeLenField(2, new Uint8Array(Math.max(0, valLen)).fill(0x61));
    const entry = encodeLenField(14, concatBytes([padKey, valField]));
    const total = baseModel.byteLength + entry.byteLength;
    if (total === targetByteLength) return concatBytes([baseModel, entry]);
    valLen += targetByteLength - total;
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
    loadedOrtResources: [
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs',
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.asyncify.wasm',
    ],
    foreignOrtResources: [],
    loadedAssetProvenance: [
      {
        role: 'wasm' as const,
        url: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.asyncify.wasm',
        sha256: 'deadbeef',
        byteLength: 12345,
        via: 'WebAssembly.instantiateStreaming' as const,
        referenceLocalDistSha256: 'deadbeef',
        matchesLocalDistSha256: true,
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
    warmRunMs: [],
  };
}

function mockShaSubtleForSyntheticM1(modelBytes: Uint8Array, f0126Feed: Float32Array) {
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
    if (u8.byteLength === f0126Feed.byteLength && h === f0126Sha) {
      return hexToArrayBuffer(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256);
    }
    return hexToArrayBuffer(h);
  });
}

describe('hitnetF0126SecondaryOutlierDiagnostic', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  if (fs.existsSync('/tmp/hitnet_sceneflow_240x320.onnx')) {
    it('dynamically resolves canonical M1 secondary-only ancestry without hardcoded indices & blocked-K=128 matches computeConv2dCandidateFullTensor 0-bit', async () => {
      const raw = new Uint8Array(fs.readFileSync('/tmp/hitnet_sceneflow_240x320.onnx'));
      if (nodeSha256Hex(raw) === HITNET_RAW_MODEL_SHA256) {
        const m1 = prepareHitnetM1FullGraphBytes(raw, { rawModelSha256: HITNET_RAW_MODEL_SHA256 });
        expect(nodeSha256Hex(m1.bytes)).toBe(HITNET_M1_FULL_GRAPH_SHA256);

        const parsed = await parseM1ModelProtoWithRawSlices(m1.bytes);
        const chainSpec = await resolveVerifiedPostConv7ChainSpec(parsed);
        const suffix9 = await buildVerifiedSuffixAfterConv7(parsed, {
          cutNodeIdx: chainSpec.node9ConvDown0.nodeIdx,
        });

        const ancestry = resolveSecondaryOutlierAncestrySpec(parsed, chainSpec, suffix9);
        expect(ancestry.coarseArgMin.opType).toBe('ArgMin');
        expect(ancestry.coarseArgMin.nodeName).toBe('level0_1/level_init/ArgMin');
        expect(ancestry.coarseArgMin.elemType).toBe(7); // int64
        expect(ancestry.coarseArgMin.dims).toEqual([1, 60, 80]);
        expect(ancestry.coarseArgMin.selectLastIndex).toBe(0);
        expect(ancestry.coarseCast?.nodeName).toBe('level0_1/level_init/Cast');
        expect(ancestry.initToPropOut?.dims).toEqual([1, 60, 80, 1, 16]);
        expect(ancestry.propWarpOut?.dims).toEqual([1, 60, 80, 1, 16]);
        expect(ancestry.prop0AddOut?.dims).toEqual([1, 60, 80, 1, 16]);
        expect(ancestry.prop0SingletonArgMin?.opType).toBe('ArgMin');
        expect(ancestry.postProp0Out?.dims).toEqual([1, 60, 80, 16]);
        expect(ancestry.refinementL0Out?.dims).toEqual([1, 120, 160, 16]);
        expect(ancestry.refinementL1Out?.dims).toEqual([1, 240, 320, 16]);
        expect(ancestry.refinementL2Add?.dims).toEqual([1, 16, 240, 320]);
        expect(ancestry.refinementL2SlantAdd?.dims).toEqual([1, 1, 240, 320]);

        // Verify populateFreshOwnedHostConv9Buffer ('blocked-k128') is 0-ULP bit-identical to computeConv2dCandidateFullTensor
        const smallIn = new Float32Array(2 * 32 * 8 * 10);
        for (let i = 0; i < smallIn.length; i++) {
          smallIn[i] = Math.fround(Math.sin(i * 0.17) * 1.3 + Math.cos(i * 0.09) * 0.5);
        }
        const hostBuf = populateFreshOwnedHostConv9Buffer({
          mode: 'blocked-k128',
          parsed,
          convNodeIdx: chainSpec.node9ConvDown0.nodeIdx,
          xData: smallIn,
          inputDims: [2, 32, 8, 10],
          outDims: [2, 40, 4, 5],
        });
        expect(hostBuf.byteOffset).toBe(0);
        expect(hostBuf.buffer.byteLength).toBe(hostBuf.byteLength);

        const refBuf = computeConv2dCandidateFullTensor({
          parsed,
          convNodeIdx: chainSpec.node9ConvDown0.nodeIdx,
          xData: smallIn,
          inputDims: [2, 32, 8, 10],
          outDims: [2, 40, 4, 5],
          candidateId: 'blocked_k128_ic_kh_kw_bias_last',
        });
        expect(new Uint32Array(hostBuf.buffer)).toEqual(new Uint32Array(refBuf.buffer));
      }
    }, 30000);
  }

  it('aborts IMMEDIATELY on mandatory native-on-candidate transport gate failure (1-bit output diff OR bypass SHA/order mismatch) with zero outlier attribution', async () => {
    const modelBytes = buildSyntheticM1SecondaryChainOnnx(
      HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength
    );
    const f0126 = new Float32Array(1 * 6 * 240 * 320).fill(0.35);
    mockShaSubtleForSyntheticM1(modelBytes, f0126);

    const parsed = await parseM1ModelProtoWithRawSlices(modelBytes);
    const chain = await resolveVerifiedPostConv7ChainSpec(parsed);
    const suffix9 = await buildVerifiedSuffixAfterConv7(parsed, {
      cutNodeIdx: chain.node9ConvDown0.nodeIdx,
    });

    const node8Data = new Float32Array(2 * 32 * 240 * 320).fill(0.125);
    const nativeGpu9Data = new Float32Array(2 * 40 * 120 * 160).fill(0.5);
    const f0126RawSha = nodeSha256Hex(new Uint8Array(f0126.buffer));
    const nativeGpu9Sha = nodeSha256Hex(new Uint8Array(nativeGpu9Data.buffer));

    const makeGateBreakerRunner = (mode: '1bit-final-diff' | 'bypass-sha-mismatch'): ResidualDiagnosticWorkerRunner => {
      let suffixCallCount = 0;
      return async (req) => {
        const mBytes = new Uint8Array(req.modelBuffer);
        const isGpu = req.provider === 'webgpu';
        const isFullUntapped = mBytes.byteLength === modelBytes.byteLength;
        const topo = inspectHitnetM1ResidualLadderTopology(mBytes);
        const outNames = Array.from(topo.graphOutputNames);
        const fShaRaw = nodeSha256Hex(new Uint8Array(req.feedBuffer));
        const feedSha = fShaRaw === f0126RawSha ? HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256 : fShaRaw;

        if (req.enableWebGpuProfiling && isGpu) {
          return {
            ...makeMockSuccessBase({ provider: 'webgpu', graphOptimizationLevel: req.graphOptimizationLevel }),
            modelByteLength: mBytes.byteLength,
            modelSha256: isFullUntapped ? HITNET_M1_FULL_GRAPH_SHA256 : nodeSha256Hex(mBytes),
            feedSha256: feedSha,
            feedTensorName: req.feedName ?? 'input',
            feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
            feedTensorDataType: 'float32',
            outputs: [],
            gpuKernelProfileSummary: {
              profilingRequested: true,
              timestampQuerySupported: true,
              webgpuComputePassCount: 2,
              webgpuDispatchWorkgroupsCount: 16,
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
              note: 'mock',
            },
          };
        }

        // Single-op #9
        if (req.allowArbitraryOutputs && req.feedName === chain.node8LeakyRelu.outputTensor) {
          return {
            ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: req.graphOptimizationLevel }),
            modelByteLength: mBytes.byteLength,
            modelSha256: nodeSha256Hex(mBytes),
            feedSha256: feedSha,
            feedTensorName: chain.node8LeakyRelu.outputTensor,
            feedTensorShape: [2, 32, 240, 320],
            feedTensorDataType: 'float32',
            outputs: [
              {
                name: chain.node9ConvDown0.outputTensor,
                dims: [2, 40, 120, 160],
                dataType: 'float32',
                buffer: nativeGpu9Data.slice().buffer,
              },
            ],
          };
        }

        // Prefix #0..#8
        if (outNames.includes(chain.node8LeakyRelu.outputTensor) && !outNames.includes('reference_output_disparity')) {
          return {
            ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: req.graphOptimizationLevel }),
            modelByteLength: mBytes.byteLength,
            modelSha256: nodeSha256Hex(mBytes),
            feedSha256: feedSha,
            feedTensorName: 'input',
            feedTensorShape: [1, 6, 240, 320],
            feedTensorDataType: 'float32',
            outputs: outNames.map((oName) => {
              if (oName === chain.node8LeakyRelu.outputTensor) {
                return { name: oName, dims: [2, 32, 240, 320], dataType: 'float32', buffer: node8Data.slice().buffer };
              }
              return {
                name: oName,
                dims: [1, 60, 25600],
                dataType: 'int32',
                buffer: new Int32Array(1 * 60 * 25600).fill(3).buffer,
              };
            }),
          };
        }

        // Suffix or Full
        const isSuffix = req.feedName === chain.node9ConvDown0.outputTensor;
        if (isSuffix) suffixCallCount++;

        const secData = new Float32Array(76800).fill(isGpu ? 12.0 : 11.9);
        if (mode === '1bit-final-diff' && isSuffix && suffixCallCount === 2) {
          // 1-bit Perturbation ONLY on the 2nd suffix call (the Gate shared-host-transport arm)
          secData[400] = Math.fround(secData[400] + Math.pow(2, -20));
        }

        const extraChecks = req.extraFeeds?.map((ef, idx) => ({
          name: ef.name,
          dims: [...ef.dims],
          dataType: ef.dataType ?? 'float32',
          byteOffset: 0,
          byteLength: ef.buffer.byteLength,
          sha256:
            mode === 'bypass-sha-mismatch' && isSuffix && suffixCallCount === 2 && idx === 0
              ? '0'.repeat(64)
              : nodeSha256Hex(new Uint8Array(ef.buffer)),
        }));

        const sessionInputNames = isSuffix ? [...suffix9.graphInputNames] : ['input'];

        return {
          ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: req.graphOptimizationLevel }),
          modelByteLength: mBytes.byteLength,
          modelSha256: isFullUntapped ? HITNET_M1_FULL_GRAPH_SHA256 : nodeSha256Hex(mBytes),
          feedSha256: feedSha,
          feedTensorName: req.feedName ?? 'input',
          feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
          feedTensorDataType: 'float32',
          feedByteOffset: 0,
          feedByteLength: req.feedBuffer.byteLength,
          extraFeedChecks: extraChecks,
          sessionInputNames,
          outputs: [
            {
              name: 'reference_output_disparity',
              dims: [1, 240, 320, 1],
              dataType: 'float32',
              buffer: new Float32Array(76800).fill(isGpu ? 14.0 : 13.9).buffer,
            },
            {
              name: 'secondary_output_disparity',
              dims: [1, 240, 320, 1],
              dataType: 'float32',
              buffer: secData.buffer,
            },
          ],
        };
      };
    };

    // Sub-case A: 1-bit final difference breaks gate immediately
    const repBit = await runHitnetF0126SecondaryOutlierDiagnostic({
      m1Bytes: modelBytes,
      f0126FeedFloat32: f0126,
      normalizeSlot0ReshapeToTranspose: false,
      runWorker: makeGateBreakerRunner('1bit-final-diff'),
      isDev: true,
    });
    expect(repBit.ok).toBe(false);
    expect(repBit.aborted).toBe(true);
    expect(repBit.gatePassed).toBe(false);
    expect(repBit.gateFirstFailure?.checkId).toBe('gate-bit-exact-vs-native-split-secondary');
    expect(repBit.outlierComparison).toBeNull();
    expect(repBit.ancestryAndTaps).toBeNull();
    expect(repBit.causalAttribution).toBeNull();
    expect(repBit.commonWasm7Secondary.source).toBe('skipped-on-gate-failure');
    expect(repBit.gateTransportRecord.execution.edges[0].sha256Before).toBe(nativeGpu9Sha);

    // Sub-case B: Bypass edge SHA disagreement post-transfer breaks gate before output attribution
    const repBypass = await runHitnetF0126SecondaryOutlierDiagnostic({
      m1Bytes: modelBytes,
      f0126FeedFloat32: f0126,
      normalizeSlot0ReshapeToTranspose: false,
      runWorker: makeGateBreakerRunner('bypass-sha-mismatch'),
      isDev: true,
    });
    expect(repBypass.ok).toBe(false);
    expect(repBypass.aborted).toBe(true);
    expect(repBypass.gatePassed).toBe(false);
    expect(repBypass.gateFirstFailure?.stage).toBe('post-transfer');
    expect(repBypass.gateFirstFailure?.checkId).toContain('post-bypass-edge-');
    expect(repBypass.outlierComparison).toBeNull();
    expect(repBypass.ancestryAndTaps).toBeNull();
    expect(repBypass.causalAttribution).toBeNull();
  }, 30000);

  it('passes Gate (0-bit exact) and attributes coarse INT64 ArgMin cost-order crossing vs exact-tie flip with ONNX select_last_index=0', async () => {
    const modelBytes = buildSyntheticM1SecondaryChainOnnx(
      HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength
    );
    const f0126 = new Float32Array(1 * 6 * 240 * 320);
    for (let i = 0; i < f0126.length; i++) f0126[i] = ((i * 29) % 247) / 255;
    mockShaSubtleForSyntheticM1(modelBytes, f0126);

    const parsed = await parseM1ModelProtoWithRawSlices(modelBytes);
    const chain = await resolveVerifiedPostConv7ChainSpec(parsed);
    const suffix9 = await buildVerifiedSuffixAfterConv7(parsed, {
      cutNodeIdx: chain.node9ConvDown0.nodeIdx,
    });
    const ancestry = resolveSecondaryOutlierAncestrySpec(parsed, chain, suffix9);

    const node8Data = new Float32Array(2 * 32 * 240 * 320).fill(0.1);
    const nativeGpu9Data = new Float32Array(2 * 40 * 120 * 160).fill(0.5);
    const nativeGpu9Sha = nodeSha256Hex(new Uint8Array(nativeGpu9Data.buffer));
    const f0126RawSha = nodeSha256Hex(new Uint8Array(f0126.buffer));

    const runScenario = async (scenario: 'order-crossing' | 'exact-tie', reuseSummary: boolean) => {
      // Outlier at tile (ty=10, tx=20) -> pixels y=40..43, x=80..83
      const peakTy = 10;
      const peakTx = 20;
      const peakTile = peakTy * 80 + peakTx;
      const dNat = 42;
      const dCand = 78;

      const buildSuffixRunner = (): ResidualDiagnosticWorkerRunner => async (req) => {
        const mBytes = new Uint8Array(req.modelBuffer);
        const isGpu = req.provider === 'webgpu';
        const isFullUntapped = mBytes.byteLength === modelBytes.byteLength;
        const topo = inspectHitnetM1ResidualLadderTopology(mBytes);
        const outNames = Array.from(topo.graphOutputNames);
        const fShaRaw = nodeSha256Hex(new Uint8Array(req.feedBuffer));
        const feedSha = fShaRaw === f0126RawSha ? HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256 : fShaRaw;

        if (req.enableWebGpuProfiling && isGpu) {
          return {
            ...makeMockSuccessBase({ provider: 'webgpu', graphOptimizationLevel: req.graphOptimizationLevel }),
            modelByteLength: mBytes.byteLength,
            modelSha256: isFullUntapped ? HITNET_M1_FULL_GRAPH_SHA256 : nodeSha256Hex(mBytes),
            feedSha256: feedSha,
            feedTensorName: req.feedName ?? 'input',
            feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
            feedTensorDataType: 'float32',
            outputs: [],
            gpuKernelProfileSummary: {
              profilingRequested: true,
              timestampQuerySupported: true,
              webgpuComputePassCount: 3,
              webgpuDispatchWorkgroupsCount: 24,
              webgpuDispatchIndirectCount: 0,
              webgpuTimestampWritePassCount: 0,
              webgpuQueueSubmitCount: 1,
              webgpuCommandBufferCount: 1,
              webgpuQuerySetCount: 0,
              webgpuSamplePassLabels: ['Conv'],
              ortProfilingEventCount: 3,
              ortProfilingKernelTypes: { Conv: 1 },
              ortProfilingSamplePrograms: ['Conv'],
              deviceObserverAttached: true,
              note: 'mock dispatch',
            },
          };
        }

        if (req.allowArbitraryOutputs && req.feedName === chain.node8LeakyRelu.outputTensor) {
          return {
            ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: req.graphOptimizationLevel }),
            modelByteLength: mBytes.byteLength,
            modelSha256: nodeSha256Hex(mBytes),
            feedSha256: feedSha,
            feedTensorName: chain.node8LeakyRelu.outputTensor,
            feedTensorShape: [2, 32, 240, 320],
            feedTensorDataType: 'float32',
            outputs: [
              {
                name: chain.node9ConvDown0.outputTensor,
                dims: [2, 40, 120, 160],
                dataType: 'float32',
                buffer: nativeGpu9Data.slice().buffer,
              },
            ],
          };
        }

        if (outNames.includes(chain.node8LeakyRelu.outputTensor) && !outNames.includes('reference_output_disparity') && !outNames.includes(chain.conv7OutputTensor)) {
          return {
            ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: req.graphOptimizationLevel }),
            modelByteLength: mBytes.byteLength,
            modelSha256: nodeSha256Hex(mBytes),
            feedSha256: feedSha,
            feedTensorName: 'input',
            feedTensorShape: [1, 6, 240, 320],
            feedTensorDataType: 'float32',
            outputs: outNames.map((oName) =>
              oName === chain.node8LeakyRelu.outputTensor
                ? { name: oName, dims: [2, 32, 240, 320], dataType: 'float32', buffer: node8Data.slice().buffer }
                : { name: oName, dims: [1, 60, 25600], dataType: 'int32', buffer: new Int32Array(1 * 60 * 25600).fill(5).buffer }
            ),
          };
        }

        const isSuffix9 = req.feedName === chain.node9ConvDown0.outputTensor;
        const isCand9Feed = isSuffix9 && feedSha !== nativeGpu9Sha;
        const secOut = new Float32Array(76800).fill(isGpu ? 8.0 : 8.005);
        const primOut = new Float32Array(76800).fill(isGpu ? 10.0 : 10.002);

        if (isCand9Feed) {
          // Inject localized 8.844 px outlier in 4x4 tile (ty=10, tx=20)
          for (let dy = 0; dy < 4; dy++) {
            for (let dx = 0; dx < 4; dx++) {
              const pIdx = (peakTy * 4 + dy) * 320 + (peakTx * 4 + dx);
              secOut[pIdx] = dx === 1 && dy === 1 ? 8.0 + 8.844 : 8.0 + 3.2;
            }
          }
          // Flip 1 pixel validity at (0, 5)
          secOut[5] = -0.25;
        }

        const outputs: OrtVersionDiagnosticOutput[] = outNames.map((oName) => {
          if (oName === 'reference_output_disparity') {
            return { name: oName, dims: [1, 240, 320, 1], dataType: 'float32', buffer: primOut.buffer };
          }
          if (oName === 'secondary_output_disparity') {
            return { name: oName, dims: [1, 240, 320, 1], dataType: 'float32', buffer: secOut.buffer };
          }
          if (oName === ancestry.coarseArgMin.outputTensor) {
            const arr = new BigInt64Array(4800).fill(BigInt(dNat));
            arr[peakTile] = BigInt(isCand9Feed ? dCand : dNat);
            return { name: oName, dims: [1, 60, 80], dataType: 'int64', buffer: arr.buffer };
          }
          if (oName === ancestry.postProp0Out?.outputTensor) {
            const arr = new Float32Array(4800 * 16).fill(1.0);
            if (isCand9Feed) arr[peakTile * 16] = 9.8;
            return { name: oName, dims: [1, 60, 80, 16], dataType: 'float32', buffer: arr.buffer };
          }
          if (oName === ancestry.coarseArgMin.inputCostTensor) {
            const cost = new Float32Array(4800 * 320).fill(10.0);
            const base = peakTile * 320;
            if (scenario === 'order-crossing') {
              if (!isCand9Feed) {
                cost[base + dNat] = Math.fround(0.5);
                cost[base + dCand] = Math.fround(0.500002);
              } else {
                cost[base + dNat] = Math.fround(0.500001);
                cost[base + dCand] = Math.fround(0.499998);
              }
            } else {
              // Exact tie in candidate arm (select_last_index=0 picks min index if tied; test both tied at dNat/dCand depending on tie semantics)
              // Specifically, make Native arm tie between dNat (42) and dCand (78) with minCost=0.5 (select_last_index=0 -> 42),
              // and Cand arm tie between dCand (78) and 120 where exact tie triggers or tieIndices includes other winner!
              cost[base + dNat] = Math.fround(0.5);
              cost[base + dCand] = Math.fround(0.5);
              if (isCand9Feed) {
                // To trigger exact tie tiebreaker in our classifier: tieIndicesNative.includes(dCand)
                // Suppose kernel picked dCand (tie-breaker flip on exact tie) while exactCost remained tied at 0.5!
                // Note: if observedMatchesOnnxExactCand must be true for EXACT_TIE_TIEBREAKER:
                // cand has tied cost at dCand and another higher index, wait: if dNat is slightly broken in tie, or native matches 42 and cand had equal min?
                // Wait: in classifier:
                // if (!matchesNat || !matchesCand) => KERNEL_MISMATCH
                // else if (exactNat.tieIndices.includes(dCand) || exactCand.tieIndices.includes(dNative)) => EXACT_TIE_TIEBREAKER
                // For both matchesNat and matchesCand to be true AND exactNat.tieIndices.includes(dCand):
                // Native: minCost at dNat=42 AND dCand=78 (tieIndicesNative=[42,78], select_last=0 -> exactNat.winner=42 === dNat).
                // Cand: cost at dNat slightly higher OR wait: if dCand < dNat? If dNat=42 < dCand=78, exactCand.winner=78 ONLY if costCand[42] > costCand[78] -> wait then exactNat had tie=[42,78]!
                cost[base + dNat] = Math.fround(0.5);
                cost[base + dCand] = Math.fround(0.5);
                if (isCand9Feed) {
                  cost[base + dNat] = Math.fround(0.5000001); // actually slightly above? Wait: exactNat.tieIndices includes 78 -> EXACT_TIE_TIEBREAKER!
                }
              }
            }
            return { name: oName, dims: [1, 60, 80, 320], dataType: 'float32', buffer: cost.buffer };
          }
          if (oName === ancestry.initToPropOut?.outputTensor) {
            const initArr = new Float32Array(4800 * 16).fill(1.0);
            initArr[peakTile * 16] = isCand9Feed ? dCand : dNat;
            return { name: oName, dims: [1, 60, 80, 1, 16], dataType: 'float32', buffer: initArr.buffer };
          }
          if (oName === chain.conv7OutputTensor || oName === chain.node8LeakyRelu.outputTensor) {
            return { name: oName, dims: [2, 32, 240, 320], dataType: 'float32', buffer: node8Data.slice().buffer };
          }
          if (oName.includes('Tile_2')) {
            return { name: oName, dims: [1, 60, 25600], dataType: 'int32', buffer: new Int32Array(1 * 60 * 25600).fill(5).buffer };
          }
          return { name: oName, dims: [1, 60, 80, 16], dataType: 'float32', buffer: new Float32Array(4800 * 16).buffer };
        });

        const sessionInputNames = isSuffix9
          ? [...suffix9.graphInputNames]
          : req.feedName === chain.conv7OutputTensor
            ? [chain.conv7OutputTensor, ...suffix9.bypassCutInputs.filter((b) => !b.name.includes('LeakyRelu')).map((b) => b.name)]
            : ['input'];

        return {
          ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: req.graphOptimizationLevel }),
          modelByteLength: mBytes.byteLength,
          modelSha256: isFullUntapped ? HITNET_M1_FULL_GRAPH_SHA256 : nodeSha256Hex(mBytes),
          feedSha256: feedSha,
          feedTensorName: req.feedName ?? 'input',
          feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
          feedTensorDataType: 'float32',
          feedByteOffset: 0,
          feedByteLength: req.feedBuffer.byteLength,
          extraFeedChecks: req.extraFeeds?.map((ef) => ({
            name: ef.name,
            dims: [...ef.dims],
            dataType: ef.dataType ?? 'float32',
            byteOffset: 0,
            byteLength: ef.buffer.byteLength,
            sha256: nodeSha256Hex(new Uint8Array(ef.buffer)),
          })),
          sessionInputNames,
          outputs,
        };
      };

      const dummyStats = {
        length: 76800,
        finiteCount: 76800,
        nonFiniteCount: 0,
        min: 0,
        max: 10,
        mean: 8,
        negativeCount: 0,
        zeroCount: 0,
        positiveCount: 76800,
        negateDisparity: false,
        maxAbsDiff: 0.05,
        p99AbsDiff: 0.01,
        meanAbsDiff: 0.002,
      };
      const existingSummary = reuseSummary
        ? {
            controlScopeNote: 'Reused precomputed F0126 Common-WASM-#7 summary.',
            dispatchConfirmedSeparatePass: true,
            causalBaselineGpuSuffixWasm7VsWasmFull: {
              primary: { stats: dummyStats, flippedPixelCount: 0, maxValidAbsDiff: 0.01 },
              secondary: { stats: { ...dummyStats, maxAbsDiff: 0.06 }, flippedPixelCount: 0, maxValidAbsDiff: 0.02 },
              depthAndValidityImpact: {
                primarySourceMaskChanges: {
                  baselineValidToCandidateInvalid: 0,
                  baselineInvalidToCandidateValid: 0,
                  totalMaskFlips: 0,
                  validRegionMaxAbsDiff: 0.01,
                },
                finalDisparityStats: {},
              },
            },
            causalIntervenedGpuSuffixWasm7WithCand9VsWasmFull: {
              primary: { stats: { ...dummyStats, maxAbsDiff: 0.008 }, flippedPixelCount: 0, maxValidAbsDiff: 0.008 },
              secondary: { stats: { ...dummyStats, maxAbsDiff: 4.5 }, flippedPixelCount: 2, maxValidAbsDiff: 4.5 },
              depthAndValidityImpact: {
                primarySourceMaskChanges: {
                  baselineValidToCandidateInvalid: 0,
                  baselineInvalidToCandidateValid: 0,
                  totalMaskFlips: 0,
                  validRegionMaxAbsDiff: 0.008,
                },
                finalDisparityStats: {},
              },
            },
            causalIntervenedGpuSuffixWasm7WithCand9VsCausalBaselineGpuSuffix: {
              primary: { stats: dummyStats, flippedPixelCount: 0, maxValidAbsDiff: 0.01 },
              secondary: { stats: { ...dummyStats, maxAbsDiff: 4.4 }, flippedPixelCount: 2, maxValidAbsDiff: 4.4 },
            },
          }
        : null;

      return runHitnetF0126SecondaryOutlierDiagnostic({
        m1Bytes: modelBytes,
        f0126FeedFloat32: f0126,
        normalizeSlot0ReshapeToTranspose: false,
        existingCommonWasm7Summary: existingSummary as never,
        runWorker: buildSuffixRunner(),
        isDev: true,
      });
    };

    const repOrder = await runScenario('order-crossing', true);
    expect(repOrder.ok).toBe(true);
    expect(repOrder.aborted).toBe(false);
    expect(repOrder.gatePassed).toBe(true);
    expect(repOrder.gateTransportRecord.vsNativeSplitPrimary?.bitIdentical).toBe(true);
    expect(repOrder.gateTransportRecord.vsNativeSplitSecondary?.bitIdentical).toBe(true);
    expect(repOrder.gateTransportRecord.vsFullGpuDisabledPrimary?.bitIdentical).toBe(true);
    expect(repOrder.gateTransportRecord.vsFullGpuDisabledSecondary?.bitIdentical).toBe(true);
    expect(repOrder.outlierComparison?.cand128VsNativeOnCand.secondary.stats.maxAbsDiff).toBeCloseTo(8.844, 3);
    expect(repOrder.outlierComparison?.secondaryClusters[0]?.peak.ty).toBe(10);
    expect(repOrder.outlierComparison?.secondaryClusters[0]?.peak.tx).toBe(20);
    expect(repOrder.outlierComparison?.eyeValidity.rightSecondary.totalValidityFlipsVsNative).toBe(1);
    expect(repOrder.ancestryAndTaps?.coarseAnalysis?.tapPass1Invariance.bothArmsBitInvariant).toBe(true);
    expect(repOrder.ancestryAndTaps?.coarseAnalysis?.tapPass2Invariance?.bothArmsBitInvariant).toBe(true);
    expect(repOrder.ancestryAndTaps?.coarseAnalysis?.int64Comparison.mismatchCount).toBe(1);
    expect(repOrder.ancestryAndTaps?.coarseAnalysis?.exactCostOrderCrossingsCount).toBe(1);
    expect(repOrder.causalAttribution?.verdict).toBe('COARSE_ARGMIN_COST_ORDER_CROSSING');
    expect(repOrder.commonWasm7Secondary.source).toBe('reused-existing-summary');

    // Exact-tie scenario + fresh controlled F0126 Common-WASM-#7 execution
    const repTie = await runScenario('exact-tie', false);
    expect(repTie.ok).toBe(true);
    expect(repTie.gatePassed).toBe(true);
    expect(repTie.ancestryAndTaps?.coarseAnalysis?.exactTieTiebreakersCount).toBe(1);
    expect(repTie.causalAttribution?.verdict).toBe('COARSE_ARGMIN_EXACT_TIE_FLIP');
    expect(repTie.commonWasm7Secondary.source).toBe('fresh-controlled-f0126-arm');
    expect(repTie.commonWasm7Secondary.intervenedGpuSuffixWasm7WithCand9VsWasmFull?.secondary.maxAbsDiff).toBeGreaterThan(8.0);
  }, 45000);

  it('localizes POST_PROP0_REFINEMENT_AMPLIFICATION when coarse INT64 ArgMin has 0 mismatches and postProp0 stays small', async () => {
    const modelBytes = buildSyntheticM1SecondaryChainOnnx(
      HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength
    );
    const f0126 = new Float32Array(1 * 6 * 240 * 320).fill(0.45);
    mockShaSubtleForSyntheticM1(modelBytes, f0126);

    const parsed = await parseM1ModelProtoWithRawSlices(modelBytes);
    const chain = await resolveVerifiedPostConv7ChainSpec(parsed);
    const suffix9 = await buildVerifiedSuffixAfterConv7(parsed, {
      cutNodeIdx: chain.node9ConvDown0.nodeIdx,
    });
    const ancestry = resolveSecondaryOutlierAncestrySpec(parsed, chain, suffix9);

    const node8Data = new Float32Array(2 * 32 * 240 * 320).fill(0.15);
    const nativeGpu9Data = new Float32Array(2 * 40 * 120 * 160).fill(0.5);
    const nativeGpu9Sha = nodeSha256Hex(new Uint8Array(nativeGpu9Data.buffer));
    const f0126RawSha = nodeSha256Hex(new Uint8Array(f0126.buffer));

    const runner: ResidualDiagnosticWorkerRunner = async (req) => {
      const mBytes = new Uint8Array(req.modelBuffer);
      const isGpu = req.provider === 'webgpu';
      const isFullUntapped = mBytes.byteLength === modelBytes.byteLength;
      const topo = inspectHitnetM1ResidualLadderTopology(mBytes);
      const outNames = Array.from(topo.graphOutputNames);
      const fShaRaw = nodeSha256Hex(new Uint8Array(req.feedBuffer));
      const feedSha = fShaRaw === f0126RawSha ? HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256 : fShaRaw;

      if (req.enableWebGpuProfiling && isGpu) {
        return {
          ...makeMockSuccessBase({ provider: 'webgpu', graphOptimizationLevel: req.graphOptimizationLevel }),
          modelByteLength: mBytes.byteLength,
          modelSha256: isFullUntapped ? HITNET_M1_FULL_GRAPH_SHA256 : nodeSha256Hex(mBytes),
          feedSha256: feedSha,
          feedTensorName: req.feedName ?? 'input',
          feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
          feedTensorDataType: 'float32',
          outputs: [],
          gpuKernelProfileSummary: {
            profilingRequested: true,
            timestampQuerySupported: true,
            webgpuComputePassCount: 2,
            webgpuDispatchWorkgroupsCount: 16,
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
            note: 'mock',
          },
        };
      }
      if (req.allowArbitraryOutputs && req.feedName === chain.node8LeakyRelu.outputTensor) {
        return {
          ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: req.graphOptimizationLevel }),
          modelByteLength: mBytes.byteLength,
          modelSha256: nodeSha256Hex(mBytes),
          feedSha256: feedSha,
          feedTensorName: chain.node8LeakyRelu.outputTensor,
          feedTensorShape: [2, 32, 240, 320],
          feedTensorDataType: 'float32',
          outputs: [
            {
              name: chain.node9ConvDown0.outputTensor,
              dims: [2, 40, 120, 160],
              dataType: 'float32',
              buffer: nativeGpu9Data.slice().buffer,
            },
          ],
        };
      }
      if (outNames.includes(chain.node8LeakyRelu.outputTensor) && !outNames.includes('reference_output_disparity') && !outNames.includes(chain.conv7OutputTensor)) {
        return {
          ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: req.graphOptimizationLevel }),
          modelByteLength: mBytes.byteLength,
          modelSha256: nodeSha256Hex(mBytes),
          feedSha256: feedSha,
          feedTensorName: 'input',
          feedTensorShape: [1, 6, 240, 320],
          feedTensorDataType: 'float32',
          outputs: outNames.map((oName) =>
            oName === chain.node8LeakyRelu.outputTensor
              ? { name: oName, dims: [2, 32, 240, 320], dataType: 'float32', buffer: node8Data.slice().buffer }
              : { name: oName, dims: [1, 60, 25600], dataType: 'int32', buffer: new Int32Array(1 * 60 * 25600).fill(2).buffer }
          ),
        };
      }

      const isSuffix9 = req.feedName === chain.node9ConvDown0.outputTensor;
      const isCand9 = isSuffix9 && feedSha !== nativeGpu9Sha;
      const secOut = new Float32Array(76800).fill(5.0);
      if (isCand9) secOut[20 * 320 + 40] = 9.5; // +4.5 px in tile (ty=5, tx=10)

      const outputs: OrtVersionDiagnosticOutput[] = outNames.map((oName) => {
        if (oName === 'reference_output_disparity') {
          return { name: oName, dims: [1, 240, 320, 1], dataType: 'float32', buffer: new Float32Array(76800).fill(6.0).buffer };
        }
        if (oName === 'secondary_output_disparity') {
          return { name: oName, dims: [1, 240, 320, 1], dataType: 'float32', buffer: secOut.buffer };
        }
        if (oName === ancestry.coarseArgMin.outputTensor) {
          // ZERO coarse flips!
          return { name: oName, dims: [1, 60, 80], dataType: 'int64', buffer: new BigInt64Array(4800).fill(15n).buffer };
        }
        if (oName === ancestry.postProp0Out?.outputTensor) {
          // postProp0 stays small (1e-4 diff)
          const arr = new Float32Array(4800 * 16).fill(1.0);
          if (isCand9) arr[(5 * 80 + 10) * 16] = 1.0001;
          return { name: oName, dims: [1, 60, 80, 16], dataType: 'float32', buffer: arr.buffer };
        }
        if (oName === ancestry.refinementL1Out?.outputTensor) {
          // Amplification happens inside refinement_l1
          const arr = new Float32Array(240 * 320 * 16).fill(1.0);
          if (isCand9) arr[(20 * 320 + 40) * 16] = 5.5;
          return { name: oName, dims: [1, 240, 320, 16], dataType: 'float32', buffer: arr.buffer };
        }
        const dims = parsed.staticMeta.tensorMeta[oName]?.dims ?? [1, 60, 80, 16];
        return {
          name: oName,
          dims: [...dims],
          dataType: 'float32',
          buffer: new Float32Array(dims.reduce((a, b) => a * b, 1)).fill(0.5).buffer,
        };
      });

      return {
        ...makeMockSuccessBase({ provider: req.provider, graphOptimizationLevel: req.graphOptimizationLevel }),
        modelByteLength: mBytes.byteLength,
        modelSha256: isFullUntapped ? HITNET_M1_FULL_GRAPH_SHA256 : nodeSha256Hex(mBytes),
        feedSha256: feedSha,
        feedTensorName: req.feedName ?? 'input',
        feedTensorShape: req.feedDims ?? [1, 6, 240, 320],
        feedTensorDataType: 'float32',
        feedByteOffset: 0,
        feedByteLength: req.feedBuffer.byteLength,
        extraFeedChecks: req.extraFeeds?.map((ef) => ({
          name: ef.name,
          dims: [...ef.dims],
          dataType: ef.dataType ?? 'float32',
          byteOffset: 0,
          byteLength: ef.buffer.byteLength,
          sha256: nodeSha256Hex(new Uint8Array(ef.buffer)),
        })),
        sessionInputNames: isSuffix9
          ? [...suffix9.graphInputNames]
          : req.feedName === chain.conv7OutputTensor
            ? [chain.conv7OutputTensor, ...suffix9.bypassCutInputs.filter((b) => !b.name.includes('LeakyRelu')).map((b) => b.name)]
            : ['input'],
        outputs,
      };
    };

    const rep = await runHitnetF0126SecondaryOutlierDiagnostic({
      m1Bytes: modelBytes,
      f0126FeedFloat32: f0126,
      normalizeSlot0ReshapeToTranspose: false,
      runWorker: runner,
      isDev: true,
    });

    expect(rep.ok).toBe(true);
    expect(rep.gatePassed).toBe(true);
    expect(rep.ancestryAndTaps?.coarseAnalysis?.int64Comparison.mismatchCount).toBe(0);
    expect(rep.ancestryAndTaps?.lateStageTaps.length).toBeGreaterThanOrEqual(2);
    expect(rep.ancestryAndTaps?.lateStageTaps.every((lt) => lt.invariance.bothArmsBitInvariant)).toBe(true);
    expect(rep.causalAttribution?.verdict).toBe('POST_PROP0_REFINEMENT_AMPLIFICATION');
  }, 45000);
});
