// @vitest-environment node
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import * as ort from 'onnxruntime-web';
import {
  HITNET_H1_FULL_GRAPH_SHA256,
  HITNET_LEVEL_INIT_REDUCEMIN_LIMITATIONS,
  HITNET_M1_FULL_GRAPH_SHA256,
  HITNET_RAW_MODEL_SHA256,
  appendOnnxGraphOutputs,
  computeOrtArgMinCacheAttributesKey,
  decomposeOnnx5dConvNodesForWebGpu,
  inferOnnxStaticTensorMeta,
  normalizeHitnetArgMinKeepDimsForWebGpu,
  normalizeHitnetGatherNdRankForWebGpu,
  normalizeHitnetInitToPropZeroSlantForWebGpu,
  normalizeHitnetLevelInitGatherNdRankForWebGpu,
  normalizeHitnetLevelInitReduceMinForWebGpu,
  normalizeHitnetPropGatherNdRankForWebGpu,
  normalizeHitnetSlot0ReshapeToTransposeForWebGpu,
  normalizeNonAscendingSliceAxesForWebGpu,
  validateOnnxGraphNodeInputs,
} from './stereoDepthWorker';
import {
  buildSyntheticArgMinControlModels,
  buildSyntheticGatherNdControlModels,
  buildSyntheticPropGatherNdControlModels,
  prepareHitnetFullGraphTapGroup,
  prepareHitnetLevelInitCandidateFullGraphBytes,
  prepareHitnetM1FullGraphBytes,
  prepareHitnetN1FullGraphBytes,
  prepareHitnetSubgraphArms,
  summarizeExactInt64IndexComparison,
  summarizeInitToPropConcat1OutSplit,
  summarizeInitToPropConcatOutSplit,
  summarizeP2bChannelwiseParity,
  summarizeRawTensorAndParity,
  verifySingletonZeroIndices,
} from './hitnetGatherNdDiagnostic';

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

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

function encodeVarintBigInt(value: bigint): Uint8Array {
  const bytes: number[] = [];
  let v = BigInt.asUintN(64, value);
  while (v >= 0x80n) {
    bytes.push(Number(v & 0x7fn) | 0x80);
    v >>= 7n;
  }
  bytes.push(Number(v));
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

function encodeVarintBigIntField(fn: number, val: bigint): Uint8Array {
  return concatBytes([encodeVarint((fn << 3) | 0), encodeVarintBigInt(val)]);
}

function encodeStrField(fn: number, value: string): Uint8Array {
  return encodeLenField(fn, textEncoder.encode(value));
}

function encodeAttrInt(name: string, val: number): Uint8Array {
  return encodeLenField(
    5,
    concatBytes([encodeStrField(1, name), encodeVarintBigIntField(3, BigInt(val)), encodeVarintField(20, 2)])
  );
}

function encodeAttrInts(name: string, vals: readonly number[]): Uint8Array {
  return encodeLenField(
    5,
    concatBytes([
      encodeStrField(1, name),
      ...vals.map((v) => encodeVarintBigIntField(8, BigInt(v))),
      encodeVarintField(20, 7),
    ])
  );
}

function encodeValueInfoTensor(
  name: string,
  elemType: number,
  dims: readonly (number | null)[]
): Uint8Array {
  const dimFields = dims.map((d) =>
    encodeLenField(1, d === null ? new Uint8Array(0) : encodeVarintField(1, d))
  );
  const shapeProto = encodeLenField(2, concatBytes(dimFields));
  const tensorTypeProto = encodeLenField(1, concatBytes([encodeVarintField(1, elemType), shapeProto]));
  return concatBytes([encodeStrField(1, name), encodeLenField(2, tensorTypeProto)]);
}

function encodeFloatTensorProto(name: string, dims: readonly number[], values: Float32Array): Uint8Array {
  const raw = new Uint8Array(values.length * 4);
  const dv = new DataView(raw.buffer);
  for (let i = 0; i < values.length; i++) {
    dv.setFloat32(i * 4, values[i], true);
  }
  return concatBytes([
    ...dims.map((d) => encodeVarintField(1, d)),
    encodeVarintField(2, 1), // FLOAT
    encodeStrField(8, name),
    encodeLenField(9, raw),
  ]);
}

type IntEncoding = 'raw_data' | 'typed_field';

function encodeIntTensorProto(
  name: string,
  dims: readonly number[],
  dataType: 6 | 7, // INT32 = 6, INT64 = 7
  values: readonly (number | bigint)[],
  encoding: IntEncoding = 'raw_data'
): Uint8Array {
  const bigVals = values.map((v) => BigInt(v));
  const parts: Uint8Array[] = [
    ...dims.map((d) => encodeVarintField(1, d)),
    encodeVarintField(2, dataType),
    encodeStrField(8, name),
  ];
  if (encoding === 'raw_data') {
    const elemSize = dataType === 7 ? 8 : 4;
    const raw = new Uint8Array(bigVals.length * elemSize);
    const dv = new DataView(raw.buffer);
    for (let i = 0; i < bigVals.length; i++) {
      if (dataType === 7) {
        dv.setBigInt64(i * 8, bigVals[i], true);
      } else {
        dv.setInt32(i * 4, Number(bigVals[i]), true);
      }
    }
    parts.push(encodeLenField(9, raw));
  } else if (dataType === 7) {
    for (const v of bigVals) {
      parts.push(encodeVarintBigIntField(7, v));
    }
  } else {
    for (const v of bigVals) {
      parts.push(encodeVarintBigIntField(5, BigInt.asIntN(32, v)));
    }
  }
  return concatBytes(parts);
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

function buildOnnxModel(
  graphParts: readonly Uint8Array[],
  opsetVersion = 12
): Uint8Array {
  return concatBytes([
    encodeVarintField(1, 7),
    encodeLenField(7, concatBytes(graphParts)),
    encodeLenField(8, concatBytes([encodeStrField(1, ''), encodeVarintField(2, opsetVersion)])),
  ]);
}

function readVarintProto(buf: Uint8Array, pos: number, end: number): [number, number] | null {
  let result = 0;
  let shift = 0;
  let cursor = pos;
  while (cursor < end && shift <= 35) {
    const byte = buf[cursor++];
    result |= (byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) return [result >>> 0, cursor];
    shift += 7;
  }
  return null;
}

function parseProto(buf: Uint8Array, start: number, end: number) {
  const fields: Array<{ fn: number; wt: number; valStart: number; valEnd: number }> = [];
  let cursor = start;
  while (cursor < end) {
    const key = readVarintProto(buf, cursor, end);
    if (!key) break;
    const fn = key[0] >>> 3;
    const wt = key[0] & 7;
    cursor = key[1];
    if (wt === 0) {
      let c = cursor;
      while (c < end && (buf[c] & 0x80) !== 0) c++;
      c++;
      fields.push({ fn, wt, valStart: cursor, valEnd: c });
      cursor = c;
    } else if (wt === 2) {
      const len = readVarintProto(buf, cursor, end);
      if (!len) break;
      fields.push({ fn, wt, valStart: len[1], valEnd: len[1] + len[0] });
      cursor = len[1] + len[0];
    } else if (wt === 5) {
      fields.push({ fn, wt, valStart: cursor, valEnd: cursor + 4 });
      cursor += 4;
    } else if (wt === 1) {
      fields.push({ fn, wt, valStart: cursor, valEnd: cursor + 8 });
      cursor += 8;
    } else {
      break;
    }
  }
  return fields;
}

function inspectNodesAndIntInits(modelBytes: Uint8Array) {
  const top = parseProto(modelBytes, 0, modelBytes.byteLength);
  const gf = top.find((f) => f.fn === 7 && f.wt === 2)!;
  const gFields = parseProto(modelBytes, gf.valStart, gf.valEnd);

  const nodes: Array<{ name: string; opType: string; inputs: string[]; outputs: string[] }> = [];
  const inits = new Map<string, bigint[]>();

  for (const field of gFields) {
    if (field.fn === 1 && field.wt === 2) {
      const nf = parseProto(modelBytes, field.valStart, field.valEnd);
      const inputs: string[] = [];
      const outputs: string[] = [];
      let name = '';
      let opType = '';
      for (const f of nf) {
        const str = textDecoder.decode(modelBytes.subarray(f.valStart, f.valEnd));
        if (f.fn === 1 && f.wt === 2) inputs.push(str);
        else if (f.fn === 2 && f.wt === 2) outputs.push(str);
        else if (f.fn === 3 && f.wt === 2) name = str;
        else if (f.fn === 4 && f.wt === 2) opType = str;
      }
      nodes.push({ name, opType, inputs, outputs });
    } else if (field.fn === 5 && field.wt === 2) {
      const tf = parseProto(modelBytes, field.valStart, field.valEnd);
      let name = '';
      let dataType = 0;
      let raw: Uint8Array | null = null;
      for (const f of tf) {
        if (f.fn === 2 && f.wt === 0) dataType = readVarintProto(modelBytes, f.valStart, f.valEnd)![0];
        else if (f.fn === 8 && f.wt === 2) name = textDecoder.decode(modelBytes.subarray(f.valStart, f.valEnd));
        else if (f.fn === 9 && f.wt === 2) raw = modelBytes.subarray(f.valStart, f.valEnd);
      }
      if (name && raw && (dataType === 6 || dataType === 7)) {
        const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
        const vals: bigint[] = [];
        if (dataType === 7) {
          for (let i = 0; i < raw.byteLength / 8; i++) vals.push(dv.getBigInt64(i * 8, true));
        } else {
          for (let i = 0; i < raw.byteLength / 4; i++) vals.push(BigInt(dv.getInt32(i * 4, true)));
        }
        inits.set(name, vals);
      }
    }
  }
  return { nodes, inits };
}

describe('normalizeNonAscendingSliceAxesForWebGpu', () => {
  ort.env.wasm.numThreads = 1;

  it('permutes (starts, ends, axes) together for HITNet axes=[0,2,3,1] pattern into 2D Conv with exact ORT parity', async () => {
    const H = 4;
    const W = 5;
    const xData = new Float32Array(1 * 6 * H * W);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = ((i * 13 + 7) % 29 - 14) * 0.08;
    }
    const wData = new Float32Array(4 * 3 * 3 * 3);
    for (let i = 0; i < wData.length; i++) {
      wData[i] = ((i * 5 + 1) % 17 - 8) * 0.07;
    }
    const bData = new Float32Array([0.1, -0.2, 0.3, -0.4]);

    // HITNet stem:
    // strided_slice_1: x[:, 0:3, :, :] via axes=[0, 2, 3, 1]
    // strided_slice:   x[:, 3:6, :, :] via axes=[0, 2, 3, 1]
    // Concat(axis=0) -> [2, 3, H, W] -> Conv([4, 3, 3, 3]) -> [2, 4, H, W]
    const INT64_MAX = 9223372036854775807n;
    const modelBytes = buildOnnxModel([
      encodeNode('Slice', 'strided_slice_1', ['x', 'starts_l', 'ends_l', 'axes_unsorted'], ['rgb_left']),
      encodeNode('Slice', 'strided_slice', ['x', 'starts_r', 'ends_r', 'axes_unsorted'], ['rgb_right']),
      encodeNode('Concat', 'concat_lr', ['rgb_left', 'rgb_right'], ['batched'], [encodeAttrInt('axis', 0)]),
      encodeNode(
        'Conv',
        'fe_conv_0',
        ['batched', 'conv_w', 'conv_b'],
        ['y'],
        [
          encodeAttrInts('kernel_shape', [3, 3]),
          encodeAttrInts('strides', [1, 1]),
          encodeAttrInts('pads', [1, 1, 1, 1]),
        ]
      ),
      encodeStrField(2, 'hitnet_slice_conv_pattern'),
      encodeLenField(5, encodeIntTensorProto('starts_l', [4], 7, [0, 0, 0, 0])),
      encodeLenField(5, encodeIntTensorProto('ends_l', [4], 7, [INT64_MAX, INT64_MAX, INT64_MAX, 3])),
      encodeLenField(5, encodeIntTensorProto('starts_r', [4], 7, [0, 0, 0, 3])),
      encodeLenField(5, encodeIntTensorProto('ends_r', [4], 7, [INT64_MAX, INT64_MAX, INT64_MAX, 6])),
      encodeLenField(5, encodeIntTensorProto('axes_unsorted', [4], 7, [0, 2, 3, 1])),
      encodeLenField(5, encodeFloatTensorProto('conv_w', [4, 3, 3, 3], wData)),
      encodeLenField(5, encodeFloatTensorProto('conv_b', [4], bData)),
      encodeLenField(11, encodeValueInfoTensor('x', 1, [1, 6, H, W])),
      encodeLenField(12, encodeValueInfoTensor('rgb_left', 1, [1, 3, H, W])),
      encodeLenField(12, encodeValueInfoTensor('rgb_right', 1, [1, 3, H, W])),
      encodeLenField(12, encodeValueInfoTensor('y', 1, [2, 4, H, W])),
    ]);

    const norm = normalizeNonAscendingSliceAxesForWebGpu(modelBytes);
    expect(norm.normalizedCount).toBe(2);
    expect(norm.normalizedNodeNames).toEqual(['strided_slice_1', 'strided_slice']);
    expect(norm.skippedDynamicCount).toBe(0);
    expect(norm.skippedInvalidOrUnprovenCount).toBe(0);
    expect(validateOnnxGraphNodeInputs(norm.bytes)).toMatchObject({ valid: true, missingInputs: [] });

    const inspected = inspectNodesAndIntInits(norm.bytes);
    const s1 = inspected.nodes.find((n) => n.name === 'strided_slice_1')!;
    const s2 = inspected.nodes.find((n) => n.name === 'strided_slice')!;
    expect(inspected.inits.get(s1.inputs[3])).toEqual([0n, 1n, 2n, 3n]);
    expect(inspected.inits.get(s1.inputs[1])).toEqual([0n, 0n, 0n, 0n]);
    expect(inspected.inits.get(s1.inputs[2])).toEqual([INT64_MAX, 3n, INT64_MAX, INT64_MAX]);
    expect(inspected.inits.get(s2.inputs[3])).toEqual([0n, 1n, 2n, 3n]);
    expect(inspected.inits.get(s2.inputs[1])).toEqual([0n, 3n, 0n, 0n]);
    expect(inspected.inits.get(s2.inputs[2])).toEqual([INT64_MAX, 6n, INT64_MAX, INT64_MAX]);

    const origSess = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      logSeverityLevel: 3,
    });
    const normSess = await ort.InferenceSession.create(norm.bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3,
    });
    try {
      const xTensor = new ort.Tensor('float32', xData, [1, 6, H, W]);
      const origOut = await origSess.run({ x: xTensor });
      const normOut = await normSess.run({ x: xTensor });

      for (const outName of ['rgb_left', 'rgb_right', 'y'] as const) {
        expect(normOut[outName].dims).toEqual(origOut[outName].dims);
        const a = origOut[outName].data as Float32Array;
        const b = normOut[outName].data as Float32Array;
        expect(b.length).toBe(a.length);
        let maxAbsDiff = 0;
        for (let i = 0; i < a.length; i++) {
          maxAbsDiff = Math.max(maxAbsDiff, Math.abs(a[i] - b[i]));
        }
        expect(maxAbsDiff, outName).toBe(0);
      }
    } finally {
      await origSess.release();
      await normSess.release();
    }
  });

  it('handles subset axes, INT32 vs INT64 raw_data and typed protobuf fields, and optional positive/negative/empty steps with exact ORT parity', async () => {
    const cases: Array<{
      label: string;
      dataType: 6 | 7;
      encoding: IntEncoding;
      starts: number[];
      ends: number[];
      axes: number[];
      steps?: number[] | 'empty';
    }> = [
      {
        label: 'subset [3,1] INT64 raw_data without steps',
        dataType: 7,
        encoding: 'raw_data',
        starts: [1, 2],
        ends: [4, 5],
        axes: [3, 1],
      },
      {
        label: '4D [2,0,3,1] INT32 typed_field with empty steps slot',
        dataType: 6,
        encoding: 'typed_field',
        starts: [1, 0, 0, 1],
        ends: [3, 2, 4, 4],
        axes: [2, 0, 3, 1],
        steps: 'empty',
      },
      {
        label: '4D [3,1,0,2] INT64 typed_field with positive and negative steps',
        dataType: 7,
        encoding: 'typed_field',
        starts: [3, 4, 0, 1],
        ends: [0, 0, 2, 4],
        axes: [3, 1, 0, 2],
        steps: [-1, -2, 1, 2],
      },
      {
        label: '4D [2,3,1,0] INT32 raw_data with mixed steps',
        dataType: 6,
        encoding: 'raw_data',
        starts: [0, 4, 1, 1],
        ends: [4, 0, 5, 2],
        axes: [2, 3, 1, 0],
        steps: [2, -1, 2, 1],
      },
    ];

    const xDims = [2, 5, 4, 5];
    const xData = new Float32Array(2 * 5 * 4 * 5);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = i * 0.25 - 10;
    }

    for (const c of cases) {
      const inputs =
        c.steps === 'empty'
          ? ['x', 's', 'e', 'a', '']
          : c.steps
            ? ['x', 's', 'e', 'a', 'st']
            : ['x', 's', 'e', 'a'];
      const graphParts: Uint8Array[] = [
        encodeNode('Slice', 'slice_0', inputs, ['y']),
        encodeStrField(2, `case_${c.label}`),
        encodeLenField(5, encodeIntTensorProto('s', [c.starts.length], c.dataType, c.starts, c.encoding)),
        encodeLenField(5, encodeIntTensorProto('e', [c.ends.length], c.dataType, c.ends, c.encoding)),
        encodeLenField(5, encodeIntTensorProto('a', [c.axes.length], c.dataType, c.axes, c.encoding)),
      ];
      if (Array.isArray(c.steps)) {
        graphParts.push(
          encodeLenField(5, encodeIntTensorProto('st', [c.steps.length], c.dataType, c.steps, c.encoding))
        );
      }
      graphParts.push(
        encodeLenField(11, encodeValueInfoTensor('x', 1, xDims)),
        encodeLenField(12, encodeValueInfoTensor('y', 1, [null, null, null, null]))
      );

      const modelBytes = buildOnnxModel(graphParts, 12);
      const norm = normalizeNonAscendingSliceAxesForWebGpu(modelBytes);
      expect(norm.normalizedCount, c.label).toBe(1);
      expect(validateOnnxGraphNodeInputs(norm.bytes), c.label).toMatchObject({
        valid: true,
        missingInputs: [],
      });

      const inspected = inspectNodesAndIntInits(norm.bytes);
      const node = inspected.nodes[0];
      if (c.steps === 'empty') {
        expect(node.inputs[4], c.label).toBe('');
      }
      const sortedAxes = inspected.inits.get(node.inputs[3])!;
      for (let i = 1; i < sortedAxes.length; i++) {
        expect(sortedAxes[i] > sortedAxes[i - 1], c.label).toBe(true);
      }

      const origSess = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'disabled',
        logSeverityLevel: 3,
      });
      const normSess = await ort.InferenceSession.create(norm.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        logSeverityLevel: 3,
      });
      try {
        const xTensor = new ort.Tensor('float32', xData, xDims);
        const origOut = (await origSess.run({ x: xTensor })).y;
        const normOut = (await normSess.run({ x: xTensor })).y;
        expect(normOut.dims, c.label).toEqual(origOut.dims);
        expect(Array.from(normOut.data as Float32Array), c.label).toEqual(
          Array.from(origOut.data as Float32Array)
        );
      } finally {
        await origSess.release();
        await normSess.release();
      }
    }
  });

  it('never mutates shared initializers or affects unrelated consumers and unrewritten Slices', async () => {
    // Reproduces HITNet initializer sharing:
    // - shared_starts: used by unsorted Slice([0,2,3,1]), already-sorted Slice([0,1,2,3]), AND unrelated Mul consumer (via Cast->Mul)!
    // - shared_ends: [100, 100, 100, 2] shared by BOTH unsorted Slice([0,2,3,1]) and sorted Slice([0,1,2,3])!
    // If mutated in place, sorted Slice or unrelated consumer would break.
    const xDims = [1, 4, 3, 4];
    const xData = new Float32Array(1 * 4 * 3 * 4).map((_, i) => i + 1);

    const modelBytes = buildOnnxModel([
      encodeNode('Slice', 'slice_unsorted', ['x', 'shared_starts', 'shared_ends', 'axes_unsorted'], ['y_unsorted']),
      encodeNode('Slice', 'slice_sorted', ['x', 'shared_starts', 'shared_ends', 'axes_sorted'], ['y_sorted']),
      encodeNode('Cast', 'cast_starts', ['shared_starts'], ['starts_f32'], [encodeAttrInt('to', 1)]),
      encodeStrField(2, 'shared_init_safety_graph'),
      encodeLenField(5, encodeIntTensorProto('shared_starts', [4], 7, [0, 1, 0, 0])),
      encodeLenField(5, encodeIntTensorProto('shared_ends', [4], 7, [1, 2, 3, 2])),
      encodeLenField(5, encodeIntTensorProto('axes_unsorted', [4], 7, [0, 2, 3, 1])),
      encodeLenField(5, encodeIntTensorProto('axes_sorted', [4], 7, [0, 1, 2, 3])),
      encodeLenField(11, encodeValueInfoTensor('x', 1, xDims)),
      encodeLenField(12, encodeValueInfoTensor('y_unsorted', 1, [null, null, null, null])),
      encodeLenField(12, encodeValueInfoTensor('y_sorted', 1, [null, null, null, null])),
      encodeLenField(12, encodeValueInfoTensor('starts_f32', 1, [4])),
    ]);

    const norm = normalizeNonAscendingSliceAxesForWebGpu(modelBytes);
    expect(norm.normalizedCount).toBe(1);
    expect(norm.normalizedNodeNames).toEqual(['slice_unsorted']);
    expect(validateOnnxGraphNodeInputs(norm.bytes)).toMatchObject({ valid: true, missingInputs: [] });

    const inspected = inspectNodesAndIntInits(norm.bytes);
    // Shared initializers must keep exact original values
    expect(inspected.inits.get('shared_starts')).toEqual([0n, 1n, 0n, 0n]);
    expect(inspected.inits.get('shared_ends')).toEqual([1n, 2n, 3n, 2n]);

    // Sorted Slice and unrelated Cast must still reference original shared initializers
    const sortedNode = inspected.nodes.find((n) => n.name === 'slice_sorted')!;
    const castNode = inspected.nodes.find((n) => n.name === 'cast_starts')!;
    expect(sortedNode.inputs).toEqual(['x', 'shared_starts', 'shared_ends', 'axes_sorted']);
    expect(castNode.inputs).toEqual(['shared_starts']);

    // Rewritten unsorted Slice points to fresh __webgpu_slice_* initializers
    const rewrittenNode = inspected.nodes.find((n) => n.name === 'slice_unsorted')!;
    expect(rewrittenNode.inputs[1]).not.toBe('shared_starts');
    expect(rewrittenNode.inputs[2]).not.toBe('shared_ends');
    expect(inspected.inits.get(rewrittenNode.inputs[3])).toEqual([0n, 1n, 2n, 3n]);
    expect(inspected.inits.get(rewrittenNode.inputs[1])).toEqual([0n, 0n, 1n, 0n]);
    expect(inspected.inits.get(rewrittenNode.inputs[2])).toEqual([1n, 2n, 2n, 3n]);

    // Real ORT WASM execution verifies all 3 outputs are identical
    const origSess = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      logSeverityLevel: 3,
    });
    const normSess = await ort.InferenceSession.create(norm.bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3,
    });
    try {
      const xTensor = new ort.Tensor('float32', xData, xDims);
      const origRes = await origSess.run({ x: xTensor });
      const normRes = await normSess.run({ x: xTensor });
      for (const outName of ['y_unsorted', 'y_sorted', 'starts_f32'] as const) {
        expect(normRes[outName].dims).toEqual(origRes[outName].dims);
        expect(Array.from(normRes[outName].data as Float32Array), outName).toEqual(
          Array.from(origRes[outName].data as Float32Array)
        );
      }
    } finally {
      await origSess.release();
      await normSess.release();
    }
  });

  it('safely skips dynamic, negative/unproven, duplicate, mismatched, zero-step, non-1D, opset<10, and already-sorted Slice nodes', () => {
    const cases: Array<{
      label: string;
      modelBytes: Uint8Array;
      expectedDynamic: number;
      expectedInvalid: number;
    }> = [
      {
        label: 'dynamic starts from graph input',
        modelBytes: buildOnnxModel([
          encodeNode('Slice', 'dyn_starts', ['x', 'dyn_s', 'e', 'a'], ['y']),
          encodeStrField(2, 'g'),
          encodeLenField(5, encodeIntTensorProto('e', [4], 7, [1, 1, 1, 1])),
          encodeLenField(5, encodeIntTensorProto('a', [4], 7, [0, 2, 3, 1])),
          encodeLenField(11, encodeValueInfoTensor('x', 1, [1, 2, 2, 2])),
          encodeLenField(11, encodeValueInfoTensor('dyn_s', 7, [4])),
          encodeLenField(12, encodeValueInfoTensor('y', 1, [1, 1, 1, 1])),
        ]),
        expectedDynamic: 1,
        expectedInvalid: 0,
      },
      {
        label: 'dynamic axes from Constant/Add node output',
        modelBytes: buildOnnxModel([
          encodeNode('Identity', 'id_axes', ['a_const'], ['dyn_a']),
          encodeNode('Slice', 'dyn_axes_slice', ['x', 's', 'e', 'dyn_a'], ['y']),
          encodeStrField(2, 'g'),
          encodeLenField(5, encodeIntTensorProto('s', [4], 7, [0, 0, 0, 0])),
          encodeLenField(5, encodeIntTensorProto('e', [4], 7, [1, 1, 1, 1])),
          encodeLenField(5, encodeIntTensorProto('a_const', [4], 7, [0, 2, 3, 1])),
          encodeLenField(11, encodeValueInfoTensor('x', 1, [1, 2, 2, 2])),
          encodeLenField(12, encodeValueInfoTensor('y', 1, [1, 1, 1, 1])),
        ]),
        expectedDynamic: 1,
        expectedInvalid: 0,
      },
      {
        label: 'negative/unproven axis [0,-1,2,1]',
        modelBytes: buildOnnxModel([
          encodeNode('Slice', 'neg_axis', ['x', 's', 'e', 'a'], ['y']),
          encodeStrField(2, 'g'),
          encodeLenField(5, encodeIntTensorProto('s', [4], 7, [0, 0, 0, 0])),
          encodeLenField(5, encodeIntTensorProto('e', [4], 7, [1, 1, 1, 1])),
          encodeLenField(5, encodeIntTensorProto('a', [4], 7, [0, -1, 2, 1])),
          encodeLenField(11, encodeValueInfoTensor('x', 1, [1, 2, 2, 2])),
          encodeLenField(12, encodeValueInfoTensor('y', 1, [1, 1, 1, 1])),
        ]),
        expectedDynamic: 0,
        expectedInvalid: 1,
      },
      {
        label: 'duplicate axis [0,2,2,1]',
        modelBytes: buildOnnxModel([
          encodeNode('Slice', 'dup_axis', ['x', 's', 'e', 'a'], ['y']),
          encodeStrField(2, 'g'),
          encodeLenField(5, encodeIntTensorProto('s', [4], 7, [0, 0, 0, 0])),
          encodeLenField(5, encodeIntTensorProto('e', [4], 7, [1, 1, 1, 1])),
          encodeLenField(5, encodeIntTensorProto('a', [4], 7, [0, 2, 2, 1])),
          encodeLenField(11, encodeValueInfoTensor('x', 1, [1, 2, 2, 2])),
          encodeLenField(12, encodeValueInfoTensor('y', 1, [1, 1, 1, 1])),
        ]),
        expectedDynamic: 0,
        expectedInvalid: 1,
      },
      {
        label: 'length mismatch starts(3) vs axes(4)',
        modelBytes: buildOnnxModel([
          encodeNode('Slice', 'len_mismatch', ['x', 's', 'e', 'a'], ['y']),
          encodeStrField(2, 'g'),
          encodeLenField(5, encodeIntTensorProto('s', [3], 7, [0, 0, 0])),
          encodeLenField(5, encodeIntTensorProto('e', [4], 7, [1, 1, 1, 1])),
          encodeLenField(5, encodeIntTensorProto('a', [4], 7, [0, 2, 3, 1])),
          encodeLenField(11, encodeValueInfoTensor('x', 1, [1, 2, 2, 2])),
          encodeLenField(12, encodeValueInfoTensor('y', 1, [1, 1, 1, 1])),
        ]),
        expectedDynamic: 0,
        expectedInvalid: 1,
      },
      {
        label: 'zero step',
        modelBytes: buildOnnxModel([
          encodeNode('Slice', 'zero_step', ['x', 's', 'e', 'a', 'st'], ['y']),
          encodeStrField(2, 'g'),
          encodeLenField(5, encodeIntTensorProto('s', [4], 7, [0, 0, 0, 0])),
          encodeLenField(5, encodeIntTensorProto('e', [4], 7, [1, 1, 1, 1])),
          encodeLenField(5, encodeIntTensorProto('a', [4], 7, [0, 2, 3, 1])),
          encodeLenField(5, encodeIntTensorProto('st', [4], 7, [1, 0, 1, 1])),
          encodeLenField(11, encodeValueInfoTensor('x', 1, [1, 2, 2, 2])),
          encodeLenField(12, encodeValueInfoTensor('y', 1, [1, 1, 1, 1])),
        ]),
        expectedDynamic: 0,
        expectedInvalid: 1,
      },
      {
        label: 'non-1D axes initializer [2,2]',
        modelBytes: buildOnnxModel([
          encodeNode('Slice', 'non_1d', ['x', 's', 'e', 'a'], ['y']),
          encodeStrField(2, 'g'),
          encodeLenField(5, encodeIntTensorProto('s', [4], 7, [0, 0, 0, 0])),
          encodeLenField(5, encodeIntTensorProto('e', [4], 7, [1, 1, 1, 1])),
          encodeLenField(5, encodeIntTensorProto('a', [2, 2], 7, [0, 2, 3, 1])),
          encodeLenField(11, encodeValueInfoTensor('x', 1, [1, 2, 2, 2])),
          encodeLenField(12, encodeValueInfoTensor('y', 1, [1, 1, 1, 1])),
        ]),
        expectedDynamic: 0,
        expectedInvalid: 1,
      },
      {
        label: 'already ascending axes [0,1,2,3]',
        modelBytes: buildOnnxModel([
          encodeNode('Slice', 'already_sorted', ['x', 's', 'e', 'a'], ['y']),
          encodeStrField(2, 'g'),
          encodeLenField(5, encodeIntTensorProto('s', [4], 7, [0, 0, 0, 0])),
          encodeLenField(5, encodeIntTensorProto('e', [4], 7, [1, 1, 1, 1])),
          encodeLenField(5, encodeIntTensorProto('a', [4], 7, [0, 1, 2, 3])),
          encodeLenField(11, encodeValueInfoTensor('x', 1, [1, 2, 2, 2])),
          encodeLenField(12, encodeValueInfoTensor('y', 1, [1, 1, 1, 1])),
        ]),
        expectedDynamic: 0,
        expectedInvalid: 0,
      },
      {
        label: 'opset 9 Slice skipped cleanly',
        modelBytes: buildOnnxModel(
          [
            encodeNode('Slice', 'opset9_slice', ['x', 's', 'e', 'a'], ['y']),
            encodeStrField(2, 'g'),
            encodeLenField(5, encodeIntTensorProto('s', [4], 7, [0, 0, 0, 0])),
            encodeLenField(5, encodeIntTensorProto('e', [4], 7, [1, 1, 1, 1])),
            encodeLenField(5, encodeIntTensorProto('a', [4], 7, [0, 2, 3, 1])),
          ],
          9
        ),
        expectedDynamic: 0,
        expectedInvalid: 0,
      },
    ];

    for (const c of cases) {
      const res = normalizeNonAscendingSliceAxesForWebGpu(c.modelBytes);
      expect(res.normalizedCount, c.label).toBe(0);
      expect(res.bytes, c.label).toBe(c.modelBytes);
      expect(res.skippedDynamicCount, c.label).toBe(c.expectedDynamic);
      expect(res.skippedInvalidOrUnprovenCount, c.label).toBe(c.expectedInvalid);
    }
  });

  it('compacts proven identity axes in compact-active mode while keeping active axes ordered and matching ORT output', async () => {
    const INT32_MAX = 2147483647;
    const xDims = [1, 6, 4, 5];
    const xData = new Float32Array(1 * 6 * 4 * 5).map((_, i) => (i % 19) * 0.25 - 1.5);

    // Reproduces all 3 HITNet 4D Slice shapes:
    // 1) stem channel slice: axes=[0,2,3,1], ends=[INT32_MAX, 4, 5, 3] -> active [1] (channel)
    // 2) spatial crop (#31 style): axes=[0,2,3,1], ends=[INT32_MAX, 2, 3, INT32_MAX] -> active [2,3]
    // 3) tail already-sorted channel slice (#193 style): axes=[0,1,2,3], ends=[INT32_MAX, INT32_MAX, INT32_MAX, 2] -> active [3]
    // 4) step=[1,1,-1,1] blocks identity compaction on axis 3 -> active [2,3] when start=0,end=INT32_MAX,step=-1
    const modelBytes = buildOnnxModel([
      encodeNode('Slice', 'stem_ch', ['x', 's4', 'e_ch', 'a_unsorted'], ['y_ch']),
      encodeNode('Slice', 'crop_hw', ['y_ch', 's4', 'e_crop', 'a_unsorted'], ['y_crop']),
      encodeNode('Slice', 'tail_sorted', ['y_crop', 's4', 'e_tail', 'a_sorted'], ['y_tail']),
      encodeNode('Slice', 'step_neg_preserve', ['y_ch', 's4', 'e_crop', 'a_unsorted', 'steps_neg'], ['y_neg']),
      encodeStrField(2, 'compact_active_graph'),
      encodeLenField(5, encodeIntTensorProto('s4', [4], 6, [0, 0, 0, 0])),
      encodeLenField(5, encodeIntTensorProto('e_ch', [4], 6, [INT32_MAX, 4, 5, 3])),
      encodeLenField(5, encodeIntTensorProto('e_crop', [4], 6, [INT32_MAX, 2, 3, INT32_MAX])),
      encodeLenField(5, encodeIntTensorProto('e_tail', [4], 6, [INT32_MAX, INT32_MAX, INT32_MAX, 2])),
      encodeLenField(5, encodeIntTensorProto('a_unsorted', [4], 6, [0, 2, 3, 1])),
      encodeLenField(5, encodeIntTensorProto('a_sorted', [4], 6, [0, 1, 2, 3])),
      encodeLenField(5, encodeIntTensorProto('steps_neg', [4], 6, [1, 1, -1, 1])),
      encodeLenField(11, encodeValueInfoTensor('x', 1, xDims)),
      encodeLenField(12, encodeValueInfoTensor('y_tail', 1, [null, null, null, null])),
      encodeLenField(12, encodeValueInfoTensor('y_neg', 1, [null, null, null, null])),
    ]);

    const res = normalizeNonAscendingSliceAxesForWebGpu(modelBytes, { mode: 'compact-active' });
    expect(res.mode).toBe('compact-active');
    expect(res.normalizedCount).toBe(4);
    expect(res.compactedCount).toBe(4);
    expect(res.normalizedNodeNames).toEqual(['stem_ch', 'crop_hw', 'tail_sorted', 'step_neg_preserve']);
    expect(validateOnnxGraphNodeInputs(res.bytes)).toMatchObject({ valid: true, missingInputs: [] });

    const inspected = inspectNodesAndIntInits(res.bytes);
    const stem = inspected.nodes.find((n) => n.name === 'stem_ch')!;
    const crop = inspected.nodes.find((n) => n.name === 'crop_hw')!;
    const tail = inspected.nodes.find((n) => n.name === 'tail_sorted')!;
    const stepNeg = inspected.nodes.find((n) => n.name === 'step_neg_preserve')!;

    expect(inspected.inits.get(stem.inputs[3])).toEqual([1n]);
    expect(inspected.inits.get(crop.inputs[3])).toEqual([2n, 3n]);
    expect(inspected.inits.get(tail.inputs[3])).toEqual([3n]);
    // Axis 3 in a_unsorted has step=-1 at index 2 -> not identity -> retained as [2, 3]
    expect(inspected.inits.get(stepNeg.inputs[3])).toEqual([2n, 3n]);
    expect(inspected.inits.get(stepNeg.inputs[4])).toEqual([1n, -1n]);

    // Real ORT WASM output must match bit-for-bit
    const origSess = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      logSeverityLevel: 3,
    });
    const compSess = await ort.InferenceSession.create(res.bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3,
    });
    try {
      const xTensor = new ort.Tensor('float32', xData, xDims);
      const origRes = await origSess.run({ x: xTensor });
      const compRes = await compSess.run({ x: xTensor });
      for (const outName of ['y_tail', 'y_neg'] as const) {
        expect(compRes[outName].dims).toEqual(origRes[outName].dims);
        expect(Array.from(compRes[outName].data as Float32Array), outName).toEqual(
          Array.from(origRes[outName].data as Float32Array)
        );
      }
    } finally {
      await origSess.release();
      await compSess.release();
    }
  });

  it('fails closed in compact-active mode on dynamic inputs, sentinel overflow, zero-active-axes, or off mode', () => {
    const INT32_MAX = 2147483647;

    // 1) Symbolic dim_param blocks compaction
    const symbolicDimInput = buildOnnxModel([
      encodeNode('Slice', 'sym_slice', ['x', 's', 'e', 'a'], ['y']),
      encodeStrField(2, 'g_sym'),
      encodeLenField(5, encodeIntTensorProto('s', [4], 6, [0, 0, 0, 0])),
      encodeLenField(5, encodeIntTensorProto('e', [4], 6, [INT32_MAX, INT32_MAX, INT32_MAX, 3])),
      encodeLenField(5, encodeIntTensorProto('a', [4], 6, [0, 2, 3, 1])),
      encodeLenField(11, encodeValueInfoTensor('x', 1, [1, null, 240, 320])),
    ]);
    const symRes = normalizeNonAscendingSliceAxesForWebGpu(symbolicDimInput, { mode: 'compact-active' });
    expect(symRes.normalizedCount).toBe(0);
    expect(symRes.skippedDynamicCount).toBe(1);
    expect(symRes.bytes).toBe(symbolicDimInput);

    // 2) Sentinel overflow (INT32_MAX + 1 = 2147483648n in INT64) fails closed
    const sentinelOverflow = buildOnnxModel([
      encodeNode('Slice', 'overflow_slice', ['x', 's', 'e', 'a'], ['y']),
      encodeStrField(2, 'g_ovf'),
      encodeLenField(5, encodeIntTensorProto('s', [4], 7, [0, 0, 0, 0])),
      encodeLenField(5, encodeIntTensorProto('e', [4], 7, [2147483648n, 2147483648n, 2147483648n, 3n])),
      encodeLenField(5, encodeIntTensorProto('a', [4], 7, [0, 2, 3, 1])),
      encodeLenField(11, encodeValueInfoTensor('x', 1, [1, 6, 4, 4])),
    ]);
    const ovfRes = normalizeNonAscendingSliceAxesForWebGpu(sentinelOverflow, { mode: 'compact-active' });
    expect(ovfRes.normalizedCount).toBe(0);
    expect(ovfRes.skippedInvalidOrUnprovenCount).toBe(1);
    expect(ovfRes.bytes).toBe(sentinelOverflow);

    // 3) Zero active axes remaining (all 4 axes are identity) fails closed
    const allIdentity = buildOnnxModel([
      encodeNode('Slice', 'all_id', ['x', 's', 'e', 'a'], ['y']),
      encodeStrField(2, 'g_all_id'),
      encodeLenField(5, encodeIntTensorProto('s', [4], 6, [0, 0, 0, 0])),
      encodeLenField(5, encodeIntTensorProto('e', [4], 6, [INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX])),
      encodeLenField(5, encodeIntTensorProto('a', [4], 6, [0, 2, 3, 1])),
      encodeLenField(11, encodeValueInfoTensor('x', 1, [1, 6, 4, 4])),
    ]);
    const zeroRes = normalizeNonAscendingSliceAxesForWebGpu(allIdentity, { mode: 'compact-active' });
    expect(zeroRes.normalizedCount).toBe(0);
    expect(zeroRes.skippedInvalidOrUnprovenCount).toBe(1);
    expect(zeroRes.bytes).toBe(allIdentity);

    // 4) Non-zero start stays active even with INT32_MAX end; unproven non-sentinel end stays active
    const partialIdentity = buildOnnxModel([
      encodeNode('Slice', 'nonzero_start', ['x', 's_nz', 'e_inf', 'a_sorted'], ['y1']),
      encodeNode('Slice', 'unproven_end', ['y1', 's_z', 'e_unp', 'a_sorted'], ['y2']),
      encodeStrField(2, 'g_partial'),
      encodeLenField(5, encodeIntTensorProto('s_nz', [4], 6, [0, 1, 0, 0])),
      encodeLenField(5, encodeIntTensorProto('s_z', [4], 6, [0, 0, 0, 0])),
      encodeLenField(5, encodeIntTensorProto('e_inf', [4], 6, [INT32_MAX, INT32_MAX, INT32_MAX, 2])),
      encodeLenField(5, encodeIntTensorProto('e_unp', [4], 6, [1, 2, 4, 2])),
      encodeLenField(5, encodeIntTensorProto('a_sorted', [4], 6, [0, 1, 2, 3])),
      encodeLenField(11, encodeValueInfoTensor('x', 1, [1, 6, 4, 4])),
    ]);
    const partialRes = normalizeNonAscendingSliceAxesForWebGpu(partialIdentity, { mode: 'compact-active' });
    expect(partialRes.normalizedCount).toBe(1);
    expect(partialRes.normalizedNodeNames).toEqual(['nonzero_start']);
    const inspPartial = inspectNodesAndIntInits(partialRes.bytes);
    const nzNode = inspPartial.nodes.find((n) => n.name === 'nonzero_start')!;
    const unpNode = inspPartial.nodes.find((n) => n.name === 'unproven_end')!;
    // start=1 on axis 1 -> axis 1 stays active alongside axis 3!
    expect(inspPartial.inits.get(nzNode.inputs[3])).toEqual([1n, 3n]);
    expect(inspPartial.inits.get(nzNode.inputs[1])).toEqual([1n, 0n]);
    // Unproven finite ends on intermediate y1 stay untouched
    expect(unpNode.inputs[3]).toBe('a_sorted');

    // 5) mode: 'off' returns untouched input bytes immediately
    const offRes = normalizeNonAscendingSliceAxesForWebGpu(partialIdentity, { mode: 'off' });
    expect(offRes.mode).toBe('off');
    expect(offRes.normalizedCount).toBe(0);
    expect(offRes.bytes).toBe(partialIdentity);
  });

  it('normalizes cached real HITNet ONNX in compact-active mode and matches raw ORT WASM output', async () => {
    const hitnetPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (!existsSync(hitnetPath)) {
      return;
    }
    const rawBytes = new Uint8Array(readFileSync(hitnetPath));
    const dec = decomposeOnnx5dConvNodesForWebGpu(rawBytes);
    expect(dec.decomposedCount).toBe(2);

    const compact = normalizeNonAscendingSliceAxesForWebGpu(dec.bytes, { mode: 'compact-active' });
    // All 68 Slice nodes in HITNet (22 unsorted 4D + 10 sorted 4D + 36 5D/6D) carry INT32_MAX identity sentinels
    expect(compact.normalizedCount).toBe(68);
    expect(compact.compactedCount).toBe(68);
    expect(compact.skippedDynamicCount).toBe(0);
    expect(compact.skippedInvalidOrUnprovenCount).toBe(0);
    expect(validateOnnxGraphNodeInputs(compact.bytes)).toMatchObject({ valid: true, missingInputs: [] });

    const xData = new Float32Array(1 * 6 * 240 * 320);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = ((i * 1315423911) % 256) / 255;
    }
    const xTensor = new ort.Tensor('float32', xData, [1, 6, 240, 320]);
    const rawSess = await ort.InferenceSession.create(rawBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      logSeverityLevel: 3,
    });
    const compSess = await ort.InferenceSession.create(compact.bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3,
    });
    try {
      const rawOut = (await rawSess.run({ [rawSess.inputNames[0]]: xTensor }))[rawSess.outputNames[0]]
        .data as Float32Array;
      const compOut = (await compSess.run({ [compSess.inputNames[0]]: xTensor }))[compSess.outputNames[0]]
        .data as Float32Array;
      expect(compOut.length).toBe(rawOut.length);
      let maxAbsDiff = 0;
      let sumAbsDiff = 0;
      for (let i = 0; i < rawOut.length; i++) {
        const d = Math.abs(compOut[i] - rawOut[i]);
        if (d > maxAbsDiff) maxAbsDiff = d;
        sumAbsDiff += d;
      }
      expect(maxAbsDiff).toBeLessThan(1e-3);
      expect(sumAbsDiff / rawOut.length).toBeLessThan(1e-5);
    } finally {
      await rawSess.release();
      await compSess.release();
    }
  }, 90000);
});

describe('normalizeHitnetGatherNdRankForWebGpu', () => {
  it('flattens non-batch axes of rank-5 GatherND indices while leaving rank-4 GatherND and negative indices unchanged', async () => {
    const suite = buildSyntheticGatherNdControlModels();
    expect(suite.patchedNodeNames).toEqual(['gnd_rank5']);

    const inspected = inspectNodesAndIntInits(suite.models.rank4ThenPatchedRank5);
    expect(inspected.nodes.map((n) => n.opType)).toEqual([
      'GatherND',
      'Reshape',
      'GatherND',
      'Reshape',
    ]);
    const preReshape = inspected.nodes[1];
    const flatGather = inspected.nodes[2];
    const postReshape = inspected.nodes[3];

    expect(inspected.inits.get(preReshape.inputs[1])).toEqual([1n, 2n, 12n, 1n]);
    expect(inspected.inits.get(postReshape.inputs[1])).toEqual([1n, 2n, 4n, 3n, 4n]);
    expect(flatGather.inputs[0]).toBe('data');
    expect(flatGather.inputs[1]).toBe(preReshape.outputs[0]);
    expect(postReshape.inputs[0]).toBe(flatGather.outputs[0]);
    expect(postReshape.outputs[0]).toBe('out5');
    expect(validateOnnxGraphNodeInputs(suite.models.rank4ThenPatchedRank5)).toMatchObject({
      valid: true,
      missingInputs: [],
    });

    // Compare against unmodified WASM (including negative indices -1..-8)
    const rawSess = await ort.InferenceSession.create(suite.models.rank4ThenRank5, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      logSeverityLevel: 3,
    });
    const patchedSess = await ort.InferenceSession.create(suite.models.rank4ThenPatchedRank5, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3,
    });
    try {
      const feeds = {
        data: new ort.Tensor('float32', suite.dataF32, suite.dataDims),
        idx4: new ort.Tensor('int64', suite.idx4I64, suite.idx4Dims),
        idx5: new ort.Tensor('int64', suite.idx5I64, suite.idx5Dims),
      };
      const rawOut = await rawSess.run(feeds);
      const patchedOut = await patchedSess.run(feeds);
      for (const outName of ['out4', 'out5'] as const) {
        expect(patchedOut[outName].dims).toEqual(rawOut[outName].dims);
        const s = summarizeRawTensorAndParity(
          patchedOut[outName].data as Float32Array,
          rawOut[outName].data as Float32Array
        );
        expect(s.maxAbsDiff).toBe(0);
        expect(s.nonFiniteMismatchCount).toBe(0);
      }
    } finally {
      await rawSess.release();
      await patchedSess.release();
    }
  });

  it('fails closed on non-HITNet variants, wasm backend, dynamic shapes, and contract mismatches', () => {
    const suite = buildSyntheticGatherNdControlModels();

    // 1) Production contract (default requireHitnetContract: true) rejects non-hitnet-sf variants
    const nonHitnet = normalizeHitnetGatherNdRankForWebGpu(suite.models.rank4ThenRank5, {
      variantId: 'lightstereo-s',
      backendPreference: 'webgpu',
    });
    expect(nonHitnet.normalizedCount).toBe(0);
    expect(nonHitnet.bytes).toBe(suite.models.rank4ThenRank5);

    // 2) Production contract rejects synthetic model that does not match HITNet 4-node warp_1d signature
    const hitnetContractOnSynthetic = normalizeHitnetGatherNdRankForWebGpu(
      suite.models.rank4ThenRank5,
      {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: true,
      }
    );
    expect(hitnetContractOnSynthetic.normalizedCount).toBe(0);
    expect(hitnetContractOnSynthetic.bytes).toBe(suite.models.rank4ThenRank5);

    // 3) WASM backend preference never mutates graph
    const wasmPref = normalizeHitnetGatherNdRankForWebGpu(suite.models.rank4ThenRank5, {
      variantId: 'hitnet-sf',
      backendPreference: 'wasm',
      requireHitnetContract: false,
    });
    expect(wasmPref.normalizedCount).toBe(0);
    expect(wasmPref.bytes).toBe(suite.models.rank4ThenRank5);

    // 4) Dynamic graph input dims fail closed even when requireHitnetContract is false
    const dynModel = buildOnnxModel([
      encodeNode('GatherND', 'dyn_gnd', ['data', 'idx5'], ['out5']),
      encodeStrField(2, 'dyn_graph'),
      encodeLenField(11, encodeValueInfoTensor('data', 1, [1, 2, null, 4])),
      encodeLenField(11, encodeValueInfoTensor('idx5', 7, [1, 2, 4, 3, 1])),
      encodeLenField(12, encodeValueInfoTensor('out5', 1, [1, 2, 4, 3, 4])),
    ]);
    const dynRes = normalizeHitnetGatherNdRankForWebGpu(dynModel, {
      requireHitnetContract: false,
    });
    expect(dynRes.normalizedCount).toBe(0);
    expect(dynRes.bytes).toBe(dynModel);
  });

  it('normalizes exact 4 warp_1d GatherND nodes on cached real HITNet ONNX and matches raw WASM on both outputs', async () => {
    const hitnetPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (!existsSync(hitnetPath)) {
      return;
    }
    const rawBytes = new Uint8Array(readFileSync(hitnetPath));
    const dec = decomposeOnnx5dConvNodesForWebGpu(rawBytes);
    const compact = normalizeNonAscendingSliceAxesForWebGpu(dec.bytes, { mode: 'compact-active' });
    const gnd = normalizeHitnetGatherNdRankForWebGpu(compact.bytes, {
      variantId: 'hitnet-sf',
      backendPreference: 'webgpu',
      requireHitnetContract: true,
    });

    expect(gnd.totalGatherNdCount).toBe(10);
    expect(gnd.normalizedCount).toBe(4);
    expect(gnd.skippedDynamicCount).toBe(0);
    expect(gnd.skippedInvalidOrUnprovenCount).toBe(0);
    expect(gnd.normalizedNodeNames).toEqual([
      'level0/shared/prop0/WarpImageWithHypotheses/warp_and_aggregate_sad/warp_1d/GatherV2;fe_shared/conv_upsample_0/mul/y',
      'level0/shared/prop0/WarpImageWithHypotheses/warp_and_aggregate_sad/warp_1d/GatherV2_1;fe_shared/conv_upsample_0/mul/y',
      'level0_1/shared/prop0/WarpImageWithHypotheses/warp_and_aggregate_sad/warp_1d/GatherV2;fe_shared/conv_upsample_0/mul/y',
      'level0_1/shared/prop0/WarpImageWithHypotheses/warp_and_aggregate_sad/warp_1d/GatherV2_1;fe_shared/conv_upsample_0/mul/y',
    ]);
    expect(validateOnnxGraphNodeInputs(gnd.bytes)).toMatchObject({
      valid: true,
      missingInputs: [],
    });

    // Running it on a non-HITNet variant fails closed
    expect(
      normalizeHitnetGatherNdRankForWebGpu(compact.bytes, {
        variantId: 'crestereo-tiny',
        backendPreference: 'webgpu',
      }).normalizedCount
    ).toBe(0);

    const xData = new Float32Array(1 * 6 * 240 * 320);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = ((i * 1315423911) % 256) / 255;
    }
    const xTensor = new ort.Tensor('float32', xData, [1, 6, 240, 320]);
    const rawSess = await ort.InferenceSession.create(rawBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      logSeverityLevel: 3,
    });
    const patchedSess = await ort.InferenceSession.create(gnd.bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3,
    });
    try {
      const rawRes = await rawSess.run({ [rawSess.inputNames[0]]: xTensor });
      const patchedRes = await patchedSess.run({ [patchedSess.inputNames[0]]: xTensor });
      for (const outName of ['reference_output_disparity', 'secondary_output_disparity'] as const) {
        expect(patchedRes[outName].dims).toEqual([1, 240, 320, 1]);
        expect(rawRes[outName].dims).toEqual([1, 240, 320, 1]);
        const s = summarizeRawTensorAndParity(
          patchedRes[outName].data as Float32Array,
          rawRes[outName].data as Float32Array
        );
        expect(s.finiteCount).toBe(76800);
        expect(s.nonFiniteMismatchCount).toBe(0);
        expect(s.maxAbsDiff).toBeLessThan(1e-3);
        expect(s.p99AbsDiff).toBeLessThan(1e-4);
        expect(s.meanAbsDiff).toBeLessThan(1e-5);
      }
    } finally {
      await rawSess.release();
      await patchedSess.release();
    }
  }, 90000);
});

describe('normalizeHitnetArgMinKeepDimsForWebGpu', () => {
  ort.env.wasm.numThreads = 1;

  it('proves ORT ArgMin cache attributes key distinguishes keepdims=0 vs keepdims=1', () => {
    const k0 = computeOrtArgMinCacheAttributesKey({
      axis: 3,
      keepDims: false,
      selectLastIndex: 0,
      inputType: 1,
      inputRank: 4,
      is1DimensionDispatch: false,
    });
    const k1 = computeOrtArgMinCacheAttributesKey({
      axis: 3,
      keepDims: true,
      selectLastIndex: 0,
      inputType: 1,
      inputRank: 4,
      is1DimensionDispatch: false,
    });
    expect(k0).toBe('ArgMin[3;false;0]:false:1;4');
    expect(k1).toBe('ArgMin[3;true;0]:false:1;4');
    expect(k1).not.toBe(k0);
  });

  it('validates safe synthetic ArgMin controls (S5_Cast, S5_GatherND, S6) including ties and last spatial location', async () => {
    const suite = buildSyntheticArgMinControlModels();
    expect(suite.patchedNodeNames).toEqual(['prop0_argmin']);
    expect(suite.untouchedNodeNames).toEqual(['level_init_argmin']);
    expect(suite.updatedValueInfoCount).toBe(1);
    expect(suite.ortCacheKeyBefore).toBe('ArgMin[3;false;0]:false:1;4');
    expect(suite.ortCacheKeyAfter).toBe('ArgMin[3;true;0]:false:1;4');
    expect(validateOnnxGraphNodeInputs(suite.models.patchedSeparatedTwoSize)).toMatchObject({
      valid: true,
      missingInputs: [],
    });

    const runs: Array<{
      name: string;
      bytes: Uint8Array;
      feeds: Record<string, ort.Tensor>;
      expected: Record<string, Float32Array>;
    }> = [
      {
        name: 'S5_Cast',
        bytes: suite.models.singleSizeCast,
        feeds: { cost8: new ort.Tensor('float32', suite.cost8F32, suite.cost8Dims) },
        expected: { out_cast: suite.expectedCastIndices },
      },
      {
        name: 'S5_GatherND',
        bytes: suite.models.singleSizeGatherNd,
        feeds: {
          cost4: new ort.Tensor('float32', suite.cost4F32, suite.cost4Dims),
          table4: new ort.Tensor('float32', suite.table4F32, suite.table4Dims),
        },
        expected: { out_gathernd: suite.expectedGatherNdValues },
      },
      {
        name: 'S6',
        bytes: suite.models.patchedSeparatedTwoSize,
        feeds: {
          cost8: new ort.Tensor('float32', suite.cost8F32, suite.cost8Dims),
          cost4: new ort.Tensor('float32', suite.cost4F32, suite.cost4Dims),
          table4: new ort.Tensor('float32', suite.table4F32, suite.table4Dims),
        },
        expected: {
          out_cast: suite.expectedCastIndices,
          out_gathernd: suite.expectedGatherNdValues,
        },
      },
    ];

    for (const r of runs) {
      const sessDisabled = await ort.InferenceSession.create(r.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'disabled',
        logSeverityLevel: 3,
      });
      const sessAll = await ort.InferenceSession.create(r.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        logSeverityLevel: 3,
      });
      try {
        const resDisabled = await sessDisabled.run(r.feeds);
        const resAll = await sessAll.run(r.feeds);
        for (const [outName, expectedArr] of Object.entries(r.expected)) {
          expect(resDisabled[outName].type, `${r.name}:${outName}`).toBe('float32');
          expect(resAll[outName].type, `${r.name}:${outName}`).toBe('float32');
          expect(Array.from(resDisabled[outName].data as Float32Array), `${r.name}:${outName} disabled`).toEqual(
            Array.from(expectedArr)
          );
          expect(Array.from(resAll[outName].data as Float32Array), `${r.name}:${outName} all`).toEqual(
            Array.from(expectedArr)
          );
        }
      } finally {
        await sessDisabled.release();
        await sessAll.release();
      }
    }
  });

  it('fails closed on contract mismatches, non-Reshape consumers, select_last_index!=0, wrong opset/irVersion, and wasm without diagnostic flag', () => {
    const initShape = encodeLenField(5, encodeIntTensorProto('norm_shape', [4], 7, [1, 60, 80, 1]));

    const buildTwoArgMinHitnetLike = (opts?: {
      irVersion?: number;
      opset?: number;
      selectLastIndex?: number;
      propConsumerOp?: 'Reshape' | 'Identity';
      propAsGraphOutput?: boolean;
      nodeNames?: [string, string, string, string];
    }) => {
      const names = opts?.nodeNames ?? [
        'level0/level_init/ArgMin',
        'level0/shared/prop0/ArgMin',
        'level0_1/level_init/ArgMin',
        'level0_1/shared/prop0/ArgMin',
      ];
      const propOutName = opts?.propConsumerOp === 'Identity' ? 'prop0_ident' : 'prop0_r_out';
      const nodes = [
        encodeNode('Identity', 'mk_cost0', ['x_cost0'], ['c0']),
        encodeNode('ArgMin', names[0], ['c0'], ['a0'], [
          encodeAttrInt('axis', 3),
          encodeAttrInt('keepdims', 0),
          encodeAttrInt('select_last_index', 0),
        ]),
        encodeNode('Reshape', 'r0', ['a0', 'norm_shape'], ['r0_out']),
        encodeNode('ArgMin', names[1], ['p0'], ['a1'], [
          encodeAttrInt('axis', 3),
          encodeAttrInt('keepdims', 0),
          encodeAttrInt('select_last_index', opts?.selectLastIndex ?? 0),
        ]),
        opts?.propConsumerOp === 'Identity'
          ? encodeNode('Identity', 'id1', ['a1'], [propOutName])
          : encodeNode('Reshape', 'r1', ['a1', 'norm_shape'], [propOutName]),
        encodeNode('ArgMin', names[2], ['c1'], ['a2'], [
          encodeAttrInt('axis', 3),
          encodeAttrInt('keepdims', 0),
          encodeAttrInt('select_last_index', 0),
        ]),
        encodeNode('Reshape', 'r2', ['a2', 'norm_shape'], ['r2_out']),
        encodeNode('ArgMin', names[3], ['p1'], ['a3'], [
          encodeAttrInt('axis', 3),
          encodeAttrInt('keepdims', 0),
          encodeAttrInt('select_last_index', 0),
        ]),
        encodeNode('Reshape', 'r3', ['a3', 'norm_shape'], ['r3_out']),
      ];
      const graphParts = [
        ...nodes,
        encodeStrField(2, 'hitnet_argmin_contract_test'),
        initShape,
        encodeLenField(11, encodeValueInfoTensor('x_cost0', 1, [1, 60, 80, 320])),
        encodeLenField(11, encodeValueInfoTensor('p0', 1, [1, 60, 80, 4])),
        encodeLenField(11, encodeValueInfoTensor('c1', 1, [1, 60, 80, 320])),
        encodeLenField(11, encodeValueInfoTensor('p1', 1, [1, 60, 80, 4])),
        encodeLenField(12, encodeValueInfoTensor('r0_out', 7, [1, 60, 80, 1])),
        encodeLenField(12, encodeValueInfoTensor(propOutName, 7, opts?.propConsumerOp === 'Identity' ? [1, 60, 80] : [1, 60, 80, 1])),
        encodeLenField(12, encodeValueInfoTensor('r2_out', 7, [1, 60, 80, 1])),
        encodeLenField(12, encodeValueInfoTensor('r3_out', 7, [1, 60, 80, 1])),
      ];
      if (opts?.propAsGraphOutput) {
        graphParts.push(encodeLenField(12, encodeValueInfoTensor('a1', 7, [1, 60, 80])));
      }
      return concatBytes([
        encodeVarintField(1, opts?.irVersion ?? 7),
        encodeLenField(7, concatBytes(graphParts)),
        encodeLenField(8, concatBytes([encodeStrField(1, ''), encodeVarintField(2, opts?.opset ?? 12)])),
      ]);
    };

    const validSynthHitnet = buildTwoArgMinHitnetLike();
    // Non-HITNet contract mode succeeds on valid 4-node graph
    const okRelaxed = normalizeHitnetArgMinKeepDimsForWebGpu(validSynthHitnet, {
      variantId: 'hitnet-sf',
      backendPreference: 'webgpu',
      requireHitnetContract: false,
    });
    expect(okRelaxed.normalizedCount).toBe(2);

    // Full HITNet contract rejects synthetic graph that lacks [1,6,240,320] input & disparity outputs
    expect(
      normalizeHitnetArgMinKeepDimsForWebGpu(validSynthHitnet, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: true,
      }).normalizedCount
    ).toBe(0);

    // Non-hitnet variant blocked when requireHitnetContract is true
    expect(
      normalizeHitnetArgMinKeepDimsForWebGpu(validSynthHitnet, {
        variantId: 'lightstereo-s',
        backendPreference: 'webgpu',
        requireHitnetContract: true,
      }).normalizedCount
    ).toBe(0);

    // WASM blocked unless allowDiagnosticTransformedWasm is true
    const wasmBlocked = normalizeHitnetArgMinKeepDimsForWebGpu(validSynthHitnet, {
      variantId: 'hitnet-sf',
      backendPreference: 'wasm',
      requireHitnetContract: false,
    });
    expect(wasmBlocked.normalizedCount).toBe(0);
    expect(wasmBlocked.bytes).toBe(validSynthHitnet);

    const wasmDiagAllowed = normalizeHitnetArgMinKeepDimsForWebGpu(validSynthHitnet, {
      variantId: 'hitnet-sf',
      backendPreference: 'wasm',
      allowDiagnosticTransformedWasm: true,
      requireHitnetContract: false,
    });
    expect(wasmDiagAllowed.normalizedCount).toBe(2);

    // Violations all fail closed cleanly
    for (const badBytes of [
      buildTwoArgMinHitnetLike({ opset: 11 }),
      buildTwoArgMinHitnetLike({ selectLastIndex: 1 }),
      buildTwoArgMinHitnetLike({ propConsumerOp: 'Identity' }),
      buildTwoArgMinHitnetLike({ propAsGraphOutput: true }),
    ]) {
      const res = normalizeHitnetArgMinKeepDimsForWebGpu(badBytes, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: false,
      });
      expect(res.normalizedCount).toBe(0);
      expect(res.bytes).toBe(badBytes);
    }
    const badIrStrict = buildTwoArgMinHitnetLike({ irVersion: 6 });
    expect(
      normalizeHitnetArgMinKeepDimsForWebGpu(badIrStrict, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: true,
      }).normalizedCount
    ).toBe(0);
  });

  it('normalizes exact 2 prop0 ArgMin nodes on cached real HITNet ONNX (G1) and verifies C-WASM controls + G1 WASM parity on both outputs', async () => {
    const hitnetPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (!existsSync(hitnetPath)) {
      return;
    }
    const rawBytes = new Uint8Array(readFileSync(hitnetPath));
    const dec = decomposeOnnx5dConvNodesForWebGpu(rawBytes);
    const compact = normalizeNonAscendingSliceAxesForWebGpu(dec.bytes, { mode: 'compact-active' });
    const f1 = normalizeHitnetGatherNdRankForWebGpu(compact.bytes, {
      variantId: 'hitnet-sf',
      backendPreference: 'webgpu',
      requireHitnetContract: true,
    });
    const g1 = normalizeHitnetArgMinKeepDimsForWebGpu(f1.bytes, {
      variantId: 'hitnet-sf',
      backendPreference: 'webgpu',
      requireHitnetContract: true,
    });

    expect(g1.totalArgMinCount).toBe(4);
    expect(g1.normalizedCount).toBe(2);
    expect(g1.skippedDynamicCount).toBe(0);
    expect(g1.skippedInvalidOrUnprovenCount).toBe(0);
    expect(g1.normalizedNodeNames).toEqual([
      'level0/shared/prop0/ArgMin',
      'level0_1/shared/prop0/ArgMin',
    ]);
    expect(g1.untouchedNodeNames).toEqual([
      'level0/level_init/ArgMin',
      'level0_1/level_init/ArgMin',
    ]);
    expect(g1.ortCacheKeyBefore).toBe('ArgMin[3;false;0]:false:1;4');
    expect(g1.ortCacheKeyAfter).toBe('ArgMin[3;true;0]:false:1;4');
    expect(validateOnnxGraphNodeInputs(g1.bytes)).toMatchObject({
      valid: true,
      missingInputs: [],
    });

    // Verify exact F1/G1 WASM control sessions match unmodified Raw WASM opt=disabled on both outputs
    const xData = new Float32Array(1 * 6 * 240 * 320);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = ((i * 1315423911) % 256) / 255;
    }
    const xTensor = new ort.Tensor('float32', xData, [1, 6, 240, 320]);

    const runWasm = async (modelBytes: Uint8Array, optLevel: 'disabled' | 'all') => {
      const sess = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: optLevel,
        logSeverityLevel: 3,
      });
      try {
        const out = await sess.run({ [sess.inputNames[0]]: xTensor });
        return {
          ref: new Float32Array(out.reference_output_disparity.data as Float32Array),
          refDims: Array.from(out.reference_output_disparity.dims),
          refType: out.reference_output_disparity.type,
          sec: new Float32Array(out.secondary_output_disparity.data as Float32Array),
          secDims: Array.from(out.secondary_output_disparity.dims),
          secType: out.secondary_output_disparity.type,
        };
      } finally {
        await sess.release();
      }
    };

    const rawDis = await runWasm(rawBytes, 'disabled');
    const rawAll = await runWasm(rawBytes, 'all');
    const f1Dis = await runWasm(f1.bytes, 'disabled');
    const f1All = await runWasm(f1.bytes, 'all');
    const g1All = await runWasm(g1.bytes, 'all');

    for (const [label, cand] of [
      ['Raw-All', rawAll],
      ['F1-Disabled', f1Dis],
      ['F1-All', f1All],
      ['G1-All', g1All],
    ] as const) {
      expect(cand.refType, label).toBe('float32');
      expect(cand.secType, label).toBe('float32');
      expect(cand.refDims, label).toEqual([1, 240, 320, 1]);
      expect(cand.secDims, label).toEqual([1, 240, 320, 1]);

      const sRef = summarizeRawTensorAndParity(cand.ref, rawDis.ref);
      const sSec = summarizeRawTensorAndParity(cand.sec, rawDis.sec);
      for (const [outLabel, s] of [
        ['primary', sRef],
        ['secondary', sSec],
      ] as const) {
        expect(s.finiteCount, `${label}:${outLabel}`).toBe(76800);
        expect(s.nonFiniteMismatchCount, `${label}:${outLabel}`).toBe(0);
        expect(s.maxAbsDiff, `${label}:${outLabel}`).toBeLessThan(1e-3);
        expect(s.p99AbsDiff, `${label}:${outLabel}`).toBeLessThan(1e-4);
        expect(s.meanAbsDiff, `${label}:${outLabel}`).toBeLessThan(1e-5);
      }
    }
  }, 120000);
});

describe('normalizeHitnetPropGatherNdRankForWebGpu (H1)', () => {
  ort.env.wasm.numThreads = 1;

  it('proves standalone rank-5 -> H1 rank-4 GatherND equivalence and validates benign S7a..S7d controls in ORT WASM', async () => {
    const suite = buildSyntheticPropGatherNdControlModels();
    expect(suite.patchedNodeNames).toEqual(['prop0_gnd_rank5']);

    // 1) Standalone rank-5 unit pair: unrewritten GatherND(batch_dims=3) vs H1 merged rank-4 (batch_dims=2)
    const rawR5Sess = await ort.InferenceSession.create(suite.models.unrewrittenRank5UnitTest, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      logSeverityLevel: 3,
    });
    const h1R5Dis = await ort.InferenceSession.create(suite.models.h1RewrittenUnitTest, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      logSeverityLevel: 3,
    });
    const h1R5All = await ort.InferenceSession.create(suite.models.h1RewrittenUnitTest, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3,
    });
    try {
      const r5Feeds = {
        data: new ort.Tensor('float32', suite.dataF32, suite.dataDims5),
        idx: new ort.Tensor('int64', suite.idxI64, suite.idxDims5),
      };
      const rawOut = await rawR5Sess.run(r5Feeds);
      const disOut = await h1R5Dis.run(r5Feeds);
      const allOut = await h1R5All.run(r5Feeds);
      expect(disOut.out.dims).toEqual([1, 2, 3, 1, 2]);
      expect(allOut.out.dims).toEqual([1, 2, 3, 1, 2]);
      expect(
        summarizeRawTensorAndParity(disOut.out.data as Float32Array, rawOut.out.data as Float32Array).maxAbsDiff
      ).toBe(0);
      expect(
        summarizeRawTensorAndParity(allOut.out.data as Float32Array, rawOut.out.data as Float32Array).maxAbsDiff
      ).toBe(0);
      expect(
        summarizeRawTensorAndParity(allOut.out.data as Float32Array, suite.expectedConstOrInputOut).maxAbsDiff
      ).toBe(0);
    } finally {
      await rawR5Sess.release();
      await h1R5Dis.release();
      await h1R5All.release();
    }

    // 2) Benign rank-4 WebGPU-safe synthetics S7a..S7d against analytical ground truth
    const s7Runs: Array<{
      label: string;
      bytes: Uint8Array;
      feeds: Record<string, ort.Tensor>;
      expected: Record<string, Float32Array>;
    }> = [
      {
        label: 'S7a_ConstIdx',
        bytes: suite.models.s7aConstIdx,
        feeds: { data4: new ort.Tensor('float32', suite.dataF32, suite.dataDims4) },
        expected: { out_s7a: suite.expectedConstOrInputOut },
      },
      {
        label: 'S7b_InputIdx',
        bytes: suite.models.s7bInputIdx,
        feeds: {
          data4: new ort.Tensor('float32', suite.dataF32, suite.dataDims4),
          idx4: new ort.Tensor('int64', suite.idxI64, suite.idxDims4),
        },
        expected: { out_s7b: suite.expectedConstOrInputOut },
      },
      {
        label: 'S7c_DynamicArgMin',
        bytes: suite.models.s7cDynamicArgMin,
        feeds: {
          cost4: new ort.Tensor('float32', suite.cost4F32, suite.cost4Dims),
          data4: new ort.Tensor('float32', suite.dataF32, suite.dataDims4),
        },
        expected: { out_s7c: suite.expectedDynamicArgMinOut },
      },
      {
        label: 'S7d_TwoBranch',
        bytes: suite.models.s7dTwoBranch,
        feeds: {
          cost8: new ort.Tensor('float32', suite.cost8F32, suite.cost8Dims),
          cost4: new ort.Tensor('float32', suite.cost4F32, suite.cost4Dims),
          data4: new ort.Tensor('float32', suite.dataF32, suite.dataDims4),
        },
        expected: {
          out_cast: suite.expectedCastOut,
          out_gathernd: suite.expectedDynamicArgMinOut,
        },
      },
    ];

    for (const runSpec of s7Runs) {
      const sDis = await ort.InferenceSession.create(runSpec.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'disabled',
        logSeverityLevel: 3,
      });
      const sAll = await ort.InferenceSession.create(runSpec.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        logSeverityLevel: 3,
      });
      try {
        const resDis = await sDis.run(runSpec.feeds);
        const resAll = await sAll.run(runSpec.feeds);
        for (const [outName, expArr] of Object.entries(runSpec.expected)) {
          expect(Array.from(resDis[outName].data as Float32Array), `${runSpec.label}:${outName} dis`).toEqual(
            Array.from(expArr)
          );
          expect(Array.from(resAll[outName].data as Float32Array), `${runSpec.label}:${outName} all`).toEqual(
            Array.from(expArr)
          );
        }
      } finally {
        await sDis.release();
        await sAll.release();
      }
    }
  });

  it('infers exact rank-5 shapes, applies guarded H1 on cached real HITNet ONNX, and verifies full-graph & P2b..P6 checkpoint parity in ORT WASM', async () => {
    const hitnetPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (!existsSync(hitnetPath)) {
      return;
    }
    const rawBytes = new Uint8Array(readFileSync(hitnetPath));
    const dec = decomposeOnnx5dConvNodesForWebGpu(rawBytes);
    const compact = normalizeNonAscendingSliceAxesForWebGpu(dec.bytes, { mode: 'compact-active' });
    const f1 = normalizeHitnetGatherNdRankForWebGpu(compact.bytes, {
      variantId: 'hitnet-sf',
      backendPreference: 'webgpu',
      requireHitnetContract: true,
    });
    const f1Snapshot = f1.bytes.slice();
    const h1 = normalizeHitnetPropGatherNdRankForWebGpu(f1.bytes, {
      variantId: 'hitnet-sf',
      backendPreference: 'webgpu',
      requireHitnetContract: true,
    });

    // Does not mutate input buffer
    expect(Buffer.compare(f1.bytes, f1Snapshot)).toBe(0);
    expect(h1.totalGatherNdCount).toBe(10);
    expect(h1.normalizedCount).toBe(2);
    expect(h1.skippedDynamicCount).toBe(0);
    expect(h1.skippedInvalidOrUnprovenCount).toBe(0);
    expect(h1.normalizedNodeNames).toEqual([
      'level0/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis',
      'level0_1/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis',
    ]);

    const rawInfer = inferOnnxStaticTensorMeta(rawBytes);
    const h1Infer = inferOnnxStaticTensorMeta(h1.bytes);
    expect(rawInfer.valid).toBe(true);
    expect(h1Infer.valid).toBe(true);
    expect(h1Infer.duplicateOutputs).toEqual([]);
    expect(h1Infer.tensorMeta['level0/level_init/init_to_prop/ExpandDims']).toEqual({
      elemType: 1,
      dims: [1, 60, 80, 1, 16],
    });
    expect(h1Infer.tensorMeta['level0/shared/prop0/WarpImageWithHypotheses/strided_slice_3']).toEqual({
      elemType: 1,
      dims: [1, 60, 80, 1, 3],
    });
    expect(h1Infer.tensorMeta['level0/shared/prop0/add']).toEqual({
      elemType: 1,
      dims: [1, 60, 80, 1, 16],
    });
    expect(h1Infer.tensorMeta['level0/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis']).toEqual({
      elemType: 1,
      dims: [1, 60, 80, 1, 16],
    });
    expect(h1Infer.tensorMeta['level0_1/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis']).toEqual({
      elemType: 1,
      dims: [1, 60, 80, 1, 16],
    });
    expect(validateOnnxGraphNodeInputs(h1.bytes)).toMatchObject({ valid: true, missingInputs: [] });

    // Full-graph H1 WASM parity vs Raw WASM opt=disabled on both disparity outputs
    const xData = new Float32Array(1 * 6 * 240 * 320);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = ((i * 1315423911) % 256) / 255;
    }
    const xTensor = new ort.Tensor('float32', xData, [1, 6, 240, 320]);

    const runFullWasm = async (modelBytes: Uint8Array, optLevel: 'disabled' | 'basic' | 'all') => {
      const sess = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: optLevel,
        logSeverityLevel: 3,
      });
      try {
        const out = await sess.run({ [sess.inputNames[0]]: xTensor });
        return {
          ref: new Float32Array(out.reference_output_disparity.data as Float32Array),
          sec: new Float32Array(out.secondary_output_disparity.data as Float32Array),
        };
      } finally {
        await sess.release();
      }
    };

    const rawDis = await runFullWasm(rawBytes, 'disabled');
    for (const optLevel of ['disabled', 'basic', 'all'] as const) {
      const h1Out = await runFullWasm(h1.bytes, optLevel);
      for (const [label, cand, ref] of [
        ['primary', h1Out.ref, rawDis.ref],
        ['secondary', h1Out.sec, rawDis.sec],
      ] as const) {
        const s = summarizeRawTensorAndParity(cand, ref);
        expect(s.finiteCount, `H1:${optLevel}:${label}`).toBe(76800);
        expect(s.nonFiniteMismatchCount, `H1:${optLevel}:${label}`).toBe(0);
        expect(s.maxAbsDiff, `H1:${optLevel}:${label}`).toBeLessThan(1e-3);
        expect(s.p99AbsDiff, `H1:${optLevel}:${label}`).toBeLessThan(1e-4);
        expect(s.meanAbsDiff, `H1:${optLevel}:${label}`).toBeLessThan(1e-5);
      }
    }

    // Per-arm P2b..P6 checkpoint subgraph verification in ORT WASM (preserving rank-5 & both branches)
    for (const mode of ['p2b', 'p3a', 'p3b', 'p3c', 'p4', 'p5', 'p6'] as const) {
      const arms = prepareHitnetSubgraphArms(rawBytes, mode, { variant: 'h1' });
      expect(arms.limitationNote).toMatch(/truncates downstream nodes/i);
      expect(arms.rawArm.targetOutputs).toEqual(arms.candidateArm.targetOutputs);
      if (mode === 'p5' || mode === 'p6') {
        expect(arms.rawArm.targetOutputs).toHaveLength(3);
      }
      const sRaw = await ort.InferenceSession.create(arms.rawArm.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'disabled',
        logSeverityLevel: 3,
      });
      const sH1 = await ort.InferenceSession.create(arms.candidateArm.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        logSeverityLevel: 3,
      });
      try {
        const rRaw = await sRaw.run({ [sRaw.inputNames[0]]: xTensor });
        const rH1 = await sH1.run({ [sH1.inputNames[0]]: xTensor });
        for (const outName of arms.rawArm.targetOutputs) {
          expect(Array.from(rH1[outName].dims), `${mode}:${outName}`).toEqual(
            arms.rawArm.outputMeta[outName].dims
          );
          const s = summarizeRawTensorAndParity(
            rH1[outName].data as Float32Array,
            rRaw[outName].data as Float32Array
          );
          expect(s.nonFiniteMismatchCount, `${mode}:${outName}`).toBe(0);
          expect(s.maxAbsDiff, `${mode}:${outName}`).toBeLessThan(1e-3);
        }
      } finally {
        await sRaw.release();
        await sH1.release();
      }
    }
  }, 180000);

  it('fails closed on wrong opset, irVersion, shape, dtype, wrong node count, extra rank-5 GatherND, and wasm without diagnostic flag', () => {
    const buildContractCandidate = (opts?: {
      irVersion?: number;
      opset?: number;
      dataDims?: number[];
      idxDims?: number[];
      idxType?: number;
      batchDims?: number;
      extraRank5GatherNd?: boolean;
      omitSecondPropNode?: boolean;
    }) => {
      const dataDims = opts?.dataDims ?? [1, 60, 80, 1, 16];
      const idxDims = opts?.idxDims ?? [1, 60, 80, 1, 1];
      const batchDims = opts?.batchDims ?? 3;
      const nodes: Uint8Array[] = [
        encodeNode(
          'GatherND',
          'level0/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis',
          ['level0/shared/prop0/add', 'Unsqueeze__675:0'],
          ['level0/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis'],
          [encodeAttrInt('batch_dims', batchDims)]
        ),
      ];
      if (!opts?.omitSecondPropNode) {
        nodes.push(
          encodeNode(
            'GatherND',
            'level0_1/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis',
            ['level0_1/shared/prop0/add', 'Unsqueeze__387:0'],
            ['level0_1/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis'],
            [encodeAttrInt('batch_dims', batchDims)]
          )
        );
      }
      if (opts?.extraRank5GatherNd) {
        nodes.push(
          encodeNode(
            'GatherND',
            'unexpected_rank5_gnd',
            ['level0/shared/prop0/add', 'Unsqueeze__675:0'],
            ['extra_out'],
            [encodeAttrInt('batch_dims', 3)]
          )
        );
      }
      const graphParts: Uint8Array[] = [
        ...nodes,
        encodeStrField(2, 'h1_guard_graph'),
        encodeLenField(11, encodeValueInfoTensor('input', 1, [1, 6, 240, 320])),
        encodeLenField(11, encodeValueInfoTensor('level0/shared/prop0/add', 1, dataDims)),
        encodeLenField(11, encodeValueInfoTensor('Unsqueeze__675:0', opts?.idxType ?? 7, idxDims)),
        encodeLenField(11, encodeValueInfoTensor('level0_1/shared/prop0/add', 1, dataDims)),
        encodeLenField(11, encodeValueInfoTensor('Unsqueeze__387:0', opts?.idxType ?? 7, idxDims)),
        encodeLenField(12, encodeValueInfoTensor('reference_output_disparity', 1, [1, 240, 320, 1])),
        encodeLenField(12, encodeValueInfoTensor('secondary_output_disparity', 1, [1, 240, 320, 1])),
      ];
      if (opts?.extraRank5GatherNd) {
        graphParts.push(encodeLenField(12, encodeValueInfoTensor('extra_out', 1, dataDims)));
      }
      return concatBytes([
        encodeVarintField(1, opts?.irVersion ?? 7),
        encodeLenField(7, concatBytes(graphParts)),
        encodeLenField(8, concatBytes([encodeStrField(1, ''), encodeVarintField(2, opts?.opset ?? 12)])),
      ]);
    };

    const baselineInvalidTotalCount = buildContractCandidate();
    // Fails strict HITNet contract because totalGatherNdCount is 2 instead of 10
    expect(
      normalizeHitnetPropGatherNdRankForWebGpu(baselineInvalidTotalCount, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: true,
      }).normalizedCount
    ).toBe(0);

    // Relaxed contract succeeds on valid IR7/opset12 shape [1,60,80,1,16], allowing targeted variant violations below
    expect(
      normalizeHitnetPropGatherNdRankForWebGpu(baselineInvalidTotalCount, {
        requireHitnetContract: false,
        backendPreference: 'webgpu',
      }).normalizedCount
    ).toBe(2);

    // Explicitly check WASM blocked without allowDiagnosticTransformedWasm
    const wasmBlocked = normalizeHitnetPropGatherNdRankForWebGpu(baselineInvalidTotalCount, {
      requireHitnetContract: false,
      backendPreference: 'wasm',
    });
    expect(wasmBlocked.normalizedCount).toBe(0);
    expect(wasmBlocked.bytes).toBe(baselineInvalidTotalCount);

    // All contract violations fail closed and return untouched bytes
    for (const badBytes of [
      buildContractCandidate({ irVersion: 6 }),
      buildContractCandidate({ opset: 11 }),
      buildContractCandidate({ dataDims: [1, 60, 80, 1, 8] }),
      buildContractCandidate({ idxDims: [1, 60, 80, 1, 2] }),
      buildContractCandidate({ idxType: 6 }), // int32 instead of int64
      buildContractCandidate({ batchDims: 2 }),
      buildContractCandidate({ extraRank5GatherNd: true }),
      buildContractCandidate({ omitSecondPropNode: true }),
    ]) {
      const res = normalizeHitnetPropGatherNdRankForWebGpu(badBytes, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: true,
      });
      expect(res.normalizedCount).toBe(0);
      expect(res.bytes).toBe(badBytes);
    }
  });

  it('appendOnnxGraphOutputs preserves source graph & fails closed on missing/dynamic/duplicate/over-limit metadata', () => {
    const buildTapTestModel = (withDynamicInput = false) => {
      const nodes = [
        encodeNode('Identity', 'id_mid', ['x'], ['mid']),
        encodeNode('Identity', 'id_out', ['mid'], ['final_out']),
      ];
      const graphParts: Uint8Array[] = [
        ...nodes,
        encodeStrField(2, 'tap_test_graph'),
        encodeLenField(11, encodeValueInfoTensor('x', 1, withDynamicInput ? [1, 0, 4, 4] : [1, 2, 4, 4])),
        encodeLenField(12, encodeValueInfoTensor('final_out', 1, withDynamicInput ? [1, 0, 4, 4] : [1, 2, 4, 4])),
      ];
      return concatBytes([
        encodeVarintField(1, 7),
        encodeLenField(7, concatBytes(graphParts)),
        encodeLenField(8, concatBytes([encodeStrField(1, ''), encodeVarintField(2, 12)])),
      ]);
    };

    const staticModel = buildTapTestModel(false);
    const snapshot = staticModel.slice();
    const tapped = appendOnnxGraphOutputs(staticModel, ['mid'], { maxExtraOutputs: 2 });
    expect(Buffer.compare(staticModel, snapshot)).toBe(0);
    expect(tapped.baseOutputNames).toEqual(['final_out']);
    expect(tapped.addedOutputs).toEqual(['mid']);
    expect(tapped.allOutputNames).toEqual(['final_out', 'mid']);
    expect(tapped.outputMeta.mid).toEqual({ elemType: 1, dims: [1, 2, 4, 4] });
    expect(tapped.outputMeta.final_out).toEqual({ elemType: 1, dims: [1, 2, 4, 4] });

    // Fail-closed guards
    expect(() => appendOnnxGraphOutputs(staticModel, [])).toThrow(/at least 1/);
    expect(() => appendOnnxGraphOutputs(staticModel, ['mid', 'mid'])).toThrow(/Duplicate/);
    expect(() => appendOnnxGraphOutputs(staticModel, ['final_out'])).toThrow(/already a GraphProto.output/);
    expect(() => appendOnnxGraphOutputs(staticModel, ['does_not_exist'])).toThrow(/Unknown or non-static/);
    expect(() => appendOnnxGraphOutputs(staticModel, ['x', 'mid'], { maxExtraOutputs: 1 })).toThrow(
      /Bounded tap limit exceeded/
    );

    const dynamicModel = buildTapTestModel(true);
    expect(() => appendOnnxGraphOutputs(dynamicModel, ['mid'])).toThrow("Missing/dynamic static metadata for existing output 'final_out'");
  });

  it('verifies post-prop0 P7..P11 checkpoints and bounded full-graph tap groups in ORT WASM on cached HITNet', async () => {
    const hitnetPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (!existsSync(hitnetPath)) {
      return;
    }
    const rawBytes = new Uint8Array(readFileSync(hitnetPath));
    const xData = new Float32Array(1 * 6 * 240 * 320);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = ((i * 1315423911) % 256) / 255;
    }
    const xTensor = new ort.Tensor('float32', xData, [1, 6, 240, 320]);

    // 1. Checkpoints P7..P11: exact dual-branch shapes + ORT WASM parity (Raw disabled vs H1 all)
    const expectedDimsByMode: Record<'p7' | 'p8' | 'p9' | 'p10' | 'p11', Record<string, number[]>> = {
      p7: {
        'shared/refinement_l0/ExpandDims': [1, 60, 80, 1, 16],
        'shared_1/refinement_l0/ExpandDims1': [1, 60, 80, 1, 16],
      },
      p8: {
        'shared/refinement_l0/UpsampleHypotheses/Reshape': [1, 120, 160, 1, 1],
        'shared/refinement_l0/UpsampleHypotheses/Reshape_1': [1, 120, 160, 1, 16],
        'shared/refinement_l0/strided_slice_17__750:0': [1, 120, 160, 16],
        'shared_1/refinement_l0/UpsampleHypotheses/Reshape': [1, 120, 160, 1, 1],
        'shared_1/refinement_l0/UpsampleHypotheses/Reshape_1': [1, 120, 160, 1, 16],
        'shared_1/refinement_l0/strided_slice_17__462:0': [1, 120, 160, 16],
      },
      p9: {
        'shared/refinement_l1/ExpandDims': [1, 120, 160, 1, 16],
        'shared_1/refinement_l1/ExpandDims1': [1, 120, 160, 1, 16],
      },
      p10: {
        'shared/refinement_l1/UpsampleHypotheses/Reshape': [1, 240, 320, 1, 1],
        'shared/refinement_l1/UpsampleHypotheses/Reshape_1': [1, 240, 320, 1, 16],
        'shared/refinement_l1/strided_slice_17__821:0': [1, 240, 320, 16],
        'shared_1/refinement_l1/UpsampleHypotheses/Reshape': [1, 240, 320, 1, 1],
        'shared_1/refinement_l1/UpsampleHypotheses/Reshape_1': [1, 240, 320, 1, 16],
        'shared_1/refinement_l1/strided_slice_173__533:0': [1, 240, 320, 16],
      },
      p11: {
        'shared/refinement_l2/add': [1, 16, 240, 320],
        'shared_1/refinement_l2/add': [1, 16, 240, 320],
      },
    };

    for (const mode of ['p7', 'p8', 'p9', 'p10', 'p11'] as const) {
      const arms = prepareHitnetSubgraphArms(rawBytes, mode, { variant: 'h1' });
      expect(arms.rawArm.targetOutputs).toEqual(Object.keys(expectedDimsByMode[mode]));
      expect(arms.candidateArm.targetOutputs).toEqual(arms.rawArm.targetOutputs);
      const sRaw = await ort.InferenceSession.create(arms.rawArm.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'disabled',
        logSeverityLevel: 3,
      });
      const sH1 = await ort.InferenceSession.create(arms.candidateArm.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        logSeverityLevel: 3,
      });
      try {
        const rRaw = await sRaw.run({ [sRaw.inputNames[0]]: xTensor });
        const rH1 = await sH1.run({ [sH1.inputNames[0]]: xTensor });
        for (const outName of arms.rawArm.targetOutputs) {
          expect(Array.from(rH1[outName].dims), `${mode}:${outName}`).toEqual(
            expectedDimsByMode[mode][outName]
          );
          const s = summarizeRawTensorAndParity(
            rH1[outName].data as Float32Array,
            rRaw[outName].data as Float32Array
          );
          expect(s.finiteCount, `${mode}:${outName}`).toBe(s.length);
          expect(s.nonFiniteMismatchCount, `${mode}:${outName}`).toBe(0);
          expect(s.maxAbsDiff, `${mode}:${outName}`).toBeLessThan(1e-3);
        }
      } finally {
        await sRaw.release();
        await sH1.release();
      }
    }

    // 2. Bounded full-graph tap groups preserve exact untouched H1 bytes SHA + retain both final disparities
    const rawSess = await ort.InferenceSession.create(rawBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      logSeverityLevel: 3,
    });
    let rawRefDisparity: Float32Array;
    let rawSecDisparity: Float32Array;
    try {
      const rawOut = await rawSess.run({ [rawSess.inputNames[0]]: xTensor });
      rawRefDisparity = new Float32Array(rawOut.reference_output_disparity.data as Float32Array);
      rawSecDisparity = new Float32Array(rawOut.secondary_output_disparity.data as Float32Array);
    } finally {
      await rawSess.release();
    }

    for (const tapId of ['p5-p6', 'p8', 'p10'] as const) {
      const prepared = prepareHitnetFullGraphTapGroup(rawBytes, tapId);
      const baseH1Sha = createHash('sha256').update(prepared.baseH1Bytes).digest('hex');
      expect(baseH1Sha).toBe('3d7278a54d3a5d8d356b71716da72b901ed75f7c21ec012f420d17205b6a69ed');
      expect(prepared.tappedH1.baseOutputNames).toEqual([
        'reference_output_disparity',
        'secondary_output_disparity',
      ]);
      expect(prepared.tappedH1.addedOutputs).toEqual(prepared.group.outputs);
      expect(prepared.tappedH1.allOutputNames).toHaveLength(4);

      const tapSess = await ort.InferenceSession.create(prepared.tappedH1.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'basic',
        logSeverityLevel: 3,
      });
      try {
        const tapOut = await tapSess.run({ [tapSess.inputNames[0]]: xTensor });
        const sRef = summarizeRawTensorAndParity(
          tapOut.reference_output_disparity.data as Float32Array,
          rawRefDisparity
        );
        const sSec = summarizeRawTensorAndParity(
          tapOut.secondary_output_disparity.data as Float32Array,
          rawSecDisparity
        );
        expect(sRef.finiteCount).toBe(76800);
        expect(sRef.maxAbsDiff).toBeLessThan(1e-3);
        expect(sSec.finiteCount).toBe(76800);
        expect(sSec.maxAbsDiff).toBeLessThan(1e-3);

        for (const tapName of prepared.group.outputs) {
          expect(Array.from(tapOut[tapName].dims)).toEqual(
            prepared.tappedH1.outputMeta[tapName].dims
          );
          const tapArr =
            tapOut[tapName].data instanceof BigInt64Array
              ? Float32Array.from(tapOut[tapName].data as BigInt64Array, Number)
              : new Float32Array(tapOut[tapName].data as Float32Array);
          const sTap = summarizeRawTensorAndParity(tapArr, null);
          expect(sTap.finiteCount).toBe(sTap.length);
        }
      } finally {
        await tapSess.release();
      }
    }
  }, 180000);

  it('verifies earlier full-graph both-branch tap pairs (P4, P2-*, P2b, P3c) and read-only rank4 GatherND identity check in ORT WASM', async () => {
    const hitnetPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (!existsSync(hitnetPath)) {
      return;
    }
    const rawBytes = new Uint8Array(readFileSync(hitnetPath));
    const xData = new Float32Array(1 * 6 * 240 * 320);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = ((i * 1315423911) % 256) / 255;
    }
    const xTensor = new ort.Tensor('float32', xData, [1, 6, 240, 320]);

    const rawSess = await ort.InferenceSession.create(rawBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      logSeverityLevel: 3,
    });
    let rawRefDisparity: Float32Array;
    let rawSecDisparity: Float32Array;
    try {
      const rawOut = await rawSess.run({ [rawSess.inputNames[0]]: xTensor });
      rawRefDisparity = new Float32Array(rawOut.reference_output_disparity.data as Float32Array);
      rawSecDisparity = new Float32Array(rawOut.secondary_output_disparity.data as Float32Array);
    } finally {
      await rawSess.release();
    }

    const toF32 = (data: unknown): Float32Array =>
      data instanceof BigInt64Array
        ? Float32Array.from(data, (v) => Number(v))
        : new Float32Array(data as ArrayLike<number>);

    const expectedEarlyMeta: Record<string, { elemType: number; dims: number[] }> = {
      'level0/shared/prop0/add': { elemType: 1, dims: [1, 60, 80, 1, 16] },
      'level0_1/shared/prop0/add': { elemType: 1, dims: [1, 60, 80, 1, 16] },
      'level0/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis': { elemType: 1, dims: [1, 60, 80, 1] },
      'level0_1/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis': { elemType: 1, dims: [1, 60, 80, 1] },
      'level0/level_init/ArgMin': { elemType: 7, dims: [1, 60, 80] },
      'level0_1/level_init/ArgMin': { elemType: 7, dims: [1, 60, 80] },
      'level0/level_init/Cast': { elemType: 1, dims: [1, 60, 80, 1] },
      'level0_1/level_init/Cast': { elemType: 1, dims: [1, 60, 80, 1] },
      'level0/level_init/init_to_prop/ExpandDims': { elemType: 1, dims: [1, 60, 80, 1, 16] },
      'level0_1/level_init/init_to_prop/ExpandDims': { elemType: 1, dims: [1, 60, 80, 1, 16] },
      'level0/shared/prop0/prop_warp/Reshape_1': { elemType: 1, dims: [1, 60, 80, 1, 16] },
      'level0_1/shared/prop0/prop_warp/Reshape_1': { elemType: 1, dims: [1, 60, 80, 1, 16] },
    };

    for (const tapId of ['p4', 'p2-gather', 'p2-argmin', 'p2-cast', 'p2b', 'p3c'] as const) {
      const prepared = prepareHitnetFullGraphTapGroup(rawBytes, tapId);
      const baseH1Sha = createHash('sha256').update(prepared.baseH1Bytes).digest('hex');
      expect(baseH1Sha).toBe('3d7278a54d3a5d8d356b71716da72b901ed75f7c21ec012f420d17205b6a69ed');
      expect(prepared.tappedH1.addedOutputs).toHaveLength(2);
      expect(prepared.tappedH1.allOutputNames).toHaveLength(4);
      for (const tapName of prepared.group.outputs) {
        expect(prepared.tappedH1.outputMeta[tapName]).toEqual(expectedEarlyMeta[tapName]);
      }

      const tapSess = await ort.InferenceSession.create(prepared.tappedH1.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'basic',
        logSeverityLevel: 3,
      });
      try {
        const tapOut = await tapSess.run({ [tapSess.inputNames[0]]: xTensor });
        const sRef = summarizeRawTensorAndParity(
          toF32(tapOut.reference_output_disparity.data),
          rawRefDisparity
        );
        const sSec = summarizeRawTensorAndParity(
          toF32(tapOut.secondary_output_disparity.data),
          rawSecDisparity
        );
        expect(sRef.finiteCount).toBe(76800);
        expect(sRef.maxAbsDiff).toBeLessThan(1e-3);
        expect(sSec.finiteCount).toBe(76800);
        expect(sSec.maxAbsDiff).toBeLessThan(1e-3);

        for (const tapName of prepared.group.outputs) {
          expect(Array.from(tapOut[tapName].dims)).toEqual(expectedEarlyMeta[tapName].dims);
          const tapStats = summarizeRawTensorAndParity(toF32(tapOut[tapName].data), null);
          expect(tapStats.finiteCount).toBe(tapStats.length);
        }
      } finally {
        await tapSess.release();
      }
    }

    // Optional rank4 GatherND read-only check: first verify ordinary singleton index values are all 0
    const idxPrepared = prepareHitnetFullGraphTapGroup(rawBytes, 'p5-rank4-indices');
    const idxSess = await ort.InferenceSession.create(idxPrepared.tappedH1.bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'basic',
      logSeverityLevel: 3,
    });
    let indicesVerifiedZero = true;
    try {
      const idxOut = await idxSess.run({ [idxSess.inputNames[0]]: xTensor });
      for (const idxName of idxPrepared.group.outputs) {
        expect(idxPrepared.tappedH1.outputMeta[idxName]).toEqual({
          elemType: 7,
          dims: [1, 4800, 1, 1],
        });
        const chk = verifySingletonZeroIndices(toF32(idxOut[idxName].data));
        expect(chk.ok).toBe(true);
        expect(chk.length).toBe(4800);
        expect(chk.nonZeroCount).toBe(0);
        if (!chk.ok) indicesVerifiedZero = false;
      }
    } finally {
      await idxSess.release();
    }

    expect(indicesVerifiedZero).toBe(true);
    for (const ioId of ['p5-rank4-io-primary', 'p5-rank4-io-secondary'] as const) {
      const ioPrepared = prepareHitnetFullGraphTapGroup(rawBytes, ioId);
      const ioSess = await ort.InferenceSession.create(ioPrepared.tappedH1.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'basic',
        logSeverityLevel: 3,
      });
      try {
        const ioOut = await ioSess.run({ [ioSess.inputNames[0]]: xTensor });
        const [dataName, outName] = ioPrepared.group.outputs;
        const dataArr = toF32(ioOut[dataName].data);
        const outArr = toF32(ioOut[outName].data);
        const idStats = summarizeRawTensorAndParity(outArr, dataArr);
        expect(idStats.finiteCount).toBe(1 * 4800 * 1 * 16);
        expect(idStats.maxAbsDiff).toBe(0);
      } finally {
        await ioSess.release();
      }
    }
  }, 240000);
});

describe('normalizeHitnetLevelInitGatherNdRankForWebGpu & normalizeHitnetLevelInitReduceMinForWebGpu', () => {
  ort.env.wasm.numThreads = 1;

  const buildSyntheticLevelInitCostModel = (opts?: {
    irVersion?: number;
    opset?: number;
    omitAbsSumLineage?: boolean;
    extraUnsqueezeConsumer?: boolean;
    badArgMinAxis?: number;
    badArgMinKeepDims?: number;
    badSelectLastIndex?: number;
    badBatchDims?: number;
  }) => {
    const H = 2;
    const W = 3;
    const D = 4;
    const C = 2;
    const nodes: Uint8Array[] = [];
    if (opts?.omitAbsSumLineage) {
      nodes.push(
        encodeNode('Reshape', 'cost_reshape', ['raw_cost', 'cost_shape'], ['cost_4d'])
      );
    } else {
      nodes.push(
        encodeNode('Sub', 'cost_sub', ['left_feat', 'right_feat'], ['cost_diff']),
        encodeNode('Abs', 'cost_abs', ['cost_diff'], ['cost_abs_out']),
        encodeNode('ReduceSum', 'cost_sum', ['cost_abs_out'], ['cost_sum_out'], [
          encodeAttrInts('axes', [3]),
          encodeAttrInt('keepdims', 0),
        ]),
        encodeNode('Reshape', 'cost_reshape', ['cost_sum_out', 'cost_shape'], ['cost_4d'])
      );
    }
    nodes.push(
      encodeNode('ArgMin', 'level_init_argmin', ['cost_4d'], ['argmin_out'], [
        encodeAttrInt('axis', opts?.badArgMinAxis ?? 3),
        encodeAttrInt('keepdims', opts?.badArgMinKeepDims ?? 0),
        encodeAttrInt('select_last_index', opts?.badSelectLastIndex ?? 0),
      ]),
      encodeNode('Reshape', 'level_init_expand', ['argmin_out', 'expand_shape'], ['expand_out']),
      encodeNode('Cast', 'level_init_cast', ['expand_out'], ['disp_f32'], [encodeAttrInt('to', 1)]),
      encodeNode('Unsqueeze', 'level_init_unsq', ['expand_out'], ['unsq_out'], [encodeAttrInts('axes', [-1])]),
      encodeNode('GatherND', 'level_init_gathernd', ['cost_4d', 'unsq_out'], ['cost_min'], [
        encodeAttrInt('batch_dims', opts?.badBatchDims ?? 3),
      ])
    );
    if (opts?.extraUnsqueezeConsumer) {
      nodes.push(encodeNode('Identity', 'extra_unsq_consumer', ['unsq_out'], ['extra_unsq_out']));
    }

    const graphParts: Uint8Array[] = [
      ...nodes,
      encodeStrField(2, 'synthetic_level_init_cost_graph'),
      encodeLenField(5, encodeIntTensorProto('cost_shape', [4], 7, [1, H, W, D])),
      encodeLenField(5, encodeIntTensorProto('expand_shape', [4], 7, [1, H, W, 1])),
    ];
    if (opts?.omitAbsSumLineage) {
      graphParts.push(encodeLenField(11, encodeValueInfoTensor('raw_cost', 1, [1, H, W, D])));
    } else {
      graphParts.push(
        encodeLenField(11, encodeValueInfoTensor('left_feat', 1, [1, H, W, C, D])),
        encodeLenField(11, encodeValueInfoTensor('right_feat', 1, [1, H, W, C, D]))
      );
    }
    graphParts.push(
      encodeLenField(12, encodeValueInfoTensor('disp_f32', 1, [1, H, W, 1])),
      encodeLenField(12, encodeValueInfoTensor('cost_min', 1, [1, H, W, 1]))
    );
    if (opts?.extraUnsqueezeConsumer) {
      graphParts.push(encodeLenField(12, encodeValueInfoTensor('extra_unsq_out', 7, [1, H, W, 1, 1])));
    }
    return concatBytes([
      encodeVarintField(1, opts?.irVersion ?? 7),
      encodeLenField(7, concatBytes(graphParts)),
      encodeLenField(8, concatBytes([encodeStrField(1, ''), encodeVarintField(2, opts?.opset ?? 12)])),
    ]);
  };

  it('proves synthetic reshape-rank4 and reduce-min equivalence in ORT WASM including tied minimums', async () => {
    const baseModel = buildSyntheticLevelInitCostModel();
    const baseSnapshot = baseModel.slice();

    const reshapeNorm = normalizeHitnetLevelInitGatherNdRankForWebGpu(baseModel, {
      variantId: 'hitnet-sf',
      backendPreference: 'webgpu',
      requireHitnetContract: false,
    });
    expect(Buffer.compare(baseModel, baseSnapshot)).toBe(0);
    expect(reshapeNorm.mode).toBe('reshape-rank4');
    expect(reshapeNorm.normalizedCount).toBe(1);
    expect(reshapeNorm.normalizedNodeNames).toEqual(['level_init_gathernd']);
    expect(reshapeNorm.totalGatherNdCount).toBe(1);
    expect(validateOnnxGraphNodeInputs(reshapeNorm.bytes)).toMatchObject({
      valid: true,
      missingInputs: [],
    });

    const inspReshape = inspectNodesAndIntInits(reshapeNorm.bytes);
    // Verify ArgMin, Cast, Unsqueeze, and all initializers remain preserved (no nodes removed)
    expect(inspReshape.nodes.some((n) => n.opType === 'ArgMin' && n.name === 'level_init_argmin')).toBe(true);
    expect(inspReshape.nodes.some((n) => n.opType === 'Cast' && n.name === 'level_init_cast')).toBe(true);
    expect(inspReshape.nodes.some((n) => n.opType === 'Unsqueeze' && n.name === 'level_init_unsq')).toBe(true);
    const reshapedGnd = inspReshape.nodes.find((n) => n.opType === 'GatherND')!;
    const preDataReshape = inspReshape.nodes.find((n) => n.outputs[0] === reshapedGnd.inputs[0])!;
    const preIdxReshape = inspReshape.nodes.find((n) => n.outputs[0] === reshapedGnd.inputs[1])!;
    const postReshape = inspReshape.nodes.find((n) => n.inputs[0] === reshapedGnd.outputs[0])!;
    expect(inspReshape.inits.get(preDataReshape.inputs[1])).toEqual([1n, 6n, 4n, 1n]);
    expect(inspReshape.inits.get(preIdxReshape.inputs[1])).toEqual([1n, 6n, 1n, 1n]);
    expect(inspReshape.inits.get(postReshape.inputs[1])).toEqual([1n, 2n, 3n, 1n]);
    expect(postReshape.outputs[0]).toBe('cost_min');

    const reduceMinNorm = normalizeHitnetLevelInitReduceMinForWebGpu(baseModel, {
      variantId: 'hitnet-sf',
      backendPreference: 'webgpu',
      requireHitnetContract: false,
    });
    expect(reduceMinNorm.mode).toBe('reduce-min');
    expect(reduceMinNorm.normalizedCount).toBe(1);
    expect(reduceMinNorm.verifiedCostLineageCount).toBe(1);
    expect(reduceMinNorm.documentedLimitations).toEqual([...HITNET_LEVEL_INIT_REDUCEMIN_LIMITATIONS]);
    expect(validateOnnxGraphNodeInputs(reduceMinNorm.bytes)).toMatchObject({
      valid: true,
      missingInputs: [],
    });
    const inspReduceMin = inspectNodesAndIntInits(reduceMinNorm.bytes);
    expect(inspReduceMin.nodes.some((n) => n.opType === 'GatherND')).toBe(false);
    expect(inspReduceMin.nodes.some((n) => n.opType === 'ArgMin' && n.name === 'level_init_argmin')).toBe(true);
    expect(inspReduceMin.nodes.some((n) => n.opType === 'Cast' && n.name === 'level_init_cast')).toBe(true);
    const rmNode = inspReduceMin.nodes.find((n) => n.opType === 'ReduceMin')!;
    expect(rmNode.inputs).toEqual(['cost_4d']);
    expect(rmNode.outputs).toEqual(['cost_min']);

    // Build cost inputs with tied minimums (including duplicate min values & exact zero Diff -> +0.0/-0.0)
    const leftFeat = new Float32Array(1 * 2 * 3 * 2 * 4);
    const rightFeat = new Float32Array(1 * 2 * 3 * 2 * 4);
    for (let i = 0; i < leftFeat.length; i++) {
      leftFeat[i] = ((i * 7 + 3) % 11) * 0.25 - 1.0;
      rightFeat[i] = ((i * 5 + 1) % 11) * 0.25 - 1.0;
    }
    // Force exact ties at pixel (0,0) and (1,2)
    for (let c = 0; c < 2; c++) {
      rightFeat[((0 * 3 + 0) * 2 + c) * 4 + 1] = leftFeat[((0 * 3 + 0) * 2 + c) * 4 + 1];
      rightFeat[((0 * 3 + 0) * 2 + c) * 4 + 3] = leftFeat[((0 * 3 + 0) * 2 + c) * 4 + 3];
    }
    const feeds = {
      left_feat: new ort.Tensor('float32', leftFeat, [1, 2, 3, 2, 4]),
      right_feat: new ort.Tensor('float32', rightFeat, [1, 2, 3, 2, 4]),
    };

    const sessOrig = await ort.InferenceSession.create(baseModel, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      logSeverityLevel: 3,
    });
    const sessReshape = await ort.InferenceSession.create(reshapeNorm.bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3,
    });
    const sessReduceMin = await ort.InferenceSession.create(reduceMinNorm.bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3,
    });
    try {
      const oOrig = await sessOrig.run(feeds);
      const oReshape = await sessReshape.run(feeds);
      const oReduceMin = await sessReduceMin.run(feeds);

      for (const outName of ['disp_f32', 'cost_min'] as const) {
        expect(oReshape[outName].dims).toEqual([1, 2, 3, 1]);
        expect(oReduceMin[outName].dims).toEqual([1, 2, 3, 1]);
        expect(
          summarizeRawTensorAndParity(
            oReshape[outName].data as Float32Array,
            oOrig[outName].data as Float32Array
          ).maxAbsDiff
        ).toBe(0);
        expect(
          summarizeRawTensorAndParity(
            oReduceMin[outName].data as Float32Array,
            oOrig[outName].data as Float32Array
          ).maxAbsDiff
        ).toBe(0);
      }
      // Verify first tied minimum index (=1) selected per select_last_index=0
      expect((oReshape.disp_f32.data as Float32Array)[0]).toBe(1);
      expect((oReduceMin.disp_f32.data as Float32Array)[0]).toBe(1);
      expect((oReshape.cost_min.data as Float32Array)[0]).toBe(0);
      expect((oReduceMin.cost_min.data as Float32Array)[0]).toBe(0);
    } finally {
      await sessOrig.release();
      await sessReshape.release();
      await sessReduceMin.release();
    }
  });

  it('fails closed on contract, opset, irVersion, wasm guard, lineage mismatch, and ReduceMin unproven cost', () => {
    const validSynth = buildSyntheticLevelInitCostModel();

    // 1) Strict HITNet contract rejects synthetic graph without exact HITNet signatures
    expect(
      normalizeHitnetLevelInitGatherNdRankForWebGpu(validSynth, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: true,
      }).normalizedCount
    ).toBe(0);
    expect(
      normalizeHitnetLevelInitReduceMinForWebGpu(validSynth, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: true,
      }).normalizedCount
    ).toBe(0);

    // 2) WASM blocked unless allowDiagnosticTransformedWasm=true
    const wasmReshapeBlocked = normalizeHitnetLevelInitGatherNdRankForWebGpu(validSynth, {
      backendPreference: 'wasm',
      requireHitnetContract: false,
    });
    expect(wasmReshapeBlocked.normalizedCount).toBe(0);
    expect(wasmReshapeBlocked.bytes).toBe(validSynth);

    const wasmRmBlocked = normalizeHitnetLevelInitReduceMinForWebGpu(validSynth, {
      backendPreference: 'wasm',
      requireHitnetContract: false,
    });
    expect(wasmRmBlocked.normalizedCount).toBe(0);
    expect(wasmRmBlocked.bytes).toBe(validSynth);

    // 3) Bad ArgMin axis / keepdims / select_last_index / batch_dims / opset 18 fail closed on both
    for (const badModel of [
      buildSyntheticLevelInitCostModel({ opset: 18 }),
      buildSyntheticLevelInitCostModel({ badArgMinAxis: 2 }),
      buildSyntheticLevelInitCostModel({ badArgMinKeepDims: 1 }),
      buildSyntheticLevelInitCostModel({ badSelectLastIndex: 1 }),
      buildSyntheticLevelInitCostModel({ badBatchDims: 2 }),
    ]) {
      const r1 = normalizeHitnetLevelInitGatherNdRankForWebGpu(badModel, {
        backendPreference: 'webgpu',
        requireHitnetContract: false,
      });
      const r2 = normalizeHitnetLevelInitReduceMinForWebGpu(badModel, {
        backendPreference: 'webgpu',
        requireHitnetContract: false,
      });
      expect(r1.normalizedCount).toBe(0);
      expect(r1.bytes).toBe(badModel);
      expect(r2.normalizedCount).toBe(0);
      expect(r2.bytes).toBe(badModel);
    }

    // 4) Unproven cost lineage (missing Abs -> ReduceSum) blocks reduce-min even though reshape-rank4 preserves exact GatherND semantics
    const noCostLineage = buildSyntheticLevelInitCostModel({ omitAbsSumLineage: true });
    expect(
      normalizeHitnetLevelInitReduceMinForWebGpu(noCostLineage, {
        backendPreference: 'webgpu',
        requireHitnetContract: false,
      }).normalizedCount
    ).toBe(0);
    expect(
      normalizeHitnetLevelInitGatherNdRankForWebGpu(noCostLineage, {
        backendPreference: 'webgpu',
        requireHitnetContract: false,
      }).normalizedCount
    ).toBe(1);

    // 5) Extra consumer on Unsqueeze must never drop the live Unsqueeze in ReduceMin mode
    const sharedUnsq = buildSyntheticLevelInitCostModel({ extraUnsqueezeConsumer: true });
    const rmShared = normalizeHitnetLevelInitReduceMinForWebGpu(sharedUnsq, {
      backendPreference: 'webgpu',
      requireHitnetContract: false,
    });
    expect(rmShared.normalizedCount).toBe(1);
    expect(validateOnnxGraphNodeInputs(rmShared.bytes)).toMatchObject({
      valid: true,
      missingInputs: [],
    });
    expect(
      inspectNodesAndIntInits(rmShared.bytes).nodes.some((n) => n.name === 'level_init_unsq')
    ).toBe(true);
  });

  it('applies L1-Reshape and L1-ReduceMin on cached real HITNet ONNX on top of H1, preserving ArgMin/Cast and matching Raw WASM on both disparities & P2a/P2b taps', async () => {
    const hitnetPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (!existsSync(hitnetPath)) {
      return;
    }
    const rawBytes = new Uint8Array(readFileSync(hitnetPath));

    const l1ReshapePrep = prepareHitnetLevelInitCandidateFullGraphBytes(rawBytes, 'reshape-rank4');
    const l1RmPrep = prepareHitnetLevelInitCandidateFullGraphBytes(rawBytes, 'reduce-min');

    expect(l1ReshapePrep.levelInitResult.normalizedCount).toBe(2);
    expect(l1ReshapePrep.levelInitResult.totalGatherNdCount).toBe(10);
    expect(l1ReshapePrep.levelInitResult.verifiedCostLineageCount).toBe(2);
    expect(l1ReshapePrep.levelInitResult.normalizedNodeNames).toEqual([
      'level0/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis',
      'level0_1/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis',
    ]);
    expect(validateOnnxGraphNodeInputs(l1ReshapePrep.bytes)).toMatchObject({
      valid: true,
      missingInputs: [],
    });
    expect(inferOnnxStaticTensorMeta(l1ReshapePrep.bytes).valid).toBe(true);

    expect(l1RmPrep.levelInitResult.normalizedCount).toBe(2);
    expect(l1RmPrep.levelInitResult.verifiedCostLineageCount).toBe(2);
    expect(l1RmPrep.levelInitResult.documentedLimitations.length).toBeGreaterThanOrEqual(3);
    expect(validateOnnxGraphNodeInputs(l1RmPrep.bytes)).toMatchObject({
      valid: true,
      missingInputs: [],
    });
    expect(inferOnnxStaticTensorMeta(l1RmPrep.bytes).valid).toBe(true);

    const xData = new Float32Array(1 * 6 * 240 * 320);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = ((i * 1315423911) % 256) / 255;
    }
    const xTensor = new ort.Tensor('float32', xData, [1, 6, 240, 320]);

    const runFull = async (bytes: Uint8Array, opt: 'disabled' | 'basic' | 'all') => {
      const sess = await ort.InferenceSession.create(bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: opt,
        logSeverityLevel: 3,
      });
      try {
        const out = await sess.run({ [sess.inputNames[0]]: xTensor });
        return {
          ref: new Float32Array(out.reference_output_disparity.data as Float32Array),
          sec: new Float32Array(out.secondary_output_disparity.data as Float32Array),
        };
      } finally {
        await sess.release();
      }
    };

    const rawDis = await runFull(rawBytes, 'disabled');
    for (const [label, candBytes] of [
      ['L1-Reshape', l1ReshapePrep.bytes],
      ['L1-ReduceMin', l1RmPrep.bytes],
    ] as const) {
      for (const opt of ['disabled', 'basic'] as const) {
        const out = await runFull(candBytes, opt);
        for (const [bName, arr, refArr] of [
          ['primary', out.ref, rawDis.ref],
          ['secondary', out.sec, rawDis.sec],
        ] as const) {
          const s = summarizeRawTensorAndParity(arr, refArr);
          expect(s.finiteCount, `${label}:${opt}:${bName}`).toBe(76800);
          expect(s.nonFiniteMismatchCount, `${label}:${opt}:${bName}`).toBe(0);
          expect(s.maxAbsDiff, `${label}:${opt}:${bName}`).toBeLessThan(1e-3);
          expect(s.p99AbsDiff, `${label}:${opt}:${bName}`).toBeLessThan(1e-4);
          expect(s.meanAbsDiff, `${label}:${opt}:${bName}`).toBeLessThan(1e-5);
        }
      }
    }

    // Verify bounded P2a & P2b channelwise tap groups + splitters on reshape-rank4 full graph
    const p2bTap = prepareHitnetFullGraphTapGroup(rawBytes, 'p2b', 'reshape-rank4');
    const p2aConcatOutTap = prepareHitnetFullGraphTapGroup(rawBytes, 'p2a-concat-out', 'reshape-rank4');
    const p2aConcatFeTap = prepareHitnetFullGraphTapGroup(rawBytes, 'p2a-concat-fe', 'reshape-rank4');

    const tapChecks: Array<{
      id: 'p2b' | 'p2a-concat-out' | 'p2a-concat-fe';
      prepared: ReturnType<typeof prepareHitnetFullGraphTapGroup>;
      expectedDims: number[];
    }> = [
      { id: 'p2b', prepared: p2bTap, expectedDims: [1, 60, 80, 1, 16] },
      { id: 'p2a-concat-out', prepared: p2aConcatOutTap, expectedDims: [1, 60, 80, 49] },
      { id: 'p2a-concat-fe', prepared: p2aConcatFeTap, expectedDims: [1, 60, 80, 48] },
    ];

    for (const tc of tapChecks) {
      const sTap = await ort.InferenceSession.create(tc.prepared.tappedH1.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'basic',
        logSeverityLevel: 3,
      });
      try {
        const res = await sTap.run({ [sTap.inputNames[0]]: xTensor });
        const refS = summarizeRawTensorAndParity(
          new Float32Array(res.reference_output_disparity.data as Float32Array),
          rawDis.ref
        );
        expect(refS.finiteCount, tc.id).toBe(76800);
        expect(refS.maxAbsDiff, tc.id).toBeLessThan(1e-3);

        const [t0, t1] = tc.prepared.group.outputs;
        expect(Array.from(res[t0].dims), `${tc.id}:${t0}`).toEqual(tc.expectedDims);
        expect(Array.from(res[t1].dims), `${tc.id}:${t1}`).toEqual(tc.expectedDims);
        const d0 = new Float32Array(res[t0].data as Float32Array);
        const d1 = new Float32Array(res[t1].data as Float32Array);

        if (tc.id === 'p2b') {
          const p2bSplit = summarizeP2bChannelwiseParity(d0, d0);
          expect(p2bSplit.ch0Disp.length).toBe(4800);
          expect(p2bSplit.ch1_2Zeros.maxAbsDiff).toBe(0);
          expect(p2bSplit.ch1CandidateDist.nonZeroCount).toBe(0);
          expect(p2bSplit.ch2CandidateDist.nonZeroCount).toBe(0);
          expect(p2bSplit.ch3_15Features.length).toBe(4800 * 13);
          expect(p2bSplit.channels).toHaveLength(16);
        } else if (tc.id === 'p2a-concat-out') {
          const cSplit = summarizeInitToPropConcatOutSplit(d0, d0);
          expect(cSplit.ch0CostMin.length).toBe(4800);
          expect(cSplit.ch0CostMin.min).toBeGreaterThanOrEqual(0);
          expect(cSplit.ch1_48Features.length).toBe(4800 * 48);
        } else {
          expect(summarizeRawTensorAndParity(d0, null).finiteCount).toBe(4800 * 48);
          expect(summarizeRawTensorAndParity(d1, null).finiteCount).toBe(4800 * 48);
        }
      } finally {
        await sTap.release();
      }
    }
  }, 180000);
});

describe('normalizeHitnetInitToPropZeroSlantForWebGpu (M1-ConstZero2Ch)', () => {
  ort.env.wasm.numThreads = 1;

  const buildSyntheticInitToPropZeroSlantModel = (opts?: {
    irVersion?: number;
    opset?: number;
    zeroInitVal?: bigint;
    extraConsumerOnMul?: boolean;
    exportZeroLikeOutput?: boolean;
    sharedZeroInitWithLiveNode?: boolean;
    argMinKeepDims?: number;
  }) => {
    const H = 2;
    const W = 3;
    const D = 4;
    const featC = 5;
    const nodes: Uint8Array[] = [
      encodeNode('ArgMin', 'level_init/ArgMin', ['cost_4d'], ['argmin_i64'], [
        encodeAttrInt('axis', 3),
        encodeAttrInt('keepdims', opts?.argMinKeepDims ?? 0),
        encodeAttrInt('select_last_index', 0),
      ]),
      encodeNode('Reshape', 'level_init/ExpandDims_2', ['argmin_i64', 'shape_hw1'], ['expand2_i64']),
      encodeNode('Cast', 'level_init/Cast', ['expand2_i64'], ['disp_f32'], [encodeAttrInt('to', 1)]),
      encodeNode('Reshape', 'disp_nchw', ['disp_f32', 'shape_1chw'], ['disp_slot0']),
      // Dead zero chain
      encodeNode('Cast', 'Cast__309', ['disp_f32'], ['cast_i64_out'], [encodeAttrInt('to', 7)]),
      encodeNode('Mul', 'Mul__311', ['cast_i64_out', 'zero__310'], ['mul_zero_out']),
      encodeNode('Cast', 'zeros_like', ['mul_zero_out'], ['zeros_like_f32'], [encodeAttrInt('to', 1)]),
      encodeNode('Reshape', 'Transpose__910', ['zeros_like_f32', 'shape_1chw'], ['zero_slot12']),
      // Concat axis=1 with duplicated zero slot (1, 1, 1, featC -> total 3+featC=8)
      encodeNode(
        'Concat',
        'init_to_prop/concat_1',
        ['disp_slot0', 'zero_slot12', 'zero_slot12', 'feat_nchw'],
        ['concat1_out'],
        [encodeAttrInt('axis', 1)]
      ),
      encodeNode('Transpose', 'Transpose__913', ['concat1_out'], ['concat1_nhwc'], [
        encodeAttrInts('perm', [0, 2, 3, 1]),
      ]),
      encodeNode('Reshape', 'init_to_prop/ExpandDims', ['concat1_nhwc', 'shape_out5d'], ['final_p2b']),
    ];

    if (opts?.extraConsumerOnMul) {
      nodes.push(encodeNode('Identity', 'extra_mul_consumer', ['mul_zero_out'], ['extra_mul_out']));
    }
    if (opts?.sharedZeroInitWithLiveNode) {
      nodes.push(encodeNode('Add', 'live_zero_user', ['argmin_i64', 'zero__310'], ['live_zero_out']));
    }

    const graphParts: Uint8Array[] = [
      ...nodes,
      encodeStrField(2, 'synthetic_m1_zero_slant_graph'),
      encodeLenField(5, encodeIntTensorProto('shape_hw1', [4], 7, [1, H, W, 1])),
      encodeLenField(5, encodeIntTensorProto('shape_1chw', [4], 7, [1, 1, H, W])),
      encodeLenField(5, encodeIntTensorProto('shape_out5d', [5], 7, [1, H, W, 1, 3 + featC])),
      encodeLenField(5, encodeIntTensorProto('zero__310', [], 7, [opts?.zeroInitVal ?? 0n])),
      encodeLenField(11, encodeValueInfoTensor('cost_4d', 1, [1, H, W, D])),
      encodeLenField(11, encodeValueInfoTensor('feat_nchw', 1, [1, featC, H, W])),
      encodeLenField(12, encodeValueInfoTensor('final_p2b', 1, [1, H, W, 1, 3 + featC])),
    ];
    if (opts?.extraConsumerOnMul) {
      graphParts.push(encodeLenField(12, encodeValueInfoTensor('extra_mul_out', 7, [1, H, W, 1])));
    }
    if (opts?.exportZeroLikeOutput) {
      graphParts.push(encodeLenField(12, encodeValueInfoTensor('zeros_like_f32', 1, [1, H, W, 1])));
    }
    if (opts?.sharedZeroInitWithLiveNode) {
      graphParts.push(encodeLenField(12, encodeValueInfoTensor('live_zero_out', 7, [1, H, W])));
    }

    return concatBytes([
      encodeVarintField(1, opts?.irVersion ?? 7),
      encodeLenField(7, concatBytes(graphParts)),
      encodeLenField(8, concatBytes([encodeStrField(1, ''), encodeVarintField(2, opts?.opset ?? 12)])),
    ]);
  };

  it('replaces duplicated zero chain with [1,2,H,W] +0.0f initializer, removes dead nodes/initializer cleanly, and matches WASM identically', async () => {
    const baseModel = buildSyntheticInitToPropZeroSlantModel();
    const snapshot = baseModel.slice();

    const m1 = normalizeHitnetInitToPropZeroSlantForWebGpu(baseModel, {
      variantId: 'hitnet-sf',
      backendPreference: 'webgpu',
      requireHitnetContract: false,
    });
    expect(Buffer.compare(baseModel, snapshot)).toBe(0);
    expect(m1.normalizedCount).toBe(1);
    expect(m1.normalizedNodeNames).toEqual(['init_to_prop/concat_1']);
    expect(m1.removedDeadNodeNames).toEqual([
      'Cast__309',
      'Mul__311',
      'zeros_like',
      'Transpose__910',
    ]);
    expect(m1.removedDeadInitializerNames).toEqual(['zero__310']);
    expect(m1.addedInitializerDims).toEqual([1, 2, 2, 3]);
    expect(m1.addedInitializerName).toBe('__webgpu_m1_init_to_prop_zero_slant_2ch');

    expect(validateOnnxGraphNodeInputs(m1.bytes)).toMatchObject({ valid: true, missingInputs: [] });
    const postMeta = inferOnnxStaticTensorMeta(m1.bytes);
    expect(postMeta.valid).toBe(true);
    expect(postMeta.tensorMeta.zeros_like_f32).toBeUndefined();
    expect(postMeta.tensorMeta.zero_slot12).toBeUndefined();
    expect(postMeta.tensorMeta.zero__310).toBeUndefined();
    expect(postMeta.tensorMeta.final_p2b?.dims).toEqual([1, 2, 3, 1, 8]);

    const cost = new Float32Array(1 * 2 * 3 * 4);
    for (let i = 0; i < cost.length; i++) cost[i] = ((i * 11 + 5) % 17) * 0.5 - 3.0;
    const feat = new Float32Array(1 * 5 * 2 * 3);
    for (let i = 0; i < feat.length; i++) feat[i] = ((i * 7 + 2) % 19) * 0.25 - 1.5;

    const feeds = {
      cost_4d: new ort.Tensor('float32', cost, [1, 2, 3, 4]),
      feat_nchw: new ort.Tensor('float32', feat, [1, 5, 2, 3]),
    };
    const sOrig = await ort.InferenceSession.create(baseModel, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      logSeverityLevel: 3,
    });
    const sM1Dis = await ort.InferenceSession.create(m1.bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      logSeverityLevel: 3,
    });
    const sM1All = await ort.InferenceSession.create(m1.bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3,
    });
    try {
      const oOrig = await sOrig.run(feeds);
      const oDis = await sM1Dis.run(feeds);
      const oAll = await sM1All.run(feeds);
      expect(Array.from(oDis.final_p2b.dims)).toEqual([1, 2, 3, 1, 8]);
      expect(
        summarizeRawTensorAndParity(
          oDis.final_p2b.data as Float32Array,
          oOrig.final_p2b.data as Float32Array
        ).maxAbsDiff
      ).toBe(0);
      expect(
        summarizeRawTensorAndParity(
          oAll.final_p2b.data as Float32Array,
          oOrig.final_p2b.data as Float32Array
        ).maxAbsDiff
      ).toBe(0);
    } finally {
      await sOrig.release();
      await sM1Dis.release();
      await sM1All.release();
    }
  });

  it('fails closed atomically on nonzero multiplier, extra consumers, graph output exports, bad guards, and preserves shared zero__310', () => {
    for (const bad of [
      buildSyntheticInitToPropZeroSlantModel({ zeroInitVal: 1n }),
      buildSyntheticInitToPropZeroSlantModel({ extraConsumerOnMul: true }),
      buildSyntheticInitToPropZeroSlantModel({ exportZeroLikeOutput: true }),
      buildSyntheticInitToPropZeroSlantModel({ opset: 18 }),
      buildSyntheticInitToPropZeroSlantModel({ irVersion: 8 }),
      buildSyntheticInitToPropZeroSlantModel({ argMinKeepDims: 1 }),
    ]) {
      const res = normalizeHitnetInitToPropZeroSlantForWebGpu(bad, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: false,
      });
      expect(res.normalizedCount).toBe(0);
      expect(res.bytes).toBe(bad);
    }

    const valid = buildSyntheticInitToPropZeroSlantModel();
    expect(
      normalizeHitnetInitToPropZeroSlantForWebGpu(valid, {
        backendPreference: 'wasm',
        requireHitnetContract: false,
      }).normalizedCount
    ).toBe(0);
    expect(
      normalizeHitnetInitToPropZeroSlantForWebGpu(valid, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: true,
      }).normalizedCount
    ).toBe(0);

    // When zero__310 is shared by another live node, dead Cast/Mul/zeros_like/Reshape are dropped
    // while zero__310 is preserved safely.
    const sharedZero = buildSyntheticInitToPropZeroSlantModel({ sharedZeroInitWithLiveNode: true });
    const sharedRes = normalizeHitnetInitToPropZeroSlantForWebGpu(sharedZero, {
      variantId: 'hitnet-sf',
      backendPreference: 'webgpu',
      requireHitnetContract: false,
    });
    expect(sharedRes.normalizedCount).toBe(1);
    expect(sharedRes.removedDeadNodeNames).toHaveLength(4);
    expect(sharedRes.removedDeadInitializerNames).toEqual([]);
    expect(validateOnnxGraphNodeInputs(sharedRes.bytes)).toMatchObject({
      valid: true,
      missingInputs: [],
    });
  });

  it('applies M1-ConstZero2Ch strictly on real cached H1 HITNet (rejecting L1/G1), matching Raw WASM & p2b/p2a-concat1-out ch1/ch2 zero parity', async () => {
    const hitnetPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (!existsSync(hitnetPath)) {
      return;
    }
    const rawBytes = new Uint8Array(readFileSync(hitnetPath));
    const rawSha = createHash('sha256').update(rawBytes).digest('hex');
    expect(rawSha).toBe(HITNET_RAW_MODEL_SHA256);

    const m1Prep = prepareHitnetM1FullGraphBytes(rawBytes, {
      rawModelSha256: rawSha,
    });
    const h1Sha = createHash('sha256').update(m1Prep.baseH1Bytes).digest('hex');
    expect(h1Sha).toBe(HITNET_H1_FULL_GRAPH_SHA256);

    expect(m1Prep.m1Result.normalizedCount).toBe(2);
    expect(m1Prep.m1Result.normalizedNodeNames).toEqual([
      'level0/level_init/init_to_prop/concat_1',
      'level0_1/level_init/init_to_prop/concat_1',
    ]);
    expect(m1Prep.m1Result.removedDeadNodeNames).toHaveLength(8);
    expect(m1Prep.m1Result.removedDeadInitializerNames).toEqual(['zero__310']);
    expect(m1Prep.m1Result.addedInitializerName).toBe('__webgpu_m1_init_to_prop_zero_slant_2ch');
    expect(m1Prep.m1Result.addedInitializerDims).toEqual([1, 2, 60, 80]);
    expect(m1Prep.m1Result.verifiedH1Provenance).toBe(true);
    expect(m1Prep.m1Result.verifiedRawIdentity).toBe(true);
    expect(validateOnnxGraphNodeInputs(m1Prep.bytes)).toMatchObject({ valid: true, missingInputs: [] });
    expect(inferOnnxStaticTensorMeta(m1Prep.bytes).valid).toBe(true);

    // Reject L1-Reshape, L1-ReduceMin, and G1 provenance when applying M1
    const l1Reshape = prepareHitnetLevelInitCandidateFullGraphBytes(rawBytes, 'reshape-rank4');
    const l1Rm = prepareHitnetLevelInitCandidateFullGraphBytes(rawBytes, 'reduce-min');
    const g1OnH1 = normalizeHitnetArgMinKeepDimsForWebGpu(m1Prep.baseH1Bytes, {
      variantId: 'hitnet-sf',
      backendPreference: 'webgpu',
    });
    for (const forbiddenBase of [l1Reshape.bytes, l1Rm.bytes, g1OnH1.bytes, rawBytes]) {
      const rej = normalizeHitnetInitToPropZeroSlantForWebGpu(forbiddenBase, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: true,
      });
      expect(rej.normalizedCount).toBe(0);
      expect(rej.bytes).toBe(forbiddenBase);
    }

    // Wrong SHA-256 fails closed
    expect(
      normalizeHitnetInitToPropZeroSlantForWebGpu(m1Prep.baseH1Bytes, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: true,
        h1ModelSha256: '0000',
      }).normalizedCount
    ).toBe(0);

    // Verify WASM equivalence on full real model (Raw WASM disabled vs M1 WASM disabled/basic)
    const xData = new Float32Array(1 * 6 * 240 * 320);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = ((i * 1315423911) % 256) / 255;
    }
    const xTensor = new ort.Tensor('float32', xData, [1, 6, 240, 320]);

    const runModel = async (bytes: Uint8Array, opt: 'disabled' | 'basic') => {
      const s = await ort.InferenceSession.create(bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: opt,
        logSeverityLevel: 3,
      });
      try {
        const out = await s.run({ [s.inputNames[0]]: xTensor });
        return {
          ref: new Float32Array(out.reference_output_disparity.data as Float32Array),
          sec: new Float32Array(out.secondary_output_disparity.data as Float32Array),
        };
      } finally {
        await s.release();
      }
    };

    const rawDis = await runModel(rawBytes, 'disabled');
    for (const opt of ['disabled', 'basic'] as const) {
      const m1Out = await runModel(m1Prep.bytes, opt);
      for (const [bName, arr, refArr] of [
        ['primary', m1Out.ref, rawDis.ref],
        ['secondary', m1Out.sec, rawDis.sec],
      ] as const) {
        const st = summarizeRawTensorAndParity(arr, refArr);
        expect(st.finiteCount, `m1:${opt}:${bName}`).toBe(76800);
        expect(st.nonFiniteMismatchCount, `m1:${opt}:${bName}`).toBe(0);
        expect(st.maxAbsDiff, `m1:${opt}:${bName}`).toBe(0);
      }
    }

    // Reject removed dead p2a-concat1-zeros tap on baseMode='m1'
    expect(() => prepareHitnetFullGraphTapGroup(rawBytes, 'p2a-concat1-zeros', 'm1')).toThrow();

    // Verify p2a-concat1-out and p2b taps on baseMode='m1'
    const concat1Tap = prepareHitnetFullGraphTapGroup(rawBytes, 'p2a-concat1-out', 'm1');
    const p2bTap = prepareHitnetFullGraphTapGroup(rawBytes, 'p2b', 'm1');
    for (const [tapId, prepared, expectedDims] of [
      ['p2a-concat1-out', concat1Tap, [1, 16, 60, 80]],
      ['p2b', p2bTap, [1, 60, 80, 1, 16]],
    ] as const) {
      const sTap = await ort.InferenceSession.create(prepared.tappedH1.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'basic',
        logSeverityLevel: 3,
      });
      try {
        const res = await sTap.run({ [sTap.inputNames[0]]: xTensor });
        for (const tName of prepared.group.outputs) {
          expect(Array.from(res[tName].dims), `${tapId}:${tName}`).toEqual(expectedDims);
          const arr = new Float32Array(res[tName].data as Float32Array);
          const split =
            tapId === 'p2b'
              ? summarizeP2bChannelwiseParity(arr, arr)
              : summarizeInitToPropConcat1OutSplit(arr, arr);
          expect(split.ch1CandidateDist.zeroCount).toBe(4800);
          expect(split.ch1CandidateDist.nonZeroCount).toBe(0);
          expect(split.ch2CandidateDist.zeroCount).toBe(4800);
          expect(split.ch2CandidateDist.nonZeroCount).toBe(0);
          expect(split.ch1VsCh2CandidateMaxAbsDiff).toBe(0);
        }
      } finally {
        await sTap.release();
      }
    }
  }, 180000);

  it('summarizes full finite stats and exact BigInt64 index comparisons accurately', () => {
    const floatCand = new Float32Array([1.5, Number.NaN, 3.5, 5.0]);
    const floatRef = new Float32Array([1.5, 2.0, 3.0, 5.0]);
    const floatStats = summarizeRawTensorAndParity(floatCand, floatRef);
    expect(floatStats.finiteCount).toBe(3);
    expect(floatStats.nonFiniteCount).toBe(1);
    expect(floatStats.nonFiniteMismatchCount).toBe(1);
    expect(floatStats.mean).toBeCloseTo((1.5 + 3.5 + 5.0) / 3, 6);
    expect(floatStats.referenceMean).toBeCloseTo((1.5 + 2.0 + 3.0 + 5.0) / 4, 6);

    const idxMatch = summarizeExactInt64IndexComparison(
      new BigInt64Array([0n, 42n, 317n, 319n]),
      new BigInt64Array([0n, 42n, 317n, 319n]),
      { minInclusive: 0, maxInclusive: 319 }
    );
    expect(idxMatch.hasExactInt64Candidate).toBe(true);
    expect(idxMatch.hasExactInt64Reference).toBe(true);
    expect(idxMatch.exactMatchCount).toBe(4);
    expect(idxMatch.mismatchCount).toBe(0);
    expect(idxMatch.outOfRangeCount).toBe(0);
    expect(idxMatch.min).toBe('0');
    expect(idxMatch.max).toBe('319');
    expect(idxMatch.maxAbsDiff).toBe('0');

    const idxDiff = summarizeExactInt64IndexComparison(
      new BigInt64Array([0n, 15n, 320n, -1n]),
      new BigInt64Array([0n, 10n, 318n, 2n]),
      { minInclusive: 0, maxInclusive: 319 }
    );
    expect(idxDiff.exactMatchCount).toBe(1);
    expect(idxDiff.mismatchCount).toBe(3);
    expect(idxDiff.outOfRangeCount).toBe(2);
    expect(idxDiff.referenceOutOfRangeCount).toBe(0);
    expect(idxDiff.maxAbsDiff).toBe('5');
    expect(idxDiff.sampleMismatches).toEqual([
      { index: 1, candidate: '15', reference: '10' },
      { index: 2, candidate: '320', reference: '318' },
      { index: 3, candidate: '-1', reference: '2' },
    ]);
  });

  it('prepares and runs the three M1 disparity-source taps (p2-argmin, p2-cast, p2-slot0-reshape) with exact shapes and Raw-WASM final parity', async () => {
    const hitnetPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (!existsSync(hitnetPath)) {
      return;
    }
    const rawBytes = new Uint8Array(readFileSync(hitnetPath));
    const m1Prep = prepareHitnetM1FullGraphBytes(rawBytes, {
      rawModelSha256: HITNET_RAW_MODEL_SHA256,
      h1ModelSha256: HITNET_H1_FULL_GRAPH_SHA256,
    });
    const m1Sha = createHash('sha256').update(m1Prep.bytes).digest('hex');
    expect(m1Sha).toBe('97e3673bdea94980f71cc1fc13998066238636b0ae0fa7f5381aa3ffa711e758');

    const xData = new Float32Array(1 * 6 * 240 * 320);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = ((i * 2654435761) % 256) / 255;
    }
    const xTensor = new ort.Tensor('float32', xData, [1, 6, 240, 320]);

    const rawSess = await ort.InferenceSession.create(rawBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      logSeverityLevel: 3,
    });
    let rawRef: Float32Array;
    let rawSec: Float32Array;
    try {
      const out = await rawSess.run({ [rawSess.inputNames[0]]: xTensor });
      rawRef = new Float32Array(out.reference_output_disparity.data as Float32Array);
      rawSec = new Float32Array(out.secondary_output_disparity.data as Float32Array);
    } finally {
      await rawSess.release();
    }

    const expectedTaps = [
      {
        id: 'p2-argmin' as const,
        outputs: ['level0/level_init/ArgMin', 'level0_1/level_init/ArgMin'],
        dims: [1, 60, 80],
        elemType: 7,
        ortType: 'int64',
      },
      {
        id: 'p2-cast' as const,
        outputs: ['level0/level_init/Cast', 'level0_1/level_init/Cast'],
        dims: [1, 60, 80, 1],
        elemType: 1,
        ortType: 'float32',
      },
      {
        id: 'p2-slot0-reshape' as const,
        outputs: ['Transpose__908:0', 'Transpose__1006:0'],
        dims: [1, 1, 60, 80],
        elemType: 1,
        ortType: 'float32',
      },
    ];

    for (const spec of expectedTaps) {
      const prepared = prepareHitnetFullGraphTapGroup(rawBytes, spec.id, 'm1');
      expect(prepared.tappedH1.addedOutputs).toEqual(spec.outputs);
      expect(prepared.tappedH1.addedOutputs.length).toBeLessThanOrEqual(2);
      expect(prepared.tappedH1.baseOutputNames).toHaveLength(2);
      expect(prepared.tappedH1.allOutputNames).toHaveLength(4);
      for (const outName of spec.outputs) {
        expect(prepared.tappedH1.outputMeta[outName]).toEqual({
          elemType: spec.elemType,
          dims: spec.dims,
        });
      }

      const sess = await ort.InferenceSession.create(prepared.tappedH1.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'basic',
        logSeverityLevel: 3,
      });
      try {
        const res = await sess.run({ [sess.inputNames[0]]: xTensor });
        expect(Object.keys(res)).toHaveLength(4);
        const pStats = summarizeRawTensorAndParity(
          new Float32Array(res.reference_output_disparity.data as Float32Array),
          rawRef
        );
        const sStats = summarizeRawTensorAndParity(
          new Float32Array(res.secondary_output_disparity.data as Float32Array),
          rawSec
        );
        expect(pStats.maxAbsDiff).toBe(0);
        expect(sStats.maxAbsDiff).toBe(0);
        for (const outName of spec.outputs) {
          expect(Array.from(res[outName].dims)).toEqual(spec.dims);
          expect(res[outName].type).toBe(spec.ortType);
          if (spec.ortType === 'int64') {
            expect(res[outName].data).toBeInstanceOf(BigInt64Array);
            const idxSummary = summarizeExactInt64IndexComparison(
              res[outName].data as BigInt64Array,
              res[outName].data as BigInt64Array
            );
            expect(idxSummary.outOfRangeCount).toBe(0);
            expect(idxSummary.exactMatchCount).toBe(4800);
          } else {
            const st = summarizeRawTensorAndParity(
              new Float32Array(res[outName].data as Float32Array)
            );
            expect(st.finiteCount).toBe(4800);
            expect(st.nonFiniteCount).toBe(0);
          }
        }
      } finally {
        await sess.release();
      }
    }
  }, 180000);
});

describe('normalizeHitnetSlot0ReshapeToTransposeForWebGpu (N1-Slot0Transpose)', () => {
  const buildSyntheticM1Slot0Model = (opts?: {
    irVersion?: number;
    opset?: number;
    shapeDims?: number[];
    shapeVals?: bigint[];
    castOutDims?: number[];
    argMinKeepDims?: number;
    extraConsumerOnCast?: boolean;
    exportCastOut?: boolean;
    extraConsumerOnSlot0?: boolean;
    exportSlot0?: boolean;
    concatAxis?: number;
    addExtraInitializer?: boolean;
  }): Uint8Array => {
    const H = 3;
    const W = 4;
    const D = 5;
    const featC = 4;
    const zero2ChData = new Uint8Array(1 * 2 * H * W * 4);
    const zero2ChInit = encodeLenField(
      5,
      concatBytes([
        encodeVarintField(1, 1),
        encodeVarintField(1, 2),
        encodeVarintField(1, H),
        encodeVarintField(1, W),
        encodeVarintField(2, 1),
        encodeStrField(8, '__webgpu_m1_init_to_prop_zero_slant_2ch'),
        encodeLenField(9, zero2ChData),
      ])
    );

    const nodes: Uint8Array[] = [
      encodeNode('ArgMin', 'level0/level_init/ArgMin', ['cost_4d'], ['argmin_i64'], [
        encodeAttrInt('axis', 3),
        encodeAttrInt('keepdims', opts?.argMinKeepDims ?? 0),
        encodeAttrInt('select_last_index', 0),
      ]),
      encodeNode('Reshape', 'level0/level_init/ExpandDims_2', ['argmin_i64', 'shape_hw1'], [
        'expand2_i64',
      ]),
      encodeNode('Cast', 'level0/level_init/Cast', ['expand2_i64'], ['disp_f32'], [
        encodeAttrInt('to', 1),
      ]),
      encodeNode('Reshape', 'Transpose__908', ['disp_f32', 'new_shape__1191'], [
        'Transpose__908:0',
      ]),
      encodeNode(
        'Concat',
        'level0/level_init/init_to_prop/concat_1',
        ['Transpose__908:0', '__webgpu_m1_init_to_prop_zero_slant_2ch', 'feat_nchw'],
        ['concat1_out'],
        [encodeAttrInt('axis', opts?.concatAxis ?? 1)]
      ),
      encodeNode('Transpose', 'Transpose__913', ['concat1_out'], ['concat1_nhwc'], [
        encodeAttrInts('perm', [0, 2, 3, 1]),
      ]),
      encodeNode('Reshape', 'level0/level_init/init_to_prop/ExpandDims', [
        'concat1_nhwc',
        'shape_out5d',
      ], ['final_p2b']),
    ];

    if (opts?.extraConsumerOnCast) {
      nodes.push(encodeNode('Identity', 'extra_cast_consumer', ['disp_f32'], ['extra_cast_out']));
    }
    if (opts?.extraConsumerOnSlot0) {
      nodes.push(
        encodeNode('Identity', 'extra_slot0_consumer', ['Transpose__908:0'], ['extra_slot0_out'])
      );
    }

    const castDims = opts?.castOutDims ?? [1, H, W, 1];
    const shapeInitDims = opts?.shapeDims ?? [4];
    const shapeInitVals = opts?.shapeVals ?? [1n, 1n, BigInt(H), BigInt(W)];
    const out1C = shapeInitVals.reduce((acc, v) => acc * Number(v), 1);
    void out1C;

    const graphParts: Uint8Array[] = [
      ...nodes,
      encodeStrField(2, 'synthetic_n1_slot0_graph'),
      encodeLenField(5, encodeIntTensorProto('shape_hw1', [4], 7, [1, H, W, 1])),
      encodeLenField(
        5,
        encodeIntTensorProto('new_shape__1191', shapeInitDims, 7, shapeInitVals)
      ),
      encodeLenField(
        5,
        encodeIntTensorProto('shape_out5d', [5], 7, [1, H, W, 1, 3 + featC])
      ),
      zero2ChInit,
      encodeLenField(11, encodeValueInfoTensor('cost_4d', 1, [1, H, W, D])),
      encodeLenField(11, encodeValueInfoTensor('feat_nchw', 1, [1, featC, H, W])),
      encodeLenField(12, encodeValueInfoTensor('final_p2b', 1, [1, H, W, 1, 3 + featC])),
    ];
    if (opts?.addExtraInitializer) {
      graphParts.push(
        encodeLenField(5, encodeIntTensorProto('extra_unused_init', [1], 7, [42n]))
      );
    }
    if (opts?.extraConsumerOnCast) {
      graphParts.push(encodeLenField(12, encodeValueInfoTensor('extra_cast_out', 1, castDims)));
    }
    if (opts?.exportCastOut) {
      graphParts.push(encodeLenField(12, encodeValueInfoTensor('disp_f32', 1, castDims)));
    }
    if (opts?.extraConsumerOnSlot0) {
      graphParts.push(
        encodeLenField(12, encodeValueInfoTensor('extra_slot0_out', 1, [1, 1, H, W]))
      );
    }
    if (opts?.exportSlot0) {
      graphParts.push(
        encodeLenField(12, encodeValueInfoTensor('Transpose__908:0', 1, [1, 1, H, W]))
      );
    }

    return concatBytes([
      encodeVarintField(1, opts?.irVersion ?? 7),
      encodeLenField(7, concatBytes(graphParts)),
      encodeLenField(
        8,
        concatBytes([encodeStrField(1, ''), encodeVarintField(2, opts?.opset ?? 12)])
      ),
    ]);
  };

  it('replaces singleton NHWC->NCHW Reshape with Transpose perm=[0,3,1,2], preserves initializers and order, and matches WASM disabled/basic identically', async () => {
    const baseModel = buildSyntheticM1Slot0Model({ addExtraInitializer: true });
    const snapshot = baseModel.slice();
    const preMeta = inferOnnxStaticTensorMeta(baseModel);
    expect(preMeta.valid).toBe(true);

    const n1 = normalizeHitnetSlot0ReshapeToTransposeForWebGpu(baseModel, {
      variantId: 'hitnet-sf',
      backendPreference: 'webgpu',
      requireHitnetContract: false,
    });
    expect(Buffer.compare(baseModel, snapshot)).toBe(0);
    expect(n1.normalizedCount).toBe(1);
    expect(n1.normalizedNodeNames).toEqual(['Transpose__908']);

    expect(validateOnnxGraphNodeInputs(n1.bytes)).toMatchObject({
      valid: true,
      missingInputs: [],
    });
    const postMeta = inferOnnxStaticTensorMeta(n1.bytes);
    expect(postMeta.valid).toBe(true);
    expect(postMeta.nodeCount).toBe(preMeta.nodeCount);
    expect(postMeta.initializerCount).toBe(preMeta.initializerCount);
    expect(postMeta.tensorMeta['new_shape__1191']?.dims).toEqual([4]);
    expect(postMeta.tensorMeta['extra_unused_init']?.dims).toEqual([1]);
    expect(postMeta.tensorMeta['Transpose__908:0']?.dims).toEqual([1, 1, 3, 4]);
    expect(postMeta.tensorMeta['final_p2b']?.dims).toEqual([1, 3, 4, 1, 7]);

    const cost = new Float32Array(1 * 3 * 4 * 5);
    for (let i = 0; i < cost.length; i++) cost[i] = ((i * 13 + 7) % 19) * 0.5 - 4.0;
    const feat = new Float32Array(1 * 4 * 3 * 4);
    for (let i = 0; i < feat.length; i++) feat[i] = ((i * 5 + 1) % 11) * 0.25 - 1.0;

    const feeds = {
      cost_4d: new ort.Tensor('float32', cost, [1, 3, 4, 5]),
      feat_nchw: new ort.Tensor('float32', feat, [1, 4, 3, 4]),
    };

    const runSess = async (bytes: Uint8Array, opt: 'disabled' | 'basic') => {
      const s = await ort.InferenceSession.create(bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: opt,
        logSeverityLevel: 3,
      });
      try {
        const res = await s.run(feeds);
        return new Float32Array(res.final_p2b.data as Float32Array);
      } finally {
        await s.release();
      }
    };

    const refOut = await runSess(baseModel, 'disabled');
    for (const opt of ['disabled', 'basic'] as const) {
      const n1Out = await runSess(n1.bytes, opt);
      const parity = summarizeRawTensorAndParity(n1Out, refOut);
      expect(parity.finiteCount).toBe(1 * 3 * 4 * 7);
      expect(parity.nonFiniteMismatchCount).toBe(0);
      expect(parity.maxAbsDiff).toBe(0);
    }
  });

  it('fails closed atomically on shape mismatches, extra-use, graph output export, missing branches, and WASM without diagnostic flag', () => {
    const dynamicShape = buildSyntheticM1Slot0Model({ shapeVals: [1n, 1n, 3n, -1n] });
    const badShapeVals = buildSyntheticM1Slot0Model({ shapeVals: [1n, 2n, 3n, 2n] });
    const badShapeRank = buildSyntheticM1Slot0Model({ shapeDims: [2, 2], shapeVals: [1n, 1n, 3n, 4n] });
    const extraCastUser = buildSyntheticM1Slot0Model({ extraConsumerOnCast: true });
    const exportedCast = buildSyntheticM1Slot0Model({ exportCastOut: true });
    const extraSlot0User = buildSyntheticM1Slot0Model({ extraConsumerOnSlot0: true });
    const exportedSlot0 = buildSyntheticM1Slot0Model({ exportSlot0: true });
    const badKeepDims = buildSyntheticM1Slot0Model({ argMinKeepDims: 1 });
    const badConcatAxis = buildSyntheticM1Slot0Model({ concatAxis: 2 });
    const wrongOpset = buildSyntheticM1Slot0Model({ opset: 18 });
    const wrongIr = buildSyntheticM1Slot0Model({ irVersion: 8 });

    for (const bad of [
      dynamicShape,
      badShapeVals,
      badShapeRank,
      extraCastUser,
      exportedCast,
      extraSlot0User,
      exportedSlot0,
      badKeepDims,
      badConcatAxis,
      wrongOpset,
      wrongIr,
    ]) {
      const res = normalizeHitnetSlot0ReshapeToTransposeForWebGpu(bad, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: false,
      });
      expect(res.normalizedCount).toBe(0);
      expect(res.bytes).toBe(bad);
    }

    const valid = buildSyntheticM1Slot0Model();
    // Gated OFF on WASM unless allowDiagnosticTransformedWasm: true
    expect(
      normalizeHitnetSlot0ReshapeToTransposeForWebGpu(valid, {
        variantId: 'hitnet-sf',
        backendPreference: 'wasm',
        requireHitnetContract: false,
      }).normalizedCount
    ).toBe(0);
    expect(
      normalizeHitnetSlot0ReshapeToTransposeForWebGpu(valid, {
        variantId: 'hitnet-sf',
        backendPreference: 'wasm',
        allowDiagnosticTransformedWasm: true,
        requireHitnetContract: false,
      }).normalizedCount
    ).toBe(1);

    // Missing second HITNet branch fails closed under requireHitnetContract: true
    expect(
      normalizeHitnetSlot0ReshapeToTransposeForWebGpu(valid, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: true,
      }).normalizedCount
    ).toBe(0);
  });

  it('applies N1-Slot0Transpose strictly on exact M1 real HITNet (rejecting Raw/H1/L1/G1), matches Raw WASM disabled/basic, and runs pre-Concat N1 taps', async () => {
    const hitnetPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (!existsSync(hitnetPath)) {
      return;
    }
    const rawBytes = new Uint8Array(readFileSync(hitnetPath));
    const rawSha = createHash('sha256').update(rawBytes).digest('hex');
    expect(rawSha).toBe(HITNET_RAW_MODEL_SHA256);

    const n1Prep = prepareHitnetN1FullGraphBytes(rawBytes, {
      rawModelSha256: rawSha,
      m1ModelSha256: HITNET_M1_FULL_GRAPH_SHA256,
    });
    const m1Sha = createHash('sha256').update(n1Prep.baseM1Bytes).digest('hex');
    expect(m1Sha).toBe(HITNET_M1_FULL_GRAPH_SHA256);

    expect(n1Prep.n1Result.normalizedCount).toBe(2);
    expect(n1Prep.n1Result.normalizedNodeNames).toEqual([
      'Transpose__908',
      'Transpose__1006',
    ]);
    expect(n1Prep.n1Result.verifiedM1Provenance).toBe(true);
    expect(n1Prep.n1Result.verifiedRawIdentity).toBe(true);
    expect(n1Prep.n1Result.skippedInvalidOrUnprovenCount).toBe(0);

    const m1Infer = inferOnnxStaticTensorMeta(n1Prep.baseM1Bytes);
    const n1Infer = inferOnnxStaticTensorMeta(n1Prep.bytes);
    expect(n1Infer.valid).toBe(true);
    expect(n1Infer.nodeCount).toBe(m1Infer.nodeCount);
    expect(n1Infer.initializerCount).toBe(m1Infer.initializerCount);
    expect(n1Infer.tensorMeta['new_shape__1191']?.dims).toEqual([4]);
    expect(n1Infer.tensorMeta['Transpose__908:0']).toEqual({
      elemType: 1,
      dims: [1, 1, 60, 80],
    });
    expect(n1Infer.tensorMeta['Transpose__1006:0']).toEqual({
      elemType: 1,
      dims: [1, 1, 60, 80],
    });
    expect(
      validateOnnxGraphNodeInputs(n1Prep.bytes)
    ).toMatchObject({ valid: true, missingInputs: [] });

    // Reject Raw, H1, L1-Reshape, L1-ReduceMin, and G1 provenance when applying N1
    const l1Reshape = prepareHitnetLevelInitCandidateFullGraphBytes(rawBytes, 'reshape-rank4');
    const l1Rm = prepareHitnetLevelInitCandidateFullGraphBytes(rawBytes, 'reduce-min');
    const g1OnM1 = normalizeHitnetArgMinKeepDimsForWebGpu(n1Prep.baseM1Bytes, {
      variantId: 'hitnet-sf',
      backendPreference: 'webgpu',
    });
    for (const forbiddenBase of [
      rawBytes,
      n1Prep.baseH1Bytes,
      l1Reshape.bytes,
      l1Rm.bytes,
      g1OnM1.bytes,
    ]) {
      const rej = normalizeHitnetSlot0ReshapeToTransposeForWebGpu(forbiddenBase, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: true,
      });
      expect(rej.normalizedCount).toBe(0);
      expect(rej.bytes).toBe(forbiddenBase);
    }

    // Wrong SHA-256 fails closed
    expect(
      normalizeHitnetSlot0ReshapeToTransposeForWebGpu(n1Prep.baseM1Bytes, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: true,
        m1ModelSha256: '0000',
      }).normalizedCount
    ).toBe(0);
    expect(
      normalizeHitnetSlot0ReshapeToTransposeForWebGpu(n1Prep.baseM1Bytes, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
        requireHitnetContract: true,
        rawModelSha256: '0000',
      }).normalizedCount
    ).toBe(0);

    // Same exact N1 bytes on WASM disabled and basic must match Raw WASM disabled on BOTH final outputs
    const xData = new Float32Array(1 * 6 * 240 * 320);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = ((i * 1315423911) % 256) / 255;
    }
    const xTensor = new ort.Tensor('float32', xData, [1, 6, 240, 320]);

    const runModel = async (bytes: Uint8Array, opt: 'disabled' | 'basic') => {
      const s = await ort.InferenceSession.create(bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: opt,
        logSeverityLevel: 3,
      });
      try {
        const out = await s.run({ [s.inputNames[0]]: xTensor });
        return {
          ref: new Float32Array(out.reference_output_disparity.data as Float32Array),
          sec: new Float32Array(out.secondary_output_disparity.data as Float32Array),
        };
      } finally {
        await s.release();
      }
    };

    const rawDis = await runModel(rawBytes, 'disabled');
    for (const opt of ['disabled', 'basic'] as const) {
      const n1Out = await runModel(n1Prep.bytes, opt);
      for (const [bName, arr, refArr] of [
        ['primary', n1Out.ref, rawDis.ref],
        ['secondary', n1Out.sec, rawDis.sec],
      ] as const) {
        const st = summarizeRawTensorAndParity(arr, refArr);
        expect(st.finiteCount, `n1:${opt}:${bName}`).toBe(76800);
        expect(st.nonFiniteMismatchCount, `n1:${opt}:${bName}`).toBe(0);
        expect(st.maxAbsDiff, `n1:${opt}:${bName}`).toBe(0);
      }
    }

    // Verify pre-Concat N1 taps (p2-slot0-reshape, p2a-concat1-out) and rejection of dead p2a-concat1-zeros
    expect(() => prepareHitnetFullGraphTapGroup(rawBytes, 'p2a-concat1-zeros', 'n1')).toThrow();
    for (const [tapId, expectedDims] of [
      ['p2-slot0-reshape', [1, 1, 60, 80]],
      ['p2a-concat1-out', [1, 16, 60, 80]],
    ] as const) {
      const tappedN1 = prepareHitnetFullGraphTapGroup(rawBytes, tapId, 'n1');
      const sTap = await ort.InferenceSession.create(tappedN1.tappedH1.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'basic',
        logSeverityLevel: 3,
      });
      try {
        const res = await sTap.run({ [sTap.inputNames[0]]: xTensor });
        expect(
          summarizeRawTensorAndParity(
            new Float32Array(res.reference_output_disparity.data as Float32Array),
            rawDis.ref
          ).maxAbsDiff
        ).toBe(0);
        expect(
          summarizeRawTensorAndParity(
            new Float32Array(res.secondary_output_disparity.data as Float32Array),
            rawDis.sec
          ).maxAbsDiff
        ).toBe(0);
        for (const outName of tappedN1.group.outputs) {
          expect(Array.from(res[outName].dims)).toEqual(expectedDims);
          const arr = new Float32Array(res[outName].data as Float32Array);
          const st = summarizeRawTensorAndParity(arr);
          expect(st.nonFiniteCount).toBe(0);
          if (tapId === 'p2a-concat1-out') {
            const split = summarizeInitToPropConcat1OutSplit(arr, arr);
            expect(split.ch1CandidateDist.zeroCount).toBe(4800);
            expect(split.ch2CandidateDist.zeroCount).toBe(4800);
          }
        }
      } finally {
        await sTap.release();
      }
    }
  }, 180000);
});



