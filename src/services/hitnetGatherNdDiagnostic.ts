import {
  appendOnnxGraphOutputs,
  decomposeOnnx5dConvNodesForWebGpu,
  HITNET_H1_FULL_GRAPH_SHA256,
  HITNET_M1_FULL_GRAPH_SHA256,
  HITNET_RAW_MODEL_SHA256,
  inferOnnxStaticTensorMeta,
  normalizeHitnetArgMinKeepDimsForWebGpu,
  normalizeHitnetGatherNdRankForWebGpu,
  normalizeHitnetInitToPropZeroSlantForWebGpu,
  normalizeHitnetLevelInitGatherNdRankForWebGpu,
  normalizeHitnetLevelInitReduceMinForWebGpu,
  normalizeHitnetPropGatherNdRankForWebGpu,
  normalizeHitnetSlot0ReshapeToTransposeForWebGpu,
  normalizeNonAscendingSliceAxesForWebGpu,
  splitLargeOnnxConcatNodes,
  stripOnnxGraphValueInfo,
  type AppendOnnxGraphOutputsOptions,
  type AppendOnnxGraphOutputsResult,
  type HitnetLevelInitGatherMode,
  type OnnxInitToPropZeroSlantNormalizeResult,
  type OnnxLevelInitGatherNormalizeResult,
  type OnnxSlot0ReshapeToTransposeResult,
  type OnnxStaticInferenceResult,
} from './stereoDepthWorker';

export { appendOnnxGraphOutputs };
export type { AppendOnnxGraphOutputsOptions, AppendOnnxGraphOutputsResult };

export interface RawTensorParityStats {
  finiteCount: number;
  nonFiniteCount: number;
  length: number;
  min: number;
  max: number;
  mean: number;
  referenceMean: number | null;
  maxAbsDiff: number;
  meanAbsDiff: number;
  rmsDiff: number;
  p99AbsDiff: number;
  nonFiniteMismatchCount: number;
}

export function summarizeRawTensorAndParity(
  candidate: Float32Array,
  reference?: Float32Array | null
): RawTensorParityStats {
  const len = candidate.length;
  let finiteCount = 0;
  let min = Infinity;
  let max = -Infinity;
  let sumFinite = 0;
  let refFiniteCount = 0;
  let sumRefFinite = 0;
  let maxAbsDiff = 0;
  let sumAbsDiff = 0;
  let sqDiff = 0;
  let nonFiniteMismatchCount = 0;
  const diffs = reference ? new Float32Array(len) : null;

  for (let i = 0; i < len; i++) {
    const v = candidate[i];
    const vFinite = Number.isFinite(v);
    if (vFinite) {
      finiteCount++;
      sumFinite += v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (reference && diffs) {
      const r = reference[i];
      const rFinite = Number.isFinite(r);
      if (rFinite) {
        refFiniteCount++;
        sumRefFinite += r;
      }
      if (vFinite !== rFinite || !vFinite || !rFinite) {
        if (!(Number.isNaN(v) && Number.isNaN(r)) && v !== r) {
          nonFiniteMismatchCount++;
        }
        diffs[i] = 0;
      } else {
        const d = Math.abs(v - r);
        diffs[i] = d;
        if (d > maxAbsDiff) maxAbsDiff = d;
        sumAbsDiff += d;
        sqDiff += d * d;
      }
    }
  }

  let p99AbsDiff = 0;
  if (diffs && len > 0) {
    diffs.sort();
    const p99Idx = Math.min(len - 1, Math.floor(0.99 * (len - 1)));
    p99AbsDiff = diffs[p99Idx];
  }

  return {
    finiteCount,
    nonFiniteCount: len - finiteCount,
    length: len,
    min,
    max,
    mean: finiteCount > 0 ? sumFinite / finiteCount : 0,
    referenceMean: reference ? (refFiniteCount > 0 ? sumRefFinite / refFiniteCount : 0) : null,
    maxAbsDiff,
    meanAbsDiff: len > 0 ? sumAbsDiff / len : 0,
    rmsDiff: len > 0 ? Math.sqrt(sqDiff / len) : 0,
    p99AbsDiff,
    nonFiniteMismatchCount,
  };
}

export interface ExactInt64IndexComparisonSummary {
  length: number;
  hasExactInt64Candidate: boolean;
  hasExactInt64Reference: boolean;
  min: string;
  max: string;
  mean: number;
  referenceMin: string | null;
  referenceMax: string | null;
  referenceMean: number | null;
  outOfRangeCount: number;
  referenceOutOfRangeCount: number | null;
  exactMatchCount: number;
  mismatchCount: number;
  maxAbsDiff: string;
  meanAbsDiff: number;
  sampleMismatches: Array<{ index: number; candidate: string; reference: string }>;
}

export function summarizeExactInt64IndexComparison(
  candidate: BigInt64Array | Float32Array,
  reference?: BigInt64Array | Float32Array | null,
  validRange: { minInclusive: number; maxInclusive: number } = { minInclusive: 0, maxInclusive: 319 }
): ExactInt64IndexComparisonSummary {
  const len = candidate.length;
  if (reference && reference.length !== len) {
    throw new Error(
      `Index tensor length mismatch: candidate=${len}, reference=${reference.length}`
    );
  }
  const hasExactInt64Candidate = candidate instanceof BigInt64Array;
  const hasExactInt64Reference = reference instanceof BigInt64Array;
  const rangeMin = BigInt(validRange.minInclusive);
  const rangeMax = BigInt(validRange.maxInclusive);

  let min: bigint | null = null;
  let max: bigint | null = null;
  let sum = 0;
  let outOfRangeCount = 0;

  let refMin: bigint | null = null;
  let refMax: bigint | null = null;
  let refSum = 0;
  let refOutOfRangeCount = 0;

  let exactMatchCount = 0;
  let mismatchCount = 0;
  let maxAbsDiff = 0n;
  let sumAbsDiff = 0;
  const sampleMismatches: Array<{ index: number; candidate: string; reference: string }> = [];

  for (let i = 0; i < len; i++) {
    const cvRaw = candidate[i];
    const cv =
      typeof cvRaw === 'bigint'
        ? cvRaw
        : Number.isFinite(cvRaw)
          ? BigInt(Math.trunc(cvRaw))
          : 0n;
    const cvValidInt = typeof cvRaw === 'bigint' || (Number.isFinite(cvRaw) && Number.isInteger(cvRaw));
    if (!cvValidInt || cv < rangeMin || cv > rangeMax) {
      outOfRangeCount++;
    }
    if (cvValidInt) {
      if (min === null || cv < min) min = cv;
      if (max === null || cv > max) max = cv;
      sum += Number(cv);
    }

    if (reference) {
      const rvRaw = reference[i];
      const rv =
        typeof rvRaw === 'bigint'
          ? rvRaw
          : Number.isFinite(rvRaw)
            ? BigInt(Math.trunc(rvRaw))
            : 0n;
      const rvValidInt = typeof rvRaw === 'bigint' || (Number.isFinite(rvRaw) && Number.isInteger(rvRaw));
      if (!rvValidInt || rv < rangeMin || rv > rangeMax) {
        refOutOfRangeCount++;
      }
      if (rvValidInt) {
        if (refMin === null || rv < refMin) refMin = rv;
        if (refMax === null || rv > refMax) refMax = rv;
        refSum += Number(rv);
      }
      if (cvValidInt && rvValidInt && cv === rv) {
        exactMatchCount++;
      } else {
        mismatchCount++;
        const diff = cvValidInt && rvValidInt ? (cv >= rv ? cv - rv : rv - cv) : 0n;
        if (diff > maxAbsDiff) maxAbsDiff = diff;
        sumAbsDiff += Number(diff);
        if (sampleMismatches.length < 5) {
          sampleMismatches.push({
            index: i,
            candidate: cvValidInt ? cv.toString() : String(cvRaw),
            reference: rvValidInt ? rv.toString() : String(rvRaw),
          });
        }
      }
    }
  }

  return {
    length: len,
    hasExactInt64Candidate,
    hasExactInt64Reference,
    min: min !== null ? min.toString() : 'n/a',
    max: max !== null ? max.toString() : 'n/a',
    mean: len > 0 ? sum / len : 0,
    referenceMin: reference ? (refMin !== null ? refMin.toString() : 'n/a') : null,
    referenceMax: reference ? (refMax !== null ? refMax.toString() : 'n/a') : null,
    referenceMean: reference && len > 0 ? refSum / len : null,
    outOfRangeCount,
    referenceOutOfRangeCount: reference ? refOutOfRangeCount : null,
    exactMatchCount,
    mismatchCount,
    maxAbsDiff: maxAbsDiff.toString(),
    meanAbsDiff: reference && len > 0 ? sumAbsDiff / len : 0,
    sampleMismatches,
  };
}

function encodeVarint(val: number): Uint8Array {
  const out: number[] = [];
  let v = val >>> 0;
  while (v >= 0x80) {
    out.push((v & 0x7f) | 0x80);
    v >>>= 7;
  }
  out.push(v);
  return new Uint8Array(out);
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

function concat(parts: readonly Uint8Array[]): Uint8Array {
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

function lenField(fn: number, payload: Uint8Array): Uint8Array {
  return concat([encodeVarint((fn << 3) | 2), encodeVarint(payload.byteLength), payload]);
}

function strField(fn: number, str: string): Uint8Array {
  return lenField(fn, new TextEncoder().encode(str));
}

function varintField(fn: number, val: number): Uint8Array {
  return concat([encodeVarint((fn << 3) | 0), encodeVarint(val)]);
}

function attrInt(name: string, val: number): Uint8Array {
  return lenField(
    5,
    concat([strField(1, name), concat([encodeVarint((3 << 3) | 0), encodeSignedVarint64(val)]), varintField(20, 2)])
  );
}

function attrInts(name: string, vals: readonly number[]): Uint8Array {
  return lenField(
    5,
    concat([
      strField(1, name),
      ...vals.map((v) => concat([encodeVarint((8 << 3) | 0), encodeSignedVarint64(v)])),
      varintField(20, 7),
    ])
  );
}

function makeInt64Initializer(name: string, dims: readonly number[], vals: readonly number[]): Uint8Array {
  const raw = new Uint8Array(vals.length * 8);
  const dv = new DataView(raw.buffer);
  for (let i = 0; i < vals.length; i++) {
    dv.setBigInt64(i * 8, BigInt(vals[i]), true);
  }
  return lenField(
    5,
    concat([
      ...dims.map((d) => varintField(1, d)),
      varintField(2, 7),
      strField(8, name),
      lenField(9, raw),
    ])
  );
}

function makeNode(
  opType: string,
  name: string,
  inputs: readonly string[],
  outputs: readonly string[],
  attrs: readonly Uint8Array[] = []
): Uint8Array {
  return lenField(
    1,
    concat([
      ...inputs.map((i) => strField(1, i)),
      ...outputs.map((o) => strField(2, o)),
      strField(3, name),
      strField(4, opType),
      ...attrs,
    ])
  );
}

function makeValueInfo(name: string, elemType: number, dims: readonly number[], fieldNum = 11): Uint8Array {
  const shape = lenField(
    2,
    concat(dims.map((d) => lenField(1, varintField(1, d))))
  );
  const tensorType = lenField(1, concat([varintField(1, elemType), shape]));
  return lenField(fieldNum, concat([strField(1, name), lenField(2, tensorType)]));
}

function makeOutputValueInfo(name: string, elemType: number, dims: readonly number[]): Uint8Array {
  return makeValueInfo(name, elemType, dims, 12);
}

function buildGraphModel(
  graphName: string,
  nodes: readonly Uint8Array[],
  inputs: readonly Uint8Array[],
  outputs: readonly Uint8Array[],
  initializers: readonly Uint8Array[] = [],
  valueInfos: readonly Uint8Array[] = [],
  opsetVersion = 12
): Uint8Array {
  const graph = lenField(
    7,
    concat([...nodes, strField(2, graphName), ...initializers, ...inputs, ...outputs, ...valueInfos])
  );
  const opset = lenField(8, concat([strField(1, ''), varintField(2, opsetVersion)]));
  return concat([varintField(1, 7), opset, graph]);
}

export interface SyntheticGatherNdSuite {
  dataDims: [number, number, number, number];
  idx4Dims: [number, number, number, number];
  idx5Dims: [number, number, number, number, number];
  dataF32: Float32Array;
  idx4I64: BigInt64Array;
  idx5I64: BigInt64Array;
  models: {
    rank4ThenRank5: Uint8Array;
    rank5ThenRank4: Uint8Array;
    rank5Only: Uint8Array;
    rank4ThenPatchedRank5: Uint8Array;
  };
  patchedNodeNames: string[];
}

/**
 * Constructs synthetic ONNX opset-12 controls matching the exact HITNet GatherND inputs:
 * - fixed float32 rank-4 data `[1, 2, 8, 4]`
 * - int64 indices (`batch_dims = 2`, trailing tuple dim `1`, including negative indices `-1..-8`)
 * - rank-4 output branch `[1, 2, 6, 4]` and rank-5 output branch `[1, 2, 4, 3, 4]`
 */
export function buildSyntheticGatherNdControlModels(): SyntheticGatherNdSuite {
  const dataDims: [number, number, number, number] = [1, 2, 8, 4];
  const idx4Dims: [number, number, number, number] = [1, 2, 6, 1];
  const idx5Dims: [number, number, number, number, number] = [1, 2, 4, 3, 1];
  const out4Dims = [1, 2, 6, 4];
  const out5Dims = [1, 2, 4, 3, 4];

  const dataF32 = new Float32Array(1 * 2 * 8 * 4);
  for (let i = 0; i < dataF32.length; i++) {
    dataF32[i] = (i + 1) * 0.25;
  }

  // Mix positive and negative indices across batch 0 and batch 1
  const idx4I64 = new BigInt64Array([
    0n, -1n, 3n, -8n, 7n, -2n,
    1n, -3n, 6n, -4n, 2n, -5n,
  ]);
  const idx5I64 = new BigInt64Array([
    0n, -1n, 1n, -8n, 2n, -2n, 3n, -7n, 4n, -6n, 5n, -5n,
    7n, -1n, 6n, -2n, 5n, -3n, 4n, -4n, 3n, -5n, 2n, -6n,
  ]);

  const gnd4 = makeNode('GatherND', 'gnd_rank4', ['data', 'idx4'], ['out4'], [attrInt('batch_dims', 2)]);
  const gnd5 = makeNode('GatherND', 'gnd_rank5', ['data', 'idx5'], ['out5'], [attrInt('batch_dims', 2)]);

  const viData = makeValueInfo('data', 1, dataDims);
  const viIdx4 = makeValueInfo('idx4', 7, idx4Dims);
  const viIdx5 = makeValueInfo('idx5', 7, idx5Dims);
  const voOut4 = makeOutputValueInfo('out4', 1, out4Dims);
  const voOut5 = makeOutputValueInfo('out5', 1, out5Dims);

  const rank4ThenRank5 = buildGraphModel(
    'synthetic_rank4_then_rank5',
    [gnd4, gnd5],
    [viData, viIdx4, viIdx5],
    [voOut4, voOut5]
  );

  const rank5ThenRank4 = buildGraphModel(
    'synthetic_rank5_then_rank4',
    [gnd5, gnd4],
    [viData, viIdx5, viIdx4],
    [voOut5, voOut4]
  );

  const rank5Only = buildGraphModel(
    'synthetic_rank5_only',
    [gnd5],
    [viData, viIdx5],
    [voOut5]
  );

  const patched = normalizeHitnetGatherNdRankForWebGpu(rank4ThenRank5, {
    requireHitnetContract: false,
  });

  return {
    dataDims,
    idx4Dims,
    idx5Dims,
    dataF32,
    idx4I64,
    idx5I64,
    models: {
      rank4ThenRank5,
      rank5ThenRank4,
      rank5Only,
      rank4ThenPatchedRank5: patched.bytes,
    },
    patchedNodeNames: patched.normalizedNodeNames,
  };
}

export interface SyntheticArgMinSuite {
  cost8Dims: [number, number, number, number];
  cost4Dims: [number, number, number, number];
  table4Dims: [number, number, number, number, number];
  cost8F32: Float32Array;
  cost4F32: Float32Array;
  table4F32: Float32Array;
  expectedCastIndices: Float32Array;
  expectedGatherNdValues: Float32Array;
  models: {
    singleSizeCast: Uint8Array;
    singleSizeGatherNd: Uint8Array;
    patchedSeparatedTwoSize: Uint8Array;
  };
  patchedNodeNames: string[];
  untouchedNodeNames: string[];
  updatedValueInfoCount: number;
  ortCacheKeyBefore: string | null;
  ortCacheKeyAfter: string | null;
}

/**
 * Constructs ONLY valid, shape-isolated single-size ArgMin operations and the patched,
 * separated-shape two-size graph (keepdims=0 for axis=8 vs keepdims=1 for axis=4).
 * Intentionally does NOT expose or run an unpatched colliding multi-size GPU model.
 * Exercises ties (`select_last_index=0`) and the last spatial location `(1, 2)`.
 */
export function buildSyntheticArgMinControlModels(): SyntheticArgMinSuite {
  const cost8Dims: [number, number, number, number] = [1, 2, 3, 8];
  const cost4Dims: [number, number, number, number] = [1, 2, 3, 4];
  const table4Dims: [number, number, number, number, number] = [1, 2, 3, 4, 2];

  // 6 spatial locations x 8 candidates; includes ties and distinct minimum at last spatial location (1,2)
  const cost8F32 = new Float32Array([
    // (0, 0) -> min idx 2 (val 0.1)
    0.8, 0.4, 0.1, 0.5, 0.9, 0.3, 0.7, 0.6,
    // (0, 1) -> tie between idx 1 and idx 4 (val 0.25) => select_last_index=0 -> 1
    0.9, 0.25, 0.7, 0.8, 0.25, 0.6, 0.4, 0.5,
    // (0, 2) -> min idx 0 (tie with idx 7) -> 0
    0.05, 0.4, 0.3, 0.2, 0.9, 0.8, 0.7, 0.05,
    // (1, 0) -> min idx 5 -> 5
    1.2, 1.1, 0.9, 0.8, 0.7, -0.5, 0.4, 0.3,
    // (1, 1) -> tie between idx 3 and idx 6 -> 3
    2.0, 1.5, 1.2, 0.125, 0.9, 0.4, 0.125, 0.8,
    // (1, 2) LAST spatial location -> min idx 7 (tie? no, unique 7) -> 7
    3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.25, -1.0,
  ]);
  const expectedCastIndices = new Float32Array([2, 1, 0, 5, 3, 7]);

  // 6 spatial locations x 4 candidates; includes ties and last spatial location (1,2)
  const cost4F32 = new Float32Array([
    // (0, 0) -> min idx 2
    0.9, 0.4, 0.1, 0.8,
    // (0, 1) -> tie idx 1 & idx 3 => 1
    0.7, 0.2, 0.6, 0.2,
    // (0, 2) -> min idx 0
    -0.3, 0.1, 0.2, 0.3,
    // (1, 0) -> min idx 3
    1.0, 0.9, 0.8, 0.25,
    // (1, 1) -> tie idx 0 & idx 2 => 0
    0.5, 0.9, 0.5, 0.7,
    // (1, 2) LAST spatial location -> tie idx 2 & idx 3 => 2
    2.5, 1.5, -0.75, -0.75,
  ]);
  const expectedBestIdx4 = [2, 1, 0, 3, 0, 2];

  const table4F32 = new Float32Array(1 * 2 * 3 * 4 * 2);
  for (let i = 0; i < table4F32.length; i++) {
    table4F32[i] = 10 + i * 0.5;
  }
  const expectedGatherNdValues = new Float32Array(6 * 2);
  for (let s = 0; s < 6; s++) {
    const c = expectedBestIdx4[s];
    const base = (s * 4 + c) * 2;
    expectedGatherNdValues[s * 2] = table4F32[base];
    expectedGatherNdValues[s * 2 + 1] = table4F32[base + 1];
  }

  const initShape = makeInt64Initializer('norm_shape', [4], [1, 2, 3, 1]);

  // S5_Cast: Valid single-size ArgMin(axis=3, keepdims=0, select_last_index=0) -> Reshape([1,2,3,1]) -> Cast(float)
  const nInitArgMin = makeNode('ArgMin', 'level_init_argmin', ['cost8'], ['init_idx_raw'], [
    attrInt('axis', 3),
    attrInt('keepdims', 0),
    attrInt('select_last_index', 0),
  ]);
  const nInitReshape = makeNode('Reshape', 'level_init_reshape', ['init_idx_raw', 'norm_shape'], ['init_idx_4d']);
  const nInitCast = makeNode('Cast', 'level_init_cast', ['init_idx_4d'], ['out_cast'], [attrInt('to', 1)]);
  const singleSizeCast = buildGraphModel(
    'synthetic_single_size_argmin_cast',
    [nInitArgMin, nInitReshape, nInitCast],
    [makeValueInfo('cost8', 1, cost8Dims)],
    [makeOutputValueInfo('out_cast', 1, [1, 2, 3, 1])],
    [initShape]
  );

  // S5_GatherND: Valid single-size ArgMin(axis=3, keepdims=0, select_last_index=0) -> Reshape([1,2,3,1]) -> Unsqueeze([-1]) -> GatherND(batch_dims=3)
  const nPropArgMin = makeNode('ArgMin', 'prop0_argmin', ['cost4'], ['prop_idx_raw'], [
    attrInt('axis', 3),
    attrInt('keepdims', 0),
    attrInt('select_last_index', 0),
  ]);
  const nPropReshape = makeNode('Reshape', 'prop0_reshape', ['prop_idx_raw', 'norm_shape'], ['prop_idx_4d']);
  const nPropUnsqueeze = makeNode('Unsqueeze', 'prop0_unsqueeze', ['prop_idx_4d'], ['prop_idx_5d'], [
    attrInts('axes', [-1]),
  ]);
  const nPropGnd = makeNode('GatherND', 'prop0_gnd', ['table4', 'prop_idx_5d'], ['out_gathernd'], [
    attrInt('batch_dims', 3),
  ]);
  const singleSizeGatherNd = buildGraphModel(
    'synthetic_single_size_argmin_gathernd',
    [nPropArgMin, nPropReshape, nPropUnsqueeze, nPropGnd],
    [makeValueInfo('cost4', 1, cost4Dims), makeValueInfo('table4', 1, table4Dims)],
    [makeOutputValueInfo('out_gathernd', 1, [1, 2, 3, 1, 2])],
    [initShape]
  );

  // S6: Combined two-size graph constructed and immediately patched via normalizeHitnetArgMinKeepDimsForWebGpu
  // (includes internal value_info on prop_idx_raw to verify internal value_info updates).
  const unpatchedTwoSizeInternal = buildGraphModel(
    'synthetic_two_size_argmin_separated',
    [nInitArgMin, nInitReshape, nInitCast, nPropArgMin, nPropReshape, nPropUnsqueeze, nPropGnd],
    [
      makeValueInfo('cost8', 1, cost8Dims),
      makeValueInfo('cost4', 1, cost4Dims),
      makeValueInfo('table4', 1, table4Dims),
    ],
    [
      makeOutputValueInfo('out_cast', 1, [1, 2, 3, 1]),
      makeOutputValueInfo('out_gathernd', 1, [1, 2, 3, 1, 2]),
    ],
    [initShape],
    [makeValueInfo('prop_idx_raw', 7, [1, 2, 3], 13)]
  );

  const patched = normalizeHitnetArgMinKeepDimsForWebGpu(unpatchedTwoSizeInternal, {
    requireHitnetContract: false,
  });
  if (patched.normalizedCount !== 1 || patched.updatedValueInfoCount !== 1) {
    throw new Error('Failed to construct valid patched separated two-size ArgMin control');
  }

  return {
    cost8Dims,
    cost4Dims,
    table4Dims,
    cost8F32,
    cost4F32,
    table4F32,
    expectedCastIndices,
    expectedGatherNdValues,
    models: {
      singleSizeCast,
      singleSizeGatherNd,
      patchedSeparatedTwoSize: patched.bytes,
    },
    patchedNodeNames: patched.normalizedNodeNames,
    untouchedNodeNames: patched.untouchedNodeNames,
    updatedValueInfoCount: patched.updatedValueInfoCount,
    ortCacheKeyBefore: patched.ortCacheKeyBefore,
    ortCacheKeyAfter: patched.ortCacheKeyAfter,
  };
}

export interface SyntheticPropGatherNdSuite {
  dataDims5: [number, number, number, number, number];
  dataDims4: [number, number, number, number];
  idxDims5: [number, number, number, number, number];
  idxDims4: [number, number, number, number];
  cost4Dims: [number, number, number, number];
  cost8Dims: [number, number, number, number];
  dataF32: Float32Array;
  idxI64: BigInt64Array;
  cost4F32: Float32Array;
  cost8F32: Float32Array;
  expectedConstOrInputOut: Float32Array;
  expectedDynamicArgMinOut: Float32Array;
  expectedCastOut: Float32Array;
  patchedNodeNames: string[];
  models: {
    // Node/WASM unit-test equivalence pair only (unrewritten rank-5 GatherND must NOT be dispatched to WebGPU)
    unrewrittenRank5UnitTest: Uint8Array;
    h1RewrittenUnitTest: Uint8Array;
    // Benign rank-4 WebGPU-safe synthetic controls (merged spatial batch 6, batch_dims=2 -> Reshape [1,2,3,1,2])
    s7aConstIdx: Uint8Array;
    s7bInputIdx: Uint8Array;
    s7cDynamicArgMin: Uint8Array;
    s7dTwoBranch: Uint8Array;
  };
}

/**
 * Constructs:
 * 1) Standalone rank-5 `GatherND(batch_dims=3)` + `H1`-rewritten pair for Node/WASM equivalence testing.
 * 2) Benign rank-4 WebGPU-safe synthetic controls (`S7a_ConstIdx`, `S7b_InputIdx`, `S7c_DynamicArgMin`, `S7d_TwoBranch`)
 *    verifying constant int64 indices (including negative `-1..-4`), graph-input int64 indices, dynamic `ArgMin`,
 *    and two-branch execution using the exact H1 rank-4 merged-batch shape pattern.
 */
export function buildSyntheticPropGatherNdControlModels(): SyntheticPropGatherNdSuite {
  const dataDims5: [number, number, number, number, number] = [1, 2, 3, 4, 2];
  const dataDims4: [number, number, number, number] = [1, 6, 4, 2];
  const idxDims5: [number, number, number, number, number] = [1, 2, 3, 1, 1];
  const idxDims4: [number, number, number, number] = [1, 6, 1, 1];
  const cost4Dims: [number, number, number, number] = [1, 2, 3, 4];
  const cost8Dims: [number, number, number, number] = [1, 2, 3, 8];
  const outDims5 = [1, 2, 3, 1, 2];

  const dataF32 = new Float32Array(1 * 6 * 4 * 2);
  for (let i = 0; i < dataF32.length; i++) {
    dataF32[i] = 10 + i * 0.5;
  }

  // Mix valid positive and negative indices in [-4, 3] across all 6 spatial positions
  const idxNums = [0, -1, 2, -4, 1, -2];
  const idxI64 = new BigInt64Array(idxNums.map((v) => BigInt(v)));

  const expectedConstOrInputOut = new Float32Array(6 * 2);
  for (let s = 0; s < 6; s++) {
    const c = idxNums[s] < 0 ? 4 + idxNums[s] : idxNums[s];
    const base = (s * 4 + c) * 2;
    expectedConstOrInputOut[s * 2] = dataF32[base];
    expectedConstOrInputOut[s * 2 + 1] = dataF32[base + 1];
  }

  const argMinSuite = buildSyntheticArgMinControlModels();
  const cost4F32 = argMinSuite.cost4F32;
  const cost8F32 = argMinSuite.cost8F32;
  const expectedCastOut = argMinSuite.expectedCastIndices;
  const expectedDynamicArgMinOut = new Float32Array(6 * 2);
  const expectedArgMinIdx4 = [2, 1, 0, 3, 0, 2];
  for (let s = 0; s < 6; s++) {
    const c = expectedArgMinIdx4[s];
    const base = (s * 4 + c) * 2;
    expectedDynamicArgMinOut[s * 2] = dataF32[base];
    expectedDynamicArgMinOut[s * 2 + 1] = dataF32[base + 1];
  }

  // 1) Node/WASM unit test model pair (rank-5 batch_dims=3 vs H1-rewritten rank-4 batch_dims=2)
  const rawGnd5 = makeNode('GatherND', 'prop0_gnd_rank5', ['data', 'idx'], ['out'], [
    attrInt('batch_dims', 3),
  ]);
  const unrewrittenRank5UnitTest = buildGraphModel(
    'synthetic_prop0_rank5_unit',
    [rawGnd5],
    [makeValueInfo('data', 1, dataDims5), makeValueInfo('idx', 7, idxDims5)],
    [makeOutputValueInfo('out', 1, outDims5)]
  );
  const h1Patched = normalizeHitnetPropGatherNdRankForWebGpu(unrewrittenRank5UnitTest, {
    requireHitnetContract: false,
  });
  if (h1Patched.normalizedCount !== 1) {
    throw new Error('Failed to apply H1 rewrite to standalone rank-5 unit model');
  }

  const initIdxConst4 = makeInt64Initializer('idx_const_4d', idxDims4, idxNums);
  const initIdxShape4 = makeInt64Initializer('idx_shape_4d', [4], idxDims4);
  const initOutShape5 = makeInt64Initializer('out_shape_5d', [5], outDims5);
  const initNormShape4 = makeInt64Initializer('norm_shape_4d', [4], [1, 2, 3, 1]);

  // S7a_ConstIdx: constant int64 [1,6,1,1] (with negative indices) -> GatherND(batch_dims=2) -> Reshape([1,2,3,1,2])
  const s7aConstIdx = buildGraphModel(
    'synthetic_h1_s7a_const_idx',
    [
      makeNode('GatherND', 's7a_gnd', ['data4', 'idx_const_4d'], ['s7a_flat'], [
        attrInt('batch_dims', 2),
      ]),
      makeNode('Reshape', 's7a_post_reshape', ['s7a_flat', 'out_shape_5d'], ['out_s7a']),
    ],
    [makeValueInfo('data4', 1, dataDims4)],
    [makeOutputValueInfo('out_s7a', 1, outDims5)],
    [initIdxConst4, initOutShape5]
  );

  // S7b_InputIdx: graph-input int64 [1,6,1,1] (with negative indices) -> GatherND(batch_dims=2) -> Reshape([1,2,3,1,2])
  const s7bInputIdx = buildGraphModel(
    'synthetic_h1_s7b_input_idx',
    [
      makeNode('GatherND', 's7b_gnd', ['data4', 'idx4'], ['s7b_flat'], [
        attrInt('batch_dims', 2),
      ]),
      makeNode('Reshape', 's7b_post_reshape', ['s7b_flat', 'out_shape_5d'], ['out_s7b']),
    ],
    [makeValueInfo('data4', 1, dataDims4), makeValueInfo('idx4', 7, idxDims4)],
    [makeOutputValueInfo('out_s7b', 1, outDims5)],
    [initOutShape5]
  );

  // S7c_DynamicArgMin: cost4 [1,2,3,4] -> ArgMin(keepdims=0) -> Reshape([1,6,1,1]) -> GatherND(data4, batch_dims=2) -> Reshape([1,2,3,1,2])
  const s7cDynamicArgMin = buildGraphModel(
    'synthetic_h1_s7c_dynamic_argmin',
    [
      makeNode('ArgMin', 's7c_argmin', ['cost4'], ['s7c_idx_3d'], [
        attrInt('axis', 3),
        attrInt('keepdims', 0),
        attrInt('select_last_index', 0),
      ]),
      makeNode('Reshape', 's7c_idx_reshape', ['s7c_idx_3d', 'idx_shape_4d'], ['s7c_idx_4d']),
      makeNode('GatherND', 's7c_gnd', ['data4', 's7c_idx_4d'], ['s7c_flat'], [
        attrInt('batch_dims', 2),
      ]),
      makeNode('Reshape', 's7c_post_reshape', ['s7c_flat', 'out_shape_5d'], ['out_s7c']),
    ],
    [makeValueInfo('cost4', 1, cost4Dims), makeValueInfo('data4', 1, dataDims4)],
    [makeOutputValueInfo('out_s7c', 1, outDims5)],
    [initIdxShape4, initOutShape5]
  );

  // S7d_TwoBranch: Branch 1 (cost8 ArgMin keepdims=0 -> Cast) + Branch 2 (cost4 ArgMin keepdims=1 -> Reshape([1,6,1,1]) -> GatherND(batch_dims=2) -> Reshape([1,2,3,1,2]))
  const s7dTwoBranch = buildGraphModel(
    'synthetic_h1_s7d_two_branch',
    [
      makeNode('ArgMin', 's7d_init_argmin', ['cost8'], ['s7d_init_raw'], [
        attrInt('axis', 3),
        attrInt('keepdims', 0),
        attrInt('select_last_index', 0),
      ]),
      makeNode('Reshape', 's7d_init_reshape', ['s7d_init_raw', 'norm_shape_4d'], ['s7d_init_4d']),
      makeNode('Cast', 's7d_init_cast', ['s7d_init_4d'], ['out_cast'], [attrInt('to', 1)]),
      makeNode('ArgMin', 's7d_prop_argmin', ['cost4'], ['s7d_prop_raw'], [
        attrInt('axis', 3),
        attrInt('keepdims', 1),
        attrInt('select_last_index', 0),
      ]),
      makeNode('Reshape', 's7d_prop_reshape', ['s7d_prop_raw', 'idx_shape_4d'], ['s7d_prop_4d']),
      makeNode('GatherND', 's7d_prop_gnd', ['data4', 's7d_prop_4d'], ['s7d_flat'], [
        attrInt('batch_dims', 2),
      ]),
      makeNode('Reshape', 's7d_prop_post_reshape', ['s7d_flat', 'out_shape_5d'], ['out_gathernd']),
    ],
    [
      makeValueInfo('cost8', 1, cost8Dims),
      makeValueInfo('cost4', 1, cost4Dims),
      makeValueInfo('data4', 1, dataDims4),
    ],
    [
      makeOutputValueInfo('out_cast', 1, [1, 2, 3, 1]),
      makeOutputValueInfo('out_gathernd', 1, outDims5),
    ],
    [initNormShape4, initIdxShape4, initOutShape5]
  );

  return {
    dataDims5,
    dataDims4,
    idxDims5,
    idxDims4,
    cost4Dims,
    cost8Dims,
    dataF32,
    idxI64,
    cost4F32,
    cost8F32,
    expectedConstOrInputOut,
    expectedDynamicArgMinOut,
    expectedCastOut,
    patchedNodeNames: h1Patched.normalizedNodeNames,
    models: {
      unrewrittenRank5UnitTest,
      h1RewrittenUnitTest: h1Patched.bytes,
      s7aConstIdx,
      s7bInputIdx,
      s7cDynamicArgMin,
      s7dTwoBranch,
    },
  };
}

export type HitnetCheckpointMode =
  | 'stem'
  | 'concat32'
  | 'p1'
  | 'p2'
  | 'p2b'
  | 'p3a'
  | 'p3b'
  | 'p3c'
  | 'p4'
  | 'p5'
  | 'p6'
  | 'p7'
  | 'p8'
  | 'p9'
  | 'p10'
  | 'p11';

export interface HitnetCheckpointSubgraphResult {
  bytes: Uint8Array;
  mode: HitnetCheckpointMode;
  targetOutputs: string[];
  outputMeta: Record<string, OnnxStaticInferenceResult['tensorMeta'][string]>;
  nodeCount: number;
  cutoffIdx: number;
  metadata: Record<string, unknown>;
}

export interface HitnetSubgraphArmsResult {
  mode: HitnetCheckpointMode;
  variant: 'h1' | 'f1' | 'g1';
  limitationNote: string;
  rawArm: HitnetCheckpointSubgraphResult;
  candidateArm: HitnetCheckpointSubgraphResult;
  fullRewriteCounts: {
    splitCount: number;
    decomposed5d: number;
    sliceNormalized: number;
    sliceCompacted: number;
    gatherNd: number;
    propGatherNd: number;
    argMin: number;
  };
}

interface SimpleProtoSlice {
  fn: number;
  wt: number;
  fStart: number;
  fEnd: number;
  valStart: number;
  valEnd: number;
}

function readSimpleVarint(buf: Uint8Array, pos: number, end: number): [number, number] | null {
  let result = 0n;
  let shift = 0n;
  let c = pos;
  while (c < end && shift <= 63n) {
    const b = buf[c++];
    result |= BigInt(b & 0x7f) << shift;
    if ((b & 0x80) === 0) {
      return [Number(result & 0xffffffffn) >>> 0, c];
    }
    shift += 7n;
  }
  return null;
}

function parseSimpleFields(buf: Uint8Array, start: number, end: number): SimpleProtoSlice[] | null {
  const out: SimpleProtoSlice[] = [];
  let c = start;
  while (c < end) {
    const fStart = c;
    const k = readSimpleVarint(buf, c, end);
    if (!k) return null;
    const fn = k[0] >>> 3;
    const wt = k[0] & 7;
    c = k[1];
    if (wt === 0) {
      const v = readSimpleVarint(buf, c, end);
      if (!v) return null;
      out.push({ fn, wt, fStart, fEnd: v[1], valStart: c, valEnd: v[1] });
      c = v[1];
    } else if (wt === 2) {
      const l = readSimpleVarint(buf, c, end);
      if (!l) return null;
      const vs = l[1];
      const ve = vs + l[0];
      if (ve > end) return null;
      out.push({ fn, wt, fStart, fEnd: ve, valStart: vs, valEnd: ve });
      c = ve;
    } else if (wt === 5) {
      if (c + 4 > end) return null;
      out.push({ fn, wt, fStart, fEnd: c + 4, valStart: c, valEnd: c + 4 });
      c += 4;
    } else if (wt === 1) {
      if (c + 8 > end) return null;
      out.push({ fn, wt, fStart, fEnd: c + 8, valStart: c, valEnd: c + 8 });
      c += 8;
    } else {
      return null;
    }
  }
  return out;
}

/**
 * Extracts a valid minimal SSA checkpoint subgraph from an ONNX byte buffer (`raw` OR already
 * transformed full HITNet) and constructs exact per-arm `ValueInfoProto` graph outputs by running
 * `inferOnnxStaticTensorMeta` independently on that arm's graph and verifying the cut graph.
 */
export function extractHitnetCheckpointSubgraph(
  armModelBytes: Uint8Array,
  mode: HitnetCheckpointMode,
  includeIntermediateTaps = false
): HitnetCheckpointSubgraphResult {
  const dec = new TextDecoder();
  const top = parseSimpleFields(armModelBytes, 0, armModelBytes.byteLength);
  if (!top) {
    throw new Error(`Invalid ONNX protobuf for checkpoint=${mode}`);
  }
  const gf = top.find((f) => f.fn === 7 && f.wt === 2);
  if (!gf) {
    throw new Error(`Missing GraphProto for checkpoint=${mode}`);
  }
  const gFields = parseSimpleFields(armModelBytes, gf.valStart, gf.valEnd);
  if (!gFields) {
    throw new Error(`Invalid GraphProto for checkpoint=${mode}`);
  }

  const fullInference = inferOnnxStaticTensorMeta(armModelBytes);
  if (!fullInference.valid) {
    throw new Error(
      `Static shape/structure validation failed before subgraph cut for checkpoint=${mode}`
    );
  }

  interface SubNode {
    idx: number;
    slice: Uint8Array;
    inputs: string[];
    outputs: string[];
    opType: string;
    name: string;
  }

  const nodes: SubNode[] = [];
  const inits = new Map<string, Uint8Array>();
  const graphInputs = new Map<string, Uint8Array>();

  for (const f of gFields) {
    if (f.fn === 1 && f.wt === 2) {
      const nf = parseSimpleFields(armModelBytes, f.valStart, f.valEnd);
      if (!nf) throw new Error(`Invalid NodeProto at checkpoint=${mode}`);
      const inputs: string[] = [];
      const outputs: string[] = [];
      let opType = '';
      let name = '';
      for (const n of nf) {
        if (n.fn === 1 && n.wt === 2) {
          inputs.push(
            n.valEnd > n.valStart ? dec.decode(armModelBytes.subarray(n.valStart, n.valEnd)) : ''
          );
        } else if (n.fn === 2 && n.wt === 2 && n.valEnd > n.valStart) {
          outputs.push(dec.decode(armModelBytes.subarray(n.valStart, n.valEnd)));
        } else if (n.fn === 3 && n.wt === 2 && n.valEnd > n.valStart) {
          name = dec.decode(armModelBytes.subarray(n.valStart, n.valEnd));
        } else if (n.fn === 4 && n.wt === 2 && n.valEnd > n.valStart) {
          opType = dec.decode(armModelBytes.subarray(n.valStart, n.valEnd));
        }
      }
      nodes.push({
        idx: nodes.length,
        slice: armModelBytes.subarray(f.fStart, f.fEnd),
        inputs,
        outputs,
        opType,
        name,
      });
    } else if (f.fn === 5 && f.wt === 2) {
      const tf = parseSimpleFields(armModelBytes, f.valStart, f.valEnd);
      const nf = tf?.find((x) => x.fn === 8 && x.wt === 2 && x.valEnd > x.valStart);
      if (nf) {
        inits.set(
          dec.decode(armModelBytes.subarray(nf.valStart, nf.valEnd)),
          armModelBytes.subarray(f.fStart, f.fEnd)
        );
      }
    } else if (f.fn === 11 && f.wt === 2) {
      const vf = parseSimpleFields(armModelBytes, f.valStart, f.valEnd);
      const nf = vf?.find((x) => x.fn === 1 && x.wt === 2 && x.valEnd > x.valStart);
      if (nf) {
        graphInputs.set(
          dec.decode(armModelBytes.subarray(nf.valStart, nf.valEnd)),
          armModelBytes.subarray(f.fStart, f.fEnd)
        );
      }
    }
  }

  let cutoffIdx: number;
  let targetOutputs: string[];
  let metadata: Record<string, unknown>;

  if (mode === 'stem') {
    const sliceNodes = nodes.filter((n) => n.opType === 'Slice');
    cutoffIdx = nodes.findIndex((n) => n.opType === 'Conv');
    if (sliceNodes.length < 2 || cutoffIdx < 0) {
      throw new Error('Missing stem Slice/Conv nodes');
    }
    targetOutputs = [sliceNodes[0].outputs[0], sliceNodes[1].outputs[0], nodes[cutoffIdx].outputs[0]];
    metadata = {
      checkpoint: 'stem (RGB Slices + first Conv)',
      sliceNames: [sliceNodes[0].name, sliceNodes[1].name],
      convName: nodes[cutoffIdx].name,
    };
  } else if (mode === 'concat32') {
    const slice31 = nodes.find((n) => n.opType === 'Slice' && n.name === 'fe_shared/strided_slice_2');
    const concat32 = nodes.find((n) => n.opType === 'Concat' && n.name === 'fe_shared/concat');
    if (!slice31 || !concat32) {
      throw new Error('Missing concat32 Slice/Concat nodes');
    }
    const conv33 = nodes.find(
      (n, idx) => idx > concat32.idx && n.opType === 'Conv' && n.inputs.includes(concat32.outputs[0])
    );
    cutoffIdx = conv33 ? conv33.idx : concat32.idx;
    const sinkOutput = conv33 ? conv33.outputs[0] : concat32.outputs[0];
    targetOutputs = includeIntermediateTaps
      ? [concat32.inputs[1], slice31.inputs[0], slice31.outputs[0], concat32.outputs[0], sinkOutput]
      : [sinkOutput];
    metadata = { checkpoint: 'concat32 (#31 -> #32 -> #33)', sinkOutput };
  } else {
    const modeTargets: Record<
      Exclude<HitnetCheckpointMode, 'stem' | 'concat32'>,
      { outputs: string[]; label: string }
    > = {
      p1: {
        outputs: ['fe_shared_5', 'fe_shared_51', 'fe_shared_4', 'fe_shared_41', 'fe_shared_3', 'fe_shared_31'],
        label: 'P1 (fe_shared multi-scale features)',
      },
      p2: {
        outputs: [
          'level0/level_init/Reshape_2',
          'level0/level_init/Cast',
          'level0/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis',
        ],
        label: 'P2 (level0 level_init cost + Cast ArgMin + GatherND)',
      },
      p2b: {
        outputs: ['level0/level_init/init_to_prop/ExpandDims'],
        label: 'P2b (level0 init_to_prop rank-5 ExpandDims [1,60,80,1,16])',
      },
      p3a: {
        outputs: ['level0/shared/prop0/WarpImageWithHypotheses/strided_slice_3'],
        label: 'P3a (level0 prop0 pre-5D-ConvTranspose strided_slice_3 [1,60,80,1,3])',
      },
      p3b: {
        outputs: ['level0/shared/prop0/WarpImageWithHypotheses/get_warp1d_coordinates_from_tile_hypotheses/Reshape_2'],
        label: 'P3b (level0 prop0 post-5D-ConvTranspose warp1d coordinates [1,240,320,1])',
      },
      p3c: {
        outputs: ['level0/shared/prop0/prop_warp/Reshape_1'],
        label: 'P3c (level0 prop0 post-warp_1d prop_warp/Reshape_1 [1,60,80,1,16])',
      },
      p4: {
        outputs: ['level0/shared/prop0/add'],
        label: 'P4 (level0 prop0 pre-#218 Add [1,60,80,1,16])',
      },
      p5: {
        outputs: [
          'level0/shared/prop0/strided_slice_4__673:0',
          'level0/shared/prop0/add',
          'level0/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis',
        ],
        label: 'P5 (level0 prop0 before & after #218, preserving both branches)',
      },
      p6: {
        outputs: [
          'level0_1/shared/prop0/strided_slice_41__385:0',
          'level0_1/shared/prop0/add',
          'level0_1/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis',
        ],
        label: 'P6 (level0_1 symmetric prop0 before & after #506, preserving both branches)',
      },
      p7: {
        outputs: [
          'shared/refinement_l0/ExpandDims',
          'shared_1/refinement_l0/ExpandDims1',
        ],
        label: 'P7 (post-prop0 refinement_l0 ExpandDims [1,60,80,1,16], both branches)',
      },
      p8: {
        outputs: [
          'shared/refinement_l0/UpsampleHypotheses/Reshape',
          'shared/refinement_l0/UpsampleHypotheses/Reshape_1',
          'shared/refinement_l0/strided_slice_17__750:0',
          'shared_1/refinement_l0/UpsampleHypotheses/Reshape',
          'shared_1/refinement_l0/UpsampleHypotheses/Reshape_1',
          'shared_1/refinement_l0/strided_slice_17__462:0',
        ],
        label: 'P8 (l0->l1 UpsampleHypotheses slant/tile Reshape & merged [1,120,160,16], both branches)',
      },
      p9: {
        outputs: [
          'shared/refinement_l1/ExpandDims',
          'shared_1/refinement_l1/ExpandDims1',
        ],
        label: 'P9 (refinement_l1 ExpandDims [1,120,160,1,16], both branches)',
      },
      p10: {
        outputs: [
          'shared/refinement_l1/UpsampleHypotheses/Reshape',
          'shared/refinement_l1/UpsampleHypotheses/Reshape_1',
          'shared/refinement_l1/strided_slice_17__821:0',
          'shared_1/refinement_l1/UpsampleHypotheses/Reshape',
          'shared_1/refinement_l1/UpsampleHypotheses/Reshape_1',
          'shared_1/refinement_l1/strided_slice_173__533:0',
        ],
        label: 'P10 (l1->l2 UpsampleHypotheses slant/tile Reshape & merged [1,240,320,16], both branches)',
      },
      p11: {
        outputs: [
          'shared/refinement_l2/add',
          'shared_1/refinement_l2/add',
        ],
        label: 'P11 (refinement_l2/add pre-head [1,16,240,320], both branches)',
      },
    };
    const spec = modeTargets[mode];
    targetOutputs = [...spec.outputs];
    const producerIndices = targetOutputs.map((outName) => {
      const idx = nodes.findIndex((n) => n.outputs.includes(outName));
      if (idx < 0) {
        throw new Error(`Target output '${outName}' producer node not found for checkpoint=${mode}`);
      }
      return idx;
    });
    cutoffIdx = Math.max(...producerIndices);
    metadata = { checkpoint: spec.label, cutoffIdx };
  }

  const needed = new Set(targetOutputs);
  const keepNodes: SubNode[] = [];
  for (let i = cutoffIdx; i >= 0; i--) {
    const n = nodes[i];
    if (n.outputs.some((o) => needed.has(o))) {
      keepNodes.push(n);
      for (const inp of n.inputs) {
        if (inp) needed.add(inp);
      }
    }
  }
  keepNodes.reverse();

  // Derive exact per-arm ValueInfoProto type and shape independently from arm inference
  const outputMeta: Record<string, OnnxStaticInferenceResult['tensorMeta'][string]> = {};
  const outputValueInfos: Uint8Array[] = [];
  for (const outName of targetOutputs) {
    const meta = fullInference.tensorMeta[outName];
    if (
      !meta ||
      !meta.elemType ||
      !meta.dims ||
      meta.dims.length === 0 ||
      meta.dims.some((d) => !Number.isInteger(d) || d <= 0)
    ) {
      throw new Error(
        `Exact static type/shape unavailable for '${outName}' in checkpoint=${mode} (inferred=${JSON.stringify(meta ?? null)})`
      );
    }
    outputMeta[outName] = { elemType: meta.elemType, dims: [...meta.dims] };
    outputValueInfos.push(makeOutputValueInfo(outName, meta.elemType, meta.dims));
  }

  const gParts: Uint8Array[] = [
    ...keepNodes.map((n) => n.slice),
    strField(2, `hitnet_${mode}_subgraph`),
  ];
  for (const [k, v] of inits) if (needed.has(k)) gParts.push(v);
  for (const [k, v] of graphInputs) if (needed.has(k) || k === 'input') gParts.push(v);
  for (const vo of outputValueInfos) gParts.push(vo);

  const subBytes = concat(
    top.map((f) =>
      f.fn === 7 && f.wt === 2 ? lenField(7, concat(gParts)) : armModelBytes.subarray(f.fStart, f.fEnd)
    )
  );

  // Independently verify the cut subgraph itself has valid structure & matching static outputs
  const cutInference = inferOnnxStaticTensorMeta(subBytes);
  if (
    !cutInference.valid ||
    cutInference.missingInputs.length > 0 ||
    cutInference.duplicateOutputs.length > 0 ||
    cutInference.duplicateInitializers.length > 0
  ) {
    throw new Error(
      `Cut subgraph verification failed for checkpoint=${mode}: missing=${cutInference.missingInputs.length}, dupOut=${cutInference.duplicateOutputs.length}`
    );
  }
  for (const outName of targetOutputs) {
    const cm = cutInference.tensorMeta[outName];
    const em = outputMeta[outName];
    if (
      !cm ||
      cm.elemType !== em.elemType ||
      cm.dims.length !== em.dims.length ||
      cm.dims.some((d, i) => d !== em.dims[i])
    ) {
      throw new Error(
        `Cut subgraph shape mismatch for '${outName}' at checkpoint=${mode}: expected=${JSON.stringify(em)} got=${JSON.stringify(cm ?? null)}`
      );
    }
  }

  return {
    bytes: subBytes,
    mode,
    targetOutputs,
    outputMeta,
    nodeCount: keepNodes.length,
    cutoffIdx,
    metadata,
  };
}

/**
 * Prepares Raw (unmodified WASM reference) and Candidate (`h1` default, `f1`, or `g1`)
 * HITNet subgraphs separately from full-model bytes, inferring each arm's output types/shapes
 * independently without leaking rank or layout assumptions across arms.
 */
export function prepareHitnetSubgraphArms(
  rawBytes: Uint8Array,
  mode: HitnetCheckpointMode,
  options: { variant?: 'h1' | 'f1' | 'g1'; includeIntermediateTaps?: boolean } = {}
): HitnetSubgraphArmsResult {
  const variant = options.variant ?? 'h1';
  const includeIntermediateTaps = options.includeIntermediateTaps ?? mode === 'stem';

  const split = splitLargeOnnxConcatNodes(rawBytes, 4);
  const afterSplit = split.splitCount > 0 ? split.bytes : stripOnnxGraphValueInfo(rawBytes);
  const dec = decomposeOnnx5dConvNodesForWebGpu(afterSplit);
  const afterConv3d = dec.decomposedCount > 0 ? dec.bytes : afterSplit;
  const slc = normalizeNonAscendingSliceAxesForWebGpu(afterConv3d, { mode: 'compact-active' });
  const gndF1 = normalizeHitnetGatherNdRankForWebGpu(slc.bytes, {
    variantId: 'hitnet-sf',
    backendPreference: 'webgpu',
    requireHitnetContract: true,
  });

  let candidateFullBytes = gndF1.bytes;
  let propGatherNdCount = 0;
  let argMinCount = 0;

  if (variant === 'h1') {
    const h1 = normalizeHitnetPropGatherNdRankForWebGpu(gndF1.bytes, {
      variantId: 'hitnet-sf',
      backendPreference: 'webgpu',
      requireHitnetContract: true,
    });
    if (h1.normalizedCount !== 2) {
      throw new Error(
        `H1 transform failed preconditions on HITNet model before subgraph cut (propGatherNd=${h1.normalizedCount})`
      );
    }
    candidateFullBytes = h1.bytes;
    propGatherNdCount = h1.normalizedCount;
  } else if (variant === 'g1') {
    const g1 = normalizeHitnetArgMinKeepDimsForWebGpu(gndF1.bytes, {
      variantId: 'hitnet-sf',
      backendPreference: 'webgpu',
      requireHitnetContract: true,
    });
    candidateFullBytes = g1.bytes;
    argMinCount = g1.normalizedCount;
  }

  const rawArm = extractHitnetCheckpointSubgraph(rawBytes, mode, includeIntermediateTaps);
  const candidateArm = extractHitnetCheckpointSubgraph(
    candidateFullBytes,
    mode,
    includeIntermediateTaps
  );

  return {
    mode,
    variant,
    limitationNote:
      'Subgraph extraction truncates downstream nodes, which can alter EP assignment (WebGPU vs ORT-CPU), graph optimization fusions, and WebGPU shader cache order compared to full HITNet.',
    rawArm,
    candidateArm,
    fullRewriteCounts: {
      splitCount: split.splitCount,
      decomposed5d: dec.decomposedCount,
      sliceNormalized: slc.normalizedCount,
      sliceCompacted: slc.compactedCount,
      gatherNd: gndF1.normalizedCount,
      propGatherNd: propGatherNdCount,
      argMin: argMinCount,
    },
  };
}

export type HitnetFullGraphTapGroupId =
  | 'p2-argmin'
  | 'p2-cast'
  | 'p2-slot0-reshape'
  | 'p2-gather'
  | 'p2a-concat-fe'
  | 'p2a-concat-out'
  | 'p2a-concat1-leakyrelu'
  | 'p2a-concat1-zeros'
  | 'p2a-concat1-out'
  | 'p2b'
  | 'p3c'
  | 'p4'
  | 'p5-p6'
  | 'p5-rank4-indices'
  | 'p5-rank4-io-primary'
  | 'p5-rank4-io-secondary'
  | 'p7'
  | 'p8'
  | 'p8-slant'
  | 'p8-tile'
  | 'p9'
  | 'p10'
  | 'p10-slant'
  | 'p10-tile'
  | 'p11';

export interface HitnetFullGraphTapGroupSpec {
  id: HitnetFullGraphTapGroupId;
  label: string;
  stageNote: string;
  outputs: [string, string];
  isIntegerTensor?: boolean;
  isRank4GatherIoPair?: boolean;
}

export const HITNET_FULL_GRAPH_TAP_GROUPS: readonly HitnetFullGraphTapGroupSpec[] = [
  {
    id: 'p4',
    label: 'In-situ P4 (prop0/add inputs to repaired GatherND, both branches)',
    stageNote: 'In-situ prop0/add [1,60,80,1,16] float32 directly feeding repaired H1 prop0 GatherND',
    outputs: [
      'level0/shared/prop0/add',
      'level0_1/shared/prop0/add',
    ],
  },
  {
    id: 'p2-gather',
    label: 'In-situ P2-Gather (level_init/GatherV2_1 output, both branches)',
    stageNote: 'In-situ level_init GatherV2_1 [1,60,80,1] float32 before init_to_prop',
    outputs: [
      'level0/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis',
      'level0_1/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis',
    ],
  },
  {
    id: 'p2-argmin',
    label: 'In-situ P2-ArgMin (level_init/ArgMin, both branches)',
    stageNote: 'In-situ level_init ArgMin [1,60,80] int64 disparity indices',
    outputs: [
      'level0/level_init/ArgMin',
      'level0_1/level_init/ArgMin',
    ],
    isIntegerTensor: true,
  },
  {
    id: 'p2-cast',
    label: 'In-situ P2-Cast (level_init/Cast, both branches)',
    stageNote: 'In-situ level_init Cast [1,60,80,1] float32 disparity indices',
    outputs: [
      'level0/level_init/Cast',
      'level0_1/level_init/Cast',
    ],
  },
  {
    id: 'p2-slot0-reshape',
    label: 'In-situ P2-Slot0-Reshape (Transpose__908:0 / Transpose__1006:0 [1,1,60,80], both branches)',
    stageNote: 'In-situ slot-0 disparity input [1,1,60,80] float32 entering init_to_prop/concat_1 (separate from concat_1 ch0)',
    outputs: [
      'Transpose__908:0',
      'Transpose__1006:0',
    ],
  },
  {
    id: 'p2a-concat-fe',
    label: 'In-situ P2a-Concat-FE (fe_shared_3 & fe_shared_31 inputs to init_to_prop/concat)',
    stageNote: '48ch feature input [1,60,80,48] float32 entering level_init/init_to_prop/concat',
    outputs: [
      'fe_shared_3',
      'fe_shared_31',
    ],
  },
  {
    id: 'p2a-concat-out',
    label: 'In-situ P2a-Concat-Out (level_init/init_to_prop/concat output, both branches)',
    stageNote: 'init_to_prop/concat [1,60,80,49] float32 (ch0=GatherV2_1/ReduceMin, ch1..48=fe_shared)',
    outputs: [
      'level0/level_init/init_to_prop/concat',
      'level0_1/level_init/init_to_prop/concat',
    ],
  },
  {
    id: 'p2a-concat1-leakyrelu',
    label: 'In-situ P2a-Concat1-LeakyRelu (init_to_prop/LeakyRelu input to concat_1)',
    stageNote: '13ch init_to_prop/LeakyRelu [1,13,60,80] float32 entering init_to_prop/concat_1',
    outputs: [
      'level0/level_init/init_to_prop/LeakyRelu',
      'level0_1/level_init/init_to_prop/LeakyRelu',
    ],
  },
  {
    id: 'p2a-concat1-zeros',
    label: 'In-situ P2a-Concat1-Zeros (Transpose__910/1008 zeros input to concat_1)',
    stageNote: 'Slant zero tensor [1,1,60,80] float32 passed to inputs 1 & 2 of init_to_prop/concat_1 (pre-M1 only)',
    outputs: [
      'Transpose__910:0',
      'Transpose__1008:0',
    ],
  },
  {
    id: 'p2a-concat1-out',
    label: 'In-situ P2a-Concat1-Out (init_to_prop/concat_1 [1,16,60,80], both branches)',
    stageNote: 'Lowest useful Concat1 output [1,16,60,80] float32 in M1/H1 (ch0=disp, ch1..2=zero slant, ch3..15=LeakyRelu)',
    outputs: [
      'level0/level_init/init_to_prop/concat_1',
      'level0_1/level_init/init_to_prop/concat_1',
    ],
  },
  {
    id: 'p2b',
    label: 'In-situ P2b (level_init/init_to_prop/ExpandDims, both branches)',
    stageNote: 'In-situ init_to_prop [1,60,80,1,16] float32 feeding prop0/add',
    outputs: [
      'level0/level_init/init_to_prop/ExpandDims',
      'level0_1/level_init/init_to_prop/ExpandDims',
    ],
  },
  {
    id: 'p3c',
    label: 'In-situ P3c (prop0/prop_warp/Reshape_1, both branches)',
    stageNote: 'In-situ post-warp_1d prop_warp [1,60,80,1,16] float32 feeding prop0 conv/add',
    outputs: [
      'level0/shared/prop0/prop_warp/Reshape_1',
      'level0_1/shared/prop0/prop_warp/Reshape_1',
    ],
  },
  {
    id: 'p5-p6',
    label: 'In-situ P5/P6 (prop0 GatherV2 primary + secondary)',
    stageNote: 'In-situ post-#218/#506 prop0 output [1,60,80,1,16] in full H1 graph',
    outputs: [
      'level0/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis',
      'level0_1/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis',
    ],
  },
  {
    id: 'p5-rank4-indices',
    label: 'Optional P5 H1 rank4 singleton indices (both branches)',
    stageNote: 'Read-only check of H1 merged rank-4 GatherND singleton indices [1,4800,1,1] int64 (expect all 0)',
    outputs: [
      'level0/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis__h1_indices_4d_4',
      'level0_1/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis__h1_indices_4d_10',
    ],
    isIntegerTensor: true,
  },
  {
    id: 'p5-rank4-io-primary',
    label: 'Optional P5 H1 primary rank4 GatherND data + out',
    stageNote: 'Read-only check of primary H1 merged rank-4 data_4d & out_4d [1,4800,1,16] float32',
    outputs: [
      'level0/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis__h1_data_4d_3',
      'level0/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis__h1_out_4d_5',
    ],
    isRank4GatherIoPair: true,
  },
  {
    id: 'p5-rank4-io-secondary',
    label: 'Optional P5 H1 secondary rank4 GatherND data + out',
    stageNote: 'Read-only check of secondary H1 merged rank-4 data_4d & out_4d [1,4800,1,16] float32',
    outputs: [
      'level0_1/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis__h1_data_4d_9',
      'level0_1/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis__h1_out_4d_11',
    ],
    isRank4GatherIoPair: true,
  },
  {
    id: 'p7',
    label: 'In-situ P7 (refinement_l0 ExpandDims primary + secondary)',
    stageNote: 'Post-prop0 refinement_l0 residual output [1,60,80,1,16]',
    outputs: [
      'shared/refinement_l0/ExpandDims',
      'shared_1/refinement_l0/ExpandDims1',
    ],
  },
  {
    id: 'p8',
    label: 'Midpoint P8 (l0->l1 UpsampleHypotheses merged strided_slice_17)',
    stageNote: 'l0->l1 UpsampleHypotheses squeezed output [1,120,160,16]',
    outputs: [
      'shared/refinement_l0/strided_slice_17__750:0',
      'shared_1/refinement_l0/strided_slice_17__462:0',
    ],
  },
  {
    id: 'p8-slant',
    label: 'P8-slant (l0->l1 UpsampleHypotheses/Reshape slant)',
    stageNote: 'l0->l1 UpsampleHypotheses slant prediction [1,120,160,1,1]',
    outputs: [
      'shared/refinement_l0/UpsampleHypotheses/Reshape',
      'shared_1/refinement_l0/UpsampleHypotheses/Reshape',
    ],
  },
  {
    id: 'p8-tile',
    label: 'P8-tile (l0->l1 UpsampleHypotheses/Reshape_1 tiled)',
    stageNote: 'l0->l1 UpsampleHypotheses Tile output [1,120,160,1,16]',
    outputs: [
      'shared/refinement_l0/UpsampleHypotheses/Reshape_1',
      'shared_1/refinement_l0/UpsampleHypotheses/Reshape_1',
    ],
  },
  {
    id: 'p9',
    label: 'In-situ P9 (refinement_l1 ExpandDims primary + secondary)',
    stageNote: 'refinement_l1 residual output [1,120,160,1,16]',
    outputs: [
      'shared/refinement_l1/ExpandDims',
      'shared_1/refinement_l1/ExpandDims1',
    ],
  },
  {
    id: 'p10',
    label: 'Midpoint P10 (l1->l2 UpsampleHypotheses merged strided_slice_17)',
    stageNote: 'l1->l2 UpsampleHypotheses squeezed output [1,240,320,16]',
    outputs: [
      'shared/refinement_l1/strided_slice_17__821:0',
      'shared_1/refinement_l1/strided_slice_173__533:0',
    ],
  },
  {
    id: 'p10-slant',
    label: 'P10-slant (l1->l2 UpsampleHypotheses/Reshape slant)',
    stageNote: 'l1->l2 UpsampleHypotheses slant prediction [1,240,320,1,1]',
    outputs: [
      'shared/refinement_l1/UpsampleHypotheses/Reshape',
      'shared_1/refinement_l1/UpsampleHypotheses/Reshape',
    ],
  },
  {
    id: 'p10-tile',
    label: 'P10-tile (l1->l2 UpsampleHypotheses/Reshape_1 tiled)',
    stageNote: 'l1->l2 UpsampleHypotheses Tile output [1,240,320,1,16]',
    outputs: [
      'shared/refinement_l1/UpsampleHypotheses/Reshape_1',
      'shared_1/refinement_l1/UpsampleHypotheses/Reshape_1',
    ],
  },
  {
    id: 'p11',
    label: 'Late P11 (refinement_l2/add pre-head primary + secondary)',
    stageNote: 'refinement_l2/add before disparity head [1,16,240,320]',
    outputs: [
      'shared/refinement_l2/add',
      'shared_1/refinement_l2/add',
    ],
  },
];

export function verifySingletonZeroIndices(values: ArrayLike<number>): {
  ok: boolean;
  length: number;
  finiteCount: number;
  nonZeroCount: number;
  min: number;
  max: number;
} {
  const length = values.length;
  let finiteCount = 0;
  let nonZeroCount = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < length; i++) {
    const v = values[i];
    if (Number.isFinite(v)) {
      finiteCount++;
      if (v < min) min = v;
      if (v > max) max = v;
      if (v !== 0) nonZeroCount++;
    }
  }
  const ok = length > 0 && finiteCount === length && nonZeroCount === 0;
  return { ok, length, finiteCount, nonZeroCount, min, max };
}

export interface ChannelScalarDistributionStats {
  length: number;
  finiteCount: number;
  zeroCount: number;
  nonZeroCount: number;
  min: number;
  max: number;
  meanAbs: number;
}

function summarizeScalarDistribution(arr: Float32Array): ChannelScalarDistributionStats {
  const length = arr.length;
  let finiteCount = 0;
  let zeroCount = 0;
  let nonZeroCount = 0;
  let min = Infinity;
  let max = -Infinity;
  let absSum = 0;
  for (let i = 0; i < length; i++) {
    const v = arr[i];
    if (Number.isFinite(v)) {
      finiteCount++;
      if (v === 0) zeroCount++;
      else nonZeroCount++;
      if (v < min) min = v;
      if (v > max) max = v;
      absSum += Math.abs(v);
    } else {
      nonZeroCount++;
    }
  }
  return {
    length,
    finiteCount,
    zeroCount,
    nonZeroCount,
    min,
    max,
    meanAbs: length > 0 ? absSum / length : 0,
  };
}

export interface P2bChannelwiseParitySummary {
  ch0Disp: RawTensorParityStats;
  ch1_2Zeros: RawTensorParityStats;
  ch3_15Features: RawTensorParityStats;
  channels: RawTensorParityStats[];
  ch1VsRef: RawTensorParityStats;
  ch2VsRef: RawTensorParityStats;
  ch1CandidateDist: ChannelScalarDistributionStats;
  ch2CandidateDist: ChannelScalarDistributionStats;
  ch1ReferenceDist: ChannelScalarDistributionStats | null;
  ch2ReferenceDist: ChannelScalarDistributionStats | null;
  ch1VsCh2CandidateMaxAbsDiff: number;
  ch1VsCh2ReferenceMaxAbsDiff: number | null;
}

/**
 * Slices `[1, 60, 80, 1, 16]` (`76800` floats) P2b `init_to_prop/ExpandDims` into:
 *   - `ch 0`: Cast(ArgMin) initial disparity
 *   - `ch 1..2`: slant placeholders (`ch1` and `ch2` also summarized individually per backend)
 *   - `ch 3..15`: init_to_prop/LeakyRelu 13 feature channels (downstream of `init_to_prop/concat`)
 */
export function summarizeP2bChannelwiseParity(
  candidate: Float32Array,
  reference?: Float32Array | null
): P2bChannelwiseParitySummary {
  const spatial = 60 * 80;
  const channelsCount = 16;
  const expectedLen = spatial * channelsCount;
  if (candidate.length !== expectedLen || (reference && reference.length !== expectedLen)) {
    throw new Error(
      `Expected P2b length=${expectedLen}, got candidate=${candidate.length} reference=${reference?.length ?? 'none'}`
    );
  }
  const ch0C = new Float32Array(spatial);
  const ch0R = reference ? new Float32Array(spatial) : null;
  const ch12C = new Float32Array(spatial * 2);
  const ch12R = reference ? new Float32Array(spatial * 2) : null;
  const ch315C = new Float32Array(spatial * 13);
  const ch315R = reference ? new Float32Array(spatial * 13) : null;
  const perChC = Array.from({ length: channelsCount }, () => new Float32Array(spatial));
  const perChR = reference
    ? Array.from({ length: channelsCount }, () => new Float32Array(spatial))
    : null;

  for (let i = 0; i < spatial; i++) {
    const base = i * channelsCount;
    ch0C[i] = candidate[base];
    ch12C[i * 2] = candidate[base + 1];
    ch12C[i * 2 + 1] = candidate[base + 2];
    for (let c = 0; c < 13; c++) {
      ch315C[i * 13 + c] = candidate[base + 3 + c];
    }
    for (let c = 0; c < channelsCount; c++) {
      perChC[c][i] = candidate[base + c];
    }
    if (reference && ch0R && ch12R && ch315R && perChR) {
      ch0R[i] = reference[base];
      ch12R[i * 2] = reference[base + 1];
      ch12R[i * 2 + 1] = reference[base + 2];
      for (let c = 0; c < 13; c++) {
        ch315R[i * 13 + c] = reference[base + 3 + c];
      }
      for (let c = 0; c < channelsCount; c++) {
        perChR[c][i] = reference[base + c];
      }
    }
  }

  const ch1VsCh2CandidateMaxAbsDiff = summarizeRawTensorAndParity(perChC[1], perChC[2]).maxAbsDiff;
  const ch1VsCh2ReferenceMaxAbsDiff = perChR
    ? summarizeRawTensorAndParity(perChR[1], perChR[2]).maxAbsDiff
    : null;
  const channels = perChC.map((cArr, cIdx) =>
    summarizeRawTensorAndParity(cArr, perChR ? perChR[cIdx] : null)
  );

  return {
    ch0Disp: summarizeRawTensorAndParity(ch0C, ch0R),
    ch1_2Zeros: summarizeRawTensorAndParity(ch12C, ch12R),
    ch3_15Features: summarizeRawTensorAndParity(ch315C, ch315R),
    channels,
    ch1VsRef: channels[1],
    ch2VsRef: channels[2],
    ch1CandidateDist: summarizeScalarDistribution(perChC[1]),
    ch2CandidateDist: summarizeScalarDistribution(perChC[2]),
    ch1ReferenceDist: perChR ? summarizeScalarDistribution(perChR[1]) : null,
    ch2ReferenceDist: perChR ? summarizeScalarDistribution(perChR[2]) : null,
    ch1VsCh2CandidateMaxAbsDiff,
    ch1VsCh2ReferenceMaxAbsDiff,
  };
}

/**
 * Slices NCHW `[1, 16, 60, 80]` (`76800` floats) `init_to_prop/concat_1` output into:
 *   - `ch 0`: disparity (`Transpose__908:0` / `Transpose__1006:0`)
 *   - `ch 1..2`: zero slant channels (in M1 supplied by `[1,2,60,80]` `+0.0f` initializer)
 *   - `ch 3..15`: `init_to_prop/LeakyRelu` 13 channels
 */
export function summarizeInitToPropConcat1OutSplit(
  candidate: Float32Array,
  reference?: Float32Array | null
): P2bChannelwiseParitySummary {
  const spatial = 60 * 80;
  const channelsCount = 16;
  const expectedLen = spatial * channelsCount;
  if (candidate.length !== expectedLen || (reference && reference.length !== expectedLen)) {
    throw new Error(
      `Expected init_to_prop/concat_1 length=${expectedLen}, got candidate=${candidate.length} reference=${reference?.length ?? 'none'}`
    );
  }
  const perChC = Array.from({ length: channelsCount }, (_, c) =>
    candidate.subarray(c * spatial, (c + 1) * spatial)
  );
  const perChR = reference
    ? Array.from({ length: channelsCount }, (_, c) =>
        reference.subarray(c * spatial, (c + 1) * spatial)
      )
    : null;
  const ch12C = candidate.subarray(1 * spatial, 3 * spatial);
  const ch12R = reference ? reference.subarray(1 * spatial, 3 * spatial) : null;
  const ch315C = candidate.subarray(3 * spatial, 16 * spatial);
  const ch315R = reference ? reference.subarray(3 * spatial, 16 * spatial) : null;

  const ch1VsCh2CandidateMaxAbsDiff = summarizeRawTensorAndParity(perChC[1], perChC[2]).maxAbsDiff;
  const ch1VsCh2ReferenceMaxAbsDiff = perChR
    ? summarizeRawTensorAndParity(perChR[1], perChR[2]).maxAbsDiff
    : null;
  const channels = perChC.map((cArr, cIdx) =>
    summarizeRawTensorAndParity(cArr, perChR ? perChR[cIdx] : null)
  );

  return {
    ch0Disp: channels[0],
    ch1_2Zeros: summarizeRawTensorAndParity(ch12C, ch12R),
    ch3_15Features: summarizeRawTensorAndParity(ch315C, ch315R),
    channels,
    ch1VsRef: channels[1],
    ch2VsRef: channels[2],
    ch1CandidateDist: summarizeScalarDistribution(perChC[1]),
    ch2CandidateDist: summarizeScalarDistribution(perChC[2]),
    ch1ReferenceDist: perChR ? summarizeScalarDistribution(perChR[1]) : null,
    ch2ReferenceDist: perChR ? summarizeScalarDistribution(perChR[2]) : null,
    ch1VsCh2CandidateMaxAbsDiff,
    ch1VsCh2ReferenceMaxAbsDiff,
  };
}

export function summarizeInitToPropConcatOutSplit(
  candidate: Float32Array,
  reference?: Float32Array | null
): { ch0CostMin: RawTensorParityStats; ch1_48Features: RawTensorParityStats } {
  const spatial = 60 * 80;
  const channelsCount = 49;
  const expectedLen = spatial * channelsCount;
  if (candidate.length !== expectedLen || (reference && reference.length !== expectedLen)) {
    throw new Error(
      `Expected init_to_prop/concat length=${expectedLen}, got candidate=${candidate.length} reference=${reference?.length ?? 'none'}`
    );
  }
  const ch0C = new Float32Array(spatial);
  const ch0R = reference ? new Float32Array(spatial) : null;
  const ch148C = new Float32Array(spatial * 48);
  const ch148R = reference ? new Float32Array(spatial * 48) : null;
  for (let i = 0; i < spatial; i++) {
    const base = i * channelsCount;
    ch0C[i] = candidate[base];
    for (let c = 0; c < 48; c++) {
      ch148C[i * 48 + c] = candidate[base + 1 + c];
    }
    if (reference && ch0R && ch148R) {
      ch0R[i] = reference[base];
      for (let c = 0; c < 48; c++) {
        ch148R[i * 48 + c] = reference[base + 1 + c];
      }
    }
  }
  return {
    ch0CostMin: summarizeRawTensorAndParity(ch0C, ch0R),
    ch1_48Features: summarizeRawTensorAndParity(ch148C, ch148R),
  };
}

export function getHitnetFullGraphTapGroup(
  id: HitnetFullGraphTapGroupId
): HitnetFullGraphTapGroupSpec {
  const found = HITNET_FULL_GRAPH_TAP_GROUPS.find((g) => g.id === id);
  if (!found) {
    throw new Error(`Unknown HITNet full-graph tap group: ${String(id)}`);
  }
  return found;
}

export function prepareHitnetH1FullGraphBytes(rawBytes: Uint8Array): {
  bytes: Uint8Array;
  counts: {
    splitCount: number;
    decomposed5d: number;
    sliceNormalized: number;
    sliceCompacted: number;
    gatherNd: number;
    propGatherNd: number;
  };
} {
  const split = splitLargeOnnxConcatNodes(rawBytes, 4);
  const afterSplit = split.splitCount > 0 ? split.bytes : stripOnnxGraphValueInfo(rawBytes);
  const dec = decomposeOnnx5dConvNodesForWebGpu(afterSplit);
  const afterConv3d = dec.decomposedCount > 0 ? dec.bytes : afterSplit;
  const slc = normalizeNonAscendingSliceAxesForWebGpu(afterConv3d, { mode: 'compact-active' });
  const gnd = normalizeHitnetGatherNdRankForWebGpu(slc.bytes, {
    variantId: 'hitnet-sf',
    backendPreference: 'webgpu',
    requireHitnetContract: true,
  });
  const h1 = normalizeHitnetPropGatherNdRankForWebGpu(gnd.bytes, {
    variantId: 'hitnet-sf',
    backendPreference: 'webgpu',
    requireHitnetContract: true,
  });
  if (
    dec.decomposedCount !== 2 ||
    slc.compactedCount === 0 ||
    gnd.normalizedCount !== 4 ||
    h1.normalizedCount !== 2
  ) {
    throw new Error(
      `H1 full-graph preconditions failed (5d=${dec.decomposedCount}, compacted=${slc.compactedCount}, f1=${gnd.normalizedCount}, h1=${h1.normalizedCount})`
    );
  }
  return {
    bytes: h1.bytes,
    counts: {
      splitCount: split.splitCount,
      decomposed5d: dec.decomposedCount,
      sliceNormalized: slc.normalizedCount,
      sliceCompacted: slc.compactedCount,
      gatherNd: gnd.normalizedCount,
      propGatherNd: h1.normalizedCount,
    },
  };
}

export function prepareHitnetLevelInitCandidateFullGraphBytes(
  rawBytes: Uint8Array,
  mode: Exclude<HitnetLevelInitGatherMode, 'off'>
): {
  bytes: Uint8Array;
  baseH1Counts: ReturnType<typeof prepareHitnetH1FullGraphBytes>['counts'];
  levelInitResult: OnnxLevelInitGatherNormalizeResult;
} {
  const h1 = prepareHitnetH1FullGraphBytes(rawBytes);
  const levelInitResult =
    mode === 'reshape-rank4'
      ? normalizeHitnetLevelInitGatherNdRankForWebGpu(h1.bytes, {
          variantId: 'hitnet-sf',
          backendPreference: 'webgpu',
          requireHitnetContract: true,
        })
      : normalizeHitnetLevelInitReduceMinForWebGpu(h1.bytes, {
          variantId: 'hitnet-sf',
          backendPreference: 'webgpu',
          requireHitnetContract: true,
        });
  if (levelInitResult.normalizedCount !== 2) {
    throw new Error(
      `HITNet level_init ${mode} full-graph preconditions failed (normalizedCount=${levelInitResult.normalizedCount}, skippedInvalid=${levelInitResult.skippedInvalidOrUnprovenCount})`
    );
  }
  return {
    bytes: levelInitResult.bytes,
    baseH1Counts: h1.counts,
    levelInitResult,
  };
}

export function prepareHitnetM1FullGraphBytes(
  rawBytes: Uint8Array,
  shaOptions?: { rawModelSha256?: string; h1ModelSha256?: string }
): {
  bytes: Uint8Array;
  baseH1Bytes: Uint8Array;
  baseH1Counts: ReturnType<typeof prepareHitnetH1FullGraphBytes>['counts'];
  m1Result: OnnxInitToPropZeroSlantNormalizeResult;
} {
  const h1 = prepareHitnetH1FullGraphBytes(rawBytes);
  const m1Result = normalizeHitnetInitToPropZeroSlantForWebGpu(h1.bytes, {
    variantId: 'hitnet-sf',
    backendPreference: 'webgpu',
    requireHitnetContract: true,
    rawModelSha256: shaOptions?.rawModelSha256,
    h1ModelSha256: shaOptions?.h1ModelSha256,
  });
  if (
    m1Result.normalizedCount !== 2 ||
    m1Result.removedDeadNodeNames.length !== 8 ||
    !m1Result.verifiedH1Provenance ||
    !m1Result.verifiedRawIdentity
  ) {
    throw new Error(
      `HITNet M1-ConstZero2Ch full-graph preconditions failed (normalized=${m1Result.normalizedCount}, removedDeadNodes=${m1Result.removedDeadNodeNames.length}, h1Prov=${m1Result.verifiedH1Provenance}, rawId=${m1Result.verifiedRawIdentity}, skippedInvalid=${m1Result.skippedInvalidOrUnprovenCount})`
    );
  }
  return {
    bytes: m1Result.bytes,
    baseH1Bytes: h1.bytes,
    baseH1Counts: h1.counts,
    m1Result,
  };
}

export function prepareHitnetN1FullGraphBytes(
  rawBytes: Uint8Array,
  shaOptions?: { rawModelSha256?: string; m1ModelSha256?: string }
): {
  bytes: Uint8Array;
  baseM1Bytes: Uint8Array;
  baseH1Bytes: Uint8Array;
  baseH1Counts: ReturnType<typeof prepareHitnetH1FullGraphBytes>['counts'];
  m1Result: OnnxInitToPropZeroSlantNormalizeResult;
  n1Result: OnnxSlot0ReshapeToTransposeResult;
} {
  const m1 = prepareHitnetM1FullGraphBytes(rawBytes, {
    rawModelSha256: shaOptions?.rawModelSha256,
  });
  const n1Result = normalizeHitnetSlot0ReshapeToTransposeForWebGpu(m1.bytes, {
    variantId: 'hitnet-sf',
    backendPreference: 'webgpu',
    requireHitnetContract: true,
    rawModelSha256: shaOptions?.rawModelSha256,
    m1ModelSha256: shaOptions?.m1ModelSha256,
  });
  if (
    n1Result.normalizedCount !== 2 ||
    !n1Result.verifiedM1Provenance ||
    !n1Result.verifiedRawIdentity
  ) {
    throw new Error(
      `HITNet N1-Slot0Transpose full-graph preconditions failed (normalized=${n1Result.normalizedCount}, m1Prov=${n1Result.verifiedM1Provenance}, rawId=${n1Result.verifiedRawIdentity}, skippedInvalid=${n1Result.skippedInvalidOrUnprovenCount})`
    );
  }
  return {
    bytes: n1Result.bytes,
    baseM1Bytes: m1.bytes,
    baseH1Bytes: m1.baseH1Bytes,
    baseH1Counts: m1.baseH1Counts,
    m1Result: m1.m1Result,
    n1Result,
  };
}

export type HitnetFullGraphTapBaseMode =
  | 'h1'
  | 'm1'
  | 'n1'
  | Exclude<HitnetLevelInitGatherMode, 'off'>;

export function prepareHitnetFullGraphTapGroup(
  rawBytes: Uint8Array,
  tapGroupId: HitnetFullGraphTapGroupId,
  baseMode: HitnetFullGraphTapBaseMode = 'h1'
): {
  group: HitnetFullGraphTapGroupSpec;
  baseMode: HitnetFullGraphTapBaseMode;
  baseH1Bytes: Uint8Array;
  tappedH1: AppendOnnxGraphOutputsResult;
} {
  if ((baseMode === 'm1' || baseMode === 'n1') && tapGroupId === 'p2a-concat1-zeros') {
    throw new Error(
      `Tap group "p2a-concat1-zeros" is removed by M1/N1; use "p2a-concat1-out" or "p2b" instead.`
    );
  }
  const group = getHitnetFullGraphTapGroup(tapGroupId);
  const baseH1Bytes =
    baseMode === 'h1'
      ? prepareHitnetH1FullGraphBytes(rawBytes).bytes
      : baseMode === 'm1'
        ? prepareHitnetM1FullGraphBytes(rawBytes).bytes
        : baseMode === 'n1'
          ? prepareHitnetN1FullGraphBytes(rawBytes).bytes
          : prepareHitnetLevelInitCandidateFullGraphBytes(rawBytes, baseMode).bytes;
  const tappedH1 = appendOnnxGraphOutputs(baseH1Bytes, group.outputs, { maxExtraOutputs: 2 });
  return { group, baseMode, baseH1Bytes, tappedH1 };
}

export {
  HITNET_RAW_MODEL_SHA256,
  HITNET_H1_FULL_GRAPH_SHA256,
  HITNET_M1_FULL_GRAPH_SHA256,
};





