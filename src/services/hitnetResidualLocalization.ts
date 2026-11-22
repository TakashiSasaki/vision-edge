import {
  appendOnnxGraphOutputs,
  getHitnetFullGraphTapGroup,
  prepareHitnetM1FullGraphBytes,
  summarizeP2bChannelwiseParity,
  summarizeRawTensorAndParity,
  type RawTensorParityStats,
} from './hitnetGatherNdDiagnostic';
import {
  HITNET_ORT_130_FAMILY_PROFILES,
  type OrtDiagnosticRuntimeFamily,
  type OrtVersionDiagnosticFailure,
  type OrtVersionDiagnosticRequest,
  type OrtVersionDiagnosticResponse,
  type OrtVersionDiagnosticSuccess,
} from './hitnetOrtComparisonHelper';
import {
  assertDevFixtureEnabled,
  HITNET_DEV_NON_JSEP_FIXTURE_SPEC,
} from './stereoDepthWebGpuDevFixture';
import {
  disparityToDepthMeters,
  extractDisparityFromOrtTensor,
  HITNET_M1_FULL_GRAPH_SHA256,
  inferOnnxStaticTensorMeta,
  type OnnxStaticInferenceResult,
} from './stereoDepthWorkerCore';

export const HITNET_RESIDUAL_PARITY_GATES = {
  expectedLength: 76800,
  expectedDims: [1, 240, 320, 1] as const,
  expectedDataType: 'float32' as const,
  maxAbsDiffMax: 1e-3,
  p99AbsDiffMax: 1e-4,
  meanAbsDiffMax: 1e-5,
} as const;

export const HITNET_RESIDUAL_SOURCE_CALIBRATION_DEFAULT = {
  presetId: 'free_nature_park',
  sourceEyeWidth: 480,
  sourceEyeHeight: 270,
  modelWidth: 320,
  modelHeight: 240,
  baselineMm: 65,
  focalLengthPx: 500,
  validDepthMinDisparityPx: 0.05,
} as const;

export interface ParsedOnnxNodeSummary {
  idx: number;
  name: string;
  opType: string;
  inputs: string[];
  outputs: string[];
  attrsSummary: string[];
}

export interface HitnetM1LadderTopology {
  nodes: ParsedOnnxNodeSummary[];
  initializerNames: Set<string>;
  graphInputNames: Set<string>;
  graphOutputNames: Set<string>;
  producerByOutput: Map<string, ParsedOnnxNodeSummary>;
  staticMeta: OnnxStaticInferenceResult;
  stemCommonNodes: ParsedOnnxNodeSummary[];
  stemMidTensor: string | null;
  stemPreSplitTensor: string | null;
  feSplitTensors: [string, string];
}

export interface FinalOutputValidationResult {
  ok: boolean;
  error: string | null;
  byName: Record<string, { dims: number[]; dataType: string; data: Float32Array }>;
}

export interface BitIdentityResult {
  bitIdentical: boolean;
  lengthMatch: boolean;
  mismatchCount: number;
  firstMismatchIndex: number | null;
  maxAbsDiff: number;
}

export interface TopDisparityResidualLocation {
  rank: number;
  flatIdx: number;
  y: number;
  x: number;
  rawRef: number;
  rawCand: number;
  rawAbsDiff: number;
  rawSignedDiff: number;
  srcRefPx: number | null;
  srcCandPx: number | null;
  srcAbsDiffPx: number | null;
  depthRefM: number | null;
  depthCandM: number | null;
  depthDiffM: number | null;
}

export interface DisparityAndDepthImpactSummary {
  frameLabel: string;
  modelWidth: number;
  modelHeight: number;
  sourceEyeWidth: number;
  sourceEyeHeight: number;
  scaleToSource: number;
  baselineMm: number;
  focalLengthPx: number;
  validDepthMinDisparityPx: number;
  primarySignBranch: {
    candNegate: boolean;
    refNegate: boolean;
    sameBranch: boolean;
    candNegSum: number;
    candPosSum: number;
    refNegSum: number;
    refPosSum: number;
  };
  primarySourceMaskChanges: {
    refValidCount: number;
    candValidCount: number;
    refValidToCandInvalid: number;
    refInvalidToCandValid: number;
    totalMaskFlips: number;
  };
  primaryDepthThresholdChanges: {
    refValidDepthCount: number;
    candValidDepthCount: number;
    refValidToCandNull: number;
    refNullToCandValid: number;
    totalDepthThresholdFlips: number;
    jointValidCount: number;
    maxAbsDepthDiffMeters: number;
    maxDepthDiffCoord: { y: number; x: number } | null;
    p99AbsDepthDiffMeters: number;
    meanAbsDepthDiffMeters: number;
  };
  rawSignedBias: {
    meanSignedDiff: number;
    minSignedDiff: number;
    maxSignedDiff: number;
    exactZeroDiffCount: number;
    positiveDiffCount: number;
    negativeDiffCount: number;
    gt1e5Count: number;
    gt1e4Count: number;
    gt5e4Count: number;
    gt1e3Count: number;
    gt5e3Count: number;
    gt1e2Count: number;
  };
  topPrimaryLocations: TopDisparityResidualLocation[];
  topSecondaryLocations: Array<{
    rank: number;
    flatIdx: number;
    y: number;
    x: number;
    rawRef: number;
    rawCand: number;
    rawAbsDiff: number;
    rawSignedDiff: number;
  }>;
}

export interface UntappedComparisonPairSummary {
  label: string;
  optimizerComparisonType: 'cross-ep-matched-opt' | 'within-ep-optimizer-delta';
  disclosedCrossEpDifferences: string[];
  primary: RawTensorParityStats & { passesStrictGate: boolean };
  secondary: RawTensorParityStats & { passesStrictGate: boolean };
  passesBothOutputs: boolean;
}

export interface LadderTapRecord {
  stepIndex: number;
  stageId: string;
  stageLabel: string;
  baseModelSha256: string;
  tappedModelSha256: string;
  tappedModelByteLength: number;
  tapOutputs: string[];
  stoppedBecauseFinalPerturbed: boolean;
  perturbationDetail: string | null;
  gpuModelSha256: string;
  wasmModelSha256: string;
  gpuFeedSha256: string;
  wasmFeedSha256: string;
  finalInvariance: {
    gpuPrimaryBitIdentical: BitIdentityResult;
    gpuSecondaryBitIdentical: BitIdentityResult;
    wasmPrimaryBitIdentical: BitIdentityResult;
    wasmSecondaryBitIdentical: BitIdentityResult;
  };
  tapComparisons: Array<{
    tensorName: string;
    dataType: string;
    dims: number[];
    stats: RawTensorParityStats;
    passesStrictGate: boolean;
    hasNonZeroDiff: boolean;
    p2bSplit?: ReturnType<typeof summarizeP2bChannelwiseParity> | null;
  }>;
}

export interface SuspectedOperatorIsolationSummary {
  branch: 'shared-stem' | 'primary' | 'secondary';
  previousCleanStageId: string;
  firstDivergingStageId: string;
  nodeIndex: number;
  nodeName: string;
  opType: string;
  attrsSummary: string[];
  outputTensorName: string;
  outputStats: RawTensorParityStats;
  immediateParentNames: string[];
  immediateParentRuns: LadderTapRecord[];
  caveat: string;
}

export interface ResidualLocalizationDiagnosticReport {
  ok: boolean;
  abortReason: string | null;
  pinnedOrtVersion: '1.30.0';
  runtimeAssetFamily: 'asyncify';
  m1ByteLength: number;
  m1Sha256: string;
  f0102FeedSha256: string;
  f0126FeedSha256: string;
  frames: Record<
    'F0102' | 'F0126',
    {
      feedSha256: string;
      arms: {
        webgpuBasic: OrtVersionDiagnosticSuccess;
        webgpuDisabled: OrtVersionDiagnosticSuccess;
        wasmBasic: OrtVersionDiagnosticSuccess;
        wasmDisabled: OrtVersionDiagnosticSuccess;
      };
      comparisons: {
        webgpuBasicVsWasmBasic: UntappedComparisonPairSummary;
        webgpuDisabledVsWasmDisabled: UntappedComparisonPairSummary;
        webgpuBasicVsDisabled: UntappedComparisonPairSummary;
        wasmBasicVsDisabled: UntappedComparisonPairSummary;
      };
      residualAndDepthBasic: DisparityAndDepthImpactSummary;
      residualAndDepthDisabled: DisparityAndDepthImpactSummary;
      nonTimingDispatchProfiles: {
        webgpuBasic: OrtVersionDiagnosticResponse | null;
        webgpuDisabled: OrtVersionDiagnosticResponse | null;
      };
    } | null
  >;
  localization: {
    executed: boolean;
    reasonSkipped: string | null;
    frame: 'F0102';
    selectedOptLevel: 'disabled' | 'basic';
    abortedByFinalPerturbation: boolean;
    perturbationStopMessage: string | null;
    earliestNonZeroTap: { stageId: string; tensorName: string; maxAbsDiff: number } | null;
    firstGateBreachingTap: { stageId: string; tensorName: string; maxAbsDiff: number; p99AbsDiff: number; meanAbsDiff: number } | null;
    coarseLadderRecords: LadderTapRecord[];
    bisectionRecords: LadderTapRecord[];
    suspectedOperator: SuspectedOperatorIsolationSummary | null;
  } | null;
}

export type ResidualDiagnosticWorkerRunner = (
  req: Omit<OrtVersionDiagnosticRequest, 'id'> & { timeoutMs?: number }
) => Promise<OrtVersionDiagnosticResponse>;

function readVarint(buf: Uint8Array, pos: number, end: number): [number, number] | null {
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

function readSignedVarint64(buf: Uint8Array, pos: number, end: number): [number, number] | null {
  let result = 0n;
  let shift = 0n;
  let c = pos;
  while (c < end && shift <= 63n) {
    const b = buf[c++];
    result |= BigInt(b & 0x7f) << shift;
    if ((b & 0x80) === 0) {
      const signed = BigInt.asIntN(64, result);
      return [Number(signed), c];
    }
    shift += 7n;
  }
  return null;
}

interface ProtoSlice {
  fn: number;
  wt: number;
  valStart: number;
  valEnd: number;
}

function parseProtoFieldsSimple(buf: Uint8Array, start: number, end: number): ProtoSlice[] | null {
  const out: ProtoSlice[] = [];
  let c = start;
  while (c < end) {
    const k = readVarint(buf, c, end);
    if (!k) return null;
    const fn = k[0] >>> 3;
    const wt = k[0] & 7;
    c = k[1];
    if (wt === 0) {
      const v = readVarint(buf, c, end);
      if (!v) return null;
      out.push({ fn, wt, valStart: c, valEnd: v[1] });
      c = v[1];
    } else if (wt === 2) {
      const l = readVarint(buf, c, end);
      if (!l) return null;
      const vs = l[1];
      const ve = vs + l[0];
      if (ve > end) return null;
      out.push({ fn, wt, valStart: vs, valEnd: ve });
      c = ve;
    } else if (wt === 5) {
      if (c + 4 > end) return null;
      out.push({ fn, wt, valStart: c, valEnd: c + 4 });
      c += 4;
    } else if (wt === 1) {
      if (c + 8 > end) return null;
      out.push({ fn, wt, valStart: c, valEnd: c + 8 });
      c += 8;
    } else {
      return null;
    }
  }
  return out;
}

function parseAttributeSummary(
  buf: Uint8Array,
  start: number,
  end: number,
  dec: TextDecoder
): string | null {
  const fields = parseProtoFieldsSimple(buf, start, end);
  if (!fields) return null;
  let attrName = '';
  let attrType = 0;
  let intVal: number | null = null;
  let floatVal: number | null = null;
  const intsVal: number[] = [];
  for (const f of fields) {
    if (f.fn === 1 && f.wt === 2) {
      attrName = dec.decode(buf.subarray(f.valStart, f.valEnd));
    } else if (f.fn === 20 && f.wt === 0) {
      const v = readVarint(buf, f.valStart, f.valEnd);
      if (v) attrType = v[0];
    } else if (f.fn === 3 && f.wt === 0) {
      const v = readSignedVarint64(buf, f.valStart, f.valEnd);
      if (v) intVal = v[0];
    } else if (f.fn === 2 && f.wt === 5 && f.valEnd - f.valStart === 4) {
      const dv = new DataView(buf.buffer, buf.byteOffset + f.valStart, 4);
      floatVal = dv.getFloat32(0, true);
    } else if (f.fn === 8 && f.wt === 0) {
      const v = readSignedVarint64(buf, f.valStart, f.valEnd);
      if (v) intsVal.push(v[0]);
    }
  }
  if (!attrName) return null;
  if (attrType === 2 && intVal !== null) return `${attrName}=${intVal}`;
  if (attrType === 1 && floatVal !== null) return `${attrName}=${floatVal}`;
  if (attrType === 7 && intsVal.length > 0) return `${attrName}=[${intsVal.join(',')}]`;
  return attrName;
}

export async function sha256HexBytes(bytes: Uint8Array): Promise<string> {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const digest = await crypto.subtle.digest('SHA-256', copy.buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

export function validateHitnetFinalOutputs(
  outputs: ReadonlyArray<{ name: string; dims: number[]; dataType: string; buffer: ArrayBuffer }>
): FinalOutputValidationResult {
  const expectedNames = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;
  if (outputs.length < expectedNames.length) {
    return {
      ok: false,
      error: `Expected at least ${expectedNames.length} outputs, got ${outputs.length}`,
      byName: {},
    };
  }
  const byName: Record<string, { dims: number[]; dataType: string; data: Float32Array }> = {};
  for (let i = 0; i < expectedNames.length; i++) {
    if (outputs[i]?.name !== expectedNames[i]) {
      return {
        ok: false,
        error: `Ordered output[${i}] mismatch: expected "${expectedNames[i]}", got "${String(outputs[i]?.name)}"`,
        byName: {},
      };
    }
  }
  for (const o of outputs) {
    if (o.name === expectedNames[0] || o.name === expectedNames[1]) {
      if (o.dataType !== HITNET_RESIDUAL_PARITY_GATES.expectedDataType) {
        return {
          ok: false,
          error: `Output "${o.name}" dtype mismatch: expected float32, got ${o.dataType}`,
          byName: {},
        };
      }
      if (JSON.stringify(o.dims) !== JSON.stringify(HITNET_RESIDUAL_PARITY_GATES.expectedDims)) {
        return {
          ok: false,
          error: `Output "${o.name}" shape mismatch: expected [1,240,320,1], got ${JSON.stringify(o.dims)}`,
          byName: {},
        };
      }
      if (o.buffer.byteLength !== HITNET_RESIDUAL_PARITY_GATES.expectedLength * 4) {
        return {
          ok: false,
          error: `Output "${o.name}" byteLength mismatch: expected ${HITNET_RESIDUAL_PARITY_GATES.expectedLength * 4}, got ${o.buffer.byteLength}`,
          byName: {},
        };
      }
      const f32 = new Float32Array(o.buffer);
      if (f32.length !== HITNET_RESIDUAL_PARITY_GATES.expectedLength) {
        return {
          ok: false,
          error: `Output "${o.name}" element length mismatch: expected 76800, got ${f32.length}`,
          byName: {},
        };
      }
      for (let i = 0; i < f32.length; i++) {
        if (!Number.isFinite(f32[i])) {
          return {
            ok: false,
            error: `Output "${o.name}" contains non-finite value at index ${i}: ${f32[i]}`,
            byName: {},
          };
        }
      }
      byName[o.name] = { dims: [...o.dims], dataType: o.dataType, data: f32 };
    }
  }
  return { ok: true, error: null, byName };
}

export function checkFloat32BitIdentity(a: Float32Array, b: Float32Array): BitIdentityResult {
  if (a.length !== b.length) {
    return {
      bitIdentical: false,
      lengthMatch: false,
      mismatchCount: Math.max(a.length, b.length),
      firstMismatchIndex: 0,
      maxAbsDiff: Infinity,
    };
  }
  const u32A = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const u32B = new Uint32Array(b.buffer, b.byteOffset, b.length);
  let mismatchCount = 0;
  let firstMismatchIndex: number | null = null;
  let maxAbsDiff = 0;
  for (let i = 0; i < a.length; i++) {
    if (u32A[i] !== u32B[i]) {
      mismatchCount++;
      if (firstMismatchIndex === null) firstMismatchIndex = i;
      const absD = Math.abs(a[i] - b[i]);
      if (Number.isFinite(absD) && absD > maxAbsDiff) {
        maxAbsDiff = absD;
      } else if (!Number.isFinite(absD)) {
        maxAbsDiff = Infinity;
      }
    }
  }
  return {
    bitIdentical: mismatchCount === 0,
    lengthMatch: true,
    mismatchCount,
    firstMismatchIndex,
    maxAbsDiff,
  };
}

export function passesStrictRawGate(stats: RawTensorParityStats): boolean {
  return (
    stats.length === HITNET_RESIDUAL_PARITY_GATES.expectedLength &&
    stats.finiteCount === HITNET_RESIDUAL_PARITY_GATES.expectedLength &&
    stats.nonFiniteMismatchCount === 0 &&
    stats.maxAbsDiff < HITNET_RESIDUAL_PARITY_GATES.maxAbsDiffMax &&
    stats.p99AbsDiff < HITNET_RESIDUAL_PARITY_GATES.p99AbsDiffMax &&
    stats.meanAbsDiff < HITNET_RESIDUAL_PARITY_GATES.meanAbsDiffMax
  );
}

function passesIntermediateTapGate(stats: RawTensorParityStats): boolean {
  return (
    stats.length > 0 &&
    stats.finiteCount === stats.length &&
    stats.nonFiniteMismatchCount === 0 &&
    stats.maxAbsDiff < HITNET_RESIDUAL_PARITY_GATES.maxAbsDiffMax &&
    stats.p99AbsDiff < HITNET_RESIDUAL_PARITY_GATES.p99AbsDiffMax &&
    stats.meanAbsDiff < HITNET_RESIDUAL_PARITY_GATES.meanAbsDiffMax
  );
}

export function summarizeUntappedPair(
  label: string,
  optimizerComparisonType: UntappedComparisonPairSummary['optimizerComparisonType'],
  candByName: Record<string, { dims: number[]; dataType: string; data: Float32Array }>,
  refByName: Record<string, { dims: number[]; dataType: string; data: Float32Array }>,
  disclosedCrossEpDifferences: string[] = []
): UntappedComparisonPairSummary {
  const [primaryName, secondaryName] = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;
  const pStats = summarizeRawTensorAndParity(candByName[primaryName].data, refByName[primaryName].data);
  const sStats = summarizeRawTensorAndParity(candByName[secondaryName].data, refByName[secondaryName].data);
  const pPass = passesStrictRawGate(pStats);
  const sPass = passesStrictRawGate(sStats);
  return {
    label,
    optimizerComparisonType,
    disclosedCrossEpDifferences,
    primary: { ...pStats, passesStrictGate: pPass },
    secondary: { ...sStats, passesStrictGate: sPass },
    passesBothOutputs: pPass && sPass,
  };
}

export function summarizeDisparityResidualAndSourceDepthImpact(
  frameLabel: string,
  candPrimary: Float32Array,
  refPrimary: Float32Array,
  candSecondary?: Float32Array | null,
  refSecondary?: Float32Array | null,
  opts?: {
    sourceEyeWidth?: number;
    sourceEyeHeight?: number;
    baselineMm?: number;
    focalLengthPx?: number;
    topK?: number;
  }
): DisparityAndDepthImpactSummary {
  const modelWidth = 320;
  const modelHeight = 240;
  const n = modelWidth * modelHeight;
  if (candPrimary.length !== n || refPrimary.length !== n) {
    throw new Error(`Expected primary length=${n}, got cand=${candPrimary.length} ref=${refPrimary.length}`);
  }
  const sourceEyeWidth = opts?.sourceEyeWidth ?? HITNET_RESIDUAL_SOURCE_CALIBRATION_DEFAULT.sourceEyeWidth;
  const sourceEyeHeight = opts?.sourceEyeHeight ?? HITNET_RESIDUAL_SOURCE_CALIBRATION_DEFAULT.sourceEyeHeight;
  const baselineMm = opts?.baselineMm ?? HITNET_RESIDUAL_SOURCE_CALIBRATION_DEFAULT.baselineMm;
  const focalLengthPx = opts?.focalLengthPx ?? HITNET_RESIDUAL_SOURCE_CALIBRATION_DEFAULT.focalLengthPx;
  const topK = Math.max(1, Math.min(20, opts?.topK ?? 8));

  const computeNegSums = (arr: Float32Array) => {
    let negSum = 0;
    let posSum = 0;
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (Number.isFinite(v)) {
        if (v < 0) negSum -= v;
        else posSum += v;
      }
    }
    return { negSum, posSum, negate: negSum > posSum * 1.1 };
  };

  const candSign = computeNegSums(candPrimary);
  const refSign = computeNegSums(refPrimary);
  const candExt = extractDisparityFromOrtTensor({ data: candPrimary, dims: [1, modelHeight, modelWidth, 1] });
  const refExt = extractDisparityFromOrtTensor({ data: refPrimary, dims: [1, modelHeight, modelWidth, 1] });
  const scaleToSource = sourceEyeWidth / refExt.width;

  let refValidCount = 0;
  let candValidCount = 0;
  let refValidToCandInvalid = 0;
  let refInvalidToCandValid = 0;

  let refValidDepthCount = 0;
  let candValidDepthCount = 0;
  let refValidToCandNull = 0;
  let refNullToCandValid = 0;
  let jointValidCount = 0;
  let maxAbsDepthDiffMeters = 0;
  let maxDepthDiffCoord: { y: number; x: number } | null = null;
  let sumAbsDepthDiff = 0;
  const jointDepthDiffs: number[] = [];

  let sumSignedDiff = 0;
  let minSignedDiff = Infinity;
  let maxSignedDiff = -Infinity;
  let exactZeroDiffCount = 0;
  let positiveDiffCount = 0;
  let negativeDiffCount = 0;
  let gt1e5Count = 0;
  let gt1e4Count = 0;
  let gt5e4Count = 0;
  let gt1e3Count = 0;
  let gt5e3Count = 0;
  let gt1e2Count = 0;

  const topPrimaryIndices: Array<{ idx: number; absDiff: number }> = [];

  for (let i = 0; i < n; i++) {
    const rRaw = refPrimary[i];
    const cRaw = candPrimary[i];
    const signedDiff = cRaw - rRaw;
    const absDiff = Math.abs(signedDiff);
    sumSignedDiff += signedDiff;
    if (signedDiff < minSignedDiff) minSignedDiff = signedDiff;
    if (signedDiff > maxSignedDiff) maxSignedDiff = signedDiff;
    if (signedDiff === 0) exactZeroDiffCount++;
    else if (signedDiff > 0) positiveDiffCount++;
    else negativeDiffCount++;

    if (absDiff > 1e-5) gt1e5Count++;
    if (absDiff > 1e-4) gt1e4Count++;
    if (absDiff > 5e-4) gt5e4Count++;
    if (absDiff > 1e-3) gt1e3Count++;
    if (absDiff > 5e-3) gt5e3Count++;
    if (absDiff > 1e-2) gt1e2Count++;

    if (
      topPrimaryIndices.length < topK ||
      absDiff > topPrimaryIndices[topPrimaryIndices.length - 1].absDiff
    ) {
      topPrimaryIndices.push({ idx: i, absDiff });
      topPrimaryIndices.sort((a, b) => b.absDiff - a.absDiff);
      if (topPrimaryIndices.length > topK) topPrimaryIndices.pop();
    }

    const rMaskValid = refExt.disparity[i] >= 0;
    const cMaskValid = candExt.disparity[i] >= 0;
    if (rMaskValid) refValidCount++;
    if (cMaskValid) candValidCount++;
    if (rMaskValid && !cMaskValid) refValidToCandInvalid++;
    if (!rMaskValid && cMaskValid) refInvalidToCandValid++;

    const rSrc = rMaskValid ? refExt.disparity[i] * scaleToSource : -1;
    const cSrc = cMaskValid ? candExt.disparity[i] * scaleToSource : -1;
    const rDepth = rSrc >= 0 ? disparityToDepthMeters(rSrc, baselineMm, focalLengthPx) : null;
    const cDepth = cSrc >= 0 ? disparityToDepthMeters(cSrc, baselineMm, focalLengthPx) : null;
    if (rDepth !== null) refValidDepthCount++;
    if (cDepth !== null) candValidDepthCount++;
    if (rDepth !== null && cDepth === null) refValidToCandNull++;
    if (rDepth === null && cDepth !== null) refNullToCandValid++;
    if (rDepth !== null && cDepth !== null) {
      jointValidCount++;
      const dZ = Math.abs(cDepth - rDepth);
      jointDepthDiffs.push(dZ);
      sumAbsDepthDiff += dZ;
      if (dZ > maxAbsDepthDiffMeters) {
        maxAbsDepthDiffMeters = dZ;
        maxDepthDiffCoord = { y: Math.floor(i / modelWidth), x: i % modelWidth };
      }
    }
  }

  let p99AbsDepthDiffMeters = 0;
  if (jointDepthDiffs.length > 0) {
    jointDepthDiffs.sort((a, b) => a - b);
    p99AbsDepthDiffMeters =
      jointDepthDiffs[Math.min(jointDepthDiffs.length - 1, Math.floor(0.99 * (jointDepthDiffs.length - 1)))];
  }

  const topPrimaryLocations: TopDisparityResidualLocation[] = topPrimaryIndices.map((entry, idx) => {
    const i = entry.idx;
    const y = Math.floor(i / modelWidth);
    const x = i % modelWidth;
    const rSrc = refExt.disparity[i] >= 0 ? refExt.disparity[i] * scaleToSource : null;
    const cSrc = candExt.disparity[i] >= 0 ? candExt.disparity[i] * scaleToSource : null;
    const dRef = rSrc !== null ? disparityToDepthMeters(rSrc, baselineMm, focalLengthPx) : null;
    const dCand = cSrc !== null ? disparityToDepthMeters(cSrc, baselineMm, focalLengthPx) : null;
    return {
      rank: idx + 1,
      flatIdx: i,
      y,
      x,
      rawRef: refPrimary[i],
      rawCand: candPrimary[i],
      rawAbsDiff: Math.abs(candPrimary[i] - refPrimary[i]),
      rawSignedDiff: candPrimary[i] - refPrimary[i],
      srcRefPx: rSrc,
      srcCandPx: cSrc,
      srcAbsDiffPx: rSrc !== null && cSrc !== null ? Math.abs(cSrc - rSrc) : null,
      depthRefM: dRef,
      depthCandM: dCand,
      depthDiffM: dRef !== null && dCand !== null ? dCand - dRef : null,
    };
  });

  const topSecondaryIndices: Array<{ idx: number; absDiff: number }> = [];
  if (candSecondary && refSecondary && candSecondary.length === n && refSecondary.length === n) {
    const secTopK = Math.min(4, topK);
    for (let i = 0; i < n; i++) {
      const absDiff = Math.abs(candSecondary[i] - refSecondary[i]);
      if (
        topSecondaryIndices.length < secTopK ||
        absDiff > topSecondaryIndices[topSecondaryIndices.length - 1].absDiff
      ) {
        topSecondaryIndices.push({ idx: i, absDiff });
        topSecondaryIndices.sort((a, b) => b.absDiff - a.absDiff);
        if (topSecondaryIndices.length > secTopK) topSecondaryIndices.pop();
      }
    }
  }

  const topSecondaryLocations = topSecondaryIndices.map((entry, idx) => {
    const i = entry.idx;
    return {
      rank: idx + 1,
      flatIdx: i,
      y: Math.floor(i / modelWidth),
      x: i % modelWidth,
      rawRef: refSecondary![i],
      rawCand: candSecondary![i],
      rawAbsDiff: entry.absDiff,
      rawSignedDiff: candSecondary![i] - refSecondary![i],
    };
  });

  return {
    frameLabel,
    modelWidth,
    modelHeight,
    sourceEyeWidth,
    sourceEyeHeight,
    scaleToSource,
    baselineMm,
    focalLengthPx,
    validDepthMinDisparityPx: HITNET_RESIDUAL_SOURCE_CALIBRATION_DEFAULT.validDepthMinDisparityPx,
    primarySignBranch: {
      candNegate: candSign.negate,
      refNegate: refSign.negate,
      sameBranch: candSign.negate === refSign.negate,
      candNegSum: candSign.negSum,
      candPosSum: candSign.posSum,
      refNegSum: refSign.negSum,
      refPosSum: refSign.posSum,
    },
    primarySourceMaskChanges: {
      refValidCount,
      candValidCount,
      refValidToCandInvalid,
      refInvalidToCandValid,
      totalMaskFlips: refValidToCandInvalid + refInvalidToCandValid,
    },
    primaryDepthThresholdChanges: {
      refValidDepthCount,
      candValidDepthCount,
      refValidToCandNull,
      refNullToCandValid,
      totalDepthThresholdFlips: refValidToCandNull + refNullToCandValid,
      jointValidCount,
      maxAbsDepthDiffMeters,
      maxDepthDiffCoord,
      p99AbsDepthDiffMeters,
      meanAbsDepthDiffMeters: jointValidCount > 0 ? sumAbsDepthDiff / jointValidCount : 0,
    },
    rawSignedBias: {
      meanSignedDiff: sumSignedDiff / n,
      minSignedDiff,
      maxSignedDiff,
      exactZeroDiffCount,
      positiveDiffCount,
      negativeDiffCount,
      gt1e5Count,
      gt1e4Count,
      gt5e4Count,
      gt1e3Count,
      gt5e3Count,
      gt1e2Count,
    },
    topPrimaryLocations,
    topSecondaryLocations,
  };
}

function isStaticFloat32TapCandidate(
  meta: OnnxStaticInferenceResult['tensorMeta'][string] | undefined,
  tensorName: string,
  initializerNames: Set<string>,
  graphInputNames: Set<string>,
  graphOutputNames: Set<string>
): boolean {
  if (!tensorName || !meta) return false;
  if (initializerNames.has(tensorName) || graphInputNames.has(tensorName) || graphOutputNames.has(tensorName)) {
    return false;
  }
  if (meta.elemType !== 1) return false;
  return meta.dims.length > 0 && meta.dims.every((d) => Number.isInteger(d) && d > 0);
}

export function inspectHitnetM1ResidualLadderTopology(m1Bytes: Uint8Array): HitnetM1LadderTopology {
  const staticMeta = inferOnnxStaticTensorMeta(m1Bytes);
  if (!staticMeta.valid) {
    throw new Error('M1 ONNX failed static tensor inference during residual ladder inspection');
  }
  const dec = new TextDecoder();
  const top = parseProtoFieldsSimple(m1Bytes, 0, m1Bytes.byteLength);
  if (!top) throw new Error('Invalid ModelProto');
  const gf = top.find((f) => f.fn === 7 && f.wt === 2);
  if (!gf) throw new Error('Missing GraphProto');
  const gFields = parseProtoFieldsSimple(m1Bytes, gf.valStart, gf.valEnd);
  if (!gFields) throw new Error('Invalid GraphProto');

  const nodes: ParsedOnnxNodeSummary[] = [];
  const initializerNames = new Set<string>();
  const graphInputNames = new Set<string>();
  const graphOutputNames = new Set<string>();
  const producerByOutput = new Map<string, ParsedOnnxNodeSummary>();

  for (const f of gFields) {
    if (f.fn === 1 && f.wt === 2) {
      const nf = parseProtoFieldsSimple(m1Bytes, f.valStart, f.valEnd);
      if (!nf) throw new Error('Invalid NodeProto');
      const inputs: string[] = [];
      const outputs: string[] = [];
      const attrsSummary: string[] = [];
      let name = '';
      let opType = '';
      for (const n of nf) {
        if (n.fn === 1 && n.wt === 2) {
          inputs.push(n.valEnd > n.valStart ? dec.decode(m1Bytes.subarray(n.valStart, n.valEnd)) : '');
        } else if (n.fn === 2 && n.wt === 2 && n.valEnd > n.valStart) {
          outputs.push(dec.decode(m1Bytes.subarray(n.valStart, n.valEnd)));
        } else if (n.fn === 3 && n.wt === 2 && n.valEnd > n.valStart) {
          name = dec.decode(m1Bytes.subarray(n.valStart, n.valEnd));
        } else if (n.fn === 4 && n.wt === 2 && n.valEnd > n.valStart) {
          opType = dec.decode(m1Bytes.subarray(n.valStart, n.valEnd));
        } else if (n.fn === 5 && n.wt === 2) {
          const summary = parseAttributeSummary(m1Bytes, n.valStart, n.valEnd, dec);
          if (summary) attrsSummary.push(summary);
        }
      }
      const node: ParsedOnnxNodeSummary = {
        idx: nodes.length,
        name,
        opType,
        inputs,
        outputs,
        attrsSummary,
      };
      nodes.push(node);
      for (const o of outputs) {
        producerByOutput.set(o, node);
      }
    } else if (f.fn === 5 && f.wt === 2) {
      const tf = parseProtoFieldsSimple(m1Bytes, f.valStart, f.valEnd);
      const nf = tf?.find((x) => x.fn === 8 && x.wt === 2 && x.valEnd > x.valStart);
      if (nf) initializerNames.add(dec.decode(m1Bytes.subarray(nf.valStart, nf.valEnd)));
    } else if (f.fn === 11 && f.wt === 2) {
      const vf = parseProtoFieldsSimple(m1Bytes, f.valStart, f.valEnd);
      const nf = vf?.find((x) => x.fn === 1 && x.wt === 2 && x.valEnd > x.valStart);
      if (nf) graphInputNames.add(dec.decode(m1Bytes.subarray(nf.valStart, nf.valEnd)));
    } else if (f.fn === 12 && f.wt === 2) {
      const vf = parseProtoFieldsSimple(m1Bytes, f.valStart, f.valEnd);
      const nf = vf?.find((x) => x.fn === 1 && x.wt === 2 && x.valEnd > x.valStart);
      if (nf) graphOutputNames.add(dec.decode(m1Bytes.subarray(nf.valStart, nf.valEnd)));
    }
  }

  const collectAncestorIdx = (targetTensors: string[]): Set<number> => {
    const visitedTensors = new Set<string>();
    const nodeIdxSet = new Set<number>();
    const queue = [...targetTensors];
    while (queue.length > 0) {
      const t = queue.pop()!;
      if (!t || visitedTensors.has(t)) continue;
      visitedTensors.add(t);
      const prod = producerByOutput.get(t);
      if (!prod) continue;
      nodeIdxSet.add(prod.idx);
      for (const inp of prod.inputs) {
        if (inp && !visitedTensors.has(inp)) {
          queue.push(inp);
        }
      }
    }
    return nodeIdxSet;
  };

  const feSplitTensors: [string, string] = ['fe_shared_3', 'fe_shared_31'];
  const ancL = collectAncestorIdx(['fe_shared_3', 'fe_shared_4', 'fe_shared_5']);
  const ancR = collectAncestorIdx(['fe_shared_31', 'fe_shared_41', 'fe_shared_51']);
  const stemCommonNodes = nodes.filter((n) => {
    if (!ancL.has(n.idx) || !ancR.has(n.idx)) return false;
    const out0 = n.outputs[0];
    return isStaticFloat32TapCandidate(
      out0 ? staticMeta.tensorMeta[out0] : undefined,
      out0 || '',
      initializerNames,
      graphInputNames,
      graphOutputNames
    );
  });

  const stemPreSplitTensor =
    stemCommonNodes.length > 0 ? stemCommonNodes[stemCommonNodes.length - 1].outputs[0] : null;
  const stemMidTensor =
    stemCommonNodes.length >= 2
      ? stemCommonNodes[Math.floor((stemCommonNodes.length - 1) / 2)].outputs[0]
      : null;

  return {
    nodes,
    initializerNames,
    graphInputNames,
    graphOutputNames,
    producerByOutput,
    staticMeta,
    stemCommonNodes,
    stemMidTensor,
    stemPreSplitTensor,
    feSplitTensors,
  };
}

export function getOrderedIntervalNodesForBisection(
  topology: HitnetM1LadderTopology,
  cleanBoundaryTensors: readonly string[],
  targetOutputTensor: string
): ParsedOnnxNodeSummary[] {
  const collectAnc = (starts: readonly string[]): Set<number> => {
    const visited = new Set<string>();
    const out = new Set<number>();
    const q = [...starts];
    while (q.length > 0) {
      const t = q.pop()!;
      if (!t || visited.has(t)) continue;
      visited.add(t);
      const p = topology.producerByOutput.get(t);
      if (!p) continue;
      out.add(p.idx);
      for (const inp of p.inputs) {
        if (inp && !visited.has(inp)) q.push(inp);
      }
    }
    return out;
  };

  const targetAnc = collectAnc([targetOutputTensor]);
  const cleanAnc = collectAnc(cleanBoundaryTensors);

  return topology.nodes.filter((n) => {
    if (!targetAnc.has(n.idx) || cleanAnc.has(n.idx)) return false;
    if (n.opType === 'Constant' || n.opType === 'Shape') return false;
    const out0 = n.outputs[0];
    if (!out0 || out0 === targetOutputTensor) return false;
    return isStaticFloat32TapCandidate(
      topology.staticMeta.tensorMeta[out0],
      out0,
      topology.initializerNames,
      topology.graphInputNames,
      topology.graphOutputNames
    );
  });
}

export function getImmediateParentTapGroups(
  topology: HitnetM1LadderTopology,
  suspectNode: ParsedOnnxNodeSummary
): string[][] {
  const parents = Array.from(
    new Set(
      suspectNode.inputs.filter((inp) =>
        isStaticFloat32TapCandidate(
          topology.staticMeta.tensorMeta[inp],
          inp,
          topology.initializerNames,
          topology.graphInputNames,
          topology.graphOutputNames
        )
      )
    )
  );
  const groups: string[][] = [];
  for (let i = 0; i < parents.length; i += 2) {
    groups.push(parents.slice(i, i + 2));
  }
  return groups;
}

export function createBrowserOrt130DiagnosticRunner(timeoutDefaultMs = 120000): ResidualDiagnosticWorkerRunner {
  return async (req) => {
    const timeoutMs = Math.max(5000, Math.trunc(req.timeoutMs ?? timeoutDefaultMs));
    const worker = new Worker('/src/services/hitnetOrt130DiagnosticWorker.ts', { type: 'module' });
    const reqId = 1;
    let timerId: ReturnType<typeof setTimeout> | null = null;
    try {
      const modelBuffer = req.modelBuffer.slice(0);
      const feedBuffer = req.feedBuffer.slice(0);
      const extraFeeds = req.extraFeeds?.map((ef) => ({
        name: ef.name,
        dims: [...ef.dims],
        buffer: ef.buffer.slice(0),
        ...(ef.dataType ? { dataType: ef.dataType } : {}),
      }));
      const res = await new Promise<OrtVersionDiagnosticResponse>((resolve) => {
        let settled = false;
        const finish = (payload: OrtVersionDiagnosticResponse) => {
          if (settled) return;
          settled = true;
          if (timerId !== null) clearTimeout(timerId);
          worker.removeEventListener('message', onMsg);
          worker.removeEventListener('error', onErr);
          resolve(payload);
        };
        const onMsg = (ev: MessageEvent<OrtVersionDiagnosticResponse>) => {
          if (ev.data?.id !== reqId) return;
          finish(ev.data);
        };
        const onErr = (ev: ErrorEvent) => {
          finish({
            id: reqId,
            ok: false,
            stage: 'init',
            message: ev?.message || 'Dedicated ORT 1.30.0 diagnostic worker error',
            pinnedOrtVersion: '1.30.0',
            runtimeAssetFamily: 'asyncify',
            entryModuleUrl: 'worker-error',
            wasmPaths: 'worker-error',
            ortVersions: null,
            loadedOrtResources: [],
            foreignOrtResources: [],
            loadedAssetProvenance: [],
            provider: req.provider,
            configuredProviders: [req.provider],
            fallbackBehavior: 'no-fallback-explicit-error',
            graphOptimizationLevel: req.graphOptimizationLevel,
          });
        };
        timerId = setTimeout(() => {
          try {
            worker.terminate();
          } catch {
            // Ignore worker terminate error on timeout
          }
          finish({
            id: reqId,
            ok: false,
            stage: 'timeout',
            message: `Worker timed out after ${timeoutMs}ms (1.30.0 asyncify ${req.provider} opt=${req.graphOptimizationLevel})`,
            pinnedOrtVersion: '1.30.0',
            runtimeAssetFamily: 'asyncify',
            entryModuleUrl: 'timeout',
            wasmPaths: 'timeout',
            ortVersions: null,
            loadedOrtResources: [],
            foreignOrtResources: [],
            loadedAssetProvenance: [],
            provider: req.provider,
            configuredProviders: [req.provider],
            fallbackBehavior: 'no-fallback-explicit-error',
            graphOptimizationLevel: req.graphOptimizationLevel,
          });
        }, timeoutMs);
        worker.addEventListener('message', onMsg);
        worker.addEventListener('error', onErr);
        const transfers: Transferable[] = [
          modelBuffer,
          feedBuffer,
          ...(extraFeeds ? extraFeeds.map((ef) => ef.buffer) : []),
        ];
        worker.postMessage(
          {
            ...req,
            id: reqId,
            expectedOrtVersion: '1.30.0',
            runtimeAssetFamily: 'asyncify' as OrtDiagnosticRuntimeFamily,
            modelBuffer,
            feedBuffer,
            ...(extraFeeds ? { extraFeeds } : {}),
          },
          transfers
        );
      });
      return res;
    } finally {
      if (timerId !== null) clearTimeout(timerId);
      try {
        worker.terminate();
      } catch {
        // Ignore worker terminate error
      }
    }
  };
}

export async function runHitnetResidualLocalizationDiagnostic(params: {
  rawModelBytes?: Uint8Array;
  m1Bytes?: Uint8Array;
  f0102FeedFloat32: Float32Array;
  f0126FeedFloat32?: Float32Array | null;
  runWorker?: ResidualDiagnosticWorkerRunner;
  disableCpuEpFallback?: boolean;
  isDev?: boolean;
  onLog?: (line: string) => void;
}): Promise<ResidualLocalizationDiagnosticReport> {
  assertDevFixtureEnabled(params.isDev);
  const log = params.onLog ?? (() => {});
  const runWorker = params.runWorker ?? createBrowserOrt130DiagnosticRunner();
  const disableCpuEpFallback = Boolean(params.disableCpuEpFallback);

  let exactM1Bytes: Uint8Array;
  if (params.m1Bytes) {
    exactM1Bytes = params.m1Bytes;
  } else if (params.rawModelBytes) {
    const rawSha = await sha256HexBytes(params.rawModelBytes);
    if (rawSha !== HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedRawSha256) {
      throw new Error(
        `Raw HITNet model SHA mismatch: expected=${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedRawSha256}, actual=${rawSha}`
      );
    }
    exactM1Bytes = prepareHitnetM1FullGraphBytes(params.rawModelBytes, {
      rawModelSha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedRawSha256,
    }).bytes;
  } else {
    throw new Error('Either m1Bytes or rawModelBytes must be provided.');
  }

  const m1Sha = await sha256HexBytes(exactM1Bytes);
  if (
    exactM1Bytes.byteLength !== HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength ||
    m1Sha !== HITNET_M1_FULL_GRAPH_SHA256
  ) {
    throw new Error(
      `Exact M1 verification failed: bytes=${exactM1Bytes.byteLength} (expected ${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength}), sha=${m1Sha} (expected ${HITNET_M1_FULL_GRAPH_SHA256})`
    );
  }

  const f0102FeedBytes = new Uint8Array(
    params.f0102FeedFloat32.buffer,
    params.f0102FeedFloat32.byteOffset,
    params.f0102FeedFloat32.byteLength
  );
  const f0102Sha = await sha256HexBytes(f0102FeedBytes);
  if (f0102Sha !== HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256) {
    throw new Error(
      `Frozen F0102 feed SHA mismatch: expected=${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256}, actual=${f0102Sha}`
    );
  }

  let f0126Sha: string = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256;
  if (params.f0126FeedFloat32) {
    const f0126FeedBytes = new Uint8Array(
      params.f0126FeedFloat32.buffer,
      params.f0126FeedFloat32.byteOffset,
      params.f0126FeedFloat32.byteLength
    );
    f0126Sha = await sha256HexBytes(f0126FeedBytes);
    if (f0126Sha !== HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256) {
      throw new Error(
        `Frozen F0126 feed SHA mismatch: expected=${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256}, actual=${f0126Sha}`
      );
    }
  }

  const asyncifyProfile = HITNET_ORT_130_FAMILY_PROFILES.asyncify;
  const crossEpDiffDisclosure = [
    'executionProviders: ["webgpu"] vs ["wasm"] (provider selection; no CPU EP fallback in diagnostic worker)',
    'sessionOptions.enableCpuMemArena & enableMemPattern: webgpu=true vs wasm=false (within-EP optimizer arms hold these identically)',
    'gpuValidationErrors / adapter: captured on WebGPU only; WASM executes CPU SIMD in ort-wasm-simd-threaded.asyncify.wasm',
    `sharedRuntimeFamily: ${asyncifyProfile.entryFile} + ${asyncifyProfile.emscriptenMjsFile}/${asyncifyProfile.wasmFile}`,
  ];

  log(
    `=== [DEV-ONLY] HITNet Non-JSEP ORT 1.30.0 Residual Localization Diagnostic ===\n` +
    `M1: bytes=${exactM1Bytes.byteLength} sha256=${m1Sha}\n` +
    `F0102: feedSha256=${f0102Sha} | F0126: feedSha256=${f0126Sha}\n` +
    `Runtime: onnxruntime-web@1.30.0/webgpu (non-JSEP ${asyncifyProfile.entryFile} + ${asyncifyProfile.wasmFile})\n` +
    `Gates: float32 [1,240,320,1], finite=76800/76800, nonFiniteMismatch=0, max<1e-3, p99<1e-4, mean<1e-5\n` +
    `Cross-EP Disclosures:\n  - ${crossEpDiffDisclosure.join('\n  - ')}`
  );

  const runVerifiedUntappedArm = async (
    label: string,
    modelBytes: Uint8Array,
    expectedModelSha: string,
    feedFloat32: Float32Array,
    expectedFeedSha: string,
    provider: 'webgpu' | 'wasm',
    optLevel: 'basic' | 'disabled',
    opts?: { warmRuns?: number; enableWebGpuProfiling?: boolean }
  ): Promise<{
    res: OrtVersionDiagnosticSuccess;
    validated: FinalOutputValidationResult;
  }> => {
    const rawRes = await runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: modelBytes.slice().buffer,
      feedBuffer: feedFloat32.slice().buffer,
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      provider,
      graphOptimizationLevel: optLevel,
      warmRuns: opts?.warmRuns ?? (provider === 'webgpu' ? 1 : 0),
      compareColdVsWarmRepeat: provider === 'webgpu' && (opts?.warmRuns ?? 1) >= 1,
      enableWebGpuProfiling: Boolean(opts?.enableWebGpuProfiling) && provider === 'webgpu',
      disableCpuEpFallback: disableCpuEpFallback && provider === 'webgpu',
      captureGpuValidationErrors: provider === 'webgpu',
    });

    if (!rawRes.ok) {
      const f = rawRes as OrtVersionDiagnosticFailure;
      throw new Error(`${label} FAILED at stage="${f.stage}": ${f.message}`);
    }
    if (
      rawRes.runtimeAssetFamily !== 'asyncify' ||
      rawRes.foreignOrtResources.length > 0 ||
      rawRes.loadedOrtResources.some((u) => u.includes('.jsep.') || u.includes('ort.bundle.'))
    ) {
      throw new Error(
        `${label} violated non-JSEP resource guard: family=${rawRes.runtimeAssetFamily}, foreign=${JSON.stringify(rawRes.foreignOrtResources)}, loaded=${JSON.stringify(rawRes.loadedOrtResources)}`
      );
    }
    if (rawRes.modelSha256 !== expectedModelSha || rawRes.modelByteLength !== modelBytes.byteLength) {
      throw new Error(
        `${label} session model mismatch: got bytes=${rawRes.modelByteLength} sha=${rawRes.modelSha256}, expected bytes=${modelBytes.byteLength} sha=${expectedModelSha}`
      );
    }
    if (rawRes.feedSha256 !== expectedFeedSha) {
      throw new Error(`${label} feed SHA mismatch: got=${rawRes.feedSha256}, expected=${expectedFeedSha}`);
    }
    if (
      rawRes.configuredProviders.length !== 1 ||
      rawRes.configuredProviders[0] !== provider
    ) {
      throw new Error(
        `${label} configuredProviders mismatch: expected ["${provider}"], got ${JSON.stringify(rawRes.configuredProviders)}`
      );
    }
    const validated = validateHitnetFinalOutputs(rawRes.outputs);
    if (!validated.ok) {
      throw new Error(`${label} output validation failed: ${validated.error}`);
    }
    if (!opts?.enableWebGpuProfiling) {
      log(
        `  [${label}] OK (non-JSEP 1.30.0 ${provider} opt=${optLevel}, modelSha=${rawRes.modelSha256.slice(0, 12)}..., feedSha=${rawRes.feedSha256.slice(0, 12)}..., createMs=${rawRes.createMs.toFixed(1)}, coldMs=${rawRes.coldRunMs.toFixed(1)}, warmMs=${JSON.stringify(rawRes.warmRunMs.map((x) => Number(x.toFixed(1))))}, gpuValErrs=${rawRes.gpuValidationErrors ? rawRes.gpuValidationErrors.length : 'n/a'})`
      );
    }
    return { res: rawRes, validated };
  };

  const formatParityLine = (p: UntappedComparisonPairSummary) =>
    `  [${p.label}] ${p.passesBothOutputs ? 'PASS' : 'FAIL'}\n` +
    `    primary  : max=${p.primary.maxAbsDiff.toExponential(3)} p99=${p.primary.p99AbsDiff.toExponential(3)} mean=${p.primary.meanAbsDiff.toExponential(3)} rms=${p.primary.rmsDiff.toExponential(3)} finite=${p.primary.finiteCount}/76800 gate=${p.primary.passesStrictGate ? 'PASS' : 'FAIL'}\n` +
    `    secondary: max=${p.secondary.maxAbsDiff.toExponential(3)} p99=${p.secondary.p99AbsDiff.toExponential(3)} mean=${p.secondary.meanAbsDiff.toExponential(3)} rms=${p.secondary.rmsDiff.toExponential(3)} finite=${p.secondary.finiteCount}/76800 gate=${p.secondary.passesStrictGate ? 'PASS' : 'FAIL'}`;

  const runFrameSuite = async (
    frameName: 'F0102' | 'F0126',
    feedFloat32: Float32Array,
    feedSha: string
  ) => {
    log(`\n--- ${frameName} Untapped 2x2 Optimizer Control (Non-JSEP 1.30.0 M1, profiling=OFF for timing) ---`);
    const wgBasic = await runVerifiedUntappedArm(
      `${frameName} WebGPU opt=basic`,
      exactM1Bytes,
      m1Sha,
      feedFloat32,
      feedSha,
      'webgpu',
      'basic',
      { warmRuns: 1, enableWebGpuProfiling: false }
    );
    const wgDis = await runVerifiedUntappedArm(
      `${frameName} WebGPU opt=disabled`,
      exactM1Bytes,
      m1Sha,
      feedFloat32,
      feedSha,
      'webgpu',
      'disabled',
      { warmRuns: 1, enableWebGpuProfiling: false }
    );
    const wasmBasic = await runVerifiedUntappedArm(
      `${frameName} WASM opt=basic`,
      exactM1Bytes,
      m1Sha,
      feedFloat32,
      feedSha,
      'wasm',
      'basic',
      { warmRuns: 0, enableWebGpuProfiling: false }
    );
    const wasmDis = await runVerifiedUntappedArm(
      `${frameName} WASM opt=disabled`,
      exactM1Bytes,
      m1Sha,
      feedFloat32,
      feedSha,
      'wasm',
      'disabled',
      { warmRuns: 0, enableWebGpuProfiling: false }
    );

    const webgpuBasicVsWasmBasic = summarizeUntappedPair(
      `${frameName} WebGPU(basic) vs WASM(basic)`,
      'cross-ep-matched-opt',
      wgBasic.validated.byName,
      wasmBasic.validated.byName,
      crossEpDiffDisclosure
    );
    const webgpuDisabledVsWasmDisabled = summarizeUntappedPair(
      `${frameName} WebGPU(disabled) vs WASM(disabled)`,
      'cross-ep-matched-opt',
      wgDis.validated.byName,
      wasmDis.validated.byName,
      crossEpDiffDisclosure
    );
    const webgpuBasicVsDisabled = summarizeUntappedPair(
      `${frameName} WebGPU(basic) vs WebGPU(disabled) [Within-EP]`,
      'within-ep-optimizer-delta',
      wgBasic.validated.byName,
      wgDis.validated.byName
    );
    const wasmBasicVsDisabled = summarizeUntappedPair(
      `${frameName} WASM(basic) vs WASM(disabled) [Within-EP]`,
      'within-ep-optimizer-delta',
      wasmBasic.validated.byName,
      wasmDis.validated.byName
    );

    log(formatParityLine(webgpuBasicVsWasmBasic));
    log(formatParityLine(webgpuDisabledVsWasmDisabled));
    log(formatParityLine(webgpuBasicVsDisabled));
    log(formatParityLine(wasmBasicVsDisabled));

    const [pOut, sOut] = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;
    const residualAndDepthBasic = summarizeDisparityResidualAndSourceDepthImpact(
      `${frameName}(opt=basic)`,
      wgBasic.validated.byName[pOut].data,
      wasmBasic.validated.byName[pOut].data,
      wgBasic.validated.byName[sOut].data,
      wasmBasic.validated.byName[sOut].data
    );
    const residualAndDepthDisabled = summarizeDisparityResidualAndSourceDepthImpact(
      `${frameName}(opt=disabled)`,
      wgDis.validated.byName[pOut].data,
      wasmDis.validated.byName[pOut].data,
      wgDis.validated.byName[sOut].data,
      wasmDis.validated.byName[sOut].data
    );

    const logDepthImpact = (imp: DisparityAndDepthImpactSummary) => {
      log(
        `  [${imp.frameLabel} Source-Space & Metric Depth Impact (480x270 eye, scale=${imp.scaleToSource.toFixed(2)}, B=${imp.baselineMm}mm, f=${imp.focalLengthPx}px, depthCutoff>${imp.validDepthMinDisparityPx}px)]\n` +
        `    signBranch: same=${imp.primarySignBranch.sameBranch} (GPU negate=${imp.primarySignBranch.candNegate}, WASM negate=${imp.primarySignBranch.refNegate}) | sourceMaskFlips=${imp.primarySourceMaskChanges.totalMaskFlips} (valid->invalid=${imp.primarySourceMaskChanges.refValidToCandInvalid}, invalid->valid=${imp.primarySourceMaskChanges.refInvalidToCandValid})\n` +
        `    depthCutoffFlips=${imp.primaryDepthThresholdChanges.totalDepthThresholdFlips} | maxAbsDepthDiff=${imp.primaryDepthThresholdChanges.maxAbsDepthDiffMeters.toFixed(5)}m @ (${imp.primaryDepthThresholdChanges.maxDepthDiffCoord?.y ?? '-'},${imp.primaryDepthThresholdChanges.maxDepthDiffCoord?.x ?? '-'}) | p99DepthDiff=${imp.primaryDepthThresholdChanges.p99AbsDepthDiffMeters.toFixed(5)}m meanDepthDiff=${imp.primaryDepthThresholdChanges.meanAbsDepthDiffMeters.toFixed(5)}m\n` +
        `    signedBias: mean=${imp.rawSignedBias.meanSignedDiff.toExponential(3)} pos=${imp.rawSignedBias.positiveDiffCount} neg=${imp.rawSignedBias.negativeDiffCount} zero=${imp.rawSignedBias.exactZeroDiffCount} | >1e-3:${imp.rawSignedBias.gt1e3Count} >5e-3:${imp.rawSignedBias.gt5e3Count} >1e-2:${imp.rawSignedBias.gt1e2Count}\n` +
        `    topPrimaryResiduals=${JSON.stringify(imp.topPrimaryLocations.slice(0, 5).map((t) => ({
          y: t.y,
          x: t.x,
          ref: Number(t.rawRef.toFixed(5)),
          gpu: Number(t.rawCand.toFixed(5)),
          dRaw: Number(t.rawSignedDiff.toExponential(3)),
          dSrcPx: t.srcAbsDiffPx !== null ? Number(t.srcAbsDiffPx.toFixed(5)) : null,
          zRefM: t.depthRefM !== null ? Number(t.depthRefM.toFixed(4)) : null,
          dZM: t.depthDiffM !== null ? Number(t.depthDiffM.toExponential(3)) : null,
        })))}`
      );
    };
    logDepthImpact(residualAndDepthBasic);
    logDepthImpact(residualAndDepthDisabled);

    // Separate NON-TIMING dispatch observation runs
    log(`  [${frameName} Separate NON-TIMING WebGPU Dispatch Profile Runs (warmRuns=0, profiling=ON)]`);
    let profBasic: OrtVersionDiagnosticResponse | null = null;
    let profDisabled: OrtVersionDiagnosticResponse | null = null;
    for (const optLevel of ['basic', 'disabled'] as const) {
      const pRes = await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: exactM1Bytes.slice().buffer,
        feedBuffer: feedFloat32.slice().buffer,
        feedName: 'input',
        feedDims: [1, 6, 240, 320],
        provider: 'webgpu',
        graphOptimizationLevel: optLevel,
        warmRuns: 0,
        enableWebGpuProfiling: true,
        disableCpuEpFallback,
        captureGpuValidationErrors: true,
      });
      if (optLevel === 'basic') profBasic = pRes;
      else profDisabled = pRes;
      log(
        `    [NON-TIMING Profile ${frameName} WebGPU opt=${optLevel}] ${JSON.stringify(
          pRes.ok
            ? {
                status: 'OK',
                configuredProviders: pRes.configuredProviders,
                gpuValidationErrors: pRes.gpuValidationErrors,
                dispatch: pRes.gpuKernelProfileSummary,
              }
            : { status: `FAILED(${pRes.stage})`, error: pRes.message }
        )}`
      );
    }

    return {
      feedSha256: feedSha,
      arms: {
        webgpuBasic: wgBasic.res,
        webgpuDisabled: wgDis.res,
        wasmBasic: wasmBasic.res,
        wasmDisabled: wasmDis.res,
      },
      comparisons: {
        webgpuBasicVsWasmBasic,
        webgpuDisabledVsWasmDisabled,
        webgpuBasicVsDisabled,
        wasmBasicVsDisabled,
      },
      residualAndDepthBasic,
      residualAndDepthDisabled,
      nonTimingDispatchProfiles: {
        webgpuBasic: profBasic,
        webgpuDisabled: profDisabled,
      },
    };
  };

  const f0102Data = await runFrameSuite('F0102', params.f0102FeedFloat32, f0102Sha);
  const f0126Data = params.f0126FeedFloat32
    ? await runFrameSuite('F0126', params.f0126FeedFloat32, f0126Sha)
    : null;

  // Bounded serial tap localization on F0102 if still divergent
  const basicDiverges = !f0102Data.comparisons.webgpuBasicVsWasmBasic.passesBothOutputs;
  const disabledDiverges = !f0102Data.comparisons.webgpuDisabledVsWasmDisabled.passesBothOutputs;

  let localization: ResidualLocalizationDiagnosticReport['localization'];
  if (!basicDiverges && !disabledDiverges) {
    localization = {
      executed: false,
      reasonSkipped: 'Both WebGPU basic and disabled passed strict raw parity gates on F0102.',
      frame: 'F0102',
      selectedOptLevel: 'disabled',
      abortedByFinalPerturbation: false,
      perturbationStopMessage: null,
      earliestNonZeroTap: null,
      firstGateBreachingTap: null,
      coarseLadderRecords: [],
      bisectionRecords: [],
      suspectedOperator: null,
    };
  } else {
    // Prefer opt='disabled' when opt='disabled' also diverges, because ONNX node/tensor names remain unfused 1:1
    const selectedOptLevel: 'disabled' | 'basic' = disabledDiverges ? 'disabled' : 'basic';
    const untappedGpuCtrl =
      selectedOptLevel === 'disabled'
        ? f0102Data.arms.webgpuDisabled
        : f0102Data.arms.webgpuBasic;
    const untappedWasmCtrl =
      selectedOptLevel === 'disabled'
        ? f0102Data.arms.wasmDisabled
        : f0102Data.arms.wasmBasic;
    const untappedGpuByName = validateHitnetFinalOutputs(untappedGpuCtrl.outputs).byName;
    const untappedWasmByName = validateHitnetFinalOutputs(untappedWasmCtrl.outputs).byName;

    const topology = inspectHitnetM1ResidualLadderTopology(exactM1Bytes);
    const coarseLadderRecords: LadderTapRecord[] = [];
    const bisectionRecords: LadderTapRecord[] = [];
    let stepCounter = 0;
    let abortedByFinalPerturbation = false;
    let perturbationStopMessage: string | null = null;
    const trackers: {
      earliestNonZeroTap: { stageId: string; tensorName: string; maxAbsDiff: number } | null;
      firstGateBreachingTap: {
        stageId: string;
        tensorName: string;
        maxAbsDiff: number;
        p99AbsDiff: number;
        meanAbsDiff: number;
      } | null;
    } = {
      earliestNonZeroTap: null,
      firstGateBreachingTap: null,
    };

    log(
      `\n--- F0102 Bounded Serial Tap Ladder (opt=${selectedOptLevel}, Non-JSEP 1.30.0, 1 tap pair <=2 outputs/run, bit-identical final invariance required) ---`
    );

    const runSingleTapStep = async (
      stageId: string,
      stageLabel: string,
      tapOutputs: string[],
      targetLog: LadderTapRecord[]
    ): Promise<LadderTapRecord> => {
      stepCounter++;
      if (tapOutputs.length === 0 || tapOutputs.length > 2) {
        throw new Error(`Invalid tapOutputs length (${tapOutputs.length}) for stage "${stageId}"`);
      }
      const tapped = appendOnnxGraphOutputs(exactM1Bytes, tapOutputs, { maxExtraOutputs: 2 });
      const tappedSha = await sha256HexBytes(tapped.bytes);
      if (tappedSha === m1Sha) {
        throw new Error(`Tapped model SHA unexpectedly equals base M1 SHA at stage "${stageId}"`);
      }

      const gpuTap = await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: tapped.bytes.slice().buffer,
        feedBuffer: params.f0102FeedFloat32.slice().buffer,
        feedName: 'input',
        feedDims: [1, 6, 240, 320],
        provider: 'webgpu',
        graphOptimizationLevel: selectedOptLevel,
        warmRuns: 0,
        enableWebGpuProfiling: false,
        disableCpuEpFallback,
        captureGpuValidationErrors: true,
      });
      const wasmTap = await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: tapped.bytes.slice().buffer,
        feedBuffer: params.f0102FeedFloat32.slice().buffer,
        feedName: 'input',
        feedDims: [1, 6, 240, 320],
        provider: 'wasm',
        graphOptimizationLevel: selectedOptLevel,
        warmRuns: 0,
        enableWebGpuProfiling: false,
        disableCpuEpFallback: false,
        captureGpuValidationErrors: false,
      });

      if (!gpuTap.ok || !wasmTap.ok) {
        throw new Error(
          `Tap step "${stageId}" failed: gpu=${gpuTap.ok ? 'OK' : gpuTap.message}, wasm=${wasmTap.ok ? 'OK' : wasmTap.message}`
        );
      }
      if (
        gpuTap.modelSha256 !== tappedSha ||
        wasmTap.modelSha256 !== tappedSha ||
        gpuTap.modelByteLength !== tapped.bytes.byteLength ||
        wasmTap.modelByteLength !== tapped.bytes.byteLength
      ) {
        throw new Error(
          `Tapped model SHA/byte mismatch at "${stageId}": expected=${tappedSha}(${tapped.bytes.byteLength}B), gpu=${gpuTap.modelSha256}(${gpuTap.modelByteLength}B), wasm=${wasmTap.modelSha256}(${wasmTap.modelByteLength}B)`
        );
      }
      if (gpuTap.feedSha256 !== f0102Sha || wasmTap.feedSha256 !== f0102Sha) {
        throw new Error(
          `Feed SHA mismatch at "${stageId}": expected=${f0102Sha}, gpu=${gpuTap.feedSha256}, wasm=${wasmTap.feedSha256}`
        );
      }

      const gpuVal = validateHitnetFinalOutputs(gpuTap.outputs);
      const wasmVal = validateHitnetFinalOutputs(wasmTap.outputs);
      if (!gpuVal.ok || !wasmVal.ok) {
        throw new Error(
          `Tapped final output contract error at "${stageId}": gpu=${gpuVal.error ?? 'ok'}, wasm=${wasmVal.error ?? 'ok'}`
        );
      }

      const [pName, sName] = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;
      const gpuPrimaryBitIdentical = checkFloat32BitIdentity(
        gpuVal.byName[pName].data,
        untappedGpuByName[pName].data
      );
      const gpuSecondaryBitIdentical = checkFloat32BitIdentity(
        gpuVal.byName[sName].data,
        untappedGpuByName[sName].data
      );
      const wasmPrimaryBitIdentical = checkFloat32BitIdentity(
        wasmVal.byName[pName].data,
        untappedWasmByName[pName].data
      );
      const wasmSecondaryBitIdentical = checkFloat32BitIdentity(
        wasmVal.byName[sName].data,
        untappedWasmByName[sName].data
      );

      const allBitIdentical =
        gpuPrimaryBitIdentical.bitIdentical &&
        gpuSecondaryBitIdentical.bitIdentical &&
        wasmPrimaryBitIdentical.bitIdentical &&
        wasmSecondaryBitIdentical.bitIdentical;

      let perturbationDetail: string | null = null;
      if (!allBitIdentical) {
        perturbationDetail =
          `Tap "${stageId}" changed final outputs relative to same-provider untapped control: ` +
          `gpuPrimary(mismatch=${gpuPrimaryBitIdentical.mismatchCount}, max=${gpuPrimaryBitIdentical.maxAbsDiff.toExponential(3)}), ` +
          `gpuSecondary(mismatch=${gpuSecondaryBitIdentical.mismatchCount}, max=${gpuSecondaryBitIdentical.maxAbsDiff.toExponential(3)}), ` +
          `wasmPrimary(mismatch=${wasmPrimaryBitIdentical.mismatchCount}, max=${wasmPrimaryBitIdentical.maxAbsDiff.toExponential(3)}), ` +
          `wasmSecondary(mismatch=${wasmSecondaryBitIdentical.mismatchCount}, max=${wasmSecondaryBitIdentical.maxAbsDiff.toExponential(3)})`;
      }

      const tapComparisons: LadderTapRecord['tapComparisons'] = [];
      for (const tapName of tapOutputs) {
        const gOut = gpuTap.outputs.find((o) => o.name === tapName);
        const wOut = wasmTap.outputs.find((o) => o.name === tapName);
        if (!gOut || !wOut) {
          throw new Error(`Missing requested tap "${tapName}" at "${stageId}"`);
        }
        const gArr = new Float32Array(gOut.buffer);
        const wArr = new Float32Array(wOut.buffer);
        const stats = summarizeRawTensorAndParity(gArr, wArr);
        const passesGate = passesIntermediateTapGate(stats);
        const hasNonZeroDiff = stats.maxAbsDiff > 0 || stats.nonFiniteMismatchCount > 0;
        if (allBitIdentical && hasNonZeroDiff && !trackers.earliestNonZeroTap) {
          trackers.earliestNonZeroTap = { stageId, tensorName: tapName, maxAbsDiff: stats.maxAbsDiff };
        }
        if (allBitIdentical && !passesGate && !trackers.firstGateBreachingTap) {
          trackers.firstGateBreachingTap = {
            stageId,
            tensorName: tapName,
            maxAbsDiff: stats.maxAbsDiff,
            p99AbsDiff: stats.p99AbsDiff,
            meanAbsDiff: stats.meanAbsDiff,
          };
        }
        const p2bSplit =
          stageId === 'p2b' && gArr.length === 76800 && wArr.length === 76800
            ? summarizeP2bChannelwiseParity(gArr, wArr)
            : null;
        tapComparisons.push({
          tensorName: tapName,
          dataType: gOut.dataType,
          dims: [...gOut.dims],
          stats,
          passesStrictGate: passesGate,
          hasNonZeroDiff,
          ...(p2bSplit ? { p2bSplit } : {}),
        });
      }

      const rec: LadderTapRecord = {
        stepIndex: stepCounter,
        stageId,
        stageLabel,
        baseModelSha256: m1Sha,
        tappedModelSha256: tappedSha,
        tappedModelByteLength: tapped.bytes.byteLength,
        tapOutputs: [...tapOutputs],
        stoppedBecauseFinalPerturbed: !allBitIdentical,
        perturbationDetail,
        gpuModelSha256: gpuTap.modelSha256,
        wasmModelSha256: wasmTap.modelSha256,
        gpuFeedSha256: gpuTap.feedSha256,
        wasmFeedSha256: wasmTap.feedSha256,
        finalInvariance: {
          gpuPrimaryBitIdentical,
          gpuSecondaryBitIdentical,
          wasmPrimaryBitIdentical,
          wasmSecondaryBitIdentical,
        },
        tapComparisons,
      };
      targetLog.push(rec);

      log(
        `  [Tap #${rec.stepIndex} ${stageId}] tappedSha=${tappedSha.slice(0, 12)}... (base=${m1Sha.slice(0, 12)}..., bytes=${tapped.bytes.byteLength}) | bitIdenticalFinals=${allBitIdentical}`
      );
      if (!allBitIdentical) {
        log(`    [STOP: TAP_PERTURBED_FINALS] ${perturbationDetail}`);
      } else {
        for (const tc of tapComparisons) {
          log(
            `    tap=${tc.tensorName} (${tc.dataType} ${JSON.stringify(tc.dims)}): ${tc.passesStrictGate ? 'PASS_GATE' : 'BREACH_GATE'} | max=${tc.stats.maxAbsDiff.toExponential(3)} p99=${tc.stats.p99AbsDiff.toExponential(3)} mean=${tc.stats.meanAbsDiff.toExponential(3)} finite=${tc.stats.finiteCount}/${tc.stats.length}`
          );
          if (tc.p2bSplit) {
            log(
              `      P2b split: ch0(disp)=${tc.p2bSplit.ch0Disp.maxAbsDiff.toExponential(3)} ch1..2(zero)=${tc.p2bSplit.ch1_2Zeros.maxAbsDiff.toExponential(3)} ch3..15(LeakyRelu)=${tc.p2bSplit.ch3_15Features.maxAbsDiff.toExponential(3)}`
            );
          }
        }
      }
      return rec;
    };

    // Build chronological coarse ladder stages (verified static float32 tensors)
    const coarseStages: Array<{ id: string; label: string; outputs: string[] }> = [];
    const stemPair: string[] = [];
    if (
      topology.stemMidTensor &&
      topology.stemPreSplitTensor &&
      topology.stemMidTensor !== topology.stemPreSplitTensor
    ) {
      stemPair.push(topology.stemMidTensor, topology.stemPreSplitTensor);
    } else if (topology.stemPreSplitTensor) {
      stemPair.push(topology.stemPreSplitTensor);
    }
    if (stemPair.length > 0) {
      coarseStages.push({
        id: 'fe-stem-pre-split',
        label: 'fe_shared stem before left/right branch split',
        outputs: stemPair,
      });
    }
    if (
      topology.staticMeta.tensorMeta[topology.feSplitTensors[0]] &&
      topology.staticMeta.tensorMeta[topology.feSplitTensors[1]]
    ) {
      coarseStages.push({
        id: 'fe-branch-split',
        label: 'fe_shared_3 / fe_shared_31 branch split [1,60,80,48]',
        outputs: [...topology.feSplitTensors],
      });
    }
    for (const gid of ['p2-cast', 'p2b', 'p5-p6', 'p7', 'p8', 'p9', 'p10', 'p11'] as const) {
      const spec = getHitnetFullGraphTapGroup(gid);
      if (
        topology.staticMeta.tensorMeta[spec.outputs[0]] &&
        topology.staticMeta.tensorMeta[spec.outputs[1]]
      ) {
        coarseStages.push({
          id: spec.id,
          label: spec.label,
          outputs: [...spec.outputs],
        });
      }
    }

    let firstBreachStageIdx = -1;
    let divergingBranch: 'shared-stem' | 'primary' | 'secondary' = 'primary';
    let firstDivergingTensor = '';
    let prevCleanTensors: string[] = [];
    let prevCleanStageId = 'graph-input';

    for (let sIdx = 0; sIdx < coarseStages.length; sIdx++) {
      const st = coarseStages[sIdx];
      const rec = await runSingleTapStep(st.id, st.label, st.outputs, coarseLadderRecords);
      if (rec.stoppedBecauseFinalPerturbed) {
        abortedByFinalPerturbation = true;
        perturbationStopMessage = rec.perturbationDetail;
        break;
      }
      const breachingTcs = rec.tapComparisons.filter((t) => !t.passesStrictGate);
      if (breachingTcs.length > 0) {
        firstBreachStageIdx = sIdx;
        if (st.id === 'fe-stem-pre-split') {
          divergingBranch = 'shared-stem';
          if (
            st.outputs.length === 2 &&
            rec.tapComparisons[0]?.passesStrictGate &&
            !rec.tapComparisons[1]?.passesStrictGate
          ) {
            prevCleanTensors = [st.outputs[0]];
            prevCleanStageId = 'fe-stem-mid';
            firstDivergingTensor = st.outputs[1];
          } else {
            firstDivergingTensor = breachingTcs[0].tensorName;
          }
        } else {
          // Pick whichever branch (primary index 0 vs secondary index 1) breached first/worse
          const primaryBreach = rec.tapComparisons[0] && !rec.tapComparisons[0].passesStrictGate;
          divergingBranch = primaryBreach ? 'primary' : 'secondary';
          firstDivergingTensor = primaryBreach
            ? rec.tapComparisons[0].tensorName
            : rec.tapComparisons[rec.tapComparisons.length - 1].tensorName;
        }
        break;
      }
      prevCleanStageId = st.id;
      prevCleanTensors = st.outputs;
    }

    // Determine interval and binary-search first suspected operator if not aborted by final perturbation
    let suspectedOperator: SuspectedOperatorIsolationSummary | null = null;
    if (!abortedByFinalPerturbation) {
      const [pFinalName, sFinalName] = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;
      if (firstBreachStageIdx < 0) {
        // Every checked coarse stage up through p11 passed strict gates; interval is [p11 -> diverging final]
        const primaryFailsFinal = !(
          selectedOptLevel === 'disabled'
            ? f0102Data.comparisons.webgpuDisabledVsWasmDisabled.primary.passesStrictGate
            : f0102Data.comparisons.webgpuBasicVsWasmBasic.primary.passesStrictGate
        );
        divergingBranch = primaryFailsFinal ? 'primary' : 'secondary';
        firstDivergingTensor = primaryFailsFinal ? pFinalName : sFinalName;
      }

      const intervalNodes = getOrderedIntervalNodesForBisection(
        topology,
        prevCleanTensors,
        firstDivergingTensor
      );

      let lo = 0;
      let hi = intervalNodes.length; // index `hi` represents `firstDivergingTensor`
      const maxBisectSteps = 10;
      let bisectStepsUsed = 0;

      while (lo < hi && bisectStepsUsed < maxBisectSteps && !abortedByFinalPerturbation) {
        const mid = Math.floor((lo + hi) / 2);
        const midNode = intervalNodes[mid];
        const midTensor = midNode.outputs[0];
        bisectStepsUsed++;
        const bRec = await runSingleTapStep(
          `bisect-${bisectStepsUsed}`,
          `Operator bisection on ${divergingBranch}: ${midNode.opType} "${midNode.name}" -> ${midTensor}`,
          [midTensor],
          bisectionRecords
        );
        if (bRec.stoppedBecauseFinalPerturbed) {
          abortedByFinalPerturbation = true;
          perturbationStopMessage = bRec.perturbationDetail;
          break;
        }
        const tc = bRec.tapComparisons[0];
        if (!tc.passesStrictGate) {
          hi = mid;
        } else {
          lo = mid + 1;
        }
      }

      if (!abortedByFinalPerturbation) {
        const suspectNode =
          hi < intervalNodes.length
            ? intervalNodes[hi]
            : topology.producerByOutput.get(firstDivergingTensor) ?? null;

        if (suspectNode) {
          const suspectOutName = suspectNode.outputs[0] || firstDivergingTensor;
          const existingRecord =
            [...bisectionRecords, ...coarseLadderRecords].find((r) =>
              r.tapComparisons.some((t) => t.tensorName === suspectOutName)
            );
          const suspectOutStats =
            existingRecord?.tapComparisons.find((t) => t.tensorName === suspectOutName)?.stats ??
            summarizeRawTensorAndParity(
              untappedGpuByName[suspectOutName]?.data ?? new Float32Array(0),
              untappedWasmByName[suspectOutName]?.data ?? new Float32Array(0)
            );

          const parentGroups = getImmediateParentTapGroups(topology, suspectNode);
          const immediateParentRuns: LadderTapRecord[] = [];
          for (let pgIdx = 0; pgIdx < parentGroups.length; pgIdx++) {
            const pRec = await runSingleTapStep(
              `suspect-parents-${pgIdx + 1}`,
              `Immediate parents of suspected ${suspectNode.opType} "${suspectNode.name}"`,
              parentGroups[pgIdx],
              bisectionRecords
            );
            immediateParentRuns.push(pRec);
            if (pRec.stoppedBecauseFinalPerturbed) {
              abortedByFinalPerturbation = true;
              perturbationStopMessage = pRec.perturbationDetail;
              break;
            }
          }

          suspectedOperator = {
            branch: divergingBranch,
            previousCleanStageId: prevCleanStageId,
            firstDivergingStageId:
              firstBreachStageIdx >= 0
                ? coarseStages[firstBreachStageIdx].id
                : `${firstDivergingTensor}(final-head)`,
            nodeIndex: suspectNode.idx,
            nodeName: suspectNode.name,
            opType: suspectNode.opType,
            attrsSummary: suspectNode.attrsSummary,
            outputTensorName: suspectOutName,
            outputStats: suspectOutStats,
            immediateParentNames: suspectNode.inputs.filter(Boolean),
            immediateParentRuns,
            caveat:
              'ORT Web does not expose per-node EP placement in JS; locating the earliest gate-breaching tensor/node at the selected graphOptimizationLevel identifies the first observed divergence interval/operator output, not proof of GPU-vs-CPU node placement or a single compiler optimization pass.',
          };

          const parentLines = immediateParentRuns
            .flatMap((r) => r.tapComparisons)
            .map(
              (tc) =>
                `${tc.tensorName}[${tc.passesStrictGate ? 'PASS' : 'BREACH'} max=${tc.stats.maxAbsDiff.toExponential(3)} p99=${tc.stats.p99AbsDiff.toExponential(3)} mean=${tc.stats.meanAbsDiff.toExponential(3)}]`
            );
          log(
            `\n  [Localization Summary (F0102 opt=${selectedOptLevel})]\n` +
            `    earliestNonZeroTap=${trackers.earliestNonZeroTap ? `${trackers.earliestNonZeroTap.stageId}@${trackers.earliestNonZeroTap.tensorName}(max=${trackers.earliestNonZeroTap.maxAbsDiff.toExponential(3)})` : 'none'}\n` +
            `    firstGateBreachingTap=${trackers.firstGateBreachingTap ? `${trackers.firstGateBreachingTap.stageId}@${trackers.firstGateBreachingTap.tensorName}(max=${trackers.firstGateBreachingTap.maxAbsDiff.toExponential(3)})` : firstDivergingTensor}\n` +
            `    firstSuspectedOp=[#${suspectNode.idx}] ${suspectNode.opType} "${suspectNode.name}" attrs=${JSON.stringify(suspectNode.attrsSummary)} out="${suspectOutName}" -> max=${suspectOutStats.maxAbsDiff.toExponential(3)} p99=${suspectOutStats.p99AbsDiff.toExponential(3)} mean=${suspectOutStats.meanAbsDiff.toExponential(3)}\n` +
            `    immediateParents=${JSON.stringify(suspectNode.inputs)}${parentLines.length > 0 ? ` | parentComparisons=${parentLines.join(', ')}` : ''}\n` +
            `    caveat: ${suspectedOperator.caveat}`
          );
        }
      }
    }

    localization = {
      executed: true,
      reasonSkipped: null,
      frame: 'F0102',
      selectedOptLevel,
      abortedByFinalPerturbation,
      perturbationStopMessage,
      earliestNonZeroTap: trackers.earliestNonZeroTap,
      firstGateBreachingTap: trackers.firstGateBreachingTap,
      coarseLadderRecords,
      bisectionRecords,
      suspectedOperator,
    };
  }

  return {
    ok: true,
    abortReason: localization?.abortedByFinalPerturbation
      ? localization.perturbationStopMessage
      : null,
    pinnedOrtVersion: '1.30.0',
    runtimeAssetFamily: 'asyncify',
    m1ByteLength: exactM1Bytes.byteLength,
    m1Sha256: m1Sha,
    f0102FeedSha256: f0102Sha,
    f0126FeedSha256: f0126Sha,
    frames: {
      F0102: f0102Data,
      F0126: f0126Data,
    },
    localization,
  };
}
