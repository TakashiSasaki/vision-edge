import {
  buildVerifiedPrefixThroughNode,
  type Stage2RealFrameInjectionReport,
  type VerifiedPrefixThroughNode,
} from './hitnetConv9BlockedInjection';
import {
  buildVerifiedSuffixAfterConv7,
  type VerifiedPostConv7Suffix,
} from './hitnetConv7ActivationSubstitution';
import {
  appendOnnxGraphOutputs,
  summarizeExactInt64IndexComparison,
  summarizeRawTensorAndParity,
  type ExactInt64IndexComparisonSummary,
  type RawTensorParityStats,
} from './hitnetGatherNdDiagnostic';
import {
  HITNET_ORT_130_FAMILY_PROFILES,
  type OrtDiagnosticExtraFeedCheck,
  type OrtVersionDiagnosticFailure,
  type OrtVersionDiagnosticOutput,
  type OrtVersionDiagnosticResponse,
  type OrtVersionDiagnosticSuccess,
} from './hitnetOrtComparisonHelper';
import {
  resolveVerifiedPostConv7ChainSpec,
  verifyCanonicalM1NoN1Replacements,
  type N1OffVerificationProof,
  type VerifiedPostConv7ChainSpec,
} from './hitnetPostConv7TapReplay';
import {
  checkFloat32BitIdentity,
  createBrowserOrt130DiagnosticRunner,
  HITNET_RESIDUAL_SOURCE_CALIBRATION_DEFAULT,
  sha256HexBytes,
  validateHitnetFinalOutputs,
  type BitIdentityResult,
  type ResidualDiagnosticWorkerRunner,
} from './hitnetResidualLocalization';
import {
  buildSingleOpCapturedInputModel,
  compareFloat32TensorsDetailed,
  parseM1ModelProtoWithRawSlices,
  type DetailedFloat32Comparison,
  type ParsedM1ModelFull,
  type ParsedM1NodeFull,
} from './hitnetRgbStemFirstNonZero';
import {
  assertDevFixtureEnabled,
  HITNET_DEV_NON_JSEP_FIXTURE_SPEC,
} from './stereoDepthWebGpuDevFixture';
import {
  disparityToDepthMeters,
  extractDisparityFromOrtTensor,
  HITNET_M1_FULL_GRAPH_SHA256,
} from './stereoDepthWorkerCore';

export type HostConv9TransportMode = 'native-gpu9-copy' | 'blocked-k128';

export interface HostConv9TransportEdgeRecord {
  name: string;
  role: 'primary-cut-9' | 'bypass';
  dims: number[];
  dataType: 'float32' | 'int32' | 'int64' | 'uint8';
  byteOffsetBefore: number;
  byteLengthBefore: number;
  sha256Before: string;
  byteOffsetAfter: number | null;
  byteLengthAfter: number | null;
  sha256After: string | null;
  orderIndexExpected: number;
  orderIndexActual: number | null;
  matchesAllChecks: boolean;
}

export interface TransportFailureDetail {
  checkId: string;
  stage: 'pre-transfer' | 'worker-execution' | 'post-transfer' | 'output-validation' | 'gate-bit-identity';
  edgeName: string | null;
  detail: string;
  expected: string;
  actual: string;
}

export interface SharedHostTransportExecutionResult {
  ok: boolean;
  label: string;
  mode: HostConv9TransportMode;
  suffixSha256: string;
  suffixByteLength: number;
  expectedInputOrder: string[];
  actualInputOrder: string[] | null;
  edges: HostConv9TransportEdgeRecord[];
  firstFailure: TransportFailureDetail | null;
  workerSuccess: OrtVersionDiagnosticSuccess | null;
  dispatchObservedSeparatePass: boolean | null;
  primaryF32: Float32Array | null;
  secondaryF32: Float32Array | null;
  outputsByName: Record<string, OrtVersionDiagnosticOutput>;
}

export interface SecondaryOutlierAncestryNodeRef {
  nodeIdx: number;
  nodeName: string;
  opType: string;
  outputTensor: string;
  dims: number[];
  elemType: number;
  attrsSummary: string[];
}

export interface SecondaryOutlierAncestrySpec {
  primaryAncestorCount: number;
  secondaryAncestorCount: number;
  secondaryOnlyNodeCount: number;
  secondaryOnlyMinNodeIdx: number;
  secondaryOnlyMaxNodeIdx: number;
  coarseArgMin: SecondaryOutlierAncestryNodeRef & {
    inputCostTensor: string;
    inputCostDims: number[];
    axis: number;
    keepdims: number;
    selectLastIndex: number;
    maxValidIndex: number;
  };
  coarseCast: SecondaryOutlierAncestryNodeRef | null;
  initToPropOut: SecondaryOutlierAncestryNodeRef | null;
  propWarpOut: SecondaryOutlierAncestryNodeRef | null;
  prop0AddOut: SecondaryOutlierAncestryNodeRef | null;
  prop0SingletonArgMin: (SecondaryOutlierAncestryNodeRef & { axis: number; inputDims: number[] }) | null;
  postProp0Out: SecondaryOutlierAncestryNodeRef | null;
  refinementL0Out: SecondaryOutlierAncestryNodeRef | null;
  refinementL1Out: SecondaryOutlierAncestryNodeRef | null;
  refinementL2Add: SecondaryOutlierAncestryNodeRef | null;
  refinementL2SlantAdd: SecondaryOutlierAncestryNodeRef | null;
}

export interface EyeValidityAndDepthFlipSummary {
  eye: 'left-primary' | 'right-secondary';
  tensorName: string;
  totalPixels: number;
  negateNative: boolean;
  negateCand: boolean;
  negateWasm: boolean;
  nativeValidCount: number;
  candValidCount: number;
  wasmValidCount: number;
  nativeValidToCandInvalid: number;
  nativeInvalidToCandValid: number;
  totalValidityFlipsVsNative: number;
  wasmValidToCandInvalid: number;
  wasmInvalidToCandValid: number;
  totalValidityFlipsVsWasm: number;
  depthThresholdFlipsVsNative: number;
  depthThresholdFlipsVsWasm: number;
}

export interface SecondaryOutlierPixelCoord {
  rank: number;
  flatIdx: number;
  y: number;
  x: number;
  ty: number;
  tx: number;
  nativeVal: number;
  candVal: number;
  wasmVal: number;
  signedDiffVsNative: number;
  absDiffVsNative: number;
  signedDiffVsWasm: number;
  absDiffVsWasm: number;
}

export interface SecondaryOutlierClusterSummary {
  clusterId: number;
  pixelCount: number;
  bbox: { minY: number; maxY: number; minX: number; maxX: number };
  tileBbox: { minTy: number; maxTy: number; minTx: number; maxTx: number };
  uniqueTileCount: number;
  sampleTiles: Array<{ ty: number; tx: number }>;
  peak: SecondaryOutlierPixelCoord;
}

export interface TapPassInvarianceRecord {
  passLabel: string;
  tapOutputs: string[];
  tappedModelByteLength: number;
  tappedModelSha256: string;
  nativeTransportOk: boolean;
  candTransportOk: boolean;
  nativePrimaryBitIdentical: BitIdentityResult | null;
  nativeSecondaryBitIdentical: BitIdentityResult | null;
  candPrimaryBitIdentical: BitIdentityResult | null;
  candSecondaryBitIdentical: BitIdentityResult | null;
  bothArmsBitInvariant: boolean;
  failureReason: string | null;
}

export interface CoarseArgMinTileCostInspection {
  ty: number;
  tx: number;
  tileFlatIdx: number;
  isSecondaryOutlierTile: boolean;
  dNative: number;
  dCand: number;
  deltaD: number;
  expectedOnnxArgMinNative: number | null;
  expectedOnnxArgMinCand: number | null;
  observedMatchesOnnxExactNative: boolean;
  observedMatchesOnnxExactCand: boolean;
  finiteCostCountNative: number;
  finiteCostCountCand: number;
  minCostNative: number;
  minCostCand: number;
  tieIndicesNative: number[];
  tieIndicesCand: number[];
  costNativeAtDNative: number;
  costNativeAtDCand: number;
  costCandAtDNative: number;
  costCandAtDCand: number;
  marginNativeCandMinusNatWinner: number;
  marginCandNatWinnerMinusCandWinner: number;
  classification:
    | 'EXACT_COST_ORDER_CROSSING'
    | 'EXACT_TIE_TIEBREAKER'
    | 'ARGMIN_KERNEL_MISMATCH'
    | 'NO_BIN_FLIP';
  initToPropDispCh0Native: number | null;
  initToPropDispCh0Cand: number | null;
  postProp0MaxAbsDiffAtTile: number | null;
}

export interface CoarseArgMinAndCostAnalysis {
  tapPass1Invariance: TapPassInvarianceRecord;
  tapPass2Invariance: TapPassInvarianceRecord | null;
  coarseArgMinSelectLastIndex: number;
  coarseArgMinDataTypeNative: string;
  coarseArgMinDataTypeCand: string;
  int64Comparison: ExactInt64IndexComparisonSummary;
  outlierTileIndices: Array<{ ty: number; tx: number; tileFlatIdx: number }>;
  outlierTilesWithArgMinFlipCount: number;
  nonOutlierTilesWithArgMinFlipCount: number;
  postProp0FullDiff: DetailedFloat32Comparison | null;
  postProp0OutlierTileMaxAbsDiff: number | null;
  coarseCostFullDiff: DetailedFloat32Comparison | null;
  coarseCostOutlierTileMaxAbsDiff: number | null;
  inspectedTiles: CoarseArgMinTileCostInspection[];
  exactCostOrderCrossingsCount: number;
  exactTieTiebreakersCount: number;
  argminKernelMismatchCount: number;
}

export interface LateStageTapComparison {
  passLabel: string;
  invariance: TapPassInvarianceRecord;
  taps: Array<{
    name: string;
    dims: number[];
    dataType: string;
    fullComparison: DetailedFloat32Comparison | null;
    roiMaxAbsDiff: number | null;
    roiMeanAbsDiff: number | null;
  }>;
}

export interface CommonWasm7SecondaryExposureReport {
  source: 'reused-existing-summary' | 'fresh-controlled-f0126-arm' | 'skipped-on-gate-failure';
  frameLabel: 'F0126';
  feedSha256: string;
  controlScopeNote: string;
  dispatchConfirmedSeparatePass: boolean | null;
  baselineGpuSuffixWasm7VsWasmFull: {
    primary: RawTensorParityStats;
    secondary: RawTensorParityStats;
    leftValidBaseline?: number;
    leftValidWasm?: number;
    rightValidBaseline?: number;
    rightValidWasm?: number;
  } | null;
  intervenedGpuSuffixWasm7WithCand9VsWasmFull: {
    primary: RawTensorParityStats;
    secondary: RawTensorParityStats;
    primaryFlipsVsWasm?: number;
    secondaryFlipsVsWasm?: number;
    leftValidCand?: number;
    rightValidCand?: number;
  } | null;
  intervenedVsBaselineGpuSuffix: {
    primary: RawTensorParityStats;
    secondary: RawTensorParityStats;
  } | null;
}

export interface HitnetF0126SecondaryOutlierReport {
  ok: boolean;
  aborted: boolean;
  abortReason: string | null;
  gatePassed: boolean;
  gateFirstFailure: TransportFailureDetail | null;
  pinnedOrtVersion: '1.30.0';
  runtimeAssetFamily: 'asyncify';
  graphOptimizationLevel: 'disabled';
  numThreads: 1;
  proxy: false;
  m1ByteLength: number;
  m1Sha256: string;
  f0126FeedSha256: string;
  n1OffProof: N1OffVerificationProof;
  chainSpec: VerifiedPostConv7ChainSpec;
  verifiedArtifacts: {
    prefix0to8ByteLength: number;
    prefix0to8Sha256: string;
    suffix9ByteLength: number;
    suffix9Sha256: string;
    singleOp9ByteLength: number;
    singleOp9Sha256: string;
    prefixNode8Sha256: string;
    gpuOneOp9RawSha256: string;
    bypassEdgeHashes: Array<{ name: string; dataType: string; dims: number[]; byteLength: number; sha256: string }>;
  };
  nativeSplitControlCheck: {
    gpuSplitVsFullGpuPrimary: BitIdentityResult;
    gpuSplitVsFullGpuSecondary: BitIdentityResult;
    passed: boolean;
  };
  gateTransportRecord: {
    execution: Omit<SharedHostTransportExecutionResult, 'primaryF32' | 'secondaryF32' | 'outputsByName'>;
    vsNativeSplitPrimary: BitIdentityResult | null;
    vsNativeSplitSecondary: BitIdentityResult | null;
    vsFullGpuDisabledPrimary: BitIdentityResult | null;
    vsFullGpuDisabledSecondary: BitIdentityResult | null;
  };
  outlierComparison: {
    cand128TransportOk: boolean;
    cand128TransportEdges: HostConv9TransportEdgeRecord[];
    cand128DispatchObservedSeparatePass: boolean;
    edge9Cand128VsNativeGpu9: DetailedFloat32Comparison;
    cand128VsNativeOnCand: {
      primary: DetailedFloat32Comparison;
      secondary: DetailedFloat32Comparison;
    };
    cand128VsFullWasmDisabled: {
      primary: DetailedFloat32Comparison;
      secondary: DetailedFloat32Comparison;
    };
    nativeOnCandVsFullWasmDisabled: {
      primary: DetailedFloat32Comparison;
      secondary: DetailedFloat32Comparison;
    };
    cand128VsFullGpuDisabled: {
      primary: DetailedFloat32Comparison;
      secondary: DetailedFloat32Comparison;
    };
    eyeValidity: {
      leftPrimary: EyeValidityAndDepthFlipSummary;
      rightSecondary: EyeValidityAndDepthFlipSummary;
    };
    secondaryClusterThresholdUsed: number;
    secondaryOutlierPixelCount: number;
    secondaryClusters: SecondaryOutlierClusterSummary[];
    topSecondaryOutlierPixels: SecondaryOutlierPixelCoord[];
  } | null;
  ancestryAndTaps: {
    ancestrySpec: SecondaryOutlierAncestrySpec;
    coarseAnalysis: CoarseArgMinAndCostAnalysis | null;
    lateStageTaps: LateStageTapComparison[];
  } | null;
  causalAttribution: {
    verdict:
      | 'COARSE_ARGMIN_COST_ORDER_CROSSING'
      | 'COARSE_ARGMIN_EXACT_TIE_FLIP'
      | 'COARSE_ARGMIN_KERNEL_MISMATCH'
      | 'POST_COARSE_PROP0_AMPLIFICATION'
      | 'POST_PROP0_REFINEMENT_AMPLIFICATION'
      | 'NO_SIGNIFICANT_SECONDARY_OUTLIER'
      | 'UNRESOLVED';
    summary: string;
    evidencePoints: string[];
  } | null;
  commonWasm7Secondary: CommonWasm7SecondaryExposureReport;
  scientificScopeNote: string;
}

// ---------- Host allocation/copy + blocked-K=128 exact shared routine ----------

function getConvAttrsForHostBuffer(parsed: ParsedM1ModelFull, convNodeIdx: number) {
  const node = parsed.nodes[convNodeIdx];
  if (!node || node.opType !== 'Conv') {
    throw new Error(`Node #${convNodeIdx} is not Conv`);
  }
  const wInit = node.inputs[1] ? parsed.initializersByName.get(node.inputs[1]) : undefined;
  const bInit = node.inputs[2] ? parsed.initializersByName.get(node.inputs[2]) : undefined;
  if (!wInit?.float32Values || wInit.dims.length !== 4) {
    throw new Error(`Conv #${convNodeIdx} missing valid 4D float32 weight initializer`);
  }
  const getInts = (name: string, fb: number[]) => {
    const a = node.attrs.find((x) => x.name === name);
    return a && a.intsVal.length > 0 ? a.intsVal : fb;
  };
  const group = node.attrs.find((x) => x.name === 'group')?.intVal ?? 1;
  const strides = getInts('strides', [1, 1]);
  const dilations = getInts('dilations', [1, 1]);
  const pads = getInts('pads', [0, 0, 0, 0]);
  return {
    wData: wInit.float32Values,
    wDims: wInit.dims,
    bData: bInit?.float32Values ?? null,
    group,
    strideH: strides[0] ?? 1,
    strideW: strides[1] ?? 1,
    dilH: dilations[0] ?? 1,
    dilW: dilations[1] ?? 1,
    padTop: pads[0] ?? 0,
    padLeft: pads[1] ?? 0,
  };
}

/**
 * Single shared fresh, owned, contiguous Float32 host-allocation and output population routine
 * used identically by both the Native-on-Candidate transport gate and the K=128 candidate arm.
 */
export function populateFreshOwnedHostConv9Buffer(params: {
  mode: HostConv9TransportMode;
  nativeGpu9F32?: Float32Array;
  parsed?: ParsedM1ModelFull;
  convNodeIdx?: number;
  xData?: Float32Array;
  inputDims?: readonly [number, number, number, number];
  outDims: readonly [number, number, number, number];
}): Float32Array {
  const [N, oM, oH, oW] = params.outDims;
  const totalOut = N * oM * oH * oW;
  if (totalOut <= 0) {
    throw new Error(`Invalid Conv #9 output shape ${JSON.stringify(params.outDims)}`);
  }
  // Always allocate a fresh, owned, zero-offset, contiguous Float32Array
  const out = new Float32Array(totalOut);
  const spatialOut = oH * oW;

  if (params.mode === 'native-gpu9-copy') {
    const src = params.nativeGpu9F32;
    if (!src || src.length !== totalOut) {
      throw new Error(
        `Native GPU #9 source length mismatch: got=${src?.length ?? 0}, expected=${totalOut}`
      );
    }
    for (let n = 0; n < N; n++) {
      const nOutBase = n * oM * spatialOut;
      for (let oh = 0; oh < oH; oh++) {
        for (let ow = 0; ow < oW; ow++) {
          const pixOff = oh * oW + ow;
          for (let m = 0; m < oM; m++) {
            const idx = nOutBase + m * spatialOut + pixOff;
            out[idx] = src[idx];
          }
        }
      }
    }
    return out;
  }

  // mode === 'blocked-k128'
  if (!params.parsed || params.convNodeIdx === undefined || !params.xData || !params.inputDims) {
    throw new Error('blocked-k128 host population requires parsed, convNodeIdx, xData, and inputDims');
  }
  const attrs = getConvAttrsForHostBuffer(params.parsed, params.convNodeIdx);
  const [xN, C, H, W] = params.inputDims;
  const [M, CPerG, kH, kW] = attrs.wDims;
  const { wData, bData, group, strideH, strideW, dilH, dilW, padTop, padLeft } = attrs;
  if (xN !== N || M !== oM || C !== group * CPerG) {
    throw new Error(
      `Invalid full-tensor Conv shapes: in=${params.inputDims.join('x')}, out=${params.outDims.join('x')}`
    );
  }
  const K = CPerG * kH * kW;
  const blockK = Math.min(128, K);
  const f32 = Math.fround;
  const gSize = M / group;
  const col = new Float32Array(K);
  const hw = H * W;

  const fast4x4Stride2 =
    group === 1 &&
    kH === 4 &&
    kW === 4 &&
    strideH === 2 &&
    strideW === 2 &&
    dilH === 1 &&
    dilW === 1 &&
    padTop === 1 &&
    padLeft === 1 &&
    K % blockK === 0 &&
    blockK % 4 === 0;

  for (let g = 0; g < group; g++) {
    const cStart = g * CPerG;
    const mStart = g * gSize;
    const mEnd = mStart + gSize;

    for (let n = 0; n < N; n++) {
      const nXBase = n * C * hw;
      const nOutBase = n * oM * spatialOut;

      for (let oh = 0; oh < oH; oh++) {
        for (let ow = 0; ow < oW; ow++) {
          let k = 0;
          if (fast4x4Stride2) {
            const h0 = oh * 2 - 1;
            const w0 = ow * 2 - 1;
            if (h0 >= 0 && h0 + 3 < H && w0 >= 0 && w0 + 3 < W) {
              for (let ic = 0; ic < CPerG; ic++) {
                let rOff = nXBase + ((cStart + ic) * H + h0) * W + w0;
                col[k++] = params.xData[rOff];
                col[k++] = params.xData[rOff + 1];
                col[k++] = params.xData[rOff + 2];
                col[k++] = params.xData[rOff + 3];
                rOff += W;
                col[k++] = params.xData[rOff];
                col[k++] = params.xData[rOff + 1];
                col[k++] = params.xData[rOff + 2];
                col[k++] = params.xData[rOff + 3];
                rOff += W;
                col[k++] = params.xData[rOff];
                col[k++] = params.xData[rOff + 1];
                col[k++] = params.xData[rOff + 2];
                col[k++] = params.xData[rOff + 3];
                rOff += W;
                col[k++] = params.xData[rOff];
                col[k++] = params.xData[rOff + 1];
                col[k++] = params.xData[rOff + 2];
                col[k++] = params.xData[rOff + 3];
              }
            } else {
              for (let ic = 0; ic < CPerG; ic++) {
                const cOff = nXBase + (cStart + ic) * hw;
                for (let kh = 0; kh < 4; kh++) {
                  const ih = h0 + kh;
                  if (ih < 0 || ih >= H) {
                    col[k++] = 0;
                    col[k++] = 0;
                    col[k++] = 0;
                    col[k++] = 0;
                  } else {
                    const rOff = cOff + ih * W;
                    col[k++] = w0 >= 0 && w0 < W ? params.xData[rOff + w0] : 0;
                    col[k++] = w0 + 1 >= 0 && w0 + 1 < W ? params.xData[rOff + w0 + 1] : 0;
                    col[k++] = w0 + 2 >= 0 && w0 + 2 < W ? params.xData[rOff + w0 + 2] : 0;
                    col[k++] = w0 + 3 >= 0 && w0 + 3 < W ? params.xData[rOff + w0 + 3] : 0;
                  }
                }
              }
            }
          } else {
            for (let ic = 0; ic < CPerG; ic++) {
              const cOff = nXBase + (cStart + ic) * hw;
              for (let kh = 0; kh < kH; kh++) {
                const ih = oh * strideH + kh * dilH - padTop;
                if (ih < 0 || ih >= H) {
                  for (let kw = 0; kw < kW; kw++) col[k++] = 0;
                } else {
                  const rOff = cOff + ih * W;
                  for (let kw = 0; kw < kW; kw++) {
                    const iw = ow * strideW + kw * dilW - padLeft;
                    col[k++] = iw >= 0 && iw < W ? params.xData[rOff + iw] : 0;
                  }
                }
              }
            }
          }

          const pixOff = oh * oW + ow;
          for (let m = mStart; m < mEnd; m++) {
            const wBase = m * K;
            const bias = bData ? f32(bData[m]) : 0.0;
            let merged = 0.0;

            if (fast4x4Stride2) {
              for (let blk = 0; blk < K; blk += blockK) {
                let s = 0.0;
                const blkEnd = blk + blockK;
                for (let kk = blk; kk < blkEnd; kk += 4) {
                  s = f32(s + f32(col[kk] * wData[wBase + kk]));
                  s = f32(s + f32(col[kk + 1] * wData[wBase + kk + 1]));
                  s = f32(s + f32(col[kk + 2] * wData[wBase + kk + 2]));
                  s = f32(s + f32(col[kk + 3] * wData[wBase + kk + 3]));
                }
                merged = f32(merged + s);
              }
            } else {
              for (let blk = 0; blk < K; blk += blockK) {
                let s = 0.0;
                const blkEnd = Math.min(K, blk + blockK);
                for (let kk = blk; kk < blkEnd; kk++) {
                  s = f32(s + f32(col[kk] * wData[wBase + kk]));
                }
                merged = f32(merged + s);
              }
            }
            out[nOutBase + m * spatialOut + pixOff] = f32(merged + bias);
          }
        }
      }
    }
  }

  return out;
}

// ---------- Dynamic secondary-only ancestry resolution ----------

function collectBackwardAncestors(
  parsed: ParsedM1ModelFull,
  targetOutputName: string
): Set<number> {
  const startNode = parsed.producerByOutput.get(targetOutputName);
  if (!startNode) {
    throw new Error(`Cannot trace ancestry: missing producer for "${targetOutputName}"`);
  }
  const visited = new Set<number>();
  const stack: number[] = [startNode.idx];
  while (stack.length > 0) {
    const idx = stack.pop()!;
    if (visited.has(idx)) continue;
    visited.add(idx);
    const node = parsed.nodes[idx];
    for (const inp of node.inputs) {
      if (!inp) continue;
      const prod = parsed.producerByOutput.get(inp);
      if (prod && !visited.has(prod.idx)) {
        stack.push(prod.idx);
      }
    }
  }
  return visited;
}

function getNodeIntAttr(node: ParsedM1NodeFull, name: string, fallback: number): number {
  const a = node.attrs.find((x) => x.name === name);
  return a && a.intVal !== null ? a.intVal : fallback;
}

function toNodeRef(parsed: ParsedM1ModelFull, node: ParsedM1NodeFull): SecondaryOutlierAncestryNodeRef {
  const out = node.outputs[0];
  if (!out) throw new Error(`Node #${node.idx} (${node.name}) has no output tensor`);
  const meta = parsed.staticMeta.tensorMeta[out];
  if (!meta) throw new Error(`Node #${node.idx} output "${out}" has no static tensor metadata`);
  return {
    nodeIdx: node.idx,
    nodeName: node.name,
    opType: node.opType,
    outputTensor: out,
    dims: [...meta.dims],
    elemType: meta.elemType,
    attrsSummary: [...node.attrsSummary],
  };
}

export function resolveSecondaryOutlierAncestrySpec(
  parsed: ParsedM1ModelFull,
  chainSpec: VerifiedPostConv7ChainSpec,
  suffix9: VerifiedPostConv7Suffix
): SecondaryOutlierAncestrySpec {
  const [primaryOutName, secondaryOutName] = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;
  const primaryAncestors = collectBackwardAncestors(parsed, primaryOutName);
  const secondaryAncestors = collectBackwardAncestors(parsed, secondaryOutName);

  const secondaryOnlyIndices = Array.from(secondaryAncestors)
    .filter((idx) => !primaryAncestors.has(idx))
    .sort((a, b) => a - b);

  if (secondaryOnlyIndices.length === 0) {
    throw new Error('Resolved empty secondary-only ancestry set relative to primary disparity output');
  }

  const suffixSecondaryNodes = secondaryOnlyIndices
    .filter((idx) => idx > chainSpec.node9ConvDown0.nodeIdx)
    .map((idx) => parsed.nodes[idx]);

  const suffixProducedSet = new Set<string>();
  for (let i = suffix9.suffixNodeRange.firstOriginalIdx; i <= suffix9.suffixNodeRange.lastOriginalIdx; i++) {
    for (const o of parsed.nodes[i].outputs) {
      if (o) suffixProducedSet.add(o);
    }
  }

  const assertInsideSuffix = (ref: SecondaryOutlierAncestryNodeRef, role: string) => {
    if (!suffixProducedSet.has(ref.outputTensor)) {
      throw new Error(
        `Resolved ${role} tensor "${ref.outputTensor}" (#${ref.nodeIdx}) is not inside suffix9`
      );
    }
  };

  // 1. Coarse non-singleton ArgMin inside secondary-only suffix nodes
  let coarseArgMinNode: ParsedM1NodeFull | null = null;
  let coarseArgMinAxis = 3;
  let coarseArgMinInDims: number[] = [];
  for (const n of suffixSecondaryNodes) {
    if (n.opType !== 'ArgMin') continue;
    const inpName = n.inputs[0];
    const inMeta = inpName ? parsed.staticMeta.tensorMeta[inpName] : undefined;
    if (!inMeta || inMeta.dims.length === 0) continue;
    const rawAxis = getNodeIntAttr(n, 'axis', 0);
    const normAxis = rawAxis < 0 ? inMeta.dims.length + rawAxis : rawAxis;
    const redDim = inMeta.dims[normAxis] ?? 0;
    if (redDim > 1) {
      coarseArgMinNode = n;
      coarseArgMinAxis = normAxis;
      coarseArgMinInDims = [...inMeta.dims];
      break;
    }
  }
  if (!coarseArgMinNode) {
    throw new Error('Could not resolve coarse non-singleton ArgMin node in secondary-only suffix ancestry');
  }
  const coarseArgMinRef = toNodeRef(parsed, coarseArgMinNode);
  assertInsideSuffix(coarseArgMinRef, 'coarseArgMin');

  const coarseArgMin = {
    ...coarseArgMinRef,
    inputCostTensor: coarseArgMinNode.inputs[0],
    inputCostDims: coarseArgMinInDims,
    axis: coarseArgMinAxis,
    keepdims: getNodeIntAttr(coarseArgMinNode, 'keepdims', 1),
    selectLastIndex: getNodeIntAttr(coarseArgMinNode, 'select_last_index', 0),
    maxValidIndex: Math.max(0, (coarseArgMinInDims[coarseArgMinAxis] ?? 1) - 1),
  };

  // 2. Coarse Cast downstream of coarseArgMin
  let coarseCast: SecondaryOutlierAncestryNodeRef | null = null;
  const castQueue: string[] = [coarseArgMin.outputTensor];
  const seenCastTensors = new Set<string>();
  while (castQueue.length > 0 && !coarseCast) {
    const cur = castQueue.shift()!;
    if (seenCastTensors.has(cur)) continue;
    seenCastTensors.add(cur);
    for (const n of suffixSecondaryNodes) {
      if (n.idx <= coarseArgMin.nodeIdx || !n.inputs.includes(cur)) continue;
      if (n.opType === 'Cast') {
        coarseCast = toNodeRef(parsed, n);
        break;
      }
      if (n.opType === 'Unsqueeze' || n.opType === 'Reshape' || n.opType === 'ExpandDims') {
        if (n.outputs[0]) castQueue.push(n.outputs[0]);
      }
    }
  }
  if (coarseCast) assertInsideSuffix(coarseCast, 'coarseCast');

  // 3. init_to_prop rank-5 [1,60,80,1,16], prop_warp rank-5, prop0/add rank-5, prop0 singleton ArgMin
  const initToPropNode =
    suffixSecondaryNodes.find((n) => {
      const o = n.outputs[0];
      const m = o ? parsed.staticMeta.tensorMeta[o] : undefined;
      return (
        Boolean(m && m.dims.length === 5 && m.dims[3] === 1 && m.dims[4] === 16) &&
        (n.name.includes('init_to_prop') || (o?.includes('init_to_prop') ?? false))
      );
    }) ?? null;
  const initToPropOut = initToPropNode ? toNodeRef(parsed, initToPropNode) : null;
  if (initToPropOut) assertInsideSuffix(initToPropOut, 'initToPropOut');

  const propWarpNode =
    suffixSecondaryNodes.find((n) => {
      const o = n.outputs[0];
      const m = o ? parsed.staticMeta.tensorMeta[o] : undefined;
      return (
        Boolean(m && m.dims.length === 5 && m.dims[3] === 1 && m.dims[4] === 16) &&
        (n.name.includes('prop_warp') || (o?.includes('prop_warp') ?? false))
      );
    }) ?? null;
  const propWarpOut = propWarpNode ? toNodeRef(parsed, propWarpNode) : null;
  if (propWarpOut) assertInsideSuffix(propWarpOut, 'propWarpOut');

  const prop0AddNode =
    suffixSecondaryNodes.find((n) => {
      if (n.opType !== 'Add') return false;
      const o = n.outputs[0];
      const m = o ? parsed.staticMeta.tensorMeta[o] : undefined;
      if (!m || m.dims.length !== 5 || m.dims[3] !== 1 || m.dims[4] !== 16) return false;
      return n.name.includes('prop0') || (initToPropOut ? n.inputs.includes(initToPropOut.outputTensor) : true);
    }) ?? null;
  const prop0AddOut = prop0AddNode ? toNodeRef(parsed, prop0AddNode) : null;
  if (prop0AddOut) assertInsideSuffix(prop0AddOut, 'prop0AddOut');

  let prop0SingletonArgMin: SecondaryOutlierAncestrySpec['prop0SingletonArgMin'] = null;
  for (const n of suffixSecondaryNodes) {
    if (n.opType !== 'ArgMin' || n.idx === coarseArgMin.nodeIdx) continue;
    const inMeta = n.inputs[0] ? parsed.staticMeta.tensorMeta[n.inputs[0]] : undefined;
    if (!inMeta || inMeta.dims.length === 0) continue;
    const rawAxis = getNodeIntAttr(n, 'axis', 0);
    const normAxis = rawAxis < 0 ? inMeta.dims.length + rawAxis : rawAxis;
    if ((inMeta.dims[normAxis] ?? 0) === 1) {
      const ref = toNodeRef(parsed, n);
      assertInsideSuffix(ref, 'prop0SingletonArgMin');
      prop0SingletonArgMin = { ...ref, axis: normAxis, inputDims: [...inMeta.dims] };
      break;
    }
  }

  // 4. postProp0Out: rank-4 [1,60,80,16] after prop0 (entering refinement_l0)
  const postProp0Node =
    suffixSecondaryNodes.find((n) => {
      if (n.idx <= (prop0AddOut?.nodeIdx ?? coarseArgMin.nodeIdx)) return false;
      const o = n.outputs[0];
      const m = o ? parsed.staticMeta.tensorMeta[o] : undefined;
      if (!m || m.elemType !== 1) return false;
      if (m.dims.join(',') !== '1,60,80,16') return false;
      return !n.name.includes('refinement_l0');
    }) ?? null;
  const postProp0Out = postProp0Node ? toNodeRef(parsed, postProp0Node) : null;
  if (postProp0Out) assertInsideSuffix(postProp0Out, 'postProp0Out');

  // 5. refinement_l0, refinement_l1, refinement_l2 checkpoints
  const findLastMatchingDims = (dimsStr: string, nameSubstr: string): SecondaryOutlierAncestryNodeRef | null => {
    const matches = suffixSecondaryNodes.filter((n) => {
      const o = n.outputs[0];
      const m = o ? parsed.staticMeta.tensorMeta[o] : undefined;
      return (
        Boolean(m && m.elemType === 1 && m.dims.join(',') === dimsStr) &&
        (n.name.includes(nameSubstr) || (o?.includes(nameSubstr) ?? false))
      );
    });
    if (matches.length === 0) return null;
    const ref = toNodeRef(parsed, matches[matches.length - 1]);
    assertInsideSuffix(ref, nameSubstr);
    return ref;
  };

  const refinementL0Out = findLastMatchingDims('1,120,160,16', 'refinement_l0');
  const refinementL1Out = findLastMatchingDims('1,240,320,16', 'refinement_l1');

  const refinementL2AddNode =
    suffixSecondaryNodes.find((n) => {
      if (n.opType !== 'Add') return false;
      const o = n.outputs[0];
      const m = o ? parsed.staticMeta.tensorMeta[o] : undefined;
      return Boolean(
        m &&
          m.elemType === 1 &&
          m.dims.join(',') === '1,16,240,320' &&
          (n.name.includes('refinement_l2') || (o?.includes('refinement_l2') ?? false))
      );
    }) ?? null;
  const refinementL2Add = refinementL2AddNode ? toNodeRef(parsed, refinementL2AddNode) : null;
  if (refinementL2Add) assertInsideSuffix(refinementL2Add, 'refinementL2Add');

  const refinementL2SlantAddNode =
    suffixSecondaryNodes.find((n) => {
      if (n.opType !== 'Add') return false;
      const o = n.outputs[0];
      const m = o ? parsed.staticMeta.tensorMeta[o] : undefined;
      return Boolean(
        m &&
          m.elemType === 1 &&
          m.dims.join(',') === '1,1,240,320' &&
          (n.name.includes('add_3') || n.name.includes('refinement_l2')) &&
          n.name !== secondaryOutName
      );
    }) ?? null;
  const refinementL2SlantAdd = refinementL2SlantAddNode ? toNodeRef(parsed, refinementL2SlantAddNode) : null;
  if (refinementL2SlantAdd) assertInsideSuffix(refinementL2SlantAdd, 'refinementL2SlantAdd');

  return {
    primaryAncestorCount: primaryAncestors.size,
    secondaryAncestorCount: secondaryAncestors.size,
    secondaryOnlyNodeCount: secondaryOnlyIndices.length,
    secondaryOnlyMinNodeIdx: secondaryOnlyIndices[0],
    secondaryOnlyMaxNodeIdx: secondaryOnlyIndices[secondaryOnlyIndices.length - 1],
    coarseArgMin,
    coarseCast,
    initToPropOut,
    propWarpOut,
    prop0AddOut,
    prop0SingletonArgMin,
    postProp0Out,
    refinementL0Out,
    refinementL1Out,
    refinementL2Add,
    refinementL2SlantAdd,
  };
}

// ---------- Helpers ----------

function copyOutputRawBytes(o: OrtVersionDiagnosticOutput): Uint8Array {
  return new Uint8Array(o.buffer).slice();
}

function hasObservedGpuDispatch(prof: OrtVersionDiagnosticResponse | null): boolean {
  if (!prof || !prof.ok || !prof.gpuKernelProfileSummary) return false;
  const s = prof.gpuKernelProfileSummary;
  return (
    s.webgpuComputePassCount > 0 ||
    s.webgpuDispatchWorkgroupsCount > 0 ||
    s.ortProfilingEventCount > 0
  );
}

function assertOrt130NonJsepSuccess(
  label: string,
  raw: OrtVersionDiagnosticResponse,
  expectedProvider: 'webgpu' | 'wasm',
  expectedModelSha: string,
  expectedModelBytes: number
): OrtVersionDiagnosticSuccess {
  if (!raw.ok) {
    const f = raw as OrtVersionDiagnosticFailure;
    throw new Error(`${label} FAILED at stage="${f.stage}": ${f.message}`);
  }
  if (
    raw.pinnedOrtVersion !== '1.30.0' ||
    raw.runtimeAssetFamily !== 'asyncify' ||
    raw.provider !== expectedProvider ||
    raw.graphOptimizationLevel !== 'disabled' ||
    raw.sessionOptionsApplied.numThreads !== 1 ||
    raw.sessionOptionsApplied.proxy !== false ||
    raw.foreignOrtResources.length > 0 ||
    raw.loadedOrtResources.some((u) => u.includes('.jsep.') || u.includes('ort.bundle.'))
  ) {
    throw new Error(
      `${label} ORT 1.30.0 non-JSEP / opt=disabled / 1-thread guard failed: opt=${raw.graphOptimizationLevel}, ep=${raw.provider}, threads=${raw.sessionOptionsApplied.numThreads}, proxy=${raw.sessionOptionsApplied.proxy}, foreign=${JSON.stringify(raw.foreignOrtResources)}`
    );
  }
  if (raw.modelSha256 !== expectedModelSha || raw.modelByteLength !== expectedModelBytes) {
    throw new Error(
      `${label} model verification failed: got=${raw.modelSha256}(${raw.modelByteLength}B), expected=${expectedModelSha}(${expectedModelBytes}B)`
    );
  }
  return raw;
}

function computeEyeValidityAndDepthFlips(params: {
  eye: 'left-primary' | 'right-secondary';
  tensorName: string;
  nativeData: Float32Array;
  candData: Float32Array;
  wasmData: Float32Array;
}): EyeValidityAndDepthFlipSummary {
  const { eye, tensorName, nativeData, candData, wasmData } = params;
  const n = nativeData.length;
  const dims = [1, 240, 320, 1] as const;
  const cal = HITNET_RESIDUAL_SOURCE_CALIBRATION_DEFAULT;
  const computeNeg = (arr: Float32Array) => {
    let neg = 0;
    let pos = 0;
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (Number.isFinite(v)) {
        if (v < 0) neg -= v;
        else pos += v;
      }
    }
    return neg > pos * 1.1;
  };
  const natExt = extractDisparityFromOrtTensor({ data: nativeData, dims });
  const candExt = extractDisparityFromOrtTensor({ data: candData, dims });
  const wasmExt = extractDisparityFromOrtTensor({ data: wasmData, dims });
  const scale = cal.sourceEyeWidth / natExt.width;

  let nativeValidCount = 0;
  let candValidCount = 0;
  let wasmValidCount = 0;
  let nativeValidToCandInvalid = 0;
  let nativeInvalidToCandValid = 0;
  let wasmValidToCandInvalid = 0;
  let wasmInvalidToCandValid = 0;
  let depthThresholdFlipsVsNative = 0;
  let depthThresholdFlipsVsWasm = 0;

  for (let i = 0; i < n; i++) {
    const nv = natExt.disparity[i] >= 0;
    const cv = candExt.disparity[i] >= 0;
    const wv = wasmExt.disparity[i] >= 0;
    if (nv) nativeValidCount++;
    if (cv) candValidCount++;
    if (wv) wasmValidCount++;
    if (nv && !cv) nativeValidToCandInvalid++;
    if (!nv && cv) nativeInvalidToCandValid++;
    if (wv && !cv) wasmValidToCandInvalid++;
    if (!wv && cv) wasmInvalidToCandValid++;

    const nd = nv ? disparityToDepthMeters(natExt.disparity[i] * scale, cal.baselineMm, cal.focalLengthPx) !== null : false;
    const cd = cv ? disparityToDepthMeters(candExt.disparity[i] * scale, cal.baselineMm, cal.focalLengthPx) !== null : false;
    const wd = wv ? disparityToDepthMeters(wasmExt.disparity[i] * scale, cal.baselineMm, cal.focalLengthPx) !== null : false;
    if (nd !== cd) depthThresholdFlipsVsNative++;
    if (wd !== cd) depthThresholdFlipsVsWasm++;
  }

  return {
    eye,
    tensorName,
    totalPixels: n,
    negateNative: computeNeg(nativeData),
    negateCand: computeNeg(candData),
    negateWasm: computeNeg(wasmData),
    nativeValidCount,
    candValidCount,
    wasmValidCount,
    nativeValidToCandInvalid,
    nativeInvalidToCandValid,
    totalValidityFlipsVsNative: nativeValidToCandInvalid + nativeInvalidToCandValid,
    wasmValidToCandInvalid,
    wasmInvalidToCandValid,
    totalValidityFlipsVsWasm: wasmValidToCandInvalid + wasmInvalidToCandValid,
    depthThresholdFlipsVsNative,
    depthThresholdFlipsVsWasm,
  };
}

function clusterSecondaryOutliers(params: {
  nativeSecondary: Float32Array;
  candSecondary: Float32Array;
  wasmSecondary: Float32Array;
  width?: number;
  height?: number;
}): {
  thresholdUsed: number;
  outlierPixelCount: number;
  clusters: SecondaryOutlierClusterSummary[];
  topPixels: SecondaryOutlierPixelCoord[];
} {
  const width = params.width ?? 320;
  const height = params.height ?? 240;
  const total = width * height;
  const { nativeSecondary, candSecondary, wasmSecondary } = params;

  let maxAbs = 0;
  const topList: Array<{ idx: number; abs: number }> = [];
  for (let i = 0; i < total; i++) {
    const abs = Math.abs(candSecondary[i] - nativeSecondary[i]);
    if (abs > maxAbs) maxAbs = abs;
    if (abs > 0 && (topList.length < 12 || abs > topList[topList.length - 1].abs)) {
      topList.push({ idx: i, abs });
      topList.sort((a, b) => b.abs - a.abs);
      if (topList.length > 12) topList.pop();
    }
  }

  const mkPixelCoord = (rank: number, flatIdx: number): SecondaryOutlierPixelCoord => {
    const y = Math.floor(flatIdx / width);
    const x = flatIdx % width;
    const nat = nativeSecondary[flatIdx];
    const cand = candSecondary[flatIdx];
    const wasm = wasmSecondary[flatIdx];
    return {
      rank,
      flatIdx,
      y,
      x,
      ty: Math.floor(y / 4),
      tx: Math.floor(x / 4),
      nativeVal: nat,
      candVal: cand,
      wasmVal: wasm,
      signedDiffVsNative: cand - nat,
      absDiffVsNative: Math.abs(cand - nat),
      signedDiffVsWasm: cand - wasm,
      absDiffVsWasm: Math.abs(cand - wasm),
    };
  };

  const topPixels = topList.map((item, k) => mkPixelCoord(k + 1, item.idx));
  const thresholdUsed = maxAbs > 0.1 ? 0.1 : maxAbs > 1e-3 ? 1e-3 : 0;
  if (thresholdUsed === 0 || maxAbs === 0) {
    return { thresholdUsed, outlierPixelCount: 0, clusters: [], topPixels };
  }

  const visited = new Uint8Array(total);
  const clusters: SecondaryOutlierClusterSummary[] = [];
  let outlierPixelCount = 0;

  for (let i = 0; i < total; i++) {
    if (visited[i]) continue;
    const abs0 = Math.abs(candSecondary[i] - nativeSecondary[i]);
    if (abs0 <= thresholdUsed) continue;

    const queue: number[] = [i];
    visited[i] = 1;
    let qPtr = 0;
    let pixelCount = 0;
    let minY = height;
    let maxY = -1;
    let minX = width;
    let maxX = -1;
    let minTy = Math.floor(height / 4);
    let maxTy = -1;
    let minTx = Math.floor(width / 4);
    let maxTx = -1;
    let peakIdx = i;
    let peakAbs = -1;
    const tilesSet = new Set<number>();

    while (qPtr < queue.length) {
      const cur = queue[qPtr++];
      pixelCount++;
      outlierPixelCount++;
      const y = Math.floor(cur / width);
      const x = cur % width;
      const ty = Math.floor(y / 4);
      const tx = Math.floor(x / 4);
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (ty < minTy) minTy = ty;
      if (ty > maxTy) maxTy = ty;
      if (tx < minTx) minTx = tx;
      if (tx > maxTx) maxTx = tx;
      tilesSet.add(ty * 80 + tx);

      const curAbs = Math.abs(candSecondary[cur] - nativeSecondary[cur]);
      if (curAbs > peakAbs) {
        peakAbs = curAbs;
        peakIdx = cur;
      }

      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          const nIdx = ny * width + nx;
          if (visited[nIdx]) continue;
          if (Math.abs(candSecondary[nIdx] - nativeSecondary[nIdx]) > thresholdUsed) {
            visited[nIdx] = 1;
            queue.push(nIdx);
          }
        }
      }
    }

    const sampleTiles = Array.from(tilesSet)
      .slice(0, 12)
      .map((t) => ({ ty: Math.floor(t / 80), tx: t % 80 }));

    clusters.push({
      clusterId: 0,
      pixelCount,
      bbox: { minY, maxY, minX, maxX },
      tileBbox: { minTy, maxTy, minTx, maxTx },
      uniqueTileCount: tilesSet.size,
      sampleTiles,
      peak: mkPixelCoord(1, peakIdx),
    });
  }

  clusters.sort((a, b) => b.peak.absDiffVsNative - a.peak.absDiffVsNative || b.pixelCount - a.pixelCount);
  const topClusters = clusters.slice(0, 10).map((c, idx) => ({ ...c, clusterId: idx + 1 }));

  return {
    thresholdUsed,
    outlierPixelCount,
    clusters: topClusters,
    topPixels,
  };
}

function readInt64OrNumberArray(out: OrtVersionDiagnosticOutput): BigInt64Array {
  if (out.dataType === 'int64') {
    return new BigInt64Array(out.buffer);
  }
  if (out.dataType === 'int32') {
    const i32 = new Int32Array(out.buffer);
    return BigInt64Array.from(i32, (v) => BigInt(v));
  }
  const f32 = new Float32Array(out.buffer);
  return BigInt64Array.from(f32, (v) => BigInt(Math.trunc(v)));
}

function readFloat32ArrayCopy(out: OrtVersionDiagnosticOutput): Float32Array {
  if (out.dataType === 'float32') {
    return new Float32Array(out.buffer).slice();
  }
  if (out.dataType === 'int64') {
    return Float32Array.from(new BigInt64Array(out.buffer), Number);
  }
  if (out.dataType === 'int32') {
    return Float32Array.from(new Int32Array(out.buffer), Number);
  }
  return new Float32Array(out.buffer).slice();
}

function computeTileCostOffsetHelper(
  dims: readonly number[],
  axis: number
): {
  tileCount: number;
  depthCount: number;
  getFlatIdx: (ty: number, tx: number, d: number) => number;
} {
  const rank = dims.length;
  const normAxis = axis < 0 ? rank + axis : axis;
  const depthCount = dims[normAxis] ?? 320;

  if (rank === 4 && dims[0] === 1 && dims[1] === 60 && dims[2] === 80 && normAxis === 3) {
    return {
      tileCount: 4800,
      depthCount,
      getFlatIdx: (ty, tx, d) => ((ty * 80 + tx) * depthCount) + d,
    };
  }

  // Generic strides
  const strides = new Array<number>(rank).fill(1);
  for (let i = rank - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  const axisStride = strides[normAxis];
  // Map remaining non-1 dims to (ty, tx)
  const spatialDims: number[] = [];
  for (let i = 0; i < rank; i++) {
    if (i !== normAxis && dims[i] > 1) spatialDims.push(i);
  }
  const hIdx = spatialDims[0] ?? 1;
  const wIdx = spatialDims[1] ?? 2;
  const hDim = dims[hIdx] ?? 60;
  const wDim = dims[wIdx] ?? 80;
  const hStride = strides[hIdx] ?? 0;
  const wStride = strides[wIdx] ?? 0;

  return {
    tileCount: hDim * wDim,
    depthCount,
    getFlatIdx: (ty, tx, d) => ty * hStride + tx * wStride + d * axisStride,
  };
}

function computeExactOnnxArgMinForVector(
  costs: Float32Array,
  selectLastIndex: number
): {
  finiteCount: number;
  minCost: number;
  tieIndices: number[];
  winner: number | null;
} {
  let minCost = Infinity;
  let finiteCount = 0;
  for (let d = 0; d < costs.length; d++) {
    const v = costs[d];
    if (!Number.isFinite(v)) continue;
    finiteCount++;
    if (v < minCost) minCost = v;
  }
  if (finiteCount === 0) {
    return { finiteCount: 0, minCost: Infinity, tieIndices: [], winner: null };
  }
  const tieIndices: number[] = [];
  for (let d = 0; d < costs.length; d++) {
    const v = costs[d];
    if (Number.isFinite(v) && v === minCost) {
      tieIndices.push(d);
    }
  }
  const winner =
    tieIndices.length === 0
      ? null
      : selectLastIndex === 1
        ? tieIndices[tieIndices.length - 1]
        : tieIndices[0];
  return { finiteCount, minCost, tieIndices, winner };
}

function computeRoiTensorDiff(
  cand: Float32Array,
  ref: Float32Array,
  dims: readonly number[],
  tileBbox: { minTy: number; maxTy: number; minTx: number; maxTx: number } | null
): { roiMaxAbsDiff: number | null; roiMeanAbsDiff: number | null } {
  if (!tileBbox || dims.length < 3 || cand.length !== ref.length) {
    return { roiMaxAbsDiff: null, roiMeanAbsDiff: null };
  }
  // Support [1, H, W, C], [1, H, W, 1, C], [1, C, H, W]
  let h: number;
  let w: number;
  let layout: 'nhwc' | 'nchw';
  let c: number;
  if (dims.length === 4 && (dims[1] === 60 || dims[1] === 120 || dims[1] === 240)) {
    h = dims[1];
    w = dims[2];
    c = dims[3];
    layout = 'nhwc';
  } else if (dims.length === 5 && (dims[1] === 60 || dims[1] === 120 || dims[1] === 240)) {
    h = dims[1];
    w = dims[2];
    c = dims[3] * dims[4];
    layout = 'nhwc';
  } else if (dims.length === 4 && (dims[2] === 60 || dims[2] === 120 || dims[2] === 240)) {
    c = dims[1];
    h = dims[2];
    w = dims[3];
    layout = 'nchw';
  } else {
    return { roiMaxAbsDiff: null, roiMeanAbsDiff: null };
  }

  const scaleY = h / 60;
  const scaleX = w / 80;
  const y0 = Math.max(0, Math.floor(tileBbox.minTy * scaleY));
  const y1 = Math.min(h - 1, Math.ceil((tileBbox.maxTy + 1) * scaleY) - 1);
  const x0 = Math.max(0, Math.floor(tileBbox.minTx * scaleX));
  const x1 = Math.min(w - 1, Math.ceil((tileBbox.maxTx + 1) * scaleX) - 1);

  let maxD = 0;
  let sumD = 0;
  let count = 0;
  if (layout === 'nhwc') {
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const base = (y * w + x) * c;
        for (let ch = 0; ch < c; ch++) {
          const d = Math.abs(cand[base + ch] - ref[base + ch]);
          if (d > maxD) maxD = d;
          sumD += d;
          count++;
        }
      }
    }
  } else {
    const hw = h * w;
    for (let ch = 0; ch < c; ch++) {
      const cOff = ch * hw;
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const d = Math.abs(cand[cOff + y * w + x] - ref[cOff + y * w + x]);
          if (d > maxD) maxD = d;
          sumD += d;
          count++;
        }
      }
    }
  }
  return {
    roiMaxAbsDiff: count > 0 ? maxD : null,
    roiMeanAbsDiff: count > 0 ? sumD / count : null,
  };
}

// ---------- Main Orchestrator ----------

export async function runHitnetF0126SecondaryOutlierDiagnostic(params: {
  m1Bytes: Uint8Array;
  f0126FeedFloat32: Float32Array;
  normalizeSlot0ReshapeToTranspose?: false;
  existingCommonWasm7Summary?: Stage2RealFrameInjectionReport['causalCommonWasm7Controls'] | null;
  runWorker?: ResidualDiagnosticWorkerRunner;
  disableCpuEpFallback?: boolean;
  isDev?: boolean;
  onLog?: (line: string) => void;
}): Promise<HitnetF0126SecondaryOutlierReport> {
  assertDevFixtureEnabled(params.isDev);
  const log = params.onLog ?? (() => {});
  const runWorker = params.runWorker ?? createBrowserOrt130DiagnosticRunner();
  const disableCpuEpFallback = Boolean(params.disableCpuEpFallback);

  if ((params as { normalizeSlot0ReshapeToTranspose?: boolean }).normalizeSlot0ReshapeToTranspose) {
    throw new Error('N1-Slot0Transpose MUST be explicitly OFF (false) for F0126 secondary outlier diagnostic.');
  }

  // 1. Canonical M1 & Frozen F0126 SHA checks
  const m1Sha = await sha256HexBytes(params.m1Bytes);
  if (
    params.m1Bytes.byteLength !== HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength ||
    m1Sha !== HITNET_M1_FULL_GRAPH_SHA256
  ) {
    throw new Error(
      `Exact canonical M1 verification failed: bytes=${params.m1Bytes.byteLength} (expected ${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength}), sha=${m1Sha} (expected ${HITNET_M1_FULL_GRAPH_SHA256})`
    );
  }

  const f0126Sha = await sha256HexBytes(
    new Uint8Array(
      params.f0126FeedFloat32.buffer,
      params.f0126FeedFloat32.byteOffset,
      params.f0126FeedFloat32.byteLength
    )
  );
  if (f0126Sha !== HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256) {
    throw new Error(
      `Frozen F0126 feed SHA mismatch: expected=${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256}, actual=${f0126Sha}`
    );
  }

  const parsedModel = await parseM1ModelProtoWithRawSlices(params.m1Bytes);
  const n1OffProof = verifyCanonicalM1NoN1Replacements(parsedModel, params.m1Bytes, m1Sha, false);
  if (!n1OffProof.verified) {
    throw new Error(n1OffProof.failureReason ?? 'Canonical M1 N1=OFF verification failed');
  }

  const chainSpec = await resolveVerifiedPostConv7ChainSpec(parsedModel);
  const suffix9 = await buildVerifiedSuffixAfterConv7(parsedModel, {
    cutNodeIdx: chainSpec.node9ConvDown0.nodeIdx,
  });
  const node8OutName = chainSpec.node8LeakyRelu.outputTensor;
  const prefixOutputsNeeded = Array.from(
    new Set([node8OutName, ...suffix9.bypassCutInputs.map((b) => b.name)])
  );
  const prefix0to8: VerifiedPrefixThroughNode = await buildVerifiedPrefixThroughNode(
    parsedModel,
    chainSpec.node8LeakyRelu.nodeIdx,
    prefixOutputsNeeded
  );
  const singleOp9 = buildSingleOpCapturedInputModel(
    parsedModel,
    parsedModel.nodes[chainSpec.node9ConvDown0.nodeIdx]
  );
  const singleOp9Sha = await sha256HexBytes(singleOp9.bytes);

  const [pName, sName] = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;
  const asyncify = HITNET_ORT_130_FAMILY_PROFILES.asyncify;

  log(
    `=== [DEV-ONLY] HITNet F0126 #9 Transport Identity Gate & Secondary Disparity Outlier Cause ===\n` +
      `Runtime target: onnxruntime-web@1.30.0/webgpu (non-JSEP ${asyncify.entryFile} + ${asyncify.wasmFile}) | opt=disabled | numThreads=1 | proxy=false\n` +
      `Canonical M1: bytes=${params.m1Bytes.byteLength} sha=${m1Sha} | N1=OFF (0 replacements)\n` +
      `Frozen F0126 feed: sha=${f0126Sha}\n` +
      `Prefix (#0..#${prefix0to8.cutNodeIdx}): sha=${prefix0to8.prefixSha256.slice(0, 12)}... | Suffix (#${suffix9.suffixNodeRange.firstOriginalIdx}..#${suffix9.suffixNodeRange.lastOriginalIdx}): sha=${suffix9.suffixSha256.slice(0, 12)}... inputs=${JSON.stringify(suffix9.graphInputNames)}`
  );

  // Helper to run simple verified model (full M1 / prefix / singleOp9 / direct split)
  const runSimpleOrt130 = async (opts: {
    label: string;
    modelBytes: Uint8Array;
    modelSha: string;
    provider: 'webgpu' | 'wasm';
    feedName: string;
    feedDims: number[];
    feedF32: Float32Array;
    feedSha: string;
    extraFeeds?: Array<{ name: string; dims: number[]; buffer: ArrayBuffer; dataType: 'float32' | 'int32' | 'int64' | 'uint8' }>;
    allowArbitraryOutputs?: boolean;
    runSeparateGpuProfilePass?: boolean;
  }) => {
    const raw = await runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: opts.modelBytes.slice().buffer,
      feedBuffer: opts.feedF32.slice().buffer,
      feedName: opts.feedName,
      feedDims: [...opts.feedDims],
      extraFeeds: opts.extraFeeds?.map((e) => ({ ...e, buffer: e.buffer.slice(0) })),
      allowArbitraryOutputs: opts.allowArbitraryOutputs,
      provider: opts.provider,
      graphOptimizationLevel: 'disabled',
      warmRuns: 0,
      enableWebGpuProfiling: false,
      disableCpuEpFallback: opts.provider === 'webgpu' ? disableCpuEpFallback : false,
      captureGpuValidationErrors: opts.provider === 'webgpu',
    });
    const okRes = assertOrt130NonJsepSuccess(
      opts.label,
      raw,
      opts.provider,
      opts.modelSha,
      opts.modelBytes.byteLength
    );
    if (okRes.feedSha256 !== opts.feedSha) {
      throw new Error(`${opts.label} feed SHA mismatch: got=${okRes.feedSha256}, expected=${opts.feedSha}`);
    }
    let dispatchObserved = false;
    if (opts.runSeparateGpuProfilePass && opts.provider === 'webgpu') {
      const prof = await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: opts.modelBytes.slice().buffer,
        feedBuffer: opts.feedF32.slice().buffer,
        feedName: opts.feedName,
        feedDims: [...opts.feedDims],
        extraFeeds: opts.extraFeeds?.map((e) => ({ ...e, buffer: e.buffer.slice(0) })),
        allowArbitraryOutputs: opts.allowArbitraryOutputs,
        provider: 'webgpu',
        graphOptimizationLevel: 'disabled',
        warmRuns: 0,
        enableWebGpuProfiling: true,
        disableCpuEpFallback,
        captureGpuValidationErrors: true,
      });
      dispatchObserved = hasObservedGpuDispatch(prof);
    }
    return { res: okRes, dispatchObserved };
  };

  // 2. Full Untapped Baselines (F0126, opt=disabled)
  const fullGpuRun = await runSimpleOrt130({
    label: 'F0126 Untapped Full M1 WebGPU (opt=disabled)',
    modelBytes: params.m1Bytes,
    modelSha: m1Sha,
    provider: 'webgpu',
    feedName: 'input',
    feedDims: [1, 6, 240, 320],
    feedF32: params.f0126FeedFloat32,
    feedSha: f0126Sha,
    runSeparateGpuProfilePass: true,
  });
  const fullGpuVal = validateHitnetFinalOutputs(fullGpuRun.res.outputs);
  if (!fullGpuVal.ok) {
    throw new Error(`F0126 Full GPU (opt=disabled) invalid outputs: ${fullGpuVal.error}`);
  }

  const fullWasmRun = await runSimpleOrt130({
    label: 'F0126 Untapped Full M1 WASM (opt=disabled)',
    modelBytes: params.m1Bytes,
    modelSha: m1Sha,
    provider: 'wasm',
    feedName: 'input',
    feedDims: [1, 6, 240, 320],
    feedF32: params.f0126FeedFloat32,
    feedSha: f0126Sha,
  });
  const fullWasmVal = validateHitnetFinalOutputs(fullWasmRun.res.outputs);
  if (!fullWasmVal.ok) {
    throw new Error(`F0126 Full WASM (opt=disabled) invalid outputs: ${fullWasmVal.error}`);
  }

  // 3. One concrete GPU Prefix Capture (#0..#8) + Bypass Extraction & Hashing
  const gpuPrefixRun = await runSimpleOrt130({
    label: 'F0126 Concrete GPU Prefix (#0..#8)',
    modelBytes: prefix0to8.prefixBytes,
    modelSha: prefix0to8.prefixSha256,
    provider: 'webgpu',
    feedName: 'input',
    feedDims: [1, 6, 240, 320],
    feedF32: params.f0126FeedFloat32,
    feedSha: f0126Sha,
    allowArbitraryOutputs: true,
    runSeparateGpuProfilePass: true,
  });
  const prefixNode8Out = gpuPrefixRun.res.outputs.find((o) => o.name === node8OutName);
  if (!prefixNode8Out) {
    throw new Error(`F0126 GPU prefix missing #8 output "${node8OutName}"`);
  }
  const prefixNode8F32 = new Float32Array(prefixNode8Out.buffer).slice();
  const prefixNode8Sha = await sha256HexBytes(
    new Uint8Array(prefixNode8F32.buffer, prefixNode8F32.byteOffset, prefixNode8F32.byteLength)
  );

  const prefixBypassList: Array<{
    name: string;
    dims: number[];
    dataType: 'float32' | 'int32' | 'int64' | 'uint8';
    rawBytes: Uint8Array;
    byteLength: number;
    sha256: string;
  }> = [];
  for (const bp of suffix9.bypassCutInputs) {
    const out = gpuPrefixRun.res.outputs.find((o) => o.name === bp.name);
    if (!out) {
      throw new Error(`F0126 GPU prefix missing required suffix9 bypass "${bp.name}"`);
    }
    const rawBytes = copyOutputRawBytes(out);
    if (rawBytes.byteLength !== bp.byteLength) {
      throw new Error(
        `F0126 GPU prefix bypass "${bp.name}" byteLength mismatch: got=${rawBytes.byteLength}, expected=${bp.byteLength}`
      );
    }
    const sha = await sha256HexBytes(rawBytes);
    prefixBypassList.push({
      name: bp.name,
      dims: [...bp.dims],
      dataType: bp.dataTypeName,
      rawBytes,
      byteLength: rawBytes.byteLength,
      sha256: sha,
    });
  }

  // 4. Native Split Control: singleOp9(prefixNode8) -> direct suffix9(gpuOneOp9Raw)
  const gpuOneOp9Run = await runSimpleOrt130({
    label: 'F0126 Native 1-Op #9 WebGPU (opt=disabled)',
    modelBytes: singleOp9.bytes,
    modelSha: singleOp9Sha,
    provider: 'webgpu',
    feedName: singleOp9.dynamicInputSpecs[0].name,
    feedDims: [...singleOp9.dynamicInputSpecs[0].dims],
    feedF32: prefixNode8F32,
    feedSha: prefixNode8Sha,
    allowArbitraryOutputs: true,
    runSeparateGpuProfilePass: true,
  });
  const gpuOneOp9Out = gpuOneOp9Run.res.outputs[0];
  if (!gpuOneOp9Out) {
    throw new Error('F0126 Native 1-Op #9 produced no outputs');
  }
  const gpuOneOp9RawF32 = new Float32Array(gpuOneOp9Out.buffer).slice();
  const gpuOneOp9RawSha = await sha256HexBytes(
    new Uint8Array(gpuOneOp9RawF32.buffer, gpuOneOp9RawF32.byteOffset, gpuOneOp9RawF32.byteLength)
  );

  const gpuSuffix9SplitRun = await runSimpleOrt130({
    label: 'F0126 Known Native GPU Split Suffix (#10..end)',
    modelBytes: suffix9.suffixBytes,
    modelSha: suffix9.suffixSha256,
    provider: 'webgpu',
    feedName: suffix9.primaryCutInput.name,
    feedDims: [...suffix9.primaryCutInput.dims],
    feedF32: gpuOneOp9RawF32,
    feedSha: gpuOneOp9RawSha,
    extraFeeds: prefixBypassList.map((b) => ({
      name: b.name,
      dims: [...b.dims],
      buffer: b.rawBytes.slice().buffer as ArrayBuffer,
      dataType: b.dataType,
    })),
  });
  const gpuSuffix9SplitVal = validateHitnetFinalOutputs(gpuSuffix9SplitRun.res.outputs);
  if (!gpuSuffix9SplitVal.ok) {
    throw new Error(`F0126 Known Native GPU Split invalid outputs: ${gpuSuffix9SplitVal.error}`);
  }
  const gpuSplitVsFullP = checkFloat32BitIdentity(
    gpuSuffix9SplitVal.byName[pName].data,
    fullGpuVal.byName[pName].data
  );
  const gpuSplitVsFullS = checkFloat32BitIdentity(
    gpuSuffix9SplitVal.byName[sName].data,
    fullGpuVal.byName[sName].data
  );
  const nativeSplitPassed = gpuSplitVsFullP.bitIdentical && gpuSplitVsFullS.bitIdentical;

  log(
    `  [Native Split Control] gpuSplitVsFullGpu: P=${gpuSplitVsFullP.bitIdentical}(mismatch=${gpuSplitVsFullP.mismatchCount}) S=${gpuSplitVsFullS.bitIdentical}(mismatch=${gpuSplitVsFullS.mismatchCount}) | prefixNode8Sha=${prefixNode8Sha.slice(0, 12)}... gpu#9Sha=${gpuOneOp9RawSha.slice(0, 12)}...`
  );

  if (!nativeSplitPassed) {
    throw new Error(
      `Known Native GPU Split does not match full untapped GPU bit-for-bit (P=${gpuSplitVsFullP.mismatchCount}, S=${gpuSplitVsFullS.mismatchCount})`
    );
  }

  // 5. Shared Host-Allocation / Copy / Byte-Extraction / Transfer / Worker-Reinsertion Routine
  const executeSuffixViaSharedHostTransport = async (opts: {
    label: string;
    mode: HostConv9TransportMode;
    modelBytes: Uint8Array;
    modelSha: string;
    expectedInputOrder: string[];
    primaryInputSpec: { name: string; dims: number[]; byteLength: number };
    bypassInputList: typeof prefixBypassList;
    xDataNode8?: Float32Array;
    runSeparateProfilePass?: boolean;
  }): Promise<SharedHostTransportExecutionResult> => {
    const outDims = opts.primaryInputSpec.dims as [number, number, number, number];
    const primaryHostF32 = populateFreshOwnedHostConv9Buffer({
      mode: opts.mode,
      nativeGpu9F32: opts.mode === 'native-gpu9-copy' ? gpuOneOp9RawF32 : undefined,
      parsed: opts.mode === 'blocked-k128' ? parsedModel : undefined,
      convNodeIdx: opts.mode === 'blocked-k128' ? chainSpec.node9ConvDown0.nodeIdx : undefined,
      xData: opts.mode === 'blocked-k128' ? (opts.xDataNode8 ?? prefixNode8F32) : undefined,
      inputDims:
        opts.mode === 'blocked-k128'
          ? (singleOp9.dynamicInputSpecs[0].dims as [number, number, number, number])
          : undefined,
      outDims,
    });

    const primaryExtractedBytes = new Uint8Array(
      primaryHostF32.buffer,
      primaryHostF32.byteOffset,
      primaryHostF32.byteLength
    );
    const primaryShaBefore = await sha256HexBytes(primaryExtractedBytes);

    // Prepare owned contiguous copies of every bypass edge in exact expected order
    const orderedBypasses: Array<{
      spec: (typeof prefixBypassList)[number];
      ownedBuffer: ArrayBuffer;
      byteOffsetBefore: number;
      byteLengthBefore: number;
      shaBefore: string;
      orderIndexExpected: number;
    }> = [];

    for (let i = 0; i < opts.bypassInputList.length; i++) {
      const bp = opts.bypassInputList[i];
      const freshU8 = new Uint8Array(bp.byteLength);
      freshU8.set(bp.rawBytes);
      const shaBefore = await sha256HexBytes(freshU8);
      const expectedIdx = opts.expectedInputOrder.indexOf(bp.name);
      orderedBypasses.push({
        spec: bp,
        ownedBuffer: freshU8.buffer,
        byteOffsetBefore: freshU8.byteOffset,
        byteLengthBefore: freshU8.byteLength,
        shaBefore,
        orderIndexExpected: expectedIdx >= 0 ? expectedIdx : i + 1,
      });
    }

    const edges: HostConv9TransportEdgeRecord[] = [
      {
        name: opts.primaryInputSpec.name,
        role: 'primary-cut-9',
        dims: [...opts.primaryInputSpec.dims],
        dataType: 'float32',
        byteOffsetBefore: primaryHostF32.byteOffset,
        byteLengthBefore: primaryHostF32.byteLength,
        sha256Before: primaryShaBefore,
        byteOffsetAfter: null,
        byteLengthAfter: null,
        sha256After: null,
        orderIndexExpected: opts.expectedInputOrder.indexOf(opts.primaryInputSpec.name),
        orderIndexActual: null,
        matchesAllChecks: false,
      },
      ...orderedBypasses.map((b) => ({
        name: b.spec.name,
        role: 'bypass' as const,
        dims: [...b.spec.dims],
        dataType: b.spec.dataType,
        byteOffsetBefore: b.byteOffsetBefore,
        byteLengthBefore: b.byteLengthBefore,
        sha256Before: b.shaBefore,
        byteOffsetAfter: null,
        byteLengthAfter: null,
        sha256After: null,
        orderIndexExpected: b.orderIndexExpected,
        orderIndexActual: null,
        matchesAllChecks: false,
      })),
    ];

    const mkFailResult = (
      firstFailure: TransportFailureDetail,
      actualInputOrder: string[] | null = null,
      workerSuccess: OrtVersionDiagnosticSuccess | null = null
    ): SharedHostTransportExecutionResult => ({
      ok: false,
      label: opts.label,
      mode: opts.mode,
      suffixSha256: opts.modelSha,
      suffixByteLength: opts.modelBytes.byteLength,
      expectedInputOrder: [...opts.expectedInputOrder],
      actualInputOrder,
      edges,
      firstFailure,
      workerSuccess,
      dispatchObservedSeparatePass: null,
      primaryF32: null,
      secondaryF32: null,
      outputsByName: {},
    });

    // Pre-transfer contiguous / zero-offset / byteLength assertions
    if (
      primaryHostF32.byteOffset !== 0 ||
      primaryHostF32.byteLength !== opts.primaryInputSpec.byteLength ||
      primaryHostF32.buffer.byteLength !== primaryHostF32.byteLength
    ) {
      return mkFailResult({
        checkId: 'pre-primary-contiguous',
        stage: 'pre-transfer',
        edgeName: opts.primaryInputSpec.name,
        detail: 'Primary #9 host buffer is not zero-offset owned contiguous',
        expected: `offset=0, bytes=${opts.primaryInputSpec.byteLength}, backing=${opts.primaryInputSpec.byteLength}`,
        actual: `offset=${primaryHostF32.byteOffset}, bytes=${primaryHostF32.byteLength}, backing=${primaryHostF32.buffer.byteLength}`,
      });
    }
    for (const bp of orderedBypasses) {
      if (bp.byteOffsetBefore !== 0 || bp.byteLengthBefore !== bp.spec.byteLength) {
        return mkFailResult({
          checkId: 'pre-bypass-contiguous',
          stage: 'pre-transfer',
          edgeName: bp.spec.name,
          detail: 'Bypass host buffer is not zero-offset exact byteLength',
          expected: `offset=0, bytes=${bp.spec.byteLength}`,
          actual: `offset=${bp.byteOffsetBefore}, bytes=${bp.byteLengthBefore}`,
        });
      }
      if (bp.shaBefore !== bp.spec.sha256) {
        return mkFailResult({
          checkId: 'pre-bypass-sha',
          stage: 'pre-transfer',
          edgeName: bp.spec.name,
          detail: 'Bypass copy SHA-256 changed before worker transfer',
          expected: bp.spec.sha256,
          actual: bp.shaBefore,
        });
      }
    }

    const reqExtraFeeds = orderedBypasses.map((b) => ({
      name: b.spec.name,
      dims: [...b.spec.dims],
      buffer: b.ownedBuffer.slice(0),
      dataType: b.spec.dataType,
    }));

    const workerResRaw = await runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: opts.modelBytes.slice().buffer,
      feedBuffer: primaryHostF32.slice().buffer,
      feedName: opts.primaryInputSpec.name,
      feedDims: [...opts.primaryInputSpec.dims],
      extraFeeds: reqExtraFeeds,
      allowArbitraryOutputs: false,
      provider: 'webgpu',
      graphOptimizationLevel: 'disabled',
      warmRuns: 0,
      enableWebGpuProfiling: false,
      disableCpuEpFallback,
      captureGpuValidationErrors: true,
    });

    if (!workerResRaw.ok) {
      const f = workerResRaw as OrtVersionDiagnosticFailure;
      return mkFailResult({
        checkId: 'worker-run-ok',
        stage: 'worker-execution',
        edgeName: null,
        detail: `Worker run failed at stage="${f.stage}": ${f.message}`,
        expected: 'ok=true (ORT 1.30.0 non-JSEP WebGPU opt=disabled)',
        actual: `ok=false (${f.stage}: ${f.message})`,
      });
    }

    const res = workerResRaw;
    if (
      res.pinnedOrtVersion !== '1.30.0' ||
      res.runtimeAssetFamily !== 'asyncify' ||
      res.provider !== 'webgpu' ||
      res.graphOptimizationLevel !== 'disabled' ||
      res.sessionOptionsApplied.numThreads !== 1 ||
      res.sessionOptionsApplied.proxy !== false ||
      res.foreignOrtResources.length > 0 ||
      res.loadedOrtResources.some((u) => u.includes('.jsep.') || u.includes('ort.bundle.'))
    ) {
      return mkFailResult(
        {
          checkId: 'worker-runtime-guard',
          stage: 'worker-execution',
          edgeName: null,
          detail: 'Worker runtime asset family / opt / threads / fallback guard violated',
          expected: '1.30.0/asyncify/webgpu/disabled/threads=1/proxy=false',
          actual: `${res.pinnedOrtVersion}/${res.runtimeAssetFamily}/${res.provider}/${res.graphOptimizationLevel}/threads=${res.sessionOptionsApplied.numThreads}/proxy=${res.sessionOptionsApplied.proxy}`,
        },
        res.sessionInputNames,
        res
      );
    }

    if (res.modelSha256 !== opts.modelSha || res.modelByteLength !== opts.modelBytes.byteLength) {
      return mkFailResult(
        {
          checkId: 'post-model-identity',
          stage: 'post-transfer',
          edgeName: null,
          detail: 'Suffix model bytes/SHA inside worker disagree with host suffix model',
          expected: `${opts.modelSha}(${opts.modelBytes.byteLength}B)`,
          actual: `${res.modelSha256}(${res.modelByteLength}B)`,
        },
        res.sessionInputNames,
        res
      );
    }

    // Check sessionInputNames exact order
    const actualOrder = res.sessionInputNames ?? [];
    if (
      actualOrder.length !== opts.expectedInputOrder.length ||
      actualOrder.some((name, idx) => name !== opts.expectedInputOrder[idx])
    ) {
      return mkFailResult(
        {
          checkId: 'post-session-input-order',
          stage: 'post-transfer',
          edgeName: null,
          detail: 'Worker sessionInputNames order/count disagrees with expected suffix graph input order',
          expected: JSON.stringify(opts.expectedInputOrder),
          actual: JSON.stringify(actualOrder),
        },
        actualOrder,
        res
      );
    }

    // Populate post-transfer edge records for primary
    const primaryEdge = edges[0];
    primaryEdge.orderIndexActual = actualOrder.indexOf(primaryEdge.name);
    primaryEdge.byteOffsetAfter = res.feedByteOffset ?? 0;
    primaryEdge.byteLengthAfter = res.feedByteLength ?? primaryHostF32.byteLength;
    primaryEdge.sha256After = res.feedSha256;

    const primaryPostOk =
      res.feedTensorName === primaryEdge.name &&
      JSON.stringify(res.feedTensorShape) === JSON.stringify(primaryEdge.dims) &&
      res.feedTensorDataType === 'float32' &&
      primaryEdge.byteOffsetAfter === 0 &&
      primaryEdge.byteLengthAfter === primaryEdge.byteLengthBefore &&
      primaryEdge.sha256After === primaryEdge.sha256Before &&
      primaryEdge.orderIndexActual === primaryEdge.orderIndexExpected;
    primaryEdge.matchesAllChecks = primaryPostOk;

    if (!primaryPostOk) {
      return mkFailResult(
        {
          checkId: 'post-primary-edge-identity',
          stage: 'post-transfer',
          edgeName: primaryEdge.name,
          detail: 'Primary #9 input tensor changed shape/dtype/offset/length/SHA/order across worker transfer',
          expected: `name=${primaryEdge.name}, dims=${JSON.stringify(primaryEdge.dims)}, dt=float32, offset=0, bytes=${primaryEdge.byteLengthBefore}, sha=${primaryEdge.sha256Before}`,
          actual: `name=${res.feedTensorName}, dims=${JSON.stringify(res.feedTensorShape)}, dt=${res.feedTensorDataType}, offset=${primaryEdge.byteOffsetAfter}, bytes=${primaryEdge.byteLengthAfter}, sha=${primaryEdge.sha256After}`,
        },
        actualOrder,
        res
      );
    }

    // Check every extraFeedChecks (if present from helper, verify exact match; always check length/order)
    const extraChecks: OrtDiagnosticExtraFeedCheck[] | undefined = res.extraFeedChecks;
    if (extraChecks) {
      if (extraChecks.length !== orderedBypasses.length) {
        return mkFailResult(
          {
            checkId: 'post-bypass-count',
            stage: 'post-transfer',
            edgeName: null,
            detail: 'Post-transfer extraFeedChecks count disagrees with bypass edge count',
            expected: String(orderedBypasses.length),
            actual: String(extraChecks.length),
          },
          actualOrder,
          res
        );
      }
      for (let i = 0; i < orderedBypasses.length; i++) {
        const edge = edges[i + 1];
        const chk = extraChecks[i];
        edge.orderIndexActual = actualOrder.indexOf(chk.name);
        edge.byteOffsetAfter = chk.byteOffset;
        edge.byteLengthAfter = chk.byteLength;
        edge.sha256After = chk.sha256;

        const edgeOk =
          chk.name === edge.name &&
          JSON.stringify(chk.dims) === JSON.stringify(edge.dims) &&
          chk.dataType === edge.dataType &&
          chk.byteOffset === 0 &&
          chk.byteLength === edge.byteLengthBefore &&
          chk.sha256 === edge.sha256Before &&
          edge.orderIndexActual === edge.orderIndexExpected;
        edge.matchesAllChecks = edgeOk;
        if (!edgeOk) {
          return mkFailResult(
            {
              checkId: `post-bypass-edge-${edge.name}`,
              stage: 'post-transfer',
              edgeName: edge.name,
              detail: `Bypass edge "${edge.name}" disagreed across worker transfer/reinsertion`,
              expected: `name=${edge.name}, dims=${JSON.stringify(edge.dims)}, dt=${edge.dataType}, offset=0, bytes=${edge.byteLengthBefore}, sha=${edge.sha256Before}, order=${edge.orderIndexExpected}`,
              actual: `name=${chk.name}, dims=${JSON.stringify(chk.dims)}, dt=${chk.dataType}, offset=${chk.byteOffset}, bytes=${chk.byteLength}, sha=${chk.sha256}, order=${edge.orderIndexActual}`,
            },
            actualOrder,
            res
          );
        }
      }
    } else {
      for (let i = 0; i < orderedBypasses.length; i++) {
        const edge = edges[i + 1];
        edge.orderIndexActual = actualOrder.indexOf(edge.name);
        edge.byteOffsetAfter = edge.byteOffsetBefore;
        edge.byteLengthAfter = edge.byteLengthBefore;
        edge.sha256After = edge.sha256Before;
        edge.matchesAllChecks = edge.orderIndexActual === edge.orderIndexExpected;
      }
    }

    const val = validateHitnetFinalOutputs(res.outputs);
    if (!val.ok) {
      return mkFailResult(
        {
          checkId: 'post-final-output-validation',
          stage: 'output-validation',
          edgeName: null,
          detail: `Suffix final output validation failed: ${val.error}`,
          expected: 'valid finite float32 [1,240,320,1] for primary & secondary',
          actual: String(val.error),
        },
        actualOrder,
        res
      );
    }

    let dispatchObservedSeparatePass: boolean | null = null;
    if (opts.runSeparateProfilePass) {
      const prof = await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: opts.modelBytes.slice().buffer,
        feedBuffer: primaryHostF32.slice().buffer,
        feedName: opts.primaryInputSpec.name,
        feedDims: [...opts.primaryInputSpec.dims],
        extraFeeds: orderedBypasses.map((b) => ({
          name: b.spec.name,
          dims: [...b.spec.dims],
          buffer: b.ownedBuffer.slice(0),
          dataType: b.spec.dataType,
        })),
        allowArbitraryOutputs: false,
        provider: 'webgpu',
        graphOptimizationLevel: 'disabled',
        warmRuns: 0,
        enableWebGpuProfiling: true,
        disableCpuEpFallback,
        captureGpuValidationErrors: true,
      });
      dispatchObservedSeparatePass = hasObservedGpuDispatch(prof);
    }

    const outputsByName: Record<string, OrtVersionDiagnosticOutput> = {};
    for (const o of res.outputs) {
      outputsByName[o.name] = o;
    }

    return {
      ok: true,
      label: opts.label,
      mode: opts.mode,
      suffixSha256: opts.modelSha,
      suffixByteLength: opts.modelBytes.byteLength,
      expectedInputOrder: [...opts.expectedInputOrder],
      actualInputOrder: actualOrder,
      edges,
      firstFailure: null,
      workerSuccess: res,
      dispatchObservedSeparatePass,
      primaryF32: val.byName[pName].data,
      secondaryF32: val.byName[sName].data,
      outputsByName,
    };
  };

  // ---------------- 6. MANDATORY GATE FIRST (Native GPU #9 through shared host transport) ----------------
  const nativeOnCandRun = await executeSuffixViaSharedHostTransport({
    label: 'F0126 GATE: Native GPU #9 via Shared Host Transport',
    mode: 'native-gpu9-copy',
    modelBytes: suffix9.suffixBytes,
    modelSha: suffix9.suffixSha256,
    expectedInputOrder: suffix9.graphInputNames,
    primaryInputSpec: suffix9.primaryCutInput,
    bypassInputList: prefixBypassList,
    runSeparateProfilePass: true,
  });

  const baseVerifiedArtifacts = {
    prefix0to8ByteLength: prefix0to8.prefixByteLength,
    prefix0to8Sha256: prefix0to8.prefixSha256,
    suffix9ByteLength: suffix9.suffixByteLength,
    suffix9Sha256: suffix9.suffixSha256,
    singleOp9ByteLength: singleOp9.bytes.byteLength,
    singleOp9Sha256: singleOp9Sha,
    prefixNode8Sha256: prefixNode8Sha,
    gpuOneOp9RawSha256: gpuOneOp9RawSha,
    bypassEdgeHashes: prefixBypassList.map((b) => ({
      name: b.name,
      dataType: b.dataType,
      dims: [...b.dims],
      byteLength: b.byteLength,
      sha256: b.sha256,
    })),
  };

  const gateExecSummary: Omit<SharedHostTransportExecutionResult, 'primaryF32' | 'secondaryF32' | 'outputsByName'> = {
    ok: nativeOnCandRun.ok,
    label: nativeOnCandRun.label,
    mode: nativeOnCandRun.mode,
    suffixSha256: nativeOnCandRun.suffixSha256,
    suffixByteLength: nativeOnCandRun.suffixByteLength,
    expectedInputOrder: nativeOnCandRun.expectedInputOrder,
    actualInputOrder: nativeOnCandRun.actualInputOrder,
    edges: nativeOnCandRun.edges,
    firstFailure: nativeOnCandRun.firstFailure,
    workerSuccess: nativeOnCandRun.workerSuccess,
    dispatchObservedSeparatePass: nativeOnCandRun.dispatchObservedSeparatePass,
  };

  let gateFailure: TransportFailureDetail | null = nativeOnCandRun.firstFailure;
  let vsNativeSplitP: BitIdentityResult | null = null;
  let vsNativeSplitS: BitIdentityResult | null = null;
  let vsFullGpuP: BitIdentityResult | null = null;
  let vsFullGpuS: BitIdentityResult | null = null;

  if (nativeOnCandRun.ok && nativeOnCandRun.primaryF32 && nativeOnCandRun.secondaryF32) {
    vsNativeSplitP = checkFloat32BitIdentity(
      nativeOnCandRun.primaryF32,
      gpuSuffix9SplitVal.byName[pName].data
    );
    vsNativeSplitS = checkFloat32BitIdentity(
      nativeOnCandRun.secondaryF32,
      gpuSuffix9SplitVal.byName[sName].data
    );
    vsFullGpuP = checkFloat32BitIdentity(nativeOnCandRun.primaryF32, fullGpuVal.byName[pName].data);
    vsFullGpuS = checkFloat32BitIdentity(nativeOnCandRun.secondaryF32, fullGpuVal.byName[sName].data);

    if (!vsNativeSplitP.bitIdentical) {
      gateFailure = {
        checkId: 'gate-bit-exact-vs-native-split-primary',
        stage: 'gate-bit-identity',
        edgeName: pName,
        detail: 'Native-on-candidate primary disparity differed from direct native split',
        expected: '0 ULP bit-identical (mismatchCount=0)',
        actual: `mismatchCount=${vsNativeSplitP.mismatchCount}, firstIdx=${vsNativeSplitP.firstMismatchIndex}, maxAbs=${vsNativeSplitP.maxAbsDiff}`,
      };
    } else if (!vsNativeSplitS.bitIdentical) {
      gateFailure = {
        checkId: 'gate-bit-exact-vs-native-split-secondary',
        stage: 'gate-bit-identity',
        edgeName: sName,
        detail: 'Native-on-candidate secondary disparity differed from direct native split',
        expected: '0 ULP bit-identical (mismatchCount=0)',
        actual: `mismatchCount=${vsNativeSplitS.mismatchCount}, firstIdx=${vsNativeSplitS.firstMismatchIndex}, maxAbs=${vsNativeSplitS.maxAbsDiff}`,
      };
    } else if (!vsFullGpuP.bitIdentical) {
      gateFailure = {
        checkId: 'gate-bit-exact-vs-full-gpu-primary',
        stage: 'gate-bit-identity',
        edgeName: pName,
        detail: 'Native-on-candidate primary disparity differed from untapped full GPU',
        expected: '0 ULP bit-identical (mismatchCount=0)',
        actual: `mismatchCount=${vsFullGpuP.mismatchCount}, firstIdx=${vsFullGpuP.firstMismatchIndex}, maxAbs=${vsFullGpuP.maxAbsDiff}`,
      };
    } else if (!vsFullGpuS.bitIdentical) {
      gateFailure = {
        checkId: 'gate-bit-exact-vs-full-gpu-secondary',
        stage: 'gate-bit-identity',
        edgeName: sName,
        detail: 'Native-on-candidate secondary disparity differed from untapped full GPU',
        expected: '0 ULP bit-identical (mismatchCount=0)',
        actual: `mismatchCount=${vsFullGpuS.mismatchCount}, firstIdx=${vsFullGpuS.firstMismatchIndex}, maxAbs=${vsFullGpuS.maxAbsDiff}`,
      };
    }
  }

  const gatePassed = gateFailure === null && Boolean(nativeOnCandRun.ok);
  log(
    `  [Transport Identity Gate] gatePassed=${gatePassed} | splitP=${vsNativeSplitP?.mismatchCount ?? 'n/a'} splitS=${vsNativeSplitS?.mismatchCount ?? 'n/a'} fullP=${vsFullGpuP?.mismatchCount ?? 'n/a'} fullS=${vsFullGpuS?.mismatchCount ?? 'n/a'}` +
      (gateFailure
        ? `\n  [STOPPED ON GATE FAILURE] ${gateFailure.checkId} (${gateFailure.stage}): ${gateFailure.detail} | expected=${gateFailure.expected} | actual=${gateFailure.actual}`
        : '')
  );

  // STOP IMMEDIATELY ON GATE FAILURE — ZERO ATTRIBUTION!
  if (!gatePassed || !nativeOnCandRun.primaryF32 || !nativeOnCandRun.secondaryF32) {
    const abortMsg = gateFailure
      ? `Transport identity gate failed at ${gateFailure.checkId} (${gateFailure.stage}): ${gateFailure.detail}`
      : 'Transport identity gate failed';
    return {
      ok: false,
      aborted: true,
      abortReason: abortMsg,
      gatePassed: false,
      gateFirstFailure: gateFailure,
      pinnedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      graphOptimizationLevel: 'disabled',
      numThreads: 1,
      proxy: false,
      m1ByteLength: params.m1Bytes.byteLength,
      m1Sha256: m1Sha,
      f0126FeedSha256: f0126Sha,
      n1OffProof,
      chainSpec,
      verifiedArtifacts: baseVerifiedArtifacts,
      nativeSplitControlCheck: {
        gpuSplitVsFullGpuPrimary: gpuSplitVsFullP,
        gpuSplitVsFullGpuSecondary: gpuSplitVsFullS,
        passed: nativeSplitPassed,
      },
      gateTransportRecord: {
        execution: gateExecSummary,
        vsNativeSplitPrimary: vsNativeSplitP,
        vsNativeSplitSecondary: vsNativeSplitS,
        vsFullGpuDisabledPrimary: vsFullGpuP,
        vsFullGpuDisabledSecondary: vsFullGpuS,
      },
      outlierComparison: null,
      ancestryAndTaps: null,
      causalAttribution: null,
      commonWasm7Secondary: {
        source: 'skipped-on-gate-failure',
        frameLabel: 'F0126',
        feedSha256: f0126Sha,
        controlScopeNote: 'Stopped on mandatory native-on-candidate transport gate failure; no attribution executed.',
        dispatchConfirmedSeparatePass: null,
        baselineGpuSuffixWasm7VsWasmFull: null,
        intervenedGpuSuffixWasm7WithCand9VsWasmFull: null,
        intervenedVsBaselineGpuSuffix: null,
      },
      scientificScopeNote:
        'Stopped on mandatory F0126 native-on-candidate transport gate failure; emitted first transport/layout disagreement only with zero outlier attribution.',
    };
  }

  // ---------------- 7. GATE PASSED: RUN BLOCKED-128 ON EXACT SAME PREFIX/SUFFIX/BYPASSES/TRANSPORT ----------------
  const nativePrimaryF32 = nativeOnCandRun.primaryF32;
  const nativeSecondaryF32 = nativeOnCandRun.secondaryF32;
  const wasmPrimaryF32 = fullWasmVal.byName[pName].data;
  const wasmSecondaryF32 = fullWasmVal.byName[sName].data;
  const fullGpuPrimaryF32 = fullGpuVal.byName[pName].data;
  const fullGpuSecondaryF32 = fullGpuVal.byName[sName].data;

  const cand128Run = await executeSuffixViaSharedHostTransport({
    label: 'F0126 INTERVENTION: Blocked-K128 #9 via Shared Host Transport',
    mode: 'blocked-k128',
    modelBytes: suffix9.suffixBytes,
    modelSha: suffix9.suffixSha256,
    expectedInputOrder: suffix9.graphInputNames,
    primaryInputSpec: suffix9.primaryCutInput,
    bypassInputList: prefixBypassList,
    runSeparateProfilePass: true,
  });

  if (!cand128Run.ok || !cand128Run.primaryF32 || !cand128Run.secondaryF32) {
    throw new Error(
      `Cand128 shared host transport failed (${cand128Run.firstFailure?.checkId ?? 'unknown'}): ${cand128Run.firstFailure?.detail ?? 'unknown error'}`
    );
  }

  const candPrimaryF32 = cand128Run.primaryF32;
  const candSecondaryF32 = cand128Run.secondaryF32;

  const cand9EdgeBufferF32 = populateFreshOwnedHostConv9Buffer({
    mode: 'blocked-k128',
    parsed: parsedModel,
    convNodeIdx: chainSpec.node9ConvDown0.nodeIdx,
    xData: prefixNode8F32,
    inputDims: singleOp9.dynamicInputSpecs[0].dims as [number, number, number, number],
    outDims: suffix9.primaryCutInput.dims as [number, number, number, number],
  });
  const edge9Cand128VsNativeGpu9 = compareFloat32TensorsDetailed(
    cand9EdgeBufferF32,
    gpuOneOp9RawF32,
    suffix9.primaryCutInput.dims
  );

  const d1 = [1, 240, 320, 1] as const;
  const cand128VsNativeOnCand = {
    primary: compareFloat32TensorsDetailed(candPrimaryF32, nativePrimaryF32, [...d1]),
    secondary: compareFloat32TensorsDetailed(candSecondaryF32, nativeSecondaryF32, [...d1]),
  };
  const cand128VsFullWasmDisabled = {
    primary: compareFloat32TensorsDetailed(candPrimaryF32, wasmPrimaryF32, [...d1]),
    secondary: compareFloat32TensorsDetailed(candSecondaryF32, wasmSecondaryF32, [...d1]),
  };
  const nativeOnCandVsFullWasmDisabled = {
    primary: compareFloat32TensorsDetailed(nativePrimaryF32, wasmPrimaryF32, [...d1]),
    secondary: compareFloat32TensorsDetailed(nativeSecondaryF32, wasmSecondaryF32, [...d1]),
  };
  const cand128VsFullGpuDisabled = {
    primary: compareFloat32TensorsDetailed(candPrimaryF32, fullGpuPrimaryF32, [...d1]),
    secondary: compareFloat32TensorsDetailed(candSecondaryF32, fullGpuSecondaryF32, [...d1]),
  };

  const leftPrimaryValidity = computeEyeValidityAndDepthFlips({
    eye: 'left-primary',
    tensorName: pName,
    nativeData: nativePrimaryF32,
    candData: candPrimaryF32,
    wasmData: wasmPrimaryF32,
  });
  const rightSecondaryValidity = computeEyeValidityAndDepthFlips({
    eye: 'right-secondary',
    tensorName: sName,
    nativeData: nativeSecondaryF32,
    candData: candSecondaryF32,
    wasmData: wasmSecondaryF32,
  });

  const secondaryClusterInfo = clusterSecondaryOutliers({
    nativeSecondary: nativeSecondaryF32,
    candSecondary: candSecondaryF32,
    wasmSecondary: wasmSecondaryF32,
  });

  log(
    `  [Cand128 vs Native-on-Candidate]\n` +
      `    Edge#9 diff (cand128 vs nativeGPU9): max=${edge9Cand128VsNativeGpu9.stats.maxAbsDiff.toExponential(3)} p99=${edge9Cand128VsNativeGpu9.stats.p99AbsDiff.toExponential(3)} mean=${edge9Cand128VsNativeGpu9.stats.meanAbsDiff.toExponential(3)}\n` +
      `    Cand128 vs NativeOnCand: P(max=${cand128VsNativeOnCand.primary.stats.maxAbsDiff.toExponential(3)} p99=${cand128VsNativeOnCand.primary.stats.p99AbsDiff.toExponential(3)} mean=${cand128VsNativeOnCand.primary.stats.meanAbsDiff.toExponential(3)}) | S(max=${cand128VsNativeOnCand.secondary.stats.maxAbsDiff.toExponential(3)} p99=${cand128VsNativeOnCand.secondary.stats.p99AbsDiff.toExponential(3)} mean=${cand128VsNativeOnCand.secondary.stats.meanAbsDiff.toExponential(3)})\n` +
      `    Cand128 vs FullWASM:     P(max=${cand128VsFullWasmDisabled.primary.stats.maxAbsDiff.toExponential(3)} p99=${cand128VsFullWasmDisabled.primary.stats.p99AbsDiff.toExponential(3)} mean=${cand128VsFullWasmDisabled.primary.stats.meanAbsDiff.toExponential(3)}) | S(max=${cand128VsFullWasmDisabled.secondary.stats.maxAbsDiff.toExponential(3)} p99=${cand128VsFullWasmDisabled.secondary.stats.p99AbsDiff.toExponential(3)} mean=${cand128VsFullWasmDisabled.secondary.stats.meanAbsDiff.toExponential(3)})\n` +
      `    Eye Validity: Left(nat=${leftPrimaryValidity.nativeValidCount}, cand=${leftPrimaryValidity.candValidCount}, wasm=${leftPrimaryValidity.wasmValidCount}, flipsVsNat=${leftPrimaryValidity.totalValidityFlipsVsNative}) | Right(nat=${rightSecondaryValidity.nativeValidCount}, cand=${rightSecondaryValidity.candValidCount}, wasm=${rightSecondaryValidity.wasmValidCount}, flipsVsNat=${rightSecondaryValidity.totalValidityFlipsVsNative})\n` +
      `    Secondary Outliers (> ${secondaryClusterInfo.thresholdUsed}): pixels=${secondaryClusterInfo.outlierPixelCount}, clusters=${secondaryClusterInfo.clusters.length}` +
      (secondaryClusterInfo.clusters[0]
        ? `, topClusterPeak=(y=${secondaryClusterInfo.clusters[0].peak.y},x=${secondaryClusterInfo.clusters[0].peak.x},tile=${secondaryClusterInfo.clusters[0].peak.ty},${secondaryClusterInfo.clusters[0].peak.tx}, absVsNat=${secondaryClusterInfo.clusters[0].peak.absDiffVsNative.toExponential(3)})`
        : '')
  );

  // ---------------- 8. BOUNDED ANCESTRY TAPS (<= 2 TAPS/PASS, BIT-INVARIANT FINALS) ----------------
  const ancestrySpec = resolveSecondaryOutlierAncestrySpec(parsedModel, chainSpec, suffix9);

  const runBoundedTappedPair = async (
    passLabel: string,
    tapNames: string[]
  ): Promise<{
    invariance: TapPassInvarianceRecord;
    nativeRes: SharedHostTransportExecutionResult | null;
    candRes: SharedHostTransportExecutionResult | null;
  }> => {
    const cleanTaps = Array.from(new Set(tapNames.filter(Boolean))).slice(0, 2);
    const tapped = appendOnnxGraphOutputs(suffix9.suffixBytes, cleanTaps, { maxExtraOutputs: 2 });
    const tappedSha = await sha256HexBytes(tapped.bytes);

    const nativeTapped = await executeSuffixViaSharedHostTransport({
      label: `${passLabel} [Native-on-Candidate]`,
      mode: 'native-gpu9-copy',
      modelBytes: tapped.bytes,
      modelSha: tappedSha,
      expectedInputOrder: suffix9.graphInputNames,
      primaryInputSpec: suffix9.primaryCutInput,
      bypassInputList: prefixBypassList,
    });
    const candTapped = await executeSuffixViaSharedHostTransport({
      label: `${passLabel} [Blocked-K128]`,
      mode: 'blocked-k128',
      modelBytes: tapped.bytes,
      modelSha: tappedSha,
      expectedInputOrder: suffix9.graphInputNames,
      primaryInputSpec: suffix9.primaryCutInput,
      bypassInputList: prefixBypassList,
    });

    const nP = nativeTapped.primaryF32
      ? checkFloat32BitIdentity(nativeTapped.primaryF32, nativePrimaryF32)
      : null;
    const nS = nativeTapped.secondaryF32
      ? checkFloat32BitIdentity(nativeTapped.secondaryF32, nativeSecondaryF32)
      : null;
    const cP = candTapped.primaryF32
      ? checkFloat32BitIdentity(candTapped.primaryF32, candPrimaryF32)
      : null;
    const cS = candTapped.secondaryF32
      ? checkFloat32BitIdentity(candTapped.secondaryF32, candSecondaryF32)
      : null;

    const bothInvariant = Boolean(
      nativeTapped.ok &&
        candTapped.ok &&
        nP?.bitIdentical &&
        nS?.bitIdentical &&
        cP?.bitIdentical &&
        cS?.bitIdentical
    );

    let failureReason: string | null = null;
    if (!nativeTapped.ok) failureReason = `Native tapped failed: ${nativeTapped.firstFailure?.detail ?? 'error'}`;
    else if (!candTapped.ok) failureReason = `Cand tapped failed: ${candTapped.firstFailure?.detail ?? 'error'}`;
    else if (!bothInvariant) {
      failureReason = `Tapped outputs perturbed finals (nP=${nP?.mismatchCount}, nS=${nS?.mismatchCount}, cP=${cP?.mismatchCount}, cS=${cS?.mismatchCount})`;
    }

    return {
      invariance: {
        passLabel,
        tapOutputs: cleanTaps,
        tappedModelByteLength: tapped.bytes.byteLength,
        tappedModelSha256: tappedSha,
        nativeTransportOk: nativeTapped.ok,
        candTransportOk: candTapped.ok,
        nativePrimaryBitIdentical: nP,
        nativeSecondaryBitIdentical: nS,
        candPrimaryBitIdentical: cP,
        candSecondaryBitIdentical: cS,
        bothArmsBitInvariant: bothInvariant,
        failureReason,
      },
      nativeRes: nativeTapped,
      candRes: candTapped,
    };
  };

  // Unique outlier tiles sorted by peak abs diff descending
  const outlierTileSet = new Map<number, { ty: number; tx: number; tileFlatIdx: number }>();
  for (const p of secondaryClusterInfo.topPixels) {
    const tIdx = p.ty * 80 + p.tx;
    if (!outlierTileSet.has(tIdx)) {
      outlierTileSet.set(tIdx, { ty: p.ty, tx: p.tx, tileFlatIdx: tIdx });
    }
  }
  for (const c of secondaryClusterInfo.clusters) {
    const pt = c.peak;
    const ptIdx = pt.ty * 80 + pt.tx;
    if (!outlierTileSet.has(ptIdx)) {
      outlierTileSet.set(ptIdx, { ty: pt.ty, tx: pt.tx, tileFlatIdx: ptIdx });
    }
    for (const st of c.sampleTiles) {
      const sIdx = st.ty * 80 + st.tx;
      if (!outlierTileSet.has(sIdx)) {
        outlierTileSet.set(sIdx, { ty: st.ty, tx: st.tx, tileFlatIdx: sIdx });
      }
    }
  }
  const outlierTilesList = Array.from(outlierTileSet.values());
  const topClusterTileBbox = secondaryClusterInfo.clusters[0]?.tileBbox ?? null;

  // Tap Pass 1: [coarseArgMin (int64), postProp0Out (float32)]
  const pass1Taps = [ancestrySpec.coarseArgMin.outputTensor];
  if (ancestrySpec.postProp0Out) pass1Taps.push(ancestrySpec.postProp0Out.outputTensor);

  const pass1 = await runBoundedTappedPair('TapPass1: coarseArgMin + postProp0', pass1Taps);
  let coarseAnalysis: CoarseArgMinAndCostAnalysis | null = null;
  const lateStageTaps: LateStageTapComparison[] = [];

  let verdict: NonNullable<HitnetF0126SecondaryOutlierReport['causalAttribution']>['verdict'];
  let summary: string;
  const evidencePoints: string[] = [];

  if (pass1.invariance.bothArmsBitInvariant && pass1.nativeRes && pass1.candRes) {
    const natArgMinOut = pass1.nativeRes.outputsByName[ancestrySpec.coarseArgMin.outputTensor];
    const candArgMinOut = pass1.candRes.outputsByName[ancestrySpec.coarseArgMin.outputTensor];
    if (!natArgMinOut || !candArgMinOut) {
      throw new Error(`Missing coarse ArgMin tap "${ancestrySpec.coarseArgMin.outputTensor}"`);
    }
    const natArgMinI64 = readInt64OrNumberArray(natArgMinOut);
    const candArgMinI64 = readInt64OrNumberArray(candArgMinOut);
    const int64Comp = summarizeExactInt64IndexComparison(candArgMinI64, natArgMinI64, {
      minInclusive: 0,
      maxInclusive: ancestrySpec.coarseArgMin.maxValidIndex,
    });

    const natPostProp0 = ancestrySpec.postProp0Out
      ? pass1.nativeRes.outputsByName[ancestrySpec.postProp0Out.outputTensor]
      : undefined;
    const candPostProp0 = ancestrySpec.postProp0Out
      ? pass1.candRes.outputsByName[ancestrySpec.postProp0Out.outputTensor]
      : undefined;
    const natPostF32 = natPostProp0 ? readFloat32ArrayCopy(natPostProp0) : null;
    const candPostF32 = candPostProp0 ? readFloat32ArrayCopy(candPostProp0) : null;
    const postProp0FullDiff =
      natPostF32 && candPostF32 && ancestrySpec.postProp0Out
        ? compareFloat32TensorsDetailed(candPostF32, natPostF32, ancestrySpec.postProp0Out.dims)
        : null;
    const postProp0Roi =
      natPostF32 && candPostF32 && ancestrySpec.postProp0Out
        ? computeRoiTensorDiff(candPostF32, natPostF32, ancestrySpec.postProp0Out.dims, topClusterTileBbox)
        : { roiMaxAbsDiff: null, roiMeanAbsDiff: null };

    let outlierFlipCount = 0;
    let nonOutlierFlipCount = 0;
    const differingTileIndices: number[] = [];
    for (let t = 0; t < Math.min(natArgMinI64.length, candArgMinI64.length); t++) {
      if (natArgMinI64[t] !== candArgMinI64[t]) {
        differingTileIndices.push(t);
        if (outlierTileSet.has(t)) outlierFlipCount++;
        else nonOutlierFlipCount++;
      }
    }

    let tapPass2Invariance: TapPassInvarianceRecord | null = null;
    let coarseCostFullDiff: DetailedFloat32Comparison | null = null;
    let coarseCostRoiMax: number | null = null;
    const inspectedTiles: CoarseArgMinTileCostInspection[] = [];
    let exactOrderCrossings = 0;
    let exactTieFlips = 0;
    let kernelMismatches = 0;

    if (int64Comp.mismatchCount > 0) {
      // Tap Pass 2: [coarseCostTensor, initToPropOut]
      const pass2Taps = [ancestrySpec.coarseArgMin.inputCostTensor];
      if (ancestrySpec.initToPropOut) pass2Taps.push(ancestrySpec.initToPropOut.outputTensor);
      const pass2 = await runBoundedTappedPair('TapPass2: coarseCost + initToPropOut', pass2Taps);
      tapPass2Invariance = pass2.invariance;

      if (pass2.invariance.bothArmsBitInvariant && pass2.nativeRes && pass2.candRes) {
        const natCostOut = pass2.nativeRes.outputsByName[ancestrySpec.coarseArgMin.inputCostTensor];
        const candCostOut = pass2.candRes.outputsByName[ancestrySpec.coarseArgMin.inputCostTensor];
        const natInitOut = ancestrySpec.initToPropOut
          ? pass2.nativeRes.outputsByName[ancestrySpec.initToPropOut.outputTensor]
          : undefined;
        const candInitOut = ancestrySpec.initToPropOut
          ? pass2.candRes.outputsByName[ancestrySpec.initToPropOut.outputTensor]
          : undefined;

        if (natCostOut && candCostOut) {
          const natCostF32 = readFloat32ArrayCopy(natCostOut);
          const candCostF32 = readFloat32ArrayCopy(candCostOut);
          coarseCostFullDiff = compareFloat32TensorsDetailed(
            candCostF32,
            natCostF32,
            ancestrySpec.coarseArgMin.inputCostDims
          );
          const roiCost = computeRoiTensorDiff(
            candCostF32,
            natCostF32,
            ancestrySpec.coarseArgMin.inputCostDims,
            topClusterTileBbox
          );
          coarseCostRoiMax = roiCost.roiMaxAbsDiff;

          const natInitF32 = natInitOut ? readFloat32ArrayCopy(natInitOut) : null;
          const candInitF32 = candInitOut ? readFloat32ArrayCopy(candInitOut) : null;

          const { depthCount, getFlatIdx } = computeTileCostOffsetHelper(
            ancestrySpec.coarseArgMin.inputCostDims,
            ancestrySpec.coarseArgMin.axis
          );
          const selectLast = ancestrySpec.coarseArgMin.selectLastIndex;

          // Pick tiles: prioritize outlier tiles that flipped, then remaining flipped tiles, then top outlier tiles (up to 16 total)
          const chosenTileIds: number[] = [];
          const pushUnique = (t: number) => {
            if (!chosenTileIds.includes(t) && chosenTileIds.length < 16) chosenTileIds.push(t);
          };
          for (const ot of outlierTilesList) {
            if (natArgMinI64[ot.tileFlatIdx] !== candArgMinI64[ot.tileFlatIdx]) pushUnique(ot.tileFlatIdx);
          }
          for (const dt of differingTileIndices) pushUnique(dt);
          for (const ot of outlierTilesList) pushUnique(ot.tileFlatIdx);

          const vecNat = new Float32Array(depthCount);
          const vecCand = new Float32Array(depthCount);
          for (const t of chosenTileIds) {
            const ty = Math.floor(t / 80);
            const tx = t % 80;
            for (let d = 0; d < depthCount; d++) {
              const fIdx = getFlatIdx(ty, tx, d);
              vecNat[d] = natCostF32[fIdx];
              vecCand[d] = candCostF32[fIdx];
            }
            const exactNat = computeExactOnnxArgMinForVector(vecNat, selectLast);
            const exactCand = computeExactOnnxArgMinForVector(vecCand, selectLast);
            const dNative = Number(natArgMinI64[t]);
            const dCand = Number(candArgMinI64[t]);
            const matchesNat = exactNat.winner === dNative;
            const matchesCand = exactCand.winner === dCand;

            const cNatAtDNat = dNative >= 0 && dNative < depthCount ? vecNat[dNative] : NaN;
            const cNatAtDCand = dCand >= 0 && dCand < depthCount ? vecNat[dCand] : NaN;
            const cCandAtDNat = dNative >= 0 && dNative < depthCount ? vecCand[dNative] : NaN;
            const cCandAtDCand = dCand >= 0 && dCand < depthCount ? vecCand[dCand] : NaN;

            let classification: CoarseArgMinTileCostInspection['classification'] = 'NO_BIN_FLIP';
            if (dNative !== dCand) {
              if (!matchesNat || !matchesCand) {
                classification = 'ARGMIN_KERNEL_MISMATCH';
                kernelMismatches++;
              } else if (
                exactNat.tieIndices.includes(dCand) ||
                exactCand.tieIndices.includes(dNative)
              ) {
                classification = 'EXACT_TIE_TIEBREAKER';
                exactTieFlips++;
              } else {
                classification = 'EXACT_COST_ORDER_CROSSING';
                exactOrderCrossings++;
              }
            }

            let postTileMax: number | null = null;
            if (natPostF32 && candPostF32) {
              let mx = 0;
              const base = t * 16;
              for (let c = 0; c < 16; c++) {
                const df = Math.abs(candPostF32[base + c] - natPostF32[base + c]);
                if (df > mx) mx = df;
              }
              postTileMax = mx;
            }

            inspectedTiles.push({
              ty,
              tx,
              tileFlatIdx: t,
              isSecondaryOutlierTile: outlierTileSet.has(t),
              dNative,
              dCand,
              deltaD: dCand - dNative,
              expectedOnnxArgMinNative: exactNat.winner,
              expectedOnnxArgMinCand: exactCand.winner,
              observedMatchesOnnxExactNative: matchesNat,
              observedMatchesOnnxExactCand: matchesCand,
              finiteCostCountNative: exactNat.finiteCount,
              finiteCostCountCand: exactCand.finiteCount,
              minCostNative: exactNat.minCost,
              minCostCand: exactCand.minCost,
              tieIndicesNative: exactNat.tieIndices.slice(0, 8),
              tieIndicesCand: exactCand.tieIndices.slice(0, 8),
              costNativeAtDNative: cNatAtDNat,
              costNativeAtDCand: cNatAtDCand,
              costCandAtDNative: cCandAtDNat,
              costCandAtDCand: cCandAtDCand,
              marginNativeCandMinusNatWinner: cNatAtDCand - cNatAtDNat,
              marginCandNatWinnerMinusCandWinner: cCandAtDNat - cCandAtDCand,
              classification,
              initToPropDispCh0Native: natInitF32 ? natInitF32[t * 16] : null,
              initToPropDispCh0Cand: candInitF32 ? candInitF32[t * 16] : null,
              postProp0MaxAbsDiffAtTile: postTileMax,
            });
          }
        }
      }
    } else {
      // Coarse INT64 ArgMin matched everywhere! Inspect later secondary stages via bounded <=2-tap passes
      const candidateLatePairs: Array<{ label: string; taps: string[] }> = [];
      if (ancestrySpec.propWarpOut || ancestrySpec.prop0AddOut) {
        candidateLatePairs.push({
          label: 'TapPass2B: propWarp + prop0Add',
          taps: [ancestrySpec.propWarpOut?.outputTensor ?? '', ancestrySpec.prop0AddOut?.outputTensor ?? ''].filter(Boolean),
        });
      }
      if (ancestrySpec.refinementL0Out || ancestrySpec.refinementL1Out) {
        candidateLatePairs.push({
          label: 'TapPass3B: refinement_l0 + refinement_l1',
          taps: [ancestrySpec.refinementL0Out?.outputTensor ?? '', ancestrySpec.refinementL1Out?.outputTensor ?? ''].filter(Boolean),
        });
      }
      if (ancestrySpec.refinementL2Add || ancestrySpec.refinementL2SlantAdd) {
        candidateLatePairs.push({
          label: 'TapPass4B: refinement_l2 add + slantAdd',
          taps: [ancestrySpec.refinementL2Add?.outputTensor ?? '', ancestrySpec.refinementL2SlantAdd?.outputTensor ?? ''].filter(Boolean),
        });
      }

      for (const pair of candidateLatePairs) {
        const resPair = await runBoundedTappedPair(pair.label, pair.taps);
        const tapSummaries: LateStageTapComparison['taps'] = [];
        if (resPair.invariance.bothArmsBitInvariant && resPair.nativeRes && resPair.candRes) {
          for (const tName of pair.taps) {
            const nO = resPair.nativeRes.outputsByName[tName];
            const cO = resPair.candRes.outputsByName[tName];
            if (!nO || !cO) continue;
            const nF = readFloat32ArrayCopy(nO);
            const cF = readFloat32ArrayCopy(cO);
            const fullComp = compareFloat32TensorsDetailed(cF, nF, nO.dims);
            const roi = computeRoiTensorDiff(cF, nF, nO.dims, topClusterTileBbox);
            tapSummaries.push({
              name: tName,
              dims: [...nO.dims],
              dataType: nO.dataType,
              fullComparison: fullComp,
              roiMaxAbsDiff: roi.roiMaxAbsDiff,
              roiMeanAbsDiff: roi.roiMeanAbsDiff,
            });
          }
        }
        lateStageTaps.push({
          passLabel: pair.label,
          invariance: resPair.invariance,
          taps: tapSummaries,
        });
        if (!resPair.invariance.bothArmsBitInvariant) break;
      }
    }

    coarseAnalysis = {
      tapPass1Invariance: pass1.invariance,
      tapPass2Invariance,
      coarseArgMinSelectLastIndex: ancestrySpec.coarseArgMin.selectLastIndex,
      coarseArgMinDataTypeNative: natArgMinOut.dataType,
      coarseArgMinDataTypeCand: candArgMinOut.dataType,
      int64Comparison: int64Comp,
      outlierTileIndices: outlierTilesList,
      outlierTilesWithArgMinFlipCount: outlierFlipCount,
      nonOutlierTilesWithArgMinFlipCount: nonOutlierFlipCount,
      postProp0FullDiff,
      postProp0OutlierTileMaxAbsDiff: postProp0Roi.roiMaxAbsDiff,
      coarseCostFullDiff,
      coarseCostOutlierTileMaxAbsDiff: coarseCostRoiMax,
      inspectedTiles,
      exactCostOrderCrossingsCount: exactOrderCrossings,
      exactTieTiebreakersCount: exactTieFlips,
      argminKernelMismatchCount: kernelMismatches,
    };

    // Determine Verdict empirically strictly from verified tap results
    const secMax = cand128VsNativeOnCand.secondary.stats.maxAbsDiff;
    if (secMax < 1e-2 && secondaryClusterInfo.outlierPixelCount === 0) {
      verdict = 'NO_SIGNIFICANT_SECONDARY_OUTLIER';
      summary = `No large secondary_output_disparity outlier reproduced under shared host transport (maxAbsDiff=${secMax.toExponential(3)}).`;
    } else if (int64Comp.mismatchCount > 0) {
      if (tapPass2Invariance && !tapPass2Invariance.bothArmsBitInvariant) {
        verdict = 'UNRESOLVED';
        summary = `Coarse INT64 ArgMin had ${int64Comp.mismatchCount} bin flips (outlierTilesFlipped=${outlierFlipCount}), but TapPass2 perturbed final outputs (${tapPass2Invariance.failureReason}).`;
      } else if (kernelMismatches > 0 && exactOrderCrossings === 0 && exactTieFlips === 0) {
        verdict = 'COARSE_ARGMIN_KERNEL_MISMATCH';
        summary = `Coarse INT64 ArgMin disagreed with exact min(coarseCost) on ${kernelMismatches} inspected tile(s) despite invariant finals.`;
      } else if (exactTieFlips > 0 && exactOrderCrossings === 0) {
        verdict = 'COARSE_ARGMIN_EXACT_TIE_FLIP';
        summary = `Secondary outlier traces to ${exactTieFlips} exact-cost tie flip(s) at coarse ArgMin (${ancestrySpec.coarseArgMin.nodeName}, select_last_index=${ancestrySpec.coarseArgMin.selectLastIndex}).`;
      } else if (exactOrderCrossings > 0) {
        verdict = 'COARSE_ARGMIN_COST_ORDER_CROSSING';
        const firstCross = inspectedTiles.find((x) => x.classification === 'EXACT_COST_ORDER_CROSSING');
        summary = `Secondary outlier (${secMax.toFixed(4)} px raw diff) traces to discrete integer ArgMin winner flips (${int64Comp.mismatchCount} total tiles, ${outlierFlipCount} in outlier ROI) where small FP32 cost perturbations reversed finite cost ordering` +
          (firstCross
            ? ` (e.g. tile (${firstCross.ty},${firstCross.tx}): d_native=${firstCross.dNative} -> d_cand=${firstCross.dCand}, marginNative=${firstCross.marginNativeCandMinusNatWinner.toExponential(3)}, marginCand=${firstCross.marginCandNatWinnerMinusCandWinner.toExponential(3)}).`
            : '.');
      } else {
        verdict = 'UNRESOLVED';
        summary = `Coarse INT64 ArgMin had ${int64Comp.mismatchCount} mismatch(es) without cost-order/tie confirmation.`;
      }
      evidencePoints.push(
        `Coarse INT64 ArgMin mismatches: ${int64Comp.mismatchCount}/${int64Comp.length} (maxAbsDelta=${int64Comp.maxAbsDiff} bins, outlierTilesFlipped=${outlierFlipCount}).`,
        `CoarseCost maxAbsDiff overall=${coarseCostFullDiff?.stats.maxAbsDiff.toExponential(3) ?? 'n/a'}, outlierROI=${coarseCostRoiMax?.toExponential(3) ?? 'n/a'}.`,
        `ONNX ArgMin select_last_index=${ancestrySpec.coarseArgMin.selectLastIndex}; inspectedTiles=${inspectedTiles.length} (orderCrossings=${exactOrderCrossings}, exactTies=${exactTieFlips}, kernelMismatches=${kernelMismatches}).`
      );
    } else {
      // Zero coarse ArgMin mismatches — check postProp0 vs refinement taps
      const p0RoiMax = postProp0Roi.roiMaxAbsDiff ?? 0;
      const anyRefJump = lateStageTaps.some((lt) =>
        lt.taps.some((t) => (t.roiMaxAbsDiff ?? t.fullComparison?.stats.maxAbsDiff ?? 0) > 0.1)
      );
      if (p0RoiMax > 0.1) {
        verdict = 'POST_COARSE_PROP0_AMPLIFICATION';
        summary = `Coarse INT64 ArgMin was 100% identical (0/${int64Comp.length} flips); the secondary outlier emerged after coarse ArgMin by post-prop0 (roiMaxAbsDiff=${p0RoiMax.toFixed(4)}).`;
      } else if (anyRefJump) {
        verdict = 'POST_PROP0_REFINEMENT_AMPLIFICATION';
        const firstJump = lateStageTaps.find((lt) =>
          lt.taps.some((t) => (t.roiMaxAbsDiff ?? t.fullComparison?.stats.maxAbsDiff ?? 0) > 0.1)
        );
        summary = `Coarse INT64 ArgMin had 0 flips and post-prop0 stayed small (roiMax=${p0RoiMax.toExponential(3)}); secondary outlier amplified during refinement (${firstJump?.passLabel ?? 'refinement'}).`;
      } else {
        verdict = 'UNRESOLVED';
        summary = `Coarse INT64 ArgMin had 0 flips and tapped intermediates stayed below outlier threshold or invariance failed.`;
      }
      evidencePoints.push(
        `Coarse INT64 ArgMin exact matches: ${int64Comp.exactMatchCount}/${int64Comp.length} (0 flips).`,
        `postProp0 ROI maxAbsDiff=${p0RoiMax.toExponential(3)}; lateStageTaps=${lateStageTaps.map((x) => `${x.passLabel}(invariant=${x.invariance.bothArmsBitInvariant})`).join(', ')}.`
      );
    }
  } else {
    verdict = 'UNRESOLVED';
    summary = `TapPass1 failed final-output invariance (${pass1.invariance.failureReason ?? 'unknown'}); secondary outlier mechanism remains UNRESOLVED.`;
    evidencePoints.push(pass1.invariance.failureReason ?? 'TapPass1 failed');
  }

  log(`  [Causal Verdict] ${verdict}: ${summary}`);

  // ---------------- 9. COMMON-WASM-#7 SECONDARY EXPOSURE ----------------
  let commonWasm7Secondary: CommonWasm7SecondaryExposureReport;
  const supplied = params.existingCommonWasm7Summary;
  if (
    supplied &&
    supplied.causalBaselineGpuSuffixWasm7VsWasmFull?.secondary?.stats &&
    supplied.causalIntervenedGpuSuffixWasm7WithCand9VsWasmFull?.secondary?.stats
  ) {
    commonWasm7Secondary = {
      source: 'reused-existing-summary',
      frameLabel: 'F0126',
      feedSha256: f0126Sha,
      controlScopeNote: supplied.controlScopeNote,
      dispatchConfirmedSeparatePass: supplied.dispatchConfirmedSeparatePass,
      baselineGpuSuffixWasm7VsWasmFull: {
        primary: supplied.causalBaselineGpuSuffixWasm7VsWasmFull.primary.stats,
        secondary: supplied.causalBaselineGpuSuffixWasm7VsWasmFull.secondary.stats,
      },
      intervenedGpuSuffixWasm7WithCand9VsWasmFull: {
        primary: supplied.causalIntervenedGpuSuffixWasm7WithCand9VsWasmFull.primary.stats,
        secondary: supplied.causalIntervenedGpuSuffixWasm7WithCand9VsWasmFull.secondary.stats,
        primaryFlipsVsWasm:
          supplied.causalIntervenedGpuSuffixWasm7WithCand9VsWasmFull.depthAndValidityImpact
            ?.primarySourceMaskChanges?.totalMaskFlips,
      },
      intervenedVsBaselineGpuSuffix: supplied.causalIntervenedGpuSuffixWasm7WithCand9VsCausalBaselineGpuSuffix
        ? {
            primary: supplied.causalIntervenedGpuSuffixWasm7WithCand9VsCausalBaselineGpuSuffix.primary.stats,
            secondary: supplied.causalIntervenedGpuSuffixWasm7WithCand9VsCausalBaselineGpuSuffix.secondary.stats,
          }
        : null,
    };
  } else {
    // Run one straightforward controlled separate F0126 Common-WASM-#7 arm
    log('  [Common-WASM-#7 Controlled Separate Arm] Running F0126 WASM-#7 baseline vs Cand9(WASM-#8)...');
    const suffix7 = await buildVerifiedSuffixAfterConv7(parsedModel, { cutNodeIdx: chainSpec.conv7Idx });
    const tapWasm78 = appendOnnxGraphOutputs(
      params.m1Bytes,
      [chainSpec.conv7OutputTensor, node8OutName],
      { maxExtraOutputs: 2 }
    );
    const tapWasm78Sha = await sha256HexBytes(tapWasm78.bytes);
    const w78Run = await runSimpleOrt130({
      label: 'F0126 Common-WASM Tap(#7,#8)',
      modelBytes: tapWasm78.bytes,
      modelSha: tapWasm78Sha,
      provider: 'wasm',
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      feedF32: params.f0126FeedFloat32,
      feedSha: f0126Sha,
    });
    const w78Val = validateHitnetFinalOutputs(w78Run.res.outputs);
    if (
      !w78Val.ok ||
      !checkFloat32BitIdentity(w78Val.byName[pName].data, wasmPrimaryF32).bitIdentical ||
      !checkFloat32BitIdentity(w78Val.byName[sName].data, wasmSecondaryF32).bitIdentical
    ) {
      throw new Error('F0126 Common-WASM Tap(#7,#8) perturbed full WASM final outputs');
    }

    const w7Out = w78Run.res.outputs.find((o) => o.name === chainSpec.conv7OutputTensor);
    const w8Out = w78Run.res.outputs.find((o) => o.name === node8OutName);
    if (!w7Out || !w8Out) {
      throw new Error('F0126 Common-WASM Tap(#7,#8) missing #7 or #8 output');
    }
    const w7F32 = new Float32Array(w7Out.buffer).slice();
    const w7Sha = await sha256HexBytes(
      new Uint8Array(w7F32.buffer, w7F32.byteOffset, w7F32.byteLength)
    );
    const w8F32 = new Float32Array(w8Out.buffer).slice();
    const w8Bytes = copyOutputRawBytes(w8Out);
    const w8Sha = await sha256HexBytes(w8Bytes);

    const tile2Names = suffix7.bypassCutInputs.map((b) => b.name);
    const tapWasmTile = appendOnnxGraphOutputs(params.m1Bytes, tile2Names, { maxExtraOutputs: 2 });
    const tapWasmTileSha = await sha256HexBytes(tapWasmTile.bytes);
    const wTileRun = await runSimpleOrt130({
      label: 'F0126 Common-WASM Tap(Tile_2)',
      modelBytes: tapWasmTile.bytes,
      modelSha: tapWasmTileSha,
      provider: 'wasm',
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      feedF32: params.f0126FeedFloat32,
      feedSha: f0126Sha,
    });
    const wTileVal = validateHitnetFinalOutputs(wTileRun.res.outputs);
    if (
      !wTileVal.ok ||
      !checkFloat32BitIdentity(wTileVal.byName[pName].data, wasmPrimaryF32).bitIdentical ||
      !checkFloat32BitIdentity(wTileVal.byName[sName].data, wasmSecondaryF32).bitIdentical
    ) {
      throw new Error('F0126 Common-WASM Tap(Tile_2) perturbed full WASM final outputs');
    }

    const wasmBypass7 = await Promise.all(
      suffix7.bypassCutInputs.map(async (b) => {
        const found = wTileRun.res.outputs.find((o) => o.name === b.name);
        if (!found) throw new Error(`Missing WASM Tile_2 bypass "${b.name}"`);
        const rawBytes = copyOutputRawBytes(found);
        return {
          name: b.name,
          dims: [...b.dims],
          dataType: b.dataTypeName,
          rawBytes,
          byteLength: rawBytes.byteLength,
          sha256: await sha256HexBytes(rawBytes),
        };
      })
    );

    const gpuSuf7Run = await runSimpleOrt130({
      label: 'F0126 Common-WASM-#7 Baseline GPU Suffix7(wasm#7)',
      modelBytes: suffix7.suffixBytes,
      modelSha: suffix7.suffixSha256,
      provider: 'webgpu',
      feedName: suffix7.primaryCutInput.name,
      feedDims: [...suffix7.primaryCutInput.dims],
      feedF32: w7F32,
      feedSha: w7Sha,
      extraFeeds: wasmBypass7.map((b) => ({
        name: b.name,
        dims: [...b.dims],
        buffer: b.rawBytes.slice().buffer as ArrayBuffer,
        dataType: b.dataType,
      })),
      runSeparateGpuProfilePass: true,
    });
    const gpuSuf7Val = validateHitnetFinalOutputs(gpuSuf7Run.res.outputs);
    if (!gpuSuf7Val.ok) {
      throw new Error(`Common-WASM-#7 GPU Suffix7 invalid outputs: ${gpuSuf7Val.error}`);
    }

    // Bypass inputs for suffix9 using WASM #8 and WASM Tile_2
    const wasmBypass9: typeof prefixBypassList = suffix9.bypassCutInputs.map((b) => {
      if (b.name === node8OutName) {
        return {
          name: node8OutName,
          dims: [...b.dims],
          dataType: 'float32',
          rawBytes: w8Bytes,
          byteLength: w8Bytes.byteLength,
          sha256: w8Sha,
        };
      }
      const match = wasmBypass7.find((x) => x.name === b.name);
      if (!match) throw new Error(`Missing WASM bypass "${b.name}" for Common-WASM-#7 suffix9`);
      return match;
    });

    const gpuSuf9Wasm8Cand = await executeSuffixViaSharedHostTransport({
      label: 'F0126 Common-WASM-#7 Intervened GPU Suffix9(cand128(wasm#8))',
      mode: 'blocked-k128',
      modelBytes: suffix9.suffixBytes,
      modelSha: suffix9.suffixSha256,
      expectedInputOrder: suffix9.graphInputNames,
      primaryInputSpec: suffix9.primaryCutInput,
      bypassInputList: wasmBypass9,
      xDataNode8: w8F32,
      runSeparateProfilePass: true,
    });
    if (!gpuSuf9Wasm8Cand.ok || !gpuSuf9Wasm8Cand.primaryF32 || !gpuSuf9Wasm8Cand.secondaryF32) {
      throw new Error(
        `Common-WASM-#7 Intervened GPU Suffix9 failed: ${gpuSuf9Wasm8Cand.firstFailure?.detail ?? 'unknown'}`
      );
    }

    const baseP = gpuSuf7Val.byName[pName].data;
    const baseS = gpuSuf7Val.byName[sName].data;
    const intP = gpuSuf9Wasm8Cand.primaryF32;
    const intS = gpuSuf9Wasm8Cand.secondaryF32;

    const baseEyeL = computeEyeValidityAndDepthFlips({
      eye: 'left-primary',
      tensorName: pName,
      nativeData: baseP,
      candData: intP,
      wasmData: wasmPrimaryF32,
    });
    const baseEyeR = computeEyeValidityAndDepthFlips({
      eye: 'right-secondary',
      tensorName: sName,
      nativeData: baseS,
      candData: intS,
      wasmData: wasmSecondaryF32,
    });

    commonWasm7Secondary = {
      source: 'fresh-controlled-f0126-arm',
      frameLabel: 'F0126',
      feedSha256: f0126Sha,
      controlScopeNote:
        'Controlled separate F0126 Common-WASM-#7 arm holding #7/#8 and Tile_2 fixed at WASM outputs while injecting blocked_k128(#9) via shared host transport.',
      dispatchConfirmedSeparatePass:
        gpuSuf7Run.dispatchObserved && Boolean(gpuSuf9Wasm8Cand.dispatchObservedSeparatePass),
      baselineGpuSuffixWasm7VsWasmFull: {
        primary: summarizeRawTensorAndParity(baseP, wasmPrimaryF32),
        secondary: summarizeRawTensorAndParity(baseS, wasmSecondaryF32),
        leftValidBaseline: baseEyeL.nativeValidCount,
        leftValidWasm: baseEyeL.wasmValidCount,
        rightValidBaseline: baseEyeR.nativeValidCount,
        rightValidWasm: baseEyeR.wasmValidCount,
      },
      intervenedGpuSuffixWasm7WithCand9VsWasmFull: {
        primary: summarizeRawTensorAndParity(intP, wasmPrimaryF32),
        secondary: summarizeRawTensorAndParity(intS, wasmSecondaryF32),
        primaryFlipsVsWasm: baseEyeL.totalValidityFlipsVsWasm,
        secondaryFlipsVsWasm: baseEyeR.totalValidityFlipsVsWasm,
        leftValidCand: baseEyeL.candValidCount,
        rightValidCand: baseEyeR.candValidCount,
      },
      intervenedVsBaselineGpuSuffix: {
        primary: summarizeRawTensorAndParity(intP, baseP),
        secondary: summarizeRawTensorAndParity(intS, baseS),
      },
    };
  }

  log(
    `  [Common-WASM-#7 Secondary Exposure (${commonWasm7Secondary.source})]\n` +
      `    Baseline(wasm#7) vs Full WASM: S(max=${commonWasm7Secondary.baselineGpuSuffixWasm7VsWasmFull?.secondary.maxAbsDiff.toExponential(3)}, p99=${commonWasm7Secondary.baselineGpuSuffixWasm7VsWasmFull?.secondary.p99AbsDiff.toExponential(3)}, mean=${commonWasm7Secondary.baselineGpuSuffixWasm7VsWasmFull?.secondary.meanAbsDiff.toExponential(3)})\n` +
      `    Cand9(wasm#8)    vs Full WASM: S(max=${commonWasm7Secondary.intervenedGpuSuffixWasm7WithCand9VsWasmFull?.secondary.maxAbsDiff.toExponential(3)}, p99=${commonWasm7Secondary.intervenedGpuSuffixWasm7WithCand9VsWasmFull?.secondary.p99AbsDiff.toExponential(3)}, mean=${commonWasm7Secondary.intervenedGpuSuffixWasm7WithCand9VsWasmFull?.secondary.meanAbsDiff.toExponential(3)})`
  );

  return {
    ok: true,
    aborted: false,
    abortReason: null,
    gatePassed: true,
    gateFirstFailure: null,
    pinnedOrtVersion: '1.30.0',
    runtimeAssetFamily: 'asyncify',
    graphOptimizationLevel: 'disabled',
    numThreads: 1,
    proxy: false,
    m1ByteLength: params.m1Bytes.byteLength,
    m1Sha256: m1Sha,
    f0126FeedSha256: f0126Sha,
    n1OffProof,
    chainSpec,
    verifiedArtifacts: baseVerifiedArtifacts,
    nativeSplitControlCheck: {
      gpuSplitVsFullGpuPrimary: gpuSplitVsFullP,
      gpuSplitVsFullGpuSecondary: gpuSplitVsFullS,
      passed: nativeSplitPassed,
    },
    gateTransportRecord: {
      execution: gateExecSummary,
      vsNativeSplitPrimary: vsNativeSplitP,
      vsNativeSplitSecondary: vsNativeSplitS,
      vsFullGpuDisabledPrimary: vsFullGpuP,
      vsFullGpuDisabledSecondary: vsFullGpuS,
    },
    outlierComparison: {
      cand128TransportOk: cand128Run.ok,
      cand128TransportEdges: cand128Run.edges,
      cand128DispatchObservedSeparatePass: Boolean(cand128Run.dispatchObservedSeparatePass),
      edge9Cand128VsNativeGpu9,
      cand128VsNativeOnCand,
      cand128VsFullWasmDisabled,
      nativeOnCandVsFullWasmDisabled,
      cand128VsFullGpuDisabled,
      eyeValidity: {
        leftPrimary: leftPrimaryValidity,
        rightSecondary: rightSecondaryValidity,
      },
      secondaryClusterThresholdUsed: secondaryClusterInfo.thresholdUsed,
      secondaryOutlierPixelCount: secondaryClusterInfo.outlierPixelCount,
      secondaryClusters: secondaryClusterInfo.clusters,
      topSecondaryOutlierPixels: secondaryClusterInfo.topPixels,
    },
    ancestryAndTaps: {
      ancestrySpec,
      coarseAnalysis,
      lateStageTaps,
    },
    causalAttribution: {
      verdict,
      summary,
      evidencePoints,
    },
    commonWasm7Secondary,
    scientificScopeNote:
      'DEV-only F0126 #9 native-on-candidate transport identity gate and secondary_output_disparity outlier-cause attribution on canonical M1 (N1=OFF, ORT 1.30.0 non-JSEP WebGPU/WASM, opt=disabled, numThreads=1). Bounded <=2-tap passes preserve 0-ULP final output bit-identity.',
  };
}
