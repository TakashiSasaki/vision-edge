import {
  buildVerifiedSuffixAfterConv7,
} from './hitnetConv7ActivationSubstitution';
import {
  appendOnnxGraphOutputs,
  summarizeRawTensorAndParity,
  type RawTensorParityStats,
} from './hitnetGatherNdDiagnostic';
import {
  HITNET_ORT_130_FAMILY_PROFILES,
  type OrtGpuKernelProfileSummary,
  type OrtLoadedAssetProvenance,
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
  sha256HexBytes,
  summarizeDisparityResidualAndSourceDepthImpact,
  validateHitnetFinalOutputs,
  type BitIdentityResult,
  type DisparityAndDepthImpactSummary,
  type ResidualDiagnosticWorkerRunner,
} from './hitnetResidualLocalization';
import {
  buildSingleOpCapturedInputModel,
  compareFloat32TensorsDetailed,
  parseM1ModelProtoWithRawSlices,
  type DetailedFloat32Comparison,
  type ParsedM1InitializerInfo,
  type ParsedM1ModelFull,
} from './hitnetRgbStemFirstNonZero';
import {
  evaluateDisparityRoiMetricsPx,
  generateSyntheticStereoGroundTruthPair,
  type GroundTruthRegionMetricsPx,
  type SyntheticGroundTruthCaseId,
  type SyntheticGroundTruthPair,
} from './hitnetSyntheticGroundTruth';
import {
  assertDevFixtureEnabled,
  HITNET_DEV_NON_JSEP_FIXTURE_SPEC,
} from './stereoDepthWebGpuDevFixture';
import {
  HITNET_M1_FULL_GRAPH_SHA256,
  inferOnnxStaticTensorMeta,
} from './stereoDepthWorkerCore';

export const CONV9_FIXED_SAMPLE_COUNT = 512;

export type Conv9CandidateId =
  | 'flat_ic_kh_kw_bias_last'
  | 'blocked_k64_ic_kh_kw_bias_last'
  | 'blocked_k128_ic_kh_kw_bias_last'
  | 'blocked_k256_ic_kh_kw_bias_last';

export type Conv9AllEvaluatedVariantId = Conv9CandidateId | 'gpu_ref_kh_kw_ic_bias_last';

export const PREDECLARED_CONV9_CANDIDATES: readonly {
  id: Conv9CandidateId;
  blockK: number; // 0 => flat (single block)
  label: string;
}[] = [
  {
    id: 'flat_ic_kh_kw_bias_last',
    blockK: 0,
    label: 'Flat sequential IC->KH->KW f32 sep mul/add + closing bias',
  },
  {
    id: 'blocked_k64_ic_kh_kw_bias_last',
    blockK: 64,
    label: 'Independent K=64 IC->KH->KW f32 blocks, sequential merge rounding + closing bias',
  },
  {
    id: 'blocked_k128_ic_kh_kw_bias_last',
    blockK: 128,
    label: 'Independent K=128 IC->KH->KW f32 blocks, sequential merge rounding + closing bias',
  },
  {
    id: 'blocked_k256_ic_kh_kw_bias_last',
    blockK: 256,
    label: 'Independent K=256 IC->KH->KW f32 blocks, sequential merge rounding + closing bias',
  },
] as const;

export const PREDECLARED_SELECTION_THRESHOLDS = {
  maxAbsDiffVsGpuWasmRatioMax: 0.5,
  p99AbsDiffVsGpuWasmRatioMax: 0.25,
  meanAbsDiffVsGpuWasmRatioMax: 0.25,
  flatMaxNonRegressionSlack: 1e-8,
} as const;

export interface VerifiedPrefixThroughNode {
  cutNodeIdx: number;
  cutNodeName: string;
  prefixBytes: Uint8Array;
  prefixByteLength: number;
  prefixSha256: string;
  prefixNodeCount: number;
  prefixInitializerCount: number;
  prefixInitializerManifestSha256: string;
  graphInputNames: string[];
  outputSpecs: Array<{
    name: string;
    dims: number[];
    elemType: number;
    dataTypeName: 'float32' | 'int32' | 'int64' | 'uint8';
    byteLength: number;
  }>;
  preservedPrefixVerified: boolean;
}

export interface RuntimeProvenanceRecord {
  armLabel: string;
  provider: 'webgpu' | 'wasm';
  graphOptimizationLevel: 'disabled' | 'basic';
  numThreads: number;
  proxy: boolean;
  disableCpuEpFallback: boolean;
  pinnedOrtVersion: '1.30.0';
  runtimeAssetFamily: 'asyncify';
  ortVersions: { common?: string; web?: string } | null;
  entryModuleUrl: string;
  wasmPaths: string;
  workerGpuAdapter: string | null;
  loadedAssetProvenance: OrtLoadedAssetProvenance[];
  loadedOrtResources: string[];
  foreignOrtResources: string[];
  modelByteLength: number;
  modelSha256: string;
  feedSha256: string;
  referenceProvenanceOnlyNote: string;
  separateProfilingPass: {
    executed: boolean;
    dispatchObserved: boolean;
    gpuValidationErrors: string[];
    summary: OrtGpuKernelProfileSummary | null;
    noSilentCpuFallbackNote: string;
  };
}

export interface ConvVariantFixedSampleMetrics {
  variantId: Conv9AllEvaluatedVariantId;
  blockK: number;
  order: 'IC_KH_KW' | 'KH_KW_IC';
  label: string;
  evaluatedCount: number;
  fixedIndexFormula: string;
  vsWasm: {
    exactBitMatches: number;
    ulp0NumericCount: number;
    ulp1Count: number;
    ulp2PlusCount: number;
    maxUlpDiff: number;
    p99UlpDiff: number;
    meanUlpDiff: number;
    meanAbsDiff: number;
    p99AbsDiff: number;
    maxAbsDiff: number;
  };
  vsGpu: {
    exactBitMatches: number;
    ulp0NumericCount: number;
    ulp1Count: number;
    ulp2PlusCount: number;
    maxUlpDiff: number;
    p99UlpDiff: number;
    meanUlpDiff: number;
    meanAbsDiff: number;
    p99AbsDiff: number;
    maxAbsDiff: number;
  };
}

export interface ConvFixedSampleEvaluationReport {
  nodeIdx: number;
  nodeName: string;
  outputTensor: string;
  dims: { N: number; C: number; H: number; W: number; M: number; oH: number; oW: number; kH: number; kW: number; K: number; group: number };
  evaluatedCount: number;
  fixedSampleIndicesSha256: string;
  baselineGpuVsWasmOnFixedIndices: {
    exactBitMatches: number;
    ulp0NumericCount: number;
    ulp1Count: number;
    ulp2PlusCount: number;
    maxUlpDiff: number;
    p99UlpDiff: number;
    meanUlpDiff: number;
    meanAbsDiff: number;
    p99AbsDiff: number;
    maxAbsDiff: number;
  };
  variants: ConvVariantFixedSampleMetrics[];
}

export interface PredeclaredCandidateSelectionResult {
  selectionRuleDescription: string;
  selectedCandidateId: Conv9CandidateId | null;
  selectedBlockK: number | null;
  f0102Promising: boolean;
  f0126HeldOutValidated: boolean;
  advancedToInjection: boolean;
  stopReason: string | null;
  f0102Eligibility: Array<{
    candidateId: Conv9CandidateId;
    maxNonRegressionVsFlat: boolean;
    improvesOverGpuWasmBaseline: boolean;
    f0102MeanAbsDiffVsWasm: number;
    f0102P99AbsDiffVsWasm: number;
    f0102MaxAbsDiffVsWasm: number;
    f0102ExactBitMatchesVsWasm: number;
    eligible: boolean;
    reason: string;
  }>;
  f0126Validation: {
    candidateId: Conv9CandidateId;
    maxNonRegressionVsFlat: boolean;
    improvesOverGpuWasmBaseline: boolean;
    f0126MeanAbsDiffVsWasm: number;
    f0126P99AbsDiffVsWasm: number;
    f0126MaxAbsDiffVsWasm: number;
    f0126ExactBitMatchesVsWasm: number;
    passed: boolean;
    reason: string;
  } | null;
}

export interface Stage1FrameConvVerification {
  frameLabel: 'F0102' | 'F0126';
  feedSha256: string;
  provenances: {
    fullM1GpuOptDisabled: RuntimeProvenanceRecord;
    fullM1WasmOptDisabled: RuntimeProvenanceRecord;
    oneOpConv9Gpu: RuntimeProvenanceRecord;
    oneOpConv9Wasm: RuntimeProvenanceRecord;
  };
  conv7ParentConcatBitwiseEqual: boolean;
  conv7ParentConcatSha256: string;
  conv7ShortControlEval: ConvFixedSampleEvaluationReport;
  node8LeakyReluBitExact: boolean;
  node8Sha256: string;
  conv9FullSuffixVsOneOpBitExactGpu: boolean;
  conv9FullSuffixVsOneOpBitExactWasm: boolean;
  conv9Eval: ConvFixedSampleEvaluationReport;
}

export interface RealFrameSplitControlCheck {
  wasmPrefixBypassEqualToFull: boolean;
  gpuPrefixBypassEqualToWasm: boolean;
  wasmSplitVsUntappedWasmFullPrimary: BitIdentityResult;
  wasmSplitVsUntappedWasmFullSecondary: BitIdentityResult;
  gpuSplitVsUntappedGpuFullPrimary: BitIdentityResult;
  gpuSplitVsUntappedGpuFullSecondary: BitIdentityResult;
  allPassed: boolean;
  failureReason: string | null;
}

export interface DualFinalParityAndValiditySummary {
  comparisonLabel: string;
  primary: DetailedFloat32Comparison;
  secondary: DetailedFloat32Comparison;
  depthAndValidityImpact: DisparityAndDepthImpactSummary;
}

export interface EndToEndPipelineWallTiming {
  timingLabel: string;
  note: string;
  unmodifiedFullGpuAppBaselineOptBasic: {
    createMs: number;
    coldRunMs: number;
    warmRunMs: number[];
    dispatchConfirmedSeparatePass: boolean;
  };
  unmodifiedFullGpuOptDisabled: {
    createMs: number;
    coldRunMs: number;
    dispatchConfirmedSeparatePass: boolean;
  };
  unmodifiedFullWasmOptDisabled: {
    createMs: number;
    coldRunMs: number;
  };
  practicalReplaced9PipelineOptDisabled: {
    prefixModelCreateMs: number;
    prefixRunAndReadbackWorkerMs: number;
    prefixWorkerRoundTripWallMs: number;
    hostCandidate9ComputeMs: number;
    suffixModelCreateMs: number;
    suffixUploadRunReadbackWorkerMs: number;
    suffixWorkerRoundTripWallMs: number;
    orchestrationOverheadMs: number;
    endToEndExecuteWallMs: number;
    endToEndColdCreatePlusRunWallMs: number;
    prefixDispatchConfirmedSeparatePass: boolean;
    suffixDispatchConfirmedSeparatePass: boolean;
    deltaVsUnmodifiedFullGpuAppBaselineColdRunMs: number;
    deltaVsUnmodifiedFullGpuOptDisabledColdRunMs: number;
    deltaVsUnmodifiedFullWasmOptDisabledColdRunMs: number;
  };
}

export interface Stage2RealFrameInjectionReport {
  frameLabel: 'F0102' | 'F0126';
  feedSha256: string;
  ok: boolean;
  aborted: boolean;
  abortReason: string | null;
  splitControls: RealFrameSplitControlCheck;
  candidate9DiffAgainstUnmodifiedGpu9OnGpuPrefixInput: RawTensorParityStats;
  practicalReplaced9VsBaselines: {
    vsUnmodifiedWasmFullOptDisabled: DualFinalParityAndValiditySummary;
    vsUnmodifiedGpuFullOptDisabled: DualFinalParityAndValiditySummary;
    vsUnmodifiedGpuAppBaselineOptBasic: DualFinalParityAndValiditySummary;
    baselineGpuOptDisabledVsWasmOptDisabled: DualFinalParityAndValiditySummary;
    baselineGpuAppBasicVsWasmOptDisabled: DualFinalParityAndValiditySummary;
  };
  causalCommonWasm7Controls: {
    controlScopeNote: string;
    causalBaselineGpuSuffixWasm7VsWasmFull: DualFinalParityAndValiditySummary;
    causalIntervenedGpuSuffixWasm7WithCand9VsWasmFull: DualFinalParityAndValiditySummary;
    causalIntervenedGpuSuffixWasm7WithCand9VsCausalBaselineGpuSuffix: DualFinalParityAndValiditySummary;
    dispatchConfirmedSeparatePass: boolean;
  };
  endToEndWallTiming: EndToEndPipelineWallTiming;
  provenances: {
    fullM1GpuAppBaselineOptBasic: RuntimeProvenanceRecord;
    practicalPrefix0to8Gpu: RuntimeProvenanceRecord;
    practicalSuffix10toEndGpu: RuntimeProvenanceRecord;
  };
}

export interface SyntheticCaseInjectionReport {
  caseId: SyntheticGroundTruthCaseId;
  disparityTruePx: 16 | 32;
  feedSha256: string;
  commonCroppedRoiPixelCount: number;
  analyticValidMaskPixelCount: { primaryLeft: number; secondaryRight: number };
  arms: {
    unmodifiedWasmFullOptDisabled: {
      primary: { commonRoi: GroundTruthRegionMetricsPx; analyticValid: GroundTruthRegionMetricsPx };
      secondary: { commonRoi: GroundTruthRegionMetricsPx; analyticValid: GroundTruthRegionMetricsPx };
      coldRunMs: number;
    };
    unmodifiedGpuFullOptDisabled: {
      primary: { commonRoi: GroundTruthRegionMetricsPx; analyticValid: GroundTruthRegionMetricsPx };
      secondary: { commonRoi: GroundTruthRegionMetricsPx; analyticValid: GroundTruthRegionMetricsPx };
      coldRunMs: number;
      dispatchConfirmedSeparatePass: boolean;
    };
    unmodifiedGpuAppBaselineOptBasic: {
      primary: { commonRoi: GroundTruthRegionMetricsPx; analyticValid: GroundTruthRegionMetricsPx };
      secondary: { commonRoi: GroundTruthRegionMetricsPx; analyticValid: GroundTruthRegionMetricsPx };
      coldRunMs: number;
      dispatchConfirmedSeparatePass: boolean;
    };
    practicalReplaced9PipelineOptDisabled: {
      primary: { commonRoi: GroundTruthRegionMetricsPx; analyticValid: GroundTruthRegionMetricsPx };
      secondary: { commonRoi: GroundTruthRegionMetricsPx; analyticValid: GroundTruthRegionMetricsPx };
      endToEndExecuteWallMs: number;
      hostCandidate9ComputeMs: number;
      prefixDispatchConfirmedSeparatePass: boolean;
      suffixDispatchConfirmedSeparatePass: boolean;
    };
  };
  fullFrameParityVsPracticalReplaced9: {
    vsUnmodifiedWasmOptDisabled: { primary: RawTensorParityStats; secondary: RawTensorParityStats };
    vsUnmodifiedGpuOptDisabled: { primary: RawTensorParityStats; secondary: RawTensorParityStats };
    vsUnmodifiedGpuAppBaselineOptBasic: { primary: RawTensorParityStats; secondary: RawTensorParityStats };
  };
  roiDeltaSummaryPx: {
    primaryEpeReplaced9MinusGpuDisabledPx: number;
    primaryEpeReplaced9MinusGpuAppBasicPx: number;
    primaryBad1DeltaVsGpuDisabled: number;
    primaryInvalidDeltaVsGpuDisabled: number;
    secondaryEpeReplaced9MinusGpuDisabledPx: number;
    secondaryEpeReplaced9MinusGpuAppBasicPx: number;
    secondaryBad1DeltaVsGpuDisabled: number;
    secondaryInvalidDeltaVsGpuDisabled: number;
  };
}

export interface HitnetConv9BlockedInjectionReport {
  ok: boolean;
  aborted: boolean;
  abortReason: string | null;
  pinnedOrtVersion: '1.30.0';
  runtimeAssetFamily: 'asyncify';
  interventionGraphOptimizationLevel: 'disabled';
  appBaselineGraphOptimizationLevel: 'basic';
  numThreads: 1;
  proxy: false;
  m1ByteLength: number;
  m1Sha256: string;
  n1OffProof: N1OffVerificationProof;
  chainSpec: VerifiedPostConv7ChainSpec;
  verifiedArtifacts: {
    prefix0to8ByteLength: number;
    prefix0to8Sha256: string;
    suffix9ByteLength: number;
    suffix9Sha256: string;
    suffix7ByteLength: number;
    suffix7Sha256: string;
    singleOp9ByteLength: number;
    singleOp9Sha256: string;
  };
  stage1: {
    f0102: Stage1FrameConvVerification;
    f0126: Stage1FrameConvVerification;
    selection: PredeclaredCandidateSelectionResult;
  };
  stage2RealFrames: {
    executed: boolean;
    stopReason: string | null;
    f0102: Stage2RealFrameInjectionReport | null;
    f0126: Stage2RealFrameInjectionReport | null;
  };
  stage3SyntheticD16D32: {
    executed: boolean;
    stopReason: string | null;
    coordinateUnitsNote: string;
    D16: SyntheticCaseInjectionReport | null;
    D32: SyntheticCaseInjectionReport | null;
  };
  scientificScopeNote: string;
}

// ---------- Binary protobuf helpers ----------

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

function encodeVarint64(val: number | bigint): Uint8Array {
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

function lenField(fn: number, payload: Uint8Array): Uint8Array {
  return concatBytes([encodeVarint((fn << 3) | 2), encodeVarint(payload.byteLength), payload]);
}

function strField(fn: number, str: string): Uint8Array {
  return lenField(fn, new TextEncoder().encode(str));
}

function varintField(fn: number, val: number | bigint): Uint8Array {
  return concatBytes([encodeVarint((fn << 3) | 0), encodeVarint64(val)]);
}

function makeValueInfo(name: string, elemType: number, dims: readonly number[], fieldNum: 11 | 12): Uint8Array {
  const shape = lenField(
    2,
    concatBytes(dims.map((d) => lenField(1, varintField(1, d))))
  );
  const tensorType = lenField(1, concatBytes([varintField(1, elemType), shape]));
  return lenField(fieldNum, concatBytes([strField(1, name), lenField(2, tensorType)]));
}

function elemTypeToOrtName(elemType: number): 'float32' | 'int32' | 'int64' | 'uint8' {
  switch (elemType) {
    case 1:
      return 'float32';
    case 6:
      return 'int32';
    case 7:
      return 'int64';
    case 2:
      return 'uint8';
    default:
      throw new Error(`Unsupported ONNX cut-edge elemType=${elemType}`);
  }
}

function elemByteSize(elemType: number): number {
  switch (elemType) {
    case 1:
    case 6:
      return 4;
    case 7:
      return 8;
    case 2:
      return 1;
    default:
      throw new Error(`Unsupported ONNX elemType=${elemType}`);
  }
}

export async function buildVerifiedPrefixThroughNode(
  parsed: ParsedM1ModelFull,
  cutNodeIdx: number,
  requiredOutputNames: readonly string[]
): Promise<VerifiedPrefixThroughNode> {
  if (cutNodeIdx < 0 || cutNodeIdx >= parsed.nodes.length) {
    throw new Error(`Invalid prefix cutNodeIdx=${cutNodeIdx}`);
  }
  const prefixNodes = parsed.nodes.slice(0, cutNodeIdx + 1);
  const prefixOutputsProduced = new Set<string>();
  const neededInitializers = new Set<string>();
  const neededGraphInputs = new Set<string>();

  for (const n of prefixNodes) {
    for (const o of n.outputs) {
      if (o) prefixOutputsProduced.add(o);
    }
    for (const inp of n.inputs) {
      if (!inp) continue;
      if (parsed.initializerNames.has(inp)) {
        neededInitializers.add(inp);
      } else if (parsed.graphInputNames.has(inp)) {
        neededGraphInputs.add(inp);
      }
    }
  }

  const preservedInitializers: ParsedM1InitializerInfo[] = [];
  for (const [, initInfo] of parsed.initializersByName) {
    if (!neededInitializers.has(initInfo.name)) continue;
    if (initInfo.isExternal) {
      throw new Error(`Prefix initializer "${initInfo.name}" uses external data_location=1`);
    }
    preservedInitializers.push(initInfo);
  }
  if (preservedInitializers.length !== neededInitializers.size) {
    throw new Error(
      `Prefix initializer mismatch: resolved ${preservedInitializers.length}, needed ${neededInitializers.size}`
    );
  }

  const dedupOutputNames = Array.from(new Set(requiredOutputNames));
  const outputSpecs: VerifiedPrefixThroughNode['outputSpecs'] = [];
  const outputValueInfos: Uint8Array[] = [];
  for (const outName of dedupOutputNames) {
    if (!prefixOutputsProduced.has(outName)) {
      throw new Error(`Required prefix output "${outName}" is not produced in nodes #0..#${cutNodeIdx}`);
    }
    const meta = parsed.staticMeta.tensorMeta[outName];
    if (!meta || meta.dims.some((d) => !Number.isInteger(d) || d <= 0)) {
      throw new Error(`Prefix output "${outName}" has missing/dynamic shape`);
    }
    const dtName = elemTypeToOrtName(meta.elemType);
    outputSpecs.push({
      name: outName,
      dims: [...meta.dims],
      elemType: meta.elemType,
      dataTypeName: dtName,
      byteLength: meta.dims.reduce((a, b) => a * b, 1) * elemByteSize(meta.elemType),
    });
    outputValueInfos.push(makeValueInfo(outName, meta.elemType, meta.dims, 12));
  }

  const graphInputNames = Array.from(neededGraphInputs);
  const inputValueInfos: Uint8Array[] = graphInputNames.map((gInp) => {
    const meta = parsed.staticMeta.tensorMeta[gInp];
    if (!meta || meta.dims.some((d) => !Number.isInteger(d) || d <= 0)) {
      throw new Error(`Prefix graph input "${gInp}" has missing/dynamic shape`);
    }
    return makeValueInfo(gInp, meta.elemType, meta.dims, 11);
  });

  const graphParts: Uint8Array[] = [
    ...prefixNodes.map((n) => lenField(1, n.rawPayloadSlice)),
    strField(2, `hitnet_m1_prefix_through_node_${cutNodeIdx}`),
    ...preservedInitializers.map((init) => lenField(5, init.rawPayloadSlice)),
    ...inputValueInfos,
    ...outputValueInfos,
  ];

  const opsets =
    parsed.opsetImportPayloads.length > 0
      ? parsed.opsetImportPayloads.map((p) => lenField(8, p))
      : [lenField(8, concatBytes([varintField(2, 13)]))];

  const prefixBytes = concatBytes([
    varintField(1, parsed.irVersion),
    ...opsets,
    lenField(7, concatBytes(graphParts)),
  ]);

  const checkMeta = inferOnnxStaticTensorMeta(prefixBytes);
  if (
    !checkMeta.valid ||
    checkMeta.hasControlFlowNode ||
    checkMeta.missingInputs.length > 0 ||
    checkMeta.duplicateOutputs.length > 0 ||
    checkMeta.duplicateInitializers.length > 0
  ) {
    throw new Error(
      `Constructed prefix (#0..#${cutNodeIdx}) failed static ONNX check: missing=${JSON.stringify(checkMeta.missingInputs)}, dupOut=${JSON.stringify(checkMeta.duplicateOutputs)}`
    );
  }

  const reparsed = await parseM1ModelProtoWithRawSlices(prefixBytes);
  const preservedPrefixVerified =
    reparsed.nodes.length === prefixNodes.length &&
    reparsed.nodes.every((rn, idx) => rn.name === prefixNodes[idx].name && rn.opType === prefixNodes[idx].opType);
  if (!preservedPrefixVerified) {
    throw new Error(`Re-parsed prefix (#0..#${cutNodeIdx}) failed 1:1 node preservation check`);
  }

  const prefixSha256 = await sha256HexBytes(prefixBytes);
  const initDigest = await sha256HexBytes(
    new TextEncoder().encode(preservedInitializers.map((i) => `${i.name}:${i.protoSha256}`).join('|'))
  );

  return {
    cutNodeIdx,
    cutNodeName: parsed.nodes[cutNodeIdx].name,
    prefixBytes,
    prefixByteLength: prefixBytes.byteLength,
    prefixSha256,
    prefixNodeCount: prefixNodes.length,
    prefixInitializerCount: preservedInitializers.length,
    prefixInitializerManifestSha256: initDigest,
    graphInputNames,
    outputSpecs,
    preservedPrefixVerified,
  };
}

// ---------- Deterministic fixed indices & IEEE-754 float32 ULP / Conv math ----------

export function buildDeterministicFixedSampleIndices(elemCount: number, sampleCount = CONV9_FIXED_SAMPLE_COUNT): number[] {
  if (elemCount <= 0) return [];
  const count = Math.min(elemCount, Math.max(1, Math.trunc(sampleCount)));
  if (count === elemCount) {
    return Array.from({ length: elemCount }, (_, i) => i);
  }
  const indices: number[] = [];
  let prev = -1;
  for (let i = 0; i < count; i++) {
    let idx = Math.floor(((i + 0.5) * elemCount) / count);
    if (idx <= prev) idx = prev + 1;
    if (idx >= elemCount) idx = elemCount - 1;
    indices.push(idx);
    prev = idx;
  }
  return indices;
}

const _ulpF32 = new Float32Array(1);
const _ulpU32 = new Uint32Array(_ulpF32.buffer);
const _ulpI32 = new Int32Array(_ulpF32.buffer);

function float32Bits(x: number): number {
  _ulpF32[0] = x;
  return _ulpU32[0] >>> 0;
}

function float32OrderedInt(x: number): number | null {
  _ulpF32[0] = x;
  const u = _ulpU32[0] >>> 0;
  // NaN / Inf check (exponent == 0xff)
  if ((u & 0x7f800000) === 0x7f800000) return null;
  if ((u & 0x7fffffff) === 0) return 0; // canonicalize -0.0 and +0.0 numerically
  const signed = _ulpI32[0];
  return signed < 0 ? 0x80000000 - (signed >>> 0) : signed;
}

export function computeFloat32UlpDistance(a: number, b: number): number {
  const oa = float32OrderedInt(a);
  const ob = float32OrderedInt(b);
  if (oa === null || ob === null) {
    return Number.isNaN(a) || Number.isNaN(b) ? Number.POSITIVE_INFINITY : a === b ? 0 : Number.POSITIVE_INFINITY;
  }
  return Math.abs(oa - ob);
}

function getConvAttrs(parsed: ParsedM1ModelFull, convNodeIdx: number) {
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
    node,
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

export function computeSingleConvElementVariant(params: {
  xData: Float32Array;
  wData: Float32Array;
  bData: Float32Array | null;
  N: number;
  C: number;
  H: number;
  W: number;
  M: number;
  oH: number;
  oW: number;
  CPerG: number;
  kH: number;
  kW: number;
  group: number;
  strideH: number;
  strideW: number;
  dilH: number;
  dilW: number;
  padTop: number;
  padLeft: number;
  flatIdx: number;
  variantId: Conv9AllEvaluatedVariantId;
}): number {
  const {
    xData,
    wData,
    bData,
    C,
    H,
    W,
    M,
    oH,
    oW,
    CPerG,
    kH,
    kW,
    group,
    strideH,
    strideW,
    dilH,
    dilW,
    padTop,
    padLeft,
    flatIdx,
    variantId,
  } = params;
  const f32 = Math.fround;
  let rem = flatIdx;
  const ow = rem % oW;
  rem = Math.floor(rem / oW);
  const oh = rem % oH;
  rem = Math.floor(rem / oH);
  const m = rem % M;
  const n = Math.floor(rem / M);

  const gSize = M / group;
  const g = Math.floor(m / gSize);
  const cStart = g * CPerG;
  const xBase = n * C * H * W;
  const wBase = m * CPerG * kH * kW;
  const bias = bData ? f32(bData[m]) : 0.0;

  if (variantId === 'gpu_ref_kh_kw_ic_bias_last') {
    let acc = 0.0;
    for (let kh = 0; kh < kH; kh++) {
      const ih = oh * strideH + kh * dilH - padTop;
      for (let kw = 0; kw < kW; kw++) {
        const iw = ow * strideW + kw * dilW - padLeft;
        for (let ic = 0; ic < CPerG; ic++) {
          const xv =
            ih >= 0 && ih < H && iw >= 0 && iw < W
              ? xData[xBase + (cStart + ic) * H * W + ih * W + iw]
              : 0.0;
          const wv = wData[wBase + (ic * kH + kh) * kW + kw];
          acc = f32(acc + f32(xv * wv));
        }
      }
    }
    return f32(acc + bias);
  }

  const blockK =
    variantId === 'blocked_k64_ic_kh_kw_bias_last'
      ? 64
      : variantId === 'blocked_k128_ic_kh_kw_bias_last'
        ? 128
        : variantId === 'blocked_k256_ic_kh_kw_bias_last'
          ? 256
          : CPerG * kH * kW;

  let merged = 0.0;
  let blockAcc = 0.0;
  let inBlockCount = 0;

  for (let ic = 0; ic < CPerG; ic++) {
    const xChanOff = xBase + (cStart + ic) * H * W;
    const wChanOff = wBase + ic * kH * kW;
    for (let kh = 0; kh < kH; kh++) {
      const ih = oh * strideH + kh * dilH - padTop;
      const wRowOff = wChanOff + kh * kW;
      for (let kw = 0; kw < kW; kw++) {
        const iw = ow * strideW + kw * dilW - padLeft;
        const xv = ih >= 0 && ih < H && iw >= 0 && iw < W ? xData[xChanOff + ih * W + iw] : 0.0;
        const wv = wData[wRowOff + kw];
        blockAcc = f32(blockAcc + f32(xv * wv));
        inBlockCount++;
        if (inBlockCount === blockK) {
          merged = f32(merged + blockAcc);
          blockAcc = 0.0;
          inBlockCount = 0;
        }
      }
    }
  }
  if (inBlockCount > 0) {
    merged = f32(merged + blockAcc);
  }
  return f32(merged + bias);
}

export async function evaluateConvReductionCandidatesOnFixedIndices(params: {
  parsed: ParsedM1ModelFull;
  convNodeIdx: number;
  inputF32: Float32Array;
  inputDims: readonly number[];
  gpuOutF32: Float32Array;
  wasmOutF32: Float32Array;
  outDims: readonly number[];
  sampleCount?: number;
}): Promise<ConvFixedSampleEvaluationReport> {
  const { parsed, convNodeIdx, inputF32, inputDims, gpuOutF32, wasmOutF32, outDims } = params;
  const attrs = getConvAttrs(parsed, convNodeIdx);
  const [N, C, H, W] = inputDims;
  const [M, CPerG, kH, kW] = attrs.wDims;
  const [, oM, oH, oW] = outDims;
  if (attrs.group * CPerG !== C || M !== oM || M % attrs.group !== 0) {
    throw new Error(
      `Invalid Conv #${convNodeIdx} dims/group: in=${JSON.stringify(inputDims)}, w=${JSON.stringify(attrs.wDims)}, out=${JSON.stringify(outDims)}, group=${attrs.group}`
    );
  }
  const elemCount = gpuOutF32.length;
  if (wasmOutF32.length !== elemCount || N * oM * oH * oW !== elemCount) {
    throw new Error(`Conv #${convNodeIdx} output length mismatch: gpu=${elemCount}, wasm=${wasmOutF32.length}`);
  }

  const fixedIndices = buildDeterministicFixedSampleIndices(
    elemCount,
    params.sampleCount ?? CONV9_FIXED_SAMPLE_COUNT
  );
  const idxBuf = new Uint8Array(new Int32Array(fixedIndices).buffer);
  const fixedSampleIndicesSha256 = await sha256HexBytes(idxBuf);

  const summarizeVs = (preds: Float32Array, refs: Float32Array) => {
    let exactBitMatches = 0;
    let ulp0NumericCount = 0;
    let ulp1Count = 0;
    let ulp2PlusCount = 0;
    let maxUlpDiff = 0;
    let ulpSum = 0;
    let absSum = 0;
    let maxAbsDiff = 0;
    const ulps: number[] = [];
    const absList: number[] = [];

    for (let i = 0; i < fixedIndices.length; i++) {
      const idx = fixedIndices[i];
      const p = preds[i];
      const r = refs[idx];
      if (float32Bits(p) === float32Bits(r)) exactBitMatches++;
      const ulp = computeFloat32UlpDistance(p, r);
      ulps.push(ulp);
      if (ulp === 0) ulp0NumericCount++;
      else if (ulp === 1) ulp1Count++;
      else ulp2PlusCount++;
      if (ulp > maxUlpDiff) maxUlpDiff = ulp;
      ulpSum += Number.isFinite(ulp) ? ulp : 0;

      const abs = Math.abs(p - r);
      absList.push(abs);
      if (abs > maxAbsDiff) maxAbsDiff = abs;
      absSum += Number.isFinite(abs) ? abs : 0;
    }

    ulps.sort((a, b) => a - b);
    absList.sort((a, b) => a - b);
    const p99Pos = Math.min(fixedIndices.length - 1, Math.floor(0.99 * Math.max(0, fixedIndices.length - 1)));

    return {
      exactBitMatches,
      ulp0NumericCount,
      ulp1Count,
      ulp2PlusCount,
      maxUlpDiff,
      p99UlpDiff: ulps[p99Pos] ?? 0,
      meanUlpDiff: fixedIndices.length > 0 ? ulpSum / fixedIndices.length : 0,
      meanAbsDiff: fixedIndices.length > 0 ? absSum / fixedIndices.length : 0,
      p99AbsDiff: absList[p99Pos] ?? 0,
      maxAbsDiff,
    };
  };

  const gpuSampleVals = new Float32Array(fixedIndices.length);
  for (let i = 0; i < fixedIndices.length; i++) {
    gpuSampleVals[i] = gpuOutF32[fixedIndices[i]];
  }
  const baselineGpuVsWasmOnFixedIndices = summarizeVs(gpuSampleVals, wasmOutF32);

  const variantSpecs: Array<{
    id: Conv9AllEvaluatedVariantId;
    blockK: number;
    order: 'IC_KH_KW' | 'KH_KW_IC';
    label: string;
  }> = [
    ...PREDECLARED_CONV9_CANDIDATES.map((c) => ({
      id: c.id as Conv9AllEvaluatedVariantId,
      blockK: c.blockK,
      order: 'IC_KH_KW' as const,
      label: c.label,
    })),
    {
      id: 'gpu_ref_kh_kw_ic_bias_last',
      blockK: 0,
      order: 'KH_KW_IC',
      label: 'Existing WebGPU reference KH->KW->IC f32 sep mul/add + closing bias',
    },
  ];

  const variants: ConvVariantFixedSampleMetrics[] = [];
  for (const vs of variantSpecs) {
    const preds = new Float32Array(fixedIndices.length);
    for (let i = 0; i < fixedIndices.length; i++) {
      preds[i] = computeSingleConvElementVariant({
        xData: inputF32,
        wData: attrs.wData,
        bData: attrs.bData,
        N,
        C,
        H,
        W,
        M,
        oH,
        oW,
        CPerG,
        kH,
        kW,
        group: attrs.group,
        strideH: attrs.strideH,
        strideW: attrs.strideW,
        dilH: attrs.dilH,
        dilW: attrs.dilW,
        padTop: attrs.padTop,
        padLeft: attrs.padLeft,
        flatIdx: fixedIndices[i],
        variantId: vs.id,
      });
    }
    variants.push({
      variantId: vs.id,
      blockK: vs.blockK,
      order: vs.order,
      label: vs.label,
      evaluatedCount: fixedIndices.length,
      fixedIndexFormula: `floor((i + 0.5) * ${elemCount} / ${fixedIndices.length})`,
      vsWasm: summarizeVs(preds, wasmOutF32),
      vsGpu: summarizeVs(preds, gpuOutF32),
    });
  }

  return {
    nodeIdx: convNodeIdx,
    nodeName: attrs.node.name,
    outputTensor: attrs.node.outputs[0] ?? '',
    dims: { N, C, H, W, M, oH, oW, kH, kW, K: CPerG * kH * kW, group: attrs.group },
    evaluatedCount: fixedIndices.length,
    fixedSampleIndicesSha256,
    baselineGpuVsWasmOnFixedIndices,
    variants,
  };
}

export function selectAndValidatePredeclaredCandidate(
  f0102Eval: ConvFixedSampleEvaluationReport,
  f0126Eval: ConvFixedSampleEvaluationReport
): PredeclaredCandidateSelectionResult {
  const findCand = (evalRep: ConvFixedSampleEvaluationReport, id: Conv9CandidateId) => {
    const v = evalRep.variants.find((x) => x.variantId === id);
    if (!v) throw new Error(`Missing evaluated candidate "${id}"`);
    return v;
  };

  const flat102 = findCand(f0102Eval, 'flat_ic_kh_kw_bias_last');
  const base102 = f0102Eval.baselineGpuVsWasmOnFixedIndices;

  const eligibilityList: PredeclaredCandidateSelectionResult['f0102Eligibility'] = [];
  const passingOnF0102: Array<{
    id: Conv9CandidateId;
    blockK: number;
    mean: number;
    p99: number;
    max: number;
  }> = [];

  for (const spec of PREDECLARED_CONV9_CANDIDATES) {
    const c = findCand(f0102Eval, spec.id);
    const maxNonRegressionVsFlat =
      c.vsWasm.maxAbsDiff <= flat102.vsWasm.maxAbsDiff + PREDECLARED_SELECTION_THRESHOLDS.flatMaxNonRegressionSlack;
    const baselineHasDiff = base102.maxAbsDiff > 0 && base102.meanAbsDiff > 0;
    const improvesOverGpuWasmBaseline = baselineHasDiff
      ? c.vsWasm.maxAbsDiff <= base102.maxAbsDiff * PREDECLARED_SELECTION_THRESHOLDS.maxAbsDiffVsGpuWasmRatioMax &&
        c.vsWasm.p99AbsDiff <= base102.p99AbsDiff * PREDECLARED_SELECTION_THRESHOLDS.p99AbsDiffVsGpuWasmRatioMax &&
        c.vsWasm.meanAbsDiff <= base102.meanAbsDiff * PREDECLARED_SELECTION_THRESHOLDS.meanAbsDiffVsGpuWasmRatioMax
      : false;
    const eligible = maxNonRegressionVsFlat && improvesOverGpuWasmBaseline;
    const reason = !baselineHasDiff
      ? 'Baseline GPU vs WASM #9 diff is zero on F0102 fixed indices.'
      : !maxNonRegressionVsFlat
        ? `Fails max non-regression vs flat on F0102 (candMax=${c.vsWasm.maxAbsDiff.toExponential(3)} > flatMax=${flat102.vsWasm.maxAbsDiff.toExponential(3)}).`
        : !improvesOverGpuWasmBaseline
          ? `Insufficient F0102 reduction vs GPU/WASM baseline (meanRatio=${(c.vsWasm.meanAbsDiff / base102.meanAbsDiff).toFixed(3)}, p99Ratio=${(base102.p99AbsDiff > 0 ? c.vsWasm.p99AbsDiff / base102.p99AbsDiff : 0).toFixed(3)}, maxRatio=${(c.vsWasm.maxAbsDiff / base102.maxAbsDiff).toFixed(3)}).`
          : `Eligible on F0102 (mean=${c.vsWasm.meanAbsDiff.toExponential(3)}, p99=${c.vsWasm.p99AbsDiff.toExponential(3)}, max=${c.vsWasm.maxAbsDiff.toExponential(3)}, bitExact=${c.vsWasm.exactBitMatches}/${c.evaluatedCount}).`;

    eligibilityList.push({
      candidateId: spec.id,
      maxNonRegressionVsFlat,
      improvesOverGpuWasmBaseline,
      f0102MeanAbsDiffVsWasm: c.vsWasm.meanAbsDiff,
      f0102P99AbsDiffVsWasm: c.vsWasm.p99AbsDiff,
      f0102MaxAbsDiffVsWasm: c.vsWasm.maxAbsDiff,
      f0102ExactBitMatchesVsWasm: c.vsWasm.exactBitMatches,
      eligible,
      reason,
    });
    if (eligible) {
      passingOnF0102.push({
        id: spec.id,
        blockK: spec.blockK,
        mean: c.vsWasm.meanAbsDiff,
        p99: c.vsWasm.p99AbsDiff,
        max: c.vsWasm.maxAbsDiff,
      });
    }
  }

  passingOnF0102.sort((a, b) => a.mean - b.mean || a.p99 - b.p99 || a.max - b.max);
  const winner = passingOnF0102[0] ?? null;

  const selectionRuleDescription =
    'Select at most 1 candidate from {flat, K=64, K=128, K=256} strictly on F0102 fixed indices by minimizing (meanAbsDiff, p99AbsDiff, maxAbsDiff) vs WASM #9, subject to maxAbsDiff <= flat.maxAbsDiff + 1e-8 and mean<=0.25x, p99<=0.25x, max<=0.5x of baseline GPU-vs-WASM #9 diff (0-ULP NOT required); then validate on held-out F0126.';

  if (!winner) {
    return {
      selectionRuleDescription,
      selectedCandidateId: null,
      selectedBlockK: null,
      f0102Promising: false,
      f0126HeldOutValidated: false,
      advancedToInjection: false,
      stopReason: 'No predeclared #9 candidate met promising F0102 mean/p99 + max non-regression criteria.',
      f0102Eligibility: eligibilityList,
      f0126Validation: null,
    };
  }

  const cand126 = findCand(f0126Eval, winner.id);
  const flat126 = findCand(f0126Eval, 'flat_ic_kh_kw_bias_last');
  const base126 = f0126Eval.baselineGpuVsWasmOnFixedIndices;

  const maxNonRegression126 =
    cand126.vsWasm.maxAbsDiff <= flat126.vsWasm.maxAbsDiff + PREDECLARED_SELECTION_THRESHOLDS.flatMaxNonRegressionSlack;
  const baseline126HasDiff = base126.maxAbsDiff > 0 && base126.meanAbsDiff > 0;
  const improves126 = baseline126HasDiff
    ? cand126.vsWasm.maxAbsDiff <= base126.maxAbsDiff * PREDECLARED_SELECTION_THRESHOLDS.maxAbsDiffVsGpuWasmRatioMax &&
      cand126.vsWasm.p99AbsDiff <= base126.p99AbsDiff * PREDECLARED_SELECTION_THRESHOLDS.p99AbsDiffVsGpuWasmRatioMax &&
      cand126.vsWasm.meanAbsDiff <= base126.meanAbsDiff * PREDECLARED_SELECTION_THRESHOLDS.meanAbsDiffVsGpuWasmRatioMax
    : false;
  const f0126Passed = maxNonRegression126 && improves126;

  return {
    selectionRuleDescription,
    selectedCandidateId: winner.id,
    selectedBlockK: winner.blockK,
    f0102Promising: true,
    f0126HeldOutValidated: f0126Passed,
    advancedToInjection: f0126Passed,
    stopReason: f0126Passed
      ? null
      : `Selected F0102 candidate "${winner.id}" failed held-out F0126 validation (mean=${cand126.vsWasm.meanAbsDiff.toExponential(3)}, p99=${cand126.vsWasm.p99AbsDiff.toExponential(3)}, max=${cand126.vsWasm.maxAbsDiff.toExponential(3)}).`,
    f0102Eligibility: eligibilityList,
    f0126Validation: {
      candidateId: winner.id,
      maxNonRegressionVsFlat: maxNonRegression126,
      improvesOverGpuWasmBaseline: improves126,
      f0126MeanAbsDiffVsWasm: cand126.vsWasm.meanAbsDiff,
      f0126P99AbsDiffVsWasm: cand126.vsWasm.p99AbsDiff,
      f0126MaxAbsDiffVsWasm: cand126.vsWasm.maxAbsDiff,
      f0126ExactBitMatchesVsWasm: cand126.vsWasm.exactBitMatches,
      passed: f0126Passed,
      reason: f0126Passed
        ? `Held-out F0126 validated (mean=${cand126.vsWasm.meanAbsDiff.toExponential(3)}, p99=${cand126.vsWasm.p99AbsDiff.toExponential(3)}, max=${cand126.vsWasm.maxAbsDiff.toExponential(3)}, bitExact=${cand126.vsWasm.exactBitMatches}/${cand126.evaluatedCount}).`
        : `Held-out F0126 failed (maxNonReg=${maxNonRegression126}, improves=${improves126}).`,
    },
  };
}

// ---------- Full-Tensor Candidate Conv2D Host Computation ----------

export function computeConv2dCandidateFullTensor(params: {
  parsed: ParsedM1ModelFull;
  convNodeIdx: number;
  xData: Float32Array;
  inputDims: readonly [number, number, number, number];
  outDims: readonly [number, number, number, number];
  candidateId: Conv9CandidateId;
}): Float32Array {
  const { parsed, convNodeIdx, xData, inputDims, outDims, candidateId } = params;
  const attrs = getConvAttrs(parsed, convNodeIdx);
  const [N, C, H, W] = inputDims;
  const [M, CPerG, kH, kW] = attrs.wDims;
  const [, oM, oH, oW] = outDims;
  const { wData, bData, group, strideH, strideW, dilH, dilW, padTop, padLeft } = attrs;

  if (N * oM * oH * oW <= 0 || M !== oM || C !== group * CPerG) {
    throw new Error(`Invalid full-tensor Conv shapes: in=${inputDims.join('x')}, out=${outDims.join('x')}`);
  }

  const out = new Float32Array(N * oM * oH * oW);
  const K = CPerG * kH * kW;
  const blockK =
    candidateId === 'blocked_k64_ic_kh_kw_bias_last'
      ? 64
      : candidateId === 'blocked_k128_ic_kh_kw_bias_last'
        ? 128
        : candidateId === 'blocked_k256_ic_kh_kw_bias_last'
          ? 256
          : K;

  const f32 = Math.fround;
  const gSize = M / group;
  const col = new Float32Array(K);
  const spatialOut = oH * oW;
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
                col[k++] = xData[rOff];
                col[k++] = xData[rOff + 1];
                col[k++] = xData[rOff + 2];
                col[k++] = xData[rOff + 3];
                rOff += W;
                col[k++] = xData[rOff];
                col[k++] = xData[rOff + 1];
                col[k++] = xData[rOff + 2];
                col[k++] = xData[rOff + 3];
                rOff += W;
                col[k++] = xData[rOff];
                col[k++] = xData[rOff + 1];
                col[k++] = xData[rOff + 2];
                col[k++] = xData[rOff + 3];
                rOff += W;
                col[k++] = xData[rOff];
                col[k++] = xData[rOff + 1];
                col[k++] = xData[rOff + 2];
                col[k++] = xData[rOff + 3];
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
                    col[k++] = w0 >= 0 && w0 < W ? xData[rOff + w0] : 0;
                    col[k++] = w0 + 1 >= 0 && w0 + 1 < W ? xData[rOff + w0 + 1] : 0;
                    col[k++] = w0 + 2 >= 0 && w0 + 2 < W ? xData[rOff + w0 + 2] : 0;
                    col[k++] = w0 + 3 >= 0 && w0 + 3 < W ? xData[rOff + w0 + 3] : 0;
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
                    col[k++] = iw >= 0 && iw < W ? xData[rOff + iw] : 0;
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

// ---------- Provenance & Helpers ----------

function hasObservedGpuDispatch(prof: OrtVersionDiagnosticResponse | null): boolean {
  if (!prof || !prof.ok || !prof.gpuKernelProfileSummary) return false;
  const s = prof.gpuKernelProfileSummary;
  return (
    s.webgpuComputePassCount > 0 ||
    s.webgpuDispatchWorkgroupsCount > 0 ||
    s.ortProfilingEventCount > 0
  );
}

function buildRuntimeProvenanceRecord(
  armLabel: string,
  res: OrtVersionDiagnosticSuccess,
  profilingPass: OrtVersionDiagnosticResponse | null
): RuntimeProvenanceRecord {
  const dispatchObserved =
    res.provider === 'webgpu' ? hasObservedGpuDispatch(profilingPass) : false;
  return {
    armLabel,
    provider: res.provider,
    graphOptimizationLevel: res.graphOptimizationLevel,
    numThreads: res.sessionOptionsApplied.numThreads,
    proxy: res.sessionOptionsApplied.proxy,
    disableCpuEpFallback: res.sessionOptionsApplied.disableCpuEpFallback,
    pinnedOrtVersion: '1.30.0',
    runtimeAssetFamily: 'asyncify',
    ortVersions: res.ortVersions,
    entryModuleUrl: res.entryModuleUrl,
    wasmPaths: res.wasmPaths,
    workerGpuAdapter: res.workerGpuAdapter,
    loadedAssetProvenance: res.loadedAssetProvenance,
    loadedOrtResources: res.loadedOrtResources,
    foreignOrtResources: res.foreignOrtResources,
    modelByteLength: res.modelByteLength,
    modelSha256: res.modelSha256,
    feedSha256: res.feedSha256,
    referenceProvenanceOnlyNote:
      'Asset URLs and SHA-256 hashes prove downloaded/instantiated ORT Web 1.30.0 asyncify runtime identity only; they do NOT prove WebGPU kernel placement without a separate dispatch-profiling pass.',
    separateProfilingPass: {
      executed: Boolean(profilingPass),
      dispatchObserved,
      gpuValidationErrors: profilingPass?.ok ? profilingPass.gpuValidationErrors ?? [] : [],
      summary: profilingPass?.ok ? profilingPass.gpuKernelProfileSummary ?? null : null,
      noSilentCpuFallbackNote:
        res.provider === 'webgpu'
          ? 'session.disable_cpu_ep_fallback / explicit ["webgpu"] provider guard + separate WebGPU profiling pass.'
          : 'Intentional WASM CPU baseline.',
    },
  };
}

function copyOutputRawBytes(o: OrtVersionDiagnosticOutput): Uint8Array {
  return new Uint8Array(o.buffer).slice();
}

function buildDualFinalSummary(
  comparisonLabel: string,
  frameLabel: string,
  candByName: Record<string, { data: Float32Array }>,
  refByName: Record<string, { data: Float32Array }>
): DualFinalParityAndValiditySummary {
  const [pName, sName] = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;
  const primary = compareFloat32TensorsDetailed(
    candByName[pName].data,
    refByName[pName].data,
    [1, 240, 320, 1]
  );
  const secondary = compareFloat32TensorsDetailed(
    candByName[sName].data,
    refByName[sName].data,
    [1, 240, 320, 1]
  );
  const depthAndValidityImpact = summarizeDisparityResidualAndSourceDepthImpact(
    frameLabel,
    candByName[pName].data,
    refByName[pName].data,
    candByName[sName].data,
    refByName[sName].data
  );
  return {
    comparisonLabel,
    primary,
    secondary,
    depthAndValidityImpact,
  };
}

// ---------- Main Orchestrator ----------

export async function runHitnetConv9BlockedInjectionDiagnostic(params: {
  m1Bytes: Uint8Array;
  f0102FeedFloat32: Float32Array;
  f0126FeedFloat32: Float32Array;
  normalizeSlot0ReshapeToTranspose?: false;
  runWorker?: ResidualDiagnosticWorkerRunner;
  disableCpuEpFallback?: boolean;
  warmRunsAppBaseline?: number;
  isDev?: boolean;
  onLog?: (line: string) => void;
}): Promise<HitnetConv9BlockedInjectionReport> {
  assertDevFixtureEnabled(params.isDev);
  const log = params.onLog ?? (() => {});
  const runWorker = params.runWorker ?? createBrowserOrt130DiagnosticRunner();
  const disableCpuEpFallback = Boolean(params.disableCpuEpFallback);
  const warmRunsAppBaseline = Math.max(0, Math.min(3, Math.trunc(params.warmRunsAppBaseline ?? 1)));

  if ((params as { normalizeSlot0ReshapeToTranspose?: boolean }).normalizeSlot0ReshapeToTranspose) {
    throw new Error('N1-Slot0Transpose MUST be explicitly OFF (false) for #9 diagnostic.');
  }

  const m1Sha = await sha256HexBytes(params.m1Bytes);
  if (
    params.m1Bytes.byteLength !== HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength ||
    m1Sha !== HITNET_M1_FULL_GRAPH_SHA256
  ) {
    throw new Error(
      `Exact canonical M1 verification failed: bytes=${params.m1Bytes.byteLength}, sha=${m1Sha}`
    );
  }

  const f0102Sha = await sha256HexBytes(
    new Uint8Array(
      params.f0102FeedFloat32.buffer,
      params.f0102FeedFloat32.byteOffset,
      params.f0102FeedFloat32.byteLength
    )
  );
  if (f0102Sha !== HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256) {
    throw new Error(
      `Frozen F0102 feed SHA mismatch: expected=${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256}, actual=${f0102Sha}`
    );
  }

  const f0126Sha = await sha256HexBytes(
    new Uint8Array(
      params.f0126FeedFloat32.buffer,
      params.f0126FeedFloat32.byteOffset,
      params.f0126FeedFloat32.byteLength
    )
  );
  if (f0126Sha === f0102Sha || f0126Sha !== HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0126FeedSha256) {
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
  const suffix7 = await buildVerifiedSuffixAfterConv7(parsedModel, { cutNodeIdx: chainSpec.conv7Idx });
  const suffix9 = await buildVerifiedSuffixAfterConv7(parsedModel, { cutNodeIdx: chainSpec.node9ConvDown0.nodeIdx });
  const conv7Node = parsedModel.nodes[chainSpec.conv7Idx];
  const conv7DynInput = conv7Node.inputs.find((inp) => inp && !parsedModel.initializerNames.has(inp));
  if (!conv7DynInput) {
    throw new Error(`Conv #${chainSpec.conv7Idx} has no dynamic input tensor`);
  }
  const conv7DynInputMeta = parsedModel.staticMeta.tensorMeta[conv7DynInput];
  if (!conv7DynInputMeta) {
    throw new Error(`Missing static metadata for #7 dynamic input "${conv7DynInput}"`);
  }

  const node8OutName = chainSpec.node8LeakyRelu.outputTensor;
  const prefixOutputsNeeded = [
    node8OutName,
    ...suffix9.bypassCutInputs.map((b) => b.name),
  ];
  const prefix0to8 = await buildVerifiedPrefixThroughNode(
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
    `=== [DEV-ONLY] HITNet #9 Blocked Accumulation & Practical Injection Comparison ===\n` +
      `Runtime target: onnxruntime-web@1.30.0/webgpu (non-JSEP ${asyncify.entryFile} + ${asyncify.wasmFile}) | interventions opt=disabled | numThreads=1\n` +
      `Canonical M1: bytes=${params.m1Bytes.byteLength} sha256=${m1Sha} | N1=OFF(0 replacements)\n` +
      `Verified Prefix (#0..#${prefix0to8.cutNodeIdx}): bytes=${prefix0to8.prefixByteLength} sha=${prefix0to8.prefixSha256.slice(0, 12)}... outputs=${JSON.stringify(prefix0to8.outputSpecs.map((o) => o.name))}\n` +
      `Verified Suffix (#${suffix9.suffixNodeRange.firstOriginalIdx}..#${suffix9.suffixNodeRange.lastOriginalIdx}): bytes=${suffix9.suffixByteLength} sha=${suffix9.suffixSha256.slice(0, 12)}... bypassInputs=${JSON.stringify(suffix9.bypassCutInputs.map((b) => b.name))}`
  );

  const assertOrt130Success = (
    label: string,
    raw: OrtVersionDiagnosticResponse,
    expectedProvider: 'webgpu' | 'wasm',
    expectedOpt: 'disabled' | 'basic',
    expectedModelSha: string,
    expectedModelBytes: number,
    expectedFeedSha?: string
  ): OrtVersionDiagnosticSuccess => {
    if (!raw.ok) {
      const f = raw as OrtVersionDiagnosticFailure;
      throw new Error(`${label} FAILED at stage="${f.stage}": ${f.message}`);
    }
    if (
      raw.pinnedOrtVersion !== '1.30.0' ||
      raw.runtimeAssetFamily !== 'asyncify' ||
      raw.provider !== expectedProvider ||
      raw.graphOptimizationLevel !== expectedOpt ||
      raw.sessionOptionsApplied.numThreads !== 1 ||
      raw.foreignOrtResources.length > 0 ||
      raw.loadedOrtResources.some((u) => u.includes('.jsep.') || u.includes('ort.bundle.'))
    ) {
      throw new Error(
        `${label} ORT 1.30.0 non-JSEP / opt=${expectedOpt} / 1-thread guard failed: opt=${raw.graphOptimizationLevel}, ep=${raw.provider}, threads=${raw.sessionOptionsApplied.numThreads}, foreign=${JSON.stringify(raw.foreignOrtResources)}`
      );
    }
    if (raw.modelSha256 !== expectedModelSha || raw.modelByteLength !== expectedModelBytes) {
      throw new Error(
        `${label} model verification failed: got=${raw.modelSha256}(${raw.modelByteLength}B), expected=${expectedModelSha}(${expectedModelBytes}B)`
      );
    }
    if (expectedFeedSha && raw.feedSha256 !== expectedFeedSha) {
      throw new Error(`${label} feed SHA mismatch: got=${raw.feedSha256}, expected=${expectedFeedSha}`);
    }
    return raw;
  };

  // Helper to run any model cleanly + optional separate profiling pass
  const runSingleModel = async (opts: {
    label: string;
    modelBytes: Uint8Array;
    modelSha: string;
    provider: 'webgpu' | 'wasm';
    opt: 'disabled' | 'basic';
    feedName: string;
    feedDims: number[];
    feedF32: Float32Array;
    feedSha: string;
    extraFeeds?: Array<{ name: string; dims: number[]; buffer: ArrayBuffer; dataType: 'float32' | 'int32' | 'int64' | 'uint8' }>;
    allowArbitraryOutputs?: boolean;
    warmRuns?: number;
    runSeparateGpuProfilePass?: boolean;
  }) => {
    const resRaw = await runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: opts.modelBytes.slice().buffer,
      feedBuffer: opts.feedF32.slice().buffer,
      feedName: opts.feedName,
      feedDims: [...opts.feedDims],
      extraFeeds: opts.extraFeeds?.map((e) => ({ ...e, buffer: e.buffer.slice(0) })),
      allowArbitraryOutputs: opts.allowArbitraryOutputs,
      provider: opts.provider,
      graphOptimizationLevel: opts.opt,
      warmRuns: opts.warmRuns ?? 0,
      enableWebGpuProfiling: false,
      disableCpuEpFallback: opts.provider === 'webgpu' ? disableCpuEpFallback : false,
      captureGpuValidationErrors: opts.provider === 'webgpu',
    });
    const okRes = assertOrt130Success(
      opts.label,
      resRaw,
      opts.provider,
      opts.opt,
      opts.modelSha,
      opts.modelBytes.byteLength,
      opts.feedSha
    );

    let profRaw: OrtVersionDiagnosticResponse | null = null;
    if (opts.runSeparateGpuProfilePass && opts.provider === 'webgpu') {
      profRaw = await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: opts.modelBytes.slice().buffer,
        feedBuffer: opts.feedF32.slice().buffer,
        feedName: opts.feedName,
        feedDims: [...opts.feedDims],
        extraFeeds: opts.extraFeeds?.map((e) => ({ ...e, buffer: e.buffer.slice(0) })),
        allowArbitraryOutputs: opts.allowArbitraryOutputs,
        provider: 'webgpu',
        graphOptimizationLevel: opts.opt,
        warmRuns: 0,
        enableWebGpuProfiling: true,
        disableCpuEpFallback,
        captureGpuValidationErrors: true,
      });
    }
    return {
      res: okRes,
      provenance: buildRuntimeProvenanceRecord(opts.label, okRes, profRaw),
    };
  };

  // ---------------- STAGE 1: Independent #9 + #7 Short Control Verification ----------------

  const frameArtifacts = new Map<
    'F0102' | 'F0126',
    {
      feedFloat32: Float32Array;
      feedSha: string;
      fullGpuDisabled: { res: OrtVersionDiagnosticSuccess; val: Record<string, { data: Float32Array }>; provenance: RuntimeProvenanceRecord };
      fullWasmDisabled: { res: OrtVersionDiagnosticSuccess; val: Record<string, { data: Float32Array }>; provenance: RuntimeProvenanceRecord };
      bypassFeeds: Array<{ name: string; dims: number[]; dataType: 'float32' | 'int32' | 'int64' | 'uint8'; rawBytes: Uint8Array; sha256: string }>;
      wasm7F32: Float32Array;
      wasm7Sha: string;
      wasm8F32: Float32Array;
      wasm8Sha: string;
      gpu9Wasm7F32: Float32Array;
      wasm9Wasm7F32: Float32Array;
      causalGpuSuffixWasm7Val: Record<string, { data: Float32Array }>;
      causalGpuSuffixWasm7DispatchObserved: boolean;
      stage1: Stage1FrameConvVerification;
    }
  >();

  const runStage1Frame = async (frameLabel: 'F0102' | 'F0126', feedFloat32: Float32Array, feedSha: string) => {
    log(`\n--- [Stage 1 / ${frameLabel}] Untapped Full M1 + #7 Short Control & #9 Reduction Verification ---`);
    const fullGpuDisabledRun = await runSingleModel({
      label: `${frameLabel} Untapped Full M1 WebGPU (opt=disabled)`,
      modelBytes: params.m1Bytes,
      modelSha: m1Sha,
      provider: 'webgpu',
      opt: 'disabled',
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      feedF32: feedFloat32,
      feedSha,
      runSeparateGpuProfilePass: true,
    });
    const fullWasmDisabledRun = await runSingleModel({
      label: `${frameLabel} Untapped Full M1 WASM (opt=disabled)`,
      modelBytes: params.m1Bytes,
      modelSha: m1Sha,
      provider: 'wasm',
      opt: 'disabled',
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      feedF32: feedFloat32,
      feedSha,
    });
    const fullGpuVal = validateHitnetFinalOutputs(fullGpuDisabledRun.res.outputs);
    const fullWasmVal = validateHitnetFinalOutputs(fullWasmDisabledRun.res.outputs);
    if (!fullGpuVal.ok || !fullWasmVal.ok) {
      throw new Error(`${frameLabel} full-M1 outputs invalid: gpu=${fullGpuVal.error}, wasm=${fullWasmVal.error}`);
    }

    // Tap [conv7DynInput (Concat), conv7Output] on full M1
    const tap67 = appendOnnxGraphOutputs(params.m1Bytes, [conv7DynInput, chainSpec.conv7OutputTensor], {
      maxExtraOutputs: 2,
    });
    const tap67Sha = await sha256HexBytes(tap67.bytes);
    const tap67Gpu = await runSingleModel({
      label: `${frameLabel} Tap(#6,#7) WebGPU`,
      modelBytes: tap67.bytes,
      modelSha: tap67Sha,
      provider: 'webgpu',
      opt: 'disabled',
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      feedF32: feedFloat32,
      feedSha,
    });
    const tap67Wasm = await runSingleModel({
      label: `${frameLabel} Tap(#6,#7) WASM`,
      modelBytes: tap67.bytes,
      modelSha: tap67Sha,
      provider: 'wasm',
      opt: 'disabled',
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      feedF32: feedFloat32,
      feedSha,
    });
    const tap67GpuVal = validateHitnetFinalOutputs(tap67Gpu.res.outputs);
    const tap67WasmVal = validateHitnetFinalOutputs(tap67Wasm.res.outputs);
    if (
      !tap67GpuVal.ok ||
      !tap67WasmVal.ok ||
      !checkFloat32BitIdentity(tap67GpuVal.byName[pName].data, fullGpuVal.byName[pName].data).bitIdentical ||
      !checkFloat32BitIdentity(tap67WasmVal.byName[pName].data, fullWasmVal.byName[pName].data).bitIdentical
    ) {
      throw new Error(`${frameLabel} Full-M1 Tap(#6,#7) perturbed final disparity outputs`);
    }

    const gConcat = tap67Gpu.res.outputs.find((o) => o.name === conv7DynInput);
    const wConcat = tap67Wasm.res.outputs.find((o) => o.name === conv7DynInput);
    const gConv7 = tap67Gpu.res.outputs.find((o) => o.name === chainSpec.conv7OutputTensor);
    const wConv7 = tap67Wasm.res.outputs.find((o) => o.name === chainSpec.conv7OutputTensor);
    if (!gConcat || !wConcat || !gConv7 || !wConv7) {
      throw new Error(`${frameLabel} missing captured #6 concat or #7 conv output`);
    }
    const gConcatSha = await sha256HexBytes(copyOutputRawBytes(gConcat));
    const wConcatSha = await sha256HexBytes(copyOutputRawBytes(wConcat));
    const conv7ParentConcatBitwiseEqual = gConcatSha === wConcatSha;
    const wConcatF32 = new Float32Array(wConcat.buffer).slice();
    const gConv7F32 = new Float32Array(gConv7.buffer).slice();
    const wConv7F32 = new Float32Array(wConv7.buffer).slice();
    const wConv7Sha = await sha256HexBytes(copyOutputRawBytes(wConv7));

    const conv7ShortControlEval = await evaluateConvReductionCandidatesOnFixedIndices({
      parsed: parsedModel,
      convNodeIdx: chainSpec.conv7Idx,
      inputF32: wConcatF32,
      inputDims: conv7DynInputMeta.dims,
      gpuOutF32: gConv7F32,
      wasmOutF32: wConv7F32,
      outDims: suffix7.primaryCutInput.dims,
    });

    // Capture suffix7 bypass Tile_2 (#2, #3)
    const tile2Names = suffix7.bypassCutInputs.map((b) => b.name);
    const tapBypass = appendOnnxGraphOutputs(params.m1Bytes, tile2Names, { maxExtraOutputs: 2 });
    const tapBypassSha = await sha256HexBytes(tapBypass.bytes);
    const tapBypassGpu = await runSingleModel({
      label: `${frameLabel} Tap Bypass(#2,#3) WebGPU`,
      modelBytes: tapBypass.bytes,
      modelSha: tapBypassSha,
      provider: 'webgpu',
      opt: 'disabled',
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      feedF32: feedFloat32,
      feedSha,
    });
    const tapBypassWasm = await runSingleModel({
      label: `${frameLabel} Tap Bypass(#2,#3) WASM`,
      modelBytes: tapBypass.bytes,
      modelSha: tapBypassSha,
      provider: 'wasm',
      opt: 'disabled',
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      feedF32: feedFloat32,
      feedSha,
    });

    const bypassFeeds: Array<{ name: string; dims: number[]; dataType: 'float32' | 'int32' | 'int64' | 'uint8'; rawBytes: Uint8Array; sha256: string }> = [];
    for (const spec of suffix7.bypassCutInputs) {
      const go = tapBypassGpu.res.outputs.find((o) => o.name === spec.name);
      const wo = tapBypassWasm.res.outputs.find((o) => o.name === spec.name);
      if (!go || !wo) throw new Error(`${frameLabel} missing bypass "${spec.name}"`);
      const gBytes = copyOutputRawBytes(go);
      const wBytes = copyOutputRawBytes(wo);
      const gSha = await sha256HexBytes(gBytes);
      const wSha = await sha256HexBytes(wBytes);
      if (gSha !== wSha || gBytes.byteLength !== wBytes.byteLength) {
        throw new Error(`${frameLabel} bypass "${spec.name}" differs between GPU and WASM`);
      }
      bypassFeeds.push({
        name: spec.name,
        dims: [...spec.dims],
        dataType: spec.dataTypeName,
        rawBytes: wBytes,
        sha256: wSha,
      });
    }

    const extraFeeds7 = bypassFeeds.map((b) => ({
      name: b.name,
      dims: b.dims,
      buffer: b.rawBytes.slice().buffer as ArrayBuffer,
      dataType: b.dataType,
    }));

    // Untapped GPU suffix7(wasm#7) for causal control comparison
    const causalGpuSuf7 = await runSingleModel({
      label: `${frameLabel} Causal Baseline GPU-Suffix(#8..end)(wasm#7)`,
      modelBytes: suffix7.suffixBytes,
      modelSha: suffix7.suffixSha256,
      provider: 'webgpu',
      opt: 'disabled',
      feedName: suffix7.primaryCutInput.name,
      feedDims: [...suffix7.primaryCutInput.dims],
      feedF32: wConv7F32,
      feedSha: wConv7Sha,
      extraFeeds: extraFeeds7,
      runSeparateGpuProfilePass: true,
    });
    const causalGpuSuf7Val = validateHitnetFinalOutputs(causalGpuSuf7.res.outputs);
    if (!causalGpuSuf7Val.ok) {
      throw new Error(`${frameLabel} Causal GPU-Suffix(wasm#7) invalid outputs: ${causalGpuSuf7Val.error}`);
    }

    // Paired tap [#8 LeakyRelu, #9 Conv] on suffix7(wasm#7)
    const tap89 = appendOnnxGraphOutputs(
      suffix7.suffixBytes,
      [chainSpec.node8LeakyRelu.outputTensor, chainSpec.node9ConvDown0.outputTensor],
      { maxExtraOutputs: 2 }
    );
    const tap89Sha = await sha256HexBytes(tap89.bytes);
    const tap89Gpu = await runSingleModel({
      label: `${frameLabel} Suffix7 Tap(#8,#9)(wasm#7) WebGPU`,
      modelBytes: tap89.bytes,
      modelSha: tap89Sha,
      provider: 'webgpu',
      opt: 'disabled',
      feedName: suffix7.primaryCutInput.name,
      feedDims: [...suffix7.primaryCutInput.dims],
      feedF32: wConv7F32,
      feedSha: wConv7Sha,
      extraFeeds: extraFeeds7,
    });
    const tap89Wasm = await runSingleModel({
      label: `${frameLabel} Suffix7 Tap(#8,#9)(wasm#7) WASM`,
      modelBytes: tap89.bytes,
      modelSha: tap89Sha,
      provider: 'wasm',
      opt: 'disabled',
      feedName: suffix7.primaryCutInput.name,
      feedDims: [...suffix7.primaryCutInput.dims],
      feedF32: wConv7F32,
      feedSha: wConv7Sha,
      extraFeeds: extraFeeds7,
    });

    const g8 = tap89Gpu.res.outputs.find((o) => o.name === chainSpec.node8LeakyRelu.outputTensor);
    const w8 = tap89Wasm.res.outputs.find((o) => o.name === chainSpec.node8LeakyRelu.outputTensor);
    const g9 = tap89Gpu.res.outputs.find((o) => o.name === chainSpec.node9ConvDown0.outputTensor);
    const w9 = tap89Wasm.res.outputs.find((o) => o.name === chainSpec.node9ConvDown0.outputTensor);
    if (!g8 || !w8 || !g9 || !w9) throw new Error(`${frameLabel} missing paired #8/#9 tap outputs`);

    const g8Sha = await sha256HexBytes(copyOutputRawBytes(g8));
    const w8Sha = await sha256HexBytes(copyOutputRawBytes(w8));
    const node8LeakyReluBitExact = g8Sha === w8Sha;
    const w8F32 = new Float32Array(w8.buffer).slice();
    const g9F32 = new Float32Array(g9.buffer).slice();
    const w9F32 = new Float32Array(w9.buffer).slice();

    // Isolated 1-op #9 replay fed identical w8F32 (with separate profiling-ON GPU dispatch check)
    const oneOpGpu = await runSingleModel({
      label: `${frameLabel} 1-Op #9 WebGPU Replay`,
      modelBytes: singleOp9.bytes,
      modelSha: singleOp9Sha,
      provider: 'webgpu',
      opt: 'disabled',
      feedName: singleOp9.dynamicInputSpecs[0].name,
      feedDims: [...singleOp9.dynamicInputSpecs[0].dims],
      feedF32: w8F32,
      feedSha: w8Sha,
      allowArbitraryOutputs: true,
      runSeparateGpuProfilePass: true,
    });
    const oneOpWasm = await runSingleModel({
      label: `${frameLabel} 1-Op #9 WASM Replay`,
      modelBytes: singleOp9.bytes,
      modelSha: singleOp9Sha,
      provider: 'wasm',
      opt: 'disabled',
      feedName: singleOp9.dynamicInputSpecs[0].name,
      feedDims: [...singleOp9.dynamicInputSpecs[0].dims],
      feedF32: w8F32,
      feedSha: w8Sha,
      allowArbitraryOutputs: true,
    });
    const oneOpGpuF32 = new Float32Array(oneOpGpu.res.outputs[0].buffer).slice();
    const oneOpWasmF32 = new Float32Array(oneOpWasm.res.outputs[0].buffer).slice();
    const conv9FullSuffixVsOneOpBitExactGpu =
      checkFloat32BitIdentity(oneOpGpuF32, g9F32).bitIdentical;
    const conv9FullSuffixVsOneOpBitExactWasm =
      checkFloat32BitIdentity(oneOpWasmF32, w9F32).bitIdentical;

    const conv9Eval = await evaluateConvReductionCandidatesOnFixedIndices({
      parsed: parsedModel,
      convNodeIdx: chainSpec.node9ConvDown0.nodeIdx,
      inputF32: w8F32,
      inputDims: singleOp9.dynamicInputSpecs[0].dims,
      gpuOutF32: g9F32,
      wasmOutF32: w9F32,
      outDims: chainSpec.node9ConvDown0.outputDims,
    });

    log(
      `  [#7 Short Control K=${conv7ShortControlEval.dims.K}] parentConcatBitEq=${conv7ParentConcatBitwiseEqual}\n` +
        conv7ShortControlEval.variants
          .map(
            (v) =>
              `    ${v.variantId}: wasmBit=${v.vsWasm.exactBitMatches}/${v.evaluatedCount} maxUlp=${v.vsWasm.maxUlpDiff} mean=${v.vsWasm.meanAbsDiff.toExponential(2)} p99=${v.vsWasm.p99AbsDiff.toExponential(2)} max=${v.vsWasm.maxAbsDiff.toExponential(2)}`
          )
          .join('\n') +
        `\n  [#8 Exact=${node8LeakyReluBitExact} | #9 K=${conv9Eval.dims.K}] 1OpBitExact(gpu=${conv9FullSuffixVsOneOpBitExactGpu}, wasm=${conv9FullSuffixVsOneOpBitExactWasm}) | GPU-vs-WASM baseline: mean=${conv9Eval.baselineGpuVsWasmOnFixedIndices.meanAbsDiff.toExponential(3)} p99=${conv9Eval.baselineGpuVsWasmOnFixedIndices.p99AbsDiff.toExponential(3)} max=${conv9Eval.baselineGpuVsWasmOnFixedIndices.maxAbsDiff.toExponential(3)}\n` +
        conv9Eval.variants
          .map(
            (v) =>
              `    ${v.variantId}: wasmBit=${v.vsWasm.exactBitMatches}/${v.evaluatedCount} ulp1=${v.vsWasm.ulp1Count} ulp2+=${v.vsWasm.ulp2PlusCount} maxUlp=${v.vsWasm.maxUlpDiff} mean=${v.vsWasm.meanAbsDiff.toExponential(2)} p99=${v.vsWasm.p99AbsDiff.toExponential(2)} max=${v.vsWasm.maxAbsDiff.toExponential(2)}`
          )
          .join('\n')
    );

    const stage1Rec: Stage1FrameConvVerification = {
      frameLabel,
      feedSha256: feedSha,
      provenances: {
        fullM1GpuOptDisabled: fullGpuDisabledRun.provenance,
        fullM1WasmOptDisabled: fullWasmDisabledRun.provenance,
        oneOpConv9Gpu: oneOpGpu.provenance,
        oneOpConv9Wasm: oneOpWasm.provenance,
      },
      conv7ParentConcatBitwiseEqual,
      conv7ParentConcatSha256: wConcatSha,
      conv7ShortControlEval,
      node8LeakyReluBitExact,
      node8Sha256: w8Sha,
      conv9FullSuffixVsOneOpBitExactGpu,
      conv9FullSuffixVsOneOpBitExactWasm,
      conv9Eval,
    };

    frameArtifacts.set(frameLabel, {
      feedFloat32,
      feedSha,
      fullGpuDisabled: { res: fullGpuDisabledRun.res, val: fullGpuVal.byName, provenance: fullGpuDisabledRun.provenance },
      fullWasmDisabled: { res: fullWasmDisabledRun.res, val: fullWasmVal.byName, provenance: fullWasmDisabledRun.provenance },
      bypassFeeds,
      wasm7F32: wConv7F32,
      wasm7Sha: wConv7Sha,
      wasm8F32: w8F32,
      wasm8Sha: w8Sha,
      gpu9Wasm7F32: g9F32,
      wasm9Wasm7F32: w9F32,
      causalGpuSuffixWasm7Val: causalGpuSuf7Val.byName,
      causalGpuSuffixWasm7DispatchObserved: causalGpuSuf7.provenance.separateProfilingPass.dispatchObserved,
      stage1: stage1Rec,
    });
    return stage1Rec;
  };

  const s1F0102 = await runStage1Frame('F0102', params.f0102FeedFloat32, f0102Sha);
  const s1F0126 = await runStage1Frame('F0126', params.f0126FeedFloat32, f0126Sha);

  const selection = selectAndValidatePredeclaredCandidate(s1F0102.conv9Eval, s1F0126.conv9Eval);
  log(
    `\n=== [Predeclared Candidate Selection (F0102 -> Held-Out F0126)] ===\n` +
      `Selected=${selection.selectedCandidateId ?? 'NONE'} (blockK=${selection.selectedBlockK ?? 'n/a'}) | f0102Promising=${selection.f0102Promising} | f0126Validated=${selection.f0126HeldOutValidated} | advance=${selection.advancedToInjection}` +
      (selection.stopReason ? `\n[STOP REASON] ${selection.stopReason}` : '')
  );

  const baseArtifactsManifest = {
    prefix0to8ByteLength: prefix0to8.prefixByteLength,
    prefix0to8Sha256: prefix0to8.prefixSha256,
    suffix9ByteLength: suffix9.suffixByteLength,
    suffix9Sha256: suffix9.suffixSha256,
    suffix7ByteLength: suffix7.suffixByteLength,
    suffix7Sha256: suffix7.suffixSha256,
    singleOp9ByteLength: singleOp9.bytes.byteLength,
    singleOp9Sha256: singleOp9Sha,
  };

  if (!selection.advancedToInjection || !selection.selectedCandidateId) {
    return {
      ok: true,
      aborted: false,
      abortReason: null,
      pinnedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      interventionGraphOptimizationLevel: 'disabled',
      appBaselineGraphOptimizationLevel: 'basic',
      numThreads: 1,
      proxy: false,
      m1ByteLength: params.m1Bytes.byteLength,
      m1Sha256: m1Sha,
      n1OffProof,
      chainSpec,
      verifiedArtifacts: baseArtifactsManifest,
      stage1: { f0102: s1F0102, f0126: s1F0126, selection },
      stage2RealFrames: {
        executed: false,
        stopReason: selection.stopReason,
        f0102: null,
        f0126: null,
      },
      stage3SyntheticD16D32: {
        executed: false,
        stopReason: selection.stopReason,
        coordinateUnitsNote:
          'D16/D32 skipped because Stage 1 predeclared candidate selection/held-out validation did not advance.',
        D16: null,
        D32: null,
      },
      scientificScopeNote:
        'Stopped cleanly after Stage 1 because no predeclared #9 candidate met promising F0102 + held-out F0126 criteria.',
    };
  }

  const winningCandidateId = selection.selectedCandidateId;

  // Helper to run Practical #0..#8 -> host cand#9 -> #10..#685 pipeline with honest wall timing & separate profiling
  const buildExtraFeedsFromPrefixOutputs = (prefixOutputs: OrtVersionDiagnosticOutput[]) => {
    const extras: Array<{ name: string; dims: number[]; buffer: ArrayBuffer; dataType: 'float32' | 'int32' | 'int64' | 'uint8' }> = [];
    for (const bypassSpec of suffix9.bypassCutInputs) {
      const match = prefixOutputs.find((o) => o.name === bypassSpec.name);
      if (!match) {
        throw new Error(`Prefix output missing required suffix9 bypass input "${bypassSpec.name}"`);
      }
      extras.push({
        name: bypassSpec.name,
        dims: [...bypassSpec.dims],
        buffer: match.buffer.slice(0),
        dataType: bypassSpec.dataTypeName,
      });
    }
    return extras;
  };

  // ---------------- STAGE 2: Practical Injection & Split Controls on F0102 / F0126 ----------------

  const runStage2RealFrame = async (frameLabel: 'F0102' | 'F0126'): Promise<Stage2RealFrameInjectionReport> => {
    const art = frameArtifacts.get(frameLabel)!;
    log(`\n--- [Stage 2 / ${frameLabel}] Practical #9 Replacement (${winningCandidateId}) + Controls ---`);

    // 1. Unmodified Full-GPU App Baseline (opt=basic) + separate profiling pass
    const fullGpuAppBasicRun = await runSingleModel({
      label: `${frameLabel} Unmodified Full-GPU App Baseline (opt=basic)`,
      modelBytes: params.m1Bytes,
      modelSha: m1Sha,
      provider: 'webgpu',
      opt: 'basic',
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      feedF32: art.feedFloat32,
      feedSha: art.feedSha,
      warmRuns: warmRunsAppBaseline,
      runSeparateGpuProfilePass: true,
    });
    const fullGpuAppBasicVal = validateHitnetFinalOutputs(fullGpuAppBasicRun.res.outputs);
    if (!fullGpuAppBasicVal.ok) {
      throw new Error(`${frameLabel} Unmodified Full-GPU App Baseline (opt=basic) invalid: ${fullGpuAppBasicVal.error}`);
    }

    // 2. Per-provider Split Controls (prefix0to8 -> 1op#9 -> suffix10toEnd) MUST match untapped runs bit-for-bit
    const wasmPrefixRun = await runSingleModel({
      label: `${frameLabel} Split Control Prefix (#0..#8) WASM`,
      modelBytes: prefix0to8.prefixBytes,
      modelSha: prefix0to8.prefixSha256,
      provider: 'wasm',
      opt: 'disabled',
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      feedF32: art.feedFloat32,
      feedSha: art.feedSha,
      allowArbitraryOutputs: true,
    });
    const wPrefixNode8Out = wasmPrefixRun.res.outputs.find((o) => o.name === node8OutName);
    if (!wPrefixNode8Out) throw new Error(`${frameLabel} WASM prefix missing #8 output`);
    const wPrefixNode8F32 = new Float32Array(wPrefixNode8Out.buffer).slice();
    const wPrefixNode8Sha = await sha256HexBytes(copyOutputRawBytes(wPrefixNode8Out));

    const wasmOneOp9Split = await runSingleModel({
      label: `${frameLabel} Split Control 1-Op #9 WASM`,
      modelBytes: singleOp9.bytes,
      modelSha: singleOp9Sha,
      provider: 'wasm',
      opt: 'disabled',
      feedName: singleOp9.dynamicInputSpecs[0].name,
      feedDims: [...singleOp9.dynamicInputSpecs[0].dims],
      feedF32: wPrefixNode8F32,
      feedSha: wPrefixNode8Sha,
      allowArbitraryOutputs: true,
    });
    const wOneOp9F32 = new Float32Array(wasmOneOp9Split.res.outputs[0].buffer).slice();
    const wOneOp9Sha = await sha256HexBytes(copyOutputRawBytes(wasmOneOp9Split.res.outputs[0]));

    const wasmSuffix9Split = await runSingleModel({
      label: `${frameLabel} Split Control Suffix (#10..end) WASM`,
      modelBytes: suffix9.suffixBytes,
      modelSha: suffix9.suffixSha256,
      provider: 'wasm',
      opt: 'disabled',
      feedName: suffix9.primaryCutInput.name,
      feedDims: [...suffix9.primaryCutInput.dims],
      feedF32: wOneOp9F32,
      feedSha: wOneOp9Sha,
      extraFeeds: buildExtraFeedsFromPrefixOutputs(wasmPrefixRun.res.outputs),
    });
    const wasmSplitVal = validateHitnetFinalOutputs(wasmSuffix9Split.res.outputs);
    if (!wasmSplitVal.ok) throw new Error(`${frameLabel} WASM Split Control invalid outputs: ${wasmSplitVal.error}`);

    // GPU Split Control (untimed here; practical pipeline run below records full uncached wall timing)
    const gpuPrefixControlRun = await runSingleModel({
      label: `${frameLabel} Split Control Prefix (#0..#8) WebGPU`,
      modelBytes: prefix0to8.prefixBytes,
      modelSha: prefix0to8.prefixSha256,
      provider: 'webgpu',
      opt: 'disabled',
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      feedF32: art.feedFloat32,
      feedSha: art.feedSha,
      allowArbitraryOutputs: true,
    });
    const gPrefixNode8Out = gpuPrefixControlRun.res.outputs.find((o) => o.name === node8OutName);
    if (!gPrefixNode8Out) throw new Error(`${frameLabel} GPU prefix missing #8 output`);
    const gPrefixNode8F32 = new Float32Array(gPrefixNode8Out.buffer).slice();
    const gPrefixNode8Sha = await sha256HexBytes(copyOutputRawBytes(gPrefixNode8Out));

    const gpuOneOp9Split = await runSingleModel({
      label: `${frameLabel} Split Control 1-Op #9 WebGPU`,
      modelBytes: singleOp9.bytes,
      modelSha: singleOp9Sha,
      provider: 'webgpu',
      opt: 'disabled',
      feedName: singleOp9.dynamicInputSpecs[0].name,
      feedDims: [...singleOp9.dynamicInputSpecs[0].dims],
      feedF32: gPrefixNode8F32,
      feedSha: gPrefixNode8Sha,
      allowArbitraryOutputs: true,
    });
    const gOneOp9F32 = new Float32Array(gpuOneOp9Split.res.outputs[0].buffer).slice();
    const gOneOp9Sha = await sha256HexBytes(copyOutputRawBytes(gpuOneOp9Split.res.outputs[0]));

    const gpuSuffix9Split = await runSingleModel({
      label: `${frameLabel} Split Control Suffix (#10..end) WebGPU`,
      modelBytes: suffix9.suffixBytes,
      modelSha: suffix9.suffixSha256,
      provider: 'webgpu',
      opt: 'disabled',
      feedName: suffix9.primaryCutInput.name,
      feedDims: [...suffix9.primaryCutInput.dims],
      feedF32: gOneOp9F32,
      feedSha: gOneOp9Sha,
      extraFeeds: buildExtraFeedsFromPrefixOutputs(gpuPrefixControlRun.res.outputs),
    });
    const gpuSplitVal = validateHitnetFinalOutputs(gpuSuffix9Split.res.outputs);
    if (!gpuSplitVal.ok) throw new Error(`${frameLabel} GPU Split Control invalid outputs: ${gpuSplitVal.error}`);

    // Verify bypass equality & bit-exact split control parity
    let wasmPrefixBypassEqualToFull = true;
    let gpuPrefixBypassEqualToWasm = true;
    for (const bf of art.bypassFeeds) {
      const wp = wasmPrefixRun.res.outputs.find((o) => o.name === bf.name);
      const gp = gpuPrefixControlRun.res.outputs.find((o) => o.name === bf.name);
      if (!wp || !gp) {
        wasmPrefixBypassEqualToFull = false;
        gpuPrefixBypassEqualToWasm = false;
        continue;
      }
      const wpSha = await sha256HexBytes(copyOutputRawBytes(wp));
      const gpSha = await sha256HexBytes(copyOutputRawBytes(gp));
      if (wpSha !== bf.sha256) wasmPrefixBypassEqualToFull = false;
      if (gpSha !== wpSha) gpuPrefixBypassEqualToWasm = false;
    }

    const wasmSplitP = checkFloat32BitIdentity(
      wasmSplitVal.byName[pName].data,
      art.fullWasmDisabled.val[pName].data
    );
    const wasmSplitS = checkFloat32BitIdentity(
      wasmSplitVal.byName[sName].data,
      art.fullWasmDisabled.val[sName].data
    );
    const gpuSplitP = checkFloat32BitIdentity(
      gpuSplitVal.byName[pName].data,
      art.fullGpuDisabled.val[pName].data
    );
    const gpuSplitS = checkFloat32BitIdentity(
      gpuSplitVal.byName[sName].data,
      art.fullGpuDisabled.val[sName].data
    );
    const splitAllPassed =
      wasmPrefixBypassEqualToFull &&
      gpuPrefixBypassEqualToWasm &&
      wasmSplitP.bitIdentical &&
      wasmSplitS.bitIdentical &&
      gpuSplitP.bitIdentical &&
      gpuSplitS.bitIdentical;

    const splitControls: RealFrameSplitControlCheck = {
      wasmPrefixBypassEqualToFull,
      gpuPrefixBypassEqualToWasm,
      wasmSplitVsUntappedWasmFullPrimary: wasmSplitP,
      wasmSplitVsUntappedWasmFullSecondary: wasmSplitS,
      gpuSplitVsUntappedGpuFullPrimary: gpuSplitP,
      gpuSplitVsUntappedGpuFullSecondary: gpuSplitS,
      allPassed: splitAllPassed,
      failureReason: splitAllPassed
        ? null
        : `Per-provider split control failed bit-exactness on ${frameLabel}: wasmBypass=${wasmPrefixBypassEqualToFull}, gpuBypass=${gpuPrefixBypassEqualToWasm}, wasm(P=${wasmSplitP.mismatchCount}, S=${wasmSplitS.mismatchCount}), gpu(P=${gpuSplitP.mismatchCount}, S=${gpuSplitS.mismatchCount})`,
    };

    log(
      `  [Per-Provider Split Controls] allPassed=${splitAllPassed} (wasmP=${wasmSplitP.mismatchCount}, wasmS=${wasmSplitS.mismatchCount}, gpuP=${gpuSplitP.mismatchCount}, gpuS=${gpuSplitS.mismatchCount})`
    );

    if (!splitAllPassed) {
      throw new Error(splitControls.failureReason!);
    }

    // 3. PRACTICAL REPLACED-#9 END-TO-END EXECUTION (NO cached arrays! Fresh prefix -> host cand#9 -> suffix!)
    const tPipelineStart = performance.now();
    const tPrefix0 = performance.now();
    const practPrefixRun = await runSingleModel({
      label: `${frameLabel} Practical Prefix (#0..#8) WebGPU (opt=disabled)`,
      modelBytes: prefix0to8.prefixBytes,
      modelSha: prefix0to8.prefixSha256,
      provider: 'webgpu',
      opt: 'disabled',
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      feedF32: art.feedFloat32,
      feedSha: art.feedSha,
      allowArbitraryOutputs: true,
      runSeparateGpuProfilePass: true,
    });
    const prefixRoundTripWallMs = performance.now() - tPrefix0;
    const practNode8Out = practPrefixRun.res.outputs.find((o) => o.name === node8OutName);
    if (!practNode8Out) throw new Error(`${frameLabel} practical prefix missing #8`);
    const practNode8F32 = new Float32Array(practNode8Out.buffer).slice();

    const tHost0 = performance.now();
    const cand9OnGpuPrefixF32 = computeConv2dCandidateFullTensor({
      parsed: parsedModel,
      convNodeIdx: chainSpec.node9ConvDown0.nodeIdx,
      xData: practNode8F32,
      inputDims: singleOp9.dynamicInputSpecs[0].dims as [number, number, number, number],
      outDims: chainSpec.node9ConvDown0.outputDims as [number, number, number, number],
      candidateId: winningCandidateId,
    });
    const hostCandidate9ComputeMs = performance.now() - tHost0;
    const cand9OnGpuPrefixSha = await sha256HexBytes(
      new Uint8Array(cand9OnGpuPrefixF32.buffer, cand9OnGpuPrefixF32.byteOffset, cand9OnGpuPrefixF32.byteLength)
    );

    const tSuffix0 = performance.now();
    const practSuffixRun = await runSingleModel({
      label: `${frameLabel} Practical Suffix (#10..end) WebGPU (opt=disabled)`,
      modelBytes: suffix9.suffixBytes,
      modelSha: suffix9.suffixSha256,
      provider: 'webgpu',
      opt: 'disabled',
      feedName: suffix9.primaryCutInput.name,
      feedDims: [...suffix9.primaryCutInput.dims],
      feedF32: cand9OnGpuPrefixF32,
      feedSha: cand9OnGpuPrefixSha,
      extraFeeds: buildExtraFeedsFromPrefixOutputs(practPrefixRun.res.outputs),
      runSeparateGpuProfilePass: true,
    });
    const suffixRoundTripWallMs = performance.now() - tSuffix0;
    const totalWallWithCreateMs = performance.now() - tPipelineStart;
    const practVal = validateHitnetFinalOutputs(practSuffixRun.res.outputs);
    if (!practVal.ok) {
      throw new Error(`${frameLabel} Practical replaced-#9 outputs invalid: ${practVal.error}`);
    }

    const cand9DiffVsUnmodGpu9 = summarizeRawTensorAndParity(cand9OnGpuPrefixF32, gOneOp9F32);
    const endToEndExecuteWallMs =
      practPrefixRun.res.coldRunMs + hostCandidate9ComputeMs + practSuffixRun.res.coldRunMs;
    const orchestrationOverheadMs = Math.max(
      0,
      totalWallWithCreateMs -
        (practPrefixRun.res.createMs +
          practPrefixRun.res.coldRunMs +
          hostCandidate9ComputeMs +
          practSuffixRun.res.createMs +
          practSuffixRun.res.coldRunMs)
    );

    // 4. CAUSAL COMMON-WASM-#7 CONTROL (strictly labeled separately from unmodified full-GPU app baseline!)
    const cand9OnWasm8F32 = computeConv2dCandidateFullTensor({
      parsed: parsedModel,
      convNodeIdx: chainSpec.node9ConvDown0.nodeIdx,
      xData: art.wasm8F32,
      inputDims: singleOp9.dynamicInputSpecs[0].dims as [number, number, number, number],
      outDims: chainSpec.node9ConvDown0.outputDims as [number, number, number, number],
      candidateId: winningCandidateId,
    });
    const cand9OnWasm8Sha = await sha256HexBytes(
      new Uint8Array(cand9OnWasm8F32.buffer, cand9OnWasm8F32.byteOffset, cand9OnWasm8F32.byteLength)
    );
    const extraFeedsCausal = suffix9.bypassCutInputs.map((bp) => {
      if (bp.name === node8OutName) {
        return {
          name: node8OutName,
          dims: [...bp.dims],
          buffer: art.wasm8F32.slice().buffer as ArrayBuffer,
          dataType: 'float32' as const,
        };
      }
      const match = art.bypassFeeds.find((x) => x.name === bp.name);
      if (!match) throw new Error(`${frameLabel} causal control missing bypass "${bp.name}"`);
      return {
        name: bp.name,
        dims: [...bp.dims],
        buffer: match.rawBytes.slice().buffer as ArrayBuffer,
        dataType: bp.dataTypeName,
      };
    });
    const causalGpuCand9 = await runSingleModel({
      label: `${frameLabel} Causal Control GPU-Suffix(cand9(wasm#8), wasm#8, bypass)`,
      modelBytes: suffix9.suffixBytes,
      modelSha: suffix9.suffixSha256,
      provider: 'webgpu',
      opt: 'disabled',
      feedName: suffix9.primaryCutInput.name,
      feedDims: [...suffix9.primaryCutInput.dims],
      feedF32: cand9OnWasm8F32,
      feedSha: cand9OnWasm8Sha,
      extraFeeds: extraFeedsCausal,
      runSeparateGpuProfilePass: true,
    });
    const causalGpuCand9Val = validateHitnetFinalOutputs(causalGpuCand9.res.outputs);
    if (!causalGpuCand9Val.ok) {
      throw new Error(`${frameLabel} Causal Control GPU-Suffix(cand9(wasm#8)) invalid: ${causalGpuCand9Val.error}`);
    }

    const endToEndWallTiming: EndToEndPipelineWallTiming = {
      timingLabel: `${frameLabel} Uncached End-to-End Timing (Profiling OFF)`,
      note:
        'All practical pipeline numbers include prefix execution+readback, full host candidate #9 computation, tensor transfer/upload/synchronization, and suffix execution+readback; separate profiling-ON passes are excluded from wall time.',
      unmodifiedFullGpuAppBaselineOptBasic: {
        createMs: fullGpuAppBasicRun.res.createMs,
        coldRunMs: fullGpuAppBasicRun.res.coldRunMs,
        warmRunMs: [...fullGpuAppBasicRun.res.warmRunMs],
        dispatchConfirmedSeparatePass: fullGpuAppBasicRun.provenance.separateProfilingPass.dispatchObserved,
      },
      unmodifiedFullGpuOptDisabled: {
        createMs: art.fullGpuDisabled.res.createMs,
        coldRunMs: art.fullGpuDisabled.res.coldRunMs,
        dispatchConfirmedSeparatePass:
          art.fullGpuDisabled.provenance.separateProfilingPass.dispatchObserved,
      },
      unmodifiedFullWasmOptDisabled: {
        createMs: art.fullWasmDisabled.res.createMs,
        coldRunMs: art.fullWasmDisabled.res.coldRunMs,
      },
      practicalReplaced9PipelineOptDisabled: {
        prefixModelCreateMs: practPrefixRun.res.createMs,
        prefixRunAndReadbackWorkerMs: practPrefixRun.res.coldRunMs,
        prefixWorkerRoundTripWallMs: prefixRoundTripWallMs,
        hostCandidate9ComputeMs: hostCandidate9ComputeMs,
        suffixModelCreateMs: practSuffixRun.res.createMs,
        suffixUploadRunReadbackWorkerMs: practSuffixRun.res.coldRunMs,
        suffixWorkerRoundTripWallMs: suffixRoundTripWallMs,
        orchestrationOverheadMs,
        endToEndExecuteWallMs,
        endToEndColdCreatePlusRunWallMs: totalWallWithCreateMs,
        prefixDispatchConfirmedSeparatePass:
          practPrefixRun.provenance.separateProfilingPass.dispatchObserved,
        suffixDispatchConfirmedSeparatePass:
          practSuffixRun.provenance.separateProfilingPass.dispatchObserved,
        deltaVsUnmodifiedFullGpuAppBaselineColdRunMs:
          endToEndExecuteWallMs - fullGpuAppBasicRun.res.coldRunMs,
        deltaVsUnmodifiedFullGpuOptDisabledColdRunMs:
          endToEndExecuteWallMs - art.fullGpuDisabled.res.coldRunMs,
        deltaVsUnmodifiedFullWasmOptDisabledColdRunMs:
          endToEndExecuteWallMs - art.fullWasmDisabled.res.coldRunMs,
      },
    };

    const practicalVsWasm = buildDualFinalSummary(
      `Practical(Prefix0..8 GPU -> ${winningCandidateId} #9 -> Suffix10..end GPU) vs Unmodified Full WASM (opt=disabled)`,
      frameLabel,
      practVal.byName,
      art.fullWasmDisabled.val
    );
    const practicalVsGpuDisabled = buildDualFinalSummary(
      `Practical(Prefix0..8 GPU -> ${winningCandidateId} #9 -> Suffix10..end GPU) vs Unmodified Full GPU (opt=disabled)`,
      frameLabel,
      practVal.byName,
      art.fullGpuDisabled.val
    );
    const practicalVsGpuAppBasic = buildDualFinalSummary(
      `Practical(Prefix0..8 GPU -> ${winningCandidateId} #9 -> Suffix10..end GPU) vs Unmodified Full-GPU App Baseline (opt=basic)`,
      frameLabel,
      practVal.byName,
      fullGpuAppBasicVal.byName
    );
    const baseGpuDisabledVsWasm = buildDualFinalSummary(
      `Unmodified Full GPU (opt=disabled) vs Unmodified Full WASM (opt=disabled)`,
      frameLabel,
      art.fullGpuDisabled.val,
      art.fullWasmDisabled.val
    );
    const baseGpuAppBasicVsWasm = buildDualFinalSummary(
      `Unmodified Full-GPU App Baseline (opt=basic) vs Unmodified Full WASM (opt=disabled)`,
      frameLabel,
      fullGpuAppBasicVal.byName,
      art.fullWasmDisabled.val
    );

    const causalBaseVsWasm = buildDualFinalSummary(
      `[CAUSAL COMMON-WASM-#7] Baseline GPU Suffix(wasm#7) vs Unmodified Full WASM`,
      frameLabel,
      art.causalGpuSuffixWasm7Val,
      art.fullWasmDisabled.val
    );
    const causalCand9VsWasm = buildDualFinalSummary(
      `[CAUSAL COMMON-WASM-#7] Intervened GPU Suffix(${winningCandidateId}(wasm#8)) vs Unmodified Full WASM`,
      frameLabel,
      causalGpuCand9Val.byName,
      art.fullWasmDisabled.val
    );
    const causalCand9VsCausalBase = buildDualFinalSummary(
      `[CAUSAL COMMON-WASM-#7] Intervened GPU Suffix(${winningCandidateId}(wasm#8)) vs Causal Baseline GPU Suffix(wasm#7)`,
      frameLabel,
      causalGpuCand9Val.byName,
      art.causalGpuSuffixWasm7Val
    );

    log(
      `  [Practical Replaced-#9 vs Unmodified Full WASM] P(max=${practicalVsWasm.primary.stats.maxAbsDiff.toExponential(3)}, p99=${practicalVsWasm.primary.stats.p99AbsDiff.toExponential(3)}, mean=${practicalVsWasm.primary.stats.meanAbsDiff.toExponential(3)}, flips=${practicalVsWasm.depthAndValidityImpact.primarySourceMaskChanges.totalMaskFlips}) | S(max=${practicalVsWasm.secondary.stats.maxAbsDiff.toExponential(3)}, p99=${practicalVsWasm.secondary.stats.p99AbsDiff.toExponential(3)}, mean=${practicalVsWasm.secondary.stats.meanAbsDiff.toExponential(3)})\n` +
        `  [Unmodified GPU opt=basic vs WASM] P(max=${baseGpuAppBasicVsWasm.primary.stats.maxAbsDiff.toExponential(3)}, p99=${baseGpuAppBasicVsWasm.primary.stats.p99AbsDiff.toExponential(3)}, mean=${baseGpuAppBasicVsWasm.primary.stats.meanAbsDiff.toExponential(3)}) | S(max=${baseGpuAppBasicVsWasm.secondary.stats.maxAbsDiff.toExponential(3)}, mean=${baseGpuAppBasicVsWasm.secondary.stats.meanAbsDiff.toExponential(3)})\n` +
        `  [CAUSAL Common-WASM-#7 Control vs Full WASM] Baseline(wasm#7) P(max=${causalBaseVsWasm.primary.stats.maxAbsDiff.toExponential(3)}, mean=${causalBaseVsWasm.primary.stats.meanAbsDiff.toExponential(3)}) -> Cand9(${winningCandidateId}) P(max=${causalCand9VsWasm.primary.stats.maxAbsDiff.toExponential(3)}, p99=${causalCand9VsWasm.primary.stats.p99AbsDiff.toExponential(3)}, mean=${causalCand9VsWasm.primary.stats.meanAbsDiff.toExponential(3)})\n` +
        `  [End-to-End Execute Wall (OFF-profile)] Practical=${endToEndExecuteWallMs.toFixed(1)}ms (prefix=${practPrefixRun.res.coldRunMs.toFixed(1)}ms + host#9=${hostCandidate9ComputeMs.toFixed(1)}ms + suffix=${practSuffixRun.res.coldRunMs.toFixed(1)}ms) | Full-GPU opt=basic=${fullGpuAppBasicRun.res.coldRunMs.toFixed(1)}ms | Full-WASM=${art.fullWasmDisabled.res.coldRunMs.toFixed(1)}ms`
    );

    return {
      frameLabel,
      feedSha256: art.feedSha,
      ok: true,
      aborted: false,
      abortReason: null,
      splitControls,
      candidate9DiffAgainstUnmodifiedGpu9OnGpuPrefixInput: cand9DiffVsUnmodGpu9,
      practicalReplaced9VsBaselines: {
        vsUnmodifiedWasmFullOptDisabled: practicalVsWasm,
        vsUnmodifiedGpuFullOptDisabled: practicalVsGpuDisabled,
        vsUnmodifiedGpuAppBaselineOptBasic: practicalVsGpuAppBasic,
        baselineGpuOptDisabledVsWasmOptDisabled: baseGpuDisabledVsWasm,
        baselineGpuAppBasicVsWasmOptDisabled: baseGpuAppBasicVsWasm,
      },
      causalCommonWasm7Controls: {
        controlScopeNote:
          'Strictly isolates causal sensitivity to replacing #9 while holding #7/#8 fixed at common WASM #7/#8 outputs; NOT comparable as an unmodified full-GPU app baseline.',
        causalBaselineGpuSuffixWasm7VsWasmFull: causalBaseVsWasm,
        causalIntervenedGpuSuffixWasm7WithCand9VsWasmFull: causalCand9VsWasm,
        causalIntervenedGpuSuffixWasm7WithCand9VsCausalBaselineGpuSuffix: causalCand9VsCausalBase,
        dispatchConfirmedSeparatePass:
          causalGpuCand9.provenance.separateProfilingPass.dispatchObserved &&
          art.causalGpuSuffixWasm7DispatchObserved,
      },
      endToEndWallTiming,
      provenances: {
        fullM1GpuAppBaselineOptBasic: fullGpuAppBasicRun.provenance,
        practicalPrefix0to8Gpu: practPrefixRun.provenance,
        practicalSuffix10toEndGpu: practSuffixRun.provenance,
      },
    };
  };

  let s2F0102: Stage2RealFrameInjectionReport | null = null;
  let s2F0126: Stage2RealFrameInjectionReport | null = null;
  try {
    s2F0102 = await runStage2RealFrame('F0102');
    s2F0126 = await runStage2RealFrame('F0126');
  } catch (err) {
    const reason = err instanceof Error ? err.message : String(err);
    log(`  [ABORT STAGE 2] ${reason}`);
    return {
      ok: false,
      aborted: true,
      abortReason: reason,
      pinnedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      interventionGraphOptimizationLevel: 'disabled',
      appBaselineGraphOptimizationLevel: 'basic',
      numThreads: 1,
      proxy: false,
      m1ByteLength: params.m1Bytes.byteLength,
      m1Sha256: m1Sha,
      n1OffProof,
      chainSpec,
      verifiedArtifacts: baseArtifactsManifest,
      stage1: { f0102: s1F0102, f0126: s1F0126, selection },
      stage2RealFrames: {
        executed: true,
        stopReason: reason,
        f0102: s2F0102,
        f0126: s2F0126,
      },
      stage3SyntheticD16D32: {
        executed: false,
        stopReason: reason,
        coordinateUnitsNote: 'Aborted because Stage 2 per-provider split control or execution failed.',
        D16: null,
        D32: null,
      },
      scientificScopeNote: `Aborted explicitly: ${reason}`,
    };
  }

  // ---------------- STAGE 3: Synthetic D16 & D32 Fixed-ROI Ground-Truth Comparison ----------------

  const runSyntheticCase = async (dispTrue: 16 | 32): Promise<SyntheticCaseInjectionReport> => {
    const gt: SyntheticGroundTruthPair = await generateSyntheticStereoGroundTruthPair(dispTrue);
    log(`\n--- [Stage 3 / ${gt.caseId}] Synthetic Known-GT (${dispTrue}.0 px) Fixed-ROI Comparison ---`);

    const wasmFullRun = await runSingleModel({
      label: `${gt.caseId} Unmodified WASM (opt=disabled)`,
      modelBytes: params.m1Bytes,
      modelSha: m1Sha,
      provider: 'wasm',
      opt: 'disabled',
      feedName: 'input',
      feedDims: [...gt.feedDims],
      feedF32: gt.feedFloat32,
      feedSha: gt.feedSha256,
    });
    const gpuDisabledRun = await runSingleModel({
      label: `${gt.caseId} Unmodified GPU (opt=disabled)`,
      modelBytes: params.m1Bytes,
      modelSha: m1Sha,
      provider: 'webgpu',
      opt: 'disabled',
      feedName: 'input',
      feedDims: [...gt.feedDims],
      feedF32: gt.feedFloat32,
      feedSha: gt.feedSha256,
      runSeparateGpuProfilePass: true,
    });
    const gpuAppBasicRun = await runSingleModel({
      label: `${gt.caseId} Unmodified GPU App Baseline (opt=basic)`,
      modelBytes: params.m1Bytes,
      modelSha: m1Sha,
      provider: 'webgpu',
      opt: 'basic',
      feedName: 'input',
      feedDims: [...gt.feedDims],
      feedF32: gt.feedFloat32,
      feedSha: gt.feedSha256,
      runSeparateGpuProfilePass: true,
    });

    const pPrefixRun = await runSingleModel({
      label: `${gt.caseId} Practical Prefix (#0..#8) WebGPU (opt=disabled)`,
      modelBytes: prefix0to8.prefixBytes,
      modelSha: prefix0to8.prefixSha256,
      provider: 'webgpu',
      opt: 'disabled',
      feedName: 'input',
      feedDims: [...gt.feedDims],
      feedF32: gt.feedFloat32,
      feedSha: gt.feedSha256,
      allowArbitraryOutputs: true,
      runSeparateGpuProfilePass: true,
    });
    const pNode8 = pPrefixRun.res.outputs.find((o) => o.name === node8OutName);
    if (!pNode8) throw new Error(`${gt.caseId} practical prefix missing #8`);
    const pNode8F32 = new Float32Array(pNode8.buffer).slice();

    const tH0 = performance.now();
    const pCand9F32 = computeConv2dCandidateFullTensor({
      parsed: parsedModel,
      convNodeIdx: chainSpec.node9ConvDown0.nodeIdx,
      xData: pNode8F32,
      inputDims: singleOp9.dynamicInputSpecs[0].dims as [number, number, number, number],
      outDims: chainSpec.node9ConvDown0.outputDims as [number, number, number, number],
      candidateId: winningCandidateId,
    });
    const hostMs = performance.now() - tH0;
    const pCand9Sha = await sha256HexBytes(
      new Uint8Array(pCand9F32.buffer, pCand9F32.byteOffset, pCand9F32.byteLength)
    );

    const pSuffixRun = await runSingleModel({
      label: `${gt.caseId} Practical Suffix (#10..end) WebGPU (opt=disabled)`,
      modelBytes: suffix9.suffixBytes,
      modelSha: suffix9.suffixSha256,
      provider: 'webgpu',
      opt: 'disabled',
      feedName: suffix9.primaryCutInput.name,
      feedDims: [...suffix9.primaryCutInput.dims],
      feedF32: pCand9F32,
      feedSha: pCand9Sha,
      extraFeeds: buildExtraFeedsFromPrefixOutputs(pPrefixRun.res.outputs),
      runSeparateGpuProfilePass: true,
    });

    const wVal = validateHitnetFinalOutputs(wasmFullRun.res.outputs);
    const gdVal = validateHitnetFinalOutputs(gpuDisabledRun.res.outputs);
    const gbVal = validateHitnetFinalOutputs(gpuAppBasicRun.res.outputs);
    const prVal = validateHitnetFinalOutputs(pSuffixRun.res.outputs);
    if (!wVal.ok || !gdVal.ok || !gbVal.ok || !prVal.ok) {
      throw new Error(`${gt.caseId} invalid disparity outputs across arms`);
    }

    const scoreArm = (byName: Record<string, { data: Float32Array }>) => ({
      primary: {
        commonRoi: evaluateDisparityRoiMetricsPx(
          byName[pName].data,
          'primary-common-roi',
          gt.commonCroppedRoi.primaryLeft,
          dispTrue
        ),
        analyticValid: evaluateDisparityRoiMetricsPx(
          byName[pName].data,
          'primary-analytic-valid',
          gt.analyticValidMask.primaryLeft,
          dispTrue
        ),
      },
      secondary: {
        commonRoi: evaluateDisparityRoiMetricsPx(
          byName[sName].data,
          'secondary-common-roi',
          gt.commonCroppedRoi.secondaryRight,
          dispTrue
        ),
        analyticValid: evaluateDisparityRoiMetricsPx(
          byName[sName].data,
          'secondary-analytic-valid',
          gt.analyticValidMask.secondaryRight,
          dispTrue
        ),
      },
    });

    const wMetrics = scoreArm(wVal.byName);
    const gdMetrics = scoreArm(gdVal.byName);
    const gbMetrics = scoreArm(gbVal.byName);
    const prMetrics = scoreArm(prVal.byName);
    const endToEndExecMs = pPrefixRun.res.coldRunMs + hostMs + pSuffixRun.res.coldRunMs;

    log(
      `  [${gt.caseId} Fixed CommonROI=${gt.commonCroppedRoi.primaryLeft.pixelCount}px]\n` +
        `    WASM(dis)   P: EPE=${wMetrics.primary.commonRoi.epePx.toFixed(4)}px Bad1=${(wMetrics.primary.commonRoi.bad1Rate * 100).toFixed(2)}% Inv=${(wMetrics.primary.commonRoi.invalidRate * 100).toFixed(2)}% | S: EPE=${wMetrics.secondary.commonRoi.epePx.toFixed(4)}px\n` +
        `    GPU(dis)    P: EPE=${gdMetrics.primary.commonRoi.epePx.toFixed(4)}px Bad1=${(gdMetrics.primary.commonRoi.bad1Rate * 100).toFixed(2)}% Inv=${(gdMetrics.primary.commonRoi.invalidRate * 100).toFixed(2)}% | S: EPE=${gdMetrics.secondary.commonRoi.epePx.toFixed(4)}px\n` +
        `    GPU(basic)  P: EPE=${gbMetrics.primary.commonRoi.epePx.toFixed(4)}px Bad1=${(gbMetrics.primary.commonRoi.bad1Rate * 100).toFixed(2)}% Inv=${(gbMetrics.primary.commonRoi.invalidRate * 100).toFixed(2)}% | S: EPE=${gbMetrics.secondary.commonRoi.epePx.toFixed(4)}px\n` +
        `    Pract(#9)   P: EPE=${prMetrics.primary.commonRoi.epePx.toFixed(4)}px Bad1=${(prMetrics.primary.commonRoi.bad1Rate * 100).toFixed(2)}% Inv=${(prMetrics.primary.commonRoi.invalidRate * 100).toFixed(2)}% | S: EPE=${prMetrics.secondary.commonRoi.epePx.toFixed(4)}px | E2E=${endToEndExecMs.toFixed(1)}ms`
    );

    return {
      caseId: gt.caseId,
      disparityTruePx: dispTrue,
      feedSha256: gt.feedSha256,
      commonCroppedRoiPixelCount: gt.commonCroppedRoi.primaryLeft.pixelCount,
      analyticValidMaskPixelCount: {
        primaryLeft: gt.analyticValidMask.primaryLeft.pixelCount,
        secondaryRight: gt.analyticValidMask.secondaryRight.pixelCount,
      },
      arms: {
        unmodifiedWasmFullOptDisabled: {
          ...wMetrics,
          coldRunMs: wasmFullRun.res.coldRunMs,
        },
        unmodifiedGpuFullOptDisabled: {
          ...gdMetrics,
          coldRunMs: gpuDisabledRun.res.coldRunMs,
          dispatchConfirmedSeparatePass: gpuDisabledRun.provenance.separateProfilingPass.dispatchObserved,
        },
        unmodifiedGpuAppBaselineOptBasic: {
          ...gbMetrics,
          coldRunMs: gpuAppBasicRun.res.coldRunMs,
          dispatchConfirmedSeparatePass: gpuAppBasicRun.provenance.separateProfilingPass.dispatchObserved,
        },
        practicalReplaced9PipelineOptDisabled: {
          ...prMetrics,
          endToEndExecuteWallMs: endToEndExecMs,
          hostCandidate9ComputeMs: hostMs,
          prefixDispatchConfirmedSeparatePass: pPrefixRun.provenance.separateProfilingPass.dispatchObserved,
          suffixDispatchConfirmedSeparatePass: pSuffixRun.provenance.separateProfilingPass.dispatchObserved,
        },
      },
      fullFrameParityVsPracticalReplaced9: {
        vsUnmodifiedWasmOptDisabled: {
          primary: summarizeRawTensorAndParity(prVal.byName[pName].data, wVal.byName[pName].data),
          secondary: summarizeRawTensorAndParity(prVal.byName[sName].data, wVal.byName[sName].data),
        },
        vsUnmodifiedGpuOptDisabled: {
          primary: summarizeRawTensorAndParity(prVal.byName[pName].data, gdVal.byName[pName].data),
          secondary: summarizeRawTensorAndParity(prVal.byName[sName].data, gdVal.byName[sName].data),
        },
        vsUnmodifiedGpuAppBaselineOptBasic: {
          primary: summarizeRawTensorAndParity(prVal.byName[pName].data, gbVal.byName[pName].data),
          secondary: summarizeRawTensorAndParity(prVal.byName[sName].data, gbVal.byName[sName].data),
        },
      },
      roiDeltaSummaryPx: {
        primaryEpeReplaced9MinusGpuDisabledPx:
          prMetrics.primary.commonRoi.epePx - gdMetrics.primary.commonRoi.epePx,
        primaryEpeReplaced9MinusGpuAppBasicPx:
          prMetrics.primary.commonRoi.epePx - gbMetrics.primary.commonRoi.epePx,
        primaryBad1DeltaVsGpuDisabled:
          prMetrics.primary.commonRoi.bad1Rate - gdMetrics.primary.commonRoi.bad1Rate,
        primaryInvalidDeltaVsGpuDisabled:
          prMetrics.primary.commonRoi.invalidRate - gdMetrics.primary.commonRoi.invalidRate,
        secondaryEpeReplaced9MinusGpuDisabledPx:
          prMetrics.secondary.commonRoi.epePx - gdMetrics.secondary.commonRoi.epePx,
        secondaryEpeReplaced9MinusGpuAppBasicPx:
          prMetrics.secondary.commonRoi.epePx - gbMetrics.secondary.commonRoi.epePx,
        secondaryBad1DeltaVsGpuDisabled:
          prMetrics.secondary.commonRoi.bad1Rate - gdMetrics.secondary.commonRoi.bad1Rate,
        secondaryInvalidDeltaVsGpuDisabled:
          prMetrics.secondary.commonRoi.invalidRate - gdMetrics.secondary.commonRoi.invalidRate,
      },
    };
  };

  const d16Report = await runSyntheticCase(16);
  const d32Report = await runSyntheticCase(32);

  return {
    ok: true,
    aborted: false,
    abortReason: null,
    pinnedOrtVersion: '1.30.0',
    runtimeAssetFamily: 'asyncify',
    interventionGraphOptimizationLevel: 'disabled',
    appBaselineGraphOptimizationLevel: 'basic',
    numThreads: 1,
    proxy: false,
    m1ByteLength: params.m1Bytes.byteLength,
    m1Sha256: m1Sha,
    n1OffProof,
    chainSpec,
    verifiedArtifacts: baseArtifactsManifest,
    stage1: { f0102: s1F0102, f0126: s1F0126, selection },
    stage2RealFrames: {
      executed: true,
      stopReason: null,
      f0102: s2F0102,
      f0126: s2F0126,
    },
    stage3SyntheticD16D32: {
      executed: true,
      stopReason: null,
      coordinateUnitsNote:
        'All disparities and error metrics (EPE, p95, Bad1>1.0px, Bad3>3.0px, invalidRate) are in native 320x240 HITNet model pixel units (px) on fixed ROI geometry; NO invented baseline/focal calibration.',
      D16: d16Report,
      D32: d32Report,
    },
    scientificScopeNote:
      'DEV-only #9 blocked accumulation verification and practical injection comparison (prefix #0..#8 GPU -> candidate #9 -> suffix #10..end GPU), strictly separating common-WASM-#7 causal controls from unmodified full-GPU app baselines, distinguishing asset download provenance from separate profiling-ON GPU dispatch proof, and accounting full end-to-end wall timing without cached arrays.',
  };
}
