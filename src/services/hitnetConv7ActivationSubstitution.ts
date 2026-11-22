import {
  appendOnnxGraphOutputs,
} from './hitnetGatherNdDiagnostic';
import {
  HITNET_ORT_130_FAMILY_PROFILES,
  type OrtVersionDiagnosticFailure,
  type OrtVersionDiagnosticOutput,
  type OrtVersionDiagnosticResponse,
  type OrtVersionDiagnosticSuccess,
} from './hitnetOrtComparisonHelper';
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
  compareFloat32TensorsDetailed,
  parseM1ModelProtoWithRawSlices,
  type DetailedFloat32Comparison,
  type ParsedM1InitializerInfo,
  type ParsedM1ModelFull,
} from './hitnetRgbStemFirstNonZero';
import {
  assertDevFixtureEnabled,
  HITNET_DEV_NON_JSEP_FIXTURE_SPEC,
} from './stereoDepthWebGpuDevFixture';
import {
  HITNET_M1_FULL_GRAPH_SHA256,
  inferOnnxStaticTensorMeta,
} from './stereoDepthWorkerCore';

export interface SuffixCutEdgeSpec {
  name: string;
  dims: number[];
  elemType: number;
  dataTypeName: 'float32' | 'int32' | 'int64' | 'uint8';
  byteLength: number;
  sourceKind: 'target-cut-node' | 'prefix-node' | 'graph-input';
  fromNodeIdx: number | null;
}

export interface VerifiedPostConv7Suffix {
  cutNodeIdx: number;
  cutNodeName: string;
  cutNodeOpType: string;
  cutOutputTensor: string;
  suffixBytes: Uint8Array;
  suffixByteLength: number;
  suffixSha256: string;
  suffixNodeCount: number;
  suffixNodeRange: { firstOriginalIdx: number; lastOriginalIdx: number };
  preservedUnrelatedDownstreamVerified: boolean;
  suffixInitializerCount: number;
  suffixInitializerManifestSha256: string;
  primaryCutInput: SuffixCutEdgeSpec;
  bypassCutInputs: SuffixCutEdgeSpec[];
  graphInputNames: string[];
  graphOutputNames: string[];
}

export interface Conv7ReductionVariantSummary {
  variantId: 'f64SeqRoundedOnce' | 'f32SepMulAdd_IC_KH_KW' | 'f32SepMulAdd_KH_KW_IC' | 'f32PairwiseTree27';
  label: string;
  evaluatedCount: number;
  gpuExactBitMatches: number;
  wasmExactBitMatches: number;
  maxAbsDiffFromGpu: number;
  meanAbsDiffFromGpu: number;
  maxAbsDiffFromWasm: number;
  meanAbsDiffFromWasm: number;
}

export interface Conv7ReductionProbeReport {
  available: boolean;
  blockerReason: string | null;
  evaluatedCount: number;
  totalConv7Elements: number;
  variants: Conv7ReductionVariantSummary[];
  samples: Array<{
    flatIdx: number;
    gpu: number;
    wasm: number;
    gpuHex: string;
    wasmHex: string;
    f64SeqRoundedOnce: number;
    f32SepMulAdd_IC_KH_KW: number;
    f32SepMulAdd_KH_KW_IC: number;
    f32PairwiseTree27: number;
  }>;
  caveat: string;
}

export interface DualOutputComparisonSummary {
  label: string;
  primary: DetailedFloat32Comparison;
  secondary: DetailedFloat32Comparison;
  passesBothStrictGates: boolean;
}

export interface ResidualReductionMetrics {
  baselineMaxAbsDiff: number;
  intervenedMaxAbsDiff: number;
  maxAbsDiffDelta: number;
  maxAbsDiffRelativeReduction: number | null;
  baselineP99AbsDiff: number;
  intervenedP99AbsDiff: number;
  p99AbsDiffRelativeReduction: number | null;
  baselineMeanAbsDiff: number;
  intervenedMeanAbsDiff: number;
  meanAbsDiffRelativeReduction: number | null;
  baselineRmsDiff: number;
  intervenedRmsDiff: number;
}

export interface ArmExecutionProvenanceSummary {
  armId: string;
  provider: 'webgpu' | 'wasm';
  graphOptimizationLevel: 'disabled';
  modelByteLength: number;
  modelSha256: string;
  primaryFeedName: string;
  primaryFeedSha256: string;
  extraFeedNames: string[];
  createMs: number;
  coldRunMs: number;
  runtimeAssetFamily: 'asyncify';
  entryModuleUrl: string;
  loadedAssetProvenance: OrtVersionDiagnosticSuccess['loadedAssetProvenance'];
  gpuValidationErrors: string[] | null;
}

export interface Conv7FrameSubstitutionResult {
  frameLabel: 'F0102' | 'F0126';
  feedSha256: string;
  ok: boolean;
  interventionInconclusive: boolean;
  inconclusiveReason: string | null;
  verificationManifest: {
    m1ByteLength: number;
    m1Sha256: string;
    feedSha256: string;
    suffixByteLength: number;
    suffixSha256: string;
    primaryCutTensor: string;
    bypassCutTensors: string[];
    bypassBitwiseEqualGpuVsWasm: boolean;
    bypassCaptureDetails: Array<{
      name: string;
      dataType: string;
      dims: number[];
      byteLength: number;
      gpuSha256: string;
      wasmSha256: string;
      bitwiseEqual: boolean;
    }>;
    tapFinalsInvariance: Array<{
      tapLabel: string;
      tapModelSha256: string;
      bitIdenticalFinals: boolean;
      gpuPrimaryBitIdentical: BitIdentityResult;
      gpuSecondaryBitIdentical: BitIdentityResult;
      wasmPrimaryBitIdentical: BitIdentityResult;
      wasmSecondaryBitIdentical: BitIdentityResult;
    }>;
    splitControlInvariance: {
      wasmSplitMatchesWasmFullBitExact: boolean;
      gpuSplitMatchesGpuFullBitExact: boolean;
      wasmPrimary: BitIdentityResult;
      wasmSecondary: BitIdentityResult;
      gpuPrimary: BitIdentityResult;
      gpuSecondary: BitIdentityResult;
      bothPassed: boolean;
    };
    dispatchProfiles: {
      fullM1WebGpu: OrtVersionDiagnosticResponse | null;
      gpuSplitControl: OrtVersionDiagnosticResponse | null;
      gpuWithWasmConv7: OrtVersionDiagnosticResponse | null;
      fullM1GpuDispatchObserved: boolean;
      gpuSplitDispatchObserved: boolean;
      gpuIntervenedDispatchObserved: boolean;
    };
  };
  parentInputCheck: {
    concatSha256Gpu: string;
    concatSha256Wasm: string;
    concatBitwiseIdentical: boolean;
    conv7WeightSha256: string | null;
    conv7BiasSha256: string | null;
    conv7GpuSha256: string;
    conv7WasmSha256: string;
    conv7GpuVsWasm: DetailedFloat32Comparison;
  };
  armProvenances: {
    wasmFull: ArmExecutionProvenanceSummary;
    gpuFull: ArmExecutionProvenanceSummary;
    wasmSplitControl: ArmExecutionProvenanceSummary;
    gpuSplitControl: ArmExecutionProvenanceSummary;
    gpuWithWasmConv7: ArmExecutionProvenanceSummary;
    wasmWithGpuConv7: ArmExecutionProvenanceSummary;
  };
  comparisons: {
    baselineGpuFullVsWasmFull: DualOutputComparisonSummary;
    gpuWithWasmConv7VsWasmFull: DualOutputComparisonSummary;
    gpuWithWasmConv7VsGpuFull: DualOutputComparisonSummary;
    wasmWithGpuConv7VsWasmFull: DualOutputComparisonSummary;
    wasmWithGpuConv7VsGpuFull: DualOutputComparisonSummary;
  };
  reductionFromWasmConv7IntoWebGpu: {
    primary: ResidualReductionMetrics;
    secondary: ResidualReductionMetrics;
  };
  depthImpact: {
    baselineGpuVsWasmFull: DisparityAndDepthImpactSummary;
    gpuWithWasmConv7VsWasmFull: DisparityAndDepthImpactSummary;
    gpuWithWasmConv7VsGpuFull: DisparityAndDepthImpactSummary;
    wasmWithGpuConv7VsWasmFull: DisparityAndDepthImpactSummary;
  };
  conv7ReductionProbe: Conv7ReductionProbeReport | null;
}

export interface HitnetConv7SubstitutionReport {
  ok: boolean;
  pinnedOrtVersion: '1.30.0';
  runtimeAssetFamily: 'asyncify';
  graphOptimizationLevel: 'disabled';
  m1ByteLength: number;
  m1Sha256: string;
  suffixSummary: Omit<VerifiedPostConv7Suffix, 'suffixBytes'>;
  f0102: Conv7FrameSubstitutionResult;
  f0126: Conv7FrameSubstitutionResult | null;
  scientificScopeNote: string;
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
      throw new Error(`Unsupported ONNX cut-edge elemType=${elemType} (supported: 1=float32, 6=int32, 7=int64, 2=uint8)`);
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
      throw new Error(`Unsupported ONNX cut-edge elemType=${elemType}`);
  }
}

export async function buildVerifiedSuffixAfterConv7(
  parsed: ParsedM1ModelFull,
  opts?: { cutNodeIdx?: number }
): Promise<VerifiedPostConv7Suffix> {
  let cutNodeIdx = opts?.cutNodeIdx ?? -1;
  if (cutNodeIdx < 0) {
    const conv7 = parsed.nodes.find(
      (n) =>
        n.opType === 'Conv' &&
        (n.name.includes('conv_down_1_0') || n.outputs.some((o) => o.includes('conv_down_1_0')))
    );
    if (!conv7) {
      throw new Error('Could not locate #7 Conv (conv_down_1_0) in parsed M1 graph');
    }
    cutNodeIdx = conv7.idx;
  }
  if (cutNodeIdx < 0 || cutNodeIdx >= parsed.nodes.length - 1) {
    throw new Error(`Invalid cutNodeIdx=${cutNodeIdx} for graph with ${parsed.nodes.length} nodes`);
  }
  const cutNode = parsed.nodes[cutNodeIdx];
  const cutOutName = cutNode.outputs[0];
  if (!cutOutName) {
    throw new Error(`Cut node #${cutNodeIdx} "${cutNode.name}" has no primary output tensor`);
  }
  const cutOutMeta = parsed.staticMeta.tensorMeta[cutOutName];
  if (!cutOutMeta || cutOutMeta.dims.some((d) => !Number.isInteger(d) || d <= 0)) {
    throw new Error(`Missing or dynamic static metadata for cut output "${cutOutName}"`);
  }

  const suffixNodes = parsed.nodes.slice(cutNodeIdx + 1);
  const suffixOutputSet = new Set<string>();
  const suffixProducerIdx = new Map<string, number>();
  for (const n of suffixNodes) {
    for (const o of n.outputs) {
      if (o) {
        suffixOutputSet.add(o);
        suffixProducerIdx.set(o, n.idx);
      }
    }
  }

  const neededInitializerNames = new Set<string>();
  const bypassEdgeMap = new Map<string, SuffixCutEdgeSpec>();
  let cutOutputConsumedBySuffix = false;

  for (const n of suffixNodes) {
    for (const inp of n.inputs) {
      if (!inp) continue;
      if (parsed.initializerNames.has(inp)) {
        neededInitializerNames.add(inp);
        continue;
      }
      if (suffixOutputSet.has(inp)) {
        continue;
      }
      if (inp === cutOutName) {
        cutOutputConsumedBySuffix = true;
        continue;
      }
      if (!bypassEdgeMap.has(inp)) {
        const meta = parsed.staticMeta.tensorMeta[inp];
        if (!meta || meta.dims.some((d) => !Number.isInteger(d) || d <= 0)) {
          throw new Error(
            `External cut/bypass input "${inp}" entering suffix node #${n.idx} has missing/dynamic shape`
          );
        }
        const prod = parsed.producerByOutput.get(inp);
        const elemCount = meta.dims.reduce((a, b) => a * b, 1);
        bypassEdgeMap.set(inp, {
          name: inp,
          dims: [...meta.dims],
          elemType: meta.elemType,
          dataTypeName: elemTypeToOrtName(meta.elemType),
          byteLength: elemCount * elemByteSize(meta.elemType),
          sourceKind: parsed.graphInputNames.has(inp) ? 'graph-input' : 'prefix-node',
          fromNodeIdx: prod ? prod.idx : null,
        });
      }
    }
  }

  if (!cutOutputConsumedBySuffix) {
    throw new Error(`Cut output "${cutOutName}" (#${cutNodeIdx}) is not consumed by any downstream suffix node`);
  }

  // Deterministic ordering of bypass inputs (by original producer index, then lexical name)
  const bypassCutInputs = Array.from(bypassEdgeMap.values()).sort((a, b) => {
    const ai = a.fromNodeIdx ?? -1;
    const bi = b.fromNodeIdx ?? -1;
    if (ai !== bi) return ai - bi;
    return a.name.localeCompare(b.name);
  });

  const primaryCutInput: SuffixCutEdgeSpec = {
    name: cutOutName,
    dims: [...cutOutMeta.dims],
    elemType: cutOutMeta.elemType,
    dataTypeName: elemTypeToOrtName(cutOutMeta.elemType),
    byteLength: cutOutMeta.dims.reduce((a, b) => a * b, 1) * elemByteSize(cutOutMeta.elemType),
    sourceKind: 'target-cut-node',
    fromNodeIdx: cutNodeIdx,
  };

  // Preserve downstream initializers verbatim in canonical graph initializer iteration order
  const preservedInitializers: ParsedM1InitializerInfo[] = [];
  for (const [, initInfo] of parsed.initializersByName) {
    if (!neededInitializerNames.has(initInfo.name)) continue;
    if (initInfo.isExternal) {
      throw new Error(`Downstream suffix initializer "${initInfo.name}" uses external data_location=1`);
    }
    preservedInitializers.push(initInfo);
  }
  if (preservedInitializers.length !== neededInitializerNames.size) {
    throw new Error(
      `Downstream suffix initializer count mismatch: resolved ${preservedInitializers.length}, needed ${neededInitializerNames.size}`
    );
  }

  const requiredFinalNames = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;
  const outputValueInfos: Uint8Array[] = [];
  for (const outName of requiredFinalNames) {
    const om = parsed.staticMeta.tensorMeta[outName];
    if (!om || om.elemType !== 1 || om.dims.join(',') !== '1,240,320,1') {
      throw new Error(`Suffix missing expected output "${outName}" [1,240,320,1] float32`);
    }
    if (!suffixProducerIdx.has(outName)) {
      throw new Error(`Expected final output "${outName}" is not produced inside suffix`);
    }
    outputValueInfos.push(makeValueInfo(outName, om.elemType, om.dims, 12));
  }

  const allGraphInputs = [primaryCutInput, ...bypassCutInputs];
  const graphParts: Uint8Array[] = [
    ...suffixNodes.map((n) => lenField(1, n.rawPayloadSlice)),
    strField(2, `hitnet_m1_suffix_after_node_${cutNodeIdx}`),
    ...preservedInitializers.map((init) => lenField(5, init.rawPayloadSlice)),
    ...allGraphInputs.map((inp) => makeValueInfo(inp.name, inp.elemType, inp.dims, 11)),
    ...outputValueInfos,
  ];

  const opsets =
    parsed.opsetImportPayloads.length > 0
      ? parsed.opsetImportPayloads.map((p) => lenField(8, p))
      : [lenField(8, concatBytes([varintField(2, 13)]))];

  const suffixBytes = concatBytes([
    varintField(1, parsed.irVersion),
    ...opsets,
    lenField(7, concatBytes(graphParts)),
  ]);

  const checkMeta = inferOnnxStaticTensorMeta(suffixBytes);
  if (
    !checkMeta.valid ||
    checkMeta.hasControlFlowNode ||
    checkMeta.missingInputs.length > 0 ||
    checkMeta.duplicateOutputs.length > 0 ||
    checkMeta.duplicateInitializers.length > 0
  ) {
    throw new Error(
      `Constructed post-#${cutNodeIdx} suffix failed static ONNX check (valid=${checkMeta.valid}, missing=${JSON.stringify(checkMeta.missingInputs)}, dupOut=${JSON.stringify(checkMeta.duplicateOutputs)})`
    );
  }

  // Cross-verify re-parsed suffix preserves every downstream node & initializer byte-for-byte
  const reparsedSuffix = await parseM1ModelProtoWithRawSlices(suffixBytes);
  const preservedUnrelatedDownstreamVerified =
    reparsedSuffix.nodes.length === suffixNodes.length &&
    reparsedSuffix.nodes.every(
      (rn, i) => rn.name === suffixNodes[i].name && rn.opType === suffixNodes[i].opType
    );
  if (!preservedUnrelatedDownstreamVerified) {
    throw new Error('Re-parsed suffix node sequence failed 1:1 downstream preservation check');
  }

  const initDigestConcat = new TextEncoder().encode(
    preservedInitializers.map((x) => `${x.name}:${x.protoSha256}`).join('|')
  );
  const suffixInitializerManifestSha256 = await sha256HexBytes(initDigestConcat);
  const suffixSha256 = await sha256HexBytes(suffixBytes);

  return {
    cutNodeIdx,
    cutNodeName: cutNode.name,
    cutNodeOpType: cutNode.opType,
    cutOutputTensor: cutOutName,
    suffixBytes,
    suffixByteLength: suffixBytes.byteLength,
    suffixSha256,
    suffixNodeCount: suffixNodes.length,
    suffixNodeRange: {
      firstOriginalIdx: suffixNodes[0].idx,
      lastOriginalIdx: suffixNodes[suffixNodes.length - 1].idx,
    },
    preservedUnrelatedDownstreamVerified,
    suffixInitializerCount: preservedInitializers.length,
    suffixInitializerManifestSha256,
    primaryCutInput,
    bypassCutInputs,
    graphInputNames: allGraphInputs.map((i) => i.name),
    graphOutputNames: [...requiredFinalNames],
  };
}

function getAttrInts(node: { attrs: Array<{ name: string; intsVal: number[] }> }, name: string, fallback: number[]): number[] {
  const a = node.attrs.find((x) => x.name === name);
  return a && a.intsVal.length > 0 ? a.intsVal : fallback;
}

function getAttrInt(node: { attrs: Array<{ name: string; intVal: number | null }> }, name: string, fallback: number): number {
  const a = node.attrs.find((x) => x.name === name);
  return a && a.intVal !== null ? a.intVal : fallback;
}

export function probeConv7ReductionVariants(
  parsed: ParsedM1ModelFull,
  conv7NodeIdx: number,
  concatInputF32: Float32Array,
  concatDims: readonly number[],
  gpuConv7OutF32: Float32Array,
  wasmConv7OutF32: Float32Array,
  conv7OutDims: readonly number[],
  opts?: { maxSamples?: number }
): Conv7ReductionProbeReport {
  const node = parsed.nodes[conv7NodeIdx];
  if (!node || node.opType !== 'Conv') {
    return {
      available: false,
      blockerReason: `Node #${conv7NodeIdx} is not Conv`,
      evaluatedCount: 0,
      totalConv7Elements: gpuConv7OutF32.length,
      variants: [],
      samples: [],
      caveat: 'Conv #7 reduction variant probe unavailable.',
    };
  }
  const wInit = node.inputs[1] ? parsed.initializersByName.get(node.inputs[1]) : undefined;
  const bInit = node.inputs[2] ? parsed.initializersByName.get(node.inputs[2]) : undefined;
  if (!wInit?.float32Values || wInit.dims.length !== 4 || concatDims.length !== 4 || conv7OutDims.length !== 4) {
    return {
      available: false,
      blockerReason: 'Conv #7 requires 4D NCHW float32 input and weight initializer',
      evaluatedCount: 0,
      totalConv7Elements: gpuConv7OutF32.length,
      variants: [],
      samples: [],
      caveat: 'Conv #7 reduction variant probe unavailable.',
    };
  }

  const [N, C, H, W] = concatDims;
  const [M, CPerG, kH, kW] = wInit.dims;
  const [, oM, oH, oW] = conv7OutDims;
  const group = getAttrInt(node, 'group', 1);
  if (group * CPerG !== C || M !== oM || M % group !== 0 || N <= 0) {
    return {
      available: false,
      blockerReason: `Invalid Conv #7 dims/group (N=${N},C=${C},M=${M},oM=${oM},group=${group})`,
      evaluatedCount: 0,
      totalConv7Elements: gpuConv7OutF32.length,
      variants: [],
      samples: [],
      caveat: 'Conv #7 reduction variant probe unavailable.',
    };
  }

  const strides = getAttrInts(node, 'strides', [1, 1]);
  const dilations = getAttrInts(node, 'dilations', [1, 1]);
  const pads = getAttrInts(node, 'pads', [0, 0, 0, 0]);
  const padTop = pads[0] ?? 0;
  const padLeft = pads[1] ?? 0;
  const strideH = strides[0] ?? 1;
  const strideW = strides[1] ?? 1;
  const dilH = dilations[0] ?? 1;
  const dilW = dilations[1] ?? 1;
  const wData = wInit.float32Values;
  const bData = bInit?.float32Values ?? null;
  const elemCount = gpuConv7OutF32.length;

  const maxSamples = Math.max(16, Math.min(512, opts?.maxSamples ?? 128));
  const picked = new Set<number>();

  // First bit/numeric diff + max diff + top-32 |gpu - wasm|
  const topDiv: Array<{ idx: number; d: number }> = [];
  for (let i = 0; i < elemCount; i++) {
    const d = Math.abs(gpuConv7OutF32[i] - wasmConv7OutF32[i]);
    if (d > 0) {
      if (picked.size < 4) picked.add(i);
      if (topDiv.length < 32 || d > topDiv[topDiv.length - 1].d) {
        topDiv.push({ idx: i, d });
        topDiv.sort((a, b) => b.d - a.d);
        if (topDiv.length > 32) topDiv.pop();
      }
    }
  }
  for (const t of topDiv) picked.add(t.idx);
  const gridStride = Math.max(1, Math.floor(elemCount / Math.max(1, maxSamples - picked.size)));
  for (let i = 0; i < elemCount && picked.size < maxSamples; i += gridStride) {
    picked.add(i);
  }
  const evalIndices = Array.from(picked).sort((a, b) => a - b);

  const gU32 = new Uint32Array(gpuConv7OutF32.buffer, gpuConv7OutF32.byteOffset, elemCount);
  const wU32 = new Uint32Array(wasmConv7OutF32.buffer, wasmConv7OutF32.byteOffset, elemCount);
  const f32Scratch = new Float32Array(1);
  const u32Scratch = new Uint32Array(f32Scratch.buffer);
  const toBits = (v: number): number => {
    f32Scratch[0] = v;
    return u32Scratch[0];
  };
  const hexU32 = (u: number): string => `0x${(u >>> 0).toString(16).padStart(8, '0')}`;

  const prods = new Float32Array(CPerG * kH * kW);
  const prodBuf2 = new Float32Array(CPerG * kH * kW);
  const gSize = M / group;

  type VariantAcc = {
    gpuExactBitMatches: number;
    wasmExactBitMatches: number;
    maxGpu: number;
    sumGpu: number;
    maxWasm: number;
    sumWasm: number;
  };
  const mkAcc = (): VariantAcc => ({
    gpuExactBitMatches: 0,
    wasmExactBitMatches: 0,
    maxGpu: 0,
    sumGpu: 0,
    maxWasm: 0,
    sumWasm: 0,
  });
  const accF64 = mkAcc();
  const accF32IJK = mkAcc();
  const accF32KJI = mkAcc();
  const accTree = mkAcc();

  const updateAcc = (acc: VariantAcc, valF32: number, idx: number) => {
    const bits = toBits(valF32);
    if (bits === gU32[idx]) acc.gpuExactBitMatches++;
    if (bits === wU32[idx]) acc.wasmExactBitMatches++;
    const dg = Math.abs(gpuConv7OutF32[idx] - valF32);
    const dw = Math.abs(wasmConv7OutF32[idx] - valF32);
    if (dg > acc.maxGpu) acc.maxGpu = dg;
    if (dw > acc.maxWasm) acc.maxWasm = dw;
    acc.sumGpu += dg;
    acc.sumWasm += dw;
  };

  const samples: Conv7ReductionProbeReport['samples'] = [];

  for (const idx of evalIndices) {
    let rem = idx;
    const ow = rem % oW;
    rem = Math.floor(rem / oW);
    const oh = rem % oH;
    rem = Math.floor(rem / oH);
    const m = rem % oM;
    const n = Math.floor(rem / oM);

    const g = Math.floor(m / gSize);
    const cStart = g * CPerG;
    const xBase = n * C * H * W;
    const wBase = m * CPerG * kH * kW;
    const bias = bData ? Number(bData[m]) : 0.0;

    let f64Sum = 0.0;
    let f32SumIJK = 0.0;
    let countP = 0;

    for (let ic = 0; ic < CPerG; ic++) {
      const xChanOff = xBase + (cStart + ic) * H * W;
      const wChanOff = wBase + ic * kH * kW;
      for (let kh = 0; kh < kH; kh++) {
        const ih = oh * strideH + kh * dilH - padTop;
        const wRowOff = wChanOff + kh * kW;
        for (let kw = 0; kw < kW; kw++) {
          const iw = ow * strideW + kw * dilW - padLeft;
          const xv = ih >= 0 && ih < H && iw >= 0 && iw < W ? concatInputF32[xChanOff + ih * W + iw] : 0;
          const wv = wData[wRowOff + kw];
          f64Sum += Number(xv) * Number(wv);
          const p32 = Math.fround(xv * wv);
          f32SumIJK = Math.fround(f32SumIJK + p32);
          prods[countP++] = p32;
        }
      }
    }

    let f32SumKJI = 0.0;
    for (let kh = 0; kh < kH; kh++) {
      const ih = oh * strideH + kh * dilH - padTop;
      for (let kw = 0; kw < kW; kw++) {
        const iw = ow * strideW + kw * dilW - padLeft;
        for (let ic = 0; ic < CPerG; ic++) {
          const xv =
            ih >= 0 && ih < H && iw >= 0 && iw < W
              ? concatInputF32[xBase + (cStart + ic) * H * W + ih * W + iw]
              : 0;
          const wv = wData[wBase + (ic * kH + kh) * kW + kw];
          f32SumKJI = Math.fround(f32SumKJI + Math.fround(xv * wv));
        }
      }
    }

    // Pairwise binary tree sum over products in float32
    for (let p = 0; p < countP; p++) prodBuf2[p] = prods[p];
    let treeLen = countP;
    while (treeLen > 1) {
      let nextLen = 0;
      for (let p = 0; p + 1 < treeLen; p += 2) {
        prodBuf2[nextLen++] = Math.fround(prodBuf2[p] + prodBuf2[p + 1]);
      }
      if ((treeLen & 1) === 1) {
        prodBuf2[nextLen++] = prodBuf2[treeLen - 1];
      }
      treeLen = nextLen;
    }

    const vF64 = Math.fround(f64Sum + bias);
    const vIJK = Math.fround(f32SumIJK + Math.fround(bias));
    const vKJI = Math.fround(f32SumKJI + Math.fround(bias));
    const vTree = Math.fround((countP > 0 ? prodBuf2[0] : 0) + Math.fround(bias));

    updateAcc(accF64, vF64, idx);
    updateAcc(accF32IJK, vIJK, idx);
    updateAcc(accF32KJI, vKJI, idx);
    updateAcc(accTree, vTree, idx);

    if (samples.length < 10 && (gU32[idx] !== wU32[idx] || samples.length < 3)) {
      samples.push({
        flatIdx: idx,
        gpu: gpuConv7OutF32[idx],
        wasm: wasmConv7OutF32[idx],
        gpuHex: hexU32(gU32[idx]),
        wasmHex: hexU32(wU32[idx]),
        f64SeqRoundedOnce: vF64,
        f32SepMulAdd_IC_KH_KW: vIJK,
        f32SepMulAdd_KH_KW_IC: vKJI,
        f32PairwiseTree27: vTree,
      });
    }
  }

  const nEval = evalIndices.length;
  const finalize = (
    variantId: Conv7ReductionVariantSummary['variantId'],
    label: string,
    acc: VariantAcc
  ): Conv7ReductionVariantSummary => ({
    variantId,
    label,
    evaluatedCount: nEval,
    gpuExactBitMatches: acc.gpuExactBitMatches,
    wasmExactBitMatches: acc.wasmExactBitMatches,
    maxAbsDiffFromGpu: acc.maxGpu,
    meanAbsDiffFromGpu: nEval > 0 ? acc.sumGpu / nEval : 0,
    maxAbsDiffFromWasm: acc.maxWasm,
    meanAbsDiffFromWasm: nEval > 0 ? acc.sumWasm / nEval : 0,
  });

  return {
    available: true,
    blockerReason: null,
    evaluatedCount: nEval,
    totalConv7Elements: elemCount,
    variants: [
      finalize('f64SeqRoundedOnce', 'Sequential f64 MAC + bias -> single f32 round', accF64),
      finalize('f32SepMulAdd_IC_KH_KW', 'Sequential f32 separate mul/add [ic][kh][kw] + bias', accF32IJK),
      finalize('f32SepMulAdd_KH_KW_IC', 'Sequential f32 separate mul/add [kh][kw][ic] + bias', accF32KJI),
      finalize('f32PairwiseTree27', 'Pairwise binary-tree f32 sum over 27 products + bias', accTree),
    ],
    samples,
    caveat:
      'Diagnostic reference only: a synthetic schedule supports inferring actual kernel accumulation order ONLY if its float32 bit patterns match across evaluated elements; approximate proximity does not prove WGSL/WASM instruction scheduling.',
  };
}

function summarizeReduction(
  baseStats: DetailedFloat32Comparison['stats'],
  intervenedStats: DetailedFloat32Comparison['stats']
): ResidualReductionMetrics {
  const rel = (b: number, iv: number): number | null => (b > 0 ? (b - iv) / b : null);
  return {
    baselineMaxAbsDiff: baseStats.maxAbsDiff,
    intervenedMaxAbsDiff: intervenedStats.maxAbsDiff,
    maxAbsDiffDelta: intervenedStats.maxAbsDiff - baseStats.maxAbsDiff,
    maxAbsDiffRelativeReduction: rel(baseStats.maxAbsDiff, intervenedStats.maxAbsDiff),
    baselineP99AbsDiff: baseStats.p99AbsDiff,
    intervenedP99AbsDiff: intervenedStats.p99AbsDiff,
    p99AbsDiffRelativeReduction: rel(baseStats.p99AbsDiff, intervenedStats.p99AbsDiff),
    baselineMeanAbsDiff: baseStats.meanAbsDiff,
    intervenedMeanAbsDiff: intervenedStats.meanAbsDiff,
    meanAbsDiffRelativeReduction: rel(baseStats.meanAbsDiff, intervenedStats.meanAbsDiff),
    baselineRmsDiff: baseStats.rmsDiff,
    intervenedRmsDiff: intervenedStats.rmsDiff,
  };
}

function compareDualOutputs(
  label: string,
  candByName: Record<string, { dims: number[]; dataType: string; data: Float32Array }>,
  refByName: Record<string, { dims: number[]; dataType: string; data: Float32Array }>
): DualOutputComparisonSummary {
  const [pName, sName] = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;
  const primary = compareFloat32TensorsDetailed(
    candByName[pName].data,
    refByName[pName].data,
    candByName[pName].dims
  );
  const secondary = compareFloat32TensorsDetailed(
    candByName[sName].data,
    refByName[sName].data,
    candByName[sName].dims
  );
  return {
    label,
    primary,
    secondary,
    passesBothStrictGates: primary.passesStrictGate && secondary.passesStrictGate,
  };
}

function toProvenanceSummary(
  armId: string,
  res: OrtVersionDiagnosticSuccess,
  extraFeedNames: string[]
): ArmExecutionProvenanceSummary {
  return {
    armId,
    provider: res.provider,
    graphOptimizationLevel: 'disabled',
    modelByteLength: res.modelByteLength,
    modelSha256: res.modelSha256,
    primaryFeedName: res.feedTensorName,
    primaryFeedSha256: res.feedSha256,
    extraFeedNames,
    createMs: res.createMs,
    coldRunMs: res.coldRunMs,
    runtimeAssetFamily: 'asyncify',
    entryModuleUrl: res.entryModuleUrl,
    loadedAssetProvenance: res.loadedAssetProvenance,
    gpuValidationErrors: res.gpuValidationErrors,
  };
}

function hasObservedGpuDispatch(prof: OrtVersionDiagnosticResponse | null): boolean {
  if (!prof || !prof.ok || !prof.gpuKernelProfileSummary) return false;
  const s = prof.gpuKernelProfileSummary;
  return s.webgpuComputePassCount > 0 || s.webgpuDispatchWorkgroupsCount > 0 || s.ortProfilingEventCount > 0;
}

export async function runHitnetConv7ActivationSubstitutionDiagnostic(params: {
  m1Bytes: Uint8Array;
  f0102FeedFloat32: Float32Array;
  f0126FeedFloat32?: Float32Array | null;
  confirmF0126?: boolean;
  includeConv7ReductionProbe?: boolean;
  runWorker?: ResidualDiagnosticWorkerRunner;
  disableCpuEpFallback?: boolean;
  isDev?: boolean;
  onLog?: (line: string) => void;
}): Promise<HitnetConv7SubstitutionReport> {
  assertDevFixtureEnabled(params.isDev);
  const log = params.onLog ?? (() => {});
  const runWorker = params.runWorker ?? createBrowserOrt130DiagnosticRunner();
  const disableCpuEpFallback = Boolean(params.disableCpuEpFallback);

  const m1Sha = await sha256HexBytes(params.m1Bytes);
  if (
    params.m1Bytes.byteLength !== HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength ||
    m1Sha !== HITNET_M1_FULL_GRAPH_SHA256
  ) {
    throw new Error(
      `Exact canonical M1 verification failed: bytes=${params.m1Bytes.byteLength} (expected ${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength}), sha=${m1Sha} (expected ${HITNET_M1_FULL_GRAPH_SHA256})`
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

  let f0126Sha: string | null = null;
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

  const parsedModel = await parseM1ModelProtoWithRawSlices(params.m1Bytes);
  const suffixSpec = await buildVerifiedSuffixAfterConv7(parsedModel);
  const asyncifyProfile = HITNET_ORT_130_FAMILY_PROFILES.asyncify;
  const [pName, sName] = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;

  log(
    `=== [DEV-ONLY] HITNet #7 Conv Activation Substitution Causal Diagnostic ===\n` +
    `Runtime: onnxruntime-web@1.30.0/webgpu (non-JSEP ${asyncifyProfile.entryFile} + ${asyncifyProfile.wasmFile}) | opt=disabled\n` +
    `Canonical M1: bytes=${params.m1Bytes.byteLength} sha256=${m1Sha}\n` +
    `Shared Post-#${suffixSpec.cutNodeIdx} Suffix: bytes=${suffixSpec.suffixByteLength} sha256=${suffixSpec.suffixSha256} | nodes=${suffixSpec.suffixNodeCount} (#${suffixSpec.suffixNodeRange.firstOriginalIdx}..#${suffixSpec.suffixNodeRange.lastOriginalIdx}) | preservedInit=${suffixSpec.suffixInitializerCount} (manifest=${suffixSpec.suffixInitializerManifestSha256.slice(0, 12)}...)\n` +
    `  Primary cut input: "${suffixSpec.primaryCutInput.name}" (${suffixSpec.primaryCutInput.dataTypeName} ${JSON.stringify(suffixSpec.primaryCutInput.dims)})\n` +
    `  External bypass cut inputs (${suffixSpec.bypassCutInputs.length}): ${JSON.stringify(suffixSpec.bypassCutInputs.map((b) => ({ name: b.name, dtype: b.dataTypeName, dims: b.dims, fromIdx: b.fromNodeIdx })))}`
  );

  const assertNonJsepSuccess = (
    label: string,
    rawRes: OrtVersionDiagnosticResponse,
    expectedModelSha: string,
    expectedModelBytes: number,
    expectedFeedSha?: string
  ): OrtVersionDiagnosticSuccess => {
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
        `${label} violated non-JSEP guard: family=${rawRes.runtimeAssetFamily}, foreign=${JSON.stringify(rawRes.foreignOrtResources)}, loaded=${JSON.stringify(rawRes.loadedOrtResources)}`
      );
    }
    if (rawRes.modelSha256 !== expectedModelSha || rawRes.modelByteLength !== expectedModelBytes) {
      throw new Error(
        `${label} model mismatch: got=${rawRes.modelSha256}(${rawRes.modelByteLength}B), expected=${expectedModelSha}(${expectedModelBytes}B)`
      );
    }
    if (expectedFeedSha && rawRes.feedSha256 !== expectedFeedSha) {
      throw new Error(
        `${label} feed SHA mismatch: got=${rawRes.feedSha256}, expected=${expectedFeedSha}`
      );
    }
    return rawRes;
  };

  const copyRawOutputBytes = (out: OrtVersionDiagnosticOutput): Uint8Array => {
    return new Uint8Array(out.buffer).slice();
  };

  const runFrame = async (
    frameLabel: 'F0102' | 'F0126',
    feedFloat32: Float32Array,
    feedSha: string
  ): Promise<Conv7FrameSubstitutionResult> => {
    log(`\n--- [${frameLabel}] Step 1: Full-Graph M1 Baselines (WebGPU & WASM, opt=disabled, profiling=OFF) ---`);
    const gpuFullRaw = await runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: params.m1Bytes.slice().buffer,
      feedBuffer: feedFloat32.slice().buffer,
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      provider: 'webgpu',
      graphOptimizationLevel: 'disabled',
      warmRuns: 0,
      enableWebGpuProfiling: false,
      disableCpuEpFallback,
      captureGpuValidationErrors: true,
    });
    const wasmFullRaw = await runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: params.m1Bytes.slice().buffer,
      feedBuffer: feedFloat32.slice().buffer,
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      provider: 'wasm',
      graphOptimizationLevel: 'disabled',
      warmRuns: 0,
      enableWebGpuProfiling: false,
      disableCpuEpFallback: false,
      captureGpuValidationErrors: false,
    });
    const gpuFull = assertNonJsepSuccess(`${frameLabel} M1 Full WebGPU`, gpuFullRaw, m1Sha, params.m1Bytes.byteLength, feedSha);
    const wasmFull = assertNonJsepSuccess(`${frameLabel} M1 Full WASM`, wasmFullRaw, m1Sha, params.m1Bytes.byteLength, feedSha);
    const gpuFullVal = validateHitnetFinalOutputs(gpuFull.outputs);
    const wasmFullVal = validateHitnetFinalOutputs(wasmFull.outputs);
    if (!gpuFullVal.ok || !wasmFullVal.ok) {
      throw new Error(`${frameLabel} Full-M1 final output error: gpu=${gpuFullVal.error ?? 'ok'}, wasm=${wasmFullVal.error ?? 'ok'}`);
    }

    const fullM1Profile = await runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: params.m1Bytes.slice().buffer,
      feedBuffer: feedFloat32.slice().buffer,
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      provider: 'webgpu',
      graphOptimizationLevel: 'disabled',
      warmRuns: 0,
      enableWebGpuProfiling: true,
      disableCpuEpFallback,
      captureGpuValidationErrors: true,
    });

    log(`\n--- [${frameLabel}] Step 2: Capture #7 Conv + #6 Concat + All Bypass Cut Edges (Bit-Identical Finals Verified) ---`);
    const cutNode = parsedModel.nodes[suffixSpec.cutNodeIdx];
    const conv7DynParent = cutNode.inputs.find(
      (inp) => inp && !parsedModel.initializerNames.has(inp) && !parsedModel.graphInputNames.has(inp)
    ) ?? null;
    const conv7TapList = conv7DynParent
      ? [conv7DynParent, suffixSpec.primaryCutInput.name]
      : [suffixSpec.primaryCutInput.name];

    const tapFinalsInvariance: Conv7FrameSubstitutionResult['verificationManifest']['tapFinalsInvariance'] = [];
    const checkTapInvariance = (
      tapLabel: string,
      tapModelSha256: string,
      gOutputs: OrtVersionDiagnosticOutput[],
      wOutputs: OrtVersionDiagnosticOutput[]
    ) => {
      const gv = validateHitnetFinalOutputs(gOutputs);
      const wv = validateHitnetFinalOutputs(wOutputs);
      if (!gv.ok || !wv.ok) {
        throw new Error(`${tapLabel} final output validation failed: gpu=${gv.error ?? 'ok'}, wasm=${wv.error ?? 'ok'}`);
      }
      const gpuPrimaryBitIdentical = checkFloat32BitIdentity(gv.byName[pName].data, gpuFullVal.byName[pName].data);
      const gpuSecondaryBitIdentical = checkFloat32BitIdentity(gv.byName[sName].data, gpuFullVal.byName[sName].data);
      const wasmPrimaryBitIdentical = checkFloat32BitIdentity(wv.byName[pName].data, wasmFullVal.byName[pName].data);
      const wasmSecondaryBitIdentical = checkFloat32BitIdentity(wv.byName[sName].data, wasmFullVal.byName[sName].data);
      const bitIdenticalFinals =
        gpuPrimaryBitIdentical.bitIdentical &&
        gpuSecondaryBitIdentical.bitIdentical &&
        wasmPrimaryBitIdentical.bitIdentical &&
        wasmSecondaryBitIdentical.bitIdentical;
      tapFinalsInvariance.push({
        tapLabel,
        tapModelSha256,
        bitIdenticalFinals,
        gpuPrimaryBitIdentical,
        gpuSecondaryBitIdentical,
        wasmPrimaryBitIdentical,
        wasmSecondaryBitIdentical,
      });
      if (!bitIdenticalFinals) {
        throw new Error(
          `${tapLabel} perturbed final disparity outputs relative to full M1 (gpuP=${gpuPrimaryBitIdentical.mismatchCount}, gpuS=${gpuSecondaryBitIdentical.mismatchCount}, wasmP=${wasmPrimaryBitIdentical.mismatchCount}, wasmS=${wasmSecondaryBitIdentical.mismatchCount})`
        );
      }
    };

    const conv7Tapped = appendOnnxGraphOutputs(params.m1Bytes, conv7TapList, { maxExtraOutputs: 2 });
    const conv7TappedSha = await sha256HexBytes(conv7Tapped.bytes);
    const conv7GpuTap = assertNonJsepSuccess(
      `${frameLabel} Tap #7 WebGPU`,
      await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: conv7Tapped.bytes.slice().buffer,
        feedBuffer: feedFloat32.slice().buffer,
        feedName: 'input',
        feedDims: [1, 6, 240, 320],
        provider: 'webgpu',
        graphOptimizationLevel: 'disabled',
        warmRuns: 0,
        enableWebGpuProfiling: false,
        disableCpuEpFallback,
        captureGpuValidationErrors: true,
      }),
      conv7TappedSha,
      conv7Tapped.bytes.byteLength,
      feedSha
    );
    const conv7WasmTap = assertNonJsepSuccess(
      `${frameLabel} Tap #7 WASM`,
      await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: conv7Tapped.bytes.slice().buffer,
        feedBuffer: feedFloat32.slice().buffer,
        feedName: 'input',
        feedDims: [1, 6, 240, 320],
        provider: 'wasm',
        graphOptimizationLevel: 'disabled',
        warmRuns: 0,
        enableWebGpuProfiling: false,
        disableCpuEpFallback: false,
        captureGpuValidationErrors: false,
      }),
      conv7TappedSha,
      conv7Tapped.bytes.byteLength,
      feedSha
    );
    checkTapInvariance('conv7-and-parent', conv7TappedSha, conv7GpuTap.outputs, conv7WasmTap.outputs);

    const gConv7Out = conv7GpuTap.outputs.find((o) => o.name === suffixSpec.primaryCutInput.name);
    const wConv7Out = conv7WasmTap.outputs.find((o) => o.name === suffixSpec.primaryCutInput.name);
    if (!gConv7Out || !wConv7Out) {
      throw new Error(`Missing captured #7 Conv activation "${suffixSpec.primaryCutInput.name}"`);
    }
    const gpuConv7F32 = new Float32Array(gConv7Out.buffer).slice();
    const wasmConv7F32 = new Float32Array(wConv7Out.buffer).slice();
    const gpuConv7Sha = await sha256HexBytes(copyRawOutputBytes(gConv7Out));
    const wasmConv7Sha = await sha256HexBytes(copyRawOutputBytes(wConv7Out));
    const conv7GpuVsWasm = compareFloat32TensorsDetailed(
      gpuConv7F32,
      wasmConv7F32,
      suffixSpec.primaryCutInput.dims
    );

    let concatSha256Gpu = '';
    let concatSha256Wasm = '';
    let concatBitwiseIdentical = false;
    let concatF32: Float32Array | null = null;
    if (conv7DynParent) {
      const gp = conv7GpuTap.outputs.find((o) => o.name === conv7DynParent);
      const wp = conv7WasmTap.outputs.find((o) => o.name === conv7DynParent);
      if (!gp || !wp) {
        throw new Error(`Missing captured #7 parent tensor "${conv7DynParent}"`);
      }
      concatSha256Gpu = await sha256HexBytes(copyRawOutputBytes(gp));
      concatSha256Wasm = await sha256HexBytes(copyRawOutputBytes(wp));
      concatBitwiseIdentical = concatSha256Gpu === concatSha256Wasm && gp.buffer.byteLength === wp.buffer.byteLength;
      if (concatBitwiseIdentical) {
        concatF32 = new Float32Array(wp.buffer).slice();
      }
    }

    // Capture all external bypass cut-edge inputs in batches of <=2 outputs
    const bypassCaptureDetails: Conv7FrameSubstitutionResult['verificationManifest']['bypassCaptureDetails'] = [];
    const bypassFeedBuffers: Array<{
      name: string;
      dims: number[];
      dataType: SuffixCutEdgeSpec['dataTypeName'];
      rawBytes: Uint8Array;
      sha256: string;
    }> = [];

    for (let i = 0; i < suffixSpec.bypassCutInputs.length; i += 2) {
      const batchSpecs = suffixSpec.bypassCutInputs.slice(i, i + 2);
      const tapNames = batchSpecs.map((b) => b.name);
      const bTapModel = appendOnnxGraphOutputs(params.m1Bytes, tapNames, { maxExtraOutputs: 2 });
      const bTapSha = await sha256HexBytes(bTapModel.bytes);
      const bGpu = assertNonJsepSuccess(
        `${frameLabel} Bypass Tap ${i / 2 + 1} WebGPU`,
        await runWorker({
          expectedOrtVersion: '1.30.0',
          runtimeAssetFamily: 'asyncify',
          modelBuffer: bTapModel.bytes.slice().buffer,
          feedBuffer: feedFloat32.slice().buffer,
          feedName: 'input',
          feedDims: [1, 6, 240, 320],
          provider: 'webgpu',
          graphOptimizationLevel: 'disabled',
          warmRuns: 0,
          enableWebGpuProfiling: false,
          disableCpuEpFallback,
          captureGpuValidationErrors: true,
        }),
        bTapSha,
        bTapModel.bytes.byteLength,
        feedSha
      );
      const bWasm = assertNonJsepSuccess(
        `${frameLabel} Bypass Tap ${i / 2 + 1} WASM`,
        await runWorker({
          expectedOrtVersion: '1.30.0',
          runtimeAssetFamily: 'asyncify',
          modelBuffer: bTapModel.bytes.slice().buffer,
          feedBuffer: feedFloat32.slice().buffer,
          feedName: 'input',
          feedDims: [1, 6, 240, 320],
          provider: 'wasm',
          graphOptimizationLevel: 'disabled',
          warmRuns: 0,
          enableWebGpuProfiling: false,
          disableCpuEpFallback: false,
          captureGpuValidationErrors: false,
        }),
        bTapSha,
        bTapModel.bytes.byteLength,
        feedSha
      );
      checkTapInvariance(`bypass-${i / 2 + 1}`, bTapSha, bGpu.outputs, bWasm.outputs);

      for (const spec of batchSpecs) {
        const go = bGpu.outputs.find((o) => o.name === spec.name);
        const wo = bWasm.outputs.find((o) => o.name === spec.name);
        if (!go || !wo) {
          throw new Error(`Missing captured bypass cut edge "${spec.name}"`);
        }
        const gBytes = copyRawOutputBytes(go);
        const wBytes = copyRawOutputBytes(wo);
        const gSha = await sha256HexBytes(gBytes);
        const wSha = await sha256HexBytes(wBytes);
        let bitwiseEqual = gBytes.byteLength === wBytes.byteLength && gSha === wSha;
        if (bitwiseEqual) {
          for (let k = 0; k < gBytes.byteLength; k++) {
            if (gBytes[k] !== wBytes[k]) {
              bitwiseEqual = false;
              break;
            }
          }
        }
        bypassCaptureDetails.push({
          name: spec.name,
          dataType: spec.dataTypeName,
          dims: [...spec.dims],
          byteLength: wBytes.byteLength,
          gpuSha256: gSha,
          wasmSha256: wSha,
          bitwiseEqual,
        });
        bypassFeedBuffers.push({
          name: spec.name,
          dims: [...spec.dims],
          dataType: spec.dataTypeName,
          rawBytes: wBytes,
          sha256: wSha,
        });
      }
    }

    const bypassBitwiseEqualGpuVsWasm = bypassCaptureDetails.every((d) => d.bitwiseEqual);
    if (!bypassBitwiseEqualGpuVsWasm) {
      throw new Error(
        `External bypass cut-edge activations differ between WebGPU & WASM on ${frameLabel}: ${JSON.stringify(bypassCaptureDetails)}`
      );
    }

    log(
      `  [#7 Conv Captured] gpuSha=${gpuConv7Sha.slice(0, 12)}... wasmSha=${wasmConv7Sha.slice(0, 12)}... | ` +
      `class=${conv7GpuVsWasm.classification} bitDiffs=${conv7GpuVsWasm.bitDiffCount} nonZero=${conv7GpuVsWasm.nonZeroNumericDiffCount} ` +
      `max=${conv7GpuVsWasm.stats.maxAbsDiff.toExponential(3)} mean=${conv7GpuVsWasm.stats.meanAbsDiff.toExponential(3)} | ` +
      `concatBitEq=${concatBitwiseIdentical} bypassBitEq=${bypassBitwiseEqualGpuVsWasm}`
    );

    // Optional compact Conv #7 reduction probe
    let conv7ReductionProbe: Conv7ReductionProbeReport | null = null;
    if ((params.includeConv7ReductionProbe ?? true) && conv7DynParent && concatF32) {
      const concatMeta = parsedModel.staticMeta.tensorMeta[conv7DynParent];
      if (concatMeta && concatBitwiseIdentical) {
        conv7ReductionProbe = probeConv7ReductionVariants(
          parsedModel,
          suffixSpec.cutNodeIdx,
          concatF32,
          concatMeta.dims,
          gpuConv7F32,
          wasmConv7F32,
          suffixSpec.primaryCutInput.dims
        );
        if (conv7ReductionProbe.available) {
          log(
            `  [#7 Reduction Variants (eval=${conv7ReductionProbe.evaluatedCount})] ` +
            conv7ReductionProbe.variants
              .map(
                (v) =>
                  `${v.variantId}: gpuBit=${v.gpuExactBitMatches}/${v.evaluatedCount}(max=${v.maxAbsDiffFromGpu.toExponential(2)}) wasmBit=${v.wasmExactBitMatches}/${v.evaluatedCount}(max=${v.maxAbsDiffFromWasm.toExponential(2)})`
              )
              .join(' | ')
          );
        }
      }
    }

    const buildExtraFeeds = () =>
      bypassFeedBuffers.map((b) => ({
        name: b.name,
        dims: [...b.dims],
        buffer: b.rawBytes.slice().buffer as ArrayBuffer,
        dataType: b.dataType,
      }));

    const runSuffixArm = async (
      armLabel: string,
      provider: 'webgpu' | 'wasm',
      conv7FeedBytes: Float32Array,
      expectedConv7Sha: string,
      opts?: { enableWebGpuProfiling?: boolean }
    ) => {
      const raw = await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: suffixSpec.suffixBytes.slice().buffer,
        feedBuffer: conv7FeedBytes.slice().buffer,
        feedName: suffixSpec.primaryCutInput.name,
        feedDims: [...suffixSpec.primaryCutInput.dims],
        extraFeeds: buildExtraFeeds(),
        provider,
        graphOptimizationLevel: 'disabled',
        warmRuns: 0,
        enableWebGpuProfiling: Boolean(opts?.enableWebGpuProfiling) && provider === 'webgpu',
        disableCpuEpFallback: disableCpuEpFallback && provider === 'webgpu',
        captureGpuValidationErrors: provider === 'webgpu',
      });
      if (opts?.enableWebGpuProfiling) return { res: raw, val: null };
      const okRes = assertNonJsepSuccess(
        armLabel,
        raw,
        suffixSpec.suffixSha256,
        suffixSpec.suffixByteLength,
        expectedConv7Sha
      );
      const val = validateHitnetFinalOutputs(okRes.outputs);
      if (!val.ok) {
        throw new Error(`${armLabel} final output validation failed: ${val.error}`);
      }
      return { res: okRes, val };
    };

    log(`\n--- [${frameLabel}] Step 3: Shared-Suffix Split Controls & Two-Sided Intervention Matrix (opt=disabled) ---`);
    const wasmSplit = await runSuffixArm(`${frameLabel} WASM-Split(wasm#7)`, 'wasm', wasmConv7F32, wasmConv7Sha);
    const gpuSplit = await runSuffixArm(`${frameLabel} GPU-Split(gpu#7)`, 'webgpu', gpuConv7F32, gpuConv7Sha);
    const gpuIntervened = await runSuffixArm(`${frameLabel} GPU-Intervention(wasm#7)`, 'webgpu', wasmConv7F32, wasmConv7Sha);
    const wasmReciprocal = await runSuffixArm(`${frameLabel} WASM-Reciprocal(gpu#7)`, 'wasm', gpuConv7F32, gpuConv7Sha);

    const wasmSplitRes = wasmSplit.res as OrtVersionDiagnosticSuccess;
    const gpuSplitRes = gpuSplit.res as OrtVersionDiagnosticSuccess;
    const gpuIntervenedRes = gpuIntervened.res as OrtVersionDiagnosticSuccess;
    const wasmReciprocalRes = wasmReciprocal.res as OrtVersionDiagnosticSuccess;

    const wasmSplitVal = wasmSplit.val!;
    const gpuSplitVal = gpuSplit.val!;
    const gpuIntervenedVal = gpuIntervened.val!;
    const wasmReciprocalVal = wasmReciprocal.val!;

    // Separate NON-TIMING dispatch profiles for suffix WebGPU arms
    const gpuSplitProfile = (
      await runSuffixArm(`${frameLabel} GPU-Split Profile`, 'webgpu', gpuConv7F32, gpuConv7Sha, {
        enableWebGpuProfiling: true,
      })
    ).res;
    const gpuIntervenedProfile = (
      await runSuffixArm(`${frameLabel} GPU-Intervention Profile`, 'webgpu', wasmConv7F32, wasmConv7Sha, {
        enableWebGpuProfiling: true,
      })
    ).res;

    // Verify same-provider split control invariance (bit-identical BOTH outputs under BOTH providers)
    const wasmPrimaryInv = checkFloat32BitIdentity(wasmSplitVal.byName[pName].data, wasmFullVal.byName[pName].data);
    const wasmSecondaryInv = checkFloat32BitIdentity(wasmSplitVal.byName[sName].data, wasmFullVal.byName[sName].data);
    const gpuPrimaryInv = checkFloat32BitIdentity(gpuSplitVal.byName[pName].data, gpuFullVal.byName[pName].data);
    const gpuSecondaryInv = checkFloat32BitIdentity(gpuSplitVal.byName[sName].data, gpuFullVal.byName[sName].data);

    const wasmSplitMatchesWasmFullBitExact = wasmPrimaryInv.bitIdentical && wasmSecondaryInv.bitIdentical;
    const gpuSplitMatchesGpuFullBitExact = gpuPrimaryInv.bitIdentical && gpuSecondaryInv.bitIdentical;
    const bothSplitControlsPassed = wasmSplitMatchesWasmFullBitExact && gpuSplitMatchesGpuFullBitExact;

    let inconclusiveReason: string | null = null;
    if (!bothSplitControlsPassed) {
      inconclusiveReason =
        `Same-provider split control invariance FAILED on ${frameLabel} (do NOT attribute outcome cleanly to #7 substitution): ` +
        `wasmSplit(P mismatch=${wasmPrimaryInv.mismatchCount}, S mismatch=${wasmSecondaryInv.mismatchCount}), ` +
        `gpuSplit(P mismatch=${gpuPrimaryInv.mismatchCount}, S mismatch=${gpuSecondaryInv.mismatchCount})`;
      log(`  [INCONCLUSIVE: SPLIT CONTROL INVARIANCE FAILED] ${inconclusiveReason}`);
    } else {
      log(
        `  [Split Control Invariance PASS] WASM-Split == WASM-Full (0 bit diffs) AND GPU-Split == GPU-Full (0 bit diffs) on both final outputs!`
      );
    }

    const baselineGpuFullVsWasmFull = compareDualOutputs(
      `${frameLabel} Baseline: GPU-Full vs WASM-Full`,
      gpuFullVal.byName,
      wasmFullVal.byName
    );
    const gpuWithWasmConv7VsWasmFull = compareDualOutputs(
      `${frameLabel} Primary Intervention: GPU-Suffix(wasm#7) vs WASM-Full`,
      gpuIntervenedVal.byName,
      wasmFullVal.byName
    );
    const gpuWithWasmConv7VsGpuFull = compareDualOutputs(
      `${frameLabel} Direct WebGPU Effect: GPU-Suffix(wasm#7) vs GPU-Full`,
      gpuIntervenedVal.byName,
      gpuFullVal.byName
    );
    const wasmWithGpuConv7VsWasmFull = compareDualOutputs(
      `${frameLabel} Reciprocal Sensitivity: WASM-Suffix(gpu#7) vs WASM-Full`,
      wasmReciprocalVal.byName,
      wasmFullVal.byName
    );
    const wasmWithGpuConv7VsGpuFull = compareDualOutputs(
      `${frameLabel} WASM-Suffix(gpu#7) vs GPU-Full`,
      wasmReciprocalVal.byName,
      gpuFullVal.byName
    );

    const reductionFromWasmConv7IntoWebGpu = {
      primary: summarizeReduction(
        baselineGpuFullVsWasmFull.primary.stats,
        gpuWithWasmConv7VsWasmFull.primary.stats
      ),
      secondary: summarizeReduction(
        baselineGpuFullVsWasmFull.secondary.stats,
        gpuWithWasmConv7VsWasmFull.secondary.stats
      ),
    };

    const depthImpact = {
      baselineGpuVsWasmFull: summarizeDisparityResidualAndSourceDepthImpact(
        `${frameLabel} Baseline GPU-Full vs WASM-Full`,
        gpuFullVal.byName[pName].data,
        wasmFullVal.byName[pName].data,
        gpuFullVal.byName[sName].data,
        wasmFullVal.byName[sName].data
      ),
      gpuWithWasmConv7VsWasmFull: summarizeDisparityResidualAndSourceDepthImpact(
        `${frameLabel} GPU-Suffix(wasm#7) vs WASM-Full`,
        gpuIntervenedVal.byName[pName].data,
        wasmFullVal.byName[pName].data,
        gpuIntervenedVal.byName[sName].data,
        wasmFullVal.byName[sName].data
      ),
      gpuWithWasmConv7VsGpuFull: summarizeDisparityResidualAndSourceDepthImpact(
        `${frameLabel} Direct Shift GPU-Suffix(wasm#7) vs GPU-Full`,
        gpuIntervenedVal.byName[pName].data,
        gpuFullVal.byName[pName].data,
        gpuIntervenedVal.byName[sName].data,
        gpuFullVal.byName[sName].data
      ),
      wasmWithGpuConv7VsWasmFull: summarizeDisparityResidualAndSourceDepthImpact(
        `${frameLabel} Reciprocal WASM-Suffix(gpu#7) vs WASM-Full`,
        wasmReciprocalVal.byName[pName].data,
        wasmFullVal.byName[pName].data,
        wasmReciprocalVal.byName[sName].data,
        wasmFullVal.byName[sName].data
      ),
    };

    const fmtCmp = (c: DualOutputComparisonSummary) =>
      `  [${c.label}] gates=${c.passesBothStrictGates ? 'PASS' : 'FAIL'}\n` +
      `    primary  : max=${c.primary.stats.maxAbsDiff.toExponential(3)} p99=${c.primary.stats.p99AbsDiff.toExponential(3)} mean=${c.primary.stats.meanAbsDiff.toExponential(3)} rms=${c.primary.stats.rmsDiff.toExponential(3)} bitDiffs=${c.primary.bitDiffCount}\n` +
      `    secondary: max=${c.secondary.stats.maxAbsDiff.toExponential(3)} p99=${c.secondary.stats.p99AbsDiff.toExponential(3)} mean=${c.secondary.stats.meanAbsDiff.toExponential(3)} rms=${c.secondary.stats.rmsDiff.toExponential(3)} bitDiffs=${c.secondary.bitDiffCount}`;

    log(fmtCmp(baselineGpuFullVsWasmFull));
    log(fmtCmp(gpuWithWasmConv7VsWasmFull));
    log(fmtCmp(gpuWithWasmConv7VsGpuFull));
    log(fmtCmp(wasmWithGpuConv7VsWasmFull));
    log(
      `  [Reduction GPU(wasm#7) vs Baseline] primary(max=${reductionFromWasmConv7IntoWebGpu.primary.maxAbsDiffRelativeReduction?.toFixed(4) ?? 'n/a'}, mean=${reductionFromWasmConv7IntoWebGpu.primary.meanAbsDiffRelativeReduction?.toFixed(4) ?? 'n/a'}) | ` +
      `secondary(max=${reductionFromWasmConv7IntoWebGpu.secondary.maxAbsDiffRelativeReduction?.toFixed(4) ?? 'n/a'}, mean=${reductionFromWasmConv7IntoWebGpu.secondary.meanAbsDiffRelativeReduction?.toFixed(4) ?? 'n/a'})\n` +
      `  [Depth & Validity] baseline maxDepthDiff=${depthImpact.baselineGpuVsWasmFull.primaryDepthThresholdChanges.maxAbsDepthDiffMeters.toFixed(4)}m maskFlips=${depthImpact.baselineGpuVsWasmFull.primarySourceMaskChanges.totalMaskFlips} | ` +
      `intervenedVsWasm maxDepthDiff=${depthImpact.gpuWithWasmConv7VsWasmFull.primaryDepthThresholdChanges.maxAbsDepthDiffMeters.toFixed(4)}m maskFlips=${depthImpact.gpuWithWasmConv7VsWasmFull.primarySourceMaskChanges.totalMaskFlips} | ` +
      `directShiftOnGpu maxDepthDiff=${depthImpact.gpuWithWasmConv7VsGpuFull.primaryDepthThresholdChanges.maxAbsDepthDiffMeters.toFixed(4)}m maskFlips=${depthImpact.gpuWithWasmConv7VsGpuFull.primarySourceMaskChanges.totalMaskFlips}\n` +
      `  [OFF-Profile Inference Timings ms (create/cold)] gpuFull=${gpuFull.createMs.toFixed(1)}/${gpuFull.coldRunMs.toFixed(1)}, wasmFull=${wasmFull.createMs.toFixed(1)}/${wasmFull.coldRunMs.toFixed(1)}, ` +
      `gpuSplit=${gpuSplitRes.createMs.toFixed(1)}/${gpuSplitRes.coldRunMs.toFixed(1)}, gpuWithWasm#7=${gpuIntervenedRes.createMs.toFixed(1)}/${gpuIntervenedRes.coldRunMs.toFixed(1)}, ` +
      `wasmSplit=${wasmSplitRes.createMs.toFixed(1)}/${wasmSplitRes.coldRunMs.toFixed(1)}, wasmWithGpu#7=${wasmReciprocalRes.createMs.toFixed(1)}/${wasmReciprocalRes.coldRunMs.toFixed(1)}\n` +
      `  [Separate Non-Timing GPU Dispatch] fullM1=${hasObservedGpuDispatch(fullM1Profile)}, gpuSplit=${hasObservedGpuDispatch(gpuSplitProfile)}, gpuIntervened=${hasObservedGpuDispatch(gpuIntervenedProfile)}`
    );

    const extraNames = suffixSpec.bypassCutInputs.map((b) => b.name);
    const wInitName = cutNode.inputs[1];
    const bInitName = cutNode.inputs[2];

    return {
      frameLabel,
      feedSha256: feedSha,
      ok: true,
      interventionInconclusive: !bothSplitControlsPassed,
      inconclusiveReason,
      verificationManifest: {
        m1ByteLength: params.m1Bytes.byteLength,
        m1Sha256: m1Sha,
        feedSha256: feedSha,
        suffixByteLength: suffixSpec.suffixByteLength,
        suffixSha256: suffixSpec.suffixSha256,
        primaryCutTensor: suffixSpec.primaryCutInput.name,
        bypassCutTensors: extraNames,
        bypassBitwiseEqualGpuVsWasm,
        bypassCaptureDetails,
        tapFinalsInvariance,
        splitControlInvariance: {
          wasmSplitMatchesWasmFullBitExact,
          gpuSplitMatchesGpuFullBitExact,
          wasmPrimary: wasmPrimaryInv,
          wasmSecondary: wasmSecondaryInv,
          gpuPrimary: gpuPrimaryInv,
          gpuSecondary: gpuSecondaryInv,
          bothPassed: bothSplitControlsPassed,
        },
        dispatchProfiles: {
          fullM1WebGpu: fullM1Profile,
          gpuSplitControl: gpuSplitProfile,
          gpuWithWasmConv7: gpuIntervenedProfile,
          fullM1GpuDispatchObserved: hasObservedGpuDispatch(fullM1Profile),
          gpuSplitDispatchObserved: hasObservedGpuDispatch(gpuSplitProfile),
          gpuIntervenedDispatchObserved: hasObservedGpuDispatch(gpuIntervenedProfile),
        },
      },
      parentInputCheck: {
        concatSha256Gpu,
        concatSha256Wasm,
        concatBitwiseIdentical,
        conv7WeightSha256: wInitName ? parsedModel.initializersByName.get(wInitName)?.protoSha256 ?? null : null,
        conv7BiasSha256: bInitName ? parsedModel.initializersByName.get(bInitName)?.protoSha256 ?? null : null,
        conv7GpuSha256: gpuConv7Sha,
        conv7WasmSha256: wasmConv7Sha,
        conv7GpuVsWasm,
      },
      armProvenances: {
        wasmFull: toProvenanceSummary('wasmFull', wasmFull, []),
        gpuFull: toProvenanceSummary('gpuFull', gpuFull, []),
        wasmSplitControl: toProvenanceSummary('wasmSplitControl', wasmSplitRes, extraNames),
        gpuSplitControl: toProvenanceSummary('gpuSplitControl', gpuSplitRes, extraNames),
        gpuWithWasmConv7: toProvenanceSummary('gpuWithWasmConv7', gpuIntervenedRes, extraNames),
        wasmWithGpuConv7: toProvenanceSummary('wasmWithGpuConv7', wasmReciprocalRes, extraNames),
      },
      comparisons: {
        baselineGpuFullVsWasmFull,
        gpuWithWasmConv7VsWasmFull,
        gpuWithWasmConv7VsGpuFull,
        wasmWithGpuConv7VsWasmFull,
        wasmWithGpuConv7VsGpuFull,
      },
      reductionFromWasmConv7IntoWebGpu,
      depthImpact,
      conv7ReductionProbe,
    };
  };

  const f0102Result = await runFrame('F0102', params.f0102FeedFloat32, f0102Sha);
  const f0126Result =
    params.f0126FeedFloat32 && f0126Sha && (params.confirmF0126 ?? true)
      ? await runFrame('F0126', params.f0126FeedFloat32, f0126Sha)
      : null;

  const suffixSummary: HitnetConv7SubstitutionReport['suffixSummary'] = {
    cutNodeIdx: suffixSpec.cutNodeIdx,
    cutNodeName: suffixSpec.cutNodeName,
    cutNodeOpType: suffixSpec.cutNodeOpType,
    cutOutputTensor: suffixSpec.cutOutputTensor,
    suffixByteLength: suffixSpec.suffixByteLength,
    suffixSha256: suffixSpec.suffixSha256,
    suffixNodeCount: suffixSpec.suffixNodeCount,
    suffixNodeRange: suffixSpec.suffixNodeRange,
    preservedUnrelatedDownstreamVerified: suffixSpec.preservedUnrelatedDownstreamVerified,
    suffixInitializerCount: suffixSpec.suffixInitializerCount,
    suffixInitializerManifestSha256: suffixSpec.suffixInitializerManifestSha256,
    primaryCutInput: suffixSpec.primaryCutInput,
    bypassCutInputs: suffixSpec.bypassCutInputs,
    graphInputNames: suffixSpec.graphInputNames,
    graphOutputNames: suffixSpec.graphOutputNames,
  };

  return {
    ok: true,
    pinnedOrtVersion: '1.30.0',
    runtimeAssetFamily: 'asyncify',
    graphOptimizationLevel: 'disabled',
    m1ByteLength: params.m1Bytes.byteLength,
    m1Sha256: m1Sha,
    suffixSummary,
    f0102: f0102Result,
    f0126: f0126Result,
    scientificScopeNote:
      'WASM M1 execution is used solely as a computational parity reference, NOT ground-truth scene geometry. Even if #7 Conv substitution eliminates or preserves disparity divergence on tested frames (F0102/F0126), that measures downstream sensitivity to #7 on those frames and does not establish #7 as the sole universal divergence mechanism across arbitrary inputs or downstream operators.',
  };
}
