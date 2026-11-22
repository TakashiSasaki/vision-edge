import {
  buildVerifiedSuffixAfterConv7,
  probeConv7ReductionVariants,
  type Conv7ReductionProbeReport,
  type VerifiedPostConv7Suffix,
} from './hitnetConv7ActivationSubstitution';
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
  validateHitnetFinalOutputs,
  type BitIdentityResult,
  type ResidualDiagnosticWorkerRunner,
} from './hitnetResidualLocalization';
import {
  buildSingleOpCapturedInputModel,
  compareFloat32TensorsDetailed,
  computeSequentialFloat64Reference,
  parseM1ModelProtoWithRawSlices,
  type DetailedFloat32Comparison,
  type ParsedM1ModelFull,
  type ParsedM1NodeFull,
  type SequentialFloat64RefResult,
} from './hitnetRgbStemFirstNonZero';
import {
  assertDevFixtureEnabled,
  HITNET_DEV_NON_JSEP_FIXTURE_SPEC,
} from './stereoDepthWebGpuDevFixture';
import { HITNET_M1_FULL_GRAPH_SHA256 } from './stereoDepthWorkerCore';

export interface PostConv7TargetNodeSpec {
  nodeIdx: number;
  nodeName: string;
  opType: string;
  outputTensor: string;
  outputDims: number[];
  attrsSummary: string[];
  dynamicInputs: string[];
  initializerNames: string[];
}

export interface VerifiedPostConv7ChainSpec {
  conv7Idx: number;
  conv7OutputTensor: string;
  node8LeakyRelu: PostConv7TargetNodeSpec;
  node9ConvDown0: PostConv7TargetNodeSpec;
  node10LeakyRelu1: PostConv7TargetNodeSpec | null;
  node11ConvDown11: PostConv7TargetNodeSpec | null;
  bypassTile2Nodes: Array<{
    nodeIdx: number;
    nodeName: string;
    outputTensor: string;
    dims: number[];
    dataTypeName: string;
  }>;
  suffixSummary: Omit<VerifiedPostConv7Suffix, 'suffixBytes'>;
}

export interface TapInvarianceCheckRecord {
  label: string;
  modelByteLength: number;
  modelSha256: string;
  referenceLabel: string;
  gpuPrimaryBitIdentical: BitIdentityResult;
  gpuSecondaryBitIdentical: BitIdentityResult;
  wasmPrimaryBitIdentical: BitIdentityResult;
  wasmSecondaryBitIdentical: BitIdentityResult;
  bothPassed: boolean;
}

export interface SuffixNodeTapObservation {
  nodeIdx: number;
  nodeName: string;
  opType: string;
  outputTensor: string;
  dims: number[];
  dataType: string;
  byteLength: number;
  gpuSha256: string;
  wasmSha256: string;
  comparison: DetailedFloat32Comparison;
  immediateDynamicInputs: Array<{
    name: string;
    source: 'common-conv7-feed' | 'paired-node-tap';
    bitwiseIdentical: boolean;
    gpuSha256: string;
    wasmSha256: string;
  }>;
  initializers: Array<{
    name: string;
    dims: number[];
    dataTypeName: string;
    protoSha256: string;
    valueSha256: string | null;
  }>;
  allImmediateInputsByteEqual: boolean;
}

export interface LeakyReluVariantProbeReport {
  available: boolean;
  alphaAttr: number;
  evaluatedCount: number;
  variants: Array<{
    variantId: 'f64MulRoundedOnce' | 'f32MulCanonicalZero' | 'f32MulSignedZero';
    label: string;
    evaluatedCount: number;
    gpuExactBitMatches: number;
    wasmExactBitMatches: number;
    maxAbsDiffFromGpu: number;
    maxAbsDiffFromWasm: number;
  }>;
  samples: Array<{
    flatIdx: number;
    inputVal: number;
    gpu: number;
    wasm: number;
    gpuHex: string;
    wasmHex: string;
    f64MulRoundedOnce: number;
    f32MulCanonicalZero: number;
    f32MulSignedZero: number;
  }>;
}

export interface PostConv7ConvExtendedVariantReport {
  available: boolean;
  blockerReason: string | null;
  evaluatedCount: number;
  standardProbe: Conv7ReductionProbeReport;
  extraVariants: Array<{
    variantId: 'f32Vec4IC_KH_KW' | 'f32Fma_IC_KH_KW' | 'f32Fma_KH_KW_IC';
    label: string;
    evaluatedCount: number;
    gpuExactBitMatches: number;
    wasmExactBitMatches: number;
    maxAbsDiffFromGpu: number;
    meanAbsDiffFromGpu: number;
    maxAbsDiffFromWasm: number;
    meanAbsDiffFromWasm: number;
  }>;
}

export interface PostConv7IsolatedReplayResult {
  executed: boolean;
  blockerReason: string | null;
  targetNodeIdx: number;
  targetNodeName: string;
  targetOpType: string;
  targetOutputTensor: string;
  submodelByteLength: number | null;
  submodelSha256: string | null;
  capturedInputName: string;
  capturedInputSha256: string;
  capturedInputBitwiseEqual: boolean;
  initializerManifest: Array<{
    name: string;
    dims: number[];
    dataTypeName: string;
    protoSha256: string;
    valueSha256: string | null;
  }>;
  webgpuDispatchObserved: boolean;
  webgpuDispatchProfile: OrtVersionDiagnosticResponse | null;
  replayGpuVsReplayWasm: DetailedFloat32Comparison | null;
  replayGpuVsFullSuffixGpu: DetailedFloat32Comparison | null;
  replayWasmVsFullSuffixWasm: DetailedFloat32Comparison | null;
  reproducesFullSuffixGpuBits: boolean | null;
  reproducesFullSuffixWasmBits: boolean | null;
  reproducesFullSuffixDiff: boolean | null;
  float64Reference: SequentialFloat64RefResult | null;
  convReductionVariants: PostConv7ConvExtendedVariantReport | null;
  leakyReluVariants: LeakyReluVariantProbeReport | null;
  interpretationNote: string;
}

export interface PostConv7FrameTapReplayResult {
  frameLabel: 'F0102' | 'F0126';
  feedSha256: string;
  ok: boolean;
  aborted: boolean;
  abortReason: string | null;
  untappedFullFinalsComparison: DetailedFloat32Comparison;
  untappedFullSecondaryComparison: DetailedFloat32Comparison;
  conv7ParentComparison: {
    gpuSha256: string;
    wasmSha256: string;
    comparison: DetailedFloat32Comparison;
  };
  bypassCaptureDetails: Array<{
    name: string;
    dataType: string;
    dims: number[];
    byteLength: number;
    gpuSha256: string;
    wasmSha256: string;
    bitwiseEqual: boolean;
  }>;
  splitBaselineInvariance: {
    wasmSplitMatchesWasmFullBitExact: boolean;
    gpuSplitMatchesGpuFullBitExact: boolean;
    wasmPrimary: BitIdentityResult;
    wasmSecondary: BitIdentityResult;
    gpuPrimary: BitIdentityResult;
    gpuSecondary: BitIdentityResult;
    bothPassed: boolean;
  };
  gpuSuffixUntappedWasm7DispatchObserved: boolean;
  gpuSuffixUntappedWasm7Profile: OrtVersionDiagnosticResponse | null;
  gpuSuffixWasm7VsWasmSplitPrimary: DetailedFloat32Comparison;
  gpuSuffixWasm7VsWasmSplitSecondary: DetailedFloat32Comparison;
  tapInvarianceChecks: TapInvarianceCheckRecord[];
  paired89Tap: {
    executed: boolean;
    tappedModelSha256: string;
    tappedModelByteLength: number;
    tappedOutputs: string[];
    invariancePassed: boolean;
    node8: SuffixNodeTapObservation | null;
    node9: SuffixNodeTapObservation | null;
  };
  paired1011Tap: {
    executed: boolean;
    skippedReason: string | null;
    tappedModelSha256: string | null;
    tappedModelByteLength: number | null;
    tappedOutputs: string[];
    invariancePassed: boolean | null;
    node10: SuffixNodeTapObservation | null;
    node11: SuffixNodeTapObservation | null;
  };
  localizedFirstDifferingOp: {
    stage: 'node8-leakyrelu' | 'node9-conv' | 'node10-leakyrelu1' | 'node11-conv11' | 'none-through-node11';
    nodeIdx: number | null;
    nodeName: string | null;
    opType: string | null;
    outputTensor: string | null;
  };
  isolatedReplay: PostConv7IsolatedReplayResult | null;
}

export interface N1OffVerificationProof {
  normalizeSlot0ReshapeToTransposeRequested: boolean;
  n1ExplicitlyOff: boolean;
  n1ReplacementCount: number;
  verifiedOriginalSlot0ReshapeNodes: string[];
  detectedN1TransposeNodes: string[];
  canonicalM1ByteLength: number;
  canonicalM1Sha256: string;
  f0102FullGpuPreSessionModelSha256: string | null;
  f0102FullWasmPreSessionModelSha256: string | null;
  verified: boolean;
  failureReason: string | null;
}

export interface HitnetPostConv7TapReplayReport {
  ok: boolean;
  aborted?: boolean;
  abortReason?: string | null;
  pinnedOrtVersion: '1.30.0';
  runtimeAssetFamily: 'asyncify';
  graphOptimizationLevel: 'disabled';
  m1ByteLength: number;
  m1Sha256: string;
  n1OffProof: N1OffVerificationProof;
  chainSpec: VerifiedPostConv7ChainSpec;
  f0102: PostConv7FrameTapReplayResult;
  f0126: PostConv7FrameTapReplayResult | null;
  scientificScopeNote: string;
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

function buildNodeSpec(parsed: ParsedM1ModelFull, node: ParsedM1NodeFull): PostConv7TargetNodeSpec {
  const out = node.outputs[0];
  if (!out) {
    throw new Error(`Node #${node.idx} "${node.name}" has no output tensor`);
  }
  const meta = parsed.staticMeta.tensorMeta[out];
  if (!meta || meta.dims.some((d) => !Number.isInteger(d) || d <= 0)) {
    throw new Error(`Node #${node.idx} output "${out}" has missing/dynamic static metadata`);
  }
  const dynamicInputs: string[] = [];
  const initializerNames: string[] = [];
  for (const inp of node.inputs) {
    if (!inp) continue;
    if (parsed.initializerNames.has(inp)) initializerNames.push(inp);
    else dynamicInputs.push(inp);
  }
  return {
    nodeIdx: node.idx,
    nodeName: node.name,
    opType: node.opType,
    outputTensor: out,
    outputDims: [...meta.dims],
    attrsSummary: [...node.attrsSummary],
    dynamicInputs,
    initializerNames,
  };
}

export async function resolveVerifiedPostConv7ChainSpec(
  parsed: ParsedM1ModelFull
): Promise<VerifiedPostConv7ChainSpec> {
  const suffixSpec = await buildVerifiedSuffixAfterConv7(parsed);
  const conv7Out = suffixSpec.primaryCutInput.name;

  const consumersOfConv7 = parsed.nodes.filter(
    (n) => n.idx > suffixSpec.cutNodeIdx && n.inputs.includes(conv7Out)
  );
  const node8 = consumersOfConv7.find((n) => n.opType === 'LeakyRelu');
  if (!node8) {
    throw new Error(
      `Could not verify post-#${suffixSpec.cutNodeIdx} LeakyRelu consumer of "${conv7Out}" in M1 graph`
    );
  }
  const node8Spec = buildNodeSpec(parsed, node8);
  if (!node8.name.includes('fe_shared/LeakyRelu') && !node8Spec.outputTensor.includes('fe_shared/LeakyRelu')) {
    throw new Error(
      `Unexpected #8 LeakyRelu consumer identity: node=#${node8.idx} name="${node8.name}" out="${node8Spec.outputTensor}"`
    );
  }

  const consumersOfNode8 = parsed.nodes.filter(
    (n) => n.idx > node8.idx && n.inputs.includes(node8Spec.outputTensor)
  );
  const node9 = consumersOfNode8.find(
    (n) => n.opType === 'Conv' && (n.name.includes('conv_down_0') || n.outputs.some((o) => o.includes('conv_down_0')))
  );
  if (!node9) {
    throw new Error(
      `Could not verify post-#${node8.idx} stride-2 Conv (conv_down_0) consumer of "${node8Spec.outputTensor}"`
    );
  }
  const node9Spec = buildNodeSpec(parsed, node9);

  // Also resolve #10 LeakyRelu_1 and #11 Conv (conv_down_1_1) dynamically by graph edge (used ONLY if #8 & #9 are both bit-exact)
  const node10 =
    parsed.nodes.find((n) => n.idx > node9.idx && n.opType === 'LeakyRelu' && n.inputs.includes(node9Spec.outputTensor)) ??
    null;
  const node10Spec = node10 ? buildNodeSpec(parsed, node10) : null;
  const node11 =
    node10Spec
      ? parsed.nodes.find(
          (n) =>
            n.idx > node10Spec.nodeIdx &&
            n.opType === 'Conv' &&
            n.inputs.includes(node10Spec.outputTensor) &&
            (n.name.includes('conv_down_1_1') || n.outputs.some((o) => o.includes('conv_down_1_1')))
        ) ?? null
      : null;
  const node11Spec = node11 ? buildNodeSpec(parsed, node11) : null;

  return {
    conv7Idx: suffixSpec.cutNodeIdx,
    conv7OutputTensor: conv7Out,
    node8LeakyRelu: node8Spec,
    node9ConvDown0: node9Spec,
    node10LeakyRelu1: node10Spec,
    node11ConvDown11: node11Spec,
    bypassTile2Nodes: suffixSpec.bypassCutInputs.map((b) => ({
      nodeIdx: b.fromNodeIdx ?? -1,
      nodeName: parsed.nodes[b.fromNodeIdx ?? -1]?.name ?? b.name,
      outputTensor: b.name,
      dims: [...b.dims],
      dataTypeName: b.dataTypeName,
    })),
    suffixSummary: {
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
    },
  };
}

export function probeLeakyReluNumericVariants(
  node: ParsedM1NodeFull,
  inputF32: Float32Array,
  gpuOutF32: Float32Array,
  wasmOutF32: Float32Array,
  opts?: { maxSamples?: number }
): LeakyReluVariantProbeReport {
  const alphaAttr = node.attrs.find((a) => a.name === 'alpha')?.floatVal ?? 0.01;
  const alphaF32 = Math.fround(alphaAttr);
  const len = inputF32.length;
  const gU32 = new Uint32Array(gpuOutF32.buffer, gpuOutF32.byteOffset, len);
  const wU32 = new Uint32Array(wasmOutF32.buffer, wasmOutF32.byteOffset, len);

  const picked = new Set<number>();
  const topDiv: Array<{ idx: number; d: number }> = [];
  for (let i = 0; i < len; i++) {
    if (gU32[i] !== wU32[i]) {
      const d = Math.abs(gpuOutF32[i] - wasmOutF32[i]);
      if (picked.size < 8) picked.add(i);
      if (topDiv.length < 64 || d > topDiv[topDiv.length - 1].d) {
        topDiv.push({ idx: i, d });
        topDiv.sort((a, b) => b.d - a.d);
        if (topDiv.length > 64) topDiv.pop();
      }
    }
  }
  for (const t of topDiv) picked.add(t.idx);
  const maxSamples = Math.max(16, Math.min(256, opts?.maxSamples ?? 128));
  const step = Math.max(1, Math.floor(len / Math.max(1, maxSamples - picked.size)));
  for (let i = 0; i < len && picked.size < maxSamples; i += step) picked.add(i);
  const evalIndices = Array.from(picked).sort((a, b) => a - b);

  const f32Scratch = new Float32Array(1);
  const u32Scratch = new Uint32Array(f32Scratch.buffer);
  const toBits = (v: number) => {
    f32Scratch[0] = v;
    return u32Scratch[0];
  };
  const hexU32 = (u: number) => `0x${(u >>> 0).toString(16).padStart(8, '0')}`;

  const acc = [
    { id: 'f64MulRoundedOnce' as const, label: 'Float64 multiply -> single f32 round', gBit: 0, wBit: 0, gMax: 0, wMax: 0 },
    { id: 'f32MulCanonicalZero' as const, label: 'Float32 multiply with +0.0 canonicalization', gBit: 0, wBit: 0, gMax: 0, wMax: 0 },
    { id: 'f32MulSignedZero' as const, label: 'Float32 multiply preserving IEEE-754 sign on zero', gBit: 0, wBit: 0, gMax: 0, wMax: 0 },
  ];
  const samples: LeakyReluVariantProbeReport['samples'] = [];

  for (const idx of evalIndices) {
    const x = inputF32[idx];
    const vF64 = x >= 0 ? Math.fround(x) : Math.fround(Number(x) * alphaAttr);
    const vCan = x > 0 ? Math.fround(x) : x === 0 ? 0 : Math.fround(x * alphaF32);
    const vSig = x >= 0 ? Math.fround(x) : Math.fround(x * alphaF32);
    const vals = [vF64, vCan, vSig];

    for (let k = 0; k < 3; k++) {
      const b = toBits(vals[k]);
      if (b === gU32[idx]) acc[k].gBit++;
      if (b === wU32[idx]) acc[k].wBit++;
      acc[k].gMax = Math.max(acc[k].gMax, Math.abs(gpuOutF32[idx] - vals[k]));
      acc[k].wMax = Math.max(acc[k].wMax, Math.abs(wasmOutF32[idx] - vals[k]));
    }
    if (samples.length < 10 && (gU32[idx] !== wU32[idx] || samples.length < 3)) {
      samples.push({
        flatIdx: idx,
        inputVal: x,
        gpu: gpuOutF32[idx],
        wasm: wasmOutF32[idx],
        gpuHex: hexU32(gU32[idx]),
        wasmHex: hexU32(wU32[idx]),
        f64MulRoundedOnce: vF64,
        f32MulCanonicalZero: vCan,
        f32MulSignedZero: vSig,
      });
    }
  }

  return {
    available: true,
    alphaAttr,
    evaluatedCount: evalIndices.length,
    variants: acc.map((a) => ({
      variantId: a.id,
      label: a.label,
      evaluatedCount: evalIndices.length,
      gpuExactBitMatches: a.gBit,
      wasmExactBitMatches: a.wBit,
      maxAbsDiffFromGpu: a.gMax,
      maxAbsDiffFromWasm: a.wMax,
    })),
    samples,
  };
}

export function probeConvOpReductionVariantsExtended(
  parsed: ParsedM1ModelFull,
  convNodeIdx: number,
  inputF32: Float32Array,
  inputDims: readonly number[],
  gpuOutF32: Float32Array,
  wasmOutF32: Float32Array,
  outDims: readonly number[],
  opts?: { maxSamples?: number }
): PostConv7ConvExtendedVariantReport {
  const maxSamples = Math.max(16, Math.min(512, opts?.maxSamples ?? 128));
  const standardProbe = probeConv7ReductionVariants(
    parsed,
    convNodeIdx,
    inputF32,
    inputDims,
    gpuOutF32,
    wasmOutF32,
    outDims,
    { maxSamples }
  );

  const node = parsed.nodes[convNodeIdx];
  const wInit = node?.inputs[1] ? parsed.initializersByName.get(node.inputs[1]) : undefined;
  const bInit = node?.inputs[2] ? parsed.initializersByName.get(node.inputs[2]) : undefined;
  if (!standardProbe.available || !node || !wInit?.float32Values) {
    return {
      available: false,
      blockerReason: standardProbe.blockerReason ?? 'Conv variant probe unavailable',
      evaluatedCount: 0,
      standardProbe,
      extraVariants: [],
    };
  }

  const [, C, H, W] = inputDims;
  const [M, CPerG, kH, kW] = wInit.dims;
  const [, oM, oH, oW] = outDims;
  const group = node.attrs.find((a) => a.name === 'group')?.intVal ?? 1;
  const strides = node.attrs.find((a) => a.name === 'strides')?.intsVal ?? [1, 1];
  const dilations = node.attrs.find((a) => a.name === 'dilations')?.intsVal ?? [1, 1];
  const pads = node.attrs.find((a) => a.name === 'pads')?.intsVal ?? [0, 0, 0, 0];
  const padTop = pads[0] ?? 0;
  const padLeft = pads[1] ?? 0;
  const strideH = strides[0] ?? 1;
  const strideW = strides[1] ?? 1;
  const dilH = dilations[0] ?? 1;
  const dilW = dilations[1] ?? 1;
  const wData = wInit.float32Values;
  const bData = bInit?.float32Values ?? null;
  const gSize = M / group;
  const elemCount = gpuOutF32.length;

  const picked = new Set<number>();
  const topDiv: Array<{ idx: number; d: number }> = [];
  for (let i = 0; i < elemCount; i++) {
    const d = Math.abs(gpuOutF32[i] - wasmOutF32[i]);
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

  const gU32 = new Uint32Array(gpuOutF32.buffer, gpuOutF32.byteOffset, elemCount);
  const wU32 = new Uint32Array(wasmOutF32.buffer, wasmOutF32.byteOffset, elemCount);
  const f32Scratch = new Float32Array(1);
  const u32Scratch = new Uint32Array(f32Scratch.buffer);
  const toBits = (v: number): number => {
    f32Scratch[0] = v;
    return u32Scratch[0];
  };
  const fma32 = (a: number, b: number, c: number): number => Math.fround(Number(a) * Number(b) + Number(c));

  const mkAcc = () => ({ gBit: 0, wBit: 0, gMax: 0, gSum: 0, wMax: 0, wSum: 0 });
  const accVec4 = mkAcc();
  const accFmaIJK = mkAcc();
  const accFmaKJI = mkAcc();

  const upd = (acc: ReturnType<typeof mkAcc>, v: number, idx: number) => {
    const bits = toBits(v);
    if (bits === gU32[idx]) acc.gBit++;
    if (bits === wU32[idx]) acc.wBit++;
    const dg = Math.abs(gpuOutF32[idx] - v);
    const dw = Math.abs(wasmOutF32[idx] - v);
    if (dg > acc.gMax) acc.gMax = dg;
    if (dw > acc.wMax) acc.wMax = dw;
    acc.gSum += dg;
    acc.wSum += dw;
  };

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
    const bias = bData ? Math.fround(bData[m]) : 0.0;

    // 1) FMA IC->KH->KW
    let fmaIjk = 0.0;
    for (let ic = 0; ic < CPerG; ic++) {
      const xChanOff = xBase + (cStart + ic) * H * W;
      const wChanOff = wBase + ic * kH * kW;
      for (let kh = 0; kh < kH; kh++) {
        const ih = oh * strideH + kh * dilH - padTop;
        const wRowOff = wChanOff + kh * kW;
        for (let kw = 0; kw < kW; kw++) {
          const iw = ow * strideW + kw * dilW - padLeft;
          const xv = ih >= 0 && ih < H && iw >= 0 && iw < W ? inputF32[xChanOff + ih * W + iw] : 0;
          const wv = wData[wRowOff + kw];
          fmaIjk = fma32(xv, wv, fmaIjk);
        }
      }
    }
    upd(accFmaIJK, Math.fround(fmaIjk + bias), idx);

    // 2) FMA KH->KW->IC + 3) Vec4 dot4 over IC groups (kh->kw->ic4 chunked sum)
    let fmaKji = 0.0;
    let vec4Sum = 0.0;
    for (let kh = 0; kh < kH; kh++) {
      const ih = oh * strideH + kh * dilH - padTop;
      for (let kw = 0; kw < kW; kw++) {
        const iw = ow * strideW + kw * dilW - padLeft;
        for (let ic = 0; ic < CPerG; ic += 4) {
          let laneSum = 0.0;
          const icEnd = Math.min(CPerG, ic + 4);
          for (let j = ic; j < icEnd; j++) {
            const xv =
              ih >= 0 && ih < H && iw >= 0 && iw < W
                ? inputF32[xBase + (cStart + j) * H * W + ih * W + iw]
                : 0;
            const wv = wData[wBase + (j * kH + kh) * kW + kw];
            fmaKji = fma32(xv, wv, fmaKji);
            laneSum = Math.fround(laneSum + Math.fround(xv * wv));
          }
          vec4Sum = Math.fround(vec4Sum + laneSum);
        }
      }
    }
    upd(accFmaKJI, Math.fround(fmaKji + bias), idx);
    upd(accVec4, Math.fround(vec4Sum + bias), idx);
  }

  const nEval = evalIndices.length;
  const fin = (
    variantId: 'f32Vec4IC_KH_KW' | 'f32Fma_IC_KH_KW' | 'f32Fma_KH_KW_IC',
    label: string,
    a: ReturnType<typeof mkAcc>
  ) => ({
    variantId,
    label,
    evaluatedCount: nEval,
    gpuExactBitMatches: a.gBit,
    wasmExactBitMatches: a.wBit,
    maxAbsDiffFromGpu: a.gMax,
    meanAbsDiffFromGpu: nEval > 0 ? a.gSum / nEval : 0,
    maxAbsDiffFromWasm: a.wMax,
    meanAbsDiffFromWasm: nEval > 0 ? a.wSum / nEval : 0,
  });

  return {
    available: true,
    blockerReason: null,
    evaluatedCount: nEval,
    standardProbe,
    extraVariants: [
      fin('f32Vec4IC_KH_KW', 'Grouped 4-channel f32 sep mul/add [kh][kw][ic4] + bias', accVec4),
      fin('f32Fma_IC_KH_KW', 'Fused FMA f32 [ic][kh][kw] + bias', accFmaIJK),
      fin('f32Fma_KH_KW_IC', 'Fused FMA f32 [kh][kw][ic] + bias', accFmaKJI),
    ],
  };
}

export function verifyCanonicalM1NoN1Replacements(
  parsed: ParsedM1ModelFull,
  m1Bytes: Uint8Array,
  m1Sha256: string,
  normalizeSlot0ReshapeToTransposeRequested = false
): N1OffVerificationProof {
  const expectedSlot0Names = ['Transpose__908', 'Transpose__1006'];
  const verifiedOriginalSlot0ReshapeNodes: string[] = [];
  const detectedN1TransposeNodes: string[] = [];

  for (const targetName of expectedSlot0Names) {
    const node = parsed.nodes.find((n) => n.name === targetName);
    if (!node) continue;
    if (node.opType === 'Reshape') {
      verifiedOriginalSlot0ReshapeNodes.push(`${node.name}@#${node.idx}`);
    } else if (node.opType === 'Transpose') {
      detectedN1TransposeNodes.push(`${node.name}@#${node.idx}(Transpose)`);
    } else {
      detectedN1TransposeNodes.push(`${node.name}@#${node.idx}(${node.opType})`);
    }
  }

  for (const n of parsed.nodes) {
    if (
      n.opType === 'Transpose' &&
      (n.name.includes('__webgpu_n1') || n.outputs.some((o) => o.includes('__webgpu_n1')))
    ) {
      detectedN1TransposeNodes.push(`${n.name}@#${n.idx}`);
    }
  }

  const n1ReplacementCount = detectedN1TransposeNodes.length;
  const n1ExplicitlyOff = !normalizeSlot0ReshapeToTransposeRequested;
  const shaOk =
    m1Bytes.byteLength === HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength &&
    m1Sha256 === HITNET_M1_FULL_GRAPH_SHA256;
  const slot0ReshapesIntact =
    verifiedOriginalSlot0ReshapeNodes.length === expectedSlot0Names.length &&
    n1ReplacementCount === 0;
  const verified = n1ExplicitlyOff && shaOk && slot0ReshapesIntact;

  let failureReason: string | null = null;
  if (!n1ExplicitlyOff) {
    failureReason = 'N1 (normalizeSlot0ReshapeToTranspose) was requested ON; must be explicitly OFF.';
  } else if (!shaOk) {
    failureReason = `Canonical M1 SHA/length mismatch: bytes=${m1Bytes.byteLength}, sha=${m1Sha256}`;
  } else if (!slot0ReshapesIntact) {
    failureReason = `N1 replacement guard failed: slot0Reshapes=${JSON.stringify(verifiedOriginalSlot0ReshapeNodes)}, detectedN1=${JSON.stringify(detectedN1TransposeNodes)}`;
  }

  return {
    normalizeSlot0ReshapeToTransposeRequested,
    n1ExplicitlyOff,
    n1ReplacementCount,
    verifiedOriginalSlot0ReshapeNodes,
    detectedN1TransposeNodes,
    canonicalM1ByteLength: m1Bytes.byteLength,
    canonicalM1Sha256: m1Sha256,
    f0102FullGpuPreSessionModelSha256: null,
    f0102FullWasmPreSessionModelSha256: null,
    verified,
    failureReason,
  };
}

export async function runHitnetPostConv7TapReplayDiagnostic(params: {
  m1Bytes: Uint8Array;
  f0102FeedFloat32: Float32Array;
  f0126FeedFloat32?: Float32Array | null;
  confirmF0126?: boolean;
  normalizeSlot0ReshapeToTranspose?: false;
  runWorker?: ResidualDiagnosticWorkerRunner;
  disableCpuEpFallback?: boolean;
  isDev?: boolean;
  onLog?: (line: string) => void;
}): Promise<HitnetPostConv7TapReplayReport> {
  assertDevFixtureEnabled(params.isDev);
  const log = params.onLog ?? (() => {});
  const runWorker = params.runWorker ?? createBrowserOrt130DiagnosticRunner();
  const disableCpuEpFallback = Boolean(params.disableCpuEpFallback);
  const normalizeSlot0ReshapeToTransposeRequested = Boolean(
    (params as { normalizeSlot0ReshapeToTranspose?: boolean }).normalizeSlot0ReshapeToTranspose
  );
  if (normalizeSlot0ReshapeToTransposeRequested) {
    throw new Error(
      'N1-Slot0Transpose (normalizeSlot0ReshapeToTranspose) MUST be explicitly OFF (false) for post-#7 diagnostic attribution.'
    );
  }

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
  const n1OffProof = verifyCanonicalM1NoN1Replacements(
    parsedModel,
    params.m1Bytes,
    m1Sha,
    normalizeSlot0ReshapeToTransposeRequested
  );
  if (!n1OffProof.verified) {
    throw new Error(n1OffProof.failureReason ?? 'N1-off / canonical M1 verification failed');
  }
  const suffixSpec = await buildVerifiedSuffixAfterConv7(parsedModel);
  const chainSpec = await resolveVerifiedPostConv7ChainSpec(parsedModel);
  const asyncifyProfile = HITNET_ORT_130_FAMILY_PROFILES.asyncify;
  const [pName, sName] = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;

  log(
    `=== [DEV-ONLY] HITNet Post-#7 (#8/#9) Paired Tap & Isolated Replay Diagnostic ===\n` +
      `Runtime: onnxruntime-web@1.30.0/webgpu (non-JSEP ${asyncifyProfile.entryFile} + ${asyncifyProfile.wasmFile}) | opt=disabled\n` +
      `Canonical M1 (pre-session.create baseline target): bytes=${params.m1Bytes.byteLength} sha256=${m1Sha}\n` +
      `N1 Guard: explicitOff=${n1OffProof.n1ExplicitlyOff} replacements=${n1OffProof.n1ReplacementCount} slot0Reshapes=${JSON.stringify(n1OffProof.verifiedOriginalSlot0ReshapeNodes)}\n` +
      `Verified Post-#${chainSpec.conv7Idx} Suffix: bytes=${suffixSpec.suffixByteLength} sha256=${suffixSpec.suffixSha256} nodes=${suffixSpec.suffixNodeCount} (#${suffixSpec.suffixNodeRange.firstOriginalIdx}..#${suffixSpec.suffixNodeRange.lastOriginalIdx})\n` +
      `Verified Target Chain:\n` +
      `  [#${chainSpec.node8LeakyRelu.nodeIdx} ${chainSpec.node8LeakyRelu.opType}] "${chainSpec.node8LeakyRelu.nodeName}" -> "${chainSpec.node8LeakyRelu.outputTensor}" ${JSON.stringify(chainSpec.node8LeakyRelu.outputDims)} attrs=${JSON.stringify(chainSpec.node8LeakyRelu.attrsSummary)}\n` +
      `  [#${chainSpec.node9ConvDown0.nodeIdx} ${chainSpec.node9ConvDown0.opType}] "${chainSpec.node9ConvDown0.nodeName}" -> "${chainSpec.node9ConvDown0.outputTensor}" ${JSON.stringify(chainSpec.node9ConvDown0.outputDims)} attrs=${JSON.stringify(chainSpec.node9ConvDown0.attrsSummary)}\n` +
      (chainSpec.node10LeakyRelu1 && chainSpec.node11ConvDown11
        ? `  [Fallback ONLY if #8&#9 both exact] [#${chainSpec.node10LeakyRelu1.nodeIdx} ${chainSpec.node10LeakyRelu1.opType}] -> [#${chainSpec.node11ConvDown11.nodeIdx} ${chainSpec.node11ConvDown11.opType}] "${chainSpec.node11ConvDown11.outputTensor}"\n`
        : '') +
      `  Bypass Tile_2 (${chainSpec.bypassTile2Nodes.length}): ${JSON.stringify(chainSpec.bypassTile2Nodes.map((b) => ({ idx: b.nodeIdx, name: b.outputTensor, dtype: b.dataTypeName, dims: b.dims })))}`
  );

  const assertNonJsepSuccess = (
    label: string,
    rawRes: OrtVersionDiagnosticResponse,
    expectedProvider: 'webgpu' | 'wasm',
    expectedModelSha: string,
    expectedModelBytes: number,
    expectedFeedSha?: string
  ): OrtVersionDiagnosticSuccess => {
    if (!rawRes.ok) {
      const f = rawRes as OrtVersionDiagnosticFailure;
      throw new Error(`${label} FAILED at stage="${f.stage}": ${f.message}`);
    }
    if (
      rawRes.pinnedOrtVersion !== '1.30.0' ||
      rawRes.graphOptimizationLevel !== 'disabled' ||
      rawRes.provider !== expectedProvider ||
      rawRes.runtimeAssetFamily !== 'asyncify' ||
      rawRes.foreignOrtResources.length > 0 ||
      rawRes.loadedOrtResources.some((u) => u.includes('.jsep.') || u.includes('ort.bundle.'))
    ) {
      throw new Error(
        `${label} violated ORT 1.30.0 non-JSEP/opt=disabled guard: version=${rawRes.pinnedOrtVersion}, opt=${rawRes.graphOptimizationLevel}, provider=${rawRes.provider}, family=${rawRes.runtimeAssetFamily}, foreign=${JSON.stringify(rawRes.foreignOrtResources)}, loaded=${JSON.stringify(rawRes.loadedOrtResources)}`
      );
    }
    if (rawRes.modelSha256 !== expectedModelSha || rawRes.modelByteLength !== expectedModelBytes) {
      throw new Error(
        `${label} pre-session model mismatch: got=${rawRes.modelSha256}(${rawRes.modelByteLength}B), expected=${expectedModelSha}(${expectedModelBytes}B)`
      );
    }
    if (expectedFeedSha && rawRes.feedSha256 !== expectedFeedSha) {
      throw new Error(
        `${label} feed SHA mismatch: got=${rawRes.feedSha256}, expected=${expectedFeedSha}`
      );
    }
    return rawRes;
  };

  const copyRawOutputBytes = (out: OrtVersionDiagnosticOutput): Uint8Array =>
    new Uint8Array(out.buffer).slice();

  const checkInvariancePair = (
    label: string,
    modelByteLength: number,
    modelSha256: string,
    referenceLabel: string,
    gpuOutputs: OrtVersionDiagnosticOutput[],
    gpuRefByName: Record<string, { data: Float32Array }>,
    wasmOutputs: OrtVersionDiagnosticOutput[],
    wasmRefByName: Record<string, { data: Float32Array }>
  ): TapInvarianceCheckRecord => {
    const gv = validateHitnetFinalOutputs(gpuOutputs);
    const wv = validateHitnetFinalOutputs(wasmOutputs);
    if (!gv.ok || !wv.ok) {
      throw new Error(`${label} final output validation failed: gpu=${gv.error ?? 'ok'}, wasm=${wv.error ?? 'ok'}`);
    }
    const gpuPrimaryBitIdentical = checkFloat32BitIdentity(gv.byName[pName].data, gpuRefByName[pName].data);
    const gpuSecondaryBitIdentical = checkFloat32BitIdentity(gv.byName[sName].data, gpuRefByName[sName].data);
    const wasmPrimaryBitIdentical = checkFloat32BitIdentity(wv.byName[pName].data, wasmRefByName[pName].data);
    const wasmSecondaryBitIdentical = checkFloat32BitIdentity(wv.byName[sName].data, wasmRefByName[sName].data);
    const bothPassed =
      gpuPrimaryBitIdentical.bitIdentical &&
      gpuSecondaryBitIdentical.bitIdentical &&
      wasmPrimaryBitIdentical.bitIdentical &&
      wasmSecondaryBitIdentical.bitIdentical;

    return {
      label,
      modelByteLength,
      modelSha256,
      referenceLabel,
      gpuPrimaryBitIdentical,
      gpuSecondaryBitIdentical,
      wasmPrimaryBitIdentical,
      wasmSecondaryBitIdentical,
      bothPassed,
    };
  };

  const runFrame = async (
    frameLabel: 'F0102' | 'F0126',
    feedFloat32: Float32Array,
    feedSha: string
  ): Promise<PostConv7FrameTapReplayResult> => {
    log(`\n--- [${frameLabel}] Step 1: Full-M1 Baselines + Capture #7 Conv & Bypass #2/#3 ---`);
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
    const gpuFull = assertNonJsepSuccess(`${frameLabel} M1 Full WebGPU`, gpuFullRaw, 'webgpu', m1Sha, params.m1Bytes.byteLength, feedSha);
    const wasmFull = assertNonJsepSuccess(`${frameLabel} M1 Full WASM`, wasmFullRaw, 'wasm', m1Sha, params.m1Bytes.byteLength, feedSha);
    if (frameLabel === 'F0102') {
      n1OffProof.f0102FullGpuPreSessionModelSha256 = gpuFull.modelSha256 ?? null;
      n1OffProof.f0102FullWasmPreSessionModelSha256 = wasmFull.modelSha256 ?? null;
      log(
        `  [Pre-session.create Canonical M1 Verified] gpu=${gpuFull.modelSha256} wasm=${wasmFull.modelSha256} match=${gpuFull.modelSha256 === m1Sha && wasmFull.modelSha256 === m1Sha} N1=${n1OffProof.n1ExplicitlyOff ? 'OFF(0 replacements)' : 'ON'}`
      );
    }
    const gpuFullVal = validateHitnetFinalOutputs(gpuFull.outputs);
    const wasmFullVal = validateHitnetFinalOutputs(wasmFull.outputs);
    if (!gpuFullVal.ok || !wasmFullVal.ok) {
      throw new Error(`${frameLabel} Full-M1 invalid outputs: gpu=${gpuFullVal.error}, wasm=${wasmFullVal.error}`);
    }

    const untappedFullFinalsComparison = compareFloat32TensorsDetailed(
      gpuFullVal.byName[pName].data,
      wasmFullVal.byName[pName].data,
      [1, 240, 320, 1]
    );
    const untappedFullSecondaryComparison = compareFloat32TensorsDetailed(
      gpuFullVal.byName[sName].data,
      wasmFullVal.byName[sName].data,
      [1, 240, 320, 1]
    );

    const tapInvarianceChecks: TapInvarianceCheckRecord[] = [];

    // Tap #7 on full M1
    const conv7TapModel = appendOnnxGraphOutputs(params.m1Bytes, [suffixSpec.primaryCutInput.name], {
      maxExtraOutputs: 1,
    });
    const conv7TapSha = await sha256HexBytes(conv7TapModel.bytes);
    const conv7Gpu = assertNonJsepSuccess(
      `${frameLabel} M1 Tap #7 WebGPU`,
      await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: conv7TapModel.bytes.slice().buffer,
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
      'webgpu',
      conv7TapSha,
      conv7TapModel.bytes.byteLength,
      feedSha
    );
    const conv7Wasm = assertNonJsepSuccess(
      `${frameLabel} M1 Tap #7 WASM`,
      await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: conv7TapModel.bytes.slice().buffer,
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
      'wasm',
      conv7TapSha,
      conv7TapModel.bytes.byteLength,
      feedSha
    );
    const conv7Inv = checkInvariancePair(
      'full-m1-tap-conv7',
      conv7TapModel.bytes.byteLength,
      conv7TapSha,
      'full-m1-untapped',
      conv7Gpu.outputs,
      gpuFullVal.byName,
      conv7Wasm.outputs,
      wasmFullVal.byName
    );
    tapInvarianceChecks.push(conv7Inv);

    const gConv7Out = conv7Gpu.outputs.find((o) => o.name === suffixSpec.primaryCutInput.name);
    const wConv7Out = conv7Wasm.outputs.find((o) => o.name === suffixSpec.primaryCutInput.name);
    if (!gConv7Out || !wConv7Out) {
      throw new Error(`${frameLabel} missing captured #7 Conv output "${suffixSpec.primaryCutInput.name}"`);
    }
    const gpuConv7F32 = new Float32Array(gConv7Out.buffer).slice();
    const wasmConv7F32 = new Float32Array(wConv7Out.buffer).slice();
    const gpuConv7Sha = await sha256HexBytes(copyRawOutputBytes(gConv7Out));
    const wasmConv7Sha = await sha256HexBytes(copyRawOutputBytes(wConv7Out));
    const conv7Comparison = compareFloat32TensorsDetailed(
      gpuConv7F32,
      wasmConv7F32,
      suffixSpec.primaryCutInput.dims
    );

    // Tap bypass #2 & #3 on full M1
    const bypassTapNames = suffixSpec.bypassCutInputs.map((b) => b.name);
    const bypassCaptureDetails: PostConv7FrameTapReplayResult['bypassCaptureDetails'] = [];
    const bypassFeedBuffers: Array<{
      name: string;
      dims: number[];
      dataType: 'int32' | 'int64' | 'float32' | 'uint8';
      rawBytes: Uint8Array;
      sha256: string;
    }> = [];

    for (let i = 0; i < suffixSpec.bypassCutInputs.length; i += 2) {
      const batch = suffixSpec.bypassCutInputs.slice(i, i + 2);
      const bModel = appendOnnxGraphOutputs(params.m1Bytes, batch.map((b) => b.name), { maxExtraOutputs: 2 });
      const bSha = await sha256HexBytes(bModel.bytes);
      const bGpu = assertNonJsepSuccess(
        `${frameLabel} M1 Tap Bypass WebGPU`,
        await runWorker({
          expectedOrtVersion: '1.30.0',
          runtimeAssetFamily: 'asyncify',
          modelBuffer: bModel.bytes.slice().buffer,
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
        'webgpu',
        bSha,
        bModel.bytes.byteLength,
        feedSha
      );
      const bWasm = assertNonJsepSuccess(
        `${frameLabel} M1 Tap Bypass WASM`,
        await runWorker({
          expectedOrtVersion: '1.30.0',
          runtimeAssetFamily: 'asyncify',
          modelBuffer: bModel.bytes.slice().buffer,
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
        'wasm',
        bSha,
        bModel.bytes.byteLength,
        feedSha
      );
      const bInv = checkInvariancePair(
        `full-m1-tap-bypass-${i / 2 + 1}`,
        bModel.bytes.byteLength,
        bSha,
        'full-m1-untapped',
        bGpu.outputs,
        gpuFullVal.byName,
        bWasm.outputs,
        wasmFullVal.byName
      );
      tapInvarianceChecks.push(bInv);

      for (const spec of batch) {
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

    const bypassAllBitEqual = bypassCaptureDetails.every((d) => d.bitwiseEqual);
    if (!conv7Inv.bothPassed || !tapInvarianceChecks.every((t) => t.bothPassed) || !bypassAllBitEqual) {
      const reason = !bypassAllBitEqual
        ? `Bypass Tile_2 tensors (${bypassTapNames.join(', ')}) are not bitwise equal between WebGPU & WASM`
        : `Full-M1 tap perturbed final disparity outputs before suffix execution`;
      log(`  [ABORT] ${reason}`);
      return {
        frameLabel,
        feedSha256: feedSha,
        ok: false,
        aborted: true,
        abortReason: reason,
        untappedFullFinalsComparison,
        untappedFullSecondaryComparison,
        conv7ParentComparison: { gpuSha256: gpuConv7Sha, wasmSha256: wasmConv7Sha, comparison: conv7Comparison },
        bypassCaptureDetails,
        splitBaselineInvariance: {
          wasmSplitMatchesWasmFullBitExact: false,
          gpuSplitMatchesGpuFullBitExact: false,
          wasmPrimary: { bitIdentical: false, lengthMatch: true, mismatchCount: -1, firstMismatchIndex: null, maxAbsDiff: 0 },
          wasmSecondary: { bitIdentical: false, lengthMatch: true, mismatchCount: -1, firstMismatchIndex: null, maxAbsDiff: 0 },
          gpuPrimary: { bitIdentical: false, lengthMatch: true, mismatchCount: -1, firstMismatchIndex: null, maxAbsDiff: 0 },
          gpuSecondary: { bitIdentical: false, lengthMatch: true, mismatchCount: -1, firstMismatchIndex: null, maxAbsDiff: 0 },
          bothPassed: false,
        },
        gpuSuffixUntappedWasm7DispatchObserved: false,
        gpuSuffixUntappedWasm7Profile: null,
        gpuSuffixWasm7VsWasmSplitPrimary: untappedFullFinalsComparison,
        gpuSuffixWasm7VsWasmSplitSecondary: untappedFullSecondaryComparison,
        tapInvarianceChecks,
        paired89Tap: {
          executed: false,
          tappedModelSha256: '',
          tappedModelByteLength: 0,
          tappedOutputs: [],
          invariancePassed: false,
          node8: null,
          node9: null,
        },
        paired1011Tap: {
          executed: false,
          skippedReason: reason,
          tappedModelSha256: null,
          tappedModelByteLength: null,
          tappedOutputs: [],
          invariancePassed: null,
          node10: null,
          node11: null,
        },
        localizedFirstDifferingOp: {
          stage: 'none-through-node11',
          nodeIdx: null,
          nodeName: null,
          opType: null,
          outputTensor: null,
        },
        isolatedReplay: null,
      };
    }

    const buildExtraFeeds = () =>
      bypassFeedBuffers.map((b) => ({
        name: b.name,
        dims: [...b.dims],
        buffer: b.rawBytes.slice().buffer as ArrayBuffer,
        dataType: b.dataType,
      }));

    log(
      `\n--- [${frameLabel}] Step 2: Existing Split Baseline Invariance + Untapped Suffix (Common wasm#7 + Bypass #2/#3) ---`
    );

    const runSuffixModel = async (
      armLabel: string,
      modelBytes: Uint8Array,
      modelSha: string,
      provider: 'webgpu' | 'wasm',
      conv7Data: Float32Array,
      expectedConv7Sha: string,
      opts?: { enableWebGpuProfiling?: boolean }
    ) => {
      const raw = await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: modelBytes.slice().buffer,
        feedBuffer: conv7Data.slice().buffer,
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
        provider,
        modelSha,
        modelBytes.byteLength,
        expectedConv7Sha
      );
      const val = validateHitnetFinalOutputs(okRes.outputs);
      if (!val.ok) {
        throw new Error(`${armLabel} final output validation failed: ${val.error}`);
      }
      return { res: okRes, val };
    };

    const wasmSplitUntapped = await runSuffixModel(
      `${frameLabel} WASM-Suffix-Untapped(wasm#7)`,
      suffixSpec.suffixBytes,
      suffixSpec.suffixSha256,
      'wasm',
      wasmConv7F32,
      wasmConv7Sha
    );
    const gpuSplitControlUntapped = await runSuffixModel(
      `${frameLabel} GPU-Suffix-Control-Untapped(gpu#7)`,
      suffixSpec.suffixBytes,
      suffixSpec.suffixSha256,
      'webgpu',
      gpuConv7F32,
      gpuConv7Sha
    );
    const gpuSuffixUntappedWasm7 = await runSuffixModel(
      `${frameLabel} GPU-Suffix-Untapped(wasm#7)`,
      suffixSpec.suffixBytes,
      suffixSpec.suffixSha256,
      'webgpu',
      wasmConv7F32,
      wasmConv7Sha
    );
    const gpuSuffixUntappedWasm7Profile = (
      await runSuffixModel(
        `${frameLabel} GPU-Suffix-Untapped(wasm#7) Profile`,
        suffixSpec.suffixBytes,
        suffixSpec.suffixSha256,
        'webgpu',
        wasmConv7F32,
        wasmConv7Sha,
        { enableWebGpuProfiling: true }
      )
    ).res;

    const wasmPrimaryInv = checkFloat32BitIdentity(
      wasmSplitUntapped.val!.byName[pName].data,
      wasmFullVal.byName[pName].data
    );
    const wasmSecondaryInv = checkFloat32BitIdentity(
      wasmSplitUntapped.val!.byName[sName].data,
      wasmFullVal.byName[sName].data
    );
    const gpuPrimaryInv = checkFloat32BitIdentity(
      gpuSplitControlUntapped.val!.byName[pName].data,
      gpuFullVal.byName[pName].data
    );
    const gpuSecondaryInv = checkFloat32BitIdentity(
      gpuSplitControlUntapped.val!.byName[sName].data,
      gpuFullVal.byName[sName].data
    );
    const splitBaselineInvariance = {
      wasmSplitMatchesWasmFullBitExact: wasmPrimaryInv.bitIdentical && wasmSecondaryInv.bitIdentical,
      gpuSplitMatchesGpuFullBitExact: gpuPrimaryInv.bitIdentical && gpuSecondaryInv.bitIdentical,
      wasmPrimary: wasmPrimaryInv,
      wasmSecondary: wasmSecondaryInv,
      gpuPrimary: gpuPrimaryInv,
      gpuSecondary: gpuSecondaryInv,
      bothPassed:
        wasmPrimaryInv.bitIdentical &&
        wasmSecondaryInv.bitIdentical &&
        gpuPrimaryInv.bitIdentical &&
        gpuSecondaryInv.bitIdentical,
    };

    const gpuSuffixWasm7VsWasmSplitPrimary = compareFloat32TensorsDetailed(
      gpuSuffixUntappedWasm7.val!.byName[pName].data,
      wasmSplitUntapped.val!.byName[pName].data,
      [1, 240, 320, 1]
    );
    const gpuSuffixWasm7VsWasmSplitSecondary = compareFloat32TensorsDetailed(
      gpuSuffixUntappedWasm7.val!.byName[sName].data,
      wasmSplitUntapped.val!.byName[sName].data,
      [1, 240, 320, 1]
    );
    const gpuSuffixDispatchObserved = hasObservedGpuDispatch(gpuSuffixUntappedWasm7Profile);

    log(
      `  [Split Baseline Invariance] bothPassed=${splitBaselineInvariance.bothPassed} (wasm=${splitBaselineInvariance.wasmSplitMatchesWasmFullBitExact}, gpu=${splitBaselineInvariance.gpuSplitMatchesGpuFullBitExact})\n` +
        `  [Untapped Suffix GPU(wasm#7) vs WASM(wasm#7)] primaryMax=${gpuSuffixWasm7VsWasmSplitPrimary.stats.maxAbsDiff.toExponential(3)} secondaryMax=${gpuSuffixWasm7VsWasmSplitSecondary.stats.maxAbsDiff.toExponential(3)} dispatchObserved=${gpuSuffixDispatchObserved}`
    );

    if (!splitBaselineInvariance.bothPassed) {
      const reason = `Existing split baseline invariance FAILED on ${frameLabel}: wasm(P=${wasmPrimaryInv.mismatchCount}, S=${wasmSecondaryInv.mismatchCount}), gpu(P=${gpuPrimaryInv.mismatchCount}, S=${gpuSecondaryInv.mismatchCount})`;
      log(`  [ABORT] ${reason}`);
      return {
        frameLabel,
        feedSha256: feedSha,
        ok: false,
        aborted: true,
        abortReason: reason,
        untappedFullFinalsComparison,
        untappedFullSecondaryComparison,
        conv7ParentComparison: { gpuSha256: gpuConv7Sha, wasmSha256: wasmConv7Sha, comparison: conv7Comparison },
        bypassCaptureDetails,
        splitBaselineInvariance,
        gpuSuffixUntappedWasm7DispatchObserved: gpuSuffixDispatchObserved,
        gpuSuffixUntappedWasm7Profile: gpuSuffixUntappedWasm7Profile,
        gpuSuffixWasm7VsWasmSplitPrimary,
        gpuSuffixWasm7VsWasmSplitSecondary,
        tapInvarianceChecks,
        paired89Tap: {
          executed: false,
          tappedModelSha256: '',
          tappedModelByteLength: 0,
          tappedOutputs: [],
          invariancePassed: false,
          node8: null,
          node9: null,
        },
        paired1011Tap: {
          executed: false,
          skippedReason: reason,
          tappedModelSha256: null,
          tappedModelByteLength: null,
          tappedOutputs: [],
          invariancePassed: null,
          node10: null,
          node11: null,
        },
        localizedFirstDifferingOp: {
          stage: 'none-through-node11',
          nodeIdx: null,
          nodeName: null,
          opType: null,
          outputTensor: null,
        },
        isolatedReplay: null,
      };
    }

    const buildInitRecords = (nodeSpec: PostConv7TargetNodeSpec) =>
      nodeSpec.initializerNames.map((name) => {
        const init = parsedModel.initializersByName.get(name);
        if (!init) throw new Error(`Missing initializer "${name}" for node #${nodeSpec.nodeIdx}`);
        return {
          name,
          dims: [...init.dims],
          dataTypeName: init.dataTypeName,
          protoSha256: init.protoSha256,
          valueSha256: init.valueSha256,
        };
      });

    const observeNodeFromPair = async (
      nodeSpec: PostConv7TargetNodeSpec,
      gpuOutputs: OrtVersionDiagnosticOutput[],
      wasmOutputs: OrtVersionDiagnosticOutput[],
      parentInfo: {
        name: string;
        source: 'common-conv7-feed' | 'paired-node-tap';
        bitwiseIdentical: boolean;
        gpuSha256: string;
        wasmSha256: string;
      }
    ): Promise<{ obs: SuffixNodeTapObservation; gpuF32: Float32Array; wasmF32: Float32Array; gpuSha: string; wasmSha: string }> => {
      const go = gpuOutputs.find((o) => o.name === nodeSpec.outputTensor);
      const wo = wasmOutputs.find((o) => o.name === nodeSpec.outputTensor);
      if (!go || !wo) {
        throw new Error(`Missing paired tap output "${nodeSpec.outputTensor}" for node #${nodeSpec.nodeIdx}`);
      }
      const gBytes = copyRawOutputBytes(go);
      const wBytes = copyRawOutputBytes(wo);
      const gpuSha = await sha256HexBytes(gBytes);
      const wasmSha = await sha256HexBytes(wBytes);
      const gpuF32 = new Float32Array(go.buffer).slice();
      const wasmF32 = new Float32Array(wo.buffer).slice();
      const comparison = compareFloat32TensorsDetailed(gpuF32, wasmF32, nodeSpec.outputDims);
      const obs: SuffixNodeTapObservation = {
        nodeIdx: nodeSpec.nodeIdx,
        nodeName: nodeSpec.nodeName,
        opType: nodeSpec.opType,
        outputTensor: nodeSpec.outputTensor,
        dims: [...nodeSpec.outputDims],
        dataType: go.dataType,
        byteLength: wBytes.byteLength,
        gpuSha256: gpuSha,
        wasmSha256: wasmSha,
        comparison,
        immediateDynamicInputs: [parentInfo],
        initializers: buildInitRecords(nodeSpec),
        allImmediateInputsByteEqual: parentInfo.bitwiseIdentical,
      };
      return { obs, gpuF32, wasmF32, gpuSha, wasmSha };
    };

    log(
      `\n--- [${frameLabel}] Step 3: Paired Tap [#${chainSpec.node8LeakyRelu.nodeIdx} ${chainSpec.node8LeakyRelu.opType}, #${chainSpec.node9ConvDown0.nodeIdx} ${chainSpec.node9ConvDown0.opType}] on Shared Suffix (wasm#7, bypass) ---`
    );
    const tap89Names = [chainSpec.node8LeakyRelu.outputTensor, chainSpec.node9ConvDown0.outputTensor];
    const tap89Model = appendOnnxGraphOutputs(suffixSpec.suffixBytes, tap89Names, { maxExtraOutputs: 2 });
    const tap89Sha = await sha256HexBytes(tap89Model.bytes);

    const gpuTap89 = (
      await runSuffixModel(
        `${frameLabel} GPU-Suffix-Tap(#8,#9)(wasm#7)`,
        tap89Model.bytes,
        tap89Sha,
        'webgpu',
        wasmConv7F32,
        wasmConv7Sha
      )
    ).res as OrtVersionDiagnosticSuccess;
    const wasmTap89 = (
      await runSuffixModel(
        `${frameLabel} WASM-Suffix-Tap(#8,#9)(wasm#7)`,
        tap89Model.bytes,
        tap89Sha,
        'wasm',
        wasmConv7F32,
        wasmConv7Sha
      )
    ).res as OrtVersionDiagnosticSuccess;

    const inv89 = checkInvariancePair(
      'suffix-tap-8-9-vs-untapped-wasm7',
      tap89Model.bytes.byteLength,
      tap89Sha,
      'suffix-untapped-wasm7',
      gpuTap89.outputs,
      gpuSuffixUntappedWasm7.val!.byName,
      wasmTap89.outputs,
      wasmSplitUntapped.val!.byName
    );
    tapInvarianceChecks.push(inv89);
    log(
      `  [Tap(#8,#9) Final Bit-Identity vs Untapped Suffix(wasm#7)] bothPassed=${inv89.bothPassed} ` +
        `(gpuP=${inv89.gpuPrimaryBitIdentical.mismatchCount}, gpuS=${inv89.gpuSecondaryBitIdentical.mismatchCount}, wasmP=${inv89.wasmPrimaryBitIdentical.mismatchCount}, wasmS=${inv89.wasmSecondaryBitIdentical.mismatchCount})`
    );

    if (!inv89.bothPassed) {
      const reason = `[${frameLabel}] Paired #8/#9 tap perturbed suffix final outputs relative to untapped suffix(wasm#7)`;
      log(`  [ABORT] ${reason}`);
      return {
        frameLabel,
        feedSha256: feedSha,
        ok: false,
        aborted: true,
        abortReason: reason,
        untappedFullFinalsComparison,
        untappedFullSecondaryComparison,
        conv7ParentComparison: { gpuSha256: gpuConv7Sha, wasmSha256: wasmConv7Sha, comparison: conv7Comparison },
        bypassCaptureDetails,
        splitBaselineInvariance,
        gpuSuffixUntappedWasm7DispatchObserved: gpuSuffixDispatchObserved,
        gpuSuffixUntappedWasm7Profile: gpuSuffixUntappedWasm7Profile,
        gpuSuffixWasm7VsWasmSplitPrimary,
        gpuSuffixWasm7VsWasmSplitSecondary,
        tapInvarianceChecks,
        paired89Tap: {
          executed: true,
          tappedModelSha256: tap89Sha,
          tappedModelByteLength: tap89Model.bytes.byteLength,
          tappedOutputs: tap89Names,
          invariancePassed: false,
          node8: null,
          node9: null,
        },
        paired1011Tap: {
          executed: false,
          skippedReason: reason,
          tappedModelSha256: null,
          tappedModelByteLength: null,
          tappedOutputs: [],
          invariancePassed: null,
          node10: null,
          node11: null,
        },
        localizedFirstDifferingOp: {
          stage: 'none-through-node11',
          nodeIdx: null,
          nodeName: null,
          opType: null,
          outputTensor: null,
        },
        isolatedReplay: null,
      };
    }

    const cap8 = await observeNodeFromPair(chainSpec.node8LeakyRelu, gpuTap89.outputs, wasmTap89.outputs, {
      name: suffixSpec.primaryCutInput.name,
      source: 'common-conv7-feed',
      bitwiseIdentical: true,
      gpuSha256: wasmConv7Sha,
      wasmSha256: wasmConv7Sha,
    });
    const cap8BitExact = cap8.obs.comparison.bitDiffCount === 0 && cap8.gpuSha === cap8.wasmSha;
    const cap9 = await observeNodeFromPair(chainSpec.node9ConvDown0, gpuTap89.outputs, wasmTap89.outputs, {
      name: chainSpec.node8LeakyRelu.outputTensor,
      source: 'paired-node-tap',
      bitwiseIdentical: cap8BitExact,
      gpuSha256: cap8.gpuSha,
      wasmSha256: cap8.wasmSha,
    });
    const cap9BitExact = cap9.obs.comparison.bitDiffCount === 0 && cap9.gpuSha === cap9.wasmSha;

    log(
      `  [#${cap8.obs.nodeIdx} ${cap8.obs.opType}] class=${cap8.obs.comparison.classification} bitDiffs=${cap8.obs.comparison.bitDiffCount} nonZero=${cap8.obs.comparison.nonZeroNumericDiffCount} max=${cap8.obs.comparison.stats.maxAbsDiff.toExponential(3)} parentByteEq=${cap8.obs.allImmediateInputsByteEqual}\n` +
        `  [#${cap9.obs.nodeIdx} ${cap9.obs.opType}] class=${cap9.obs.comparison.classification} bitDiffs=${cap9.obs.comparison.bitDiffCount} nonZero=${cap9.obs.comparison.nonZeroNumericDiffCount} max=${cap9.obs.comparison.stats.maxAbsDiff.toExponential(3)} p99=${cap9.obs.comparison.stats.p99AbsDiff.toExponential(3)} mean=${cap9.obs.comparison.stats.meanAbsDiff.toExponential(3)} parentByteEq=${cap9.obs.allImmediateInputsByteEqual}`
    );

    const runIsolatedReplayForTarget = async (
      targetSpec: PostConv7TargetNodeSpec,
      targetObs: SuffixNodeTapObservation,
      fullSuffixGpuF32: Float32Array,
      fullSuffixWasmF32: Float32Array,
      capturedInputF32: Float32Array,
      capturedInputSha: string,
      capturedInputBitEqual: boolean
    ): Promise<PostConv7IsolatedReplayResult> => {
      const targetNode = parsedModel.nodes[targetSpec.nodeIdx];
      if (!capturedInputBitEqual) {
        return {
          executed: false,
          blockerReason: `Immediate parent of #${targetSpec.nodeIdx} is NOT bitwise-equal`,
          targetNodeIdx: targetSpec.nodeIdx,
          targetNodeName: targetSpec.nodeName,
          targetOpType: targetSpec.opType,
          targetOutputTensor: targetSpec.outputTensor,
          submodelByteLength: null,
          submodelSha256: null,
          capturedInputName: targetSpec.dynamicInputs[0] ?? '',
          capturedInputSha256: capturedInputSha,
          capturedInputBitwiseEqual: false,
          initializerManifest: targetObs.initializers,
          webgpuDispatchObserved: false,
          webgpuDispatchProfile: null,
          replayGpuVsReplayWasm: null,
          replayGpuVsFullSuffixGpu: null,
          replayWasmVsFullSuffixWasm: null,
          reproducesFullSuffixGpuBits: null,
          reproducesFullSuffixWasmBits: null,
          reproducesFullSuffixDiff: null,
          float64Reference: null,
          convReductionVariants: null,
          leakyReluVariants: null,
          interpretationNote: 'Single-op replay blocked because immediate dynamic input differed across EPs.',
        };
      }

      log(`\n--- [${frameLabel}] Step 4: Isolated 1-Op Replay (#${targetSpec.nodeIdx} ${targetSpec.opType}) ---`);
      const sub = buildSingleOpCapturedInputModel(parsedModel, targetNode);
      const subSha = await sha256HexBytes(sub.bytes);
      const dynIn = sub.dynamicInputSpecs[0];

      const [repGpuRaw, repWasmRaw] = await Promise.all([
        runWorker({
          expectedOrtVersion: '1.30.0',
          runtimeAssetFamily: 'asyncify',
          modelBuffer: sub.bytes.slice().buffer,
          feedBuffer: capturedInputF32.slice().buffer,
          feedName: dynIn.name,
          feedDims: [...dynIn.dims],
          allowArbitraryOutputs: true,
          provider: 'webgpu',
          graphOptimizationLevel: 'disabled',
          warmRuns: 0,
          enableWebGpuProfiling: false,
          disableCpuEpFallback,
          captureGpuValidationErrors: true,
        }),
        runWorker({
          expectedOrtVersion: '1.30.0',
          runtimeAssetFamily: 'asyncify',
          modelBuffer: sub.bytes.slice().buffer,
          feedBuffer: capturedInputF32.slice().buffer,
          feedName: dynIn.name,
          feedDims: [...dynIn.dims],
          allowArbitraryOutputs: true,
          provider: 'wasm',
          graphOptimizationLevel: 'disabled',
          warmRuns: 0,
          enableWebGpuProfiling: false,
          disableCpuEpFallback: false,
          captureGpuValidationErrors: false,
        }),
      ]);

      const repGpu = assertNonJsepSuccess(
        `${frameLabel} Replay #${targetSpec.nodeIdx} WebGPU`,
        repGpuRaw,
        'webgpu',
        subSha,
        sub.bytes.byteLength,
        capturedInputSha
      );
      const repWasm = assertNonJsepSuccess(
        `${frameLabel} Replay #${targetSpec.nodeIdx} WASM`,
        repWasmRaw,
        'wasm',
        subSha,
        sub.bytes.byteLength,
        capturedInputSha
      );

      const profRaw = await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: sub.bytes.slice().buffer,
        feedBuffer: capturedInputF32.slice().buffer,
        feedName: dynIn.name,
        feedDims: [...dynIn.dims],
        allowArbitraryOutputs: true,
        provider: 'webgpu',
        graphOptimizationLevel: 'disabled',
        warmRuns: 0,
        enableWebGpuProfiling: true,
        disableCpuEpFallback,
        captureGpuValidationErrors: true,
      });
      const webgpuDispatchObserved = hasObservedGpuDispatch(profRaw);

      const gOut = repGpu.outputs.find((o) => o.name === sub.outputSpec.name) ?? repGpu.outputs[0];
      const wOut = repWasm.outputs.find((o) => o.name === sub.outputSpec.name) ?? repWasm.outputs[0];
      if (!gOut || !wOut) {
        throw new Error(`Missing output "${sub.outputSpec.name}" from isolated replay of #${targetSpec.nodeIdx}`);
      }
      const repGpuF32 = new Float32Array(gOut.buffer).slice();
      const repWasmF32 = new Float32Array(wOut.buffer).slice();

      const replayGpuVsReplayWasm = compareFloat32TensorsDetailed(repGpuF32, repWasmF32, sub.outputSpec.dims);
      const replayGpuVsFullSuffixGpu = compareFloat32TensorsDetailed(repGpuF32, fullSuffixGpuF32, sub.outputSpec.dims);
      const replayWasmVsFullSuffixWasm = compareFloat32TensorsDetailed(repWasmF32, fullSuffixWasmF32, sub.outputSpec.dims);

      const reproducesFullSuffixGpuBits = replayGpuVsFullSuffixGpu.bitDiffCount === 0;
      const reproducesFullSuffixWasmBits = replayWasmVsFullSuffixWasm.bitDiffCount === 0;
      const reproducesFullSuffixDiff =
        reproducesFullSuffixGpuBits &&
        reproducesFullSuffixWasmBits &&
        replayGpuVsReplayWasm.bitDiffCount > 0;

      const priority: number[] = [];
      if (replayGpuVsReplayWasm.firstBitDiff) priority.push(replayGpuVsReplayWasm.firstBitDiff.flatIdx);
      if (replayGpuVsReplayWasm.firstNonZeroNumericDiff) priority.push(replayGpuVsReplayWasm.firstNonZeroNumericDiff.flatIdx);
      if (replayGpuVsReplayWasm.maxAbsDiffLocation) priority.push(replayGpuVsReplayWasm.maxAbsDiffLocation.flatIdx);

      const float64Reference = computeSequentialFloat64Reference(
        parsedModel,
        targetNode,
        { [dynIn.name]: { dims: [...dynIn.dims], data: capturedInputF32 } },
        sub.outputSpec.dims,
        repGpuF32,
        repWasmF32,
        priority
      );

      let convReductionVariants: PostConv7ConvExtendedVariantReport | null = null;
      let leakyReluVariants: LeakyReluVariantProbeReport | null = null;
      if (targetNode.opType === 'Conv') {
        convReductionVariants = probeConvOpReductionVariantsExtended(
          parsedModel,
          targetNode.idx,
          capturedInputF32,
          dynIn.dims,
          repGpuF32,
          repWasmF32,
          sub.outputSpec.dims,
          { maxSamples: 128 }
        );
      } else if (targetNode.opType === 'LeakyRelu') {
        leakyReluVariants = probeLeakyReluNumericVariants(
          targetNode,
          capturedInputF32,
          repGpuF32,
          repWasmF32,
          { maxSamples: 128 }
        );
      }

      const interpretationNote = !webgpuDispatchObserved
        ? `1-op WebGPU replay of #${targetNode.idx} did NOT observe WebGPU dispatch.`
        : reproducesFullSuffixDiff
          ? `1-op replay of #${targetNode.idx} ${targetNode.opType} with verified WebGPU dispatch reproduced BOTH full-suffix WebGPU and WASM outputs bit-identically (maxAbsDiff=${replayGpuVsReplayWasm.stats.maxAbsDiff.toExponential(3)}, bitDiffs=${replayGpuVsReplayWasm.bitDiffCount}).`
          : `1-op replay vs full-suffix bitDiffs: gpu=${replayGpuVsFullSuffixGpu.bitDiffCount}, wasm=${replayWasmVsFullSuffixWasm.bitDiffCount}.`;

      log(
        `  [1-Op Replay #${targetNode.idx} ${targetNode.opType}] subBytes=${sub.bytes.byteLength} sha=${subSha.slice(0, 12)}... dispatchObserved=${webgpuDispatchObserved}\n` +
          `  [Replay GPU vs WASM] ${replayGpuVsReplayWasm.classification} bitDiffs=${replayGpuVsReplayWasm.bitDiffCount} nonZero=${replayGpuVsReplayWasm.nonZeroNumericDiffCount} max=${replayGpuVsReplayWasm.stats.maxAbsDiff.toExponential(3)} p99=${replayGpuVsReplayWasm.stats.p99AbsDiff.toExponential(3)} mean=${replayGpuVsReplayWasm.stats.meanAbsDiff.toExponential(3)}\n` +
          `  [Replay vs Full-Suffix] gpuBitDiffs=${replayGpuVsFullSuffixGpu.bitDiffCount} (max=${replayGpuVsFullSuffixGpu.stats.maxAbsDiff.toExponential(3)}) | wasmBitDiffs=${replayWasmVsFullSuffixWasm.bitDiffCount} (max=${replayWasmVsFullSuffixWasm.stats.maxAbsDiff.toExponential(3)})\n` +
          `  [Float64 Ref] ${float64Reference.available ? `mode=${float64Reference.mode} eval=${float64Reference.evaluatedCount} maxGpuErr=${float64Reference.maxAbsGpuVsF64?.toExponential(3)} maxWasmErr=${float64Reference.maxAbsWasmVsF64?.toExponential(3)} closer(gpu=${float64Reference.gpuCloserToF64Count}, wasm=${float64Reference.wasmCloserToF64Count}, eq=${float64Reference.equalDistToF64Count})` : `UNAVAILABLE(${float64Reference.blockerReason})`}`
      );
      if (convReductionVariants?.available) {
        log(
          `  [Conv Reduction Variants (eval=${convReductionVariants.evaluatedCount})] ` +
            convReductionVariants.standardProbe.variants
              .map(
                (v) =>
                  `${v.variantId}: gpuBit=${v.gpuExactBitMatches}/${v.evaluatedCount}(max=${v.maxAbsDiffFromGpu.toExponential(2)}) wasmBit=${v.wasmExactBitMatches}/${v.evaluatedCount}(max=${v.maxAbsDiffFromWasm.toExponential(2)})`
              )
              .join(' | ') +
            ' | ' +
            convReductionVariants.extraVariants
              .map(
                (v) =>
                  `${v.variantId}: gpuBit=${v.gpuExactBitMatches}/${v.evaluatedCount}(max=${v.maxAbsDiffFromGpu.toExponential(2)}) wasmBit=${v.wasmExactBitMatches}/${v.evaluatedCount}(max=${v.maxAbsDiffFromWasm.toExponential(2)})`
              )
              .join(' | ')
        );
      } else if (leakyReluVariants?.available) {
        log(
          `  [LeakyRelu Variants (eval=${leakyReluVariants.evaluatedCount})] ` +
            leakyReluVariants.variants
              .map(
                (v) =>
                  `${v.variantId}: gpuBit=${v.gpuExactBitMatches}/${v.evaluatedCount}(max=${v.maxAbsDiffFromGpu.toExponential(2)}) wasmBit=${v.wasmExactBitMatches}/${v.evaluatedCount}(max=${v.maxAbsDiffFromWasm.toExponential(2)})`
              )
              .join(' | ')
        );
      }

      return {
        executed: true,
        blockerReason: null,
        targetNodeIdx: targetSpec.nodeIdx,
        targetNodeName: targetSpec.nodeName,
        targetOpType: targetSpec.opType,
        targetOutputTensor: targetSpec.outputTensor,
        submodelByteLength: sub.bytes.byteLength,
        submodelSha256: subSha,
        capturedInputName: dynIn.name,
        capturedInputSha256: capturedInputSha,
        capturedInputBitwiseEqual: true,
        initializerManifest: targetObs.initializers,
        webgpuDispatchObserved,
        webgpuDispatchProfile: profRaw,
        replayGpuVsReplayWasm,
        replayGpuVsFullSuffixGpu,
        replayWasmVsFullSuffixWasm,
        reproducesFullSuffixGpuBits,
        reproducesFullSuffixWasmBits,
        reproducesFullSuffixDiff,
        float64Reference,
        convReductionVariants,
        leakyReluVariants,
        interpretationNote,
      };
    };

    const paired89Tap = {
      executed: true,
      tappedModelSha256: tap89Sha,
      tappedModelByteLength: tap89Model.bytes.byteLength,
      tappedOutputs: tap89Names,
      invariancePassed: true,
      node8: cap8.obs,
      node9: cap9.obs,
    };

    // Stop immediately if #8 differs!
    if (!cap8BitExact) {
      const replay = await runIsolatedReplayForTarget(
        chainSpec.node8LeakyRelu,
        cap8.obs,
        cap8.gpuF32,
        cap8.wasmF32,
        wasmConv7F32,
        wasmConv7Sha,
        true
      );
      return {
        frameLabel,
        feedSha256: feedSha,
        ok: true,
        aborted: false,
        abortReason: null,
        untappedFullFinalsComparison,
        untappedFullSecondaryComparison,
        conv7ParentComparison: { gpuSha256: gpuConv7Sha, wasmSha256: wasmConv7Sha, comparison: conv7Comparison },
        bypassCaptureDetails,
        splitBaselineInvariance,
        gpuSuffixUntappedWasm7DispatchObserved: gpuSuffixDispatchObserved,
        gpuSuffixUntappedWasm7Profile: gpuSuffixUntappedWasm7Profile,
        gpuSuffixWasm7VsWasmSplitPrimary,
        gpuSuffixWasm7VsWasmSplitSecondary,
        tapInvarianceChecks,
        paired89Tap,
        paired1011Tap: {
          executed: false,
          skippedReason: `Stopped: #${chainSpec.node8LeakyRelu.nodeIdx} LeakyRelu already differed (bitDiffs=${cap8.obs.comparison.bitDiffCount}).`,
          tappedModelSha256: null,
          tappedModelByteLength: null,
          tappedOutputs: [],
          invariancePassed: null,
          node10: null,
          node11: null,
        },
        localizedFirstDifferingOp: {
          stage: 'node8-leakyrelu',
          nodeIdx: chainSpec.node8LeakyRelu.nodeIdx,
          nodeName: chainSpec.node8LeakyRelu.nodeName,
          opType: chainSpec.node8LeakyRelu.opType,
          outputTensor: chainSpec.node8LeakyRelu.outputTensor,
        },
        isolatedReplay: replay,
      };
    }

    // #8 is exact; check #9 Conv from the same paired tap!
    if (!cap9BitExact) {
      const replay = await runIsolatedReplayForTarget(
        chainSpec.node9ConvDown0,
        cap9.obs,
        cap9.gpuF32,
        cap9.wasmF32,
        cap8.wasmF32,
        cap8.wasmSha,
        true
      );
      return {
        frameLabel,
        feedSha256: feedSha,
        ok: true,
        aborted: false,
        abortReason: null,
        untappedFullFinalsComparison,
        untappedFullSecondaryComparison,
        conv7ParentComparison: { gpuSha256: gpuConv7Sha, wasmSha256: wasmConv7Sha, comparison: conv7Comparison },
        bypassCaptureDetails,
        splitBaselineInvariance,
        gpuSuffixUntappedWasm7DispatchObserved: gpuSuffixDispatchObserved,
        gpuSuffixUntappedWasm7Profile: gpuSuffixUntappedWasm7Profile,
        gpuSuffixWasm7VsWasmSplitPrimary,
        gpuSuffixWasm7VsWasmSplitSecondary,
        tapInvarianceChecks,
        paired89Tap,
        paired1011Tap: {
          executed: false,
          skippedReason: `Stopped: #${chainSpec.node8LeakyRelu.nodeIdx} LeakyRelu exact (0 bitDiffs) and #${chainSpec.node9ConvDown0.nodeIdx} stride-2 Conv localized (bitDiffs=${cap9.obs.comparison.bitDiffCount}).`,
          tappedModelSha256: null,
          tappedModelByteLength: null,
          tappedOutputs: [],
          invariancePassed: null,
          node10: null,
          node11: null,
        },
        localizedFirstDifferingOp: {
          stage: 'node9-conv',
          nodeIdx: chainSpec.node9ConvDown0.nodeIdx,
          nodeName: chainSpec.node9ConvDown0.nodeName,
          opType: chainSpec.node9ConvDown0.opType,
          outputTensor: chainSpec.node9ConvDown0.outputTensor,
        },
        isolatedReplay: replay,
      };
    }

    // Both #8 AND #9 were bit-identical: ONLY THEN run #10/#11 paired tap!
    if (!chainSpec.node10LeakyRelu1 || !chainSpec.node11ConvDown11) {
      return {
        frameLabel,
        feedSha256: feedSha,
        ok: true,
        aborted: false,
        abortReason: null,
        untappedFullFinalsComparison,
        untappedFullSecondaryComparison,
        conv7ParentComparison: { gpuSha256: gpuConv7Sha, wasmSha256: wasmConv7Sha, comparison: conv7Comparison },
        bypassCaptureDetails,
        splitBaselineInvariance,
        gpuSuffixUntappedWasm7DispatchObserved: gpuSuffixDispatchObserved,
        gpuSuffixUntappedWasm7Profile: gpuSuffixUntappedWasm7Profile,
        gpuSuffixWasm7VsWasmSplitPrimary,
        gpuSuffixWasm7VsWasmSplitSecondary,
        tapInvarianceChecks,
        paired89Tap,
        paired1011Tap: {
          executed: false,
          skippedReason: 'Both #8 and #9 were exact, and #10/#11 nodes were not present in graph.',
          tappedModelSha256: null,
          tappedModelByteLength: null,
          tappedOutputs: [],
          invariancePassed: null,
          node10: null,
          node11: null,
        },
        localizedFirstDifferingOp: {
          stage: 'none-through-node11',
          nodeIdx: null,
          nodeName: null,
          opType: null,
          outputTensor: null,
        },
        isolatedReplay: null,
      };
    }

    log(
      `\n--- [${frameLabel}] Step 3b (#8&#9 Both Exact): Paired Tap [#${chainSpec.node10LeakyRelu1.nodeIdx}, #${chainSpec.node11ConvDown11.nodeIdx}] ---`
    );
    const tap1011Names = [chainSpec.node10LeakyRelu1.outputTensor, chainSpec.node11ConvDown11.outputTensor];
    const tap1011Model = appendOnnxGraphOutputs(suffixSpec.suffixBytes, tap1011Names, { maxExtraOutputs: 2 });
    const tap1011Sha = await sha256HexBytes(tap1011Model.bytes);

    const gpuTap1011 = (
      await runSuffixModel(
        `${frameLabel} GPU-Suffix-Tap(#10,#11)(wasm#7)`,
        tap1011Model.bytes,
        tap1011Sha,
        'webgpu',
        wasmConv7F32,
        wasmConv7Sha
      )
    ).res as OrtVersionDiagnosticSuccess;
    const wasmTap1011 = (
      await runSuffixModel(
        `${frameLabel} WASM-Suffix-Tap(#10,#11)(wasm#7)`,
        tap1011Model.bytes,
        tap1011Sha,
        'wasm',
        wasmConv7F32,
        wasmConv7Sha
      )
    ).res as OrtVersionDiagnosticSuccess;

    const inv1011 = checkInvariancePair(
      'suffix-tap-10-11-vs-untapped-wasm7',
      tap1011Model.bytes.byteLength,
      tap1011Sha,
      'suffix-untapped-wasm7',
      gpuTap1011.outputs,
      gpuSuffixUntappedWasm7.val!.byName,
      wasmTap1011.outputs,
      wasmSplitUntapped.val!.byName
    );
    tapInvarianceChecks.push(inv1011);

    if (!inv1011.bothPassed) {
      const reason = `[${frameLabel}] Paired #10/#11 tap perturbed suffix final outputs relative to untapped suffix(wasm#7)`;
      return {
        frameLabel,
        feedSha256: feedSha,
        ok: false,
        aborted: true,
        abortReason: reason,
        untappedFullFinalsComparison,
        untappedFullSecondaryComparison,
        conv7ParentComparison: { gpuSha256: gpuConv7Sha, wasmSha256: wasmConv7Sha, comparison: conv7Comparison },
        bypassCaptureDetails,
        splitBaselineInvariance,
        gpuSuffixUntappedWasm7DispatchObserved: gpuSuffixDispatchObserved,
        gpuSuffixUntappedWasm7Profile: gpuSuffixUntappedWasm7Profile,
        gpuSuffixWasm7VsWasmSplitPrimary,
        gpuSuffixWasm7VsWasmSplitSecondary,
        tapInvarianceChecks,
        paired89Tap,
        paired1011Tap: {
          executed: true,
          skippedReason: null,
          tappedModelSha256: tap1011Sha,
          tappedModelByteLength: tap1011Model.bytes.byteLength,
          tappedOutputs: tap1011Names,
          invariancePassed: false,
          node10: null,
          node11: null,
        },
        localizedFirstDifferingOp: {
          stage: 'none-through-node11',
          nodeIdx: null,
          nodeName: null,
          opType: null,
          outputTensor: null,
        },
        isolatedReplay: null,
      };
    }

    const cap10 = await observeNodeFromPair(chainSpec.node10LeakyRelu1, gpuTap1011.outputs, wasmTap1011.outputs, {
      name: chainSpec.node9ConvDown0.outputTensor,
      source: 'paired-node-tap',
      bitwiseIdentical: cap9BitExact,
      gpuSha256: cap9.gpuSha,
      wasmSha256: cap9.wasmSha,
    });
    const cap10BitExact = cap10.obs.comparison.bitDiffCount === 0 && cap10.gpuSha === cap10.wasmSha;
    const cap11 = await observeNodeFromPair(chainSpec.node11ConvDown11, gpuTap1011.outputs, wasmTap1011.outputs, {
      name: chainSpec.node10LeakyRelu1.outputTensor,
      source: 'paired-node-tap',
      bitwiseIdentical: cap10BitExact,
      gpuSha256: cap10.gpuSha,
      wasmSha256: cap10.wasmSha,
    });
    const cap11BitExact = cap11.obs.comparison.bitDiffCount === 0 && cap11.gpuSha === cap11.wasmSha;

    const paired1011Tap = {
      executed: true,
      skippedReason: null,
      tappedModelSha256: tap1011Sha,
      tappedModelByteLength: tap1011Model.bytes.byteLength,
      tappedOutputs: tap1011Names,
      invariancePassed: true,
      node10: cap10.obs,
      node11: cap11.obs,
    };

    if (!cap10BitExact) {
      const replay = await runIsolatedReplayForTarget(
        chainSpec.node10LeakyRelu1,
        cap10.obs,
        cap10.gpuF32,
        cap10.wasmF32,
        cap9.wasmF32,
        cap9.wasmSha,
        true
      );
      return {
        frameLabel,
        feedSha256: feedSha,
        ok: true,
        aborted: false,
        abortReason: null,
        untappedFullFinalsComparison,
        untappedFullSecondaryComparison,
        conv7ParentComparison: { gpuSha256: gpuConv7Sha, wasmSha256: wasmConv7Sha, comparison: conv7Comparison },
        bypassCaptureDetails,
        splitBaselineInvariance,
        gpuSuffixUntappedWasm7DispatchObserved: gpuSuffixDispatchObserved,
        gpuSuffixUntappedWasm7Profile: gpuSuffixUntappedWasm7Profile,
        gpuSuffixWasm7VsWasmSplitPrimary,
        gpuSuffixWasm7VsWasmSplitSecondary,
        tapInvarianceChecks,
        paired89Tap,
        paired1011Tap,
        localizedFirstDifferingOp: {
          stage: 'node10-leakyrelu1',
          nodeIdx: chainSpec.node10LeakyRelu1.nodeIdx,
          nodeName: chainSpec.node10LeakyRelu1.nodeName,
          opType: chainSpec.node10LeakyRelu1.opType,
          outputTensor: chainSpec.node10LeakyRelu1.outputTensor,
        },
        isolatedReplay: replay,
      };
    }

    if (!cap11BitExact) {
      const replay = await runIsolatedReplayForTarget(
        chainSpec.node11ConvDown11,
        cap11.obs,
        cap11.gpuF32,
        cap11.wasmF32,
        cap10.wasmF32,
        cap10.wasmSha,
        true
      );
      return {
        frameLabel,
        feedSha256: feedSha,
        ok: true,
        aborted: false,
        abortReason: null,
        untappedFullFinalsComparison,
        untappedFullSecondaryComparison,
        conv7ParentComparison: { gpuSha256: gpuConv7Sha, wasmSha256: wasmConv7Sha, comparison: conv7Comparison },
        bypassCaptureDetails,
        splitBaselineInvariance,
        gpuSuffixUntappedWasm7DispatchObserved: gpuSuffixDispatchObserved,
        gpuSuffixUntappedWasm7Profile: gpuSuffixUntappedWasm7Profile,
        gpuSuffixWasm7VsWasmSplitPrimary,
        gpuSuffixWasm7VsWasmSplitSecondary,
        tapInvarianceChecks,
        paired89Tap,
        paired1011Tap,
        localizedFirstDifferingOp: {
          stage: 'node11-conv11',
          nodeIdx: chainSpec.node11ConvDown11.nodeIdx,
          nodeName: chainSpec.node11ConvDown11.nodeName,
          opType: chainSpec.node11ConvDown11.opType,
          outputTensor: chainSpec.node11ConvDown11.outputTensor,
        },
        isolatedReplay: replay,
      };
    }

    return {
      frameLabel,
      feedSha256: feedSha,
      ok: true,
      aborted: false,
      abortReason: null,
      untappedFullFinalsComparison,
      untappedFullSecondaryComparison,
      conv7ParentComparison: { gpuSha256: gpuConv7Sha, wasmSha256: wasmConv7Sha, comparison: conv7Comparison },
      bypassCaptureDetails,
      splitBaselineInvariance,
      gpuSuffixUntappedWasm7DispatchObserved: gpuSuffixDispatchObserved,
      gpuSuffixUntappedWasm7Profile: gpuSuffixUntappedWasm7Profile,
      gpuSuffixWasm7VsWasmSplitPrimary,
      gpuSuffixWasm7VsWasmSplitSecondary,
      tapInvarianceChecks,
      paired89Tap,
      paired1011Tap,
      localizedFirstDifferingOp: {
        stage: 'none-through-node11',
        nodeIdx: null,
        nodeName: null,
        opType: null,
        outputTensor: null,
      },
      isolatedReplay: null,
    };
  };

  const f0102Result = await runFrame('F0102', params.f0102FeedFloat32, f0102Sha);
  const f0126Result =
    !f0102Result.aborted &&
    params.f0126FeedFloat32 &&
    f0126Sha &&
    (params.confirmF0126 ?? true)
      ? await runFrame('F0126', params.f0126FeedFloat32, f0126Sha)
      : null;

  const overallOk =
    n1OffProof.verified &&
    f0102Result.ok &&
    !f0102Result.aborted &&
    (f0126Result ? f0126Result.ok && !f0126Result.aborted : true);
  const overallAbortReason =
    n1OffProof.failureReason ?? f0102Result.abortReason ?? f0126Result?.abortReason ?? null;

  return {
    ok: overallOk,
    aborted: !overallOk,
    abortReason: overallAbortReason,
    pinnedOrtVersion: '1.30.0',
    runtimeAssetFamily: 'asyncify',
    graphOptimizationLevel: 'disabled',
    m1ByteLength: params.m1Bytes.byteLength,
    m1Sha256: m1Sha,
    n1OffProof,
    chainSpec,
    f0102: f0102Result,
    f0126: f0126Result,
    scientificScopeNote:
      'Isolates the first independent post-#7 divergence inside the verified suffix (#8..#725) under bitwise-identical (wasm#7, #2 Tile_2, #3 Tile_2) inputs. WASM M1 outputs serve only as a computational parity reference, not ground-truth geometry.',
  };
}
