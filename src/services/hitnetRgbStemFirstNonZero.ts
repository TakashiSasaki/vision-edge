import {
  appendOnnxGraphOutputs,
  summarizeRawTensorAndParity,
  type RawTensorParityStats,
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
  assertDevFixtureEnabled,
  HITNET_DEV_NON_JSEP_FIXTURE_SPEC,
} from './stereoDepthWebGpuDevFixture';
import {
  HITNET_M1_FULL_GRAPH_SHA256,
  inferOnnxStaticTensorMeta,
  type OnnxStaticInferenceResult,
} from './stereoDepthWorkerCore';

export type StemElementComparisonClassification =
  | 'BIT_IDENTICAL'
  | 'SIGNED_ZERO_ONLY_BIT_DIFF'
  | 'NONZERO_NUMERIC_SUBGATE'
  | 'STRICT_GATE_BREACH';

export interface ProtoSlice {
  fn: number;
  wt: number;
  valStart: number;
  valEnd: number;
  fieldStart: number;
  fieldEnd: number;
}

export interface ParsedM1Attribute {
  name: string;
  attrType: number;
  intVal: number | null;
  floatVal: number | null;
  intsVal: number[];
  summary: string;
}

export interface ParsedM1NodeFull {
  idx: number;
  name: string;
  opType: string;
  inputs: string[];
  outputs: string[];
  attrs: ParsedM1Attribute[];
  attrsSummary: string[];
  rawPayloadSlice: Uint8Array;
}

export interface ParsedM1InitializerInfo {
  name: string;
  dims: number[];
  dataType: number;
  dataTypeName: string;
  dataLocation: number;
  isExternal: boolean;
  externalInfo: string | null;
  protoByteLength: number;
  protoSha256: string;
  rawPayloadSlice: Uint8Array;
  float32Values: Float32Array | null;
  int64Values: BigInt64Array | null;
  valueSha256: string | null;
  blockerReason: string | null;
}

export interface ParsedM1ModelFull {
  irVersion: number;
  opsetImportPayloads: Uint8Array[];
  nodes: ParsedM1NodeFull[];
  initializersByName: Map<string, ParsedM1InitializerInfo>;
  initializerNames: Set<string>;
  graphInputNames: Set<string>;
  graphOutputNames: Set<string>;
  producerByOutput: Map<string, ParsedM1NodeFull>;
  staticMeta: OnnxStaticInferenceResult;
}

export interface AncestorClosureVerification {
  seedNodeIdx: number;
  seedNodeName: string;
  seedOutputTensor: string;
  selectionRule: string;
  totalClosureNodes: number;
  closureNodeIndices: number[];
  omittedInterleavedIndices: number[];
  topologicalOrderVerified: boolean;
  ancestryVerified: boolean;
  missingProducers: string[];
  tapBatches: Array<{
    batchIndex: number;
    nodeIndices: number[];
    tensorNames: string[];
  }>;
  orderedNodes: Array<{
    idx: number;
    opType: string;
    name: string;
    inputs: string[];
    outputs: string[];
    outputDims: number[] | null;
    outputElemType: number | null;
    attrsSummary: string[];
    initializerInputs: string[];
    dynamicInputs: string[];
  }>;
}

export interface BitOrNumericLocationSample {
  flatIdx: number;
  coord: number[];
  candValue: number;
  refValue: number;
  candHex: string;
  refHex: string;
  absDiff: number;
  signedDiff: number;
  ulpDiff: number | null;
  signedZeroPair: boolean;
}

export interface DetailedFloat32Comparison {
  length: number;
  dims: number[];
  bitDiffCount: number;
  signedZeroOnlyCount: number;
  nonZeroNumericDiffCount: number;
  passesStrictGate: boolean;
  classification: StemElementComparisonClassification;
  stats: RawTensorParityStats;
  firstBitDiff: BitOrNumericLocationSample | null;
  firstNonZeroNumericDiff: BitOrNumericLocationSample | null;
  maxAbsDiffLocation: BitOrNumericLocationSample | null;
}

export interface ImmediateInputVerification {
  inputName: string;
  kind: 'graph-input' | 'dynamic-tensor' | 'initializer' | 'empty-optional' | 'missing';
  dims: number[] | null;
  elemTypeName: string | null;
  verifiedIdentity: boolean;
  identityStatus:
    | 'VERIFIED_SHARED_MODEL_INITIALIZER'
    | 'VERIFIED_FEED_SHA'
    | 'OBSERVED_BIT_IDENTICAL'
    | 'OBSERVED_SIGNED_ZERO_ONLY'
    | 'OBSERVED_NONZERO_DIFF'
    | 'BLOCKER_EXTERNAL_INITIALIZER'
    | 'BLOCKER_UNSUPPORTED_INITIALIZER_DTYPE'
    | 'BLOCKER_MISSING_OBSERVATION'
    | 'OPTIONAL_EMPTY';
  sha256: string | null;
  bitDiffCount: number | null;
  signedZeroOnlyCount: number | null;
  nonZeroNumericDiffCount: number | null;
  maxAbsDiff: number | null;
  blockerReason: string | null;
}

export interface ClosureNodeTapObservation {
  nodeIdx: number;
  opType: string;
  nodeName: string;
  outputTensor: string;
  dims: number[];
  dataType: string;
  batchIndex: number;
  tappedModelSha256: string;
  tappedModelByteLength: number;
  gpuBufferSha256: string;
  wasmBufferSha256: string;
  comparison: DetailedFloat32Comparison;
  immediateInputs: ImmediateInputVerification[];
  allImmediateInputsByteEqual: boolean;
  inputBlockers: string[];
}

export interface SequentialFloat64RefResult {
  available: boolean;
  mode: 'full-tensor' | 'sampled-coords' | 'unavailable';
  opType: string;
  elementCount: number;
  evaluatedCount: number;
  scalarWorkEstimate: number;
  blockerReason: string | null;
  maxAbsGpuVsF64: number | null;
  meanAbsGpuVsF64: number | null;
  maxAbsWasmVsF64: number | null;
  meanAbsWasmVsF64: number | null;
  gpuCloserToF64Count: number;
  wasmCloserToF64Count: number;
  equalDistToF64Count: number;
  sampleCoords: Array<{
    flatIdx: number;
    coord: number[];
    f64Ref: number;
    f64RoundedF32: number;
    gpu: number;
    wasm: number;
    gpuAbsErrFromF64: number;
    wasmAbsErrFromF64: number;
  }>;
}

export interface SingleOpReplayReport {
  executed: boolean;
  blockerReason: string | null;
  targetNodeIdx: number;
  targetOpType: string;
  targetNodeName: string;
  targetOutputTensor: string;
  submodelByteLength: number | null;
  submodelSha256: string | null;
  capturedFeedSpecs: Array<{
    name: string;
    dims: number[];
    byteLength: number;
    gpuCapturedSha256: string;
    wasmCapturedSha256: string;
    identicalCapturedBytes: boolean;
  }>;
  initializerSpecs: Array<{
    name: string;
    dims: number[];
    dataTypeName: string;
    protoByteLength: number;
    protoSha256: string;
    valueSha256: string | null;
  }>;
  webgpuReplaySuccess: OrtVersionDiagnosticSuccess | null;
  wasmReplaySuccess: OrtVersionDiagnosticSuccess | null;
  webgpuProfileRun: OrtVersionDiagnosticResponse | null;
  webgpuKernelDispatchObserved: boolean;
  replayGpuVsReplayWasm: DetailedFloat32Comparison | null;
  replayGpuVsFullM1Gpu: DetailedFloat32Comparison | null;
  replayWasmVsFullM1Wasm: DetailedFloat32Comparison | null;
  reproducesFullM1GpuBits: boolean | null;
  reproducesFullM1WasmBits: boolean | null;
  reproducesFullM1Diff: boolean | null;
  float64Reference: SequentialFloat64RefResult | null;
  interpretationNote: string;
}

export interface F0126ConfirmationSummary {
  executed: boolean;
  feedSha256: string;
  untappedGpuVsWasmPrimary: RawTensorParityStats;
  untappedGpuVsWasmSecondary: RawTensorParityStats;
  candidateTap: {
    targetNodeIdx: number;
    targetOutputTensor: string;
    parentTensor: string | null;
    tappedModelSha256: string;
    bitIdenticalFinals: boolean;
    perturbationDetail: string | null;
    parentComparison: DetailedFloat32Comparison | null;
    targetComparison: DetailedFloat32Comparison | null;
  } | null;
}

export interface RgbStemFirstNonZeroReport {
  ok: boolean;
  abortReason: string | null;
  frame: 'F0102';
  pinnedOrtVersion: '1.30.0';
  runtimeAssetFamily: 'asyncify';
  graphOptimizationLevel: 'disabled';
  m1ByteLength: number;
  m1Sha256: string;
  f0102FeedSha256: string;
  closure: AncestorClosureVerification;
  untappedBaseline: {
    gpuModelSha256: string;
    wasmModelSha256: string;
    gpuFeedSha256: string;
    wasmFeedSha256: string;
    primary: DetailedFloat32Comparison;
    secondary: DetailedFloat32Comparison;
    gpuDispatchProfile: OrtVersionDiagnosticResponse | null;
  };
  tapBatchSummaries: Array<{
    batchIndex: number;
    tapOutputs: string[];
    tappedModelSha256: string;
    tappedModelByteLength: number;
    bitIdenticalFinals: boolean;
    perturbationDetail: string | null;
    finalInvariance: {
      gpuPrimaryBitIdentical: BitIdentityResult;
      gpuSecondaryBitIdentical: BitIdentityResult;
      wasmPrimaryBitIdentical: BitIdentityResult;
      wasmSecondaryBitIdentical: BitIdentityResult;
    };
  }>;
  nodeObservations: ClosureNodeTapObservation[];
  negativeControlsBitIdentical: Array<{
    nodeIdx: number;
    opType: string;
    nodeName: string;
    outputTensor: string;
    classification: StemElementComparisonClassification;
  }>;
  earliestBitDiffNode: ClosureNodeTapObservation | null;
  earliestSignedZeroOnlyNode: ClosureNodeTapObservation | null;
  earliestNonZeroNumericNode: ClosureNodeTapObservation | null;
  earliestStrictGateBreachNode: ClosureNodeTapObservation | null;
  singleOpReplay: SingleOpReplayReport | null;
  f0126Confirmation: F0126ConfirmationSummary | null;
}

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

function readVarint64(buf: Uint8Array, pos: number, end: number): [bigint, number] | null {
  let result = 0n;
  let shift = 0n;
  let c = pos;
  while (c < end && shift <= 63n) {
    const b = buf[c++];
    result |= BigInt(b & 0x7f) << shift;
    if ((b & 0x80) === 0) {
      return [result, c];
    }
    shift += 7n;
  }
  return null;
}

function readSignedVarint64(buf: Uint8Array, pos: number, end: number): [number, number] | null {
  const v = readVarint64(buf, pos, end);
  if (!v) return null;
  return [Number(BigInt.asIntN(64, v[0])), v[1]];
}

function parseProtoFieldsWithBounds(
  buf: Uint8Array,
  start: number,
  end: number
): ProtoSlice[] | null {
  const out: ProtoSlice[] = [];
  let c = start;
  while (c < end) {
    const fieldStart = c;
    const k = readVarint(buf, c, end);
    if (!k) return null;
    const fn = k[0] >>> 3;
    const wt = k[0] & 7;
    c = k[1];
    if (wt === 0) {
      const v = readVarint64(buf, c, end);
      if (!v) return null;
      out.push({ fn, wt, valStart: c, valEnd: v[1], fieldStart, fieldEnd: v[1] });
      c = v[1];
    } else if (wt === 2) {
      const l = readVarint(buf, c, end);
      if (!l) return null;
      const vs = l[1];
      const ve = vs + l[0];
      if (ve > end) return null;
      out.push({ fn, wt, valStart: vs, valEnd: ve, fieldStart, fieldEnd: ve });
      c = ve;
    } else if (wt === 5) {
      if (c + 4 > end) return null;
      out.push({ fn, wt, valStart: c, valEnd: c + 4, fieldStart, fieldEnd: c + 4 });
      c += 4;
    } else if (wt === 1) {
      if (c + 8 > end) return null;
      out.push({ fn, wt, valStart: c, valEnd: c + 8, fieldStart, fieldEnd: c + 8 });
      c += 8;
    } else {
      return null;
    }
  }
  return out;
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

function onnxDataTypeToName(dt: number): string {
  switch (dt) {
    case 1:
      return 'float32';
    case 2:
      return 'uint8';
    case 3:
      return 'int8';
    case 5:
      return 'int16';
    case 6:
      return 'int32';
    case 7:
      return 'int64';
    case 9:
      return 'bool';
    case 10:
      return 'float16';
    case 11:
      return 'float64';
    default:
      return `onnx-type(${dt})`;
  }
}

function parseAttributeFull(
  buf: Uint8Array,
  start: number,
  end: number,
  dec: TextDecoder
): ParsedM1Attribute | null {
  const fields = parseProtoFieldsWithBounds(buf, start, end);
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
  let summary = attrName;
  if (attrType === 2 && intVal !== null) summary = `${attrName}=${intVal}`;
  else if (attrType === 1 && floatVal !== null) summary = `${attrName}=${floatVal}`;
  else if (attrType === 7 && intsVal.length > 0) summary = `${attrName}=[${intsVal.join(',')}]`;
  return { name: attrName, attrType, intVal, floatVal, intsVal, summary };
}

export async function parseM1ModelProtoWithRawSlices(
  m1Bytes: Uint8Array
): Promise<ParsedM1ModelFull> {
  const staticMeta = inferOnnxStaticTensorMeta(m1Bytes);
  if (!staticMeta.valid) {
    throw new Error('M1 ONNX failed static tensor inference during full graph parse');
  }
  const dec = new TextDecoder();
  const top = parseProtoFieldsWithBounds(m1Bytes, 0, m1Bytes.byteLength);
  if (!top) throw new Error('Invalid ONNX ModelProto');

  let irVersion = 8;
  const opsetImportPayloads: Uint8Array[] = [];
  let gf: ProtoSlice | null = null;
  for (const f of top) {
    if (f.fn === 1 && f.wt === 0) {
      const v = readSignedVarint64(m1Bytes, f.valStart, f.valEnd);
      if (v) irVersion = v[0];
    } else if (f.fn === 8 && f.wt === 2) {
      opsetImportPayloads.push(m1Bytes.slice(f.valStart, f.valEnd));
    } else if (f.fn === 7 && f.wt === 2) {
      gf = f;
    }
  }
  if (!gf) throw new Error('Missing GraphProto in ModelProto');
  const gFields = parseProtoFieldsWithBounds(m1Bytes, gf.valStart, gf.valEnd);
  if (!gFields) throw new Error('Invalid GraphProto');

  const nodes: ParsedM1NodeFull[] = [];
  const initializersByName = new Map<string, ParsedM1InitializerInfo>();
  const initializerNames = new Set<string>();
  const graphInputNames = new Set<string>();
  const graphOutputNames = new Set<string>();
  const producerByOutput = new Map<string, ParsedM1NodeFull>();

  for (const f of gFields) {
    if (f.fn === 1 && f.wt === 2) {
      const nf = parseProtoFieldsWithBounds(m1Bytes, f.valStart, f.valEnd);
      if (!nf) throw new Error('Invalid NodeProto');
      const inputs: string[] = [];
      const outputs: string[] = [];
      const attrs: ParsedM1Attribute[] = [];
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
          const parsed = parseAttributeFull(m1Bytes, n.valStart, n.valEnd, dec);
          if (parsed) attrs.push(parsed);
        }
      }
      const node: ParsedM1NodeFull = {
        idx: nodes.length,
        name,
        opType,
        inputs,
        outputs,
        attrs,
        attrsSummary: attrs.map((a) => a.summary),
        rawPayloadSlice: m1Bytes.slice(f.valStart, f.valEnd),
      };
      nodes.push(node);
      for (const o of outputs) {
        if (o) producerByOutput.set(o, node);
      }
    } else if (f.fn === 5 && f.wt === 2) {
      const tf = parseProtoFieldsWithBounds(m1Bytes, f.valStart, f.valEnd);
      if (!tf) continue;
      const dims: number[] = [];
      let dataType = 0;
      let name = '';
      let dataLocation = 0;
      let rawSlice: Uint8Array | null = null;
      const floatDataInline: number[] = [];
      const int64DataInline: bigint[] = [];
      const externalKv: string[] = [];

      for (const t of tf) {
        if (t.fn === 1 && t.wt === 0) {
          const v = readSignedVarint64(m1Bytes, t.valStart, t.valEnd);
          if (v) dims.push(v[0]);
        } else if (t.fn === 2 && t.wt === 0) {
          const v = readVarint(m1Bytes, t.valStart, t.valEnd);
          if (v) dataType = v[0];
        } else if (t.fn === 4 && t.wt === 5 && t.valEnd - t.valStart === 4) {
          const dv = new DataView(m1Bytes.buffer, m1Bytes.byteOffset + t.valStart, 4);
          floatDataInline.push(dv.getFloat32(0, true));
        } else if (t.fn === 7 && t.wt === 0) {
          const v = readVarint64(m1Bytes, t.valStart, t.valEnd);
          if (v) int64DataInline.push(BigInt.asIntN(64, v[0]));
        } else if (t.fn === 8 && t.wt === 2 && t.valEnd > t.valStart) {
          name = dec.decode(m1Bytes.subarray(t.valStart, t.valEnd));
        } else if (t.fn === 9 && t.wt === 2) {
          rawSlice = m1Bytes.slice(t.valStart, t.valEnd);
        } else if (t.fn === 14 && t.wt === 0) {
          const v = readVarint(m1Bytes, t.valStart, t.valEnd);
          if (v) dataLocation = v[0];
        } else if (t.fn === 13 && t.wt === 2) {
          const kvf = parseProtoFieldsWithBounds(m1Bytes, t.valStart, t.valEnd);
          const kF = kvf?.find((x) => x.fn === 1 && x.wt === 2);
          const vF = kvf?.find((x) => x.fn === 2 && x.wt === 2);
          if (kF && vF) {
            externalKv.push(
              `${dec.decode(m1Bytes.subarray(kF.valStart, kF.valEnd))}=${dec.decode(m1Bytes.subarray(vF.valStart, vF.valEnd))}`
            );
          }
        }
      }
      if (!name) continue;
      initializerNames.add(name);
      const rawPayloadSlice = m1Bytes.slice(f.valStart, f.valEnd);
      const protoSha256 = await sha256HexBytes(rawPayloadSlice);
      const totalElements = dims.length === 0 ? 1 : dims.reduce((a, b) => a * b, 1);
      const isExternal = dataLocation === 1;
      let float32Values: Float32Array | null = null;
      let int64Values: BigInt64Array | null = null;
      let valueSha256: string | null = null;
      let blockerReason: string | null = null;

      if (isExternal) {
        blockerReason = `Initializer "${name}" uses external data_location=1 (${externalKv.join(', ') || 'no location kv'})`;
      } else if (dataType === 1) {
        if (rawSlice) {
          const outF32 = new Float32Array(totalElements);
          const dv = new DataView(rawSlice.buffer, rawSlice.byteOffset, rawSlice.byteLength);
          for (let i = 0; i < totalElements && (i + 1) * 4 <= rawSlice.byteLength; i++) {
            outF32[i] = dv.getFloat32(i * 4, true);
          }
          float32Values = outF32;
          valueSha256 = await sha256HexBytes(rawSlice);
        } else if (floatDataInline.length === totalElements) {
          float32Values = new Float32Array(floatDataInline);
          valueSha256 = await sha256HexBytes(
            new Uint8Array(float32Values.buffer, float32Values.byteOffset, float32Values.byteLength)
          );
        } else {
          blockerReason = `Float32 initializer "${name}" has neither raw_data (${totalElements * 4}B) nor matching float_data (${floatDataInline.length})`;
        }
      } else if (dataType === 7) {
        if (rawSlice) {
          const outI64 = new BigInt64Array(totalElements);
          const dv = new DataView(rawSlice.buffer, rawSlice.byteOffset, rawSlice.byteLength);
          for (let i = 0; i < totalElements && (i + 1) * 8 <= rawSlice.byteLength; i++) {
            outI64[i] = dv.getBigInt64(i * 8, true);
          }
          int64Values = outI64;
          valueSha256 = await sha256HexBytes(rawSlice);
        } else if (int64DataInline.length === totalElements) {
          int64Values = new BigInt64Array(int64DataInline);
          valueSha256 = await sha256HexBytes(
            new Uint8Array(int64Values.buffer, int64Values.byteOffset, int64Values.byteLength)
          );
        } else {
          blockerReason = `Int64 initializer "${name}" has neither raw_data (${totalElements * 8}B) nor matching int64_data (${int64DataInline.length})`;
        }
      } else if (rawSlice) {
        valueSha256 = await sha256HexBytes(rawSlice);
        blockerReason = `Initializer "${name}" has unsupported embedded dataType=${onnxDataTypeToName(dataType)} for direct JS reference`;
      } else {
        blockerReason = `Initializer "${name}" has unsupported dataType=${onnxDataTypeToName(dataType)} without raw_data`;
      }

      initializersByName.set(name, {
        name,
        dims,
        dataType,
        dataTypeName: onnxDataTypeToName(dataType),
        dataLocation,
        isExternal,
        externalInfo: externalKv.length > 0 ? externalKv.join(', ') : null,
        protoByteLength: rawPayloadSlice.byteLength,
        protoSha256,
        rawPayloadSlice,
        float32Values,
        int64Values,
        valueSha256,
        blockerReason,
      });
    } else if (f.fn === 11 && f.wt === 2) {
      const vf = parseProtoFieldsWithBounds(m1Bytes, f.valStart, f.valEnd);
      const nf = vf?.find((x) => x.fn === 1 && x.wt === 2 && x.valEnd > x.valStart);
      if (nf) graphInputNames.add(dec.decode(m1Bytes.subarray(nf.valStart, nf.valEnd)));
    } else if (f.fn === 12 && f.wt === 2) {
      const vf = parseProtoFieldsWithBounds(m1Bytes, f.valStart, f.valEnd);
      const nf = vf?.find((x) => x.fn === 1 && x.wt === 2 && x.valEnd > x.valStart);
      if (nf) graphOutputNames.add(dec.decode(m1Bytes.subarray(nf.valStart, nf.valEnd)));
    }
  }

  return {
    irVersion,
    opsetImportPayloads,
    nodes,
    initializersByName,
    initializerNames,
    graphInputNames,
    graphOutputNames,
    producerByOutput,
    staticMeta,
  };
}

export function resolveRgbStemSeedAndAncestorClosure(
  parsed: ParsedM1ModelFull,
  opts?: {
    seedOutputTensorOrSubstring?: string;
    seedNodeNameSubstring?: string;
  }
): AncestorClosureVerification {
  let seedNode: ParsedM1NodeFull | undefined;
  let selectionRule: string;

  if (opts?.seedOutputTensorOrSubstring) {
    const query = opts.seedOutputTensorOrSubstring;
    seedNode =
      parsed.nodes.find((n) => n.outputs.some((o) => o === query)) ??
      parsed.nodes.find((n) => n.outputs.some((o) => o.includes(query)));
    selectionRule = `explicit-output-match(${query})`;
  } else if (opts?.seedNodeNameSubstring) {
    const query = opts.seedNodeNameSubstring;
    seedNode = parsed.nodes.find((n) => n.name.includes(query));
    selectionRule = `explicit-node-match(${query})`;
  } else {
    // Earliest previously sampled differing RGB-stem node in prior tufa15 telemetry:
    // fe_shared/conv_down_1_1/BiasAdd;... (#11 Conv)
    seedNode = parsed.nodes.find(
      (n) =>
        n.opType === 'Conv' &&
        (n.name.includes('conv_down_1_1') || n.outputs.some((o) => o.includes('conv_down_1_1')))
    );
    selectionRule = 'earliest-prior-sampled-rgb-stem-node(conv_down_1_1)';
  }

  if (!seedNode || !seedNode.outputs[0]) {
    throw new Error(
      `Failed to locate RGB-stem seed node in ONNX graph (${selectionRule || 'conv_down_1_1'})`
    );
  }

  const visitedTensors = new Set<string>();
  const closureIdxSet = new Set<number>();
  const missingProducers: string[] = [];
  const queue: string[] = [seedNode.outputs[0]];

  while (queue.length > 0) {
    const t = queue.pop()!;
    if (!t || visitedTensors.has(t)) continue;
    visitedTensors.add(t);
    if (parsed.graphInputNames.has(t) || parsed.initializerNames.has(t)) {
      continue;
    }
    const prod = parsed.producerByOutput.get(t);
    if (!prod) {
      missingProducers.push(t);
      continue;
    }
    closureIdxSet.add(prod.idx);
    for (const inp of prod.inputs) {
      if (inp && !visitedTensors.has(inp)) {
        queue.push(inp);
      }
    }
  }

  if (missingProducers.length > 0) {
    throw new Error(
      `Ancestor closure contains missing non-initializer/non-input producers: ${missingProducers.join(', ')}`
    );
  }

  const closureNodes = parsed.nodes
    .filter((n) => closureIdxSet.has(n.idx))
    .sort((a, b) => a.idx - b.idx);

  const closureNodeIndices = closureNodes.map((n) => n.idx);
  const maxIdx = closureNodeIndices[closureNodeIndices.length - 1];
  const omittedInterleavedIndices: number[] = [];
  for (let i = 0; i <= maxIdx; i++) {
    if (!closureIdxSet.has(i)) {
      omittedInterleavedIndices.push(i);
    }
  }

  // Programmatic topological + ancestry verification:
  // 1) Deduplicated nodes (no repeated idx).
  // 2) Every dynamic input edge (parent -> child) satisfies parent.idx < child.idx.
  // 3) Every closure node (except seed) must actually be reachable from seed via closure edges.
  let topologicalOrderVerified = closureNodes.length === closureIdxSet.size;
  const reverseChildren = new Map<number, number[]>();
  for (const child of closureNodes) {
    for (const inp of child.inputs) {
      if (!inp || parsed.initializerNames.has(inp) || parsed.graphInputNames.has(inp)) continue;
      const parent = parsed.producerByOutput.get(inp);
      if (!parent || !closureIdxSet.has(parent.idx) || parent.idx >= child.idx) {
        topologicalOrderVerified = false;
      } else {
        const list = reverseChildren.get(parent.idx) ?? [];
        list.push(child.idx);
        reverseChildren.set(parent.idx, list);
      }
    }
  }

  let ancestryVerified = closureIdxSet.has(seedNode.idx);
  for (const n of closureNodes) {
    if (n.idx === seedNode.idx) continue;
    const seen = new Set<number>([n.idx]);
    const bfs = [n.idx];
    let reachesSeed = false;
    while (bfs.length > 0) {
      const cur = bfs.pop()!;
      if (cur === seedNode.idx) {
        reachesSeed = true;
        break;
      }
      for (const ch of reverseChildren.get(cur) ?? []) {
        if (!seen.has(ch)) {
          seen.add(ch);
          bfs.push(ch);
        }
      }
    }
    if (!reachesSeed) {
      ancestryVerified = false;
      break;
    }
  }

  if (!topologicalOrderVerified || !ancestryVerified) {
    throw new Error(
      `Ancestor closure verification failed (topological=${topologicalOrderVerified}, ancestry=${ancestryVerified})`
    );
  }

  const orderedNodes = closureNodes.map((n) => {
    const out0 = n.outputs[0];
    const meta = out0 ? parsed.staticMeta.tensorMeta[out0] : undefined;
    const initializerInputs: string[] = [];
    const dynamicInputs: string[] = [];
    for (const inp of n.inputs) {
      if (!inp) continue;
      if (parsed.initializerNames.has(inp)) {
        initializerInputs.push(inp);
      } else {
        dynamicInputs.push(inp);
      }
    }
    return {
      idx: n.idx,
      opType: n.opType,
      name: n.name,
      inputs: [...n.inputs],
      outputs: [...n.outputs],
      outputDims: meta ? [...meta.dims] : null,
      outputElemType: meta ? meta.elemType : null,
      attrsSummary: [...n.attrsSummary],
      initializerInputs,
      dynamicInputs,
    };
  });

  const tapBatches: AncestorClosureVerification['tapBatches'] = [];
  for (let i = 0; i < orderedNodes.length; i += 2) {
    const slice = orderedNodes.slice(i, i + 2);
    tapBatches.push({
      batchIndex: tapBatches.length + 1,
      nodeIndices: slice.map((x) => x.idx),
      tensorNames: slice.map((x) => x.outputs[0]),
    });
  }

  return {
    seedNodeIdx: seedNode.idx,
    seedNodeName: seedNode.name,
    seedOutputTensor: seedNode.outputs[0],
    selectionRule,
    totalClosureNodes: orderedNodes.length,
    closureNodeIndices,
    omittedInterleavedIndices,
    topologicalOrderVerified,
    ancestryVerified,
    missingProducers,
    tapBatches,
    orderedNodes,
  };
}

function flatIdxToCoord(flatIdx: number, dims: readonly number[]): number[] {
  if (dims.length === 0) return [flatIdx];
  const coord = new Array<number>(dims.length);
  let rem = flatIdx;
  for (let i = dims.length - 1; i >= 0; i--) {
    const d = dims[i] > 0 ? dims[i] : 1;
    coord[i] = rem % d;
    rem = Math.floor(rem / d);
  }
  return coord;
}

function u32ToHex(u: number): string {
  return `0x${(u >>> 0).toString(16).padStart(8, '0')}`;
}

function computeUlpDiff(aBits: number, bBits: number, aVal: number, bVal: number): number | null {
  if (!Number.isFinite(aVal) || !Number.isFinite(bVal)) return null;
  const aOrd = (aBits & 0x80000000) !== 0 ? 0x80000000 - (aBits & 0x7fffffff) : 0x80000000 + aBits;
  const bOrd = (bBits & 0x80000000) !== 0 ? 0x80000000 - (bBits & 0x7fffffff) : 0x80000000 + bBits;
  return Math.abs(aOrd - bOrd);
}

function isSignedZeroPairBits(aBits: number, bBits: number): boolean {
  const isZeroA = aBits === 0x00000000 || aBits === 0x80000000;
  const isZeroB = bBits === 0x00000000 || bBits === 0x80000000;
  return isZeroA && isZeroB && aBits !== bBits;
}

export function compareFloat32TensorsDetailed(
  cand: Float32Array,
  ref: Float32Array,
  dims: readonly number[]
): DetailedFloat32Comparison {
  if (cand.length !== ref.length) {
    throw new Error(`Float32 tensor length mismatch: cand=${cand.length}, ref=${ref.length}`);
  }
  const len = cand.length;
  const u32Cand = new Uint32Array(cand.buffer, cand.byteOffset, len);
  const u32Ref = new Uint32Array(ref.buffer, ref.byteOffset, len);

  let bitDiffCount = 0;
  let signedZeroOnlyCount = 0;
  let nonZeroNumericDiffCount = 0;
  let firstBitIdx = -1;
  let firstNonZeroIdx = -1;
  let maxAbsIdx = -1;
  let maxAbsObserved = -1;
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
  let nonZeroDiffTotal = 0;

  for (let i = 0; i < len; i++) {
    const cB = u32Cand[i];
    const rB = u32Ref[i];
    const cV = cand[i];
    const rV = ref[i];
    const cFin = Number.isFinite(cV);
    const rFin = Number.isFinite(rV);
    if (cFin) {
      finiteCount++;
      sumFinite += cV;
      if (cV < min) min = cV;
      if (cV > max) max = cV;
    }
    if (rFin) {
      refFiniteCount++;
      sumRefFinite += rV;
    }

    if (cB !== rB) {
      bitDiffCount++;
      if (firstBitIdx < 0) firstBitIdx = i;
      if (isSignedZeroPairBits(cB, rB)) {
        signedZeroOnlyCount++;
      } else {
        const isNaNPair = Number.isNaN(cV) && Number.isNaN(rV);
        if (!isNaNPair && cV !== rV) {
          nonZeroNumericDiffCount++;
          if (firstNonZeroIdx < 0) firstNonZeroIdx = i;
        }
      }
    }

    if (!cFin || !rFin) {
      if (!(Number.isNaN(cV) && Number.isNaN(rV)) && cV !== rV) {
        nonFiniteMismatchCount++;
      }
    } else if (cB !== rB) {
      const d = Math.abs(cV - rV);
      if (d > 0) {
        nonZeroDiffTotal++;
        sumAbsDiff += d;
        sqDiff += d * d;
        if (d > maxAbsDiff) {
          maxAbsDiff = d;
        }
        if (d > maxAbsObserved) {
          maxAbsObserved = d;
          maxAbsIdx = i;
        }
      }
    }
  }

  let p99AbsDiff = 0;
  if (len > 0 && nonZeroDiffTotal > 0) {
    const p99Idx = Math.min(len - 1, Math.floor(0.99 * (len - 1)));
    const zeroCount = len - nonZeroDiffTotal;
    if (p99Idx >= zeroCount) {
      const nzDiffs = new Float32Array(nonZeroDiffTotal);
      let w = 0;
      for (let i = 0; i < len && w < nonZeroDiffTotal; i++) {
        const cV = cand[i];
        const rV = ref[i];
        if (Number.isFinite(cV) && Number.isFinite(rV)) {
          const d = Math.abs(cV - rV);
          if (d > 0) nzDiffs[w++] = d;
        }
      }
      nzDiffs.sort();
      p99AbsDiff = nzDiffs[Math.min(nonZeroDiffTotal - 1, p99Idx - zeroCount)];
    }
  }

  const stats: RawTensorParityStats = {
    finiteCount,
    nonFiniteCount: len - finiteCount,
    length: len,
    min,
    max,
    mean: finiteCount > 0 ? sumFinite / finiteCount : 0,
    referenceMean: refFiniteCount > 0 ? sumRefFinite / refFiniteCount : 0,
    maxAbsDiff,
    meanAbsDiff: len > 0 ? sumAbsDiff / len : 0,
    rmsDiff: len > 0 ? Math.sqrt(sqDiff / len) : 0,
    p99AbsDiff,
    nonFiniteMismatchCount,
  };

  const passesGate =
    len > 0 &&
    stats.finiteCount === len &&
    stats.nonFiniteMismatchCount === 0 &&
    stats.maxAbsDiff < 1e-3 &&
    stats.p99AbsDiff < 1e-4 &&
    stats.meanAbsDiff < 1e-5;

  let classification: StemElementComparisonClassification;
  if (bitDiffCount === 0 && nonZeroNumericDiffCount === 0) {
    classification = 'BIT_IDENTICAL';
  } else if (nonZeroNumericDiffCount === 0 && stats.nonFiniteMismatchCount === 0) {
    classification = 'SIGNED_ZERO_ONLY_BIT_DIFF';
  } else if (passesGate) {
    classification = 'NONZERO_NUMERIC_SUBGATE';
  } else {
    classification = 'STRICT_GATE_BREACH';
  }

  const sampleAt = (idx: number): BitOrNumericLocationSample | null => {
    if (idx < 0) return null;
    const cV = cand[idx];
    const rV = ref[idx];
    const cB = u32Cand[idx];
    const rB = u32Ref[idx];
    return {
      flatIdx: idx,
      coord: flatIdxToCoord(idx, dims),
      candValue: cV,
      refValue: rV,
      candHex: u32ToHex(cB),
      refHex: u32ToHex(rB),
      absDiff: Math.abs(cV - rV),
      signedDiff: cV - rV,
      ulpDiff: computeUlpDiff(cB, rB, cV, rV),
      signedZeroPair: isSignedZeroPairBits(cB, rB),
    };
  };

  return {
    length: len,
    dims: [...dims],
    bitDiffCount,
    signedZeroOnlyCount,
    nonZeroNumericDiffCount,
    passesStrictGate: passesGate,
    classification,
    stats,
    firstBitDiff: sampleAt(firstBitIdx),
    firstNonZeroNumericDiff: sampleAt(firstNonZeroIdx),
    maxAbsDiffLocation: sampleAt(maxAbsIdx >= 0 ? maxAbsIdx : firstBitIdx),
  };
}

export function buildSingleOpCapturedInputModel(
  parsed: ParsedM1ModelFull,
  targetNode: ParsedM1NodeFull
): {
  bytes: Uint8Array;
  dynamicInputSpecs: Array<{ name: string; dims: number[]; elemType: number }>;
  initializerSpecs: ParsedM1InitializerInfo[];
  outputSpec: { name: string; dims: number[]; elemType: number };
} {
  const dynamicInputSpecs: Array<{ name: string; dims: number[]; elemType: number }> = [];
  const seenDynamic = new Set<string>();
  const initializerSpecs: ParsedM1InitializerInfo[] = [];
  const seenInit = new Set<string>();

  for (const inp of targetNode.inputs) {
    if (!inp) continue;
    if (parsed.initializerNames.has(inp)) {
      if (!seenInit.has(inp)) {
        seenInit.add(inp);
        const initInfo = parsed.initializersByName.get(inp);
        if (!initInfo) {
          throw new Error(`Missing parsed initializer "${inp}" for single-op submodel`);
        }
        if (initInfo.isExternal) {
          throw new Error(
            `Cannot build single-op submodel with external initializer "${inp}" (${initInfo.externalInfo || 'external'})`
          );
        }
        initializerSpecs.push(initInfo);
      }
    } else if (!seenDynamic.has(inp)) {
      seenDynamic.add(inp);
      const meta = parsed.staticMeta.tensorMeta[inp];
      if (!meta || meta.dims.some((d) => !Number.isInteger(d) || d <= 0)) {
        throw new Error(
          `Cannot build single-op submodel: missing or dynamic shape for input "${inp}"`
        );
      }
      if (meta.elemType !== 1) {
        throw new Error(
          `Single-op replay requires float32 dynamic inputs; "${inp}" has elemType=${meta.elemType}`
        );
      }
      dynamicInputSpecs.push({
        name: inp,
        dims: [...meta.dims],
        elemType: meta.elemType,
      });
    }
  }

  if (dynamicInputSpecs.length === 0) {
    throw new Error(`Target node "${targetNode.name}" has zero dynamic inputs`);
  }

  const outName = targetNode.outputs[0];
  if (!outName) {
    throw new Error(`Target node "${targetNode.name}" has no output tensor`);
  }
  const outMeta = parsed.staticMeta.tensorMeta[outName];
  if (!outMeta || outMeta.dims.some((d) => !Number.isInteger(d) || d <= 0)) {
    throw new Error(`Missing static output shape for "${outName}"`);
  }
  const outputSpec = {
    name: outName,
    dims: [...outMeta.dims],
    elemType: outMeta.elemType,
  };

  const graphParts: Uint8Array[] = [
    lenField(1, targetNode.rawPayloadSlice),
    strField(2, `hitnet_single_op_replay_node_${targetNode.idx}`),
    ...initializerSpecs.map((init) => lenField(5, init.rawPayloadSlice)),
    ...dynamicInputSpecs.map((inp) => makeValueInfo(inp.name, inp.elemType, inp.dims, 11)),
    makeValueInfo(outputSpec.name, outputSpec.elemType, outputSpec.dims, 12),
  ];

  const opsets =
    parsed.opsetImportPayloads.length > 0
      ? parsed.opsetImportPayloads.map((p) => lenField(8, p))
      : [lenField(8, concatBytes([varintField(2, 13)]))];

  const modelBytes = concatBytes([
    varintField(1, parsed.irVersion),
    ...opsets,
    lenField(7, concatBytes(graphParts)),
  ]);

  const checkMeta = inferOnnxStaticTensorMeta(modelBytes);
  if (
    !checkMeta.valid ||
    checkMeta.hasControlFlowNode ||
    checkMeta.missingInputs.length > 0 ||
    checkMeta.duplicateOutputs.length > 0
  ) {
    throw new Error(
      `Constructed single-op submodel failed static ONNX validation (valid=${checkMeta.valid}, missing=${JSON.stringify(checkMeta.missingInputs)})`
    );
  }

  return {
    bytes: modelBytes,
    dynamicInputSpecs,
    initializerSpecs,
    outputSpec,
  };
}

function getAttrInts(node: ParsedM1NodeFull, name: string, fallback: number[]): number[] {
  const a = node.attrs.find((x) => x.name === name);
  return a && a.intsVal.length > 0 ? a.intsVal : fallback;
}

function getAttrInt(node: ParsedM1NodeFull, name: string, fallback: number): number {
  const a = node.attrs.find((x) => x.name === name);
  return a && a.intVal !== null ? a.intVal : fallback;
}

function getAttrFloat(node: ParsedM1NodeFull, name: string, fallback: number): number {
  const a = node.attrs.find((x) => x.name === name);
  return a && a.floatVal !== null ? a.floatVal : fallback;
}

function computeStrides(dims: readonly number[]): number[] {
  const s = new Array<number>(dims.length);
  let acc = 1;
  for (let i = dims.length - 1; i >= 0; i--) {
    s[i] = acc;
    acc *= dims[i];
  }
  return s;
}

function evaluateElementSequentialF64(
  parsed: ParsedM1ModelFull,
  node: ParsedM1NodeFull,
  dynamicInputs: Record<string, { dims: number[]; data: Float32Array }>,
  outDims: readonly number[],
  flatIdx: number
): { val: number | null; blocker: string | null } {
  const inName0 = node.inputs[0];
  const getArrayAndDims = (name: string): { dims: number[]; data: Float32Array } | null => {
    if (dynamicInputs[name]) return dynamicInputs[name];
    const init = parsed.initializersByName.get(name);
    if (init?.float32Values) return { dims: init.dims, data: init.float32Values };
    return null;
  };
  const readBroadcast = (
    src: { dims: number[]; data: Float32Array },
    coord: readonly number[],
    targetDims: readonly number[]
  ): number => {
    const offset = targetDims.length - src.dims.length;
    let idx = 0;
    const strides = computeStrides(src.dims);
    for (let i = 0; i < src.dims.length; i++) {
      const sd = src.dims[i];
      const c = coord[i + offset] ?? 0;
      idx += (sd === 1 ? 0 : c) * strides[i];
    }
    return src.dims.length === 0 ? src.data[0] : src.data[idx];
  };

  if (node.opType === 'Conv') {
    const xSpec = dynamicInputs[inName0];
    const wInit = node.inputs[1] ? parsed.initializersByName.get(node.inputs[1]) : undefined;
    const bInit = node.inputs[2] ? parsed.initializersByName.get(node.inputs[2]) : undefined;
    if (!xSpec || xSpec.dims.length !== 4 || !wInit?.float32Values || wInit.dims.length !== 4) {
      return { val: null, blocker: 'Conv reference requires 4D NCHW float32 input and weight initializer' };
    }
    const [N, C, H, W] = xSpec.dims;
    const [M, CPerG, kH, kW] = wInit.dims;
    const group = getAttrInt(node, 'group', 1);
    if (group * CPerG !== C || M % group !== 0 || N <= 0) {
      return { val: null, blocker: `Invalid Conv group=${group} for C=${C}, CPerG=${CPerG}, M=${M}` };
    }
    const autoPad = node.attrs.find((a) => a.name === 'auto_pad');
    if (autoPad && autoPad.summary !== 'NOTSET') {
      // If auto_pad string isn't handled, check pads presence
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

    const coord = flatIdxToCoord(flatIdx, outDims);
    const n = coord[0];
    const m = coord[1];
    const oh = coord[2];
    const ow = coord[3];
    const gSize = M / group;
    const g = Math.floor(m / gSize);
    const cStart = g * CPerG;
    const xData = xSpec.data;
    const wData = wInit.float32Values;
    const xBaseBatch = n * C * H * W;
    const wBaseFilter = m * CPerG * kH * kW;

    let acc = 0.0;
    for (let ic = 0; ic < CPerG; ic++) {
      const cIdx = cStart + ic;
      const xChanOff = xBaseBatch + cIdx * H * W;
      const wChanOff = wBaseFilter + ic * kH * kW;
      for (let kh = 0; kh < kH; kh++) {
        const ih = oh * strideH + kh * dilH - padTop;
        if (ih < 0 || ih >= H) continue;
        const xRowOff = xChanOff + ih * W;
        const wRowOff = wChanOff + kh * kW;
        for (let kw = 0; kw < kW; kw++) {
          const iw = ow * strideW + kw * dilW - padLeft;
          if (iw < 0 || iw >= W) continue;
          acc += Number(xData[xRowOff + iw]) * Number(wData[wRowOff + kw]);
        }
      }
    }
    if (bInit?.float32Values) {
      acc += Number(bInit.float32Values[m]);
    }
    return { val: acc, blocker: null };
  }

  if (node.opType === 'Mul' || node.opType === 'Sub' || node.opType === 'Add') {
    const a = getArrayAndDims(node.inputs[0]);
    const b = getArrayAndDims(node.inputs[1]);
    if (!a || !b) {
      return { val: null, blocker: `${node.opType} inputs unavailable as float32 tensors` };
    }
    const coord = flatIdxToCoord(flatIdx, outDims);
    const va = readBroadcast(a, coord, outDims);
    const vb = readBroadcast(b, coord, outDims);
    const val =
      node.opType === 'Mul'
        ? va * vb
        : node.opType === 'Sub'
          ? va - vb
          : va + vb;
    return { val, blocker: null };
  }

  if (node.opType === 'LeakyRelu') {
    const x = getArrayAndDims(node.inputs[0]);
    if (!x) return { val: null, blocker: 'LeakyRelu input unavailable' };
    const alpha = getAttrFloat(node, 'alpha', 0.01);
    const v = Number(x.data[flatIdx]);
    return { val: v >= 0 ? v : v * alpha, blocker: null };
  }

  if (node.opType === 'Concat') {
    const axisRaw = getAttrInt(node, 'axis', 0);
    const axis = axisRaw < 0 ? outDims.length + axisRaw : axisRaw;
    const coord = flatIdxToCoord(flatIdx, outDims);
    let axCoord = coord[axis];
    for (const inpName of node.inputs) {
      const inp = getArrayAndDims(inpName);
      if (!inp) return { val: null, blocker: `Concat input "${inpName}" unavailable` };
      const dimAx = inp.dims[axis];
      if (axCoord < dimAx) {
        const cIn = [...coord];
        cIn[axis] = axCoord;
        const strides = computeStrides(inp.dims);
        let inIdx = 0;
        for (let i = 0; i < cIn.length; i++) inIdx += cIn[i] * strides[i];
        return { val: Number(inp.data[inIdx]), blocker: null };
      }
      axCoord -= dimAx;
    }
    return { val: null, blocker: 'Concat index out of bounds' };
  }

  if (node.opType === 'Slice') {
    const x = getArrayAndDims(node.inputs[0]);
    const startsI = node.inputs[1] ? parsed.initializersByName.get(node.inputs[1])?.int64Values : null;
    const endsI = node.inputs[2] ? parsed.initializersByName.get(node.inputs[2])?.int64Values : null;
    const axesI = node.inputs[3] ? parsed.initializersByName.get(node.inputs[3])?.int64Values : null;
    const stepsI = node.inputs[4] ? parsed.initializersByName.get(node.inputs[4])?.int64Values : null;
    if (!x || !startsI || !endsI) {
      return { val: null, blocker: 'Slice reference requires float32 data and int64 constant starts/ends initializers' };
    }
    const rank = x.dims.length;
    const coord = flatIdxToCoord(flatIdx, outDims);
    const inCoord = [...coord];
    for (let k = 0; k < startsI.length; k++) {
      const axRaw = axesI ? Number(axesI[k]) : k;
      const ax = axRaw < 0 ? rank + axRaw : axRaw;
      const dimLen = x.dims[ax];
      const step = stepsI ? Number(stepsI[k]) : 1;
      let start = Number(startsI[k]);
      if (start < 0) start += dimLen;
      start = Math.max(0, Math.min(dimLen, start));
      inCoord[ax] = start + coord[ax] * step;
    }
    const strides = computeStrides(x.dims);
    let inIdx = 0;
    for (let i = 0; i < rank; i++) inIdx += inCoord[i] * strides[i];
    return { val: Number(x.data[inIdx]), blocker: null };
  }

  return {
    val: null,
    blocker: `No sequential float64 reference implemented for opType="${node.opType}"`,
  };
}

export function computeSequentialFloat64Reference(
  parsed: ParsedM1ModelFull,
  node: ParsedM1NodeFull,
  dynamicInputs: Record<string, { dims: number[]; data: Float32Array }>,
  outDims: readonly number[],
  gpuOut: Float32Array,
  wasmOut: Float32Array,
  priorityIndices: readonly number[] = []
): SequentialFloat64RefResult {
  const elemCount = outDims.reduce((a, b) => a * b, 1);
  let perElemWork = 1;
  if (node.opType === 'Conv') {
    const wInit = node.inputs[1] ? parsed.initializersByName.get(node.inputs[1]) : undefined;
    const group = getAttrInt(node, 'group', 1);
    const xSpec = dynamicInputs[node.inputs[0]];
    if (wInit && wInit.dims.length === 4 && xSpec && xSpec.dims.length === 4) {
      perElemWork = wInit.dims[1] * wInit.dims[2] * wInit.dims[3];
      if (group <= 0) perElemWork = 1;
    }
  }
  const totalScalarWork = elemCount * perElemWork;
  const FULL_TENSOR_WORK_BUDGET = 10_000_000;
  const useFullTensor = totalScalarWork <= FULL_TENSOR_WORK_BUDGET;

  let indicesToEval: number[];
  if (useFullTensor) {
    indicesToEval = new Array<number>(elemCount);
    for (let i = 0; i < elemCount; i++) indicesToEval[i] = i;
  } else {
    const picked = new Set<number>();
    for (const idx of priorityIndices) {
      if (idx >= 0 && idx < elemCount) picked.add(idx);
    }
    // Also capture top diverging coordinates across the entire tensor
    const topDiv: Array<{ idx: number; diff: number }> = [];
    const maxTrack = 64;
    for (let i = 0; i < elemCount; i++) {
      const d = Math.abs(gpuOut[i] - wasmOut[i]);
      if (d > 0 && (topDiv.length < maxTrack || d > topDiv[topDiv.length - 1].diff)) {
        topDiv.push({ idx: i, diff: d });
        topDiv.sort((a, b) => b.diff - a.diff);
        if (topDiv.length > maxTrack) topDiv.pop();
      }
    }
    for (const td of topDiv) picked.add(td.idx);
    const stride = Math.max(1, Math.floor(elemCount / 1024));
    for (let i = 0; i < elemCount && picked.size < 2048; i += stride) {
      picked.add(i);
    }
    indicesToEval = Array.from(picked).sort((a, b) => a - b);
  }

  if (indicesToEval.length === 0) {
    return {
      available: false,
      mode: 'unavailable',
      opType: node.opType,
      elementCount: elemCount,
      evaluatedCount: 0,
      scalarWorkEstimate: totalScalarWork,
      blockerReason: 'No output elements to evaluate',
      maxAbsGpuVsF64: null,
      meanAbsGpuVsF64: null,
      maxAbsWasmVsF64: null,
      meanAbsWasmVsF64: null,
      gpuCloserToF64Count: 0,
      wasmCloserToF64Count: 0,
      equalDistToF64Count: 0,
      sampleCoords: [],
    };
  }

  let maxGpuErr = 0;
  let sumGpuErr = 0;
  let maxWasmErr = 0;
  let sumWasmErr = 0;
  let gpuCloser = 0;
  let wasmCloser = 0;
  let equalDist = 0;

  const topSamples: Array<{
    score: number;
    entry: SequentialFloat64RefResult['sampleCoords'][number];
  }> = [];
  const f32RoundBuf = new Float32Array(1);

  for (const idx of indicesToEval) {
    const res = evaluateElementSequentialF64(parsed, node, dynamicInputs, outDims, idx);
    if (res.val === null || res.blocker) {
      return {
        available: false,
        mode: 'unavailable',
        opType: node.opType,
        elementCount: elemCount,
        evaluatedCount: 0,
        scalarWorkEstimate: totalScalarWork,
        blockerReason: res.blocker ?? 'float64 reference evaluation failed',
        maxAbsGpuVsF64: null,
        meanAbsGpuVsF64: null,
        maxAbsWasmVsF64: null,
        meanAbsWasmVsF64: null,
        gpuCloserToF64Count: 0,
        wasmCloserToF64Count: 0,
        equalDistToF64Count: 0,
        sampleCoords: [],
      };
    }
    const f64Ref = res.val;
    f32RoundBuf[0] = f64Ref;
    const f64RoundedF32 = f32RoundBuf[0];
    const g = gpuOut[idx];
    const w = wasmOut[idx];
    const gErr = Math.abs(g - f64Ref);
    const wErr = Math.abs(w - f64Ref);
    if (gErr > maxGpuErr) maxGpuErr = gErr;
    if (wErr > maxWasmErr) maxWasmErr = wErr;
    sumGpuErr += gErr;
    sumWasmErr += wErr;
    if (gErr < wErr) gpuCloser++;
    else if (wErr < gErr) wasmCloser++;
    else equalDist++;

    const gwDiff = Math.abs(g - w);
    const isPriority = priorityIndices.includes(idx);
    if (
      isPriority ||
      topSamples.length < 8 ||
      gwDiff > topSamples[topSamples.length - 1].score
    ) {
      topSamples.push({
        score: gwDiff + (isPriority ? 1e6 : 0),
        entry: {
          flatIdx: idx,
          coord: flatIdxToCoord(idx, outDims),
          f64Ref,
          f64RoundedF32,
          gpu: g,
          wasm: w,
          gpuAbsErrFromF64: gErr,
          wasmAbsErrFromF64: wErr,
        },
      });
      topSamples.sort((a, b) => b.score - a.score);
      if (topSamples.length > 12) topSamples.pop();
    }
  }

  return {
    available: true,
    mode: useFullTensor ? 'full-tensor' : 'sampled-coords',
    opType: node.opType,
    elementCount: elemCount,
    evaluatedCount: indicesToEval.length,
    scalarWorkEstimate: totalScalarWork,
    blockerReason: null,
    maxAbsGpuVsF64: maxGpuErr,
    meanAbsGpuVsF64: sumGpuErr / indicesToEval.length,
    maxAbsWasmVsF64: maxWasmErr,
    meanAbsWasmVsF64: sumWasmErr / indicesToEval.length,
    gpuCloserToF64Count: gpuCloser,
    wasmCloserToF64Count: wasmCloser,
    equalDistToF64Count: equalDist,
    sampleCoords: topSamples.map((s) => s.entry),
  };
}

export async function runHitnetRgbStemFirstNonZeroDiagnostic(params: {
  m1Bytes: Uint8Array;
  f0102FeedFloat32: Float32Array;
  f0126FeedFloat32?: Float32Array | null;
  confirmF0126Candidate?: boolean;
  seedOutputTensorOrSubstring?: string;
  runWorker?: ResidualDiagnosticWorkerRunner;
  disableCpuEpFallback?: boolean;
  isDev?: boolean;
  onLog?: (line: string) => void;
}): Promise<RgbStemFirstNonZeroReport> {
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
      `Exact M1 verification failed: bytes=${params.m1Bytes.byteLength} (expected ${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength}), sha=${m1Sha} (expected ${HITNET_M1_FULL_GRAPH_SHA256})`
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
  const closure = resolveRgbStemSeedAndAncestorClosure(parsedModel, {
    seedOutputTensorOrSubstring: params.seedOutputTensorOrSubstring,
  });

  const asyncifyProfile = HITNET_ORT_130_FAMILY_PROFILES.asyncify;
  log(
    `=== [DEV-ONLY] HITNet RGB-Stem First Nonzero Localization & Single-Op Replay ===\n` +
    `Runtime: onnxruntime-web@1.30.0/webgpu (non-JSEP ${asyncifyProfile.entryFile} + ${asyncifyProfile.wasmFile}) | opt=disabled\n` +
    `M1: bytes=${params.m1Bytes.byteLength} sha256=${m1Sha}\n` +
    `F0102: float32 [1,6,240,320] sha256=${f0102Sha}${f0126Sha ? ` | F0126 sha256=${f0126Sha}` : ''}\n` +
    `Seed: [#${closure.seedNodeIdx}] ${closure.orderedNodes[closure.orderedNodes.length - 1]?.opType} "${closure.seedNodeName}" -> "${closure.seedOutputTensor}" (rule=${closure.selectionRule})\n` +
    `Verified Ancestor Closure (${closure.totalClosureNodes} nodes, indices=${JSON.stringify(closure.closureNodeIndices)}, omittedInterleaved=${JSON.stringify(closure.omittedInterleavedIndices)}, topoOk=${closure.topologicalOrderVerified}, ancestryOk=${closure.ancestryVerified}):`
  );
  for (const n of closure.orderedNodes) {
    log(
      `  [#${n.idx}] ${n.opType} "${n.name}" | in=${JSON.stringify(n.inputs)} -> out="${n.outputs[0]}" ${JSON.stringify(n.outputDims)} ${onnxDataTypeToName(n.outputElemType ?? 0)} | attrs=${JSON.stringify(n.attrsSummary)}`
    );
  }

  const assertNonJsepSuccess = async (
    label: string,
    rawRes: OrtVersionDiagnosticResponse,
    expectedModelSha: string,
    expectedModelBytes: number,
    expectedFeedSha: string
  ): Promise<OrtVersionDiagnosticSuccess> => {
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
        `${label} model hash/length mismatch: got=${rawRes.modelSha256}(${rawRes.modelByteLength}B), expected=${expectedModelSha}(${expectedModelBytes}B)`
      );
    }
    if (rawRes.feedSha256 !== expectedFeedSha) {
      throw new Error(
        `${label} feed SHA mismatch: got=${rawRes.feedSha256}, expected=${expectedFeedSha}`
      );
    }
    return rawRes;
  };

  log(`\n--- Step 1: F0102 Untapped M1 Control (WebGPU vs WASM, opt=disabled) ---`);
  const [untappedGpuRaw, untappedWasmRaw] = await Promise.all([
    runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: params.m1Bytes.slice().buffer,
      feedBuffer: params.f0102FeedFloat32.slice().buffer,
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
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
      modelBuffer: params.m1Bytes.slice().buffer,
      feedBuffer: params.f0102FeedFloat32.slice().buffer,
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      provider: 'wasm',
      graphOptimizationLevel: 'disabled',
      warmRuns: 0,
      enableWebGpuProfiling: false,
      disableCpuEpFallback: false,
      captureGpuValidationErrors: false,
    }),
  ]);

  const untappedGpu = await assertNonJsepSuccess(
    'F0102 Untapped M1 WebGPU(disabled)',
    untappedGpuRaw,
    m1Sha,
    params.m1Bytes.byteLength,
    f0102Sha
  );
  const untappedWasm = await assertNonJsepSuccess(
    'F0102 Untapped M1 WASM(disabled)',
    untappedWasmRaw,
    m1Sha,
    params.m1Bytes.byteLength,
    f0102Sha
  );

  const untappedGpuVal = validateHitnetFinalOutputs(untappedGpu.outputs);
  const untappedWasmVal = validateHitnetFinalOutputs(untappedWasm.outputs);
  if (!untappedGpuVal.ok || !untappedWasmVal.ok) {
    throw new Error(
      `Untapped F0102 final validation failed: gpu=${untappedGpuVal.error ?? 'ok'}, wasm=${untappedWasmVal.error ?? 'ok'}`
    );
  }

  const [pName, sName] = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;
  const untappedPrimary = compareFloat32TensorsDetailed(
    untappedGpuVal.byName[pName].data,
    untappedWasmVal.byName[pName].data,
    untappedGpuVal.byName[pName].dims
  );
  const untappedSecondary = compareFloat32TensorsDetailed(
    untappedGpuVal.byName[sName].data,
    untappedWasmVal.byName[sName].data,
    untappedGpuVal.byName[sName].dims
  );

  const gpuProfileUntapped = await runWorker({
    expectedOrtVersion: '1.30.0',
    runtimeAssetFamily: 'asyncify',
    modelBuffer: params.m1Bytes.slice().buffer,
    feedBuffer: params.f0102FeedFloat32.slice().buffer,
    feedName: 'input',
    feedDims: [1, 6, 240, 320],
    provider: 'webgpu',
    graphOptimizationLevel: 'disabled',
    warmRuns: 0,
    enableWebGpuProfiling: true,
    disableCpuEpFallback,
    captureGpuValidationErrors: true,
  });

  log(
    `  [Untapped F0102 opt=disabled] primary(${untappedPrimary.classification}): max=${untappedPrimary.stats.maxAbsDiff.toExponential(3)} p99=${untappedPrimary.stats.p99AbsDiff.toExponential(3)} mean=${untappedPrimary.stats.meanAbsDiff.toExponential(3)} | ` +
    `secondary(${untappedSecondary.classification}): max=${untappedSecondary.stats.maxAbsDiff.toExponential(3)}\n` +
    `  [Untapped Non-Timing GPU Profile] ${gpuProfileUntapped.ok ? JSON.stringify(gpuProfileUntapped.gpuKernelProfileSummary) : `FAILED(${gpuProfileUntapped.stage}): ${gpuProfileUntapped.message}`}`
  );

  log(`\n--- Step 2: Serial Ancestor-Closure Tap Sweep (<=2 outputs/run, bit-identical finals required) ---`);
  const capturedByTensor = new Map<
    string,
    {
      dims: number[];
      dataType: string;
      gpuData: Float32Array;
      wasmData: Float32Array;
      gpuSha256: string;
      wasmSha256: string;
      comparison: DetailedFloat32Comparison;
    }
  >();

  // Record graph input feed as exact verified baseline tensor
  capturedByTensor.set('input', {
    dims: [1, 6, 240, 320],
    dataType: 'float32',
    gpuData: params.f0102FeedFloat32.slice(),
    wasmData: params.f0102FeedFloat32.slice(),
    gpuSha256: f0102Sha,
    wasmSha256: f0102Sha,
    comparison: compareFloat32TensorsDetailed(
      params.f0102FeedFloat32,
      params.f0102FeedFloat32,
      [1, 6, 240, 320]
    ),
  });

  const tapBatchSummaries: RgbStemFirstNonZeroReport['tapBatchSummaries'] = [];
  const nodeObservations: ClosureNodeTapObservation[] = [];
  let abortedByFinalPerturbation = false;
  let perturbationStopMessage: string | null = null;

  for (const batch of closure.tapBatches) {
    const tapped = appendOnnxGraphOutputs(params.m1Bytes, batch.tensorNames, { maxExtraOutputs: 2 });
    const tappedSha = await sha256HexBytes(tapped.bytes);
    if (tappedSha === m1Sha) {
      throw new Error(`Tapped model SHA unexpectedly matches base M1 SHA at batch #${batch.batchIndex}`);
    }

    const gpuRaw = await runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: tapped.bytes.slice().buffer,
      feedBuffer: params.f0102FeedFloat32.slice().buffer,
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      provider: 'webgpu',
      graphOptimizationLevel: 'disabled',
      warmRuns: 0,
      enableWebGpuProfiling: false,
      disableCpuEpFallback,
      captureGpuValidationErrors: true,
    });
    const wasmRaw = await runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: tapped.bytes.slice().buffer,
      feedBuffer: params.f0102FeedFloat32.slice().buffer,
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      provider: 'wasm',
      graphOptimizationLevel: 'disabled',
      warmRuns: 0,
      enableWebGpuProfiling: false,
      disableCpuEpFallback: false,
      captureGpuValidationErrors: false,
    });

    const gpuTap = await assertNonJsepSuccess(
      `Closure Batch #${batch.batchIndex} WebGPU`,
      gpuRaw,
      tappedSha,
      tapped.bytes.byteLength,
      f0102Sha
    );
    const wasmTap = await assertNonJsepSuccess(
      `Closure Batch #${batch.batchIndex} WASM`,
      wasmRaw,
      tappedSha,
      tapped.bytes.byteLength,
      f0102Sha
    );

    const gpuVal = validateHitnetFinalOutputs(gpuTap.outputs);
    const wasmVal = validateHitnetFinalOutputs(wasmTap.outputs);
    if (!gpuVal.ok || !wasmVal.ok) {
      throw new Error(
        `Closure Batch #${batch.batchIndex} final output contract error: gpu=${gpuVal.error ?? 'ok'}, wasm=${wasmVal.error ?? 'ok'}`
      );
    }

    const gpuPrimaryBitIdentical = checkFloat32BitIdentity(
      gpuVal.byName[pName].data,
      untappedGpuVal.byName[pName].data
    );
    const gpuSecondaryBitIdentical = checkFloat32BitIdentity(
      gpuVal.byName[sName].data,
      untappedGpuVal.byName[sName].data
    );
    const wasmPrimaryBitIdentical = checkFloat32BitIdentity(
      wasmVal.byName[pName].data,
      untappedWasmVal.byName[pName].data
    );
    const wasmSecondaryBitIdentical = checkFloat32BitIdentity(
      wasmVal.byName[sName].data,
      untappedWasmVal.byName[sName].data
    );

    const allBitIdentical =
      gpuPrimaryBitIdentical.bitIdentical &&
      gpuSecondaryBitIdentical.bitIdentical &&
      wasmPrimaryBitIdentical.bitIdentical &&
      wasmSecondaryBitIdentical.bitIdentical;

    let perturbationDetail: string | null = null;
    if (!allBitIdentical) {
      perturbationDetail =
        `Closure Batch #${batch.batchIndex} perturbed final outputs: ` +
        `gpuP=${gpuPrimaryBitIdentical.mismatchCount}, gpuS=${gpuSecondaryBitIdentical.mismatchCount}, ` +
        `wasmP=${wasmPrimaryBitIdentical.mismatchCount}, wasmS=${wasmSecondaryBitIdentical.mismatchCount}`;
    }

    tapBatchSummaries.push({
      batchIndex: batch.batchIndex,
      tapOutputs: [...batch.tensorNames],
      tappedModelSha256: tappedSha,
      tappedModelByteLength: tapped.bytes.byteLength,
      bitIdenticalFinals: allBitIdentical,
      perturbationDetail,
      finalInvariance: {
        gpuPrimaryBitIdentical,
        gpuSecondaryBitIdentical,
        wasmPrimaryBitIdentical,
        wasmSecondaryBitIdentical,
      },
    });

    if (!allBitIdentical) {
      abortedByFinalPerturbation = true;
      perturbationStopMessage = perturbationDetail;
      log(`  [STOP: TAP_PERTURBED_FINALS @ Batch #${batch.batchIndex}] ${perturbationDetail}`);
      break;
    }

    for (let k = 0; k < batch.nodeIndices.length; k++) {
      const nIdx = batch.nodeIndices[k];
      const tName = batch.tensorNames[k];
      const nodeObj = parsedModel.nodes[nIdx];
      const gOut = gpuTap.outputs.find((o) => o.name === tName);
      const wOut = wasmTap.outputs.find((o) => o.name === tName);
      if (!gOut || !wOut) {
        throw new Error(`Missing tapped output "${tName}" at node #${nIdx}`);
      }
      if (gOut.dataType !== 'float32' || wOut.dataType !== 'float32') {
        throw new Error(
          `Expected float32 tap output for "${tName}" (#${nIdx}), got gpu=${gOut.dataType}, wasm=${wOut.dataType}`
        );
      }
      if (JSON.stringify(gOut.dims) !== JSON.stringify(wOut.dims)) {
        throw new Error(
          `Tap dims mismatch for "${tName}" (#${nIdx}): gpu=${JSON.stringify(gOut.dims)}, wasm=${JSON.stringify(wOut.dims)}`
        );
      }
      const gpuData = new Float32Array(gOut.buffer).slice();
      const wasmData = new Float32Array(wOut.buffer).slice();
      const gpuSha256 = await sha256HexBytes(
        new Uint8Array(gpuData.buffer, gpuData.byteOffset, gpuData.byteLength)
      );
      const wasmSha256 = await sha256HexBytes(
        new Uint8Array(wasmData.buffer, wasmData.byteOffset, wasmData.byteLength)
      );
      const comparison = compareFloat32TensorsDetailed(gpuData, wasmData, gOut.dims);

      capturedByTensor.set(tName, {
        dims: [...gOut.dims],
        dataType: gOut.dataType,
        gpuData,
        wasmData,
        gpuSha256,
        wasmSha256,
        comparison,
      });

      const immediateInputs: ImmediateInputVerification[] = [];
      const inputBlockers: string[] = [];

      for (const inpName of nodeObj.inputs) {
        if (!inpName) {
          immediateInputs.push({
            inputName: '',
            kind: 'empty-optional',
            dims: null,
            elemTypeName: null,
            verifiedIdentity: true,
            identityStatus: 'OPTIONAL_EMPTY',
            sha256: null,
            bitDiffCount: null,
            signedZeroOnlyCount: null,
            nonZeroNumericDiffCount: null,
            maxAbsDiff: null,
            blockerReason: null,
          });
          continue;
        }
        const init = parsedModel.initializersByName.get(inpName);
        if (init) {
          if (init.isExternal) {
            const msg = init.blockerReason ?? `Initializer "${inpName}" is external`;
            inputBlockers.push(msg);
            immediateInputs.push({
              inputName: inpName,
              kind: 'initializer',
              dims: [...init.dims],
              elemTypeName: init.dataTypeName,
              verifiedIdentity: false,
              identityStatus: 'BLOCKER_EXTERNAL_INITIALIZER',
              sha256: init.protoSha256,
              bitDiffCount: null,
              signedZeroOnlyCount: null,
              nonZeroNumericDiffCount: null,
              maxAbsDiff: null,
              blockerReason: msg,
            });
          } else {
            const status = init.float32Values || init.int64Values
              ? 'VERIFIED_SHARED_MODEL_INITIALIZER'
              : 'BLOCKER_UNSUPPORTED_INITIALIZER_DTYPE';
            if (status !== 'VERIFIED_SHARED_MODEL_INITIALIZER' && init.blockerReason) {
              inputBlockers.push(init.blockerReason);
            }
            immediateInputs.push({
              inputName: inpName,
              kind: 'initializer',
              dims: [...init.dims],
              elemTypeName: init.dataTypeName,
              verifiedIdentity: status === 'VERIFIED_SHARED_MODEL_INITIALIZER',
              identityStatus: status,
              sha256: init.valueSha256 ?? init.protoSha256,
              bitDiffCount: 0,
              signedZeroOnlyCount: 0,
              nonZeroNumericDiffCount: 0,
              maxAbsDiff: 0,
              blockerReason: status === 'VERIFIED_SHARED_MODEL_INITIALIZER' ? null : init.blockerReason,
            });
          }
          continue;
        }

        const cap = capturedByTensor.get(inpName);
        if (cap) {
          const isFeed = parsedModel.graphInputNames.has(inpName);
          const c = cap.comparison;
          let idStatus: ImmediateInputVerification['identityStatus'];
          if (c.classification === 'BIT_IDENTICAL') {
            idStatus = isFeed ? 'VERIFIED_FEED_SHA' : 'OBSERVED_BIT_IDENTICAL';
          } else if (c.classification === 'SIGNED_ZERO_ONLY_BIT_DIFF') {
            idStatus = 'OBSERVED_SIGNED_ZERO_ONLY';
          } else {
            idStatus = 'OBSERVED_NONZERO_DIFF';
          }
          immediateInputs.push({
            inputName: inpName,
            kind: isFeed ? 'graph-input' : 'dynamic-tensor',
            dims: [...cap.dims],
            elemTypeName: cap.dataType,
            verifiedIdentity: c.classification === 'BIT_IDENTICAL',
            identityStatus: idStatus,
            sha256: cap.wasmSha256,
            bitDiffCount: c.bitDiffCount,
            signedZeroOnlyCount: c.signedZeroOnlyCount,
            nonZeroNumericDiffCount: c.nonZeroNumericDiffCount,
            maxAbsDiff: c.stats.maxAbsDiff,
            blockerReason: null,
          });
        } else {
          const meta = parsedModel.staticMeta.tensorMeta[inpName];
          const msg = `Missing captured runtime observation for dynamic parent "${inpName}"`;
          inputBlockers.push(msg);
          immediateInputs.push({
            inputName: inpName,
            kind: parsedModel.producerByOutput.has(inpName) ? 'dynamic-tensor' : 'missing',
            dims: meta ? [...meta.dims] : null,
            elemTypeName: meta ? onnxDataTypeToName(meta.elemType) : null,
            verifiedIdentity: false,
            identityStatus: 'BLOCKER_MISSING_OBSERVATION',
            sha256: null,
            bitDiffCount: null,
            signedZeroOnlyCount: null,
            nonZeroNumericDiffCount: null,
            maxAbsDiff: null,
            blockerReason: msg,
          });
        }
      }

      const allImmediateInputsByteEqual =
        inputBlockers.length === 0 &&
        immediateInputs.every(
          (ii) => ii.kind === 'empty-optional' || ii.verifiedIdentity
        );

      const obs: ClosureNodeTapObservation = {
        nodeIdx: nIdx,
        opType: nodeObj.opType,
        nodeName: nodeObj.name,
        outputTensor: tName,
        dims: [...gOut.dims],
        dataType: gOut.dataType,
        batchIndex: batch.batchIndex,
        tappedModelSha256: tappedSha,
        tappedModelByteLength: tapped.bytes.byteLength,
        gpuBufferSha256: gpuSha256,
        wasmBufferSha256: wasmSha256,
        comparison,
        immediateInputs,
        allImmediateInputsByteEqual,
        inputBlockers,
      };
      nodeObservations.push(obs);

      log(
        `  [Node #${nIdx} ${nodeObj.opType}] "${nodeObj.name}" -> "${tName}" ${JSON.stringify(gOut.dims)} | class=${comparison.classification} | ` +
        `bitDiffs=${comparison.bitDiffCount} (signedZero=${comparison.signedZeroOnlyCount}, nonZero=${comparison.nonZeroNumericDiffCount}) | ` +
        `max=${comparison.stats.maxAbsDiff.toExponential(3)} p99=${comparison.stats.p99AbsDiff.toExponential(3)} mean=${comparison.stats.meanAbsDiff.toExponential(3)} | ` +
        `parentsByteEq=${allImmediateInputsByteEqual}${inputBlockers.length > 0 ? ` blockers=${JSON.stringify(inputBlockers)}` : ''}`
      );
    }
  }

  // Topological order is guaranteed by nodeObservations order (closure.orderedNodes idx asc)
  const earliestBitDiffNode =
    nodeObservations.find((o) => o.comparison.bitDiffCount > 0) ?? null;
  const earliestSignedZeroOnlyNode =
    nodeObservations.find((o) => o.comparison.classification === 'SIGNED_ZERO_ONLY_BIT_DIFF') ?? null;
  const earliestNonZeroNumericNode =
    nodeObservations.find((o) => o.comparison.nonZeroNumericDiffCount > 0 || o.comparison.stats.nonFiniteMismatchCount > 0) ?? null;
  const earliestStrictGateBreachNode =
    nodeObservations.find((o) => o.comparison.classification === 'STRICT_GATE_BREACH') ?? null;

  const negativeControlsBitIdentical = nodeObservations
    .filter((o) => o.comparison.classification === 'BIT_IDENTICAL')
    .map((o) => ({
      nodeIdx: o.nodeIdx,
      opType: o.opType,
      nodeName: o.nodeName,
      outputTensor: o.outputTensor,
      classification: o.comparison.classification,
    }));

  // Step 3: Single-op captured-input replay on earliest nonzero candidate (if any & unblocked)
  let singleOpReplay: SingleOpReplayReport | null = null;
  if (!abortedByFinalPerturbation && earliestNonZeroNumericNode) {
    const targetNode = parsedModel.nodes[earliestNonZeroNumericNode.nodeIdx];
    log(
      `\n--- Step 3: One Valid Single-Op Captured-Input Replay ([Node #${targetNode.idx}] ${targetNode.opType} "${targetNode.name}") ---`
    );

    const dynamicInputsForRef: Record<string, { dims: number[]; data: Float32Array }> = {};
    const capturedFeedSpecs: SingleOpReplayReport['capturedFeedSpecs'] = [];
    const replayBlockers: string[] = [...earliestNonZeroNumericNode.inputBlockers];

    for (const ii of earliestNonZeroNumericNode.immediateInputs) {
      if (ii.kind === 'graph-input' || ii.kind === 'dynamic-tensor') {
        const cap = capturedByTensor.get(ii.inputName);
        if (!cap) {
          replayBlockers.push(`Missing captured tensor "${ii.inputName}"`);
          continue;
        }
        const identicalBytes = cap.gpuSha256 === cap.wasmSha256 && cap.comparison.bitDiffCount === 0;
        if (!identicalBytes) {
          replayBlockers.push(
            `Parent dynamic tensor "${ii.inputName}" is NOT byte-equal across WebGPU & WASM (${cap.comparison.classification}, bitDiffs=${cap.comparison.bitDiffCount}, nonZero=${cap.comparison.nonZeroNumericDiffCount}, max=${cap.comparison.stats.maxAbsDiff.toExponential(3)})`
          );
        }
        capturedFeedSpecs.push({
          name: ii.inputName,
          dims: [...cap.dims],
          byteLength: cap.wasmData.byteLength,
          gpuCapturedSha256: cap.gpuSha256,
          wasmCapturedSha256: cap.wasmSha256,
          identicalCapturedBytes: identicalBytes,
        });
        dynamicInputsForRef[ii.inputName] = { dims: [...cap.dims], data: cap.wasmData };
      }
    }

    if (replayBlockers.length > 0) {
      singleOpReplay = {
        executed: false,
        blockerReason: replayBlockers.join('; '),
        targetNodeIdx: targetNode.idx,
        targetOpType: targetNode.opType,
        targetNodeName: targetNode.name,
        targetOutputTensor: earliestNonZeroNumericNode.outputTensor,
        submodelByteLength: null,
        submodelSha256: null,
        capturedFeedSpecs,
        initializerSpecs: [],
        webgpuReplaySuccess: null,
        wasmReplaySuccess: null,
        webgpuProfileRun: null,
        webgpuKernelDispatchObserved: false,
        replayGpuVsReplayWasm: null,
        replayGpuVsFullM1Gpu: null,
        replayWasmVsFullM1Wasm: null,
        reproducesFullM1GpuBits: null,
        reproducesFullM1WasmBits: null,
        reproducesFullM1Diff: null,
        float64Reference: null,
        interpretationNote: `Single-op replay blocked cleanly: ${replayBlockers.join('; ')}`,
      };
      log(`  [Replay BLOCKED] ${singleOpReplay.blockerReason}`);
    } else {
      try {
        const sub = buildSingleOpCapturedInputModel(parsedModel, targetNode);
        const subSha = await sha256HexBytes(sub.bytes);
        const primaryDyn = sub.dynamicInputSpecs[0];
        const primaryCap = capturedByTensor.get(primaryDyn.name)!;
        const primaryFeedBytes = new Uint8Array(
          primaryCap.wasmData.buffer,
          primaryCap.wasmData.byteOffset,
          primaryCap.wasmData.byteLength
        );
        const primaryFeedSha = await sha256HexBytes(primaryFeedBytes);
        const extraFeeds = sub.dynamicInputSpecs.slice(1).map((spec) => {
          const c = capturedByTensor.get(spec.name)!;
          return {
            name: spec.name,
            dims: [...spec.dims],
            buffer: c.wasmData.slice().buffer as ArrayBuffer,
          };
        });

        const [repGpuRaw, repWasmRaw] = await Promise.all([
          runWorker({
            expectedOrtVersion: '1.30.0',
            runtimeAssetFamily: 'asyncify',
            modelBuffer: sub.bytes.slice().buffer,
            feedBuffer: primaryCap.wasmData.slice().buffer,
            feedName: primaryDyn.name,
            feedDims: [...primaryDyn.dims],
            extraFeeds,
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
            feedBuffer: primaryCap.wasmData.slice().buffer,
            feedName: primaryDyn.name,
            feedDims: [...primaryDyn.dims],
            extraFeeds: extraFeeds.map((ef) => ({
              name: ef.name,
              dims: [...ef.dims],
              buffer: ef.buffer.slice(0),
            })),
            allowArbitraryOutputs: true,
            provider: 'wasm',
            graphOptimizationLevel: 'disabled',
            warmRuns: 0,
            enableWebGpuProfiling: false,
            disableCpuEpFallback: false,
            captureGpuValidationErrors: false,
          }),
        ]);

        const repGpu = await assertNonJsepSuccess(
          'Single-Op Replay WebGPU(disabled)',
          repGpuRaw,
          subSha,
          sub.bytes.byteLength,
          primaryFeedSha
        );
        const repWasm = await assertNonJsepSuccess(
          'Single-Op Replay WASM(disabled)',
          repWasmRaw,
          subSha,
          sub.bytes.byteLength,
          primaryFeedSha
        );

        const webgpuProfileRun = await runWorker({
          expectedOrtVersion: '1.30.0',
          runtimeAssetFamily: 'asyncify',
          modelBuffer: sub.bytes.slice().buffer,
          feedBuffer: primaryCap.wasmData.slice().buffer,
          feedName: primaryDyn.name,
          feedDims: [...primaryDyn.dims],
          extraFeeds: extraFeeds.map((ef) => ({
            name: ef.name,
            dims: [...ef.dims],
            buffer: ef.buffer.slice(0),
          })),
          allowArbitraryOutputs: true,
          provider: 'webgpu',
          graphOptimizationLevel: 'disabled',
          warmRuns: 0,
          enableWebGpuProfiling: true,
          disableCpuEpFallback,
          captureGpuValidationErrors: true,
        });

        const dispatchSummary = webgpuProfileRun.ok ? webgpuProfileRun.gpuKernelProfileSummary : null;
        const webgpuKernelDispatchObserved = Boolean(
          dispatchSummary &&
            (dispatchSummary.webgpuComputePassCount > 0 ||
              dispatchSummary.webgpuDispatchWorkgroupsCount > 0 ||
              dispatchSummary.ortProfilingEventCount > 0)
        );

        const gOut: OrtVersionDiagnosticOutput | undefined =
          repGpu.outputs.find((o) => o.name === sub.outputSpec.name) ?? repGpu.outputs[0];
        const wOut: OrtVersionDiagnosticOutput | undefined =
          repWasm.outputs.find((o) => o.name === sub.outputSpec.name) ?? repWasm.outputs[0];
        if (!gOut || !wOut) {
          throw new Error(`Missing output "${sub.outputSpec.name}" from single-op replay`);
        }

        const repGpuF32 = new Float32Array(gOut.buffer);
        const repWasmF32 = new Float32Array(wOut.buffer);
        const m1Captured = capturedByTensor.get(earliestNonZeroNumericNode.outputTensor)!;

        const replayGpuVsReplayWasm = compareFloat32TensorsDetailed(repGpuF32, repWasmF32, gOut.dims);
        const replayGpuVsFullM1Gpu = compareFloat32TensorsDetailed(
          repGpuF32,
          m1Captured.gpuData,
          gOut.dims
        );
        const replayWasmVsFullM1Wasm = compareFloat32TensorsDetailed(
          repWasmF32,
          m1Captured.wasmData,
          wOut.dims
        );

        const priorityIndices: number[] = [];
        if (replayGpuVsReplayWasm.firstNonZeroNumericDiff) {
          priorityIndices.push(replayGpuVsReplayWasm.firstNonZeroNumericDiff.flatIdx);
        }
        if (replayGpuVsReplayWasm.maxAbsDiffLocation) {
          priorityIndices.push(replayGpuVsReplayWasm.maxAbsDiffLocation.flatIdx);
        }
        if (earliestNonZeroNumericNode.comparison.firstNonZeroNumericDiff) {
          priorityIndices.push(earliestNonZeroNumericNode.comparison.firstNonZeroNumericDiff.flatIdx);
        }
        if (earliestNonZeroNumericNode.comparison.maxAbsDiffLocation) {
          priorityIndices.push(earliestNonZeroNumericNode.comparison.maxAbsDiffLocation.flatIdx);
        }

        const float64Reference = computeSequentialFloat64Reference(
          parsedModel,
          targetNode,
          dynamicInputsForRef,
          gOut.dims,
          repGpuF32,
          repWasmF32,
          priorityIndices
        );

        const reproducesFullM1GpuBits = replayGpuVsFullM1Gpu.bitDiffCount === 0;
        const reproducesFullM1WasmBits = replayWasmVsFullM1Wasm.bitDiffCount === 0;
        const reproducesFullM1Diff =
          reproducesFullM1GpuBits &&
          reproducesFullM1WasmBits &&
          replayGpuVsReplayWasm.nonZeroNumericDiffCount > 0;

        let interpretationNote = '';
        if (!webgpuKernelDispatchObserved) {
          interpretationNote =
            'Single-op WebGPU replay did not show WebGPU compute passes / profiling events (possible internal CPU placement or unattached observer).';
        } else if (!reproducesFullM1GpuBits || !reproducesFullM1WasmBits) {
          interpretationNote =
            `Single-op replay outputs differ from full-M1 outputs at "${sub.outputSpec.name}" ` +
            `(gpuReplayVsM1 bitDiffs=${replayGpuVsFullM1Gpu.bitDiffCount}, wasmReplayVsM1 bitDiffs=${replayWasmVsFullM1Wasm.bitDiffCount}); ` +
            'different kernel/layout selection or graph context may apply between single-op and full-M1.';
        } else if (reproducesFullM1Diff) {
          interpretationNote =
            `Isolated single-op replay bit-identically reproduces BOTH full-M1 WebGPU and full-M1 WASM outputs ` +
            `(maxAbsDiff=${replayGpuVsReplayWasm.stats.maxAbsDiff.toExponential(3)}, bitDiffs=${replayGpuVsReplayWasm.bitDiffCount}) with verified WebGPU dispatch.`;
        } else {
          interpretationNote =
            `Single-op replay matches full-M1 bits without divergence (classification=${replayGpuVsReplayWasm.classification}).`;
        }

        singleOpReplay = {
          executed: true,
          blockerReason: null,
          targetNodeIdx: targetNode.idx,
          targetOpType: targetNode.opType,
          targetNodeName: targetNode.name,
          targetOutputTensor: sub.outputSpec.name,
          submodelByteLength: sub.bytes.byteLength,
          submodelSha256: subSha,
          capturedFeedSpecs,
          initializerSpecs: sub.initializerSpecs.map((init) => ({
            name: init.name,
            dims: [...init.dims],
            dataTypeName: init.dataTypeName,
            protoByteLength: init.protoByteLength,
            protoSha256: init.protoSha256,
            valueSha256: init.valueSha256,
          })),
          webgpuReplaySuccess: repGpu,
          wasmReplaySuccess: repWasm,
          webgpuProfileRun,
          webgpuKernelDispatchObserved,
          replayGpuVsReplayWasm,
          replayGpuVsFullM1Gpu,
          replayWasmVsFullM1Wasm,
          reproducesFullM1GpuBits,
          reproducesFullM1WasmBits,
          reproducesFullM1Diff,
          float64Reference,
          interpretationNote,
        };

        log(
          `  [Replay submodel] bytes=${sub.bytes.byteLength} sha=${subSha.slice(0, 12)}... | dispatchObserved=${webgpuKernelDispatchObserved}\n` +
          `  [Replay WebGPU vs WASM] ${replayGpuVsReplayWasm.classification} | bitDiffs=${replayGpuVsReplayWasm.bitDiffCount} nonZero=${replayGpuVsReplayWasm.nonZeroNumericDiffCount} | max=${replayGpuVsReplayWasm.stats.maxAbsDiff.toExponential(3)} p99=${replayGpuVsReplayWasm.stats.p99AbsDiff.toExponential(3)} mean=${replayGpuVsReplayWasm.stats.meanAbsDiff.toExponential(3)}\n` +
          `  [Replay vs Full-M1] gpuBitDiffs=${replayGpuVsFullM1Gpu.bitDiffCount} (max=${replayGpuVsFullM1Gpu.stats.maxAbsDiff.toExponential(3)}) | wasmBitDiffs=${replayWasmVsFullM1Wasm.bitDiffCount} (max=${replayWasmVsFullM1Wasm.stats.maxAbsDiff.toExponential(3)})\n` +
          `  [Float64 Sequential Ref] ${float64Reference.available ? `mode=${float64Reference.mode} eval=${float64Reference.evaluatedCount}/${float64Reference.elementCount} maxGpuErr=${float64Reference.maxAbsGpuVsF64?.toExponential(3)} maxWasmErr=${float64Reference.maxAbsWasmVsF64?.toExponential(3)} closer(gpu=${float64Reference.gpuCloserToF64Count}, wasm=${float64Reference.wasmCloserToF64Count}, eq=${float64Reference.equalDistToF64Count})` : `UNAVAILABLE(${float64Reference.blockerReason})`}\n` +
          `  [Dispatch Profile] ${dispatchSummary ? JSON.stringify({ passes: dispatchSummary.webgpuComputePassCount, dispatch: dispatchSummary.webgpuDispatchWorkgroupsCount, submits: dispatchSummary.webgpuQueueSubmitCount, ortEvents: dispatchSummary.ortProfilingEventCount, samplePrograms: dispatchSummary.ortProfilingSamplePrograms }) : 'none'}\n` +
          `  [Note] ${interpretationNote}`
        );
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        singleOpReplay = {
          executed: false,
          blockerReason: msg,
          targetNodeIdx: targetNode.idx,
          targetOpType: targetNode.opType,
          targetNodeName: targetNode.name,
          targetOutputTensor: earliestNonZeroNumericNode.outputTensor,
          submodelByteLength: null,
          submodelSha256: null,
          capturedFeedSpecs,
          initializerSpecs: [],
          webgpuReplaySuccess: null,
          wasmReplaySuccess: null,
          webgpuProfileRun: null,
          webgpuKernelDispatchObserved: false,
          replayGpuVsReplayWasm: null,
          replayGpuVsFullM1Gpu: null,
          replayWasmVsFullM1Wasm: null,
          reproducesFullM1GpuBits: null,
          reproducesFullM1WasmBits: null,
          reproducesFullM1Diff: null,
          float64Reference: null,
          interpretationNote: `Single-op replay errored: ${msg}`,
        };
        log(`  [Replay ERROR] ${msg}`);
      }
    }
  }

  // Step 4: Optional F0126 candidate confirmation (untapped + single pair tap [parent, target])
  let f0126Confirmation: F0126ConfirmationSummary | null = null;
  if (
    !abortedByFinalPerturbation &&
    params.f0126FeedFloat32 &&
    f0126Sha &&
    (params.confirmF0126Candidate ?? true)
  ) {
    log(`\n--- Step 4: F0126 Confirmation (opt=disabled, untapped + localized candidate pair) ---`);
    const [f2GpuRaw, f2WasmRaw] = await Promise.all([
      runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: params.m1Bytes.slice().buffer,
        feedBuffer: params.f0126FeedFloat32.slice().buffer,
        feedName: 'input',
        feedDims: [1, 6, 240, 320],
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
        modelBuffer: params.m1Bytes.slice().buffer,
        feedBuffer: params.f0126FeedFloat32.slice().buffer,
        feedName: 'input',
        feedDims: [1, 6, 240, 320],
        provider: 'wasm',
        graphOptimizationLevel: 'disabled',
        warmRuns: 0,
        enableWebGpuProfiling: false,
        disableCpuEpFallback: false,
        captureGpuValidationErrors: false,
      }),
    ]);

    const f2Gpu = await assertNonJsepSuccess(
      'F0126 Untapped M1 WebGPU(disabled)',
      f2GpuRaw,
      m1Sha,
      params.m1Bytes.byteLength,
      f0126Sha
    );
    const f2Wasm = await assertNonJsepSuccess(
      'F0126 Untapped M1 WASM(disabled)',
      f2WasmRaw,
      m1Sha,
      params.m1Bytes.byteLength,
      f0126Sha
    );
    const f2GpuVal = validateHitnetFinalOutputs(f2Gpu.outputs);
    const f2WasmVal = validateHitnetFinalOutputs(f2Wasm.outputs);
    if (!f2GpuVal.ok || !f2WasmVal.ok) {
      throw new Error(`F0126 untapped final validation failed`);
    }
    const f2Prim = summarizeRawTensorAndParity(f2GpuVal.byName[pName].data, f2WasmVal.byName[pName].data);
    const f2Sec = summarizeRawTensorAndParity(f2GpuVal.byName[sName].data, f2WasmVal.byName[sName].data);

    let candidateTap: F0126ConfirmationSummary['candidateTap'] = null;
    if (earliestNonZeroNumericNode) {
      const parentDyn = parsedModel.nodes[earliestNonZeroNumericNode.nodeIdx].inputs.find(
        (inp) => inp && !parsedModel.initializerNames.has(inp) && !parsedModel.graphInputNames.has(inp)
      ) ?? null;
      const f2TapOutputs = parentDyn
        ? [parentDyn, earliestNonZeroNumericNode.outputTensor]
        : [earliestNonZeroNumericNode.outputTensor];
      const f2Tapped = appendOnnxGraphOutputs(params.m1Bytes, f2TapOutputs, { maxExtraOutputs: 2 });
      const f2TappedSha = await sha256HexBytes(f2Tapped.bytes);

      const f2TapGpuRaw = await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: f2Tapped.bytes.slice().buffer,
        feedBuffer: params.f0126FeedFloat32.slice().buffer,
        feedName: 'input',
        feedDims: [1, 6, 240, 320],
        provider: 'webgpu',
        graphOptimizationLevel: 'disabled',
        warmRuns: 0,
        enableWebGpuProfiling: false,
        disableCpuEpFallback,
        captureGpuValidationErrors: true,
      });
      const f2TapWasmRaw = await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: f2Tapped.bytes.slice().buffer,
        feedBuffer: params.f0126FeedFloat32.slice().buffer,
        feedName: 'input',
        feedDims: [1, 6, 240, 320],
        provider: 'wasm',
        graphOptimizationLevel: 'disabled',
        warmRuns: 0,
        enableWebGpuProfiling: false,
        disableCpuEpFallback: false,
        captureGpuValidationErrors: false,
      });

      const f2TapGpu = await assertNonJsepSuccess(
        'F0126 Candidate Tap WebGPU',
        f2TapGpuRaw,
        f2TappedSha,
        f2Tapped.bytes.byteLength,
        f0126Sha
      );
      const f2TapWasm = await assertNonJsepSuccess(
        'F0126 Candidate Tap WASM',
        f2TapWasmRaw,
        f2TappedSha,
        f2Tapped.bytes.byteLength,
        f0126Sha
      );
      const f2TapGpuVal = validateHitnetFinalOutputs(f2TapGpu.outputs);
      const f2TapWasmVal = validateHitnetFinalOutputs(f2TapWasm.outputs);
      const f2BitIdentical =
        f2TapGpuVal.ok &&
        f2TapWasmVal.ok &&
        checkFloat32BitIdentity(f2TapGpuVal.byName[pName].data, f2GpuVal.byName[pName].data).bitIdentical &&
        checkFloat32BitIdentity(f2TapGpuVal.byName[sName].data, f2GpuVal.byName[sName].data).bitIdentical &&
        checkFloat32BitIdentity(f2TapWasmVal.byName[pName].data, f2WasmVal.byName[pName].data).bitIdentical &&
        checkFloat32BitIdentity(f2TapWasmVal.byName[sName].data, f2WasmVal.byName[sName].data).bitIdentical;

      let parentComparison: DetailedFloat32Comparison | null = null;
      let targetComparison: DetailedFloat32Comparison | null = null;
      if (f2BitIdentical) {
        if (parentDyn) {
          const pg = f2TapGpu.outputs.find((o) => o.name === parentDyn);
          const pw = f2TapWasm.outputs.find((o) => o.name === parentDyn);
          if (pg && pw) {
            parentComparison = compareFloat32TensorsDetailed(
              new Float32Array(pg.buffer),
              new Float32Array(pw.buffer),
              pg.dims
            );
          }
        }
        const tg = f2TapGpu.outputs.find((o) => o.name === earliestNonZeroNumericNode.outputTensor);
        const tw = f2TapWasm.outputs.find((o) => o.name === earliestNonZeroNumericNode.outputTensor);
        if (tg && tw) {
          targetComparison = compareFloat32TensorsDetailed(
            new Float32Array(tg.buffer),
            new Float32Array(tw.buffer),
            tg.dims
          );
        }
      }

      candidateTap = {
        targetNodeIdx: earliestNonZeroNumericNode.nodeIdx,
        targetOutputTensor: earliestNonZeroNumericNode.outputTensor,
        parentTensor: parentDyn,
        tappedModelSha256: f2TappedSha,
        bitIdenticalFinals: f2BitIdentical,
        perturbationDetail: f2BitIdentical ? null : 'F0126 candidate tap perturbed final outputs',
        parentComparison,
        targetComparison,
      };
      log(
        `  [F0126 Confirmation] untappedPrimaryMax=${f2Prim.maxAbsDiff.toExponential(3)} | ` +
        `tapBitIdFinals=${f2BitIdentical} | ` +
        `parent=${parentComparison ? `${parentComparison.classification}(bitDiff=${parentComparison.bitDiffCount}, max=${parentComparison.stats.maxAbsDiff.toExponential(3)})` : 'graph-input/none'} | ` +
        `target(#${earliestNonZeroNumericNode.nodeIdx})=${targetComparison ? `${targetComparison.classification}(bitDiff=${targetComparison.bitDiffCount}, nonZero=${targetComparison.nonZeroNumericDiffCount}, max=${targetComparison.stats.maxAbsDiff.toExponential(3)})` : 'n/a'}`
      );
    }

    f0126Confirmation = {
      executed: true,
      feedSha256: f0126Sha,
      untappedGpuVsWasmPrimary: f2Prim,
      untappedGpuVsWasmSecondary: f2Sec,
      candidateTap,
    };
  }

  log(
    `\n=== Summary (F0102 opt=disabled, non-JSEP ORT 1.30.0) ===\n` +
    `  earliestBitDiffNode=${earliestBitDiffNode ? `[#${earliestBitDiffNode.nodeIdx}] ${earliestBitDiffNode.opType} "${earliestBitDiffNode.outputTensor}" (${earliestBitDiffNode.comparison.classification}, bitDiffs=${earliestBitDiffNode.comparison.bitDiffCount})` : 'none'}\n` +
    `  earliestSignedZeroOnlyNode=${earliestSignedZeroOnlyNode ? `[#${earliestSignedZeroOnlyNode.nodeIdx}] ${earliestSignedZeroOnlyNode.opType} "${earliestSignedZeroOnlyNode.outputTensor}" (signedZero=${earliestSignedZeroOnlyNode.comparison.signedZeroOnlyCount})` : 'none'}\n` +
    `  earliestNonZeroNumericNode=${earliestNonZeroNumericNode ? `[#${earliestNonZeroNumericNode.nodeIdx}] ${earliestNonZeroNumericNode.opType} "${earliestNonZeroNumericNode.outputTensor}" (max=${earliestNonZeroNumericNode.comparison.stats.maxAbsDiff.toExponential(3)}, p99=${earliestNonZeroNumericNode.comparison.stats.p99AbsDiff.toExponential(3)}, mean=${earliestNonZeroNumericNode.comparison.stats.meanAbsDiff.toExponential(3)})` : 'none'}\n` +
    `  earliestStrictGateBreachNode=${earliestStrictGateBreachNode ? `[#${earliestStrictGateBreachNode.nodeIdx}] ${earliestStrictGateBreachNode.opType} "${earliestStrictGateBreachNode.outputTensor}"` : 'none-within-closure'}\n` +
    `  negativeControls(BIT_IDENTICAL)=${JSON.stringify(negativeControlsBitIdentical.map((c) => `#${c.nodeIdx}:${c.opType}`))}\n` +
    `  singleOpReplay=${singleOpReplay ? (singleOpReplay.executed ? `EXECUTED(reproducesFullM1Diff=${singleOpReplay.reproducesFullM1Diff}, dispatchObserved=${singleOpReplay.webgpuKernelDispatchObserved})` : `BLOCKED(${singleOpReplay.blockerReason})`) : 'skipped'}`
  );

  return {
    ok: true,
    abortReason: abortedByFinalPerturbation ? perturbationStopMessage : null,
    frame: 'F0102',
    pinnedOrtVersion: '1.30.0',
    runtimeAssetFamily: 'asyncify',
    graphOptimizationLevel: 'disabled',
    m1ByteLength: params.m1Bytes.byteLength,
    m1Sha256: m1Sha,
    f0102FeedSha256: f0102Sha,
    closure,
    untappedBaseline: {
      gpuModelSha256: untappedGpu.modelSha256,
      wasmModelSha256: untappedWasm.modelSha256,
      gpuFeedSha256: untappedGpu.feedSha256,
      wasmFeedSha256: untappedWasm.feedSha256,
      primary: untappedPrimary,
      secondary: untappedSecondary,
      gpuDispatchProfile: gpuProfileUntapped,
    },
    tapBatchSummaries,
    nodeObservations,
    negativeControlsBitIdentical,
    earliestBitDiffNode,
    earliestSignedZeroOnlyNode,
    earliestNonZeroNumericNode,
    earliestStrictGateBreachNode,
    singleOpReplay,
    f0126Confirmation,
  };
}
