import {
  HITNET_ORT_130_FAMILY_PROFILES,
  type OrtGpuKernelProfileSummary,
  type OrtVersionDiagnosticFailure,
  type OrtVersionDiagnosticResponse,
  type OrtVersionDiagnosticSuccess,
} from './hitnetOrtComparisonHelper';
import {
  createBrowserOrt130DiagnosticRunner,
  sha256HexBytes,
  validateHitnetFinalOutputs,
  type ResidualDiagnosticWorkerRunner,
} from './hitnetResidualLocalization';
import {
  verifyCanonicalM1NoN1Replacements,
  type N1OffVerificationProof,
} from './hitnetPostConv7TapReplay';
import { parseM1ModelProtoWithRawSlices } from './hitnetRgbStemFirstNonZero';
import {
  assertDevFixtureEnabled,
  HITNET_DEV_NON_JSEP_FIXTURE_SPEC,
} from './stereoDepthWebGpuDevFixture';
import { HITNET_M1_FULL_GRAPH_SHA256 } from './stereoDepthWorkerCore';
import { summarizeRawTensorAndParity, type RawTensorParityStats } from './hitnetGatherNdDiagnostic';

export const HITNET_SYNTHETIC_GT_CONSTANTS = {
  modelWidth: 320,
  modelHeight: 240,
  borderPadPx: 16,
  maxCommonDisparityPx: 32,
  seed: 0x48495431, // "HIT1" deterministic fixed seed
  disparities: [16, 32] as const,
} as const;

export type SyntheticGroundTruthCaseId = 'D16' | 'D32';

export interface StereoRoiRect {
  xStart: number;
  xEndExclusive: number;
  yStart: number;
  yEndExclusive: number;
  pixelCount: number;
}

export interface SyntheticGroundTruthPair {
  caseId: SyntheticGroundTruthCaseId;
  disparityTruePx: 16 | 32;
  seed: number;
  width: 320;
  height: 240;
  channels: 6;
  feedDims: [1, 6, 240, 320];
  feedFloat32: Float32Array;
  feedSha256: string;
  leftRgba: Uint8ClampedArray;
  rightRgba: Uint8ClampedArray;
  signConventionNote: string;
  analyticValidMask: {
    primaryLeft: StereoRoiRect;
    secondaryRight: StereoRoiRect;
  };
  commonCroppedRoi: {
    primaryLeft: StereoRoiRect;
    secondaryRight: StereoRoiRect;
  };
}

export interface GroundTruthRegionMetricsPx {
  roiLabel: string;
  roi: StereoRoiRect;
  disparityTruePx: number;
  evaluatedCount: number;
  finiteCount: number;
  nonFiniteCount: number;
  invalidCount: number; // !isFinite || d < 0 (zero prediction-dependent exclusion from error metrics)
  invalidRate: number;
  epePx: number;
  p95AbsErrorPx: number;
  maxAbsErrorPx: number;
  bad1Count: number;
  bad1Rate: number;
  bad3Count: number;
  bad3Rate: number;
  meanPredPx: number;
  medianPredPx: number;
  minPredPx: number;
  maxPredPx: number;
}

export interface SyntheticGtArmExecutionResult {
  provider: 'webgpu' | 'wasm';
  graphOptimizationLevel: 'basic' | 'disabled';
  modelSha256: string;
  modelByteLength: number;
  feedSha256: string;
  profilingOffTimings: {
    createMs: number;
    coldRunMs: number;
    warmRunMs: number[];
    warmMedianMs: number | null;
  };
  dispatchCheckProfilingOn: {
    executed: boolean;
    dispatchObserved: boolean;
    summary: OrtGpuKernelProfileSummary | null;
  } | null;
  workerGpuAdapter: string | null;
  gpuValidationErrors: string[];
  primary: {
    analyticValid: GroundTruthRegionMetricsPx;
    commonRoi: GroundTruthRegionMetricsPx;
  };
  secondary: {
    analyticValid: GroundTruthRegionMetricsPx;
    commonRoi: GroundTruthRegionMetricsPx;
  };
}

export interface SyntheticGtCaseReport {
  caseId: SyntheticGroundTruthCaseId;
  disparityTruePx: 16 | 32;
  feedSha256: string;
  feedByteLength: number;
  signConventionNote: string;
  analyticValidMask: SyntheticGroundTruthPair['analyticValidMask'];
  commonCroppedRoi: SyntheticGroundTruthPair['commonCroppedRoi'];
  webgpu: SyntheticGtArmExecutionResult;
  wasm: SyntheticGtArmExecutionResult;
  webgpuVsWasmFullFrameParity: {
    primary: RawTensorParityStats;
    secondary: RawTensorParityStats;
  };
}

export interface HitnetSyntheticGroundTruthReport {
  ok: boolean;
  pinnedOrtVersion: '1.30.0';
  runtimeAssetFamily: 'asyncify';
  graphOptimizationLevel: 'basic' | 'disabled';
  warmRuns: number;
  m1ByteLength: number;
  m1Sha256: string;
  n1OffProof: N1OffVerificationProof;
  coordinateUnitsNote: string;
  cases: Record<SyntheticGroundTruthCaseId, SyntheticGtCaseReport>;
  scientificScopeNote: string;
}

function createMulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function buildDeterministicRgbTextureUint8(opts?: {
  sceneWidth?: number;
  height?: number;
  seed?: number;
}): { sceneWidth: number; height: number; rgb: Uint8Array } {
  const sceneWidth = opts?.sceneWidth ?? HITNET_SYNTHETIC_GT_CONSTANTS.modelWidth + HITNET_SYNTHETIC_GT_CONSTANTS.maxCommonDisparityPx;
  const height = opts?.height ?? HITNET_SYNTHETIC_GT_CONSTANTS.modelHeight;
  const seed = opts?.seed ?? HITNET_SYNTHETIC_GT_CONSTANTS.seed;
  const rand = createMulberry32(seed);

  // Coarse 8x8 tile + medium 4x4 + fine 1x1 + diagonal grid/dot landmarks using pure integer PRNG
  // so every browser & Node produces exact identical uint8 texture bytes with zero Math.sin libm variance.
  const coarseW = Math.ceil(sceneWidth / 8);
  const coarseH = Math.ceil(height / 8);
  const medW = Math.ceil(sceneWidth / 4);
  const medH = Math.ceil(height / 4);

  const coarse = new Float32Array(coarseH * coarseW * 3);
  for (let i = 0; i < coarse.length; i++) coarse[i] = rand();

  const med = new Float32Array(medH * medW * 3);
  for (let i = 0; i < med.length; i++) med[i] = rand();

  const fine = new Float32Array(height * sceneWidth * 3);
  for (let i = 0; i < fine.length; i++) fine[i] = rand();

  const rgb = new Uint8Array(height * sceneWidth * 3);

  for (let y = 0; y < height; y++) {
    const cy = Math.floor(y / 8);
    const my = Math.floor(y / 4);
    for (let x = 0; x < sceneWidth; x++) {
      const cx = Math.floor(x / 8);
      const mx = Math.floor(x / 4);
      // Distinct non-periodic landmarks so correlation peaks uniquely at +d_true without 16px/32px aliasing
      const hashBlock = (((x * 73856093) ^ (y * 19349663) ^ seed) >>> 24) & 0xff;
      const diagStripe = ((x + y * 2) % 37 < 7 ? 32 : 0) + ((x * 3 - y) % 53 < 6 ? 24 : 0);

      for (let c = 0; c < 3; c++) {
        const cVal = coarse[(cy * coarseW + cx) * 3 + c];
        const mVal = med[(my * medW + mx) * 3 + c];
        const fVal = fine[(y * sceneWidth + x) * 3 + c];
        const channelBias = c === 0 ? (hashBlock & 0x3f) : c === 1 ? ((hashBlock >> 2) & 0x3f) : ((hashBlock >> 4) & 0x3f);
        const raw =
          24 +
          cVal * 92 +
          mVal * 72 +
          fVal * 44 +
          (diagStripe + channelBias) * 0.45;
        rgb[(y * sceneWidth + x) * 3 + c] = Math.max(0, Math.min(255, Math.round(raw)));
      }
    }
  }

  return { sceneWidth, height, rgb };
}

export function computeSyntheticGtRois(disparityTruePx: 16 | 32): {
  analyticValidMask: SyntheticGroundTruthPair['analyticValidMask'];
  commonCroppedRoi: SyntheticGroundTruthPair['commonCroppedRoi'];
} {
  const bPad = HITNET_SYNTHETIC_GT_CONSTANTS.borderPadPx;
  const w = HITNET_SYNTHETIC_GT_CONSTANTS.modelWidth;
  const h = HITNET_SYNTHETIC_GT_CONSTANTS.modelHeight;
  const dMax = HITNET_SYNTHETIC_GT_CONSTANTS.maxCommonDisparityPx;

  const yStart = bPad;
  const yEndExclusive = h - bPad;
  const rows = yEndExclusive - yStart;

  const pXStart = bPad + disparityTruePx;
  const pXEnd = w - bPad;
  const sXStart = bPad;
  const sXEnd = w - bPad - disparityTruePx;

  const cPXStart = bPad + dMax;
  const cPXEnd = w - bPad;
  const cSXStart = bPad;
  const cSXEnd = w - bPad - dMax;

  return {
    analyticValidMask: {
      primaryLeft: {
        xStart: pXStart,
        xEndExclusive: pXEnd,
        yStart,
        yEndExclusive,
        pixelCount: rows * (pXEnd - pXStart),
      },
      secondaryRight: {
        xStart: sXStart,
        xEndExclusive: sXEnd,
        yStart,
        yEndExclusive,
        pixelCount: rows * (sXEnd - sXStart),
      },
    },
    commonCroppedRoi: {
      primaryLeft: {
        xStart: cPXStart,
        xEndExclusive: cPXEnd,
        yStart,
        yEndExclusive,
        pixelCount: rows * (cPXEnd - cPXStart),
      },
      secondaryRight: {
        xStart: cSXStart,
        xEndExclusive: cSXEnd,
        yStart,
        yEndExclusive,
        pixelCount: rows * (cSXEnd - cSXStart),
      },
    },
  };
}

export async function generateSyntheticStereoGroundTruthPair(
  disparityTruePx: 16 | 32,
  opts?: { seed?: number }
): Promise<SyntheticGroundTruthPair> {
  const width = HITNET_SYNTHETIC_GT_CONSTANTS.modelWidth;
  const height = HITNET_SYNTHETIC_GT_CONSTANTS.modelHeight;
  const seed = opts?.seed ?? HITNET_SYNTHETIC_GT_CONSTANTS.seed;
  const sceneWidth = width + HITNET_SYNTHETIC_GT_CONSTANTS.maxCommonDisparityPx;
  const tex = buildDeterministicRgbTextureUint8({ sceneWidth, height, seed });

  const pixelCount = width * height;
  const feedFloat32 = new Float32Array(6 * pixelCount);
  const leftRgba = new Uint8ClampedArray(pixelCount * 4);
  const rightRgba = new Uint8ClampedArray(pixelCount * 4);
  const inv255 = Math.fround(1 / 255);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dstPx = y * width + x;
      const srcLeftBase = (y * sceneWidth + x) * 3;
      const srcRightBase = (y * sceneWidth + (x + disparityTruePx)) * 3;

      const lr = tex.rgb[srcLeftBase];
      const lg = tex.rgb[srcLeftBase + 1];
      const lb = tex.rgb[srcLeftBase + 2];
      const rr = tex.rgb[srcRightBase];
      const rg = tex.rgb[srcRightBase + 1];
      const rb = tex.rgb[srcRightBase + 2];

      leftRgba[dstPx * 4] = lr;
      leftRgba[dstPx * 4 + 1] = lg;
      leftRgba[dstPx * 4 + 2] = lb;
      leftRgba[dstPx * 4 + 3] = 255;

      rightRgba[dstPx * 4] = rr;
      rightRgba[dstPx * 4 + 1] = rg;
      rightRgba[dstPx * 4 + 2] = rb;
      rightRgba[dstPx * 4 + 3] = 255;

      // Exact NCHW [1, 6, 240, 320]: channels 0..2 = Left RGB, channels 3..5 = Right RGB
      feedFloat32[dstPx] = Math.fround(lr * inv255);
      feedFloat32[pixelCount + dstPx] = Math.fround(lg * inv255);
      feedFloat32[pixelCount * 2 + dstPx] = Math.fround(lb * inv255);
      feedFloat32[pixelCount * 3 + dstPx] = Math.fround(rr * inv255);
      feedFloat32[pixelCount * 4 + dstPx] = Math.fround(rg * inv255);
      feedFloat32[pixelCount * 5 + dstPx] = Math.fround(rb * inv255);
    }
  }

  const feedSha256 = await sha256HexBytes(
    new Uint8Array(feedFloat32.buffer, feedFloat32.byteOffset, feedFloat32.byteLength)
  );
  const rois = computeSyntheticGtRois(disparityTruePx);

  return {
    caseId: disparityTruePx === 16 ? 'D16' : 'D32',
    disparityTruePx,
    seed,
    width,
    height,
    channels: 6,
    feedDims: [1, 6, 240, 320],
    feedFloat32,
    feedSha256,
    leftRgba,
    rightRgba,
    signConventionNote:
      `Scene texture T(c,y,x): I_L(c,y,x)=T(c,y,x) [ch0..2], I_R(c,y,x)=T(c,y,x+${disparityTruePx}) [ch3..5]. ` +
      `Left correspondence x_R = x_L - ${disparityTruePx} gives reference_output_disparity = +${disparityTruePx}.0 (matching M1 #139 Sub x_R = x_L - d_ref); ` +
      `Right correspondence x_L = x_R + ${disparityTruePx} gives secondary_output_disparity = +${disparityTruePx}.0 (matching M1 #426/#427 x_L = x_R + d_sec).`,
    analyticValidMask: rois.analyticValidMask,
    commonCroppedRoi: rois.commonCroppedRoi,
  };
}

export function evaluateDisparityRoiMetricsPx(
  pred: Float32Array,
  roiLabel: string,
  roi: StereoRoiRect,
  disparityTruePx: number,
  width = HITNET_SYNTHETIC_GT_CONSTANTS.modelWidth
): GroundTruthRegionMetricsPx {
  const absErrors: number[] = [];
  const finitePreds: number[] = [];
  let finiteCount = 0;
  let nonFiniteCount = 0;
  let invalidCount = 0;
  let bad1Count = 0;
  let bad3Count = 0;
  let absErrorSum = 0;
  let maxAbsErrorPx = 0;
  let predSum = 0;
  let minPredPx = Infinity;
  let maxPredPx = -Infinity;

  for (let y = roi.yStart; y < roi.yEndExclusive; y++) {
    const rowOff = y * width;
    for (let x = roi.xStart; x < roi.xEndExclusive; x++) {
      const v = pred[rowOff + x];
      if (!Number.isFinite(v)) {
        nonFiniteCount++;
        invalidCount++;
        // Non-finite predictions in ROI always count as errors (never excluded by prediction-dependent masks)
        bad1Count++;
        bad3Count++;
        absErrors.push(Infinity);
        maxAbsErrorPx = Infinity;
        continue;
      }
      finiteCount++;
      if (v < 0) {
        invalidCount++;
      }
      finitePreds.push(v);
      predSum += v;
      if (v < minPredPx) minPredPx = v;
      if (v > maxPredPx) maxPredPx = v;

      const absErr = Math.abs(v - disparityTruePx);
      absErrors.push(absErr);
      absErrorSum += absErr;
      if (absErr > maxAbsErrorPx) maxAbsErrorPx = absErr;
      if (absErr > 1.0) bad1Count++;
      if (absErr > 3.0) bad3Count++;
    }
  }

  const evaluatedCount = roi.pixelCount;
  absErrors.sort((a, b) => a - b);
  finitePreds.sort((a, b) => a - b);
  const p95Idx = absErrors.length > 0 ? Math.min(absErrors.length - 1, Math.floor(0.95 * (absErrors.length - 1))) : 0;
  const medIdx = finitePreds.length > 0 ? Math.floor(finitePreds.length / 2) : 0;

  return {
    roiLabel,
    roi: { ...roi },
    disparityTruePx,
    evaluatedCount,
    finiteCount,
    nonFiniteCount,
    invalidCount,
    invalidRate: evaluatedCount > 0 ? invalidCount / evaluatedCount : 0,
    epePx: nonFiniteCount > 0 ? Infinity : finiteCount > 0 ? absErrorSum / finiteCount : 0,
    p95AbsErrorPx: absErrors.length > 0 ? absErrors[p95Idx] : 0,
    maxAbsErrorPx: absErrors.length > 0 ? maxAbsErrorPx : 0,
    bad1Count,
    bad1Rate: evaluatedCount > 0 ? bad1Count / evaluatedCount : 0,
    bad3Count,
    bad3Rate: evaluatedCount > 0 ? bad3Count / evaluatedCount : 0,
    meanPredPx: finiteCount > 0 ? predSum / finiteCount : NaN,
    medianPredPx: finitePreds.length > 0 ? finitePreds[medIdx] : NaN,
    minPredPx: finiteCount > 0 ? minPredPx : NaN,
    maxPredPx: finiteCount > 0 ? maxPredPx : NaN,
  };
}

function medianNumber(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
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

export async function runHitnetSyntheticGroundTruthTest(params: {
  m1Bytes: Uint8Array;
  graphOptimizationLevel?: 'basic' | 'disabled';
  warmRuns?: number;
  runWorker?: ResidualDiagnosticWorkerRunner;
  disableCpuEpFallback?: boolean;
  isDev?: boolean;
  onLog?: (line: string) => void;
}): Promise<HitnetSyntheticGroundTruthReport> {
  assertDevFixtureEnabled(params.isDev);
  const log = params.onLog ?? (() => {});
  const runWorker = params.runWorker ?? createBrowserOrt130DiagnosticRunner();
  const graphOptimizationLevel = params.graphOptimizationLevel ?? 'basic';
  const warmRuns = Math.max(0, Math.min(5, Math.trunc(params.warmRuns ?? 1)));
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

  const parsed = await parseM1ModelProtoWithRawSlices(params.m1Bytes);
  const n1OffProof = verifyCanonicalM1NoN1Replacements(parsed, params.m1Bytes, m1Sha, false);
  if (!n1OffProof.verified) {
    throw new Error(n1OffProof.failureReason ?? 'Canonical M1 N1-off guard failed');
  }

  const [pName, sName] = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;
  const asyncify = HITNET_ORT_130_FAMILY_PROFILES.asyncify;

  log(
    `=== [DEV-ONLY] HITNet Known Ground-Truth Synthetic Test (D16 & D32, model-px @ 320x240) ===\n` +
      `Runtime: onnxruntime-web@1.30.0/webgpu (${asyncify.entryFile} + ${asyncify.wasmFile}) | opt=${graphOptimizationLevel} | warmRuns=${warmRuns}\n` +
      `Canonical Untapped M1: bytes=${params.m1Bytes.byteLength} sha256=${m1Sha} (N1=OFF, 0 replacements)\n` +
      `Coordinate Units: native 320x240 model-coordinate pixels (px) only; NO invented camera baseline/focal calibration.`
  );

  const assertSuccess = (
    label: string,
    raw: OrtVersionDiagnosticResponse,
    expectedProvider: 'webgpu' | 'wasm',
    expectedFeedSha: string
  ): OrtVersionDiagnosticSuccess => {
    if (!raw.ok) {
      const f = raw as OrtVersionDiagnosticFailure;
      throw new Error(`${label} FAILED at stage="${f.stage}": ${f.message}`);
    }
    if (
      raw.pinnedOrtVersion !== '1.30.0' ||
      raw.runtimeAssetFamily !== 'asyncify' ||
      raw.graphOptimizationLevel !== graphOptimizationLevel ||
      raw.provider !== expectedProvider ||
      raw.foreignOrtResources.length > 0 ||
      raw.loadedOrtResources.some((u) => u.includes('.jsep.') || u.includes('ort.bundle.'))
    ) {
      throw new Error(
        `${label} runtime/opt guard violation: ver=${raw.pinnedOrtVersion}, fam=${raw.runtimeAssetFamily}, opt=${raw.graphOptimizationLevel}, ep=${raw.provider}`
      );
    }
    if (raw.modelSha256 !== m1Sha || raw.modelByteLength !== params.m1Bytes.byteLength) {
      throw new Error(
        `${label} M1 pre-session SHA mismatch: got=${raw.modelSha256}, expected=${m1Sha}`
      );
    }
    if (raw.feedSha256 !== expectedFeedSha) {
      throw new Error(`${label} feed SHA mismatch: got=${raw.feedSha256}, expected=${expectedFeedSha}`);
    }
    return raw;
  };

  const runCase = async (dispTrue: 16 | 32): Promise<SyntheticGtCaseReport> => {
    const gt = await generateSyntheticStereoGroundTruthPair(dispTrue);
    if (gt.caseId === 'D16') {
      log(
        `\n--- [${gt.caseId}] trueDisparity=${dispTrue}.0 px | feedSha256=${gt.feedSha256} ---\n` +
          `  Sign convention: ${gt.signConventionNote}`
      );
    } else {
      log(`\n--- [${gt.caseId}] trueDisparity=${dispTrue}.0 px | feedSha256=${gt.feedSha256} ---`);
    }

    const gpuTimingRaw = await runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: params.m1Bytes.slice().buffer,
      feedBuffer: gt.feedFloat32.slice().buffer,
      feedName: 'input',
      feedDims: [...gt.feedDims],
      provider: 'webgpu',
      graphOptimizationLevel,
      warmRuns,
      enableWebGpuProfiling: false,
      disableCpuEpFallback,
      captureGpuValidationErrors: true,
    });
    const wasmTimingRaw = await runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: params.m1Bytes.slice().buffer,
      feedBuffer: gt.feedFloat32.slice().buffer,
      feedName: 'input',
      feedDims: [...gt.feedDims],
      provider: 'wasm',
      graphOptimizationLevel,
      warmRuns,
      enableWebGpuProfiling: false,
      disableCpuEpFallback: false,
      captureGpuValidationErrors: false,
    });

    const gpuOk = assertSuccess(`${gt.caseId} WebGPU`, gpuTimingRaw, 'webgpu', gt.feedSha256);
    const wasmOk = assertSuccess(`${gt.caseId} WASM`, wasmTimingRaw, 'wasm', gt.feedSha256);
    const gpuVal = validateHitnetFinalOutputs(gpuOk.outputs);
    const wasmVal = validateHitnetFinalOutputs(wasmOk.outputs);
    if (!gpuVal.ok || !wasmVal.ok) {
      throw new Error(`${gt.caseId} final output shape/dtype/finite validation failed: gpu=${gpuVal.error}, wasm=${wasmVal.error}`);
    }

    // Separate profiling-ON pass strictly for GPU dispatch proof (excluded from OFF-profile timings)
    const gpuProfRaw = await runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: params.m1Bytes.slice().buffer,
      feedBuffer: gt.feedFloat32.slice().buffer,
      feedName: 'input',
      feedDims: [...gt.feedDims],
      provider: 'webgpu',
      graphOptimizationLevel,
      warmRuns: 0,
      enableWebGpuProfiling: true,
      disableCpuEpFallback,
      captureGpuValidationErrors: true,
    });
    const dispatchObserved = hasObservedGpuDispatch(gpuProfRaw);

    const buildArmMetrics = (
      provider: 'webgpu' | 'wasm',
      okRes: OrtVersionDiagnosticSuccess,
      valByName: Record<string, { data: Float32Array }>,
      dispatchCheck: SyntheticGtArmExecutionResult['dispatchCheckProfilingOn']
    ): SyntheticGtArmExecutionResult => {
      const pData = valByName[pName].data;
      const sData = valByName[sName].data;
      return {
        provider,
        graphOptimizationLevel,
        modelSha256: okRes.modelSha256!,
        modelByteLength: okRes.modelByteLength!,
        feedSha256: okRes.feedSha256!,
        profilingOffTimings: {
          createMs: okRes.createMs,
          coldRunMs: okRes.coldRunMs,
          warmRunMs: [...okRes.warmRunMs],
          warmMedianMs: medianNumber(okRes.warmRunMs),
        },
        dispatchCheckProfilingOn: dispatchCheck,
        workerGpuAdapter: okRes.workerGpuAdapter ?? null,
        gpuValidationErrors: okRes.gpuValidationErrors ?? [],
        primary: {
          analyticValid: evaluateDisparityRoiMetricsPx(
            pData,
            'primary-analytic-valid',
            gt.analyticValidMask.primaryLeft,
            dispTrue
          ),
          commonRoi: evaluateDisparityRoiMetricsPx(
            pData,
            'primary-common-roi',
            gt.commonCroppedRoi.primaryLeft,
            dispTrue
          ),
        },
        secondary: {
          analyticValid: evaluateDisparityRoiMetricsPx(
            sData,
            'secondary-analytic-valid',
            gt.analyticValidMask.secondaryRight,
            dispTrue
          ),
          commonRoi: evaluateDisparityRoiMetricsPx(
            sData,
            'secondary-common-roi',
            gt.commonCroppedRoi.secondaryRight,
            dispTrue
          ),
        },
      };
    };

    const webgpuRes = buildArmMetrics('webgpu', gpuOk, gpuVal.byName, {
      executed: true,
      dispatchObserved,
      summary: gpuProfRaw.ok ? gpuProfRaw.gpuKernelProfileSummary ?? null : null,
    });
    const wasmRes = buildArmMetrics('wasm', wasmOk, wasmVal.byName, null);

    const primaryParity = summarizeRawTensorAndParity(gpuVal.byName[pName].data, wasmVal.byName[pName].data);
    const secondaryParity = summarizeRawTensorAndParity(gpuVal.byName[sName].data, wasmVal.byName[sName].data);

    const fmtMetrics = (label: string, m: GroundTruthRegionMetricsPx) =>
      `${label}(n=${m.evaluatedCount}): EPE=${m.epePx.toFixed(4)}px p95=${m.p95AbsErrorPx.toFixed(4)}px Bad1=${(m.bad1Rate * 100).toFixed(2)}% Bad3=${(m.bad3Rate * 100).toFixed(2)}% invalid=${(m.invalidRate * 100).toFixed(2)}% meanPred=${m.meanPredPx.toFixed(4)}px`;

    log(
      `  [WebGPU OFF-profile timing] create=${webgpuRes.profilingOffTimings.createMs.toFixed(1)}ms cold=${webgpuRes.profilingOffTimings.coldRunMs.toFixed(1)}ms warmMed=${webgpuRes.profilingOffTimings.warmMedianMs?.toFixed(1) ?? 'n/a'}ms | [Separate Profiling-ON] dispatchObserved=${dispatchObserved}\n` +
        `    P: ${fmtMetrics('commonROI', webgpuRes.primary.commonRoi)} | ${fmtMetrics('validMask', webgpuRes.primary.analyticValid)}\n` +
        `    S: ${fmtMetrics('commonROI', webgpuRes.secondary.commonRoi)} | ${fmtMetrics('validMask', webgpuRes.secondary.analyticValid)}\n` +
        `  [WASM   OFF-profile timing] create=${wasmRes.profilingOffTimings.createMs.toFixed(1)}ms cold=${wasmRes.profilingOffTimings.coldRunMs.toFixed(1)}ms warmMed=${wasmRes.profilingOffTimings.warmMedianMs?.toFixed(1) ?? 'n/a'}ms\n` +
        `    P: ${fmtMetrics('commonROI', wasmRes.primary.commonRoi)} | ${fmtMetrics('validMask', wasmRes.primary.analyticValid)}\n` +
        `    S: ${fmtMetrics('commonROI', wasmRes.secondary.commonRoi)} | ${fmtMetrics('validMask', wasmRes.secondary.analyticValid)}\n` +
        `  [GPU vs WASM Full-Frame Parity] P(max=${primaryParity.maxAbsDiff.toExponential(3)}, mean=${primaryParity.meanAbsDiff.toExponential(3)}) | S(max=${secondaryParity.maxAbsDiff.toExponential(3)}, mean=${secondaryParity.meanAbsDiff.toExponential(3)})`
    );

    return {
      caseId: gt.caseId,
      disparityTruePx: dispTrue,
      feedSha256: gt.feedSha256,
      feedByteLength: gt.feedFloat32.byteLength,
      signConventionNote: gt.signConventionNote,
      analyticValidMask: gt.analyticValidMask,
      commonCroppedRoi: gt.commonCroppedRoi,
      webgpu: webgpuRes,
      wasm: wasmRes,
      webgpuVsWasmFullFrameParity: {
        primary: primaryParity,
        secondary: secondaryParity,
      },
    };
  };

  const d16 = await runCase(16);
  const d32 = await runCase(32);
  if (d16.feedSha256 === d32.feedSha256) {
    throw new Error('D16 and D32 feed SHA-256 unexpectedly matched');
  }
  if (d16.feedSha256 === HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedF0102FeedSha256) {
    throw new Error('Synthetic D16 feed SHA unexpectedly collided with frozen F0102');
  }

  return {
    ok: true,
    pinnedOrtVersion: '1.30.0',
    runtimeAssetFamily: 'asyncify',
    graphOptimizationLevel,
    warmRuns,
    m1ByteLength: params.m1Bytes.byteLength,
    m1Sha256: m1Sha,
    n1OffProof,
    coordinateUnitsNote:
      'All disparities and error metrics (EPE, p95, Bad1>1.0px, Bad3>3.0px) are strictly in native 320x240 HITNet model-coordinate pixel units (px). No camera baseline or focal calibration is assumed.',
    cases: {
      D16: d16,
      D32: d32,
    },
    scientificScopeNote:
      'Evaluates full-graph untapped canonical M1 accuracy against analytic synthetic ground truth (+16.0px and +32.0px) on ORT 1.30.0 non-JSEP WebGPU vs WASM without excluding predictions via prediction-dependent masks.',
  };
}
