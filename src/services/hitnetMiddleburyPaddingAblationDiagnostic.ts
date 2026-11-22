import {
  HITNET_MIDDLEBURY_MODEL_HEIGHT,
  HITNET_MIDDLEBURY_MODEL_WIDTH,
  HITNET_MIDDLEBURY_PIXEL_COUNT,
  loadAndVerifyMiddleburySceneAssets,
  packRgbaStereoEyesToHitnetFeed,
  type LoadedMiddleburySceneAssets,
  type MiddleburyCalibData,
  type MiddleburyEyeId,
} from './hitnetMiddleburyEval3Benchmark';
import { buildMiddleburyGeometryTransforms } from './hitnetMiddleburyFactorialDiagnosis';
import {
  decodePngRgbTopToBottomBitExact,
  type DecodedRgbImageGrid,
} from './hitnetMiddleburyRawAndGtDiagnostic';
import {
  buildWindowScalePlacements,
  buildWindowSharedPhysicalPointsForEye,
  drawIsolatedRgbaIntoModelCanvasBrowser,
  evaluateWindowArmEyeMetrics,
  MOTORCYCLE_SCALE_FACTOR_320_741,
  PREDECLARED_MOTORCYCLE_320X240_WINDOWS,
  rgbGridWindowToRgba,
  type HitnetMotorcycleScaleContextReport,
  type ScaleContextModelRunSummary,
  type ScaleContextWindowId,
  type ScaleWindowSpec,
  type WindowEyeArmEvaluation,
  type WindowEyeSharedSupport,
  type WindowScaleGeometryPlacement,
} from './hitnetMiddleburyScaleContextDiagnostic';
import {
  HITNET_ORT_130_FAMILY_PROFILES,
  type OrtGpuKernelProfileSummary,
  type OrtVersionDiagnosticFailure,
  type OrtVersionDiagnosticResponse,
  type OrtVersionDiagnosticSuccess,
} from './hitnetOrtComparisonHelper';
import {
  verifyCanonicalM1NoN1Replacements,
  type N1OffVerificationProof,
} from './hitnetPostConv7TapReplay';
import {
  createBrowserOrt130DiagnosticRunner,
  sha256HexBytes,
  validateHitnetFinalOutputs,
  type ResidualDiagnosticWorkerRunner,
} from './hitnetResidualLocalization';
import { parseM1ModelProtoWithRawSlices } from './hitnetRgbStemFirstNonZero';
import {
  assertDevFixtureEnabled,
  HITNET_DEV_NON_JSEP_FIXTURE_SPEC,
} from './stereoDepthWebGpuDevFixture';
import { HITNET_M1_FULL_GRAPH_SHA256 } from './stereoDepthWorkerCore';
import { summarizeRawTensorAndParity, type RawTensorParityStats } from './hitnetGatherNdDiagnostic';

export type FixedActivePaddingMode = 'pad_black' | 'pad_replicate' | 'pad_reflect';

export const FIXED_ACTIVE_PADDING_MODES: readonly FixedActivePaddingMode[] = [
  'pad_black',
  'pad_replicate',
  'pad_reflect',
];

export interface ActiveInteriorPixelBox {
  specPolicy: 'strict-full-coverage-interior';
  colStart: number;
  colEnd: number;
  rowStart: number;
  rowEnd: number;
  width: number;
  height: number;
  pixelCountPerEye: number;
  fractionalRimNote: string;
}

export interface PaddingAblationFeedBundle {
  windowId: ScaleContextWindowId;
  activeInteriorBox: ActiveInteriorPixelBox;
  leftActiveFloat32Sha256: string;
  rightActiveFloat32Sha256: string;
  bothEyesActiveFloat32Sha256: string;
  feeds: Record<
    FixedActivePaddingMode,
    {
      paddingMode: FixedActivePaddingMode;
      feedFloat32: Float32Array;
      feedSha256: string;
    }
  >;
}

export interface FrozenWindowSupportCheck {
  windowId: ScaleContextWindowId;
  leftSupportSha256: string;
  rightSupportSha256: string;
  leftCounts: WindowEyeSharedSupport['counts'];
  rightCounts: WindowEyeSharedSupport['counts'];
  activeSupportMatchesInteriorGuard: boolean;
}

export interface PaddingModeArmResult {
  paddingMode: FixedActivePaddingMode;
  run: ScaleContextModelRunSummary;
  left: WindowEyeArmEvaluation;
  right: WindowEyeArmEvaluation;
}

export interface PaddingDeltaVsBlack {
  paddingMode: 'pad_replicate' | 'pad_reflect';
  leftNonoccEpeNativeQDeltaPx: number;
  leftNonoccSignedBiasNativeQDeltaPx: number;
  leftNonoccP95NativeQDeltaPx: number;
  leftNonoccBad1NativeQDeltaPctPts: number;
  leftNonoccBad3NativeQDeltaPctPts: number;
  leftLowDisp24EpeNativeQDeltaPx: number;
  leftLowDisp24Bad3NativeQDeltaPctPts: number;
  rightNonoccEpeNativeQDeltaPx: number;
  rightNonoccSignedBiasNativeQDeltaPx: number;
  rightNonoccP95NativeQDeltaPx: number;
  rightNonoccBad1NativeQDeltaPctPts: number;
  rightNonoccBad3NativeQDeltaPctPts: number;
  rightLowDisp24EpeNativeQDeltaPx: number;
  rightLowDisp24Bad3NativeQDeltaPctPts: number;
  meanBothEyeNonoccEpeDeltaPx: number;
}

export interface WindowPaddingAblationReport {
  windowSpec: ScaleWindowSpec;
  s1Geometry: WindowScaleGeometryPlacement;
  activeInteriorBox: ActiveInteriorPixelBox;
  frozenSupport: FrozenWindowSupportCheck;
  activeByteIdentityProof: {
    verified: true;
    leftActiveFloat32Sha256: string;
    rightActiveFloat32Sha256: string;
    bothEyesActiveFloat32Sha256: string;
    priorBlackBaselineFeedShaMatched: boolean | null;
    priorBlackBaselineFeedShaExpected: string | null;
  };
  wasmArms: Record<FixedActivePaddingMode, PaddingModeArmResult>;
  deltaVsPadBlackWasm: {
    pad_replicate: PaddingDeltaVsBlack;
    pad_reflect: PaddingDeltaVsBlack;
  };
  priorReferencesNonCausal: {
    reusedFromPriorReport: boolean;
    note: string;
    native1xLeftNonoccEpeQ: number | null;
    native1xRightNonoccEpeQ: number | null;
    wholeFrameContextLeftNonoccEpeQ: number | null;
    wholeFrameContextRightNonoccEpeQ: number | null;
  };
}

export interface BoundedPromisingGpuPairedCheck {
  attempted: boolean;
  selectedWindowId: ScaleContextWindowId;
  selectedCandidatePadding: 'pad_replicate' | 'pad_reflect';
  selectionRationale: string;
  realDispatchObservedBothArms: boolean;
  padBlackGpu: PaddingModeArmResult | null;
  candidateGpu: PaddingModeArmResult | null;
  rawParityGpuVsWasm: {
    padBlack: {
      reference_output_disparity: RawTensorParityStats;
      secondary_output_disparity: RawTensorParityStats;
    };
    candidate: {
      reference_output_disparity: RawTensorParityStats;
      secondary_output_disparity: RawTensorParityStats;
    };
  } | null;
  gpuDeltaCandidateMinusBlack: PaddingDeltaVsBlack | null;
  failureOrFallbackNote: string | null;
}

export interface HitnetMotorcyclePaddingAblationReport {
  ok: true;
  sceneId: 'Motorcycle';
  edition: string;
  calib: MiddleburyCalibData;
  pinnedOrtVersion: '1.30.0';
  runtimeAssetFamily: 'asyncify';
  graphOptimizationLevel: 'basic';
  channelOrder: 'C0_RGB';
  scaleFactor: number;
  m1ByteLength: number;
  m1Sha256: string;
  n1OffProof: N1OffVerificationProof;
  fixtureSha256s: LoadedMiddleburySceneAssets['fixtureSha256s'];
  windows: Record<ScaleContextWindowId, WindowPaddingAblationReport>;
  boundedPromisingGpuPairedCheck: BoundedPromisingGpuPairedCheck;
  conclusion: {
    paddingAblationEffectSummary: string;
    nativeGapRemains: boolean;
    strongestNextControlledTest: string;
  };
}

export type PaddingAblationFeedBundleBuilder = (params: {
  im0Decoded: DecodedRgbImageGrid;
  im1Decoded: DecodedRgbImageGrid;
  windowSpec: ScaleWindowSpec;
  s1Geometry: WindowScaleGeometryPlacement;
  activeInteriorBox: ActiveInteriorPixelBox;
}) => Promise<PaddingAblationFeedBundle>;

export function computeStrictActiveInteriorBox(
  s1Geometry: WindowScaleGeometryPlacement
): ActiveInteriorPixelBox {
  const colStart = Math.ceil(s1Geometry.dstRect.x);
  const colEnd = Math.floor(s1Geometry.dstRect.x + s1Geometry.dstRect.width) - 1;
  const rowStart = Math.ceil(s1Geometry.dstRect.y);
  const rowEnd = Math.floor(s1Geometry.dstRect.y + s1Geometry.dstRect.height) - 1;
  if (
    colStart < 0 ||
    rowStart < 0 ||
    colEnd >= HITNET_MIDDLEBURY_MODEL_WIDTH ||
    rowEnd >= HITNET_MIDDLEBURY_MODEL_HEIGHT ||
    colStart > s1Geometry.activeColStart ||
    colEnd < s1Geometry.activeColEnd ||
    rowStart > s1Geometry.activeRowStart ||
    rowEnd < s1Geometry.activeRowEnd
  ) {
    throw new Error(
      `Active interior box [${colStart}..${colEnd}]x[${rowStart}..${rowEnd}] inconsistent with guarded support [${s1Geometry.activeColStart}..${s1Geometry.activeColEnd}]x[${s1Geometry.activeRowStart}..${s1Geometry.activeRowEnd}]`
    );
  }
  const width = colEnd - colStart + 1;
  const height = rowEnd - rowStart + 1;
  return {
    specPolicy: 'strict-full-coverage-interior',
    colStart,
    colEnd,
    rowStart,
    rowEnd,
    width,
    height,
    pixelCountPerEye: width * height,
    fractionalRimNote:
      `Strict full-coverage interior [${colStart}..${colEnd}]x[${rowStart}..${rowEnd}] (${width}x${height}=${width * height} px/eye) is preserved bit-identical across pad_black/pad_replicate/pad_reflect. ` +
      `Fractional canvas rim pixels outside this box (cols ${Math.floor(s1Geometry.dstRect.x)},${Math.ceil(s1Geometry.dstRect.x + s1Geometry.dstRect.width) - 1}; rows ${Math.floor(s1Geometry.dstRect.y)},${Math.ceil(s1Geometry.dstRect.y + s1Geometry.dstRect.height) - 1}) remain untouched in pad_black (matching prior full feed hash) and are filled outside [${colStart}..${colEnd}]x[${rowStart}..${rowEnd}] in pad_replicate/pad_reflect; common guard=2 px evaluates strictly within [${s1Geometry.activeColStart}..${s1Geometry.activeColEnd}]x[${s1Geometry.activeRowStart}..${s1Geometry.activeRowEnd}].`,
  };
}

function reflect101Index(idx: number, min: number, max: number): number {
  if (min >= max) return min;
  let v = idx;
  while (v < min || v > max) {
    if (v < min) v = min + (min - v);
    if (v > max) v = max - (v - max);
  }
  return v;
}

export function fillPaddingOutsideActiveBoxRgba(
  blackBaselineRgba: Uint8ClampedArray,
  activeBox: ActiveInteriorPixelBox,
  mode: 'pad_replicate' | 'pad_reflect'
): Uint8ClampedArray {
  const mw = HITNET_MIDDLEBURY_MODEL_WIDTH;
  const mh = HITNET_MIDDLEBURY_MODEL_HEIGHT;
  if (blackBaselineRgba.length !== mw * mh * 4) {
    throw new Error(`Expected ${mw * mh * 4} RGBA bytes, got ${blackBaselineRgba.length}`);
  }
  const out = new Uint8ClampedArray(mw * mh * 4);
  for (let y = 0; y < mh; y++) {
    const insideY = y >= activeBox.rowStart && y <= activeBox.rowEnd;
    const srcY = insideY
      ? y
      : mode === 'pad_replicate'
        ? Math.max(activeBox.rowStart, Math.min(activeBox.rowEnd, y))
        : reflect101Index(y, activeBox.rowStart, activeBox.rowEnd);

    for (let x = 0; x < mw; x++) {
      const insideX = x >= activeBox.colStart && x <= activeBox.colEnd;
      const dstOff = (y * mw + x) * 4;
      if (insideX && insideY) {
        // Exact copy inside active interior support
        out[dstOff] = blackBaselineRgba[dstOff];
        out[dstOff + 1] = blackBaselineRgba[dstOff + 1];
        out[dstOff + 2] = blackBaselineRgba[dstOff + 2];
        out[dstOff + 3] = blackBaselineRgba[dstOff + 3];
      } else {
        const srcX = insideX
          ? x
          : mode === 'pad_replicate'
            ? Math.max(activeBox.colStart, Math.min(activeBox.colEnd, x))
            : reflect101Index(x, activeBox.colStart, activeBox.colEnd);
        const srcOff = (srcY * mw + srcX) * 4;
        out[dstOff] = blackBaselineRgba[srcOff];
        out[dstOff + 1] = blackBaselineRgba[srcOff + 1];
        out[dstOff + 2] = blackBaselineRgba[srcOff + 2];
        out[dstOff + 3] = 255;
      }
    }
  }
  return out;
}

export function extractEyeActiveFloat32Bytes(
  feedFloat32: Float32Array,
  eye: MiddleburyEyeId,
  activeBox: ActiveInteriorPixelBox
): Uint8Array {
  const px = HITNET_MIDDLEBURY_PIXEL_COUNT;
  if (feedFloat32.length !== px * 6) {
    throw new Error(`Expected NCHW feed length ${px * 6}, got ${feedFloat32.length}`);
  }
  const chStart = eye === 'left' ? 0 : 3;
  const mw = HITNET_MIDDLEBURY_MODEL_WIDTH;
  const sub = new Float32Array(3 * activeBox.width * activeBox.height);
  let writeIdx = 0;
  for (let c = 0; c < 3; c++) {
    const chBase = (chStart + c) * px;
    for (let y = activeBox.rowStart; y <= activeBox.rowEnd; y++) {
      const rowBase = chBase + y * mw;
      for (let x = activeBox.colStart; x <= activeBox.colEnd; x++) {
        sub[writeIdx++] = feedFloat32[rowBase + x];
      }
    }
  }
  return new Uint8Array(sub.buffer, sub.byteOffset, sub.byteLength);
}

export async function assertActiveSupportByteIdentityBeforeInference(params: {
  windowId: ScaleContextWindowId;
  bundle: PaddingAblationFeedBundle;
}): Promise<{
  leftActiveFloat32Sha256: string;
  rightActiveFloat32Sha256: string;
  bothEyesActiveFloat32Sha256: string;
}> {
  const { windowId, bundle } = params;
  const box = bundle.activeInteriorBox;

  const blackL = extractEyeActiveFloat32Bytes(bundle.feeds.pad_black.feedFloat32, 'left', box);
  const blackR = extractEyeActiveFloat32Bytes(bundle.feeds.pad_black.feedFloat32, 'right', box);
  const blackLSha = await sha256HexBytes(blackL);
  const blackRSha = await sha256HexBytes(blackR);
  const blackCombined = new Uint8Array(blackL.byteLength + blackR.byteLength);
  blackCombined.set(blackL, 0);
  blackCombined.set(blackR, blackL.byteLength);
  const blackBothSha = await sha256HexBytes(blackCombined);

  if (
    blackLSha !== bundle.leftActiveFloat32Sha256 ||
    blackRSha !== bundle.rightActiveFloat32Sha256 ||
    blackBothSha !== bundle.bothEyesActiveFloat32Sha256
  ) {
    throw new Error(
      `Array guard failed on ${windowId} pad_black: declared active SHAs mismatch actual packed Float32 active sub-arrays`
    );
  }

  for (const altMode of ['pad_replicate', 'pad_reflect'] as const) {
    const altFeed = bundle.feeds[altMode];
    if (altFeed.feedSha256 === bundle.feeds.pad_black.feedSha256) {
      throw new Error(
        `Padding ablation invalid on ${windowId}: ${altMode} full feedSha256 equals pad_black`
      );
    }

    const altL = extractEyeActiveFloat32Bytes(altFeed.feedFloat32, 'left', box);
    const altR = extractEyeActiveFloat32Bytes(altFeed.feedFloat32, 'right', box);

    for (let i = 0; i < blackL.byteLength; i++) {
      if (altL[i] !== blackL[i]) {
        throw new Error(
          `Array guard failed on ${windowId}: Left eye active interior byte diff at byteOffset=${i} between pad_black and ${altMode}`
        );
      }
    }
    for (let i = 0; i < blackR.byteLength; i++) {
      if (altR[i] !== blackR[i]) {
        throw new Error(
          `Array guard failed on ${windowId}: Right eye active interior byte diff at byteOffset=${i} between pad_black and ${altMode}`
        );
      }
    }
  }

  if (bundle.feeds.pad_replicate.feedSha256 === bundle.feeds.pad_reflect.feedSha256) {
    throw new Error(
      `Padding ablation invalid on ${windowId}: pad_replicate and pad_reflect full feed hashes unexpectedly match`
    );
  }

  return {
    leftActiveFloat32Sha256: blackLSha,
    rightActiveFloat32Sha256: blackRSha,
    bothEyesActiveFloat32Sha256: blackBothSha,
  };
}

async function computeSupportSha256(support: WindowEyeSharedSupport): Promise<string> {
  const n = support.sharedNonoccPoints.length;
  const buf = new ArrayBuffer(n * 12);
  const dv = new DataView(buf);
  for (let i = 0; i < n; i++) {
    const pt = support.sharedNonoccPoints[i];
    dv.setInt32(i * 12, pt.ix, true);
    dv.setInt32(i * 12 + 4, pt.iy, true);
    dv.setFloat32(i * 12 + 8, pt.gtNativePx, true);
  }
  return sha256HexBytes(new Uint8Array(buf));
}

export const defaultBrowserPaddingAblationFeedBuilder: PaddingAblationFeedBundleBuilder = async ({
  im0Decoded,
  im1Decoded,
  windowSpec,
  s1Geometry,
  activeInteriorBox,
}) => {
  const winLeftRgba = rgbGridWindowToRgba(
    im0Decoded,
    windowSpec.x,
    windowSpec.y,
    windowSpec.width,
    windowSpec.height
  );
  const winRightRgba = rgbGridWindowToRgba(
    im1Decoded,
    windowSpec.x,
    windowSpec.y,
    windowSpec.width,
    windowSpec.height
  );

  // Build exact prior S1 black baseline RGBA via isolated 320x240 canvas drawImage
  const blackLeftRgba = drawIsolatedRgbaIntoModelCanvasBrowser(
    winLeftRgba,
    windowSpec.width,
    windowSpec.height,
    s1Geometry.dstRect
  );
  const blackRightRgba = drawIsolatedRgbaIntoModelCanvasBrowser(
    winRightRgba,
    windowSpec.width,
    windowSpec.height,
    s1Geometry.dstRect
  );

  const repLeftRgba = fillPaddingOutsideActiveBoxRgba(
    blackLeftRgba,
    activeInteriorBox,
    'pad_replicate'
  );
  const repRightRgba = fillPaddingOutsideActiveBoxRgba(
    blackRightRgba,
    activeInteriorBox,
    'pad_replicate'
  );

  const refLeftRgba = fillPaddingOutsideActiveBoxRgba(
    blackLeftRgba,
    activeInteriorBox,
    'pad_reflect'
  );
  const refRightRgba = fillPaddingOutsideActiveBoxRgba(
    blackRightRgba,
    activeInteriorBox,
    'pad_reflect'
  );

  const blackFeed = packRgbaStereoEyesToHitnetFeed(blackLeftRgba, blackRightRgba);
  const repFeed = packRgbaStereoEyesToHitnetFeed(repLeftRgba, repRightRgba);
  const refFeed = packRgbaStereoEyesToHitnetFeed(refLeftRgba, refRightRgba);

  const [blackSha, repSha, refSha] = await Promise.all([
    sha256HexBytes(new Uint8Array(blackFeed.buffer, blackFeed.byteOffset, blackFeed.byteLength)),
    sha256HexBytes(new Uint8Array(repFeed.buffer, repFeed.byteOffset, repFeed.byteLength)),
    sha256HexBytes(new Uint8Array(refFeed.buffer, refFeed.byteOffset, refFeed.byteLength)),
  ]);

  const activeL = extractEyeActiveFloat32Bytes(blackFeed, 'left', activeInteriorBox);
  const activeR = extractEyeActiveFloat32Bytes(blackFeed, 'right', activeInteriorBox);
  const activeBoth = new Uint8Array(activeL.byteLength + activeR.byteLength);
  activeBoth.set(activeL, 0);
  activeBoth.set(activeR, activeL.byteLength);

  const [leftActiveFloat32Sha256, rightActiveFloat32Sha256, bothEyesActiveFloat32Sha256] =
    await Promise.all([
      sha256HexBytes(activeL),
      sha256HexBytes(activeR),
      sha256HexBytes(activeBoth),
    ]);

  return {
    windowId: windowSpec.windowId,
    activeInteriorBox,
    leftActiveFloat32Sha256,
    rightActiveFloat32Sha256,
    bothEyesActiveFloat32Sha256,
    feeds: {
      pad_black: { paddingMode: 'pad_black', feedFloat32: blackFeed, feedSha256: blackSha },
      pad_replicate: { paddingMode: 'pad_replicate', feedFloat32: repFeed, feedSha256: repSha },
      pad_reflect: { paddingMode: 'pad_reflect', feedFloat32: refFeed, feedSha256: refSha },
    },
  };
};

function computePaddingDelta(
  mode: 'pad_replicate' | 'pad_reflect',
  candL: WindowEyeArmEvaluation,
  candR: WindowEyeArmEvaluation,
  blackL: WindowEyeArmEvaluation,
  blackR: WindowEyeArmEvaluation
): PaddingDeltaVsBlack {
  const lDelta = candL.nonocc.epeNativeQPx - blackL.nonocc.epeNativeQPx;
  const rDelta = candR.nonocc.epeNativeQPx - blackR.nonocc.epeNativeQPx;
  return {
    paddingMode: mode,
    leftNonoccEpeNativeQDeltaPx: lDelta,
    leftNonoccSignedBiasNativeQDeltaPx:
      candL.nonocc.signedBiasNativeQPx - blackL.nonocc.signedBiasNativeQPx,
    leftNonoccP95NativeQDeltaPx:
      candL.nonocc.p95AbsErrorNativeQPx - blackL.nonocc.p95AbsErrorNativeQPx,
    leftNonoccBad1NativeQDeltaPctPts:
      (candL.nonocc.bad1NativeQRate - blackL.nonocc.bad1NativeQRate) * 100,
    leftNonoccBad3NativeQDeltaPctPts:
      (candL.nonocc.bad3NativeQRate - blackL.nonocc.bad3NativeQRate) * 100,
    leftLowDisp24EpeNativeQDeltaPx:
      candL.lowDisp24Nonocc.epeNativeQPx - blackL.lowDisp24Nonocc.epeNativeQPx,
    leftLowDisp24Bad3NativeQDeltaPctPts:
      (candL.lowDisp24Nonocc.bad3NativeQRate - blackL.lowDisp24Nonocc.bad3NativeQRate) * 100,
    rightNonoccEpeNativeQDeltaPx: rDelta,
    rightNonoccSignedBiasNativeQDeltaPx:
      candR.nonocc.signedBiasNativeQPx - blackR.nonocc.signedBiasNativeQPx,
    rightNonoccP95NativeQDeltaPx:
      candR.nonocc.p95AbsErrorNativeQPx - blackR.nonocc.p95AbsErrorNativeQPx,
    rightNonoccBad1NativeQDeltaPctPts:
      (candR.nonocc.bad1NativeQRate - blackR.nonocc.bad1NativeQRate) * 100,
    rightNonoccBad3NativeQDeltaPctPts:
      (candR.nonocc.bad3NativeQRate - blackR.nonocc.bad3NativeQRate) * 100,
    rightLowDisp24EpeNativeQDeltaPx:
      candR.lowDisp24Nonocc.epeNativeQPx - blackR.lowDisp24Nonocc.epeNativeQPx,
    rightLowDisp24Bad3NativeQDeltaPctPts:
      (candR.lowDisp24Nonocc.bad3NativeQRate - blackR.lowDisp24Nonocc.bad3NativeQRate) * 100,
    meanBothEyeNonoccEpeDeltaPx: (lDelta + rDelta) / 2,
  };
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

export async function runHitnetMiddleburyPaddingAblationDiagnostic(params: {
  m1Bytes: Uint8Array;
  isDev?: boolean;
  includeWebGpuCheck?: boolean;
  disableCpuEpFallback?: boolean;
  priorScaleContextReport?: HitnetMotorcycleScaleContextReport | null;
  runWorker?: ResidualDiagnosticWorkerRunner;
  loadMotorcycleAssets?: () => Promise<LoadedMiddleburySceneAssets>;
  buildFeeds?: PaddingAblationFeedBundleBuilder;
  onLog?: (line: string) => void;
}): Promise<HitnetMotorcyclePaddingAblationReport> {
  assertDevFixtureEnabled(params.isDev);
  const log = params.onLog ?? (() => {});
  const runWorker = params.runWorker ?? createBrowserOrt130DiagnosticRunner();
  const loadMotorcycle =
    params.loadMotorcycleAssets ?? (() => loadAndVerifyMiddleburySceneAssets('Motorcycle'));
  const buildFeeds = params.buildFeeds ?? defaultBrowserPaddingAblationFeedBuilder;
  const includeWebGpuCheck = params.includeWebGpuCheck ?? true;
  const disableCpuEpFallback = Boolean(params.disableCpuEpFallback);
  const priorReport = params.priorScaleContextReport ?? null;

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
    throw new Error(n1OffProof.failureReason ?? 'Canonical M1 N1=OFF check failed');
  }

  const assets = await loadMotorcycle();
  const im0Decoded = await decodePngRgbTopToBottomBitExact(assets.im0Bytes);
  const im1Decoded = await decodePngRgbTopToBottomBitExact(assets.im1Bytes);
  if (
    im0Decoded.width !== assets.calib.width ||
    im0Decoded.height !== assets.calib.height ||
    im1Decoded.width !== assets.calib.width ||
    im1Decoded.height !== assets.calib.height
  ) {
    throw new Error('Decoded Motorcycle PNG dimensions disagree with calib.txt');
  }

  const { g1: wholeFrameGeometry } = buildMiddleburyGeometryTransforms(
    'Motorcycle',
    assets.calib.width,
    assets.calib.height
  );

  const asyncify = HITNET_ORT_130_FAMILY_PROFILES.asyncify;
  const [pName, sName] = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;

  log(
    `=== [DEV-ONLY] Motorcycle Fixed-Active-Pixels Padding Ablation (pad_black vs pad_replicate vs pad_reflect) ===\n` +
      `Runtime: onnxruntime-web@1.30.0 (${asyncify.entryFile} + ${asyncify.wasmFile}) | opt=basic | C0_RGB | scale=320/741 | M1=${m1Sha} (N1=OFF, 0 replacements)\n` +
      `Prior reference reused in session: ${priorReport ? `YES (M1=${priorReport.m1Sha256.slice(0, 12)}..., non-causal reference only)` : 'NO (standalone padding-only run without repeating S0/whole-frame)'}`
  );

  const executeRun = async (req: {
    runId: string;
    windowId: ScaleContextWindowId;
    paddingMode: FixedActivePaddingMode;
    provider: 'wasm' | 'webgpu';
    feedFloat32: Float32Array;
    feedSha256: string;
    verifyGpuDispatch?: boolean;
  }): Promise<{
    summary: ScaleContextModelRunSummary;
    leftOut: Float32Array;
    rightOut: Float32Array;
  }> => {
    const raw = await runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: params.m1Bytes.slice().buffer,
      feedBuffer: req.feedFloat32.slice().buffer,
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      provider: req.provider,
      graphOptimizationLevel: 'basic',
      warmRuns: 1,
      enableWebGpuProfiling: false,
      disableCpuEpFallback: req.provider === 'webgpu' ? disableCpuEpFallback : false,
      captureGpuValidationErrors: req.provider === 'webgpu',
    });

    if (!raw.ok) {
      const f = raw as OrtVersionDiagnosticFailure;
      throw new Error(`${req.runId} FAILED (${f.stage}): ${f.message}`);
    }
    const okRes = raw as OrtVersionDiagnosticSuccess;
    if (
      okRes.pinnedOrtVersion !== '1.30.0' ||
      okRes.runtimeAssetFamily !== 'asyncify' ||
      okRes.graphOptimizationLevel !== 'basic' ||
      okRes.provider !== req.provider ||
      okRes.configuredProviders.length !== 1 ||
      okRes.configuredProviders[0] !== req.provider ||
      okRes.foreignOrtResources.length > 0 ||
      okRes.loadedOrtResources.some((u) => u.includes('.jsep.') || u.includes('ort.bundle.'))
    ) {
      throw new Error(
        `${req.runId} runtime/provider/opt guard violation: provider=${okRes.provider}`
      );
    }
    if (okRes.modelSha256 !== m1Sha || okRes.modelByteLength !== params.m1Bytes.byteLength) {
      throw new Error(`${req.runId} pre-session M1 SHA mismatch: ${okRes.modelSha256}`);
    }
    if (okRes.feedSha256 !== req.feedSha256) {
      throw new Error(`${req.runId} feed SHA mismatch: ${okRes.feedSha256} vs ${req.feedSha256}`);
    }

    const val = validateHitnetFinalOutputs(okRes.outputs);
    if (!val.ok) {
      throw new Error(`${req.runId} output contract error: ${val.error}`);
    }

    let dispatchObserved: boolean | null = null;
    let gpuKernelProfileSummary: OrtGpuKernelProfileSummary | null = null;
    if (req.provider === 'webgpu' && req.verifyGpuDispatch) {
      const gpuProf = await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: params.m1Bytes.slice().buffer,
        feedBuffer: req.feedFloat32.slice().buffer,
        feedName: 'input',
        feedDims: [1, 6, 240, 320],
        provider: 'webgpu',
        graphOptimizationLevel: 'basic',
        warmRuns: 0,
        enableWebGpuProfiling: true,
        disableCpuEpFallback,
        captureGpuValidationErrors: true,
      });
      dispatchObserved = hasObservedGpuDispatch(gpuProf);
      gpuKernelProfileSummary = gpuProf.ok ? gpuProf.gpuKernelProfileSummary ?? null : null;
    }

    return {
      leftOut: val.byName[pName].data,
      rightOut: val.byName[sName].data,
      summary: {
        runId: req.runId,
        windowId: req.windowId,
        armKind: 'S1_downscale_320_741',
        scaleX: MOTORCYCLE_SCALE_FACTOR_320_741,
        scaleY: MOTORCYCLE_SCALE_FACTOR_320_741,
        channelOrder: 'C0_RGB',
        provider: req.provider,
        graphOptimizationLevel: 'basic',
        modelByteLength: okRes.modelByteLength,
        modelSha256: okRes.modelSha256,
        feedSha256: okRes.feedSha256,
        configuredProviders: [...okRes.configuredProviders],
        fallbackBehavior: okRes.fallbackBehavior,
        loadedOrtResources: [...okRes.loadedOrtResources],
        foreignOrtResources: [...okRes.foreignOrtResources],
        gpuValidationErrors: [...(okRes.gpuValidationErrors ?? [])],
        dispatchObserved,
        gpuKernelProfileSummary,
        createMs: okRes.createMs,
        coldRunMs: okRes.coldRunMs,
        warmRunMs: [...okRes.warmRunMs],
        paddedBorderNonZeroCount: 0,
      },
    };
  };

  const windowIds: ScaleContextWindowId[] = ['top_left', 'center'];
  const windowReports = {} as Record<ScaleContextWindowId, WindowPaddingAblationReport>;
  const bundleCache = {} as Record<
    ScaleContextWindowId,
    {
      bundle: PaddingAblationFeedBundle;
      leftSupport: WindowEyeSharedSupport;
      rightSupport: WindowEyeSharedSupport;
      wasmRawOut: Record<FixedActivePaddingMode, { left: Float32Array; right: Float32Array }>;
    }
  >;

  for (const windowId of windowIds) {
    const windowSpec = PREDECLARED_MOTORCYCLE_320X240_WINDOWS[windowId];
    const { s1: s1Geometry } = buildWindowScalePlacements(windowSpec);
    const activeInteriorBox = computeStrictActiveInteriorBox(s1Geometry);

    const leftSupport = buildWindowSharedPhysicalPointsForEye({
      eye: 'left',
      windowSpec,
      gtPfm: assets.disp0GtPfm,
      maskNocc: assets.mask0Nocc,
      calib: assets.calib,
      wholeFrameGeometry,
    });
    const rightSupport = buildWindowSharedPhysicalPointsForEye({
      eye: 'right',
      windowSpec,
      gtPfm: assets.disp1GtPfm,
      maskNocc: assets.mask1Nocc,
      calib: assets.calib,
      wholeFrameGeometry,
    });

    const [leftSupportSha256, rightSupportSha256] = await Promise.all([
      computeSupportSha256(leftSupport),
      computeSupportSha256(rightSupport),
    ]);

    const frozenSupport: FrozenWindowSupportCheck = {
      windowId,
      leftSupportSha256,
      rightSupportSha256,
      leftCounts: leftSupport.counts,
      rightCounts: rightSupport.counts,
      activeSupportMatchesInteriorGuard:
        s1Geometry.activeColStart === activeInteriorBox.colStart + s1Geometry.padGuardModelPx &&
        s1Geometry.activeColEnd === activeInteriorBox.colEnd - s1Geometry.padGuardModelPx &&
        s1Geometry.activeRowStart === activeInteriorBox.rowStart + s1Geometry.padGuardModelPx &&
        s1Geometry.activeRowEnd === activeInteriorBox.rowEnd - s1Geometry.padGuardModelPx,
    };
    if (!frozenSupport.activeSupportMatchesInteriorGuard) {
      throw new Error(`Frozen support guard check failed for ${windowId}`);
    }

    const bundle = await buildFeeds({
      im0Decoded,
      im1Decoded,
      windowSpec,
      s1Geometry,
      activeInteriorBox,
    });

    // Assert BOTH-EYE active byte hashes & zero active diffs strictly BEFORE inference
    const verifiedActive = await assertActiveSupportByteIdentityBeforeInference({
      windowId,
      bundle,
    });

    const priorWin = priorReport?.windows?.[windowId] ?? null;
    const priorBlackSha = priorWin?.downscale320_741Wasm.run.feedSha256 ?? null;
    const priorBlackMatched =
      priorBlackSha !== null ? bundle.feeds.pad_black.feedSha256 === priorBlackSha : null;
    if (priorBlackMatched === false) {
      throw new Error(
        `${windowId} pad_black full feedSha256=${bundle.feeds.pad_black.feedSha256} disagrees with prior S1_downscale_320_741 baseline (${priorBlackSha})`
      );
    }

    log(
      `\n=== Window: ${windowId} | Active Interior [${activeInteriorBox.colStart}..${activeInteriorBox.colEnd}]x[${activeInteriorBox.rowStart}..${activeInteriorBox.rowEnd}] (${activeInteriorBox.width}x${activeInteriorBox.height}) ===\n` +
        `  Pre-Inference Guard PASS: LeftActiveSHA=${verifiedActive.leftActiveFloat32Sha256.slice(0, 12)}... RightActiveSHA=${verifiedActive.rightActiveFloat32Sha256.slice(0, 12)}... BothSHA=${verifiedActive.bothEyesActiveFloat32Sha256.slice(0, 12)}...\n` +
        `  Frozen Common Support: Left nonocc=${leftSupport.counts.sharedNonoccCount} (low<24Q=${leftSupport.counts.sharedLowDisp24NonoccCount}, sha=${leftSupportSha256.slice(0, 12)}...) | Right nonocc=${rightSupport.counts.sharedNonoccCount} (low<24Q=${rightSupport.counts.sharedLowDisp24NonoccCount}, sha=${rightSupportSha256.slice(0, 12)}...)` +
        (priorBlackMatched !== null ? ` | PriorBlackFeedMatch=${String(priorBlackMatched)}` : '')
    );

    const wasmArms = {} as Record<FixedActivePaddingMode, PaddingModeArmResult>;
    const wasmRawOut = {} as Record<
      FixedActivePaddingMode,
      { left: Float32Array; right: Float32Array }
    >;

    for (const mode of FIXED_ACTIVE_PADDING_MODES) {
      const entry = bundle.feeds[mode];
      const exec = await executeRun({
        runId: `${windowId}_${mode}_wasm`,
        windowId,
        paddingMode: mode,
        provider: 'wasm',
        feedFloat32: entry.feedFloat32,
        feedSha256: entry.feedSha256,
      });

      const leftEval = evaluateWindowArmEyeMetrics({
        pred: exec.leftOut,
        armKind: 'S1_downscale_320_741',
        scaleX: MOTORCYCLE_SCALE_FACTOR_320_741,
        support: leftSupport,
      });
      const rightEval = evaluateWindowArmEyeMetrics({
        pred: exec.rightOut,
        armKind: 'S1_downscale_320_741',
        scaleX: MOTORCYCLE_SCALE_FACTOR_320_741,
        support: rightSupport,
      });

      wasmArms[mode] = {
        paddingMode: mode,
        run: exec.summary,
        left: leftEval,
        right: rightEval,
      };
      wasmRawOut[mode] = { left: exec.leftOut, right: exec.rightOut };

      log(
        `  [WASM ${mode}] feed=${exec.summary.feedSha256} cold=${exec.summary.coldRunMs.toFixed(1)}ms warm=${JSON.stringify(exec.summary.warmRunMs.map((n) => Number(n.toFixed(1))))}\n` +
          `    L nonocc(n=${leftEval.nonocc.evaluatedCount}): EPE=${leftEval.nonocc.epeNativeQPx.toFixed(3)}Q bias=${leftEval.nonocc.signedBiasNativeQPx.toFixed(3)}Q p95=${leftEval.nonocc.p95AbsErrorNativeQPx.toFixed(2)}Q Bad1=${(leftEval.nonocc.bad1NativeQRate * 100).toFixed(2)}% Bad3=${(leftEval.nonocc.bad3NativeQRate * 100).toFixed(2)}% inv=${leftEval.nonocc.invalidPredCount} | low<24Q(n=${leftEval.lowDisp24Nonocc.evaluatedCount}): EPE=${leftEval.lowDisp24Nonocc.epeNativeQPx.toFixed(3)}Q Bad3=${(leftEval.lowDisp24Nonocc.bad3NativeQRate * 100).toFixed(2)}%\n` +
          `    R nonocc(n=${rightEval.nonocc.evaluatedCount}): EPE=${rightEval.nonocc.epeNativeQPx.toFixed(3)}Q bias=${rightEval.nonocc.signedBiasNativeQPx.toFixed(3)}Q p95=${rightEval.nonocc.p95AbsErrorNativeQPx.toFixed(2)}Q Bad1=${(rightEval.nonocc.bad1NativeQRate * 100).toFixed(2)}% Bad3=${(rightEval.nonocc.bad3NativeQRate * 100).toFixed(2)}% inv=${rightEval.nonocc.invalidPredCount} | low<24Q(n=${rightEval.lowDisp24Nonocc.evaluatedCount}): EPE=${rightEval.lowDisp24Nonocc.epeNativeQPx.toFixed(3)}Q Bad3=${(rightEval.lowDisp24Nonocc.bad3NativeQRate * 100).toFixed(2)}%`
      );
    }

    const blackArm = wasmArms.pad_black;
    const repDelta = computePaddingDelta(
      'pad_replicate',
      wasmArms.pad_replicate.left,
      wasmArms.pad_replicate.right,
      blackArm.left,
      blackArm.right
    );
    const refDelta = computePaddingDelta(
      'pad_reflect',
      wasmArms.pad_reflect.left,
      wasmArms.pad_reflect.right,
      blackArm.left,
      blackArm.right
    );

    windowReports[windowId] = {
      windowSpec,
      s1Geometry,
      activeInteriorBox,
      frozenSupport,
      activeByteIdentityProof: {
        verified: true,
        ...verifiedActive,
        priorBlackBaselineFeedShaMatched: priorBlackMatched,
        priorBlackBaselineFeedShaExpected: priorBlackSha,
      },
      wasmArms,
      deltaVsPadBlackWasm: {
        pad_replicate: repDelta,
        pad_reflect: refDelta,
      },
      priorReferencesNonCausal: {
        reusedFromPriorReport: priorWin !== null,
        note:
          priorWin !== null
            ? 'Reused from prior session Scale/Context report as NON-CAUSAL reference only (different active FOV / unpadded native scale).'
            : 'No prior Scale/Context report in session; standalone padding ablation did not repeat prior native 1:1 or whole-frame runs.',
        native1xLeftNonoccEpeQ: priorWin?.native1xWasm.left.nonocc.epeNativeQPx ?? null,
        native1xRightNonoccEpeQ: priorWin?.native1xWasm.right.nonocc.epeNativeQPx ?? null,
        wholeFrameContextLeftNonoccEpeQ:
          priorWin?.wholeFrameContextRefOnWindowPoints.left.nonocc.epeNativeQPx ?? null,
        wholeFrameContextRightNonoccEpeQ:
          priorWin?.wholeFrameContextRefOnWindowPoints.right.nonocc.epeNativeQPx ?? null,
      },
    };

    bundleCache[windowId] = {
      bundle,
      leftSupport,
      rightSupport,
      wasmRawOut,
    };
  }

  // Pick single most promising window + padding mode (largest EPE reduction vs pad_black; fallback to top_left)
  let selectedWindowId: ScaleContextWindowId = 'top_left';
  let selectedCandidatePadding: 'pad_replicate' | 'pad_reflect' = 'pad_reflect';
  let bestMeanDelta = Infinity;
  for (const wId of windowIds) {
    for (const cand of ['pad_replicate', 'pad_reflect'] as const) {
      const d = windowReports[wId].deltaVsPadBlackWasm[cand].meanBothEyeNonoccEpeDeltaPx;
      if (d < bestMeanDelta) {
        bestMeanDelta = d;
        selectedWindowId = wId;
        selectedCandidatePadding = cand;
      }
    }
  }

  const selectionRationale =
    bestMeanDelta < 0
      ? `Selected ${selectedWindowId} + ${selectedCandidatePadding} for bounded paired WebGPU check (largest both-eye WASM EPE reduction vs pad_black: ${bestMeanDelta.toFixed(3)}Q).`
      : `Neither alternative padding lowered both-eye EPE below pad_black (best delta=${bestMeanDelta.toFixed(3)}Q); paired WebGPU check runs on ${selectedWindowId} (${selectedCandidatePadding} vs pad_black).`;

  const boundedPromisingGpuPairedCheck: BoundedPromisingGpuPairedCheck = {
    attempted: false,
    selectedWindowId,
    selectedCandidatePadding,
    selectionRationale,
    realDispatchObservedBothArms: false,
    padBlackGpu: null,
    candidateGpu: null,
    rawParityGpuVsWasm: null,
    gpuDeltaCandidateMinusBlack: null,
    failureOrFallbackNote: null,
  };

  if (includeWebGpuCheck) {
    boundedPromisingGpuPairedCheck.attempted = true;
    const sel = bundleCache[selectedWindowId];
    try {
      const gpuBlack = await executeRun({
        runId: `${selectedWindowId}_pad_black_webgpu`,
        windowId: selectedWindowId,
        paddingMode: 'pad_black',
        provider: 'webgpu',
        feedFloat32: sel.bundle.feeds.pad_black.feedFloat32,
        feedSha256: sel.bundle.feeds.pad_black.feedSha256,
        verifyGpuDispatch: true,
      });
      const gpuCand = await executeRun({
        runId: `${selectedWindowId}_${selectedCandidatePadding}_webgpu`,
        windowId: selectedWindowId,
        paddingMode: selectedCandidatePadding,
        provider: 'webgpu',
        feedFloat32: sel.bundle.feeds[selectedCandidatePadding].feedFloat32,
        feedSha256: sel.bundle.feeds[selectedCandidatePadding].feedSha256,
        verifyGpuDispatch: true,
      });

      const blackLeftEval = evaluateWindowArmEyeMetrics({
        pred: gpuBlack.leftOut,
        armKind: 'S1_downscale_320_741',
        scaleX: MOTORCYCLE_SCALE_FACTOR_320_741,
        support: sel.leftSupport,
      });
      const blackRightEval = evaluateWindowArmEyeMetrics({
        pred: gpuBlack.rightOut,
        armKind: 'S1_downscale_320_741',
        scaleX: MOTORCYCLE_SCALE_FACTOR_320_741,
        support: sel.rightSupport,
      });

      const candLeftEval = evaluateWindowArmEyeMetrics({
        pred: gpuCand.leftOut,
        armKind: 'S1_downscale_320_741',
        scaleX: MOTORCYCLE_SCALE_FACTOR_320_741,
        support: sel.leftSupport,
      });
      const candRightEval = evaluateWindowArmEyeMetrics({
        pred: gpuCand.rightOut,
        armKind: 'S1_downscale_320_741',
        scaleX: MOTORCYCLE_SCALE_FACTOR_320_741,
        support: sel.rightSupport,
      });

      boundedPromisingGpuPairedCheck.padBlackGpu = {
        paddingMode: 'pad_black',
        run: gpuBlack.summary,
        left: blackLeftEval,
        right: blackRightEval,
      };
      boundedPromisingGpuPairedCheck.candidateGpu = {
        paddingMode: selectedCandidatePadding,
        run: gpuCand.summary,
        left: candLeftEval,
        right: candRightEval,
      };
      boundedPromisingGpuPairedCheck.rawParityGpuVsWasm = {
        padBlack: {
          reference_output_disparity: summarizeRawTensorAndParity(
            gpuBlack.leftOut,
            sel.wasmRawOut.pad_black.left
          ),
          secondary_output_disparity: summarizeRawTensorAndParity(
            gpuBlack.rightOut,
            sel.wasmRawOut.pad_black.right
          ),
        },
        candidate: {
          reference_output_disparity: summarizeRawTensorAndParity(
            gpuCand.leftOut,
            sel.wasmRawOut[selectedCandidatePadding].left
          ),
          secondary_output_disparity: summarizeRawTensorAndParity(
            gpuCand.rightOut,
            sel.wasmRawOut[selectedCandidatePadding].right
          ),
        },
      };
      boundedPromisingGpuPairedCheck.gpuDeltaCandidateMinusBlack = computePaddingDelta(
        selectedCandidatePadding,
        candLeftEval,
        candRightEval,
        blackLeftEval,
        blackRightEval
      );
      boundedPromisingGpuPairedCheck.realDispatchObservedBothArms =
        gpuBlack.summary.dispatchObserved === true && gpuCand.summary.dispatchObserved === true;

      log(
        `\n[Bounded WebGPU Paired Check (${selectedWindowId}: ${selectedCandidatePadding} vs pad_black)] ` +
          `realDispatchBoth=${String(boundedPromisingGpuPairedCheck.realDispatchObservedBothArms)} | ` +
          `Black EPE(L=${blackLeftEval.nonocc.epeNativeQPx.toFixed(3)}Q, R=${blackRightEval.nonocc.epeNativeQPx.toFixed(3)}Q) vs ` +
          `${selectedCandidatePadding} EPE(L=${candLeftEval.nonocc.epeNativeQPx.toFixed(3)}Q, R=${candRightEval.nonocc.epeNativeQPx.toFixed(3)}Q) | ` +
          `GPUvsWASM maxDiff(Black=${boundedPromisingGpuPairedCheck.rawParityGpuVsWasm.padBlack.reference_output_disparity.maxAbsDiff.toExponential(2)}, Cand=${boundedPromisingGpuPairedCheck.rawParityGpuVsWasm.candidate.reference_output_disparity.maxAbsDiff.toExponential(2)})`
      );
    } catch (err) {
      boundedPromisingGpuPairedCheck.failureOrFallbackNote =
        err instanceof Error ? err.message : String(err);
      log(
        `\n[Bounded WebGPU Paired Check] Unavailable/Failed cleanly: ${boundedPromisingGpuPairedCheck.failureOrFallbackNote}`
      );
    }
  }

  // Synthesize conclusion + strongest next controlled test if native gap remains
  const bestAltEpeTL = Math.min(
    (windowReports.top_left.wasmArms.pad_replicate.left.nonocc.epeNativeQPx +
      windowReports.top_left.wasmArms.pad_replicate.right.nonocc.epeNativeQPx) /
      2,
    (windowReports.top_left.wasmArms.pad_reflect.left.nonocc.epeNativeQPx +
      windowReports.top_left.wasmArms.pad_reflect.right.nonocc.epeNativeQPx) /
      2
  );
  const bestAltEpeCenter = Math.min(
    (windowReports.center.wasmArms.pad_replicate.left.nonocc.epeNativeQPx +
      windowReports.center.wasmArms.pad_replicate.right.nonocc.epeNativeQPx) /
      2,
    (windowReports.center.wasmArms.pad_reflect.left.nonocc.epeNativeQPx +
      windowReports.center.wasmArms.pad_reflect.right.nonocc.epeNativeQPx) /
      2
  );

  const priorNatTL = windowReports.top_left.priorReferencesNonCausal.native1xLeftNonoccEpeQ;
  const nativeGapRemains =
    priorNatTL !== null
      ? bestAltEpeTL > priorNatTL * 1.25 || bestAltEpeCenter > 2.0
      : bestAltEpeTL > 2.0 || bestAltEpeCenter > 2.0;

  const tlWfRef = windowReports.top_left.priorReferencesNonCausal.wholeFrameContextLeftNonoccEpeQ;
  const paddingAblationEffectSummary =
    `top_left EPE deltas vs pad_black: replicate=${windowReports.top_left.deltaVsPadBlackWasm.pad_replicate.meanBothEyeNonoccEpeDeltaPx.toFixed(3)}Q, reflect=${windowReports.top_left.deltaVsPadBlackWasm.pad_reflect.meanBothEyeNonoccEpeDeltaPx.toFixed(3)}Q | ` +
    `center deltas: replicate=${windowReports.center.deltaVsPadBlackWasm.pad_replicate.meanBothEyeNonoccEpeDeltaPx.toFixed(3)}Q, reflect=${windowReports.center.deltaVsPadBlackWasm.pad_reflect.meanBothEyeNonoccEpeDeltaPx.toFixed(3)}Q.`;

  let strongestNextControlledTest: string;
  if (!nativeGapRemains) {
    strongestNextControlledTest =
      'Native-Q gap largely closed by padding replacement; verify full 741x497 vertical-letterbox SceneFlow eval under replicate/reflect border padding across Teddy/Motorcycle.';
  } else if (tlWfRef !== null && bestAltEpeTL - tlWfRef > 1.0) {
    strongestNextControlledTest =
      'Strongest next controlled test: fixed-320/741 scale stepped-FOV ablation WITH zero side border padding — compare 138x102 isolated box vs 512x384 window vs full 741x497 frame (320/741 scale) using an outer spatial crop that fills all 320x240 model inputs (or symmetric full-canvas FOV expansion), isolating global semantic context / tile-neighborhood structure from padding pixel fill.';
  } else {
    strongestNextControlledTest =
      'Strongest next controlled test: unpadded full-320x240-filled isotropic scale ladder (e.g. 1.0x [320x240 crop] vs 0.75x [427x320 crop -> 320x240] vs 0.50x [640x480 crop -> 320x240]) centered identically on center & top_left with zero border padding, pinpointing whether SceneFlow HITNet degrades monotonically with physical spatial decimation (px/scene-unit) rather than border padding.';
  }

  log(
    `\n=== Padding Ablation Conclusion ===\n` +
      `  ${paddingAblationEffectSummary}\n` +
      `  nativeGapRemains=${String(nativeGapRemains)}\n` +
      `  Next Control: ${strongestNextControlledTest}`
  );

  return {
    ok: true,
    sceneId: 'Motorcycle',
    edition: assets.spec.edition,
    calib: assets.calib,
    pinnedOrtVersion: '1.30.0',
    runtimeAssetFamily: 'asyncify',
    graphOptimizationLevel: 'basic',
    channelOrder: 'C0_RGB',
    scaleFactor: MOTORCYCLE_SCALE_FACTOR_320_741,
    m1ByteLength: params.m1Bytes.byteLength,
    m1Sha256: m1Sha,
    n1OffProof,
    fixtureSha256s: assets.fixtureSha256s,
    windows: windowReports,
    boundedPromisingGpuPairedCheck,
    conclusion: {
      paddingAblationEffectSummary,
      nativeGapRemains,
      strongestNextControlledTest,
    },
  };
}
