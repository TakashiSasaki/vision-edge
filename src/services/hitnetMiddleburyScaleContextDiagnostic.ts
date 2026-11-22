import {
  HITNET_MIDDLEBURY_MODEL_HEIGHT,
  HITNET_MIDDLEBURY_MODEL_WIDTH,
  HITNET_MIDDLEBURY_PIXEL_COUNT,
  loadAndVerifyMiddleburySceneAssets,
  MIDDLEBURY_EVAL3_Q_SCENES,
  packRgbaStereoEyesToHitnetFeed,
  type LoadedMiddleburySceneAssets,
  type MiddleburyCalibData,
  type MiddleburyEyeId,
  type ParsedPfmGrid,
  type ParsedPngMaskGrid,
} from './hitnetMiddleburyEval3Benchmark';
import {
  buildMiddleburyGeometryTransforms,
  type GeometryTransformSpec,
} from './hitnetMiddleburyFactorialDiagnosis';
import {
  decodePngRgbTopToBottomBitExact,
  type DecodedRgbImageGrid,
} from './hitnetMiddleburyRawAndGtDiagnostic';
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

export type ScaleContextWindowId = 'top_left' | 'center';
export type ScaleContextArmKind =
  | 'S0_native_1x'
  | 'S1_downscale_320_741'
  | 'whole_frame_context_ref';

export interface ScaleWindowSpec {
  windowId: ScaleContextWindowId;
  label: string;
  x: number;
  y: number;
  width: 320;
  height: 240;
  note: string;
}

export const MOTORCYCLE_SCALE_FACTOR_320_741 =
  HITNET_MIDDLEBURY_MODEL_WIDTH / MIDDLEBURY_EVAL3_Q_SCENES.Motorcycle.expectedWidth; // 320 / 741
export const MOTORCYCLE_ACTIVE_PAD_GUARD_MODEL_PX = 2;
export const MOTORCYCLE_LOW_DISPARITY_NATIVE_Q_MAX_PX = 24;
export const MOTORCYCLE_LOW_DISPARITY_MODEL12_NATIVE_Q_PX =
  12 / MOTORCYCLE_SCALE_FACTOR_320_741; // 27.7875 native Q px (= 12 model px at 320/741)

export const PREDECLARED_MOTORCYCLE_320X240_WINDOWS: Readonly<
  Record<ScaleContextWindowId, ScaleWindowSpec>
> = {
  top_left: {
    windowId: 'top_left',
    label: 'Top-Left 320x240 Window',
    x: 0,
    y: 0,
    width: 320,
    height: 240,
    note: 'Native-Q [0..320) x [0..240) covering upper-left shelf/frame/wheel trouble area.',
  },
  center: {
    windowId: 'center',
    label: 'Center 320x240 Window',
    x: Math.floor((741 - 320) / 2), // 210
    y: Math.floor((497 - 240) / 2), // 128
    width: 320,
    height: 240,
    note: 'Centered native-Q [210..530) x [128..368) covering foreground motorcycle body and background.',
  },
};

export interface WindowScaleGeometryPlacement {
  windowSpec: ScaleWindowSpec;
  armKind: ScaleContextArmKind;
  scaleX: number;
  scaleY: number;
  dstRect: { x: number; y: number; width: number; height: number };
  paddingMode: 'none' | 'centered-symmetric-zero-letterbox-no-external-context';
  padGuardModelPx: number;
  activeColStart: number;
  activeColEnd: number;
  activeRowStart: number;
  activeRowEnd: number;
  activeSrcXMin: number;
  activeSrcXMax: number;
  activeSrcYMin: number;
  activeSrcYMax: number;
  roundingNote: string;
}

export interface ScaleMatchedPhysicalPoint {
  ix: number;
  iy: number;
  lx: number;
  ly: number;
  idxNative1x: number;
  idxDownscale: number;
  idxWholeFrameRef: number;
  gtNativePx: number;
  maskByte: number;
}

export interface WindowEyeSharedSupport {
  windowId: ScaleContextWindowId;
  eye: MiddleburyEyeId;
  outputTensorName: 'reference_output_disparity' | 'secondary_output_disparity';
  s0Geometry: WindowScaleGeometryPlacement;
  s1Geometry: WindowScaleGeometryPlacement;
  wholeFrameGeometry: GeometryTransformSpec;
  counts: {
    rawDownscaleActiveGridPixels: number;
    excludedInvalidOrNonPositiveGtCount: number;
    excludedOverNdispCount: number;
    excludedCounterpartOutsideCropCount: number;
    excludedCounterpartOutsideActiveSupportCount: number;
    sharedAllValidCount: number;
    sharedNonoccCount: number;
    sharedLowDisp24NonoccCount: number;
    sharedLowDispModel12EqNonoccCount: number;
  };
  sharedAllValidPoints: ScaleMatchedPhysicalPoint[];
  sharedNonoccPoints: ScaleMatchedPhysicalPoint[];
}

export interface NativeQDisparityMetrics {
  label: string;
  scaleXUsedForNativeMapping: number;
  evaluatedCount: number;
  finitePredCount: number;
  nonFinitePredCount: number;
  negativePredCount: number;
  invalidPredCount: number;
  invalidPredRate: number;
  epeNativeQPx: number;
  signedBiasNativeQPx: number;
  medianSignedErrorNativeQPx: number;
  p95AbsErrorNativeQPx: number;
  maxAbsErrorNativeQPx: number;
  bad1NativeQRate: number;
  bad3NativeQRate: number;
  /** Model-px equivalents reported alongside Native-Q px for cross-reference only */
  epeModelPx: number;
  signedBiasModelPx: number;
  p95AbsErrorModelPx: number;
  bad1ModelPxRate: number;
  bad3ModelPxRate: number;
}

export interface WindowEyeArmEvaluation {
  eye: MiddleburyEyeId;
  outputTensorName: 'reference_output_disparity' | 'secondary_output_disparity';
  nonocc: NativeQDisparityMetrics;
  lowDisp24Nonocc: NativeQDisparityMetrics;
  lowDispModel12EqNonocc: NativeQDisparityMetrics;
  allValidGt: NativeQDisparityMetrics;
}

export interface ScaleContextModelRunSummary {
  runId: string;
  windowId: ScaleContextWindowId | 'whole_frame';
  armKind: ScaleContextArmKind;
  scaleX: number;
  scaleY: number;
  channelOrder: 'C0_RGB';
  provider: 'wasm' | 'webgpu';
  graphOptimizationLevel: 'basic';
  modelByteLength: number;
  modelSha256: string;
  feedSha256: string;
  configuredProviders: readonly string[];
  fallbackBehavior: 'no-fallback-explicit-error';
  loadedOrtResources: string[];
  foreignOrtResources: string[];
  gpuValidationErrors: string[];
  dispatchObserved: boolean | null;
  gpuKernelProfileSummary: OrtGpuKernelProfileSummary | null;
  createMs: number;
  coldRunMs: number;
  warmRunMs: number[];
  paddedBorderNonZeroCount: number;
}

export interface WindowScaleComparisonReport {
  windowSpec: ScaleWindowSpec;
  leftSupport: WindowEyeSharedSupport['counts'];
  rightSupport: WindowEyeSharedSupport['counts'];
  native1xWasm: {
    run: ScaleContextModelRunSummary;
    left: WindowEyeArmEvaluation;
    right: WindowEyeArmEvaluation;
  };
  downscale320_741Wasm: {
    run: ScaleContextModelRunSummary;
    left: WindowEyeArmEvaluation;
    right: WindowEyeArmEvaluation;
  };
  wholeFrameContextRefOnWindowPoints: {
    note: string;
    runMeta: ScaleContextModelRunSummary;
    left: WindowEyeArmEvaluation;
    right: WindowEyeArmEvaluation;
  };
  deltaDownscaleMinusNativeWasm: {
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
  };
  deltaContextRefMinusDownscaleSameScale: {
    leftNonoccEpeNativeQDeltaPx: number;
    leftNonoccBad3NativeQDeltaPctPts: number;
    leftLowDisp24EpeNativeQDeltaPx: number;
    rightNonoccEpeNativeQDeltaPx: number;
    rightNonoccBad3NativeQDeltaPctPts: number;
    rightLowDisp24EpeNativeQDeltaPx: number;
  };
}

export interface BoundedWebGpuVerificationReport {
  attempted: boolean;
  windowId: ScaleContextWindowId;
  executedArms: Array<{
    armKind: 'S0_native_1x' | 'S1_downscale_320_741';
    run: ScaleContextModelRunSummary;
    left: WindowEyeArmEvaluation;
    right: WindowEyeArmEvaluation;
    rawParityVsSameWindowWasm: {
      reference_output_disparity: RawTensorParityStats;
      secondary_output_disparity: RawTensorParityStats;
    };
  }>;
  realDispatchObserved: boolean;
  failureOrFallbackNote: string | null;
}

export interface HitnetMotorcycleScaleContextReport {
  ok: true;
  sceneId: 'Motorcycle';
  edition: string;
  calib: MiddleburyCalibData;
  pinnedOrtVersion: '1.30.0';
  runtimeAssetFamily: 'asyncify';
  graphOptimizationLevel: 'basic';
  channelOrder: 'C0_RGB';
  m1ByteLength: number;
  m1Sha256: string;
  n1OffProof: N1OffVerificationProof;
  fixtureSha256s: LoadedMiddleburySceneAssets['fixtureSha256s'];
  methodologyAndCausationNote: string;
  windows: Record<ScaleContextWindowId, WindowScaleComparisonReport>;
  wholeFrameContextRefRun: ScaleContextModelRunSummary;
  boundedWebGpuVerification: BoundedWebGpuVerificationReport;
  verdictSummary: {
    interventionEffectObserved: string;
    epistemicBoundary: string;
  };
}

export type WindowScaleFeedBuilder = (params: {
  im0Decoded: DecodedRgbImageGrid;
  im1Decoded: DecodedRgbImageGrid;
  windowSpec: ScaleWindowSpec;
  s0Geometry: WindowScaleGeometryPlacement;
  s1Geometry: WindowScaleGeometryPlacement;
}) => Promise<{
  s0FeedFloat32: Float32Array;
  s0FeedSha256: string;
  s1FeedFloat32: Float32Array;
  s1FeedSha256: string;
  s1PaddedBorderNonZeroCount: number;
}>;

export type WholeFrameContextFeedBuilder = (params: {
  im0Decoded: DecodedRgbImageGrid;
  im1Decoded: DecodedRgbImageGrid;
  wholeFrameGeometry: GeometryTransformSpec;
}) => Promise<{
  feedFloat32: Float32Array;
  feedSha256: string;
  paddedBorderNonZeroCount: number;
}>;

export function buildWindowScalePlacements(
  windowSpec: ScaleWindowSpec
): {
  s0: WindowScaleGeometryPlacement;
  s1: WindowScaleGeometryPlacement;
} {
  const mw = HITNET_MIDDLEBURY_MODEL_WIDTH;
  const mh = HITNET_MIDDLEBURY_MODEL_HEIGHT;
  const scale = MOTORCYCLE_SCALE_FACTOR_320_741; // 320 / 741
  const dstW = windowSpec.width * scale;
  const dstH = windowSpec.height * scale;
  const dstX = (mw - dstW) / 2;
  const dstY = (mh - dstH) / 2;
  const guard = MOTORCYCLE_ACTIVE_PAD_GUARD_MODEL_PX;

  const activeColStart = Math.ceil(dstX) + guard;
  const activeColEnd = Math.floor(dstX + dstW) - 1 - guard;
  const activeRowStart = Math.ceil(dstY) + guard;
  const activeRowEnd = Math.floor(dstY + dstH) - 1 - guard;

  if (
    activeColStart >= activeColEnd ||
    activeRowStart >= activeRowEnd ||
    activeColStart < 0 ||
    activeRowStart < 0
  ) {
    throw new Error(`Invalid active downscaled region for ${windowSpec.windowId}`);
  }

  const mapDownXToLocalSrc = (mx: number): number =>
    Math.max(0, Math.min(windowSpec.width - 1, Math.round((mx - dstX + 0.5) / scale - 0.5)));
  const mapDownYToLocalSrc = (my: number): number =>
    Math.max(0, Math.min(windowSpec.height - 1, Math.round((my - dstY + 0.5) / scale - 0.5)));

  const localXMin = mapDownXToLocalSrc(activeColStart);
  const localXMax = mapDownXToLocalSrc(activeColEnd);
  const localYMin = mapDownYToLocalSrc(activeRowStart);
  const localYMax = mapDownYToLocalSrc(activeRowEnd);

  const activeSrcXMin = windowSpec.x + localXMin;
  const activeSrcXMax = windowSpec.x + localXMax;
  const activeSrcYMin = windowSpec.y + localYMin;
  const activeSrcYMax = windowSpec.y + localYMax;

  const s0: WindowScaleGeometryPlacement = {
    windowSpec,
    armKind: 'S0_native_1x',
    scaleX: 1.0,
    scaleY: 1.0,
    dstRect: { x: 0, y: 0, width: mw, height: mh },
    paddingMode: 'none',
    padGuardModelPx: guard,
    activeColStart: localXMin,
    activeColEnd: localXMax,
    activeRowStart: localYMin,
    activeRowEnd: localYMax,
    activeSrcXMin,
    activeSrcXMax,
    activeSrcYMin,
    activeSrcYMax,
    roundingNote:
      `1:1 native pixel copy of window [${windowSpec.x}..${windowSpec.x + 320}) x [${windowSpec.y}..${windowSpec.y + 240}); ` +
      `evaluated on common guarded interior source box [${activeSrcXMin}..${activeSrcXMax}] x [${activeSrcYMin}..${activeSrcYMax}].`,
  };

  const s1: WindowScaleGeometryPlacement = {
    windowSpec,
    armKind: 'S1_downscale_320_741',
    scaleX: scale,
    scaleY: scale,
    dstRect: { x: dstX, y: dstY, width: dstW, height: dstH },
    paddingMode: 'centered-symmetric-zero-letterbox-no-external-context',
    padGuardModelPx: guard,
    activeColStart,
    activeColEnd,
    activeRowStart,
    activeRowEnd,
    activeSrcXMin,
    activeSrcXMax,
    activeSrcYMin,
    activeSrcYMax,
    roundingNote:
      `Isolated 320x240 window downscaled at exact isotropic scale=320/741=${scale.toFixed(6)} to ` +
      `dstRect=[x=${dstX.toFixed(4)}, y=${dstY.toFixed(4)}, w=${dstW.toFixed(4)}, h=${dstH.toFixed(4)}] on zero #000000 canvas (zero external scene context). ` +
      `Common guard excludes ${guard} model-px inside strict interior: active model cols [${activeColStart}..${activeColEnd}], rows [${activeRowStart}..${activeRowEnd}].`,
  };

  return { s0, s1 };
}

export function buildWindowSharedPhysicalPointsForEye(params: {
  eye: MiddleburyEyeId;
  windowSpec: ScaleWindowSpec;
  gtPfm: ParsedPfmGrid;
  maskNocc: ParsedPngMaskGrid;
  calib: MiddleburyCalibData;
  wholeFrameGeometry: GeometryTransformSpec;
}): WindowEyeSharedSupport {
  const { eye, windowSpec, gtPfm, maskNocc, calib, wholeFrameGeometry } = params;
  const { s0, s1 } = buildWindowScalePlacements(windowSpec);
  const mw = HITNET_MIDDLEBURY_MODEL_WIDTH;
  const mh = HITNET_MIDDLEBURY_MODEL_HEIGHT;
  const scale = s1.scaleX;

  const sharedAllValidPoints: ScaleMatchedPhysicalPoint[] = [];
  const sharedNonoccPoints: ScaleMatchedPhysicalPoint[] = [];
  const seenSrcIdx = new Set<number>();

  let rawDownscaleActiveGridPixels = 0;
  let excludedInvalidOrNonPositiveGtCount = 0;
  let excludedOverNdispCount = 0;
  let excludedCounterpartOutsideCropCount = 0;
  let excludedCounterpartOutsideActiveSupportCount = 0;
  let sharedLowDisp24NonoccCount = 0;
  let sharedLowDispModel12EqNonoccCount = 0;

  for (let myDown = s1.activeRowStart; myDown <= s1.activeRowEnd; myDown++) {
    const vLocal = (myDown - s1.dstRect.y + 0.5) / scale - 0.5;
    const ly = Math.max(0, Math.min(windowSpec.height - 1, Math.round(vLocal)));
    const iy = windowSpec.y + ly;

    const myWhole = Math.max(
      wholeFrameGeometry.activeFullModelRowStart,
      Math.min(
        wholeFrameGeometry.activeFullModelRowEndInclusive,
        Math.round(
          wholeFrameGeometry.dstRect.y +
            ((iy + 0.5 - wholeFrameGeometry.srcRect.y) * wholeFrameGeometry.dstRect.height) /
              wholeFrameGeometry.srcRect.height -
            0.5
        )
      )
    );

    for (let mxDown = s1.activeColStart; mxDown <= s1.activeColEnd; mxDown++) {
      rawDownscaleActiveGridPixels++;
      const uLocal = (mxDown - s1.dstRect.x + 0.5) / scale - 0.5;
      const lx = Math.max(0, Math.min(windowSpec.width - 1, Math.round(uLocal)));
      const ix = windowSpec.x + lx;
      const srcIdx = iy * calib.width + ix;

      if (seenSrcIdx.has(srcIdx)) continue;
      seenSrcIdx.add(srcIdx);

      // Direct discrete native GT lookup; strictly NO GT smoothing or interpolation.
      const gtNativePx = gtPfm.dataTopToBottom[srcIdx];
      if (!Number.isFinite(gtNativePx) || gtNativePx <= 0) {
        excludedInvalidOrNonPositiveGtCount++;
        continue;
      }
      if (gtNativePx > calib.ndisp) {
        excludedOverNdispCount++;
        continue;
      }

      const xMatch = eye === 'left' ? ix - gtNativePx : ix + gtNativePx;
      if (xMatch < windowSpec.x || xMatch > windowSpec.x + windowSpec.width - 1) {
        excludedCounterpartOutsideCropCount++;
        continue;
      }
      if (xMatch < s1.activeSrcXMin || xMatch > s1.activeSrcXMax) {
        excludedCounterpartOutsideActiveSupportCount++;
        continue;
      }

      const mxWhole = Math.max(
        0,
        Math.min(
          mw - 1,
          Math.round(
            wholeFrameGeometry.dstRect.x +
              ((ix + 0.5 - wholeFrameGeometry.srcRect.x) * wholeFrameGeometry.dstRect.width) /
                wholeFrameGeometry.srcRect.width -
              0.5
          )
        )
      );

      if (myWhole < 0 || myWhole >= mh) continue;

      const maskByte = maskNocc.grayTopToBottom[srcIdx];
      const pt: ScaleMatchedPhysicalPoint = {
        ix,
        iy,
        lx,
        ly,
        idxNative1x: ly * mw + lx,
        idxDownscale: myDown * mw + mxDown,
        idxWholeFrameRef: myWhole * mw + mxWhole,
        gtNativePx,
        maskByte,
      };
      sharedAllValidPoints.push(pt);
      if (maskByte === 255) {
        sharedNonoccPoints.push(pt);
        if (gtNativePx < MOTORCYCLE_LOW_DISPARITY_NATIVE_Q_MAX_PX) {
          sharedLowDisp24NonoccCount++;
        }
        if (gtNativePx < MOTORCYCLE_LOW_DISPARITY_MODEL12_NATIVE_Q_PX) {
          sharedLowDispModel12EqNonoccCount++;
        }
      }
    }
  }

  return {
    windowId: windowSpec.windowId,
    eye,
    outputTensorName:
      eye === 'left' ? 'reference_output_disparity' : 'secondary_output_disparity',
    s0Geometry: s0,
    s1Geometry: s1,
    wholeFrameGeometry,
    counts: {
      rawDownscaleActiveGridPixels,
      excludedInvalidOrNonPositiveGtCount,
      excludedOverNdispCount,
      excludedCounterpartOutsideCropCount,
      excludedCounterpartOutsideActiveSupportCount,
      sharedAllValidCount: sharedAllValidPoints.length,
      sharedNonoccCount: sharedNonoccPoints.length,
      sharedLowDisp24NonoccCount,
      sharedLowDispModel12EqNonoccCount,
    },
    sharedAllValidPoints,
    sharedNonoccPoints,
  };
}

export function evaluateWindowArmEyeMetrics(params: {
  pred: Float32Array;
  armKind: ScaleContextArmKind;
  scaleX: number;
  support: WindowEyeSharedSupport;
}): WindowEyeArmEvaluation {
  const { pred, armKind, scaleX, support } = params;
  if (pred.length !== HITNET_MIDDLEBURY_PIXEL_COUNT) {
    throw new Error(
      `Expected prediction length ${HITNET_MIDDLEBURY_PIXEL_COUNT}, got ${pred.length}`
    );
  }

  const pickIdx = (pt: ScaleMatchedPhysicalPoint): number =>
    armKind === 'S0_native_1x'
      ? pt.idxNative1x
      : armKind === 'S1_downscale_320_741'
        ? pt.idxDownscale
        : pt.idxWholeFrameRef;

  const evalSubset = (
    label: string,
    points: ScaleMatchedPhysicalPoint[]
  ): NativeQDisparityMetrics => {
    const absErrsQ: number[] = [];
    const signedErrsQ: number[] = [];

    let finitePredCount = 0;
    let nonFinitePredCount = 0;
    let negativePredCount = 0;
    let absSumQ = 0;
    let signedSumQ = 0;
    let absSumM = 0;
    let signedSumM = 0;
    let maxAbsQ = 0;
    let bad1Q = 0;
    let bad3Q = 0;
    let bad1M = 0;
    let bad3M = 0;
    const absErrsM: number[] = [];

    for (const pt of points) {
      const pModel = pred[pickIdx(pt)];
      if (!Number.isFinite(pModel)) {
        nonFinitePredCount++;
        bad1Q++;
        bad3Q++;
        bad1M++;
        bad3M++;
        absErrsQ.push(Infinity);
        absErrsM.push(Infinity);
        maxAbsQ = Infinity;
        continue;
      }
      finitePredCount++;
      if (pModel < 0) negativePredCount++;

      // Map model-pixel disparity back to native Q pixels via horizontal scale factor.
      const pNativeQ = pModel / scaleX;
      const gtNativeQ = pt.gtNativePx;
      const gtModel = gtNativeQ * scaleX;

      const signedQ = pNativeQ - gtNativeQ;
      const absQ = Math.abs(signedQ);
      const signedM = pModel - gtModel;
      const absM = Math.abs(signedM);

      absErrsQ.push(absQ);
      signedErrsQ.push(signedQ);
      absErrsM.push(absM);

      absSumQ += absQ;
      signedSumQ += signedQ;
      absSumM += absM;
      signedSumM += signedM;

      if (absQ > maxAbsQ) maxAbsQ = absQ;
      if (absQ > 1.0) bad1Q++;
      if (absQ > 3.0) bad3Q++;
      if (absM > 1.0) bad1M++;
      if (absM > 3.0) bad3M++;
    }

    absErrsQ.sort((a, b) => a - b);
    absErrsM.sort((a, b) => a - b);
    signedErrsQ.sort((a, b) => a - b);

    const n = points.length;
    const p95Idx =
      absErrsQ.length > 0
        ? Math.min(absErrsQ.length - 1, Math.floor(0.95 * (absErrsQ.length - 1)))
        : 0;
    const medIdx = signedErrsQ.length > 0 ? Math.floor(signedErrsQ.length / 2) : 0;

    return {
      label,
      scaleXUsedForNativeMapping: scaleX,
      evaluatedCount: n,
      finitePredCount,
      nonFinitePredCount,
      negativePredCount,
      invalidPredCount: nonFinitePredCount + negativePredCount,
      invalidPredRate: n > 0 ? (nonFinitePredCount + negativePredCount) / n : 0,
      epeNativeQPx:
        n === 0 ? NaN : nonFinitePredCount > 0 ? Infinity : absSumQ / finitePredCount,
      signedBiasNativeQPx: finitePredCount > 0 ? signedSumQ / finitePredCount : NaN,
      medianSignedErrorNativeQPx:
        signedErrsQ.length > 0 ? signedErrsQ[medIdx] : NaN,
      p95AbsErrorNativeQPx: absErrsQ.length > 0 ? absErrsQ[p95Idx] : NaN,
      maxAbsErrorNativeQPx: absErrsQ.length > 0 ? maxAbsQ : NaN,
      bad1NativeQRate: n > 0 ? bad1Q / n : 0,
      bad3NativeQRate: n > 0 ? bad3Q / n : 0,
      epeModelPx:
        n === 0 ? NaN : nonFinitePredCount > 0 ? Infinity : absSumM / finitePredCount,
      signedBiasModelPx: finitePredCount > 0 ? signedSumM / finitePredCount : NaN,
      p95AbsErrorModelPx: absErrsM.length > 0 ? absErrsM[p95Idx] : NaN,
      bad1ModelPxRate: n > 0 ? bad1M / n : 0,
      bad3ModelPxRate: n > 0 ? bad3M / n : 0,
    };
  };

  const low24 = support.sharedNonoccPoints.filter(
    (pt) => pt.gtNativePx < MOTORCYCLE_LOW_DISPARITY_NATIVE_Q_MAX_PX
  );
  const low12Eq = support.sharedNonoccPoints.filter(
    (pt) => pt.gtNativePx < MOTORCYCLE_LOW_DISPARITY_MODEL12_NATIVE_Q_PX
  );

  return {
    eye: support.eye,
    outputTensorName: support.outputTensorName,
    nonocc: evalSubset('nonocc_shared', support.sharedNonoccPoints),
    lowDisp24Nonocc: evalSubset('low_disp_lt24Q_nonocc', low24),
    lowDispModel12EqNonocc: evalSubset('low_disp_lt27.79Q_eq12modelPx_nonocc', low12Eq),
    allValidGt: evalSubset('allValidGt_shared', support.sharedAllValidPoints),
  };
}

export function rgbGridWindowToRgba(
  img: DecodedRgbImageGrid,
  x0: number,
  y0: number,
  w: number,
  h: number
): Uint8ClampedArray {
  if (x0 < 0 || y0 < 0 || x0 + w > img.width || y0 + h > img.height) {
    throw new Error(`Window ${x0},${y0},${w}x${h} out of bounds for ${img.width}x${img.height}`);
  }
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let row = 0; row < h; row++) {
    const srcRowOff = ((y0 + row) * img.width + x0) * 3;
    const dstRowOff = row * w * 4;
    for (let col = 0; col < w; col++) {
      const s = srcRowOff + col * 3;
      const d = dstRowOff + col * 4;
      rgba[d] = img.rgbTopToBottom[s];
      rgba[d + 1] = img.rgbTopToBottom[s + 1];
      rgba[d + 2] = img.rgbTopToBottom[s + 2];
      rgba[d + 3] = 255;
    }
  }
  return rgba;
}

export function drawIsolatedRgbaIntoModelCanvasBrowser(
  srcRgba: Uint8ClampedArray,
  srcW: number,
  srcH: number,
  dstRect: { x: number; y: number; width: number; height: number }
): Uint8ClampedArray {
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = srcW;
  srcCanvas.height = srcH;
  const srcCtx = srcCanvas.getContext('2d');
  if (!srcCtx) throw new Error('2D canvas unavailable for isolated source buffer');
  const clampedCopy = new Uint8ClampedArray(srcRgba.length);
  clampedCopy.set(srcRgba);
  srcCtx.putImageData(new ImageData(clampedCopy, srcW, srcH), 0, 0);

  const dstCanvas = document.createElement('canvas');
  dstCanvas.width = HITNET_MIDDLEBURY_MODEL_WIDTH;
  dstCanvas.height = HITNET_MIDDLEBURY_MODEL_HEIGHT;
  const dstCtx = dstCanvas.getContext('2d', { willReadFrequently: true });
  if (!dstCtx) throw new Error('2D canvas unavailable for padded target buffer');

  dstCtx.fillStyle = '#000000';
  dstCtx.fillRect(0, 0, HITNET_MIDDLEBURY_MODEL_WIDTH, HITNET_MIDDLEBURY_MODEL_HEIGHT);
  dstCtx.drawImage(
    srcCanvas,
    0,
    0,
    srcW,
    srcH,
    dstRect.x,
    dstRect.y,
    dstRect.width,
    dstRect.height
  );

  return dstCtx.getImageData(
    0,
    0,
    HITNET_MIDDLEBURY_MODEL_WIDTH,
    HITNET_MIDDLEBURY_MODEL_HEIGHT
  ).data;
}

export function countNonZeroOutsideRect(
  rgba: Uint8ClampedArray,
  rect: { x: number; y: number; width: number; height: number }
): number {
  const mw = HITNET_MIDDLEBURY_MODEL_WIDTH;
  const mh = HITNET_MIDDLEBURY_MODEL_HEIGHT;
  const xMin = Math.floor(rect.x);
  const xMax = Math.ceil(rect.x + rect.width) - 1;
  const yMin = Math.floor(rect.y);
  const yMax = Math.ceil(rect.y + rect.height) - 1;

  let nonZero = 0;
  for (let y = 0; y < mh; y++) {
    for (let x = 0; x < mw; x++) {
      if (x >= xMin && x <= xMax && y >= yMin && y <= yMax) continue;
      const off = (y * mw + x) * 4;
      if (rgba[off] !== 0 || rgba[off + 1] !== 0 || rgba[off + 2] !== 0) {
        nonZero++;
      }
    }
  }
  return nonZero;
}

export const defaultBrowserWindowScaleFeedBuilder: WindowScaleFeedBuilder = async ({
  im0Decoded,
  im1Decoded,
  windowSpec,
  s1Geometry,
}) => {
  // Isolate exact 320x240 source window FIRST so external pixels can never bleed across crop borders.
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

  const s0FeedFloat32 = packRgbaStereoEyesToHitnetFeed(winLeftRgba, winRightRgba);
  const s0FeedSha256 = await sha256HexBytes(
    new Uint8Array(s0FeedFloat32.buffer, s0FeedFloat32.byteOffset, s0FeedFloat32.byteLength)
  );

  const downLeftRgba = drawIsolatedRgbaIntoModelCanvasBrowser(
    winLeftRgba,
    windowSpec.width,
    windowSpec.height,
    s1Geometry.dstRect
  );
  const downRightRgba = drawIsolatedRgbaIntoModelCanvasBrowser(
    winRightRgba,
    windowSpec.width,
    windowSpec.height,
    s1Geometry.dstRect
  );

  const s1PaddedBorderNonZeroCount =
    countNonZeroOutsideRect(downLeftRgba, s1Geometry.dstRect) +
    countNonZeroOutsideRect(downRightRgba, s1Geometry.dstRect);
  if (s1PaddedBorderNonZeroCount !== 0) {
    throw new Error(
      `${windowSpec.windowId} S1 symmetric zero padding corrupted: nonZeroBorderPixels=${s1PaddedBorderNonZeroCount}`
    );
  }

  const s1FeedFloat32 = packRgbaStereoEyesToHitnetFeed(downLeftRgba, downRightRgba);
  const s1FeedSha256 = await sha256HexBytes(
    new Uint8Array(s1FeedFloat32.buffer, s1FeedFloat32.byteOffset, s1FeedFloat32.byteLength)
  );

  return {
    s0FeedFloat32,
    s0FeedSha256,
    s1FeedFloat32,
    s1FeedSha256,
    s1PaddedBorderNonZeroCount,
  };
};

export const defaultBrowserWholeFrameContextFeedBuilder: WholeFrameContextFeedBuilder = async ({
  im0Decoded,
  im1Decoded,
  wholeFrameGeometry,
}) => {
  const fullLeftRgba = rgbGridWindowToRgba(
    im0Decoded,
    0,
    0,
    im0Decoded.width,
    im0Decoded.height
  );
  const fullRightRgba = rgbGridWindowToRgba(
    im1Decoded,
    0,
    0,
    im1Decoded.width,
    im1Decoded.height
  );

  const dstLeftRgba = drawIsolatedRgbaIntoModelCanvasBrowser(
    fullLeftRgba,
    im0Decoded.width,
    im0Decoded.height,
    wholeFrameGeometry.dstRect
  );
  const dstRightRgba = drawIsolatedRgbaIntoModelCanvasBrowser(
    fullRightRgba,
    im1Decoded.width,
    im1Decoded.height,
    wholeFrameGeometry.dstRect
  );

  const paddedBorderNonZeroCount =
    countNonZeroOutsideRect(dstLeftRgba, wholeFrameGeometry.dstRect) +
    countNonZeroOutsideRect(dstRightRgba, wholeFrameGeometry.dstRect);

  const feedFloat32 = packRgbaStereoEyesToHitnetFeed(dstLeftRgba, dstRightRgba);
  const feedSha256 = await sha256HexBytes(
    new Uint8Array(feedFloat32.buffer, feedFloat32.byteOffset, feedFloat32.byteLength)
  );
  return { feedFloat32, feedSha256, paddedBorderNonZeroCount };
};

function hasObservedGpuDispatch(prof: OrtVersionDiagnosticResponse | null): boolean {
  if (!prof || !prof.ok || !prof.gpuKernelProfileSummary) return false;
  const s = prof.gpuKernelProfileSummary;
  return (
    s.webgpuComputePassCount > 0 ||
    s.webgpuDispatchWorkgroupsCount > 0 ||
    s.ortProfilingEventCount > 0
  );
}

export async function runHitnetMiddleburyScaleContextDiagnostic(params: {
  m1Bytes: Uint8Array;
  isDev?: boolean;
  includeWebGpuVerification?: boolean;
  disableCpuEpFallback?: boolean;
  runWorker?: ResidualDiagnosticWorkerRunner;
  loadMotorcycleAssets?: () => Promise<LoadedMiddleburySceneAssets>;
  buildWindowFeeds?: WindowScaleFeedBuilder;
  buildWholeFrameFeed?: WholeFrameContextFeedBuilder;
  onLog?: (line: string) => void;
}): Promise<HitnetMotorcycleScaleContextReport> {
  assertDevFixtureEnabled(params.isDev);
  const log = params.onLog ?? (() => {});
  const runWorker = params.runWorker ?? createBrowserOrt130DiagnosticRunner();
  const loadMotorcycle =
    params.loadMotorcycleAssets ?? (() => loadAndVerifyMiddleburySceneAssets('Motorcycle'));
  const buildWindowFeeds =
    params.buildWindowFeeds ?? defaultBrowserWindowScaleFeedBuilder;
  const buildWholeFrameFeed =
    params.buildWholeFrameFeed ?? defaultBrowserWholeFrameContextFeedBuilder;
  const includeWebGpuVerification = params.includeWebGpuVerification ?? true;
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
  const methodologyAndCausationNote =
    'Controlled within-window comparison isolating 320x240 native Q windows (top_left [0..320)x[0..240), center [210..530)x[128..368)) with ZERO external scene pixels. ' +
    'S0 (native 1:1) vs S1 (exact same window downscaled by isotropic 320/741 + centered symmetric #000000 padding, common padding guard=2 model-px) measures the combined downsample+padding intervention; ' +
    'whole-frame 320/741 isotropic letterbox serves strictly as an external-context reference evaluated at identical physical source points (not a pure scale comparator). No scale-only causation beyond this combined intervention is claimed.';

  log(
    `=== [DEV-ONLY] Motorcycle Native-Q Scale/Context-Controlled Comparison (Canonical M1, C0_RGB) ===\n` +
      `Runtime: onnxruntime-web@1.30.0 (${asyncify.entryFile} + ${asyncify.wasmFile}) | opt=basic | M1=${m1Sha} (N1=OFF, 0 replacements)\n` +
      `Policy: ${methodologyAndCausationNote}`
  );

  const executeModelRun = async (req: {
    runId: string;
    windowId: ScaleContextWindowId | 'whole_frame';
    armKind: ScaleContextArmKind;
    scaleX: number;
    scaleY: number;
    provider: 'wasm' | 'webgpu';
    feedFloat32: Float32Array;
    feedSha256: string;
    paddedBorderNonZeroCount: number;
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
        `${req.runId} runtime/provider/opt guard violation: provider=${okRes.provider}, configured=${JSON.stringify(okRes.configuredProviders)}`
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
        armKind: req.armKind,
        scaleX: req.scaleX,
        scaleY: req.scaleY,
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
        paddedBorderNonZeroCount: req.paddedBorderNonZeroCount,
      },
    };
  };

  // Single non-duplicated whole-frame isotropic 320/741 WASM run (context reference)
  const wholeBuilt = await buildWholeFrameFeed({
    im0Decoded,
    im1Decoded,
    wholeFrameGeometry,
  });
  const wholeExec = await executeModelRun({
    runId: 'whole_frame_context_ref_wasm',
    windowId: 'whole_frame',
    armKind: 'whole_frame_context_ref',
    scaleX: wholeFrameGeometry.scaleX,
    scaleY: wholeFrameGeometry.scaleY,
    provider: 'wasm',
    feedFloat32: wholeBuilt.feedFloat32,
    feedSha256: wholeBuilt.feedSha256,
    paddedBorderNonZeroCount: wholeBuilt.paddedBorderNonZeroCount,
  });

  log(
    `[Whole-Frame Context Ref WASM] feed=${wholeExec.summary.feedSha256} cold=${wholeExec.summary.coldRunMs.toFixed(1)}ms warm=${JSON.stringify(wholeExec.summary.warmRunMs.map((n) => Number(n.toFixed(1))))}`
  );

  const windowIds: ScaleContextWindowId[] = ['top_left', 'center'];
  const windowsReport = {} as Record<ScaleContextWindowId, WindowScaleComparisonReport>;
  const executedFeedShas = new Set<string>([wholeBuilt.feedSha256]);

  let topLeftFeedsCache: {
    s0FeedFloat32: Float32Array;
    s0FeedSha256: string;
    s1FeedFloat32: Float32Array;
    s1FeedSha256: string;
    s1PaddedBorderNonZeroCount: number;
    leftSupport: WindowEyeSharedSupport;
    rightSupport: WindowEyeSharedSupport;
    s0WasmLeftOut: Float32Array;
    s0WasmRightOut: Float32Array;
    s1WasmLeftOut: Float32Array;
    s1WasmRightOut: Float32Array;
  } | null = null;

  for (const windowId of windowIds) {
    const windowSpec = PREDECLARED_MOTORCYCLE_320X240_WINDOWS[windowId];
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

    const built = await buildWindowFeeds({
      im0Decoded,
      im1Decoded,
      windowSpec,
      s0Geometry: leftSupport.s0Geometry,
      s1Geometry: leftSupport.s1Geometry,
    });

    if (built.s0FeedSha256 === built.s1FeedSha256) {
      throw new Error(`${windowId}: S0 and S1 feed hashes unexpectedly match (${built.s0FeedSha256})`);
    }
    if (executedFeedShas.has(built.s0FeedSha256) || executedFeedShas.has(built.s1FeedSha256)) {
      throw new Error(`Duplicate feed SHA detected for window=${windowId}`);
    }
    executedFeedShas.add(built.s0FeedSha256);
    executedFeedShas.add(built.s1FeedSha256);

    const s0Exec = await executeModelRun({
      runId: `${windowId}_S0_native_1x_wasm`,
      windowId,
      armKind: 'S0_native_1x',
      scaleX: 1.0,
      scaleY: 1.0,
      provider: 'wasm',
      feedFloat32: built.s0FeedFloat32,
      feedSha256: built.s0FeedSha256,
      paddedBorderNonZeroCount: 0,
    });
    const s1Exec = await executeModelRun({
      runId: `${windowId}_S1_downscale_320_741_wasm`,
      windowId,
      armKind: 'S1_downscale_320_741',
      scaleX: MOTORCYCLE_SCALE_FACTOR_320_741,
      scaleY: MOTORCYCLE_SCALE_FACTOR_320_741,
      provider: 'wasm',
      feedFloat32: built.s1FeedFloat32,
      feedSha256: built.s1FeedSha256,
      paddedBorderNonZeroCount: built.s1PaddedBorderNonZeroCount,
    });

    const s0Left = evaluateWindowArmEyeMetrics({
      pred: s0Exec.leftOut,
      armKind: 'S0_native_1x',
      scaleX: 1.0,
      support: leftSupport,
    });
    const s0Right = evaluateWindowArmEyeMetrics({
      pred: s0Exec.rightOut,
      armKind: 'S0_native_1x',
      scaleX: 1.0,
      support: rightSupport,
    });

    const s1Left = evaluateWindowArmEyeMetrics({
      pred: s1Exec.leftOut,
      armKind: 'S1_downscale_320_741',
      scaleX: MOTORCYCLE_SCALE_FACTOR_320_741,
      support: leftSupport,
    });
    const s1Right = evaluateWindowArmEyeMetrics({
      pred: s1Exec.rightOut,
      armKind: 'S1_downscale_320_741',
      scaleX: MOTORCYCLE_SCALE_FACTOR_320_741,
      support: rightSupport,
    });

    const wfLeft = evaluateWindowArmEyeMetrics({
      pred: wholeExec.leftOut,
      armKind: 'whole_frame_context_ref',
      scaleX: wholeFrameGeometry.scaleX,
      support: leftSupport,
    });
    const wfRight = evaluateWindowArmEyeMetrics({
      pred: wholeExec.rightOut,
      armKind: 'whole_frame_context_ref',
      scaleX: wholeFrameGeometry.scaleX,
      support: rightSupport,
    });

    if (windowId === 'top_left') {
      topLeftFeedsCache = {
        ...built,
        leftSupport,
        rightSupport,
        s0WasmLeftOut: s0Exec.leftOut,
        s0WasmRightOut: s0Exec.rightOut,
        s1WasmLeftOut: s1Exec.leftOut,
        s1WasmRightOut: s1Exec.rightOut,
      };
    }

    const fmtMetrics = (tag: string, m: WindowEyeArmEvaluation) =>
      `${tag} nonocc(n=${m.nonocc.evaluatedCount}): EPE=${m.nonocc.epeNativeQPx.toFixed(3)}Q bias=${m.nonocc.signedBiasNativeQPx.toFixed(3)}Q p95=${m.nonocc.p95AbsErrorNativeQPx.toFixed(2)}Q Bad1=${(m.nonocc.bad1NativeQRate * 100).toFixed(2)}% Bad3=${(m.nonocc.bad3NativeQRate * 100).toFixed(2)}% inv=${m.nonocc.invalidPredCount} | low24Q(n=${m.lowDisp24Nonocc.evaluatedCount}): EPE=${m.lowDisp24Nonocc.epeNativeQPx.toFixed(3)}Q Bad3=${(m.lowDisp24Nonocc.bad3NativeQRate * 100).toFixed(2)}%`;

    log(
      `\n=== Window: ${windowSpec.windowId} [${windowSpec.x}..${windowSpec.x + 320})x[${windowSpec.y}..${windowSpec.y + 240}) ===\n` +
        `  Shared Support: Left nonocc=${leftSupport.counts.sharedNonoccCount} (low<24Q=${leftSupport.counts.sharedLowDisp24NonoccCount}, low<27.8Q=${leftSupport.counts.sharedLowDispModel12EqNonoccCount}, allValid=${leftSupport.counts.sharedAllValidCount}) | ` +
        `Right nonocc=${rightSupport.counts.sharedNonoccCount} (low<24Q=${rightSupport.counts.sharedLowDisp24NonoccCount}, allValid=${rightSupport.counts.sharedAllValidCount})\n` +
        `  S0 native 1:1       feed=${s0Exec.summary.feedSha256} cold=${s0Exec.summary.coldRunMs.toFixed(1)}ms warm=${JSON.stringify(s0Exec.summary.warmRunMs.map((n) => Number(n.toFixed(1))))}\n` +
        `    ${fmtMetrics('L', s0Left)}\n` +
        `    ${fmtMetrics('R', s0Right)}\n` +
        `  S1 downscale 320/741 feed=${s1Exec.summary.feedSha256} cold=${s1Exec.summary.coldRunMs.toFixed(1)}ms warm=${JSON.stringify(s1Exec.summary.warmRunMs.map((n) => Number(n.toFixed(1))))}\n` +
        `    ${fmtMetrics('L', s1Left)}\n` +
        `    ${fmtMetrics('R', s1Right)}\n` +
        `  Whole-Frame Ref [Context Reference Only, same 320/741 scale]\n` +
        `    ${fmtMetrics('L', wfLeft)}\n` +
        `    ${fmtMetrics('R', wfRight)}`
    );

    windowsReport[windowId] = {
      windowSpec,
      leftSupport: leftSupport.counts,
      rightSupport: rightSupport.counts,
      native1xWasm: { run: s0Exec.summary, left: s0Left, right: s0Right },
      downscale320_741Wasm: { run: s1Exec.summary, left: s1Left, right: s1Right },
      wholeFrameContextRefOnWindowPoints: {
        note: 'Full 741x497 Motorcycle image scaled isotropically 320/741 WITH full external scene context, evaluated on identical window points (context reference only; not pure scale comparator).',
        runMeta: wholeExec.summary,
        left: wfLeft,
        right: wfRight,
      },
      deltaDownscaleMinusNativeWasm: {
        leftNonoccEpeNativeQDeltaPx: s1Left.nonocc.epeNativeQPx - s0Left.nonocc.epeNativeQPx,
        leftNonoccSignedBiasNativeQDeltaPx:
          s1Left.nonocc.signedBiasNativeQPx - s0Left.nonocc.signedBiasNativeQPx,
        leftNonoccP95NativeQDeltaPx:
          s1Left.nonocc.p95AbsErrorNativeQPx - s0Left.nonocc.p95AbsErrorNativeQPx,
        leftNonoccBad1NativeQDeltaPctPts:
          (s1Left.nonocc.bad1NativeQRate - s0Left.nonocc.bad1NativeQRate) * 100,
        leftNonoccBad3NativeQDeltaPctPts:
          (s1Left.nonocc.bad3NativeQRate - s0Left.nonocc.bad3NativeQRate) * 100,
        leftLowDisp24EpeNativeQDeltaPx:
          s1Left.lowDisp24Nonocc.epeNativeQPx - s0Left.lowDisp24Nonocc.epeNativeQPx,
        leftLowDisp24Bad3NativeQDeltaPctPts:
          (s1Left.lowDisp24Nonocc.bad3NativeQRate - s0Left.lowDisp24Nonocc.bad3NativeQRate) * 100,
        rightNonoccEpeNativeQDeltaPx: s1Right.nonocc.epeNativeQPx - s0Right.nonocc.epeNativeQPx,
        rightNonoccSignedBiasNativeQDeltaPx:
          s1Right.nonocc.signedBiasNativeQPx - s0Right.nonocc.signedBiasNativeQPx,
        rightNonoccP95NativeQDeltaPx:
          s1Right.nonocc.p95AbsErrorNativeQPx - s0Right.nonocc.p95AbsErrorNativeQPx,
        rightNonoccBad1NativeQDeltaPctPts:
          (s1Right.nonocc.bad1NativeQRate - s0Right.nonocc.bad1NativeQRate) * 100,
        rightNonoccBad3NativeQDeltaPctPts:
          (s1Right.nonocc.bad3NativeQRate - s0Right.nonocc.bad3NativeQRate) * 100,
        rightLowDisp24EpeNativeQDeltaPx:
          s1Right.lowDisp24Nonocc.epeNativeQPx - s0Right.lowDisp24Nonocc.epeNativeQPx,
        rightLowDisp24Bad3NativeQDeltaPctPts:
          (s1Right.lowDisp24Nonocc.bad3NativeQRate - s0Right.lowDisp24Nonocc.bad3NativeQRate) * 100,
      },
      deltaContextRefMinusDownscaleSameScale: {
        leftNonoccEpeNativeQDeltaPx: wfLeft.nonocc.epeNativeQPx - s1Left.nonocc.epeNativeQPx,
        leftNonoccBad3NativeQDeltaPctPts:
          (wfLeft.nonocc.bad3NativeQRate - s1Left.nonocc.bad3NativeQRate) * 100,
        leftLowDisp24EpeNativeQDeltaPx:
          wfLeft.lowDisp24Nonocc.epeNativeQPx - s1Left.lowDisp24Nonocc.epeNativeQPx,
        rightNonoccEpeNativeQDeltaPx: wfRight.nonocc.epeNativeQPx - s1Right.nonocc.epeNativeQPx,
        rightNonoccBad3NativeQDeltaPctPts:
          (wfRight.nonocc.bad3NativeQRate - s1Right.nonocc.bad3NativeQRate) * 100,
        rightLowDisp24EpeNativeQDeltaPx:
          wfRight.lowDisp24Nonocc.epeNativeQPx - s1Right.lowDisp24Nonocc.epeNativeQPx,
      },
    };
  }

  // Bounded WebGPU verification on top_left window (S0 vs S1, with 1 real dispatch proof)
  const boundedWebGpuVerification: BoundedWebGpuVerificationReport = {
    attempted: false,
    windowId: 'top_left',
    executedArms: [],
    realDispatchObserved: false,
    failureOrFallbackNote: null,
  };

  if (includeWebGpuVerification && topLeftFeedsCache) {
    boundedWebGpuVerification.attempted = true;
    try {
      const gpuS0 = await executeModelRun({
        runId: 'top_left_S0_native_1x_webgpu',
        windowId: 'top_left',
        armKind: 'S0_native_1x',
        scaleX: 1.0,
        scaleY: 1.0,
        provider: 'webgpu',
        feedFloat32: topLeftFeedsCache.s0FeedFloat32,
        feedSha256: topLeftFeedsCache.s0FeedSha256,
        paddedBorderNonZeroCount: 0,
        verifyGpuDispatch: true,
      });
      const gpuS1 = await executeModelRun({
        runId: 'top_left_S1_downscale_320_741_webgpu',
        windowId: 'top_left',
        armKind: 'S1_downscale_320_741',
        scaleX: MOTORCYCLE_SCALE_FACTOR_320_741,
        scaleY: MOTORCYCLE_SCALE_FACTOR_320_741,
        provider: 'webgpu',
        feedFloat32: topLeftFeedsCache.s1FeedFloat32,
        feedSha256: topLeftFeedsCache.s1FeedSha256,
        paddedBorderNonZeroCount: topLeftFeedsCache.s1PaddedBorderNonZeroCount,
        verifyGpuDispatch: false,
      });

      const s0GpuLeft = evaluateWindowArmEyeMetrics({
        pred: gpuS0.leftOut,
        armKind: 'S0_native_1x',
        scaleX: 1.0,
        support: topLeftFeedsCache.leftSupport,
      });
      const s0GpuRight = evaluateWindowArmEyeMetrics({
        pred: gpuS0.rightOut,
        armKind: 'S0_native_1x',
        scaleX: 1.0,
        support: topLeftFeedsCache.rightSupport,
      });
      const s1GpuLeft = evaluateWindowArmEyeMetrics({
        pred: gpuS1.leftOut,
        armKind: 'S1_downscale_320_741',
        scaleX: MOTORCYCLE_SCALE_FACTOR_320_741,
        support: topLeftFeedsCache.leftSupport,
      });
      const s1GpuRight = evaluateWindowArmEyeMetrics({
        pred: gpuS1.rightOut,
        armKind: 'S1_downscale_320_741',
        scaleX: MOTORCYCLE_SCALE_FACTOR_320_741,
        support: topLeftFeedsCache.rightSupport,
      });

      boundedWebGpuVerification.executedArms.push(
        {
          armKind: 'S0_native_1x',
          run: gpuS0.summary,
          left: s0GpuLeft,
          right: s0GpuRight,
          rawParityVsSameWindowWasm: {
            reference_output_disparity: summarizeRawTensorAndParity(
              gpuS0.leftOut,
              topLeftFeedsCache.s0WasmLeftOut
            ),
            secondary_output_disparity: summarizeRawTensorAndParity(
              gpuS0.rightOut,
              topLeftFeedsCache.s0WasmRightOut
            ),
          },
        },
        {
          armKind: 'S1_downscale_320_741',
          run: gpuS1.summary,
          left: s1GpuLeft,
          right: s1GpuRight,
          rawParityVsSameWindowWasm: {
            reference_output_disparity: summarizeRawTensorAndParity(
              gpuS1.leftOut,
              topLeftFeedsCache.s1WasmLeftOut
            ),
            secondary_output_disparity: summarizeRawTensorAndParity(
              gpuS1.rightOut,
              topLeftFeedsCache.s1WasmRightOut
            ),
          },
        }
      );

      boundedWebGpuVerification.realDispatchObserved =
        gpuS0.summary.dispatchObserved === true &&
        gpuS0.summary.fallbackBehavior === 'no-fallback-explicit-error' &&
        gpuS1.summary.fallbackBehavior === 'no-fallback-explicit-error';

      log(
        `\n[Bounded WebGPU Verification (top_left)] realDispatchObserved=${String(boundedWebGpuVerification.realDispatchObserved)} | ` +
          `S0 EPE(L=${s0GpuLeft.nonocc.epeNativeQPx.toFixed(3)}Q, R=${s0GpuRight.nonocc.epeNativeQPx.toFixed(3)}Q, gpuVsWasmMax=${boundedWebGpuVerification.executedArms[0].rawParityVsSameWindowWasm.reference_output_disparity.maxAbsDiff.toExponential(2)}) | ` +
          `S1 EPE(L=${s1GpuLeft.nonocc.epeNativeQPx.toFixed(3)}Q, R=${s1GpuRight.nonocc.epeNativeQPx.toFixed(3)}Q, gpuVsWasmMax=${boundedWebGpuVerification.executedArms[1].rawParityVsSameWindowWasm.reference_output_disparity.maxAbsDiff.toExponential(2)})`
      );
    } catch (err) {
      boundedWebGpuVerification.failureOrFallbackNote =
        err instanceof Error ? err.message : String(err);
      log(
        `\n[Bounded WebGPU Verification (top_left)] Unavailable/Failed cleanly without altering WASM comparison: ${boundedWebGpuVerification.failureOrFallbackNote}`
      );
    }
  }

  const tlDelta = windowsReport.top_left.deltaDownscaleMinusNativeWasm;
  const ctDelta = windowsReport.center.deltaDownscaleMinusNativeWasm;

  const verdictSummary = {
    interventionEffectObserved:
      `top_left delta(S1 downscale+pad minus S0 native 1:1): Left EPE ${tlDelta.leftNonoccEpeNativeQDeltaPx >= 0 ? '+' : ''}${tlDelta.leftNonoccEpeNativeQDeltaPx.toFixed(3)}Q (Bad3 ${tlDelta.leftNonoccBad3NativeQDeltaPctPts >= 0 ? '+' : ''}${tlDelta.leftNonoccBad3NativeQDeltaPctPts.toFixed(2)}pp), ` +
      `Right EPE ${tlDelta.rightNonoccEpeNativeQDeltaPx >= 0 ? '+' : ''}${tlDelta.rightNonoccEpeNativeQDeltaPx.toFixed(3)}Q | ` +
      `center delta(S1 minus S0): Left EPE ${ctDelta.leftNonoccEpeNativeQDeltaPx >= 0 ? '+' : ''}${ctDelta.leftNonoccEpeNativeQDeltaPx.toFixed(3)}Q (Bad3 ${ctDelta.leftNonoccBad3NativeQDeltaPctPts >= 0 ? '+' : ''}${ctDelta.leftNonoccBad3NativeQDeltaPctPts.toFixed(2)}pp), ` +
      `Right EPE ${ctDelta.rightNonoccEpeNativeQDeltaPx >= 0 ? '+' : ''}${ctDelta.rightNonoccEpeNativeQDeltaPx.toFixed(3)}Q.`,
    epistemicBoundary:
      'Observed differences between S0_native_1x and S1_downscale_320_741 reflect the combined downsample (320/741) + symmetric zero-letterbox padding intervention inside identical 320x240 windows with zero external context; pure scale causation independent of symmetric border padding or effective FOV-in-model is NOT asserted.',
  };

  log(`\n=== Scale/Context Verdict ===\n  ${verdictSummary.interventionEffectObserved}\n  Boundary: ${verdictSummary.epistemicBoundary}`);

  return {
    ok: true,
    sceneId: 'Motorcycle',
    edition: assets.spec.edition,
    calib: assets.calib,
    pinnedOrtVersion: '1.30.0',
    runtimeAssetFamily: 'asyncify',
    graphOptimizationLevel: 'basic',
    channelOrder: 'C0_RGB',
    m1ByteLength: params.m1Bytes.byteLength,
    m1Sha256: m1Sha,
    n1OffProof,
    fixtureSha256s: assets.fixtureSha256s,
    methodologyAndCausationNote,
    windows: windowsReport,
    wholeFrameContextRefRun: wholeExec.summary,
    boundedWebGpuVerification,
    verdictSummary,
  };
}
