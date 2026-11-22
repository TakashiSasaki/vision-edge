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
  decodePngRgbTopToBottomBitExact,
  type DecodedRgbImageGrid,
} from './hitnetMiddleburyRawAndGtDiagnostic';
import {
  MOTORCYCLE_ACTIVE_PAD_GUARD_MODEL_PX,
  MOTORCYCLE_LOW_DISPARITY_MODEL12_NATIVE_Q_PX,
  MOTORCYCLE_LOW_DISPARITY_NATIVE_Q_MAX_PX,
  type NativeQDisparityMetrics,
  type ScaleContextModelRunSummary,
  type WindowEyeArmEvaluation,
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

export type NoPadLadderLocationId = 'upper_left' | 'center';
export type NoPadLadderScaleId = 'scale_1_00' | 'scale_0_75' | 'scale_0_50';

export const NO_PAD_LADDER_SCALES: ReadonlyArray<{
  scaleId: NoPadLadderScaleId;
  scale: number;
  label: string;
}> = [
  { scaleId: 'scale_1_00', scale: 1.0, label: '1.00x (320x240 native)' },
  { scaleId: 'scale_0_75', scale: 0.75, label: '0.75x (426.667x320 -> 320x240)' },
  { scaleId: 'scale_0_50', scale: 0.5, label: '0.50x (640x480 -> 320x240)' },
];

export interface FractionalSourceCropRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface NoPadLadderArmPlacement {
  locationId: NoPadLadderLocationId;
  scaleId: NoPadLadderScaleId;
  scaleX: number;
  scaleY: number;
  srcRect: FractionalSourceCropRect;
  dstRect: { x: 0; y: 0; width: 320; height: 240 };
  placementPolicy: 'top-left-anchored-nested' | 'center-anchored-nested';
  paddingPixelsPerEye: 0;
  contextExpansionNote: string;
}

export interface NoPadLadderPhysicalPoint {
  ix: number;
  iy: number;
  gtNativePx: number;
  maskByte: number;
  idxByScale: Record<NoPadLadderScaleId, number>;
}

export interface NoPadLadderEyeSharedSupport {
  locationId: NoPadLadderLocationId;
  eye: MiddleburyEyeId;
  outputTensorName: 'reference_output_disparity' | 'secondary_output_disparity';
  marginModelPx: number;
  activeSrcBoxIntersection: {
    xMin: number;
    xMax: number;
    yMin: number;
    yMax: number;
  };
  counts: {
    candidatePhysicalPixelsInIntersection: number;
    excludedMarginOrOutOfBoundsCount: number;
    excludedInvalidOrNonPositiveGtCount: number;
    excludedOverNdispCount: number;
    excludedCounterpartOutsideNativeCropOrMarginCount: number;
    sharedAllValidCount: number;
    sharedNonoccCount: number;
    sharedLowDisp24NonoccCount: number;
    sharedLowDispModel12EqNonoccCount: number;
  };
  supportSha256: string;
  sharedAllValidPoints: NoPadLadderPhysicalPoint[];
  sharedNonoccPoints: NoPadLadderPhysicalPoint[];
}

export interface NoPadLadderArmRunSummary {
  placement: NoPadLadderArmPlacement;
  run: Omit<ScaleContextModelRunSummary, 'windowId' | 'armKind'> & {
    locationId: NoPadLadderLocationId | 'shifted_context_control';
    scaleId: NoPadLadderScaleId | 'shifted_left_top_1_00';
    paddingPixels: 0;
  };
  left: WindowEyeArmEvaluation;
  right: WindowEyeArmEvaluation;
}

export interface NoPadScaleDeltaVsNative1x {
  scaleId: 'scale_0_75' | 'scale_0_50';
  scale: number;
  leftNonoccEpeNativeQDeltaPx: number;
  leftNonoccSignedBiasNativeQDeltaPx: number;
  leftNonoccP95NativeQDeltaPx: number;
  leftNonoccBad1NativeQDeltaPctPts: number;
  leftNonoccBad3NativeQDeltaPctPts: number;
  leftLowDisp24EpeNativeQDeltaPx: number;
  rightNonoccEpeNativeQDeltaPx: number;
  rightNonoccSignedBiasNativeQDeltaPx: number;
  rightNonoccP95NativeQDeltaPx: number;
  rightNonoccBad1NativeQDeltaPctPts: number;
  rightNonoccBad3NativeQDeltaPctPts: number;
  rightLowDisp24EpeNativeQDeltaPx: number;
  meanBothEyeNonoccEpeDeltaPx: number;
}

export interface LocationNoPadScaleLadderReport {
  locationId: NoPadLadderLocationId;
  placementSummary: string;
  contextChangesExplicitNote: string;
  placements: Record<NoPadLadderScaleId, NoPadLadderArmPlacement>;
  leftSupport: NoPadLadderEyeSharedSupport['counts'] & {
    supportSha256: string;
    activeSrcBoxIntersection: NoPadLadderEyeSharedSupport['activeSrcBoxIntersection'];
  };
  rightSupport: NoPadLadderEyeSharedSupport['counts'] & {
    supportSha256: string;
    activeSrcBoxIntersection: NoPadLadderEyeSharedSupport['activeSrcBoxIntersection'];
  };
  wasmArms: Record<NoPadLadderScaleId, NoPadLadderArmRunSummary>;
  deltasVsScale1_00Wasm: {
    scale_0_75: NoPadScaleDeltaVsNative1x;
    scale_0_50: NoPadScaleDeltaVsNative1x;
  };
}

export interface SameScaleShiftedCropPoint {
  ix: number;
  iy: number;
  gtNativePx: number;
  maskByte: number;
  idxShiftedA: number;
  idxCenterB: number;
}

export interface SameScaleShiftedCropContextControlReport {
  controlKind: 'same-scale-1x-shifted-crop-context-only';
  scale: 1.0;
  cropA: {
    cropId: 'shifted_left_top_1_00';
    srcRect: { x: number; y: number; width: 320; height: 240 };
  };
  cropB: {
    cropId: 'center_scale_1_00';
    reusedFromCenterLadder: true;
    srcRect: { x: number; y: number; width: 320; height: 240 };
  };
  intersectionBoxNativeQ: {
    xMin: number;
    xMax: number;
    yMin: number;
    yMax: number;
    marginModelPx: number;
    widthPx: number;
    heightPx: number;
  };
  leftSupportCounts: {
    sharedAllValidCount: number;
    sharedNonoccCount: number;
    sharedLowDisp24NonoccCount: number;
    supportSha256: string;
  };
  rightSupportCounts: {
    sharedAllValidCount: number;
    sharedNonoccCount: number;
    sharedLowDisp24NonoccCount: number;
    supportSha256: string;
  };
  cropAWasm: NoPadLadderArmRunSummary;
  cropBWasmReused: NoPadLadderArmRunSummary;
  deltaCenterMinusShiftedContextOnly: {
    leftNonoccEpeDeltaPx: number;
    leftNonoccBad3DeltaPctPts: number;
    rightNonoccEpeDeltaPx: number;
    rightNonoccBad3DeltaPctPts: number;
    meanBothEyeAbsEpeShiftPx: number;
  };
  interpretation: string;
}

export interface BoundedLadderGpuDispatchCheck {
  attempted: boolean;
  selectedLocationId: NoPadLadderLocationId;
  selectedComparisonScaleId: 'scale_0_75' | 'scale_0_50';
  selectionRationale: string;
  realDispatchObservedBothArms: boolean;
  native1xGpu: NoPadLadderArmRunSummary | null;
  scaledCandidateGpu: NoPadLadderArmRunSummary | null;
  rawParityGpuVsWasm: {
    native1x: {
      reference_output_disparity: RawTensorParityStats;
      secondary_output_disparity: RawTensorParityStats;
    };
    scaledCandidate: {
      reference_output_disparity: RawTensorParityStats;
      secondary_output_disparity: RawTensorParityStats;
    };
  } | null;
  failureOrFallbackNote: string | null;
}

export interface HitnetMotorcycleNoPadScaleLadderReport {
  ok: true;
  sceneId: 'Motorcycle';
  edition: string;
  sourceDimensions: { width: number; height: number };
  calib: MiddleburyCalibData;
  officialMatchedDyNote: string;
  pinnedOrtVersion: '1.30.0';
  runtimeAssetFamily: 'asyncify';
  graphOptimizationLevel: 'basic';
  channelOrder: 'C0_RGB';
  m1ByteLength: number;
  m1Sha256: string;
  n1OffProof: N1OffVerificationProof;
  fixtureSha256s: LoadedMiddleburySceneAssets['fixtureSha256s'];
  distinctionFromPriorDiagnostics: {
    priorNativeAndDownscalePlusPadSummary: string;
    currentNoPadScaleLadderSummary: string;
    epistemicRule: string;
  };
  locations: Record<NoPadLadderLocationId, LocationNoPadScaleLadderReport>;
  sameScaleContextOnlyControl: SameScaleShiftedCropContextControlReport;
  boundedGpuDispatchCheck: BoundedLadderGpuDispatchCheck;
  summary: {
    upperLeftTrend: string;
    centerTrend: string;
    shifted1xContextControlDelta: string;
    nextStepOrInterpretation: string;
  };
}

export type NoPadCropFeedBuilder = (params: {
  im0Decoded: DecodedRgbImageGrid;
  im1Decoded: DecodedRgbImageGrid;
  srcRect: FractionalSourceCropRect;
}) => Promise<{
  feedFloat32: Float32Array;
  feedSha256: string;
}>;

export function buildPredeclaredNoPadScaleLadderPlacements(
  sourceWidth = MIDDLEBURY_EVAL3_Q_SCENES.Motorcycle.expectedWidth,
  sourceHeight = MIDDLEBURY_EVAL3_Q_SCENES.Motorcycle.expectedHeight
): Record<NoPadLadderLocationId, Record<NoPadLadderScaleId, NoPadLadderArmPlacement>> {
  const mw = HITNET_MIDDLEBURY_MODEL_WIDTH;
  const mh = HITNET_MIDDLEBURY_MODEL_HEIGHT;

  // Anchored center matches prior Scale/Context center 1:1 window [210..530)x[128..368)
  const centerAnchorX = 210 + mw / 2; // 370
  const centerAnchorY = 128 + mh / 2; // 248

  const makePlacement = (
    locationId: NoPadLadderLocationId,
    scaleId: NoPadLadderScaleId,
    scale: number
  ): NoPadLadderArmPlacement => {
    const srcW = mw / scale;
    const srcH = mh / scale;
    const x = locationId === 'upper_left' ? 0 : centerAnchorX - srcW / 2;
    const y = locationId === 'upper_left' ? 0 : centerAnchorY - srcH / 2;

    if (
      x < -1e-9 ||
      y < -1e-9 ||
      x + srcW > sourceWidth + 1e-9 ||
      y + srcH > sourceHeight + 1e-9
    ) {
      throw new Error(
        `Predeclared no-pad window out of bounds for ${locationId} ${scaleId}: [${x}, ${y}, ${srcW}, ${srcH}] vs ${sourceWidth}x${sourceHeight}`
      );
    }

    const clampedX = Math.max(0, x);
    const clampedY = Math.max(0, y);

    return {
      locationId,
      scaleId,
      scaleX: mw / srcW,
      scaleY: mh / srcH,
      srcRect: { x: clampedX, y: clampedY, width: srcW, height: srcH },
      dstRect: { x: 0, y: 0, width: mw, height: mh },
      placementPolicy:
        locationId === 'upper_left' ? 'top-left-anchored-nested' : 'center-anchored-nested',
      paddingPixelsPerEye: 0,
      contextExpansionNote:
        locationId === 'upper_left'
          ? `Anchored at (0,0); scaling to ${scale.toFixed(2)}x expands physical FOV rightward/downward to [${clampedX.toFixed(2)}..${(clampedX + srcW).toFixed(2)}) x [${clampedY.toFixed(2)}..${(clampedY + srcH).toFixed(2)}). Scale is coupled with added right/bottom scene context (NOT pure scale).`
          : `Centered on (${centerAnchorX},${centerAnchorY}); scaling to ${scale.toFixed(2)}x expands physical FOV symmetrically across [${clampedX.toFixed(2)}..${(clampedX + srcW).toFixed(2)}) x [${clampedY.toFixed(2)}..${(clampedY + srcH).toFixed(2)}). Scale is coupled with added outer scene context (NOT pure scale).`,
    };
  };

  return {
    upper_left: {
      scale_1_00: makePlacement('upper_left', 'scale_1_00', 1.0),
      scale_0_75: makePlacement('upper_left', 'scale_0_75', 0.75),
      scale_0_50: makePlacement('upper_left', 'scale_0_50', 0.5),
    },
    center: {
      scale_1_00: makePlacement('center', 'scale_1_00', 1.0),
      scale_0_75: makePlacement('center', 'scale_0_75', 0.75),
      scale_0_50: makePlacement('center', 'scale_0_50', 0.5),
    },
  };
}

export const PREDECLARED_SHIFTED_CONTEXT_CONTROL_CROP_A = {
  cropId: 'shifted_left_top_1_00' as const,
  srcRect: { x: 120, y: 80, width: 320 as const, height: 240 as const },
};

export function mapSourcePointToModelGridIndex(
  ix: number,
  iy: number,
  srcRect: FractionalSourceCropRect,
  scale: number,
  marginModelPx: number
): number | null {
  const mw = HITNET_MIDDLEBURY_MODEL_WIDTH;
  const mh = HITNET_MIDDLEBURY_MODEL_HEIGHT;
  const mx = Math.round((ix + 0.5 - srcRect.x) * scale - 0.5);
  const my = Math.round((iy + 0.5 - srcRect.y) * scale - 0.5);
  if (
    mx < marginModelPx ||
    mx > mw - 1 - marginModelPx ||
    my < marginModelPx ||
    my > mh - 1 - marginModelPx
  ) {
    return null;
  }
  return my * mw + mx;
}

async function hashPointCoordinates(
  points: ReadonlyArray<{ ix: number; iy: number; gtNativePx: number }>
): Promise<string> {
  const buf = new ArrayBuffer(points.length * 12);
  const dv = new DataView(buf);
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    dv.setInt32(i * 12, p.ix, true);
    dv.setInt32(i * 12 + 4, p.iy, true);
    dv.setFloat32(i * 12 + 8, p.gtNativePx, true);
  }
  return sha256HexBytes(new Uint8Array(buf));
}

export async function buildNoPadLadderSharedPointsForEye(params: {
  eye: MiddleburyEyeId;
  locationId: NoPadLadderLocationId;
  placements: Record<NoPadLadderScaleId, NoPadLadderArmPlacement>;
  gtPfm: ParsedPfmGrid;
  maskNocc: ParsedPngMaskGrid;
  calib: MiddleburyCalibData;
  marginModelPx?: number;
}): Promise<NoPadLadderEyeSharedSupport> {
  const { eye, locationId, placements, gtPfm, maskNocc, calib } = params;
  const marginModelPx = params.marginModelPx ?? MOTORCYCLE_ACTIVE_PAD_GUARD_MODEL_PX;
  const p100 = placements.scale_1_00;
  const p075 = placements.scale_0_75;
  const p050 = placements.scale_0_50;

  // Tightest crop (1.0x) establishes all physical candidates inside all nested placements.
  const xStart = Math.ceil(p100.srcRect.x);
  const xEnd = Math.floor(p100.srcRect.x + p100.srcRect.width) - 1;
  const yStart = Math.ceil(p100.srcRect.y);
  const yEnd = Math.floor(p100.srcRect.y + p100.srcRect.height) - 1;

  // Verify every valid source X stays within ALL three arms' guarded margins
  let activeXMin = Infinity;
  let activeXMax = -Infinity;
  let activeYMin = Infinity;
  let activeYMax = -Infinity;

  const isSourcePointInsideAllArms = (sx: number, sy: number): Record<NoPadLadderScaleId, number> | null => {
    const i100 = mapSourcePointToModelGridIndex(sx, sy, p100.srcRect, p100.scaleX, marginModelPx);
    if (i100 === null) return null;
    const i075 = mapSourcePointToModelGridIndex(sx, sy, p075.srcRect, p075.scaleX, marginModelPx);
    if (i075 === null) return null;
    const i050 = mapSourcePointToModelGridIndex(sx, sy, p050.srcRect, p050.scaleX, marginModelPx);
    if (i050 === null) return null;
    return { scale_1_00: i100, scale_0_75: i075, scale_0_50: i050 };
  };

  for (let iy = yStart; iy <= yEnd; iy++) {
    for (let ix = xStart; ix <= xEnd; ix++) {
      if (isSourcePointInsideAllArms(ix, iy) !== null) {
        if (ix < activeXMin) activeXMin = ix;
        if (ix > activeXMax) activeXMax = ix;
        if (iy < activeYMin) activeYMin = iy;
        if (iy > activeYMax) activeYMax = iy;
      }
    }
  }

  let candidatePhysicalPixelsInIntersection = 0;
  let excludedMarginOrOutOfBoundsCount = 0;
  let excludedInvalidOrNonPositiveGtCount = 0;
  let excludedOverNdispCount = 0;
  let excludedCounterpartOutsideNativeCropOrMarginCount = 0;
  let sharedLowDisp24NonoccCount = 0;
  let sharedLowDispModel12EqNonoccCount = 0;

  const sharedAllValidPoints: NoPadLadderPhysicalPoint[] = [];
  const sharedNonoccPoints: NoPadLadderPhysicalPoint[] = [];

  for (let iy = yStart; iy <= yEnd; iy++) {
    for (let ix = xStart; ix <= xEnd; ix++) {
      candidatePhysicalPixelsInIntersection++;
      const idxByScale = isSourcePointInsideAllArms(ix, iy);
      if (idxByScale === null) {
        excludedMarginOrOutOfBoundsCount++;
        continue;
      }

      const srcIdx = iy * calib.width + ix;
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

      // Require counterpart X in BOTH eyes to remain within the guarded intersection of all 3 arms.
      const xMatch = eye === 'left' ? ix - gtNativePx : ix + gtNativePx;
      const xMatchRound = Math.round(xMatch);
      if (
        xMatch < activeXMin ||
        xMatch > activeXMax ||
        isSourcePointInsideAllArms(xMatchRound, iy) === null
      ) {
        excludedCounterpartOutsideNativeCropOrMarginCount++;
        continue;
      }

      const maskByte = maskNocc.grayTopToBottom[srcIdx];
      const pt: NoPadLadderPhysicalPoint = {
        ix,
        iy,
        gtNativePx,
        maskByte,
        idxByScale,
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

  const supportSha256 = await hashPointCoordinates(sharedNonoccPoints);

  return {
    locationId,
    eye,
    outputTensorName:
      eye === 'left' ? 'reference_output_disparity' : 'secondary_output_disparity',
    marginModelPx,
    activeSrcBoxIntersection: {
      xMin: activeXMin,
      xMax: activeXMax,
      yMin: activeYMin,
      yMax: activeYMax,
    },
    counts: {
      candidatePhysicalPixelsInIntersection,
      excludedMarginOrOutOfBoundsCount,
      excludedInvalidOrNonPositiveGtCount,
      excludedOverNdispCount,
      excludedCounterpartOutsideNativeCropOrMarginCount,
      sharedAllValidCount: sharedAllValidPoints.length,
      sharedNonoccCount: sharedNonoccPoints.length,
      sharedLowDisp24NonoccCount,
      sharedLowDispModel12EqNonoccCount,
    },
    supportSha256,
    sharedAllValidPoints,
    sharedNonoccPoints,
  };
}

export async function buildShiftedCropContextControlSupportForEye(params: {
  eye: MiddleburyEyeId;
  cropA: { x: number; y: number; width: number; height: number };
  cropB: { x: number; y: number; width: number; height: number };
  gtPfm: ParsedPfmGrid;
  maskNocc: ParsedPngMaskGrid;
  calib: MiddleburyCalibData;
  marginModelPx?: number;
}): Promise<{
  intersectionBoxNativeQ: {
    xMin: number;
    xMax: number;
    yMin: number;
    yMax: number;
    marginModelPx: number;
    widthPx: number;
    heightPx: number;
  };
  sharedAllValidPoints: SameScaleShiftedCropPoint[];
  sharedNonoccPoints: SameScaleShiftedCropPoint[];
  lowDisp24NonoccCount: number;
  supportSha256: string;
}> {
  const { eye, cropA, cropB, gtPfm, maskNocc, calib } = params;
  const margin = params.marginModelPx ?? MOTORCYCLE_ACTIVE_PAD_GUARD_MODEL_PX;

  const xMin = Math.max(cropA.x + margin, cropB.x + margin);
  const xMax = Math.min(cropA.x + cropA.width - 1 - margin, cropB.x + cropB.width - 1 - margin);
  const yMin = Math.max(cropA.y + margin, cropB.y + margin);
  const yMax = Math.min(cropA.y + cropA.height - 1 - margin, cropB.y + cropB.height - 1 - margin);

  if (xMin >= xMax || yMin >= yMax) {
    throw new Error('Shifted 1x crops have empty intersection');
  }

  const sharedAllValidPoints: SameScaleShiftedCropPoint[] = [];
  const sharedNonoccPoints: SameScaleShiftedCropPoint[] = [];
  let lowDisp24NonoccCount = 0;
  const mw = HITNET_MIDDLEBURY_MODEL_WIDTH;

  for (let iy = yMin; iy <= yMax; iy++) {
    for (let ix = xMin; ix <= xMax; ix++) {
      const srcIdx = iy * calib.width + ix;
      const gtNativePx = gtPfm.dataTopToBottom[srcIdx];
      if (!Number.isFinite(gtNativePx) || gtNativePx <= 0 || gtNativePx > calib.ndisp) continue;

      const xMatch = eye === 'left' ? ix - gtNativePx : ix + gtNativePx;
      if (xMatch < xMin || xMatch > xMax) continue;

      const maskByte = maskNocc.grayTopToBottom[srcIdx];
      const pt: SameScaleShiftedCropPoint = {
        ix,
        iy,
        gtNativePx,
        maskByte,
        idxShiftedA: (iy - cropA.y) * mw + (ix - cropA.x),
        idxCenterB: (iy - cropB.y) * mw + (ix - cropB.x),
      };
      sharedAllValidPoints.push(pt);
      if (maskByte === 255) {
        sharedNonoccPoints.push(pt);
        if (gtNativePx < MOTORCYCLE_LOW_DISPARITY_NATIVE_Q_MAX_PX) {
          lowDisp24NonoccCount++;
        }
      }
    }
  }

  const supportSha256 = await hashPointCoordinates(sharedNonoccPoints);

  return {
    intersectionBoxNativeQ: {
      xMin,
      xMax,
      yMin,
      yMax,
      marginModelPx: margin,
      widthPx: xMax - xMin + 1,
      heightPx: yMax - yMin + 1,
    },
    sharedAllValidPoints,
    sharedNonoccPoints,
    lowDisp24NonoccCount,
    supportSha256,
  };
}

export function evaluateNativeQPointsFromIndices<T>(params: {
  pred: Float32Array;
  scaleX: number;
  points: ReadonlyArray<T>;
  getPredIdx: (pt: T) => number;
  getGtNativePx: (pt: T) => number;
  label: string;
}): NativeQDisparityMetrics {
  const { pred, scaleX, points, getPredIdx, getGtNativePx, label } = params;
  if (pred.length !== HITNET_MIDDLEBURY_PIXEL_COUNT) {
    throw new Error(`Expected prediction length ${HITNET_MIDDLEBURY_PIXEL_COUNT}, got ${pred.length}`);
  }

  const absErrsQ: number[] = [];
  const signedErrsQ: number[] = [];
  const absErrsM: number[] = [];

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

  for (const pt of points) {
    const pModel = pred[getPredIdx(pt)];
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

    const gtNativeQ = getGtNativePx(pt);
    const pNativeQ = pModel / scaleX;
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
    epeNativeQPx: n === 0 ? NaN : nonFinitePredCount > 0 ? Infinity : absSumQ / finitePredCount,
    signedBiasNativeQPx: finitePredCount > 0 ? signedSumQ / finitePredCount : NaN,
    medianSignedErrorNativeQPx: signedErrsQ.length > 0 ? signedErrsQ[medIdx] : NaN,
    p95AbsErrorNativeQPx: absErrsQ.length > 0 ? absErrsQ[p95Idx] : NaN,
    maxAbsErrorNativeQPx: n > 0 ? maxAbsQ : NaN,
    bad1NativeQRate: n > 0 ? bad1Q / n : NaN,
    bad3NativeQRate: n > 0 ? bad3Q / n : NaN,
    epeModelPx: n === 0 ? NaN : nonFinitePredCount > 0 ? Infinity : absSumM / finitePredCount,
    signedBiasModelPx: finitePredCount > 0 ? signedSumM / finitePredCount : NaN,
    p95AbsErrorModelPx: absErrsM.length > 0 ? absErrsM[p95Idx] : NaN,
    bad1ModelPxRate: n > 0 ? bad1M / n : NaN,
    bad3ModelPxRate: n > 0 ? bad3M / n : NaN,
  };
}

export function evaluateNoPadEyeMetrics<T extends { gtNativePx: number }>(params: {
  pred: Float32Array;
  eye: MiddleburyEyeId;
  scaleX: number;
  allValidPoints: ReadonlyArray<T>;
  nonoccPoints: ReadonlyArray<T>;
  getPredIdx: (pt: T) => number;
}): WindowEyeArmEvaluation {
  const { pred, eye, scaleX, allValidPoints, nonoccPoints, getPredIdx } = params;
  const low24 = nonoccPoints.filter(
    (p) => p.gtNativePx < MOTORCYCLE_LOW_DISPARITY_NATIVE_Q_MAX_PX
  );
  const low12Eq = nonoccPoints.filter(
    (p) => p.gtNativePx < MOTORCYCLE_LOW_DISPARITY_MODEL12_NATIVE_Q_PX
  );
  return {
    eye,
    outputTensorName:
      eye === 'left' ? 'reference_output_disparity' : 'secondary_output_disparity',
    nonocc: evaluateNativeQPointsFromIndices({
      pred,
      scaleX,
      points: nonoccPoints,
      getPredIdx,
      getGtNativePx: (p) => p.gtNativePx,
      label: `${eye}_nonocc`,
    }),
    lowDisp24Nonocc: evaluateNativeQPointsFromIndices({
      pred,
      scaleX,
      points: low24,
      getPredIdx,
      getGtNativePx: (p) => p.gtNativePx,
      label: `${eye}_lowDisp_lt24Q_nonocc`,
    }),
    lowDispModel12EqNonocc: evaluateNativeQPointsFromIndices({
      pred,
      scaleX,
      points: low12Eq,
      getPredIdx,
      getGtNativePx: (p) => p.gtNativePx,
      label: `${eye}_lowDisp_lt12ModelEqQ_nonocc`,
    }),
    allValidGt: evaluateNativeQPointsFromIndices({
      pred,
      scaleX,
      points: allValidPoints,
      getPredIdx,
      getGtNativePx: (p) => p.gtNativePx,
      label: `${eye}_allValidGt`,
    }),
  };
}

function rgbGridToFullRgba(img: DecodedRgbImageGrid): Uint8ClampedArray {
  const total = img.width * img.height;
  const rgba = new Uint8ClampedArray(total * 4);
  for (let i = 0; i < total; i++) {
    rgba[i * 4] = img.rgbTopToBottom[i * 3];
    rgba[i * 4 + 1] = img.rgbTopToBottom[i * 3 + 1];
    rgba[i * 4 + 2] = img.rgbTopToBottom[i * 3 + 2];
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

function drawFullSourceCropToModelBrowser(
  fullRgba: Uint8ClampedArray,
  srcW: number,
  srcH: number,
  srcRect: FractionalSourceCropRect
): Uint8ClampedArray {
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = srcW;
  srcCanvas.height = srcH;
  const srcCtx = srcCanvas.getContext('2d');
  if (!srcCtx) throw new Error('2D canvas unavailable for source buffer');
  const clampedCopy = new Uint8ClampedArray(fullRgba.length);
  clampedCopy.set(fullRgba);
  srcCtx.putImageData(new ImageData(clampedCopy, srcW, srcH), 0, 0);

  const dstCanvas = document.createElement('canvas');
  dstCanvas.width = HITNET_MIDDLEBURY_MODEL_WIDTH;
  dstCanvas.height = HITNET_MIDDLEBURY_MODEL_HEIGHT;
  const dstCtx = dstCanvas.getContext('2d', { willReadFrequently: true });
  if (!dstCtx) throw new Error('2D canvas unavailable for target 320x240 buffer');

  // Direct full-canvas fractional source-crop resize (NO black fillRect/padding anywhere)
  dstCtx.drawImage(
    srcCanvas,
    srcRect.x,
    srcRect.y,
    srcRect.width,
    srcRect.height,
    0,
    0,
    HITNET_MIDDLEBURY_MODEL_WIDTH,
    HITNET_MIDDLEBURY_MODEL_HEIGHT
  );

  return dstCtx.getImageData(
    0,
    0,
    HITNET_MIDDLEBURY_MODEL_WIDTH,
    HITNET_MIDDLEBURY_MODEL_HEIGHT
  ).data;
}

export const defaultBrowserNoPadCropFeedBuilder: NoPadCropFeedBuilder = async ({
  im0Decoded,
  im1Decoded,
  srcRect,
}) => {
  const fullL = rgbGridToFullRgba(im0Decoded);
  const fullR = rgbGridToFullRgba(im1Decoded);
  const dstL = drawFullSourceCropToModelBrowser(
    fullL,
    im0Decoded.width,
    im0Decoded.height,
    srcRect
  );
  const dstR = drawFullSourceCropToModelBrowser(
    fullR,
    im1Decoded.width,
    im1Decoded.height,
    srcRect
  );
  const feedFloat32 = packRgbaStereoEyesToHitnetFeed(dstL, dstR);
  const feedSha256 = await sha256HexBytes(
    new Uint8Array(feedFloat32.buffer, feedFloat32.byteOffset, feedFloat32.byteLength)
  );
  return { feedFloat32, feedSha256 };
};

function computeScaleDeltaVsNative1x(
  scaleId: 'scale_0_75' | 'scale_0_50',
  scale: number,
  candL: WindowEyeArmEvaluation,
  candR: WindowEyeArmEvaluation,
  natL: WindowEyeArmEvaluation,
  natR: WindowEyeArmEvaluation
): NoPadScaleDeltaVsNative1x {
  const lDelta = candL.nonocc.epeNativeQPx - natL.nonocc.epeNativeQPx;
  const rDelta = candR.nonocc.epeNativeQPx - natR.nonocc.epeNativeQPx;
  return {
    scaleId,
    scale,
    leftNonoccEpeNativeQDeltaPx: lDelta,
    leftNonoccSignedBiasNativeQDeltaPx:
      candL.nonocc.signedBiasNativeQPx - natL.nonocc.signedBiasNativeQPx,
    leftNonoccP95NativeQDeltaPx:
      candL.nonocc.p95AbsErrorNativeQPx - natL.nonocc.p95AbsErrorNativeQPx,
    leftNonoccBad1NativeQDeltaPctPts:
      (candL.nonocc.bad1NativeQRate - natL.nonocc.bad1NativeQRate) * 100,
    leftNonoccBad3NativeQDeltaPctPts:
      (candL.nonocc.bad3NativeQRate - natL.nonocc.bad3NativeQRate) * 100,
    leftLowDisp24EpeNativeQDeltaPx:
      candL.lowDisp24Nonocc.epeNativeQPx - natL.lowDisp24Nonocc.epeNativeQPx,
    rightNonoccEpeNativeQDeltaPx: rDelta,
    rightNonoccSignedBiasNativeQDeltaPx:
      candR.nonocc.signedBiasNativeQPx - natR.nonocc.signedBiasNativeQPx,
    rightNonoccP95NativeQDeltaPx:
      candR.nonocc.p95AbsErrorNativeQPx - natR.nonocc.p95AbsErrorNativeQPx,
    rightNonoccBad1NativeQDeltaPctPts:
      (candR.nonocc.bad1NativeQRate - natR.nonocc.bad1NativeQRate) * 100,
    rightNonoccBad3NativeQDeltaPctPts:
      (candR.nonocc.bad3NativeQRate - natR.nonocc.bad3NativeQRate) * 100,
    rightLowDisp24EpeNativeQDeltaPx:
      candR.lowDisp24Nonocc.epeNativeQPx - natR.lowDisp24Nonocc.epeNativeQPx,
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

export async function runHitnetMiddleburyNoPadScaleLadderDiagnostic(params: {
  m1Bytes: Uint8Array;
  isDev?: boolean;
  includeWebGpuCheck?: boolean;
  disableCpuEpFallback?: boolean;
  runWorker?: ResidualDiagnosticWorkerRunner;
  loadMotorcycleAssets?: () => Promise<LoadedMiddleburySceneAssets>;
  buildCropFeed?: NoPadCropFeedBuilder;
  onLog?: (line: string) => void;
}): Promise<HitnetMotorcycleNoPadScaleLadderReport> {
  assertDevFixtureEnabled(params.isDev);
  const log = params.onLog ?? (() => {});
  const runWorker = params.runWorker ?? createBrowserOrt130DiagnosticRunner();
  const loadMotorcycle =
    params.loadMotorcycleAssets ?? (() => loadAndVerifyMiddleburySceneAssets('Motorcycle'));
  const buildCropFeed = params.buildCropFeed ?? defaultBrowserNoPadCropFeedBuilder;
  const includeWebGpuCheck = params.includeWebGpuCheck ?? true;
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

  const placementsByLocation = buildPredeclaredNoPadScaleLadderPlacements(
    assets.calib.width,
    assets.calib.height
  );

  const asyncify = HITNET_ORT_130_FAMILY_PROFILES.asyncify;
  const [pName, sName] = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;
  const officialMatchedDyNote =
    `Motorcycle Q (${assets.spec.edition}) calib.txt reports scalar dyavg=${assets.calib.dyavg} px, dymax=${assets.calib.dymax} px; no per-pixel vertical dy disparity PFM exists in the official matched Q assets (omitted cleanly without adding datasets).`;

  log(
    `=== [DEV-ONLY] Motorcycle Q No-Padding Scale Ladder (1.0x / 0.75x / 0.50x) ===\n` +
      `Runtime: onnxruntime-web@1.30.0 (${asyncify.entryFile} + ${asyncify.wasmFile}) | opt=basic | C0_RGB | M1=${m1Sha} (N1=OFF, 0 replacements)\n` +
      `All arms fill full 320x240 with strictly in-bounds Motorcycle Q pixels (0 padding pixels). Expanding FOV shifts outer scene context alongside scale.`
  );

  const executeModelRun = async (req: {
    runId: string;
    locationId: NoPadLadderLocationId | 'shifted_context_control';
    scaleId: NoPadLadderScaleId | 'shifted_left_top_1_00';
    scaleX: number;
    scaleY: number;
    provider: 'wasm' | 'webgpu';
    feedFloat32: Float32Array;
    feedSha256: string;
    verifyGpuDispatch?: boolean;
  }): Promise<{
    runMeta: NoPadLadderArmRunSummary['run'];
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
      throw new Error(`${req.runId} runtime/provider/opt guard violation: provider=${okRes.provider}`);
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
      runMeta: {
        runId: req.runId,
        locationId: req.locationId,
        scaleId: req.scaleId,
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
        paddedBorderNonZeroCount: 0,
        paddingPixels: 0,
      },
    };
  };

  const locationIds: NoPadLadderLocationId[] = ['upper_left', 'center'];
  const locationReports = {} as Record<NoPadLadderLocationId, LocationNoPadScaleLadderReport>;
  const locationCache = {} as Record<
    NoPadLadderLocationId,
    {
      leftSupport: NoPadLadderEyeSharedSupport;
      rightSupport: NoPadLadderEyeSharedSupport;
      feeds: Record<NoPadLadderScaleId, { feedFloat32: Float32Array; feedSha256: string }>;
      wasmRawOut: Record<NoPadLadderScaleId, { left: Float32Array; right: Float32Array }>;
    }
  >;

  for (const locationId of locationIds) {
    const placements = placementsByLocation[locationId];
    const [leftSupport, rightSupport] = await Promise.all([
      buildNoPadLadderSharedPointsForEye({
        eye: 'left',
        locationId,
        placements,
        gtPfm: assets.disp0GtPfm,
        maskNocc: assets.mask0Nocc,
        calib: assets.calib,
      }),
      buildNoPadLadderSharedPointsForEye({
        eye: 'right',
        locationId,
        placements,
        gtPfm: assets.disp1GtPfm,
        maskNocc: assets.mask1Nocc,
        calib: assets.calib,
      }),
    ]);

    log(
      `\n=== Location: ${locationId} (${placements.scale_1_00.placementPolicy}) ===\n` +
        `  Shared Physical Support (both eyes contained in 1.0x/0.75x/0.50x + 2px model margin): ` +
        `L nonocc=${leftSupport.counts.sharedNonoccCount} (low<24Q=${leftSupport.counts.sharedLowDisp24NonoccCount}, sha=${leftSupport.supportSha256.slice(0, 12)}...) | ` +
        `R nonocc=${rightSupport.counts.sharedNonoccCount} (low<24Q=${rightSupport.counts.sharedLowDisp24NonoccCount}, sha=${rightSupport.supportSha256.slice(0, 12)}...)`
    );

    const feeds = {} as Record<NoPadLadderScaleId, { feedFloat32: Float32Array; feedSha256: string }>;
    const wasmArms = {} as Record<NoPadLadderScaleId, NoPadLadderArmRunSummary>;
    const wasmRawOut = {} as Record<NoPadLadderScaleId, { left: Float32Array; right: Float32Array }>;

    for (const sEntry of NO_PAD_LADDER_SCALES) {
      const placement = placements[sEntry.scaleId];
      const feed = await buildCropFeed({
        im0Decoded,
        im1Decoded,
        srcRect: placement.srcRect,
      });
      feeds[sEntry.scaleId] = feed;

      const exec = await executeModelRun({
        runId: `${locationId}_${sEntry.scaleId}_wasm`,
        locationId,
        scaleId: sEntry.scaleId,
        scaleX: placement.scaleX,
        scaleY: placement.scaleY,
        provider: 'wasm',
        feedFloat32: feed.feedFloat32,
        feedSha256: feed.feedSha256,
      });

      const leftEval = evaluateNoPadEyeMetrics({
        pred: exec.leftOut,
        eye: 'left',
        scaleX: placement.scaleX,
        allValidPoints: leftSupport.sharedAllValidPoints,
        nonoccPoints: leftSupport.sharedNonoccPoints,
        getPredIdx: (p) => p.idxByScale[sEntry.scaleId],
      });
      const rightEval = evaluateNoPadEyeMetrics({
        pred: exec.rightOut,
        eye: 'right',
        scaleX: placement.scaleX,
        allValidPoints: rightSupport.sharedAllValidPoints,
        nonoccPoints: rightSupport.sharedNonoccPoints,
        getPredIdx: (p) => p.idxByScale[sEntry.scaleId],
      });

      wasmArms[sEntry.scaleId] = {
        placement,
        run: exec.runMeta,
        left: leftEval,
        right: rightEval,
      };
      wasmRawOut[sEntry.scaleId] = { left: exec.leftOut, right: exec.rightOut };

      log(
        `  [WASM ${sEntry.label}] crop=[${placement.srcRect.x.toFixed(1)},${placement.srcRect.y.toFixed(1)} ${placement.srcRect.width.toFixed(1)}x${placement.srcRect.height.toFixed(1)}] feed=${feed.feedSha256.slice(0, 12)}... cold=${exec.runMeta.coldRunMs.toFixed(1)}ms warm=${JSON.stringify(exec.runMeta.warmRunMs.map((n) => Number(n.toFixed(1))))}\n` +
          `    L nonocc(n=${leftEval.nonocc.evaluatedCount}): EPE=${leftEval.nonocc.epeNativeQPx.toFixed(3)}Q bias=${leftEval.nonocc.signedBiasNativeQPx.toFixed(3)}Q p95=${leftEval.nonocc.p95AbsErrorNativeQPx.toFixed(2)}Q Bad1=${(leftEval.nonocc.bad1NativeQRate * 100).toFixed(2)}% Bad3=${(leftEval.nonocc.bad3NativeQRate * 100).toFixed(2)}% inv=${leftEval.nonocc.invalidPredCount} | low<24Q(n=${leftEval.lowDisp24Nonocc.evaluatedCount}): EPE=${leftEval.lowDisp24Nonocc.epeNativeQPx.toFixed(3)}Q\n` +
          `    R nonocc(n=${rightEval.nonocc.evaluatedCount}): EPE=${rightEval.nonocc.epeNativeQPx.toFixed(3)}Q bias=${rightEval.nonocc.signedBiasNativeQPx.toFixed(3)}Q p95=${rightEval.nonocc.p95AbsErrorNativeQPx.toFixed(2)}Q Bad1=${(rightEval.nonocc.bad1NativeQRate * 100).toFixed(2)}% Bad3=${(rightEval.nonocc.bad3NativeQRate * 100).toFixed(2)}% inv=${rightEval.nonocc.invalidPredCount} | low<24Q(n=${rightEval.lowDisp24Nonocc.evaluatedCount}): EPE=${rightEval.lowDisp24Nonocc.epeNativeQPx.toFixed(3)}Q`
      );
    }

    const nat = wasmArms.scale_1_00;
    const d075 = computeScaleDeltaVsNative1x(
      'scale_0_75',
      0.75,
      wasmArms.scale_0_75.left,
      wasmArms.scale_0_75.right,
      nat.left,
      nat.right
    );
    const d050 = computeScaleDeltaVsNative1x(
      'scale_0_50',
      0.5,
      wasmArms.scale_0_50.left,
      wasmArms.scale_0_50.right,
      nat.left,
      nat.right
    );

    locationReports[locationId] = {
      locationId,
      placementSummary:
        locationId === 'upper_left'
          ? 'Anchored at (0,0): 1.00x [0..320)x[0..240) ⊂ 0.75x [0..426.67)x[0..320) ⊂ 0.50x [0..640)x[0..480).'
          : 'Centered at (370,248): 1.00x [210..530)x[128..368) ⊂ 0.75x [156.67..583.33)x[88..408) ⊂ 0.50x [50..690)x[8..488).',
      contextChangesExplicitNote: placements.scale_0_50.contextExpansionNote,
      placements,
      leftSupport: {
        ...leftSupport.counts,
        supportSha256: leftSupport.supportSha256,
        activeSrcBoxIntersection: leftSupport.activeSrcBoxIntersection,
      },
      rightSupport: {
        ...rightSupport.counts,
        supportSha256: rightSupport.supportSha256,
        activeSrcBoxIntersection: rightSupport.activeSrcBoxIntersection,
      },
      wasmArms,
      deltasVsScale1_00Wasm: {
        scale_0_75: d075,
        scale_0_50: d050,
      },
    };

    locationCache[locationId] = {
      leftSupport,
      rightSupport,
      feeds,
      wasmRawOut,
    };
  }

  // Same-scale (1.0x) shifted-crop context-only control:
  // Shift Crop A [120..440)x[80..320) vs already-run Center 1.0x Crop B [210..530)x[128..368).
  const cropARect = PREDECLARED_SHIFTED_CONTEXT_CONTROL_CROP_A.srcRect;
  const cropBRect = { x: 210, y: 128, width: 320 as const, height: 240 as const };
  const [shiftLeftSup, shiftRightSup] = await Promise.all([
    buildShiftedCropContextControlSupportForEye({
      eye: 'left',
      cropA: cropARect,
      cropB: cropBRect,
      gtPfm: assets.disp0GtPfm,
      maskNocc: assets.mask0Nocc,
      calib: assets.calib,
    }),
    buildShiftedCropContextControlSupportForEye({
      eye: 'right',
      cropA: cropARect,
      cropB: cropBRect,
      gtPfm: assets.disp1GtPfm,
      maskNocc: assets.mask1Nocc,
      calib: assets.calib,
    }),
  ]);

  const shiftFeed = await buildCropFeed({
    im0Decoded,
    im1Decoded,
    srcRect: cropARect,
  });
  const shiftExec = await executeModelRun({
    runId: 'shifted_left_top_1_00_wasm',
    locationId: 'shifted_context_control',
    scaleId: 'shifted_left_top_1_00',
    scaleX: 1.0,
    scaleY: 1.0,
    provider: 'wasm',
    feedFloat32: shiftFeed.feedFloat32,
    feedSha256: shiftFeed.feedSha256,
  });

  const shiftALeft = evaluateNoPadEyeMetrics({
    pred: shiftExec.leftOut,
    eye: 'left',
    scaleX: 1.0,
    allValidPoints: shiftLeftSup.sharedAllValidPoints,
    nonoccPoints: shiftLeftSup.sharedNonoccPoints,
    getPredIdx: (p) => p.idxShiftedA,
  });
  const shiftARight = evaluateNoPadEyeMetrics({
    pred: shiftExec.rightOut,
    eye: 'right',
    scaleX: 1.0,
    allValidPoints: shiftRightSup.sharedAllValidPoints,
    nonoccPoints: shiftRightSup.sharedNonoccPoints,
    getPredIdx: (p) => p.idxShiftedA,
  });

  const centerBLeftOut = locationCache.center.wasmRawOut.scale_1_00.left;
  const centerBRightOut = locationCache.center.wasmRawOut.scale_1_00.right;
  const shiftBLeft = evaluateNoPadEyeMetrics({
    pred: centerBLeftOut,
    eye: 'left',
    scaleX: 1.0,
    allValidPoints: shiftLeftSup.sharedAllValidPoints,
    nonoccPoints: shiftLeftSup.sharedNonoccPoints,
    getPredIdx: (p) => p.idxCenterB,
  });
  const shiftBRight = evaluateNoPadEyeMetrics({
    pred: centerBRightOut,
    eye: 'right',
    scaleX: 1.0,
    allValidPoints: shiftRightSup.sharedAllValidPoints,
    nonoccPoints: shiftRightSup.sharedNonoccPoints,
    getPredIdx: (p) => p.idxCenterB,
  });

  const shiftLeftDelta = shiftBLeft.nonocc.epeNativeQPx - shiftALeft.nonocc.epeNativeQPx;
  const shiftRightDelta = shiftBRight.nonocc.epeNativeQPx - shiftARight.nonocc.epeNativeQPx;
  const meanShiftAbs = (Math.abs(shiftLeftDelta) + Math.abs(shiftRightDelta)) / 2;

  const sameScaleContextOnlyControl: SameScaleShiftedCropContextControlReport = {
    controlKind: 'same-scale-1x-shifted-crop-context-only',
    scale: 1.0,
    cropA: {
      cropId: 'shifted_left_top_1_00',
      srcRect: cropARect,
    },
    cropB: {
      cropId: 'center_scale_1_00',
      reusedFromCenterLadder: true,
      srcRect: cropBRect,
    },
    intersectionBoxNativeQ: shiftLeftSup.intersectionBoxNativeQ,
    leftSupportCounts: {
      sharedAllValidCount: shiftLeftSup.sharedAllValidPoints.length,
      sharedNonoccCount: shiftLeftSup.sharedNonoccPoints.length,
      sharedLowDisp24NonoccCount: shiftLeftSup.lowDisp24NonoccCount,
      supportSha256: shiftLeftSup.supportSha256,
    },
    rightSupportCounts: {
      sharedAllValidCount: shiftRightSup.sharedAllValidPoints.length,
      sharedNonoccCount: shiftRightSup.sharedNonoccPoints.length,
      sharedLowDisp24NonoccCount: shiftRightSup.lowDisp24NonoccCount,
      supportSha256: shiftRightSup.supportSha256,
    },
    cropAWasm: {
      placement: {
        locationId: 'center',
        scaleId: 'scale_1_00',
        scaleX: 1.0,
        scaleY: 1.0,
        srcRect: cropARect,
        dstRect: { x: 0, y: 0, width: 320, height: 240 },
        placementPolicy: 'center-anchored-nested',
        paddingPixelsPerEye: 0,
        contextExpansionNote:
          'Same-scale 1.0x shifted crop [120..440)x[80..320) evaluated strictly on fixed intersection with center 1.0x.',
      },
      run: shiftExec.runMeta,
      left: shiftALeft,
      right: shiftARight,
    },
    cropBWasmReused: {
      placement: placementsByLocation.center.scale_1_00,
      run: locationReports.center.wasmArms.scale_1_00.run,
      left: shiftBLeft,
      right: shiftBRight,
    },
    deltaCenterMinusShiftedContextOnly: {
      leftNonoccEpeDeltaPx: shiftLeftDelta,
      leftNonoccBad3DeltaPctPts:
        (shiftBLeft.nonocc.bad3NativeQRate - shiftALeft.nonocc.bad3NativeQRate) * 100,
      rightNonoccEpeDeltaPx: shiftRightDelta,
      rightNonoccBad3DeltaPctPts:
        (shiftBRight.nonocc.bad3NativeQRate - shiftARight.nonocc.bad3NativeQRate) * 100,
      meanBothEyeAbsEpeShiftPx: meanShiftAbs,
    },
    interpretation:
      `On the exact same 1.0x scale over the fixed intersection [${shiftLeftSup.intersectionBoxNativeQ.xMin}..${shiftLeftSup.intersectionBoxNativeQ.xMax}]x[${shiftLeftSup.intersectionBoxNativeQ.yMin}..${shiftLeftSup.intersectionBoxNativeQ.yMax}], outer crop shift alone moved EPE by L=${shiftLeftDelta.toFixed(3)}Q, R=${shiftRightDelta.toFixed(3)}Q (mean |shift|=${meanShiftAbs.toFixed(3)}Q), directly bounding same-scale boundary/context sensitivity.`,
  };

  log(
    `\n=== Same-Scale (1.0x) Shifted-Crop Context-Only Control ===\n` +
      `  Intersection [${shiftLeftSup.intersectionBoxNativeQ.xMin}..${shiftLeftSup.intersectionBoxNativeQ.xMax}]x[${shiftLeftSup.intersectionBoxNativeQ.yMin}..${shiftLeftSup.intersectionBoxNativeQ.yMax}] (${shiftLeftSup.intersectionBoxNativeQ.widthPx}x${shiftLeftSup.intersectionBoxNativeQ.heightPx}): ` +
      `Shifted[120,80] EPE(L=${shiftALeft.nonocc.epeNativeQPx.toFixed(3)}Q, R=${shiftARight.nonocc.epeNativeQPx.toFixed(3)}Q) vs Center[210,128] EPE(L=${shiftBLeft.nonocc.epeNativeQPx.toFixed(3)}Q, R=${shiftBRight.nonocc.epeNativeQPx.toFixed(3)}Q)`
  );

  // Bounded small real WebGPU dispatch check (1.0x vs best candidate scale on single selected location)
  let selectedLocationId: NoPadLadderLocationId = 'upper_left';
  let selectedComparisonScaleId: 'scale_0_75' | 'scale_0_50' = 'scale_0_50';
  let bestDelta = Infinity;
  for (const loc of locationIds) {
    for (const sId of ['scale_0_75', 'scale_0_50'] as const) {
      const d = locationReports[loc].deltasVsScale1_00Wasm[sId].meanBothEyeNonoccEpeDeltaPx;
      if (d < bestDelta) {
        bestDelta = d;
        selectedLocationId = loc;
        selectedComparisonScaleId = sId;
      }
    }
  }

  const boundedGpuDispatchCheck: BoundedLadderGpuDispatchCheck = {
    attempted: false,
    selectedLocationId,
    selectedComparisonScaleId,
    selectionRationale: `Selected ${selectedLocationId} (${selectedComparisonScaleId} vs scale_1_00) for bounded paired WebGPU dispatch verification (both-eye WASM EPE delta vs 1.0x: ${bestDelta.toFixed(3)}Q).`,
    realDispatchObservedBothArms: false,
    native1xGpu: null,
    scaledCandidateGpu: null,
    rawParityGpuVsWasm: null,
    failureOrFallbackNote: null,
  };

  if (includeWebGpuCheck) {
    boundedGpuDispatchCheck.attempted = true;
    const selCache = locationCache[selectedLocationId];
    const selPlaceNat = placementsByLocation[selectedLocationId].scale_1_00;
    const selPlaceCand = placementsByLocation[selectedLocationId][selectedComparisonScaleId];

    try {
      const gpuNat = await executeModelRun({
        runId: `${selectedLocationId}_scale_1_00_webgpu`,
        locationId: selectedLocationId,
        scaleId: 'scale_1_00',
        scaleX: selPlaceNat.scaleX,
        scaleY: selPlaceNat.scaleY,
        provider: 'webgpu',
        feedFloat32: selCache.feeds.scale_1_00.feedFloat32,
        feedSha256: selCache.feeds.scale_1_00.feedSha256,
        verifyGpuDispatch: true,
      });
      const gpuCand = await executeModelRun({
        runId: `${selectedLocationId}_${selectedComparisonScaleId}_webgpu`,
        locationId: selectedLocationId,
        scaleId: selectedComparisonScaleId,
        scaleX: selPlaceCand.scaleX,
        scaleY: selPlaceCand.scaleY,
        provider: 'webgpu',
        feedFloat32: selCache.feeds[selectedComparisonScaleId].feedFloat32,
        feedSha256: selCache.feeds[selectedComparisonScaleId].feedSha256,
        verifyGpuDispatch: true,
      });

      const natLeftEval = evaluateNoPadEyeMetrics({
        pred: gpuNat.leftOut,
        eye: 'left',
        scaleX: selPlaceNat.scaleX,
        allValidPoints: selCache.leftSupport.sharedAllValidPoints,
        nonoccPoints: selCache.leftSupport.sharedNonoccPoints,
        getPredIdx: (p) => p.idxByScale.scale_1_00,
      });
      const natRightEval = evaluateNoPadEyeMetrics({
        pred: gpuNat.rightOut,
        eye: 'right',
        scaleX: selPlaceNat.scaleX,
        allValidPoints: selCache.rightSupport.sharedAllValidPoints,
        nonoccPoints: selCache.rightSupport.sharedNonoccPoints,
        getPredIdx: (p) => p.idxByScale.scale_1_00,
      });

      const candLeftEval = evaluateNoPadEyeMetrics({
        pred: gpuCand.leftOut,
        eye: 'left',
        scaleX: selPlaceCand.scaleX,
        allValidPoints: selCache.leftSupport.sharedAllValidPoints,
        nonoccPoints: selCache.leftSupport.sharedNonoccPoints,
        getPredIdx: (p) => p.idxByScale[selectedComparisonScaleId],
      });
      const candRightEval = evaluateNoPadEyeMetrics({
        pred: gpuCand.rightOut,
        eye: 'right',
        scaleX: selPlaceCand.scaleX,
        allValidPoints: selCache.rightSupport.sharedAllValidPoints,
        nonoccPoints: selCache.rightSupport.sharedNonoccPoints,
        getPredIdx: (p) => p.idxByScale[selectedComparisonScaleId],
      });

      boundedGpuDispatchCheck.native1xGpu = {
        placement: selPlaceNat,
        run: gpuNat.runMeta,
        left: natLeftEval,
        right: natRightEval,
      };
      boundedGpuDispatchCheck.scaledCandidateGpu = {
        placement: selPlaceCand,
        run: gpuCand.runMeta,
        left: candLeftEval,
        right: candRightEval,
      };
      boundedGpuDispatchCheck.rawParityGpuVsWasm = {
        native1x: {
          reference_output_disparity: summarizeRawTensorAndParity(
            gpuNat.leftOut,
            selCache.wasmRawOut.scale_1_00.left
          ),
          secondary_output_disparity: summarizeRawTensorAndParity(
            gpuNat.rightOut,
            selCache.wasmRawOut.scale_1_00.right
          ),
        },
        scaledCandidate: {
          reference_output_disparity: summarizeRawTensorAndParity(
            gpuCand.leftOut,
            selCache.wasmRawOut[selectedComparisonScaleId].left
          ),
          secondary_output_disparity: summarizeRawTensorAndParity(
            gpuCand.rightOut,
            selCache.wasmRawOut[selectedComparisonScaleId].right
          ),
        },
      };
      boundedGpuDispatchCheck.realDispatchObservedBothArms =
        gpuNat.runMeta.dispatchObserved === true && gpuCand.runMeta.dispatchObserved === true;

      log(
        `\n[Bounded WebGPU Dispatch Check (${selectedLocationId}: 1.00x vs ${selectedComparisonScaleId})] ` +
          `realDispatchBoth=${String(boundedGpuDispatchCheck.realDispatchObservedBothArms)} | ` +
          `GPU vs WASM maxDiff(1.0x=${boundedGpuDispatchCheck.rawParityGpuVsWasm.native1x.reference_output_disparity.maxAbsDiff.toExponential(2)}, ${selectedComparisonScaleId}=${boundedGpuDispatchCheck.rawParityGpuVsWasm.scaledCandidate.reference_output_disparity.maxAbsDiff.toExponential(2)})`
      );
    } catch (err) {
      boundedGpuDispatchCheck.failureOrFallbackNote =
        err instanceof Error ? err.message : String(err);
      log(
        `\n[Bounded WebGPU Dispatch Check] Unavailable/Failed cleanly: ${boundedGpuDispatchCheck.failureOrFallbackNote}`
      );
    }
  }

  const ulD075 = locationReports.upper_left.deltasVsScale1_00Wasm.scale_0_75;
  const ulD050 = locationReports.upper_left.deltasVsScale1_00Wasm.scale_0_50;
  const ctD075 = locationReports.center.deltasVsScale1_00Wasm.scale_0_75;
  const ctD050 = locationReports.center.deltasVsScale1_00Wasm.scale_0_50;

  const upperLeftTrend =
    `upper_left both-eye EPE delta vs 1.00x: 0.75x=${ulD075.meanBothEyeNonoccEpeDeltaPx.toFixed(3)}Q, 0.50x=${ulD050.meanBothEyeNonoccEpeDeltaPx.toFixed(3)}Q`;
  const centerTrend =
    `center both-eye EPE delta vs 1.00x: 0.75x=${ctD075.meanBothEyeNonoccEpeDeltaPx.toFixed(3)}Q, 0.50x=${ctD050.meanBothEyeNonoccEpeDeltaPx.toFixed(3)}Q`;
  const shifted1xContextControlDelta =
    `same-scale 1.00x shifted-crop EPE delta (center - shifted_left_top): L=${shiftLeftDelta.toFixed(3)}Q, R=${shiftRightDelta.toFixed(3)}Q`;

  const nextStepOrInterpretation =
    'All ladder arms fill 320x240 with zero padding; differences vs 1.00x reflect joint physical scale reduction + expanded outer scene context (bounded by the 1.00x shifted-crop control), distinct from earlier 138x102 isolated window down+pad runs.';

  log(`\n=== No-Padding Scale Ladder Summary ===\n  ${upperLeftTrend}\n  ${centerTrend}\n  ${shifted1xContextControlDelta}`);

  return {
    ok: true,
    sceneId: 'Motorcycle',
    edition: assets.spec.edition,
    sourceDimensions: { width: assets.calib.width, height: assets.calib.height },
    calib: assets.calib,
    officialMatchedDyNote,
    pinnedOrtVersion: '1.30.0',
    runtimeAssetFamily: 'asyncify',
    graphOptimizationLevel: 'basic',
    channelOrder: 'C0_RGB',
    m1ByteLength: params.m1Bytes.byteLength,
    m1Sha256: m1Sha,
    n1OffProof,
    fixtureSha256s: assets.fixtureSha256s,
    distinctionFromPriorDiagnostics: {
      priorNativeAndDownscalePlusPadSummary:
        'Prior Scale/Context (S1_downscale_320_741) and Padding Ablation fixed the exact 320x240 physical FOV and shrank it into a 138x102 box surrounded by 81.7% synthetic padding pixels (black, edge-replicated, or reflected).',
      currentNoPadScaleLadderSummary:
        'This ladder fills all 320x240 model pixels with real in-bounds Motorcycle Q pixels (0 padding pixels) by expanding source crop FOV as scale decreases (320x240 @ 1.00x -> 426.667x320 @ 0.75x -> 640x480 @ 0.50x).',
      epistemicRule:
        'Scale and outer scene context co-vary along this no-padding ladder; therefore NO pure-scale causal attribution is made without factoring out same-scale context shift.',
    },
    locations: locationReports,
    sameScaleContextOnlyControl,
    boundedGpuDispatchCheck,
    summary: {
      upperLeftTrend,
      centerTrend,
      shifted1xContextControlDelta,
      nextStepOrInterpretation,
    },
  };
}
