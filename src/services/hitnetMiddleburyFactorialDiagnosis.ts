import {
  HITNET_MIDDLEBURY_MODEL_HEIGHT,
  HITNET_MIDDLEBURY_MODEL_WIDTH,
  HITNET_MIDDLEBURY_PIXEL_COUNT,
  loadAndVerifyMiddleburySceneAssets,
  MIDDLEBURY_EVAL3_Q_SCENES,
  packRgbaStereoEyesToHitnetFeed,
  type EyeSignSanityCheck,
  type LoadedMiddleburySceneAssets,
  type MiddleburyCalibData,
  type MiddleburyEval3SceneId,
  type MiddleburyEyeId,
  type ParsedPfmGrid,
  type ParsedPngMaskGrid,
} from './hitnetMiddleburyEval3Benchmark';
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

export type FactorialGeometryMode = 'G0_direct_320x240' | 'G1_isotropic_same_scaleX';
export type FactorialChannelOrder = 'C0_RGB' | 'C1_BGR';
export type FactorialArmId =
  | 'G0_C0_direct_RGB'
  | 'G0_C1_direct_BGR'
  | 'G1_C0_isotropic_RGB'
  | 'G1_C1_isotropic_BGR';

export const FACTORIAL_ARM_SPECS: ReadonlyArray<{
  armId: FactorialArmId;
  geometry: FactorialGeometryMode;
  channelOrder: FactorialChannelOrder;
  label: string;
}> = [
  {
    armId: 'G0_C0_direct_RGB',
    geometry: 'G0_direct_320x240',
    channelOrder: 'C0_RGB',
    label: 'G0 Direct 320x240 + RGB (baseline)',
  },
  {
    armId: 'G0_C1_direct_BGR',
    geometry: 'G0_direct_320x240',
    channelOrder: 'C1_BGR',
    label: 'G0 Direct 320x240 + R/B-swap BGR',
  },
  {
    armId: 'G1_C0_isotropic_RGB',
    geometry: 'G1_isotropic_same_scaleX',
    channelOrder: 'C0_RGB',
    label: 'G1 Isotropic same-scaleX + RGB',
  },
  {
    armId: 'G1_C1_isotropic_BGR',
    geometry: 'G1_isotropic_same_scaleX',
    channelOrder: 'C1_BGR',
    label: 'G1 Isotropic same-scaleX + R/B-swap BGR',
  },
];

export interface GeometryTransformSpec {
  geometry: FactorialGeometryMode;
  sceneId: MiddleburyEval3SceneId;
  srcWidth: number;
  srcHeight: number;
  modelWidth: 320;
  modelHeight: 240;
  /** Source rect passed to Canvas drawImage */
  srcRect: { x: number; y: number; width: number; height: number };
  /** Destination rect passed to Canvas drawImage */
  dstRect: { x: number; y: number; width: number; height: number };
  scaleX: number;
  scaleY: number;
  scaleYOverScaleX: number;
  modeKind: 'direct-anisotropic' | 'centered-vertical-letterbox' | 'centered-vertical-crop';
  /** Active full interior model rows (excludes any padded/blended edge row) */
  activeFullModelRowStart: number;
  activeFullModelRowEndInclusive: number;
  excludedBoundaryPadModelRows: number[];
  /** Rounding note documenting fractional vs rounded destination/source rectangles */
  roundingNote: string;
}

export interface SharedPhysicalSamplePoint {
  /** Model X column [0..319] (identical in G0 and G1) */
  mx: number;
  /** Nearest model Y row in G0 [0..239] */
  myG0: number;
  /** Exact interior model Y row in G1 */
  myG1: number;
  /** Flat index in 320x240 G0 output */
  idxG0: number;
  /** Flat index in 320x240 G1 output */
  idxG1: number;
  /** Native source pixel coordinates (no GT interpolation!) */
  ix: number;
  iy: number;
  /** GT native disparity (px) and scaled 320x240 model disparity (px) */
  gtNativePx: number;
  gtModelPx: number;
  /** Mask byte in native mask{0,1}nocc.png: 255=nonocc, 128=occluded */
  maskByte: number;
  /** Discrete 3x3 native neighborhood max |Δgt| in model px (for edge category) */
  localGtJumpModelPx: number;
  /** True if any 3x3 native neighbor has maskByte !== 255 or !isFinite(gt) */
  hasNonNonoccNeighbor3x3: boolean;
}

export interface SharedSceneEyeSampleSet {
  eye: MiddleburyEyeId;
  outputTensorName: 'reference_output_disparity' | 'secondary_output_disparity';
  g0Transform: GeometryTransformSpec;
  g1Transform: GeometryTransformSpec;
  /** Support comparison vs prior single-geometry G0 baseline */
  support: {
    priorG0BaselineAllValidCount: number;
    priorG0BaselineNonoccCount: number;
    sharedAllValidCount: number;
    sharedNonoccCount: number;
    excludedByG1ActiveOrEdgeCount: number;
    excludedInvalidOrNonPosCount: number;
    excludedOverNdispCount: number;
    excludedCounterpartOutOfBoundsCount: number;
    samplingPolicyNote: string;
  };
  sharedAllValidPoints: SharedPhysicalSamplePoint[];
  sharedNonoccPoints: SharedPhysicalSamplePoint[];
}

export interface BandBreakdownEntry {
  bandId: string;
  label: string;
  count: number;
  fractionOfEvaluated: number;
  epeModelPx: number;
  signedBiasModelPx: number;
  p95AbsErrorModelPx: number;
  bad1Rate: number;
  bad3Rate: number;
  invalidPredCount: number;
}

export interface SpatialClusterSpotCheck {
  rank: number;
  tileCol: number;
  tileRow: number;
  modelBox: { xStart: number; xEndExclusive: number; yStart: number; yEndExclusive: number };
  sourceBox: { ixMin: number; ixMax: number; iyMin: number; iyMax: number };
  sampleCount: number;
  epeModelPx: number;
  signedBiasModelPx: number;
  bad3Rate: number;
  gtModelRangePx: [number, number];
  gtNativeRangePx: [number, number];
  allGtWithinNdisp: boolean;
  edgeOrBoundaryFraction: number;
  lowTextureFraction: number;
  yResidualStatus: string;
}

export interface FactorialEyeEvaluation {
  eye: MiddleburyEyeId;
  outputTensorName: 'reference_output_disparity' | 'secondary_output_disparity';
  nonocc: {
    evaluatedCount: number;
    finitePredCount: number;
    nonFinitePredCount: number;
    negativePredCount: number;
    invalidPredCount: number;
    signedBiasModelPx: number;
    medianSignedErrorModelPx: number;
    epeModelPx: number;
    epeNativeQPx: number;
    p95AbsErrorModelPx: number;
    maxAbsErrorModelPx: number;
    bad1Rate: number;
    bad3Rate: number;
    olsSlope: number;
    olsInterceptModelPx: number;
  };
  allValidGt: {
    evaluatedCount: number;
    invalidPredCount: number;
    signedBiasModelPx: number;
    epeModelPx: number;
    p95AbsErrorModelPx: number;
    bad1Rate: number;
    bad3Rate: number;
  };
  disparityBands: BandBreakdownEntry[];
  edgeCategories: BandBreakdownEntry[];
  textureCategories: BandBreakdownEntry[];
  worstClustersTop5: SpatialClusterSpotCheck[];
  eyeSignSanity: EyeSignSanityCheck;
}

export interface FactorialArmRunResult {
  armId: FactorialArmId;
  geometry: FactorialGeometryMode;
  channelOrder: FactorialChannelOrder;
  provider: 'wasm' | 'webgpu';
  graphOptimizationLevel: 'basic';
  modelSha256: string;
  modelByteLength: number;
  feedSha256: string;
  configuredProviders: readonly string[];
  foreignOrtResources: string[];
  gpuValidationErrors: string[];
  dispatchObserved: boolean | null;
  gpuKernelProfileSummary: OrtGpuKernelProfileSummary | null;
  createMs: number;
  coldRunMs: number;
  warmRunMs: number[];
  leftPrimary: FactorialEyeEvaluation;
  rightSecondary: FactorialEyeEvaluation;
}

export interface SceneFactorialReport {
  sceneId: MiddleburyEval3SceneId;
  edition: string;
  calib: MiddleburyCalibData;
  rectificationCaveat: string;
  yMapStatusNote: string;
  transforms: {
    g0: GeometryTransformSpec;
    g1: GeometryTransformSpec;
  };
  leftSupport: SharedSceneEyeSampleSet['support'];
  rightSupport: SharedSceneEyeSampleSet['support'];
  wasmArms: Record<FactorialArmId, FactorialArmRunResult>;
  gpuConfirmation: {
    baselineArm: FactorialArmRunResult;
    candidateArm: FactorialArmRunResult | null;
    baselineRawParityVsWasm: {
      leftFullFrame: RawTensorParityStats;
      rightFullFrame: RawTensorParityStats;
    };
    candidateRawParityVsWasm: {
      leftFullFrame: RawTensorParityStats;
      rightFullFrame: RawTensorParityStats;
    } | null;
  };
  deltaVsG0RgbWasm: Record<
    FactorialArmId,
    {
      leftNonoccEpeDeltaPx: number;
      leftNonoccBad3DeltaPctPts: number;
      rightNonoccEpeDeltaPx: number;
      rightNonoccBad3DeltaPctPts: number;
    }
  >;
}

export interface FactorialDiagnosticVerdict {
  anyMeaningfulImprovement: boolean;
  bestWasmArmByMeanNonoccEpe: Record<MiddleburyEval3SceneId, FactorialArmId>;
  summaryVerdict: string;
  smallestSubsequentFollowUpIfNoImprovement: string[];
}

export interface HitnetMiddleburyFactorialReport {
  ok: true;
  pinnedOrtVersion: '1.30.0';
  runtimeAssetFamily: 'asyncify';
  graphOptimizationLevel: 'basic';
  m1ByteLength: number;
  m1Sha256: string;
  n1OffProof: N1OffVerificationProof;
  samplingAndMaskPolicy: string;
  scenes: Record<MiddleburyEval3SceneId, SceneFactorialReport>;
  verdict: FactorialDiagnosticVerdict;
}

export type FactorialFeedPairBuilder = (params: {
  sceneAssets: LoadedMiddleburySceneAssets;
  g0Transform: GeometryTransformSpec;
  g1Transform: GeometryTransformSpec;
}) => Promise<{
  G0: { leftRgba: Uint8ClampedArray; rightRgba: Uint8ClampedArray };
  G1: { leftRgba: Uint8ClampedArray; rightRgba: Uint8ClampedArray };
}>;

export function buildMiddleburyGeometryTransforms(
  sceneId: MiddleburyEval3SceneId,
  srcWidth: number,
  srcHeight: number
): { g0: GeometryTransformSpec; g1: GeometryTransformSpec } {
  const mw = HITNET_MIDDLEBURY_MODEL_WIDTH;
  const mh = HITNET_MIDDLEBURY_MODEL_HEIGHT;
  const g0ScaleX = mw / srcWidth;
  const g0ScaleY = mh / srcHeight;

  const g0: GeometryTransformSpec = {
    geometry: 'G0_direct_320x240',
    sceneId,
    srcWidth,
    srcHeight,
    modelWidth: mw,
    modelHeight: mh,
    srcRect: { x: 0, y: 0, width: srcWidth, height: srcHeight },
    dstRect: { x: 0, y: 0, width: mw, height: mh },
    scaleX: g0ScaleX,
    scaleY: g0ScaleY,
    scaleYOverScaleX: g0ScaleY / g0ScaleX,
    modeKind: 'direct-anisotropic',
    activeFullModelRowStart: 0,
    activeFullModelRowEndInclusive: mh - 1,
    excludedBoundaryPadModelRows: [],
    roundingNote: `Exact rational scaling scaleX=320/${srcWidth}=${g0ScaleX.toFixed(6)}, scaleY=240/${srcHeight}=${g0ScaleY.toFixed(6)}, Y/X anisotropy=${(g0ScaleY / g0ScaleX).toFixed(6)}.`,
  };

  const isoScale = mw / srcWidth;
  const uncroppedDstH = srcHeight * isoScale;

  if (uncroppedDstH <= mh) {
    // Wider than 4:3 (Motorcycle 741x497 -> 320x214.62888): centered fractional vertical letterbox.
    const dstYExact = (mh - uncroppedDstH) / 2;
    const fullRowStart = Math.ceil(dstYExact);
    const fullRowEndInclusive = Math.floor(dstYExact + uncroppedDstH) - 1;
    const excluded: number[] = [];
    for (let r = 0; r < mh; r++) {
      if (r < fullRowStart || r > fullRowEndInclusive) excluded.push(r);
    }
    return {
      g0,
      g1: {
        geometry: 'G1_isotropic_same_scaleX',
        sceneId,
        srcWidth,
        srcHeight,
        modelWidth: mw,
        modelHeight: mh,
        srcRect: { x: 0, y: 0, width: srcWidth, height: srcHeight },
        dstRect: { x: 0, y: dstYExact, width: mw, height: uncroppedDstH },
        scaleX: isoScale,
        scaleY: isoScale,
        scaleYOverScaleX: 1.0,
        modeKind: 'centered-vertical-letterbox',
        activeFullModelRowStart: fullRowStart,
        activeFullModelRowEndInclusive: fullRowEndInclusive,
        excludedBoundaryPadModelRows: excluded,
        roundingNote:
          `Fractional centered vertical letterbox in Canvas drawImage: dstY=${dstYExact.toFixed(6)}, dstH=${uncroppedDstH.toFixed(6)}, exact scaleX=scaleY=320/${srcWidth}=${isoScale.toFixed(6)} (0.000% integer-height rounding error). ` +
          `Pure interior rows strictly inside [dstY, dstY+dstH] are [${fullRowStart}..${fullRowEndInclusive}] (${fullRowEndInclusive - fullRowStart + 1} rows); boundary/pad rows ${JSON.stringify(excluded)} excluded pre-prediction.`,
      },
    };
  }

  // Taller than 4:3 (Teddy 450x375 -> source crop 450x337.5 -> 320x240): centered fractional vertical crop.
  const cropSrcH = mh / isoScale;
  const cropSrcY = (srcHeight - cropSrcH) / 2;
  return {
    g0,
    g1: {
      geometry: 'G1_isotropic_same_scaleX',
      sceneId,
      srcWidth,
      srcHeight,
      modelWidth: mw,
      modelHeight: mh,
      srcRect: { x: 0, y: cropSrcY, width: srcWidth, height: cropSrcH },
      dstRect: { x: 0, y: 0, width: mw, height: mh },
      scaleX: isoScale,
      scaleY: isoScale,
      scaleYOverScaleX: 1.0,
      modeKind: 'centered-vertical-crop',
      activeFullModelRowStart: 0,
      activeFullModelRowEndInclusive: mh - 1,
      excludedBoundaryPadModelRows: [],
      roundingNote:
        `Fractional centered vertical crop in Canvas drawImage: srcY=${cropSrcY.toFixed(6)}, srcH=${cropSrcH.toFixed(6)}, exact scaleX=scaleY=320/${srcWidth}=${isoScale.toFixed(6)} (0.000% rounding error; top/bottom ${cropSrcY.toFixed(2)} source rows cropped out pre-prediction).`,
    },
  };
}

export function packRgbaStereoEyesWithChannelOrder(
  leftRgba: Uint8ClampedArray,
  rightRgba: Uint8ClampedArray,
  channelOrder: FactorialChannelOrder
): Float32Array {
  if (channelOrder === 'C0_RGB') {
    return packRgbaStereoEyesToHitnetFeed(leftRgba, rightRgba);
  }
  const expectedLen = HITNET_MIDDLEBURY_PIXEL_COUNT * 4;
  if (leftRgba.length !== expectedLen || rightRgba.length !== expectedLen) {
    throw new Error(
      `Expected 320x240 RGBA buffers (${expectedLen} bytes), got left=${leftRgba.length}, right=${rightRgba.length}`
    );
  }
  // C1_BGR: swap ONLY R(0) and B(2) in BOTH eyes; keep Left in ch0..2 and Right in ch3..5.
  const feed = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT * 6);
  const inv255 = Math.fround(1 / 255);
  const px = HITNET_MIDDLEBURY_PIXEL_COUNT;
  for (let i = 0; i < px; i++) {
    const off = i * 4;
    feed[i] = Math.fround(leftRgba[off + 2] * inv255);
    feed[px + i] = Math.fround(leftRgba[off + 1] * inv255);
    feed[px * 2 + i] = Math.fround(leftRgba[off] * inv255);
    feed[px * 3 + i] = Math.fround(rightRgba[off + 2] * inv255);
    feed[px * 4 + i] = Math.fround(rightRgba[off + 1] * inv255);
    feed[px * 5 + i] = Math.fround(rightRgba[off] * inv255);
  }
  return feed;
}

function mapSourceYToModelRowNearest(iy: number, t: GeometryTransformSpec): number | null {
  const myCont =
    t.dstRect.y + ((iy + 0.5 - t.srcRect.y) * t.dstRect.height) / t.srcRect.height - 0.5;
  const my = Math.round(myCont);
  if (my < t.activeFullModelRowStart || my > t.activeFullModelRowEndInclusive) {
    return null;
  }
  return my;
}

function mapModelColToNearestSourceX(mx: number, t: GeometryTransformSpec): number {
  const u = t.srcRect.x + ((mx - t.dstRect.x + 0.5) * t.srcRect.width) / t.dstRect.width - 0.5;
  return Math.max(0, Math.min(t.srcWidth - 1, Math.round(u)));
}

function mapModelRowToNearestSourceY(my: number, t: GeometryTransformSpec): number {
  const v = t.srcRect.y + ((my - t.dstRect.y + 0.5) * t.srcRect.height) / t.dstRect.height - 0.5;
  return Math.max(0, Math.min(t.srcHeight - 1, Math.round(v)));
}

export function buildSharedPhysicalSampleSetForEye(params: {
  eye: MiddleburyEyeId;
  sceneId: MiddleburyEval3SceneId;
  gtPfm: ParsedPfmGrid;
  maskNocc: ParsedPngMaskGrid;
  calib: MiddleburyCalibData;
  g0: GeometryTransformSpec;
  g1: GeometryTransformSpec;
}): SharedSceneEyeSampleSet {
  const { eye, gtPfm, maskNocc, calib, g0, g1 } = params;
  const mw = HITNET_MIDDLEBURY_MODEL_WIDTH;
  const mh = HITNET_MIDDLEBURY_MODEL_HEIGHT;
  const scaleX = g1.scaleX; // identical across G0 and G1

  // Compute prior baseline G0 counts directly for support comparison reporting
  let priorG0BaselineAllValidCount = 0;
  let priorG0BaselineNonoccCount = 0;
  for (let my0 = 0; my0 < mh; my0++) {
    const iy0 = mapModelRowToNearestSourceY(my0, g0);
    for (let mx = 0; mx < mw; mx++) {
      const ix0 = mapModelColToNearestSourceX(mx, g0);
      const srcIdx0 = iy0 * calib.width + ix0;
      const gt0 = gtPfm.dataTopToBottom[srcIdx0];
      if (!Number.isFinite(gt0) || gt0 <= 0 || gt0 > calib.ndisp) continue;
      const xMatch0 = eye === 'left' ? ix0 - gt0 : ix0 + gt0;
      if (xMatch0 < 0 || xMatch0 > calib.width - 1) continue;
      priorG0BaselineAllValidCount++;
      if (maskNocc.grayTopToBottom[srcIdx0] === 255) {
        priorG0BaselineNonoccCount++;
      }
    }
  }

  // Canonical common physical source grid: generated from G1's strict active interior rows x G1/G0 shared columns,
  // verified unique in (ix, iy) and mapped onto exact interior G0 & G1 model indices.
  const sharedAllValidPoints: SharedPhysicalSamplePoint[] = [];
  const sharedNonoccPoints: SharedPhysicalSamplePoint[] = [];
  const seenSourceIndices = new Set<number>();

  let excludedByG1ActiveOrEdgeCount = 0;
  let excludedInvalidOrNonPosCount = 0;
  let excludedOverNdispCount = 0;
  let excludedCounterpartOutOfBoundsCount = 0;

  for (let my1 = 0; my1 < mh; my1++) {
    if (my1 < g1.activeFullModelRowStart || my1 > g1.activeFullModelRowEndInclusive) {
      excludedByG1ActiveOrEdgeCount += mw;
      continue;
    }
    const iy = mapModelRowToNearestSourceY(my1, g1);
    const myG0 = mapSourceYToModelRowNearest(iy, g0);
    if (myG0 === null) {
      excludedByG1ActiveOrEdgeCount += mw;
      continue;
    }

    for (let mx = 0; mx < mw; mx++) {
      const ix = mapModelColToNearestSourceX(mx, g1);
      const srcIdx = iy * calib.width + ix;
      if (seenSourceIndices.has(srcIdx)) continue;
      seenSourceIndices.add(srcIdx);

      const gtNative = gtPfm.dataTopToBottom[srcIdx];
      if (!Number.isFinite(gtNative) || gtNative <= 0) {
        excludedInvalidOrNonPosCount++;
        continue;
      }
      if (gtNative > calib.ndisp) {
        excludedOverNdispCount++;
        continue;
      }

      // Check counterpart lies strictly inside both images (no out-of-bounds correspondence)
      const xMatch = eye === 'left' ? ix - gtNative : ix + gtNative;
      if (xMatch < 0 || xMatch > calib.width - 1) {
        excludedCounterpartOutOfBoundsCount++;
        continue;
      }

      const maskByte = maskNocc.grayTopToBottom[srcIdx];
      let localJumpModelPx = 0;
      let hasNonNonoccNeighbor3x3 = false;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = iy + dy;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = ix + dx;
          if (ny < 0 || ny >= calib.height || nx < 0 || nx >= calib.width) {
            hasNonNonoccNeighbor3x3 = true;
            continue;
          }
          const nIdx = ny * calib.width + nx;
          if (maskNocc.grayTopToBottom[nIdx] !== 255) {
            hasNonNonoccNeighbor3x3 = true;
          }
          const ngt = gtPfm.dataTopToBottom[nIdx];
          if (!Number.isFinite(ngt) || ngt <= 0) {
            hasNonNonoccNeighbor3x3 = true;
          } else {
            const diffModel = Math.abs(ngt - gtNative) * scaleX;
            if (diffModel > localJumpModelPx) localJumpModelPx = diffModel;
          }
        }
      }

      const pt: SharedPhysicalSamplePoint = {
        mx,
        myG0,
        myG1: my1,
        idxG0: myG0 * mw + mx,
        idxG1: my1 * mw + mx,
        ix,
        iy,
        gtNativePx: gtNative,
        gtModelPx: gtNative * scaleX,
        maskByte,
        localGtJumpModelPx: localJumpModelPx,
        hasNonNonoccNeighbor3x3,
      };
      sharedAllValidPoints.push(pt);
      if (maskByte === 255) {
        sharedNonoccPoints.push(pt);
      }
    }
  }

  return {
    eye,
    outputTensorName:
      eye === 'left' ? 'reference_output_disparity' : 'secondary_output_disparity',
    g0Transform: g0,
    g1Transform: g1,
    support: {
      priorG0BaselineAllValidCount,
      priorG0BaselineNonoccCount,
      sharedAllValidCount: sharedAllValidPoints.length,
      sharedNonoccCount: sharedNonoccPoints.length,
      excludedByG1ActiveOrEdgeCount,
      excludedInvalidOrNonPosCount,
      excludedOverNdispCount,
      excludedCounterpartOutOfBoundsCount,
      samplingPolicyNote:
        'Common physical source points (ix, iy) derived 1-to-1 from G1 interior active rows x 320 cols; GT and mask255 are read directly from discrete native (ix, iy) without smoothing; G1 prediction read at exact (mx, my1) and G0 prediction read at nearest interior row (mx, myG0) without cross-edge prediction blending.',
    },
    sharedAllValidPoints,
    sharedNonoccPoints,
  };
}

function computeLocalRgbGradientMag(
  feedFloat32: Float32Array,
  eye: MiddleburyEyeId,
  mx: number,
  my: number
): number {
  const chOffset = eye === 'left' ? 0 : 3;
  const w = HITNET_MIDDLEBURY_MODEL_WIDTH;
  const h = HITNET_MIDDLEBURY_MODEL_HEIGHT;
  const pxCount = HITNET_MIDDLEBURY_PIXEL_COUNT;
  const xL = Math.max(0, mx - 1);
  const xR = Math.min(w - 1, mx + 1);
  const yU = Math.max(0, my - 1);
  const yD = Math.min(h - 1, my + 1);

  let sumSq = 0;
  for (let c = 0; c < 3; c++) {
    const base = (chOffset + c) * pxCount;
    const gx = Math.abs(feedFloat32[base + my * w + xR] - feedFloat32[base + my * w + xL]);
    const gy = Math.abs(feedFloat32[base + yD * w + mx] - feedFloat32[base + yU * w + mx]);
    sumSq += gx * gx + gy * gy;
  }
  return Math.sqrt(sumSq / 3);
}

interface AggregatedEvalBucket {
  bandId: string;
  label: string;
  absErrors: number[];
  signedSum: number;
  absSum: number;
  finiteCount: number;
  invalidPredCount: number;
  bad1Count: number;
  bad3Count: number;
  totalCount: number;
}

function createBucket(bandId: string, label: string): AggregatedEvalBucket {
  return {
    bandId,
    label,
    absErrors: [],
    signedSum: 0,
    absSum: 0,
    finiteCount: 0,
    invalidPredCount: 0,
    bad1Count: 0,
    bad3Count: 0,
    totalCount: 0,
  };
}

function pushBucketSample(
  bucket: AggregatedEvalBucket,
  pred: number,
  gtModel: number
): void {
  bucket.totalCount++;
  if (!Number.isFinite(pred)) {
    bucket.invalidPredCount++;
    bucket.bad1Count++;
    bucket.bad3Count++;
    bucket.absErrors.push(Infinity);
    return;
  }
  if (pred < 0) bucket.invalidPredCount++;
  bucket.finiteCount++;
  const signed = pred - gtModel;
  const absErr = Math.abs(signed);
  bucket.signedSum += signed;
  bucket.absSum += absErr;
  bucket.absErrors.push(absErr);
  if (absErr > 1.0) bucket.bad1Count++;
  if (absErr > 3.0) bucket.bad3Count++;
}

function finalizeBucket(b: AggregatedEvalBucket, totalEvaluated: number): BandBreakdownEntry {
  const sorted = [...b.absErrors].sort((a, c) => a - c);
  const p95Idx =
    sorted.length > 0 ? Math.min(sorted.length - 1, Math.floor(0.95 * (sorted.length - 1))) : 0;
  const hasNonFinite = b.finiteCount < b.totalCount;
  return {
    bandId: b.bandId,
    label: b.label,
    count: b.totalCount,
    fractionOfEvaluated: totalEvaluated > 0 ? b.totalCount / totalEvaluated : 0,
    epeModelPx:
      b.totalCount === 0 ? NaN : hasNonFinite ? Infinity : b.absSum / b.finiteCount,
    signedBiasModelPx: b.finiteCount > 0 ? b.signedSum / b.finiteCount : NaN,
    p95AbsErrorModelPx: sorted.length > 0 ? sorted[p95Idx] : NaN,
    bad1Rate: b.totalCount > 0 ? b.bad1Count / b.totalCount : 0,
    bad3Rate: b.totalCount > 0 ? b.bad3Count / b.totalCount : 0,
    invalidPredCount: b.invalidPredCount,
  };
}

export function evaluateFactorialArmEye(params: {
  pred: Float32Array;
  geometry: FactorialGeometryMode;
  sampleSet: SharedSceneEyeSampleSet;
  oppositeSampleSet: SharedSceneEyeSampleSet;
  feedFloat32: Float32Array;
  calib: MiddleburyCalibData;
  rectificationCaveat: string;
}): FactorialEyeEvaluation {
  const {
    pred,
    geometry,
    sampleSet,
    oppositeSampleSet,
    feedFloat32,
    calib,
    rectificationCaveat,
  } = params;
  const isG0 = geometry === 'G0_direct_320x240';
  const getIdx = (pt: SharedPhysicalSamplePoint) => (isG0 ? pt.idxG0 : pt.idxG1);
  const getMy = (pt: SharedPhysicalSamplePoint) => (isG0 ? pt.myG0 : pt.myG1);

  const summarizeMask = (points: SharedPhysicalSamplePoint[]) => {
    const absErrors: number[] = [];
    const signedErrors: number[] = [];
    let finitePredCount = 0;
    let nonFinitePredCount = 0;
    let negativePredCount = 0;
    let bad1Count = 0;
    let bad3Count = 0;
    let absSum = 0;
    let signedSum = 0;
    let maxAbsErr = 0;
    let sumGt = 0;
    let sumPred = 0;
    let sumGtSq = 0;
    let sumGtPred = 0;

    for (const pt of points) {
      const p = pred[getIdx(pt)];
      const gt = pt.gtModelPx;
      if (!Number.isFinite(p)) {
        nonFinitePredCount++;
        bad1Count++;
        bad3Count++;
        absErrors.push(Infinity);
        maxAbsErr = Infinity;
        continue;
      }
      finitePredCount++;
      if (p < 0) negativePredCount++;
      const signed = p - gt;
      const absErr = Math.abs(signed);
      absErrors.push(absErr);
      signedErrors.push(signed);
      absSum += absErr;
      signedSum += signed;
      if (absErr > maxAbsErr) maxAbsErr = absErr;
      if (absErr > 1.0) bad1Count++;
      if (absErr > 3.0) bad3Count++;
      sumGt += gt;
      sumPred += p;
      sumGtSq += gt * gt;
      sumGtPred += gt * p;
    }

    absErrors.sort((a, b) => a - b);
    signedErrors.sort((a, b) => a - b);
    const p95Idx =
      absErrors.length > 0
        ? Math.min(absErrors.length - 1, Math.floor(0.95 * (absErrors.length - 1)))
        : 0;
    const medIdx = signedErrors.length > 0 ? Math.floor(signedErrors.length / 2) : 0;
    const n = points.length;
    const epeModelPx =
      nonFinitePredCount > 0 ? Infinity : finitePredCount > 0 ? absSum / finitePredCount : NaN;
    const meanGt = finitePredCount > 0 ? sumGt / finitePredCount : NaN;
    const meanPred = finitePredCount > 0 ? sumPred / finitePredCount : NaN;
    const varGt = finitePredCount > 0 ? sumGtSq - finitePredCount * meanGt * meanGt : 0;
    const olsSlope =
      finitePredCount > 1 && varGt > 1e-9
        ? (sumGtPred - finitePredCount * meanGt * meanPred) / varGt
        : NaN;
    const olsInterceptModelPx =
      Number.isFinite(olsSlope) ? meanPred - olsSlope * meanGt : NaN;

    return {
      evaluatedCount: n,
      finitePredCount,
      nonFinitePredCount,
      negativePredCount,
      invalidPredCount: nonFinitePredCount + negativePredCount,
      signedBiasModelPx: finitePredCount > 0 ? signedSum / finitePredCount : NaN,
      medianSignedErrorModelPx: signedErrors.length > 0 ? signedErrors[medIdx] : NaN,
      epeModelPx,
      epeNativeQPx: epeModelPx / sampleSet.g1Transform.scaleX,
      p95AbsErrorModelPx: absErrors.length > 0 ? absErrors[p95Idx] : NaN,
      maxAbsErrorModelPx: absErrors.length > 0 ? maxAbsErr : NaN,
      bad1Rate: n > 0 ? bad1Count / n : 0,
      bad3Rate: n > 0 ? bad3Count / n : 0,
      olsSlope,
      olsInterceptModelPx,
    };
  };

  const nonoccSummary = summarizeMask(sampleSet.sharedNonoccPoints);
  const allValidSummary = summarizeMask(sampleSet.sharedAllValidPoints);

  // Stratified breakdowns on fixed shared nonocc points
  const dispLow = createBucket('disp_0_12', 'Low Disparity [0, 12) model-px');
  const dispMid = createBucket('disp_12_20', 'Mid Disparity [12, 20) model-px');
  const dispHigh = createBucket('disp_20_plus', 'High Disparity [20, +inf) model-px');

  const edgeSmooth = createBucket('smooth', 'Smooth (3x3 max|Δgt|<=0.5px, interior nonocc)');
  const edgeSlant = createBucket('slant', 'Moderate Slant (0.5<max|Δgt|<=2.0px, interior nonocc)');
  const edgeBoundary = createBucket('boundary', 'Depth Edge / Occlusion-Adj (max|Δgt|>2.0px or non-nonocc neighbor)');

  const texLow = createBucket('tex_low', 'Low Texture (RGB grad < 0.02)');
  const texMed = createBucket('tex_medium', 'Medium Texture (0.02 <= grad < 0.08)');
  const texHigh = createBucket('tex_high', 'High Texture (grad >= 0.08)');

  // 8x6 spatial tile cluster stats (40x40 model-px tiles)
  const tileBuckets = new Map<
    string,
    {
      tileCol: number;
      tileRow: number;
      ixMin: number;
      ixMax: number;
      iyMin: number;
      iyMax: number;
      gtModelMin: number;
      gtModelMax: number;
      gtNativeMin: number;
      gtNativeMax: number;
      allWithinNdisp: boolean;
      edgeCount: number;
      lowTexCount: number;
      bucket: AggregatedEvalBucket;
    }
  >();

  for (const pt of sampleSet.sharedNonoccPoints) {
    const p = pred[getIdx(pt)];
    const my = getMy(pt);
    if (pt.gtModelPx < 12) pushBucketSample(dispLow, p, pt.gtModelPx);
    else if (pt.gtModelPx < 20) pushBucketSample(dispMid, p, pt.gtModelPx);
    else pushBucketSample(dispHigh, p, pt.gtModelPx);

    const isEdge = pt.hasNonNonoccNeighbor3x3 || pt.localGtJumpModelPx > 2.0;
    if (isEdge) pushBucketSample(edgeBoundary, p, pt.gtModelPx);
    else if (pt.localGtJumpModelPx > 0.5) pushBucketSample(edgeSlant, p, pt.gtModelPx);
    else pushBucketSample(edgeSmooth, p, pt.gtModelPx);

    const grad = computeLocalRgbGradientMag(feedFloat32, sampleSet.eye, pt.mx, my);
    const isLowTex = grad < 0.02;
    if (isLowTex) pushBucketSample(texLow, p, pt.gtModelPx);
    else if (grad < 0.08) pushBucketSample(texMed, p, pt.gtModelPx);
    else pushBucketSample(texHigh, p, pt.gtModelPx);

    const tileCol = Math.floor(pt.mx / 40);
    const tileRow = Math.floor(my / 40);
    const key = `${tileCol}:${tileRow}`;
    let tb = tileBuckets.get(key);
    if (!tb) {
      tb = {
        tileCol,
        tileRow,
        ixMin: pt.ix,
        ixMax: pt.ix,
        iyMin: pt.iy,
        iyMax: pt.iy,
        gtModelMin: pt.gtModelPx,
        gtModelMax: pt.gtModelPx,
        gtNativeMin: pt.gtNativePx,
        gtNativeMax: pt.gtNativePx,
        allWithinNdisp: true,
        edgeCount: 0,
        lowTexCount: 0,
        bucket: createBucket(key, key),
      };
      tileBuckets.set(key, tb);
    }
    if (pt.ix < tb.ixMin) tb.ixMin = pt.ix;
    if (pt.ix > tb.ixMax) tb.ixMax = pt.ix;
    if (pt.iy < tb.iyMin) tb.iyMin = pt.iy;
    if (pt.iy > tb.iyMax) tb.iyMax = pt.iy;
    if (pt.gtModelPx < tb.gtModelMin) tb.gtModelMin = pt.gtModelPx;
    if (pt.gtModelPx > tb.gtModelMax) tb.gtModelMax = pt.gtModelPx;
    if (pt.gtNativePx < tb.gtNativeMin) tb.gtNativeMin = pt.gtNativePx;
    if (pt.gtNativePx > tb.gtNativeMax) tb.gtNativeMax = pt.gtNativePx;
    if (pt.gtNativePx <= 0 || pt.gtNativePx > calib.ndisp) tb.allWithinNdisp = false;
    if (isEdge) tb.edgeCount++;
    if (isLowTex) tb.lowTexCount++;
    pushBucketSample(tb.bucket, p, pt.gtModelPx);
  }

  const totalN = sampleSet.sharedNonoccPoints.length;
  const yResidualStatus =
    calib.dymax > 0
      ? `Scene-level calib: dyavg=${calib.dyavg}, dymax=${calib.dymax} px (${rectificationCaveat}); per-pixel disp*GTy.pfm absent in static repo fixtures.`
      : `Scene-level calib: dyavg=0, dymax=0 px (rectified).`;

  const worstClustersTop5: SpatialClusterSpotCheck[] = Array.from(tileBuckets.values())
    .filter((tb) => tb.bucket.totalCount >= 50)
    .map((tb) => {
      const fin = finalizeBucket(tb.bucket, totalN);
      return {
        rank: 0,
        tileCol: tb.tileCol,
        tileRow: tb.tileRow,
        modelBox: {
          xStart: tb.tileCol * 40,
          xEndExclusive: Math.min(320, (tb.tileCol + 1) * 40),
          yStart: tb.tileRow * 40,
          yEndExclusive: Math.min(240, (tb.tileRow + 1) * 40),
        },
        sourceBox: {
          ixMin: tb.ixMin,
          ixMax: tb.ixMax,
          iyMin: tb.iyMin,
          iyMax: tb.iyMax,
        },
        sampleCount: tb.bucket.totalCount,
        epeModelPx: fin.epeModelPx,
        signedBiasModelPx: fin.signedBiasModelPx,
        bad3Rate: fin.bad3Rate,
        gtModelRangePx: [tb.gtModelMin, tb.gtModelMax] as [number, number],
        gtNativeRangePx: [tb.gtNativeMin, tb.gtNativeMax] as [number, number],
        allGtWithinNdisp: tb.allWithinNdisp,
        edgeOrBoundaryFraction: tb.edgeCount / tb.bucket.totalCount,
        lowTextureFraction: tb.lowTexCount / tb.bucket.totalCount,
        yResidualStatus,
      };
    })
    .sort((a, b) => b.epeModelPx - a.epeModelPx)
    .slice(0, 5)
    .map((c, idx) => ({ ...c, rank: idx + 1 }));

  // Positive sign sanity check on shared nonocc points
  let corrSum = 0;
  let negSum = 0;
  let corrCnt = 0;
  let swapSum = 0;
  let swapCnt = 0;
  const oppMapBySrc = new Map<number, number>();
  for (const op of oppositeSampleSet.sharedNonoccPoints) {
    oppMapBySrc.set(op.iy * calib.width + op.ix, op.gtModelPx);
  }
  for (const pt of sampleSet.sharedNonoccPoints) {
    const p = pred[getIdx(pt)];
    if (!Number.isFinite(p)) continue;
    corrSum += Math.abs(p - pt.gtModelPx);
    negSum += Math.abs(p - -pt.gtModelPx);
    corrCnt++;
    const oppGt = oppMapBySrc.get(pt.iy * calib.width + pt.ix);
    if (oppGt !== undefined) {
      swapSum += Math.abs(p - oppGt);
      swapCnt++;
    }
  }
  const eyeSignSanity: EyeSignSanityCheck =
    corrCnt > 0
      ? {
          correctEyePosSignEpeModelPx: corrSum / corrCnt,
          negatedGtEpeModelPx: negSum / corrCnt,
          swappedEyePosSignEpeModelPx: swapCnt > 0 ? swapSum / swapCnt : NaN,
          sharedSwapCheckCount: swapCnt,
          positiveSignConfirmed: corrSum / corrCnt < negSum / corrCnt,
        }
      : {
          correctEyePosSignEpeModelPx: NaN,
          negatedGtEpeModelPx: NaN,
          swappedEyePosSignEpeModelPx: NaN,
          sharedSwapCheckCount: 0,
          positiveSignConfirmed: false,
        };

  return {
    eye: sampleSet.eye,
    outputTensorName: sampleSet.outputTensorName,
    nonocc: nonoccSummary,
    allValidGt: allValidSummary,
    disparityBands: [
      finalizeBucket(dispLow, totalN),
      finalizeBucket(dispMid, totalN),
      finalizeBucket(dispHigh, totalN),
    ],
    edgeCategories: [
      finalizeBucket(edgeSmooth, totalN),
      finalizeBucket(edgeSlant, totalN),
      finalizeBucket(edgeBoundary, totalN),
    ],
    textureCategories: [
      finalizeBucket(texLow, totalN),
      finalizeBucket(texMed, totalN),
      finalizeBucket(texHigh, totalN),
    ],
    worstClustersTop5,
    eyeSignSanity,
  };
}

async function decodePngToGeometryRgbaBrowser(
  pngBytes: Uint8Array,
  transform: GeometryTransformSpec
): Promise<Uint8ClampedArray> {
  const copy = new Uint8Array(pngBytes.byteLength);
  copy.set(pngBytes);
  const blob = new Blob([copy], { type: 'image/png' });
  const canvas = document.createElement('canvas');
  canvas.width = HITNET_MIDDLEBURY_MODEL_WIDTH;
  canvas.height = HITNET_MIDDLEBURY_MODEL_HEIGHT;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new Error('2D canvas context unavailable for Middlebury factorial preprocessing');
  }
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, HITNET_MIDDLEBURY_MODEL_WIDTH, HITNET_MIDDLEBURY_MODEL_HEIGHT);

  const drawSource = (img: ImageBitmap | HTMLImageElement, w: number, h: number) => {
    if (w !== transform.srcWidth || h !== transform.srcHeight) {
      throw new Error(
        `Decoded PNG size ${w}x${h} mismatches expected ${transform.srcWidth}x${transform.srcHeight}`
      );
    }
    ctx.drawImage(
      img,
      transform.srcRect.x,
      transform.srcRect.y,
      transform.srcRect.width,
      transform.srcRect.height,
      transform.dstRect.x,
      transform.dstRect.y,
      transform.dstRect.width,
      transform.dstRect.height
    );
    return ctx.getImageData(0, 0, HITNET_MIDDLEBURY_MODEL_WIDTH, HITNET_MIDDLEBURY_MODEL_HEIGHT)
      .data;
  };

  if (typeof createImageBitmap === 'function') {
    const bmp = await createImageBitmap(blob);
    try {
      return drawSource(bmp, bmp.width, bmp.height);
    } finally {
      bmp.close();
    }
  }

  const url = URL.createObjectURL(blob);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error('Failed to decode Middlebury PNG image'));
      el.src = url;
    });
    return drawSource(img, img.naturalWidth, img.naturalHeight);
  } finally {
    URL.revokeObjectURL(url);
  }
}

export const defaultBrowserFactorialFeedPairBuilder: FactorialFeedPairBuilder = async ({
  sceneAssets,
  g0Transform,
  g1Transform,
}) => {
  const [g0L, g0R, g1L, g1R] = await Promise.all([
    decodePngToGeometryRgbaBrowser(sceneAssets.im0Bytes, g0Transform),
    decodePngToGeometryRgbaBrowser(sceneAssets.im1Bytes, g0Transform),
    decodePngToGeometryRgbaBrowser(sceneAssets.im0Bytes, g1Transform),
    decodePngToGeometryRgbaBrowser(sceneAssets.im1Bytes, g1Transform),
  ]);
  return {
    G0: { leftRgba: g0L, rightRgba: g0R },
    G1: { leftRgba: g1L, rightRgba: g1R },
  };
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

export async function runHitnetMiddleburyFactorialDiagnosis(params: {
  m1Bytes: Uint8Array;
  sceneIds?: MiddleburyEval3SceneId[];
  disableCpuEpFallback?: boolean;
  isDev?: boolean;
  runWorker?: ResidualDiagnosticWorkerRunner;
  loadSceneAssets?: (sceneId: MiddleburyEval3SceneId) => Promise<LoadedMiddleburySceneAssets>;
  buildGeometryRgba?: FactorialFeedPairBuilder;
  onLog?: (line: string) => void;
}): Promise<HitnetMiddleburyFactorialReport> {
  assertDevFixtureEnabled(params.isDev);
  const log = params.onLog ?? (() => {});
  const runWorker = params.runWorker ?? createBrowserOrt130DiagnosticRunner();
  const loadScene = params.loadSceneAssets ?? ((id) => loadAndVerifyMiddleburySceneAssets(id));
  const buildGeometryRgba = params.buildGeometryRgba ?? defaultBrowserFactorialFeedPairBuilder;
  const sceneIds: MiddleburyEval3SceneId[] =
    params.sceneIds && params.sceneIds.length > 0 ? params.sceneIds : ['Motorcycle', 'Teddy'];
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
  const samplingAndMaskPolicy =
    'All 4 arms (G0/G1 x RGB/BGR) per scene are evaluated on the exact same pre-prediction intersected physical source coordinates (ix, iy) derived 1-to-1 from G1 interior active rows x 320 cols. GT & mask0/1nocc are read directly at native (ix, iy) with zero GT smoothing or interpolation across invalid/occluded holes; predictions are read at nearest active interior model row without cross-edge blending.';

  log(
    `=== [DEV-ONLY] Middlebury Eval3 Q 2x2 Factorial Diagnosis (Geometry x RGB/BGR, Canonical M1) ===\n` +
      `Runtime: onnxruntime-web@1.30.0 (${asyncify.entryFile} + ${asyncify.wasmFile}) | opt=basic | M1 sha256=${m1Sha} (N1=OFF, 0 replacements)\n` +
      `Policy: ${samplingAndMaskPolicy}`
  );

  const assertSuccess = (
    label: string,
    raw: OrtVersionDiagnosticResponse,
    expectedProvider: 'wasm' | 'webgpu',
    expectedFeedSha: string
  ): OrtVersionDiagnosticSuccess => {
    if (!raw.ok) {
      const f = raw as OrtVersionDiagnosticFailure;
      throw new Error(`${label} FAILED at stage="${f.stage}": ${f.message}`);
    }
    if (
      raw.pinnedOrtVersion !== '1.30.0' ||
      raw.runtimeAssetFamily !== 'asyncify' ||
      raw.graphOptimizationLevel !== 'basic' ||
      raw.provider !== expectedProvider ||
      raw.configuredProviders.length !== 1 ||
      raw.configuredProviders[0] !== expectedProvider ||
      raw.foreignOrtResources.length > 0 ||
      raw.loadedOrtResources.some((u) => u.includes('.jsep.') || u.includes('ort.bundle.'))
    ) {
      throw new Error(
        `${label} runtime/provider/opt guard violation: provider=${raw.provider}, configured=${JSON.stringify(raw.configuredProviders)}`
      );
    }
    if (raw.modelSha256 !== m1Sha || raw.modelByteLength !== params.m1Bytes.byteLength) {
      throw new Error(`${label} pre-session M1 SHA mismatch: ${raw.modelSha256}`);
    }
    if (raw.feedSha256 !== expectedFeedSha) {
      throw new Error(`${label} feed SHA mismatch: ${raw.feedSha256} vs ${expectedFeedSha}`);
    }
    return raw;
  };

  const sceneReports = {} as Record<MiddleburyEval3SceneId, SceneFactorialReport>;
  const bestArmPerScene = {} as Record<MiddleburyEval3SceneId, FactorialArmId>;
  let anyMeaningfulImprovement = false;

  for (const sceneId of sceneIds) {
    const assets = await loadScene(sceneId);
    const { g0, g1 } = buildMiddleburyGeometryTransforms(
      sceneId,
      assets.calib.width,
      assets.calib.height
    );
    const leftSet = buildSharedPhysicalSampleSetForEye({
      eye: 'left',
      sceneId,
      gtPfm: assets.disp0GtPfm,
      maskNocc: assets.mask0Nocc,
      calib: assets.calib,
      g0,
      g1,
    });
    const rightSet = buildSharedPhysicalSampleSetForEye({
      eye: 'right',
      sceneId,
      gtPfm: assets.disp1GtPfm,
      maskNocc: assets.mask1Nocc,
      calib: assets.calib,
      g0,
      g1,
    });

    const rgbaPair = await buildGeometryRgba({
      sceneAssets: assets,
      g0Transform: g0,
      g1Transform: g1,
    });

    log(
      `\n=== Scene: ${sceneId} (${assets.spec.edition}) | native=${assets.calib.width}x${assets.calib.height} ===\n` +
        `  G0: ${g0.roundingNote}\n` +
        `  G1: ${g1.roundingNote}\n` +
        `  Shared Support vs Prior G0 Baseline: Left nonocc ${leftSet.support.sharedNonoccCount}/${leftSet.support.priorG0BaselineNonoccCount} (allValid ${leftSet.support.sharedAllValidCount}/${leftSet.support.priorG0BaselineAllValidCount}, pad/cropExcluded=${leftSet.support.excludedByG1ActiveOrEdgeCount}, counterpartOOB=${leftSet.support.excludedCounterpartOutOfBoundsCount}) | ` +
        `Right nonocc ${rightSet.support.sharedNonoccCount}/${rightSet.support.priorG0BaselineNonoccCount}`
    );

    const feedsByArm = {} as Record<
      FactorialArmId,
      { feedFloat32: Float32Array; feedSha256: string }
    >;
    const rawOutputsByWasmArm = {} as Record<
      FactorialArmId,
      { left: Float32Array; right: Float32Array }
    >;
    const wasmArms = {} as Record<FactorialArmId, FactorialArmRunResult>;

    const executeSingleArm = async (
      armSpec: (typeof FACTORIAL_ARM_SPECS)[number],
      provider: 'wasm' | 'webgpu'
    ): Promise<{
      result: FactorialArmRunResult;
      leftOut: Float32Array;
      rightOut: Float32Array;
    }> => {
      let feedEntry = feedsByArm[armSpec.armId];
      if (!feedEntry) {
        const geomRgba = armSpec.geometry === 'G0_direct_320x240' ? rgbaPair.G0 : rgbaPair.G1;
        const feedFloat32 = packRgbaStereoEyesWithChannelOrder(
          geomRgba.leftRgba,
          geomRgba.rightRgba,
          armSpec.channelOrder
        );
        const feedSha256 = await sha256HexBytes(
          new Uint8Array(feedFloat32.buffer, feedFloat32.byteOffset, feedFloat32.byteLength)
        );
        feedEntry = { feedFloat32, feedSha256 };
        feedsByArm[armSpec.armId] = feedEntry;
      }

      const raw = await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: params.m1Bytes.slice().buffer,
        feedBuffer: feedEntry.feedFloat32.slice().buffer,
        feedName: 'input',
        feedDims: [1, 6, 240, 320],
        provider,
        graphOptimizationLevel: 'basic',
        warmRuns: 1,
        enableWebGpuProfiling: false,
        disableCpuEpFallback: provider === 'webgpu' ? disableCpuEpFallback : false,
        captureGpuValidationErrors: provider === 'webgpu',
      });
      const okRes = assertSuccess(`${sceneId} ${armSpec.armId} ${provider}`, raw, provider, feedEntry.feedSha256);
      const val = validateHitnetFinalOutputs(okRes.outputs);
      if (!val.ok) {
        throw new Error(`${sceneId} ${armSpec.armId} ${provider} output invalid: ${val.error}`);
      }

      let dispatchObserved: boolean | null = null;
      let gpuKernelProfileSummary: OrtGpuKernelProfileSummary | null = null;
      if (provider === 'webgpu') {
        const gpuProf = await runWorker({
          expectedOrtVersion: '1.30.0',
          runtimeAssetFamily: 'asyncify',
          modelBuffer: params.m1Bytes.slice().buffer,
          feedBuffer: feedEntry.feedFloat32.slice().buffer,
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

      const leftOut = val.byName[pName].data;
      const rightOut = val.byName[sName].data;
      const leftEval = evaluateFactorialArmEye({
        pred: leftOut,
        geometry: armSpec.geometry,
        sampleSet: leftSet,
        oppositeSampleSet: rightSet,
        feedFloat32: feedEntry.feedFloat32,
        calib: assets.calib,
        rectificationCaveat: MIDDLEBURY_EVAL3_Q_SCENES[sceneId].rectificationCaveat,
      });
      const rightEval = evaluateFactorialArmEye({
        pred: rightOut,
        geometry: armSpec.geometry,
        sampleSet: rightSet,
        oppositeSampleSet: leftSet,
        feedFloat32: feedEntry.feedFloat32,
        calib: assets.calib,
        rectificationCaveat: MIDDLEBURY_EVAL3_Q_SCENES[sceneId].rectificationCaveat,
      });

      return {
        leftOut,
        rightOut,
        result: {
          armId: armSpec.armId,
          geometry: armSpec.geometry,
          channelOrder: armSpec.channelOrder,
          provider,
          graphOptimizationLevel: 'basic',
          modelSha256: okRes.modelSha256,
          modelByteLength: okRes.modelByteLength,
          feedSha256: feedEntry.feedSha256,
          configuredProviders: okRes.configuredProviders,
          foreignOrtResources: okRes.foreignOrtResources,
          gpuValidationErrors: okRes.gpuValidationErrors ?? [],
          dispatchObserved,
          gpuKernelProfileSummary,
          createMs: okRes.createMs,
          coldRunMs: okRes.coldRunMs,
          warmRunMs: [...okRes.warmRunMs],
          leftPrimary: leftEval,
          rightSecondary: rightEval,
        },
      };
    };

    for (const armSpec of FACTORIAL_ARM_SPECS) {
      const exec = await executeSingleArm(armSpec, 'wasm');
      wasmArms[armSpec.armId] = exec.result;
      rawOutputsByWasmArm[armSpec.armId] = { left: exec.leftOut, right: exec.rightOut };
      const l = exec.result.leftPrimary.nonocc;
      const r = exec.result.rightSecondary.nonocc;
      log(
        `  [WASM ${armSpec.armId}] feed=${exec.result.feedSha256.slice(0, 12)}...\n` +
          `    Left  nonocc(n=${l.evaluatedCount}): bias=${l.signedBiasModelPx.toFixed(4)} EPE=${l.epeModelPx.toFixed(4)}px (${l.epeNativeQPx.toFixed(3)}Q) p95=${l.p95AbsErrorModelPx.toFixed(3)} Bad1=${(l.bad1Rate * 100).toFixed(2)}% Bad3=${(l.bad3Rate * 100).toFixed(2)}% inv=${l.invalidPredCount} OLS=${l.olsSlope.toFixed(3)}*gt+${l.olsInterceptModelPx.toFixed(3)}\n` +
          `      Bands: ${exec.result.leftPrimary.disparityBands.map((b) => `${b.bandId}(n=${b.count},EPE=${b.epeModelPx.toFixed(2)},Bad3=${(b.bad3Rate * 100).toFixed(1)}%)`).join(' | ')}\n` +
          `      Edges: ${exec.result.leftPrimary.edgeCategories.map((b) => `${b.bandId}(n=${b.count},EPE=${b.epeModelPx.toFixed(2)},Bad3=${(b.bad3Rate * 100).toFixed(1)}%)`).join(' | ')}\n` +
          `      Tex:   ${exec.result.leftPrimary.textureCategories.map((b) => `${b.bandId}(n=${b.count},EPE=${b.epeModelPx.toFixed(2)},Bad3=${(b.bad3Rate * 100).toFixed(1)}%)`).join(' | ')}\n` +
          `      WorstClusters: ${exec.result.leftPrimary.worstClustersTop5.slice(0, 3).map((c) => `#${c.rank}[x=${c.modelBox.xStart}:${c.modelBox.xEndExclusive},y=${c.modelBox.yStart}:${c.modelBox.yEndExclusive}|src=${c.sourceBox.ixMin}..${c.sourceBox.ixMax}x${c.sourceBox.iyMin}..${c.sourceBox.iyMax}]:EPE=${c.epeModelPx.toFixed(2)},bias=${c.signedBiasModelPx.toFixed(2)},gt=[${c.gtNativeRangePx[0].toFixed(1)},${c.gtNativeRangePx[1].toFixed(1)}]<=ndisp:${String(c.allGtWithinNdisp)}`).join(' ; ')}\n` +
          `    Right nonocc(n=${r.evaluatedCount}): bias=${r.signedBiasModelPx.toFixed(4)} EPE=${r.epeModelPx.toFixed(4)}px (${r.epeNativeQPx.toFixed(3)}Q) p95=${r.p95AbsErrorModelPx.toFixed(3)} Bad1=${(r.bad1Rate * 100).toFixed(2)}% Bad3=${(r.bad3Rate * 100).toFixed(2)}% inv=${r.invalidPredCount} OLS=${r.olsSlope.toFixed(3)}*gt+${r.olsInterceptModelPx.toFixed(3)}`
      );
    }

    const baseL = wasmArms.G0_C0_direct_RGB.leftPrimary.nonocc;
    const baseR = wasmArms.G0_C0_direct_RGB.rightSecondary.nonocc;
    const deltaVsG0RgbWasm = {} as SceneFactorialReport['deltaVsG0RgbWasm'];

    let bestArmId: FactorialArmId = 'G0_C0_direct_RGB';
    let bestMeanEpe = (baseL.epeModelPx + baseR.epeModelPx) / 2;

    for (const armSpec of FACTORIAL_ARM_SPECS) {
      const aL = wasmArms[armSpec.armId].leftPrimary.nonocc;
      const aR = wasmArms[armSpec.armId].rightSecondary.nonocc;
      deltaVsG0RgbWasm[armSpec.armId] = {
        leftNonoccEpeDeltaPx: aL.epeModelPx - baseL.epeModelPx,
        leftNonoccBad3DeltaPctPts: (aL.bad3Rate - baseL.bad3Rate) * 100,
        rightNonoccEpeDeltaPx: aR.epeModelPx - baseR.epeModelPx,
        rightNonoccBad3DeltaPctPts: (aR.bad3Rate - baseR.bad3Rate) * 100,
      };
      const meanEpe = (aL.epeModelPx + aR.epeModelPx) / 2;
      if (meanEpe < bestMeanEpe) {
        bestMeanEpe = meanEpe;
        bestArmId = armSpec.armId;
      }
    }
    bestArmPerScene[sceneId] = bestArmId;

    const meanBaseEpe = (baseL.epeModelPx + baseR.epeModelPx) / 2;
    const relGain = meanBaseEpe > 0 ? (meanBaseEpe - bestMeanEpe) / meanBaseEpe : 0;
    const hasPromising = bestArmId !== 'G0_C0_direct_RGB' && relGain >= 0.05;
    if (hasPromising) anyMeaningfulImprovement = true;

    // Bounded same-pair GPU confirmation: always run unchanged G0_C0_direct_RGB baseline,
    // plus candidate arm (best alternative arm, prioritizing promising candidate or G1_C0_isotropic_RGB control).
    const gpuBaseSpec = FACTORIAL_ARM_SPECS[0];
    const gpuCandId: FactorialArmId = hasPromising ? bestArmId : 'G1_C0_isotropic_RGB';
    const gpuCandSpec = FACTORIAL_ARM_SPECS.find((s) => s.armId === gpuCandId)!;

    const gpuBaseExec = await executeSingleArm(gpuBaseSpec, 'webgpu');
    const baselineRawParityVsWasm = {
      leftFullFrame: summarizeRawTensorAndParity(
        gpuBaseExec.leftOut,
        rawOutputsByWasmArm.G0_C0_direct_RGB.left
      ),
      rightFullFrame: summarizeRawTensorAndParity(
        gpuBaseExec.rightOut,
        rawOutputsByWasmArm.G0_C0_direct_RGB.right
      ),
    };

    const gpuCandExec = await executeSingleArm(gpuCandSpec, 'webgpu');
    const candidateRawParityVsWasm = {
      leftFullFrame: summarizeRawTensorAndParity(
        gpuCandExec.leftOut,
        rawOutputsByWasmArm[gpuCandId].left
      ),
      rightFullFrame: summarizeRawTensorAndParity(
        gpuCandExec.rightOut,
        rawOutputsByWasmArm[gpuCandId].right
      ),
    };

    log(
      `  [WebGPU Confirmation (${sceneId})] ${gpuBaseSpec.armId}: dispatchObserved=${String(gpuBaseExec.result.dispatchObserved)}, Left nonocc EPE=${gpuBaseExec.result.leftPrimary.nonocc.epeModelPx.toFixed(4)}px | Raw GPU vs WASM parity L(max=${baselineRawParityVsWasm.leftFullFrame.maxAbsDiff.toExponential(3)}, p99=${baselineRawParityVsWasm.leftFullFrame.p99AbsDiff.toExponential(3)}) R(max=${baselineRawParityVsWasm.rightFullFrame.maxAbsDiff.toExponential(3)})\n` +
        `  [WebGPU Candidate    (${sceneId})] ${gpuCandSpec.armId}: dispatchObserved=${String(gpuCandExec.result.dispatchObserved)}, Left nonocc EPE=${gpuCandExec.result.leftPrimary.nonocc.epeModelPx.toFixed(4)}px | Raw GPU vs WASM parity L(max=${candidateRawParityVsWasm.leftFullFrame.maxAbsDiff.toExponential(3)}, p99=${candidateRawParityVsWasm.leftFullFrame.p99AbsDiff.toExponential(3)})`
    );

    const yMapStatusNote =
      assets.calib.dymax > 0
        ? `Scene-level calib.txt vertical y-disparity: dyavg=${assets.calib.dyavg} px, dymax=${assets.calib.dymax} px. Per-pixel signed y-disparity files (disp0GTy.pfm / disp1GTy.pfm) are absent in /public/dev-fixtures/middlebury-eval3/${sceneId}/, so per-pixel y-residual correlation is omitted.`
        : `Scene-level calib.txt vertical y-disparity: dyavg=0 px, dymax=0 px (ideal epipolar alignment).`;

    sceneReports[sceneId] = {
      sceneId,
      edition: assets.spec.edition,
      calib: assets.calib,
      rectificationCaveat: assets.spec.rectificationCaveat,
      yMapStatusNote,
      transforms: { g0, g1 },
      leftSupport: leftSet.support,
      rightSupport: rightSet.support,
      wasmArms,
      gpuConfirmation: {
        baselineArm: gpuBaseExec.result,
        candidateArm: gpuCandExec.result,
        baselineRawParityVsWasm,
        candidateRawParityVsWasm,
      },
      deltaVsG0RgbWasm,
    };
  }

  const verdict: FactorialDiagnosticVerdict = {
    anyMeaningfulImprovement,
    bestWasmArmByMeanNonoccEpe: bestArmPerScene,
    summaryVerdict: anyMeaningfulImprovement
      ? `IMPROVEMENT_OBSERVED: At least one factorial intervention improved mean both-eye nonocc EPE by >=5% (best arms: ${JSON.stringify(bestArmPerScene)}).`
      : `NO_PREPROCESSING_FIX_FOUND: Neither R/B channel swap (BGR) nor fractional isotropic same-scaleX geometry materially fixed Motorcycle/Teddy accuracy over G0_RGB (best arms: ${JSON.stringify(bestArmPerScene)}).`,
    smallestSubsequentFollowUpIfNoImprovement: [
      '1. Raw pre-M1 ONNX weights (hitnet_sceneflow_240x320.onnx, 8,829,487 B) vs canonical M1 (97e3673bdea...) on WASM (opt=basic) on Motorcycle + Teddy G0_C0 to verify zero scene-dependent regression from M1 transforms.',
      '2. Native-resolution 320x240 center crop control (scaleX=scaleY=1.0, zero 2.32x spatial decimation) on Motorcycle Q (741x497) and Teddy Q (450x375) with shared counterpart-in-bounds masks to separate decimation/thin-structure loss from SceneFlow zero-shot domain gap.',
    ],
  };

  log(
    `\n=== Factorial Verdict: ${verdict.summaryVerdict} ===\n` +
      (!anyMeaningfulImprovement
        ? `Smallest Next Controls:\n${verdict.smallestSubsequentFollowUpIfNoImprovement.map((l) => `  ${l}`).join('\n')}`
        : '')
  );

  return {
    ok: true,
    pinnedOrtVersion: '1.30.0',
    runtimeAssetFamily: 'asyncify',
    graphOptimizationLevel: 'basic',
    m1ByteLength: params.m1Bytes.byteLength,
    m1Sha256: m1Sha,
    n1OffProof,
    samplingAndMaskPolicy,
    scenes: sceneReports,
    verdict,
  };
}
