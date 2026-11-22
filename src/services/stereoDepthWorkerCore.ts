import type * as ort from 'onnxruntime-web';
import {
  attachWebGpuDeviceDispatchObserver,
  type OrtLikeGpuDevice,
  type OrtLoadedAssetProvenance,
} from './hitnetOrtComparisonHelper';
import type {
  OnnxBackendInfo,
  StereoBackendPreference,
  StereoColormap,
  StereoDepthVariantId,
} from '../types/vision';

export type StereoInputLayout = 'dual' | 'stacked';
export type StereoNormalizationMode = 'imagenet' | 'zero-one' | 'zero-255' | 'minus-one-one';
export type StereoOnlineDeliveryMode = 'direct-cors' | 'cors-proxy' | 'local-only';

export interface StereoModelProfile {
  variantId: StereoDepthVariantId;
  tier: 'baseline' | 'lightweight' | 'high-accuracy' | 'custom';
  engineType: 'sad-cpu' | 'webgpu-wgsl' | 'onnx';
  fileName: string;
  defaultWidth: number;
  defaultHeight: number;
  inputLayout: StereoInputLayout;
  normalization: StereoNormalizationMode;
  preferWebGpu: boolean;
  license: string;
  licenseNote: string;
  upstreamUrl: string;
  onlineDelivery: StereoOnlineDeliveryMode;
  onlineSourceLabel?: string;
  defaultModelUrl?: string;
  fallbackModelUrl?: string;
  corsProxyUrl?: string;
}

export interface SadComputeOptions {
  maxDisparity?: number;
  windowSize?: 5 | 7 | 9;
  confidenceThreshold?: number;
}

export interface SadDisparityOutput {
  disparity: Float32Array;
  width: number;
  height: number;
  validRatio: number;
}

export interface DisparityStatsResult {
  minDisparity: number;
  maxDisparity: number;
  medianDisparity: number;
  scaleMinDisparity: number;
  scaleMaxDisparity: number;
  validRatio: number;
}

export interface StereoWorkerEstimateOptions {
  confidenceThreshold?: number;
  maxDisparity?: number;
  sadWindowSize?: 5 | 7 | 9;
  autoNormalizeDisparity?: boolean;
  minDisparityPx?: number;
  maxDisparityPx?: number;
  colormap?: StereoColormap;
}

export type OrtGraphOptimizationLevel = 'disabled' | 'basic' | 'extended' | 'layout' | 'all';
export type SliceAxesNormalizeMode = 'off' | 'sort-only' | 'compact-active';
export type HitnetLevelInitGatherMode = 'off' | 'reshape-rank4' | 'reduce-min';

export interface NormalizeSliceAxesOptions {
  mode?: SliceAxesNormalizeMode;
}

export interface NormalizeGatherNdRankOptions {
  variantId?: StereoDepthVariantId | string;
  backendPreference?: StereoBackendPreference;
  requireHitnetContract?: boolean;
  allowDiagnosticTransformedWasm?: boolean;
}

export interface NormalizeArgMinKeepDimsOptions {
  variantId?: StereoDepthVariantId | string;
  backendPreference?: StereoBackendPreference;
  requireHitnetContract?: boolean;
  allowDiagnosticTransformedWasm?: boolean;
}

export interface NormalizeInitToPropZeroSlantOptions {
  variantId?: StereoDepthVariantId | string;
  backendPreference?: StereoBackendPreference;
  requireHitnetContract?: boolean;
  allowDiagnosticTransformedWasm?: boolean;
  rawModelSha256?: string;
  h1ModelSha256?: string;
}

export interface OnnxInitToPropZeroSlantNormalizeResult {
  bytes: Uint8Array;
  normalizedCount: number;
  normalizedNodeNames: string[];
  removedDeadNodeNames: string[];
  removedDeadInitializerNames: string[];
  addedInitializerName: string | null;
  addedInitializerDims: number[] | null;
  verifiedH1Provenance: boolean;
  verifiedRawIdentity: boolean;
  skippedInvalidOrUnprovenCount: number;
}

export interface NormalizeSlot0ReshapeToTransposeOptions {
  variantId?: StereoDepthVariantId | string;
  backendPreference?: StereoBackendPreference;
  requireHitnetContract?: boolean;
  allowDiagnosticTransformedWasm?: boolean;
  rawModelSha256?: string;
  m1ModelSha256?: string;
}

export interface OnnxSlot0ReshapeToTransposeResult {
  bytes: Uint8Array;
  normalizedCount: number;
  normalizedNodeNames: string[];
  verifiedM1Provenance: boolean;
  verifiedRawIdentity: boolean;
  skippedInvalidOrUnprovenCount: number;
}

export interface OnnxLevelInitGatherNormalizeResult extends OnnxGatherNdRankNormalizeResult {
  mode: HitnetLevelInitGatherMode;
  verifiedCostLineageCount: number;
  documentedLimitations: string[];
}

export interface GpuKernelProfileSummary {
  eventCount: number;
  kernelTypes: Record<string, number>;
  samplePrograms: string[];
  timestampQuerySupported: boolean;
  webgpuComputePassCount?: number;
  webgpuDispatchWorkgroupsCount?: number;
  webgpuDispatchIndirectCount?: number;
  webgpuTimestampWritePassCount?: number;
  webgpuQueueSubmitCount?: number;
  webgpuCommandBufferCount?: number;
  webgpuQuerySetCount?: number;
  webgpuSamplePassLabels?: string[];
  deviceObserverAttached?: boolean;
  note: string;
}

export interface StereoSessionOptionsApplied {
  executionProviders: string[];
  graphOptimizationLevel: OrtGraphOptimizationLevel;
  disableCpuEpFallback: boolean;
  wasmPaths: string;
  sliceNormalizeMode: SliceAxesNormalizeMode;
  gatherNdRankNormalized: boolean;
  propGatherNdRankNormalized?: boolean;
  levelInitGatherMode?: HitnetLevelInitGatherMode;
  initToPropZeroSlantNormalized?: boolean;
  slot0ReshapeToTransposeNormalized?: boolean;
  argMinKeepDimsNormalized: boolean;
  allowDiagnosticTransformedWasm?: boolean;
  pretransformedExactM1SkipConcatSplit?: boolean;
  diagnosticTapOutputs?: string[];
  enableWebGpuProfiling: boolean;
}

export interface StereoWorkerInitTiming {
  prepMs: number;
  sessionCreateMs: number;
  totalInitMs: number;
}

export interface StereoWorkerEstimateTiming {
  preprocessMs: number;
  runMs: number;
  postprocessMs: number;
  totalEstimateMs: number;
}

export interface StereoRawTensorOutput {
  name: string;
  dims: number[];
  dataType?: string;
  buffer: ArrayBuffer;
  exactInt64Buffer?: ArrayBuffer;
}

export type StereoWorkerRequest =
  | {
      type: 'init-sad';
      id: number;
      profile: StereoModelProfile;
    }
  | {
      type: 'init-wgsl';
      id: number;
      profile: StereoModelProfile;
      backendPreference?: StereoBackendPreference;
    }
  | {
      type: 'init-onnx';
      id: number;
      profile: StereoModelProfile;
      onnxBuffer: ArrayBuffer;
      wasmPrimaryUrl: string;
      wasmFallbackUrl: string | null;
      backendPreference?: StereoBackendPreference;
      normalizeSliceAxes?: boolean | SliceAxesNormalizeMode;
      normalizeGatherNdRank?: boolean;
      normalizePropGatherNdRank?: boolean;
      levelInitGatherMode?: HitnetLevelInitGatherMode;
      normalizeInitToPropZeroSlant?: boolean;
      normalizeSlot0ReshapeToTranspose?: boolean;
      normalizeArgMinKeepDims?: boolean;
      allowDiagnosticTransformedWasm?: boolean;
      skipImplicitConcatSplitForExactM1Fixture?: boolean;
      diagnosticTapOutputs?: string[];
      strictWebGpu?: boolean;
      graphOptimizationLevel?: OrtGraphOptimizationLevel;
      disableCpuEpFallback?: boolean;
      enableWebGpuProfiling?: boolean;
    }
  | {
      type: 'estimate';
      id: number;
      leftRgbaBuffer: ArrayBuffer;
      rightRgbaBuffer: ArrayBuffer;
      width: number;
      height: number;
      sourceEyeWidth: number;
      sourceEyeHeight: number;
      options: StereoWorkerEstimateOptions;
      includeRawTensorOutputs?: boolean;
      fetchPrimaryOutputOnly?: boolean;
      captureGpuValidationErrors?: boolean;
    }
  | {
      type: 'dispose';
      id: number;
    };

export type StereoWorkerResponse =
  | {
      type: 'init-result';
      id: number;
      ok: true;
      backend: OnnxBackendInfo;
      wasmSourceUsed: 'primary' | 'fallback';
      targetWidth: number;
      targetHeight: number;
      workerGpuAdapter?: string | null;
      ortVersions?: {
        common?: string;
        web?: string;
      } | null;
      runtimeAssetFamily?: 'asyncify' | 'jsep';
      entryModuleUrl?: string;
      loadedOrtResources?: string[];
      foreignOrtResources?: string[];
      loadedAssetProvenance?: OrtLoadedAssetProvenance[];
      rawInputModelSha256?: string;
      baseSessionModelSha256?: string;
      sessionModelSha256?: string;
      sessionModelByteLength?: number;
      tappedOutputs?: string[];
      sessionOptionsApplied?: StereoSessionOptionsApplied;
      initTiming?: StereoWorkerInitTiming;
      gatherNdStats?: {
        normalizedCount: number;
        normalizedNodeNames: string[];
        skippedDynamicCount: number;
        skippedInvalidOrUnprovenCount: number;
        totalGatherNdCount: number;
      };
      propGatherNdStats?: {
        normalizedCount: number;
        normalizedNodeNames: string[];
        skippedDynamicCount: number;
        skippedInvalidOrUnprovenCount: number;
        totalGatherNdCount: number;
      };
      levelInitGatherStats?: {
        mode: HitnetLevelInitGatherMode;
        normalizedCount: number;
        normalizedNodeNames: string[];
        skippedDynamicCount: number;
        skippedInvalidOrUnprovenCount: number;
        totalGatherNdCount: number;
        verifiedCostLineageCount: number;
        documentedLimitations: string[];
      };
      initToPropZeroSlantStats?: {
        normalizedCount: number;
        normalizedNodeNames: string[];
        removedDeadNodeNames: string[];
        removedDeadInitializerNames: string[];
        addedInitializerName: string | null;
        addedInitializerDims: number[] | null;
        verifiedH1Provenance: boolean;
        verifiedRawIdentity: boolean;
        skippedInvalidOrUnprovenCount: number;
      };
      slot0ReshapeToTransposeStats?: {
        normalizedCount: number;
        normalizedNodeNames: string[];
        verifiedM1Provenance: boolean;
        verifiedRawIdentity: boolean;
        skippedInvalidOrUnprovenCount: number;
      };
      argMinStats?: {
        normalizedCount: number;
        normalizedNodeNames: string[];
        untouchedNodeNames: string[];
        totalArgMinCount: number;
        skippedDynamicCount: number;
        skippedInvalidOrUnprovenCount: number;
        updatedValueInfoCount: number;
        ortCacheKeyBefore: string | null;
        ortCacheKeyAfter: string | null;
      };
      transformStats?: {
        splitCount: number;
        decomposed5dCount: number;
        sliceNormalizedCount: number;
        sliceCompactedCount: number;
        sliceSkippedDynamicCount: number;
        sliceSkippedInvalidOrUnprovenCount: number;
      };
    }
  | {
      type: 'estimate-result';
      id: number;
      ok: true;
      width: number;
      height: number;
      disparityBuffer: ArrayBuffer;
      overlayRgbaBuffer: ArrayBuffer;
      stats: DisparityStatsResult;
      backend: OnnxBackendInfo;
      gpuKernelProfileSummary?: GpuKernelProfileSummary | null;
      gpuValidationErrors?: string[];
      estimateTiming?: StereoWorkerEstimateTiming;
      feedSha256s?: Record<string, string>;
      feedMetadata?: Record<string, { dims: number[]; dataType: string }>;
      fetchedOutputNames?: string[];
      rawOutputs?: StereoRawTensorOutput[];
    }
  | {
      type: 'dispose-result';
      id: number;
      ok: true;
    }
  | {
      type: 'error';
      id: number;
      ok: false;
      stage: 'wasm' | 'task' | 'input' | 'run' | 'output';
      message: string;
    };

/**
 * Separable sliding-window SAD stereo block matcher with sub-pixel parabolic interpolation.
 * Runs in O(W * H * D) integer arithmetic without allocating 3D cost volumes.
 */
export function computeSadDisparity(
  leftRgba: Uint8ClampedArray,
  rightRgba: Uint8ClampedArray,
  width: number,
  height: number,
  options: SadComputeOptions = {}
): SadDisparityOutput {
  const n = width * height;
  if (n <= 0 || leftRgba.length < n * 4 || rightRgba.length < n * 4) {
    throw new RangeError(`Invalid RGBA buffers for SAD stereo (${width}x${height})`);
  }

  const maxD = Math.max(4, Math.min(width - 2, Math.floor(options.maxDisparity ?? 48)));
  const winSize = options.windowSize === 5 || options.windowSize === 9 ? options.windowSize : 7;
  const radius = Math.floor(winSize / 2);
  const confThresh = options.confidenceThreshold ?? 0.35;
  const maxMeanSad = Math.max(18, Math.round(115 - confThresh * 85));

  // 1. Convert RGBA to 8-bit luminance + align dual-camera mean exposure + horizontal gradient
  const leftGray = new Uint8Array(n);
  const rightGray = new Uint8Array(n);
  const leftGrad = new Uint8Array(n);
  const rightGrad = new Uint8Array(n);
  let sumL = 0;
  let sumR = 0;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const lg = (77 * leftRgba[o] + 150 * leftRgba[o + 1] + 29 * leftRgba[o + 2]) >> 8;
    const rg = (77 * rightRgba[o] + 150 * rightRgba[o + 1] + 29 * rightRgba[o + 2]) >> 8;
    leftGray[i] = lg;
    rightGray[i] = rg;
    sumL += lg;
    sumR += rg;
  }
  const meanDiff = Math.round((sumL - sumR) / Math.max(1, n));
  if (Math.abs(meanDiff) >= 2) {
    for (let i = 0; i < n; i++) {
      const adj = rightGray[i] + meanDiff;
      rightGray[i] = adj < 0 ? 0 : adj > 255 ? 255 : adj;
    }
  }
  for (let y = 0; y < height; y++) {
    const rowOff = y * width;
    for (let x = 1; x < width - 1; x++) {
      const gl = leftGray[rowOff + x + 1] - leftGray[rowOff + x - 1];
      const gr = rightGray[rowOff + x + 1] - rightGray[rowOff + x - 1];
      leftGrad[rowOff + x] = gl < 0 ? -gl : gl;
      rightGrad[rowOff + x] = gr < 0 ? -gr : gr;
    }
  }

  // 2. Separable sliding box filter buffers
  const diffRow = new Int32Array(n);
  const horiz = new Int32Array(n);
  const winArea = winSize * winSize;

  const bestCost = new Int32Array(n);
  bestCost.fill(0x7fffffff);
  const secondBestCost = new Int32Array(n);
  secondBestCost.fill(0x7fffffff);
  const bestDisp = new Int16Array(n);
  const prevCostAtBest = new Int32Array(n);
  const nextCostAtBest = new Int32Array(n);
  prevCostAtBest.fill(0x7fffffff);
  nextCostAtBest.fill(0x7fffffff);

  let prevCost = new Int32Array(n);
  let currCost = new Int32Array(n);
  const colSum = new Int32Array(width);
  const costHistory = new Int32Array((maxD + 1) * n);

  for (let d = 0; d <= maxD; d++) {
    for (let y = 0; y < height; y++) {
      const rowOff = y * width;
      for (let x = 0; x < d; x++) {
        const diff = leftGray[rowOff + x] - rightGray[rowOff];
        const gDiff = leftGrad[rowOff + x] - rightGrad[rowOff];
        diffRow[rowOff + x] =
          (diff < 0 ? -diff : diff) + ((gDiff < 0 ? -gDiff : gDiff) >> 1) + 12;
      }
      for (let x = d; x < width; x++) {
        const diff = leftGray[rowOff + x] - rightGray[rowOff + x - d];
        const gDiff = leftGrad[rowOff + x] - rightGrad[rowOff + x - d];
        diffRow[rowOff + x] =
          (diff < 0 ? -diff : diff) + ((gDiff < 0 ? -gDiff : gDiff) >> 1);
      }

      let sum = 0;
      for (let k = -radius; k <= radius; k++) {
        const cx = k < 0 ? 0 : k >= width ? width - 1 : k;
        sum += diffRow[rowOff + cx];
      }
      horiz[rowOff] = sum;
      for (let x = 1; x < width; x++) {
        const addX = x + radius < width ? x + radius : width - 1;
        const subX = x - radius - 1 >= 0 ? x - radius - 1 : 0;
        sum += diffRow[rowOff + addX] - diffRow[rowOff + subX];
        horiz[rowOff + x] = sum;
      }
    }

    colSum.fill(0);
    for (let ky = -radius; ky <= radius; ky++) {
      const cy = ky < 0 ? 0 : ky >= height ? height - 1 : ky;
      const rOff = cy * width;
      for (let x = 0; x < width; x++) {
        colSum[x] += horiz[rOff + x];
      }
    }
    for (let x = 0; x < width; x++) {
      currCost[x] = colSum[x];
    }
    for (let y = 1; y < height; y++) {
      const addY = y + radius < height ? y + radius : height - 1;
      const subY = y - radius - 1 >= 0 ? y - radius - 1 : 0;
      const addOff = addY * width;
      const subOff = subY * width;
      const dstOff = y * width;
      for (let x = 0; x < width; x++) {
        const v = colSum[x] + horiz[addOff + x] - horiz[subOff + x];
        colSum[x] = v;
        currCost[dstOff + x] = v;
      }
    }

    const regPenalty = d;
    const dBase = d * n;

    for (let i = 0; i < n; i++) {
      const c = currCost[i] + regPenalty;
      costHistory[dBase + i] = c;
      if (d > 0 && bestDisp[i] === d - 1) {
        nextCostAtBest[i] = c;
      }
      if (c < bestCost[i]) {
        bestCost[i] = c;
        bestDisp[i] = d;
        prevCostAtBest[i] = d > 0 ? prevCost[i] + (d - 1) : c;
        nextCostAtBest[i] = c;
      }
    }

    const tmp = prevCost;
    prevCost = currCost;
    currCost = tmp;
  }

  for (let d = 0; d <= maxD; d++) {
    const dBase = d * n;
    for (let i = 0; i < n; i++) {
      const bd = bestDisp[i];
      if (d - bd >= 2 || bd - d >= 2) {
        const c = costHistory[dBase + i];
        if (c < secondBestCost[i]) {
          secondBestCost[i] = c;
        }
      }
    }
  }

  // 3. Subpixel parabolic refinement + validity & uniqueness gating
  const rawDisp = new Float32Array(n);
  const maxCostSum = maxMeanSad * winArea;
  let passCount = 0;
  for (let i = 0; i < n; i++) {
    if (bestCost[i] <= maxCostSum) passCount++;
  }
  let effectiveMaxCostSum = maxCostSum;
  if (passCount < n * 0.25) {
    const sortedCosts = Int32Array.from(bestCost).sort();
    const p85Cost = sortedCosts[Math.min(n - 1, Math.floor(n * 0.85))];
    effectiveMaxCostSum = Math.max(maxCostSum, p85Cost);
  }

  for (let y = 0; y < height; y++) {
    const rowOff = y * width;
    for (let x = 0; x < width; x++) {
      const i = rowOff + x;
      const cBest = bestCost[i];
      if (cBest > effectiveMaxCostSum) {
        rawDisp[i] = -1;
        continue;
      }
      const d = bestDisp[i];
      // Reject high-disparity false matches in low-uniqueness / flat regions (sky, walls, compression blocks)
      const cSecond = secondBestCost[i];
      if (d > 16 && cSecond < 0x7fffffff) {
        const uniqMargin = (cSecond - cBest) / Math.max(cBest, winArea * 3);
        const localGrad = leftGrad[i];
        if (uniqMargin < 0.12 || (localGrad < 4 && uniqMargin < 0.24)) {
          rawDisp[i] = -1;
          continue;
        }
      }
      let sub = d;
      if (d > 0 && d < maxD) {
        const cPrev = prevCostAtBest[i];
        const cNext = nextCostAtBest[i];
        const denom = 2 * (cPrev - 2 * cBest + cNext);
        if (denom > 0) {
          const delta = (cPrev - cNext) / denom;
          if (Number.isFinite(delta)) {
            sub = d + Math.max(-0.5, Math.min(0.5, delta));
          }
        }
      }
      rawDisp[i] = sub;
    }
  }

  // 4. Edge-aware 5x5 strided median + bilateral filter to suppress real-video speckles
  const filtered = new Float32Array(n);
  const windowVals = new Float32Array(13);
  let validCount = 0;

  const offsetsY = [-2, -2, -2, -1, -1, -1, 0, 0, 0, 1, 1, 1, 2];
  const offsetsX = [-2, 0, 2, -1, 0, 1, -2, 0, 2, -1, 0, 1, 0];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const center = rawDisp[idx];
      if (center < 0) {
        filtered[idx] = -1;
        continue;
      }
      let count = 0;
      for (let k = 0; k < 13; k++) {
        const sy = Math.max(0, Math.min(height - 1, y + offsetsY[k]));
        const sx = Math.max(0, Math.min(width - 1, x + offsetsX[k]));
        const v = rawDisp[sy * width + sx];
        if (v >= 0) {
          windowVals[count++] = v;
        }
      }
      if (count === 0) {
        filtered[idx] = -1;
        continue;
      }
      for (let a = 1; a < count; a++) {
        const key = windowVals[a];
        let b = a - 1;
        while (b >= 0 && windowVals[b] > key) {
          windowVals[b + 1] = windowVals[b];
          b--;
        }
        windowVals[b + 1] = key;
      }
      const med = windowVals[count >> 1];
      const baseDisp = Math.abs(center - med) > 4.5 ? med : center;
      const centerGray = leftGray[idx];
      let wSum = 0;
      let dSum = 0;
      for (let dy = -2; dy <= 2; dy++) {
        const sy = Math.max(0, Math.min(height - 1, y + dy));
        const rOff = sy * width;
        for (let dx = -2; dx <= 2; dx++) {
          const sx = Math.max(0, Math.min(width - 1, x + dx));
          const nIdx = rOff + sx;
          const nv = rawDisp[nIdx];
          if (nv >= 0 && Math.abs(nv - med) <= 5.0) {
            const gDiff = Math.abs(leftGray[nIdx] - centerGray);
            const w = 1 / (1 + gDiff * 0.08);
            wSum += w;
            dSum += w * nv;
          }
        }
      }
      filtered[idx] = wSum > 0 ? dSum / wSum : baseDisp;
      validCount++;
    }
  }

  return {
    disparity: filtered,
    width,
    height,
    validRatio: n > 0 ? validCount / n : 0,
  };
}

// ============================================================================
// Colormap 256-entry RGB Lookup Tables (Perceptual Turbo / Viridis / Inferno)
// ============================================================================

const LUT_CACHE = new Map<StereoColormap, Uint8ClampedArray>();

function lerpColorStops(
  stops: ReadonlyArray<readonly [number, number, number, number]>,
  t: number
): [number, number, number] {
  if (t <= stops[0][0]) return [stops[0][1], stops[0][2], stops[0][3]];
  const last = stops[stops.length - 1];
  if (t >= last[0]) return [last[1], last[2], last[3]];
  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i];
    const b = stops[i + 1];
    if (t >= a[0] && t <= b[0]) {
      const f = (t - a[0]) / Math.max(1e-6, b[0] - a[0]);
      return [
        Math.round(a[1] + (b[1] - a[1]) * f),
        Math.round(a[2] + (b[2] - a[2]) * f),
        Math.round(a[3] + (b[3] - a[3]) * f),
      ];
    }
  }
  return [last[1], last[2], last[3]];
}

const TURBO_STOPS: ReadonlyArray<readonly [number, number, number, number]> = [
  [0.0, 48, 18, 59],
  [0.125, 70, 98, 216],
  [0.25, 53, 171, 248],
  [0.375, 27, 229, 181],
  [0.5, 116, 254, 93],
  [0.625, 210, 232, 52],
  [0.75, 254, 170, 51],
  [0.875, 238, 86, 16],
  [1.0, 122, 4, 3],
];

const VIRIDIS_STOPS: ReadonlyArray<readonly [number, number, number, number]> = [
  [0.0, 68, 1, 84],
  [0.25, 59, 82, 139],
  [0.5, 33, 145, 140],
  [0.75, 94, 201, 98],
  [1.0, 253, 231, 37],
];

const INFERNO_STOPS: ReadonlyArray<readonly [number, number, number, number]> = [
  [0.0, 0, 0, 4],
  [0.25, 87, 16, 110],
  [0.5, 188, 55, 84],
  [0.75, 249, 142, 9],
  [1.0, 252, 255, 164],
];

export function getColormapLut(colormap: StereoColormap): Uint8ClampedArray {
  const cached = LUT_CACHE.get(colormap);
  if (cached) return cached;
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let rgb: [number, number, number];
    if (colormap === 'grayscale') {
      const v = i;
      rgb = [v, v, v];
    } else if (colormap === 'viridis') {
      rgb = lerpColorStops(VIRIDIS_STOPS, t);
    } else if (colormap === 'inferno') {
      rgb = lerpColorStops(INFERNO_STOPS, t);
    } else {
      rgb = lerpColorStops(TURBO_STOPS, t);
    }
    lut[i * 3] = rgb[0];
    lut[i * 3 + 1] = rgb[1];
    lut[i * 3 + 2] = rgb[2];
  }
  LUT_CACHE.set(colormap, lut);
  return lut;
}

export function renderDisparityColormapToRgba(
  disparity: Float32Array,
  width: number,
  height: number,
  scaleMinDisp: number,
  scaleMaxDisp: number,
  colormap: StereoColormap,
  targetRgba?: Uint8ClampedArray
): Uint8ClampedArray {
  const n = width * height;
  const data =
    targetRgba && targetRgba.length === n * 4
      ? targetRgba
      : new Uint8ClampedArray(n * 4);
  const lut = getColormapLut(colormap);
  const span = Math.max(1e-3, scaleMaxDisp - scaleMinDisp);

  for (let i = 0; i < n; i++) {
    const d = disparity[i];
    const p = i * 4;
    if (d < 0 || !Number.isFinite(d)) {
      data[p] = 0;
      data[p + 1] = 0;
      data[p + 2] = 0;
      data[p + 3] = 0;
      continue;
    }
    const norm = Math.max(0, Math.min(1, (d - scaleMinDisp) / span));
    const idx = Math.round(norm * 255) * 3;
    data[p] = lut[idx];
    data[p + 1] = lut[idx + 1];
    data[p + 2] = lut[idx + 2];
    data[p + 3] = 255;
  }
  return data;
}

export function computeDisparityStats(
  disparity: Float32Array,
  autoNormalize: boolean,
  manualMin: number,
  manualMax: number
): DisparityStatsResult {
  const validValues: number[] = [];
  let minD = Infinity;
  let maxD = -Infinity;
  for (let i = 0; i < disparity.length; i++) {
    const d = disparity[i];
    if (d >= 0 && Number.isFinite(d)) {
      validValues.push(d);
      if (d < minD) minD = d;
      if (d > maxD) maxD = d;
    }
  }
  if (validValues.length === 0) {
    return {
      minDisparity: 0,
      maxDisparity: 0,
      medianDisparity: 0,
      scaleMinDisparity: manualMin,
      scaleMaxDisparity: Math.max(manualMin + 1, manualMax),
      validRatio: 0,
    };
  }
  validValues.sort((a, b) => a - b);
  const median = validValues[validValues.length >> 1];
  const p02 = validValues[Math.floor(validValues.length * 0.02)];
  const p98 = validValues[Math.min(validValues.length - 1, Math.floor(validValues.length * 0.98))];

  const scaleMin = autoNormalize ? p02 : manualMin;
  const scaleMax = autoNormalize ? Math.max(p02 + 1, p98) : Math.max(manualMin + 1, manualMax);

  return {
    minDisparity: minD,
    maxDisparity: maxD,
    medianDisparity: median,
    scaleMinDisparity: scaleMin,
    scaleMaxDisparity: scaleMax,
    validRatio: validValues.length / Math.max(1, disparity.length),
  };
}

export function disparityToDepthMeters(
  disparityPx: number,
  baselineMm: number,
  focalLengthPx: number
): number | null {
  if (
    !Number.isFinite(disparityPx) ||
    !Number.isFinite(baselineMm) ||
    !Number.isFinite(focalLengthPx) ||
    disparityPx <= 0.05 ||
    baselineMm <= 0 ||
    focalLengthPx <= 0
  ) {
    return null;
  }
  const baselineM = baselineMm / 1000;
  return (focalLengthPx * baselineM) / disparityPx;
}

const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
const IMAGENET_STD = [0.229, 0.224, 0.225] as const;

export function rgbaToChwInto(
  rgba: Uint8ClampedArray,
  pixelCount: number,
  out: Float32Array,
  offsetFloats: number,
  normalization: StereoNormalizationMode
): void {
  const c1 = offsetFloats;
  const c2 = offsetFloats + pixelCount;
  const c3 = offsetFloats + pixelCount * 2;

  if (normalization === 'zero-255') {
    for (let i = 0; i < pixelCount; i++) {
      const o = i * 4;
      out[c1 + i] = rgba[o];
      out[c2 + i] = rgba[o + 1];
      out[c3 + i] = rgba[o + 2];
    }
  } else if (normalization === 'zero-one') {
    const inv255 = 1 / 255;
    for (let i = 0; i < pixelCount; i++) {
      const o = i * 4;
      out[c1 + i] = rgba[o] * inv255;
      out[c2 + i] = rgba[o + 1] * inv255;
      out[c3 + i] = rgba[o + 2] * inv255;
    }
  } else if (normalization === 'minus-one-one') {
    const inv127 = 1 / 127.5;
    for (let i = 0; i < pixelCount; i++) {
      const o = i * 4;
      out[c1 + i] = rgba[o] * inv127 - 1;
      out[c2 + i] = rgba[o + 1] * inv127 - 1;
      out[c3 + i] = rgba[o + 2] * inv127 - 1;
    }
  } else {
    const inv255 = 1 / 255;
    for (let i = 0; i < pixelCount; i++) {
      const o = i * 4;
      out[c1 + i] = (rgba[o] * inv255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      out[c2 + i] = (rgba[o] * inv255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      out[c3 + i] = (rgba[o] * inv255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
  }
}

/**
 * Pure bilinear/box downscaler for RGBA buffers (DOM-free, safe in Web Workers).
 */
export function downscaleRgbaBuffer(
  srcRgba: Uint8ClampedArray,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(dstW * dstH * 4);
  if (srcW === dstW && srcH === dstH) {
    out.set(srcRgba.subarray(0, out.length));
    return out;
  }
  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;
  for (let y = 0; y < dstH; y++) {
    const sy0 = Math.min(srcH - 1, Math.floor(y * scaleY));
    const sy1 = Math.min(srcH - 1, Math.floor((y + 0.5) * scaleY));
    const row0 = sy0 * srcW;
    const row1 = sy1 * srcW;
    const dstRow = y * dstW;
    for (let x = 0; x < dstW; x++) {
      const sx0 = Math.min(srcW - 1, Math.floor(x * scaleX));
      const sx1 = Math.min(srcW - 1, Math.floor((x + 0.5) * scaleX));
      const p00 = (row0 + sx0) * 4;
      const p01 = (row0 + sx1) * 4;
      const p10 = (row1 + sx0) * 4;
      const p11 = (row1 + sx1) * 4;
      const d = (dstRow + x) * 4;
      out[d] = (srcRgba[p00] + srcRgba[p01] + srcRgba[p10] + srcRgba[p11]) >> 2;
      out[d + 1] = (srcRgba[p00 + 1] + srcRgba[p01 + 1] + srcRgba[p10 + 1] + srcRgba[p11 + 1]) >> 2;
      out[d + 2] = (srcRgba[p00 + 2] + srcRgba[p01 + 2] + srcRgba[p10 + 2] + srcRgba[p11 + 2]) >> 2;
      out[d + 3] = 255;
    }
  }
  return out;
}

export function extractDisparityFromOrtTensor(
  tensor: { data: unknown; dims: readonly number[] }
): { disparity: Float32Array; width: number; height: number } {
  const dims = tensor.dims;
  if (dims.length < 2) {
    throw new Error(`stereo-depth output: Unexpected output tensor rank: [${dims.join(',')}]`);
  }
  const isNhwcSingleChannel = dims.length === 4 && dims[3] === 1 && dims[1] > 1 && dims[2] > 1;
  const height = isNhwcSingleChannel ? dims[1] : dims[dims.length - 2];
  const width = isNhwcSingleChannel ? dims[2] : dims[dims.length - 1];
  const n = width * height;
  if (n <= 0) {
    throw new Error(`stereo-depth output: Invalid output dimensions: ${width}x${height}`);
  }
  const channels = isNhwcSingleChannel
    ? 1
    : dims.length === 4
      ? dims[1]
      : dims.length === 3 && dims[0] > 1
        ? dims[0]
        : 1;
  const raw = tensor.data as Float32Array | number[];
  if (!raw || raw.length < n) {
    throw new Error('stereo-depth output: Output tensor buffer smaller than spatial size');
  }

  const out = new Float32Array(n);
  let negSum = 0;
  let posSum = 0;
  for (let i = 0; i < n; i++) {
    const v = raw[i];
    if (Number.isFinite(v)) {
      if (v < 0) negSum -= v;
      else posSum += v;
    }
  }
  const negate = negSum > posSum * 1.1;

  if (channels === 2 || negate) {
    for (let i = 0; i < n; i++) {
      const v = raw[i];
      if (!Number.isFinite(v)) {
        out[i] = -1;
      } else {
        const d = negate ? -v : v;
        out[i] = d >= 0 ? d : 0;
      }
    }
  } else {
    for (let i = 0; i < n; i++) {
      const v = raw[i];
      out[i] = Number.isFinite(v) && v >= 0 ? v : -1;
    }
  }
  return { disparity: out, width, height };
}

export function isWebGpuRuntimeKernelError(error: unknown): boolean {
  const parts: string[] = [];
  if (error instanceof Error) {
    parts.push(error.message);
    if (error.cause) parts.push(String(error.cause));
  } else {
    parts.push(String(error));
  }
  return /jsep|webgpu|wgsl|gpuvalidationerror|gpubuffer|createcomputepipeline|createbindgroup|executekernel/i.test(
    parts.join(' ')
  );
}

// ============================================================================
// ONNX Protobuf ValueInfo Sanitizer & WebGPU Compatibility Inspector
// ============================================================================

interface ProtobufFieldSlice {
  fn: number;
  wt: number;
  fStart: number;
  fEnd: number;
  valStart: number;
  valEnd: number;
}

function readProtoVarint(buf: Uint8Array, pos: number, end: number): [number, number] | null {
  let result = 0n;
  let shift = 0n;
  let cursor = pos;
  while (cursor < end && shift <= 63n) {
    const byte = buf[cursor++];
    result |= BigInt(byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) {
      return [Number(result), cursor];
    }
    shift += 7n;
  }
  return null;
}

function encodeProtoVarint(value: number): Uint8Array {
  const bytes: number[] = [];
  let v = BigInt(value);
  while (v >= 0x80n) {
    bytes.push(Number(v & 0x7fn) | 0x80);
    v >>= 7n;
  }
  bytes.push(Number(v));
  return new Uint8Array(bytes);
}

function parseProtoFields(buf: Uint8Array, start: number, end: number): ProtobufFieldSlice[] | null {
  const fields: ProtobufFieldSlice[] = [];
  let pos = start;
  while (pos < end) {
    const fStart = pos;
    const tagRead = readProtoVarint(buf, pos, end);
    if (!tagRead) return null;
    const [tag, p1] = tagRead;
    const fn = tag >>> 3;
    const wt = tag & 7;
    if (fn <= 0) return null;

    let valStart = p1;
    let valEnd: number;
    if (wt === 0) {
      const next = readProtoVarint(buf, p1, end);
      if (!next) return null;
      valEnd = next[1];
    } else if (wt === 1) {
      valEnd = p1 + 8;
    } else if (wt === 2) {
      const lenRead = readProtoVarint(buf, p1, end);
      if (!lenRead) return null;
      const [len, p2] = lenRead;
      if (len < 0) return null;
      valStart = p2;
      valEnd = p2 + len;
    } else if (wt === 5) {
      valEnd = p1 + 4;
    } else {
      return null;
    }
    if (valEnd > end) return null;
    fields.push({ fn, wt, fStart, fEnd: valEnd, valStart, valEnd });
    pos = valEnd;
  }
  return fields;
}

export function stripOnnxGraphValueInfo(bytes: Uint8Array): Uint8Array {
  if (bytes.byteLength < 16) return bytes;
  try {
    const topFields = parseProtoFields(bytes, 0, bytes.byteLength);
    if (!topFields) return bytes;

    const graphIndex = topFields.findIndex((f) => f.fn === 7 && f.wt === 2);
    if (graphIndex < 0) return bytes;

    const graphField = topFields[graphIndex];
    const graphFields = parseProtoFields(bytes, graphField.valStart, graphField.valEnd);
    if (!graphFields) return bytes;

    const hasValueInfo = graphFields.some((f) => f.fn === 13);
    if (!hasValueInfo) return bytes;

    const keptGraphSlices = graphFields.filter((f) => f.fn !== 13);
    let strippedGraphLen = 0;
    for (const slice of keptGraphSlices) {
      strippedGraphLen += slice.fEnd - slice.fStart;
    }

    const graphHeaderTag = encodeProtoVarint((7 << 3) | 2);
    const graphHeaderLen = encodeProtoVarint(strippedGraphLen);

    let totalLen = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        totalLen += graphHeaderTag.byteLength + graphHeaderLen.byteLength + strippedGraphLen;
      } else {
        totalLen += topFields[i].fEnd - topFields[i].fStart;
      }
    }

    const out = new Uint8Array(totalLen);
    let offset = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        out.set(graphHeaderTag, offset);
        offset += graphHeaderTag.byteLength;
        out.set(graphHeaderLen, offset);
        offset += graphHeaderLen.byteLength;
        for (const slice of keptGraphSlices) {
          const chunk = bytes.subarray(slice.fStart, slice.fEnd);
          out.set(chunk, offset);
          offset += chunk.byteLength;
        }
      } else {
        const chunk = bytes.subarray(topFields[i].fStart, topFields[i].fEnd);
        out.set(chunk, offset);
        offset += chunk.byteLength;
      }
    }
    return out;
  } catch {
    return bytes;
  }
}

export const DEFAULT_WEBGPU_MAX_STORAGE_BUFFERS_PER_STAGE = 8;

export interface WebGpuStorageBufferInspection {
  exceeds: boolean;
  maxNodeBuffers: number;
  opType: string | null;
}

export function exceedsWebGpuStorageBufferLimit(
  bytes: Uint8Array,
  maxStorageBuffersPerStage = DEFAULT_WEBGPU_MAX_STORAGE_BUFFERS_PER_STAGE
): WebGpuStorageBufferInspection {
  const safeResult: WebGpuStorageBufferInspection = {
    exceeds: false,
    maxNodeBuffers: 0,
    opType: null,
  };
  if (bytes.byteLength < 16) return safeResult;
  try {
    const topFields = parseProtoFields(bytes, 0, bytes.byteLength);
    if (!topFields) return safeResult;

    const graphField = topFields.find((f) => f.fn === 7 && f.wt === 2);
    if (!graphField) return safeResult;

    const graphFields = parseProtoFields(bytes, graphField.valStart, graphField.valEnd);
    if (!graphFields) return safeResult;

    const decoder = new TextDecoder();
    let maxNodeBuffers = 0;
    let worstOpType: string | null = null;

    for (const gf of graphFields) {
      if (gf.fn !== 1 || gf.wt !== 2) continue;
      const nodeFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
      if (!nodeFields) continue;

      let tensorBindings = 0;
      let opType = '';
      for (const nf of nodeFields) {
        if ((nf.fn === 1 || nf.fn === 2) && nf.wt === 2 && nf.valEnd > nf.valStart) {
          tensorBindings++;
        } else if (nf.fn === 4 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          opType = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
        }
      }

      if (tensorBindings > maxNodeBuffers) {
        maxNodeBuffers = tensorBindings;
        worstOpType = opType || null;
      }
      if (tensorBindings > maxStorageBuffersPerStage) {
        return {
          exceeds: true,
          maxNodeBuffers: tensorBindings,
          opType: opType || null,
        };
      }
    }

    return {
      exceeds: false,
      maxNodeBuffers,
      opType: worstOpType,
    };
  } catch {
    return safeResult;
  }
}

export interface WebGpuCompatibilityInspection {
  compatible: boolean;
  reason: string | null;
  storageBuffers: WebGpuStorageBufferInspection;
  hasConv3d: boolean;
  conv3dNodeName: string | null;
}

function readTensorDimsCount(buf: Uint8Array, start: number, end: number): number {
  const tensorFields = parseProtoFields(buf, start, end);
  if (!tensorFields) return 0;
  let count = 0;
  for (const tf of tensorFields) {
    if (tf.fn !== 1) continue;
    if (tf.wt === 0) {
      count++;
    } else if (tf.wt === 2) {
      let p = tf.valStart;
      while (p < tf.valEnd) {
        const next = readProtoVarint(buf, p, tf.valEnd);
        if (!next) break;
        count++;
        p = next[1];
      }
    }
  }
  return count;
}

function getInitializerName(buf: Uint8Array, start: number, end: number, decoder: TextDecoder): string | null {
  const tensorFields = parseProtoFields(buf, start, end);
  if (!tensorFields) return null;
  const nameField = tensorFields.find((tf) => tf.fn === 8 && tf.wt === 2);
  if (!nameField || nameField.valEnd <= nameField.valStart) return null;
  return decoder.decode(buf.subarray(nameField.valStart, nameField.valEnd));
}

function readConvKernelShapeCount(buf: Uint8Array, nodeFields: ProtobufFieldSlice[], decoder: TextDecoder): number {
  for (const nf of nodeFields) {
    if (nf.fn !== 5 || nf.wt !== 2) continue;
    const attrFields = parseProtoFields(buf, nf.valStart, nf.valEnd);
    if (!attrFields) continue;
    const nameField = attrFields.find((af) => af.fn === 1 && af.wt === 2);
    if (!nameField) continue;
    const attrName = decoder.decode(buf.subarray(nameField.valStart, nameField.valEnd));
    if (attrName !== 'kernel_shape') continue;
    let count = 0;
    for (const af of attrFields) {
      if (af.fn !== 7) continue;
      if (af.wt === 0) {
        count++;
      } else if (af.wt === 2) {
        let p = af.valStart;
        while (p < af.valEnd) {
          const next = readProtoVarint(buf, p, af.valEnd);
          if (!next) break;
          count++;
          p = next[1];
        }
      }
    }
    return count;
  }
  return 0;
}

export function inspectStereoOnnxWebGpuCompatibility(
  bytes: Uint8Array,
  maxStorageBuffersPerStage = DEFAULT_WEBGPU_MAX_STORAGE_BUFFERS_PER_STAGE
): WebGpuCompatibilityInspection {
  const storageBuffers = exceedsWebGpuStorageBufferLimit(bytes, maxStorageBuffersPerStage);
  if (storageBuffers.exceeds) {
    return {
      compatible: false,
      reason: `Node ${storageBuffers.opType ?? 'unknown'} uses ${storageBuffers.maxNodeBuffers} storage buffers (limit ${maxStorageBuffersPerStage})`,
      storageBuffers,
      hasConv3d: false,
      conv3dNodeName: null,
    };
  }
  if (bytes.byteLength < 16) {
    return {
      compatible: true,
      reason: null,
      storageBuffers,
      hasConv3d: false,
      conv3dNodeName: null,
    };
  }
  try {
    const topFields = parseProtoFields(bytes, 0, bytes.byteLength);
    if (!topFields) {
      return { compatible: true, reason: null, storageBuffers, hasConv3d: false, conv3dNodeName: null };
    }
    const graphField = topFields.find((f) => f.fn === 7 && f.wt === 2);
    if (!graphField) {
      return { compatible: true, reason: null, storageBuffers, hasConv3d: false, conv3dNodeName: null };
    }
    const graphFields = parseProtoFields(bytes, graphField.valStart, graphField.valEnd);
    if (!graphFields) {
      return { compatible: true, reason: null, storageBuffers, hasConv3d: false, conv3dNodeName: null };
    }

    const decoder = new TextDecoder();
    const initializerRanks = new Map<string, number>();
    for (const gf of graphFields) {
      if (gf.fn !== 5 || gf.wt !== 2) continue;
      const initName = getInitializerName(bytes, gf.valStart, gf.valEnd, decoder);
      if (!initName) continue;
      const rank = readTensorDimsCount(bytes, gf.valStart, gf.valEnd);
      if (rank > 0) {
        initializerRanks.set(initName, rank);
      }
    }

    for (const gf of graphFields) {
      if (gf.fn !== 1 || gf.wt !== 2) continue;
      const nodeFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
      if (!nodeFields) continue;
      const opField = nodeFields.find((nf) => nf.fn === 4 && nf.wt === 2 && nf.valEnd > nf.valStart);
      if (!opField) continue;
      const opType = decoder.decode(bytes.subarray(opField.valStart, opField.valEnd));
      if (opType !== 'Conv' && opType !== 'ConvTranspose') continue;

      const inputs = nodeFields
        .filter((nf) => nf.fn === 1 && nf.wt === 2 && nf.valEnd > nf.valStart)
        .map((nf) => decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)));
      const weightRank = inputs[1] ? (initializerRanks.get(inputs[1]) ?? 0) : 0;
      const kernelRank = readConvKernelShapeCount(bytes, nodeFields, decoder);
      if (weightRank > 4 || kernelRank > 2) {
        const nameField = nodeFields.find((nf) => nf.fn === 3 && nf.wt === 2 && nf.valEnd > nf.valStart);
        const nodeName = nameField
          ? decoder.decode(bytes.subarray(nameField.valStart, nameField.valEnd))
          : opType;
        return {
          compatible: false,
          reason: `3D convolution node ${nodeName} (weight rank ${weightRank || 'dynamic'}) is incompatible with WebGPU JSEP kernels`,
          storageBuffers,
          hasConv3d: true,
          conv3dNodeName: nodeName,
        };
      }
    }

    return {
      compatible: true,
      reason: null,
      storageBuffers,
      hasConv3d: false,
      conv3dNodeName: null,
    };
  } catch {
    return { compatible: true, reason: null, storageBuffers, hasConv3d: false, conv3dNodeName: null };
  }
}

export interface OnnxConcatSplitResult {
  bytes: Uint8Array;
  splitCount: number;
  originalMaxConcatInputs: number;
}

/**
 * Splits ONNX `Concat` nodes with more than `maxInputsPerConcat` (default 4) inputs into
 * balanced hierarchical `Concat` subtrees preserving channel order and all attributes (e.g. `axis`).
 * Also removes `GraphProto.value_info` fields so intermediate shape hints never conflict.
 * Reduces per-node WebGPU storage buffer bindings from up to 49 (LightStereo-S) / 38 (CREStereo) down to <= 5.
 */
export function splitLargeOnnxConcatNodes(
  bytes: Uint8Array,
  maxInputsPerConcat = 4
): OnnxConcatSplitResult {
  const safeMax = Math.max(2, Math.floor(maxInputsPerConcat));
  if (bytes.byteLength < 16) {
    return { bytes, splitCount: 0, originalMaxConcatInputs: 0 };
  }
  try {
    const topFields = parseProtoFields(bytes, 0, bytes.byteLength);
    if (!topFields) return { bytes, splitCount: 0, originalMaxConcatInputs: 0 };

    const graphIndex = topFields.findIndex((f) => f.fn === 7 && f.wt === 2);
    if (graphIndex < 0) return { bytes, splitCount: 0, originalMaxConcatInputs: 0 };

    const graphField = topFields[graphIndex];
    const graphFields = parseProtoFields(bytes, graphField.valStart, graphField.valEnd);
    if (!graphFields) return { bytes, splitCount: 0, originalMaxConcatInputs: 0 };

    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    const usedNames = new Set<string>();
    let hasOversizedConcat = false;
    let originalMaxConcatInputs = 0;

    for (const gf of graphFields) {
      if (gf.fn === 5 && gf.wt === 2) {
        const initName = getInitializerName(bytes, gf.valStart, gf.valEnd, decoder);
        if (initName) usedNames.add(initName);
        continue;
      }
      if (gf.fn !== 1 || gf.wt !== 2) continue;
      const nodeFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
      if (!nodeFields) continue;
      let opType = '';
      let inputCount = 0;
      let outputCount = 0;
      for (const nf of nodeFields) {
        if (nf.fn === 1 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          inputCount++;
          usedNames.add(decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)));
        } else if (nf.fn === 2 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          outputCount++;
          usedNames.add(decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)));
        } else if (nf.fn === 4 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          opType = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
        }
      }
      if (opType === 'Concat' && inputCount > safeMax && outputCount === 1) {
        hasOversizedConcat = true;
        if (inputCount > originalMaxConcatInputs) {
          originalMaxConcatInputs = inputCount;
        }
      }
    }

    if (!hasOversizedConcat) {
      return { bytes, splitCount: 0, originalMaxConcatInputs: 0 };
    }

    const encodeProtoField = (fn: number, payload: Uint8Array): Uint8Array => {
      const tag = encodeProtoVarint((fn << 3) | 2);
      const len = encodeProtoVarint(payload.byteLength);
      const out = new Uint8Array(tag.byteLength + len.byteLength + payload.byteLength);
      out.set(tag, 0);
      out.set(len, tag.byteLength);
      out.set(payload, tag.byteLength + len.byteLength);
      return out;
    };
    const encodeProtoStringField = (fn: number, value: string): Uint8Array =>
      encodeProtoField(fn, encoder.encode(value));

    const buildNodeSlice = (
      inputs: readonly string[],
      output: string,
      name: string | null,
      preservedParts: readonly Uint8Array[]
    ): Uint8Array => {
      const parts: Uint8Array[] = [];
      for (const inp of inputs) parts.push(encodeProtoStringField(1, inp));
      parts.push(encodeProtoStringField(2, output));
      if (name) parts.push(encodeProtoStringField(3, name));
      for (const p of preservedParts) parts.push(p);
      let nodeLen = 0;
      for (const p of parts) nodeLen += p.byteLength;
      const nodePayload = new Uint8Array(nodeLen);
      let off = 0;
      for (const p of parts) {
        nodePayload.set(p, off);
        off += p.byteLength;
      }
      return encodeProtoField(1, nodePayload);
    };

    const nextGraphParts: Uint8Array[] = [];
    let splitCount = 0;
    let nodeIdx = 0;
    let tempSeq = 0;

    for (const gf of graphFields) {
      if (gf.fn === 13) continue; // Drop value_info in the same pass
      if (gf.fn !== 1 || gf.wt !== 2) {
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      const currentNodeIdx = nodeIdx++;
      const nodeFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
      if (!nodeFields) {
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      let opType = '';
      let origName: string | null = null;
      const inputs: string[] = [];
      const outputs: string[] = [];
      const preservedParts: Uint8Array[] = [];

      for (const nf of nodeFields) {
        if (nf.fn === 1 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          inputs.push(decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)));
        } else if (nf.fn === 2 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          outputs.push(decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)));
        } else if (nf.fn === 3 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          origName = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
        } else {
          if (nf.fn === 4 && nf.wt === 2 && nf.valEnd > nf.valStart) {
            opType = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
          }
          preservedParts.push(bytes.subarray(nf.fStart, nf.fEnd));
        }
      }

      if (opType !== 'Concat' || inputs.length <= safeMax || outputs.length !== 1) {
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      splitCount++;
      let currentLevel = inputs;
      while (currentLevel.length > safeMax) {
        const nextLevel: string[] = [];
        for (let i = 0; i < currentLevel.length; i += safeMax) {
          const chunk = currentLevel.slice(i, i + safeMax);
          if (chunk.length === 1) {
            nextLevel.push(chunk[0]);
            continue;
          }
          let interName = `__webgpu_concat_${currentNodeIdx}_${tempSeq++}`;
          while (usedNames.has(interName)) {
            interName = `__webgpu_concat_${currentNodeIdx}_${tempSeq++}`;
          }
          usedNames.add(interName);
          const interNodeName = origName ? `${origName}__split_${tempSeq}` : `Concat_split_${currentNodeIdx}_${tempSeq}`;
          nextGraphParts.push(buildNodeSlice(chunk, interName, interNodeName, preservedParts));
          nextLevel.push(interName);
        }
        currentLevel = nextLevel;
      }
      nextGraphParts.push(buildNodeSlice(currentLevel, outputs[0], origName, preservedParts));
    }

    let graphPayloadLen = 0;
    for (const part of nextGraphParts) graphPayloadLen += part.byteLength;

    const graphTag = encodeProtoVarint((7 << 3) | 2);
    const graphLen = encodeProtoVarint(graphPayloadLen);
    let totalLen = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        totalLen += graphTag.byteLength + graphLen.byteLength + graphPayloadLen;
      } else {
        totalLen += topFields[i].fEnd - topFields[i].fStart;
      }
    }

    const out = new Uint8Array(totalLen);
    let offset = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        out.set(graphTag, offset);
        offset += graphTag.byteLength;
        out.set(graphLen, offset);
        offset += graphLen.byteLength;
        for (const part of nextGraphParts) {
          out.set(part, offset);
          offset += part.byteLength;
        }
      } else {
        const chunk = bytes.subarray(topFields[i].fStart, topFields[i].fEnd);
        out.set(chunk, offset);
        offset += chunk.byteLength;
      }
    }
    return { bytes: out, splitCount, originalMaxConcatInputs };
  } catch {
    return { bytes, splitCount: 0, originalMaxConcatInputs: 0 };
  }
}

export interface OnnxConv3dDecomposeResult {
  bytes: Uint8Array;
  decomposedCount: number;
  decomposedNodeNames: string[];
}

interface Parsed5dInitializer {
  name: string;
  dims: number[];
  dataType: number;
  floatData: Float32Array;
  fieldSlice: ProtobufFieldSlice;
}

function readTensorDimsArray(buf: Uint8Array, start: number, end: number): number[] {
  const tensorFields = parseProtoFields(buf, start, end);
  if (!tensorFields) return [];
  const dims: number[] = [];
  for (const tf of tensorFields) {
    if (tf.fn !== 1) continue;
    if (tf.wt === 0) {
      const v = readProtoVarint(buf, tf.valStart, tf.valEnd);
      if (v) dims.push(v[0]);
    } else if (tf.wt === 2) {
      let p = tf.valStart;
      while (p < tf.valEnd) {
        const next = readProtoVarint(buf, p, tf.valEnd);
        if (!next) break;
        dims.push(next[0]);
        p = next[1];
      }
    }
  }
  return dims;
}

function parse5dFloatInitializer(
  buf: Uint8Array,
  slice: ProtobufFieldSlice,
  decoder: TextDecoder
): Parsed5dInitializer | null {
  const tensorFields = parseProtoFields(buf, slice.valStart, slice.valEnd);
  if (!tensorFields) return null;

  const dims: number[] = [];
  let dataType = 1; // FLOAT = 1
  let name = '';
  const floatValues: number[] = [];
  let rawBytes: Uint8Array | null = null;
  let hasExternalLocation = false;

  for (const tf of tensorFields) {
    if (tf.fn === 1) {
      if (tf.wt === 0) {
        const v = readProtoVarint(buf, tf.valStart, tf.valEnd);
        if (v) dims.push(v[0]);
      } else if (tf.wt === 2) {
        let p = tf.valStart;
        while (p < tf.valEnd) {
          const next = readProtoVarint(buf, p, tf.valEnd);
          if (!next) break;
          dims.push(next[0]);
          p = next[1];
        }
      }
    } else if (tf.fn === 2 && tf.wt === 0) {
      const v = readProtoVarint(buf, tf.valStart, tf.valEnd);
      if (v) dataType = v[0];
    } else if (tf.fn === 4) {
      if (tf.wt === 5 && tf.valEnd - tf.valStart === 4) {
        const dv = new DataView(buf.buffer, buf.byteOffset + tf.valStart, 4);
        floatValues.push(dv.getFloat32(0, true));
      } else if (tf.wt === 2) {
        const byteLen = tf.valEnd - tf.valStart;
        const dv = new DataView(buf.buffer, buf.byteOffset + tf.valStart, byteLen);
        for (let off = 0; off + 4 <= byteLen; off += 4) {
          floatValues.push(dv.getFloat32(off, true));
        }
      }
    } else if (tf.fn === 8 && tf.wt === 2 && tf.valEnd > tf.valStart) {
      name = decoder.decode(buf.subarray(tf.valStart, tf.valEnd));
    } else if (tf.fn === 9 && tf.wt === 2) {
      rawBytes = buf.subarray(tf.valStart, tf.valEnd);
    } else if (tf.fn === 13 && tf.wt === 0) {
      const v = readProtoVarint(buf, tf.valStart, tf.valEnd);
      if (v && v[0] !== 0) hasExternalLocation = true;
    }
  }

  if (!name || dims.length !== 5 || dataType !== 1 || hasExternalLocation) {
    return null;
  }

  let totalElements = 1;
  for (const d of dims) {
    if (d <= 0) return null;
    totalElements *= d;
  }

  let floatData: Float32Array;
  if (rawBytes && rawBytes.byteLength === totalElements * 4) {
    floatData = new Float32Array(totalElements);
    const dv = new DataView(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength);
    for (let i = 0; i < totalElements; i++) {
      floatData[i] = dv.getFloat32(i * 4, true);
    }
  } else if (floatValues.length === totalElements) {
    floatData = new Float32Array(floatValues);
  } else if (!rawBytes && floatValues.length === 0) {
    // Support synthetic minimal test protos where raw_data is omitted
    floatData = new Float32Array(totalElements);
  } else {
    return null;
  }

  return {
    name,
    dims,
    dataType,
    floatData,
    fieldSlice: slice,
  };
}

function readNodeIntsAttribute(
  buf: Uint8Array,
  nodeFields: ProtobufFieldSlice[],
  targetAttrName: string,
  decoder: TextDecoder
): number[] | null {
  for (const nf of nodeFields) {
    if (nf.fn !== 5 || nf.wt !== 2) continue;
    const attrFields = parseProtoFields(buf, nf.valStart, nf.valEnd);
    if (!attrFields) continue;
    const nameField = attrFields.find((af) => af.fn === 1 && af.wt === 2);
    if (!nameField) continue;
    const attrName = decoder.decode(buf.subarray(nameField.valStart, nameField.valEnd));
    if (attrName !== targetAttrName) continue;
    const values: number[] = [];
    for (const af of attrFields) {
      if (af.fn !== 8) continue;
      if (af.wt === 0) {
        const v = readProtoVarint(buf, af.valStart, af.valEnd);
        if (v) values.push(v[0]);
      } else if (af.wt === 2) {
        let p = af.valStart;
        while (p < af.valEnd) {
          const next = readProtoVarint(buf, p, af.valEnd);
          if (!next) break;
          values.push(next[0]);
          p = next[1];
        }
      }
    }
    return values;
  }
  return null;
}

function readNodeIntAttribute(
  buf: Uint8Array,
  nodeFields: ProtobufFieldSlice[],
  targetAttrName: string,
  decoder: TextDecoder
): number | null {
  for (const nf of nodeFields) {
    if (nf.fn !== 5 || nf.wt !== 2) continue;
    const attrFields = parseProtoFields(buf, nf.valStart, nf.valEnd);
    if (!attrFields) continue;
    const nameField = attrFields.find((af) => af.fn === 1 && af.wt === 2);
    if (!nameField) continue;
    const attrName = decoder.decode(buf.subarray(nameField.valStart, nameField.valEnd));
    if (attrName !== targetAttrName) continue;
    for (const af of attrFields) {
      if (af.fn === 3 && af.wt === 0) {
        const v = readProtoVarint(buf, af.valStart, af.valEnd);
        if (v) return v[0];
      }
    }
  }
  return null;
}

function readNodeStringAttribute(
  buf: Uint8Array,
  nodeFields: ProtobufFieldSlice[],
  targetAttrName: string,
  decoder: TextDecoder
): string | null {
  for (const nf of nodeFields) {
    if (nf.fn !== 5 || nf.wt !== 2) continue;
    const attrFields = parseProtoFields(buf, nf.valStart, nf.valEnd);
    if (!attrFields) continue;
    const nameField = attrFields.find((af) => af.fn === 1 && af.wt === 2);
    if (!nameField) continue;
    const attrName = decoder.decode(buf.subarray(nameField.valStart, nameField.valEnd));
    if (attrName !== targetAttrName) continue;
    for (const af of attrFields) {
      if (af.fn === 4 && af.wt === 2) {
        return decoder.decode(buf.subarray(af.valStart, af.valEnd));
      }
    }
  }
  return null;
}

export interface OnnxGraphValidationResult {
  valid: boolean;
  missingInputs: Array<{ nodeName: string; opType: string; inputName: string }>;
  nodeCount: number;
  hasGraphInputs: boolean;
}

function getValueInfoName(
  buf: Uint8Array,
  start: number,
  end: number,
  decoder: TextDecoder
): string | null {
  const fields = parseProtoFields(buf, start, end);
  if (!fields) return null;
  for (const f of fields) {
    if (f.fn === 1 && f.wt === 2 && f.valEnd > f.valStart) {
      return decoder.decode(buf.subarray(f.valStart, f.valEnd));
    }
  }
  return null;
}

function readOnnxDefaultOpsetVersion(
  buf: Uint8Array,
  topFields: ProtobufFieldSlice[],
  decoder: TextDecoder
): number | null {
  let defaultVersion: number | null = null;
  for (const f of topFields) {
    if (f.fn !== 8 || f.wt !== 2) continue;
    const opsetFields = parseProtoFields(buf, f.valStart, f.valEnd);
    if (!opsetFields) continue;
    let domain = '';
    let version: number | null = null;
    for (const ofield of opsetFields) {
      if (ofield.fn === 1 && ofield.wt === 2) {
        domain = decoder.decode(buf.subarray(ofield.valStart, ofield.valEnd));
      } else if (ofield.fn === 2 && ofield.wt === 0) {
        const v = readProtoVarint(buf, ofield.valStart, ofield.valEnd);
        if (v) version = v[0];
      }
    }
    if ((domain === '' || domain === 'ai.onnx') && version !== null) {
      defaultVersion = version;
    }
  }
  return defaultVersion;
}

/**
 * Validates that every non-empty node input in an ONNX `GraphProto` is defined by a graph input,
 * initializer, or output of a preceding node in topological order (matching ORT `ERROR_CODE: 2`
 * graph validation semantics).
 */
export function validateOnnxGraphNodeInputs(bytes: Uint8Array): OnnxGraphValidationResult {
  if (bytes.byteLength < 8) {
    return { valid: false, missingInputs: [], nodeCount: 0, hasGraphInputs: false };
  }
  const topFields = parseProtoFields(bytes, 0, bytes.byteLength);
  if (!topFields) {
    return { valid: false, missingInputs: [], nodeCount: 0, hasGraphInputs: false };
  }
  const graphField = topFields.find((f) => f.fn === 7 && f.wt === 2);
  if (!graphField) {
    return { valid: false, missingInputs: [], nodeCount: 0, hasGraphInputs: false };
  }
  const graphFields = parseProtoFields(bytes, graphField.valStart, graphField.valEnd);
  if (!graphFields) {
    return { valid: false, missingInputs: [], nodeCount: 0, hasGraphInputs: false };
  }

  const decoder = new TextDecoder();
  const defined = new Set<string>();
  let hasGraphInputs = false;

  for (const gf of graphFields) {
    if (gf.fn === 11 && gf.wt === 2) {
      const inputName = getValueInfoName(bytes, gf.valStart, gf.valEnd, decoder);
      if (inputName) {
        defined.add(inputName);
        hasGraphInputs = true;
      }
    } else if (gf.fn === 5 && gf.wt === 2) {
      const initName = getInitializerName(bytes, gf.valStart, gf.valEnd, decoder);
      if (initName) {
        defined.add(initName);
      }
    } else if (gf.fn === 15 && gf.wt === 2) {
      const sparseFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
      if (sparseFields) {
        for (const sf of sparseFields) {
          if ((sf.fn === 1 || sf.fn === 2) && sf.wt === 2) {
            const sparseName = getInitializerName(bytes, sf.valStart, sf.valEnd, decoder);
            if (sparseName) defined.add(sparseName);
          }
        }
      }
    }
  }

  const missingInputs: Array<{ nodeName: string; opType: string; inputName: string }> = [];
  let nodeCount = 0;

  for (const gf of graphFields) {
    if (gf.fn !== 1 || gf.wt !== 2) continue;
    const nodeFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
    if (!nodeFields) continue;
    nodeCount += 1;

    const opField = nodeFields.find((nf) => nf.fn === 4 && nf.wt === 2 && nf.valEnd > nf.valStart);
    const opType = opField ? decoder.decode(bytes.subarray(opField.valStart, opField.valEnd)) : '';
    const nameField = nodeFields.find(
      (nf) => nf.fn === 3 && nf.wt === 2 && nf.valEnd > nf.valStart
    );
    const nodeName = nameField
      ? decoder.decode(bytes.subarray(nameField.valStart, nameField.valEnd))
      : `node_${nodeCount - 1}`;

    for (const nf of nodeFields) {
      if (nf.fn === 1 && nf.wt === 2 && nf.valEnd > nf.valStart) {
        const inputName = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
        if (!defined.has(inputName)) {
          missingInputs.push({ nodeName, opType, inputName });
        }
      }
    }
    for (const nf of nodeFields) {
      if (nf.fn === 2 && nf.wt === 2 && nf.valEnd > nf.valStart) {
        defined.add(decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)));
      }
    }
  }

  return {
    valid: missingInputs.length === 0,
    missingInputs,
    nodeCount,
    hasGraphInputs,
  };
}

/**
 * Rewrites 5D `Conv` (`Conv3D`) nodes in an ONNX `GraphProto` into mathematically equivalent
 * WebGPU-compatible 4D `Conv` (`Conv2D`) + `Transpose` / `Reshape` / `Slice` / `Add` subgraphs.
 *
 * Specifically:
 * - For `kD = 1, sD = 1, pD0 = 0, pD1 = 0`:
 *   Folds the depth dimension `D` into the batch dimension `[N * D, C_in, H, W]`, applies a single
 *   4D `Conv` with squeezed weights `[M, C_in/group, kH, kW]`, and restores `[N, M, D, H_out, W_out]`.
 * - For general `kD >= 1`:
 *   Decomposes the 3D convolution along the kernel depth axis `m = 0 .. kD - 1` into `kD` 2D convolutions
 *   over depth-sliced windows `[N * D_out, C_in, H, W]` (with optional zero-padding along `D`) and sums
 *   their outputs via `Add` before restoring `[N, M, D_out, H_out, W_out]`.
 */
export function decomposeOnnx5dConvNodesForWebGpu(bytes: Uint8Array): OnnxConv3dDecomposeResult {
  if (bytes.byteLength < 16) {
    return { bytes, decomposedCount: 0, decomposedNodeNames: [] };
  }
  try {
    const topFields = parseProtoFields(bytes, 0, bytes.byteLength);
    if (!topFields) return { bytes, decomposedCount: 0, decomposedNodeNames: [] };

    const decoder = new TextDecoder();
    const defaultOpset = readOnnxDefaultOpsetVersion(bytes, topFields, decoder);
    // 3-input Pad and 5-input Slice require ONNX opset >= 11
    if (defaultOpset !== null && defaultOpset < 11) {
      return { bytes, decomposedCount: 0, decomposedNodeNames: [] };
    }

    const graphIndex = topFields.findIndex((f) => f.fn === 7 && f.wt === 2);
    if (graphIndex < 0) return { bytes, decomposedCount: 0, decomposedNodeNames: [] };

    const graphField = topFields[graphIndex];
    const graphFields = parseProtoFields(bytes, graphField.valStart, graphField.valEnd);
    if (!graphFields) return { bytes, decomposedCount: 0, decomposedNodeNames: [] };
    const encoder = new TextEncoder();
    const usedNames = new Set<string>();
    const initializers5d = new Map<string, Parsed5dInitializer>();
    const initializerUseCounts = new Map<string, number>();

    for (const gf of graphFields) {
      if (gf.fn === 5 && gf.wt === 2) {
        const initName = getInitializerName(bytes, gf.valStart, gf.valEnd, decoder);
        if (initName) {
          usedNames.add(initName);
          const dims = readTensorDimsArray(bytes, gf.valStart, gf.valEnd);
          if (dims.length === 5) {
            const parsed = parse5dFloatInitializer(bytes, gf, decoder);
            if (parsed) {
              initializers5d.set(initName, parsed);
            }
          }
        }
        continue;
      }
      if (gf.fn !== 1 || gf.wt !== 2) continue;
      const nodeFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
      if (!nodeFields) continue;
      for (const nf of nodeFields) {
        if (nf.fn === 1 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          const inpName = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
          usedNames.add(inpName);
          initializerUseCounts.set(inpName, (initializerUseCounts.get(inpName) ?? 0) + 1);
        } else if (nf.fn === 2 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          usedNames.add(decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)));
        } else if (nf.fn === 3 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          usedNames.add(decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)));
        }
      }
    }

    if (initializers5d.size === 0) {
      return { bytes, decomposedCount: 0, decomposedNodeNames: [] };
    }

    let tempSeq = 0;
    const allocName = (prefix: string): string => {
      let candidate = `${prefix}_${tempSeq++}`;
      while (usedNames.has(candidate)) {
        candidate = `${prefix}_${tempSeq++}`;
      }
      usedNames.add(candidate);
      return candidate;
    };

    const encodeLenField = (fn: number, payload: Uint8Array): Uint8Array => {
      const tag = encodeProtoVarint((fn << 3) | 2);
      const len = encodeProtoVarint(payload.byteLength);
      const out = new Uint8Array(tag.byteLength + len.byteLength + payload.byteLength);
      out.set(tag, 0);
      out.set(len, tag.byteLength);
      out.set(payload, tag.byteLength + len.byteLength);
      return out;
    };

    const encodeVarintField = (fn: number, val: number): Uint8Array => {
      const tag = encodeProtoVarint((fn << 3) | 0);
      const vBytes =
        val >= 0
          ? encodeProtoVarint(val)
          : (() => {
              // Signed 64-bit two's complement varint for negative int64 (e.g. -1)
              let u = BigInt.asUintN(64, BigInt(val));
              const arr: number[] = [];
              while (u >= 0x80n) {
                arr.push(Number(u & 0x7fn) | 0x80);
                u >>= 7n;
              }
              arr.push(Number(u));
              return new Uint8Array(arr);
            })();
      const out = new Uint8Array(tag.byteLength + vBytes.byteLength);
      out.set(tag, 0);
      out.set(vBytes, tag.byteLength);
      return out;
    };

    const encodeStrField = (fn: number, str: string): Uint8Array =>
      encodeLenField(fn, encoder.encode(str));

    const concatBytes = (parts: readonly Uint8Array[]): Uint8Array => {
      let total = 0;
      for (const p of parts) total += p.byteLength;
      const out = new Uint8Array(total);
      let off = 0;
      for (const p of parts) {
        out.set(p, off);
        off += p.byteLength;
      }
      return out;
    };

    // AttributeProto builders: field 1=name, field 3=i, field 4=s, field 8=ints, field 20=type (INT=2, STRING=3, INTS=7)
    const encodeAttrInt = (name: string, val: number): Uint8Array =>
      encodeLenField(
        5,
        concatBytes([encodeStrField(1, name), encodeVarintField(3, val), encodeVarintField(20, 2)])
      );

    const encodeAttrInts = (name: string, vals: readonly number[]): Uint8Array =>
      encodeLenField(
        5,
        concatBytes([
          encodeStrField(1, name),
          ...vals.map((v) => encodeVarintField(8, v)),
          encodeVarintField(20, 7),
        ])
      );

    const encodeAttrString = (name: string, val: string): Uint8Array =>
      encodeLenField(
        5,
        concatBytes([encodeStrField(1, name), encodeStrField(4, val), encodeVarintField(20, 3)])
      );

    const encodeNode = (
      opType: string,
      inputs: readonly string[],
      outputs: readonly string[],
      name: string,
      attrs: readonly Uint8Array[] = []
    ): Uint8Array =>
      encodeLenField(
        1,
        concatBytes([
          ...inputs.map((inp) => encodeStrField(1, inp)),
          ...outputs.map((out) => encodeStrField(2, out)),
          encodeStrField(3, name),
          encodeStrField(4, opType),
          ...attrs,
        ])
      );

    // TensorProto builders: field 1=dims, field 2=data_type (1=FLOAT, 7=INT64), field 8=name, field 9=raw_data
    const encodeInt64Initializer = (
      name: string,
      dims: readonly number[],
      values: readonly number[]
    ): Uint8Array => {
      const raw = new Uint8Array(values.length * 8);
      const dv = new DataView(raw.buffer);
      for (let i = 0; i < values.length; i++) {
        dv.setBigInt64(i * 8, BigInt(values[i]), true);
      }
      return encodeLenField(
        5,
        concatBytes([
          ...dims.map((d) => encodeVarintField(1, d)),
          encodeVarintField(2, 7), // INT64 = 7
          encodeStrField(8, name),
          encodeLenField(9, raw),
        ])
      );
    };

    const encodeFloat32Initializer = (
      name: string,
      dims: readonly number[],
      values: Float32Array
    ): Uint8Array => {
      const raw = new Uint8Array(values.length * 4);
      const dv = new DataView(raw.buffer);
      for (let i = 0; i < values.length; i++) {
        dv.setFloat32(i * 4, values[i], true);
      }
      return encodeLenField(
        5,
        concatBytes([
          ...dims.map((d) => encodeVarintField(1, d)),
          encodeVarintField(2, 1), // FLOAT = 1
          encodeStrField(8, name),
          encodeLenField(9, raw),
        ])
      );
    };

    const nextGraphParts: Uint8Array[] = [];
    const addedInitializers: Uint8Array[] = [];
    const replacedWeightCounts = new Map<string, number>();
    const reused4dWeights = new Map<string, string>();
    const decomposedNodeNames: string[] = [];

    // Shared helper constants emitted lazily if at least one 5D Conv/ConvTranspose is decomposed
    let sharedConstReady = false;
    let cIdx0 = '';
    let cIdx1 = '';
    let cIdx2 = '';
    let cIdx3 = '';
    let cIdx4 = '';
    let cStep1 = '';
    let cInt64Max = '';
    let cZeroFloat = '';

    const ensureSharedConstants = () => {
      if (sharedConstReady) return;
      sharedConstReady = true;
      cIdx0 = allocName('__c3d_idx0');
      cIdx1 = allocName('__c3d_idx1');
      cIdx2 = allocName('__c3d_idx2');
      cIdx3 = allocName('__c3d_idx3');
      cIdx4 = allocName('__c3d_idx4');
      cStep1 = allocName('__c3d_step1');
      cInt64Max = allocName('__c3d_int64_max');
      cZeroFloat = allocName('__c3d_zero_f32');
      addedInitializers.push(
        encodeInt64Initializer(cIdx0, [1], [0]),
        encodeInt64Initializer(cIdx1, [1], [1]),
        encodeInt64Initializer(cIdx2, [1], [2]),
        encodeInt64Initializer(cIdx3, [1], [3]),
        encodeInt64Initializer(cIdx4, [1], [4]),
        encodeInt64Initializer(cStep1, [1], [1]),
        encodeInt64Initializer(cInt64Max, [1], [2147483647]),
        encodeFloat32Initializer(cZeroFloat, [], new Float32Array([0]))
      );
    };

    for (const gf of graphFields) {
      if (gf.fn === 13) continue; // Strip value_info so 4D intermediate tensors never conflict with 5D hints
      if (gf.fn !== 1 || gf.wt !== 2) {
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      const nodeFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
      if (!nodeFields) {
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      const opField = nodeFields.find((nf) => nf.fn === 4 && nf.wt === 2 && nf.valEnd > nf.valStart);
      const opType = opField ? decoder.decode(bytes.subarray(opField.valStart, opField.valEnd)) : '';

      if (opType === 'ConvTranspose') {
        const inputs = nodeFields
          .filter((nf) => nf.fn === 1 && nf.wt === 2 && nf.valEnd > nf.valStart)
          .map((nf) => decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)));
        const outputs = nodeFields
          .filter((nf) => nf.fn === 2 && nf.wt === 2 && nf.valEnd > nf.valStart)
          .map((nf) => decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)));
        const nameField = nodeFields.find(
          (nf) => nf.fn === 3 && nf.wt === 2 && nf.valEnd > nf.valStart
        );
        const origName = nameField
          ? decoder.decode(bytes.subarray(nameField.valStart, nameField.valEnd))
          : allocName('ConvTranspose3D');

        const weightName = inputs[1] ?? '';
        const weightInit = initializers5d.get(weightName);
        if (!weightInit || inputs.length < 2 || inputs.length > 3 || outputs.length !== 1) {
          nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
          continue;
        }

        const [cIn, mPerGroup, k0, k1, k2] = weightInit.dims;
        const autoPadAttr = readNodeStringAttribute(bytes, nodeFields, 'auto_pad', decoder);
        const outputShapeAttr = readNodeIntsAttribute(bytes, nodeFields, 'output_shape', decoder);
        const kernelShapeAttr = readNodeIntsAttribute(bytes, nodeFields, 'kernel_shape', decoder);
        const stridesAttr = readNodeIntsAttribute(bytes, nodeFields, 'strides', decoder);
        const padsAttr = readNodeIntsAttribute(bytes, nodeFields, 'pads', decoder);
        const dilationsAttr = readNodeIntsAttribute(bytes, nodeFields, 'dilations', decoder);
        const outputPaddingAttr = readNodeIntsAttribute(
          bytes,
          nodeFields,
          'output_padding',
          decoder
        );
        const groupAttr = readNodeIntAttribute(bytes, nodeFields, 'group', decoder) ?? 1;

        if (autoPadAttr !== null && autoPadAttr !== 'NOTSET') {
          nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
          continue;
        }
        if (outputShapeAttr !== null) {
          nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
          continue;
        }
        if (
          kernelShapeAttr !== null &&
          (kernelShapeAttr.length !== 3 ||
            kernelShapeAttr[0] !== k0 ||
            kernelShapeAttr[1] !== k1 ||
            kernelShapeAttr[2] !== k2)
        ) {
          nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
          continue;
        }
        if (stridesAttr !== null && stridesAttr.length !== 3) {
          nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
          continue;
        }
        if (dilationsAttr !== null && dilationsAttr.length !== 3) {
          nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
          continue;
        }
        if (padsAttr !== null && padsAttr.length !== 6) {
          nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
          continue;
        }
        if (outputPaddingAttr !== null && outputPaddingAttr.length !== 3) {
          nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
          continue;
        }

        const [s0, s1, s2] = stridesAttr ?? [1, 1, 1];
        const [d0, d1, d2] = dilationsAttr ?? [1, 1, 1];
        const [p0a, p1a, p2a, p0b, p1b, p2b] = padsAttr ?? [0, 0, 0, 0, 0, 0];
        const [op0, op1, op2] = outputPaddingAttr ?? [0, 0, 0];

        if (
          groupAttr < 1 ||
          s0 < 1 ||
          s1 < 1 ||
          s2 < 1 ||
          d0 < 1 ||
          d1 < 1 ||
          d2 < 1 ||
          p0a < 0 ||
          p1a < 0 ||
          p2a < 0 ||
          p0b < 0 ||
          p1b < 0 ||
          p2b < 0 ||
          op0 < 0 ||
          op1 < 0 ||
          op2 < 0
        ) {
          nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
          continue;
        }

        // Only fold trailing spatial axis into batch when it is strictly unit/no-op
        if (k2 !== 1 || s2 !== 1 || d2 !== 1 || p2a !== 0 || p2b !== 0 || op2 !== 0) {
          nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
          continue;
        }

        ensureSharedConstants();
        decomposedNodeNames.push(origName);
        replacedWeightCounts.set(weightName, (replacedWeightCounts.get(weightName) ?? 0) + 1);

        const xInput = inputs[0];
        const biasInput = inputs[2] ?? null;
        const yOutput = outputs[0];

        const weightKey = `ct:${weightName}`;
        let w4dName = reused4dWeights.get(weightKey);
        if (!w4dName) {
          w4dName = allocName(`${weightName}__4d`);
          addedInitializers.push(
            encodeFloat32Initializer(w4dName, [cIn, mPerGroup, k0, k1], weightInit.floatData)
          );
          reused4dWeights.set(weightKey, w4dName);
        }

        const convT2dAttrs: Uint8Array[] = [
          encodeAttrInts('kernel_shape', [k0, k1]),
          encodeAttrInts('strides', [s0, s1]),
          encodeAttrInts('dilations', [d0, d1]),
          encodeAttrInts('pads', [p0a, p1a, p0b, p1b]),
        ];
        if (groupAttr > 1) {
          convT2dAttrs.push(encodeAttrInt('group', groupAttr));
        }
        if (autoPadAttr === 'NOTSET') {
          convT2dAttrs.push(encodeAttrString('auto_pad', 'NOTSET'));
        }
        if (outputPaddingAttr !== null) {
          convT2dAttrs.push(encodeAttrInts('output_padding', [op0, op1]));
        }

        const xShape = allocName(`${origName}__x_shape`);
        const dimN = allocName(`${origName}__dim_N`);
        const dimC = allocName(`${origName}__dim_C`);
        const dimD0 = allocName(`${origName}__dim_D0`);
        const dimD1 = allocName(`${origName}__dim_D1`);
        const dimD2 = allocName(`${origName}__dim_D2`);
        const xTrans = allocName(`${origName}__x_nd2chw`);
        const dimND2 = allocName(`${origName}__dim_ND2`);
        const x4dShape = allocName(`${origName}__x4d_shape`);
        const x4d = allocName(`${origName}__x_4d`);
        const y4d = allocName(`${origName}__y_4d`);
        const y4dShape = allocName(`${origName}__y4d_shape`);
        const dimCout = allocName(`${origName}__dim_Cout`);
        const dimOut0 = allocName(`${origName}__dim_Out0`);
        const dimOut1 = allocName(`${origName}__dim_Out1`);
        const y5dShape = allocName(`${origName}__y5d_shape`);
        const y5d = allocName(`${origName}__y_nd2mhw`);

        const convTInputs = biasInput ? [x4d, w4dName, biasInput] : [x4d, w4dName];

        nextGraphParts.push(
          encodeNode('Shape', [xInput], [xShape], allocName(`${origName}__ShapeX`)),
          encodeNode('Gather', [xShape, cIdx0], [dimN], allocName(`${origName}__GatherN`), [
            encodeAttrInt('axis', 0),
          ]),
          encodeNode('Gather', [xShape, cIdx1], [dimC], allocName(`${origName}__GatherC`), [
            encodeAttrInt('axis', 0),
          ]),
          encodeNode('Gather', [xShape, cIdx2], [dimD0], allocName(`${origName}__GatherD0`), [
            encodeAttrInt('axis', 0),
          ]),
          encodeNode('Gather', [xShape, cIdx3], [dimD1], allocName(`${origName}__GatherD1`), [
            encodeAttrInt('axis', 0),
          ]),
          encodeNode('Gather', [xShape, cIdx4], [dimD2], allocName(`${origName}__GatherD2`), [
            encodeAttrInt('axis', 0),
          ]),
          encodeNode('Transpose', [xInput], [xTrans], allocName(`${origName}__TransposeIn`), [
            encodeAttrInts('perm', [0, 4, 1, 2, 3]),
          ]),
          encodeNode('Mul', [dimN, dimD2], [dimND2], allocName(`${origName}__MulND2`)),
          encodeNode(
            'Concat',
            [dimND2, dimC, dimD0, dimD1],
            [x4dShape],
            allocName(`${origName}__ConcatX4dShape`),
            [encodeAttrInt('axis', 0)]
          ),
          encodeNode('Reshape', [xTrans, x4dShape], [x4d], allocName(`${origName}__ReshapeIn`)),
          encodeNode(
            'ConvTranspose',
            convTInputs,
            [y4d],
            allocName(`${origName}__ConvTranspose2D`),
            convT2dAttrs
          ),
          encodeNode('Shape', [y4d], [y4dShape], allocName(`${origName}__ShapeY4d`)),
          encodeNode('Gather', [y4dShape, cIdx1], [dimCout], allocName(`${origName}__GatherCout`), [
            encodeAttrInt('axis', 0),
          ]),
          encodeNode('Gather', [y4dShape, cIdx2], [dimOut0], allocName(`${origName}__GatherOut0`), [
            encodeAttrInt('axis', 0),
          ]),
          encodeNode('Gather', [y4dShape, cIdx3], [dimOut1], allocName(`${origName}__GatherOut1`), [
            encodeAttrInt('axis', 0),
          ]),
          encodeNode(
            'Concat',
            [dimN, dimD2, dimCout, dimOut0, dimOut1],
            [y5dShape],
            allocName(`${origName}__ConcatY5dShape`),
            [encodeAttrInt('axis', 0)]
          ),
          encodeNode('Reshape', [y4d, y5dShape], [y5d], allocName(`${origName}__ReshapeOut`)),
          encodeNode('Transpose', [y5d], [yOutput], origName, [
            encodeAttrInts('perm', [0, 2, 3, 4, 1]),
          ])
        );
        continue;
      }

      if (opType !== 'Conv') {
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      const inputs = nodeFields
        .filter((nf) => nf.fn === 1 && nf.wt === 2 && nf.valEnd > nf.valStart)
        .map((nf) => decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)));
      const outputs = nodeFields
        .filter((nf) => nf.fn === 2 && nf.wt === 2 && nf.valEnd > nf.valStart)
        .map((nf) => decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)));
      const nameField = nodeFields.find((nf) => nf.fn === 3 && nf.wt === 2 && nf.valEnd > nf.valStart);
      const origName = nameField
        ? decoder.decode(bytes.subarray(nameField.valStart, nameField.valEnd))
        : allocName('Conv3D');

      const weightName = inputs[1] ?? '';
      const weightInit = initializers5d.get(weightName);
      if (!weightInit || inputs.length < 2 || outputs.length !== 1) {
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      const [outC, inCPerGroup, kD, kH, kW] = weightInit.dims;
      const stridesAttr = readNodeIntsAttribute(bytes, nodeFields, 'strides', decoder);
      const padsAttr = readNodeIntsAttribute(bytes, nodeFields, 'pads', decoder);
      const dilationsAttr = readNodeIntsAttribute(bytes, nodeFields, 'dilations', decoder);
      const groupAttr = readNodeIntAttribute(bytes, nodeFields, 'group', decoder) ?? 1;
      const autoPadAttr = readNodeStringAttribute(bytes, nodeFields, 'auto_pad', decoder);

      const sD = stridesAttr && stridesAttr.length === 3 ? stridesAttr[0] : 1;
      const sH = stridesAttr && stridesAttr.length === 3 ? stridesAttr[1] : 1;
      const sW = stridesAttr && stridesAttr.length === 3 ? stridesAttr[2] : 1;

      const dD = dilationsAttr && dilationsAttr.length === 3 ? dilationsAttr[0] : 1;
      const dH = dilationsAttr && dilationsAttr.length === 3 ? dilationsAttr[1] : 1;
      const dW = dilationsAttr && dilationsAttr.length === 3 ? dilationsAttr[2] : 1;

      let pD0 = padsAttr && padsAttr.length === 6 ? padsAttr[0] : 0;
      const pH0 = padsAttr && padsAttr.length === 6 ? padsAttr[1] : 0;
      const pW0 = padsAttr && padsAttr.length === 6 ? padsAttr[2] : 0;
      let pD1 = padsAttr && padsAttr.length === 6 ? padsAttr[3] : 0;
      const pH1 = padsAttr && padsAttr.length === 6 ? padsAttr[4] : 0;
      const pW1 = padsAttr && padsAttr.length === 6 ? padsAttr[5] : 0;

      if (autoPadAttr === 'SAME_UPPER' || autoPadAttr === 'SAME_LOWER') {
        if (sD > 1) {
          nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
          continue;
        }
        const totalPadD = Math.max(0, (kD - 1) * dD);
        pD0 = autoPadAttr === 'SAME_UPPER' ? Math.floor(totalPadD / 2) : Math.ceil(totalPadD / 2);
        pD1 = totalPadD - pD0;
      }

      // Limit decomposition to manageable kernel depths so graph size stays tiny
      if (kD < 1 || kD > 32 || sD < 1) {
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      ensureSharedConstants();
      decomposedNodeNames.push(origName);
      replacedWeightCounts.set(weightName, (replacedWeightCounts.get(weightName) ?? 0) + 1);

      const xInput = inputs[0];
      const biasInput = inputs[2] ?? null;
      const yOutput = outputs[0];

      const conv2dAttrs: Uint8Array[] = [
        encodeAttrInts('kernel_shape', [kH, kW]),
        encodeAttrInts('strides', [sH, sW]),
        encodeAttrInts('dilations', [dH, dW]),
      ];
      if (groupAttr > 1) {
        conv2dAttrs.push(encodeAttrInt('group', groupAttr));
      }
      if (autoPadAttr && autoPadAttr !== 'NOTSET') {
        conv2dAttrs.push(encodeAttrString('auto_pad', autoPadAttr));
      } else {
        conv2dAttrs.push(encodeAttrInts('pads', [pH0, pW0, pH1, pW1]));
      }

      // Extract input shape components [N, C_in, D_in, H_in, W_in]
      const xShape = allocName(`${origName}__x_shape`);
      const dimN = allocName(`${origName}__dim_N`);
      const dimC = allocName(`${origName}__dim_C`);
      const dimD = allocName(`${origName}__dim_D`);
      nextGraphParts.push(
        encodeNode('Shape', [xInput], [xShape], allocName(`${origName}__ShapeX`)),
        encodeNode('Gather', [xShape, cIdx0], [dimN], allocName(`${origName}__GatherN`), [
          encodeAttrInt('axis', 0),
        ]),
        encodeNode('Gather', [xShape, cIdx1], [dimC], allocName(`${origName}__GatherC`), [
          encodeAttrInt('axis', 0),
        ]),
        encodeNode('Gather', [xShape, cIdx2], [dimD], allocName(`${origName}__GatherD`), [
          encodeAttrInt('axis', 0),
        ])
      );

      const cOutConst = allocName(`${origName}__c_out`);
      addedInitializers.push(encodeInt64Initializer(cOutConst, [1], [outC]));

      if (kD === 1 && sD === 1 && pD0 === 0 && pD1 === 0) {
        // Fast path: 1xkxk 3D Conv -> Transpose [0,2,1,3,4] -> Reshape [-1, C_in, H, W] -> Conv2D -> Reshape [N, D, M, H_out, W_out] -> Transpose [0,2,1,3,4]
        const w4dName = allocName(`${weightName}__4d`);
        addedInitializers.push(
          encodeFloat32Initializer(w4dName, [outC, inCPerGroup, kH, kW], weightInit.floatData)
        );

        const xTrans = allocName(`${origName}__x_ndchw`);
        const dimND = allocName(`${origName}__dim_ND`);
        const x4dShape = allocName(`${origName}__x4d_shape`);
        const x4d = allocName(`${origName}__x_4d`);
        const y4d = allocName(`${origName}__y_4d`);
        const y4dShape = allocName(`${origName}__y4d_shape`);
        const dimHOut = allocName(`${origName}__dim_Hout`);
        const dimWOut = allocName(`${origName}__dim_Wout`);
        const y5dShape = allocName(`${origName}__y5d_ndmhw_shape`);
        const y5dNdmhw = allocName(`${origName}__y_ndmhw`);

        const convInputs = biasInput ? [x4d, w4dName, biasInput] : [x4d, w4dName];

        const dimHIn = allocName(`${origName}__dim_Hin`);
        const dimWIn = allocName(`${origName}__dim_Win`);
        nextGraphParts.push(
          encodeNode('Transpose', [xInput], [xTrans], allocName(`${origName}__TransposeIn`), [
            encodeAttrInts('perm', [0, 2, 1, 3, 4]),
          ]),
          encodeNode('Mul', [dimN, dimD], [dimND], allocName(`${origName}__MulND`)),
          encodeNode('Gather', [xShape, cIdx3], [dimHIn], allocName(`${origName}__GatherHin`), [
            encodeAttrInt('axis', 0),
          ]),
          encodeNode('Gather', [xShape, cIdx4], [dimWIn], allocName(`${origName}__GatherWin`), [
            encodeAttrInt('axis', 0),
          ]),
          encodeNode(
            'Concat',
            [dimND, dimC, dimHIn, dimWIn],
            [x4dShape],
            allocName(`${origName}__ConcatX4dShape`),
            [encodeAttrInt('axis', 0)]
          ),
          encodeNode('Reshape', [xTrans, x4dShape], [x4d], allocName(`${origName}__ReshapeIn`)),
          encodeNode('Conv', convInputs, [y4d], allocName(`${origName}__Conv2D`), conv2dAttrs),
          encodeNode('Shape', [y4d], [y4dShape], allocName(`${origName}__ShapeY4d`)),
          encodeNode('Gather', [y4dShape, cIdx2], [dimHOut], allocName(`${origName}__GatherHout`), [
            encodeAttrInt('axis', 0),
          ]),
          encodeNode('Gather', [y4dShape, cIdx3], [dimWOut], allocName(`${origName}__GatherWout`), [
            encodeAttrInt('axis', 0),
          ]),
          encodeNode(
            'Concat',
            [dimN, dimD, cOutConst, dimHOut, dimWOut],
            [y5dShape],
            allocName(`${origName}__ConcatY5dShape`),
            [encodeAttrInt('axis', 0)]
          ),
          encodeNode('Reshape', [y4d, y5dShape], [y5dNdmhw], allocName(`${origName}__ReshapeOut`)),
          encodeNode('Transpose', [y5dNdmhw], [yOutput], origName, [
            encodeAttrInts('perm', [0, 2, 1, 3, 4]),
          ])
        );
        continue;
      }

      // General 5D Conv decomposition along depth kernel kD:
      // 1. Optional Pad along D axis (axis 2) if pD0 > 0 or pD1 > 0
      let xDepthReady = xInput;
      if (pD0 > 0 || pD1 > 0) {
        const padsConst = allocName(`${origName}__pads_5d`);
        addedInitializers.push(
          encodeInt64Initializer(padsConst, [10], [0, 0, pD0, 0, 0, 0, 0, pD1, 0, 0])
        );
        const xPadded = allocName(`${origName}__x_padded`);
        nextGraphParts.push(
          encodeNode(
            'Pad',
            [xInput, padsConst, cZeroFloat],
            [xPadded],
            allocName(`${origName}__PadDepth`),
            [encodeAttrString('mode', 'constant')]
          )
        );
        xDepthReady = xPadded;
      }

      const dimHIn = allocName(`${origName}__dim_Hin`);
      const dimWIn = allocName(`${origName}__dim_Win`);
      nextGraphParts.push(
        encodeNode('Gather', [xShape, cIdx3], [dimHIn], allocName(`${origName}__GatherHin`), [
          encodeAttrInt('axis', 0),
        ]),
        encodeNode('Gather', [xShape, cIdx4], [dimWIn], allocName(`${origName}__GatherWin`), [
          encodeAttrInt('axis', 0),
        ])
      );

      // Compute D_out dynamically: for m = 0, start = 0, end = D_padded - (kD - 1) * dD, step = sD
      // Specifically, for each m in 0 .. kD - 1:
      // start_m = m * dD
      // end_m = D_in + pD0 + pD1 - (kD - 1 - m) * dD
      const cPadSum = pD0 + pD1;
      let dPaddedTensor = dimD;
      if (cPadSum > 0) {
        const padSumConst = allocName(`${origName}__pad_sum`);
        addedInitializers.push(encodeInt64Initializer(padSumConst, [1], [cPadSum]));
        const dPadOut = allocName(`${origName}__dim_Dpadded`);
        nextGraphParts.push(
          encodeNode('Add', [dimD, padSumConst], [dPadOut], allocName(`${origName}__AddDpad`))
        );
        dPaddedTensor = dPadOut;
      }

      const cStepSD = sD === 1 ? cStep1 : allocName(`${origName}__step_sD`);
      if (sD !== 1) {
        addedInitializers.push(encodeInt64Initializer(cStepSD, [1], [sD]));
      }

      const spatialKernelSize = kH * kW;
      const depthStrideInWeight = kD * spatialKernelSize;
      let accum4d: string | null = null;
      let firstSlice3dForDout: string | null = null;
      let sharedX4dShape: string | null = null;

      for (let m = 0; m < kD; m++) {
        // Extract 4D weight slice W_m of shape [outC, inCPerGroup, kH, kW]
        const wSliceData = new Float32Array(outC * inCPerGroup * spatialKernelSize);
        let dstOff = 0;
        for (let oc = 0; oc < outC; oc++) {
          for (let ic = 0; ic < inCPerGroup; ic++) {
            const srcBase = (oc * inCPerGroup + ic) * depthStrideInWeight + m * spatialKernelSize;
            for (let sp = 0; sp < spatialKernelSize; sp++) {
              wSliceData[dstOff++] = weightInit.floatData[srcBase + sp];
            }
          }
        }
        const wSliceName = allocName(`${weightName}__m${m}`);
        addedInitializers.push(
          encodeFloat32Initializer(wSliceName, [outC, inCPerGroup, kH, kW], wSliceData)
        );

        const startVal = m * dD;
        const tailMargin = (kD - 1 - m) * dD;
        const startConst = startVal === 0 ? cIdx0 : allocName(`${origName}__start_m${m}`);
        if (startVal !== 0) {
          addedInitializers.push(encodeInt64Initializer(startConst, [1], [startVal]));
        }

        let endTensor = cInt64Max;
        if (tailMargin > 0) {
          const marginConst = allocName(`${origName}__margin_m${m}`);
          addedInitializers.push(encodeInt64Initializer(marginConst, [1], [tailMargin]));
          const computedEnd = allocName(`${origName}__end_m${m}`);
          nextGraphParts.push(
            encodeNode(
              'Sub',
              [dPaddedTensor, marginConst],
              [computedEnd],
              allocName(`${origName}__SubEnd_m${m}`)
            )
          );
          endTensor = computedEnd;
        } else if (sD > 1) {
          endTensor = dPaddedTensor;
        }

        const xSlice5d =
          kD === 1 && sD === 1
            ? xDepthReady
            : (() => {
                const sliced = allocName(`${origName}__x_slice_m${m}`);
                nextGraphParts.push(
                  encodeNode(
                    'Slice',
                    [xDepthReady, startConst, endTensor, cIdx2, cStepSD],
                    [sliced],
                    allocName(`${origName}__Slice_m${m}`)
                  )
                );
                return sliced;
              })();

        if (m === 0) {
          firstSlice3dForDout = xSlice5d;
        }

        // Transpose [N, C, D_out, H, W] -> [N, D_out, C, H, W] and Reshape to [N * D_out, C, H, W]
        const xTransM = allocName(`${origName}__x_trans_m${m}`);
        nextGraphParts.push(
          encodeNode(
            'Transpose',
            [xSlice5d],
            [xTransM],
            allocName(`${origName}__TransposeIn_m${m}`),
            [encodeAttrInts('perm', [0, 2, 1, 3, 4])]
          )
        );

        let x4dShapeM: string;
        if (m === 0) {
          const sliceShape0 = allocName(`${origName}__slice0_shape`);
          const dimDOut = allocName(`${origName}__dim_Dout`);
          const dimNDOut = allocName(`${origName}__dim_NDout`);
          sharedX4dShape = allocName(`${origName}__x4d_shape`);
          nextGraphParts.push(
            encodeNode('Shape', [xSlice5d], [sliceShape0], allocName(`${origName}__ShapeSlice0`)),
            encodeNode(
              'Gather',
              [sliceShape0, cIdx2],
              [dimDOut],
              allocName(`${origName}__GatherDout`),
              [encodeAttrInt('axis', 0)]
            ),
            encodeNode('Mul', [dimN, dimDOut], [dimNDOut], allocName(`${origName}__MulNDout`)),
            encodeNode(
              'Concat',
              [dimNDOut, dimC, dimHIn, dimWIn],
              [sharedX4dShape],
              allocName(`${origName}__ConcatX4dShape`),
              [encodeAttrInt('axis', 0)]
            )
          );
          x4dShapeM = sharedX4dShape;
          // Save for reuse across m > 0 and final output reshape
          firstSlice3dForDout = dimDOut;
        } else {
          // Reuse the exact allocated [N * D_out, C, H_in, W_in] shape tensor computed at m=0
          x4dShapeM = sharedX4dShape!;
        }

        const x4dM = allocName(`${origName}__x4d_m${m}`);
        const y4dM = allocName(`${origName}__y4d_m${m}`);
        const convInputsM =
          m === 0 && biasInput ? [x4dM, wSliceName, biasInput] : [x4dM, wSliceName];

        nextGraphParts.push(
          encodeNode(
            'Reshape',
            [xTransM, x4dShapeM],
            [x4dM],
            allocName(`${origName}__ReshapeIn_m${m}`)
          ),
          encodeNode(
            'Conv',
            convInputsM,
            [y4dM],
            allocName(`${origName}__Conv2D_m${m}`),
            conv2dAttrs
          )
        );

        if (accum4d === null) {
          accum4d = y4dM;
        } else {
          const nextAccum = allocName(`${origName}__accum_m${m}`);
          nextGraphParts.push(
            encodeNode('Add', [accum4d, y4dM], [nextAccum], allocName(`${origName}__Add_m${m}`))
          );
          accum4d = nextAccum;
        }
      }

      // Restore [N * D_out, M, H_out, W_out] -> [N, D_out, M, H_out, W_out] -> Transpose [0, 2, 1, 3, 4] -> [N, M, D_out, H_out, W_out]
      const y4dShape = allocName(`${origName}__y4d_shape`);
      const dimHOut = allocName(`${origName}__dim_Hout`);
      const dimWOut = allocName(`${origName}__dim_Wout`);
      const y5dShape = allocName(`${origName}__y5d_ndmhw_shape`);
      const y5dNdmhw = allocName(`${origName}__y_ndmhw`);

      nextGraphParts.push(
        encodeNode('Shape', [accum4d!], [y4dShape], allocName(`${origName}__ShapeY4d`)),
        encodeNode('Gather', [y4dShape, cIdx2], [dimHOut], allocName(`${origName}__GatherHout`), [
          encodeAttrInt('axis', 0),
        ]),
        encodeNode('Gather', [y4dShape, cIdx3], [dimWOut], allocName(`${origName}__GatherWout`), [
          encodeAttrInt('axis', 0),
        ]),
        encodeNode(
          'Concat',
          [dimN, firstSlice3dForDout!, cOutConst, dimHOut, dimWOut],
          [y5dShape],
          allocName(`${origName}__ConcatY5dShape`),
          [encodeAttrInt('axis', 0)]
        ),
        encodeNode('Reshape', [accum4d!, y5dShape], [y5dNdmhw], allocName(`${origName}__ReshapeOut`)),
        encodeNode('Transpose', [y5dNdmhw], [yOutput], origName, [
          encodeAttrInts('perm', [0, 2, 1, 3, 4]),
        ])
      );
    }

    if (decomposedNodeNames.length === 0) {
      return { bytes, decomposedCount: 0, decomposedNodeNames: [] };
    }

    const replacedWeightNames = new Set<string>();
    for (const [wName, replacedCount] of replacedWeightCounts) {
      if (replacedCount >= (initializerUseCounts.get(wName) ?? 0)) {
        replacedWeightNames.add(wName);
      }
    }

    // Filter out replaced 5D weight initializers (and any legacy GraphProto.input duplicates)
    // and append the new 4D weight & constant initializers
    const finalGraphParts: Uint8Array[] = [];
    for (const part of nextGraphParts) {
      const sliceFields = parseProtoFields(part, 0, part.byteLength);
      if (sliceFields && sliceFields.length === 1) {
        const sf = sliceFields[0];
        if (sf.fn === 5 && sf.wt === 2) {
          const initName = getInitializerName(part, sf.valStart, sf.valEnd, decoder);
          if (initName && replacedWeightNames.has(initName)) {
            continue;
          }
        } else if (sf.fn === 11 && sf.wt === 2) {
          const inputName = getValueInfoName(part, sf.valStart, sf.valEnd, decoder);
          if (inputName && replacedWeightNames.has(inputName)) {
            continue;
          }
        }
      }
      finalGraphParts.push(part);
    }
    for (const initPart of addedInitializers) {
      finalGraphParts.push(initPart);
    }

    let graphPayloadLen = 0;
    for (const part of finalGraphParts) graphPayloadLen += part.byteLength;

    const graphTag = encodeProtoVarint((7 << 3) | 2);
    const graphLen = encodeProtoVarint(graphPayloadLen);
    let totalLen = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        totalLen += graphTag.byteLength + graphLen.byteLength + graphPayloadLen;
      } else {
        totalLen += topFields[i].fEnd - topFields[i].fStart;
      }
    }

    const out = new Uint8Array(totalLen);
    let offset = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        out.set(graphTag, offset);
        offset += graphTag.byteLength;
        out.set(graphLen, offset);
        offset += graphLen.byteLength;
        for (const part of finalGraphParts) {
          out.set(part, offset);
          offset += part.byteLength;
        }
      } else {
        const chunk = bytes.subarray(topFields[i].fStart, topFields[i].fEnd);
        out.set(chunk, offset);
        offset += chunk.byteLength;
      }
    }

    // Static SSA integrity gate: ensure rewrite introduced zero dangling node input references
    const beforeValidation = validateOnnxGraphNodeInputs(bytes);
    const afterValidation = validateOnnxGraphNodeInputs(out);
    const preexistingMissing = new Set(beforeValidation.missingInputs.map((m) => m.inputName));
    const newlyMissing = afterValidation.missingInputs.filter(
      (m) => !preexistingMissing.has(m.inputName)
    );
    if (
      newlyMissing.length > 0 ||
      (beforeValidation.hasGraphInputs && !afterValidation.valid)
    ) {
      const sample = newlyMissing[0] ?? afterValidation.missingInputs[0];
      console.warn(
        `[StereoDepth] Aborting 5D Conv3D rewrite due to unresolved node input '${sample?.inputName ?? 'unknown'}' on '${sample?.nodeName ?? 'unknown'}'`
      );
      return { bytes, decomposedCount: 0, decomposedNodeNames: [] };
    }

    return {
      bytes: out,
      decomposedCount: decomposedNodeNames.length,
      decomposedNodeNames,
    };
  } catch {
    return { bytes, decomposedCount: 0, decomposedNodeNames: [] };
  }
}

export interface OnnxSliceAxesNormalizeResult {
  bytes: Uint8Array;
  normalizedCount: number;
  normalizedNodeNames: string[];
  skippedDynamicCount: number;
  skippedInvalidOrUnprovenCount: number;
  compactedCount: number;
  mode: SliceAxesNormalizeMode;
}

interface Parsed1dIntInitializer {
  name: string;
  dataType: 6 | 7; // INT32 = 6, INT64 = 7
  values: bigint[];
}

function readProtoVarintBigInt(buf: Uint8Array, pos: number, end: number): [bigint, number] | null {
  let result = 0n;
  let shift = 0n;
  let cursor = pos;
  while (cursor < end && shift <= 63n) {
    const byte = buf[cursor++];
    result |= BigInt(byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) {
      return [result, cursor];
    }
    shift += 7n;
  }
  return null;
}

function parse1dIntConstantInitializer(
  buf: Uint8Array,
  slice: ProtobufFieldSlice,
  decoder: TextDecoder
): Parsed1dIntInitializer | null {
  const tensorFields = parseProtoFields(buf, slice.valStart, slice.valEnd);
  if (!tensorFields) return null;

  const dims: number[] = [];
  let dataType = 0;
  let name = '';
  const int32Values: bigint[] = [];
  const int64Values: bigint[] = [];
  let rawBytes: Uint8Array | null = null;
  let hasExternalLocation = false;

  for (const tf of tensorFields) {
    if (tf.fn === 1) {
      if (tf.wt === 0) {
        const v = readProtoVarint(buf, tf.valStart, tf.valEnd);
        if (v) dims.push(v[0]);
      } else if (tf.wt === 2) {
        let p = tf.valStart;
        while (p < tf.valEnd) {
          const next = readProtoVarint(buf, p, tf.valEnd);
          if (!next) break;
          dims.push(next[0]);
          p = next[1];
        }
      }
    } else if (tf.fn === 2 && tf.wt === 0) {
      const v = readProtoVarint(buf, tf.valStart, tf.valEnd);
      if (v) dataType = v[0];
    } else if (tf.fn === 5) {
      if (tf.wt === 0) {
        const v = readProtoVarintBigInt(buf, tf.valStart, tf.valEnd);
        if (!v) return null;
        int32Values.push(BigInt.asIntN(32, v[0]));
      } else if (tf.wt === 2) {
        let p = tf.valStart;
        while (p < tf.valEnd) {
          const v = readProtoVarintBigInt(buf, p, tf.valEnd);
          if (!v) return null;
          int32Values.push(BigInt.asIntN(32, v[0]));
          p = v[1];
        }
      }
    } else if (tf.fn === 7) {
      if (tf.wt === 0) {
        const v = readProtoVarintBigInt(buf, tf.valStart, tf.valEnd);
        if (!v) return null;
        int64Values.push(BigInt.asIntN(64, v[0]));
      } else if (tf.wt === 2) {
        let p = tf.valStart;
        while (p < tf.valEnd) {
          const v = readProtoVarintBigInt(buf, p, tf.valEnd);
          if (!v) return null;
          int64Values.push(BigInt.asIntN(64, v[0]));
          p = v[1];
        }
      }
    } else if (tf.fn === 8 && tf.wt === 2 && tf.valEnd > tf.valStart) {
      name = decoder.decode(buf.subarray(tf.valStart, tf.valEnd));
    } else if (tf.fn === 9 && tf.wt === 2) {
      rawBytes = buf.subarray(tf.valStart, tf.valEnd);
    } else if (tf.fn === 13 && tf.wt === 2) {
      hasExternalLocation = true;
    } else if (tf.fn === 14 && tf.wt === 0) {
      const v = readProtoVarint(buf, tf.valStart, tf.valEnd);
      if (v && v[0] !== 0) hasExternalLocation = true;
    }
  }

  if (!name || hasExternalLocation || (dataType !== 6 && dataType !== 7)) {
    return null;
  }
  if (dims.length !== 1 || dims[0] <= 0) {
    return null;
  }

  const len = dims[0];
  let values: bigint[] | null = null;
  if (dataType === 7) {
    if (rawBytes && rawBytes.byteLength === len * 8) {
      const dv = new DataView(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength);
      values = [];
      for (let i = 0; i < len; i++) {
        values.push(dv.getBigInt64(i * 8, true));
      }
    } else if (int64Values.length === len) {
      values = int64Values;
    }
  } else {
    if (rawBytes && rawBytes.byteLength === len * 4) {
      const dv = new DataView(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength);
      values = [];
      for (let i = 0; i < len; i++) {
        values.push(BigInt(dv.getInt32(i * 4, true)));
      }
    } else if (int32Values.length === len) {
      values = int32Values;
    }
  }

  if (!values || values.length !== len) {
    return null;
  }

  return {
    name,
    dataType: dataType as 6 | 7,
    values,
  };
}

const INT32_MIN_BIGINT = -2147483648n;
const INT32_MAX_BIGINT = 2147483647n;
const INT64_MAX_BIGINT = 9223372036854775807n;

interface ParsedValueInfoShape {
  name: string;
  dims: number[] | null;
}

function parseValueInfoStaticShape(
  buf: Uint8Array,
  start: number,
  end: number,
  decoder: TextDecoder
): ParsedValueInfoShape | null {
  const viFields = parseProtoFields(buf, start, end);
  if (!viFields) return null;
  const nameField = viFields.find((f) => f.fn === 1 && f.wt === 2);
  if (!nameField || nameField.valEnd <= nameField.valStart) return null;
  const name = decoder.decode(buf.subarray(nameField.valStart, nameField.valEnd));
  const typeField = viFields.find((f) => f.fn === 2 && f.wt === 2);
  if (!typeField) return { name, dims: null };
  const typeFields = parseProtoFields(buf, typeField.valStart, typeField.valEnd);
  const tensorTypeField = typeFields?.find((f) => f.fn === 1 && f.wt === 2);
  if (!tensorTypeField) return { name, dims: null };
  const ttFields = parseProtoFields(buf, tensorTypeField.valStart, tensorTypeField.valEnd);
  const shapeField = ttFields?.find((f) => f.fn === 2 && f.wt === 2);
  if (!shapeField) return { name, dims: null };
  const shapeFields = parseProtoFields(buf, shapeField.valStart, shapeField.valEnd);
  if (!shapeFields) return { name, dims: null };

  const dims: number[] = [];
  for (const sf of shapeFields) {
    if (sf.fn !== 1 || sf.wt !== 2) continue;
    const dimFields = parseProtoFields(buf, sf.valStart, sf.valEnd);
    if (!dimFields) return { name, dims: null };
    const dimParam = dimFields.find((df) => df.fn === 2 && df.wt === 2);
    if (dimParam) return { name, dims: null };
    const dimVal = dimFields.find((df) => df.fn === 1 && df.wt === 0);
    if (!dimVal) return { name, dims: null };
    const decodedVal = readProtoVarintBigInt(buf, dimVal.valStart, dimVal.valEnd);
    if (!decodedVal || decodedVal[0] <= 0n || decodedVal[0] > INT32_MAX_BIGINT) {
      return { name, dims: null };
    }
    dims.push(Number(decodedVal[0]));
  }
  return { name, dims: dims.length > 0 ? dims : null };
}

function isSentinelOverflow(val: bigint, dataType: 6 | 7): boolean {
  if (val >= INT32_MIN_BIGINT && val <= INT32_MAX_BIGINT) {
    return false;
  }
  if (dataType === 7 && val === INT64_MAX_BIGINT) {
    return false;
  }
  return true;
}

/**
 * Reorders `(starts, ends, axes[, steps])` together by ascending `axis` index for opset >= 10
 * `Slice` nodes whose `axes` input is a 1D constant integer initializer with non-negative,
 * pairwise-distinct indices.
 *
 * Modes:
 * - `'off'`: no-op
 * - `'sort-only'` (default): reorders non-ascending tuples together by ascending `axis`
 * - `'compact-active'`: removes only Slice tuple entries mathematically proven to be full-axis
 *   identity (`start === 0`, `step === 1` when present, `end === INT32_MAX`/`INT64_MAX` or `>= knownDim`,
 *   with static positive graph input dims and strict bounds/overflow validation), keeping at least
 *   1 active axis and ascending axis order.
 *
 * Existing shared initializers are never mutated in-place; rewritten `Slice` nodes are wired to
 * fresh constant initializers so unrelated consumers and other `Slice` nodes remain untouched.
 * Dynamic, negative/unproven, duplicate, overflowed, zero-active-axis, or invalid `Slice` tuples
 * fail closed (are safely skipped).
 */
export function normalizeNonAscendingSliceAxesForWebGpu(
  bytes: Uint8Array,
  options: NormalizeSliceAxesOptions = {}
): OnnxSliceAxesNormalizeResult {
  const mode: SliceAxesNormalizeMode = options.mode ?? 'sort-only';
  const emptyResult: OnnxSliceAxesNormalizeResult = {
    bytes,
    normalizedCount: 0,
    normalizedNodeNames: [],
    skippedDynamicCount: 0,
    skippedInvalidOrUnprovenCount: 0,
    compactedCount: 0,
    mode,
  };
  if (mode === 'off' || bytes.byteLength < 16) {
    return emptyResult;
  }
  try {
    const topFields = parseProtoFields(bytes, 0, bytes.byteLength);
    if (!topFields) return emptyResult;

    const decoder = new TextDecoder();
    const defaultOpset = readOnnxDefaultOpsetVersion(bytes, topFields, decoder);
    // ONNX opset < 10 Slice takes attributes rather than tensor inputs
    if (defaultOpset !== null && defaultOpset < 10) {
      return emptyResult;
    }

    const graphIndex = topFields.findIndex((f) => f.fn === 7 && f.wt === 2);
    if (graphIndex < 0) return emptyResult;

    const graphField = topFields[graphIndex];
    const graphFields = parseProtoFields(bytes, graphField.valStart, graphField.valEnd);
    if (!graphFields) return emptyResult;

    const encoder = new TextEncoder();
    const usedNames = new Set<string>();
    const graphOutputNames = new Set<string>();
    const nodeOutputNames = new Set<string>();
    const constant1dIntInits = new Map<string, Parsed1dIntInitializer>();
    const allInitializerNames = new Set<string>();
    const initializerUseCounts = new Map<string, number>();
    const rawGraphInputShapes: ParsedValueInfoShape[] = [];
    const staticTensorShapes = new Map<string, number[]>();
    let hasControlFlowNode = false;

    for (const gf of graphFields) {
      if (gf.fn === 5 && gf.wt === 2) {
        const initName = getInitializerName(bytes, gf.valStart, gf.valEnd, decoder);
        if (initName) {
          usedNames.add(initName);
          allInitializerNames.add(initName);
          const parsed = parse1dIntConstantInitializer(bytes, gf, decoder);
          if (parsed) {
            constant1dIntInits.set(initName, parsed);
          }
        }
        continue;
      }
      if (gf.fn === 11 && gf.wt === 2) {
        const parsedShape = parseValueInfoStaticShape(bytes, gf.valStart, gf.valEnd, decoder);
        if (parsedShape) {
          usedNames.add(parsedShape.name);
          rawGraphInputShapes.push(parsedShape);
          if (parsedShape.dims) {
            staticTensorShapes.set(parsedShape.name, parsedShape.dims);
          }
        }
        continue;
      }
      if (gf.fn === 12 && gf.wt === 2) {
        const outName = getValueInfoName(bytes, gf.valStart, gf.valEnd, decoder);
        if (outName) {
          usedNames.add(outName);
          graphOutputNames.add(outName);
        }
        continue;
      }
      if (gf.fn !== 1 || gf.wt !== 2) continue;
      const nodeFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
      if (!nodeFields) continue;
      for (const nf of nodeFields) {
        if (nf.fn === 1 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          const inpName = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
          usedNames.add(inpName);
          initializerUseCounts.set(inpName, (initializerUseCounts.get(inpName) ?? 0) + 1);
        } else if (nf.fn === 2 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          const outName = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
          usedNames.add(outName);
          nodeOutputNames.add(outName);
        } else if (nf.fn === 3 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          usedNames.add(decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)));
        } else if (nf.fn === 4 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          const op = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
          if (op === 'If' || op === 'Loop' || op === 'Scan') {
            hasControlFlowNode = true;
          }
        }
      }
    }

    const nonInitGraphInputs = rawGraphInputShapes.filter((g) => !allInitializerNames.has(g.name));
    const hasStaticPositiveGraphInputs =
      nonInitGraphInputs.length > 0 && nonInitGraphInputs.every((g) => g.dims !== null);

    if (constant1dIntInits.size === 0) {
      return emptyResult;
    }

    let tempSeq = 0;
    const allocName = (prefix: string): string => {
      let candidate = `${prefix}_${tempSeq++}`;
      while (usedNames.has(candidate)) {
        candidate = `${prefix}_${tempSeq++}`;
      }
      usedNames.add(candidate);
      return candidate;
    };

    const concatBytes = (parts: readonly Uint8Array[]): Uint8Array => {
      let total = 0;
      for (const p of parts) total += p.byteLength;
      const out = new Uint8Array(total);
      let off = 0;
      for (const p of parts) {
        out.set(p, off);
        off += p.byteLength;
      }
      return out;
    };

    const encodeLenField = (fn: number, payload: Uint8Array): Uint8Array => {
      const tag = encodeProtoVarint((fn << 3) | 2);
      const len = encodeProtoVarint(payload.byteLength);
      const out = new Uint8Array(tag.byteLength + len.byteLength + payload.byteLength);
      out.set(tag, 0);
      out.set(len, tag.byteLength);
      out.set(payload, tag.byteLength + len.byteLength);
      return out;
    };

    const encodeVarintField = (fn: number, val: number): Uint8Array => {
      const tag = encodeProtoVarint((fn << 3) | 0);
      const vBytes = encodeProtoVarint(val);
      const out = new Uint8Array(tag.byteLength + vBytes.byteLength);
      out.set(tag, 0);
      out.set(vBytes, tag.byteLength);
      return out;
    };

    const encodeStrField = (fn: number, str: string): Uint8Array =>
      encodeLenField(fn, encoder.encode(str));

    const encode1dIntInitializer = (
      name: string,
      dataType: 6 | 7,
      values: readonly bigint[]
    ): Uint8Array => {
      const elemSize = dataType === 7 ? 8 : 4;
      const raw = new Uint8Array(values.length * elemSize);
      const dv = new DataView(raw.buffer);
      for (let i = 0; i < values.length; i++) {
        if (dataType === 7) {
          dv.setBigInt64(i * 8, values[i], true);
        } else {
          dv.setInt32(i * 4, Number(values[i]), true);
        }
      }
      return encodeLenField(
        5,
        concatBytes([
          encodeVarintField(1, values.length),
          encodeVarintField(2, dataType),
          encodeStrField(8, name),
          encodeLenField(9, raw),
        ])
      );
    };

    const addedInitializers: Uint8Array[] = [];
    const createdInitCache = new Map<string, string>();
    const getOrCreateIntInit = (
      rolePrefix: string,
      dataType: 6 | 7,
      values: readonly bigint[]
    ): string => {
      const key = `${dataType}:${values.map((v) => v.toString()).join(',')}`;
      const existing = createdInitCache.get(key);
      if (existing) return existing;
      const name = allocName(`__webgpu_slice_${rolePrefix}`);
      createdInitCache.set(key, name);
      addedInitializers.push(encode1dIntInitializer(name, dataType, values));
      return name;
    };

    const nextGraphParts: Uint8Array[] = [];
    const replacedInitCounts = new Map<string, number>();
    const normalizedNodeNames: string[] = [];
    let compactedCount = 0;
    let skippedDynamicCount = 0;
    let skippedInvalidOrUnprovenCount = 0;
    let sliceSeq = 0;

    for (const gf of graphFields) {
      if (gf.fn === 13) continue; // Drop value_info so reordered shapes never conflict with stale hints
      if (gf.fn !== 1 || gf.wt !== 2) {
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      const nodeFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
      if (!nodeFields) {
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      const opField = nodeFields.find((nf) => nf.fn === 4 && nf.wt === 2 && nf.valEnd > nf.valStart);
      const opType = opField ? decoder.decode(bytes.subarray(opField.valStart, opField.valEnd)) : '';
      if (opType !== 'Slice') {
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      const inputs: string[] = [];
      for (const nf of nodeFields) {
        if (nf.fn === 1 && nf.wt === 2) {
          inputs.push(
            nf.valEnd > nf.valStart ? decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)) : ''
          );
        }
      }
      const nameField = nodeFields.find((nf) => nf.fn === 3 && nf.wt === 2 && nf.valEnd > nf.valStart);
      const nodeName = nameField
        ? decoder.decode(bytes.subarray(nameField.valStart, nameField.valEnd))
        : `Slice_${sliceSeq}`;
      sliceSeq++;

      // Slice(data, starts, ends[, axes[, steps]]): 3-input or empty axes defaults to [0..r-1] ascending
      if (inputs.length < 4 || inputs.length > 5 || inputs[3] === '') {
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      const dataName = inputs[0];
      const startsName = inputs[1];
      const endsName = inputs[2];
      const axesName = inputs[3];
      const hasSteps = inputs.length === 5 && inputs[4] !== '';
      const stepsName = hasSteps ? inputs[4] : null;

      if (
        !startsName ||
        !endsName ||
        nodeOutputNames.has(startsName) ||
        nodeOutputNames.has(endsName) ||
        nodeOutputNames.has(axesName) ||
        (stepsName && nodeOutputNames.has(stepsName))
      ) {
        skippedDynamicCount++;
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      const startsInit = constant1dIntInits.get(startsName);
      const endsInit = constant1dIntInits.get(endsName);
      const axesInit = constant1dIntInits.get(axesName);
      const stepsInit = stepsName ? constant1dIntInits.get(stepsName) : undefined;

      if (!startsInit || !endsInit || !axesInit || (stepsName && !stepsInit)) {
        const isInvalidInit =
          (!startsInit && allInitializerNames.has(startsName)) ||
          (!endsInit && allInitializerNames.has(endsName)) ||
          (!axesInit && allInitializerNames.has(axesName)) ||
          (stepsName != null && !stepsInit && allInitializerNames.has(stepsName));
        if (isInvalidInit) {
          skippedInvalidOrUnprovenCount++;
        } else {
          skippedDynamicCount++;
        }
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      const len = axesInit.values.length;
      if (
        startsInit.values.length !== len ||
        endsInit.values.length !== len ||
        (stepsInit && stepsInit.values.length !== len)
      ) {
        skippedInvalidOrUnprovenCount++;
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      if (stepsInit && stepsInit.values.some((s) => s === 0n)) {
        skippedInvalidOrUnprovenCount++;
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      const knownDataDims = dataName ? (staticTensorShapes.get(dataName) ?? null) : null;
      let hasNegativeOrInvalidAxis = false;
      let hasDuplicateAxis = false;
      const seenAxes = new Set<bigint>();
      for (let i = 0; i < len; i++) {
        const ax = axesInit.values[i];
        if (ax < 0n || (knownDataDims !== null && ax >= BigInt(knownDataDims.length))) {
          hasNegativeOrInvalidAxis = true;
          break;
        }
        if (seenAxes.has(ax)) {
          hasDuplicateAxis = true;
          break;
        }
        seenAxes.add(ax);
      }
      if (hasNegativeOrInvalidAxis || hasDuplicateAxis) {
        skippedInvalidOrUnprovenCount++;
        nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        continue;
      }

      let isAlreadyAscending = true;
      for (let i = 1; i < len; i++) {
        if (axesInit.values[i] <= axesInit.values[i - 1]) {
          isAlreadyAscending = false;
          break;
        }
      }

      let order: number[] = [];
      let didCompact = false;

      if (mode === 'compact-active') {
        if (!hasStaticPositiveGraphInputs) {
          skippedDynamicCount++;
          nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
          continue;
        }

        let hasOverflow = false;
        for (let i = 0; i < len; i++) {
          if (
            isSentinelOverflow(startsInit.values[i], startsInit.dataType) ||
            isSentinelOverflow(endsInit.values[i], endsInit.dataType)
          ) {
            hasOverflow = true;
            break;
          }
        }
        if (hasOverflow) {
          skippedInvalidOrUnprovenCount++;
          nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
          continue;
        }

        const activeIndices: number[] = [];
        for (let i = 0; i < len; i++) {
          const axNum = Number(axesInit.values[i]);
          const knownDim =
            knownDataDims && axNum >= 0 && axNum < knownDataDims.length
              ? knownDataDims[axNum]
              : null;
          const s = startsInit.values[i];
          const e = endsInit.values[i];
          const st = stepsInit ? stepsInit.values[i] : 1n;

          const endCoversFullDim =
            e === INT32_MAX_BIGINT ||
            (endsInit.dataType === 7 && e === INT64_MAX_BIGINT) ||
            (knownDim !== null && e >= BigInt(knownDim));
          const isIdentityAxis = s === 0n && st === 1n && endCoversFullDim;
          if (!isIdentityAxis) {
            activeIndices.push(i);
          }
        }

        if (activeIndices.length === 0) {
          // Zero active axes remaining (pure identity Slice): fail closed so we never emit empty 0-length tuples.
          skippedInvalidOrUnprovenCount++;
          nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
          continue;
        }

        didCompact = activeIndices.length < len;
        if (!didCompact && isAlreadyAscending) {
          nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
          continue;
        }
        order = activeIndices;
      } else {
        if (len <= 1 || isAlreadyAscending) {
          nextGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
          continue;
        }
        order = Array.from({ length: len }, (_, i) => i);
      }

      order.sort((a, b) =>
        axesInit.values[a] < axesInit.values[b]
          ? -1
          : axesInit.values[a] > axesInit.values[b]
            ? 1
            : 0
      );

      const newStarts = order.map((i) => startsInit.values[i]);
      const newEnds = order.map((i) => endsInit.values[i]);
      const newAxes = order.map((i) => axesInit.values[i]);
      const newSteps = stepsInit ? order.map((i) => stepsInit.values[i]) : null;

      const newStartsName = getOrCreateIntInit('starts', startsInit.dataType, newStarts);
      const newEndsName = getOrCreateIntInit('ends', endsInit.dataType, newEnds);
      const newAxesName = getOrCreateIntInit('axes', axesInit.dataType, newAxes);
      const newStepsName =
        newSteps && stepsInit
          ? getOrCreateIntInit('steps', stepsInit.dataType, newSteps)
          : null;

      replacedInitCounts.set(startsName, (replacedInitCounts.get(startsName) ?? 0) + 1);
      replacedInitCounts.set(endsName, (replacedInitCounts.get(endsName) ?? 0) + 1);
      replacedInitCounts.set(axesName, (replacedInitCounts.get(axesName) ?? 0) + 1);
      if (stepsName) {
        replacedInitCounts.set(stepsName, (replacedInitCounts.get(stepsName) ?? 0) + 1);
      }

      const rewrittenParts: Uint8Array[] = [];
      let inputSlot = 0;
      for (const nf of nodeFields) {
        if (nf.fn === 1 && nf.wt === 2) {
          if (inputSlot === 1) {
            rewrittenParts.push(encodeStrField(1, newStartsName));
          } else if (inputSlot === 2) {
            rewrittenParts.push(encodeStrField(1, newEndsName));
          } else if (inputSlot === 3) {
            rewrittenParts.push(encodeStrField(1, newAxesName));
          } else if (inputSlot === 4 && newStepsName) {
            rewrittenParts.push(encodeStrField(1, newStepsName));
          } else {
            rewrittenParts.push(bytes.subarray(nf.fStart, nf.fEnd));
          }
          inputSlot++;
        } else {
          rewrittenParts.push(bytes.subarray(nf.fStart, nf.fEnd));
        }
      }

      nextGraphParts.push(encodeLenField(1, concatBytes(rewrittenParts)));
      normalizedNodeNames.push(nodeName);
      if (didCompact) {
        compactedCount++;
      }
    }

    if (normalizedNodeNames.length === 0) {
      return {
        bytes,
        normalizedCount: 0,
        normalizedNodeNames: [],
        skippedDynamicCount,
        skippedInvalidOrUnprovenCount,
        compactedCount: 0,
        mode,
      };
    }

    // Only remove an original initializer when ALL references across the entire graph were replaced
    // and it is not a graph output; shared initializers remain byte-for-byte untouched.
    const removableInitNames = new Set<string>();
    if (!hasControlFlowNode) {
      for (const [initName, replacedCount] of replacedInitCounts) {
        if (
          replacedCount >= (initializerUseCounts.get(initName) ?? 0) &&
          !graphOutputNames.has(initName)
        ) {
          removableInitNames.add(initName);
        }
      }
    }

    const finalGraphParts: Uint8Array[] = [];
    for (const part of nextGraphParts) {
      const sliceFields = parseProtoFields(part, 0, part.byteLength);
      if (sliceFields && sliceFields.length === 1) {
        const sf = sliceFields[0];
        if (sf.fn === 5 && sf.wt === 2) {
          const initName = getInitializerName(part, sf.valStart, sf.valEnd, decoder);
          if (initName && removableInitNames.has(initName)) {
            continue;
          }
        } else if (sf.fn === 11 && sf.wt === 2) {
          const inputName = getValueInfoName(part, sf.valStart, sf.valEnd, decoder);
          if (inputName && removableInitNames.has(inputName)) {
            continue;
          }
        }
      }
      finalGraphParts.push(part);
    }
    for (const initPart of addedInitializers) {
      finalGraphParts.push(initPart);
    }

    let graphPayloadLen = 0;
    for (const part of finalGraphParts) graphPayloadLen += part.byteLength;

    const graphTag = encodeProtoVarint((7 << 3) | 2);
    const graphLen = encodeProtoVarint(graphPayloadLen);
    let totalLen = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        totalLen += graphTag.byteLength + graphLen.byteLength + graphPayloadLen;
      } else {
        totalLen += topFields[i].fEnd - topFields[i].fStart;
      }
    }

    const out = new Uint8Array(totalLen);
    let offset = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        out.set(graphTag, offset);
        offset += graphTag.byteLength;
        out.set(graphLen, offset);
        offset += graphLen.byteLength;
        for (const part of finalGraphParts) {
          out.set(part, offset);
          offset += part.byteLength;
        }
      } else {
        const chunk = bytes.subarray(topFields[i].fStart, topFields[i].fEnd);
        out.set(chunk, offset);
        offset += chunk.byteLength;
      }
    }

    const beforeValidation = validateOnnxGraphNodeInputs(bytes);
    const afterValidation = validateOnnxGraphNodeInputs(out);
    const preexistingMissing = new Set(beforeValidation.missingInputs.map((m) => m.inputName));
    const newlyMissing = afterValidation.missingInputs.filter(
      (m) => !preexistingMissing.has(m.inputName)
    );
    if (
      newlyMissing.length > 0 ||
      (beforeValidation.hasGraphInputs && !afterValidation.valid)
    ) {
      const sample = newlyMissing[0] ?? afterValidation.missingInputs[0];
      console.warn(
        `[StereoDepth] Aborting Slice rewrite (${mode}) due to unresolved node input '${sample?.inputName ?? 'unknown'}' on '${sample?.nodeName ?? 'unknown'}'`
      );
      return emptyResult;
    }

    return {
      bytes: out,
      normalizedCount: normalizedNodeNames.length,
      normalizedNodeNames,
      skippedDynamicCount,
      skippedInvalidOrUnprovenCount,
      compactedCount,
      mode,
    };
  } catch {
    return emptyResult;
  }
}

export interface OnnxGatherNdRankNormalizeResult {
  bytes: Uint8Array;
  normalizedCount: number;
  normalizedNodeNames: string[];
  skippedDynamicCount: number;
  skippedInvalidOrUnprovenCount: number;
  totalGatherNdCount: number;
}

interface StaticTensorMeta {
  elemType: number;
  dims: number[];
  intVals?: number[];
}

interface ParsedStaticValueInfo {
  name: string;
  elemType: number | null;
  dims: number[] | null;
}

function parseValueInfoStaticTypeAndShape(
  buf: Uint8Array,
  start: number,
  end: number,
  decoder: TextDecoder
): ParsedStaticValueInfo | null {
  const viFields = parseProtoFields(buf, start, end);
  if (!viFields) return null;
  const nameField = viFields.find((f) => f.fn === 1 && f.wt === 2);
  if (!nameField || nameField.valEnd <= nameField.valStart) return null;
  const name = decoder.decode(buf.subarray(nameField.valStart, nameField.valEnd));
  const typeField = viFields.find((f) => f.fn === 2 && f.wt === 2);
  if (!typeField) return { name, elemType: null, dims: null };
  const typeFields = parseProtoFields(buf, typeField.valStart, typeField.valEnd);
  const tensorTypeField = typeFields?.find((f) => f.fn === 1 && f.wt === 2);
  if (!tensorTypeField) return { name, elemType: null, dims: null };
  const ttFields = parseProtoFields(buf, tensorTypeField.valStart, tensorTypeField.valEnd);
  if (!ttFields) return { name, elemType: null, dims: null };
  const elemTypeField = ttFields.find((f) => f.fn === 1 && f.wt === 0);
  const elemType = elemTypeField
    ? (readProtoVarint(buf, elemTypeField.valStart, elemTypeField.valEnd)?.[0] ?? null)
    : null;
  const shapeField = ttFields.find((f) => f.fn === 2 && f.wt === 2);
  if (!shapeField) return { name, elemType, dims: null };
  const shapeFields = parseProtoFields(buf, shapeField.valStart, shapeField.valEnd);
  if (!shapeFields) return { name, elemType, dims: null };

  const dims: number[] = [];
  for (const sf of shapeFields) {
    if (sf.fn !== 1 || sf.wt !== 2) continue;
    const dimFields = parseProtoFields(buf, sf.valStart, sf.valEnd);
    if (!dimFields) return { name, elemType, dims: null };
    const dimParam = dimFields.find((df) => df.fn === 2 && df.wt === 2);
    if (dimParam) return { name, elemType, dims: null };
    const dimVal = dimFields.find((df) => df.fn === 1 && df.wt === 0);
    if (!dimVal) return { name, elemType, dims: null };
    const decodedVal = readProtoVarintBigInt(buf, dimVal.valStart, dimVal.valEnd);
    if (!decodedVal || decodedVal[0] <= 0n || decodedVal[0] > INT32_MAX_BIGINT) {
      return { name, elemType, dims: null };
    }
    dims.push(Number(decodedVal[0]));
  }
  return { name, elemType, dims: dims.length > 0 ? dims : null };
}

function parseInitializerStaticMeta(
  buf: Uint8Array,
  slice: ProtobufFieldSlice,
  decoder: TextDecoder
): { name: string; meta: StaticTensorMeta } | null {
  const tensorFields = parseProtoFields(buf, slice.valStart, slice.valEnd);
  if (!tensorFields) return null;
  let name = '';
  let elemType = 0;
  const dims: number[] = [];
  let rawData: Uint8Array | null = null;
  const i32s: number[] = [];
  const i64s: number[] = [];
  let hasExternal = false;

  for (const tf of tensorFields) {
    if (tf.fn === 1) {
      if (tf.wt === 0) {
        const v = readProtoVarintBigInt(buf, tf.valStart, tf.valEnd);
        if (v) dims.push(Number(BigInt.asIntN(64, v[0])));
      } else if (tf.wt === 2) {
        let p = tf.valStart;
        while (p < tf.valEnd) {
          const v = readProtoVarintBigInt(buf, p, tf.valEnd);
          if (!v) break;
          dims.push(Number(BigInt.asIntN(64, v[0])));
          p = v[1];
        }
      }
    } else if (tf.fn === 2 && tf.wt === 0) {
      elemType = readProtoVarint(buf, tf.valStart, tf.valEnd)?.[0] ?? 0;
    } else if (tf.fn === 4) {
      if (tf.wt === 0) {
        const v = readProtoVarintBigInt(buf, tf.valStart, tf.valEnd);
        if (v) i32s.push(Number(BigInt.asIntN(32, v[0])));
      } else if (tf.wt === 2) {
        let p = tf.valStart;
        while (p < tf.valEnd) {
          const v = readProtoVarintBigInt(buf, p, tf.valEnd);
          if (!v) break;
          i32s.push(Number(BigInt.asIntN(32, v[0])));
          p = v[1];
        }
      }
    } else if (tf.fn === 7) {
      if (tf.wt === 0) {
        const v = readProtoVarintBigInt(buf, tf.valStart, tf.valEnd);
        if (v) i64s.push(Number(BigInt.asIntN(64, v[0])));
      } else if (tf.wt === 2) {
        let p = tf.valStart;
        while (p < tf.valEnd) {
          const v = readProtoVarintBigInt(buf, p, tf.valEnd);
          if (!v) break;
          i64s.push(Number(BigInt.asIntN(64, v[0])));
          p = v[1];
        }
      }
    } else if (tf.fn === 8 && tf.wt === 2 && tf.valEnd > tf.valStart) {
      name = decoder.decode(buf.subarray(tf.valStart, tf.valEnd));
    } else if (tf.fn === 9 && tf.wt === 2) {
      rawData = buf.subarray(tf.valStart, tf.valEnd);
    } else if (tf.fn === 13 && tf.wt === 2) {
      hasExternal = true;
    } else if (tf.fn === 14 && tf.wt === 0) {
      const v = readProtoVarint(buf, tf.valStart, tf.valEnd);
      if (v && v[0] !== 0) hasExternal = true;
    }
  }
  if (!name || !elemType || hasExternal) return null;

  let intVals: number[] | undefined;
  if (elemType === 7) {
    if (rawData && rawData.byteLength % 8 === 0 && rawData.byteLength <= 128) {
      const dv = new DataView(rawData.buffer, rawData.byteOffset, rawData.byteLength);
      intVals = Array.from({ length: rawData.byteLength / 8 }, (_, i) =>
        Number(dv.getBigInt64(i * 8, true))
      );
    } else if (i64s.length > 0 && i64s.length <= 16) {
      intVals = i64s;
    } else if (dims.length === 1 && dims[0] === 0) {
      intVals = [];
    }
  } else if (elemType === 6) {
    if (rawData && rawData.byteLength % 4 === 0 && rawData.byteLength <= 64) {
      const dv = new DataView(rawData.buffer, rawData.byteOffset, rawData.byteLength);
      intVals = Array.from({ length: rawData.byteLength / 4 }, (_, i) => dv.getInt32(i * 4, true));
    } else if (i32s.length > 0 && i32s.length <= 16) {
      intVals = i32s;
    }
  }
  return { name, meta: { elemType, dims, intVals } };
}

function broadcastStaticDims(a: readonly number[], b: readonly number[]): number[] | null {
  const rank = Math.max(a.length, b.length);
  const out = new Array<number>(rank);
  for (let i = 0; i < rank; i++) {
    const da = i < rank - a.length ? 1 : a[i - (rank - a.length)];
    const db = i < rank - b.length ? 1 : b[i - (rank - b.length)];
    if (da === db) out[i] = da;
    else if (da === 1) out[i] = db;
    else if (db === 1) out[i] = da;
    else return null;
  }
  return out;
}

function sameStaticDims(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/**
 * Workaround for ORT JSEP 1.30.0 GatherND Pass 2 shader-cache key collision when a model
 * executes both rank-4 and rank-5 output `GatherND` nodes with fixed rank-4 float32 data
 * and rank-1 slice offsets (`batch_dims=2`): flattens only the non-batch axes of `indices`
 * (`[1,240,320,3,1] -> [1,240,960,1]`) via Pre-`Reshape`, keeps all index values/dtypes,
 * `data` tensor, and `batch_dims` untouched, then restores `[1,240,320,3,32]` via Post-`Reshape`.
 *
 * Strictly opt-in and fail-closed on non-HITNet models, dynamic inputs, or mismatched topology.
 */
export function normalizeHitnetGatherNdRankForWebGpu(
  bytes: Uint8Array,
  options: NormalizeGatherNdRankOptions = {}
): OnnxGatherNdRankNormalizeResult {
  const emptyResult: OnnxGatherNdRankNormalizeResult = {
    bytes,
    normalizedCount: 0,
    normalizedNodeNames: [],
    skippedDynamicCount: 0,
    skippedInvalidOrUnprovenCount: 0,
    totalGatherNdCount: 0,
  };
  const requireHitnetContract = options.requireHitnetContract ?? true;
  if (
    (options.backendPreference === 'wasm' && !options.allowDiagnosticTransformedWasm) ||
    bytes.byteLength < 16
  ) {
    return emptyResult;
  }
  if (requireHitnetContract && options.variantId !== 'hitnet-sf') {
    return emptyResult;
  }

  try {
    const topFields = parseProtoFields(bytes, 0, bytes.byteLength);
    if (!topFields) return emptyResult;

    const decoder = new TextDecoder();
    const defaultOpset = readOnnxDefaultOpsetVersion(bytes, topFields, decoder);
    if (defaultOpset !== null && defaultOpset < 5) {
      return emptyResult;
    }

    const graphIndex = topFields.findIndex((f) => f.fn === 7 && f.wt === 2);
    if (graphIndex < 0) return emptyResult;

    const graphField = topFields[graphIndex];
    const graphFields = parseProtoFields(bytes, graphField.valStart, graphField.valEnd);
    if (!graphFields) return emptyResult;

    const encoder = new TextEncoder();
    const usedNames = new Set<string>();
    const allInitializerNames = new Set<string>();
    const parsedGraphInputs: ParsedStaticValueInfo[] = [];
    const parsedGraphOutputs: ParsedStaticValueInfo[] = [];
    const tensorMeta = new Map<string, StaticTensorMeta>();
    let hasControlFlowNode = false;

    interface ParsedNodeEntry {
      gf: ProtobufFieldSlice;
      nodeFields: ProtobufFieldSlice[];
      opType: string;
      name: string;
      inputs: string[];
      outputs: string[];
      attrs: Map<string, { i?: number; ints?: number[] }>;
    }
    const parsedNodes: ParsedNodeEntry[] = [];

    for (const gf of graphFields) {
      if (gf.fn === 5 && gf.wt === 2) {
        const init = parseInitializerStaticMeta(bytes, gf, decoder);
        if (init) {
          usedNames.add(init.name);
          allInitializerNames.add(init.name);
          tensorMeta.set(init.name, init.meta);
        } else {
          const nm = getInitializerName(bytes, gf.valStart, gf.valEnd, decoder);
          if (nm) {
            usedNames.add(nm);
            allInitializerNames.add(nm);
          }
        }
        continue;
      }
      if (gf.fn === 11 && gf.wt === 2) {
        const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
        if (vi) {
          usedNames.add(vi.name);
          parsedGraphInputs.push(vi);
          if (vi.elemType && vi.dims) {
            tensorMeta.set(vi.name, { elemType: vi.elemType, dims: vi.dims });
          }
        }
        continue;
      }
      if (gf.fn === 12 && gf.wt === 2) {
        const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
        if (vi) {
          usedNames.add(vi.name);
          parsedGraphOutputs.push(vi);
        }
        continue;
      }
      if (gf.fn !== 1 || gf.wt !== 2) continue;

      const nodeFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
      if (!nodeFields) return emptyResult;
      const inputs: string[] = [];
      const outputs: string[] = [];
      let name = '';
      let opType = '';
      const attrs = new Map<string, { i?: number; ints?: number[] }>();

      for (const nf of nodeFields) {
        if (nf.fn === 1 && nf.wt === 2) {
          inputs.push(
            nf.valEnd > nf.valStart ? decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)) : ''
          );
        } else if (nf.fn === 2 && nf.wt === 2) {
          const outName =
            nf.valEnd > nf.valStart ? decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)) : '';
          outputs.push(outName);
          if (outName) usedNames.add(outName);
        } else if (nf.fn === 3 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          name = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
          usedNames.add(name);
        } else if (nf.fn === 4 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          opType = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
          if (opType === 'If' || opType === 'Loop' || opType === 'Scan') {
            hasControlFlowNode = true;
          }
        } else if (nf.fn === 5 && nf.wt === 2) {
          const af = parseProtoFields(bytes, nf.valStart, nf.valEnd);
          if (!af) continue;
          let an = '';
          let ai: number | undefined;
          const aints: number[] = [];
          for (const a of af) {
            if (a.fn === 1 && a.wt === 2 && a.valEnd > a.valStart) {
              an = decoder.decode(bytes.subarray(a.valStart, a.valEnd));
            } else if (a.fn === 3 && a.wt === 0) {
              const v = readProtoVarintBigInt(bytes, a.valStart, a.valEnd);
              if (v) ai = Number(BigInt.asIntN(64, v[0]));
            } else if (a.fn === 8 && a.wt === 0) {
              const v = readProtoVarintBigInt(bytes, a.valStart, a.valEnd);
              if (v) aints.push(Number(BigInt.asIntN(64, v[0])));
            }
          }
          if (an) attrs.set(an, { i: ai, ints: aints });
        }
      }
      for (const inp of inputs) {
        if (inp) usedNames.add(inp);
      }
      parsedNodes.push({ gf, nodeFields, opType, name, inputs, outputs, attrs });
    }

    if (hasControlFlowNode) {
      return emptyResult;
    }

    const nonInitInputs = parsedGraphInputs.filter((g) => !allInitializerNames.has(g.name));
    if (
      nonInitInputs.length === 0 ||
      nonInitInputs.some((g) => g.dims === null || g.elemType === null)
    ) {
      return emptyResult;
    }

    if (requireHitnetContract) {
      if (
        nonInitInputs.length !== 1 ||
        nonInitInputs[0].name !== 'input' ||
        nonInitInputs[0].elemType !== 1 ||
        !nonInitInputs[0].dims ||
        !sameStaticDims(nonInitInputs[0].dims, [1, 6, 240, 320])
      ) {
        return emptyResult;
      }
      const outNames = parsedGraphOutputs.map((o) => o.name);
      if (
        outNames.length !== 2 ||
        !outNames.includes('reference_output_disparity') ||
        !outNames.includes('secondary_output_disparity')
      ) {
        return emptyResult;
      }
    }

    // Static shape, elemType, and small 1D integer value propagation across graph nodes
    for (const node of parsedNodes) {
      const ins = node.inputs.map((inp) => (inp ? tensorMeta.get(inp) : undefined));
      const m0 = ins[0];
      if (!m0) continue;

      let outMetas: StaticTensorMeta[] | null = null;
      switch (node.opType) {
        case 'Identity':
        case 'LeakyRelu':
        case 'Relu':
        case 'Abs':
        case 'Floor':
        case 'Neg':
          outMetas = [{ elemType: m0.elemType, dims: [...m0.dims], intVals: m0.intVals }];
          break;
        case 'Cast': {
          const to = node.attrs.get('to')?.i;
          if (to) outMetas = [{ elemType: to, dims: [...m0.dims], intVals: m0.intVals }];
          break;
        }
        case 'Add':
        case 'Sub':
        case 'Mul':
        case 'Min':
        case 'Max': {
          const m1 = ins[1];
          if (m1) {
            const d = broadcastStaticDims(m0.dims, m1.dims);
            if (d) {
              let intVals: number[] | undefined;
              if (
                m0.intVals &&
                m1.intVals &&
                (d.length === 0 || (d.length === 1 && d[0] <= 16))
              ) {
                const len = d.length === 0 ? 1 : d[0];
                intVals = Array.from({ length: len }, (_, i) => {
                  const va = m0.intVals![m0.intVals!.length === 1 ? 0 : i];
                  const vb = m1.intVals![m1.intVals!.length === 1 ? 0 : i];
                  if (node.opType === 'Add') return va + vb;
                  if (node.opType === 'Sub') return va - vb;
                  if (node.opType === 'Mul') return va * vb;
                  if (node.opType === 'Min') return Math.min(va, vb);
                  return Math.max(va, vb);
                });
              }
              outMetas = [{ elemType: m0.elemType, dims: d, intVals }];
            }
          }
          break;
        }
        case 'Transpose': {
          const perm = node.attrs.get('perm')?.ints;
          if (perm && perm.length === m0.dims.length) {
            outMetas = [{ elemType: m0.elemType, dims: perm.map((p) => m0.dims[p]) }];
          }
          break;
        }
        case 'Reshape': {
          const m1 = ins[1];
          if (m1?.intVals) {
            const inTotal = m0.dims.reduce((a, c) => a * c, 1);
            const target = m1.intVals.map((d, idx) => (d === 0 ? m0.dims[idx] : d));
            const negIdx = target.indexOf(-1);
            if (negIdx >= 0) {
              const knownProd = target.reduce(
                (a, c, idx) => (idx === negIdx ? a : a * c),
                1
              );
              if (knownProd > 0 && inTotal % knownProd === 0) {
                target[negIdx] = inTotal / knownProd;
              }
            }
            if (target.every((d) => d > 0)) {
              outMetas = [{ elemType: m0.elemType, dims: target }];
            }
          }
          break;
        }
        case 'Shape':
          outMetas = [{ elemType: 7, dims: [m0.dims.length], intVals: [...m0.dims] }];
          break;
        case 'Gather': {
          const axisRaw = node.attrs.get('axis')?.i ?? 0;
          const axis = axisRaw < 0 ? m0.dims.length + axisRaw : axisRaw;
          const m1 = ins[1];
          if (axis === 0 && m1 && m1.dims.length <= 1 && m1.intVals && m1.intVals.length === 1) {
            const idx0 = m1.intVals[0] < 0 ? m0.dims[0] + m1.intVals[0] : m1.intVals[0];
            if (idx0 >= 0 && idx0 < m0.dims[0]) {
              const val = m0.intVals ? m0.intVals[idx0] : undefined;
              outMetas = [
                {
                  elemType: m0.elemType,
                  dims: [...m1.dims, ...m0.dims.slice(1)],
                  intVals: val !== undefined ? [val] : undefined,
                },
              ];
            }
          }
          break;
        }
        case 'Concat': {
          const axRaw = node.attrs.get('axis')?.i ?? 0;
          const ax = axRaw < 0 ? m0.dims.length + axRaw : axRaw;
          if (ax >= 0 && ax < m0.dims.length && ins.every((x) => Boolean(x))) {
            const outDims = [...m0.dims];
            let sum = 0;
            for (const inp of ins) sum += inp!.dims[ax];
            outDims[ax] = sum;
            const intVals =
              outDims.length === 1 && ins.every((x) => Array.isArray(x?.intVals))
                ? ins.flatMap((x) => x!.intVals!)
                : undefined;
            outMetas = [{ elemType: m0.elemType, dims: outDims, intVals }];
          }
          break;
        }
        case 'Unsqueeze': {
          const axes = node.attrs.get('axes')?.ints;
          if (axes) {
            const outRank = m0.dims.length + axes.length;
            const normAxes = axes
              .map((a) => (a < 0 ? outRank + a : a))
              .sort((x, y) => x - y);
            const outDims: number[] = [];
            let j = 0;
            for (let i = 0; i < outRank; i++) {
              if (normAxes.includes(i)) outDims.push(1);
              else outDims.push(m0.dims[j++]);
            }
            outMetas = [{ elemType: m0.elemType, dims: outDims }];
          }
          break;
        }
        case 'Squeeze': {
          const axes = node.attrs.get('axes')?.ints ?? [];
          const normAxes = new Set(axes.map((a) => (a < 0 ? m0.dims.length + a : a)));
          outMetas = [
            {
              elemType: m0.elemType,
              dims: m0.dims.filter((d, i) => (normAxes.size > 0 ? !normAxes.has(i) : d !== 1)),
            },
          ];
          break;
        }
        case 'ReduceMin':
        case 'ReduceSum':
        case 'ArgMin': {
          const keepdims = node.attrs.get('keepdims')?.i ?? 1;
          const axes = node.attrs.get('axes')?.ints ?? [node.attrs.get('axis')?.i ?? 0];
          const normAxes = new Set(axes.map((a) => (a < 0 ? m0.dims.length + a : a)));
          const outDims: number[] = [];
          m0.dims.forEach((d, i) => {
            if (normAxes.has(i)) {
              if (keepdims) outDims.push(1);
            } else {
              outDims.push(d);
            }
          });
          outMetas = [{ elemType: node.opType === 'ArgMin' ? 7 : m0.elemType, dims: outDims }];
          break;
        }
        case 'Tile': {
          const m1 = ins[1];
          if (m1?.intVals && m1.intVals.length === m0.dims.length) {
            outMetas = [
              {
                elemType: m0.elemType,
                dims: m0.dims.map((d, i) => d * m1.intVals![i]),
              },
            ];
          }
          break;
        }
        case 'Expand': {
          const m1 = ins[1];
          if (m1?.intVals) {
            const d = broadcastStaticDims(m0.dims, m1.intVals);
            if (d) outMetas = [{ elemType: m0.elemType, dims: d }];
          }
          break;
        }
        case 'Slice': {
          const ms = ins[1]?.intVals;
          const me = ins[2]?.intVals;
          const ma = ins[3]?.intVals ?? Array.from({ length: ms?.length ?? 0 }, (_, i) => i);
          const mst = ins[4]?.intVals ?? Array.from({ length: ms?.length ?? 0 }, () => 1);
          if (ms && me && ms.length === me.length && ma.length === ms.length) {
            const outDims = [...m0.dims];
            let ok = true;
            for (let i = 0; i < ma.length; i++) {
              const ax = ma[i] < 0 ? m0.dims.length + ma[i] : ma[i];
              if (ax < 0 || ax >= m0.dims.length) {
                ok = false;
                break;
              }
              const dim = m0.dims[ax];
              const step = mst[i];
              if (step === 0) {
                ok = false;
                break;
              }
              let s = ms[i] < 0 ? dim + ms[i] : ms[i];
              let e = me[i] < 0 ? dim + me[i] : me[i];
              if (step > 0) {
                s = Math.max(0, Math.min(dim, s));
                e = Math.max(0, Math.min(dim, e));
                outDims[ax] = e <= s ? 0 : Math.ceil((e - s) / step);
              } else {
                s = Math.max(0, Math.min(dim - 1, s));
                e = Math.max(-1, Math.min(dim - 1, e));
                outDims[ax] = s <= e ? 0 : Math.ceil((s - e) / -step);
              }
            }
            if (ok) outMetas = [{ elemType: m0.elemType, dims: outDims }];
          }
          break;
        }
        case 'Conv': {
          const w = ins[1];
          if (w && m0.dims.length === 4 && w.dims.length === 4) {
            const ks = node.attrs.get('kernel_shape')?.ints ?? [w.dims[2], w.dims[3]];
            const st = node.attrs.get('strides')?.ints ?? [1, 1];
            const dil = node.attrs.get('dilations')?.ints ?? [1, 1];
            const pads = node.attrs.get('pads')?.ints ?? [0, 0, 0, 0];
            const oH =
              Math.floor((m0.dims[2] + pads[0] + pads[2] - dil[0] * (ks[0] - 1) - 1) / st[0]) +
              1;
            const oW =
              Math.floor((m0.dims[3] + pads[1] + pads[3] - dil[1] * (ks[1] - 1) - 1) / st[1]) +
              1;
            outMetas = [{ elemType: m0.elemType, dims: [m0.dims[0], w.dims[0], oH, oW] }];
          }
          break;
        }
        case 'ConvTranspose': {
          const w = ins[1];
          if (w && m0.dims.length === 4 && w.dims.length === 4) {
            const ks = node.attrs.get('kernel_shape')?.ints ?? [w.dims[2], w.dims[3]];
            const st = node.attrs.get('strides')?.ints ?? [1, 1];
            const dil = node.attrs.get('dilations')?.ints ?? [1, 1];
            const pads = node.attrs.get('pads')?.ints ?? [0, 0, 0, 0];
            const opad = node.attrs.get('output_padding')?.ints ?? [0, 0];
            const g = node.attrs.get('group')?.i ?? 1;
            const outC = w.dims[1] * g;
            const oH =
              st[0] * (m0.dims[2] - 1) + ((ks[0] - 1) * dil[0] + 1) - pads[0] - pads[2] + opad[0];
            const oW =
              st[1] * (m0.dims[3] - 1) + ((ks[1] - 1) * dil[1] + 1) - pads[1] - pads[3] + opad[1];
            outMetas = [{ elemType: m0.elemType, dims: [m0.dims[0], outC, oH, oW] }];
          }
          break;
        }
        case 'Split': {
          const axRaw = node.attrs.get('axis')?.i ?? 0;
          const ax = axRaw < 0 ? m0.dims.length + axRaw : axRaw;
          const splits =
            node.attrs.get('split')?.ints ??
            Array.from({ length: node.outputs.length }, () =>
              Math.floor(m0.dims[ax] / node.outputs.length)
            );
          outMetas = splits.map((sp) => {
            const d = [...m0.dims];
            d[ax] = sp;
            return { elemType: m0.elemType, dims: d };
          });
          break;
        }
        case 'GatherND': {
          const m1 = ins[1];
          const bdims = node.attrs.get('batch_dims')?.i ?? 0;
          if (m1 && m1.dims.length >= 1 && bdims >= 0 && bdims < m1.dims.length) {
            const k = m1.dims[m1.dims.length - 1];
            if (bdims + k <= m0.dims.length) {
              outMetas = [
                {
                  elemType: m0.elemType,
                  dims: [...m1.dims.slice(0, -1), ...m0.dims.slice(bdims + k)],
                },
              ];
            }
          }
          break;
        }
      }

      if (outMetas && outMetas.length === node.outputs.length) {
        for (let i = 0; i < node.outputs.length; i++) {
          if (node.outputs[i]) {
            tensorMeta.set(node.outputs[i], outMetas[i]);
          }
        }
      }
    }

    if (requireHitnetContract) {
      const refOutMeta = tensorMeta.get('reference_output_disparity');
      const secOutMeta = tensorMeta.get('secondary_output_disparity');
      if (
        !refOutMeta ||
        !secOutMeta ||
        refOutMeta.elemType !== 1 ||
        secOutMeta.elemType !== 1 ||
        !sameStaticDims(refOutMeta.dims, [1, 240, 320, 1]) ||
        !sameStaticDims(secOutMeta.dims, [1, 240, 320, 1])
      ) {
        return emptyResult;
      }
    }

    let tempSeq = 0;
    const allocName = (prefix: string): string => {
      let candidate = `${prefix}_${tempSeq++}`;
      while (usedNames.has(candidate)) {
        candidate = `${prefix}_${tempSeq++}`;
      }
      usedNames.add(candidate);
      return candidate;
    };

    const concatBytes = (parts: readonly Uint8Array[]): Uint8Array => {
      let total = 0;
      for (const p of parts) total += p.byteLength;
      const out = new Uint8Array(total);
      let off = 0;
      for (const p of parts) {
        out.set(p, off);
        off += p.byteLength;
      }
      return out;
    };

    const encodeLenField = (fn: number, payload: Uint8Array): Uint8Array => {
      const tag = encodeProtoVarint((fn << 3) | 2);
      const len = encodeProtoVarint(payload.byteLength);
      const out = new Uint8Array(tag.byteLength + len.byteLength + payload.byteLength);
      out.set(tag, 0);
      out.set(len, tag.byteLength);
      out.set(payload, tag.byteLength + len.byteLength);
      return out;
    };

    const encodeStrField = (fn: number, str: string): Uint8Array =>
      encodeLenField(fn, encoder.encode(str));

    const encodeNodeSimple = (
      opType: string,
      inputs: readonly string[],
      outputs: readonly string[],
      name: string
    ): Uint8Array => {
      const parts: Uint8Array[] = [];
      for (const inp of inputs) parts.push(encodeStrField(1, inp));
      for (const out of outputs) parts.push(encodeStrField(2, out));
      parts.push(encodeStrField(3, name));
      parts.push(encodeStrField(4, opType));
      return encodeLenField(1, concatBytes(parts));
    };

    const encodeInt64ShapeInitializer = (name: string, shapeDims: readonly number[]): Uint8Array => {
      const raw = new Uint8Array(shapeDims.length * 8);
      const dv = new DataView(raw.buffer);
      for (let i = 0; i < shapeDims.length; i++) {
        dv.setBigInt64(i * 8, BigInt(shapeDims[i]), true);
      }
      const varintField = (fn: number, val: number): Uint8Array => {
        const tag = encodeProtoVarint((fn << 3) | 0);
        const vBytes = encodeProtoVarint(val);
        const out = new Uint8Array(tag.byteLength + vBytes.byteLength);
        out.set(tag, 0);
        out.set(vBytes, tag.byteLength);
        return out;
      };
      return encodeLenField(
        5,
        concatBytes([
          varintField(1, shapeDims.length),
          varintField(2, 7), // TensorProto.INT64
          encodeStrField(8, name),
          encodeLenField(9, raw),
        ])
      );
    };

    const addedInitializers: Uint8Array[] = [];
    const shapeInitCache = new Map<string, string>();
    const getOrCreateInt64ShapeInit = (rolePrefix: string, shapeDims: readonly number[]): string => {
      const key = shapeDims.join(',');
      const existing = shapeInitCache.get(key);
      if (existing) return existing;
      const name = allocName(`__webgpu_gathernd_${rolePrefix}`);
      shapeInitCache.set(key, name);
      addedInitializers.push(encodeInt64ShapeInitializer(name, shapeDims));
      return name;
    };

    const nodeReplacementMap = new Map<number, Uint8Array[]>();
    const normalizedNodeNames: string[] = [];
    let totalGatherNdCount = 0;
    let skippedDynamicCount = 0;
    let skippedInvalidOrUnprovenCount = 0;

    for (let nodeIdx = 0; nodeIdx < parsedNodes.length; nodeIdx++) {
      const node = parsedNodes[nodeIdx];
      if (node.opType !== 'GatherND') continue;
      totalGatherNdCount++;

      if (node.inputs.length < 2 || node.outputs.length !== 1 || !node.outputs[0]) {
        skippedInvalidOrUnprovenCount++;
        continue;
      }

      const dataMeta = node.inputs[0] ? tensorMeta.get(node.inputs[0]) : undefined;
      const idxMeta = node.inputs[1] ? tensorMeta.get(node.inputs[1]) : undefined;
      const outMeta = tensorMeta.get(node.outputs[0]);
      if (!dataMeta || !idxMeta || !outMeta) {
        if (!dataMeta && !allInitializerNames.has(node.inputs[0])) {
          skippedDynamicCount++;
        } else if (!idxMeta && !allInitializerNames.has(node.inputs[1])) {
          skippedDynamicCount++;
        } else {
          skippedInvalidOrUnprovenCount++;
        }
        continue;
      }

      const batchDims = node.attrs.get('batch_dims')?.i ?? 0;
      const idxRank = idxMeta.dims.length;
      const dataRank = dataMeta.dims.length;
      const outRank = outMeta.dims.length;
      const lastIdxDim = idxMeta.dims[idxRank - 1];

      if (
        batchDims < 0 ||
        batchDims >= idxRank - 1 ||
        batchDims + lastIdxDim > dataRank ||
        dataMeta.elemType !== 1 ||
        idxMeta.elemType !== 7
      ) {
        if (outRank >= 5) {
          skippedInvalidOrUnprovenCount++;
        }
        continue;
      }

      if (requireHitnetContract) {
        const isWarp1dTargetName =
          (node.name.includes('level0/shared/prop0/') ||
            node.name.includes('level0_1/shared/prop0/')) &&
          node.name.includes('/warp_1d/') &&
          node.name.includes('GatherV2');
        const matchesExactWarp1dShapes =
          batchDims === 2 &&
          sameStaticDims(dataMeta.dims, [1, 240, 320, 32]) &&
          sameStaticDims(idxMeta.dims, [1, 240, 320, 3, 1]) &&
          sameStaticDims(outMeta.dims, [1, 240, 320, 3, 32]);
        if (!isWarp1dTargetName) {
          continue;
        }
        if (!matchesExactWarp1dShapes) {
          skippedInvalidOrUnprovenCount++;
          continue;
        }
      } else {
        // Synthetic diagnostic controls: flatten only rank-5 output GatherND with rank-4 float32 data and >= 2 non-batch index axes
        const nonBatchIndexAxisCount = idxRank - batchDims - 1;
        if (outRank !== 5 || dataRank !== 4 || batchDims !== 2 || nonBatchIndexAxisCount < 2) {
          continue;
        }
      }

      const nonBatchIndexDims = idxMeta.dims.slice(batchDims, -1);
      const nonBatchProd = nonBatchIndexDims.reduce((a, b) => a * b, 1);
      if (!Number.isSafeInteger(nonBatchProd) || nonBatchProd <= 0) {
        skippedInvalidOrUnprovenCount++;
        continue;
      }

      const preShapeDims = [...idxMeta.dims.slice(0, batchDims), nonBatchProd, lastIdxDim];
      const postShapeDims = [...outMeta.dims];
      if (preShapeDims.length !== 4 || postShapeDims.length !== 5) {
        skippedInvalidOrUnprovenCount++;
        continue;
      }

      const origNodeName = node.name || `GatherND_${nodeIdx}`;
      const preShapeName = getOrCreateInt64ShapeInit('pre_shape', preShapeDims);
      const postShapeName = getOrCreateInt64ShapeInit('post_shape', postShapeDims);
      const flatIndicesName = allocName(`${origNodeName}__flat_indices`);
      const flatOutName = allocName(`${origNodeName}__flat_out`);

      // Rebuild GatherND preserving exact name, domain, attributes (batch_dims=2), and unchanged data input[0],
      // substituting only flattened indices input[1] and rank-4 output[0].
      const rewrittenNodeParts: Uint8Array[] = [];
      let inSlot = 0;
      let outSlot = 0;
      for (const nf of node.nodeFields) {
        if (nf.fn === 1 && nf.wt === 2) {
          if (inSlot === 1) {
            rewrittenNodeParts.push(encodeStrField(1, flatIndicesName));
          } else {
            rewrittenNodeParts.push(bytes.subarray(nf.fStart, nf.fEnd));
          }
          inSlot++;
        } else if (nf.fn === 2 && nf.wt === 2) {
          if (outSlot === 0) {
            rewrittenNodeParts.push(encodeStrField(2, flatOutName));
          } else {
            rewrittenNodeParts.push(bytes.subarray(nf.fStart, nf.fEnd));
          }
          outSlot++;
        } else {
          rewrittenNodeParts.push(bytes.subarray(nf.fStart, nf.fEnd));
        }
      }

      nodeReplacementMap.set(nodeIdx, [
        encodeNodeSimple(
          'Reshape',
          [node.inputs[1], preShapeName],
          [flatIndicesName],
          allocName(`${origNodeName}__PreReshape`)
        ),
        encodeLenField(1, concatBytes(rewrittenNodeParts)),
        encodeNodeSimple(
          'Reshape',
          [flatOutName, postShapeName],
          [node.outputs[0]],
          allocName(`${origNodeName}__PostReshape`)
        ),
      ]);
      normalizedNodeNames.push(origNodeName);
    }

    if (requireHitnetContract) {
      const level0Count = normalizedNodeNames.filter((n) => n.startsWith('level0/')).length;
      const level01Count = normalizedNodeNames.filter((n) => n.startsWith('level0_1/')).length;
      if (
        totalGatherNdCount !== 10 ||
        normalizedNodeNames.length !== 4 ||
        level0Count !== 2 ||
        level01Count !== 2 ||
        skippedDynamicCount > 0 ||
        skippedInvalidOrUnprovenCount > 0
      ) {
        return emptyResult;
      }
    }

    if (normalizedNodeNames.length === 0) {
      return {
        bytes,
        normalizedCount: 0,
        normalizedNodeNames: [],
        skippedDynamicCount,
        skippedInvalidOrUnprovenCount,
        totalGatherNdCount,
      };
    }

    const finalGraphParts: Uint8Array[] = [];
    let currentNodeIdx = 0;
    for (const gf of graphFields) {
      if (gf.fn === 13) continue; // Drop stale value_info entries
      if (gf.fn === 1 && gf.wt === 2) {
        const replacement = nodeReplacementMap.get(currentNodeIdx);
        currentNodeIdx++;
        if (replacement) {
          for (const part of replacement) {
            finalGraphParts.push(part);
          }
        } else {
          finalGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        }
      } else {
        finalGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
      }
    }
    for (const initPart of addedInitializers) {
      finalGraphParts.push(initPart);
    }

    let graphPayloadLen = 0;
    for (const part of finalGraphParts) graphPayloadLen += part.byteLength;

    const graphTag = encodeProtoVarint((7 << 3) | 2);
    const graphLen = encodeProtoVarint(graphPayloadLen);
    let totalLen = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        totalLen += graphTag.byteLength + graphLen.byteLength + graphPayloadLen;
      } else {
        totalLen += topFields[i].fEnd - topFields[i].fStart;
      }
    }

    const out = new Uint8Array(totalLen);
    let offset = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        out.set(graphTag, offset);
        offset += graphTag.byteLength;
        out.set(graphLen, offset);
        offset += graphLen.byteLength;
        for (const part of finalGraphParts) {
          out.set(part, offset);
          offset += part.byteLength;
        }
      } else {
        const chunk = bytes.subarray(topFields[i].fStart, topFields[i].fEnd);
        out.set(chunk, offset);
        offset += chunk.byteLength;
      }
    }

    const beforeValidation = validateOnnxGraphNodeInputs(bytes);
    const afterValidation = validateOnnxGraphNodeInputs(out);
    const preexistingMissing = new Set(beforeValidation.missingInputs.map((m) => m.inputName));
    const newlyMissing = afterValidation.missingInputs.filter(
      (m) => !preexistingMissing.has(m.inputName)
    );
    if (
      newlyMissing.length > 0 ||
      (beforeValidation.hasGraphInputs && !afterValidation.valid)
    ) {
      const sample = newlyMissing[0] ?? afterValidation.missingInputs[0];
      console.warn(
        `[StereoDepth] Aborting GatherND rank rewrite due to unresolved node input '${sample?.inputName ?? 'unknown'}' on '${sample?.nodeName ?? 'unknown'}'`
      );
      return emptyResult;
    }

    return {
      bytes: out,
      normalizedCount: normalizedNodeNames.length,
      normalizedNodeNames,
      skippedDynamicCount,
      skippedInvalidOrUnprovenCount,
      totalGatherNdCount,
    };
  } catch {
    return emptyResult;
  }
}

export interface OnnxStaticInferenceResult {
  valid: boolean;
  irVersion: number | null;
  defaultOpset: number | null;
  hasControlFlowNode: boolean;
  tensorMeta: Record<string, StaticTensorMeta>;
  duplicateOutputs: string[];
  duplicateInitializers: string[];
  duplicateNodeNames: string[];
  missingInputs: Array<{ nodeName: string; opType: string; inputName: string }>;
}

/**
 * Infers static ONNX tensor `(elemType, dims, intVals)` across nodes by source tensor name
 * without mutating `bytes`, and checks SSA output uniqueness, initializer uniqueness, and input resolution.
 */
export function inferOnnxStaticTensorMeta(bytes: Uint8Array): OnnxStaticInferenceResult {
  const empty: OnnxStaticInferenceResult = {
    valid: false,
    irVersion: null,
    defaultOpset: null,
    hasControlFlowNode: false,
    tensorMeta: {},
    duplicateOutputs: [],
    duplicateInitializers: [],
    duplicateNodeNames: [],
    missingInputs: [],
  };
  if (bytes.byteLength < 16) return empty;

  try {
    const topFields = parseProtoFields(bytes, 0, bytes.byteLength);
    if (!topFields) return empty;
    const decoder = new TextDecoder();
    const irField = topFields.find((f) => f.fn === 1 && f.wt === 0);
    const irVersion = irField
      ? (readProtoVarint(bytes, irField.valStart, irField.valEnd)?.[0] ?? null)
      : null;
    const defaultOpset = readOnnxDefaultOpsetVersion(bytes, topFields, decoder);

    const graphField = topFields.find((f) => f.fn === 7 && f.wt === 2);
    if (!graphField) return empty;
    const graphFields = parseProtoFields(bytes, graphField.valStart, graphField.valEnd);
    if (!graphFields) return empty;

    const tensorMeta = new Map<string, StaticTensorMeta>();
    const seenInitializers = new Set<string>();
    const duplicateInitializers: string[] = [];
    const seenOutputs = new Set<string>();
    const duplicateOutputs: string[] = [];
    const seenNodeNames = new Set<string>();
    const duplicateNodeNames: string[] = [];
    let hasControlFlowNode = false;

    interface InferNodeEntry {
      opType: string;
      name: string;
      inputs: string[];
      outputs: string[];
      attrs: Map<string, { i?: number; ints?: number[] }>;
    }
    const parsedNodes: InferNodeEntry[] = [];

    for (const gf of graphFields) {
      if (gf.fn === 5 && gf.wt === 2) {
        const init = parseInitializerStaticMeta(bytes, gf, decoder);
        const nm = init?.name ?? getInitializerName(bytes, gf.valStart, gf.valEnd, decoder);
        if (nm) {
          if (seenInitializers.has(nm)) duplicateInitializers.push(nm);
          seenInitializers.add(nm);
        }
        if (init) {
          tensorMeta.set(init.name, init.meta);
        }
        continue;
      }
      if (gf.fn === 11 && gf.wt === 2) {
        const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
        if (vi && vi.elemType && vi.dims) {
          tensorMeta.set(vi.name, { elemType: vi.elemType, dims: vi.dims });
        }
        continue;
      }
      if (gf.fn !== 1 || gf.wt !== 2) continue;

      const nodeFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
      if (!nodeFields) return empty;
      const inputs: string[] = [];
      const outputs: string[] = [];
      let name = '';
      let opType = '';
      const attrs = new Map<string, { i?: number; ints?: number[] }>();

      for (const nf of nodeFields) {
        if (nf.fn === 1 && nf.wt === 2) {
          inputs.push(
            nf.valEnd > nf.valStart ? decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)) : ''
          );
        } else if (nf.fn === 2 && nf.wt === 2) {
          const outName =
            nf.valEnd > nf.valStart ? decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)) : '';
          outputs.push(outName);
          if (outName) {
            if (seenOutputs.has(outName)) duplicateOutputs.push(outName);
            seenOutputs.add(outName);
          }
        } else if (nf.fn === 3 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          name = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
          if (seenNodeNames.has(name)) duplicateNodeNames.push(name);
          seenNodeNames.add(name);
        } else if (nf.fn === 4 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          opType = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
          if (opType === 'If' || opType === 'Loop' || opType === 'Scan') {
            hasControlFlowNode = true;
          }
        } else if (nf.fn === 5 && nf.wt === 2) {
          const af = parseProtoFields(bytes, nf.valStart, nf.valEnd);
          if (!af) continue;
          let an = '';
          let ai: number | undefined;
          const aints: number[] = [];
          for (const a of af) {
            if (a.fn === 1 && a.wt === 2 && a.valEnd > a.valStart) {
              an = decoder.decode(bytes.subarray(a.valStart, a.valEnd));
            } else if (a.fn === 3 && a.wt === 0) {
              const v = readProtoVarintBigInt(bytes, a.valStart, a.valEnd);
              if (v) ai = Number(BigInt.asIntN(64, v[0]));
            } else if (a.fn === 8 && a.wt === 0) {
              const v = readProtoVarintBigInt(bytes, a.valStart, a.valEnd);
              if (v) aints.push(Number(BigInt.asIntN(64, v[0])));
            }
          }
          if (an) attrs.set(an, { i: ai, ints: aints });
        }
      }
      parsedNodes.push({ opType, name, inputs, outputs, attrs });
    }

    const validation = validateOnnxGraphNodeInputs(bytes);

    for (const node of parsedNodes) {
      const ins = node.inputs.map((inp) => (inp ? tensorMeta.get(inp) : undefined));
      const m0 = ins[0];
      if (!m0) continue;

      let outMetas: StaticTensorMeta[] | null = null;
      switch (node.opType) {
        case 'Identity':
        case 'LeakyRelu':
        case 'Relu':
        case 'Sigmoid':
        case 'Tanh':
        case 'Sqrt':
        case 'Clip':
        case 'Abs':
        case 'Floor':
        case 'Neg':
          outMetas = [{ elemType: m0.elemType, dims: [...m0.dims], intVals: m0.intVals }];
          break;
        case 'Cast': {
          const to = node.attrs.get('to')?.i;
          if (to) outMetas = [{ elemType: to, dims: [...m0.dims], intVals: m0.intVals }];
          break;
        }
        case 'Add':
        case 'Sub':
        case 'Mul':
        case 'Div':
        case 'Min':
        case 'Max': {
          const m1 = ins[1];
          if (m1) {
            const d = broadcastStaticDims(m0.dims, m1.dims);
            if (d) {
              let intVals: number[] | undefined;
              if (
                node.opType !== 'Div' &&
                m0.intVals &&
                m1.intVals &&
                (d.length === 0 || (d.length === 1 && d[0] <= 16))
              ) {
                const len = d.length === 0 ? 1 : d[0];
                intVals = Array.from({ length: len }, (_, i) => {
                  const va = m0.intVals![m0.intVals!.length === 1 ? 0 : i];
                  const vb = m1.intVals![m1.intVals!.length === 1 ? 0 : i];
                  if (node.opType === 'Add') return va + vb;
                  if (node.opType === 'Sub') return va - vb;
                  if (node.opType === 'Mul') return va * vb;
                  if (node.opType === 'Min') return Math.min(va, vb);
                  return Math.max(va, vb);
                });
              }
              outMetas = [{ elemType: m0.elemType, dims: d, intVals }];
            }
          }
          break;
        }
        case 'Transpose': {
          const perm = node.attrs.get('perm')?.ints;
          if (perm && perm.length === m0.dims.length) {
            outMetas = [{ elemType: m0.elemType, dims: perm.map((p) => m0.dims[p]) }];
          }
          break;
        }
        case 'Reshape': {
          const m1 = ins[1];
          if (m1?.intVals) {
            const inTotal = m0.dims.reduce((a, c) => a * c, 1);
            const target = m1.intVals.map((d, idx) => (d === 0 ? m0.dims[idx] : d));
            const negIdx = target.indexOf(-1);
            if (negIdx >= 0) {
              const knownProd = target.reduce(
                (a, c, idx) => (idx === negIdx ? a : a * c),
                1
              );
              if (knownProd > 0 && inTotal % knownProd === 0) {
                target[negIdx] = inTotal / knownProd;
              }
            }
            if (target.every((d) => d > 0)) {
              outMetas = [{ elemType: m0.elemType, dims: target, intVals: m0.intVals }];
            }
          }
          break;
        }
        case 'Shape':
          outMetas = [{ elemType: 7, dims: [m0.dims.length], intVals: [...m0.dims] }];
          break;
        case 'Gather': {
          const axisRaw = node.attrs.get('axis')?.i ?? 0;
          const axis = axisRaw < 0 ? m0.dims.length + axisRaw : axisRaw;
          const m1 = ins[1];
          if (axis === 0 && m1 && m1.dims.length <= 1 && m1.intVals && m1.intVals.length === 1) {
            const idx0 = m1.intVals[0] < 0 ? m0.dims[0] + m1.intVals[0] : m1.intVals[0];
            if (idx0 >= 0 && idx0 < m0.dims[0]) {
              const val = m0.intVals ? m0.intVals[idx0] : undefined;
              outMetas = [
                {
                  elemType: m0.elemType,
                  dims: [...m1.dims, ...m0.dims.slice(1)],
                  intVals: val !== undefined ? [val] : undefined,
                },
              ];
            }
          }
          break;
        }
        case 'Concat': {
          const axRaw = node.attrs.get('axis')?.i ?? 0;
          const ax = axRaw < 0 ? m0.dims.length + axRaw : axRaw;
          if (ax >= 0 && ax < m0.dims.length && ins.every((x) => Boolean(x))) {
            const outDims = [...m0.dims];
            let sum = 0;
            for (const inp of ins) sum += inp!.dims[ax];
            outDims[ax] = sum;
            const intVals =
              outDims.length === 1 && ins.every((x) => Array.isArray(x?.intVals))
                ? ins.flatMap((x) => x!.intVals!)
                : undefined;
            outMetas = [{ elemType: m0.elemType, dims: outDims, intVals }];
          }
          break;
        }
        case 'Unsqueeze': {
          const axes = node.attrs.get('axes')?.ints ?? ins[1]?.intVals;
          if (axes) {
            const outRank = m0.dims.length + axes.length;
            const normAxes = axes
              .map((a) => (a < 0 ? outRank + a : a))
              .sort((x, y) => x - y);
            const outDims: number[] = [];
            let j = 0;
            for (let i = 0; i < outRank; i++) {
              if (normAxes.includes(i)) outDims.push(1);
              else outDims.push(m0.dims[j++]);
            }
            outMetas = [{ elemType: m0.elemType, dims: outDims }];
          }
          break;
        }
        case 'Squeeze': {
          const axes = node.attrs.get('axes')?.ints ?? ins[1]?.intVals ?? [];
          const normAxes = new Set(axes.map((a) => (a < 0 ? m0.dims.length + a : a)));
          outMetas = [
            {
              elemType: m0.elemType,
              dims: m0.dims.filter((d, i) => (normAxes.size > 0 ? !normAxes.has(i) : d !== 1)),
            },
          ];
          break;
        }
        case 'ReduceMin':
        case 'ReduceSum':
        case 'ArgMin': {
          const keepdims = node.attrs.get('keepdims')?.i ?? 1;
          const axes = node.attrs.get('axes')?.ints ?? [node.attrs.get('axis')?.i ?? 0];
          const normAxes = new Set(axes.map((a) => (a < 0 ? m0.dims.length + a : a)));
          const outDims: number[] = [];
          m0.dims.forEach((d, i) => {
            if (normAxes.has(i)) {
              if (keepdims) outDims.push(1);
            } else {
              outDims.push(d);
            }
          });
          outMetas = [{ elemType: node.opType === 'ArgMin' ? 7 : m0.elemType, dims: outDims }];
          break;
        }
        case 'Tile': {
          const m1 = ins[1];
          if (m1?.intVals && m1.intVals.length === m0.dims.length) {
            outMetas = [
              {
                elemType: m0.elemType,
                dims: m0.dims.map((d, i) => d * m1.intVals![i]),
              },
            ];
          }
          break;
        }
        case 'Expand': {
          const m1 = ins[1];
          if (m1?.intVals) {
            const d = broadcastStaticDims(m0.dims, m1.intVals);
            if (d) outMetas = [{ elemType: m0.elemType, dims: d }];
          }
          break;
        }
        case 'Pad': {
          const padsAttr = node.attrs.get('pads')?.ints;
          const padsVals = ins[1]?.intVals ?? padsAttr;
          if (padsVals && padsVals.length === m0.dims.length * 2) {
            outMetas = [
              {
                elemType: m0.elemType,
                dims: m0.dims.map((d, i) => d + padsVals[i] + padsVals[i + m0.dims.length]),
              },
            ];
          }
          break;
        }
        case 'Slice': {
          const ms = ins[1]?.intVals;
          const me = ins[2]?.intVals;
          const ma = ins[3]?.intVals ?? Array.from({ length: ms?.length ?? 0 }, (_, i) => i);
          const mst = ins[4]?.intVals ?? Array.from({ length: ms?.length ?? 0 }, () => 1);
          if (ms && me && ms.length === me.length && ma.length === ms.length) {
            const outDims = [...m0.dims];
            let ok = true;
            for (let i = 0; i < ma.length; i++) {
              const ax = ma[i] < 0 ? m0.dims.length + ma[i] : ma[i];
              if (ax < 0 || ax >= m0.dims.length) {
                ok = false;
                break;
              }
              const dim = m0.dims[ax];
              const step = mst[i];
              if (step === 0) {
                ok = false;
                break;
              }
              let s = ms[i] < 0 ? dim + ms[i] : ms[i];
              let e = me[i] < 0 ? dim + me[i] : me[i];
              if (step > 0) {
                s = Math.max(0, Math.min(dim, s));
                e = Math.max(0, Math.min(dim, e));
                outDims[ax] = e <= s ? 0 : Math.ceil((e - s) / step);
              } else {
                s = Math.max(0, Math.min(dim - 1, s));
                e = Math.max(-1, Math.min(dim - 1, e));
                outDims[ax] = s <= e ? 0 : Math.ceil((s - e) / -step);
              }
            }
            if (ok) outMetas = [{ elemType: m0.elemType, dims: outDims }];
          }
          break;
        }
        case 'Conv': {
          const w = ins[1];
          if (w && m0.dims.length === 4 && w.dims.length === 4) {
            const ks = node.attrs.get('kernel_shape')?.ints ?? [w.dims[2], w.dims[3]];
            const st = node.attrs.get('strides')?.ints ?? [1, 1];
            const dil = node.attrs.get('dilations')?.ints ?? [1, 1];
            const pads = node.attrs.get('pads')?.ints ?? [0, 0, 0, 0];
            const oH =
              Math.floor((m0.dims[2] + pads[0] + pads[2] - dil[0] * (ks[0] - 1) - 1) / st[0]) +
              1;
            const oW =
              Math.floor((m0.dims[3] + pads[1] + pads[3] - dil[1] * (ks[1] - 1) - 1) / st[1]) +
              1;
            outMetas = [{ elemType: m0.elemType, dims: [m0.dims[0], w.dims[0], oH, oW] }];
          }
          break;
        }
        case 'ConvTranspose': {
          const w = ins[1];
          if (w && m0.dims.length === 4 && w.dims.length === 4) {
            const ks = node.attrs.get('kernel_shape')?.ints ?? [w.dims[2], w.dims[3]];
            const st = node.attrs.get('strides')?.ints ?? [1, 1];
            const dil = node.attrs.get('dilations')?.ints ?? [1, 1];
            const pads = node.attrs.get('pads')?.ints ?? [0, 0, 0, 0];
            const opad = node.attrs.get('output_padding')?.ints ?? [0, 0];
            const g = node.attrs.get('group')?.i ?? 1;
            const outC = w.dims[1] * g;
            const oH =
              st[0] * (m0.dims[2] - 1) + ((ks[0] - 1) * dil[0] + 1) - pads[0] - pads[2] + opad[0];
            const oW =
              st[1] * (m0.dims[3] - 1) + ((ks[1] - 1) * dil[1] + 1) - pads[1] - pads[3] + opad[1];
            outMetas = [{ elemType: m0.elemType, dims: [m0.dims[0], outC, oH, oW] }];
          } else if (w && m0.dims.length === 5 && w.dims.length === 5) {
            const ks = node.attrs.get('kernel_shape')?.ints ?? [w.dims[2], w.dims[3], w.dims[4]];
            const st = node.attrs.get('strides')?.ints ?? [1, 1, 1];
            const dil = node.attrs.get('dilations')?.ints ?? [1, 1, 1];
            const pads = node.attrs.get('pads')?.ints ?? [0, 0, 0, 0, 0, 0];
            const opad = node.attrs.get('output_padding')?.ints ?? [0, 0, 0];
            const g = node.attrs.get('group')?.i ?? 1;
            const outC = w.dims[1] * g;
            const oD =
              st[0] * (m0.dims[2] - 1) + ((ks[0] - 1) * dil[0] + 1) - pads[0] - pads[3] + opad[0];
            const oH =
              st[1] * (m0.dims[3] - 1) + ((ks[1] - 1) * dil[1] + 1) - pads[1] - pads[4] + opad[1];
            const oW =
              st[2] * (m0.dims[4] - 1) + ((ks[2] - 1) * dil[2] + 1) - pads[2] - pads[5] + opad[2];
            outMetas = [{ elemType: m0.elemType, dims: [m0.dims[0], outC, oD, oH, oW] }];
          }
          break;
        }
        case 'Split': {
          const axRaw = node.attrs.get('axis')?.i ?? 0;
          const ax = axRaw < 0 ? m0.dims.length + axRaw : axRaw;
          const splits =
            node.attrs.get('split')?.ints ??
            Array.from({ length: node.outputs.length }, () =>
              Math.floor(m0.dims[ax] / node.outputs.length)
            );
          outMetas = splits.map((sp) => {
            const d = [...m0.dims];
            d[ax] = sp;
            return { elemType: m0.elemType, dims: d };
          });
          break;
        }
        case 'GatherND': {
          const m1 = ins[1];
          const bdims = node.attrs.get('batch_dims')?.i ?? 0;
          if (m1 && m1.dims.length >= 1 && bdims >= 0 && bdims < m1.dims.length) {
            const k = m1.dims[m1.dims.length - 1];
            if (bdims + k <= m0.dims.length) {
              outMetas = [
                {
                  elemType: m0.elemType,
                  dims: [...m1.dims.slice(0, -1), ...m0.dims.slice(bdims + k)],
                },
              ];
            }
          }
          break;
        }
      }

      if (outMetas && outMetas.length === node.outputs.length) {
        for (let i = 0; i < node.outputs.length; i++) {
          if (node.outputs[i]) {
            tensorMeta.set(node.outputs[i], outMetas[i]);
          }
        }
      }
    }

    const record: Record<string, StaticTensorMeta> = {};
    for (const [k, v] of tensorMeta.entries()) {
      record[k] = {
        elemType: v.elemType,
        dims: [...v.dims],
        ...(v.intVals ? { intVals: [...v.intVals] } : {}),
      };
    }

    return {
      valid:
        validation.valid &&
        duplicateOutputs.length === 0 &&
        duplicateInitializers.length === 0 &&
        duplicateNodeNames.length === 0 &&
        !hasControlFlowNode,
      irVersion,
      defaultOpset,
      hasControlFlowNode,
      tensorMeta: record,
      duplicateOutputs,
      duplicateInitializers,
      duplicateNodeNames,
      missingInputs: validation.missingInputs,
    };
  } catch {
    return empty;
  }
}

/**
 * Guarded H1 pass built on top of F1 for `hitnet-sf` WebGPU:
 * Normalizes ONLY the two `prop0` `GatherND` nodes (#218 and #506) whose `data` and `indices`
 * are rank-5 with `batch_dims=3`:
 * - Merges the two spatial batch axes `(60, 80) -> 4800`:
 *   - `data`: `float32 [1,60,80,1,16]` -> `PreReshape` -> `float32 [1,4800,1,16]` (rank 4)
 *   - `indices`: `int64 [1,60,80,1,1]` -> `PreReshape` -> `int64 [1,4800,1,1]` (rank 4)
 *   - `GatherND`: `batch_dims=3` -> `batch_dims=2` -> `float32 [1,4800,1,16]` (rank 4)
 *   - `PostReshape`: restores exact original output name and shape `[1,60,80,1,16]`
 *
 * Strictly opt-in and atomic fail-closed: preserves all integer index values/offsets, leaves
 * the other 8 `GatherND` nodes untouched, validates unique names & full graph structure, and
 * aborts returning untouched bytes if any guard fails.
 */
export function normalizeHitnetPropGatherNdRankForWebGpu(
  bytes: Uint8Array,
  options: NormalizeGatherNdRankOptions = {}
): OnnxGatherNdRankNormalizeResult {
  const emptyResult: OnnxGatherNdRankNormalizeResult = {
    bytes,
    normalizedCount: 0,
    normalizedNodeNames: [],
    skippedDynamicCount: 0,
    skippedInvalidOrUnprovenCount: 0,
    totalGatherNdCount: 0,
  };
  const requireHitnetContract = options.requireHitnetContract ?? true;
  if (
    (options.backendPreference === 'wasm' && !options.allowDiagnosticTransformedWasm) ||
    bytes.byteLength < 16
  ) {
    return emptyResult;
  }
  if (requireHitnetContract && options.variantId !== 'hitnet-sf') {
    return emptyResult;
  }

  try {
    const preInference = inferOnnxStaticTensorMeta(bytes);
    if (
      !preInference.valid ||
      preInference.hasControlFlowNode ||
      preInference.duplicateOutputs.length > 0 ||
      preInference.duplicateInitializers.length > 0 ||
      preInference.duplicateNodeNames.length > 0
    ) {
      return emptyResult;
    }

    const topFields = parseProtoFields(bytes, 0, bytes.byteLength);
    if (!topFields) return emptyResult;
    const decoder = new TextDecoder();
    if (requireHitnetContract) {
      if (preInference.irVersion !== 7 || preInference.defaultOpset !== 12) {
        return emptyResult;
      }
    } else if (
      preInference.defaultOpset !== null &&
      (preInference.defaultOpset < 11 || preInference.defaultOpset > 21)
    ) {
      return emptyResult;
    }

    const graphIndex = topFields.findIndex((f) => f.fn === 7 && f.wt === 2);
    if (graphIndex < 0) return emptyResult;
    const graphField = topFields[graphIndex];
    const graphFields = parseProtoFields(bytes, graphField.valStart, graphField.valEnd);
    if (!graphFields) return emptyResult;

    const encoder = new TextEncoder();
    const usedNames = new Set<string>();
    const allInitializerNames = new Set<string>();
    const parsedGraphInputs: ParsedStaticValueInfo[] = [];
    const parsedGraphOutputs: ParsedStaticValueInfo[] = [];

    interface PropParsedNode {
      gf: ProtobufFieldSlice;
      nodeFields: ProtobufFieldSlice[];
      opType: string;
      name: string;
      inputs: string[];
      outputs: string[];
      attrs: Map<string, { i?: number; ints?: number[] }>;
    }
    const parsedNodes: PropParsedNode[] = [];

    for (const gf of graphFields) {
      if (gf.fn === 5 && gf.wt === 2) {
        const nm = getInitializerName(bytes, gf.valStart, gf.valEnd, decoder);
        if (nm) {
          usedNames.add(nm);
          allInitializerNames.add(nm);
        }
        continue;
      }
      if (gf.fn === 11 && gf.wt === 2) {
        const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
        if (vi) {
          usedNames.add(vi.name);
          parsedGraphInputs.push(vi);
        }
        continue;
      }
      if (gf.fn === 12 && gf.wt === 2) {
        const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
        if (vi) {
          usedNames.add(vi.name);
          parsedGraphOutputs.push(vi);
        }
        continue;
      }
      if (gf.fn === 13 && gf.wt === 2) {
        const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
        if (vi) usedNames.add(vi.name);
        continue;
      }
      if (gf.fn !== 1 || gf.wt !== 2) continue;

      const nodeFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
      if (!nodeFields) return emptyResult;
      const inputs: string[] = [];
      const outputs: string[] = [];
      let name = '';
      let opType = '';
      const attrs = new Map<string, { i?: number; ints?: number[] }>();

      for (const nf of nodeFields) {
        if (nf.fn === 1 && nf.wt === 2) {
          const inp =
            nf.valEnd > nf.valStart ? decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)) : '';
          inputs.push(inp);
          if (inp) usedNames.add(inp);
        } else if (nf.fn === 2 && nf.wt === 2) {
          const outName =
            nf.valEnd > nf.valStart ? decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)) : '';
          outputs.push(outName);
          if (outName) usedNames.add(outName);
        } else if (nf.fn === 3 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          name = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
          usedNames.add(name);
        } else if (nf.fn === 4 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          opType = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
        } else if (nf.fn === 5 && nf.wt === 2) {
          const af = parseProtoFields(bytes, nf.valStart, nf.valEnd);
          if (!af) continue;
          let an = '';
          let ai: number | undefined;
          const aints: number[] = [];
          for (const a of af) {
            if (a.fn === 1 && a.wt === 2 && a.valEnd > a.valStart) {
              an = decoder.decode(bytes.subarray(a.valStart, a.valEnd));
            } else if (a.fn === 3 && a.wt === 0) {
              const v = readProtoVarintBigInt(bytes, a.valStart, a.valEnd);
              if (v) ai = Number(BigInt.asIntN(64, v[0]));
            } else if (a.fn === 8 && a.wt === 0) {
              const v = readProtoVarintBigInt(bytes, a.valStart, a.valEnd);
              if (v) aints.push(Number(BigInt.asIntN(64, v[0])));
            }
          }
          if (an) attrs.set(an, { i: ai, ints: aints });
        }
      }
      parsedNodes.push({ gf, nodeFields, opType, name, inputs, outputs, attrs });
    }

    const nonInitInputs = parsedGraphInputs.filter((g) => !allInitializerNames.has(g.name));
    if (
      nonInitInputs.length === 0 ||
      nonInitInputs.some((g) => g.dims === null || g.elemType === null)
    ) {
      return emptyResult;
    }

    if (requireHitnetContract) {
      if (
        nonInitInputs.length !== 1 ||
        nonInitInputs[0].name !== 'input' ||
        nonInitInputs[0].elemType !== 1 ||
        !nonInitInputs[0].dims ||
        !sameStaticDims(nonInitInputs[0].dims, [1, 6, 240, 320])
      ) {
        return emptyResult;
      }
      const outNames = parsedGraphOutputs.map((o) => o.name);
      if (
        outNames.length !== 2 ||
        !outNames.includes('reference_output_disparity') ||
        !outNames.includes('secondary_output_disparity')
      ) {
        return emptyResult;
      }
      const refOutMeta = preInference.tensorMeta['reference_output_disparity'];
      const secOutMeta = preInference.tensorMeta['secondary_output_disparity'];
      if (
        !refOutMeta ||
        !secOutMeta ||
        refOutMeta.elemType !== 1 ||
        secOutMeta.elemType !== 1 ||
        !sameStaticDims(refOutMeta.dims, [1, 240, 320, 1]) ||
        !sameStaticDims(secOutMeta.dims, [1, 240, 320, 1])
      ) {
        return emptyResult;
      }
    }

    let tempSeq = 0;
    const allocName = (prefix: string): string => {
      let candidate = `${prefix}_${tempSeq++}`;
      while (usedNames.has(candidate)) {
        candidate = `${prefix}_${tempSeq++}`;
      }
      usedNames.add(candidate);
      return candidate;
    };

    const concatBytes = (parts: readonly Uint8Array[]): Uint8Array => {
      let total = 0;
      for (const p of parts) total += p.byteLength;
      const out = new Uint8Array(total);
      let off = 0;
      for (const p of parts) {
        out.set(p, off);
        off += p.byteLength;
      }
      return out;
    };

    const encodeLenField = (fn: number, payload: Uint8Array): Uint8Array => {
      const tag = encodeProtoVarint((fn << 3) | 2);
      const len = encodeProtoVarint(payload.byteLength);
      const out = new Uint8Array(tag.byteLength + len.byteLength + payload.byteLength);
      out.set(tag, 0);
      out.set(len, tag.byteLength);
      out.set(payload, tag.byteLength + len.byteLength);
      return out;
    };

    const encodeStrField = (fn: number, str: string): Uint8Array =>
      encodeLenField(fn, encoder.encode(str));

    const encodeVarintField = (fn: number, val: number): Uint8Array =>
      concatBytes([encodeProtoVarint((fn << 3) | 0), encodeProtoVarint(val)]);

    const encodeAttrInt = (attrName: string, val: number): Uint8Array =>
      encodeLenField(
        5,
        concatBytes([
          encodeStrField(1, attrName),
          concatBytes([encodeProtoVarint((3 << 3) | 0), encodeSignedVarint64Bytes(val)]),
          encodeVarintField(20, 2),
        ])
      );

    const encodeNodeSimple = (
      opType: string,
      inputs: readonly string[],
      outputs: readonly string[],
      name: string
    ): Uint8Array => {
      const parts: Uint8Array[] = [];
      for (const inp of inputs) parts.push(encodeStrField(1, inp));
      for (const out of outputs) parts.push(encodeStrField(2, out));
      parts.push(encodeStrField(3, name));
      parts.push(encodeStrField(4, opType));
      return encodeLenField(1, concatBytes(parts));
    };

    const encodeInt64ShapeInitializer = (name: string, shapeDims: readonly number[]): Uint8Array => {
      const raw = new Uint8Array(shapeDims.length * 8);
      const dv = new DataView(raw.buffer);
      for (let i = 0; i < shapeDims.length; i++) {
        dv.setBigInt64(i * 8, BigInt(shapeDims[i]), true);
      }
      return encodeLenField(
        5,
        concatBytes([
          encodeVarintField(1, shapeDims.length),
          encodeVarintField(2, 7),
          encodeStrField(8, name),
          encodeLenField(9, raw),
        ])
      );
    };

    const addedInitializers: Uint8Array[] = [];
    const shapeInitCache = new Map<string, string>();
    const getOrCreateInt64ShapeInit = (rolePrefix: string, shapeDims: readonly number[]): string => {
      const key = shapeDims.join(',');
      const existing = shapeInitCache.get(key);
      if (existing) return existing;
      const name = allocName(`__webgpu_h1_prop_${rolePrefix}`);
      shapeInitCache.set(key, name);
      addedInitializers.push(encodeInt64ShapeInitializer(name, shapeDims));
      return name;
    };

    const nodeReplacementMap = new Map<number, Uint8Array[]>();
    const normalizedNodeNames: string[] = [];
    const expectedAfterMetaByOutName = new Map<string, { elemType: number; outDims: number[] }>();
    let totalGatherNdCount = 0;
    let skippedDynamicCount = 0;
    let skippedInvalidOrUnprovenCount = 0;

    for (let nodeIdx = 0; nodeIdx < parsedNodes.length; nodeIdx++) {
      const node = parsedNodes[nodeIdx];
      if (node.opType !== 'GatherND') continue;
      totalGatherNdCount++;

      if (node.inputs.length !== 2 || node.outputs.length !== 1 || !node.outputs[0]) {
        skippedInvalidOrUnprovenCount++;
        continue;
      }

      const dataMeta = node.inputs[0] ? preInference.tensorMeta[node.inputs[0]] : undefined;
      const idxMeta = node.inputs[1] ? preInference.tensorMeta[node.inputs[1]] : undefined;
      const outMeta = preInference.tensorMeta[node.outputs[0]];
      if (!dataMeta || !idxMeta || !outMeta) {
        skippedDynamicCount++;
        continue;
      }

      const batchDims = node.attrs.get('batch_dims')?.i ?? 0;

      if (requireHitnetContract) {
        const isL0Prop =
          node.name === 'level0/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis' &&
          node.inputs[0] === 'level0/shared/prop0/add' &&
          node.inputs[1] === 'Unsqueeze__675:0' &&
          node.outputs[0] === 'level0/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis';
        const isL01Prop =
          node.name === 'level0_1/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis' &&
          node.inputs[0] === 'level0_1/shared/prop0/add' &&
          node.inputs[1] === 'Unsqueeze__387:0' &&
          node.outputs[0] === 'level0_1/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis';

        if (!isL0Prop && !isL01Prop) {
          // Verify other GatherND nodes: allow level_init & warp_1d nodes untouched,
          // fail closed if another unexpected rank-5 data GatherND exists.
          if (dataMeta.dims.length >= 5) {
            skippedInvalidOrUnprovenCount++;
          }
          continue;
        }

        const matchesExactPropContract =
          batchDims === 3 &&
          dataMeta.elemType === 1 &&
          idxMeta.elemType === 7 &&
          outMeta.elemType === 1 &&
          sameStaticDims(dataMeta.dims, [1, 60, 80, 1, 16]) &&
          sameStaticDims(idxMeta.dims, [1, 60, 80, 1, 1]) &&
          sameStaticDims(outMeta.dims, [1, 60, 80, 1, 16]);
        if (!matchesExactPropContract) {
          skippedInvalidOrUnprovenCount++;
          continue;
        }
      } else {
        if (
          batchDims !== 3 ||
          dataMeta.elemType !== 1 ||
          idxMeta.elemType !== 7 ||
          outMeta.elemType !== 1 ||
          dataMeta.dims.length !== 5 ||
          idxMeta.dims.length !== 5 ||
          outMeta.dims.length !== 5 ||
          idxMeta.dims[4] !== 1 ||
          dataMeta.dims[0] !== idxMeta.dims[0] ||
          dataMeta.dims[1] !== idxMeta.dims[1] ||
          dataMeta.dims[2] !== idxMeta.dims[2]
        ) {
          continue;
        }
      }

      const b0 = dataMeta.dims[0];
      const mergedBatch = dataMeta.dims[1] * dataMeta.dims[2];
      if (
        !Number.isSafeInteger(b0) ||
        b0 <= 0 ||
        !Number.isSafeInteger(mergedBatch) ||
        mergedBatch <= 0
      ) {
        skippedInvalidOrUnprovenCount++;
        continue;
      }

      const mergedDataDims = [b0, mergedBatch, dataMeta.dims[3], dataMeta.dims[4]];
      const mergedIdxDims = [b0, mergedBatch, idxMeta.dims[3], idxMeta.dims[4]];
      const origOutDims = [...outMeta.dims];
      const mergedOutDims = [b0, mergedBatch, ...origOutDims.slice(3)];

      const origNodeName = node.name || `GatherND_${nodeIdx}`;
      const dataShapeInit = getOrCreateInt64ShapeInit('data_shape', mergedDataDims);
      const idxShapeInit = getOrCreateInt64ShapeInit('idx_shape', mergedIdxDims);
      const outShapeInit = getOrCreateInt64ShapeInit('out_shape', origOutDims);

      const mergedDataName = allocName(`${origNodeName}__h1_data_4d`);
      const mergedIdxName = allocName(`${origNodeName}__h1_indices_4d`);
      const mergedOutName = allocName(`${origNodeName}__h1_out_4d`);

      // Rebuild GatherND preserving exact node name, domain, and non-batch_dims attributes,
      // switching batch_dims from 3 to 2 and wiring merged 4D inputs & outputs.
      const rewrittenNodeParts: Uint8Array[] = [];
      let inSlot = 0;
      let outSlot = 0;
      let replacedBatchDims = false;
      for (const nf of node.nodeFields) {
        if (nf.fn === 1 && nf.wt === 2) {
          rewrittenNodeParts.push(
            inSlot === 0
              ? encodeStrField(1, mergedDataName)
              : inSlot === 1
                ? encodeStrField(1, mergedIdxName)
                : bytes.subarray(nf.fStart, nf.fEnd)
          );
          inSlot++;
        } else if (nf.fn === 2 && nf.wt === 2) {
          rewrittenNodeParts.push(
            outSlot === 0 ? encodeStrField(2, mergedOutName) : bytes.subarray(nf.fStart, nf.fEnd)
          );
          outSlot++;
        } else if (nf.fn === 5 && nf.wt === 2) {
          const af = parseProtoFields(bytes, nf.valStart, nf.valEnd);
          const nameF = af?.find((a) => a.fn === 1 && a.wt === 2 && a.valEnd > a.valStart);
          const attrName = nameF ? decoder.decode(bytes.subarray(nameF.valStart, nameF.valEnd)) : '';
          if (attrName === 'batch_dims') {
            rewrittenNodeParts.push(encodeAttrInt('batch_dims', 2));
            replacedBatchDims = true;
          } else {
            rewrittenNodeParts.push(bytes.subarray(nf.fStart, nf.fEnd));
          }
        } else {
          rewrittenNodeParts.push(bytes.subarray(nf.fStart, nf.fEnd));
        }
      }
      if (!replacedBatchDims) {
        rewrittenNodeParts.push(encodeAttrInt('batch_dims', 2));
      }

      nodeReplacementMap.set(nodeIdx, [
        encodeNodeSimple(
          'Reshape',
          [node.inputs[0], dataShapeInit],
          [mergedDataName],
          allocName(`${origNodeName}__H1PreReshapeData`)
        ),
        encodeNodeSimple(
          'Reshape',
          [node.inputs[1], idxShapeInit],
          [mergedIdxName],
          allocName(`${origNodeName}__H1PreReshapeIndices`)
        ),
        encodeLenField(1, concatBytes(rewrittenNodeParts)),
        encodeNodeSimple(
          'Reshape',
          [mergedOutName, outShapeInit],
          [node.outputs[0]],
          allocName(`${origNodeName}__H1PostReshapeOut`)
        ),
      ]);
      expectedAfterMetaByOutName.set(mergedDataName, {
        elemType: 1,
        outDims: mergedDataDims,
      });
      expectedAfterMetaByOutName.set(mergedIdxName, {
        elemType: 7,
        outDims: mergedIdxDims,
      });
      expectedAfterMetaByOutName.set(mergedOutName, {
        elemType: 1,
        outDims: mergedOutDims,
      });
      expectedAfterMetaByOutName.set(node.outputs[0], {
        elemType: 1,
        outDims: origOutDims,
      });
      normalizedNodeNames.push(origNodeName);
    }

    if (requireHitnetContract) {
      const hasL0 = normalizedNodeNames.includes(
        'level0/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis'
      );
      const hasL01 = normalizedNodeNames.includes(
        'level0_1/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis'
      );
      if (
        totalGatherNdCount !== 10 ||
        normalizedNodeNames.length !== 2 ||
        !hasL0 ||
        !hasL01 ||
        skippedDynamicCount > 0 ||
        skippedInvalidOrUnprovenCount > 0
      ) {
        return emptyResult;
      }
    }

    if (normalizedNodeNames.length === 0) {
      return {
        bytes,
        normalizedCount: 0,
        normalizedNodeNames: [],
        skippedDynamicCount,
        skippedInvalidOrUnprovenCount,
        totalGatherNdCount,
      };
    }

    const finalGraphParts: Uint8Array[] = [];
    let currentNodeIdx = 0;
    for (const gf of graphFields) {
      if (gf.fn === 13) continue; // Drop stale value_info entries
      if (gf.fn === 1 && gf.wt === 2) {
        const replacement = nodeReplacementMap.get(currentNodeIdx);
        currentNodeIdx++;
        if (replacement) {
          for (const part of replacement) {
            finalGraphParts.push(part);
          }
        } else {
          finalGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        }
      } else {
        finalGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
      }
    }
    for (const initPart of addedInitializers) {
      finalGraphParts.push(initPart);
    }

    let graphPayloadLen = 0;
    for (const part of finalGraphParts) graphPayloadLen += part.byteLength;

    const graphTag = encodeProtoVarint((7 << 3) | 2);
    const graphLen = encodeProtoVarint(graphPayloadLen);
    let totalLen = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        totalLen += graphTag.byteLength + graphLen.byteLength + graphPayloadLen;
      } else {
        totalLen += topFields[i].fEnd - topFields[i].fStart;
      }
    }

    const out = new Uint8Array(totalLen);
    let offset = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        out.set(graphTag, offset);
        offset += graphTag.byteLength;
        out.set(graphLen, offset);
        offset += graphLen.byteLength;
        for (const part of finalGraphParts) {
          out.set(part, offset);
          offset += part.byteLength;
        }
      } else {
        const chunk = bytes.subarray(topFields[i].fStart, topFields[i].fEnd);
        out.set(chunk, offset);
        offset += chunk.byteLength;
      }
    }

    // Post-validate rewritten graph structure, unique names, and inferred tensor shapes atomically
    const beforeValidation = validateOnnxGraphNodeInputs(bytes);
    const afterValidation = validateOnnxGraphNodeInputs(out);
    const preexistingMissing = new Set(beforeValidation.missingInputs.map((m) => m.inputName));
    const newlyMissing = afterValidation.missingInputs.filter(
      (m) => !preexistingMissing.has(m.inputName)
    );
    if (
      newlyMissing.length > 0 ||
      (beforeValidation.hasGraphInputs && !afterValidation.valid)
    ) {
      return emptyResult;
    }

    const postInference = inferOnnxStaticTensorMeta(out);
    if (
      !postInference.valid ||
      postInference.duplicateOutputs.length > 0 ||
      postInference.duplicateInitializers.length > 0 ||
      postInference.duplicateNodeNames.length > 0
    ) {
      return emptyResult;
    }

    for (const [tensorName, exp] of expectedAfterMetaByOutName.entries()) {
      const m = postInference.tensorMeta[tensorName];
      if (!m || m.elemType !== exp.elemType || !sameStaticDims(m.dims, exp.outDims)) {
        return emptyResult;
      }
    }
    if (requireHitnetContract) {
      const refPost = postInference.tensorMeta['reference_output_disparity'];
      const secPost = postInference.tensorMeta['secondary_output_disparity'];
      if (
        !refPost ||
        !secPost ||
        refPost.elemType !== 1 ||
        secPost.elemType !== 1 ||
        !sameStaticDims(refPost.dims, [1, 240, 320, 1]) ||
        !sameStaticDims(secPost.dims, [1, 240, 320, 1])
      ) {
        return emptyResult;
      }
    }

    return {
      bytes: out,
      normalizedCount: normalizedNodeNames.length,
      normalizedNodeNames,
      skippedDynamicCount,
      skippedInvalidOrUnprovenCount,
      totalGatherNdCount,
    };
  } catch {
    return emptyResult;
  }
}

interface ParsedLevelInitNode {
  nodeIdx: number;
  gf: ProtobufFieldSlice;
  nodeFields: ProtobufFieldSlice[];
  opType: string;
  name: string;
  inputs: string[];
  outputs: string[];
  attrs: Map<string, { i?: number; ints?: number[] }>;
}

function inspectLevelInitGatherContext(
  bytes: Uint8Array,
  requireHitnetContract: boolean
): {
  valid: boolean;
  topFields: ProtobufFieldSlice[];
  graphIndex: number;
  graphFields: ProtobufFieldSlice[];
  preInference: OnnxStaticInferenceResult;
  parsedNodes: ParsedLevelInitNode[];
  usedNames: Set<string>;
  producerByOutput: Map<string, ParsedLevelInitNode>;
  consumersByTensor: Map<string, Set<number>>;
  totalGatherNdCount: number;
  skippedDynamicCount: number;
  skippedInvalidOrUnprovenCount: number;
  targets: Array<{
    node: ParsedLevelInitNode;
    unsqueezeNode: ParsedLevelInitNode;
    expandNode: ParsedLevelInitNode;
    argMinNode: ParsedLevelInitNode;
    castNode: ParsedLevelInitNode | null;
    dataDims: [number, number, number, number];
    idxDims: [number, number, number, number, number];
    outDims: [number, number, number, number];
    hasNonNegativeSadCostLineage: boolean;
  }>;
} | null {
  const preInference = inferOnnxStaticTensorMeta(bytes);
  if (
    !preInference.valid ||
    preInference.hasControlFlowNode ||
    preInference.duplicateOutputs.length > 0 ||
    preInference.duplicateInitializers.length > 0 ||
    preInference.duplicateNodeNames.length > 0
  ) {
    return null;
  }

  if (requireHitnetContract) {
    if (preInference.irVersion !== 7 || preInference.defaultOpset !== 12) {
      return null;
    }
  } else if (
    preInference.defaultOpset !== null &&
    (preInference.defaultOpset < 11 || preInference.defaultOpset > 17)
  ) {
    return null;
  }

  const topFields = parseProtoFields(bytes, 0, bytes.byteLength);
  if (!topFields) return null;
  const graphIndex = topFields.findIndex((f) => f.fn === 7 && f.wt === 2);
  if (graphIndex < 0) return null;
  const graphField = topFields[graphIndex];
  const graphFields = parseProtoFields(bytes, graphField.valStart, graphField.valEnd);
  if (!graphFields) return null;

  const decoder = new TextDecoder();
  const usedNames = new Set<string>();
  const allInitializerNames = new Set<string>();
  const parsedGraphInputs: ParsedStaticValueInfo[] = [];
  const parsedGraphOutputs: ParsedStaticValueInfo[] = [];
  const parsedNodes: ParsedLevelInitNode[] = [];
  const producerByOutput = new Map<string, ParsedLevelInitNode>();
  const consumersByTensor = new Map<string, Set<number>>();

  for (const gf of graphFields) {
    if (gf.fn === 5 && gf.wt === 2) {
      const nm = getInitializerName(bytes, gf.valStart, gf.valEnd, decoder);
      if (nm) {
        usedNames.add(nm);
        allInitializerNames.add(nm);
      }
      continue;
    }
    if (gf.fn === 11 && gf.wt === 2) {
      const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
      if (vi) {
        usedNames.add(vi.name);
        parsedGraphInputs.push(vi);
      }
      continue;
    }
    if (gf.fn === 12 && gf.wt === 2) {
      const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
      if (vi) {
        usedNames.add(vi.name);
        parsedGraphOutputs.push(vi);
      }
      continue;
    }
    if (gf.fn === 13 && gf.wt === 2) {
      const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
      if (vi) usedNames.add(vi.name);
      continue;
    }
    if (gf.fn !== 1 || gf.wt !== 2) continue;

    const nodeFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
    if (!nodeFields) return null;
    const inputs: string[] = [];
    const outputs: string[] = [];
    let name = '';
    let opType = '';
    const attrs = new Map<string, { i?: number; ints?: number[] }>();

    for (const nf of nodeFields) {
      if (nf.fn === 1 && nf.wt === 2) {
        const inp =
          nf.valEnd > nf.valStart ? decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)) : '';
        inputs.push(inp);
        if (inp) usedNames.add(inp);
      } else if (nf.fn === 2 && nf.wt === 2) {
        const outName =
          nf.valEnd > nf.valStart ? decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)) : '';
        outputs.push(outName);
        if (outName) usedNames.add(outName);
      } else if (nf.fn === 3 && nf.wt === 2 && nf.valEnd > nf.valStart) {
        name = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
        usedNames.add(name);
      } else if (nf.fn === 4 && nf.wt === 2 && nf.valEnd > nf.valStart) {
        opType = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
      } else if (nf.fn === 5 && nf.wt === 2) {
        const af = parseProtoFields(bytes, nf.valStart, nf.valEnd);
        if (!af) continue;
        let an = '';
        let ai: number | undefined;
        const aints: number[] = [];
        for (const a of af) {
          if (a.fn === 1 && a.wt === 2 && a.valEnd > a.valStart) {
            an = decoder.decode(bytes.subarray(a.valStart, a.valEnd));
          } else if (a.fn === 3 && a.wt === 0) {
            const v = readProtoVarintBigInt(bytes, a.valStart, a.valEnd);
            if (v) ai = Number(BigInt.asIntN(64, v[0]));
          } else if (a.fn === 8 && a.wt === 0) {
            const v = readProtoVarintBigInt(bytes, a.valStart, a.valEnd);
            if (v) aints.push(Number(BigInt.asIntN(64, v[0])));
          }
        }
        if (an) attrs.set(an, { i: ai, ints: aints });
      }
    }

    const nodeIdx = parsedNodes.length;
    const entry: ParsedLevelInitNode = {
      nodeIdx,
      gf,
      nodeFields,
      opType,
      name,
      inputs,
      outputs,
      attrs,
    };
    parsedNodes.push(entry);
    for (const outName of outputs) {
      if (outName) producerByOutput.set(outName, entry);
    }
    for (const inpName of inputs) {
      if (!inpName) continue;
      let set = consumersByTensor.get(inpName);
      if (!set) {
        set = new Set<number>();
        consumersByTensor.set(inpName, set);
      }
      set.add(nodeIdx);
    }
  }

  const nonInitInputs = parsedGraphInputs.filter((g) => !allInitializerNames.has(g.name));
  if (
    nonInitInputs.length === 0 ||
    nonInitInputs.some((g) => g.dims === null || g.elemType === null)
  ) {
    return null;
  }

  if (requireHitnetContract) {
    if (
      nonInitInputs.length !== 1 ||
      nonInitInputs[0].name !== 'input' ||
      nonInitInputs[0].elemType !== 1 ||
      !nonInitInputs[0].dims ||
      !sameStaticDims(nonInitInputs[0].dims, [1, 6, 240, 320])
    ) {
      return null;
    }
    const outNames = parsedGraphOutputs.map((o) => o.name);
    if (
      outNames.length !== 2 ||
      !outNames.includes('reference_output_disparity') ||
      !outNames.includes('secondary_output_disparity')
    ) {
      return null;
    }
    const refMeta = preInference.tensorMeta['reference_output_disparity'];
    const secMeta = preInference.tensorMeta['secondary_output_disparity'];
    if (
      !refMeta ||
      !secMeta ||
      refMeta.elemType !== 1 ||
      secMeta.elemType !== 1 ||
      !sameStaticDims(refMeta.dims, [1, 240, 320, 1]) ||
      !sameStaticDims(secMeta.dims, [1, 240, 320, 1])
    ) {
      return null;
    }
  }

  let totalGatherNdCount = 0;
  let skippedDynamicCount = 0;
  let skippedInvalidOrUnprovenCount = 0;
  const targets: NonNullable<ReturnType<typeof inspectLevelInitGatherContext>>['targets'] = [];

  for (const node of parsedNodes) {
    if (node.opType !== 'GatherND') continue;
    totalGatherNdCount++;

    if (node.inputs.length !== 2 || node.outputs.length !== 1 || !node.outputs[0]) {
      skippedInvalidOrUnprovenCount++;
      continue;
    }

    const dataMeta = node.inputs[0] ? preInference.tensorMeta[node.inputs[0]] : undefined;
    const idxMeta = node.inputs[1] ? preInference.tensorMeta[node.inputs[1]] : undefined;
    const outMeta = preInference.tensorMeta[node.outputs[0]];
    if (!dataMeta || !idxMeta || !outMeta) {
      skippedDynamicCount++;
      continue;
    }

    const batchDims = node.attrs.get('batch_dims')?.i ?? 0;

    let branchPrefix = '';
    if (requireHitnetContract) {
      const isL0Init =
        node.name === 'level0/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis' &&
        node.inputs[0] === 'level0/level_init/Reshape_2' &&
        node.inputs[1] === 'Unsqueeze__308:0' &&
        node.outputs[0] === 'level0/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis';
      const isL01Init =
        node.name === 'level0_1/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis' &&
        node.inputs[0] === 'level0_1/level_init/Reshape_2' &&
        node.inputs[1] === 'Unsqueeze__282:0' &&
        node.outputs[0] === 'level0_1/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis';

      if (!isL0Init && !isL01Init) {
        if (batchDims === 3 && idxMeta.dims.length >= 5) {
          skippedInvalidOrUnprovenCount++;
        }
        continue;
      }
      branchPrefix = isL0Init ? 'level0/' : 'level0_1/';

      if (
        batchDims !== 3 ||
        dataMeta.elemType !== 1 ||
        idxMeta.elemType !== 7 ||
        outMeta.elemType !== 1 ||
        !sameStaticDims(dataMeta.dims, [1, 60, 80, 320]) ||
        !sameStaticDims(idxMeta.dims, [1, 60, 80, 1, 1]) ||
        !sameStaticDims(outMeta.dims, [1, 60, 80, 1])
      ) {
        skippedInvalidOrUnprovenCount++;
        continue;
      }
    } else {
      if (
        batchDims !== 3 ||
        dataMeta.elemType !== 1 ||
        idxMeta.elemType !== 7 ||
        outMeta.elemType !== 1 ||
        dataMeta.dims.length !== 4 ||
        idxMeta.dims.length !== 5 ||
        outMeta.dims.length !== 4 ||
        idxMeta.dims[3] !== 1 ||
        idxMeta.dims[4] !== 1 ||
        dataMeta.dims[0] !== idxMeta.dims[0] ||
        dataMeta.dims[1] !== idxMeta.dims[1] ||
        dataMeta.dims[2] !== idxMeta.dims[2] ||
        outMeta.dims[0] !== dataMeta.dims[0] ||
        outMeta.dims[1] !== dataMeta.dims[1] ||
        outMeta.dims[2] !== dataMeta.dims[2] ||
        outMeta.dims[3] !== 1
      ) {
        continue;
      }
    }

    // Verify exact index lineage:
    //   data === Reshape_2 <-- ArgMin(axis=3, keepdims=0, select_last_index=0)
    //   --> ExpandDims_2(Reshape([B0,B1,B2,1])) --> Unsqueeze(axes=[-1|4]) --> GatherND
    //                                           \-> Cast(to=FLOAT)
    const unsqueezeNode = producerByOutput.get(node.inputs[1]);
    if (!unsqueezeNode || unsqueezeNode.opType !== 'Unsqueeze' || unsqueezeNode.inputs.length < 1) {
      skippedInvalidOrUnprovenCount++;
      continue;
    }
    const unsqAxesAttr = unsqueezeNode.attrs.get('axes')?.ints;
    const unsqAxesInput =
      unsqueezeNode.inputs.length >= 2 && unsqueezeNode.inputs[1]
        ? preInference.tensorMeta[unsqueezeNode.inputs[1]]?.intVals
        : undefined;
    const unsqAxes = unsqAxesAttr && unsqAxesAttr.length > 0 ? unsqAxesAttr : unsqAxesInput;
    if (!unsqAxes || unsqAxes.length !== 1 || (unsqAxes[0] !== -1 && unsqAxes[0] !== 4)) {
      skippedInvalidOrUnprovenCount++;
      continue;
    }

    const expandNode = producerByOutput.get(unsqueezeNode.inputs[0]);
    if (!expandNode || expandNode.opType !== 'Reshape' || expandNode.inputs.length < 2) {
      skippedInvalidOrUnprovenCount++;
      continue;
    }
    const expandMeta = preInference.tensorMeta[expandNode.outputs[0]];
    if (
      !expandMeta ||
      expandMeta.elemType !== 7 ||
      !sameStaticDims(expandMeta.dims, [
        dataMeta.dims[0],
        dataMeta.dims[1],
        dataMeta.dims[2],
        1,
      ])
    ) {
      skippedInvalidOrUnprovenCount++;
      continue;
    }

    const argMinNode = producerByOutput.get(expandNode.inputs[0]);
    if (
      !argMinNode ||
      argMinNode.opType !== 'ArgMin' ||
      argMinNode.inputs.length !== 1 ||
      argMinNode.inputs[0] !== node.inputs[0]
    ) {
      skippedInvalidOrUnprovenCount++;
      continue;
    }
    const argMinAxis = argMinNode.attrs.get('axis')?.i ?? 0;
    const argMinKeepDims = argMinNode.attrs.get('keepdims')?.i ?? 1;
    const argMinSelectLast = argMinNode.attrs.get('select_last_index')?.i ?? 0;
    if (argMinAxis !== 3 && argMinAxis !== -1) {
      skippedInvalidOrUnprovenCount++;
      continue;
    }
    if (argMinKeepDims !== 0 || argMinSelectLast !== 0) {
      skippedInvalidOrUnprovenCount++;
      continue;
    }

    if (requireHitnetContract) {
      if (
        !argMinNode.name.startsWith(`${branchPrefix}level_init/`) ||
        !expandNode.name.startsWith(`${branchPrefix}level_init/`)
      ) {
        skippedInvalidOrUnprovenCount++;
        continue;
      }
    }

    const expandConsumers = consumersByTensor.get(expandNode.outputs[0]);
    let castNode: ParsedLevelInitNode | null = null;
    if (expandConsumers) {
      for (const cIdx of expandConsumers) {
        const cn = parsedNodes[cIdx];
        if (cn.opType === 'Cast' && (cn.attrs.get('to')?.i ?? 0) === 1) {
          if (!requireHitnetContract || cn.name.startsWith(`${branchPrefix}level_init/`)) {
            castNode = cn;
            break;
          }
        }
      }
    }
    if (requireHitnetContract && !castNode) {
      skippedInvalidOrUnprovenCount++;
      continue;
    }

    // Verify optional non-negative L1 SAD cost lineage: Reshape_2 <- ReduceSum(axes=[3],keepdims=0) <- Abs <- Sub
    let hasNonNegativeSadCostLineage = false;
    const costReshapeNode = producerByOutput.get(node.inputs[0]);
    if (costReshapeNode?.opType === 'Reshape' && costReshapeNode.inputs[0]) {
      const reduceSumNode = producerByOutput.get(costReshapeNode.inputs[0]);
      if (reduceSumNode?.opType === 'ReduceSum' && reduceSumNode.inputs[0]) {
        const rsAxes = reduceSumNode.attrs.get('axes')?.ints;
        const rsKeep = reduceSumNode.attrs.get('keepdims')?.i ?? 1;
        const absNode = producerByOutput.get(reduceSumNode.inputs[0]);
        if (
          rsAxes &&
          rsAxes.length === 1 &&
          (rsAxes[0] === 3 || rsAxes[0] === -1) &&
          rsKeep === 0 &&
          absNode?.opType === 'Abs'
        ) {
          hasNonNegativeSadCostLineage = true;
        }
      }
    }

    targets.push({
      node,
      unsqueezeNode,
      expandNode,
      argMinNode,
      castNode,
      dataDims: [dataMeta.dims[0], dataMeta.dims[1], dataMeta.dims[2], dataMeta.dims[3]],
      idxDims: [
        idxMeta.dims[0],
        idxMeta.dims[1],
        idxMeta.dims[2],
        idxMeta.dims[3],
        idxMeta.dims[4],
      ],
      outDims: [outMeta.dims[0], outMeta.dims[1], outMeta.dims[2], outMeta.dims[3]],
      hasNonNegativeSadCostLineage,
    });
  }

  return {
    valid: true,
    topFields,
    graphIndex,
    graphFields,
    preInference,
    parsedNodes,
    usedNames,
    producerByOutput,
    consumersByTensor,
    totalGatherNdCount,
    skippedDynamicCount,
    skippedInvalidOrUnprovenCount,
    targets,
  };
}

/**
 * Operation-preserving HITNet-only opt-in guard & rewrite for the two `level_init/GatherV2_1`
 * cost-minimum `GatherND(batch_dims=3)` nodes (`#127` in `level0` and `#438` in `level0_1`):
 *   - Verifies exact node names, `opset=12` (`ir=7`), dtypes, static shapes `[1,60,80,320]` /
 *     `[1,60,80,1,1]`, and same-branch `ArgMin -> ExpandDims_2 -> Unsqueeze -> GatherND` & `Cast`
 *     lineage on the identical cost tensor.
 *   - Merges spatial batch axes `(60, 80) -> 4800`:
 *       - `data`: `float32 [1,60,80,320]` -> `PreReshape` -> `float32 [1,4800,320,1]` (rank 4)
 *       - `indices`: `int64 [1,60,80,1,1]` -> `PreReshape` -> `int64 [1,4800,1,1]` (rank 4)
 *       - `GatherND`: `batch_dims=3` -> `batch_dims=2` -> `float32 [1,4800,1,1]` (rank 4)
 *       - `PostReshape`: restores exact original output name and shape `[1,60,80,1]`
 *   - Keeps `ArgMin` and `Cast` outside the rewrite, removes NO shared node or initializer,
 *     and fails atomically returning untouched bytes if any guard fails.
 */
export function normalizeHitnetLevelInitGatherNdRankForWebGpu(
  bytes: Uint8Array,
  options: NormalizeGatherNdRankOptions = {}
): OnnxLevelInitGatherNormalizeResult {
  const emptyResult: OnnxLevelInitGatherNormalizeResult = {
    bytes,
    mode: 'reshape-rank4',
    normalizedCount: 0,
    normalizedNodeNames: [],
    skippedDynamicCount: 0,
    skippedInvalidOrUnprovenCount: 0,
    totalGatherNdCount: 0,
    verifiedCostLineageCount: 0,
    documentedLimitations: [],
  };
  const requireHitnetContract = options.requireHitnetContract ?? true;
  if (
    (options.backendPreference === 'wasm' && !options.allowDiagnosticTransformedWasm) ||
    bytes.byteLength < 16
  ) {
    return emptyResult;
  }
  if (requireHitnetContract && options.variantId !== 'hitnet-sf') {
    return emptyResult;
  }

  try {
    const ctx = inspectLevelInitGatherContext(bytes, requireHitnetContract);
    if (!ctx) return emptyResult;

    const {
      topFields,
      graphIndex,
      graphFields,
      usedNames,
      totalGatherNdCount,
      skippedDynamicCount,
      skippedInvalidOrUnprovenCount,
      targets,
    } = ctx;

    if (requireHitnetContract) {
      if (
        totalGatherNdCount !== 10 ||
        targets.length !== 2 ||
        skippedDynamicCount > 0 ||
        skippedInvalidOrUnprovenCount > 0
      ) {
        return {
          ...emptyResult,
          skippedDynamicCount,
          skippedInvalidOrUnprovenCount,
          totalGatherNdCount,
        };
      }
      const names = targets.map((t) => t.node.name);
      if (
        !names.includes('level0/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis') ||
        !names.includes('level0_1/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis')
      ) {
        return { ...emptyResult, totalGatherNdCount };
      }
    } else if (
      targets.length === 0 ||
      skippedDynamicCount > 0 ||
      skippedInvalidOrUnprovenCount > 0
    ) {
      return {
        ...emptyResult,
        skippedDynamicCount,
        skippedInvalidOrUnprovenCount,
        totalGatherNdCount,
      };
    }

    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    let tempSeq = 0;
    const allocName = (prefix: string): string => {
      let candidate = `${prefix}_${tempSeq++}`;
      while (usedNames.has(candidate)) {
        candidate = `${prefix}_${tempSeq++}`;
      }
      usedNames.add(candidate);
      return candidate;
    };

    const concatBytes = (parts: readonly Uint8Array[]): Uint8Array => {
      let total = 0;
      for (const p of parts) total += p.byteLength;
      const out = new Uint8Array(total);
      let off = 0;
      for (const p of parts) {
        out.set(p, off);
        off += p.byteLength;
      }
      return out;
    };
    const encodeLenField = (fn: number, payload: Uint8Array): Uint8Array => {
      const tag = encodeProtoVarint((fn << 3) | 2);
      const len = encodeProtoVarint(payload.byteLength);
      const out = new Uint8Array(tag.byteLength + len.byteLength + payload.byteLength);
      out.set(tag, 0);
      out.set(len, tag.byteLength);
      out.set(payload, tag.byteLength + len.byteLength);
      return out;
    };
    const encodeStrField = (fn: number, str: string): Uint8Array =>
      encodeLenField(fn, encoder.encode(str));
    const encodeVarintField = (fn: number, val: number): Uint8Array =>
      concatBytes([encodeProtoVarint((fn << 3) | 0), encodeProtoVarint(val)]);
    const encodeAttrInt = (attrName: string, val: number): Uint8Array =>
      encodeLenField(
        5,
        concatBytes([
          encodeStrField(1, attrName),
          concatBytes([encodeProtoVarint((3 << 3) | 0), encodeSignedVarint64Bytes(val)]),
          encodeVarintField(20, 2),
        ])
      );
    const encodeNodeSimple = (
      opType: string,
      inputs: readonly string[],
      outputs: readonly string[],
      name: string
    ): Uint8Array => {
      const parts: Uint8Array[] = [];
      for (const inp of inputs) parts.push(encodeStrField(1, inp));
      for (const out of outputs) parts.push(encodeStrField(2, out));
      parts.push(encodeStrField(3, name));
      parts.push(encodeStrField(4, opType));
      return encodeLenField(1, concatBytes(parts));
    };
    const encodeInt64ShapeInitializer = (name: string, shapeDims: readonly number[]): Uint8Array => {
      const raw = new Uint8Array(shapeDims.length * 8);
      const dv = new DataView(raw.buffer);
      for (let i = 0; i < shapeDims.length; i++) {
        dv.setBigInt64(i * 8, BigInt(shapeDims[i]), true);
      }
      return encodeLenField(
        5,
        concatBytes([
          encodeVarintField(1, shapeDims.length),
          encodeVarintField(2, 7),
          encodeStrField(8, name),
          encodeLenField(9, raw),
        ])
      );
    };

    const addedInitializers: Uint8Array[] = [];
    const shapeInitCache = new Map<string, string>();
    const getOrCreateInt64ShapeInit = (rolePrefix: string, shapeDims: readonly number[]): string => {
      const key = `${rolePrefix}:${shapeDims.join(',')}`;
      const existing = shapeInitCache.get(key);
      if (existing) return existing;
      const name = allocName(`__webgpu_li_gnd_${rolePrefix}`);
      shapeInitCache.set(key, name);
      addedInitializers.push(encodeInt64ShapeInitializer(name, shapeDims));
      return name;
    };

    const nodeReplacementMap = new Map<number, Uint8Array[]>();
    const normalizedNodeNames: string[] = [];
    const expectedAfterMetaByOutName = new Map<string, { elemType: number; outDims: number[] }>();
    let verifiedCostLineageCount = 0;

    for (const target of targets) {
      const { node, dataDims, outDims } = target;
      const mergedBatch = dataDims[1] * dataDims[2];
      if (!Number.isSafeInteger(mergedBatch) || mergedBatch <= 0) {
        return emptyResult;
      }

      const mergedDataDims = [dataDims[0], mergedBatch, dataDims[3], 1];
      const mergedIdxDims = [dataDims[0], mergedBatch, 1, 1];
      const mergedOutDims = [dataDims[0], mergedBatch, 1, 1];
      const origOutDims = [...outDims];

      const origNodeName = node.name || `GatherND_level_init_${node.nodeIdx}`;
      const dataShapeInit = getOrCreateInt64ShapeInit('data_shape', mergedDataDims);
      const idxShapeInit = getOrCreateInt64ShapeInit('idx_shape', mergedIdxDims);
      const outShapeInit = getOrCreateInt64ShapeInit('out_shape', origOutDims);

      const mergedDataName = allocName(`${origNodeName}__li_data_4d`);
      const mergedIdxName = allocName(`${origNodeName}__li_indices_4d`);
      const mergedOutName = allocName(`${origNodeName}__li_out_4d`);

      const rewrittenNodeParts: Uint8Array[] = [];
      let inSlot = 0;
      let outSlot = 0;
      let replacedBatchDims = false;
      for (const nf of node.nodeFields) {
        if (nf.fn === 1 && nf.wt === 2) {
          rewrittenNodeParts.push(
            inSlot === 0
              ? encodeStrField(1, mergedDataName)
              : inSlot === 1
                ? encodeStrField(1, mergedIdxName)
                : bytes.subarray(nf.fStart, nf.fEnd)
          );
          inSlot++;
        } else if (nf.fn === 2 && nf.wt === 2) {
          rewrittenNodeParts.push(
            outSlot === 0 ? encodeStrField(2, mergedOutName) : bytes.subarray(nf.fStart, nf.fEnd)
          );
          outSlot++;
        } else if (nf.fn === 5 && nf.wt === 2) {
          const af = parseProtoFields(bytes, nf.valStart, nf.valEnd);
          const nameF = af?.find((a) => a.fn === 1 && a.wt === 2 && a.valEnd > a.valStart);
          const attrName = nameF ? decoder.decode(bytes.subarray(nameF.valStart, nameF.valEnd)) : '';
          if (attrName === 'batch_dims') {
            rewrittenNodeParts.push(encodeAttrInt('batch_dims', 2));
            replacedBatchDims = true;
          } else {
            rewrittenNodeParts.push(bytes.subarray(nf.fStart, nf.fEnd));
          }
        } else {
          rewrittenNodeParts.push(bytes.subarray(nf.fStart, nf.fEnd));
        }
      }
      if (!replacedBatchDims) {
        rewrittenNodeParts.push(encodeAttrInt('batch_dims', 2));
      }

      nodeReplacementMap.set(node.nodeIdx, [
        encodeNodeSimple(
          'Reshape',
          [node.inputs[0], dataShapeInit],
          [mergedDataName],
          allocName(`${origNodeName}__LiPreReshapeData`)
        ),
        encodeNodeSimple(
          'Reshape',
          [node.inputs[1], idxShapeInit],
          [mergedIdxName],
          allocName(`${origNodeName}__LiPreReshapeIndices`)
        ),
        encodeLenField(1, concatBytes(rewrittenNodeParts)),
        encodeNodeSimple(
          'Reshape',
          [mergedOutName, outShapeInit],
          [node.outputs[0]],
          allocName(`${origNodeName}__LiPostReshapeOut`)
        ),
      ]);

      expectedAfterMetaByOutName.set(mergedDataName, { elemType: 1, outDims: mergedDataDims });
      expectedAfterMetaByOutName.set(mergedIdxName, { elemType: 7, outDims: mergedIdxDims });
      expectedAfterMetaByOutName.set(mergedOutName, { elemType: 1, outDims: mergedOutDims });
      expectedAfterMetaByOutName.set(node.outputs[0], { elemType: 1, outDims: origOutDims });
      normalizedNodeNames.push(origNodeName);
      verifiedCostLineageCount++;
    }

    const finalGraphParts: Uint8Array[] = [];
    let currentNodeIdx = 0;
    for (const gf of graphFields) {
      if (gf.fn === 13) continue; // Drop stale value_info entries
      if (gf.fn === 1 && gf.wt === 2) {
        const replacement = nodeReplacementMap.get(currentNodeIdx);
        currentNodeIdx++;
        if (replacement) {
          for (const part of replacement) finalGraphParts.push(part);
        } else {
          finalGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        }
      } else {
        finalGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
      }
    }
    for (const initPart of addedInitializers) {
      finalGraphParts.push(initPart);
    }

    let graphPayloadLen = 0;
    for (const part of finalGraphParts) graphPayloadLen += part.byteLength;
    const graphTag = encodeProtoVarint((7 << 3) | 2);
    const graphLen = encodeProtoVarint(graphPayloadLen);
    let totalLen = 0;
    for (let i = 0; i < topFields.length; i++) {
      totalLen +=
        i === graphIndex
          ? graphTag.byteLength + graphLen.byteLength + graphPayloadLen
          : topFields[i].fEnd - topFields[i].fStart;
    }

    const out = new Uint8Array(totalLen);
    let offset = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        out.set(graphTag, offset);
        offset += graphTag.byteLength;
        out.set(graphLen, offset);
        offset += graphLen.byteLength;
        for (const part of finalGraphParts) {
          out.set(part, offset);
          offset += part.byteLength;
        }
      } else {
        const chunk = bytes.subarray(topFields[i].fStart, topFields[i].fEnd);
        out.set(chunk, offset);
        offset += chunk.byteLength;
      }
    }

    const beforeValidation = validateOnnxGraphNodeInputs(bytes);
    const afterValidation = validateOnnxGraphNodeInputs(out);
    const preexistingMissing = new Set(beforeValidation.missingInputs.map((m) => m.inputName));
    const newlyMissing = afterValidation.missingInputs.filter(
      (m) => !preexistingMissing.has(m.inputName)
    );
    if (
      newlyMissing.length > 0 ||
      (beforeValidation.hasGraphInputs && !afterValidation.valid)
    ) {
      return emptyResult;
    }

    const postInference = inferOnnxStaticTensorMeta(out);
    if (
      !postInference.valid ||
      postInference.duplicateOutputs.length > 0 ||
      postInference.duplicateInitializers.length > 0 ||
      postInference.duplicateNodeNames.length > 0
    ) {
      return emptyResult;
    }

    for (const [tensorName, exp] of expectedAfterMetaByOutName.entries()) {
      const m = postInference.tensorMeta[tensorName];
      if (!m || m.elemType !== exp.elemType || !sameStaticDims(m.dims, exp.outDims)) {
        return emptyResult;
      }
    }
    if (requireHitnetContract) {
      const refPost = postInference.tensorMeta['reference_output_disparity'];
      const secPost = postInference.tensorMeta['secondary_output_disparity'];
      if (
        !refPost ||
        !secPost ||
        refPost.elemType !== 1 ||
        secPost.elemType !== 1 ||
        !sameStaticDims(refPost.dims, [1, 240, 320, 1]) ||
        !sameStaticDims(secPost.dims, [1, 240, 320, 1])
      ) {
        return emptyResult;
      }
    }

    return {
      bytes: out,
      mode: 'reshape-rank4',
      normalizedCount: normalizedNodeNames.length,
      normalizedNodeNames,
      skippedDynamicCount,
      skippedInvalidOrUnprovenCount,
      totalGatherNdCount,
      verifiedCostLineageCount,
      documentedLimitations: [
        'Operation-preserving rank-4 GatherND(batch_dims=2) reshape; preserves ArgMin, Cast, and all shared nodes/initializers.',
      ],
    };
  } catch {
    return emptyResult;
  }
}

export const HITNET_LEVEL_INIT_REDUCEMIN_LIMITATIONS: readonly string[] = [
  'Diagnostic candidate only (NOT a production default): replaces two level_init/GatherV2_1 nodes with ReduceMin(axes=[3], keepdims=1) while retaining ArgMin and Cast untouched.',
  'Graph-proven algebraic equivalence holds for finite non-negative cost tensors (proven Reshape(ReduceSum(Abs(Sub(...))))); Abs eliminates -0.0 in finite sums.',
  'IEEE-754 special-value limitation: if any cost candidate is NaN, WGSL ReduceMinShared (select(best, cand, cand < best)) preserves finite lane 0 vs CPU ArgMin NaN behavior; runtime verification enforces finiteCount===length.',
];

/**
 * Independent opt-in diagnostic candidate replacing the two `level_init/GatherV2_1` nodes
 * (`#127` and `#438`) with `ReduceMin(axes=[3], keepdims=1)` ONLY when static graph proof
 * establishes:
 *   1. Verified opset (`=== 12` for HITNet contract, or `[11..17]` attribute-based `ReduceMin`).
 *   2. Exact same cost tensor `Reshape_2 [1,60,80,320]` produced by `Reshape(ReduceSum(Abs(...)))`
 *      (non-negative, signed-zero-free) feeding BOTH `ArgMin(axis=3, keepdims=0, select_last_index=0)`
 *      and `GatherND(batch_dims=3)`.
 *   3. `ArgMin -> ExpandDims_2 -> Cast` remains completely preserved outside the rewrite.
 */
export function normalizeHitnetLevelInitReduceMinForWebGpu(
  bytes: Uint8Array,
  options: NormalizeGatherNdRankOptions = {}
): OnnxLevelInitGatherNormalizeResult {
  const emptyResult: OnnxLevelInitGatherNormalizeResult = {
    bytes,
    mode: 'reduce-min',
    normalizedCount: 0,
    normalizedNodeNames: [],
    skippedDynamicCount: 0,
    skippedInvalidOrUnprovenCount: 0,
    totalGatherNdCount: 0,
    verifiedCostLineageCount: 0,
    documentedLimitations: [...HITNET_LEVEL_INIT_REDUCEMIN_LIMITATIONS],
  };
  const requireHitnetContract = options.requireHitnetContract ?? true;
  if (
    (options.backendPreference === 'wasm' && !options.allowDiagnosticTransformedWasm) ||
    bytes.byteLength < 16
  ) {
    return emptyResult;
  }
  if (requireHitnetContract && options.variantId !== 'hitnet-sf') {
    return emptyResult;
  }

  try {
    const ctx = inspectLevelInitGatherContext(bytes, requireHitnetContract);
    if (!ctx) return emptyResult;

    const {
      topFields,
      graphIndex,
      graphFields,
      consumersByTensor,
      totalGatherNdCount,
      skippedDynamicCount,
      skippedInvalidOrUnprovenCount,
      targets,
    } = ctx;

    // Every ReduceMin target MUST prove same non-negative Abs->ReduceSum->Reshape cost lineage
    const provenTargets = targets.filter((t) => t.hasNonNegativeSadCostLineage);
    const unprovenCostCount = targets.length - provenTargets.length;
    if (unprovenCostCount > 0) {
      return {
        ...emptyResult,
        skippedDynamicCount,
        skippedInvalidOrUnprovenCount: skippedInvalidOrUnprovenCount + unprovenCostCount,
        totalGatherNdCount,
      };
    }

    if (requireHitnetContract) {
      if (
        totalGatherNdCount !== 10 ||
        provenTargets.length !== 2 ||
        skippedDynamicCount > 0 ||
        skippedInvalidOrUnprovenCount > 0
      ) {
        return {
          ...emptyResult,
          skippedDynamicCount,
          skippedInvalidOrUnprovenCount,
          totalGatherNdCount,
        };
      }
      const names = provenTargets.map((t) => t.node.name);
      if (
        !names.includes('level0/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis') ||
        !names.includes('level0_1/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis')
      ) {
        return { ...emptyResult, totalGatherNdCount };
      }
    } else if (
      provenTargets.length === 0 ||
      skippedDynamicCount > 0 ||
      skippedInvalidOrUnprovenCount > 0
    ) {
      return {
        ...emptyResult,
        skippedDynamicCount,
        skippedInvalidOrUnprovenCount,
        totalGatherNdCount,
      };
    }

    const encoder = new TextEncoder();
    const concatBytes = (parts: readonly Uint8Array[]): Uint8Array => {
      let total = 0;
      for (const p of parts) total += p.byteLength;
      const out = new Uint8Array(total);
      let off = 0;
      for (const p of parts) {
        out.set(p, off);
        off += p.byteLength;
      }
      return out;
    };
    const encodeLenField = (fn: number, payload: Uint8Array): Uint8Array => {
      const tag = encodeProtoVarint((fn << 3) | 2);
      const len = encodeProtoVarint(payload.byteLength);
      const out = new Uint8Array(tag.byteLength + len.byteLength + payload.byteLength);
      out.set(tag, 0);
      out.set(len, tag.byteLength);
      out.set(payload, tag.byteLength + len.byteLength);
      return out;
    };
    const encodeStrField = (fn: number, str: string): Uint8Array =>
      encodeLenField(fn, encoder.encode(str));
    const encodeVarintField = (fn: number, val: number): Uint8Array =>
      concatBytes([encodeProtoVarint((fn << 3) | 0), encodeProtoVarint(val)]);
    const encodeAttrInt = (attrName: string, val: number): Uint8Array =>
      encodeLenField(
        5,
        concatBytes([
          encodeStrField(1, attrName),
          concatBytes([encodeProtoVarint((3 << 3) | 0), encodeSignedVarint64Bytes(val)]),
          encodeVarintField(20, 2),
        ])
      );
    const encodeAttrInts = (attrName: string, vals: readonly number[]): Uint8Array =>
      encodeLenField(
        5,
        concatBytes([
          encodeStrField(1, attrName),
          ...vals.map((v) =>
            concatBytes([encodeProtoVarint((8 << 3) | 0), encodeSignedVarint64Bytes(v)])
          ),
          encodeVarintField(20, 7),
        ])
      );
    const encodeReduceMinNode = (name: string, inputName: string, outputName: string): Uint8Array =>
      encodeLenField(
        1,
        concatBytes([
          encodeStrField(1, inputName),
          encodeStrField(2, outputName),
          encodeStrField(3, name),
          encodeStrField(4, 'ReduceMin'),
          encodeAttrInts('axes', [3]),
          encodeAttrInt('keepdims', 1),
        ])
      );

    const nodeReplacementMap = new Map<number, Uint8Array[]>();
    const deadUnsharedNodeIndices = new Set<number>();
    const normalizedNodeNames: string[] = [];
    const expectedAfterMetaByOutName = new Map<string, { elemType: number; outDims: number[] }>();

    for (const target of provenTargets) {
      const { node, unsqueezeNode, outDims } = target;
      const origNodeName = node.name || `ReduceMin_level_init_${node.nodeIdx}`;
      nodeReplacementMap.set(node.nodeIdx, [
        encodeReduceMinNode(origNodeName, node.inputs[0], node.outputs[0]),
      ]);
      // Drop dead Unsqueeze ONLY if its output tensor is unshared and solely consumed by this GatherND
      const unsqConsumers = consumersByTensor.get(unsqueezeNode.outputs[0]);
      if (
        unsqueezeNode.outputs.length === 1 &&
        unsqConsumers &&
        unsqConsumers.size === 1 &&
        unsqConsumers.has(node.nodeIdx)
      ) {
        deadUnsharedNodeIndices.add(unsqueezeNode.nodeIdx);
      }
      expectedAfterMetaByOutName.set(node.outputs[0], { elemType: 1, outDims: [...outDims] });
      normalizedNodeNames.push(origNodeName);
    }

    const finalGraphParts: Uint8Array[] = [];
    let currentNodeIdx = 0;
    for (const gf of graphFields) {
      if (gf.fn === 13) continue; // Drop stale value_info entries
      if (gf.fn === 1 && gf.wt === 2) {
        const idx = currentNodeIdx++;
        if (deadUnsharedNodeIndices.has(idx)) {
          continue;
        }
        const replacement = nodeReplacementMap.get(idx);
        if (replacement) {
          for (const part of replacement) finalGraphParts.push(part);
        } else {
          finalGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        }
      } else {
        finalGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
      }
    }

    let graphPayloadLen = 0;
    for (const part of finalGraphParts) graphPayloadLen += part.byteLength;
    const graphTag = encodeProtoVarint((7 << 3) | 2);
    const graphLen = encodeProtoVarint(graphPayloadLen);
    let totalLen = 0;
    for (let i = 0; i < topFields.length; i++) {
      totalLen +=
        i === graphIndex
          ? graphTag.byteLength + graphLen.byteLength + graphPayloadLen
          : topFields[i].fEnd - topFields[i].fStart;
    }

    const out = new Uint8Array(totalLen);
    let offset = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        out.set(graphTag, offset);
        offset += graphTag.byteLength;
        out.set(graphLen, offset);
        offset += graphLen.byteLength;
        for (const part of finalGraphParts) {
          out.set(part, offset);
          offset += part.byteLength;
        }
      } else {
        const chunk = bytes.subarray(topFields[i].fStart, topFields[i].fEnd);
        out.set(chunk, offset);
        offset += chunk.byteLength;
      }
    }

    const beforeValidation = validateOnnxGraphNodeInputs(bytes);
    const afterValidation = validateOnnxGraphNodeInputs(out);
    const preexistingMissing = new Set(beforeValidation.missingInputs.map((m) => m.inputName));
    const newlyMissing = afterValidation.missingInputs.filter(
      (m) => !preexistingMissing.has(m.inputName)
    );
    if (
      newlyMissing.length > 0 ||
      (beforeValidation.hasGraphInputs && !afterValidation.valid)
    ) {
      return emptyResult;
    }

    const postInference = inferOnnxStaticTensorMeta(out);
    if (
      !postInference.valid ||
      postInference.duplicateOutputs.length > 0 ||
      postInference.duplicateInitializers.length > 0 ||
      postInference.duplicateNodeNames.length > 0
    ) {
      return emptyResult;
    }

    for (const [tensorName, exp] of expectedAfterMetaByOutName.entries()) {
      const m = postInference.tensorMeta[tensorName];
      if (!m || m.elemType !== exp.elemType || !sameStaticDims(m.dims, exp.outDims)) {
        return emptyResult;
      }
    }
    if (requireHitnetContract) {
      const refPost = postInference.tensorMeta['reference_output_disparity'];
      const secPost = postInference.tensorMeta['secondary_output_disparity'];
      if (
        !refPost ||
        !secPost ||
        refPost.elemType !== 1 ||
        secPost.elemType !== 1 ||
        !sameStaticDims(refPost.dims, [1, 240, 320, 1]) ||
        !sameStaticDims(secPost.dims, [1, 240, 320, 1])
      ) {
        return emptyResult;
      }
    }

    return {
      bytes: out,
      mode: 'reduce-min',
      normalizedCount: normalizedNodeNames.length,
      normalizedNodeNames,
      skippedDynamicCount,
      skippedInvalidOrUnprovenCount,
      totalGatherNdCount,
      verifiedCostLineageCount: provenTargets.length,
      documentedLimitations: [...HITNET_LEVEL_INIT_REDUCEMIN_LIMITATIONS],
    };
  } catch {
    return emptyResult;
  }
}

export interface OnnxArgMinKeepDimsNormalizeResult {
  bytes: Uint8Array;
  normalizedCount: number;
  normalizedNodeNames: string[];
  untouchedNodeNames: string[];
  totalArgMinCount: number;
  skippedDynamicCount: number;
  skippedInvalidOrUnprovenCount: number;
  updatedValueInfoCount: number;
  ortCacheKeyBefore: string | null;
  ortCacheKeyAfter: string | null;
}

/**
 * Mirrors ORT Web/JSEP 1.30.0 `AttributeWithCacheKeyImpl2` (`Object.getOwnPropertyNames(this).sort().map(...)`)
 * + `getProgramInfoUniqueKey2` (`createReduceProgramInfo2(..., inputDependencies: ["rank"])`).
 * Proves statically that `keepDims=0` vs `keepDims=1` generate distinct shader-cache keys.
 */
export function computeOrtArgMinCacheAttributesKey(params: {
  axis: number;
  keepDims: boolean | number;
  selectLastIndex: number;
  inputType?: number;
  inputRank?: number;
  is1DimensionDispatch?: boolean;
}): string {
  const attrObj: Record<string, unknown> = {
    axis: params.axis,
    keepDims: params.keepDims,
    selectLastIndex: params.selectLastIndex,
  };
  const hint = Object.getOwnPropertyNames(attrObj)
    .sort()
    .map((k) => `${attrObj[k]}`)
    .join(';');
  const is1Dim = params.is1DimensionDispatch ?? false;
  const inputType = params.inputType ?? 1;
  const inputRank = params.inputRank ?? 4;
  return `ArgMin[${hint}]:${is1Dim}:${inputType};${inputRank}`;
}

async function computeSha256Hex(bytes: Uint8Array): Promise<string> {
  if (
    typeof globalThis.crypto !== 'undefined' &&
    globalThis.crypto.subtle &&
    typeof globalThis.crypto.subtle.digest === 'function'
  ) {
    const copy = new Uint8Array(bytes.byteLength);
    copy.set(bytes);
    const digest = await globalThis.crypto.subtle.digest('SHA-256', copy.buffer);
    return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
  }
  return 'unavailable';
}

function encodeSignedVarint64Bytes(val: number): Uint8Array {
  let v = BigInt.asUintN(64, BigInt(val));
  const out: number[] = [];
  while (v >= 0x80n) {
    out.push(Number(v & 0x7fn) | 0x80);
    v >>= 7n;
  }
  out.push(Number(v));
  return new Uint8Array(out);
}

/**
 * Smallest HITNet-only opt-in repair for the ORT JSEP 1.30.0 `ArgMin` shader cache
 * collision between `level_init/ArgMin` (`[1,60,80,320]`, 320 disparity candidates) and
 * `shared/prop0/ArgMin` (`[1,60,80,4]`, 4 propagation candidates).
 *
 * Sets `keepdims=1` on ONLY the two 4-candidate `prop0` `ArgMin` nodes when static graph proof
 * establishes that each original `ArgMin` output reaches ONLY its existing shape-normalizing
 * `Reshape(..., [1,60,80,1])` node (keeping the downstream signature and graph contract identical).
 * Fail-closed on any contract/opset/dtype/shape/consumer mismatch.
 */
export function normalizeHitnetArgMinKeepDimsForWebGpu(
  bytes: Uint8Array,
  options: NormalizeArgMinKeepDimsOptions = {}
): OnnxArgMinKeepDimsNormalizeResult {
  const emptyResult: OnnxArgMinKeepDimsNormalizeResult = {
    bytes,
    normalizedCount: 0,
    normalizedNodeNames: [],
    untouchedNodeNames: [],
    totalArgMinCount: 0,
    skippedDynamicCount: 0,
    skippedInvalidOrUnprovenCount: 0,
    updatedValueInfoCount: 0,
    ortCacheKeyBefore: null,
    ortCacheKeyAfter: null,
  };
  const requireHitnetContract = options.requireHitnetContract ?? true;
  if (
    (options.backendPreference === 'wasm' && !options.allowDiagnosticTransformedWasm) ||
    bytes.byteLength < 16
  ) {
    return emptyResult;
  }
  if (requireHitnetContract && options.variantId !== 'hitnet-sf') {
    return emptyResult;
  }

  try {
    const topFields = parseProtoFields(bytes, 0, bytes.byteLength);
    if (!topFields) return emptyResult;

    const decoder = new TextDecoder();
    const irField = topFields.find((f) => f.fn === 1 && f.wt === 0);
    const irVersion = irField ? (readProtoVarint(bytes, irField.valStart, irField.valEnd)?.[0] ?? null) : null;
    const defaultOpset = readOnnxDefaultOpsetVersion(bytes, topFields, decoder);
    if (requireHitnetContract) {
      if (irVersion !== 7 || defaultOpset !== 12) {
        return emptyResult;
      }
    } else if (defaultOpset !== null && (defaultOpset < 12 || defaultOpset > 21)) {
      return emptyResult;
    }

    const graphIndex = topFields.findIndex((f) => f.fn === 7 && f.wt === 2);
    if (graphIndex < 0) return emptyResult;

    const graphField = topFields[graphIndex];
    const graphFields = parseProtoFields(bytes, graphField.valStart, graphField.valEnd);
    if (!graphFields) return emptyResult;

    const encoder = new TextEncoder();
    const allInitializerNames = new Set<string>();
    const parsedGraphInputs: ParsedStaticValueInfo[] = [];
    const parsedGraphOutputs: ParsedStaticValueInfo[] = [];
    const tensorMeta = new Map<string, StaticTensorMeta>();
    let hasControlFlowNode = false;

    interface ParsedNodeEntry {
      gf: ProtobufFieldSlice;
      nodeFields: ProtobufFieldSlice[];
      opType: string;
      name: string;
      inputs: string[];
      outputs: string[];
      attrs: Map<string, { i?: number; ints?: number[] }>;
    }
    const parsedNodes: ParsedNodeEntry[] = [];

    for (const gf of graphFields) {
      if (gf.fn === 5 && gf.wt === 2) {
        const init = parseInitializerStaticMeta(bytes, gf, decoder);
        if (init) {
          allInitializerNames.add(init.name);
          tensorMeta.set(init.name, init.meta);
        } else {
          const nm = getInitializerName(bytes, gf.valStart, gf.valEnd, decoder);
          if (nm) allInitializerNames.add(nm);
        }
        continue;
      }
      if (gf.fn === 11 && gf.wt === 2) {
        const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
        if (vi) {
          parsedGraphInputs.push(vi);
          if (vi.elemType && vi.dims) {
            tensorMeta.set(vi.name, { elemType: vi.elemType, dims: vi.dims });
          }
        }
        continue;
      }
      if (gf.fn === 12 && gf.wt === 2) {
        const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
        if (vi) {
          parsedGraphOutputs.push(vi);
        }
        continue;
      }
      if (gf.fn !== 1 || gf.wt !== 2) continue;

      const nodeFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
      if (!nodeFields) return emptyResult;
      const inputs: string[] = [];
      const outputs: string[] = [];
      let name = '';
      let opType = '';
      const attrs = new Map<string, { i?: number; ints?: number[] }>();

      for (const nf of nodeFields) {
        if (nf.fn === 1 && nf.wt === 2) {
          inputs.push(
            nf.valEnd > nf.valStart ? decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)) : ''
          );
        } else if (nf.fn === 2 && nf.wt === 2) {
          outputs.push(
            nf.valEnd > nf.valStart ? decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)) : ''
          );
        } else if (nf.fn === 3 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          name = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
        } else if (nf.fn === 4 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          opType = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
          if (opType === 'If' || opType === 'Loop' || opType === 'Scan') {
            hasControlFlowNode = true;
          }
        } else if (nf.fn === 5 && nf.wt === 2) {
          const af = parseProtoFields(bytes, nf.valStart, nf.valEnd);
          if (!af) continue;
          let an = '';
          let ai: number | undefined;
          const aints: number[] = [];
          for (const a of af) {
            if (a.fn === 1 && a.wt === 2 && a.valEnd > a.valStart) {
              an = decoder.decode(bytes.subarray(a.valStart, a.valEnd));
            } else if (a.fn === 3 && a.wt === 0) {
              const v = readProtoVarintBigInt(bytes, a.valStart, a.valEnd);
              if (v) ai = Number(BigInt.asIntN(64, v[0]));
            } else if (a.fn === 8 && a.wt === 0) {
              const v = readProtoVarintBigInt(bytes, a.valStart, a.valEnd);
              if (v) aints.push(Number(BigInt.asIntN(64, v[0])));
            }
          }
          if (an) attrs.set(an, { i: ai, ints: aints });
        }
      }
      parsedNodes.push({ gf, nodeFields, opType, name, inputs, outputs, attrs });
    }

    if (hasControlFlowNode) return emptyResult;

    const nonInitInputs = parsedGraphInputs.filter((g) => !allInitializerNames.has(g.name));
    if (
      nonInitInputs.length === 0 ||
      nonInitInputs.some((g) => g.dims === null || g.elemType === null)
    ) {
      return emptyResult;
    }

    if (requireHitnetContract) {
      if (
        nonInitInputs.length !== 1 ||
        nonInitInputs[0].name !== 'input' ||
        nonInitInputs[0].elemType !== 1 ||
        !nonInitInputs[0].dims ||
        !sameStaticDims(nonInitInputs[0].dims, [1, 6, 240, 320])
      ) {
        return emptyResult;
      }
      const outNames = parsedGraphOutputs.map((o) => o.name);
      if (
        outNames.length !== 2 ||
        !outNames.includes('reference_output_disparity') ||
        !outNames.includes('secondary_output_disparity')
      ) {
        return emptyResult;
      }
    }

    // Build consumer map across the entire graph and check duplicates within node inputs
    const consumersByTensor = new Map<string, Array<{ nodeIdx: number; slot: number }>>();
    for (let i = 0; i < parsedNodes.length; i++) {
      const node = parsedNodes[i];
      for (let s = 0; s < node.inputs.length; s++) {
        const inp = node.inputs[s];
        if (!inp) continue;
        let list = consumersByTensor.get(inp);
        if (!list) {
          list = [];
          consumersByTensor.set(inp, list);
        }
        list.push({ nodeIdx: i, slot: s });
      }
    }

    // Propagate static shapes & dtypes across all graph ops
    for (const node of parsedNodes) {
      const ins = node.inputs.map((inp) => (inp ? tensorMeta.get(inp) : undefined));
      const m0 = ins[0];
      if (!m0) continue;

      let outMetas: StaticTensorMeta[] | null = null;
      switch (node.opType) {
        case 'Identity':
        case 'LeakyRelu':
        case 'Relu':
        case 'Abs':
        case 'Floor':
        case 'Neg':
          outMetas = [{ elemType: m0.elemType, dims: [...m0.dims], intVals: m0.intVals }];
          break;
        case 'Cast': {
          const to = node.attrs.get('to')?.i;
          if (to) outMetas = [{ elemType: to, dims: [...m0.dims], intVals: m0.intVals }];
          break;
        }
        case 'Add':
        case 'Sub':
        case 'Mul':
        case 'Min':
        case 'Max': {
          const m1 = ins[1];
          if (m1) {
            const d = broadcastStaticDims(m0.dims, m1.dims);
            if (d) {
              let intVals: number[] | undefined;
              if (
                m0.intVals &&
                m1.intVals &&
                (d.length === 0 || (d.length === 1 && d[0] <= 16))
              ) {
                const len = d.length === 0 ? 1 : d[0];
                intVals = Array.from({ length: len }, (_, i) => {
                  const va = m0.intVals![m0.intVals!.length === 1 ? 0 : i];
                  const vb = m1.intVals![m1.intVals!.length === 1 ? 0 : i];
                  if (node.opType === 'Add') return va + vb;
                  if (node.opType === 'Sub') return va - vb;
                  if (node.opType === 'Mul') return va * vb;
                  if (node.opType === 'Min') return Math.min(va, vb);
                  return Math.max(va, vb);
                });
              }
              outMetas = [{ elemType: m0.elemType, dims: d, intVals }];
            }
          }
          break;
        }
        case 'Transpose': {
          const perm = node.attrs.get('perm')?.ints;
          if (perm && perm.length === m0.dims.length) {
            outMetas = [{ elemType: m0.elemType, dims: perm.map((p) => m0.dims[p]) }];
          }
          break;
        }
        case 'Reshape': {
          const m1 = ins[1];
          if (m1?.intVals) {
            const inTotal = m0.dims.reduce((a, c) => a * c, 1);
            const target = m1.intVals.map((d, idx) => (d === 0 ? m0.dims[idx] : d));
            const negIdx = target.indexOf(-1);
            if (negIdx >= 0) {
              const knownProd = target.reduce(
                (a, c, idx) => (idx === negIdx ? a : a * c),
                1
              );
              if (knownProd > 0 && inTotal % knownProd === 0) {
                target[negIdx] = inTotal / knownProd;
              }
            }
            if (target.every((d) => d > 0)) {
              outMetas = [{ elemType: m0.elemType, dims: target }];
            }
          }
          break;
        }
        case 'Shape':
          outMetas = [{ elemType: 7, dims: [m0.dims.length], intVals: [...m0.dims] }];
          break;
        case 'Gather': {
          const axisRaw = node.attrs.get('axis')?.i ?? 0;
          const axis = axisRaw < 0 ? m0.dims.length + axisRaw : axisRaw;
          const m1 = ins[1];
          if (axis === 0 && m1 && m1.dims.length <= 1 && m1.intVals && m1.intVals.length === 1) {
            const idx0 = m1.intVals[0] < 0 ? m0.dims[0] + m1.intVals[0] : m1.intVals[0];
            if (idx0 >= 0 && idx0 < m0.dims[0]) {
              const val = m0.intVals ? m0.intVals[idx0] : undefined;
              outMetas = [
                {
                  elemType: m0.elemType,
                  dims: [...m1.dims, ...m0.dims.slice(1)],
                  intVals: val !== undefined ? [val] : undefined,
                },
              ];
            }
          }
          break;
        }
        case 'Concat': {
          const axRaw = node.attrs.get('axis')?.i ?? 0;
          const ax = axRaw < 0 ? m0.dims.length + axRaw : axRaw;
          if (ax >= 0 && ax < m0.dims.length && ins.every((x) => Boolean(x))) {
            const outDims = [...m0.dims];
            let sum = 0;
            for (const inp of ins) sum += inp!.dims[ax];
            outDims[ax] = sum;
            const intVals =
              outDims.length === 1 && ins.every((x) => Array.isArray(x?.intVals))
                ? ins.flatMap((x) => x!.intVals!)
                : undefined;
            outMetas = [{ elemType: m0.elemType, dims: outDims, intVals }];
          }
          break;
        }
        case 'Unsqueeze': {
          const axes = node.attrs.get('axes')?.ints ?? ins[1]?.intVals;
          if (axes) {
            const outRank = m0.dims.length + axes.length;
            const normAxes = axes
              .map((a) => (a < 0 ? outRank + a : a))
              .sort((x, y) => x - y);
            const outDims: number[] = [];
            let j = 0;
            for (let i = 0; i < outRank; i++) {
              if (normAxes.includes(i)) outDims.push(1);
              else outDims.push(m0.dims[j++]);
            }
            outMetas = [{ elemType: m0.elemType, dims: outDims }];
          }
          break;
        }
        case 'Squeeze': {
          const axes = node.attrs.get('axes')?.ints ?? ins[1]?.intVals ?? [];
          const normAxes = new Set(axes.map((a) => (a < 0 ? m0.dims.length + a : a)));
          outMetas = [
            {
              elemType: m0.elemType,
              dims: m0.dims.filter((d, i) => (normAxes.size > 0 ? !normAxes.has(i) : d !== 1)),
            },
          ];
          break;
        }
        case 'ReduceMin':
        case 'ReduceSum':
        case 'ArgMin': {
          const keepdims = node.attrs.get('keepdims')?.i ?? 1;
          const axes = node.attrs.get('axes')?.ints ?? [node.attrs.get('axis')?.i ?? 0];
          const normAxes = new Set(axes.map((a) => (a < 0 ? m0.dims.length + a : a)));
          const outDims: number[] = [];
          m0.dims.forEach((d, i) => {
            if (normAxes.has(i)) {
              if (keepdims) outDims.push(1);
            } else {
              outDims.push(d);
            }
          });
          outMetas = [{ elemType: node.opType === 'ArgMin' ? 7 : m0.elemType, dims: outDims }];
          break;
        }
        case 'Tile': {
          const m1 = ins[1];
          if (m1?.intVals && m1.intVals.length === m0.dims.length) {
            outMetas = [
              {
                elemType: m0.elemType,
                dims: m0.dims.map((d, i) => d * m1.intVals![i]),
              },
            ];
          }
          break;
        }
        case 'Expand': {
          const m1 = ins[1];
          if (m1?.intVals) {
            const d = broadcastStaticDims(m0.dims, m1.intVals);
            if (d) outMetas = [{ elemType: m0.elemType, dims: d }];
          }
          break;
        }
        case 'Slice': {
          const ms = ins[1]?.intVals;
          const me = ins[2]?.intVals;
          const ma = ins[3]?.intVals ?? Array.from({ length: ms?.length ?? 0 }, (_, i) => i);
          const mst = ins[4]?.intVals ?? Array.from({ length: ms?.length ?? 0 }, () => 1);
          if (ms && me && ms.length === me.length && ma.length === ms.length) {
            const outDims = [...m0.dims];
            let ok = true;
            for (let i = 0; i < ma.length; i++) {
              const ax = ma[i] < 0 ? m0.dims.length + ma[i] : ma[i];
              if (ax < 0 || ax >= m0.dims.length) {
                ok = false;
                break;
              }
              const dim = m0.dims[ax];
              const step = mst[i];
              if (step === 0) {
                ok = false;
                break;
              }
              let s = ms[i] < 0 ? dim + ms[i] : ms[i];
              let e = me[i] < 0 ? dim + me[i] : me[i];
              if (step > 0) {
                s = Math.max(0, Math.min(dim, s));
                e = Math.max(0, Math.min(dim, e));
                outDims[ax] = e <= s ? 0 : Math.ceil((e - s) / step);
              } else {
                s = Math.max(0, Math.min(dim - 1, s));
                e = Math.max(-1, Math.min(dim - 1, e));
                outDims[ax] = s <= e ? 0 : Math.ceil((s - e) / -step);
              }
            }
            if (ok) outMetas = [{ elemType: m0.elemType, dims: outDims }];
          }
          break;
        }
        case 'Conv': {
          const w = ins[1];
          if (w && m0.dims.length === 4 && w.dims.length === 4) {
            const ks = node.attrs.get('kernel_shape')?.ints ?? [w.dims[2], w.dims[3]];
            const st = node.attrs.get('strides')?.ints ?? [1, 1];
            const dil = node.attrs.get('dilations')?.ints ?? [1, 1];
            const pads = node.attrs.get('pads')?.ints ?? [0, 0, 0, 0];
            const oH =
              Math.floor((m0.dims[2] + pads[0] + pads[2] - dil[0] * (ks[0] - 1) - 1) / st[0]) + 1;
            const oW =
              Math.floor((m0.dims[3] + pads[1] + pads[3] - dil[1] * (ks[1] - 1) - 1) / st[1]) + 1;
            outMetas = [{ elemType: m0.elemType, dims: [m0.dims[0], w.dims[0], oH, oW] }];
          }
          break;
        }
        case 'ConvTranspose': {
          const w = ins[1];
          if (w && m0.dims.length === 4 && w.dims.length === 4) {
            const ks = node.attrs.get('kernel_shape')?.ints ?? [w.dims[2], w.dims[3]];
            const st = node.attrs.get('strides')?.ints ?? [1, 1];
            const dil = node.attrs.get('dilations')?.ints ?? [1, 1];
            const pads = node.attrs.get('pads')?.ints ?? [0, 0, 0, 0];
            const opad = node.attrs.get('output_padding')?.ints ?? [0, 0];
            const g = node.attrs.get('group')?.i ?? 1;
            const outC = w.dims[1] * g;
            const oH =
              st[0] * (m0.dims[2] - 1) + ((ks[0] - 1) * dil[0] + 1) - pads[0] - pads[2] + opad[0];
            const oW =
              st[1] * (m0.dims[3] - 1) + ((ks[1] - 1) * dil[1] + 1) - pads[1] - pads[3] + opad[1];
            outMetas = [{ elemType: m0.elemType, dims: [m0.dims[0], outC, oH, oW] }];
          } else if (w && m0.dims.length === 5 && w.dims.length === 5) {
            const ks = node.attrs.get('kernel_shape')?.ints ?? [w.dims[2], w.dims[3], w.dims[4]];
            const st = node.attrs.get('strides')?.ints ?? [1, 1, 1];
            const dil = node.attrs.get('dilations')?.ints ?? [1, 1, 1];
            const pads = node.attrs.get('pads')?.ints ?? [0, 0, 0, 0, 0, 0];
            const opad = node.attrs.get('output_padding')?.ints ?? [0, 0, 0];
            const g = node.attrs.get('group')?.i ?? 1;
            const outC = w.dims[1] * g;
            const oD =
              st[0] * (m0.dims[2] - 1) + ((ks[0] - 1) * dil[0] + 1) - pads[0] - pads[3] + opad[0];
            const oH =
              st[1] * (m0.dims[3] - 1) + ((ks[1] - 1) * dil[1] + 1) - pads[1] - pads[4] + opad[1];
            const oW =
              st[2] * (m0.dims[4] - 1) + ((ks[2] - 1) * dil[2] + 1) - pads[2] - pads[5] + opad[2];
            outMetas = [{ elemType: m0.elemType, dims: [m0.dims[0], outC, oD, oH, oW] }];
          }
          break;
        }
        case 'Split': {
          const axRaw = node.attrs.get('axis')?.i ?? 0;
          const ax = axRaw < 0 ? m0.dims.length + axRaw : axRaw;
          const splits =
            node.attrs.get('split')?.ints ??
            Array.from({ length: node.outputs.length }, () =>
              Math.floor(m0.dims[ax] / node.outputs.length)
            );
          outMetas = splits.map((sp) => {
            const d = [...m0.dims];
            d[ax] = sp;
            return { elemType: m0.elemType, dims: d };
          });
          break;
        }
        case 'GatherND': {
          const m1 = ins[1];
          const bdims = node.attrs.get('batch_dims')?.i ?? 0;
          if (m1 && m1.dims.length >= 1 && bdims >= 0 && bdims < m1.dims.length) {
            const k = m1.dims[m1.dims.length - 1];
            if (bdims + k <= m0.dims.length) {
              outMetas = [
                {
                  elemType: m0.elemType,
                  dims: [...m1.dims.slice(0, -1), ...m0.dims.slice(bdims + k)],
                },
              ];
            }
          }
          break;
        }
      }

      if (outMetas && outMetas.length === node.outputs.length) {
        for (let i = 0; i < node.outputs.length; i++) {
          if (node.outputs[i]) {
            tensorMeta.set(node.outputs[i], outMetas[i]);
          }
        }
      }
    }

    if (requireHitnetContract) {
      const refOutMeta = tensorMeta.get('reference_output_disparity');
      const secOutMeta = tensorMeta.get('secondary_output_disparity');
      if (
        !refOutMeta ||
        !secOutMeta ||
        refOutMeta.elemType !== 1 ||
        secOutMeta.elemType !== 1 ||
        !sameStaticDims(refOutMeta.dims, [1, 240, 320, 1]) ||
        !sameStaticDims(secOutMeta.dims, [1, 240, 320, 1])
      ) {
        return emptyResult;
      }
    }

    const graphOutputNames = new Set(parsedGraphOutputs.map((o) => o.name));
    const argMinEntries: Array<{
      nodeIdx: number;
      node: ParsedNodeEntry;
      inDims: number[];
      outDims: number[];
      axis: number;
      keepdims: number;
      selectLastIndex: number;
    }> = [];
    let skippedDynamicCount = 0;
    let skippedInvalidOrUnprovenCount = 0;

    for (let nodeIdx = 0; nodeIdx < parsedNodes.length; nodeIdx++) {
      const node = parsedNodes[nodeIdx];
      if (node.opType !== 'ArgMin') continue;

      if (node.inputs.length !== 1 || node.outputs.length !== 1 || !node.inputs[0] || !node.outputs[0]) {
        skippedInvalidOrUnprovenCount++;
        continue;
      }
      const inMeta = tensorMeta.get(node.inputs[0]);
      const outMeta = tensorMeta.get(node.outputs[0]);
      if (!inMeta || !outMeta) {
        skippedDynamicCount++;
        continue;
      }

      const axis = node.attrs.get('axis')?.i ?? 0;
      const keepdims = node.attrs.get('keepdims')?.i ?? 1;
      const selectLastIndex = node.attrs.get('select_last_index')?.i ?? 0;

      if (
        inMeta.elemType !== 1 ||
        outMeta.elemType !== 7 ||
        inMeta.dims.length !== 4 ||
        axis !== 3 ||
        selectLastIndex !== 0
      ) {
        skippedInvalidOrUnprovenCount++;
        continue;
      }

      argMinEntries.push({
        nodeIdx,
        node,
        inDims: inMeta.dims,
        outDims: outMeta.dims,
        axis,
        keepdims,
        selectLastIndex,
      });
    }

    if (argMinEntries.length === 0 || skippedDynamicCount > 0 || skippedInvalidOrUnprovenCount > 0) {
      return {
        ...emptyResult,
        totalArgMinCount:
          argMinEntries.length + skippedDynamicCount + skippedInvalidOrUnprovenCount,
        skippedDynamicCount,
        skippedInvalidOrUnprovenCount,
      };
    }

    const totalArgMinCount = argMinEntries.length;
    if (requireHitnetContract && totalArgMinCount !== 4) {
      return { ...emptyResult, totalArgMinCount };
    }

    // Determine target ArgMin nodes to patch vs leave untouched
    const targetEntries: typeof argMinEntries = [];
    const untouchedNodeNames: string[] = [];

    if (requireHitnetContract) {
      for (const entry of argMinEntries) {
        if (
          (entry.node.name === 'level0/level_init/ArgMin' ||
            entry.node.name === 'level0_1/level_init/ArgMin') &&
          sameStaticDims(entry.inDims, [1, 60, 80, 320]) &&
          sameStaticDims(entry.outDims, [1, 60, 80]) &&
          entry.keepdims === 0
        ) {
          untouchedNodeNames.push(entry.node.name);
        } else if (
          (entry.node.name === 'level0/shared/prop0/ArgMin' ||
            entry.node.name === 'level0_1/shared/prop0/ArgMin') &&
          sameStaticDims(entry.inDims, [1, 60, 80, 1]) &&
          sameStaticDims(entry.outDims, [1, 60, 80]) &&
          entry.keepdims === 0
        ) {
          targetEntries.push(entry);
        } else {
          return {
            ...emptyResult,
            totalArgMinCount,
            skippedInvalidOrUnprovenCount: skippedInvalidOrUnprovenCount + 1,
          };
        }
      }
      const hasL0Init = untouchedNodeNames.includes('level0/level_init/ArgMin');
      const hasL01Init = untouchedNodeNames.includes('level0_1/level_init/ArgMin');
      const hasL0Prop = targetEntries.some((e) => e.node.name === 'level0/shared/prop0/ArgMin');
      const hasL01Prop = targetEntries.some((e) => e.node.name === 'level0_1/shared/prop0/ArgMin');
      if (!hasL0Init || !hasL01Init || !hasL0Prop || !hasL01Prop) {
        return { ...emptyResult, totalArgMinCount };
      }
    } else {
      // Synthetic non-HITNet diagnostic control: only patch smaller candidate-size rank-4 ArgMin(keepdims=0)
      // when distinct candidate counts exist in the graph.
      const axis3Sizes = argMinEntries.map((e) => e.inDims[3]);
      const minAxis3 = Math.min(...axis3Sizes);
      const maxAxis3 = Math.max(...axis3Sizes);
      if (minAxis3 === maxAxis3) {
        return { ...emptyResult, totalArgMinCount };
      }
      for (const entry of argMinEntries) {
        if (entry.inDims[3] === minAxis3 && entry.keepdims === 0) {
          targetEntries.push(entry);
        } else {
          untouchedNodeNames.push(entry.node.name || `ArgMin_${entry.nodeIdx}`);
        }
      }
      if (targetEntries.length === 0) {
        return { ...emptyResult, totalArgMinCount };
      }
    }

    // Prove every target ArgMin output reaches ONLY its shape-normalizing Reshape(..., [d0,d1,d2,1])
    const patchedOutputShapeByTensor = new Map<string, number[]>();
    const nodeIdxToNewDims = new Map<number, number[]>();

    for (const entry of targetEntries) {
      const outName = entry.node.outputs[0];
      if (graphOutputNames.has(outName)) {
        return { ...emptyResult, totalArgMinCount, skippedInvalidOrUnprovenCount: 1 };
      }
      const consumers = consumersByTensor.get(outName) ?? [];
      if (consumers.length !== 1 || consumers[0].slot !== 0) {
        return { ...emptyResult, totalArgMinCount, skippedInvalidOrUnprovenCount: 1 };
      }
      const consumerNode = parsedNodes[consumers[0].nodeIdx];
      if (consumerNode.opType !== 'Reshape' || consumerNode.outputs.length !== 1) {
        return { ...emptyResult, totalArgMinCount, skippedInvalidOrUnprovenCount: 1 };
      }
      const shapeInputMeta = consumerNode.inputs[1]
        ? tensorMeta.get(consumerNode.inputs[1])
        : undefined;
      const reshapeOutMeta = tensorMeta.get(consumerNode.outputs[0]);
      const expectedNormalizedDims = [entry.inDims[0], entry.inDims[1], entry.inDims[2], 1];
      if (
        !shapeInputMeta?.intVals ||
        !reshapeOutMeta ||
        shapeInputMeta.elemType !== 7 ||
        !sameStaticDims(shapeInputMeta.intVals, expectedNormalizedDims) ||
        reshapeOutMeta.elemType !== 7 ||
        !sameStaticDims(reshapeOutMeta.dims, expectedNormalizedDims)
      ) {
        return { ...emptyResult, totalArgMinCount, skippedInvalidOrUnprovenCount: 1 };
      }

      patchedOutputShapeByTensor.set(outName, expectedNormalizedDims);
      nodeIdxToNewDims.set(entry.nodeIdx, expectedNormalizedDims);
    }

    const concatBytes = (parts: readonly Uint8Array[]): Uint8Array => {
      let total = 0;
      for (const p of parts) total += p.byteLength;
      const out = new Uint8Array(total);
      let off = 0;
      for (const p of parts) {
        out.set(p, off);
        off += p.byteLength;
      }
      return out;
    };

    const encodeLenField = (fn: number, payload: Uint8Array): Uint8Array => {
      const tag = encodeProtoVarint((fn << 3) | 2);
      const len = encodeProtoVarint(payload.byteLength);
      const out = new Uint8Array(tag.byteLength + len.byteLength + payload.byteLength);
      out.set(tag, 0);
      out.set(len, tag.byteLength);
      out.set(payload, tag.byteLength + len.byteLength);
      return out;
    };

    const encodeVarintField = (fn: number, val: number): Uint8Array =>
      concatBytes([encodeProtoVarint((fn << 3) | 0), encodeProtoVarint(val)]);

    const encodeAttrInt = (attrName: string, val: number): Uint8Array =>
      encodeLenField(
        5,
        concatBytes([
          encodeLenField(1, encoder.encode(attrName)),
          concatBytes([encodeProtoVarint((3 << 3) | 0), encodeSignedVarint64Bytes(val)]),
          encodeVarintField(20, 2),
        ])
      );

    const encodeValueInfoTensor = (
      tensorName: string,
      elemType: number,
      dims: readonly number[]
    ): Uint8Array => {
      const dimMsgs = dims.map((d) => encodeLenField(1, encodeVarintField(1, d)));
      const tensorShape = encodeLenField(2, concatBytes(dimMsgs));
      const tensorType = encodeLenField(
        1,
        concatBytes([encodeVarintField(1, elemType), tensorShape])
      );
      const typeProto = encodeLenField(2, tensorType);
      return encodeLenField(
        13,
        concatBytes([encodeLenField(1, encoder.encode(tensorName)), typeProto])
      );
    };

    const rewriteArgMinKeepDimsNode = (node: ParsedNodeEntry): Uint8Array => {
      const parts: Uint8Array[] = [];
      let replacedKeepDims = false;
      for (const nf of node.nodeFields) {
        if (nf.fn === 5 && nf.wt === 2) {
          const af = parseProtoFields(bytes, nf.valStart, nf.valEnd);
          let attrName = '';
          if (af) {
            for (const a of af) {
              if (a.fn === 1 && a.wt === 2 && a.valEnd > a.valStart) {
                attrName = decoder.decode(bytes.subarray(a.valStart, a.valEnd));
              }
            }
          }
          if (attrName === 'keepdims') {
            parts.push(encodeAttrInt('keepdims', 1));
            replacedKeepDims = true;
            continue;
          }
        }
        parts.push(bytes.subarray(nf.fStart, nf.fEnd));
      }
      if (!replacedKeepDims) {
        parts.push(encodeAttrInt('keepdims', 1));
      }
      return encodeLenField(1, concatBytes(parts));
    };

    const normalizedNodeNames: string[] = [];
    let updatedValueInfoCount = 0;
    const seenPatchedValueInfos = new Set<string>();
    const finalGraphParts: Uint8Array[] = [];
    let currentNodeIdx = 0;

    for (const gf of graphFields) {
      if (gf.fn === 13 && gf.wt === 2) {
        const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
        if (vi && patchedOutputShapeByTensor.has(vi.name)) {
          finalGraphParts.push(
            encodeValueInfoTensor(vi.name, 7, patchedOutputShapeByTensor.get(vi.name)!)
          );
          seenPatchedValueInfos.add(vi.name);
          updatedValueInfoCount++;
        } else {
          finalGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        }
        continue;
      }
      if (gf.fn === 1 && gf.wt === 2) {
        const node = parsedNodes[currentNodeIdx];
        if (nodeIdxToNewDims.has(currentNodeIdx)) {
          finalGraphParts.push(rewriteArgMinKeepDimsNode(node));
          normalizedNodeNames.push(node.name || `ArgMin_${currentNodeIdx}`);
        } else {
          finalGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        }
        currentNodeIdx++;
      } else {
        finalGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
      }
    }
    for (const [tensorName, newDims] of patchedOutputShapeByTensor.entries()) {
      if (!seenPatchedValueInfos.has(tensorName)) {
        finalGraphParts.push(encodeValueInfoTensor(tensorName, 7, newDims));
        updatedValueInfoCount++;
      }
    }

    let graphPayloadLen = 0;
    for (const part of finalGraphParts) graphPayloadLen += part.byteLength;

    const graphTag = encodeProtoVarint((7 << 3) | 2);
    const graphLen = encodeProtoVarint(graphPayloadLen);
    let totalLen = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        totalLen += graphTag.byteLength + graphLen.byteLength + graphPayloadLen;
      } else {
        totalLen += topFields[i].fEnd - topFields[i].fStart;
      }
    }

    const out = new Uint8Array(totalLen);
    let offset = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        out.set(graphTag, offset);
        offset += graphTag.byteLength;
        out.set(graphLen, offset);
        offset += graphLen.byteLength;
        for (const part of finalGraphParts) {
          out.set(part, offset);
          offset += part.byteLength;
        }
      } else {
        const chunk = bytes.subarray(topFields[i].fStart, topFields[i].fEnd);
        out.set(chunk, offset);
        offset += chunk.byteLength;
      }
    }

    const beforeValidation = validateOnnxGraphNodeInputs(bytes);
    const afterValidation = validateOnnxGraphNodeInputs(out);
    const preexistingMissing = new Set(beforeValidation.missingInputs.map((m) => m.inputName));
    const newlyMissing = afterValidation.missingInputs.filter(
      (m) => !preexistingMissing.has(m.inputName)
    );
    if (
      newlyMissing.length > 0 ||
      (beforeValidation.hasGraphInputs && !afterValidation.valid)
    ) {
      const sample = newlyMissing[0] ?? afterValidation.missingInputs[0];
      console.warn(
        `[StereoDepth] Aborting ArgMin keepdims rewrite due to unresolved node input '${sample?.inputName ?? 'unknown'}' on '${sample?.nodeName ?? 'unknown'}'`
      );
      return emptyResult;
    }

    const ortCacheKeyBefore = computeOrtArgMinCacheAttributesKey({
      axis: 3,
      keepDims: false,
      selectLastIndex: 0,
      inputType: 1,
      inputRank: 4,
      is1DimensionDispatch: false,
    });
    const ortCacheKeyAfter = computeOrtArgMinCacheAttributesKey({
      axis: 3,
      keepDims: true,
      selectLastIndex: 0,
      inputType: 1,
      inputRank: 4,
      is1DimensionDispatch: false,
    });
    if (ortCacheKeyBefore === ortCacheKeyAfter) {
      return emptyResult;
    }

    return {
      bytes: out,
      normalizedCount: normalizedNodeNames.length,
      normalizedNodeNames,
      untouchedNodeNames,
      totalArgMinCount,
      skippedDynamicCount,
      skippedInvalidOrUnprovenCount,
      updatedValueInfoCount,
      ortCacheKeyBefore,
      ortCacheKeyAfter,
    };
  } catch {
    return emptyResult;
  }
}

export const HITNET_RAW_MODEL_SHA256 =
  '67df2f91798a169ab8113574f122a07403197e21b9d3dc71029cebd06d96284c';
export const HITNET_H1_FULL_GRAPH_SHA256 =
  '3d7278a54d3a5d8d356b71716da72b901ed75f7c21ec012f420d17205b6a69ed';
export const HITNET_M1_FULL_GRAPH_SHA256 =
  '97e3673bdea94980f71cc1fc13998066238636b0ae0fa7f5381aa3ffa711e758';
export const HITNET_M1_FULL_GRAPH_BYTE_LENGTH = 8888846 as const;

function readExactInt64InitializerBigInts(
  buf: Uint8Array,
  slice: ProtobufFieldSlice,
  decoder: TextDecoder
): {
  name: string;
  elemType: number;
  dims: number[];
  values: bigint[];
  hasExternal: boolean;
} | null {
  const tensorFields = parseProtoFields(buf, slice.valStart, slice.valEnd);
  if (!tensorFields) return null;
  let name = '';
  let elemType = 0;
  const dims: number[] = [];
  let rawData: Uint8Array | null = null;
  const i64s: bigint[] = [];
  let hasExternal = false;

  for (const tf of tensorFields) {
    if (tf.fn === 1) {
      if (tf.wt === 0) {
        const v = readProtoVarintBigInt(buf, tf.valStart, tf.valEnd);
        if (v) dims.push(Number(BigInt.asIntN(64, v[0])));
      } else if (tf.wt === 2) {
        let p = tf.valStart;
        while (p < tf.valEnd) {
          const v = readProtoVarintBigInt(buf, p, tf.valEnd);
          if (!v) break;
          dims.push(Number(BigInt.asIntN(64, v[0])));
          p = v[1];
        }
      }
    } else if (tf.fn === 2 && tf.wt === 0) {
      elemType = readProtoVarint(buf, tf.valStart, tf.valEnd)?.[0] ?? 0;
    } else if (tf.fn === 7) {
      if (tf.wt === 0) {
        const v = readProtoVarintBigInt(buf, tf.valStart, tf.valEnd);
        if (v) i64s.push(BigInt.asIntN(64, v[0]));
      } else if (tf.wt === 2) {
        let p = tf.valStart;
        while (p < tf.valEnd) {
          const v = readProtoVarintBigInt(buf, p, tf.valEnd);
          if (!v) break;
          i64s.push(BigInt.asIntN(64, v[0]));
          p = v[1];
        }
      }
    } else if (tf.fn === 8 && tf.wt === 2 && tf.valEnd > tf.valStart) {
      name = decoder.decode(buf.subarray(tf.valStart, tf.valEnd));
    } else if (tf.fn === 9 && tf.wt === 2) {
      rawData = buf.subarray(tf.valStart, tf.valEnd);
    } else if (tf.fn === 13 && tf.wt === 2) {
      hasExternal = true;
    } else if (tf.fn === 14 && tf.wt === 0) {
      const v = readProtoVarint(buf, tf.valStart, tf.valEnd);
      if (v && v[0] !== 0) hasExternal = true;
    }
  }
  if (!name || elemType !== 7 || hasExternal) return null;
  let values: bigint[];
  if (rawData && rawData.byteLength > 0 && rawData.byteLength % 8 === 0) {
    const dv = new DataView(rawData.buffer, rawData.byteOffset, rawData.byteLength);
    values = Array.from({ length: rawData.byteLength / 8 }, (_, i) =>
      dv.getBigInt64(i * 8, true)
    );
  } else if (i64s.length > 0) {
    values = i64s;
  } else {
    return null;
  }
  return { name, elemType, dims, values, hasExternal };
}

/**
 * Guarded M1-ConstZero2Ch pass for `hitnet-sf` WebGPU built strictly on top of H1 (never L1):
 * Replaces both duplicated verified-zero input slots (`ch1` and `ch2` slant placeholders) in
 * `level0/level_init/init_to_prop/concat_1` and `level0_1/level_init/init_to_prop/concat_1`
 * with a single static `float32 [1,2,60,80]` `+0.0f` initializer while preserving `ch0` disparity,
 * `ch3..15` LeakyRelu features, exact node/output names, and output shapes `[1,16,60,80]` -> `[1,60,80,1,16]`.
 *
 * Atomically fails closed with no mutation unless:
 *   1. Exact `irVersion=7`, `defaultOpset=12`, raw HITNet I/O contract (and SHA-256 when provided) hold.
 *   2. Exact H1 provenance holds on both branches (5D convs decomposed, slices compacted, 4 F1 warp_1d rank-4,
 *      2 H1 prop0 rank-4 GatherNDs, both `level_init/GatherV2_1` untouched rank-5, no L1-Reshape/L1-ReduceMin, no G1).
 *   3. Both `init_to_prop/concat_1` (`Concat(axis=1)`) branches match exact 4-slot signatures where slots 1 & 2
 *      share one tensor produced by bounded `ArgMin(axis=3,keepdims=0) -> ExpandDims_2 -> level_init/Cast(to=FLOAT)`
 *      followed by `Cast(to=INT64) -> Mul(scalar int64 0n initializer) -> Cast(to=FLOAT) -> Reshape([1,1,60,80])`.
 *   4. All intermediate dead tensors & `zero__310` initializer have zero other consumers or graph outputs before removal.
 */
export function normalizeHitnetInitToPropZeroSlantForWebGpu(
  bytes: Uint8Array,
  options: NormalizeInitToPropZeroSlantOptions = {}
): OnnxInitToPropZeroSlantNormalizeResult {
  const emptyResult: OnnxInitToPropZeroSlantNormalizeResult = {
    bytes,
    normalizedCount: 0,
    normalizedNodeNames: [],
    removedDeadNodeNames: [],
    removedDeadInitializerNames: [],
    addedInitializerName: null,
    addedInitializerDims: null,
    verifiedH1Provenance: false,
    verifiedRawIdentity: false,
    skippedInvalidOrUnprovenCount: 0,
  };
  const requireHitnetContract = options.requireHitnetContract ?? true;
  if (
    (options.backendPreference === 'wasm' && !options.allowDiagnosticTransformedWasm) ||
    bytes.byteLength < 16
  ) {
    return emptyResult;
  }
  if (requireHitnetContract && options.variantId !== 'hitnet-sf') {
    return emptyResult;
  }
  if (options.rawModelSha256 !== undefined && options.rawModelSha256 !== HITNET_RAW_MODEL_SHA256) {
    return { ...emptyResult, skippedInvalidOrUnprovenCount: 1 };
  }
  if (options.h1ModelSha256 !== undefined && options.h1ModelSha256 !== HITNET_H1_FULL_GRAPH_SHA256) {
    return { ...emptyResult, skippedInvalidOrUnprovenCount: 1 };
  }

  try {
    const preInference = inferOnnxStaticTensorMeta(bytes);
    if (
      !preInference.valid ||
      preInference.hasControlFlowNode ||
      preInference.duplicateOutputs.length > 0 ||
      preInference.duplicateInitializers.length > 0 ||
      preInference.duplicateNodeNames.length > 0
    ) {
      return emptyResult;
    }

    if (preInference.irVersion !== 7) {
      return emptyResult;
    }
    if (requireHitnetContract) {
      if (preInference.defaultOpset !== 12) {
        return emptyResult;
      }
    } else if (
      preInference.defaultOpset !== null &&
      (preInference.defaultOpset < 11 || preInference.defaultOpset > 17)
    ) {
      return emptyResult;
    }

    const topFields = parseProtoFields(bytes, 0, bytes.byteLength);
    if (!topFields) return emptyResult;
    const graphIndex = topFields.findIndex((f) => f.fn === 7 && f.wt === 2);
    if (graphIndex < 0) return emptyResult;
    const graphField = topFields[graphIndex];
    const graphFields = parseProtoFields(bytes, graphField.valStart, graphField.valEnd);
    if (!graphFields) return emptyResult;

    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    const usedNames = new Set<string>();
    const allInitializerNames = new Set<string>();
    const int64InitMap = new Map<string, { dims: number[]; values: bigint[] }>();
    const parsedGraphInputs: ParsedStaticValueInfo[] = [];
    const parsedGraphOutputs: ParsedStaticValueInfo[] = [];
    const graphOutputSet = new Set<string>();

    interface M1ParsedNode {
      nodeIdx: number;
      gf: ProtobufFieldSlice;
      nodeFields: ProtobufFieldSlice[];
      opType: string;
      name: string;
      inputs: string[];
      outputs: string[];
      attrs: Map<string, { i?: number; ints?: number[] }>;
    }
    const parsedNodes: M1ParsedNode[] = [];
    const producerByOutput = new Map<string, M1ParsedNode>();
    const consumersByTensor = new Map<string, Set<number>>();

    for (const gf of graphFields) {
      if (gf.fn === 5 && gf.wt === 2) {
        const nm = getInitializerName(bytes, gf.valStart, gf.valEnd, decoder);
        if (nm) {
          usedNames.add(nm);
          allInitializerNames.add(nm);
        }
        const i64 = readExactInt64InitializerBigInts(bytes, gf, decoder);
        if (i64) {
          int64InitMap.set(i64.name, { dims: i64.dims, values: i64.values });
        }
        continue;
      }
      if (gf.fn === 11 && gf.wt === 2) {
        const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
        if (vi) {
          usedNames.add(vi.name);
          parsedGraphInputs.push(vi);
        }
        continue;
      }
      if (gf.fn === 12 && gf.wt === 2) {
        const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
        if (vi) {
          usedNames.add(vi.name);
          parsedGraphOutputs.push(vi);
          graphOutputSet.add(vi.name);
        } else {
          const nm = getValueInfoName(bytes, gf.valStart, gf.valEnd, decoder);
          if (nm) {
            usedNames.add(nm);
            graphOutputSet.add(nm);
          }
        }
        continue;
      }
      if (gf.fn === 13 && gf.wt === 2) {
        const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
        if (vi) usedNames.add(vi.name);
        continue;
      }
      if (gf.fn !== 1 || gf.wt !== 2) continue;

      const nodeFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
      if (!nodeFields) return emptyResult;
      const inputs: string[] = [];
      const outputs: string[] = [];
      let name = '';
      let opType = '';
      const attrs = new Map<string, { i?: number; ints?: number[] }>();

      for (const nf of nodeFields) {
        if (nf.fn === 1 && nf.wt === 2) {
          const inp =
            nf.valEnd > nf.valStart ? decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)) : '';
          inputs.push(inp);
          if (inp) usedNames.add(inp);
        } else if (nf.fn === 2 && nf.wt === 2) {
          const outName =
            nf.valEnd > nf.valStart ? decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)) : '';
          outputs.push(outName);
          if (outName) usedNames.add(outName);
        } else if (nf.fn === 3 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          name = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
          usedNames.add(name);
        } else if (nf.fn === 4 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          opType = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
        } else if (nf.fn === 5 && nf.wt === 2) {
          const af = parseProtoFields(bytes, nf.valStart, nf.valEnd);
          if (!af) continue;
          let an = '';
          let ai: number | undefined;
          const aints: number[] = [];
          for (const a of af) {
            if (a.fn === 1 && a.wt === 2 && a.valEnd > a.valStart) {
              an = decoder.decode(bytes.subarray(a.valStart, a.valEnd));
            } else if (a.fn === 3 && a.wt === 0) {
              const v = readProtoVarintBigInt(bytes, a.valStart, a.valEnd);
              if (v) ai = Number(BigInt.asIntN(64, v[0]));
            } else if (a.fn === 8 && a.wt === 0) {
              const v = readProtoVarintBigInt(bytes, a.valStart, a.valEnd);
              if (v) aints.push(Number(BigInt.asIntN(64, v[0])));
            }
          }
          if (an) attrs.set(an, { i: ai, ints: aints });
        }
      }

      const nodeIdx = parsedNodes.length;
      const entry: M1ParsedNode = { nodeIdx, gf, nodeFields, opType, name, inputs, outputs, attrs };
      parsedNodes.push(entry);
      for (const o of outputs) {
        if (o) producerByOutput.set(o, entry);
      }
      for (const inp of inputs) {
        if (!inp) continue;
        let set = consumersByTensor.get(inp);
        if (!set) {
          set = new Set<number>();
          consumersByTensor.set(inp, set);
        }
        set.add(nodeIdx);
      }
    }

    const nonInitInputs = parsedGraphInputs.filter((g) => !allInitializerNames.has(g.name));
    if (
      nonInitInputs.length === 0 ||
      nonInitInputs.some((g) => g.dims === null || g.elemType === null)
    ) {
      return emptyResult;
    }

    let verifiedRawIdentity = !requireHitnetContract;
    let verifiedH1Provenance = !requireHitnetContract;

    if (requireHitnetContract) {
      if (
        nonInitInputs.length !== 1 ||
        nonInitInputs[0].name !== 'input' ||
        nonInitInputs[0].elemType !== 1 ||
        !nonInitInputs[0].dims ||
        !sameStaticDims(nonInitInputs[0].dims, [1, 6, 240, 320])
      ) {
        return emptyResult;
      }
      const outNames = parsedGraphOutputs.map((o) => o.name);
      if (
        outNames.length !== 2 ||
        !outNames.includes('reference_output_disparity') ||
        !outNames.includes('secondary_output_disparity')
      ) {
        return emptyResult;
      }
      const refMeta = preInference.tensorMeta['reference_output_disparity'];
      const secMeta = preInference.tensorMeta['secondary_output_disparity'];
      if (
        !refMeta ||
        !secMeta ||
        refMeta.elemType !== 1 ||
        secMeta.elemType !== 1 ||
        !sameStaticDims(refMeta.dims, [1, 240, 320, 1]) ||
        !sameStaticDims(secMeta.dims, [1, 240, 320, 1])
      ) {
        return emptyResult;
      }
      verifiedRawIdentity = true;

      // Verify exact H1 provenance (5D conv decomposed, slices compacted, F1 warp_1d, H1 prop0 rank-4)
      // and strictly reject either L1 variant (Reshape-rank4 / ReduceMin) or G1 (ArgMin keepdims=1).
      const has5dConv = parsedNodes.some((n) => {
        if (n.opType !== 'Conv') return false;
        const m = n.inputs[0] ? preInference.tensorMeta[n.inputs[0]] : undefined;
        return Boolean(m && m.dims.length >= 5);
      });
      const hasReduceMin = parsedNodes.some((n) => n.opType === 'ReduceMin');
      if (has5dConv || hasReduceMin) {
        return { ...emptyResult, verifiedRawIdentity, skippedInvalidOrUnprovenCount: 1 };
      }

      const gatherNdNodes = parsedNodes.filter((n) => n.opType === 'GatherND');
      if (gatherNdNodes.length !== 10) {
        return { ...emptyResult, verifiedRawIdentity, skippedInvalidOrUnprovenCount: 1 };
      }
      let f1Count = 0;
      let h1PropCount = 0;
      let untouchedLevelInitCount = 0;
      for (const gn of gatherNdNodes) {
        const bd = gn.attrs.get('batch_dims')?.i ?? 0;
        const dm = gn.inputs[0] ? preInference.tensorMeta[gn.inputs[0]] : undefined;
        const im = gn.inputs[1] ? preInference.tensorMeta[gn.inputs[1]] : undefined;
        const om = gn.outputs[0] ? preInference.tensorMeta[gn.outputs[0]] : undefined;
        if (!dm || !im || !om) {
          return { ...emptyResult, verifiedRawIdentity, skippedInvalidOrUnprovenCount: 1 };
        }
        if (
          (gn.name === 'level0/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis' ||
            gn.name === 'level0_1/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis') &&
          bd === 2 &&
          dm.elemType === 1 &&
          im.elemType === 7 &&
          om.elemType === 1 &&
          sameStaticDims(dm.dims, [1, 4800, 1, 16]) &&
          sameStaticDims(im.dims, [1, 4800, 1, 1]) &&
          sameStaticDims(om.dims, [1, 4800, 1, 16]) &&
          gn.inputs[0].includes('__h1_data_4d_') &&
          gn.inputs[1].includes('__h1_indices_4d_') &&
          gn.outputs[0].includes('__h1_out_4d_')
        ) {
          h1PropCount++;
        } else if (
          gn.name.includes('/warp_1d/') &&
          bd === 2 &&
          sameStaticDims(dm.dims, [1, 240, 320, 32]) &&
          sameStaticDims(im.dims, [1, 240, 960, 1]) &&
          sameStaticDims(om.dims, [1, 240, 960, 32]) &&
          gn.inputs[1].includes('__flat_indices_') &&
          gn.outputs[0].includes('__flat_out_')
        ) {
          f1Count++;
        } else if (
          (gn.name === 'level0/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis' ||
            gn.name === 'level0_1/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis') &&
          bd === 3 &&
          dm.elemType === 1 &&
          im.elemType === 7 &&
          om.elemType === 1 &&
          sameStaticDims(dm.dims, [1, 60, 80, 320]) &&
          sameStaticDims(im.dims, [1, 60, 80, 1, 1]) &&
          sameStaticDims(om.dims, [1, 60, 80, 1])
        ) {
          untouchedLevelInitCount++;
        }
      }
      if (h1PropCount !== 2 || f1Count !== 4 || untouchedLevelInitCount !== 2) {
        return { ...emptyResult, verifiedRawIdentity, skippedInvalidOrUnprovenCount: 1 };
      }

      const argMinNodes = parsedNodes.filter((n) => n.opType === 'ArgMin');
      if (
        argMinNodes.length !== 4 ||
        argMinNodes.some((an) => (an.attrs.get('keepdims')?.i ?? 1) !== 0)
      ) {
        return { ...emptyResult, verifiedRawIdentity, skippedInvalidOrUnprovenCount: 1 };
      }
      verifiedH1Provenance = true;
    }

    interface VerifiedBranchSpec {
      branchName: string;
      concatNode: M1ParsedNode;
      slot0DisparityTensor: string;
      slot12ZeroTensor: string;
      slot3FeatureTensor: string;
      deadNodeIndices: number[];
      deadNodeNames: string[];
      deadTensorNames: string[];
      zeroInitName: string;
      batch0: number;
      height: number;
      width: number;
      totalOutChannels: number;
      expectedExpandDimsOut: string | null;
      expectedTransposeOut: string | null;
    }

    const expectedHitnetBranches = [
      {
        branchName: 'level0',
        concatName: 'level0/level_init/init_to_prop/concat_1',
        slot0: 'Transpose__908:0',
        slot12: 'Transpose__910:0',
        slot3: 'level0/level_init/init_to_prop/LeakyRelu',
        castF32Name: 'level0/level_init/Cast',
        castI64Name: 'Cast__309',
        mulName: 'Mul__311',
        zerosLikeName: 'level0/level_init/init_to_prop/zeros_like',
        reshapeZeroName: 'Transpose__910',
        zeroInitName: 'zero__310',
        transposeOutName: 'Transpose__913:0',
        expandDimsOutName: 'level0/level_init/init_to_prop/ExpandDims',
      },
      {
        branchName: 'level0_1',
        concatName: 'level0_1/level_init/init_to_prop/concat_1',
        slot0: 'Transpose__1006:0',
        slot12: 'Transpose__1008:0',
        slot3: 'level0_1/level_init/init_to_prop/LeakyRelu',
        castF32Name: 'level0_1/level_init/Cast',
        castI64Name: 'Cast__283',
        mulName: 'Mul__285',
        zerosLikeName: 'level0_1/level_init/init_to_prop/zeros_like',
        reshapeZeroName: 'Transpose__1008',
        zeroInitName: 'zero__310',
        transposeOutName: 'Transpose__1011:0',
        expandDimsOutName: 'level0_1/level_init/init_to_prop/ExpandDims',
      },
    ];

    const verifySingleBranch = (
      concatNode: M1ParsedNode,
      branchSpec?: (typeof expectedHitnetBranches)[number]
    ): VerifiedBranchSpec | null => {
      if (concatNode.opType !== 'Concat' || concatNode.outputs.length !== 1 || !concatNode.outputs[0]) {
        return null;
      }
      const axisAttr = concatNode.attrs.get('axis')?.i;
      if (axisAttr !== 1) return null;
      if (concatNode.inputs.length !== 4) return null;

      const [slot0, slot1, slot2, slot3] = concatNode.inputs;
      if (!slot0 || !slot1 || !slot2 || !slot3 || slot1 !== slot2 || slot0 === slot1 || slot3 === slot1 || slot0 === slot3) {
        return null;
      }
      if (branchSpec) {
        if (
          concatNode.name !== branchSpec.concatName ||
          concatNode.outputs[0] !== branchSpec.concatName ||
          slot0 !== branchSpec.slot0 ||
          slot1 !== branchSpec.slot12 ||
          slot3 !== branchSpec.slot3
        ) {
          return null;
        }
      }

      const m0 = preInference.tensorMeta[slot0];
      const m12 = preInference.tensorMeta[slot1];
      const m3 = preInference.tensorMeta[slot3];
      const mOut = preInference.tensorMeta[concatNode.outputs[0]];
      if (
        !m0 ||
        !m12 ||
        !m3 ||
        !mOut ||
        m0.elemType !== 1 ||
        m12.elemType !== 1 ||
        m3.elemType !== 1 ||
        mOut.elemType !== 1 ||
        m0.dims.length !== 4 ||
        m12.dims.length !== 4 ||
        m3.dims.length !== 4 ||
        mOut.dims.length !== 4
      ) {
        return null;
      }
      const [b0, c0, h, w] = m0.dims;
      if (c0 !== 1 || b0 <= 0 || h <= 0 || w <= 0) return null;
      if (requireHitnetContract && (b0 !== 1 || h !== 60 || w !== 80)) return null;
      if (!sameStaticDims(m12.dims, [b0, 1, h, w])) return null;
      if (m3.dims[0] !== b0 || m3.dims[2] !== h || m3.dims[3] !== w || m3.dims[1] <= 0) return null;
      if (requireHitnetContract && m3.dims[1] !== 13) return null;
      if (!sameStaticDims(mOut.dims, [b0, 1 + 2 + m3.dims[1], h, w])) return null;

      // Verify ch0 disparity producer remains bounded integer ArgMin -> Reshape -> Cast(to=FLOAT) -> Reshape
      const ch0Reshape = producerByOutput.get(slot0);
      if (!ch0Reshape || ch0Reshape.opType !== 'Reshape' || ch0Reshape.inputs.length < 2) return null;
      const castF32 = producerByOutput.get(ch0Reshape.inputs[0]);
      if (
        !castF32 ||
        castF32.opType !== 'Cast' ||
        castF32.inputs.length !== 1 ||
        (castF32.attrs.get('to')?.i ?? 0) !== 1
      ) {
        return null;
      }
      if (branchSpec && castF32.name !== branchSpec.castF32Name) return null;
      const expand2 = producerByOutput.get(castF32.inputs[0]);
      if (!expand2 || expand2.opType !== 'Reshape' || expand2.inputs.length < 2) return null;
      const expand2Meta = preInference.tensorMeta[expand2.outputs[0]];
      if (!expand2Meta || expand2Meta.elemType !== 7 || !sameStaticDims(expand2Meta.dims, [b0, h, w, 1])) {
        return null;
      }
      const argMin = producerByOutput.get(expand2.inputs[0]);
      if (!argMin || argMin.opType !== 'ArgMin' || argMin.inputs.length !== 1) return null;
      const argMinMeta = preInference.tensorMeta[argMin.inputs[0]];
      const argMinAxis = argMin.attrs.get('axis')?.i ?? 0;
      const argMinKeep = argMin.attrs.get('keepdims')?.i ?? 1;
      const argMinSelLast = argMin.attrs.get('select_last_index')?.i ?? 0;
      if (
        !argMinMeta ||
        argMinMeta.elemType !== 1 ||
        argMinMeta.dims.length !== 4 ||
        (argMinAxis !== 3 && argMinAxis !== -1) ||
        argMinKeep !== 0 ||
        argMinSelLast !== 0 ||
        argMinMeta.dims[3] <= 0 ||
        !sameStaticDims([argMinMeta.dims[0], argMinMeta.dims[1], argMinMeta.dims[2]], [b0, h, w])
      ) {
        return null;
      }

      // Trace slot12 bounded int64 Cast(to=7) -> Mul(scalar 0n) -> Cast(to=1) -> Reshape([b0,1,h,w])
      const reshapeZero = producerByOutput.get(slot1);
      if (
        !reshapeZero ||
        reshapeZero.opType !== 'Reshape' ||
        reshapeZero.inputs.length !== 2 ||
        reshapeZero.outputs.length !== 1
      ) {
        return null;
      }
      if (branchSpec && reshapeZero.name !== branchSpec.reshapeZeroName) return null;
      const reshapeShapeMeta = preInference.tensorMeta[reshapeZero.inputs[1]];
      if (!reshapeShapeMeta?.intVals || !sameStaticDims(reshapeShapeMeta.intVals, [b0, 1, h, w])) {
        return null;
      }

      const castToFloat = producerByOutput.get(reshapeZero.inputs[0]);
      if (
        !castToFloat ||
        castToFloat.opType !== 'Cast' ||
        castToFloat.inputs.length !== 1 ||
        castToFloat.outputs.length !== 1 ||
        (castToFloat.attrs.get('to')?.i ?? 0) !== 1
      ) {
        return null;
      }
      if (branchSpec && castToFloat.name !== branchSpec.zerosLikeName) return null;
      const castToFloatInMeta = preInference.tensorMeta[castToFloat.inputs[0]];
      const castToFloatOutMeta = preInference.tensorMeta[castToFloat.outputs[0]];
      if (
        !castToFloatInMeta ||
        !castToFloatOutMeta ||
        castToFloatInMeta.elemType !== 7 ||
        castToFloatOutMeta.elemType !== 1 ||
        !sameStaticDims(castToFloatOutMeta.dims, [b0, h, w, 1])
      ) {
        return null;
      }

      const mulZero = producerByOutput.get(castToFloat.inputs[0]);
      if (
        !mulZero ||
        mulZero.opType !== 'Mul' ||
        mulZero.inputs.length !== 2 ||
        mulZero.outputs.length !== 1
      ) {
        return null;
      }
      if (branchSpec && mulZero.name !== branchSpec.mulName) return null;

      let castToInt64OutName = '';
      let zeroInitName = '';
      for (const mi of mulZero.inputs) {
        if (int64InitMap.has(mi) && allInitializerNames.has(mi)) {
          zeroInitName = mi;
        } else {
          castToInt64OutName = mi;
        }
      }
      if (!castToInt64OutName || !zeroInitName) return null;
      if (branchSpec && zeroInitName !== branchSpec.zeroInitName) return null;
      const zeroInit = int64InitMap.get(zeroInitName);
      if (
        !zeroInit ||
        (zeroInit.dims.length !== 0 && !sameStaticDims(zeroInit.dims, [1])) ||
        zeroInit.values.length !== 1 ||
        zeroInit.values[0] !== 0n
      ) {
        return null;
      }

      const castToInt64 = producerByOutput.get(castToInt64OutName);
      if (
        !castToInt64 ||
        castToInt64.opType !== 'Cast' ||
        castToInt64.inputs.length !== 1 ||
        castToInt64.outputs.length !== 1 ||
        (castToInt64.attrs.get('to')?.i ?? 0) !== 7
      ) {
        return null;
      }
      if (branchSpec && castToInt64.name !== branchSpec.castI64Name) return null;
      if (castToInt64.inputs[0] !== castF32.outputs[0]) return null;

      // Verify zero other consumers and zero graph output exports for all 4 intermediate tensors
      const chainTensors = [
        castToInt64.outputs[0],
        mulZero.outputs[0],
        castToFloat.outputs[0],
        reshapeZero.outputs[0],
      ];
      for (const ct of chainTensors) {
        if (graphOutputSet.has(ct)) return null;
      }
      const c1 = consumersByTensor.get(castToInt64.outputs[0]);
      if (!c1 || c1.size !== 1 || !c1.has(mulZero.nodeIdx)) return null;
      if (mulZero.inputs[0] === mulZero.inputs[1]) return null;

      const c2 = consumersByTensor.get(mulZero.outputs[0]);
      if (!c2 || c2.size !== 1 || !c2.has(castToFloat.nodeIdx)) return null;

      const c3 = consumersByTensor.get(castToFloat.outputs[0]);
      if (!c3 || c3.size !== 1 || !c3.has(reshapeZero.nodeIdx)) return null;

      const c4 = consumersByTensor.get(reshapeZero.outputs[0]);
      if (!c4 || c4.size !== 1 || !c4.has(concatNode.nodeIdx)) return null;

      // Also verify downstream Transpose(perm=[0,2,3,1]) -> ExpandDims(Reshape [b0,h,w,1,16]) in HITNet
      let expectedTransposeOut: string | null = null;
      let expectedExpandDimsOut: string | null = null;
      if (branchSpec) {
        const concatConsumers = consumersByTensor.get(concatNode.outputs[0]);
        if (!concatConsumers || concatConsumers.size !== 1) return null;
        const tIdx = Array.from(concatConsumers)[0];
        const tNode = parsedNodes[tIdx];
        if (
          tNode.opType !== 'Transpose' ||
          tNode.outputs[0] !== branchSpec.transposeOutName ||
          !sameStaticDims(tNode.attrs.get('perm')?.ints ?? [], [0, 2, 3, 1])
        ) {
          return null;
        }
        expectedTransposeOut = tNode.outputs[0];
        const eConsumers = consumersByTensor.get(tNode.outputs[0]);
        if (!eConsumers || eConsumers.size !== 1) return null;
        const eIdx = Array.from(eConsumers)[0];
        const eNode = parsedNodes[eIdx];
        if (eNode.opType !== 'Reshape' || eNode.outputs[0] !== branchSpec.expandDimsOutName) {
          return null;
        }
        const eMeta = preInference.tensorMeta[eNode.outputs[0]];
        if (!eMeta || eMeta.elemType !== 1 || !sameStaticDims(eMeta.dims, [b0, h, w, 1, 16])) {
          return null;
        }
        expectedExpandDimsOut = eNode.outputs[0];
      }

      const deadNodes = [castToInt64, mulZero, castToFloat, reshapeZero];
      return {
        branchName: branchSpec?.branchName ?? concatNode.name,
        concatNode,
        slot0DisparityTensor: slot0,
        slot12ZeroTensor: slot1,
        slot3FeatureTensor: slot3,
        deadNodeIndices: deadNodes.map((n) => n.nodeIdx),
        deadNodeNames: deadNodes.map((n) => n.name),
        deadTensorNames: chainTensors,
        zeroInitName,
        batch0: b0,
        height: h,
        width: w,
        totalOutChannels: mOut.dims[1],
        expectedExpandDimsOut,
        expectedTransposeOut,
      };
    };

    const verifiedBranches: VerifiedBranchSpec[] = [];
    if (requireHitnetContract) {
      for (const bs of expectedHitnetBranches) {
        const cn = parsedNodes.find(
          (n) => n.opType === 'Concat' && n.name === bs.concatName
        );
        if (!cn) {
          return {
            ...emptyResult,
            verifiedH1Provenance,
            verifiedRawIdentity,
            skippedInvalidOrUnprovenCount: 1,
          };
        }
        const verified = verifySingleBranch(cn, bs);
        if (!verified) {
          return {
            ...emptyResult,
            verifiedH1Provenance,
            verifiedRawIdentity,
            skippedInvalidOrUnprovenCount: 1,
          };
        }
        verifiedBranches.push(verified);
      }
      if (verifiedBranches.length !== 2) {
        return {
          ...emptyResult,
          verifiedH1Provenance,
          verifiedRawIdentity,
          skippedInvalidOrUnprovenCount: 1,
        };
      }
      const [b0Spec, b1Spec] = verifiedBranches;
      if (
        b0Spec.batch0 !== b1Spec.batch0 ||
        b0Spec.height !== b1Spec.height ||
        b0Spec.width !== b1Spec.width ||
        b0Spec.zeroInitName !== b1Spec.zeroInitName
      ) {
        return {
          ...emptyResult,
          verifiedH1Provenance,
          verifiedRawIdentity,
          skippedInvalidOrUnprovenCount: 1,
        };
      }
    } else {
      const candidateConcats = parsedNodes.filter(
        (n) =>
          n.opType === 'Concat' &&
          (n.attrs.get('axis')?.i ?? null) === 1 &&
          n.inputs.length === 4 &&
          Boolean(n.inputs[1]) &&
          n.inputs[1] === n.inputs[2]
      );
      for (const cn of candidateConcats) {
        const verified = verifySingleBranch(cn);
        if (verified) {
          verifiedBranches.push(verified);
        } else {
          return {
            ...emptyResult,
            verifiedH1Provenance,
            verifiedRawIdentity,
            skippedInvalidOrUnprovenCount: 1,
          };
        }
      }
      if (verifiedBranches.length === 0) {
        return { ...emptyResult, verifiedH1Provenance, verifiedRawIdentity };
      }
      const first = verifiedBranches[0];
      if (
        verifiedBranches.some(
          (b) =>
            b.batch0 !== first.batch0 || b.height !== first.height || b.width !== first.width
        )
      ) {
        return {
          ...emptyResult,
          verifiedH1Provenance,
          verifiedRawIdentity,
          skippedInvalidOrUnprovenCount: 1,
        };
      }
    }

    // Collect all dead nodes & verify dead zero initializer has NO other consumers before dropping it
    const deadNodeIndexSet = new Set<number>();
    const removedDeadNodeNames: string[] = [];
    const deadMulNodeIdxSet = new Set<number>();
    const deadZeroInitCandidates = new Set<string>();
    const deadTensorNamesAll = new Set<string>();

    for (const vb of verifiedBranches) {
      for (let i = 0; i < vb.deadNodeIndices.length; i++) {
        deadNodeIndexSet.add(vb.deadNodeIndices[i]);
        removedDeadNodeNames.push(vb.deadNodeNames[i]);
      }
      // Index 1 in deadNodeIndices is always mulZero
      deadMulNodeIdxSet.add(vb.deadNodeIndices[1]);
      deadZeroInitCandidates.add(vb.zeroInitName);
      for (const dt of vb.deadTensorNames) deadTensorNamesAll.add(dt);
    }

    const deadInitializerNameSet = new Set<string>();
    const removedDeadInitializerNames: string[] = [];
    for (const zi of deadZeroInitCandidates) {
      if (graphOutputSet.has(zi)) continue;
      const ziConsumers = consumersByTensor.get(zi);
      if (ziConsumers && ziConsumers.size > 0) {
        let allConsumersRemoved = true;
        for (const cIdx of ziConsumers) {
          if (!deadMulNodeIdxSet.has(cIdx)) {
            allConsumersRemoved = false;
            break;
          }
        }
        if (allConsumersRemoved) {
          deadInitializerNameSet.add(zi);
          removedDeadInitializerNames.push(zi);
        }
      }
    }

    const concatBytes = (parts: readonly Uint8Array[]): Uint8Array => {
      let total = 0;
      for (const p of parts) total += p.byteLength;
      const out = new Uint8Array(total);
      let off = 0;
      for (const p of parts) {
        out.set(p, off);
        off += p.byteLength;
      }
      return out;
    };
    const encodeLenField = (fn: number, payload: Uint8Array): Uint8Array => {
      const tag = encodeProtoVarint((fn << 3) | 2);
      const len = encodeProtoVarint(payload.byteLength);
      const out = new Uint8Array(tag.byteLength + len.byteLength + payload.byteLength);
      out.set(tag, 0);
      out.set(len, tag.byteLength);
      out.set(payload, tag.byteLength + len.byteLength);
      return out;
    };
    const encodeStrField = (fn: number, str: string): Uint8Array =>
      encodeLenField(fn, encoder.encode(str));
    const encodeVarintField = (fn: number, val: number): Uint8Array =>
      concatBytes([encodeProtoVarint((fn << 3) | 0), encodeProtoVarint(val)]);

    let zero2ChInitName = '__webgpu_m1_init_to_prop_zero_slant_2ch';
    let suffix = 0;
    while (usedNames.has(zero2ChInitName)) {
      zero2ChInitName = `__webgpu_m1_init_to_prop_zero_slant_2ch_${suffix++}`;
    }
    usedNames.add(zero2ChInitName);

    const refBranch = verifiedBranches[0];
    const addedInitializerDims = [refBranch.batch0, 2, refBranch.height, refBranch.width];
    const zeroFloatRaw = new Uint8Array(
      addedInitializerDims[0] *
        addedInitializerDims[1] *
        addedInitializerDims[2] *
        addedInitializerDims[3] *
        4
    ); // All zero bytes == IEEE-754 +0.0f
    const addedZero2ChInitializer = encodeLenField(
      5,
      concatBytes([
        ...addedInitializerDims.map((d) => encodeVarintField(1, d)),
        encodeVarintField(2, 1), // TensorProto.DataType.FLOAT
        encodeStrField(8, zero2ChInitName),
        encodeLenField(9, zeroFloatRaw),
      ])
    );

    const concatReplacementByIdx = new Map<number, Uint8Array>();
    const normalizedNodeNames: string[] = [];
    const expectedAfterMetaByOutName = new Map<string, { elemType: number; outDims: number[] }>();

    for (const vb of verifiedBranches) {
      const cn = vb.concatNode;
      const rewrittenParts: Uint8Array[] = [];
      let emittedNewInputs = false;
      for (const nf of cn.nodeFields) {
        if (nf.fn === 1 && nf.wt === 2) {
          if (!emittedNewInputs) {
            rewrittenParts.push(
              encodeStrField(1, vb.slot0DisparityTensor),
              encodeStrField(1, zero2ChInitName),
              encodeStrField(1, vb.slot3FeatureTensor)
            );
            emittedNewInputs = true;
          }
        } else {
          rewrittenParts.push(bytes.subarray(nf.fStart, nf.fEnd));
        }
      }
      concatReplacementByIdx.set(cn.nodeIdx, encodeLenField(1, concatBytes(rewrittenParts)));
      normalizedNodeNames.push(cn.name);
      expectedAfterMetaByOutName.set(cn.outputs[0], {
        elemType: 1,
        outDims: [vb.batch0, vb.totalOutChannels, vb.height, vb.width],
      });
      if (vb.expectedTransposeOut) {
        expectedAfterMetaByOutName.set(vb.expectedTransposeOut, {
          elemType: 1,
          outDims: [vb.batch0, vb.height, vb.width, vb.totalOutChannels],
        });
      }
      if (vb.expectedExpandDimsOut) {
        expectedAfterMetaByOutName.set(vb.expectedExpandDimsOut, {
          elemType: 1,
          outDims: [vb.batch0, vb.height, vb.width, 1, vb.totalOutChannels],
        });
      }
    }

    const finalGraphParts: Uint8Array[] = [];
    let currentNodeIdx = 0;
    for (const gf of graphFields) {
      if (gf.fn === 13) continue; // Drop stale value_info entries
      if (gf.fn === 5 && gf.wt === 2) {
        const initName = getInitializerName(bytes, gf.valStart, gf.valEnd, decoder);
        if (initName && deadInitializerNameSet.has(initName)) {
          continue;
        }
        finalGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
      } else if (gf.fn === 1 && gf.wt === 2) {
        const idx = currentNodeIdx++;
        if (deadNodeIndexSet.has(idx)) {
          continue;
        }
        const replacement = concatReplacementByIdx.get(idx);
        if (replacement) {
          finalGraphParts.push(replacement);
        } else {
          finalGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
        }
      } else {
        finalGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
      }
    }
    finalGraphParts.push(addedZero2ChInitializer);

    let graphPayloadLen = 0;
    for (const part of finalGraphParts) graphPayloadLen += part.byteLength;
    const graphTag = encodeProtoVarint((7 << 3) | 2);
    const graphLen = encodeProtoVarint(graphPayloadLen);
    let totalLen = 0;
    for (let i = 0; i < topFields.length; i++) {
      totalLen +=
        i === graphIndex
          ? graphTag.byteLength + graphLen.byteLength + graphPayloadLen
          : topFields[i].fEnd - topFields[i].fStart;
    }

    const out = new Uint8Array(totalLen);
    let offset = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        out.set(graphTag, offset);
        offset += graphTag.byteLength;
        out.set(graphLen, offset);
        offset += graphLen.byteLength;
        for (const part of finalGraphParts) {
          out.set(part, offset);
          offset += part.byteLength;
        }
      } else {
        const chunk = bytes.subarray(topFields[i].fStart, topFields[i].fEnd);
        out.set(chunk, offset);
        offset += chunk.byteLength;
      }
    }

    const beforeValidation = validateOnnxGraphNodeInputs(bytes);
    const afterValidation = validateOnnxGraphNodeInputs(out);
    const preexistingMissing = new Set(beforeValidation.missingInputs.map((m) => m.inputName));
    const newlyMissing = afterValidation.missingInputs.filter(
      (m) => !preexistingMissing.has(m.inputName)
    );
    if (
      newlyMissing.length > 0 ||
      (beforeValidation.hasGraphInputs && !afterValidation.valid)
    ) {
      return emptyResult;
    }

    const postInference = inferOnnxStaticTensorMeta(out);
    if (
      !postInference.valid ||
      postInference.duplicateOutputs.length > 0 ||
      postInference.duplicateInitializers.length > 0 ||
      postInference.duplicateNodeNames.length > 0
    ) {
      return emptyResult;
    }

    for (const dt of deadTensorNamesAll) {
      if (postInference.tensorMeta[dt]) {
        return emptyResult;
      }
    }
    for (const di of removedDeadInitializerNames) {
      if (postInference.tensorMeta[di]) {
        return emptyResult;
      }
    }
    const addedInitMeta = postInference.tensorMeta[zero2ChInitName];
    if (
      !addedInitMeta ||
      addedInitMeta.elemType !== 1 ||
      !sameStaticDims(addedInitMeta.dims, addedInitializerDims)
    ) {
      return emptyResult;
    }
    for (const [tensorName, exp] of expectedAfterMetaByOutName.entries()) {
      const m = postInference.tensorMeta[tensorName];
      if (!m || m.elemType !== exp.elemType || !sameStaticDims(m.dims, exp.outDims)) {
        return emptyResult;
      }
    }
    if (requireHitnetContract) {
      const refPost = postInference.tensorMeta['reference_output_disparity'];
      const secPost = postInference.tensorMeta['secondary_output_disparity'];
      if (
        !refPost ||
        !secPost ||
        refPost.elemType !== 1 ||
        secPost.elemType !== 1 ||
        !sameStaticDims(refPost.dims, [1, 240, 320, 1]) ||
        !sameStaticDims(secPost.dims, [1, 240, 320, 1])
      ) {
        return emptyResult;
      }
    }

    return {
      bytes: out,
      normalizedCount: normalizedNodeNames.length,
      normalizedNodeNames,
      removedDeadNodeNames,
      removedDeadInitializerNames,
      addedInitializerName: zero2ChInitName,
      addedInitializerDims,
      verifiedH1Provenance,
      verifiedRawIdentity,
      skippedInvalidOrUnprovenCount: 0,
    };
  } catch {
    return emptyResult;
  }
}

/**
 * Guarded N1-Slot0Transpose pass for `hitnet-sf` WebGPU built strictly on top of exact M1:
 * Replaces only the two singleton static NHWC `[1,60,80,1]` -> NCHW `[1,1,60,80]` `Reshape`
 * nodes (`Transpose__908` and `Transpose__1006`) with explicit equivalent `Transpose(perm=[0,3,1,2])`,
 * preserving exact node/tensor names, types, topological order, and all other consumers/initializers.
 * Atomically fails closed with no mutation on any identity, lineage, shape, or extra-use mismatch.
 */
export function normalizeHitnetSlot0ReshapeToTransposeForWebGpu(
  bytes: Uint8Array,
  options: NormalizeSlot0ReshapeToTransposeOptions = {}
): OnnxSlot0ReshapeToTransposeResult {
  const emptyResult: OnnxSlot0ReshapeToTransposeResult = {
    bytes,
    normalizedCount: 0,
    normalizedNodeNames: [],
    verifiedM1Provenance: false,
    verifiedRawIdentity: false,
    skippedInvalidOrUnprovenCount: 0,
  };
  const requireHitnetContract = options.requireHitnetContract ?? true;
  if (
    (options.backendPreference === 'wasm' && !options.allowDiagnosticTransformedWasm) ||
    bytes.byteLength < 16
  ) {
    return emptyResult;
  }
  if (requireHitnetContract && options.variantId !== 'hitnet-sf') {
    return emptyResult;
  }
  if (options.rawModelSha256 !== undefined && options.rawModelSha256 !== HITNET_RAW_MODEL_SHA256) {
    return { ...emptyResult, skippedInvalidOrUnprovenCount: 1 };
  }
  if (options.m1ModelSha256 !== undefined && options.m1ModelSha256 !== HITNET_M1_FULL_GRAPH_SHA256) {
    return { ...emptyResult, skippedInvalidOrUnprovenCount: 1 };
  }

  try {
    const preInference = inferOnnxStaticTensorMeta(bytes);
    if (
      !preInference.valid ||
      preInference.hasControlFlowNode ||
      preInference.duplicateOutputs.length > 0 ||
      preInference.duplicateInitializers.length > 0 ||
      preInference.duplicateNodeNames.length > 0
    ) {
      return emptyResult;
    }

    if (preInference.irVersion !== 7) {
      return emptyResult;
    }
    if (requireHitnetContract) {
      if (preInference.defaultOpset !== 12) {
        return emptyResult;
      }
    } else if (
      preInference.defaultOpset !== null &&
      (preInference.defaultOpset < 11 || preInference.defaultOpset > 17)
    ) {
      return emptyResult;
    }

    const topFields = parseProtoFields(bytes, 0, bytes.byteLength);
    if (!topFields) return emptyResult;
    const graphIndex = topFields.findIndex((f) => f.fn === 7 && f.wt === 2);
    if (graphIndex < 0) return emptyResult;
    const graphField = topFields[graphIndex];
    const graphFields = parseProtoFields(bytes, graphField.valStart, graphField.valEnd);
    if (!graphFields) return emptyResult;

    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    const allInitializerNames = new Set<string>();
    const int64InitMap = new Map<string, { dims: number[]; values: bigint[] }>();
    const parsedGraphInputs: ParsedStaticValueInfo[] = [];
    const parsedGraphOutputs: ParsedStaticValueInfo[] = [];
    const graphOutputSet = new Set<string>();

    interface N1ParsedNode {
      nodeIdx: number;
      gf: ProtobufFieldSlice;
      opType: string;
      name: string;
      inputs: string[];
      outputs: string[];
      attrs: Map<string, { i?: number; ints?: number[] }>;
    }
    const parsedNodes: N1ParsedNode[] = [];
    const producerByOutput = new Map<string, N1ParsedNode>();
    const consumersByTensor = new Map<string, Set<number>>();

    for (const gf of graphFields) {
      if (gf.fn === 5 && gf.wt === 2) {
        const nm = getInitializerName(bytes, gf.valStart, gf.valEnd, decoder);
        if (nm) {
          allInitializerNames.add(nm);
        }
        const i64 = readExactInt64InitializerBigInts(bytes, gf, decoder);
        if (i64) {
          int64InitMap.set(i64.name, { dims: i64.dims, values: i64.values });
        }
        continue;
      }
      if (gf.fn === 11 && gf.wt === 2) {
        const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
        if (vi) {
          parsedGraphInputs.push(vi);
        }
        continue;
      }
      if (gf.fn === 12 && gf.wt === 2) {
        const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
        if (vi) {
          parsedGraphOutputs.push(vi);
          graphOutputSet.add(vi.name);
        } else {
          const nm = getValueInfoName(bytes, gf.valStart, gf.valEnd, decoder);
          if (nm) graphOutputSet.add(nm);
        }
        continue;
      }
      if (gf.fn !== 1 || gf.wt !== 2) continue;

      const nodeFields = parseProtoFields(bytes, gf.valStart, gf.valEnd);
      if (!nodeFields) return emptyResult;
      const inputs: string[] = [];
      const outputs: string[] = [];
      let name = '';
      let opType = '';
      const attrs = new Map<string, { i?: number; ints?: number[] }>();

      for (const nf of nodeFields) {
        if (nf.fn === 1 && nf.wt === 2) {
          const inp =
            nf.valEnd > nf.valStart ? decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)) : '';
          inputs.push(inp);
        } else if (nf.fn === 2 && nf.wt === 2) {
          const outName =
            nf.valEnd > nf.valStart ? decoder.decode(bytes.subarray(nf.valStart, nf.valEnd)) : '';
          outputs.push(outName);
        } else if (nf.fn === 3 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          name = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
        } else if (nf.fn === 4 && nf.wt === 2 && nf.valEnd > nf.valStart) {
          opType = decoder.decode(bytes.subarray(nf.valStart, nf.valEnd));
        } else if (nf.fn === 5 && nf.wt === 2) {
          const af = parseProtoFields(bytes, nf.valStart, nf.valEnd);
          if (!af) continue;
          let an = '';
          let ai: number | undefined;
          const aints: number[] = [];
          for (const a of af) {
            if (a.fn === 1 && a.wt === 2 && a.valEnd > a.valStart) {
              an = decoder.decode(bytes.subarray(a.valStart, a.valEnd));
            } else if (a.fn === 3 && a.wt === 0) {
              const v = readProtoVarintBigInt(bytes, a.valStart, a.valEnd);
              if (v) ai = Number(BigInt.asIntN(64, v[0]));
            } else if (a.fn === 8 && a.wt === 0) {
              const v = readProtoVarintBigInt(bytes, a.valStart, a.valEnd);
              if (v) aints.push(Number(BigInt.asIntN(64, v[0])));
            }
          }
          if (an) attrs.set(an, { i: ai, ints: aints });
        }
      }

      const nodeIdx = parsedNodes.length;
      const entry: N1ParsedNode = { nodeIdx, gf, opType, name, inputs, outputs, attrs };
      parsedNodes.push(entry);
      for (const o of outputs) {
        if (o) producerByOutput.set(o, entry);
      }
      for (const inp of inputs) {
        if (!inp) continue;
        let set = consumersByTensor.get(inp);
        if (!set) {
          set = new Set<number>();
          consumersByTensor.set(inp, set);
        }
        set.add(nodeIdx);
      }
    }

    const nonInitInputs = parsedGraphInputs.filter((g) => !allInitializerNames.has(g.name));
    if (
      nonInitInputs.length === 0 ||
      nonInitInputs.some((g) => g.dims === null || g.elemType === null)
    ) {
      return emptyResult;
    }

    let verifiedRawIdentity = !requireHitnetContract;
    let verifiedM1Provenance = !requireHitnetContract;

    if (requireHitnetContract) {
      if (
        nonInitInputs.length !== 1 ||
        nonInitInputs[0].name !== 'input' ||
        nonInitInputs[0].elemType !== 1 ||
        !nonInitInputs[0].dims ||
        !sameStaticDims(nonInitInputs[0].dims, [1, 6, 240, 320])
      ) {
        return emptyResult;
      }
      const outNames = parsedGraphOutputs.map((o) => o.name);
      if (
        outNames.length !== 2 ||
        !outNames.includes('reference_output_disparity') ||
        !outNames.includes('secondary_output_disparity')
      ) {
        return emptyResult;
      }
      const refMeta = preInference.tensorMeta['reference_output_disparity'];
      const secMeta = preInference.tensorMeta['secondary_output_disparity'];
      if (
        !refMeta ||
        !secMeta ||
        refMeta.elemType !== 1 ||
        secMeta.elemType !== 1 ||
        !sameStaticDims(refMeta.dims, [1, 240, 320, 1]) ||
        !sameStaticDims(secMeta.dims, [1, 240, 320, 1])
      ) {
        return emptyResult;
      }
      verifiedRawIdentity = true;

      // Reject 5D Conv, L1-ReduceMin, L1-Reshape, G1 ArgMin keepdims!=0, or non-M1 H1/raw models
      const has5dConv = parsedNodes.some((n) => {
        if (n.opType !== 'Conv') return false;
        const m = n.inputs[0] ? preInference.tensorMeta[n.inputs[0]] : undefined;
        return Boolean(m && m.dims.length >= 5);
      });
      const hasReduceMin = parsedNodes.some((n) => n.opType === 'ReduceMin');
      if (has5dConv || hasReduceMin) {
        return { ...emptyResult, verifiedRawIdentity, skippedInvalidOrUnprovenCount: 1 };
      }

      const gatherNdNodes = parsedNodes.filter((n) => n.opType === 'GatherND');
      if (gatherNdNodes.length !== 10) {
        return { ...emptyResult, verifiedRawIdentity, skippedInvalidOrUnprovenCount: 1 };
      }
      let f1Count = 0;
      let h1PropCount = 0;
      let untouchedLevelInitCount = 0;
      for (const gn of gatherNdNodes) {
        const bd = gn.attrs.get('batch_dims')?.i ?? 0;
        const dm = gn.inputs[0] ? preInference.tensorMeta[gn.inputs[0]] : undefined;
        const im = gn.inputs[1] ? preInference.tensorMeta[gn.inputs[1]] : undefined;
        const om = gn.outputs[0] ? preInference.tensorMeta[gn.outputs[0]] : undefined;
        if (!dm || !im || !om) {
          return { ...emptyResult, verifiedRawIdentity, skippedInvalidOrUnprovenCount: 1 };
        }
        if (
          (gn.name === 'level0/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis' ||
            gn.name === 'level0_1/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis') &&
          bd === 2 &&
          dm.elemType === 1 &&
          im.elemType === 7 &&
          om.elemType === 1 &&
          sameStaticDims(dm.dims, [1, 4800, 1, 16]) &&
          sameStaticDims(im.dims, [1, 4800, 1, 1]) &&
          sameStaticDims(om.dims, [1, 4800, 1, 16]) &&
          gn.inputs[0].includes('__h1_data_4d_') &&
          gn.inputs[1].includes('__h1_indices_4d_') &&
          gn.outputs[0].includes('__h1_out_4d_')
        ) {
          h1PropCount++;
        } else if (
          gn.name.includes('/warp_1d/') &&
          bd === 2 &&
          sameStaticDims(dm.dims, [1, 240, 320, 32]) &&
          sameStaticDims(im.dims, [1, 240, 960, 1]) &&
          sameStaticDims(om.dims, [1, 240, 960, 32]) &&
          gn.inputs[1].includes('__flat_indices_') &&
          gn.outputs[0].includes('__flat_out_')
        ) {
          f1Count++;
        } else if (
          (gn.name === 'level0/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis' ||
            gn.name === 'level0_1/level_init/GatherV2_1;level0/level_init/GatherV2_1/axis') &&
          bd === 3 &&
          dm.elemType === 1 &&
          im.elemType === 7 &&
          om.elemType === 1 &&
          sameStaticDims(dm.dims, [1, 60, 80, 320]) &&
          sameStaticDims(im.dims, [1, 60, 80, 1, 1]) &&
          sameStaticDims(om.dims, [1, 60, 80, 1])
        ) {
          untouchedLevelInitCount++;
        }
      }
      if (h1PropCount !== 2 || f1Count !== 4 || untouchedLevelInitCount !== 2) {
        return { ...emptyResult, verifiedRawIdentity, skippedInvalidOrUnprovenCount: 1 };
      }

      const argMinNodes = parsedNodes.filter((n) => n.opType === 'ArgMin');
      if (
        argMinNodes.length !== 4 ||
        argMinNodes.some((an) => (an.attrs.get('keepdims')?.i ?? 1) !== 0)
      ) {
        return { ...emptyResult, verifiedRawIdentity, skippedInvalidOrUnprovenCount: 1 };
      }

      // Exact M1 provenance: must have M1 2ch initializer and none of H1's removed dead nodes/initializer
      const m1ZeroInitMeta = preInference.tensorMeta['__webgpu_m1_init_to_prop_zero_slant_2ch'];
      if (
        !allInitializerNames.has('__webgpu_m1_init_to_prop_zero_slant_2ch') ||
        !m1ZeroInitMeta ||
        m1ZeroInitMeta.elemType !== 1 ||
        !sameStaticDims(m1ZeroInitMeta.dims, [1, 2, 60, 80]) ||
        allInitializerNames.has('zero__310') ||
        parsedNodes.some(
          (n) =>
            n.name === 'Transpose__910' ||
            n.name === 'Transpose__1008' ||
            n.name === 'Cast__309' ||
            n.name === 'Cast__283'
        )
      ) {
        return { ...emptyResult, verifiedRawIdentity, skippedInvalidOrUnprovenCount: 1 };
      }
      verifiedM1Provenance = true;
    }

    interface VerifiedN1Branch {
      reshapeNode: N1ParsedNode;
      castInputName: string;
      reshapeOutName: string;
      concatOutName: string;
      batch0: number;
      height: number;
      width: number;
      expectedTransposeOut: string | null;
      expectedExpandDimsOut: string | null;
    }

    const expectedHitnetBranches = [
      {
        concatName: 'level0/level_init/init_to_prop/concat_1',
        slot0Tensor: 'Transpose__908:0',
        reshapeName: 'Transpose__908',
        castF32Name: 'level0/level_init/Cast',
        expand2Name: 'level0/level_init/ExpandDims_2',
        argMinName: 'level0/level_init/ArgMin',
        zero2ChInitName: '__webgpu_m1_init_to_prop_zero_slant_2ch',
        slot2FeatureTensor: 'level0/level_init/init_to_prop/LeakyRelu',
        shapeInitName: 'new_shape__1191',
        transposeOutName: 'Transpose__913:0',
        expandDimsOutName: 'level0/level_init/init_to_prop/ExpandDims',
      },
      {
        concatName: 'level0_1/level_init/init_to_prop/concat_1',
        slot0Tensor: 'Transpose__1006:0',
        reshapeName: 'Transpose__1006',
        castF32Name: 'level0_1/level_init/Cast',
        expand2Name: 'level0_1/level_init/ExpandDims_2',
        argMinName: 'level0_1/level_init/ArgMin',
        zero2ChInitName: '__webgpu_m1_init_to_prop_zero_slant_2ch',
        slot2FeatureTensor: 'level0_1/level_init/init_to_prop/LeakyRelu',
        shapeInitName: 'new_shape__1191',
        transposeOutName: 'Transpose__1011:0',
        expandDimsOutName: 'level0_1/level_init/init_to_prop/ExpandDims',
      },
    ];

    const verifyN1Branch = (
      concatNode: N1ParsedNode,
      branchSpec?: (typeof expectedHitnetBranches)[number]
    ): VerifiedN1Branch | null => {
      if (
        concatNode.opType !== 'Concat' ||
        concatNode.inputs.length !== 3 ||
        concatNode.outputs.length !== 1 ||
        !concatNode.outputs[0] ||
        (concatNode.attrs.get('axis')?.i ?? null) !== 1
      ) {
        return null;
      }
      const [slot0, slot1, slot2] = concatNode.inputs;
      if (!slot0 || !slot1 || !slot2 || slot0 === slot1 || slot0 === slot2 || slot1 === slot2) {
        return null;
      }
      if (!allInitializerNames.has(slot1)) return null;
      if (branchSpec) {
        if (
          concatNode.name !== branchSpec.concatName ||
          concatNode.outputs[0] !== branchSpec.concatName ||
          slot0 !== branchSpec.slot0Tensor ||
          slot1 !== branchSpec.zero2ChInitName ||
          slot2 !== branchSpec.slot2FeatureTensor
        ) {
          return null;
        }
      }

      const m0 = preInference.tensorMeta[slot0];
      const m1 = preInference.tensorMeta[slot1];
      const m2 = preInference.tensorMeta[slot2];
      const mOut = preInference.tensorMeta[concatNode.outputs[0]];
      if (
        !m0 ||
        !m1 ||
        !m2 ||
        !mOut ||
        m0.elemType !== 1 ||
        m1.elemType !== 1 ||
        m2.elemType !== 1 ||
        mOut.elemType !== 1 ||
        m0.dims.length !== 4 ||
        m1.dims.length !== 4 ||
        m2.dims.length !== 4 ||
        mOut.dims.length !== 4
      ) {
        return null;
      }
      const [b0, c0, h, w] = m0.dims;
      if (c0 !== 1 || b0 <= 0 || h <= 0 || w <= 0) return null;
      if (requireHitnetContract && (b0 !== 1 || h !== 60 || w !== 80)) return null;
      if (!sameStaticDims(m1.dims, [b0, 2, h, w])) return null;
      if (m2.dims[0] !== b0 || m2.dims[2] !== h || m2.dims[3] !== w || m2.dims[1] <= 0) return null;
      if (requireHitnetContract && m2.dims[1] !== 13) return null;
      if (!sameStaticDims(mOut.dims, [b0, 1 + 2 + m2.dims[1], h, w])) return null;

      // Extra-use and graph-output guard on slot0 (Transpose__908:0 / Transpose__1006:0)
      if (graphOutputSet.has(slot0)) return null;
      const slot0Consumers = consumersByTensor.get(slot0);
      if (!slot0Consumers || slot0Consumers.size !== 1 || !slot0Consumers.has(concatNode.nodeIdx)) {
        return null;
      }

      const reshapeNode = producerByOutput.get(slot0);
      if (
        !reshapeNode ||
        reshapeNode.opType !== 'Reshape' ||
        reshapeNode.inputs.length !== 2 ||
        reshapeNode.outputs.length !== 1 ||
        reshapeNode.outputs[0] !== slot0 ||
        (reshapeNode.attrs.get('allowzero')?.i ?? 0) !== 0
      ) {
        return null;
      }
      if (branchSpec && reshapeNode.name !== branchSpec.reshapeName) return null;

      const [castInputName, shapeInputName] = reshapeNode.inputs;
      if (!castInputName || !shapeInputName) return null;
      if (branchSpec && shapeInputName !== branchSpec.shapeInitName) return null;

      // Extra-use and graph-output guard on Reshape input (level_init/Cast)
      if (graphOutputSet.has(castInputName)) return null;
      const castConsumers = consumersByTensor.get(castInputName);
      if (!castConsumers || castConsumers.size !== 1 || !castConsumers.has(reshapeNode.nodeIdx)) {
        return null;
      }

      // Shape tensor must be an exact static int64 [4] initializer equal to [b0, 1, h, w] (no -1/0)
      const shapeInit = int64InitMap.get(shapeInputName);
      if (
        !shapeInit ||
        !allInitializerNames.has(shapeInputName) ||
        !sameStaticDims(shapeInit.dims, [4]) ||
        shapeInit.values.length !== 4 ||
        shapeInit.values[0] !== BigInt(b0) ||
        shapeInit.values[1] !== 1n ||
        shapeInit.values[2] !== BigInt(h) ||
        shapeInit.values[3] !== BigInt(w)
      ) {
        return null;
      }

      // Input shape/dtype guard: must be float32 [b0, h, w, 1]
      const castInMeta = preInference.tensorMeta[castInputName];
      if (
        !castInMeta ||
        castInMeta.elemType !== 1 ||
        !sameStaticDims(castInMeta.dims, [b0, h, w, 1])
      ) {
        return null;
      }

      // Upstream lineage guard: ArgMin(axis=3/-1, keepdims=0, select_last_index=0) -> ExpandDims_2 -> Cast(to=1)
      const castF32 = producerByOutput.get(castInputName);
      if (
        !castF32 ||
        castF32.opType !== 'Cast' ||
        castF32.inputs.length !== 1 ||
        castF32.outputs.length !== 1 ||
        (castF32.attrs.get('to')?.i ?? 0) !== 1
      ) {
        return null;
      }
      if (branchSpec && castF32.name !== branchSpec.castF32Name) return null;

      const expand2 = producerByOutput.get(castF32.inputs[0]);
      if (!expand2 || expand2.opType !== 'Reshape' || expand2.inputs.length !== 2) return null;
      if (branchSpec && expand2.name !== branchSpec.expand2Name) return null;
      const expand2Meta = preInference.tensorMeta[expand2.outputs[0]];
      if (!expand2Meta || expand2Meta.elemType !== 7 || !sameStaticDims(expand2Meta.dims, [b0, h, w, 1])) {
        return null;
      }

      const argMin = producerByOutput.get(expand2.inputs[0]);
      if (!argMin || argMin.opType !== 'ArgMin' || argMin.inputs.length !== 1) return null;
      if (branchSpec && argMin.name !== branchSpec.argMinName) return null;
      const argMinInMeta = preInference.tensorMeta[argMin.inputs[0]];
      const argMinAxis = argMin.attrs.get('axis')?.i ?? 0;
      const argMinKeep = argMin.attrs.get('keepdims')?.i ?? 1;
      const argMinSelLast = argMin.attrs.get('select_last_index')?.i ?? 0;
      if (
        !argMinInMeta ||
        argMinInMeta.elemType !== 1 ||
        argMinInMeta.dims.length !== 4 ||
        (argMinAxis !== 3 && argMinAxis !== -1) ||
        argMinKeep !== 0 ||
        argMinSelLast !== 0 ||
        argMinInMeta.dims[3] <= 0 ||
        !sameStaticDims([argMinInMeta.dims[0], argMinInMeta.dims[1], argMinInMeta.dims[2]], [b0, h, w])
      ) {
        return null;
      }

      let expectedTransposeOut: string | null = null;
      let expectedExpandDimsOut: string | null = null;
      if (branchSpec) {
        const concatConsumers = consumersByTensor.get(concatNode.outputs[0]);
        if (!concatConsumers || concatConsumers.size !== 1) return null;
        const tNode = parsedNodes[Array.from(concatConsumers)[0]];
        if (
          tNode.opType !== 'Transpose' ||
          tNode.outputs[0] !== branchSpec.transposeOutName ||
          !sameStaticDims(tNode.attrs.get('perm')?.ints ?? [], [0, 2, 3, 1])
        ) {
          return null;
        }
        expectedTransposeOut = tNode.outputs[0];
        const eConsumers = consumersByTensor.get(tNode.outputs[0]);
        if (!eConsumers || eConsumers.size !== 1) return null;
        const eNode = parsedNodes[Array.from(eConsumers)[0]];
        if (eNode.opType !== 'Reshape' || eNode.outputs[0] !== branchSpec.expandDimsOutName) {
          return null;
        }
        const eMeta = preInference.tensorMeta[eNode.outputs[0]];
        if (!eMeta || eMeta.elemType !== 1 || !sameStaticDims(eMeta.dims, [b0, h, w, 1, 16])) {
          return null;
        }
        expectedExpandDimsOut = eNode.outputs[0];
      }

      return {
        reshapeNode,
        castInputName,
        reshapeOutName: slot0,
        concatOutName: concatNode.outputs[0],
        batch0: b0,
        height: h,
        width: w,
        expectedTransposeOut,
        expectedExpandDimsOut,
      };
    };

    const verifiedBranches: VerifiedN1Branch[] = [];
    if (requireHitnetContract) {
      for (const bs of expectedHitnetBranches) {
        const cn = parsedNodes.find(
          (n) => n.opType === 'Concat' && n.name === bs.concatName
        );
        if (!cn) {
          return {
            ...emptyResult,
            verifiedM1Provenance,
            verifiedRawIdentity,
            skippedInvalidOrUnprovenCount: 1,
          };
        }
        const verified = verifyN1Branch(cn, bs);
        if (!verified) {
          return {
            ...emptyResult,
            verifiedM1Provenance,
            verifiedRawIdentity,
            skippedInvalidOrUnprovenCount: 1,
          };
        }
        verifiedBranches.push(verified);
      }
      if (verifiedBranches.length !== 2) {
        return {
          ...emptyResult,
          verifiedM1Provenance,
          verifiedRawIdentity,
          skippedInvalidOrUnprovenCount: 1,
        };
      }
    } else {
      const candidateConcats = parsedNodes.filter(
        (n) =>
          n.opType === 'Concat' &&
          (n.attrs.get('axis')?.i ?? null) === 1 &&
          n.inputs.length === 3
      );
      for (const cn of candidateConcats) {
        const verified = verifyN1Branch(cn);
        if (verified) {
          verifiedBranches.push(verified);
        } else {
          return {
            ...emptyResult,
            verifiedM1Provenance,
            verifiedRawIdentity,
            skippedInvalidOrUnprovenCount: 1,
          };
        }
      }
      if (verifiedBranches.length === 0) {
        return { ...emptyResult, verifiedM1Provenance, verifiedRawIdentity };
      }
    }

    const concatBytes = (parts: readonly Uint8Array[]): Uint8Array => {
      let total = 0;
      for (const p of parts) total += p.byteLength;
      const out = new Uint8Array(total);
      let off = 0;
      for (const p of parts) {
        out.set(p, off);
        off += p.byteLength;
      }
      return out;
    };
    const encodeLenField = (fn: number, payload: Uint8Array): Uint8Array => {
      const tag = encodeProtoVarint((fn << 3) | 2);
      const len = encodeProtoVarint(payload.byteLength);
      const out = new Uint8Array(tag.byteLength + len.byteLength + payload.byteLength);
      out.set(tag, 0);
      out.set(len, tag.byteLength);
      out.set(payload, tag.byteLength + len.byteLength);
      return out;
    };
    const encodeStrField = (fn: number, str: string): Uint8Array =>
      encodeLenField(fn, encoder.encode(str));
    const encodeVarintField = (fn: number, val: number): Uint8Array =>
      concatBytes([encodeProtoVarint((fn << 3) | 0), encodeProtoVarint(val)]);
    const encodeSignedVarint64 = (val: number): Uint8Array => {
      let v = BigInt.asUintN(64, BigInt(val));
      const out: number[] = [];
      while (v >= 0x80n) {
        out.push(Number(v & 0x7fn) | 0x80);
        v >>= 7n;
      }
      out.push(Number(v));
      return new Uint8Array(out);
    };
    const encodePermAttr = (): Uint8Array =>
      encodeLenField(
        5,
        concatBytes([
          encodeStrField(1, 'perm'),
          ...[0, 3, 1, 2].map((v) =>
            concatBytes([encodeProtoVarint((8 << 3) | 0), encodeSignedVarint64(v)])
          ),
          encodeVarintField(20, 7),
        ])
      );

    const replacementByNodeIdx = new Map<number, Uint8Array>();
    const normalizedNodeNames: string[] = [];
    for (const vb of verifiedBranches) {
      const rn = vb.reshapeNode;
      const transposeNodeProto = encodeLenField(
        1,
        concatBytes([
          encodeStrField(1, vb.castInputName),
          encodeStrField(2, vb.reshapeOutName),
          encodeStrField(3, rn.name),
          encodeStrField(4, 'Transpose'),
          encodePermAttr(),
        ])
      );
      replacementByNodeIdx.set(rn.nodeIdx, transposeNodeProto);
      normalizedNodeNames.push(rn.name);
    }

    // Preserve topological node order and ALL initializers, inputs, and outputs verbatim
    const finalGraphParts: Uint8Array[] = [];
    let currentNodeIdx = 0;
    for (const gf of graphFields) {
      if (gf.fn === 1 && gf.wt === 2) {
        const idx = currentNodeIdx++;
        const repl = replacementByNodeIdx.get(idx);
        finalGraphParts.push(repl ?? bytes.subarray(gf.fStart, gf.fEnd));
      } else {
        finalGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
      }
    }

    let graphPayloadLen = 0;
    for (const part of finalGraphParts) graphPayloadLen += part.byteLength;
    const graphTag = encodeProtoVarint((7 << 3) | 2);
    const graphLen = encodeProtoVarint(graphPayloadLen);
    let totalLen = 0;
    for (let i = 0; i < topFields.length; i++) {
      totalLen +=
        i === graphIndex
          ? graphTag.byteLength + graphLen.byteLength + graphPayloadLen
          : topFields[i].fEnd - topFields[i].fStart;
    }

    const out = new Uint8Array(totalLen);
    let offset = 0;
    for (let i = 0; i < topFields.length; i++) {
      if (i === graphIndex) {
        out.set(graphTag, offset);
        offset += graphTag.byteLength;
        out.set(graphLen, offset);
        offset += graphLen.byteLength;
        for (const part of finalGraphParts) {
          out.set(part, offset);
          offset += part.byteLength;
        }
      } else {
        const chunk = bytes.subarray(topFields[i].fStart, topFields[i].fEnd);
        out.set(chunk, offset);
        offset += chunk.byteLength;
      }
    }

    const beforeValidation = validateOnnxGraphNodeInputs(bytes);
    const afterValidation = validateOnnxGraphNodeInputs(out);
    const preexistingMissing = new Set(beforeValidation.missingInputs.map((m) => m.inputName));
    const newlyMissing = afterValidation.missingInputs.filter(
      (m) => !preexistingMissing.has(m.inputName)
    );
    if (
      newlyMissing.length > 0 ||
      (beforeValidation.hasGraphInputs && !afterValidation.valid)
    ) {
      return emptyResult;
    }

    const postInference = inferOnnxStaticTensorMeta(out);
    if (
      !postInference.valid ||
      postInference.duplicateOutputs.length > 0 ||
      postInference.duplicateInitializers.length > 0 ||
      postInference.duplicateNodeNames.length > 0 ||
      postInference.nodeCount !== preInference.nodeCount ||
      postInference.initializerCount !== preInference.initializerCount
    ) {
      return emptyResult;
    }

    for (const vb of verifiedBranches) {
      const slot0Post = postInference.tensorMeta[vb.reshapeOutName];
      const concatPost = postInference.tensorMeta[vb.concatOutName];
      if (
        !slot0Post ||
        slot0Post.elemType !== 1 ||
        !sameStaticDims(slot0Post.dims, [vb.batch0, 1, vb.height, vb.width]) ||
        !concatPost ||
        concatPost.elemType !== 1 ||
        concatPost.dims.length !== 4
      ) {
        return emptyResult;
      }
      if (vb.expectedTransposeOut) {
        const tm = postInference.tensorMeta[vb.expectedTransposeOut];
        if (!tm || tm.elemType !== 1 || !sameStaticDims(tm.dims, [vb.batch0, vb.height, vb.width, 16])) {
          return emptyResult;
        }
      }
      if (vb.expectedExpandDimsOut) {
        const em = postInference.tensorMeta[vb.expectedExpandDimsOut];
        if (!em || em.elemType !== 1 || !sameStaticDims(em.dims, [vb.batch0, vb.height, vb.width, 1, 16])) {
          return emptyResult;
        }
      }
    }

    if (requireHitnetContract) {
      const refPost = postInference.tensorMeta['reference_output_disparity'];
      const secPost = postInference.tensorMeta['secondary_output_disparity'];
      if (
        !refPost ||
        !secPost ||
        refPost.elemType !== 1 ||
        secPost.elemType !== 1 ||
        !sameStaticDims(refPost.dims, [1, 240, 320, 1]) ||
        !sameStaticDims(secPost.dims, [1, 240, 320, 1])
      ) {
        return emptyResult;
      }
    }

    return {
      bytes: out,
      normalizedCount: normalizedNodeNames.length,
      normalizedNodeNames,
      verifiedM1Provenance,
      verifiedRawIdentity,
      skippedInvalidOrUnprovenCount: 0,
    };
  } catch {
    return emptyResult;
  }
}

export interface AppendOnnxGraphOutputsOptions {
  maxExtraOutputs?: number;
}

export interface AppendOnnxGraphOutputsResult {
  bytes: Uint8Array;
  baseOutputNames: string[];
  addedOutputs: string[];
  allOutputNames: string[];
  outputMeta: Record<string, StaticTensorMeta>;
}

/**
 * Appends a bounded set of existing static tensors as `GraphProto.output` (`fn = 12`) entries
 * using exact static `elemType` + `dims` from `inferOnnxStaticTensorMeta(bytes)`.
 * Preserves all existing nodes, initializers, inputs, outputs, and `value_info` untouched,
 * and fails closed by throwing an Error on empty/duplicate/over-limit/unknown/dynamic metadata.
 */
export function appendOnnxGraphOutputs(
  bytes: Uint8Array,
  extraOutputNames: readonly string[],
  options: AppendOnnxGraphOutputsOptions = {}
): AppendOnnxGraphOutputsResult {
  const maxExtraOutputs = options.maxExtraOutputs ?? 2;
  if (!Number.isInteger(maxExtraOutputs) || maxExtraOutputs < 1) {
    throw new Error(`Invalid maxExtraOutputs=${String(maxExtraOutputs)}`);
  }
  if (!Array.isArray(extraOutputNames) || extraOutputNames.length === 0) {
    throw new Error('appendOnnxGraphOutputs requires at least 1 extra output tensor name');
  }
  if (extraOutputNames.length > maxExtraOutputs) {
    throw new Error(
      `Bounded tap limit exceeded: requested ${extraOutputNames.length} extra outputs (max=${maxExtraOutputs})`
    );
  }
  const requestedSet = new Set<string>();
  for (const name of extraOutputNames) {
    if (typeof name !== 'string' || name.length === 0) {
      throw new Error('appendOnnxGraphOutputs requires non-empty string tensor names');
    }
    if (requestedSet.has(name)) {
      throw new Error(`Duplicate extra output tensor requested: '${name}'`);
    }
    requestedSet.add(name);
  }

  const preInference = inferOnnxStaticTensorMeta(bytes);
  if (
    !preInference.valid ||
    preInference.hasControlFlowNode ||
    preInference.missingInputs.length > 0 ||
    preInference.duplicateOutputs.length > 0 ||
    preInference.duplicateInitializers.length > 0 ||
    preInference.duplicateNodeNames.length > 0
  ) {
    throw new Error(
      `Source ONNX failed static integrity/shape check before appending outputs (valid=${preInference.valid}, missing=${preInference.missingInputs.length}, dupOut=${preInference.duplicateOutputs.length})`
    );
  }

  const topFields = parseProtoFields(bytes, 0, bytes.byteLength);
  if (!topFields) {
    throw new Error('Invalid ONNX ModelProto protobuf');
  }
  const graphIndex = topFields.findIndex((f) => f.fn === 7 && f.wt === 2);
  if (graphIndex < 0) {
    throw new Error('Missing GraphProto in ONNX model');
  }
  const graphField = topFields[graphIndex];
  const graphFields = parseProtoFields(bytes, graphField.valStart, graphField.valEnd);
  if (!graphFields) {
    throw new Error('Invalid GraphProto protobuf');
  }

  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  const baseOutputNames: string[] = [];
  const baseOutputSet = new Set<string>();

  for (const gf of graphFields) {
    if (gf.fn === 12 && gf.wt === 2) {
      const vi = parseValueInfoStaticTypeAndShape(bytes, gf.valStart, gf.valEnd, decoder);
      const outName = vi?.name ?? getValueInfoName(bytes, gf.valStart, gf.valEnd, decoder);
      if (!outName) {
        throw new Error('Existing GraphProto.output entry is missing tensor name');
      }
      baseOutputNames.push(outName);
      baseOutputSet.add(outName);
    }
  }

  for (const name of extraOutputNames) {
    if (baseOutputSet.has(name)) {
      throw new Error(`Requested tap tensor '${name}' is already a GraphProto.output`);
    }
  }

  const concatLocal = (parts: readonly Uint8Array[]): Uint8Array => {
    let total = 0;
    for (const p of parts) total += p.byteLength;
    const out = new Uint8Array(total);
    let off = 0;
    for (const p of parts) {
      out.set(p, off);
      off += p.byteLength;
    }
    return out;
  };
  const encodeLen = (fn: number, payload: Uint8Array): Uint8Array => {
    const tag = encodeProtoVarint((fn << 3) | 2);
    const len = encodeProtoVarint(payload.byteLength);
    const out = new Uint8Array(tag.byteLength + len.byteLength + payload.byteLength);
    out.set(tag, 0);
    out.set(len, tag.byteLength);
    out.set(payload, tag.byteLength + len.byteLength);
    return out;
  };
  const encodeVarintFld = (fn: number, val: number): Uint8Array =>
    concatLocal([encodeProtoVarint((fn << 3) | 0), encodeProtoVarint(val)]);
  const encodeOutputValueInfoTensor = (
    tensorName: string,
    elemType: number,
    dims: readonly number[]
  ): Uint8Array => {
    const dimMsgs = dims.map((d) => encodeLen(1, encodeVarintFld(1, d)));
    const tensorShape = encodeLen(2, concatLocal(dimMsgs));
    const tensorType = encodeLen(
      1,
      concatLocal([encodeVarintFld(1, elemType), tensorShape])
    );
    const typeProto = encodeLen(2, tensorType);
    return encodeLen(12, concatLocal([encodeLen(1, encoder.encode(tensorName)), typeProto]));
  };

  const addedValueInfoParts: Uint8Array[] = [];
  const addedOutputs: string[] = [];
  const expectedOutputMeta: Record<string, StaticTensorMeta> = {};

  for (const baseName of baseOutputNames) {
    const bm = preInference.tensorMeta[baseName];
    if (
      !bm ||
      !Number.isInteger(bm.elemType) ||
      bm.elemType <= 0 ||
      !Array.isArray(bm.dims) ||
      bm.dims.length === 0 ||
      bm.dims.some((d) => !Number.isInteger(d) || d <= 0)
    ) {
      throw new Error(
        `Missing/dynamic static metadata for existing output '${baseName}': ${JSON.stringify(bm ?? null)}`
      );
    }
    expectedOutputMeta[baseName] = { elemType: bm.elemType, dims: [...bm.dims] };
  }

  for (const outName of extraOutputNames) {
    const meta = preInference.tensorMeta[outName];
    if (
      !meta ||
      !Number.isInteger(meta.elemType) ||
      meta.elemType <= 0 ||
      !Array.isArray(meta.dims) ||
      meta.dims.length === 0 ||
      meta.dims.some((d) => !Number.isInteger(d) || d <= 0)
    ) {
      throw new Error(
        `Unknown or non-static metadata for tap tensor '${outName}': ${JSON.stringify(meta ?? null)}`
      );
    }
    expectedOutputMeta[outName] = { elemType: meta.elemType, dims: [...meta.dims] };
    addedOutputs.push(outName);
    addedValueInfoParts.push(encodeOutputValueInfoTensor(outName, meta.elemType, meta.dims));
  }

  const finalGraphParts: Uint8Array[] = [];
  for (const gf of graphFields) {
    finalGraphParts.push(bytes.subarray(gf.fStart, gf.fEnd));
  }
  for (const vo of addedValueInfoParts) {
    finalGraphParts.push(vo);
  }

  let graphPayloadLen = 0;
  for (const part of finalGraphParts) graphPayloadLen += part.byteLength;
  const graphTag = encodeProtoVarint((7 << 3) | 2);
  const graphLen = encodeProtoVarint(graphPayloadLen);

  let totalLen = 0;
  for (let i = 0; i < topFields.length; i++) {
    totalLen +=
      i === graphIndex
        ? graphTag.byteLength + graphLen.byteLength + graphPayloadLen
        : topFields[i].fEnd - topFields[i].fStart;
  }

  const outBytes = new Uint8Array(totalLen);
  let offset = 0;
  for (let i = 0; i < topFields.length; i++) {
    if (i === graphIndex) {
      outBytes.set(graphTag, offset);
      offset += graphTag.byteLength;
      outBytes.set(graphLen, offset);
      offset += graphLen.byteLength;
      for (const part of finalGraphParts) {
        outBytes.set(part, offset);
        offset += part.byteLength;
      }
    } else {
      const chunk = bytes.subarray(topFields[i].fStart, topFields[i].fEnd);
      outBytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
  }

  const postInference = inferOnnxStaticTensorMeta(outBytes);
  if (
    !postInference.valid ||
    postInference.missingInputs.length > 0 ||
    postInference.duplicateOutputs.length > 0 ||
    postInference.duplicateInitializers.length > 0 ||
    postInference.duplicateNodeNames.length > 0
  ) {
    throw new Error(
      `Post-tap graph validation failed (valid=${postInference.valid}, missing=${postInference.missingInputs.length}, dupOut=${postInference.duplicateOutputs.length})`
    );
  }

  const allOutputNames = [...baseOutputNames, ...addedOutputs];
  for (const outName of allOutputNames) {
    const pm = postInference.tensorMeta[outName];
    const em = expectedOutputMeta[outName];
    if (
      !pm ||
      pm.elemType !== em.elemType ||
      pm.dims.length !== em.dims.length ||
      pm.dims.some((d, i) => d !== em.dims[i])
    ) {
      throw new Error(
        `Post-tap shape mismatch for '${outName}': expected=${JSON.stringify(em)} got=${JSON.stringify(pm ?? null)}`
      );
    }
  }

  return {
    bytes: outBytes,
    baseOutputNames,
    addedOutputs,
    allOutputNames,
    outputMeta: expectedOutputMeta,
  };
}

export function resolveStereoOrtDelegate(
  profile: Pick<StereoModelProfile, 'preferWebGpu'>,
  onnxBytes: Uint8Array,
  maxStorageBuffersPerStage = DEFAULT_WEBGPU_MAX_STORAGE_BUFFERS_PER_STAGE,
  backendPreference: StereoBackendPreference = 'auto'
): 'WebGPU' | 'WASM' {
  if (backendPreference === 'wasm') {
    return 'WASM';
  }
  if (backendPreference === 'webgpu') {
    return 'WebGPU';
  }
  if (!profile.preferWebGpu) {
    return 'WASM';
  }
  const compatibility = inspectStereoOnnxWebGpuCompatibility(onnxBytes, maxStorageBuffersPerStage);
  return compatibility.compatible ? 'WebGPU' : 'WASM';
}

export function resolveSessionTargetResolution(
  session: ort.InferenceSession,
  profile: Pick<StereoModelProfile, 'defaultWidth' | 'defaultHeight'>
): { width: number; height: number } {
  const inputMeta = (session as unknown as {
    inputMetadata?: ReadonlyArray<{
      shape?: readonly (number | string)[];
      dimensions?: readonly (number | string)[];
    }>;
  }).inputMetadata;
  const primaryMeta =
    inputMeta && inputMeta.length > 0 ? inputMeta[inputMeta.length - 1] : undefined;
  const targetDims = primaryMeta?.shape ?? primaryMeta?.dimensions;
  if (targetDims && targetDims.length === 4) {
    const mh = targetDims[2];
    const mw = targetDims[3];
    if (typeof mh === 'number' && mh > 0 && typeof mw === 'number' && mw > 0) {
      return { width: mw, height: mh };
    }
  }
  return { width: profile.defaultWidth, height: profile.defaultHeight };
}

function disposeOrtTensors(values: Iterable<ort.Tensor>): void {
  for (const t of new Set(values)) {
    try {
      t.dispose?.();
    } catch {
      // Ignore cleanup errors
    }
  }
}

export interface StereoOrtSessionOptionOverrides {
  graphOptimizationLevel?: OrtGraphOptimizationLevel;
  disableCpuEpFallback?: boolean;
  enableProfiling?: boolean;
  logSeverityLevel?: 0 | 1 | 2 | 3 | 4;
}

export function resolveStereoOrtSessionOptions(
  executionProviders: string[],
  overrides: StereoOrtSessionOptionOverrides = {}
): ort.InferenceSession.SessionOptions {
  const isWasmOnly =
    executionProviders.length > 0 && executionProviders.every((ep) => ep === 'wasm');
  // ORT's basic/all graph optimization performs ConstantFolding at session creation time, which
  // expands MobileStereoNet-2D disparity/correlation subgraphs to >2.3 GB WASM heap and crashes
  // browser tabs on load. Disabling graph optimization and memory arenas for WASM prevents peak
  // intermediate buffers from remaining pinned in the WASM heap across frames on mobile devices.
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders,
    graphOptimizationLevel: overrides.graphOptimizationLevel ?? (isWasmOnly ? 'disabled' : 'all'),
    enableCpuMemArena: !isWasmOnly,
    enableMemPattern: !isWasmOnly,
    logSeverityLevel: overrides.logSeverityLevel ?? 3,
  };
  if (overrides.enableProfiling) {
    options.enableProfiling = true;
  }
  if (overrides.disableCpuEpFallback) {
    options.extra = {
      session: {
        disable_cpu_ep_fallback: '1',
      },
    };
  }
  return options;
}

export interface OrtWorkerRuntimeLike {
  env: {
    logLevel?: string;
    versions?: { common?: string; web?: string };
    wasm: {
      numThreads?: number;
      proxy?: boolean;
      wasmPaths?: string;
    };
    webgpu?: {
      adapter?: {
        info?: { vendor?: string; architecture?: string; device?: string; description?: string };
        features?: { has?: (name: string) => boolean };
      };
      device?: OrtLikeGpuDevice;
      profiling?: {
        mode?: 'off' | 'default';
        ondata?: (data: { kernelType?: string; kernelName?: string; programName?: string }) => void;
      };
    };
  };
  Tensor: new (dtype: string, data: Float32Array, dims: number[]) => ort.Tensor;
  InferenceSession: {
    create: (bytes: Uint8Array, options: ort.InferenceSession.SessionOptions) => Promise<ort.InferenceSession>;
  };
}

async function createWorkerOrtSessionWithProvidersForRuntime(
  ortRuntime: OrtWorkerRuntimeLike,
  onnxBytes: Uint8Array,
  delegate: 'WebGPU' | 'WASM',
  strictWebGpu = false,
  overrides: StereoOrtSessionOptionOverrides = {},
  webGpuAvailableOverride?: boolean
): Promise<{ session: ort.InferenceSession; backend: OnnxBackendInfo }> {
  const requestedProvider = delegate === 'WASM' ? 'wasm' : 'webgpu';
  const available =
    webGpuAvailableOverride ??
    (typeof navigator !== 'undefined' && Boolean((navigator as { gpu?: unknown }).gpu));
  if (strictWebGpu && delegate !== 'WASM' && !available) {
    throw new Error('WebGPU API unavailable (strictWebGpu=true)');
  }
  const providers =
    delegate === 'WASM' || !available
      ? ['wasm']
      : strictWebGpu
        ? ['webgpu']
        : ['webgpu', 'wasm'];
  const backend: OnnxBackendInfo = {
    runtime: 'onnxruntime-web',
    requestedProvider,
    configuredProviders: providers,
    fallbackReason:
      delegate !== 'WASM' && !available ? 'WebGPU API unavailable; WASM configured' : null,
    actualBackend: 'unmeasured',
  };
  const create = (executionProviders: string[]) =>
    ortRuntime.InferenceSession.create(
      onnxBytes,
      resolveStereoOrtSessionOptions(executionProviders, overrides)
    );
  try {
    return { session: await create(providers), backend };
  } catch (cause) {
    if (!providers.includes('webgpu') || strictWebGpu) {
      throw new Error(`ONNX session initialization failed: ${String(cause)}`, { cause });
    }
    const fallback: OnnxBackendInfo = {
      ...backend,
      configuredProviders: ['wasm'],
      fallbackReason: `WebGPU session creation failed: ${String(cause)}`,
    };
    try {
      return { session: await create(['wasm']), backend: fallback };
    } catch (error) {
      throw new Error(`ONNX WASM fallback initialization failed: ${String(error)}`, {
        cause: error,
      });
    }
  }
}

export async function createWorkerOrtSessionForRuntime(
  ortRuntime: OrtWorkerRuntimeLike,
  onnxBytes: Uint8Array,
  delegate: 'WebGPU' | 'WASM',
  wasmPrimaryUrl: string,
  wasmFallbackUrl: string | null,
  strictWebGpu = false,
  overrides: StereoOrtSessionOptionOverrides = {},
  webGpuAvailableOverride?: boolean
): Promise<{
  session: ort.InferenceSession;
  backend: OnnxBackendInfo;
  wasmSourceUsed: 'primary' | 'fallback';
}> {
  ortRuntime.env.logLevel = overrides.enableProfiling ? 'verbose' : 'error';
  // Running inside a dedicated Worker thread already isolates ONNX execution from the UI thread;
  // disable nested pthread sub-workers so Vite's worker bundle never tries to spawn itself via import.meta.url.
  ortRuntime.env.wasm.numThreads = 1;
  ortRuntime.env.wasm.proxy = false;
  if (wasmPrimaryUrl) {
    ortRuntime.env.wasm.wasmPaths = wasmPrimaryUrl;
  }

  try {
    const created = await createWorkerOrtSessionWithProvidersForRuntime(
      ortRuntime,
      onnxBytes,
      delegate,
      strictWebGpu,
      overrides,
      webGpuAvailableOverride
    );
    return { ...created, wasmSourceUsed: 'primary' };
  } catch (primaryErr) {
    if (wasmFallbackUrl && wasmFallbackUrl !== wasmPrimaryUrl) {
      ortRuntime.env.wasm.wasmPaths = wasmFallbackUrl;
      const created = await createWorkerOrtSessionWithProvidersForRuntime(
        ortRuntime,
        onnxBytes,
        delegate,
        strictWebGpu,
        overrides,
        webGpuAvailableOverride
      );
      return { ...created, wasmSourceUsed: 'fallback' };
    }
    throw primaryErr;
  }
}

// ============================================================================
// Native WebGPU WGSL Stereo Compute Shader Engine (Census + Gradient + Parabolic)
// ============================================================================

const WEBGPU_BUFFER_USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
} as const;

const WEBGPU_MAP_MODE_READ = 0x0001;

interface GpuBufferLike {
  size: number;
  mapAsync: (mode: number) => Promise<void>;
  getMappedRange: () => ArrayBuffer;
  unmap: () => void;
  destroy: () => void;
}

interface GpuCommandEncoderLike {
  beginComputePass: () => {
    setPipeline: (pipeline: unknown) => void;
    setBindGroup: (index: number, bindGroup: unknown) => void;
    dispatchWorkgroups: (x: number, y?: number, z?: number) => void;
    end: () => void;
  };
  copyBufferToBuffer: (
    source: GpuBufferLike,
    sourceOffset: number,
    destination: GpuBufferLike,
    destinationOffset: number,
    size: number
  ) => void;
  finish: () => unknown;
}

interface GpuDeviceLike {
  queue: {
    writeBuffer: (buffer: GpuBufferLike, bufferOffset: number, data: BufferSource) => void;
    submit: (commandBuffers: unknown[]) => void;
  };
  lost?: Promise<{ reason?: string; message?: string }>;
  createShaderModule: (desc: { code: string }) => unknown;
  createComputePipelineAsync: (desc: {
    layout: 'auto';
    compute: { module: unknown; entryPoint: string };
  }) => Promise<{ getBindGroupLayout: (index: number) => unknown }>;
  createBuffer: (desc: { size: number; usage: number }) => GpuBufferLike;
  createBindGroup: (desc: {
    layout: unknown;
    entries: Array<{ binding: number; resource: { buffer: GpuBufferLike } }>;
  }) => unknown;
  createCommandEncoder: () => GpuCommandEncoderLike;
  destroy?: () => void;
}

export const STEREO_WGSL_SHADER = /* wgsl */ `
struct StereoParams {
  width: u32,
  height: u32,
  maxDisp: u32,
  aggRadius: u32,
  meanDiff: f32,
  maxCostNorm: f32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: StereoParams;
@group(0) @binding(1) var<storage, read> leftRgba: array<u32>;
@group(0) @binding(2) var<storage, read> rightRgba: array<u32>;
@group(0) @binding(3) var<storage, read_write> leftGray: array<f32>;
@group(0) @binding(4) var<storage, read_write> rightGray: array<f32>;

@compute @workgroup_size(16, 16)
fn prepare_gray(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  let idx = gid.y * params.width + gid.x;
  let lp = leftRgba[idx];
  let rp = rightRgba[idx];
  let lLum = (77.0 * f32(lp & 0xffu) + 150.0 * f32((lp >> 8u) & 0xffu) + 29.0 * f32((lp >> 16u) & 0xffu)) / 256.0;
  let rLum = (77.0 * f32(rp & 0xffu) + 150.0 * f32((rp >> 8u) & 0xffu) + 29.0 * f32((rp >> 16u) & 0xffu)) / 256.0;
  leftGray[idx] = lLum;
  rightGray[idx] = clamp(rLum + params.meanDiff, 0.0, 255.0);
}

@group(0) @binding(1) var<storage, read> grayLIn: array<f32>;
@group(0) @binding(2) var<storage, read> grayRIn: array<f32>;
@group(0) @binding(3) var<storage, read_write> leftCensus: array<u32>;
@group(0) @binding(4) var<storage, read_write> rightCensus: array<u32>;
@group(0) @binding(5) var<storage, read_write> leftGrad: array<f32>;
@group(0) @binding(6) var<storage, read_write> rightGrad: array<f32>;

fn sample_clamped(buf: ptr<storage, array<f32>, read>, x: i32, y: i32, w: i32, h: i32) -> f32 {
  let cx = clamp(x, 0, w - 1);
  let cy = clamp(y, 0, h - 1);
  return (*buf)[u32(cy * w + cx)];
}

@compute @workgroup_size(16, 16)
fn build_descriptors(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  let w = i32(params.width);
  let h = i32(params.height);
  let ix = i32(gid.x);
  let iy = i32(gid.y);
  let idx = gid.y * params.width + gid.x;

  let cL = grayLIn[idx];
  let cR = grayRIn[idx];

  let gxL = abs(sample_clamped(&grayLIn, ix + 1, iy, w, h) - sample_clamped(&grayLIn, ix - 1, iy, w, h));
  let gyL = abs(sample_clamped(&grayLIn, ix, iy + 1, w, h) - sample_clamped(&grayLIn, ix, iy - 1, w, h));
  let gxR = abs(sample_clamped(&grayRIn, ix + 1, iy, w, h) - sample_clamped(&grayRIn, ix - 1, iy, w, h));
  let gyR = abs(sample_clamped(&grayRIn, ix, iy + 1, w, h) - sample_clamped(&grayRIn, ix, iy - 1, w, h));
  leftGrad[idx] = gxL + 0.5 * gyL;
  rightGrad[idx] = gxR + 0.5 * gyR;

  var maskL: u32 = 0u;
  var maskR: u32 = 0u;
  var bit: u32 = 0u;
  for (var dy: i32 = -2; dy <= 2; dy = dy + 1) {
    for (var dx: i32 = -2; dx <= 2; dx = dx + 1) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      if (sample_clamped(&grayLIn, ix + dx, iy + dy, w, h) > cL + 1.25) {
        maskL = maskL | (1u << bit);
      }
      if (sample_clamped(&grayRIn, ix + dx, iy + dy, w, h) > cR + 1.25) {
        maskR = maskR | (1u << bit);
      }
      bit = bit + 1u;
    }
  }
  leftCensus[idx] = maskL;
  rightCensus[idx] = maskR;
}

@group(0) @binding(1) var<storage, read> mLeftGray: array<f32>;
@group(0) @binding(2) var<storage, read> mRightGray: array<f32>;
@group(0) @binding(3) var<storage, read> mLeftCensus: array<u32>;
@group(0) @binding(4) var<storage, read> mRightCensus: array<u32>;
@group(0) @binding(5) var<storage, read> mLeftGrad: array<f32>;
@group(0) @binding(6) var<storage, read> mRightGrad: array<f32>;
@group(0) @binding(7) var<storage, read_write> rawDisp: array<f32>;

@compute @workgroup_size(16, 16)
fn match_disparity(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  let w = i32(params.width);
  let h = i32(params.height);
  let ix = i32(gid.x);
  let iy = i32(gid.y);
  let r = i32(params.aggRadius);
  let maxD = i32(params.maxDisp);

  var bestD: i32 = 0;
  var bestCost: f32 = 1e20;
  var secondBestCost: f32 = 1e20;
  var cAtBestMinus1: f32 = 1e20;
  var cAtBestPlus1: f32 = 1e20;
  var prevCost: f32 = 1e20;

  let side = r * 2 + 1;
  let sampleCount = f32(side * side);

  var gradSum: f32 = 0.0;
  for (var ky: i32 = -r; ky <= r; ky = ky + 1) {
    let sy = clamp(iy + ky, 0, h - 1);
    let rowBase = sy * w;
    for (var kx: i32 = -r; kx <= r; kx = kx + 1) {
      let lx = clamp(ix + kx, 0, w - 1);
      gradSum = gradSum + mLeftGrad[u32(rowBase + lx)];
    }
  }
  let meanGrad = gradSum / sampleCount;

  var costHist: array<f32, 96>;

  for (var d: i32 = 0; d <= maxD; d = d + 1) {
    var accCost: f32 = 0.0;
    for (var ky: i32 = -r; ky <= r; ky = ky + 1) {
      let sy = clamp(iy + ky, 0, h - 1);
      let rowBase = sy * w;
      for (var kx: i32 = -r; kx <= r; kx = kx + 1) {
        let lx = clamp(ix + kx, 0, w - 1);
        let rxRaw = lx - d;
        let rx = clamp(rxRaw, 0, w - 1);
        let li = u32(rowBase + lx);
        let ri = u32(rowBase + rx);
        let cDiff = f32(countOneBits(mLeftCensus[li] ^ mRightCensus[ri]));
        let lDiff = min(abs(mLeftGray[li] - mRightGray[ri]), 42.0);
        let gDiff = min(abs(mLeftGrad[li] - mRightGrad[ri]), 36.0);
        let borderPenalty = select(0.0, 28.0, rxRaw < 0);
        accCost = accCost + (3.0 * cDiff + 0.24 * lDiff + 0.20 * gDiff + borderPenalty);
      }
    }
    let totalCost = accCost + 0.09 * f32(d) * sampleCount;
    if (d < 96) {
      costHist[u32(d)] = totalCost;
    }
    if (d > 0 && bestD == d - 1) {
      cAtBestPlus1 = totalCost;
    }
    if (totalCost < bestCost) {
      bestCost = totalCost;
      bestD = d;
      cAtBestMinus1 = select(totalCost, prevCost, d > 0);
      cAtBestPlus1 = totalCost;
    }
    prevCost = totalCost;
  }

  let histLimit = min(maxD, 95);
  for (var d: i32 = 0; d <= histLimit; d = d + 1) {
    if (abs(d - bestD) >= 2) {
      let c = costHist[u32(d)];
      if (c < secondBestCost) {
        secondBestCost = c;
      }
    }
  }

  let meanCost = bestCost / sampleCount;
  let outIdx = gid.y * params.width + gid.x;
  if (meanCost > params.maxCostNorm) {
    rawDisp[outIdx] = -1.0;
    return;
  }

  // Reject ambiguous high-disparity false peaks in low-texture / compressed real-world footage
  let uniqRatio = (secondBestCost - bestCost) / max(bestCost, sampleCount * 2.5);
  if (bestD > 16 && (uniqRatio < 0.11 || (meanGrad < 4.5 && uniqRatio < 0.22))) {
    rawDisp[outIdx] = -1.0;
    return;
  }
  if (meanGrad < 2.2 && uniqRatio < 0.08 && bestD > 8) {
    rawDisp[outIdx] = -1.0;
    return;
  }

  var sub = f32(bestD);
  if (bestD > 0 && bestD < maxD) {
    let denom = 2.0 * (cAtBestMinus1 - 2.0 * bestCost + cAtBestPlus1);
    if (denom > 1e-4) {
      let delta = clamp((cAtBestMinus1 - cAtBestPlus1) / denom, -0.5, 0.5);
      sub = sub + delta;
    }
  }
  rawDisp[outIdx] = sub;
}

@group(0) @binding(1) var<storage, read> filtInDisp: array<f32>;
@group(0) @binding(2) var<storage, read_write> filtOutDisp: array<f32>;
@group(0) @binding(3) var<storage, read> filtLeftGray: array<f32>;

@compute @workgroup_size(16, 16)
fn median_filter_3x3(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  let idx = gid.y * params.width + gid.x;
  let center = filtInDisp[idx];
  if (center < 0.0) {
    filtOutDisp[idx] = -1.0;
    return;
  }

  let w = i32(params.width);
  let h = i32(params.height);
  let ix = i32(gid.x);
  let iy = i32(gid.y);

  // 13-tap 5x5 strided median to reject macroblock speckles + 5x5 edge-aware bilateral average
  var vals: array<f32, 13>;
  var cnt: u32 = 0u;
  let offX = array<i32, 13>(-2, 0, 2, -1, 0, 1, -2, 0, 2, -1, 0, 1, 0);
  let offY = array<i32, 13>(-2, -2, -2, -1, -1, -1, 0, 0, 0, 1, 1, 1, 2);

  for (var k: u32 = 0u; k < 13u; k = k + 1u) {
    let sx = clamp(ix + offX[k], 0, w - 1);
    let sy = clamp(iy + offY[k], 0, h - 1);
    let v = filtInDisp[u32(sy * w + sx)];
    if (v >= 0.0) {
      vals[cnt] = v;
      cnt = cnt + 1u;
    }
  }

  if (cnt == 0u) {
    filtOutDisp[idx] = -1.0;
    return;
  }

  for (var a: u32 = 1u; a < cnt; a = a + 1u) {
    let key = vals[a];
    var b: i32 = i32(a) - 1;
    while (b >= 0 && vals[u32(b)] > key) {
      vals[u32(b + 1)] = vals[u32(b)];
      b = b - 1;
    }
    vals[u32(b + 1)] = key;
  }

  let med = vals[cnt >> 1u];
  let centerLum = filtLeftGray[idx];
  var wSum: f32 = 0.0;
  var dSum: f32 = 0.0;

  for (var dy: i32 = -2; dy <= 2; dy = dy + 1) {
    let sy = clamp(iy + dy, 0, h - 1);
    let rBase = sy * w;
    for (var dx: i32 = -2; dx <= 2; dx = dx + 1) {
      let sx = clamp(ix + dx, 0, w - 1);
      let nIdx = u32(rBase + sx);
      let nv = filtInDisp[nIdx];
      if (nv >= 0.0 && abs(nv - med) <= 4.5) {
        let lumDiff = abs(filtLeftGray[nIdx] - centerLum);
        let weight = exp(-lumDiff * 0.06);
        wSum = wSum + weight;
        dSum = dSum + weight * nv;
      }
    }
  }

  filtOutDisp[idx] = select(med, dSum / wSum, wSum > 1e-4);
}
`;

export interface WebGpuStereoComputeEngine {
  estimate: (
    leftRgba: Uint8ClampedArray,
    rightRgba: Uint8ClampedArray,
    width: number,
    height: number,
    options: {
      maxDisparity: number;
      windowSize: 5 | 7 | 9;
      confidenceThreshold: number;
    }
  ) => Promise<{ disparity: Float32Array; width: number; height: number }>;
  destroy: () => void;
}

export async function createWebGpuStereoComputeEngine(): Promise<WebGpuStereoComputeEngine> {
  const gpuNav =
    typeof navigator !== 'undefined'
      ? (navigator as {
          gpu?: {
            requestAdapter?: (opts?: { powerPreference?: string }) => Promise<{
              requestDevice?: () => Promise<GpuDeviceLike>;
            } | null>;
          };
        }).gpu
      : undefined;

  if (!gpuNav || typeof gpuNav.requestAdapter !== 'function') {
    throw new Error('navigator.gpu unavailable');
  }

  const adapter = await gpuNav.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter || typeof adapter.requestDevice !== 'function') {
    throw new Error('WebGPU adapter unavailable');
  }

  const device = await adapter.requestDevice();
  let lost = false;
  device.lost?.then(() => {
    lost = true;
  }).catch(() => {
    lost = true;
  });

  const shaderModule = device.createShaderModule({ code: STEREO_WGSL_SHADER });
  const [prepPipeline, descPipeline, matchPipeline, medianPipeline] = await Promise.all([
    device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'prepare_gray' },
    }),
    device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'build_descriptors' },
    }),
    device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'match_disparity' },
    }),
    device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'median_filter_3x3' },
    }),
  ]);

  const paramsBuffer = device.createBuffer({
    size: 32,
    usage: WEBGPU_BUFFER_USAGE.UNIFORM | WEBGPU_BUFFER_USAGE.COPY_DST,
  });

  let allocatedPixels = 0;
  let leftRgbaBuf: GpuBufferLike | null = null;
  let rightRgbaBuf: GpuBufferLike | null = null;
  let leftGrayBuf: GpuBufferLike | null = null;
  let rightGrayBuf: GpuBufferLike | null = null;
  let leftCensusBuf: GpuBufferLike | null = null;
  let rightCensusBuf: GpuBufferLike | null = null;
  let leftGradBuf: GpuBufferLike | null = null;
  let rightGradBuf: GpuBufferLike | null = null;
  let rawDispBuf: GpuBufferLike | null = null;
  let outDispBuf: GpuBufferLike | null = null;
  let readbackBuf: GpuBufferLike | null = null;

  let prepBg: unknown = null;
  let descBg: unknown = null;
  let matchBg: unknown = null;
  let medianBg: unknown = null;

  const releasePixelBuffers = () => {
    for (const buf of [
      leftRgbaBuf,
      rightRgbaBuf,
      leftGrayBuf,
      rightGrayBuf,
      leftCensusBuf,
      rightCensusBuf,
      leftGradBuf,
      rightGradBuf,
      rawDispBuf,
      outDispBuf,
      readbackBuf,
    ]) {
      try {
        buf?.destroy();
      } catch {
        // Ignore buffer cleanup error
      }
    }
    allocatedPixels = 0;
    leftRgbaBuf = null;
    rightRgbaBuf = null;
    leftGrayBuf = null;
    rightGrayBuf = null;
    leftCensusBuf = null;
    rightCensusBuf = null;
    leftGradBuf = null;
    rightGradBuf = null;
    rawDispBuf = null;
    outDispBuf = null;
    readbackBuf = null;
    prepBg = null;
    descBg = null;
    matchBg = null;
    medianBg = null;
  };

  const ensurePixelBuffers = (pixelCount: number) => {
    if (allocatedPixels === pixelCount && prepBg && descBg && matchBg && medianBg) {
      return;
    }
    releasePixelBuffers();
    const byteSize = Math.max(16, pixelCount * 4);
    leftRgbaBuf = device.createBuffer({
      size: byteSize,
      usage: WEBGPU_BUFFER_USAGE.STORAGE | WEBGPU_BUFFER_USAGE.COPY_DST,
    });
    rightRgbaBuf = device.createBuffer({
      size: byteSize,
      usage: WEBGPU_BUFFER_USAGE.STORAGE | WEBGPU_BUFFER_USAGE.COPY_DST,
    });
    leftGrayBuf = device.createBuffer({ size: byteSize, usage: WEBGPU_BUFFER_USAGE.STORAGE });
    rightGrayBuf = device.createBuffer({ size: byteSize, usage: WEBGPU_BUFFER_USAGE.STORAGE });
    leftCensusBuf = device.createBuffer({ size: byteSize, usage: WEBGPU_BUFFER_USAGE.STORAGE });
    rightCensusBuf = device.createBuffer({ size: byteSize, usage: WEBGPU_BUFFER_USAGE.STORAGE });
    leftGradBuf = device.createBuffer({ size: byteSize, usage: WEBGPU_BUFFER_USAGE.STORAGE });
    rightGradBuf = device.createBuffer({ size: byteSize, usage: WEBGPU_BUFFER_USAGE.STORAGE });
    rawDispBuf = device.createBuffer({ size: byteSize, usage: WEBGPU_BUFFER_USAGE.STORAGE });
    outDispBuf = device.createBuffer({
      size: byteSize,
      usage: WEBGPU_BUFFER_USAGE.STORAGE | WEBGPU_BUFFER_USAGE.COPY_SRC,
    });
    readbackBuf = device.createBuffer({
      size: byteSize,
      usage: WEBGPU_BUFFER_USAGE.MAP_READ | WEBGPU_BUFFER_USAGE.COPY_DST,
    });

    prepBg = device.createBindGroup({
      layout: prepPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: leftRgbaBuf } },
        { binding: 2, resource: { buffer: rightRgbaBuf } },
        { binding: 3, resource: { buffer: leftGrayBuf } },
        { binding: 4, resource: { buffer: rightGrayBuf } },
      ],
    });
    descBg = device.createBindGroup({
      layout: descPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: leftGrayBuf } },
        { binding: 2, resource: { buffer: rightGrayBuf } },
        { binding: 3, resource: { buffer: leftCensusBuf } },
        { binding: 4, resource: { buffer: rightCensusBuf } },
        { binding: 5, resource: { buffer: leftGradBuf } },
        { binding: 6, resource: { buffer: rightGradBuf } },
      ],
    });
    matchBg = device.createBindGroup({
      layout: matchPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: leftGrayBuf } },
        { binding: 2, resource: { buffer: rightGrayBuf } },
        { binding: 3, resource: { buffer: leftCensusBuf } },
        { binding: 4, resource: { buffer: rightCensusBuf } },
        { binding: 5, resource: { buffer: leftGradBuf } },
        { binding: 6, resource: { buffer: rightGradBuf } },
        { binding: 7, resource: { buffer: rawDispBuf } },
      ],
    });
    medianBg = device.createBindGroup({
      layout: medianPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: rawDispBuf } },
        { binding: 2, resource: { buffer: outDispBuf } },
        { binding: 3, resource: { buffer: leftGrayBuf } },
      ],
    });
    allocatedPixels = pixelCount;
  };

  return {
    async estimate(leftRgba, rightRgba, width, height, options) {
      if (lost) {
        throw new Error('WebGPU device was lost');
      }
      const n = width * height;
      if (n <= 0 || leftRgba.byteLength < n * 4 || rightRgba.byteLength < n * 4) {
        throw new RangeError(`Invalid RGBA buffers for WGSL stereo (${width}x${height})`);
      }

      // Strided luminance exposure compensation across left & right eyes (< 0.05ms)
      let sumL = 0;
      let sumR = 0;
      let sampleCnt = 0;
      const stride = Math.max(1, Math.floor(n / 4096));
      for (let i = 0; i < n; i += stride) {
        const o = i * 4;
        sumL += (77 * leftRgba[o] + 150 * leftRgba[o + 1] + 29 * leftRgba[o + 2]) >> 8;
        sumR += (77 * rightRgba[o] + 150 * rightRgba[o + 1] + 29 * rightRgba[o + 2]) >> 8;
        sampleCnt++;
      }
      const meanDiff = sampleCnt > 0 ? Math.round((sumL - sumR) / sampleCnt) : 0;
      const effectiveMeanDiff = Math.abs(meanDiff) >= 2 ? meanDiff : 0;

      const maxDisp = Math.max(4, Math.min(width - 2, Math.floor(options.maxDisparity)));
      const aggRadius = options.windowSize === 9 ? 4 : options.windowSize === 7 ? 3 : 2;
      const confThresh = Math.max(0, Math.min(1, options.confidenceThreshold));
      const maxCostNorm = Math.max(16, 62 - confThresh * 34);

      ensurePixelBuffers(n);

      const paramBuf = new ArrayBuffer(32);
      const paramU32 = new Uint32Array(paramBuf);
      const paramF32 = new Float32Array(paramBuf);
      paramU32[0] = width;
      paramU32[1] = height;
      paramU32[2] = maxDisp;
      paramU32[3] = aggRadius;
      paramF32[4] = effectiveMeanDiff;
      paramF32[5] = maxCostNorm;

      device.queue.writeBuffer(paramsBuffer, 0, paramBuf);
      device.queue.writeBuffer(
        leftRgbaBuf!,
        0,
        leftRgba.buffer.slice(leftRgba.byteOffset, leftRgba.byteOffset + n * 4) as ArrayBuffer
      );
      device.queue.writeBuffer(
        rightRgbaBuf!,
        0,
        rightRgba.buffer.slice(rightRgba.byteOffset, rightRgba.byteOffset + n * 4) as ArrayBuffer
      );

      const wgX = Math.ceil(width / 16);
      const wgY = Math.ceil(height / 16);
      const encoder = device.createCommandEncoder();

      for (const [pipeline, bg] of [
        [prepPipeline, prepBg],
        [descPipeline, descBg],
        [matchPipeline, matchBg],
        [medianPipeline, medianBg],
      ] as const) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(wgX, wgY, 1);
        pass.end();
      }

      encoder.copyBufferToBuffer(outDispBuf!, 0, readbackBuf!, 0, n * 4);
      device.queue.submit([encoder.finish()]);

      await readbackBuf!.mapAsync(WEBGPU_MAP_MODE_READ);
      const mapped = readbackBuf!.getMappedRange();
      const disparity = new Float32Array(mapped.slice(0));
      readbackBuf!.unmap();

      return { disparity, width, height };
    },
    destroy() {
      releasePixelBuffers();
      try {
        paramsBuffer.destroy();
      } catch {
        // Ignore cleanup error
      }
      try {
        device.destroy?.();
      } catch {
        // Ignore device cleanup error
      }
    },
  };
}

export interface StereoWorkerSessionCreator {
  (
    onnxBytes: Uint8Array,
    delegate: 'WebGPU' | 'WASM',
    wasmPrimaryUrl: string,
    wasmFallbackUrl: string | null,
    strictWebGpu?: boolean,
    sessionOptionOverrides?: StereoOrtSessionOptionOverrides
  ): Promise<{
    session: ort.InferenceSession;
    backend: OnnxBackendInfo;
    wasmSourceUsed: 'primary' | 'fallback';
    runtimeAssetFamily?: 'asyncify' | 'jsep';
    entryModuleUrl?: string;
    loadedOrtResources?: string[];
    foreignOrtResources?: string[];
    loadedAssetProvenance?: OrtLoadedAssetProvenance[];
  }>;
}

export interface StereoWorkerProcessorOptions {
  allowDevPretransformedExactM1SkipConcatSplit?: boolean;
}

export function createStereoWorkerProcessorWithOrt(
  ortRuntimeOrGetter: OrtWorkerRuntimeLike | (() => OrtWorkerRuntimeLike | null),
  sessionCreator?: StereoWorkerSessionCreator,
  wgslEngineFactory: () => Promise<WebGpuStereoComputeEngine> = createWebGpuStereoComputeEngine,
  onDisposeRuntime?: () => Promise<void> | void,
  processorOptions?: StereoWorkerProcessorOptions
) {
  const getOrt = (): OrtWorkerRuntimeLike => {
    const resolved =
      typeof ortRuntimeOrGetter === 'function' ? ortRuntimeOrGetter() : ortRuntimeOrGetter;
    if (!resolved) {
      throw new Error('ORT runtime is not initialized');
    }
    return resolved;
  };
  const resolvedSessionCreator: StereoWorkerSessionCreator =
    sessionCreator ??
    ((
      onnxBytes,
      delegate,
      wasmPrimaryUrl,
      wasmFallbackUrl,
      strictWebGpu,
      sessionOptionOverrides
    ) =>
      createWorkerOrtSessionForRuntime(
        getOrt(),
        onnxBytes,
        delegate,
        wasmPrimaryUrl,
        wasmFallbackUrl,
        strictWebGpu,
        sessionOptionOverrides
      ));

  let activeProfile: StereoModelProfile | null = null;
  let activeSession: ort.InferenceSession | null = null;
  let activeWgslEngine: WebGpuStereoComputeEngine | null = null;
  let activeBackend: OnnxBackendInfo | null = null;
  let activeOnnxBytes: Uint8Array | null = null;
  let activeWasmPrimaryUrl = '';
  let activeWasmFallbackUrl: string | null = null;
  let activeStrictWebGpu = false;
  let activeEnableWebGpuProfiling = false;
  let gpuProfileEvents: Array<{ kernelType: string; programName: string }> = [];
  let attemptedWasmFallback = false;
  let wgslRuntimeFailed = false;

  const configureWebGpuProfiling = (enabled: boolean) => {
    gpuProfileEvents = [];
    let ortEnv: OrtWorkerRuntimeLike['env'];
    try {
      ortEnv = getOrt().env;
    } catch {
      return;
    }
    if (!ortEnv.webgpu) return;
    if (enabled) {
      ortEnv.webgpu.profiling = {
        mode: 'default',
        ondata: (data) => {
          gpuProfileEvents.push({
            kernelType: data?.kernelType || 'unknown',
            programName: data?.programName || data?.kernelName || '',
          });
        },
      };
    } else {
      ortEnv.webgpu.profiling = { mode: 'off' };
    }
  };

  const releaseActiveSession = async () => {
    const current = activeSession;
    const currentWgsl = activeWgslEngine;
    activeSession = null;
    activeWgslEngine = null;
    activeOnnxBytes = null;
    activeStrictWebGpu = false;
    activeEnableWebGpuProfiling = false;
    gpuProfileEvents = [];
    attemptedWasmFallback = false;
    wgslRuntimeFailed = false;
    configureWebGpuProfiling(false);
    if (currentWgsl) {
      try {
        currentWgsl.destroy();
      } catch {
        // Ignore WebGPU cleanup errors
      }
    }
    if (current) {
      try {
        await current.release();
      } catch {
        // Ignore session release errors
      }
    }
    if (onDisposeRuntime) {
      try {
        await onDisposeRuntime();
      } catch {
        // Ignore runtime cleanup errors
      }
    }
  };

  let messageChain: Promise<void> = Promise.resolve();

  const processMessage = async (
    msg: StereoWorkerRequest,
    postResponse: (response: StereoWorkerResponse, transfer?: Transferable[]) => void
  ): Promise<void> => {
    if (!msg || typeof msg !== 'object') return;

    if (msg.type === 'init-sad') {
      await releaseActiveSession();
      activeProfile = msg.profile;
      activeBackend = {
        runtime: 'stereo-sad-cpu',
        requestedProvider: 'cpu',
        configuredProviders: ['cpu-sad'],
        fallbackReason: null,
        actualBackend: 'unmeasured',
      };
      postResponse({
        type: 'init-result',
        id: msg.id,
        ok: true,
        backend: activeBackend,
        wasmSourceUsed: 'primary',
        targetWidth: msg.profile.defaultWidth,
        targetHeight: msg.profile.defaultHeight,
      });
      return;
    }

    if (msg.type === 'init-wgsl') {
      await releaseActiveSession();
      activeProfile = msg.profile;
      const pref = msg.backendPreference ?? 'auto';
      if (pref === 'wasm') {
        activeBackend = {
          runtime: 'stereo-webgpu-wgsl',
          requestedProvider: 'wasm',
          configuredProviders: ['cpu-sad'],
          fallbackReason: '手動設定により WASM/CPU SAD で実行',
          actualBackend: 'unmeasured',
        };
        postResponse({
          type: 'init-result',
          id: msg.id,
          ok: true,
          backend: activeBackend,
          wasmSourceUsed: 'primary',
          targetWidth: 320,
          targetHeight: 180,
        });
        return;
      }

      try {
        activeWgslEngine = await wgslEngineFactory();
        activeBackend = {
          runtime: 'stereo-webgpu-wgsl',
          requestedProvider: 'webgpu',
          configuredProviders: ['webgpu-wgsl'],
          fallbackReason: null,
          actualBackend: 'unmeasured',
        };
        postResponse({
          type: 'init-result',
          id: msg.id,
          ok: true,
          backend: activeBackend,
          wasmSourceUsed: 'primary',
          targetWidth: msg.profile.defaultWidth,
          targetHeight: msg.profile.defaultHeight,
        });
      } catch (err) {
        const reason = err instanceof Error ? err.message : String(err);
        activeWgslEngine = null;
        activeBackend = {
          runtime: 'stereo-webgpu-wgsl',
          requestedProvider: 'webgpu',
          configuredProviders: ['cpu-sad'],
          fallbackReason: `WebGPU WGSL利用不可 (${reason}); CPU SAD へ自動フォールバック`,
          actualBackend: 'unmeasured',
        };
        postResponse({
          type: 'init-result',
          id: msg.id,
          ok: true,
          backend: activeBackend,
          wasmSourceUsed: 'primary',
          targetWidth: 320,
          targetHeight: 180,
        });
      }
      return;
    }

    if (msg.type === 'init-onnx') {
      try {
        const tInitStart = performance.now();
        await releaseActiveSession();
        const rawBytes = new Uint8Array(msg.onnxBuffer);
        const rawInputModelSha256 = await computeSha256Hex(rawBytes);
        const backendPref = msg.backendPreference ?? 'auto';
        const strictWebGpu = Boolean(msg.strictWebGpu);
        const isHitnetVariant = msg.profile.variantId === 'hitnet-sf';
        // Diagnostic-only override: allows running the exact WebGPU-transformed graph on WASM inside
        // bounded same-Worker A/B comparisons without ever changing normal/explicit/fallback WASM bytes.
        const diagnosticTransformedWasm =
          Boolean(msg.allowDiagnosticTransformedWasm) && backendPref === 'wasm' && isHitnetVariant;

        // Narrow DEV-only exact-M1 pretransformed fixture path: skips ONLY implicit Concat
        // splitting on exact verified M1 bytes (already containing decomposed 5-input Concats)
        // when authorized by the DEV fixture worker protocol. Production worker keeps this false.
        const authorizedDevFixtureProcessor =
          Boolean(import.meta.env.DEV) &&
          Boolean(processorOptions?.allowDevPretransformedExactM1SkipConcatSplit);
        const requestedSkipConcatSplit = Boolean(msg.skipImplicitConcatSplitForExactM1Fixture);
        const skipImplicitConcatSplit =
          authorizedDevFixtureProcessor &&
          requestedSkipConcatSplit &&
          isHitnetVariant &&
          msg.profile.engineType === 'onnx' &&
          backendPref === 'webgpu' &&
          strictWebGpu &&
          msg.graphOptimizationLevel === 'basic' &&
          !diagnosticTransformedWasm &&
          (!msg.normalizeSliceAxes || msg.normalizeSliceAxes === 'off') &&
          !msg.normalizeGatherNdRank &&
          !msg.normalizePropGatherNdRank &&
          (!msg.levelInitGatherMode || msg.levelInitGatherMode === 'off') &&
          !msg.normalizeInitToPropZeroSlant &&
          !msg.normalizeSlot0ReshapeToTranspose &&
          !msg.normalizeArgMinKeepDims &&
          (!Array.isArray(msg.diagnosticTapOutputs) || msg.diagnosticTapOutputs.length === 0) &&
          rawBytes.byteLength === HITNET_M1_FULL_GRAPH_BYTE_LENGTH &&
          rawInputModelSha256 === HITNET_M1_FULL_GRAPH_SHA256;
        if (authorizedDevFixtureProcessor && requestedSkipConcatSplit && !skipImplicitConcatSplit) {
          throw new Error(
            `[model-guard] DEV exact-M1 Concat-split skip preconditions failed: len=${rawBytes.byteLength} (expected ${HITNET_M1_FULL_GRAPH_BYTE_LENGTH}), sha=${rawInputModelSha256} (expected ${HITNET_M1_FULL_GRAPH_SHA256}), backend=${backendPref}, strict=${strictWebGpu}, opt=${String(msg.graphOptimizationLevel)}`
          );
        }

        const splitResult =
          (backendPref === 'wasm' && !diagnosticTransformedWasm) || skipImplicitConcatSplit
            ? { bytes: rawBytes, splitCount: 0, originalMaxConcatInputs: 0 }
            : splitLargeOnnxConcatNodes(rawBytes, 4);
        const afterSplitBytes =
          backendPref === 'wasm' && !diagnosticTransformedWasm
            ? rawBytes
            : splitResult.splitCount > 0
              ? splitResult.bytes
              : stripOnnxGraphValueInfo(rawBytes);

        const allowGraphRewrites = backendPref !== 'wasm' || diagnosticTransformedWasm;
        const conv3dResult = allowGraphRewrites
          ? decomposeOnnx5dConvNodesForWebGpu(afterSplitBytes)
          : { bytes: afterSplitBytes, decomposedCount: 0, decomposedNodeNames: [] };
        const afterConv3dBytes =
          conv3dResult.decomposedCount > 0 ? conv3dResult.bytes : afterSplitBytes;

        // Gate Slice, GatherND & ArgMin normalizations strictly to hitnet-sf and default production to OFF
        // until real tufa15 browser A/B verification confirms full-graph WebGPU parity & speedup.
        const requestedSliceMode: SliceAxesNormalizeMode =
          msg.normalizeSliceAxes === 'compact-active'
            ? 'compact-active'
            : msg.normalizeSliceAxes === 'sort-only' || msg.normalizeSliceAxes === true
              ? 'sort-only'
              : 'off';
        const sliceNormalizeMode: SliceAxesNormalizeMode =
          allowGraphRewrites && isHitnetVariant ? requestedSliceMode : 'off';

        const sliceNormResult =
          sliceNormalizeMode !== 'off'
            ? normalizeNonAscendingSliceAxesForWebGpu(afterConv3dBytes, {
                mode: sliceNormalizeMode,
              })
            : {
                bytes: afterConv3dBytes,
                normalizedCount: 0,
                normalizedNodeNames: [],
                skippedDynamicCount: 0,
                skippedInvalidOrUnprovenCount: 0,
                compactedCount: 0,
                mode: sliceNormalizeMode,
              };
        const afterSliceBytes =
          sliceNormResult.normalizedCount > 0 ? sliceNormResult.bytes : afterConv3dBytes;

        const gatherNdRankNormalized =
          Boolean(msg.normalizeGatherNdRank) && allowGraphRewrites && isHitnetVariant;
        const gatherNdResult = gatherNdRankNormalized
          ? normalizeHitnetGatherNdRankForWebGpu(afterSliceBytes, {
              variantId: msg.profile.variantId,
              backendPreference: backendPref,
              requireHitnetContract: true,
              allowDiagnosticTransformedWasm: diagnosticTransformedWasm,
            })
          : {
              bytes: afterSliceBytes,
              normalizedCount: 0,
              normalizedNodeNames: [],
              skippedDynamicCount: 0,
              skippedInvalidOrUnprovenCount: 0,
              totalGatherNdCount: 0,
            };
        const afterGatherNdBytes =
          gatherNdResult.normalizedCount > 0 ? gatherNdResult.bytes : afterSliceBytes;

        const propGatherNdRequested =
          Boolean(msg.normalizePropGatherNdRank) && allowGraphRewrites && isHitnetVariant;
        const propGatherNdPreconditionsMet =
          conv3dResult.decomposedCount > 0 &&
          sliceNormResult.compactedCount > 0 &&
          gatherNdResult.normalizedCount === 4;
        const propGatherNdResult =
          propGatherNdRequested && propGatherNdPreconditionsMet
            ? normalizeHitnetPropGatherNdRankForWebGpu(afterGatherNdBytes, {
                variantId: msg.profile.variantId,
                backendPreference: backendPref,
                requireHitnetContract: true,
                allowDiagnosticTransformedWasm: diagnosticTransformedWasm,
              })
            : {
                bytes: afterGatherNdBytes,
                normalizedCount: 0,
                normalizedNodeNames: [],
                skippedDynamicCount: 0,
                skippedInvalidOrUnprovenCount: propGatherNdRequested ? 1 : 0,
                totalGatherNdCount: 0,
              };
        if (
          propGatherNdRequested &&
          (msg.strictWebGpu || diagnosticTransformedWasm) &&
          (!propGatherNdPreconditionsMet || propGatherNdResult.normalizedCount !== 2)
        ) {
          throw new Error(
            `H1 prop0 GatherND normalization fail-closed: preconditionsMet=${propGatherNdPreconditionsMet}, normalizedCount=${propGatherNdResult.normalizedCount}, skippedInvalid=${propGatherNdResult.skippedInvalidOrUnprovenCount}`
          );
        }
        const afterPropGatherNdBytes =
          propGatherNdResult.normalizedCount > 0 ? propGatherNdResult.bytes : afterGatherNdBytes;

        const requestedLevelInitGatherMode: HitnetLevelInitGatherMode =
          msg.levelInitGatherMode === 'reshape-rank4' || msg.levelInitGatherMode === 'reduce-min'
            ? msg.levelInitGatherMode
            : 'off';
        const levelInitGatherAppliedMode: HitnetLevelInitGatherMode =
          allowGraphRewrites && isHitnetVariant ? requestedLevelInitGatherMode : 'off';
        const levelInitGatherResult: OnnxLevelInitGatherNormalizeResult =
          levelInitGatherAppliedMode === 'reshape-rank4' && propGatherNdResult.normalizedCount === 2
            ? normalizeHitnetLevelInitGatherNdRankForWebGpu(afterPropGatherNdBytes, {
                variantId: msg.profile.variantId,
                backendPreference: backendPref,
                requireHitnetContract: true,
                allowDiagnosticTransformedWasm: diagnosticTransformedWasm,
              })
            : levelInitGatherAppliedMode === 'reduce-min' && propGatherNdResult.normalizedCount === 2
              ? normalizeHitnetLevelInitReduceMinForWebGpu(afterPropGatherNdBytes, {
                  variantId: msg.profile.variantId,
                  backendPreference: backendPref,
                  requireHitnetContract: true,
                  allowDiagnosticTransformedWasm: diagnosticTransformedWasm,
                })
              : {
                  bytes: afterPropGatherNdBytes,
                  mode: levelInitGatherAppliedMode,
                  normalizedCount: 0,
                  normalizedNodeNames: [],
                  skippedDynamicCount: 0,
                  skippedInvalidOrUnprovenCount: levelInitGatherAppliedMode !== 'off' ? 1 : 0,
                  totalGatherNdCount: 0,
                  verifiedCostLineageCount: 0,
                  documentedLimitations: [],
                };
        if (
          levelInitGatherAppliedMode !== 'off' &&
          (msg.strictWebGpu || diagnosticTransformedWasm) &&
          (!propGatherNdPreconditionsMet ||
            propGatherNdResult.normalizedCount !== 2 ||
            levelInitGatherResult.normalizedCount !== 2)
        ) {
          throw new Error(
            `HITNet level_init ${levelInitGatherAppliedMode} fail-closed: h1Count=${propGatherNdResult.normalizedCount}, normalizedCount=${levelInitGatherResult.normalizedCount}, skippedInvalid=${levelInitGatherResult.skippedInvalidOrUnprovenCount}`
          );
        }
        const afterLevelInitBytes =
          levelInitGatherResult.normalizedCount > 0
            ? levelInitGatherResult.bytes
            : afterPropGatherNdBytes;

        const argMinKeepDimsNormalized =
          Boolean(msg.normalizeArgMinKeepDims) && allowGraphRewrites && isHitnetVariant;
        const argMinResult = argMinKeepDimsNormalized
          ? normalizeHitnetArgMinKeepDimsForWebGpu(afterLevelInitBytes, {
              variantId: msg.profile.variantId,
              backendPreference: backendPref,
              requireHitnetContract: true,
              allowDiagnosticTransformedWasm: diagnosticTransformedWasm,
            })
          : {
              bytes: afterLevelInitBytes,
              normalizedCount: 0,
              normalizedNodeNames: [],
              untouchedNodeNames: [],
              totalArgMinCount: 0,
              skippedDynamicCount: 0,
              skippedInvalidOrUnprovenCount: 0,
              updatedValueInfoCount: 0,
              ortCacheKeyBefore: null,
              ortCacheKeyAfter: null,
            };
        const afterArgMinBytes =
          argMinResult.normalizedCount > 0 ? argMinResult.bytes : afterLevelInitBytes;

        const m1ZeroSlantRequested =
          Boolean(msg.normalizeInitToPropZeroSlant) && allowGraphRewrites && isHitnetVariant;
        if (
          m1ZeroSlantRequested &&
          (levelInitGatherAppliedMode !== 'off' ||
            argMinKeepDimsNormalized ||
            !propGatherNdPreconditionsMet ||
            propGatherNdResult.normalizedCount !== 2)
        ) {
          throw new Error(
            `M1-ConstZero2Ch is restricted strictly to H1 (never L1 or G1): levelInitMode=${levelInitGatherAppliedMode}, argMin=${argMinKeepDimsNormalized}, h1Count=${propGatherNdResult.normalizedCount}`
          );
        }
        const h1SourceSha256 = m1ZeroSlantRequested
          ? await computeSha256Hex(afterArgMinBytes)
          : undefined;
        const m1ZeroSlantResult = m1ZeroSlantRequested
          ? normalizeHitnetInitToPropZeroSlantForWebGpu(afterArgMinBytes, {
              variantId: msg.profile.variantId,
              backendPreference: backendPref,
              requireHitnetContract: true,
              allowDiagnosticTransformedWasm: diagnosticTransformedWasm,
              rawModelSha256: rawInputModelSha256,
              h1ModelSha256: h1SourceSha256,
            })
          : {
              bytes: afterArgMinBytes,
              normalizedCount: 0,
              normalizedNodeNames: [],
              removedDeadNodeNames: [],
              removedDeadInitializerNames: [],
              addedInitializerName: null,
              addedInitializerDims: null,
              verifiedH1Provenance: false,
              verifiedRawIdentity: false,
              skippedInvalidOrUnprovenCount: 0,
            };
        if (
          m1ZeroSlantRequested &&
          (msg.strictWebGpu || diagnosticTransformedWasm) &&
          m1ZeroSlantResult.normalizedCount !== 2
        ) {
          throw new Error(
            `M1-ConstZero2Ch fail-closed: normalizedCount=${m1ZeroSlantResult.normalizedCount}, verifiedRaw=${rawInputModelSha256 === HITNET_RAW_MODEL_SHA256}, verifiedH1=${h1SourceSha256 === HITNET_H1_FULL_GRAPH_SHA256}, skippedInvalid=${m1ZeroSlantResult.skippedInvalidOrUnprovenCount}`
          );
        }
        const afterM1Bytes =
          m1ZeroSlantResult.normalizedCount > 0 ? m1ZeroSlantResult.bytes : afterArgMinBytes;

        const n1Slot0Requested =
          Boolean(msg.normalizeSlot0ReshapeToTranspose) && allowGraphRewrites && isHitnetVariant;
        if (
          n1Slot0Requested &&
          (!m1ZeroSlantRequested ||
            m1ZeroSlantResult.normalizedCount !== 2 ||
            levelInitGatherAppliedMode !== 'off' ||
            argMinKeepDimsNormalized ||
            !propGatherNdPreconditionsMet ||
            propGatherNdResult.normalizedCount !== 2)
        ) {
          throw new Error(
            `N1-Slot0Transpose is restricted strictly to exact M1: m1Req=${m1ZeroSlantRequested}, m1Count=${m1ZeroSlantResult.normalizedCount}, levelInitMode=${levelInitGatherAppliedMode}, argMin=${argMinKeepDimsNormalized}, h1Count=${propGatherNdResult.normalizedCount}`
          );
        }
        const m1SourceSha256 = n1Slot0Requested
          ? await computeSha256Hex(afterM1Bytes)
          : undefined;
        const n1Slot0Result = n1Slot0Requested
          ? normalizeHitnetSlot0ReshapeToTransposeForWebGpu(afterM1Bytes, {
              variantId: msg.profile.variantId,
              backendPreference: backendPref,
              requireHitnetContract: true,
              allowDiagnosticTransformedWasm: diagnosticTransformedWasm,
              rawModelSha256: rawInputModelSha256,
              m1ModelSha256: m1SourceSha256,
            })
          : {
              bytes: afterM1Bytes,
              normalizedCount: 0,
              normalizedNodeNames: [],
              verifiedM1Provenance: false,
              verifiedRawIdentity: false,
              skippedInvalidOrUnprovenCount: 0,
            };
        if (
          n1Slot0Requested &&
          (msg.strictWebGpu || diagnosticTransformedWasm) &&
          n1Slot0Result.normalizedCount !== 2
        ) {
          throw new Error(
            `N1-Slot0Transpose fail-closed: normalizedCount=${n1Slot0Result.normalizedCount}, verifiedRaw=${rawInputModelSha256 === HITNET_RAW_MODEL_SHA256}, verifiedM1=${m1SourceSha256 === HITNET_M1_FULL_GRAPH_SHA256}, skippedInvalid=${n1Slot0Result.skippedInvalidOrUnprovenCount}`
          );
        }
        const preparedBytes =
          n1Slot0Result.normalizedCount > 0 ? n1Slot0Result.bytes : afterM1Bytes;
        const baseSessionModelSha256 = await computeSha256Hex(preparedBytes);

        const requestedTapOutputs = Array.isArray(msg.diagnosticTapOutputs)
          ? msg.diagnosticTapOutputs.filter((s): s is string => typeof s === 'string' && s.length > 0)
          : [];
        if (requestedTapOutputs.length > 0) {
          if (!isHitnetVariant || (!strictWebGpu && !diagnosticTransformedWasm)) {
            throw new Error(
              'diagnosticTapOutputs is restricted to diagnostic hitnet-sf strictWebGpu or allowDiagnosticTransformedWasm runs'
            );
          }
        }
        const tapResult =
          requestedTapOutputs.length > 0
            ? appendOnnxGraphOutputs(preparedBytes, requestedTapOutputs, { maxExtraOutputs: 2 })
            : null;
        const sessionInputBytes = tapResult ? tapResult.bytes : preparedBytes;
        const tPrepEnd = performance.now();

        const delegate = diagnosticTransformedWasm
          ? 'WASM'
          : resolveStereoOrtDelegate(
              msg.profile,
              sessionInputBytes,
              DEFAULT_WEBGPU_MAX_STORAGE_BUFFERS_PER_STAGE,
              strictWebGpu ? 'webgpu' : backendPref
            );
        const enableWebGpuProfiling = Boolean(msg.enableWebGpuProfiling);
        const disableCpuEpFallback =
          !diagnosticTransformedWasm && Boolean(msg.disableCpuEpFallback);
        const effectiveOptLevel: OrtGraphOptimizationLevel =
          msg.graphOptimizationLevel ?? (delegate === 'WASM' ? 'disabled' : 'all');
        const sessionOptionOverrides: StereoOrtSessionOptionOverrides = {
          graphOptimizationLevel: msg.graphOptimizationLevel,
          disableCpuEpFallback,
          enableProfiling: enableWebGpuProfiling,
          logSeverityLevel: enableWebGpuProfiling ? 0 : 3,
        };
        configureWebGpuProfiling(enableWebGpuProfiling);

        if (skipImplicitConcatSplit) {
          const preSessionModelSha256 = await computeSha256Hex(sessionInputBytes);
          if (
            sessionInputBytes.byteLength !== HITNET_M1_FULL_GRAPH_BYTE_LENGTH ||
            preSessionModelSha256 !== HITNET_M1_FULL_GRAPH_SHA256 ||
            baseSessionModelSha256 !== HITNET_M1_FULL_GRAPH_SHA256 ||
            rawInputModelSha256 !== HITNET_M1_FULL_GRAPH_SHA256
          ) {
            throw new Error(
              `[model-guard] Pre-session exact M1 verification failed after Concat-split skip: len=${sessionInputBytes.byteLength} (expected ${HITNET_M1_FULL_GRAPH_BYTE_LENGTH}), preSessionSha=${preSessionModelSha256}, baseSha=${baseSessionModelSha256}, expected=${HITNET_M1_FULL_GRAPH_SHA256}`
            );
          }
        }

        let created: Awaited<ReturnType<StereoWorkerSessionCreator>>;
        let transformFallbackReason: string | null = null;
        let sessionModelBytesUsed = sessionInputBytes;
        const tSessionStart = performance.now();
        try {
          created = await resolvedSessionCreator(
            sessionInputBytes,
            delegate,
            msg.wasmPrimaryUrl,
            msg.wasmFallbackUrl,
            strictWebGpu && !diagnosticTransformedWasm,
            sessionOptionOverrides
          );
        } catch (primaryInitErr) {
          if (
            !strictWebGpu &&
            !diagnosticTransformedWasm &&
            !tapResult &&
            (conv3dResult.decomposedCount > 0 ||
              sliceNormResult.normalizedCount > 0 ||
              gatherNdResult.normalizedCount > 0 ||
              propGatherNdResult.normalizedCount > 0 ||
              levelInitGatherResult.normalizedCount > 0 ||
              m1ZeroSlantResult.normalizedCount > 0 ||
              argMinResult.normalizedCount > 0) &&
            preparedBytes !== afterSplitBytes
          ) {
            const errMsg =
              primaryInitErr instanceof Error ? primaryInitErr.message : String(primaryInitErr);
            sessionModelBytesUsed = afterSplitBytes;
            created = await resolvedSessionCreator(
              afterSplitBytes,
              'WASM',
              msg.wasmPrimaryUrl,
              msg.wasmFallbackUrl,
              false
            );
            transformFallbackReason = `WebGPU互換グラフ変換後セッション初期化失敗 (${errMsg}); 未変換ONNXでWASMへ安全フォールバック`;
          } else {
            throw primaryInitErr;
          }
        }
        const tSessionEnd = performance.now();
        const sessionModelSha256 = await computeSha256Hex(sessionModelBytesUsed);
        const staticCompatibility =
          backendPref === 'auto' && delegate === 'WASM' && msg.profile.preferWebGpu
            ? inspectStereoOnnxWebGpuCompatibility(sessionInputBytes)
            : null;
        const requestedProvider: OnnxBackendInfo['requestedProvider'] =
          backendPref === 'wasm'
            ? 'wasm'
            : backendPref === 'webgpu' || strictWebGpu
              ? 'webgpu'
              : delegate === 'WebGPU'
                ? 'webgpu'
                : 'auto';
        const transformNotes: string[] = [];
        if (splitResult.splitCount > 0) {
          transformNotes.push(
            `Concat分割(${splitResult.splitCount}箇所・最大${splitResult.originalMaxConcatInputs}→4入力ツリー化)`
          );
        }
        if (conv3dResult.decomposedCount > 0 && !transformFallbackReason) {
          transformNotes.push(
            `5D Conv3D分解(${conv3dResult.decomposedCount}箇所→2D Conv化)`
          );
        }
        if (sliceNormResult.normalizedCount > 0 && !transformFallbackReason) {
          transformNotes.push(
            `Slice軸正規化[${sliceNormalizeMode}](${sliceNormResult.normalizedCount}箇所)`
          );
        }
        if (gatherNdResult.normalizedCount > 0 && !transformFallbackReason) {
          transformNotes.push(
            `GatherNDランク正規化(${gatherNdResult.normalizedCount}箇所)`
          );
        }
        if (propGatherNdResult.normalizedCount > 0 && !transformFallbackReason) {
          transformNotes.push(
            `H1 prop0 GatherND正規化(${propGatherNdResult.normalizedCount}箇所)`
          );
        }
        if (levelInitGatherResult.normalizedCount > 0 && !transformFallbackReason) {
          transformNotes.push(
            `level_init ${levelInitGatherResult.mode}(${levelInitGatherResult.normalizedCount}箇所)`
          );
        }
        if (m1ZeroSlantResult.normalizedCount > 0 && !transformFallbackReason) {
          transformNotes.push(
            `M1 ConstZero2Ch(${m1ZeroSlantResult.normalizedCount}箇所)`
          );
        }
        if (argMinResult.normalizedCount > 0 && !transformFallbackReason) {
          transformNotes.push(
            `ArgMin keepdims=1分離(${argMinResult.normalizedCount}箇所)`
          );
        }
        const graphTransformNote =
          transformNotes.length > 0 && created.backend.configuredProviders.includes('webgpu')
            ? `${transformNotes.join('＋')}でWebGPU有効化`
            : null;
        const adjustedBackend: OnnxBackendInfo = {
          ...created.backend,
          requestedProvider,
          fallbackReason:
            transformFallbackReason ??
            created.backend.fallbackReason ??
            staticCompatibility?.reason ??
            graphTransformNote,
        };
        let workerGpuAdapter: string | null = null;
        if (created.backend.configuredProviders.includes('webgpu')) {
          try {
            const ortEnv = getOrt().env;
            let info = ortEnv.webgpu?.adapter?.info;
            let features = ortEnv.webgpu?.adapter?.features;
            if ((!info || !features) && typeof navigator !== 'undefined') {
              const gpu = (navigator as unknown as {
                gpu?: {
                  requestAdapter?: () => Promise<{
                    info?: { vendor?: string; architecture?: string; device?: string; description?: string };
                    features?: { has?: (name: string) => boolean };
                  } | null>;
                };
              }).gpu;
              const ad = await gpu?.requestAdapter?.();
              info = info ?? ad?.info;
              features = features ?? ad?.features;
            }
            if (info) {
              const hasTq = features?.has?.('timestamp-query') ?? false;
              const hasF16 = features?.has?.('shader-f16') ?? false;
              workerGpuAdapter = `vendor=${info.vendor || '-'} arch=${info.architecture || '-'} device=${info.device || '-'} desc=${info.description || '-'} shader-f16=${hasF16} timestamp-query=${hasTq}`;
            }
          } catch {
            workerGpuAdapter = null;
          }
        }
        const ortVersions = getOrt().env.versions ?? null;
        const wasmPathsUsed =
          created.wasmSourceUsed === 'fallback' && msg.wasmFallbackUrl
            ? msg.wasmFallbackUrl
            : msg.wasmPrimaryUrl;

        activeProfile = msg.profile;
        activeSession = created.session;
        activeBackend = adjustedBackend;
        activeOnnxBytes =
          !strictWebGpu && adjustedBackend.configuredProviders.includes('webgpu')
            ? afterSplitBytes
            : null;
        activeWasmPrimaryUrl = msg.wasmPrimaryUrl;
        activeWasmFallbackUrl = msg.wasmFallbackUrl;
        activeStrictWebGpu = strictWebGpu;
        activeEnableWebGpuProfiling = enableWebGpuProfiling;
        attemptedWasmFallback = false;
        const { width: targetWidth, height: targetHeight } = resolveSessionTargetResolution(
          created.session,
          msg.profile
        );
        const tInitEnd = performance.now();
        postResponse({
          type: 'init-result',
          id: msg.id,
          ok: true,
          backend: adjustedBackend,
          wasmSourceUsed: created.wasmSourceUsed,
          targetWidth,
          targetHeight,
          workerGpuAdapter,
          ortVersions,
          ...(created.runtimeAssetFamily ? { runtimeAssetFamily: created.runtimeAssetFamily } : {}),
          ...(created.entryModuleUrl ? { entryModuleUrl: created.entryModuleUrl } : {}),
          ...(created.loadedOrtResources ? { loadedOrtResources: created.loadedOrtResources } : {}),
          ...(created.foreignOrtResources ? { foreignOrtResources: created.foreignOrtResources } : {}),
          ...(created.loadedAssetProvenance
            ? { loadedAssetProvenance: created.loadedAssetProvenance }
            : {}),
          rawInputModelSha256,
          baseSessionModelSha256,
          sessionModelSha256,
          sessionModelByteLength: sessionModelBytesUsed.byteLength,
          ...(tapResult ? { tappedOutputs: tapResult.addedOutputs } : {}),
          initTiming: {
            prepMs: tPrepEnd - tInitStart,
            sessionCreateMs: tSessionEnd - tSessionStart,
            totalInitMs: tInitEnd - tInitStart,
          },
          gatherNdStats: {
            normalizedCount: gatherNdResult.normalizedCount,
            normalizedNodeNames: gatherNdResult.normalizedNodeNames,
            skippedDynamicCount: gatherNdResult.skippedDynamicCount,
            skippedInvalidOrUnprovenCount: gatherNdResult.skippedInvalidOrUnprovenCount,
            totalGatherNdCount: gatherNdResult.totalGatherNdCount,
          },
          propGatherNdStats: {
            normalizedCount: propGatherNdResult.normalizedCount,
            normalizedNodeNames: propGatherNdResult.normalizedNodeNames,
            skippedDynamicCount: propGatherNdResult.skippedDynamicCount,
            skippedInvalidOrUnprovenCount: propGatherNdResult.skippedInvalidOrUnprovenCount,
            totalGatherNdCount: propGatherNdResult.totalGatherNdCount,
          },
          levelInitGatherStats: {
            mode: levelInitGatherResult.mode,
            normalizedCount: levelInitGatherResult.normalizedCount,
            normalizedNodeNames: levelInitGatherResult.normalizedNodeNames,
            skippedDynamicCount: levelInitGatherResult.skippedDynamicCount,
            skippedInvalidOrUnprovenCount: levelInitGatherResult.skippedInvalidOrUnprovenCount,
            totalGatherNdCount: levelInitGatherResult.totalGatherNdCount,
            verifiedCostLineageCount: levelInitGatherResult.verifiedCostLineageCount,
            documentedLimitations: levelInitGatherResult.documentedLimitations,
          },
          initToPropZeroSlantStats: {
            normalizedCount: m1ZeroSlantResult.normalizedCount,
            normalizedNodeNames: m1ZeroSlantResult.normalizedNodeNames,
            removedDeadNodeNames: m1ZeroSlantResult.removedDeadNodeNames,
            removedDeadInitializerNames: m1ZeroSlantResult.removedDeadInitializerNames,
            addedInitializerName: m1ZeroSlantResult.addedInitializerName,
            addedInitializerDims: m1ZeroSlantResult.addedInitializerDims,
            verifiedH1Provenance: m1ZeroSlantResult.verifiedH1Provenance,
            verifiedRawIdentity: m1ZeroSlantResult.verifiedRawIdentity,
            skippedInvalidOrUnprovenCount: m1ZeroSlantResult.skippedInvalidOrUnprovenCount,
          },
          slot0ReshapeToTransposeStats: {
            normalizedCount: n1Slot0Result.normalizedCount,
            normalizedNodeNames: n1Slot0Result.normalizedNodeNames,
            verifiedM1Provenance: n1Slot0Result.verifiedM1Provenance,
            verifiedRawIdentity: n1Slot0Result.verifiedRawIdentity,
            skippedInvalidOrUnprovenCount: n1Slot0Result.skippedInvalidOrUnprovenCount,
          },
          argMinStats: {
            normalizedCount: argMinResult.normalizedCount,
            normalizedNodeNames: argMinResult.normalizedNodeNames,
            untouchedNodeNames: argMinResult.untouchedNodeNames,
            totalArgMinCount: argMinResult.totalArgMinCount,
            skippedDynamicCount: argMinResult.skippedDynamicCount,
            skippedInvalidOrUnprovenCount: argMinResult.skippedInvalidOrUnprovenCount,
            updatedValueInfoCount: argMinResult.updatedValueInfoCount,
            ortCacheKeyBefore: argMinResult.ortCacheKeyBefore,
            ortCacheKeyAfter: argMinResult.ortCacheKeyAfter,
          },
          sessionOptionsApplied: {
            executionProviders: [...created.backend.configuredProviders],
            graphOptimizationLevel: effectiveOptLevel,
            disableCpuEpFallback:
              disableCpuEpFallback && created.backend.configuredProviders.includes('webgpu'),
            wasmPaths: wasmPathsUsed,
            sliceNormalizeMode,
            gatherNdRankNormalized: gatherNdResult.normalizedCount > 0,
            propGatherNdRankNormalized: propGatherNdResult.normalizedCount > 0,
            levelInitGatherMode: levelInitGatherAppliedMode,
            initToPropZeroSlantNormalized: m1ZeroSlantResult.normalizedCount > 0,
            slot0ReshapeToTransposeNormalized: n1Slot0Result.normalizedCount > 0,
            argMinKeepDimsNormalized: argMinResult.normalizedCount > 0,
            allowDiagnosticTransformedWasm: diagnosticTransformedWasm,
            pretransformedExactM1SkipConcatSplit: skipImplicitConcatSplit,
            ...(tapResult ? { diagnosticTapOutputs: tapResult.addedOutputs } : {}),
            enableWebGpuProfiling,
          },
          transformStats: {
            splitCount: splitResult.splitCount,
            decomposed5dCount: conv3dResult.decomposedCount,
            sliceNormalizedCount: sliceNormResult.normalizedCount,
            sliceCompactedCount: sliceNormResult.compactedCount,
            sliceSkippedDynamicCount: sliceNormResult.skippedDynamicCount,
            sliceSkippedInvalidOrUnprovenCount: sliceNormResult.skippedInvalidOrUnprovenCount,
          },
        });
      } catch (err) {
        await releaseActiveSession().catch(() => {});
        activeProfile = null;
        activeBackend = null;
        postResponse({
          type: 'error',
          id: msg.id,
          ok: false,
          stage: 'task',
          message: err instanceof Error ? err.message : String(err),
        });
      }
      return;
    }

    if (msg.type === 'estimate') {
      try {
        if (!activeProfile || !activeBackend) {
          throw new Error('Stereo Depth Worker is not initialized');
        }
        const leftRgba = new Uint8ClampedArray(msg.leftRgbaBuffer);
        const rightRgba = new Uint8ClampedArray(msg.rightRgbaBuffer);
        const width = msg.width;
        const height = msg.height;
        const sourceEyeW = Math.max(1, msg.sourceEyeWidth);
        const options = msg.options ?? {};

        if (activeProfile.engineType === 'sad-cpu' || activeProfile.engineType === 'webgpu-wgsl') {
          const userMaxDisp = Math.max(8, Math.min(256, options.maxDisparity ?? 64));
          const downscaleRatio = width / sourceEyeW;
          const minScaledDisp = sourceEyeW >= 1280 ? 24 : 6;
          const scaledMaxDisp = Math.max(
            minScaledDisp,
            Math.min(width - 2, Math.round(userMaxDisp * downscaleRatio))
          );

          let rawDisparity: Float32Array;
          if (activeProfile.engineType === 'webgpu-wgsl' && activeWgslEngine && !wgslRuntimeFailed) {
            try {
              const wgslOut = await activeWgslEngine.estimate(leftRgba, rightRgba, width, height, {
                maxDisparity: scaledMaxDisp,
                windowSize: options.sadWindowSize ?? 7,
                confidenceThreshold: options.confidenceThreshold ?? 0.35,
              });
              rawDisparity = wgslOut.disparity;
            } catch (wgslErr) {
              wgslRuntimeFailed = true;
              const errMsg = wgslErr instanceof Error ? wgslErr.message : String(wgslErr);
              activeBackend = {
                ...activeBackend,
                configuredProviders: ['cpu-sad'],
                fallbackReason: `WebGPU WGSL実行エラー (${errMsg}); CPU SAD へ自動フォールバック`,
              };
              const sadFallback = computeSadDisparity(leftRgba, rightRgba, width, height, {
                maxDisparity: scaledMaxDisp,
                windowSize: options.sadWindowSize ?? 7,
                confidenceThreshold: options.confidenceThreshold ?? 0.35,
              });
              rawDisparity = sadFallback.disparity;
            }
          } else {
            const sadResult = computeSadDisparity(leftRgba, rightRgba, width, height, {
              maxDisparity: scaledMaxDisp,
              windowSize: options.sadWindowSize ?? 7,
              confidenceThreshold: options.confidenceThreshold ?? 0.35,
            });
            rawDisparity = sadResult.disparity;
          }

          const dispScaleToSource = sourceEyeW / width;
          const sourceDisparity = new Float32Array(rawDisparity.length);
          for (let i = 0; i < rawDisparity.length; i++) {
            const v = rawDisparity[i];
            sourceDisparity[i] = v >= 0 ? v * dispScaleToSource : -1;
          }
          const autoNorm = options.autoNormalizeDisparity ?? true;
          const manualMin = options.minDisparityPx ?? 0;
          const manualMax = options.maxDisparityPx ?? 64;
          const colormap: StereoColormap = options.colormap ?? 'turbo';
          const stats = computeDisparityStats(sourceDisparity, autoNorm, manualMin, manualMax);
          const overlayRgba = renderDisparityColormapToRgba(
            sourceDisparity,
            width,
            height,
            stats.scaleMinDisparity,
            stats.scaleMaxDisparity,
            colormap
          );
          const dispBuf = sourceDisparity.buffer as ArrayBuffer;
          const ovBuf = overlayRgba.buffer as ArrayBuffer;
          postResponse(
            {
              type: 'estimate-result',
              id: msg.id,
              ok: true,
              width,
              height,
              disparityBuffer: dispBuf,
              overlayRgbaBuffer: ovBuf,
              stats,
              backend: activeBackend,
            },
            [dispBuf, ovBuf]
          );
          return;
        }

        if (!activeSession) {
          throw new Error('Stereo ONNX session is not initialized');
        }

        const ortRuntime = getOrt();
        const pixelCount = width * height;
        const inputNames = activeSession.inputNames;
        const feeds: Record<string, ort.Tensor> = {};
        const createdTensors: ort.Tensor[] = [];

        try {
          const inputMeta = (activeSession as unknown as {
            inputMetadata?: ReadonlyArray<{
              shape?: readonly (number | string)[];
              dimensions?: readonly (number | string)[];
            }>;
          }).inputMetadata;

          const tEstStart = performance.now();
          if (inputNames.length === 1) {
            const singleDims = inputMeta?.[0]?.shape ?? inputMeta?.[0]?.dimensions;
            const channels =
              typeof singleDims?.[1] === 'number' ? (singleDims[1] as number) : 6;
            const data = new Float32Array(channels * pixelCount);
            rgbaToChwInto(leftRgba, pixelCount, data, 0, activeProfile.normalization);
            if (channels >= 6) {
              rgbaToChwInto(
                rightRgba,
                pixelCount,
                data,
                pixelCount * 3,
                activeProfile.normalization
              );
            }
            const tensor = new ortRuntime.Tensor('float32', data, [1, channels, height, width]);
            feeds[inputNames[0]] = tensor;
            createdTensors.push(tensor);
          } else if (inputNames.length === 2) {
            const leftData = new Float32Array(3 * pixelCount);
            const rightData = new Float32Array(3 * pixelCount);
            rgbaToChwInto(leftRgba, pixelCount, leftData, 0, activeProfile.normalization);
            rgbaToChwInto(rightRgba, pixelCount, rightData, 0, activeProfile.normalization);
            const leftTensor = new ortRuntime.Tensor('float32', leftData, [1, 3, height, width]);
            const rightTensor = new ortRuntime.Tensor('float32', rightData, [1, 3, height, width]);
            const secondIsLeft =
              /left/i.test(inputNames[1]) && !/left/i.test(inputNames[0]);
            feeds[inputNames[0]] = secondIsLeft ? rightTensor : leftTensor;
            feeds[inputNames[1]] = secondIsLeft ? leftTensor : rightTensor;
            createdTensors.push(leftTensor, rightTensor);
          } else if (inputNames.length === 4) {
            const halfW = Math.max(16, Math.floor(width / 2));
            const halfH = Math.max(16, Math.floor(height / 2));
            const initLRgba = downscaleRgbaBuffer(leftRgba, width, height, halfW, halfH);
            const initRRgba = downscaleRgbaBuffer(rightRgba, width, height, halfW, halfH);
            const initPixels = halfW * halfH;
            const initL = new Float32Array(3 * initPixels);
            const initR = new Float32Array(3 * initPixels);
            rgbaToChwInto(initLRgba, initPixels, initL, 0, activeProfile.normalization);
            rgbaToChwInto(initRRgba, initPixels, initR, 0, activeProfile.normalization);

            const nextL = new Float32Array(3 * pixelCount);
            const nextR = new Float32Array(3 * pixelCount);
            rgbaToChwInto(leftRgba, pixelCount, nextL, 0, activeProfile.normalization);
            rgbaToChwInto(rightRgba, pixelCount, nextR, 0, activeProfile.normalization);

            const tInitL = new ortRuntime.Tensor('float32', initL, [1, 3, halfH, halfW]);
            const tInitR = new ortRuntime.Tensor('float32', initR, [1, 3, halfH, halfW]);
            const tNextL = new ortRuntime.Tensor('float32', nextL, [1, 3, height, width]);
            const tNextR = new ortRuntime.Tensor('float32', nextR, [1, 3, height, width]);
            createdTensors.push(tInitL, tInitR, tNextL, tNextR);

            for (let i = 0; i < inputNames.length; i++) {
              const nLower = inputNames[i].toLowerCase();
              if (nLower.includes('init') && nLower.includes('left')) feeds[inputNames[i]] = tInitL;
              else if (nLower.includes('init') && nLower.includes('right')) feeds[inputNames[i]] = tInitR;
              else if (nLower.includes('left')) feeds[inputNames[i]] = tNextL;
              else if (nLower.includes('right')) feeds[inputNames[i]] = tNextR;
              else if (i === 0) feeds[inputNames[i]] = tInitL;
              else if (i === 1) feeds[inputNames[i]] = tInitR;
              else if (i === 2) feeds[inputNames[i]] = tNextL;
              else feeds[inputNames[i]] = tNextR;
            }
          } else {
            throw new Error(
              `Unsupported ONNX stereo input count (${inputNames.length}): [${inputNames.join(', ')}]`
            );
          }

          const tPreprocessEnd = performance.now();
          let feedSha256s: Record<string, string> | undefined;
          let feedMetadata: Record<string, { dims: number[]; dataType: string }> | undefined;
          if (msg.includeRawTensorOutputs) {
            feedSha256s = {};
            feedMetadata = {};
            for (const [feedName, feedTensor] of Object.entries(feeds)) {
              const td = feedTensor.data as ArrayBufferView;
              const view = new Uint8Array(td.buffer, td.byteOffset, td.byteLength);
              feedSha256s[feedName] = await computeSha256Hex(view);
              feedMetadata[feedName] = {
                dims: Array.from(feedTensor.dims),
                dataType: feedTensor.type,
              };
            }
          }
          const fetchesArg =
            msg.fetchPrimaryOutputOnly && activeSession.outputNames.length > 0
              ? [activeSession.outputNames[0]]
              : undefined;
          let results: Record<string, ort.Tensor>;
          const gpuValidationErrors: string[] = [];
          const activeWebGpuEnv = ortRuntime.env.webgpu;
          const gpuDevice =
            (msg.captureGpuValidationErrors || activeEnableWebGpuProfiling) &&
            activeBackend.configuredProviders.includes('webgpu')
              ? activeWebGpuEnv?.device
              : undefined;
          const validationGpuDevice = msg.captureGpuValidationErrors ? gpuDevice : undefined;
          const dispatchObserver =
            activeEnableWebGpuProfiling && activeBackend.configuredProviders.includes('webgpu')
              ? attachWebGpuDeviceDispatchObserver(gpuDevice)
              : null;
          const onUncaptured = (ev: { error?: { message?: string } }) => {
            const m = ev?.error?.message;
            if (m) gpuValidationErrors.push(`uncaptured: ${m}`);
          };
          let pushedErrorScope = false;
          if (validationGpuDevice) {
            try {
              validationGpuDevice.addEventListener?.('uncapturederror', onUncaptured);
              validationGpuDevice.pushErrorScope?.('validation');
              pushedErrorScope = typeof validationGpuDevice.popErrorScope === 'function';
            } catch {
              pushedErrorScope = false;
            }
          }
          gpuProfileEvents = [];
          const tRunStart = performance.now();
          try {
            try {
              results = fetchesArg
                ? await activeSession.run(feeds, fetchesArg)
                : await activeSession.run(feeds);
              // Once the first frame succeeds on WebGPU, free the fallback ONNX byte buffer (10-25MB)
              activeOnnxBytes = null;
            } catch (runError) {
              if (
                !activeStrictWebGpu &&
                !attemptedWasmFallback &&
                activeOnnxBytes &&
                activeBackend.configuredProviders.includes('webgpu') &&
                isWebGpuRuntimeKernelError(runError)
              ) {
                attemptedWasmFallback = true;
                const failedSession = activeSession;
                activeSession = null;
                try {
                  await failedSession.release();
                } catch {
                  // Ignore cleanup error on failed WebGPU session
                }
                const recovered = await resolvedSessionCreator(
                  activeOnnxBytes,
                  'WASM',
                  activeWasmPrimaryUrl,
                  activeWasmFallbackUrl,
                  false
                );
                activeSession = recovered.session;
                activeOnnxBytes = null;
                const errMsg = runError instanceof Error ? runError.message : String(runError);
                activeBackend = {
                  ...recovered.backend,
                  requestedProvider: activeBackend.requestedProvider,
                  fallbackReason: `WebGPU kernel runtime failure (${errMsg}); fell back to WASM`,
                };
                const fallbackFetches =
                  msg.fetchPrimaryOutputOnly && activeSession.outputNames.length > 0
                    ? [activeSession.outputNames[0]]
                    : undefined;
                results = fallbackFetches
                  ? await activeSession.run(feeds, fallbackFetches)
                  : await activeSession.run(feeds);
              } else {
                throw runError;
              }
            }
          } finally {
            dispatchObserver?.detach();
            if (pushedErrorScope && validationGpuDevice?.popErrorScope) {
              try {
                const scopedErr = await validationGpuDevice.popErrorScope();
                if (scopedErr?.message) {
                  gpuValidationErrors.push(`validation-scope: ${scopedErr.message}`);
                }
              } catch {
                // Ignore popErrorScope error
              }
            }
            if (validationGpuDevice) {
              try {
                validationGpuDevice.removeEventListener?.('uncapturederror', onUncaptured);
              } catch {
                // Ignore listener cleanup
              }
            }
          }
          const tRunEnd = performance.now();

          let gpuKernelProfileSummary: GpuKernelProfileSummary | null = null;
          if (activeEnableWebGpuProfiling) {
            const tqSupported = Boolean(activeWebGpuEnv?.adapter?.features?.has?.('timestamp-query'));
            const kernelTypes: Record<string, number> = {};
            const samplePrograms: string[] = [];
            for (const ev of gpuProfileEvents) {
              kernelTypes[ev.kernelType] = (kernelTypes[ev.kernelType] ?? 0) + 1;
              if (samplePrograms.length < 10 && ev.programName) {
                samplePrograms.push(ev.programName);
              }
            }
            const c = dispatchObserver?.counters;
            const placementDisclosure =
              'ORT Web does not expose per-node EP placement in JS; GPU pass/submit counts prove WebGPU kernel dispatches occurred, not that every node avoided ORT-internal CPU fallback.';
            gpuKernelProfileSummary = {
              eventCount: gpuProfileEvents.length,
              kernelTypes,
              samplePrograms,
              timestampQuerySupported: tqSupported,
              ...(c
                ? {
                    webgpuComputePassCount: c.computePassCount,
                    webgpuDispatchWorkgroupsCount: c.dispatchWorkgroupsCount,
                    webgpuDispatchIndirectCount: c.dispatchIndirectCount,
                    webgpuTimestampWritePassCount: c.timestampWritePassCount,
                    webgpuQueueSubmitCount: c.queueSubmitCount,
                    webgpuCommandBufferCount: c.commandBufferCount,
                    webgpuQuerySetCount: c.querySetCount,
                    webgpuSamplePassLabels: c.samplePassLabels,
                    deviceObserverAttached: c.attached,
                  }
                : {}),
              note: c
                ? `Observed WebGPU GPU dispatch (${c.computePassCount} compute passes, ${c.dispatchWorkgroupsCount} dispatchWorkgroups, ${c.queueSubmitCount} queue.submit, ${gpuProfileEvents.length} ORT profiling events; timestamp-query=${tqSupported}). ${placementDisclosure}`
                : tqSupported
                  ? `timestamp-query enabled: WebGPU kernel ondata callbacks captured. ${placementDisclosure}`
                  : `timestamp-query unavailable on WebGPU adapter: ORT WebGPU profiling ondata does not fire without timestamp-query. ${placementDisclosure}`,
            };
          }

          const outputValues = Object.values(results);
          for (const t of outputValues) {
            createdTensors.push(t);
          }
          const fetchedOutputNames = Object.keys(results);
          const rawOutputs: StereoRawTensorOutput[] | undefined = msg.includeRawTensorOutputs
            ? Object.entries(results).map(([outName, outTensor]) => {
                const isBigInt64 = outTensor.data instanceof BigInt64Array;
                const rawCopy = isBigInt64
                  ? Float32Array.from(outTensor.data as BigInt64Array, (v) => Number(v))
                  : new Float32Array(outTensor.data as ArrayLike<number>);
                const exactInt64Copy = isBigInt64
                  ? new BigInt64Array(outTensor.data as BigInt64Array)
                  : null;
                return {
                  name: outName,
                  dims: Array.from(outTensor.dims),
                  dataType: outTensor.type,
                  buffer: rawCopy.buffer as ArrayBuffer,
                  ...(exactInt64Copy
                    ? { exactInt64Buffer: exactInt64Copy.buffer as ArrayBuffer }
                    : {}),
                };
              })
            : undefined;
          const primaryOutput =
            results[activeSession.outputNames[0]] ?? outputValues[0];
          if (!primaryOutput) {
            throw new Error('ONNX session returned no output tensor');
          }

          const extracted = extractDisparityFromOrtTensor(primaryOutput);
          const scaleToSource = sourceEyeW / extracted.width;
          const scaledDisparity = new Float32Array(extracted.disparity.length);
          for (let i = 0; i < extracted.disparity.length; i++) {
            const v = extracted.disparity[i];
            scaledDisparity[i] = v >= 0 ? v * scaleToSource : -1;
          }

          const autoNorm = options.autoNormalizeDisparity ?? true;
          const manualMin = options.minDisparityPx ?? 0;
          const manualMax = options.maxDisparityPx ?? 64;
          const colormap: StereoColormap = options.colormap ?? 'turbo';
          const stats = computeDisparityStats(scaledDisparity, autoNorm, manualMin, manualMax);
          const overlayRgba = renderDisparityColormapToRgba(
            scaledDisparity,
            extracted.width,
            extracted.height,
            stats.scaleMinDisparity,
            stats.scaleMaxDisparity,
            colormap
          );
          const tEstEnd = performance.now();

          const scaledBuf = scaledDisparity.buffer as ArrayBuffer;
          const ovBuf = overlayRgba.buffer as ArrayBuffer;
          const transfers: Transferable[] = [scaledBuf, ovBuf];
          if (rawOutputs) {
            for (const ro of rawOutputs) {
              transfers.push(ro.buffer);
              if (ro.exactInt64Buffer) {
                transfers.push(ro.exactInt64Buffer);
              }
            }
          }
          postResponse(
            {
              type: 'estimate-result',
              id: msg.id,
              ok: true,
              width: extracted.width,
              height: extracted.height,
              disparityBuffer: scaledBuf,
              overlayRgbaBuffer: ovBuf,
              stats,
              backend: activeBackend,
              gpuKernelProfileSummary,
              ...(msg.captureGpuValidationErrors ? { gpuValidationErrors } : {}),
              estimateTiming: {
                preprocessMs: tPreprocessEnd - tEstStart,
                runMs: tRunEnd - tRunStart,
                postprocessMs: tEstEnd - tRunEnd,
                totalEstimateMs: tEstEnd - tEstStart,
              },
              feedSha256s,
              feedMetadata,
              fetchedOutputNames,
              rawOutputs,
            },
            transfers
          );
        } finally {
          disposeOrtTensors(createdTensors);
        }
      } catch (err) {
        postResponse({
          type: 'error',
          id: msg.id,
          ok: false,
          stage: 'run',
          message: err instanceof Error ? err.message : String(err),
        });
      }
      return;
    }

    if (msg.type === 'dispose') {
      try {
        await releaseActiveSession();
        activeProfile = null;
        activeBackend = null;
      } finally {
        postResponse({
          type: 'dispose-result',
          id: msg.id,
          ok: true,
        });
      }
    }
  };

  return {
    handleMessage(
      msg: StereoWorkerRequest,
      postResponse: (response: StereoWorkerResponse, transfer?: Transferable[]) => void
    ): Promise<void> {
      const run = messageChain.then(
        () => processMessage(msg, postResponse),
        () => processMessage(msg, postResponse)
      );
      messageChain = run.then(
        () => {},
        () => {}
      );
      return run;
    },
  };
}

