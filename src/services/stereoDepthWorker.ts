import * as ort from 'onnxruntime-web';
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

export function resolveStereoOrtSessionOptions(
  executionProviders: string[]
): ort.InferenceSession.SessionOptions {
  const isWasmOnly =
    executionProviders.length > 0 && executionProviders.every((ep) => ep === 'wasm');
  // ORT's basic/all graph optimization performs ConstantFolding at session creation time, which
  // expands MobileStereoNet-2D disparity/correlation subgraphs to >2.3 GB WASM heap and crashes
  // browser tabs on load. Disabling graph optimization for WASM reduces peak create memory by >13x
  // and speeds up session creation across MobileStereoNet-2D, HITNet, and CREStereo.
  return {
    executionProviders,
    graphOptimizationLevel: isWasmOnly ? 'disabled' : 'all',
    logSeverityLevel: 3,
  };
}

async function createWorkerOrtSessionWithProviders(
  onnxBytes: Uint8Array,
  delegate: 'WebGPU' | 'WASM'
): Promise<{ session: ort.InferenceSession; backend: OnnxBackendInfo }> {
  const requestedProvider = delegate === 'WASM' ? 'wasm' : 'webgpu';
  const available = typeof navigator !== 'undefined' && Boolean((navigator as { gpu?: unknown }).gpu);
  const providers = delegate === 'WASM' || !available ? ['wasm'] : ['webgpu', 'wasm'];
  const backend: OnnxBackendInfo = {
    runtime: 'onnxruntime-web',
    requestedProvider,
    configuredProviders: providers,
    fallbackReason:
      delegate !== 'WASM' && !available ? 'WebGPU API unavailable; WASM configured' : null,
    actualBackend: 'unmeasured',
  };
  const create = (executionProviders: string[]) =>
    ort.InferenceSession.create(onnxBytes, resolveStereoOrtSessionOptions(executionProviders));
  try {
    return { session: await create(providers), backend };
  } catch (cause) {
    if (!providers.includes('webgpu')) {
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

async function createWorkerOrtSession(
  onnxBytes: Uint8Array,
  delegate: 'WebGPU' | 'WASM',
  wasmPrimaryUrl: string,
  wasmFallbackUrl: string | null
): Promise<{
  session: ort.InferenceSession;
  backend: OnnxBackendInfo;
  wasmSourceUsed: 'primary' | 'fallback';
}> {
  ort.env.logLevel = 'error';
  // Running inside a dedicated Worker thread already isolates ONNX execution from the UI thread;
  // disable nested pthread sub-workers so Vite's worker bundle never tries to spawn itself via import.meta.url.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  if (wasmPrimaryUrl) {
    ort.env.wasm.wasmPaths = wasmPrimaryUrl;
  }

  try {
    const created = await createWorkerOrtSessionWithProviders(onnxBytes, delegate);
    return { ...created, wasmSourceUsed: 'primary' };
  } catch (primaryErr) {
    if (wasmFallbackUrl && wasmFallbackUrl !== wasmPrimaryUrl) {
      ort.env.wasm.wasmPaths = wasmFallbackUrl;
      const created = await createWorkerOrtSessionWithProviders(onnxBytes, delegate);
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
    wasmFallbackUrl: string | null
  ): Promise<{
    session: ort.InferenceSession;
    backend: OnnxBackendInfo;
    wasmSourceUsed: 'primary' | 'fallback';
  }>;
}

export function createStereoWorkerProcessor(
  sessionCreator: StereoWorkerSessionCreator = createWorkerOrtSession,
  wgslEngineFactory: () => Promise<WebGpuStereoComputeEngine> = createWebGpuStereoComputeEngine
) {
  let activeProfile: StereoModelProfile | null = null;
  let activeSession: ort.InferenceSession | null = null;
  let activeWgslEngine: WebGpuStereoComputeEngine | null = null;
  let activeBackend: OnnxBackendInfo | null = null;
  let activeOnnxBytes: Uint8Array | null = null;
  let activeWasmPrimaryUrl = '';
  let activeWasmFallbackUrl: string | null = null;
  let attemptedWasmFallback = false;
  let wgslRuntimeFailed = false;

  const releaseActiveSession = async () => {
    const current = activeSession;
    const currentWgsl = activeWgslEngine;
    activeSession = null;
    activeWgslEngine = null;
    activeOnnxBytes = null;
    attemptedWasmFallback = false;
    wgslRuntimeFailed = false;
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
        await releaseActiveSession();
        const rawBytes = new Uint8Array(msg.onnxBuffer);
        const splitResult = splitLargeOnnxConcatNodes(rawBytes, 4);
        const preparedBytes =
          splitResult.splitCount > 0 ? splitResult.bytes : stripOnnxGraphValueInfo(rawBytes);
        const backendPref = msg.backendPreference ?? 'auto';
        const delegate = resolveStereoOrtDelegate(
          msg.profile,
          preparedBytes,
          DEFAULT_WEBGPU_MAX_STORAGE_BUFFERS_PER_STAGE,
          backendPref
        );
        const created = await sessionCreator(
          preparedBytes,
          delegate,
          msg.wasmPrimaryUrl,
          msg.wasmFallbackUrl
        );
        const staticCompatibility =
          backendPref === 'auto' && delegate === 'WASM' && msg.profile.preferWebGpu
            ? inspectStereoOnnxWebGpuCompatibility(preparedBytes)
            : null;
        const requestedProvider: OnnxBackendInfo['requestedProvider'] =
          backendPref === 'wasm'
            ? 'wasm'
            : backendPref === 'webgpu'
              ? 'webgpu'
              : delegate === 'WebGPU'
                ? 'webgpu'
                : 'auto';
        const concatNote =
          splitResult.splitCount > 0 && created.backend.configuredProviders.includes('webgpu')
            ? `Concat分割(${splitResult.splitCount}箇所・最大${splitResult.originalMaxConcatInputs}→4入力ツリー化)でWebGPU有効化`
            : null;
        const adjustedBackend: OnnxBackendInfo = {
          ...created.backend,
          requestedProvider,
          fallbackReason:
            created.backend.fallbackReason ??
            staticCompatibility?.reason ??
            concatNote,
        };
        activeProfile = msg.profile;
        activeSession = created.session;
        activeBackend = adjustedBackend;
        activeOnnxBytes = adjustedBackend.configuredProviders.includes('webgpu')
          ? preparedBytes
          : null;
        activeWasmPrimaryUrl = msg.wasmPrimaryUrl;
        activeWasmFallbackUrl = msg.wasmFallbackUrl;
        attemptedWasmFallback = false;
        const { width: targetWidth, height: targetHeight } = resolveSessionTargetResolution(
          created.session,
          msg.profile
        );
        postResponse({
          type: 'init-result',
          id: msg.id,
          ok: true,
          backend: adjustedBackend,
          wasmSourceUsed: created.wasmSourceUsed,
          targetWidth,
          targetHeight,
        });
      } catch (err) {
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
            const tensor = new ort.Tensor('float32', data, [1, channels, height, width]);
            feeds[inputNames[0]] = tensor;
            createdTensors.push(tensor);
          } else if (inputNames.length === 2) {
            const leftData = new Float32Array(3 * pixelCount);
            const rightData = new Float32Array(3 * pixelCount);
            rgbaToChwInto(leftRgba, pixelCount, leftData, 0, activeProfile.normalization);
            rgbaToChwInto(rightRgba, pixelCount, rightData, 0, activeProfile.normalization);
            const leftTensor = new ort.Tensor('float32', leftData, [1, 3, height, width]);
            const rightTensor = new ort.Tensor('float32', rightData, [1, 3, height, width]);
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

            const tInitL = new ort.Tensor('float32', initL, [1, 3, halfH, halfW]);
            const tInitR = new ort.Tensor('float32', initR, [1, 3, halfH, halfW]);
            const tNextL = new ort.Tensor('float32', nextL, [1, 3, height, width]);
            const tNextR = new ort.Tensor('float32', nextR, [1, 3, height, width]);
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

          let results: Record<string, ort.Tensor>;
          try {
            results = await activeSession.run(feeds);
            // Once the first frame succeeds on WebGPU, free the fallback ONNX byte buffer (10-25MB)
            activeOnnxBytes = null;
          } catch (runError) {
            if (
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
              const recovered = await sessionCreator(
                activeOnnxBytes,
                'WASM',
                activeWasmPrimaryUrl,
                activeWasmFallbackUrl
              );
              activeSession = recovered.session;
              activeOnnxBytes = null;
              const errMsg = runError instanceof Error ? runError.message : String(runError);
              activeBackend = {
                ...recovered.backend,
                requestedProvider: activeBackend.requestedProvider,
                fallbackReason: `WebGPU kernel runtime failure (${errMsg}); fell back to WASM`,
              };
              results = await activeSession.run(feeds);
            } else {
              throw runError;
            }
          }

          const outputValues = Object.values(results);
          for (const t of outputValues) {
            createdTensors.push(t);
          }
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

          const scaledBuf = scaledDisparity.buffer as ArrayBuffer;
          const ovBuf = overlayRgba.buffer as ArrayBuffer;
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
            },
            [scaledBuf, ovBuf]
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

const workerScope =
  typeof self !== 'undefined' &&
  typeof (self as unknown as { postMessage?: unknown }).postMessage === 'function' &&
  typeof window === 'undefined'
    ? (self as unknown as {
        onmessage: ((ev: MessageEvent<StereoWorkerRequest>) => void) | null;
        postMessage: (msg: StereoWorkerResponse, transfer?: Transferable[]) => void;
      })
    : null;

if (workerScope) {
  const processor = createStereoWorkerProcessor();
  workerScope.onmessage = (event: MessageEvent<StereoWorkerRequest>) => {
    void processor.handleMessage(event.data, (res, transfer) => {
      if (transfer && transfer.length > 0) {
        workerScope.postMessage(res, transfer);
      } else {
        workerScope.postMessage(res);
      }
    });
  };
}
