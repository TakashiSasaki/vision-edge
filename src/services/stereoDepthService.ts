import * as ort from 'onnxruntime-web';
import type {
  InferenceInput,
  OnnxBackendInfo,
  StereoBackendPreference,
  StereoColormap,
  StereoDepthInferenceParams,
  StereoDepthMapData,
  StereoDepthProbeResult,
  StereoDepthVariantId,
  StereoViewMode,
} from '../types/vision';
import {
  createOrtSession,
  disposeTensors,
  OnnxInferenceError,
  OnnxSessionOwner,
  unknownOnnxBackend,
} from './onnxSession';
import {
  resolveOrtWasmUrls,
  type AssetDeliverySource,
  type AssetSourceRecorder,
} from './modelAssetResolver';
import { fetchModelBytesWithBrowserCache, STEREO_CORS_PROXY_PREFIX } from './browserModelCache';
import {
  computeDisparityStats,
  computeSadDisparity,
  createStereoWorkerProcessor,
  createWebGpuStereoComputeEngine,
  DEFAULT_WEBGPU_MAX_STORAGE_BUFFERS_PER_STAGE,
  downscaleRgbaBuffer,
  exceedsWebGpuStorageBufferLimit,
  extractDisparityFromOrtTensor,
  getColormapLut,
  inspectStereoOnnxWebGpuCompatibility,
  isWebGpuRuntimeKernelError,
  renderDisparityColormapToRgba,
  resolveStereoOrtDelegate,
  resolveStereoOrtSessionOptions,
  rgbaToChwInto,
  splitLargeOnnxConcatNodes,
  STEREO_WGSL_SHADER,
  stripOnnxGraphValueInfo,
  type DisparityStatsResult,
  type OnnxConcatSplitResult,
  type SadComputeOptions,
  type SadDisparityOutput,
  type StereoInputLayout,
  type StereoModelProfile,
  type StereoNormalizationMode,
  type StereoOnlineDeliveryMode,
  type StereoWorkerEstimateOptions,
  type StereoWorkerRequest,
  type StereoWorkerResponse,
  type StereoWorkerSessionCreator,
  type WebGpuCompatibilityInspection,
  type WebGpuStereoComputeEngine,
  type WebGpuStorageBufferInspection,
} from './stereoDepthWorker';

export {
  computeDisparityStats,
  computeSadDisparity,
  createStereoWorkerProcessor,
  createWebGpuStereoComputeEngine,
  DEFAULT_WEBGPU_MAX_STORAGE_BUFFERS_PER_STAGE,
  downscaleRgbaBuffer,
  exceedsWebGpuStorageBufferLimit,
  extractDisparityFromOrtTensor,
  getColormapLut,
  inspectStereoOnnxWebGpuCompatibility,
  isWebGpuRuntimeKernelError,
  renderDisparityColormapToRgba,
  resolveStereoOrtDelegate,
  resolveStereoOrtSessionOptions,
  splitLargeOnnxConcatNodes,
  STEREO_WGSL_SHADER,
  stripOnnxGraphValueInfo,
  type DisparityStatsResult,
  type OnnxConcatSplitResult,
  type SadComputeOptions,
  type SadDisparityOutput,
  type StereoInputLayout,
  type StereoModelProfile,
  type StereoNormalizationMode,
  type StereoOnlineDeliveryMode,
  type StereoWorkerEstimateOptions,
  type StereoWorkerRequest,
  type StereoWorkerResponse,
  type StereoWorkerSessionCreator,
  type WebGpuCompatibilityInspection,
  type WebGpuStereoComputeEngine,
  type WebGpuStorageBufferInspection,
};

export const STEREO_MODEL_PROFILES: Record<StereoDepthVariantId, StereoModelProfile> = {
  'stereo-sad-baseline': {
    variantId: 'stereo-sad-baseline',
    tier: 'baseline',
    engineType: 'sad-cpu',
    fileName: 'builtin-stereo-sad',
    defaultWidth: 320,
    defaultHeight: 180,
    inputLayout: 'dual',
    normalization: 'zero-255',
    preferWebGpu: false,
    license: 'Built-in (MIT)',
    licenseNote: 'ブラウザ標準Web Workerでの高速スライディングウィンドウSAD視差計算（重み不要・即座に動作）',
    upstreamUrl: 'https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API',
    onlineDelivery: 'local-only',
  },
  'stereo-webgpu-wgsl': {
    variantId: 'stereo-webgpu-wgsl',
    tier: 'lightweight',
    engineType: 'webgpu-wgsl',
    fileName: 'builtin-stereo-webgpu-wgsl',
    defaultWidth: 480,
    defaultHeight: 270,
    inputLayout: 'dual',
    normalization: 'zero-255',
    preferWebGpu: true,
    license: 'Built-in (MIT)',
    licenseNote: 'ネイティブ WebGPU WGSL コンピュートシェーダ（5×5 Census＋勾配＋3×3メジアン · 非対応時はCPU自動フォールバック）',
    upstreamUrl: 'https://www.w3.org/TR/webgpu/',
    onlineDelivery: 'local-only',
  },
  'lightstereo-s': {
    variantId: 'lightstereo-s',
    tier: 'lightweight',
    engineType: 'onnx',
    fileName: 'lightstereo_s_sceneflow_general_opt_256_512.onnx',
    defaultWidth: 512,
    defaultHeight: 256,
    inputLayout: 'dual',
    normalization: 'imagenet',
    preferWebGpu: true,
    license: 'Apache-2.0',
    licenseNote: 'LightStereo-S (OpenStereo / lightstereo_cpp ONNX) / Apache-2.0（Concat自動分割によるWebGPU高速実行 + WASMフォールバック対応 · オンライン自動取得＋ローカル.onnx上書き対応）',
    upstreamUrl: 'https://github.com/zz990099/lightstereo_cpp',
    onlineDelivery: 'direct-cors',
    onlineSourceLabel: 'jsDelivr CDN / GitHub Raw',
    defaultModelUrl:
      'https://cdn.jsdelivr.net/gh/zz990099/lightstereo_cpp@master/models/lightstereo_s_sceneflow_general_opt_256_512.onnx',
    fallbackModelUrl:
      'https://raw.githubusercontent.com/zz990099/lightstereo_cpp/master/models/lightstereo_s_sceneflow_general_opt_256_512.onnx',
  },
  'mobilestereonet-2d': {
    variantId: 'mobilestereonet-2d',
    tier: 'lightweight',
    engineType: 'onnx',
    fileName: 'model_528_240_float32.onnx',
    defaultWidth: 528,
    defaultHeight: 240,
    inputLayout: 'dual',
    normalization: 'imagenet',
    preferWebGpu: false,
    license: 'MIT',
    licenseNote: 'MobileStereoNet-2D (Shamsafar et al. / ONNX-MobileStereoNet) / MIT（3D畳み込み集約ノードのため自動時はWASM構成 · 240×528 · 手動でWebGPU強制試行可能）',
    upstreamUrl: 'https://github.com/ibaiGorordo/ONNX-MobileStereoNet',
    onlineDelivery: 'cors-proxy',
    onlineSourceLabel: 'Google Drive (CORS中継 + ブラウザキャッシュ)',
    defaultModelUrl:
      'https://drive.usercontent.google.com/download?id=1Dkyrg5Fu554gqxfclkHC6zJBCYrdzOO0&export=download&confirm=t',
    corsProxyUrl: `${STEREO_CORS_PROXY_PREFIX}model_528_240_float32.onnx`,
  },
  'hitnet-sf': {
    variantId: 'hitnet-sf',
    tier: 'lightweight',
    engineType: 'onnx',
    fileName: 'hitnet_sceneflow_240x320.onnx',
    defaultWidth: 320,
    defaultHeight: 240,
    inputLayout: 'stacked',
    normalization: 'zero-one',
    preferWebGpu: true,
    license: 'Apache-2.0',
    licenseNote: 'HITNet SceneFlow (Tankovich et al. / PINTO0309 変換 ONNX) / Apache-2.0（WebGPU優先 + WASMフォールバック · [1,6,240,320]連結入力）',
    upstreamUrl: 'https://github.com/nburrus/stereodemo/releases/tag/v0.1-hitnet',
    onlineDelivery: 'cors-proxy',
    onlineSourceLabel: 'GitHub Releases (CORS中継 + ブラウザキャッシュ)',
    defaultModelUrl:
      'https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_sceneflow_240x320.onnx',
    corsProxyUrl: `${STEREO_CORS_PROXY_PREFIX}hitnet_sceneflow_240x320.onnx`,
  },
  'crestereo-init-iter2': {
    variantId: 'crestereo-init-iter2',
    tier: 'high-accuracy',
    engineType: 'onnx',
    fileName: 'crestereo_init_iter2_180x320.onnx',
    defaultWidth: 320,
    defaultHeight: 180,
    inputLayout: 'dual',
    normalization: 'zero-255',
    preferWebGpu: true,
    license: 'Apache-2.0',
    licenseNote: 'CREStereo Init Iter2 (MEGVII / PINTO0309 変換 ONNX) / Apache-2.0（Concat自動分割によるWebGPU高速実行 + WASMフォールバック対応 · 180×320）',
    upstreamUrl: 'https://github.com/nburrus/stereodemo/releases/tag/v0.1-crestereo',
    onlineDelivery: 'cors-proxy',
    onlineSourceLabel: 'GitHub Releases (CORS中継 + ブラウザキャッシュ)',
    defaultModelUrl:
      'https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter2_180x320.onnx',
    corsProxyUrl: `${STEREO_CORS_PROXY_PREFIX}crestereo_init_iter2_180x320.onnx`,
  },
  'crestereo-combined-iter5': {
    variantId: 'crestereo-combined-iter5',
    tier: 'high-accuracy',
    engineType: 'onnx',
    fileName: 'crestereo_combined_iter5_240x320.onnx',
    defaultWidth: 320,
    defaultHeight: 240,
    inputLayout: 'dual',
    normalization: 'zero-255',
    preferWebGpu: true,
    license: 'Apache-2.0',
    licenseNote: 'CREStereo Combined Iter5 (MEGVII / PINTO0309 変換 ONNX) / Apache-2.0（Concat自動分割によるWebGPU高速実行 + WASMフォールバック対応 · 240×320）',
    upstreamUrl: 'https://github.com/nburrus/stereodemo/releases/tag/v0.1-crestereo',
    onlineDelivery: 'cors-proxy',
    onlineSourceLabel: 'GitHub Releases (CORS中継 + ブラウザキャッシュ)',
    defaultModelUrl:
      'https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter5_240x320.onnx',
    corsProxyUrl: `${STEREO_CORS_PROXY_PREFIX}crestereo_combined_iter5_240x320.onnx`,
  },
  'custom-onnx': {
    variantId: 'custom-onnx',
    tier: 'custom',
    engineType: 'onnx',
    fileName: 'custom_stereo.onnx',
    defaultWidth: 640,
    defaultHeight: 384,
    inputLayout: 'dual',
    normalization: 'imagenet',
    preferWebGpu: true,
    license: 'ユーザー指定',
    licenseNote: '任意のステレオ深度ONNXファイル（1入力[1,6,H,W]または2入力[1,3,H,W]を自動判定・Web Worker推論）',
    upstreamUrl: 'https://onnxruntime.ai/',
    onlineDelivery: 'local-only',
  },
};

export interface LocalStereoOnnxInfo {
  variantId: StereoDepthVariantId;
  fileName: string;
  size: number;
  loadedAt: number;
}

interface LocalStereoOnnxRecord extends LocalStereoOnnxInfo {
  bytes: Uint8Array;
}

const localStereoOnnxStore = new Map<StereoDepthVariantId, LocalStereoOnnxRecord>();
const localStereoListeners = new Set<() => void>();
let localStereoVersion = 0;

function notifyLocalStereoListeners(): void {
  localStereoVersion++;
  localStereoListeners.forEach((listener) => listener());
}

export function registerLocalStereoOnnxBytes(
  variantId: StereoDepthVariantId,
  bytes: Uint8Array,
  fileName: string
): void {
  localStereoOnnxStore.set(variantId, {
    variantId,
    bytes,
    fileName,
    size: bytes.byteLength,
    loadedAt: Date.now(),
  });
  notifyLocalStereoListeners();
}

export function clearLocalStereoOnnxBytes(variantId: StereoDepthVariantId): void {
  if (localStereoOnnxStore.delete(variantId)) {
    notifyLocalStereoListeners();
  }
}

export function getLocalStereoOnnxBytes(variantId: StereoDepthVariantId): Uint8Array | null {
  return localStereoOnnxStore.get(variantId)?.bytes ?? null;
}

export function getLocalStereoOnnxInfo(variantId: StereoDepthVariantId): LocalStereoOnnxInfo | null {
  const entry = localStereoOnnxStore.get(variantId);
  if (!entry) return null;
  return {
    variantId: entry.variantId,
    fileName: entry.fileName,
    size: entry.size,
    loadedAt: entry.loadedAt,
  };
}

export function subscribeLocalStereoOnnx(listener: () => void): () => void {
  localStereoListeners.add(listener);
  return () => {
    localStereoListeners.delete(listener);
  };
}

export function getLocalStereoOnnxVersion(): number {
  return localStereoVersion;
}

export function getStereoInputDimensions(input: InferenceInput): { width: number; height: number } {
  const width =
    'videoWidth' in input
      ? input.videoWidth
      : 'displayWidth' in input
        ? input.displayWidth
        : input.width;
  const height =
    'videoHeight' in input
      ? input.videoHeight
      : 'displayHeight' in input
        ? input.displayHeight
        : input.height;
  return { width, height };
}

export interface SplitSideBySideResult {
  leftCanvas: HTMLCanvasElement;
  rightCanvas: HTMLCanvasElement;
  eyeWidth: number;
  eyeHeight: number;
}

export function splitSideBySideFrame(
  input: InferenceInput,
  swapLeftRight = false,
  targetWidth?: number,
  targetHeight?: number,
  leftCanvasArg?: HTMLCanvasElement,
  rightCanvasArg?: HTMLCanvasElement,
  rotate180 = false
): SplitSideBySideResult {
  const { width: totalW, height: totalH } = getStereoInputDimensions(input);
  if (totalW < 2 || totalH < 1) {
    throw new RangeError(`Invalid stereo input dimensions: ${totalW}x${totalH}`);
  }
  const sourceEyeW = Math.floor(totalW / 2);
  const outW = targetWidth && targetWidth > 0 ? targetWidth : sourceEyeW;
  const outH = targetHeight && targetHeight > 0 ? targetHeight : totalH;

  const leftCanvas = leftCanvasArg ?? document.createElement('canvas');
  const rightCanvas = rightCanvasArg ?? document.createElement('canvas');
  if (leftCanvas.width !== outW || leftCanvas.height !== outH) {
    leftCanvas.width = outW;
    leftCanvas.height = outH;
  }
  if (rightCanvas.width !== outW || rightCanvas.height !== outH) {
    rightCanvas.width = outW;
    rightCanvas.height = outH;
  }

  const leftCtx = leftCanvas.getContext('2d', { willReadFrequently: true });
  const rightCtx = rightCanvas.getContext('2d', { willReadFrequently: true });
  if (!leftCtx || !rightCtx) {
    throw new Error('2D canvas context unavailable for Side-by-Side frame split');
  }

  // Rotating an inverted Side-by-Side frame by 180deg swaps its raw halves:
  // rot180([L | R]) = [rot180(R) | rot180(L)]. XOR keeps manual swapLeftRight intuitive.
  const effectiveSwap = swapLeftRight !== rotate180;
  const leftSrcX = effectiveSwap ? sourceEyeW : 0;
  const rightSrcX = effectiveSwap ? 0 : sourceEyeW;

  const drawEye = (ctx: CanvasRenderingContext2D, srcX: number) => {
    if (rotate180) {
      ctx.save?.();
      if (typeof ctx.translate === 'function' && typeof ctx.rotate === 'function') {
        ctx.translate(outW, outH);
        ctx.rotate(Math.PI);
      }
      ctx.drawImage(input as CanvasImageSource, srcX, 0, sourceEyeW, totalH, 0, 0, outW, outH);
      ctx.restore?.();
    } else {
      ctx.drawImage(input as CanvasImageSource, srcX, 0, sourceEyeW, totalH, 0, 0, outW, outH);
    }
  };

  drawEye(leftCtx, leftSrcX);
  drawEye(rightCtx, rightSrcX);

  return {
    leftCanvas,
    rightCanvas,
    eyeWidth: sourceEyeW,
    eyeHeight: totalH,
  };
}

function writeRgbaToCanvas(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  targetCanvas?: HTMLCanvasElement
): HTMLCanvasElement {
  const canvas = targetCanvas ?? document.createElement('canvas');
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;
  const img = ctx.createImageData(width, height);
  img.data.set(rgba);
  ctx.putImageData(img, 0, 0);
  return canvas;
}

export function renderDisparityColormap(
  disparity: Float32Array,
  width: number,
  height: number,
  scaleMinDisp: number,
  scaleMaxDisp: number,
  colormap: StereoColormap,
  targetCanvas?: HTMLCanvasElement
): HTMLCanvasElement {
  const rgba = renderDisparityColormapToRgba(
    disparity,
    width,
    height,
    scaleMinDisp,
    scaleMaxDisp,
    colormap
  );
  return writeRgbaToCanvas(rgba, width, height, targetCanvas);
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

export function inspectDisparityMapAtPoint(
  depthMap: StereoDepthMapData,
  normX: number,
  normY: number,
  baselineMm: number,
  focalLengthPx: number
): StereoDepthProbeResult | null {
  if (
    !Number.isFinite(normX) ||
    !Number.isFinite(normY) ||
    normX < 0 ||
    normX > 1 ||
    normY < 0 ||
    normY > 1 ||
    depthMap.width <= 0 ||
    depthMap.height <= 0
  ) {
    return null;
  }

  const mx = Math.min(depthMap.width - 1, Math.max(0, Math.floor(normX * depthMap.width)));
  const my = Math.min(depthMap.height - 1, Math.max(0, Math.floor(normY * depthMap.height)));
  const pixelX = Math.min(
    Math.max(0, depthMap.sourceEyeWidth - 1),
    Math.max(0, Math.floor(normX * depthMap.sourceEyeWidth))
  );
  const pixelY = Math.min(
    Math.max(0, depthMap.sourceEyeHeight - 1),
    Math.max(0, Math.floor(normY * depthMap.sourceEyeHeight))
  );

  const disp = depthMap.disparity[my * depthMap.width + mx];
  const valid = Number.isFinite(disp) && disp >= 0;
  const disparityPx = valid ? disp : 0;
  const depthMeters = valid ? disparityToDepthMeters(disparityPx, baselineMm, focalLengthPx) : null;

  return {
    normX,
    normY,
    pixelX,
    pixelY,
    disparityPx,
    valid,
    depthMeters,
  };
}

export interface StereoDepthRunOptions extends Partial<StereoDepthInferenceParams> {
  confidenceThreshold?: number;
  rotate180?: boolean;
}

export interface StereoDepthTask {
  backend?: OnnxBackendInfo;
  estimateForVideo: (
    input: InferenceInput,
    options?: StereoDepthRunOptions
  ) => Promise<StereoDepthMapData> | StereoDepthMapData;
  rerenderOverlay?: (
    depthMap: StereoDepthMapData,
    options: StereoDepthRunOptions
  ) => StereoDepthMapData;
  close: () => Promise<void> | void;
}

function buildDepthMapData(
  disparityInSourcePx: Float32Array,
  mapWidth: number,
  mapHeight: number,
  leftCanvas: HTMLCanvasElement,
  rightCanvas: HTMLCanvasElement,
  sourceEyeWidth: number,
  sourceEyeHeight: number,
  options: StereoDepthRunOptions,
  overlayCanvas?: HTMLCanvasElement
): StereoDepthMapData {
  const autoNorm = options.autoNormalizeDisparity ?? true;
  const manualMin = options.minDisparityPx ?? 0;
  const manualMax = options.maxDisparityPx ?? 64;
  const colormap: StereoColormap = options.colormap ?? 'turbo';
  const viewMode: StereoViewMode = options.viewMode ?? 'left-overlay';
  const swapLeftRight = options.swapLeftRight ?? false;
  const rotate180 = options.rotate180 ?? false;

  const stats = computeDisparityStats(disparityInSourcePx, autoNorm, manualMin, manualMax);
  const renderedCanvas = renderDisparityColormap(
    disparityInSourcePx,
    mapWidth,
    mapHeight,
    stats.scaleMinDisparity,
    stats.scaleMaxDisparity,
    colormap,
    overlayCanvas
  );

  return {
    width: mapWidth,
    height: mapHeight,
    disparity: disparityInSourcePx,
    minDisparity: stats.minDisparity,
    maxDisparity: stats.maxDisparity,
    medianDisparity: stats.medianDisparity,
    scaleMinDisparity: stats.scaleMinDisparity,
    scaleMaxDisparity: stats.scaleMaxDisparity,
    validRatio: stats.validRatio,
    overlayCanvas: renderedCanvas,
    leftCanvas,
    rightCanvas,
    colormap,
    viewMode,
    swapLeftRight,
    rotate180,
    sourceEyeWidth,
    sourceEyeHeight,
  };
}

export function rerenderStereoDepthMap(
  depthMap: StereoDepthMapData,
  options: StereoDepthRunOptions
): StereoDepthMapData {
  return buildDepthMapData(
    depthMap.disparity,
    depthMap.width,
    depthMap.height,
    depthMap.leftCanvas,
    depthMap.rightCanvas,
    depthMap.sourceEyeWidth,
    depthMap.sourceEyeHeight,
    {
      ...options,
      swapLeftRight: options.swapLeftRight ?? depthMap.swapLeftRight,
      rotate180: options.rotate180 ?? depthMap.rotate180,
    }
  );
}

// ============================================================================
// Baseline Engine: Built-in Separable SAD Stereo Matcher (In-Process Fallback)
// ============================================================================

export class StereoSadEngine implements StereoDepthTask {
  public readonly backend: OnnxBackendInfo = {
    runtime: 'stereo-sad-cpu',
    requestedProvider: 'cpu',
    configuredProviders: ['cpu-sad'],
    fallbackReason: null,
    actualBackend: 'unmeasured',
  };

  private leftScratch: HTMLCanvasElement | null = null;
  private rightScratch: HTMLCanvasElement | null = null;
  private closed = false;

  estimateForVideo(
    input: InferenceInput,
    options: StereoDepthRunOptions = {}
  ): StereoDepthMapData {
    if (this.closed) {
      throw new OnnxInferenceError('stereo-depth', 'closed', 'task is closed');
    }
    const { width: totalW, height: totalH } = getStereoInputDimensions(input);
    const sourceEyeW = Math.max(1, Math.floor(totalW / 2));
    const sourceEyeH = Math.max(1, totalH);
    const targetW = Math.min(320, sourceEyeW);
    const targetH = Math.max(48, Math.round((sourceEyeH * targetW) / sourceEyeW));

    this.leftScratch ??= document.createElement('canvas');
    this.rightScratch ??= document.createElement('canvas');

    const { leftCanvas, rightCanvas } = splitSideBySideFrame(
      input,
      options.swapLeftRight ?? false,
      targetW,
      targetH,
      this.leftScratch,
      this.rightScratch,
      options.rotate180 ?? false
    );

    const leftCtx = leftCanvas.getContext('2d', { willReadFrequently: true });
    const rightCtx = rightCanvas.getContext('2d', { willReadFrequently: true });
    if (!leftCtx || !rightCtx) {
      throw new Error('2D context unavailable for stereo SAD estimation');
    }

    const leftImg = leftCtx.getImageData(0, 0, targetW, targetH);
    const rightImg = rightCtx.getImageData(0, 0, targetW, targetH);

    // Scale user maxDisparity (in source-eye pixels) down to processing resolution
    const userMaxDisp = Math.max(8, Math.min(256, options.maxDisparity ?? 64));
    const downscaleRatio = targetW / sourceEyeW;
    const minScaledDisp = sourceEyeW >= 1280 ? 24 : 6;
    const scaledMaxDisp = Math.max(minScaledDisp, Math.min(targetW - 2, Math.round(userMaxDisp * downscaleRatio)));

    const sadResult = computeSadDisparity(leftImg.data, rightImg.data, targetW, targetH, {
      maxDisparity: scaledMaxDisp,
      windowSize: options.sadWindowSize ?? 7,
      confidenceThreshold: options.confidenceThreshold ?? 0.35,
    });

    // Convert disparity values back into true source-eye pixels so focalLengthPx matches 1:1
    const dispScaleToSource = sourceEyeW / targetW;
    const sourceDisparity = new Float32Array(sadResult.disparity.length);
    for (let i = 0; i < sadResult.disparity.length; i++) {
      const v = sadResult.disparity[i];
      sourceDisparity[i] = v >= 0 ? v * dispScaleToSource : -1;
    }

    // Clone scratch canvases so the returned DetectionFrame owns immutable snapshots
    const leftCopy = document.createElement('canvas');
    leftCopy.width = targetW;
    leftCopy.height = targetH;
    leftCopy.getContext('2d')?.drawImage(leftCanvas, 0, 0);

    const rightCopy = document.createElement('canvas');
    rightCopy.width = targetW;
    rightCopy.height = targetH;
    rightCopy.getContext('2d')?.drawImage(rightCanvas, 0, 0);

    return buildDepthMapData(
      sourceDisparity,
      targetW,
      targetH,
      leftCopy,
      rightCopy,
      sourceEyeW,
      sourceEyeH,
      options
    );
  }

  rerenderOverlay(depthMap: StereoDepthMapData, options: StereoDepthRunOptions): StereoDepthMapData {
    return rerenderStereoDepthMap(depthMap, options);
  }

  close(): void {
    this.closed = true;
    this.leftScratch = null;
    this.rightScratch = null;
  }
}

export class StereoWgslEngine implements StereoDepthTask {
  public backend: OnnxBackendInfo;
  private wgslEngine: WebGpuStereoComputeEngine | null;
  private leftScratch: HTMLCanvasElement | null = null;
  private rightScratch: HTMLCanvasElement | null = null;
  private closed = false;

  private constructor(
    wgslEngine: WebGpuStereoComputeEngine | null,
    backend: OnnxBackendInfo
  ) {
    this.wgslEngine = wgslEngine;
    this.backend = backend;
  }

  static async create(
    backendPreference: StereoBackendPreference = 'auto'
  ): Promise<StereoWgslEngine> {
    if (backendPreference === 'wasm') {
      return new StereoWgslEngine(null, {
        runtime: 'stereo-webgpu-wgsl',
        requestedProvider: 'wasm',
        configuredProviders: ['cpu-sad'],
        fallbackReason: '手動設定により WASM/CPU SAD で実行',
        actualBackend: 'unmeasured',
      });
    }
    try {
      const gpuEngine = await createWebGpuStereoComputeEngine();
      return new StereoWgslEngine(gpuEngine, {
        runtime: 'stereo-webgpu-wgsl',
        requestedProvider: 'webgpu',
        configuredProviders: ['webgpu-wgsl'],
        fallbackReason: null,
        actualBackend: 'unmeasured',
      });
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      return new StereoWgslEngine(null, {
        runtime: 'stereo-webgpu-wgsl',
        requestedProvider: 'webgpu',
        configuredProviders: ['cpu-sad'],
        fallbackReason: `WebGPU WGSL利用不可 (${reason}); CPU SAD へ自動フォールバック`,
        actualBackend: 'unmeasured',
      });
    }
  }

  async estimateForVideo(
    input: InferenceInput,
    options: StereoDepthRunOptions = {}
  ): Promise<StereoDepthMapData> {
    if (this.closed) {
      throw new OnnxInferenceError('stereo-depth', 'closed', 'task is closed');
    }
    const { width: totalW, height: totalH } = getStereoInputDimensions(input);
    const sourceEyeW = Math.max(1, Math.floor(totalW / 2));
    const sourceEyeH = Math.max(1, totalH);
    const maxW = this.wgslEngine ? 480 : 320;
    const targetW = Math.min(maxW, sourceEyeW);
    const targetH = Math.max(48, Math.round((sourceEyeH * targetW) / sourceEyeW));

    this.leftScratch ??= document.createElement('canvas');
    this.rightScratch ??= document.createElement('canvas');

    const { leftCanvas, rightCanvas } = splitSideBySideFrame(
      input,
      options.swapLeftRight ?? false,
      targetW,
      targetH,
      this.leftScratch,
      this.rightScratch,
      options.rotate180 ?? false
    );

    const leftCtx = leftCanvas.getContext('2d', { willReadFrequently: true });
    const rightCtx = rightCanvas.getContext('2d', { willReadFrequently: true });
    if (!leftCtx || !rightCtx) {
      throw new Error('2D context unavailable for stereo WGSL estimation');
    }

    const leftImg = leftCtx.getImageData(0, 0, targetW, targetH);
    const rightImg = rightCtx.getImageData(0, 0, targetW, targetH);

    const userMaxDisp = Math.max(8, Math.min(256, options.maxDisparity ?? 64));
    const downscaleRatio = targetW / sourceEyeW;
    const minScaledDisp = sourceEyeW >= 1280 ? 24 : 6;
    const scaledMaxDisp = Math.max(
      minScaledDisp,
      Math.min(targetW - 2, Math.round(userMaxDisp * downscaleRatio))
    );

    let rawDisparity: Float32Array;
    if (this.wgslEngine) {
      try {
        const res = await this.wgslEngine.estimate(leftImg.data, rightImg.data, targetW, targetH, {
          maxDisparity: scaledMaxDisp,
          windowSize: options.sadWindowSize ?? 7,
          confidenceThreshold: options.confidenceThreshold ?? 0.35,
        });
        rawDisparity = res.disparity;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        this.wgslEngine.destroy();
        this.wgslEngine = null;
        this.backend = {
          ...this.backend,
          configuredProviders: ['cpu-sad'],
          fallbackReason: `WebGPU WGSL実行エラー (${msg}); CPU SAD へ自動フォールバック`,
        };
        rawDisparity = computeSadDisparity(leftImg.data, rightImg.data, targetW, targetH, {
          maxDisparity: scaledMaxDisp,
          windowSize: options.sadWindowSize ?? 7,
          confidenceThreshold: options.confidenceThreshold ?? 0.35,
        }).disparity;
      }
    } else {
      rawDisparity = computeSadDisparity(leftImg.data, rightImg.data, targetW, targetH, {
        maxDisparity: scaledMaxDisp,
        windowSize: options.sadWindowSize ?? 7,
        confidenceThreshold: options.confidenceThreshold ?? 0.35,
      }).disparity;
    }

    const dispScaleToSource = sourceEyeW / targetW;
    const sourceDisparity = new Float32Array(rawDisparity.length);
    for (let i = 0; i < rawDisparity.length; i++) {
      const v = rawDisparity[i];
      sourceDisparity[i] = v >= 0 ? v * dispScaleToSource : -1;
    }

    const leftCopy = document.createElement('canvas');
    leftCopy.width = targetW;
    leftCopy.height = targetH;
    leftCopy.getContext('2d')?.drawImage(leftCanvas, 0, 0);

    const rightCopy = document.createElement('canvas');
    rightCopy.width = targetW;
    rightCopy.height = targetH;
    rightCopy.getContext('2d')?.drawImage(rightCanvas, 0, 0);

    return buildDepthMapData(
      sourceDisparity,
      targetW,
      targetH,
      leftCopy,
      rightCopy,
      sourceEyeW,
      sourceEyeH,
      options
    );
  }

  rerenderOverlay(depthMap: StereoDepthMapData, options: StereoDepthRunOptions): StereoDepthMapData {
    return rerenderStereoDepthMap(depthMap, options);
  }

  close(): void {
    this.closed = true;
    this.leftScratch = null;
    this.rightScratch = null;
    if (this.wgslEngine) {
      this.wgslEngine.destroy();
      this.wgslEngine = null;
    }
  }
}

// ============================================================================
// Neural Stereo Engine: In-Process ONNX Runtime Web (Test / Fallback Engine)
// ============================================================================

export type StereoWasmRecoveryFactory = () => Promise<{
  session: ort.InferenceSession;
  backend: OnnxBackendInfo;
}>;

export class StereoOnnxEngine implements StereoDepthTask {
  private readonly owner: OnnxSessionOwner;
  private session: ort.InferenceSession;
  public backend: OnnxBackendInfo;
  private readonly fallbackToWasm?: StereoWasmRecoveryFactory;
  private attemptedWasmFallback = false;
  private leftScratch: HTMLCanvasElement | null = null;
  private rightScratch: HTMLCanvasElement | null = null;

  constructor(
    session: ort.InferenceSession,
    private readonly profile: StereoModelProfile,
    backend?: OnnxBackendInfo,
    fallbackToWasm?: StereoWasmRecoveryFactory
  ) {
    this.session = session;
    this.backend = backend ?? unknownOnnxBackend();
    this.fallbackToWasm = fallbackToWasm;
    this.owner = new OnnxSessionOwner(
      { release: () => this.session.release() },
      `stereo-depth:${profile.variantId}`
    );
  }

  private resolveTargetResolution(): { width: number; height: number } {
    const inputMeta = (this.session as unknown as {
      inputMetadata?: ReadonlyArray<{
        shape?: readonly (number | string)[];
        dimensions?: readonly (number | string)[];
      }>;
    }).inputMetadata;
    const primaryMeta = inputMeta && inputMeta.length > 0 ? inputMeta[inputMeta.length - 1] : undefined;
    const targetDims = primaryMeta?.shape ?? primaryMeta?.dimensions;
    if (targetDims && targetDims.length === 4) {
      const mh = targetDims[2];
      const mw = targetDims[3];
      if (typeof mh === 'number' && mh > 0 && typeof mw === 'number' && mw > 0) {
        return { width: mw, height: mh };
      }
    }
    return { width: this.profile.defaultWidth, height: this.profile.defaultHeight };
  }

  async estimateForVideo(
    input: InferenceInput,
    options: StereoDepthRunOptions = {}
  ): Promise<StereoDepthMapData> {
    return this.owner.use(async () => {
      const { width: totalW, height: totalH } = getStereoInputDimensions(input);
      const sourceEyeW = Math.max(1, Math.floor(totalW / 2));
      const sourceEyeH = Math.max(1, totalH);
      const { width: modelW, height: modelH } = this.resolveTargetResolution();

      this.leftScratch ??= document.createElement('canvas');
      this.rightScratch ??= document.createElement('canvas');

      const { leftCanvas, rightCanvas } = splitSideBySideFrame(
        input,
        options.swapLeftRight ?? false,
        modelW,
        modelH,
        this.leftScratch,
        this.rightScratch,
        options.rotate180 ?? false
      );

      const leftCtx = leftCanvas.getContext('2d', { willReadFrequently: true });
      const rightCtx = rightCanvas.getContext('2d', { willReadFrequently: true });
      if (!leftCtx || !rightCtx) {
        throw new OnnxInferenceError('stereo-depth', 'input', '2D context unavailable');
      }

      const leftRgba = leftCtx.getImageData(0, 0, modelW, modelH).data;
      const rightRgba = rightCtx.getImageData(0, 0, modelW, modelH).data;
      const pixelCount = modelW * modelH;

      const inputNames = this.session.inputNames;
      const feeds: Record<string, ort.Tensor> = {};
      const createdTensors: ort.Tensor[] = [];

      try {
        const inputMeta = (this.session as unknown as {
          inputMetadata?: ReadonlyArray<{
            shape?: readonly (number | string)[];
            dimensions?: readonly (number | string)[];
          }>;
        }).inputMetadata;

        if (inputNames.length === 1) {
          const singleDims = inputMeta?.[0]?.shape ?? inputMeta?.[0]?.dimensions;
          const channels =
            typeof singleDims?.[1] === 'number'
              ? (singleDims[1] as number)
              : this.profile.inputLayout === 'stacked'
                ? 6
                : 6;
          const data = new Float32Array(channels * pixelCount);
          rgbaToChwInto(leftRgba, pixelCount, data, 0, this.profile.normalization);
          if (channels >= 6) {
            rgbaToChwInto(rightRgba, pixelCount, data, pixelCount * 3, this.profile.normalization);
          }
          const tensor = new ort.Tensor('float32', data, [1, channels, modelH, modelW]);
          feeds[inputNames[0]] = tensor;
          createdTensors.push(tensor);
        } else if (inputNames.length === 2) {
          const leftData = new Float32Array(3 * pixelCount);
          const rightData = new Float32Array(3 * pixelCount);
          rgbaToChwInto(leftRgba, pixelCount, leftData, 0, this.profile.normalization);
          rgbaToChwInto(rightRgba, pixelCount, rightData, 0, this.profile.normalization);
          const leftTensor = new ort.Tensor('float32', leftData, [1, 3, modelH, modelW]);
          const rightTensor = new ort.Tensor('float32', rightData, [1, 3, modelH, modelW]);
          const secondIsLeft =
            /left/i.test(inputNames[1]) && !/left/i.test(inputNames[0]);
          feeds[inputNames[0]] = secondIsLeft ? rightTensor : leftTensor;
          feeds[inputNames[1]] = secondIsLeft ? leftTensor : rightTensor;
          createdTensors.push(leftTensor, rightTensor);
        } else if (inputNames.length === 4) {
          const halfW = Math.max(16, Math.floor(modelW / 2));
          const halfH = Math.max(16, Math.floor(modelH / 2));
          const initLRgba = downscaleRgbaBuffer(leftRgba, modelW, modelH, halfW, halfH);
          const initRRgba = downscaleRgbaBuffer(rightRgba, modelW, modelH, halfW, halfH);

          const initPixels = halfW * halfH;
          const initL = new Float32Array(3 * initPixels);
          const initR = new Float32Array(3 * initPixels);
          rgbaToChwInto(initLRgba, initPixels, initL, 0, this.profile.normalization);
          rgbaToChwInto(initRRgba, initPixels, initR, 0, this.profile.normalization);

          const nextL = new Float32Array(3 * pixelCount);
          const nextR = new Float32Array(3 * pixelCount);
          rgbaToChwInto(leftRgba, pixelCount, nextL, 0, this.profile.normalization);
          rgbaToChwInto(rightRgba, pixelCount, nextR, 0, this.profile.normalization);

          const tInitL = new ort.Tensor('float32', initL, [1, 3, halfH, halfW]);
          const tInitRTensor = new ort.Tensor('float32', initR, [1, 3, halfH, halfW]);
          const tNextL = new ort.Tensor('float32', nextL, [1, 3, modelH, modelW]);
          const tNextR = new ort.Tensor('float32', nextR, [1, 3, modelH, modelW]);
          createdTensors.push(tInitL, tInitRTensor, tNextL, tNextR);

          for (let i = 0; i < inputNames.length; i++) {
            const nLower = inputNames[i].toLowerCase();
            if (nLower.includes('init') && nLower.includes('left')) feeds[inputNames[i]] = tInitL;
            else if (nLower.includes('init') && nLower.includes('right')) feeds[inputNames[i]] = tInitRTensor;
            else if (nLower.includes('left')) feeds[inputNames[i]] = tNextL;
            else if (nLower.includes('right')) feeds[inputNames[i]] = tNextR;
            else if (i === 0) feeds[inputNames[i]] = tInitL;
            else if (i === 1) feeds[inputNames[i]] = tInitRTensor;
            else if (i === 2) feeds[inputNames[i]] = tNextL;
            else feeds[inputNames[i]] = tNextR;
          }
        } else {
          throw new OnnxInferenceError(
            'stereo-depth',
            'input',
            `Unsupported ONNX stereo input count (${inputNames.length}): [${inputNames.join(', ')}]`
          );
        }

        let results: Record<string, ort.Tensor>;
        try {
          results = await this.session.run(feeds);
        } catch (runError) {
          if (
            !this.attemptedWasmFallback &&
            this.fallbackToWasm &&
            this.backend.configuredProviders.includes('webgpu') &&
            isWebGpuRuntimeKernelError(runError)
          ) {
            this.attemptedWasmFallback = true;
            const failedSession = this.session;
            try {
              await failedSession.release();
            } catch {
              // Ignore cleanup errors on the failed WebGPU session before switching to WASM
            }
            const recovered = await this.fallbackToWasm();
            if (this.owner.closed) {
              try {
                await recovered.session.release();
              } catch {
                // Ignore cleanup error if task was closed mid-recovery
              }
              throw new OnnxInferenceError(`stereo-depth:${this.profile.variantId}`, 'closed', 'task is closed');
            }
            this.session = recovered.session;
            const errMsg = runError instanceof Error ? runError.message : String(runError);
            this.backend = {
              ...recovered.backend,
              requestedProvider: this.backend.requestedProvider,
              fallbackReason: `WebGPU kernel runtime failure (${errMsg}); fell back to WASM`,
            };
            results = await this.session.run(feeds);
          } else {
            throw runError;
          }
        }
        const outputValues = Object.values(results);
        for (const t of outputValues) {
          createdTensors.push(t);
        }
        const primaryOutput =
          results[this.session.outputNames[0]] ?? outputValues[0];
        if (!primaryOutput) {
          throw new OnnxInferenceError('stereo-depth', 'output', 'ONNX session returned no output tensor');
        }

        const extracted = extractDisparityFromOrtTensor(primaryOutput);
        const scaleToSource = sourceEyeW / extracted.width;
        const scaledDisparity = new Float32Array(extracted.disparity.length);
        for (let i = 0; i < extracted.disparity.length; i++) {
          const v = extracted.disparity[i];
          scaledDisparity[i] = v >= 0 ? v * scaleToSource : -1;
        }

        const leftCopy = document.createElement('canvas');
        leftCopy.width = modelW;
        leftCopy.height = modelH;
        leftCopy.getContext('2d')?.drawImage(leftCanvas, 0, 0);

        const rightCopy = document.createElement('canvas');
        rightCopy.width = modelW;
        rightCopy.height = modelH;
        rightCopy.getContext('2d')?.drawImage(rightCanvas, 0, 0);

        return buildDepthMapData(
          scaledDisparity,
          extracted.width,
          extracted.height,
          leftCopy,
          rightCopy,
          sourceEyeW,
          sourceEyeH,
          options
        );
      } finally {
        disposeTensors(createdTensors);
      }
    });
  }

  rerenderOverlay(depthMap: StereoDepthMapData, options: StereoDepthRunOptions): StereoDepthMapData {
    return rerenderStereoDepthMap(depthMap, options);
  }

  close(): Promise<void> {
    this.leftScratch = null;
    this.rightScratch = null;
    return this.owner.close();
  }
}

// ============================================================================
// Dedicated Web Worker Engine (Offloads SAD & ONNX WASM/WebGPU off Main Thread)
// ============================================================================

export interface StereoWorkerLike {
  postMessage: (message: StereoWorkerRequest, transfer?: Transferable[]) => void;
  onmessage: ((event: MessageEvent<StereoWorkerResponse>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  terminate: () => void;
}

interface PendingStereoWorkerRequest {
  resolve: (msg: StereoWorkerResponse) => void;
  reject: (err: unknown) => void;
}

export interface StereoWorkerInitConfig {
  variantId: StereoDepthVariantId;
  profile: StereoModelProfile;
  onnxBytes?: Uint8Array;
  backendPreference?: StereoBackendPreference;
  primarySource?: AssetDeliverySource;
  recordAssetSource?: AssetSourceRecorder;
  workerFactory?: () => StereoWorkerLike;
}

let activeInitController: AbortController | null = null;
let activeInitWorker: StereoWorkerLike | null = null;

/**
 * Cancels any in-flight Stereo Depth model download or Web Worker session compilation.
 * Called automatically when a new Stereo Depth initialization starts so variant switches
 * respond immediately without waiting for an older model's download or compilation.
 */
export function cancelInFlightStereoInit(): void {
  if (activeInitController) {
    activeInitController.abort();
    activeInitController = null;
  }
  if (activeInitWorker) {
    try {
      activeInitWorker.terminate();
    } catch {
      // Ignore worker termination errors
    }
    activeInitWorker = null;
  }
}

export class StereoWorkerEngine implements StereoDepthTask {
  public backend: OnnxBackendInfo;
  private targetWidth: number;
  private targetHeight: number;
  private readonly variantId: StereoDepthVariantId;
  private readonly engineType: 'sad-cpu' | 'webgpu-wgsl' | 'onnx';
  private nextRequestId = 1;
  private closed = false;
  private leftScratch: HTMLCanvasElement | null = null;
  private rightScratch: HTMLCanvasElement | null = null;
  private readonly pendingRequests = new Map<number, PendingStereoWorkerRequest>();
  private inFlightEstimate: Promise<StereoDepthMapData> | null = null;
  private queuedEstimate: {
    input: InferenceInput;
    options: StereoDepthRunOptions;
    promise: Promise<StereoDepthMapData>;
    resolve: (val: StereoDepthMapData) => void;
    reject: (err: unknown) => void;
  } | null = null;

  private constructor(
    private readonly worker: StereoWorkerLike,
    variantId: StereoDepthVariantId,
    engineType: 'sad-cpu' | 'webgpu-wgsl' | 'onnx',
    backend: OnnxBackendInfo,
    targetWidth: number,
    targetHeight: number
  ) {
    this.variantId = variantId;
    this.engineType = engineType;
    this.backend = backend;
    this.targetWidth = targetWidth;
    this.targetHeight = targetHeight;

    this.worker.onmessage = (event: MessageEvent<StereoWorkerResponse>) => {
      const msg = event.data;
      if (!msg || typeof msg.id !== 'number') return;
      const pending = this.pendingRequests.get(msg.id);
      if (!pending) return;
      this.pendingRequests.delete(msg.id);
      if (msg.type === 'error') {
        pending.reject(
          new OnnxInferenceError(
            `stereo-depth:${this.variantId}`,
            'run',
            msg.message
          )
        );
      } else {
        pending.resolve(msg);
      }
    };

    this.worker.onerror = (event: ErrorEvent) => {
      const err = new OnnxInferenceError(
        `stereo-depth:${this.variantId}`,
        'run',
        event?.message || 'Stereo Depth Web Worker runtime error'
      );
      this.rejectAllPending(err);
    };
  }

  private static normalizeWasmBaseUrl(url: string | null | undefined): string | null {
    if (!url) return null;
    if (url.startsWith('/') && typeof window !== 'undefined' && window.location?.origin) {
      return `${window.location.origin}${url}`;
    }
    return url;
  }

  static async create(config: StereoWorkerInitConfig): Promise<StereoWorkerEngine> {
    if (config.profile.engineType === 'onnx' && !config.onnxBytes) {
      throw new Error(`Missing ONNX bytes for ${config.variantId}`);
    }

    const resolvedWasmUrls =
      config.profile.engineType === 'onnx' ? await resolveOrtWasmUrls() : null;
    const wasmPrimaryUrl =
      StereoWorkerEngine.normalizeWasmBaseUrl(resolvedWasmUrls?.primaryUrl) ?? '';
    const wasmFallbackUrl = StereoWorkerEngine.normalizeWasmBaseUrl(
      resolvedWasmUrls?.fallbackUrl
    );

    const worker: StereoWorkerLike = config.workerFactory
      ? config.workerFactory()
      : (new Worker(new URL('./stereoDepthWorker.ts', import.meta.url), {
          type: 'module',
        }) as unknown as StereoWorkerLike);

    activeInitWorker = worker;

    try {
      const initResult = await new Promise<Extract<StereoWorkerResponse, { type: 'init-result' }>>(
        (resolve, reject) => {
          const requestId = 0;
          worker.onmessage = (event: MessageEvent<StereoWorkerResponse>) => {
            const msg = event.data;
            if (!msg || msg.id !== requestId) return;
            if (msg.type === 'init-result') {
              resolve(msg);
            } else if (msg.type === 'error') {
              reject(new Error(msg.message));
            }
          };
          worker.onerror = (event: ErrorEvent) => {
            reject(new Error(event?.message || 'Stereo Depth Web Worker failed during initialization'));
          };

          if (config.profile.engineType === 'sad-cpu') {
            worker.postMessage({
              type: 'init-sad',
              id: requestId,
              profile: config.profile,
            });
          } else if (config.profile.engineType === 'webgpu-wgsl') {
            worker.postMessage({
              type: 'init-wgsl',
              id: requestId,
              profile: config.profile,
              backendPreference: config.backendPreference,
            });
          } else {
            const copy = new Uint8Array(config.onnxBytes!.byteLength);
            copy.set(config.onnxBytes!);
            worker.postMessage(
              {
                type: 'init-onnx',
                id: requestId,
                profile: config.profile,
                onnxBuffer: copy.buffer,
                wasmPrimaryUrl,
                wasmFallbackUrl,
                backendPreference: config.backendPreference,
              },
              [copy.buffer]
            );
          }
        }
      );

      if (activeInitWorker === worker) {
        activeInitWorker = null;
      }

      if (config.profile.engineType === 'sad-cpu' || config.profile.engineType === 'webgpu-wgsl') {
        config.recordAssetSource?.('stereo-depth', 'browser-cache');
        config.recordAssetSource?.(`stereo-depth:${config.variantId}`, 'browser-cache');
      } else {
        const src = config.primarySource ?? 'external';
        config.recordAssetSource?.('stereo-depth', src);
        config.recordAssetSource?.(`stereo-depth:${config.variantId}`, src);
        const wasmSource =
          initResult.wasmSourceUsed === 'fallback'
            ? 'external-fallback'
            : (resolvedWasmUrls?.source ?? 'external');
        config.recordAssetSource?.('wasm-ort', wasmSource);
      }

      return new StereoWorkerEngine(
        worker,
        config.variantId,
        config.profile.engineType,
        initResult.backend,
        initResult.targetWidth,
        initResult.targetHeight
      );
    } catch (err) {
      if (activeInitWorker === worker) {
        activeInitWorker = null;
      }
      try {
        worker.terminate();
      } catch {
        // Ignore cleanup error
      }
      throw err;
    }
  }

  private rejectAllPending(error: unknown): void {
    for (const [, pending] of this.pendingRequests) {
      pending.reject(error);
    }
    this.pendingRequests.clear();
    if (this.queuedEstimate) {
      const queued = this.queuedEstimate;
      this.queuedEstimate = null;
      queued.reject(error);
    }
    this.inFlightEstimate = null;
  }

  async estimateForVideo(
    input: InferenceInput,
    options: StereoDepthRunOptions = {}
  ): Promise<StereoDepthMapData> {
    if (this.closed) {
      throw new OnnxInferenceError(`stereo-depth:${this.variantId}`, 'closed', 'task is closed');
    }

    if (this.inFlightEstimate) {
      if (this.queuedEstimate) {
        this.queuedEstimate.input = input;
        this.queuedEstimate.options = options;
        return this.queuedEstimate.promise;
      }
      let resolveQueued!: (val: StereoDepthMapData) => void;
      let rejectQueued!: (err: unknown) => void;
      const promise = new Promise<StereoDepthMapData>((resolve, reject) => {
        resolveQueued = resolve;
        rejectQueued = reject;
      });
      this.queuedEstimate = {
        input,
        options,
        promise,
        resolve: resolveQueued,
        reject: rejectQueued,
      };
      return promise;
    }

    return this.dispatchEstimate(input, options);
  }

  private dispatchEstimate(
    input: InferenceInput,
    options: StereoDepthRunOptions
  ): Promise<StereoDepthMapData> {
    const runPromise = this.runSingleEstimate(input, options).finally(() => {
      if (this.inFlightEstimate === runPromise) {
        this.inFlightEstimate = null;
      }
      if (!this.closed && this.queuedEstimate) {
        const next = this.queuedEstimate;
        this.queuedEstimate = null;
        this.dispatchEstimate(next.input, next.options).then(next.resolve, next.reject);
      }
    });
    this.inFlightEstimate = runPromise;
    return runPromise;
  }

  private async runSingleEstimate(
    input: InferenceInput,
    options: StereoDepthRunOptions = {}
  ): Promise<StereoDepthMapData> {
    if (this.closed) {
      throw new OnnxInferenceError(`stereo-depth:${this.variantId}`, 'closed', 'task is closed');
    }

    const { width: totalW, height: totalH } = getStereoInputDimensions(input);
    const sourceEyeW = Math.max(1, Math.floor(totalW / 2));
    const sourceEyeH = Math.max(1, totalH);

    const isDynamicAspect = this.engineType === 'sad-cpu' || this.engineType === 'webgpu-wgsl';
    const maxAspectWidth =
      this.engineType === 'webgpu-wgsl' ? Math.max(320, this.targetWidth) : 320;
    const targetW = isDynamicAspect ? Math.min(maxAspectWidth, sourceEyeW) : this.targetWidth;
    const targetH = isDynamicAspect
      ? Math.max(48, Math.round((sourceEyeH * targetW) / sourceEyeW))
      : this.targetHeight;

    this.leftScratch ??= document.createElement('canvas');
    this.rightScratch ??= document.createElement('canvas');

    const { leftCanvas, rightCanvas } = splitSideBySideFrame(
      input,
      options.swapLeftRight ?? false,
      targetW,
      targetH,
      this.leftScratch,
      this.rightScratch,
      options.rotate180 ?? false
    );

    const leftCtx = leftCanvas.getContext('2d', { willReadFrequently: true });
    const rightCtx = rightCanvas.getContext('2d', { willReadFrequently: true });
    if (!leftCtx || !rightCtx) {
      throw new OnnxInferenceError(
        `stereo-depth:${this.variantId}`,
        'input',
        '2D context unavailable for stereo Worker estimation'
      );
    }

    const leftImg = leftCtx.getImageData(0, 0, targetW, targetH);
    const rightImg = rightCtx.getImageData(0, 0, targetW, targetH);

    const leftCopy = document.createElement('canvas');
    leftCopy.width = targetW;
    leftCopy.height = targetH;
    leftCopy.getContext('2d')?.drawImage(leftCanvas, 0, 0);

    const rightCopy = document.createElement('canvas');
    rightCopy.width = targetW;
    rightCopy.height = targetH;
    rightCopy.getContext('2d')?.drawImage(rightCanvas, 0, 0);

    const requestId = this.nextRequestId++;
    const leftBuffer = leftImg.data.buffer.slice(
      leftImg.data.byteOffset,
      leftImg.data.byteOffset + leftImg.data.byteLength
    );
    const rightBuffer = rightImg.data.buffer.slice(
      rightImg.data.byteOffset,
      rightImg.data.byteOffset + rightImg.data.byteLength
    );

    const response = await new Promise<Extract<StereoWorkerResponse, { type: 'estimate-result' }>>(
      (resolve, reject) => {
        if (this.closed) {
          reject(new OnnxInferenceError(`stereo-depth:${this.variantId}`, 'closed', 'task is closed'));
          return;
        }
        this.pendingRequests.set(requestId, {
          resolve: (msg) => {
            if (msg.type === 'estimate-result') {
              resolve(msg);
            } else {
              reject(
                new OnnxInferenceError(
                  `stereo-depth:${this.variantId}`,
                  'run',
                  `Unexpected worker response: ${msg.type}`
                )
              );
            }
          },
          reject,
        });

        this.worker.postMessage(
          {
            type: 'estimate',
            id: requestId,
            leftRgbaBuffer: leftBuffer,
            rightRgbaBuffer: rightBuffer,
            width: targetW,
            height: targetH,
            sourceEyeWidth: sourceEyeW,
            sourceEyeHeight: sourceEyeH,
            options: {
              confidenceThreshold: options.confidenceThreshold,
              maxDisparity: options.maxDisparity,
              sadWindowSize: options.sadWindowSize,
              autoNormalizeDisparity: options.autoNormalizeDisparity,
              minDisparityPx: options.minDisparityPx,
              maxDisparityPx: options.maxDisparityPx,
              colormap: options.colormap,
            },
          },
          [leftBuffer, rightBuffer]
        );
      }
    );

    if (this.closed) {
      throw new OnnxInferenceError(`stereo-depth:${this.variantId}`, 'closed', 'task is closed');
    }

    this.backend = response.backend;
    const disparity = new Float32Array(response.disparityBuffer);
    const overlayRgba = new Uint8ClampedArray(response.overlayRgbaBuffer);
    const overlayCanvas = writeRgbaToCanvas(overlayRgba, response.width, response.height);

    return {
      width: response.width,
      height: response.height,
      disparity,
      minDisparity: response.stats.minDisparity,
      maxDisparity: response.stats.maxDisparity,
      medianDisparity: response.stats.medianDisparity,
      scaleMinDisparity: response.stats.scaleMinDisparity,
      scaleMaxDisparity: response.stats.scaleMaxDisparity,
      validRatio: response.stats.validRatio,
      overlayCanvas,
      leftCanvas: leftCopy,
      rightCanvas: rightCopy,
      colormap: options.colormap ?? 'turbo',
      viewMode: options.viewMode ?? 'left-overlay',
      swapLeftRight: options.swapLeftRight ?? false,
      sourceEyeWidth: sourceEyeW,
      sourceEyeHeight: sourceEyeH,
    };
  }

  rerenderOverlay(depthMap: StereoDepthMapData, options: StereoDepthRunOptions): StereoDepthMapData {
    return rerenderStereoDepthMap(depthMap, options);
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.leftScratch = null;
    this.rightScratch = null;
    // Reject all in-flight frame requests immediately so VisionEngine / FrameScheduler locks release in < 1ms
    this.rejectAllPending(
      new OnnxInferenceError(`stereo-depth:${this.variantId}`, 'closed', 'task is closed')
    );
    // Terminate the Web Worker immediately so in-flight WASM session.run() halts at the OS thread level
    try {
      this.worker.terminate();
    } catch {
      // Ignore worker termination errors
    }
  }
}

// ============================================================================
// Remote Weight Loader & Task Factory
// ============================================================================

const remoteStereoOnnxCache = new Map<StereoDepthVariantId, Uint8Array>();

export function clearRemoteStereoOnnxCache(): void {
  remoteStereoOnnxCache.clear();
}

async function fetchStereoOnnxBytes(
  url: string,
  fetchImpl: typeof fetch,
  signal?: AbortSignal
): Promise<{ bytes: Uint8Array; fromCache: boolean }> {
  const { bytes: rawBytes, fromCache } = await fetchModelBytesWithBrowserCache(url, {
    fetchImpl,
    signal,
  });
  return {
    bytes: stripOnnxGraphValueInfo(rawBytes),
    fromCache,
  };
}

export interface StereoDepthTaskOptions {
  variantId?: StereoDepthVariantId | string;
  modelUrl?: string;
  modelBytes?: Uint8Array;
  backendPreference?: StereoBackendPreference;
  recordAssetSource?: AssetSourceRecorder;
  fetchImpl?: typeof fetch;
  useWorker?: boolean;
  workerFactory?: () => StereoWorkerLike;
}

function shouldUseStereoWorkerByDefault(options: StereoDepthTaskOptions): boolean {
  if (typeof options.workerFactory === 'function') {
    return true;
  }
  if (typeof options.useWorker === 'boolean') {
    return options.useWorker;
  }
  if (typeof Worker !== 'function') {
    return false;
  }
  // When running under Vitest unit tests (which mock createOrtSession or 2D canvas in jsdom),
  // keep the synchronous/in-process engine unless workerFactory or useWorker: true is passed.
  const isVitest = Boolean((import.meta as { env?: { VITEST?: unknown } }).env?.VITEST);
  const isCreateOrtMocked = Boolean((createOrtSession as unknown as { mock?: unknown }).mock);
  return !isVitest && !isCreateOrtMocked;
}

export async function createStereoDepthTask(
  options: StereoDepthTaskOptions = {}
): Promise<StereoDepthTask> {
  // Preempt any earlier in-flight stereo model download or worker initialization
  cancelInFlightStereoInit();

  const variantId = (options.variantId || 'stereo-sad-baseline') as StereoDepthVariantId;
  const profile =
    STEREO_MODEL_PROFILES[variantId] ?? STEREO_MODEL_PROFILES['stereo-sad-baseline'];
  const backendPreference: StereoBackendPreference = options.backendPreference ?? 'auto';
  const useWorker = shouldUseStereoWorkerByDefault(options);

  if (profile.engineType === 'sad-cpu') {
    if (useWorker) {
      return StereoWorkerEngine.create({
        variantId: profile.variantId,
        profile,
        backendPreference,
        recordAssetSource: options.recordAssetSource,
        workerFactory: options.workerFactory,
      });
    }
    options.recordAssetSource?.('stereo-depth', 'browser-cache');
    options.recordAssetSource?.(`stereo-depth:${profile.variantId}`, 'browser-cache');
    return new StereoSadEngine();
  }

  if (profile.engineType === 'webgpu-wgsl') {
    if (useWorker) {
      return StereoWorkerEngine.create({
        variantId: profile.variantId,
        profile,
        backendPreference,
        recordAssetSource: options.recordAssetSource,
        workerFactory: options.workerFactory,
      });
    }
    options.recordAssetSource?.('stereo-depth', 'browser-cache');
    options.recordAssetSource?.(`stereo-depth:${profile.variantId}`, 'browser-cache');
    return StereoWgslEngine.create(backendPreference);
  }

  const buildStereoEngine = async (
    onnxBytes: Uint8Array,
    primarySource: AssetDeliverySource
  ): Promise<StereoDepthTask> => {
    if (useWorker) {
      return StereoWorkerEngine.create({
        variantId: profile.variantId,
        profile,
        onnxBytes,
        backendPreference,
        primarySource,
        recordAssetSource: options.recordAssetSource,
        workerFactory: options.workerFactory,
      });
    }
    const splitResult = splitLargeOnnxConcatNodes(onnxBytes, 4);
    const preparedBytes = splitResult.splitCount > 0 ? splitResult.bytes : onnxBytes;
    const delegate = resolveStereoOrtDelegate(
      profile,
      preparedBytes,
      DEFAULT_WEBGPU_MAX_STORAGE_BUFFERS_PER_STAGE,
      backendPreference
    );
    const { session, backend } = await createOrtSession(preparedBytes, delegate, {
      recordAssetSource: options.recordAssetSource,
      modelId: 'stereo-depth',
      primarySource,
      variantId: profile.variantId,
      graphOptimizationLevel: delegate === 'WASM' ? 'disabled' : 'all',
    });
    const staticCompat =
      backendPreference === 'auto' && delegate === 'WASM' && profile.preferWebGpu
        ? inspectStereoOnnxWebGpuCompatibility(preparedBytes)
        : null;
    const concatNote =
      splitResult.splitCount > 0 && backend.configuredProviders.includes('webgpu')
        ? `Concat分割(${splitResult.splitCount}箇所・最大${splitResult.originalMaxConcatInputs}→4入力ツリー化)でWebGPU有効化`
        : null;
    const adjustedBackend: OnnxBackendInfo = {
      ...backend,
      requestedProvider:
        backendPreference === 'wasm'
          ? 'wasm'
          : backendPreference === 'webgpu'
            ? 'webgpu'
            : backend.requestedProvider,
      fallbackReason: backend.fallbackReason ?? staticCompat?.reason ?? concatNote,
    };
    const fallbackToWasm: StereoWasmRecoveryFactory | undefined =
      delegate === 'WebGPU' && adjustedBackend.configuredProviders.includes('webgpu')
        ? () =>
            createOrtSession(preparedBytes, 'WASM', {
              recordAssetSource: options.recordAssetSource,
              modelId: 'stereo-depth',
              primarySource,
              variantId: profile.variantId,
              graphOptimizationLevel: 'disabled',
            })
        : undefined;
    return new StereoOnnxEngine(session, profile, adjustedBackend, fallbackToWasm);
  };

  const localBytes = options.modelBytes ?? getLocalStereoOnnxBytes(profile.variantId);

  if (localBytes) {
    const preparedLocal = stripOnnxGraphValueInfo(localBytes);
    return buildStereoEngine(preparedLocal, 'browser-cache');
  }

  interface CandidateSource {
    url: string;
    source: AssetDeliverySource;
  }

  const candidates: CandidateSource[] = [];

  if (options.modelUrl) {
    candidates.push({ url: options.modelUrl, source: 'external' });
    if (
      options.modelUrl === profile.defaultModelUrl &&
      profile.fallbackModelUrl &&
      profile.fallbackModelUrl !== options.modelUrl
    ) {
      candidates.push({ url: profile.fallbackModelUrl, source: 'external-fallback' });
    }
  } else if (profile.onlineDelivery === 'cors-proxy') {
    if (!profile.corsProxyUrl) {
      throw new Error(
        `${profile.variantId} の ONNX モデル (${profile.fileName}) の取得URLが未設定です。右パネル「モデル調整」からローカルの .onnx ファイルを選択してください。`
      );
    }
    candidates.push({ url: profile.corsProxyUrl, source: 'cors-proxy' });
  } else if (profile.defaultModelUrl) {
    candidates.push({ url: profile.defaultModelUrl, source: 'external' });
    if (profile.fallbackModelUrl && profile.fallbackModelUrl !== profile.defaultModelUrl) {
      candidates.push({ url: profile.fallbackModelUrl, source: 'external-fallback' });
    }
  } else {
    throw new Error(
      `${profile.variantId} の ONNX モデルファイル (${profile.fileName}) が未読み込みです。右パネル「モデル調整」からローカルの .onnx ファイルを選択するか、「SAD Baseline (内蔵・即時実行)」またはオンライン対応の「LightStereo-S / MobileStereoNet-2D」へ切り替えてください。`
    );
  }

  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  if (!options.modelUrl) {
    const cachedRemote = remoteStereoOnnxCache.get(profile.variantId);
    if (cachedRemote) {
      return buildStereoEngine(cachedRemote, 'browser-cache');
    }
  }

  const controller = typeof AbortController === 'function' && useWorker ? new AbortController() : null;
  activeInitController = controller;

  let loadedBytes: Uint8Array | null = null;
  let resolvedSource: AssetDeliverySource = candidates[0].source;
  const failureMessages: string[] = [];
  let lastError: unknown;

  try {
    for (const candidate of candidates) {
      if (controller?.signal.aborted) {
        throw new Error('Stereo model download aborted due to variant switch');
      }
      try {
        const fetched = await fetchStereoOnnxBytes(
          candidate.url,
          fetchImpl,
          controller?.signal
        );
        loadedBytes = fetched.bytes;
        resolvedSource = fetched.fromCache ? 'browser-cache' : candidate.source;
        break;
      } catch (err) {
        if (controller?.signal.aborted) {
          throw err;
        }
        lastError = err;
        failureMessages.push(err instanceof Error ? err.message : String(err));
      }
    }
  } finally {
    if (activeInitController === controller) {
      activeInitController = null;
    }
  }

  if (!loadedBytes) {
    if (candidates.length > 1) {
      throw new Error(
        `${profile.variantId} の ONNX モデル (${profile.fileName}) のオンライン取得に失敗しました (${failureMessages.join(' / fallback: ')})。ネットワーク状態を確認して再試行するか、右パネル「モデル調整」からローカルの .onnx ファイルを選択、または「SAD Baseline」へ切り替えてください。`,
        { cause: lastError }
      );
    }
    throw new Error(
      `${profile.variantId} の ONNX モデル (${profile.fileName}) の読み込みに失敗しました (${failureMessages[0] ?? 'unknown error'})。右パネル「モデル調整」からローカルの .onnx ファイルを選択するか、「SAD Baseline」へ切り替えてください。`,
      { cause: lastError }
    );
  }

  if (!options.modelUrl) {
    remoteStereoOnnxCache.clear();
    remoteStereoOnnxCache.set(profile.variantId, loadedBytes);
  }

  return buildStereoEngine(loadedBytes, resolvedSource);
}
