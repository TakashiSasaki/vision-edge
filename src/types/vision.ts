export type ModelType = 'selfie-segmenter' | 'yolo-seg' | 'mediapipe' | 'segformer' | 'mirage' | 'stereo-depth';

export type RunnableModelType = 'selfie-segmenter' | 'mediapipe' | 'yolo-seg' | 'segformer' | 'mirage' | 'stereo-depth';

export type SelfieVariantId = 'general' | 'landscape';
export type MediapipeVariantId = 'lite0' | 'lite2';
export type YoloSegVariantId = 'yolov8n-seg' | 'yolov8s-seg';
export type SegFormerVariantId = 'b0' | 'b1' | 'b2';
export type MirageVariantId =
  | 'mirage-51m'
  | 'mirage-onnx-exp-wasm'
  | 'mirage-onnx-exp-webgpu';
export type StereoDepthVariantId =
  | 'stereo-sad-baseline'
  | 'stereo-webgpu-wgsl'
  | 'lightstereo-s'
  | 'mobilestereonet-2d'
  | 'hitnet-sf'
  | 'crestereo-init-iter2'
  | 'crestereo-combined-iter5'
  | 'custom-onnx';

export type ModelVariantId =
  | SelfieVariantId
  | MediapipeVariantId
  | YoloSegVariantId
  | SegFormerVariantId
  | MirageVariantId
  | StereoDepthVariantId;

export interface ModelVariantMetadata {
  id: string;
  name: string;
  description: string;
  fileName: string;
  upstreamUrl: string;
  localUrl: string;
  isDefault?: boolean;
}

// An immutable browser video frame is preferred when the platform supports it.
export type InferenceInput = HTMLVideoElement | VideoFrame | HTMLCanvasElement;

export type CameraFacingMode = 'user' | 'environment';

export interface BoundingBox {
  x: number;      // 0 to 1 normalized
  y: number;      // 0 to 1 normalized
  width: number;  // 0 to 1 normalized
  height: number; // 0 to 1 normalized
}

export interface DetectedObject {
  id: string;
  label: string;
  labelJa: string;
  category: string;
  confidence: number;
  bbox: BoundingBox;
  color: string;
  maskPolygon?: Array<{ x: number; y: number }>;
  maskCanvas?: HTMLCanvasElement;
  semanticClassId?: number;
  areaFraction?: number;
  confidenceKind?: 'semantic-softmax-mean' | 'semantic-softmax-pixel';
}

export interface TTSConfig {
  enabled: boolean;
  autoReadout: boolean;
  cooldownSec: number;
  rate: number;
  pitch: number;
  volume: number;
  voiceURI: string;
}

export interface RenderOptions {
  showBoxes: boolean;
  showMasks: boolean;
  showLabels: boolean;
  showConfidence: boolean;
  maskOpacity: number;
  confidenceThreshold: number;
}

export interface SelfieInferenceParams {
  confidenceThreshold: number;
  minAreaPercent: number;
}

export interface MediapipeInferenceParams {
  confidenceThreshold: number;
  maxDetections: number;
  bboxSmoothing: number;
}

export interface YoloSegInferenceParams {
  confidenceThreshold: number;
  iouThreshold: number;
  maxDetections: number;
  maskThreshold: number;
  bboxSmoothing: number;
}

export interface SegFormerInferenceParams {
  confidenceThreshold: number;
  minAreaPercent: number;
}

export type MirageAnalysisMode = 'fast' | 'full';

export interface MirageInferenceParams {
  confidenceThreshold: number;
  maxDetections: number;
  iouThreshold: number;
  analysisMode: MirageAnalysisMode;
  mirageLanguage?: string;
  includeOcr: boolean;
}

export type StereoColormap = 'turbo' | 'viridis' | 'inferno' | 'grayscale';
export type StereoViewMode = 'left-overlay' | 'split-3pane' | 'depth-only';
export type StereoBackendPreference = 'auto' | 'webgpu' | 'wasm';

export interface StereoDepthInferenceParams {
  confidenceThreshold: number;
  backendPreference: StereoBackendPreference;
  maxDisparity: number;
  sadWindowSize: 5 | 7 | 9;
  autoNormalizeDisparity: boolean;
  minDisparityPx: number;
  maxDisparityPx: number;
  swapLeftRight: boolean;
  cameraRotate180: boolean;
  colormap: StereoColormap;
  viewMode: StereoViewMode;
  baselineMm: number;
  focalLengthPx: number;
  showCalibrationDepth: boolean;
}

export interface StereoDepthMapData {
  width: number;
  height: number;
  disparity: Float32Array;
  minDisparity: number;
  maxDisparity: number;
  medianDisparity: number;
  scaleMinDisparity: number;
  scaleMaxDisparity: number;
  validRatio: number;
  overlayCanvas: HTMLCanvasElement;
  leftCanvas: HTMLCanvasElement;
  rightCanvas: HTMLCanvasElement;
  colormap: StereoColormap;
  viewMode: StereoViewMode;
  swapLeftRight: boolean;
  rotate180?: boolean;
  sourceEyeWidth: number;
  sourceEyeHeight: number;
}

export interface StereoDepthProbeResult {
  normX: number;
  normY: number;
  pixelX: number;
  pixelY: number;
  disparityPx: number;
  valid: boolean;
  depthMeters: number | null;
}

export interface StereoPinnedProbe extends StereoDepthProbeResult {
  id: string;
  label: string;
  variantId: string;
}

export type ModelInferenceParamsMap = {
  'selfie-segmenter': SelfieInferenceParams;
  'mediapipe': MediapipeInferenceParams;
  'yolo-seg': YoloSegInferenceParams;
  'segformer': SegFormerInferenceParams;
  'mirage': MirageInferenceParams;
  'stereo-depth': StereoDepthInferenceParams;
};

export interface ModelParameterConfig<T = unknown> {
  inference: T;
  render: RenderOptions;
}

export type PerModelParameterConfig = {
  [K in RunnableModelType]: ModelParameterConfig<ModelInferenceParamsMap[K]>;
};

export const DEFAULT_MODEL_PARAMS: PerModelParameterConfig = {
  'selfie-segmenter': {
    inference: { confidenceThreshold: 0.35, minAreaPercent: 0.5 },
    render: { showBoxes: true, showMasks: true, showLabels: true, showConfidence: true, maskOpacity: 0.35, confidenceThreshold: 0.35 }
  },
  'mediapipe': {
    inference: { confidenceThreshold: 0.35, maxDetections: 10, bboxSmoothing: 0.20 },
    render: { showBoxes: true, showMasks: false, showLabels: true, showConfidence: true, maskOpacity: 0.35, confidenceThreshold: 0.35 }
  },
  'yolo-seg': {
    inference: { confidenceThreshold: 0.35, iouThreshold: 0.45, maxDetections: 20, maskThreshold: 0.50, bboxSmoothing: 0.20 },
    render: { showBoxes: true, showMasks: true, showLabels: true, showConfidence: true, maskOpacity: 0.35, confidenceThreshold: 0.35 }
  },
  'segformer': {
    inference: { confidenceThreshold: 0.35, minAreaPercent: 0.5 },
    render: { showBoxes: true, showMasks: true, showLabels: true, showConfidence: true, maskOpacity: 0.35, confidenceThreshold: 0.35 }
  },
  'mirage': {
    inference: { confidenceThreshold: 0.25, maxDetections: 15, iouThreshold: 0.45, analysisMode: 'fast', mirageLanguage: 'en', includeOcr: true },
    render: { showBoxes: true, showMasks: true, showLabels: true, showConfidence: true, maskOpacity: 0.35, confidenceThreshold: 0.25 }
  },
  'stereo-depth': {
    inference: {
      confidenceThreshold: 0.35,
      backendPreference: 'auto',
      maxDisparity: 64,
      sadWindowSize: 7,
      autoNormalizeDisparity: true,
      minDisparityPx: 0,
      maxDisparityPx: 64,
      swapLeftRight: false,
      cameraRotate180: true,
      colormap: 'turbo',
      viewMode: 'left-overlay',
      baselineMm: 60,
      focalLengthPx: 500,
      showCalibrationDepth: true,
    },
    render: { showBoxes: false, showMasks: true, showLabels: true, showConfidence: false, maskOpacity: 0.72, confidenceThreshold: 0.35 }
  }
};

export interface ModelMetadata {
  id: ModelType;
  name: string;
  implemented: boolean;
  runtime: string;
  outputType: string;
  description: string;
  license?: string;
  licenseNote?: string;
  upstreamUrl?: string;
  localCachePath?: string;
  defaultVariantId: string;
  variants: ModelVariantMetadata[];
}

export type ModelState =
  | { status: 'idle'; model: null; variantId?: string; requestId: number }
  | { status: 'loading'; model: RunnableModelType; variantId?: string; requestId: number }
  | { status: 'ready'; model: RunnableModelType; variantId?: string; requestId: number; delegate?: 'GPU'; backend?: OnnxBackendInfo }
  | { status: 'error'; model: RunnableModelType; variantId?: string; requestId: number; stage: 'wasm' | 'task'; cause: unknown };

export interface DetectionFrame {
  model: RunnableModelType;
  requestId: number;
  objects: DetectedObject[];
  semanticMask?: HTMLCanvasElement;
  depthMap?: StereoDepthMapData;
}

export interface OnnxBackendInfo {
  runtime:'onnxruntime-web' | 'stereo-sad-cpu' | 'stereo-webgpu-wgsl';
  requestedProvider:'auto'|'webgpu'|'wasm'|'unreported'|'cpu';
  configuredProviders:readonly string[];
  fallbackReason:string|null;
  actualBackend:'unmeasured';
}

export type FrameResult =
  | ({ status: 'ready' } & DetectionFrame)
  | { status: 'not-ready'; model: RunnableModelType; requestId: number };

