export type ModelType = 'selfie-segmenter' | 'yolo-seg' | 'mediapipe' | 'segformer';

export type RunnableModelType = 'selfie-segmenter' | 'mediapipe' | 'yolo-seg' | 'segformer';

export type SelfieVariantId = 'general' | 'landscape';
export type MediapipeVariantId = 'lite0' | 'lite2' | 'ssd-mobilenet-v2';
export type YoloSegVariantId = 'yolov8n-seg' | 'yolov8s-seg';
export type SegFormerVariantId = 'b0' | 'b1' | 'b2';

export type ModelVariantId = SelfieVariantId | MediapipeVariantId | YoloSegVariantId | SegFormerVariantId;

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

export type ModelInferenceParamsMap = {
  'selfie-segmenter': SelfieInferenceParams;
  'mediapipe': MediapipeInferenceParams;
  'yolo-seg': YoloSegInferenceParams;
  'segformer': SegFormerInferenceParams;
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
}

export interface OnnxBackendInfo {
  runtime:'onnxruntime-web';
  requestedProvider:'auto'|'webgpu'|'wasm'|'unreported';
  configuredProviders:readonly string[];
  fallbackReason:string|null;
  actualBackend:'unmeasured';
}

export type FrameResult =
  | ({ status: 'ready' } & DetectionFrame)
  | { status: 'not-ready'; model: RunnableModelType; requestId: number };

