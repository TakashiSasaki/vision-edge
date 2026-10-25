export type ModelType = 'selfie-segmenter' | 'yolo-seg' | 'mediapipe' | 'segformer';

export type RunnableModelType = 'selfie-segmenter' | 'mediapipe' | 'yolo-seg';

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

export interface ModelMetadata {
  id: ModelType;
  name: string;
  implemented: boolean;
  runtime: string;
  outputType: string;
  description: string;
}

export type ModelState =
  | { status: 'idle'; model: null; requestId: number }
  | { status: 'loading'; model: RunnableModelType; requestId: number }
  | { status: 'ready'; model: RunnableModelType; requestId: number; delegate: 'GPU' }
  | { status: 'error'; model: RunnableModelType; requestId: number; stage: 'wasm' | 'task'; cause: unknown };

export interface DetectionFrame {
  model: RunnableModelType;
  requestId: number;
  objects: DetectedObject[];
}

export type FrameResult =
  | ({ status: 'ready' } & DetectionFrame)
  | { status: 'not-ready'; model: RunnableModelType; requestId: number };
