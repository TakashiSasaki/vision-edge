export type ModelType = 'selfie-segmenter' | 'yolo-seg' | 'mediapipe' | 'segformer';

export type BackendOption = 'auto' | 'webgpu' | 'wasm-simd' | 'webgl';

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

export interface BenchmarkStats {
  inferenceTimeMs: number;
  fps: number;
  objectCount: number;
  backend: string;
  resolution: string;
  processedFrames: number;
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
  tagline: string;
  runtime: string;
  outputType: string;
  classesCount: number;
  datasetName: string;
  typicalFps: string;
  description: string;
  pros: string[];
}
