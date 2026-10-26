import type { ModelType } from './vision';

export type ImageClassifierVariantId = 'lite0' | 'lite2';
export type ClassificationDelegate = 'GPU' | 'CPU';
export type ModelFamilyType = ModelType | 'image-classifier';

// SDK output IDs are preserved. They are not ranking positions or translated IDs.
export interface ImageCategory {
  readonly index: number;
  readonly categoryName: string;
  readonly displayName: string;
  readonly score: number;
}
export interface ImageClassifications {
  readonly headIndex: number;
  readonly headName: string;
  readonly categories: readonly ImageCategory[];
}
export interface ClassificationOptions {
  scoreThreshold?: number;
  maxResults?: number;
}
export type ClassificationState =
  | { status: 'idle'; model: null; requestId: number }
  | { status: 'loading'; model: 'image-classifier'; variantId: ImageClassifierVariantId; requestedDelegate: ClassificationDelegate; requestId: number }
  | { status: 'ready'; model: 'image-classifier'; variantId: ImageClassifierVariantId; requestedDelegate: ClassificationDelegate; sdkDelegate: ClassificationDelegate | 'unreported'; actualBackend: 'unmeasured'; requestId: number }
  | { status: 'error'; model: 'image-classifier'; variantId: ImageClassifierVariantId; requestedDelegate: ClassificationDelegate; requestId: number; stage: 'asset' | 'wasm' | 'task'; cause: unknown };
export interface WholeImageClassificationFrame {
  readonly status: 'ready';
  readonly kind: 'whole-image-classification';
  readonly model: 'image-classifier';
  readonly variantId: ImageClassifierVariantId;
  readonly requestId: number;
  readonly frameId: number;
  readonly outputGeneration: number;
  readonly timestampMs: number;
  readonly sdkTimestampMs?: number;
  readonly classifications: readonly ImageClassifications[];
}
export type ClassificationFrameResult = WholeImageClassificationFrame
  | { status: 'not-ready'; model: 'image-classifier'; requestId: number };
