import type { ImageClassifierVariantId } from '../types/classification';

// Official sample references only. No bytes, hashes or exact-weight license have
// been verified. Keep these outside the acquisition/redistribution catalog.
export const IMAGE_CLASSIFIER_ASSETS = {
  lite0: {
    modelId: 'image-classifier', variantId: 'lite0', file: 'efficientnet_lite0.tflite',
    upstreamUrl: 'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite',
  },
  lite2: {
    modelId: 'image-classifier', variantId: 'lite2', file: 'efficientnet_lite2.tflite',
    upstreamUrl: 'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite2/float32/1/efficientnet_lite2.tflite',
  },
} as const;
export function classifierAsset(variantId: string = 'lite0') {
  if (typeof variantId !== 'string' || !Object.hasOwn(IMAGE_CLASSIFIER_ASSETS, variantId)) throw new RangeError(`Unsupported classifier variant: ${String(variantId)}`);
  return IMAGE_CLASSIFIER_ASSETS[variantId as ImageClassifierVariantId];
}
export class ClassificationAssetsBlockedError extends Error {
  constructor(public readonly variantId: ImageClassifierVariantId) {
    super(`image-classifier:${variantId} BLOCKED: exact-weight license, bytes/SHA256 and model metadata require review before acquisition or redistribution`);
    this.name = 'ClassificationAssetsBlockedError';
  }
}
export interface ClassifierAssetResolution {
  modelId: 'image-classifier';
  variantId: ImageClassifierVariantId;
  modelAssetPath: string;
  source: 'external';
}
// A future reviewed asset policy must replace this gate. No I/O, fallback,
// prefetch or fabricated pins occur here; even the SDK is not started by default.
export async function resolveClassifierAsset(variantId: ImageClassifierVariantId): Promise<ClassifierAssetResolution> {
  classifierAsset(variantId);
  throw new ClassificationAssetsBlockedError(variantId);
}
export function classifierAssetStatuses() {
  return Object.fromEntries(Object.entries(IMAGE_CLASSIFIER_ASSETS).map(([key, entry]) => [key, {
    ...entry, status: 'blocked' as const, reason: 'exact-weight verification and acquisition approval pending',
    sha256: null, size: null, license: null, localUrl: null, acquisitionEnabled: false as const,
  }]));
}

export type ClassifierAssetStatuses = ReturnType<typeof classifierAssetStatuses>;
