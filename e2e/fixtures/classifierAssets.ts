// Explicit --mode e2e only. No acquisition: the SDK is also the existing CI
// double. Production must never import this marker/resolver.
export * from '../../src/services/imageClassifierAssets.ts';
import { classifierAsset } from '../../src/services/imageClassifierAssets.ts';
import type { ImageClassifierVariantId } from '../../src/types/classification';
export async function resolveClassifierAsset(variant: ImageClassifierVariantId) {
  const entry = classifierAsset(variant);
  return {
    modelId: 'image-classifier' as const,
    variantId: variant,
    modelAssetBuffer: new Uint8Array(entry.expectedSize),
    modelAssetPath: entry.upstreamUrl,
    source: 'browser-cache' as const,
  };
}
export const CLASSIFIER_CI_ONLY = 'CLASSIFIER_CI_ONLY';
