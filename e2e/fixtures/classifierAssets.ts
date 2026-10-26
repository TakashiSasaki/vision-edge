// Explicit --mode e2e only. No acquisition: the SDK is also the existing CI
// double. Production must never import this marker/resolver.
export * from '../../src/services/imageClassifierAssets.ts';
import { classifierAsset } from '../../src/services/imageClassifierAssets.ts';
import type { ImageClassifierVariantId } from '../../src/types/classification';
export async function resolveClassifierAsset(variant: ImageClassifierVariantId) {
  return { modelId: 'image-classifier' as const, variantId: variant, modelAssetPath: classifierAsset(variant).upstreamUrl, source: 'external' as const };
}
export const CLASSIFIER_CI_ONLY = 'CLASSIFIER_CI_ONLY';
