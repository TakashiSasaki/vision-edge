import type { ImageClassifierVariantId } from '../types/classification.ts';
import { IMAGE_CLASSIFIER_ASSETS } from './classifierAssetManifest.ts';
import { fetchContainerCacheStatus, isContainerModelCacheEnabled, type ContainerModelStatusEntry } from './modelAssetResolver.ts';
export { IMAGE_CLASSIFIER_ASSETS } from './classifierAssetManifest.ts';
export function classifierAsset(variantId: string = 'lite0') {
  if (typeof variantId !== 'string' || !Object.hasOwn(IMAGE_CLASSIFIER_ASSETS, variantId)) throw new RangeError(`Unsupported classifier variant: ${String(variantId)}`);
  return IMAGE_CLASSIFIER_ASSETS[variantId as ImageClassifierVariantId];
}
export class ClassificationAssetsBlockedError extends Error {
  constructor(public readonly variantId: ImageClassifierVariantId, reason = 'verified container cache API unavailable; static/preview serving cannot initialize this asset') {
    super(`image-classifier:${variantId} BLOCKED: ${reason}`);
    this.name = 'ClassificationAssetsBlockedError';
  }
}
export interface ClassifierAssetResolution {
  modelId: 'image-classifier';
  variantId: ImageClassifierVariantId;
  modelAssetPath: string;
  source: 'container-cache';
}
// Only a matching server catalog can authorize the pinned local URL. No direct
// upstream fallback: the server verifies the exact bytes before serving them.
export async function resolveClassifierAsset(variantId: ImageClassifierVariantId): Promise<ClassifierAssetResolution> {
  const entry = classifierAsset(variantId);
  if (!isContainerModelCacheEnabled()) throw new ClassificationAssetsBlockedError(variantId);
  const status = await fetchContainerCacheStatus(true);
  const row = status?.enabled && status.variants?.[`image-classifier:${variantId}`];
  if (!row || row.id !== entry.modelId || row.variantId !== entry.variantId || row.file !== entry.file || row.localUrl !== entry.localUrl || row.upstreamUrl !== entry.upstreamUrl || row.license !== entry.license || row.expectedSize !== entry.expectedSize || row.expectedSha256 !== entry.expectedSha256) throw new ClassificationAssetsBlockedError(variantId, 'container asset identity/pins missing or mismatched');
  if (row.status === 'cached') {
    if (row.filePresent !== true || row.integrity !== 'pinned' || row.size !== entry.expectedSize || row.sha256 !== entry.expectedSha256 || row.error !== null) throw new ClassificationAssetsBlockedError(variantId, 'cached asset failed pinned integrity verification');
  } else if (row.error !== null || row.size !== null || row.sha256 !== null || (row.status === 'unavailable' ? row.filePresent !== false || row.integrity !== 'not-present' : row.status !== 'downloading' || row.integrity !== 'pending')) {
    throw new ClassificationAssetsBlockedError(variantId, 'container acquisition failed or cache is unverified');
  }
  return { modelId: entry.modelId, variantId, modelAssetPath: entry.localUrl, source: 'container-cache' };
}
export function classifierAssetStatuses(states: Record<string, ContainerModelStatusEntry> = {}) {
  return Object.fromEntries(Object.entries(IMAGE_CLASSIFIER_ASSETS).map(([key, entry]) => {
    const state = states[`image-classifier:${key}`];
    return [key, { ...entry, status: state?.status ?? 'unavailable',
      sha256: state?.sha256 ?? null, size: state?.size ?? null,
      filePresent: state?.filePresent ?? null, integrity: state?.integrity ?? 'unknown',
      error: state?.error ?? null, acquisitionEnabled: true as const }];
  }));
}
export type ClassifierAssetStatuses = ReturnType<typeof classifierAssetStatuses>;
