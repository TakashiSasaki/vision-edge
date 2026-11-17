import type { ImageClassifierVariantId } from '../types/classification.ts';
import { IMAGE_CLASSIFIER_ASSETS } from './classifierAssetManifest.ts';
import { fetchModelBytesWithBrowserCache } from './browserModelCache.ts';
import type { ContainerModelStatusEntry } from './modelAssetResolver.ts';

export { IMAGE_CLASSIFIER_ASSETS } from './classifierAssetManifest.ts';

export function classifierAsset(variantId: string = 'lite0') {
  if (typeof variantId !== 'string' || !Object.hasOwn(IMAGE_CLASSIFIER_ASSETS, variantId)) {
    throw new RangeError(`Unsupported classifier variant: ${String(variantId)}`);
  }
  return IMAGE_CLASSIFIER_ASSETS[variantId as ImageClassifierVariantId];
}

export class ClassificationAssetsBlockedError extends Error {
  constructor(
    public readonly variantId: ImageClassifierVariantId,
    reason = 'classifier model download or Web Crypto SHA-256 verification failed; unverified models cannot be initialized'
  ) {
    super(`image-classifier:${variantId} BLOCKED: ${reason}`);
    this.name = 'ClassificationAssetsBlockedError';
  }
}

export interface ClassifierAssetResolution {
  modelId: 'image-classifier';
  variantId: ImageClassifierVariantId;
  modelAssetBuffer: Uint8Array;
  modelAssetPath: string;
  source: 'browser-cache' | 'external';
}

export async function resolveClassifierAsset(
  variantId: ImageClassifierVariantId,
  options: { fetchImpl?: typeof fetch } = {}
): Promise<ClassifierAssetResolution> {
  const entry = classifierAsset(variantId);
  try {
    const { bytes, fromCache } = await fetchModelBytesWithBrowserCache(entry.upstreamUrl, {
      expectedSize: entry.expectedSize,
      expectedSha256: entry.expectedSha256,
      fetchImpl: options.fetchImpl,
    });
    return {
      modelId: entry.modelId,
      variantId,
      modelAssetBuffer: bytes,
      modelAssetPath: entry.upstreamUrl,
      source: fromCache ? 'browser-cache' : 'external',
    };
  } catch (cause) {
    throw new ClassificationAssetsBlockedError(
      variantId,
      cause instanceof Error ? cause.message : String(cause)
    );
  }
}

export function classifierAssetStatuses(states: Record<string, ContainerModelStatusEntry> = {}) {
  return Object.fromEntries(
    Object.entries(IMAGE_CLASSIFIER_ASSETS).map(([key, entry]) => {
      const state = states[`image-classifier:${key}`];
      return [
        key,
        {
          ...entry,
          status: state?.status ?? 'unavailable',
          sha256: state?.sha256 ?? null,
          size: state?.size ?? null,
          filePresent: state?.filePresent ?? null,
          integrity: state?.integrity ?? 'unknown',
          error: state?.error ?? null,
          acquisitionEnabled: true as const,
        },
      ];
    })
  );
}

export type ClassifierAssetStatuses = ReturnType<typeof classifierAssetStatuses>;
