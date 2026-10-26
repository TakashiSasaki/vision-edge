import { afterEach, expect, it, vi } from 'vitest';
import { IMAGE_CLASSIFIER_ASSETS, classifierAssetStatuses, resolveClassifierAsset, ClassificationAssetsBlockedError } from './imageClassifierAssets';
import { setContainerModelCacheEnabledForTest } from './modelAssetResolver';
const variants = ['lite0', 'lite2'] as const;
function row(variant: typeof variants[number], overrides = {}) {
  const e = IMAGE_CLASSIFIER_ASSETS[variant];
  return { ...e, id: e.modelId, status: 'cached', size: e.expectedSize, sha256: e.expectedSha256, filePresent: true, integrity: 'pinned', error: null, ...overrides };
}
function status(variant: typeof variants[number], value: unknown) {
  setContainerModelCacheEnabledForTest(true);
  return vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ enabled: true, models: {}, wasm: {}, variants: { ['image-classifier:'+variant]: value } }), { headers: { 'Content-Type': 'application/json' } }));
}
afterEach(() => { setContainerModelCacheEnabledForTest(null); vi.restoreAllMocks(); });
it.each(variants)('%s has exact inspected metadata, not an invented acquisition result', variant => {
  const e = IMAGE_CLASSIFIER_ASSETS[variant];
  expect(e.upstreamUrl).toBe(`https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_${variant}/float32/1/efficientnet_${variant}.tflite`);
  expect(e.expectedSize).toBe(variant === 'lite0' ? 18582189 : 24325765);
  expect(e.expectedSha256).toBe(variant === 'lite0' ? '6c7ab0a6e5dcbf38a8c33b960996a55a3b4300b36a018c4545801de3a3c8bde0' : '94b6c84a5d6de20f932558de213b50e0338dcc06ab83d8d64b12128a476b4b83');
  expect(e.inputShape).toEqual([1, variant === 'lite0' ? 224 : 260, variant === 'lite0' ? 224 : 260, 3]); expect(e.outputShape).toEqual([1,1000]);
  expect(classifierAssetStatuses()[variant]).toMatchObject({ status: 'unavailable', size: null, sha256: null, filePresent: null, license: 'Apache-2.0', acquisitionEnabled: true });
});
it.each(variants)('%s resolves only to the matching pinned container path', async variant => {
  const fetch = status(variant, row(variant));
  expect(await resolveClassifierAsset(variant)).toEqual({ modelId: 'image-classifier', variantId: variant, source: 'container-cache', modelAssetPath: IMAGE_CLASSIFIER_ASSETS[variant].localUrl });
  expect(fetch).toHaveBeenCalledTimes(1); expect(fetch.mock.calls[0][0]).toBe('/api/model-cache/status');
});
for (const variant of variants) it.each([
  { id: 'mediapipe' }, { variantId: variant === 'lite0' ? 'lite2' : 'lite0' }, { file: 'wrong.tflite' }, { localUrl: IMAGE_CLASSIFIER_ASSETS[variant].upstreamUrl }, { upstreamUrl: 'https://wrong.invalid/model' }, { license: 'unknown' }, { expectedSize: 1 }, { expectedSha256: '0'.repeat(64) }, { sha256: '0'.repeat(64) }, { size: 1 }, { filePresent: false }, { integrity: 'self-recorded' }, { error: 'bad cache' }, { status: 'unknown' },
])('%s rejects mismatched/unverified cached state %j', async overrides => {
  status(variant, row(variant, overrides)); await expect(resolveClassifierAsset(variant)).rejects.toBeInstanceOf(ClassificationAssetsBlockedError);
});
it.each(variants)('%s allows only verified server acquisition states; no binary fetch by resolver', async variant => {
  const fetch = status(variant, row(variant, { status: 'unavailable', size: null, sha256: null, filePresent: false, integrity: 'not-present' }));
  expect((await resolveClassifierAsset(variant)).modelAssetPath).toBe(IMAGE_CLASSIFIER_ASSETS[variant].localUrl); expect(fetch).toHaveBeenCalledTimes(1);
});
it('static/preview mode blocks without network; missing API never enables upstream fallback', async () => {
  setContainerModelCacheEnabledForTest(false); const fetch = vi.spyOn(globalThis, 'fetch');
  await expect(resolveClassifierAsset('lite0')).rejects.toThrow('API unavailable'); expect(fetch).not.toHaveBeenCalled();
  setContainerModelCacheEnabledForTest(true); fetch.mockResolvedValue(new Response('<html>SPA</html>'));
  await expect(resolveClassifierAsset('lite0')).rejects.toThrow('identity/pins'); expect(fetch).toHaveBeenCalledTimes(1);
});
