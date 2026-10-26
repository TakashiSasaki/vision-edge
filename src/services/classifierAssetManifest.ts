// Pins from independently inspected local files, downloaded from these exact
// URLs according to the owner. No independent remote hash comparison occurred.
const lite0Hash = '6c7ab0a6e5dcbf38a8c33b960996a55a3b4300b36a018c4545801de3a3c8bde0';
const lite2Hash = '94b6c84a5d6de20f932558de213b50e0338dcc06ab83d8d64b12128a476b4b83';
function asset(variantId: 'lite0' | 'lite2', expectedSize: number, expectedSha256: string, inputSize: number) {
  const file = `efficientnet_${variantId}.tflite`;
  return Object.freeze({
    modelId: 'image-classifier' as const, variantId, file,
    upstreamUrl: `https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_${variantId}/float32/1/${file}`,
    localUrl: `/api/model-cache/models/${file}?sha256=${expectedSha256}`,
    expectedSize, expectedSha256, license: 'Apache-2.0', author: 'MediaPipe',
    licenseNote: 'Exact-file ModelMetadata license declaration; see /licenses/classifier-models.txt',
    inputShape: [1, inputSize, inputSize, 3] as const, outputShape: [1, 1000] as const,
    tensorType: 'FLOAT32', labels: 'labels_without_background.txt', labelCount: 1000,
  });
}
export const IMAGE_CLASSIFIER_ASSETS = Object.freeze({
  lite0: asset('lite0', 18_582_189, lite0Hash, 224),
  lite2: asset('lite2', 24_325_765, lite2Hash, 260),
});
