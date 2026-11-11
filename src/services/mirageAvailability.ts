/** No faithful Mirage ONNX artifact has passed reference-output validation yet. */
export const MIRAGE_WEBGPU_UNAVAILABLE_CODE = 'MIRAGE_ONNX_NOT_VALIDATED';

export const MIRAGE_WEBGPU_UNAVAILABLE_REASON =
  'Mirage ONNX is unavailable: a faithful export of the learned model has not been validated. Use the original Mirage .cact WASM variant.';

export const MIRAGE_WEBGPU_UNAVAILABLE_REASON_JA =
  'ONNX版は元モデルとの一致を未検証のため利用できません。公式 .cact のWASM版を選択してください。';

export function mirageWebGpuUnavailableError(): Error & { code: string } {
  return Object.assign(new Error(MIRAGE_WEBGPU_UNAVAILABLE_REASON), {
    code: MIRAGE_WEBGPU_UNAVAILABLE_CODE,
  });
}
