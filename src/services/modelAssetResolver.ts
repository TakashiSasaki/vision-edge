import type { ClassifierAssetStatuses } from './imageClassifierAssets.ts';
import type { RunnableModelType } from '../types/vision.ts';
import {
  MIRAGE_DECODER_TABLES_EXPECTED_SHA256,
  MIRAGE_DECODER_TABLES_EXPECTED_SIZE,
  MIRAGE_DECODER_TABLES_FILE,
} from './mirageCactCodec.ts';
import { STEREO_CORS_PROXY_PREFIX } from './browserModelCache.ts';

export type AssetDeliverySource =
  | 'browser-cache'
  | 'cors-proxy'
  | 'external-fallback'
  | 'external';
export type AssetSourceRecorder = (assetKey: string, source: AssetDeliverySource) => void;

export type CacheAssetModelId = RunnableModelType | 'image-classifier';
export type CacheIntegrity =
  | 'pinned'
  | 'self-recorded'
  | 'invalid'
  | 'not-present'
  | 'pending'
  | 'unknown'
  | 'blocked';

export type AssetCacheState = 'cached' | 'downloading' | 'unavailable';

export interface ModelAssetCatalogEntry {
  id: RunnableModelType;
  file: string;
  upstreamUrl: string;
  localUrl: string;
  license: string;
  licenseNote: string;
}

export interface WasmAssetCatalogEntry {
  id: 'mediapipe' | 'ort';
  upstreamUrl: string;
  localUrl: string;
  license: string;
}

export interface ContainerModelStatusEntry {
  acquisitionEnabled?: boolean;
  blockedReason?: string | null;
  id: CacheAssetModelId;
  filePresent?: boolean | null;
  integrity?: CacheIntegrity;
  sha256?: string | null;
  expectedSize?: number | null;
  expectedSha256?: string | null;
  variantId?: string;
  file: string;
  status: AssetCacheState;
  size: number | null;
  cachedAt: string | null;
  error: string | null;
  localUrl: string;
  upstreamUrl: string;
  license: string;
}

export interface ContainerCacheStatusResponse {
  schemaVersion?: 2;
  observedAt?: string;
  catalogCount?: number;
  classification?: ClassifierAssetStatuses;
  enabled: boolean;
  wasm: {
    mediapipe: {
      available: boolean;
      localUrl: string;
      upstreamUrl: string;
      license: string;
    };
    ort: {
      available: boolean;
      localUrl: string;
      upstreamUrl: string;
      license: string;
    };
  };
  models: Record<RunnableModelType, ContainerModelStatusEntry>;
  variants?: Record<string, ContainerModelStatusEntry>;
}

export interface ResolvedAssetUrls {
  primaryUrl: string;
  fallbackUrl: string | null;
  source: AssetDeliverySource;
}

export const UPSTREAM_MEDIAPIPE_WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
export const LOCAL_MEDIAPIPE_WASM_URL = UPSTREAM_MEDIAPIPE_WASM_URL;

export const UPSTREAM_ORT_WASM_URL =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
export const LOCAL_ORT_WASM_URL = UPSTREAM_ORT_WASM_URL;

export const WASM_ASSET_CATALOG: Record<'mediapipe' | 'ort', WasmAssetCatalogEntry> = {
  mediapipe: {
    id: 'mediapipe',
    upstreamUrl: UPSTREAM_MEDIAPIPE_WASM_URL,
    localUrl: LOCAL_MEDIAPIPE_WASM_URL,
    license: 'Apache-2.0',
  },
  ort: {
    id: 'ort',
    upstreamUrl: UPSTREAM_ORT_WASM_URL,
    localUrl: LOCAL_ORT_WASM_URL,
    license: 'MIT',
  },
};

export interface ModelVariantCatalogEntry {
  licenseNoticeUrl?: string;
  expectedSize?: number;
  expectedSha256?: string;
  modelId: RunnableModelType;
  variantId: string;
  file: string;
  upstreamUrl: string;
  localUrl: string;
  license: string;
  licenseNote: string;
  isDefault?: boolean;
}

export const MODEL_VARIANT_CATALOG: Record<string, ModelVariantCatalogEntry> = {
  'selfie-segmenter:general': {
    modelId: 'selfie-segmenter',
    variantId: 'general',
    file: 'selfie_segmenter.tflite',
    upstreamUrl:
      'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite',
    localUrl:
      'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite',
    license: 'Apache-2.0',
    licenseNote: 'Google MediaPipe / Apache License 2.0（ブラウザ Service Worker キャッシュ配信）',
    isDefault: true,
  },
  'selfie-segmenter:landscape': {
    modelId: 'selfie-segmenter',
    variantId: 'landscape',
    file: 'selfie_segmenter_landscape.tflite',
    upstreamUrl:
      'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite',
    localUrl:
      'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite',
    license: 'Apache-2.0',
    licenseNote: 'Google MediaPipe / Apache License 2.0（ブラウザ Service Worker キャッシュ配信）',
  },
  'mediapipe:lite0': {
    modelId: 'mediapipe',
    variantId: 'lite0',
    file: 'efficientdet_lite0.tflite',
    upstreamUrl:
      'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
    localUrl:
      'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
    license: 'Apache-2.0',
    licenseNote: 'Google MediaPipe / Apache License 2.0（ブラウザ Service Worker キャッシュ配信）',
    isDefault: true,
  },
  'mediapipe:lite2': {
    modelId: 'mediapipe',
    variantId: 'lite2',
    file: 'efficientdet_lite2.tflite',
    upstreamUrl:
      'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/1/efficientdet_lite2.tflite',
    localUrl:
      'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/1/efficientdet_lite2.tflite',
    license: 'Apache-2.0',
    licenseNote: 'Google MediaPipe / Apache License 2.0（ブラウザ Service Worker キャッシュ配信）',
  },
  'yolo-seg:yolov8n-seg': {
    modelId: 'yolo-seg',
    variantId: 'yolov8n-seg',
    file: 'yolov8n-seg.onnx',
    upstreamUrl:
      'https://huggingface.co/unileon-robotics/YOLO8-ONNX/resolve/main/yolov8n-seg.onnx',
    localUrl:
      'https://huggingface.co/unileon-robotics/YOLO8-ONNX/resolve/main/yolov8n-seg.onnx',
    license: 'AGPL-3.0',
    licenseNote: 'Ultralytics YOLOv8 / GNU AGPLv3（OSS・非商用デモ前提・ブラウザ Service Worker キャッシュ）',
    isDefault: true,
  },
  'yolo-seg:yolov8s-seg': {
    modelId: 'yolo-seg',
    variantId: 'yolov8s-seg',
    file: 'yolov8s-seg.onnx',
    upstreamUrl:
      'https://huggingface.co/unileon-robotics/YOLO8-ONNX/resolve/main/dynamic/yolov8s-seg.onnx',
    localUrl:
      'https://huggingface.co/unileon-robotics/YOLO8-ONNX/resolve/main/dynamic/yolov8s-seg.onnx',
    license: 'AGPL-3.0',
    licenseNote: 'Ultralytics YOLOv8 / GNU AGPLv3（OSS・非商用デモ前提・ブラウザ Service Worker キャッシュ）',
  },
  'segformer:b0': {
    modelId: 'segformer',
    variantId: 'b0',
    file: 'segformer-b0-quantized.onnx',
    upstreamUrl:
      'https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx',
    localUrl:
      'https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx',
    license: 'NVIDIA Source Code License (Non-Commercial)',
    licenseNote: 'NVIDIA SegFormer-B0 / 研究・非商用デモ限定ライセンス',
    isDefault: true,
  },
  'segformer:b1': {
    modelId: 'segformer',
    variantId: 'b1',
    file: 'segformer-b1-quantized.onnx',
    upstreamUrl:
      'https://huggingface.co/Xenova/segformer-b1-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx',
    localUrl:
      'https://huggingface.co/Xenova/segformer-b1-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx',
    license: 'NVIDIA Source Code License (Non-Commercial)',
    licenseNote: 'NVIDIA SegFormer-B1 / 研究・非商用デモ限定ライセンス',
  },
  'segformer:b2': {
    modelId: 'segformer',
    variantId: 'b2',
    file: 'segformer-b2-quantized.onnx',
    upstreamUrl:
      'https://huggingface.co/Xenova/segformer-b2-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx',
    localUrl:
      'https://huggingface.co/Xenova/segformer-b2-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx',
    license: 'NVIDIA Source Code License (Non-Commercial)',
    licenseNote: 'NVIDIA SegFormer-B2 / 研究・非商用デモ限定ライセンス',
  },
};

export type LegacyRunnableModelType = Exclude<RunnableModelType, 'mirage' | 'stereo-depth'>;

export const MIRAGE_UPSTREAM_URL =
  'https://huggingface.co/Cactus-Compute/mirage/resolve/76c1e9e53381aca8fc3843c636216c96ff220122/mirage.cact';
export const MIRAGE_LOCAL_URL = MIRAGE_UPSTREAM_URL;

export const MIRAGE_EXPERIMENTAL_ONNX_FILE =
  'cact-dequantized-reference-order.onnx';
export const MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE = 34_281_503;
export const MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256 =
  '52d52d2918160404cd806702a270736819712025cf3c6aa8e907a06bc3cbdd07';
export const MIRAGE_EXPERIMENTAL_ONNX_LOCAL_URL = `/api/model-cache/mirage/${MIRAGE_EXPERIMENTAL_ONNX_FILE}?sha256=${MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256}`;

export {
  MIRAGE_DECODER_TABLES_FILE,
  MIRAGE_DECODER_TABLES_EXPECTED_SIZE,
  MIRAGE_DECODER_TABLES_EXPECTED_SHA256,
};

export const MIRAGE_READER_ONNX_FILE = 'mirage-reader.onnx';
export const MIRAGE_READER_ONNX_EXPECTED_SIZE = 7_880_013;
export const MIRAGE_READER_ONNX_EXPECTED_SHA256 =
  '79b39a53ae24cd6904eabb94814ec5ad8ee171d23800e9c56a637dc1e0c6d34b';
export const MIRAGE_READER_ONNX_LOCAL_URL = `/api/model-cache/mirage/${MIRAGE_READER_ONNX_FILE}?sha256=${MIRAGE_READER_ONNX_EXPECTED_SHA256}`;

export const MIRAGE_CROSS_KV_ONNX_FILE = 'mirage-cross-kv.onnx';
export const MIRAGE_CROSS_KV_ONNX_EXPECTED_SIZE = 16_668_140;
export const MIRAGE_CROSS_KV_ONNX_EXPECTED_SHA256 =
  '7978dfc75e3e0584ca44379d402b8e671e0e10d10c2eb337400b999338ef9c3e';
export const MIRAGE_CROSS_KV_ONNX_LOCAL_URL = `/api/model-cache/mirage/${MIRAGE_CROSS_KV_ONNX_FILE}?sha256=${MIRAGE_CROSS_KV_ONNX_EXPECTED_SHA256}`;

export const MIRAGE_DECODER_STEP_ONNX_FILE = 'mirage-decoder-step.onnx';
export const MIRAGE_DECODER_STEP_ONNX_EXPECTED_SIZE = 73_637_843;
export const MIRAGE_DECODER_STEP_ONNX_EXPECTED_SHA256 =
  '4249e1f404a318e80bb18cc95960d6de81dad13587455680e5321e018dc079f6';
export const MIRAGE_DECODER_STEP_ONNX_LOCAL_URL = `/api/model-cache/mirage/${MIRAGE_DECODER_STEP_ONNX_FILE}?sha256=${MIRAGE_DECODER_STEP_ONNX_EXPECTED_SHA256}`;

export const MIRAGE_DECODER_TABLES_LOCAL_URL = `/api/model-cache/mirage/${MIRAGE_DECODER_TABLES_FILE}?sha256=${MIRAGE_DECODER_TABLES_EXPECTED_SHA256}`;

export type MirageFullPipelineComponentKey =
  | 'reader'
  | 'crossKv'
  | 'decoderStep'
  | 'decoderTables';

export interface MiragePinnedComponentSpec {
  key: MirageFullPipelineComponentKey;
  file: string;
  expectedSize: number;
  expectedSha256: string;
  localUrl: string;
}

export const MIRAGE_FULL_PIPELINE_COMPONENTS: Readonly<
  Record<MirageFullPipelineComponentKey, MiragePinnedComponentSpec>
> = {
  reader: {
    key: 'reader',
    file: MIRAGE_READER_ONNX_FILE,
    expectedSize: MIRAGE_READER_ONNX_EXPECTED_SIZE,
    expectedSha256: MIRAGE_READER_ONNX_EXPECTED_SHA256,
    localUrl: MIRAGE_READER_ONNX_LOCAL_URL,
  },
  crossKv: {
    key: 'crossKv',
    file: MIRAGE_CROSS_KV_ONNX_FILE,
    expectedSize: MIRAGE_CROSS_KV_ONNX_EXPECTED_SIZE,
    expectedSha256: MIRAGE_CROSS_KV_ONNX_EXPECTED_SHA256,
    localUrl: MIRAGE_CROSS_KV_ONNX_LOCAL_URL,
  },
  decoderStep: {
    key: 'decoderStep',
    file: MIRAGE_DECODER_STEP_ONNX_FILE,
    expectedSize: MIRAGE_DECODER_STEP_ONNX_EXPECTED_SIZE,
    expectedSha256: MIRAGE_DECODER_STEP_ONNX_EXPECTED_SHA256,
    localUrl: MIRAGE_DECODER_STEP_ONNX_LOCAL_URL,
  },
  decoderTables: {
    key: 'decoderTables',
    file: MIRAGE_DECODER_TABLES_FILE,
    expectedSize: MIRAGE_DECODER_TABLES_EXPECTED_SIZE,
    expectedSha256: MIRAGE_DECODER_TABLES_EXPECTED_SHA256,
    localUrl: MIRAGE_DECODER_TABLES_LOCAL_URL,
  },
} as const;

export const LOCAL_CACTUS_WASM_JS_URL = '/wasm/cactus/cactus.js';
export const LOCAL_CACTUS_WASM_BINARY_URL = '/wasm/cactus/cactus.wasm';
export const CACHE_CACTUS_WASM_JS_URL = LOCAL_CACTUS_WASM_JS_URL;
export const CACHE_CACTUS_WASM_BINARY_URL = LOCAL_CACTUS_WASM_BINARY_URL;

export const MIRAGE_VARIANT_CATALOG: Record<string, ModelVariantCatalogEntry> = {
  'mirage:mirage-51m': {
    modelId: 'mirage',
    variantId: 'mirage-51m',
    file: 'mirage.cact',
    upstreamUrl: MIRAGE_UPSTREAM_URL,
    localUrl: MIRAGE_LOCAL_URL,
    license: 'Apache-2.0',
    licenseNote: 'Cactus Compute, Inc. / Apache License 2.0（ブラウザWASM + Service Worker キャッシュ）',
    isDefault: true,
  },
  'mirage:mirage-onnx-exp-wasm': {
    modelId: 'mirage',
    variantId: 'mirage-onnx-exp-wasm',
    file: MIRAGE_EXPERIMENTAL_ONNX_FILE,
    upstreamUrl: MIRAGE_EXPERIMENTAL_ONNX_LOCAL_URL,
    localUrl: MIRAGE_EXPERIMENTAL_ONNX_LOCAL_URL,
    license: 'Apache-2.0',
    licenseNote: 'Cactus Compute, Inc. / Apache License 2.0（公式.cact復元重み・部分ONNX実験用 452/583テンソル）',
    expectedSize: MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE,
    expectedSha256: MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256,
  },
  'mirage:mirage-onnx-exp-webgpu': {
    modelId: 'mirage',
    variantId: 'mirage-onnx-exp-webgpu',
    file: MIRAGE_EXPERIMENTAL_ONNX_FILE,
    upstreamUrl: MIRAGE_EXPERIMENTAL_ONNX_LOCAL_URL,
    localUrl: MIRAGE_EXPERIMENTAL_ONNX_LOCAL_URL,
    license: 'Apache-2.0',
    licenseNote: 'Cactus Compute, Inc. / Apache License 2.0（公式.cact復元重み・部分ONNX実験用 452/583テンソル）',
    expectedSize: MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE,
    expectedSha256: MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256,
  },
};

export const MIRAGE_MODEL_ASSET_ENTRY: ModelAssetCatalogEntry = {
  id: 'mirage',
  file: MIRAGE_VARIANT_CATALOG['mirage:mirage-51m'].file,
  upstreamUrl: MIRAGE_VARIANT_CATALOG['mirage:mirage-51m'].upstreamUrl,
  localUrl: MIRAGE_VARIANT_CATALOG['mirage:mirage-51m'].localUrl,
  license: MIRAGE_VARIANT_CATALOG['mirage:mirage-51m'].license,
  licenseNote: MIRAGE_VARIANT_CATALOG['mirage:mirage-51m'].licenseNote,
};

export const STEREO_VARIANT_CATALOG: Record<string, ModelVariantCatalogEntry> = {
  'stereo-depth:stereo-sad-baseline': {
    modelId: 'stereo-depth',
    variantId: 'stereo-sad-baseline',
    file: 'builtin-stereo-sad',
    upstreamUrl: 'https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API',
    localUrl: 'builtin-stereo-sad',
    license: 'MIT (Built-in)',
    licenseNote: '内蔵CPUステレオマッチング (Separable Sliding-Window SAD · 重み取得不要)',
    isDefault: true,
  },
  'stereo-depth:stereo-webgpu-wgsl': {
    modelId: 'stereo-depth',
    variantId: 'stereo-webgpu-wgsl',
    file: 'builtin-stereo-webgpu-wgsl',
    upstreamUrl: 'https://www.w3.org/TR/webgpu/',
    localUrl: 'builtin-stereo-webgpu-wgsl',
    license: 'MIT (Built-in)',
    licenseNote: '内蔵WebGPU WGSLコンピュートシェーダ (Census + Gradient GPU視差集約 · 重み取得不要 · WASM/CPU自動フォールバック)',
  },
  'stereo-depth:lightstereo-s': {
    modelId: 'stereo-depth',
    variantId: 'lightstereo-s',
    file: 'lightstereo_s_sceneflow_general_opt_256_512.onnx',
    upstreamUrl:
      'https://cdn.jsdelivr.net/gh/zz990099/lightstereo_cpp@master/models/lightstereo_s_sceneflow_general_opt_256_512.onnx',
    localUrl:
      'https://cdn.jsdelivr.net/gh/zz990099/lightstereo_cpp@master/models/lightstereo_s_sceneflow_general_opt_256_512.onnx',
    license: 'Apache-2.0',
    licenseNote: 'LightStereo-S / Apache-2.0（オンライン自動取得・ブラウザキャッシュ保存 + ローカル.onnx上書き対応 · Concat自動分割WebGPU優先 / WASMフォールバック）',
  },
  'stereo-depth:mobilestereonet-2d': {
    modelId: 'stereo-depth',
    variantId: 'mobilestereonet-2d',
    file: 'model_528_240_float32.onnx',
    upstreamUrl:
      'https://drive.usercontent.google.com/download?id=1Dkyrg5Fu554gqxfclkHC6zJBCYrdzOO0&export=download&confirm=t',
    localUrl: `${STEREO_CORS_PROXY_PREFIX}model_528_240_float32.onnx`,
    license: 'MIT',
    licenseNote: 'MobileStereoNet-2D / MIT（CORS中継取得 + ブラウザキャッシュ保存 / ローカル.onnx上書き対応 · 5D Conv3d含むためWASM推奨）',
  },
  'stereo-depth:hitnet-sf': {
    modelId: 'stereo-depth',
    variantId: 'hitnet-sf',
    file: 'hitnet_sceneflow_240x320.onnx',
    upstreamUrl:
      'https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_sceneflow_240x320.onnx',
    localUrl: `${STEREO_CORS_PROXY_PREFIX}hitnet_sceneflow_240x320.onnx`,
    license: 'Apache-2.0',
    licenseNote: 'HITNet SceneFlow / Apache-2.0（CORS中継取得 + ブラウザキャッシュ保存 / ローカル.onnx上書き対応 · WebGPU優先 / WASMフォールバック）',
  },
  'stereo-depth:crestereo-init-iter2': {
    modelId: 'stereo-depth',
    variantId: 'crestereo-init-iter2',
    file: 'crestereo_init_iter2_180x320.onnx',
    upstreamUrl:
      'https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_init_iter2_180x320.onnx',
    localUrl: `${STEREO_CORS_PROXY_PREFIX}crestereo_init_iter2_180x320.onnx`,
    license: 'Apache-2.0',
    licenseNote: 'CREStereo Init Iter2 / Apache-2.0（CORS中継取得 + ブラウザキャッシュ保存 / ローカル.onnx上書き対応 · Concat自動分割WebGPU優先 / WASMフォールバック）',
  },
  'stereo-depth:crestereo-combined-iter5': {
    modelId: 'stereo-depth',
    variantId: 'crestereo-combined-iter5',
    file: 'crestereo_combined_iter5_240x320.onnx',
    upstreamUrl:
      'https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter5_240x320.onnx',
    localUrl: `${STEREO_CORS_PROXY_PREFIX}crestereo_combined_iter5_240x320.onnx`,
    license: 'Apache-2.0',
    licenseNote: 'CREStereo Combined Iter5 / Apache-2.0（CORS中継取得 + ブラウザキャッシュ保存 / ローカル.onnx上書き対応 · Concat自動分割WebGPU優先 / WASMフォールバック）',
  },
  'stereo-depth:custom-onnx': {
    modelId: 'stereo-depth',
    variantId: 'custom-onnx',
    file: 'custom_stereo.onnx',
    upstreamUrl: 'https://onnxruntime.ai/',
    localUrl: 'custom_stereo.onnx',
    license: 'ユーザー指定',
    licenseNote: '任意のステレオONNX重み（1入力/2入力自動対応 · Concat自動分割WebGPU優先）',
  },
};

export const STEREO_MODEL_ASSET_ENTRY: ModelAssetCatalogEntry = {
  id: 'stereo-depth',
  file: STEREO_VARIANT_CATALOG['stereo-depth:stereo-sad-baseline'].file,
  upstreamUrl: STEREO_VARIANT_CATALOG['stereo-depth:stereo-sad-baseline'].upstreamUrl,
  localUrl: STEREO_VARIANT_CATALOG['stereo-depth:stereo-sad-baseline'].localUrl,
  license: STEREO_VARIANT_CATALOG['stereo-depth:stereo-sad-baseline'].license,
  licenseNote: STEREO_VARIANT_CATALOG['stereo-depth:stereo-sad-baseline'].licenseNote,
};

export function getVariantCatalogEntry(
  model: RunnableModelType,
  variantId?: string
): ModelVariantCatalogEntry {
  if (variantId !== undefined && typeof variantId !== 'string') {
    throw new RangeError('Variant id must be a string');
  }
  if (model === 'mirage') {
    if (variantId !== undefined && variantId !== '') {
      const key = `mirage:${variantId}`;
      if (Object.hasOwn(MIRAGE_VARIANT_CATALOG, key)) return MIRAGE_VARIANT_CATALOG[key];
      throw new RangeError(`Unknown model variant: ${key}`);
    }
    return MIRAGE_VARIANT_CATALOG['mirage:mirage-51m'];
  }
  if (model === 'stereo-depth') {
    if (variantId !== undefined && variantId !== '') {
      const key = `stereo-depth:${variantId}`;
      if (Object.hasOwn(STEREO_VARIANT_CATALOG, key)) return STEREO_VARIANT_CATALOG[key];
      throw new RangeError(`Unknown model variant: ${key}`);
    }
    return STEREO_VARIANT_CATALOG['stereo-depth:stereo-sad-baseline'];
  }
  if (variantId !== undefined && variantId !== '') {
    const key = `${model}:${variantId}`;
    if (Object.hasOwn(MODEL_VARIANT_CATALOG, key)) return MODEL_VARIANT_CATALOG[key];
    throw new RangeError(`Unknown model variant: ${key}`);
  }
  const entry = Object.values(MODEL_VARIANT_CATALOG).find(
    (e) => e.modelId === model && e.isDefault
  );
  if (!entry) throw new RangeError(`Unknown model: ${model}`);
  return entry;
}

export const MODEL_ASSET_CATALOG: Record<LegacyRunnableModelType, ModelAssetCatalogEntry> = {
  'selfie-segmenter': {
    id: 'selfie-segmenter',
    file: MODEL_VARIANT_CATALOG['selfie-segmenter:general'].file,
    upstreamUrl: MODEL_VARIANT_CATALOG['selfie-segmenter:general'].upstreamUrl,
    localUrl: MODEL_VARIANT_CATALOG['selfie-segmenter:general'].localUrl,
    license: MODEL_VARIANT_CATALOG['selfie-segmenter:general'].license,
    licenseNote: MODEL_VARIANT_CATALOG['selfie-segmenter:general'].licenseNote,
  },
  mediapipe: {
    id: 'mediapipe',
    file: MODEL_VARIANT_CATALOG['mediapipe:lite0'].file,
    upstreamUrl: MODEL_VARIANT_CATALOG['mediapipe:lite0'].upstreamUrl,
    localUrl: MODEL_VARIANT_CATALOG['mediapipe:lite0'].localUrl,
    license: MODEL_VARIANT_CATALOG['mediapipe:lite0'].license,
    licenseNote: MODEL_VARIANT_CATALOG['mediapipe:lite0'].licenseNote,
  },
  'yolo-seg': {
    id: 'yolo-seg',
    file: MODEL_VARIANT_CATALOG['yolo-seg:yolov8n-seg'].file,
    upstreamUrl: MODEL_VARIANT_CATALOG['yolo-seg:yolov8n-seg'].upstreamUrl,
    localUrl: MODEL_VARIANT_CATALOG['yolo-seg:yolov8n-seg'].localUrl,
    license: MODEL_VARIANT_CATALOG['yolo-seg:yolov8n-seg'].license,
    licenseNote: MODEL_VARIANT_CATALOG['yolo-seg:yolov8n-seg'].licenseNote,
  },
  segformer: {
    id: 'segformer',
    file: MODEL_VARIANT_CATALOG['segformer:b0'].file,
    upstreamUrl: MODEL_VARIANT_CATALOG['segformer:b0'].upstreamUrl,
    localUrl: MODEL_VARIANT_CATALOG['segformer:b0'].localUrl,
    license: MODEL_VARIANT_CATALOG['segformer:b0'].license,
    licenseNote: MODEL_VARIANT_CATALOG['segformer:b0'].licenseNote,
  },
};

const resolvedSources: Record<string, AssetDeliverySource> = {};
const listeners = new Set<() => void>();

function notifyListeners(): void {
  listeners.forEach((listener) => listener());
}

export function subscribeAssetResolution(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function isContainerModelCacheEnabled(): boolean {
  return false;
}

export function resetResolvedSourcesForTest(): void {
  for (const key of Object.keys(resolvedSources)) {
    delete resolvedSources[key];
  }
  notifyListeners();
}

export function recordResolvedSource(
  assetKey: RunnableModelType | 'wasm-mediapipe' | 'wasm-ort' | string,
  source: AssetDeliverySource
): void {
  resolvedSources[assetKey] = source;
  notifyListeners();
}

export function recordResolvedSources(
  sources: ReadonlyMap<string, AssetDeliverySource>
): void {
  if (sources.size === 0) return;
  for (const [key, source] of sources) resolvedSources[key] = source;
  notifyListeners();
}

export function getResolvedSource(
  assetKey: RunnableModelType | 'wasm-mediapipe' | 'wasm-ort' | string
): AssetDeliverySource {
  return resolvedSources[assetKey] ?? 'external';
}

export async function resolveMediaPipeWasmUrls(): Promise<ResolvedAssetUrls> {
  return {
    primaryUrl: UPSTREAM_MEDIAPIPE_WASM_URL,
    fallbackUrl: null,
    source: 'external',
  };
}

export async function resolveOrtWasmUrls(): Promise<ResolvedAssetUrls> {
  return {
    primaryUrl: UPSTREAM_ORT_WASM_URL,
    fallbackUrl: null,
    source: 'external',
  };
}

export class ModelAssetsBlockedError extends Error {
  constructor(
    public readonly model: RunnableModelType,
    public readonly variantId: string,
    public readonly reason: string
  ) {
    super(`${model}:${variantId} BLOCKED: ${reason}`);
    this.name = 'ModelAssetsBlockedError';
  }
}

export async function resolveModelAssetUrls(
  model: RunnableModelType,
  variantId?: string
): Promise<ResolvedAssetUrls> {
  const entry = getVariantCatalogEntry(model, variantId);
  if (model === 'stereo-depth') {
    const isCorsProxy = entry.localUrl.startsWith(STEREO_CORS_PROXY_PREFIX);
    return {
      primaryUrl: isCorsProxy ? entry.localUrl : entry.upstreamUrl,
      fallbackUrl: null,
      source: isCorsProxy ? 'cors-proxy' : 'external',
    };
  }
  return {
    primaryUrl: entry.upstreamUrl,
    fallbackUrl: null,
    source: 'external',
  };
}
