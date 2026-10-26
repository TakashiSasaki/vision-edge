import type { ClassifierAssetStatuses } from './imageClassifierAssets.ts';
import type { RunnableModelType } from '../types/vision.ts';

export type AssetDeliverySource = 'container-cache' | 'external-fallback' | 'external';
export type AssetSourceRecorder = (assetKey: string, source: AssetDeliverySource) => void;

export type CacheAssetModelId = RunnableModelType | 'image-classifier';
export type CacheIntegrity = 'pinned' | 'self-recorded' | 'invalid' | 'not-present' | 'pending' | 'unknown';

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
export const LOCAL_MEDIAPIPE_WASM_URL = '/api/model-cache/wasm/mediapipe';

export const UPSTREAM_ORT_WASM_URL =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
export const LOCAL_ORT_WASM_URL = '/api/model-cache/wasm/ort/';

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
    localUrl: '/api/model-cache/models/selfie_segmenter.tflite',
    license: 'Apache-2.0',
    licenseNote: 'Google MediaPipe / Apache License 2.0（コンテナキャッシュ・再配布可）',
    isDefault: true,
  },
  'selfie-segmenter:landscape': {
    modelId: 'selfie-segmenter',
    variantId: 'landscape',
    file: 'selfie_segmenter_landscape.tflite',
    upstreamUrl:
      'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite',
    localUrl: '/api/model-cache/models/selfie_segmenter_landscape.tflite',
    license: 'Apache-2.0',
    licenseNote: 'Google MediaPipe / Apache License 2.0（コンテナキャッシュ・再配布可）',
  },
  'mediapipe:lite0': {
    modelId: 'mediapipe',
    variantId: 'lite0',
    file: 'efficientdet_lite0.tflite',
    upstreamUrl:
      'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
    localUrl: '/api/model-cache/models/efficientdet_lite0.tflite',
    license: 'Apache-2.0',
    licenseNote: 'Google MediaPipe / Apache License 2.0（コンテナキャッシュ・再配布可）',
    isDefault: true,
  },
  'mediapipe:lite2': {
    modelId: 'mediapipe',
    variantId: 'lite2',
    file: 'efficientdet_lite2.tflite',
    upstreamUrl:
      'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/1/efficientdet_lite2.tflite',
    localUrl: '/api/model-cache/models/efficientdet_lite2.tflite',
    license: 'Apache-2.0',
    licenseNote: 'Google MediaPipe / Apache License 2.0（コンテナキャッシュ・再配布可）',
  },
  'yolo-seg:yolov8n-seg': {
    modelId: 'yolo-seg',
    variantId: 'yolov8n-seg',
    file: 'yolov8n-seg.onnx',
    upstreamUrl:
      'https://huggingface.co/unileon-robotics/YOLO8-ONNX/resolve/main/yolov8n-seg.onnx',
    localUrl: '/api/model-cache/models/yolov8n-seg.onnx',
    license: 'AGPL-3.0',
    licenseNote: 'Ultralytics YOLOv8 / GNU AGPLv3（OSS・非商用デモ前提でキャッシュ配信）',
    isDefault: true,
  },
  'yolo-seg:yolov8s-seg': {
    modelId: 'yolo-seg',
    variantId: 'yolov8s-seg',
    file: 'yolov8s-seg.onnx',
    upstreamUrl:
      'https://huggingface.co/unileon-robotics/YOLO8-ONNX/resolve/main/dynamic/yolov8s-seg.onnx',
    localUrl: '/api/model-cache/models/yolov8s-seg.onnx',
    license: 'AGPL-3.0',
    licenseNote: 'Ultralytics YOLOv8 / GNU AGPLv3（OSS・非商用デモ前提でキャッシュ配信）',
  },
  'segformer:b0': {
    modelId: 'segformer',
    variantId: 'b0',
    file: 'segformer-b0-quantized.onnx',
    upstreamUrl:
      'https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx',
    localUrl: '/api/model-cache/models/segformer-b0-quantized.onnx',
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
    localUrl: '/api/model-cache/models/segformer-b1-quantized.onnx',
    license: 'NVIDIA Source Code License (Non-Commercial)',
    licenseNote: 'NVIDIA SegFormer-B1 / 研究・非商用デモ限定ライセンス',
  },
  'segformer:b2': {
    modelId: 'segformer',
    variantId: 'b2',
    file: 'segformer-b2-quantized.onnx',
    upstreamUrl:
      'https://huggingface.co/Xenova/segformer-b2-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx',
    localUrl: '/api/model-cache/models/segformer-b2-quantized.onnx',
    license: 'NVIDIA Source Code License (Non-Commercial)',
    licenseNote: 'NVIDIA SegFormer-B2 / 研究・非商用デモ限定ライセンス',
  },
};

export function getVariantCatalogEntry(
  model: RunnableModelType,
  variantId?: string
): ModelVariantCatalogEntry {
  if (variantId !== undefined && typeof variantId !== 'string') throw new RangeError('Variant id must be a string');
  if (variantId !== undefined && variantId !== '') {
    const key = `${model}:${variantId}`;
    if (Object.hasOwn(MODEL_VARIANT_CATALOG, key)) return MODEL_VARIANT_CATALOG[key];
    throw new RangeError(`Unknown model variant: ${key}`);
  }
  const entry = Object.values(MODEL_VARIANT_CATALOG).find(e => e.modelId === model && e.isDefault);
  if (!entry) throw new RangeError(`Unknown model: ${model}`);
  return entry;
}

export const MODEL_ASSET_CATALOG: Record<RunnableModelType, ModelAssetCatalogEntry> = {
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

declare global {
  interface Window {
    __CONTAINER_MODEL_CACHE_ENABLED__?: boolean;
  }
}

let testOverrideEnabled: boolean | null = null;
let cachedStatusResponse: ContainerCacheStatusResponse | null = null;
let cachedStatusAt = 0;
const STATUS_TTL_MS = 5_000;

const resolvedSources: Record<string, AssetDeliverySource> = {};
const listeners = new Set<() => void>();

function notifyListeners(): void {
  listeners.forEach(listener => listener());
}

export function subscribeAssetResolution(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function isContainerModelCacheEnabled(): boolean {
  if (testOverrideEnabled !== null) return testOverrideEnabled;
  if (typeof window === 'undefined') return false;
  return Boolean(window.__CONTAINER_MODEL_CACHE_ENABLED__);
}

export function setContainerModelCacheEnabledForTest(enabled: boolean | null): void {
  testOverrideEnabled = enabled;
  cachedStatusResponse = null;
  cachedStatusAt = 0;
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

// Commit one accepted initialization atomically before synchronous observers run.
export function recordResolvedSources(sources: ReadonlyMap<string, AssetDeliverySource>): void {
  if (sources.size === 0) return;
  for (const [key, source] of sources) resolvedSources[key] = source;
  notifyListeners();
}

export function getResolvedSource(
  assetKey: RunnableModelType | 'wasm-mediapipe' | 'wasm-ort' | string
): AssetDeliverySource {
  if (resolvedSources[assetKey]) return resolvedSources[assetKey];
  return isContainerModelCacheEnabled() ? 'container-cache' : 'external';
}

export async function fetchContainerCacheStatus(
  forceRefresh = false
): Promise<ContainerCacheStatusResponse | null> {
  if (!isContainerModelCacheEnabled()) return null;
  const now = Date.now();
  if (!forceRefresh && cachedStatusResponse && now - cachedStatusAt < STATUS_TTL_MS) {
    return cachedStatusResponse;
  }
  if (typeof fetch !== 'function') return null;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 3_500);
  try {
    const response = await fetch('/api/model-cache/status', {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });
    if (!response.ok) throw new Error('Cache status HTTP failure');
    const contentType = response.headers.get('content-type') ?? '';
    if (!contentType.includes('application/json')) throw new Error('Cache status is not JSON');
    const data = (await response.json()) as ContainerCacheStatusResponse;
    if (!isRecord(data) || typeof data.enabled !== 'boolean' || !isRecord(data.models) || !isRecord(data.wasm)) throw new Error('Malformed cache status');
    cachedStatusResponse = data;
    cachedStatusAt = Date.now();
    notifyListeners();
    return data;
  } catch {
    cachedStatusResponse = null; cachedStatusAt = 0;
    return null;
  } finally {
    clearTimeout(timer);
  }
}

export function getLatestContainerCacheStatus(): ContainerCacheStatusResponse | null {
  return cachedStatusResponse;
}

export async function resolveMediaPipeWasmUrls(): Promise<ResolvedAssetUrls> {
  if (!isContainerModelCacheEnabled()) {
    return {
      primaryUrl: UPSTREAM_MEDIAPIPE_WASM_URL,
      fallbackUrl: null,
      source: 'external',
    };
  }
  const status = await fetchContainerCacheStatus();
  if (status?.enabled && status.wasm?.mediapipe?.available === true && status.wasm.mediapipe.localUrl === LOCAL_MEDIAPIPE_WASM_URL && status.wasm.mediapipe.upstreamUrl === UPSTREAM_MEDIAPIPE_WASM_URL) {
    return {
      primaryUrl: status.wasm.mediapipe.localUrl || LOCAL_MEDIAPIPE_WASM_URL,
      fallbackUrl: UPSTREAM_MEDIAPIPE_WASM_URL,
      source: 'container-cache',
    };
  }
  return {
    primaryUrl: UPSTREAM_MEDIAPIPE_WASM_URL,
    fallbackUrl: null,
    source: 'external-fallback',
  };
}

export async function resolveOrtWasmUrls(): Promise<ResolvedAssetUrls> {
  if (!isContainerModelCacheEnabled()) {
    return {
      primaryUrl: UPSTREAM_ORT_WASM_URL,
      fallbackUrl: null,
      source: 'external',
    };
  }
  const status = await fetchContainerCacheStatus();
  if (status?.enabled && status.wasm?.ort?.available === true && status.wasm.ort.localUrl === LOCAL_ORT_WASM_URL && status.wasm.ort.upstreamUrl === UPSTREAM_ORT_WASM_URL) {
    const rawLocal = status.wasm.ort.localUrl || LOCAL_ORT_WASM_URL;
    const localWithSlash = rawLocal.endsWith('/') ? rawLocal : `${rawLocal}/`;
    const origin =
      typeof window !== 'undefined' && window.location?.origin
        ? window.location.origin
        : '';
    return {
      primaryUrl: origin ? `${origin}${localWithSlash}` : localWithSlash,
      fallbackUrl: UPSTREAM_ORT_WASM_URL,
      source: 'container-cache',
    };
  }
  return {
    primaryUrl: UPSTREAM_ORT_WASM_URL,
    fallbackUrl: null,
    source: 'external-fallback',
  };
}

export async function resolveModelAssetUrls(
  model: RunnableModelType,
  variantId?: string
): Promise<ResolvedAssetUrls> {
  const entry = getVariantCatalogEntry(model, variantId);
  if (!isContainerModelCacheEnabled()) {
    return {
      primaryUrl: entry.upstreamUrl,
      fallbackUrl: null,
      source: 'external',
    };
  }
  const status = await fetchContainerCacheStatus();
  const variantKey = `${entry.modelId}:${entry.variantId}`;
  // A present variants map is authoritative, including a missing/invalid row.
  // Only a legacy response with no variants field may identify a default model.
  const modern = !!status && Object.hasOwn(status, 'variants');
  const candidate = modern
    ? (isRecord(status?.variants) ? status.variants[variantKey] : undefined)
    : (entry.isDefault ? status?.models?.[model] : undefined);
  if (status?.enabled && matchesVariantStatus(candidate, entry, modern)) {
    return { primaryUrl: entry.localUrl, fallbackUrl: entry.upstreamUrl, source: 'container-cache' };
  }
  return {
    primaryUrl: entry.upstreamUrl,
    fallbackUrl: null,
    source: 'external-fallback',
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}
function matchesVariantStatus(value: unknown, entry: ModelVariantCatalogEntry, modern: boolean): boolean {
  if (!isRecord(value)) return false;
  return value.id === entry.modelId
    && (modern ? value.variantId === entry.variantId : value.variantId === undefined || value.variantId === entry.variantId)
    && value.file === entry.file && value.localUrl === entry.localUrl && value.upstreamUrl === entry.upstreamUrl
    && value.license === entry.license && value.error === null
    && ((value.status === 'cached' && Number.isSafeInteger(value.size) && Number(value.size) >= 1024 && typeof value.cachedAt === 'string')
      || (value.status === 'downloading' && value.size === null && value.cachedAt === null));
}
