import { IMAGE_CLASSIFIER_ASSETS } from './imageClassifierAssets';
import type { ModelFamilyType } from '../types/classification';
import type { ModelMetadata, ModelVariantMetadata, ModelState, RunnableModelType } from '../types/vision';
import { MODEL_ASSET_CATALOG, MODEL_VARIANT_CATALOG } from './modelAssetResolver';

export const MODEL_REGISTRY: readonly ModelMetadata[] = [
  { id: 'selfie-segmenter', name: 'Selfie Segmenter', implemented: true,
    runtime: 'MediaPipe ImageSegmenter', outputType: '人物領域マスクと外接矩形',
    description: '人物領域を推定します。複数人物の個体別インスタンス分割ではありません。',
    license: MODEL_ASSET_CATALOG['selfie-segmenter'].license,
    licenseNote: MODEL_ASSET_CATALOG['selfie-segmenter'].licenseNote,
    upstreamUrl: MODEL_ASSET_CATALOG['selfie-segmenter'].upstreamUrl,
    localCachePath: MODEL_ASSET_CATALOG['selfie-segmenter'].localUrl,
    defaultVariantId: 'general',
    variants: [
      {
        id: 'general',
        name: 'General (標準 256×256)',
        description: '一般用途向けの正方形入力セルフィーモデル。人物境界を鮮明に検出します。',
        fileName: MODEL_VARIANT_CATALOG['selfie-segmenter:general'].file,
        upstreamUrl: MODEL_VARIANT_CATALOG['selfie-segmenter:general'].upstreamUrl,
        localUrl: MODEL_VARIANT_CATALOG['selfie-segmenter:general'].localUrl,
        isDefault: true,
      },
      {
        id: 'landscape',
        name: 'Landscape (高速 144×256)',
        description: '横長入力の軽量モデル。処理速度重視でリアルタイム応答性に優れます。',
        fileName: MODEL_VARIANT_CATALOG['selfie-segmenter:landscape'].file,
        upstreamUrl: MODEL_VARIANT_CATALOG['selfie-segmenter:landscape'].upstreamUrl,
        localUrl: MODEL_VARIANT_CATALOG['selfie-segmenter:landscape'].localUrl,
      },
    ],
  },
  { id: 'mediapipe', name: 'EfficientDet-Lite', implemented: true,
    runtime: 'MediaPipe ObjectDetector', outputType: '検出矩形とクラスラベル',
    description: 'COCO の物体クラスを検出します。輪郭マスクは生成しません。',
    license: MODEL_ASSET_CATALOG.mediapipe.license,
    licenseNote: MODEL_ASSET_CATALOG.mediapipe.licenseNote,
    upstreamUrl: MODEL_ASSET_CATALOG.mediapipe.upstreamUrl,
    localCachePath: MODEL_ASSET_CATALOG.mediapipe.localUrl,
    defaultVariantId: 'lite0',
    variants: [
      {
        id: 'lite0',
        name: 'EfficientDet-Lite0 (軽量・高速)',
        description: '超軽量な物体検出モデル。モバイル・低スペック端末向け。',
        fileName: MODEL_VARIANT_CATALOG['mediapipe:lite0'].file,
        upstreamUrl: MODEL_VARIANT_CATALOG['mediapipe:lite0'].upstreamUrl,
        localUrl: MODEL_VARIANT_CATALOG['mediapipe:lite0'].localUrl,
        isDefault: true,
      },
      {
        id: 'lite2',
        name: 'EfficientDet-Lite2 (中量・高精度)',
        description: '中規模な物体検出モデル。小型・遠くの物体に対する検出精度が向上します。',
        fileName: MODEL_VARIANT_CATALOG['mediapipe:lite2'].file,
        upstreamUrl: MODEL_VARIANT_CATALOG['mediapipe:lite2'].upstreamUrl,
        localUrl: MODEL_VARIANT_CATALOG['mediapipe:lite2'].localUrl,
      },
    ],
  },
  { id: 'yolo-seg', name: 'YOLO-seg', implemented: true,
    runtime: 'ONNX Runtime Web (YOLOv8-seg)', outputType: '80クラス矩形とインスタンス領域マスク',
    description: 'COCOの物体を検出し、物体ごとの輪郭マスク（インスタンスセグメンテーション）を生成します。',
    license: MODEL_ASSET_CATALOG['yolo-seg'].license,
    licenseNote: MODEL_ASSET_CATALOG['yolo-seg'].licenseNote,
    upstreamUrl: MODEL_ASSET_CATALOG['yolo-seg'].upstreamUrl,
    localCachePath: MODEL_ASSET_CATALOG['yolo-seg'].localUrl,
    defaultVariantId: 'yolov8n-seg',
    variants: [
      {
        id: 'yolov8n-seg',
        name: 'YOLOv8n-seg (Nano・超高速)',
        description: 'NanoサイズのYOLOv8インスタンス分割。リアルタイム性に優れます。',
        fileName: MODEL_VARIANT_CATALOG['yolo-seg:yolov8n-seg'].file,
        upstreamUrl: MODEL_VARIANT_CATALOG['yolo-seg:yolov8n-seg'].upstreamUrl,
        localUrl: MODEL_VARIANT_CATALOG['yolo-seg:yolov8n-seg'].localUrl,
        isDefault: true,
      },
      {
        id: 'yolov8s-seg',
        name: 'YOLOv8s-seg (Small・高精度)',
        description: 'SmallサイズのYOLOv8インスタンス分割。複雑なシーンで高精度なマスクを生成します。',
        fileName: MODEL_VARIANT_CATALOG['yolo-seg:yolov8s-seg'].file,
        upstreamUrl: MODEL_VARIANT_CATALOG['yolo-seg:yolov8s-seg'].upstreamUrl,
        localUrl: MODEL_VARIANT_CATALOG['yolo-seg:yolov8s-seg'].localUrl,
      },
    ],
  },
  { id: 'segformer', name: 'SegFormer (ADE20K)', implemented: true,
    runtime: 'ONNX Runtime Web (SegFormer)', outputType: '150クラス全画素セマンティックマスク',
    description: '室内・都市景観の全画素（壁、床、家具、人物など150種）をリアルタイムに分類します。',
    license: MODEL_ASSET_CATALOG.segformer.license,
    licenseNote: MODEL_ASSET_CATALOG.segformer.licenseNote,
    upstreamUrl: MODEL_ASSET_CATALOG.segformer.upstreamUrl,
    localCachePath: MODEL_ASSET_CATALOG.segformer.localUrl,
    defaultVariantId: 'b0',
    variants: [
      {
        id: 'b0',
        name: 'SegFormer-B0 (軽量・標準)',
        description: '軽量バックボーンMiT-B0。全画素150クラスを省リソースでリアルタイム分類。',
        fileName: MODEL_VARIANT_CATALOG['segformer:b0'].file,
        upstreamUrl: MODEL_VARIANT_CATALOG['segformer:b0'].upstreamUrl,
        localUrl: MODEL_VARIANT_CATALOG['segformer:b0'].localUrl,
        isDefault: true,
      },
      {
        id: 'b1',
        name: 'SegFormer-B1 (中量・バランス)',
        description: '中量バックボーンMiT-B1。速度とセグメンテーション精度のバランスモデル。',
        fileName: MODEL_VARIANT_CATALOG['segformer:b1'].file,
        upstreamUrl: MODEL_VARIANT_CATALOG['segformer:b1'].upstreamUrl,
        localUrl: MODEL_VARIANT_CATALOG['segformer:b1'].localUrl,
      },
      {
        id: 'b2',
        name: 'SegFormer-B2 (大型・高精度)',
        description: '大型バックボーンMiT-B2。細部や複雑な背景で高精細な全画素マスクを出力。',
        fileName: MODEL_VARIANT_CATALOG['segformer:b2'].file,
        upstreamUrl: MODEL_VARIANT_CATALOG['segformer:b2'].upstreamUrl,
        localUrl: MODEL_VARIANT_CATALOG['segformer:b2'].localUrl,
      },
    ],
  },
];

export function isRunnableModel(id: unknown): id is RunnableModelType {
  return MODEL_REGISTRY.some(model => model.id === id && model.implemented);
}

export function requireRunnableModel(id: unknown): asserts id is RunnableModelType {
  if (!isRunnableModel(id)) throw new Error(`Unsupported model ID: ${String(id)}`);
}

export function modelMetadata(id: RunnableModelType): ModelMetadata {
  requireRunnableModel(id);
  return MODEL_REGISTRY.find(model => model.id === id)!;
}

export function getVariantMetadata(
  modelId: RunnableModelType,
  variantId?: string
): ModelVariantMetadata {
  const meta = modelMetadata(modelId);
  const targetId = variantId || meta.defaultVariantId;
  const found = meta.variants.find(v => v.id === targetId);
  if (found) return found;
  return meta.variants.find(v => v.isDefault) || meta.variants[0];
}

export function backendDescription(state: ModelState): string {
  if(state.status!=='ready')return 'バックエンド未準備';
  if(state.model==='mediapipe'||state.model==='selfie-segmenter')return `${state.delegate} delegate 指定・初期化済み（GPU実動作は未計測）`;
  const backend=state.backend;
  const requested=backend?.requestedProvider==='auto'?'自動(WebGPU優先)':backend?.requestedProvider==='webgpu'?'WebGPU優先':backend?.requestedProvider==='wasm'?'WASM':'未報告';
  const configured=backend?.configuredProviders.length?backend.configuredProviders.map(p=>p==='wasm'?'WASM':'WebGPU').join(' → '):'未報告';
  return `ONNX 要求: ${requested}／設定: ${configured}${backend?.fallbackReason?`／理由: ${backend.fallbackReason.slice(0,180)}`:''}（実行バックエンドは未計測）`;
}

// Phase A exposes a discoverable backend family without putting classification
// into the current object/mask UI or its detection-only RunnableModelType.
export const IMAGE_CLASSIFIER_MODEL = {
  id: 'image-classifier', name: 'EfficientNet-Lite', implemented: true, availableInUI: true,
  runtime: 'MediaPipe ImageClassifier0.10.14', outputType: 'whole-image-classification',
  defaultVariantId: 'lite0', assetVerification: 'blocked',
  variants: Object.values(IMAGE_CLASSIFIER_ASSETS).map(entry => ({
    id: entry.variantId, name: entry.variantId === 'lite0' ? 'EfficientNet-Lite0' : 'EfficientNet-Lite2', fileName: entry.file,
    upstreamUrl: entry.upstreamUrl, isDefault: entry.variantId === 'lite0',
  })),
} as const;
export const ALL_MODEL_REGISTRY = [...MODEL_REGISTRY, IMAGE_CLASSIFIER_MODEL] as const;
export function modelFamilyMetadata(id: ModelFamilyType) {
  const found = ALL_MODEL_REGISTRY.find(entry => entry.id === id);
  if (!found) throw new RangeError(`Unsupported model family: ${String(id)}`);
  return found;
}
