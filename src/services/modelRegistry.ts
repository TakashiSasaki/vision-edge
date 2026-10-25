import type { ModelMetadata, ModelState, RunnableModelType } from '../types/vision';
import { MODEL_ASSET_CATALOG } from './modelAssetResolver';

export const MODEL_REGISTRY: readonly ModelMetadata[] = [
  { id: 'selfie-segmenter', name: 'Selfie Segmenter', implemented: true,
    runtime: 'MediaPipe ImageSegmenter', outputType: '人物領域マスクと外接矩形',
    description: '人物領域を推定します。複数人物の個体別インスタンス分割ではありません。',
    license: MODEL_ASSET_CATALOG['selfie-segmenter'].license,
    licenseNote: MODEL_ASSET_CATALOG['selfie-segmenter'].licenseNote,
    upstreamUrl: MODEL_ASSET_CATALOG['selfie-segmenter'].upstreamUrl,
    localCachePath: MODEL_ASSET_CATALOG['selfie-segmenter'].localUrl },
  { id: 'mediapipe', name: 'EfficientDet-Lite0', implemented: true,
    runtime: 'MediaPipe ObjectDetector', outputType: '検出矩形とクラスラベル',
    description: 'COCO の物体クラスを検出します。輪郭マスクは生成しません。',
    license: MODEL_ASSET_CATALOG.mediapipe.license,
    licenseNote: MODEL_ASSET_CATALOG.mediapipe.licenseNote,
    upstreamUrl: MODEL_ASSET_CATALOG.mediapipe.upstreamUrl,
    localCachePath: MODEL_ASSET_CATALOG.mediapipe.localUrl },
  { id: 'yolo-seg', name: 'YOLO-seg', implemented: true,
    runtime: 'ONNX Runtime Web (YOLOv8n-seg)', outputType: '80クラス矩形とインスタンス領域マスク',
    description: 'COCOの物体を検出し、物体ごとの輪郭マスク（インスタンスセグメンテーション）を生成します。',
    license: MODEL_ASSET_CATALOG['yolo-seg'].license,
    licenseNote: MODEL_ASSET_CATALOG['yolo-seg'].licenseNote,
    upstreamUrl: MODEL_ASSET_CATALOG['yolo-seg'].upstreamUrl,
    localCachePath: MODEL_ASSET_CATALOG['yolo-seg'].localUrl },
  { id: 'segformer', name: 'SegFormer (ADE20K)', implemented: true,
    runtime: 'ONNX Runtime Web (SegFormer-B0)', outputType: '150クラス全画素セマンティックマスク',
    description: '室内・都市景観の全画素（壁、床、家具、人物など150種）をリアルタイムに分類します。',
    license: MODEL_ASSET_CATALOG.segformer.license,
    licenseNote: MODEL_ASSET_CATALOG.segformer.licenseNote,
    upstreamUrl: MODEL_ASSET_CATALOG.segformer.upstreamUrl,
    localCachePath: MODEL_ASSET_CATALOG.segformer.localUrl },
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

export function backendDescription(state: ModelState): string {
  if(state.status!=='ready')return 'バックエンド未準備';
  if(state.model==='mediapipe'||state.model==='selfie-segmenter')return `${state.delegate} delegate 指定・初期化済み（GPU実動作は未計測）`;
  const backend=state.backend;
  const requested=backend?.requestedProvider==='auto'?'自動(WebGPU優先)':backend?.requestedProvider==='webgpu'?'WebGPU優先':backend?.requestedProvider==='wasm'?'WASM':'未報告';
  const configured=backend?.configuredProviders.length?backend.configuredProviders.map(p=>p==='wasm'?'WASM':'WebGPU').join(' → '):'未報告';
  return `ONNX 要求: ${requested}／設定: ${configured}${backend?.fallbackReason?`／理由: ${backend.fallbackReason.slice(0,180)}`:''}（実行バックエンドは未計測）`;
}
