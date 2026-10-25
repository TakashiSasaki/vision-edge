import type { ModelMetadata, ModelState, RunnableModelType } from '../types/vision';

export const MODEL_REGISTRY: readonly ModelMetadata[] = [
  { id: 'selfie-segmenter', name: 'Selfie Segmenter', implemented: true,
    runtime: 'MediaPipe ImageSegmenter', outputType: '人物領域マスクと外接矩形',
    description: '人物領域を推定します。複数人物の個体別インスタンス分割ではありません。' },
  { id: 'mediapipe', name: 'EfficientDet-Lite0', implemented: true,
    runtime: 'MediaPipe ObjectDetector', outputType: '検出矩形とクラスラベル',
    description: 'COCO の物体クラスを検出します。輪郭マスクは生成しません。' },
  { id: 'yolo-seg', name: 'YOLO-seg', implemented: true,
    runtime: 'ONNX Runtime Web (YOLOv8n-seg)', outputType: '80クラス矩形とインスタンス領域マスク',
    description: 'COCOの物体を検出し、物体ごとの輪郭マスク（インスタンスセグメンテーション）を生成します。' },
  { id: 'segformer', name: 'SegFormer', implemented: false,
    runtime: '未実装', outputType: '未実装', description: '将来の実装候補です。現在は実行できません。' },
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
  return state.status === 'ready'
    ? `${state.delegate} delegate 指定・初期化済み（GPU実動作は未計測）`
    : 'バックエンド未準備';
}
