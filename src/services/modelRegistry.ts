import { IMAGE_CLASSIFIER_ASSETS } from './imageClassifierAssets';
import type { ModelFamilyType } from '../types/classification';
import type { ModelMetadata, ModelVariantMetadata, ModelState, RunnableModelType } from '../types/vision';
import {
  MIRAGE_MODEL_ASSET_ENTRY,
  MIRAGE_VARIANT_CATALOG,
  MODEL_ASSET_CATALOG,
  MODEL_VARIANT_CATALOG,
  STEREO_MODEL_ASSET_ENTRY,
  STEREO_VARIANT_CATALOG,
} from './modelAssetResolver';

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
  { id: 'mediapipe', name: 'MediaPipe Object Detector', implemented: true,
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

export const MIRAGE_MODEL: ModelMetadata = {
  id: 'mirage',
  name: 'Mirage (Cactus WASM)',
  implemented: true,
  runtime: 'Cactus WASM / Web Worker · 実験用ONNX (WebGPU/WASM)',
  outputType: '検出物体・輪郭マスク・OCRテキスト・方向/距離（実験用ONNXはフル推論: 4段ONNX+CQテーブル／高速モード: 領域候補・枠・マスクのみ）',
  description: '公式の Mirage .cact を WASM Worker で実行します。完成版単一ONNXは未検証のため利用不可ですが、公式.cact復元重みによる実験用ONNX（高速領域候補 452/583テンソル、およびパラメータ設定で Encoder + CTC OCR + CrossKV + 8層デコーダのフル4段ONNX 582/583推論テンソル）を試せます。',
  license: MIRAGE_MODEL_ASSET_ENTRY.license,
  licenseNote: MIRAGE_MODEL_ASSET_ENTRY.licenseNote,
  upstreamUrl: MIRAGE_MODEL_ASSET_ENTRY.upstreamUrl,
  localCachePath: MIRAGE_MODEL_ASSET_ENTRY.localUrl,
  defaultVariantId: 'mirage-51m',
  variants: [
    {
      id: 'mirage-51m',
      name: 'Mirage 51.3M (2-bit/4-bit .cact WASM · 20.6MB)',
      description: '公式量子化アーカイブ (.cact) を Web Worker 上の CPU WebAssembly で非同期推論するフルモデル版。',
      fileName: MIRAGE_VARIANT_CATALOG['mirage:mirage-51m'].file,
      upstreamUrl: MIRAGE_VARIANT_CATALOG['mirage:mirage-51m'].upstreamUrl,
      localUrl: MIRAGE_VARIANT_CATALOG['mirage:mirage-51m'].localUrl,
      isDefault: true,
    },
    {
      id: 'mirage-onnx-exp-wasm',
      name: 'Mirage 部分ONNX 実験用 (WASM · 34.3MB)',
      description: '【実験用機能】公式.cact復元重みによるONNX（Encoder SHA256 52d52d29 · 34.3MB）。WASMで高速領域候補（452/583テンソル）を推論し、パラメータタブの「フル推論」有効化時は 4段ONNX + CQテーブル（Encoder + CTC OCR + CrossKV + 8層デコーダ · 582/583推論テンソル）で物体名・OCR・出力選択まで実行します。',
      fileName: MIRAGE_VARIANT_CATALOG['mirage:mirage-onnx-exp-wasm'].file,
      upstreamUrl: MIRAGE_VARIANT_CATALOG['mirage:mirage-onnx-exp-wasm'].upstreamUrl,
      localUrl: MIRAGE_VARIANT_CATALOG['mirage:mirage-onnx-exp-wasm'].localUrl,
    },
    {
      id: 'mirage-onnx-exp-webgpu',
      name: 'Mirage 部分ONNX 実験用 (GPU/WebGPU優先 · 34.3MB)',
      description: '【実験用機能】公式.cact復元重みによるONNX（Encoder SHA256 52d52d29 · 34.3MB）。WebGPU優先（WASMフォールバック付き）で高速領域候補（452/583テンソル）を推論し、パラメータタブの「フル推論」有効化時は 4段ONNX + CQテーブル（Encoder + CTC OCR + CrossKV + 8層デコーダ · 582/583推論テンソル）で物体名・OCR・出力選択まで実行します。',
      fileName: MIRAGE_VARIANT_CATALOG['mirage:mirage-onnx-exp-webgpu'].file,
      upstreamUrl: MIRAGE_VARIANT_CATALOG['mirage:mirage-onnx-exp-webgpu'].upstreamUrl,
      localUrl: MIRAGE_VARIANT_CATALOG['mirage:mirage-onnx-exp-webgpu'].localUrl,
    },
  ],
};

export const STEREO_DEPTH_MODEL: ModelMetadata = {
  id: 'stereo-depth',
  name: 'Stereo Depth (視差・深度)',
  implemented: true,
  runtime: 'WebGPU WGSL Compute · ONNX Runtime Web (WebGPU/WASM) · 内蔵CPU SAD',
  outputType: '視差マップ (px) と校正深度 (m) のカラーマップ可視化・任意座標プローブ',
  description: 'Side-by-Side (左右分割) ステレオ映像から視差と深度を推定します。即時動作する軽量CPU SADベースラインとネイティブWebGPU WGSLコンピュートシェーダ版に加え、Concat自動分割によるWebGPU高速化対応の各ステレオONNXモデルを比較できます。',
  license: STEREO_MODEL_ASSET_ENTRY.license,
  licenseNote: STEREO_MODEL_ASSET_ENTRY.licenseNote,
  upstreamUrl: STEREO_MODEL_ASSET_ENTRY.upstreamUrl,
  localCachePath: STEREO_MODEL_ASSET_ENTRY.localUrl,
  defaultVariantId: 'stereo-sad-baseline',
  variants: [
    {
      id: 'stereo-sad-baseline',
      name: 'SAD Baseline (内蔵CPU・軽量高速)',
      description: 'スライディングウィンドウSAD＋放物線サブピクセル補間。外部ファイル不要で即座にリアルタイム動作します。',
      fileName: STEREO_VARIANT_CATALOG['stereo-depth:stereo-sad-baseline'].file,
      upstreamUrl: STEREO_VARIANT_CATALOG['stereo-depth:stereo-sad-baseline'].upstreamUrl,
      localUrl: STEREO_VARIANT_CATALOG['stereo-depth:stereo-sad-baseline'].localUrl,
      isDefault: true,
    },
    {
      id: 'stereo-webgpu-wgsl',
      name: 'WebGPU WGSL Compute (超高速・GPU視差集約)',
      description: 'ブラウザWebGPUコンピュートシェーダによるCensus+Gradient視差推定＋中央値平滑化 (重み不要・GPU非対応時はCPU SADへ自動フォールバック)。',
      fileName: STEREO_VARIANT_CATALOG['stereo-depth:stereo-webgpu-wgsl'].file,
      upstreamUrl: STEREO_VARIANT_CATALOG['stereo-depth:stereo-webgpu-wgsl'].upstreamUrl,
      localUrl: STEREO_VARIANT_CATALOG['stereo-depth:stereo-webgpu-wgsl'].localUrl,
    },
    {
      id: 'lightstereo-s',
      name: 'LightStereo-S (ONNX・超軽量リアルタイム)',
      description: '2D CNNベースの最速級リアルタイムステレオONNX (256×512 · Concat自動分割WebGPU優先 · オンライン自動取得対応)。ローカル .onnx での上書きも可能です。',
      fileName: STEREO_VARIANT_CATALOG['stereo-depth:lightstereo-s'].file,
      upstreamUrl: STEREO_VARIANT_CATALOG['stereo-depth:lightstereo-s'].upstreamUrl,
      localUrl: STEREO_VARIANT_CATALOG['stereo-depth:lightstereo-s'].localUrl,
    },
    {
      id: 'mobilestereonet-2d',
      name: 'MobileStereoNet-2D (ONNX・モバイル軽量)',
      description: '省メモリ・軽量ステレオONNX (240×528 · 5D Conv3D→2D Conv自動グラフ変換WebGPU優先 · オンライン自動取得対応)。ローカル .onnx での上書きも可能です。',
      fileName: STEREO_VARIANT_CATALOG['stereo-depth:mobilestereonet-2d'].file,
      upstreamUrl: STEREO_VARIANT_CATALOG['stereo-depth:mobilestereonet-2d'].upstreamUrl,
      localUrl: STEREO_VARIANT_CATALOG['stereo-depth:mobilestereonet-2d'].localUrl,
    },
    {
      id: 'hitnet-sf',
      name: 'HITNet (ONNX・軽量タイル精密化)',
      description: '反復タイル精密化による高速ステレオONNX (240×320 · WebGPU優先 · CORS中継 + ブラウザキャッシュ対応)。ローカル .onnx 上書きも可能です。',
      fileName: STEREO_VARIANT_CATALOG['stereo-depth:hitnet-sf'].file,
      upstreamUrl: STEREO_VARIANT_CATALOG['stereo-depth:hitnet-sf'].upstreamUrl,
      localUrl: STEREO_VARIANT_CATALOG['stereo-depth:hitnet-sf'].localUrl,
    },
    {
      id: 'crestereo-init-iter2',
      name: 'CREStereo Init Iter2 (ONNX・高精度)',
      description: '階層的再帰精密化＋適応相関による高精度ステレオONNX (180×320 · Concat自動分割WebGPU優先 · CORS中継 + ブラウザキャッシュ対応)。',
      fileName: STEREO_VARIANT_CATALOG['stereo-depth:crestereo-init-iter2'].file,
      upstreamUrl: STEREO_VARIANT_CATALOG['stereo-depth:crestereo-init-iter2'].upstreamUrl,
      localUrl: STEREO_VARIANT_CATALOG['stereo-depth:crestereo-init-iter2'].localUrl,
    },
    {
      id: 'crestereo-combined-iter5',
      name: 'CREStereo Combined Iter5 (ONNX・最高精度)',
      description: '2段階入力（init＋next · 240×320 · Concat自動分割WebGPU優先）で細部まで精密化する最高精度ステレオONNX（CORS中継 + ブラウザキャッシュ対応）。',
      fileName: STEREO_VARIANT_CATALOG['stereo-depth:crestereo-combined-iter5'].file,
      upstreamUrl: STEREO_VARIANT_CATALOG['stereo-depth:crestereo-combined-iter5'].upstreamUrl,
      localUrl: STEREO_VARIANT_CATALOG['stereo-depth:crestereo-combined-iter5'].localUrl,
    },
    {
      id: 'custom-onnx',
      name: 'Custom Stereo ONNX (任意モデル読込)',
      description: '任意のステレオONNXモデル（1入力[1,6,H,W]・2入力[1,3,H,W]・2chフロー出力を自動認識・Concat自動分割WebGPU対応）をテストできます。',
      fileName: STEREO_VARIANT_CATALOG['stereo-depth:custom-onnx'].file,
      upstreamUrl: STEREO_VARIANT_CATALOG['stereo-depth:custom-onnx'].upstreamUrl,
      localUrl: STEREO_VARIANT_CATALOG['stereo-depth:custom-onnx'].localUrl,
    },
  ],
};

export const RUNNABLE_MODEL_REGISTRY: readonly ModelMetadata[] = [
  ...MODEL_REGISTRY,
  MIRAGE_MODEL,
  STEREO_DEPTH_MODEL,
];

export function isRunnableModel(id: unknown): id is RunnableModelType {
  return RUNNABLE_MODEL_REGISTRY.some(model => model.id === id && model.implemented);
}

export function requireRunnableModel(id: unknown): asserts id is RunnableModelType {
  if (!isRunnableModel(id)) throw new Error(`Unsupported model ID: ${String(id)}`);
}

export function modelMetadata(id: RunnableModelType): ModelMetadata {
  requireRunnableModel(id);
  return RUNNABLE_MODEL_REGISTRY.find(model => model.id === id)!;
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
  if(state.model==='mirage'){
    if (
      state.variantId === 'mirage-onnx-exp-wasm' ||
      state.variantId === 'mirage-onnx-exp-webgpu'
    ) {
      const backend = state.backend;
      const requested =
        backend?.requestedProvider === 'webgpu'
          ? 'WebGPU優先'
          : backend?.requestedProvider === 'wasm'
            ? 'WASM'
            : backend?.requestedProvider === 'auto'
              ? '自動(WebGPU優先)'
              : '未報告';
      const configured = backend?.configuredProviders.length
        ? backend.configuredProviders
            .map(p => (p === 'wasm' ? 'WASM' : 'WebGPU'))
            .join(' → ')
        : '未報告';
      return `実験用部分ONNX 要求: ${requested}／設定: ${configured}${backend?.fallbackReason ? `／理由: ${backend.fallbackReason.slice(0, 180)}` : ''}（実行バックエンドは未計測）`;
    }
    return 'Cactus WebAssembly (CPU / Web Worker 非同期推論)';
  }
  if (state.model === 'stereo-depth') {
    const backend = state.backend;
    if (backend?.runtime === 'stereo-sad-cpu') {
      return 'Stereo SAD (CPU スライディングウィンドウ視差推定)';
    }
    if (backend?.runtime === 'stereo-webgpu-wgsl') {
      const usingGpu = backend.configuredProviders.includes('webgpu-wgsl');
      if (usingGpu) {
        return 'Stereo WGSL Compute (WebGPU GPUコンピュート視差推定)';
      }
      return `Stereo WGSL Compute → CPU SAD フォールバック${backend.fallbackReason ? ` (${backend.fallbackReason.slice(0, 120)})` : ''}`;
    }
    const requested =
      backend?.requestedProvider === 'auto'
        ? '自動(WebGPU優先)'
        : backend?.requestedProvider === 'webgpu'
          ? 'WebGPU優先'
          : backend?.requestedProvider === 'wasm'
            ? 'WASM'
            : '未報告';
    const configured = backend?.configuredProviders.length
      ? backend.configuredProviders.map(p => (p === 'wasm' ? 'WASM' : 'WebGPU')).join(' → ')
      : '未報告';
    return `Stereo ONNX 要求: ${requested}／設定: ${configured}${backend?.fallbackReason ? `／理由: ${backend.fallbackReason.slice(0, 180)}` : ''}（実行バックエンドは未計測）`;
  }
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
  defaultVariantId: 'lite0', assetVerification: 'pinned-client-sha256-required',
  variants: Object.values(IMAGE_CLASSIFIER_ASSETS).map(entry => ({
    id: entry.variantId, name: entry.variantId === 'lite0' ? 'EfficientNet-Lite0' : 'EfficientNet-Lite2', fileName: entry.file,
    upstreamUrl: entry.upstreamUrl, isDefault: entry.variantId === 'lite0',
  })),
} as const;
export const ALL_MODEL_REGISTRY = [...RUNNABLE_MODEL_REGISTRY, IMAGE_CLASSIFIER_MODEL] as const;
export function modelFamilyMetadata(id: ModelFamilyType) {
  const found = ALL_MODEL_REGISTRY.find(entry => entry.id === id);
  if (!found) throw new RangeError(`Unsupported model family: ${String(id)}`);
  return found;
}
