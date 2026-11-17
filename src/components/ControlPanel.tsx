import React, { useState, useEffect, useSyncExternalStore } from 'react';
import {
  Sliders,
  Volume2,
  Layers,
  RotateCcw,
  SlidersHorizontal,
  Eye,
  Crosshair,
  Trash2,
  Upload,
} from 'lucide-react';
import {
  DetectedObject,
  RunnableModelType,
  RenderOptions,
  TTSConfig,
  StereoDepthMapData,
  StereoDepthProbeResult,
  StereoPinnedProbe,
  StereoDepthVariantId,
  StereoBackendPreference,
  StereoColormap,
  StereoViewMode,
  ModelState,
} from '../types/vision';
import type { ExtraInferenceOptions } from '../services/visionEngine';
import { speechService } from '../services/speechService';
import { confidenceDescription } from '../services/confidenceDescription';
import { modelMetadata } from '../services/modelRegistry';
import {
  isMirageExperimentalVariant,
  type MirageExperimentalExecutionDiagnostics,
} from '../services/mirageExperimentalOnnxService';
import { MIRAGE_SUPPORTED_LANGUAGES } from '../services/mirageCactCodec';
import {
  STEREO_MODEL_PROFILES,
  getLocalStereoOnnxInfo,
  getLocalStereoOnnxVersion,
  inspectDisparityMapAtPoint,
  subscribeLocalStereoOnnx,
} from '../services/stereoDepthService';
import {
  FREE_STEREO_VIDEO_PRESET_IDS,
  STEREO_VIDEO_PRESETS,
  SYNTHETIC_STEREO_VIDEO_PRESET_IDS,
  type StereoVideoGroundTruthRegion,
  type StereoVideoPresetId,
} from '../services/stereoVideoService';
import type { StereoVideoSource } from './VisionCanvas';

const MIRAGE_LANGUAGE_LABELS: Record<string, string> = {
  en: '英語 (en)',
  es: 'スペイン語 (es)',
  fr: 'フランス語 (fr)',
  pt: 'ポルトガル語 (pt)',
  de: 'ドイツ語 (de)',
  it: 'イタリア語 (it)',
  nl: 'オランダ語 (nl)',
  pl: 'ポーランド語 (pl)',
  ru: 'ロシア語 (ru)',
  ar: 'アラビア語 (ar)',
  zh: '中国語 (zh)',
};

interface ControlPanelProps {
  activeModel: RunnableModelType;
  activeVariant?: string;
  modelState?: ModelState;
  renderOptions: RenderOptions;
  onChangeRenderOptions: (opts: Partial<RenderOptions>) => void;
  inferenceParams?: ExtraInferenceOptions;
  onChangeInferenceParams?: (params: Partial<ExtraInferenceOptions>) => void;
  onResetModelParams?: () => void;
  ttsConfig: TTSConfig;
  onChangeTTSConfig: (cfg: Partial<TTSConfig>) => void;
  detectedObjects: DetectedObject[];
  hasCurrentFrame: boolean;
  onSelectObject: (obj: DetectedObject) => void;
  selectedObjectId?: string;
  isSpeaking: boolean;
  experimentalDiagnostics?: MirageExperimentalExecutionDiagnostics | null;
  stillImageLabel?: string | null;
  onSelectStillImagePreset?: (
    preset:
      | 'camera'
      | 'overlap_shapes'
      | 'text_regions'
      | 'coco_000000458054'
      | 'coco_000000331352'
  ) => void;
  onUploadStillImage?: (file: File) => void;
  stereoDepthMap?: StereoDepthMapData | null;
  stereoHoverProbe?: StereoDepthProbeResult | null;
  stereoPinnedProbes?: StereoPinnedProbe[];
  onClearStereoPins?: () => void;
  onRemoveStereoPin?: (id: string) => void;
  onPinStereoProbe?: (probe: StereoDepthProbeResult) => void;
  onSelectStereoStillPreset?: (preset: 'camera' | 'stereo_blocks' | 'stereo_corridor') => void;
  stereoVideoSource?: StereoVideoSource | null;
  stereoVideoRegions?: StereoVideoGroundTruthRegion[];
  onSelectStereoVideoPreset?: (presetId: StereoVideoPresetId) => void;
  onUploadStereoVideoFile?: (file: File) => void;
  onLoadStereoOnnxFile?: (variantId: StereoDepthVariantId, file: File) => void;
  onClearStereoOnnxFile?: (variantId: StereoDepthVariantId) => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  activeModel,
  activeVariant,
  modelState,
  renderOptions,
  onChangeRenderOptions,
  inferenceParams = {},
  onChangeInferenceParams,
  onResetModelParams,
  ttsConfig,
  onChangeTTSConfig,
  detectedObjects,
  hasCurrentFrame,
  onSelectObject,
  selectedObjectId,
  experimentalDiagnostics = null,
  stillImageLabel = null,
  onSelectStillImagePreset,
  onUploadStillImage,
  stereoDepthMap = null,
  stereoHoverProbe = null,
  stereoPinnedProbes = [],
  onClearStereoPins,
  onRemoveStereoPin,
  onPinStereoProbe,
  onSelectStereoStillPreset,
  stereoVideoSource = null,
  stereoVideoRegions = [],
  onSelectStereoVideoPreset,
  onUploadStereoVideoFile,
  onLoadStereoOnnxFile,
  onClearStereoOnnxFile,
}) => {
  const [activeTab, setActiveTab] = useState<'detection' | 'params' | 'settings'>('detection');
  const [, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([]);
  const isExperimentalMirage =
    activeModel === 'mirage' && isMirageExperimentalVariant(activeVariant);
  useSyncExternalStore(subscribeLocalStereoOnnx, getLocalStereoOnnxVersion);

  useEffect(() => {
    setAvailableVoices(speechService.getVoices());
  }, []);

  const handleManualSpeak = (obj: DetectedObject) => {
    onSelectObject(obj);
    speechService.speak(
      `${obj.labelJa}。${confidenceDescription(obj)}は${Math.round(obj.confidence * 100)}%です。`,
      obj.labelJa,
      { ...ttsConfig, enabled: true },
      'manual'
    );
  };

  const handleTestSpeech = () => {
    speechService.speak(
      '音声合成テストです。カメラの物体を認識して読み上げます。',
      'test',
      { ...ttsConfig, enabled: true },
      'manual'
    );
  };

  const confidenceThreshold =
    inferenceParams.confidenceThreshold ??
    renderOptions.confidenceThreshold ??
    (isExperimentalMirage ? 0.3 : 0.35);
  const minAreaPercent = inferenceParams.minAreaPercent ?? 0.5;
  const maxDetections =
    inferenceParams.maxDetections ??
    (activeModel === 'yolo-seg'
      ? 20
      : isExperimentalMirage
        ? 64
        : activeModel === 'mirage'
          ? 15
          : 10);
  const bboxSmoothing = inferenceParams.bboxSmoothing ?? 0.20;
  const iouThreshold = inferenceParams.iouThreshold ?? 0.45;
  const maskThreshold = inferenceParams.maskThreshold ?? 0.50;
  const mirageAnalysisMode = inferenceParams.analysisMode ?? 'fast';
  const mirageLanguage = inferenceParams.mirageLanguage ?? 'en';
  const mirageIncludeOcr = inferenceParams.includeOcr ?? true;
  const experimentalNmsIou =
    inferenceParams.experimentalNmsIou !== undefined
      ? inferenceParams.experimentalNmsIou
      : null;
  const showObjectProposals = inferenceParams.showObjectProposals ?? true;
  const showTextProposals = inferenceParams.showTextProposals ?? true;
  const showDiagnosticMasks = inferenceParams.showDiagnosticMasks ?? true;
  const overlayNativeWasm = inferenceParams.overlayNativeWasm ?? false;

  const stereoMaxDisparity = inferenceParams.maxDisparity ?? 64;
  const stereoSadWindowSize = inferenceParams.sadWindowSize ?? 7;
  const stereoAutoNorm = inferenceParams.autoNormalizeDisparity ?? true;
  const stereoMinDispPx = inferenceParams.minDisparityPx ?? 0;
  const stereoMaxDispPx = inferenceParams.maxDisparityPx ?? 64;
  const stereoSwapLR = inferenceParams.swapLeftRight ?? false;
  const stereoCameraRotate180 = inferenceParams.cameraRotate180 ?? true;
  const isLiveStereoCameraActive = !stillImageLabel && !stereoVideoSource;
  const stereoColormap: StereoColormap = inferenceParams.colormap ?? 'turbo';
  const stereoViewMode: StereoViewMode = inferenceParams.viewMode ?? 'left-overlay';
  const stereoBaselineMm = inferenceParams.baselineMm ?? 60;
  const stereoFocalPx = inferenceParams.focalLengthPx ?? 500;
  const stereoShowCalibDepth = inferenceParams.showCalibrationDepth ?? true;
  const stereoBackendPref: StereoBackendPreference = inferenceParams.backendPreference ?? 'auto';
  const activeStereoVariantId = (activeVariant || 'stereo-sad-baseline') as StereoDepthVariantId;
  const activeStereoProfile =
    STEREO_MODEL_PROFILES[activeStereoVariantId] ?? STEREO_MODEL_PROFILES['stereo-sad-baseline'];
  const localStereoOnnx = getLocalStereoOnnxInfo(activeStereoVariantId);

  const updateInferenceParam = <K extends keyof ExtraInferenceOptions>(key: K, value: ExtraInferenceOptions[K]) => {
    if (onChangeInferenceParams) {
      onChangeInferenceParams({ [key]: value });
    } else {
      if (key === 'confidenceThreshold' && typeof value === 'number') {
        onChangeRenderOptions({ confidenceThreshold: value });
      }
    }
  };

  return (
    <div className="w-full h-full min-h-0 flex flex-col bg-[#0c111e] border border-slate-800 rounded-xl overflow-hidden shadow-lg select-none">
      {/* Tab Switcher */}
      <div className="flex border-b border-slate-800 bg-[#090d18] shrink-0">
        <button
          onClick={() => setActiveTab('detection')}
          className={`flex-1 py-2.5 px-2 text-xs font-semibold flex items-center justify-center gap-1 border-b-2 transition-colors ${
            activeTab === 'detection'
              ? 'border-blue-500 text-blue-400 bg-slate-900/60'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          {activeModel === 'stereo-depth' ? (
            <>
              <Crosshair className="w-3.5 h-3.5" />
              <span className="whitespace-nowrap">
                深度計測 {hasCurrentFrame ? `(ピン ${stereoPinnedProbes.length})` : '（未取得）'}
              </span>
            </>
          ) : (
            <>
              <Layers className="w-3.5 h-3.5" />
              <span className="whitespace-nowrap">認識物体 {hasCurrentFrame ? `(${detectedObjects.length})` : '（未取得）'}</span>
            </>
          )}
        </button>

        <button
          onClick={() => setActiveTab('params')}
          className={`flex-1 py-2.5 px-2 text-xs font-semibold flex items-center justify-center gap-1 border-b-2 transition-colors ${
            activeTab === 'params'
              ? 'border-blue-500 text-blue-400 bg-slate-900/60'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <SlidersHorizontal className="w-3.5 h-3.5" />
          <span className="whitespace-nowrap">モデル調整</span>
        </button>

        <button
          onClick={() => setActiveTab('settings')}
          className={`flex-1 py-2.5 px-2 text-xs font-semibold flex items-center justify-center gap-1 border-b-2 transition-colors ${
            activeTab === 'settings'
              ? 'border-blue-500 text-blue-400 bg-slate-900/60'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Sliders className="w-3.5 h-3.5" />
          <span className="flex items-center gap-1 whitespace-nowrap">
            <span>音声</span>
            {ttsConfig.enabled && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />}
          </span>
        </button>
      </div>

      {/* Tab Body (Internal scroll only) */}
      <div className="flex-1 min-h-0 overflow-y-auto p-3.5 space-y-3.5 text-xs">
        {activeModel === 'segformer' && (
          <p className="text-slate-400">
            全画素マスクは最大スコアの分類です。softmaxスコアは正答率の保証ではありません。しきい値は一覧・ラベルに適用し、タップでは小さい領域や低スコアの画素も確認できます。面積は別表示です。
          </p>
        )}
        {activeModel === 'mirage' && !isExperimentalMirage && (
          <p className="text-slate-400">
            Mirage は公式 .cact を WASM Worker で実行します。検出物体・輪郭マスク・OCR・方向/距離（近/中/遠）を表示・読み上げます。
          </p>
        )}
        {isExperimentalMirage && (
          <div
            role="note"
            aria-label="実験用部分Mirage ONNXの制限事項"
            className="p-2.5 rounded-lg border border-amber-500/40 bg-amber-950/25 text-amber-100 space-y-1.5"
          >
            <div className="font-bold text-amber-300 flex items-center justify-between">
              <span>
                {mirageAnalysisMode === 'full'
                  ? '【実験用機能】Mirage ONNX フルパイプライン推論'
                  : '【実験用機能】部分Mirage ONNX 領域候補推論'}
              </span>
              <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-amber-900/60 border border-amber-600/40">
                {mirageAnalysisMode === 'full' ? '4段ONNX + CQテーブル · opset 17' : '452/583 tensors · opset 17'}
              </span>
            </div>
            <p className="text-[11px] text-amber-200/90 leading-relaxed">
              {mirageAnalysisMode === 'full'
                ? '公式.cactから再構成した Encoder + Reader (OCR) + CrossKV + Autoregressive Decoder の全段をブラウザ内ONNX Runtimeで実行します（初回切替時は約105MBの追加入手・SHA-256検証を実施）。既定の安全側は引き続き公式WASM版です。'
                : '公式.cactから復元した部分モデル（SHA-256: 52d52d29...）です。パラメータタブで「フル推論」に切り替えると OCR・語彙命名・構造デコーダを有効化できます。高速モードでは物体領域候補・テキスト領域候補・枠・128×128診断用マスクを表示します。'}
            </p>
            {onSelectStillImagePreset && (
              <div className="pt-1 border-t border-amber-800/40 space-y-1.5">
                <div className="flex items-center justify-between text-[10px] text-amber-300">
                  <span>入力画像テスト（カメラまたは検証画像を選択）:</span>
                  <span className="font-mono">{stillImageLabel ?? 'カメラ映像'}</span>
                </div>
                <div className="flex flex-wrap gap-1">
                  <button
                    type="button"
                    onClick={() => onSelectStillImagePreset('camera')}
                    className={`px-2 py-1 rounded text-[10px] font-semibold border transition-colors ${
                      !stillImageLabel
                        ? 'bg-blue-600 border-blue-400 text-white'
                        : 'bg-slate-900/80 border-slate-700 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    カメラ映像
                  </button>
                  <button
                    type="button"
                    onClick={() => onSelectStillImagePreset('overlap_shapes')}
                    className={`px-2 py-1 rounded text-[10px] font-semibold border transition-colors ${
                      stillImageLabel === 'overlap_shapes'
                        ? 'bg-cyan-700 border-cyan-400 text-white'
                        : 'bg-slate-900/80 border-slate-700 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    検証: 重なり図形
                  </button>
                  <button
                    type="button"
                    onClick={() => onSelectStillImagePreset('text_regions')}
                    className={`px-2 py-1 rounded text-[10px] font-semibold border transition-colors ${
                      stillImageLabel === 'text_regions'
                        ? 'bg-cyan-700 border-cyan-400 text-white'
                        : 'bg-slate-900/80 border-slate-700 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    検証: テキスト領域
                  </button>
                  <button
                    type="button"
                    onClick={() => onSelectStillImagePreset('coco_000000458054')}
                    className={`px-2 py-1 rounded text-[10px] font-semibold border transition-colors ${
                      stillImageLabel === 'coco_000000458054'
                        ? 'bg-cyan-700 border-cyan-400 text-white'
                        : 'bg-slate-900/80 border-slate-700 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    検証: COCO 458054
                  </button>
                  {onUploadStillImage && (
                    <label className="px-2 py-1 rounded text-[10px] font-semibold border bg-slate-900/80 border-slate-700 text-slate-300 hover:bg-slate-800 cursor-pointer">
                      <span>画像ファイル...</span>
                      <input
                        type="file"
                        accept="image/*"
                        aria-label="検証用の画像ファイルを選択"
                        className="hidden"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) onUploadStillImage(file);
                          e.currentTarget.value = '';
                        }}
                      />
                    </label>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {activeModel === 'stereo-depth' && (
          <div className="p-2.5 rounded-lg border border-cyan-500/35 bg-cyan-950/20 text-slate-200 space-y-2">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-bold text-cyan-300">Side-by-Side ステレオ深度推定</span>
              <span className="font-mono text-[10px] text-slate-300">
                {activeStereoProfile.engineType === 'webgpu-wgsl'
                  ? 'WGSL Compute'
                  : activeStereoProfile.tier === 'baseline'
                    ? 'CPU SAD'
                    : activeStereoProfile.tier === 'high-accuracy'
                      ? '高精度 ONNX'
                      : '軽量 ONNX'}
              </span>
            </div>
            <p className="text-[11px] text-slate-300 leading-relaxed">
              横2分割 (左|右) 映像から視差マップを生成します。カメラ使用時は横3840×縦1080ピクセルのステレオカメラを自動判別・接続し、実機がない場合でも合成SBSステレオ動画（3シーン）やフリー素材SBSステレオ動画でデモンストレーションできます。
            </p>

            {/* 1. Synthetic Side-by-Side Stereo Videos */}
            {onSelectStereoVideoPreset && (
              <div className="pt-1.5 border-t border-cyan-900/50 space-y-1.5">
                <div className="flex items-center justify-between text-[10px] text-cyan-300">
                  <span>合成SBSステレオ動画デモ (3シーン):</span>
                  <span className="font-mono text-slate-300">
                    {stereoVideoSource?.kind === 'preset' &&
                    STEREO_VIDEO_PRESETS[stereoVideoSource.presetId]?.kind === 'synthetic'
                      ? STEREO_VIDEO_PRESETS[stereoVideoSource.presetId].shortLabel
                      : '未選択'}
                  </span>
                </div>
                <div className="flex flex-wrap gap-1" role="group" aria-label="合成SBSステレオ動画選択">
                  {SYNTHETIC_STEREO_VIDEO_PRESET_IDS.map((presetId) => {
                    const preset = STEREO_VIDEO_PRESETS[presetId];
                    const isSelected =
                      stereoVideoSource?.kind === 'preset' &&
                      stereoVideoSource.presetId === presetId;
                    return (
                      <button
                        key={presetId}
                        type="button"
                        onClick={() => onSelectStereoVideoPreset(presetId)}
                        className={`px-2 py-1 rounded text-[10px] font-semibold border transition-colors whitespace-nowrap ${
                          isSelected
                            ? 'bg-cyan-700 border-cyan-400 text-white'
                            : 'bg-slate-900/80 border-slate-700 text-slate-300 hover:bg-slate-800'
                        }`}
                      >
                        {preset.shortLabel}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* 2. Free Open-License Side-by-Side Stereo Videos & Local Video Upload */}
            {(onSelectStereoVideoPreset || onUploadStereoVideoFile) && (
              <div className="pt-1.5 border-t border-cyan-900/50 space-y-1.5">
                <div className="flex items-center justify-between text-[10px] text-cyan-300">
                  <span>フリー素材SBSステレオ動画 / ファイル読込:</span>
                  <span className="font-mono text-slate-300">
                    {stereoVideoSource?.kind === 'preset' &&
                    STEREO_VIDEO_PRESETS[stereoVideoSource.presetId]?.kind === 'free-preset'
                      ? STEREO_VIDEO_PRESETS[stereoVideoSource.presetId].shortLabel
                      : stereoVideoSource?.kind === 'uploaded-file'
                        ? stereoVideoSource.fileName
                        : '未選択'}
                  </span>
                </div>
                <div className="flex flex-wrap gap-1" role="group" aria-label="フリー素材SBSステレオ動画選択">
                  {onSelectStereoVideoPreset &&
                    FREE_STEREO_VIDEO_PRESET_IDS.map((presetId) => {
                      const preset = STEREO_VIDEO_PRESETS[presetId];
                      const isSelected =
                        stereoVideoSource?.kind === 'preset' &&
                        stereoVideoSource.presetId === presetId;
                      return (
                        <button
                          key={presetId}
                          type="button"
                          onClick={() => onSelectStereoVideoPreset(presetId)}
                          className={`px-2 py-1 rounded text-[10px] font-semibold border transition-colors whitespace-nowrap ${
                            isSelected
                              ? 'bg-emerald-700 border-emerald-400 text-white'
                              : 'bg-slate-900/80 border-slate-700 text-slate-300 hover:bg-slate-800'
                          }`}
                        >
                          {preset.shortLabel}
                        </button>
                      );
                    })}
                  {onUploadStereoVideoFile && (
                    <label
                      className={`px-2 py-1 rounded text-[10px] font-semibold border cursor-pointer transition-colors whitespace-nowrap ${
                        stereoVideoSource?.kind === 'uploaded-file'
                          ? 'bg-emerald-700 border-emerald-400 text-white'
                          : 'bg-slate-900/80 border-slate-700 text-slate-300 hover:bg-slate-800'
                      }`}
                    >
                      <span>SBS動画ファイル...</span>
                      <input
                        type="file"
                        accept="video/*,.mp4,.webm,.mov,.ogg"
                        aria-label="Side-by-Sideステレオ動画ファイルを選択"
                        className="hidden"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) onUploadStereoVideoFile(file);
                          e.currentTarget.value = '';
                        }}
                      />
                    </label>
                  )}
                </div>
              </div>
            )}

            {/* Active Stereo Video Metadata & Calibration Sync */}
            {stereoVideoSource && (
              <div
                aria-label="選択中のステレオ動画情報"
                className="p-2 rounded bg-slate-950/75 border border-slate-800/90 space-y-1 text-[10px]"
              >
                {stereoVideoSource.kind === 'preset' ? (
                  (() => {
                    const preset = STEREO_VIDEO_PRESETS[stereoVideoSource.presetId];
                    return (
                      <>
                        <div className="font-semibold text-cyan-200">{preset.label}</div>
                        <p className="text-slate-300 leading-relaxed">{preset.description}</p>
                        <div className="flex flex-wrap items-center gap-1.5 text-slate-400 font-mono pt-0.5">
                          <span>{preset.eyeWidth * 2}×{preset.eyeHeight} SBS</span>
                          <span aria-hidden="true">·</span>
                          <span>片目 {preset.eyeWidth}×{preset.eyeHeight}</span>
                          <span aria-hidden="true">·</span>
                          <span>{preset.fps} FPS</span>
                          <span aria-hidden="true">·</span>
                          <span>{preset.license}</span>
                        </div>
                        <div className="flex flex-wrap items-center justify-between gap-2 text-slate-400 font-mono">
                          <span className="truncate">{preset.attribution}</span>
                          {preset.videoUrl && (
                            <a
                              href={preset.upstreamUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-emerald-400 hover:text-emerald-300 underline whitespace-nowrap"
                            >
                              Wikimedia原典 ↗
                            </a>
                          )}
                        </div>
                        <div className="flex items-center justify-between gap-2 pt-1 border-t border-slate-800/80">
                          <span className="font-mono text-slate-400">
                            推奨校正: B={preset.recommendedBaselineMm}mm · f={preset.recommendedFocalPx}px
                          </span>
                          <button
                            type="button"
                            onClick={() => {
                              updateInferenceParam('baselineMm', preset.recommendedBaselineMm);
                              updateInferenceParam('focalLengthPx', preset.recommendedFocalPx);
                            }}
                            className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-cyan-300 font-semibold transition-colors whitespace-nowrap"
                          >
                            推奨校正値を適用
                          </button>
                        </div>
                      </>
                    );
                  })()
                ) : (
                  <div className="flex items-center justify-between text-slate-300 font-mono">
                    <span>ローカル動画: {stereoVideoSource.fileName}</span>
                    <span>カスタムSBS映像</span>
                  </div>
                )}
              </div>
            )}

            {/* 3. Live Camera & Still Image Presets */}
            {onSelectStereoStillPreset && (
              <div className="pt-1.5 border-t border-cyan-900/50 space-y-1.5">
                <div className="flex items-center justify-between text-[10px] text-cyan-300">
                  <span>入力切替 (カメラ / テストSBS画像):</span>
                  <span className="font-mono">
                    {stereoVideoSource
                      ? stereoVideoSource.label
                      : stillImageLabel ?? 'カメラ映像'}
                  </span>
                </div>
                <div className="flex flex-wrap gap-1">
                  <button
                    type="button"
                    onClick={() => onSelectStereoStillPreset('camera')}
                    className={`px-2 py-1 rounded text-[10px] font-semibold border transition-colors ${
                      !stillImageLabel && !stereoVideoSource
                        ? 'bg-blue-600 border-blue-400 text-white'
                        : 'bg-slate-900/80 border-slate-700 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    カメラ映像
                  </button>
                  <button
                    type="button"
                    onClick={() => onSelectStereoStillPreset('stereo_blocks')}
                    className={`px-2 py-1 rounded text-[10px] font-semibold border transition-colors ${
                      stillImageLabel === 'stereo_blocks' && !stereoVideoSource
                        ? 'bg-cyan-700 border-cyan-400 text-white'
                        : 'bg-slate-900/80 border-slate-700 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    検証: 深度ブロック (SBS)
                  </button>
                  <button
                    type="button"
                    onClick={() => onSelectStereoStillPreset('stereo_corridor')}
                    className={`px-2 py-1 rounded text-[10px] font-semibold border transition-colors ${
                      stillImageLabel === 'stereo_corridor' && !stereoVideoSource
                        ? 'bg-cyan-700 border-cyan-400 text-white'
                        : 'bg-slate-900/80 border-slate-700 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    検証: 遠近シーン (SBS)
                  </button>
                  {onUploadStillImage && (
                    <label className="px-2 py-1 rounded text-[10px] font-semibold border bg-slate-900/80 border-slate-700 text-slate-300 hover:bg-slate-800 cursor-pointer">
                      <span>SBS画像...</span>
                      <input
                        type="file"
                        accept="image/*"
                        aria-label="Side-by-Sideステレオ画像ファイルを選択"
                        className="hidden"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) onUploadStillImage(file);
                          e.currentTarget.value = '';
                        }}
                      />
                    </label>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 1: Detection List OR Stereo Depth Telemetry/Pins */}
        {activeTab === 'detection' && activeModel === 'stereo-depth' && (
          <div className="space-y-3">
            <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-2">
              <div className="text-[11px] font-bold text-cyan-300 flex items-center justify-between">
                <span>フレーム視差サマリー</span>
                <span className="font-mono text-[10px] text-slate-400">
                  {stereoDepthMap ? `${stereoDepthMap.sourceEyeWidth}×${stereoDepthMap.sourceEyeHeight}px` : '未取得'}
                </span>
              </div>
              {stereoDepthMap ? (
                <div className="grid grid-cols-2 gap-2 font-mono tabular-nums text-[11px]">
                  <div className="p-2 rounded bg-slate-950/70 border border-slate-800/80">
                    <div className="text-[10px] text-slate-400 font-sans">最小〜最大視差</div>
                    <div className="text-slate-100 font-semibold mt-0.5">
                      {stereoDepthMap.minDisparity.toFixed(1)} .. {stereoDepthMap.maxDisparity.toFixed(1)} px
                    </div>
                  </div>
                  <div className="p-2 rounded bg-slate-950/70 border border-slate-800/80">
                    <div className="text-[10px] text-slate-400 font-sans">中央値 / 有効画素</div>
                    <div className="text-emerald-400 font-semibold mt-0.5">
                      {stereoDepthMap.medianDisparity.toFixed(1)} px · {Math.round(stereoDepthMap.validRatio * 100)}%
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-[11px] text-slate-400">
                  カメラ映像、合成SBS動画、または「検証: 深度ブロック (SBS)」を選択すると視差統計を表示します。
                </p>
              )}

              {/* Ground-Truth Target Reference List for Active Stereo Video */}
              {stereoVideoRegions.length > 0 && (
                <div className="pt-2 border-t border-slate-800 space-y-1.5">
                  <div className="flex items-center justify-between text-[10px] text-slate-400">
                    <span>現在フレームの基準ターゲット視差 (クリックでピン配置):</span>
                  </div>
                  <div className="grid grid-cols-1 gap-1">
                    {stereoVideoRegions.map((reg) => {
                      const measured = stereoDepthMap
                        ? inspectDisparityMapAtPoint(
                            stereoDepthMap,
                            reg.normX,
                            reg.normY,
                            stereoBaselineMm,
                            stereoFocalPx
                          )
                        : null;
                      return (
                        <div
                          key={reg.id}
                          className="px-2 py-1.5 rounded bg-slate-950/80 border border-slate-800/90 flex items-center justify-between gap-2 text-[11px] font-mono tabular-nums"
                        >
                          <div className="flex items-center gap-1.5 min-w-0">
                            <span className="font-sans font-medium text-slate-200 truncate">
                              {reg.label}
                            </span>
                            <span className="text-slate-400">基準 {reg.disparityPx}px</span>
                            {measured && measured.valid && (
                              <>
                                <span aria-hidden="true" className="text-slate-600">·</span>
                                <span className="text-cyan-300">
                                  実測 {measured.disparityPx.toFixed(1)}px
                                </span>
                              </>
                            )}
                          </div>
                          {measured && onPinStereoProbe && (
                            <button
                              type="button"
                              onClick={() => onPinStereoProbe(measured)}
                              className="px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[10px] font-sans font-semibold text-cyan-300 shrink-0"
                            >
                              ピン追加
                            </button>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5">
              <div className="text-[11px] font-semibold text-slate-300">カーソル座標のリアルタイム計測</div>
              {stereoHoverProbe ? (
                <div className="font-mono tabular-nums text-xs text-cyan-200 flex flex-wrap items-center justify-between gap-2 bg-slate-950/70 border border-slate-800 rounded px-2.5 py-2">
                  <span>({stereoHoverProbe.pixelX}, {stereoHoverProbe.pixelY})</span>
                  <span>
                    {stereoHoverProbe.valid
                      ? `視差 ${stereoHoverProbe.disparityPx.toFixed(2)} px`
                      : '視差未確定'}
                  </span>
                  {stereoShowCalibDepth && (
                    <span className="text-emerald-300 font-semibold">
                      {stereoHoverProbe.depthMeters !== null
                        ? `${stereoHoverProbe.depthMeters.toFixed(2)} m`
                        : '--'}
                    </span>
                  )}
                </div>
              ) : (
                <p className="text-[11px] text-slate-400">
                  プレビュー画像上にカーソルを合わせると現在座標の視差(px)と推定深度(m)を表示します。
                </p>
              )}
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between text-[11px] text-slate-400">
                <span>固定ポイント ({stereoPinnedProbes.length}件)</span>
                {stereoPinnedProbes.length > 0 && onClearStereoPins && (
                  <button
                    type="button"
                    onClick={onClearStereoPins}
                    className="flex items-center gap-1 text-[10px] text-slate-300 hover:text-rose-300"
                  >
                    <Trash2 className="w-3 h-3" />
                    <span>全ピン消去</span>
                  </button>
                )}
              </div>

              {stereoPinnedProbes.length === 0 ? (
                <div className="p-4 text-center border border-dashed border-slate-800 rounded-lg bg-slate-900/30 text-[11px] text-slate-400">
                  キャンバス上の任意の点をクリックすると計測値をピン固定して比較できます。
                </div>
              ) : (
                <div className="space-y-1.5">
                  {stereoPinnedProbes.map((pin) => {
                    const livePin = stereoDepthMap
                      ? inspectDisparityMapAtPoint(
                          stereoDepthMap,
                          pin.normX,
                          pin.normY,
                          stereoBaselineMm,
                          stereoFocalPx
                        )
                      : null;
                    const dispVal = livePin ? livePin.disparityPx : pin.disparityPx;
                    const isValid = livePin ? livePin.valid : pin.valid;
                    const depthVal = livePin ? livePin.depthMeters : pin.depthMeters;
                    return (
                      <div
                        key={pin.id}
                        className="p-2.5 rounded-lg border border-slate-800 bg-slate-900/70 flex items-center justify-between gap-2"
                      >
                        <div className="min-w-0">
                          <div className="flex items-center gap-1.5">
                            <span className="font-mono text-xs font-bold text-cyan-300">{pin.label}</span>
                            <span className="font-mono text-[10px] text-slate-400 tabular-nums">
                              ({pin.pixelX}, {pin.pixelY})
                            </span>
                            <span className="font-mono text-[10px] text-slate-400 truncate">
                              {pin.variantId}
                            </span>
                          </div>
                          <div className="font-mono text-xs tabular-nums text-slate-100 mt-0.5 flex items-center gap-2">
                            <span>{isValid ? `${dispVal.toFixed(2)} px` : '未確定'}</span>
                            {stereoShowCalibDepth && depthVal !== null && (
                              <span className="text-emerald-400 font-semibold">
                                {depthVal.toFixed(2)} m
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center gap-1 shrink-0">
                          {isValid && (
                            <button
                              type="button"
                              title="計測値を読み上げ"
                              onClick={() => {
                                const depthText =
                                  depthVal !== null
                                    ? `、推定距離${depthVal.toFixed(2)}メートル`
                                    : '';
                                speechService.speak(
                                  `${pin.label}。視差${dispVal.toFixed(1)}ピクセル${depthText}`,
                                  pin.id,
                                  { ...ttsConfig, enabled: true },
                                  'manual'
                                );
                              }}
                              className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800"
                            >
                              <Volume2 className="w-3.5 h-3.5" />
                            </button>
                          )}
                          {onRemoveStereoPin && (
                            <button
                              type="button"
                              title="このピンを削除"
                              onClick={() => onRemoveStereoPin(pin.id)}
                              className="p-1 rounded text-slate-400 hover:text-rose-300 hover:bg-slate-800"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'detection' && activeModel !== 'stereo-depth' && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span>{hasCurrentFrame ? `認識結果 (${detectedObjects.length}件)` : '認識結果は未取得です'}</span>
              <span>タップで個別発話</span>
            </div>

            {detectedObjects.length === 0 ? (
              <div className="p-6 text-center border border-dashed border-slate-800 rounded-lg bg-slate-900/30">
                <Layers className="w-6 h-6 mx-auto text-slate-600 mb-1.5" />
                <p className="text-slate-400">
                  {hasCurrentFrame ? '現在しきい値を超える物体はありません' : 'モデルとカメラの準備後、推論結果を表示します'}
                </p>
                <p className="text-[10px] text-slate-500 mt-1">
                  {hasCurrentFrame
                    ? activeModel === 'segformer'
                      ? '全画素の分類はマスクで確認できます'
                      : 'カメラを物体に向けてください'
                    : '未取得は正常な空検出を意味しません'}
                </p>
              </div>
            ) : (
              <div className="space-y-1.5">
                {detectedObjects.map((obj) => {
                  const isSelected = obj.id === selectedObjectId;
                  return (
                    <div
                      key={obj.id}
                      onClick={() => handleManualSpeak(obj)}
                      className={`p-2.5 rounded-lg border transition-all cursor-pointer flex items-center justify-between group ${
                        isSelected
                          ? 'bg-blue-950/40 border-blue-500/80 shadow-sm'
                          : 'bg-slate-900/60 border-slate-800/80 hover:border-slate-700 hover:bg-slate-900'
                      }`}
                    >
                      <div className="flex items-center gap-2.5">
                        <div
                          className="w-3 h-3 rounded-full shrink-0"
                          style={{ backgroundColor: obj.color }}
                        />
                        <div>
                          <div className="font-semibold text-white text-xs flex items-center gap-1">
                            <span>{obj.labelJa}</span>
                            <span className="text-[10px] text-slate-400 font-normal">({obj.label})</span>
                          </div>
                          <div className="text-[10px] text-slate-400 mt-0.5 flex items-center gap-1.5">
                            <span>{obj.category}</span>
                            <span>·</span>
                            <span className="font-mono text-emerald-400 tabular-nums">
                              {obj.confidenceKind ? `${confidenceDescription(obj)} ` : ''}
                              {Math.round(obj.confidence * 100)}%
                            </span>
                            {obj.areaFraction !== undefined && <span>面積 {Math.round(obj.areaFraction * 100)}%</span>}
                            {obj.maskPolygon && (
                              <>
                                <span>·</span>
                                <span className="text-blue-400">輪郭あり</span>
                              </>
                            )}
                          </div>
                        </div>
                      </div>

                      <button
                        title="読み上げる"
                        className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800"
                      >
                        <Volume2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Model Parameter Tuning */}
        {activeTab === 'params' && (
          <div className="space-y-4">
            {/* Model Header & Reset Button */}
            <div className="p-3 bg-slate-900/90 border border-slate-800 rounded-lg flex items-center justify-between gap-2">
              <div>
                <div className="text-[10px] text-slate-400 uppercase tracking-wider font-mono">対象モデル</div>
                <div className="font-bold text-white text-xs mt-0.5">{modelMetadata(activeModel).name}</div>
              </div>

              {onResetModelParams && (
                <button
                  onClick={onResetModelParams}
                  className="px-2.5 py-1.5 text-[11px] font-semibold text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-md border border-slate-700 transition-colors flex items-center gap-1 shrink-0"
                  title="このモデルの設定を初期値に戻す"
                >
                  <RotateCcw className="w-3 h-3 text-blue-400" />
                  <span>初期値にリセット</span>
                </button>
              )}
            </div>

            {/* Section 1: Inference & Post-processing Parameters */}
            <div className="space-y-3">
              <div className="text-[11px] font-bold text-blue-400 flex items-center gap-1.5 border-b border-slate-800/80 pb-1">
                <SlidersHorizontal className="w-3.5 h-3.5" />
                <span>推論・後処理パラメータ</span>
              </div>

              {/* Confidence Threshold (All models) */}
              <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-medium text-slate-200">
                    {activeModel === 'segformer' ? '平均softmaxスコアしきい値' : '認識しきい値 (Confidence)'}
                  </span>
                  <span className="font-mono text-blue-400 tabular-nums font-semibold">
                    {Math.round(confidenceThreshold * 100)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0.15"
                  max="0.90"
                  step="0.05"
                  value={confidenceThreshold}
                  onChange={(e) => updateInferenceParam('confidenceThreshold', parseFloat(e.target.value))}
                  className="w-full accent-blue-500 cursor-pointer"
                />
                <p className="text-[10px] text-slate-400 leading-tight">
                  {activeModel === 'segformer'
                    ? '抽出・表示対象とする領域クラスの平均確定確率の下限値'
                    : '物体検出候補を判定するスコアの下限値'}
                </p>
              </div>

              {/* Model Specific Parameters */}
              {activeModel === 'selfie-segmenter' && (
                <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-medium text-slate-200">最小人物画面比率 (Min Area)</span>
                    <span className="font-mono text-emerald-400 tabular-nums font-semibold">
                      {minAreaPercent.toFixed(1)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="10.0"
                    step="0.1"
                    value={minAreaPercent}
                    onChange={(e) => updateInferenceParam('minAreaPercent', parseFloat(e.target.value))}
                    className="w-full accent-blue-500 cursor-pointer"
                  />
                  <p className="text-[10px] text-slate-400 leading-tight">
                    背景ノイズや極小誤検出を除外するための最低ピクセル面積割合
                  </p>
                </div>
              )}

              {activeModel === 'mediapipe' && (
                <>
                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-medium text-slate-200">最大検出オブジェクト数</span>
                      <span className="font-mono text-emerald-400 tabular-nums font-semibold">
                        {maxDetections} 件
                      </span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="25"
                      step="1"
                      value={maxDetections}
                      onChange={(e) => updateInferenceParam('maxDetections', parseInt(e.target.value, 10))}
                      className="w-full accent-blue-500 cursor-pointer"
                    />
                    <p className="text-[10px] text-slate-400 leading-tight">
                      1フレームで同時に描画・保持する高スコア物体の最大数
                    </p>
                  </div>

                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-medium text-slate-200">枠ブレ平滑化 (EMA)</span>
                      <span className="font-mono text-emerald-400 tabular-nums font-semibold">
                        {Math.round(bboxSmoothing * 100)}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="0.80"
                      step="0.05"
                      value={bboxSmoothing}
                      onChange={(e) => updateInferenceParam('bboxSmoothing', parseFloat(e.target.value))}
                      className="w-full accent-blue-500 cursor-pointer"
                    />
                    <p className="text-[10px] text-slate-400 leading-tight">
                      前フレームの位置を保持して検出枠のチラつき・振動を抑える割合
                    </p>
                  </div>
                </>
              )}

              {activeModel === 'yolo-seg' && (
                <>
                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-medium text-slate-200">NMS 重複除去しきい値 (IoU)</span>
                      <span className="font-mono text-emerald-400 tabular-nums font-semibold">
                        {Math.round(iouThreshold * 100)}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.20"
                      max="0.85"
                      step="0.05"
                      value={iouThreshold}
                      onChange={(e) => updateInferenceParam('iouThreshold', parseFloat(e.target.value))}
                      className="w-full accent-blue-500 cursor-pointer"
                    />
                    <p className="text-[10px] text-slate-400 leading-tight">
                      重なる検出ボックスを非最大値抑制 (NMS) で結合・除外する割合
                    </p>
                  </div>

                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-medium text-slate-200">最大検出オブジェクト数</span>
                      <span className="font-mono text-emerald-400 tabular-nums font-semibold">
                        {maxDetections} 件
                      </span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="30"
                      step="1"
                      value={maxDetections}
                      onChange={(e) => updateInferenceParam('maxDetections', parseInt(e.target.value, 10))}
                      className="w-full accent-blue-500 cursor-pointer"
                    />
                    <p className="text-[10px] text-slate-400 leading-tight">
                      NMS 処理後に保持するインスタンスマスクの最大件数
                    </p>
                  </div>

                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-medium text-slate-200">マスク二値化スレショルド</span>
                      <span className="font-mono text-emerald-400 tabular-nums font-semibold">
                        {Math.round(maskThreshold * 100)}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.25"
                      max="0.80"
                      step="0.05"
                      value={maskThreshold}
                      onChange={(e) => updateInferenceParam('maskThreshold', parseFloat(e.target.value))}
                      className="w-full accent-blue-500 cursor-pointer"
                    />
                    <p className="text-[10px] text-slate-400 leading-tight">
                      プロトタイプマスク合成時にピクセルを輪郭に採用するシグモイド基準
                    </p>
                  </div>

                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-medium text-slate-200">枠ブレ平滑化 (EMA)</span>
                      <span className="font-mono text-emerald-400 tabular-nums font-semibold">
                        {Math.round(bboxSmoothing * 100)}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="0.80"
                      step="0.05"
                      value={bboxSmoothing}
                      onChange={(e) => updateInferenceParam('bboxSmoothing', parseFloat(e.target.value))}
                      className="w-full accent-blue-500 cursor-pointer"
                    />
                    <p className="text-[10px] text-slate-400 leading-tight">
                      ボックス枠移動の指数移動平均平滑化割合
                    </p>
                  </div>
                </>
              )}

              {activeModel === 'segformer' && (
                <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-medium text-slate-200">最小クラス画面比率 (Min Area)</span>
                    <span className="font-mono text-emerald-400 tabular-nums font-semibold">
                      {minAreaPercent.toFixed(1)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="5.0"
                    step="0.1"
                    value={minAreaPercent}
                    onChange={(e) => updateInferenceParam('minAreaPercent', parseFloat(e.target.value))}
                    className="w-full accent-blue-500 cursor-pointer"
                  />
                  <p className="text-[10px] text-slate-400 leading-tight">
                    一覧表示および領域ラベルから除外する極小カテゴリの最小画面割合
                  </p>
                </div>
              )}

              {activeModel === 'mirage' && !isExperimentalMirage && (
                <>
                  {/* Fast vs Full Analysis Mode Selector */}
                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-2">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-medium text-slate-200">Mirage 解析モード</span>
                      <span className="font-mono text-emerald-400 font-semibold text-[11px]">
                        {mirageAnalysisMode === 'fast' ? '高速モード (320px)' : 'フル解析 (512px)'}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-1.5" role="group" aria-label="Mirage 解析モード">
                      <button
                        type="button"
                        onClick={() => updateInferenceParam('analysisMode', 'fast')}
                        className={`p-2 rounded border text-left transition-colors ${
                          mirageAnalysisMode === 'fast'
                            ? 'bg-blue-950/70 border-blue-500 text-white font-semibold'
                            : 'bg-slate-800/60 border-slate-700 text-slate-300 hover:bg-slate-800'
                        }`}
                      >
                        <div className="text-[11px]">高速モード（推奨）</div>
                        <div className="text-[10px] text-slate-400 font-normal mt-0.5">
                          物体・輪郭・OCR優先（WASM Worker/高速）
                        </div>
                      </button>
                      <button
                        type="button"
                        onClick={() => updateInferenceParam('analysisMode', 'full')}
                        className={`p-2 rounded border text-left transition-colors ${
                          mirageAnalysisMode === 'full'
                            ? 'bg-blue-950/70 border-blue-500 text-white font-semibold'
                            : 'bg-slate-800/60 border-slate-700 text-slate-300 hover:bg-slate-800'
                        }`}
                      >
                        <div className="text-[11px]">フル解析モード</div>
                        <div className="text-[10px] text-slate-400 font-normal mt-0.5">
                          全項目・細部文字（512px）
                        </div>
                      </button>
                    </div>
                    <p className="text-[10px] text-slate-400 leading-tight">
                      ヘッダーの「バリアント」から公式 WASM (.cact 版) と実験用部分ONNX (WASM / GPU) を切り替えられます
                    </p>
                  </div>

                  {/* Include OCR Toggle */}
                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg flex items-center justify-between gap-2">
                    <div>
                      <div className="font-medium text-slate-200 text-xs">OCR テキスト領域の表示・読み上げ</div>
                      <div className="text-[10px] text-slate-400 mt-0.5">
                        検出物体・輪郭マスクに加えて画像内の文字領域も枠表示・音声発話
                      </div>
                    </div>
                    <input
                      type="checkbox"
                      aria-label="OCR テキスト領域の表示・読み上げ"
                      checked={mirageIncludeOcr}
                      onChange={(e) => updateInferenceParam('includeOcr', e.target.checked)}
                      className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer shrink-0"
                    />
                  </div>

                  {/* Max Detections */}
                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-medium text-slate-200">最大検出数（物体＋OCR）</span>
                      <span className="font-mono text-emerald-400 tabular-nums font-semibold">
                        {maxDetections} 件
                      </span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="30"
                      step="1"
                      value={maxDetections}
                      onChange={(e) => updateInferenceParam('maxDetections', parseInt(e.target.value, 10))}
                      className="w-full accent-blue-500 cursor-pointer"
                    />
                  </div>

                  {/* NMS IoU */}
                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-medium text-slate-200">NMS 重複除去しきい値 (IoU)</span>
                      <span className="font-mono text-emerald-400 tabular-nums font-semibold">
                        {Math.round(iouThreshold * 100)}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.20"
                      max="0.85"
                      step="0.05"
                      value={iouThreshold}
                      onChange={(e) => updateInferenceParam('iouThreshold', parseFloat(e.target.value))}
                      className="w-full accent-blue-500 cursor-pointer"
                    />
                  </div>
                </>
              )}

              {isExperimentalMirage && (
                <>
                  {/* Experimental ONNX Pipeline Mode & Decoder Language */}
                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-2">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-medium text-slate-200">ONNX パイプライン構成</span>
                      <span className="font-mono text-cyan-400 font-semibold text-[11px]">
                        {mirageAnalysisMode === 'full' ? 'フル推論 (全4グラフ)' : '高速 (Encoder候補のみ)'}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-1.5" role="group" aria-label="ONNX パイプライン構成">
                      <button
                        type="button"
                        onClick={() => updateInferenceParam('analysisMode', 'fast')}
                        className={`p-2 rounded border text-left transition-colors ${
                          mirageAnalysisMode === 'fast'
                            ? 'bg-cyan-950/70 border-cyan-500 text-white font-semibold'
                            : 'bg-slate-800/60 border-slate-700 text-slate-300 hover:bg-slate-800'
                        }`}
                      >
                        <div className="text-[11px]">高速（候補のみ）</div>
                        <div className="text-[10px] text-slate-400 font-normal mt-0.5">
                          Encoder のみ (34.3MB)
                        </div>
                      </button>
                      <button
                        type="button"
                        onClick={() => updateInferenceParam('analysisMode', 'full')}
                        className={`p-2 rounded border text-left transition-colors ${
                          mirageAnalysisMode === 'full'
                            ? 'bg-cyan-950/70 border-cyan-500 text-white font-semibold'
                            : 'bg-slate-800/60 border-slate-700 text-slate-300 hover:bg-slate-800'
                        }`}
                      >
                        <div className="text-[11px]">フル推論（OCR・命名）</div>
                        <div className="text-[10px] text-slate-400 font-normal mt-0.5">
                          OCR + CrossKV + デコーダ
                        </div>
                      </button>
                    </div>
                    {mirageAnalysisMode === 'full' && (
                      <div className="pt-1.5 border-t border-slate-800/80 space-y-1.5">
                        <div className="flex items-center justify-between gap-2 text-xs">
                          <label htmlFor="mirage-onnx-language" className="text-slate-300">
                            デコーダ出力言語
                          </label>
                          <select
                            id="mirage-onnx-language"
                            aria-label="Mirage 出力言語"
                            value={mirageLanguage}
                            onChange={(e) => updateInferenceParam('mirageLanguage', e.target.value)}
                            className="bg-slate-950 border border-slate-700 rounded px-2 py-1 text-[11px] text-slate-100"
                          >
                            {MIRAGE_SUPPORTED_LANGUAGES.map((code) => (
                              <option key={code} value={code}>
                                {MIRAGE_LANGUAGE_LABELS[code] ?? code}
                              </option>
                            ))}
                          </select>
                        </div>
                        <label className="flex items-center justify-between text-xs cursor-pointer">
                          <span className="text-slate-300">CTC OCR 文字認識 (reader.onnx)</span>
                          <input
                            type="checkbox"
                            aria-label="CTC OCR 文字認識"
                            checked={mirageIncludeOcr}
                            onChange={(e) => updateInferenceParam('includeOcr', e.target.checked)}
                            className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer"
                          />
                        </label>
                      </div>
                    )}
                  </div>

                  {/* Candidate Post-processing / NMS Policy */}
                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-2">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-medium text-slate-200">候補選別・NMS方針</span>
                      <span className="font-mono text-cyan-400 font-semibold text-[11px]">
                        {experimentalNmsIou === null ? 'NMSなし (既定)' : `Greedy NMS IoU ${experimentalNmsIou}`}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-1.5" role="group" aria-label="候補選別・NMS方針">
                      <button
                        type="button"
                        onClick={() => updateInferenceParam('experimentalNmsIou', null)}
                        className={`p-2 rounded border text-left transition-colors ${
                          experimentalNmsIou === null
                            ? 'bg-cyan-950/70 border-cyan-500 text-white font-semibold'
                            : 'bg-slate-800/60 border-slate-700 text-slate-300 hover:bg-slate-800'
                        }`}
                      >
                        <div className="text-[11px]">NMSなし（既定）</div>
                        <div className="text-[10px] text-slate-400 font-normal mt-0.5">
                          3×3 local peaks（52/54一致・余分80件）
                        </div>
                      </button>
                      <button
                        type="button"
                        onClick={() => updateInferenceParam('experimentalNmsIou', 0.5)}
                        className={`p-2 rounded border text-left transition-colors ${
                          experimentalNmsIou === 0.5
                            ? 'bg-cyan-950/70 border-cyan-500 text-white font-semibold'
                            : 'bg-slate-800/60 border-slate-700 text-slate-300 hover:bg-slate-800'
                        }`}
                      >
                        <div className="text-[11px]">Greedy NMS (IoU 0.5)</div>
                        <div className="text-[10px] text-slate-400 font-normal mt-0.5">
                          重複候補抑制仮説（51/54一致・余分27件）
                        </div>
                      </button>
                    </div>
                  </div>

                  {/* Proposal Layer Toggles */}
                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-2">
                    <div className="font-medium text-slate-200 text-xs">部分ONNX 領域候補・マスク表示対象</div>
                    <label className="flex items-center justify-between text-xs cursor-pointer">
                      <span className="text-slate-300">物体領域候補を表示（最大64件・水色枠）</span>
                      <input
                        type="checkbox"
                        checked={showObjectProposals}
                        onChange={(e) => updateInferenceParam('showObjectProposals', e.target.checked)}
                        className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer"
                      />
                    </label>
                    <label className="flex items-center justify-between text-xs cursor-pointer">
                      <span className="text-slate-300">テキスト領域候補を表示（最大32件・琥珀枠）</span>
                      <input
                        type="checkbox"
                        checked={showTextProposals}
                        onChange={(e) => updateInferenceParam('showTextProposals', e.target.checked)}
                        className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer"
                      />
                    </label>
                    <label className="flex items-center justify-between text-xs cursor-pointer">
                      <span className="text-slate-300">128×128 診断用物体マスクを合成</span>
                      <input
                        type="checkbox"
                        checked={showDiagnosticMasks}
                        onChange={(e) => updateInferenceParam('showDiagnosticMasks', e.target.checked)}
                        className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer"
                      />
                    </label>
                  </div>

                  {/* Optional Official WASM Overlay Toggle */}
                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg flex items-center justify-between gap-2">
                    <div>
                      <div className="font-medium text-slate-200 text-xs">公式.cact WASM結果の比較オーバーレイ</div>
                      <div className="text-[10px] text-slate-400 mt-0.5">
                        有効時は <code className="font-mono">[ONNX候補]</code> と <code className="font-mono">[公式WASM]</code> の出力元を明示区別して重ね表示します
                      </div>
                    </div>
                    <input
                      type="checkbox"
                      aria-label="公式.cact WASM結果の比較オーバーレイ"
                      checked={overlayNativeWasm}
                      onChange={(e) => updateInferenceParam('overlayNativeWasm', e.target.checked)}
                      className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer shrink-0"
                    />
                  </div>

                  {/* Execution Diagnostics Readout */}
                  {experimentalDiagnostics && (
                    <div
                      aria-label="実験用部分ONNX実行診断"
                      className="p-3 bg-slate-950/90 border border-slate-800 rounded-lg space-y-1 font-mono text-[10px] text-slate-300"
                    >
                      <div className="text-cyan-400 font-sans font-bold text-[11px] mb-1">
                        {experimentalDiagnostics.pipelineMode === 'full_pipeline'
                          ? '実験用 Mirage ONNX フル推論 診断'
                          : '実験用部分ONNX 直近推論診断'}
                      </div>
                      <div>パイプライン: {experimentalDiagnostics.pipelineMode === 'full_pipeline' ? 'full_pipeline (Encoder+Reader+CrossKV+Decoder)' : 'encoder_only (候補抽出のみ)'}</div>
                      <div>要求EP: {experimentalDiagnostics.requestedProvider} ／ 設定EP: {experimentalDiagnostics.configuredProviders.join(' → ')} ／ 実効EP: {experimentalDiagnostics.activeProvider}</div>
                      {experimentalDiagnostics.webgpuAdapter && (
                        <div className={experimentalDiagnostics.webgpuAdapter.isSoftwareAdapter || !experimentalDiagnostics.webgpuAdapter.available ? 'text-amber-300' : 'text-emerald-300/90'}>
                          WebGPUアダプター: {experimentalDiagnostics.webgpuAdapter.available
                            ? `${[experimentalDiagnostics.webgpuAdapter.vendor, experimentalDiagnostics.webgpuAdapter.architecture, experimentalDiagnostics.webgpuAdapter.description].filter(Boolean).join(' / ') || '検出済'}${experimentalDiagnostics.webgpuAdapter.isSoftwareAdapter ? ' (ソフトウェアエミュレーション)' : ' (ハードウェアGPU)'}`
                            : `利用不可${experimentalDiagnostics.webgpuAdapter.error ? ` (${experimentalDiagnostics.webgpuAdapter.error})` : ''}`}
                        </div>
                      )}
                      {experimentalDiagnostics.fallbackReason && (
                        <div className="text-amber-300">フォールバック理由: {experimentalDiagnostics.fallbackReason}</div>
                      )}
                      <div>入力 valid_hw: [{experimentalDiagnostics.validHw[0]}, {experimentalDiagnostics.validHw[1]}] (元画像 {experimentalDiagnostics.metadata.W}×{experimentalDiagnostics.metadata.H} · ceil32)</div>
                      <div>座標逆変換倍率: [{experimentalDiagnostics.metadata.coordinate_scale.map(v => v.toFixed(3)).join(', ')}]</div>
                      <div>抽出候補: 物体 {experimentalDiagnostics.objectProposalCount}件 ／ テキスト {experimentalDiagnostics.textProposalCount}件{experimentalDiagnostics.nativeOverlayEnabled ? ` ／ 公式WASM ${experimentalDiagnostics.nativeOverlayCount}件` : ''}</div>
                      {experimentalDiagnostics.pipelineMode === 'full_pipeline' && (
                        <>
                          <div>
                            デコーダ: 言語 {experimentalDiagnostics.decoderLanguage ?? 'en'} · {experimentalDiagnostics.generatedTokenCount ?? 0}トークン ({experimentalDiagnostics.decoderStepsExecuted ?? 0}step) · OCR {experimentalDiagnostics.ocrLineCount ?? 0}行
                          </div>
                          {experimentalDiagnostics.assembledObservation && (
                            <div className="text-emerald-300/90 break-words">
                              領域: {experimentalDiagnostics.assembledObservation.regions.length}件 ／ 関係: {experimentalDiagnostics.assembledObservation.relations.length}件
                              {experimentalDiagnostics.assembledObservation.text.length > 0
                                ? ` ／ OCR: ${experimentalDiagnostics.assembledObservation.text.map(t => t.string).slice(0, 3).join(' | ')}`
                                : ''}
                            </div>
                          )}
                        </>
                      )}
                      <div>処理時間: 前処理 {experimentalDiagnostics.timingsMs.preprocess.toFixed(1)}ms · 推論 {experimentalDiagnostics.timingsMs.inference.toFixed(1)}ms · 後処理 {experimentalDiagnostics.timingsMs.postprocess.toFixed(1)}ms</div>
                    </div>
                  )}
                </>
              )}
              {activeModel === 'stereo-depth' && (
                <>
                  {/* Execution Backend Selector (WebGPU / WASM) */}
                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-2">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-medium text-slate-200">実行バックエンド (WebGPU / WASM)</span>
                      <span className="font-mono text-cyan-400 text-[11px] tabular-nums">
                        {modelState?.status === 'ready' &&
                        modelState.model === 'stereo-depth' &&
                        modelState.backend
                          ? modelState.backend.configuredProviders.join(' → ')
                          : modelState?.status === 'loading'
                            ? '読込中...'
                            : stereoBackendPref}
                      </span>
                    </div>
                    <div
                      className="grid grid-cols-3 gap-1.5"
                      role="group"
                      aria-label="ステレオ実行バックエンド"
                    >
                      {([
                        ['auto', '自動 (GPU優先)'],
                        ['webgpu', 'WebGPU 強制'],
                        ['wasm', 'WASM (CPU)'],
                      ] as const).map(([pref, label]) => (
                        <button
                          key={pref}
                          type="button"
                          aria-pressed={stereoBackendPref === pref}
                          onClick={() => updateInferenceParam('backendPreference', pref)}
                          className={`py-1.5 px-2 rounded border text-[11px] transition-colors ${
                            stereoBackendPref === pref
                              ? 'bg-cyan-950/70 border-cyan-500 text-white font-semibold'
                              : 'bg-slate-800/60 border-slate-700 text-slate-300 hover:bg-slate-800'
                          }`}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                    {modelState?.status === 'ready' &&
                      modelState.model === 'stereo-depth' &&
                      modelState.backend && (
                        <div className="font-mono tabular-nums text-[10px] text-slate-300 bg-slate-950/75 border border-slate-800 rounded px-2 py-1.5 space-y-0.5">
                          <div>
                            runtime: {modelState.backend.runtime} · 要求:{' '}
                            {modelState.backend.requestedProvider}
                          </div>
                          {modelState.backend.fallbackReason && (
                            <div className="text-amber-300 break-words font-sans">
                              フォールバック理由: {modelState.backend.fallbackReason}
                            </div>
                          )}
                        </div>
                      )}
                  </div>

                  {/* Local ONNX File Loader for Neural Variants */}
                  {activeStereoProfile.engineType === 'onnx' && (
                    <div className="p-3 bg-slate-900/80 border border-cyan-800/50 rounded-lg space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-cyan-300">ONNX モデル重みファイル</span>
                        <span className="font-mono text-[10px] text-slate-400">{activeStereoProfile.fileName}</span>
                      </div>
                      <p className="text-[10px] text-slate-400 leading-relaxed">
                        {activeStereoProfile.licenseNote}
                      </p>
                      {localStereoOnnx ? (
                        <div className="flex items-center justify-between gap-2 p-2 rounded bg-emerald-950/40 border border-emerald-700/50 text-[11px]">
                          <span className="font-mono text-emerald-300 truncate">
                            読込済: {localStereoOnnx.fileName} ({(localStereoOnnx.size / (1024 * 1024)).toFixed(1)} MB)
                          </span>
                          {onClearStereoOnnxFile && (
                            <button
                              type="button"
                              onClick={() => onClearStereoOnnxFile(activeStereoVariantId)}
                              className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-[10px] shrink-0"
                            >
                              解除
                            </button>
                          )}
                        </div>
                      ) : activeStereoProfile.onlineDelivery === 'direct-cors' ||
                        activeStereoProfile.onlineDelivery === 'cors-proxy' ? (
                        <div className="p-2 rounded bg-cyan-950/35 border border-cyan-700/45 text-[10px] text-cyan-200 leading-relaxed">
                          オンライン自動取得対応 ({activeStereoProfile.onlineSourceLabel ?? 'HTTPS'} · {activeStereoProfile.defaultWidth}×{activeStereoProfile.defaultHeight})。必要に応じてローカルの .onnx ファイルで上書きできます。
                        </div>
                      ) : (
                        <div className="text-[10px] text-amber-300">
                          未読込です。ローカルの .onnx ファイルを選択して WebGPU / WASM 推論を開始してください。
                        </div>
                      )}
                      {onLoadStereoOnnxFile && (
                        <label className="w-full py-1.5 px-3 rounded bg-cyan-900/60 hover:bg-cyan-800/70 border border-cyan-600/50 text-cyan-100 text-xs font-semibold flex items-center justify-center gap-1.5 cursor-pointer transition-colors">
                          <Upload className="w-3.5 h-3.5" />
                          <span>{localStereoOnnx ? '別の .onnx を読み込む' : `${activeStereoProfile.fileName} (.onnx) を選択`}</span>
                          <input
                            type="file"
                            accept=".onnx"
                            aria-label="ステレオONNXモデルファイルを選択"
                            className="hidden"
                            onChange={(e) => {
                              const file = e.target.files?.[0];
                              if (file) onLoadStereoOnnxFile(activeStereoVariantId, file);
                              e.currentTarget.value = '';
                            }}
                          />
                        </label>
                      )}
                    </div>
                  )}

                  {/* Stereo View Mode */}
                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-2">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-medium text-slate-200">表示レイアウト (Side-by-Side)</span>
                      <span className="font-mono text-cyan-400 text-[11px]">{stereoViewMode}</span>
                    </div>
                    <div className="grid grid-cols-3 gap-1.5" role="group" aria-label="ステレオ表示レイアウト">
                      {([
                        ['left-overlay', '左+重畳'],
                        ['split-3pane', '3画面比較'],
                        ['depth-only', '深度のみ'],
                      ] as const).map(([mode, label]) => (
                        <button
                          key={mode}
                          type="button"
                          onClick={() => updateInferenceParam('viewMode', mode)}
                          className={`py-1.5 px-2 rounded border text-[11px] transition-colors ${
                            stereoViewMode === mode
                              ? 'bg-blue-950/70 border-blue-500 text-white font-semibold'
                              : 'bg-slate-800/60 border-slate-700 text-slate-300 hover:bg-slate-800'
                          }`}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Colormap & Swap L/R */}
                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-2.5">
                    <div className="flex items-center justify-between gap-2 text-xs">
                      <label htmlFor="stereo-colormap-select" className="font-medium text-slate-200">
                        カラーマップ
                      </label>
                      <select
                        id="stereo-colormap-select"
                        aria-label="視差カラーマップ"
                        value={stereoColormap}
                        onChange={(e) => updateInferenceParam('colormap', e.target.value as StereoColormap)}
                        className="bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-slate-100"
                      >
                        <option value="turbo">Turbo (高識別レインボー)</option>
                        <option value="viridis">Viridis (知覚均一)</option>
                        <option value="inferno">Inferno (高コントラスト)</option>
                        <option value="grayscale">Grayscale (グレースケール)</option>
                      </select>
                    </div>

                    <label className="flex items-center justify-between text-xs cursor-pointer">
                      <span className="text-slate-300">左右映像を入れ替える (交差法 / R|L 補正)</span>
                      <input
                        type="checkbox"
                        checked={stereoSwapLR}
                        onChange={(e) => updateInferenceParam('swapLeftRight', e.target.checked)}
                        className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer"
                      />
                    </label>

                    <div className="pt-2 border-t border-slate-800 space-y-1">
                      <label className="flex items-center justify-between gap-2 text-xs cursor-pointer">
                        <span className="text-slate-200 font-medium">
                          実カメラ入力を180°回転 (逆さ設置補正)
                        </span>
                        <input
                          type="checkbox"
                          aria-label="実カメラ入力を180°回転 (逆さ設置補正)"
                          checked={stereoCameraRotate180}
                          onChange={(e) => updateInferenceParam('cameraRotate180', e.target.checked)}
                          className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer shrink-0"
                        />
                      </label>
                      <div className="flex items-center justify-between text-[10px] text-slate-400">
                        <span>全体180°回転（左右目自動入替）</span>
                        <span
                          className={`font-mono tabular-nums ${
                            !isLiveStereoCameraActive
                              ? 'text-slate-400'
                              : stereoCameraRotate180
                                ? 'text-cyan-400'
                                : 'text-slate-400'
                          }`}
                        >
                          {!isLiveStereoCameraActive
                            ? '動画・静止画では無効'
                            : stereoCameraRotate180
                              ? '実カメラ適用中'
                              : 'OFF (正立入力)'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Disparity Scale Range */}
                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-2">
                    <label className="flex items-center justify-between text-xs cursor-pointer">
                      <span className="font-medium text-slate-200">視差レンジ自動正規化 (2%–98%)</span>
                      <input
                        type="checkbox"
                        checked={stereoAutoNorm}
                        onChange={(e) => updateInferenceParam('autoNormalizeDisparity', e.target.checked)}
                        className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer"
                      />
                    </label>
                    {!stereoAutoNorm && (
                      <div className="space-y-2 pt-1 border-t border-slate-800">
                        <div className="space-y-1">
                          <div className="flex justify-between text-[11px]">
                            <span className="text-slate-300">最小視差 (遠景下限)</span>
                            <span className="font-mono tabular-nums text-cyan-300">{stereoMinDispPx} px</span>
                          </div>
                          <input
                            type="range"
                            min="0"
                            max="64"
                            step="1"
                            value={stereoMinDispPx}
                            onChange={(e) => updateInferenceParam('minDisparityPx', parseInt(e.target.value, 10))}
                            className="w-full accent-blue-500 cursor-pointer"
                          />
                        </div>
                        <div className="space-y-1">
                          <div className="flex justify-between text-[11px]">
                            <span className="text-slate-300">最大視差 (近景上限)</span>
                            <span className="font-mono tabular-nums text-cyan-300">{stereoMaxDispPx} px</span>
                          </div>
                          <input
                            type="range"
                            min="16"
                            max="192"
                            step="4"
                            value={stereoMaxDispPx}
                            onChange={(e) => updateInferenceParam('maxDisparityPx', parseInt(e.target.value, 10))}
                            className="w-full accent-blue-500 cursor-pointer"
                          />
                        </div>
                      </div>
                    )}
                  </div>

                  {/* SAD / WGSL Disparity Search Parameters */}
                  {(activeStereoVariantId === 'stereo-sad-baseline' ||
                    activeStereoVariantId === 'stereo-webgpu-wgsl') && (
                    <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-2">
                      <div className="space-y-1">
                        <div className="flex justify-between items-center text-xs">
                          <span className="font-medium text-slate-200">
                            {activeStereoVariantId === 'stereo-webgpu-wgsl'
                              ? 'WGSL 探索最大視差'
                              : 'SAD 探索最大視差'}
                          </span>
                          <span className="font-mono text-emerald-400 tabular-nums font-semibold">
                            {stereoMaxDisparity} px
                          </span>
                        </div>
                        <input
                          type="range"
                          min="16"
                          max="128"
                          step="8"
                          value={stereoMaxDisparity}
                          onChange={(e) => updateInferenceParam('maxDisparity', parseInt(e.target.value, 10))}
                          className="w-full accent-blue-500 cursor-pointer"
                        />
                      </div>

                      <div className="flex items-center justify-between text-xs pt-1 border-t border-slate-800">
                        <span className="text-slate-300">
                          {activeStereoVariantId === 'stereo-webgpu-wgsl'
                            ? 'WGSL 窓サイズ (CPUフォールバック兼用)'
                            : 'SAD ブロックサイズ'}
                        </span>
                        <div className="flex gap-1">
                          {([5, 7, 9] as const).map((sz) => (
                            <button
                              key={sz}
                              type="button"
                              onClick={() => updateInferenceParam('sadWindowSize', sz)}
                              className={`px-2 py-0.5 rounded border font-mono text-[11px] ${
                                stereoSadWindowSize === sz
                                  ? 'bg-blue-600 border-blue-400 text-white'
                                  : 'bg-slate-800 border-slate-700 text-slate-300'
                              }`}
                            >
                              {sz}×{sz}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Stereo Camera Calibration */}
                  <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-2">
                    <label className="flex items-center justify-between text-xs cursor-pointer">
                      <span className="font-medium text-slate-200">校正深度 (m) を計算表示</span>
                      <input
                        type="checkbox"
                        checked={stereoShowCalibDepth}
                        onChange={(e) => updateInferenceParam('showCalibrationDepth', e.target.checked)}
                        className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer"
                      />
                    </label>
                    <div className="space-y-1">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-slate-300">カメラ基線長 (Baseline B)</span>
                        <span className="font-mono tabular-nums text-emerald-400">{stereoBaselineMm} mm</span>
                      </div>
                      <input
                        type="range"
                        min="10"
                        max="300"
                        step="5"
                        value={stereoBaselineMm}
                        onChange={(e) => updateInferenceParam('baselineMm', parseInt(e.target.value, 10))}
                        className="w-full accent-blue-500 cursor-pointer"
                      />
                    </div>
                    <div className="space-y-1">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-slate-300">焦点距離 (Focal Length f)</span>
                        <span className="font-mono tabular-nums text-emerald-400">{stereoFocalPx} px</span>
                      </div>
                      <input
                        type="range"
                        min="150"
                        max="2000"
                        step="25"
                        value={stereoFocalPx}
                        onChange={(e) => updateInferenceParam('focalLengthPx', parseInt(e.target.value, 10))}
                        className="w-full accent-blue-500 cursor-pointer"
                      />
                    </div>
                    <p className="text-[10px] font-mono text-slate-400">
                      Z = (f · B) / d = ({stereoFocalPx} · {(stereoBaselineMm / 1000).toFixed(3)}) / d [m]
                    </p>
                  </div>
                </>
              )}
            </div>

            {/* Section 2: Render & Overlay Display Options */}
            <div className="space-y-3 pt-1">
              <div className="text-[11px] font-bold text-blue-400 flex items-center gap-1.5 border-b border-slate-800/80 pb-1">
                <Eye className="w-3.5 h-3.5" />
                <span>表示・オーバーレイ設定</span>
              </div>

              {/* Mask Opacity (For models that generate masks or depth colormaps) */}
              {activeModel !== 'mediapipe' && (
                <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-medium text-slate-200">
                      {activeModel === 'stereo-depth' ? 'カラーマップ重畳不透明度' : '輪郭マスク不透明度'}
                    </span>
                    <span className="font-mono text-blue-400 tabular-nums font-semibold">
                      {Math.round(renderOptions.maskOpacity * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.95"
                    step="0.05"
                    value={renderOptions.maskOpacity}
                    onChange={(e) => onChangeRenderOptions({ maskOpacity: parseFloat(e.target.value) })}
                    className="w-full accent-blue-500 cursor-pointer"
                  />
                </div>
              )}

              {/* Render Option Checkboxes */}
              <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-2">
                {activeModel !== 'mediapipe' && (
                  <label className="flex items-center justify-between text-xs cursor-pointer">
                    <span className="text-slate-300">
                      {activeModel === 'stereo-depth' ? '視差カラーマップを重ねる' : '輪郭・セグメンテーションマスク描画'}
                    </span>
                    <input
                      type="checkbox"
                      checked={renderOptions.showMasks}
                      onChange={(e) => onChangeRenderOptions({ showMasks: e.target.checked })}
                      className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer"
                    />
                  </label>
                )}

                {activeModel !== 'stereo-depth' && (
                  <>
                    <label className="flex items-center justify-between text-xs cursor-pointer">
                      <span className="text-slate-300">外接バウンディングボックス描画</span>
                      <input
                        type="checkbox"
                        checked={renderOptions.showBoxes}
                        onChange={(e) => onChangeRenderOptions({ showBoxes: e.target.checked })}
                        className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer"
                      />
                    </label>

                    <label className="flex items-center justify-between text-xs cursor-pointer">
                      <span className="text-slate-300">クラス名・ラベル描画</span>
                      <input
                        type="checkbox"
                        checked={renderOptions.showLabels}
                        onChange={(e) => onChangeRenderOptions({ showLabels: e.target.checked })}
                        className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer"
                      />
                    </label>

                    <label className="flex items-center justify-between text-xs cursor-pointer">
                      <span className="text-slate-300">信頼度スコア数値表示</span>
                      <input
                        type="checkbox"
                        checked={renderOptions.showConfidence}
                        onChange={(e) => onChangeRenderOptions({ showConfidence: e.target.checked })}
                        className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer"
                      />
                    </label>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Tab 3: Speech & TTS Settings */}
        {activeTab === 'settings' && (
          <div className="space-y-3">
            {/* Master Speech Switch */}
            <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg flex items-center justify-between">
              <div>
                <div className="font-semibold text-white text-xs">音声読み上げ (TTS)</div>
                <div className="text-[10px] text-slate-400">初期状態: オフ</div>
              </div>
              <input
                type="checkbox"
                checked={ttsConfig.enabled}
                onChange={(e) => onChangeTTSConfig({ enabled: e.target.checked })}
                className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer"
              />
            </div>

            {/* Auto Readout Toggle */}
            <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg flex items-center justify-between">
              <div>
                <div className="font-semibold text-white text-xs">新規物体の自動読み上げ</div>
                <div className="text-[10px] text-slate-400">新しく検出された物体を自動発話</div>
              </div>
              <input
                type="checkbox"
                checked={ttsConfig.autoReadout}
                disabled={!ttsConfig.enabled}
                onChange={(e) => onChangeTTSConfig({ autoReadout: e.target.checked })}
                className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer disabled:opacity-40"
              />
            </div>

            {/* Debounce Cooldown */}
            <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5">
              <div className="flex justify-between items-center text-xs">
                <span className="font-medium text-slate-200">読み上げ間隔 (クールダウン)</span>
                <span className="font-mono text-emerald-400 tabular-nums font-semibold">
                  {ttsConfig.cooldownSec.toFixed(1)} 秒
                </span>
              </div>
              <input
                type="range"
                min="1.0"
                max="8.0"
                step="0.5"
                value={ttsConfig.cooldownSec}
                disabled={!ttsConfig.enabled || !ttsConfig.autoReadout}
                onChange={(e) => onChangeTTSConfig({ cooldownSec: parseFloat(e.target.value) })}
                className="w-full accent-blue-500 cursor-pointer disabled:opacity-40"
              />
              <p className="text-[10px] text-slate-400 leading-tight">
                同じ物体の連続発話を防止する待機時間
              </p>
            </div>

            <button
              onClick={handleTestSpeech}
              className="w-full py-2 px-3 text-xs font-semibold text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-700 transition-colors flex items-center justify-center gap-1.5"
            >
              <Volume2 className="w-3.5 h-3.5 text-blue-400" />
              <span>テスト音声を出力</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
