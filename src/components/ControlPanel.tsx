import React, { useState, useEffect } from 'react';
import {
  Sliders,
  Volume2,
  Layers,
  RotateCcw,
  SlidersHorizontal,
  Eye
} from 'lucide-react';
import {
  DetectedObject,
  RunnableModelType,
  RenderOptions,
  TTSConfig
} from '../types/vision';
import type { ExtraInferenceOptions } from '../services/visionEngine';
import { speechService } from '../services/speechService';
import { confidenceDescription } from '../services/confidenceDescription';
import { modelMetadata } from '../services/modelRegistry';
import {
  isMirageExperimentalVariant,
  type MirageExperimentalExecutionDiagnostics,
} from '../services/mirageExperimentalOnnxService';

interface ControlPanelProps {
  activeModel: RunnableModelType;
  activeVariant?: string;
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
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  activeModel,
  activeVariant,
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
}) => {
  const [activeTab, setActiveTab] = useState<'detection' | 'params' | 'settings'>('detection');
  const [, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([]);
  const isExperimentalMirage =
    activeModel === 'mirage' && isMirageExperimentalVariant(activeVariant);

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
  const mirageIncludeOcr = inferenceParams.includeOcr ?? true;
  const experimentalNmsIou =
    inferenceParams.experimentalNmsIou !== undefined
      ? inferenceParams.experimentalNmsIou
      : null;
  const showObjectProposals = inferenceParams.showObjectProposals ?? true;
  const showTextProposals = inferenceParams.showTextProposals ?? true;
  const showDiagnosticMasks = inferenceParams.showDiagnosticMasks ?? true;
  const overlayNativeWasm = inferenceParams.overlayNativeWasm ?? false;

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
          <Layers className="w-3.5 h-3.5" />
          <span className="whitespace-nowrap">認識物体 {hasCurrentFrame ? `(${detectedObjects.length})` : '（未取得）'}</span>
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
              <span>【実験用機能】部分Mirage ONNX 領域候補推論</span>
              <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-amber-900/60 border border-amber-600/40">
                452/583 tensors · opset 17
              </span>
            </div>
            <p className="text-[11px] text-amber-200/90 leading-relaxed">
              公式.cactから復元した部分モデル（SHA-256: <code className="font-mono">52d52d29...</code>）です。物体名・OCR文字列・言語/領域処理・最終候補選別は未実装のため、<strong>物体領域候補・テキスト領域候補・枠・128×128診断用マスク</strong>として表示します（検証4画像で公式54領域中52件対応・余分候補80件、中間層167件の不一致あり）。
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

        {/* Tab 1: Detection List */}
        {activeTab === 'detection' && (
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
                          物体・輪郭・OCR優先（WebGPU/高速）
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
                        実験用部分ONNX 直近推論診断
                      </div>
                      <div>要求EP: {experimentalDiagnostics.requestedProvider} ／ 設定EP: {experimentalDiagnostics.configuredProviders.join(' → ')}</div>
                      {experimentalDiagnostics.fallbackReason && (
                        <div className="text-amber-300">フォールバック理由: {experimentalDiagnostics.fallbackReason}</div>
                      )}
                      <div>入力 valid_hw: [{experimentalDiagnostics.validHw[0]}, {experimentalDiagnostics.validHw[1]}] (元画像 {experimentalDiagnostics.metadata.W}×{experimentalDiagnostics.metadata.H} · ceil32)</div>
                      <div>座標逆変換倍率: [{experimentalDiagnostics.metadata.coordinate_scale.map(v => v.toFixed(3)).join(', ')}]</div>
                      <div>抽出候補: 物体 {experimentalDiagnostics.objectProposalCount}件 ／ テキスト {experimentalDiagnostics.textProposalCount}件{experimentalDiagnostics.nativeOverlayEnabled ? ` ／ 公式WASM ${experimentalDiagnostics.nativeOverlayCount}件` : ''}</div>
                      <div>処理時間: 前処理 {experimentalDiagnostics.timingsMs.preprocess.toFixed(1)}ms · 推論 {experimentalDiagnostics.timingsMs.inference.toFixed(1)}ms · 後処理 {experimentalDiagnostics.timingsMs.postprocess.toFixed(1)}ms</div>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Section 2: Render & Overlay Display Options */}
            <div className="space-y-3 pt-1">
              <div className="text-[11px] font-bold text-blue-400 flex items-center gap-1.5 border-b border-slate-800/80 pb-1">
                <Eye className="w-3.5 h-3.5" />
                <span>表示・オーバーレイ設定</span>
              </div>

              {/* Mask Opacity (For models that generate masks) */}
              {activeModel !== 'mediapipe' && (
                <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-medium text-slate-200">輪郭マスク不透明度</span>
                    <span className="font-mono text-blue-400 tabular-nums font-semibold">
                      {Math.round(renderOptions.maskOpacity * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
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
                    <span className="text-slate-300">輪郭・セグメンテーションマスク描画</span>
                    <input
                      type="checkbox"
                      checked={renderOptions.showMasks}
                      onChange={(e) => onChangeRenderOptions({ showMasks: e.target.checked })}
                      className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 bg-slate-800 border-slate-700 cursor-pointer"
                    />
                  </label>
                )}

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
