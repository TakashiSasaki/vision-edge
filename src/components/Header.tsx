import React from 'react';
import { Camera, CameraOff, Volume2, VolumeX, Sparkles, Layers, Cpu } from 'lucide-react';
import type { ClassificationState, ModelFamilyType } from '../types/classification';
import type { ModelState } from '../types/vision';
import { ALL_MODEL_REGISTRY, isRunnableModel, modelFamilyMetadata, backendDescription } from '../services/modelRegistry';

interface HeaderProps {
  activeModel: ModelFamilyType;
  onSelectModel: (model: ModelFamilyType) => void;
  activeVariant: string;
  onSelectVariant: (variantId: string) => void;
  modelState: ModelState | ClassificationState;
  isCameraActive: boolean;
  onToggleCamera: () => void;
  ttsEnabled: boolean;
  onToggleTTS: () => void;
  isSpeaking: boolean;
  onOpenComparison: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeModel,
  onSelectModel,
  activeVariant,
  onSelectVariant,
  modelState,
  isCameraActive,
  onToggleCamera,
  ttsEnabled,
  onToggleTTS,
  isSpeaking,
  onOpenComparison
}) => {
  const currentMeta = modelFamilyMetadata(activeModel);
  const backend = modelState.model === 'image-classifier'
    ? modelState.status === 'ready' ? `要求${modelState.requestedDelegate}／SDK設定${modelState.sdkDelegate}／実バックエンド未計測` : '分類バックエンド未準備'
    : backendDescription(modelState);

  return (
    <header className="min-h-12 py-2 flex-wrap border-b border-slate-800 bg-[#0c111e] px-3 sm:px-5 flex items-center justify-between gap-2 shrink-0 select-none z-30">
      {/* Zone 1: Brand Wordmark */}
      <div className="flex items-center gap-2 shrink-0 order-1">
        <div className="w-6 h-6 rounded-md bg-blue-600 flex items-center justify-center text-white shadow-sm shadow-blue-500/20">
          <Layers className="w-3.5 h-3.5" />
        </div>
        <span className="text-sm font-bold tracking-tight text-white whitespace-nowrap">
          VisionStudio AI
        </span>
      </div>

      {/* Zone 2: Compact Selectors (Model, Variant & Backend) */}
      <div className="flex items-center flex-wrap gap-1.5 sm:gap-2 order-3 w-full sm:order-2 sm:w-auto min-w-0">
        {/* Model Selector Pulldown */}
        <div className="flex items-center gap-1 bg-slate-900/90 border border-slate-700/80 rounded-md px-2 py-1 text-xs">
          <span className="text-[10px] text-slate-400 font-medium whitespace-nowrap hidden sm:inline">
            モデル:
          </span>
          <select
            value={activeModel}
            aria-label="モデル"
            onChange={(e) => { if (e.target.value === 'image-classifier' || isRunnableModel(e.target.value)) onSelectModel(e.target.value); }}
            className="bg-transparent text-xs font-semibold text-white focus:outline-none cursor-pointer"
          >
            {ALL_MODEL_REGISTRY.map(model => (
              <option key={model.id} value={model.id} disabled={!model.implemented} className="bg-slate-900 text-white">
                {model.name}{model.implemented ? '' : '（未実装・選択不可）'}
              </option>
            ))}
          </select>
        </div>

        {/* Variant Selector Pulldown */}
        <div className="flex items-center gap-1 bg-slate-900/90 border border-slate-700/80 rounded-md px-2 py-1 text-xs">
          <span className="text-[10px] text-slate-400 font-medium whitespace-nowrap hidden sm:inline">
            バリアント:
          </span>
          <select
            value={activeVariant}
            aria-label="バリアント"
            onChange={(e) => onSelectVariant(e.target.value)}
            className="bg-transparent text-xs font-semibold text-blue-300 focus:outline-none cursor-pointer"
          >
            {currentMeta.variants.map(variant => (
              <option key={variant.id} value={variant.id} className="bg-slate-900 text-white">
                {variant.name}
              </option>
            ))}
          </select>
        </div>

        {/* Acceleration Backend Selector Pulldown */}
        <div className="hidden lg:flex items-center gap-1 bg-slate-900/90 border border-slate-700/80 rounded-md px-2 py-1 text-xs">
          <Cpu className="w-3 h-3 text-blue-400 shrink-0" />
          <span aria-label="バックエンド状態">{backend}</span>
        </div>
      </div>

      {/* Zone 3: Actions */}
      <div className="flex items-center gap-1.5 sm:gap-2 shrink-0 order-2 sm:order-3">
        <a href="/model-cache.html" className="text-xs text-blue-300 underline px-1 py-2" aria-label="コンテナのモデルキャッシュ一覧">キャッシュ</a>
        <a href="/sitemap.html" className="text-xs text-blue-300 underline px-1 py-2" aria-label="サイトマップ">サイトマップ</a>
        {/* Audio Mute/Unmute */}
        <button
          onClick={onToggleTTS}
          disabled={activeModel === 'image-classifier'}
          aria-label={activeModel === 'image-classifier' ? '分類音声は候補ボタンから手動で開始' : undefined}
          title={ttsEnabled ? '音声読み上げ ON (クリックでミュート)' : '音声読み上げ OFF (クリックで有効化)'}
          className={`p-1.5 rounded-md text-xs font-medium border transition-colors ${
            ttsEnabled
              ? 'bg-emerald-950/60 border-emerald-600/50 text-emerald-400 hover:bg-emerald-900/60'
              : 'bg-slate-800/80 border-slate-700 text-slate-400 hover:text-slate-200'
          }`}
        >
          {ttsEnabled ? (
            <Volume2 className={`w-3.5 h-3.5 ${isSpeaking ? 'animate-bounce text-emerald-300' : ''}`} />
          ) : (
            <VolumeX className="w-3.5 h-3.5" />
          )}
        </button>

        {/* Specs Modal Trigger */}
        <button
          onClick={onOpenComparison}
          className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-slate-300 bg-slate-800/80 border border-slate-700 rounded-md hover:bg-slate-700 hover:text-white transition-colors whitespace-nowrap"
        >
          <Sparkles className="w-3 h-3 text-blue-400" />
          <span>仕様</span>
        </button>

        {/* Camera Toggle */}
        <button
          onClick={onToggleCamera}
          className={`flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-md transition-all whitespace-nowrap ${
            isCameraActive
              ? 'bg-rose-600 text-white hover:bg-rose-700 shadow-rose-900/30'
              : 'bg-blue-600 text-white hover:bg-blue-500 shadow-blue-900/30'
          }`}
        >
          {isCameraActive ? (
            <>
              <CameraOff className="w-3.5 h-3.5" />
              <span>停止</span>
            </>
          ) : (
            <>
              <Camera className="w-3.5 h-3.5" />
              <span>カメラ開始</span>
            </>
          )}
        </button>
      </div>
    </header>
  );
};
