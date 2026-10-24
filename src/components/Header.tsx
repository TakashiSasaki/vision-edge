import React from 'react';
import { Camera, CameraOff, Volume2, VolumeX, Sparkles, Layers, Cpu } from 'lucide-react';
import type { ModelState, RunnableModelType } from '../types/vision';
import { MODEL_REGISTRY, isRunnableModel, backendDescription } from '../services/modelRegistry';

interface HeaderProps {
  activeModel: RunnableModelType;
  onSelectModel: (model: RunnableModelType) => void;
  modelState: ModelState;
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
  modelState,
  isCameraActive,
  onToggleCamera,
  ttsEnabled,
  onToggleTTS,
  isSpeaking,
  onOpenComparison
}) => {
  return (
    <header className="h-12 border-b border-slate-800 bg-[#0c111e] px-3 sm:px-5 flex items-center justify-between gap-2 shrink-0 select-none z-30">
      {/* Zone 1: Brand Wordmark */}
      <div className="flex items-center gap-2 shrink-0">
        <div className="w-6 h-6 rounded-md bg-blue-600 flex items-center justify-center text-white shadow-sm shadow-blue-500/20">
          <Layers className="w-3.5 h-3.5" />
        </div>
        <span className="text-sm font-bold tracking-tight text-white whitespace-nowrap">
          VisionStudio AI
        </span>
      </div>

      {/* Zone 2: Compact Selectors (Model & Backend) */}
      <div className="flex items-center gap-1.5 sm:gap-2">
        {/* Model Selector Pulldown */}
        <div className="flex items-center gap-1 bg-slate-900/90 border border-slate-700/80 rounded-md px-2 py-1 text-xs">
          <span className="text-[10px] text-slate-400 font-medium whitespace-nowrap hidden sm:inline">
            モデル:
          </span>
          <select
            value={activeModel}
            aria-label="モデル"
            onChange={(e) => { if (isRunnableModel(e.target.value)) onSelectModel(e.target.value); }}
            className="bg-transparent text-xs font-semibold text-white focus:outline-none cursor-pointer"
          >
            {MODEL_REGISTRY.map(model => (
              <option key={model.id} value={model.id} disabled={!model.implemented} className="bg-slate-900 text-white">
                {model.name}{model.implemented ? '' : '（未実装・選択不可）'}
              </option>
            ))}
          </select>
        </div>

        {/* Acceleration Backend Selector Pulldown */}
        <div className="hidden md:flex items-center gap-1 bg-slate-900/90 border border-slate-700/80 rounded-md px-2 py-1 text-xs">
          <Cpu className="w-3 h-3 text-blue-400 shrink-0" />
          <span aria-label="バックエンド状態">{backendDescription(modelState)}</span>
        </div>
      </div>

      {/* Zone 3: Actions */}
      <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
        {/* Audio Mute/Unmute */}
        <button
          onClick={onToggleTTS}
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
          className="hidden sm:flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-slate-300 bg-slate-800/80 border border-slate-700 rounded-md hover:bg-slate-700 hover:text-white transition-colors whitespace-nowrap"
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
