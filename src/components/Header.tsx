import React from 'react';
import { Camera, CameraOff, Volume2, VolumeX, Sparkles, Layers, Cpu } from 'lucide-react';
import { BackendOption, ModelType } from '../types/vision';

interface HeaderProps {
  activeModel: ModelType;
  onSelectModel: (model: ModelType) => void;
  selectedBackend: BackendOption;
  onSelectBackend: (backend: BackendOption) => void;
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
  selectedBackend,
  onSelectBackend,
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
            onChange={(e) => onSelectModel(e.target.value as ModelType)}
            className="bg-transparent text-xs font-semibold text-white focus:outline-none cursor-pointer"
          >
            <option value="selfie-segmenter" className="bg-slate-900 text-emerald-400 font-bold">✨ Selfie Segmenter (人物輪郭・推奨)</option>
            <option value="yolo-seg" className="bg-slate-900 text-white">YOLO-seg (輪郭＋矩形)</option>
            <option value="mediapipe" className="bg-slate-900 text-white">MediaPipe (高速矩形)</option>
            <option value="segformer" className="bg-slate-900 text-white">SegFormer (全領域)</option>
          </select>
        </div>

        {/* Acceleration Backend Selector Pulldown */}
        <div className="hidden md:flex items-center gap-1 bg-slate-900/90 border border-slate-700/80 rounded-md px-2 py-1 text-xs">
          <Cpu className="w-3 h-3 text-blue-400 shrink-0" />
          <select
            value={selectedBackend}
            onChange={(e) => onSelectBackend(e.target.value as BackendOption)}
            className="bg-transparent text-xs font-medium text-slate-200 focus:outline-none cursor-pointer"
          >
            <option value="auto" className="bg-slate-900 text-white">自動 (Auto)</option>
            <option value="webgpu" className="bg-slate-900 text-white">WebGPU</option>
            <option value="wasm-simd" className="bg-slate-900 text-white">WASM SIMD</option>
            <option value="webgl" className="bg-slate-900 text-white">WebGL</option>
          </select>
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
