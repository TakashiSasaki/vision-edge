import React, { useState, useEffect } from 'react';
import {
  Sliders,
  Volume2,
  Layers
} from 'lucide-react';
import { DetectedObject, ModelType, RenderOptions, TTSConfig } from '../types/vision';
import { speechService } from '../services/speechService';

interface ControlPanelProps {
  activeModel: ModelType;
  renderOptions: RenderOptions;
  onChangeRenderOptions: (opts: Partial<RenderOptions>) => void;
  ttsConfig: TTSConfig;
  onChangeTTSConfig: (cfg: Partial<TTSConfig>) => void;
  detectedObjects: DetectedObject[];
  hasCurrentFrame: boolean;
  onSelectObject: (obj: DetectedObject) => void;
  selectedObjectId?: string;
  isSpeaking: boolean;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  renderOptions,
  onChangeRenderOptions,
  ttsConfig,
  onChangeTTSConfig,
  detectedObjects,
  hasCurrentFrame,
  onSelectObject,
  selectedObjectId
}) => {
  const [activeTab, setActiveTab] = useState<'detection' | 'settings'>('detection');
  const [, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([]);

  useEffect(() => {
    setAvailableVoices(speechService.getVoices());
  }, []);

  const handleManualSpeak = (obj: DetectedObject) => {
    onSelectObject(obj);
    speechService.speak(
      `${obj.labelJa}。信頼度は${Math.round(obj.confidence * 100)}%です。`,
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

  return (
    <div className="w-full h-full flex flex-col bg-[#0c111e] border border-slate-800 rounded-xl overflow-hidden shadow-lg select-none">
      {/* Tab Switcher */}
      <div className="flex border-b border-slate-800 bg-[#090d18] shrink-0">
        <button
          onClick={() => setActiveTab('detection')}
          className={`flex-1 py-2.5 px-3 text-xs font-semibold flex items-center justify-center gap-1.5 border-b-2 transition-colors ${
            activeTab === 'detection'
              ? 'border-blue-500 text-blue-400 bg-slate-900/60'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>認識物体 {hasCurrentFrame ? `(${detectedObjects.length})` : '（未取得）'}</span>
        </button>

        <button
          onClick={() => setActiveTab('settings')}
          className={`flex-1 py-2.5 px-3 text-xs font-semibold flex items-center justify-center gap-1.5 border-b-2 transition-colors ${
            activeTab === 'settings'
              ? 'border-blue-500 text-blue-400 bg-slate-900/60'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Sliders className="w-3.5 h-3.5" />
          <span className="flex items-center gap-1">
            <span>音声・設定</span>
            {ttsConfig.enabled && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />}
          </span>
        </button>
      </div>

      {/* Tab Body (Internal scroll only) */}
      <div className="flex-1 overflow-y-auto p-3.5 space-y-3.5 text-xs">
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
                <p className="text-slate-400">{hasCurrentFrame ? '現在しきい値を超える物体はありません' : 'モデルとカメラの準備後、推論結果を表示します'}</p>
                <p className="text-[10px] text-slate-500 mt-1">{hasCurrentFrame ? 'カメラを物体に向けてください' : '未取得は正常な空検出を意味しません'}</p>
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
                              {Math.round(obj.confidence * 100)}%
                            </span>
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

        {/* Tab 2: Speech & Model Settings */}
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

            {/* Confidence Threshold */}
            <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5">
              <div className="flex justify-between items-center text-xs">
                <span className="font-medium text-slate-200">認識しきい値 (Confidence)</span>
                <span className="font-mono text-blue-400 tabular-nums font-semibold">
                  {Math.round(renderOptions.confidenceThreshold * 100)}%
                </span>
              </div>
              <input
                type="range"
                min="0.15"
                max="0.90"
                step="0.05"
                value={renderOptions.confidenceThreshold}
                onChange={(e) => onChangeRenderOptions({ confidenceThreshold: parseFloat(e.target.value) })}
                className="w-full accent-blue-500 cursor-pointer"
              />
            </div>

            {/* Mask Opacity */}
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
