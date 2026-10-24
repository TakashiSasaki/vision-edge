import React from 'react';
import { X, Sparkles } from 'lucide-react';
import { ModelType } from '../types/vision';

interface ModelComparisonTableProps {
  isOpen: boolean;
  onClose: () => void;
  activeModel: ModelType;
  onSelectModel: (m: ModelType) => void;
}

export const ModelComparisonTable: React.FC<ModelComparisonTableProps> = ({
  isOpen,
  onClose,
  activeModel,
  onSelectModel
}) => {
  if (!isOpen) return null;

  const comparisonData = [
    {
      id: 'selfie-segmenter' as ModelType,
      model: '✨ Selfie Segmenter (人物・上半身リアルタイム輪郭)',
      engine: 'MediaPipe Tasks Vision (WebGL GPU)',
      output: 'ピクセル単位人物輪郭マスク＋矩形追従',
      classes: '人物 / セルフィー特化',
      labelAvailable: true,
      fpsRange: '45 ~ 60 FPS',
      strength: 'インカメラや近距離での頭部欠落・下ズレを根本解消。頭髪・顔・肩・上半身に完全密着し、ミリ秒単位でリアルタイム追随。'
    },
    {
      id: 'yolo-seg' as ModelType,
      model: 'YOLO-seg (YOLOv8/11-seg)',
      engine: 'ONNX Runtime Web (WebGPU / WASM)',
      output: 'インスタンス輪郭マスク＋矩形',
      classes: 'COCO 80クラス',
      labelAvailable: true,
      fpsRange: '25 ~ 45 FPS',
      strength: '輪郭マスクと矩形を同時出力。重なり合う物体の切り分けに最優秀。'
    },
    {
      id: 'mediapipe' as ModelType,
      model: 'MediaPipe Object Detector',
      engine: 'MediaPipe Tasks Vision (WASM / WebGL)',
      output: 'バウンディングボックス (矩形)',
      classes: 'COCO 80クラス',
      labelAvailable: true,
      fpsRange: '45 ~ 60+ FPS',
      strength: '超軽量で省電力。モバイルでも極めて安定したフレームレート。'
    },
    {
      id: 'segformer' as ModelType,
      model: 'SegFormer (ADE20K)',
      engine: 'Transformers.js v3 (WebGPU / WASM)',
      output: 'セマンティック全領域分割 (150領域)',
      classes: 'ADE20K 150クラス',
      labelAvailable: true,
      fpsRange: '15 ~ 30 FPS',
      strength: '壁・床・天井・家具など空間全体のピクセルを幅広く分類。'
    }
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 bg-black/80 backdrop-blur-sm">
      <div className="bg-[#0f172a] border border-slate-800 rounded-xl max-w-2xl w-full max-h-[85vh] overflow-hidden flex flex-col shadow-2xl">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-blue-400" />
            <h3 className="text-sm font-bold text-white">物体認識・セグメンテーションモデル仕様</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white rounded-lg transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-4 overflow-y-auto space-y-3 text-xs">
          {comparisonData.map((row) => {
            const isActive = row.id === activeModel;
            return (
              <div
                key={row.id}
                onClick={() => {
                  onSelectModel(row.id);
                  onClose();
                }}
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  isActive
                    ? 'bg-blue-950/40 border-blue-500 shadow-sm'
                    : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="font-semibold text-white flex items-center gap-1.5">
                    <span>{row.model}</span>
                    {isActive && (
                      <span className="text-[10px] bg-blue-600 text-white px-1.5 py-0.2 rounded font-normal">
                        稼働中
                      </span>
                    )}
                  </div>
                  <span className="font-mono text-emerald-400">{row.fpsRange}</span>
                </div>
                <div className="text-[11px] text-slate-400 mb-1">
                  {row.engine} · {row.classes}
                </div>
                <div className="text-slate-300 text-[11px] leading-relaxed">
                  {row.strength}
                </div>
              </div>
            );
          })}
        </div>

        <div className="p-3 border-t border-slate-800 bg-slate-900/40 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 text-xs font-semibold text-slate-300 bg-slate-800 hover:bg-slate-700 rounded-md transition-colors"
          >
            閉じる
          </button>
        </div>
      </div>
    </div>
  );
};
