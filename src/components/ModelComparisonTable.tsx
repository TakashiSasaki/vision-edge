import type { RunnableModelType } from '../types/vision';
import { MODEL_REGISTRY, isRunnableModel } from '../services/modelRegistry';
import { getResolvedSource } from '../services/modelAssetResolver';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  activeModel: RunnableModelType;
  onSelectModel: (model: RunnableModelType) => void;
}

function sourceDescription(modelId: RunnableModelType): string {
  const source = getResolvedSource(modelId);
  if (source === 'container-cache') return 'コンテナ内キャッシュ配信（未取得時は外部URLへ自動フォールバック）';
  if (source === 'external-fallback') return '外部配布元URLへの自動フォールバック配信';
  return '外部配布元URL直接取得（コンテナキャッシュ有効時はローカル配信）';
}

export function ModelComparisonTable({ isOpen, onClose, activeModel, onSelectModel }: Props) {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 bg-black/80">
      <section role="dialog" aria-modal="true" aria-labelledby="model-specs-title" className="bg-slate-900 border border-slate-700 rounded-xl p-4 max-w-2xl max-h-[85vh] overflow-y-auto space-y-3">
        <h2 id="model-specs-title" className="text-white font-semibold">物体認識・セグメンテーションモデル仕様</h2>
        {MODEL_REGISTRY.map(model => (
          <div key={model.id} className="border border-slate-700 rounded-lg p-3 text-xs space-y-1">
            <button type="button" disabled={!model.implemented} aria-pressed={activeModel === model.id}
              onClick={() => { if (isRunnableModel(model.id)) { onSelectModel(model.id); onClose(); } }}
              className="font-semibold text-white disabled:text-slate-500 disabled:cursor-not-allowed">
              {model.name}{model.implemented ? '' : '（未実装・選択不可）'}
            </button>
            {activeModel === model.id && <span className="ml-2 text-blue-400">選択中</span>}
            <p>{model.runtime} · {model.outputType}</p>
            <p>{model.description}</p>
            {model.license && (
              <p className="text-slate-400">
                ライセンス: {model.licenseNote ?? model.license} ／ 供給経路: {sourceDescription(model.id)}
              </p>
            )}
          </div>
        ))}
        <p className="text-xs text-slate-400">MediaPipeはGPU delegateを指定します。ONNXはWebGPU優先で、利用不能時はWASMを設定します。設定とフォールバック理由はヘッダーに表示し、実際の実行バックエンド・速度は未計測です。</p>
        <button type="button" onClick={onClose} className="px-3 py-1 bg-slate-800 rounded">閉じる</button>
      </section>
    </div>
  );
}
