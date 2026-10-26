import type { ModelFamilyType } from '../types/classification';
import type { RunnableModelType } from '../types/vision';
import { MODEL_REGISTRY, IMAGE_CLASSIFIER_MODEL, isRunnableModel } from '../services/modelRegistry';
import { getResolvedSource } from '../services/modelAssetResolver';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  activeModel: ModelFamilyType;
  activeVariant: string;
  onSelectModelAndVariant: (model: ModelFamilyType, variantId: string) => void;
}

function sourceDescription(modelId: RunnableModelType): string {
  const source = getResolvedSource(modelId);
  if (source === 'container-cache') return 'コンテナ内キャッシュ配信（未取得時は外部URLへ自動フォールバック）';
  if (source === 'external-fallback') return '外部配布元URLへの自動フォールバック配信';
  return '外部配布元URL直接取得（コンテナキャッシュ有効時はローカル配信）';
}

export function ModelComparisonTable({
  isOpen,
  onClose,
  activeModel,
  activeVariant,
  onSelectModelAndVariant
}: Props) {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 bg-black/80">
      <section role="dialog" aria-modal="true" aria-labelledby="model-specs-title" className="bg-slate-900 border border-slate-700 rounded-xl p-4 max-w-2xl max-h-[85vh] overflow-y-auto space-y-3">
        <h2 id="model-specs-title" className="text-white font-semibold">物体認識・セグメンテーションモデル仕様・バリアント／画像全体分類</h2>
        {MODEL_REGISTRY.map(model => (
          <div key={model.id} className="border border-slate-700 rounded-lg p-3 text-xs space-y-2">
            <div className="flex items-center justify-between">
              <div className="font-semibold text-white">
                {model.name}{model.implemented ? '' : '（未実装・選択不可）'}
              </div>
              {activeModel === model.id && <span className="text-blue-400 font-medium">現在使用中</span>}
            </div>
            <p className="text-slate-300">{model.runtime} · {model.outputType}</p>
            <p className="text-slate-400">{model.description}</p>

            {/* Variant List */}
            <div className="space-y-1.5 pt-1">
              <div className="text-[11px] font-semibold text-slate-300">利用可能なモデルサイズ・構造バリアント:</div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                {model.variants.map(variant => {
                  const isSelected = activeModel === model.id && activeVariant === variant.id;
                  return (
                    <button
                      key={variant.id}
                      type="button"
                      disabled={!model.implemented}
                      onClick={() => {
                        if (isRunnableModel(model.id)) {
                          onSelectModelAndVariant(model.id, variant.id);
                          onClose();
                        }
                      }}
                      className={`p-2 rounded border text-left transition-all ${
                        isSelected
                          ? 'bg-blue-950/70 border-blue-500 text-white font-medium shadow-sm'
                          : 'bg-slate-800/60 border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-xs">{variant.name}</span>
                        {isSelected && <span className="text-[10px] text-blue-400 font-bold">選択中</span>}
                      </div>
                      <p className="text-[10px] text-slate-400 mt-0.5 leading-tight">{variant.description}</p>
                    </button>
                  );
                })}
              </div>
            </div>

            {model.license && (
              <p className="text-slate-400 text-[10px] pt-1">
                {model.id === 'mediapipe' ? '既定Lite0のライセンス' : 'ライセンス'}: {model.licenseNote ?? model.license} ／ 供給経路: {sourceDescription(model.id)}
              </p>
            )}
          </div>
        ))}
        <section className="border border-slate-700 rounded-lg p-3 text-xs space-y-2" aria-label="画像全体分類仕様">
          <h3>{IMAGE_CLASSIFIER_MODEL.name}：画像全体の分類</h3>
          <p>Top5は画像全体の候補です。矩形・マスク・位置や物体数は生成しません。スコアは正答率ではありません。</p>
          <p>分類モデルはApache-2.0・固定SHA256/サイズを確認済み。Lite0入力224、Lite2入力260。対応コンテナで検証後に配信します。静的配信・previewはAPI不在でBLOCKED。実推論は未検証です。 <a href="/licenses/classifier-models.txt" className="underline text-blue-300">出典・ライセンス</a></p>
          <p>GPU/CPU要求は明示し、実バックエンドは未計測です。翻訳は完全一致ラベルのみ。手動音声のみ使用します。</p>
          {IMAGE_CLASSIFIER_MODEL.variants.map(variant => <button type="button" key={variant.id} className="rounded border border-slate-600 p-2 mr-2" onClick={() => { onSelectModelAndVariant('image-classifier', variant.id); onClose(); }}>{variant.name}（画像全体分類）</button>)}
        </section>
        <p className="text-xs text-slate-400">SSD MobileNetV2はSDK0.10.14での実行互換性が未検証のため取得・配信を行いません。Apache-2.0のプロジェクト声明と残る検証事項は <a href="/licenses/ssd-mobilenet-v2.txt" className="underline">SSD出典</a> を参照してください。</p>
        <p className="text-xs text-slate-400">MediaPipeはGPU delegateを指定します。ONNXはWebGPU優先で、利用不能時はWASMを設定します。対応する開発サーバーでは既存4モデル9種と分類2種を起動時にプリフェッチします。分類2種は固定ハッシュ・サイズ一致が必須です。</p>
        <button type="button" onClick={onClose} className="px-3 py-1 bg-slate-800 hover:bg-slate-700 rounded text-xs text-slate-200">閉じる</button>
      </section>
    </div>
  );
}
