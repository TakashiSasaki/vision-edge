import type { ModelState } from '../types/vision';

export function ModelStatus({ state, onRetry }: { state: ModelState; onRetry: () => void }) {
  if (state.status === 'error') return (
    <div role="alert" className="px-4 py-2 text-xs text-rose-300 bg-slate-900">
      <span>{state.stage === 'wasm' ? 'モデル実行環境の読み込みに失敗しました。' : 'モデルの生成に失敗しました。'}</span>
      <button type="button" onClick={onRetry} className="ml-3 underline">モデルを再試行</button>
    </div>
  );
  return <p role="status" className="px-4 py-2 text-xs text-slate-300">
    {state.status === 'ready' ? 'モデル準備完了（カメラの準備・推論成功とは別です）'
      : state.status === 'loading' ? 'モデル読込中・推論待機' : 'モデル未準備'}
  </p>;
}
