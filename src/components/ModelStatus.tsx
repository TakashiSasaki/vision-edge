import { ModelAssetsBlockedError } from '../services/modelAssetResolver';
import type { ModelState } from '../types/vision';
import { MIRAGE_WEBGPU_UNAVAILABLE_REASON_JA } from '../services/mirageAvailability';

export function ModelStatus({
  state,
  onRetry,
  onOpenDiagnostics,
}: {
  state: ModelState;
  onRetry: () => void;
  onOpenDiagnostics?: () => void;
}) {
  if (state.status === 'error' && state.model === 'mirage' && state.variantId === 'mirage-webgpu') return (
    <div role="alert" className="px-4 py-2 text-xs text-rose-300 bg-slate-900 flex items-center justify-between border-b border-rose-900/40">
      <span>{MIRAGE_WEBGPU_UNAVAILABLE_REASON_JA}</span>
      {onOpenDiagnostics && (
        <button
          type="button"
          onClick={onOpenDiagnostics}
          className="px-2.5 py-1 text-xs font-medium rounded bg-slate-800 text-cyan-300 hover:bg-slate-700 transition-colors shrink-0 ml-3"
        >
          詳細ログ・自己診断
        </button>
      )}
    </div>
  );

  if (state.status === 'error') {
    const rawCause =
      state.cause instanceof Error
        ? state.cause.message
        : state.cause
          ? String(state.cause)
          : '';
    const causeSnippet = rawCause ? ` (${rawCause})` : '';
    const errorText =
      state.cause instanceof ModelAssetsBlockedError
        ? `BLOCKED：${state.cause.reason}。取得・配信・実推論は無効です。`
        : state.stage === 'wasm'
          ? `モデル実行環境の読み込みに失敗しました。${causeSnippet}`
          : `モデルの生成に失敗しました。${causeSnippet}`;

    return (
      <div
        role="alert"
        className="px-4 py-2 text-xs text-rose-300 bg-slate-900 flex flex-wrap items-center justify-between gap-2 border-b border-rose-900/40"
      >
        <span className="font-medium break-all">{errorText}</span>
        <div className="flex items-center gap-2 shrink-0">
          {onOpenDiagnostics && (
            <button
              type="button"
              onClick={onOpenDiagnostics}
              className="px-2.5 py-1 text-xs font-medium rounded bg-slate-800 text-cyan-300 hover:bg-slate-700 transition-colors"
            >
              詳細ログ・自己診断
            </button>
          )}
          <button
            type="button"
            onClick={onRetry}
            className="px-2.5 py-1 text-xs font-medium rounded bg-rose-950/80 text-rose-200 border border-rose-800/60 hover:bg-rose-900 transition-colors"
          >
            モデルを再試行
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="px-4 py-2 text-xs text-slate-300 bg-slate-900/60 flex items-center justify-between border-b border-slate-800/40">
      <p role="status">
        {state.status === 'ready'
          ? 'モデル準備完了（カメラの準備・推論成功とは別です）'
          : state.status === 'loading'
            ? 'モデル読込中・推論待機'
            : 'モデル未準備'}
      </p>
      {onOpenDiagnostics && (
        <button
          type="button"
          onClick={onOpenDiagnostics}
          className="text-xs text-slate-400 hover:text-cyan-400 transition-colors underline"
        >
          診断ログ
        </button>
      )}
    </div>
  );
}
