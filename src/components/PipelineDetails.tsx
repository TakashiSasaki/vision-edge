import { memo } from 'react';
import { PIPELINE_STAGES, type AttemptOutcome, type PipelineSnapshot, type StageStatistics } from '../services/pipelineMetrics';
const meanings = {
  capture: 'captureInputの全試行 (bootstrap/到着/実行時)。RGBA読出・所有コピーを含む。',
  comparison: 'samePixelsの全呼出。推論しない重複入力の比較も含む。',
  queue: 'タイマー予約成功から実行開始までの実測待機。ロック取得不可はcancelled。',
  job: 'ロック取得後から入力release終了までの占有。capture/engine/結果採用/入力cleanupを包含。非推論の実行時重複はcancelled。',
  sdk: 'detectForVideo / segmentForVideo の同期呼出区間。',
  postprocess: 'SDK後の変換・マスク走査/RGBA作成/putImageData/BBox・フィルター/平滑化。結果closeを除く区間合計。',
  cleanup: '入力releaseまたは推論結果closeの各呼出。初期化/モデルdisposeは対象外。',
  engine: '準備済みdetectVideoFrameラッパー全体。SDK/変換/結果closeを含む。入力releaseは範囲外。',
  draw: 'Canvas寸法/バッファ設定・オーバーレイ描画・表示Canvasコピー/クリア。onFrame通知を含まない。',
  latency: '実行時の不変入力取得開始から最初のCanvasコピー完了。queueは範囲外。live-video入力は適用不可。',
};
const format = (value: number | null | undefined) => value == null ? '未計測' : value.toFixed(2);
function StatisticsTable({ snapshot, outcome }: { snapshot: PipelineSnapshot | null; outcome: AttemptOutcome }) {
  return <table className="border-separate border-spacing-1" aria-label={`${outcome} interval statistics`}>
    <caption>{outcome} 区間 (ms)</caption>
    <thead><tr>{['stage / 意味', 'count / 有効 / 欠測', 'mean ms', 'median ms', 'p95 ms', 'max ms', '合計 ms', '分位標本 / 有効標本 / coverage'].map(label => <th scope="col" key={label}>{label}</th>)}</tr></thead>
    <tbody>{PIPELINE_STAGES.map(stage => {
      const row: StageStatistics | undefined = snapshot?.stages[stage][outcome];
      return <tr key={stage}>
        <th scope="row" title={meanings[stage]}>{stage}</th>
        <td>{row ? `${row.count} / ${row.measured} / ${row.missing}` : '未計測'}</td>
        <td>{format(row?.mean)}</td><td>{format(row?.median)}</td><td>{format(row?.p95)}</td><td>{format(row?.max)}</td><td>{format(row?.totalMs)}</td>
        <td>{row ? `${row.quantileCount} / ${row.measured} / ${row.truncated ? '切捨あり・部分集計' : '保持窓全体'}` : '未計測'}</td>
      </tr>;
    })}</tbody>
  </table>;
}
// Native details state stays in the DOM and never participates in pipeline effects.
export const PipelineDetails = memo(function PipelineDetails({ snapshot }: { snapshot: PipelineSnapshot | null }) {
  return <details aria-label="パイプライン計測詳細" className="absolute top-12 right-2 z-30 max-h-[65%] max-w-[95%] overflow-auto rounded bg-slate-950/95 p-2 text-[10px] text-slate-200">
    <summary className="cursor-pointer">計測詳細 (JS wall time)</summary>
    <p>直近5秒・完了時刻基準。CPU/GPU使用率、カメラFPS、物理画面表示は未計測。</p>
    <p>count/mean/max/合計: 保持イベント (最大2048件/区間/結果)。中央値/p95: 最新最大300有効標本。中央値は中央2値の平均、p95はnearest-rank。欠測 ≠ 0 ms。</p>
    <StatisticsTable snapshot={snapshot} outcome="success" />
    <details><summary>失敗・取消・旧世代の区間統計</summary>{(['failed', 'cancelled', 'stale'] as const).map(outcome => <StatisticsTable key={outcome} snapshot={snapshot} outcome={outcome} />)}</details>
    <p>captureDuty: {snapshot?.captureDutyPercent == null ? '未計測' : `${snapshot.captureDutyPercent.toFixed(2)} % (JS wall)`}。captureInputのみの完了区間合計 ÷ 5000 ms (窓境界で区間を分割しない)。外部比較はcomparisonに表示。</p>
    <p>jobは実行時capture/engine/結果採用/入力cleanupを包含。engineはSDK/postprocess/結果cleanupを包含。captureは入力コピーを包含。comparison/queue/入力cleanup/drawは別区間。包含区間を合算しない。</p>
    <p>latencyは実際にSDKへ渡した不変入力の取得開始 → 最初のCanvasコピー完了。queueは範囲外。live-videoは適用不可。物理表示/カメラから画面の遅延は不明。</p>
    <p>結果 outcomes (件 / 5秒): {snapshot ? (Object.entries(snapshot.results).map(([key, count]) => `${key}: ${count}`).join(', ') || '観測なし') : '未計測'}{snapshot?.resultsTruncated ? ' (切捨あり・部分集計)' : ''}。未描画保持: {snapshot?.pendingResults ?? '未計測'} (最大64件/5秒)。</p>
    <p>未描画: overwritten=更新前に上書き、cancelled=停止/無効化、hidden=非表示、draw-failed=描画例外、expired=保持期限、capacity=上限。unbound=描画成功・不変入力時刻なし。繰返しRAFはlatency標本を追加しない。</p>
    <p>世代変更で統計を破棄。時刻欠測 (世代累計): {snapshot?.unclocked ?? '未計測'}。無効/逆行時刻は欠測となり、時刻が追いつくまで統計を表示しない。</p>
    <details><summary>区間の意味</summary><dl>{PIPELINE_STAGES.map(stage => <div key={stage}><dt>{stage} (ms)</dt><dd>{meanings[stage]}</dd></div>)}</dl></details>
  </details>;
});
