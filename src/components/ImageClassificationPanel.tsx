import type { WholeImageClassificationFrame, ImageCategory } from '../types/classification';
import { japaneseClassificationLabel } from '../services/classificationLabels';
interface Props {
  frame: WholeImageClassificationFrame | null;
  threshold: number;
  onThreshold: (value: number) => void;
  onSpeak: (frame: WholeImageClassificationFrame, category: ImageCategory) => void;
  holding: boolean;
}
export function ImageClassificationPanel({ frame, threshold, onThreshold, onSpeak, holding }: Props) {
  return <section aria-label="画像全体の分類結果" className="min-h-0 overflow-y-auto rounded-xl border border-slate-700 bg-slate-900 p-4 space-y-4">
    <h2 className="font-semibold">画像全体の分類 Top5</h2>
    <p className="text-xs text-slate-300">順位は出力スコア順です。スコアは正答率を保証しません。SDK indexは順位とは別のモデル出力IDです。</p>
    <label className="block text-sm">分類スコアしきい値: {threshold.toFixed(2)}<input aria-label="分類スコアしきい値" className="block w-full" type="range" min="0" max="1" step="0.01" value={threshold} onChange={e => onThreshold(Number(e.target.value))} /></label>
    <fieldset disabled className="text-xs text-slate-400 space-y-1"><legend>分類では使用しない空間・自動音声操作</legend>
      {['検出矩形', '領域マスク', '位置タップ', '自動読み上げ'].map(label => <label key={label} className="inline-flex gap-1 mr-3"><input type="checkbox" checked={false} readOnly />{label}</label>)}
    </fieldset>
    <p className="text-xs">音声は候補の「読み上げ」ボタンだけで開始します。読み上げ中は対象フレームを保持します。</p>
    {holding && <p role="status">読み上げ対象のフレームを保持中。終了後に分類を更新します。</p>}
    {!frame ? <p>現在の分類結果は未取得です。</p> : frame.classifications.length === 0 ? <p>SDKの分類候補は空です。</p> : frame.classifications.map((head, h) => <div key={`${head.headIndex}:${h}`} className="space-y-2">
      <h3 className="text-xs text-slate-300">分類ヘッド {head.headIndex}{head.headName ? ` · ${head.headName}` : ''}</h3>
      {head.categories.length === 0 && <p>現在のしきい値を満たす候補はありません。</p>}
      <ol className="space-y-2">{head.categories.slice(0, 5).map((category, index) => <li key={`${index}:${category.index}`} className="rounded border border-slate-700 p-3 space-y-1 break-words">
        <p className="text-sm font-semibold">順位 {index + 1} <span className="text-xs text-slate-400">SDK index {category.index}</span></p>
        <p lang="en" data-classifier-label="raw">{category.categoryName || '（SDKラベル空欄）'}</p>
        <p lang="ja">{japaneseClassificationLabel(category.categoryName) ?? '日本語訳未登録（SDK原文を保持）'}</p>
        <p>スコア <output>{String(category.score)}</output></p>
        <button type="button" onClick={() => onSpeak(frame, category)} className="px-3 py-2 rounded border border-slate-600 text-sm" aria-label={`順位${index + 1} ${category.categoryName}を読み上げ`}>読み上げ</button>
      </li>)}</ol>
    </div>)}
  </section>;
}
