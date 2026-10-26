import { flushSync } from 'react-dom';
import { useCallback, useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore } from 'react';
import type { CameraFacingMode, RunnableModelType, TTSConfig } from '../types/vision';
import type { ClassificationDelegate, ClassificationState, ImageCategory, ImageClassifierVariantId, ModelFamilyType, WholeImageClassificationFrame } from '../types/classification';
import { ImageClassificationEngine, ClassificationInitializationError } from '../services/imageClassificationEngine';
import { ClassificationAssetsBlockedError, classifierAsset } from '../services/imageClassifierAssets';
import { startImageClassificationScheduler } from '../services/imageClassificationScheduler';
import { JobLock } from '../services/frameScheduler';
import { ClassificationSpeech } from '../services/classificationSpeech';
import { japaneseClassificationLabel } from '../services/classificationLabels';
import { useCameraStream } from '../hooks/useCameraStream';
import { Header } from './Header';
import { ImageClassificationPanel } from './ImageClassificationPanel';
import { ModelComparisonTable } from './ModelComparisonTable';
export function classifierBlocked(state: ClassificationState): boolean {
  return state.status === 'error' && state.cause instanceof ClassificationInitializationError && state.cause.cause instanceof ClassificationAssetsBlockedError;
}
interface Props {
  engine?: ImageClassificationEngine;
  variant: ImageClassifierVariantId;
  onVariant: (variant: ImageClassifierVariantId) => void;
  onSelectDetection: (model: RunnableModelType, variant?: string) => void;
  cameraActive: boolean;
  facingMode: CameraFacingMode;
  onToggleFacing: () => void;
  onStart: () => void;
  onStop: () => void;
  ttsConfig: TTSConfig;
}
export function ImageClassificationWorkspace(props: Props) {
  const [engine] = useState(() => props.engine ?? new ImageClassificationEngine());
  const state = useSyncExternalStore(engine.subscribe, engine.getState);
  const [speech] = useState(() => new ClassificationSpeech());
  const [delegate, setDelegate] = useState<ClassificationDelegate>('GPU');
  const [frame, setFrame] = useState<WholeImageClassificationFrame | null>(null);
  const [threshold, setThreshold] = useState(0);
  const [holding, setHolding] = useState(false);
  const [speechError, setSpeechError] = useState(false);
  const [fault, setFault] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  const [comparison, setComparison] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null), lock = useRef(new JobLock());
  const paused = useRef(false), live = useRef(true), faultRef = useRef(false);
  const activeProps = useRef(props); useLayoutEffect(() => { activeProps.current = props; });
  const invalidate = useCallback(() => { speech.cancel(); engine.clearResults(); setFrame(null); setSpeechError(false); }, [engine, speech]);
  const cameraStop = useCallback(() => { live.current = false; invalidate(); activeProps.current.onStop(); }, [invalidate]);
  const { cameraReady, cameraRequestId, cameraError, clearError } = useCameraStream(videoRef, props.cameraActive, props.facingMode, cameraStop);
  const hold = useCallback((value: boolean) => { paused.current = value; setHolding(value); }, []);
  useLayoutEffect(() => {
    live.current = props.cameraActive && cameraReady; faultRef.current = false; setFault(null); invalidate();
    return () => { live.current = false; invalidate(); };
  }, [props.cameraActive, props.facingMode, cameraReady, cameraRequestId, props.variant, delegate, threshold, retry, state.requestId, invalidate]);
  useEffect(() => { void engine.initialize(props.variant, delegate); return () => engine.dispose(); }, [engine, props.variant, delegate]);
  useEffect(() => () => { speech.cancel(); engine.dispose(); }, [engine, speech]);
  useEffect(() => {
    if (state.status !== 'ready' || !props.cameraActive || !cameraReady || fault) return;
    const requestId = state.requestId, variant = props.variant;
    return startImageClassificationScheduler({ engine, video: videoRef.current!, lock: lock.current,
      options: { scoreThreshold: threshold, maxResults: 5 },
      current: () => live.current && !paused.current && !faultRef.current && activeProps.current.variant === variant && engine.getState().requestId === requestId,
      // The SDK owner changes synchronously. Commit the corresponding visible
      // candidate before another input event can observe an obsolete render.
      adopt: accepted => { if (engine.isCurrentFrame(accepted)) flushSync(() => setFrame(accepted)); }, input: () => {},
      error: () => { faultRef.current = true; invalidate(); setFault('分類推論に失敗しました。'); },
      captureError: () => { faultRef.current = true; invalidate(); setFault('分類映像フレームの取得に失敗しました。'); },
    });
  }, [engine, state, props.cameraActive, cameraReady, cameraRequestId, props.variant, threshold, fault, retry, invalidate]);
  const validFrame = frame && props.cameraActive && cameraReady && !fault && state.status === 'ready' && state.variantId === props.variant && engine.isCurrentFrame(frame) ? frame : null;
  const select = (model: ModelFamilyType, variant?: string) => {
    if (model === 'image-classifier') { const entry = classifierAsset(variant ?? props.variant); if (entry.variantId !== props.variant) { invalidate(); engine.dispose(); props.onVariant(entry.variantId); } }
    else { live.current = false; invalidate(); engine.dispose(); props.onSelectDetection(model, variant); }
  };
  const speak = (accepted: WholeImageClassificationFrame, category: ImageCategory) => {
    const valid = () => live.current && !faultRef.current && activeProps.current.variant === accepted.variantId && engine.isCurrentFrame(accepted);
    if (accepted !== validFrame || !valid() || !accepted.classifications.some(head => head.categories.includes(category))) return;
    const label = (japaneseClassificationLabel(category.categoryName) ?? category.categoryName) || `SDK index ${category.index}`;
    setSpeechError(!speech.speak(`画像全体の分類候補、${label}。スコア${String(category.score)}。`, valid, hold, props.ttsConfig));
  };
  const blocked = classifierBlocked(state);
  return <div className="h-screen w-screen overflow-hidden bg-[#070b14] text-slate-100 flex flex-col">
    <Header activeModel="image-classifier" onSelectModel={model => select(model)} activeVariant={props.variant} onSelectVariant={variant => select('image-classifier', variant)} modelState={state} isCameraActive={props.cameraActive} onToggleCamera={() => { if (props.cameraActive) cameraStop(); else { clearError(); props.onStart(); } }} ttsEnabled={false} onToggleTTS={() => {}} isSpeaking={holding} onOpenComparison={() => setComparison(true)} />
    <div className="px-4 py-2 text-xs space-y-1">
      {blocked ? <p role="status">BLOCKED：検証済みコンテナキャッシュを利用できません。静的配信・previewではキャッシュAPIが必要です。取得・整合性エラー時に未検証の外部モデルへ切り替えません。</p>
        : state.status === 'error' ? <p role="alert">分類モデルの初期化に失敗しました。<button type="button" className="ml-2 underline" onClick={() => { invalidate(); void engine.initialize(props.variant, delegate); }}>分類モデルを再試行</button></p>
        : <p role="status">{state.status === 'ready' ? '分類タスク準備完了（実モデル推論の受入確認とは別です）' : '分類モデル準備中'}</p>}
      <p>モデルのライセンス・固定SHA256・入力形状（Lite0:224、Lite2:260）は確認済みです。実モデル推論・動作環境の受け入れは未検証です。</p>
      <label>分類delegate要求 <select className="bg-slate-800 rounded px-2 py-1" aria-label="分類delegate要求" value={delegate} onChange={e => { const value = e.target.value; if (value === 'CPU' || value === 'GPU') { invalidate(); engine.dispose(); setDelegate(value); } }}><option value="GPU">GPU</option><option value="CPU">CPU</option></select></label>
      <p aria-label="分類経路記録">受理した初期化の経路記録: モデル {state.status === 'ready' ? state.assetSource : '未取得'} ／ WASM {state.status === 'ready' ? state.wasmSource : '未取得'}（実バイト未確認）</p>
      <p aria-label="分類バックエンド状態">要求: {delegate} ／ SDK設定: {state.status === 'ready' ? state.sdkDelegate : '未準備'} ／ 実バックエンド: 未計測</p>
    </div>
    <main className="flex-1 min-h-0 overflow-y-auto p-3 space-y-3 md:space-y-0 md:grid md:grid-cols-2 gap-3">
      <section aria-label="画像分類プレビュー" className="min-w-0 space-y-2">
        <video ref={videoRef} autoPlay muted playsInline aria-label="画像分類カメラ" className="w-full rounded-xl bg-black aspect-video object-contain" />
        <p className="text-xs">画像全体の分類です。位置タップ・検出矩形・領域マスクは使用しません。</p>
        <button type="button" onClick={() => { invalidate(); props.onToggleFacing(); }} className="rounded border border-slate-600 px-3 py-2">カメラを切り替え</button>
        {cameraError && <p role="alert">{cameraError}</p>}
        {!props.cameraActive && <button type="button" onClick={() => { clearError(); props.onStart(); }}>分類カメラ開始</button>}
        {fault && <p role="alert">{fault}<button type="button" className="ml-2 underline" onClick={() => { faultRef.current = false; setFault(null); setRetry(n => n + 1); }}>分類推論を再試行</button></p>}
        {speechError && <p role="status">音声を開始できませんでした。この環境の音声合成設定を確認してください。</p>}
      </section>
      <ImageClassificationPanel frame={validFrame} threshold={threshold} onThreshold={value => { invalidate(); setThreshold(value); }} onSpeak={speak} holding={holding} />
    </main>
    <ModelComparisonTable isOpen={comparison} onClose={() => setComparison(false)} activeModel="image-classifier" activeVariant={props.variant} onSelectModelAndVariant={select} />
  </div>;
}
