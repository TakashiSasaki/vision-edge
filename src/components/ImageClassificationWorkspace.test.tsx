import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { ImageClassificationWorkspace } from './ImageClassificationWorkspace';
import { ImageClassificationEngine, ClassificationInitializationError } from '../services/imageClassificationEngine';
import type { ImageClassificationTask, ImageClassificationFactory } from '../services/imageClassificationEngine';
import { ClassificationAssetsBlockedError } from '../services/imageClassifierAssets';
import type { ImageClassifierVariantId } from '../types/classification';
import { useState } from 'react';
vi.mock('../hooks/useCameraStream', () => ({ useCameraStream: (_ref: unknown, enabled: boolean) => ({ cameraReady: enabled, cameraRequestId: enabled ? 1 : 2, cameraError: null, clearError: vi.fn() }) }));
let callbacks: Map<number, FrameRequestCallback>, next: number, utterances: { text: string; onend?: () => void }[];
let synth: { speak: ReturnType<typeof vi.fn>; cancel: ReturnType<typeof vi.fn>; getVoices: ReturnType<typeof vi.fn> };
const tts = { enabled: false, autoReadout: true, cooldownSec: 3, rate: 1, pitch: 1, volume: 1, voiceURI: '' };
const categories = [ { index: 1000, categoryName: 'UNKNOWN Exact Raw Label', displayName: '', score: .7 }, { index: 7, categoryName: 'goldfish, Carassius auratus', displayName: '', score: .95 }, { index: 0, categoryName: 'banana', displayName: '', score: .8 }, { index: 3, categoryName: 'volcano', displayName: '', score: .6 }, { index: 5, categoryName: 'soccer ball', displayName: '', score: .5 }, { index: 2, categoryName: 'orange', displayName: '', score: .1 } ];
function task(label?: string) { return { sdkDelegate: 'GPU' as const, close: vi.fn(), classifyForVideo: vi.fn<ImageClassificationTask['classifyForVideo']>(() => ({ classifications: [{ headIndex: 0, headName: '', categories: label ? [{...categories[0], categoryName: label}] : categories }] })) }; }
const engines: ImageClassificationEngine[] = [];
beforeEach(() => {
  vi.useFakeTimers(); callbacks = new Map(); next = 0; utterances = [];
  vi.stubGlobal('requestAnimationFrame', vi.fn((cb: FrameRequestCallback) => { callbacks.set(++next, cb); return next; })); vi.stubGlobal('cancelAnimationFrame', vi.fn((id: number) => callbacks.delete(id)));
  vi.stubGlobal('VideoFrame', class { displayWidth=640; displayHeight=480; timestamp: number; close=vi.fn(); constructor(v: HTMLVideoElement) { this.timestamp=v.currentTime*1e6; } });
  vi.spyOn(HTMLMediaElement.prototype, 'readyState','get').mockReturnValue(2); vi.spyOn(HTMLVideoElement.prototype,'videoWidth','get').mockReturnValue(640); vi.spyOn(HTMLVideoElement.prototype,'videoHeight','get').mockReturnValue(480);
  synth={ speak:vi.fn((u: {text:string})=>utterances.push(u)), cancel:vi.fn(), getVoices:vi.fn(()=>[]) }; vi.stubGlobal('speechSynthesis',synth); vi.stubGlobal('SpeechSynthesisUtterance',class { constructor(public text:string){} });
});
afterEach(() => { engines.splice(0).forEach(e=>e.dispose()); vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });
function Harness({ engine }: {engine:ImageClassificationEngine}) { const [variant,onVariant]=useState<ImageClassifierVariantId>('lite0'), [active,setActive]=useState(true); return <ImageClassificationWorkspace engine={engine} variant={variant} onVariant={onVariant} onSelectDetection={vi.fn()} cameraActive={active} facingMode="user" onToggleFacing={vi.fn()} onStart={()=>setActive(true)} onStop={()=>setActive(false)} ttsConfig={tts}/>; }
async function mount(factory?:ImageClassificationFactory) { const e=new ImageClassificationEngine(factory); engines.push(e); const view=render(<Harness engine={e}/>); await act(async()=>{}); await act(async()=>{vi.runOnlyPendingTimers();}); return {e,...view}; }
async function tick(time:number) { const video=screen.getByLabelText('画像分類カメラ') as HTMLVideoElement; video.currentTime=time; await act(async()=>{const pending=[...callbacks]; callbacks.clear(); pending.forEach(([,cb])=>cb(0)); vi.runOnlyPendingTimers();}); }
it('Top5 keeps rank separate from SDK IDs, exact raw/translated labels, raw scores and disabled spatial controls', async()=>{
  const t=task(); await mount(async()=>t); const panel=screen.getByRole('region',{name:'画像全体の分類結果'}), items=within(panel).getAllByRole('listitem'); expect(items).toHaveLength(5);
  expect(items[0]).toHaveTextContent('順位 1'); expect(items[0]).toHaveTextContent('SDK index 7'); expect(items[0]).toHaveTextContent('goldfish, Carassius auratus'); expect(items[0]).toHaveTextContent('金魚'); expect(items[0]).toHaveTextContent('0.95'); expect(items[2]).toHaveTextContent('UNKNOWN Exact Raw Label'); expect(items[2]).toHaveTextContent('日本語訳未登録');
  expect(panel).not.toHaveTextContent('%'); expect(panel).not.toHaveTextContent('認識物体'); expect(screen.queryByRole('button',{name:'新規物体の自動読み上げ'})).toBeNull(); within(panel).getAllByRole('checkbox').forEach(c=>expect(c).toBeDisabled());
  fireEvent.click(screen.getByLabelText('画像分類カメラ')); expect(synth.speak).not.toHaveBeenCalled(); expect(t.classifyForVideo).toHaveBeenCalledTimes(1);
});
it('manual speech holds the exact current frame, resumes on matching end, and stop clears speech/results', async()=>{
  const t=task(); await mount(async()=>t); expect(synth.speak).not.toHaveBeenCalled(); fireEvent.click(screen.getByRole('button',{name:/順位1 .*読み上げ/})); expect(synth.speak).toHaveBeenCalledTimes(1); expect(utterances[0].text).toContain('金魚'); expect(utterances[0].text).not.toContain('%');
  await tick(1); expect(t.classifyForVideo).toHaveBeenCalledTimes(1); act(()=>utterances[0].onend?.()); await tick(2); expect(t.classifyForVideo).toHaveBeenCalledTimes(2);
  fireEvent.click(screen.getByRole('button',{name:/順位1 .*読み上げ/})); fireEvent.click(within(screen.getByRole('banner')).getByRole('button',{name:'停止'})); expect(screen.queryAllByRole('listitem')).toHaveLength(0); expect(synth.cancel).toHaveBeenCalled();
  fireEvent.click(within(screen.getByRole('banner')).getByRole('button',{name:'カメラ開始'})); await act(async()=>{vi.runOnlyPendingTimers();}); expect(screen.getAllByRole('listitem')).toHaveLength(5);
});
it('threshold change clears the speaking frame and updates Top5 without automatic speech', async()=>{
  await mount(async()=>task()); fireEvent.click(screen.getByRole('button',{name:/順位1 .*読み上げ/})); fireEvent.change(screen.getByLabelText('分類スコアしきい値'),{target:{value:'.9'}}); expect(synth.cancel).toHaveBeenCalled(); expect(screen.queryAllByRole('listitem')).toHaveLength(0);
  await act(async()=>{vi.runOnlyPendingTimers();}); expect(screen.getAllByRole('listitem')).toHaveLength(1); expect(synth.speak).toHaveBeenCalledTimes(1);
});
it('both variants stay BLOCKED with the production resolver and no model/network action', async()=>{
  const fetch=vi.spyOn(globalThis,'fetch'); await mount(); expect(screen.getByText(/BLOCKED：/)).toBeInTheDocument(); expect(screen.queryAllByRole('listitem')).toHaveLength(0);
  fireEvent.change(within(screen.getByRole('banner')).getByLabelText('バリアント'),{target:{value:'lite2'}}); await act(async()=>{}); expect(screen.getByText(/BLOCKED：/)).toBeInTheDocument(); expect(fetch).not.toHaveBeenCalled();
});
it('initialization failure is not BLOCKED; explicit retry recovers and run failure clears output until retry', async()=>{
  const t=task(), factory=vi.fn<ImageClassificationFactory>().mockRejectedValueOnce(Error('INIT_FAILURE')).mockResolvedValue(t); await mount(factory); expect(screen.getByRole('alert')).toHaveTextContent('初期化に失敗'); expect(screen.queryByText(/BLOCKED：/)).toBeNull(); fireEvent.click(screen.getByText('分類モデルを再試行')); await act(async()=>{}); await act(async()=>{vi.runOnlyPendingTimers();}); expect(screen.getAllByRole('listitem')).toHaveLength(5);
  t.classifyForVideo.mockImplementationOnce(()=>{throw Error('RUN_FAILURE');}); await tick(1); expect(screen.queryAllByRole('listitem')).toHaveLength(0); expect(screen.getByRole('alert')).toHaveTextContent('分類推論に失敗'); fireEvent.click(screen.getByText('分類推論を再試行')); await act(async()=>{vi.runOnlyPendingTimers();}); expect(screen.getAllByRole('listitem')).toHaveLength(5);
});
it('unrelated asset-stage initialization error is a failure, never the intended license gate', async()=>{
  await mount(async()=>{throw new ClassificationInitializationError('asset',Error('WRONG_IDENTITY'));}); expect(screen.getByRole('alert')).toHaveTextContent('初期化に失敗'); expect(screen.queryByText(/BLOCKED：/)).toBeNull();
});
it('rapid ABA switches discard old initialization and preserve only latest variant/frame', async()=>{
  let resolve!:(task:ImageClassificationTask)=>void; const old=task('OBSOLETE'), current=task('CURRENT'); let count=0;
  const {e}=await mount(async()=>++count===1?new Promise<ImageClassificationTask>(yes=>{resolve=yes;}):current);
  const variant=within(screen.getByRole('banner')).getByLabelText('バリアント'); fireEvent.change(variant,{target:{value:'lite2'}}); await act(async()=>{}); fireEvent.change(variant,{target:{value:'lite0'}}); await act(async()=>{}); await act(async()=>{vi.runOnlyPendingTimers();});
  await act(async()=>resolve(old)); await tick(1); expect(screen.queryByText('OBSOLETE')).toBeNull(); expect(screen.getByText('CURRENT')).toBeInTheDocument(); expect(old.close).toHaveBeenCalledTimes(1); expect(e.getState()).toMatchObject({status:'ready',variantId:'lite0'});
});
it('a genuine blocked initialization cause remains distinguishable', async()=>{await mount(async()=>{throw new ClassificationInitializationError('asset',new ClassificationAssetsBlockedError('lite0'));}); expect(screen.getByText(/BLOCKED：/)).toBeInTheDocument();});

it('unavailable manual speech reports failure without hiding valid classification results or leaving a hold', async()=>{
  await mount(async()=>task()); synth.speak.mockImplementationOnce(()=>{throw Error('SPEAK_FAILURE');}); fireEvent.click(screen.getByRole('button',{name:/順位1 .*読み上げ/})); expect(screen.getByText(/音声を開始できません/)).toBeInTheDocument(); expect(screen.getAllByRole('listitem')).toHaveLength(5); expect(screen.queryByText(/フレームを保持中/)).toBeNull();
});
it('variant switch cancels manual speech and late completion cannot alter new variant results', async()=>{
  await mount(async variant=>task(variant==='lite2'?'coffee mug':undefined)); fireEvent.click(screen.getByRole('button',{name:/順位1 .*読み上げ/})); const old=utterances[0];
  fireEvent.change(within(screen.getByRole('banner')).getByLabelText('バリアント'),{target:{value:'lite2'}}); await act(async()=>{}); await act(async()=>{vi.runOnlyPendingTimers();}); act(()=>old.onend?.()); expect(screen.getByText('coffee mug')).toBeInTheDocument(); expect(synth.cancel).toHaveBeenCalled(); expect(synth.speak).toHaveBeenCalledTimes(1); expect(screen.queryByText(/フレームを保持中/)).toBeNull();
});

it('provenance is request-owned rather than inferred from the cache default or a previous model', async()=>{
  const {e}=await mount(async (variant,_delegate,record)=>{if(variant==='lite0'){record('image-classifier:lite0','external');record('wasm-mediapipe','external-fallback');}return task();});
  expect(screen.getByLabelText('分類経路記録')).toHaveTextContent('モデル external ／ WASM external-fallback');
  fireEvent.change(within(screen.getByRole('banner')).getByLabelText('バリアント'),{target:{value:'lite2'}}); await act(async()=>{}); expect(e.getState()).toMatchObject({status:'ready',assetSource:'unreported',wasmSource:'unreported'}); expect(screen.getByLabelText('分類経路記録')).toHaveTextContent('モデル unreported ／ WASM unreported');
});
