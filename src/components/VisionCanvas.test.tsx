import { flushSync } from 'react-dom';
import { StrictMode } from 'react';
import { speechService } from '../services/speechService';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import App from '../App';
import { VisionCanvas } from './VisionCanvas';
import { VisionEngine, type SegmenterTask, type VisionTaskFactory } from '../services/visionEngine';

let raf: Map<number, FrameRequestCallback>;
let nextId: number;
let clearRect: ReturnType<typeof vi.fn>;
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(yes => { resolve = yes; });
  return { promise, resolve };
}
function fixture() {
  const detect = vi.fn(() => ({ detections: [{
    categories: [{ categoryName: 'dog', displayName: '', index: 0, score: 0.9 }],
    boundingBox: { originX: 64, originY: 48, width: 128, height: 96, angle: 0 }, keypoints: [],
  }] }));
  const segment = vi.fn(() => ({ confidenceMasks: [], categoryMask: undefined, close: vi.fn() }));
  const detector = { detectForVideo: detect, close: vi.fn() };
  const segmenter: SegmenterTask = { segmentForVideo: segment, close: vi.fn() };
  const factory: VisionTaskFactory = {
    load: vi.fn(async () => ({ wasmLoaderPath: 'loader', wasmBinaryPath: 'wasm' })),
    detector: vi.fn(async () => detector), segmenter: vi.fn(async () => segmenter),
  };
  return { engine: new VisionEngine(factory), factory, detector, segmenter, detect, segment };
}
function step() {
  act(() => {
    document.querySelectorAll('video').forEach(video => { video.currentTime += 1 / 30; });
    for (const [id, callback] of [...raf]) {
      raf.delete(id);
      callback(performance.now());
    }
  });
}
function makeVideoReady(container: HTMLElement) {
  Object.defineProperties(container.querySelector('video')!, {
    videoWidth: { value: 640 }, videoHeight: { value: 480 }, readyState: { value: 2 },
  });
}
beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'performance'] });
  raf = new Map(); nextId = 0;
  vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation(callback => { raf.set(++nextId, callback); return nextId; });
  vi.spyOn(globalThis, 'cancelAnimationFrame').mockImplementation(id => { raf.delete(id); });
  clearRect = vi.fn();
  const ctx = { clearRect, drawImage: vi.fn(), save: vi.fn(), restore: vi.fn(), beginPath: vi.fn(), arc: vi.fn(), roundRect: vi.fn(), stroke: vi.fn(), fill: vi.fn(), fillText: vi.fn(), measureText: vi.fn(() => ({ width: 20 })) };
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: vi.fn(async () => {
    const track = Object.assign(new EventTarget(), { stop: vi.fn() });
    return { getTracks: () => [track] } as unknown as MediaStream;
  }) } });
});
afterEach(() => { cleanup(); vi.useRealTimers(); Reflect.deleteProperty(navigator, 'mediaDevices'); });

it('runs the ready model through the real App/canvas bridge and removes old list/overlays on switch', async () => {
  const f = fixture();
  let container!: HTMLElement;
  await act(async () => { container = render(<App engine={f.engine} />).container; });
  await act(async () => { fireEvent.change(screen.getByRole('combobox', { name: 'モデル' }), { target: { value: 'mediapipe' } }); });
  expect(f.engine.getState()).toMatchObject({ status: 'ready', model: 'mediapipe' });
  makeVideoReady(container);
  step();
  act(() => { vi.advanceTimersByTime(0); });
  expect(f.detect).toHaveBeenCalledTimes(1);
  act(() => { vi.advanceTimersByTime(250); });
  step();
  expect(screen.getByText('(dog)')).toBeInTheDocument();
  expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1);
  const pending = deferred<SegmenterTask>();
  vi.mocked(f.factory.segmenter).mockReturnValueOnce(pending.promise);
  const clearCount = clearRect.mock.calls.length;
  await act(async () => { fireEvent.change(screen.getByRole('combobox', { name: 'モデル' }), { target: { value: 'selfie-segmenter' } }); });
  expect(screen.queryByText('(dog)')).not.toBeInTheDocument();
  expect(clearRect.mock.calls.length).toBeGreaterThan(clearCount);
  expect(screen.getByRole('status')).toHaveTextContent('モデル読込中');
  const previousSegments = f.segment.mock.calls.length;
  step();
  act(() => { vi.advanceTimersByTime(0); });
  expect(f.segment).toHaveBeenCalledTimes(previousSegments);
  await act(async () => { pending.resolve(f.segmenter); });
  step();
  act(() => { vi.advanceTimersByTime(0); });
  expect(f.segment.mock.calls.length).toBeGreaterThan(previousSegments);
  act(() => { vi.advanceTimersByTime(250); });
  step();
  expect(screen.queryByText('(dog)')).not.toBeInTheDocument();
  expect(screen.getByText('現在しきい値を超える物体はありません')).toBeInTheDocument();
});

it('cancels a queued old-generation inference and allows the replacement model to run', async () => {
  const f = fixture();
  await f.engine.initialize('mediapipe');
  const onFrame = vi.fn();
  const common = {
    engine: f.engine, facingMode: 'user' as const, onToggleFacingMode: vi.fn(),
    renderOptions: { showBoxes: true, showMasks: true, showLabels: true, showConfidence: true, maskOpacity: 0.35, confidenceThreshold: 0.35 },
    ttsConfig: { enabled: false, autoReadout: false, cooldownSec: 3, rate: 1, pitch: 1, volume: 1, voiceURI: '' },
    isCameraActive: true, onCameraStart: vi.fn(), onCameraStop: vi.fn(), onSelectObject: vi.fn(), onFrame,
  };
  const view = render(<VisionCanvas {...common} activeModel="mediapipe" modelState={f.engine.getState()} />);
  await act(async () => {}); // Complete the real camera hook with the synthetic stream.
  makeVideoReady(view.container);
  step(); // Queued detector call has not executed yet.
  expect(f.detect).not.toHaveBeenCalled();
  await f.engine.initialize('selfie-segmenter');
  view.rerender(<VisionCanvas {...common} activeModel="selfie-segmenter" modelState={f.engine.getState()} />);
  act(() => { vi.advanceTimersByTime(0); });
  expect(f.detect).not.toHaveBeenCalled();
  step();
  act(() => { vi.advanceTimersByTime(0); });
  expect(f.segment).toHaveBeenCalledTimes(1);
  act(() => { vi.advanceTimersByTime(250); });
  step();
  expect(onFrame).toHaveBeenLastCalledWith(expect.objectContaining({ model: 'selfie-segmenter', requestId: f.engine.getState().requestId, objects: [] }));
});

it('camera stop/restart clears the old scene and selection without recreating the ready model', async () => {
  const f = fixture();
  let container!: HTMLElement;
  await act(async () => { container = render(<App engine={f.engine} />).container; });
  await act(async () => { fireEvent.change(screen.getByRole('combobox', { name: 'モデル' }), { target: { value: 'mediapipe' } }); });
  makeVideoReady(container);
  step();
  act(() => { vi.advanceTimersByTime(0); vi.advanceTimersByTime(250); });
  step();
  expect(screen.getByText('(dog)')).toBeInTheDocument();
  const detection = f.detect.getMockImplementation()!;
  act(() => { fireEvent.click(screen.getByText('(dog)')); });
  expect(screen.getByText('(dog)').closest('.group')).toHaveClass('border-blue-500/80');
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: '停止' })); });
  expect(screen.queryByText('(dog)')).not.toBeInTheDocument();
  expect(f.engine.getState()).toMatchObject({ status: 'ready', model: 'mediapipe' });
  expect(f.detector.close).not.toHaveBeenCalled();
  f.detect.mockReturnValue({ detections: [] });
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'カメラ開始' })); });
  expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(2);
  step();
  act(() => { vi.advanceTimersByTime(0); vi.advanceTimersByTime(250); });
  step();
  expect(screen.queryByText('(dog)')).not.toBeInTheDocument();
  expect(screen.getByText('現在しきい値を超える物体はありません')).toBeInTheDocument();
  f.detect.mockImplementation(detection);
  act(() => { vi.advanceTimersByTime(0); vi.advanceTimersByTime(250); });
  step();
  expect(screen.getByText('(dog)').closest('.group')).not.toHaveClass('border-blue-500/80');
  expect(f.factory.detector).toHaveBeenCalledTimes(1);
  expect(f.detector.close).not.toHaveBeenCalled();
});

it('does not infer from stale video frames while camera acquisition or playback is pending', async () => {
  const f = fixture();
  await f.engine.initialize('mediapipe');
  const acquisition = deferred<MediaStream>();
  const playback = deferred<void>();
  vi.mocked(navigator.mediaDevices.getUserMedia).mockReturnValue(acquisition.promise);
  vi.mocked(HTMLMediaElement.prototype.play).mockReturnValue(playback.promise);
  const props = {
    engine: f.engine, facingMode: 'user' as const, onToggleFacingMode: vi.fn(),
    renderOptions: { showBoxes: true, showMasks: true, showLabels: true, showConfidence: true, maskOpacity: 0.35, confidenceThreshold: 0.35 },
    ttsConfig: { enabled: false, autoReadout: false, cooldownSec: 3, rate: 1, pitch: 1, volume: 1, voiceURI: '' },
    isCameraActive: true, onCameraStart: vi.fn(), onCameraStop: vi.fn(), onSelectObject: vi.fn(), onFrame: vi.fn(),
  };
  const view = render(<VisionCanvas {...props} activeModel="mediapipe" modelState={f.engine.getState()} />);
  makeVideoReady(view.container); // A previous video can still expose readyState >= 2.
  step();
  act(() => { vi.advanceTimersByTime(0); });
  expect(f.detect).not.toHaveBeenCalled();
  const track = Object.assign(new EventTarget(), { stop: vi.fn() });
  await act(async () => { acquisition.resolve({ getTracks: () => [track] } as unknown as MediaStream); });
  step();
  act(() => { vi.advanceTimersByTime(0); });
  expect(f.detect).not.toHaveBeenCalled();
  await act(async () => { playback.resolve(); });
  step();
  act(() => { vi.advanceTimersByTime(0); });
  expect(f.detect).toHaveBeenCalledTimes(1);
  view.unmount();
  expect(track.stop).toHaveBeenCalledTimes(1);
  f.engine.dispose();
});

function canvasProps(f: ReturnType<typeof fixture>) {
  return {
    engine: f.engine, activeModel: 'mediapipe' as const, modelState: f.engine.getState(),
    facingMode: 'user' as const, onToggleFacingMode: vi.fn(),
    renderOptions: { showBoxes: true, showMasks: true, showLabels: true, showConfidence: true, maskOpacity: 0.35, confidenceThreshold: 0.35 },
    ttsConfig: { enabled: false, autoReadout: false, cooldownSec: 3, rate: 1, pitch: 1, volume: 1, voiceURI: '' },
    isCameraActive: true, onCameraStart: vi.fn(), onCameraStop: vi.fn(), onSelectObject: vi.fn(), onFrame: vi.fn(),
  };
}
it('does not infer the same fallback frame repeatedly, while still drawing and publishing at bounded frequency', async () => {
  const f = fixture(); await f.engine.initialize('mediapipe'); const props = canvasProps(f);
  props.ttsConfig = { ...props.ttsConfig, enabled: true, autoReadout: true };
  const decision = vi.spyOn(speechService, 'canSpeakLabel').mockReturnValue(true);
  vi.spyOn(speechService, 'speak').mockReturnValue(true);
  const view = render(<VisionCanvas {...props} />); await act(async () => {}); makeVideoReady(view.container);
  step(); act(() => { vi.advanceTimersByTime(0); });
  for (let i = 0; i < 20; i++) {
    act(() => {
      for (const [id, callback] of [...raf]) { raf.delete(id); callback(performance.now()); }
      vi.advanceTimersByTime(10);
    });
  }
  expect(f.detect).toHaveBeenCalledTimes(1);
  expect(decision).toHaveBeenCalledTimes(1);
  expect(props.onFrame.mock.calls.filter(([frame]) => frame !== null)).toHaveLength(0);
  act(() => { vi.advanceTimersByTime(50); });
  expect(props.onFrame.mock.calls.filter(([frame]) => frame !== null)).toHaveLength(1);
  act(() => { vi.advanceTimersByTime(1000); });
  expect(props.onFrame.mock.calls.filter(([frame]) => frame !== null)).toHaveLength(1);
});
it.each(['draw', 'notification', 'tts'] as const)('exposes a stopped %s fault and recovers explicitly without recreating camera or model', async stage => {
  const f = fixture(); await f.engine.initialize('mediapipe'); const props = canvasProps(f);
  if (stage === 'tts') props.ttsConfig = { ...props.ttsConfig, enabled: true, autoReadout: true };
  const speak = vi.spyOn(speechService, 'speak').mockImplementation(() => true);
  vi.spyOn(speechService, 'canSpeakLabel').mockReturnValue(true);
  const view = render(<VisionCanvas {...props} />); await act(async () => {}); makeVideoReady(view.container);
  const ctx = view.container.querySelector('canvas')!.getContext('2d')!;
  if (stage === 'draw') vi.mocked(ctx.drawImage).mockImplementationOnce(() => { throw new Error('draw fault'); });
  if (stage === 'notification') props.onFrame.mockImplementationOnce(() => { throw new Error('notify fault'); });
  if (stage === 'tts') speak.mockImplementationOnce(() => { throw new Error('speech fault'); });
  step(); act(() => { vi.advanceTimersByTime(0); }); step();
  act(() => { vi.advanceTimersByTime(250); });
  expect(screen.getByRole('alert')).toHaveTextContent('処理を停止しました');
  const count = f.detect.mock.calls.length;
  step(); act(() => { vi.advanceTimersByTime(500); }); expect(f.detect).toHaveBeenCalledTimes(count);
  act(() => { fireEvent.click(screen.getByRole('button', { name: '処理を再試行' })); });
  step(); act(() => { vi.advanceTimersByTime(0); }); step(); act(() => { vi.advanceTimersByTime(250); });
  expect(screen.queryByRole('alert')).not.toBeInTheDocument(); expect(f.detect.mock.calls.length).toBeGreaterThan(count);
  expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1); expect(f.factory.detector).toHaveBeenCalledTimes(1);
});
it('keeps latest callbacks, appearance and ripple without restarting frame acquisition', async () => {
  const f = fixture(); await f.engine.initialize('mediapipe'); const props = canvasProps(f);
  const view = render(<VisionCanvas {...props} />); await act(async () => {}); makeVideoReady(view.container);
  step(); act(() => { vi.advanceTimersByTime(0); vi.advanceTimersByTime(250); }); step();
  const canvas = view.container.querySelector('canvas')!; const ctx = canvas.getContext('2d')!;
  vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({ left: 0, top: 0, width: 640, height: 480 } as DOMRect);
  act(() => { fireEvent.click(canvas, { clientX: 100, clientY: 100 }); }); step();
  expect(ctx.arc).toHaveBeenCalled();
  const callback = vi.fn(); const pendingIds = [...raf.keys()];
  view.rerender(<VisionCanvas {...props} onFrame={callback} renderOptions={{ ...props.renderOptions, showBoxes: false }} />);
  expect([...raf.keys()]).toEqual(pendingIds);
  const strokes = vi.mocked(ctx.roundRect).mock.calls.length; step();
  expect(vi.mocked(ctx.roundRect).mock.calls.length).toBe(strokes);
  act(() => { vi.advanceTimersByTime(0); vi.advanceTimersByTime(250); });
  expect(callback).toHaveBeenCalledWith(expect.objectContaining({ model: 'mediapipe' }));
  expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1);
});
it('cancels both loops and pending execution on StrictMode teardown without duplicated callbacks', async () => {
  const f = fixture(); await f.engine.initialize('mediapipe');
  const view = render(<StrictMode><VisionCanvas {...canvasProps(f)} /></StrictMode>); await act(async () => {});
  makeVideoReady(view.container); expect(raf.size).toBe(2);
  step(); view.unmount(); act(() => { vi.advanceTimersByTime(1000); });
  expect(raf.size).toBe(0); expect(f.detect).not.toHaveBeenCalled();
});
it('counts valid empty inference completions and actual draws, excluding failed work and aging idle metrics', async () => {
  const f = fixture(); await f.engine.initialize('mediapipe'); const props = canvasProps(f);
  let now = 0; vi.spyOn(performance, 'now').mockImplementation(() => now);
  f.detect.mockImplementation(() => { now += 5; return { detections: [] }; });
  const view = render(<VisionCanvas {...props} />); await act(async () => {}); makeVideoReady(view.container);
  for (let i = 0; i < 4; i++) {
    now = i * 500;
    step(); act(() => { vi.advanceTimersByTime(0); });
    act(() => { vi.advanceTimersByTime(250); });
  }
  expect(screen.getByLabelText('推論FPS')).toHaveTextContent('2.0 FPS');
  expect(screen.getByLabelText('フレーム処理時間')).toHaveTextContent('5.0 ms');
  expect(screen.getByLabelText('描画FPS')).toHaveTextContent('2.0 FPS');
  expect(props.onFrame).toHaveBeenLastCalledWith(expect.objectContaining({ objects: [] }));
  now = 2000; f.detect.mockImplementationOnce(() => { now += 10; throw new Error('SDK fault'); });
  step(); act(() => { vi.advanceTimersByTime(0); vi.advanceTimersByTime(250); });
  expect(screen.getByLabelText('フレーム処理時間')).toHaveTextContent('5.0 ms');
  expect(screen.getByLabelText('性能状態')).toHaveTextContent('推論エラー');
  expect(props.onFrame).toHaveBeenLastCalledWith(null);
  now = 5000; act(() => { vi.advanceTimersByTime(250); });
  expect(screen.getByLabelText('推論FPS')).toHaveTextContent('未計測');
  expect(screen.getByLabelText('フレーム処理時間')).toHaveTextContent('未計測');
  now = 5500; step(); act(() => { vi.advanceTimersByTime(0); vi.advanceTimersByTime(250); });
  expect(screen.getByLabelText('性能状態')).toHaveTextContent('計測中');
});
it.each(['model', 'camera', 'threshold', 'unmount'] as const)('rejects a result invalidated synchronously during inference by %s generation', async kind => {
  const f = fixture(); await f.engine.initialize('mediapipe'); const props = canvasProps(f);
  const view = render(<VisionCanvas {...props} />); await act(async () => {}); makeVideoReady(view.container);
  const detect = f.detect.getMockImplementation()!;
  f.detect.mockImplementationOnce(() => {
    if (kind === 'model') f.engine.dispose();
    else if (kind === 'unmount') view.unmount();
    else if (kind === 'threshold') flushSync(() => { view.rerender(<VisionCanvas {...props} renderOptions={{ ...props.renderOptions, confidenceThreshold: 0.95 }} />); });
    else flushSync(() => { view.rerender(<VisionCanvas {...props} isCameraActive={false} />); });
    return detect();
  });
  step(); act(() => { vi.advanceTimersByTime(0); vi.advanceTimersByTime(250); });
  expect(props.onFrame.mock.calls.filter(([frame]) => frame !== null)).toHaveLength(0);
  expect(screen.queryByText('(dog)')).not.toBeInTheDocument();
});
