import { flushSync } from 'react-dom';
import { StrictMode } from 'react';
import { speechService } from '../services/speechService';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import App from '../App';
import { PipelineMetrics } from '../services/pipelineMetrics';
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
function step(advanceVideo = true) {
  act(() => {
    if (advanceVideo) document.querySelectorAll('video').forEach(video => { video.currentTime += 1 / 30; });
    for (const [id, callback] of [...raf]) {
      raf.delete(id);
      callback(performance.now());
    }
  });
}
function makeVideoReady(container: HTMLElement) {
  Object.defineProperties(container.querySelector('video')!, {
    videoWidth: { configurable: true, value: 640 }, videoHeight: { configurable: true, value: 480 }, readyState: { configurable: true, value: 2 },
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
  Object.defineProperty(HTMLVideoElement.prototype, 'getVideoPlaybackQuality', { configurable: true, value: function(this: HTMLVideoElement) { return { totalVideoFrames: Math.floor(this.currentTime * 1000000) + 1 }; } });
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: vi.fn(async () => {
    const track = Object.assign(new EventTarget(), { stop: vi.fn() });
    return { getTracks: () => [track] } as unknown as MediaStream;
  }) } });
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); Reflect.deleteProperty(HTMLVideoElement.prototype, 'getVideoPlaybackQuality'); Reflect.deleteProperty(navigator, 'mediaDevices'); });

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
  expect(f.segment).toHaveBeenCalledTimes(1); // Ready replacement bootstraps the existing frame.
  step(false); // An arrival for that same image must not run it twice.
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
it.each(['draw', 'notification', 'tts', 'capture'] as const)('exposes a stopped %s fault and recovers explicitly without recreating camera or model', async stage => {
  const f = fixture(); await f.engine.initialize('mediapipe'); const props = canvasProps(f);
  if (stage === 'tts') props.ttsConfig = { ...props.ttsConfig, enabled: true, autoReadout: true };
  const speak = vi.spyOn(speechService, 'speak').mockImplementation(() => true);
  vi.spyOn(speechService, 'canSpeakLabel').mockReturnValue(true);
  const view = render(<VisionCanvas {...props} />); await act(async () => {}); makeVideoReady(view.container);
  const ctx = view.container.querySelector('canvas')!.getContext('2d')!;
  if (stage === 'capture') vi.stubGlobal('VideoFrame', class { constructor() { throw new Error('capture fault'); } });
  if (stage === 'draw') vi.mocked(ctx.drawImage).mockImplementationOnce(() => { throw new Error('draw fault'); });
  if (stage === 'notification') props.onFrame.mockImplementationOnce(() => { throw new Error('notify fault'); });
  if (stage === 'tts') speak.mockImplementationOnce(() => { throw new Error('speech fault'); });
  step(); act(() => { vi.advanceTimersByTime(0); }); step();
  act(() => { vi.advanceTimersByTime(250); });
  expect(screen.getByRole('alert')).toHaveTextContent('処理を停止しました');
  const count = f.detect.mock.calls.length;
  step(); act(() => { vi.advanceTimersByTime(500); }); expect(f.detect).toHaveBeenCalledTimes(count);
  if (stage === 'capture') vi.stubGlobal('VideoFrame', undefined);
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
  vi.spyOn(view.container.querySelector('video')!, 'getBoundingClientRect').mockReturnValue({ left: 0, top: 0, width: 640, height: 480 } as DOMRect);
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

it.each([
  { name: 'horizontal', rect: { left: 20, top: 40, width: 960, height: 540 }, margin: [120, 148], point: [351.5, 195.25] },
  { name: 'vertical', rect: { left: 20, top: 40, width: 360, height: 640 }, margin: [92, 200], point: [125.75, 302.625] },
])('selects the visible BBox and ignores $name margins without selection, ripple or speech', async ({ rect, margin, point }) => {
  const f = fixture(); await f.engine.initialize('mediapipe'); const props = canvasProps(f);
  const speak = vi.spyOn(speechService, 'speak').mockReturnValue(true);
  const find = vi.spyOn(f.engine, 'findObjectAtPoint');
  const view = render(<VisionCanvas {...props} />); await act(async () => {}); makeVideoReady(view.container);
  step(); act(() => { vi.advanceTimersByTime(0); vi.advanceTimersByTime(250); }); step();
  const video = view.container.querySelector('video')!; const canvas = view.container.querySelector('canvas')!;
  const ctx = canvas.getContext('2d')!;
  vi.spyOn(video, 'getBoundingClientRect').mockReturnValue(rect as DOMRect);
  // A stale backing size or different Canvas CSS box must not supply video geometry.
  vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({ left: 0, top: 0, width: 300, height: 150 } as DOMRect);
  act(() => { fireEvent.click(canvas, { clientX: margin[0], clientY: margin[1] }); }); step(false);
  expect(find).not.toHaveBeenCalled(); expect(props.onSelectObject).not.toHaveBeenCalled();
  expect(speak).not.toHaveBeenCalled(); expect(ctx.arc).not.toHaveBeenCalled();
  act(() => { fireEvent.click(canvas, { clientX: point[0], clientY: point[1] }); }); step(false);
  expect(find).toHaveBeenLastCalledWith(0.29375, 0.2875, expect.any(Array));
  expect(props.onSelectObject).toHaveBeenCalledWith(expect.objectContaining({ label: 'dog' }));
  expect(ctx.arc).toHaveBeenLastCalledWith(188, 138, 24, 0, Math.PI * 2);
  expect(speak).toHaveBeenCalledTimes(1);
  // A margin click preserves an existing selection/ripple and never speaks again.
  act(() => { fireEvent.click(canvas, { clientX: margin[0], clientY: margin[1] }); }); step(false);
  expect(props.onSelectObject).toHaveBeenCalledTimes(1); expect(speak).toHaveBeenCalledTimes(1);
  const callbacks = [...raf.keys()];
  view.rerender(<VisionCanvas {...props} selectedObjectId="det-0-dog" renderOptions={{ ...props.renderOptions, maskOpacity: 0.6 }} />);
  expect([...raf.keys()]).toEqual(callbacks);
  expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1);
  expect(f.factory.detector).toHaveBeenCalledTimes(1); expect(f.detector.close).not.toHaveBeenCalled();
});

it.each(['maskCanvas', 'maskPolygon'] as const)('passes video-normalized coordinates to existing %s hit testing and drawing', async kind => {
  const f = fixture(); await f.engine.initialize('mediapipe'); const props = canvasProps(f);
  vi.spyOn(speechService, 'speak').mockReturnValue(true);
  const view = render(<VisionCanvas {...props} />); await act(async () => {}); makeVideoReady(view.container);
  step(); act(() => { vi.advanceTimersByTime(0); vi.advanceTimersByTime(250); }); step();
  const frame = props.onFrame.mock.calls.at(-1)![0]!;
  const object = frame.objects[0]; object.bbox = { x: 0.8, y: 0.8, width: 0.1, height: 0.1 };
  const canvas = view.container.querySelector('canvas')!; const ctx = canvas.getContext('2d')!;
  const pixel = vi.fn(() => ({ data: new Uint8ClampedArray([0, 0, 0, 255]) }));
  if (kind === 'maskCanvas') {
    const mask = document.createElement('canvas'); mask.width = 320; mask.height = 240;
    Object.defineProperty(mask, 'getContext', { value: () => ({ getImageData: pixel }) });
    object.maskCanvas = mask;
  } else {
    object.maskPolygon = [{ x: 0.1, y: 0.1 }, { x: 0.3, y: 0.1 }, { x: 0.3, y: 0.3 }, { x: 0.1, y: 0.3 }];
    Object.assign(ctx, { moveTo: vi.fn(), lineTo: vi.fn(), closePath: vi.fn() });
  }
  vi.spyOn(view.container.querySelector('video')!, 'getBoundingClientRect').mockReturnValue({ left: 0, top: 0, width: 960, height: 540 } as DOMRect);
  act(() => { fireEvent.click(canvas, { clientX: 331.5, clientY: 155.25 }); }); step(false);
  expect(props.onSelectObject).toHaveBeenCalledWith(object);
  expect(ctx.arc).toHaveBeenLastCalledWith(188, 138, 24, 0, Math.PI * 2);
  if (kind === 'maskCanvas') {
    expect(pixel).toHaveBeenCalledWith(94, 69, 1, 1);
    expect(ctx.drawImage).toHaveBeenCalledWith(object.maskCanvas, 0, 0, 640, 480);
  } else expect(ctx.moveTo).toHaveBeenCalledWith(64, 48);
});

it('reads resized/scrolled boxes and changed intrinsic dimensions immediately before input without recreating resources', async () => {
  const f = fixture(); await f.engine.initialize('mediapipe'); const props = canvasProps(f);
  const speak = vi.spyOn(speechService, 'speak').mockReturnValue(true);
  const find = vi.spyOn(f.engine, 'findObjectAtPoint');
  const view = render(<VisionCanvas {...props} />); await act(async () => {}); makeVideoReady(view.container);
  step(); act(() => { vi.advanceTimersByTime(0); vi.advanceTimersByTime(250); }); step();
  const video = view.container.querySelector('video')!; const canvas = view.container.querySelector('canvas')!;
  const rect = vi.spyOn(video, 'getBoundingClientRect').mockReturnValue({ left: 50, top: -100, width: 960, height: 540 } as DOMRect);
  const callbacks = [...raf.keys()];
  // Intrinsic resize precedes both a render and the next drawing frame.
  Object.defineProperties(video, { videoWidth: { value: 1280 }, videoHeight: { value: 720 } });
  act(() => { fireEvent.click(canvas, { clientX: 242, clientY: 8 }); });
  expect(find).toHaveBeenLastCalledWith(0.2, 0.2, expect.any(Array));
  expect(canvas.width).toBe(640); expect(canvas.height).toBe(480);
  expect([...raf.keys()]).toEqual(callbacks); step(false);
  expect(canvas.width).toBe(1280); expect(canvas.height).toBe(720);
  expect(canvas.getContext('2d')!.arc).toHaveBeenLastCalledWith(256, 144, 24, 0, Math.PI * 2);
  rect.mockReturnValue({ left: -20, top: 80, width: 800, height: 600 } as DOMRect);
  act(() => { fireEvent.click(canvas, { clientX: 140, clientY: 245 }); }); step(false);
  expect(find).toHaveBeenLastCalledWith(0.2, 0.2, expect.any(Array));
  expect(props.onSelectObject).toHaveBeenCalledTimes(2); expect(speak).toHaveBeenCalledTimes(2);
  expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1);
  expect(f.factory.detector).toHaveBeenCalledTimes(1); expect(f.detector.close).not.toHaveBeenCalled();
});

it('ignores unavailable/zero/nonfinite video geometry before hit testing or speech', async () => {
  const f = fixture(); await f.engine.initialize('mediapipe'); const props = canvasProps(f);
  const speak = vi.spyOn(speechService, 'speak').mockReturnValue(true);
  const find = vi.spyOn(f.engine, 'findObjectAtPoint');
  const view = render(<VisionCanvas {...props} />); await act(async () => {}); makeVideoReady(view.container);
  step(); act(() => { vi.advanceTimersByTime(0); vi.advanceTimersByTime(250); }); step();
  const video = view.container.querySelector('video')!; const canvas = view.container.querySelector('canvas')!;
  const rect = vi.spyOn(video, 'getBoundingClientRect');
  for (const width of [0, NaN, Infinity]) {
    rect.mockReturnValue({ left: 0, top: 0, width, height: 540 } as DOMRect);
    fireEvent.click(canvas, { clientX: 200, clientY: 100 });
  }
  rect.mockReturnValue({ left: 0, top: 0, width: 960, height: 540 } as DOMRect);
  Object.defineProperty(video, 'videoWidth', { value: 0 }); fireEvent.click(canvas, { clientX: 200, clientY: 100 });
  Object.defineProperty(video, 'videoWidth', { value: 640 });
  Object.defineProperty(video, 'readyState', { value: 0 }); fireEvent.click(canvas, { clientX: 200, clientY: 100 });
  expect(find).not.toHaveBeenCalled(); expect(props.onSelectObject).not.toHaveBeenCalled();
  expect(speak).not.toHaveBeenCalled(); expect(canvas.getContext('2d')!.arc).not.toHaveBeenCalled();
});

it('selects exact rational content edges but keeps adjacent fractional margins silent (PR05-F01)', async () => {
  const f = fixture(); await f.engine.initialize('mediapipe'); const props = canvasProps(f);
  const speak = vi.spyOn(speechService, 'speak').mockReturnValue(true);
  const find = vi.spyOn(f.engine, 'findObjectAtPoint');
  const view = render(<VisionCanvas {...props} />); await act(async () => {}); makeVideoReady(view.container);
  step(); act(() => { vi.advanceTimersByTime(0); vi.advanceTimersByTime(250); }); step();
  props.onFrame.mock.calls.at(-1)![0]!.objects[0].bbox = { x: 0, y: 0, width: 1, height: 1 };
  const canvas = view.container.querySelector('canvas')!;
  vi.spyOn(view.container.querySelector('video')!, 'getBoundingClientRect').mockReturnValue({ left: 37.25, top: 82.5, width: 960.75, height: 540.25 } as DOMRect);
  for (const clientX of [157.45833333333331, 877.7916666666667]) {
    act(() => { fireEvent.click(canvas, { clientX, clientY: 352.625 }); });
  }
  expect(find).not.toHaveBeenCalled(); expect(speak).not.toHaveBeenCalled();
  expect(props.onSelectObject).not.toHaveBeenCalled();
  for (const [clientX, x] of [[3779 / 24, 0], [21067 / 24, 1]]) {
    act(() => { fireEvent.click(canvas, { clientX, clientY: 352.625 }); }); step(false);
    expect(find).toHaveBeenLastCalledWith(x, 0.5, expect.any(Array));
    expect(canvas.getContext('2d')!.arc).toHaveBeenLastCalledWith(x * 640, 240, 24, 0, Math.PI * 2);
  }
  expect(props.onSelectObject).toHaveBeenCalledTimes(2); expect(speak).toHaveBeenCalledTimes(2);
  expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1);
  expect(f.factory.detector).toHaveBeenCalledTimes(1);
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
it('F01 real canvas/engine: a delayed arrival cannot adopt or announce the same live image twice', async () => {
  const f = fixture(); await f.engine.initialize('mediapipe'); const props = canvasProps(f);
  props.ttsConfig = { ...props.ttsConfig, enabled: true, autoReadout: true };
  const gate = vi.spyOn(speechService, 'canSpeakLabel').mockReturnValue(true);
  vi.spyOn(speechService, 'speak').mockReturnValue(true);
  const inputs: number[] = [];
  const detect = f.detect.getMockImplementation()!;
  f.detect.mockImplementation((videoElement?: HTMLVideoElement) => {
    // Read the actual SDK-boundary argument rather than the scheduler candidate.
    inputs.push(videoElement!.currentTime);
    return detect();
  });
  const view = render(<VisionCanvas {...props} />); await act(async () => {}); makeVideoReady(view.container);
  const video = view.container.querySelector('video')!;
  video.currentTime = 1; step(false);
  video.currentTime = 2; act(() => { vi.advanceTimersByTime(0); });
  act(() => { vi.advanceTimersByTime(500); }); step(false);
  act(() => { vi.advanceTimersByTime(0); vi.advanceTimersByTime(250); });
  expect(inputs).toEqual([2]); expect(gate).toHaveBeenCalledTimes(1);
  expect(props.onFrame.mock.calls.filter(([frame]) => frame !== null)).toHaveLength(1);
  expect(screen.getByLabelText('推論FPS')).toHaveTextContent('未計測'); // One distinct successful input, not 2 FPS.
});

// PR06 uses the actual App, scheduler and Canvas copy, not onFrame callbacks.
it('records a successful first Canvas draw for an empty immutable result once, including measured zero latency', async () => {
  vi.stubGlobal('VideoFrame', class {
    timestamp: number; displayWidth = 640; displayHeight = 480;
    constructor(video: HTMLVideoElement) { this.timestamp = video.currentTime * 1000000; }
    close() {}
  });
  const f = fixture(); const accepts = vi.spyOn(PipelineMetrics.prototype, 'accept'); const firstDraws = vi.spyOn(PipelineMetrics.prototype, 'firstDraw');
  let container!: HTMLElement; await act(async () => { container = render(<App engine={f.engine} />).container; });
  makeVideoReady(container); step(); act(() => { vi.advanceTimersByTime(0); });
  expect(accepts).toHaveBeenCalledTimes(1); expect(firstDraws).not.toHaveBeenCalled();
  step(false); step(false); step(false);
  expect(firstDraws).toHaveBeenCalledTimes(1);
  expect(firstDraws.mock.calls[0][0]).toBe(accepts.mock.calls[0][0]);
  const metrics = accepts.mock.contexts[0] as PipelineMetrics;
  expect(metrics.snapshot()!.stages.latency.success).toMatchObject({ count: 1, mean: 0 });
  expect(metrics.snapshot()!.results['first-draw']).toBe(1);
  expect(f.segment).toHaveBeenCalledTimes(1);
});
it('records overwritten results separately and binds the first draw to the latest accepted job', async () => {
  const f = fixture(); const accepts = vi.spyOn(PipelineMetrics.prototype, 'accept'); const draws = vi.spyOn(PipelineMetrics.prototype, 'firstDraw');
  let container!: HTMLElement; await act(async () => { container = render(<App engine={f.engine} />).container; });
  makeVideoReady(container); step(); act(() => { vi.advanceTimersByTime(0); });
  expect(accepts).toHaveBeenCalledTimes(1);
  act(() => {
    container.querySelector('video')!.currentTime += 1;
    const [id, callback] = [...raf][0]; raf.delete(id); callback(0); // acquisition only; draw has not run
    vi.advanceTimersByTime(0);
  });
  expect(accepts).toHaveBeenCalledTimes(2); expect(draws).not.toHaveBeenCalled();
  const metrics = accepts.mock.contexts[0] as PipelineMetrics;
  expect(metrics.snapshot()!.results.overwritten).toBe(1);
  step(false); expect(draws.mock.calls[0][0]).toBe(accepts.mock.calls[1][0]);
  expect(metrics.snapshot()!.results).toMatchObject({ overwritten: 1, 'first-draw': 1, unbound: 1 });
});
it.each(['hidden', 'stop', 'draw-failed'] as const)('never substitutes zero for %s result and prevents a later recovery latency', async outcome => {
  const f = fixture(); const accepts = vi.spyOn(PipelineMetrics.prototype, 'accept'); const draws = vi.spyOn(PipelineMetrics.prototype, 'firstDraw');
  let container!: HTMLElement; await act(async () => { container = render(<App engine={f.engine} />).container; });
  makeVideoReady(container); step(); act(() => { vi.advanceTimersByTime(0); });
  const metrics = accepts.mock.contexts[0] as PipelineMetrics;
  if (outcome === 'hidden') {
    vi.spyOn(document, 'hidden', 'get').mockReturnValue(true);
    act(() => { document.dispatchEvent(new Event('visibilitychange')); });
    step(false);
  } else if (outcome === 'stop') {
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: /^停止$/ })); });
  } else {
    const ctx = container.querySelector('canvas')!.getContext('2d')!;
    vi.mocked(ctx.drawImage).mockImplementationOnce(() => { throw Error('copy'); }); step(false);
    expect(screen.getByRole('alert')).toHaveTextContent('描画エラー');
  }
  expect(draws).not.toHaveBeenCalled();
  expect(metrics.snapshot()!.stages.latency.success.mean).toBeNull();
  expect(metrics.snapshot()!.results[outcome === 'stop' ? 'cancelled' : outcome]).toBe(1);
});
it('statistics exceptions and details toggles do not change SDK, camera, model or Canvas ownership', async () => {
  const f = fixture(); const snapshot = vi.spyOn(PipelineMetrics.prototype, 'snapshot').mockImplementation(() => { throw Error('statistics'); });
  let container!: HTMLElement; await act(async () => { container = render(<App engine={f.engine} />).container; });
  makeVideoReady(container); step(); act(() => { vi.advanceTimersByTime(0); }); step(false);
  act(() => { vi.advanceTimersByTime(250); }); expect(snapshot).toHaveBeenCalled();
  const loadCount = vi.mocked(f.factory.load).mock.calls.length;
  for (let i = 0; i < 4; i++) fireEvent.click(screen.getByText('計測詳細 (JS wall time)'));
  step(); act(() => { vi.advanceTimersByTime(0); }); step(false);
  expect(f.segment).toHaveBeenCalledTimes(2); expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1);
  expect(f.factory.load).toHaveBeenCalledTimes(loadCount); expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});

it('invalidates the published detail snapshot when a fatal draw fault stops its generation', async () => {
  const f = fixture(); let container!: HTMLElement;
  await act(async () => { container = render(<App engine={f.engine} />).container; });
  makeVideoReady(container); step(); act(() => { vi.advanceTimersByTime(0); }); step(false);
  act(() => { vi.advanceTimersByTime(250); });
  const count = () => container.querySelector('table[aria-label="success interval statistics"] tbody tr td')!.textContent;
  expect(count()).toMatch(/[1-9]\d* \/ [1-9]\d* \/ 0/);
  vi.mocked(container.querySelector('canvas')!.getContext('2d')!.drawImage).mockImplementationOnce(() => { throw Error('copy'); });
  step(false); expect(screen.getByRole('alert')).toHaveTextContent('描画エラー'); expect(count()).toBe('未計測');
});
