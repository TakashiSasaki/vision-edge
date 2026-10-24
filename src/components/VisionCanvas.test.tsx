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
    const next = raf.entries().next().value;
    if (!next) throw new Error('Render loop stopped scheduling');
    raf.delete(next[0]);
    next[1](performance.now());
  });
}
function makeVideoReady(container: HTMLElement) {
  Object.defineProperties(container.querySelector('video')!, {
    videoWidth: { value: 640 }, videoHeight: { value: 480 }, readyState: { value: 2 },
  });
}
beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'performance'] });
  raf = new Map(); nextId = 0;
  vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation(callback => { raf.set(++nextId, callback); return nextId; });
  vi.spyOn(globalThis, 'cancelAnimationFrame').mockImplementation(id => { raf.delete(id); });
  clearRect = vi.fn();
  const ctx = { clearRect, drawImage: vi.fn(), save: vi.fn(), restore: vi.fn(), beginPath: vi.fn(), roundRect: vi.fn(), stroke: vi.fn(), fill: vi.fn(), fillText: vi.fn(), measureText: vi.fn(() => ({ width: 20 })) };
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
