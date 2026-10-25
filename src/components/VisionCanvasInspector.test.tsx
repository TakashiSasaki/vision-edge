// Actual component/scheduler/engine/service integration; only ORT and browser I/O
// are controlled. No model bytes, network or inference-success claim is involved.
import { act, cleanup, fireEvent, render } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type * as ort from 'onnxruntime-web';
import { VisionCanvas } from './VisionCanvas';
import { VisionEngine, type VisionTaskFactory } from '../services/visionEngine';
import { SegFormerEngine } from '../services/segformerService';
import { speechService } from '../services/speechService';
import type { DetectionFrame } from '../types/vision';

vi.mock('onnxruntime-web', () => ({
  Tensor: class { constructor(public type: string, public data: Float32Array, public dims: number[]) {} dispose() {} },
}));
let raf: Map<number, FrameRequestCallback>;
let nextId: number;
let drawImage: ReturnType<typeof vi.fn>;
const pixels = new WeakMap<HTMLCanvasElement, ImageData>();
function context(canvas: HTMLCanvasElement) {
  return {
    drawImage, clearRect: vi.fn(), save: vi.fn(), restore: vi.fn(), beginPath: vi.fn(), arc: vi.fn(), roundRect: vi.fn(), stroke: vi.fn(), fill: vi.fn(), fillText: vi.fn(), measureText: () => ({ width: 20 }),
    createImageData: (w: number, h: number) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: (image: ImageData) => pixels.set(canvas, image),
    getImageData: (_x: number, _y: number, w: number, h: number) => ({ width: w, height: h, data: pixels.get(canvas)?.data ?? new Uint8ClampedArray(w * h * 4).fill(255) }),
  } as unknown as CanvasRenderingContext2D;
}
function logits(selected = 0, weak = false) {
  const data = new Float32Array(150 * 100);
  data.fill(weak ? .1 : 10, selected * 100, (selected + 1) * 100);
  return { logits: { dims: [1, 150, 10, 10], data } };
}
function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>(yes => { resolve = yes; }); return { promise, resolve }; }
function step(advanceVideo = false) {
  act(() => {
    if (advanceVideo) document.querySelectorAll('video').forEach(v => { v.currentTime += 1 / 30; });
    for (const [id, callback] of [...raf]) { raf.delete(id); callback(performance.now()); }
  });
}
async function runFrame() {
  step(true);
  await act(async () => { vi.advanceTimersByTime(0); });
  step(); // Draw the newly adopted result, without advancing the throttled HUD.
}
async function fixture(weak = false) {
  const session = { inputNames: ['pixel_values'], outputNames: ['logits'], run: vi.fn(async () => logits(0, weak)), release: vi.fn(async () => {}) };
  const task = new SegFormerEngine(session as unknown as ort.InferenceSession);
  const factory: VisionTaskFactory = {
    load: async () => ({ wasmLoaderPath: 'unused', wasmBinaryPath: 'unused' }),
    detector: async () => ({ detectForVideo: () => ({ detections: [] }), close: vi.fn() }),
    segmenter: async () => ({ segmentForVideo: () => ({ confidenceMasks: [], categoryMask: undefined, close: vi.fn() }), close: vi.fn() }),
    segformer: async () => task,
  };
  const engine = new VisionEngine(factory);
  await engine.initialize('segformer');
  const props = {
    engine, activeModel: 'segformer' as const, modelState: engine.getState(), facingMode: 'user' as const, onToggleFacingMode: vi.fn(),
    renderOptions: { showBoxes: true, showMasks: true, showLabels: true, showConfidence: true, maskOpacity: .35, confidenceThreshold: .3 },
    ttsConfig: { enabled: false, autoReadout: false, cooldownSec: 3, rate: 1, pitch: 1, volume: 1, voiceURI: '' },
    isCameraActive: true, onCameraStart: vi.fn(), onCameraStop: vi.fn(), onSelectObject: vi.fn(), onFrame: vi.fn(),
  };
  const view = render(<VisionCanvas {...props} />);
  await act(async () => {}); // Complete the real camera hook with a controlled stream.
  const video = view.container.querySelector('video')!;
  const canvas = view.container.querySelector('canvas')!;
  Object.defineProperties(video, { videoWidth: { configurable: true, value: 640 }, videoHeight: { configurable: true, value: 480 }, readyState: { configurable: true, value: 2 } });
  vi.spyOn(video, 'getBoundingClientRect').mockReturnValue({ left: 0, top: 0, width: 640, height: 480 } as DOMRect);
  const tap = () => act(() => { fireEvent.click(canvas, { clientX: 320, clientY: 240 }); });
  return { ...view, engine, session, task, props, tap };
}
beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'performance'] });
  raf = new Map(); nextId = 0; drawImage = vi.fn();
  vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation(cb => { raf.set(++nextId, cb); return nextId; });
  vi.spyOn(globalThis, 'cancelAnimationFrame').mockImplementation(id => { raf.delete(id); });
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (this: HTMLCanvasElement) { return context(this); });
  Object.defineProperty(HTMLVideoElement.prototype, 'getVideoPlaybackQuality', { configurable: true, value: function (this: HTMLVideoElement) { return { totalVideoFrames: Math.floor(this.currentTime * 1000000) + 1 }; } });
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: vi.fn(async () => ({ getTracks: () => [Object.assign(new EventTarget(), { stop: vi.fn() })] })) } });
  vi.spyOn(speechService, 'speak').mockReturnValue(true);
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); Reflect.deleteProperty(HTMLVideoElement.prototype, 'getVideoPlaybackQuality'); Reflect.deleteProperty(navigator, 'mediaDevices'); });

it.each([false, true])('F5 taps the newly drawn accepted semantic frame before HUD refresh (empty prominent list=%s)', async weak => {
  const f = await fixture(weak);
  const find = vi.spyOn(f.engine, 'findObjectAtPoint');
  await runFrame();
  act(() => { vi.advanceTimersByTime(250); });
  const hudA = f.props.onFrame.mock.lastCall![0] as DetectionFrame;
  expect(hudA.objects).toHaveLength(weak ? 0 : 1);
  f.tap(); expect(f.props.onSelectObject).toHaveBeenLastCalledWith(expect.objectContaining({ label: 'wall' }));
  f.props.onSelectObject.mockClear(); vi.mocked(speechService.speak).mockClear();
  act(() => { vi.advanceTimersByTime(20); });
  f.session.run.mockResolvedValueOnce(logits(12, weak));
  drawImage.mockClear();
  await runFrame();
  expect(f.props.onFrame.mock.lastCall![0]).toBe(hudA); // HUD intentionally still A.
  expect(drawImage.mock.calls.some(([source]) => source instanceof HTMLCanvasElement && source.width === 10 && pixels.has(source))).toBe(true);
  f.tap();
  expect(f.props.onSelectObject).toHaveBeenCalledTimes(1);
  expect(f.props.onSelectObject).toHaveBeenLastCalledWith(expect.objectContaining({ label: 'person', confidenceKind: 'semantic-softmax-pixel' }));
  expect(find.mock.lastCall![2]).not.toBe(hudA.objects);
  expect(speechService.speak).toHaveBeenCalledTimes(1);
  const selected = f.props.onSelectObject.mock.lastCall![0];
  expect(selected.confidence).toBeCloseTo(1 / (1 + 149 * Math.exp(weak ? -.1 : -10)));
  f.engine.dispose();
});

it('F5 clear during delayed inference invalidates inspection and rejects its late result', async () => {
  const f = await fixture(); await runFrame();
  act(() => { vi.advanceTimersByTime(250); }); f.tap();
  f.props.onSelectObject.mockClear(); vi.mocked(speechService.speak).mockClear();
  const pending = deferred<ReturnType<typeof logits>>(); f.session.run.mockImplementationOnce(() => pending.promise);
  step(true); await act(async () => { vi.advanceTimersByTime(0); });
  f.tap(); expect(f.props.onSelectObject).toHaveBeenLastCalledWith(expect.objectContaining({ label: 'wall' }));
  f.props.onSelectObject.mockClear(); vi.mocked(speechService.speak).mockClear();
  act(() => { f.engine.clearDetections(); }); f.tap();
  expect(f.props.onSelectObject).not.toHaveBeenCalled();
  await act(async () => { pending.resolve(logits(12)); }); step(); f.tap();
  expect(f.props.onSelectObject).not.toHaveBeenCalled(); expect(speechService.speak).not.toHaveBeenCalled(); expect(f.task.getPixelClass(.5, .5)).toBeNull();
  await runFrame(); f.tap(); expect(f.props.onSelectObject).toHaveBeenLastCalledWith(expect.objectContaining({ label: 'wall' }));
  f.engine.dispose();
});

it('F5 requires an accepted result to be copied before inspecting it', async () => {
  const f = await fixture(); await runFrame(); act(() => { vi.advanceTimersByTime(250); });
  f.session.run.mockResolvedValueOnce(logits(12));
  step(true); await act(async () => { vi.advanceTimersByTime(0); });
  // B is accepted, but the visible Canvas still contains A. Do not inspect B
  // prematurely or resurrect A using its older exact array.
  f.tap(); expect(f.props.onSelectObject).not.toHaveBeenCalled(); expect(speechService.speak).not.toHaveBeenCalled();
  step(); f.tap(); expect(f.props.onSelectObject).toHaveBeenLastCalledWith(expect.objectContaining({ label: 'person' }));
  f.engine.dispose();
});

it('F5 model switch during delayed semantic inference cannot inspect its late completion', async () => {
  const f = await fixture(); await runFrame(); act(() => { vi.advanceTimersByTime(250); }); f.tap();
  f.props.onSelectObject.mockClear(); vi.mocked(speechService.speak).mockClear();
  const pending = deferred<ReturnType<typeof logits>>(); f.session.run.mockImplementationOnce(() => pending.promise);
  step(true); await act(async () => { vi.advanceTimersByTime(0); });
  await act(async () => {
    const switching = f.engine.initialize('mediapipe');
    f.tap(); // Engine already invalidated its model, before React receives props.
    await switching;
  });
  expect(f.props.onSelectObject).not.toHaveBeenCalled(); expect(speechService.speak).not.toHaveBeenCalled();
  f.rerender(<VisionCanvas {...f.props} activeModel="mediapipe" modelState={f.engine.getState()} />);
  await act(async () => { pending.resolve(logits(12)); }); step(); f.tap();
  expect(f.props.onSelectObject).not.toHaveBeenCalled(); expect(speechService.speak).not.toHaveBeenCalled(); expect(f.task.getPixelClass(.5, .5)).toBeNull();
  expect(f.engine.getState()).toMatchObject({ status: 'ready', model: 'mediapipe' }); expect(f.session.release).toHaveBeenCalledTimes(1);
  f.engine.dispose();
});
