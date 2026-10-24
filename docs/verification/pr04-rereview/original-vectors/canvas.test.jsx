import React, { StrictMode } from 'react';
import { flushSync } from 'react-dom';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { VisionCanvas } from '../../../../src/components/VisionCanvas';
import { VisionEngine } from '../../../../src/services/visionEngine';
import { speechService } from '../../../../src/services/speechService';
let callbacks, serial, contexts, now, tracks;
beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
  callbacks = new Map(); serial = 0; contexts = new Map(); tracks = []; now = 0;
  vi.spyOn(performance, 'now').mockImplementation(() => now);
  vi.stubGlobal('requestAnimationFrame', cb => { callbacks.set(++serial, cb); return serial; });
  vi.stubGlobal('cancelAnimationFrame', id => callbacks.delete(id));
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function() {
    if (!contexts.has(this)) contexts.set(this, Object.fromEntries(
      ['clearRect','drawImage','save','restore','beginPath','roundRect','stroke','fill','fillText','arc'].map(name => [name, vi.fn()])
    ));
    const ctx = contexts.get(this); ctx.measureText = () => ({ width: 10 }); return ctx;
  });
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: vi.fn(async () => {
    const track = Object.assign(new EventTarget(), { stop: vi.fn() }); tracks.push(track);
    return { getTracks: () => [track] };
  }) } });
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); Reflect.deleteProperty(navigator, 'mediaDevices'); });
function drawTick() { act(() => { for (const [id, cb] of [...callbacks]) { callbacks.delete(id); cb(now); } }); }
function flush(ms = 0) { act(() => { vi.advanceTimersByTime(ms); }); }
async function mount({ strict = false, segmenter = false, speech = false } = {}) {
  const detect = vi.fn(() => ({ detections: [{
    categories: [{ categoryName: 'dog', score: 0.9 }],
    boundingBox: { originX: 0, originY: 0, width: 32, height: 32 },
  }] }));
  const closeResult = vi.fn();
  const segment = vi.fn(() => ({ confidenceMasks: [], close: closeResult }));
  const detector = { detectForVideo: detect, close: vi.fn() }, segmentTask = { segmentForVideo: segment, close: vi.fn() };
  const factory = { load: vi.fn(async () => ({})), detector: vi.fn(async () => detector), segmenter: vi.fn(async () => segmentTask) };
  const engine = new VisionEngine(factory);
  const model = segmenter ? 'selfie-segmenter' : 'mediapipe';
  await engine.initialize(model);
  const props = {
    activeModel: model, engine, modelState: engine.getState(), facingMode: 'user', isCameraActive: true,
    onToggleFacingMode: vi.fn(), onCameraStart: vi.fn(), onCameraStop: vi.fn(), onSelectObject: vi.fn(), onFrame: vi.fn(),
    renderOptions: { showBoxes: true, showMasks: true, showLabels: true, showConfidence: true, confidenceThreshold: 0.35, maskOpacity: 0.3 },
    ttsConfig: { enabled: speech, autoReadout: speech, cooldownSec: 3, rate: 1, pitch: 1, volume: 1, voiceURI: '' },
  };
  const element = () => strict ? <StrictMode><VisionCanvas {...props} /></StrictMode> : <VisionCanvas {...props} />;
  const view = render(element()); await act(async () => {});
  const video = view.container.querySelector('video'), canvas = view.container.querySelector('canvas');
  Object.defineProperties(video, { readyState: { value: 2, configurable: true }, videoWidth: { value: 64 }, videoHeight: { value: 64 } });
  return { ...view, props, engine, detect, segment, detector, segmentTask, factory, closeResult, video, canvas, element };
}
it.each(['offscreen-draw', 'visible-copy', 'missing-context', 'notification', 'tts-gate', 'tts-speak'])('stage exception reaches explicit stop and recovers: %s', async stage => {
  const gate = vi.spyOn(speechService, 'canSpeakLabel').mockReturnValue(true);
  const speak = vi.spyOn(speechService, 'speak').mockReturnValue(true);
  const f = await mount({ speech: stage.startsWith('tts') });
  const visible = contexts.get(f.canvas);
  if (stage === 'visible-copy') visible.drawImage.mockImplementationOnce(() => { throw Error('copy'); });
  if (stage === 'missing-context') vi.spyOn(f.canvas, 'getContext').mockReturnValueOnce(null);
  if (stage === 'notification') f.props.onFrame.mockImplementationOnce(() => { throw Error('parent'); });
  if (stage === 'tts-gate') gate.mockImplementationOnce(() => { throw Error('gate'); });
  if (stage === 'tts-speak') speak.mockImplementationOnce(() => { throw Error('speak'); });
  drawTick(); flush();
  if (stage === 'offscreen-draw') {
    // First successful draw allocates its buffer. Throw inside the second draw.
    drawTick();
    const offscreen = [...contexts.keys()].find(canvas => canvas !== f.canvas);
    expect(offscreen).toBeDefined();
    contexts.get(offscreen).roundRect.mockImplementationOnce(() => { throw Error('overlay'); });
  }
  drawTick(); flush(250);
  expect(screen.getByRole('alert').textContent).toContain('処理を停止しました');
  expect(screen.getByLabelText('描画FPS').textContent).toContain('未計測');
  const calls = f.detect.mock.calls.length;
  for (let i = 0; i < 5; i++) { f.video.currentTime++; drawTick(); flush(250); }
  expect(f.detect).toHaveBeenCalledTimes(calls); expect(callbacks.size).toBe(0);
  fireEvent.click(screen.getByRole('button', { name: '処理を再試行' }));
  drawTick(); flush(); drawTick(); flush(250);
  expect(screen.queryByRole('alert')).toBeNull(); expect(f.detect.mock.calls.length).toBeGreaterThan(calls);
  expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1);
  expect(f.factory.detector).toHaveBeenCalledTimes(1);
});
it('duration includes separately injected SDK and post-processing work, and draw callbacks before results count nothing', async () => {
  const f = await mount();
  drawTick(); // No SDK timer executed and no actual overlay copied.
  now = 500; flush(250);
  expect(screen.getByLabelText('描画FPS').textContent).toContain('未計測');
  f.detect.mockImplementation(() => {
    now += 5;
    return { detections: [{ categories: [{ get categoryName() { now += 11; return 'dog'; }, score: 0.9 }],
      boundingBox: { originX: 0, originY: 0, width: 32, height: 32 } }] };
  });
  // First queued execution already ran at flush; use a fresh actual candidate.
  f.video.currentTime = 1; drawTick(); flush();
  now = 1000; flush(250);
  // 5 SDK + 11 post-processing = 16. Zero-duration successes have no duration sample.
  expect(screen.getByLabelText('フレーム処理時間').textContent).toContain('16.0 ms');
});
it('segmenter duration includes result cleanup, not just SDK timestamp or call', async () => {
  const f = await mount({ segmenter: true });
  f.segment.mockImplementation(() => { now += 5; return { confidenceMasks: [], close() { now += 13; } }; });
  drawTick(); flush(); now = 250; flush(250);
  expect(screen.getByLabelText('フレーム処理時間').textContent).toContain('18.0 ms');
});
it.each(['model', 'camera', 'threshold', 'unmount'])('stale SDK reentry cannot publish metrics, parent result or speech: %s', async kind => {
  const gate = vi.spyOn(speechService, 'canSpeakLabel').mockReturnValue(true);
  vi.spyOn(speechService, 'speak').mockReturnValue(true);
  const f = await mount({ speech: true });
  const result = f.detect.getMockImplementation();
  f.detect.mockImplementationOnce(() => {
    now += 20;
    if (kind === 'model') f.engine.dispose();
    if (kind === 'unmount') f.unmount();
    if (kind === 'camera') flushSync(() => { f.props.isCameraActive = false; f.rerender(f.element()); });
    if (kind === 'threshold') flushSync(() => { f.props.renderOptions = { ...f.props.renderOptions, confidenceThreshold: 0.99 }; f.rerender(f.element()); });
    return result();
  });
  drawTick(); flush(); now = 250; flush(250);
  expect(f.props.onFrame.mock.calls.filter(([frame]) => frame !== null)).toEqual([]);
  expect(gate).not.toHaveBeenCalled();
  if (kind !== 'unmount') expect(screen.getByLabelText('推論FPS').textContent).toContain('未計測');
});
it('processing failure repairs smoothing; fixed timestamp never retried; empty recovery counts success', async () => {
  const f = await mount();
  drawTick(); flush(); now = 250; flush(250);
  expect(f.props.onFrame).toHaveBeenLastCalledWith(expect.objectContaining({ objects: [expect.objectContaining({ label: 'dog' })] }));
  f.detect.mockReturnValueOnce({ get detections() { throw Error('postprocess'); } });
  f.video.currentTime = 1; drawTick(); flush(); now = 500; flush(250);
  expect(f.props.onFrame).toHaveBeenLastCalledWith(null);
  expect(screen.getByLabelText('性能状態').textContent).toContain('推論エラー');
  expect(screen.getByLabelText('推論FPS').textContent).toContain('未計測');
  const calls = f.detect.mock.calls.length;
  for (let i = 0; i < 4; i++) { drawTick(); flush(); }
  expect(f.detect).toHaveBeenCalledTimes(calls);
  f.detect.mockImplementation(() => { now += 4; return { detections: [] }; });
  f.video.currentTime = 2; drawTick(); flush(); now = 750; flush(250);
  expect(f.props.onFrame).toHaveBeenLastCalledWith(expect.objectContaining({ objects: [] }));
  expect(screen.getByLabelText('フレーム処理時間').textContent).toContain('4.0 ms');
  expect(screen.getByLabelText('推論FPS').textContent).toContain('2.0 FPS');
  expect(f.engine.getState().requestId).toBe(1); expect(f.detector.close).not.toHaveBeenCalled();
});
it('ready model refusing a reentrantly invalidated result creates no success or draw sample', async () => {
  const f = await mount(); const result = f.detect.getMockImplementation();
  f.detect.mockImplementationOnce(() => { now += 8; f.engine.clearDetections(); return result(); });
  drawTick(); flush(); drawTick(); now = 250; flush(250);
  expect(f.props.onFrame.mock.calls.filter(([frame]) => frame !== null)).toEqual([]);
  expect(screen.getByLabelText('推論FPS').textContent).toContain('未計測');
  expect(screen.getByLabelText('描画FPS').textContent).toContain('未計測');
  expect(screen.getByLabelText('フレーム処理時間').textContent).toContain('未計測');
  f.detect.mockImplementation(() => { now += 3; return { detections: [] }; });
  f.video.currentTime = 1; drawTick(); flush(); now = 500; flush(250);
  expect(f.props.onFrame).toHaveBeenLastCalledWith(expect.objectContaining({ objects: [] }));
  expect(screen.getByLabelText('フレーム処理時間').textContent).toContain('3.0 ms');
  expect(f.engine.getState().requestId).toBe(1);
});
it('StrictMode releases streams and all queued execution on unmount', async () => {
  const f = await mount({ strict: true }); drawTick(); f.unmount(); flush(1000);
  expect(callbacks.size).toBe(0); expect(f.detect).not.toHaveBeenCalled();
  expect(tracks.every(track => track.stop.mock.calls.length === 1)).toBe(true);
});
it('F01 production bridge: repeated live image reaches real engine and repeats TTS decisions', async () => {
  const gate = vi.spyOn(speechService, 'canSpeakLabel').mockReturnValue(true);
  vi.spyOn(speechService, 'speak').mockReturnValue(true);
  const f = await mount({ speech: true }); const images = [];
  f.detect.mockImplementation(video => { images.push(video.currentTime); now += 1; return { detections: [{
    categories: [{ categoryName: 'dog', score: 0.9 }], boundingBox: { originX: 0, originY: 0, width: 32, height: 32 },
  }] }; });
  f.video.currentTime = 1; drawTick();
  // The actual SDK input is still the live video, not the observed candidate.
  f.video.currentTime = 2; flush();
  now = 500; drawTick(); flush(); flush(250);
  console.log('F01 real canvas/engine', { images, ttsDecisions: gate.mock.calls.length, fps: screen.getByLabelText('推論FPS').textContent });
  expect(images).toEqual([2]);
  expect(gate).toHaveBeenCalledTimes(1);
});
