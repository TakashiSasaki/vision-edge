import { beforeEach, expect, it, vi } from 'vitest';
import { VisionEngine, type VisionTaskFactory, type SegmenterTask } from './visionEngine';
import { backendDescription } from './modelRegistry';
import type { RunnableModelType } from '../types/vision';
import { FilesetResolver, ObjectDetector, ImageSegmenter, type MPMask } from '@mediapipe/tasks-vision';

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function fixture() {
  const detect = vi.fn(() => ({ detections: [{
    categories: [{ categoryName: 'dog', displayName: '', score: 0.9, index: 0 }],
    boundingBox: { originX: 64, originY: 48, width: 128, height: 96, angle: 0 }, keypoints: [],
  }] }));
  const detector = { detectForVideo: detect, close: vi.fn() };
  const result = { confidenceMasks: [], categoryMask: undefined, close: vi.fn() };
  const segment = vi.fn(() => result);
  const segmenter: SegmenterTask = { segmentForVideo: segment, close: vi.fn() };
  const factory: VisionTaskFactory = {
    load: vi.fn(async () => ({ wasmLoaderPath: 'loader', wasmBinaryPath: 'wasm' })),
    detector: vi.fn(async () => detector),
    segmenter: vi.fn(async () => segmenter),
  };
  return { engine: new VisionEngine(factory), factory, detector, segmenter, detect, segment, result };
}
function video() {
  return Object.defineProperties(document.createElement('video'), {
    videoWidth: { value: 640 }, videoHeight: { value: 480 }, readyState: { value: 2 },
  });
}
beforeEach(() => { vi.spyOn(console, 'warn').mockImplementation(() => {}); });

it.each(['unknown', 'constructor', 'invalid-model'])('rejects unsupported runtime ID %s without creating or running another model', id => {
  const f = fixture();
  expect(() => f.engine.initialize(id as RunnableModelType)).toThrow('Unsupported model ID');
  expect(() => f.engine.detectVideoFrame(video(), id as RunnableModelType, 0.3)).toThrow('Unsupported model ID');
  expect(f.factory.load).not.toHaveBeenCalled();
  expect(f.detect).not.toHaveBeenCalled();
});

it('initializes segformer through factory and runs asynchronous inference with full semantic mask and pixel inspection', async () => {
  const f = fixture();
  const mockMaskCanvas = document.createElement('canvas');
  const mockDetect = vi.fn(async () => [
    {
      id: 'segformer-0',
      label: 'wall',
      labelJa: '壁',
      category: '建築',
      confidence: 0.95,
      bbox: { x: 0, y: 0, width: 1, height: 1 },
      color: '#94a3b8',
      maskCanvas: mockMaskCanvas,
    }
  ]);
  const mockGetPixelClass = vi.fn(() => ({ id: 0, en: 'wall', ja: '壁', category: '建築', color: '#94a3b8' }));
  const segformerTask = { detectForVideo: mockDetect, getPixelClass: mockGetPixelClass, close: vi.fn() };
  f.factory.segformer = vi.fn(async () => segformerTask);

  const initResult = await f.engine.initialize('segformer');
  expect(initResult.status).toBe('ready');
  expect(f.factory.segformer).toHaveBeenCalledTimes(1);
  expect(f.factory.load).not.toHaveBeenCalled();

  const v = video();
  const frameResult = await f.engine.detectVideoFrame(v, 'segformer', 0.3);
  expect(frameResult.status).toBe('ready');
  if (frameResult.status === 'ready') {
    expect(frameResult.objects).toHaveLength(1);
    expect(frameResult.objects[0].label).toBe('wall');
    expect(frameResult.objects[0].maskCanvas).toBe(mockMaskCanvas);
  }

  const clickedObj = f.engine.findObjectAtPoint(0.5, 0.5, frameResult.status === 'ready' ? frameResult.objects : []);
  expect(clickedObj).toMatchObject({ id: 'segformer-0', label: 'wall', labelJa: '壁' });
  expect(mockGetPixelClass).toHaveBeenCalledWith(0.5, 0.5);

  f.engine.dispose();
  expect(segformerTask.close).toHaveBeenCalledTimes(1);
});

it('initializes yolo-seg through factory and runs asynchronous inference with maskCanvas', async () => {
  const f = fixture();
  const mockDetect = vi.fn(async () => [
    {
      id: 'yolo-test-1',
      label: 'cat',
      labelJa: '猫',
      category: '動物',
      confidence: 0.92,
      bbox: { x: 0.2, y: 0.2, width: 0.4, height: 0.4 },
      color: '#fbbf24',
      maskCanvas: document.createElement('canvas'),
    }
  ]);
  const yoloTask = { detectForVideo: mockDetect, close: vi.fn() };
  f.factory.yoloSeg = vi.fn(async () => yoloTask);

  const initResult = await f.engine.initialize('yolo-seg');
  expect(initResult.status).toBe('ready');
  expect(f.factory.yoloSeg).toHaveBeenCalledTimes(1);
  expect(f.factory.load).not.toHaveBeenCalled();

  const v = video();
  const frameResult = await f.engine.detectVideoFrame(v, 'yolo-seg', 0.3);
  expect(frameResult.status).toBe('ready');
  if (frameResult.status === 'ready') {
    expect(frameResult.objects).toHaveLength(1);
    expect(frameResult.objects[0].label).toBe('cat');
    expect(frameResult.objects[0].maskCanvas).toBeDefined();
  }

  f.engine.dispose();
  expect(yoloTask.close).toHaveBeenCalledTimes(1);
});

it('shares the in-progress Promise and creates only the selected task; ready reaches actual detector dispatch without polygons', async () => {
  const f = fixture();
  const task = deferred<typeof f.detector>();
  vi.mocked(f.factory.detector).mockReturnValue(task.promise);
  const first = f.engine.initialize('mediapipe');
  const second = f.engine.initialize('mediapipe');
  expect(second).toBe(first);
  expect(f.engine.getState().status).toBe('loading');
  expect(backendDescription(f.engine.getState())).toBe('バックエンド未準備');
  expect(f.engine.detectVideoFrame(video(), 'mediapipe', 0.3).status).toBe('not-ready');
  expect(f.detect).not.toHaveBeenCalled();
  task.resolve(f.detector);
  expect((await first).status).toBe('ready');
  expect(f.factory.detector).toHaveBeenCalledWith(expect.anything(), 'GPU', expect.anything());
  expect(f.factory.detector).toHaveBeenCalledTimes(1);
  expect(f.factory.segmenter).not.toHaveBeenCalled();
  const frame = f.engine.detectVideoFrame(video(), 'mediapipe', 0.3);
  expect(frame.status).toBe('ready');
  if (frame.status !== 'ready') throw new Error('Expected actual inference result');
  expect(f.detect).toHaveBeenCalledTimes(1);
  expect(frame.objects[0]).toMatchObject({ label: 'dog', bbox: { x: 0.1, y: 0.1, width: 0.2, height: 0.2 } });
  expect(frame.objects[0].maskPolygon).toBeUndefined();
  expect(backendDescription(f.engine.getState())).toContain('GPU実動作は未計測');
  await f.engine.initialize('mediapipe');
  expect(f.factory.detector).toHaveBeenCalledTimes(1);
});

it('reaches the real segmenter path only after ready and closes its inference result', async () => {
  const f = fixture();
  expect(f.engine.detectVideoFrame(video(), 'selfie-segmenter', 0.3).status).toBe('not-ready');
  expect(f.segment).not.toHaveBeenCalled();
  await f.engine.initialize('selfie-segmenter');
  expect(f.factory.detector).not.toHaveBeenCalled();
  expect(f.factory.segmenter).toHaveBeenCalledWith(expect.anything(), 'GPU', expect.anything());
  expect(f.engine.detectVideoFrame(video(), 'selfie-segmenter', 0.3)).toMatchObject({ status: 'ready', objects: [] });
  expect(f.segment).toHaveBeenCalledTimes(1);
  expect(f.result.close).toHaveBeenCalledTimes(1);
});

it('distinguishes WASM load failure, keeps its cause, and explicitly retries without retaining a failed Promise', async () => {
  const f = fixture();
  const cause = new Error('WASM unavailable');
  vi.mocked(f.factory.load).mockRejectedValueOnce(cause);
  const failed = f.engine.initialize('mediapipe');
  expect(await failed).toMatchObject({ status: 'error', stage: 'wasm', cause });
  expect(f.factory.detector).not.toHaveBeenCalled();
  expect(f.engine.detectVideoFrame(video(), 'mediapipe', 0.3).status).toBe('not-ready');
  expect(f.detect).not.toHaveBeenCalled();
  expect(backendDescription(f.engine.getState())).toBe('バックエンド未準備');
  const retry = f.engine.initialize('mediapipe');
  expect(retry).not.toBe(failed);
  expect(f.engine.initialize('mediapipe')).toBe(retry);
  expect((await retry).status).toBe('ready');
  expect(f.factory.load).toHaveBeenCalledTimes(2);
});

it.each(['mediapipe', 'selfie-segmenter'] as const)('reports selected %s task failure as error and recovers on explicit retry', async model => {
  const f = fixture();
  const cause = new Error('task unavailable');
  if (model === 'mediapipe') vi.mocked(f.factory.detector).mockRejectedValueOnce(cause);
  else vi.mocked(f.factory.segmenter).mockRejectedValueOnce(cause);
  expect(await f.engine.initialize(model)).toMatchObject({ status: 'error', stage: 'task', cause });
  expect(f.engine.detectVideoFrame(video(), model, 0.3).status).toBe('not-ready');
  expect((await f.engine.initialize(model)).status).toBe('ready');
});

it('rejects a null task at the factory boundary', async () => {
  const f = fixture();
  vi.mocked(f.factory.segmenter).mockResolvedValueOnce(null as unknown as SegmenterTask);
  expect(await f.engine.initialize('selfie-segmenter')).toMatchObject({ status: 'error', stage: 'task' });
});

it('keeps B ready when delayed A finishes or fails; closes only obsolete A', async () => {
  const f = fixture();
  const old = deferred<SegmenterTask>();
  vi.mocked(f.factory.segmenter).mockReturnValue(old.promise);
  const a = f.engine.initialize('selfie-segmenter');
  await Promise.resolve(); // The resolved fileset reaches the controlled task factory.
  expect(f.factory.segmenter).toHaveBeenCalledTimes(1);
  await f.engine.initialize('mediapipe');
  const b = f.engine.getState();
  old.resolve(f.segmenter);
  expect((await a).status).toBe('cancelled');
  expect(f.engine.getState()).toBe(b);
  expect(f.segmenter.close).toHaveBeenCalledTimes(1);
  expect(f.detector.close).not.toHaveBeenCalled();
  expect(f.engine.detectVideoFrame(video(), 'mediapipe', 0.3).status).toBe('ready');
});

it('ignores an obsolete rejected initialization without replacing the current ready state', async () => {
  const f = fixture();
  const old = deferred<SegmenterTask>();
  vi.mocked(f.factory.segmenter).mockReturnValue(old.promise);
  const a = f.engine.initialize('selfie-segmenter');
  await Promise.resolve();
  await f.engine.initialize('mediapipe');
  const ready = f.engine.getState();
  old.reject(new Error('late obsolete failure'));
  expect((await a).status).toBe('cancelled');
  expect(f.engine.getState()).toBe(ready);
});

it('handles A/B/A out of order without closing the newer A task', async () => {
  const f = fixture();
  const old = deferred<SegmenterTask>();
  const next = { segmentForVideo: f.segmenter.segmentForVideo, close: vi.fn() };
  vi.mocked(f.factory.segmenter).mockReturnValueOnce(old.promise).mockResolvedValueOnce(next);
  const a = f.engine.initialize('selfie-segmenter');
  await Promise.resolve();
  await f.engine.initialize('mediapipe');
  await f.engine.initialize('selfie-segmenter');
  const ready = f.engine.getState();
  old.resolve(f.segmenter);
  await a;
  expect(f.engine.getState()).toBe(ready);
  expect(f.detector.close).toHaveBeenCalledTimes(1);
  expect(f.segmenter.close).toHaveBeenCalledTimes(1);
  expect(next.close).not.toHaveBeenCalled();
  f.engine.dispose();
  f.engine.dispose();
  expect(next.close).toHaveBeenCalledTimes(1);
});

it('closes a task completed after disposal and can initialize after a StrictMode-style cleanup', async () => {
  const f = fixture();
  const old = deferred<SegmenterTask>();
  vi.mocked(f.factory.segmenter).mockReturnValueOnce(old.promise);
  const pending = f.engine.initialize('selfie-segmenter');
  await Promise.resolve();
  f.engine.dispose();
  expect(f.engine.getState().status).toBe('idle');
  old.resolve(f.segmenter);
  expect((await pending).status).toBe('cancelled');
  expect(f.segmenter.close).toHaveBeenCalledTimes(1);
  expect((await f.engine.initialize('mediapipe')).status).toBe('ready');
});

it('does not create an obsolete task if disposal occurs during WASM loading', async () => {
  const f = fixture();
  const load = deferred<Awaited<ReturnType<VisionTaskFactory['load']>>>();
  vi.mocked(f.factory.load).mockReturnValue(load.promise);
  const pending = f.engine.initialize('mediapipe');
  f.engine.dispose();
  load.resolve({ wasmLoaderPath: 'loader', wasmBinaryPath: 'wasm' });
  expect((await pending).status).toBe('cancelled');
  expect(f.factory.detector).not.toHaveBeenCalled();
});

it('clears the old model detections and requires the matching ready model before dispatch', async () => {
  const f = fixture();
  await f.engine.initialize('mediapipe');
  expect(f.engine.detectVideoFrame(video(), 'mediapipe', 0.3)).toMatchObject({ objects: [{ label: 'dog' }] });
  await f.engine.initialize('selfie-segmenter');
  expect(f.detector.close).toHaveBeenCalledTimes(1);
  expect(f.engine.detectVideoFrame(video(), 'mediapipe', 0.3).status).toBe('not-ready');
  expect(f.detect).toHaveBeenCalledTimes(1);
  expect(f.engine.detectVideoFrame(video(), 'selfie-segmenter', 0.3)).toMatchObject({ status: 'ready', objects: [] });
});

it('does not convert an inference exception into successful empty detection', async () => {
  const f = fixture();
  await f.engine.initialize('mediapipe');
  f.detect.mockImplementation(() => { throw new Error('inference failure'); });
  expect(() => f.engine.detectVideoFrame(video(), 'mediapipe', 0.3)).toThrow('inference failure');
});

it('preserves pixel-mask processing and the enclosing person box', async () => {
  const f = fixture();
  const putImageData = vi.fn();
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    createImageData: (width: number, height: number) => ({ data: new Uint8ClampedArray(width * height * 4) }),
    putImageData,
  } as unknown as CanvasRenderingContext2D);
  const mask = { width: 2, height: 2, getAsFloat32Array: () => new Float32Array([0, 0.9, 0.9, 0.9]) } as MPMask;
  const close = vi.fn();
  const task: SegmenterTask = { segmentForVideo: vi.fn(() => ({ confidenceMasks: [mask], close })), close: vi.fn() };
  vi.mocked(f.factory.segmenter).mockResolvedValue(task);
  await f.engine.initialize('selfie-segmenter');
  const frame = f.engine.detectVideoFrame(video(), 'selfie-segmenter', 0.35);
  if (frame.status !== 'ready') throw new Error('Expected prepared model');
  expect(frame.objects[0]).toMatchObject({ label: 'person', bbox: { x: 0, y: 0, width: 0.5, height: 0.5 } });
  expect(frame.objects[0].maskCanvas).toBeInstanceOf(HTMLCanvasElement);
  expect(frame.objects[0].maskCanvas?.width).toBe(2);
  expect(frame.objects[0].maskPolygon).toBeUndefined();
  expect(putImageData).toHaveBeenCalledTimes(1);
  expect(close).toHaveBeenCalledTimes(1);
});

it('keeps model ready distinct from an unready video', async () => {
  const f = fixture();
  await f.engine.initialize('mediapipe');
  expect(f.engine.detectVideoFrame(document.createElement('video'), 'mediapipe', 0.3).status).toBe('not-ready');
  expect(f.engine.getState().status).toBe('ready');
  expect(f.detect).not.toHaveBeenCalled();
});

it.each(['mediapipe', 'selfie-segmenter'] as const)('passes the reported delegate to the actual SDK creation boundary for %s', async model => {
  const f = fixture();
  vi.spyOn(FilesetResolver, 'forVisionTasks').mockResolvedValue({ wasmLoaderPath: 'loader', wasmBinaryPath: 'wasm' });
  const detector = vi.spyOn(ObjectDetector, 'createFromOptions').mockResolvedValue(f.detector as unknown as ObjectDetector);
  const segmenter = vi.spyOn(ImageSegmenter, 'createFromOptions').mockResolvedValue(f.segmenter as unknown as ImageSegmenter);
  const engine = new VisionEngine();
  await engine.initialize(model);
  expect(model === 'mediapipe' ? detector : segmenter).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({
    baseOptions: expect.objectContaining({ delegate: 'GPU' }), runningMode: 'VIDEO',
  }));
  expect(model === 'mediapipe' ? segmenter : detector).not.toHaveBeenCalled();
  expect(engine.getState()).toMatchObject({ status: 'ready', delegate: 'GPU' });
  engine.dispose();
});

it('repairs temporary smoothing state after an SDK exception so empty recovery cannot resurrect old objects', async () => {
  const f = fixture(); await f.engine.initialize('mediapipe'); const v = video();
  expect(f.engine.detectVideoFrame(v, 'mediapipe', 0.3)).toMatchObject({ status: 'ready', objects: [expect.objectContaining({ label: 'dog' })] });
  f.detect.mockImplementationOnce(() => { throw new Error('SDK failure'); });
  expect(() => f.engine.detectVideoFrame(v, 'mediapipe', 0.3)).toThrow('SDK failure');
  f.detect.mockReturnValue({ detections: [] });
  expect(f.engine.detectVideoFrame(v, 'mediapipe', 0.3)).toMatchObject({ status: 'ready', objects: [] });
  expect(f.factory.detector).toHaveBeenCalledTimes(1); expect(f.detector.close).not.toHaveBeenCalled();
});
it('does not refill smoothing state if the scene is cleared reentrantly during SDK execution', async () => {
  const f = fixture(); await f.engine.initialize('mediapipe'); const v = video();
  const detect = f.detect.getMockImplementation()!;
  f.detect.mockImplementationOnce(() => { f.engine.clearDetections(); return detect(); });
  expect(f.engine.detectVideoFrame(v, 'mediapipe', 0.3)).toMatchObject({ status: 'not-ready' });
  f.detect.mockReturnValue({ detections: [] });
  expect(f.engine.detectVideoFrame(v, 'mediapipe', 0.3)).toMatchObject({ status: 'ready', objects: [] });
  expect(f.engine.getState()).toMatchObject({ status: 'ready', requestId: 1 });
});
it('routes an immutable VideoFrame input to the correct SDK task using its intrinsic dimensions without closing its caller-owned handle', async () => {
  const f = fixture(); const close = vi.fn();
  const frame = { displayWidth: 640, displayHeight: 480, timestamp: 0, close } as unknown as VideoFrame;
  await f.engine.initialize('mediapipe');
  expect(f.engine.detectVideoFrame(frame, 'mediapipe', 0.3)).toMatchObject({ status: 'ready', objects: [expect.objectContaining({ bbox: { x: 0.1, y: 0.1, width: 0.2, height: 0.2 } })] });
  expect(f.detect).toHaveBeenCalledWith(frame, expect.any(Number));
  await f.engine.initialize('selfie-segmenter');
  expect(f.engine.detectVideoFrame(frame, 'selfie-segmenter', 0.3)).toMatchObject({ status: 'ready', objects: [] });
  expect(f.segment).toHaveBeenCalledWith(frame, expect.any(Number)); expect(close).not.toHaveBeenCalled();
});
it('uses a caller-owned compatibility canvas as the actual SDK image with its own dimensions', async () => {
  const f = fixture(); const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 480;
  await f.engine.initialize('mediapipe');
  expect(f.engine.detectVideoFrame(canvas, 'mediapipe', 0.3)).toMatchObject({ status: 'ready', objects: [expect.objectContaining({ bbox: { x: 0.1, y: 0.1, width: 0.2, height: 0.2 } })] });
  expect(f.detect).toHaveBeenCalledWith(canvas, expect.any(Number)); expect(canvas.width).toBe(640);
  await f.engine.initialize('selfie-segmenter');
  expect(f.engine.detectVideoFrame(canvas, 'selfie-segmenter', 0.3)).toMatchObject({ status: 'ready', objects: [] });
  expect(f.segment).toHaveBeenCalledWith(canvas, expect.any(Number)); expect(canvas.width).toBe(640);
});

