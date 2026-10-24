import { beforeEach, expect, it, vi } from 'vitest';
import type { MPMask } from '@mediapipe/tasks-vision';
import { PipelineMetrics, type PipelineObserver } from './pipelineMetrics';
import { VisionEngine, type VisionTaskFactory } from './visionEngine';
function fixture() {
  let now = 0;
  const image = new Uint8ClampedArray(16);
  const context = { createImageData: vi.fn(() => { now += 2; return { data: image }; }), putImageData: vi.fn(() => { now += 4; }) };
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as unknown as CanvasRenderingContext2D);
  const mask = { width: 2, height: 2, getAsFloat32Array: vi.fn(() => { now += 3; return new Float32Array([0.9, 0.9, 0.9, 0.9]); }) } as unknown as MPMask;
  const result = { confidenceMasks: [mask], categoryMask: undefined, close: vi.fn(() => { now += 5; }) };
  const detect = vi.fn(() => { now += 10; return { detections: [] }; });
  const segment = vi.fn(() => { now += 10; return result; });
  const factory: VisionTaskFactory = {
    load: async () => ({ wasmLoaderPath: '', wasmBinaryPath: '' }),
    detector: async () => ({ detectForVideo: detect, close: vi.fn() }),
    segmenter: async () => ({ segmentForVideo: segment, close: vi.fn() }),
  };
  const metrics = new PipelineMetrics(() => now);
  const input = Object.assign(document.createElement('canvas'), { width: 640, height: 480 });
  return { engine: new VisionEngine(factory), metrics, input, detect, segment, result, mask, context, time: () => now };
}
beforeEach(() => { vi.spyOn(performance, 'now').mockReturnValue(123); vi.spyOn(console, 'warn').mockImplementation(() => {}); });
it('separates synchronous SDK, summed exclusive segmentation transforms, cleanup, and inclusive wrapper', async () => {
  const f = fixture(); await f.engine.initialize('selfie-segmenter');
  const result = f.engine.detectVideoFrame(f.input, 'selfie-segmenter', 0.3, f.metrics);
  expect(result).toMatchObject({ status: 'ready', objects: [{ label: 'person' }] });
  const stages = f.metrics.snapshot()!.stages;
  expect(stages.sdk.success).toMatchObject({ count: 1, mean: 10 });
  expect(stages.postprocess.success).toMatchObject({ count: 1, mean: 9 });
  expect(stages.cleanup.success).toMatchObject({ count: 1, mean: 5 });
  expect(stages.engine.success).toMatchObject({ count: 1, mean: 24 });
  expect(f.context.putImageData).toHaveBeenCalledTimes(1); expect(f.result.close).toHaveBeenCalledTimes(1);
  expect(f.segment).toHaveBeenCalledWith(f.input, 123);
});
it('detector timings preserve measured zero transformation work and do not fabricate result-close work', async () => {
  const f = fixture(); await f.engine.initialize('mediapipe');
  expect(f.engine.detectVideoFrame(f.input, 'mediapipe', 0.3, f.metrics)).toMatchObject({ status: 'ready', objects: [] });
  const stages = f.metrics.snapshot()!.stages;
  expect(stages.sdk.success.mean).toBe(10); expect(stages.postprocess.success).toMatchObject({ count: 1, mean: 0 });
  expect(stages.cleanup.success).toMatchObject({ count: 0, mean: null }); expect(stages.engine.success.mean).toBe(10);
  expect(f.detect).toHaveBeenCalledWith(f.input, 123);
});
it.each(['mediapipe', 'selfie-segmenter'] as const)('preserves %s SDK exceptions and separates failure timing from successes', async model => {
  const f = fixture(); await f.engine.initialize(model); const cause = Error('sdk');
  (model === 'mediapipe' ? f.detect : f.segment).mockImplementationOnce(() => { throw cause; });
  expect(() => f.engine.detectVideoFrame(f.input, model, 0.3, f.metrics)).toThrow(cause);
  const stages = f.metrics.snapshot()!.stages;
  expect(stages.sdk.failed).toMatchObject({ count: 1, mean: 0 }); expect(stages.engine.failed.count).toBe(1);
  expect(stages.sdk.success.count).toBe(0); expect(stages.postprocess.failed.count).toBe(0); expect(f.result.close).not.toHaveBeenCalled();
  expect(f.engine.detectVideoFrame(f.input, model, 0.3, f.metrics).status).toBe('ready');
});
it('records failed transformation work and still closes the result exactly once', async () => {
  const f = fixture(); await f.engine.initialize('selfie-segmenter'); const cause = Error('mask');
  vi.mocked(f.mask.getAsFloat32Array).mockImplementationOnce(() => { throw cause; });
  expect(() => f.engine.detectVideoFrame(f.input, 'selfie-segmenter', 0.3, f.metrics)).toThrow(cause);
  expect(f.result.close).toHaveBeenCalledTimes(1);
  expect(f.metrics.snapshot()!.stages.postprocess.failed.count).toBe(1);
  expect(f.metrics.snapshot()!.stages.engine.failed.mean).toBe(15);
});
it('retains product cleanup exception policy and separately records failed cleanup', async () => {
  const f = fixture(); await f.engine.initialize('selfie-segmenter');
  f.result.close.mockImplementationOnce(() => { throw Error('close'); });
  expect(f.engine.detectVideoFrame(f.input, 'selfie-segmenter', 0.3, f.metrics).status).toBe('ready');
  expect(f.result.close).toHaveBeenCalledTimes(1);
  expect(f.metrics.snapshot()!.stages.cleanup.failed.count).toBe(1);
  expect(f.metrics.snapshot()!.stages.engine.success.count).toBe(1);
});
it.each(['mediapipe', 'selfie-segmenter'] as const)('separates stale %s output after reentrant clear from successful engine attempts', async model => {
  const f = fixture(); await f.engine.initialize(model);
  if (model === 'mediapipe') f.detect.mockImplementationOnce(() => { f.engine.clearDetections(); return { detections: [] }; });
  else f.segment.mockImplementationOnce(() => { f.engine.clearDetections(); return f.result; });
  expect(f.engine.detectVideoFrame(f.input, model, 0.3, f.metrics).status).toBe('not-ready');
  expect(f.metrics.snapshot()!.stages.engine.stale.count).toBe(1); expect(f.metrics.snapshot()!.stages.engine.success.count).toBe(0);
  expect(f.result.close).toHaveBeenCalledTimes(model === 'selfie-segmenter' ? 1 : 0);
});
for (const model of ['mediapipe', 'selfie-segmenter'] as const) {
  it.each(['disabled', 'valid', 'throwing-observer', 'throwing-clock', 'NaN-clock', 'backwards-clock'] as const)(`${model}: enabled/disabled equivalence with %s instrumentation`, async mode => {
    const f = fixture(); await f.engine.initialize(model);
    const now = vi.fn(() => {
      if (mode === 'throwing-clock') throw Error('clock');
      if (mode === 'NaN-clock') return NaN;
      if (mode === 'backwards-clock') return 100 - f.time();
      return f.time();
    });
    const observer: PipelineObserver | undefined = mode === 'disabled' ? undefined : { now, record: event => {
      if (mode === 'throwing-observer') throw Error('observer'); f.metrics.record(event);
    } };
    const result = f.engine.detectVideoFrame(f.input, model, 0.3, observer);
    expect(result).toMatchObject({ status: 'ready', model, requestId: 1 });
    const sdk = model === 'mediapipe' ? f.detect : f.segment;
    expect(sdk).toHaveBeenCalledExactlyOnceWith(f.input, 123);
    expect(performance.now).toHaveBeenCalledTimes(1); // Original product timestamp only; instrumentation uses its injected clock.
    expect(f.result.close).toHaveBeenCalledTimes(model === 'selfie-segmenter' ? 1 : 0);
    if (mode === 'disabled') expect(now).not.toHaveBeenCalled();
    if (model === 'selfie-segmenter') {
      expect(imageHash(f.context.createImageData.mock.results[0].value.data)).toBe('20,184,166,188,20,184,166,188,20,184,166,188,20,184,166,188');
      expect(f.context.putImageData).toHaveBeenCalledTimes(1);
    }
  });
}
function imageHash(data: Uint8ClampedArray) { return Array.from(data).join(','); }

it('observer failure during an SDK failure preserves the identical product exception', async () => {
  const f = fixture(); await f.engine.initialize('mediapipe'); const cause = Error('original SDK');
  f.detect.mockImplementationOnce(() => { throw cause; });
  expect(() => f.engine.detectVideoFrame(f.input, 'mediapipe', 0.3, { now: () => 0, record: () => { throw Error('observer'); } })).toThrow(cause);
  expect(f.detect).toHaveBeenCalledExactlyOnceWith(f.input, 123); expect(f.result.close).not.toHaveBeenCalled();
});
