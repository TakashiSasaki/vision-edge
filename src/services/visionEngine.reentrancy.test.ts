import { beforeEach, expect, it, vi } from 'vitest';
import { VisionEngine, type VisionTaskFactory } from './visionEngine';

function fixture() {
  const detector = { detectForVideo: vi.fn(() => ({ detections: [] })), close: vi.fn() };
  const segmenter = { segmentForVideo: vi.fn(), close: vi.fn() };
  const factory: VisionTaskFactory = {
    load: vi.fn(async () => ({ wasmLoaderPath: 'loader', wasmBinaryPath: 'wasm' })),
    detector: vi.fn(async () => detector),
    segmenter: vi.fn(async () => segmenter),
  };
  return { engine: new VisionEngine(factory), factory, detector, segmenter };
}

type Initialization = ReturnType<VisionEngine['initialize']>;

beforeEach(() => { vi.spyOn(console, 'warn').mockImplementation(() => {}); });

it('shares the exact Promise with a same-model request made inside the loading notification', async () => {
  const f = fixture();
  let reentered = false;
  let nested: Initialization | undefined;
  const unsubscribe = f.engine.subscribe(() => {
    if (f.engine.getState().status === 'loading' && !reentered) {
      reentered = true;
      nested = f.engine.initialize('mediapipe');
    }
  });
  const outer = f.engine.initialize('mediapipe');
  const results = await Promise.all([outer, nested]);
  unsubscribe();
  expect(nested).toBe(outer);
  expect(results[0]).toMatchObject({ status: 'ready', model: 'mediapipe', requestId: 1 });
  expect(results[1]).toBe(results[0]);
  expect(f.factory.load).toHaveBeenCalledTimes(1);
  expect(f.factory.detector).toHaveBeenCalledTimes(1);
  expect(f.factory.segmenter).not.toHaveBeenCalled();
  f.engine.dispose();
});

it('does not overwrite the new-model Promise when loading notification switches models', async () => {
  const f = fixture();
  let replacement: Initialization | undefined;
  const unsubscribe = f.engine.subscribe(() => {
    const state = f.engine.getState();
    if (state.status === 'loading' && state.model === 'selfie-segmenter') {
      replacement = f.engine.initialize('mediapipe');
    }
  });
  const obsolete = f.engine.initialize('selfie-segmenter');
  const shared = f.engine.initialize('mediapipe');
  const [oldResult, currentResult] = await Promise.all([obsolete, replacement]);
  unsubscribe();
  expect(shared).toBe(replacement);
  expect(shared).not.toBe(obsolete);
  expect(oldResult).toMatchObject({ status: 'cancelled', model: 'selfie-segmenter' });
  expect(currentResult).toMatchObject({ status: 'ready', model: 'mediapipe' });
  expect(f.engine.getState()).toBe(currentResult);
  expect(f.factory.load).toHaveBeenCalledTimes(1);
  expect(f.factory.segmenter).not.toHaveBeenCalled();
  expect(f.detector.close).not.toHaveBeenCalled();
  f.engine.dispose();
});

it('returns its own cancelled Promise without starting a load after synchronous disposal', async () => {
  const f = fixture();
  const unsubscribe = f.engine.subscribe(() => {
    if (f.engine.getState().status === 'loading') f.engine.dispose();
  });
  const pending = f.engine.initialize('mediapipe');
  const result = await pending;
  unsubscribe();
  expect(pending).toBeInstanceOf(Promise);
  expect(result).toMatchObject({ status: 'cancelled', model: 'mediapipe' });
  expect(f.engine.getState().status).toBe('idle');
  expect(f.factory.load).not.toHaveBeenCalled();
  expect(f.factory.detector).not.toHaveBeenCalled();
  expect((await f.engine.initialize('mediapipe')).status).toBe('ready');
  f.engine.dispose();
});

it('preserves replacement ownership when a ready notification selects another model', async () => {
  const f = fixture();
  let replacement: Initialization | undefined;
  const unsubscribe = f.engine.subscribe(() => {
    const state = f.engine.getState();
    if (state.status === 'ready' && state.model === 'selfie-segmenter') {
      replacement = f.engine.initialize('mediapipe');
    }
  });
  await f.engine.initialize('selfie-segmenter');
  await replacement;
  unsubscribe();
  expect(f.engine.getState()).toMatchObject({ status: 'ready', model: 'mediapipe' });
  expect(f.segmenter.close).toHaveBeenCalledTimes(1);
  expect(f.detector.close).not.toHaveBeenCalled();
  f.engine.dispose();
  expect(f.detector.close).toHaveBeenCalledTimes(1);
});

it('keeps an explicit error-notification retry separate from the failed request', async () => {
  const f = fixture();
  const cause = new Error('first load failed');
  vi.mocked(f.factory.load).mockRejectedValueOnce(cause);
  let retry: Initialization | undefined;
  const unsubscribe = f.engine.subscribe(() => {
    if (f.engine.getState().status === 'error') retry = f.engine.initialize('mediapipe');
  });
  const original = f.engine.initialize('mediapipe');
  const failed = await original;
  await retry;
  unsubscribe();
  expect(failed).toMatchObject({ status: 'error', cause });
  expect(retry).not.toBe(original);
  expect(f.engine.getState()).toMatchObject({ status: 'ready', model: 'mediapipe' });
  expect(f.factory.load).toHaveBeenCalledTimes(2);
  expect(f.factory.detector).toHaveBeenCalledTimes(1);
  f.engine.dispose();
});

it('handles a synchronous factory exception and permits an ordinary retry', async () => {
  const f = fixture();
  const cause = new Error('synchronous loader failure');
  vi.mocked(f.factory.load).mockImplementationOnce(() => { throw cause; });
  const failed = f.engine.initialize('mediapipe');
  expect(await failed).toMatchObject({ status: 'error', stage: 'wasm', cause });
  const retry = f.engine.initialize('mediapipe');
  expect(f.engine.initialize('mediapipe')).toBe(retry);
  expect((await retry).status).toBe('ready');
  expect(f.factory.detector).toHaveBeenCalledTimes(1);
  f.engine.dispose();
});
