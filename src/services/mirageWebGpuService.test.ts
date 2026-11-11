import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as ort from 'onnxruntime-web';
import type { DetectedObject, InferenceInput } from '../types/vision';
import type { MirageTask } from './mirageWasmService';
import { createMirageWebGpuTask, MirageWebGpuEngine } from './mirageWebGpuService';
import { MIRAGE_WEBGPU_UNAVAILABLE_CODE } from './mirageAvailability';
import * as assetResolver from './modelAssetResolver';
import * as onnxSession from './onnxSession';

// These are synthetic decoder/lifecycle controls. They do not load Mirage weights
// or establish numerical equivalence with the original model.
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

function observation(label: string, overrides: Partial<DetectedObject> = {}): DetectedObject {
  return {
    id: label,
    label,
    labelJa: label,
    category: '物体',
    confidence: 0.98,
    bbox: { x: 0.35, y: 0.25, width: 0.3, height: 0.5 },
    color: '#3b82f6',
    ...overrides,
  };
}

function outputTensors({ object = false, ocr = false } = {}): Record<string, ort.Tensor> {
  const area = 32 * 32;
  const dense = new Float32Array(10 * area);
  const classes = new Float32Array(32 * area).fill(-5);
  if (object) {
    const idx = 16 * 32 + 16;
    dense[idx] = 0.92;
    dense[area + idx] = 0.3;
    dense[2 * area + idx] = 0.5;
    dense[3 * area + idx] = 0.5;
    dense[4 * area + idx] = 0.5;
    classes[idx] = 5;
  }
  if (ocr) {
    for (let x = 6; x <= 8; x++) dense[5 * area + 4 * 32 + x] = 0.88;
  }
  return {
    dense_out: new ort.Tensor('float32', dense, [1, 10, 32, 32]),
    mask_coeffs: new ort.Tensor('float32', new Float32Array(32 * area), [1, 32, 32, 32]),
    mask_protos: new ort.Tensor('float32', new Float32Array(32 * 64 * 64), [1, 32, 64, 64]),
    class_logits: new ort.Tensor('float32', classes, [1, 32, 32, 32]),
  };
}

function sessionWithOutputs(options: { object?: boolean; ocr?: boolean } = {}) {
  return {
    inputNames: ['images'],
    outputNames: ['dense_out', 'mask_coeffs', 'mask_protos', 'class_logits'],
    run: vi.fn(async (_feeds: Record<string, ort.Tensor>) => {
      void _feeds;
      return outputTensors(options);
    }),
    release: vi.fn(async () => {}),
  };
}

describe('Mirage ONNX availability and frame ownership', () => {
  let pixels: WeakMap<object, number>;

  beforeEach(() => {
    pixels = new WeakMap();
    const contexts = new WeakMap<HTMLCanvasElement, CanvasRenderingContext2D>();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (this: HTMLCanvasElement) {
      let ctx = contexts.get(this);
      if (!ctx) {
        ctx = {
          drawImage: vi.fn((source: CanvasImageSource) => {
            pixels.set(this, pixels.get(source) ?? 80);
          }),
          getImageData: (_x: number, _y: number, width: number, height: number) => ({
            data: new Uint8ClampedArray(width * height * 4).fill(pixels.get(this) ?? 0),
            width,
            height,
          }),
          createImageData: (width: number, height: number) => ({
            data: new Uint8ClampedArray(width * height * 4), width, height,
          }),
          putImageData: vi.fn(),
        } as unknown as CanvasRenderingContext2D;
        contexts.set(this, ctx);
      }
      return ctx;
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('rejects default and custom-URL creation before fetching, resolving assets or creating sessions', async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    const resolveSpy = vi.spyOn(assetResolver, 'resolveModelAssetUrls');
    const createSpy = vi.spyOn(onnxSession, 'createOrtSession');
    const sessionSpy = vi.spyOn(ort.InferenceSession, 'create');
    const recordAssetSource = vi.fn();
    const wasmTask = { detectForVideo: vi.fn(), close: vi.fn() };
    for (const options of [
      undefined,
      { modelUrl: '/models/claimed-valid.onnx', delegate: 'WebGPU' as const, recordAssetSource, wasmTask },
      { modelUrl: '/models/claimed-valid.onnx', delegate: 'WASM' as const },
    ]) {
      await expect(createMirageWebGpuTask(options)).rejects.toMatchObject({
        code: MIRAGE_WEBGPU_UNAVAILABLE_CODE,
      });
    }
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(resolveSpy).not.toHaveBeenCalled();
    expect(createSpy).not.toHaveBeenCalled();
    expect(sessionSpy).not.toHaveBeenCalled();
    expect(recordAssetSource).not.toHaveBeenCalled();
    expect(wasmTask.detectForVideo).not.toHaveBeenCalled();
    expect(wasmTask.close).not.toHaveBeenCalled();
  });

  it('rejects before even reading a custom model URL', async () => {
    const getModelUrl = vi.fn(() => { throw new Error('must not read caller model URL'); });
    const options = Object.defineProperty({}, 'modelUrl', { get: getModelUrl });
    await expect(createMirageWebGpuTask(options)).rejects.toMatchObject({
      code: MIRAGE_WEBGPU_UNAVAILABLE_CODE,
    });
    expect(getModelUrl).not.toHaveBeenCalled();
  });

  it('enriches first and every fifth frame across 20 calls without replaying any native object', async () => {
    const input = document.createElement('canvas');
    const session = sessionWithOutputs();
    const nativeFrames: number[] = [];
    const wasmTask: MirageTask = {
      detectForVideo: vi.fn(async (snapshot: InferenceInput) => {
        const frame = pixels.get(snapshot)!;
        nativeFrames.push(frame);
        return [observation(`native-frame-${frame}`)];
      }),
      close: vi.fn(),
    };
    const engine = new MirageWebGpuEngine(session as unknown as ort.InferenceSession, undefined, wasmTask);
    for (let frame = 1; frame <= 20; frame++) {
      pixels.set(input, frame);
      const actual = await engine.detectForVideo(input);
      expect(actual.map(obj => obj.label)).toEqual(
        frame === 1 || frame % 5 === 0 ? [`native-frame-${frame}`] : []
      );
    }
    expect(nativeFrames).toEqual([1, 5, 10, 15, 20]);
    expect(session.run).toHaveBeenCalledTimes(20);
    engine.close();
  });

  it('does not reuse names, OCR strings or unmatched boxes on a frame without native enrichment', async () => {
    const input = document.createElement('canvas');
    const session = sessionWithOutputs({ object: true, ocr: true });
    const wasmTask: MirageTask = {
      detectForVideo: vi.fn(async () => [
        observation('native-object'),
        observation('first-frame-only', { bbox: { x: 0.02, y: 0.7, width: 0.1, height: 0.1 } }),
        observation('ocr:FIRST', {
          labelJa: '文字: "FIRST"',
          category: 'テキスト (OCR)',
          bbox: { x: 0.17, y: 0.10, width: 0.15, height: 0.08 },
        }),
      ]),
      close: vi.fn(),
    };
    const engine = new MirageWebGpuEngine(session as unknown as ort.InferenceSession, undefined, wasmTask);
    const first = await engine.detectForVideo(input);
    expect(first.map(obj => obj.label)).toEqual(['native-object', 'first-frame-only', 'ocr:FIRST']);
    const second = await engine.detectForVideo(input, 0.25);
    expect(second.map(obj => obj.label)).toEqual(['person', 'ocr:text-region']);
    expect(second.every(obj => !obj.labelJa.includes('FIRST'))).toBe(true);
    engine.close();
  });

  it('rechecks current confidence, OCR choice and result limit on native observations', async () => {
    const input = document.createElement('canvas');
    const session = sessionWithOutputs({ ocr: true });
    const wasmTask: MirageTask = {
      detectForVideo: vi.fn(async () => [
        observation('weak', { confidence: 0.5 }),
        observation('ocr:DISABLED', { category: 'テキスト (OCR)' }),
        observation('accepted-1', { bbox: { x: 0.05, y: 0.05, width: 0.1, height: 0.1 } }),
        observation('accepted-2', { bbox: { x: 0.8, y: 0.05, width: 0.1, height: 0.1 } }),
        observation('over-limit'),
      ]),
      close: vi.fn(),
    };
    const engine = new MirageWebGpuEngine(session as unknown as ort.InferenceSession, undefined, wasmTask);
    expect((await engine.detectForVideo(input, 0.9, { includeOcr: false, maxDetections: 2 })).map(obj => obj.label))
      .toEqual(['accepted-1', 'accepted-2']);
    // A changed threshold cannot be bypassed by replaying a previous result.
    expect(await engine.detectForVideo(input, 0.99, { includeOcr: false })).toEqual([]);
    for (let frame = 3; frame <= 4; frame++) await engine.detectForVideo(input, 0.99);
    expect(await engine.detectForVideo(input, 0.99)).toEqual([]);
    engine.close();
  });

  it('uses the same captured pixels for both runtimes and rejects overlapping calls', async () => {
    const input = document.createElement('canvas');
    pixels.set(input, 40);
    const entered = deferred<void>();
    const continueRun = deferred<void>();
    const session = sessionWithOutputs();
    let onnxRed = 0;
    session.run.mockImplementation(async (feeds) => {
      onnxRed = (feeds.images.data as Float32Array)[0];
      entered.resolve();
      await continueRun.promise;
      return outputTensors();
    });
    const nativeInputs: InferenceInput[] = [];
    const nativePixels: number[] = [];
    const wasmTask: MirageTask = {
      detectForVideo: vi.fn(async (snapshot) => {
        nativeInputs.push(snapshot);
        nativePixels.push(pixels.get(snapshot)!);
        return [];
      }),
      close: vi.fn(),
    };
    const engine = new MirageWebGpuEngine(session as unknown as ort.InferenceSession, undefined, wasmTask);
    const requestOptions = { includeOcr: false, maxDetections: 2 };
    const first = engine.detectForVideo(input, 0.4, requestOptions);
    await entered.promise;
    pixels.set(input, 220); // A live input advances while ONNX is running.
    requestOptions.includeOcr = true;
    await expect(engine.detectForVideo(input)).rejects.toThrow('inference is already in progress');
    expect(() => engine.setWasmTask(null)).toThrow('cannot replace the WASM task during inference');
    continueRun.resolve();
    await first;
    expect(onnxRed).toBeCloseTo(40 / 255, 6);
    expect(nativeInputs[0]).not.toBe(input);
    expect(nativeInputs[0]).toBeInstanceOf(HTMLCanvasElement);
    expect(nativePixels).toEqual([40]);
    expect(wasmTask.detectForVideo).toHaveBeenCalledWith(nativeInputs[0], 0.4, {
      includeOcr: false, maxDetections: 2,
    });
    expect(session.run).toHaveBeenCalledTimes(1);
    engine.close();
  });

  it('retries failed native inference at later cadence slots and does not replay the last success', async () => {
    const input = document.createElement('canvas');
    const session = sessionWithOutputs();
    const detectForVideo = vi.fn()
      .mockRejectedValueOnce(new Error('first native failure'))
      .mockResolvedValueOnce([observation('frame-5')])
      .mockRejectedValueOnce(new Error('later native failure'))
      .mockResolvedValueOnce([observation('frame-15')]);
    const engine = new MirageWebGpuEngine(session as unknown as ort.InferenceSession, undefined, {
      detectForVideo,
      close: vi.fn(),
    });
    for (let frame = 1; frame <= 15; frame++) {
      const expected = frame === 5 || frame === 15 ? [`frame-${frame}`] : [];
      expect((await engine.detectForVideo(input)).map(obj => obj.label)).toEqual(expected);
    }
    expect(detectForVideo).toHaveBeenCalledTimes(4);
    engine.close();
  });

  it('closes replaced and detached native tasks once while preserving the same task', () => {
    const session = sessionWithOutputs();
    const initialTask: MirageTask = {
      detectForVideo: vi.fn(),
      close: vi.fn(() => { throw new Error('retired task cleanup failed'); }),
    };
    const replacementTask: MirageTask = { detectForVideo: vi.fn(), close: vi.fn() };
    const finalTask: MirageTask = { detectForVideo: vi.fn(), close: vi.fn() };
    const engine = new MirageWebGpuEngine(session as unknown as ort.InferenceSession, undefined, initialTask);

    engine.setWasmTask(initialTask);
    expect(initialTask.close).not.toHaveBeenCalled();
    expect(() => engine.setWasmTask(replacementTask)).not.toThrow();
    expect(initialTask.close).toHaveBeenCalledTimes(1);
    engine.setWasmTask(replacementTask);
    expect(replacementTask.close).not.toHaveBeenCalled();

    engine.setWasmTask(null);
    engine.setWasmTask(null);
    expect(replacementTask.close).toHaveBeenCalledTimes(1);
    engine.setWasmTask(finalTask);
    engine.setWasmTask(finalTask);
    expect(finalTask.close).not.toHaveBeenCalled();

    engine.close();
    engine.close();
    expect(initialTask.close).toHaveBeenCalledTimes(1);
    expect(replacementTask.close).toHaveBeenCalledTimes(1);
    expect(finalTask.close).toHaveBeenCalledTimes(1);
    expect(session.release).toHaveBeenCalledTimes(1);
    expect(session.run).not.toHaveBeenCalled();
  });

  it('discards a native completion after close and releases the session when the call settles', async () => {
    const input = document.createElement('canvas');
    const session = sessionWithOutputs();
    const entered = deferred<void>();
    const nativeResult = deferred<DetectedObject[]>();
    const wasmTask: MirageTask = {
      detectForVideo: vi.fn(async () => {
        entered.resolve();
        return nativeResult.promise;
      }),
      close: vi.fn(),
    };
    const engine = new MirageWebGpuEngine(session as unknown as ort.InferenceSession, undefined, wasmTask);
    const pending = engine.detectForVideo(input);
    await entered.promise;
    engine.close();
    engine.close();
    expect(wasmTask.close).toHaveBeenCalledTimes(1);
    expect(session.release).not.toHaveBeenCalled();
    nativeResult.resolve([observation('too-late')]);
    expect(await pending).toEqual([]);
    expect(session.release).toHaveBeenCalledTimes(1);
    await expect(engine.detectForVideo(input)).rejects.toThrow('task is closed');
    expect(() => engine.setWasmTask(wasmTask)).toThrow('task is closed');
    expect(wasmTask.detectForVideo).toHaveBeenCalledTimes(1);
  });
});
