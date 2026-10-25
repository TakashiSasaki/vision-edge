import { describe, expect, it, vi } from 'vitest';
import type * as ort from 'onnxruntime-web';
import { COCO_CLASSES, YoloSegEngine, type YoloSegTask } from './yoloSegService';

describe('YoloSegService and Engine', () => {
  it('exposes standard 80 COCO classes starting with person and vehicle categories', () => {
    expect(COCO_CLASSES).toHaveLength(80);
    expect(COCO_CLASSES[0]).toBe('person');
    expect(COCO_CLASSES[1]).toBe('bicycle');
    expect(COCO_CLASSES[2]).toBe('car');
    expect(COCO_CLASSES[79]).toBe('toothbrush');
  });

  it('provides a YoloSegEngine that adheres to YoloSegTask interface and handles lifecycle close', () => {
    const mockSession = {
      inputNames: ['images'],
      outputNames: ['output0', 'output1'],
      run: vi.fn(async () => ({})),
      release: vi.fn(async () => {}),
    };

    const engine: YoloSegTask = new YoloSegEngine(mockSession as unknown as ort.InferenceSession);
    expect(typeof engine.detectForVideo).toBe('function');
    expect(typeof engine.close).toBe('function');

    engine.close();
    expect(mockSession.release).toHaveBeenCalled();
  });
});
