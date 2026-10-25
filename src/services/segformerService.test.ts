import { describe, expect, it, vi } from 'vitest';
import type * as ort from 'onnxruntime-web';
import { ADE20K_CLASSES, getADE20KInfo } from './ade20kDictionary';
import { SegFormerEngine, type SegFormerTask } from './segformerService';

describe('SegFormerService and Engine', () => {
  it('exposes standard 150 ADE20K classes with Japanese labels', () => {
    expect(ADE20K_CLASSES).toHaveLength(150);
    expect(ADE20K_CLASSES[0].en).toBe('wall');
    expect(ADE20K_CLASSES[0].ja).toBe('壁');
    expect(ADE20K_CLASSES[2].en).toBe('sky');
    expect(ADE20K_CLASSES[2].ja).toBe('空');
    expect(ADE20K_CLASSES[12].en).toBe('person');
    expect(ADE20K_CLASSES[12].ja).toBe('人物');
  });

  it('retrieves class info via getADE20KInfo safely with fallback', () => {
    const wall = getADE20KInfo(0);
    expect(wall.ja).toBe('壁');
    const outOfBounds = getADE20KInfo(999);
    expect(outOfBounds.ja).toBe('クラス 999');
  });

  it('provides a SegFormerEngine that adheres to SegFormerTask interface and handles lifecycle close', () => {
    const mockSession = {
      inputNames: ['pixel_values'],
      outputNames: ['logits'],
      run: vi.fn(async () => ({})),
      release: vi.fn(async () => {}),
    };

    const engine: SegFormerTask = new SegFormerEngine(mockSession as unknown as ort.InferenceSession);
    expect(typeof engine.detectForVideo).toBe('function');
    expect(typeof engine.close).toBe('function');

    engine.close();
    expect(mockSession.release).toHaveBeenCalled();
  });
});
