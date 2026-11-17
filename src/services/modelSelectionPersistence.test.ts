import { describe, expect, it } from 'vitest';
import {
  getDefaultModelSelection,
  loadPersistedModelSelection,
  MODEL_SELECTION_STORAGE_KEY,
  savePersistedModelSelection,
} from './modelSelectionPersistence';

function makeMemoryStorage(initial: Record<string, string> = {}): Storage {
  const map = new Map<string, string>(Object.entries(initial));
  return {
    get length() {
      return map.size;
    },
    clear() {
      map.clear();
    },
    getItem(key: string) {
      return map.get(key) ?? null;
    },
    key(index: number) {
      return Array.from(map.keys())[index] ?? null;
    },
    removeItem(key: string) {
      map.delete(key);
    },
    setItem(key: string, value: string) {
      map.set(key, value);
    },
  };
}

describe('modelSelectionPersistence', () => {
  it('returns default selection when storage is empty or unavailable', () => {
    expect(loadPersistedModelSelection(null)).toEqual(getDefaultModelSelection());
    expect(loadPersistedModelSelection(makeMemoryStorage())).toEqual(getDefaultModelSelection());
  });

  it('saves and restores activeModel, per-model selectedVariants, and classifier selection', () => {
    const storage = makeMemoryStorage();
    const defaults = getDefaultModelSelection();
    const custom = {
      activeModel: 'stereo-depth' as const,
      selectedVariants: {
        ...defaults.selectedVariants,
        'stereo-depth': 'lightstereo-s',
        'yolo-seg': 'yolov8s-seg',
      },
      classifierMode: true,
      classifierVariant: 'lite2' as const,
    };

    savePersistedModelSelection(custom, storage);
    expect(loadPersistedModelSelection(storage)).toEqual(custom);
  });

  it('sanitizes invalid models/variants or malformed JSON back to safe defaults', () => {
    const corrupt = makeMemoryStorage({ [MODEL_SELECTION_STORAGE_KEY]: '{not-json' });
    expect(loadPersistedModelSelection(corrupt)).toEqual(getDefaultModelSelection());

    const partialInvalid = makeMemoryStorage({
      [MODEL_SELECTION_STORAGE_KEY]: JSON.stringify({
        activeModel: 'unknown-model',
        classifierMode: 'yes',
        classifierVariant: 'invalid-lite',
        selectedVariants: {
          'stereo-depth': 'lightstereo-s',
          'yolo-seg': 'nonexistent-variant',
        },
      }),
    });
    const loaded = loadPersistedModelSelection(partialInvalid);
    expect(loaded.activeModel).toBe('selfie-segmenter');
    expect(loaded.classifierMode).toBe(false);
    expect(loaded.classifierVariant).toBe('lite0');
    expect(loaded.selectedVariants['stereo-depth']).toBe('lightstereo-s');
    expect(loaded.selectedVariants['yolo-seg']).toBe('yolov8n-seg');
  });

  it('does not throw when storage throws quota or security errors', () => {
    const throwingStorage = {
      getItem() {
        throw new Error('SecurityError');
      },
      setItem() {
        throw new Error('QuotaExceededError');
      },
    };
    expect(loadPersistedModelSelection(throwingStorage)).toEqual(getDefaultModelSelection());
    expect(() =>
      savePersistedModelSelection(getDefaultModelSelection(), throwingStorage)
    ).not.toThrow();
  });
});
