import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { TTSConfig } from '../types/vision';

const config: TTSConfig = {
  enabled: true, autoReadout: true, cooldownSec: 3,
  rate: 1, pitch: 1, volume: 1, voiceURI: '',
};
const originalSpeech = Object.getOwnPropertyDescriptor(window, 'speechSynthesis');

beforeEach(() => vi.resetModules());
afterEach(() => {
  vi.unstubAllGlobals();
  if (originalSpeech) Object.defineProperty(window, 'speechSynthesis', originalSpeech);
  else Reflect.deleteProperty(window, 'speechSynthesis');
});

describe('speechService', () => {
  it('starts with empty history and available browser voices without speaking', async () => {
    const voices = [{ voiceURI: 'ja-test', lang: 'ja-JP' }] as SpeechSynthesisVoice[];
    const synth = { getVoices: vi.fn(() => voices), speak: vi.fn(), cancel: vi.fn(), onvoiceschanged: null };
    vi.stubGlobal('speechSynthesis', synth);
    const { speechService } = await import('./speechService');

    expect(speechService.getHistory()).toEqual([]);
    expect(speechService.getVoices()).toEqual(voices);
    expect(speechService.canSpeakLabel('dog', 3)).toBe(true);
    expect(synth.speak).not.toHaveBeenCalled();
    expect(speechService.speak('犬', 'dog', { ...config, enabled: false })).toBe(false);
    expect(synth.speak).not.toHaveBeenCalled();
  });

  it('returns false safely when speech synthesis is unavailable', async () => {
    Reflect.deleteProperty(window, 'speechSynthesis');
    const { speechService } = await import('./speechService');

    expect(speechService.loadVoices()).toEqual([]);
    expect(speechService.getVoices()).toEqual([]);
    for (const trigger of ['auto', 'tap', 'manual'] as const) {
      expect(speechService.speak('犬', 'dog', config, trigger)).toBe(false);
    }
    expect(() => speechService.cancel()).not.toThrow();
    expect(speechService.getHistory()).toEqual([]);
  });

  it('notifies subscribers when history is cleared and respects unsubscribe', async () => {
    Reflect.deleteProperty(window, 'speechSynthesis');
    const { speechService } = await import('./speechService');
    const listener = vi.fn();
    const unsubscribe = speechService.subscribe(listener);
    speechService.clearHistory();
    expect(listener).toHaveBeenCalledWith(false, undefined);
    unsubscribe();
    speechService.clearHistory();
    expect(listener).toHaveBeenCalledTimes(1);
  });
});
