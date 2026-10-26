import { afterEach, expect, it, vi } from 'vitest';
import { japaneseClassificationLabel } from './classificationLabels';
import { ClassificationSpeech } from './classificationSpeech';
import type { TTSConfig } from '../types/vision';
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });
const config: TTSConfig = { enabled: false, autoReadout: true, cooldownSec: 3, rate: 1, pitch: 1, volume: 1, voiceURI: '' };
it.each([['goldfish, Carassius auratus','金魚'],['coffee mug','コーヒーマグ'],['banana','バナナ'],['volcano','火山']])('exact translation %s', (raw, expected) => expect(japaneseClassificationLabel(raw)).toBe(expected));
it.each(['goldfish', 'Goldfish, Carassius auratus', 'a coffee mug', 'banana split', 'UNKNOWN Exact Raw Label', '', '__proto__'])('unknown/partial/case label stays unregistered %s', raw => expect(japaneseClassificationLabel(raw)).toBeNull());
function fixture() {
  class Utterance { text: string; onstart?: () => void; onend?: () => void; onerror?: () => void; constructor(text: string) { this.text = text; } }
  vi.stubGlobal('SpeechSynthesisUtterance', Utterance);
  const utterances: Utterance[] = [], synth = { getVoices: vi.fn(() => []), speak: vi.fn((u: Utterance) => utterances.push(u)), cancel: vi.fn() };
  return { utterances, synth, owner: new ClassificationSpeech(synth as unknown as SpeechSynthesis) };
}
it('speech starts only on deliberate invocation and owns the frame hold until the matching finish', () => {
  const f = fixture(), hold = vi.fn(); expect(f.synth.speak).not.toHaveBeenCalled();
  expect(f.owner.speak('raw label, score0.7', () => true, hold, config)).toBe(true); expect(hold).toHaveBeenLastCalledWith(true); expect(f.utterances[0].text).toBe('raw label, score0.7');
  f.utterances[0].onend?.(); expect(hold).toHaveBeenLastCalledWith(false); f.owner.cancel(); vi.unstubAllGlobals();
});
it('obsolete utterance completion cannot release a replacement frame hold', () => {
  const f = fixture(), old = vi.fn(), current = vi.fn();
  f.owner.speak('A', () => true, old, config); f.owner.speak('B', () => true, current, config); const before = current.mock.calls.length;
  f.utterances[0].onend?.(); f.utterances[0].onerror?.(); expect(current).toHaveBeenCalledTimes(before); expect(current).toHaveBeenLastCalledWith(true);
  f.owner.cancel(); expect(current).toHaveBeenLastCalledWith(false); const count = current.mock.calls.length; f.utterances[1].onend?.(); expect(current).toHaveBeenCalledTimes(count); vi.unstubAllGlobals();
});
it('stale selection and unsupported speech never start or hold', () => {
  const f = fixture(), hold = vi.fn(); expect(f.owner.speak('A', () => false, hold, config)).toBe(false); expect(f.synth.speak).not.toHaveBeenCalled(); expect(hold).not.toHaveBeenCalled();
  expect(new ClassificationSpeech(null).speak('A', () => true, hold, config)).toBe(false); vi.unstubAllGlobals();
});
it('queued onstart cancels a no-longer-current owner; browser throw also releases hold', () => {
  const f = fixture(), hold = vi.fn(); let valid = true; f.owner.speak('A', () => valid, hold, config); valid = false; f.utterances[0].onstart?.(); expect(hold).toHaveBeenLastCalledWith(false);
  f.synth.speak.mockImplementationOnce(() => { throw Error('TTS_FAILURE'); }); expect(f.owner.speak('B', () => true, hold, config)).toBe(false); expect(hold).toHaveBeenLastCalledWith(false); vi.unstubAllGlobals();
});

it('voice enumeration failure returns an explicit unsuccessful start with no stuck hold', () => {
  const f=fixture(), hold=vi.fn(); f.synth.getVoices.mockImplementationOnce(()=>{throw Error('VOICE_FAILURE');}); expect(f.owner.speak('A',()=>true,hold,config)).toBe(false); expect(f.synth.speak).not.toHaveBeenCalled(); expect(hold).not.toHaveBeenCalled();
});
it('cancel failure is handled, still releases the old owner, and is reported', () => {
  const f=fixture(), hold=vi.fn(); const warn=vi.spyOn(console,'warn').mockImplementation(()=>{}); f.owner.speak('A',()=>true,hold,config); f.synth.cancel.mockImplementationOnce(()=>{throw Error('CANCEL_FAILURE');}); expect(()=>f.owner.cancel()).not.toThrow(); expect(hold).toHaveBeenLastCalledWith(false); expect(warn).toHaveBeenCalled();
});
