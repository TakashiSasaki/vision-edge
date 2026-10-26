import type { TTSConfig } from '../types/vision';
// A deliberate UI action owns one utterance. Late browser events from cancelled
// utterances cannot release another frame's hold or resume the wrong session.
export class ClassificationSpeech {
  private token = 0;
  private active: { token: number; settled: () => void } | null = null;
  constructor(private readonly synth: SpeechSynthesis | null = typeof window !== 'undefined' ? window.speechSynthesis ?? null : null) {}
  cancel(): void {
    this.token++;
    const old = this.active; this.active = null;
    if (old) { try { this.synth?.cancel(); } catch (cause) { console.warn('Classification speech cancellation failed', cause); } finally { old.settled(); } }
  }
  speak(text: string, current: () => boolean, hold: (holding: boolean) => void, config: TTSConfig): boolean {
    this.cancel();
    if (!this.synth || typeof SpeechSynthesisUtterance !== 'function' || !current()) return false;
    try {
    const token = ++this.token, utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'ja-JP'; utterance.rate = config.rate; utterance.pitch = config.pitch; utterance.volume = config.volume;
    const voice = this.synth.getVoices().find(v => v.voiceURI === config.voiceURI); if (voice) { utterance.voice = voice; utterance.lang = voice.lang; }
    const finish = () => { if (this.active?.token !== token) return; this.active = null; hold(false); };
    this.active = { token, settled: () => hold(false) }; hold(true);
    utterance.onstart = () => { if (this.active?.token === token && !current()) this.cancel(); };
    utterance.onend = finish; utterance.onerror = finish;
    if (!current()) { this.cancel(); return false; }
    this.synth.speak(utterance); return true;
    } catch { this.cancel(); return false; }
  }
}
