import { TTSConfig } from '../types/vision';

export interface SpeechHistoryItem {
  id: string;
  text: string;
  timestamp: Date;
  triggeredBy: 'auto' | 'tap' | 'manual';
  label: string;
}

class SpeechService {
  private synth: SpeechSynthesis | null = null;
  private voices: SpeechSynthesisVoice[] = [];
  private lastSpokenMap = new Map<string, number>();
  private listeners = new Set<(isSpeaking: boolean, currentItem?: SpeechHistoryItem) => void>();
  private history: SpeechHistoryItem[] = [];
  private isSpeaking = false;

  constructor() {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      this.synth = window.speechSynthesis;
      this.loadVoices();
      if (this.synth.onvoiceschanged !== undefined) {
        this.synth.onvoiceschanged = () => this.loadVoices();
      }
    }
  }

  public loadVoices(): SpeechSynthesisVoice[] {
    if (!this.synth) return [];
    this.voices = this.synth.getVoices();
    return this.voices;
  }

  public getVoices(): SpeechSynthesisVoice[] {
    if (this.voices.length === 0 && this.synth) {
      this.voices = this.synth.getVoices();
    }
    return this.voices;
  }

  public subscribe(callback: (isSpeaking: boolean, currentItem?: SpeechHistoryItem) => void): () => void {
    this.listeners.add(callback);
    return () => {
      this.listeners.delete(callback);
    };
  }

  private notify(speaking: boolean, item?: SpeechHistoryItem) {
    this.isSpeaking = speaking;
    this.listeners.forEach(cb => cb(speaking, item));
  }

  public getHistory(): SpeechHistoryItem[] {
    return [...this.history];
  }

  public clearHistory() {
    this.history = [];
    this.notify(this.isSpeaking);
  }

  public cancel() {
    if (this.synth) {
      this.synth.cancel();
      this.notify(false);
    }
  }

  public canSpeakLabel(label: string, cooldownSec: number): boolean {
    const now = Date.now();
    const last = this.lastSpokenMap.get(label.toLowerCase()) || 0;
    const cooldownMs = cooldownSec * 1000;
    return now - last >= cooldownMs;
  }

  public speak(
    text: string,
    labelKey: string,
    config: TTSConfig,
    triggeredBy: 'auto' | 'tap' | 'manual' = 'auto'
  ): boolean {
    if (!this.synth || !config.enabled) return false;

    if (triggeredBy === 'auto') {
      if (!config.autoReadout) return false;
      if (!this.canSpeakLabel(labelKey, config.cooldownSec)) {
        return false;
      }
      if (this.synth.speaking) {
        return false;
      }
    } else {
      this.synth.cancel();
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.volume = Math.max(0, Math.min(1, config.volume));
    utterance.rate = Math.max(0.5, Math.min(2.0, config.rate));
    utterance.pitch = Math.max(0.5, Math.min(1.5, config.pitch));

    const available = this.getVoices();
    let selectedVoice = config.voiceURI ? available.find(v => v.voiceURI === config.voiceURI) : undefined;

    if (!selectedVoice) {
      selectedVoice = available.find(v => v.lang.startsWith('ja') || v.lang.includes('JP'));
    }

    if (selectedVoice) {
      utterance.voice = selectedVoice;
      utterance.lang = selectedVoice.lang;
    } else {
      utterance.lang = 'ja-JP';
    }

    const historyItem: SpeechHistoryItem = {
      id: Math.random().toString(36).substring(2, 9),
      text,
      timestamp: new Date(),
      triggeredBy,
      label: labelKey
    };

    this.lastSpokenMap.set(labelKey.toLowerCase(), Date.now());

    utterance.onstart = () => {
      this.history.unshift(historyItem);
      if (this.history.length > 20) this.history.pop();
      this.notify(true, historyItem);
    };

    utterance.onend = () => {
      this.notify(false);
    };

    utterance.onerror = () => {
      this.notify(false);
    };

    try {
      this.synth.speak(utterance);
      return true;
    } catch {
      this.notify(false);
      return false;
    }
  }
}

export const speechService = new SpeechService();
