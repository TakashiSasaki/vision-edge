// Human-authored translations of complete, exact SDK label strings. This is not
// an ImageNet index list and makes no claim about an uninspected model's labels.
const JAPANESE_LABELS: Readonly<Record<string, string>> = Object.freeze({
  'tench, Tinca tinca': 'テンチ（魚）',
  'goldfish, Carassius auratus': '金魚',
  'tabby, tabby cat': 'トラ猫',
  'Labrador retriever': 'ラブラドール・レトリバー',
  'coffee mug': 'コーヒーマグ',
  banana: 'バナナ', orange: 'オレンジ', espresso: 'エスプレッソ',
  'soccer ball': 'サッカーボール', volcano: '火山',
});
export function japaneseClassificationLabel(raw: string): string | null {
  return Object.hasOwn(JAPANESE_LABELS, raw) ? JAPANESE_LABELS[raw] : null;
}
