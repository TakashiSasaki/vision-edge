import { describe, expect, it } from 'vitest';
import { getCategoryColor, getLabelInfo } from './dictionary';

describe('getLabelInfo', () => {
  it('translates known COCO labels and normalizes whitespace and case', () => {
    expect(getLabelInfo(' DOG ')).toEqual({ ja: '犬', category: '動物' });
    expect(getLabelInfo('cell phone')).toEqual({ ja: 'スマートフォン', category: '電子機器' });
  });

  it('uses ADE20K labels when requested, with COCO fallback', () => {
    expect(getLabelInfo('sky', true)).toEqual({ ja: '空', category: '屋外' });
    expect(getLabelInfo('bicycle', true)).toEqual({ ja: '自転車', category: '乗り物' });
  });

  it('provides a readable fallback for unknown and empty labels', () => {
    expect(getLabelInfo(' QUANTUM WIDGET ')).toEqual({ ja: 'Quantum widget', category: 'その他' });
    expect(getLabelInfo('')).toEqual({ ja: '', category: 'その他' });
  });
});

describe('getCategoryColor', () => {
  it('returns stable hex colors across repeated and reordered calls', () => {
    const labels = ['person', 'dog', 'sky', '人物', '', 'quantum widget'];
    const colors = new Map(labels.map(label => [label, getCategoryColor(label)]));
    for (const label of [...labels].reverse()) {
      expect(getCategoryColor(label)).toBe(colors.get(label));
      expect(getCategoryColor(label)).toMatch(/^#[0-9A-F]{6}$/);
    }
    expect(new Set(colors.values()).size).toBeGreaterThan(1);
  });
});
