import { describe, expect, it } from 'vitest';
import { computeSadDisparity, renderDisparityColormapToRgba } from './stereoDepthWorker';

const w = 128, h = 64;
function texture(x: number, y: number) {
  let v = x + 1 + Math.imul(y + 1, 4099);
  v = Math.imul(v ^ (v >>> 16), 0x45d9f3b);
  v = Math.imul(v ^ (v >>> 16), 0x45d9f3b);
  return (v ^ (v >>> 16)) & 255;
}
function frame(pattern: (x: number, y: number) => number) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = pattern(x, y);
    data.set([v, v, v, 255], (y * w + x) * 4);
  }
  return data;
}
const options = { maxDisparity: 24, windowSize: 7 as const, confidenceThreshold: 0.35 };
describe('stereo match validity', () => {
  for (const kind of ['flat', 'periodic']) {
    it(`rejects ${kind} ambiguity instead of painting false depth`, () => {
      const left = frame(x => kind === 'flat' ? 128 : x % 8 < 4 ? 40 : 210);
      const out = computeSadDisparity(left, left, w, h, options);
      for (let y = 8; y < h - 8; y++) for (let x = 32; x < w - 8; x++) {
        expect(out.disparity[y * w + x]).toBe(-1);
      }
      const rgba = renderDisparityColormapToRgba(out.disparity, w, h, 0, 24, 'turbo');
      expect(rgba[(32 * w + 64) * 4 + 3]).toBe(0);
    });
  }
  it('retains accurate unambiguous shifts and rejects unsupported borders', () => {
    const out = computeSadDisparity(frame(texture), frame((x, y) => texture(x + 8, y)), w, h, options);
    let valid = 0, tested = 0;
    for (let y = 8; y < h - 8; y++) for (let x = 32; x < w - 8; x++) {
      tested++;
      const d = out.disparity[y * w + x];
      if (d >= 0) { valid++; expect(Math.abs(d - 8)).toBeLessThan(0.5); }
    }
    expect(valid / tested).toBeGreaterThan(0.98);
    for (let y = 0; y < h; y++) for (let x = 0; x < 4; x++) expect(out.disparity[y * w + x]).toBe(-1);
  });
  it('does not relax confidence to fill an unrelated pair with matches', () => {
    const out = computeSadDisparity(frame(texture), frame((x, y) => texture(x + 999, y + 333)), w, h,
      { ...options, confidenceThreshold: 1 });
    expect(out.validRatio).toBeLessThan(0.01);
  });
});
