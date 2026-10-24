import { afterEach, describe, expect, it, vi } from 'vitest';
import { clientPointToVideo } from './videoCoordinates';

afterEach(() => vi.unstubAllGlobals());

// Fixed reference rectangles/coordinates, independently worked out in CSS pixels.
const cases = [
  { name: '16:9 to 16:9', video: [1280, 720], box: [960, 540], content: [0, 0, 960, 540] },
  { name: '4:3 to 16:9', video: [640, 480], box: [960, 540], content: [120, 0, 720, 540] },
  { name: '16:9 to 4:3', video: [1280, 720], box: [800, 600], content: [0, 75, 800, 450] },
  { name: 'square to wide', video: [600, 600], box: [960, 540], content: [210, 0, 540, 540] },
];
for (const c of cases) {
  describe(c.name, () => {
    const [left, top, width, height] = c.content;
    const rect = { left: 35, top: 67, width: c.box[0], height: c.box[1] };
    const map = (x: number, y: number) => clientPointToVideo(x + 35, y + 67, rect, c.video[0], c.video[1]);
    it.each([
      [left, top, 0, 0], [left + width, top, 1, 0],
      [left, top + height, 0, 1], [left + width, top + height, 1, 1],
      [left + width / 2, top + height / 2, 0.5, 0.5],
      [left + width / 4, top + height * 0.75, 0.25, 0.75],
    ])('maps reference (%s, %s) to (%s, %s)', (x, y, nx, ny) => {
      expect(map(x, y)).toEqual({ x: nx, y: ny });
    });
    it('accepts just inside and excludes just outside every edge', () => {
      const epsilon = 0.001;
      expect(map(left + epsilon, top + height / 2)?.x).toBeCloseTo(epsilon / width, 12);
      expect(map(left + width - epsilon, top + height / 2)?.x).toBeCloseTo(1 - epsilon / width, 12);
      expect(map(left + width / 2, top + epsilon)?.y).toBeCloseTo(epsilon / height, 12);
      expect(map(left + width / 2, top + height - epsilon)?.y).toBeCloseTo(1 - epsilon / height, 12);
      expect(map(left - epsilon, top + height / 2)).toBeNull();
      expect(map(left + width + epsilon, top + height / 2)).toBeNull();
      expect(map(left + width / 2, top - epsilon)).toBeNull();
      expect(map(left + width / 2, top + height + epsilon)).toBeNull();
    });
  });
}

it('uses the independent 640x480 in 960x540 numeric reference and rejects both horizontal margins', () => {
  const rect = { left: 0, top: 0, width: 960, height: 540 };
  expect(clientPointToVideo(300, 405, rect, 640, 480)).toEqual({ x: 0.25, y: 0.75 });
  expect(clientPointToVideo(119.999, 270, rect, 640, 480)).toBeNull();
  expect(clientPointToVideo(840.001, 270, rect, 640, 480)).toBeNull();
});

it('retains fractional CSS pixels at DPR 1, 1.25, 2 and 3 without using backing pixels', () => {
  // 4:3 input, 960.5x540.5 box: content 720.6666667x540.5, left inset 119.9166667.
  for (const dpr of [1, 1.25, 2, 3]) {
    vi.stubGlobal('devicePixelRatio', dpr);
    const point = clientPointToVideo(310.5833333333333, 182.125, { left: 10.5, top: 47, width: 960.5, height: 540.5 }, 640, 480);
    expect(point?.x).toBeCloseTo(0.25, 12); expect(point?.y).toBe(0.25);
  }
});

it('shares the client viewport origin with page offsets and scrolled negative rect positions', () => {
  expect(clientPointToVideo(800, 655, { left: 500, top: 250, width: 960, height: 540 }, 640, 480)).toEqual({ x: 0.25, y: 0.75 });
  expect(clientPointToVideo(280, 155, { left: -20, top: -250, width: 960, height: 540 }, 640, 480)).toEqual({ x: 0.25, y: 0.75 });
});

it.each([0, -1, NaN, Infinity, -Infinity])('rejects invalid dimensions %s for every dimension', bad => {
  for (const key of ['width', 'height']) {
    expect(clientPointToVideo(50, 50, { left: 0, top: 0, width: 100, height: 100, [key]: bad }, 640, 480)).toBeNull();
  }
  expect(clientPointToVideo(50, 50, { left: 0, top: 0, width: 100, height: 100 }, bad, 480)).toBeNull();
  expect(clientPointToVideo(50, 50, { left: 0, top: 0, width: 100, height: 100 }, 640, bad)).toBeNull();
});
it.each([NaN, Infinity, -Infinity])('rejects nonfinite pointer/rectangle origins %s', bad => {
  const rect = { left: 0, top: 0, width: 100, height: 100 };
  expect(clientPointToVideo(bad, 50, rect, 640, 480)).toBeNull();
  expect(clientPointToVideo(50, bad, rect, 640, 480)).toBeNull();
  expect(clientPointToVideo(50, 50, { ...rect, left: bad }, 640, 480)).toBeNull();
  expect(clientPointToVideo(50, 50, { ...rect, top: bad }, 640, 480)).toBeNull();
});

it('rejects finite inputs whose calculated content underflows', () => {
  expect(clientPointToVideo(0, 0, { left: 0, top: 0, width: Number.MIN_VALUE, height: Number.MIN_VALUE }, Number.MAX_VALUE, 1)).toBeNull();
});

// PR05-F01: fixed exact-rational edges, rounded only at the final division.
// These values do not come from the production geometry implementation.
const fractionalEdges = [
  { left: 37.25, top: 82.5, lo: 3779 / 24, hi: 21067 / 24 },
  { left: 500, top: 250, lo: 14885 / 24, hi: 32173 / 24 },
  { left: -20, top: -250, lo: 2405 / 24, hi: 19693 / 24 },
];
function adjacentFloat(value: number, direction: -1 | 1): number {
  if (value === 0) return direction * Number.MIN_VALUE;
  const bits = new DataView(new ArrayBuffer(8));
  bits.setFloat64(0, value);
  bits.setBigUint64(0, bits.getBigUint64(0) + BigInt((value > 0) === (direction > 0) ? 1 : -1));
  return bits.getFloat64(0);
}
for (const transpose of [false, true]) {
  for (const dpr of [1, 1.25, 2, 3]) {
    it.each(fractionalEdges)(`accepts rational edges and rejects the next outside float (transpose=${transpose}, DPR=${dpr}, origin=$left)`, c => {
      vi.stubGlobal('devicePixelRatio', dpr);
      const rect = transpose ? { left: c.top, top: c.left, width: 540.25, height: 960.75 }
        : { left: c.left, top: c.top, width: 960.75, height: 540.25 };
      const point = (x: number, y: number) => transpose
        ? clientPointToVideo(y, x, rect, 480, 640) : clientPointToVideo(x, y, rect, 640, 480);
      for (const [x, nx] of [[c.lo, 0], [c.hi, 1]]) {
        for (const [y, ny] of [[c.top, 0], [c.top + 540.25, 1], [c.top + 270.125, 0.5]]) {
          expect(point(x, y)).toEqual(transpose ? { x: ny, y: nx } : { x: nx, y: ny });
        }
      }
      const middleX = (c.lo + c.hi) / 2;
      const middleY = c.top + 270.125;
      for (const [edge, direction] of [[c.lo, -1], [c.hi, 1]] as const) {
        expect(point(adjacentFloat(edge, direction), middleY)).toBeNull();
        expect(point(adjacentFloat(edge, direction === -1 ? 1 : -1), middleY)).not.toBeNull();
        for (const distance of [1e-12, 0.001]) expect(point(edge + direction * distance, middleY)).toBeNull();
      }
      for (const [edge, direction] of [[c.top, -1], [c.top + 540.25, 1]] as const) {
        expect(point(middleX, adjacentFloat(edge, direction))).toBeNull();
        expect(point(middleX, adjacentFloat(edge, direction === -1 ? 1 : -1))).not.toBeNull();
      }
    });
  }
}
