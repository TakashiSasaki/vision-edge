import type { Stamp } from './acceptance';
export function inputStamp(input: CanvasImageSource): Stamp | null {
  try {
    const canvas = document.createElement('canvas'); canvas.width = 184; canvas.height = 1;
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(input, 0, 508, 184, 1, 0, 0, 184, 1);
    const pixels = ctx.getImageData(0, 0, 184, 1).data; let code = 0;
    for (let bit = 0; bit < 23; bit++) if (pixels[(bit * 8 + 4) * 4] > 127) code |= 1 << bit;
    if (code >>> 19 !== 10) return null;
    const scene = ['', 'astronaut.png', 'coffee.png', 'background', 'empty'][code & 7];
    return scene ? { scene, revision: (code >>> 3) & 255, stream: (code >>> 11) & 255 } : null;
  } catch { return null; } // Missing provenance cannot satisfy acceptance.
}
