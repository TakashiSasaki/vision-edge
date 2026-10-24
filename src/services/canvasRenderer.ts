import type { DetectionFrame, DetectedObject } from '../types/vision';

// The synchronous Canvas work shared by the UI and instrumentation experiment.
// True means a result (including an empty one) was copied to the visible Canvas.
export function drawCanvasFrame(
  canvas: HTMLCanvasElement,
  video: Pick<HTMLVideoElement, 'videoWidth' | 'videoHeight' | 'readyState'>,
  frame: DetectionFrame | null,
  buffer: { current: HTMLCanvasElement | null },
  overlays: (ctx: CanvasRenderingContext2D, width: number, height: number, objects: DetectedObject[]) => void,
  current: () => boolean,
): boolean {
  if (video.videoWidth > 0 && canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
  if (video.videoHeight > 0 && canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Overlay context unavailable');
  if (frame && video.readyState >= 2) {
    const offscreen = buffer.current ??= document.createElement('canvas');
    // Preserve context reset and the existing native backing-coordinate contract.
    offscreen.width = canvas.width; offscreen.height = canvas.height;
    const offCtx = offscreen.getContext('2d');
    if (!offCtx) throw new Error('Overlay buffer unavailable');
    overlays(offCtx, offscreen.width, offscreen.height, frame.objects);
    if (!current()) return false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(offscreen, 0, 0);
    return true;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  return false;
}
