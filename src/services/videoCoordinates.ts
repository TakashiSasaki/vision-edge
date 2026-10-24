interface ClientRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

// VisionCanvas's video and overlay use object-fit: contain; object-position: 50% 50%.
// DOM client coordinates and CSS rectangles already share units (no DPR factor).
export function clientPointToVideo(
  clientX: number,
  clientY: number,
  rect: ClientRect,
  videoWidth: number,
  videoHeight: number,
): { x: number; y: number } | null {
  if (![clientX, clientY, rect.left, rect.top, rect.width, rect.height, videoWidth, videoHeight].every(Number.isFinite)
    || rect.width <= 0 || rect.height <= 0 || videoWidth <= 0 || videoHeight <= 0) return null;

  const scale = Math.min(rect.width / videoWidth, rect.height / videoHeight);
  const width = videoWidth * scale;
  const height = videoHeight * scale;
  const left = rect.left + (rect.width - width) / 2;
  const top = rect.top + (rect.height - height) / 2;
  if (![width, height, left, top].every(Number.isFinite) || width <= 0 || height <= 0) return null;
  const x = (clientX - left) / width;
  const y = (clientY - top) / height;
  if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || x > 1 || y < 0 || y > 1) return null;
  return { x, y };
}
