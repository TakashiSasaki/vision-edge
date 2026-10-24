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

  // Compare client coordinates to content edges before normalizing. Form each
  // contained edge as a single quotient, rather than centering an already
  // rounded scaled size (which can move an inclusive edge by one float).
  const containedEdges = (origin: number, box: number, otherBox: number, source: number, otherSource: number) => {
    const center = origin * otherSource + box * otherSource / 2;
    const halfContent = otherBox * source / 2;
    return [(center - halfContent) / otherSource, (center + halfContent) / otherSource];
  };
  const widthFits = rect.width / videoWidth <= rect.height / videoHeight;
  const [left, right] = widthFits ? [rect.left, rect.left + rect.width]
    : containedEdges(rect.left, rect.width, rect.height, videoWidth, videoHeight);
  const [top, bottom] = widthFits ? containedEdges(rect.top, rect.height, rect.width, videoHeight, videoWidth)
    : [rect.top, rect.top + rect.height];
  const width = right - left;
  const height = bottom - top;
  if (![left, right, top, bottom, width, height].every(Number.isFinite) || width <= 0 || height <= 0
    || clientX < left || clientX > right || clientY < top || clientY > bottom) return null;
  const x = clientX === right ? 1 : (clientX - left) / width;
  const y = clientY === bottom ? 1 : (clientY - top) / height;
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y };
}
