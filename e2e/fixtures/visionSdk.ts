// Only Vite's explicit --mode e2e aliases the SDK to this module.
// Production App, VisionEngine, scheduler, metrics and camera hook are unchanged.
interface SdkProbe {
  calls: number; failures: number; creates: number; empty: boolean;
  observeInputs: boolean;
  inputs: Array<{ sourceKind: 'video-frame' | 'live-video' | 'canvas'; time: number | null; qualityFrames: number | null; pixels: number[] }>;
  afterInput?: (video: HTMLVideoElement) => void;
}
declare global { interface Window { visionSdkProbe: SdkProbe; visionSdkGate?: Promise<void>; visionSdkObserverFrame?: typeof VideoFrame } }
const probe: SdkProbe = window.visionSdkProbe = { calls: 0, failures: 0, creates: 0, empty: false, observeInputs: false, inputs: [] };
function execute(input: HTMLVideoElement | VideoFrame | HTMLCanvasElement) {
  const video = document.querySelector('video')!;
  if (probe.observeInputs) {
    const canvas = document.createElement('canvas');
    canvas.width = 'videoWidth' in input ? input.videoWidth : 'displayWidth' in input ? input.displayWidth : input.width;
    canvas.height = 'videoHeight' in input ? input.videoHeight : 'displayHeight' in input ? input.displayHeight : input.height;
    const context = canvas.getContext('2d', { willReadFrequently: true })!;
    const Frame = window.visionSdkObserverFrame ?? VideoFrame;
    const isCanvas = 'getContext' in input;
    const observed = 'timestamp' in input ? input : new Frame(input, isCanvas ? { timestamp: 0 } : undefined);
    try {
      context.drawImage(observed, 0, 0);
      probe.inputs.push({ sourceKind: 'timestamp' in input ? 'video-frame' : isCanvas ? 'canvas' : 'live-video', time: isCanvas ? null : observed.timestamp / 1000000, qualityFrames: video.getVideoPlaybackQuality?.()?.totalVideoFrames ?? null,
        pixels: Array.from(context.getImageData(0, 0, canvas.width, canvas.height).data) });
    } finally { if (observed !== input) observed.close(); }
    if (probe.inputs.length > 16) probe.inputs.shift();
    probe.afterInput?.(video);
  }
  probe.calls++;
  if (probe.failures > 0) { probe.failures--; throw new Error('Test-only SDK fault'); }
  // Deliberately observable synchronous SDK work; metrics include post-processing too.
  const until = performance.now() + 1;
  while (performance.now() < until) { /* test work */ }
}
export const FilesetResolver = { forVisionTasks: async () => { await window.visionSdkGate; return {}; } };
export const ObjectDetector = { createFromOptions: async () => {
  probe.creates++;
  return { close() {}, detectForVideo(video: HTMLVideoElement | VideoFrame | HTMLCanvasElement) {
    execute(video);
    return { detections: probe.empty ? [] : [{ categories: [{ categoryName: 'dog', score: 0.9 }], boundingBox: { originX: 64, originY: 48, width: 128, height: 96 } }] };
  } };
} };
export const ImageSegmenter = { createFromOptions: async () => {
  probe.creates++;
  return { close() {}, segmentForVideo(video: HTMLVideoElement | VideoFrame | HTMLCanvasElement) { execute(video); return { confidenceMasks: [], close() {} }; } };
} };
