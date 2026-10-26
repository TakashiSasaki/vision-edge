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
interface ClassifierCi {
  marker: 'CLASSIFIER_CI_ONLY'; creates: string[]; calls: number; closes: number;
  failCreates: number; failRuns: number; gate?: Promise<void>;
}
declare global { interface Window { classifierCi: ClassifierCi } }
const classifierCi: ClassifierCi = window.classifierCi = { marker: 'CLASSIFIER_CI_ONLY', creates: [], calls: 0, closes: 0, failCreates: 0, failRuns: 0 };
export const ImageClassifier = { createFromOptions: async (_fileset: unknown, options: { baseOptions: { modelAssetPath: string; delegate: string }; runningMode: string }) => {
  const variant = options.baseOptions.modelAssetPath.includes('efficientnet_lite2') ? 'lite2' : 'lite0';
  classifierCi.creates.push(`${variant}:${options.baseOptions.delegate}`);
  const gate = classifierCi.gate; if (gate) await gate;
  if (classifierCi.failCreates > 0) { classifierCi.failCreates--; throw Error('CLASSIFIER_CI_ONLY initialization failure'); }
  return { close() { classifierCi.closes++; }, classifyForVideo(_input: HTMLVideoElement | VideoFrame | HTMLCanvasElement, timestampMs: number) {
    classifierCi.calls++;
    if (classifierCi.failRuns > 0) { classifierCi.failRuns--; throw Error('CLASSIFIER_CI_ONLY run failure'); }
    const label = variant === 'lite2' ? 'coffee mug' : 'goldfish, Carassius auratus';
    return { timestampMs, classifications: [{ headIndex: 0, headName: 'CI model-free head', categories: [
      { index: 1000, categoryName: 'UNKNOWN, Exact Raw Label', displayName: '', score: .7 },
      { index: 7, categoryName: label, displayName: '', score: .95 },
      { index: 0, categoryName: 'banana', displayName: '', score: .8 },
      { index: 3, categoryName: 'volcano', displayName: '', score: .6 },
      { index: 8, categoryName: 'soccer ball', displayName: '', score: .5 },
      { index: 2, categoryName: 'orange', displayName: '', score: .1 },
    ] }] };
  } };
} };
