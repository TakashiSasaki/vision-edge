// Only Vite's explicit --mode e2e aliases the SDK to this module.
// Production App, VisionEngine, scheduler, metrics and camera hook are unchanged.
interface SdkProbe { calls: number; failures: number; creates: number; empty: boolean }
declare global { interface Window { visionSdkProbe: SdkProbe } }
const probe = window.visionSdkProbe = { calls: 0, failures: 0, creates: 0, empty: false };
function execute() {
  probe.calls++;
  if (probe.failures > 0) { probe.failures--; throw new Error('Test-only SDK fault'); }
  // Deliberately observable synchronous SDK work; metrics include post-processing too.
  const until = performance.now() + 1;
  while (performance.now() < until) { /* test work */ }
}
export const FilesetResolver = { forVisionTasks: async () => ({}) };
export const ObjectDetector = { createFromOptions: async () => {
  probe.creates++;
  return { close() {}, detectForVideo() {
    execute();
    return { detections: probe.empty ? [] : [{ categories: [{ categoryName: 'dog', score: 0.9 }], boundingBox: { originX: 64, originY: 48, width: 128, height: 96 } }] };
  } };
} };
export const ImageSegmenter = { createFromOptions: async () => {
  probe.creates++;
  return { close() {}, segmentForVideo() { execute(); return { confidenceMasks: [], close() {} }; } };
} };
