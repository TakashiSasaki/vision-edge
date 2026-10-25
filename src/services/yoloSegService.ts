import * as ort from 'onnxruntime-web';
import type { DetectedObject, InferenceInput } from '../types/vision';
import { getCategoryColor, getLabelInfo } from './dictionary';

export const COCO_CLASSES: readonly string[] = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
  'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
  'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
  'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
] as const;

export interface YoloSegTask {
  detectForVideo: (input: InferenceInput, confidenceThreshold: number) => Promise<DetectedObject[]> | DetectedObject[];
  close: () => void;
}

export interface YoloSegTaskOptions {
  modelUrl?: string;
  delegate?: 'GPU' | 'WebGPU' | 'WASM';
}

const DEFAULT_YOLO_SEG_URL = 'https://huggingface.co/unileon-robotics/YOLO8-ONNX/resolve/main/yolov8n-seg.onnx';
const TARGET_INPUT_WIDTH = 640;
const TARGET_INPUT_HEIGHT = 480;
const NUM_MASK_COEFFS = 32;

function parseHexColor(hex: string): [number, number, number] {
  const clean = hex.replace('#', '');
  if (clean.length === 6) {
    const r = parseInt(clean.substring(0, 2), 16);
    const g = parseInt(clean.substring(2, 4), 16);
    const b = parseInt(clean.substring(4, 6), 16);
    return [r, g, b];
  }
  return [56, 189, 248]; // default sky blue
}

function calculateIoU(
  boxA: { x: number; y: number; width: number; height: number },
  boxB: { x: number; y: number; width: number; height: number }
): number {
  const xA = Math.max(boxA.x, boxB.x);
  const yA = Math.max(boxA.y, boxB.y);
  const xB = Math.min(boxA.x + boxA.width, boxB.x + boxB.width);
  const yB = Math.min(boxA.y + boxA.height, boxB.y + boxB.height);

  const interWidth = Math.max(0, xB - xA);
  const interHeight = Math.max(0, yB - yA);
  const interArea = interWidth * interHeight;

  const areaA = boxA.width * boxA.height;
  const areaB = boxB.width * boxB.height;
  const unionArea = areaA + areaB - interArea;

  return unionArea <= 0 ? 0 : interArea / unionArea;
}

export class YoloSegEngine implements YoloSegTask {
  private inputCanvas: HTMLCanvasElement | null = null;
  private inputCtx: CanvasRenderingContext2D | null = null;
  private isClosed = false;

  constructor(private readonly session: ort.InferenceSession) {}

  public close(): void {
    this.isClosed = true;
    try {
      void this.session.release();
    } catch (cause) {
      console.warn('YOLO-seg session release error', cause);
    }
    this.inputCanvas = null;
    this.inputCtx = null;
  }

  public async detectForVideo(
    input: InferenceInput,
    confidenceThreshold: number
  ): Promise<DetectedObject[]> {
    if (this.isClosed) return [];

    // 1. Prepare 640x480 input canvas and get normalized tensor
    if (!this.inputCanvas) {
      this.inputCanvas = document.createElement('canvas');
      this.inputCanvas.width = TARGET_INPUT_WIDTH;
      this.inputCanvas.height = TARGET_INPUT_HEIGHT;
      this.inputCtx = this.inputCanvas.getContext('2d', { willReadFrequently: true });
    }
    const ctx = this.inputCtx;
    if (!ctx) return [];

    ctx.drawImage(input, 0, 0, TARGET_INPUT_WIDTH, TARGET_INPUT_HEIGHT);
    const imgData = ctx.getImageData(0, 0, TARGET_INPUT_WIDTH, TARGET_INPUT_HEIGHT);
    const { data } = imgData;

    // Convert RGBA to planar RGB Float32Array [1, 3, 480, 640] normalized to [0, 1]
    const pixels = TARGET_INPUT_WIDTH * TARGET_INPUT_HEIGHT;
    const float32Data = new Float32Array(3 * pixels);
    for (let i = 0; i < pixels; i++) {
      float32Data[i] = data[i * 4] / 255.0;                   // R channel
      float32Data[pixels + i] = data[i * 4 + 1] / 255.0;       // G channel
      float32Data[2 * pixels + i] = data[i * 4 + 2] / 255.0;   // B channel
    }

    const inputName = this.session.inputNames[0] || 'images';
    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, TARGET_INPUT_HEIGHT, TARGET_INPUT_WIDTH]);

    // 2. Run inference
    let results: Record<string, ort.Tensor>;
    try {
      const feeds: Record<string, ort.Tensor> = { [inputName]: inputTensor };
      results = await this.session.run(feeds);
    } catch (err) {
      console.error('[YOLO-seg] session.run failed:', err);
      return [];
    }
    if (this.isClosed) return [];

    // 3. Identify output0 (boxes & mask coefficients) and output1 (proto masks)
    const outputNames = this.session.outputNames;
    let detTensor: ort.Tensor | null = null;
    let protoTensor: ort.Tensor | null = null;

    for (const name of outputNames) {
      const tensor = results[name];
      if (!tensor) continue;
      const dims = tensor.dims;
      if (dims.length === 3 && (dims[1] === 116 || dims[2] === 116)) {
        detTensor = tensor;
      } else if (dims.length === 4 && dims[1] === NUM_MASK_COEFFS) {
        protoTensor = tensor;
      }
    }

    // Fallback detection if shapes differ slightly
    if (!detTensor && outputNames.length >= 1) detTensor = results[outputNames[0]];
    if (!protoTensor && outputNames.length >= 2) protoTensor = results[outputNames[1]];

    if (!detTensor || !protoTensor) {
      return [];
    }

    const detData = detTensor.data as Float32Array;
    const protoData = protoTensor.data as Float32Array;
    const detDims = detTensor.dims; // e.g. [1, 116, 8400] or [1, 8400, 116]

    const isChannelsFirst = detDims[1] === 116;
    const numCandidates = isChannelsFirst ? detDims[2] : detDims[1];

    interface Candidate {
      classId: number;
      className: string;
      score: number;
      bbox: { x: number; y: number; width: number; height: number };
      coeffs: Float32Array;
    }

    const threshold = Math.max(0.15, confidenceThreshold);
    const candidates: Candidate[] = [];

    // Extract boxes above confidence threshold
    for (let i = 0; i < numCandidates; i++) {
      let cx: number;
      let cy: number;
      let w: number;
      let h: number;
      let maxScore = 0;
      let bestClass = 0;

      if (isChannelsFirst) {
        // detData layout: [116, 8400]
        cx = detData[0 * numCandidates + i];
        cy = detData[1 * numCandidates + i];
        w = detData[2 * numCandidates + i];
        h = detData[3 * numCandidates + i];

        for (let c = 0; c < 80; c++) {
          const score = detData[(4 + c) * numCandidates + i];
          if (score > maxScore) {
            maxScore = score;
            bestClass = c;
          }
        }
      } else {
        // detData layout: [8400, 116]
        const offset = i * 116;
        cx = detData[offset];
        cy = detData[offset + 1];
        w = detData[offset + 2];
        h = detData[offset + 3];

        for (let c = 0; c < 80; c++) {
          const score = detData[offset + 4 + c];
          if (score > maxScore) {
            maxScore = score;
            bestClass = c;
          }
        }
      }

      if (maxScore < threshold) continue;

      const normX = Math.max(0, (cx - w / 2) / TARGET_INPUT_WIDTH);
      const normY = Math.max(0, (cy - h / 2) / TARGET_INPUT_HEIGHT);
      const normW = Math.min(1 - normX, w / TARGET_INPUT_WIDTH);
      const normH = Math.min(1 - normY, h / TARGET_INPUT_HEIGHT);

      if (normW <= 0.01 || normH <= 0.01) continue;

      const coeffs = new Float32Array(NUM_MASK_COEFFS);
      if (isChannelsFirst) {
        for (let m = 0; m < NUM_MASK_COEFFS; m++) {
          coeffs[m] = detData[(84 + m) * numCandidates + i];
        }
      } else {
        const offset = i * 116 + 84;
        for (let m = 0; m < NUM_MASK_COEFFS; m++) {
          coeffs[m] = detData[offset + m];
        }
      }

      candidates.push({
        classId: bestClass,
        className: COCO_CLASSES[bestClass] || `class-${bestClass}`,
        score: maxScore,
        bbox: { x: normX, y: normY, width: normW, height: normH },
        coeffs
      });
    }

    // Sort descending by score
    candidates.sort((a, b) => b.score - a.score);

    // Non-Maximum Suppression (IoU >= 0.45)
    const selected: Candidate[] = [];
    const IOU_THRESHOLD = 0.45;
    const MAX_DETECTIONS = 20;

    for (const cand of candidates) {
      if (selected.length >= MAX_DETECTIONS) break;
      let overlap = false;
      for (const sel of selected) {
        if (cand.classId === sel.classId && calculateIoU(cand.bbox, sel.bbox) > IOU_THRESHOLD) {
          overlap = true;
          break;
        }
      }
      if (!overlap) {
        selected.push(cand);
      }
    }

    // 4. Generate Instance Segmentation Masks using Prototype Masks
    const detectedObjects: DetectedObject[] = [];
    const protoDims = protoTensor.dims; // [1, 32, protoH, protoW] (e.g. [1, 32, 120, 160])
    const protoH = protoDims.length >= 4 ? protoDims[2] : 120;
    const protoW = protoDims.length >= 4 ? protoDims[3] : 160;
    const protoArea = protoW * protoH;

    for (let idx = 0; idx < selected.length; idx++) {
      const item = selected[idx];
      const colorHex = getCategoryColor(item.className);
      const [cr, cg, cb] = parseHexColor(colorHex);

      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = protoW;
      maskCanvas.height = protoH;
      const maskCtx = maskCanvas.getContext('2d');

      if (maskCtx) {
        const maskImgData = maskCtx.createImageData(protoW, protoH);
        const mPixels = maskImgData.data;

        // Bounding box bounds in proto coordinates
        const minPx = Math.max(0, Math.floor(item.bbox.x * protoW));
        const maxPx = Math.min(protoW - 1, Math.ceil((item.bbox.x + item.bbox.width) * protoW));
        const minPy = Math.max(0, Math.floor(item.bbox.y * protoH));
        const maxPy = Math.min(protoH - 1, Math.ceil((item.bbox.y + item.bbox.height) * protoH));

        for (let py = minPy; py <= maxPy; py++) {
          const rowOffset = py * protoW;
          for (let px = minPx; px <= maxPx; px++) {
            let logit = 0;
            const pIdx = rowOffset + px;

            // Dot product between 32 coefficients and 32 prototype masks at (px, py)
            for (let m = 0; m < NUM_MASK_COEFFS; m++) {
              logit += item.coeffs[m] * protoData[m * protoArea + pIdx];
            }

            // Sigmoid activation
            const sig = 1 / (1 + Math.exp(-logit));
            if (sig > 0.5) {
              const pixelIdx = pIdx * 4;
              mPixels[pixelIdx] = cr;
              mPixels[pixelIdx + 1] = cg;
              mPixels[pixelIdx + 2] = cb;
              mPixels[pixelIdx + 3] = Math.min(240, Math.floor(sig * 220));
            }
          }
        }

        maskCtx.putImageData(maskImgData, 0, 0);
      }

      const labelInfo = getLabelInfo(item.className);

      detectedObjects.push({
        id: `yolo-${item.classId}-${idx}-${Math.round(item.score * 100)}`,
        label: item.className,
        labelJa: labelInfo.ja,
        category: labelInfo.category,
        confidence: Math.min(0.99, item.score),
        bbox: item.bbox,
        color: colorHex,
        maskCanvas
      });
    }

    if (detectedObjects.length > 0) {
      console.log(`[YOLO-seg] Detected ${detectedObjects.length} object(s):`,
        detectedObjects.map(o => `${o.labelJa} (${Math.round(o.confidence * 100)}%)`).join(', '));
    }

    return detectedObjects;
  }
}

export async function createYoloSegTask(options: YoloSegTaskOptions = {}): Promise<YoloSegTask> {
  const modelUrl = options.modelUrl || DEFAULT_YOLO_SEG_URL;

  if (typeof window !== 'undefined') {
    ort.env.logLevel = 'error';
    const isIsolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
    ort.env.wasm.numThreads = isIsolated ? Math.min(4, Math.max(1, (navigator.hardwareConcurrency || 2) - 1)) : 1;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
  }

  // Choose WebGPU if available and not explicitly disabled
  const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;
  const executionProviders = options.delegate === 'WASM'
    ? ['wasm']
    : hasWebGPU
    ? ['webgpu', 'wasm']
    : ['wasm'];

  const sessionOptions: ort.InferenceSession.SessionOptions = {
    executionProviders,
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3,
  };

  try {
    const session = await ort.InferenceSession.create(modelUrl, sessionOptions);
    return new YoloSegEngine(session);
  } catch (error) {
    // If WebGPU failed, fallback to WASM directly
    if (executionProviders.includes('webgpu')) {
      try {
        const fallbackSession = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
          logSeverityLevel: 3,
        });
        return new YoloSegEngine(fallbackSession);
      } catch (fallbackError) {
        throw new Error(`Failed to create YOLO-seg session with WASM fallback: ${String(fallbackError)}`, {
          cause: fallbackError,
        });
      }
    }
    throw new Error(`Failed to initialize YOLO-seg model: ${String(error)}`, { cause: error });
  }
}
