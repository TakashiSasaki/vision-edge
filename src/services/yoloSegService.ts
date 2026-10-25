import * as ort from 'onnxruntime-web';
import type { DetectedObject, InferenceInput, OnnxBackendInfo, YoloSegVariantId } from '../types/vision';
import { OnnxSessionOwner, OnnxInferenceError, createOrtSession, disposeTensors, unknownOnnxBackend } from './onnxSession';
import { getCategoryColor, getLabelInfo } from './dictionary';
import { resolveModelAssetUrls } from './modelAssetResolver';

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
  close: () => void | Promise<void>;
  readonly backend?: OnnxBackendInfo;
}

export interface YoloSegTaskOptions {
  modelUrl?: string;
  variantId?: YoloSegVariantId;
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
  private readonly owner: OnnxSessionOwner;

  constructor(private readonly session: ort.InferenceSession, public readonly backend: OnnxBackendInfo = unknownOnnxBackend()) {this.owner=new OnnxSessionOwner(session,'YOLO-seg');}

  public close(): Promise<void> {
    this.inputCanvas=null;this.inputCtx=null;return this.owner.close();
  }

  public async detectForVideo(
    input: InferenceInput,
    confidenceThreshold: number
  ): Promise<DetectedObject[]> {
    return this.owner.use(async()=>{

    // 1. Prepare 640x480 input canvas and get normalized tensor
    if (!this.inputCanvas) {
      this.inputCanvas = document.createElement('canvas');
      this.inputCanvas.width = TARGET_INPUT_WIDTH;
      this.inputCanvas.height = TARGET_INPUT_HEIGHT;
      this.inputCtx = this.inputCanvas.getContext('2d', { willReadFrequently: true });
    }
    const ctx = this.inputCtx;
    if (!ctx) throw new OnnxInferenceError('YOLO-seg','input','2D context unavailable');

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

    const owned:ort.Tensor[]=[inputTensor];
    try {
    // 2. Run inference
    let results: Record<string, ort.Tensor>;
    try {
      const feeds: Record<string, ort.Tensor> = { [inputName]: inputTensor };
      results = await this.session.run(feeds);
    } catch (err) {
      throw new OnnxInferenceError('YOLO-seg','run',err);
    }
    owned.push(...Object.values(results));
    if (this.owner.closed) return [];

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
      throw new OnnxInferenceError('YOLO-seg','output','missing detection/prototype tensors');
    }

    const detData = detTensor.data as Float32Array;
    const protoData = protoTensor.data as Float32Array;
    const detDims = detTensor.dims; // e.g. [1, 116, 8400] or [1, 8400, 116]

    if(detDims.length!==3||detDims[0]!==1||!detDims.every(n=>Number.isSafeInteger(n)&&n>=0)||!(detDims[1]===116||detDims[2]===116)||protoTensor.dims.length!==4||!protoTensor.dims.every(n=>Number.isSafeInteger(n)&&n>0)||protoTensor.dims[0]!==1||protoTensor.dims[1]!==32)throw new OnnxInferenceError('YOLO-seg','output','unexpected tensor shape');
    if(detData.length!==detDims.reduce((a,b)=>a*b,1)||protoData.length!==protoTensor.dims.reduce((a,b)=>a*b,1)||!detData.every(Number.isFinite)||!protoData.every(Number.isFinite))throw new OnnxInferenceError('YOLO-seg','output','invalid tensor data');
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

      if(w<=0||h<=0)continue;
      const clamp=(n:number)=>Math.max(0,Math.min(1,n));
      const normX=clamp((cx-w/2)/TARGET_INPUT_WIDTH),right=clamp((cx+w/2)/TARGET_INPUT_WIDTH);
      const normY=clamp((cy-h/2)/TARGET_INPUT_HEIGHT),bottom=clamp((cy+h/2)/TARGET_INPUT_HEIGHT);
      const normW=right-normX,normH=bottom-normY;

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
        const maxPx = Math.min(protoW, Math.ceil((item.bbox.x + item.bbox.width) * protoW));
        const minPy = Math.max(0, Math.floor(item.bbox.y * protoH));
        const maxPy = Math.min(protoH, Math.ceil((item.bbox.y + item.bbox.height) * protoH));

        for (let py = minPy; py < maxPy; py++) {
          const rowOffset = py * protoW;
          for (let px = minPx; px < maxPx; px++) {
            const x=(px+.5)/protoW,y=(py+.5)/protoH;
            if(x<item.bbox.x||x>=item.bbox.x+item.bbox.width||y<item.bbox.y||y>=item.bbox.y+item.bbox.height)continue;
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

    return detectedObjects;
    } finally {disposeTensors(owned);}
    });
  }
}

export async function createYoloSegTask(options: YoloSegTaskOptions = {}): Promise<YoloSegTask> {
  if (options.modelUrl) {
    const { session, backend } = await createOrtSession(options.modelUrl, options.delegate, {
      modelId: 'yolo-seg',
      primarySource: 'external',
    });
    return new YoloSegEngine(session, backend);
  }
  const resolved = await resolveModelAssetUrls('yolo-seg', options.variantId);
  const { session, backend } = await createOrtSession(
    resolved.primaryUrl || DEFAULT_YOLO_SEG_URL,
    options.delegate,
    {
      fallbackModelUrl: resolved.fallbackUrl,
      modelId: 'yolo-seg',
      primarySource: resolved.source,
    }
  );
  return new YoloSegEngine(session, backend);
}
