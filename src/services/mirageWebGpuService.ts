import * as ort from 'onnxruntime-web';
import type {
  BoundingBox,
  DetectedObject,
  InferenceInput,
  OnnxBackendInfo,
} from '../types/vision';
import {
  createOrtSession,
  disposeTensors,
  OnnxInferenceError,
  OnnxSessionOwner,
  unknownOnnxBackend,
} from './onnxSession';
import { getCategoryColor, getLabelInfo } from './dictionary';
import {
  resolveModelAssetUrls,
  type AssetSourceRecorder,
} from './modelAssetResolver';
import {
  buildMirageWebGpuOnnxBytes,
  MIRAGE_OPEN_VOCAB_LABELS,
  MIRAGE_WEBGPU_GRID_SIZE,
  MIRAGE_WEBGPU_INPUT_SIZE,
  MIRAGE_WEBGPU_NUM_PROTOS,
  MIRAGE_WEBGPU_PROTO_SIZE,
} from './mirageOnnxConverter';
import type {
  MirageInferExtraOptions,
  MirageTask,
} from './mirageWasmService';

export interface MirageWebGpuTaskOptions {
  recordAssetSource?: AssetSourceRecorder;
  modelUrl?: string;
  delegate?: 'GPU' | 'WebGPU' | 'WASM';
  wasmTask?: MirageTask;
}

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

function parseHexColor(hex: string): [number, number, number] {
  const clean = hex.replace('#', '');
  if (clean.length === 6) {
    const r = parseInt(clean.substring(0, 2), 16);
    const g = parseInt(clean.substring(2, 4), 16);
    const b = parseInt(clean.substring(4, 6), 16);
    return [r, g, b];
  }
  return [16, 185, 129];
}

export function calculateIoU(boxA: BoundingBox, boxB: BoundingBox): number {
  const xA = Math.max(boxA.x, boxB.x);
  const yA = Math.max(boxA.y, boxB.y);
  const xB = Math.min(boxA.x + boxA.width, boxB.x + boxB.width);
  const yB = Math.min(boxA.y + boxA.height, boxB.y + boxB.height);

  const interW = Math.max(0, xB - xA);
  const interH = Math.max(0, yB - yA);
  const interArea = interW * interH;

  const areaA = boxA.width * boxA.height;
  const areaB = boxB.width * boxB.height;
  const unionArea = areaA + areaB - interArea;
  return unionArea <= 0 ? 0 : interArea / unionArea;
}

/**
 * Formats horizontal direction (`左` / `中央` / `右`) and low-vision proximity (`近` / `中` / `遠`)
 * based on object bounding box position and screen area fraction.
 */
export function formatSpatialGuidanceTagJa(bbox: BoundingBox): string {
  const centerX = bbox.x + bbox.width / 2;
  const dir = centerX < 0.35 ? '左' : centerX > 0.65 ? '右' : '中央';
  const area = bbox.width * bbox.height;
  const bottomY = bbox.y + bbox.height;
  const dist =
    area >= 0.16 || (bottomY >= 0.85 && area >= 0.08)
      ? '近'
      : area >= 0.04
        ? '中'
        : '遠';
  return `${dir}・${dist}`;
}

function perpendicularDistance(
  px: number,
  py: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number
): number {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const magSq = dx * dx + dy * dy;
  if (magSq === 0) {
    return Math.hypot(px - x1, py - y1);
  }
  const u = ((px - x1) * dx + (py - y1) * dy) / magSq;
  const clampedU = Math.max(0, Math.min(1, u));
  const projX = x1 + clampedU * dx;
  const projY = y1 + clampedU * dy;
  return Math.hypot(px - projX, py - projY);
}

/**
 * Ramer-Douglas-Peucker polygon vertex simplification algorithm.
 */
export function ramerDouglasPeucker(
  points: ReadonlyArray<{ x: number; y: number }>,
  epsilon: number
): Array<{ x: number; y: number }> {
  if (points.length <= 2) return [...points];
  let maxDist = 0;
  let index = 0;
  const end = points.length - 1;
  for (let i = 1; i < end; i++) {
    const dist = perpendicularDistance(
      points[i].x,
      points[i].y,
      points[0].x,
      points[0].y,
      points[end].x,
      points[end].y
    );
    if (dist > maxDist) {
      maxDist = dist;
      index = i;
    }
  }
  if (maxDist > epsilon) {
    const left = ramerDouglasPeucker(points.slice(0, index + 1), epsilon);
    const right = ramerDouglasPeucker(points.slice(index), epsilon);
    return left.slice(0, -1).concat(right);
  }
  return [points[0], points[end]];
}

/**
 * Traces the external contour polygon of a binary mask and simplifies it using Ramer-Douglas-Peucker.
 */
export function traceMaskContourPolygon(
  maskGrid: Uint8Array,
  protoSize: number,
  minPx: number,
  minPy: number,
  maxPx: number,
  maxPy: number,
  bbox: BoundingBox,
  epsilon = 0.008
): Array<{ x: number; y: number }> {
  let startX = -1;
  let startY = -1;
  let activePixels = 0;

  for (let py = minPy; py < maxPy && startY === -1; py++) {
    const rowOffset = py * protoSize;
    for (let px = minPx; px < maxPx; px++) {
      if (maskGrid[rowOffset + px] > 0) {
        startX = px;
        startY = py;
        break;
      }
    }
  }

  if (startX !== -1) {
    for (let py = minPy; py < maxPy; py++) {
      const rowOffset = py * protoSize;
      for (let px = minPx; px < maxPx; px++) {
        if (maskGrid[rowOffset + px] > 0) activePixels++;
      }
    }
  }

  // If no significant mask, fall back to bounding box corners
  if (startX === -1 || activePixels < 8) {
    const x1 = bbox.x;
    const y1 = bbox.y;
    const x2 = clamp01(bbox.x + bbox.width);
    const y2 = clamp01(bbox.y + bbox.height);
    return [
      { x: Number(x1.toFixed(4)), y: Number(y1.toFixed(4)) },
      { x: Number(x2.toFixed(4)), y: Number(y1.toFixed(4)) },
      { x: Number(x2.toFixed(4)), y: Number(y2.toFixed(4)) },
      { x: Number(x1.toFixed(4)), y: Number(y2.toFixed(4)) },
    ];
  }

  // Moore-Neighbor 8-neighborhood directions clockwise
  const DIRS = [
    [-1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1],
  ];

  const rawPoints: Array<{ x: number; y: number }> = [];
  let currX = startX;
  let currY = startY;
  let backtrackDir = 0;
  const maxSteps = 300;

  for (let step = 0; step < maxSteps; step++) {
    rawPoints.push({
      x: clamp01(currX / protoSize),
      y: clamp01(currY / protoSize),
    });

    let foundNext = false;
    for (let i = 0; i < 8; i++) {
      const dirIdx = (backtrackDir + i) % 8;
      const [dx, dy] = DIRS[dirIdx];
      const nx = currX + dx;
      const ny = currY + dy;
      if (
        nx >= 0 &&
        nx < protoSize &&
        ny >= 0 &&
        ny < protoSize &&
        maskGrid[ny * protoSize + nx] > 0
      ) {
        currX = nx;
        currY = ny;
        backtrackDir = (dirIdx + 5) % 8;
        foundNext = true;
        break;
      }
    }

    if (!foundNext || (currX === startX && currY === startY && step > 2)) {
      break;
    }
  }

  if (rawPoints.length < 3) {
    const x1 = bbox.x;
    const y1 = bbox.y;
    const x2 = clamp01(bbox.x + bbox.width);
    const y2 = clamp01(bbox.y + bbox.height);
    return [
      { x: Number(x1.toFixed(4)), y: Number(y1.toFixed(4)) },
      { x: Number(x2.toFixed(4)), y: Number(y1.toFixed(4)) },
      { x: Number(x2.toFixed(4)), y: Number(y2.toFixed(4)) },
      { x: Number(x1.toFixed(4)), y: Number(y2.toFixed(4)) },
    ];
  }

  const simplified = ramerDouglasPeucker(rawPoints, epsilon);
  return simplified.map((p) => ({
    x: Number(p.x.toFixed(4)),
    y: Number(p.y.toFixed(4)),
  }));
}

interface CandidatePeak {
  gx: number;
  gy: number;
  score: number;
  bbox: BoundingBox;
  coeffs: Float32Array;
  label: string;
}

export class MirageWebGpuEngine implements MirageTask {
  private inputCanvas: HTMLCanvasElement | null = null;
  private inputCtx: CanvasRenderingContext2D | null = null;
  private readonly owner: OnnxSessionOwner;
  private wasmCache: DetectedObject[] = [];
  private wasmTask: MirageTask | null = null;
  private isWasmInFlight = false;
  private frameCount = 0;

  constructor(
    private readonly session: ort.InferenceSession,
    public readonly backend: OnnxBackendInfo = unknownOnnxBackend(),
    wasmTask?: MirageTask
  ) {
    this.owner = new OnnxSessionOwner(session, 'Mirage-WebGPU');
    this.wasmTask = wasmTask ?? null;
  }

  public setWasmTask(wasmTask: MirageTask | null): void {
    this.wasmTask = wasmTask;
  }

  public fuseWasmObservations(objects: DetectedObject[]): void {
    this.wasmCache = objects;
  }

  public close(): void {
    this.inputCanvas = null;
    this.inputCtx = null;
    if (this.wasmTask) {
      try {
        this.wasmTask.close();
      } catch {
        // Ignore wasm close errors
      }
      this.wasmTask = null;
    }
    void this.owner.close();
  }

  public async detectForVideo(
    input: InferenceInput,
    confidenceThreshold = 0.25,
    extraOptions?: MirageInferExtraOptions
  ): Promise<DetectedObject[]> {
    return this.owner.use(async () => {
      const S = MIRAGE_WEBGPU_INPUT_SIZE;
      if (!this.inputCanvas) {
        this.inputCanvas = document.createElement('canvas');
        this.inputCanvas.width = S;
        this.inputCanvas.height = S;
        this.inputCtx = this.inputCanvas.getContext('2d', {
          willReadFrequently: true,
        });
      }
      const ctx = this.inputCtx;
      if (!ctx) {
        throw new OnnxInferenceError(
          'Mirage-WebGPU',
          'input',
          '2D canvas context unavailable'
        );
      }

      ctx.drawImage(input as CanvasImageSource, 0, 0, S, S);
      const imgData = ctx.getImageData(0, 0, S, S);
      const rgba = imgData.data;

      const pixels = S * S;
      const float32Data = new Float32Array(3 * pixels);
      for (let i = 0; i < pixels; i++) {
        float32Data[i] = rgba[i * 4] / 255.0;
        float32Data[pixels + i] = rgba[i * 4 + 1] / 255.0;
        float32Data[2 * pixels + i] = rgba[i * 4 + 2] / 255.0;
      }

      const inputName = this.session.inputNames[0] || 'images';
      const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, S, S]);
      const owned: ort.Tensor[] = [inputTensor];

      try {
        let results: Record<string, ort.Tensor>;
        try {
          results = await this.session.run({ [inputName]: inputTensor });
        } catch (err) {
          throw new OnnxInferenceError('Mirage-WebGPU', 'run', err);
        }
        owned.push(...Object.values(results));
        if (this.owner.closed) return [];

        const denseTensor =
          results.dense_out ?? results[this.session.outputNames[0]];
        const coefTensor =
          results.mask_coeffs ?? results[this.session.outputNames[1]];
        const protoTensor =
          results.mask_protos ?? results[this.session.outputNames[2]];
        const classTensor =
          results.class_logits ?? results[this.session.outputNames[3]];

        if (!denseTensor || !coefTensor || !protoTensor) {
          throw new OnnxInferenceError(
            'Mirage-WebGPU',
            'output',
            'Missing dense_out, mask_coeffs, or mask_protos tensors'
          );
        }

        const denseData = denseTensor.data as Float32Array;
        const coefData = coefTensor.data as Float32Array;
        const protoData = protoTensor.data as Float32Array;
        const classData = (classTensor?.data as Float32Array | undefined) ?? null;

        const G = MIRAGE_WEBGPU_GRID_SIZE; // 32
        const gridArea = G * G;
        const threshold = Math.max(0.15, Math.min(0.95, confidenceThreshold));
        const iouThreshold = extraOptions?.iouThreshold ?? 0.45;
        const maxDetections = extraOptions?.maxDetections ?? 15;
        const includeOcr = extraOptions?.includeOcr ?? true;

        // Trigger background WASM engine perception for open-vocabulary and OCR label enrichment
        if (
          this.wasmTask &&
          !this.isWasmInFlight &&
          (this.frameCount === 0 || this.frameCount % 5 === 0)
        ) {
          this.isWasmInFlight = true;
          this.frameCount++;
          const wasmPromise = this.wasmTask.detectForVideo(input, confidenceThreshold, extraOptions);
          if (this.wasmCache.length === 0) {
            try {
              const objs = await wasmPromise;
              if (!this.owner.closed) {
                this.fuseWasmObservations(objs);
              }
            } catch {
              // Standalone WebGPU fallback
            } finally {
              this.isWasmInFlight = false;
            }
          } else {
            wasmPromise
              .then((objs) => {
                if (!this.owner.closed) {
                  this.fuseWasmObservations(objs);
                }
              })
              .catch(() => {})
              .finally(() => {
                this.isWasmInFlight = false;
              });
          }
        }

        const candidates: CandidatePeak[] = [];

        // Scan 32x32 CenterNet heatmap for local 3x3 peaks
        for (let gy = 1; gy < G - 1; gy++) {
          for (let gx = 1; gx < G - 1; gx++) {
            const idx = gy * G + gx;
            // Raw CenterNet objectness score is Sigmoided channel 0 output in [0, 1]
            const score = denseData[idx];
            if (score < threshold) continue;

            // 3x3 local maximum check (allow ties only toward top-left)
            let isPeak = true;
            for (let dy = -1; dy <= 1 && isPeak; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                if (dx === 0 && dy === 0) continue;
                const nIdx = (gy + dy) * G + (gx + dx);
                const nVal = denseData[nIdx];
                if (
                  nVal > score ||
                  (nVal === score && (dy < 0 || (dy === 0 && dx < 0)))
                ) {
                  isPeak = false;
                  break;
                }
              }
            }
            if (!isPeak) continue;

            // Direct CenterNet normalized width and height regression from channels 1 and 2
            const regW = denseData[1 * gridArea + idx];
            const regH = denseData[2 * gridArea + idx];

            // Sub-grid center offsets from channels 3 and 4
            const regDx = (denseData[3 * gridArea + idx] - 0.5) * 0.5;
            const regDy = (denseData[4 * gridArea + idx] - 0.5) * 0.5;

            const cx = clamp01((gx + 0.5 + regDx) / G);
            const cy = clamp01((gy + 0.5 + regDy) / G);
            const boxW = Math.max(0.04, Math.min(0.96, regW));
            const boxH = Math.max(0.04, Math.min(0.96, regH));

            const x1 = clamp01(cx - boxW / 2);
            const y1 = clamp01(cy - boxH / 2);
            const x2 = clamp01(cx + boxW / 2);
            const y2 = clamp01(cy + boxH / 2);

            const width = Number(Math.max(0.04, x2 - x1).toFixed(4));
            const height = Number(Math.max(0.04, y2 - y1).toFixed(4));
            const bbox: BoundingBox = {
              x: Number(x1.toFixed(4)),
              y: Number(y1.toFixed(4)),
              width,
              height,
            };

            const coeffs = new Float32Array(MIRAGE_WEBGPU_NUM_PROTOS);
            for (let m = 0; m < MIRAGE_WEBGPU_NUM_PROTOS; m++) {
              coeffs[m] = coefData[m * gridArea + idx];
            }

            const label = this.selectOpenVocabLabel(
              idx,
              gridArea,
              bbox,
              classData,
              rgba,
              S
            );

            candidates.push({
              gx,
              gy,
              score: Number(Math.min(0.99, score).toFixed(4)),
              bbox,
              coeffs,
              label,
            });
          }
        }

        candidates.sort((a, b) => b.score - a.score);

        // Non-Maximum Suppression
        const selected: CandidatePeak[] = [];
        for (const cand of candidates) {
          if (selected.length >= maxDetections) break;
          let overlap = false;
          for (const sel of selected) {
            if (calculateIoU(cand.bbox, sel.bbox) > iouThreshold) {
              overlap = true;
              break;
            }
          }
          if (!overlap) {
            selected.push(cand);
          }
        }

        const detectedObjects: DetectedObject[] = [];
        const P = MIRAGE_WEBGPU_PROTO_SIZE; // 64
        const protoArea = P * P;

        for (let i = 0; i < selected.length; i++) {
          const item = selected[i];

          // Check if an overlapping WASM observation exists in the fusion cache
          let finalLabel = item.label;
          let finalLabelJa: string;
          let finalCategory: string;

          let bestIoU = 0;
          let bestWasm: DetectedObject | null = null;
          for (const wasmObj of this.wasmCache) {
            if (wasmObj.category === 'テキスト (OCR)') continue;
            const iou = calculateIoU(item.bbox, wasmObj.bbox);
            if (iou > bestIoU) {
              bestIoU = iou;
              bestWasm = wasmObj;
            }
          }

          const spatialTag = formatSpatialGuidanceTagJa(item.bbox);

          if (bestWasm && bestIoU >= 0.20) {
            finalLabel = bestWasm.label;
            finalCategory = bestWasm.category;
            const baseJa = bestWasm.labelJa.split('（')[0].trim();
            finalLabelJa = `${baseJa}（${spatialTag}）`;
          } else {
            const labelInfo = getLabelInfo(item.label);
            finalCategory = labelInfo.category;
            finalLabelJa = labelInfo.ja.includes('（')
              ? labelInfo.ja
              : `${labelInfo.ja}（${spatialTag}）`;
          }

          const colorHex = getCategoryColor(finalLabel);
          const [cr, cg, cb] = parseHexColor(colorHex);

          const { maskCanvas, maskPolygon } =
            this.buildInstanceMaskAndContour(
              item.bbox,
              item.coeffs,
              protoData,
              P,
              protoArea,
              cr,
              cg,
              cb
            );

          const finalBbox = item.bbox;

          detectedObjects.push({
            id: `mirage-webgpu-obj-${i}-${finalLabel.replace(/\s+/g, '-')}`,
            label: finalLabel,
            labelJa: finalLabelJa,
            category: finalCategory,
            confidence: item.score,
            bbox: finalBbox,
            color: colorHex,
            ...(maskCanvas || bestWasm?.maskCanvas ? { maskCanvas: maskCanvas ?? bestWasm?.maskCanvas } : {}),
            ...(maskPolygon || bestWasm?.maskPolygon ? { maskPolygon: maskPolygon ?? bestWasm?.maskPolygon } : {}),
            areaFraction: Number(
              (finalBbox.width * finalBbox.height).toFixed(4)
            ),
          });
        }

        // Recover any high-confidence WASM objects that had no matching CenterNet peak
        if (this.wasmCache.length > 0) {
          for (const wasmObj of this.wasmCache) {
            if (detectedObjects.length >= maxDetections) break;
            if (wasmObj.category === 'テキスト (OCR)') continue;
            const overlaps = detectedObjects.some(
              (d) => calculateIoU(d.bbox, wasmObj.bbox) > 0.3
            );
            if (!overlaps) {
              const spatialTag = formatSpatialGuidanceTagJa(wasmObj.bbox);
              const baseJa = wasmObj.labelJa.split('（')[0].trim();
              detectedObjects.push({
                ...wasmObj,
                id: `mirage-fused-${wasmObj.id}`,
                labelJa: `${baseJa}（${spatialTag}）`,
              });
            }
          }
        }

        // Extract OCR text regions from channel 5 of dense_out when enabled
        if (includeOcr && detectedObjects.length < maxDetections) {
          const ocrRegions = this.extractOcrRegions(
            denseData,
            G,
            gridArea,
            Math.max(0.25, threshold),
            maxDetections - detectedObjects.length,
            this.wasmCache
          );
          detectedObjects.push(...ocrRegions);
        }

        return detectedObjects;
      } finally {
        disposeTensors(owned);
      }
    });
  }

  private selectOpenVocabLabel(
    gridIdx: number,
    gridArea: number,
    bbox: BoundingBox,
    classData: Float32Array | null,
    rgba: Uint8ClampedArray,
    imgSize: number
  ): string {
    const x0 = Math.max(0, Math.floor(bbox.x * imgSize));
    const y0 = Math.max(0, Math.floor(bbox.y * imgSize));
    const x1 = Math.min(imgSize - 1, Math.ceil((bbox.x + bbox.width) * imgSize));
    const y1 = Math.min(imgSize - 1, Math.ceil((bbox.y + bbox.height) * imgSize));

    let rSum = 0;
    let gSum = 0;
    let bSum = 0;
    let count = 0;
    const step = 4;
    for (let y = y0; y <= y1; y += step) {
      for (let x = x0; x <= x1; x += step) {
        const p = (y * imgSize + x) * 4;
        rSum += rgba[p];
        gSum += rgba[p + 1];
        bSum += rgba[p + 2];
        count++;
      }
    }
    const rMean = count > 0 ? rSum / (count * 255) : 0.5;
    const gMean = count > 0 ? gSum / (count * 255) : 0.5;
    const bMean = count > 0 ? bSum / (count * 255) : 0.5;
    const aspect = bbox.width / Math.max(0.02, bbox.height);
    const area = bbox.width * bbox.height;

    // 1. High-confidence class logits from ONNX descriptor head
    if (classData) {
      let bestIdx = 0;
      let bestVal = -Infinity;
      const numClasses = Math.min(
        MIRAGE_OPEN_VOCAB_LABELS.length,
        Math.floor(classData.length / gridArea)
      );
      for (let c = 0; c < numClasses; c++) {
        const val = classData[c * gridArea + gridIdx];
        if (val > bestVal) {
          bestVal = val;
          bestIdx = c;
        }
      }
      if (bestVal > 0.05 && MIRAGE_OPEN_VOCAB_LABELS[bestIdx]) {
        return MIRAGE_OPEN_VOCAB_LABELS[bestIdx];
      }
    }

    // 2. Warm skin-like / portrait detection
    const isWarmSkin =
      rMean > 0.35 &&
      gMean > 0.22 &&
      bMean > 0.15 &&
      rMean > gMean * 1.05 &&
      gMean >= bMean * 0.90;

    if (isWarmSkin && area >= 0.04 && aspect >= 0.35 && aspect <= 1.60) {
      return 'person';
    }

    // 3. Common scene items
    if (aspect < 0.55 && bbox.height >= 0.18) {
      return 'bottle';
    }

    if (aspect > 1.30 && area >= 0.08) {
      return 'table';
    }

    if (aspect >= 0.65 && aspect <= 1.35 && area >= 0.03 && area < 0.15) {
      return 'chair';
    }

    if (area < 0.04 && aspect > 0.60 && aspect < 1.40) {
      return 'cup';
    }

    return 'object';
  }

  private buildInstanceMaskAndContour(
    bbox: BoundingBox,
    coeffs: Float32Array,
    protoData: Float32Array,
    protoSize: number,
    protoArea: number,
    cr: number,
    cg: number,
    cb: number
  ): {
    maskCanvas?: HTMLCanvasElement;
    maskPolygon?: Array<{ x: number; y: number }>;
    tightBbox?: BoundingBox;
  } {
    const padX = Math.max(2, Math.round(bbox.width * protoSize * 0.15));
    const padY = Math.max(2, Math.round(bbox.height * protoSize * 0.15));
    const minPx = Math.max(0, Math.floor(bbox.x * protoSize) - padX);
    const maxPx = Math.min(protoSize, Math.ceil((bbox.x + bbox.width) * protoSize) + padX);
    const minPy = Math.max(0, Math.floor(bbox.y * protoSize) - padY);
    const maxPy = Math.min(protoSize, Math.ceil((bbox.y + bbox.height) * protoSize) + padY);

    let minActiveX = protoSize;
    let maxActiveX = -1;
    let minActiveY = protoSize;
    let maxActiveY = -1;
    let activePixels = 0;

    const binaryMask = new Uint8Array(protoSize * protoSize);
    let maskCanvas: HTMLCanvasElement | undefined;
    try {
      const canvas = document.createElement('canvas');
      canvas.width = protoSize;
      canvas.height = protoSize;
      const maskCtx = canvas.getContext('2d');
      if (maskCtx && typeof maskCtx.createImageData === 'function') {
        const maskImgData = maskCtx.createImageData(protoSize, protoSize);
        const mPixels = maskImgData.data;

        for (let py = minPy; py < maxPy; py++) {
          const rowOffset = py * protoSize;
          for (let px = minPx; px < maxPx; px++) {
            const pIdx = rowOffset + px;
            let logit = 0;
            for (let m = 0; m < MIRAGE_WEBGPU_NUM_PROTOS; m++) {
              logit += coeffs[m] * protoData[m * protoArea + pIdx];
            }
            const sig = 1 / (1 + Math.exp(-logit));
            if (sig > 0.45) {
              binaryMask[pIdx] = 1;
              activePixels++;
              if (px < minActiveX) minActiveX = px;
              if (px > maxActiveX) maxActiveX = px;
              if (py < minActiveY) minActiveY = py;
              if (py > maxActiveY) maxActiveY = py;
              const pixelIdx = pIdx * 4;
              mPixels[pixelIdx] = cr;
              mPixels[pixelIdx + 1] = cg;
              mPixels[pixelIdx + 2] = cb;
              mPixels[pixelIdx + 3] = Math.min(235, Math.floor(sig * 215));
            }
          }
        }
        maskCtx.putImageData?.(maskImgData, 0, 0);
        maskCanvas = canvas;
      }
    } catch {
      maskCanvas = undefined;
    }

    let tightBbox: BoundingBox | undefined;
    if (activePixels >= 10 && maxActiveX >= minActiveX && maxActiveY >= minActiveY) {
      const tx1 = clamp01((minActiveX - 0.5) / protoSize);
      const ty1 = clamp01((minActiveY - 0.5) / protoSize);
      const tx2 = clamp01((maxActiveX + 1.5) / protoSize);
      const ty2 = clamp01((maxActiveY + 1.5) / protoSize);
      tightBbox = {
        x: Number(tx1.toFixed(4)),
        y: Number(ty1.toFixed(4)),
        width: Number(Math.max(0.04, tx2 - tx1).toFixed(4)),
        height: Number(Math.max(0.04, ty2 - ty1).toFixed(4)),
      };
    }

    // Trace true boundary contour from the active prototype mask using Moore-Neighbor & Douglas-Peucker
    const maskPolygon = traceMaskContourPolygon(
      binaryMask,
      protoSize,
      minPx,
      minPy,
      maxPx,
      maxPy,
      tightBbox ?? bbox,
      0.008
    );

    return { maskCanvas, maskPolygon, tightBbox };
  }

  private extractOcrRegions(
    denseData: Float32Array,
    G: number,
    gridArea: number,
    textThreshold: number,
    maxLines: number,
    wasmCache?: readonly DetectedObject[]
  ): DetectedObject[] {
    if (maxLines <= 0) return [];
    const textOffset = 5 * gridArea;
    let meanTxt = 0;
    for (let i = 0; i < gridArea; i++) {
      meanTxt += denseData[textOffset + i];
    }
    meanTxt /= gridArea;

    const ocrOut: DetectedObject[] = [];
    const visited = new Uint8Array(gridArea);

    for (let gy = 1; gy < G - 1 && ocrOut.length < maxLines; gy++) {
      for (let gx = 1; gx < G - 2 && ocrOut.length < maxLines; gx++) {
        const idx = gy * G + gx;
        if (visited[idx]) continue;
        const raw = denseData[textOffset + idx];
        const prom = raw - meanTxt;
        if (prom < 0.08) continue;
        // Require horizontal continuity (at least 2 adjacent stroke cells)
        const rightProm = denseData[textOffset + idx + 1] - meanTxt;
        if (rightProm < 0.06) continue;

        let endGx = gx + 1;
        while (
          endGx + 1 < G - 1 &&
          denseData[textOffset + gy * G + (endGx + 1)] - meanTxt >= 0.05
        ) {
          endGx++;
        }
        for (let fillX = gx; fillX <= endGx; fillX++) {
          visited[gy * G + fillX] = 1;
          if (gy + 1 < G) visited[(gy + 1) * G + fillX] = 1;
        }

        const conf = raw;
        if (conf < textThreshold) continue;

        const bbox: BoundingBox = {
          x: Number(clamp01((gx - 0.5) / G).toFixed(4)),
          y: Number(clamp01((gy - 0.5) / G).toFixed(4)),
          width: Number(clamp01((endGx - gx + 2) / G).toFixed(4)),
          height: Number(clamp01(2.2 / G).toFixed(4)),
        };
        const spatialTag = formatSpatialGuidanceTagJa(bbox);

        // Check if WASM engine recognized the exact OCR string for this region
        let matchedText: string | null = null;
        if (wasmCache && wasmCache.length > 0) {
          let bestOcrIoU = 0;
          for (const item of wasmCache) {
            if (item.category !== 'テキスト (OCR)') continue;
            const iou = calculateIoU(bbox, item.bbox);
            if (iou > bestOcrIoU) {
              bestOcrIoU = iou;
              const extracted = item.labelJa.replace(/.*?"(.*?)"(?:（.*）)?/, '$1');
              matchedText = extracted && extracted !== item.labelJa ? extracted : item.label;
            }
          }
        }

        const labelJa = matchedText
          ? `文字: "${matchedText}"（${spatialTag}）`
          : `文字領域（${spatialTag}）`;

        ocrOut.push({
          id: `mirage-webgpu-ocr-${ocrOut.length}-${gy}-${gx}`,
          label: matchedText ? `ocr:${matchedText}` : 'ocr:text-region',
          labelJa,
          category: 'テキスト (OCR)',
          confidence: Number(Math.min(0.98, conf).toFixed(4)),
          bbox,
          color: '#38bdf8',
          areaFraction: Number((bbox.width * bbox.height).toFixed(4)),
        });
      }
    }
    return ocrOut;
  }
}

export async function createMirageWebGpuTask(
  options: MirageWebGpuTaskOptions = {}
): Promise<MirageTask> {
  if (options.modelUrl) {
    const { session, backend } = await createOrtSession(
      options.modelUrl,
      options.delegate,
      {
        recordAssetSource: options.recordAssetSource,
        modelId: 'mirage',
        variantId: 'mirage-webgpu',
        primarySource: 'external',
        fallbackModelBytes: () => buildMirageWebGpuOnnxBytes(),
      }
    );
    return new MirageWebGpuEngine(session, backend, options.wasmTask);
  }

  const resolved = await resolveModelAssetUrls('mirage', 'mirage-webgpu');
  const { session, backend } = await createOrtSession(
    resolved.primaryUrl,
    options.delegate,
    {
      fallbackModelUrl: resolved.fallbackUrl,
      fallbackModelBytes: () => buildMirageWebGpuOnnxBytes(),
      variantId: 'mirage-webgpu',
      recordAssetSource: options.recordAssetSource,
      modelId: 'mirage',
      primarySource: resolved.source,
    }
  );
  return new MirageWebGpuEngine(session, backend, options.wasmTask);
}
