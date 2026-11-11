import * as ort from 'onnxruntime-web';
import type {
  BoundingBox,
  DetectedObject,
  InferenceInput,
  MirageAnalysisMode,
  MirageVariantId,
  OnnxBackendInfo,
} from '../types/vision';
import {
  resolveModelAssetUrls,
  type AssetSourceRecorder,
} from './modelAssetResolver';
import type { MirageInferExtraOptions, MirageTask } from './mirageWasmService';
import {
  createOrtSession,
  disposeTensors,
  OnnxInferenceError,
  OnnxSessionOwner,
  unknownOnnxBackend,
} from './onnxSession';

export const MIRAGE_EXPERIMENTAL_ONNX_FILE = 'cact-dequantized-reference-order.onnx';
export const MIRAGE_EXPERIMENTAL_ONNX_RELATIVE_PATH =
  '.cache/mirage-refinement/numerical-cact-dequantized-pixel-0-r3/cact-dequantized-reference-order.onnx';
export const MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE = 34_281_503;
export const MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256 =
  '52d52d2918160404cd806702a270736819712025cf3c6aa8e907a06bc3cbdd07';
export const MIRAGE_EXPERIMENTAL_ONNX_OPSET = 17;
export const MIRAGE_EXPERIMENTAL_ONNX_TENSOR_COUNT = 452;
export const MIRAGE_TOTAL_CHECKPOINT_TENSOR_COUNT = 583;
export const MIRAGE_EXPERIMENTAL_ONNX_FINAL_OUTPUTS = [
  'detections',
  'prototypes',
  'coefficients',
] as const;

export type MirageExperimentalVariantId =
  | 'mirage-onnx-exp-wasm'
  | 'mirage-onnx-exp-webgpu';

export interface MirageCeil32PreprocessMetadata {
  W: number;
  H: number;
  w: number;
  h: number;
  r: number;
  coordinate_scale: [number, number, number, number];
  resize_rule: 'ceil32';
}

export interface MiragePreprocessedInput {
  image: Float32Array;
  valid_hw: [number, number];
  metadata: MirageCeil32PreprocessMetadata;
}

export interface MirageRawHeadOutputs {
  detections: Float32Array;
  prototypes: Float32Array;
  coefficients: Float32Array;
}

export interface MirageDecodedProposal {
  kind: 'object_proposal' | 'text_proposal';
  index: number;
  cell: [number, number];
  score: number;
  /** Scaled [x1, y1, x2, y2] in [0, 1] relative to original image dimensions */
  box: [number, number, number, number];
  /** [x1, y1, x2, y2] in [0, 1000] integer coordinates matching perception_diagnostics.py `bbox` */
  bbox: [number, number, number, number];
  /** Present for object proposals: 128x128 diagnostic mask (0 or 1) matching `_mask` */
  mask?: Uint8Array;
  maskActivePixels?: number;
}

export interface MirageDecodedPerceptionResult {
  confidence: number;
  nms_iou: number | null;
  objects: MirageDecodedProposal[];
  text: MirageDecodedProposal[];
}

export interface MirageExperimentalExecutionDiagnostics {
  variantId: MirageExperimentalVariantId;
  requestedProvider: 'wasm' | 'webgpu';
  configuredProviders: readonly ('webgpu' | 'wasm')[];
  activeProvider: 'wasm' | 'webgpu';
  fallbackReason: string | null;
  artifactSha256: string;
  artifactSize: number;
  opset: number;
  tensorCoverage: string;
  validHw: [number, number];
  metadata: MirageCeil32PreprocessMetadata;
  nmsIou: number | null;
  confidenceThreshold: number;
  objectProposalCount: number;
  textProposalCount: number;
  nativeOverlayEnabled: boolean;
  nativeOverlayCount: number;
  timingsMs: {
    preprocess: number;
    inference: number;
    postprocess: number;
    total: number;
  };
}

export function isMirageExperimentalVariant(
  variantId?: string | null
): variantId is MirageExperimentalVariantId {
  return (
    variantId === 'mirage-onnx-exp-wasm' ||
    variantId === 'mirage-onnx-exp-webgpu'
  );
}

/**
 * Reproduces scripts/mirage/refinement_inputs.py `extent(width, height, 'ceil32')`
 * and `preprocess(image, 'ceil32')` metadata.
 */
export function computeMirageCeil32Metadata(
  width: number,
  height: number
): MirageCeil32PreprocessMetadata {
  const W = Math.max(1, Math.round(width));
  const H = Math.max(1, Math.round(height));
  const r = 512 / Math.max(W, H);
  const w = Math.min(512, Math.max(32, Math.ceil((W * r) / 32) * 32));
  const h = Math.min(512, Math.max(32, Math.ceil((H * r) / 32) * 32));
  return {
    W,
    H,
    w,
    h,
    r,
    coordinate_scale: [512 / w, 512 / h, 512 / w, 512 / h],
    resize_rule: 'ceil32',
  };
}

/**
 * Packs an RGBA or RGB buffer of already-resized `(w, h)` pixels into a `[1, 3, 512, 512]`
 * NCHW Float32Array placed at top-left `[:h, :w]` with `0.0` padding and `v / 127.5 - 1` normalization.
 */
export function packResizedPixelsToMirageTensor(
  pixels: Uint8ClampedArray | Uint8Array | Float32Array,
  channels: 3 | 4,
  metadata: MirageCeil32PreprocessMetadata
): MiragePreprocessedInput {
  const { w, h } = metadata;
  const planeSize = 512 * 512;
  const image = new Float32Array(3 * planeSize); // Zero-filled by default (padding = 0.0)
  for (let row = 0; row < h; row++) {
    const dstRowOffset = row * 512;
    const srcRowOffset = row * w * channels;
    for (let col = 0; col < w; col++) {
      const srcIdx = srcRowOffset + col * channels;
      const dstIdx = dstRowOffset + col;
      image[dstIdx] = pixels[srcIdx] / 127.5 - 1.0;
      image[planeSize + dstIdx] = pixels[srcIdx + 1] / 127.5 - 1.0;
      image[2 * planeSize + dstIdx] = pixels[srcIdx + 2] / 127.5 - 1.0;
    }
  }
  return {
    image,
    valid_hw: [h, w],
    metadata,
  };
}

/**
 * Python round-half-to-even (banker's rounding) to match `round(float(v) * 1000)` in perception_diagnostics.py.
 */
export function roundHalfToEven(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

const GRID_128: Float64Array = (() => {
  const g = new Float64Array(128);
  for (let k = 0; k < 128; k++) {
    g[k] = (k + 0.5) / 128;
  }
  return g;
})();

/**
 * Computes box IoU matching `box_iou` in `scripts/mirage/perception_diagnostics.py`
 * (`[x1, y1, x2, y2]` format).
 */
export function computeBoxIou(
  a: readonly [number, number, number, number] | readonly number[],
  b: readonly [number, number, number, number] | readonly number[]
): number {
  const [ax1, ay1, ax2, ay2] = a;
  const [bx1, by1, bx2, by2] = b;
  const interW = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
  const interH = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
  const inter = interW * interH;
  const union =
    Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1) +
    Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1) -
    inter;
  return union > 0 ? inter / union : 0.0;
}

/**
 * Greedy IoU NMS on decoded proposals when needed as a standalone helper.
 */
export function applyGreedyNms(
  items: readonly MirageDecodedProposal[],
  iouThreshold: number | null,
  limit?: number
): MirageDecodedProposal[] {
  if (iouThreshold === null || iouThreshold === undefined) {
    return limit !== undefined ? items.slice(0, limit) : items.slice();
  }
  const kept: MirageDecodedProposal[] = [];
  for (const item of items) {
    const overlaps = kept.some(
      (prev) => computeBoxIou(item.box, prev.box) > iouThreshold
    );
    if (!overlaps) {
      kept.push({
        ...item,
        index: kept.length,
      });
      if (limit !== undefined && kept.length >= limit) break;
    }
  }
  return kept;
}

function stableSigmoidFloat32(x: number): number {
  // Matches numpy float32 `np.exp(-np.logaddexp(0, -x))`
  const f = Math.fround(x);
  return Math.fround(1 / (1 + Math.exp(-f)));
}

/**
 * Exact TypeScript port of `decode` in `scripts/mirage/perception_diagnostics.py`
 * and `decode_policy` in `scripts/mirage/corpus_postprocessing.py`.
 *
 * - Default confidence: 0.3
 * - 3x3 local peak filtering (`heat === peak`)
 * - Valid cell constraint `(row * 8 < h) && (col * 8 < w)`
 * - Deterministic sort: `(-score, row, col)`
 * - Top-K: max 64 object proposals (channel 0), max 32 text proposals (channel 1)
 * - Optional greedy IoU NMS (`null` by default, or `0.5` matching `corpus_postprocessing.py`)
 * - 128x128 diagnostic mask for object proposals
 */
export function decodeMirageExperimentalOutputs(
  outputs: MirageRawHeadOutputs,
  meta: MirageCeil32PreprocessMetadata,
  options: {
    confidence?: number;
    nmsIou?: number | null;
    maxObjects?: number;
    maxText?: number;
    includeMasks?: boolean;
  } = {}
): MirageDecodedPerceptionResult {
  const confidence = options.confidence ?? 0.3;
  const nmsIou = options.nmsIou === undefined ? null : options.nmsIou;
  const maxObjects = options.maxObjects ?? 64;
  const maxText = options.maxText ?? 32;
  const includeMasks = options.includeMasks ?? true;

  const { detections, prototypes, coefficients } = outputs;
  if (detections.length !== 64 * 64 * 10) {
    throw new OnnxInferenceError(
      'Mirage-Exp-ONNX',
      'output',
      `Unexpected detections length ${detections.length} (expected ${64 * 64 * 10})`
    );
  }
  if (prototypes.length !== 32 * 128 * 128) {
    throw new OnnxInferenceError(
      'Mirage-Exp-ONNX',
      'output',
      `Unexpected prototypes length ${prototypes.length} (expected ${32 * 128 * 128})`
    );
  }
  if (coefficients.length !== 32 * 64 * 64) {
    throw new OnnxInferenceError(
      'Mirage-Exp-ONNX',
      'output',
      `Unexpected coefficients length ${coefficients.length} (expected ${32 * 64 * 64})`
    );
  }

  const scale = meta.coordinate_scale ?? [
    512 / (meta.W * meta.r),
    512 / (meta.H * meta.r),
    512 / (meta.W * meta.r),
    512 / (meta.H * meta.r),
  ];

  // Precompute resampling indices for 128x128 diagnostic mask:
  // rows = np.minimum(127, ((np.arange(128)+0.5)/128 * meta['h']/4).astype(int))
  const resampleRows = new Int32Array(128);
  const resampleCols = new Int32Array(128);
  for (let k = 0; k < 128; k++) {
    resampleRows[k] = Math.min(127, Math.trunc(GRID_128[k] * (meta.h / 4)));
    resampleCols[k] = Math.min(127, Math.trunc(GRID_128[k] * (meta.w / 4)));
  }

  const decodeChannel = (
    channel: 0 | 1,
    limit: number
  ): MirageDecodedProposal[] => {
    const heat = new Float32Array(64 * 64);
    for (let i = 0; i < 64; i++) {
      for (let j = 0; j < 64; j++) {
        const logit = detections[(i * 64 + j) * 10 + channel];
        heat[i * 64 + j] = stableSigmoidFloat32(logit);
      }
    }

    const candidates: Array<{ i: number; j: number; score: number }> = [];
    for (let i = 0; i < 64; i++) {
      if (i * 8 >= meta.h) continue;
      const rMin = Math.max(0, i - 1);
      const rMax = Math.min(63, i + 1);
      for (let j = 0; j < 64; j++) {
        if (j * 8 >= meta.w) continue;
        const val = heat[i * 64 + j];
        if (val < confidence) continue;
        const cMin = Math.max(0, j - 1);
        const cMax = Math.min(63, j + 1);
        let peak = -Infinity;
        for (let ni = rMin; ni <= rMax; ni++) {
          const rowOff = ni * 64;
          for (let nj = cMin; nj <= cMax; nj++) {
            const neighbor = heat[rowOff + nj];
            if (neighbor > peak) peak = neighbor;
          }
        }
        if (val === peak) {
          candidates.push({ i, j, score: val });
        }
      }
    }

    // Sort by (-score, i, j)
    candidates.sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      if (a.i !== b.i) return a.i - b.i;
      return a.j - b.j;
    });

    const items: MirageDecodedProposal[] = [];
    const selectedRawBoxes: Array<[number, number, number, number]> = [];
    const maxWFrac = meta.w / 512;
    const maxHFrac = meta.h / 512;

    const candidatePool =
      nmsIou === null ? candidates.slice(0, limit) : candidates;

    for (let idx = 0; idx < candidatePool.length; idx++) {
      const { i, j, score } = candidatePool[idx];
      const base = (i * 64 + j) * 10;
      const width = Math.abs(Number(detections[base + 2 + 2 * channel]));
      const height = Math.abs(Number(detections[base + 3 + 2 * channel]));
      const cx = j + Number(detections[base + 6 + 2 * channel]);
      const cy = i + Number(detections[base + 7 + 2 * channel]);

      const bx0 = Math.min(maxWFrac, Math.max(0, (cx - width / 2) / 64));
      const by0 = Math.min(maxHFrac, Math.max(0, (cy - height / 2) / 64));
      const bx1 = Math.min(maxWFrac, Math.max(0, (cx + width / 2) / 64));
      const by1 = Math.min(maxHFrac, Math.max(0, (cy + height / 2) / 64));
      const rawBox: [number, number, number, number] = [bx0, by0, bx1, by1];

      if (nmsIou !== null) {
        const overlaps = selectedRawBoxes.some(
          (oldBox) => computeBoxIou(rawBox, oldBox) > nmsIou
        );
        if (overlaps) continue;
        selectedRawBoxes.push(rawBox);
      }

      const normX0 = bx0 * scale[0];
      const normY0 = by0 * scale[1];
      const normX1 = bx1 * scale[2];
      const normY1 = by1 * scale[3];

      const bbox: [number, number, number, number] = [
        roundHalfToEven(normX0 * 1000),
        roundHalfToEven(normY0 * 1000),
        roundHalfToEven(normX1 * 1000),
        roundHalfToEven(normY1 * 1000),
      ];

      const proposal: MirageDecodedProposal = {
        kind: channel === 0 ? 'object_proposal' : 'text_proposal',
        index: items.length,
        cell: [i, j],
        score,
        box: [
          Math.min(1, Math.max(0, normX0)),
          Math.min(1, Math.max(0, normY0)),
          Math.min(1, Math.max(0, normX1)),
          Math.min(1, Math.max(0, normY1)),
        ],
        bbox,
      };

      if (channel === 0 && includeMasks) {
        const coeffVec = new Float32Array(32);
        const cellOffset = i * 64 + j;
        for (let r = 0; r < 32; r++) {
          coeffVec[r] = coefficients[r * 4096 + cellOffset];
        }

        const centerRow = Math.min(127, Math.trunc((by0 + by1) * 64));
        const centerCol = Math.min(127, Math.trunc((bx0 + bx1) * 64));

        // First compute the 128x128 base mask before resampling with np.ix_(rows, cols)
        const baseMask = new Uint8Array(128 * 128);
        for (let py = 0; py < 128; py++) {
          const gy = GRID_128[py];
          const yInBox = gy >= by0 && gy <= by1;
          const protoRowBase = py * 128;
          for (let px = 0; px < 128; px++) {
            const gx = GRID_128[px];
            const inside =
              (yInBox && gx >= bx0 && gx <= bx1) ||
              (py === centerRow && px === centerCol);
            if (!inside) continue;

            const protoPixelIdx = protoRowBase + px;
            let sum = 0.0;
            for (let r = 0; r < 32; r++) {
              sum = Math.fround(
                sum +
                  Math.fround(
                    coeffVec[r] * prototypes[r * 16384 + protoPixelIdx]
                  )
              );
            }
            const prob = stableSigmoidFloat32(sum);
            if (prob >= 0.5) {
              baseMask[protoPixelIdx] = 1;
            }
          }
        }

        const mask = new Uint8Array(128 * 128);
        let activePixels = 0;
        for (let outY = 0; outY < 128; outY++) {
          const py = resampleRows[outY];
          const baseRowOff = py * 128;
          const outRowOff = outY * 128;
          for (let outX = 0; outX < 128; outX++) {
            const px = resampleCols[outX];
            const val = baseMask[baseRowOff + px];
            if (val) {
              mask[outRowOff + outX] = 1;
              activePixels++;
            }
          }
        }
        proposal.mask = mask;
        proposal.maskActivePixels = activePixels;
      }

      items.push(proposal);
      if (items.length >= limit) break;
    }

    return items;
  };

  return {
    confidence,
    nms_iou: nmsIou,
    objects: decodeChannel(0, maxObjects),
    text: decodeChannel(1, maxText),
  };
}

/**
 * Renders a 128x128 diagnostic binary mask into an HTMLCanvasElement for overlay display.
 */
export function renderDiagnosticMaskToCanvas(
  mask: Uint8Array,
  hexColor: string
): HTMLCanvasElement | undefined {
  if (typeof document === 'undefined') return undefined;
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  if (!ctx) return undefined;

  const clean = hexColor.replace('#', '');
  const r = parseInt(clean.substring(0, 2), 16) || 56;
  const g = parseInt(clean.substring(2, 4), 16) || 189;
  const b = parseInt(clean.substring(4, 6), 16) || 248;

  const imageData = ctx.createImageData(128, 128);
  const data = imageData.data;
  for (let i = 0; i < 128 * 128; i++) {
    if (mask[i]) {
      const p = i * 4;
      data[p] = r;
      data[p + 1] = g;
      data[p + 2] = b;
      data[p + 3] = 165;
    }
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

/**
 * Converts decoded ONNX perception proposals into VisionStudio `DetectedObject[]`
 * without fabricating any object class names or OCR text strings.
 */
export function convertProposalsToDetectedObjects(
  decoded: MirageDecodedPerceptionResult,
  options: {
    showObjectProposals?: boolean;
    showTextProposals?: boolean;
    showDiagnosticMasks?: boolean;
    maxDetections?: number;
    sourceAttributionPrefix?: boolean;
  } = {}
): DetectedObject[] {
  const showObjects = options.showObjectProposals ?? true;
  const showText = options.showTextProposals ?? true;
  const showMasks = options.showDiagnosticMasks ?? true;
  const prefix = options.sourceAttributionPrefix ? '[ONNX候補] ' : '';

  const results: DetectedObject[] = [];

  if (showObjects) {
    const objectCap = options.maxDetections ?? decoded.objects.length;
    const objects = decoded.objects.slice(0, objectCap);
    for (let i = 0; i < objects.length; i++) {
      const item = objects[i];
      const [x1, y1, x2, y2] = item.box;
      const bbox: BoundingBox = {
        x: x1,
        y: y1,
        width: Math.max(0.002, x2 - x1),
        height: Math.max(0.002, y2 - y1),
      };
      const color = '#22d3ee'; // Cyan technical frame for ONNX object region proposals
      const maskCanvas =
        showMasks && item.mask
          ? renderDiagnosticMaskToCanvas(item.mask, color)
          : undefined;
      results.push({
        id: `mirage-onnx-exp-obj-${item.cell[0]}-${item.cell[1]}-${i}`,
        label: `object_region_candidate_${i + 1}`,
        labelJa: `${prefix}物体領域候補 #${i + 1} (${item.cell[0]},${item.cell[1]})`,
        category: '物体領域候補 (部分ONNX)',
        confidence: item.score,
        bbox,
        color,
        maskCanvas,
      });
    }
  }

  if (showText) {
    for (let i = 0; i < decoded.text.length; i++) {
      const item = decoded.text[i];
      const [x1, y1, x2, y2] = item.box;
      const bbox: BoundingBox = {
        x: x1,
        y: y1,
        width: Math.max(0.002, x2 - x1),
        height: Math.max(0.002, y2 - y1),
      };
      results.push({
        id: `mirage-onnx-exp-txt-${item.cell[0]}-${item.cell[1]}-${i}`,
        label: `text_region_candidate_${i + 1}`,
        labelJa: `${prefix}テキスト領域候補 #${i + 1} (${item.cell[0]},${item.cell[1]})`,
        category: 'テキスト領域候補 (部分ONNX)',
        confidence: item.score,
        bbox,
        color: '#fbbf24', // Amber technical frame for ONNX text region proposals
      });
    }
  }

  return results;
}

function getInputDimensions(input: InferenceInput): { width: number; height: number } {
  if ('videoWidth' in input) {
    return {
      width: input.videoWidth || 640,
      height: input.videoHeight || 480,
    };
  }
  if ('displayWidth' in input) {
    return {
      width: input.displayWidth || 640,
      height: input.displayHeight || 480,
    };
  }
  return {
    width: input.width || 640,
    height: input.height || 480,
  };
}

export interface MirageExperimentalInferOptions extends MirageInferExtraOptions {
  /** Override NMS policy: null = raw 3x3 peaks (default 52/54 corpus baseline), number = greedy IoU NMS (e.g. 0.5) */
  experimentalNmsIou?: number | null;
  /** Toggle object region proposals visibility */
  showObjectProposals?: boolean;
  /** Toggle text region proposals visibility */
  showTextProposals?: boolean;
  /** Toggle 128x128 diagnostic object masks */
  showDiagnosticMasks?: boolean;
  /** Optional toggle to overlay official .cact WASM detections with explicit source attribution */
  overlayNativeWasm?: boolean;
}

export async function verifyMirageExperimentalOnnxBytes(
  bytes: ArrayBuffer | Uint8Array
): Promise<{ size: number; sha256: string }> {
  const view =
    bytes instanceof Uint8Array
      ? bytes
      : new Uint8Array(bytes);
  if (view.byteLength !== MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE) {
    throw new Error(
      `Pinned model size mismatch: expected ${MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE}, got ${view.byteLength}`
    );
  }
  const copy = new Uint8Array(view.byteLength);
  copy.set(view);
  const digestBuf = await globalThis.crypto.subtle.digest(
    'SHA-256',
    copy.buffer
  );
  const sha256 = Array.from(new Uint8Array(digestBuf))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
  if (sha256 !== MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256) {
    throw new Error(
      `Pinned model SHA256 mismatch: expected ${MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256}, got ${sha256}`
    );
  }
  return { size: view.byteLength, sha256 };
}

export class MirageExperimentalOnnxEngine implements MirageTask {
  public readonly backend: OnnxBackendInfo;
  public readonly variantId: MirageExperimentalVariantId;
  private readonly rawSession: ort.InferenceSession;
  private readonly owner: OnnxSessionOwner;
  private resizeCanvas: HTMLCanvasElement | null = null;
  private resizeCtx: CanvasRenderingContext2D | null = null;
  private nativeOverlayTask: MirageTask | null = null;
  private readonly nativeOverlayTaskFactory?: () => Promise<MirageTask>;
  private lastDiagnostics: MirageExperimentalExecutionDiagnostics | null = null;
  private runMutex: Promise<unknown> = Promise.resolve();

  constructor(
    session: ort.InferenceSession,
    variantId: MirageExperimentalVariantId,
    backend?: OnnxBackendInfo,
    nativeOverlayTaskFactory?: () => Promise<MirageTask>
  ) {
    this.variantId = variantId;
    const requestedProvider =
      variantId === 'mirage-onnx-exp-webgpu' ? 'webgpu' : 'wasm';
    this.backend =
      backend ??
      ({
        runtime: 'onnxruntime-web',
        requestedProvider,
        configuredProviders:
          requestedProvider === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
        fallbackReason: null,
        actualBackend: 'unmeasured',
      } satisfies OnnxBackendInfo);
    this.rawSession = session;
    this.owner = new OnnxSessionOwner(session, 'Mirage-Exp-ONNX');
    this.nativeOverlayTaskFactory = nativeOverlayTaskFactory;
  }

  public getLastDiagnostics(): MirageExperimentalExecutionDiagnostics | null {
    return this.lastDiagnostics;
  }

  private enqueueExclusive<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.runMutex.then(fn, fn);
    this.runMutex = next.then(
      () => undefined,
      () => undefined
    );
    return next;
  }

  public preprocessInput(input: InferenceInput): MiragePreprocessedInput {
    const { width: srcW, height: srcH } = getInputDimensions(input);
    const metadata = computeMirageCeil32Metadata(srcW, srcH);
    const { w, h } = metadata;

    if (!this.resizeCanvas) {
      this.resizeCanvas = document.createElement('canvas');
      this.resizeCtx = this.resizeCanvas.getContext('2d', {
        willReadFrequently: true,
      });
    }
    if (this.resizeCanvas.width !== w || this.resizeCanvas.height !== h) {
      this.resizeCanvas.width = w;
      this.resizeCanvas.height = h;
    }

    const ctx = this.resizeCtx;
    if (!ctx) {
      throw new OnnxInferenceError(
        'Mirage-Exp-ONNX',
        'input',
        'Failed to acquire 2D canvas context for Mirage ceil32 preprocessing'
      );
    }

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(input as CanvasImageSource, 0, 0, w, h);
    const imageData = ctx.getImageData(0, 0, w, h);
    return packResizedPixelsToMirageTensor(imageData.data, 4, metadata);
  }

  /**
   * Executes inference on a precomputed or preprocessed `{ image, valid_hw, metadata }` input,
   * fetching only the 3 terminal perception outputs (`detections`, `prototypes`, `coefficients`)
   * from the 64-output verified ONNX graph.
   */
  public async runRawHeads(
    preprocessed: MiragePreprocessedInput
  ): Promise<MirageRawHeadOutputs> {
    return this.owner.use(() =>
      this.enqueueExclusive(async () => {
        const imageTensor = new ort.Tensor('float32', preprocessed.image, [
          1,
          3,
          512,
          512,
        ]);
        const validHwArray = new BigInt64Array([
          BigInt(preprocessed.valid_hw[0]),
          BigInt(preprocessed.valid_hw[1]),
        ]);
        const validHwTensor = new ort.Tensor('int64', validHwArray, [2]);

        const feeds: Record<string, ort.Tensor> = {
          image: imageTensor,
          valid_hw: validHwTensor,
        };

        const owned: ort.Tensor[] = [imageTensor, validHwTensor];
        try {
          let results: ort.InferenceSession.OnnxValueMapType;
          try {
            results = await this.rawSession.run(feeds, [
              ...MIRAGE_EXPERIMENTAL_ONNX_FINAL_OUTPUTS,
            ]);
          } catch (err) {
            throw new OnnxInferenceError('Mirage-Exp-ONNX', 'run', err);
          }
          if (results && typeof results === 'object') {
            for (const val of Object.values(results)) {
              if (val && typeof val === 'object' && 'data' in val) {
                owned.push(val as ort.Tensor);
              }
            }
          }
          const detTensor = results?.detections;
          const protoTensor = results?.prototypes;
          const coeffTensor = results?.coefficients;
          if (!detTensor?.data || !protoTensor?.data || !coeffTensor?.data) {
            throw new OnnxInferenceError(
              'Mirage-Exp-ONNX',
              'output',
              'Missing required Mirage terminal outputs (detections, prototypes, coefficients)'
            );
          }

          // Copy out Float32Array views before disposing ORT tensors
          const detections = new Float32Array(detTensor.data as Float32Array);
          const prototypes = new Float32Array(protoTensor.data as Float32Array);
          const coefficients = new Float32Array(
            coeffTensor.data as Float32Array
          );
          return { detections, prototypes, coefficients };
        } finally {
          disposeTensors(owned);
        }
      })
    );
  }

  public async detectForVideo(
    videoElement: InferenceInput,
    confidenceThreshold = 0.3,
    extraOptions?: MirageExperimentalInferOptions
  ): Promise<DetectedObject[]> {
    if (this.owner.closed) return [];
    const totalStart = performance.now();

    const preStart = performance.now();
    const preprocessed = this.preprocessInput(videoElement);
    const preprocessMs = performance.now() - preStart;

    const infStart = performance.now();
    const rawHeads = await this.runRawHeads(preprocessed);
    const inferenceMs = performance.now() - infStart;
    if (this.owner.closed) return [];

    const postStart = performance.now();
    const effectiveConfidence = Math.max(
      0.05,
      Math.min(0.95, confidenceThreshold)
    );
    const nmsIou =
      extraOptions?.experimentalNmsIou !== undefined
        ? extraOptions.experimentalNmsIou
        : null;
    const includeOcr = extraOptions?.includeOcr ?? true;
    const showObjects = extraOptions?.showObjectProposals ?? true;
    const showText = (extraOptions?.showTextProposals ?? true) && includeOcr;
    const showMasks = extraOptions?.showDiagnosticMasks ?? true;
    const overlayNative = extraOptions?.overlayNativeWasm ?? false;

    const decoded = decodeMirageExperimentalOutputs(
      rawHeads,
      preprocessed.metadata,
      {
        confidence: effectiveConfidence,
        nmsIou,
        maxObjects: 64,
        maxText: 32,
        includeMasks: showMasks,
      }
    );

    const onnxObjects = convertProposalsToDetectedObjects(decoded, {
      showObjectProposals: showObjects,
      showTextProposals: showText,
      showDiagnosticMasks: showMasks,
      maxDetections: extraOptions?.maxDetections ?? 64,
      sourceAttributionPrefix: true,
    });

    let nativeObjects: DetectedObject[] = [];
    if (overlayNative && !this.owner.closed && this.nativeOverlayTaskFactory) {
      try {
        if (!this.nativeOverlayTask) {
          this.nativeOverlayTask = await this.nativeOverlayTaskFactory();
        }
        if (!this.owner.closed && this.nativeOverlayTask) {
          const rawNative = await this.nativeOverlayTask.detectForVideo(
            videoElement,
            effectiveConfidence,
            {
              analysisMode:
                extraOptions?.analysisMode ?? ('fast' as MirageAnalysisMode),
              includeOcr,
              maxDetections: extraOptions?.maxDetections ?? 15,
            }
          );
          nativeObjects = rawNative.map((obj, idx) => ({
            ...obj,
            id: `mirage-native-overlay-${idx}-${obj.id}`,
            labelJa: `[公式WASM] ${obj.labelJa}`,
            category: `${obj.category} (公式WASM)`,
            color: obj.category === 'テキスト (OCR)' ? '#34d399' : '#10b981',
          }));
        }
      } catch (err) {
        console.warn(
          'Optional native WASM overlay failed during experimental ONNX run',
          err
        );
      }
    }

    const postprocessMs = performance.now() - postStart;
    const totalMs = performance.now() - totalStart;

    const requestedProvider =
      this.variantId === 'mirage-onnx-exp-webgpu' ? 'webgpu' : 'wasm';
    const activeProvider: 'wasm' | 'webgpu' =
      requestedProvider === 'webgpu' && !this.backend.fallbackReason
        ? 'webgpu'
        : 'wasm';

    this.lastDiagnostics = {
      variantId: this.variantId,
      requestedProvider,
      configuredProviders: this.backend.configuredProviders as readonly (
        | 'webgpu'
        | 'wasm'
      )[],
      activeProvider,
      fallbackReason: this.backend.fallbackReason,
      artifactSha256: MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256,
      artifactSize: MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE,
      opset: MIRAGE_EXPERIMENTAL_ONNX_OPSET,
      tensorCoverage: `${MIRAGE_EXPERIMENTAL_ONNX_TENSOR_COUNT}/${MIRAGE_TOTAL_CHECKPOINT_TENSOR_COUNT}`,
      validHw: preprocessed.valid_hw,
      metadata: preprocessed.metadata,
      nmsIou,
      confidenceThreshold: effectiveConfidence,
      objectProposalCount: decoded.objects.length,
      textProposalCount: decoded.text.length,
      nativeOverlayEnabled: overlayNative,
      nativeOverlayCount: nativeObjects.length,
      timingsMs: {
        preprocess: preprocessMs,
        inference: inferenceMs,
        postprocess: postprocessMs,
        total: totalMs,
      },
    };

    return [...onnxObjects, ...nativeObjects];
  }

  public async close(): Promise<void> {
    this.resizeCanvas = null;
    this.resizeCtx = null;
    if (this.nativeOverlayTask) {
      try {
        this.nativeOverlayTask.close();
      } catch {
        // Ignore cleanup errors
      }
      this.nativeOverlayTask = null;
    }
    await this.owner.close();
  }
}

export interface CreateMirageExperimentalOnnxOptions {
  variantId: MirageExperimentalVariantId;
  recordAssetSource?: AssetSourceRecorder;
  nativeOverlayTaskFactory?: () => Promise<MirageTask>;
}

export async function createMirageExperimentalOnnxTask(
  options: CreateMirageExperimentalOnnxOptions
): Promise<MirageExperimentalOnnxEngine> {
  const { variantId, recordAssetSource, nativeOverlayTaskFactory } = options;
  if (!isMirageExperimentalVariant(variantId)) {
    throw new RangeError(
      `Unsupported experimental Mirage ONNX variant: ${String(variantId)}`
    );
  }

  const asset = await resolveModelAssetUrls(
    'mirage',
    variantId as MirageVariantId
  );
  const delegate: 'WebGPU' | 'WASM' =
    variantId === 'mirage-onnx-exp-webgpu' ? 'WebGPU' : 'WASM';

  const { session, backend } = await createOrtSession(
    asset.primaryUrl,
    delegate,
    {
      fallbackModelUrl: asset.fallbackUrl,
      variantId,
      recordAssetSource,
      modelId: 'mirage',
      primarySource: asset.source,
    }
  );

  return new MirageExperimentalOnnxEngine(
    session,
    variantId,
    backend ?? unknownOnnxBackend(),
    nativeOverlayTaskFactory
  );
}

export interface MirageExperimentalBrowserBridge {
  ort: typeof ort;
  createMirageExperimentalOnnxTask: typeof createMirageExperimentalOnnxTask;
  verifyMirageExperimentalOnnxBytes: typeof verifyMirageExperimentalOnnxBytes;
  computeMirageCeil32Metadata: typeof computeMirageCeil32Metadata;
  packResizedPixelsToMirageTensor: typeof packResizedPixelsToMirageTensor;
  decodeMirageExperimentalOutputs: typeof decodeMirageExperimentalOutputs;
  convertProposalsToDetectedObjects: typeof convertProposalsToDetectedObjects;
}

declare global {
  interface Window {
    __mirageExperimentalBridge?: MirageExperimentalBrowserBridge;
  }
}

if (typeof window !== 'undefined') {
  window.__mirageExperimentalBridge = {
    ort,
    createMirageExperimentalOnnxTask,
    verifyMirageExperimentalOnnxBytes,
    computeMirageCeil32Metadata,
    packResizedPixelsToMirageTensor,
    decodeMirageExperimentalOutputs,
    convertProposalsToDetectedObjects,
  };
}

