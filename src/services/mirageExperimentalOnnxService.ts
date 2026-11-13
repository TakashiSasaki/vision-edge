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
import {
  cancelInFlightMirageWasmInit,
  type MirageInferExtraOptions,
  type MirageSnapshotLike,
  type MirageTask,
} from './mirageWasmService';
import {
  createOrtSession,
  disposeTensors,
  OnnxInferenceError,
  OnnxSessionOwner,
  unknownOnnxBackend,
} from './onnxSession';
import { diagnosticLogger } from './diagnosticLogger';

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

export interface MirageSourceFrameSnapshot extends MirageSnapshotLike {
  readonly frameId: number;
  readonly sourceGeneration: number;
  readonly width: number;
  readonly height: number;
  readonly rgba: Uint8ClampedArray<ArrayBuffer>;
  readonly pixelHash: string;
  readonly rgbaSha256: string;
  readonly canvas: HTMLCanvasElement;
  readonly refCount: number;
  readonly disposed: boolean;
  retain(): MirageSourceFrameSnapshot;
  release(): void;
}

export interface MirageFrameAlignmentDiagnostics {
  frameId: number;
  sourceGeneration: number;
  sourceWidth: number;
  sourceHeight: number;
  sourcePixelHash: string;
  onnxInputPixelHash: string;
  nativeInputPixelHash: string | null;
  sameSourceSnapshot: boolean;
  onnxFrameId: number;
  nativeFrameId: number | null;
  onnxSourceGeneration: number;
  nativeSourceGeneration: number | null;
  onnxRgbaSha256: string;
  nativeRgbaSha256: string | null;
  sameSourceFrame: boolean;
}

export interface WebGpuAdapterDiagnostics {
  available: boolean;
  navigatorGpuPresent: boolean;
  adapterAvailable: boolean;
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  isFallbackAdapter: boolean;
  isSoftwareAdapter: boolean;
  error: string | null;
}

export interface MirageTensorParityMetrics {
  length: number;
  allFinite: boolean;
  actualFinite: boolean;
  expectedFinite: boolean;
  maxAbsDiff: number;
  meanAbsDiff: number;
  rmsDiff: number;
  maxRelDiff: number;
  nearZeroEpsilon: number;
  atol: number;
  rtol: number;
  withinAtolRtol: boolean;
  allcloseAt2e4: boolean;
  violationCount: number;
  violationsAt2e4: number;
  cosineSimilarity: number;
}

export interface MirageExperimentalExecutionDiagnostics {
  variantId: MirageExperimentalVariantId;
  requestedProvider: 'wasm' | 'webgpu';
  configuredProviders: readonly ('webgpu' | 'wasm')[];
  /** Inferred from requestedProvider and absence of fallbackReason; not a direct GPU kernel profiler measurement */
  activeProvider: 'wasm' | 'webgpu';
  /** Direct per-node kernel execution provider attribution from ORT Web ('unmeasured' unless ORT profiling is enabled) */
  actualKernelBackend: 'unmeasured';
  allowWasmFallback: boolean;
  fallbackReason: string | null;
  webgpuAdapter?: WebGpuAdapterDiagnostics | null;
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
  frameAlignment?: MirageFrameAlignmentDiagnostics;
  timingsMs: {
    preprocess: number;
    inference: number;
    postprocess: number;
    total: number;
  };
}

const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

/**
 * Synchronous FIPS 180-4 SHA-256 hex digest for frame provenance hashing before async boundaries.
 */
export function sha256HexSync(data: Uint8Array | Uint8ClampedArray): string {
  const len = data.byteLength;
  const bitLenHi = Math.floor((len * 8) / 0x100000000);
  const bitLenLo = (len * 8) >>> 0;
  const paddedLen = ((len + 9 + 63) >> 6) << 6;
  const padded = new Uint8Array(paddedLen);
  padded.set(data);
  padded[len] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(paddedLen - 8, bitLenHi, false);
  view.setUint32(paddedLen - 4, bitLenLo, false);

  let h0 = 0x6a09e667;
  let h1 = 0xbb67ae85;
  let h2 = 0x3c6ef372;
  let h3 = 0xa54ff53a;
  let h4 = 0x510e527f;
  let h5 = 0x9b05688c;
  let h6 = 0x1f83d9ab;
  let h7 = 0x5be0cd19;

  const w = new Uint32Array(64);
  for (let offset = 0; offset < paddedLen; offset += 64) {
    for (let i = 0; i < 16; i++) {
      w[i] = view.getUint32(offset + i * 4, false);
    }
    for (let i = 16; i < 64; i++) {
      const v15 = w[i - 15];
      const s0 = ((v15 >>> 7) | (v15 << 25)) ^ ((v15 >>> 18) | (v15 << 14)) ^ (v15 >>> 3);
      const v2 = w[i - 2];
      const s1 = ((v2 >>> 17) | (v2 << 15)) ^ ((v2 >>> 19) | (v2 << 13)) ^ (v2 >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }

    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    let f = h5;
    let g = h6;
    let h = h7;

    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + S1 + ch + SHA256_K[i] + w[i]) >>> 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) >>> 0;

      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }

    h0 = (h0 + a) >>> 0;
    h1 = (h1 + b) >>> 0;
    h2 = (h2 + c) >>> 0;
    h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0;
    h5 = (h5 + f) >>> 0;
    h6 = (h6 + g) >>> 0;
    h7 = (h7 + h) >>> 0;
  }

  return [h0, h1, h2, h3, h4, h5, h6, h7]
    .map((v) => v.toString(16).padStart(8, '0'))
    .join('');
}

/**
 * Compares two Float32Array head tensors with explicit near-zero relative-error policy
 * (`denom = max(|expected|, nearZeroEpsilon)`), finite-value checks, RMS, and `atol`/`rtol` verification.
 */
export function compareMirageTensorParity(
  actual: Float32Array,
  expected: Float32Array,
  optionsOrAtol:
    | {
        atol?: number;
        rtol?: number;
        nearZeroEpsilon?: number;
      }
    | number = {},
  rtolArg?: number,
  nearZeroFloorArg?: number
): MirageTensorParityMetrics {
  const opts =
    typeof optionsOrAtol === 'number'
      ? {
          atol: optionsOrAtol,
          rtol: rtolArg ?? 2e-4,
          nearZeroEpsilon: nearZeroFloorArg ?? 1e-6,
        }
      : optionsOrAtol;
  const atol = opts.atol ?? 2e-4;
  const rtol = opts.rtol ?? 2e-4;
  const nearZeroEpsilon = opts.nearZeroEpsilon ?? 1e-6;
  const n = Math.min(actual.length, expected.length);

  let actualFinite = true;
  let expectedFinite = true;
  let maxAbsDiff = 0;
  let sumAbsDiff = 0;
  let sumSqDiff = 0;
  let maxRelDiff = 0;
  let violationCount =
    actual.length === expected.length
      ? 0
      : Math.abs(actual.length - expected.length);
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < n; i++) {
    const a = actual[i];
    const b = expected[i];
    const aFin = Number.isFinite(a);
    const bFin = Number.isFinite(b);
    if (!aFin) actualFinite = false;
    if (!bFin) expectedFinite = false;
    if (!aFin || !bFin) {
      violationCount++;
      continue;
    }
    const diff = Math.abs(a - b);
    if (diff > maxAbsDiff) maxAbsDiff = diff;
    sumAbsDiff += diff;
    sumSqDiff += diff * diff;

    const denom = Math.max(Math.abs(b), nearZeroEpsilon);
    const rel = diff / denom;
    if (rel > maxRelDiff) maxRelDiff = rel;

    if (diff > atol + rtol * Math.abs(b)) {
      violationCount++;
    }

    dot += a * b;
    normA += a * a;
    normB += b * b;
  }

  const allFinite =
    actualFinite && expectedFinite && actual.length === expected.length;
  const meanAbsDiff = n > 0 ? sumAbsDiff / n : 0;
  const rmsDiff = n > 0 ? Math.sqrt(sumSqDiff / n) : 0;
  const cosineSimilarity =
    normA > 0 && normB > 0 ? dot / (Math.sqrt(normA) * Math.sqrt(normB)) : 0;
  const withinAtolRtol = allFinite && violationCount === 0;

  return {
    length: n,
    allFinite,
    actualFinite,
    expectedFinite,
    maxAbsDiff,
    meanAbsDiff,
    rmsDiff,
    maxRelDiff,
    nearZeroEpsilon,
    atol,
    rtol,
    withinAtolRtol,
    allcloseAt2e4: withinAtolRtol,
    violationCount,
    violationsAt2e4: violationCount,
    cosineSimilarity,
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

function getInputDimensions(
  input: InferenceInput | HTMLImageElement | MirageSourceFrameSnapshot
): { width: number; height: number } {
  if ('frameId' in input && 'rgba' in input) {
    return {
      width: input.width || 640,
      height: input.height || 480,
    };
  }
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
  if ('naturalWidth' in input) {
    return {
      width: input.naturalWidth || input.width || 640,
      height: input.naturalHeight || input.height || 480,
    };
  }
  return {
    width: input.width || 640,
    height: input.height || 480,
  };
}

function isSnapshotObject(
  input: unknown
): input is MirageSourceFrameSnapshot {
  return (
    Boolean(input) &&
    typeof input === 'object' &&
    'frameId' in (input as Record<string, unknown>) &&
    'rgba' in (input as Record<string, unknown>) &&
    'pixelHash' in (input as Record<string, unknown>) &&
    typeof (input as MirageSourceFrameSnapshot).retain === 'function'
  );
}

function resampleSnapshotRgba(
  srcRgba: Uint8ClampedArray<ArrayBuffer>,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number
): Uint8ClampedArray<ArrayBuffer> {
  if (srcW === dstW && srcH === dstH) {
    return srcRgba;
  }
  const out = new Uint8ClampedArray(dstW * dstH * 4);
  for (let y = 0; y < dstH; y++) {
    const sy = Math.min(srcH - 1, Math.floor(((y + 0.5) * srcH) / dstH));
    const srcRow = sy * srcW * 4;
    const dstRow = y * dstW * 4;
    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(srcW - 1, Math.floor(((x + 0.5) * srcW) / dstW));
      const sp = srcRow + sx * 4;
      const dp = dstRow + x * 4;
      out[dp] = srcRgba[sp];
      out[dp + 1] = srcRgba[sp + 1];
      out[dp + 2] = srcRgba[sp + 2];
      out[dp + 3] = srcRgba[sp + 3];
    }
  }
  return out;
}

/**
 * Captures source pixels synchronously before any `await`, returning a reference-counted
 * immutable `MirageSourceFrameSnapshot` with its own isolated offscreen `HTMLCanvasElement`.
 */
export function captureMirageSourceFrameSnapshot(
  input: InferenceInput | HTMLImageElement,
  frameId: number,
  sourceGeneration: number
): MirageSourceFrameSnapshot {
  const existing = (input as { __mirageSnapshot?: MirageSourceFrameSnapshot })
    .__mirageSnapshot;
  if (existing && !existing.disposed) {
    return existing.retain();
  }

  const { width: srcW, height: srcH } = getInputDimensions(input);
  const canvas = document.createElement('canvas');
  canvas.width = srcW;
  canvas.height = srcH;
  const ctx = canvas.getContext('2d', {
    willReadFrequently: true,
  });
  if (!ctx) {
    throw new OnnxInferenceError(
      'Mirage-Exp-ONNX',
      'input',
      'Failed to acquire 2D canvas context for Mirage frame snapshot capture'
    );
  }

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.clearRect(0, 0, srcW, srcH);
  ctx.drawImage(input as CanvasImageSource, 0, 0, srcW, srcH);
  const rawData = ctx.getImageData(0, 0, srcW, srcH).data;
  const rgba = new Uint8ClampedArray(rawData.length);
  rgba.set(rawData);
  const pixelHash = sha256HexSync(rgba);

  let refCount = 1;
  let disposed = false;

  const snapshot: MirageSourceFrameSnapshot = {
    frameId,
    sourceGeneration,
    width: srcW,
    height: srcH,
    rgba,
    pixelHash,
    rgbaSha256: pixelHash,
    canvas,
    get refCount() {
      return refCount;
    },
    get disposed() {
      return disposed;
    },
    retain(): MirageSourceFrameSnapshot {
      if (disposed) {
        throw new Error(
          `Cannot retain disposed MirageSourceFrameSnapshot (frameId=${frameId})`
        );
      }
      refCount++;
      return snapshot;
    },
    release(): void {
      if (disposed || refCount <= 0) return;
      refCount--;
      if (refCount === 0) {
        disposed = true;
        delete (canvas as { __mirageSnapshot?: MirageSourceFrameSnapshot })
          .__mirageSnapshot;
        canvas.width = 0;
        canvas.height = 0;
      }
    },
  };

  (canvas as { __mirageSnapshot?: MirageSourceFrameSnapshot }).__mirageSnapshot =
    snapshot;
  return snapshot;
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
  public backend: OnnxBackendInfo;
  public readonly variantId: MirageExperimentalVariantId;
  public readonly webgpuAdapter: WebGpuAdapterDiagnostics | null;
  private rawSession: ort.InferenceSession;
  private owner: OnnxSessionOwner;
  private nativeOverlayTask: MirageTask | null = null;
  private nativeOverlayInitPromise: Promise<MirageTask> | null = null;
  private readonly nativeOverlayTaskFactory?: () => Promise<MirageTask>;
  private readonly modelBytes?: Uint8Array;
  private readonly allowWasmFallback: boolean;
  private lastDiagnostics: MirageExperimentalExecutionDiagnostics | null = null;
  private runMutex: Promise<unknown> = Promise.resolve();
  private nextFrameId = 0;
  private sourceGeneration = 1;
  private lastSourceElement: InferenceInput | null = null;

  constructor(
    session: ort.InferenceSession,
    variantId: MirageExperimentalVariantId,
    backend?: OnnxBackendInfo,
    nativeOverlayTaskFactory?: () => Promise<MirageTask>,
    modelBytes?: Uint8Array,
    allowWasmFallback = true,
    webgpuAdapter: WebGpuAdapterDiagnostics | null = null
  ) {
    this.variantId = variantId;
    this.allowWasmFallback = allowWasmFallback;
    this.webgpuAdapter = webgpuAdapter;
    const requestedProvider =
      variantId === 'mirage-onnx-exp-webgpu' ? 'webgpu' : 'wasm';
    this.backend =
      backend ??
      ({
        runtime: 'onnxruntime-web',
        requestedProvider,
        configuredProviders:
          requestedProvider === 'webgpu'
            ? allowWasmFallback
              ? ['webgpu', 'wasm']
              : ['webgpu']
            : ['wasm'],
        fallbackReason: null,
        actualBackend: 'unmeasured',
      } satisfies OnnxBackendInfo);
    this.rawSession = session;
    this.owner = new OnnxSessionOwner(session, 'Mirage-Exp-ONNX');
    this.nativeOverlayTaskFactory = nativeOverlayTaskFactory;
    this.modelBytes = modelBytes;
  }

  public getLastDiagnostics(): MirageExperimentalExecutionDiagnostics | null {
    return this.lastDiagnostics;
  }

  public notifySourceChanged(): number {
    this.lastSourceElement = null;
    this.nativeOverlayTask?.notifySourceChanged?.();
    return ++this.sourceGeneration;
  }

  public getSourceGeneration(): number {
    return this.sourceGeneration;
  }

  private enqueueExclusive<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.runMutex.then(fn, fn);
    this.runMutex = next.then(
      () => undefined,
      () => undefined
    );
    return next;
  }

  public preprocessInput(
    input: InferenceInput | HTMLImageElement | MirageSourceFrameSnapshot
  ): MiragePreprocessedInput {
    const snapshot = isSnapshotObject(input)
      ? input
      : (input as { __mirageSnapshot?: MirageSourceFrameSnapshot })
          .__mirageSnapshot;

    if (snapshot) {
      if (snapshot.disposed) {
        throw new OnnxInferenceError(
          'Mirage-Exp-ONNX',
          'input',
          'Cannot preprocess a disposed MirageSourceFrameSnapshot'
        );
      }
      snapshot.retain();
    }

    try {
      const { width: srcW, height: srcH } = getInputDimensions(input);
      const metadata = computeMirageCeil32Metadata(srcW, srcH);
      const { w, h } = metadata;

      const resizeCanvas = document.createElement('canvas');
      resizeCanvas.width = w;
      resizeCanvas.height = h;
      const ctx = resizeCanvas.getContext('2d', {
        willReadFrequently: true,
      });
      if (!ctx) {
        throw new OnnxInferenceError(
          'Mirage-Exp-ONNX',
          'input',
          'Failed to acquire 2D canvas context for Mirage ceil32 preprocessing'
        );
      }

      const drawSource = isSnapshotObject(input)
        ? input.canvas
        : (input as CanvasImageSource);

      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(drawSource, 0, 0, w, h);
      const imageData = ctx.getImageData(0, 0, w, h);
      let pixels: Uint8ClampedArray<ArrayBuffer> = imageData.data;

      if (snapshot) {
        let canvasHasNonZero = false;
        for (let i = 0; i < pixels.length; i += 4) {
          if (pixels[i] !== 0 || pixels[i + 1] !== 0 || pixels[i + 2] !== 0) {
            canvasHasNonZero = true;
            break;
          }
        }
        let snapshotHasNonZero = false;
        if (!canvasHasNonZero) {
          for (let i = 0; i < snapshot.rgba.length; i += 4) {
            if (
              snapshot.rgba[i] !== 0 ||
              snapshot.rgba[i + 1] !== 0 ||
              snapshot.rgba[i + 2] !== 0
            ) {
              snapshotHasNonZero = true;
              break;
            }
          }
        }
        if (!canvasHasNonZero && snapshotHasNonZero) {
          pixels = resampleSnapshotRgba(
            snapshot.rgba,
            snapshot.width,
            snapshot.height,
            w,
            h
          );
        }
      }

      return packResizedPixelsToMirageTensor(pixels, 4, metadata);
    } finally {
      if (snapshot) {
        snapshot.release();
      }
    }
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
            if (
              this.allowWasmFallback &&
              this.backend.configuredProviders.includes('webgpu') &&
              this.modelBytes
            ) {
              const fallbackSession = await ort.InferenceSession.create(
                this.modelBytes,
                {
                  executionProviders: ['wasm'],
                  graphOptimizationLevel: 'all',
                  logSeverityLevel: 3,
                }
              );
              try {
                await Promise.resolve(this.rawSession.release?.());
              } catch {
                // Ignore release failure of broken WebGPU session
              }
              this.rawSession = fallbackSession;
              this.owner = new OnnxSessionOwner(
                fallbackSession,
                'Mirage-Exp-ONNX'
              );
              this.backend = {
                ...this.backend,
                configuredProviders: ['wasm'],
                fallbackReason: `WebGPU runtime inference failed; fell back to WASM: ${
                  err instanceof Error ? err.message : String(err)
                }`,
              };
              results = await this.rawSession.run(feeds, [
                ...MIRAGE_EXPERIMENTAL_ONNX_FINAL_OUTPUTS,
              ]);
            } else {
              throw new OnnxInferenceError('Mirage-Exp-ONNX', 'run', err);
            }
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

  private async ensureNativeOverlayTask(): Promise<MirageTask | null> {
    if (this.owner.closed || !this.nativeOverlayTaskFactory) return null;
    if (this.nativeOverlayTask) return this.nativeOverlayTask;
    if (!this.nativeOverlayInitPromise) {
      const factory = this.nativeOverlayTaskFactory;
      this.nativeOverlayInitPromise = (async () => {
        try {
          const created = await factory();
          if (this.owner.closed) {
            try {
              await Promise.resolve(created.close());
            } catch {
              // Ignore cleanup error after close
            }
            throw new Error('MirageExperimentalOnnxEngine closed during native overlay initialization');
          }
          this.nativeOverlayTask = created;
          return created;
        } finally {
          this.nativeOverlayInitPromise = null;
        }
      })();
    }
    return this.nativeOverlayInitPromise;
  }

  public async detectForVideo(
    videoElement: InferenceInput,
    confidenceThreshold = 0.3,
    extraOptions?: MirageExperimentalInferOptions
  ): Promise<DetectedObject[]> {
    if (this.owner.closed) return [];
    const totalStart = performance.now();

    if (
      this.lastSourceElement !== null &&
      this.lastSourceElement !== videoElement &&
      extraOptions?.sourceGeneration === undefined
    ) {
      this.sourceGeneration++;
    }
    this.lastSourceElement = videoElement;

    if (
      extraOptions?.sourceGeneration !== undefined &&
      extraOptions.sourceGeneration !== this.sourceGeneration
    ) {
      this.sourceGeneration = extraOptions.sourceGeneration;
    }

    const requestGeneration = this.sourceGeneration;
    const frameId = ++this.nextFrameId;
    const isStale = () =>
      this.owner.closed || this.sourceGeneration !== requestGeneration;

    // Synchronously capture an immutable source frame snapshot before any await
    const preStart = performance.now();
    const snapshot =
      extraOptions?.sourceSnapshot && !extraOptions.sourceSnapshot.disposed
        ? (extraOptions.sourceSnapshot.retain() as MirageSourceFrameSnapshot)
        : captureMirageSourceFrameSnapshot(
            videoElement,
            frameId,
            requestGeneration
          );

    try {
      const preprocessed = this.preprocessInput(snapshot);
      const onnxInputPixelHash = snapshot.pixelHash;
      const preprocessMs = performance.now() - preStart;

      const infStart = performance.now();
      const rawHeads = await this.runRawHeads(preprocessed);
      const inferenceMs = performance.now() - infStart;
      if (isStale()) return [];

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
      let nativeInputPixelHash: string | null = null;

      if (overlayNative && !isStale() && this.nativeOverlayTaskFactory) {
        snapshot.retain();
        try {
          const overlayTask = await this.ensureNativeOverlayTask();
          if (!isStale() && overlayTask) {
            nativeInputPixelHash = snapshot.pixelHash;
            const rawNative = await overlayTask.detectForVideo(
              snapshot.canvas,
              effectiveConfidence,
              {
                analysisMode:
                  extraOptions?.analysisMode ?? ('fast' as MirageAnalysisMode),
                includeOcr,
                maxDetections: extraOptions?.maxDetections ?? 15,
                sourceGeneration: requestGeneration,
                sourceSnapshot: snapshot,
              }
            );
            if (isStale()) return [];
            nativeObjects = rawNative.map((obj, idx) => ({
              ...obj,
              id: `mirage-native-overlay-${idx}-${obj.id}`,
              labelJa: `[公式WASM] ${obj.labelJa}`,
              category: `${obj.category} (公式WASM)`,
              color: obj.category === 'テキスト (OCR)' ? '#34d399' : '#10b981',
            }));
          }
        } catch (err) {
          if (isStale()) return [];
          console.warn(
            'Optional native WASM overlay failed during experimental ONNX run',
            err
          );
        } finally {
          snapshot.release();
        }
      }

      if (isStale()) return [];

      const postprocessMs = performance.now() - postStart;
      const totalMs = performance.now() - totalStart;

      const requestedProvider =
        this.variantId === 'mirage-onnx-exp-webgpu' ? 'webgpu' : 'wasm';
      const activeProvider: 'wasm' | 'webgpu' =
        requestedProvider === 'webgpu' && !this.backend.fallbackReason
          ? 'webgpu'
          : 'wasm';

      const sameSourceSnapshot =
        onnxInputPixelHash === snapshot.pixelHash &&
        (!overlayNative ||
          nativeInputPixelHash === null ||
          nativeInputPixelHash === snapshot.pixelHash);

      this.lastDiagnostics = {
        variantId: this.variantId,
        requestedProvider,
        configuredProviders: this.backend.configuredProviders as readonly (
          | 'webgpu'
          | 'wasm'
        )[],
        activeProvider,
        actualKernelBackend: 'unmeasured',
        allowWasmFallback: this.allowWasmFallback,
        fallbackReason: this.backend.fallbackReason,
        webgpuAdapter: this.webgpuAdapter,
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
        frameAlignment: {
          frameId: snapshot.frameId,
          sourceGeneration: snapshot.sourceGeneration,
          sourceWidth: snapshot.width,
          sourceHeight: snapshot.height,
          sourcePixelHash: snapshot.pixelHash,
          onnxInputPixelHash,
          nativeInputPixelHash,
          sameSourceSnapshot,
          onnxFrameId: snapshot.frameId,
          nativeFrameId: nativeInputPixelHash ? snapshot.frameId : null,
          onnxSourceGeneration: snapshot.sourceGeneration,
          nativeSourceGeneration: nativeInputPixelHash
            ? snapshot.sourceGeneration
            : null,
          onnxRgbaSha256: onnxInputPixelHash,
          nativeRgbaSha256: nativeInputPixelHash,
          sameSourceFrame: sameSourceSnapshot,
        },
        timingsMs: {
          preprocess: preprocessMs,
          inference: inferenceMs,
          postprocess: postprocessMs,
          total: totalMs,
        },
      };

      return [...onnxObjects, ...nativeObjects];
    } finally {
      snapshot.release();
    }
  }

  public async close(): Promise<void> {
    this.sourceGeneration++;
    this.lastSourceElement = null;
    if (this.nativeOverlayTask) {
      const taskToClose = this.nativeOverlayTask;
      this.nativeOverlayTask = null;
      try {
        await Promise.resolve(taskToClose.close());
      } catch {
        // Ignore cleanup errors
      }
    }
    await this.owner.close();
  }
}

export interface FetchModelBufferOptions {
  expectedSize?: number;
  onChunk?: (loaded: number, total: number) => void;
  variantId?: string;
}

/**
 * Fetches model bytes from the given URL.
 * When the expected size exceeds Cloud Run's 32 MiB response threshold (e.g. >28 MiB),
 * fetches in 16 MiB chunks using HTTP Range requests to avoid connection drops / synthetic HTTP 500.
 */
export async function fetchModelBuffer(
  url: string,
  options: FetchModelBufferOptions = {}
): Promise<ArrayBuffer> {
  const { expectedSize = MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE } = options;
  const CHUNK_SIZE = 16 * 1024 * 1024; // 16 MiB per chunk

  if (expectedSize && expectedSize > 28 * 1024 * 1024) {
    const totalChunks = Math.ceil(expectedSize / CHUNK_SIZE);
    const combined = new Uint8Array(expectedSize);
    let offset = 0;

    for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
      const start = chunkIdx * CHUNK_SIZE;
      const end = Math.min((chunkIdx + 1) * CHUNK_SIZE - 1, expectedSize - 1);

      const response = await fetch(url, {
        headers: {
          Range: `bytes=${start}-${end}`,
        },
      });

      if (response.status === 206) {
        const chunkBuf = await response.arrayBuffer();
        combined.set(new Uint8Array(chunkBuf), start);
        offset += chunkBuf.byteLength;
        if (options.onChunk) {
          options.onChunk(offset, expectedSize);
        }
        continue;
      }

      if (response.status === 200 && chunkIdx === 0) {
        const fullBuf = await response.arrayBuffer();
        if (fullBuf.byteLength === expectedSize) {
          return fullBuf;
        }
      }

      const errorText = await response.text().catch(() => '');
      throw new Error(
        `Failed to fetch chunk ${chunkIdx + 1}/${totalChunks} (bytes ${start}-${end}) from ${url}: HTTP ${response.status} ${response.statusText} (${errorText})`
      );
    }

    return combined.buffer;
  }

  const response = await fetch(url);
  if (!response.ok) {
    const errorText = await response.text().catch(() => '');
    throw new Error(
      `Failed to fetch model from ${url}: HTTP ${response.status} ${response.statusText} (${errorText})`
    );
  }
  return await response.arrayBuffer();
}

export async function probeWebGpuAdapterDiagnostics(): Promise<WebGpuAdapterDiagnostics> {
  if (
    typeof navigator === 'undefined' ||
    !('gpu' in navigator) ||
    !navigator.gpu
  ) {
    return {
      available: false,
      navigatorGpuPresent: false,
      adapterAvailable: false,
      vendor: '',
      architecture: '',
      device: '',
      description: '',
      isFallbackAdapter: false,
      isSoftwareAdapter: false,
      error: 'navigator.gpu is not present in this runtime',
    };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return {
        available: false,
        navigatorGpuPresent: true,
        adapterAvailable: false,
        vendor: '',
        architecture: '',
        device: '',
        description: '',
        isFallbackAdapter: false,
        isSoftwareAdapter: false,
        error: 'navigator.gpu.requestAdapter() returned null',
      };
    }
    const rawAdapter = adapter as unknown as {
      info?: {
        vendor?: string;
        architecture?: string;
        device?: string;
        description?: string;
      };
      isFallbackAdapter?: boolean;
      requestAdapterInfo?: () => Promise<{
        vendor?: string;
        architecture?: string;
        device?: string;
        description?: string;
      }>;
    };
    const info =
      rawAdapter.info ??
      (typeof rawAdapter.requestAdapterInfo === 'function'
        ? await rawAdapter.requestAdapterInfo().catch(() => undefined)
        : undefined);
    const vendor = String(info?.vendor ?? '');
    const architecture = String(info?.architecture ?? '');
    const device = String(info?.device ?? '');
    const description = String(info?.description ?? '');
    const isFallbackAdapter = Boolean(rawAdapter.isFallbackAdapter);
    const combined = `${vendor} ${architecture} ${device} ${description}`;
    const isSoftwareAdapter =
      isFallbackAdapter ||
      /swiftshader|llvmpipe|lavapipe|software|warp|basic render/i.test(
        combined
      );
    return {
      available: true,
      navigatorGpuPresent: true,
      adapterAvailable: true,
      vendor,
      architecture,
      device,
      description,
      isFallbackAdapter,
      isSoftwareAdapter,
      error: null,
    };
  } catch (err) {
    return {
      available: false,
      navigatorGpuPresent: true,
      adapterAvailable: false,
      vendor: '',
      architecture: '',
      device: '',
      description: '',
      isFallbackAdapter: false,
      isSoftwareAdapter: false,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

export interface CreateMirageExperimentalOnnxOptions {
  variantId: MirageExperimentalVariantId;
  recordAssetSource?: AssetSourceRecorder;
  nativeOverlayTaskFactory?: () => Promise<MirageTask>;
  /** When false and variantId is mirage-onnx-exp-webgpu, strictly requires ['webgpu'] without falling back to WASM */
  allowWasmFallback?: boolean;
}

export async function createMirageExperimentalOnnxTask(
  options: CreateMirageExperimentalOnnxOptions
): Promise<MirageExperimentalOnnxEngine> {
  const {
    variantId,
    recordAssetSource,
    nativeOverlayTaskFactory,
    allowWasmFallback = true,
  } = options;
  if (!isMirageExperimentalVariant(variantId)) {
    throw new RangeError(
      `Unsupported experimental Mirage ONNX variant: ${String(variantId)}`
    );
  }

  cancelInFlightMirageWasmInit();

  diagnosticLogger.log('info', 'resolve', 'mirage', variantId, 'Resolving model asset URLs', {
    details: { variantId, allowWasmFallback },
  });

  const asset = await resolveModelAssetUrls(
    'mirage',
    variantId as MirageVariantId
  );
  const delegate: 'WebGPU' | 'WASM' =
    variantId === 'mirage-onnx-exp-webgpu' ? 'WebGPU' : 'WASM';

  diagnosticLogger.log('info', 'resolve', 'mirage', variantId, 'Asset resolved', {
    details: {
      primaryUrl: asset.primaryUrl,
      fallbackUrl: asset.fallbackUrl,
      source: asset.source,
      delegate,
      allowWasmFallback,
    },
  });

  const webgpuAdapter =
    delegate === 'WebGPU' ? await probeWebGpuAdapterDiagnostics() : null;

  if (delegate === 'WebGPU' && !allowWasmFallback) {
    if (
      typeof navigator === 'undefined' ||
      !('gpu' in navigator) ||
      !navigator.gpu
    ) {
      throw new Error(
        'Strict WebGPU session requested (allowWasmFallback=false), but navigator.gpu is unavailable'
      );
    }
    if (!webgpuAdapter?.available) {
      throw new Error(
        'Strict WebGPU session requested (allowWasmFallback=false), but navigator.gpu.requestAdapter() returned null'
      );
    }
  }

  let verifiedBytes: Uint8Array | undefined;
  if (typeof fetch === 'function') {
    diagnosticLogger.log('info', 'fetch', 'mirage', variantId, 'Starting direct pre-fetch and verification', {
      details: { url: asset.primaryUrl },
    });
    const fetchStart = performance.now();
    try {
      const buffer = await fetchModelBuffer(asset.primaryUrl, {
        expectedSize: MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE,
        variantId,
        onChunk: (loaded, total) => {
          diagnosticLogger.log(
            'info',
            'fetch',
            'mirage',
            variantId,
            `Downloaded ${loaded}/${total} bytes (${Math.round((loaded / total) * 100)}%)`,
            { details: { loaded, total } }
          );
        },
      });
      const fetchMs = Math.round(performance.now() - fetchStart);
      diagnosticLogger.log('info', 'verify', 'mirage', variantId, `Received ${buffer.byteLength} bytes. Verifying SHA-256...`, {
        details: { byteLength: buffer.byteLength, fetchMs },
      });
      const verifyStart = performance.now();
      const verified = await verifyMirageExperimentalOnnxBytes(buffer);
      const verifyMs = Math.round(performance.now() - verifyStart);
      verifiedBytes = new Uint8Array(buffer);
      diagnosticLogger.log('info', 'verify', 'mirage', variantId, 'SHA-256 integrity verified', {
        details: { sha256: verified.sha256, size: verified.size, verifyMs },
      });
    } catch (fetchErr) {
      diagnosticLogger.log('error', 'fetch', 'mirage', variantId, 'Pre-fetch and verification failed', {
        error: fetchErr,
        sendToServer: typeof window !== 'undefined',
      });
      throw fetchErr;
    }
  }

  diagnosticLogger.log('info', 'ort-session', 'mirage', variantId, 'Creating ONNX Runtime session', {
    details: {
      delegate,
      allowWasmFallback,
      hasVerifiedBytes: Boolean(verifiedBytes),
      byteLength: verifiedBytes?.byteLength,
    },
  });

  try {
    const sessionStart = performance.now();
    let session: ort.InferenceSession;
    let backend: OnnxBackendInfo | undefined;

    if (delegate === 'WebGPU' && !allowWasmFallback) {
      if (!verifiedBytes) {
        throw new Error(
          'Strict WebGPU session requires verified model bytes'
        );
      }
      session = await ort.InferenceSession.create(verifiedBytes, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
        logSeverityLevel: 3,
      });
      recordAssetSource?.(
        `mirage:${variantId}`,
        asset.source
      );
      backend = {
        runtime: 'onnxruntime-web',
        requestedProvider: 'webgpu',
        configuredProviders: ['webgpu'],
        fallbackReason: null,
        actualBackend: 'unmeasured',
      };
    } else {
      const created = await createOrtSession(
        verifiedBytes ?? asset.primaryUrl,
        delegate,
        {
          fallbackModelUrl: asset.fallbackUrl,
          fallbackModelBytes: verifiedBytes ? () => verifiedBytes! : undefined,
          variantId,
          recordAssetSource,
          modelId: 'mirage',
          primarySource: asset.source,
        }
      );
      session = created.session;
      backend = created.backend;
    }

    const sessionMs = Math.round(performance.now() - sessionStart);
    diagnosticLogger.log('info', 'ort-session', 'mirage', variantId, 'ONNX session created successfully', {
      details: {
        backend,
        webgpuAdapter,
        sessionMs,
        allowWasmFallback,
        inputNames: session.inputNames,
        outputNames: session.outputNames,
      },
    });

    return new MirageExperimentalOnnxEngine(
      session,
      variantId,
      backend ?? unknownOnnxBackend(),
      nativeOverlayTaskFactory,
      verifiedBytes,
      allowWasmFallback,
      webgpuAdapter
    );
  } catch (ortErr) {
    diagnosticLogger.log('error', 'ort-session', 'mirage', variantId, 'ONNX Runtime session creation failed', {
      error: ortErr,
      sendToServer: typeof window !== 'undefined',
    });
    throw ortErr;
  }
}

export interface MirageExperimentalBrowserBridge {
  ort: typeof ort;
  createMirageExperimentalOnnxTask: typeof createMirageExperimentalOnnxTask;
  fetchModelBuffer: typeof fetchModelBuffer;
  verifyMirageExperimentalOnnxBytes: typeof verifyMirageExperimentalOnnxBytes;
  computeMirageCeil32Metadata: typeof computeMirageCeil32Metadata;
  packResizedPixelsToMirageTensor: typeof packResizedPixelsToMirageTensor;
  decodeMirageExperimentalOutputs: typeof decodeMirageExperimentalOutputs;
  convertProposalsToDetectedObjects: typeof convertProposalsToDetectedObjects;
  compareMirageTensorParity: typeof compareMirageTensorParity;
  probeWebGpuAdapterDiagnostics: typeof probeWebGpuAdapterDiagnostics;
  captureMirageSourceFrameSnapshot: typeof captureMirageSourceFrameSnapshot;
  sha256HexSync: typeof sha256HexSync;
  cancelInFlightMirageWasmInit: typeof cancelInFlightMirageWasmInit;
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
    fetchModelBuffer,
    verifyMirageExperimentalOnnxBytes,
    computeMirageCeil32Metadata,
    packResizedPixelsToMirageTensor,
    decodeMirageExperimentalOutputs,
    convertProposalsToDetectedObjects,
    compareMirageTensorParity,
    probeWebGpuAdapterDiagnostics,
    captureMirageSourceFrameSnapshot,
    sha256HexSync,
    cancelInFlightMirageWasmInit,
  };
}



