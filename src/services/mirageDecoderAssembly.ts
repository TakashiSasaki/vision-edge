/**
 * Full-pipeline Mirage browser inference helpers:
 * - Bilinear text crop resizing and sliding strip window extraction for `mirage-reader.onnx`
 * - Region slot feature & Fourier box assembly for `mirage-cross-kv.onnx`
 * - Autoregressive greedy decoding loop with KV/tap/engram history state for `mirage-decoder-step.onnx`
 * - Grammar parser (`parseMirageDecodedText`) and structured output assembler (`assembleMirageObservation`)
 * - Mapping into VisionStudio `MirageRawObservation` & `DetectedObject[]` without heuristic hallucination
 */
import * as ort from 'onnxruntime-web';
import type { BoundingBox, DetectedObject } from '../types/vision';
import {
  ctcCollapse,
  MIRAGE_EMBEDDING_DIM,
  MIRAGE_SUPPORTED_LANGUAGES,
  MirageDecoderTableReader,
} from './mirageCactCodec';
import type {
  MirageRawObservation,
  MirageRegion,
  MirageObject,
} from './mirageVisionAdapter';
import { traceMaskContourPolygon } from './mirageWebGpuService';
import { disposeTensors, OnnxInferenceError } from './onnxSession';

export const MIRAGE_READER_STRIP_H = 48;
export const MIRAGE_READER_STRIP_W = 320;
export const MIRAGE_READER_STRIDE = 224;
export const MIRAGE_MAX_SLOTS = 96;
export const MIRAGE_MAX_M = 256 + MIRAGE_MAX_SLOTS; // 352
export const MIRAGE_MAX_SEQ = 516;
export const MIRAGE_CROSS_KV_HEADS = 8;
export const MIRAGE_CROSS_HEAD_DIM = 48;
export const MIRAGE_CROSS_V_DIM = 64;
export const MIRAGE_DECODER_LAYERS = 8;
export const MIRAGE_DECODER_KV_HEADS = 2;
export const MIRAGE_DECODER_ROPE_THETA = 100_000.0;
export const MIRAGE_FULL_DEFAULT_MAX_OBJECTS = 20;
export const MIRAGE_FULL_DEFAULT_MAX_TEXT = 20;
export const MIRAGE_FULL_DEFAULT_MAX_TOKENS = 512;

export const MIRAGE_FULL_ENCODER_OUTPUTS = [
  'memory',
  'dense_features',
  'detections',
  'prototypes',
  'coefficients',
] as const;

export interface MirageProposalLike {
  kind: 'object_proposal' | 'text_proposal';
  index: number;
  cell: [number, number];
  score: number;
  /** Normalized [x1, y1, x2, y2] in [0, 1] relative to original image */
  box: [number, number, number, number];
  /** [x1, y1, x2, y2] in [0, 1000] integer coordinates */
  bbox: [number, number, number, number];
  /** Normalized [x1, y1, x2, y2] on the padded 512x512 canvas (`0..w/512`, `0..h/512`) */
  canvasBox?: [number, number, number, number];
  mask?: Uint8Array;
  maskActivePixels?: number;
}

export interface MirageEncoderMetaLike {
  W: number;
  H: number;
  w: number;
  h: number;
  r: number;
  coordinate_scale?: readonly [number, number, number, number];
}

export interface MirageReaderWindow {
  tensorData: Float32Array;
  lo: number;
  hi: number;
  xs: number;
}

export interface MirageReaderCropPlan {
  stripWidth: number;
  nFrames: number;
  windows: MirageReaderWindow[];
}

export interface MirageCrossKvInputArrays {
  validCells: Float32Array;
  slotDFourier: Float32Array;
  slotPooled: Float32Array;
  slotEmb: Float32Array;
  slotValid: Float32Array;
  refs: number[];
}

export interface MirageParsedEntity {
  n: number;
  name: string;
  position: string | null;
  refs: Array<{ kind: 'obj' | 'txt'; index: number }>;
  description: Record<string, string>;
  count?: number | string;
}

export interface MirageParsedDecodedText {
  scene: Record<string, string>;
  entities: MirageParsedEntity[];
  text_refs: number[];
  relations: Array<{ subject: number; predicate: string; object: number }>;
  events: string[];
  unparsed: string[];
  reasoning: string;
}

export interface MirageAssembledObject {
  id: number;
  name: string;
  bbox?: [number, number, number, number];
  score?: number;
  count?: number | string;
  instances?: Array<{ bbox: [number, number, number, number]; score: number }>;
  position?: string;
  description?: Record<string, string>;
  mask?: number[] | number[][];
}

export interface MirageAssembledRegion {
  id: number;
  name: string;
  count?: number | string;
  position?: string;
  description?: Record<string, string>;
}

export interface MirageAssembledTextLine {
  string: string;
  bbox: [number, number, number, number];
  score: number;
  mentioned: boolean;
}

export interface MirageAssembledObservation {
  scene: Record<string, string>;
  objects: MirageAssembledObject[];
  regions: MirageAssembledRegion[];
  text: MirageAssembledTextLine[];
  relations: Array<{ subject: number; predicate: string; object: number }>;
  events: string[];
  reasoning: string;
  unparsed?: string[];
  language: string;
  rawDecodedText: string;
  generatedTokenIds: number[];
}

export interface MirageOrtSessionRunner {
  run(
    feeds: ort.InferenceSession.FeedsType,
    fetches: readonly string[]
  ): Promise<ort.InferenceSession.OnnxValueMapType>;
}

let cachedRopeCos: Float32Array | null = null;
let cachedRopeSin: Float32Array | null = null;

/**
 * Returns precomputed RoPE cosine/sine tables `[MIRAGE_MAX_SEQ, 24]` for `MirageDecoderStep`.
 */
export function getMirageRopeTables(): {
  cosTable: Float32Array;
  sinTable: Float32Array;
} {
  if (cachedRopeCos && cachedRopeSin) {
    return { cosTable: cachedRopeCos, sinTable: cachedRopeSin };
  }
  const halfDim = MIRAGE_CROSS_HEAD_DIM / 2; // 24
  const cosTable = new Float32Array(MIRAGE_MAX_SEQ * halfDim);
  const sinTable = new Float32Array(MIRAGE_MAX_SEQ * halfDim);
  const freqs = new Float64Array(halfDim);
  for (let k = 0; k < halfDim; k++) {
    freqs[k] = 1.0 / Math.pow(MIRAGE_DECODER_ROPE_THETA, (2 * k) / MIRAGE_CROSS_HEAD_DIM);
  }
  for (let pos = 0; pos < MIRAGE_MAX_SEQ; pos++) {
    const base = pos * halfDim;
    for (let k = 0; k < halfDim; k++) {
      const ang = pos * freqs[k];
      cosTable[base + k] = Math.fround(Math.cos(ang));
      sinTable[base + k] = Math.fround(Math.sin(ang));
    }
  }
  cachedRopeCos = cosTable;
  cachedRopeSin = sinTable;
  return { cosTable, sinTable };
}

/**
 * Computes `[64]` Fourier positional features matching `Mirage.fourier(box)` in `reference.py`.
 */
export function buildFourierBox(
  box: readonly [number, number, number, number]
): Float32Array {
  const out = new Float32Array(64);
  for (let c = 0; c < 4; c++) {
    const coord = box[c];
    const base = c * 16;
    for (let r = 0; r < 8; r++) {
      const angle = Math.pow(2, r) * Math.PI * coord;
      out[base + r] = Math.fround(Math.sin(angle));
      out[base + 8 + r] = Math.fround(Math.cos(angle));
    }
  }
  return out;
}

/**
 * Builds the `48 x 320` sliding reader windows (`stride = 224`, neutral `127` padding, `v / 127.5 - 1`)
 * from a source RGBA buffer and `[x0, y0, x1, y1]` box on a `0..1000` scale.
 */
export function extractReaderCropPlan(
  rgba: Uint8ClampedArray | Uint8Array,
  srcW: number,
  srcH: number,
  bbox1000: readonly [number, number, number, number]
): MirageReaderCropPlan {
  const x0f = (bbox1000[0] / 1000) * srcW;
  const y0f = (bbox1000[1] / 1000) * srcH;
  const x1f = (bbox1000[2] / 1000) * srcW;
  const y1f = (bbox1000[3] / 1000) * srcH;

  const ix0 = Math.max(0, Math.min(srcW, Math.floor(Math.min(x0f, x1f))));
  const iy0 = Math.max(0, Math.min(srcH, Math.floor(Math.min(y0f, y1f))));
  const ix1 = Math.max(0, Math.min(srcW, Math.ceil(Math.max(x0f, x1f))));
  const iy1 = Math.max(0, Math.min(srcH, Math.ceil(Math.max(y0f, y1f))));

  const cw = ix1 - ix0;
  const ch = iy1 - iy0;
  if (cw < 1 || ch < 1) {
    return { stripWidth: 0, nFrames: 0, windows: [] };
  }

  const U = Math.max(1, Math.round((MIRAGE_READER_STRIP_H * cw) / ch));
  // Separable H-then-V bilinear resize with uint8 rounding per pass (matching Pillow BILINEAR)
  const hResized = new Uint8Array(ch * U * 3);
  const sx0Arr = new Int32Array(U);
  const sx1Arr = new Int32Array(U);
  const wxArr = new Float32Array(U);
  for (let dx = 0; dx < U; dx++) {
    const sxFloat = Math.max(
      0,
      Math.min(cw - 1, ((dx + 0.5) * cw) / U - 0.5)
    );
    const sx0 = Math.floor(sxFloat);
    sx0Arr[dx] = sx0;
    sx1Arr[dx] = Math.min(cw - 1, sx0 + 1);
    wxArr[dx] = sxFloat - sx0;
  }
  for (let y = 0; y < ch; y++) {
    const srcRowBase = ((iy0 + y) * srcW + ix0) * 4;
    const dstRowBase = y * U * 3;
    for (let dx = 0; dx < U; dx++) {
      const p0 = srcRowBase + sx0Arr[dx] * 4;
      const p1 = srcRowBase + sx1Arr[dx] * 4;
      const wx = wxArr[dx];
      const w0x = 1 - wx;
      const out = dstRowBase + dx * 3;
      for (let c = 0; c < 3; c++) {
        hResized[out + c] = Math.round(rgba[p0 + c] * w0x + rgba[p1 + c] * wx);
      }
    }
  }

  const stripRgb = new Float32Array(MIRAGE_READER_STRIP_H * U * 3);
  for (let dy = 0; dy < MIRAGE_READER_STRIP_H; dy++) {
    const syFloat = Math.max(
      0,
      Math.min(ch - 1, ((dy + 0.5) * ch) / MIRAGE_READER_STRIP_H - 0.5)
    );
    const sy0 = Math.floor(syFloat);
    const sy1 = Math.min(ch - 1, sy0 + 1);
    const wy = syFloat - sy0;
    const w0y = 1 - wy;
    const row0Base = sy0 * U * 3;
    const row1Base = sy1 * U * 3;
    const dstRowBase = dy * U * 3;
    for (let dx = 0; dx < U; dx++) {
      const i0 = row0Base + dx * 3;
      const i1 = row1Base + dx * 3;
      const out = dstRowBase + dx * 3;
      for (let c = 0; c < 3; c++) {
        stripRgb[out + c] = Math.round(
          hResized[i0 + c] * w0y + hResized[i1 + c] * wy
        );
      }
    }
  }

  const starts: number[] = [];
  if (U <= MIRAGE_READER_STRIP_W) {
    starts.push(0);
  } else {
    const stop = U - MIRAGE_READER_STRIP_W + (MIRAGE_READER_STRIDE - 1);
    for (let s = 0; s < stop; s += MIRAGE_READER_STRIDE) {
      if (s === 0 || s < U) {
        starts.push(s);
      }
    }
  }

  const nFrames = Math.ceil(U / 4);
  const padNorm = 127.0 / 127.5 - 1.0;
  const planeSize = MIRAGE_READER_STRIP_H * MIRAGE_READER_STRIP_W;
  const windows: MirageReaderWindow[] = [];

  for (const xs of starts) {
    const tensorData = new Float32Array(3 * planeSize);
    tensorData.fill(padNorm);
    const segW = Math.min(MIRAGE_READER_STRIP_W, U - xs);
    for (let dy = 0; dy < MIRAGE_READER_STRIP_H; dy++) {
      const dstRowBase = dy * MIRAGE_READER_STRIP_W;
      const srcRowBase = dy * U + xs;
      for (let dx = 0; dx < segW; dx++) {
        const srcIdx = (srcRowBase + dx) * 3;
        const dstIdx = dstRowBase + dx;
        tensorData[dstIdx] = stripRgb[srcIdx] / 127.5 - 1.0;
        tensorData[planeSize + dstIdx] = stripRgb[srcIdx + 1] / 127.5 - 1.0;
        tensorData[2 * planeSize + dstIdx] = stripRgb[srcIdx + 2] / 127.5 - 1.0;
      }
    }
    const lo = xs === 0 ? 0 : 12;
    const hi =
      xs + MIRAGE_READER_STRIP_W >= U
        ? Math.ceil((U - xs) / 4)
        : 68;
    windows.push({
      tensorData,
      lo,
      hi: Math.min(hi, 80),
      xs,
    });
  }

  return { stripWidth: U, nFrames, windows };
}

/**
 * Runs `mirage-reader.onnx` across every text crop and collapses frame probabilities using CTC greedy decoding.
 */
export async function readMirageTextLines(
  readerSession: MirageOrtSessionRunner,
  rgba: Uint8ClampedArray | Uint8Array,
  srcW: number,
  srcH: number,
  textProposals: readonly MirageProposalLike[],
  charset: string,
  isStale?: () => boolean
): Promise<string[]> {
  const results: string[] = [];
  for (const proposal of textProposals) {
    if (isStale?.()) {
      break;
    }
    const plan = extractReaderCropPlan(rgba, srcW, srcH, proposal.bbox);
    if (plan.windows.length === 0) {
      results.push('');
      continue;
    }
    const labels: number[] = [];
    for (const win of plan.windows) {
      if (isStale?.()) {
        break;
      }
      const cropTensor = new ort.Tensor('float32', win.tensorData, [
        1,
        3,
        MIRAGE_READER_STRIP_H,
        MIRAGE_READER_STRIP_W,
      ]);
      const owned: ort.Tensor[] = [cropTensor];
      try {
        const outputs = await readerSession.run({ crop: cropTensor }, ['probs']);
        const probsTensor = outputs?.probs;
        if (!probsTensor?.data) {
          throw new OnnxInferenceError(
            'Mirage-Exp-ONNX',
            'output',
            'MirageReader output `probs` missing'
          );
        }
        owned.push(probsTensor);
        const probs = probsTensor.data as Float32Array;
        const dims = probsTensor.dims;
        const classes =
          dims && dims.length >= 1 && Number(dims[dims.length - 1]) > 0
            ? Number(dims[dims.length - 1])
            : 307;
        for (let t = win.lo; t < win.hi; t++) {
          if (Math.floor(win.xs / 4) + t >= plan.nFrames) {
            break;
          }
          const frameOff = t * classes;
          let bestClass = 0;
          let bestVal = -Infinity;
          for (let c = 0; c < classes; c++) {
            const v = probs[frameOff + c];
            if (v > bestVal) {
              bestVal = v;
              bestClass = c;
            }
          }
          labels.push(bestClass);
        }
      } finally {
        disposeTensors(owned);
      }
    }
    results.push(ctcCollapse(labels, charset));
  }
  return results;
}

const GRID_16_CENTRES: Float64Array = (() => {
  const arr = new Float64Array(16);
  for (let i = 0; i < 16; i++) {
    arr[i] = (i + 0.5) / 16;
  }
  return arr;
})();

function resolveCanvasBox(
  proposal: MirageProposalLike,
  meta: MirageEncoderMetaLike
): [number, number, number, number] {
  if (proposal.canvasBox) {
    return proposal.canvasBox;
  }
  const scale = meta.coordinate_scale ?? [
    512 / meta.w,
    512 / meta.h,
    512 / meta.w,
    512 / meta.h,
  ];
  return [
    proposal.box[0] / (scale[0] || 1),
    proposal.box[1] / (scale[1] || 1),
    proposal.box[2] / (scale[2] || 1),
    proposal.box[3] / (scale[3] || 1),
  ];
}

/**
 * Constructs `slot_d_fourier`, `slot_pooled`, `slot_emb`, `valid_cells`, `slot_valid`, and
 * permitted reference token IDs for `mirage-cross-kv.onnx`.
 */
export function buildCrossKvInputArrays(
  E: Float32Array,
  D: Float32Array,
  meta: MirageEncoderMetaLike,
  objects: readonly MirageProposalLike[],
  texts: readonly MirageProposalLike[],
  tableReader: MirageDecoderTableReader
): MirageCrossKvInputArrays {
  if (E.length !== 256 * MIRAGE_EMBEDDING_DIM) {
    throw new OnnxInferenceError(
      'Mirage-Exp-ONNX',
      'output',
      `Unexpected encoder memory length ${E.length} (expected ${256 * MIRAGE_EMBEDDING_DIM})`
    );
  }
  if (D.length !== 128 * 64 * 64) {
    throw new OnnxInferenceError(
      'Mirage-Exp-ONNX',
      'output',
      `Unexpected dense_features length ${D.length} (expected ${128 * 64 * 64})`
    );
  }

  const validCells = new Float32Array(256);
  for (let r = 0; r < 16; r++) {
    const validR = r * 32 < meta.h;
    for (let c = 0; c < 16; c++) {
      validCells[r * 16 + c] = validR && c * 32 < meta.w ? 1.0 : 0.0;
    }
  }

  const slotDFourier = new Float32Array(MIRAGE_MAX_SLOTS * 192);
  const slotPooled = new Float32Array(MIRAGE_MAX_SLOTS * MIRAGE_EMBEDDING_DIM);
  const slotEmb = new Float32Array(MIRAGE_MAX_SLOTS * MIRAGE_EMBEDDING_DIM);
  const slotValid = new Float32Array(MIRAGE_MAX_SLOTS);
  const refs: number[] = [];

  const scaleEmb = Math.sqrt(MIRAGE_EMBEDDING_DIM);
  const rowScratch = new Float32Array(MIRAGE_EMBEDDING_DIM);
  const insideMask = new Uint8Array(256);

  const combined: Array<{
    kind: 'object' | 'text';
    kindIndex: number;
    proposal: MirageProposalLike;
  }> = [];

  for (let i = 0; i < objects.length && combined.length < MIRAGE_MAX_SLOTS; i++) {
    combined.push({ kind: 'object', kindIndex: i, proposal: objects[i] });
  }
  for (let i = 0; i < texts.length && combined.length < MIRAGE_MAX_SLOTS; i++) {
    combined.push({ kind: 'text', kindIndex: i, proposal: texts[i] });
  }

  for (let s = 0; s < combined.length; s++) {
    const { kind, kindIndex, proposal } = combined[s];
    const b = resolveCanvasBox(proposal, meta);
    const [ci, cj] = proposal.cell;

    // 1. D[:, ci, cj] (first 128) + fourier(b) (next 64)
    const dOutBase = s * 192;
    const cellOffset = ci * 64 + cj;
    for (let ch = 0; ch < 128; ch++) {
      slotDFourier[dOutBase + ch] = D[ch * 4096 + cellOffset];
    }
    const fourier = buildFourierBox(b);
    slotDFourier.set(fourier, dOutBase + 128);

    // 2. Pooled E cells inside box `b` (or center cell fallback)
    insideMask.fill(0);
    let insideCount = 0;
    for (let r = 0; r < 16; r++) {
      const cy = GRID_16_CENTRES[r];
      const inY = cy >= b[1] && cy <= b[3];
      for (let c = 0; c < 16; c++) {
        const idx = r * 16 + c;
        if (
          inY &&
          GRID_16_CENTRES[c] >= b[0] &&
          GRID_16_CENTRES[c] <= b[2] &&
          validCells[idx] > 0.5
        ) {
          insideMask[idx] = 1;
          insideCount++;
        }
      }
    }
    const centerR = Math.min(15, Math.max(0, Math.trunc(((b[1] + b[3]) / 2) * 16)));
    const centerC = Math.min(15, Math.max(0, Math.trunc(((b[0] + b[2]) / 2) * 16)));
    const centerIdx = centerR * 16 + centerC;
    if (validCells[centerIdx] > 0.5 && insideMask[centerIdx] === 0) {
      insideMask[centerIdx] = 1;
      insideCount++;
    }
    if (insideCount === 0) {
      insideMask[centerIdx] = 1;
      insideCount = 1;
    }

    const poolOutBase = s * MIRAGE_EMBEDDING_DIM;
    const invCount = 1 / insideCount;
    for (let cellIdx = 0; cellIdx < 256; cellIdx++) {
      if (insideMask[cellIdx]) {
        const eBase = cellIdx * MIRAGE_EMBEDDING_DIM;
        for (let d = 0; d < MIRAGE_EMBEDDING_DIM; d++) {
          slotPooled[poolOutBase + d] += E[eBase + d];
        }
      }
    }
    for (let d = 0; d < MIRAGE_EMBEDDING_DIM; d++) {
      slotPooled[poolOutBase + d] *= invCount;
    }

    // 3. Scaled token embedding for `<objK>` / `<txtK>`
    const rid = tableReader.vocab.refId(kind, kindIndex);
    tableReader.readTokenEmbedding(rid, rowScratch);
    const embOutBase = s * MIRAGE_EMBEDDING_DIM;
    for (let d = 0; d < MIRAGE_EMBEDDING_DIM; d++) {
      slotEmb[embOutBase + d] = rowScratch[d] * scaleEmb;
    }

    slotValid[s] = 1.0;
    refs.push(rid);
  }

  return {
    validCells,
    slotDFourier,
    slotPooled,
    slotEmb,
    slotValid,
    refs,
  };
}

export interface MirageCrossKvOutputs {
  crossK: Float32Array;
  crossV: Float32Array;
  crossMask: Float32Array;
  refs: number[];
}

/**
 * Executes `mirage-cross-kv.onnx` on encoder features + built slot inputs.
 */
export async function runMirageCrossKv(
  crossKvSession: MirageOrtSessionRunner,
  E: Float32Array,
  D: Float32Array,
  meta: MirageEncoderMetaLike,
  objects: readonly MirageProposalLike[],
  texts: readonly MirageProposalLike[],
  tableReader: MirageDecoderTableReader
): Promise<MirageCrossKvOutputs> {
  const inputs = buildCrossKvInputArrays(E, D, meta, objects, texts, tableReader);
  const owned: ort.Tensor[] = [
    new ort.Tensor('float32', new Float32Array(E), [1, 256, MIRAGE_EMBEDDING_DIM]),
    new ort.Tensor('float32', inputs.slotDFourier, [1, MIRAGE_MAX_SLOTS, 192]),
    new ort.Tensor('float32', inputs.slotPooled, [
      1,
      MIRAGE_MAX_SLOTS,
      MIRAGE_EMBEDDING_DIM,
    ]),
    new ort.Tensor('float32', inputs.slotEmb, [
      1,
      MIRAGE_MAX_SLOTS,
      MIRAGE_EMBEDDING_DIM,
    ]),
    new ort.Tensor('float32', inputs.validCells, [256]),
    new ort.Tensor('float32', inputs.slotValid, [MIRAGE_MAX_SLOTS]),
  ];
  try {
    const out = await crossKvSession.run(
      {
        E: owned[0],
        slot_d_fourier: owned[1],
        slot_pooled: owned[2],
        slot_emb: owned[3],
        valid_cells: owned[4],
        slot_valid: owned[5],
      },
      ['ck', 'cv', 'cross_mask']
    );
    const ckTensor = out?.ck;
    const cvTensor = out?.cv;
    const cmTensor = out?.cross_mask;
    if (!ckTensor?.data || !cvTensor?.data || !cmTensor?.data) {
      throw new OnnxInferenceError(
        'Mirage-Exp-ONNX',
        'output',
        'MirageCrossKv outputs missing'
      );
    }
    owned.push(ckTensor, cvTensor, cmTensor);
    return {
      crossK: new Float32Array(ckTensor.data as Float32Array),
      crossV: new Float32Array(cvTensor.data as Float32Array),
      crossMask: new Float32Array(cmTensor.data as Float32Array),
      refs: inputs.refs,
    };
  } finally {
    disposeTensors(owned);
  }
}

export interface MirageGreedyDecodeOptions {
  language?: string;
  maxTokens?: number;
  isStale?: () => boolean;
  /** Captures step-0 decoder logits (`logits` right after `[bos, <|lang|>]`) for numerical parity checks */
  captureStep0Logits?: boolean;
}

export interface MirageGreedyDecodeResult {
  language: string;
  generatedTokenIds: number[];
  decodedText: string;
  stepsExecuted: number;
  step0Logits?: Float32Array;
}

/**
 * Executes the autoregressive `mirage-decoder-step.onnx` loop from `[bos, <|lang|>]` until
 * `</s>` (`vocab.eos`) or `maxTokens` steps.
 */
export async function runMirageGreedyDecode(
  decoderStepSession: MirageOrtSessionRunner,
  crossKv: MirageCrossKvOutputs,
  tableReader: MirageDecoderTableReader,
  options: MirageGreedyDecodeOptions = {}
): Promise<MirageGreedyDecodeResult> {
  const vocab = tableReader.vocab;
  const requestedLang = options.language ?? 'en';
  const language =
    requestedLang in vocab.langIds
      ? requestedLang
      : MIRAGE_SUPPORTED_LANGUAGES.find((l) => l in vocab.langIds) ?? 'en';
  const langId = vocab.langIds[language];
  if (langId === undefined) {
    throw new OnnxInferenceError(
      'Mirage-Exp-ONNX',
      'input',
      `Language token <|${language}|> not present in Mirage vocabulary`
    );
  }

  const maxTokens = Math.min(
    options.maxTokens ?? MIRAGE_FULL_DEFAULT_MAX_TOKENS,
    MIRAGE_MAX_SEQ - 4
  );

  // Build banned mask: forbid language switch tokens and unreferenced `<objK>`/`<txtK>` slots
  const banned = new Uint8Array(vocab.size);
  for (const id of Object.values(vocab.langIds)) {
    banned[id] = 1;
  }
  for (let id = vocab.refBase; id < vocab.size; id++) {
    banned[id] = 1;
  }
  for (const r of crossKv.refs) {
    if (r >= 0 && r < vocab.size) {
      banned[r] = 0;
    }
  }

  const { cosTable, sinTable } = getMirageRopeTables();
  const halfDim = MIRAGE_CROSS_HEAD_DIM / 2;

  let pastK = new Float32Array(
    MIRAGE_DECODER_LAYERS *
      MIRAGE_DECODER_KV_HEADS *
      MIRAGE_MAX_SEQ *
      MIRAGE_CROSS_HEAD_DIM
  );
  let pastV = new Float32Array(
    MIRAGE_DECODER_LAYERS *
      MIRAGE_DECODER_KV_HEADS *
      MIRAGE_MAX_SEQ *
      MIRAGE_CROSS_V_DIM
  );
  let rawQHist = new Float32Array(MIRAGE_DECODER_LAYERS * 2 * 384);
  let rawKHist = new Float32Array(MIRAGE_DECODER_LAYERS * 2 * 96);
  let rawVHist = new Float32Array(MIRAGE_DECODER_LAYERS * 2 * 128);
  let engVHist = new Float32Array(2 * 9 * MIRAGE_EMBEDDING_DIM);

  const posOnehot = new Float32Array(MIRAGE_MAX_SEQ);
  const selfMask = new Float32Array(MIRAGE_MAX_SEQ).fill(-1e9);

  const tokens: number[] = [vocab.bos, langId];
  const generated: number[] = [];
  let stepsExecuted = 0;

  const stepOne = async (tokenId: number, t: number): Promise<Float32Array> => {
    posOnehot.fill(0);
    posOnehot[t] = 1.0;
    selfMask[t] = 0.0;

    const tokenEmb = tableReader.readTokenEmbedding(tokenId);
    const eng0 = tableReader.readEngramVector(0, tokens, t);
    const eng1 = tableReader.readEngramVector(1, tokens, t);
    const ropeCos = cosTable.subarray(t * halfDim, (t + 1) * halfDim);
    const ropeSin = sinTable.subarray(t * halfDim, (t + 1) * halfDim);

    const owned: ort.Tensor[] = [
      new ort.Tensor('float32', tokenEmb, [MIRAGE_EMBEDDING_DIM]),
      new ort.Tensor('float32', eng0, [MIRAGE_EMBEDDING_DIM]),
      new ort.Tensor('float32', eng1, [MIRAGE_EMBEDDING_DIM]),
      new ort.Tensor('float32', new Float32Array(ropeCos), [halfDim]),
      new ort.Tensor('float32', new Float32Array(ropeSin), [halfDim]),
      new ort.Tensor('float32', new Float32Array(posOnehot), [MIRAGE_MAX_SEQ]),
      new ort.Tensor('float32', new Float32Array(selfMask), [MIRAGE_MAX_SEQ]),
      new ort.Tensor('float32', crossKv.crossK, [
        MIRAGE_DECODER_LAYERS,
        MIRAGE_CROSS_KV_HEADS,
        MIRAGE_MAX_M,
        MIRAGE_CROSS_HEAD_DIM,
      ]),
      new ort.Tensor('float32', crossKv.crossV, [
        MIRAGE_DECODER_LAYERS,
        MIRAGE_CROSS_KV_HEADS,
        MIRAGE_MAX_M,
        MIRAGE_CROSS_V_DIM,
      ]),
      new ort.Tensor('float32', crossKv.crossMask, [MIRAGE_MAX_M]),
      new ort.Tensor('float32', pastK, [
        MIRAGE_DECODER_LAYERS,
        MIRAGE_DECODER_KV_HEADS,
        MIRAGE_MAX_SEQ,
        MIRAGE_CROSS_HEAD_DIM,
      ]),
      new ort.Tensor('float32', pastV, [
        MIRAGE_DECODER_LAYERS,
        MIRAGE_DECODER_KV_HEADS,
        MIRAGE_MAX_SEQ,
        MIRAGE_CROSS_V_DIM,
      ]),
      new ort.Tensor('float32', rawQHist, [MIRAGE_DECODER_LAYERS, 2, 384]),
      new ort.Tensor('float32', rawKHist, [MIRAGE_DECODER_LAYERS, 2, 96]),
      new ort.Tensor('float32', rawVHist, [MIRAGE_DECODER_LAYERS, 2, 128]),
      new ort.Tensor('float32', engVHist, [2, 9, MIRAGE_EMBEDDING_DIM]),
    ];

    try {
      const outputs = await decoderStepSession.run(
        {
          token_emb: owned[0],
          eng0_emb: owned[1],
          eng1_emb: owned[2],
          rope_cos: owned[3],
          rope_sin: owned[4],
          pos_onehot: owned[5],
          self_mask: owned[6],
          cross_k: owned[7],
          cross_v: owned[8],
          cross_mask: owned[9],
          past_k: owned[10],
          past_v: owned[11],
          raw_q_hist: owned[12],
          raw_k_hist: owned[13],
          raw_v_hist: owned[14],
          eng_v_hist: owned[15],
        },
        [
          'logits',
          'new_past_k',
          'new_past_v',
          'new_raw_q_hist',
          'new_raw_k_hist',
          'new_raw_v_hist',
          'new_eng_v_hist',
        ]
      );

      for (const val of Object.values(outputs)) {
        if (val && typeof val === 'object' && 'data' in val) {
          owned.push(val);
        }
      }

      const logitsT = outputs?.logits;
      const npk = outputs?.new_past_k;
      const npv = outputs?.new_past_v;
      const nrq = outputs?.new_raw_q_hist;
      const nrk = outputs?.new_raw_k_hist;
      const nrv = outputs?.new_raw_v_hist;
      const neh = outputs?.new_eng_v_hist;
      if (
        !logitsT?.data ||
        !npk?.data ||
        !npv?.data ||
        !nrq?.data ||
        !nrk?.data ||
        !nrv?.data ||
        !neh?.data
      ) {
        throw new OnnxInferenceError(
          'Mirage-Exp-ONNX',
          'output',
          `MirageDecoderStep outputs missing at step ${t}`
        );
      }

      pastK = new Float32Array(npk.data as Float32Array);
      pastV = new Float32Array(npv.data as Float32Array);
      rawQHist = new Float32Array(nrq.data as Float32Array);
      rawKHist = new Float32Array(nrk.data as Float32Array);
      rawVHist = new Float32Array(nrv.data as Float32Array);
      engVHist = new Float32Array(neh.data as Float32Array);
      stepsExecuted++;
      return new Float32Array(logitsT.data as Float32Array);
    } finally {
      disposeTensors(owned);
    }
  };

  await stepOne(vocab.bos, 0);
  if (options.isStale?.()) {
    return {
      language,
      generatedTokenIds: generated,
      decodedText: '',
      stepsExecuted,
    };
  }
  let logits = await stepOne(langId, 1);
  const step0Logits = options.captureStep0Logits
    ? new Float32Array(logits)
    : undefined;

  for (let genIdx = 0; genIdx < maxTokens; genIdx++) {
    if (options.isStale?.()) {
      break;
    }
    let bestId = vocab.eos;
    let bestScore = -Infinity;
    const nLogits = Math.min(logits.length, banned.length);
    for (let id = 0; id < nLogits; id++) {
      if (banned[id]) continue;
      const s = logits[id];
      if (s > bestScore) {
        bestScore = s;
        bestId = id;
      }
    }
    if (bestId === vocab.eos) {
      break;
    }
    generated.push(bestId);
    tokens.push(bestId);
    const t = tokens.length - 1;
    if (t >= MIRAGE_MAX_SEQ) {
      break;
    }
    logits = await stepOne(bestId, t);
  }

  return {
    language,
    generatedTokenIds: generated,
    decodedText: vocab.decode(generated),
    stepsExecuted,
    ...(step0Logits ? { step0Logits } : {}),
  };
}

const THINK_RE = /<think>([\s\S]*?)<\/think>/;
const ENTITY_RE = /^E(\d+)\s+(.*)$/;
const TRAILING_ATTRS_RE = /\{(.*)\}\s*$/;
const REF_GLOBAL_RE = /<(obj|txt)(\d+)>/g;
const COUNT_SUFFIX_RE = /\s*#(\S+)$/;
const RELATION_RE = /^R\s+(\d+)\s+(\S+)\s+(\d+)\s*$/;

function parseAttrs(s: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const part of s.split(';')) {
    const eqIdx = part.indexOf('=');
    if (eqIdx >= 0) {
      const k = part.slice(0, eqIdx).trim();
      const v = part.slice(eqIdx + 1).trim();
      if (k) {
        out[k] = v;
      }
    } else if (part.trim() && out.note === undefined) {
      out.note = part.trim();
    }
  }
  return out;
}

/**
 * Exact TypeScript port of `parse_decoded(text)` in `oasis/reference.py`.
 */
export function parseMirageDecodedText(text: string): MirageParsedDecodedText {
  let trace = '';
  let body = text;
  const tm = THINK_RE.exec(text);
  if (tm) {
    trace = tm[1].trim();
    body = text.slice(tm.index + tm[0].length);
  }
  body = body.replace(/<\/s>/g, '');

  const rec: MirageParsedDecodedText = {
    scene: {},
    entities: [],
    text_refs: [],
    relations: [],
    events: [],
    unparsed: [],
    reasoning: trace,
  };

  for (const rawLine of body.split('\n')) {
    const line = rawLine.trim();
    if (!line) continue;

    if (line.startsWith('S ')) {
      Object.assign(rec.scene, parseAttrs(line.slice(2)));
      continue;
    }

    const em = ENTITY_RE.exec(line);
    if (em) {
      const n = Number.parseInt(em[1], 10);
      let rest = em[2];
      let attrs: Record<string, string> = {};
      const am = TRAILING_ATTRS_RE.exec(rest);
      if (am) {
        attrs = parseAttrs(am[1]);
        rest = rest.slice(0, am.index).trim();
      }
      const refs: Array<{ kind: 'obj' | 'txt'; index: number }> = [];
      for (const m of rest.matchAll(REF_GLOBAL_RE)) {
        refs.push({
          kind: m[1] === 'obj' ? 'obj' : 'txt',
          index: Number.parseInt(m[2], 10),
        });
      }
      rest = rest.replace(REF_GLOBAL_RE, '').replace(/\*/g, ' ').trim();
      let pos: string | null = null;
      const atIdx = rest.indexOf('@');
      if (atIdx >= 0) {
        pos = rest.slice(atIdx + 1).trim();
        rest = rest.slice(0, atIdx);
      }
      let name = rest.trim();
      let count: number | string | undefined;
      const cm = COUNT_SUFFIX_RE.exec(name);
      if (cm) {
        const rawCount = cm[1];
        name = name.slice(0, cm.index).trim();
        count = /^\d+$/.test(rawCount) ? Number.parseInt(rawCount, 10) : rawCount;
      }
      const ent: MirageParsedEntity = {
        n,
        name,
        position: pos,
        refs,
        description: attrs,
      };
      if (count !== undefined) {
        ent.count = count;
      }
      rec.entities.push(ent);
      continue;
    }

    if (line.startsWith('T')) {
      for (const m of line.matchAll(REF_GLOBAL_RE)) {
        if (m[1] === 'txt') {
          rec.text_refs.push(Number.parseInt(m[2], 10));
        }
      }
      continue;
    }

    const rm = RELATION_RE.exec(line);
    if (rm) {
      rec.relations.push({
        subject: Number.parseInt(rm[1], 10),
        predicate: rm[2],
        object: Number.parseInt(rm[3], 10),
      });
      continue;
    }

    if (line.startsWith('V ') || line.startsWith('EV')) {
      const sp = line.indexOf(' ');
      rec.events.push(sp >= 0 ? line.slice(sp + 1) : line);
      continue;
    }

    rec.unparsed.push(line);
  }

  return rec;
}

/**
 * Extracts a `0..1000` scale boundary polygon `[x1, y1, x2, y2, ...]` from a `128x128` proposal mask
 * using the existing Moore-Neighbor + Douglas-Peucker contour tracer.
 */
export function extractProposalMaskPolygon1000(
  proposal: MirageProposalLike
): number[] | undefined {
  if (!proposal.mask || proposal.mask.length !== 128 * 128) {
    return undefined;
  }
  const [x1, y1, x2, y2] = proposal.box;
  const bbox: BoundingBox = {
    x: x1,
    y: y1,
    width: Math.max(1 / 128, x2 - x1),
    height: Math.max(1 / 128, y2 - y1),
  };
  const minPx = Math.max(0, Math.floor(x1 * 128) - 1);
  const minPy = Math.max(0, Math.floor(y1 * 128) - 1);
  const maxPx = Math.min(128, Math.ceil(x2 * 128) + 1);
  const maxPy = Math.min(128, Math.ceil(y2 * 128) + 1);
  const poly = traceMaskContourPolygon(
    proposal.mask,
    128,
    minPx,
    minPy,
    maxPx,
    maxPy,
    bbox
  );
  if (poly.length < 3) return undefined;
  const flat: number[] = [];
  for (const pt of poly) {
    flat.push(Math.round(pt.x * 1000), Math.round(pt.y * 1000));
  }
  return flat;
}

/**
 * Assembles decoded text, object/text detector proposals, CTC OCR strings, and mask contours
 * into a structured Mirage record matching `oasis/reference.py` `assemble(...)`.
 */
export function assembleMirageObservation(params: {
  decodedText: string;
  generatedTokenIds: number[];
  language: string;
  objects: readonly MirageProposalLike[];
  texts: readonly MirageProposalLike[];
  ocrStrings: readonly string[];
  includeMasks?: boolean;
}): MirageAssembledObservation {
  const {
    decodedText,
    generatedTokenIds,
    language,
    objects,
    texts,
    ocrStrings,
    includeMasks = true,
  } = params;
  const rec = parseMirageDecodedText(decodedText);
  const mentionedTextSet = new Set(rec.text_refs);

  const assembledObjects: MirageAssembledObject[] = [];
  const assembledRegions: MirageAssembledRegion[] = [];

  for (const e of rec.entities) {
    const objRefs = e.refs
      .filter((r) => r.kind === 'obj' && r.index >= 0 && r.index < objects.length)
      .map((r) => r.index);

    if (objRefs.length > 0) {
      const primary = objects[objRefs[0]];
      const item: MirageAssembledObject = {
        id: e.n,
        name: e.name,
        bbox: [...primary.bbox],
        score: Number(primary.score.toFixed(3)),
      };
      if (e.count !== undefined) item.count = e.count;
      if (e.position) item.position = e.position;
      if (Object.keys(e.description).length > 0) {
        item.description = e.description;
      }
      if (objRefs.length > 1) {
        item.count = objRefs.length;
        item.instances = objRefs.map((idx) => ({
          bbox: [...objects[idx].bbox],
          score: Number(objects[idx].score.toFixed(3)),
        }));
      }
      if (includeMasks) {
        const poly = extractProposalMaskPolygon1000(primary);
        if (poly) item.mask = poly;
      }
      assembledObjects.push(item);
    } else {
      const region: MirageAssembledRegion = {
        id: e.n,
        name: e.name,
      };
      if (e.count !== undefined) region.count = e.count;
      if (e.position) region.position = e.position;
      if (Object.keys(e.description).length > 0) {
        region.description = e.description;
      }
      assembledRegions.push(region);
    }
  }

  const assembledText: MirageAssembledTextLine[] = [];
  for (let k = 0; k < texts.length; k++) {
    const s = ocrStrings[k] ?? '';
    if (!s) continue;
    assembledText.push({
      string: s,
      bbox: [...texts[k].bbox],
      score: Number(texts[k].score.toFixed(3)),
      mentioned: mentionedTextSet.has(k),
    });
  }

  const out: MirageAssembledObservation = {
    scene: rec.scene,
    objects: assembledObjects,
    regions: assembledRegions,
    text: assembledText,
    relations: rec.relations,
    events: rec.events,
    reasoning: rec.reasoning,
    language,
    rawDecodedText: decodedText,
    generatedTokenIds,
  };
  if (rec.unparsed.length > 0) {
    out.unparsed = rec.unparsed;
  }
  return out;
}

/**
 * Converts a `MirageAssembledObservation` to `MirageRawObservation` compatible with
 * `mapMirageObservationToDetectedObjects(...)`, preserving bbox/mask integrity.
 */
export function toMirageRawObservation(
  assembled: MirageAssembledObservation
): MirageRawObservation {
  const objects: MirageObject[] = assembled.objects
    .filter((o): o is MirageAssembledObject & { bbox: [number, number, number, number]; score: number } =>
      Array.isArray(o.bbox) && typeof o.score === 'number'
    )
    .map((o) => ({
      id: o.id,
      name: o.name,
      bbox: o.bbox,
      score: o.score,
      ...(o.mask ? { mask: o.mask } : {}),
      ...(o.position ? { position: o.position } : {}),
      ...(typeof o.count === 'number' ? { count: o.count } : {}),
      ...(o.description ? { description: o.description } : {}),
    }));

  const regions: MirageRegion[] = assembled.regions.map((r) => ({
    id: r.id,
    name: r.name,
    score: 1.0,
    ...(r.position ? { position: r.position } : {}),
    ...(r.description ? { description: r.description } : {}),
  }));

  return {
    scene: assembled.scene,
    objects,
    regions,
    text: assembled.text.map((t) => ({
      string: t.string,
      bbox: t.bbox,
      score: t.score,
    })),
    relations: assembled.relations,
    events: assembled.events.map((description) => ({ description })),
    reasoning: assembled.reasoning,
  };
}

/**
 * Attaches rendered `128x128` `HTMLCanvasElement` masks onto mapped semantic `DetectedObject`s
 * by matching primary object proposal references (`<objK>`) or tight bbox correspondence.
 */
export function attachDiagnosticMasksToSemanticObjects(
  mappedObjects: DetectedObject[],
  assembled: MirageAssembledObservation,
  decodedText: string,
  proposals: readonly MirageProposalLike[],
  renderMaskCanvas: (mask: Uint8Array, color: string) => HTMLCanvasElement | undefined
): void {
  const rec = parseMirageDecodedText(decodedText);
  const objProposalByEntityId = new Map<number, MirageProposalLike>();
  for (const ent of rec.entities) {
    const firstObjRef = ent.refs.find(
      (r) => r.kind === 'obj' && r.index >= 0 && r.index < proposals.length
    );
    if (firstObjRef !== undefined) {
      objProposalByEntityId.set(ent.n, proposals[firstObjRef.index]);
    }
  }

  let nextAssembledIdx = 0;
  for (const det of mappedObjects) {
    if (det.id.startsWith('mirage-reg-') || det.id.startsWith('mirage-ocr-')) {
      continue;
    }
    let matchedObj = assembled.objects[nextAssembledIdx];
    const idMatch = /^mirage-obj-(\d+)-(\d+)-/.exec(det.id);
    if (idMatch) {
      const obsIdx = Number.parseInt(idMatch[1], 10);
      if (obsIdx >= 0 && obsIdx < assembled.objects.length) {
        matchedObj = assembled.objects[obsIdx];
        nextAssembledIdx = obsIdx + 1;
      }
    } else if (matchedObj) {
      nextAssembledIdx++;
    }
    if (!matchedObj) {
      continue;
    }
    const srcProposal = objProposalByEntityId.get(matchedObj.id);
    if (srcProposal?.mask && !det.maskCanvas) {
      const canvas = renderMaskCanvas(srcProposal.mask, det.color);
      if (canvas) {
        det.maskCanvas = canvas;
      }
    }
  }
}
