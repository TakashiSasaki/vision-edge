/**
 * Generic Walsh-Hadamard, tensor-layout and ONNX serialization helpers.
 * These helpers alone do not implement or validate the learned Mirage model.
 */
import { mirageWebGpuUnavailableError } from './mirageAvailability.ts';

export const MIRAGE_WEBGPU_INPUT_SIZE = 256;
export const MIRAGE_WEBGPU_GRID_SIZE = 32; // H/8 x W/8 (256 / 8 = 32)
export const MIRAGE_WEBGPU_PROTO_SIZE = 64; // H/4 x W/4 (256 / 4 = 64)
export const MIRAGE_WEBGPU_NUM_PROTOS = 32;

export const MIRAGE_OPEN_VOCAB_LABELS: readonly string[] = [
  'person',
  'face',
  'hand',
  'chair',
  'table',
  'laptop',
  'cell phone',
  'cup',
  'bottle',
  'book',
  'box',
  'bag',
  'tv',
  'keyboard',
  'mouse',
  'door',
  'window',
  'potted plant',
  'car',
  'dog',
  'cat',
  'clock',
  'glasses',
  'pen',
  'paper',
  'card',
  'bowl',
  'spoon',
  'fork',
  'knife',
  'couch',
  'bed',
] as const;

/**
 * Bit-reversal of an `nBits`-wide unsigned integer `x`.
 */
export function bitReverse(x: number, nBits: number): number {
  let rev = 0;
  for (let i = 0; i < nBits; i++) {
    if ((x >> i) & 1) {
      rev |= 1 << (nBits - 1 - i);
    }
  }
  return rev;
}

/**
 * Popcount (Hamming weight) of a non-negative integer.
 */
export function popcount32(x: number): number {
  let v = x >>> 0;
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

/**
 * Builds the exact sequency-ordered Walsh-Hadamard orthogonal matrix W in R^{D x D}
 * as defined in Cactus Compute's HConvGLU architecture:
 *   s(j) = bit_reverse(j ^ (j >> 1))
 *   W[j, k] = (-1)^popcount(s(j) & k) / sqrt(D)
 */
export function buildSequencyWalshMatrix(dim: number): Float32Array {
  const nBits = Math.round(Math.log2(dim));
  if ((1 << nBits) !== dim) {
    throw new RangeError(`Walsh-Hadamard dimension must be a power of 2, got ${dim}`);
  }
  const scale = 1 / Math.sqrt(dim);
  const mat = new Float32Array(dim * dim);
  for (let j = 0; j < dim; j++) {
    const gray = j ^ (j >> 1);
    const seq = bitReverse(gray, nBits);
    const rowOffset = j * dim;
    for (let k = 0; k < dim; k++) {
      const parity = popcount32(seq & k) & 1;
      mat[rowOffset + k] = parity ? -scale : scale;
    }
  }
  return mat;
}

/**
 * Fuses a 1D banded spectral convolution kernel `bandedConv` of shape `[dim, 2 * radius + 1]`
 * with the sequency-ordered Walsh-Hadamard orthogonal matrix `W` (`[dim, dim]`), producing
 * a standard dense `1x1` Conv weight matrix `[dim, dim, 1, 1]` suitable for WebGPU execution.
 *
 * - When `direction === 'forward'`, computes `B * W` (spectral analysis + banded filter).
 * - When `direction === 'inverse'`, computes `W * B` (banded filter + inverse Walsh synthesis).
 */
export function fuseHConvGluKernel(
  bandedConv: Float32Array,
  dim: number,
  radius: number,
  direction: 'forward' | 'inverse' = 'forward'
): Float32Array {
  const bandWidth = 2 * radius + 1;
  if (bandedConv.length !== dim * bandWidth) {
    throw new RangeError(
      `Expected bandedConv of length ${dim * bandWidth}, got ${bandedConv.length}`
    );
  }
  const W = buildSequencyWalshMatrix(dim);
  const B = new Float32Array(dim * dim);

  for (let c = 0; c < dim; c++) {
    const bandRow = c * bandWidth;
    for (let delta = -radius; delta <= radius; delta++) {
      const col = c + delta;
      if (col >= 0 && col < dim) {
        B[c * dim + col] = bandedConv[bandRow + (delta + radius)];
      }
    }
  }

  const out = new Float32Array(dim * dim);
  const A = direction === 'forward' ? B : W;
  const M = direction === 'forward' ? W : B;

  for (let i = 0; i < dim; i++) {
    const iRow = i * dim;
    for (let k = 0; k < dim; k++) {
      const aik = A[iRow + k];
      if (aik === 0) continue;
      const kRow = k * dim;
      for (let j = 0; j < dim; j++) {
        out[iRow + j] += aik * M[kRow + j];
      }
    }
  }
  return out;
}

/**
 * Transposes a Safetensors convolution weight from `[H, W, C_in, C_out]` (HWIO)
 * to ONNX `Conv` layout `[C_out, C_in, H, W]` (OIHW).
 */
export function transposeHwioToOihw(
  src: Float32Array,
  kh: number,
  kw: number,
  cIn: number,
  cOut: number
): Float32Array {
  const dst = new Float32Array(cOut * cIn * kh * kw);
  for (let h = 0; h < kh; h++) {
    for (let w = 0; w < kw; w++) {
      for (let i = 0; i < cIn; i++) {
        for (let o = 0; o < cOut; o++) {
          const srcIdx = ((h * kw + w) * cIn + i) * cOut + o;
          const dstIdx = ((o * cIn + i) * kh + h) * kw + w;
          dst[dstIdx] = src[srcIdx];
        }
      }
    }
  }
  return dst;
}

/**
 * Transposes a 2D linear kernel `[C_in, C_out]` from `mirage.safetensors`
 * into an ONNX 1x1 `Conv` weight `[C_out, C_in, 1, 1]`.
 */
export function transposeLinearToConv1x1(
  src: Float32Array,
  cIn: number,
  cOut: number
): Float32Array {
  const dst = new Float32Array(cOut * cIn);
  for (let i = 0; i < cIn; i++) {
    for (let o = 0; o < cOut; o++) {
      dst[o * cIn + i] = src[i * cOut + o];
    }
  }
  return dst;
}

// ============================================================================
// Minimal, Zero-Dependency ONNX Protobuf (IR v8, Opset 17) Binary Encoder
// ============================================================================

function encodeVarint(value: number | bigint): number[] {
  let v = BigInt(value);
  if (v < 0n) {
    v = BigInt.asUintN(64, v);
  }
  const bytes: number[] = [];
  while (v >= 0x80n) {
    bytes.push(Number(v & 0x7fn) | 0x80);
    v >>= 7n;
  }
  bytes.push(Number(v));
  return bytes;
}

function encodeTag(fieldNumber: number, wireType: number): Uint8Array {
  return new Uint8Array(encodeVarint((fieldNumber << 3) | wireType));
}

function concatBytes(chunks: readonly Uint8Array[]): Uint8Array {
  let total = 0;
  for (const c of chunks) total += c.byteLength;
  const out = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.byteLength;
  }
  return out;
}

function encodeLengthDelimited(fieldNumber: number, payload: Uint8Array): Uint8Array {
  const tag = encodeTag(fieldNumber, 2);
  const len = new Uint8Array(encodeVarint(payload.byteLength));
  return concatBytes([tag, len, payload]);
}

function encodeStringField(fieldNumber: number, str: string): Uint8Array {
  return encodeLengthDelimited(fieldNumber, new TextEncoder().encode(str));
}

function encodeVarintField(fieldNumber: number, value: number | bigint): Uint8Array {
  const tag = encodeTag(fieldNumber, 0);
  const val = new Uint8Array(encodeVarint(value));
  return concatBytes([tag, val]);
}

export interface OnnxTensorInit {
  name: string;
  dims: readonly number[];
  data: Float32Array;
}

export interface OnnxAttributeInit {
  name: string;
  ints?: readonly number[];
  int?: number;
  float?: number;
}

export interface OnnxNodeInit {
  name: string;
  opType: string;
  inputs: readonly string[];
  outputs: readonly string[];
  attributes?: readonly OnnxAttributeInit[];
}

export interface OnnxValueInfoInit {
  name: string;
  dims: readonly number[];
}

function encodeTensorProto(tensor: OnnxTensorInit): Uint8Array {
  const parts: Uint8Array[] = [];
  for (const d of tensor.dims) {
    parts.push(encodeVarintField(1, d));
  }
  // data_type = 1 (FLOAT)
  parts.push(encodeVarintField(2, 1));
  parts.push(encodeStringField(8, tensor.name));
  const rawBytes = new Uint8Array(
    tensor.data.buffer,
    tensor.data.byteOffset,
    tensor.data.byteLength
  );
  parts.push(encodeLengthDelimited(9, rawBytes));
  return concatBytes(parts);
}

function encodeAttributeProto(attr: OnnxAttributeInit): Uint8Array {
  const parts: Uint8Array[] = [encodeStringField(1, attr.name)];
  if (attr.ints !== undefined) {
    for (const v of attr.ints) {
      parts.push(encodeVarintField(8, v));
    }
    // AttributeType.INTS = 7
    parts.push(encodeVarintField(20, 7));
  } else if (attr.int !== undefined) {
    parts.push(encodeVarintField(3, attr.int));
    // AttributeType.INT = 2
    parts.push(encodeVarintField(20, 2));
  } else if (attr.float !== undefined) {
    const fBuf = new ArrayBuffer(4);
    new DataView(fBuf).setFloat32(0, attr.float, true);
    parts.push(concatBytes([encodeTag(2, 5), new Uint8Array(fBuf)]));
    // AttributeType.FLOAT = 1
    parts.push(encodeVarintField(20, 1));
  }
  return concatBytes(parts);
}

function encodeNodeProto(node: OnnxNodeInit): Uint8Array {
  const parts: Uint8Array[] = [];
  for (const inp of node.inputs) {
    parts.push(encodeStringField(1, inp));
  }
  for (const out of node.outputs) {
    parts.push(encodeStringField(2, out));
  }
  parts.push(encodeStringField(3, node.name));
  parts.push(encodeStringField(4, node.opType));
  if (node.attributes) {
    for (const attr of node.attributes) {
      parts.push(encodeLengthDelimited(5, encodeAttributeProto(attr)));
    }
  }
  return concatBytes(parts);
}

function encodeValueInfoProto(info: OnnxValueInfoInit): Uint8Array {
  const dimMessages: Uint8Array[] = info.dims.map((d) =>
    encodeLengthDelimited(1, encodeVarintField(1, d))
  );
  const shapeProto = concatBytes(dimMessages);
  const tensorTypeProto = concatBytes([
    encodeVarintField(1, 1), // elem_type = FLOAT (1)
    encodeLengthDelimited(2, shapeProto),
  ]);
  const typeProto = encodeLengthDelimited(1, tensorTypeProto);
  return concatBytes([
    encodeStringField(1, info.name),
    encodeLengthDelimited(2, typeProto),
  ]);
}

export function serializeOnnxModelProto(spec: {
  graphName: string;
  inputs: readonly OnnxValueInfoInit[];
  outputs: readonly OnnxValueInfoInit[];
  initializers: readonly OnnxTensorInit[];
  nodes: readonly OnnxNodeInit[];
}): Uint8Array {
  const graphParts: Uint8Array[] = [];
  for (const node of spec.nodes) {
    graphParts.push(encodeLengthDelimited(1, encodeNodeProto(node)));
  }
  graphParts.push(encodeStringField(2, spec.graphName));
  for (const init of spec.initializers) {
    graphParts.push(encodeLengthDelimited(5, encodeTensorProto(init)));
  }
  for (const inp of spec.inputs) {
    graphParts.push(encodeLengthDelimited(11, encodeValueInfoProto(inp)));
  }
  for (const out of spec.outputs) {
    graphParts.push(encodeLengthDelimited(12, encodeValueInfoProto(out)));
  }
  const graphProto = concatBytes(graphParts);

  const opsetProto = concatBytes([
    encodeStringField(1, ''),
    encodeVarintField(2, 17),
  ]);

  return concatBytes([
    encodeVarintField(1, 8), // ir_version = 8
    encodeStringField(2, 'cactus-mirage-webgpu-converter'), // producer_name = 2
    encodeLengthDelimited(7, graphProto), // graph = 7
    encodeLengthDelimited(8, opsetProto), // opset_import = 8
  ]);
}

/**
 * A faithful Mirage export is not available yet. The former implementation
 * generated hand-written filters instead of preserving the learned model.
 * Never emit an artifact until an exporter passes reference-output validation.
 */
export function buildMirageWebGpuOnnxBytes(): never {
  throw mirageWebGpuUnavailableError();
}
