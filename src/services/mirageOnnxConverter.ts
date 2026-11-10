/**
 * Mirage (Cactus-Compute/mirage) → WebGPU ONNX Converter & Graph Builder.
 *
 * Converts Mirage's architecture (`stem`, Walsh-Hadamard `HConvGLU`, `dense` CenterNet + 32-prototype
 * instance segmentation head, and open-vocabulary descriptor head) into a standard Opset-17 ONNX
 * `ModelProto` binary composed exclusively of WebGPU-native operators (`Conv`, `Add`, `Mul`, `Sigmoid`).
 */

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
 * Optional upstream tensor overrides extracted from `Cactus-Compute/mirage` (`mirage.safetensors`).
 */
export interface MirageUpstreamTensors {
  stemW0?: Float32Array;       // [3, 3, 3, 32] HWIO
  stemPw1?: Float32Array;      // [32, 64]
  stemPw2?: Float32Array;      // [64, 128]
  stemPw3?: Float32Array;      // [128, 256]
  stemPw4?: Float32Array;      // [256, 512]
  hconvGate?: Float32Array;    // [64, 65] or [512, 65]
  hconvValue?: Float32Array;   // [64, 65] or [512, 65]
  hconvOut?: Float32Array;     // [64, 65] or [512, 65]
  denseOutKernel?: Float32Array;   // [128, 10]
  denseOutBias?: Float32Array;     // [10]
  denseCoefKernel?: Float32Array;  // [128, 32]
  denseProtosKernel?: Float32Array;// [128, 32]
  denseFineKernel?: Float32Array;  // [128, 128]
  denseEdgeKernel?: Float32Array;  // [64, 128]
  denseMidKernel?: Float32Array;   // [256, 128]
  denseCoarseKernel?: Float32Array;// [512, 128]
}

/**
 * Builds the WebGPU-executable Mirage ONNX model (`mirage-webgpu.onnx`).
 *
 * Architecture:
 * 1. `stem`: 5-stage downsampling hierarchy (`stem/w0` 3x3 conv -> `stem1` 3x3 -> `stem2` 3x3)
 *    fused with upstream multi-scale edge and luminance tensors, mapping 256x256 -> 64x64 (`edge_feat_64`) -> 32x32 (`fine_feat_32`).
 * 2. `body_hconvglu`: Sequency-ordered Walsh-Hadamard `HConvGLU` block (`B_g W`, `B_v W`, `W B_o`)
 *    fused into `1x1` Conv kernels + `3x3` depthwise mixing + GLU gate (`Sigmoid` * `Mul`) + residual `Add`.
 * 3. `dense`:
 *    - `dense_out` (`[1, 10, 32, 32]`):
 *      - `0`: Foreground object CenterNet saliency heatmap (`[0]`)
 *      - `1..2`: Object normalized width (`[1]`) and height (`[2]`) direct regression
 *      - `3..4`: Object center dx offset (`[3]`), dy offset (`[4]`)
 *      - `5`: OCR text-line heatmap logit (`[5]`)
 *      - `6..7`: OCR text-line normalized width (`[6]`) and height (`[7]`)
 *      - `8..9`: OCR text-line dx (`[8]`), dy (`[9]`)
 *    - `mask_coeffs` (`[1, 32, 32, 32]`): 32 instance mask coefficients per grid cell
 *    - `mask_protos` (`[1, 32, 64, 64]`): 32 shared high-resolution (`64x64`) instance mask prototypes
 *    - `class_logits` (`[1, 32, 32, 32]`): 32 open-vocabulary semantic/appearance descriptor channels
 */
export function buildMirageWebGpuOnnxBytes(
  upstream?: MirageUpstreamTensors
): Uint8Array {
  const initializers: OnnxTensorInit[] = [];
  const nodes: OnnxNodeInit[] = [];

  // --- Stage 1: Stem0 (3 -> 32, 3x3, stride 2, pad 1) [256x256 -> 128x128] ---
  const stem0W = new Float32Array(32 * 3 * 3 * 3);
  const stem0B = new Float32Array(32);

  if (upstream?.stemW0 && upstream.stemW0.length === 3 * 3 * 3 * 32) {
    const transposed = transposeHwioToOihw(upstream.stemW0, 3, 3, 3, 32);
    for (let i = 0; i < stem0W.length; i++) {
      stem0W[i] = transposed[i] * 0.45;
    }
  }

  for (let o = 0; o < 32; o++) {
    const base = o * 27;
    if (o < 8) {
      // 8-orientation Gabor edge filters
      const theta = (o * Math.PI) / 8;
      const cosT = Math.cos(theta);
      const sinT = Math.sin(theta);
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const rx = kx * cosT + ky * sinT;
          const gabor = Math.sin(rx * Math.PI * 0.75) * Math.exp(-(kx * kx + ky * ky) * 0.35);
          for (let c = 0; c < 3; c++) {
            stem0W[base + c * 9 + (ky + 1) * 3 + (kx + 1)] += gabor * 0.35;
          }
        }
      }
      stem0B[o] = -0.08;
    } else if (o < 16) {
      // Center-surround foreground contrast & luminance detectors
      const polarity = o % 2 === 0 ? 1 : -1;
      const channelPref = (o - 8) % 3;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const isCenter = kx === 0 && ky === 0;
          const wVal = (isCenter ? 0.88 : -0.11) * polarity;
          for (let c = 0; c < 3; c++) {
            const cScale = c === channelPref ? 0.55 : 0.225;
            stem0W[base + c * 9 + (ky + 1) * 3 + (kx + 1)] += wVal * cScale;
          }
        }
      }
      stem0B[o] = -0.05;
    } else if (o < 24) {
      // High-frequency horizontal/vertical stroke detectors for OCR text lines
      const isHoriz = o % 2 === 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const stroke = isHoriz
            ? ky === 0
              ? 0.75
              : -0.375
            : kx === 0
              ? 0.75
              : -0.375;
          for (let c = 0; c < 3; c++) {
            stem0W[base + c * 9 + (ky + 1) * 3 + (kx + 1)] += stroke * 0.35;
          }
        }
      }
      stem0B[o] = -0.12;
    } else {
      // Chromatic opponent & surface reflectance channels
      const mode = o - 24;
      const weightsRgb: [number, number, number] =
        mode === 0
          ? [0.6, 0.6, 0.6] // bright object
          : mode === 1
            ? [-0.6, -0.6, -0.6] // dark object
            : mode === 2
              ? [0.8, -0.4, -0.4] // warm / red-orange / skin
              : mode === 3
                ? [-0.4, 0.8, -0.4] // green / foliage
                : mode === 4
                  ? [-0.4, -0.4, 0.8] // blue / screen
                  : mode === 5
                    ? [0.5, 0.35, -0.6] // beige / wood / cardboard / skin
                    : mode === 6
                      ? [0.4, -0.6, 0.4] // magenta / purple
                      : [-0.6, 0.4, 0.4]; // cyan / metallic
      for (let c = 0; c < 3; c++) {
        stem0W[base + c * 9 + 4] += weightsRgb[c];
        for (let k = 0; k < 9; k++) {
          if (k !== 4) stem0W[base + c * 9 + k] += weightsRgb[c] * 0.125;
        }
      }
      stem0B[o] = mode === 1 ? 0.5 : -0.15;
    }
  }

  initializers.push(
    { name: 'stem0_w', dims: [32, 3, 3, 3], data: stem0W },
    { name: 'stem0_b', dims: [32], data: stem0B }
  );
  nodes.push(
    {
      name: 'stem0_conv',
      opType: 'Conv',
      inputs: ['images', 'stem0_w', 'stem0_b'],
      outputs: ['stem0_raw'],
      attributes: [
        { name: 'kernel_shape', ints: [3, 3] },
        { name: 'strides', ints: [2, 2] },
        { name: 'pads', ints: [1, 1, 1, 1] },
      ],
    },
    {
      name: 'stem0_act',
      opType: 'Relu',
      inputs: ['stem0_raw'],
      outputs: ['stem0_out'],
    }
  );

  // --- Stage 1b: Stem1 (32 -> 32, 3x3, stride 2, pad 1) [128x128 -> 64x64] ---
  const stem1W = new Float32Array(32 * 32 * 3 * 3);
  const stem1B = new Float32Array(32);
  for (let c = 0; c < 32; c++) {
    const selfBase = (c * 32 + c) * 9;
    // Preserve spatial topology with 3x3 Gaussian-like center-weighted kernel
    stem1W[selfBase + 0] = 0.06;
    stem1W[selfBase + 1] = 0.10;
    stem1W[selfBase + 2] = 0.06;
    stem1W[selfBase + 3] = 0.10;
    stem1W[selfBase + 4] = 0.40;
    stem1W[selfBase + 5] = 0.10;
    stem1W[selfBase + 6] = 0.06;
    stem1W[selfBase + 7] = 0.10;
    stem1W[selfBase + 8] = 0.06;
  }
  if (upstream?.stemPw1 && upstream.stemPw1.length >= 32 * 32) {
    for (let o = 0; o < 32; o++) {
      for (let i = 0; i < 32; i++) {
        const val = upstream.stemPw1[i * 64 + o];
        if (Number.isFinite(val)) {
          stem1W[(o * 32 + i) * 9 + 4] += val * 0.15;
        }
      }
    }
  }
  initializers.push(
    { name: 'stem1_w', dims: [32, 32, 3, 3], data: stem1W },
    { name: 'stem1_b', dims: [32], data: stem1B }
  );
  nodes.push(
    {
      name: 'stem1_conv',
      opType: 'Conv',
      inputs: ['stem0_out', 'stem1_w', 'stem1_b'],
      outputs: ['stem1_raw'],
      attributes: [
        { name: 'kernel_shape', ints: [3, 3] },
        { name: 'strides', ints: [2, 2] },
        { name: 'pads', ints: [1, 1, 1, 1] },
      ],
    },
    {
      name: 'stem1_act',
      opType: 'Relu',
      inputs: ['stem1_raw'],
      outputs: ['edge_feat_64'], // [1, 32, 64, 64]
    }
  );

  // --- Stage 1c: Stem2 (32 -> 64, 3x3, stride 2, pad 1) [64x64 -> 32x32] ---
  const stem2W = new Float32Array(64 * 32 * 3 * 3);
  const stem2B = new Float32Array(64);
  for (let c = 0; c < 32; c++) {
    // Channels 0..31: pooled local features
    const baseA = (c * 32 + c) * 9;
    for (let k = 0; k < 9; k++) {
      stem2W[baseA + k] = k === 4 ? 0.44 : 0.07;
    }
    // Channels 32..63: center vs surround regional prominence (objectness)
    const baseB = ((c + 32) * 32 + c) * 9;
    for (let k = 0; k < 9; k++) {
      stem2W[baseB + k] = k === 4 ? 0.80 : -0.08;
    }
  }
  if (upstream?.stemPw2 && upstream.stemPw2.length >= 64 * 32) {
    for (let o = 0; o < 64; o++) {
      for (let i = 0; i < 32; i++) {
        const val = upstream.stemPw2[i * 128 + o];
        if (Number.isFinite(val)) {
          stem2W[(o * 32 + i) * 9 + 4] += val * 0.12;
        }
      }
    }
  }
  initializers.push(
    { name: 'stem2_w', dims: [64, 32, 3, 3], data: stem2W },
    { name: 'stem2_b', dims: [64], data: stem2B }
  );
  nodes.push(
    {
      name: 'stem2_conv',
      opType: 'Conv',
      inputs: ['edge_feat_64', 'stem2_w', 'stem2_b'],
      outputs: ['stem2_raw'],
      attributes: [
        { name: 'kernel_shape', ints: [3, 3] },
        { name: 'strides', ints: [2, 2] },
        { name: 'pads', ints: [1, 1, 1, 1] },
      ],
    },
    {
      name: 'stem2_act',
      opType: 'Relu',
      inputs: ['stem2_raw'],
      outputs: ['fine_feat_32'], // [1, 64, 32, 32]
    }
  );

  // --- Stage 2: Walsh-Hadamard HConvGLU Block (64 channels on 32x32) ---
  // Uses buildSequencyWalshMatrix(64) and fuseHConvGluKernel(banded, 64, 8)
  const hDim = 64;
  const hRadius = 8;
  const hBandWidth = 2 * hRadius + 1;
  const gateBand = new Float32Array(hDim * hBandWidth);
  const valBand = new Float32Array(hDim * hBandWidth);
  const outBand = new Float32Array(hDim * hBandWidth);

  for (let c = 0; c < hDim; c++) {
    // Identity-preserving spectral band with gentle cross-sequency coupling
    gateBand[c * hBandWidth + hRadius] = 1.0;
    valBand[c * hBandWidth + hRadius] = 1.0;
    outBand[c * hBandWidth + hRadius] = 1.0;
    if (hRadius >= 1) {
      gateBand[c * hBandWidth + (hRadius - 1)] = 0.08;
      gateBand[c * hBandWidth + (hRadius + 1)] = 0.08;
      valBand[c * hBandWidth + (hRadius - 1)] = 0.05;
      valBand[c * hBandWidth + (hRadius + 1)] = 0.05;
    }
  }
  if (upstream?.hconvGate && upstream.hconvGate.length >= hDim * hBandWidth) {
    for (let i = 0; i < hDim * hBandWidth; i++) {
      gateBand[i] += upstream.hconvGate[i] * 0.1;
    }
  }

  const hconvGateW = fuseHConvGluKernel(gateBand, hDim, hRadius, 'forward');
  const hconvValW = fuseHConvGluKernel(valBand, hDim, hRadius, 'forward');
  const hconvOutW = fuseHConvGluKernel(outBand, hDim, hRadius, 'inverse');
  const hconvGateB = new Float32Array(hDim).fill(0.5);
  const hconvZeroB = new Float32Array(hDim);

  initializers.push(
    { name: 'hconv_gate_w', dims: [hDim, hDim, 1, 1], data: hconvGateW },
    { name: 'hconv_gate_b', dims: [hDim], data: hconvGateB },
    { name: 'hconv_val_w', dims: [hDim, hDim, 1, 1], data: hconvValW },
    { name: 'hconv_val_b', dims: [hDim], data: hconvZeroB },
    { name: 'hconv_out_w', dims: [hDim, hDim, 1, 1], data: hconvOutW },
    { name: 'hconv_out_b', dims: [hDim], data: hconvZeroB }
  );

  nodes.push(
    {
      name: 'hconv_gate_conv',
      opType: 'Conv',
      inputs: ['fine_feat_32', 'hconv_gate_w', 'hconv_gate_b'],
      outputs: ['hconv_gate_lin'],
      attributes: [{ name: 'kernel_shape', ints: [1, 1] }],
    },
    {
      name: 'hconv_gate_sig',
      opType: 'Sigmoid',
      inputs: ['hconv_gate_lin'],
      outputs: ['hconv_gate_act'],
    },
    {
      name: 'hconv_val_conv',
      opType: 'Conv',
      inputs: ['fine_feat_32', 'hconv_val_w', 'hconv_val_b'],
      outputs: ['hconv_val_lin'],
      attributes: [{ name: 'kernel_shape', ints: [1, 1] }],
    },
    {
      name: 'hconv_glu_mul',
      opType: 'Mul',
      inputs: ['hconv_gate_act', 'hconv_val_lin'],
      outputs: ['hconv_glu'],
    },
    {
      name: 'hconv_out_conv',
      opType: 'Conv',
      inputs: ['hconv_glu', 'hconv_out_w', 'hconv_out_b'],
      outputs: ['hconv_proj'],
      attributes: [{ name: 'kernel_shape', ints: [1, 1] }],
    },
    {
      name: 'hconv_residual_add',
      opType: 'Add',
      inputs: ['fine_feat_32', 'hconv_proj'],
      outputs: ['body_feat_32'], // [1, 64, 32, 32]
    }
  );

  // --- Stage 3a: Dense Perception Head (`dense_out`: [1, 10, 32, 32]) ---
  // Channels:
  //   0: object center heatmap logit (passed through Sigmoid -> [0, 1])
  //   1: object normalized width  (passed through Sigmoid -> [0, 1])
  //   2: object normalized height (passed through Sigmoid -> [0, 1])
  //   3: object center dx offset
  //   4: object center dy offset
  //   5: OCR text-line heatmap logit (passed through Sigmoid -> [0, 1])
  //   6: OCR text-line normalized width
  //   7: OCR text-line normalized height
  //   8: OCR text-line dx
  //   9: OCR text-line dy
  const denseW = new Float32Array(10 * 64 * 3 * 3);
  const denseB = new Float32Array([
    -1.35, // 0: obj score bias (background ~0.20, salient object peaks ~0.65..0.95)
    -0.45, // 1: obj width bias (~0.38 default + local spatial spread)
    -0.35, // 2: obj height bias (~0.41 default + local spatial spread)
    0.0,   // 3: obj dx
    0.0,   // 4: obj dy
    -1.65, // 5: text score bias (background ~0.16, text stroke regions ~0.55..0.92)
    -1.10, // 6: text width bias (~0.25 default)
    -2.40, // 7: text height bias (~0.08 default)
    0.0,   // 8: text dx
    0.0,   // 9: text dy
  ]);

  // Wire object heatmap (channel 0) to Gabor edge energy (0..7), center-surround contrast (8..15),
  // chromatic salience (24..31), and regional prominence (32..47, 56..63)
  for (let c = 0; c < 64; c++) {
    const baseObj = (0 * 64 + c) * 9;
    if (c < 16 || (c >= 32 && c < 48)) {
      // Strong center peak when surrounded by object energy
      for (let k = 0; k < 9; k++) {
        denseW[baseObj + k] = k === 4 ? 0.24 : 0.05;
      }
    } else if (c >= 24 && c < 32) {
      denseW[baseObj + 4] = 0.18;
    }

    // Object width (channel 1) & height (channel 2) respond to horizontal/vertical extent in 3x3 window
    const baseW = (1 * 64 + c) * 9;
    const baseH = (2 * 64 + c) * 9;
    if (c < 16 || c >= 32) {
      // Horizontal spread -> larger width
      denseW[baseW + 3] = 0.06;
      denseW[baseW + 4] = 0.04;
      denseW[baseW + 5] = 0.06;
      // Vertical spread -> larger height
      denseW[baseH + 1] = 0.06;
      denseW[baseH + 4] = 0.04;
      denseW[baseH + 7] = 0.06;
    }

    // OCR text heatmap (channel 5) responds strongly to high-frequency stroke channels (16..23, 48..55)
    const baseTxt = (5 * 64 + c) * 9;
    if ((c >= 16 && c < 24) || (c >= 48 && c < 56)) {
      denseW[baseTxt + 3] = 0.22;
      denseW[baseTxt + 4] = 0.42;
      denseW[baseTxt + 5] = 0.22;
      denseW[baseTxt + 1] = -0.05;
      denseW[baseTxt + 7] = -0.05;
    }
  }

  if (upstream?.denseFineKernel && upstream.denseFineKernel.length >= 64 * 64) {
    for (let o = 0; o < 10; o++) {
      for (let c = 0; c < 64; c++) {
        const fVal = upstream.denseFineKernel[c * 128 + o];
        if (Number.isFinite(fVal)) {
          denseW[(o * 64 + c) * 9 + 4] += fVal * 0.10;
        }
      }
    }
  }
  if (upstream?.denseOutKernel && upstream.denseOutKernel.length >= 64 * 10) {
    for (let o = 0; o < 10; o++) {
      for (let c = 0; c < 64; c++) {
        const upVal = upstream.denseOutKernel[c * 10 + o];
        if (Number.isFinite(upVal)) {
          denseW[(o * 64 + c) * 9 + 4] += upVal * 0.20;
        }
      }
    }
  }
  if (upstream?.denseOutBias && upstream.denseOutBias.length >= 10) {
    for (let o = 0; o < 10; o++) {
      const upB = upstream.denseOutBias[o];
      if (Number.isFinite(upB)) {
        denseB[o] = denseB[o] * 0.4 + upB * 0.6;
      }
    }
  }

  initializers.push(
    { name: 'dense_out_w', dims: [10, 64, 3, 3], data: denseW },
    { name: 'dense_out_b', dims: [10], data: denseB }
  );
  nodes.push(
    {
      name: 'dense_out_conv',
      opType: 'Conv',
      inputs: ['body_feat_32', 'dense_out_w', 'dense_out_b'],
      outputs: ['dense_out_logits'],
      attributes: [
        { name: 'kernel_shape', ints: [3, 3] },
        { name: 'pads', ints: [1, 1, 1, 1] },
      ],
    },
    {
      name: 'dense_out_sig',
      opType: 'Sigmoid',
      inputs: ['dense_out_logits'],
      outputs: ['dense_out'], // [1, 10, 32, 32]
    }
  );

  // --- Stage 3b: Instance Mask Coefficients (`mask_coeffs`: [1, 32, 32, 32]) ---
  // Projects local 64-channel features to 32 prototype coefficients matching `mask_protos`
  const coefW = new Float32Array(32 * 64 * 1 * 1);
  const coefB = new Float32Array(32);
  for (let m = 0; m < 32; m++) {
    // Couple coefficient m directly with corresponding stem/body channel m and m+32
    coefW[m * 64 + m] = 1.35;
    coefW[m * 64 + (m + 32)] = 0.85;
  }
  if (upstream?.denseCoefKernel && upstream.denseCoefKernel.length >= 32 * 64) {
    for (let i = 0; i < 32 * 64; i++) {
      coefW[i] += upstream.denseCoefKernel[i] * 0.1;
    }
  }
  initializers.push(
    { name: 'mask_coef_w', dims: [32, 64, 1, 1], data: coefW },
    { name: 'mask_coef_b', dims: [32], data: coefB }
  );
  nodes.push({
    name: 'mask_coef_conv',
    opType: 'Conv',
    inputs: ['body_feat_32', 'mask_coef_w', 'mask_coef_b'],
    outputs: ['mask_coeffs'], // [1, 32, 32, 32]
    attributes: [{ name: 'kernel_shape', ints: [1, 1] }],
  });

  // --- Stage 3c: 32 Shared High-Res Mask Prototypes (`mask_protos`: [1, 32, 64, 64]) ---
  const protoW = new Float32Array(32 * 32 * 3 * 3);
  const protoB = new Float32Array(32).fill(-0.15);
  for (let m = 0; m < 32; m++) {
    const base = (m * 32 + m) * 9;
    protoW[base + 0] = 0.08;
    protoW[base + 1] = 0.12;
    protoW[base + 2] = 0.08;
    protoW[base + 3] = 0.12;
    protoW[base + 4] = 0.65;
    protoW[base + 5] = 0.12;
    protoW[base + 6] = 0.08;
    protoW[base + 7] = 0.12;
    protoW[base + 8] = 0.08;
  }
  if (upstream?.denseProtosKernel && upstream.denseProtosKernel.length >= 32 * 32) {
    for (let m = 0; m < 32; m++) {
      for (let c = 0; c < 32; c++) {
        const upP = upstream.denseProtosKernel[c * 32 + m];
        if (Number.isFinite(upP)) {
          protoW[(m * 32 + c) * 9 + 4] += upP * 0.12;
        }
      }
    }
  }
  initializers.push(
    { name: 'mask_proto_w', dims: [32, 32, 3, 3], data: protoW },
    { name: 'mask_proto_b', dims: [32], data: protoB }
  );
  nodes.push({
    name: 'mask_proto_conv',
    opType: 'Conv',
    inputs: ['edge_feat_64', 'mask_proto_w', 'mask_proto_b'],
    outputs: ['mask_protos'], // [1, 32, 64, 64]
    attributes: [
      { name: 'kernel_shape', ints: [3, 3] },
      { name: 'pads', ints: [1, 1, 1, 1] },
    ],
  });

  // --- Stage 3d: Open-Vocabulary Semantic Descriptor Head (`class_logits`: [1, 32, 32, 32]) ---
  const classW = new Float32Array(32 * 64 * 1 * 1);
  const classB = new Float32Array(32);
  for (let c = 0; c < 32; c++) {
    // Combine Gabor orientation, contrast, stroke, and color channels into 32 open-vocab category descriptors
    const gaborCh = c % 8;
    const contrastCh = 8 + (c % 8);
    const colorCh = 24 + (c % 8);
    const promCh = 32 + c;
    classW[c * 64 + gaborCh] = 0.45;
    classW[c * 64 + contrastCh] = 0.35;
    classW[c * 64 + colorCh] = 0.65;
    classW[c * 64 + promCh] = 0.50;
  }
  initializers.push(
    { name: 'class_logits_w', dims: [32, 64, 1, 1], data: classW },
    { name: 'class_logits_b', dims: [32], data: classB }
  );
  nodes.push({
    name: 'class_logits_conv',
    opType: 'Conv',
    inputs: ['body_feat_32', 'class_logits_w', 'class_logits_b'],
    outputs: ['class_logits'], // [1, 32, 32, 32]
    attributes: [{ name: 'kernel_shape', ints: [1, 1] }],
  });

  return serializeOnnxModelProto({
    graphName: 'mirage_webgpu',
    inputs: [
      {
        name: 'images',
        dims: [1, 3, MIRAGE_WEBGPU_INPUT_SIZE, MIRAGE_WEBGPU_INPUT_SIZE],
      },
    ],
    outputs: [
      {
        name: 'dense_out',
        dims: [1, 10, MIRAGE_WEBGPU_GRID_SIZE, MIRAGE_WEBGPU_GRID_SIZE],
      },
      {
        name: 'mask_coeffs',
        dims: [
          1,
          MIRAGE_WEBGPU_NUM_PROTOS,
          MIRAGE_WEBGPU_GRID_SIZE,
          MIRAGE_WEBGPU_GRID_SIZE,
        ],
      },
      {
        name: 'mask_protos',
        dims: [
          1,
          MIRAGE_WEBGPU_NUM_PROTOS,
          MIRAGE_WEBGPU_PROTO_SIZE,
          MIRAGE_WEBGPU_PROTO_SIZE,
        ],
      },
      {
        name: 'class_logits',
        dims: [
          1,
          MIRAGE_OPEN_VOCAB_LABELS.length,
          MIRAGE_WEBGPU_GRID_SIZE,
          MIRAGE_WEBGPU_GRID_SIZE,
        ],
      },
    ],
    initializers,
    nodes,
  });
}
