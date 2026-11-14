/**
 * Pure TypeScript codec for Cactus `.cact` archives used by Mirage:
 * - Parses `.cact` archive headers, 28-float Walsh-Hadamard codebooks, and 806 entry descriptors
 *   (supporting both full `mirage.cact` and sparse `mirage-decoder-tables.cact` containers).
 * - Dequantizes 2-bit and 4-bit CQ (Codebook-Quantized Walsh-Hadamard) matrices and single rows
 *   with exact IEEE-754 float64 butterfly arithmetic and float32 `Math.fround` rounding.
 * - Decodes and encodes SentencePiece RAW vocabularies (`entry 805`) with UTF-8 `<0xNN>` byte
 *   fallback and `<objK>` / `<txtK>` / `<|lang|>` extension tokens.
 * - Evaluates CTC greedy collapse (`entry 804` charset) and 32-bit unsigned multiplicative hash
 *   indices for N-gram Engram lookup tables (`entries 228, 232`).
 *
 * Intentionally free of `node:*` imports so it runs identically in browsers, Web Workers, and Node.js.
 */

export const MIRAGE_CACT_MAGIC = 0x05e12a84;
export const MIRAGE_CACT_ENTRY_COUNT = 806;
export const MIRAGE_CACT_CODEBOOK_LEN = 28;
export const MIRAGE_CACT_HEADER_BYTES = 196;
export const MIRAGE_CACT_MIN_DATA_OFFSET =
  MIRAGE_CACT_HEADER_BYTES +
  4 * MIRAGE_CACT_CODEBOOK_LEN +
  44 * MIRAGE_CACT_ENTRY_COUNT; // 35,772 bytes

export const MIRAGE_DECODER_TABLE_ENTRY_INDICES = [0, 228, 232, 804, 805] as const;
export const MIRAGE_DECODER_TABLES_FILE = 'mirage-decoder-tables.cact';
export const MIRAGE_DECODER_TABLES_EXPECTED_SIZE = 7_350_719;
export const MIRAGE_DECODER_TABLES_EXPECTED_SHA256 =
  '8ab3fa723be57d23116ae974e318c9cd611645c3ca7e9390aa2e1982ec562ef7';

export const MIRAGE_ENGRAM_SLOTS = 18_432;
export const MIRAGE_ENGRAM_SEED = 0x9e3779b9;
export const MIRAGE_ENGRAM_PRIME = 0x01000193;
export const MIRAGE_ENGRAM_ORDERS = [2, 3] as const;
export const MIRAGE_ENGRAM_HEADS = 2;

export const MIRAGE_N_OBJECT_REFS = 64;
export const MIRAGE_N_TEXT_REFS = 32;
export const MIRAGE_EMBEDDING_ROWS = 8299;
export const MIRAGE_EMBEDDING_DIM = 512;
export const MIRAGE_VOCAB_ENTRY = 805;
export const MIRAGE_CHARSET_ENTRY = 804;
export const MIRAGE_CHARSET_LEN = 306;
export const MIRAGE_TOKEN_EMBEDDING_ENTRY = 0;
export const MIRAGE_ENGRAM0_EMBEDDING_ENTRY = 228;
export const MIRAGE_ENGRAM1_EMBEDDING_ENTRY = 232;

export const MIRAGE_SUPPORTED_LANGUAGES = [
  'en',
  'es',
  'fr',
  'pt',
  'de',
  'it',
  'nl',
  'pl',
  'ru',
  'ar',
  'zh',
] as const;

export type MirageSupportedLanguage = (typeof MIRAGE_SUPPORTED_LANGUAGES)[number];

export interface MirageCactEntry {
  index: number;
  dtype: number;
  shape: number[];
  offset: number;
  nbytes: number;
  group: number;
  bits: number;
}

export interface ParsedMirageCactDirectory {
  ropeTheta: number;
  codebook: Float64Array;
  entries: MirageCactEntry[];
}

export interface MiragePlacementRecord {
  tensor: string;
  layer: number | null;
  onnxName: string;
  entry: number;
  entryMode: string;
  entryShape: number[];
  transform: string;
  row?: number;
  offset: number;
  length: number;
  shape: number[];
}

const CODEBOOK_OFFSET_BY_BITS: Readonly<Record<number, number>> = {
  2: 0,
  3: 4,
  4: 12,
};

const CODEBOOK_LEN_BY_BITS: Readonly<Record<number, number>> = {
  2: 4,
  3: 8,
  4: 16,
};

let fp16ToFp32Table: Float32Array | null = null;

/**
 * Exact IEEE-754 binary16 (uint16) to binary32 (float32) conversion lookup table
 * precomputed once for all 65,536 16-bit bit patterns.
 */
export function getFp16ToFp32Table(): Float32Array {
  if (fp16ToFp32Table) {
    return fp16ToFp32Table;
  }
  const table = new Float32Array(65536);
  const u32 = new Uint32Array(table.buffer);
  for (let h = 0; h < 65536; h++) {
    const sign = (h & 0x8000) << 16;
    const exp = (h >> 10) & 0x1f;
    const mant = h & 0x03ff;
    if (exp === 0) {
      if (mant === 0) {
        u32[h] = sign;
      } else {
        let m = mant;
        let e = -14;
        while ((m & 0x0400) === 0) {
          m <<= 1;
          e--;
        }
        m &= 0x03ff;
        u32[h] = sign | ((e + 127) << 23) | (m << 13);
      }
    } else if (exp === 31) {
      u32[h] = sign | 0x7f800000 | (mant << 13);
    } else {
      u32[h] = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
  }
  fp16ToFp32Table = table;
  return table;
}

/**
 * Parses the 196-byte header, 28-float codebook, and 806-entry directory from either a full
 * `mirage.cact` archive or a sparse `mirage-decoder-tables.cact` subset.
 */
export function parseMirageCactDirectory(
  bytes: Uint8Array
): ParsedMirageCactDirectory {
  if (bytes.byteLength < MIRAGE_CACT_MIN_DATA_OFFSET) {
    throw new Error(
      `Archive too small (${bytes.byteLength} B); expected at least ${MIRAGE_CACT_MIN_DATA_OFFSET} B`
    );
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = view.getUint32(0, true);
  const numEntries = view.getUint32(4, true);
  const codebookCount = view.getUint32(8, true);
  if (
    magic !== MIRAGE_CACT_MAGIC ||
    numEntries !== MIRAGE_CACT_ENTRY_COUNT ||
    codebookCount !== MIRAGE_CACT_CODEBOOK_LEN
  ) {
    throw new Error(
      `Unexpected .cact archive header: magic=0x${magic.toString(16)}, entries=${numEntries}, codebook=${codebookCount}`
    );
  }

  const ropeTheta = view.getFloat32(192, true);
  const codebook = new Float64Array(codebookCount);
  for (let i = 0; i < codebookCount; i++) {
    codebook[i] = view.getFloat32(MIRAGE_CACT_HEADER_BYTES + i * 4, true);
  }

  let dirOffset = MIRAGE_CACT_HEADER_BYTES + 4 * codebookCount;
  const entries: MirageCactEntry[] = [];
  for (let i = 0; i < numEntries; i++) {
    const dtype = bytes[dirOffset];
    const ndim = bytes[dirOffset + 1];
    if (dtype < 1 || dtype > 4 || ndim > 4) {
      throw new Error(
        `Invalid archive entry header at index ${i}: dtype=${dtype}, ndim=${ndim}`
      );
    }
    const shape: number[] = [];
    for (let d = 0; d < ndim; d++) {
      shape.push(view.getUint32(dirOffset + 4 + d * 4, true));
    }
    const startLow = view.getUint32(dirOffset + 20, true);
    const startHigh = view.getUint32(dirOffset + 24, true);
    const lenLow = view.getUint32(dirOffset + 28, true);
    const lenHigh = view.getUint32(dirOffset + 32, true);
    const offset = startLow + startHigh * 0x1_0000_0000;
    const nbytes = lenLow + lenHigh * 0x1_0000_0000;
    const group = view.getUint32(dirOffset + 36, true);
    const bits = view.getUint32(dirOffset + 40, true);

    const isOmittedSparse = offset === 0 && nbytes === 0;
    if (
      !isOmittedSparse &&
      (offset < MIRAGE_CACT_MIN_DATA_OFFSET || offset + nbytes > bytes.byteLength)
    ) {
      throw new Error(
        `Archive entry ${i} out of bounds: offset=${offset}, nbytes=${nbytes}, total=${bytes.byteLength}`
      );
    }

    entries.push({
      index: i,
      dtype,
      shape,
      offset,
      nbytes,
      group,
      bits,
    });
    dirOffset += 44;
  }

  return { ropeTheta, codebook, entries };
}

/**
 * Extracts a compact, deterministic sparse `.cact` containing only the entries needed at
 * runtime outside ONNX (`embedding/embedding`, `engrams_0/embedding`, `engrams_1/embedding`,
 * `charset`, `vocab`), reducing browser download from 21.6 MB to 7,350,719 bytes.
 */
export function extractDecoderTablesCactBytes(
  fullCactBytes: Uint8Array
): Uint8Array {
  const { entries } = parseMirageCactDirectory(fullCactBytes);
  const keptSet = new Set<number>(MIRAGE_DECODER_TABLE_ENTRY_INDICES);
  let totalPayload = 0;
  for (const idx of MIRAGE_DECODER_TABLE_ENTRY_INDICES) {
    const entry = entries[idx];
    if (!entry || entry.nbytes === 0) {
      throw new Error(`Required decoder-table entry ${idx} missing in archive`);
    }
    totalPayload += entry.nbytes;
  }

  const out = new Uint8Array(MIRAGE_CACT_MIN_DATA_OFFSET + totalPayload);
  // Copy header + codebook verbatim.
  const codebookEnd = MIRAGE_CACT_HEADER_BYTES + 4 * MIRAGE_CACT_CODEBOOK_LEN;
  out.set(fullCactBytes.subarray(0, codebookEnd), 0);

  const outView = new DataView(out.buffer, out.byteOffset, out.byteLength);
  let payloadPos = MIRAGE_CACT_MIN_DATA_OFFSET;

  for (let i = 0; i < MIRAGE_CACT_ENTRY_COUNT; i++) {
    const srcDirOff = codebookEnd + i * 44;
    out.set(fullCactBytes.subarray(srcDirOff, srcDirOff + 44), srcDirOff);
    if (keptSet.has(i)) {
      const entry = entries[i];
      const lowOff = payloadPos >>> 0;
      const highOff = Math.floor(payloadPos / 0x1_0000_0000) >>> 0;
      const lowLen = entry.nbytes >>> 0;
      const highLen = Math.floor(entry.nbytes / 0x1_0000_0000) >>> 0;
      outView.setUint32(srcDirOff + 20, lowOff, true);
      outView.setUint32(srcDirOff + 24, highOff, true);
      outView.setUint32(srcDirOff + 28, lowLen, true);
      outView.setUint32(srcDirOff + 32, highLen, true);
      out.set(
        fullCactBytes.subarray(entry.offset, entry.offset + entry.nbytes),
        payloadPos
      );
      payloadPos += entry.nbytes;
    } else {
      outView.setUint32(srcDirOff + 20, 0, true);
      outView.setUint32(srcDirOff + 24, 0, true);
      outView.setUint32(srcDirOff + 28, 0, true);
      outView.setUint32(srcDirOff + 32, 0, true);
    }
  }

  if (out.byteLength !== MIRAGE_DECODER_TABLES_EXPECTED_SIZE) {
    throw new Error(
      `Extracted decoder-tables size mismatch: expected ${MIRAGE_DECODER_TABLES_EXPECTED_SIZE}, got ${out.byteLength}`
    );
  }
  return out;
}

interface CqLayout {
  rows: number;
  cols: number;
  groupsPerRow: number;
  paddedCols: number;
  rowIndexBytes: number;
  normsBaseOffset: number;
  perByte: number;
  mask: number;
  cbOffset: number;
}

function validateCqEntry(
  entry: Pick<MirageCactEntry, 'dtype' | 'shape' | 'offset' | 'nbytes' | 'group' | 'bits'>
): CqLayout {
  if (entry.dtype !== 3) {
    throw new Error(`Expected CQ dtype=3, got dtype=${entry.dtype}`);
  }
  if (entry.shape.length !== 2) {
    throw new Error(`Expected 2D CQ entry shape, got [${entry.shape.join(', ')}]`);
  }
  const [rows, cols] = entry.shape;
  const { group, bits } = entry;
  if (group !== 128 || (bits !== 2 && bits !== 4)) {
    throw new Error(`Unsupported CQ quantization parameters: group=${group}, bits=${bits}`);
  }
  const cbOffset = CODEBOOK_OFFSET_BY_BITS[bits];
  if (cbOffset === undefined || CODEBOOK_LEN_BY_BITS[bits] === undefined) {
    throw new Error(`Unsupported CQ bit width: ${bits}`);
  }
  const groupsPerRow = Math.ceil(cols / group);
  const paddedCols = groupsPerRow * group;
  const rowIndexBytes = (paddedCols * bits) / 8;
  const totalIndexBytes = rows * rowIndexBytes;
  const expectedNbytes = totalIndexBytes + rows * groupsPerRow * 2;
  if (entry.nbytes !== expectedNbytes) {
    throw new Error(
      `CQ byte count mismatch: expected ${expectedNbytes}, got ${entry.nbytes}`
    );
  }
  return {
    rows,
    cols,
    groupsPerRow,
    paddedCols,
    rowIndexBytes,
    normsBaseOffset: entry.offset + totalIndexBytes,
    perByte: 8 / bits,
    mask: (1 << bits) - 1,
    cbOffset,
  };
}

function walshGroupInto(
  bytes: Uint8Array,
  packedStart: number,
  normU16: number,
  groupSize: number,
  bits: number,
  perByte: number,
  mask: number,
  cbOffset: number,
  codebook: Float64Array,
  fp16Table: Float32Array,
  block: Float64Array,
  temp: Float64Array,
  validCount: number,
  out: Float32Array,
  outIndexFn: (k: number) => number
): void {
  const norm = fp16Table[normU16];
  const bytesPerGroup = (groupSize * bits) / 8;
  let packedIdx = packedStart;

  if (bits === 2) {
    for (let b = 0; b < bytesPerGroup; b++) {
      const byteVal = bytes[packedIdx++];
      const base = b * 4;
      block[base] = codebook[cbOffset + (byteVal & 3)] * norm;
      block[base + 1] = codebook[cbOffset + ((byteVal >> 2) & 3)] * norm;
      block[base + 2] = codebook[cbOffset + ((byteVal >> 4) & 3)] * norm;
      block[base + 3] = codebook[cbOffset + ((byteVal >> 6) & 3)] * norm;
    }
  } else {
    for (let b = 0; b < bytesPerGroup; b++) {
      const byteVal = bytes[packedIdx++];
      const base = b * perByte;
      for (let j = 0; j < perByte; j++) {
        const code = (byteVal >> (j * bits)) & mask;
        block[base + j] = codebook[cbOffset + code] * norm;
      }
    }
  }

  for (let width = 1; width < groupSize; width <<= 1) {
    const stride = width << 1;
    for (let start = 0; start < groupSize; start += stride) {
      for (let k = 0; k < width; k++) {
        const a = block[start + k];
        const c = block[start + width + k];
        temp[start + k] = a + c;
        temp[start + width + k] = a - c;
      }
    }
    block.set(temp);
  }

  const invSqrtGroup = 1.0 / Math.sqrt(groupSize);
  for (let k = 0; k < validCount; k++) {
    const val = Math.fround(block[k] * invSqrtGroup);
    if (!Number.isFinite(val)) {
      throw new Error('Non-finite CQ dequantization output');
    }
    out[outIndexFn(k)] = val;
  }
}

/**
 * Lazily dequantizes a single row `rowIndex` of a 2D CQ entry directly into `out[outOffset .. outOffset + cols)`.
 */
export function decodeCqRowInto(
  bytes: Uint8Array,
  entry: Pick<MirageCactEntry, 'dtype' | 'shape' | 'offset' | 'nbytes' | 'group' | 'bits'>,
  codebook: Float64Array,
  fp16Table: Float32Array,
  rowIndex: number,
  out: Float32Array,
  outOffset: number = 0
): void {
  const layout = validateCqEntry(entry);
  if (!Number.isInteger(rowIndex) || rowIndex < 0 || rowIndex >= layout.rows) {
    throw new RangeError(`CQ row index ${rowIndex} out of bounds [0, ${layout.rows})`);
  }
  if (outOffset + layout.cols > out.length) {
    throw new RangeError('Output Float32Array too small for decoded CQ row');
  }

  const block = new Float64Array(entry.group);
  const temp = new Float64Array(entry.group);
  const bytesPerGroup = (entry.group * entry.bits) / 8;
  const rowPackedStart = entry.offset + rowIndex * layout.rowIndexBytes;
  const rowNormStart = layout.normsBaseOffset + rowIndex * layout.groupsPerRow * 2;

  for (let g = 0; g < layout.groupsPerRow; g++) {
    const normByteOff = rowNormStart + g * 2;
    const normU16 = bytes[normByteOff] | (bytes[normByteOff + 1] << 8);
    const colBase = g * entry.group;
    const validCount = Math.min(entry.group, layout.cols - colBase);
    walshGroupInto(
      bytes,
      rowPackedStart + g * bytesPerGroup,
      normU16,
      entry.group,
      entry.bits,
      layout.perByte,
      layout.mask,
      layout.cbOffset,
      codebook,
      fp16Table,
      block,
      temp,
      validCount,
      out,
      (k) => outOffset + colBase + k
    );
  }
}

/**
 * Dequantizes a complete 2-bit or 4-bit CQ tensor `[rows, cols]` into Float32Array,
 * optionally transposing to `[cols, rows]`.
 */
export function decodeCqMatrix(
  bytes: Uint8Array,
  entry: Pick<MirageCactEntry, 'dtype' | 'shape' | 'offset' | 'nbytes' | 'group' | 'bits'>,
  codebook: Float64Array,
  fp16Table: Float32Array,
  options: { transpose?: boolean } = {}
): Float32Array {
  const layout = validateCqEntry(entry);
  const transpose = options.transpose === true;
  const out = new Float32Array(layout.rows * layout.cols);
  const block = new Float64Array(entry.group);
  const temp = new Float64Array(entry.group);
  const bytesPerGroup = (entry.group * entry.bits) / 8;

  for (let r = 0; r < layout.rows; r++) {
    const rowPackedStart = entry.offset + r * layout.rowIndexBytes;
    const rowNormStart = layout.normsBaseOffset + r * layout.groupsPerRow * 2;
    for (let g = 0; g < layout.groupsPerRow; g++) {
      const normByteOff = rowNormStart + g * 2;
      const normU16 = bytes[normByteOff] | (bytes[normByteOff + 1] << 8);
      const colBase = g * entry.group;
      const validCount = Math.min(entry.group, layout.cols - colBase);
      walshGroupInto(
        bytes,
        rowPackedStart + g * bytesPerGroup,
        normU16,
        entry.group,
        entry.bits,
        layout.perByte,
        layout.mask,
        layout.cbOffset,
        codebook,
        fp16Table,
        block,
        temp,
        validCount,
        out,
        transpose
          ? (k) => (colBase + k) * layout.rows + r
          : (k) => r * layout.cols + colBase + k
      );
    }
  }
  return out;
}

export function decodeFp16Entry(
  bytes: Uint8Array,
  entry: Pick<MirageCactEntry, 'dtype' | 'offset' | 'nbytes'>,
  fp16Table: Float32Array
): Float32Array {
  if (entry.dtype !== 1 || (entry.nbytes & 1) !== 0) {
    throw new Error(`Invalid fp16 entry: dtype=${entry.dtype}, nbytes=${entry.nbytes}`);
  }
  const count = entry.nbytes >> 1;
  const arr = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const off = entry.offset + (i << 1);
    arr[i] = fp16Table[bytes[off] | (bytes[off + 1] << 8)];
  }
  return arr;
}

export function decodeFp32Entry(
  bytes: Uint8Array,
  entry: Pick<MirageCactEntry, 'dtype' | 'offset' | 'nbytes'>
): Float32Array {
  if (entry.dtype !== 2 || (entry.nbytes & 3) !== 0) {
    throw new Error(`Invalid fp32 entry: dtype=${entry.dtype}, nbytes=${entry.nbytes}`);
  }
  const count = entry.nbytes >> 2;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const arr = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    arr[i] = view.getFloat32(entry.offset + (i << 2), true);
  }
  return arr;
}

export function transpose2D(
  src: Float32Array,
  rows: number,
  cols: number
): Float32Array {
  if (src.length !== rows * cols) {
    throw new Error(`transpose2D length mismatch: ${src.length} vs ${rows}x${cols}`);
  }
  const dst = new Float32Array(src.length);
  for (let r = 0; r < rows; r++) {
    const srcBase = r * cols;
    for (let c = 0; c < cols; c++) {
      dst[c * rows + r] = src[srcBase + c];
    }
  }
  return dst;
}

export function permute3201(src: Float32Array, shape: number[]): Float32Array {
  if (shape.length !== 4) {
    throw new Error(`permute3201 requires 4D shape, got [${shape.join(', ')}]`);
  }
  const [d0, d1, d2, d3] = shape;
  if (src.length !== d0 * d1 * d2 * d3) {
    throw new Error(`permute3201 size mismatch: ${src.length} vs ${shape.join('x')}`);
  }
  const dst = new Float32Array(src.length);
  let dstIdx = 0;
  for (let i3 = 0; i3 < d3; i3++) {
    for (let i2 = 0; i2 < d2; i2++) {
      for (let i0 = 0; i0 < d0; i0++) {
        for (let i1 = 0; i1 < d1; i1++) {
          const srcIdx = ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
          dst[dstIdx++] = src[srcIdx];
        }
      }
    }
  }
  return dst;
}

/**
 * Decodes and transforms an archive entry according to a full-pipeline `MiragePlacementRecord`.
 */
export function resolvePlacementWeight(
  cactBytes: Uint8Array,
  entries: MirageCactEntry[],
  codebook: Float64Array,
  fp16Table: Float32Array,
  placement: MiragePlacementRecord,
  entryCache: Map<number, Float32Array>
): Float32Array {
  const entryIdx = placement.entry;
  const entry = entries[entryIdx];
  if (!entry || entry.nbytes === 0) {
    throw new Error(`Archive entry ${entryIdx} missing for ${placement.tensor}`);
  }

  let base = entryCache.get(entryIdx);
  if (!base) {
    if (placement.entryMode === 'fp16' || placement.entryMode === 'fp16_scalar') {
      base = decodeFp16Entry(cactBytes, entry, fp16Table);
    } else if (placement.entryMode === 'fp32') {
      base = decodeFp32Entry(cactBytes, entry);
    } else if (placement.entryMode === 'cq_id') {
      base = decodeCqMatrix(cactBytes, entry, codebook, fp16Table, { transpose: false });
    } else if (placement.entryMode === 'cq_T') {
      base = decodeCqMatrix(cactBytes, entry, codebook, fp16Table, { transpose: true });
    } else if (placement.entryMode === 'cq_T_per_layer') {
      const raw = decodeCqMatrix(cactBytes, entry, codebook, fp16Table, { transpose: false });
      const [rows, cols] = placement.entryShape;
      if (rows % 8 !== 0) {
        throw new Error(`Expected rows divisible by 8 for cq_T_per_layer, got ${rows}`);
      }
      const perLayer = rows / 8;
      const stacked = new Float32Array(raw.length);
      for (let l = 0; l < 8; l++) {
        const layerIn = raw.subarray(l * perLayer * cols, (l + 1) * perLayer * cols);
        const layerOut = transpose2D(layerIn, perLayer, cols);
        stacked.set(layerOut, l * cols * perLayer);
      }
      base = stacked;
    } else {
      throw new Error(`Unsupported placement entryMode: ${placement.entryMode}`);
    }
    entryCache.set(entryIdx, base);
  }

  const layer = placement.layer;
  const tr = placement.transform;

  // Multi-layer single-archive tensors (`stack/mhc_a_*`, `stack/mhc_b_*`, `stack/mhc_phi_*`)
  // store all 8 layers stacked in one archive entry (total length = 8 * placementElems).
  // Individual per-layer entries (e.g. `cond_u` shape [8, 512]) equal `placementElems` directly.
  const placementElems = placement.shape.reduce((acc, dim) => acc * dim, 1);
  const isStackedSingleArchive =
    layer !== null &&
    tr !== 'layer_row' &&
    base.length === 8 * placementElems;

  const getLayerSlice = (): Float32Array => {
    if (layer === null) {
      throw new Error(`Layer index required for transform ${tr} on ${placement.tensor}`);
    }
    if (!isStackedSingleArchive) {
      return base;
    }
    if (layer < 0 || layer >= 8 || base.length % 8 !== 0) {
      throw new Error(`Invalid stacked slice for ${placement.tensor} layer=${layer}`);
    }
    const perLayerCount = base.length / 8;
    return base.subarray(layer * perLayerCount, (layer + 1) * perLayerCount);
  };

  if (tr === 'id') {
    return base;
  }
  if (tr === 'plus1') {
    const out = new Float32Array(base.length);
    for (let i = 0; i < base.length; i++) {
      out[i] = Math.fround(base[i] + 1.0);
    }
    return out;
  }
  if (tr === 'T') {
    if (placement.entryShape.length !== 2) {
      throw new Error(`T transform expects 2D entryShape on ${placement.tensor}`);
    }
    const [r, c] =
      placement.entryMode === 'cq_T'
        ? [placement.entryShape[1], placement.entryShape[0]]
        : [placement.entryShape[0], placement.entryShape[1]];
    return transpose2D(base, r, c);
  }
  if (tr === 'perm3201') {
    return permute3201(base, placement.entryShape);
  }
  if (tr === 'mix_5_384') {
    return base;
  }
  if (tr === 'mix_row' || tr === 'tap_row') {
    if (placement.row === undefined) {
      throw new Error(`Missing row index for ${tr} on ${placement.tensor}`);
    }
    const cols = placement.shape[placement.shape.length - 1];
    const start = placement.row * cols;
    return base.subarray(start, start + cols);
  }
  if (tr === 'layer_id') {
    return getLayerSlice();
  }
  if (tr === 'layer_plus1') {
    const sl = getLayerSlice();
    const out = new Float32Array(sl.length);
    for (let i = 0; i < sl.length; i++) {
      out[i] = Math.fround(sl[i] + 1.0);
    }
    return out;
  }
  if (tr === 'layer_T') {
    const sl = getLayerSlice();
    const rows =
      placement.entryMode === 'cq_T'
        ? placement.entryShape[1]
        : placement.entryShape[0];
    const cols =
      placement.entryMode === 'cq_T'
        ? placement.entryShape[0]
        : placement.entryShape[1];
    return transpose2D(sl, rows, cols);
  }
  if (tr === 'layer_row') {
    if (placement.row === undefined) {
      throw new Error(`Missing row index for layer_row on ${placement.tensor}`);
    }
    const sl = getLayerSlice();
    const cols = placement.shape[placement.shape.length - 1];
    const start = placement.row * cols;
    return sl.subarray(start, start + cols);
  }

  throw new Error(`Unsupported full-pipeline transform: ${tr}`);
}

const BYTE_TOKEN_RE = /^<0x([0-9A-Fa-f]{2})>$/;
const LANG_TOKEN_RE = /^<\|([a-z]{2})\|>$/;

/**
 * SentencePiece RAW vocabulary with exact byte-fallback decoding, `<objK>`/`<txtK>` refs,
 * `<|lang|>` lookup, and Unigram Viterbi tokenization for `encode(text)`.
 */
export class MirageVocab {
  readonly count: number;
  readonly pad: number;
  readonly eos: number;
  readonly bos: number;
  readonly unk: number;
  readonly addDummyPrefix: boolean;
  readonly byteFallback: boolean;
  readonly pieces: readonly string[];
  readonly types: readonly number[];
  readonly scores: readonly number[];
  readonly size: number;
  readonly langIds: Readonly<Record<string, number>>;

  private readonly pieceToIdMap: Map<string, number>;
  private readonly bytePieceIds: number[];
  private readonly utf8Decoder = new TextDecoder('utf-8', { fatal: false });
  private readonly utf8Encoder = new TextEncoder();
  private maxPieceCodeUnits = 0;

  constructor(params: {
    count: number;
    pad: number;
    eos: number;
    bos: number;
    unk: number;
    addDummyPrefix: boolean;
    byteFallback: boolean;
    pieces: string[];
    types: number[];
    scores: number[];
    size: number;
    langIds: Record<string, number>;
  }) {
    this.count = params.count;
    this.pad = params.pad;
    this.eos = params.eos;
    this.bos = params.bos;
    this.unk = params.unk;
    this.addDummyPrefix = params.addDummyPrefix;
    this.byteFallback = params.byteFallback;
    this.pieces = params.pieces;
    this.types = params.types;
    this.scores = params.scores;
    this.size = params.size;
    this.langIds = params.langIds;

    this.pieceToIdMap = new Map<string, number>();
    this.bytePieceIds = new Array<number>(256).fill(-1);
    for (let i = 0; i < params.count; i++) {
      const p = params.pieces[i];
      if (!this.pieceToIdMap.has(p)) {
        this.pieceToIdMap.set(p, i);
      }
      if (params.types[i] === 0 || params.types[i] === 3) {
        if (p.length > this.maxPieceCodeUnits) {
          this.maxPieceCodeUnits = p.length;
        }
      } else if (params.types[i] === 4) {
        const m = BYTE_TOKEN_RE.exec(p);
        if (m) {
          this.bytePieceIds[parseInt(m[1], 16)] = i;
        }
      }
    }
  }

  get refBase(): number {
    return this.count;
  }

  idOf(piece: string): number | undefined {
    return this.pieceToIdMap.get(piece);
  }

  refId(kind: 'object' | 'text', k: number): number {
    return this.refBase + (kind === 'object' ? k : MIRAGE_N_OBJECT_REFS + k);
  }

  piece(idx: number): string {
    if (idx >= this.count) {
      const k = idx - this.refBase;
      return k < MIRAGE_N_OBJECT_REFS
        ? `<obj${k}>`
        : `<txt${k - MIRAGE_N_OBJECT_REFS}>`;
    }
    return this.pieces[idx] ?? '';
  }

  decode(ids: ArrayLike<number>, showSpecial: boolean = true): string {
    const out: string[] = [];
    const byteBuf: number[] = [];

    const flushBytes = (): void => {
      if (byteBuf.length > 0) {
        out.push(this.utf8Decoder.decode(new Uint8Array(byteBuf)));
        byteBuf.length = 0;
      }
    };

    for (let n = 0; n < ids.length; n++) {
      const idx = ids[n];
      if (idx >= this.count) {
        flushBytes();
        out.push(this.piece(idx));
        continue;
      }
      const typ = this.types[idx];
      const pc = this.pieces[idx];
      if (typ === 4) {
        const m = BYTE_TOKEN_RE.exec(pc);
        if (m) {
          byteBuf.push(parseInt(m[1], 16));
          continue;
        }
      }
      flushBytes();
      if (typ === 1 || typ === 2 || typ === 3) {
        if (showSpecial) {
          out.push(pc);
        }
      } else {
        out.push(pc.replace(/\u2581/g, ' '));
      }
    }
    flushBytes();
    return out.join('');
  }

  /**
   * Encodes UTF-8 text into SentencePiece token IDs using Unigram Viterbi scoring,
   * preserving `USER_DEFINED` (type=3) special tokens when present literally in `text`.
   */
  encode(text: string): number[] {
    if (text.length === 0) {
      return [];
    }
    // Split on user-defined tokens (`<think>`, `<|en|>`, `<schema>`, etc.) when matched verbatim.
    const normalized = text.replace(/ /g, '\u2581');
    const finalStr =
      this.addDummyPrefix && !normalized.startsWith('\u2581')
        ? `\u2581${normalized}`
        : normalized;

    const len = finalStr.length;
    const bestScore = new Float64Array(len + 1).fill(-Infinity);
    const prevPos = new Int32Array(len + 1).fill(-1);
    const chosenIds: number[][] = Array.from({ length: len + 1 }, () => []);
    bestScore[0] = 0;

    for (let i = 0; i < len; ) {
      const cp = finalStr.codePointAt(i)!;
      const cpLen = cp > 0xffff ? 2 : 1;
      if (Number.isFinite(bestScore[i])) {
        const maxEnd = Math.min(len, i + Math.max(this.maxPieceCodeUnits, cpLen));
        let matchedAny = false;
        for (let end = i + cpLen; end <= maxEnd; ) {
          const slice = finalStr.slice(i, end);
          const id = this.pieceToIdMap.get(slice);
          if (id !== undefined) {
            const typ = this.types[id];
            if (typ === 0 || typ === 3) {
              const cand = bestScore[i] + (typ === 3 ? 0 : this.scores[id]);
              if (cand > bestScore[end]) {
                bestScore[end] = cand;
                prevPos[end] = i;
                chosenIds[end] = [id];
              }
              matchedAny = true;
            }
          }
          if (end >= len) {
            break;
          }
          const nextCp = finalStr.codePointAt(end)!;
          end += nextCp > 0xffff ? 2 : 1;
        }
        if (!matchedAny) {
          const end = i + cpLen;
          const penalty = bestScore[i] - 100.0;
          if (penalty > bestScore[end]) {
            bestScore[end] = penalty;
            prevPos[end] = i;
            if (this.byteFallback) {
              const ch = finalStr.slice(i, end);
              const bytes = this.utf8Encoder.encode(ch === '\u2581' ? ' ' : ch);
              const ids: number[] = [];
              let ok = true;
              for (let b = 0; b < bytes.length; b++) {
                const bid = this.bytePieceIds[bytes[b]];
                if (bid < 0) {
                  ok = false;
                  break;
                }
                ids.push(bid);
              }
              chosenIds[end] = ok ? ids : [this.unk];
            } else {
              chosenIds[end] = [this.unk];
            }
          }
        }
      }
      i += cpLen;
    }

    const segments: number[][] = [];
    let cur = len;
    while (cur > 0) {
      const p = prevPos[cur];
      if (p < 0) {
        throw new Error(`SentencePiece encode failed at index ${cur}`);
      }
      segments.push(chosenIds[cur]);
      cur = p;
    }
    const result: number[] = [];
    for (let s = segments.length - 1; s >= 0; s--) {
      for (const id of segments[s]) {
        result.push(id);
      }
    }
    return result;
  }
}

export function parseVocabRaw(
  raw: Uint8Array,
  embeddingRows: number = MIRAGE_EMBEDDING_ROWS
): MirageVocab {
  if (raw.byteLength < 24) {
    throw new Error(`SentencePiece RAW buffer too small: ${raw.byteLength} B`);
  }
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const count = view.getUint32(0, true);
  const pad = view.getUint32(4, true);
  const eos = view.getUint32(8, true);
  const bos = view.getUint32(12, true);
  const unk = view.getUint32(16, true);
  const addDummyPrefix = raw[20] !== 0;
  const byteFallback = raw[21] !== 0;

  let off = 24;
  const utf8 = new TextDecoder('utf-8', { fatal: false });
  const pieces: string[] = [];
  const types: number[] = [];
  const scores: number[] = [];
  const langIds: Record<string, number> = {};

  for (let i = 0; i < count; i++) {
    if (off + 7 > raw.byteLength) {
      throw new Error(`Truncated vocab entry header at index ${i}`);
    }
    const score = view.getFloat32(off, true);
    const typ = raw[off + 4];
    const length = view.getUint16(off + 5, true);
    off += 7;
    if (off + length > raw.byteLength) {
      throw new Error(`Truncated vocab piece bytes at index ${i}`);
    }
    const piece = utf8.decode(raw.subarray(off, off + length));
    off += length;
    pieces.push(piece);
    types.push(typ);
    scores.push(score);
    const lm = LANG_TOKEN_RE.exec(piece);
    if (lm) {
      langIds[lm[1]] = i;
    }
  }

  return new MirageVocab({
    count,
    pad,
    eos,
    bos,
    unk,
    addDummyPrefix,
    byteFallback,
    pieces,
    types,
    scores,
    size: embeddingRows,
    langIds,
  });
}

export function parseCharsetRaw(
  bytes: Uint8Array,
  entry: Pick<MirageCactEntry, 'dtype' | 'shape' | 'offset' | 'nbytes'>
): string {
  const arr = decodeFp32Entry(bytes, entry);
  let out = '';
  for (let i = 0; i < arr.length; i++) {
    out += String.fromCodePoint(Math.round(arr[i]));
  }
  return out;
}

/**
 * Greedy CTC collapse: skips blank (0) and repeated identical consecutive frame labels.
 */
export function ctcCollapse(labels: ArrayLike<number>, charset: string): string {
  const out: string[] = [];
  let prev = 0;
  for (let i = 0; i < labels.length; i++) {
    const lab = labels[i];
    if (lab !== 0 && lab !== prev) {
      const idx = lab - 1;
      if (idx >= 0 && idx < charset.length) {
        out.push(charset[idx]);
      }
    }
    prev = lab;
  }
  return out.join('');
}

/**
 * Exact 32-bit unsigned FNV/multiplicative hash for Engram table indexing.
 */
export function engramIndex(
  tokens: ArrayLike<number>,
  t: number,
  orderIdx: number,
  order: number,
  head: number
): number {
  let acc = Math.imul(
    MIRAGE_ENGRAM_SEED,
    orderIdx * MIRAGE_ENGRAM_HEADS + head + 1
  ) >>> 0;
  for (let j = 0; j < order; j++) {
    const tok = t - j >= 0 ? tokens[t - j] >>> 0 : 0;
    acc = Math.imul((acc ^ tok) >>> 0, MIRAGE_ENGRAM_PRIME) >>> 0;
  }
  acc = (acc ^ (acc >>> 15)) >>> 0;
  return acc % MIRAGE_ENGRAM_SLOTS;
}

/**
 * Lazy runtime accessor over either `mirage.cact` or compact `mirage-decoder-tables.cact`
 * that serves vocab, CTC charset, token embeddings, and Engram vectors without bulk-unpacking
 * all 73,728-row Engram tables into memory.
 */
export class MirageDecoderTableReader {
  readonly codebook: Float64Array;
  readonly entries: MirageCactEntry[];
  readonly fp16Table: Float32Array;
  readonly vocab: MirageVocab;
  readonly charset: string;

  private readonly bytes: Uint8Array;
  private readonly tokenEntry: MirageCactEntry;
  private readonly eng0Entry: MirageCactEntry;
  private readonly eng1Entry: MirageCactEntry;
  private readonly tokenRowCache = new Map<number, Float32Array>();

  constructor(archiveBytes: Uint8Array) {
    this.bytes = archiveBytes;
    const { codebook, entries } = parseMirageCactDirectory(archiveBytes);
    this.codebook = codebook;
    this.entries = entries;
    this.fp16Table = getFp16ToFp32Table();

    const vocabEntry = entries[MIRAGE_VOCAB_ENTRY];
    const charsetEntry = entries[MIRAGE_CHARSET_ENTRY];
    const tokenEntry = entries[MIRAGE_TOKEN_EMBEDDING_ENTRY];
    const eng0Entry = entries[MIRAGE_ENGRAM0_EMBEDDING_ENTRY];
    const eng1Entry = entries[MIRAGE_ENGRAM1_EMBEDDING_ENTRY];

    for (const req of [vocabEntry, charsetEntry, tokenEntry, eng0Entry, eng1Entry]) {
      if (!req || req.nbytes === 0) {
        throw new Error(
          `Archive missing required decoder-table entry ${req?.index ?? 'unknown'}`
        );
      }
    }

    this.tokenEntry = tokenEntry;
    this.eng0Entry = eng0Entry;
    this.eng1Entry = eng1Entry;
    this.vocab = parseVocabRaw(
      archiveBytes.subarray(vocabEntry.offset, vocabEntry.offset + vocabEntry.nbytes),
      tokenEntry.shape[0]
    );
    this.charset = parseCharsetRaw(archiveBytes, charsetEntry);
  }

  /**
   * Decodes one `[512]` token embedding row lazily and caches repeated token IDs (e.g. structural tokens).
   */
  readTokenEmbedding(tokenId: number, out?: Float32Array): Float32Array {
    let cached = this.tokenRowCache.get(tokenId);
    if (!cached) {
      cached = new Float32Array(MIRAGE_EMBEDDING_DIM);
      decodeCqRowInto(
        this.bytes,
        this.tokenEntry,
        this.codebook,
        this.fp16Table,
        tokenId,
        cached,
        0
      );
      if (this.tokenRowCache.size < 512) {
        this.tokenRowCache.set(tokenId, cached);
      }
    }
    if (out) {
      out.set(cached);
      return out;
    }
    return new Float32Array(cached);
  }

  /**
   * Computes `[512]` Engram lookup concatenation (`orders=(2,3), heads=2`, 128 dims per slice)
   * at step `t` for site `0` (`engrams_0`, layer 3) or `1` (`engrams_1`, layer 7).
   */
  readEngramVector(
    site: 0 | 1,
    tokens: ArrayLike<number>,
    t: number,
    out: Float32Array = new Float32Array(MIRAGE_EMBEDDING_DIM)
  ): Float32Array {
    const entry = site === 0 ? this.eng0Entry : this.eng1Entry;
    const partDim = 128;
    out.fill(0);
    for (let oi = 0; oi < MIRAGE_ENGRAM_ORDERS.length; oi++) {
      const order = MIRAGE_ENGRAM_ORDERS[oi];
      if (t - (order - 1) < 0) {
        continue;
      }
      for (let h = 0; h < MIRAGE_ENGRAM_HEADS; h++) {
        const idx = engramIndex(tokens, t, oi, order, h);
        const sliceIdx = oi * MIRAGE_ENGRAM_HEADS + h;
        const rowIndex = sliceIdx * MIRAGE_ENGRAM_SLOTS + idx;
        decodeCqRowInto(
          this.bytes,
          entry,
          this.codebook,
          this.fp16Table,
          rowIndex,
          out,
          sliceIdx * partDim
        );
      }
    }
    return out;
  }
}
