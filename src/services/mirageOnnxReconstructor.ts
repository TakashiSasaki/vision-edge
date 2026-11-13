import { createHash, randomUUID } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync, gzipSync } from 'node:zlib';
import {
  MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256,
  MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE,
} from './mirageExperimentalOnnxService';

export const MIRAGE_CACT_EXPECTED_SIZE = 21_611_043;
export const MIRAGE_CACT_EXPECTED_SHA256 =
  '4249a0f7515272ff76b9414bec17d7e7daf84b15b66c7fb4f17fce1e66f9f237';

export function getMirageCactExpectedPin(): { size: number; sha256: string } {
  return {
    size: MIRAGE_CACT_EXPECTED_SIZE,
    sha256: MIRAGE_CACT_EXPECTED_SHA256,
  };
}

function resolveMiragePackageRoot(): string {
  const meta = import.meta as unknown as { dirname?: string; url?: string };
  if (typeof meta.dirname === 'string' && meta.dirname.length > 0) {
    return path.resolve(meta.dirname, '../..');
  }
  if (typeof meta.url === 'string' && meta.url.startsWith('file://')) {
    return path.resolve(path.dirname(fileURLToPath(meta.url)), '../..');
  }
  if (typeof meta.url === 'string' && meta.url.startsWith('/')) {
    return path.resolve(path.dirname(meta.url), '../..');
  }
  return process.cwd();
}

export const MIRAGE_PACKAGE_ROOT = resolveMiragePackageRoot();
export const MIRAGE_RECONSTRUCTION_REL_DIR = path.join('src', 'assets', 'mirage-reconstruction');

export const MIRAGE_RECONSTRUCTION_ASSETS_PIN = {
  mapping: {
    file: 'mirage-encoder-archive-mapping.json',
    size: 95_347,
    sha256: 'b520189ac608d27279243494c6ac3cbde40edf1519c6d5a4d5c4d9c2f0a12892',
  },
  skeletonGz: {
    file: 'cact_onnx_skeleton.bin.gz',
    size: 124_958,
    sha256: '0e9f92e01fe5142805f7168063cad3ec183c88b58d5957ccb703ba0a442393eb',
  },
  placements: {
    file: 'cact_onnx_placements.json',
    size: 68_826,
    sha256: '04f56475718c14ac0120fa54c85ef0dca4ebc70eb6f2a812e2bf7232fedfc95c',
  },
} as const;

export interface MirageReconstructionAssetPaths {
  assetsDir: string;
  mappingPath: string;
  skeletonGzPath: string;
  placementsPath: string;
  presetsDir: string;
}

export function resolveMirageReconstructionAssetPaths(
  rootDir: string = MIRAGE_PACKAGE_ROOT
): MirageReconstructionAssetPaths {
  const assetsDir = path.resolve(rootDir, MIRAGE_RECONSTRUCTION_REL_DIR);
  return {
    assetsDir,
    mappingPath: path.join(assetsDir, MIRAGE_RECONSTRUCTION_ASSETS_PIN.mapping.file),
    skeletonGzPath: path.join(assetsDir, MIRAGE_RECONSTRUCTION_ASSETS_PIN.skeletonGz.file),
    placementsPath: path.join(assetsDir, MIRAGE_RECONSTRUCTION_ASSETS_PIN.placements.file),
    presetsDir: path.join(assetsDir, 'presets'),
  };
}

export function hasMirageReconstructionAssets(
  rootDir: string = MIRAGE_PACKAGE_ROOT
): boolean {
  const paths = resolveMirageReconstructionAssetPaths(rootDir);
  return (
    existsSync(paths.mappingPath) &&
    existsSync(paths.skeletonGzPath) &&
    existsSync(paths.placementsPath)
  );
}

function readVerifiedAssetFile(
  filePath: string,
  expected: { file: string; size: number; sha256: string }
): Buffer {
  const bytes = readFileSync(filePath);
  if (bytes.byteLength !== expected.size) {
    throw new Error(
      `Reconstruction asset ${expected.file} size mismatch: expected ${expected.size}, got ${bytes.byteLength}`
    );
  }
  const actualSha256 = createHash('sha256').update(bytes).digest('hex');
  if (actualSha256 !== expected.sha256) {
    throw new Error(
      `Reconstruction asset ${expected.file} SHA-256 mismatch: expected ${expected.sha256}, got ${actualSha256}`
    );
  }
  return bytes;
}

interface ArchiveMappingEntry {
  index: number;
  dtype: number;
  shape: number[];
  offset: number;
  bytes: number;
  group: number;
  bits: number;
}

interface ArchiveMappingRow {
  name: string;
  entry: ArchiveMappingEntry;
}

interface OnnxPlacement {
  name: string;
  mode: string;
  offset: number;
  length: number;
  shape: number[];
}

/**
 * Exact IEEE-754 binary16 (uint16) to binary32 (float32) conversion table,
 * precomputed once for all 65,536 16-bit bit patterns.
 */
let fp16ToFp32Table: Float32Array | null = null;

function getFp16ToFp32Table(): Float32Array {
  if (fp16ToFp32Table) return fp16ToFp32Table;
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
 * Decodes a Cactus CQ (2-bit Walsh-Hadamard codebook quantized) matrix and
 * returns its transposed Float32Array `decode_cq(...).T.copy()`, matching
 * `scripts/mirage/archive_encoder_weights.py` bit-for-bit.
 */
function decodeCqTransposed(
  cactBytes: Buffer,
  byteOffset: number,
  byteLength: number,
  entryShape: number[],
  group: number,
  bits: number,
  codebook: Float64Array,
  fp16Table: Float32Array
): Float32Array {
  if (entryShape.length !== 2) {
    throw new Error(`Expected 2D CQ entry shape, got [${entryShape.join(', ')}]`);
  }
  const rows = entryShape[0];
  const cols = entryShape[1];
  if (group !== 128 || bits !== 2 || cols % group !== 0) {
    throw new Error(`Unsupported CQ format: group=${group}, bits=${bits}, cols=${cols}`);
  }
  const count = rows * cols;
  const indexBytes = (count * bits) / 8;
  const normCount = count / group;
  if (byteLength !== indexBytes + 2 * normCount) {
    throw new Error(
      `CQ byte count mismatch: expected ${indexBytes + 2 * normCount}, got ${byteLength}`
    );
  }

  const invSqrt128 = 1.0 / Math.sqrt(128);
  const groupsPerRow = cols / group;
  // Output is transposed: shape is [cols, rows]
  const outTransposed = new Float32Array(cols * rows);
  const block = new Float64Array(128);
  const temp = new Float64Array(128);

  let packedIdx = byteOffset;
  let normByteIdx = byteOffset + indexBytes;

  for (let r = 0; r < rows; r++) {
    for (let g = 0; g < groupsPerRow; g++) {
      const u16Norm =
        cactBytes[normByteIdx] | (cactBytes[normByteIdx + 1] << 8);
      normByteIdx += 2;
      const norm = fp16Table[u16Norm];

      // Unpack 128 2-bit codes from 32 bytes
      for (let b = 0; b < 32; b++) {
        const byteVal = cactBytes[packedIdx++];
        const base = b * 4;
        block[base] = codebook[byteVal & 3] * norm;
        block[base + 1] = codebook[(byteVal >> 2) & 3] * norm;
        block[base + 2] = codebook[(byteVal >> 4) & 3] * norm;
        block[base + 3] = codebook[(byteVal >> 6) & 3] * norm;
      }

      // In-place 128-point natural Walsh-Hadamard transform in Float64
      for (let width = 1; width < 128; width <<= 1) {
        const stride = width << 1;
        for (let start = 0; start < 128; start += stride) {
          for (let k = 0; k < width; k++) {
            const a = block[start + k];
            const c = block[start + width + k];
            temp[start + k] = a + c;
            temp[start + width + k] = a - c;
          }
        }
        block.set(temp);
      }

      const colBase = g * 128;
      for (let k = 0; k < 128; k++) {
        const val = Math.fround(block[k] * invSqrt128);
        // Write transposed at (colBase + k, r) -> index (colBase + k) * rows + r
        outTransposed[(colBase + k) * rows + r] = val;
      }
    }
  }

  return outTransposed;
}

function permute3201(src: Float32Array, shape: number[]): Float32Array {
  const [d0, d1, d2, d3] = shape;
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
 * Reconstructs the verified `cact-dequantized-reference-order.onnx` (`34,281,503` bytes,
 * SHA-256 `52d52d2918160404cd806702a270736819712025cf3c6aa8e907a06bc3cbdd07`) in pure Node.js
 * directly from the verified `mirage.cact` archive and the packaged runtime reconstruction assets.
 */
export function reconstructVerifiedMirageOnnxBytes(
  cactBytes: Buffer,
  rootDir: string = MIRAGE_PACKAGE_ROOT
): Buffer {
  if (cactBytes.byteLength !== MIRAGE_CACT_EXPECTED_SIZE) {
    throw new Error(
      `mirage.cact byte length mismatch: expected ${MIRAGE_CACT_EXPECTED_SIZE}, got ${cactBytes.byteLength}`
    );
  }
  const cactSha256 = createHash('sha256').update(cactBytes).digest('hex');
  if (cactSha256 !== MIRAGE_CACT_EXPECTED_SHA256) {
    throw new Error(
      `mirage.cact SHA-256 mismatch: expected ${MIRAGE_CACT_EXPECTED_SHA256}, got ${cactSha256}`
    );
  }

  const magic = cactBytes.readUInt32LE(0);
  const numEntries = cactBytes.readUInt32LE(4);
  const codebookCount = cactBytes.readUInt32LE(8);
  if (magic !== 0x05e12a84 || numEntries !== 806 || codebookCount !== 28) {
    throw new Error('Unexpected mirage.cact archive header');
  }

  const codebook = new Float64Array(28);
  for (let i = 0; i < 28; i++) {
    codebook[i] = cactBytes.readFloatLE(196 + i * 4);
  }

  const { mappingPath, skeletonGzPath, placementsPath } =
    resolveMirageReconstructionAssetPaths(rootDir);

  const mappingBytes = readVerifiedAssetFile(
    mappingPath,
    MIRAGE_RECONSTRUCTION_ASSETS_PIN.mapping
  );
  const placementsBytes = readVerifiedAssetFile(
    placementsPath,
    MIRAGE_RECONSTRUCTION_ASSETS_PIN.placements
  );
  const skeletonGzBytes = readVerifiedAssetFile(
    skeletonGzPath,
    MIRAGE_RECONSTRUCTION_ASSETS_PIN.skeletonGz
  );

  const mappingRows = JSON.parse(
    mappingBytes.toString('utf8')
  ) as ArchiveMappingRow[];
  const placements = JSON.parse(
    placementsBytes.toString('utf8')
  ) as OnnxPlacement[];
  const onnxBuffer = gunzipSync(skeletonGzBytes);

  if (onnxBuffer.byteLength !== MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE) {
    throw new Error(
      `Skeleton size mismatch: expected ${MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE}, got ${onnxBuffer.byteLength}`
    );
  }

  const fp16Table = getFp16ToFp32Table();
  const decodedWeights = new Map<string, Float32Array>();

  for (const row of mappingRows) {
    const { name, entry } = row;
    const { offset, bytes, dtype, shape, group, bits } = entry;
    if (dtype === 3) {
      decodedWeights.set(
        name,
        decodeCqTransposed(
          cactBytes,
          offset,
          bytes,
          shape,
          group,
          bits,
          codebook,
          fp16Table
        )
      );
    } else if (dtype === 1) {
      const count = bytes >> 1;
      const arr = new Float32Array(count);
      for (let i = 0; i < count; i++) {
        const u16 =
          cactBytes[offset + (i << 1)] |
          (cactBytes[offset + (i << 1) + 1] << 8);
        arr[i] = fp16Table[u16];
      }
      decodedWeights.set(name, arr);
    } else if (dtype === 2) {
      const count = bytes >> 2;
      const arr = new Float32Array(count);
      for (let i = 0; i < count; i++) {
        arr[i] = cactBytes.readFloatLE(offset + (i << 2));
      }
      decodedWeights.set(name, arr);
    } else {
      throw new Error(`Unsupported archive dtype ${dtype} for ${name}`);
    }
  }

  for (const placement of placements) {
    const baseWeight = decodedWeights.get(placement.name);
    if (!baseWeight) {
      throw new Error(`Missing decoded weight for placement: ${placement.name}`);
    }
    let transformed: Float32Array;
    if (placement.mode === 'id') {
      transformed = baseWeight;
    } else if (placement.mode === 'plus1') {
      transformed = new Float32Array(baseWeight.length);
      for (let i = 0; i < baseWeight.length; i++) {
        transformed[i] = Math.fround(baseWeight[i] + 1.0);
      }
    } else if (placement.mode === 'perm3201') {
      transformed = permute3201(baseWeight, placement.shape);
    } else if (placement.mode.startsWith('row_')) {
      const rowIdx = Number(placement.mode.slice(4));
      const cols = placement.shape[1];
      transformed = baseWeight.subarray(rowIdx * cols, (rowIdx + 1) * cols);
    } else {
      throw new Error(`Unsupported placement mode: ${placement.mode}`);
    }

    const srcBytes = Buffer.from(
      transformed.buffer,
      transformed.byteOffset,
      transformed.byteLength
    );
    if (srcBytes.byteLength !== placement.length) {
      throw new Error(
        `Placement byte length mismatch for ${placement.name}: expected ${placement.length}, got ${srcBytes.byteLength}`
      );
    }
    srcBytes.copy(onnxBuffer, placement.offset);
  }

  if (onnxBuffer.byteLength !== MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE) {
    throw new Error(
      `Reconstructed ONNX size mismatch: expected ${MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE}, got ${onnxBuffer.byteLength}`
    );
  }
  const actualSha256 = createHash('sha256').update(onnxBuffer).digest('hex');
  if (actualSha256 !== MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256) {
    throw new Error(
      `Reconstructed ONNX SHA-256 mismatch: expected ${MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256}, got ${actualSha256}`
    );
  }

  return onnxBuffer;
}

export function ensureReconstructedMirageOnnxFile(
  targetPath: string,
  cactBytes: Buffer,
  rootDir: string = MIRAGE_PACKAGE_ROOT
): Buffer {
  const onnxBuffer = reconstructVerifiedMirageOnnxBytes(cactBytes, rootDir);
  mkdirSync(path.dirname(targetPath), { recursive: true });
  const tempPath = `${targetPath}.part-${randomUUID()}`;
  try {
    writeFileSync(tempPath, onnxBuffer);
    renameSync(tempPath, targetPath);
  } catch (err) {
    try {
      rmSync(tempPath, { force: true });
    } catch {
      // Ignore cleanup error
    }
    throw err;
  }
  const tempGz = `${targetPath}.gz.part-${randomUUID()}`;
  try {
    const gzBytes = gzipSync(onnxBuffer, { level: 9 });
    writeFileSync(tempGz, gzBytes);
    renameSync(tempGz, `${targetPath}.gz`);
  } catch {
    try {
      rmSync(tempGz, { force: true });
    } catch {
      // Non-critical compression fallback
    }
  }
  return onnxBuffer;
}

