import { createHash, randomUUID } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync, gzipSync } from 'node:zlib';
import {
  decodeCqMatrix,
  decodeFp16Entry,
  decodeFp32Entry,
  extractDecoderTablesCactBytes,
  getFp16ToFp32Table,
  MIRAGE_DECODER_TABLES_EXPECTED_SHA256,
  MIRAGE_DECODER_TABLES_EXPECTED_SIZE,
  MIRAGE_DECODER_TABLES_FILE,
  parseMirageCactDirectory,
  permute3201,
  resolvePlacementWeight,
  type MiragePlacementRecord,
} from './mirageCactCodec.ts';
import {
  MIRAGE_CROSS_KV_ONNX_EXPECTED_SHA256,
  MIRAGE_CROSS_KV_ONNX_EXPECTED_SIZE,
  MIRAGE_CROSS_KV_ONNX_FILE,
  MIRAGE_DECODER_STEP_ONNX_EXPECTED_SHA256,
  MIRAGE_DECODER_STEP_ONNX_EXPECTED_SIZE,
  MIRAGE_DECODER_STEP_ONNX_FILE,
  MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SHA256,
  MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE,
  MIRAGE_READER_ONNX_EXPECTED_SHA256,
  MIRAGE_READER_ONNX_EXPECTED_SIZE,
  MIRAGE_READER_ONNX_FILE,
  type MirageFullPipelineComponentKey,
} from './modelAssetResolver.ts';

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

export interface MirageFullOnnxComponentPin {
  key: Exclude<MirageFullPipelineComponentKey, 'decoderTables'>;
  onnxFile: string;
  expectedSize: number;
  expectedSha256: string;
  skeletonFile: string;
  skeletonSize: number;
  skeletonSha256: string;
  placementsFile: string;
  placementsSize: number;
  placementsSha256: string;
}

export interface MirageDecoderTablesPin {
  key: 'decoderTables';
  file: string;
  expectedSize: number;
  expectedSha256: string;
}

export const MIRAGE_FULL_PIPELINE_PIN: {
  reader: MirageFullOnnxComponentPin;
  crossKv: MirageFullOnnxComponentPin;
  decoderStep: MirageFullOnnxComponentPin;
  decoderTables: MirageDecoderTablesPin;
} = {
  reader: {
    key: 'reader',
    onnxFile: MIRAGE_READER_ONNX_FILE,
    expectedSize: MIRAGE_READER_ONNX_EXPECTED_SIZE,
    expectedSha256: MIRAGE_READER_ONNX_EXPECTED_SHA256,
    skeletonFile: 'mirage_reader_skeleton.bin.gz',
    skeletonSize: 14_693,
    skeletonSha256:
      '2cd018ecfd98b15256fc428df1ee5fc59c6c89ec08307295b12907b70c04e51d',
    placementsFile: 'mirage_reader_placements.json',
    placementsSize: 14_052,
    placementsSha256:
      '0fc3004ef4cc44625592ec80a9137510358b5073b6269b2b74b088f781a98ef3',
  },
  crossKv: {
    key: 'crossKv',
    onnxFile: MIRAGE_CROSS_KV_ONNX_FILE,
    expectedSize: MIRAGE_CROSS_KV_ONNX_EXPECTED_SIZE,
    expectedSha256: MIRAGE_CROSS_KV_ONNX_EXPECTED_SHA256,
    skeletonFile: 'mirage_cross_kv_skeleton.bin.gz',
    skeletonSize: 187_130,
    skeletonSha256:
      '116c7349d970a129beef859c701ebf24c011f8e68f87c2610620fdf710fb2346',
    placementsFile: 'mirage_cross_kv_placements.json',
    placementsSize: 8_773,
    placementsSha256:
      '0285d6af9aebc00d49c8c44daaee351fe7342ced862bd99e28d7d5cfe5c33c07',
  },
  decoderStep: {
    key: 'decoderStep',
    onnxFile: MIRAGE_DECODER_STEP_ONNX_FILE,
    expectedSize: MIRAGE_DECODER_STEP_ONNX_EXPECTED_SIZE,
    expectedSha256: MIRAGE_DECODER_STEP_ONNX_EXPECTED_SHA256,
    skeletonFile: 'mirage_decoder_step_skeleton.bin.gz',
    skeletonSize: 138_840,
    skeletonSha256:
      'c68591eebea7c2e400e8a066b3778ae794239609bf52d1d1770cd0fa97941863',
    placementsFile: 'mirage_decoder_step_placements.json',
    placementsSize: 123_684,
    placementsSha256:
      'ea3c1313d49916c0c790975656ab5a5af3c9d5b98b542a939b9612c1b87b7930',
  },
  decoderTables: {
    key: 'decoderTables',
    file: MIRAGE_DECODER_TABLES_FILE,
    expectedSize: MIRAGE_DECODER_TABLES_EXPECTED_SIZE,
    expectedSha256: MIRAGE_DECODER_TABLES_EXPECTED_SHA256,
  },
} as const;

export function hasMirageFullComponentAssets(
  componentKey: MirageFullPipelineComponentKey,
  rootDir: string = MIRAGE_PACKAGE_ROOT
): boolean {
  if (componentKey === 'decoderTables') {
    return true;
  }
  const pin = MIRAGE_FULL_PIPELINE_PIN[componentKey];
  const assetsDir = path.resolve(rootDir, MIRAGE_RECONSTRUCTION_REL_DIR);
  return (
    existsSync(path.join(assetsDir, pin.skeletonFile)) &&
    existsSync(path.join(assetsDir, pin.placementsFile))
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

function verifyMirageCactArchiveBytes(cactBytes: Buffer): {
  codebook: Float64Array;
  entries: ReturnType<typeof parseMirageCactDirectory>['entries'];
} {
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
  const parsed = parseMirageCactDirectory(cactBytes);
  return { codebook: parsed.codebook, entries: parsed.entries };
}

function writeVerifiedArtifactWithGzip(targetPath: string, payload: Buffer): void {
  mkdirSync(path.dirname(targetPath), { recursive: true });
  const tempPath = `${targetPath}.part-${randomUUID()}`;
  try {
    writeFileSync(tempPath, payload);
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
    const gzBytes = gzipSync(payload, { level: 9 });
    writeFileSync(tempGz, gzBytes);
    renameSync(tempGz, `${targetPath}.gz`);
  } catch {
    try {
      rmSync(tempGz, { force: true });
    } catch {
      // Non-critical compression fallback
    }
  }
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
  const { codebook } = verifyMirageCactArchiveBytes(cactBytes);

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
    const descriptor = {
      dtype: entry.dtype,
      shape: entry.shape,
      offset: entry.offset,
      nbytes: entry.bytes,
      group: entry.group,
      bits: entry.bits,
    };
    if (entry.dtype === 3) {
      decodedWeights.set(
        name,
        decodeCqMatrix(cactBytes, descriptor, codebook, fp16Table, {
          transpose: true,
        })
      );
    } else if (entry.dtype === 1) {
      decodedWeights.set(name, decodeFp16Entry(cactBytes, descriptor, fp16Table));
    } else if (entry.dtype === 2) {
      decodedWeights.set(name, decodeFp32Entry(cactBytes, descriptor));
    } else {
      throw new Error(`Unsupported archive dtype ${entry.dtype} for ${name}`);
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
  writeVerifiedArtifactWithGzip(targetPath, onnxBuffer);
  return onnxBuffer;
}

/**
 * Extracts and SHA-256-verifies the compact `7,350,719`-byte `mirage-decoder-tables.cact`
 * asset containing token embeddings, Engram tables, CTC charset, and SentencePiece vocab.
 */
export function extractVerifiedMirageDecoderTablesBytes(
  cactBytes: Buffer
): Buffer {
  verifyMirageCactArchiveBytes(cactBytes);
  const raw = extractDecoderTablesCactBytes(cactBytes);
  const buf = Buffer.from(raw.buffer, raw.byteOffset, raw.byteLength);
  const pin = MIRAGE_FULL_PIPELINE_PIN.decoderTables;
  if (buf.byteLength !== pin.expectedSize) {
    throw new Error(
      `Extracted ${pin.file} size mismatch: expected ${pin.expectedSize}, got ${buf.byteLength}`
    );
  }
  const sha256 = createHash('sha256').update(buf).digest('hex');
  if (sha256 !== pin.expectedSha256) {
    throw new Error(
      `Extracted ${pin.file} SHA-256 mismatch: expected ${pin.expectedSha256}, got ${sha256}`
    );
  }
  return buf;
}

/**
 * Reconstructs one of the full-pipeline ONNX graphs (`reader`, `crossKv`, `decoderStep`)
 * directly from `mirage.cact` using zeroed skeletons and deterministic placements.
 */
export function reconstructVerifiedMirageFullOnnxBytes(
  componentKey: Exclude<MirageFullPipelineComponentKey, 'decoderTables'>,
  cactBytes: Buffer,
  rootDir: string = MIRAGE_PACKAGE_ROOT
): Buffer {
  const { codebook, entries } = verifyMirageCactArchiveBytes(cactBytes);
  const pin = MIRAGE_FULL_PIPELINE_PIN[componentKey];
  const assetsDir = path.resolve(rootDir, MIRAGE_RECONSTRUCTION_REL_DIR);
  const skeletonGzBytes = readVerifiedAssetFile(path.join(assetsDir, pin.skeletonFile), {
    file: pin.skeletonFile,
    size: pin.skeletonSize,
    sha256: pin.skeletonSha256,
  });
  const placementsBytes = readVerifiedAssetFile(path.join(assetsDir, pin.placementsFile), {
    file: pin.placementsFile,
    size: pin.placementsSize,
    sha256: pin.placementsSha256,
  });

  const placements = JSON.parse(
    placementsBytes.toString('utf8')
  ) as MiragePlacementRecord[];
  const onnxBuffer = gunzipSync(skeletonGzBytes);
  if (onnxBuffer.byteLength !== pin.expectedSize) {
    throw new Error(
      `${pin.onnxFile} skeleton size mismatch: expected ${pin.expectedSize}, got ${onnxBuffer.byteLength}`
    );
  }

  const fp16Table = getFp16ToFp32Table();
  const entryCache = new Map<number, Float32Array>();

  for (const placement of placements) {
    const transformed = resolvePlacementWeight(
      cactBytes,
      entries,
      codebook,
      fp16Table,
      placement,
      entryCache
    );
    const srcBytes = Buffer.from(
      transformed.buffer,
      transformed.byteOffset,
      transformed.byteLength
    );
    if (srcBytes.byteLength !== placement.length) {
      throw new Error(
        `Placement byte length mismatch for ${placement.tensor}: expected ${placement.length}, got ${srcBytes.byteLength}`
      );
    }
    srcBytes.copy(onnxBuffer, placement.offset);
  }

  const actualSha256 = createHash('sha256').update(onnxBuffer).digest('hex');
  if (actualSha256 !== pin.expectedSha256) {
    throw new Error(
      `Reconstructed ${pin.onnxFile} SHA-256 mismatch: expected ${pin.expectedSha256}, got ${actualSha256}`
    );
  }
  return onnxBuffer;
}

export function reconstructVerifiedMirageComponentBytes(
  componentKey: MirageFullPipelineComponentKey,
  cactBytes: Buffer,
  rootDir: string = MIRAGE_PACKAGE_ROOT
): Buffer {
  if (componentKey === 'decoderTables') {
    return extractVerifiedMirageDecoderTablesBytes(cactBytes);
  }
  return reconstructVerifiedMirageFullOnnxBytes(componentKey, cactBytes, rootDir);
}

export function ensureReconstructedMirageComponentFile(
  componentKey: MirageFullPipelineComponentKey,
  targetPath: string,
  cactBytes: Buffer,
  rootDir: string = MIRAGE_PACKAGE_ROOT
): Buffer {
  const payload = reconstructVerifiedMirageComponentBytes(
    componentKey,
    cactBytes,
    rootDir
  );
  writeVerifiedArtifactWithGzip(targetPath, payload);
  return payload;
}

