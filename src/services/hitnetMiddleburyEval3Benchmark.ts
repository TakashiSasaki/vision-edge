import {
  HITNET_ORT_130_FAMILY_PROFILES,
  type OrtGpuKernelProfileSummary,
  type OrtVersionDiagnosticFailure,
  type OrtVersionDiagnosticResponse,
  type OrtVersionDiagnosticSuccess,
} from './hitnetOrtComparisonHelper';
import {
  createBrowserOrt130DiagnosticRunner,
  sha256HexBytes,
  validateHitnetFinalOutputs,
  type ResidualDiagnosticWorkerRunner,
} from './hitnetResidualLocalization';
import {
  verifyCanonicalM1NoN1Replacements,
  type N1OffVerificationProof,
} from './hitnetPostConv7TapReplay';
import { parseM1ModelProtoWithRawSlices } from './hitnetRgbStemFirstNonZero';
import {
  assertDevFixtureEnabled,
  HITNET_DEV_NON_JSEP_FIXTURE_SPEC,
} from './stereoDepthWebGpuDevFixture';
import { HITNET_M1_FULL_GRAPH_SHA256 } from './stereoDepthWorkerCore';
import { summarizeRawTensorAndParity, type RawTensorParityStats } from './hitnetGatherNdDiagnostic';

export type MiddleburyEval3SceneId = 'Motorcycle' | 'Teddy';
export type MiddleburyEyeId = 'left' | 'right';

export interface MiddleburyExpectedFileSpec {
  relativePath: string;
  byteLength: number;
  sha256: string;
  sourceArchiveUrl: string;
  zipMember: string;
}

export interface MiddleburySceneSpec {
  sceneId: MiddleburyEval3SceneId;
  edition: string;
  baseFixturePath: string;
  expectedWidth: number;
  expectedHeight: number;
  expectedNdisp: number;
  expectedDyavg: number;
  expectedDymax: number;
  rectificationCaveat: string;
  citation: string;
  files: {
    im0: MiddleburyExpectedFileSpec;
    im1: MiddleburyExpectedFileSpec;
    calib: MiddleburyExpectedFileSpec;
    disp0GT: MiddleburyExpectedFileSpec;
    mask0nocc: MiddleburyExpectedFileSpec;
    disp1GT: MiddleburyExpectedFileSpec;
    mask1nocc: MiddleburyExpectedFileSpec;
  };
}

export const MIDDLEBURY_EVAL3_Q_ARCHIVES = {
  dataQ: {
    url: 'https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip',
    byteLength: 30937533,
    sha256: 'a1411f283b523541e0d2e8b3a1dd6618974a0e539aea0122490791a041fffc09',
  },
  gt0Q: {
    url: 'https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip',
    byteLength: 13555667,
    sha256: '489b6a4015951a3c865aa54c0f37dcb648b5d0efad8019264e22499cef35ca45',
  },
  gt1Q: {
    url: 'https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT1-Q.zip',
    byteLength: 13523003,
    sha256: '476af471b1e2697c4baee9d12735502a17fc22c26eae5f379008f41268273682',
  },
} as const;

export const MIDDLEBURY_EVAL3_Q_SCENES: Readonly<Record<MiddleburyEval3SceneId, MiddleburySceneSpec>> = {
  Motorcycle: {
    sceneId: 'Motorcycle',
    edition: 'MiddEval3-trainingQ-2014-imperfect',
    baseFixturePath: '/dev-fixtures/middlebury-eval3/Motorcycle',
    expectedWidth: 741,
    expectedHeight: 497,
    expectedNdisp: 70,
    expectedDyavg: 0.23,
    expectedDymax: 0.379,
    rectificationCaveat:
      '2014 imperfect-rectification Q edition: calibrated residual vertical y-disparity dyavg=0.23 px, dymax=0.379 px at native Q resolution.',
    citation:
      'D. Scharstein et al., "High-resolution stereo datasets with subpixel-accurate ground truth," GCPR 2014.',
    files: {
      im0: {
        relativePath: 'im0.png',
        byteLength: 653821,
        sha256: 'a7b3988b02447c190309b0503da3db036d0c6c0380c3b9b6acda3d66594f5312',
        sourceArchiveUrl: MIDDLEBURY_EVAL3_Q_ARCHIVES.dataQ.url,
        zipMember: 'MiddEval3/trainingQ/Motorcycle/im0.png',
      },
      im1: {
        relativePath: 'im1.png',
        byteLength: 649348,
        sha256: '9bb3c7fa53a27f003223b2fc504f86e1647f70cb2d25f3baedf3127957934a87',
        sourceArchiveUrl: MIDDLEBURY_EVAL3_Q_ARCHIVES.dataQ.url,
        zipMember: 'MiddEval3/trainingQ/Motorcycle/im1.png',
      },
      calib: {
        relativePath: 'calib.txt',
        byteLength: 207,
        sha256: 'd886e8f78085eb3a13f45e1a70a5fdbeb09a2a2091e364536269318d4ffa5e52',
        sourceArchiveUrl: MIDDLEBURY_EVAL3_Q_ARCHIVES.dataQ.url,
        zipMember: 'MiddEval3/trainingQ/Motorcycle/calib.txt',
      },
      disp0GT: {
        relativePath: 'disp0GT.pfm',
        byteLength: 1473129,
        sha256: 'b7c4557ec53314d97643f3e3f7c4098eb992aaab193937e03a7759b42dfa6f11',
        sourceArchiveUrl: MIDDLEBURY_EVAL3_Q_ARCHIVES.gt0Q.url,
        zipMember: 'MiddEval3/trainingQ/Motorcycle/disp0GT.pfm',
      },
      mask0nocc: {
        relativePath: 'mask0nocc.png',
        byteLength: 17577,
        sha256: 'dfa4e6db93fde4bdc33a32286478ebff23a58d775be95d859e8d72e1c20d6bd8',
        sourceArchiveUrl: MIDDLEBURY_EVAL3_Q_ARCHIVES.gt0Q.url,
        zipMember: 'MiddEval3/trainingQ/Motorcycle/mask0nocc.png',
      },
      disp1GT: {
        relativePath: 'disp1GT.pfm',
        byteLength: 1473129,
        sha256: 'c93be985670060d9d8a124156a1567b35c8c52f8fc63b3ad56ea546720609d92',
        sourceArchiveUrl: MIDDLEBURY_EVAL3_Q_ARCHIVES.gt1Q.url,
        zipMember: 'MiddEval3/trainingQ/Motorcycle/disp1GT.pfm',
      },
      mask1nocc: {
        relativePath: 'mask1nocc.png',
        byteLength: 17347,
        sha256: 'ede24bba9051025ccdd3fce65ac95842997a768d19dda8de07fb810e74293dfe',
        sourceArchiveUrl: MIDDLEBURY_EVAL3_Q_ARCHIVES.gt1Q.url,
        zipMember: 'MiddEval3/trainingQ/Motorcycle/mask1nocc.png',
      },
    },
  },
  Teddy: {
    sceneId: 'Teddy',
    edition: 'MiddEval3-trainingQ-2003-rectified',
    baseFixturePath: '/dev-fixtures/middlebury-eval3/Teddy',
    expectedWidth: 450,
    expectedHeight: 375,
    expectedNdisp: 64,
    expectedDyavg: 0,
    expectedDymax: 0,
    rectificationCaveat:
      '2003 structured-light rectified Q edition included in MiddEval3: dyavg=0 px, dymax=0 px.',
    citation:
      'D. Scharstein and R. Szeliski, "High-accuracy stereo depth maps using structured light," CVPR 2003.',
    files: {
      im0: {
        relativePath: 'im0.png',
        byteLength: 309929,
        sha256: '0620f5cb513ca67aaaf3702c0b1e261641d67171ad56fa0934ca4cede638b023',
        sourceArchiveUrl: MIDDLEBURY_EVAL3_Q_ARCHIVES.dataQ.url,
        zipMember: 'MiddEval3/trainingQ/Teddy/im0.png',
      },
      im1: {
        relativePath: 'im1.png',
        byteLength: 311265,
        sha256: '2c377edb3c2144f32c56c4c969ec5dcb6b6bf02db9c95e21d24e70ac5fe6b350',
        sourceArchiveUrl: MIDDLEBURY_EVAL3_Q_ARCHIVES.dataQ.url,
        zipMember: 'MiddEval3/trainingQ/Teddy/im1.png',
      },
      calib: {
        relativePath: 'calib.txt',
        byteLength: 165,
        sha256: '25c523c597aa84679f09e19f9cc79d899a4172ca40dc7b33314d7a9a752e25bc',
        sourceArchiveUrl: MIDDLEBURY_EVAL3_Q_ARCHIVES.dataQ.url,
        zipMember: 'MiddEval3/trainingQ/Teddy/calib.txt',
      },
      disp0GT: {
        relativePath: 'disp0GT.pfm',
        byteLength: 675021,
        sha256: '895744a0bed14c428b98a8aeba63053f4b52279ba8284ecf9d15fa045444217e',
        sourceArchiveUrl: MIDDLEBURY_EVAL3_Q_ARCHIVES.gt0Q.url,
        zipMember: 'MiddEval3/trainingQ/Teddy/disp0GT.pfm',
      },
      mask0nocc: {
        relativePath: 'mask0nocc.png',
        byteLength: 5827,
        sha256: 'f654d8b5e8494f0ed054ae75cea90096a60ab79093fba7f408b629ad0b462229',
        sourceArchiveUrl: MIDDLEBURY_EVAL3_Q_ARCHIVES.gt0Q.url,
        zipMember: 'MiddEval3/trainingQ/Teddy/mask0nocc.png',
      },
      disp1GT: {
        relativePath: 'disp1GT.pfm',
        byteLength: 675021,
        sha256: '74ed7de8083674576aca379b604d287699bd75e23a873f2b96c73cedd0c25d28',
        sourceArchiveUrl: MIDDLEBURY_EVAL3_Q_ARCHIVES.gt1Q.url,
        zipMember: 'MiddEval3/trainingQ/Teddy/disp1GT.pfm',
      },
      mask1nocc: {
        relativePath: 'mask1nocc.png',
        byteLength: 5837,
        sha256: '31b1c9678c2e64ed4a237ed4930ac0763d389f1b29cb2a9f6bb7ee6ce94cc4c4',
        sourceArchiveUrl: MIDDLEBURY_EVAL3_Q_ARCHIVES.gt1Q.url,
        zipMember: 'MiddEval3/trainingQ/Teddy/mask1nocc.png',
      },
    },
  },
};

export const HITNET_MIDDLEBURY_MODEL_WIDTH = 320;
export const HITNET_MIDDLEBURY_MODEL_HEIGHT = 240;
export const HITNET_MIDDLEBURY_PIXEL_COUNT =
  HITNET_MIDDLEBURY_MODEL_WIDTH * HITNET_MIDDLEBURY_MODEL_HEIGHT;

export interface MiddleburyCalibData {
  cam0: number[];
  cam1: number[];
  doffs: number;
  baseline: number;
  width: number;
  height: number;
  ndisp: number;
  isint: number;
  vmin: number;
  vmax: number;
  dyavg: number;
  dymax: number;
}

export interface ParsedPfmGrid {
  magic: 'Pf';
  width: number;
  height: number;
  scaleFactor: number;
  littleEndian: boolean;
  /** Row-major top-to-bottom Float32 disparity map matching PNG image coordinates. */
  dataTopToBottom: Float32Array;
}

export interface ParsedPngMaskGrid {
  width: number;
  height: number;
  colorType: number;
  /** Single-channel mask bytes in top-to-bottom row-major order: 255=nonocc, 128=occ, 0=invalid. */
  grayTopToBottom: Uint8Array;
}

export interface CropRectPx {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ResampledMiddleburyEyeGtGrid {
  eye: MiddleburyEyeId;
  outputTensorName: 'reference_output_disparity' | 'secondary_output_disparity';
  srcWidth: number;
  srcHeight: number;
  cropRect: CropRectPx;
  modelWidth: 320;
  modelHeight: 240;
  scaleX: number;
  scaleY: number;
  ndispNative: number;
  ndispModelPx: number;
  /** GT disparity scaled to 320x240 model pixel units at nonocc/allValid pixels; NaN elsewhere. */
  gtDisparityModelPx: Float32Array;
  /** Fixed pre-prediction masks: 1=included, 0=excluded. */
  allValidMask: Uint8Array;
  nonoccMask: Uint8Array;
  coverage: {
    totalModelPixels: number;
    allValidGtCount: number;
    allValidGtRate: number;
    nonoccValidGtCount: number;
    nonoccValidGtRate: number;
    occludedValidGtCount: number;
    gtInvalidOrNonPositiveCount: number;
    gtOverNdispCount: number;
    cropLostCorrespondenceCount: number;
  };
}

export interface MiddleburyMaskMetrics {
  maskLabel: 'nonocc' | 'allValidGt';
  evaluatedCount: number;
  coverageRateOfModelFrame: number;
  finitePredCount: number;
  nonFinitePredCount: number;
  negativePredCount: number;
  invalidPredCount: number;
  invalidPredRate: number;
  epeModelPx: number;
  epeNativeQPx: number;
  p95AbsErrorModelPx: number;
  p95AbsErrorNativeQPx: number;
  maxAbsErrorModelPx: number;
  bad1Count: number;
  bad1Rate: number;
  bad3Count: number;
  bad3Rate: number;
  meanSignedErrorModelPx: number;
  meanPredModelPx: number;
  medianPredModelPx: number;
  minPredModelPx: number;
  maxPredModelPx: number;
}

export interface EyeSignSanityCheck {
  correctEyePosSignEpeModelPx: number;
  negatedGtEpeModelPx: number;
  swappedEyePosSignEpeModelPx: number;
  sharedSwapCheckCount: number;
  positiveSignConfirmed: boolean;
}

export interface MiddleburyEyeArmMetrics {
  eye: MiddleburyEyeId;
  outputTensorName: 'reference_output_disparity' | 'secondary_output_disparity';
  nonocc: MiddleburyMaskMetrics;
  allValidGt: MiddleburyMaskMetrics;
  eyeSignSanity: EyeSignSanityCheck;
}

export interface MiddleburyArmExecutionResult {
  provider: 'webgpu' | 'wasm';
  graphOptimizationLevel: 'basic';
  modelSha256: string;
  modelByteLength: number;
  feedSha256: string;
  configuredProviders: readonly string[];
  fallbackReason: string | null;
  loadedOrtResources: string[];
  foreignOrtResources: string[];
  workerGpuAdapter: string | null;
  gpuValidationErrors: string[];
  profilingOffTimings: {
    createMs: number;
    coldRunMs: number;
    warmRunMs: number[];
    warmMedianMs: number | null;
  };
  dispatchCheckProfilingOn: {
    executed: boolean;
    dispatchObserved: boolean;
    summary: OrtGpuKernelProfileSummary | null;
  } | null;
  leftPrimary: MiddleburyEyeArmMetrics;
  rightSecondary: MiddleburyEyeArmMetrics;
}

export interface MiddleburySceneBenchmarkReport {
  sceneId: MiddleburyEval3SceneId;
  edition: string;
  rectificationCaveat: string;
  citation: string;
  calib: MiddleburyCalibData;
  cropRect: CropRectPx;
  scaleX: number;
  scaleY: number;
  feedSha256: string;
  feedByteLength: number;
  fixtureSha256s: Record<keyof MiddleburySceneSpec['files'], string>;
  leftGtCoverage: ResampledMiddleburyEyeGtGrid['coverage'];
  rightGtCoverage: ResampledMiddleburyEyeGtGrid['coverage'];
  webgpu: MiddleburyArmExecutionResult;
  wasm: MiddleburyArmExecutionResult;
  webgpuVsWasmParity: {
    leftPrimaryFullFrame: RawTensorParityStats;
    leftPrimaryNonocc: RawTensorParityStats;
    rightSecondaryFullFrame: RawTensorParityStats;
    rightSecondaryNonocc: RawTensorParityStats;
  };
}

export interface ProductionGoNoGoEvaluation {
  candidateScope: 'DEV-only unmodified canonical M1 non-JSEP ORT 1.30.0 (opt=basic)';
  gates: Array<{
    id: string;
    label: string;
    passed: boolean;
    detail: string;
  }>;
  overallStatus: 'PASS' | 'FAIL_ACCURACY_OR_PARITY' | 'FAIL_RUNTIME_OR_DISPATCH';
  explicitLimits: string[];
}

export interface HitnetMiddleburyEval3BenchmarkReport {
  ok: boolean;
  benchmarkMode: 'local-public-fixture-320x240-not-official-leaderboard';
  pinnedOrtVersion: '1.30.0';
  runtimeAssetFamily: 'asyncify';
  graphOptimizationLevel: 'basic';
  warmRuns: number;
  m1ByteLength: number;
  m1Sha256: string;
  n1OffProof: N1OffVerificationProof;
  perEyeSignMappingNote: string;
  scenes: Record<MiddleburyEval3SceneId, MiddleburySceneBenchmarkReport>;
  productionGoNoGoEvaluation: ProductionGoNoGoEvaluation;
}

export interface LoadedMiddleburySceneAssets {
  sceneId: MiddleburyEval3SceneId;
  spec: MiddleburySceneSpec;
  calibText: string;
  calib: MiddleburyCalibData;
  im0Bytes: Uint8Array;
  im1Bytes: Uint8Array;
  disp0GtPfm: ParsedPfmGrid;
  disp1GtPfm: ParsedPfmGrid;
  mask0Nocc: ParsedPngMaskGrid;
  mask1Nocc: ParsedPngMaskGrid;
  fixtureSha256s: Record<keyof MiddleburySceneSpec['files'], string>;
}

export type MiddleburyStereoFeedBuilder = (params: {
  im0Bytes: Uint8Array;
  im1Bytes: Uint8Array;
  srcWidth: number;
  srcHeight: number;
  cropRect: CropRectPx;
}) => Promise<{ feedFloat32: Float32Array; feedSha256: string }>;

export function parseMiddleburyCalibTxt(text: string): MiddleburyCalibData {
  const map = new Map<string, string>();
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq <= 0) continue;
    const key = line.slice(0, eq).trim();
    const val = line.slice(eq + 1).trim();
    map.set(key, val);
  }

  const parseNum = (k: string): number => {
    const v = map.get(k);
    if (v === undefined) throw new Error(`Missing calib.txt field "${k}"`);
    const n = Number(v);
    if (!Number.isFinite(n)) throw new Error(`Invalid calib.txt numeric field ${k}="${v}"`);
    return n;
  };
  const parseCam = (k: string): number[] => {
    const raw = map.get(k);
    if (!raw || !raw.startsWith('[') || !raw.endsWith(']')) {
      throw new Error(`Invalid calib.txt matrix field ${k}="${String(raw)}"`);
    }
    const nums = raw
      .slice(1, -1)
      .replace(/;/g, ' ')
      .trim()
      .split(/\s+/)
      .map(Number);
    if (nums.length !== 9 || nums.some((n) => !Number.isFinite(n))) {
      throw new Error(`Invalid 3x3 matrix in calib.txt for ${k}`);
    }
    return nums;
  };

  const width = Math.trunc(parseNum('width'));
  const height = Math.trunc(parseNum('height'));
  const ndisp = parseNum('ndisp');
  if (width <= 0 || height <= 0 || ndisp <= 0) {
    throw new Error(`Invalid calib.txt dimensions/ndisp: ${width}x${height}, ndisp=${ndisp}`);
  }

  return {
    cam0: parseCam('cam0'),
    cam1: parseCam('cam1'),
    doffs: parseNum('doffs'),
    baseline: parseNum('baseline'),
    width,
    height,
    ndisp,
    isint: Math.trunc(parseNum('isint')),
    vmin: parseNum('vmin'),
    vmax: parseNum('vmax'),
    dyavg: parseNum('dyavg'),
    dymax: parseNum('dymax'),
  };
}

export function parsePfmSingleBand(bytes: Uint8Array): ParsedPfmGrid {
  const readLine = (start: number): { line: string; next: number } => {
    let i = start;
    while (i < bytes.byteLength && bytes[i] !== 0x0a) i++;
    if (i >= bytes.byteLength) {
      throw new Error('Truncated PFM header before newline');
    }
    let end = i;
    if (end > start && bytes[end - 1] === 0x0d) end--;
    const line = new TextDecoder('ascii').decode(bytes.subarray(start, end));
    return { line, next: i + 1 };
  };

  const l1 = readLine(0);
  if (l1.line.trim() !== 'Pf') {
    throw new Error(`Expected single-band PFM header "Pf", got "${l1.line}"`);
  }
  const l2 = readLine(l1.next);
  const dims = l2.line.trim().split(/\s+/);
  if (dims.length !== 2) {
    throw new Error(`Invalid PFM dimensions line "${l2.line}"`);
  }
  const width = Number.parseInt(dims[0], 10);
  const height = Number.parseInt(dims[1], 10);
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error(`Invalid PFM dimensions ${width}x${height}`);
  }
  const l3 = readLine(l2.next);
  const scaleFactor = Number.parseFloat(l3.line.trim());
  if (!Number.isFinite(scaleFactor) || scaleFactor === 0) {
    throw new Error(`Invalid PFM scale factor "${l3.line}"`);
  }

  const littleEndian = scaleFactor < 0;
  const expectedPayloadBytes = width * height * 4;
  const actualPayloadBytes = bytes.byteLength - l3.next;
  if (actualPayloadBytes !== expectedPayloadBytes) {
    throw new Error(
      `PFM payload size mismatch: got ${actualPayloadBytes} bytes, expected ${expectedPayloadBytes} for ${width}x${height}`
    );
  }

  const dv = new DataView(bytes.buffer, bytes.byteOffset + l3.next, expectedPayloadBytes);
  const dataTopToBottom = new Float32Array(width * height);
  // PFM stores raster rows bottom-to-top; flip vertically to top-to-bottom image coordinates.
  for (let yFile = 0; yFile < height; yFile++) {
    const yImg = height - 1 - yFile;
    const srcRowByteBase = yFile * width * 4;
    const dstRowBase = yImg * width;
    for (let x = 0; x < width; x++) {
      dataTopToBottom[dstRowBase + x] = dv.getFloat32(srcRowByteBase + x * 4, littleEndian);
    }
  }

  return {
    magic: 'Pf',
    width,
    height,
    scaleFactor,
    littleEndian,
    dataTopToBottom,
  };
}

async function zlibDecompressBytes(compressed: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream === 'function') {
    const ds = new DecompressionStream('deflate');
    const copy = new Uint8Array(compressed.byteLength);
    copy.set(compressed);
    const stream = new Blob([copy]).stream().pipeThrough(ds);
    const buf = await new Response(stream).arrayBuffer();
    return new Uint8Array(buf);
  }
  throw new Error('DecompressionStream("deflate") unavailable for bit-exact PNG mask decoding');
}

function paethPredictor(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

export async function parsePngGrayscaleMaskBitExact(bytes: Uint8Array): Promise<ParsedPngMaskGrid> {
  const PNG_SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (bytes.byteLength < 8 || PNG_SIG.some((v, i) => bytes[i] !== v)) {
    throw new Error('Invalid PNG signature in mask file');
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let pos = 8;
  let width = 0;
  let height = 0;
  let colorType = 0;
  const idatChunks: Uint8Array[] = [];

  while (pos + 12 <= bytes.byteLength) {
    const chunkLen = dv.getUint32(pos, false);
    const type = String.fromCharCode(bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]);
    const dataStart = pos + 8;
    const dataEnd = dataStart + chunkLen;
    if (dataEnd + 4 > bytes.byteLength) {
      throw new Error(`Truncated PNG chunk ${type}`);
    }
    if (type === 'IHDR') {
      width = dv.getUint32(dataStart, false);
      height = dv.getUint32(dataStart + 4, false);
      const bitDepth = bytes[dataStart + 8];
      colorType = bytes[dataStart + 9];
      const compression = bytes[dataStart + 10];
      const filter = bytes[dataStart + 11];
      const interlace = bytes[dataStart + 12];
      if (bitDepth !== 8 || compression !== 0 || filter !== 0 || interlace !== 0) {
        throw new Error(
          `Unsupported PNG mask encoding: bitDepth=${bitDepth}, colorType=${colorType}, interlace=${interlace}`
        );
      }
      if (colorType !== 0 && colorType !== 2 && colorType !== 6) {
        throw new Error(`Unsupported PNG mask colorType=${colorType} (expected 0 grayscale, 2 RGB, or 6 RGBA)`);
      }
    } else if (type === 'IDAT') {
      idatChunks.push(bytes.subarray(dataStart, dataEnd));
    } else if (type === 'IEND') {
      break;
    }
    pos = dataEnd + 4;
  }

  if (width <= 0 || height <= 0 || idatChunks.length === 0) {
    throw new Error('Incomplete PNG mask IHDR/IDAT');
  }

  const totalIdat = idatChunks.reduce((acc, c) => acc + c.byteLength, 0);
  const compressed = new Uint8Array(totalIdat);
  let offset = 0;
  for (const c of idatChunks) {
    compressed.set(c, offset);
    offset += c.byteLength;
  }

  const raw = await zlibDecompressBytes(compressed);
  const bpp = colorType === 0 ? 1 : colorType === 2 ? 3 : 4;
  const stride = width * bpp;
  const expectedRaw = height * (stride + 1);
  if (raw.byteLength !== expectedRaw) {
    throw new Error(`PNG decompressed length mismatch: got ${raw.byteLength}, expected ${expectedRaw}`);
  }

  const reconImage = new Uint8Array(height * stride);
  const grayTopToBottom = new Uint8Array(width * height);

  for (let y = 0; y < height; y++) {
    const srcRowStart = y * (stride + 1);
    const filterType = raw[srcRowStart];
    const rowOutStart = y * stride;
    for (let i = 0; i < stride; i++) {
      const xVal = raw[srcRowStart + 1 + i];
      const a = i >= bpp ? reconImage[rowOutStart + i - bpp] : 0;
      const b = y > 0 ? reconImage[(y - 1) * stride + i] : 0;
      const c = y > 0 && i >= bpp ? reconImage[(y - 1) * stride + i - bpp] : 0;
      let recon: number;
      if (filterType === 0) recon = xVal;
      else if (filterType === 1) recon = (xVal + a) & 0xff;
      else if (filterType === 2) recon = (xVal + b) & 0xff;
      else if (filterType === 3) recon = (xVal + ((a + b) >> 1)) & 0xff;
      else if (filterType === 4) recon = (xVal + paethPredictor(a, b, c)) & 0xff;
      else throw new Error(`Unsupported PNG scanline filter ${filterType}`);
      reconImage[rowOutStart + i] = recon;
    }
    const dstGrayRow = y * width;
    if (bpp === 1) {
      grayTopToBottom.set(reconImage.subarray(rowOutStart, rowOutStart + width), dstGrayRow);
    } else {
      for (let x = 0; x < width; x++) {
        grayTopToBottom[dstGrayRow + x] = reconImage[rowOutStart + x * bpp];
      }
    }
  }

  return { width, height, colorType, grayTopToBottom };
}

export function resampleMiddleburyEyeGtNearestGrid(params: {
  eye: MiddleburyEyeId;
  gtPfm: ParsedPfmGrid;
  maskNocc: ParsedPngMaskGrid;
  calib: MiddleburyCalibData;
  cropRect?: CropRectPx;
}): ResampledMiddleburyEyeGtGrid {
  const { eye, gtPfm, maskNocc, calib } = params;
  if (gtPfm.width !== calib.width || gtPfm.height !== calib.height) {
    throw new Error(
      `GT PFM size ${gtPfm.width}x${gtPfm.height} disagrees with calib.txt ${calib.width}x${calib.height}`
    );
  }
  if (maskNocc.width !== calib.width || maskNocc.height !== calib.height) {
    throw new Error(
      `Mask PNG size ${maskNocc.width}x${maskNocc.height} disagrees with calib.txt ${calib.width}x${calib.height}`
    );
  }

  const cropRect: CropRectPx = params.cropRect ?? {
    x: 0,
    y: 0,
    width: calib.width,
    height: calib.height,
  };
  if (
    cropRect.x < 0 ||
    cropRect.y < 0 ||
    cropRect.width <= 0 ||
    cropRect.height <= 0 ||
    cropRect.x + cropRect.width > calib.width ||
    cropRect.y + cropRect.height > calib.height
  ) {
    throw new Error(
      `Invalid cropRect ${JSON.stringify(cropRect)} for ${calib.width}x${calib.height}`
    );
  }

  const modelWidth = HITNET_MIDDLEBURY_MODEL_WIDTH;
  const modelHeight = HITNET_MIDDLEBURY_MODEL_HEIGHT;
  const scaleX = modelWidth / cropRect.width;
  const scaleY = modelHeight / cropRect.height;

  const gtDisparityModelPx = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT);
  gtDisparityModelPx.fill(NaN);
  const allValidMask = new Uint8Array(HITNET_MIDDLEBURY_PIXEL_COUNT);
  const nonoccMask = new Uint8Array(HITNET_MIDDLEBURY_PIXEL_COUNT);

  let allValidGtCount = 0;
  let nonoccValidGtCount = 0;
  let occludedValidGtCount = 0;
  let gtInvalidOrNonPositiveCount = 0;
  let gtOverNdispCount = 0;
  let cropLostCorrespondenceCount = 0;

  const cropMinX = cropRect.x;
  const cropMaxX = cropRect.x + cropRect.width - 1;
  const cropMinY = cropRect.y;
  const cropMaxY = cropRect.y + cropRect.height - 1;

  for (let my = 0; my < modelHeight; my++) {
    const vSrc = cropRect.y + (my + 0.5) * (cropRect.height / modelHeight) - 0.5;
    const iy = Math.max(cropMinY, Math.min(cropMaxY, Math.round(vSrc)));
    const srcRowOff = iy * calib.width;
    const dstRowOff = my * modelWidth;

    for (let mx = 0; mx < modelWidth; mx++) {
      const dstIdx = dstRowOff + mx;
      const uSrc = cropRect.x + (mx + 0.5) * (cropRect.width / modelWidth) - 0.5;
      const ix = Math.max(cropMinX, Math.min(cropMaxX, Math.round(uSrc)));
      const srcIdx = srcRowOff + ix;

      // Nearest-neighbor lookup directly on the discrete source GT grid avoids interpolation
      // across Infinity holes or depth discontinuities.
      const gtNative = gtPfm.dataTopToBottom[srcIdx];
      if (!Number.isFinite(gtNative) || gtNative <= 0) {
        gtInvalidOrNonPositiveCount++;
        continue;
      }
      if (gtNative > calib.ndisp) {
        gtOverNdispCount++;
        continue;
      }

      // Pre-prediction correspondence check: verify matching pixel stays within cropRect.
      // Left view0: x_R = x_L - d_0; Right view1: x_L = x_R + d_1.
      const xMatch = eye === 'left' ? ix - gtNative : ix + gtNative;
      if (xMatch < cropMinX || xMatch > cropMaxX) {
        cropLostCorrespondenceCount++;
        continue;
      }

      allValidMask[dstIdx] = 1;
      allValidGtCount++;
      gtDisparityModelPx[dstIdx] = gtNative * scaleX;

      if (maskNocc.grayTopToBottom[srcIdx] === 255) {
        nonoccMask[dstIdx] = 1;
        nonoccValidGtCount++;
      } else {
        occludedValidGtCount++;
      }
    }
  }

  return {
    eye,
    outputTensorName:
      eye === 'left' ? 'reference_output_disparity' : 'secondary_output_disparity',
    srcWidth: calib.width,
    srcHeight: calib.height,
    cropRect,
    modelWidth,
    modelHeight,
    scaleX,
    scaleY,
    ndispNative: calib.ndisp,
    ndispModelPx: calib.ndisp * scaleX,
    gtDisparityModelPx,
    allValidMask,
    nonoccMask,
    coverage: {
      totalModelPixels: HITNET_MIDDLEBURY_PIXEL_COUNT,
      allValidGtCount,
      allValidGtRate: allValidGtCount / HITNET_MIDDLEBURY_PIXEL_COUNT,
      nonoccValidGtCount,
      nonoccValidGtRate: nonoccValidGtCount / HITNET_MIDDLEBURY_PIXEL_COUNT,
      occludedValidGtCount,
      gtInvalidOrNonPositiveCount,
      gtOverNdispCount,
      cropLostCorrespondenceCount,
    },
  };
}

export function evaluateMiddleburyDisparityMask(
  pred: Float32Array,
  gtGrid: ResampledMiddleburyEyeGtGrid,
  maskLabel: 'nonocc' | 'allValidGt'
): MiddleburyMaskMetrics {
  if (pred.length !== HITNET_MIDDLEBURY_PIXEL_COUNT) {
    throw new Error(
      `Expected prediction length ${HITNET_MIDDLEBURY_PIXEL_COUNT}, got ${pred.length}`
    );
  }
  const mask = maskLabel === 'nonocc' ? gtGrid.nonoccMask : gtGrid.allValidMask;
  const gtArr = gtGrid.gtDisparityModelPx;
  const absErrors: number[] = [];
  const finitePreds: number[] = [];

  let evaluatedCount = 0;
  let finitePredCount = 0;
  let nonFinitePredCount = 0;
  let negativePredCount = 0;
  let bad1Count = 0;
  let bad3Count = 0;
  let absErrorSum = 0;
  let signedErrorSum = 0;
  let maxAbsErrorModelPx = 0;
  let predSum = 0;
  let minPredModelPx = Infinity;
  let maxPredModelPx = -Infinity;

  for (let i = 0; i < HITNET_MIDDLEBURY_PIXEL_COUNT; i++) {
    if (mask[i] !== 1) continue;
    evaluatedCount++;
    const gt = gtArr[i];
    const p = pred[i];

    if (!Number.isFinite(p)) {
      nonFinitePredCount++;
      bad1Count++;
      bad3Count++;
      absErrors.push(Infinity);
      maxAbsErrorModelPx = Infinity;
      continue;
    }

    finitePredCount++;
    if (p < 0) negativePredCount++;
    finitePreds.push(p);
    predSum += p;
    if (p < minPredModelPx) minPredModelPx = p;
    if (p > maxPredModelPx) maxPredModelPx = p;

    const signedErr = p - gt;
    const absErr = Math.abs(signedErr);
    absErrors.push(absErr);
    absErrorSum += absErr;
    signedErrorSum += signedErr;
    if (absErr > maxAbsErrorModelPx) maxAbsErrorModelPx = absErr;
    if (absErr > 1.0) bad1Count++;
    if (absErr > 3.0) bad3Count++;
  }

  absErrors.sort((a, b) => a - b);
  finitePreds.sort((a, b) => a - b);
  const p95Idx =
    absErrors.length > 0
      ? Math.min(absErrors.length - 1, Math.floor(0.95 * (absErrors.length - 1)))
      : 0;
  const medIdx = finitePreds.length > 0 ? Math.floor(finitePreds.length / 2) : 0;
  const invalidPredCount = nonFinitePredCount + negativePredCount;
  const epeModelPx =
    nonFinitePredCount > 0 ? Infinity : finitePredCount > 0 ? absErrorSum / finitePredCount : NaN;
  const p95AbsErrorModelPx = absErrors.length > 0 ? absErrors[p95Idx] : NaN;

  return {
    maskLabel,
    evaluatedCount,
    coverageRateOfModelFrame: evaluatedCount / HITNET_MIDDLEBURY_PIXEL_COUNT,
    finitePredCount,
    nonFinitePredCount,
    negativePredCount,
    invalidPredCount,
    invalidPredRate: evaluatedCount > 0 ? invalidPredCount / evaluatedCount : 0,
    epeModelPx,
    epeNativeQPx: epeModelPx / gtGrid.scaleX,
    p95AbsErrorModelPx,
    p95AbsErrorNativeQPx: p95AbsErrorModelPx / gtGrid.scaleX,
    maxAbsErrorModelPx: absErrors.length > 0 ? maxAbsErrorModelPx : NaN,
    bad1Count,
    bad1Rate: evaluatedCount > 0 ? bad1Count / evaluatedCount : 0,
    bad3Count,
    bad3Rate: evaluatedCount > 0 ? bad3Count / evaluatedCount : 0,
    meanSignedErrorModelPx: finitePredCount > 0 ? signedErrorSum / finitePredCount : NaN,
    meanPredModelPx: finitePredCount > 0 ? predSum / finitePredCount : NaN,
    medianPredModelPx: finitePreds.length > 0 ? finitePreds[medIdx] : NaN,
    minPredModelPx: finitePredCount > 0 ? minPredModelPx : NaN,
    maxPredModelPx: finitePredCount > 0 ? maxPredModelPx : NaN,
  };
}

export function evaluateEyeSignSanity(
  pred: Float32Array,
  expectedEyeGt: ResampledMiddleburyEyeGtGrid,
  oppositeEyeGt: ResampledMiddleburyEyeGtGrid
): EyeSignSanityCheck {
  let correctSum = 0;
  let correctCount = 0;
  let negSum = 0;
  let swapSum = 0;
  let swapCount = 0;

  for (let i = 0; i < HITNET_MIDDLEBURY_PIXEL_COUNT; i++) {
    if (expectedEyeGt.nonoccMask[i] !== 1) continue;
    const p = pred[i];
    if (!Number.isFinite(p)) continue;
    const gt = expectedEyeGt.gtDisparityModelPx[i];
    correctSum += Math.abs(p - gt);
    negSum += Math.abs(p - -gt);
    correctCount++;

    if (oppositeEyeGt.nonoccMask[i] === 1) {
      swapSum += Math.abs(p - oppositeEyeGt.gtDisparityModelPx[i]);
      swapCount++;
    }
  }

  const correctEpe = correctCount > 0 ? correctSum / correctCount : NaN;
  const negEpe = correctCount > 0 ? negSum / correctCount : NaN;
  const swapEpe = swapCount > 0 ? swapSum / swapCount : NaN;
  return {
    correctEyePosSignEpeModelPx: correctEpe,
    negatedGtEpeModelPx: negEpe,
    swappedEyePosSignEpeModelPx: swapEpe,
    sharedSwapCheckCount: swapCount,
    positiveSignConfirmed: Number.isFinite(correctEpe) && Number.isFinite(negEpe) && correctEpe < negEpe,
  };
}

export function summarizeMaskedTensorParity(
  a: Float32Array,
  b: Float32Array,
  mask: Uint8Array
): RawTensorParityStats {
  const maskedA: number[] = [];
  const maskedB: number[] = [];
  for (let i = 0; i < HITNET_MIDDLEBURY_PIXEL_COUNT; i++) {
    if (mask[i] === 1) {
      maskedA.push(a[i]);
      maskedB.push(b[i]);
    }
  }
  return summarizeRawTensorAndParity(new Float32Array(maskedA), new Float32Array(maskedB));
}

export function packRgbaStereoEyesToHitnetFeed(
  leftRgba: Uint8ClampedArray,
  rightRgba: Uint8ClampedArray
): Float32Array {
  const expectedLen = HITNET_MIDDLEBURY_PIXEL_COUNT * 4;
  if (leftRgba.length !== expectedLen || rightRgba.length !== expectedLen) {
    throw new Error(
      `Expected 320x240 RGBA buffers (${expectedLen} bytes), got left=${leftRgba.length}, right=${rightRgba.length}`
    );
  }
  const feed = new Float32Array(HITNET_MIDDLEBURY_PIXEL_COUNT * 6);
  const inv255 = Math.fround(1 / 255);
  const px = HITNET_MIDDLEBURY_PIXEL_COUNT;
  for (let i = 0; i < px; i++) {
    const off = i * 4;
    feed[i] = Math.fround(leftRgba[off] * inv255);
    feed[px + i] = Math.fround(leftRgba[off + 1] * inv255);
    feed[px * 2 + i] = Math.fround(leftRgba[off + 2] * inv255);
    feed[px * 3 + i] = Math.fround(rightRgba[off] * inv255);
    feed[px * 4 + i] = Math.fround(rightRgba[off + 1] * inv255);
    feed[px * 5 + i] = Math.fround(rightRgba[off + 2] * inv255);
  }
  return feed;
}

async function decodePngTo320x240RgbaBrowser(
  pngBytes: Uint8Array,
  expectedWidth: number,
  expectedHeight: number,
  cropRect: CropRectPx
): Promise<Uint8ClampedArray> {
  const copy = new Uint8Array(pngBytes.byteLength);
  copy.set(pngBytes);
  const blob = new Blob([copy], { type: 'image/png' });
  const canvas = document.createElement('canvas');
  canvas.width = HITNET_MIDDLEBURY_MODEL_WIDTH;
  canvas.height = HITNET_MIDDLEBURY_MODEL_HEIGHT;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new Error('2D canvas context unavailable for Middlebury stereo preprocessing');
  }

  if (typeof createImageBitmap === 'function') {
    const bmp = await createImageBitmap(blob);
    try {
      if (bmp.width !== expectedWidth || bmp.height !== expectedHeight) {
        throw new Error(
          `Decoded PNG size ${bmp.width}x${bmp.height} mismatches expected ${expectedWidth}x${expectedHeight}`
        );
      }
      ctx.drawImage(
        bmp,
        cropRect.x,
        cropRect.y,
        cropRect.width,
        cropRect.height,
        0,
        0,
        HITNET_MIDDLEBURY_MODEL_WIDTH,
        HITNET_MIDDLEBURY_MODEL_HEIGHT
      );
      return ctx.getImageData(0, 0, HITNET_MIDDLEBURY_MODEL_WIDTH, HITNET_MIDDLEBURY_MODEL_HEIGHT)
        .data;
    } finally {
      bmp.close();
    }
  }

  const url = URL.createObjectURL(blob);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error('Failed to decode Middlebury PNG image'));
      el.src = url;
    });
    if (img.naturalWidth !== expectedWidth || img.naturalHeight !== expectedHeight) {
      throw new Error(
        `Decoded image size ${img.naturalWidth}x${img.naturalHeight} mismatches expected ${expectedWidth}x${expectedHeight}`
      );
    }
    ctx.drawImage(
      img,
      cropRect.x,
      cropRect.y,
      cropRect.width,
      cropRect.height,
      0,
      0,
      HITNET_MIDDLEBURY_MODEL_WIDTH,
      HITNET_MIDDLEBURY_MODEL_HEIGHT
    );
    return ctx.getImageData(0, 0, HITNET_MIDDLEBURY_MODEL_WIDTH, HITNET_MIDDLEBURY_MODEL_HEIGHT)
      .data;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export const defaultBrowserMiddleburyFeedBuilder: MiddleburyStereoFeedBuilder = async ({
  im0Bytes,
  im1Bytes,
  srcWidth,
  srcHeight,
  cropRect,
}) => {
  const leftRgba = await decodePngTo320x240RgbaBrowser(im0Bytes, srcWidth, srcHeight, cropRect);
  const rightRgba = await decodePngTo320x240RgbaBrowser(im1Bytes, srcWidth, srcHeight, cropRect);
  const feedFloat32 = packRgbaStereoEyesToHitnetFeed(leftRgba, rightRgba);
  const feedSha256 = await sha256HexBytes(
    new Uint8Array(feedFloat32.buffer, feedFloat32.byteOffset, feedFloat32.byteLength)
  );
  return { feedFloat32, feedSha256 };
};

export async function loadAndVerifyMiddleburySceneAssets(
  sceneId: MiddleburyEval3SceneId,
  fetchImpl: typeof fetch = globalThis.fetch
): Promise<LoadedMiddleburySceneAssets> {
  const spec = MIDDLEBURY_EVAL3_Q_SCENES[sceneId];
  const fetchVerified = async (fileSpec: MiddleburyExpectedFileSpec): Promise<Uint8Array> => {
    const url = `${spec.baseFixturePath}/${fileSpec.relativePath}`;
    const res = await fetchImpl(url, { cache: 'no-store' });
    if (!res.ok) {
      throw new Error(`Failed to fetch Middlebury fixture ${url}: HTTP ${res.status}`);
    }
    const bytes = new Uint8Array(await res.arrayBuffer());
    if (bytes.byteLength !== fileSpec.byteLength) {
      throw new Error(
        `Middlebury fixture size mismatch for ${url}: got ${bytes.byteLength}, expected ${fileSpec.byteLength}`
      );
    }
    const sha = await sha256HexBytes(bytes);
    if (sha !== fileSpec.sha256) {
      throw new Error(
        `Middlebury fixture SHA-256 mismatch for ${url}: got ${sha}, expected ${fileSpec.sha256}`
      );
    }
    return bytes;
  };

  const [im0Bytes, im1Bytes, calibBytes, disp0Bytes, mask0Bytes, disp1Bytes, mask1Bytes] =
    await Promise.all([
      fetchVerified(spec.files.im0),
      fetchVerified(spec.files.im1),
      fetchVerified(spec.files.calib),
      fetchVerified(spec.files.disp0GT),
      fetchVerified(spec.files.mask0nocc),
      fetchVerified(spec.files.disp1GT),
      fetchVerified(spec.files.mask1nocc),
    ]);

  const calibText = new TextDecoder('utf-8').decode(calibBytes);
  const calib = parseMiddleburyCalibTxt(calibText);
  if (
    calib.width !== spec.expectedWidth ||
    calib.height !== spec.expectedHeight ||
    calib.ndisp !== spec.expectedNdisp ||
    calib.isint !== 0
  ) {
    throw new Error(
      `Unexpected calib.txt metadata for ${sceneId}: got ${JSON.stringify(calib)}`
    );
  }

  const disp0GtPfm = parsePfmSingleBand(disp0Bytes);
  const disp1GtPfm = parsePfmSingleBand(disp1Bytes);
  const mask0Nocc = await parsePngGrayscaleMaskBitExact(mask0Bytes);
  const mask1Nocc = await parsePngGrayscaleMaskBitExact(mask1Bytes);

  return {
    sceneId,
    spec,
    calibText,
    calib,
    im0Bytes,
    im1Bytes,
    disp0GtPfm,
    disp1GtPfm,
    mask0Nocc,
    mask1Nocc,
    fixtureSha256s: {
      im0: spec.files.im0.sha256,
      im1: spec.files.im1.sha256,
      calib: spec.files.calib.sha256,
      disp0GT: spec.files.disp0GT.sha256,
      mask0nocc: spec.files.mask0nocc.sha256,
      disp1GT: spec.files.disp1GT.sha256,
      mask1nocc: spec.files.mask1nocc.sha256,
    },
  };
}

function medianNumber(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function hasObservedGpuDispatch(prof: OrtVersionDiagnosticResponse | null): boolean {
  if (!prof || !prof.ok || !prof.gpuKernelProfileSummary) return false;
  const s = prof.gpuKernelProfileSummary;
  return (
    s.webgpuComputePassCount > 0 ||
    s.webgpuDispatchWorkgroupsCount > 0 ||
    s.ortProfilingEventCount > 0
  );
}

export function evaluateProductionGoNoGo(
  scenes: Record<MiddleburyEval3SceneId, MiddleburySceneBenchmarkReport>
): ProductionGoNoGoEvaluation {
  const allReports = [scenes.Motorcycle, scenes.Teddy];
  const gates: ProductionGoNoGoEvaluation['gates'] = [];

  const runtimeAllOk = allReports.every(
    (s) =>
      s.webgpu.fallbackReason === null &&
      s.webgpu.gpuValidationErrors.length === 0 &&
      s.webgpu.dispatchCheckProfilingOn?.dispatchObserved === true &&
      s.webgpu.foreignOrtResources.length === 0 &&
      s.wasm.foreignOrtResources.length === 0
  );
  gates.push({
    id: 'gate-runtime-dispatch',
    label: 'WebGPU Dispatch, Zero Fallback & Zero GPU Validation Errors (ORT 1.30.0 non-JSEP M1 opt=basic)',
    passed: runtimeAllOk,
    detail: allReports
      .map(
        (s) =>
          `${s.sceneId}: fallback=${String(s.webgpu.fallbackReason)}, gpuErr=${s.webgpu.gpuValidationErrors.length}, dispatch=${String(s.webgpu.dispatchCheckProfilingOn?.dispatchObserved)}`
      )
      .join(' | '),
  });

  const maxInvalidRate = Math.max(
    ...allReports.flatMap((s) => [
      s.webgpu.leftPrimary.nonocc.invalidPredRate,
      s.webgpu.rightSecondary.nonocc.invalidPredRate,
      s.wasm.leftPrimary.nonocc.invalidPredRate,
      s.wasm.rightSecondary.nonocc.invalidPredRate,
    ])
  );
  const zeroInvalidPreds = maxInvalidRate === 0;
  gates.push({
    id: 'gate-zero-invalid-preds',
    label: 'Zero Non-Finite or Negative Predictions in Fixed Nonocc GT',
    passed: zeroInvalidPreds,
    detail: `maxInvalidPredRate=${(maxInvalidRate * 100).toFixed(4)}%`,
  });

  // Explicit accuracy thresholds at 320x240 model coordinates across both eyes on Q fixtures
  const maxBad3 = Math.max(
    ...allReports.flatMap((s) => [
      s.webgpu.leftPrimary.nonocc.bad3Rate,
      s.webgpu.rightSecondary.nonocc.bad3Rate,
      s.wasm.leftPrimary.nonocc.bad3Rate,
      s.wasm.rightSecondary.nonocc.bad3Rate,
    ])
  );
  const maxEpe = Math.max(
    ...allReports.flatMap((s) => [
      s.webgpu.leftPrimary.nonocc.epeModelPx,
      s.webgpu.rightSecondary.nonocc.epeModelPx,
      s.wasm.leftPrimary.nonocc.epeModelPx,
      s.wasm.rightSecondary.nonocc.epeModelPx,
    ])
  );
  const accuracyPassed = Number.isFinite(maxEpe) && maxEpe <= 2.5 && maxBad3 <= 0.15;
  gates.push({
    id: 'gate-accuracy-nonocc-320x240',
    label: 'Both-Eye Nonocc Accuracy Gate (EPE <= 2.50 model-px, Bad3 <= 15.0%)',
    passed: accuracyPassed,
    detail: `worstEPE=${maxEpe.toFixed(4)} model-px, worstBad3=${(maxBad3 * 100).toFixed(2)}%`,
  });

  const worstParityMax = Math.max(
    ...allReports.flatMap((s) => [
      s.webgpuVsWasmParity.leftPrimaryFullFrame.maxAbsDiff,
      s.webgpuVsWasmParity.rightSecondaryFullFrame.maxAbsDiff,
    ])
  );
  const worstParityP99 = Math.max(
    ...allReports.flatMap((s) => [
      s.webgpuVsWasmParity.leftPrimaryFullFrame.p99AbsDiff,
      s.webgpuVsWasmParity.rightSecondaryFullFrame.p99AbsDiff,
    ])
  );
  const parityTight = worstParityMax < 1e-3 && worstParityP99 < 1e-4;
  gates.push({
    id: 'gate-gpu-wasm-parity',
    label: 'Unmodified M1 WebGPU vs WASM Cross-EP Parity (maxAbsDiff < 1e-3, p99AbsDiff < 1e-4)',
    passed: parityTight,
    detail: `worstFullFrameMax=${worstParityMax.toExponential(3)}, worstP99=${worstParityP99.toExponential(3)}`,
  });

  const overallStatus = !runtimeAllOk || !zeroInvalidPreds
    ? 'FAIL_RUNTIME_OR_DISPATCH'
    : !accuracyPassed || !parityTight
      ? 'FAIL_ACCURACY_OR_PARITY'
      : 'PASS';

  return {
    candidateScope: 'DEV-only unmodified canonical M1 non-JSEP ORT 1.30.0 (opt=basic)',
    gates,
    overallStatus,
    explicitLimits: [
      'Local DEV-only 320x240 direct stereo resize of MiddEval3 trainingQ scenes; NOT an official full-resolution Middlebury leaderboard submission.',
      'Motorcycle Q uses the 2014 imperfect-rectification edition (dyavg=0.23 px, dymax=0.379 px); Teddy Q uses the 2003 rectified edition (dyavg=0, dymax=0). Neither mixes perfect and imperfect editions.',
      'HITNet SceneFlow (hitnet_sceneflow_240x320.onnx) is evaluated zero-shot without Middlebury fine-tuning.',
      'Separate WebGPU profiling-ON pass verifies GPU compute dispatch events, but does not prove every operator executes on GPU without node-by-node EP assignment proof.',
      'Static /dev-fixtures/middlebury-eval3/ delivery avoids CORS/zip blockers on vision.middlebury.edu without adding any new proxy or dependency.',
    ],
  };
}

export async function runHitnetMiddleburyEval3Benchmark(params: {
  m1Bytes: Uint8Array;
  sceneIds?: MiddleburyEval3SceneId[];
  warmRuns?: number;
  disableCpuEpFallback?: boolean;
  isDev?: boolean;
  runWorker?: ResidualDiagnosticWorkerRunner;
  loadSceneAssets?: (sceneId: MiddleburyEval3SceneId) => Promise<LoadedMiddleburySceneAssets>;
  buildFeed?: MiddleburyStereoFeedBuilder;
  onLog?: (line: string) => void;
}): Promise<HitnetMiddleburyEval3BenchmarkReport> {
  assertDevFixtureEnabled(params.isDev);
  const log = params.onLog ?? (() => {});
  const runWorker = params.runWorker ?? createBrowserOrt130DiagnosticRunner();
  const loadScene = params.loadSceneAssets ?? ((id) => loadAndVerifyMiddleburySceneAssets(id));
  const buildFeed = params.buildFeed ?? defaultBrowserMiddleburyFeedBuilder;
  const sceneIds: MiddleburyEval3SceneId[] =
    params.sceneIds && params.sceneIds.length > 0 ? params.sceneIds : ['Motorcycle', 'Teddy'];
  const warmRuns = Math.max(2, Math.min(5, Math.trunc(params.warmRuns ?? 3)));
  const disableCpuEpFallback = Boolean(params.disableCpuEpFallback);

  const m1Sha = await sha256HexBytes(params.m1Bytes);
  if (
    params.m1Bytes.byteLength !== HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength ||
    m1Sha !== HITNET_M1_FULL_GRAPH_SHA256
  ) {
    throw new Error(
      `Exact canonical M1 verification failed: bytes=${params.m1Bytes.byteLength} (expected ${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength}), sha=${m1Sha} (expected ${HITNET_M1_FULL_GRAPH_SHA256})`
    );
  }

  const parsed = await parseM1ModelProtoWithRawSlices(params.m1Bytes);
  const n1OffProof = verifyCanonicalM1NoN1Replacements(parsed, params.m1Bytes, m1Sha, false);
  if (!n1OffProof.verified) {
    throw new Error(n1OffProof.failureReason ?? 'Canonical M1 N1-off guard failed');
  }

  const [pName, sName] = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;
  const asyncify = HITNET_ORT_130_FAMILY_PROFILES.asyncify;
  const perEyeSignMappingNote =
    'reference_output_disparity (+d px, x_R = x_L - d) mapped to Left eye disp0GT.pfm + mask0nocc.png; secondary_output_disparity (+d px, x_L = x_R + d) mapped to Right eye disp1GT.pfm + mask1nocc.png (verified against MiddEval3 computemask.cpp & HITNet D16/D32 analytic validation).';

  log(
    `=== [DEV-ONLY] Middlebury Eval3 Q Stereo Benchmark (Unmodified Canonical M1, non-JSEP ORT 1.30.0 opt=basic) ===\n` +
      `Runtime: onnxruntime-web@1.30.0 (${asyncify.entryFile} + ${asyncify.wasmFile}) | warmRuns=${warmRuns} | disableCpuEpFallback=${String(disableCpuEpFallback)}\n` +
      `Canonical Untapped M1: bytes=${params.m1Bytes.byteLength} sha256=${m1Sha} (N1=OFF, 0 replacements, NO #9 replacement)\n` +
      `Scope: Local public-fixture evaluation at 320x240 direct stereo resize (NOT an official full-res Middlebury leaderboard submission).\n` +
      `Eye Mapping: ${perEyeSignMappingNote}`
  );

  const assertSuccess = (
    label: string,
    raw: OrtVersionDiagnosticResponse,
    expectedProvider: 'webgpu' | 'wasm',
    expectedFeedSha: string
  ): OrtVersionDiagnosticSuccess => {
    if (!raw.ok) {
      const f = raw as OrtVersionDiagnosticFailure;
      throw new Error(`${label} FAILED at stage="${f.stage}": ${f.message}`);
    }
    if (
      raw.pinnedOrtVersion !== '1.30.0' ||
      raw.runtimeAssetFamily !== 'asyncify' ||
      raw.graphOptimizationLevel !== 'basic' ||
      raw.provider !== expectedProvider ||
      raw.foreignOrtResources.length > 0 ||
      raw.loadedOrtResources.some((u) => u.includes('.jsep.') || u.includes('ort.bundle.'))
    ) {
      throw new Error(
        `${label} runtime/opt guard violation: ver=${raw.pinnedOrtVersion}, fam=${raw.runtimeAssetFamily}, opt=${raw.graphOptimizationLevel}, ep=${raw.provider}, foreign=${JSON.stringify(raw.foreignOrtResources)}`
      );
    }
    if (
      expectedProvider === 'webgpu' &&
      (raw.configuredProviders.length !== 1 || raw.configuredProviders[0] !== 'webgpu')
    ) {
      throw new Error(
        `${label} unexpected WebGPU fallback/providers: ${JSON.stringify(raw.configuredProviders)}`
      );
    }
    if (raw.modelSha256 !== m1Sha || raw.modelByteLength !== params.m1Bytes.byteLength) {
      throw new Error(
        `${label} pre-session M1 SHA mismatch: got=${raw.modelSha256}, expected=${m1Sha}`
      );
    }
    if (raw.feedSha256 !== expectedFeedSha) {
      throw new Error(
        `${label} feed SHA mismatch: got=${raw.feedSha256}, expected=${expectedFeedSha}`
      );
    }
    return raw;
  };

  const sceneResults = {} as Record<MiddleburyEval3SceneId, MiddleburySceneBenchmarkReport>;

  for (const sceneId of sceneIds) {
    const assets = await loadScene(sceneId);
    const cropRect: CropRectPx = {
      x: 0,
      y: 0,
      width: assets.calib.width,
      height: assets.calib.height,
    };
    const leftGt = resampleMiddleburyEyeGtNearestGrid({
      eye: 'left',
      gtPfm: assets.disp0GtPfm,
      maskNocc: assets.mask0Nocc,
      calib: assets.calib,
      cropRect,
    });
    const rightGt = resampleMiddleburyEyeGtNearestGrid({
      eye: 'right',
      gtPfm: assets.disp1GtPfm,
      maskNocc: assets.mask1Nocc,
      calib: assets.calib,
      cropRect,
    });

    const { feedFloat32, feedSha256 } = await buildFeed({
      im0Bytes: assets.im0Bytes,
      im1Bytes: assets.im1Bytes,
      srcWidth: assets.calib.width,
      srcHeight: assets.calib.height,
      cropRect,
    });
    const feedByteLength = feedFloat32.byteLength;
    if (feedByteLength !== HITNET_MIDDLEBURY_PIXEL_COUNT * 6 * 4) {
      throw new Error(`Unexpected Middlebury feed byteLength ${feedByteLength}`);
    }

    log(
      `\n--- Scene: ${sceneId} (${assets.spec.edition}) | native=${assets.calib.width}x${assets.calib.height} -> 320x240 | scaleX=${leftGt.scaleX.toFixed(6)} | ndisp=${assets.calib.ndisp} (${leftGt.ndispModelPx.toFixed(2)} model-px) ---\n` +
        `  Caveat: ${assets.spec.rectificationCaveat}\n` +
        `  Feed SHA-256: ${feedSha256} (${feedByteLength} B)\n` +
        `  Pre-prediction GT coverage: Left nonocc=${leftGt.coverage.nonoccValidGtCount}/76800 (${(leftGt.coverage.nonoccValidGtRate * 100).toFixed(2)}%, allValid=${leftGt.coverage.allValidGtCount}, invalidOr<=0=${leftGt.coverage.gtInvalidOrNonPositiveCount}, >ndisp=${leftGt.coverage.gtOverNdispCount}, cropLost=${leftGt.coverage.cropLostCorrespondenceCount}) | ` +
        `Right nonocc=${rightGt.coverage.nonoccValidGtCount}/76800 (${(rightGt.coverage.nonoccValidGtRate * 100).toFixed(2)}%, allValid=${rightGt.coverage.allValidGtCount}, invalidOr<=0=${rightGt.coverage.gtInvalidOrNonPositiveCount}, >ndisp=${rightGt.coverage.gtOverNdispCount}, cropLost=${rightGt.coverage.cropLostCorrespondenceCount})`
    );

    const gpuTimingRaw = await runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: params.m1Bytes.slice().buffer,
      feedBuffer: feedFloat32.slice().buffer,
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      provider: 'webgpu',
      graphOptimizationLevel: 'basic',
      warmRuns,
      enableWebGpuProfiling: false,
      disableCpuEpFallback,
      captureGpuValidationErrors: true,
    });
    const wasmTimingRaw = await runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: params.m1Bytes.slice().buffer,
      feedBuffer: feedFloat32.slice().buffer,
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      provider: 'wasm',
      graphOptimizationLevel: 'basic',
      warmRuns,
      enableWebGpuProfiling: false,
      disableCpuEpFallback: false,
      captureGpuValidationErrors: false,
    });

    const gpuOk = assertSuccess(`${sceneId} WebGPU`, gpuTimingRaw, 'webgpu', feedSha256);
    const wasmOk = assertSuccess(`${sceneId} WASM`, wasmTimingRaw, 'wasm', feedSha256);
    const gpuVal = validateHitnetFinalOutputs(gpuOk.outputs);
    const wasmVal = validateHitnetFinalOutputs(wasmOk.outputs);
    if (!gpuVal.ok || !wasmVal.ok) {
      throw new Error(
        `${sceneId} output validation failed: gpu=${String(gpuVal.error)}, wasm=${String(wasmVal.error)}`
      );
    }

    // Separate profiling-ON pass strictly to verify actual WebGPU kernel dispatch (excluded from timings)
    const gpuProfRaw = await runWorker({
      expectedOrtVersion: '1.30.0',
      runtimeAssetFamily: 'asyncify',
      modelBuffer: params.m1Bytes.slice().buffer,
      feedBuffer: feedFloat32.slice().buffer,
      feedName: 'input',
      feedDims: [1, 6, 240, 320],
      provider: 'webgpu',
      graphOptimizationLevel: 'basic',
      warmRuns: 0,
      enableWebGpuProfiling: true,
      disableCpuEpFallback,
      captureGpuValidationErrors: true,
    });
    const dispatchObserved = hasObservedGpuDispatch(gpuProfRaw);

    const buildArm = (
      provider: 'webgpu' | 'wasm',
      okRes: OrtVersionDiagnosticSuccess,
      valByName: Record<string, { data: Float32Array }>,
      dispatchCheck: MiddleburyArmExecutionResult['dispatchCheckProfilingOn']
    ): MiddleburyArmExecutionResult => {
      const pData = valByName[pName].data;
      const sData = valByName[sName].data;
      return {
        provider,
        graphOptimizationLevel: 'basic',
        modelSha256: okRes.modelSha256!,
        modelByteLength: okRes.modelByteLength!,
        feedSha256: okRes.feedSha256!,
        configuredProviders: okRes.configuredProviders,
        fallbackReason:
          okRes.configuredProviders.length === 1 && okRes.configuredProviders[0] === provider
            ? null
            : `unexpected-providers:${okRes.configuredProviders.join(',')}`,
        loadedOrtResources: okRes.loadedOrtResources,
        foreignOrtResources: okRes.foreignOrtResources,
        workerGpuAdapter: okRes.workerGpuAdapter ?? null,
        gpuValidationErrors: okRes.gpuValidationErrors ?? [],
        profilingOffTimings: {
          createMs: okRes.createMs,
          coldRunMs: okRes.coldRunMs,
          warmRunMs: [...okRes.warmRunMs],
          warmMedianMs: medianNumber(okRes.warmRunMs),
        },
        dispatchCheckProfilingOn: dispatchCheck,
        leftPrimary: {
          eye: 'left',
          outputTensorName: 'reference_output_disparity',
          nonocc: evaluateMiddleburyDisparityMask(pData, leftGt, 'nonocc'),
          allValidGt: evaluateMiddleburyDisparityMask(pData, leftGt, 'allValidGt'),
          eyeSignSanity: evaluateEyeSignSanity(pData, leftGt, rightGt),
        },
        rightSecondary: {
          eye: 'right',
          outputTensorName: 'secondary_output_disparity',
          nonocc: evaluateMiddleburyDisparityMask(sData, rightGt, 'nonocc'),
          allValidGt: evaluateMiddleburyDisparityMask(sData, rightGt, 'allValidGt'),
          eyeSignSanity: evaluateEyeSignSanity(sData, rightGt, leftGt),
        },
      };
    };

    const webgpuArm = buildArm('webgpu', gpuOk, gpuVal.byName, {
      executed: true,
      dispatchObserved,
      summary: gpuProfRaw.ok ? gpuProfRaw.gpuKernelProfileSummary ?? null : null,
    });
    const wasmArm = buildArm('wasm', wasmOk, wasmVal.byName, null);

    const leftPrimaryFullFrame = summarizeRawTensorAndParity(
      gpuVal.byName[pName].data,
      wasmVal.byName[pName].data
    );
    const leftPrimaryNonocc = summarizeMaskedTensorParity(
      gpuVal.byName[pName].data,
      wasmVal.byName[pName].data,
      leftGt.nonoccMask
    );
    const rightSecondaryFullFrame = summarizeRawTensorAndParity(
      gpuVal.byName[sName].data,
      wasmVal.byName[sName].data
    );
    const rightSecondaryNonocc = summarizeMaskedTensorParity(
      gpuVal.byName[sName].data,
      wasmVal.byName[sName].data,
      rightGt.nonoccMask
    );

    const fmtEye = (tag: string, m: MiddleburyMaskMetrics, s: EyeSignSanityCheck) =>
      `${tag}(n=${m.evaluatedCount}): EPE=${m.epeModelPx.toFixed(4)} model-px (${m.epeNativeQPx.toFixed(3)} Q-px) p95=${m.p95AbsErrorModelPx.toFixed(4)}px Bad1=${(m.bad1Rate * 100).toFixed(2)}% Bad3=${(m.bad3Rate * 100).toFixed(2)}% invalidPred=${m.invalidPredCount} | signCheck(+GT=${s.correctEyePosSignEpeModelPx.toFixed(3)} < -GT=${s.negatedGtEpeModelPx.toFixed(3)}, oppEye=${s.swappedEyePosSignEpeModelPx.toFixed(3)})`;

    log(
      `  [WebGPU OFF-profile] create=${webgpuArm.profilingOffTimings.createMs.toFixed(1)}ms cold=${webgpuArm.profilingOffTimings.coldRunMs.toFixed(1)}ms warm=[${webgpuArm.profilingOffTimings.warmRunMs.map((v) => v.toFixed(1)).join(', ')}]ms (med=${webgpuArm.profilingOffTimings.warmMedianMs?.toFixed(1) ?? 'n/a'}ms) | [Separate Profiling-ON] dispatchObserved=${dispatchObserved}\n` +
        `    Left  nonocc: ${fmtEye('ref_disp', webgpuArm.leftPrimary.nonocc, webgpuArm.leftPrimary.eyeSignSanity)}\n` +
        `    Right nonocc: ${fmtEye('sec_disp', webgpuArm.rightSecondary.nonocc, webgpuArm.rightSecondary.eyeSignSanity)}\n` +
        `  [WASM   OFF-profile] create=${wasmArm.profilingOffTimings.createMs.toFixed(1)}ms cold=${wasmArm.profilingOffTimings.coldRunMs.toFixed(1)}ms warm=[${wasmArm.profilingOffTimings.warmRunMs.map((v) => v.toFixed(1)).join(', ')}]ms (med=${wasmArm.profilingOffTimings.warmMedianMs?.toFixed(1) ?? 'n/a'}ms)\n` +
        `    Left  nonocc: ${fmtEye('ref_disp', wasmArm.leftPrimary.nonocc, wasmArm.leftPrimary.eyeSignSanity)}\n` +
        `    Right nonocc: ${fmtEye('sec_disp', wasmArm.rightSecondary.nonocc, wasmArm.rightSecondary.eyeSignSanity)}\n` +
        `  [GPU vs WASM Parity] Left full(max=${leftPrimaryFullFrame.maxAbsDiff.toExponential(3)}, p99=${leftPrimaryFullFrame.p99AbsDiff.toExponential(3)}, mean=${leftPrimaryFullFrame.meanAbsDiff.toExponential(3)}) nonocc(max=${leftPrimaryNonocc.maxAbsDiff.toExponential(3)}, mean=${leftPrimaryNonocc.meanAbsDiff.toExponential(3)}) | ` +
        `Right full(max=${rightSecondaryFullFrame.maxAbsDiff.toExponential(3)}, p99=${rightSecondaryFullFrame.p99AbsDiff.toExponential(3)}, mean=${rightSecondaryFullFrame.meanAbsDiff.toExponential(3)}) nonocc(max=${rightSecondaryNonocc.maxAbsDiff.toExponential(3)}, mean=${rightSecondaryNonocc.meanAbsDiff.toExponential(3)})`
    );

    sceneResults[sceneId] = {
      sceneId,
      edition: assets.spec.edition,
      rectificationCaveat: assets.spec.rectificationCaveat,
      citation: assets.spec.citation,
      calib: assets.calib,
      cropRect,
      scaleX: leftGt.scaleX,
      scaleY: leftGt.scaleY,
      feedSha256,
      feedByteLength,
      fixtureSha256s: assets.fixtureSha256s,
      leftGtCoverage: leftGt.coverage,
      rightGtCoverage: rightGt.coverage,
      webgpu: webgpuArm,
      wasm: wasmArm,
      webgpuVsWasmParity: {
        leftPrimaryFullFrame,
        leftPrimaryNonocc,
        rightSecondaryFullFrame,
        rightSecondaryNonocc,
      },
    };
  }

  const productionGoNoGoEvaluation =
    sceneResults.Motorcycle && sceneResults.Teddy
      ? evaluateProductionGoNoGo(sceneResults)
      : {
          candidateScope: 'DEV-only unmodified canonical M1 non-JSEP ORT 1.30.0 (opt=basic)' as const,
          gates: [],
          overallStatus: 'FAIL_ACCURACY_OR_PARITY' as const,
          explicitLimits: ['Incomplete scene set executed; run both Motorcycle and Teddy for full gate summary.'],
        };

  log(
    `\n=== Production Go/No-Go Evaluation (${productionGoNoGoEvaluation.overallStatus}) ===\n` +
      productionGoNoGoEvaluation.gates
        .map((g) => `  [${g.passed ? 'PASS' : 'FAIL'}] ${g.label}: ${g.detail}`)
        .join('\n') +
      `\nExplicit Limits:\n` +
      productionGoNoGoEvaluation.explicitLimits.map((l) => `  - ${l}`).join('\n')
  );

  return {
    ok: true,
    benchmarkMode: 'local-public-fixture-320x240-not-official-leaderboard',
    pinnedOrtVersion: '1.30.0',
    runtimeAssetFamily: 'asyncify',
    graphOptimizationLevel: 'basic',
    warmRuns,
    m1ByteLength: params.m1Bytes.byteLength,
    m1Sha256: m1Sha,
    n1OffProof,
    perEyeSignMappingNote,
    scenes: sceneResults,
    productionGoNoGoEvaluation,
  };
}
