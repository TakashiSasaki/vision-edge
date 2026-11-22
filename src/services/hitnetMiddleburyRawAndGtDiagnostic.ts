import {
  DEFAULT_HITNET_DEV_FIXTURE_PROFILE,
  assertDevFixtureEnabled,
  HITNET_DEV_NON_JSEP_FIXTURE_SPEC,
} from './stereoDepthWebGpuDevFixture';
import {
  STEREO_CORS_PROXY_PREFIX,
  STEREO_CORS_PROXY_UPSTREAMS,
} from './browserModelCache';
import {
  prepareHitnetM1FullGraphBytes,
  summarizeRawTensorAndParity,
  type RawTensorParityStats,
} from './hitnetGatherNdDiagnostic';
import {
  HITNET_MIDDLEBURY_PIXEL_COUNT,
  MIDDLEBURY_EVAL3_Q_SCENES,
  loadAndVerifyMiddleburySceneAssets,
  type LoadedMiddleburySceneAssets,
  type MiddleburyCalibData,
  type MiddleburyEval3SceneId,
  type MiddleburyEyeId,
  type ParsedPfmGrid,
  type ParsedPngMaskGrid,
} from './hitnetMiddleburyEval3Benchmark';
import {
  buildMiddleburyGeometryTransforms,
  buildSharedPhysicalSampleSetForEye,
  defaultBrowserFactorialFeedPairBuilder,
  evaluateFactorialArmEye,
  packRgbaStereoEyesWithChannelOrder,
  type FactorialChannelOrder,
  type FactorialEyeEvaluation,
  type FactorialFeedPairBuilder,
  type GeometryTransformSpec,
  type SharedSceneEyeSampleSet,
} from './hitnetMiddleburyFactorialDiagnosis';
import {
  type OrtVersionDiagnosticFailure,
  type OrtVersionDiagnosticResponse,
  type OrtVersionDiagnosticSuccess,
} from './hitnetOrtComparisonHelper';
import {
  verifyCanonicalM1NoN1Replacements,
  type N1OffVerificationProof,
} from './hitnetPostConv7TapReplay';
import {
  createBrowserOrt130DiagnosticRunner,
  sha256HexBytes,
  validateHitnetFinalOutputs,
  type ResidualDiagnosticWorkerRunner,
} from './hitnetResidualLocalization';
import { parseM1ModelProtoWithRawSlices } from './hitnetRgbStemFirstNonZero';
import {
  HITNET_M1_FULL_GRAPH_SHA256,
  HITNET_RAW_MODEL_SHA256,
} from './stereoDepthWorkerCore';

export const EXPECTED_RAW_HITNET_RELEASE_URL =
  'https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_sceneflow_240x320.onnx';
export const EXPECTED_RAW_HITNET_RELEASE_PAGE =
  'https://github.com/nburrus/stereodemo/releases/tag/v0.1-hitnet';
export const EXPECTED_RAW_HITNET_PROXY_PATH =
  `${STEREO_CORS_PROXY_PREFIX}hitnet_sceneflow_240x320.onnx`;
export const EXPECTED_RAW_HITNET_BYTE_LENGTH = 8829487;

export interface RawHitnetProvenanceCheck {
  verified: boolean;
  failureReason: string | null;
  configuredProfileUpstreamUrl: string;
  configuredProxyPrefix: string;
  configuredProxyUpstreamUrl: string | null;
  fetchedUrl: string;
  httpStatus: number | null;
  proxyCacheHeader: string | null;
  proxyUpstreamHostHeader: string | null;
  rawByteLength: number | null;
  rawSha256: string | null;
  expectedRawSha256: string;
  expectedRawByteLength: number;
  contractMatches: boolean;
  graphInputs: Array<{ name: string; dims: number[]; elemType: number | null }>;
  graphOutputs: Array<{ name: string; dims: number[]; elemType: number | null }>;
  rawNodeCount: number | null;
}

export interface DecodedRgbImageGrid {
  width: number;
  height: number;
  colorType: number;
  /** RGB top-to-bottom uint8 (length = width * height * 3) */
  rgbTopToBottom: Uint8Array;
}

export interface NativeRoiBoxPx {
  roiId: string;
  label: string;
  xStart: number;
  xEndExclusive: number;
  yStart: number;
  yEndExclusive: number;
  note: string;
}

export interface RawVsM1ArmRunSummary {
  modelVariant: 'raw_release' | 'canonical_m1';
  channelOrder: FactorialChannelOrder;
  feedSha256: string;
  feedTensorName: string;
  feedTensorShape: number[];
  feedTensorDataType: string;
  modelByteLength: number;
  modelSha256: string;
  sessionInputNames: string[];
  sessionOutputNames: string[];
  outputContracts: Array<{ name: string; dims: number[]; dataType: string }>;
  configuredProviders: readonly string[];
  loadedOrtResources: string[];
  foreignOrtResources: string[];
  wasmAssetShas: Array<{ url: string; sha256: string | null; matchesLocalDist: boolean | null }>;
  createMs: number;
  coldRunMs: number;
  warmRunMs: number[];
  leftPrimary: FactorialEyeEvaluation;
  rightSecondary: FactorialEyeEvaluation;
}

export interface RawVsM1FeedComparison {
  channelOrder: FactorialChannelOrder;
  feedSha256: string;
  rawArm: RawVsM1ArmRunSummary;
  m1Arm: RawVsM1ArmRunSummary;
  fullTensorParityRawVsM1: {
    reference_output_disparity: RawTensorParityStats;
    secondary_output_disparity: RawTensorParityStats;
    bitIdenticalBothOutputs: boolean;
  };
  accuracyDeltaRawMinusM1: {
    leftNonoccEpeDeltaPx: number;
    leftNonoccP95DeltaPx: number;
    leftNonoccBad1DeltaPctPts: number;
    leftNonoccBad3DeltaPctPts: number;
    leftNonoccInvalidDeltaCount: number;
    rightNonoccEpeDeltaPx: number;
    rightNonoccP95DeltaPx: number;
    rightNonoccBad1DeltaPctPts: number;
    rightNonoccBad3DeltaPctPts: number;
    rightNonoccInvalidDeltaCount: number;
  };
}

export interface EyeSourceGtRoiConsistency {
  roiId: string;
  label: string;
  eye: MiddleburyEyeId;
  note: string;
  evaluatedRecipCount: number;
  oppositeNonoccHitRate: number;
  oppositeOccOrInvalidHitCount: number;
  gtNativeMinPx: number;
  gtNativeMaxPx: number;
  /** |d_A(x,y) - d_B(round(x_B), y)| where x_B = x - d_A (left) or x + d_A (right) */
  nearestRecipAbsMeanPx: number;
  nearestRecipAbsP95Px: number;
  nearestRecipAbsMaxPx: number;
  nearestRecipLe1PxRate: number;
  /** Valid-only bilinear interpolation (strictly skips if any 4 neighbors invalid/non-positive) */
  bilinearRecipValidCount: number;
  bilinearRecipAbsMeanPx: number;
  bilinearRecipAbsP95Px: number;
  signedRecipBiasPx: number;
  /** Photometric + exposure-invariant gradient/patch stats on valid 5x5 patch points */
  evaluatedPatchCount: number;
  meanSrcLum01: number;
  meanTargetLum01: number;
  signedExposureOffsetLum01: number;
  absLumResidual01: number;
  absRgbResidual01: number;
  meanGradDiff01: number;
  p95GradDiff01: number;
  relGradDiffToSrcGrad: number;
  /** Exposure-invariant zero-mean 5x5 SAD per pixel & ZNCC (textured patches std >= 0.01) */
  zsad5x5MeanPerPx01: number;
  zncc5x5TexturedCount: number;
  zncc5x5TexturedMean: number;
  /** Unit & sign sanity check directly from source-native images + GT */
  signAndUnitCheck: {
    positive1xAbsLum01: number;
    negatedSignAbsLum01: number;
    halfScale05xAbsLum01: number;
    doubleScale2xAbsLum01: number;
    positive1xZncc5x5: number;
    negatedSignZncc5x5: number;
    halfScale05xZncc5x5: number;
    doubleScale2xZncc5x5: number;
    confirmedUnitAndSign: boolean;
  };
}

export interface SignedYMapAvailabilityStatus {
  checkedUrls: string[];
  available: boolean;
  statusNote: string;
  calibDyavgPx: number;
  calibDymaxPx: number;
}

export interface SceneSourceGtConsistencyReport {
  sceneId: MiddleburyEval3SceneId;
  edition: string;
  calib: MiddleburyCalibData;
  yGtMapStatus: SignedYMapAvailabilityStatus;
  rois: EyeSourceGtRoiConsistency[];
}

export interface SceneRawVsM1ComparisonReport {
  sceneId: MiddleburyEval3SceneId;
  edition: string;
  g0Transform: GeometryTransformSpec;
  leftSupport: SharedSceneEyeSampleSet['support'];
  rightSupport: SharedSceneEyeSampleSet['support'];
  byFeed: Record<FactorialChannelOrder, RawVsM1FeedComparison>;
}

export interface HitnetMiddleburyRawAndGtReport {
  ok: true;
  pinnedOrtVersion: '1.30.0';
  runtimeAssetFamily: 'asyncify';
  graphOptimizationLevel: 'basic';
  rawProvenance: RawHitnetProvenanceCheck;
  rawComparisonExecuted: boolean;
  rawComparisonAbortedReason: string | null;
  m1ByteLength: number | null;
  m1Sha256: string | null;
  n1OffProof: N1OffVerificationProof | null;
  rawVsM1ByScene: Partial<Record<MiddleburyEval3SceneId, SceneRawVsM1ComparisonReport>>;
  sourceGtConsistencyByScene: Record<MiddleburyEval3SceneId, SceneSourceGtConsistencyReport>;
  diagnosisSummary: {
    rawVsM1Verdict: string;
    sourceGtConsistencyVerdict: string;
    smallestNextExperiment: string;
  };
}

export const PREDECLARED_NATIVE_ROIS: Readonly<
  Record<MiddleburyEval3SceneId, readonly NativeRoiBoxPx[]>
> = {
  Motorcycle: [
    {
      roiId: 'motorcycle_full',
      label: 'Motorcycle Full Frame',
      xStart: 0,
      xEndExclusive: 741,
      yStart: 0,
      yEndExclusive: 497,
      note: 'Full native 741x497 Q scene.',
    },
    {
      roiId: 'motorcycle_upper_left_trouble',
      label: 'Motorcycle Upper-Left Trouble ROI',
      xStart: 0,
      xEndExclusive: 247,
      yStart: 0,
      yEndExclusive: 166,
      note: 'Predeclared upper-left trouble region (native [0..247)x[0..166), corresponding to model [0..107)x[0..80)).',
    },
    {
      roiId: 'motorcycle_control_center',
      label: 'Motorcycle Fixed Control Center ROI',
      xStart: 247,
      xEndExclusive: 555,
      yStart: 200,
      yEndExclusive: 420,
      note: 'Fixed foreground control ROI (native [247..555)x[200..420), textured motorcycle frame/engine).',
    },
  ],
  Teddy: [
    {
      roiId: 'teddy_full',
      label: 'Teddy Full Frame (Rectified Control)',
      xStart: 0,
      xEndExclusive: 450,
      yStart: 0,
      yEndExclusive: 375,
      note: 'Full native 450x375 rectified Q scene (dyavg=0, dymax=0).',
    },
  ],
};

async function zlibDecompressDeflate(compressed: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream !== 'function') {
    throw new Error('DecompressionStream("deflate") unavailable for bit-exact PNG RGB decoding');
  }
  const copy = new Uint8Array(compressed.byteLength);
  copy.set(compressed);
  const stream = new Blob([copy]).stream().pipeThrough(new DecompressionStream('deflate'));
  const buf = await new Response(stream).arrayBuffer();
  return new Uint8Array(buf);
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

export async function decodePngRgbTopToBottomBitExact(
  bytes: Uint8Array
): Promise<DecodedRgbImageGrid> {
  const PNG_SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (bytes.byteLength < 8 || PNG_SIG.some((v, i) => bytes[i] !== v)) {
    throw new Error('Invalid PNG signature');
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
          `Unsupported PNG encoding: bitDepth=${bitDepth}, colorType=${colorType}, interlace=${interlace}`
        );
      }
      if (colorType !== 0 && colorType !== 2 && colorType !== 6) {
        throw new Error(`Unsupported PNG colorType=${colorType} (expected 0, 2, or 6)`);
      }
    } else if (type === 'IDAT') {
      idatChunks.push(bytes.subarray(dataStart, dataEnd));
    } else if (type === 'IEND') {
      break;
    }
    pos = dataEnd + 4;
  }

  if (width <= 0 || height <= 0 || idatChunks.length === 0) {
    throw new Error('Incomplete PNG IHDR/IDAT');
  }

  const totalIdat = idatChunks.reduce((acc, c) => acc + c.byteLength, 0);
  const compressed = new Uint8Array(totalIdat);
  let off = 0;
  for (const c of idatChunks) {
    compressed.set(c, off);
    off += c.byteLength;
  }

  const raw = await zlibDecompressDeflate(compressed);
  const bpp = colorType === 0 ? 1 : colorType === 2 ? 3 : 4;
  const stride = width * bpp;
  const expectedRaw = height * (stride + 1);
  if (raw.byteLength !== expectedRaw) {
    throw new Error(`PNG decompressed length mismatch: got ${raw.byteLength}, expected ${expectedRaw}`);
  }

  const recon = new Uint8Array(height * stride);
  const rgbTopToBottom = new Uint8Array(width * height * 3);

  for (let y = 0; y < height; y++) {
    const srcRowStart = y * (stride + 1);
    const filterType = raw[srcRowStart];
    const rowOutStart = y * stride;
    for (let i = 0; i < stride; i++) {
      const xVal = raw[srcRowStart + 1 + i];
      const a = i >= bpp ? recon[rowOutStart + i - bpp] : 0;
      const b = y > 0 ? recon[(y - 1) * stride + i] : 0;
      const c = y > 0 && i >= bpp ? recon[(y - 1) * stride + i - bpp] : 0;
      let pxVal: number;
      if (filterType === 0) pxVal = xVal;
      else if (filterType === 1) pxVal = (xVal + a) & 0xff;
      else if (filterType === 2) pxVal = (xVal + b) & 0xff;
      else if (filterType === 3) pxVal = (xVal + ((a + b) >> 1)) & 0xff;
      else if (filterType === 4) pxVal = (xVal + paethPredictor(a, b, c)) & 0xff;
      else throw new Error(`Unsupported PNG scanline filter ${filterType}`);
      recon[rowOutStart + i] = pxVal;
    }

    const dstRgbRow = y * width * 3;
    if (bpp === 3) {
      rgbTopToBottom.set(recon.subarray(rowOutStart, rowOutStart + width * 3), dstRgbRow);
    } else if (bpp === 4) {
      for (let x = 0; x < width; x++) {
        const si = rowOutStart + x * 4;
        const di = dstRgbRow + x * 3;
        rgbTopToBottom[di] = recon[si];
        rgbTopToBottom[di + 1] = recon[si + 1];
        rgbTopToBottom[di + 2] = recon[si + 2];
      }
    } else {
      for (let x = 0; x < width; x++) {
        const g = recon[rowOutStart + x];
        const di = dstRgbRow + x * 3;
        rgbTopToBottom[di] = g;
        rgbTopToBottom[di + 1] = g;
        rgbTopToBottom[di + 2] = g;
      }
    }
  }

  return { width, height, colorType, rgbTopToBottom };
}

export async function verifyRawHitnetProvenanceAndContract(
  fetchImpl: typeof fetch = globalThis.fetch
): Promise<{ check: RawHitnetProvenanceCheck; rawBytes: Uint8Array | null }> {
  const configuredProfileUpstreamUrl = DEFAULT_HITNET_DEV_FIXTURE_PROFILE.upstreamUrl ?? '';
  const configuredProxyUpstreamUrl =
    STEREO_CORS_PROXY_UPSTREAMS['hitnet_sceneflow_240x320.onnx'] ?? null;

  const baseFail = (
    failureReason: string,
    partial: Partial<RawHitnetProvenanceCheck> = {}
  ): { check: RawHitnetProvenanceCheck; rawBytes: null } => ({
    rawBytes: null,
    check: {
      verified: false,
      failureReason,
      configuredProfileUpstreamUrl,
      configuredProxyPrefix: STEREO_CORS_PROXY_PREFIX,
      configuredProxyUpstreamUrl,
      fetchedUrl: EXPECTED_RAW_HITNET_PROXY_PATH,
      httpStatus: null,
      proxyCacheHeader: null,
      proxyUpstreamHostHeader: null,
      rawByteLength: null,
      rawSha256: null,
      expectedRawSha256: HITNET_RAW_MODEL_SHA256,
      expectedRawByteLength: EXPECTED_RAW_HITNET_BYTE_LENGTH,
      contractMatches: false,
      graphInputs: [],
      graphOutputs: [],
      rawNodeCount: null,
      ...partial,
    },
  });

  if (configuredProfileUpstreamUrl !== EXPECTED_RAW_HITNET_RELEASE_PAGE) {
    return baseFail(
      `Configured HITNet profile upstreamUrl "${configuredProfileUpstreamUrl}" != expected "${EXPECTED_RAW_HITNET_RELEASE_PAGE}"`
    );
  }
  if (configuredProxyUpstreamUrl !== EXPECTED_RAW_HITNET_RELEASE_URL) {
    return baseFail(
      `Configured STEREO_CORS_PROXY_UPSTREAMS['hitnet_sceneflow_240x320.onnx']="${String(configuredProxyUpstreamUrl)}" != expected "${EXPECTED_RAW_HITNET_RELEASE_URL}"`
    );
  }

  let resp: Response;
  try {
    resp = await fetchImpl(EXPECTED_RAW_HITNET_PROXY_PATH, {
      method: 'GET',
      cache: 'no-store',
    });
  } catch (err) {
    return baseFail(
      `Proxy fetch failed for ${EXPECTED_RAW_HITNET_PROXY_PATH}: ${err instanceof Error ? err.message : String(err)}`
    );
  }

  const httpStatus = resp.status;
  const proxyCacheHeader = resp.headers.get('x-proxy-cache');
  const proxyUpstreamHostHeader = resp.headers.get('x-upstream-host');

  if (!resp.ok) {
    return baseFail(`Proxy returned HTTP ${httpStatus} for ${EXPECTED_RAW_HITNET_PROXY_PATH}`, {
      httpStatus,
      proxyCacheHeader,
      proxyUpstreamHostHeader,
    });
  }

  const contentType = resp.headers.get('content-type')?.toLowerCase() ?? '';
  if (contentType.startsWith('text/html')) {
    return baseFail(`Proxy returned unexpected text/html instead of ONNX binary`, {
      httpStatus,
      proxyCacheHeader,
      proxyUpstreamHostHeader,
    });
  }

  const rawBytes = new Uint8Array(await resp.arrayBuffer());
  const rawSha256 = await sha256HexBytes(rawBytes);

  if (
    rawBytes.byteLength !== EXPECTED_RAW_HITNET_BYTE_LENGTH ||
    rawSha256 !== HITNET_RAW_MODEL_SHA256
  ) {
    return baseFail(
      `Raw HITNet release binary mismatch: bytes=${rawBytes.byteLength} (expected ${EXPECTED_RAW_HITNET_BYTE_LENGTH}), sha=${rawSha256} (expected ${HITNET_RAW_MODEL_SHA256})`,
      {
        httpStatus,
        proxyCacheHeader,
        proxyUpstreamHostHeader,
        rawByteLength: rawBytes.byteLength,
        rawSha256,
      }
    );
  }

  let parsed: Awaited<ReturnType<typeof parseM1ModelProtoWithRawSlices>>;
  try {
    parsed = await parseM1ModelProtoWithRawSlices(rawBytes);
  } catch (err) {
    return baseFail(
      `Raw HITNet protobuf parse failed: ${err instanceof Error ? err.message : String(err)}`,
      {
        httpStatus,
        proxyCacheHeader,
        proxyUpstreamHostHeader,
        rawByteLength: rawBytes.byteLength,
        rawSha256,
      }
    );
  }

  const graphInputs = Array.from(parsed.graphInputNames).map((name) => {
    const meta = parsed.staticMeta.tensorMeta[name];
    return {
      name,
      dims: meta ? [...meta.dims] : [],
      elemType: meta ? meta.elemType : null,
    };
  });
  const graphOutputs = Array.from(parsed.graphOutputNames).map((name) => {
    const meta = parsed.staticMeta.tensorMeta[name];
    return {
      name,
      dims: meta ? [...meta.dims] : [],
      elemType: meta ? meta.elemType : null,
    };
  });

  const inOk =
    graphInputs.length === 1 &&
    graphInputs[0].name === 'input' &&
    graphInputs[0].elemType === 1 &&
    JSON.stringify(graphInputs[0].dims) === JSON.stringify([1, 6, 240, 320]);

  const reqOuts = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;
  const outOk =
    graphOutputs.length === 2 &&
    reqOuts.every((reqName) => {
      const found = graphOutputs.find((o) => o.name === reqName);
      return (
        Boolean(found) &&
        found!.elemType === 1 &&
        JSON.stringify(found!.dims) === JSON.stringify([1, 240, 320, 1])
      );
    });

  const contractMatches = inOk && outOk;
  if (!contractMatches) {
    return baseFail(
      `Raw ONNX IO contract mismatch: inputs=${JSON.stringify(graphInputs)}, outputs=${JSON.stringify(graphOutputs)}`,
      {
        httpStatus,
        proxyCacheHeader,
        proxyUpstreamHostHeader,
        rawByteLength: rawBytes.byteLength,
        rawSha256,
        contractMatches: false,
        graphInputs,
        graphOutputs,
        rawNodeCount: parsed.nodes.length,
      }
    );
  }

  return {
    rawBytes,
    check: {
      verified: true,
      failureReason: null,
      configuredProfileUpstreamUrl,
      configuredProxyPrefix: STEREO_CORS_PROXY_PREFIX,
      configuredProxyUpstreamUrl,
      fetchedUrl: EXPECTED_RAW_HITNET_PROXY_PATH,
      httpStatus,
      proxyCacheHeader,
      proxyUpstreamHostHeader,
      rawByteLength: rawBytes.byteLength,
      rawSha256,
      expectedRawSha256: HITNET_RAW_MODEL_SHA256,
      expectedRawByteLength: EXPECTED_RAW_HITNET_BYTE_LENGTH,
      contractMatches: true,
      graphInputs,
      graphOutputs,
      rawNodeCount: parsed.nodes.length,
    },
  };
}

function rgbAtLum01(img: DecodedRgbImageGrid, x: number, y: number): number {
  const o = (y * img.width + x) * 3;
  return (0.299 * img.rgbTopToBottom[o] + 0.587 * img.rgbTopToBottom[o + 1] + 0.114 * img.rgbTopToBottom[o + 2]) / 255;
}

function rgbTuple01(img: DecodedRgbImageGrid, x: number, y: number): [number, number, number] {
  const o = (y * img.width + x) * 3;
  return [
    img.rgbTopToBottom[o] / 255,
    img.rgbTopToBottom[o + 1] / 255,
    img.rgbTopToBottom[o + 2] / 255,
  ];
}

function sampleBilinearLum01(
  img: DecodedRgbImageGrid,
  fx: number,
  fy: number
): number {
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(img.width - 1, x0 + 1);
  const y1 = Math.min(img.height - 1, y0 + 1);
  const tx = fx - x0;
  const ty = fy - y0;
  const w00 = (1 - tx) * (1 - ty);
  const w10 = tx * (1 - ty);
  const w01 = (1 - tx) * ty;
  const w11 = tx * ty;
  return (
    w00 * rgbAtLum01(img, x0, y0) +
    w10 * rgbAtLum01(img, x1, y0) +
    w01 * rgbAtLum01(img, x0, y1) +
    w11 * rgbAtLum01(img, x1, y1)
  );
}

function sampleBilinearRgbLum01(
  img: DecodedRgbImageGrid,
  fx: number,
  fy: number
): { lum: number; rgb: [number, number, number] } {
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(img.width - 1, x0 + 1);
  const y1 = Math.min(img.height - 1, y0 + 1);
  const tx = fx - x0;
  const ty = fy - y0;
  const w00 = (1 - tx) * (1 - ty);
  const w10 = tx * (1 - ty);
  const w01 = (1 - tx) * ty;
  const w11 = tx * ty;
  const p00 = rgbTuple01(img, x0, y0);
  const p10 = rgbTuple01(img, x1, y0);
  const p01 = rgbTuple01(img, x0, y1);
  const p11 = rgbTuple01(img, x1, y1);
  const r = w00 * p00[0] + w10 * p10[0] + w01 * p01[0] + w11 * p11[0];
  const g = w00 * p00[1] + w10 * p10[1] + w01 * p01[1] + w11 * p11[1];
  const b = w00 * p00[2] + w10 * p10[2] + w01 * p01[2] + w11 * p11[2];
  return { lum: 0.299 * r + 0.587 * g + 0.114 * b, rgb: [r, g, b] };
}

const PATCH_A_BUF = new Float64Array(25);
const PATCH_B_BUF = new Float64Array(25);

function compute5x5PatchMetrics(
  srcImg: DecodedRgbImageGrid,
  dstImg: DecodedRgbImageGrid,
  cx: number,
  cy: number,
  targetFx: number
): { zsadPerPx: number; zncc: number | null; srcStd: number } {
  let meanA = 0;
  let meanB = 0;
  let k = 0;
  for (let dy = -2; dy <= 2; dy++) {
    const sy = cy + dy;
    for (let dx = -2; dx <= 2; dx++) {
      const av = rgbAtLum01(srcImg, cx + dx, sy);
      const bv = sampleBilinearLum01(dstImg, targetFx + dx, sy);
      PATCH_A_BUF[k] = av;
      PATCH_B_BUF[k] = bv;
      meanA += av;
      meanB += bv;
      k++;
    }
  }
  meanA /= 25;
  meanB /= 25;

  let zsadSum = 0;
  let varA = 0;
  let varB = 0;
  let cov = 0;
  for (let i = 0; i < 25; i++) {
    const da = PATCH_A_BUF[i] - meanA;
    const db = PATCH_B_BUF[i] - meanB;
    zsadSum += Math.abs(da - db);
    varA += da * da;
    varB += db * db;
    cov += da * db;
  }
  const stdA = Math.sqrt(varA / 25);
  const denom = Math.sqrt(varA * varB);
  const zncc = stdA >= 0.01 && denom > 1e-12 ? cov / denom : null;
  return { zsadPerPx: zsadSum / 25, zncc, srcStd: stdA };
}

function p95Sorted(sortedAsc: number[]): number {
  if (sortedAsc.length === 0) return NaN;
  const idx = Math.min(sortedAsc.length - 1, Math.floor(0.95 * (sortedAsc.length - 1)));
  return sortedAsc[idx];
}

export function evaluateEyeSourceGtRoiConsistency(params: {
  eye: MiddleburyEyeId;
  roi: NativeRoiBoxPx;
  srcImg: DecodedRgbImageGrid;
  dstImg: DecodedRgbImageGrid;
  srcGt: ParsedPfmGrid;
  dstGt: ParsedPfmGrid;
  srcMask: ParsedPngMaskGrid;
  dstMask: ParsedPngMaskGrid;
  calib: MiddleburyCalibData;
}): EyeSourceGtRoiConsistency {
  const { eye, roi, srcImg, dstImg, srcGt, dstGt, srcMask, dstMask, calib } = params;
  const w = calib.width;
  const h = calib.height;
  const sign = eye === 'left' ? -1 : 1; // left: x_R = x_L - d0; right: x_L = x_R + d1

  const nearestRecipAbs: number[] = [];
  const bilinearRecipAbs: number[] = [];
  const gradDiffs: number[] = [];

  let oppositeNonoccHits = 0;
  let oppositeOccOrInvalidHits = 0;
  let nearestLe1Count = 0;
  let nearestMax = 0;
  let nearestSum = 0;
  let nearestSignedSum = 0;
  let bilinearSum = 0;

  let gtMin = Infinity;
  let gtMax = -Infinity;

  let evaluatedPatchCount = 0;
  let sumSrcLum = 0;
  let sumDstLum = 0;
  let sumAbsLum = 0;
  let sumAbsRgb = 0;
  let sumGradDiff = 0;
  let sumSrcGrad = 0;
  let sumZsad = 0;
  let znccSum = 0;
  let znccCount = 0;

  // Alternative unit & sign comparisons on exact same nonocc points where all 4 shift projections lie in bounds
  let altCount = 0;
  let pos1xLumSum = 0;
  let neg1xLumSum = 0;
  let half05xLumSum = 0;
  let double2xLumSum = 0;
  let pos1xZnccSum = 0;
  let pos1xZnccCnt = 0;
  let neg1xZnccSum = 0;
  let neg1xZnccCnt = 0;
  let half05xZnccSum = 0;
  let half05xZnccCnt = 0;
  let double2xZnccSum = 0;
  let double2xZnccCnt = 0;

  const yMin = Math.max(0, roi.yStart);
  const yMax = Math.min(h, roi.yEndExclusive);
  const xMin = Math.max(0, roi.xStart);
  const xMax = Math.min(w, roi.xEndExclusive);

  for (let y = yMin; y < yMax; y++) {
    for (let x = xMin; x < xMax; x++) {
      const idx = y * w + x;
      if (srcMask.grayTopToBottom[idx] !== 255) continue;
      const dSrc = srcGt.dataTopToBottom[idx];
      if (!Number.isFinite(dSrc) || dSrc <= 0 || dSrc > calib.ndisp) continue;

      const xMatch = x + sign * dSrc;
      if (xMatch < 0 || xMatch > w - 1) continue;

      if (dSrc < gtMin) gtMin = dSrc;
      if (dSrc > gtMax) gtMax = dSrc;

      // 1. Reciprocal L<->R disparity consistency (nearest non-interpolated + valid-only bilinear)
      const xNearest = Math.round(xMatch);
      const dstIdxNear = y * w + xNearest;
      const dstMaskVal = dstMask.grayTopToBottom[dstIdxNear];
      if (dstMaskVal === 255) oppositeNonoccHits++;
      else oppositeOccOrInvalidHits++;

      const dDstNear = dstGt.dataTopToBottom[dstIdxNear];
      if (Number.isFinite(dDstNear) && dDstNear > 0) {
        const signedDiff = dSrc - dDstNear;
        const absDiff = Math.abs(signedDiff);
        nearestRecipAbs.push(absDiff);
        nearestSum += absDiff;
        nearestSignedSum += signedDiff;
        if (absDiff > nearestMax) nearestMax = absDiff;
        if (absDiff <= 1.0) nearestLe1Count++;
      }

      // Valid-only bilinear GT lookup in opposite eye (strictly NO interpolation across Infinity/<=0 holes)
      const x0 = Math.floor(xMatch);
      const x1 = Math.min(w - 1, x0 + 1);
      const tx = xMatch - x0;
      const d0 = dstGt.dataTopToBottom[y * w + x0];
      const d1 = dstGt.dataTopToBottom[y * w + x1];
      if (Number.isFinite(d0) && d0 > 0 && Number.isFinite(d1) && d1 > 0) {
        const dDstBil = (1 - tx) * d0 + tx * d1;
        const absBil = Math.abs(dSrc - dDstBil);
        bilinearRecipAbs.push(absBil);
        bilinearSum += absBil;
      }

      // 2. Photometric, gradient & exposure-invariant 5x5 patch consistency (interior >= 3px from borders)
      if (y >= 3 && y < h - 3 && x >= 3 && x < w - 3 && xMatch >= 3 && xMatch <= w - 4) {
        const pSrc = rgbTuple01(srcImg, x, y);
        const lSrc = 0.299 * pSrc[0] + 0.587 * pSrc[1] + 0.114 * pSrc[2];
        const sDst = sampleBilinearRgbLum01(dstImg, xMatch, y);

        evaluatedPatchCount++;
        sumSrcLum += lSrc;
        sumDstLum += sDst.lum;
        sumAbsLum += Math.abs(lSrc - sDst.lum);
        sumAbsRgb +=
          (Math.abs(pSrc[0] - sDst.rgb[0]) +
            Math.abs(pSrc[1] - sDst.rgb[1]) +
            Math.abs(pSrc[2] - sDst.rgb[2])) /
          3;

        const gxSrc = 0.5 * (rgbAtLum01(srcImg, x + 1, y) - rgbAtLum01(srcImg, x - 1, y));
        const gySrc = 0.5 * (rgbAtLum01(srcImg, x, y + 1) - rgbAtLum01(srcImg, x, y - 1));
        const gxDst =
          0.5 *
          (sampleBilinearRgbLum01(dstImg, xMatch + 1, y).lum -
            sampleBilinearRgbLum01(dstImg, xMatch - 1, y).lum);
        const gyDst =
          0.5 *
          (sampleBilinearRgbLum01(dstImg, xMatch, y + 1).lum -
            sampleBilinearRgbLum01(dstImg, xMatch, y - 1).lum);
        const gd = Math.hypot(gxSrc - gxDst, gySrc - gyDst);
        gradDiffs.push(gd);
        sumGradDiff += gd;
        sumSrcGrad += Math.hypot(gxSrc, gySrc);

        const patch = compute5x5PatchMetrics(srcImg, dstImg, x, y, xMatch);
        sumZsad += patch.zsadPerPx;
        if (patch.zncc !== null) {
          znccSum += patch.zncc;
          znccCount++;
        }

        // Unit & sign hypotheses when all 4 reprojections lie cleanly inside 5x5 borders
        const xNeg = x - sign * dSrc;
        const xHalf = x + sign * (0.5 * dSrc);
        const xDbl = x + sign * (2.0 * dSrc);
        if (
          xNeg >= 3 &&
          xNeg <= w - 4 &&
          xHalf >= 3 &&
          xHalf <= w - 4 &&
          xDbl >= 3 &&
          xDbl <= w - 4
        ) {
          altCount++;
          pos1xLumSum += Math.abs(lSrc - sDst.lum);
          neg1xLumSum += Math.abs(lSrc - sampleBilinearRgbLum01(dstImg, xNeg, y).lum);
          half05xLumSum += Math.abs(lSrc - sampleBilinearRgbLum01(dstImg, xHalf, y).lum);
          double2xLumSum += Math.abs(lSrc - sampleBilinearRgbLum01(dstImg, xDbl, y).lum);

          if (patch.zncc !== null) {
            pos1xZnccSum += patch.zncc;
            pos1xZnccCnt++;
            const pNeg = compute5x5PatchMetrics(srcImg, dstImg, x, y, xNeg);
            if (pNeg.zncc !== null) {
              neg1xZnccSum += pNeg.zncc;
              neg1xZnccCnt++;
            }
            const pHalf = compute5x5PatchMetrics(srcImg, dstImg, x, y, xHalf);
            if (pHalf.zncc !== null) {
              half05xZnccSum += pHalf.zncc;
              half05xZnccCnt++;
            }
            const pDbl = compute5x5PatchMetrics(srcImg, dstImg, x, y, xDbl);
            if (pDbl.zncc !== null) {
              double2xZnccSum += pDbl.zncc;
              double2xZnccCnt++;
            }
          }
        }
      }
    }
  }

  nearestRecipAbs.sort((a, b) => a - b);
  bilinearRecipAbs.sort((a, b) => a - b);
  gradDiffs.sort((a, b) => a - b);

  const nNear = nearestRecipAbs.length;
  const nBil = bilinearRecipAbs.length;
  const totalMatchAttempts = oppositeNonoccHits + oppositeOccOrInvalidHits;

  const positive1xAbsLum01 = altCount > 0 ? pos1xLumSum / altCount : NaN;
  const negatedSignAbsLum01 = altCount > 0 ? neg1xLumSum / altCount : NaN;
  const halfScale05xAbsLum01 = altCount > 0 ? half05xLumSum / altCount : NaN;
  const doubleScale2xAbsLum01 = altCount > 0 ? double2xLumSum / altCount : NaN;
  const positive1xZncc5x5 = pos1xZnccCnt > 0 ? pos1xZnccSum / pos1xZnccCnt : NaN;
  const negatedSignZncc5x5 = neg1xZnccCnt > 0 ? neg1xZnccSum / neg1xZnccCnt : NaN;
  const halfScale05xZncc5x5 = half05xZnccCnt > 0 ? half05xZnccSum / half05xZnccCnt : NaN;
  const doubleScale2xZncc5x5 = double2xZnccCnt > 0 ? double2xZnccSum / double2xZnccCnt : NaN;

  const confirmedUnitAndSign =
    Number.isFinite(positive1xZncc5x5) &&
    positive1xZncc5x5 > negatedSignZncc5x5 &&
    positive1xZncc5x5 > halfScale05xZncc5x5 &&
    positive1xZncc5x5 > doubleScale2xZncc5x5;

  return {
    roiId: roi.roiId,
    label: roi.label,
    eye,
    note: roi.note,
    evaluatedRecipCount: nNear,
    oppositeNonoccHitRate: totalMatchAttempts > 0 ? oppositeNonoccHits / totalMatchAttempts : 0,
    oppositeOccOrInvalidHitCount: oppositeOccOrInvalidHits,
    gtNativeMinPx: Number.isFinite(gtMin) ? gtMin : NaN,
    gtNativeMaxPx: Number.isFinite(gtMax) ? gtMax : NaN,
    nearestRecipAbsMeanPx: nNear > 0 ? nearestSum / nNear : NaN,
    nearestRecipAbsP95Px: p95Sorted(nearestRecipAbs),
    nearestRecipAbsMaxPx: nNear > 0 ? nearestMax : NaN,
    nearestRecipLe1PxRate: nNear > 0 ? nearestLe1Count / nNear : 0,
    bilinearRecipValidCount: nBil,
    bilinearRecipAbsMeanPx: nBil > 0 ? bilinearSum / nBil : NaN,
    bilinearRecipAbsP95Px: p95Sorted(bilinearRecipAbs),
    signedRecipBiasPx: nNear > 0 ? nearestSignedSum / nNear : NaN,
    evaluatedPatchCount,
    meanSrcLum01: evaluatedPatchCount > 0 ? sumSrcLum / evaluatedPatchCount : NaN,
    meanTargetLum01: evaluatedPatchCount > 0 ? sumDstLum / evaluatedPatchCount : NaN,
    signedExposureOffsetLum01:
      evaluatedPatchCount > 0 ? (sumSrcLum - sumDstLum) / evaluatedPatchCount : NaN,
    absLumResidual01: evaluatedPatchCount > 0 ? sumAbsLum / evaluatedPatchCount : NaN,
    absRgbResidual01: evaluatedPatchCount > 0 ? sumAbsRgb / evaluatedPatchCount : NaN,
    meanGradDiff01: evaluatedPatchCount > 0 ? sumGradDiff / evaluatedPatchCount : NaN,
    p95GradDiff01: p95Sorted(gradDiffs),
    relGradDiffToSrcGrad: sumSrcGrad > 1e-9 ? sumGradDiff / sumSrcGrad : NaN,
    zsad5x5MeanPerPx01: evaluatedPatchCount > 0 ? sumZsad / evaluatedPatchCount : NaN,
    zncc5x5TexturedCount: znccCount,
    zncc5x5TexturedMean: znccCount > 0 ? znccSum / znccCount : NaN,
    signAndUnitCheck: {
      positive1xAbsLum01,
      negatedSignAbsLum01,
      halfScale05xAbsLum01,
      doubleScale2xAbsLum01,
      positive1xZncc5x5,
      negatedSignZncc5x5,
      halfScale05xZncc5x5,
      doubleScale2xZncc5x5,
      confirmedUnitAndSign,
    },
  };
}

async function checkSignedYMapStatus(
  sceneId: MiddleburyEval3SceneId,
  calib: MiddleburyCalibData,
  fetchImpl: typeof fetch
): Promise<SignedYMapAvailabilityStatus> {
  const base = MIDDLEBURY_EVAL3_Q_SCENES[sceneId].baseFixturePath;
  const urls = [`${base}/disp0GTy.pfm`, `${base}/disp1GTy.pfm`];
  let allFound = true;
  for (const url of urls) {
    try {
      const res = await fetchImpl(url, { method: 'GET' });
      const ct = res.headers.get('content-type')?.toLowerCase() ?? '';
      if (!res.ok || ct.startsWith('text/html')) {
        allFound = false;
        break;
      }
      const u8 = new Uint8Array(await res.arrayBuffer());
      // Must start with 'Pf' (0x50 0x66) to avoid Vite SPA HTML fallback
      if (u8.byteLength < 10 || u8[0] !== 0x50 || u8[1] !== 0x66) {
        allFound = false;
        break;
      }
    } catch {
      allFound = false;
      break;
    }
  }
  const statusNote = allFound
    ? `Official ${MIDDLEBURY_EVAL3_Q_SCENES[sceneId].edition} signed disp0GTy.pfm / disp1GTy.pfm verified alongside calib (dyavg=${calib.dyavg}, dymax=${calib.dymax}).`
    : calib.dymax > 0
      ? `Official same-edition disp0GTy.pfm / disp1GTy.pfm unavailable in local static fixtures (calib.txt reports scene-level dyavg=${calib.dyavg} px, dymax=${calib.dymax} px; no mismatched perfect/imperfect substitution performed).`
      : `Rectified 2003 Q scene (calib.txt dyavg=0 px, dymax=0 px); disp*GTy.pfm absent/not needed.`;
  return {
    checkedUrls: urls,
    available: allFound,
    statusNote,
    calibDyavgPx: calib.dyavg,
    calibDymaxPx: calib.dymax,
  };
}

export async function analyzeMiddleburySourceGtConsistency(params: {
  sceneAssets: LoadedMiddleburySceneAssets;
  fetchImpl?: typeof fetch;
}): Promise<SceneSourceGtConsistencyReport> {
  const { sceneAssets } = params;
  const fetchImpl = params.fetchImpl ?? globalThis.fetch;
  const [im0Rgb, im1Rgb, yGtMapStatus] = await Promise.all([
    decodePngRgbTopToBottomBitExact(sceneAssets.im0Bytes),
    decodePngRgbTopToBottomBitExact(sceneAssets.im1Bytes),
    checkSignedYMapStatus(sceneAssets.sceneId, sceneAssets.calib, fetchImpl),
  ]);

  if (
    im0Rgb.width !== sceneAssets.calib.width ||
    im0Rgb.height !== sceneAssets.calib.height ||
    im1Rgb.width !== sceneAssets.calib.width ||
    im1Rgb.height !== sceneAssets.calib.height
  ) {
    throw new Error(
      `Decoded RGB sizes (${im0Rgb.width}x${im0Rgb.height}, ${im1Rgb.width}x${im1Rgb.height}) disagree with calib ${sceneAssets.calib.width}x${sceneAssets.calib.height}`
    );
  }

  const rois: EyeSourceGtRoiConsistency[] = [];
  for (const roiBox of PREDECLARED_NATIVE_ROIS[sceneAssets.sceneId]) {
    rois.push(
      evaluateEyeSourceGtRoiConsistency({
        eye: 'left',
        roi: roiBox,
        srcImg: im0Rgb,
        dstImg: im1Rgb,
        srcGt: sceneAssets.disp0GtPfm,
        dstGt: sceneAssets.disp1GtPfm,
        srcMask: sceneAssets.mask0Nocc,
        dstMask: sceneAssets.mask1Nocc,
        calib: sceneAssets.calib,
      }),
      evaluateEyeSourceGtRoiConsistency({
        eye: 'right',
        roi: roiBox,
        srcImg: im1Rgb,
        dstImg: im0Rgb,
        srcGt: sceneAssets.disp1GtPfm,
        dstGt: sceneAssets.disp0GtPfm,
        srcMask: sceneAssets.mask1Nocc,
        dstMask: sceneAssets.mask0Nocc,
        calib: sceneAssets.calib,
      })
    );
  }

  return {
    sceneId: sceneAssets.sceneId,
    edition: sceneAssets.spec.edition,
    calib: sceneAssets.calib,
    yGtMapStatus,
    rois,
  };
}

export async function runHitnetMiddleburyRawAndGtDiagnostic(params?: {
  isDev?: boolean;
  fetchImpl?: typeof fetch;
  runWorker?: ResidualDiagnosticWorkerRunner;
  loadSceneAssets?: (sceneId: MiddleburyEval3SceneId) => Promise<LoadedMiddleburySceneAssets>;
  buildGeometryRgba?: FactorialFeedPairBuilder;
  onLog?: (line: string) => void;
}): Promise<HitnetMiddleburyRawAndGtReport> {
  assertDevFixtureEnabled(params?.isDev);
  const fetchImpl = params?.fetchImpl ?? globalThis.fetch;
  const log = params?.onLog ?? (() => {});
  const runWorker = params?.runWorker ?? createBrowserOrt130DiagnosticRunner();
  const loadScene = params?.loadSceneAssets ?? ((id) => loadAndVerifyMiddleburySceneAssets(id, fetchImpl));
  const buildGeometryRgba = params?.buildGeometryRgba ?? defaultBrowserFactorialFeedPairBuilder;
  const sceneIds: MiddleburyEval3SceneId[] = ['Motorcycle', 'Teddy'];

  log(
    `=== [DEV-ONLY] Middlebury HITNet Raw Release vs Canonical M1 & Source-GT Consistency Diagnostic ===\n` +
      `Step 1: Verifying stereodemo v0.1-hitnet public raw asset provenance via configured proxy (${EXPECTED_RAW_HITNET_PROXY_PATH})...`
  );

  const { check: rawProvenance, rawBytes } = await verifyRawHitnetProvenanceAndContract(fetchImpl);

  log(
    `  Raw Provenance: verified=${String(rawProvenance.verified)} | bytes=${String(rawProvenance.rawByteLength)} | sha256=${String(rawProvenance.rawSha256)}\n` +
      `  Upstream: ${rawProvenance.configuredProfileUpstreamUrl} | ProxyUpstream: ${String(rawProvenance.configuredProxyUpstreamUrl)}\n` +
      `  Contract: in=${JSON.stringify(rawProvenance.graphInputs)} | out=${JSON.stringify(rawProvenance.graphOutputs)} | nodes=${String(rawProvenance.rawNodeCount)}` +
      (rawProvenance.failureReason ? `\n  [PROVENANCE STOP] ${rawProvenance.failureReason}` : '')
  );

  // Load both official MiddEval3 Q scenes (Motorcycle + Teddy)
  const sceneAssetsMap = {} as Record<MiddleburyEval3SceneId, LoadedMiddleburySceneAssets>;
  const sourceGtConsistencyByScene = {} as Record<
    MiddleburyEval3SceneId,
    SceneSourceGtConsistencyReport
  >;

  for (const sceneId of sceneIds) {
    const assets = await loadScene(sceneId);
    sceneAssetsMap[sceneId] = assets;
    log(`\n--- Step 2: Source-Native GT Consistency (${sceneId}, ${assets.spec.edition}) ---`);
    const gtCons = await analyzeMiddleburySourceGtConsistency({
      sceneAssets: assets,
      fetchImpl,
    });
    sourceGtConsistencyByScene[sceneId] = gtCons;
    log(`  Y-GT Status: ${gtCons.yGtMapStatus.statusNote}`);
    for (const r of gtCons.rois) {
      log(
        `  [${r.roiId} | ${r.eye}] nRecip=${r.evaluatedRecipCount} oppNonoccHit=${(r.oppositeNonoccHitRate * 100).toFixed(2)}% | ` +
          `recipDiff mean=${r.nearestRecipAbsMeanPx.toFixed(4)}px p95=${r.nearestRecipAbsP95Px.toFixed(4)}px max=${r.nearestRecipAbsMaxPx.toFixed(3)}px <=1px=${(r.nearestRecipLe1PxRate * 100).toFixed(2)}% bilMean=${r.bilinearRecipAbsMeanPx.toFixed(4)}px | ` +
          `photo: LumDiff=${r.absLumResidual01.toFixed(4)} expOffset=${r.signedExposureOffsetLum01.toFixed(4)} GradDiff=${r.meanGradDiff01.toFixed(4)} (rel=${(r.relGradDiffToSrcGrad * 100).toFixed(1)}%) ZSAD5x5=${r.zsad5x5MeanPerPx01.toFixed(4)} ZNCC5x5=${r.zncc5x5TexturedMean.toFixed(4)} | ` +
          `sign/unit(+1x vs -1x vs 0.5x vs 2x ZNCC)=[${r.signAndUnitCheck.positive1xZncc5x5.toFixed(3)}, ${r.signAndUnitCheck.negatedSignZncc5x5.toFixed(3)}, ${r.signAndUnitCheck.halfScale05xZncc5x5.toFixed(3)}, ${r.signAndUnitCheck.doubleScale2xZncc5x5.toFixed(3)}] confirmed=${String(r.signAndUnitCheck.confirmedUnitAndSign)}`
      );
    }
  }

  let rawComparisonExecuted = false;
  const rawComparisonAbortedReason: string | null = rawProvenance.verified
    ? null
    : rawProvenance.failureReason ?? 'Raw provenance verification failed; aborted RAW comparison without weight substitution.';
  let m1ByteLength: number | null = null;
  let m1Sha256: string | null = null;
  let n1OffProof: N1OffVerificationProof | null = null;
  const rawVsM1ByScene: Partial<Record<MiddleburyEval3SceneId, SceneRawVsM1ComparisonReport>> = {};

  if (rawProvenance.verified && rawBytes) {
    const m1Prep = prepareHitnetM1FullGraphBytes(rawBytes, {
      rawModelSha256: HITNET_RAW_MODEL_SHA256,
    });
    m1ByteLength = m1Prep.bytes.byteLength;
    m1Sha256 = await sha256HexBytes(m1Prep.bytes);
    if (
      m1ByteLength !== HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength ||
      m1Sha256 !== HITNET_M1_FULL_GRAPH_SHA256
    ) {
      throw new Error(
        `Canonical M1 verification failed: bytes=${m1ByteLength}, sha=${m1Sha256} (expected ${HITNET_M1_FULL_GRAPH_SHA256})`
      );
    }
    const parsedM1 = await parseM1ModelProtoWithRawSlices(m1Prep.bytes);
    n1OffProof = verifyCanonicalM1NoN1Replacements(parsedM1, m1Prep.bytes, m1Sha256, false);
    if (!n1OffProof.verified) {
      throw new Error(n1OffProof.failureReason ?? 'Canonical M1 N1=OFF check failed');
    }

    const [pName, sName] = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs;
    log(
      `\n--- Step 3: Raw Release vs Canonical M1 Inference Comparison (ORT 1.30.0 WASM opt=basic, NO implicit rewrites) ---\n` +
        `  Raw=${rawProvenance.rawByteLength}B (${rawProvenance.rawSha256}) vs M1=${m1ByteLength}B (${m1Sha256}, N1=OFF 0 replacements)`
    );

    const runWasmModelArm = async (paramsRun: {
      sceneId: MiddleburyEval3SceneId;
      assets: LoadedMiddleburySceneAssets;
      modelVariant: 'raw_release' | 'canonical_m1';
      modelBytes: Uint8Array;
      expectedModelSha: string;
      channelOrder: FactorialChannelOrder;
      feedFloat32: Float32Array;
      feedSha256: string;
      g0: GeometryTransformSpec;
      leftSet: SharedSceneEyeSampleSet;
      rightSet: SharedSceneEyeSampleSet;
    }): Promise<{
      summary: RawVsM1ArmRunSummary;
      leftOut: Float32Array;
      rightOut: Float32Array;
    }> => {
      const {
        sceneId,
        assets,
        modelVariant,
        modelBytes,
        expectedModelSha,
        channelOrder,
        feedFloat32,
        feedSha256,
        leftSet,
        rightSet,
      } = paramsRun;
      const rawRes: OrtVersionDiagnosticResponse = await runWorker({
        expectedOrtVersion: '1.30.0',
        runtimeAssetFamily: 'asyncify',
        modelBuffer: modelBytes.slice().buffer,
        feedBuffer: feedFloat32.slice().buffer,
        feedName: 'input',
        feedDims: [1, 6, 240, 320],
        provider: 'wasm',
        graphOptimizationLevel: 'basic',
        warmRuns: 1,
        enableWebGpuProfiling: false,
        disableCpuEpFallback: false,
        captureGpuValidationErrors: false,
      });
      if (!rawRes.ok) {
        const fail = rawRes as OrtVersionDiagnosticFailure;
        throw new Error(`${sceneId} ${modelVariant} ${channelOrder} FAILED (${fail.stage}): ${fail.message}`);
      }
      const okRes = rawRes as OrtVersionDiagnosticSuccess;
      if (
        okRes.pinnedOrtVersion !== '1.30.0' ||
        okRes.runtimeAssetFamily !== 'asyncify' ||
        okRes.graphOptimizationLevel !== 'basic' ||
        okRes.provider !== 'wasm' ||
        okRes.configuredProviders.length !== 1 ||
        okRes.configuredProviders[0] !== 'wasm' ||
        okRes.foreignOrtResources.length > 0 ||
        okRes.loadedOrtResources.some((u) => u.includes('.jsep.') || u.includes('ort.bundle.'))
      ) {
        throw new Error(
          `${sceneId} ${modelVariant} ${channelOrder} runtime guard violation: ${JSON.stringify(okRes.configuredProviders)}`
        );
      }
      if (okRes.modelSha256 !== expectedModelSha || okRes.modelByteLength !== modelBytes.byteLength) {
        throw new Error(
          `${sceneId} ${modelVariant} ${channelOrder} pre-session model SHA mismatch: ${okRes.modelSha256} vs ${expectedModelSha}`
        );
      }
      if (
        okRes.feedSha256 !== feedSha256 ||
        okRes.feedTensorName !== 'input' ||
        okRes.feedTensorDataType !== 'float32' ||
        JSON.stringify(okRes.feedTensorShape) !== JSON.stringify([1, 6, 240, 320])
      ) {
        throw new Error(
          `${sceneId} ${modelVariant} ${channelOrder} pre-session feed mismatch: sha=${okRes.feedSha256}, shape=${JSON.stringify(okRes.feedTensorShape)}`
        );
      }

      const val = validateHitnetFinalOutputs(okRes.outputs);
      if (!val.ok) {
        throw new Error(`${sceneId} ${modelVariant} ${channelOrder} output contract error: ${val.error}`);
      }
      const leftOut = val.byName[pName].data;
      const rightOut = val.byName[sName].data;
      const caveat = MIDDLEBURY_EVAL3_Q_SCENES[sceneId].rectificationCaveat;

      const leftEval = evaluateFactorialArmEye({
        pred: leftOut,
        geometry: 'G0_direct_320x240',
        sampleSet: leftSet,
        oppositeSampleSet: rightSet,
        feedFloat32,
        calib: assets.calib,
        rectificationCaveat: caveat,
      });
      const rightEval = evaluateFactorialArmEye({
        pred: rightOut,
        geometry: 'G0_direct_320x240',
        sampleSet: rightSet,
        oppositeSampleSet: leftSet,
        feedFloat32,
        calib: assets.calib,
        rectificationCaveat: caveat,
      });

      return {
        leftOut,
        rightOut,
        summary: {
          modelVariant,
          channelOrder,
          feedSha256: okRes.feedSha256,
          feedTensorName: okRes.feedTensorName,
          feedTensorShape: [...okRes.feedTensorShape],
          feedTensorDataType: okRes.feedTensorDataType,
          modelByteLength: okRes.modelByteLength,
          modelSha256: okRes.modelSha256,
          sessionInputNames: [...okRes.sessionInputNames],
          sessionOutputNames: [...okRes.sessionOutputNames],
          outputContracts: okRes.outputs.map((o) => ({
            name: o.name,
            dims: [...o.dims],
            dataType: o.dataType,
          })),
          configuredProviders: [...okRes.configuredProviders],
          loadedOrtResources: [...okRes.loadedOrtResources],
          foreignOrtResources: [...okRes.foreignOrtResources],
          wasmAssetShas: (okRes.loadedAssetProvenance ?? [])
            .filter((p) => p.role === 'wasm')
            .map((p) => ({
              url: p.url,
              sha256: p.sha256,
              matchesLocalDist: p.matchesLocalDistSha256,
            })),
          createMs: okRes.createMs,
          coldRunMs: okRes.coldRunMs,
          warmRunMs: [...okRes.warmRunMs],
          leftPrimary: leftEval,
          rightSecondary: rightEval,
        },
      };
    };

    for (const sceneId of sceneIds) {
      const assets = sceneAssetsMap[sceneId];
      const { g0, g1 } = buildMiddleburyGeometryTransforms(
        sceneId,
        assets.calib.width,
        assets.calib.height
      );
      const leftSet = buildSharedPhysicalSampleSetForEye({
        eye: 'left',
        sceneId,
        gtPfm: assets.disp0GtPfm,
        maskNocc: assets.mask0Nocc,
        calib: assets.calib,
        g0,
        g1,
      });
      const rightSet = buildSharedPhysicalSampleSetForEye({
        eye: 'right',
        sceneId,
        gtPfm: assets.disp1GtPfm,
        maskNocc: assets.mask1Nocc,
        calib: assets.calib,
        g0,
        g1,
      });
      const rgbaPair = await buildGeometryRgba({
        sceneAssets: assets,
        g0Transform: g0,
        g1Transform: g1,
      });

      const byFeed = {} as Record<FactorialChannelOrder, RawVsM1FeedComparison>;
      for (const channelOrder of ['C0_RGB', 'C1_BGR'] as const) {
        const feedFloat32 = packRgbaStereoEyesWithChannelOrder(
          rgbaPair.G0.leftRgba,
          rgbaPair.G0.rightRgba,
          channelOrder
        );
        const feedSha256 = await sha256HexBytes(
          new Uint8Array(feedFloat32.buffer, feedFloat32.byteOffset, feedFloat32.byteLength)
        );

        const rawExec = await runWasmModelArm({
          sceneId,
          assets,
          modelVariant: 'raw_release',
          modelBytes: rawBytes,
          expectedModelSha: HITNET_RAW_MODEL_SHA256,
          channelOrder,
          feedFloat32,
          feedSha256,
          g0,
          leftSet,
          rightSet,
        });
        const m1Exec = await runWasmModelArm({
          sceneId,
          assets,
          modelVariant: 'canonical_m1',
          modelBytes: m1Prep.bytes,
          expectedModelSha: HITNET_M1_FULL_GRAPH_SHA256,
          channelOrder,
          feedFloat32,
          feedSha256,
          g0,
          leftSet,
          rightSet,
        });

        const pParity = summarizeRawTensorAndParity(rawExec.leftOut, m1Exec.leftOut);
        const sParity = summarizeRawTensorAndParity(rawExec.rightOut, m1Exec.rightOut);
        let pNonZeroDiff = 0;
        const pRawU32 = new Uint32Array(rawExec.leftOut.buffer, rawExec.leftOut.byteOffset, rawExec.leftOut.length);
        const pM1U32 = new Uint32Array(m1Exec.leftOut.buffer, m1Exec.leftOut.byteOffset, m1Exec.leftOut.length);
        const pMinLen = Math.min(pRawU32.length, pM1U32.length);
        for (let i = 0; i < pMinLen; i++) {
          if (pRawU32[i] !== pM1U32[i]) pNonZeroDiff++;
        }
        pNonZeroDiff += Math.abs(pRawU32.length - pM1U32.length);

        let sNonZeroDiff = 0;
        const sRawU32 = new Uint32Array(rawExec.rightOut.buffer, rawExec.rightOut.byteOffset, rawExec.rightOut.length);
        const sM1U32 = new Uint32Array(m1Exec.rightOut.buffer, m1Exec.rightOut.byteOffset, m1Exec.rightOut.length);
        const sMinLen = Math.min(sRawU32.length, sM1U32.length);
        for (let i = 0; i < sMinLen; i++) {
          if (sRawU32[i] !== sM1U32[i]) sNonZeroDiff++;
        }
        sNonZeroDiff += Math.abs(sRawU32.length - sM1U32.length);

        const bitIdenticalBothOutputs =
          pParity.length === HITNET_MIDDLEBURY_PIXEL_COUNT &&
          sParity.length === HITNET_MIDDLEBURY_PIXEL_COUNT &&
          pNonZeroDiff === 0 &&
          sNonZeroDiff === 0 &&
          pParity.nonFiniteMismatchCount === 0 &&
          sParity.nonFiniteMismatchCount === 0;

        const rL = rawExec.summary.leftPrimary.nonocc;
        const mL = m1Exec.summary.leftPrimary.nonocc;
        const rR = rawExec.summary.rightSecondary.nonocc;
        const mR = m1Exec.summary.rightSecondary.nonocc;

        byFeed[channelOrder] = {
          channelOrder,
          feedSha256,
          rawArm: rawExec.summary,
          m1Arm: m1Exec.summary,
          fullTensorParityRawVsM1: {
            reference_output_disparity: pParity,
            secondary_output_disparity: sParity,
            bitIdenticalBothOutputs,
          },
          accuracyDeltaRawMinusM1: {
            leftNonoccEpeDeltaPx: rL.epeModelPx - mL.epeModelPx,
            leftNonoccP95DeltaPx: rL.p95AbsErrorModelPx - mL.p95AbsErrorModelPx,
            leftNonoccBad1DeltaPctPts: (rL.bad1Rate - mL.bad1Rate) * 100,
            leftNonoccBad3DeltaPctPts: (rL.bad3Rate - mL.bad3Rate) * 100,
            leftNonoccInvalidDeltaCount: rL.invalidPredCount - mL.invalidPredCount,
            rightNonoccEpeDeltaPx: rR.epeModelPx - mR.epeModelPx,
            rightNonoccP95DeltaPx: rR.p95AbsErrorModelPx - mR.p95AbsErrorModelPx,
            rightNonoccBad1DeltaPctPts: (rR.bad1Rate - mR.bad1Rate) * 100,
            rightNonoccBad3DeltaPctPts: (rR.bad3Rate - mR.bad3Rate) * 100,
            rightNonoccInvalidDeltaCount: rR.invalidPredCount - mR.invalidPredCount,
          },
        };

        log(
          `  [${sceneId} | G0 ${channelOrder}] feed=${feedSha256.slice(0, 12)}... | Raw vs M1 parity: ` +
            `ref(max=${pParity.maxAbsDiff.toExponential(3)}, mean=${pParity.meanAbsDiff.toExponential(3)}, nonzero=${pNonZeroDiff}) ` +
            `sec(max=${sParity.maxAbsDiff.toExponential(3)}, mean=${sParity.meanAbsDiff.toExponential(3)}, nonzero=${sNonZeroDiff})\n` +
            `    RAW Left nonocc(n=${rL.evaluatedCount}): EPE=${rL.epeModelPx.toFixed(4)}px p95=${rL.p95AbsErrorModelPx.toFixed(3)} Bad1=${(rL.bad1Rate * 100).toFixed(2)}% Bad3=${(rL.bad3Rate * 100).toFixed(2)}% inv=${rL.invalidPredCount}\n` +
            `    M1  Left nonocc(n=${mL.evaluatedCount}): EPE=${mL.epeModelPx.toFixed(4)}px p95=${mL.p95AbsErrorModelPx.toFixed(3)} Bad1=${(mL.bad1Rate * 100).toFixed(2)}% Bad3=${(mL.bad3Rate * 100).toFixed(2)}% inv=${mL.invalidPredCount}\n` +
            `    RAW Right nonocc(n=${rR.evaluatedCount}): EPE=${rR.epeModelPx.toFixed(4)}px p95=${rR.p95AbsErrorModelPx.toFixed(3)} Bad1=${(rR.bad1Rate * 100).toFixed(2)}% Bad3=${(rR.bad3Rate * 100).toFixed(2)}% inv=${rR.invalidPredCount}\n` +
            `    M1  Right nonocc(n=${mR.evaluatedCount}): EPE=${mR.epeModelPx.toFixed(4)}px p95=${mR.p95AbsErrorModelPx.toFixed(3)} Bad1=${(mR.bad1Rate * 100).toFixed(2)}% Bad3=${(mR.bad3Rate * 100).toFixed(2)}% inv=${mR.invalidPredCount}`
        );
      }

      rawVsM1ByScene[sceneId] = {
        sceneId,
        edition: assets.spec.edition,
        g0Transform: g0,
        leftSupport: leftSet.support,
        rightSupport: rightSet.support,
        byFeed,
      };
    }

    rawComparisonExecuted = true;
  }

  // Synthesize data-driven summary & next smallest step
  let rawVsM1Verdict: string;
  let anyRawDiff = false;
  if (!rawComparisonExecuted) {
    rawVsM1Verdict = `RAW_COMPARISON_STOPPED: ${rawComparisonAbortedReason ?? 'Provenance/contract failed'}`;
  } else {
    let worstMaxDiff = 0;
    for (const sId of sceneIds) {
      const sc = rawVsM1ByScene[sId];
      if (!sc) continue;
      for (const ch of ['C0_RGB', 'C1_BGR'] as const) {
        const fp = sc.byFeed[ch].fullTensorParityRawVsM1;
        worstMaxDiff = Math.max(
          worstMaxDiff,
          fp.reference_output_disparity.maxAbsDiff,
          fp.secondary_output_disparity.maxAbsDiff
        );
        if (fp.reference_output_disparity.maxAbsDiff > 1e-4 || fp.secondary_output_disparity.maxAbsDiff > 1e-4) {
          anyRawDiff = true;
        }
      }
    }
    rawVsM1Verdict = anyRawDiff
      ? `RAW_VS_M1_DIVERGED: maxAbsDiff=${worstMaxDiff.toExponential(3)} between stereodemo raw release and canonical M1 on WASM.`
      : `RAW_VS_M1_MATCHED: maxAbsDiff=${worstMaxDiff.toExponential(3)} across both Motorcycle & Teddy RGB/BGR WASM runs; canonical M1 preserves raw HITNet release accuracy.`;
  }

  const mcycleLeftFull = sourceGtConsistencyByScene.Motorcycle.rois.find(
    (r) => r.roiId === 'motorcycle_full' && r.eye === 'left'
  )!;
  const mcycleLeftUL = sourceGtConsistencyByScene.Motorcycle.rois.find(
    (r) => r.roiId === 'motorcycle_upper_left_trouble' && r.eye === 'left'
  )!;
  const mcycleLeftCtrl = sourceGtConsistencyByScene.Motorcycle.rois.find(
    (r) => r.roiId === 'motorcycle_control_center' && r.eye === 'left'
  )!;
  const teddyLeftFull = sourceGtConsistencyByScene.Teddy.rois.find(
    (r) => r.roiId === 'teddy_full' && r.eye === 'left'
  )!;

  const gtRecipHealthy =
    mcycleLeftFull.nearestRecipAbsMeanPx < 0.2 &&
    mcycleLeftUL.nearestRecipAbsMeanPx < 0.2 &&
    mcycleLeftCtrl.nearestRecipAbsMeanPx < 0.2 &&
    teddyLeftFull.nearestRecipAbsMeanPx < 0.2 &&
    mcycleLeftFull.signAndUnitCheck.confirmedUnitAndSign &&
    teddyLeftFull.signAndUnitCheck.confirmedUnitAndSign;

  const sourceGtConsistencyVerdict = gtRecipHealthy
    ? `SOURCE_GT_CONSISTENT: Reciprocal L/R mean diff <= 0.2px across Motorcycle full (${mcycleLeftFull.nearestRecipAbsMeanPx.toFixed(4)}px), upper-left ROI (${mcycleLeftUL.nearestRecipAbsMeanPx.toFixed(4)}px), control ROI (${mcycleLeftCtrl.nearestRecipAbsMeanPx.toFixed(4)}px), and Teddy (${teddyLeftFull.nearestRecipAbsMeanPx.toFixed(4)}px), with unit & positive-sign ZNCC confirmed on both eyes.`
    : `SOURCE_GT_ANOMALY_DETECTED: Inspect reciprocal disparity / ZNCC unit-sign metrics across ROIs.`;

  const smallestNextExperiment = !rawComparisonExecuted
    ? 'Establish verified stereodemo v0.1-hitnet raw ONNX provenance via proxy before any weight or resolution follow-up.'
    : anyRawDiff
      ? 'Compare M1 subgraphs (#0..#7 vs level_init/init_to_prop/prop0) against raw stereodemo ONNX nodes on Motorcycle G0_RGB WASM to isolate which M1 rewrite shifted accuracy.'
      : 'Native 1.0x resolution (320x240) center crop control on Motorcycle (741x497 -> 1.0x crop vs 2.32x decimation) and Teddy (450x375 -> 1.0x crop) on WASM/WebGPU to isolate spatial downsampling / thin-structure loss and SceneFlow zero-shot domain gap from model/GT correctness.';

  log(
    `\n=== Diagnostic Summary ===\n` +
      `  Raw vs M1 Verdict: ${rawVsM1Verdict}\n` +
      `  Source-GT Consistency Verdict: ${sourceGtConsistencyVerdict}\n` +
      `  Smallest Next Step: ${smallestNextExperiment}`
  );

  return {
    ok: true,
    pinnedOrtVersion: '1.30.0',
    runtimeAssetFamily: 'asyncify',
    graphOptimizationLevel: 'basic',
    rawProvenance,
    rawComparisonExecuted,
    rawComparisonAbortedReason,
    m1ByteLength,
    m1Sha256,
    n1OffProof,
    rawVsM1ByScene,
    sourceGtConsistencyByScene,
    diagnosisSummary: {
      rawVsM1Verdict,
      sourceGtConsistencyVerdict,
      smallestNextExperiment,
    },
  };
}
