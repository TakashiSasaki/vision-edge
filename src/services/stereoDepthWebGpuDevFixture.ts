import {
  assertLoadedOrtVersionMatches,
  HITNET_M1_REQUIRED_FINAL_OUTPUTS,
  HITNET_ORT_130_FAMILY_PROFILES,
  type OrtLoadedAssetProvenance,
} from './hitnetOrtComparisonHelper';
import {
  HITNET_M1_FULL_GRAPH_SHA256,
  HITNET_RAW_MODEL_SHA256,
  type GpuKernelProfileSummary,
  type StereoModelProfile,
  type StereoWorkerRequest,
  type StereoWorkerResponse,
} from './stereoDepthWorkerCore';

export interface DevFixtureWorkerLike {
  postMessage: (message: StereoWorkerRequest, transfer?: Transferable[]) => void;
  onmessage: ((event: MessageEvent<StereoWorkerResponse>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  terminate: () => void;
}

export const HITNET_DEV_NON_JSEP_FIXTURE_SPEC = {
  pinnedOrtVersion: '1.30.0' as const,
  runtimeAssetFamily: 'asyncify' as const,
  packageExport: 'onnxruntime-web@1.30.0/webgpu',
  distBaseUrl: HITNET_ORT_130_FAMILY_PROFILES.asyncify.distBaseUrl,
  entryFile: HITNET_ORT_130_FAMILY_PROFILES.asyncify.entryFile,
  emscriptenMjsFile: HITNET_ORT_130_FAMILY_PROFILES.asyncify.emscriptenMjsFile,
  wasmFile: HITNET_ORT_130_FAMILY_PROFILES.asyncify.wasmFile,
  expectedEntrySha256: HITNET_ORT_130_FAMILY_PROFILES.asyncify.referenceLocalDistSha256.entry,
  expectedEmscriptenMjsSha256:
    HITNET_ORT_130_FAMILY_PROFILES.asyncify.referenceLocalDistSha256.emscriptenMjs,
  expectedWasmSha256: HITNET_ORT_130_FAMILY_PROFILES.asyncify.referenceLocalDistSha256.wasm,
  expectedRawSha256: HITNET_RAW_MODEL_SHA256,
  expectedM1ByteLength: 8888846,
  expectedM1Sha256: HITNET_M1_FULL_GRAPH_SHA256,
  expectedF0102FeedSha256: '49427b336f1f946db75fbf67402c9a7a217698890e4a8cc04ea75f481029ba29',
  expectedF0126FeedSha256: '7b91b078655e9be0e2460dff5146337de6124b05a0562b3696d7069fe6810e93',
  expectedFeedShape: [1, 6, 240, 320] as const,
  expectedFeedDataType: 'float32' as const,
  expectedOutputShape: [1, 240, 320, 1] as const,
  requiredOutputs: HITNET_M1_REQUIRED_FINAL_OUTPUTS,
} as const;

export const DEFAULT_HITNET_DEV_FIXTURE_PROFILE: StereoModelProfile = {
  variantId: 'hitnet-sf',
  tier: 'lightweight',
  engineType: 'onnx',
  fileName: 'hitnet_sceneflow_240x320.onnx',
  defaultWidth: 320,
  defaultHeight: 240,
  inputLayout: 'stacked',
  normalization: 'zero-one',
  preferWebGpu: true,
  license: 'Apache-2.0',
  licenseNote: 'HITNet SceneFlow DEV-ONLY non-JSEP app worker fixture',
  upstreamUrl: 'https://github.com/nburrus/stereodemo/releases/tag/v0.1-hitnet',
  onlineDelivery: 'cors-proxy',
};

export function assertDevFixtureEnabled(isDev?: boolean): void {
  const allowed = isDev ?? Boolean(import.meta.env.DEV);
  if (!allowed) {
    throw new Error(
      '[dev-gate] DEV-ONLY HITNet non-JSEP WebGPU app worker fixture is disabled outside development builds.'
    );
  }
}

export function createDevOnlyHitnetNonJsepAppWorker(isDev?: boolean): DevFixtureWorkerLike {
  assertDevFixtureEnabled(isDev);
  return new Worker(new URL('./stereoDepthWebGpuDevFixtureWorker.ts', import.meta.url), {
    type: 'module',
  }) as unknown as DevFixtureWorkerLike;
}

export interface DevAppWorkerEstimateOutput {
  name: string;
  dims: number[];
  dataType: string;
  data: Float32Array;
}

export interface DevAppWorkerRunRecord {
  runIndex: number;
  label: 'cold' | `warm-${number}`;
  wallMs: number;
  runMs: number;
  preprocessMs: number;
  postprocessMs: number;
  configuredProviders: readonly string[];
  fallbackReason: string | null;
  feedSha256s: Record<string, string>;
  feedMetadata: Record<string, { dims: number[]; dataType: string }>;
  fetchedOutputNames: string[];
  gpuValidationErrors: string[];
  gpuKernelProfileSummary: GpuKernelProfileSummary | null;
  outputs: Record<string, DevAppWorkerEstimateOutput>;
}

type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

export interface DevAppWorkerFixtureRunParams {
  m1Bytes: Uint8Array;
  leftRgbaBuffer: ArrayBuffer | Uint8ClampedArray;
  rightRgbaBuffer: ArrayBuffer | Uint8ClampedArray;
  profile?: StereoModelProfile;
  expectedM1Sha256?: string;
  expectedFeedSha256?: string;
  width?: number;
  height?: number;
  sourceEyeWidth?: number;
  sourceEyeHeight?: number;
  warmRuns?: number;
  enableWebGpuProfiling?: boolean;
  disableCpuEpFallback?: boolean;
  timeoutMs?: number;
  isDev?: boolean;
  workerFactory?: () => DevFixtureWorkerLike;
  sha256Hex?: (bytes: Uint8Array) => Promise<string>;
}

export type DevAppWorkerFixtureResult =
  | {
      ok: true;
      pinnedOrtVersion: '1.30.0';
      runtimeAssetFamily: 'asyncify';
      entryModuleUrl: string;
      wasmPaths: string;
      ortVersions: { common?: string; web?: string } | null;
      loadedOrtResources: string[];
      foreignOrtResources: string[];
      loadedAssetProvenance: OrtLoadedAssetProvenance[];
      workerGpuAdapter: string | null;
      configuredProviders: readonly string[];
      fallbackReason: string | null;
      sessionOptionsApplied: Record<string, unknown> | null;
      rawInputModelSha256: string;
      baseSessionModelSha256: string;
      sessionModelSha256: string;
      modelByteLength: number;
      sessionModelByteLength: number;
      pretransformedExactM1SkipConcatSplit: boolean;
      splitCount: number;
      initPrepMs: number;
      initSessionCreateMs: number;
      initTotalWallMs: number;
      coldRun: DevAppWorkerRunRecord;
      warmRuns: DevAppWorkerRunRecord[];
      lastRun: DevAppWorkerRunRecord;
    }
  | {
      ok: false;
      stage: string;
      message: string;
      ortVersions?: { common?: string; web?: string } | null;
      loadedOrtResources?: string[];
      foreignOrtResources?: string[];
      loadedAssetProvenance?: OrtLoadedAssetProvenance[];
      configuredProviders?: readonly string[];
      fallbackReason?: string | null;
      sessionModelSha256?: string | null;
    };

async function computeSha256Hex(bytes: Uint8Array): Promise<string> {
  if (typeof crypto === 'undefined' || !crypto.subtle) {
    throw new Error('crypto.subtle unavailable');
  }
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const digest = await crypto.subtle.digest('SHA-256', copy.buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

function toArrayBufferCopy(src: ArrayBuffer | Uint8ClampedArray): ArrayBuffer {
  if (src instanceof Uint8ClampedArray) {
    const out = new Uint8ClampedArray(src.byteLength);
    out.set(src);
    return out.buffer;
  }
  return src.slice(0);
}

function inferFailureStage(workerStage: string | undefined, message: string): string {
  const m = message.match(/^\[(dev-gate|opt-in-guard|webgpu-support|version-guard|resource-guard|fallback-guard)\]\s*/);
  if (m) return m[1];
  return workerStage || 'worker';
}

export async function runDevOnlyHitnetAppWorkerFixture(
  params: DevAppWorkerFixtureRunParams
): Promise<DevAppWorkerFixtureResult> {
  try {
    assertDevFixtureEnabled(params.isDev);
  } catch (err) {
    return {
      ok: false,
      stage: 'dev-gate',
      message: err instanceof Error ? err.message : String(err),
    };
  }

  const sha256 = params.sha256Hex ?? computeSha256Hex;
  const expectedM1Sha = params.expectedM1Sha256 ?? HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1Sha256;
  const m1ActualSha = await sha256(params.m1Bytes);
  if (
    params.m1Bytes.byteLength !== HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength ||
    m1ActualSha !== expectedM1Sha
  ) {
    return {
      ok: false,
      stage: 'input',
      message: `M1 model bytes/SHA mismatch before worker launch: bytes=${params.m1Bytes.byteLength} (expected ${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength}), sha=${m1ActualSha} (expected ${expectedM1Sha})`,
    };
  }
  const isOfficialExactM1 =
    expectedM1Sha === HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1Sha256 &&
    m1ActualSha === HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1Sha256;

  const profile = params.profile ?? DEFAULT_HITNET_DEV_FIXTURE_PROFILE;
  const width = params.width ?? 320;
  const height = params.height ?? 240;
  const sourceEyeWidth = params.sourceEyeWidth ?? 480;
  const sourceEyeHeight = params.sourceEyeHeight ?? 270;
  const warmRunsCount = Math.max(0, Math.min(5, Math.trunc(params.warmRuns ?? 1)));
  const enableWebGpuProfiling = Boolean(params.enableWebGpuProfiling);
  const disableCpuEpFallback = Boolean(params.disableCpuEpFallback);
  const timeoutMs = Math.max(5000, Math.trunc(params.timeoutMs ?? 120000));

  const worker = params.workerFactory
    ? params.workerFactory()
    : createDevOnlyHitnetNonJsepAppWorker(params.isDev);
  let nextId = 1;
  let workerCrashedOrTimedOut = false;

  const call = <T extends StereoWorkerResponse>(
    payload: DistributiveOmit<StereoWorkerRequest, 'id'>,
    transfer: Transferable[] = [],
    overrideTimeoutMs = timeoutMs
  ): Promise<T> =>
    new Promise<T>((resolve, reject) => {
      const id = nextId++;
      let settled = false;
      const timerId = setTimeout(() => {
        if (settled) return;
        settled = true;
        workerCrashedOrTimedOut = true;
        worker.onmessage = null;
        worker.onerror = null;
        reject(new Error(`[timeout] Worker request id=${id} (${payload.type}) timed out after ${overrideTimeoutMs}ms`));
      }, overrideTimeoutMs);

      const cleanup = () => {
        clearTimeout(timerId);
        worker.onmessage = null;
        worker.onerror = null;
      };

      worker.onmessage = (ev: MessageEvent<StereoWorkerResponse>) => {
        const data = ev.data;
        if (!data || data.id !== id) return;
        if (settled) return;
        settled = true;
        cleanup();
        if (data.type === 'error') {
          reject(
            new Error(data.message, {
              cause: { stage: inferFailureStage(data.stage, data.message) },
            })
          );
          return;
        }
        resolve(data as T);
      };

      worker.onerror = (ev: ErrorEvent) => {
        if (settled) return;
        settled = true;
        workerCrashedOrTimedOut = true;
        cleanup();
        reject(
          new Error(ev?.message || 'DEV WebGPU app fixture worker crashed', {
            cause: { stage: 'worker-error' },
          })
        );
      };

      worker.postMessage({ ...payload, id } as StereoWorkerRequest, transfer);
    });

  let initRes: Extract<StereoWorkerResponse, { type: 'init-result' }> | null = null;
  try {
    const onnxCopy = new Uint8Array(params.m1Bytes.byteLength);
    onnxCopy.set(params.m1Bytes);
    const tInit0 = performance.now();
    initRes = await call<Extract<StereoWorkerResponse, { type: 'init-result' }>>(
      {
        type: 'init-onnx',
        profile,
        onnxBuffer: onnxCopy.buffer,
        wasmPrimaryUrl: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl,
        wasmFallbackUrl: null,
        backendPreference: 'webgpu',
        normalizeSliceAxes: 'off',
        normalizeGatherNdRank: false,
        normalizePropGatherNdRank: false,
        levelInitGatherMode: 'off',
        normalizeInitToPropZeroSlant: false,
        normalizeSlot0ReshapeToTranspose: false,
        normalizeArgMinKeepDims: false,
        allowDiagnosticTransformedWasm: false,
        skipImplicitConcatSplitForExactM1Fixture: isOfficialExactM1,
        strictWebGpu: true,
        graphOptimizationLevel: 'basic',
        disableCpuEpFallback,
        enableWebGpuProfiling,
      },
      [onnxCopy.buffer]
    );
    const initTotalWallMs = performance.now() - tInit0;

    if (
      initRes.rawInputModelSha256 !== expectedM1Sha ||
      initRes.baseSessionModelSha256 !== expectedM1Sha ||
      initRes.sessionModelSha256 !== expectedM1Sha
    ) {
      return {
        ok: false,
        stage: 'model-guard',
        message: `App worker modified or mismatched M1 model SHA: raw=${initRes.rawInputModelSha256}, base=${initRes.baseSessionModelSha256}, session=${initRes.sessionModelSha256}, expected=${expectedM1Sha}`,
        ortVersions: initRes.ortVersions ?? null,
        loadedOrtResources: initRes.loadedOrtResources,
        foreignOrtResources: initRes.foreignOrtResources,
        loadedAssetProvenance: initRes.loadedAssetProvenance,
        configuredProviders: initRes.backend.configuredProviders,
        fallbackReason: initRes.backend.fallbackReason,
        sessionModelSha256: initRes.sessionModelSha256 ?? null,
      };
    }

    if (
      (initRes.transformStats?.splitCount ?? 0) !== 0 ||
      (initRes.transformStats?.decomposed5dCount ?? 0) !== 0 ||
      (isOfficialExactM1 &&
        (initRes.sessionModelByteLength !== HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength ||
          initRes.sessionOptionsApplied?.pretransformedExactM1SkipConcatSplit !== true))
    ) {
      return {
        ok: false,
        stage: 'model-guard',
        message: `Expected zero in-worker graph rewrites and preserved exact M1 bytes (${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength} B) on exact M1, got sessionModelByteLength=${String(initRes.sessionModelByteLength)}, pretransformedSkip=${String(initRes.sessionOptionsApplied?.pretransformedExactM1SkipConcatSplit)}, transformStats=${JSON.stringify(initRes.transformStats)}`,
        sessionModelSha256: initRes.sessionModelSha256 ?? null,
      };
    }

    try {
      assertLoadedOrtVersionMatches(
        initRes.ortVersions ?? null,
        HITNET_DEV_NON_JSEP_FIXTURE_SPEC.pinnedOrtVersion
      );
    } catch (verErr) {
      return {
        ok: false,
        stage: 'version-guard',
        message: verErr instanceof Error ? verErr.message : String(verErr),
        ortVersions: initRes.ortVersions ?? null,
        loadedOrtResources: initRes.loadedOrtResources,
        foreignOrtResources: initRes.foreignOrtResources,
        loadedAssetProvenance: initRes.loadedAssetProvenance,
      };
    }

    if (
      initRes.backend.configuredProviders.length !== 1 ||
      initRes.backend.configuredProviders[0] !== 'webgpu' ||
      initRes.backend.fallbackReason !== null
    ) {
      return {
        ok: false,
        stage: 'fallback-guard',
        message: `Unexpected backend fallback/providers on init: configuredProviders=${JSON.stringify(initRes.backend.configuredProviders)}, fallbackReason=${String(initRes.backend.fallbackReason)}`,
        ortVersions: initRes.ortVersions ?? null,
        configuredProviders: initRes.backend.configuredProviders,
        fallbackReason: initRes.backend.fallbackReason,
      };
    }

    const foreignResources = initRes.foreignOrtResources ?? [];
    const loadedResources = initRes.loadedOrtResources ?? [];
    const loadedProvenance = initRes.loadedAssetProvenance ?? [];
    const hasUnexpectedJsep =
      loadedResources.some((u) => u.includes('.jsep.') || u.includes('ort.bundle.')) ||
      loadedProvenance.some(
        (p) => p.url.includes('.jsep.') || p.url.includes('ort.bundle.')
      );
    if (foreignResources.length > 0 || hasUnexpectedJsep) {
      return {
        ok: false,
        stage: 'resource-guard',
        message: `Unexpected foreign or JSEP resources loaded: foreign=${JSON.stringify(foreignResources)}, loaded=${JSON.stringify(loadedResources)}`,
        ortVersions: initRes.ortVersions ?? null,
        loadedOrtResources: loadedResources,
        foreignOrtResources: foreignResources,
        loadedAssetProvenance: loadedProvenance,
      };
    }

    const executeRun = async (
      runIndex: number,
      label: DevAppWorkerRunRecord['label']
    ): Promise<DevAppWorkerRunRecord> => {
      const lBuf = toArrayBufferCopy(params.leftRgbaBuffer);
      const rBuf = toArrayBufferCopy(params.rightRgbaBuffer);
      const tRun0 = performance.now();
      const estRes = await call<Extract<StereoWorkerResponse, { type: 'estimate-result' }>>(
        {
          type: 'estimate',
          width,
          height,
          sourceEyeWidth,
          sourceEyeHeight,
          leftRgbaBuffer: lBuf,
          rightRgbaBuffer: rBuf,
          options: { autoNormalizeDisparity: true, colormap: 'turbo' },
          includeRawTensorOutputs: true,
          fetchPrimaryOutputOnly: false,
          captureGpuValidationErrors: true,
        },
        [lBuf, rBuf]
      );
      const wallMs = performance.now() - tRun0;

      if (
        estRes.backend.configuredProviders.length !== 1 ||
        estRes.backend.configuredProviders[0] !== 'webgpu' ||
        estRes.backend.fallbackReason !== null
      ) {
        throw new Error(
          `[fallback-guard] Runtime provider fallback detected during ${label}: providers=${JSON.stringify(estRes.backend.configuredProviders)}, fallbackReason=${String(estRes.backend.fallbackReason)}`
        );
      }

      const feedSha256s = estRes.feedSha256s ?? {};
      const feedMetadata = estRes.feedMetadata ?? {};
      const actualFeedSha = feedSha256s.input;
      if (params.expectedFeedSha256 && actualFeedSha !== params.expectedFeedSha256) {
        throw new Error(
          `[input] Feed SHA-256 mismatch on ${label}: got=${String(actualFeedSha)} expected=${params.expectedFeedSha256}`
        );
      }
      const inputMeta = feedMetadata.input;
      if (
        !inputMeta ||
        inputMeta.dataType !== HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedFeedDataType ||
        JSON.stringify(inputMeta.dims) !==
          JSON.stringify(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedFeedShape)
      ) {
        throw new Error(
          `[input] Feed tensor metadata mismatch on ${label}: got=${JSON.stringify(inputMeta)}, expected=float32 ${JSON.stringify(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedFeedShape)}`
        );
      }

      const outputs: Record<string, DevAppWorkerEstimateOutput> = {};
      for (const ro of estRes.rawOutputs ?? []) {
        outputs[ro.name] = {
          name: ro.name,
          dims: ro.dims,
          dataType: ro.dataType ?? 'float32',
          data: new Float32Array(ro.buffer),
        };
      }
      for (const reqOut of HITNET_DEV_NON_JSEP_FIXTURE_SPEC.requiredOutputs) {
        const got = outputs[reqOut];
        if (
          !got ||
          got.dataType !== 'float32' ||
          JSON.stringify(got.dims) !==
            JSON.stringify(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedOutputShape) ||
          got.data.length !== 76800
        ) {
          throw new Error(
            `[output] Invalid or missing HITNet output "${reqOut}" on ${label}: ${got ? `${got.dataType} ${JSON.stringify(got.dims)} len=${got.data.length}` : 'missing'}`
          );
        }
      }

      return {
        runIndex,
        label,
        wallMs,
        runMs: estRes.estimateTiming?.runMs ?? wallMs,
        preprocessMs: estRes.estimateTiming?.preprocessMs ?? 0,
        postprocessMs: estRes.estimateTiming?.postprocessMs ?? 0,
        configuredProviders: estRes.backend.configuredProviders,
        fallbackReason: estRes.backend.fallbackReason,
        feedSha256s,
        feedMetadata,
        fetchedOutputNames: estRes.fetchedOutputNames ?? Object.keys(outputs),
        gpuValidationErrors: estRes.gpuValidationErrors ?? [],
        gpuKernelProfileSummary: estRes.gpuKernelProfileSummary ?? null,
        outputs,
      };
    };

    const coldRun = await executeRun(0, 'cold');
    const warmRuns: DevAppWorkerRunRecord[] = [];
    for (let i = 1; i <= warmRunsCount; i++) {
      warmRuns.push(await executeRun(i, `warm-${i}`));
    }
    const lastRun = warmRuns.length > 0 ? warmRuns[warmRuns.length - 1] : coldRun;

    return {
      ok: true,
      pinnedOrtVersion: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.pinnedOrtVersion,
      runtimeAssetFamily: 'asyncify',
      entryModuleUrl:
        initRes.entryModuleUrl ??
        `${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl}${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.entryFile}`,
      wasmPaths:
        initRes.sessionOptionsApplied?.wasmPaths ?? HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl,
      ortVersions: initRes.ortVersions ?? null,
      loadedOrtResources: loadedResources,
      foreignOrtResources: foreignResources,
      loadedAssetProvenance: loadedProvenance,
      workerGpuAdapter: initRes.workerGpuAdapter ?? null,
      configuredProviders: initRes.backend.configuredProviders,
      fallbackReason: initRes.backend.fallbackReason,
      sessionOptionsApplied: (initRes.sessionOptionsApplied as unknown as Record<string, unknown>) ?? null,
      rawInputModelSha256: initRes.rawInputModelSha256 ?? expectedM1Sha,
      baseSessionModelSha256: initRes.baseSessionModelSha256 ?? expectedM1Sha,
      sessionModelSha256: initRes.sessionModelSha256 ?? expectedM1Sha,
      modelByteLength: params.m1Bytes.byteLength,
      sessionModelByteLength: initRes.sessionModelByteLength ?? params.m1Bytes.byteLength,
      pretransformedExactM1SkipConcatSplit: Boolean(
        initRes.sessionOptionsApplied?.pretransformedExactM1SkipConcatSplit
      ),
      splitCount: initRes.transformStats?.splitCount ?? 0,
      initPrepMs: initRes.initTiming?.prepMs ?? 0,
      initSessionCreateMs: initRes.initTiming?.sessionCreateMs ?? initTotalWallMs,
      initTotalWallMs,
      coldRun,
      warmRuns,
      lastRun,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const causeStage =
      err instanceof Error &&
      err.cause &&
      typeof err.cause === 'object' &&
      'stage' in err.cause &&
      typeof (err.cause as { stage?: unknown }).stage === 'string'
        ? (err.cause as { stage: string }).stage
        : undefined;
    return {
      ok: false,
      stage: inferFailureStage(causeStage, message),
      message,
      ortVersions: initRes?.ortVersions ?? null,
      loadedOrtResources: initRes?.loadedOrtResources,
      foreignOrtResources: initRes?.foreignOrtResources,
      loadedAssetProvenance: initRes?.loadedAssetProvenance,
      configuredProviders: initRes?.backend.configuredProviders,
      fallbackReason: initRes?.backend.fallbackReason ?? null,
      sessionModelSha256: initRes?.sessionModelSha256 ?? null,
    };
  } finally {
    if (!workerCrashedOrTimedOut) {
      try {
        await call({ type: 'dispose' }, [], Math.min(2000, timeoutMs));
      } catch {
        // Best-effort dispose before terminating worker
      }
    }
    try {
      worker.terminate();
    } catch {
      // Ignore worker terminate error
    }
  }
}
