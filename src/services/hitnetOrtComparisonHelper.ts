export type OrtDiagnosticVersion = '1.30.0' | '1.31.1';

export interface OrtVersionProfile {
  version: OrtDiagnosticVersion;
  distBaseUrl: string;
  entryFile: 'ort.webgpu.min.mjs';
  expectedAssetFiles: readonly [
    'ort.webgpu.min.mjs',
    'ort-wasm-simd-threaded.jsep.mjs',
    'ort-wasm-simd-threaded.jsep.wasm',
    'ort-wasm-simd-threaded.mjs',
    'ort-wasm-simd-threaded.wasm',
  ];
}

export const HITNET_ORT_VERSION_PROFILES: Readonly<Record<OrtDiagnosticVersion, OrtVersionProfile>> = {
  '1.30.0': {
    version: '1.30.0',
    distBaseUrl: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
    entryFile: 'ort.webgpu.min.mjs',
    expectedAssetFiles: [
      'ort.webgpu.min.mjs',
      'ort-wasm-simd-threaded.jsep.mjs',
      'ort-wasm-simd-threaded.jsep.wasm',
      'ort-wasm-simd-threaded.mjs',
      'ort-wasm-simd-threaded.wasm',
    ],
  },
  '1.31.1': {
    version: '1.31.1',
    distBaseUrl: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/',
    entryFile: 'ort.webgpu.min.mjs',
    expectedAssetFiles: [
      'ort.webgpu.min.mjs',
      'ort-wasm-simd-threaded.jsep.mjs',
      'ort-wasm-simd-threaded.jsep.wasm',
      'ort-wasm-simd-threaded.mjs',
      'ort-wasm-simd-threaded.wasm',
    ],
  },
};

export const HITNET_M1_REQUIRED_FINAL_OUTPUTS = [
  'reference_output_disparity',
  'secondary_output_disparity',
] as const;

export interface OrtVersionDiagnosticRequest {
  id: number;
  expectedOrtVersion: OrtDiagnosticVersion;
  modelBuffer: ArrayBuffer;
  feedBuffer: ArrayBuffer;
  feedName?: string;
  feedDims?: number[];
  provider: 'webgpu' | 'wasm';
  graphOptimizationLevel: 'disabled' | 'basic';
  warmRuns?: number;
  disableCpuEpFallback?: boolean;
  captureGpuValidationErrors?: boolean;
}

export interface OrtVersionDiagnosticOutput {
  name: string;
  dims: number[];
  dataType: string;
  buffer: ArrayBuffer;
}

export interface OrtVersionDiagnosticSuccess {
  id: number;
  ok: true;
  pinnedOrtVersion: OrtDiagnosticVersion;
  entryModuleUrl: string;
  wasmPaths: string;
  ortVersions: { common?: string; web?: string } | null;
  loadedOrtResources: string[];
  foreignOrtResources: string[];
  provider: 'webgpu' | 'wasm';
  configuredProviders: string[];
  fallbackBehavior: 'no-fallback-explicit-error';
  graphOptimizationLevel: 'disabled' | 'basic';
  modelSha256: string;
  feedSha256: string;
  workerGpuAdapter: string | null;
  gpuValidationErrors: string[] | null;
  createMs: number;
  coldRunMs: number;
  warmRunMs: number[];
  outputs: OrtVersionDiagnosticOutput[];
}

export interface OrtVersionDiagnosticFailure {
  id: number;
  ok: false;
  stage:
    | 'init'
    | 'version-guard'
    | 'resource-guard'
    | 'webgpu-support'
    | 'create'
    | 'run'
    | 'output'
    | 'timeout';
  message: string;
  pinnedOrtVersion: OrtDiagnosticVersion;
  entryModuleUrl: string;
  wasmPaths: string;
  ortVersions: { common?: string; web?: string } | null;
  loadedOrtResources: string[];
  foreignOrtResources: string[];
  provider: 'webgpu' | 'wasm';
  configuredProviders: string[];
  fallbackBehavior: 'no-fallback-explicit-error';
  graphOptimizationLevel: 'disabled' | 'basic';
  modelSha256?: string;
  feedSha256?: string;
  workerGpuAdapter?: string | null;
  gpuValidationErrors?: string[];
}

export type OrtVersionDiagnosticResponse =
  | OrtVersionDiagnosticSuccess
  | OrtVersionDiagnosticFailure;

export interface OrtLikeTensor {
  dims: readonly number[];
  type: string;
  data: Float32Array | BigInt64Array | Int32Array | Uint8Array | ArrayLike<number>;
  dispose?: () => void;
}

export interface OrtLikeSession {
  inputNames?: readonly string[];
  outputNames?: readonly string[];
  run: (feeds: Record<string, OrtLikeTensor>) => Promise<Record<string, OrtLikeTensor>>;
  release: () => Promise<void>;
}

export interface OrtLikeGpuDevice {
  pushErrorScope?: (filter: 'validation' | 'out-of-memory' | 'internal') => void;
  popErrorScope?: () => Promise<{ message?: string } | null>;
  addEventListener?: (type: string, cb: (ev: { error?: { message?: string } }) => void) => void;
  removeEventListener?: (type: string, cb: (ev: { error?: { message?: string } }) => void) => void;
}

export interface OrtLikeGpuAdapter {
  info?: { vendor?: string; architecture?: string; device?: string; description?: string };
  features?: { has?: (name: string) => boolean };
}

export interface OrtLikeModule {
  env: {
    logLevel?: string;
    versions?: { common?: string; web?: string };
    wasm: {
      numThreads?: number;
      proxy?: boolean;
      wasmPaths?: string;
    };
    webgpu?: {
      adapter?: OrtLikeGpuAdapter;
      device?: OrtLikeGpuDevice;
    };
  };
  Tensor: new (dtype: string, data: Float32Array, dims: number[]) => OrtLikeTensor;
  InferenceSession: {
    create: (bytes: Uint8Array, options: Record<string, unknown>) => Promise<OrtLikeSession>;
  };
}

export interface OrtWorkerExecutionDeps {
  loadOrtModule: () => Promise<OrtLikeModule>;
  sha256Hex?: (bytes: Uint8Array) => Promise<string>;
  getResourceUrls?: () => string[];
  requestWebGpuAdapter?: () => Promise<OrtLikeGpuAdapter | null>;
  now?: () => number;
}

export interface OrtWorkerExecutionResult {
  response: OrtVersionDiagnosticResponse;
  transfers: Transferable[];
}

async function defaultSha256Hex(u8: Uint8Array): Promise<string> {
  const copy = u8.slice();
  const digest = await crypto.subtle.digest('SHA-256', copy.buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

function defaultGetResourceUrls(): string[] {
  if (typeof performance === 'undefined' || typeof performance.getEntriesByType !== 'function') {
    return [];
  }
  return performance
    .getEntriesByType('resource')
    .map((entry) => entry.name)
    .filter((name): name is string => typeof name === 'string' && name.length > 0);
}

async function defaultRequestWebGpuAdapter(): Promise<OrtLikeGpuAdapter | null> {
  if (typeof navigator === 'undefined') return null;
  const gpu = (navigator as unknown as {
    gpu?: { requestAdapter?: (opts?: { powerPreference?: string }) => Promise<OrtLikeGpuAdapter | null> };
  }).gpu;
  if (!gpu || typeof gpu.requestAdapter !== 'function') return null;
  try {
    return await gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch {
    return null;
  }
}

function describeGpuAdapter(adapter: OrtLikeGpuAdapter | null | undefined): string | null {
  if (!adapter?.info) return null;
  const info = adapter.info;
  const hasF16 = adapter.features?.has?.('shader-f16') ?? false;
  const hasTq = adapter.features?.has?.('timestamp-query') ?? false;
  return `vendor=${info.vendor || '-'} arch=${info.architecture || '-'} device=${info.device || '-'} desc=${info.description || '-'} shader-f16=${hasF16} timestamp-query=${hasTq}`;
}

export function classifyOrtLoadedResources(
  resourceUrls: readonly string[],
  expectedDistBaseUrl: string
): { loadedOrtResources: string[]; foreignOrtResources: string[] } {
  const unique = Array.from(
    new Set(
      resourceUrls.filter(
        (u) =>
          u.includes('onnxruntime-web') ||
          u.includes('ort-wasm') ||
          u.includes('ort.webgpu') ||
          u.includes('ort.wasm') ||
          u.includes('ort.all')
      )
    )
  );
  const loadedOrtResources = unique.filter((u) => u.startsWith(expectedDistBaseUrl));
  const foreignOrtResources = unique.filter((u) => !u.startsWith(expectedDistBaseUrl));
  return { loadedOrtResources, foreignOrtResources };
}

function copyOrtFloat32OrTypedBuffer(tensor: OrtLikeTensor): ArrayBuffer {
  const data = tensor.data;
  if (data instanceof Float32Array) {
    return data.slice().buffer as ArrayBuffer;
  }
  if (data instanceof BigInt64Array) {
    return data.slice().buffer as ArrayBuffer;
  }
  if (data instanceof Int32Array) {
    return data.slice().buffer as ArrayBuffer;
  }
  if (data instanceof Uint8Array) {
    return data.slice().buffer as ArrayBuffer;
  }
  if (ArrayBuffer.isView(data)) {
    const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    return view.slice().buffer as ArrayBuffer;
  }
  return new Float32Array(Array.from(data as ArrayLike<number>, Number)).buffer as ArrayBuffer;
}

export function assertLoadedOrtVersionMatches(
  ortVersions: { common?: string; web?: string } | null | undefined,
  expectedVersion: OrtDiagnosticVersion
): void {
  const webVer = ortVersions?.web;
  const commonVer = ortVersions?.common;
  if (!webVer || webVer !== expectedVersion || (commonVer !== undefined && commonVer !== expectedVersion)) {
    throw new Error(
      `ORT version mismatch: expected=${expectedVersion}, observed=${JSON.stringify(ortVersions ?? null)}`
    );
  }
}

export async function executeOrtDiagnosticWorkerRun(
  pinnedVersion: OrtDiagnosticVersion,
  req: OrtVersionDiagnosticRequest,
  deps: OrtWorkerExecutionDeps
): Promise<OrtWorkerExecutionResult> {
  const profile = HITNET_ORT_VERSION_PROFILES[pinnedVersion];
  const entryModuleUrl = `${profile.distBaseUrl}${profile.entryFile}`;
  const configuredProviders = [req.provider];
  const now = deps.now ?? (() => performance.now());
  const sha256 = deps.sha256Hex ?? defaultSha256Hex;
  const getResourceUrls = deps.getResourceUrls ?? defaultGetResourceUrls;
  const requestAdapter = deps.requestWebGpuAdapter ?? defaultRequestWebGpuAdapter;

  const feedName = req.feedName || 'input';
  const feedDims = Array.isArray(req.feedDims) && req.feedDims.length > 0 ? req.feedDims : [1, 6, 240, 320];
  const warmRuns = Math.max(0, Math.min(5, Math.trunc(req.warmRuns ?? 1)));
  const captureValidation = (req.captureGpuValidationErrors ?? true) && req.provider === 'webgpu';

  let stage: OrtVersionDiagnosticFailure['stage'] = 'init';
  let ort: OrtLikeModule | null = null;
  let ortVersions: { common?: string; web?: string } | null = null;
  let workerGpuAdapter: string | null = null;
  let modelSha256: string | undefined;
  let feedSha256: string | undefined;
  let session: OrtLikeSession | null = null;
  const gpuValidationErrors: string[] = [];

  const fail = (message: string): OrtWorkerExecutionResult => {
    const resCheck = classifyOrtLoadedResources(getResourceUrls(), profile.distBaseUrl);
    return {
      response: {
        id: req.id,
        ok: false,
        stage,
        message,
        pinnedOrtVersion: pinnedVersion,
        entryModuleUrl,
        wasmPaths: profile.distBaseUrl,
        ortVersions,
        loadedOrtResources: resCheck.loadedOrtResources,
        foreignOrtResources: resCheck.foreignOrtResources,
        provider: req.provider,
        configuredProviders,
        fallbackBehavior: 'no-fallback-explicit-error',
        graphOptimizationLevel: req.graphOptimizationLevel,
        modelSha256,
        feedSha256,
        workerGpuAdapter,
        ...(gpuValidationErrors.length > 0 ? { gpuValidationErrors } : {}),
      },
      transfers: [],
    };
  };

  try {
    if (req.expectedOrtVersion !== pinnedVersion) {
      stage = 'version-guard';
      return fail(
        `Worker request expectedOrtVersion="${req.expectedOrtVersion}" does not match pinned worker version "${pinnedVersion}"`
      );
    }

    const modelBytes = new Uint8Array(req.modelBuffer);
    if (modelBytes.byteLength === 0) {
      stage = 'init';
      return fail('Model buffer is empty');
    }

    const feedF32 = new Float32Array(req.feedBuffer);
    const expectedElements = feedDims.reduce((acc, d) => acc * d, 1);
    if (feedF32.length !== expectedElements || expectedElements <= 0) {
      stage = 'init';
      return fail(
        `Feed Float32 length ${feedF32.length} does not match expected dims ${JSON.stringify(feedDims)} (${expectedElements})`
      );
    }

    modelSha256 = await sha256(modelBytes);
    feedSha256 = await sha256(new Uint8Array(feedF32.buffer, feedF32.byteOffset, feedF32.byteLength));

    let preAdapter: OrtLikeGpuAdapter | null = null;
    if (req.provider === 'webgpu') {
      stage = 'webgpu-support';
      preAdapter = await requestAdapter();
      if (!preAdapter) {
        return fail('WebGPU adapter unavailable in worker (explicit error, no CPU/WASM fallback)');
      }
      workerGpuAdapter = describeGpuAdapter(preAdapter);
    }

    stage = 'version-guard';
    ort = await deps.loadOrtModule();
    ortVersions = ort.env.versions
      ? { common: ort.env.versions.common, web: ort.env.versions.web }
      : null;
    assertLoadedOrtVersionMatches(ortVersions, pinnedVersion);

    ort.env.logLevel = 'error';
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = profile.distBaseUrl;

    if (ort.env.wasm.wasmPaths !== profile.distBaseUrl) {
      stage = 'resource-guard';
      return fail(`Failed to pin ort.env.wasm.wasmPaths to ${profile.distBaseUrl}`);
    }

    stage = 'create';
    const sessionOptions: Record<string, unknown> = {
      executionProviders: configuredProviders,
      graphOptimizationLevel: req.graphOptimizationLevel,
      enableCpuMemArena: req.provider === 'webgpu',
      enableMemPattern: req.provider === 'webgpu',
      logSeverityLevel: 3,
    };
    if (req.disableCpuEpFallback && req.provider === 'webgpu') {
      sessionOptions.extra = { session: { disable_cpu_ep_fallback: '1' } };
    }

    const tCreate0 = now();
    session = await ort.InferenceSession.create(modelBytes, sessionOptions);
    const createMs = now() - tCreate0;

    if (req.provider === 'webgpu') {
      workerGpuAdapter = describeGpuAdapter(ort.env.webgpu?.adapter ?? preAdapter) ?? workerGpuAdapter;
    }

    const afterCreateRes = classifyOrtLoadedResources(getResourceUrls(), profile.distBaseUrl);
    if (afterCreateRes.foreignOrtResources.length > 0) {
      stage = 'resource-guard';
      return fail(
        `Unexpected foreign ORT assets loaded during session create: ${afterCreateRes.foreignOrtResources.join(', ')}`
      );
    }

    const resolvedFeedName = session.inputNames?.[0] || feedName;
    const runOnce = async (): Promise<{
      runMs: number;
      result: Record<string, OrtLikeTensor>;
    }> => {
      const feedTensor = new ort!.Tensor('float32', feedF32, feedDims);
      const gpuDevice = captureValidation ? ort!.env.webgpu?.device : undefined;
      const onUncaptured = (ev: { error?: { message?: string } }) => {
        const msg = ev?.error?.message;
        if (msg) gpuValidationErrors.push(`uncaptured: ${msg}`);
      };
      let pushedScope = false;
      if (gpuDevice) {
        try {
          gpuDevice.addEventListener?.('uncapturederror', onUncaptured);
          gpuDevice.pushErrorScope?.('validation');
          pushedScope = typeof gpuDevice.popErrorScope === 'function';
        } catch {
          pushedScope = false;
        }
      }

      const tRun0 = now();
      try {
        const result = await session!.run({ [resolvedFeedName]: feedTensor });
        const runMs = now() - tRun0;
        return { runMs, result };
      } finally {
        try {
          feedTensor.dispose?.();
        } catch {
          // Ignore input tensor dispose error
        }
        if (pushedScope && gpuDevice?.popErrorScope) {
          try {
            const scopedErr = await gpuDevice.popErrorScope();
            if (scopedErr?.message) {
              gpuValidationErrors.push(`validation-scope: ${scopedErr.message}`);
            }
          } catch {
            // Ignore popErrorScope error
          }
        }
        if (gpuDevice) {
          try {
            gpuDevice.removeEventListener?.('uncapturederror', onUncaptured);
          } catch {
            // Ignore removeEventListener error
          }
        }
      }
    };

    const disposeResultTensors = (res: Record<string, OrtLikeTensor>) => {
      for (const tensor of Object.values(res)) {
        try {
          tensor.dispose?.();
        } catch {
          // Ignore output tensor dispose error
        }
      }
    };

    stage = 'run';
    const cold = await runOnce();
    const coldRunMs = cold.runMs;
    const warmRunMs: number[] = [];
    let finalResult = cold.result;

    for (let i = 0; i < warmRuns; i++) {
      disposeResultTensors(finalResult);
      const warm = await runOnce();
      warmRunMs.push(warm.runMs);
      finalResult = warm.result;
    }

    stage = 'output';
    const outputs: OrtVersionDiagnosticOutput[] = [];
    const transfers: Transferable[] = [];
    try {
      for (const reqName of HITNET_M1_REQUIRED_FINAL_OUTPUTS) {
        if (!finalResult[reqName]) {
          throw new Error(
            `Missing required HITNet final output "${reqName}" (present: ${Object.keys(finalResult).join(', ')})`
          );
        }
      }
      for (const [name, tensor] of Object.entries(finalResult)) {
        const buffer = copyOrtFloat32OrTypedBuffer(tensor);
        outputs.push({
          name,
          dims: Array.from(tensor.dims),
          dataType: tensor.type,
          buffer,
        });
        transfers.push(buffer);
      }
    } finally {
      disposeResultTensors(finalResult);
    }

    const finalResCheck = classifyOrtLoadedResources(getResourceUrls(), profile.distBaseUrl);
    if (finalResCheck.foreignOrtResources.length > 0) {
      stage = 'resource-guard';
      return fail(
        `Unexpected foreign ORT assets loaded during inference: ${finalResCheck.foreignOrtResources.join(', ')}`
      );
    }

    return {
      response: {
        id: req.id,
        ok: true,
        pinnedOrtVersion: pinnedVersion,
        entryModuleUrl,
        wasmPaths: profile.distBaseUrl,
        ortVersions,
        loadedOrtResources: finalResCheck.loadedOrtResources,
        foreignOrtResources: finalResCheck.foreignOrtResources,
        provider: req.provider,
        configuredProviders,
        fallbackBehavior: 'no-fallback-explicit-error',
        graphOptimizationLevel: req.graphOptimizationLevel,
        modelSha256,
        feedSha256,
        workerGpuAdapter,
        gpuValidationErrors: captureValidation ? gpuValidationErrors : null,
        createMs,
        coldRunMs,
        warmRunMs,
        outputs,
      },
      transfers,
    };
  } catch (err) {
    return fail(err instanceof Error ? err.message : String(err));
  } finally {
    if (session) {
      try {
        await session.release();
      } catch {
        // Ignore session release error
      }
    }
  }
}
