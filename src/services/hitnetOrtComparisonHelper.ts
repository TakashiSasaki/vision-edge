export type OrtDiagnosticVersion = '1.30.0' | '1.31.1';
export type OrtDiagnosticRuntimeFamily = 'asyncify' | 'jsep';

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

export interface OrtRuntimeFamilyProfile {
  family: OrtDiagnosticRuntimeFamily;
  version: '1.30.0';
  distBaseUrl: string;
  entryFile: 'ort.webgpu.min.mjs' | 'ort.bundle.min.mjs';
  emscriptenMjsFile: 'ort-wasm-simd-threaded.asyncify.mjs' | 'ort-wasm-simd-threaded.jsep.mjs';
  wasmFile: 'ort-wasm-simd-threaded.asyncify.wasm' | 'ort-wasm-simd-threaded.jsep.wasm';
  forbiddenAssetSubstrings: readonly string[];
  referenceLocalDistSha256: {
    entry: string;
    emscriptenMjs: string;
    wasm: string;
  };
  provenanceNote: string;
}

export const HITNET_ORT_130_FAMILY_PROFILES: Readonly<
  Record<OrtDiagnosticRuntimeFamily, OrtRuntimeFamilyProfile>
> = {
  asyncify: {
    family: 'asyncify',
    version: '1.30.0',
    distBaseUrl: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
    entryFile: 'ort.webgpu.min.mjs',
    emscriptenMjsFile: 'ort-wasm-simd-threaded.asyncify.mjs',
    wasmFile: 'ort-wasm-simd-threaded.asyncify.wasm',
    forbiddenAssetSubstrings: ['.jsep.'],
    referenceLocalDistSha256: {
      entry: '3dffff71811bc13a3a3d9591c57ca5ee5735b8f5f3eaf789510b02bb040ae3a0',
      emscriptenMjs: '3d1c85995364bb643302fc6fd877a0c3ba5ae72401815e0f24828a53d9191e28',
      wasm: '39f9f0894d478800487ed9f7dbe92618498db320cf55c8e3d89adff8dce658da',
    },
    provenanceNote:
      'Dedicated WebGPU package subpath export ("onnxruntime-web/webgpu" -> dist/ort.webgpu.min.mjs -> ort-wasm-simd-threaded.asyncify.mjs/.wasm).',
  },
  jsep: {
    family: 'jsep',
    version: '1.30.0',
    distBaseUrl: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
    entryFile: 'ort.bundle.min.mjs',
    emscriptenMjsFile: 'ort-wasm-simd-threaded.jsep.mjs',
    wasmFile: 'ort-wasm-simd-threaded.jsep.wasm',
    forbiddenAssetSubstrings: ['.asyncify.'],
    referenceLocalDistSha256: {
      entry: 'd5fd02577ab51ec31243809e146c81600cd614786a7f9bf848ea8efbfc07af1c',
      emscriptenMjs: '709853412fd1ffc34247af1e73569227b5b79629c5ca3f59cc39cf7e500e4947',
      wasm: '3ad23231b5bd6d9dda55a7f84606315e0bf35b6750c28ee993c987c54cacab0f',
    },
    provenanceNote:
      'Real app default export ("onnxruntime-web" package.json exports["."].import.default -> dist/ort.bundle.min.mjs) + string wasmPaths loading ort-wasm-simd-threaded.jsep.mjs/.wasm.',
  },
};

export interface OrtLoadedAssetProvenance {
  role: 'entry' | 'wasm' | 'emscripten-mjs';
  url: string;
  byteLength: number | null;
  sha256: string | null;
  via:
    | 'fetch+blob-import'
    | 'direct-import-unhashed'
    | 'WebAssembly.instantiateStreaming'
    | 'WebAssembly.instantiate'
    | 'performance-resource-entry';
  referenceLocalDistSha256: string | null;
  matchesLocalDistSha256: boolean | null;
  note?: string;
}

export interface OrtDiagnosticSessionOptionsApplied {
  executionProviders: string[];
  graphOptimizationLevel: 'disabled' | 'basic';
  enableCpuMemArena: boolean;
  enableMemPattern: boolean;
  numThreads: number;
  proxy: boolean;
  wasmPaths: string;
  disableCpuEpFallback: boolean;
  fetchMode: 'all-session-outputs';
  profilingRequested: boolean;
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

export interface OrtGpuKernelProfileSummary {
  profilingRequested: boolean;
  timestampQuerySupported: boolean;
  ortProfilingEventCount: number;
  ortProfilingKernelTypes: Record<string, number>;
  ortProfilingSamplePrograms: string[];
  webgpuComputePassCount: number;
  webgpuDispatchWorkgroupsCount: number;
  webgpuDispatchIndirectCount: number;
  webgpuTimestampWritePassCount: number;
  webgpuQueueSubmitCount: number;
  webgpuCommandBufferCount: number;
  webgpuQuerySetCount: number;
  webgpuSamplePassLabels: string[];
  deviceObserverAttached: boolean;
  note: string;
}

export interface OrtVersionDiagnosticRequest {
  id: number;
  expectedOrtVersion: OrtDiagnosticVersion;
  runtimeAssetFamily?: OrtDiagnosticRuntimeFamily;
  modelBuffer: ArrayBuffer;
  feedBuffer: ArrayBuffer;
  feedName?: string;
  feedDims?: number[];
  extraFeeds?: Array<{
    name: string;
    dims: number[];
    buffer: ArrayBuffer;
    dataType?: 'float32' | 'int32' | 'int64' | 'uint8';
  }>;
  allowArbitraryOutputs?: boolean;
  provider: 'webgpu' | 'wasm';
  graphOptimizationLevel: 'disabled' | 'basic';
  warmRuns?: number;
  compareColdVsWarmRepeat?: boolean;
  enableWebGpuProfiling?: boolean;
  disableCpuEpFallback?: boolean;
  captureGpuValidationErrors?: boolean;
}

export interface OrtVersionDiagnosticOutput {
  name: string;
  dims: number[];
  dataType: string;
  buffer: ArrayBuffer;
}

export interface OrtDiagnosticExtraFeedCheck {
  name: string;
  dims: number[];
  dataType: 'float32' | 'int32' | 'int64' | 'uint8';
  byteOffset: number;
  byteLength: number;
  sha256: string;
}

export interface OrtVersionDiagnosticSuccess {
  id: number;
  ok: true;
  pinnedOrtVersion: OrtDiagnosticVersion;
  runtimeAssetFamily: OrtDiagnosticRuntimeFamily;
  entryModuleUrl: string;
  wasmPaths: string;
  ortVersions: { common?: string; web?: string } | null;
  loadedOrtResources: string[];
  foreignOrtResources: string[];
  loadedAssetProvenance: OrtLoadedAssetProvenance[];
  provider: 'webgpu' | 'wasm';
  configuredProviders: string[];
  fallbackBehavior: 'no-fallback-explicit-error';
  graphOptimizationLevel: 'disabled' | 'basic';
  sessionOptionsApplied: OrtDiagnosticSessionOptionsApplied;
  modelByteLength: number;
  modelSha256: string;
  feedSha256: string;
  feedTensorName: string;
  feedTensorShape: number[];
  feedTensorDataType: 'float32';
  feedByteOffset?: number;
  feedByteLength?: number;
  extraFeedChecks?: OrtDiagnosticExtraFeedCheck[];
  sessionInputNames: string[];
  sessionOutputNames: string[];
  workerGpuAdapter: string | null;
  gpuValidationErrors: string[] | null;
  gpuKernelProfileSummary: OrtGpuKernelProfileSummary | null;
  createMs: number;
  coldRunMs: number;
  warmRunMs: number[];
  outputs: OrtVersionDiagnosticOutput[];
  coldOutputs?: OrtVersionDiagnosticOutput[];
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
  runtimeAssetFamily: OrtDiagnosticRuntimeFamily;
  entryModuleUrl: string;
  wasmPaths: string;
  ortVersions: { common?: string; web?: string } | null;
  loadedOrtResources: string[];
  foreignOrtResources: string[];
  loadedAssetProvenance: OrtLoadedAssetProvenance[];
  provider: 'webgpu' | 'wasm';
  configuredProviders: string[];
  fallbackBehavior: 'no-fallback-explicit-error';
  graphOptimizationLevel: 'disabled' | 'basic';
  sessionOptionsApplied?: OrtDiagnosticSessionOptionsApplied;
  modelByteLength?: number;
  modelSha256?: string;
  feedSha256?: string;
  feedTensorName?: string;
  feedTensorShape?: number[];
  feedTensorDataType?: 'float32';
  sessionInputNames?: string[];
  sessionOutputNames?: string[];
  workerGpuAdapter?: string | null;
  gpuValidationErrors?: string[];
  gpuKernelProfileSummary?: OrtGpuKernelProfileSummary | null;
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
  startProfiling?: () => void;
  endProfiling?: () => void;
}

export interface OrtLikeGpuComputePass {
  dispatchWorkgroups?: (...args: unknown[]) => unknown;
  dispatchWorkgroupsIndirect?: (...args: unknown[]) => unknown;
}

export interface OrtLikeGpuCommandEncoder {
  beginComputePass?: (descriptor?: {
    label?: string;
    timestampWrites?: unknown;
  }) => OrtLikeGpuComputePass;
}

export interface OrtLikeGpuQueue {
  submit?: (commandBuffers: readonly unknown[]) => unknown;
}

export interface OrtLikeGpuDevice {
  pushErrorScope?: (filter: 'validation' | 'out-of-memory' | 'internal') => void;
  popErrorScope?: () => Promise<{ message?: string } | null>;
  addEventListener?: (type: string, cb: (ev: { error?: { message?: string } }) => void) => void;
  removeEventListener?: (type: string, cb: (ev: { error?: { message?: string } }) => void) => void;
  createCommandEncoder?: (...args: unknown[]) => OrtLikeGpuCommandEncoder;
  createQuerySet?: (descriptor?: { type?: string; count?: number }) => unknown;
  queue?: OrtLikeGpuQueue;
}

export interface OrtLikeGpuAdapter {
  info?: { vendor?: string; architecture?: string; device?: string; description?: string };
  features?: { has?: (name: string) => boolean };
}

export interface OrtLikeProfilingEvent {
  kernelType?: string;
  kernelName?: string;
  programName?: string;
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
      profiling?: {
        mode?: 'off' | 'default';
        ondata?: (data: OrtLikeProfilingEvent) => void;
      };
    };
  };
  Tensor: new (
    dtype: string,
    data: Float32Array | Int32Array | BigInt64Array | Uint8Array,
    dims: number[]
  ) => OrtLikeTensor;
  InferenceSession: {
    create: (bytes: Uint8Array, options: Record<string, unknown>) => Promise<OrtLikeSession>;
  };
}

export interface OrtLoadedModuleWithEntryAsset<TModule = OrtLikeModule> {
  module: TModule;
  entryAsset?: OrtLoadedAssetProvenance;
}

export interface OrtWorkerExecutionDeps {
  loadOrtModule: (
    entryModuleUrl: string,
    runtimeAssetFamily: OrtDiagnosticRuntimeFamily
  ) => Promise<OrtLikeModule | OrtLoadedModuleWithEntryAsset>;
  sha256Hex?: (bytes: Uint8Array) => Promise<string>;
  getResourceUrls?: () => string[];
  getResourceEntries?: () => Array<{ name: string; decodedBodySize?: number }>;
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

function defaultGetResourceEntries(): Array<{ name: string; decodedBodySize?: number }> {
  if (typeof performance === 'undefined' || typeof performance.getEntriesByType !== 'function') {
    return [];
  }
  return performance
    .getEntriesByType('resource')
    .filter((entry): entry is PerformanceResourceTiming => typeof entry.name === 'string' && entry.name.length > 0)
    .map((entry) => ({
      name: entry.name,
      decodedBodySize: typeof entry.decodedBodySize === 'number' && entry.decodedBodySize > 0
        ? entry.decodedBodySize
        : undefined,
    }));
}

export async function loadOrtEntryModuleWithBoundarySha256<TModule = OrtLikeModule>(
  entryUrl: string,
  opts: {
    fetchFn?: typeof fetch;
    importFn?: (url: string) => Promise<TModule>;
    sha256Hex?: (bytes: Uint8Array) => Promise<string>;
    referenceLocalDistSha256?: string | null;
    createObjectURL?: (blob: Blob) => string;
    revokeObjectURL?: (url: string) => void;
  } = {}
): Promise<OrtLoadedModuleWithEntryAsset<TModule>> {
  const fetchFn = opts.fetchFn ?? globalThis.fetch.bind(globalThis);
  const sha256 = opts.sha256Hex ?? defaultSha256Hex;
  const createUrl = opts.createObjectURL ?? URL.createObjectURL.bind(URL);
  const revokeUrl = opts.revokeObjectURL ?? URL.revokeObjectURL.bind(URL);
  const importFn =
    opts.importFn ??
    ((u: string) => import(/* @vite-ignore */ u) as Promise<TModule>);

  const resp = await fetchFn(entryUrl, { credentials: 'same-origin' });
  if (!resp.ok) {
    throw new Error(`Failed to fetch ORT entry module (${resp.status} ${resp.statusText || ''}): ${entryUrl}`);
  }
  const buf = await resp.arrayBuffer();
  const bytes = new Uint8Array(buf);
  const sha256HexStr = await sha256(bytes);
  const refSha = opts.referenceLocalDistSha256 ?? null;

  const blobUrl = createUrl(new Blob([bytes], { type: 'text/javascript' }));
  try {
    const mod = await importFn(blobUrl);
    return {
      module: mod,
      entryAsset: {
        role: 'entry',
        url: resp.url || entryUrl,
        byteLength: bytes.byteLength,
        sha256: sha256HexStr,
        via: 'fetch+blob-import',
        referenceLocalDistSha256: refSha,
        matchesLocalDistSha256: refSha ? sha256HexStr === refSha : null,
      },
    };
  } finally {
    try {
      revokeUrl(blobUrl);
    } catch {
      // Ignore revokeObjectURL errors
    }
  }
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

interface DeviceDispatchCounters {
  computePassCount: number;
  dispatchWorkgroupsCount: number;
  dispatchIndirectCount: number;
  timestampWritePassCount: number;
  queueSubmitCount: number;
  commandBufferCount: number;
  querySetCount: number;
  samplePassLabels: string[];
  attached: boolean;
}

export function attachWebGpuDeviceDispatchObserver(
  gpuDevice: OrtLikeGpuDevice | undefined
): { counters: DeviceDispatchCounters; detach: () => void } {
  const counters: DeviceDispatchCounters = {
    computePassCount: 0,
    dispatchWorkgroupsCount: 0,
    dispatchIndirectCount: 0,
    timestampWritePassCount: 0,
    queueSubmitCount: 0,
    commandBufferCount: 0,
    querySetCount: 0,
    samplePassLabels: [],
    attached: false,
  };
  if (!gpuDevice) {
    return { counters, detach: () => {} };
  }

  const restoreCallbacks: Array<() => void> = [];

  try {
    if (typeof gpuDevice.createCommandEncoder === 'function') {
      const origCreateEncoder = gpuDevice.createCommandEncoder;
      const hadOwn = Object.prototype.hasOwnProperty.call(gpuDevice, 'createCommandEncoder');
      gpuDevice.createCommandEncoder = function wrappedCreateCommandEncoder(...args: unknown[]) {
        const encoder = origCreateEncoder.apply(this, args);
        try {
          if (encoder && typeof encoder.beginComputePass === 'function') {
            const origBeginPass = encoder.beginComputePass;
            encoder.beginComputePass = function wrappedBeginComputePass(
              descriptor?: { label?: string; timestampWrites?: unknown }
            ) {
              counters.computePassCount++;
              if (descriptor?.timestampWrites) {
                counters.timestampWritePassCount++;
              }
              if (descriptor?.label && counters.samplePassLabels.length < 12) {
                counters.samplePassLabels.push(descriptor.label);
              }
              const pass = origBeginPass.apply(this, descriptor ? [descriptor] : []);
              try {
                if (pass && typeof pass.dispatchWorkgroups === 'function') {
                  const origDispatch = pass.dispatchWorkgroups;
                  pass.dispatchWorkgroups = function wrappedDispatchWorkgroups(...dArgs: unknown[]) {
                    counters.dispatchWorkgroupsCount++;
                    return origDispatch.apply(this, dArgs);
                  };
                }
                if (pass && typeof pass.dispatchWorkgroupsIndirect === 'function') {
                  const origIndirect = pass.dispatchWorkgroupsIndirect;
                  pass.dispatchWorkgroupsIndirect = function wrappedDispatchIndirect(...dArgs: unknown[]) {
                    counters.dispatchIndirectCount++;
                    return origIndirect.apply(this, dArgs);
                  };
                }
              } catch {
                // Non-fatal: pass wrapper skipped
              }
              return pass;
            };
          }
        } catch {
          // Non-fatal: encoder wrapper skipped
        }
        return encoder;
      };
      counters.attached = true;
      restoreCallbacks.push(() => {
        if (hadOwn) {
          gpuDevice.createCommandEncoder = origCreateEncoder;
        } else {
          delete gpuDevice.createCommandEncoder;
        }
      });
    }

    if (typeof gpuDevice.createQuerySet === 'function') {
      const origCreateQuerySet = gpuDevice.createQuerySet;
      const hadOwn = Object.prototype.hasOwnProperty.call(gpuDevice, 'createQuerySet');
      gpuDevice.createQuerySet = function wrappedCreateQuerySet(
        descriptor?: { type?: string; count?: number }
      ) {
        counters.querySetCount++;
        return origCreateQuerySet.apply(this, descriptor ? [descriptor] : []);
      };
      counters.attached = true;
      restoreCallbacks.push(() => {
        if (hadOwn) {
          gpuDevice.createQuerySet = origCreateQuerySet;
        } else {
          delete gpuDevice.createQuerySet;
        }
      });
    }

    const queue = gpuDevice.queue;
    const origSubmit = queue?.submit;
    if (queue && typeof origSubmit === 'function') {
      const hadOwn = Object.prototype.hasOwnProperty.call(queue, 'submit');
      queue.submit = function wrappedQueueSubmit(commandBuffers: readonly unknown[]) {
        counters.queueSubmitCount++;
        if (Array.isArray(commandBuffers)) {
          counters.commandBufferCount += commandBuffers.length;
        }
        return origSubmit.apply(queue, [commandBuffers]);
      };
      counters.attached = true;
      restoreCallbacks.push(() => {
        if (hadOwn) {
          queue.submit = origSubmit;
        } else {
          delete queue.submit;
        }
      });
    }
  } catch {
    // Non-fatal: keep partial or zero attachment cleanly
  }

  return {
    counters,
    detach: () => {
      for (const fn of restoreCallbacks) {
        try {
          fn();
        } catch {
          // Ignore restore error
        }
      }
    },
  };
}

export function attachWasmBoundaryObserver(
  sha256: (bytes: Uint8Array) => Promise<string>,
  referenceWasmSha256: string | null
): {
  wasmProvenances: OrtLoadedAssetProvenance[];
  detach: () => void;
} {
  const wasmProvenances: OrtLoadedAssetProvenance[] = [];
  const restoreCallbacks: Array<() => void> = [];
  let lastWasmFetchUrl: string | null = null;

  try {
    if (typeof globalThis.fetch === 'function') {
      const origFetch = globalThis.fetch;
      globalThis.fetch = async function wrappedDiagnosticFetch(
        input: RequestInfo | URL,
        init?: RequestInit
      ): Promise<Response> {
        const reqUrl =
          typeof input === 'string'
            ? input
            : input instanceof URL
              ? input.href
              : input.url;
        const resp = await origFetch.call(this, input, init);
        const resolvedUrl = resp.url || reqUrl;
        if (resolvedUrl.includes('.wasm')) {
          lastWasmFetchUrl = resolvedUrl;
        }
        return resp;
      };
      restoreCallbacks.push(() => {
        globalThis.fetch = origFetch;
      });
    }

    const wasmNs = (globalThis as unknown as {
      WebAssembly?: {
        instantiateStreaming?: (
          source: Response | PromiseLike<Response>,
          importObject?: WebAssembly.Imports
        ) => Promise<WebAssembly.WebAssemblyInstantiatedSource>;
        instantiate?: (
          bytes: BufferSource | WebAssembly.Module,
          importObject?: WebAssembly.Imports
        ) => Promise<unknown>;
      };
    }).WebAssembly;

    if (wasmNs && typeof wasmNs.instantiateStreaming === 'function') {
      const origInstantiateStreaming = wasmNs.instantiateStreaming;
      wasmNs.instantiateStreaming = async function wrappedInstantiateStreaming(
        source: Response | PromiseLike<Response>,
        importObject?: WebAssembly.Imports
      ): Promise<WebAssembly.WebAssemblyInstantiatedSource> {
        const resp = await source;
        const url = resp?.url || lastWasmFetchUrl || 'unknown-wasm-streaming';
        try {
          if (resp && typeof resp.clone === 'function') {
            const cloneBuf = await resp.clone().arrayBuffer();
            const u8 = new Uint8Array(cloneBuf);
            const hash = await sha256(u8);
            wasmProvenances.push({
              role: 'wasm',
              url,
              byteLength: u8.byteLength,
              sha256: hash,
              via: 'WebAssembly.instantiateStreaming',
              referenceLocalDistSha256: referenceWasmSha256,
              matchesLocalDistSha256: referenceWasmSha256 ? hash === referenceWasmSha256 : null,
            });
          } else {
            wasmProvenances.push({
              role: 'wasm',
              url,
              byteLength: null,
              sha256: null,
              via: 'WebAssembly.instantiateStreaming',
              referenceLocalDistSha256: referenceWasmSha256,
              matchesLocalDistSha256: null,
              note: 'Response.clone() unavailable; could not read bytes without consuming stream.',
            });
          }
        } catch (err) {
          wasmProvenances.push({
            role: 'wasm',
            url,
            byteLength: null,
            sha256: null,
            via: 'WebAssembly.instantiateStreaming',
            referenceLocalDistSha256: referenceWasmSha256,
            matchesLocalDistSha256: null,
            note: `WASM clone/hash error at instantiateStreaming boundary: ${err instanceof Error ? err.message : String(err)}`,
          });
        }
        return origInstantiateStreaming.call(this, Promise.resolve(resp), importObject);
      };
      restoreCallbacks.push(() => {
        wasmNs.instantiateStreaming = origInstantiateStreaming;
      });
    }

    if (wasmNs && typeof wasmNs.instantiate === 'function') {
      const origInstantiate = wasmNs.instantiate;
      wasmNs.instantiate = async function wrappedInstantiate(
        bytesOrModule: BufferSource | WebAssembly.Module,
        importObject?: WebAssembly.Imports
      ): Promise<unknown> {
        try {
          let view: Uint8Array | null = null;
          if (bytesOrModule instanceof ArrayBuffer) {
            view = new Uint8Array(bytesOrModule);
          } else if (ArrayBuffer.isView(bytesOrModule)) {
            view = new Uint8Array(
              bytesOrModule.buffer,
              bytesOrModule.byteOffset,
              bytesOrModule.byteLength
            );
          }
          if (view) {
            const hash = await sha256(view);
            wasmProvenances.push({
              role: 'wasm',
              url: lastWasmFetchUrl || 'in-memory-wasm-bytes',
              byteLength: view.byteLength,
              sha256: hash,
              via: 'WebAssembly.instantiate',
              referenceLocalDistSha256: referenceWasmSha256,
              matchesLocalDistSha256: referenceWasmSha256 ? hash === referenceWasmSha256 : null,
            });
          }
        } catch {
          // Non-fatal: avoid interrupting instantiate fallback
        }
        return origInstantiate.call(this, bytesOrModule, importObject);
      };
      restoreCallbacks.push(() => {
        wasmNs.instantiate = origInstantiate;
      });
    }
  } catch {
    // Non-fatal if globals are read-only
  }

  return {
    wasmProvenances,
    detach: () => {
      for (let i = restoreCallbacks.length - 1; i >= 0; i--) {
        try {
          restoreCallbacks[i]();
        } catch {
          // Ignore restore error
        }
      }
    },
  };
}

export async function executeOrtDiagnosticWorkerRun(
  pinnedVersion: OrtDiagnosticVersion,
  req: OrtVersionDiagnosticRequest,
  deps: OrtWorkerExecutionDeps
): Promise<OrtWorkerExecutionResult> {
  const versionProfile = HITNET_ORT_VERSION_PROFILES[pinnedVersion];
  const requestedFamily: OrtDiagnosticRuntimeFamily =
    req.runtimeAssetFamily === 'jsep' ? 'jsep' : 'asyncify';
  const familyProfile =
    pinnedVersion === '1.30.0' ? HITNET_ORT_130_FAMILY_PROFILES[requestedFamily] : null;
  const distBaseUrl = familyProfile ? familyProfile.distBaseUrl : versionProfile.distBaseUrl;
  const entryFile = familyProfile ? familyProfile.entryFile : versionProfile.entryFile;
  const entryModuleUrl = `${distBaseUrl}${entryFile}`;
  const configuredProviders = [req.provider];
  const now = deps.now ?? (() => performance.now());
  const sha256 = deps.sha256Hex ?? defaultSha256Hex;
  const getResourceEntries =
    deps.getResourceEntries ??
    (deps.getResourceUrls
      ? () => deps.getResourceUrls!().map((name) => ({ name }))
      : defaultGetResourceEntries);
  const requestAdapter = deps.requestWebGpuAdapter ?? defaultRequestWebGpuAdapter;

  const feedName = req.feedName || 'input';
  const feedDims =
    Array.isArray(req.feedDims) && req.feedDims.length > 0 ? req.feedDims : [1, 6, 240, 320];
  const warmRuns = Math.max(0, Math.min(5, Math.trunc(req.warmRuns ?? 1)));
  const captureValidation = (req.captureGpuValidationErrors ?? true) && req.provider === 'webgpu';
  const profilingRequested = Boolean(req.enableWebGpuProfiling) && req.provider === 'webgpu';
  const captureColdOutputs = Boolean(req.compareColdVsWarmRepeat) && warmRuns >= 1;
  const disableCpuEpFallback = Boolean(req.disableCpuEpFallback) && req.provider === 'webgpu';

  const sessionOptionsApplied: OrtDiagnosticSessionOptionsApplied = {
    executionProviders: configuredProviders,
    graphOptimizationLevel: req.graphOptimizationLevel,
    enableCpuMemArena: req.provider === 'webgpu',
    enableMemPattern: req.provider === 'webgpu',
    numThreads: 1,
    proxy: false,
    wasmPaths: distBaseUrl,
    disableCpuEpFallback,
    fetchMode: 'all-session-outputs',
    profilingRequested,
  };

  let stage: OrtVersionDiagnosticFailure['stage'] = 'init';
  let ort: OrtLikeModule | null = null;
  let ortVersions: { common?: string; web?: string } | null = null;
  let workerGpuAdapter: string | null = null;
  let modelByteLength: number | undefined;
  let modelSha256: string | undefined;
  let feedSha256: string | undefined;
  let sessionInputNames: string[] | undefined;
  let sessionOutputNames: string[] | undefined;
  let session: OrtLikeSession | null = null;
  const gpuValidationErrors: string[] = [];
  let gpuKernelProfileSummary: OrtGpuKernelProfileSummary | null = null;
  let entryProvenance: OrtLoadedAssetProvenance | null = null;
  let wasmProvenances: OrtLoadedAssetProvenance[] = [];

  const collectLoadedAssetProvenance = (
    resEntries: Array<{ name: string; decodedBodySize?: number }>
  ): OrtLoadedAssetProvenance[] => {
    const out: OrtLoadedAssetProvenance[] = [];
    if (entryProvenance) out.push(entryProvenance);
    for (const wp of wasmProvenances) out.push(wp);
    const seenUrls = new Set(out.map((p) => p.url));
    for (const ent of resEntries) {
      if (seenUrls.has(ent.name)) continue;
      if (
        ent.name.endsWith('.jsep.mjs') ||
        ent.name.endsWith('.asyncify.mjs') ||
        ent.name.endsWith('ort-wasm-simd-threaded.mjs')
      ) {
        seenUrls.add(ent.name);
        out.push({
          role: 'emscripten-mjs',
          url: ent.name,
          byteLength: ent.decodedBodySize ?? null,
          sha256: null,
          via: 'performance-resource-entry',
          referenceLocalDistSha256: familyProfile?.referenceLocalDistSha256.emscriptenMjs ?? null,
          matchesLocalDistSha256: null,
          note: 'Loaded via native dynamic import(httpsUrl) from ORT string wasmPaths; raw JS bytes are not observable in worker JS without changing ORT wasmPaths.',
        });
      }
    }
    return out;
  };

  const fail = (message: string): OrtWorkerExecutionResult => {
    const entries = getResourceEntries();
    const resCheck = classifyOrtLoadedResources(
      entries.map((e) => e.name),
      distBaseUrl
    );
    return {
      response: {
        id: req.id,
        ok: false,
        stage,
        message,
        pinnedOrtVersion: pinnedVersion,
        runtimeAssetFamily: requestedFamily,
        entryModuleUrl,
        wasmPaths: distBaseUrl,
        ortVersions,
        loadedOrtResources: resCheck.loadedOrtResources,
        foreignOrtResources: resCheck.foreignOrtResources,
        loadedAssetProvenance: collectLoadedAssetProvenance(entries),
        provider: req.provider,
        configuredProviders,
        fallbackBehavior: 'no-fallback-explicit-error',
        graphOptimizationLevel: req.graphOptimizationLevel,
        sessionOptionsApplied,
        modelByteLength,
        modelSha256,
        feedSha256,
        feedTensorName: feedName,
        feedTensorShape: feedDims,
        feedTensorDataType: 'float32',
        sessionInputNames,
        sessionOutputNames,
        workerGpuAdapter,
        ...(gpuValidationErrors.length > 0 ? { gpuValidationErrors } : {}),
        ...(gpuKernelProfileSummary ? { gpuKernelProfileSummary } : {}),
      },
      transfers: [],
    };
  };

  let detachObserver: (() => void) | null = null;
  let detachWasmObserver: (() => void) | null = null;

  try {
    if (req.expectedOrtVersion !== pinnedVersion) {
      stage = 'version-guard';
      return fail(
        `Worker request expectedOrtVersion="${req.expectedOrtVersion}" does not match pinned worker version "${pinnedVersion}"`
      );
    }

    if (requestedFamily === 'jsep' && pinnedVersion !== '1.30.0') {
      stage = 'init';
      return fail(
        `runtimeAssetFamily="jsep" is only defined on pinned ORT 1.30.0 diagnostic harness (got "${pinnedVersion}")`
      );
    }

    const modelBytes = new Uint8Array(req.modelBuffer);
    if (modelBytes.byteLength === 0) {
      stage = 'init';
      return fail('Model buffer is empty');
    }
    modelByteLength = modelBytes.byteLength;

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
    const loadedMod = await deps.loadOrtModule(entryModuleUrl, requestedFamily);
    if ('module' in loadedMod) {
      ort = loadedMod.module;
      if (loadedMod.entryAsset) {
        const refEntrySha = familyProfile?.referenceLocalDistSha256.entry ?? loadedMod.entryAsset.referenceLocalDistSha256;
        entryProvenance = {
          ...loadedMod.entryAsset,
          referenceLocalDistSha256: refEntrySha,
          matchesLocalDistSha256:
            refEntrySha && loadedMod.entryAsset.sha256
              ? loadedMod.entryAsset.sha256 === refEntrySha
              : loadedMod.entryAsset.matchesLocalDistSha256,
        };
      }
    } else {
      ort = loadedMod;
      entryProvenance = {
        role: 'entry',
        url: entryModuleUrl,
        byteLength: null,
        sha256: null,
        via: 'direct-import-unhashed',
        referenceLocalDistSha256: familyProfile?.referenceLocalDistSha256.entry ?? null,
        matchesLocalDistSha256: null,
      };
    }

    ortVersions = ort.env.versions
      ? { common: ort.env.versions.common, web: ort.env.versions.web }
      : null;
    assertLoadedOrtVersionMatches(ortVersions, pinnedVersion);

    ort.env.logLevel = profilingRequested ? 'verbose' : 'error';
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = distBaseUrl;

    if (ort.env.wasm.wasmPaths !== distBaseUrl) {
      stage = 'resource-guard';
      return fail(`Failed to pin ort.env.wasm.wasmPaths to ${distBaseUrl}`);
    }

    const ortProfilingEvents: OrtLikeProfilingEvent[] = [];
    if (ort.env.webgpu) {
      ort.env.webgpu.profiling = profilingRequested
        ? {
            mode: 'default',
            ondata: (data) => {
              ortProfilingEvents.push(data);
            },
          }
        : { mode: 'off' };
    }

    stage = 'create';
    const sessionOptions: Record<string, unknown> = {
      executionProviders: configuredProviders,
      graphOptimizationLevel: req.graphOptimizationLevel,
      enableCpuMemArena: sessionOptionsApplied.enableCpuMemArena,
      enableMemPattern: sessionOptionsApplied.enableMemPattern,
      logSeverityLevel: profilingRequested ? 0 : 3,
      ...(profilingRequested ? { enableProfiling: true } : {}),
    };
    if (disableCpuEpFallback) {
      sessionOptions.extra = { session: { disable_cpu_ep_fallback: '1' } };
    }

    const wasmObs = attachWasmBoundaryObserver(
      sha256,
      familyProfile?.referenceLocalDistSha256.wasm ?? null
    );
    wasmProvenances = wasmObs.wasmProvenances;
    detachWasmObserver = wasmObs.detach;

    let createMs = 0;
    try {
      const tCreate0 = now();
      session = await ort.InferenceSession.create(modelBytes, sessionOptions);
      createMs = now() - tCreate0;
    } finally {
      wasmObs.detach();
      detachWasmObserver = null;
    }

    sessionInputNames = Array.from(session.inputNames ?? []);
    sessionOutputNames = Array.from(session.outputNames ?? []);

    if (req.provider === 'webgpu') {
      workerGpuAdapter = describeGpuAdapter(ort.env.webgpu?.adapter ?? preAdapter) ?? workerGpuAdapter;
    }

    const afterCreateEntries = getResourceEntries();
    const afterCreateRes = classifyOrtLoadedResources(
      afterCreateEntries.map((e) => e.name),
      distBaseUrl
    );
    if (afterCreateRes.foreignOrtResources.length > 0) {
      stage = 'resource-guard';
      return fail(
        `Unexpected foreign ORT assets loaded during session create: ${afterCreateRes.foreignOrtResources.join(', ')}`
      );
    }
    if (familyProfile) {
      const forbiddenLoaded = afterCreateRes.loadedOrtResources.filter((u) =>
        familyProfile.forbiddenAssetSubstrings.some((sub) => u.includes(sub))
      );
      if (forbiddenLoaded.length > 0) {
        stage = 'resource-guard';
        return fail(
          `Asset family mismatch for "${requestedFamily}": unexpected ORT assets loaded (${forbiddenLoaded.join(', ')})`
        );
      }
      const wasmUrlViolations = wasmProvenances.filter((wp) =>
        familyProfile.forbiddenAssetSubstrings.some((sub) => wp.url.includes(sub))
      );
      if (wasmUrlViolations.length > 0) {
        stage = 'resource-guard';
        return fail(
          `Instantiated WASM mismatch for "${requestedFamily}": ${wasmUrlViolations.map((w) => w.url).join(', ')}`
        );
      }
    }

    const activeAdapter = ort.env.webgpu?.adapter ?? preAdapter;
    const timestampQuerySupported = Boolean(activeAdapter?.features?.has?.('timestamp-query'));
    const observer = profilingRequested
      ? attachWebGpuDeviceDispatchObserver(ort.env.webgpu?.device)
      : null;
    detachObserver = observer?.detach ?? null;

    const extraFeedSpecs = await Promise.all(
      (req.extraFeeds ?? []).map(async (ef) => {
        const dt = ef.dataType ?? 'float32';
        const expectedEf = ef.dims.reduce((acc, d) => acc * d, 1);
        let typed: Float32Array | Int32Array | BigInt64Array | Uint8Array;
        if (dt === 'int32') typed = new Int32Array(ef.buffer);
        else if (dt === 'int64') typed = new BigInt64Array(ef.buffer);
        else if (dt === 'uint8') typed = new Uint8Array(ef.buffer);
        else typed = new Float32Array(ef.buffer);
        if (typed.length !== expectedEf || expectedEf <= 0) {
          throw new Error(
            `Extra feed "${ef.name}" (${dt}) length ${typed.length} does not match expected dims ${JSON.stringify(ef.dims)} (${expectedEf})`
          );
        }
        const extraSha = await sha256(
          new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength)
        );
        return {
          name: ef.name,
          dims: [...ef.dims],
          dt,
          typed,
          byteOffset: typed.byteOffset,
          byteLength: typed.byteLength,
          sha256: extraSha,
        };
      })
    );
    const extraFeedChecks: OrtDiagnosticExtraFeedCheck[] = extraFeedSpecs.map((ef) => ({
      name: ef.name,
      dims: [...ef.dims],
      dataType: ef.dt,
      byteOffset: ef.byteOffset,
      byteLength: ef.byteLength,
      sha256: ef.sha256,
    }));

    const resolvedFeedName = session.inputNames?.includes(feedName)
      ? feedName
      : session.inputNames?.[0] || feedName;
    const runOnce = async (): Promise<{
      runMs: number;
      result: Record<string, OrtLikeTensor>;
    }> => {
      const feedTensor = new ort!.Tensor('float32', feedF32, feedDims);
      const extraTensors = extraFeedSpecs.map((ef) => ({
        name: ef.name,
        tensor: new ort!.Tensor(ef.dt, ef.typed, ef.dims),
      }));
      const feedDict: Record<string, OrtLikeTensor> = {
        [resolvedFeedName]: feedTensor,
      };
      for (const et of extraTensors) {
        feedDict[et.name] = et.tensor;
      }
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
        const result = await session!.run(feedDict);
        const runMs = now() - tRun0;
        return { runMs, result };
      } finally {
        try {
          feedTensor.dispose?.();
        } catch {
          // Ignore input tensor dispose error
        }
        for (const et of extraTensors) {
          try {
            et.tensor.dispose?.();
          } catch {
            // Ignore extra input tensor dispose error
          }
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

    const serializeOutputs = (
      res: Record<string, OrtLikeTensor>,
      transfersAcc: Transferable[]
    ): OrtVersionDiagnosticOutput[] => {
      if (!req.allowArbitraryOutputs) {
        for (const reqName of HITNET_M1_REQUIRED_FINAL_OUTPUTS) {
          if (!res[reqName]) {
            throw new Error(
              `Missing required HITNet final output "${reqName}" (present: ${Object.keys(res).join(', ')})`
            );
          }
        }
      }
      const list: OrtVersionDiagnosticOutput[] = [];
      for (const [name, tensor] of Object.entries(res)) {
        const buffer = copyOrtFloat32OrTypedBuffer(tensor);
        list.push({
          name,
          dims: Array.from(tensor.dims),
          dataType: tensor.type,
          buffer,
        });
        transfersAcc.push(buffer);
      }
      return list;
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
    const transfers: Transferable[] = [];
    let coldOutputs: OrtVersionDiagnosticOutput[] | undefined;
    let finalResult = cold.result;

    if (captureColdOutputs) {
      stage = 'output';
      try {
        coldOutputs = serializeOutputs(cold.result, transfers);
      } finally {
        disposeResultTensors(cold.result);
      }
      stage = 'run';
      for (let i = 0; i < warmRuns; i++) {
        const warm = await runOnce();
        warmRunMs.push(warm.runMs);
        if (i < warmRuns - 1) {
          disposeResultTensors(warm.result);
        } else {
          finalResult = warm.result;
        }
      }
    } else {
      for (let i = 0; i < warmRuns; i++) {
        disposeResultTensors(finalResult);
        const warm = await runOnce();
        warmRunMs.push(warm.runMs);
        finalResult = warm.result;
      }
    }

    if (profilingRequested) {
      try {
        session.endProfiling?.();
      } catch {
        // Non-fatal: Emscripten MEMFS endProfiling may be no-op or unavailable
      }
    }
    if (observer) {
      observer.detach();
      detachObserver = null;
      const kernelTypes: Record<string, number> = {};
      const samplePrograms: string[] = [];
      for (const ev of ortProfilingEvents) {
        const k = ev?.kernelType || 'unknown';
        kernelTypes[k] = (kernelTypes[k] ?? 0) + 1;
        const pName = ev?.programName || ev?.kernelName;
        if (pName && samplePrograms.length < 10) {
          samplePrograms.push(pName);
        }
      }
      const c = observer.counters;
      const hasKernelExecution =
        c.dispatchWorkgroupsCount > 0 ||
        c.dispatchIndirectCount > 0 ||
        c.computePassCount > 0 ||
        ortProfilingEvents.length > 0;
      const placementDisclosure =
        'ORT Web does not expose per-node EP placement in JS; GPU pass/submit counts prove WebGPU kernel dispatches occurred, not that every node avoided ORT-internal CPU fallback.';
      gpuKernelProfileSummary = {
        profilingRequested: true,
        timestampQuerySupported,
        ortProfilingEventCount: ortProfilingEvents.length,
        ortProfilingKernelTypes: kernelTypes,
        ortProfilingSamplePrograms: samplePrograms,
        webgpuComputePassCount: c.computePassCount,
        webgpuDispatchWorkgroupsCount: c.dispatchWorkgroupsCount,
        webgpuDispatchIndirectCount: c.dispatchIndirectCount,
        webgpuTimestampWritePassCount: c.timestampWritePassCount,
        webgpuQueueSubmitCount: c.queueSubmitCount,
        webgpuCommandBufferCount: c.commandBufferCount,
        webgpuQuerySetCount: c.querySetCount,
        webgpuSamplePassLabels: c.samplePassLabels,
        deviceObserverAttached: c.attached,
        note: hasKernelExecution
          ? `Observed WebGPU GPU kernel execution (${c.computePassCount} compute passes, ${c.dispatchWorkgroupsCount} dispatchWorkgroups, ${c.queueSubmitCount} queue.submit, ${ortProfilingEvents.length} ORT profiling events; timestamp-query=${timestampQuerySupported}; disable_cpu_ep_fallback=${disableCpuEpFallback ? '1' : '0/default'}). ${placementDisclosure}`
          : `No WebGPU compute passes or ORT GPU profiling events captured (deviceObserverAttached=${c.attached}, timestamp-query=${timestampQuerySupported}; disable_cpu_ep_fallback=${disableCpuEpFallback ? '1' : '0/default'}). ${placementDisclosure}`,
      };
    }

    stage = 'output';
    let outputs: OrtVersionDiagnosticOutput[] = [];
    try {
      outputs = serializeOutputs(finalResult, transfers);
    } finally {
      disposeResultTensors(finalResult);
    }

    const finalEntries = getResourceEntries();
    const finalResCheck = classifyOrtLoadedResources(
      finalEntries.map((e) => e.name),
      distBaseUrl
    );
    if (finalResCheck.foreignOrtResources.length > 0) {
      stage = 'resource-guard';
      return fail(
        `Unexpected foreign ORT assets loaded during inference: ${finalResCheck.foreignOrtResources.join(', ')}`
      );
    }
    if (familyProfile) {
      const forbiddenLoaded = finalResCheck.loadedOrtResources.filter((u) =>
        familyProfile.forbiddenAssetSubstrings.some((sub) => u.includes(sub))
      );
      if (forbiddenLoaded.length > 0) {
        stage = 'resource-guard';
        return fail(
          `Asset family mismatch for "${requestedFamily}" during inference: ${forbiddenLoaded.join(', ')}`
        );
      }
    }

    return {
      response: {
        id: req.id,
        ok: true,
        pinnedOrtVersion: pinnedVersion,
        runtimeAssetFamily: requestedFamily,
        entryModuleUrl,
        wasmPaths: distBaseUrl,
        ortVersions,
        loadedOrtResources: finalResCheck.loadedOrtResources,
        foreignOrtResources: finalResCheck.foreignOrtResources,
        loadedAssetProvenance: collectLoadedAssetProvenance(finalEntries),
        provider: req.provider,
        configuredProviders,
        fallbackBehavior: 'no-fallback-explicit-error',
        graphOptimizationLevel: req.graphOptimizationLevel,
        sessionOptionsApplied,
        modelByteLength,
        modelSha256,
        feedSha256,
        feedTensorName: resolvedFeedName,
        feedTensorShape: feedDims,
        feedTensorDataType: 'float32',
        feedByteOffset: feedF32.byteOffset,
        feedByteLength: feedF32.byteLength,
        extraFeedChecks,
        sessionInputNames,
        sessionOutputNames,
        workerGpuAdapter,
        gpuValidationErrors: captureValidation ? gpuValidationErrors : null,
        gpuKernelProfileSummary,
        createMs,
        coldRunMs,
        warmRunMs,
        outputs,
        ...(coldOutputs ? { coldOutputs } : {}),
      },
      transfers,
    };
  } catch (err) {
    return fail(err instanceof Error ? err.message : String(err));
  } finally {
    if (detachWasmObserver) {
      try {
        detachWasmObserver();
      } catch {
        // Ignore wasm observer detach error
      }
    }
    if (detachObserver) {
      try {
        detachObserver();
      } catch {
        // Ignore observer detach error
      }
    }
    if (ort) {
      try {
        ort.env.logLevel = 'error';
        if (ort.env.webgpu) {
          ort.env.webgpu.profiling = { mode: 'off' };
        }
      } catch {
        // Ignore env reset error
      }
    }
    if (session) {
      try {
        await session.release();
      } catch {
        // Ignore session release error
      }
    }
  }
}
