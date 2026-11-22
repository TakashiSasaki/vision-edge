import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type * as ort from 'onnxruntime-web';
import { HITNET_ORT_130_FAMILY_PROFILES } from './hitnetOrtComparisonHelper';
import { prepareHitnetM1FullGraphBytes } from './hitnetGatherNdDiagnostic';
import {
  DEFAULT_HITNET_DEV_FIXTURE_PROFILE,
  HITNET_DEV_NON_JSEP_FIXTURE_SPEC,
  assertDevFixtureEnabled,
  runDevOnlyHitnetAppWorkerFixture,
  type DevFixtureWorkerLike,
} from './stereoDepthWebGpuDevFixture';
import {
  PINNED_DEV_FIXTURE_ORT_VERSION,
  createDevWebGpuFixtureProcessor,
} from './stereoDepthWebGpuDevFixtureWorker';
import { createStereoWorkerProcessor } from './stereoDepthWorker';
import type {
  OrtWorkerRuntimeLike,
  StereoWorkerRequest,
  StereoWorkerResponse,
} from './stereoDepthWorkerCore';
import { STEREO_MODEL_PROFILES, StereoWorkerEngine } from './stereoDepthService';
import * as modelAssetResolver from './modelAssetResolver';

function nodeSha256(bytes: Uint8Array): string {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function makeFakeM1Bytes(): Uint8Array {
  const bytes = new Uint8Array(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength);
  for (let i = 0; i < 64; i++) {
    bytes[i] = i + 1;
  }
  return bytes;
}

function makeFakeRgbaBuffer(width = 320, height = 240, offset = 0): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const o = i * 4;
    rgba[o] = (i + offset) & 0xff;
    rgba[o + 1] = (i * 3 + offset) & 0xff;
    rgba[o + 2] = (i * 5 + offset) & 0xff;
    rgba[o + 3] = 255;
  }
  return rgba;
}

function createBridgeWorker(
  processor: ReturnType<typeof createDevWebGpuFixtureProcessor>
): DevFixtureWorkerLike & { terminatedCount: number; flush: () => Promise<void> } {
  let pending: Promise<void> = Promise.resolve();
  const w: DevFixtureWorkerLike & { terminatedCount: number; flush: () => Promise<void> } = {
    onmessage: null,
    onerror: null,
    terminatedCount: 0,
    postMessage(msg: StereoWorkerRequest) {
      pending = pending.then(
        () =>
          new Promise<void>((resolve) => {
            queueMicrotask(() => {
              void processor
                .handleMessage(msg, (resp: StereoWorkerResponse) => {
                  w.onmessage?.({ data: resp } as MessageEvent<StereoWorkerResponse>);
                })
                .finally(resolve);
            });
          })
      );
    },
    terminate() {
      w.terminatedCount++;
    },
    flush() {
      return pending;
    },
  };
  return w;
}

function makeMockOrtRuntime(opts?: {
  versions?: { common?: string; web?: string } | null;
  withWebGpuDevice?: boolean;
}) {
  let releasedSessions = 0;
  let disposedTensors = 0;
  const sessionCreateCalls: Array<Record<string, unknown>> = [];
  const sessionCreateModels: Array<{ byteLength: number; sha256: string }> = [];
  const capturedFeeds: Array<{ name: string; dims: number[]; type: string }> = [];

  class MockTensor {
    dims: number[];
    type: string;
    data: Float32Array;
    constructor(type: string, data: Float32Array, dims: number[]) {
      this.type = type;
      this.data = data;
      this.dims = [...dims];
    }
    dispose() {
      disposedTensors++;
    }
  }

  const fakeGpuDevice = opts?.withWebGpuDevice
    ? {
        features: { has: () => false },
        addEventListener: () => {},
        removeEventListener: () => {},
        pushErrorScope: () => {},
        popErrorScope: async () => null,
        createCommandEncoder: () => ({
          beginComputePass: () => ({
            dispatchWorkgroups: () => {},
            dispatchWorkgroupsIndirect: () => {},
            end: () => {},
          }),
          finish: () => ({}),
        }),
        queue: {
          submit: (_buffers?: readonly unknown[]) => {
            void _buffers;
          },
        },
      }
    : undefined;

  const ortRuntime: OrtWorkerRuntimeLike = {
    env: {
      versions:
        opts?.versions === undefined ? { common: '1.30.0', web: '1.30.0' } : opts.versions ?? undefined,
      wasm: {
        numThreads: 1,
        proxy: false,
        wasmPaths: '',
      },
      webgpu: {
        adapter: {
          info: { vendor: 'amd', architecture: 'rdna3', device: '0x15bf', description: 'Radeon' },
          features: { has: () => false },
        },
        device: fakeGpuDevice as unknown as NonNullable<OrtWorkerRuntimeLike['env']['webgpu']>['device'],
        profiling: { mode: 'off' },
      },
    },
    Tensor: MockTensor as unknown as OrtWorkerRuntimeLike['Tensor'],
    InferenceSession: {
      create: async (bytes, options) => {
        sessionCreateCalls.push(options as Record<string, unknown>);
        sessionCreateModels.push({
          byteLength: bytes.byteLength,
          sha256: nodeSha256(bytes),
        });
        const fakeSession = {
          inputNames: ['input'],
          outputNames: ['reference_output_disparity', 'secondary_output_disparity'],
          inputMetadata: [{ dimensions: [1, 6, 240, 320], type: 'float32' }],
          run: async (feeds: Record<string, ort.Tensor>) => {
            for (const [k, v] of Object.entries(feeds)) {
              capturedFeeds.push({ name: k, dims: [...v.dims], type: v.type });
            }
            if (fakeGpuDevice) {
              for (let p = 0; p < 3; p++) {
                const enc = fakeGpuDevice.createCommandEncoder();
                const pass = enc.beginComputePass();
                pass.dispatchWorkgroups();
                pass.end();
                fakeGpuDevice.queue.submit([enc.finish()]);
              }
            }
            const refData = new Float32Array(76800).fill(12.25);
            const secData = new Float32Array(76800).fill(12.5);
            const tRef = new MockTensor('float32', refData, [1, 240, 320, 1]);
            const tSec = new MockTensor('float32', secData, [1, 240, 320, 1]);
            return {
              reference_output_disparity: tRef as unknown as ort.Tensor,
              secondary_output_disparity: tSec as unknown as ort.Tensor,
            };
          },
          release: async () => {
            releasedSessions++;
          },
        };
        return fakeSession as unknown as ort.InferenceSession;
      },
    },
  };

  return {
    ortRuntime,
    sessionCreateCalls,
    sessionCreateModels,
    capturedFeeds,
    getReleasedSessions: () => releasedSessions,
    getDisposedTensors: () => disposedTensors,
  };
}

describe('stereoDepthWebGpuDevFixture (DEV-ONLY non-JSEP real app Worker path)', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('pins official non-JSEP 1.30.0 webgpu export metadata and leaves production worker unchanged', () => {
    const asyncifyProf = HITNET_ORT_130_FAMILY_PROFILES.asyncify;
    expect(PINNED_DEV_FIXTURE_ORT_VERSION).toBe('1.30.0');
    expect(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.packageExport).toBe('onnxruntime-web@1.30.0/webgpu');
    expect(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.entryFile).toBe('ort.webgpu.min.mjs');
    expect(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.emscriptenMjsFile).toBe(
      'ort-wasm-simd-threaded.asyncify.mjs'
    );
    expect(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.wasmFile).toBe(
      'ort-wasm-simd-threaded.asyncify.wasm'
    );
    expect(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedEntrySha256).toBe(
      asyncifyProf.referenceLocalDistSha256.entry
    );
    expect(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1ByteLength).toBe(8888846);
    expect(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedM1Sha256).toBe(
      '97e3673bdea94980f71cc1fc13998066238636b0ae0fa7f5381aa3ffa711e758'
    );

    const root = process.cwd();
    const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')) as {
      devDependencies?: Record<string, string>;
    };
    expect(pkg.devDependencies?.['onnxruntime-web']).toBe('1.30.0');

    const prodWorker = fs.readFileSync(path.join(root, 'src/services/stereoDepthWorker.ts'), 'utf8');
    expect(prodWorker).toContain("from 'onnxruntime-web'");
    expect(prodWorker).not.toContain('ort.webgpu.min.mjs');
    expect(prodWorker).toContain('createStereoWorkerProcessor');

    const ortWebgpuPkg = JSON.parse(
      fs.readFileSync(path.join(root, 'node_modules/onnxruntime-web/package.json'), 'utf8')
    ) as {
      exports?: {
        './webgpu'?: {
          import?: {
            'onnxruntime-web-use-extern-wasm'?: string;
            default?: string;
          };
        };
      };
    };
    expect(ortWebgpuPkg.exports?.['./webgpu']?.import?.['onnxruntime-web-use-extern-wasm']).toBe(
      './dist/ort.webgpu.min.mjs'
    );
  });

  it('fails closed when not in DEV mode without loading ORT or touching WebGPU', async () => {
    expect(() => assertDevFixtureEnabled(false)).toThrow(/\[dev-gate\]/);

    let loadCalled = false;
    let adapterCalled = false;
    const processor = createDevWebGpuFixtureProcessor({
      isDev: false,
      loadOrtModule: async () => {
        loadCalled = true;
        throw new Error('should not be called');
      },
      requestWebGpuAdapter: async () => {
        adapterCalled = true;
        return null;
      },
    });

    const responses: StereoWorkerResponse[] = [];
    await processor.handleMessage(
      {
        type: 'init-onnx',
        id: 1,
        profile: DEFAULT_HITNET_DEV_FIXTURE_PROFILE,
        onnxBuffer: new ArrayBuffer(16),
        wasmPrimaryUrl: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl,
        wasmFallbackUrl: null,
        backendPreference: 'webgpu',
        strictWebGpu: true,
        graphOptimizationLevel: 'basic',
      },
      (r) => responses.push(r)
    );
    expect(responses[0]).toMatchObject({ type: 'error', id: 1, ok: false, stage: 'wasm' });
    expect((responses[0] as { message: string }).message).toContain('[dev-gate]');
    expect(loadCalled).toBe(false);
    expect(adapterCalled).toBe(false);

    const m1 = makeFakeM1Bytes();
    const res = await runDevOnlyHitnetAppWorkerFixture({
      m1Bytes: m1,
      expectedM1Sha256: nodeSha256(m1),
      leftRgbaBuffer: makeFakeRgbaBuffer(),
      rightRgbaBuffer: makeFakeRgbaBuffer(),
      isDev: false,
      sha256Hex: async (u8) => nodeSha256(u8),
    });
    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.stage).toBe('dev-gate');
    }
  });

  it('rejects silent provider swaps, rewrite flags, and missing WebGPU adapter with no WASM fallback', async () => {
    let loadCalled = false;
    const processor = createDevWebGpuFixtureProcessor({
      isDev: true,
      loadOrtModule: async () => {
        loadCalled = true;
        throw new Error('should not load');
      },
      requestWebGpuAdapter: async () => null,
    });

    const send = async (req: StereoWorkerRequest) => {
      const out: StereoWorkerResponse[] = [];
      await processor.handleMessage(req, (r) => out.push(r));
      return out[0];
    };

    const rSad = await send({
      type: 'init-sad',
      id: 1,
      profile: DEFAULT_HITNET_DEV_FIXTURE_PROFILE,
    });
    expect(rSad.type).toBe('error');

    const rWasm = await send({
      type: 'init-onnx',
      id: 2,
      profile: DEFAULT_HITNET_DEV_FIXTURE_PROFILE,
      onnxBuffer: new ArrayBuffer(16),
      wasmPrimaryUrl: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl,
      wasmFallbackUrl: null,
      backendPreference: 'wasm',
      strictWebGpu: true,
      graphOptimizationLevel: 'basic',
    });
    expect(rWasm.type).toBe('error');
    expect((rWasm as { message: string }).message).toContain('[opt-in-guard]');

    const rRewrite = await send({
      type: 'init-onnx',
      id: 3,
      profile: DEFAULT_HITNET_DEV_FIXTURE_PROFILE,
      onnxBuffer: new ArrayBuffer(16),
      wasmPrimaryUrl: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl,
      wasmFallbackUrl: null,
      backendPreference: 'webgpu',
      normalizeSliceAxes: 'compact-active',
      strictWebGpu: true,
      graphOptimizationLevel: 'basic',
    });
    expect(rRewrite.type).toBe('error');
    expect((rRewrite as { message: string }).message).toContain('[opt-in-guard]');

    const rNoGpu = await send({
      type: 'init-onnx',
      id: 4,
      profile: DEFAULT_HITNET_DEV_FIXTURE_PROFILE,
      onnxBuffer: new ArrayBuffer(16),
      wasmPrimaryUrl: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl,
      wasmFallbackUrl: null,
      backendPreference: 'webgpu',
      strictWebGpu: true,
      graphOptimizationLevel: 'basic',
    });
    expect(rNoGpu.type).toBe('error');
    expect((rNoGpu as { message: string }).message).toContain('[webgpu-support]');
    expect(loadCalled).toBe(false);
  });

  it('releases ORT session on failed init when JSEP/foreign assets or wrong ORT version appear', async () => {
    const m1 = makeFakeM1Bytes();
    const m1Sha = nodeSha256(m1);

    const mockJsepLeak = makeMockOrtRuntime();
    const jsepProcessor = createDevWebGpuFixtureProcessor({
      isDev: true,
      sha256Hex: async (u8) => nodeSha256(u8),
      requestWebGpuAdapter: async () => ({ info: { vendor: 'amd' } }),
      getResourceEntries: () => [
        {
          name: `${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl}ort-wasm-simd-threaded.jsep.mjs`,
          decodedBodySize: 1234,
        },
        {
          name: `${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl}ort-wasm-simd-threaded.jsep.wasm`,
          decodedBodySize: 5678,
        },
      ],
      loadOrtModule: async (url) => ({
        module: mockJsepLeak.ortRuntime,
        entryAsset: {
          role: 'entry',
          url,
          byteLength: 66349,
          sha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedEntrySha256,
          via: 'fetch+blob-import',
          referenceLocalDistSha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedEntrySha256,
          matchesLocalDistSha256: true,
        },
      }),
    });

    const wJsep = createBridgeWorker(jsepProcessor);
    const resJsep = await runDevOnlyHitnetAppWorkerFixture({
      m1Bytes: m1,
      expectedM1Sha256: m1Sha,
      leftRgbaBuffer: makeFakeRgbaBuffer(),
      rightRgbaBuffer: makeFakeRgbaBuffer(),
      isDev: true,
      workerFactory: () => wJsep,
      sha256Hex: async (u8) => nodeSha256(u8),
    });

    expect(resJsep.ok).toBe(false);
    if (!resJsep.ok) {
      expect(resJsep.stage).toBe('resource-guard');
      expect(resJsep.message).toContain('jsep');
    }
    // Critical: created session MUST be released even when post-create guard fails
    expect(mockJsepLeak.getReleasedSessions()).toBe(1);
    expect(wJsep.terminatedCount).toBe(1);

    const mockWrongVersion = makeMockOrtRuntime({
      versions: { common: '1.31.1', web: '1.31.1' },
    });
    const verProcessor = createDevWebGpuFixtureProcessor({
      isDev: true,
      sha256Hex: async (u8) => nodeSha256(u8),
      requestWebGpuAdapter: async () => ({ info: { vendor: 'amd' } }),
      loadOrtModule: async (url) => ({
        module: mockWrongVersion.ortRuntime,
        entryAsset: {
          role: 'entry',
          url,
          byteLength: 66349,
          sha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedEntrySha256,
          via: 'fetch+blob-import',
          referenceLocalDistSha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedEntrySha256,
          matchesLocalDistSha256: true,
        },
      }),
    });
    const wVer = createBridgeWorker(verProcessor);
    const resVer = await runDevOnlyHitnetAppWorkerFixture({
      m1Bytes: m1,
      expectedM1Sha256: m1Sha,
      leftRgbaBuffer: makeFakeRgbaBuffer(),
      rightRgbaBuffer: makeFakeRgbaBuffer(),
      isDev: true,
      workerFactory: () => wVer,
      sha256Hex: async (u8) => nodeSha256(u8),
    });
    expect(resVer.ok).toBe(false);
    if (!resVer.ok) {
      expect(resVer.stage).toBe('version-guard');
    }
    expect(mockWrongVersion.sessionCreateCalls).toHaveLength(0);
    expect(wVer.terminatedCount).toBe(1);
  });

  it('executes real app worker protocol on exact M1 + asyncify non-JSEP runtime with dispatch proof and full teardown', async () => {
    const m1 = makeFakeM1Bytes();
    const m1Sha = nodeSha256(m1);
    const left = makeFakeRgbaBuffer(320, 240, 1);
    const right = makeFakeRgbaBuffer(320, 240, 9);

    const mockOrt = makeMockOrtRuntime({ withWebGpuDevice: true });
    const asyncifyBase = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl;

    const processor = createDevWebGpuFixtureProcessor({
      isDev: true,
      sha256Hex: async (u8) => nodeSha256(u8),
      requestWebGpuAdapter: async () => ({
        info: { vendor: 'amd', architecture: 'rdna3', device: '0x15bf', description: 'Radeon' },
        features: { has: () => false },
      }),
      getResourceEntries: () => [
        {
          name: `${asyncifyBase}ort-wasm-simd-threaded.asyncify.mjs`,
          decodedBodySize: 218075,
        },
        {
          name: `${asyncifyBase}ort-wasm-simd-threaded.asyncify.wasm`,
          decodedBodySize: 24756195,
        },
      ],
      loadOrtModule: async (url) => ({
        module: mockOrt.ortRuntime,
        entryAsset: {
          role: 'entry',
          url,
          byteLength: 66349,
          sha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedEntrySha256,
          via: 'fetch+blob-import',
          referenceLocalDistSha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedEntrySha256,
          matchesLocalDistSha256: true,
        },
      }),
    });

    const w = createBridgeWorker(processor);
    const res = await runDevOnlyHitnetAppWorkerFixture({
      m1Bytes: m1,
      expectedM1Sha256: m1Sha,
      leftRgbaBuffer: left,
      rightRgbaBuffer: right,
      warmRuns: 1,
      enableWebGpuProfiling: true,
      disableCpuEpFallback: false,
      isDev: true,
      workerFactory: () => w,
      sha256Hex: async (u8) => nodeSha256(u8),
    });

    expect(res.ok).toBe(true);
    if (res.ok) {
      expect(res.pinnedOrtVersion).toBe('1.30.0');
      expect(res.runtimeAssetFamily).toBe('asyncify');
      expect(res.entryModuleUrl).toBe(`${asyncifyBase}ort.webgpu.min.mjs`);
      expect(res.rawInputModelSha256).toBe(m1Sha);
      expect(res.baseSessionModelSha256).toBe(m1Sha);
      expect(res.sessionModelSha256).toBe(m1Sha);
      expect(res.configuredProviders).toEqual(['webgpu']);
      expect(res.fallbackReason).toBeNull();
      expect(res.foreignOrtResources).toEqual([]);

      const entryProv = res.loadedAssetProvenance.find((p) => p.role === 'entry');
      expect(entryProv?.sha256).toBe(HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedEntrySha256);
      expect(entryProv?.matchesLocalDistSha256).toBe(true);

      const mjsProv = res.loadedAssetProvenance.find((p) => p.role === 'emscripten-mjs');
      expect(mjsProv?.url).toBe(`${asyncifyBase}ort-wasm-simd-threaded.asyncify.mjs`);
      expect(mjsProv?.sha256).toBeNull();
      expect(mjsProv?.via).toBe('performance-resource-entry');

      expect(res.coldRun.feedMetadata.input).toEqual({
        dims: [1, 6, 240, 320],
        dataType: 'float32',
      });
      expect(res.warmRuns).toHaveLength(1);
      expect(res.warmRuns[0].feedSha256s.input).toBe(res.coldRun.feedSha256s.input);
      expect(res.coldRun.outputs.reference_output_disparity.data.length).toBe(76800);
      expect(res.coldRun.outputs.secondary_output_disparity.data.length).toBe(76800);

      expect(res.lastRun.gpuKernelProfileSummary?.deviceObserverAttached).toBe(true);
      expect(res.lastRun.gpuKernelProfileSummary?.webgpuComputePassCount).toBe(3);
      expect(res.lastRun.gpuKernelProfileSummary?.webgpuDispatchWorkgroupsCount).toBe(3);
      expect(res.lastRun.gpuKernelProfileSummary?.webgpuQueueSubmitCount).toBe(3);
      expect(res.lastRun.gpuKernelProfileSummary?.note).toContain(
        'ORT Web does not expose per-node EP placement in JS'
      );
    }

    // 2 runs * (1 input tensor + 2 output tensors) = 6 tensors disposed, 1 session released, worker terminated
    expect(mockOrt.getDisposedTensors()).toBe(6);
    expect(mockOrt.getReleasedSessions()).toBe(1);
    expect(w.terminatedCount).toBe(1);
    expect(mockOrt.sessionCreateCalls[0]).toMatchObject({
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'basic',
    });
  });

  it('works cleanly through StereoWorkerEngine.create when explicitly passed as workerFactory', async () => {
    vi.spyOn(modelAssetResolver, 'resolveOrtWasmUrls').mockResolvedValue({
      primaryUrl: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl,
      fallbackUrl: null,
      source: 'external',
    });

    const m1 = makeFakeM1Bytes();
    const mockOrt = makeMockOrtRuntime({ withWebGpuDevice: true });
    const processor = createDevWebGpuFixtureProcessor({
      isDev: true,
      sha256Hex: async (u8) => nodeSha256(u8),
      requestWebGpuAdapter: async () => ({ info: { vendor: 'amd' } }),
      getResourceEntries: () => [
        {
          name: `${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl}ort-wasm-simd-threaded.asyncify.mjs`,
        },
        {
          name: `${HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl}ort-wasm-simd-threaded.asyncify.wasm`,
        },
      ],
      loadOrtModule: async (url) => ({
        module: mockOrt.ortRuntime,
        entryAsset: {
          role: 'entry',
          url,
          byteLength: 66349,
          sha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedEntrySha256,
          via: 'fetch+blob-import',
          referenceLocalDistSha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedEntrySha256,
          matchesLocalDistSha256: true,
        },
      }),
    });
    const w = createBridgeWorker(processor);

    const engine = await StereoWorkerEngine.create({
      variantId: 'hitnet-sf',
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBytes: m1,
      backendPreference: 'webgpu',
      strictWebGpu: true,
      graphOptimizationLevel: 'basic',
      workerFactory: () => w as unknown as import('./stereoDepthService').StereoWorkerLike,
    });

    expect(engine.backend.configuredProviders).toEqual(['webgpu']);
    expect(engine.backend.fallbackReason).toBeNull();
    engine.close();
    await w.flush();
    expect(mockOrt.getReleasedSessions()).toBe(1);
    expect(w.terminatedCount).toBe(1);
  });

  it('rejects mismatched entry module SHA-256 without creating an ORT session', async () => {
    const m1 = makeFakeM1Bytes();
    const m1Sha = nodeSha256(m1);
    const mockOrt = makeMockOrtRuntime();
    const processor = createDevWebGpuFixtureProcessor({
      isDev: true,
      sha256Hex: async (u8) => nodeSha256(u8),
      requestWebGpuAdapter: async () => ({ info: { vendor: 'amd' } }),
      loadOrtModule: async (url) => ({
        module: mockOrt.ortRuntime,
        entryAsset: {
          role: 'entry',
          url,
          byteLength: 100,
          sha256: '0000000000000000000000000000000000000000000000000000000000000000',
          via: 'fetch+blob-import',
          referenceLocalDistSha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedEntrySha256,
          matchesLocalDistSha256: false,
        },
      }),
    });
    const w = createBridgeWorker(processor);
    const res = await runDevOnlyHitnetAppWorkerFixture({
      m1Bytes: m1,
      expectedM1Sha256: m1Sha,
      leftRgbaBuffer: makeFakeRgbaBuffer(),
      rightRgbaBuffer: makeFakeRgbaBuffer(),
      isDev: true,
      workerFactory: () => w,
      sha256Hex: async (u8) => nodeSha256(u8),
    });
    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.stage).toBe('resource-guard');
      expect(res.message).toContain('SHA-256 mismatch');
    }
    expect(mockOrt.sessionCreateCalls).toHaveLength(0);
    expect(w.terminatedCount).toBe(1);
  });

  it('keeps production worker taking original Concat-split path even if skipImplicitConcatSplitForExactM1Fixture is sent', async () => {
    const enc = new TextEncoder();
    const field = (fn: number, payload: Uint8Array) => {
      const len = payload.byteLength;
      const out = new Uint8Array(2 + len);
      out[0] = (fn << 3) | 2;
      out[1] = len;
      out.set(payload, 2);
      return out;
    };
    const concatParts = [
      field(1, enc.encode('a0')),
      field(1, enc.encode('a1')),
      field(1, enc.encode('a2')),
      field(1, enc.encode('a3')),
      field(1, enc.encode('a4')),
      field(2, enc.encode('out')),
      field(3, enc.encode('test_concat5')),
      field(4, enc.encode('Concat')),
    ];
    const nodeLen = concatParts.reduce((s, p) => s + p.byteLength, 0);
    const nodePayload = new Uint8Array(nodeLen);
    let off = 0;
    for (const p of concatParts) {
      nodePayload.set(p, off);
      off += p.byteLength;
    }
    const graphNode = field(1, nodePayload);
    const syntheticOnnx = field(7, graphNode);
    // Pad >= 16 bytes with trailing model_version varint (fn=6, wt=0, val=1) if needed
    const paddedSynthetic =
      syntheticOnnx.byteLength >= 16
        ? syntheticOnnx
        : (() => {
            const p = new Uint8Array(syntheticOnnx.byteLength + 2);
            p.set(syntheticOnnx, 0);
            p[syntheticOnnx.byteLength] = 6 << 3;
            p[syntheticOnnx.byteLength + 1] = 1;
            return p;
          })();
    const rawSyntheticSha = nodeSha256(paddedSynthetic);

    const seenProdModels: Array<{ byteLength: number; sha256: string }> = [];
    const prodProcessor = createStereoWorkerProcessor(async (onnxBytes, delegate) => {
      seenProdModels.push({ byteLength: onnxBytes.byteLength, sha256: nodeSha256(onnxBytes) });
      return {
        session: {
          inputNames: ['input'],
          outputNames: ['reference_output_disparity', 'secondary_output_disparity'],
          inputMetadata: [{ dimensions: [1, 6, 240, 320], type: 'float32' }],
          run: async () => ({}),
          release: async () => {},
        } as unknown as ort.InferenceSession,
        backend: {
          runtime: 'onnxruntime-web',
          requestedProvider: delegate === 'WebGPU' ? 'webgpu' : 'wasm',
          configuredProviders: ['webgpu'],
          fallbackReason: null,
          actualBackend: 'unmeasured',
        },
        wasmSourceUsed: 'primary',
      };
    });

    const prodResponses: StereoWorkerResponse[] = [];
    await prodProcessor.handleMessage(
      {
        type: 'init-onnx',
        id: 1,
        profile: DEFAULT_HITNET_DEV_FIXTURE_PROFILE,
        onnxBuffer: paddedSynthetic.slice().buffer,
        wasmPrimaryUrl: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl,
        wasmFallbackUrl: null,
        backendPreference: 'webgpu',
        strictWebGpu: true,
        graphOptimizationLevel: 'basic',
        skipImplicitConcatSplitForExactM1Fixture: true,
      },
      (r) => prodResponses.push(r)
    );

    expect(prodResponses[0].type).toBe('init-result');
    if (prodResponses[0].type === 'init-result') {
      expect(prodResponses[0].transformStats?.splitCount).toBe(1);
      expect(prodResponses[0].sessionOptionsApplied?.pretransformedExactM1SkipConcatSplit).toBe(
        false
      );
      expect(prodResponses[0].sessionModelSha256).not.toBe(rawSyntheticSha);
      expect(seenProdModels[0]?.sha256).toBe(prodResponses[0].sessionModelSha256);
    }

    // DEV fixture worker rejects forged skipImplicitConcatSplitForExactM1Fixture with wrong bytes/SHA before ORT load
    let ortLoadedOnBadSkip = false;
    const devGuardProcessor = createDevWebGpuFixtureProcessor({
      isDev: true,
      sha256Hex: async (u8) => nodeSha256(u8),
      requestWebGpuAdapter: async () => ({ info: { vendor: 'amd' } }),
      loadOrtModule: async () => {
        ortLoadedOnBadSkip = true;
        throw new Error('should not load ORT on bad M1 skip');
      },
    });
    const guardResponses: StereoWorkerResponse[] = [];
    await devGuardProcessor.handleMessage(
      {
        type: 'init-onnx',
        id: 2,
        profile: DEFAULT_HITNET_DEV_FIXTURE_PROFILE,
        onnxBuffer: makeFakeM1Bytes().buffer as ArrayBuffer,
        wasmPrimaryUrl: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl,
        wasmFallbackUrl: null,
        backendPreference: 'webgpu',
        strictWebGpu: true,
        graphOptimizationLevel: 'basic',
        skipImplicitConcatSplitForExactM1Fixture: true,
      },
      (r) => guardResponses.push(r)
    );
    expect(guardResponses[0]).toMatchObject({ type: 'error', id: 2, ok: false, stage: 'input' });
    expect((guardResponses[0] as { message: string }).message).toContain('[model-guard]');
    expect(ortLoadedOnBadSkip).toBe(false);
  });

  it('preserves exact M1 bytes (97e3673b..., 8888846 B) before InferenceSession.create and in manifest while production still splits', async () => {
    const hitnetPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (!fs.existsSync(hitnetPath)) {
      return;
    }
    const rawBytes = new Uint8Array(fs.readFileSync(hitnetPath));
    const m1Bytes = prepareHitnetM1FullGraphBytes(rawBytes).bytes;
    const m1Sha = nodeSha256(m1Bytes);
    expect(m1Bytes.byteLength).toBe(8888846);
    expect(m1Sha).toBe('97e3673bdea94980f71cc1fc13998066238636b0ae0fa7f5381aa3ffa711e758');

    // 1. Production createStereoWorkerProcessor on exact M1 still splits 2 Concats (5->4) even with skip flag
    const prodModels: Array<{ byteLength: number; sha256: string }> = [];
    const prodProcessor = createStereoWorkerProcessor(async (onnxBytes) => {
      prodModels.push({ byteLength: onnxBytes.byteLength, sha256: nodeSha256(onnxBytes) });
      return {
        session: {
          inputNames: ['input'],
          outputNames: ['reference_output_disparity', 'secondary_output_disparity'],
          inputMetadata: [{ dimensions: [1, 6, 240, 320], type: 'float32' }],
          run: async () => ({}),
          release: async () => {},
        } as unknown as ort.InferenceSession,
        backend: {
          runtime: 'onnxruntime-web',
          requestedProvider: 'webgpu',
          configuredProviders: ['webgpu'],
          fallbackReason: null,
          actualBackend: 'unmeasured',
        },
        wasmSourceUsed: 'primary',
      };
    });
    const prodRes: StereoWorkerResponse[] = [];
    await prodProcessor.handleMessage(
      {
        type: 'init-onnx',
        id: 1,
        profile: DEFAULT_HITNET_DEV_FIXTURE_PROFILE,
        onnxBuffer: m1Bytes.slice().buffer,
        wasmPrimaryUrl: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl,
        wasmFallbackUrl: null,
        backendPreference: 'webgpu',
        strictWebGpu: true,
        graphOptimizationLevel: 'basic',
        skipImplicitConcatSplitForExactM1Fixture: true,
      },
      (r) => prodRes.push(r)
    );
    expect(prodRes[0].type).toBe('init-result');
    if (prodRes[0].type === 'init-result') {
      expect(prodRes[0].transformStats?.splitCount).toBe(2);
      expect(prodRes[0].sessionOptionsApplied?.pretransformedExactM1SkipConcatSplit).toBe(false);
      expect(prodRes[0].sessionModelSha256).toBe(
        '9e0fda247d65c7734e7e02f8d0f9dc001c7160a52776a27ad9c095f4fb9b5848'
      );
      expect(prodModels[0]).toEqual({
        byteLength: 8889263,
        sha256: '9e0fda247d65c7734e7e02f8d0f9dc001c7160a52776a27ad9c095f4fb9b5848',
      });
    }

    // 2. DEV-only fixture skips ONLY implicit Concat split and delivers exact M1 bytes to InferenceSession.create
    const mockOrt = makeMockOrtRuntime({ withWebGpuDevice: true });
    const asyncifyBase = HITNET_DEV_NON_JSEP_FIXTURE_SPEC.distBaseUrl;
    const devProcessor = createDevWebGpuFixtureProcessor({
      isDev: true,
      sha256Hex: async (u8) => nodeSha256(u8),
      requestWebGpuAdapter: async () => ({
        info: { vendor: 'amd', architecture: 'rdna3', device: '0x15bf', description: 'Radeon' },
        features: { has: () => false },
      }),
      getResourceEntries: () => [
        {
          name: `${asyncifyBase}ort-wasm-simd-threaded.asyncify.mjs`,
          decodedBodySize: 218075,
        },
        {
          name: `${asyncifyBase}ort-wasm-simd-threaded.asyncify.wasm`,
          decodedBodySize: 24756195,
        },
      ],
      loadOrtModule: async (url) => ({
        module: mockOrt.ortRuntime,
        entryAsset: {
          role: 'entry',
          url,
          byteLength: 66349,
          sha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedEntrySha256,
          via: 'fetch+blob-import',
          referenceLocalDistSha256: HITNET_DEV_NON_JSEP_FIXTURE_SPEC.expectedEntrySha256,
          matchesLocalDistSha256: true,
        },
      }),
    });

    const w = createBridgeWorker(devProcessor);
    const res = await runDevOnlyHitnetAppWorkerFixture({
      m1Bytes,
      leftRgbaBuffer: makeFakeRgbaBuffer(320, 240, 2),
      rightRgbaBuffer: makeFakeRgbaBuffer(320, 240, 7),
      warmRuns: 1,
      enableWebGpuProfiling: true,
      isDev: true,
      workerFactory: () => w,
      sha256Hex: async (u8) => nodeSha256(u8),
    });

    expect(res.ok).toBe(true);
    if (res.ok) {
      expect(res.rawInputModelSha256).toBe(m1Sha);
      expect(res.baseSessionModelSha256).toBe(m1Sha);
      expect(res.sessionModelSha256).toBe(m1Sha);
      expect(res.modelByteLength).toBe(8888846);
      expect(res.sessionModelByteLength).toBe(8888846);
      expect(res.pretransformedExactM1SkipConcatSplit).toBe(true);
      expect(res.splitCount).toBe(0);
      expect(res.fallbackReason).toBeNull();
      expect(res.configuredProviders).toEqual(['webgpu']);
    }
    expect(mockOrt.sessionCreateModels).toEqual([{ byteLength: 8888846, sha256: m1Sha }]);
    expect(mockOrt.getReleasedSessions()).toBe(1);
    expect(mockOrt.getDisposedTensors()).toBe(6);
    expect(w.terminatedCount).toBe(1);
  });
});
