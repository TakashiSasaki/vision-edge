import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { describe, expect, it } from 'vitest';
import {
  assertLoadedOrtVersionMatches,
  classifyOrtLoadedResources,
  executeOrtDiagnosticWorkerRun,
  HITNET_ORT_130_FAMILY_PROFILES,
  HITNET_ORT_VERSION_PROFILES,
  loadOrtEntryModuleWithBoundarySha256,
  type OrtLikeGpuDevice,
  type OrtLikeModule,
  type OrtLikeTensor,
} from './hitnetOrtComparisonHelper';
import { HITNET_M1_FULL_GRAPH_SHA256 } from './stereoDepthWorker';

function makeValidFeedBuffer(seed = 0): ArrayBuffer {
  const f32 = new Float32Array(1 * 6 * 240 * 320);
  for (let i = 0; i < f32.length; i += 997) {
    f32[i] = ((i + seed) % 255) / 255;
  }
  return f32.buffer;
}

function makeValidModelBuffer(): ArrayBuffer {
  return new Uint8Array([0x08, 0x07, 0x12, 0x04, 0x74, 0x65, 0x73, 0x74]).buffer;
}

function nodeSha256(bytes: Uint8Array): string {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

describe('hitnetOrtComparisonHelper (isolated 1.30.0 Asyncify/JSEP & 1.31.1 M1 diagnostic)', () => {
  it('pins distinct official CDN asset profiles for 1.30.0 and 1.31.1 and preserves 1.30.0 in package.json', () => {
    const p130 = HITNET_ORT_VERSION_PROFILES['1.30.0'];
    const p131 = HITNET_ORT_VERSION_PROFILES['1.31.1'];

    expect(p130.distBaseUrl).toBe('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/');
    expect(p131.distBaseUrl).toBe('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/');
    expect(p130.entryFile).toBe('ort.webgpu.min.mjs');
    expect(p131.entryFile).toBe('ort.webgpu.min.mjs');
    expect(p130.expectedAssetFiles).toContain('ort-wasm-simd-threaded.jsep.mjs');
    expect(p130.expectedAssetFiles).toContain('ort-wasm-simd-threaded.jsep.wasm');
    expect(p131.expectedAssetFiles).toContain('ort-wasm-simd-threaded.mjs');
    expect(p131.expectedAssetFiles).toContain('ort-wasm-simd-threaded.wasm');

    const asyncifyProf = HITNET_ORT_130_FAMILY_PROFILES.asyncify;
    const jsepProf = HITNET_ORT_130_FAMILY_PROFILES.jsep;
    expect(asyncifyProf.entryFile).toBe('ort.webgpu.min.mjs');
    expect(asyncifyProf.emscriptenMjsFile).toBe('ort-wasm-simd-threaded.asyncify.mjs');
    expect(asyncifyProf.wasmFile).toBe('ort-wasm-simd-threaded.asyncify.wasm');
    expect(asyncifyProf.forbiddenAssetSubstrings).toEqual(['.jsep.']);

    expect(jsepProf.entryFile).toBe('ort.bundle.min.mjs');
    expect(jsepProf.emscriptenMjsFile).toBe('ort-wasm-simd-threaded.jsep.mjs');
    expect(jsepProf.wasmFile).toBe('ort-wasm-simd-threaded.jsep.wasm');
    expect(jsepProf.forbiddenAssetSubstrings).toEqual(['.asyncify.']);

    const root = process.cwd();
    const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')) as {
      devDependencies?: Record<string, string>;
      dependencies?: Record<string, string>;
    };
    expect(pkg.devDependencies?.['onnxruntime-web']).toBe('1.30.0');
    expect(pkg.dependencies?.['onnxruntime-web']).toBeUndefined();

    // Verify local installed 1.30.0 dist artifact SHA-256 constants match package exports
    const ortDistDir = path.join(root, 'node_modules/onnxruntime-web/dist');
    if (fs.existsSync(ortDistDir)) {
      const ortPkg = JSON.parse(
        fs.readFileSync(path.join(root, 'node_modules/onnxruntime-web/package.json'), 'utf8')
      ) as { exports?: { '.'?: { import?: { default?: string } } } };
      expect(ortPkg.exports?.['.']?.import?.default).toBe('./dist/ort.bundle.min.mjs');
      expect(nodeSha256(fs.readFileSync(path.join(ortDistDir, asyncifyProf.entryFile)))).toBe(
        asyncifyProf.referenceLocalDistSha256.entry
      );
      expect(nodeSha256(fs.readFileSync(path.join(ortDistDir, asyncifyProf.emscriptenMjsFile)))).toBe(
        asyncifyProf.referenceLocalDistSha256.emscriptenMjs
      );
      expect(nodeSha256(fs.readFileSync(path.join(ortDistDir, asyncifyProf.wasmFile)))).toBe(
        asyncifyProf.referenceLocalDistSha256.wasm
      );
      expect(nodeSha256(fs.readFileSync(path.join(ortDistDir, jsepProf.entryFile)))).toBe(
        jsepProf.referenceLocalDistSha256.entry
      );
      expect(nodeSha256(fs.readFileSync(path.join(ortDistDir, jsepProf.emscriptenMjsFile)))).toBe(
        jsepProf.referenceLocalDistSha256.emscriptenMjs
      );
      expect(nodeSha256(fs.readFileSync(path.join(ortDistDir, jsepProf.wasmFile)))).toBe(
        jsepProf.referenceLocalDistSha256.wasm
      );
    }

    const w130 = fs.readFileSync(path.join(root, 'src/services/hitnetOrt130DiagnosticWorker.ts'), 'utf8');
    const w131 = fs.readFileSync(path.join(root, 'src/services/hitnetOrt131DiagnosticWorker.ts'), 'utf8');
    const prodWorker = fs.readFileSync(path.join(root, 'src/services/stereoDepthWorker.ts'), 'utf8');
    const prodSession = fs.readFileSync(path.join(root, 'src/services/onnxSession.ts'), 'utf8');

    expect(w130).toContain("PINNED_ORT_VERSION = '1.30.0'");
    expect(w130).not.toContain('1.31.1');
    expect(w131).toContain("PINNED_ORT_VERSION = '1.31.1'");
    expect(w131).not.toContain('1.30.0');
    expect(prodWorker).toContain("from 'onnxruntime-web'");
    expect(prodWorker).not.toContain('1.31.1');
    expect(prodSession).not.toContain('1.31.1');
    expect(HITNET_M1_FULL_GRAPH_SHA256).toBe(
      '97e3673bdea94980f71cc1fc13998066238636b0ae0fa7f5381aa3ffa711e758'
    );
  });

  it('hashes exact entry module bytes at fetch+blob-import boundary and cleans up Blob URL', async () => {
    const entryBytes = new TextEncoder().encode('export const ortTest = 1;');
    const expectedHash = nodeSha256(entryBytes);
    let fetchCount = 0;
    const revokedUrls: string[] = [];
    const importedUrls: string[] = [];

    const mockMod = {
      env: { versions: { common: '1.30.0', web: '1.30.0' }, wasm: {} },
      Tensor: class MockT {
        dims = [1];
        type = 'float32';
        data = new Float32Array(1);
      },
      InferenceSession: {
        create: async () => {
          throw new Error('unused');
        },
      },
    } satisfies OrtLikeModule;

    const loaded = await loadOrtEntryModuleWithBoundarySha256(
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.bundle.min.mjs',
      {
        fetchFn: async () => {
          fetchCount++;
          return new Response(entryBytes, { status: 200 });
        },
        sha256Hex: async (u8) => nodeSha256(u8),
        createObjectURL: () => 'blob:mock-ort-entry-1',
        revokeObjectURL: (u) => {
          revokedUrls.push(u);
        },
        importFn: async (u) => {
          importedUrls.push(u);
          return mockMod;
        },
        referenceLocalDistSha256: expectedHash,
      }
    );

    expect(fetchCount).toBe(1);
    expect(importedUrls).toEqual(['blob:mock-ort-entry-1']);
    expect(revokedUrls).toEqual(['blob:mock-ort-entry-1']);
    expect(loaded.entryAsset?.via).toBe('fetch+blob-import');
    expect(loaded.entryAsset?.byteLength).toBe(entryBytes.byteLength);
    expect(loaded.entryAsset?.sha256).toBe(expectedHash);
    expect(loaded.entryAsset?.matchesLocalDistSha256).toBe(true);

    await expect(
      loadOrtEntryModuleWithBoundarySha256('https://example.invalid/missing.mjs', {
        fetchFn: async () => new Response(null, { status: 404, statusText: 'Not Found' }),
      })
    ).rejects.toThrow(/404/);
  });

  it('validates loaded ORT version metadata and classifies asyncify/jsep resources cleanly', () => {
    expect(() => assertLoadedOrtVersionMatches({ common: '1.31.1', web: '1.31.1' }, '1.31.1')).not.toThrow();
    expect(() => assertLoadedOrtVersionMatches({ common: '1.30.0', web: '1.30.0' }, '1.31.1')).toThrow(
      /ORT version mismatch/
    );
    expect(() => assertLoadedOrtVersionMatches(null, '1.30.0')).toThrow(/ORT version mismatch/);

    const classified = classifyOrtLoadedResources(
      [
        'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs',
        'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort-wasm-simd-threaded.asyncify.mjs',
        'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort-wasm-simd-threaded.asyncify.wasm',
        'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.jsep.wasm',
      ],
      HITNET_ORT_VERSION_PROFILES['1.31.1'].distBaseUrl
    );
    expect(classified.loadedOrtResources).toHaveLength(3);
    expect(classified.foreignOrtResources).toEqual([
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.jsep.wasm',
    ]);
  });

  it('runs valid cold + warm repeat WebGPU and WASM sessions with verified hashes, shapes, and dtypes', async () => {
    const capturedSessionOpts: Record<string, unknown>[] = [];
    const dist130 = HITNET_ORT_VERSION_PROFILES['1.30.0'].distBaseUrl;
    let disposedCount = 0;
    let releasedCount = 0;
    let runCounter = 0;

    const mockModule: OrtLikeModule = {
      env: {
        versions: { common: '1.30.0', web: '1.30.0' },
        wasm: {},
        webgpu: {
          adapter: {
            info: { vendor: 'amd', architecture: 'rdna3', device: 'test', description: 'Radeon' },
            features: { has: (name) => name === 'shader-f16' || name === 'timestamp-query' },
          },
          device: {
            pushErrorScope: () => {},
            popErrorScope: async () => null,
            addEventListener: () => {},
            removeEventListener: () => {},
          },
        },
      },
      Tensor: class MockTensor implements OrtLikeTensor {
        dims: readonly number[];
        type: string;
        data: Float32Array | Int32Array | BigInt64Array | Uint8Array;
        constructor(
          dtype: string,
          data: Float32Array | Int32Array | BigInt64Array | Uint8Array,
          dims: number[]
        ) {
          this.type = dtype;
          this.data = data;
          this.dims = dims;
        }
        dispose() {
          disposedCount++;
        }
      },
      InferenceSession: {
        create: async (_bytes, options) => {
          capturedSessionOpts.push(options);
          return {
            inputNames: ['input'],
            outputNames: ['reference_output_disparity', 'secondary_output_disparity'],
            run: async () => {
              const callIdx = runCounter++;
              return {
                reference_output_disparity: {
                  dims: [1, 240, 320, 1],
                  type: 'float32',
                  data: new Float32Array(76800).fill(callIdx === 0 ? 1.25 : 1.5),
                  dispose: () => {
                    disposedCount++;
                  },
                },
                secondary_output_disparity: {
                  dims: [1, 240, 320, 1],
                  type: 'float32',
                  data: new Float32Array(76800).fill(2.5),
                  dispose: () => {
                    disposedCount++;
                  },
                },
              };
            },
            release: async () => {
              releasedCount++;
            },
          };
        },
      },
    };

    const gpuRes = await executeOrtDiagnosticWorkerRun(
      '1.30.0',
      {
        id: 1,
        expectedOrtVersion: '1.30.0',
        modelBuffer: makeValidModelBuffer(),
        feedBuffer: makeValidFeedBuffer(1),
        provider: 'webgpu',
        graphOptimizationLevel: 'basic',
        warmRuns: 2,
        compareColdVsWarmRepeat: true,
      },
      {
        loadOrtModule: async () => mockModule,
        sha256Hex: async (u8) => `sha-${u8.byteLength}`,
        getResourceUrls: () => [
          `${dist130}ort.webgpu.min.mjs`,
          `${dist130}ort-wasm-simd-threaded.asyncify.mjs`,
          `${dist130}ort-wasm-simd-threaded.asyncify.wasm`,
        ],
        requestWebGpuAdapter: async () => mockModule.env.webgpu?.adapter ?? null,
      }
    );

    expect(gpuRes.response.ok).toBe(true);
    if (!gpuRes.response.ok) return;
    expect(gpuRes.response.pinnedOrtVersion).toBe('1.30.0');
    expect(gpuRes.response.runtimeAssetFamily).toBe('asyncify');
    expect(gpuRes.response.configuredProviders).toEqual(['webgpu']);
    expect(gpuRes.response.fallbackBehavior).toBe('no-fallback-explicit-error');
    expect(gpuRes.response.modelByteLength).toBe(8);
    expect(gpuRes.response.feedTensorName).toBe('input');
    expect(gpuRes.response.feedTensorShape).toEqual([1, 6, 240, 320]);
    expect(gpuRes.response.feedTensorDataType).toBe('float32');
    expect(gpuRes.response.sessionInputNames).toEqual(['input']);
    expect(gpuRes.response.sessionOutputNames).toEqual([
      'reference_output_disparity',
      'secondary_output_disparity',
    ]);
    expect(gpuRes.response.warmRunMs).toHaveLength(2);
    expect(gpuRes.response.outputs.map((o) => o.name)).toEqual([
      'reference_output_disparity',
      'secondary_output_disparity',
    ]);
    expect(gpuRes.response.coldOutputs?.map((o) => o.name)).toEqual([
      'reference_output_disparity',
      'secondary_output_disparity',
    ]);
    expect(new Float32Array(gpuRes.response.coldOutputs![0].buffer)[0]).toBe(1.25);
    expect(new Float32Array(gpuRes.response.outputs[0].buffer)[0]).toBe(1.5);
    expect(gpuRes.response.gpuValidationErrors).toEqual([]);
    expect(gpuRes.response.gpuKernelProfileSummary).toBeNull();
    expect(gpuRes.transfers).toHaveLength(4);
    expect(capturedSessionOpts[0].executionProviders).toEqual(['webgpu']);
    expect(capturedSessionOpts[0].graphOptimizationLevel).toBe('basic');
    expect(disposedCount).toBe(9); // 3 input tensors + 6 output tensors across cold + 2 warm
    expect(releasedCount).toBe(1);

    const wasmRes = await executeOrtDiagnosticWorkerRun(
      '1.30.0',
      {
        id: 2,
        expectedOrtVersion: '1.30.0',
        modelBuffer: makeValidModelBuffer(),
        feedBuffer: makeValidFeedBuffer(2),
        provider: 'wasm',
        graphOptimizationLevel: 'disabled',
        warmRuns: 0,
      },
      {
        loadOrtModule: async () => mockModule,
        sha256Hex: async (u8) => `sha-${u8.byteLength}`,
        getResourceUrls: () => [`${dist130}ort.webgpu.min.mjs`, `${dist130}ort-wasm-simd-threaded.wasm`],
      }
    );
    expect(wasmRes.response.ok).toBe(true);
    if (wasmRes.response.ok) {
      expect(wasmRes.response.configuredProviders).toEqual(['wasm']);
      expect(wasmRes.response.graphOptimizationLevel).toBe('disabled');
      expect(wasmRes.response.gpuValidationErrors).toBeNull();
      expect(wasmRes.response.gpuKernelProfileSummary).toBeNull();
    }
  });

  it('captures JSEP runtime-boundary WASM SHA-256 via WebAssembly.instantiateStreaming, restores globals, and guards family assets', async () => {
    const jsepProf = HITNET_ORT_130_FAMILY_PROFILES.jsep;
    const dist130 = jsepProf.distBaseUrl;
    const fakeWasmBytes = new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
    const expectedWasmHash = nodeSha256(fakeWasmBytes);
    const fakeEntryHash = jsepProf.referenceLocalDistSha256.entry;

    const origFetch = globalThis.fetch;
    const origInstantiateStreaming = WebAssembly.instantiateStreaming;
    const origInstantiate = WebAssembly.instantiate;
    let capturedOpts: Record<string, unknown> | null = null;
    let wasmStreamConsumedCorrectly = false;

    const fakeWasmFetch: typeof fetch = async (input) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      const resp = new Response(fakeWasmBytes, { status: 200 });
      Object.defineProperty(resp, 'url', { value: url });
      return resp;
    };
    const fakeInstantiateStreaming = async (
      source: Response | PromiseLike<Response>
    ): Promise<WebAssembly.WebAssemblyInstantiatedSource> => {
      const r = await source;
      const buf = await r.arrayBuffer();
      wasmStreamConsumedCorrectly = buf.byteLength === fakeWasmBytes.byteLength;
      return {
        module: {} as WebAssembly.Module,
        instance: {} as WebAssembly.Instance,
      };
    };

    globalThis.fetch = fakeWasmFetch;
    WebAssembly.instantiateStreaming = fakeInstantiateStreaming;

    try {
      const mockJsepModule: OrtLikeModule = {
        env: {
          versions: { common: '1.30.0', web: '1.30.0' },
          wasm: {},
          webgpu: {
            adapter: {
              info: { vendor: 'amd', architecture: 'rdna3', device: 'test', description: 'Radeon' },
              features: { has: () => true },
            },
            device: {
              pushErrorScope: () => {},
              popErrorScope: async () => null,
              addEventListener: () => {},
              removeEventListener: () => {},
            },
          },
        },
        Tensor: class MockT implements OrtLikeTensor {
          dims: readonly number[];
          type: string;
          data: Float32Array | Int32Array | BigInt64Array | Uint8Array;
          constructor(
            dtype: string,
            data: Float32Array | Int32Array | BigInt64Array | Uint8Array,
            dims: number[]
          ) {
            this.type = dtype;
            this.data = data;
            this.dims = dims;
          }
        },
        InferenceSession: {
          create: async (_bytes, options) => {
            capturedOpts = options;
            const resp = await globalThis.fetch(`${dist130}${jsepProf.wasmFile}`);
            await WebAssembly.instantiateStreaming(resp);
            return {
              inputNames: ['input'],
              outputNames: ['reference_output_disparity', 'secondary_output_disparity'],
              run: async () => ({
                reference_output_disparity: {
                  dims: [1, 240, 320, 1],
                  type: 'float32',
                  data: new Float32Array(76800).fill(3.0),
                },
                secondary_output_disparity: {
                  dims: [1, 240, 320, 1],
                  type: 'float32',
                  data: new Float32Array(76800).fill(4.0),
                },
              }),
              release: async () => {},
            };
          },
        },
      };

      const jsepRes = await executeOrtDiagnosticWorkerRun(
        '1.30.0',
        {
          id: 20,
          expectedOrtVersion: '1.30.0',
          runtimeAssetFamily: 'jsep',
          modelBuffer: makeValidModelBuffer(),
          feedBuffer: makeValidFeedBuffer(4),
          provider: 'webgpu',
          graphOptimizationLevel: 'basic',
          warmRuns: 1,
          disableCpuEpFallback: true,
        },
        {
          loadOrtModule: async (entryUrl, family) => {
            expect(entryUrl).toBe(`${dist130}${jsepProf.entryFile}`);
            expect(family).toBe('jsep');
            return {
              module: mockJsepModule,
              entryAsset: {
                role: 'entry',
                url: entryUrl,
                byteLength: 413523,
                sha256: fakeEntryHash,
                via: 'fetch+blob-import',
                referenceLocalDistSha256: fakeEntryHash,
                matchesLocalDistSha256: true,
              },
            };
          },
          sha256Hex: async (u8) => nodeSha256(u8),
          getResourceEntries: () => [
            { name: `${dist130}ort-wasm-simd-threaded.jsep.mjs`, decodedBodySize: 162165 },
            { name: `${dist130}ort-wasm-simd-threaded.jsep.wasm`, decodedBodySize: 28000000 },
          ],
          requestWebGpuAdapter: async () => mockJsepModule.env.webgpu?.adapter ?? null,
        }
      );

      expect(wasmStreamConsumedCorrectly).toBe(true);
      expect(globalThis.fetch).toBe(fakeWasmFetch);
      expect(WebAssembly.instantiateStreaming).toBe(fakeInstantiateStreaming);
      expect(WebAssembly.instantiate).toBe(origInstantiate);

      expect(jsepRes.response.ok).toBe(true);
      if (!jsepRes.response.ok) return;
      expect(jsepRes.response.runtimeAssetFamily).toBe('jsep');
      expect(jsepRes.response.entryModuleUrl).toBe(`${dist130}${jsepProf.entryFile}`);
      expect(jsepRes.response.sessionOptionsApplied).toEqual({
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'basic',
        enableCpuMemArena: true,
        enableMemPattern: true,
        numThreads: 1,
        proxy: false,
        wasmPaths: dist130,
        disableCpuEpFallback: true,
        fetchMode: 'all-session-outputs',
        profilingRequested: false,
      });
      expect((capturedOpts as Record<string, unknown> | null)?.extra).toEqual({
        session: { disable_cpu_ep_fallback: '1' },
      });

      const entryProv = jsepRes.response.loadedAssetProvenance.find((p) => p.role === 'entry');
      const wasmProv = jsepRes.response.loadedAssetProvenance.find((p) => p.role === 'wasm');
      const mjsProv = jsepRes.response.loadedAssetProvenance.find((p) => p.role === 'emscripten-mjs');
      expect(entryProv?.sha256).toBe(fakeEntryHash);
      expect(entryProv?.matchesLocalDistSha256).toBe(true);
      expect(wasmProv?.via).toBe('WebAssembly.instantiateStreaming');
      expect(wasmProv?.sha256).toBe(expectedWasmHash);
      expect(wasmProv?.byteLength).toBe(fakeWasmBytes.byteLength);
      expect(wasmProv?.url).toBe(`${dist130}${jsepProf.wasmFile}`);
      expect(mjsProv?.via).toBe('performance-resource-entry');
      expect(mjsProv?.sha256).toBeNull();
      expect(mjsProv?.note).toContain('native dynamic import');

      // Family guard rejects asyncify assets when runtimeAssetFamily='jsep'
      const contaminatedRes = await executeOrtDiagnosticWorkerRun(
        '1.30.0',
        {
          id: 21,
          expectedOrtVersion: '1.30.0',
          runtimeAssetFamily: 'jsep',
          modelBuffer: makeValidModelBuffer(),
          feedBuffer: makeValidFeedBuffer(4),
          provider: 'webgpu',
          graphOptimizationLevel: 'basic',
        },
        {
          loadOrtModule: async () => mockJsepModule,
          sha256Hex: async (u8) => nodeSha256(u8),
          getResourceEntries: () => [
            { name: `${dist130}ort-wasm-simd-threaded.asyncify.mjs` },
            { name: `${dist130}ort-wasm-simd-threaded.asyncify.wasm` },
          ],
          requestWebGpuAdapter: async () => mockJsepModule.env.webgpu?.adapter ?? null,
        }
      );
      expect(contaminatedRes.response.ok).toBe(false);
      if (!contaminatedRes.response.ok) {
        expect(contaminatedRes.response.stage).toBe('resource-guard');
        expect(contaminatedRes.response.message).toContain('Asset family mismatch for "jsep"');
      }

      // Reject JSEP family on 1.31.1
      const wrongVerFamilyRes = await executeOrtDiagnosticWorkerRun(
        '1.31.1',
        {
          id: 22,
          expectedOrtVersion: '1.31.1',
          runtimeAssetFamily: 'jsep',
          modelBuffer: makeValidModelBuffer(),
          feedBuffer: makeValidFeedBuffer(4),
          provider: 'webgpu',
          graphOptimizationLevel: 'basic',
        },
        {
          loadOrtModule: async () => mockJsepModule,
        }
      );
      expect(wrongVerFamilyRes.response.ok).toBe(false);
      if (!wrongVerFamilyRes.response.ok) {
        expect(wrongVerFamilyRes.response.stage).toBe('init');
        expect(wrongVerFamilyRes.response.message).toContain('only defined on pinned ORT 1.30.0');
      }
    } finally {
      globalThis.fetch = origFetch;
      WebAssembly.instantiateStreaming = origInstantiateStreaming;
      WebAssembly.instantiate = origInstantiate;
    }
  });

  it('captures WebGPU kernel dispatch and profiling evidence when enableWebGpuProfiling is true and restores device hooks', async () => {
    const dist130 = HITNET_ORT_VERSION_PROFILES['1.30.0'].distBaseUrl;
    let endProfilingCalled = 0;

    const protoSubmit = () => {};
    const queue = Object.create({ submit: protoSubmit }) as { submit: (bufs: readonly unknown[]) => void };
    const gpuDevice: OrtLikeGpuDevice = {
      pushErrorScope: () => {},
      popErrorScope: async () => null,
      addEventListener: () => {},
      removeEventListener: () => {},
      createQuerySet: () => ({}),
      createCommandEncoder: () => ({
        beginComputePass: () => ({
          dispatchWorkgroups: () => {},
          dispatchWorkgroupsIndirect: () => {},
        }),
      }),
      queue,
    };

    const mockModule: OrtLikeModule = {
      env: {
        versions: { common: '1.30.0', web: '1.30.0' },
        wasm: {},
        webgpu: {
          adapter: {
            info: { vendor: 'amd', architecture: 'rdna3', device: 'test', description: 'Radeon' },
            features: { has: (name) => name === 'timestamp-query' },
          },
          device: gpuDevice,
        },
      },
      Tensor: class MockT implements OrtLikeTensor {
        dims: readonly number[];
        type: string;
        data: Float32Array | Int32Array | BigInt64Array | Uint8Array;
        constructor(
          dtype: string,
          data: Float32Array | Int32Array | BigInt64Array | Uint8Array,
          dims: number[]
        ) {
          this.type = dtype;
          this.data = data;
          this.dims = dims;
        }
      },
      InferenceSession: {
        create: async () => ({
          inputNames: ['input'],
          outputNames: ['reference_output_disparity', 'secondary_output_disparity'],
          run: async () => {
            mockModule.env.webgpu?.profiling?.ondata?.({
              kernelType: 'Conv',
              programName: 'Conv2dNaive',
            });
            gpuDevice.createQuerySet?.();
            const enc = gpuDevice.createCommandEncoder?.();
            const pass = enc?.beginComputePass?.({ label: 'Conv_0', timestampWrites: {} });
            pass?.dispatchWorkgroups?.(10, 10, 1);
            pass?.dispatchWorkgroups?.(5, 5, 1);
            gpuDevice.queue?.submit?.(['cmd0']);
            return {
              reference_output_disparity: {
                dims: [1, 240, 320, 1],
                type: 'float32',
                data: new Float32Array(76800),
              },
              secondary_output_disparity: {
                dims: [1, 240, 320, 1],
                type: 'float32',
                data: new Float32Array(76800),
              },
            };
          },
          endProfiling: () => {
            endProfilingCalled++;
          },
          release: async () => {},
        }),
      },
    };

    const res = await executeOrtDiagnosticWorkerRun(
      '1.30.0',
      {
        id: 10,
        expectedOrtVersion: '1.30.0',
        modelBuffer: makeValidModelBuffer(),
        feedBuffer: makeValidFeedBuffer(3),
        provider: 'webgpu',
        graphOptimizationLevel: 'basic',
        warmRuns: 0,
        enableWebGpuProfiling: true,
      },
      {
        loadOrtModule: async () => mockModule,
        sha256Hex: async (u8) => `sha-${u8.byteLength}`,
        getResourceUrls: () => [`${dist130}ort.webgpu.min.mjs`, `${dist130}ort-wasm-simd-threaded.asyncify.wasm`],
        requestWebGpuAdapter: async () => mockModule.env.webgpu?.adapter ?? null,
      }
    );

    expect(res.response.ok).toBe(true);
    if (!res.response.ok) return;
    const kp = res.response.gpuKernelProfileSummary;
    expect(kp).not.toBeNull();
    expect(kp?.profilingRequested).toBe(true);
    expect(kp?.timestampQuerySupported).toBe(true);
    expect(kp?.deviceObserverAttached).toBe(true);
    expect(kp?.webgpuComputePassCount).toBe(1);
    expect(kp?.webgpuDispatchWorkgroupsCount).toBe(2);
    expect(kp?.webgpuTimestampWritePassCount).toBe(1);
    expect(kp?.webgpuQueueSubmitCount).toBe(1);
    expect(kp?.webgpuCommandBufferCount).toBe(1);
    expect(kp?.webgpuQuerySetCount).toBe(1);
    expect(kp?.webgpuSamplePassLabels).toEqual(['Conv_0']);
    expect(kp?.ortProfilingEventCount).toBe(1);
    expect(kp?.ortProfilingKernelTypes).toEqual({ Conv: 1 });
    expect(kp?.note).toContain('ORT Web does not expose per-node EP placement in JS');
    expect(endProfilingCalled).toBe(1);
    // Verify env & observer restoration after profiled run
    expect(mockModule.env.logLevel).toBe('error');
    expect(mockModule.env.webgpu?.profiling?.mode).toBe('off');
    expect(Object.prototype.hasOwnProperty.call(queue, 'submit')).toBe(false);
  });

  it('fails explicitly without WASM fallback when WebGPU is unavailable or versions mismatch', async () => {
    const noAdapterRes = await executeOrtDiagnosticWorkerRun(
      '1.31.1',
      {
        id: 3,
        expectedOrtVersion: '1.31.1',
        modelBuffer: makeValidModelBuffer(),
        feedBuffer: makeValidFeedBuffer(),
        provider: 'webgpu',
        graphOptimizationLevel: 'basic',
      },
      {
        loadOrtModule: async () => {
          throw new Error('Should not load ORT when adapter is missing');
        },
        requestWebGpuAdapter: async () => null,
      }
    );
    expect(noAdapterRes.response.ok).toBe(false);
    if (!noAdapterRes.response.ok) {
      expect(noAdapterRes.response.stage).toBe('webgpu-support');
      expect(noAdapterRes.response.configuredProviders).toEqual(['webgpu']);
      expect(noAdapterRes.response.fallbackBehavior).toBe('no-fallback-explicit-error');
    }

    const wrongVersionRes = await executeOrtDiagnosticWorkerRun(
      '1.31.1',
      {
        id: 4,
        expectedOrtVersion: '1.31.1',
        modelBuffer: makeValidModelBuffer(),
        feedBuffer: makeValidFeedBuffer(),
        provider: 'wasm',
        graphOptimizationLevel: 'disabled',
      },
      {
        loadOrtModule: async () => ({
          env: { versions: { common: '1.30.0', web: '1.30.0' }, wasm: {} },
          Tensor: class MockT {
            dims = [1];
            type = 'float32';
            data = new Float32Array(1);
          },
          InferenceSession: {
            create: async () => {
              throw new Error('Should not create session on version mismatch');
            },
          },
        }),
      }
    );
    expect(wrongVersionRes.response.ok).toBe(false);
    if (!wrongVersionRes.response.ok) {
      expect(wrongVersionRes.response.stage).toBe('version-guard');
    }
  });
});
