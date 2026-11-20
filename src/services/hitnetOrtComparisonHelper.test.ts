import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  assertLoadedOrtVersionMatches,
  classifyOrtLoadedResources,
  executeOrtDiagnosticWorkerRun,
  HITNET_ORT_VERSION_PROFILES,
  type OrtLikeModule,
  type OrtLikeTensor,
} from './hitnetOrtComparisonHelper';
import { HITNET_M1_FULL_GRAPH_SHA256 } from './stereoDepthWorker';

function makeValidFeedBuffer(): ArrayBuffer {
  const f32 = new Float32Array(1 * 6 * 240 * 320);
  for (let i = 0; i < f32.length; i += 997) {
    f32[i] = (i % 255) / 255;
  }
  return f32.buffer;
}

function makeValidModelBuffer(): ArrayBuffer {
  return new Uint8Array([0x08, 0x07, 0x12, 0x04, 0x74, 0x65, 0x73, 0x74]).buffer;
}

describe('hitnetOrtComparisonHelper (isolated 1.30.0 vs 1.31.1 M1 diagnostic)', () => {
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

    const root = process.cwd();
    const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')) as {
      devDependencies?: Record<string, string>;
      dependencies?: Record<string, string>;
    };
    expect(pkg.devDependencies?.['onnxruntime-web']).toBe('1.30.0');
    expect(pkg.dependencies?.['onnxruntime-web']).toBeUndefined();

    const w130 = fs.readFileSync(path.join(root, 'src/services/hitnetOrt130DiagnosticWorker.ts'), 'utf8');
    const w131 = fs.readFileSync(path.join(root, 'src/services/hitnetOrt131DiagnosticWorker.ts'), 'utf8');
    const prodWorker = fs.readFileSync(path.join(root, 'src/services/stereoDepthWorker.ts'), 'utf8');
    const prodSession = fs.readFileSync(path.join(root, 'src/services/onnxSession.ts'), 'utf8');

    expect(w130).toContain("PINNED_ORT_VERSION = '1.30.0'");
    expect(w130).not.toContain('1.31.1');
    expect(w131).toContain("PINNED_ORT_VERSION = '1.31.1'");
    expect(w131).not.toContain('1.30.0');
    expect(prodWorker).not.toContain('1.31.1');
    expect(prodSession).not.toContain('1.31.1');
    expect(HITNET_M1_FULL_GRAPH_SHA256).toBe(
      '97e3673bdea94980f71cc1fc13998066238636b0ae0fa7f5381aa3ffa711e758'
    );
  });

  it('validates loaded ORT version metadata and detects foreign version resource loads', () => {
    expect(() => assertLoadedOrtVersionMatches({ common: '1.31.1', web: '1.31.1' }, '1.31.1')).not.toThrow();
    expect(() => assertLoadedOrtVersionMatches({ common: '1.30.0', web: '1.30.0' }, '1.31.1')).toThrow(
      /ORT version mismatch/
    );
    expect(() => assertLoadedOrtVersionMatches(null, '1.30.0')).toThrow(/ORT version mismatch/);

    const classified = classifyOrtLoadedResources(
      [
        'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs',
        'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort-wasm-simd-threaded.jsep.wasm',
        'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.jsep.wasm',
      ],
      HITNET_ORT_VERSION_PROFILES['1.31.1'].distBaseUrl
    );
    expect(classified.loadedOrtResources).toHaveLength(2);
    expect(classified.foreignOrtResources).toEqual([
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.jsep.wasm',
    ]);
  });

  it('runs valid cold + warm WebGPU and WASM sessions with explicit no-fallback EP config and preserves both final tensors', async () => {
    const capturedSessionOpts: Record<string, unknown>[] = [];
    const dist131 = HITNET_ORT_VERSION_PROFILES['1.31.1'].distBaseUrl;
    let disposedCount = 0;
    let releasedCount = 0;

    const mockModule: OrtLikeModule = {
      env: {
        versions: { common: '1.31.1', web: '1.31.1' },
        wasm: {},
        webgpu: {
          adapter: {
            info: { vendor: 'amd', architecture: 'rdna3', device: 'test', description: 'Radeon' },
            features: { has: (name) => name === 'shader-f16' },
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
        data: Float32Array;
        constructor(dtype: string, data: Float32Array, dims: number[]) {
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
            run: async () => ({
              reference_output_disparity: {
                dims: [1, 240, 320, 1],
                type: 'float32',
                data: new Float32Array(76800).fill(1.25),
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
            }),
            release: async () => {
              releasedCount++;
            },
          };
        },
      },
    };

    const gpuRes = await executeOrtDiagnosticWorkerRun(
      '1.31.1',
      {
        id: 1,
        expectedOrtVersion: '1.31.1',
        modelBuffer: makeValidModelBuffer(),
        feedBuffer: makeValidFeedBuffer(),
        provider: 'webgpu',
        graphOptimizationLevel: 'basic',
        warmRuns: 1,
      },
      {
        loadOrtModule: async () => mockModule,
        sha256Hex: async (u8) => `sha-${u8.byteLength}`,
        getResourceUrls: () => [
          `${dist131}ort.webgpu.min.mjs`,
          `${dist131}ort-wasm-simd-threaded.jsep.mjs`,
          `${dist131}ort-wasm-simd-threaded.jsep.wasm`,
        ],
        requestWebGpuAdapter: async () => mockModule.env.webgpu?.adapter ?? null,
      }
    );

    expect(gpuRes.response.ok).toBe(true);
    if (!gpuRes.response.ok) return;
    expect(gpuRes.response.pinnedOrtVersion).toBe('1.31.1');
    expect(gpuRes.response.configuredProviders).toEqual(['webgpu']);
    expect(gpuRes.response.fallbackBehavior).toBe('no-fallback-explicit-error');
    expect(gpuRes.response.warmRunMs).toHaveLength(1);
    expect(gpuRes.response.outputs.map((o) => o.name)).toEqual([
      'reference_output_disparity',
      'secondary_output_disparity',
    ]);
    expect(gpuRes.response.gpuValidationErrors).toEqual([]);
    expect(gpuRes.transfers).toHaveLength(2);
    expect(capturedSessionOpts[0].executionProviders).toEqual(['webgpu']);
    expect(capturedSessionOpts[0].graphOptimizationLevel).toBe('basic');
    expect(disposedCount).toBe(6); // 2 input tensors + 4 output tensors across cold+warm
    expect(releasedCount).toBe(1);

    const wasmRes = await executeOrtDiagnosticWorkerRun(
      '1.31.1',
      {
        id: 2,
        expectedOrtVersion: '1.31.1',
        modelBuffer: makeValidModelBuffer(),
        feedBuffer: makeValidFeedBuffer(),
        provider: 'wasm',
        graphOptimizationLevel: 'disabled',
        warmRuns: 0,
      },
      {
        loadOrtModule: async () => mockModule,
        sha256Hex: async (u8) => `sha-${u8.byteLength}`,
        getResourceUrls: () => [`${dist131}ort.webgpu.min.mjs`, `${dist131}ort-wasm-simd-threaded.wasm`],
      }
    );
    expect(wasmRes.response.ok).toBe(true);
    if (wasmRes.response.ok) {
      expect(wasmRes.response.configuredProviders).toEqual(['wasm']);
      expect(wasmRes.response.graphOptimizationLevel).toBe('disabled');
      expect(wasmRes.response.gpuValidationErrors).toBeNull();
    }
  });

  it('fails explicitly without WASM fallback when WebGPU is unavailable or resources mismatch version', async () => {
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
