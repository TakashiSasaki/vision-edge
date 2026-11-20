import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type * as ort from 'onnxruntime-web';
import {
  cancelInFlightStereoInit,
  clearLocalStereoOnnxBytes,
  clearRemoteStereoOnnxCache,
  computeDisparityStats,
  computeSadDisparity,
  createStereoDepthTask,
  createStereoWorkerProcessor,
  createWebGpuStereoComputeEngine,
  decomposeOnnx5dConvNodesForWebGpu,
  disparityToDepthMeters,
  exceedsWebGpuStorageBufferLimit,
  extractDisparityFromOrtTensor,
  getColormapLut,
  getLocalStereoOnnxInfo,
  inferOnnxStaticTensorMeta,
  inspectDisparityMapAtPoint,
  inspectStereoOnnxWebGpuCompatibility,
  isWebGpuRuntimeKernelError,
  normalizeHitnetArgMinKeepDimsForWebGpu,
  normalizeHitnetGatherNdRankForWebGpu,
  normalizeHitnetPropGatherNdRankForWebGpu,
  normalizeNonAscendingSliceAxesForWebGpu,
  registerLocalStereoOnnxBytes,
  resolveStereoOrtDelegate,
  resolveStereoOrtSessionOptions,
  splitLargeOnnxConcatNodes,
  splitSideBySideFrame,
  STEREO_MODEL_PROFILES,
  StereoOnnxEngine,
  StereoSadEngine,
  stripOnnxGraphValueInfo,
  validateOnnxGraphNodeInputs,
  type StereoWorkerLike,
  type StereoWorkerRequest,
  type StereoWorkerResponse,
} from './stereoDepthService';
import * as onnxSessionModule from './onnxSession';
import {
  HITNET_H1_FULL_GRAPH_SHA256,
  HITNET_M1_FULL_GRAPH_SHA256,
  HITNET_RAW_MODEL_SHA256,
  normalizeHitnetInitToPropZeroSlantForWebGpu,
  normalizeHitnetSlot0ReshapeToTransposeForWebGpu,
} from './stereoDepthWorker';
import { VisionEngine, type VisionTaskFactory } from './visionEngine';
import type { StereoDepthMapData } from '../types/vision';

function makeTexturedRgba(width: number, height: number, shiftXFn: (x: number, y: number) => number): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4);
  const pattern = (sx: number, sy: number) => {
    const v1 = Math.sin(sx * 0.45) * 60 + Math.cos(sy * 0.35) * 45;
    const v2 = ((Math.floor(sx / 4) * 37 + Math.floor(sy / 4) * 97) % 80) - 40;
    return Math.max(0, Math.min(255, Math.round(128 + v1 + v2)));
  };
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const shift = shiftXFn(x, y);
      const val = pattern(x + shift, y);
      const o = (y * width + x) * 4;
      rgba[o] = val;
      rgba[o + 1] = val;
      rgba[o + 2] = val;
      rgba[o + 3] = 255;
    }
  }
  return rgba;
}

describe('stereoDepthService', () => {
  afterEach(() => {
    clearLocalStereoOnnxBytes('lightstereo-s');
    clearLocalStereoOnnxBytes('mobilestereonet-2d');
    clearLocalStereoOnnxBytes('hitnet-sf');
    clearLocalStereoOnnxBytes('crestereo-init-iter2');
    clearLocalStereoOnnxBytes('crestereo-combined-iter5');
    clearLocalStereoOnnxBytes('custom-onnx');
    clearRemoteStereoOnnxCache();
    vi.restoreAllMocks();
  });

  it('estimates accurate disparity on textured stereo pairs with known horizontal shifts', () => {
    const w = 96;
    const h = 48;
    // Left image has sample at sx = x. Right image shifts foreground patch left by 10px (sx = x + 10)
    // and background by 2px (sx = x + 2) so R(x - d) == L(x) matches d = 10 in center and d = 2 on sides.
    const left = makeTexturedRgba(w, h, () => 0);
    const right = makeTexturedRgba(w, h, (x, y) => {
      const inBox = x >= 28 && x < 68 && y >= 12 && y < 36;
      return inBox ? 10 : 2;
    });

    const out = computeSadDisparity(left, right, w, h, {
      maxDisparity: 24,
      windowSize: 7,
      confidenceThreshold: 0.35,
    });

    expect(out.width).toBe(w);
    expect(out.height).toBe(h);
    expect(out.validRatio).toBeGreaterThan(0.75);

    // Center of foreground box in left image (x ~ 48, y ~ 24) shifts to x ~ 38 in right image => disparity ~ 10
    const centerDisp = out.disparity[24 * w + 48];
    expect(centerDisp).toBeGreaterThanOrEqual(8.5);
    expect(centerDisp).toBeLessThanOrEqual(11.5);

    // Background point (x = 80, y = 8) => disparity ~ 2
    const bgDisp = out.disparity[8 * w + 80];
    expect(bgDisp).toBeGreaterThanOrEqual(1.0);
    expect(bgDisp).toBeLessThanOrEqual(3.0);
  });

  it('converts pixel disparity to calibrated depth in meters via Z = (f * B) / d', () => {
    // f = 500px, B = 60mm (0.06m), d = 15px => Z = 30 / 15 = 2.0m
    expect(disparityToDepthMeters(15, 60, 500)).toBeCloseTo(2.0, 5);
    expect(disparityToDepthMeters(30, 60, 500)).toBeCloseTo(1.0, 5);
    expect(disparityToDepthMeters(0, 60, 500)).toBeNull();
    expect(disparityToDepthMeters(-5, 60, 500)).toBeNull();
    expect(disparityToDepthMeters(10, 0, 500)).toBeNull();
  });

  it('generates 256-entry RGB colormap LUTs and computes disparity percentile bounds', () => {
    for (const cmap of ['turbo', 'viridis', 'inferno', 'grayscale'] as const) {
      const lut = getColormapLut(cmap);
      expect(lut).toHaveLength(256 * 3);
      expect(lut[0]).toBeGreaterThanOrEqual(0);
      expect(lut[255 * 3]).toBeLessThanOrEqual(255);
    }

    const disp = new Float32Array([2, 4, 8, 12, 16, 20, -1]);
    const stats = computeDisparityStats(disp, true, 0, 64);
    expect(stats.minDisparity).toBe(2);
    expect(stats.maxDisparity).toBe(20);
    expect(stats.validRatio).toBeCloseTo(6 / 7, 4);
    expect(stats.scaleMinDisparity).toBeGreaterThanOrEqual(2);
    expect(stats.scaleMaxDisparity).toBeLessThanOrEqual(20);
  });

  it('extracts disparity from [1,1,H,W], [1,H,W], HITNet [1,H,W,1], and 2-channel CREStereo [1,2,H,W] tensors', () => {
    const t1 = extractDisparityFromOrtTensor({
      dims: [1, 1, 2, 2],
      data: new Float32Array([4.5, 9.0, 12.0, -1]),
    });
    expect(t1.width).toBe(2);
    expect(t1.height).toBe(2);
    expect(Array.from(t1.disparity)).toEqual([4.5, 9.0, 12.0, -1]);

    // HITNet NHWC [1, H, W, 1]
    const tNhwc = extractDisparityFromOrtTensor({
      dims: [1, 2, 3, 1],
      data: new Float32Array([1, 2, 3, 4, 5, 6]),
    });
    expect(tNhwc.height).toBe(2);
    expect(tNhwc.width).toBe(3);
    expect(Array.from(tNhwc.disparity)).toEqual([1, 2, 3, 4, 5, 6]);

    // CREStereo negative horizontal flow in channel 0
    const tFlow = extractDisparityFromOrtTensor({
      dims: [1, 2, 2, 2],
      data: new Float32Array([-5, -10, -15, -20, 0, 0, 0, 0]),
    });
    expect(Array.from(tFlow.disparity)).toEqual([5, 10, 15, 20]);
  });

  it('inspects depth coordinates and rejects out-of-bounds queries', () => {
    const dummyCanvas = document.createElement('canvas');
    const map: StereoDepthMapData = {
      width: 4,
      height: 2,
      disparity: new Float32Array([4, 8, 12, 16, -1, 10, 20, 30]),
      minDisparity: 4,
      maxDisparity: 30,
      medianDisparity: 12,
      scaleMinDisparity: 4,
      scaleMaxDisparity: 30,
      validRatio: 7 / 8,
      overlayCanvas: dummyCanvas,
      leftCanvas: dummyCanvas,
      rightCanvas: dummyCanvas,
      colormap: 'turbo',
      viewMode: 'left-overlay',
      swapLeftRight: false,
      sourceEyeWidth: 400,
      sourceEyeHeight: 200,
    };

    const probe = inspectDisparityMapAtPoint(map, 0.75, 0.75, 60, 500);
    expect(probe).toMatchObject({
      pixelX: 300,
      pixelY: 150,
      disparityPx: 30,
      valid: true,
    });
    expect(probe?.depthMeters).toBeCloseTo(1.0, 4);

    const invalidProbe = inspectDisparityMapAtPoint(map, 0.1, 0.75, 60, 500);
    expect(invalidProbe).toMatchObject({ valid: false, disparityPx: 0, depthMeters: null });
    expect(inspectDisparityMapAtPoint(map, -0.1, 0.5, 60, 500)).toBeNull();
  });

  it('manages local stereo ONNX weights and fails clearly when a local-only variant has no file loaded', async () => {
    await expect(createStereoDepthTask({ variantId: 'custom-onnx' })).rejects.toThrow(
      /custom-onnx の ONNX モデルファイル/
    );

    const sampleBytes = new Uint8Array([1, 2, 3, 4]);
    registerLocalStereoOnnxBytes('lightstereo-s', sampleBytes, 'my_lightstereo.onnx');
    expect(getLocalStereoOnnxInfo('lightstereo-s')).toMatchObject({
      variantId: 'lightstereo-s',
      fileName: 'my_lightstereo.onnx',
      size: 4,
    });

    clearLocalStereoOnnxBytes('lightstereo-s');
    expect(getLocalStereoOnnxInfo('lightstereo-s')).toBeNull();
  });

  it('strips GraphProto.value_info fields cleanly while preserving non-protobuf buffers', () => {
    // Non-protobuf tiny bytes remain untouched
    const raw = new Uint8Array([1, 2, 3, 4]);
    expect(stripOnnxGraphValueInfo(raw)).toBe(raw);

    // Construct minimal ModelProto:
    // field 1 (ir_version=8): 0x08 0x08
    // field 7 (graph): tag 0x3a, length, payload containing:
    //   field 2 (name='g'): 0x12 0x01 0x67
    //   field 13 (value_info='v1234567890'): 0x6a 0x0b + 11 bytes
    const graphNode = [0x12, 0x01, 0x67];
    const valueInfo = [0x6a, 0x0b, 0x76, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x30];
    const graphPayload = [...graphNode, ...valueInfo];
    const modelBytes = new Uint8Array([0x08, 0x08, 0x3a, graphPayload.length, ...graphPayload]);

    const stripped = stripOnnxGraphValueInfo(modelBytes);
    expect(Array.from(stripped)).toEqual([0x08, 0x08, 0x3a, graphNode.length, ...graphNode]);
  });

  it('downloads LightStereo-S online with fallback and reuses cached weights across initializations', async () => {
    const fakeOnnxBytes = new Uint8Array(2048).fill(0x11);
    const mockFetch = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(new Response('Not found', { status: 404 }))
      .mockResolvedValueOnce(new Response(fakeOnnxBytes, { status: 200 }));

    const mockSession = {
      inputNames: ['left_img', 'right_img'],
      outputNames: ['disp_pred'],
      run: vi.fn(),
      release: vi.fn(async () => {}),
    } as unknown as ort.InferenceSession;

    const createSpy = vi.spyOn(onnxSessionModule, 'createOrtSession').mockResolvedValue({
      session: mockSession,
      backend: {
        runtime: 'onnxruntime-web',
        requestedProvider: 'webgpu',
        configuredProviders: ['wasm'],
        fallbackReason: null,
        actualBackend: 'unmeasured',
      },
    });

    const recordedSources = new Map<string, string>();
    const task = await createStereoDepthTask({
      variantId: 'lightstereo-s',
      fetchImpl: mockFetch,
      recordAssetSource: (k, s) => recordedSources.set(k, s),
    });

    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(mockFetch.mock.calls[0][0]).toBe(STEREO_MODEL_PROFILES['lightstereo-s'].defaultModelUrl);
    expect(mockFetch.mock.calls[1][0]).toBe(STEREO_MODEL_PROFILES['lightstereo-s'].fallbackModelUrl);
    expect(createSpy).toHaveBeenCalledTimes(1);
    expect(createSpy.mock.calls[0][1]).toBe('WebGPU');
    expect(createSpy.mock.calls[0][2]?.primarySource).toBe('external-fallback');
    await task.close();

    // Second creation reuses the in-memory remote cache without fetching again
    const task2 = await createStereoDepthTask({
      variantId: 'lightstereo-s',
      fetchImpl: mockFetch,
    });
    expect(mockFetch).toHaveBeenCalledTimes(2);
    await task2.close();
  });

  it('splits oversized ONNX Concat nodes into <=4-input subtrees to enable WebGPU execution', async () => {
    const writeVarint = (val: number): number[] => {
      const out: number[] = [];
      let v = val >>> 0;
      while (v >= 0x80) {
        out.push((v & 0x7f) | 0x80);
        v >>>= 7;
      }
      out.push(v);
      return out;
    };
    const encodeLenField = (fieldNum: number, payload: number[]): number[] => [
      ...writeVarint((fieldNum << 3) | 2),
      ...writeVarint(payload.length),
      ...payload,
    ];
    const strBytes = (s: string): number[] => Array.from(s).map((ch) => ch.charCodeAt(0));

    const makeConcatModelProto = (numInputs: number, includeAxisAttr = false): Uint8Array => {
      const nodeParts: number[] = [];
      for (let i = 0; i < numInputs; i++) {
        nodeParts.push(...encodeLenField(1, strBytes(`x${i}`)));
      }
      // NodeProto field 2 (output 'y'), field 3 (name 'BigConcat'), field 4 (op_type 'Concat')
      nodeParts.push(...encodeLenField(2, strBytes('y')));
      nodeParts.push(...encodeLenField(3, strBytes('BigConcat')));
      nodeParts.push(...encodeLenField(4, strBytes('Concat')));
      if (includeAxisAttr) {
        // AttributeProto: field 1 name='axis', field 3 i=1, field 20 type=INT(2)
        const attrPayload = [
          ...encodeLenField(1, strBytes('axis')),
          0x18, 0x01,
          0xa0, 0x01, 0x02,
        ];
        nodeParts.push(...encodeLenField(5, attrPayload));
      }

      const graphPayload = encodeLenField(1, nodeParts);
      return new Uint8Array([0x08, 0x08, ...encodeLenField(7, graphPayload)]);
    };

    const safeConcat = makeConcatModelProto(4); // 4 inputs + 1 output = 5 <= 8
    expect(exceedsWebGpuStorageBufferLimit(safeConcat)).toEqual({
      exceeds: false,
      maxNodeBuffers: 5,
      opType: 'Concat',
    });
    expect(resolveStereoOrtDelegate(STEREO_MODEL_PROFILES['custom-onnx'], safeConcat)).toBe('WebGPU');

    const oversizedConcat = makeConcatModelProto(9); // 9 inputs + 1 output = 10 > 8
    expect(exceedsWebGpuStorageBufferLimit(oversizedConcat)).toEqual({
      exceeds: true,
      maxNodeBuffers: 10,
      opType: 'Concat',
    });
    // Raw unsplit bytes alone would route to WASM
    expect(resolveStereoOrtDelegate(STEREO_MODEL_PROFILES['custom-onnx'], oversizedConcat)).toBe('WASM');

    // Splitting converts a 48-input Concat (as in LightStereo-S) into <=4-input trees, under the 8-buffer limit
    const largeConcat = makeConcatModelProto(48, true);
    expect(exceedsWebGpuStorageBufferLimit(largeConcat).exceeds).toBe(true);
    const splitLarge = splitLargeOnnxConcatNodes(largeConcat, 4);
    expect(splitLarge.splitCount).toBe(1);
    expect(splitLarge.originalMaxConcatInputs).toBe(48);
    expect(exceedsWebGpuStorageBufferLimit(splitLarge.bytes)).toMatchObject({
      exceeds: false,
    });
    expect(exceedsWebGpuStorageBufferLimit(splitLarge.bytes).maxNodeBuffers).toBeLessThanOrEqual(5);
    expect(resolveStereoOrtDelegate(STEREO_MODEL_PROFILES['lightstereo-s'], splitLarge.bytes)).toBe('WebGPU');

    const mockSession = {
      inputNames: ['left', 'right'],
      outputNames: ['disp'],
      run: vi.fn(),
      release: vi.fn(async () => {}),
    } as unknown as ort.InferenceSession;

    const createSpy = vi.spyOn(onnxSessionModule, 'createOrtSession').mockResolvedValue({
      session: mockSession,
      backend: {
        runtime: 'onnxruntime-web',
        requestedProvider: 'webgpu',
        configuredProviders: ['webgpu', 'wasm'],
        fallbackReason: null,
        actualBackend: 'unmeasured',
      },
    });

    const task = await createStereoDepthTask({
      variantId: 'custom-onnx',
      modelBytes: oversizedConcat,
    });
    expect(createSpy).toHaveBeenCalledTimes(1);
    expect(createSpy.mock.calls[0][1]).toBe('WebGPU');
    expect(exceedsWebGpuStorageBufferLimit(createSpy.mock.calls[0][0] as Uint8Array).exceeds).toBe(false);
    await task.close();
  });

  it('surfaces an informative error when both LightStereo-S online download URLs fail', async () => {
    const mockFetch = vi
      .fn<typeof fetch>()
      .mockRejectedValueOnce(new Error('CDN unreachable'))
      .mockResolvedValueOnce(new Response('Server error', { status: 503 }));

    await expect(
      createStereoDepthTask({
        variantId: 'lightstereo-s',
        fetchImpl: mockFetch,
      })
    ).rejects.toThrow(/オンライン取得に失敗しました/);
  });

  it('downloads MobileStereoNet-2D directly via CORS and HITNet/CREStereo via stateless CORS proxy', async () => {
    const fakeOnnxBytes = new Uint8Array(2048).fill(0x22);
    const mockFetch = vi.fn<typeof fetch>(async () => new Response(fakeOnnxBytes, { status: 200 }));
    const mockSession = {
      inputNames: ['left', 'right'],
      outputNames: ['disp'],
      run: vi.fn(),
      release: vi.fn(async () => {}),
    } as unknown as ort.InferenceSession;

    const createSpy = vi.spyOn(onnxSessionModule, 'createOrtSession').mockResolvedValue({
      session: mockSession,
      backend: {
        runtime: 'onnxruntime-web',
        requestedProvider: 'wasm',
        configuredProviders: ['wasm'],
        fallbackReason: null,
        actualBackend: 'unmeasured',
      },
    });

    // 1. MobileStereoNet-2D downloads via same-origin stateless CORS proxy (Google Drive blocks cross-site browser Sec-Fetch-Mode: cors) and runs on WebGPU after 5D Conv3D -> 2D Conv decomposition
    const msTask = await createStereoDepthTask({
      variantId: 'mobilestereonet-2d',
      fetchImpl: mockFetch,
    });
    expect(mockFetch.mock.calls[0][0]).toBe(
      STEREO_MODEL_PROFILES['mobilestereonet-2d'].corsProxyUrl
    );
    expect(createSpy.mock.calls[0][1]).toBe('WebGPU');
    await msTask.close();

    // 2. HITNet & CREStereo download from same-origin stateless /api/cors-proxy/stereo/
    for (const variantId of [
      'hitnet-sf',
      'crestereo-init-iter2',
      'crestereo-combined-iter5',
    ] as const) {
      const task = await createStereoDepthTask({
        variantId,
        fetchImpl: mockFetch,
      });
      expect(mockFetch).toHaveBeenLastCalledWith(
        STEREO_MODEL_PROFILES[variantId].corsProxyUrl,
        { method: 'GET', redirect: 'follow' }
      );
      await task.close();
    }
  });

  it('detects 5D Conv3d weights and automatically falls back from JSEP runtime kernel errors to WASM', async () => {
    const strBytes = (s: string) => Array.from(s).map((ch) => ch.charCodeAt(0));
    const encodeStrField = (fieldNum: number, s: string) => {
      const b = strBytes(s);
      return [(fieldNum << 3) | 2, b.length, ...b];
    };
    const makeConvProto = (weightDims: number[], nodeName: string): Uint8Array => {
      // TensorProto: field 1 = repeated int64 dims, field 8 = string name 'w'
      const tensorParts: number[] = [];
      for (const d of weightDims) {
        tensorParts.push(0x08, d);
      }
      tensorParts.push(...encodeStrField(8, 'w'));
      // NodeProto: input 'x', input 'w', output 'y', name nodeName, op_type 'Conv'
      const nodeParts = [
        ...encodeStrField(1, 'x'),
        ...encodeStrField(1, 'w'),
        ...encodeStrField(2, 'y'),
        ...encodeStrField(3, nodeName),
        ...encodeStrField(4, 'Conv'),
      ];
      const graphParts = [
        0x0a, nodeParts.length, ...nodeParts, // GraphProto.node (field 1)
        0x2a, tensorParts.length, ...tensorParts, // GraphProto.initializer (field 5)
      ];
      return new Uint8Array([0x08, 0x08, 0x3a, graphParts.length, ...graphParts]);
    };

    const conv2dProto = makeConvProto([16, 3, 3, 3], 'Conv_2d');
    expect(inspectStereoOnnxWebGpuCompatibility(conv2dProto)).toMatchObject({
      compatible: true,
      hasConv3d: false,
    });
    expect(resolveStereoOrtDelegate(STEREO_MODEL_PROFILES['custom-onnx'], conv2dProto)).toBe('WebGPU');

    const conv3dProto = makeConvProto([2, 1, 3, 3, 3], 'Conv_16834');
    expect(inspectStereoOnnxWebGpuCompatibility(conv3dProto)).toMatchObject({
      compatible: false,
      hasConv3d: true,
      conv3dNodeName: 'Conv_16834',
    });
    expect(resolveStereoOrtDelegate(STEREO_MODEL_PROFILES['custom-onnx'], conv3dProto)).toBe('WASM');

    // Decomposing 5D Conv3D nodes rewrites them into WebGPU-compatible 4D Conv2D subgraphs
    // and must keep every synthesized node input resolvable (regression test for Conv_1005__x4d_shape)
    const decomposed = decomposeOnnx5dConvNodesForWebGpu(conv3dProto);
    expect(decomposed.decomposedCount).toBe(1);
    expect(decomposed.decomposedNodeNames).toEqual(['Conv_16834']);
    const decomposedValidation = validateOnnxGraphNodeInputs(decomposed.bytes);
    expect(
      decomposedValidation.missingInputs.filter((m) => m.inputName !== 'x')
    ).toEqual([]);
    expect(inspectStereoOnnxWebGpuCompatibility(decomposed.bytes)).toMatchObject({
      compatible: true,
      hasConv3d: false,
    });
    expect(
      resolveStereoOrtDelegate(STEREO_MODEL_PROFILES['mobilestereonet-2d'], decomposed.bytes)
    ).toBe('WebGPU');

    // Also verify kD = 1 fast-path 5D Conv decomposition
    const conv3dKd1Proto = makeConvProto([4, 2, 1, 3, 3], 'Conv_kd1');
    const decomposedKd1 = decomposeOnnx5dConvNodesForWebGpu(conv3dKd1Proto);
    expect(decomposedKd1.decomposedCount).toBe(1);
    const decomposedKd1Validation = validateOnnxGraphNodeInputs(decomposedKd1.bytes);
    expect(
      decomposedKd1Validation.missingInputs.filter((m) => m.inputName !== 'x')
    ).toEqual([]);
    expect(inspectStereoOnnxWebGpuCompatibility(decomposedKd1.bytes)).toMatchObject({
      compatible: true,
      hasConv3d: false,
    });

    // Worker init-onnx must fall back cleanly to untransformed ONNX bytes on WASM if the
    // decomposed graph fails session creation, and skip 5D decomposition in explicit WASM mode.
    const sessionCalls: Array<{ delegate: 'WebGPU' | 'WASM'; hasConv3d: boolean }> = [];
    const fallbackWorkerProcessor = createStereoWorkerProcessor(async (onnxBytes, delegate) => {
      const hasConv3d = inspectStereoOnnxWebGpuCompatibility(onnxBytes).hasConv3d;
      sessionCalls.push({ delegate, hasConv3d });
      if (delegate === 'WebGPU' && !hasConv3d) {
        throw new Error('Simulated WebGPU session creation failure on decomposed graph');
      }
      return {
        session: {
          inputNames: ['left', 'right'],
          outputNames: ['disp'],
          run: vi.fn(),
          release: vi.fn(async () => {}),
        } as unknown as ort.InferenceSession,
        backend: {
          runtime: 'onnxruntime-web',
          requestedProvider: delegate === 'WebGPU' ? 'webgpu' : 'wasm',
          configuredProviders: [delegate === 'WebGPU' ? 'webgpu' : 'wasm'],
          fallbackReason: null,
          actualBackend: 'unmeasured',
        },
        wasmSourceUsed: 'primary',
      };
    });
    const workerResponses: StereoWorkerResponse[] = [];
    await fallbackWorkerProcessor.handleMessage(
      {
        type: 'init-onnx',
        id: 77,
        profile: STEREO_MODEL_PROFILES['mobilestereonet-2d'],
        onnxBuffer: conv3dProto.slice().buffer,
        wasmPrimaryUrl: 'https://example.invalid/ort/',
        wasmFallbackUrl: null,
        backendPreference: 'auto',
      },
      (res) => workerResponses.push(res)
    );
    expect(sessionCalls).toEqual([
      { delegate: 'WebGPU', hasConv3d: false },
      { delegate: 'WASM', hasConv3d: true },
    ]);
    expect(workerResponses[0]).toMatchObject({
      type: 'init-result',
      id: 77,
      ok: true,
    });

    sessionCalls.length = 0;
    workerResponses.length = 0;
    await fallbackWorkerProcessor.handleMessage(
      {
        type: 'init-onnx',
        id: 78,
        profile: STEREO_MODEL_PROFILES['mobilestereonet-2d'],
        onnxBuffer: conv3dProto.slice().buffer,
        wasmPrimaryUrl: 'https://example.invalid/ort/',
        wasmFallbackUrl: null,
        backendPreference: 'wasm',
      },
      (res) => workerResponses.push(res)
    );
    expect(sessionCalls).toEqual([{ delegate: 'WASM', hasConv3d: true }]);
    expect(workerResponses[0]).toMatchObject({
      type: 'init-result',
      id: 78,
      ok: true,
    });
    await fallbackWorkerProcessor.handleMessage({ type: 'dispose', id: 79 }, () => {});

    expect(isWebGpuRuntimeKernelError(new Error("Non-zero status code returned while running Conv node. Name:'Conv_16834_token_966' Status Message: Failed to run JSEP kernel"))).toBe(true);

    // Also verify runtime JSEP fallback for models that pass static checks (e.g. Custom ONNX on WebGPU)
    const pixelStore = new WeakMap<HTMLCanvasElement, Uint8ClampedArray>();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (this: HTMLCanvasElement) {
      return {
        drawImage: () => {
          pixelStore.set(this, new Uint8ClampedArray(this.width * this.height * 4).fill(110));
        },
        getImageData: (_x: number, _y: number, w: number, h: number) => ({
          data: pixelStore.get(this) ?? new Uint8ClampedArray(w * h * 4).fill(110),
          width: w,
          height: h,
        }),
        createImageData: (w: number, h: number) => ({
          data: new Uint8ClampedArray(w * h * 4),
          width: w,
          height: h,
        }),
        putImageData: () => {},
      } as unknown as CanvasRenderingContext2D;
    });

    const webgpuRun = vi.fn(async () => {
      throw new Error("Non-zero status code returned while running Conv node. Name:'Conv_16834_token_966' Status Message: Failed to run JSEP kernel");
    });
    const webgpuRelease = vi.fn(async () => {});
    const wasmRun = vi.fn(async () => ({
      disp: {
        dims: [1, 1, 16, 32],
        data: new Float32Array(16 * 32).fill(5),
        dispose: vi.fn(),
      } as unknown as ort.Tensor,
    }));
    const wasmRelease = vi.fn(async () => {});

    const webgpuSession = {
      inputNames: ['left', 'right'],
      outputNames: ['disp'],
      inputMetadata: [{ shape: [1, 3, 16, 32] }, { shape: [1, 3, 16, 32] }],
      run: webgpuRun,
      release: webgpuRelease,
    } as unknown as ort.InferenceSession;

    const wasmSession = {
      inputNames: ['left', 'right'],
      outputNames: ['disp'],
      inputMetadata: [{ shape: [1, 3, 16, 32] }, { shape: [1, 3, 16, 32] }],
      run: wasmRun,
      release: wasmRelease,
    } as unknown as ort.InferenceSession;

    const createSpy = vi
      .spyOn(onnxSessionModule, 'createOrtSession')
      .mockResolvedValueOnce({
        session: webgpuSession,
        backend: {
          runtime: 'onnxruntime-web',
          requestedProvider: 'webgpu',
          configuredProviders: ['webgpu', 'wasm'],
          fallbackReason: null,
          actualBackend: 'unmeasured',
        },
      })
      .mockResolvedValueOnce({
        session: wasmSession,
        backend: {
          runtime: 'onnxruntime-web',
          requestedProvider: 'wasm',
          configuredProviders: ['wasm'],
          fallbackReason: null,
          actualBackend: 'unmeasured',
        },
      });

    const task = await createStereoDepthTask({
      variantId: 'custom-onnx',
      modelBytes: conv2dProto,
    });
    const sbsCanvas = Object.assign(document.createElement('canvas'), { width: 128, height: 32 });
    const first = await task.estimateForVideo(sbsCanvas);
    expect(first.medianDisparity).toBeCloseTo(10, 4);
    expect(webgpuRun).toHaveBeenCalledTimes(1);
    expect(webgpuRelease).toHaveBeenCalledTimes(1);
    expect(wasmRun).toHaveBeenCalledTimes(1);
    expect(createSpy).toHaveBeenNthCalledWith(1, conv2dProto, 'WebGPU', expect.any(Object));
    expect(createSpy).toHaveBeenNthCalledWith(2, conv2dProto, 'WASM', expect.any(Object));
    expect(task.backend?.configuredProviders).toEqual(['wasm']);
    expect(task.backend?.fallbackReason).toMatch(/Failed to run JSEP kernel/);

    // Subsequent frames execute directly on recovered WASM session
    await task.estimateForVideo(sbsCanvas);
    expect(webgpuRun).toHaveBeenCalledTimes(1);
    expect(wasmRun).toHaveBeenCalledTimes(2);

    await task.close();
    expect(wasmRelease).toHaveBeenCalledTimes(1);
  });

  it('enables WebGPU for hitnet-sf via trailing-unit-axis 5D ConvTranspose decomposition and falls back safely to unmodified WASM', async () => {
    const encoder = new TextEncoder();
    const encodeVarint = (val: number): number[] => {
      const out: number[] = [];
      let v = val >>> 0;
      while (v >= 0x80) {
        out.push((v & 0x7f) | 0x80);
        v >>>= 7;
      }
      out.push(v);
      return out;
    };
    const lenField = (fn: number, payload: number[]) => [
      (fn << 3) | 2,
      ...encodeVarint(payload.length),
      ...payload,
    ];
    const strField = (fn: number, s: string) => lenField(fn, Array.from(encoder.encode(s)));
    const intAttr = (name: string, vals: number[]) =>
      lenField(5, [
        ...strField(1, name),
        ...vals.flatMap((v) => [(8 << 3) | 0, ...encodeVarint(v)]),
        (20 << 3) | 0,
        7,
      ]);

    const makeConvTransposeProto = (weightDims: number[], strides: number[]): Uint8Array => {
      const tensorParts: number[] = [];
      for (const d of weightDims) {
        tensorParts.push(0x08, d);
      }
      tensorParts.push((2 << 3) | 0, 1); // FLOAT
      tensorParts.push(...strField(8, 'ct_w'));
      const nodeParts = [
        ...strField(1, 'x'),
        ...strField(1, 'ct_w'),
        ...strField(2, 'y'),
        ...strField(3, 'level0/shared/prop0/WarpImageWithHypotheses/get_warp1d_coordinates_from_tile_hypotheses/get_warp_disparity'),
        ...strField(4, 'ConvTranspose'),
        ...intAttr('kernel_shape', weightDims.slice(2)),
        ...intAttr('strides', strides),
        ...intAttr('pads', [0, 0, 0, 0, 0, 0]),
      ];
      const graphParts = [
        ...lenField(1, nodeParts),
        ...lenField(5, tensorParts),
        ...strField(2, 'hitnet_ct_graph'),
      ];
      const opsetParts = [...strField(1, ''), (2 << 3) | 0, 12];
      return new Uint8Array([
        0x08,
        0x07,
        ...lenField(7, graphParts),
        ...lenField(8, opsetParts),
      ]);
    };

    const compatibleCtProto = makeConvTransposeProto([3, 1, 4, 4, 1], [4, 4, 1]);
    expect(inspectStereoOnnxWebGpuCompatibility(compatibleCtProto)).toMatchObject({
      compatible: false,
      hasConv3d: true,
    });
    expect(resolveStereoOrtDelegate(STEREO_MODEL_PROFILES['hitnet-sf'], compatibleCtProto)).toBe('WASM');

    const decomposed = decomposeOnnx5dConvNodesForWebGpu(compatibleCtProto);
    expect(decomposed.decomposedCount).toBe(1);
    expect(inspectStereoOnnxWebGpuCompatibility(decomposed.bytes)).toMatchObject({
      compatible: true,
      hasConv3d: false,
    });
    expect(resolveStereoOrtDelegate(STEREO_MODEL_PROFILES['hitnet-sf'], decomposed.bytes)).toBe('WebGPU');

    const incompatibleCtProto = makeConvTransposeProto([3, 1, 4, 4, 2], [4, 4, 1]);
    const untouched = decomposeOnnx5dConvNodesForWebGpu(incompatibleCtProto);
    expect(untouched.decomposedCount).toBe(0);
    expect(untouched.bytes).toBe(incompatibleCtProto);
    expect(resolveStereoOrtDelegate(STEREO_MODEL_PROFILES['hitnet-sf'], untouched.bytes)).toBe('WASM');

    // Runtime JSEP failure on decomposed graph must fall back cleanly to original unmodified bytes on WASM
    const sessionCalls: Array<{ delegate: 'WebGPU' | 'WASM'; hasConv3d: boolean }> = [];
    const webgpuRun = vi.fn(async () => {
      throw new Error('Failed to run JSEP kernel on ConvTranspose');
    });
    const wasmRun = vi.fn(async () => ({
      reference_output_disparity: {
        dims: [1, 240, 320, 1],
        data: new Float32Array(240 * 320).fill(3.5),
        dispose: vi.fn(),
      } as unknown as ort.Tensor,
    }));
    const workerProcessor = createStereoWorkerProcessor(async (onnxBytes, delegate) => {
      const hasConv3d = inspectStereoOnnxWebGpuCompatibility(onnxBytes).hasConv3d;
      sessionCalls.push({ delegate, hasConv3d });
      return {
        session: {
          inputNames: ['input'],
          outputNames: ['reference_output_disparity'],
          inputMetadata: [{ shape: [1, 6, 240, 320] }],
          run: delegate === 'WebGPU' ? webgpuRun : wasmRun,
          release: vi.fn(async () => {}),
        } as unknown as ort.InferenceSession,
        backend: {
          runtime: 'onnxruntime-web',
          requestedProvider: delegate === 'WebGPU' ? 'webgpu' : 'wasm',
          configuredProviders: [delegate === 'WebGPU' ? 'webgpu' : 'wasm'],
          fallbackReason: null,
          actualBackend: 'unmeasured',
        },
        wasmSourceUsed: 'primary',
      };
    });

    const responses: StereoWorkerResponse[] = [];
    await workerProcessor.handleMessage(
      {
        type: 'init-onnx',
        id: 1,
        profile: STEREO_MODEL_PROFILES['hitnet-sf'],
        onnxBuffer: compatibleCtProto.slice().buffer,
        wasmPrimaryUrl: 'https://example.invalid/ort/',
        wasmFallbackUrl: null,
        backendPreference: 'auto',
      },
      (res) => responses.push(res)
    );
    expect(sessionCalls).toEqual([{ delegate: 'WebGPU', hasConv3d: false }]);
    expect(responses[0]).toMatchObject({
      type: 'init-result',
      id: 1,
      ok: true,
      backend: { configuredProviders: ['webgpu'] },
    });

    const rgbaBytes = 320 * 240 * 4;
    await workerProcessor.handleMessage(
      {
        type: 'estimate',
        id: 2,
        width: 320,
        height: 240,
        sourceEyeWidth: 480,
        sourceEyeHeight: 270,
        leftRgbaBuffer: new Uint8ClampedArray(rgbaBytes).fill(100).buffer,
        rightRgbaBuffer: new Uint8ClampedArray(rgbaBytes).fill(100).buffer,
        options: {},
      },
      (res) => responses.push(res)
    );

    expect(sessionCalls).toEqual([
      { delegate: 'WebGPU', hasConv3d: false },
      { delegate: 'WASM', hasConv3d: true },
    ]);
    expect(responses[1]).toMatchObject({
      type: 'estimate-result',
      id: 2,
      ok: true,
      backend: { configuredProviders: ['wasm'] },
    });

    // Explicit WASM mode must skip 5D ConvTranspose conversion altogether
    sessionCalls.length = 0;
    await workerProcessor.handleMessage(
      {
        type: 'init-onnx',
        id: 3,
        profile: STEREO_MODEL_PROFILES['hitnet-sf'],
        onnxBuffer: compatibleCtProto.slice().buffer,
        wasmPrimaryUrl: 'https://example.invalid/ort/',
        wasmFallbackUrl: null,
        backendPreference: 'wasm',
      },
      (res) => responses.push(res)
    );
    expect(sessionCalls).toEqual([{ delegate: 'WASM', hasConv3d: true }]);
    await workerProcessor.handleMessage({ type: 'dispose', id: 4 }, () => {});
  });

  it('runs StereoOnnxEngine with dual/stacked/4-input ONNX sessions and disposes all tensors cleanly', async () => {
    const pixelStore = new WeakMap<HTMLCanvasElement, Uint8ClampedArray>();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (this: HTMLCanvasElement) {
      return {
        drawImage: () => {
          pixelStore.set(this, new Uint8ClampedArray(this.width * this.height * 4).fill(128));
        },
        getImageData: (_x: number, _y: number, w: number, h: number) => ({
          data: pixelStore.get(this) ?? new Uint8ClampedArray(w * h * 4).fill(128),
          width: w,
          height: h,
        }),
        createImageData: (w: number, h: number) => ({
          data: new Uint8ClampedArray(w * h * 4),
          width: w,
          height: h,
        }),
        putImageData: () => {},
      } as unknown as CanvasRenderingContext2D;
    });

    const release = vi.fn(async () => {});
    const run = vi.fn(async (feeds: Record<string, ort.Tensor>) => {
      expect(Object.keys(feeds)).toEqual(['left', 'right']);
      expect(feeds.left.dims).toEqual([1, 3, 16, 32]);
      expect(feeds.right.dims).toEqual([1, 3, 16, 32]);
      return {
        disparity: {
          dims: [1, 1, 16, 32],
          data: new Float32Array(16 * 32).fill(8),
          dispose: vi.fn(),
        } as unknown as ort.Tensor,
      };
    });

    const mockSession = {
      inputNames: ['left', 'right'],
      outputNames: ['disparity'],
      inputMetadata: [{ dimensions: [1, 3, 16, 32] }, { dimensions: [1, 3, 16, 32] }],
      run,
      release,
    } as unknown as ort.InferenceSession;

    const engine = new StereoOnnxEngine(mockSession, STEREO_MODEL_PROFILES['lightstereo-s']);
    const sbsCanvas = Object.assign(document.createElement('canvas'), { width: 128, height: 32 });

    const depthMap = await engine.estimateForVideo(sbsCanvas, { colormap: 'viridis' });
    // Source eye width = 64, model width = 32 => scale = 2.0, 8 * 2 = 16px
    expect(depthMap.sourceEyeWidth).toBe(64);
    expect(depthMap.sourceEyeHeight).toBe(32);
    expect(depthMap.medianDisparity).toBeCloseTo(16, 4);
    expect(depthMap.colormap).toBe('viridis');

    await engine.close();
    expect(release).toHaveBeenCalledTimes(1);
  });

  it('resolves static resolution from onnxruntime-web .shape using the last input for 4-input CREStereo', async () => {
    const pixelStore = new WeakMap<HTMLCanvasElement, Uint8ClampedArray>();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (this: HTMLCanvasElement) {
      return {
        drawImage: () => {
          pixelStore.set(this, new Uint8ClampedArray(this.width * this.height * 4).fill(100));
        },
        getImageData: (_x: number, _y: number, w: number, h: number) => ({
          data: pixelStore.get(this) ?? new Uint8ClampedArray(w * h * 4).fill(100),
          width: w,
          height: h,
        }),
        createImageData: (w: number, h: number) => ({
          data: new Uint8ClampedArray(w * h * 4),
          width: w,
          height: h,
        }),
        putImageData: () => {},
      } as unknown as CanvasRenderingContext2D;
    });

    const run = vi.fn(async (feeds: Record<string, ort.Tensor>) => {
      expect(feeds.init_left.dims).toEqual([1, 3, 24, 32]);
      expect(feeds.init_right.dims).toEqual([1, 3, 24, 32]);
      expect(feeds.next_left.dims).toEqual([1, 3, 48, 64]);
      expect(feeds.next_right.dims).toEqual([1, 3, 48, 64]);
      return {
        next_output: {
          dims: [1, 2, 48, 64],
          data: new Float32Array(2 * 48 * 64).fill(4),
          dispose: vi.fn(),
        } as unknown as ort.Tensor,
      };
    });

    const mockSession = {
      inputNames: ['init_left', 'init_right', 'next_left', 'next_right'],
      outputNames: ['next_output'],
      inputMetadata: [
        { shape: [1, 3, 24, 32] },
        { shape: [1, 3, 24, 32] },
        { shape: [1, 3, 48, 64] },
        { shape: [1, 3, 48, 64] },
      ],
      run,
      release: vi.fn(async () => {}),
    } as unknown as ort.InferenceSession;

    const engine = new StereoOnnxEngine(mockSession, STEREO_MODEL_PROFILES['crestereo-combined-iter5']);
    const sbsCanvas = Object.assign(document.createElement('canvas'), { width: 256, height: 96 });
    const depthMap = await engine.estimateForVideo(sbsCanvas);
    expect(depthMap.width).toBe(64);
    expect(depthMap.height).toBe(48);
    expect(depthMap.medianDisparity).toBeCloseTo(8, 4);
    await engine.close();
  });

  it('guards VisionEngine stereo depth inspection against stale generations', async () => {
    const dummyCanvas = document.createElement('canvas');
    const mockDepthMap: StereoDepthMapData = {
      width: 10,
      height: 10,
      disparity: new Float32Array(100).fill(20),
      minDisparity: 20,
      maxDisparity: 20,
      medianDisparity: 20,
      scaleMinDisparity: 20,
      scaleMaxDisparity: 20,
      validRatio: 1,
      overlayCanvas: dummyCanvas,
      leftCanvas: dummyCanvas,
      rightCanvas: dummyCanvas,
      colormap: 'turbo',
      viewMode: 'left-overlay',
      swapLeftRight: false,
      sourceEyeWidth: 320,
      sourceEyeHeight: 240,
    };

    const stereoTask = {
      backend: {
        runtime: 'stereo-sad-cpu' as const,
        requestedProvider: 'cpu' as const,
        configuredProviders: ['cpu-sad'],
        fallbackReason: null,
        actualBackend: 'unmeasured' as const,
      },
      estimateForVideo: vi.fn(() => mockDepthMap),
      close: vi.fn(),
    };

    const factory: VisionTaskFactory = {
      load: vi.fn(async () => ({ wasmLoaderPath: '', wasmBinaryPath: '' })),
      detector: vi.fn(),
      segmenter: vi.fn(),
      stereoDepth: vi.fn(async () => stereoTask),
    };

    const engine = new VisionEngine(factory);
    const init = await engine.initialize('stereo-depth', 'stereo-sad-baseline');
    expect(init.status).toBe('ready');

    const input = Object.assign(document.createElement('canvas'), { width: 640, height: 240 });
    const frame = await engine.detectVideoFrame(input, 'stereo-depth', 0.35);
    expect(frame.status).toBe('ready');
    if (frame.status === 'ready' && frame.depthMap) {
      const hit = engine.inspectDepthAtPoint(0.5, 0.5, frame.depthMap, 60, 500);
      expect(hit).toMatchObject({ valid: true, disparityPx: 20 });
      expect(hit?.depthMeters).toBeCloseTo(1.5, 4);

      engine.clearDetections();
      expect(engine.inspectDepthAtPoint(0.5, 0.5, frame.depthMap, 60, 500)).toBeNull();
    }
    engine.dispose();
    expect(stereoTask.close).toHaveBeenCalledTimes(1);
  });

  it('executes SAD and ONNX stereo depth inside StereoWorkerEngine via createStereoWorkerProcessor', async () => {
    const pixelStore = new WeakMap<HTMLCanvasElement, Uint8ClampedArray>();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (this: HTMLCanvasElement) {
      return {
        drawImage: () => {
          pixelStore.set(this, makeTexturedRgba(this.width || 64, this.height || 32, () => 4));
        },
        getImageData: (_x: number, _y: number, w: number, h: number) => ({
          data: pixelStore.get(this) ?? makeTexturedRgba(w, h, () => 4),
          width: w,
          height: h,
        }),
        createImageData: (w: number, h: number) => ({
          data: new Uint8ClampedArray(w * h * 4),
          width: w,
          height: h,
        }),
        putImageData: () => {},
      } as unknown as CanvasRenderingContext2D;
    });

    const createPairedWorker = (
      sessionCreator?: Parameters<typeof createStereoWorkerProcessor>[0]
    ): { worker: StereoWorkerLike; terminated: () => boolean } => {
      let isTerminated = false;
      const processor = createStereoWorkerProcessor(sessionCreator);
      const worker: StereoWorkerLike = {
        onmessage: null,
        onerror: null,
        postMessage: (msg: StereoWorkerRequest) => {
          if (isTerminated) return;
          void processor
            .handleMessage(msg, (response) => {
              if (!isTerminated) {
                worker.onmessage?.({ data: response } as MessageEvent<StereoWorkerResponse>);
              }
            })
            .catch((err) => {
              if (!isTerminated) {
                worker.onerror?.({ message: String(err) } as ErrorEvent);
              }
            });
        },
        terminate: () => {
          isTerminated = true;
          void processor.handleMessage({ type: 'dispose', id: -1 }, () => {});
        },
      };
      return { worker, terminated: () => isTerminated };
    };

    // 1. SAD Baseline in Worker
    const sadPair = createPairedWorker();
    const sadTask = await createStereoDepthTask({
      variantId: 'stereo-sad-baseline',
      workerFactory: () => sadPair.worker,
    });
    const sbsCanvas = Object.assign(document.createElement('canvas'), { width: 128, height: 64 });
    const sadDepth = await sadTask.estimateForVideo(sbsCanvas, { colormap: 'inferno' });
    expect(sadDepth.width).toBe(64);
    expect(sadDepth.height).toBe(64);
    expect(sadDepth.colormap).toBe('inferno');
    expect(sadPair.terminated()).toBe(false);
    await sadTask.close();
    expect(sadPair.terminated()).toBe(true);

    // 2. ONNX variant in Worker
    const mockSession = {
      inputNames: ['left', 'right'],
      outputNames: ['disp'],
      inputMetadata: [{ shape: [1, 3, 24, 32] }, { shape: [1, 3, 24, 32] }],
      run: vi.fn(async () => ({
        disp: {
          dims: [1, 1, 24, 32],
          data: new Float32Array(24 * 32).fill(6),
          dispose: vi.fn(),
        } as unknown as ort.Tensor,
      })),
      release: vi.fn(async () => {}),
    } as unknown as ort.InferenceSession;

    const onnxPair = createPairedWorker(async () => ({
      session: mockSession,
      backend: {
        runtime: 'onnxruntime-web',
        requestedProvider: 'wasm',
        configuredProviders: ['wasm'],
        fallbackReason: null,
        actualBackend: 'unmeasured',
      },
      wasmSourceUsed: 'primary',
    }));
    const onnxTask = await createStereoDepthTask({
      variantId: 'lightstereo-s',
      modelBytes: new Uint8Array(64).fill(1),
      workerFactory: () => onnxPair.worker,
    });
    const onnxDepth = await onnxTask.estimateForVideo(sbsCanvas, { colormap: 'turbo' });
    // Source eye width = 64, model output width = 32 => scale = 2.0 => 6 * 2 = 12
    expect(onnxDepth.width).toBe(32);
    expect(onnxDepth.height).toBe(24);
    expect(onnxDepth.medianDisparity).toBeCloseTo(12, 4);
    await onnxTask.close();
    expect(onnxPair.terminated()).toBe(true);
  });

  it('immediately terminates an in-flight Worker and rejects pending frame promises when switching variants', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (this: HTMLCanvasElement) {
      return {
        drawImage: () => {},
        getImageData: (_x: number, _y: number, w: number, h: number) => ({
          data: new Uint8ClampedArray(w * h * 4).fill(128),
          width: w,
          height: h,
        }),
        createImageData: (w: number, h: number) => ({
          data: new Uint8ClampedArray(w * h * 4),
          width: w,
          height: h,
        }),
        putImageData: () => {},
      } as unknown as CanvasRenderingContext2D;
    });

    const terminateSpy = vi.fn();
    const stalledWorker: StereoWorkerLike = {
      onmessage: null,
      onerror: null,
      postMessage: (msg: StereoWorkerRequest) => {
        if (msg.type === 'init-sad') {
          stalledWorker.onmessage?.({
            data: {
              type: 'init-result',
              id: msg.id,
              ok: true,
              backend: {
                runtime: 'stereo-sad-cpu',
                requestedProvider: 'cpu',
                configuredProviders: ['cpu-sad'],
                fallbackReason: null,
                actualBackend: 'unmeasured',
              },
              wasmSourceUsed: 'primary',
              targetWidth: 64,
              targetHeight: 32,
            },
          } as unknown as MessageEvent<StereoWorkerResponse>);
        }
        // Intentionally never respond to 'estimate' to simulate a slow 1,500ms WASM frame
      },
      terminate: terminateSpy,
    };

    const task = await createStereoDepthTask({
      variantId: 'stereo-sad-baseline',
      workerFactory: () => stalledWorker,
    });
    const sbsCanvas = Object.assign(document.createElement('canvas'), { width: 128, height: 64 });
    const pendingEstimate = Promise.resolve(task.estimateForVideo(sbsCanvas));

    // Closing the task (as VisionEngine.initialize does on variant change) must immediately
    // terminate the Worker and reject the in-flight frame promise so UI/locks never wait.
    await task.close();
    expect(terminateSpy).toHaveBeenCalledTimes(1);
    await expect(pendingEstimate).rejects.toThrow(/task is closed/);

    cancelInFlightStereoInit();
  });

  it('awaits resolveOrtWasmUrls in StereoWorkerEngine.create and records wasm-ort asset source', async () => {
    const sentMessages: StereoWorkerRequest[] = [];
    const recordedSources: Array<[string, string]> = [];
    const worker: StereoWorkerLike = {
      onmessage: null,
      onerror: null,
      postMessage: (msg: StereoWorkerRequest) => {
        sentMessages.push(msg);
        if (msg.type === 'init-onnx') {
          worker.onmessage?.({
            data: {
              type: 'init-result',
              id: msg.id,
              ok: true,
              backend: {
                runtime: 'onnxruntime-web',
                requestedProvider: 'wasm',
                configuredProviders: ['wasm'],
                fallbackReason: null,
                actualBackend: 'unmeasured',
              },
              wasmSourceUsed: 'primary',
              targetWidth: 320,
              targetHeight: 240,
            },
          } as unknown as MessageEvent<StereoWorkerResponse>);
        }
      },
      terminate: vi.fn(),
    };

    const task = await createStereoDepthTask({
      variantId: 'mobilestereonet-2d',
      modelBytes: new Uint8Array(2048).fill(1),
      workerFactory: () => worker,
      recordAssetSource: (assetKey, source) => {
        recordedSources.push([assetKey, source]);
      },
    });

    const initMsg = sentMessages.find(
      (m): m is Extract<StereoWorkerRequest, { type: 'init-onnx' }> => m.type === 'init-onnx'
    );
    expect(initMsg).toBeDefined();
    expect(typeof initMsg?.wasmPrimaryUrl).toBe('string');
    expect((initMsg?.wasmPrimaryUrl ?? '').length).toBeGreaterThan(0);
    expect(recordedSources.some(([key]) => key === 'wasm-ort')).toBe(true);

    await task.close();
  });

  it('normalizes predominantly negative 1-channel disparity tensors and handles dual-camera exposure offsets in SAD', () => {
    const negSingleChannel = extractDisparityFromOrtTensor({
      dims: [1, 1, 2, 2],
      data: new Float32Array([-4.5, -9.0, -12.0, -2.0]),
    });
    expect(Array.from(negSingleChannel.disparity)).toEqual([4.5, 9.0, 12.0, 2.0]);

    const w = 96;
    const h = 48;
    const left = makeTexturedRgba(w, h, () => 0);
    const right = makeTexturedRgba(w, h, (x, y) => (x >= 28 && x < 68 && y >= 12 && y < 36 ? 10 : 2));
    // Simulate a +24 sensor gain/exposure brightness offset in the right eye
    for (let i = 0; i < right.length; i += 4) {
      right[i] = Math.min(255, right[i] + 24);
      right[i + 1] = Math.min(255, right[i + 1] + 24);
      right[i + 2] = Math.min(255, right[i + 2] + 24);
    }

    const sad = computeSadDisparity(left, right, w, h, {
      maxDisparity: 24,
      windowSize: 7,
      confidenceThreshold: 0.35,
    });
    expect(sad.validRatio).toBeGreaterThan(0.7);
    const centerDisp = sad.disparity[24 * w + 48];
    expect(centerDisp).toBeGreaterThanOrEqual(8.0);
    expect(centerDisp).toBeLessThanOrEqual(12.0);
  });

  it('disables ORT graph constant folding and memory arenas on stereo WASM sessions to prevent multi-GB OOM crashes', () => {
    expect(resolveStereoOrtSessionOptions(['wasm'])).toEqual({
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      enableCpuMemArena: false,
      enableMemPattern: false,
      logSeverityLevel: 3,
    });
    expect(resolveStereoOrtSessionOptions(['webgpu', 'wasm'])).toEqual({
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: true,
      enableMemPattern: true,
      logSeverityLevel: 3,
    });
  });

  it('coalesces intermediate queued frames in StereoWorkerEngine while a slow ONNX frame is in-flight', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (
      this: HTMLCanvasElement
    ) {
      return {
        drawImage: () => {},
        getImageData: (_x: number, _y: number, w: number, h: number) => ({
          data: new Uint8ClampedArray(w * h * 4).fill(100),
          width: w,
          height: h,
        }),
        createImageData: (w: number, h: number) => ({
          data: new Uint8ClampedArray(w * h * 4),
          width: w,
          height: h,
        }),
        putImageData: () => {},
      } as unknown as CanvasRenderingContext2D;
    });

    const estimateRequests: Array<Extract<StereoWorkerRequest, { type: 'estimate' }>> = [];
    const pendingResponders: Array<() => void> = [];

    const controlledWorker: StereoWorkerLike = {
      onmessage: null,
      onerror: null,
      postMessage: (msg: StereoWorkerRequest) => {
        if (msg.type === 'init-onnx') {
          controlledWorker.onmessage?.({
            data: {
              type: 'init-result',
              id: msg.id,
              ok: true,
              backend: {
                runtime: 'onnxruntime-web',
                requestedProvider: 'wasm',
                configuredProviders: ['wasm'],
                fallbackReason: null,
                actualBackend: 'unmeasured',
              },
              wasmSourceUsed: 'primary',
              targetWidth: 32,
              targetHeight: 24,
            },
          } as unknown as MessageEvent<StereoWorkerResponse>);
          return;
        }
        if (msg.type === 'estimate') {
          estimateRequests.push(msg);
          pendingResponders.push(() => {
            const disp = new Float32Array(msg.width * msg.height).fill(10);
            const rgba = new Uint8ClampedArray(msg.width * msg.height * 4).fill(200);
            controlledWorker.onmessage?.({
              data: {
                type: 'estimate-result',
                id: msg.id,
                ok: true,
                width: msg.width,
                height: msg.height,
                disparityBuffer: disp.buffer,
                overlayRgbaBuffer: rgba.buffer,
                stats: {
                  minDisparity: 10,
                  maxDisparity: 10,
                  medianDisparity: 10,
                  scaleMinDisparity: 10,
                  scaleMaxDisparity: 10,
                  validRatio: 1,
                },
                backend: {
                  runtime: 'onnxruntime-web',
                  requestedProvider: 'wasm',
                  configuredProviders: ['wasm'],
                  fallbackReason: null,
                  actualBackend: 'unmeasured',
                },
              },
            } as unknown as MessageEvent<StereoWorkerResponse>);
          });
        }
      },
      terminate: vi.fn(),
    };

    const task = await createStereoDepthTask({
      variantId: 'hitnet-sf',
      modelBytes: new Uint8Array(512).fill(1),
      workerFactory: () => controlledWorker,
    });

    const sbsCanvas = Object.assign(document.createElement('canvas'), { width: 128, height: 64 });

    // Start frame 1 (immediately sent to Worker)
    const p1 = task.estimateForVideo(sbsCanvas, { colormap: 'turbo' });
    // Queue frames 2, 3, and 4 while frame 1 is still in-flight in the Worker
    const p2 = task.estimateForVideo(sbsCanvas, { colormap: 'viridis' });
    const p3 = task.estimateForVideo(sbsCanvas, { colormap: 'inferno' });
    const p4 = task.estimateForVideo(sbsCanvas, { colormap: 'grayscale' });

    // Only frame 1 should have been posted to the Worker so far
    expect(estimateRequests).toHaveLength(1);
    expect(estimateRequests[0].options.colormap).toBe('turbo');

    // Complete frame 1 -> should automatically dispatch ONLY the coalesced latest frame (frame 4: grayscale)
    pendingResponders.shift()?.();
    const res1 = await p1;
    expect(res1.colormap).toBe('turbo');

    expect(estimateRequests).toHaveLength(2);
    expect(estimateRequests[1].options.colormap).toBe('grayscale');

    // Complete coalesced frame 4 -> resolves p2, p3, and p4 with the latest result
    pendingResponders.shift()?.();
    const [res2, res3, res4] = await Promise.all([p2, p3, p4]);
    expect(res2.colormap).toBe('grayscale');
    expect(res3.colormap).toBe('grayscale');
    expect(res4.colormap).toBe('grayscale');

    await task.close();
  });

  it('respects explicit backendPreference (auto / webgpu / wasm) in resolveStereoOrtDelegate and createStereoDepthTask', async () => {
    const tinyOnnx = new Uint8Array([1, 2, 3, 4]);
    expect(resolveStereoOrtDelegate(STEREO_MODEL_PROFILES['lightstereo-s'], tinyOnnx, 8, 'auto')).toBe('WebGPU');
    expect(resolveStereoOrtDelegate(STEREO_MODEL_PROFILES['lightstereo-s'], tinyOnnx, 8, 'wasm')).toBe('WASM');
    expect(resolveStereoOrtDelegate(STEREO_MODEL_PROFILES['mobilestereonet-2d'], tinyOnnx, 8, 'auto')).toBe('WebGPU');
    expect(resolveStereoOrtDelegate(STEREO_MODEL_PROFILES['mobilestereonet-2d'], tinyOnnx, 8, 'wasm')).toBe('WASM');
    expect(resolveStereoOrtDelegate(STEREO_MODEL_PROFILES['mobilestereonet-2d'], tinyOnnx, 8, 'webgpu')).toBe('WebGPU');

    const mockSession = {
      inputNames: ['left', 'right'],
      outputNames: ['disp'],
      run: vi.fn(),
      release: vi.fn(async () => {}),
    } as unknown as ort.InferenceSession;

    const createSpy = vi.spyOn(onnxSessionModule, 'createOrtSession').mockResolvedValue({
      session: mockSession,
      backend: {
        runtime: 'onnxruntime-web',
        requestedProvider: 'wasm',
        configuredProviders: ['wasm'],
        fallbackReason: null,
        actualBackend: 'unmeasured',
      },
    });

    const task = await createStereoDepthTask({
      variantId: 'custom-onnx',
      modelBytes: new Uint8Array(64).fill(7),
      backendPreference: 'wasm',
    });
    expect(createSpy).toHaveBeenCalledTimes(1);
    expect(createSpy.mock.calls[0][1]).toBe('WASM');
    await task.close();
  });

  it('runs stereo-webgpu-wgsl via WebGPU compute when available and falls back to CPU SAD when unavailable or forced WASM', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (this: HTMLCanvasElement) {
      return {
        drawImage: () => {},
        getImageData: (_x: number, _y: number, w: number, h: number) => ({
          data: makeTexturedRgba(w, h, (x) => (x < w / 2 ? 0 : 8)),
          width: w,
          height: h,
        }),
        createImageData: (w: number, h: number) => ({
          data: new Uint8ClampedArray(w * h * 4),
          width: w,
          height: h,
        }),
        putImageData: () => {},
      } as unknown as CanvasRenderingContext2D;
    });

    const createPairedWorker = (): StereoWorkerLike => {
      let terminated = false;
      const processor = createStereoWorkerProcessor();
      const worker: StereoWorkerLike = {
        onmessage: null,
        onerror: null,
        postMessage: (msg: StereoWorkerRequest) => {
          if (terminated) return;
          void processor.handleMessage(msg, (response) => {
            if (!terminated) {
              worker.onmessage?.({ data: response } as MessageEvent<StereoWorkerResponse>);
            }
          });
        },
        terminate: () => {
          terminated = true;
          void processor.handleMessage({ type: 'dispose', id: -1 }, () => {});
        },
      };
      return worker;
    };

    // 1. Without navigator.gpu, stereo-webgpu-wgsl automatically falls back to CPU SAD without crashing
    const origGpu = (navigator as unknown as { gpu?: unknown }).gpu;
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: undefined });
    try {
      const fallbackTask = await createStereoDepthTask({
        variantId: 'stereo-webgpu-wgsl',
        workerFactory: createPairedWorker,
      });
      expect(fallbackTask.backend?.runtime).toBe('stereo-webgpu-wgsl');
      expect(fallbackTask.backend?.configuredProviders).toEqual(['cpu-sad']);
      expect(fallbackTask.backend?.fallbackReason).toBeTruthy();

      const sbsCanvas = Object.assign(document.createElement('canvas'), { width: 128, height: 64 });
      const fallbackResult = await fallbackTask.estimateForVideo(sbsCanvas);
      expect(fallbackResult.width).toBe(64);
      expect(fallbackResult.height).toBe(64);
      await fallbackTask.close();

      // 2. With mocked WebGPU device, createWebGpuStereoComputeEngine executes the 4-pass compute pipeline
      const createdBuffers: Array<{ size: number; usage: number; data: ArrayBuffer }> = [];
      let readbackPixelCount = 0;
      const mockDevice = {
        createShaderModule: vi.fn(() => ({})),
        createComputePipelineAsync: vi.fn(async () => ({
          getBindGroupLayout: vi.fn(() => ({})),
        })),
        createBindGroup: vi.fn(() => ({})),
        createBuffer: vi.fn(({ size, usage }: { size: number; usage: number }) => {
          const data = new ArrayBuffer(size);
          if (usage & 0x0001) {
            // MAP_READ buffer (readback): populate simulated GPU disparity output (14.5px)
            readbackPixelCount = Math.floor(size / 4);
            new Float32Array(data).fill(14.5);
          }
          const buf = {
            size,
            usage,
            data,
            mapAsync: vi.fn(async () => {}),
            getMappedRange: vi.fn(() => {
              new Float32Array(data).fill(14.5);
              return data;
            }),
            unmap: vi.fn(),
            destroy: vi.fn(),
          };
          createdBuffers.push(buf);
          return buf;
        }),
        createCommandEncoder: vi.fn(() => ({
          beginComputePass: vi.fn(() => ({
            setPipeline: vi.fn(),
            setBindGroup: vi.fn(),
            dispatchWorkgroups: vi.fn(),
            end: vi.fn(),
          })),
          copyBufferToBuffer: vi.fn(),
          finish: vi.fn(() => ({})),
        })),
        queue: {
          writeBuffer: vi.fn(),
          submit: vi.fn(),
        },
        destroy: vi.fn(),
      };
      const requestAdapterSpy = vi.fn(async () => ({
        requestDevice: vi.fn(async () => mockDevice),
      }));
      Object.defineProperty(navigator, 'gpu', {
        configurable: true,
        value: { requestAdapter: requestAdapterSpy },
      });

      const directEngine = await createWebGpuStereoComputeEngine();
      expect(directEngine).not.toBeNull();
      const leftRgba = new Uint8ClampedArray(32 * 16 * 4).fill(120);
      const rightRgba = new Uint8ClampedArray(32 * 16 * 4).fill(120);
      const gpuOut = await directEngine!.estimate(leftRgba, rightRgba, 32, 16, {
        maxDisparity: 48,
        windowSize: 5,
        confidenceThreshold: 0.35,
      });
      expect(readbackPixelCount).toBe(32 * 16);
      expect(gpuOut.width).toBe(32);
      expect(gpuOut.height).toBe(16);
      expect(gpuOut.disparity[0]).toBeCloseTo(14.5, 4);
      expect(mockDevice.queue.submit).toHaveBeenCalledTimes(1);
      directEngine!.destroy();
      expect(mockDevice.destroy).toHaveBeenCalledTimes(1);

      // Inside Worker with GPU available
      const gpuTask = await createStereoDepthTask({
        variantId: 'stereo-webgpu-wgsl',
        workerFactory: createPairedWorker,
      });
      expect(gpuTask.backend?.runtime).toBe('stereo-webgpu-wgsl');
      expect(gpuTask.backend?.configuredProviders).toEqual(['webgpu-wgsl']);
      const gpuFrame = await gpuTask.estimateForVideo(sbsCanvas);
      expect(gpuFrame.medianDisparity).toBeCloseTo(14.5, 4);
      await gpuTask.close();

      // Forcing WASM/CPU overrides WebGPU even when navigator.gpu exists
      requestAdapterSpy.mockClear();
      const forcedCpuTask = await createStereoDepthTask({
        variantId: 'stereo-webgpu-wgsl',
        backendPreference: 'wasm',
        workerFactory: createPairedWorker,
      });
      expect(requestAdapterSpy).not.toHaveBeenCalled();
      expect(forcedCpuTask.backend?.configuredProviders).toEqual(['cpu-sad']);
      await forcedCpuTask.close();
    } finally {
      Object.defineProperty(navigator, 'gpu', { configurable: true, value: origGpu });
    }
  });

  it('rotates the entire Side-by-Side frame by 180deg (swapping raw halves and rotating each eye) when rotate180 is enabled', () => {
    // 4x2 raw SBS input: leftHalf is 2x2 [pixels [10,20] / [30,40]], rightHalf is 2x2 [pixels [50,60] / [70,80]]
    // Physically inverted SBS capture is rot180([L | R]) = [rot180(R) | rot180(L)],
    // so rotating back by 180deg yields upright Left = rot180(rightHalf) = [[80,70],[60,50]]
    // and upright Right = rot180(leftHalf) = [[40,30],[20,10]].
    const sbsW = 4;
    const sbsH = 2;
    const rawPixels = [
      // y = 0: leftHalf(0..1), rightHalf(2..3)
      10, 20, 50, 60,
      // y = 1:
      30, 40, 70, 80,
    ];
    const canvasPixels = new WeakMap<HTMLCanvasElement, Uint8ClampedArray>();
    const sbsCanvas = Object.assign(document.createElement('canvas'), { width: sbsW, height: sbsH });
    const rawRgba = new Uint8ClampedArray(sbsW * sbsH * 4);
    rawPixels.forEach((v, idx) => {
      rawRgba[idx * 4] = v;
      rawRgba[idx * 4 + 1] = v;
      rawRgba[idx * 4 + 2] = v;
      rawRgba[idx * 4 + 3] = 255;
    });
    canvasPixels.set(sbsCanvas, rawRgba);

    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (this: HTMLCanvasElement) {
      let tx = 0;
      let ty = 0;
      let angle = 0;
      const stack: Array<{ tx: number; ty: number; angle: number }> = [];
      return {
        save: () => {
          stack.push({ tx, ty, angle });
        },
        restore: () => {
          const prev = stack.pop();
          if (prev) {
            tx = prev.tx;
            ty = prev.ty;
            angle = prev.angle;
          }
        },
        translate: (x: number, y: number) => {
          tx += x;
          ty += y;
        },
        rotate: (rad: number) => {
          angle += rad;
        },
        drawImage: (
          img: HTMLCanvasElement,
          sx: number,
          sy: number,
          sw: number,
          sh: number,
          dx: number,
          dy: number,
          dw: number,
          dh: number
        ) => {
          const src = canvasPixels.get(img);
          if (!src) return;
          const outW = this.width;
          const outH = this.height;
          const dst = new Uint8ClampedArray(outW * outH * 4);
          const isRotated180 = Math.abs(Math.abs(angle) - Math.PI) < 1e-4;
          for (let oy = 0; oy < dh; oy++) {
            for (let ox = 0; ox < dw; ox++) {
              const rx = isRotated180 ? Math.round(tx - 1 - (dx + ox)) : dx + ox;
              const ry = isRotated180 ? Math.round(ty - 1 - (dy + oy)) : dy + oy;
              if (rx < 0 || rx >= outW || ry < 0 || ry >= outH) continue;
              const srcX = sx + Math.floor((ox * sw) / dw);
              const srcY = sy + Math.floor((oy * sh) / dh);
              const sIdx = (srcY * img.width + srcX) * 4;
              const dIdx = (ry * outW + rx) * 4;
              dst[dIdx] = src[sIdx];
              dst[dIdx + 1] = src[sIdx + 1];
              dst[dIdx + 2] = src[sIdx + 2];
              dst[dIdx + 3] = src[sIdx + 3];
            }
          }
          canvasPixels.set(this, dst);
        },
        getImageData: (_x: number, _y: number, w: number, h: number) => ({
          data: canvasPixels.get(this) ?? new Uint8ClampedArray(w * h * 4),
          width: w,
          height: h,
        }),
        createImageData: (w: number, h: number) => ({
          data: new Uint8ClampedArray(w * h * 4),
          width: w,
          height: h,
        }),
        putImageData: (imgData: ImageData) => {
          canvasPixels.set(this, new Uint8ClampedArray(imgData.data));
        },
      } as unknown as CanvasRenderingContext2D;
    });

    const extractGray = (c: HTMLCanvasElement) => {
      const buf = canvasPixels.get(c)!;
      const out: number[] = [];
      for (let i = 0; i < buf.length; i += 4) out.push(buf[i]);
      return out;
    };

    // 1. Default (rotate180=false, swapLeftRight=false): direct left/right halves
    const normal = splitSideBySideFrame(sbsCanvas, false, 2, 2, undefined, undefined, false);
    expect(extractGray(normal.leftCanvas)).toEqual([10, 20, 30, 40]);
    expect(extractGray(normal.rightCanvas)).toEqual([50, 60, 70, 80]);

    // 2. Full-frame 180deg rotation (rotate180=true, swapLeftRight=false):
    // swaps halves AND rotates each eye 180deg
    const rotated = splitSideBySideFrame(sbsCanvas, false, 2, 2, undefined, undefined, true);
    expect(extractGray(rotated.leftCanvas)).toEqual([80, 70, 60, 50]);
    expect(extractGray(rotated.rightCanvas)).toEqual([40, 30, 20, 10]);

    // 3. XOR swap (rotate180=true, swapLeftRight=true): rotates each eye 180deg without swapping halves
    const rotatedAndSwapped = splitSideBySideFrame(sbsCanvas, true, 2, 2, undefined, undefined, true);
    expect(extractGray(rotatedAndSwapped.leftCanvas)).toEqual([40, 30, 20, 10]);
    expect(extractGray(rotatedAndSwapped.rightCanvas)).toEqual([80, 70, 60, 50]);

    // 4. StereoSadEngine tracks rotate180 on returned StereoDepthMapData and preserves it on rerender
    const sadEngine = new StereoSadEngine();
    const depth = sadEngine.estimateForVideo(sbsCanvas, { rotate180: true });
    expect(depth.rotate180).toBe(true);
    const rerendered = sadEngine.rerenderOverlay(depth, { colormap: 'viridis' });
    expect(rerendered.rotate180).toBe(true);
    sadEngine.close();
  });

  it('supports A/B normalizeSliceAxes toggle and strictWebGpu no-fallback mode in createStereoWorkerProcessor while keeping unmodified WASM reference', async () => {
    const te = new TextEncoder();
    const uVarint = (val: number): Uint8Array => {
      const b: number[] = [];
      let v = val >>> 0;
      while (v >= 0x80) {
        b.push((v & 0x7f) | 0x80);
        v >>>= 7;
      }
      b.push(v);
      return new Uint8Array(b);
    };
    const cat = (parts: readonly Uint8Array[]): Uint8Array => {
      const out = new Uint8Array(parts.reduce((acc, p) => acc + p.byteLength, 0));
      let off = 0;
      for (const p of parts) {
        out.set(p, off);
        off += p.byteLength;
      }
      return out;
    };
    const lf = (fn: number, payload: Uint8Array) =>
      cat([uVarint((fn << 3) | 2), uVarint(payload.byteLength), payload]);
    const vf = (fn: number, val: number) => cat([uVarint((fn << 3) | 0), uVarint(val)]);
    const sf = (fn: number, s: string) => lf(fn, te.encode(s));
    const attrInts = (name: string, vals: number[]) =>
      lf(5, cat([sf(1, name), ...vals.map((v) => vf(8, v)), vf(20, 7)]));
    const int64Init = (name: string, vals: number[]) => {
      const raw = new Uint8Array(vals.length * 8);
      const dv = new DataView(raw.buffer);
      vals.forEach((v, i) => dv.setBigInt64(i * 8, BigInt(v), true));
      return lf(5, cat([vf(1, vals.length), vf(2, 7), sf(8, name), lf(9, raw)]));
    };
    const floatInit = (name: string, dims: number[]) => {
      const len = dims.reduce((a, b) => a * b, 1);
      return lf(
        5,
        cat([...dims.map((d) => vf(1, d)), vf(2, 1), sf(8, name), lf(9, new Uint8Array(len * 4))])
      );
    };
    const valueInfo = (name: string, dims: number[]) =>
      lf(
        1,
        cat([
          vf(1, 1),
          lf(2, cat(dims.map((d) => lf(1, vf(1, d))))),
        ])
      );
    const hitnetLikeOnnx = cat([
      vf(1, 7),
      lf(
        7,
        cat([
          lf(1, cat([sf(1, 'input'), sf(1, 's'), sf(1, 'e'), sf(1, 'a_unsorted'), sf(2, 'sliced'), sf(3, 'Slice_0'), sf(4, 'Slice')])),
          lf(
            1,
            cat([
              sf(1, 'sliced'),
              sf(1, 'w5d'),
              sf(2, 'output'),
              sf(3, 'ConvT_0'),
              sf(4, 'ConvTranspose'),
              attrInts('kernel_shape', [4, 4, 1]),
              attrInts('strides', [4, 4, 1]),
              attrInts('pads', [0, 0, 0, 0, 0, 0]),
            ])
          ),
          sf(2, 'worker_ab_graph'),
          int64Init('s', [0, 0, 0, 0]),
          int64Init('e', [1, 16, 16, 3]),
          int64Init('a_unsorted', [0, 2, 3, 1]),
          floatInit('w5d', [3, 1, 4, 4, 1]),
          lf(11, cat([sf(1, 'input'), lf(2, valueInfo('input', [1, 6, 16, 16]))])),
          lf(12, cat([sf(1, 'output'), lf(2, valueInfo('output', [1, 1, 16, 16]))])),
        ])
      ),
      lf(8, cat([sf(1, ''), vf(2, 12)])),
    ]);

    const sendMsg = (
      processor: ReturnType<typeof createStereoWorkerProcessor>,
      msg: StereoWorkerRequest
    ): Promise<StereoWorkerResponse> =>
      new Promise((resolve) => {
        void processor.handleMessage(msg, (res) => resolve(res));
      });

    const estimatePayload = (): StereoWorkerRequest => {
      const left = new Uint8ClampedArray(16 * 16 * 4).buffer;
      const right = new Uint8ClampedArray(16 * 16 * 4).buffer;
      return {
        type: 'estimate',
        id: 2,
        width: 16,
        height: 16,
        sourceEyeWidth: 16,
        sourceEyeHeight: 16,
        leftRgbaBuffer: left,
        rightRgbaBuffer: right,
        options: { colormap: 'turbo' },
      };
    };

    const calls: Array<{
      bytes: Uint8Array;
      delegate: 'WebGPU' | 'WASM';
      strictWebGpu: boolean | undefined;
    }> = [];
    let shouldFailRun = false;

    const processor = createStereoWorkerProcessor(async (onnxBytes, delegate, _p, _f, strictWebGpu) => {
      calls.push({ bytes: new Uint8Array(onnxBytes), delegate, strictWebGpu });
      const mockSession = {
        inputNames: ['input'],
        outputNames: ['output'],
        inputMetadata: [{ shape: [1, 6, 16, 16] }],
        run: vi.fn(async () => {
          if (shouldFailRun) {
            throw new Error(
              '[WebGPU] Kernel [Conv] fe_shared/conv_down_1_0/BiasAdd failed: FILTER_IN_CHANNEL should be equal to DATA_CHANNEL'
            );
          }
          return {
            output: {
              dims: [1, 1, 16, 16],
              data: new Float32Array(16 * 16).fill(4),
              dispose: vi.fn(),
            } as unknown as ort.Tensor,
          };
        }),
        release: vi.fn(async () => {}),
      } as unknown as ort.InferenceSession;
      return {
        session: mockSession,
        backend: {
          runtime: 'onnxruntime-web',
          requestedProvider: delegate === 'WebGPU' ? 'webgpu' : 'wasm',
          configuredProviders: delegate === 'WebGPU' ? ['webgpu'] : ['wasm'],
          fallbackReason: null,
          actualBackend: 'unmeasured',
        },
        wasmSourceUsed: 'primary',
      };
    });

    // A) Original (5D ConvTranspose decomposed only, normalizeSliceAxes: false, strictWebGpu: true)
    shouldFailRun = true;
    const initA = await sendMsg(processor, {
      type: 'init-onnx',
      id: 1,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: hitnetLikeOnnx.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'webgpu',
      normalizeSliceAxes: false,
      strictWebGpu: true,
    });
    expect(initA.type).toBe('init-result');
    if (initA.type === 'init-result') {
      expect(initA.transformStats).toMatchObject({
        splitCount: 0,
        decomposed5dCount: 1,
        sliceNormalizedCount: 0,
      });
    }
    expect(calls[0].delegate).toBe('WebGPU');
    expect(calls[0].strictWebGpu).toBe(true);
    expect(inspectStereoOnnxWebGpuCompatibility(calls[0].bytes).hasConv3d).toBe(false);
    // Unsorted Slice must still be present in Arm A bytes
    expect(normalizeNonAscendingSliceAxesForWebGpu(calls[0].bytes).normalizedCount).toBe(1);

    // Strict WebGPU must NOT silently fall back to WASM when kernel fails
    const estA = await sendMsg(processor, estimatePayload());
    expect(estA.type).toBe('error');
    expect(calls).toHaveLength(1);

    // B) Tuple-sorted (5D ConvTranspose + normalizeSliceAxes: true, strictWebGpu: true)
    shouldFailRun = false;
    const initB = await sendMsg(processor, {
      type: 'init-onnx',
      id: 3,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: hitnetLikeOnnx.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'webgpu',
      normalizeSliceAxes: true,
      strictWebGpu: true,
    });
    expect(initB.type).toBe('init-result');
    if (initB.type === 'init-result') {
      expect(initB.transformStats).toMatchObject({
        splitCount: 0,
        decomposed5dCount: 1,
        sliceNormalizedCount: 1,
      });
    }
    expect(calls[1].delegate).toBe('WebGPU');
    expect(calls[1].strictWebGpu).toBe(true);
    expect(inspectStereoOnnxWebGpuCompatibility(calls[1].bytes).hasConv3d).toBe(false);
    expect(normalizeNonAscendingSliceAxesForWebGpu(calls[1].bytes).normalizedCount).toBe(0);

    // C) Non-strict Auto fallback still recovers to UNMODIFIED WASM bytes when WebGPU kernel throws
    let remainingKernelFailures = 1;
    const autoProcessor = createStereoWorkerProcessor(
      async (onnxBytes, delegate, _p, _f, strictWebGpu) => {
        calls.push({ bytes: new Uint8Array(onnxBytes), delegate, strictWebGpu });
        const mockSession = {
          inputNames: ['input'],
          outputNames: ['output'],
          inputMetadata: [{ shape: [1, 6, 16, 16] }],
          run: vi.fn(async () => {
            if (remainingKernelFailures > 0) {
              remainingKernelFailures--;
              throw new Error(
                '[WebGPU] Kernel [Conv] fe_shared/conv_down_1_0/BiasAdd failed: FILTER_IN_CHANNEL should be equal to DATA_CHANNEL'
              );
            }
            return {
              output: {
                dims: [1, 1, 16, 16],
                data: new Float32Array(16 * 16).fill(4),
                dispose: vi.fn(),
              } as unknown as ort.Tensor,
            };
          }),
          release: vi.fn(async () => {}),
        } as unknown as ort.InferenceSession;
        return {
          session: mockSession,
          backend: {
            runtime: 'onnxruntime-web',
            requestedProvider: delegate === 'WebGPU' ? 'webgpu' : 'wasm',
            configuredProviders: delegate === 'WebGPU' ? ['webgpu'] : ['wasm'],
            fallbackReason: null,
            actualBackend: 'unmeasured',
          },
          wasmSourceUsed: 'primary',
        };
      }
    );
    await sendMsg(autoProcessor, {
      type: 'init-onnx',
      id: 4,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: hitnetLikeOnnx.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'auto',
      normalizeSliceAxes: true,
      strictWebGpu: false,
    });
    const estAuto = await sendMsg(autoProcessor, estimatePayload());
    expect(estAuto.type).toBe('estimate-result');
    if (estAuto.type === 'estimate-result') {
      expect(estAuto.backend.configuredProviders).toEqual(['wasm']);
      expect(estAuto.backend.fallbackReason).toContain('fell back to WASM');
    }
    const recoveredWasmCall = calls[calls.length - 1];
    expect(recoveredWasmCall.delegate).toBe('WASM');
    expect(inspectStereoOnnxWebGpuCompatibility(recoveredWasmCall.bytes).hasConv3d).toBe(true);
    expect(normalizeNonAscendingSliceAxesForWebGpu(recoveredWasmCall.bytes).normalizedCount).toBe(1);
    await sendMsg(autoProcessor, { type: 'dispose', id: 40 });

    // D) Omitting normalizeSliceAxes in production defaults to OFF (no unverified Slice rewrite)
    const initDefaultOff = await sendMsg(processor, {
      type: 'init-onnx',
      id: 41,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: hitnetLikeOnnx.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'webgpu',
      strictWebGpu: true,
    });
    expect(initDefaultOff.type).toBe('init-result');
    if (initDefaultOff.type === 'init-result') {
      expect(initDefaultOff.transformStats?.sliceNormalizedCount).toBe(0);
      expect(initDefaultOff.sessionOptionsApplied?.sliceNormalizeMode).toBe('off');
    }

    // E) Non-HITNet variant NEVER rewrites Slice nodes even when compact-active is passed
    const initOtherModel = await sendMsg(processor, {
      type: 'init-onnx',
      id: 42,
      profile: STEREO_MODEL_PROFILES['lightstereo-s'],
      onnxBuffer: hitnetLikeOnnx.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'webgpu',
      normalizeSliceAxes: 'compact-active',
      strictWebGpu: true,
    });
    expect(initOtherModel.type).toBe('init-result');
    if (initOtherModel.type === 'init-result') {
      expect(initOtherModel.transformStats?.sliceNormalizedCount).toBe(0);
      expect(initOtherModel.sessionOptionsApplied?.sliceNormalizeMode).toBe('off');
    }

    // F) compact-active + graphOptimizationLevel + disableCpuEpFallback plumbing on hitnet-sf
    const optOverridesSeen: unknown[] = [];
    const strictMatrixProcessor = createStereoWorkerProcessor(
      async (onnxBytes, delegate, _p, _f, strictWebGpu, overrides) => {
        calls.push({ bytes: new Uint8Array(onnxBytes), delegate, strictWebGpu });
        optOverridesSeen.push(overrides);
        const mockSession = {
          inputNames: ['input'],
          outputNames: ['output'],
          inputMetadata: [{ shape: [1, 6, 16, 16] }],
          run: vi.fn(async () => ({
            output: {
              dims: [1, 1, 16, 16],
              data: new Float32Array(16 * 16).fill(4),
              dispose: vi.fn(),
            } as unknown as ort.Tensor,
          })),
          release: vi.fn(async () => {}),
        } as unknown as ort.InferenceSession;
        return {
          session: mockSession,
          backend: {
            runtime: 'onnxruntime-web',
            requestedProvider: delegate === 'WebGPU' ? 'webgpu' : 'wasm',
            configuredProviders: delegate === 'WebGPU' ? ['webgpu'] : ['wasm'],
            fallbackReason: null,
            actualBackend: 'unmeasured',
          },
          wasmSourceUsed: 'primary',
        };
      }
    );
    const initCompact = await sendMsg(strictMatrixProcessor, {
      type: 'init-onnx',
      id: 43,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: hitnetLikeOnnx.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'webgpu',
      normalizeSliceAxes: 'compact-active',
      strictWebGpu: true,
      graphOptimizationLevel: 'basic',
      disableCpuEpFallback: true,
    });
    expect(initCompact.type).toBe('init-result');
    if (initCompact.type === 'init-result') {
      expect(initCompact.transformStats).toMatchObject({
        decomposed5dCount: 1,
        sliceNormalizedCount: 1,
        sliceCompactedCount: 1,
      });
      expect(initCompact.sessionOptionsApplied).toMatchObject({
        graphOptimizationLevel: 'basic',
        disableCpuEpFallback: true,
        sliceNormalizeMode: 'compact-active',
      });
    }
    expect(optOverridesSeen[0]).toMatchObject({
      graphOptimizationLevel: 'basic',
      disableCpuEpFallback: true,
    });
    expect(
      resolveStereoOrtSessionOptions(['webgpu'], {
        graphOptimizationLevel: 'basic',
        disableCpuEpFallback: true,
      })
    ).toMatchObject({
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'basic',
      extra: { session: { disable_cpu_ep_fallback: '1' } },
    });
    await sendMsg(strictMatrixProcessor, { type: 'dispose', id: 44 });

    // G) Explicit backendPreference: 'wasm' skips BOTH 5D decomposition and Slice normalization
    const initWasm = await sendMsg(processor, {
      type: 'init-onnx',
      id: 5,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: hitnetLikeOnnx.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'wasm',
    });
    expect(initWasm.type).toBe('init-result');
    if (initWasm.type === 'init-result') {
      expect(initWasm.transformStats).toEqual({
        splitCount: 0,
        decomposed5dCount: 0,
        sliceNormalizedCount: 0,
        sliceCompactedCount: 0,
        sliceSkippedDynamicCount: 0,
        sliceSkippedInvalidOrUnprovenCount: 0,
      });
    }
    const lastCall = calls[calls.length - 1];
    expect(lastCall.delegate).toBe('WASM');
    expect(inspectStereoOnnxWebGpuCompatibility(lastCall.bytes).hasConv3d).toBe(true);
    expect(normalizeNonAscendingSliceAxesForWebGpu(lastCall.bytes).normalizedCount).toBe(1);

    await sendMsg(processor, { type: 'dispose', id: 6 });
  });

  it('gates normalizeGatherNdRank as opt-in on hitnet-sf, preserves raw WASM fallback bytes, and returns pre-conversion rawOutputs when requested', async () => {
    const calls: Array<{ bytes: Uint8Array; delegate: 'WebGPU' | 'WASM'; strictWebGpu: boolean }> = [];
    const seenFetches: Array<readonly string[] | undefined> = [];

    const processor = createStereoWorkerProcessor(
      async (onnxBytes, delegate, _p, _f, strictWebGpu) => {
        calls.push({ bytes: new Uint8Array(onnxBytes), delegate, strictWebGpu: Boolean(strictWebGpu) });
        const mockSession = {
          inputNames: ['input'],
          outputNames: ['reference_output_disparity', 'secondary_output_disparity'],
          inputMetadata: [{ shape: [1, 6, 16, 16] }],
          run: vi.fn(async (_feeds, fetches?: readonly string[]) => {
            seenFetches.push(fetches);
            const rawRef = new Float32Array(16 * 16).fill(1.25);
            rawRef[0] = 3.25;
            rawRef[1] = -2.5;
            const rawSec = new Float32Array(16 * 16).fill(1.75);
            const full = {
              reference_output_disparity: {
                dims: [1, 16, 16, 1],
                data: rawRef,
                dispose: vi.fn(),
              } as unknown as ort.Tensor,
              secondary_output_disparity: {
                dims: [1, 16, 16, 1],
                data: rawSec,
                dispose: vi.fn(),
              } as unknown as ort.Tensor,
            };
            if (fetches && fetches.length === 1 && fetches[0] === 'reference_output_disparity') {
              return { reference_output_disparity: full.reference_output_disparity };
            }
            return full;
          }),
          release: vi.fn(async () => {}),
        } as unknown as ort.InferenceSession;
        return {
          session: mockSession,
          backend: {
            runtime: 'onnxruntime-web',
            requestedProvider: delegate === 'WebGPU' ? 'webgpu' : 'wasm',
            configuredProviders: delegate === 'WebGPU' ? ['webgpu'] : ['wasm'],
            fallbackReason: null,
            actualBackend: 'unmeasured',
          },
          wasmSourceUsed: 'primary',
        };
      }
    );

    const sendMsg = (req: StereoWorkerRequest) =>
      new Promise<StereoWorkerResponse>((resolve) => {
        void processor.handleMessage(req, (res) => resolve(res));
      });

    // Minimal valid ONNX header bytes suffice to verify worker default-off & rawOutputs contract
    const rawOnnx = new Uint8Array([0x08, 0x07]);
    const initDefault = await sendMsg({
      type: 'init-onnx',
      id: 100,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawOnnx.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'webgpu',
      strictWebGpu: true,
    });
    expect(initDefault.type).toBe('init-result');
    if (initDefault.type === 'init-result') {
      expect(initDefault.sessionOptionsApplied?.gatherNdRankNormalized).toBe(false);
      expect(initDefault.gatherNdStats?.normalizedCount).toBe(0);
      expect(initDefault.initTiming?.totalInitMs).toBeGreaterThanOrEqual(0);
    }

    const rgba = new Uint8Array(16 * 16 * 4);
    // Request raw pre-conversion outputs (preserving negative disparity -2.5 & unscaled 3.25 before display scaling!)
    const estAll = await sendMsg({
      type: 'estimate',
      id: 101,
      leftRgbaBuffer: rgba.slice().buffer,
      rightRgbaBuffer: rgba.slice().buffer,
      width: 16,
      height: 16,
      sourceEyeWidth: 32,
      sourceEyeHeight: 16,
      options: { autoNormalizeDisparity: true, colormap: 'turbo' },
      includeRawTensorOutputs: true,
    });
    expect(estAll.type).toBe('estimate-result');
    if (estAll.type === 'estimate-result') {
      expect(seenFetches[0]).toBeUndefined();
      expect(estAll.fetchedOutputNames).toEqual([
        'reference_output_disparity',
        'secondary_output_disparity',
      ]);
      expect(estAll.rawOutputs).toHaveLength(2);
      const rawRefArr = new Float32Array(estAll.rawOutputs![0].buffer);
      // Unclamped and unscaled raw model values preserved
      expect(rawRefArr[0]).toBeCloseTo(3.25, 5);
      expect(rawRefArr[1]).toBeCloseTo(-2.5, 5);
      // Display buffer is clamped (<0 -> -1) and scaled by sourceEyeWidth / width = 2
      const dispArr = new Float32Array(estAll.disparityBuffer);
      expect(dispArr[0]).toBeCloseTo(6.5, 5);
      expect(dispArr[1]).toBe(-1);
      expect(estAll.estimateTiming?.runMs).toBeGreaterThanOrEqual(0);
    }

    // Diagnostic-only fetchPrimaryOutputOnly skips secondary output cleanly without altering production defaults
    const estPrimary = await sendMsg({
      type: 'estimate',
      id: 102,
      leftRgbaBuffer: rgba.slice().buffer,
      rightRgbaBuffer: rgba.slice().buffer,
      width: 16,
      height: 16,
      sourceEyeWidth: 32,
      sourceEyeHeight: 16,
      options: {},
      includeRawTensorOutputs: true,
      fetchPrimaryOutputOnly: true,
    });
    expect(estPrimary.type).toBe('estimate-result');
    if (estPrimary.type === 'estimate-result') {
      expect(seenFetches[1]).toEqual(['reference_output_disparity']);
      expect(estPrimary.fetchedOutputNames).toEqual(['reference_output_disparity']);
      expect(estPrimary.rawOutputs).toHaveLength(1);
    }

    await sendMsg({ type: 'dispose', id: 103 });

    // Real cached HITNet worker plumbing verification: opt-in only, non-HITNet blocked, and unmodified WASM fallback
    const hitnetPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (existsSync(hitnetPath)) {
      const hitnetBytes = new Uint8Array(readFileSync(hitnetPath));

      // Non-HITNet variant must NOT run GatherND rewrite
      const initOther = await sendMsg({
        type: 'init-onnx',
        id: 104,
        profile: STEREO_MODEL_PROFILES['lightstereo-s'],
        onnxBuffer: hitnetBytes.slice().buffer,
        wasmPrimaryUrl: 'https://example.test/ort/',
        wasmFallbackUrl: null,
        backendPreference: 'webgpu',
        normalizeSliceAxes: 'compact-active',
        normalizeGatherNdRank: true,
        strictWebGpu: true,
      });
      expect(initOther.type).toBe('init-result');
      if (initOther.type === 'init-result') {
        expect(initOther.gatherNdStats?.normalizedCount).toBe(0);
        expect(initOther.sessionOptionsApplied?.gatherNdRankNormalized).toBe(false);
      }

      // HITNet-SF with opt-in normalizeGatherNdRank: true rewrites exact 4 nodes and still preserves unmodified WASM fallback
      const shouldThrowWebGpuRun = true;
      const fallbackCalls: Array<{ bytes: Uint8Array; delegate: 'WebGPU' | 'WASM' }> = [];
      const fallbackProcessor = createStereoWorkerProcessor(
        async (onnxBytes, delegate) => {
          fallbackCalls.push({ bytes: new Uint8Array(onnxBytes), delegate });
          const mockSession = {
            inputNames: ['input'],
            outputNames: ['reference_output_disparity', 'secondary_output_disparity'],
            inputMetadata: [{ shape: [1, 6, 240, 320] }],
            run: vi.fn(async () => {
              if (delegate === 'WebGPU' && shouldThrowWebGpuRun) {
                throw new Error('[WebGPU] Kernel [GatherND] simulated runtime error');
              }
              return {
                reference_output_disparity: {
                  dims: [1, 240, 320, 1],
                  data: new Float32Array(240 * 320).fill(1.5),
                  dispose: vi.fn(),
                } as unknown as ort.Tensor,
              };
            }),
            release: vi.fn(async () => {}),
          } as unknown as ort.InferenceSession;
          return {
            session: mockSession,
            backend: {
              runtime: 'onnxruntime-web',
              requestedProvider: delegate === 'WebGPU' ? 'webgpu' : 'wasm',
              configuredProviders: delegate === 'WebGPU' ? ['webgpu'] : ['wasm'],
              fallbackReason: null,
              actualBackend: 'unmeasured',
            },
            wasmSourceUsed: 'primary',
          };
        }
      );

      const sendFb = (req: StereoWorkerRequest) =>
        new Promise<StereoWorkerResponse>((resolve) => {
          void fallbackProcessor.handleMessage(req, (res) => resolve(res));
        });

      const initHitnetOptIn = await sendFb({
        type: 'init-onnx',
        id: 105,
        profile: STEREO_MODEL_PROFILES['hitnet-sf'],
        onnxBuffer: hitnetBytes.slice().buffer,
        wasmPrimaryUrl: 'https://example.test/ort/',
        wasmFallbackUrl: null,
        backendPreference: 'auto',
        normalizeSliceAxes: 'compact-active',
        normalizeGatherNdRank: true,
        strictWebGpu: false,
      });
      expect(initHitnetOptIn.type).toBe('init-result');
      if (initHitnetOptIn.type === 'init-result') {
        expect(initHitnetOptIn.gatherNdStats?.normalizedCount).toBe(4);
        expect(initHitnetOptIn.sessionOptionsApplied?.gatherNdRankNormalized).toBe(true);
      }

      const rgbHit = new Uint8Array(320 * 240 * 4);
      const estRecovered = await sendFb({
        type: 'estimate',
        id: 106,
        leftRgbaBuffer: rgbHit.slice().buffer,
        rightRgbaBuffer: rgbHit.slice().buffer,
        width: 320,
        height: 240,
        sourceEyeWidth: 480,
        sourceEyeHeight: 270,
        options: {},
      });
      expect(estRecovered.type).toBe('estimate-result');
      expect(fallbackCalls).toHaveLength(2);
      expect(fallbackCalls[0].delegate).toBe('WebGPU');
      expect(fallbackCalls[1].delegate).toBe('WASM');
      // Fallback WASM session must receive UNMODIFIED raw model bytes (exact raw HITNet bytes with 5D Conv3D & unnormalized GatherND)
      expect(fallbackCalls[1].bytes.byteLength).toBe(hitnetBytes.byteLength);
      expect(Buffer.compare(fallbackCalls[1].bytes, hitnetBytes)).toBe(0);
      expect(inspectStereoOnnxWebGpuCompatibility(fallbackCalls[1].bytes).hasConv3d).toBe(true);
      const fbDecomposed = decomposeOnnx5dConvNodesForWebGpu(fallbackCalls[1].bytes);
      const fbSliced = normalizeNonAscendingSliceAxesForWebGpu(fbDecomposed.bytes, { mode: 'compact-active' });
      expect(
        normalizeHitnetGatherNdRankForWebGpu(fbSliced.bytes, {
          variantId: 'hitnet-sf',
          backendPreference: 'webgpu',
        }).normalizedCount
      ).toBe(4);

      await sendFb({ type: 'dispose', id: 107 });
    }
  }, 30000);

  it('gates normalizeArgMinKeepDims & allowDiagnosticTransformedWasm, logs sessionModelSha256 & feedSha256s, and preserves raw explicit/fallback WASM bytes', async () => {
    const calls: Array<{
      bytes: Uint8Array;
      delegate: 'WebGPU' | 'WASM';
      sha256: string;
    }> = [];
    const seenFeedData: Array<Float32Array> = [];

    const processor = createStereoWorkerProcessor(
      async (onnxBytes, delegate) => {
        const copy = new Uint8Array(onnxBytes);
        const sha256 = createHash('sha256').update(copy).digest('hex');
        calls.push({ bytes: copy, delegate, sha256 });
        const mockSession = {
          inputNames: ['input'],
          outputNames: ['reference_output_disparity', 'secondary_output_disparity'],
          inputMetadata: [{ shape: [1, 6, 240, 320] }],
          run: vi.fn(async (feeds: Record<string, ort.Tensor>) => {
            const inp = feeds.input;
            seenFeedData.push(new Float32Array(inp.data as Float32Array));
            if (delegate === 'WebGPU' && calls.length === 4) {
              throw new Error('[WebGPU] Kernel [ArgMin] simulated runtime failure');
            }
            return {
              reference_output_disparity: {
                type: 'float32',
                dims: [1, 240, 320, 1],
                data: new Float32Array(240 * 320).fill(1.5),
                dispose: vi.fn(),
              } as unknown as ort.Tensor,
              secondary_output_disparity: {
                type: 'float32',
                dims: [1, 240, 320, 1],
                data: new Float32Array(240 * 320).fill(1.0),
                dispose: vi.fn(),
              } as unknown as ort.Tensor,
            };
          }),
          release: vi.fn(async () => {}),
        } as unknown as ort.InferenceSession;
        return {
          session: mockSession,
          backend: {
            runtime: 'onnxruntime-web',
            requestedProvider: delegate === 'WebGPU' ? 'webgpu' : 'wasm',
            configuredProviders: delegate === 'WebGPU' ? ['webgpu'] : ['wasm'],
            fallbackReason: null,
            actualBackend: 'unmeasured',
          },
          wasmSourceUsed: 'primary',
        };
      }
    );

    const sendMsg = (req: StereoWorkerRequest) =>
      new Promise<StereoWorkerResponse>((resolve) => {
        void processor.handleMessage(req, (res) => resolve(res));
      });

    const hitnetPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (!existsSync(hitnetPath)) {
      return;
    }
    const rawBytes = new Uint8Array(readFileSync(hitnetPath));
    const rawSha = createHash('sha256').update(rawBytes).digest('hex');

    // 1) Default WebGPU init keeps argMinKeepDims OFF
    const initDefault = await sendMsg({
      type: 'init-onnx',
      id: 201,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'webgpu',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      strictWebGpu: true,
    });
    expect(initDefault.type).toBe('init-result');
    if (initDefault.type === 'init-result') {
      expect(initDefault.rawInputModelSha256).toBe(rawSha);
      expect(initDefault.sessionModelSha256).toBe(calls[0].sha256);
      expect(initDefault.sessionModelSha256).not.toBe(rawSha);
      expect(initDefault.sessionOptionsApplied?.argMinKeepDimsNormalized).toBe(false);
      expect(initDefault.sessionOptionsApplied?.allowDiagnosticTransformedWasm).toBe(false);
      expect(initDefault.argMinStats?.normalizedCount).toBe(0);
    }

    // 2) Explicit WASM WITHOUT allowDiagnosticTransformedWasm ignores transform flags and sends exact raw bytes
    const initExplicitWasm = await sendMsg({
      type: 'init-onnx',
      id: 202,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'wasm',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizeArgMinKeepDims: true,
      allowDiagnosticTransformedWasm: false,
    });
    expect(initExplicitWasm.type).toBe('init-result');
    if (initExplicitWasm.type === 'init-result') {
      expect(initExplicitWasm.rawInputModelSha256).toBe(rawSha);
      expect(initExplicitWasm.sessionModelSha256).toBe(rawSha);
      expect(calls[1].sha256).toBe(rawSha);
      expect(Buffer.compare(calls[1].bytes, rawBytes)).toBe(0);
      expect(initExplicitWasm.argMinStats?.normalizedCount).toBe(0);
      expect(initExplicitWasm.sessionOptionsApplied?.allowDiagnosticTransformedWasm).toBe(false);
    }

    // 3) Explicit WASM WITH allowDiagnosticTransformedWasm=true (diagnostic control only) transforms cleanly to F1/G1 bytes
    const initDiagWasmG1 = await sendMsg({
      type: 'init-onnx',
      id: 203,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'wasm',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizeArgMinKeepDims: true,
      allowDiagnosticTransformedWasm: true,
    });
    expect(initDiagWasmG1.type).toBe('init-result');
    if (initDiagWasmG1.type === 'init-result') {
      expect(initDiagWasmG1.rawInputModelSha256).toBe(rawSha);
      expect(initDiagWasmG1.sessionModelSha256).toBe(calls[2].sha256);
      expect(initDiagWasmG1.sessionModelSha256).not.toBe(rawSha);
      expect(initDiagWasmG1.argMinStats).toMatchObject({
        normalizedCount: 2,
        totalArgMinCount: 4,
        ortCacheKeyBefore: 'ArgMin[3;false;0]:false:1;4',
        ortCacheKeyAfter: 'ArgMin[3;true;0]:false:1;4',
      });
      expect(initDiagWasmG1.sessionOptionsApplied?.argMinKeepDimsNormalized).toBe(true);
      expect(initDiagWasmG1.sessionOptionsApplied?.allowDiagnosticTransformedWasm).toBe(true);
    }

    const rgba = new Uint8Array(320 * 240 * 4).fill(120);
    const estDiag = await sendMsg({
      type: 'estimate',
      id: 204,
      leftRgbaBuffer: rgba.slice().buffer,
      rightRgbaBuffer: rgba.slice().buffer,
      width: 320,
      height: 240,
      sourceEyeWidth: 480,
      sourceEyeHeight: 270,
      options: {},
      includeRawTensorOutputs: true,
    });
    expect(estDiag.type).toBe('estimate-result');
    if (estDiag.type === 'estimate-result') {
      const expectedFeedSha = createHash('sha256')
        .update(new Uint8Array(seenFeedData[0].buffer, seenFeedData[0].byteOffset, seenFeedData[0].byteLength))
        .digest('hex');
      expect(estDiag.feedSha256s).toEqual({ input: expectedFeedSha });
      expect(estDiag.rawOutputs?.[0].dataType).toBe('float32');
      expect(estDiag.rawOutputs?.[1].dataType).toBe('float32');
    }

    // 4) Auto mode with G1 enabled: WebGPU gets G1 bytes, fallback WASM gets EXACT untouched raw bytes
    const initAutoG1 = await sendMsg({
      type: 'init-onnx',
      id: 205,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'auto',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizeArgMinKeepDims: true,
      strictWebGpu: false,
    });
    expect(initAutoG1.type).toBe('init-result');
    if (initAutoG1.type === 'init-result') {
      expect(initAutoG1.sessionModelSha256).toBe(calls[3].sha256);
      expect(initAutoG1.sessionModelSha256).toBe(calls[2].sha256);
    }

    const estFallback = await sendMsg({
      type: 'estimate',
      id: 206,
      leftRgbaBuffer: rgba.slice().buffer,
      rightRgbaBuffer: rgba.slice().buffer,
      width: 320,
      height: 240,
      sourceEyeWidth: 480,
      sourceEyeHeight: 270,
      options: {},
    });
    expect(estFallback.type).toBe('estimate-result');
    expect(calls).toHaveLength(5);
    expect(calls[3].delegate).toBe('WebGPU');
    expect(calls[4].delegate).toBe('WASM');
    expect(calls[4].sha256).toBe(rawSha);
    expect(Buffer.compare(calls[4].bytes, rawBytes)).toBe(0);
    // Double-check the helper directly still sees 2 unpatched prop0 ArgMin nodes after stripping nothing on raw
    expect(
      normalizeHitnetArgMinKeepDimsForWebGpu(calls[4].bytes, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
      }).normalizedCount
    ).toBe(2);

    await sendMsg({ type: 'dispose', id: 207 });
  }, 30000);

  it('gates normalizePropGatherNdRank (H1) as opt-in on hitnet-sf, fails closed on missing preconditions, and preserves raw explicit/fallback WASM bytes', async () => {
    const calls: Array<{ bytes: Uint8Array; delegate: 'WebGPU' | 'WASM'; sha256: string }> = [];
    const processor = createStereoWorkerProcessor(async (onnxBytes, delegate) => {
      const copy = new Uint8Array(onnxBytes);
      const sha256 = createHash('sha256').update(copy).digest('hex');
      calls.push({ bytes: copy, delegate, sha256 });
      const mockSession = {
        inputNames: ['input'],
        outputNames: ['reference_output_disparity', 'secondary_output_disparity'],
        inputMetadata: [{ shape: [1, 6, 240, 320] }],
        run: vi.fn(async () => {
          if (delegate === 'WebGPU' && calls.length >= 4) {
            throw new Error('[WebGPU] Kernel [GatherND] simulated H1 fallback trigger');
          }
          return {
            reference_output_disparity: {
              type: 'float32',
              dims: [1, 240, 320, 1],
              data: new Float32Array(240 * 320).fill(1.25),
              dispose: vi.fn(),
            } as unknown as ort.Tensor,
            secondary_output_disparity: {
              type: 'float32',
              dims: [1, 240, 320, 1],
              data: new Float32Array(240 * 320).fill(0.75),
              dispose: vi.fn(),
            } as unknown as ort.Tensor,
          };
        }),
        release: vi.fn(async () => {}),
      } as unknown as ort.InferenceSession;
      return {
        session: mockSession,
        backend: {
          runtime: 'onnxruntime-web',
          requestedProvider: delegate === 'WebGPU' ? 'webgpu' : 'wasm',
          configuredProviders: delegate === 'WebGPU' ? ['webgpu'] : ['wasm'],
          fallbackReason: null,
          actualBackend: 'unmeasured',
        },
        wasmSourceUsed: 'primary',
      };
    });

    const sendMsg = (req: StereoWorkerRequest) =>
      new Promise<StereoWorkerResponse>((resolve) => {
        void processor.handleMessage(req, (res) => resolve(res));
      });

    const hitnetPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (!existsSync(hitnetPath)) {
      return;
    }
    const rawBytes = new Uint8Array(readFileSync(hitnetPath));
    const rawSnapshot = rawBytes.slice();
    const rawSha = createHash('sha256').update(rawBytes).digest('hex');

    // Direct helper fail-closed guards & non-mutation
    expect(
      normalizeHitnetPropGatherNdRankForWebGpu(rawBytes, {
        variantId: 'lightstereo-s',
        backendPreference: 'webgpu',
      }).normalizedCount
    ).toBe(0);
    expect(
      normalizeHitnetPropGatherNdRankForWebGpu(rawBytes, {
        variantId: 'hitnet-sf',
        backendPreference: 'wasm',
      }).normalizedCount
    ).toBe(0);
    expect(Buffer.compare(rawBytes, rawSnapshot)).toBe(0);
    expect(inferOnnxStaticTensorMeta(rawBytes)).toMatchObject({
      valid: true,
      irVersion: 7,
      defaultOpset: 12,
      duplicateOutputs: [],
    });

    // 1) Missing F1 preconditions under strictWebGpu must fail closed with error
    const missingPrecond = await sendMsg({
      type: 'init-onnx',
      id: 301,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'webgpu',
      normalizeSliceAxes: 'off',
      normalizeGatherNdRank: false,
      normalizePropGatherNdRank: true,
      strictWebGpu: true,
    });
    expect(missingPrecond.type).toBe('error');

    // 2) Full F1 + H1 opt-in rewrites exact 2 prop0 nodes, keeps ArgMin keepdims=1 OFF
    const initH1 = await sendMsg({
      type: 'init-onnx',
      id: 302,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'webgpu',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizePropGatherNdRank: true,
      strictWebGpu: true,
    });
    expect(initH1.type).toBe('init-result');
    if (initH1.type === 'init-result') {
      expect(initH1.sessionOptionsApplied?.propGatherNdRankNormalized).toBe(true);
      expect(initH1.sessionOptionsApplied?.argMinKeepDimsNormalized).toBe(false);
      expect(initH1.propGatherNdStats).toMatchObject({
        normalizedCount: 2,
        totalGatherNdCount: 10,
        skippedInvalidOrUnprovenCount: 0,
      });
      expect(initH1.propGatherNdStats?.normalizedNodeNames).toEqual([
        'level0/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis',
        'level0_1/shared/prop0/GatherV2;level0/level_init/GatherV2_1/axis',
      ]);
      expect(initH1.sessionModelSha256).toBe(calls[0].sha256);
    }

    // 3) Diagnostic H1 WASM matches exact H1 WebGPU session bytes
    const initDiagWasmH1 = await sendMsg({
      type: 'init-onnx',
      id: 303,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'wasm',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizePropGatherNdRank: true,
      allowDiagnosticTransformedWasm: true,
    });
    expect(initDiagWasmH1.type).toBe('init-result');
    if (initDiagWasmH1.type === 'init-result') {
      expect(initDiagWasmH1.sessionModelSha256).toBe(calls[0].sha256);
      expect(initDiagWasmH1.sessionOptionsApplied?.propGatherNdRankNormalized).toBe(true);
    }

    // 4) Non-HITNet variant ignores normalizePropGatherNdRank cleanly
    const initOther = await sendMsg({
      type: 'init-onnx',
      id: 304,
      profile: STEREO_MODEL_PROFILES['lightstereo-s'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'webgpu',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizePropGatherNdRank: true,
      strictWebGpu: true,
    });
    expect(initOther.type).toBe('init-result');
    if (initOther.type === 'init-result') {
      expect(initOther.propGatherNdStats?.normalizedCount).toBe(0);
      expect(initOther.sessionOptionsApplied?.propGatherNdRankNormalized).toBe(false);
    }

    // 5) Auto mode with H1 falls back on runtime error to EXACT raw untouched WASM bytes
    await sendMsg({
      type: 'init-onnx',
      id: 305,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'auto',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizePropGatherNdRank: true,
      strictWebGpu: false,
    });
    const rgba = new Uint8Array(320 * 240 * 4).fill(90);
    const estH1Fallback = await sendMsg({
      type: 'estimate',
      id: 306,
      leftRgbaBuffer: rgba.slice().buffer,
      rightRgbaBuffer: rgba.slice().buffer,
      width: 320,
      height: 240,
      sourceEyeWidth: 480,
      sourceEyeHeight: 270,
      options: {},
      captureGpuValidationErrors: true,
    });
    expect(estH1Fallback.type).toBe('estimate-result');
    const fallbackCall = calls[calls.length - 1];
    expect(fallbackCall.delegate).toBe('WASM');
    expect(fallbackCall.sha256).toBe(rawSha);
    expect(Buffer.compare(fallbackCall.bytes, rawBytes)).toBe(0);

    await sendMsg({ type: 'dispose', id: 307 });
  }, 30000);

  it('gates normalizeInitToPropZeroSlant (M1-ConstZero2Ch) strictly on H1 hitnet-sf, fails closed on L1/G1/WASM, and preserves raw fallback bytes', async () => {
    const calls: Array<{ bytes: Uint8Array; delegate: 'WebGPU' | 'WASM'; sha256: string }> = [];
    const processor = createStereoWorkerProcessor(async (onnxBytes, delegate) => {
      const copy = new Uint8Array(onnxBytes);
      const sha256 = createHash('sha256').update(copy).digest('hex');
      calls.push({ bytes: copy, delegate, sha256 });
      const mockSession = {
        inputNames: ['input'],
        outputNames: ['reference_output_disparity', 'secondary_output_disparity'],
        inputMetadata: [{ shape: [1, 6, 240, 320] }],
        run: vi.fn(async () => {
          if (delegate === 'WebGPU' && calls.length >= 4) {
            throw new Error('[WebGPU] Kernel [Concat] simulated M1 fallback trigger');
          }
          return {
            reference_output_disparity: {
              type: 'float32',
              dims: [1, 240, 320, 1],
              data: new Float32Array(240 * 320).fill(1.0),
              dispose: vi.fn(),
            } as unknown as ort.Tensor,
            secondary_output_disparity: {
              type: 'float32',
              dims: [1, 240, 320, 1],
              data: new Float32Array(240 * 320).fill(0.5),
              dispose: vi.fn(),
            } as unknown as ort.Tensor,
          };
        }),
        release: vi.fn(async () => {}),
      } as unknown as ort.InferenceSession;
      return {
        session: mockSession,
        backend: {
          runtime: 'onnxruntime-web',
          requestedProvider: delegate === 'WebGPU' ? 'webgpu' : 'wasm',
          configuredProviders: delegate === 'WebGPU' ? ['webgpu'] : ['wasm'],
          fallbackReason: null,
          actualBackend: 'unmeasured',
        },
        wasmSourceUsed: 'primary',
      };
    });

    const sendMsg = (req: StereoWorkerRequest) =>
      new Promise<StereoWorkerResponse>((resolve) => {
        void processor.handleMessage(req, (res) => resolve(res));
      });

    const hitnetPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (!existsSync(hitnetPath)) {
      return;
    }
    const rawBytes = new Uint8Array(readFileSync(hitnetPath));
    const rawSha = createHash('sha256').update(rawBytes).digest('hex');
    expect(rawSha).toBe(HITNET_RAW_MODEL_SHA256);

    // 1) Missing H1, L1 conflict, or G1 conflict must fail closed
    for (const badReq of [
      {
        normalizeSliceAxes: 'compact-active' as const,
        normalizeGatherNdRank: true,
        normalizePropGatherNdRank: false,
      },
      {
        normalizeSliceAxes: 'compact-active' as const,
        normalizeGatherNdRank: true,
        normalizePropGatherNdRank: true,
        levelInitGatherMode: 'reshape-rank4' as const,
      },
      {
        normalizeSliceAxes: 'compact-active' as const,
        normalizeGatherNdRank: true,
        normalizePropGatherNdRank: true,
        normalizeArgMinKeepDims: true,
      },
    ]) {
      const errRes = await sendMsg({
        type: 'init-onnx',
        id: 401,
        profile: STEREO_MODEL_PROFILES['hitnet-sf'],
        onnxBuffer: rawBytes.slice().buffer,
        wasmPrimaryUrl: 'https://example.test/ort/',
        wasmFallbackUrl: null,
        backendPreference: 'webgpu',
        ...badReq,
        normalizeInitToPropZeroSlant: true,
        strictWebGpu: true,
      });
      expect(errRes.type).toBe('error');
    }

    // 2) Production-like default (normalizeInitToPropZeroSlant omitted) keeps M1 OFF even on H1
    const initDefaultOff = await sendMsg({
      type: 'init-onnx',
      id: 402,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'webgpu',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizePropGatherNdRank: true,
      strictWebGpu: true,
    });
    expect(initDefaultOff.type).toBe('init-result');
    if (initDefaultOff.type === 'init-result') {
      expect(initDefaultOff.sessionModelSha256).toBe(HITNET_H1_FULL_GRAPH_SHA256);
      expect(initDefaultOff.initToPropZeroSlantStats?.normalizedCount).toBe(0);
      expect(initDefaultOff.sessionOptionsApplied?.initToPropZeroSlantNormalized).toBe(false);
    }

    // 3) Opt-in M1 on H1 succeeds cleanly and verifies H1/raw provenance + 8 dead nodes + zero__310
    const initM1WebGpu = await sendMsg({
      type: 'init-onnx',
      id: 403,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'webgpu',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizePropGatherNdRank: true,
      normalizeInitToPropZeroSlant: true,
      strictWebGpu: true,
    });
    expect(initM1WebGpu.type).toBe('init-result');
    let m1Sha = '';
    if (initM1WebGpu.type === 'init-result') {
      m1Sha = initM1WebGpu.sessionModelSha256 ?? '';
      expect(m1Sha).not.toBe(HITNET_H1_FULL_GRAPH_SHA256);
      expect(initM1WebGpu.initToPropZeroSlantStats).toMatchObject({
        normalizedCount: 2,
        removedDeadInitializerNames: ['zero__310'],
        addedInitializerName: '__webgpu_m1_init_to_prop_zero_slant_2ch',
        addedInitializerDims: [1, 2, 60, 80],
        verifiedH1Provenance: true,
        verifiedRawIdentity: true,
        skippedInvalidOrUnprovenCount: 0,
      });
      expect(initM1WebGpu.initToPropZeroSlantStats?.removedDeadNodeNames).toHaveLength(8);
      expect(initM1WebGpu.sessionOptionsApplied?.initToPropZeroSlantNormalized).toBe(true);
    }

    // 4) Explicit WASM without allowDiagnosticTransformedWasm leaves rawBytes untouched;
    //    diagnostic transformed WASM receives EXACT M1 bytes matching WebGPU.
    const initExplicitWasm = await sendMsg({
      type: 'init-onnx',
      id: 404,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'wasm',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizePropGatherNdRank: true,
      normalizeInitToPropZeroSlant: true,
      allowDiagnosticTransformedWasm: false,
    });
    expect(initExplicitWasm.type).toBe('init-result');
    if (initExplicitWasm.type === 'init-result') {
      expect(initExplicitWasm.sessionModelSha256).toBe(rawSha);
      expect(initExplicitWasm.initToPropZeroSlantStats?.normalizedCount).toBe(0);
    }

    const initDiagWasmM1 = await sendMsg({
      type: 'init-onnx',
      id: 405,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'wasm',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizePropGatherNdRank: true,
      normalizeInitToPropZeroSlant: true,
      allowDiagnosticTransformedWasm: true,
    });
    expect(initDiagWasmM1.type).toBe('init-result');
    if (initDiagWasmM1.type === 'init-result') {
      expect(initDiagWasmM1.sessionModelSha256).toBe(m1Sha);
      expect(initDiagWasmM1.sessionOptionsApplied?.initToPropZeroSlantNormalized).toBe(true);
    }

    // 5) Auto mode runtime fallback still uses exact raw untouched WASM bytes
    await sendMsg({
      type: 'init-onnx',
      id: 406,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'auto',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizePropGatherNdRank: true,
      normalizeInitToPropZeroSlant: true,
      strictWebGpu: false,
    });
    const rgba = new Uint8Array(320 * 240 * 4).fill(90);
    const estFallback = await sendMsg({
      type: 'estimate',
      id: 407,
      leftRgbaBuffer: rgba.slice().buffer,
      rightRgbaBuffer: rgba.slice().buffer,
      width: 320,
      height: 240,
      sourceEyeWidth: 480,
      sourceEyeHeight: 270,
      options: {},
    });
    expect(estFallback.type).toBe('estimate-result');
    const lastCall = calls[calls.length - 1];
    expect(lastCall.delegate).toBe('WASM');
    expect(lastCall.sha256).toBe(rawSha);
    expect(Buffer.compare(lastCall.bytes, rawBytes)).toBe(0);
    expect(
      normalizeHitnetInitToPropZeroSlantForWebGpu(lastCall.bytes, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
      }).normalizedCount
    ).toBe(0);

    await sendMsg({ type: 'dispose', id: 408 });
  }, 30000);

  it('gates normalizeSlot0ReshapeToTranspose (N1-Slot0Transpose) strictly on exact M1 hitnet-sf, fails closed without M1 or on L1/G1/WASM, and preserves raw fallback bytes', async () => {
    const calls: Array<{ bytes: Uint8Array; delegate: 'WebGPU' | 'WASM'; sha256: string }> = [];
    const processor = createStereoWorkerProcessor(async (onnxBytes, delegate) => {
      const copy = new Uint8Array(onnxBytes);
      const sha256 = createHash('sha256').update(copy).digest('hex');
      calls.push({ bytes: copy, delegate, sha256 });
      const mockSession = {
        inputNames: ['input'],
        outputNames: ['reference_output_disparity', 'secondary_output_disparity'],
        inputMetadata: [{ shape: [1, 6, 240, 320] }],
        run: vi.fn(async () => {
          if (delegate === 'WebGPU' && calls.length >= 4) {
            throw new Error('[WebGPU] Kernel [Transpose] simulated N1 fallback trigger');
          }
          return {
            reference_output_disparity: {
              type: 'float32',
              dims: [1, 240, 320, 1],
              data: new Float32Array(240 * 320).fill(1.0),
              dispose: vi.fn(),
            } as unknown as ort.Tensor,
            secondary_output_disparity: {
              type: 'float32',
              dims: [1, 240, 320, 1],
              data: new Float32Array(240 * 320).fill(0.5),
              dispose: vi.fn(),
            } as unknown as ort.Tensor,
          };
        }),
        release: vi.fn(async () => {}),
      } as unknown as ort.InferenceSession;
      return {
        session: mockSession,
        backend: {
          runtime: 'onnxruntime-web',
          requestedProvider: delegate === 'WebGPU' ? 'webgpu' : 'wasm',
          configuredProviders: delegate === 'WebGPU' ? ['webgpu'] : ['wasm'],
          fallbackReason: null,
          actualBackend: 'unmeasured',
        },
        wasmSourceUsed: 'primary',
      };
    });

    const sendMsg = (req: StereoWorkerRequest) =>
      new Promise<StereoWorkerResponse>((resolve) => {
        void processor.handleMessage(req, (res) => resolve(res));
      });

    const hitnetPath = '/tmp/hitnet_sceneflow_240x320.onnx';
    if (!existsSync(hitnetPath)) {
      return;
    }
    const rawBytes = new Uint8Array(readFileSync(hitnetPath));
    const rawSha = createHash('sha256').update(rawBytes).digest('hex');
    expect(rawSha).toBe(HITNET_RAW_MODEL_SHA256);

    // 1) Missing M1, missing H1, L1 conflict, or G1 conflict must fail closed
    for (const badReq of [
      {
        normalizeSliceAxes: 'compact-active' as const,
        normalizeGatherNdRank: true,
        normalizePropGatherNdRank: true,
        normalizeInitToPropZeroSlant: false,
      },
      {
        normalizeSliceAxes: 'compact-active' as const,
        normalizeGatherNdRank: true,
        normalizePropGatherNdRank: false,
        normalizeInitToPropZeroSlant: false,
      },
      {
        normalizeSliceAxes: 'compact-active' as const,
        normalizeGatherNdRank: true,
        normalizePropGatherNdRank: true,
        normalizeInitToPropZeroSlant: true,
        levelInitGatherMode: 'reshape-rank4' as const,
      },
      {
        normalizeSliceAxes: 'compact-active' as const,
        normalizeGatherNdRank: true,
        normalizePropGatherNdRank: true,
        normalizeInitToPropZeroSlant: true,
        normalizeArgMinKeepDims: true,
      },
    ]) {
      const errRes = await sendMsg({
        type: 'init-onnx',
        id: 501,
        profile: STEREO_MODEL_PROFILES['hitnet-sf'],
        onnxBuffer: rawBytes.slice().buffer,
        wasmPrimaryUrl: 'https://example.test/ort/',
        wasmFallbackUrl: null,
        backendPreference: 'webgpu',
        ...badReq,
        normalizeSlot0ReshapeToTranspose: true,
        strictWebGpu: true,
      });
      expect(errRes.type).toBe('error');
    }

    // 2) Production-like default (normalizeSlot0ReshapeToTranspose omitted) keeps N1 OFF even on M1
    const initDefaultOff = await sendMsg({
      type: 'init-onnx',
      id: 502,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'webgpu',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizePropGatherNdRank: true,
      normalizeInitToPropZeroSlant: true,
      strictWebGpu: true,
    });
    expect(initDefaultOff.type).toBe('init-result');
    if (initDefaultOff.type === 'init-result') {
      expect(initDefaultOff.sessionModelSha256).toBe(HITNET_M1_FULL_GRAPH_SHA256);
      expect(initDefaultOff.slot0ReshapeToTransposeStats?.normalizedCount).toBe(0);
      expect(initDefaultOff.sessionOptionsApplied?.slot0ReshapeToTransposeNormalized).toBe(false);
    }

    // 3) Opt-in N1 on exact M1 succeeds cleanly
    const initN1WebGpu = await sendMsg({
      type: 'init-onnx',
      id: 503,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'webgpu',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizePropGatherNdRank: true,
      normalizeInitToPropZeroSlant: true,
      normalizeSlot0ReshapeToTranspose: true,
      strictWebGpu: true,
    });
    expect(initN1WebGpu.type).toBe('init-result');
    let n1Sha = '';
    if (initN1WebGpu.type === 'init-result') {
      n1Sha = initN1WebGpu.sessionModelSha256 ?? '';
      expect(n1Sha).not.toBe(HITNET_M1_FULL_GRAPH_SHA256);
      expect(n1Sha).not.toBe(HITNET_H1_FULL_GRAPH_SHA256);
      expect(initN1WebGpu.slot0ReshapeToTransposeStats).toMatchObject({
        normalizedCount: 2,
        normalizedNodeNames: ['Transpose__908', 'Transpose__1006'],
        verifiedM1Provenance: true,
        verifiedRawIdentity: true,
        skippedInvalidOrUnprovenCount: 0,
      });
      expect(initN1WebGpu.sessionOptionsApplied?.slot0ReshapeToTransposeNormalized).toBe(true);
    }

    // 4) Explicit WASM without allowDiagnosticTransformedWasm leaves rawBytes untouched;
    //    diagnostic transformed WASM receives EXACT N1 bytes matching WebGPU.
    const initExplicitWasm = await sendMsg({
      type: 'init-onnx',
      id: 504,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'wasm',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizePropGatherNdRank: true,
      normalizeInitToPropZeroSlant: true,
      normalizeSlot0ReshapeToTranspose: true,
      allowDiagnosticTransformedWasm: false,
    });
    expect(initExplicitWasm.type).toBe('init-result');
    if (initExplicitWasm.type === 'init-result') {
      expect(initExplicitWasm.sessionModelSha256).toBe(rawSha);
      expect(initExplicitWasm.slot0ReshapeToTransposeStats?.normalizedCount).toBe(0);
    }

    const initDiagWasmN1 = await sendMsg({
      type: 'init-onnx',
      id: 505,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'wasm',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizePropGatherNdRank: true,
      normalizeInitToPropZeroSlant: true,
      normalizeSlot0ReshapeToTranspose: true,
      allowDiagnosticTransformedWasm: true,
    });
    expect(initDiagWasmN1.type).toBe('init-result');
    if (initDiagWasmN1.type === 'init-result') {
      expect(initDiagWasmN1.sessionModelSha256).toBe(n1Sha);
      expect(initDiagWasmN1.sessionOptionsApplied?.slot0ReshapeToTransposeNormalized).toBe(true);
    }

    // 5) Auto mode runtime fallback still uses exact raw untouched WASM bytes
    await sendMsg({
      type: 'init-onnx',
      id: 506,
      profile: STEREO_MODEL_PROFILES['hitnet-sf'],
      onnxBuffer: rawBytes.slice().buffer,
      wasmPrimaryUrl: 'https://example.test/ort/',
      wasmFallbackUrl: null,
      backendPreference: 'auto',
      normalizeSliceAxes: 'compact-active',
      normalizeGatherNdRank: true,
      normalizePropGatherNdRank: true,
      normalizeInitToPropZeroSlant: true,
      normalizeSlot0ReshapeToTranspose: true,
      strictWebGpu: false,
    });
    const rgba = new Uint8Array(320 * 240 * 4).fill(90);
    const estFallback = await sendMsg({
      type: 'estimate',
      id: 507,
      leftRgbaBuffer: rgba.slice().buffer,
      rightRgbaBuffer: rgba.slice().buffer,
      width: 320,
      height: 240,
      sourceEyeWidth: 480,
      sourceEyeHeight: 270,
      options: {},
    });
    expect(estFallback.type).toBe('estimate-result');
    const lastCall = calls[calls.length - 1];
    expect(lastCall.delegate).toBe('WASM');
    expect(lastCall.sha256).toBe(rawSha);
    expect(Buffer.compare(lastCall.bytes, rawBytes)).toBe(0);
    expect(
      normalizeHitnetSlot0ReshapeToTransposeForWebGpu(lastCall.bytes, {
        variantId: 'hitnet-sf',
        backendPreference: 'webgpu',
      }).normalizedCount
    ).toBe(0);

    await sendMsg({ type: 'dispose', id: 508 });
  }, 30000);
});


