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
  disparityToDepthMeters,
  exceedsWebGpuStorageBufferLimit,
  extractDisparityFromOrtTensor,
  getColormapLut,
  getLocalStereoOnnxInfo,
  inspectDisparityMapAtPoint,
  inspectStereoOnnxWebGpuCompatibility,
  isWebGpuRuntimeKernelError,
  registerLocalStereoOnnxBytes,
  resolveStereoOrtDelegate,
  resolveStereoOrtSessionOptions,
  splitLargeOnnxConcatNodes,
  splitSideBySideFrame,
  STEREO_MODEL_PROFILES,
  StereoOnnxEngine,
  StereoSadEngine,
  stripOnnxGraphValueInfo,
  type StereoWorkerLike,
  type StereoWorkerRequest,
  type StereoWorkerResponse,
} from './stereoDepthService';
import * as onnxSessionModule from './onnxSession';
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

    // 1. MobileStereoNet-2D downloads via same-origin stateless CORS proxy (Google Drive blocks cross-site browser Sec-Fetch-Mode: cors) and runs on WASM (5D Conv3d cost aggregation)
    const msTask = await createStereoDepthTask({
      variantId: 'mobilestereonet-2d',
      fetchImpl: mockFetch,
    });
    expect(mockFetch.mock.calls[0][0]).toBe(
      STEREO_MODEL_PROFILES['mobilestereonet-2d'].corsProxyUrl
    );
    expect(createSpy.mock.calls[0][1]).toBe('WASM');
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

    const conv3dProto = makeConvProto([16, 1, 8, 3, 3], 'Conv_16834');
    expect(inspectStereoOnnxWebGpuCompatibility(conv3dProto)).toMatchObject({
      compatible: false,
      hasConv3d: true,
      conv3dNodeName: 'Conv_16834',
    });
    expect(resolveStereoOrtDelegate(STEREO_MODEL_PROFILES['custom-onnx'], conv3dProto)).toBe('WASM');

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

  it('disables ORT graph constant folding on stereo WASM sessions to prevent multi-GB OOM crashes', () => {
    expect(resolveStereoOrtSessionOptions(['wasm'])).toEqual({
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      logSeverityLevel: 3,
    });
    expect(resolveStereoOrtSessionOptions(['webgpu', 'wasm'])).toEqual({
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all',
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
    expect(resolveStereoOrtDelegate(STEREO_MODEL_PROFILES['mobilestereonet-2d'], tinyOnnx, 8, 'auto')).toBe('WASM');
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
});
