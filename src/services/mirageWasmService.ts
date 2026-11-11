import type {
  DetectedObject,
  InferenceInput,
  MirageAnalysisMode,
  MirageVariantId,
  OnnxBackendInfo,
} from '../types/vision';
import {
  LOCAL_CACTUS_WASM_BINARY_URL,
  LOCAL_CACTUS_WASM_JS_URL,
  resolveModelAssetUrls,
  type AssetSourceRecorder,
} from './modelAssetResolver';
import {
  mapMirageObservationToDetectedObjects,
  type MirageDescribeOptions,
  type MirageRawObservation,
} from './mirageVisionAdapter';
import type {
  MirageWorkerRequest,
  MirageWorkerResponse,
} from './mirageWasmWorker';
import { mirageWebGpuUnavailableError } from './mirageAvailability';

export interface MirageInferExtraOptions {
  iouThreshold?: number;
  maxDetections?: number;
  analysisMode?: MirageAnalysisMode;
  includeOcr?: boolean;
}

export interface MirageTask {
  readonly backend?: OnnxBackendInfo;
  detectForVideo(
    videoElement: InferenceInput,
    confidenceThreshold: number,
    extraOptions?: MirageInferExtraOptions
  ): Promise<DetectedObject[]>;
  close(): void;
}

export interface MirageWorkerLike {
  postMessage(message: MirageWorkerRequest, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<MirageWorkerResponse>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  terminate(): void;
}

export interface MirageTaskOptions {
  variantId?: MirageVariantId;
  recordAssetSource?: AssetSourceRecorder;
  workerFactory?: () => MirageWorkerLike;
}

const FAST_MODE_MAX_DIM = 320;
const FULL_MODE_MAX_DIM = 512;

export function buildMirageDescribeOptions(
  confidenceThreshold: number,
  extraOptions?: MirageInferExtraOptions
): { describeOptions: MirageDescribeOptions; maxDim: number; includeOcr: boolean } {
  const analysisMode: MirageAnalysisMode = extraOptions?.analysisMode ?? 'fast';
  const includeOcr = extraOptions?.includeOcr ?? true;
  const maxObjects = extraOptions?.maxDetections ?? 15;
  const iou = extraOptions?.iouThreshold ?? 0.45;
  const conf = Math.max(0.15, Math.min(0.95, confidenceThreshold));

  if (analysisMode === 'full') {
    const sections: MirageDescribeOptions['sections'] = includeOcr
      ? ['scene', 'objects', 'regions', 'text', 'relations']
      : ['scene', 'objects', 'regions', 'relations'];
    return {
      maxDim: FULL_MODE_MAX_DIM,
      includeOcr,
      describeOptions: {
        sections,
        fields: ['bbox', 'score', 'mask', 'position', 'description', 'count'],
        mask_format: 'polygons',
        conf,
        text_conf: Math.max(0.25, conf),
        iou,
        max_objects: maxObjects,
        max_text_lines: 12,
        max_tokens: 768,
      },
    };
  }

  const sections: MirageDescribeOptions['sections'] = includeOcr
    ? ['objects', 'text']
    : ['objects'];
  return {
    maxDim: FAST_MODE_MAX_DIM,
    includeOcr,
    describeOptions: {
      sections,
      fields: ['bbox', 'score', 'mask', 'position'],
      mask_format: 'polygons',
      conf,
      text_conf: Math.max(0.25, conf),
      iou,
      max_objects: maxObjects,
      max_text_lines: 8,
      max_tokens: 384,
    },
  };
}

function getInputDimensions(input: InferenceInput): { width: number; height: number } {
  if ('videoWidth' in input) {
    return {
      width: input.videoWidth || 640,
      height: input.videoHeight || 480,
    };
  }
  if ('displayWidth' in input) {
    return {
      width: input.displayWidth || 640,
      height: input.displayHeight || 480,
    };
  }
  return {
    width: input.width || 640,
    height: input.height || 480,
  };
}

function formatPositionTagJa(bbox: { x: number; y: number; width: number; height: number }): string {
  const centerX = bbox.x + bbox.width / 2;
  const dir = centerX < 0.35 ? '左' : centerX > 0.65 ? '右' : '中央';
  const area = bbox.width * bbox.height;
  const bottomY = bbox.y + bbox.height;
  const dist =
    area >= 0.16 || (bottomY >= 0.85 && area >= 0.08)
      ? '近'
      : area >= 0.04
        ? '中'
        : '遠';
  return `${dir}・${dist}`;
}

export class MirageWasmEngine implements MirageTask {
  public readonly backend: OnnxBackendInfo = {
    runtime: 'onnxruntime-web',
    requestedProvider: 'wasm',
    configuredProviders: ['wasm'],
    fallbackReason: null,
    actualBackend: 'unmeasured',
  };

  private readonly worker: MirageWorkerLike;
  private offscreenCanvas: HTMLCanvasElement | null = null;
  private offscreenCtx: CanvasRenderingContext2D | null = null;
  private nextRequestId = 1;
  private readonly pendingRequests = new Map<
    number,
    {
      resolve: (res: MirageWorkerResponse) => void;
      reject: (err: Error) => void;
    }
  >();
  private inFlight = 0;
  private closed = false;
  private terminated = false;

  constructor(worker: MirageWorkerLike) {
    this.worker = worker;
    this.worker.onmessage = (event: MessageEvent<MirageWorkerResponse>) => {
      const msg = event.data;
      if (!msg || typeof msg.id !== 'number') return;
      const pending = this.pendingRequests.get(msg.id);
      if (!pending) return;
      this.pendingRequests.delete(msg.id);
      if (!msg.ok) {
        pending.reject(new Error(msg.message || 'Mirage WASM Worker error'));
      } else {
        pending.resolve(msg);
      }
    };
    this.worker.onerror = (event: ErrorEvent) => {
      const err = new Error(event.message || 'Mirage WASM Worker fatal error');
      for (const [, pending] of this.pendingRequests) {
        pending.reject(err);
      }
      this.pendingRequests.clear();
    };
  }

  public async initializeWorker(
    jsUrl: string,
    wasmUrl: string,
    modelUrl: string,
    fallbackModelUrl: string | null
  ): Promise<'primary' | 'fallback'> {
    const id = this.nextRequestId++;
    const response = await new Promise<MirageWorkerResponse>((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject });
      this.worker.postMessage({
        type: 'init',
        id,
        jsUrl,
        wasmUrl,
        modelUrl,
        fallbackModelUrl,
      });
    });
    if (response.type !== 'init-result') {
      throw new Error('Unexpected worker initialization response');
    }
    return response.sourceUsed;
  }

  private extractPackedRgb(
    input: InferenceInput,
    maxDim: number
  ): { rgbBuffer: ArrayBuffer; width: number; height: number } {
    const { width: srcW, height: srcH } = getInputDimensions(input);
    const scale = Math.min(1, maxDim / Math.max(srcW, srcH, 1));
    const targetW = Math.max(32, Math.round(srcW * scale));
    const targetH = Math.max(32, Math.round(srcH * scale));

    if (!this.offscreenCanvas) {
      this.offscreenCanvas = document.createElement('canvas');
      this.offscreenCtx = this.offscreenCanvas.getContext('2d', {
        willReadFrequently: true,
      });
    }
    if (
      this.offscreenCanvas.width !== targetW ||
      this.offscreenCanvas.height !== targetH
    ) {
      this.offscreenCanvas.width = targetW;
      this.offscreenCanvas.height = targetH;
    }

    const ctx = this.offscreenCtx;
    if (!ctx) {
      throw new Error('Failed to acquire 2D canvas context for Mirage preprocessing');
    }

    ctx.drawImage(input as CanvasImageSource, 0, 0, targetW, targetH);
    const imageData = ctx.getImageData(0, 0, targetW, targetH);
    const rgba = imageData.data;
    const pixelCount = targetW * targetH;
    const rgb = new Uint8Array(pixelCount * 3);

    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
      rgb[j] = rgba[i];
      rgb[j + 1] = rgba[i + 1];
      rgb[j + 2] = rgba[i + 2];
    }

    return {
      rgbBuffer: rgb.buffer,
      width: targetW,
      height: targetH,
    };
  }

  public async detectForVideo(
    videoElement: InferenceInput,
    confidenceThreshold = 0.25,
    extraOptions?: MirageInferExtraOptions
  ): Promise<DetectedObject[]> {
    if (this.closed) return [];
    this.inFlight++;
    try {
      const { describeOptions, maxDim, includeOcr } = buildMirageDescribeOptions(
        confidenceThreshold,
        extraOptions
      );
      const { rgbBuffer, width, height } = this.extractPackedRgb(
        videoElement,
        maxDim
      );
      if (this.closed) return [];

      const id = this.nextRequestId++;
      const optionsJson = JSON.stringify(describeOptions);
      const response = await new Promise<MirageWorkerResponse>((resolve, reject) => {
        this.pendingRequests.set(id, { resolve, reject });
        this.worker.postMessage(
          {
            type: 'describe',
            id,
            rgbBuffer,
            width,
            height,
            optionsJson,
          },
          [rgbBuffer]
        );
      });

      if (this.closed || response.type !== 'describe-result') {
        return [];
      }

      return this.postprocessObservation(
        response.observation,
        confidenceThreshold,
        includeOcr,
        extraOptions?.maxDetections ?? 15
      );
    } finally {
      this.inFlight--;
      this.terminateIfIdle();
    }
  }

  private postprocessObservation(
    observation: MirageRawObservation,
    confidenceThreshold: number,
    includeOcr: boolean,
    maxDetections: number
  ): DetectedObject[] {
    const { objects } = mapMirageObservationToDetectedObjects(observation, {
      confidenceThreshold,
      textConfidenceThreshold: Math.max(0.25, confidenceThreshold),
      maxDetections,
      includeOcrAsDetections: includeOcr,
      includeRegionsAsDetections: true,
      bboxFormat: 'yxyx',
    });

    return objects.map((obj) => {
      const posJa = formatPositionTagJa(obj.bbox);
      const labelJaWithPosition = obj.labelJa.includes('（')
        ? obj.labelJa
        : `${obj.labelJa}（${posJa}）`;
      return {
        ...obj,
        labelJa: labelJaWithPosition,
      };
    });
  }

  private terminateIfIdle(): void {
    if (!this.closed || this.inFlight > 0 || this.terminated) return;
    this.terminated = true;
    for (const [, pending] of this.pendingRequests) {
      pending.reject(new Error('Mirage task closed'));
    }
    this.pendingRequests.clear();
    try {
      this.worker.terminate();
    } catch {
      // Ignore worker termination errors
    }
  }

  public close(): void {
    this.closed = true;
    this.offscreenCanvas = null;
    this.offscreenCtx = null;
    this.terminateIfIdle();
  }
}

function createDefaultBrowserWorker(): MirageWorkerLike {
  return new Worker(new URL('./mirageWasmWorker.ts', import.meta.url), {
    type: 'module',
  });
}

export async function createMirageWasmTask(
  options: MirageTaskOptions = {}
): Promise<MirageTask> {
  const variantId: MirageVariantId = options.variantId ?? 'mirage-51m';

  if (variantId === 'mirage-webgpu') {
    throw mirageWebGpuUnavailableError();
  }

  const asset = await resolveModelAssetUrls('mirage', variantId);
  const origin =
    typeof window !== 'undefined' && window.location?.origin
      ? window.location.origin
      : '';
  const resolveAbsolute = (url: string) =>
    url.startsWith('http://') || url.startsWith('https://')
      ? url
      : `${origin}${url}`;

  const jsUrl = resolveAbsolute(LOCAL_CACTUS_WASM_JS_URL);
  const wasmUrl = resolveAbsolute(LOCAL_CACTUS_WASM_BINARY_URL);
  const modelUrl = resolveAbsolute(asset.primaryUrl);
  const fallbackModelUrl = asset.fallbackUrl
    ? resolveAbsolute(asset.fallbackUrl)
    : null;

  const worker = options.workerFactory
    ? options.workerFactory()
    : createDefaultBrowserWorker();
  const engine = new MirageWasmEngine(worker);

  try {
    const sourceUsed = await engine.initializeWorker(
      jsUrl,
      wasmUrl,
      modelUrl,
      fallbackModelUrl
    );
    const deliverySource =
      sourceUsed === 'fallback' ? 'external-fallback' : asset.source;
    options.recordAssetSource?.('mirage', deliverySource);
    options.recordAssetSource?.(`mirage:${variantId}`, deliverySource);
    return engine;
  } catch (err) {
    engine.close();
    throw err;
  }
}
