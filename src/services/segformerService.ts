import * as ort from 'onnxruntime-web';
import type { DetectedObject, InferenceInput } from '../types/vision';
import { ADE20K_CLASSES, getADE20KInfo } from './ade20kDictionary';

export interface SegFormerTaskOptions {
  modelUrl?: string;
  delegate?: 'GPU' | 'WebGPU' | 'WASM';
}

export interface SegFormerTask {
  detectForVideo: (input: InferenceInput, confidenceThreshold: number) => Promise<DetectedObject[]>;
  getPixelClass?: (normX: number, normY: number) => { id: number; en: string; ja: string; category: string; color: string } | null;
  close: () => void;
}

const DEFAULT_SEGFORMER_URL = 'https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx';
const TARGET_SIZE = 512;
const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
const IMAGENET_STD = [0.229, 0.224, 0.225] as const;

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.replace('#', '');
  if (clean.length === 6) {
    return [
      parseInt(clean.substring(0, 2), 16),
      parseInt(clean.substring(2, 4), 16),
      parseInt(clean.substring(4, 6), 16),
    ];
  }
  return [148, 163, 184];
}

// Pre-compute RGB lookup for 150 classes
const CLASS_RGB_TABLE: [number, number, number][] = ADE20K_CLASSES.map(cls => hexToRgb(cls.color));

export class SegFormerEngine implements SegFormerTask {
  private inputCanvas: HTMLCanvasElement | null = null;
  private inputCtx: CanvasRenderingContext2D | null = null;
  private outputMaskCanvas: HTMLCanvasElement | null = null;
  private latestClassGrid: Uint8Array | null = null;
  private gridWidth = 128;
  private gridHeight = 128;
  private isClosed = false;

  constructor(private readonly session: ort.InferenceSession) {}

  public close(): void {
    this.isClosed = true;
    try {
      void this.session.release();
    } catch (cause) {
      console.warn('SegFormer session release error', cause);
    }
    this.inputCanvas = null;
    this.inputCtx = null;
    this.outputMaskCanvas = null;
    this.latestClassGrid = null;
  }

  public getPixelClass(normX: number, normY: number): { id: number; en: string; ja: string; category: string; color: string } | null {
    if (!this.latestClassGrid) return null;
    const px = Math.min(this.gridWidth - 1, Math.max(0, Math.floor(normX * this.gridWidth)));
    const py = Math.min(this.gridHeight - 1, Math.max(0, Math.floor(normY * this.gridHeight)));
    const classId = this.latestClassGrid[py * this.gridWidth + px];
    return getADE20KInfo(classId);
  }

  public async detectForVideo(
    input: InferenceInput,
    confidenceThreshold: number
  ): Promise<DetectedObject[]> {
    if (this.isClosed) return [];

    // 1. Prepare 512x512 input canvas
    if (!this.inputCanvas) {
      this.inputCanvas = document.createElement('canvas');
      this.inputCanvas.width = TARGET_SIZE;
      this.inputCanvas.height = TARGET_SIZE;
      this.inputCtx = this.inputCanvas.getContext('2d', { willReadFrequently: true });
    }
    const ctx = this.inputCtx;
    if (!ctx) return [];

    ctx.drawImage(input, 0, 0, TARGET_SIZE, TARGET_SIZE);
    const imgData = ctx.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE);
    const { data } = imgData;

    // 2. Preprocess with ImageNet mean/std normalization into planar [1, 3, 512, 512]
    const pixels = TARGET_SIZE * TARGET_SIZE;
    const float32Data = new Float32Array(3 * pixels);
    for (let i = 0; i < pixels; i++) {
      const offset = i * 4;
      const r = data[offset] / 255.0;
      const g = data[offset + 1] / 255.0;
      const b = data[offset + 2] / 255.0;

      float32Data[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      float32Data[pixels + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      float32Data[2 * pixels + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }

    const inputName = this.session.inputNames[0] || 'pixel_values';
    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, TARGET_SIZE, TARGET_SIZE]);

    // 3. Run inference
    let results: Record<string, ort.Tensor>;
    try {
      results = await this.session.run({ [inputName]: inputTensor });
    } catch (err) {
      console.error('[SegFormer] session.run failed:', err);
      return [];
    }
    if (this.isClosed) return [];

    // 4. Output logits tensor: [1, 150, 128, 128]
    const outputName = this.session.outputNames[0] || 'logits';
    const logitsTensor = results[outputName] || Object.values(results)[0];
    if (!logitsTensor) return [];

    const dims = logitsTensor.dims; // [1, 150, 128, 128]
    const numClasses = dims[1] || 150;
    const outH = dims[2] || 128;
    const outW = dims[3] || 128;
    this.gridWidth = outW;
    this.gridHeight = outH;

    const logitsData = logitsTensor.data as Float32Array;
    const gridPixels = outW * outH;

    if (!this.outputMaskCanvas) {
      this.outputMaskCanvas = document.createElement('canvas');
      this.outputMaskCanvas.width = outW;
      this.outputMaskCanvas.height = outH;
    }
    const maskCtx = this.outputMaskCanvas.getContext('2d');
    if (!maskCtx) return [];

    const maskImgData = maskCtx.createImageData(outW, outH);
    const mPixels = maskImgData.data;

    // Track pixel count and bounding boxes per class for HUD / inspector
    const classPixelCounts = new Uint32Array(numClasses);
    const classMinX = new Int32Array(numClasses).fill(outW);
    const classMaxX = new Int32Array(numClasses).fill(-1);
    const classMinY = new Int32Array(numClasses).fill(outH);
    const classMaxY = new Int32Array(numClasses).fill(-1);

    const classGrid = new Uint8Array(gridPixels);
    this.latestClassGrid = classGrid;

    // Fast Argmax across 150 channels
    for (let p = 0; p < gridPixels; p++) {
      let maxLogit = -Infinity;
      let bestClass = 0;

      for (let c = 0; c < numClasses; c++) {
        const logit = logitsData[c * gridPixels + p];
        if (logit > maxLogit) {
          maxLogit = logit;
          bestClass = c;
        }
      }

      classGrid[p] = bestClass;
      classPixelCounts[bestClass]++;

      const px = p % outW;
      const py = Math.floor(p / outW);
      if (px < classMinX[bestClass]) classMinX[bestClass] = px;
      if (px > classMaxX[bestClass]) classMaxX[bestClass] = px;
      if (py < classMinY[bestClass]) classMinY[bestClass] = py;
      if (py > classMaxY[bestClass]) classMaxY[bestClass] = py;

      const rgb = CLASS_RGB_TABLE[bestClass] || [148, 163, 184];
      const pOffset = p * 4;
      mPixels[pOffset] = rgb[0];
      mPixels[pOffset + 1] = rgb[1];
      mPixels[pOffset + 2] = rgb[2];
      // Semi-transparent overlay mask alpha
      mPixels[pOffset + 3] = 200;
    }

    maskCtx.putImageData(maskImgData, 0, 0);

    // 5. Build DetectedObject list for prominent classes
    const detectedObjects: DetectedObject[] = [];
    const minPixelCount = Math.max(40, Math.floor(gridPixels * 0.005)); // 0.5% area threshold

    // Collect prominent classes sorted by area
    const activeClasses: number[] = [];
    for (let c = 0; c < numClasses; c++) {
      if (classPixelCounts[c] >= minPixelCount) {
        activeClasses.push(c);
      }
    }
    activeClasses.sort((a, b) => classPixelCounts[b] - classPixelCounts[a]);

    for (let i = 0; i < activeClasses.length; i++) {
      const classId = activeClasses[i];
      const info = getADE20KInfo(classId);
      const pixelCount = classPixelCounts[classId];
      const confidence = Math.min(0.99, 0.70 + (pixelCount / gridPixels) * 0.29);

      if (confidence < confidenceThreshold && i > 0) continue;

      const bx = classMinX[classId] / outW;
      const by = classMinY[classId] / outH;
      const bw = Math.max(0.04, (classMaxX[classId] - classMinX[classId] + 1) / outW);
      const bh = Math.max(0.04, (classMaxY[classId] - classMinY[classId] + 1) / outH);

      detectedObjects.push({
        id: `segformer-${classId}`,
        label: info.en,
        labelJa: info.ja,
        category: info.category,
        confidence,
        bbox: { x: bx, y: by, width: bw, height: bh },
        color: info.color,
        // The first prominent object carries the full composite semantic mask
        maskCanvas: i === 0 ? this.outputMaskCanvas : undefined,
      });
    }

    // Ensure the full semantic mask is always attached to at least one object
    if (detectedObjects.length > 0 && !detectedObjects[0].maskCanvas) {
      detectedObjects[0].maskCanvas = this.outputMaskCanvas;
    }

    if (detectedObjects.length > 0) {
      console.log(`[SegFormer] Frame classified ${detectedObjects.length} scene segment(s):`,
        detectedObjects.map(o => `${o.labelJa} (${Math.round((classPixelCounts[parseInt(o.id.replace('segformer-', ''), 10)] / gridPixels) * 100)}%)`).join(', '));
    }

    return detectedObjects;
  }
}

export async function createSegFormerTask(options: SegFormerTaskOptions = {}): Promise<SegFormerTask> {
  const modelUrl = options.modelUrl || DEFAULT_SEGFORMER_URL;

  if (typeof window !== 'undefined') {
    ort.env.logLevel = 'error';
    const isIsolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
    ort.env.wasm.numThreads = isIsolated ? Math.min(4, Math.max(1, (navigator.hardwareConcurrency || 2) - 1)) : 1;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
  }

  const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;
  const executionProviders = options.delegate === 'WASM'
    ? ['wasm']
    : hasWebGPU
    ? ['webgpu', 'wasm']
    : ['wasm'];

  const sessionOptions: ort.InferenceSession.SessionOptions = {
    executionProviders,
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3,
  };

  try {
    const session = await ort.InferenceSession.create(modelUrl, sessionOptions);
    return new SegFormerEngine(session);
  } catch (error) {
    if (executionProviders.includes('webgpu')) {
      try {
        const fallbackSession = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
          logSeverityLevel: 3,
        });
        return new SegFormerEngine(fallbackSession);
      } catch (fallbackError) {
        throw new Error(`Failed to create SegFormer session with WASM fallback: ${String(fallbackError)}`, {
          cause: fallbackError,
        });
      }
    }
    throw new Error(`Failed to initialize SegFormer model: ${String(error)}`, { cause: error });
  }
}
