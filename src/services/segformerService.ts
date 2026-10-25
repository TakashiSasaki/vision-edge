import * as ort from 'onnxruntime-web';
import type { BoundingBox, DetectedObject, InferenceInput, OnnxBackendInfo } from '../types/vision';
import { ADE20K_CLASSES, getADE20KInfo } from './ade20kDictionary';
import { OnnxSessionOwner, OnnxInferenceError, createOrtSession, disposeTensors, unknownOnnxBackend } from './onnxSession';
export interface SegFormerTaskOptions {
    modelUrl?: string;
    delegate?: 'GPU' | 'WebGPU' | 'WASM';
}
export interface SemanticPixel {
    id: number;
    en: string;
    ja: string;
    category: string;
    color: string;
    confidence?: number;
    areaFraction?: number;
    bbox?: BoundingBox;
}
export interface SegFormerTask {
    detectForVideo: (input: InferenceInput, confidenceThreshold: number) => Promise<DetectedObject[]>;
    getPixelClass?: (x: number, y: number) => SemanticPixel | null;
    acceptResult?: (objects: DetectedObject[]) => {
        maskCanvas: HTMLCanvasElement;
    } | null;
    clearResult?: () => void;
    close: () => void | Promise<void>;
    readonly backend?: OnnxBackendInfo;
}
const DEFAULT_SEGFORMER_URL = 'https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx';
const TARGET_SIZE = 512;
const MEAN = [.485, .456, .406], STD = [.229, .224, .225];
const RGB = ADE20K_CLASSES.map(c => [parseInt(c.color.slice(1, 3), 16), parseInt(c.color.slice(3, 5), 16), parseInt(c.color.slice(5, 7), 16)]);
interface Snapshot {
    maskCanvas: HTMLCanvasElement;
    grid: Uint8Array;
    scores: Float32Array;
    width: number;
    height: number;
    counts: Uint32Array;
    boxes: BoundingBox[];
}
export class SegFormerEngine implements SegFormerTask {
    private inputCanvas: HTMLCanvasElement | null = null;
    private inputCtx: CanvasRenderingContext2D | null = null;
    private readonly owner: OnnxSessionOwner;
    private readonly snapshots = new WeakMap<DetectedObject[], Snapshot>();
    private accepted: Snapshot | null = null;
    constructor(private readonly session: ort.InferenceSession, public readonly backend: OnnxBackendInfo = unknownOnnxBackend()) { this.owner = new OnnxSessionOwner(session, 'SegFormer'); }
    close(): Promise<void> { this.clearResult(); this.inputCanvas = null; this.inputCtx = null; return this.owner.close(); }
    clearResult = () => { this.accepted = null; };
    // Only the engine's generation-checked acceptance publishes an inspector grid.
    acceptResult(objects: DetectedObject[]) { this.accepted = this.owner.closed ? null : this.snapshots.get(objects) ?? null; return this.accepted; }
    getPixelClass(x: number, y: number): SemanticPixel | null {
        const s = this.accepted;
        if (!s || this.owner.closed || !Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0 || x > 1 || y > 1)
            return null;
        const px = Math.min(s.width - 1, Math.floor(x * s.width)), py = Math.min(s.height - 1, Math.floor(y * s.height));
        const i = py * s.width + px, id = s.grid[i];
        return { ...getADE20KInfo(id), confidence: s.scores[i], areaFraction: s.counts[id] / s.grid.length, bbox: s.boxes[id] };
    }
    detectForVideo(input: InferenceInput, threshold: number): Promise<DetectedObject[]> {
        return this.owner.use(async () => {
            if (!this.inputCanvas) {
                this.inputCanvas = document.createElement('canvas');
                this.inputCanvas.width = TARGET_SIZE;
                this.inputCanvas.height = TARGET_SIZE;
                this.inputCtx = this.inputCanvas.getContext('2d', { willReadFrequently: true });
            }
            const ctx = this.inputCtx;
            if (!ctx)
                throw new OnnxInferenceError('SegFormer', 'input', '2D context unavailable');
            ctx.drawImage(input, 0, 0, TARGET_SIZE, TARGET_SIZE);
            const rgba = ctx.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE).data;
            const n = TARGET_SIZE * TARGET_SIZE, data = new Float32Array(3 * n);
            for (let i = 0; i < n; i++)
                for (let c = 0; c < 3; c++)
                    data[c * n + i] = (rgba[i * 4 + c] / 255 - MEAN[c]) / STD[c];
            const tensor = new ort.Tensor('float32', data, [1, 3, TARGET_SIZE, TARGET_SIZE]);
            const owned: ort.Tensor[] = [tensor];
            try {
                let result: Record<string, ort.Tensor>;
                try {
                    result = await this.session.run({ [this.session.inputNames[0] || 'pixel_values']: tensor });
                }
                catch (cause) {
                    throw new OnnxInferenceError('SegFormer', 'run', cause);
                }
                owned.push(...Object.values(result));
                if (this.owner.closed)
                    return [];
                const logits = result[this.session.outputNames[0] || 'logits'] ?? Object.values(result)[0];
                if (!logits || logits.dims.length !== 4 || logits.dims[0] !== 1 || logits.dims[1] !== 150)
                    throw new OnnxInferenceError('SegFormer', 'output', 'expected [1,150,height,width] logits');
                const height = logits.dims[2], width = logits.dims[3], area = width * height, classes = 150;
                const values = logits.data as Float32Array;
                if (!Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0 || values.length !== classes * area || !values.every(Number.isFinite))
                    throw new OnnxInferenceError('SegFormer', 'output', 'invalid logits data/dimensions');
                const maskCanvas = document.createElement('canvas');
                maskCanvas.width = width;
                maskCanvas.height = height;
                const out = maskCanvas.getContext('2d');
                if (!out)
                    throw new OnnxInferenceError('SegFormer', 'output', 'mask context unavailable');
                const image = out.createImageData(width, height), grid = new Uint8Array(area), scores = new Float32Array(area), counts = new Uint32Array(classes), sums = new Float64Array(classes);
                const minX = new Int32Array(classes).fill(width), maxX = new Int32Array(classes).fill(-1), minY = new Int32Array(classes).fill(height), maxY = new Int32Array(classes).fill(-1);
                for (let p = 0; p < area; p++) {
                    let best = 0, max = -Infinity;
                    for (let c = 0; c < classes; c++)
                        if (values[c * area + p] > max) {
                            max = values[c * area + p];
                            best = c;
                        }
                    let denominator = 0;
                    for (let c = 0; c < classes; c++)
                        denominator += Math.exp(values[c * area + p] - max);
                    const probability = 1 / denominator;
                    grid[p] = best;
                    scores[p] = probability;
                    counts[best]++;
                    sums[best] += probability;
                    const x = p % width, y = Math.floor(p / width);
                    minX[best] = Math.min(minX[best], x);
                    maxX[best] = Math.max(maxX[best], x);
                    minY[best] = Math.min(minY[best], y);
                    maxY[best] = Math.max(maxY[best], y);
                    const rgb = RGB[best];
                    image.data.set([rgb[0], rgb[1], rgb[2], 200], p * 4);
                }
                out.putImageData(image, 0, 0);
                const boxes = Array.from({ length: classes }, (_, c) => counts[c] ? { x: minX[c] / width, y: minY[c] / height, width: (maxX[c] - minX[c] + 1) / width, height: (maxY[c] - minY[c] + 1) / height } : { x: 0, y: 0, width: 0, height: 0 });
                const objects: DetectedObject[] = [];
                const minArea = Math.max(40, Math.floor(area * .005));
                for (let c = 0; c < classes; c++) {
                    if (counts[c] < minArea)
                        continue;
                    const confidence = sums[c] / counts[c];
                    if (confidence < threshold)
                        continue;
                    const info = getADE20KInfo(c);
                    objects.push({ id: `segformer-${c}`, label: info.en, labelJa: info.ja, category: info.category, color: info.color, semanticClassId: c, confidence, confidenceKind: 'semantic-softmax-mean', areaFraction: counts[c] / area, bbox: boxes[c] });
                }
                objects.sort((a, b) => (b.areaFraction ?? 0) - (a.areaFraction ?? 0));
                // Snapshot mask/grid are per-result; no mutation of accepted prior frames.
                this.snapshots.set(objects, { maskCanvas, grid, scores, width, height, counts, boxes });
                return objects;
            }
            finally {
                disposeTensors(owned);
            }
        });
    }
}
export async function createSegFormerTask(options: SegFormerTaskOptions = {}): Promise<SegFormerTask> { const { session, backend } = await createOrtSession(options.modelUrl || DEFAULT_SEGFORMER_URL, options.delegate); return new SegFormerEngine(session, backend); }
