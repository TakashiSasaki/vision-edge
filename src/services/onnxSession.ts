import * as ort from 'onnxruntime-web';
import type { OnnxBackendInfo, RunnableModelType } from '../types/vision';
import {
    recordResolvedSource,
    resolveOrtWasmUrls,
    type AssetDeliverySource,
    type AssetSourceRecorder,
} from './modelAssetResolver';
export class OnnxInferenceError extends Error {
    constructor(public readonly model: string, public readonly stage: 'input' | 'run' | 'output' | 'closed', cause: unknown) {
        super(`${model} ${stage}: ${cause instanceof Error ? cause.message : String(cause)}`, { cause });
        this.name = 'OnnxInferenceError';
    }
}
export const unknownOnnxBackend = (): OnnxBackendInfo => ({ runtime: 'onnxruntime-web', requestedProvider: 'unreported', configuredProviders: [], fallbackReason: 'factory configuration not reported', actualBackend: 'unmeasured' });
// A lease covers preprocessing, run, output reads and tensor cleanup. Closing
// denies new leases and releases exactly once after every current lease ends.
export class OnnxSessionOwner {
    closed = false;
    private active = 0;
    private releaseStarted = false;
    private done: Promise<void> | null = null;
    private resolve!: () => void;
    constructor(private readonly session: Pick<ort.InferenceSession, 'release'>, private readonly model: string) { }
    async use<T>(work: () => Promise<T>): Promise<T> {
        if (this.closed)
            throw new OnnxInferenceError(this.model, 'closed', 'task is closed');
        this.active++;
        try {
            return await work();
        }
        finally {
            this.active--;
            if (this.closed)
                this.releaseWhenIdle();
        }
    }
    close(): Promise<void> {
        if (this.done)
            return this.done;
        this.closed = true;
        this.done = new Promise<void>(resolve => { this.resolve = resolve; });
        this.releaseWhenIdle();
        return this.done;
    }
    private releaseWhenIdle() {
        if (this.active || this.releaseStarted)
            return;
        this.releaseStarted = true;
        const failed = (cause: unknown) => { console.warn(`${this.model} session release failed`, cause); this.resolve(); };
        try {
            Promise.resolve(this.session.release()).then(() => this.resolve(), failed);
        }
        catch (cause) {
            failed(cause);
        }
    }
}
export function disposeTensors(values: Iterable<ort.Tensor>) { for (const t of new Set(values))
    try {
        t.dispose?.();
    }
    catch (cause) {
        console.warn('ONNX tensor cleanup failed', cause);
    } }

export interface OrtSessionResolutionOptions {
    recordAssetSource?: AssetSourceRecorder;
    fallbackModelUrl?: string | null;
    modelId?: Extract<RunnableModelType, 'yolo-seg' | 'segformer'>;
    primarySource?: AssetDeliverySource;
    variantId?: string;
}

async function createOrtSessionWithUrl(
    targetUrl: string,
    delegate?: 'GPU' | 'WebGPU' | 'WASM'
): Promise<{ session: ort.InferenceSession; backend: OnnxBackendInfo }> {
    const requestedProvider = delegate === 'WASM' ? 'wasm' : delegate ? 'webgpu' : 'auto';
    const available = typeof navigator !== 'undefined' && !!navigator.gpu;
    const providers = delegate === 'WASM' || !available ? ['wasm'] : ['webgpu', 'wasm'];
    const backend: OnnxBackendInfo = { runtime: 'onnxruntime-web', requestedProvider, configuredProviders: providers, fallbackReason: delegate !== 'WASM' && !available ? 'WebGPU API unavailable; WASM configured' : null, actualBackend: 'unmeasured' };
    const create = (executionProviders: string[]) => ort.InferenceSession.create(targetUrl, { executionProviders, graphOptimizationLevel: 'all', logSeverityLevel: 3 });
    try {
        return { session: await create(providers), backend };
    }
    catch (cause) {
        if (!providers.includes('webgpu'))
            throw new Error(`ONNX session initialization failed: ${String(cause)}`, { cause });
        const fallback = { ...backend, configuredProviders: ['wasm'], fallbackReason: `WebGPU session creation failed: ${String(cause)}` };
        try {
            return { session: await create(['wasm']), backend: fallback };
        }
        catch (error) {
            throw new Error(`ONNX WASM fallback initialization failed: ${String(error)}`, { cause: error });
        }
    }
}

export async function createOrtSession(
    modelUrl: string,
    delegate?: 'GPU' | 'WebGPU' | 'WASM',
    resolution?: OrtSessionResolutionOptions
) {
    const recordAssetSource = resolution?.recordAssetSource ?? recordResolvedSource;
    const wasm = await resolveOrtWasmUrls();
    if (typeof window !== 'undefined') {
        ort.env.logLevel = 'error';
        const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
        ort.env.wasm.numThreads = isolated ? Math.min(4, Math.max(1, (navigator.hardwareConcurrency || 2) - 1)) : 1;
        ort.env.wasm.wasmPaths = wasm.primaryUrl;
    }
    try {
        const created = await createOrtSessionWithUrl(modelUrl, delegate);
        recordAssetSource('wasm-ort', wasm.source);
        if (resolution?.modelId) {
            recordAssetSource(resolution.modelId, resolution.primarySource ?? 'external');
            if (resolution.variantId) recordAssetSource(`${resolution.modelId}:${resolution.variantId}`, resolution.primarySource ?? 'external');
        }
        return created;
    }
    catch (primaryError) {
        if (resolution?.fallbackModelUrl && resolution.fallbackModelUrl !== modelUrl) {
            if (typeof window !== 'undefined' && wasm.fallbackUrl) {
                ort.env.wasm.wasmPaths = wasm.fallbackUrl;
            }
            const created = await createOrtSessionWithUrl(resolution.fallbackModelUrl, delegate);
            recordAssetSource('wasm-ort', wasm.fallbackUrl ? 'external-fallback' : wasm.source);
            if (resolution.modelId) {
                recordAssetSource(resolution.modelId, 'external-fallback');
                if (resolution.variantId) recordAssetSource(`${resolution.modelId}:${resolution.variantId}`, 'external-fallback');
            }
            return created;
        }
        throw primaryError;
    }
}
