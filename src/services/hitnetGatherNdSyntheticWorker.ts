import * as ort from 'onnxruntime-web';

export interface SyntheticWorkerFeedSpec {
  name: string;
  dtype: 'float32' | 'int64';
  dims: number[];
  buffer: ArrayBuffer;
}

export interface SyntheticWorkerRunRequest {
  id: number;
  modelBuffer: ArrayBuffer;
  wasmPaths: string;
  executionProviders: Array<'webgpu' | 'wasm'>;
  graphOptimizationLevel: 'disabled' | 'basic' | 'extended' | 'all';
  disableCpuEpFallback?: boolean;
  captureGpuValidationErrors?: boolean;
  feeds: SyntheticWorkerFeedSpec[];
}

export interface SyntheticWorkerRunOutput {
  name: string;
  dims: number[];
  dataType?: string;
  buffer: ArrayBuffer;
}

export type SyntheticWorkerRunResponse =
  | {
      id: number;
      ok: true;
      createMs: number;
      runMs: number;
      ortVersions: { common?: string; web?: string } | null;
      workerGpuAdapter: string | null;
      gpuValidationErrors?: string[];
      outputs: SyntheticWorkerRunOutput[];
    }
  | {
      id: number;
      ok: false;
      stage: 'create' | 'run';
      message: string;
      ortVersions: { common?: string; web?: string } | null;
      workerGpuAdapter: string | null;
      gpuValidationErrors?: string[];
    };

async function describeWorkerAdapter(): Promise<string | null> {
  try {
    const ortEnv = ort.env as unknown as {
      webgpu?: {
        adapter?: {
          info?: { vendor?: string; architecture?: string; device?: string; description?: string };
          features?: { has?: (name: string) => boolean };
        };
      };
    };
    let info = ortEnv.webgpu?.adapter?.info;
    let features = ortEnv.webgpu?.adapter?.features;
    if ((!info || !features) && typeof navigator !== 'undefined') {
      const gpu = (navigator as unknown as {
        gpu?: {
          requestAdapter?: () => Promise<{
            info?: { vendor?: string; architecture?: string; device?: string; description?: string };
            features?: { has?: (name: string) => boolean };
          } | null>;
        };
      }).gpu;
      const ad = await gpu?.requestAdapter?.();
      info = info ?? ad?.info;
      features = features ?? ad?.features;
    }
    if (!info) return null;
    const hasF16 = features?.has?.('shader-f16') ?? false;
    const hasTq = features?.has?.('timestamp-query') ?? false;
    return `vendor=${info.vendor || '-'} arch=${info.architecture || '-'} device=${info.device || '-'} desc=${info.description || '-'} shader-f16=${hasF16} timestamp-query=${hasTq}`;
  } catch {
    return null;
  }
}

function copyOrtTensorBuffer(tensor: ort.Tensor): ArrayBuffer {
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
  return new Float32Array(Array.from(data as Iterable<unknown>, (v) => Number(v))).buffer as ArrayBuffer;
}

const workerScope =
  typeof self !== 'undefined' &&
  typeof (self as unknown as { postMessage?: unknown }).postMessage === 'function' &&
  typeof window === 'undefined'
    ? (self as unknown as {
        onmessage: ((ev: MessageEvent<SyntheticWorkerRunRequest>) => void) | null;
        postMessage: (msg: SyntheticWorkerRunResponse, transfer?: Transferable[]) => void;
      })
    : null;

if (workerScope) {
  workerScope.onmessage = async (event: MessageEvent<SyntheticWorkerRunRequest>) => {
    const req = event.data;
    if (!req) return;
    let stage: 'create' | 'run' = 'create';
    let session: ort.InferenceSession | null = null;
    const createdTensors: ort.Tensor[] = [];
    const gpuValidationErrors: string[] = [];
    const ortVersions =
      (ort.env as unknown as { versions?: { common?: string; web?: string } }).versions ?? null;

    try {
      ort.env.logLevel = 'error';
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      if (req.wasmPaths) {
        ort.env.wasm.wasmPaths = req.wasmPaths;
      }

      const opts: ort.InferenceSession.SessionOptions = {
        executionProviders: req.executionProviders,
        graphOptimizationLevel: req.graphOptimizationLevel,
        enableCpuMemArena: !req.executionProviders.every((e) => e === 'wasm'),
        enableMemPattern: !req.executionProviders.every((e) => e === 'wasm'),
        logSeverityLevel: 3,
      };
      if (req.disableCpuEpFallback && req.executionProviders.includes('webgpu')) {
        opts.extra = { session: { disable_cpu_ep_fallback: '1' } };
      }

      const tC0 = performance.now();
      session = await ort.InferenceSession.create(new Uint8Array(req.modelBuffer), opts);
      const createMs = performance.now() - tC0;

      stage = 'run';
      const feeds: Record<string, ort.Tensor> = {};
      for (const spec of req.feeds) {
        const typedData =
          spec.dtype === 'int64'
            ? new BigInt64Array(spec.buffer)
            : new Float32Array(spec.buffer);
        const t = new ort.Tensor(spec.dtype, typedData, spec.dims);
        feeds[spec.name] = t;
        createdTensors.push(t);
      }

      const captureValidation =
        (req.captureGpuValidationErrors ?? true) && req.executionProviders.includes('webgpu');
      const gpuDevice = captureValidation
        ? (ort.env as unknown as {
            webgpu?: {
              device?: {
                pushErrorScope?: (filter: 'validation' | 'out-of-memory' | 'internal') => void;
                popErrorScope?: () => Promise<{ message?: string } | null>;
                addEventListener?: (type: string, cb: (ev: { error?: { message?: string } }) => void) => void;
                removeEventListener?: (type: string, cb: (ev: { error?: { message?: string } }) => void) => void;
              };
            };
          }).webgpu?.device
        : undefined;
      const onUncaptured = (ev: { error?: { message?: string } }) => {
        const m = ev?.error?.message;
        if (m) gpuValidationErrors.push(`uncaptured: ${m}`);
      };
      let pushedErrorScope = false;
      if (gpuDevice) {
        try {
          gpuDevice.addEventListener?.('uncapturederror', onUncaptured);
          gpuDevice.pushErrorScope?.('validation');
          pushedErrorScope = typeof gpuDevice.popErrorScope === 'function';
        } catch {
          pushedErrorScope = false;
        }
      }

      let tR0 = 0;
      let runMs = 0;
      let res: Record<string, ort.Tensor>;
      try {
        tR0 = performance.now();
        res = await session.run(feeds);
        runMs = performance.now() - tR0;
      } finally {
        if (pushedErrorScope && gpuDevice?.popErrorScope) {
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
            // Ignore cleanup error
          }
        }
      }

      const outputs: SyntheticWorkerRunOutput[] = [];
      const transfers: Transferable[] = [];
      for (const [name, tensor] of Object.entries(res)) {
        createdTensors.push(tensor);
        const buf = copyOrtTensorBuffer(tensor);
        outputs.push({
          name,
          dims: Array.from(tensor.dims),
          dataType: tensor.type,
          buffer: buf,
        });
        transfers.push(buf);
      }

      const workerGpuAdapter = req.executionProviders.includes('webgpu')
        ? await describeWorkerAdapter()
        : null;

      workerScope.postMessage(
        {
          id: req.id,
          ok: true,
          createMs,
          runMs,
          ortVersions,
          workerGpuAdapter,
          ...(captureValidation ? { gpuValidationErrors } : {}),
          outputs,
        },
        transfers
      );
    } catch (err) {
      const workerGpuAdapter = req.executionProviders.includes('webgpu')
        ? await describeWorkerAdapter()
        : null;
      workerScope.postMessage({
        id: req.id,
        ok: false,
        stage,
        message: err instanceof Error ? err.message : String(err),
        ortVersions,
        workerGpuAdapter,
        ...(gpuValidationErrors.length > 0 ? { gpuValidationErrors } : {}),
      });
    } finally {
      for (const t of createdTensors) {
        try {
          t.dispose?.();
        } catch {
          // Ignore dispose error
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
  };
}
