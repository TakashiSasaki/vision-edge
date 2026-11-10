import type { MirageRawObservation } from './mirageVisionAdapter';

export interface CactusEmscriptenModule {
  HEAPU8: Uint8Array;
  _malloc(size: number): number;
  _free(ptr: number): void;
  _cactus_load(ptr: number, byteLength: bigint): number;
  _cactus_unload(): void;
  _cactus_models(): number;
  _cactus_describe(
    rgbPtr: number,
    width: number,
    height: number,
    optsPtr: number,
    langPtr: number,
    keywordsPtr: number,
    outPtr: number,
    outCap: number
  ): number;
  _cactus_error(): number;
  UTF8ToString(ptr: number, maxBytesToRead?: number): string;
}

export type MirageWorkerRequest =
  | {
      type: 'init';
      id: number;
      jsUrl: string;
      wasmUrl: string;
      modelUrl: string;
      fallbackModelUrl: string | null;
    }
  | {
      type: 'describe';
      id: number;
      rgbBuffer: ArrayBuffer;
      width: number;
      height: number;
      optionsJson: string;
    }
  | {
      type: 'dispose';
      id: number;
    };

export type MirageWorkerResponse =
  | {
      type: 'init-result';
      id: number;
      ok: true;
      sourceUsed: 'primary' | 'fallback';
    }
  | {
      type: 'describe-result';
      id: number;
      ok: true;
      observation: MirageRawObservation;
    }
  | {
      type: 'dispose-result';
      id: number;
      ok: true;
    }
  | {
      type: 'error';
      id: number;
      ok: false;
      stage: 'wasm' | 'task';
      message: string;
    };

export function sanitizeCactusJson(rawJson: string): string {
  return rawJson.replace(/(:\s*)-?(?:nan|inf|infinity)\b/gi, '$10.0');
}

let cactusMod: CactusEmscriptenModule | null = null;

async function fetchBinaryWithCheck(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} fetching ${url}`);
  }
  const buf = await res.arrayBuffer();
  if (buf.byteLength < 1024) {
    throw new Error(`Payload too small (${buf.byteLength} bytes) from ${url}`);
  }
  return new Uint8Array(buf);
}

async function instantiateCactusModule(
  jsUrl: string,
  wasmUrl: string
): Promise<CactusEmscriptenModule> {
  const [jsRes, wasmBytes] = await Promise.all([
    fetch(jsUrl),
    fetchBinaryWithCheck(wasmUrl),
  ]);
  if (!jsRes.ok) {
    throw new Error(`HTTP ${jsRes.status} fetching ${jsUrl}`);
  }
  const jsCode = await jsRes.text();
  const factoryFn = new Function(
    `${jsCode}\nreturn typeof createCactus !== 'undefined' ? createCactus : null;`
  ) as () => ((arg?: Record<string, unknown>) => Promise<CactusEmscriptenModule>) | null;

  const createCactus = factoryFn();
  if (typeof createCactus !== 'function') {
    throw new Error('Failed to evaluate createCactus factory from cactus.js');
  }

  return createCactus({
    wasmBinary: wasmBytes,
  });
}

function loadMirageArchiveIntoModule(
  mod: CactusEmscriptenModule,
  cactBytes: Uint8Array
): void {
  const ptr = mod._malloc(cactBytes.byteLength);
  if (!ptr) {
    throw new Error('Failed to allocate WASM heap for mirage.cact');
  }
  try {
    mod.HEAPU8.set(cactBytes, ptr);
    const rc = mod._cactus_load(ptr, BigInt(cactBytes.byteLength));
    if (rc !== 0) {
      const errPtr = mod._cactus_error();
      const errMsg = errPtr ? mod.UTF8ToString(errPtr) : `code ${rc}`;
      throw new Error(`cactus_load failed: ${errMsg}`);
    }
  } finally {
    mod._free(ptr);
  }
}

export function runCactusDescribe(
  mod: CactusEmscriptenModule,
  rgbBytes: Uint8Array,
  width: number,
  height: number,
  optionsJson: string
): MirageRawObservation {
  const rgbPtr = mod._malloc(rgbBytes.byteLength);
  if (!rgbPtr) {
    throw new Error('Failed to allocate RGB buffer in WASM heap');
  }
  mod.HEAPU8.set(rgbBytes, rgbPtr);

  const encodedOpts = new TextEncoder().encode(optionsJson);
  const optsByteLen = encodedOpts.byteLength + 1;
  const optsPtr = mod._malloc(optsByteLen);
  if (!optsPtr) {
    mod._free(rgbPtr);
    throw new Error('Failed to allocate options string in WASM heap');
  }
  mod.HEAPU8.set(encodedOpts, optsPtr);
  mod.HEAPU8[optsPtr + encodedOpts.byteLength] = 0;

  const outCap = 256 * 1024;
  const outPtr = mod._malloc(outCap);
  if (!outPtr) {
    mod._free(optsPtr);
    mod._free(rgbPtr);
    throw new Error('Failed to allocate output buffer in WASM heap');
  }

  try {
    const written = mod._cactus_describe(
      rgbPtr,
      width,
      height,
      optsPtr,
      0,
      0,
      outPtr,
      outCap
    );
    if (written < 0) {
      const errPtr = mod._cactus_error();
      const errMsg = errPtr ? mod.UTF8ToString(errPtr) : `code ${written}`;
      throw new Error(`cactus_describe failed: ${errMsg}`);
    }
    const rawJson = mod.UTF8ToString(outPtr, outCap);
    const sanitized = sanitizeCactusJson(rawJson);
    return JSON.parse(sanitized) as MirageRawObservation;
  } finally {
    mod._free(outPtr);
    mod._free(optsPtr);
    mod._free(rgbPtr);
  }
}

const workerScope =
  typeof self !== 'undefined' &&
  typeof (self as unknown as { postMessage?: unknown }).postMessage === 'function' &&
  typeof window === 'undefined'
    ? (self as unknown as {
        onmessage: ((ev: MessageEvent<MirageWorkerRequest>) => void) | null;
        postMessage: (msg: MirageWorkerResponse) => void;
      })
    : null;

if (workerScope) {
  workerScope.onmessage = async (event: MessageEvent<MirageWorkerRequest>) => {
    const msg = event.data;
    if (!msg || typeof msg !== 'object') return;

    if (msg.type === 'init') {
      let stage: 'wasm' | 'task' = 'wasm';
      try {
        const mod = await instantiateCactusModule(msg.jsUrl, msg.wasmUrl);
        stage = 'task';
        let cactBytes: Uint8Array;
        let sourceUsed: 'primary' | 'fallback' = 'primary';
        try {
          cactBytes = await fetchBinaryWithCheck(msg.modelUrl);
        } catch (primaryErr) {
          if (msg.fallbackModelUrl) {
            cactBytes = await fetchBinaryWithCheck(msg.fallbackModelUrl);
            sourceUsed = 'fallback';
          } else {
            throw primaryErr;
          }
        }
        loadMirageArchiveIntoModule(mod, cactBytes);
        cactusMod = mod;
        workerScope.postMessage({
          type: 'init-result',
          id: msg.id,
          ok: true,
          sourceUsed,
        });
      } catch (err) {
        workerScope.postMessage({
          type: 'error',
          id: msg.id,
          ok: false,
          stage,
          message: err instanceof Error ? err.message : String(err),
        });
      }
      return;
    }

    if (msg.type === 'describe') {
      try {
        if (!cactusMod) {
          throw new Error('Mirage WASM module is not initialized');
        }
        const rgbBytes = new Uint8Array(msg.rgbBuffer);
        const observation = runCactusDescribe(
          cactusMod,
          rgbBytes,
          msg.width,
          msg.height,
          msg.optionsJson
        );
        workerScope.postMessage({
          type: 'describe-result',
          id: msg.id,
          ok: true,
          observation,
        });
      } catch (err) {
        workerScope.postMessage({
          type: 'error',
          id: msg.id,
          ok: false,
          stage: 'task',
          message: err instanceof Error ? err.message : String(err),
        });
      }
      return;
    }

    if (msg.type === 'dispose') {
      try {
        if (cactusMod) {
          cactusMod._cactus_unload();
          cactusMod = null;
        }
      } finally {
        workerScope.postMessage({
          type: 'dispose-result',
          id: msg.id,
          ok: true,
        });
      }
    }
  };
}
