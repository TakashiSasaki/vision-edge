import {
  assertLoadedOrtVersionMatches,
  attachWasmBoundaryObserver,
  classifyOrtLoadedResources,
  HITNET_ORT_130_FAMILY_PROFILES,
  loadOrtEntryModuleWithBoundarySha256,
  type OrtLikeGpuAdapter,
  type OrtLoadedAssetProvenance,
  type OrtLoadedModuleWithEntryAsset,
} from './hitnetOrtComparisonHelper';
import {
  createStereoWorkerProcessorWithOrt,
  createWorkerOrtSessionForRuntime,
  HITNET_M1_FULL_GRAPH_BYTE_LENGTH,
  HITNET_M1_FULL_GRAPH_SHA256,
  type OrtWorkerRuntimeLike,
  type StereoWorkerRequest,
  type StereoWorkerResponse,
  type StereoWorkerSessionCreator,
} from './stereoDepthWorkerCore';

export const PINNED_DEV_FIXTURE_ORT_VERSION = '1.30.0' as const;

export interface DevWebGpuFixtureDeps {
  isDev?: boolean;
  loadOrtModule?: (
    entryModuleUrl: string
  ) => Promise<OrtLoadedModuleWithEntryAsset<OrtWorkerRuntimeLike> | OrtWorkerRuntimeLike>;
  requestWebGpuAdapter?: () => Promise<OrtLikeGpuAdapter | null>;
  getResourceEntries?: () => Array<{ name: string; decodedBodySize?: number }>;
  sha256Hex?: (bytes: Uint8Array) => Promise<string>;
}

async function defaultSha256Hex(bytes: Uint8Array): Promise<string> {
  if (typeof crypto === 'undefined' || !crypto.subtle) {
    throw new Error('crypto.subtle unavailable in DEV WebGPU fixture worker');
  }
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const digest = await crypto.subtle.digest('SHA-256', copy.buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

async function defaultRequestWebGpuAdapter(): Promise<OrtLikeGpuAdapter | null> {
  if (typeof navigator === 'undefined') return null;
  const gpu = (navigator as unknown as {
    gpu?: {
      requestAdapter?: (opts?: { powerPreference?: string }) => Promise<OrtLikeGpuAdapter | null>;
    };
  }).gpu;
  if (!gpu || typeof gpu.requestAdapter !== 'function') return null;
  try {
    return await gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch {
    return null;
  }
}

function defaultGetResourceEntries(): Array<{ name: string; decodedBodySize?: number }> {
  if (typeof performance === 'undefined' || typeof performance.getEntriesByType !== 'function') {
    return [];
  }
  return performance.getEntriesByType('resource').map((entry) => {
    const rec = entry as { name?: string; decodedBodySize?: number };
    return {
      name: rec.name || '',
      decodedBodySize: typeof rec.decodedBodySize === 'number' ? rec.decodedBodySize : undefined,
    };
  });
}

function collectNonJsepAssetProvenance(
  entryProvenance: OrtLoadedAssetProvenance | null,
  wasmProvenances: OrtLoadedAssetProvenance[],
  resourceEntries: Array<{ name: string; decodedBodySize?: number }>,
  expectedEmscriptenMjsSha256: string
): OrtLoadedAssetProvenance[] {
  const out: OrtLoadedAssetProvenance[] = [];
  if (entryProvenance) out.push(entryProvenance);
  for (const wp of wasmProvenances) out.push(wp);
  const seenUrls = new Set(out.map((p) => p.url));
  for (const ent of resourceEntries) {
    if (!ent.name || seenUrls.has(ent.name)) continue;
    if (
      ent.name.endsWith('.asyncify.mjs') ||
      ent.name.endsWith('.jsep.mjs') ||
      ent.name.endsWith('ort-wasm-simd-threaded.mjs')
    ) {
      seenUrls.add(ent.name);
      out.push({
        role: 'emscripten-mjs',
        url: ent.name,
        byteLength: ent.decodedBodySize ?? null,
        sha256: null,
        via: 'performance-resource-entry',
        referenceLocalDistSha256: ent.name.endsWith('.asyncify.mjs')
          ? expectedEmscriptenMjsSha256
          : null,
        matchesLocalDistSha256: null,
        note: 'Loaded via native dynamic import(httpsUrl) from ORT string wasmPaths; raw JS bytes are not observable in worker JS without changing ORT wasmPaths.',
      });
    }
  }
  return out;
}

export function createDevWebGpuFixtureProcessor(deps: DevWebGpuFixtureDeps = {}) {
  const isDev = deps.isDev ?? Boolean(import.meta.env.DEV);
  const profile = HITNET_ORT_130_FAMILY_PROFILES.asyncify;
  const entryModuleUrl = `${profile.distBaseUrl}${profile.entryFile}`;
  const sha256 = deps.sha256Hex ?? defaultSha256Hex;
  const requestAdapter = deps.requestWebGpuAdapter ?? defaultRequestWebGpuAdapter;
  const getResourceEntries = deps.getResourceEntries ?? defaultGetResourceEntries;
  const loadOrtModule =
    deps.loadOrtModule ??
    ((url: string) =>
      loadOrtEntryModuleWithBoundarySha256<OrtWorkerRuntimeLike>(url, {
        referenceLocalDistSha256: profile.referenceLocalDistSha256.entry,
      }));

  let activeOrtRuntime: OrtWorkerRuntimeLike | null = null;
  let hasVerifiedAdapter = false;
  let expectExactM1InSessionCreator = false;

  const devSessionCreator: StereoWorkerSessionCreator = async (
    onnxBytes,
    delegate,
    wasmPrimaryUrl,
    wasmFallbackUrl,
    strictWebGpu,
    sessionOptionOverrides
  ) => {
    if (delegate !== 'WebGPU' || !strictWebGpu) {
      throw new Error(
        `[fallback-guard] DEV WebGPU fixture requires delegate="WebGPU" and strictWebGpu=true (got delegate=${delegate}, strictWebGpu=${String(strictWebGpu)})`
      );
    }
    if (wasmPrimaryUrl !== profile.distBaseUrl || wasmFallbackUrl) {
      throw new Error(
        `[resource-guard] DEV WebGPU fixture requires pinned ORT ${PINNED_DEV_FIXTURE_ORT_VERSION} asyncify distBaseUrl=${profile.distBaseUrl} without fallback (got primary=${wasmPrimaryUrl}, fallback=${String(wasmFallbackUrl)})`
      );
    }

    const loaded = await loadOrtModule(entryModuleUrl);
    let ortMod: OrtWorkerRuntimeLike;
    let entryProvenance: OrtLoadedAssetProvenance;
    if ('module' in loaded) {
      ortMod = loaded.module;
      entryProvenance = loaded.entryAsset ?? {
        role: 'entry',
        url: entryModuleUrl,
        byteLength: null,
        sha256: null,
        via: 'fetch+blob-import',
        referenceLocalDistSha256: profile.referenceLocalDistSha256.entry,
        matchesLocalDistSha256: null,
      };
    } else {
      ortMod = loaded;
      entryProvenance = {
        role: 'entry',
        url: entryModuleUrl,
        byteLength: null,
        sha256: null,
        via: 'direct-import-unhashed',
        referenceLocalDistSha256: profile.referenceLocalDistSha256.entry,
        matchesLocalDistSha256: null,
      };
    }

    if (
      entryProvenance.sha256 &&
      entryProvenance.sha256 !== profile.referenceLocalDistSha256.entry
    ) {
      throw new Error(
        `[resource-guard] Loaded entry module SHA-256 mismatch for ${profile.entryFile}: expected=${profile.referenceLocalDistSha256.entry}, got=${entryProvenance.sha256}`
      );
    }

    try {
      assertLoadedOrtVersionMatches(ortMod.env.versions ?? null, PINNED_DEV_FIXTURE_ORT_VERSION);
    } catch (verErr) {
      throw new Error(
        `[version-guard] ${verErr instanceof Error ? verErr.message : String(verErr)}`,
        { cause: verErr }
      );
    }

    activeOrtRuntime = ortMod;

    const verifiedAdapterForInit = hasVerifiedAdapter || Boolean(await requestAdapter());
    hasVerifiedAdapter = false;
    if (!verifiedAdapterForInit) {
      expectExactM1InSessionCreator = false;
      throw new Error(
        '[webgpu-support] WebGPU adapter unavailable in DEV app fixture worker (explicit error, no WASM fallback).'
      );
    }

    const enforceExactM1InSessionCreator = expectExactM1InSessionCreator;
    expectExactM1InSessionCreator = false;
    if (enforceExactM1InSessionCreator) {
      const preSessionCreateSha = await sha256(onnxBytes);
      if (
        onnxBytes.byteLength !== HITNET_M1_FULL_GRAPH_BYTE_LENGTH ||
        preSessionCreateSha !== HITNET_M1_FULL_GRAPH_SHA256
      ) {
        throw new Error(
          `[model-guard] Session creation input bytes drifted from exact M1 immediately before InferenceSession.create: len=${onnxBytes.byteLength} (expected ${HITNET_M1_FULL_GRAPH_BYTE_LENGTH}), sha=${preSessionCreateSha} (expected ${HITNET_M1_FULL_GRAPH_SHA256})`
        );
      }
    }

    const wasmObs = attachWasmBoundaryObserver(sha256, profile.referenceLocalDistSha256.wasm);
    let created: Awaited<ReturnType<typeof createWorkerOrtSessionForRuntime>>;
    try {
      created = await createWorkerOrtSessionForRuntime(
        ortMod,
        onnxBytes,
        'WebGPU',
        profile.distBaseUrl,
        null,
        true,
        sessionOptionOverrides,
        verifiedAdapterForInit
      );
    } catch (initErr) {
      activeOrtRuntime = null;
      throw initErr;
    } finally {
      wasmObs.detach();
    }

    try {
      if (
        created.backend.configuredProviders.length !== 1 ||
        created.backend.configuredProviders[0] !== 'webgpu' ||
        created.backend.fallbackReason !== null ||
        created.wasmSourceUsed !== 'primary'
      ) {
        throw new Error(
          `[fallback-guard] Unexpected provider/fallback state in DEV non-JSEP WebGPU fixture: configuredProviders=${JSON.stringify(created.backend.configuredProviders)}, fallbackReason=${String(created.backend.fallbackReason)}, wasmSourceUsed=${created.wasmSourceUsed}`
        );
      }
      if (ortMod.env.wasm.wasmPaths !== profile.distBaseUrl) {
        throw new Error(
          `[resource-guard] Failed to keep ort.env.wasm.wasmPaths pinned to ${profile.distBaseUrl} (got ${String(ortMod.env.wasm.wasmPaths)})`
        );
      }

      const resourceEntries = getResourceEntries();
      const resCheck = classifyOrtLoadedResources(
        resourceEntries.map((e) => e.name),
        profile.distBaseUrl
      );
      if (resCheck.foreignOrtResources.length > 0) {
        throw new Error(
          `[resource-guard] Unexpected foreign ORT assets loaded: ${resCheck.foreignOrtResources.join(', ')}`
        );
      }

      const forbiddenInLoaded = resCheck.loadedOrtResources.filter(
        (u) =>
          profile.forbiddenAssetSubstrings.some((sub) => u.includes(sub)) ||
          u.includes('ort.bundle.')
      );
      if (forbiddenInLoaded.length > 0) {
        throw new Error(
          `[resource-guard] Unexpected JSEP/bundle ORT asset loaded in non-JSEP fixture: ${forbiddenInLoaded.join(', ')}`
        );
      }

      const forbiddenInWasm = wasmObs.wasmProvenances.filter(
        (wp) =>
          profile.forbiddenAssetSubstrings.some((sub) => wp.url.includes(sub)) ||
          wp.matchesLocalDistSha256 === false
      );
      if (forbiddenInWasm.length > 0) {
        throw new Error(
          `[resource-guard] Unexpected or mismatched WASM asset instantiated in non-JSEP fixture: ${JSON.stringify(forbiddenInWasm)}`
        );
      }

      const loadedAssetProvenance = collectNonJsepAssetProvenance(
        entryProvenance,
        wasmObs.wasmProvenances,
        resourceEntries,
        profile.referenceLocalDistSha256.emscriptenMjs
      );

      return {
        ...created,
        runtimeAssetFamily: 'asyncify',
        entryModuleUrl,
        loadedOrtResources: resCheck.loadedOrtResources,
        foreignOrtResources: resCheck.foreignOrtResources,
        loadedAssetProvenance,
      };
    } catch (guardErr) {
      activeOrtRuntime = null;
      await created.session.release().catch(() => {});
      throw guardErr;
    }
  };

  const innerProcessor = createStereoWorkerProcessorWithOrt(
    () => activeOrtRuntime,
    devSessionCreator,
    undefined,
    () => {
      activeOrtRuntime = null;
    },
    {
      allowDevPretransformedExactM1SkipConcatSplit: isDev,
    }
  );

  return {
    async handleMessage(
      msg: StereoWorkerRequest,
      postResponse: (response: StereoWorkerResponse, transfer?: Transferable[]) => void
    ): Promise<void> {
      if (!msg || typeof msg !== 'object') return;

      if (msg.type === 'dispose') {
        await innerProcessor.handleMessage(msg, postResponse);
        return;
      }

      if (!isDev) {
        postResponse({
          type: 'error',
          id: msg.id,
          ok: false,
          stage: 'wasm',
          message:
            '[dev-gate] DEV-ONLY WebGPU non-JSEP app worker fixture is disabled outside development builds.',
        });
        return;
      }

      if (msg.type === 'init-sad' || msg.type === 'init-wgsl') {
        postResponse({
          type: 'error',
          id: msg.id,
          ok: false,
          stage: 'input',
          message: `[opt-in-guard] DEV WebGPU non-JSEP fixture does not permit "${msg.type}".`,
        });
        return;
      }

      if (msg.type === 'init-onnx') {
        if (msg.profile?.variantId !== 'hitnet-sf' || msg.profile?.engineType !== 'onnx') {
          postResponse({
            type: 'error',
            id: msg.id,
            ok: false,
            stage: 'input',
            message: `[opt-in-guard] DEV WebGPU non-JSEP fixture is restricted to hitnet-sf ONNX (got variantId=${String(msg.profile?.variantId)}, engineType=${String(msg.profile?.engineType)}).`,
          });
          return;
        }
        if (msg.backendPreference !== 'webgpu' || msg.strictWebGpu !== true) {
          postResponse({
            type: 'error',
            id: msg.id,
            ok: false,
            stage: 'input',
            message: `[opt-in-guard] DEV WebGPU non-JSEP fixture requires backendPreference="webgpu" and strictWebGpu=true (got backendPreference=${String(msg.backendPreference)}, strictWebGpu=${String(msg.strictWebGpu)}).`,
          });
          return;
        }
        if (msg.graphOptimizationLevel !== 'basic') {
          postResponse({
            type: 'error',
            id: msg.id,
            ok: false,
            stage: 'input',
            message: `[opt-in-guard] DEV WebGPU non-JSEP fixture requires graphOptimizationLevel="basic" (got ${String(msg.graphOptimizationLevel)}).`,
          });
          return;
        }
        if (msg.wasmPrimaryUrl !== profile.distBaseUrl || msg.wasmFallbackUrl) {
          postResponse({
            type: 'error',
            id: msg.id,
            ok: false,
            stage: 'wasm',
            message: `[resource-guard] DEV WebGPU non-JSEP fixture requires wasmPrimaryUrl=${profile.distBaseUrl} and wasmFallbackUrl=null (got primary=${String(msg.wasmPrimaryUrl)}, fallback=${String(msg.wasmFallbackUrl)}).`,
          });
          return;
        }
        if (
          msg.allowDiagnosticTransformedWasm ||
          (msg.normalizeSliceAxes && msg.normalizeSliceAxes !== 'off') ||
          msg.normalizeGatherNdRank ||
          msg.normalizePropGatherNdRank ||
          (msg.levelInitGatherMode && msg.levelInitGatherMode !== 'off') ||
          msg.normalizeInitToPropZeroSlant ||
          msg.normalizeSlot0ReshapeToTranspose ||
          msg.normalizeArgMinKeepDims
        ) {
          postResponse({
            type: 'error',
            id: msg.id,
            ok: false,
            stage: 'input',
            message:
              '[opt-in-guard] DEV WebGPU non-JSEP fixture expects exact prebuilt M1 bytes with all in-worker graph rewrite flags disabled.',
          });
          return;
        }

        if (msg.skipImplicitConcatSplitForExactM1Fixture) {
          if (Array.isArray(msg.diagnosticTapOutputs) && msg.diagnosticTapOutputs.length > 0) {
            postResponse({
              type: 'error',
              id: msg.id,
              ok: false,
              stage: 'input',
              message:
                '[opt-in-guard] DEV exact-M1 Concat-split skip does not permit diagnosticTapOutputs.',
            });
            return;
          }
          const rawM1Sha =
            msg.onnxBuffer instanceof ArrayBuffer
              ? await sha256(new Uint8Array(msg.onnxBuffer))
              : '';
          if (
            !(msg.onnxBuffer instanceof ArrayBuffer) ||
            msg.onnxBuffer.byteLength !== HITNET_M1_FULL_GRAPH_BYTE_LENGTH ||
            rawM1Sha !== HITNET_M1_FULL_GRAPH_SHA256
          ) {
            postResponse({
              type: 'error',
              id: msg.id,
              ok: false,
              stage: 'input',
              message: `[model-guard] DEV exact-M1 Concat-split skip requires exact M1 bytes (${HITNET_M1_FULL_GRAPH_BYTE_LENGTH} B, sha=${HITNET_M1_FULL_GRAPH_SHA256}), got bytes=${msg.onnxBuffer instanceof ArrayBuffer ? msg.onnxBuffer.byteLength : 'invalid'}, sha=${rawM1Sha}.`,
            });
            return;
          }
        }

        hasVerifiedAdapter = false;
        expectExactM1InSessionCreator = false;
        const adapter = await requestAdapter();
        if (!adapter) {
          postResponse({
            type: 'error',
            id: msg.id,
            ok: false,
            stage: 'wasm',
            message:
              '[webgpu-support] WebGPU adapter unavailable in DEV app fixture worker (explicit error, no WASM fallback).',
          });
          return;
        }
        hasVerifiedAdapter = true;
        expectExactM1InSessionCreator = Boolean(msg.skipImplicitConcatSplitForExactM1Fixture);
      }

      await innerProcessor.handleMessage(msg, postResponse);
    },
  };
}

const workerScope =
  typeof self !== 'undefined' &&
  typeof (self as unknown as { postMessage?: unknown }).postMessage === 'function' &&
  typeof window === 'undefined'
    ? (self as unknown as {
        onmessage: ((ev: MessageEvent<StereoWorkerRequest>) => void) | null;
        postMessage: (msg: StereoWorkerResponse, transfer?: Transferable[]) => void;
      })
    : null;

if (workerScope) {
  const processor = createDevWebGpuFixtureProcessor();
  workerScope.onmessage = (event: MessageEvent<StereoWorkerRequest>) => {
    void processor.handleMessage(event.data, (res, transfer) => {
      if (transfer && transfer.length > 0) {
        workerScope.postMessage(res, transfer);
      } else {
        workerScope.postMessage(res);
      }
    });
  };
}
