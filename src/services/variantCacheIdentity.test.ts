// Model-free routing controls: actual cache/resolver/engine, controlled fetch/SDK.
import { beforeEach, afterEach, expect, it, vi } from 'vitest';
import { mkdtemp, rm, readFile, writeFile, readdir, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { FilesetResolver, ObjectDetector, ImageSegmenter } from '@mediapipe/tasks-vision';
import { ContainerModelCacheManager } from './containerModelCacheServer';
import { MODEL_VARIANT_CATALOG, resolveModelAssetUrls, getVariantCatalogEntry, getResolvedSource, fetchContainerCacheStatus, setContainerModelCacheEnabledForTest } from './modelAssetResolver';
import { VisionEngine } from './visionEngine';
const boundary = vi.hoisted(() => ({ create: vi.fn() }));
vi.mock('onnxruntime-web', () => ({ env: { wasm: {}, logLevel: 'error' }, InferenceSession: { create: boundary.create } }));
const entries = Object.entries(MODEL_VARIANT_CATALOG);
const row = (e: typeof entries[number][1], status = 'cached', error: string | null = null) => ({ id: e.modelId, variantId: e.variantId, file: e.file, localUrl: e.localUrl, upstreamUrl: e.upstreamUrl, license: e.license, status, error, size: status === 'cached' ? 2048 : null, cachedAt: status === 'cached' ? new Date().toISOString() : null });
function statusPayload(variants = true, state = 'cached') {
    return { enabled: true, wasm: { mediapipe: { available: false }, ort: { available: false } }, models: Object.fromEntries(entries.filter(([, e]) => e.isDefault).map(([, e]) => [e.modelId, row(e, state)])), ...(variants ? { variants: Object.fromEntries(entries.map(([k, e]) => [k, row(e, state)])) } : {}) };
}
let directory: string;
beforeEach(async () => { directory = await mkdtemp(path.join(tmpdir(), 'variant-identity-')); setContainerModelCacheEnabledForTest(true); boundary.create.mockReset(); });
afterEach(async () => { setContainerModelCacheEnabledForTest(null); vi.restoreAllMocks(); vi.useRealTimers(); await rm(directory, { recursive: true, force: true }); });
function statusFetch(value: unknown) { return vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } })); }
function manager() { return new ContainerModelCacheManager({ cacheDir: directory, fetchImpl: vi.fn(async (url) => new Response(Buffer.alloc(2048, entries.findIndex(([, e]) => e.upstreamUrl === String(url)) + 1))) }); }
for (const state of ['cached', 'downloading'])
    it.each(entries)('legacy ' + state + ' never supplies a different variant: %s', async (_key, e) => {
        statusFetch(statusPayload(false, state));
        const r = await resolveModelAssetUrls(e.modelId, e.variantId);
        expect(r.primaryUrl).toBe(e.isDefault ? e.localUrl : e.upstreamUrl);
        expect(r.fallbackUrl).toBe(e.isDefault ? e.upstreamUrl : null);
    });
it('server exposes all9 variant identities and default aliases from one snapshot', async () => {
    const m = manager();
    await m.prefetchAllModels();
    const s = await m.getStatusResponse();
    expect(s).toHaveProperty('variants');
    expect(Object.keys(s.variants!)).toHaveLength(9);
    for (const [key, e] of entries) {
        expect(s).toHaveProperty(['variants', key], expect.objectContaining({ ...row(e), cachedAt: expect.any(String) }));
        if (e.isDefault)
            expect(s.models[e.modelId]).toEqual(s.variants![key]);
    }
});
it('restart rejects swapped bytes rather than blessing a filename by size', async () => {
    const m = manager();
    await m.prefetchAllModels();
    await writeFile(m.getVariantFilePath('mediapipe:lite2'), Buffer.alloc(2048, 99));
    const s = await manager().getStatusResponse();
    expect(s).toHaveProperty(['variants', 'mediapipe:lite2', 'status'], 'unavailable');
});
it('unknown explicit variant cannot be relabelled as a default asset', () => {
    expect(() => getVariantCatalogEntry('mediapipe', 'unknown')).toThrow();
    const engine = new VisionEngine();
    expect(() => engine.initialize('mediapipe', 'unknown')).toThrow();
    expect(engine.getState().status).toBe('idle');
    engine.dispose();
});
for (const state of ['cached', 'downloading'])
    it.each(entries)('modern ' + state + ' routes exact identity %s', async (_key, e) => {
        statusFetch(statusPayload(true, state));
        const r = await resolveModelAssetUrls(e.modelId, e.variantId);
        expect(r).toEqual({ primaryUrl: e.localUrl, fallbackUrl: e.upstreamUrl, source: 'container-cache' });
    });
for (const mode of ['unavailable', 'missing-row', 'disabled-server', 'disabled-client'])
    it.each(entries)(mode + ' uses only requested upstream %s', async (key, e) => {
        const payload = statusPayload();
        if (mode === 'unavailable')
            payload.variants![key] = row(e, 'unavailable', 'CONTROLLED_UNAVAILABLE');
        if (mode === 'missing-row')
            delete payload.variants![key];
        if (mode === 'disabled-server')
            payload.enabled = false;
        if (mode === 'disabled-client')
            setContainerModelCacheEnabledForTest(false);
        const fetch = statusFetch(payload), r = await resolveModelAssetUrls(e.modelId, e.variantId);
        expect(r).toEqual({ primaryUrl: e.upstreamUrl, fallbackUrl: null, source: mode === 'disabled-client' ? 'external' : 'external-fallback' });
        if (mode === 'disabled-client')
            expect(fetch).not.toHaveBeenCalled();
    });
for (const [field, value] of [['id', 'mediapipe-wrong'], ['variantId', 'wrong'], ['file', 'wrong.onnx'], ['localUrl', '/api/model-cache/models/wrong.onnx'], ['upstreamUrl', 'https://wrong.invalid/wrong'], ['error', 'CONTROLLED_FAILURE'], ['size', '2048'], ['cachedAt', null], ['status', 'unknown']] as const) {
    it.each(entries)(`mismatched ${field} cannot be used for %s`, async (key, e) => { const payload = statusPayload(); Object.assign(payload.variants![key], { [field]: value }); statusFetch(payload); expect(await resolveModelAssetUrls(e.modelId, e.variantId)).toMatchObject({ primaryUrl: e.upstreamUrl, fallbackUrl: null }); });
}
it.each(entries.filter(([, e]) => !e.isDefault))('modern wrong default row at nondefault key is rejected %s', async (key, e) => { const payload = statusPayload(); const d = getVariantCatalogEntry(e.modelId); payload.variants![key] = row(d); statusFetch(payload); expect((await resolveModelAssetUrls(e.modelId, e.variantId)).primaryUrl).toBe(e.upstreamUrl); });
for (const bad of [null, [], { enabled: 'true', models: {}, wasm: {} }, { enabled: true, models: [], wasm: {} }])
    it('malformed status fails closed ' + JSON.stringify(bad), async () => { statusFetch(bad); const e = MODEL_VARIANT_CATALOG['mediapipe:lite2']; expect((await resolveModelAssetUrls(e.modelId, e.variantId)).primaryUrl).toBe(e.upstreamUrl); });
it('malformed modern variants field does not become a legacy-default response', async () => { const payload = { ...statusPayload(), variants: null }; statusFetch(payload); const e = getVariantCatalogEntry('mediapipe'); expect((await resolveModelAssetUrls(e.modelId, e.variantId)).primaryUrl).toBe(e.upstreamUrl); });
it('HTTP/JSON errors clear prior status instead of reusing old cache selection', async () => {
    const fetch = statusFetch(statusPayload());
    await fetchContainerCacheStatus();
    fetch.mockResolvedValueOnce(new Response('unavailable', { status: 503 }));
    expect(await fetchContainerCacheStatus(true)).toBeNull();
    fetch.mockResolvedValueOnce(new Response('bad-json', { headers: { 'Content-Type': 'application/json' } }));
    expect(await fetchContainerCacheStatus()).toBeNull();
    fetch.mockResolvedValueOnce(new Response('{}', { headers: { 'Content-Type': 'text/plain' } }));
    const e = getVariantCatalogEntry('mediapipe', 'lite2');
    expect((await resolveModelAssetUrls(e.modelId, e.variantId)).primaryUrl).toBe(e.upstreamUrl);
});
async function sdkBoundary() {
    vi.spyOn(FilesetResolver, 'forVisionTasks').mockResolvedValue({ wasmLoaderPath: 'mock-loader', wasmBinaryPath: 'mock-wasm' });
    const detector = vi.spyOn(ObjectDetector, 'createFromOptions').mockResolvedValue({ detectForVideo: vi.fn(), close: vi.fn() } as unknown as ObjectDetector);
    const segmenter = vi.spyOn(ImageSegmenter, 'createFromOptions').mockResolvedValue({ segmentForVideo: vi.fn(), close: vi.fn() } as unknown as ImageSegmenter);
    boundary.create.mockResolvedValue({ inputNames: [], outputNames: [], run: vi.fn(), release: vi.fn(async () => { }) });
    return { detector, segmenter };
}
for (const mode of ['local', 'legacy', 'fallback'])
    it.each(entries)(`actual initialization ${mode} binds requested URL and ready identity %s`, async (_key, e) => {
        statusFetch(statusPayload(mode !== 'legacy'));
        const sdk = await sdkBoundary();
        if (mode === 'fallback') {
            if (e.modelId === 'mediapipe')
                sdk.detector.mockRejectedValueOnce(new Error('CONTROLLED_LOCAL_MODEL_FAILURE'));
            else if (e.modelId === 'selfie-segmenter')
                sdk.segmenter.mockRejectedValueOnce(new Error('CONTROLLED_LOCAL_MODEL_FAILURE'));
            else
                boundary.create.mockRejectedValueOnce(new Error('CONTROLLED_LOCAL_MODEL_FAILURE'));
        }
        const engine = new VisionEngine();
        const state = await engine.initialize(e.modelId, e.variantId);
        expect(state).toMatchObject({ status: 'ready', model: e.modelId, variantId: e.variantId });
        const urls = e.modelId === 'mediapipe' ? sdk.detector.mock.calls.map(c => c[1].baseOptions?.modelAssetPath) : e.modelId === 'selfie-segmenter' ? sdk.segmenter.mock.calls.map(c => c[1].baseOptions?.modelAssetPath) : boundary.create.mock.calls.map(c => c[0]);
        expect(urls).toEqual(mode === 'fallback' ? [e.localUrl, e.upstreamUrl] : [mode === 'legacy' && !e.isDefault ? e.upstreamUrl : e.localUrl]);
        expect(getResolvedSource(`${e.modelId}:${e.variantId}`)).toBe(mode === 'fallback' || (mode === 'legacy' && !e.isDefault) ? 'external-fallback' : 'container-cache');
        engine.dispose();
        await Promise.resolve();
    });
// Invoke actual middleware with controlled HTTP I/O, not a separate routing clone.
async function request(m: ContainerModelCacheManager, url: string, method = 'GET') {
    const headers: Record<string, string> = {};
    let body: Buffer | string = '';
    const res = { statusCode: 200, setHeader: (k: string, v: string) => { headers[k.toLowerCase()] = String(v); }, end: (b: Buffer | string = '') => { body = b; } };
    await m.createMiddleware()({ url, method } as import('node:http').IncomingMessage, res as unknown as import('node:http').ServerResponse, (error?: unknown) => { if (error)
        throw error; });
    return { status: res.statusCode, headers, body: Buffer.isBuffer(body) ? body : Buffer.from(body) };
}
it('restart reuses verified9 payloads; HTTP filenames, status and returned bytes correspond', async () => {
    const first = manager();
    await first.prefetchAllModels();
    const fetch = vi.fn(async () => { throw new Error('UNEXPECTED_FETCH'); });
    const restarted = new ContainerModelCacheManager({ cacheDir: directory, fetchImpl: fetch });
    const status = JSON.parse((await request(restarted, '/api/model-cache/status')).body.toString());
    expect(Object.keys(status.variants)).toHaveLength(9);
    for (const [index, [key, e]] of entries.entries()) {
        expect(status.variants[key]).toMatchObject({ id: e.modelId, variantId: e.variantId, file: e.file, localUrl: e.localUrl, status: 'cached' });
        const r = await request(restarted, e.localUrl);
        expect(r.status).toBe(200);
        expect(r.headers['x-model-id']).toBe(e.modelId);
        expect(r.headers['x-model-variant']).toBe(e.variantId);
        expect(r.body).toEqual(Buffer.alloc(2048, index + 1));
        const head = await request(restarted, e.localUrl, 'HEAD');
        expect(head.status).toBe(200);
        expect(head.body.length).toBe(0);
    }
    expect((await request(restarted, '/api/model-cache/models/unknown.onnx')).status).toBe(404);
    expect(fetch).not.toHaveBeenCalled();
});
it('unrecorded old bytes are not silently adopted or served', async () => { const m = manager(), e = getVariantCatalogEntry('mediapipe', 'lite2'); await writeFile(m.getVariantFilePath('mediapipe:lite2'), Buffer.alloc(2048)); const s = await m.getStatusResponse(); expect(s.variants!['mediapipe:lite2']).toMatchObject({ status: 'unavailable', size: null, error: expect.stringContaining('Unverified') }); expect((await request(m, e.localUrl)).status).toBe(503); });
for (const field of ['variantId', 'modelId', 'file', 'upstreamUrl', 'sha256'])
    it(`restart rejects corrupted identity ${field}`, async () => { const m = manager(); await m.prefetchVariant('segformer:b2'); const meta = m.getVariantFilePath('segformer:b2') + '.identity.json'; const r = JSON.parse(await readFile(meta, 'utf8')); r[field] = 'wrong'; await writeFile(meta, JSON.stringify(r)); const s = await manager().getStatusResponse(); expect(s.variants!['segformer:b2'].status).toBe('unavailable'); });
it('status invalidates deleted verified files', async () => { const m = manager(); await m.prefetchVariant('segformer:b1'); await rm(m.getVariantFilePath('segformer:b1')); const s = await m.getStatusResponse(); expect(s.variants!['segformer:b1']).toMatchObject({ status: 'unavailable', size: null, cachedAt: null }); });
it('a file changed after status validation is not served as cached weights', async () => { const m = manager(); await m.prefetchVariant('segformer:b1'); const original = m.syncVariantDiskState.bind(m); vi.spyOn(m, 'syncVariantDiskState').mockImplementationOnce(async (key) => { const s = await original(key); await writeFile(m.getVariantFilePath(key), Buffer.alloc(2048, 99)); return s; }); expect((await request(m, MODEL_VARIANT_CATALOG['segformer:b1'].localUrl)).status).toBe(503); });
it('failed acquisition remains unavailable without implicit request retries; temp files cleaned', async () => { const fetch = vi.fn(async () => new Response('controlled denied', { status: 403 })); const m = new ContainerModelCacheManager({ cacheDir: directory, fetchImpl: fetch }); await m.prefetchVariant('mediapipe:lite2'); expect((await m.getStatusResponse()).variants!['mediapipe:lite2'].error).toContain('HTTP 403'); expect((await request(m, MODEL_VARIANT_CATALOG['mediapipe:lite2'].localUrl)).status).toBe(503); expect(fetch).toHaveBeenCalledTimes(1); expect((await readdir(directory)).some(p => p.includes('.part-'))).toBe(false); });
it('concurrent requests deduplicate each variant; downloading identities stay separate', async () => { let release!: () => void; const gate = new Promise<void>(r => { release = r; }); const fetch = vi.fn(async () => { await gate; return new Response(Buffer.alloc(2048, 5)); }); const m = new ContainerModelCacheManager({ cacheDir: directory, fetchImpl: fetch }); const a = m.prefetchVariant('mediapipe:lite2'), b = m.prefetchVariant('mediapipe:lite2'), c = m.prefetchVariant('mediapipe:lite0'); await vi.waitFor(() => expect(fetch).toHaveBeenCalledTimes(2)); const s = await m.getStatusResponse(); expect(s.variants!['mediapipe:lite2']).toMatchObject({ status: 'downloading', variantId: 'lite2' }); expect(s.models.mediapipe).toMatchObject({ status: 'downloading', variantId: 'lite0' }); release(); await Promise.all([a, b, c]); expect(fetch).toHaveBeenCalledTimes(2); });
it.each(entries)('both local/upstream failure cannot yield ready identity %s', async (_key, e) => { statusFetch(statusPayload()); const sdk = await sdkBoundary(); const error = new Error('CONTROLLED_ALL_MODEL_FAILURE'); if (e.modelId === 'mediapipe')
    sdk.detector.mockRejectedValue(error);
else if (e.modelId === 'selfie-segmenter')
    sdk.segmenter.mockRejectedValue(error);
else
    boundary.create.mockRejectedValue(error); const engine = new VisionEngine(); expect(await engine.initialize(e.modelId, e.variantId)).toMatchObject({ status: 'error', model: e.modelId, variantId: e.variantId }); expect(engine.getState().status).not.toBe('ready'); engine.dispose(); });
it('invalid request leaves an already-ready valid variant accurately labelled', async () => { statusFetch(statusPayload()); const sdk = await sdkBoundary(); const engine = new VisionEngine(); const ready = await engine.initialize('mediapipe', 'lite2'); expect(() => engine.initialize('mediapipe', 'not-a-variant')).toThrow(RangeError); expect(engine.getState()).toBe(ready); expect(sdk.detector).toHaveBeenCalledTimes(1); engine.dispose(); });
it('late initialization cannot replace a newer variant or retain its task', async () => { statusFetch(statusPayload()); const sdk = await sdkBoundary(); let resolve!: (value: ObjectDetector) => void; const pending = new Promise<ObjectDetector>(yes => { resolve = yes; }); sdk.detector.mockImplementationOnce(() => pending); const engine = new VisionEngine(); const older = engine.initialize('mediapipe', 'lite2'); await vi.waitFor(() => expect(sdk.detector).toHaveBeenCalledTimes(1)); expect(await engine.initialize('mediapipe', 'lite0')).toMatchObject({ status: 'ready', variantId: 'lite0' }); const close = vi.fn(); resolve({ close, detectForVideo: vi.fn() } as unknown as ObjectDetector); expect(await older).toMatchObject({ status: 'cancelled' }); expect(engine.getState()).toMatchObject({ status: 'ready', variantId: 'lite0' }); expect(close).toHaveBeenCalledTimes(1); engine.dispose(); });
it('all9 actual downloading status entries remain distinct until completion', async () => { let release!: () => void; const gate = new Promise<void>(r => { release = r; }); const fetch = vi.fn(async () => { await gate; return new Response(Buffer.alloc(2048, 7)); }); const m = new ContainerModelCacheManager({ cacheDir: directory, fetchImpl: fetch }); const pending = m.prefetchAllModels(); await vi.waitFor(() => expect(fetch).toHaveBeenCalledTimes(9)); const status = await m.getStatusResponse(); for (const [key, e] of entries)
    expect(status.variants![key]).toMatchObject({ status: 'downloading', id: e.modelId, variantId: e.variantId, file: e.file, size: null, cachedAt: null }); release(); await pending; });
for (const kind of ['fetch', 'body'])
    it(`bounded ${kind} timeout returns an unavailable status even for an ignoring mock`, async () => { vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] }); const fetch = vi.fn(async () => kind === 'fetch' ? new Promise<Response>(() => { }) : new Response(new ReadableStream({ start() { } }))); const m = new ContainerModelCacheManager({ cacheDir: directory, fetchImpl: fetch }); const pending = m.prefetchVariant('mediapipe:lite2'); await vi.waitFor(() => expect(fetch).toHaveBeenCalledTimes(1)); await vi.advanceTimersByTimeAsync(45000); expect(await pending).toMatchObject({ status: 'unavailable', error: expect.stringContaining('timeout') }); expect((await readdir(directory)).some(p => p.includes('.part-'))).toBe(false); });
for (const kind of ['small', 'large-header'])
    it(`invalid ${kind} body cannot become a valid cached model`, async () => { const fetch = vi.fn(async () => new Response(Buffer.alloc(1023), kind === 'large-header' ? { headers: { 'Content-Length': String(512 * 1024 * 1024 + 1) } } : undefined)); const m = new ContainerModelCacheManager({ cacheDir: directory, fetchImpl: fetch }); expect(await m.prefetchVariant('segformer:b2')).toMatchObject({ status: 'unavailable', size: null, error: expect.stringContaining(kind === 'small' ? 'too small' : 'size limit') }); expect((await readdir(directory)).length).toBe(0); });
for (const kind of ['file', 'metadata'])
    it(`restart rejects a ${kind} symlink`, async () => { const m = manager(); await m.prefetchVariant('segformer:b2'); const file = m.getVariantFilePath('segformer:b2'), target = kind === 'file' ? file : file + '.identity.json'; const bytes = await readFile(target); await writeFile(path.join(directory, 'elsewhere'), bytes); await rm(target); await symlink(path.join(directory, 'elsewhere'), target); expect((await manager().getStatusResponse()).variants!['segformer:b2'].status).toBe('unavailable'); });
it('unknown cache key/path never becomes a default fetch', async () => { const fetch = vi.fn(async () => { throw Error('NO_FETCH'); }); const m = new ContainerModelCacheManager({ cacheDir: directory, fetchImpl: fetch }); expect(() => m.getVariantFilePath('../wrong')).toThrow(); await expect(m.prefetchVariant('mediapipe:unknown')).rejects.toThrow(); expect(fetch).not.toHaveBeenCalled(); });
