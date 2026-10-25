// Model-free controls using actual services/engine and a recording ORT boundary.
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, readFileSync } from 'node:fs';
import path from 'node:path';
import type * as ort from 'onnxruntime-web';
import { createYoloSegTask, YoloSegEngine } from './yoloSegService';
import { createSegFormerTask, SegFormerEngine } from './segformerService';
import { VisionEngine, type VisionTaskFactory } from './visionEngine';
import type { PipelineObserver } from './pipelineMetrics';
import { backendDescription } from './modelRegistry';
const boundary = vi.hoisted(() => ({ create: vi.fn() }));
vi.mock('onnxruntime-web', () => ({ env: { wasm: {}, logLevel: 'error' }, InferenceSession: { create: boundary.create }, Tensor: class {
        constructor(public type: string, public data: Float32Array, public dims: number[]) { }
        dispose = vi.fn();
    } }));
const pixels = new WeakMap<HTMLCanvasElement, ImageData>();
afterEach(() => vi.unstubAllGlobals());
function canvasContext(canvas: HTMLCanvasElement) { return { drawImage: vi.fn(), createImageData: (w: number, h: number) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }), putImageData: (image: ImageData) => pixels.set(canvas, image), getImageData: (x: number, y: number, w: number, h: number) => { const a = new Uint8ClampedArray(w * h * 4); const old = pixels.get(canvas); for (let j = 0; j < h; j++)
        for (let i = 0; i < w; i++) {
            const p = (j * w + i) * 4;
            if (old)
                a.set(old.data.subarray(((y + j) * old.width + x + i) * 4, ((y + j) * old.width + x + i) * 4 + 4), p);
            else
                a.set([255, 128, 0, 255], p);
        } return { width: w, height: h, data: a }; } } as unknown as CanvasRenderingContext2D; }
beforeEach(() => {
    vi.spyOn(console, 'log').mockImplementation(() => { });
    vi.spyOn(console, 'warn').mockImplementation(() => { });
    vi.spyOn(console, 'error').mockImplementation(() => { });
    boundary.create.mockReset();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (this: HTMLCanvasElement) { return canvasContext(this); });
});
function input() { const c = document.createElement('canvas'); c.width = 640; c.height = 480; return c; }
function deferred<T>() { let resolve!: (v: T) => void; let reject!: (e: Error) => void; const promise = new Promise<T>((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; }
function yolo(cx = 320, cy = 240, w = 128, h = 96, score = .9) { const data = new Float32Array(116); data.set([cx, cy, w, h]); data[4] = score; data[84] = 10; const proto = new Float32Array(32 * 20 * 20); proto.fill(1, 0, 400); return { output0: { dims: [1, 116, 1], data }, output1: { dims: [1, 32, 20, 20], data: proto } }; }
function logits(selected = 0, uniform = false) { const data = new Float32Array(150 * 100); if (!uniform)
    data.fill(10, selected * 100, (selected + 1) * 100); return { logits: { dims: [1, 150, 10, 10], data } }; }
function session(outputs: object, kind: 'yolo' | 'seg' = 'yolo') { return { inputNames: [kind === 'yolo' ? 'images' : 'pixel_values'], outputNames: kind === 'yolo' ? ['output0', 'output1'] : ['logits'], run: vi.fn(async () => outputs), release: vi.fn(async () => { }) }; }
const asSession = (s: ReturnType<typeof session>) => s as unknown as ort.InferenceSession;
function engineWith(task: YoloSegEngine | SegFormerEngine, model: 'yolo-seg' | 'segformer') {
    const f: VisionTaskFactory = { load: async () => ({ wasmLoaderPath: 'unused', wasmBinaryPath: 'unused' }), detector: async () => ({ detectForVideo: () => ({ detections: [] }), close: vi.fn() }), segmenter: async () => ({ segmentForVideo: () => ({ confidenceMasks: [], categoryMask: undefined, close: vi.fn() }), close: vi.fn() }) };
    if (model === 'yolo-seg')
        f.yoloSeg = async () => task as YoloSegEngine;
    else
        f.segformer = async () => task as SegFormerEngine;
    return new VisionEngine(f);
}
it.each(['yolo-seg', 'segformer'] as const)('F2 %s propagates actual run rejection, clears scene, and marks failed probes without destroying the model', async (model) => {
    const s = session(model === 'yolo-seg' ? yolo() : logits(), model === 'yolo-seg' ? 'yolo' : 'seg');
    const task = model === 'yolo-seg' ? new YoloSegEngine(asSession(s)) : new SegFormerEngine(asSession(s));
    const e = engineWith(task, model);
    await e.initialize(model);
    const good = await e.detectVideoFrame(input(), model, .3);
    expect(good.status).toBe('ready');
    const events: Array<{
        stage: string;
        outcome: string;
    }> = [];
    const observer: PipelineObserver = { now: () => performance.now(), record: r => events.push({ stage: r.stage, outcome: r.outcome }) };
    s.run.mockRejectedValueOnce(new Error('REVIEW_ORT_FAILURE'));
    const failure = e.detectVideoFrame(input(), model, .3, observer);
    await expect(failure).rejects.toThrow('REVIEW_ORT_FAILURE');
    await expect(failure).rejects.toMatchObject({ name: 'OnnxInferenceError', stage: 'run' });
    expect(events).toContainEqual({ stage: 'sdk', outcome: 'failed' });
    expect(events).toContainEqual({ stage: 'engine', outcome: 'failed' });
    expect(events).not.toContainEqual({ stage: 'engine', outcome: 'success' });
    expect(e.getState().status).toBe('ready');
    expect(s.release).not.toHaveBeenCalled();
    expect(e.findObjectAtPoint(.5, .5, [])).toBeNull();
    if (task instanceof SegFormerEngine)
        expect(task.getPixelClass(.5, .5)).toBeNull();
    if (model === 'yolo-seg') {
        s.run.mockResolvedValueOnce(yolo(320, 240, 128, 96, 0));
        const empty = await e.detectVideoFrame(input(), model, .3);
        expect(empty).toMatchObject({ status: 'ready', objects: [] });
    }
    e.dispose();
});
it.each(['yolo', 'seg'] as const)('positive %s preprocessing keeps the supplied image tensor contract', async (kind) => { const s = session(kind === 'yolo' ? yolo() : logits(), kind); const task = kind === 'yolo' ? new YoloSegEngine(asSession(s)) : new SegFormerEngine(asSession(s)); await task.detectForVideo(input(), .3); const feeds = s.run.mock.calls[0] as unknown as [
    Record<string, {
        dims: number[];
        data: Float32Array;
    }>
]; const t = Object.values(feeds[0])[0]; const area = kind === 'yolo' ? 640 * 480 : 512 * 512; expect(t.dims).toEqual(kind === 'yolo' ? [1, 3, 480, 640] : [1, 3, 512, 512]); expect(t.data[0]).toBeCloseTo(kind === 'yolo' ? 1 : (1 - .485) / .229, 5); expect(t.data[area]).toBeCloseTo(kind === 'yolo' ? 128 / 255 : (128 / 255 - .456) / .224, 5); expect(t.data[2 * area]).toBeCloseTo(kind === 'yolo' ? 0 : -.406 / .225, 5); await task.close(); });
it.each([
    [0, 240, 128, 96, { x: 0, y: .4, width: .1, height: .2 }],
    [640, 240, 128, 96, { x: .9, y: .4, width: .1, height: .2 }],
    [320, 0, 128, 96, { x: .4, y: 0, width: .2, height: .1 }],
    [320, 480, 128, 96, { x: .4, y: .9, width: .2, height: .1 }],
    [0, 0, 128, 96, { x: 0, y: 0, width: .1, height: .1 }],
    [640, 480, 128, 96, { x: .9, y: .9, width: .1, height: .1 }],
] as const)('F3 clips YOLO endpoints and mask at cx=%s cy=%s', async (cx, cy, w, h, box) => { const s = session(yolo(cx, cy, w, h)); const task = new YoloSegEngine(asSession(s)); const [o] = await task.detectForVideo(input(), .3); for (const k of ['x', 'y', 'width', 'height'] as const)
    expect(o.bbox[k]).toBeCloseTo(box[k], 5); const image = pixels.get(o.maskCanvas!)!; let count = 0; for (let py = 0; py < 20; py++)
    for (let px = 0; px < 20; px++)
        if (image.data[(py * 20 + px) * 4 + 3]) {
            count++;
            expect((px + .5) / 20).toBeGreaterThanOrEqual(box.x - 1e-6);
            expect((px + .5) / 20).toBeLessThan(box.x + box.width + 1e-6);
            expect((py + .5) / 20).toBeGreaterThanOrEqual(box.y - 1e-6);
            expect((py + .5) / 20).toBeLessThan(box.y + box.height + 1e-6);
        } expect(count).toBeGreaterThan(0); await task.close(); });
it('F3 entirely offscreen boxes produce a legitimate empty result', async () => { const s = session(yolo(-200, 240, 128, 96)); const t = new YoloSegEngine(asSession(s)); expect(await t.detectForVideo(input(), .3)).toEqual([]); await t.close(); });
it('F4 uniform logits have low softmax score, no dominant-class threshold exemption, and retain a full semantic mask', async () => { const s = session(logits(0, true), 'seg'); const e = engineWith(new SegFormerEngine(asSession(s)), 'segformer'); await e.initialize('segformer'); const frame = await e.detectVideoFrame(input(), 'segformer', .9); expect(frame).toMatchObject({ status: 'ready', objects: [] }); expect(frame).toHaveProperty('semanticMask'); if (frame.status !== 'ready')
    throw Error('expected frame'); const hit = e.findObjectAtPoint(.5, .5, frame.objects); expect(hit?.confidence).toBeCloseTo(1 / 150, 6); expect(hit).toHaveProperty('confidenceKind', 'semantic-softmax-pixel'); expect(hit).toHaveProperty('areaFraction', 1); e.dispose(); });
it('F4 equal-area strongly separated classes both obey softmax score threshold', async () => { const out = logits(); out.logits.data.fill(-10); out.logits.data.fill(10, 0, 50); out.logits.data.fill(10, 100 + 50, 200); const s = session(out, 'seg'); const t = new SegFormerEngine(asSession(s)); const objects = await t.detectForVideo(input(), .9); expect(objects).toHaveLength(2); for (const o of objects) {
    expect(o.confidence).toBeGreaterThan(.99);
    expect(o).toHaveProperty('areaFraction', .5);
    expect(o).toHaveProperty('confidenceKind', 'semantic-softmax-mean');
} await t.close(); });
it('F5 clear invalidates the inspector even after an accepted frame', async () => { const s = session(logits(), 'seg'); const t = new SegFormerEngine(asSession(s)); const e = engineWith(t, 'segformer'); await e.initialize('segformer'); const frame = await e.detectVideoFrame(input(), 'segformer', .3); if (frame.status !== 'ready')
    throw Error('expected frame'); expect(e.findObjectAtPoint(.5, .5, frame.objects)?.label).toBe('wall'); e.clearDetections(); expect(e.findObjectAtPoint(.5, .5, [])).toBeNull(); expect(e.findObjectAtPoint(.5, .5, frame.objects)).toBeNull(); expect(t.getPixelClass(.5, .5)).toBeNull(); e.dispose(); });
it('F5 clearing an in-flight first result cannot publish its person grid', async () => { const s = session(logits(), 'seg'); const p = deferred<object>(); s.run.mockImplementationOnce(() => p.promise); const t = new SegFormerEngine(asSession(s)); const e = engineWith(t, 'segformer'); await e.initialize('segformer'); const pending = e.detectVideoFrame(input(), 'segformer', .3); e.clearDetections(); p.resolve(logits(12)); expect(await pending).toMatchObject({ status: 'not-ready' }); expect(e.findObjectAtPoint(.5, .5, [])).toBeNull(); expect(t.getPixelClass(.5, .5)).toBeNull(); e.dispose(); });
it('F5 pixel inspection retains small classes absent from prominent objects', async () => { const out = logits(); out.logits.data[0] = 0; out.logits.data[12 * 100] = 20; const s = session(out, 'seg'); const e = engineWith(new SegFormerEngine(asSession(s)), 'segformer'); await e.initialize('segformer'); const r = await e.detectVideoFrame(input(), 'segformer', .3); if (r.status !== 'ready')
    throw Error('expected ready'); expect(r.objects.some(o => o.label === 'person')).toBe(false); expect(e.findObjectAtPoint(.01, .01, r.objects)).toMatchObject({ label: 'person', confidenceKind: 'semantic-softmax-pixel' }); e.dispose(); });
it.each(['yolo-seg', 'segformer'] as const)('F6 %s WASM configuration is not reported as GPU hardware', async (model) => { vi.stubGlobal('navigator', {}); const s = session(model === 'yolo-seg' ? yolo() : logits(), model === 'yolo-seg' ? 'yolo' : 'seg'); boundary.create.mockResolvedValue(asSession(s)); const e = new VisionEngine(); const r = await e.initialize(model); expect(r.status).toBe('ready'); expect(boundary.create.mock.calls[0][1].executionProviders).toEqual(['wasm']); expect(r).not.toHaveProperty('delegate', 'GPU'); expect(r).toHaveProperty('backend.actualBackend', 'unmeasured'); expect(backendDescription(e.getState())).toContain('WASM'); e.dispose(); vi.unstubAllGlobals(); });
it.each([createYoloSegTask, createSegFormerTask])('F6 records observed WebGPU creation failure and explicit WASM retry', async (create) => { vi.stubGlobal('navigator', { gpu: {} }); const s = session({}); boundary.create.mockRejectedValueOnce(new Error('WEBGPU_CREATE_FAILED')).mockResolvedValueOnce(asSession(s)); const t = await create(); expect(boundary.create.mock.calls.map(c => c[1].executionProviders)).toEqual([['webgpu', 'wasm'], ['wasm']]); expect(t).toHaveProperty('backend.fallbackReason', expect.stringContaining('WEBGPU_CREATE_FAILED')); expect(t).toHaveProperty('backend.actualBackend', 'unmeasured'); await t.close(); vi.unstubAllGlobals(); });
it.each(['yolo-seg', 'segformer'] as const)('F7 %s close waits for the in-flight run and releases exactly once', async (model) => { const p = deferred<object>(); const s = session({}); s.run.mockImplementationOnce(() => p.promise); const t = model === 'yolo-seg' ? new YoloSegEngine(asSession(s)) : new SegFormerEngine(asSession(s)); const running = t.detectForVideo(input(), .3); const closed = t.close(); const again = t.close(); expect(s.release).not.toHaveBeenCalled(); p.resolve(model === 'yolo-seg' ? yolo() : logits()); await running; await Promise.all([closed, again]); expect(s.release).toHaveBeenCalledTimes(1); });
it('positive model-switch generation guard keeps a late SegFormer result not-ready and releases after the run', async () => { const p = deferred<object>(); const s = session({}, 'seg'); s.run.mockImplementationOnce(() => p.promise); const e = engineWith(new SegFormerEngine(asSession(s)), 'segformer'); await e.initialize('segformer'); const pending = e.detectVideoFrame(input(), 'segformer', .3); await e.initialize('mediapipe'); expect(s.release).not.toHaveBeenCalled(); p.resolve(logits(12)); expect(await pending).toMatchObject({ status: 'not-ready' }); expect(e.getState()).toMatchObject({ status: 'ready', model: 'mediapipe' }); expect(s.release).toHaveBeenCalledTimes(1); e.dispose(); });
it('F7 async release failures are handled by real service owners, not unhandled in the process', () => {
    const dir = mkdtempSync(path.resolve('.cache/ort-release-'));
    const file = path.join(dir, 'check.mts');
    writeFileSync(file, `import {YoloSegEngine} from ${JSON.stringify(path.resolve('src/services/yoloSegService.ts'))};import {SegFormerEngine} from ${JSON.stringify(path.resolve('src/services/segformerService.ts'))};let unhandled=0;process.on('unhandledRejection',()=>unhandled++);let releases=0;for(const Task of [YoloSegEngine,SegFormerEngine]){const t=new Task({release:async()=>{releases++;throw Error('RELEASE_FAILED_SENTINEL')}});await t.close();await t.close();}await new Promise(r=>setTimeout(r,40));console.log(JSON.stringify({unhandled,releases}));process.exitCode=unhandled?1:0;`);
    const r = spawnSync(process.execPath, ['--import', 'tsx', file], { encoding: 'utf8', timeout: 10000 });
    writeFileSync(path.join(dir, 'stdout.log'), r.stdout ?? '');
    writeFileSync(path.join(dir, 'stderr.log'), r.stderr ?? '');
    expect(r.error).toBeUndefined();
    expect(r.status, `${r.stderr}\nlogs: ${dir}`).toBe(0);
    expect(JSON.parse(readFileSync(path.join(dir, 'stdout.log'), 'utf8'))).toEqual({ unhandled: 0, releases: 2 });
    expect(r.stderr).toContain('RELEASE_FAILED_SENTINEL');
});
it.each(['yolo-seg', 'segformer'] as const)('F7 %s close and run rejection settle ownership without releasing twice', async (model) => { const p = deferred<object>(); const s = session({}); s.run.mockImplementationOnce(() => p.promise); const t = model === 'yolo-seg' ? new YoloSegEngine(asSession(s)) : new SegFormerEngine(asSession(s)); const running = t.detectForVideo(input(), .3); const rejection = expect(running).rejects.toThrow('RUN_DURING_CLOSE_FAILED'); const closing = t.close(); expect(s.release).not.toHaveBeenCalled(); p.reject(new Error('RUN_DURING_CLOSE_FAILED')); await rejection; await closing; await t.close(); expect(s.release).toHaveBeenCalledTimes(1); });
it('F3 instance masks keep their paired box instead of box-only EMA drift', async () => { const s = session(yolo()); const e = engineWith(new YoloSegEngine(asSession(s)), 'yolo-seg'); await e.initialize('yolo-seg'); await e.detectVideoFrame(input(), 'yolo-seg', .3); s.run.mockResolvedValueOnce(yolo(0, 240)); const r = await e.detectVideoFrame(input(), 'yolo-seg', .3); if (r.status !== 'ready')
    throw Error('expected frame'); expect(r.objects[0].bbox.x).toBe(0); expect(r.objects[0].bbox.width).toBeCloseTo(.1); e.dispose(); });
it('F2 legitimate YOLO empty frames retain only the existing bounded two-frame policy', async () => { const s = session(yolo()); const e = engineWith(new YoloSegEngine(asSession(s)), 'yolo-seg'); await e.initialize('yolo-seg'); await e.detectVideoFrame(input(), 'yolo-seg', .3); s.run.mockResolvedValue(yolo(320, 240, 128, 96, 0)); for (let i = 0; i < 2; i++) {
    const r = await e.detectVideoFrame(input(), 'yolo-seg', .3);
    if (r.status !== 'ready')
        throw Error('expected ready');
    expect(r.objects).toHaveLength(1);
    expect(r.objects[0].confidence).toBeCloseTo(.9 * .85);
} expect(await e.detectVideoFrame(input(), 'yolo-seg', .3)).toMatchObject({ status: 'ready', objects: [] }); e.dispose(); });
it.each(['yolo', 'seg'] as const)('F2 %s missing outputs fail rather than becoming valid empty inference', async (kind) => { const s = session({}, kind); const t = kind === 'yolo' ? new YoloSegEngine(asSession(s)) : new SegFormerEngine(asSession(s)); await expect(t.detectForVideo(input(), .3)).rejects.toMatchObject({ name: 'OnnxInferenceError', stage: 'output' }); await t.close(); });
it('F5 immutable mask snapshots and inspector identity stay paired across accepted frames', async () => { const s = session(logits(), 'seg'); const e = engineWith(new SegFormerEngine(asSession(s)), 'segformer'); await e.initialize('segformer'); const first = await e.detectVideoFrame(input(), 'segformer', .3); if (first.status !== 'ready' || !first.semanticMask)
    throw Error('expected mask'); const bytes = new Uint8ClampedArray(pixels.get(first.semanticMask)!.data); s.run.mockResolvedValueOnce(logits(12)); const second = await e.detectVideoFrame(input(), 'segformer', .3); if (second.status !== 'ready')
    throw Error('expected ready'); expect(second.semanticMask).not.toBe(first.semanticMask); expect(pixels.get(first.semanticMask)!.data).toEqual(bytes); expect(e.findObjectAtPoint(.5, .5, first.objects)).toBeNull(); expect(e.findObjectAtPoint(.5, .5, second.objects)?.label).toBe('person'); e.dispose(); });
it.each([createYoloSegTask, createSegFormerTask])('F6 explicit WASM remains a request, not an observed hardware claim', async (create) => { vi.stubGlobal('navigator', { gpu: {} }); boundary.create.mockResolvedValue(asSession(session({}))); const t = await create({ delegate: 'WASM' }); expect(t).toHaveProperty('backend.requestedProvider', 'wasm'); expect(t).toHaveProperty('backend.configuredProviders', ['wasm']); expect(t).toHaveProperty('backend.fallbackReason', null); expect(t).toHaveProperty('backend.actualBackend', 'unmeasured'); expect(boundary.create).toHaveBeenCalledTimes(1); await t.close(); });
it.each([createYoloSegTask, createSegFormerTask])('F6 WebGPU preference records configured alternatives without asserting the actual provider', async (create) => { vi.stubGlobal('navigator', { gpu: {} }); boundary.create.mockResolvedValue(asSession(session({}))); const t = await create(); expect(t).toHaveProperty('backend.requestedProvider', 'auto'); expect(t).toHaveProperty('backend.configuredProviders', ['webgpu', 'wasm']); expect(t).toHaveProperty('backend.actualBackend', 'unmeasured'); expect(t).toHaveProperty('backend.fallbackReason', null); await t.close(); });
it('F7 engine handles asynchronous close rejection from an injected task without corrupting the next model', async () => { const task = new YoloSegEngine(asSession(session(yolo()))); const e = engineWith(task, 'yolo-seg'); vi.spyOn(task, 'close').mockImplementation(async () => { throw Error('INJECTED_CLOSE_REJECT'); }); await e.initialize('yolo-seg'); await e.initialize('mediapipe'); await Promise.resolve(); expect(e.getState()).toMatchObject({ status: 'ready', model: 'mediapipe' }); expect(console.warn).toHaveBeenCalledWith('Vision task cleanup failed', expect.objectContaining({ model: 'yolo-seg', cause: expect.objectContaining({ message: 'INJECTED_CLOSE_REJECT' }) })); e.dispose(); });
