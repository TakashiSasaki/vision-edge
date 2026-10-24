import { expect, type BrowserContext, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { manifest, assetPath, verifyAsset } from '../scripts/real-sdk/assets.mjs';
import type { probe } from './observer';
export type InputMode = 'VideoFrame' | 'Canvas' | 'HTMLVideoElement';
type Camera = { scene: string; frames: number; requests: number; framesCreated: number; framesClosed: number; doubleCloses: number; liveTracks: () => number; dispose: () => void };
declare global { interface Window { __realCamera: Camera; __realSdkProbe: typeof probe } }
export async function routeAssets(context: BrowserContext, failModel = false) {
  const requests: Array<{ url: string; status: string }> = [];
  await context.route('**/*', async route => {
    const url = route.request().url();
    const asset = manifest.assets.find(asset => asset.url === url || url === `http://127.0.0.1:4175/fixtures/${asset.file}`);
    if (asset) {
      if (failModel && asset.file.endsWith('.tflite')) {
        requests.push({ url, status: 'controlled HTTP503' });
        await route.fulfill({ status: 503, body: 'PR07 controlled asset failure', headers: { 'Access-Control-Allow-Origin': '*' } }); return;
      }
      const checked = await verifyAsset(asset);
      if (checked.status !== 'pass') { requests.push({ url, status: checked.status }); await route.abort('failed'); return; }
      requests.push({ url, status: 'verified local bytes' });
      await route.fulfill({ body: await readFile(assetPath(asset)), contentType: asset.file.endsWith('.wasm') ? 'application/wasm' : asset.file.endsWith('.js') ? 'application/javascript' : asset.file.endsWith('.png') ? 'image/png' : 'application/octet-stream', headers: { 'Access-Control-Allow-Origin': '*' } }); return;
    }
    if (new URL(url).origin === 'http://127.0.0.1:4175') await route.continue();
    else { requests.push({ url, status: 'external network blocked' }); await route.abort('blockedbyclient'); }
  });
  return requests;
}
export async function installCamera(context: BrowserContext, mode: InputMode, initialScene: string) {
  await context.addInitScript(({ mode, initialScene }) => {
    const streams: MediaStream[] = []; const intervals: number[] = []; const closed = new WeakSet<VideoFrame>();
    const camera: Camera = window.__realCamera = { scene: initialScene, frames: 0, requests: 0, framesCreated: 0, framesClosed: 0, doubleCloses: 0,
      liveTracks: () => streams.flatMap(s => s.getTracks()).filter(t => t.readyState === 'live').length,
      dispose: () => { intervals.forEach(clearInterval); streams.forEach(s => s.getTracks().forEach(t => t.stop())); } };
    if (mode !== 'VideoFrame') Object.defineProperty(window, 'VideoFrame', { configurable: true, value: undefined });
    else {
      const Native = VideoFrame;
      Native.prototype.close = new Proxy(Native.prototype.close, { apply(target, receiver, args) {
        if (closed.has(receiver)) camera.doubleCloses++; else { closed.add(receiver); camera.framesClosed++; }
        return Reflect.apply(target, receiver, args);
      } });
      window.VideoFrame = new Proxy(Native, { construct(target, args) { const frame = Reflect.construct(target, args); camera.framesCreated++; return frame; } });
    }
    if (mode === 'Canvas') {
      Object.defineProperty(HTMLVideoElement.prototype, 'getVideoPlaybackQuality', { configurable: true, value: () => ({ totalVideoFrames: 0 }) });
      Object.defineProperty(HTMLVideoElement.prototype, 'requestVideoFrameCallback', { configurable: true, value: undefined });
    }
    Object.defineProperty(navigator.mediaDevices, 'getUserMedia', { configurable: true, value: async () => {
      camera.requests++;
      const images: Record<string, HTMLImageElement> = {};
      for (const name of ['astronaut.png', 'coffee.png']) {
        const image = new Image(); image.src = `/fixtures/${name}`; await image.decode(); images[name] = image;
      }
      const canvas = document.createElement('canvas'); canvas.width = 512; canvas.height = 512;
      const ctx = canvas.getContext('2d')!;
      const draw = () => {
        if (images[camera.scene]) ctx.drawImage(images[camera.scene], 0, 0, 512, 512);
        else {
          ctx.fillStyle = '#080808'; ctx.fillRect(0, 0, 512, 512);
          if (camera.scene === 'background') for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) { ctx.fillStyle = (x + y) % 2 ? '#283848' : '#8193a5'; ctx.fillRect(x * 32, y * 32, 32, 32); }
        }
        // Distinct native frames at 5Hz, not physical-camera FPS evidence.
        ctx.fillStyle = ++camera.frames % 2 ? '#000000' : '#ffffff'; ctx.fillRect(511, 511, 1, 1);
      };
      draw(); const stream = canvas.captureStream(5); streams.push(stream);
      intervals.push(window.setInterval(() => { if (stream.getTracks().some(t => t.readyState === 'live')) draw(); }, 200));
      return stream;
    } });
  }, { mode, initialScene });
}
export async function runtimeEvidence(page: Page) {
  const evidence = await page.evaluate(() => {
    const canvas = document.createElement('canvas'); const gl = canvas.getContext('webgl2') ?? canvas.getContext('webgl');
    const extension = gl?.getExtension('WEBGL_debug_renderer_info');
    const runtime = { userAgent: navigator.userAgent, platform: navigator.platform, webgl: !!gl,
      vendor: gl && extension ? gl.getParameter(extension.UNMASKED_VENDOR_WEBGL) : null,
      renderer: gl && extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : null,
      physicalGpu: 'unknown; GPU delegate request is not hardware proof', physicalCamera: 'not-run',
      videoFrame: typeof VideoFrame, resolution: [512, 512], nativeQuality: document.querySelector('video')?.getVideoPlaybackQuality?.()?.totalVideoFrames };
    gl?.getExtension('WEBGL_lose_context')?.loseContext(); return runtime;
  });
  return { ...evidence, browserVersion: page.context().browser()?.version() ?? 'unknown (disconnected browser)' };
}
export async function ready(page: Page) { await expect(page.getByRole('status')).toContainText('モデル準備完了', { timeout: 30_000 }); }
export async function release(page: Page) {
  if (page.isClosed()) return;
  const stop = page.getByRole('banner').getByRole('button', { name: '停止', exact: true });
  if (await stop.count()) await stop.click();
  await page.evaluate(() => { window.__realCamera?.dispose(); window.__realSdkProbe?.stop(); });
}
