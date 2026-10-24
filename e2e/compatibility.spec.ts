import { test, expect } from '@playwright/test';
interface CompatibilityProbe { first: number[]; release: () => void; change: (color: string) => void; streams: MediaStream[] }
declare global { interface Window { compatibilityProbe: CompatibilityProbe } }
test.use({ baseURL: 'http://127.0.0.1:4174' });
for (const mode of ['immutable', 'quality', 'pixels-rvfc', 'pixels-raf', 'zero-quality-rvfc', 'zero-quality-raf'] as const) {
  test(`native still input ages and real pixel changes recover (${mode})`, async ({ context, page }, testInfo) => {
    const pixelMode = mode.includes('pixels') || mode.includes('zero-quality');
    const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
    await context.route('**/*', route => new URL(route.request().url()).origin === 'http://127.0.0.1:4174' ? route.continue() : route.abort());
    await context.addInitScript(mode => {
      const Frame = window.VideoFrame, rvfc = HTMLVideoElement.prototype.requestVideoFrameCallback, play = HTMLMediaElement.prototype.play;
      window.visionSdkObserverFrame = Frame; // SDK observer only; never visible through the product API.
      const probe: CompatibilityProbe = { first: [], release() {}, change() {}, streams: [] };
      window.compatibilityProbe = probe;
      window.visionSdkGate = new Promise<void>(resolve => { probe.release = resolve; });
      if (mode !== 'immutable') Object.defineProperty(window, 'VideoFrame', { configurable: true, value: undefined });
      if (mode !== 'immutable' && mode !== 'quality') Object.defineProperty(HTMLVideoElement.prototype, 'getVideoPlaybackQuality', {
        configurable: true, value: mode.startsWith('zero-quality') ? () => ({ totalVideoFrames: 0 }) : undefined,
      });
      if (mode.endsWith('raf') || mode === 'immutable' || mode === 'quality') Object.defineProperty(HTMLVideoElement.prototype, 'requestVideoFrameCallback', { configurable: true, value: undefined });
      HTMLMediaElement.prototype.play = function() {
        if (this instanceof HTMLVideoElement && this.srcObject instanceof MediaStream)
          rvfc.call(this, (_now, metadata) => { probe.first.push(metadata.mediaTime); });
        return play.call(this);
      };
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: async () => {
        const canvas = document.createElement('canvas'); canvas.width = 64; canvas.height = 64;
        const context = canvas.getContext('2d')!; context.fillStyle = 'red'; context.fillRect(0, 0, 64, 64);
        const stream = canvas.captureStream(0); probe.streams.push(stream);
        probe.change = color => { context.fillStyle = color; context.fillRect(0, 0, 64, 64); (stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack).requestFrame(); };
        setTimeout(() => (stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack).requestFrame(), 30);
        return stream;
      } } });
    }, mode);
    await page.goto('/');
    try {
      await expect.poll(() => page.evaluate(() => window.compatibilityProbe.first.length)).toBe(1);
      await page.evaluate(() => { window.visionSdkProbe.observeInputs = true; window.compatibilityProbe.release(); });
      await expect.poll(() => page.evaluate(() => window.visionSdkProbe.calls)).toBeGreaterThan(0);
      await page.waitForTimeout(pixelMode ? 2300 : 500);
      const still = await page.evaluate(() => ({ calls: window.visionSdkProbe.calls, input: window.visionSdkProbe.inputs[0],
        fps: document.querySelector('[aria-label="推論FPS"]')!.textContent, status: document.querySelector('[aria-label="性能状態"]')!.textContent }));
      expect(still.calls).toBe(1); expect(still.fps).toContain('未計測'); expect(still.input.pixels.slice(0, 4)).toEqual([255, 0, 0, 255]);
      if (pixelMode) expect(still.status).toBe('映像入力停滞');
      await page.evaluate(() => window.compatibilityProbe.change('blue'));
      await expect.poll(() => page.evaluate(() => window.visionSdkProbe.calls)).toBe(2);
      await page.waitForTimeout(350); expect(await page.evaluate(() => window.visionSdkProbe.calls)).toBe(2);
      expect(await page.evaluate(() => window.visionSdkProbe.inputs[1].pixels.slice(0, 4))).toEqual([0, 0, 255, 255]);
      if (pixelMode) await expect(page.getByLabel('性能状態')).toHaveText('計測中');
      expect(await page.evaluate(() => window.compatibilityProbe.streams.length)).toBe(1);
      expect(await page.evaluate(() => window.visionSdkProbe.creates)).toBe(1);
      await testInfo.attach(`compatibility-${mode}.json`, { body: JSON.stringify({ mode, callsAfterStill: still.calls, fpsAfterStill: still.fps,
        statusAfterStill: still.status, sourceKind: still.input.sourceKind, red: still.input.pixels.slice(0, 4), blue: [0, 0, 255, 255] }), contentType: 'application/json' });
      console.log('COMPATIBILITY', JSON.stringify({ mode, callsAfterStill: still.calls, fpsAfterStill: still.fps, statusAfterStill: still.status, sourceKind: still.input.sourceKind }));
      expect(errors).toEqual([]);
    } finally {
      await page.getByRole('banner').getByRole('button', { name: '停止', exact: true }).click();
      expect(await page.evaluate(() => window.compatibilityProbe.streams.every(stream => stream.getTracks().every(track => track.readyState === 'ended')))).toBe(true);
    }
  });
}
