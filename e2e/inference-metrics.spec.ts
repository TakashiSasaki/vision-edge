import { expect, test } from '@playwright/test';
interface InputProbe { requests: number; frames: number; callbacks: number; paused: boolean; streams: MediaStream[]; dispose: () => void }
declare global { interface Window { inputProbe: InputProbe } }
test.use({ baseURL: 'http://127.0.0.1:4174' });
for (const fallback of [false, true]) {
  test(`successful inference, measurements, idle and recovery with native video (${fallback ? 'RAF fallback' : 'rVFC'})`, async ({ context, page }) => {
    const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
    await context.route('**/*', route => new URL(route.request().url()).origin === 'http://127.0.0.1:4174' ? route.continue() : route.abort());
    await context.addInitScript(({ fallback }) => {
      const intervals: number[] = [];
      const probe: InputProbe = { requests: 0, frames: 0, callbacks: 0, paused: false, streams: [], dispose() {
        intervals.forEach(clearInterval); probe.streams.forEach(stream => stream.getTracks().forEach(track => track.stop()));
      } };
      window.inputProbe = probe;
      const original = HTMLVideoElement.prototype.requestVideoFrameCallback;
      Object.defineProperty(HTMLVideoElement.prototype, 'requestVideoFrameCallback', { configurable: true, value: fallback ? undefined : function(this: HTMLVideoElement, callback: VideoFrameRequestCallback) {
        probe.callbacks++; return original.call(this, callback);
      } });
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: async () => {
        probe.requests++;
        const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 480;
        const ctx = canvas.getContext('2d')!;
        const draw = () => { if (!probe.paused) { ctx.fillStyle = ++probe.frames % 2 ? '#257354' : '#123456'; ctx.fillRect(0, 0, 640, 480); } };
        draw(); const stream = canvas.captureStream(10); probe.streams.push(stream);
        intervals.push(window.setInterval(draw, 100)); return stream;
      } } });
    }, { fallback });
    await page.goto('/');
    try {
      await expect(page.getByLabel('推論FPS')).toContainText(/\d+\.\d+ FPS/);
      await expect(page.getByLabel('描画FPS')).toContainText(/\d+\.\d+ FPS/);
      await expect(page.getByLabel('フレーム処理時間')).toContainText(/\d+\.\d+ ms/);
      await expect(page.getByText('現在しきい値を超える物体はありません')).toBeVisible();
      expect(await page.evaluate(() => window.inputProbe.callbacks > 0)).toBe(!fallback);
      await page.getByRole('banner').getByRole('combobox', { name: 'モデル', exact: true }).selectOption('mediapipe');
      await expect(page.getByText('(dog)')).toBeVisible();
      expect(await page.evaluate(() => window.inputProbe.requests)).toBe(1);
      await page.evaluate(() => { window.inputProbe.paused = true; });
      await expect(page.getByLabel('性能状態')).toHaveText('映像入力停滞');
      const calls = await page.evaluate(() => window.visionSdkProbe.calls);
      await expect(page.getByLabel('推論FPS')).toContainText('未計測');
      await page.waitForTimeout(350);
      expect(await page.evaluate(() => window.visionSdkProbe.calls)).toBe(calls);
      await page.evaluate(() => { window.visionSdkProbe.failures = 1; window.visionSdkProbe.empty = true; window.inputProbe.paused = false; });
      await expect(page.getByText('(dog)')).toHaveCount(0);
      await expect(page.getByLabel('推論FPS')).toContainText(/\d+\.\d+ FPS/);
      const creates = await page.evaluate(() => window.visionSdkProbe.creates);
      await page.getByRole('banner').getByRole('button', { name: '停止', exact: true }).click();
      await expect(page.getByLabel('性能状態')).toHaveText('停止中');
      expect(await page.evaluate(() => window.inputProbe.streams.every(s => s.getTracks().every(t => t.readyState === 'ended')))).toBe(true);
      const stoppedCalls = await page.evaluate(() => window.visionSdkProbe.calls);
      await page.waitForTimeout(300); expect(await page.evaluate(() => window.visionSdkProbe.calls)).toBe(stoppedCalls);
      await page.getByRole('banner').getByRole('button', { name: 'カメラ開始', exact: true }).click();
      await expect(page.getByLabel('推論FPS')).toContainText(/\d+\.\d+ FPS/);
      expect(await page.evaluate(() => window.visionSdkProbe.creates)).toBe(creates);
      expect(errors).toEqual([]);
    } finally { await page.evaluate(() => window.inputProbe.dispose()); }
  });
}
