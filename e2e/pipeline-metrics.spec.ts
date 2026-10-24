import { expect, test } from '@playwright/test';
interface PipelineCameraProbe { requests: number; frames: number; paused: boolean; dispose: () => void }
declare global { interface Window { pipelineCameraProbe: PipelineCameraProbe; pipelineGateResolve: () => void } }
test.use({ baseURL: 'http://127.0.0.1:4174' });
for (const mode of ['immutable', 'rgba', 'live-quality'] as const) {
  test(`pipeline details show measured units, expiry and lifecycle without toggle restarts (${mode})`, async ({ context, page }) => {
    await context.route('**/*', route => new URL(route.request().url()).origin === 'http://127.0.0.1:4174' ? route.continue() : route.abort());
    await context.addInitScript(({ mode }) => {
      window.visionSdkGate = new Promise<void>(resolve => { window.pipelineGateResolve = resolve; });
      if (mode !== 'immutable') Object.defineProperty(window, 'VideoFrame', { configurable: true, value: undefined });
      if (mode === 'rgba') {
        Object.defineProperty(HTMLVideoElement.prototype, 'getVideoPlaybackQuality', { configurable: true, value: () => ({ totalVideoFrames: 0 }) });
        Object.defineProperty(HTMLVideoElement.prototype, 'requestVideoFrameCallback', { configurable: true, value: undefined });
      }
      const streams: MediaStream[] = []; const intervals: number[] = [];
      const probe: PipelineCameraProbe = window.pipelineCameraProbe = { requests: 0, frames: 0, paused: false, dispose() {
        intervals.forEach(clearInterval); streams.forEach(stream => stream.getTracks().forEach(track => track.stop()));
      } };
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: async () => {
        probe.requests++; const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 480;
        const ctx = canvas.getContext('2d')!;
        const draw = () => { if (!probe.paused) { ctx.fillStyle = ++probe.frames % 2 ? '#123456' : '#abcdef'; ctx.fillRect(0, 0, 640, 480); } };
        draw(); const stream = canvas.captureStream(10); streams.push(stream); intervals.push(window.setInterval(draw, 100)); return stream;
      } } });
    }, { mode });
    const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
    await page.goto('/');
    try {
      const panel = page.getByLabel('パイプライン計測詳細', { exact: true });
      await expect(panel).not.toHaveAttribute('open');
      await expect(page.getByLabel('推論FPS', { exact: true })).toContainText('未計測');
      await panel.locator(':scope > summary').click();
      const table = panel.getByRole('table', { name: 'success interval statistics', exact: true });
      const row = (stage: string) => table.getByRole('row').filter({ has: page.getByRole('rowheader', { name: stage, exact: true }) });
      await expect(row('sdk').getByRole('cell').nth(1)).toHaveText('未計測');
      await page.evaluate(() => window.pipelineGateResolve());
      await expect(row('sdk').getByRole('cell').nth(1)).toHaveText(/\d+\.\d{2}/);
      await expect(table.getByRole('columnheader', { name: 'p95 ms', exact: true })).toBeVisible();
      await expect(panel.getByText(/CPU\/GPU使用率/)).toBeVisible();
      await expect(row('capture').getByRole('cell').nth(0)).toHaveText(/[1-9]\d* \/ [1-9]\d* \/ 0/);
      if (mode === 'live-quality') {
        await expect(row('latency').getByRole('cell').nth(1)).toHaveText('未計測');
        await expect(panel.getByText(/結果 outcomes/)).toContainText('unbound:');
      } else await expect(row('latency').getByRole('cell').nth(1)).toHaveText(/\d+\.\d{2}/);
      if (mode === 'rgba') await expect(row('comparison').getByRole('cell').nth(0)).toHaveText(/[1-9]\d* \/ [1-9]\d* \/ 0/);
      const before = await page.evaluate(() => ({ cameras: window.pipelineCameraProbe.requests, creates: window.visionSdkProbe.creates }));
      await panel.locator(':scope > summary').click();
      await page.screenshot({ path: `docs/verification/pr06/screenshots/normal-${mode}.png` });
      await panel.locator(':scope > summary').click();
      expect(await page.evaluate(() => ({ cameras: window.pipelineCameraProbe.requests, creates: window.visionSdkProbe.creates }))).toEqual(before);
      await page.screenshot({ path: `docs/verification/pr06/screenshots/details-${mode}.png` });
      await page.evaluate(() => { window.pipelineCameraProbe.paused = true; });
      await expect(row('sdk').getByRole('cell').nth(1)).toHaveText('未計測', { timeout: 10_000 });
      await expect(row('latency').getByRole('cell').nth(1)).toHaveText('未計測');
      await expect(page.getByLabel('推論FPS', { exact: true })).toContainText('未計測');
      // Duplicate RGBA polling still costs time while successful SDK timings expire.
      if (mode === 'rgba') await expect(row('comparison').getByRole('cell').nth(1)).toHaveText(/\d+\.\d{2}/);
      await page.getByRole('banner').getByRole('button', { name: '停止', exact: true }).click();
      await expect(page.getByLabel('性能状態')).toHaveText('停止中');
      await expect(row('capture').getByRole('cell').nth(0)).toHaveText('未計測');
      await page.evaluate(() => { window.pipelineCameraProbe.paused = false; });
      await page.getByRole('banner').getByRole('button', { name: 'カメラ開始', exact: true }).click();
      await expect(row('sdk').getByRole('cell').nth(1)).toHaveText(/\d+\.\d{2}/);
      expect(await page.evaluate(() => window.visionSdkProbe.creates)).toBe(before.creates);
      await page.getByRole('banner').getByRole('combobox', { name: 'モデル', exact: true }).selectOption('mediapipe');
      await expect(page.getByText('(dog)')).toBeVisible();
      expect(await page.evaluate(() => window.visionSdkProbe.creates)).toBe(before.creates + 1);
      await page.getByTitle('アウトカメラ（背面）に切り替え').click();
      await expect.poll(() => page.evaluate(() => window.pipelineCameraProbe.requests)).toBe(before.cameras + 2);
      await expect(row('sdk').getByRole('cell').nth(1)).toHaveText(/\d+\.\d{2}/);
      expect(await page.evaluate(() => window.visionSdkProbe.creates)).toBe(before.creates + 1);
      await panel.getByText('失敗・取消・旧世代の区間統計', { exact: true }).click();
      await page.evaluate(() => { window.visionSdkProbe.failures = 1; });
      const failedSdk = panel.getByRole('table', { name: 'failed interval statistics', exact: true }).getByRole('row').filter({ has: page.getByRole('rowheader', { name: 'sdk', exact: true }) });
      await expect(failedSdk.getByRole('cell').nth(0)).toHaveText(/[1-9]\d* \/ [1-9]\d* \/ 0/);
      await expect(failedSdk.getByRole('cell').nth(1)).toHaveText(/\d+\.\d{2}/);
      await expect(row('sdk').getByRole('cell').nth(1)).toHaveText(/\d+\.\d{2}/);
      expect(errors).toEqual([]);
    } finally { await page.evaluate(() => window.pipelineCameraProbe.dispose()); }
  });
}
