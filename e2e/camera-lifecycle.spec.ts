import { expect, test } from '@playwright/test';

interface CameraProbe {
  requests: Array<{ constraints: MediaStreamConstraints; resolve: () => void }>;
  streams: Record<number, MediaStream>;
  dispose: () => void;
}
declare global { interface Window { cameraProbe: CameraProbe } }

test('camera sessions survive UI renders and release stopped, replaced and late synthetic streams', async ({ context, page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await context.route('**/*', route => new URL(route.request().url()).origin === 'http://127.0.0.1:4173'
    ? route.continue() : route.abort('blockedbyclient'));
  // Only acquisition is controlled. The real built app, native MediaStream /
  // canvas-capture tracks, video srcObject, playback and track.stop all execute.
  await context.addInitScript(() => {
    const intervals: number[] = [];
    const canvases: HTMLCanvasElement[] = [];
    const probe: CameraProbe = {
      requests: [], streams: {},
      dispose() {
        intervals.forEach(id => clearInterval(id));
        Object.values(probe.streams).forEach(stream => stream.getTracks().forEach(track => track.stop()));
      },
    };
    window.cameraProbe = probe;
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: {
      getUserMedia: (constraints: MediaStreamConstraints) => new Promise<MediaStream>(resolve => {
        const id = probe.requests.length;
        probe.requests.push({ constraints, resolve() {
          const canvas = document.createElement('canvas');
          canvas.width = 640; canvas.height = 480;
          canvases.push(canvas);
          const ctx = canvas.getContext('2d')!;
          const stream = canvas.captureStream(10);
          const draw = () => { ctx.fillStyle = '#257354'; ctx.fillRect(0, 0, 640, 480); };
          draw();
          intervals.push(window.setInterval(draw, 100));
          probe.streams[id] = stream;
          resolve(stream);
        } });
      }),
    } });
  });
  await page.goto('/');
  const header = page.getByRole('banner');
  const video = page.locator('video');
  try {
    await expect.poll(() => page.evaluate(() => window.cameraProbe.requests.length)).toBe(1);
    await page.evaluate(() => window.cameraProbe.requests[0].resolve());
    await expect(page.getByLabel('カメラ状態')).toHaveText('カメラ映像を取得しました');
    await expect.poll(() => video.evaluate(v => (v as HTMLVideoElement).readyState)).toBeGreaterThanOrEqual(2);

    // Parent state/model initialization updates must not request another camera.
    await header.getByRole('combobox', { name: 'モデル', exact: true }).selectOption('mediapipe');
    await expect(page.getByRole('alert')).toContainText(/モデル(実行環境の読み込み|の生成)に失敗しました/);
    await header.getByRole('button', { name: '仕様', exact: true }).click();
    await page.getByRole('dialog').getByRole('button', { name: '閉じる', exact: true }).click();
    await page.getByRole('button', { name: '音声・設定' }).click();
    await page.locator('input[type=range]').nth(1).focus();
    await page.locator('input[type=range]').nth(1).press('ArrowRight');
    expect(await page.evaluate(() => window.cameraProbe.requests.length)).toBe(1);
    expect(await video.evaluate(v => (v as HTMLVideoElement).srcObject === window.cameraProbe.streams[0])).toBe(true);
    expect(await page.evaluate(() => window.cameraProbe.streams[0].getTracks().every(track => track.readyState === 'live'))).toBe(true);

    await header.getByRole('button', { name: '停止', exact: true }).click();
    await expect.poll(() => page.evaluate(() => window.cameraProbe.streams[0].getTracks().every(track => track.readyState === 'ended'))).toBe(true);
    expect(await video.evaluate(v => (v as HTMLVideoElement).srcObject)).toBeNull();

    // Stop a pending second request; start a third and complete it first.
    await header.getByRole('button', { name: 'カメラ開始', exact: true }).click();
    await expect.poll(() => page.evaluate(() => window.cameraProbe.requests.length)).toBe(2);
    await header.getByRole('button', { name: '停止', exact: true }).click();
    await header.getByRole('button', { name: 'カメラ開始', exact: true }).click();
    await expect.poll(() => page.evaluate(() => window.cameraProbe.requests.length)).toBe(3);
    await page.evaluate(() => window.cameraProbe.requests[2].resolve());
    await expect(page.getByLabel('カメラ状態')).toHaveText('カメラ映像を取得しました');
    await page.evaluate(() => window.cameraProbe.requests[1].resolve());
    await expect.poll(() => page.evaluate(() => window.cameraProbe.streams[1].getTracks().every(track => track.readyState === 'ended'))).toBe(true);
    expect(await video.evaluate(v => (v as HTMLVideoElement).srcObject === window.cameraProbe.streams[2])).toBe(true);
    expect(await page.evaluate(() => window.cameraProbe.streams[2].getTracks().every(track => track.readyState === 'live'))).toBe(true);

    // Rapid user -> environment -> user requests, also completed out of order.
    await page.getByTitle('アウトカメラ（背面）に切り替え').click();
    await expect.poll(() => page.evaluate(() => window.cameraProbe.requests.length)).toBe(4);
    expect(await page.evaluate(() => window.cameraProbe.requests[3].constraints.video)).toMatchObject({ facingMode: { ideal: 'environment' } });
    expect(await page.evaluate(() => window.cameraProbe.streams[2].getTracks().every(track => track.readyState === 'ended'))).toBe(true);
    await page.getByTitle('インカメラ（前面）に切り替え').click();
    await expect.poll(() => page.evaluate(() => window.cameraProbe.requests.length)).toBe(5);
    await page.evaluate(() => window.cameraProbe.requests[4].resolve());
    await expect(page.getByLabel('カメラ状態')).toHaveText('カメラ映像を取得しました');
    await page.evaluate(() => window.cameraProbe.requests[3].resolve());
    await expect.poll(() => page.evaluate(() => window.cameraProbe.streams[3].getTracks().every(track => track.readyState === 'ended'))).toBe(true);
    expect(await video.evaluate(v => (v as HTMLVideoElement).srcObject === window.cameraProbe.streams[4])).toBe(true);
    expect(await page.evaluate(() => window.cameraProbe.streams[4].getTracks().every(track => track.readyState === 'live'))).toBe(true);
    await header.getByRole('button', { name: '停止', exact: true }).click();
    await expect.poll(() => page.evaluate(() => Object.values(window.cameraProbe.streams).every(stream => stream.getTracks().every(track => track.readyState === 'ended')))).toBe(true);
    expect(await video.evaluate(v => (v as HTMLVideoElement).srcObject)).toBeNull();
    expect(pageErrors).toEqual([]);
  } finally {
    await page.evaluate(() => window.cameraProbe.dispose());
  }
});
