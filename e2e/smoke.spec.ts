import { expect, test } from '@playwright/test';

test('production app loads and model controls work without camera or external services', async ({ context, page }) => {
  const pageErrors: string[] = [];
  const localFailures: string[] = [];
  const externalRequests: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('response', response => {
    if (response.url().startsWith('http://127.0.0.1:4173/') && response.status() >= 400) {
      localFailures.push(`${response.status()} ${response.url()}`);
    }
  });
  page.on('requestfailed', request => {
    if (request.url().startsWith('http://127.0.0.1:4173/')) localFailures.push(request.url());
  });

  // Real local HTML, CSS and JS run unchanged. Every external resource is unavailable,
  // including fonts, MediaPipe WASM/loader and model assets; no CDN response is needed.
  await context.route('**/*', route => {
    const url = new URL(route.request().url());
    if (url.origin === 'http://127.0.0.1:4173') return route.continue();
    externalRequests.push(url.href);
    return route.abort('blockedbyclient');
  });
  await context.addInitScript(() => {
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: {
        getUserMedia: () => Promise.reject(new DOMException('Smoke test camera denied', 'NotAllowedError')),
      },
    });
  });

  await page.goto('/');
  await expect(page).toHaveTitle('Webcam Vision & Segmentation Studio');
  const header = page.getByRole('banner');
  await expect(header.getByText('VisionStudio AI')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'カメラエラー' })).toBeVisible();
  // Observe the real loader reaching the blocked network boundary before asserting errors.
  await expect.poll(() => externalRequests.some(url => url.includes('cdn.jsdelivr.net'))).toBe(true);

  const model = header.getByRole('combobox').nth(0);
  await expect(model).toHaveValue('selfie-segmenter');
  await expect(model.getByRole('option')).toHaveCount(4);
  await model.selectOption('mediapipe');
  await expect(model).toHaveValue('mediapipe');
  await header.getByRole('button', { name: '仕様', exact: true }).click();
  await expect(page.getByRole('heading', { name: '物体認識・セグメンテーションモデル仕様' })).toBeVisible();
  await page.getByText('YOLO-seg (YOLOv8/11-seg)', { exact: true }).click();
  await expect(model).toHaveValue('yolo-seg');
  await expect(page.getByRole('heading', { name: '物体認識・セグメンテーションモデル仕様' })).toBeHidden();
  await expect(header).toBeVisible();
  expect(localFailures).toEqual([]);
  expect(pageErrors).toEqual([]);
});
