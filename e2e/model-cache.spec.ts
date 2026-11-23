import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.route('**/*', (route) =>
    new URL(route.request().url()).origin === 'http://127.0.0.1:4173'
      ? route.continue()
      : route.abort('blockedbyclient')
  );
});

test('browser cache inventory: displays Cache Storage entries, deletes single/all items, fits 320px mobile, and links from main header', async ({
  page,
}, info) => {
  await page.setViewportSize({ width: 320, height: 640 });
  const externalModelRequests: string[] = [];
  page.on('request', (r) => {
    if (r.url().endsWith('.tflite') || r.url().endsWith('.onnx')) {
      externalModelRequests.push(r.url());
    }
  });

  await page.goto('/model-cache.html');
  await expect(page.getByRole('status')).toContainText('ブラウザ保存済み 0件');
  await expect(page.locator('#page-url')).toContainText('/model-cache.html');
  await expect(page.getByRole('button', { name: 'ページURLをクリップボードにコピー' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'キャッシュをすべて削除' })).toBeDisabled();

  // Seed two model binaries in the browser's real Cache Storage ('vision-studio-models-v1')
  await page.evaluate(async () => {
    const cache = await caches.open('vision-studio-models-v1');
    await cache.put(
      'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite',
      new Response(new Uint8Array(2048), {
        status: 200,
        headers: {
          'Content-Type': 'application/octet-stream',
          'Content-Length': '2048',
          'X-Vision-Cached-At': '2026-11-16T05:00:00.000Z',
          'X-Vision-Sha256': '6c7ab0a6e5dcbf38a8c33b960996a55a3b4300b36a018c4545801de3a3c8bde0',
        },
      })
    );
    await cache.put(
      'http://127.0.0.1:4173/api/cors-proxy/stereo/hitnet_sceneflow_240x320.onnx',
      new Response(new Uint8Array(4096), {
        status: 200,
        headers: {
          'Content-Type': 'application/octet-stream',
          'Content-Length': '4096',
          'X-Vision-Cached-At': '2026-11-16T05:05:00.000Z',
        },
      })
    );
  });

  await page.getByRole('button', { name: '状態を更新' }).click();
  await expect(page.getByRole('status')).toContainText('ブラウザ保存済み 2件');
  await expect(page.locator('#cached')).toContainText('efficientnet_lite0.tflite');
  await expect(page.locator('#cached')).toContainText('hitnet_sceneflow_240x320.onnx');
  await expect(page.locator('#cached')).toContainText('6c7ab0a6e5dc...');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(externalModelRequests).toEqual([]);

  // Delete single entry
  await page
    .getByRole('button', { name: 'efficientnet_lite0.tflite をキャッシュから削除' })
    .click();
  await expect(page.getByRole('status')).toContainText('ブラウザ保存済み 1件');
  await expect(page.locator('#cached')).not.toContainText('efficientnet_lite0.tflite');
  await expect(page.locator('#cached')).toContainText('hitnet_sceneflow_240x320.onnx');

  // Clear all entries
  await page.getByRole('button', { name: 'キャッシュをすべて削除' }).click();
  await expect(page.getByRole('status')).toContainText('ブラウザ保存済み 0件');
  await expect(page.getByRole('button', { name: 'キャッシュをすべて削除' })).toBeDisabled();

  await page.screenshot({ path: info.outputPath('cache-inventory-320.png'), fullPage: true });

  await page.goto('/');
  await expect(page.getByRole('link', { name: 'ブラウザのモデルキャッシュ一覧' })).toHaveAttribute(
    'href',
    '/model-cache.html'
  );
  await expect(page.getByRole('link', { name: 'サイトマップ' })).toHaveAttribute(
    'href',
    '/sitemap.html'
  );
  await expect(page.locator('link[rel="sitemap"]')).toHaveAttribute('href', '/sitemap.xml');
  await expect(page.locator('meta[name="sitemap"]')).toHaveAttribute('content', '/sitemap.xml');
});

test('non-JS user agent can read all catalog model URLs on /model-cache.html without server-cache endpoints', async ({
  browser,
  baseURL,
}) => {
  const context = await browser.newContext({ javaScriptEnabled: false, baseURL });
  try {
    const page = await context.newPage();
    await page.goto('/model-cache.html');
    const links = page.locator('#static-model-catalog a.catalog-model-url');
    // Independent inventory: fail on missing, duplicate, unexpected, or redirected entries.
    const expectedHrefs = [
      'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite',
      'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite2/float32/1/efficientnet_lite2.tflite',
      'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite',
      'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite',
      'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
      'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/1/efficientdet_lite2.tflite',
      'https://huggingface.co/unileon-robotics/YOLO8-ONNX/resolve/main/yolov8n-seg.onnx',
      'https://huggingface.co/unileon-robotics/YOLO8-ONNX/resolve/main/dynamic/yolov8s-seg.onnx',
      'https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx',
      'https://huggingface.co/Xenova/segformer-b1-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx',
      'https://huggingface.co/Xenova/segformer-b2-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx',
      'https://huggingface.co/Cactus-Compute/mirage/resolve/76c1e9e53381aca8fc3843c636216c96ff220122/mirage.cact',
      'https://cdn.jsdelivr.net/gh/zz990099/lightstereo_cpp@master/models/lightstereo_s_sceneflow_general_opt_256_512.onnx',
      '/api/cors-proxy/stereo/model_528_240_float32.onnx',
      '/api/cors-proxy/stereo/hitnet_sceneflow_240x320.onnx',
      '/api/cors-proxy/stereo/crestereo_init_iter2_180x320.onnx',
      '/api/cors-proxy/stereo/crestereo_combined_iter5_240x320.onnx',
    ];
    await expect(links).toHaveCount(expectedHrefs.length);
    const hrefs = await links.evaluateAll((els) => els.map((e) => e.getAttribute('href') ?? ''));
    expect([...hrefs].sort()).toEqual([...expectedHrefs].sort());
    for (const href of hrefs) {
      expect(href).not.toContain('/api/model-cache/');
    }
  } finally {
    await context.close();
  }
});
