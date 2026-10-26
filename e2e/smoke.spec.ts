import { expect, test, type Response } from '@playwright/test';

test('production app loads and model controls work without camera or external services', async ({ context, page }) => {
  const pageErrors: string[] = [];
  const localFailures: string[] = [];
  const externalRequests: string[] = [];
  const stylesheetResponses = new Map<string, Response>();
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('response', response => {
    if (response.request().resourceType() === 'stylesheet') {
      stylesheetResponses.set(response.url(), response);
    }
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
  // Discover the build's links rather than relying on Vite's hashed filenames.
  // HTTP 200 alone is insufficient: preview serves HTML for missing asset URLs.
  const stylesheets = await page.locator('link[rel~="stylesheet"]').evaluateAll(links =>
    links.map(link => (link as HTMLLinkElement).href).filter(href => new URL(href).origin === location.origin));
  expect(stylesheets.length, 'required local stylesheet links').toBeGreaterThan(0);
  for (const href of stylesheets) {
    const response = stylesheetResponses.get(href);
    expect(response, `required stylesheet response: ${href}`).toBeDefined();
    expect(response!.status(), `required stylesheet status: ${href}`).toBe(200);
    expect(response!.headers()['content-type'], `required stylesheet MIME: ${href}`).toMatch(/^text\/css(?:;|$)/i);
  }
  const parsedStylesheets = await page.locator('link[rel~="stylesheet"]').evaluateAll(links =>
    links.filter(link => new URL((link as HTMLLinkElement).href).origin === location.origin).map(link => {
      const sheet = (link as HTMLLinkElement).sheet;
      return { enabled: sheet !== null && !sheet.disabled, rules: sheet?.cssRules.length ?? 0 };
    }));
  for (const sheet of parsedStylesheets) {
    expect(sheet.enabled, 'required stylesheet enabled').toBe(true);
    expect(sheet.rules, 'required stylesheet parsed rules').toBeGreaterThan(0);
  }
  // Check both the base reset/custom CSS and a utility used by the real app.
  await expect(page.locator('body'), 'required stylesheet applied body reset').toHaveCSS('margin', '0px');
  await expect(page.locator('#root'), 'required stylesheet applied overflow').toHaveCSS('overflow', 'hidden');
  await expect(header, 'required stylesheet applied header layout').toHaveCSS('display', 'flex');
  // Observe the real loader reaching the blocked network boundary before asserting errors.
  await expect.poll(() => externalRequests.some(url => url.includes('cdn.jsdelivr.net'))).toBe(true);

  await expect(page.getByRole('alert')).toContainText(/モデル(実行環境の読み込み|の生成)に失敗しました/);
  await expect(header.getByLabel('バックエンド状態')).toHaveText('バックエンド未準備');
  await expect(page.getByText('現在しきい値を超える物体はありません')).toHaveCount(0);
  await expect(page.getByLabel('推論FPS')).toBeVisible();
  await expect(page.getByLabel('推論FPS')).toHaveText('推論: 未計測');
  await page.getByRole('button', { name: 'モデルを再試行' }).click();
  await expect(page.getByRole('alert')).toContainText(/モデル(実行環境の読み込み|の生成)に失敗しました/);

  const model = header.getByRole('combobox', { name: 'モデル', exact: true });
  await expect(model).toHaveValue('selfie-segmenter');
  expect(await model.locator('option:not(:disabled)').evaluateAll(options => options.map(option => (option as HTMLOptionElement).value))).toEqual(['selfie-segmenter', 'mediapipe', 'yolo-seg', 'segformer', 'image-classifier']);
  await expect(header.getByRole('combobox')).toHaveCount(2);
  await model.selectOption('mediapipe');
  await expect(model).toHaveValue('mediapipe');
  await header.getByRole('button', { name: '仕様', exact: true }).click();
  await expect(page.getByRole('heading', { name: /物体認識・セグメンテーションモデル仕様/ })).toBeVisible();
  const specs = page.getByRole('dialog');
  const defaultSegFormer = specs.getByRole('button', { name: /^SegFormer-B0 \(軽量・標準\)/ });
  await expect(defaultSegFormer).toHaveCount(1);
  await expect(defaultSegFormer).not.toBeDisabled();
  await specs.getByRole('button', { name: /General/ }).click();
  await expect(model).toHaveValue('selfie-segmenter');
  await expect(page.getByRole('heading', { name: '物体認識・セグメンテーションモデル仕様' })).toBeHidden();
  await expect(page.getByRole('alert')).toContainText(/モデル(実行環境の読み込み|の生成)に失敗しました/);
  await header.getByRole('button', { name: 'カメラ開始', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'カメラエラー' })).toBeVisible();
  await expect(header).toBeVisible();
  expect(localFailures).toEqual([]);
  expect(pageErrors).toEqual([]);
});
