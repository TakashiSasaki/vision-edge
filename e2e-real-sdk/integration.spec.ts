import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import { manifest, verifyAsset } from '../scripts/real-sdk/assets.mjs';
import { installCamera, routeAssets, ready, release, runtimeEvidence, type InputMode } from './fixtures';

for (const model of ['mediapipe', 'selfie-segmenter'] as const) {
  for (const input of ['VideoFrame', 'Canvas', 'HTMLVideoElement'] as const satisfies readonly InputMode[]) {
    test(`real inference ${model} ${input}`, async ({ context, page }, testInfo) => {
      // Explicit failure for direct Playwright invocations; the supported runner
      // records each blocked scenario and exits2 without skip/xfail success.
      for (const asset of manifest.assets) expect((await verifyAsset(asset)).status, `BLOCKED prerequisite ${asset.id}`).toBe('pass');
      await installCamera(context, input, model === 'mediapipe' ? 'coffee.png' : 'astronaut.png');
      const requests = await routeAssets(context);
      const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
      try {
        await page.goto('/');
        if (model === 'mediapipe') await page.getByRole('banner').getByRole('combobox').selectOption(model);
        await ready(page);
        await expect.poll(() => page.evaluate(model => window.__realSdkProbe.processed.filter(r => r.model === model && r.status === 'ready').length, model)).toBeGreaterThanOrEqual(4);
        const positive = await page.evaluate(() => ({ raw: window.__realSdkProbe.raw, processed: window.__realSdkProbe.processed, draws: window.__realSdkProbe.draws }));
        const raws = positive.raw.filter(r => r.model === model && !r.error);
        expect(raws.length).toBeGreaterThanOrEqual(4);
        expect(raws.every(r => r.input === input)).toBe(true);
        const expected = model === 'mediapipe' ? 'cup' : 'person';
        await expect.poll(() => page.evaluate(({ model, expected }) => window.__realSdkProbe.processed.some(r => r.model === model && r.status === 'ready' && r.objects.some(obj => obj.label === expected) && window.__realSdkProbe.draws.some(d => d.id === r.id)), { model, expected })).toBe(true);
        const result = await page.evaluate(({ model, expected }) => window.__realSdkProbe.processed.find(r => r.model === model && r.status === 'ready' && r.objects.some(obj => obj.label === expected) && window.__realSdkProbe.draws.some(d => d.id === r.id)), { model, expected });
        expect(result, `semantic positive ${expected}`).toBeTruthy();
        // Same actual result must traverse postprocessing and displayed Canvas.
        await expect.poll(() => page.evaluate(id => window.__realSdkProbe.draws.some(d => d.id === id && d.alphaPixels > 0 && d.width === 512 && d.height === 512), result!.id)).toBe(true);
        for (const obj of result!.objects as Array<{ confidence: number; bbox: { x: number; y: number; width: number; height: number } }>) {
          expect(Number.isFinite(obj.confidence) && obj.confidence >= 0 && obj.confidence <= 1).toBe(true);
          expect(Object.values(obj.bbox).every(Number.isFinite)).toBe(true);
          expect(obj.bbox.x).toBeGreaterThanOrEqual(0); expect(obj.bbox.y).toBeGreaterThanOrEqual(0);
          expect(obj.bbox.width).toBeGreaterThan(0); expect(obj.bbox.width).toBeLessThanOrEqual(1);
          expect(obj.bbox.height).toBeGreaterThan(0); expect(obj.bbox.height).toBeLessThanOrEqual(1);
        }
        if (model === 'mediapipe') {
          for (const raw of raws) for (const det of raw.detections ?? []) {
            expect(det.bbox && Object.values(det.bbox).every(Number.isFinite) && det.bbox.width > 0 && det.bbox.height > 0).toBe(true);
            expect(det.categories.every(c => Number.isFinite(c.score) && c.score >= 0 && c.score <= 1)).toBe(true);
          }
          expect(raws.some(r => r.detections?.some((d: { categories: { label: string; score: number }[] }) => d.categories.some(c => c.label === 'cup' && Number.isFinite(c.score) && c.score >= 0.35)))).toBe(true);
        } else {
          const masks = raws.flatMap(r => r.masks ?? []);
          expect(masks.length).toBeGreaterThan(0);
          expect(await page.evaluate(() => window.__realSdkProbe.draws.some(d => d.maskDraws > 0 && d.alphaPixels > 0))).toBe(true);
          for (const mask of masks) {
            expect(mask.width > 0 && mask.height > 0 && mask.length === mask.width * mask.height && mask.finite).toBe(true);
            expect(mask.min).toBeGreaterThanOrEqual(0); expect(mask.max).toBeLessThanOrEqual(1);
          }
          expect(masks.some(m => m.personRegion > 0.5 && m.personRegion > m.backgroundRegion + 0.2 && m.foregroundFraction > 0.05 && m.foregroundFraction < 0.95)).toBe(true);
        }
        await expect.poll(() => page.evaluate(() => window.__realSdkProbe.snapshots.some(s => (['sdk', 'postprocess', 'draw'] as const).every(stage => s.stages[stage].success.measured > 0)))).toBe(true);
        const metric = await page.evaluate(() => window.__realSdkProbe.snapshots.at(-1)!);
        if (input === 'HTMLVideoElement') { expect(metric.stages.latency.success.count).toBe(0); expect(metric.results.unbound).toBeGreaterThan(0); }
        else expect(metric.stages.latency.success.measured).toBeGreaterThan(0);
        const creations = await page.evaluate(() => window.__realSdkProbe.creates.length);
        // Observed reentrant cancellation follows a real SDK result, never a fake result.
        await page.evaluate(() => { window.__realSdkProbe.invalidateAfterNextSdk = true; });
        await expect.poll(() => page.evaluate(() => window.__realSdkProbe.processed.some(r => r.status === 'not-ready'))).toBe(true);
        const stale = await page.evaluate(() => window.__realSdkProbe.processed.filter(r => r.status === 'not-ready').map(r => r.id));
        expect(await page.evaluate(ids => window.__realSdkProbe.draws.some(d => ids.includes(d.id)), stale)).toBe(false);
        await page.getByRole('banner').getByRole('button', { name: '停止', exact: true }).click();
        await expect.poll(() => page.evaluate(() => window.__realCamera.liveTracks())).toBe(0);
        if (input === 'VideoFrame') expect(await page.evaluate(() => window.__realCamera.framesCreated - window.__realCamera.framesClosed)).toBe(0);
        await page.getByRole('banner').getByRole('button', { name: 'カメラ開始', exact: true }).click();
        await expect.poll(() => page.evaluate(() => window.__realCamera.requests)).toBe(2);
        expect(await page.evaluate(() => window.__realSdkProbe.creates.length)).toBe(creations);
        // Both textured background and empty controls; allow existing 2-frame retention.
        for (const scene of ['background', 'empty']) {
          const boundary = await page.evaluate(scene => {
            const probe = window.__realSdkProbe;
            const latest = probe.processed.at(-1)!;
            window.__realCamera.scene = scene;
            return { afterCallId: probe.calls, afterResultId: latest.id, model: latest.model, input: latest.input, requestId: latest.requestId,
              stamp: { scene, revision: window.__realCamera.revision + 1, stream: window.__realCamera.requests } };
          }, scene);
          await expect.poll(async () => {
            const result = await page.evaluate(({ boundary, expected }) => window.__realSdkProbe.negativeCheck(boundary, expected), { boundary, expected });
            expect(result.status, result.reason).not.toBe('fail');
            return result.status;
          }).toBe('pass');
        }
        const other = model === 'mediapipe' ? 'selfie-segmenter' : 'mediapipe';
        await page.getByRole('banner').getByRole('combobox').selectOption(other);
        await ready(page);
        await expect.poll(() => page.evaluate(other => window.__realSdkProbe.processed.some(r => r.model === other && r.status === 'ready'), other)).toBe(true);
        expect(await page.evaluate(() => window.__realSdkProbe.creates.length)).toBe(creations + 1);
        expect(errors).toEqual([]);
        await page.screenshot({ path: testInfo.outputPath('app.png') });
      } finally {
        await release(page);
        const evidence = await page.evaluate(() => ({ probe: window.__realSdkProbe, camera: { ...window.__realCamera, liveTracks: window.__realCamera.liveTracks() } }));
        await writeFile(testInfo.outputPath('evidence.json'), JSON.stringify({ runtime: await runtimeEvidence(page), requests, evidence, errors }, null, 2));
        expect(evidence.camera.liveTracks).toBe(0); expect(evidence.camera.doubleCloses).toBe(0);
        expect(evidence.probe.creates.every(c => c.delegate === 'GPU')).toBe(true);
        expect(evidence.probe.closes).toBe(evidence.probe.creates.filter(c => !c.error).length);
      }
    });
  }
}
for (const model of ['mediapipe', 'selfie-segmenter'] as const) {
  test(`asset diagnostic ${model}`, async ({ context, page }, testInfo) => {
    await installCamera(context, 'VideoFrame', 'astronaut.png');
    const requests = await routeAssets(context, true);
    const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
    try {
      await page.goto('/');
      if (model === 'mediapipe') await page.getByRole('banner').getByRole('combobox').selectOption(model);
      await expect(page.getByRole('alert')).toContainText('モデルの生成に失敗しました。', { timeout: 30_000 });
      await expect.poll(() => page.evaluate(model => window.__realSdkProbe.creates.some(c => c.model === model && !!c.error), model)).toBe(true);
      expect(requests.some(r => r.url.includes(model === 'mediapipe' ? 'efficientdet_lite0.tflite' : 'selfie_segmenter.tflite') && r.status === 'controlled HTTP503')).toBe(true);
      const modelAsset = manifest.assets.find(asset => asset.file === (model === 'mediapipe' ? 'efficientdet_lite0.tflite' : 'selfie_segmenter.tflite'))!;
      expect(await page.evaluate(model => window.__realSdkProbe.creates.find(c => c.model === model && c.error)?.error, model)).toBe(`Error: Failed to fetch model: ${modelAsset.url} (503)`);
      expect(await page.evaluate(() => window.__realSdkProbe.calls)).toBe(0);
      expect(errors).toEqual([]);
      await expect(page.locator('video')).toHaveCount(1);
      await page.screenshot({ path: testInfo.outputPath('diagnostic.png') });
    } finally {
      await release(page);
      await writeFile(testInfo.outputPath('diagnostic.json'), JSON.stringify({ model, requests, runtime: await runtimeEvidence(page), observation: await page.evaluate(() => window.__realSdkProbe), errors }, null, 2));
    }
  });
}
