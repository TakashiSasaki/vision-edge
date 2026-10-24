import { expect, test } from '@playwright/test';

interface CoordinateProbe {
  requests: number;
  speech: string[];
  arcs: number[][];
  dispose: () => void;
}
declare global { interface Window { coordinateProbe: CoordinateProbe } }
test.use({ baseURL: 'http://127.0.0.1:4174', deviceScaleFactor: 2 });

for (const scenario of [
  { name: '4x3 in wide display', width: 640, height: 480, viewport: { width: 1600, height: 550 } },
  { name: '16x9 in narrow display', width: 1280, height: 720, viewport: { width: 550, height: 900 } },
]) {
  test(`pointer selects displayed object and excludes margins: ${scenario.name}`, async ({ context, page }, testInfo) => {
    await page.setViewportSize(scenario.viewport);
    await context.route('**/*', route => new URL(route.request().url()).origin === 'http://127.0.0.1:4174' ? route.continue() : route.abort());
    await context.addInitScript(({ width, height }) => {
      const streams: MediaStream[] = [];
      const sources: HTMLCanvasElement[] = [];
      const probe: CoordinateProbe = window.coordinateProbe = { requests: 0, speech: [], arcs: [], dispose() { streams.forEach(s => s.getTracks().forEach(t => t.stop())); sources.length = 0; } };
      const arc = CanvasRenderingContext2D.prototype.arc;
      CanvasRenderingContext2D.prototype.arc = function(...args: Parameters<typeof arc>) { probe.arcs.push([args[0], args[1], args[2]]); if (probe.arcs.length > 20) probe.arcs.shift(); return arc.apply(this, args); };
      window.speechSynthesis.speak = utterance => { probe.speech.push(utterance.text); };
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: async () => {
        probe.requests++;
        const source = document.createElement('canvas'); source.width = width; source.height = height;
        const ctx = source.getContext('2d')!; ctx.fillStyle = '#18324f'; ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = '#c7e6a4'; ctx.fillRect(64, 48, 128, 96);
        sources.push(source); // Keep the actual image alive for the whole test.
        const stream = source.captureStream(0); streams.push(stream); return stream;
      } } });
    }, scenario);
    await page.goto('/');
    try {
      await page.evaluate(() => { window.visionSdkProbe.observeInputs = true; });
      await page.getByRole('banner').getByRole('combobox', { name: 'モデル', exact: true }).selectOption('mediapipe');
      await expect(page.getByText('(dog)')).toBeVisible();
      const geometry = await page.evaluate(() => {
        const video = document.querySelector('video')!; const canvas = document.querySelector('canvas')!;
        const box = (element: Element) => { const r = element.getBoundingClientRect(); const s = getComputedStyle(element); return { left: r.left, top: r.top, width: r.width, height: r.height, objectFit: s.objectFit, objectPosition: s.objectPosition, transform: s.transform }; };
        const snapshot = document.createElement('canvas'); snapshot.width = video.videoWidth; snapshot.height = video.videoHeight;
        const ctx = snapshot.getContext('2d')!; ctx.drawImage(video, 0, 0);
        const input = window.visionSdkProbe.inputs.at(-1)!;
        const sample = (x: number, y: number) => input.pixels.slice((y * video.videoWidth + x) * 4, (y * video.videoWidth + x) * 4 + 4);
        const pixels = { videoBackground: Array.from(ctx.getImageData(0, 0, 1, 1).data), videoObject: Array.from(ctx.getImageData(100, 100, 1, 1).data),
          sdkBackground: sample(0, 0), sdkObject: sample(100, 100), sdkPixelLength: input.pixels.length, sdkSourceKind: input.sourceKind };
        return { pixels, video: box(video), canvas: box(canvas), intrinsic: [video.videoWidth, video.videoHeight], backing: [canvas.width, canvas.height], dpr: devicePixelRatio };
      });
      const scale = Math.min(geometry.video.width / scenario.width, geometry.video.height / scenario.height);
      const left = geometry.video.left + (geometry.video.width - scenario.width * scale) / 2;
      const top = geometry.video.top + (geometry.video.height - scenario.height * scale) / 2;
      const point = { x: left + 188 * scale, y: top + 138 * scale };
      const margin = scenario.width === 640
        ? { x: left - 2, y: top + 100 * scale }
        : { x: left + 100 * scale, y: top - 2 };
      expect(geometry.video.objectFit).toBe('contain'); expect(geometry.canvas.objectFit).toBe('contain');
      expect(geometry.video.objectPosition).toBe('50% 50%'); expect(geometry.canvas.objectPosition).toBe('50% 50%');
      expect(geometry.video.transform).toBe('none'); expect(geometry.backing).toEqual([scenario.width, scenario.height]);
      expect(margin.x).toBeGreaterThan(geometry.video.left); expect(margin.x).toBeLessThan(geometry.video.left + geometry.video.width);
      expect(margin.y).toBeGreaterThan(geometry.video.top); expect(margin.y).toBeLessThan(geometry.video.top + geometry.video.height);
      expect(Math.abs(geometry.video.width / geometry.video.height - scenario.width / scenario.height)).toBeGreaterThan(0.1);
      expect(geometry.pixels.sdkPixelLength).toBe(scenario.width * scenario.height * 4);
      for (const pixels of [geometry.pixels.videoBackground, geometry.pixels.sdkBackground]) {
        [24, 50, 79, 255].forEach((value, i) => expect(Math.abs(pixels[i] - value)).toBeLessThanOrEqual(2));
      }
      for (const pixels of [geometry.pixels.videoObject, geometry.pixels.sdkObject]) {
        [199, 230, 164, 255].forEach((value, i) => expect(Math.abs(pixels[i] - value)).toBeLessThanOrEqual(2));
      }
      const creates = await page.evaluate(() => window.visionSdkProbe.creates);
      await page.mouse.click(margin.x, margin.y);
      await page.waitForTimeout(100);
      const marginObservation = await page.evaluate(() => ({ speech: [...window.coordinateProbe.speech], arcs: [...window.coordinateProbe.arcs] }));
      await testInfo.attach('margin.png', { body: await page.screenshot(), contentType: 'image/png' });
      const group = page.locator('.group').filter({ has: page.getByText('(dog)') });
      const marginSelected = /border-blue-500\/80/.test(await group.getAttribute('class') ?? '');
      await page.mouse.click(point.x, point.y);
      await page.waitForTimeout(150);
      const observations = await page.evaluate(() => ({ speech: window.coordinateProbe.speech, arcs: window.coordinateProbe.arcs, requests: window.coordinateProbe.requests, creates: window.visionSdkProbe.creates }));
      const evidence = { scenario, geometry, content: { left, top, width: scenario.width * scale, height: scenario.height * scale }, point, margin, marginObservation, marginSelected, observations };
      console.log('coordinate observation', JSON.stringify(evidence));
      await testInfo.attach('geometry.json', { body: JSON.stringify(evidence, null, 2), contentType: 'application/json' });
      await testInfo.attach('display.png', { body: await page.screenshot(), contentType: 'image/png' });
      expect(marginObservation.speech).toEqual([]); expect(marginObservation.arcs).toEqual([]); expect(marginSelected).toBe(false);
      await expect(group).toHaveClass(/border-blue-500\/80/);
      expect(observations.speech).toHaveLength(1);
      expect(observations.arcs.length).toBeGreaterThan(0);
      // Native MouseEvent coordinates can be quantized to a CSS pixel.
      expect(Math.abs(observations.arcs.at(-1)![0] - 188)).toBeLessThanOrEqual(1 / scale);
      expect(Math.abs(observations.arcs.at(-1)![1] - 138)).toBeLessThanOrEqual(1 / scale);
      expect(observations.requests).toBe(1); expect(observations.creates).toBe(creates);
      // Exercise a real responsive resize without camera or model replacement.
      await page.setViewportSize({ width: 900, height: 700 });
      const resized = await page.locator('video').boundingBox(); expect(resized).not.toBeNull();
      const nextScale = Math.min(resized!.width / scenario.width, resized!.height / scenario.height);
      await page.mouse.click(resized!.x + (resized!.width - scenario.width * nextScale) / 2 + 90 * nextScale,
        resized!.y + (resized!.height - scenario.height * nextScale) / 2 + 90 * nextScale);
      await expect.poll(() => page.evaluate(() => window.coordinateProbe.speech.length)).toBe(2);
      await expect.poll(() => page.evaluate(() => Math.abs((window.coordinateProbe.arcs.at(-1)?.[0] ?? -1) - 90))).toBeLessThanOrEqual(1 / nextScale);
      const ripple = await page.evaluate(() => window.coordinateProbe.arcs.at(-1)![0]);
      expect(Math.abs(ripple - 90)).toBeLessThanOrEqual(1 / nextScale);
      const afterResize = await page.evaluate(() => ({ arcs: window.coordinateProbe.arcs, requests: window.coordinateProbe.requests, creates: window.visionSdkProbe.creates }));
      expect(Math.abs(afterResize.arcs.at(-1)![1] - 90)).toBeLessThanOrEqual(1 / nextScale);
      expect(afterResize.requests).toBe(1); expect(afterResize.creates).toBe(creates);
      await testInfo.attach('resize.json', { body: JSON.stringify({ resized, nextScale, afterResize }, null, 2), contentType: 'application/json' });
    } finally { await page.evaluate(() => window.coordinateProbe.dispose()); }
  });
}
