import { test, expect } from '@playwright/test';
import { readFile, writeFile } from 'node:fs/promises';
import { installCamera, routeAssets, type InputMode } from './fixtures';
import type { Stamp } from './acceptance';
import { delegatePaths, requestedDelegate } from '../scripts/real-sdk/delegate.mjs';
const paths=delegatePaths(requestedDelegate());

for (const mode of ['VideoFrame', 'Canvas', 'HTMLVideoElement'] as InputMode[]) {
  test(`input provenance without models ${mode}`, async ({ context, page }, info) => {
    await installCamera(context, mode, 'astronaut.png');
    const requests = await routeAssets(context);
    await context.route(`${paths.origin}/`, route => route.fulfill({ contentType: 'text/html', body: '<video autoplay muted playsinline></video>' }));
    await context.route('**/native-probe.js', async route => route.fulfill({ contentType: 'application/javascript', body: await readFile(`${paths.probe}/native-probe.js`) }));
    try {
      await page.goto('/');
      const result = await page.evaluate(async () => {
        const video = document.querySelector('video')!;
        const stream = await navigator.mediaDevices.getUserMedia({ video: true }); video.srcObject = stream; await video.play();
        const modulePath = '/native-probe.js';
        const scheduler = await import(/* @vite-ignore */ modulePath);
        const rows: Array<{ kind: string; stamp: Stamp | null }> = []; const errors: string[] = [];
        let resolve!: () => void; let reject!: (error: unknown) => void;
        const done = new Promise<void>((yes, no) => { resolve = yes; reject = no; });
        const timeout = setTimeout(() => reject(new Error('native provenance condition timeout')), 8000);
        let changed = false;
        const stop = scheduler.startFrameScheduler({ video, lock: new scheduler.JobLock(), current: () => true, input: () => {},
          work: (input: VideoFrame | HTMLCanvasElement | HTMLVideoElement) => {
            const stamp = scheduler.inputStamp(input);
            rows.push({ kind: input instanceof HTMLVideoElement ? 'HTMLVideoElement' : input instanceof HTMLCanvasElement ? 'Canvas' : 'VideoFrame', stamp });
            if (!changed && stamp?.scene === 'astronaut.png') { changed = true; window.__realCamera.scene = 'empty'; }
            if (stamp?.scene === 'empty') resolve();
          }, adopt: () => {}, error: (error: unknown) => { errors.push(String(error)); reject(error); }, captureError: (error: unknown) => reject(error) });
        try { await done; }
        finally { clearTimeout(timeout); stop(); window.__realCamera.dispose(); }
        return { rows, errors, sdkObservationPresent: !!window.__realSdkProbe, created: window.__realCamera.framesCreated, closed: window.__realCamera.framesClosed, liveTracks: window.__realCamera.liveTracks() };
      });
      await writeFile(info.outputPath('provenance.json'), JSON.stringify({ mode, result, requests, browserVersion: context.browser()?.version(), scope: 'actual native inputs/unchanged scheduler ONLY; NO SDK/model inference' }, null, 2));
      expect(result.errors).toEqual([]); expect(result.sdkObservationPresent).toBe(false); expect(result.rows.every(row => row.kind === mode)).toBe(true);
      const first = result.rows.find(row => row.stamp?.scene === 'astronaut.png')!.stamp!;
      const next = result.rows.find(row => row.stamp?.scene === 'empty')!.stamp!;
      expect(next.revision).toBe(first.revision + 1); expect(next.stream).toBe(first.stream);
      expect(result.created).toBe(result.closed); expect(result.liveTracks).toBe(0);
      expect(requests.some(row => row.url.endsWith('.tflite'))).toBe(false);
    } finally { await page.evaluate(() => window.__realCamera?.dispose()); }
  });
}
