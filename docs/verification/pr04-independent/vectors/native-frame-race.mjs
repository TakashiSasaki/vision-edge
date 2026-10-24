import fs from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';
import ts from 'typescript';
const root = process.cwd();
const source = fs.readFileSync(path.join(root, 'src/services/frameScheduler.ts'), 'utf8');
const moduleCode = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
const clip = fs.readFileSync(path.join(root, '.cache/review-video.webm')).toString('base64');
const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true });
const results = [];
try {
  for (const native of [true, false, true, false, true, false]) {
    const page = await browser.newPage();
    await page.setContent('<video muted playsinline></video>');
    await page.addScriptTag({ content: 'window.scheduler = {}; (function(exports) {' + moduleCode + '})(window.scheduler);' });
    const result = await page.evaluate(async ({ clip, native }) => {
      const video = document.querySelector('video');
      video.src = 'data:video/webm;base64,' + clip;
      await video.play();
      const canvas = document.createElement('canvas'); canvas.width = 64; canvas.height = 64;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      const observations = [], arrivals = [];
      let previousPixels = null;
      let stalledOnce = false;
      const rvfc = video.requestVideoFrameCallback.bind(video);
      const raf = window.requestAnimationFrame;
      if (native) video.requestVideoFrameCallback = cb => rvfc((now, metadata) => {
        if (arrivals.length < 10) arrivals.push({ time: metadata.mediaTime, count: metadata.presentedFrames, at: performance.now() });
        cb(now, metadata);
        if (!stalledOnce) {
          stalledOnce = true;
          const until = performance.now() + 320;
          while (performance.now() < until) { /* reproduce delayed timer with native decoding */ }
        }
      });
      else {
        video.requestVideoFrameCallback = undefined;
        window.requestAnimationFrame = cb => raf(now => {
          if (arrivals.length < 10) arrivals.push({ time: video.currentTime, count: video.getVideoPlaybackQuality().totalVideoFrames, at: performance.now() });
          cb(now);
          if (!stalledOnce) {
            stalledOnce = true;
            const until = performance.now() + 320;
            while (performance.now() < until) { /* main-thread contention */ }
          }
        });
      }
      const stop = window.scheduler.startFrameScheduler({
        video, lock: new window.scheduler.JobLock(), current: () => true,
        input() {}, error(cause) { observations.push({ error: String(cause) }); }, adopt() {},
        work() {
          ctx.drawImage(video, 0, 0);
          const bytes = ctx.getImageData(0, 0, 64, 64).data;
          let hash = 2166136261;
          for (const byte of bytes) hash = Math.imul(hash ^ byte, 16777619) >>> 0;
          const identicalPixelsToPrevious = previousPixels !== null && bytes.every((byte, i) => byte === previousPixels[i]);
          previousPixels = new Uint8ClampedArray(bytes);
          observations.push({ time: video.currentTime, total: video.getVideoPlaybackQuality().totalVideoFrames, hash, identicalPixelsToPrevious, at: performance.now() });
          video.pause();
          return hash;
        },
      });
      await new Promise(resolve => setTimeout(resolve, 1000));
      stop(); window.requestAnimationFrame = raf;
      return { native, observations, arrivals };
    }, { clip, native });
    results.push(result); await page.close();
  }
} finally { await browser.close(); }
console.log(JSON.stringify(results, null, 2));
if (results.some(result => result.observations.some(o => o.identicalPixelsToPrevious))) {
  console.error('PR04-F01: native browser consumed byte-for-byte identical video frames twice');
  process.exitCode = 1;
}
