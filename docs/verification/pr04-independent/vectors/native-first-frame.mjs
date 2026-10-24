import fs from 'node:fs';
import { chromium } from '@playwright/test';
import ts from 'typescript';
const source = fs.readFileSync('src/services/frameScheduler.ts', 'utf8');
const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true });
const results = [];
try {
  for (const native of [true, false]) {
    const page = await browser.newPage();
    await page.setContent('<video muted playsinline></video>');
    await page.addScriptTag({ content: 'window.scheduler = {}; (function(exports) {' + code + '})(window.scheduler);' });
    results.push(await page.evaluate(async native => {
      const source = document.createElement('canvas'); source.width = 64; source.height = 64;
      source.getContext('2d').fillRect(0, 0, 64, 64);
      const stream = source.captureStream(0);
      const video = document.querySelector('video');
      const firstPresented = new Promise(resolve => video.requestVideoFrameCallback((now, meta) => resolve({ mediaTime: meta.mediaTime, presentedFrames: meta.presentedFrames })));
      video.srcObject = stream;
      await video.play(); stream.getVideoTracks()[0].requestFrame();
      const first = await firstPresented;
      await new Promise(resolve => setTimeout(resolve, 100));
      if (!native) video.requestVideoFrameCallback = undefined;
      let calls = 0;
      const stop = window.scheduler.startFrameScheduler({
        video, current: () => true, lock: new window.scheduler.JobLock(),
        work() { calls++; return []; }, adopt() {}, error() {}, input() {},
      });
      await new Promise(resolve => setTimeout(resolve, 400));
      stop(); stream.getTracks().forEach(track => track.stop());
      return { native, first, readyState: video.readyState, width: video.videoWidth, height: video.videoHeight, calls };
    }, native));
    await page.close();
  }
} finally { await browser.close(); }
console.log(JSON.stringify(results, null, 2));
if (results.some(result => result.calls !== 1)) {
  console.error('PR04-F02: existing ready first frame was not processed once in both paths');
  process.exitCode = 1;
}
