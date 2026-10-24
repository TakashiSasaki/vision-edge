import { execFileSync } from 'node:child_process';
import { chromium } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
const output = process.argv[2] ?? 'docs/verification/pr06/overhead.json';
const browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
const page = await browser.newPage();
try {
  await page.goto('http://127.0.0.1:4175/docs/verification/pr06/experiment/index.html');
  await page.waitForFunction(() => !!window.runPipelineTrial);
  const trials = []; const warmup = [];
  for (const [width, height] of [[640, 480], [1280, 720], [1920, 1080]]) for (const path of ['rgba', 'video-frame']) for (const model of ['mediapipe', 'selfie-segmenter']) {
    const configuration = { width, height, path, model };
    for (const enabled of [false, true]) warmup.push(await page.evaluate(opts => window.runPipelineTrial(opts), { ...configuration, enabled, iterations: path === 'rgba' ? 8 : 16 }));
    for (let pair = 0; pair < 12; pair++) {
      for (const enabled of (pair % 2 ? [true, false] : [false, true])) {
        const trial = await page.evaluate(opts => window.runPipelineTrial(opts), { ...configuration, enabled, iterations: path === 'rgba' ? 16 : 32 });
        trials.push({ pair, ...trial });
      }
    }
    console.log(JSON.stringify({ configuration, completedTrials: trials.length }));
    await mkdir(output.slice(0, output.lastIndexOf('/')), { recursive: true });
    await writeFile(output, JSON.stringify({ environment: { sourceHead: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(), sourceTree: execFileSync('git', ['rev-parse', 'HEAD^{tree}'], { encoding: 'utf8' }).trim(), capturedAt: new Date().toISOString(), build: 'production Vite separate experiment entry', browser: browser.version(), node: process.version, userAgent: await page.evaluate(() => navigator.userAgent), hardwareConcurrency: await page.evaluate(() => navigator.hardwareConcurrency) }, warmup, trials }, null, 2));
  }
} finally { await browser.close(); }
