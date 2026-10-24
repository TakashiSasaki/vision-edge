import { chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';
const browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
const page = await browser.newPage();
const warmup = []; const trials = [];
const persist = (completed, error) => writeFileSync('docs/verification/pr06-f01-correction/hud-cost-final.json', JSON.stringify({ environment: { browser: browser.version(), node: process.version, capturedAt: new Date().toISOString() }, completed, error, identicalHealthyHTML: completed, warmup, trials }, null, 2) + '\n');
try {
  await page.goto('http://127.0.0.1:4176/docs/verification/pr06-f01-correction/experiment/index.html');
  await page.waitForFunction(() => !!window.hudCost);
  for (const profile of ['fresh', '300', 'truncated']) {
    if (!await page.evaluate(profile => window.hudCost.equivalent(profile), profile)) throw Error(`Valid ${profile} HTML changed`);
    for (const open of [false, true]) {
      for (const version of ['before', 'after']) warmup.push(await page.evaluate(opts => window.hudCost.trial(opts), { version, profile, open, iterations: 16 }));
      for (let pair = 0; pair < 12; pair++) for (const version of pair % 2 ? ['after', 'before'] : ['before', 'after']) {
        const result = await page.evaluate(opts => window.hudCost.trial(opts), { version, profile, open, iterations: 32 });
        if (result.snapshots !== 32 || result.renders !== 32) throw Error('Unmatched HUD work');
        trials.push({ pair, ...result });
        persist(false);
      }
    }
  }
  persist(true);
  console.log(`Completed ${trials.length} matched HUD trials with valid HTML equivalence`);
} catch (error) { persist(false, String(error)); throw error; } finally { await browser.close(); }
