import { defineConfig, devices } from '@playwright/test';
import path from 'node:path';
import { requestedDelegate, delegatePaths } from './scripts/real-sdk/delegate.mjs';
const paths=delegatePaths(requestedDelegate());
const output = process.env.REAL_SDK_RUN_DIR ?? `${paths.runs}/manual`;
export default defineConfig({
  testDir: './e2e-real-sdk',
  outputDir: path.join(output, 'browser'),
  timeout: 60_000,
  globalTimeout: 600_000,
  expect: { timeout: 15_000 },
  forbidOnly: true,
  retries: 0,
  workers: 1,
  reporter: [['list'], ['json', { outputFile: path.join(output, 'playwright.json') }]],
  use: {
    ...devices['Desktop Chrome'],
    baseURL: paths.origin,
    serviceWorkers: 'block',
    launchOptions: { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH },
    trace: 'retain-on-failure', screenshot: 'only-on-failure',
  },
  webServer: {
    command: `node scripts/real-sdk/build-native-probe.mjs && npm run build -- --config vite.real-sdk.config.ts && npm run preview -- --outDir ${paths.build} --host 127.0.0.1 --port ${paths.port} --strictPort`,
    url: paths.origin, timeout: 45_000, reuseExistingServer: false,
    gracefulShutdown: { signal: 'SIGTERM', timeout: 5000 },
  },
});
