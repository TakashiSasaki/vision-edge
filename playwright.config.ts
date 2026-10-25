import { defineConfig, devices } from '@playwright/test';
import path from 'node:path';
const logDir = process.env.TEST_LOG_DIR;

export default defineConfig({
  testDir: './e2e',
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  reporter: logDir ? [['json', { outputFile: path.join(logDir, 'browser.json') }], ['html', { open: 'never', outputFolder: path.join(logDir, 'html') }]] : [['list'], ['html', { open: 'never' }]],
  ...(logDir ? { outputDir: path.join(logDir, 'results') } : {}),
  use: {
    baseURL: 'http://127.0.0.1:4173',
    launchOptions: { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH },
    serviceWorkers: 'block',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [{
    command: 'npm run preview -- --host 127.0.0.1 --port 4173 --strictPort',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: false,
    timeout: 30_000,
  }, {
    command: 'npm run build -- --mode e2e --outDir .cache/vision-edge-e2e && npm run preview -- --outDir .cache/vision-edge-e2e --host 127.0.0.1 --port 4174 --strictPort',
    url: 'http://127.0.0.1:4174',
    reuseExistingServer: false,
    timeout: 30_000,
  }],
});
