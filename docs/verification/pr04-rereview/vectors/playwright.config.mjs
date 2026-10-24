import { defineConfig } from '@playwright/test';
import { fileURLToPath } from 'node:url';
export default defineConfig({
 testDir:'.',testMatch:'app-compatibility.spec.mjs',workers:1,retries:0,timeout:20000,
 use:{baseURL:'http://127.0.0.1:4185',launchOptions:{executablePath:'/usr/bin/chromium'},serviceWorkers:'block'},
 webServer:{cwd:fileURLToPath(new URL('../../../../',import.meta.url)),command:'npx --no-install vite build --config docs/verification/pr04-rereview/vectors/vite.config.mjs --outDir .cache/pr04-independent-rereview && npm run preview -- --outDir .cache/pr04-independent-rereview --host 127.0.0.1 --port 4185 --strictPort',url:'http://127.0.0.1:4185',reuseExistingServer:false,timeout:30000}
});
