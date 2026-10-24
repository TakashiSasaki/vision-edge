import {defineConfig} from '@playwright/test';
import {fileURLToPath} from 'node:url';
export default defineConfig({
 testDir:'.',testMatch:'pixel-app.spec.mjs',workers:1,retries:0,timeout:30000,
 use:{baseURL:'http://127.0.0.1:4186',launchOptions:{executablePath:'/usr/bin/chromium'},serviceWorkers:'block'},
 webServer:{cwd:fileURLToPath(new URL('../../../../',import.meta.url)),command:'npx --no-install vite build --config docs/verification/pr04-rereview2/vectors/vite.config.mjs --outDir .cache/pr04-rereview2 && npm run preview -- --outDir .cache/pr04-rereview2 --host 127.0.0.1 --port 4186 --strictPort',url:'http://127.0.0.1:4186',reuseExistingServer:false,timeout:30000}
});
