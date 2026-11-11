import { rmSync } from 'node:fs';

// Explicit cleanup is limited to production build artifacts. Test runs, model
// caches and reports are evidence and must survive both clean and later builds.
for (const path of ['dist', 'server.js']) {
  rmSync(new URL(`../${path}`, import.meta.url), { recursive: true, force: true });
}

