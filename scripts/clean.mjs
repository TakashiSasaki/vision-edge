import { mkdirSync, rmSync } from 'node:fs';

for (const path of ['dist', 'server.js', '.cache', 'coverage', 'test-results', 'playwright-report', 'blob-report']) {
  rmSync(new URL(`../${path}`, import.meta.url), { recursive: true, force: true });
}
mkdirSync(new URL('../.cache', import.meta.url), { recursive: true });

