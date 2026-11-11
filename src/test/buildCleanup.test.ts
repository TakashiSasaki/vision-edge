// @vitest-environment node
import { spawnSync } from 'node:child_process';
import { copyFile, mkdir, mkdtemp, readFile, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { afterEach, beforeEach, expect, it } from 'vitest';

const repositoryRoot = process.cwd();
const preservedFiles = [
  '.cache/test-runs/active/output.log',
  '.cache/container-models/mirage.cact',
  'coverage/coverage-final.json',
  'test-results/trace.zip',
  'playwright-report/index.html',
  'blob-report/report.zip',
];
let fixture: string;

async function writeFixture(relativePath: string, content: string) {
  const target = path.join(fixture, relativePath);
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, content);
}

function runNpm(args: string[]) {
  // Inherit the capture wrapper's process group, including npm's Vite child.
  // Never detach subprocesses or run cleanup against the shared repository.
  const npmCli = process.env.npm_execpath;
  const result = spawnSync(npmCli ? process.execPath : 'npm', npmCli ? [npmCli, ...args] : args, {
    cwd: fixture,
    encoding: 'utf8',
    timeout: 20_000,
    env: {
      ...process.env,
      CI: 'true',
      TEST_LOG_DIR: path.join(fixture, '.cache/test-runs/active'),
      CONTAINER_MODEL_CACHE: 'false',
    },
  });
  // The normal test wrapper preserves these raw child diagnostics in output.log.
  if (result.stdout) process.stdout.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);
  expect(result.error).toBeUndefined();
  expect(result.signal).toBeNull();
  expect(result.status).toBe(0);
}

async function expectEvidencePreserved() {
  for (const file of preservedFiles) {
    expect(await readFile(path.join(fixture, file), 'utf8')).toBe(`preserved: ${file}`);
  }
}

beforeEach(async () => {
  fixture = await mkdtemp(path.join(tmpdir(), 'vision-build-cleanup-'));
  const pkg = JSON.parse(await readFile(path.join(repositoryRoot, 'package.json'), 'utf8'));
  await writeFixture('package.json', JSON.stringify({
    name: 'vision-build-cleanup-fixture',
    private: true,
    type: 'module',
    scripts: pkg.scripts,
  }));
  await mkdir(path.join(fixture, 'scripts'));
  await copyFile(path.join(repositoryRoot, 'scripts/clean.mjs'), path.join(fixture, 'scripts/clean.mjs'));
  for (const file of preservedFiles) await writeFixture(file, `preserved: ${file}`);
  await writeFixture('dist/index.html', 'production preview stays available');
  await writeFixture('server.js', 'legacy generated server');
});

afterEach(async () => {
  await rm(fixture, { recursive: true, force: true });
});

it('explicit npm clean removes only production artifacts and preserves verification evidence', async () => {
  runNpm(['run', '--silent', 'clean']);

  await expect(stat(path.join(fixture, 'dist'))).rejects.toMatchObject({ code: 'ENOENT' });
  await expect(stat(path.join(fixture, 'server.js'))).rejects.toMatchObject({ code: 'ENOENT' });
  await expectEvidencePreserved();

  runNpm(['run', '--silent', 'clean']);
  await expectEvidencePreserved();
}, 30_000);

it('an alternate npm build empties its own output while preserving production output, model cache and logs', async () => {
  await symlink(path.join(repositoryRoot, 'node_modules'), path.join(fixture, 'node_modules'), 'junction');
  await writeFixture('index.html', '<div id="app">Build fixture</div><script type="module" src="/main.js"></script>');
  await writeFixture('main.js', 'document.querySelector("#app").textContent = "Built fixture";');
  await writeFixture('.cache/vision-edge-e2e/obsolete.js', 'old test build');

  runNpm(['run', '--silent', 'build', '--', '--mode', 'e2e', '--outDir', '.cache/vision-edge-e2e']);

  expect(await readFile(path.join(fixture, 'dist/index.html'), 'utf8')).toBe('production preview stays available');
  expect(await readFile(path.join(fixture, 'server.js'), 'utf8')).toBe('legacy generated server');
  expect(await readFile(path.join(fixture, '.cache/vision-edge-e2e/index.html'), 'utf8')).toContain('Build fixture');
  await expect(stat(path.join(fixture, '.cache/vision-edge-e2e/obsolete.js'))).rejects.toMatchObject({ code: 'ENOENT' });
  await expectEvidencePreserved();
}, 30_000);
