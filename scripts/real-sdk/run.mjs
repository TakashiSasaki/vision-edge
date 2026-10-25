import { spawn, execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { summarizeRun } from './summarize.mjs';
import { randomUUID } from 'node:crypto';
import { excerpt } from '../testing/capture.mjs';
import { root, manifest, verifyAsset, hash } from './assets.mjs';

const run = path.join(root, '.cache/real-sdk-runs', `${new Date().toISOString().replaceAll(':', '-')}-${randomUUID().slice(0, 8)}`);
await mkdir(run, { recursive: true });
const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim();
const checked = [];
for (const asset of manifest.assets) checked.push(await verifyAsset(asset));
const complete = checked.every(a => a.status === 'pass');
const corrupt = checked.some(a => a.status === 'fail');
const start = new Date().toISOString();
const environment = {
  start, sourceHead: git('rev-parse', 'HEAD'), sourceTree: git('rev-parse', 'HEAD^{tree}'),
  worktreeStatus: git('status', '--porcelain'), diffSha256: hash(git('diff', 'HEAD')),
  manifestSha256: hash(await readFile(path.join(root, 'e2e-real-sdk/assets.lock.json'))),
  os: { platform: os.platform(), release: os.release(), arch: os.arch() }, node: process.version,
  npm: execFileSync('npm', ['--version'], { encoding: 'utf8' }).trim(),
  chromium: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ?? 'Playwright matched browser (no system fallback)',
  mode: 'verified local routing; all other external browser requests blocked', resolution: [512, 512],
  delegate: 'GPU (unchanged product request); physical device unknown',
  population: 'Per scenario: initialization separate; first successful SDK call warmup; next >=3 positive inference calls steady state; background/empty >=5 calls each. HUD rolling 5s/2048 events/latest300 quantiles; no physical timing threshold.',
};
await writeFile(path.join(run, 'environment.json'), JSON.stringify(environment, null, 2) + '\n');
await writeFile(path.join(run, 'assets.json'), JSON.stringify(checked, null, 2) + '\n');
console.log(`Real SDK run: ${run}\nPrerequisites: ${complete ? 'all verified' : 'BLOCKED or corrupt; running asset diagnostics only'}`);
let childExit = null;
if (!corrupt) {
  const args = ['node_modules/@playwright/test/cli.js', 'test', '--config', 'playwright.real-sdk.config.ts'];
  if (!complete) args.push('--grep', 'asset diagnostic');
  await writeFile(path.join(run, 'command.json'), JSON.stringify({ argv: [process.execPath, ...args], cwd: root, env: { REAL_SDK_RUN_DIR: run, PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ?? null } }, null, 2) + '\n');
  const log = [];
  const child = spawn(process.execPath, args, { cwd: root, detached: true, env: { ...process.env, REAL_SDK_RUN_DIR: run }, stdio: ['ignore', 'pipe', 'pipe'] });
  const stop = () => { try { process.kill(-child.pid, 'SIGTERM'); } catch { /* Already closed. */ } };
  process.once('SIGINT', stop); process.once('SIGTERM', stop);
  const timer = setTimeout(stop, 660_000);
  for (const stream of [child.stdout, child.stderr]) stream.on('data', bytes => { log.push(bytes); if (process.env.TEST_OUTPUT === 'verbose') process.stdout.write(bytes); });
  childExit = await new Promise(resolve => { child.once('error', error => { log.push(Buffer.from(String(error))); resolve(1); }); child.once('close', code => resolve(code ?? 1)); });
  clearTimeout(timer); process.removeListener('SIGINT', stop); process.removeListener('SIGTERM', stop);
  await writeFile(path.join(run, 'command.txt'), Buffer.concat(log));
}
let report;
try { report = JSON.parse(await readFile(path.join(run, 'playwright.json'), 'utf8')); } catch { /* startup failure is visible below */ }
const specs = [];
function collect(suites) { for (const suite of suites ?? []) { specs.push(...suite.specs ?? []); collect(suite.suites); } }
collect(report?.suites);
const outcome = title => {
  const spec = specs.find(s => s.title === title);
  if (!spec) return { status: 'not-run', reason: 'No completed test record (startup/launch failure possible)' };
  return { status: spec.ok ? 'pass' : 'fail', tests: spec.tests.map(t => ({ status: t.status, results: t.results.map(r => ({ status: r.status, duration: r.duration, errors: r.errors })) })) };
};
const matrix = [];
for (const model of ['mediapipe', 'selfie-segmenter']) for (const input of ['VideoFrame', 'Canvas', 'HTMLVideoElement']) {
  matrix.push({ model, input, runtime: 'cloud Chromium / unchanged GPU request', ...(complete ? outcome(`real inference ${model} ${input}`) : { status: corrupt ? 'fail' : 'blocked', reason: checked.filter(c => c.status !== 'pass'), executedInference: 0 }) });
}
for (const model of ['mediapipe', 'selfie-segmenter']) matrix.push({ model, input: 'controlled asset HTTP503', runtime: 'cloud Chromium actual SDK', ...outcome(`asset diagnostic ${model}`) });
for (const runtime of ['physical desktop camera/GPU/audio', 'physical mobile camera/GPU/audio', 'public application code equality', 'CPU diagnostic (product delegate unchanged)']) matrix.push({ runtime, model: 'both', status: 'not-run', reason: 'Outside this cloud run; separate authorization/environment needed' });
const exit = corrupt || childExit !== 0 ? 1 : complete ? 0 : 2;
await summarizeRun(run);
const result = { start, end: new Date().toISOString(), childExit, exit, complete, matrix };
await writeFile(path.join(run, 'matrix.json'), JSON.stringify(result, null, 2) + '\n');
await writeFile(path.join(root, '.cache/real-sdk-last-run.txt'), run + '\n');
const counts = Object.fromEntries(['pass', 'fail', 'blocked', 'not-run'].map(status => [status, matrix.filter(row => row.status === status).length]));
console.log(`Real SDK: ${exit === 0 ? 'PASS' : exit === 2 ? 'BLOCKED' : 'FAIL'} exit=${exit} ${JSON.stringify(counts)}; matrix: ${path.join(run, 'matrix.json')}`);
for (const asset of checked.filter(row => row.status !== 'pass')) console.log(`${asset.id}: ${asset.status}; ${asset.reason}`);
if (childExit !== 0 && childExit !== null && process.env.TEST_OUTPUT !== 'verbose') console.error(excerpt(await readFile(path.join(run, 'command.txt'), 'utf8')));
process.exitCode = exit;
