import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm, symlink, copyFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { checkRepository, repositoryRoot, installedNpmVersion } from './preflight.mjs';
const hash = b => createHash('sha256').update(b).digest('hex');
async function fixture(work) {
  const root = await mkdtemp(path.join(tmpdir(), 'repository-preflight-'));
  const pkg = { name: 'fixture', version: '1.0.4', packageManager: 'npm@11.9.0', engines: { node: '>=24.22.1 <25', npm: '>=11 <12' }, dependencies: { dep: '^1.0.0' } };
  const lock = { name: pkg.name, version: pkg.version, lockfileVersion: 3, packages: { '': { name: pkg.name, version: pkg.version, engines: pkg.engines, dependencies: pkg.dependencies }, 'node_modules/dep': { version: '1.0.0', resolved: 'https://registry.npmjs.org/dep/-/dep-1.0.0.tgz' } } };
  const ledger = { schemaVersion: 1, sourceCommit: 'a'.repeat(40), entryCount: 2, files: [{ path: 'docs/verification/old.log', sha256: hash('original\n') }, { path: 'docs/verification/empty.log', sha256: hash('') }] };
  const write = async (file, value) => { await mkdir(path.dirname(path.join(root, file)), { recursive: true }); await writeFile(path.join(root, file), typeof value === 'string' ? value : JSON.stringify(value)); };
  await write('package.json', pkg); await write('package-lock.json', lock); await write('.nvmrc', '24.22.1\n'); await write('scripts/testing/protected-evidence.json', ledger); await write('docs/verification/old.log', 'original\n'); await write('docs/verification/empty.log', '');
  const check = options => checkRepository({ root, nodeVersion: '24.22.1', npmVersion: '11.9.0', ...options });
  try { return await work({ root, pkg, lock, ledger, write, check }); } finally { await rm(root, { recursive: true, force: true }); }
}
const has = (r, code) => { assert.equal(r.status, 'fail'); assert(r.issues.some(i => i.code === code), JSON.stringify(r)); };
test('valid locked project, original empty evidence and compatible npm11 minor pass', () => fixture(async f => { assert.equal((await f.check()).status, 'pass'); assert.equal((await f.check({ npmVersion: '11.17.0' })).status, 'pass'); }));
for (const value of [null, 0, [], '{broken']) test(`missing/invalid lock ${JSON.stringify(value)} fails explicitly`, () => fixture(async f => { if (value === null) await rm(path.join(f.root, 'package-lock.json')); else await f.write('package-lock.json', value); has(await f.check(), 'LOCK_MISSING_OR_INVALID'); }));
for (const name of ['bun.lock', 'bun.lockb', 'yarn.lock', 'pnpm-lock.yaml', 'npm-shrinkwrap.json']) test(`competing ${name} fails even with valid npm lock`, () => fixture(async f => { await f.write(name, 'competing'); has(await f.check(), 'CONFLICTING_LOCK'); }));
for (const field of ['name', 'version']) test(`top/root lock ${field} drift fails`, () => fixture(async f => { f.lock[field] = 'wrong'; f.lock.packages[''][field] = 'wrong'; await f.write('package-lock.json', f.lock); has(await f.check(), 'LOCK_IDENTITY'); }));
for (const field of ['dependencies', 'devDependencies', 'optionalDependencies', 'peerDependencies', 'engines']) test(`root ${field} mismatch fails`, () => fixture(async f => { f.lock.packages[''][field] = { changed: '1' }; await f.write('package-lock.json', f.lock); has(await f.check(), 'LOCK_ROOT_MISMATCH'); }));
test('synchronized PATCH and dependency changes plus new source/evidence remain allowed', () => fixture(async f => { f.pkg.version = '1.0.5'; f.pkg.dependencies = { dep: '^2.0.0' }; f.lock.version = '1.0.5'; Object.assign(f.lock.packages[''], { version: '1.0.5', dependencies: f.pkg.dependencies }); f.lock.packages['node_modules/dep'] = { version: '2.0.0', resolved: 'https://registry.npmjs.org/dep/-/dep-2.0.0.tgz' }; await f.write('package.json', f.pkg); await f.write('package-lock.json', f.lock); await f.write('src/new-feature.ts', 'new source'); await f.write('docs/verification/new-report.txt', 'new evidence'); assert.equal((await f.check()).status, 'pass'); }));
for (const value of ['http://registry.npmjs.org/a.tgz', 'https://other.invalid/a.tgz', 'https://user:secret@registry.npmjs.org/a.tgz']) test(`noncanonical lock URL ${new URL(value).hostname} fails without exposing credentials`, () => fixture(async f => { f.lock.packages['node_modules/dep'].resolved = value; await f.write('package-lock.json', f.lock); const r = await f.check(); has(r, 'LOCK_REGISTRY'); assert(!JSON.stringify(r).includes('secret')); }));
for (const nodeVersion of ['22.0.0', '24.19.0', '24.22.0', '25.0.0', 'garbage']) test(`runtime Node ${nodeVersion} contract fails`, () => fixture(async f => has(await f.check({ nodeVersion }), 'NODE_CONTRACT')));
test('matching pin and runtime below the security floor fail', () => fixture(async f => { await f.write('.nvmrc', '24.22.0'); has(await f.check({ nodeVersion: '24.22.0' }), 'NODE_CONTRACT'); }));
for (const npmVersion of ['10.0.0', 'unavailable']) test(`runtime npm ${npmVersion} contract fails`, () => fixture(async f => has(await f.check({ npmVersion }), 'NPM_CONTRACT')));
test('matching legitimate Node24 pin advance passes', () => fixture(async f => { await f.write('.nvmrc', '24.23.0'); assert.equal((await f.check({ nodeVersion: '24.23.0' })).status, 'pass'); }));
test('wrong package-manager declaration fails', () => fixture(async f => { f.pkg.packageManager = 'bun@1.0.0'; await f.write('package.json', f.pkg); has(await f.check(), 'NPM_CONTRACT'); }));
for (const file of ['old.log', 'empty.log']) test(`missing protected ${file} fails including empty originals`, () => fixture(async f => { await rm(path.join(f.root, 'docs/verification', file)); has(await f.check(), 'EVIDENCE_MISSING_OR_INVALID'); }));
test('changed historical bytes fail and original bytes recover', () => fixture(async f => { await f.write('docs/verification/old.log', 'regenerated'); has(await f.check(), 'EVIDENCE_CHANGED'); await f.write('docs/verification/old.log', 'original\n'); assert.equal((await f.check()).status, 'pass'); }));
for (const kind of ['empty', 'count', 'duplicate', 'traversal', 'absolute', 'bad-hash']) test(`invalid ledger ${kind} fails closed`, () => fixture(async f => { if (kind === 'empty') f.ledger.files = []; if (kind === 'count') f.ledger.files.pop(); if (kind === 'duplicate') f.ledger.files[1] = f.ledger.files[0]; if (kind === 'traversal') f.ledger.files[0].path = 'docs/verification/../../outside'; if (kind === 'absolute') f.ledger.files[0].path = '/etc/passwd'; if (kind === 'bad-hash') f.ledger.files[0].sha256 = 'invalid'; await f.write('scripts/testing/protected-evidence.json', f.ledger); has(await f.check(), 'EVIDENCE_LEDGER_INVALID'); }));
test('protected file symlink is rejected', () => fixture(async f => { await rm(path.join(f.root, 'docs/verification/old.log')); await symlink(path.join(f.root, 'docs/verification/empty.log'), path.join(f.root, 'docs/verification/old.log')); has(await f.check(), 'EVIDENCE_MISSING_OR_INVALID'); }));
test('guard is read-only on success and failure', () => fixture(async f => { const snapshot = async () => { const names = await readdir(f.root, { recursive: true, withFileTypes: true }); const result = {}; for (const p of names.filter(p => p.isFile())) { const name = path.join(p.parentPath, p.name); result[name] = hash(await readFile(name)); } return result; }; const a = await snapshot(); await f.check(); assert.deepEqual(await snapshot(), a); await f.write('bun.lock', 'unintended'); const b = await snapshot(); await f.check(); assert.deepEqual(await snapshot(), b); }));
test('actual CLI returns exit1 and structured intended cause for missing canonical lock', () => fixture(async f => { await copyFile(path.join(repositoryRoot, 'scripts/testing/preflight.mjs'), path.join(f.root, 'scripts/testing/preflight.mjs')); const good = spawnSync(process.execPath, [path.join(f.root, 'scripts/testing/preflight.mjs'), '--json'], { encoding: 'utf8', timeout: 15_000 }); assert.equal(good.status, 0, good.stdout + good.stderr); assert.equal(JSON.parse(good.stdout).status, 'pass'); await rm(path.join(f.root, 'package-lock.json')); const r = spawnSync(process.execPath, [path.join(f.root, 'scripts/testing/preflight.mjs'), '--json'], { encoding: 'utf8', timeout: 15_000 }); assert.equal(r.error, undefined); assert.equal(r.status, 1); has(JSON.parse(r.stdout), 'LOCK_MISSING_OR_INVALID'); }));
test('real repository checks the complete reviewed ledger', async () => { const expected = JSON.parse(await readFile(path.join(repositoryRoot, 'scripts/testing/protected-evidence.json'), 'utf8')).entryCount; const r = await checkRepository({ npmVersion: installedNpmVersion() }); assert(expected > 0); assert.equal(r.protectedCount, expected); assert.equal(r.status, 'pass', JSON.stringify(r)); });

test('missing evidence ledger fails rather than skipping protection', () => fixture(async f => { await rm(path.join(f.root, 'scripts/testing/protected-evidence.json')); has(await f.check(), 'EVIDENCE_LEDGER_INVALID'); }));
test('literal null lock fails with an intended diagnostic', () => fixture(async f => { await f.write('package-lock.json', 'null'); has(await f.check(), 'LOCK_MISSING_OR_INVALID'); }));
test('missing Node pin fails explicitly', () => fixture(async f => { await rm(path.join(f.root, '.nvmrc')); has(await f.check(), 'NODE_PIN_INVALID'); }));
test('invalid npm lock schema fails', () => fixture(async f => { f.lock.lockfileVersion = 2; await f.write('package-lock.json', f.lock); has(await f.check(), 'LOCK_SCHEMA'); }));
test('protected parent-directory symlink is rejected', () => fixture(async f => { await mkdir(path.join(f.root, 'elsewhere')); await rm(path.join(f.root, 'docs/verification'), { recursive: true }); await symlink(path.join(f.root, 'elsewhere'), path.join(f.root, 'docs/verification')); has(await f.check(), 'EVIDENCE_MISSING_OR_INVALID'); }));
test('canonical lock symlink is rejected', () => fixture(async f => { await f.write('elsewhere.json', f.lock); await rm(path.join(f.root, 'package-lock.json')); await symlink(path.join(f.root, 'elsewhere.json'), path.join(f.root, 'package-lock.json')); has(await f.check(), 'LOCK_MISSING_OR_INVALID'); }));

for (const [kind, value] of [['array', ['1.0.4']], ['object', { version: '1.0.4' }], ['number', 104], ['boolean', true], ['null', null]]) {
  test(`S1-F2 actual CLI rejects ${kind} package version with structured diagnosis`, () => fixture(async f => {
    f.pkg.version = value;
    await f.write('package.json', f.pkg);
    await copyFile(path.join(repositoryRoot, 'scripts/testing/preflight.mjs'), path.join(f.root, 'scripts/testing/preflight.mjs'));
    const r = spawnSync(process.execPath, [path.join(f.root, 'scripts/testing/preflight.mjs'), '--json'], { encoding: 'utf8', timeout: 15_000 });
    assert.equal(r.error, undefined); assert.equal(r.status, 1); assert.equal(r.signal, null);
    assert(r.stdout.length > 0, r.stderr); has(JSON.parse(r.stdout), 'PACKAGE_VERSION'); assert(!r.stderr.includes('TypeError'));
  }));
}
test('S1-F2 array runtime Node returns NODE_CONTRACT without throwing', () => fixture(async f => has(await f.check({ nodeVersion: ['24.19.0'] }), 'NODE_CONTRACT')));
test('S1-F2 array runtime npm returns NPM_CONTRACT without throwing', () => fixture(async f => has(await f.check({ npmVersion: ['11.9.0'] }), 'NPM_CONTRACT')));
test('S1-F2 array packageManager returns NPM_CONTRACT', () => fixture(async f => { f.pkg.packageManager = ['npm@11.9.0']; await f.write('package.json', f.pkg); has(await f.check(), 'NPM_CONTRACT'); }));
test('S1-F2 array engine range returns NODE_CONTRACT', () => fixture(async f => { f.pkg.engines.node = ['>=24.15.0 <25']; await f.write('package.json', f.pkg); has(await f.check(), 'NODE_CONTRACT'); }));
test('S1-F1 actual negative control accepts only the intended missing-lock failure', () => fixture(async f => {
  for (const name of ['preflight.mjs', 'missing-lock-control.mjs']) await copyFile(path.join(repositoryRoot, 'scripts/testing', name), path.join(f.root, 'scripts/testing', name));
  const run = () => spawnSync(process.execPath, [path.join(f.root, 'scripts/testing/missing-lock-control.mjs')], { encoding: 'utf8', timeout: 15_000 });
  const valid = run(); assert.equal(valid.status, 1); assert.match(valid.stderr, /requires an already-missing/);
  await rm(path.join(f.root, 'package-lock.json'));
  const intended = run(); assert.equal(intended.error, undefined); assert.equal(intended.status, 0, intended.stdout + intended.stderr); assert.match(intended.stdout, /PASS expectedChildExit=1 intended=LOCK_MISSING_OR_INVALID/);
  await f.write('bun.lock', 'unexpected additional failure');
  const unrelated = run(); assert.equal(unrelated.status, 1); assert(!unrelated.stdout.includes('PASS'));
}));
