// Dependency-free, read-only repository contract. Run after pinning Node, before
// setup-node's dependency cache. npm ci still validates the complete lock graph.
import { readFile, lstat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createReadStream, existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { isDeepStrictEqual } from 'node:util';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const repositoryRoot = fileURLToPath(new URL('../../', import.meta.url));
const conflicts = ['bun.lock', 'bun.lockb', 'yarn.lock', 'pnpm-lock.yaml', 'npm-shrinkwrap.json'];
const version = s => /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(s ?? '') ? s.split('.').map(Number) : null;
function matches(v, range) {
  const actual = version(v);
  if (!actual || typeof range !== 'string') return false;
  const parts = range.trim().split(/\s+/);
  return parts.length > 0 && parts.every(part => {
    const m = /^(>=|<=|>|<|=)?(\d+)(?:\.(\d+))?(?:\.(\d+))?$/.exec(part);
    if (!m) return false;
    const expected = [+m[2], +(m[3] ?? 0), +(m[4] ?? 0)];
    let diff = 0;
    for (let i = 0; i < 3; i++) { if (actual[i] !== expected[i]) { diff = Math.sign(actual[i] - expected[i]); break; } }
    return ({ '>=': diff >= 0, '<=': diff <= 0, '>': diff > 0, '<': diff < 0, '=': diff === 0 })[m[1] ?? '='];
  });
}
async function plainFile(root, relative) {
  let p = root;
  for (const segment of relative.split('/')) {
    p = path.join(p, segment);
    const info = await lstat(p);
    if (info.isSymbolicLink()) throw new Error('symbolic links are not protected repository files');
  }
  const info = await lstat(p);
  if (!info.isFile()) throw new Error('expected a regular file');
  return p;
}
export async function checkRepository({ root = repositoryRoot, nodeVersion = process.versions.node, npmVersion } = {}) {
  const issues = [];
  const fail = (code, file, message) => issues.push({ code, file, message });
  const json = async relative => {
    try { const file = await plainFile(root, relative); if ((await lstat(file)).size > 4 * 1024 * 1024) throw new Error('contract JSON exceeds4MiB'); const value = JSON.parse(await readFile(file, 'utf8')); if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('expected JSON object'); return value; }
    catch (error) { fail(relative === 'package-lock.json' ? 'LOCK_MISSING_OR_INVALID' : 'CONTRACT_FILE_INVALID', relative, error.message); return null; }
  };
  const pkg = await json('package.json'), lock = await json('package-lock.json');
  for (const file of conflicts) {
    try { await lstat(path.join(root, file)); fail('CONFLICTING_LOCK', file, 'package-lock.json is the sole supported dependency lock'); }
    catch (error) { if (error.code !== 'ENOENT') fail('LOCK_INSPECTION_FAILED', file, error.message); }
  }
  let pin = '';
  try { pin = (await readFile(await plainFile(root, '.nvmrc'), 'utf8')).trim(); }
  catch (error) { fail('NODE_PIN_INVALID', '.nvmrc', error.message); }
  if (!version(pin) || version(pin)[0] !== 24) fail('NODE_PIN_INVALID', '.nvmrc', 'expected a precise Node24 version');
  if (!version(nodeVersion) || version(nodeVersion)[0] !== 24 || nodeVersion !== pin || !matches(nodeVersion, pkg?.engines?.node) || !matches(pin, pkg?.engines?.node)) fail('NODE_CONTRACT', 'package.json/.nvmrc', 'Node24 runtime must match .nvmrc; runtime and pin must satisfy the declared comparator engine range');
  if (!/^npm@11\.\d+\.\d+$/.test(pkg?.packageManager ?? '') || !version(npmVersion) || version(npmVersion)[0] !== 11 || !matches(npmVersion, pkg?.engines?.npm) || !matches(pkg?.packageManager?.slice(4), pkg?.engines?.npm)) fail('NPM_CONTRACT', 'package.json', 'npm11 runtime and packageManager must satisfy the declared comparator engine range');
  if (!version(pkg?.version)) fail('PACKAGE_VERSION', 'package.json', 'expected MAJOR.MINOR.PATCH version');
  if (lock) {
    const entry = lock.packages?.[''];
    if (lock.lockfileVersion !== 3 || !entry || typeof entry !== 'object') fail('LOCK_SCHEMA', 'package-lock.json', 'expected npm11 lockfileVersion3 and root package');
    if (!pkg || lock.name !== pkg.name || lock.version !== pkg.version || entry?.name !== pkg.name || entry?.version !== pkg.version) fail('LOCK_IDENTITY', 'package-lock.json', 'package/lock name and version must match; use npm version --no-git-tag-version');
    for (const field of ['dependencies', 'devDependencies', 'optionalDependencies', 'peerDependencies', 'engines', 'workspaces']) {
      if (!isDeepStrictEqual(pkg?.[field] ?? {}, entry?.[field] ?? {})) fail('LOCK_ROOT_MISMATCH', 'package-lock.json', `${field} differ from package.json; reconcile with npm before publication`);
    }
    for (const [name, value] of Object.entries(lock.packages ?? {})) {
      if (value?.resolved !== undefined) {
        try { const url = new URL(value.resolved); if (url.protocol !== 'https:' || url.hostname !== 'registry.npmjs.org' || url.username || url.password || url.port) throw new Error('noncanonical registry'); }
        catch { fail('LOCK_REGISTRY', 'package-lock.json', `noncanonical resolved URL at ${name}`); }
      }
    }
  }
  const ledger = await json('scripts/testing/protected-evidence.json');
  let protectedCount = 0;
  if (!ledger || ledger.schemaVersion !== 1 || !/^[a-f0-9]{40}$/.test(ledger.sourceCommit ?? '') || !Number.isSafeInteger(ledger.entryCount) || ledger.entryCount < 1 || ledger.entryCount > 10_000 || !Array.isArray(ledger.files) || ledger.entryCount !== ledger.files.length) {
    fail('EVIDENCE_LEDGER_INVALID', 'scripts/testing/protected-evidence.json', 'expected nonempty versioned path/SHA256 ledger with matching entryCount');
  } else {
    const seen = new Set();
    for (const row of ledger.files) {
      const relative = row?.path;
      if (typeof relative !== 'string' || !relative.startsWith('docs/verification/') || relative.includes('\\') || relative.includes('\0') || relative.split('/').some(p => !p || p === '.' || p === '..') || !/^[a-f0-9]{64}$/.test(row?.sha256 ?? '') || seen.has(relative)) {
        fail('EVIDENCE_LEDGER_INVALID', 'scripts/testing/protected-evidence.json', 'invalid/duplicate protected path or SHA256'); continue;
      }
      seen.add(relative); protectedCount++;
      try {
        const filename = await plainFile(root, relative);
        const hash = createHash('sha256');
        for await (const chunk of createReadStream(filename)) hash.update(chunk);
        if (hash.digest('hex') !== row.sha256) fail('EVIDENCE_CHANGED', relative, 'historical bytes differ from the reviewed ledger; do not regenerate evidence');
      } catch (error) { fail('EVIDENCE_MISSING_OR_INVALID', relative, error.message); }
    }
  }
  return { status: issues.length ? 'fail' : 'pass', protectedCount, violations: issues.length, issues };
}
export function installedNpmVersion() {
  // Node executes npm's JS entry on Windows too; no shell quoting or .cmd launch.
  const npm = [process.env.npm_execpath, path.join(path.dirname(process.execPath), 'node_modules/npm/bin/npm-cli.js'), path.resolve(path.dirname(process.execPath), '../lib/node_modules/npm/bin/npm-cli.js')].find(p => p && existsSync(p));
  const result = npm ? spawnSync(process.execPath, [npm, '--version'], { encoding: 'utf8', timeout: 10_000, maxBuffer: 64 * 1024 }) : spawnSync(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['--version'], { encoding: 'utf8', timeout: 10_000, maxBuffer: 64 * 1024, shell: false });
  return result.status === 0 && !result.error ? result.stdout.trim() : 'unavailable';
}
if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  if (process.argv.slice(2).some(arg => arg !== '--json')) { console.error('Usage: node scripts/testing/preflight.mjs [--json]'); process.exitCode = 1; }
  else {
    const result = await checkRepository({ npmVersion: installedNpmVersion() });
    if (process.argv.includes('--json')) console.log(JSON.stringify(result, null, 2));
    else { console.log(`preflight: ${result.status.toUpperCase()} protected=${result.protectedCount} violations=${result.violations}`); for (const issue of result.issues.slice(0, 8)) console.error(`${issue.code} ${JSON.stringify(issue.file)}: ${issue.message}`); if (result.issues.length > 8) console.error('More issues: run with --json for complete read-only diagnostics.'); }
    process.exitCode = result.status === 'pass' ? 0 : 1;
  }
}
