// Only for an isolated CI checkout whose canonical lock was deliberately removed.
// A setup-node cache failure cannot satisfy this control: setup must finish first.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdir, writeFile, lstat } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../../', import.meta.url));
const dir = path.join(root, '.cache/preflight-negative', randomUUID());
await mkdir(dir, { recursive: true });
let missing = false;
try { await lstat(path.join(root, 'package-lock.json')); }
catch (error) { if (error.code === 'ENOENT') missing = true; else throw error; }
assert(missing, 'negative control requires an already-missing canonical lock');
const r = spawnSync(process.execPath, [path.join(root, 'scripts/testing/preflight.mjs'), '--json'], { cwd: root, encoding: 'utf8', timeout: 30_000, maxBuffer: 4 * 1024 * 1024 });
await writeFile(path.join(dir, 'stdout.json'), r.stdout ?? '');
await writeFile(path.join(dir, 'stderr.log'), r.stderr ?? '');
await writeFile(path.join(dir, 'result.json'), JSON.stringify({ exit: r.status, signal: r.signal, error: r.error ? String(r.error) : null, node: process.version }, null, 2) + '\n');
assert.equal(r.error, undefined); assert.equal(r.signal, null); assert.equal(r.status, 1);
const report = JSON.parse(r.stdout);
assert.equal(report.status, 'fail'); assert.equal(report.violations, 1);
assert.equal(report.issues.length, 1); assert(report.protectedCount > 0);
assert.equal(report.issues[0].code, 'LOCK_MISSING_OR_INVALID');
assert.equal(report.issues[0].file, 'package-lock.json');
assert.match(report.issues[0].message, /ENOENT/);
console.log(`missing-lock-control: PASS expectedChildExit=1 intended=LOCK_MISSING_OR_INVALID protected=${report.protectedCount}; artifacts: ${dir}`);
