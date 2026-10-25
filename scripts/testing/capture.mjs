// Keep complete diagnostics on disk; expose a bounded, truthful command receipt.
import { spawn, execFileSync } from 'node:child_process';
import { openSync, writeSync, closeSync } from 'node:fs';
import { mkdir, writeFile, open } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import path from 'node:path';
export const hash = value => createHash('sha256').update(value).digest('hex');
export const DISPLAY_BYTES = 6000;
export function excerpt(value, limit = DISPLAY_BYTES) {
  const bytes = Buffer.from(value); if (bytes.length <= limit) return value;
  const first = Math.floor(limit / 3);
  return bytes.subarray(0, first).toString('utf8') + '\n[excerpt truncated; full diagnostics in output.log]\n' + bytes.subarray(-(limit - first - 100)).toString('utf8');
}
export async function readTail(file, limit = 16_384) {
  const handle = await open(file, 'r');
  try { const { size } = await handle.stat(); const bytes = Buffer.alloc(Math.min(size, limit)); await handle.read(bytes, 0, bytes.length, Math.max(0, size - bytes.length)); return bytes.toString('utf8'); }
  finally { await handle.close(); }
}
export async function newRun(root, base = process.env.TEST_OUTPUT_BASE ?? path.join(root, '.cache/test-runs')) {
  const dir = path.resolve(base, `${new Date().toISOString().replaceAll(':', '-')}-${randomUUID().slice(0, 8)}`);
  await mkdir(dir, { recursive: true });
  const git = (...args) => { try { return execFileSync('git', args, { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], maxBuffer: 32 * 1024 * 1024 }).trim(); } catch { return null; } };
  const status = git('status', '--porcelain'); const diff = git('diff', '--binary', 'HEAD');
  await writeFile(path.join(dir, 'source.json'), JSON.stringify({ head: git('rev-parse', 'HEAD'), tree: git('rev-parse', 'HEAD^{tree}'), dirty: status === null ? null : !!status, diffSha256: diff === null ? null : hash(diff), node: process.version, npm: process.env.npm_config_user_agent ?? null, start: new Date().toISOString() }, null, 2) + '\n');
  return dir;
}
export async function execute(task, { root, runDir, emit = line => console.log(line), verbose = process.env.TEST_OUTPUT === 'verbose', timeoutMs = 900_000, graceMs = 5000, maxLogBytes = 64 * 1024 * 1024 } = {}) {
  if (!/^[a-z][a-z0-9-]*$/.test(task.name)) throw new Error('Invalid task name');
  const dir = path.join(runDir, task.name); await mkdir(dir); // Never reuse a previous task's report.
  const log = path.join(dir, 'output.log'); const receipt = { task: task.name, argv: [task.command, ...task.args], cwd: root, start: new Date().toISOString(), state: 'RUNNING', selection: task.selection ?? 'default', node: process.version };
  const save = () => writeFile(path.join(dir, 'result.json'), JSON.stringify(receipt, null, 2) + '\n');
  await save();
  const fd = openSync(log, 'wx'); const digest = createHash('sha256'); let bytes = 0; let stdoutTail = Buffer.alloc(0); let child; let timer; let killTimer; let problem; let interrupted; let closed = false;
  const terminate = () => { if (!child?.pid || closed) return; try { if (process.platform === 'win32') child.kill('SIGTERM'); else process.kill(-child.pid, 'SIGTERM'); } catch { /* already closed */ }
    killTimer ??= setTimeout(() => { try { if (process.platform === 'win32') child.kill('SIGKILL'); else process.kill(-child.pid, 'SIGKILL'); } catch { /* already closed */ } }, graceMs); };
  const onInterrupt = () => { interrupted = 'SIGINT'; terminate(); }; const onTerminate = () => { interrupted = 'SIGTERM'; terminate(); };
  const append = (chunk, isStdout) => {
    const kept = chunk.subarray(0, Math.max(0, maxLogBytes - bytes)); writeSync(fd, kept); digest.update(kept); bytes += kept.length;
    if (isStdout) stdoutTail = Buffer.concat([stdoutTail, kept]).subarray(-32_768);
    if (verbose) (isStdout ? process.stdout : process.stderr).write(kept);
    if (kept.length !== chunk.length) { problem = 'log limit exceeded; capture incomplete'; terminate(); }
  };
  let outcome;
  try {
    child = spawn(task.command, task.args, { cwd: root, env: { ...process.env, ...task.env }, detached: process.platform !== 'win32', stdio: ['ignore', 'pipe', 'pipe'] });
    process.once('SIGINT', onInterrupt); process.once('SIGTERM', onTerminate);
    timer = setTimeout(() => { problem = 'command timeout'; terminate(); }, timeoutMs);
    const consume = (b, stdout) => { try { append(b, stdout); } catch(error) { problem = `capture failed: ${String(error)}`; terminate(); } };
    child.stdout.on('data', b => consume(b, true)); child.stderr.on('data', b => consume(b, false));
    outcome = await new Promise(resolve => { child.once('error', error => { problem = String(error); }); child.once('close', (code, signal) => { if (process.platform !== 'win32' && (code !== 0 || signal || problem || interrupted)) { try { process.kill(-child.pid, 'SIGKILL'); } catch { /* no descendants remain */ } } closed = true; resolve({ code, signal }); }); });
  } finally {
    clearTimeout(timer); clearTimeout(killTimer); process.removeListener('SIGINT', onInterrupt); process.removeListener('SIGTERM', onTerminate); closeSync(fd);
  }
  let exit = interrupted ? (interrupted === 'SIGINT' ? 130 : 143) : problem ? 1 : outcome.code ?? 1;
  let counts;
  if (task.summarize) {
    try { counts = await task.summarize({ dir, stdout: stdoutTail.toString('utf8'), childExit: outcome.code });
      if (outcome.code === 0 && ((counts.failed ?? 0) > 0 || (counts.cancelled ?? 0) > 0)) throw new Error('Child exited0 but test report contains failures/cancellations'); }
    catch (error) { receipt.reportError = String(error); if (exit === 0) exit = 1; }
  }
  Object.assign(receipt, { end: new Date().toISOString(), childExit: outcome.code, signal: outcome.signal, interrupted: interrupted ?? null, problem: problem ?? null, exit, state: exit === 0 ? 'PASS' : task.blockedCodes?.includes(exit) && !problem && !interrupted ? 'BLOCKED' : 'FAIL', counts: counts ?? null, logBytes: bytes, logSha256: digest.digest('hex') });
  await save();
  emit(`${task.name}: ${receipt.state} exit=${exit}${receipt.selection !== 'default' ? ' selection=custom' : ''}${counts ? ' ' + Object.entries(counts).map(([k,v]) => `${k}=${v}`).join(' ') : ''}`);
  emit(`artifacts: ${dir}`);
  if (receipt.state !== 'PASS' && !verbose) {
    let details; try { details = await task.diagnose?.(dir); } catch { /* raw files still identify report failure */ }
    emit(excerpt([problem, receipt.reportError, details, details ? null : await readTail(log, 4000)].filter(Boolean).join('\n')));
  }
  return receipt;
}
