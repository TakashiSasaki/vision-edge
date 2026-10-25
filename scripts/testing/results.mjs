import { readFile } from 'node:fs/promises';
import path from 'node:path';
const integer = value => Number.isSafeInteger(value) && value >= 0;
function checked(counts) {
  if (!Object.values(counts).every(integer) || counts.total === 0) throw new Error('Missing/invalid/empty test counts');
  if (Object.values(counts).slice(1).reduce((a,b)=>a+b,0) !== counts.total) throw new Error('Contradictory test counts');
  return counts;
}
export function tapCounts({ stdout }) {
  const names = ['tests', 'pass', 'fail', 'cancelled', 'skipped', 'todo'];
  const values = names.map(name => [...stdout.matchAll(new RegExp(`^# ${name} (\\d+)\\r?$`, 'gm'))].at(-1)?.[1]);
  if (values.some(v => v === undefined)) throw new Error('Node TAP totals missing');
  return checked(Object.fromEntries(['total','passed','failed','cancelled','skipped','todo'].map((key,i)=>[key,Number(values[i])])));
}
export async function unitCounts({ dir, childExit }) {
  const r = JSON.parse(await readFile(path.join(dir, 'unit.json'), 'utf8'));
  if (childExit === 0 && r.success !== true) throw new Error('Unit report does not confirm a successful run');
  return checked({ total:r.numTotalTests, passed:r.numPassedTests, failed:r.numFailedTests, skipped:r.numPendingTests, todo:r.numTodoTests });
}
export async function browserCounts({ dir }) {
  const r = JSON.parse(await readFile(path.join(dir, 'browser.json'), 'utf8')); const s=r.stats;
  if (r.errors?.length) throw new Error('Browser global/startup error; inspect browser.json');
  return checked({ total:s.expected+s.unexpected+s.skipped+s.flaky, passed:s.expected, failed:s.unexpected, skipped:s.skipped, flaky:s.flaky });
}
export function controlCounts({ stdout }) {
  const rows = [...stdout.matchAll(/^([^\r\n]+): exit=(\d+); passed=(\d+); failed=(\d+)$/gm)];
  if (!rows.length) throw new Error('Control outcomes missing');
  return { total: rows.length, healthy:rows.filter(r=>r[2]==='0').length, expectedFaults:rows.filter(r=>r[2]==='1').length };
}

export async function unitFailures(dir) {
  const r = JSON.parse(await readFile(path.join(dir, 'unit.json'), 'utf8'));
  return r.testResults.flatMap(file => file.assertionResults.filter(t => t.status === 'failed').slice(0,3).map(t => `${t.fullName}:\n${(t.failureMessages ?? []).join('\n')}`)).slice(0,3).join('\n');
}
export async function browserFailures(dir) {
  const r = JSON.parse(await readFile(path.join(dir, 'browser.json'), 'utf8')); const errors = [...r.errors ?? []];
  function walk(suites) { for (const s of suites ?? []) { for (const spec of s.specs ?? []) for (const t of spec.tests) for (const result of t.results) errors.push(...result.errors ?? []); walk(s.suites); } }
  walk(r.suites); return errors.slice(0,3).map(e => e.message ?? e.value ?? JSON.stringify(e)).join('\n');
}
export async function tapFailures(dir) {
  const { readTail } = await import('./capture.mjs');
  const lines = (await readTail(path.join(dir, 'output.log'), 32_768)).split('\n'); const blocks=[];
  for(let i=0;i<lines.length&&blocks.length<3;i++) {
    if(!/^\s*not ok \d+/.test(lines[i])) continue;
    const block=[lines[i]];
    for(let j=i+1;j<lines.length;j++) { block.push(lines[j]); if(/^\s+\.\.\.$/.test(lines[j])) { i=j;break; } }
    blocks.push(block.join('\n'));
  }
  return blocks.join('\n');
}
