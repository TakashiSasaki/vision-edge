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
  const counts=checked({ total:r.numTotalTests, passed:r.numPassedTests, failed:r.numFailedTests, skipped:r.numPendingTests, todo:r.numTodoTests });
  if(!Array.isArray(r.testResults))throw new Error('Unit detail records missing');
  const details={total:0,passed:0,failed:0,skipped:0,todo:0};
  for(const file of r.testResults) {
    if(!Array.isArray(file.assertionResults))throw new Error('Unit assertion records missing');
    if(childExit===0 && (file.status==='failed'||file.message))throw new Error('Unit suite failure contradicts success');
    for(const t of file.assertionResults) {
      if(t.status==='pending')throw new Error('Incomplete unit assertion is not a skipped test');
      const key={passed:'passed',failed:'failed',pending:'skipped',skipped:'skipped',todo:'todo'}[t.status];
      if(!key)throw new Error('Unknown unit assertion status');
      if(key==='passed' && t.failureMessages?.length)throw new Error('Passing unit assertion contains failures');
      details.total++;details[key]++;
    }
  }
  if(Object.keys(details).some(k=>details[k]!==counts[k]))throw new Error('Unit aggregate/detail contradiction');
  return counts;
}
export async function browserCounts({ dir }) {
  const r = JSON.parse(await readFile(path.join(dir, 'browser.json'), 'utf8')); const s=r.stats;
  if (r.errors?.length) throw new Error('Browser global/startup error; inspect browser.json');
  const counts=checked({ total:s.expected+s.unexpected+s.skipped+s.flaky, passed:s.expected, failed:s.unexpected, skipped:s.skipped, flaky:s.flaky });
  if(!Array.isArray(r.suites))throw new Error('Browser suite records missing');
  const details={total:0,passed:0,failed:0,skipped:0,flaky:0};
  function walk(suites) { for(const suite of suites) {
    for(const spec of suite.specs??[]) {
      if(!Array.isArray(spec.tests)||!spec.tests.length)throw new Error('Browser test records missing');
      let ok=true;
      for(const t of spec.tests) {
        const key={expected:'passed',unexpected:'failed',skipped:'skipped',flaky:'flaky'}[t.status];
        if(!key||!Array.isArray(t.results))throw new Error('Unknown/missing browser result');
        const last=t.results.at(-1);const expected=t.expectedStatus??'passed';
        if(!['passed','failed','skipped'].includes(expected)||t.results.some(x=>!['passed','failed','skipped','timedOut','interrupted'].includes(x.status)))throw new Error('Unknown browser attempt/expectation status');
        if(expected==='skipped'&&t.status!=='skipped')throw new Error('Skipped expectation cannot count as a pass');
        if(t.status==='expected' && (!last||last.status!==expected))throw new Error('Browser expected result contradicts actual outcome');
        if(t.status==='unexpected')ok=false;
        if(t.status==='unexpected' && last?.status===expected)throw new Error('Browser unexpected result contradicts actual outcome');
        if(t.status==='skipped' && t.results.some(x=>x.status!=='skipped'))throw new Error('Browser skipped result contains execution');
        if(t.status==='flaky' && (!last||last.status!==expected||!t.results.some(x=>x.status!==expected)))throw new Error('Browser flaky result contradicts attempts');
        if(last?.status==='passed' && last.errors?.length)throw new Error('Passing browser attempt contains errors');
        details.total++;details[key]++;
      }
      if(spec.ok!==ok)throw new Error('Browser spec status contradicts test details');
    }
    walk(suite.suites??[]);
  } }
  walk(r.suites);
  if(Object.keys(details).some(k=>details[k]!==counts[k]))throw new Error('Browser aggregate/detail contradiction');
  return counts;
}
export function controlCounts({ stdout }) {
  const rows = [...stdout.matchAll(/^([^\r\n]+): exit=(\d+); passed=(\d+); failed=(\d+)$/gm)];
  if (!rows.length) throw new Error('Control outcomes missing');
  return { total: rows.length, healthy:rows.filter(r=>r[2]==='0').length, expectedFaults:rows.filter(r=>r[2]==='1').length };
}

export async function unitFailures(dir) {
  const r = JSON.parse(await readFile(path.join(dir, 'unit.json'), 'utf8'));
  return r.testResults.flatMap(file => {
    const assertions = (file.assertionResults ?? []).filter(t => t.status === 'failed').slice(0,3).map(t => `${t.fullName}:\n${(t.failureMessages ?? []).join('\n')}`);
    return assertions.length ? assertions : file.message ? [`${file.name ?? 'suite'}:\n${file.message}`] : [];
  }).slice(0,3).join('\n');
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
