import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { execute, hash, excerpt } from './capture.mjs';
import { tapCounts, unitCounts, browserCounts, unitFailures, tapFailures } from './results.mjs';
const base=path.resolve(import.meta.dirname,'../..');
async function fixture(code, options={}) {
 const root=await mkdtemp(path.join(tmpdir(),'compact-output-'));const lines=[];
 try {
  const {task:extraTask,...rest}=options;
  const task={name:'probe',command:process.execPath,args:['-e',code],...extraTask};
  const result=await execute(task,{root,runDir:root,emit:s=>lines.push(s),...rest});
  const log=await readFile(path.join(root,'probe/output.log'));
  const saved=JSON.parse(await readFile(path.join(root,'probe/result.json'),'utf8'));
  assert.deepEqual(saved,result);assert.equal(saved.logSha256,hash(log));if(saved.state!=='PASS'){const diagnostic=await readFile(path.join(root,'probe/diagnostic.txt'),'utf8');assert(diagnostic.length>0);assert(Buffer.byteLength(diagnostic)<6100);}
  return {result,log,lines,root};
 } finally{await rm(root,{recursive:true,force:true});}
}
test('large successful output is saved byte-for-byte while console stays small',async()=>{
 const {result,log,lines}=await fixture("process.stdout.write('x'.repeat(80_000));process.stderr.write('warning: retained');");
 assert.equal(result.state,'PASS');assert.equal(result.exit,0);assert.equal(log.length,80017);assert(log.includes('warning: retained'));assert(Buffer.byteLength(lines.join('\n'))<500);
});
for(const code of [1,7])test(`nonzero exit${code} propagates with diagnostics`,async()=>{
 const {result,lines}=await fixture(`console.error('ROOT_CAUSE_SENTINEL');process.exitCode=${code}`);assert.equal(result.exit,code);assert.equal(result.state,'FAIL');assert(lines.join('\n').includes('ROOT_CAUSE_SENTINEL'));
});
test('exit2 is BLOCKED only for a task that explicitly declares that meaning',async()=>{
 const a=await fixture('process.exitCode=2');const b=await fixture('process.exitCode=2',{task:{blockedCodes:[2]}});
 assert.equal(a.result.state,'FAIL');assert.equal(b.result.state,'BLOCKED');assert.equal(b.result.exit,2);
});
test('spawn error cannot look like a successful empty test run',async()=>{
 const {result}=await fixture('',{task:{command:'/nonexistent/compact-output-command'}});assert.equal(result.state,'FAIL');assert.equal(result.exit,1);assert.match(result.problem,/ENOENT/);
});
test('a missing machine report cannot turn child exit0 into PASS',async()=>{
 const {result}=await fixture('console.log("finished")',{task:{summarize:unitCounts}});assert.equal(result.childExit,0);assert.equal(result.exit,1);assert.match(result.reportError,/ENOENT/);
});
test('a report contradiction is failure, and file-only assertion details are displayed',async()=>{
 const report={numTotalTests:1,numPassedTests:0,numFailedTests:1,numPendingTests:0,numTodoTests:0,testResults:[{assertionResults:[{status:'failed',fullName:'fixture',failureMessages:['FILE_ONLY_ROOT_CAUSE']}]}]};
 const {result,lines}=await fixture(`require('node:fs').writeFileSync('probe/unit.json',${JSON.stringify(JSON.stringify(report))})`,{task:{summarize:unitCounts,diagnose:unitFailures}});
 assert.equal(result.exit,1);assert(lines.join('\n').includes('FILE_ONLY_ROOT_CAUSE'));
});
test('malformed or zero-test TAP is rejected rather than summarized as a pass',()=>{
 for(const stdout of ['# tests 73\n# pass 73\n','# tests 0\n# pass 0\n# fail 0\n# cancelled 0\n# skipped 0\n# todo 0\n'])assert.throws(()=>tapCounts({stdout}));
});
test('skip, todo and cancellation counts stay visible and are not called passes',()=>{
 assert.deepEqual(tapCounts({stdout:'# tests 5\n# pass 1\n# fail 1\n# cancelled 1\n# skipped 1\n# todo 1\n'}),{total:5,passed:1,failed:1,cancelled:1,skipped:1,todo:1});
});
test('structured browser startup errors cannot produce a green summary',async()=>{
 const report={errors:[{message:'SERVER_STARTUP_SENTINEL'}],stats:{expected:0,unexpected:0,skipped:0,flaky:0}};
 const {result}=await fixture(`require('node:fs').writeFileSync('probe/browser.json',${JSON.stringify(JSON.stringify(report))})`,{task:{summarize:browserCounts}});assert.equal(result.exit,1);assert.match(result.reportError,/startup/);
});
test('runaway output fails explicitly at its byte limit',async()=>{
 const {result,log}=await fixture("process.stdout.write('z'.repeat(50000))",{maxLogBytes:1024});assert.equal(result.state,'FAIL');assert.match(result.problem,/capture incomplete/);assert.equal(log.length,1024);
});
test('timeout kills a child that ignores SIGTERM, without a success receipt',async()=>{
 const {result}=await fixture("process.on('SIGTERM',()=>{});setInterval(()=>{},1000)",{timeoutMs:150,graceMs:30});assert.equal(result.state,'FAIL');assert.equal(result.exit,1);assert.match(result.problem,/timeout/);
});
test('interrupt is reported with its conventional nonzero exit',async()=>{
 const timer=setTimeout(()=>process.emit('SIGINT'),100);
 try{const {result}=await fixture('setInterval(()=>{},1000)',{timeoutMs:2000,graceMs:30});assert.equal(result.exit,130);assert.equal(result.interrupted,'SIGINT');assert.equal(result.state,'FAIL');}finally{clearTimeout(timer);}
});
test('failure display is bounded but original logs remain complete',async()=>{
 const {result,log,lines}=await fixture("console.log('head');console.error('e'.repeat(40000));console.error('TAIL_SENTINEL');process.exitCode=1");assert.equal(result.exit,1);assert(log.length>40000);assert(Buffer.byteLength(lines.join('\n'))<6500);assert(lines.join('\n').includes('TAIL_SENTINEL'));assert(Buffer.byteLength(excerpt('界'.repeat(20_000)))<6100);
});
test('separate invocations never overwrite an existing task report',async()=>{
 const root=await mkdtemp(path.join(tmpdir(),'compact-collision-'));const task={name:'probe',command:process.execPath,args:['-e','console.log("first")']};
 try{await execute(task,{root,runDir:root,emit:()=>{}});await assert.rejects(()=>execute(task,{root,runDir:root,emit:()=>{}}),/EEXIST/);assert.match(await readFile(path.join(root,'probe/output.log'),'utf8'),/first/);}finally{await rm(root,{recursive:true,force:true});}
});
test('log reader selects bounded diagnostics and discloses the complete file path',async()=>{
 const root=await mkdtemp(path.join(tmpdir(),'compact-reader-'));
 try{await execute({name:'probe',command:process.execPath,args:['-e',"console.log('needle: important');console.log('x'.repeat(30000))"]},{root,runDir:root,emit:()=>{}});const r=spawnSync(process.execPath,[path.join(base,'scripts/testing/read-log.mjs'),root,'probe','needle'],{encoding:'utf8'});assert.equal(r.status,0);assert.match(r.stdout,/needle: important/);assert.match(r.stdout,/full log:/);assert(r.stdout.length<1000);const direct=spawnSync(process.execPath,[path.join(base,'scripts/testing/read-log.mjs'),path.join(root,'probe'),'needle'],{encoding:'utf8'});assert.equal(direct.status,0);assert.equal(direct.stdout,r.stdout);}finally{await rm(root,{recursive:true,force:true});}
});

test('CLI test filters select a subset without inventing skipped counts',async()=>{
 const dir=await mkdtemp(path.join(tmpdir(),'compact-cli-filter-'));
 try{const env={...process.env,TEST_OUTPUT_BASE:dir};delete env.NODE_TEST_CONTEXT;const r=spawnSync(process.execPath,[path.join(base,'scripts/testing/run.mjs'),'foundation','--test-name-pattern=asset pin verified'],{cwd:base,env,encoding:'utf8',timeout:15000});assert.equal(r.status,0,r.stdout+r.stderr);const [id]=await readdir(dir);const result=JSON.parse(await readFile(path.join(dir,id,'foundation/result.json'),'utf8'));assert(result.counts.total<73);assert.equal(result.counts.failed,0);assert.equal(result.selection,'custom');const tap=await readFile(path.join(dir,id,'foundation/output.log'),'utf8');assert.match(tap,/# Subtest: asset pin verified/);assert(!tap.includes('# Subtest: asset pin tampered'));assert.match(r.stdout,/selection=custom/);}finally{await rm(dir,{recursive:true,force:true});}
});
test('a false global unit report flag is not ignored even when individual counts pass',async()=>{
 const report={success:false,numTotalTests:1,numPassedTests:1,numFailedTests:0,numPendingTests:0,numTodoTests:0};
 const {result}=await fixture(`require('node:fs').writeFileSync('probe/unit.json',${JSON.stringify(JSON.stringify(report))})`,{task:{summarize:unitCounts}});assert.equal(result.exit,1);assert.match(result.reportError,/does not confirm/);
});

test('a nested TAP failure exposes the inner diagnostic without passing-case chatter',async()=>{
 const tap='# Subtest: outer\n    not ok 1 - inner\n      ---\n      error: NESTED_ROOT_SENTINEL\n      ...\nnot ok 1 - outer\n  ---\n  error: subtest failed\n  ...\n1..1\n# tests 2\n# pass 0\n# fail 2\n# cancelled 0\n# skipped 0\n# todo 0\n';
 const {result,lines}=await fixture(`process.stdout.write(${JSON.stringify(tap)});process.exitCode=1`,{task:{summarize:tapCounts,diagnose:tapFailures}});assert.equal(result.exit,1);assert(lines.join('\n').includes('NESTED_ROOT_SENTINEL'));
});

test('a malformed report cannot be mislabeled BLOCKED just because child exited2',async()=>{
 const {result}=await fixture('process.exitCode=2',{task:{blockedCodes:[2],summarize:unitCounts}});assert.equal(result.childExit,2);assert.equal(result.exit,1);assert.equal(result.state,'FAIL');
});

test('outer npm silence does not erase the child command diagnostic level',async()=>{
 const {result,log}=await fixture("console.log(process.env.npm_config_loglevel);console.warn('retained warning')",{task:{env:{npm_config_loglevel:'silent'}}});assert.equal(result.exit,0);assert(log.includes('notice'));assert(log.includes('retained warning'));
});
