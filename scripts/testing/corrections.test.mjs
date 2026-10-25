import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, readdir, rm, copyFile } from 'node:fs/promises';
import { spawnSync, execFileSync } from 'node:child_process';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { execute } from './capture.mjs';
import { unitCounts, browserCounts } from './results.mjs';
const base=path.resolve(import.meta.dirname,'../..');
const cli=(args,env={})=>{const e={...process.env,...env};delete e.NODE_TEST_CONTEXT;return spawnSync(process.execPath,args,{cwd:base,env:e,encoding:'utf8',timeout:15000});};
const read=(dir,pattern)=>cli(['scripts/testing/read-log.mjs',dir,pattern]);
const unit={success:true,numTotalTests:1,numPassedTests:1,numFailedTests:0,numPendingTests:0,numTodoTests:0,testResults:[{status:'passed',assertionResults:[{status:'passed',failureMessages:[]}]}]};
const browser={errors:[],stats:{expected:1,unexpected:0,skipped:0,flaky:0},suites:[{specs:[{ok:true,tests:[{status:'expected',expectedStatus:'passed',results:[{status:'passed',errors:[]}]}]}]}]};
async function report(data,file,summarize){const dir=await mkdtemp(path.join(tmpdir(),'pr22-report-'));try{return await execute({name:'probe',command:process.execPath,args:['-e',`require('fs').writeFileSync('probe/${file}',${JSON.stringify(JSON.stringify(data))})`],summarize},{root:dir,runDir:dir,emit:()=>{}});}finally{await rm(dir,{recursive:true,force:true});}}
test('R1 native fourth assertion is searchable alongside actual pass/skip/todo counts',async()=>{
 const dir=await mkdtemp(path.join(base,'.cache','native-correction-'));
 try{
  const config=path.join(dir,'vitest.config.mjs');await writeFile(config,`export default {test:{include:[${JSON.stringify(path.join(dir,'native.test.mjs'))}],environment:'node'}};`);
  await writeFile(path.join(dir,'native.test.mjs'),"import {test,expect} from 'vitest';for(let i=0;i<4;i++)test('failure '+i,()=>expect(i,'NATIVE_FILE_ONLY_SENTINEL_'+i).toBe(99));test('pass',()=>{});test.skip('skip',()=>{});test.todo('todo');");
  const result=cli(['scripts/testing/run.mjs','unit',`--config=${config}`],{TEST_OUTPUT_BASE:path.join(dir,'runs')});assert.equal(result.status,1,result.stderr);
  const artifact=result.stdout.match(/^artifacts: (.+)$/m)[1];const counts=JSON.parse(await readFile(path.join(artifact,'result.json'),'utf8')).counts;
  assert.deepEqual(counts,{total:7,passed:1,failed:4,skipped:1,todo:1});
  const found=read(artifact,'NATIVE_FILE_ONLY_SENTINEL_3');assert.equal(found.status,0);assert(!found.stdout.includes('No match'));assert.match(found.stdout,/unit.json/);assert.match(found.stdout,/NATIVE_FILE_ONLY_SENTINEL_3/);assert(Buffer.byteLength(found.stdout)<6500);
 }finally{await rm(dir,{recursive:true,force:true});}
});
test('R1 complete browser report is searched beyond the diagnostic excerpt',async()=>{
 const dir=await mkdtemp(path.join(tmpdir(),'pr22-browser-search-'));
 try{await execute({name:'probe',command:process.execPath,args:['-e',"console.log('empty stdout')"]},{root:dir,runDir:dir,emit:()=>{}});const d=path.join(dir,'probe');await writeFile(path.join(d,'browser.json'),JSON.stringify({errors:[0,1,2,3].map(i=>({message:'BROWSER_ERROR_'+i}))}));await writeFile(path.join(d,'diagnostic.txt'),'BROWSER_ERROR_0');const r=read(d,'BROWSER_ERROR_3');assert.equal(r.status,0);assert.match(r.stdout,/browser.json:1/);assert.match(r.stdout,/BROWSER_ERROR_3/);}finally{await rm(dir,{recursive:true,force:true});}
});
test('R1 long matching lines disclose truncation and keep the matched suffix visible',async()=>{
 const dir=await mkdtemp(path.join(tmpdir(),'pr22-long-line-'));
 try{await execute({name:'probe',command:process.execPath,args:['-e',"console.log('x'.repeat(5000)+'CRITICAL_SUFFIX'+'y'.repeat(5000))"]},{root:dir,runDir:dir,emit:()=>{}});const r=read(path.join(dir,'probe'),'CRITICAL_SUFFIX');assert.equal(r.status,0);assert.match(r.stdout,/CRITICAL_SUFFIX/);assert.match(r.stdout,/prefix truncated/);assert.match(r.stdout,/suffix truncated/);assert(Buffer.byteLength(r.stdout)<6500);}finally{await rm(dir,{recursive:true,force:true});}
});
for(const [name,change] of [
 ['failed detail with passing totals',r=>{r.testResults[0].assertionResults[0].status='failed';}],
 ['missing detail',r=>{r.testResults=[];}],
 ['passing detail with failure message',r=>{r.testResults[0].assertionResults[0].failureMessages=['hidden error'];}],
 ['failed suite with passing totals',r=>{r.testResults[0].status='failed';}],
 ['unfinished pending assertion',r=>{r.numPassedTests=0;r.numPendingTests=1;r.testResults[0].assertionResults[0].status='pending';}],
])test(`R4 unit rejects ${name}`,async()=>{const r=structuredClone(unit);change(r);const result=await report(r,'unit.json',unitCounts);assert.equal(result.childExit,0);assert.equal(result.exit,1);assert.equal(result.state,'FAIL');});
for(const [name,change] of [
 ['unexpected failure with passing totals',r=>{const s=r.suites[0].specs[0];s.ok=false;s.tests[0].status='unexpected';s.tests[0].results[0].status='failed';}],
 ['expected label with failed attempt',r=>{r.suites[0].specs[0].tests[0].results[0].status='failed';}],
 ['missing detail',r=>{r.suites=[];}],
 ['skipped expectation counted as a pass',r=>{const t=r.suites[0].specs[0].tests[0];t.expectedStatus='skipped';t.results[0].status='skipped';}],
])test(`R4 browser rejects ${name}`,async()=>{const r=structuredClone(browser);change(r);const result=await report(r,'browser.json',browserCounts);assert.equal(result.childExit,0);assert.equal(result.exit,1);assert.equal(result.state,'FAIL');});
test('R4 browser distinguishes expected failure, skipped and flaky native statuses',async()=>{
 const r=structuredClone(browser);r.stats={expected:1,unexpected:0,skipped:1,flaky:1};r.suites[0].specs=[{ok:true,tests:[{status:'expected',expectedStatus:'failed',results:[{status:'failed',errors:[{message:'expected'}]}]},{status:'skipped',results:[{status:'skipped'}]},{status:'flaky',results:[{status:'failed'},{status:'passed'}]}]}];
 const result=await report(r,'browser.json',browserCounts);assert.equal(result.exit,0);assert.deepEqual(result.counts,{total:3,passed:1,failed:0,skipped:1,flaky:1});
});
function live(pid){try{const s=execFileSync('ps',['-o','stat=','-p',String(pid)],{encoding:'utf8'}).trim();return !!s&&!s.startsWith('Z');}catch{return false;}}
test('R2 Linux detached descendant and inherited pipes are bounded after timeout',{skip:process.platform!=='linux'},async()=>{
 const dir=await mkdtemp(path.join(tmpdir(),'pr22-process-'));let pid;
 try{const code=`const {spawn}=require('child_process');const c=spawn(process.execPath,['-e',"process.on('SIGTERM',()=>{});setInterval(()=>{},100)"],{detached:true,stdio:['ignore',1,2]});require('fs').writeFileSync('pid',String(c.pid));setTimeout(()=>process.exit(0),400);`;
 const start=Date.now();const r=await execute({name:'probe',command:process.execPath,args:['-e',code]},{root:dir,runDir:dir,timeoutMs:1500,graceMs:100,emit:()=>{}});pid=Number(await readFile(path.join(dir,'pid'),'utf8'));assert.equal(r.state,'FAIL');assert.match(r.problem,/timeout/);assert(Date.now()-start<4000);await new Promise(r=>setTimeout(r,100));assert.equal(live(pid),false);
 }finally{if(pid&&live(pid))process.kill(pid,'SIGKILL');await rm(dir,{recursive:true,force:true});}
});
test('R2 actual SDK runner joins capture ownership and retains child stderr',{skip:process.platform!=='linux'},async()=>{
 const dir=await mkdtemp(path.join(tmpdir(),'pr22-sdk-process-'));const work=path.join(dir,'work');let pid;let added=false;
 try{
  execFileSync('git',['worktree','add','--detach',work,'HEAD'],{cwd:base,stdio:'pipe'});added=true;
  for(const f of ['scripts/real-sdk/run.mjs','scripts/real-sdk/delegate.mjs','scripts/testing/capture.mjs','scripts/testing/process-tree.mjs'])await copyFile(path.join(base,f),path.join(work,f));
  await mkdir(path.join(work,'node_modules/@playwright/test'),{recursive:true});
  const pidFile=path.join(dir,'pid');await writeFile(path.join(work,'node_modules/@playwright/test/cli.js'),`require('fs').writeFileSync(${JSON.stringify(pidFile)},String(process.pid));console.error('SDK_CHILD_STDERR');process.on('SIGTERM',()=>{});setInterval(()=>{},100);`);
  const r=await execute({name:'sdk',command:process.execPath,args:['scripts/real-sdk/run.mjs']},{root:work,runDir:dir,timeoutMs:1500,graceMs:100,emit:()=>{}});pid=Number(await readFile(pidFile,'utf8'));assert.equal(r.state,'FAIL');assert.match(r.problem,/timeout/);await new Promise(r=>setTimeout(r,100));assert.equal(live(pid),false);
  const runs=path.join(work,'.cache/real-sdk-runs');const [id]=await readdir(runs);assert.match(await readFile(path.join(runs,id,'command.txt'),'utf8'),/SDK_CHILD_STDERR/);
 }finally{if(pid&&live(pid))process.kill(pid,'SIGKILL');if(added)execFileSync('git',['worktree','remove','--force',work],{cwd:base,stdio:'pipe'});await rm(dir,{recursive:true,force:true});}
});
