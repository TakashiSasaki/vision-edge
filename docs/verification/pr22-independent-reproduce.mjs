// Independent review evidence. Run with Node24 against an installed disposable checkout.
// Usage: node reproduce.mjs /absolute/path/to/frozen-pr22-worktree [output-directory]
// No network/model acquisition, source fixes, merge, or deployment.
import assert from 'node:assert/strict';
import { spawnSync, execFileSync } from 'node:child_process';
import { mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const target = path.resolve(process.argv[2] ?? '.');
const sha = '1c9a485841d89edcd79dabb6a6d1efdd6e632630';
assert.equal(execFileSync('git', ['rev-parse','HEAD'], {cwd:target,encoding:'utf8'}).trim(),sha);
const output = process.argv[3] ? path.resolve(process.argv[3]) : await mkdtemp(path.join(tmpdir(),'pr22-independent-'));
await mkdir(output,{recursive:true});
const {execute,hash} = await import(pathToFileURL(path.join(target,'scripts/testing/capture.mjs')));
const {unitCounts,unitFailures,browserCounts} = await import(pathToFileURL(path.join(target,'scripts/testing/results.mjs')));
const observations = {head:sha,output};
const call = (args,options={}) => spawnSync(process.execPath,args,{cwd:target,encoding:'utf8',...options});
const native = path.join(target,'.cache',`independent-review-${Date.now()}`);
await mkdir(native,{recursive:true});
const config = path.join(native,'vitest.config.mjs');
await writeFile(config,`export default {test:{include:[${JSON.stringify(path.join(native,'native.test.mjs'))}],environment:'node'}};\n`);
await writeFile(path.join(native,'native.test.mjs'),"import {test,expect} from 'vitest';\nfor(let i=0;i<4;i++)test('independent '+i,()=>expect(i,'NATIVE_FILE_ONLY_SENTINEL_'+i).toBe(99));\n");
const runBase = path.join(output,'native-runs');
const env = {...process.env,TEST_OUTPUT_BASE:runBase}; delete env.NODE_TEST_CONTEXT;
const failed = call(['scripts/testing/run.mjs','unit',`--config=${config}`],{env,timeout:15000});
assert.equal(failed.status,1,failed.stderr);
const match = failed.stdout.match(/^artifacts: (.+)$/m); assert(match);
const dir = match[1];
const report = JSON.parse(await readFile(path.join(dir,'unit.json'),'utf8'));
assert.equal(report.numFailedTests,4);
assert(JSON.stringify(report).includes('NATIVE_FILE_ONLY_SENTINEL_3'));
const search = call(['scripts/testing/read-log.mjs',dir,'NATIVE_FILE_ONLY_SENTINEL_3']);
observations.nativeFourthFailure = {testExit:failed.status,readerExit:search.status,falseNoMatch:search.stdout.includes('No match'),fullReportContainsSentinel:true};
await writeFile(path.join(output,'native-search.txt'),search.stdout);
const binary=Buffer.from('日本語🙂\u001b[31mWARNING\u001b[0m\0\r\n'.repeat(3000));
const positive=await execute({name:'binary-positive',command:process.execPath,args:['-e',`process.stdout.write(Buffer.from('${binary.toString('base64')}','base64'));setTimeout(()=>process.stderr.write('STDERR_END'),30)`]},{root:target,runDir:output,emit:()=>{}});
const log=await readFile(path.join(output,'binary-positive/output.log'));
assert(log.equals(Buffer.concat([binary,Buffer.from('STDERR_END')])));
assert.equal(hash(log),positive.logSha256);
observations.binary = {bytes:log.length,exact:true,state:positive.state};
const unit={success:true,numTotalTests:1,numPassedTests:1,numFailedTests:0,numPendingTests:0,numTodoTests:0,testResults:[{assertionResults:[{status:'failed',fullName:'contradiction',failureMessages:['INDEPENDENT_HIDDEN_FAILURE']}]}]};
const browser={errors:[],stats:{expected:1,unexpected:0,skipped:0,flaky:0},suites:[{specs:[{ok:false,tests:[{status:'unexpected',results:[{status:'failed',errors:[{message:'INDEPENDENT_HIDDEN_FAILURE'}]}]}]}]}]};
for(const [name,data,file,summarize] of [['unit-contradiction',unit,'unit.json',unitCounts],['browser-contradiction',browser,'browser.json',browserCounts]]){
 const receipt=await execute({name,command:process.execPath,args:['-e',`require('fs').writeFileSync(${JSON.stringify(path.join(output,name,file))},${JSON.stringify(JSON.stringify(data))})`],summarize,diagnose:name==='unit-contradiction'?unitFailures:undefined},{root:target,runDir:output,emit:()=>{}});
 observations[name]={state:receipt.state,exit:receipt.exit,counts:receipt.counts};
}
// Exercise the actual nested runner without models/browser/network: replace only
// the installed Playwright executable in a new disposable worktree with a
// SIGTERM-resistant fixture. The reviewed implementation is never edited.
const nested = path.join(output,'nested-worktree');
execFileSync('git',['worktree','add','--detach',nested,sha],{cwd:target,stdio:'pipe'});
let pid;let completed=false;let pending;
try {
 await mkdir(path.join(nested,'node_modules/@playwright/test'),{recursive:true});
 const pidFile=path.join(output,'nested.pid');
 await writeFile(path.join(nested,'node_modules/@playwright/test/cli.js'),`require('fs').writeFileSync(${JSON.stringify(pidFile)},String(process.pid));process.on('SIGTERM',()=>{});setInterval(()=>{},100);`);
 pending=execute({name:'nested-real-sdk',command:process.execPath,args:['scripts/real-sdk/run.mjs'],blockedCodes:[2]},{root:nested,runDir:output,timeoutMs:1500,graceMs:100,emit:()=>{}}).then(r=>{completed=true;return r});
 await new Promise(resolve=>setTimeout(resolve,3000));
 pid=Number(await readFile(pidFile,'utf8'));
 let alive=false;try{process.kill(pid,0);alive=true;}catch{/* exited */}
 const before=JSON.parse(await readFile(path.join(output,'nested-real-sdk/result.json'),'utf8'));
 observations.nestedCleanup={timeoutMs:1500,graceMs:100,observedAfterMs:3000,completed,receiptState:before.state,detachedDescendantAlive:alive};
} finally {
 if(pid){try{process.kill(-pid,'SIGKILL');}catch{/* already gone */}}
 if(pending)await pending;
 execFileSync('git',['worktree','remove','--force',nested],{cwd:target,stdio:'pipe'});
}
await writeFile(path.join(output,'observations.json'),JSON.stringify(observations,null,2)+'\n');
console.log(JSON.stringify(observations,null,2));
