// Model-free mode/integrity controls; no SDK call or model acquisition.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdtemp, mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { requestedDelegate, commandDelegate, delegatePaths, cpuEngineVariant } from './delegate.mjs';
import { summarizeRun } from './summarize.mjs';
const root=path.resolve(import.meta.dirname,'../..');
test('default CLI stays GPU despite explicit ambient CPU setting',()=>{
 const old=process.env.REAL_SDK_DELEGATE;try{process.env.REAL_SDK_DELEGATE='CPU';assert.equal(requestedDelegate(),'CPU');assert.equal(commandDelegate([]),'GPU');}finally{if(old===undefined)delete process.env.REAL_SDK_DELEGATE;else process.env.REAL_SDK_DELEGATE=old;}
});
test('explicit CPU/GPU CLI modes are accepted without fallback',()=>{assert.equal(commandDelegate(['--delegate=CPU']),'CPU');assert.equal(commandDelegate(['--delegate=GPU']),'GPU');});
for(const value of ['cpu','','AUTO','GPU,CPU'])test(`invalid delegate ${JSON.stringify(value)} is rejected`,()=>{assert.throws(()=>requestedDelegate(value),/INVALID_REAL_SDK_DELEGATE/);assert.throws(()=>commandDelegate([`--delegate=${value}`]),/INVALID_REAL_SDK_DELEGATE/);});
test('unknown or repeated arguments fail closed',()=>{assert.throws(()=>commandDelegate(['--cpu']),/INVALID/);assert.throws(()=>commandDelegate(['--delegate=CPU','--delegate=GPU']),/INVALID/);});
test('npm entrypoints reject ignored selectors/delegate arguments instead of silent fallback',()=>{for(const name of ['real-sdk','real-sdk-cpu']){const r=spawnSync(process.execPath,['scripts/testing/run.mjs',name,'--delegate=CPU'],{cwd:root,encoding:'utf8',timeout:5000});assert.equal(r.status,1);assert.match(r.stderr,/accept no extra arguments/);assert(!r.stdout.includes('Real SDK run:'));}});
test('GPU and CPU isolate origin, build, run, pointer and native probe paths',()=>{const a=delegatePaths('GPU'),b=delegatePaths('CPU');for(const key of Object.keys(a))assert.notEqual(a[key],b[key]);assert.equal(a.port,4175);assert.equal(b.port,4176);});
test('CPU test variant changes only the engine delegate constant, never factory/results',async()=>{const source=await readFile(path.join(root,'src/services/visionEngine.ts'),'utf8');const changed=cpuEngineVariant(source,'visionEngine.ts');assert.equal(changed.replace("const DELEGATE = 'CPU' as const;","const DELEGATE = 'GPU' as const;"),source);assert.match(source,/const DELEGATE = 'GPU' as const;/);assert.equal(cpuEngineVariant(source,'observer.ts'),undefined);});
test('missing/duplicate CPU build anchor fails closed',()=>{assert.throws(()=>cpuEngineVariant('','visionEngine.ts'),/anchor changed/);assert.throws(()=>cpuEngineVariant("const DELEGATE = 'GPU' as const;const DELEGATE = 'GPU' as const;",'visionEngine.ts'),/anchor changed/);});
test('installed SDK declaration and implementation support explicit CPU option',async()=>{const declaration=await readFile(path.join(root,'node_modules/@mediapipe/tasks-vision/vision.d.ts'),'utf8');assert.match(declaration,/delegate\?: "CPU" \| "GPU"/);const sdk=await readFile(path.join(root,'node_modules/@mediapipe/tasks-vision/vision_bundle.mjs'),'utf8');assert(sdk.includes('"GPU"===e.delegate'));});
test('actual CPU Vite pipeline transforms the engine before TS erasure without models',async()=>{
 const parent=process.env.TEST_LOG_DIR??path.join(root,'.cache/delegate-controls');await mkdir(parent,{recursive:true});const dir=await mkdtemp(path.join(parent,'cpu-build-'));
 const result=spawnSync(process.execPath,['node_modules/vite/bin/vite.js','build','--config','vite.real-sdk.config.ts','--outDir',path.join(dir,'dist')],{cwd:root,env:{...process.env,REAL_SDK_DELEGATE:'CPU'},encoding:'utf8',timeout:20000});
 await writeFile(path.join(dir,'stdout.log'),result.stdout??'');await writeFile(path.join(dir,'stderr.log'),result.stderr??'');await writeFile(path.join(dir,'exit.json'),JSON.stringify({exit:result.status,signal:result.signal,error:result.error?String(result.error):null})+'\n');
 assert.ifError(result.error);assert.equal(result.status,0,`CPU build failed; logs: ${dir}\n${result.stderr}`);assert((await readFile(path.join(dir,'dist/index.html'),'utf8')).includes('<script'));
});
for(const observed of ['CPU','GPU'])test(`summary preserves requested CPU versus observed ${observed}, with zero inference`,async()=>{
 const dir=await mkdtemp(path.join(tmpdir(),'cpu-summary-'));
 try{await mkdir(path.join(dir,'browser'));await writeFile(path.join(dir,'environment.json'),JSON.stringify({requestedDelegate:'CPU'}));await writeFile(path.join(dir,'browser/diagnostic.json'),JSON.stringify({observation:{creates:[{model:'mediapipe',delegate:observed,error:'CHECKER-ONLY missing model'}],calls:0,sdkSuccesses:0,rawDropped:0,raw:[],processed:[],draws:[],adoptions:[]}}));await summarizeRun(dir);const s=JSON.parse(await readFile(path.join(dir,'measurements.json'),'utf8'));assert.equal(s.requestedDelegate,'CPU');assert.deepEqual(s.execution[0].observedCreateDelegates,[observed]);assert.equal(s.inferenceExecuted,false);assert.equal(s.inferenceAttempted,false);assert.equal(s.resultRendered,false);assert.deepEqual(s.populations,[]);assert.equal(s.execution[0].evidenceErrors.length,observed==='CPU'?0:1);}finally{await rm(dir,{recursive:true,force:true});}
});
