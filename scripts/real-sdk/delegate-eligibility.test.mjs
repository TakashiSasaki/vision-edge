// Dependency-free CHECKER-ONLY data. No SDK/model/runtime or asset enrollment.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { summarizeRun } from './summarize.mjs';
function fixture(delegate) {
 const stamp={scene:'coffee.png',revision:1,stream:1};
 return {creates:[{model:'mediapipe',delegate}],calls:1,sdkSuccesses:1,rawDropped:0,
  raw:[{id:1,model:'mediapipe',input:'Canvas',sdkSucceeded:true,requestId:1,stamp,ms:1}],
  processed:[{id:11,rawId:1,model:'mediapipe',input:'Canvas',requestId:1,stamp,status:'ready',objects:[]}],
  adoptions:[{id:11}],draws:[{id:11,width:512,height:512,alphaPixels:0}],snapshots:[]};
}
async function summarize(probe,delegate) {
 const dir=await mkdtemp(path.join(tmpdir(),'pr23-f1-'));
 try{await mkdir(path.join(dir,'browser'));if(delegate!==undefined)await writeFile(path.join(dir,'environment.json'),JSON.stringify({requestedDelegate:delegate}));await writeFile(path.join(dir,'browser/evidence.json'),JSON.stringify({runtime:{kind:'CHECKER ONLY'},evidence:{probe}}));await summarizeRun(dir);return JSON.parse(await readFile(path.join(dir,'measurements.json'),'utf8'));}finally{await rm(dir,{recursive:true,force:true});}
}
function suppressed(s) {
 assert.equal(s.inferenceAttempted,false);assert.equal(s.inferenceExecuted,false);assert.equal(s.resultAdopted,false);assert.equal(s.resultRendered,false);assert.deepEqual(s.populations,[]);assert.equal(s.execution[0].modeEvidence.status,'unverified');assert(s.execution[0].evidenceErrors.length>0);
}
function addSecondModel(p,delegate,creation) {
 p.calls++;p.sdkSuccesses++;
 p.raw.push({...p.raw[0],id:2,model:'selfie-segmenter',stamp:{scene:'astronaut.png',revision:1,stream:1}});
 if(creation)p.creates.push({model:'selfie-segmenter',delegate});
}
for(const delegate of ['CPU','GPU']) {
 const edits={
  'empty creates':p=>{p.creates=[];},
  'missing creates property':p=>{delete p.creates;},
  'wrong-model-only creation':p=>{p.creates[0].model='selfie-segmenter';},
  'missing delegate':p=>{delete p.creates[0].delegate;},
  'wrong delegate':p=>{p.creates[0].delegate=delegate==='CPU'?'GPU':'CPU';},
  'failed creation only':p=>{p.creates[0].error='CHECKER ONLY initialization failure';},
  'second call model has no creation':p=>addSecondModel(p,delegate,false),
 };
 for(const [name,edit] of Object.entries(edits))test(`F1 ${delegate} rejects ${name} without discarding generic records`,async()=>{
  const p=fixture(delegate);edit(p);const s=await summarize(p,delegate);suppressed(s);assert.equal(s.execution[0].attempts,p.calls);assert.equal(s.execution[0].successfulSdkCalls,p.sdkSuccesses);assert.equal(s.execution[0].retainedAttempts,p.raw.length);assert.equal(s.execution[0].readyProcessedResultsRetained,1);assert.equal(s.execution[0].adoptedResultsRetained,1);assert.equal(s.execution[0].renderedResultsRetained,1);
 });
 test(`F1 ${delegate} accepts matching model/delegate observation as checker control`,async()=>{const s=await summarize(fixture(delegate),delegate);assert.equal(s.execution[0].evidenceErrors.length,0);assert.equal(s.execution[0].modeEvidence.status,'verified-for-retained-models');assert.equal(s.inferenceExecuted,true);assert.equal(s.resultRendered,true);assert.equal(s.populations.length,1);});
 test(`F1 ${delegate} requires and accepts separate creation for both retained models`,async()=>{const p=fixture(delegate);addSecondModel(p,delegate,true);const s=await summarize(p,delegate);assert.equal(s.execution[0].evidenceErrors.length,0);assert.equal(s.execution[0].modeEvidence.modelDelegateChecks.length,2);assert(s.execution[0].modeEvidence.modelDelegateChecks.every(c=>c.matchingCreation));assert.equal(s.populations.length,2);});
}
test('F1 legacy undeclared evidence retains generic semantics without claiming a CPU mode',async()=>{const p=fixture('CPU');p.creates=[];const s=await summarize(p,undefined);assert.equal(s.requestedDelegate,null);assert.equal(s.execution[0].modeEvidence.status,'not-declared');assert.equal(s.inferenceExecuted,true);assert.equal(s.populations.length,1);});
test('F1 invalid declared mode cannot become eligible through matching text',async()=>{suppressed(await summarize(fixture('AUTO'),'AUTO'));});
test('F1 evicted-only counters do not identify a retained model/delegate',async()=>{const p=fixture('CPU');p.raw=[];p.rawDropped=1;p.processed=[];p.adoptions=[];p.draws=[];const s=await summarize(p,'CPU');suppressed(s);assert.equal(s.execution[0].successfulSdkCalls,1);});
test('F1 unbound failed attempts remain generic observations, not CPU-qualified attempts',async()=>{const p=fixture('CPU');p.creates=[];p.sdkSuccesses=0;p.raw[0].sdkSucceeded=false;p.raw[0].error='CHECKER ONLY failed call';p.processed=[];p.adoptions=[];p.draws=[];const s=await summarize(p,'CPU');suppressed(s);assert.equal(s.execution[0].failedSdkCallsRetained,1);assert.equal(s.execution[0].attempts,1);});
test('F1 zero-call initialization diagnostics never establish execution or timing',async()=>{const p=fixture('CPU');p.creates[0].error='CHECKER ONLY HTTP503';p.calls=0;p.sdkSuccesses=0;p.raw=[];p.processed=[];p.adoptions=[];p.draws=[];const s=await summarize(p,'CPU');assert.deepEqual(s.execution[0].evidenceErrors,[]);assert.equal(s.execution[0].modeEvidence.status,'no-inference-call-models');assert.equal(s.inferenceExecuted,false);assert.equal(s.resultRendered,false);assert.deepEqual(s.populations,[]);});
