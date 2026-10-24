// Independent CHECKER-ONLY vectors: no SDK/model imports/calls, no invented inference acceptance.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const source = process.env.PR07_REVIEW_SOURCE ?? '/tmp/pr07-rereview-current';
const { validateManifest, manifest } = await import(pathToFileURL(path.join(source, 'scripts/real-sdk/assets.mjs')));
const { checkNegativeScene } = await import(pathToFileURL(path.join(source, 'e2e-real-sdk/acceptance.ts')));
const { summarizeRun } = await import(pathToFileURL(path.join(source, 'scripts/real-sdk/summarize.mjs')));
const clone = value => structuredClone(value);
for (const [name, value] of [['null', null], ['boolean', true], ['array', []], ['scalar', 1], ['schema string', {schema:'1', assets:[]}], ['schema omitted', {assets:[]}], ['assets object', {schema:1, assets:{}}]]) {
  test(`independent F01 invalid top level: ${name}`, () => assert.throws(() => validateManifest(value), /INVALID_ASSET_MANIFEST/));
}
const manifestEdits = {
  'empty array': m => { m.assets=[]; },
  'unsupported negative schema': m => {m.schema=-1;},
  'SDK-only inventory': m => {m.assets=m.assets.filter(a=>a.packageFile);},
  'null entry': m => {m.assets[0]=null;},
  'array entry': m => {m.assets[0]=[];},
  'prototype name identity': m => {m.assets[0].id='constructor';},
  'prototype key identity': m => {m.assets[0].id='__proto__';},
  'swapped versions': m => {const a=m.assets[0];a.version='0.10.15';},
  'SDK alternate package': m => {m.assets.at(-1).packageFile='node_modules/other/vision_bundle.mjs';},
  'non-SDK package path': m => {m.assets[0].packageFile='node_modules/@mediapipe/tasks-vision/wasm/'+m.assets[0].file;},
  'wrong boolean required': m => {m.assets[0].required='true';},
  'uppercase hash': m => {m.assets[0].sha256=m.assets[0].sha256.toUpperCase();},
  'fractional size': m => {m.assets[0].size=2.5;},
  'unsafe size': m => {m.assets[0].size=Number.MAX_SAFE_INTEGER+1;},
  'negative size': m => {m.assets[0].size=-1;},
  'model half pin hash': m => {m.assets.find(a=>a.id.endsWith('.tflite')).sha256='a'.repeat(64);},
  'model half pin size': m => {m.assets.find(a=>a.id.endsWith('.tflite')).size=4;},
  'missing blocked model reason': m => {m.assets.find(a=>a.id.endsWith('.tflite')).acquisitionStatus='ready';},
  'null nonmodel despite blocked reason': m => {Object.assign(m.assets[0],{size:null,sha256:null,acquisitionStatus:'blocked: independent control'});},
  'empty provenance': m => {m.assets[0].production='  ';},
  'missing license': m => {delete m.assets[0].license;},
};
for (const a of manifest.assets) manifestEdits[`remove ${a.id}`]=m=>{m.assets=m.assets.filter(row=>row.id!==a.id);};
for (const [name, edit] of Object.entries(manifestEdits)) test(`independent F01 rejects ${name}`, () => {const m=clone(manifest);edit(m);assert.throws(()=>validateManifest(m),/INVALID_ASSET_MANIFEST/);});
test('independent F01 full blocked inventory is eligible, unchanged and order independent',()=>{
 const m=clone(manifest);m.assets.reverse();const before=JSON.stringify(m);assert.equal(validateManifest(m),m);assert.equal(JSON.stringify(m),before);assert.equal(m.assets.length,9);assert.equal(m.assets.filter(a=>a.sha256===null&&a.size===null).length,2);
});
function chain(model, input, count=5) {
 const stamp={scene:'empty',revision:4,stream:2}; const boundary={afterCallId:10,afterResultId:10,model,input,requestId:42,stamp};
 const raw=Array.from({length:count},(_,i)=>({id:11+i,model,input,requestId:42,stamp:clone(stamp),sdkSucceeded:true,ms:1}));
 const processed=raw.map((r,i)=>({id:11+i,rawId:r.id,model,input,requestId:42,stamp:clone(stamp),status:'ready',objects:i<2?[{label:model==='mediapipe'?'cup':'person'}]:[]}));
 const last=processed.at(-1).id;const probe={raw,processed,adoptions:[{id:last}],draws:[{id:last,width:512,height:512,alphaPixels:0}]};return {probe,boundary,expected:model==='mediapipe'?'cup':'person'};
}
const mutations={
 'all new attempts throw, old empty processed record': p=>{p.raw.forEach(r=>{r.sdkSucceeded=false;r.error='CHECKER ONLY failed attempt';});p.processed=[{...p.processed[0],id:1,rawId:1,objects:[]}];},
 'zero new attempts with stale outputs':p=>{p.raw=[];},
 'unknown return outcome':p=>{delete p.raw.at(-1).sdkSucceeded;},
 'unknown provenance':p=>{p.raw.forEach(r=>{r.stamp=null;});},
 'previous input scene':p=>{p.raw.forEach(r=>{r.stamp.scene='background';});},
 'previous camera session':p=>{p.raw.forEach(r=>{r.stamp.stream=1;});},
 'previous transition':p=>{p.raw.forEach(r=>{r.stamp.revision=3;});},
 'attempt ID gap':p=>{p.raw.shift();},
 'duplicate attempt IDs':p=>{p.raw.at(-1).id=p.raw[0].id;},
 'only four successful returns':p=>{p.raw.pop();p.processed.pop();},
 'future generation':p=>{p.raw.at(-1).requestId++;},
 'other model at last call':p=>{p.raw.at(-1).model=p.raw[0].model==='mediapipe'?'selfie-segmenter':'mediapipe';},
 'implicit input fallback':p=>{p.raw.at(-1).input=p.raw[0].input==='Canvas'?'HTMLVideoElement':'Canvas';},
 'stale result IDs':p=>{p.processed.forEach((r,i)=>{r.id=i+1;});},
 'result bound to earlier call':p=>{p.processed.at(-1).rawId=10;},
 'result from other generation':p=>{p.processed.at(-1).requestId++;},
 'result has wrong actual stamp':p=>{p.processed.at(-1).stamp.stream=1;},
 'not-ready latest output':p=>{p.processed.at(-1).status='not-ready';},
 'missing latest processed result':p=>{p.processed.pop();},
 'duplicate result IDs':p=>{p.processed.at(-1).id=p.processed[0].id;},
 'no actual adoption':p=>{p.adoptions=[];},
 'only stale adoption':p=>{p.adoptions[0].id=1;},
 'no actual visible copy':p=>{p.draws=[];},
 'only stale visible copy':p=>{p.draws[0].id=1;},
 'invalid visible dimensions':p=>{p.draws[0].width=0;},
 'unknown alpha coverage':p=>{p.draws[0].alphaPixels=NaN;},
 'later failed call after complete eligible chain':p=>{p.raw.push({...p.raw[0],id:16,sdkSucceeded:false,error:'CHECKER ONLY late failure'});},
};
for(const model of ['mediapipe','selfie-segmenter']) for(const input of ['VideoFrame','Canvas','HTMLVideoElement']) {
 test(`independent F02 eligible chain with retained positives ${model}/${input}`,()=>{const {probe,boundary,expected}=chain(model,input);assert.equal(checkNegativeScene(probe,boundary,expected).status,'pass');});
 for(const [name,edit] of Object.entries(mutations)) test(`independent F02 ${model}/${input} rejects ${name}`,()=>{const {probe,boundary,expected}=chain(model,input);edit(probe);assert.notEqual(checkNegativeScene(probe,boundary,expected).status,'pass');});
 test(`independent F02 ${model}/${input} rejects residual object after retention`,()=>{const {probe,boundary,expected}=chain(model,input,9);probe.processed[3].objects=[{label:expected}];assert.equal(checkNegativeScene(probe,boundary,expected).status,'fail');});
}
async function summarize(probe) {
 const dir=await mkdtemp(path.join(os.tmpdir(),'pr07-independent-summary-'));
 try{await mkdir(path.join(dir,'browser/case'),{recursive:true});await writeFile(path.join(dir,'browser/case/evidence.json'),JSON.stringify({runtime:{scope:'CHECKER ONLY; no SDK/model'},evidence:{probe}}));await summarizeRun(dir);return JSON.parse(await readFile(path.join(dir,'measurements.json'),'utf8'));}
 finally{await rm(dir,{recursive:true,force:true});}
}
function summaryChain(model,input) {
 const {probe}=chain(model,input);probe.raw.forEach((r,i)=>{r.id=i+1;r.stamp.scene=model==='mediapipe'?'coffee.png':'astronaut.png';});probe.processed.forEach((r,i)=>{r.id=i+1;r.rawId=i+1;r.stamp.scene=model==='mediapipe'?'coffee.png':'astronaut.png';});probe.adoptions=[{id:5}];probe.draws=[{id:5,width:512,height:512,alphaPixels:30}];return {...probe,calls:5,sdkSuccesses:5,rawDropped:0,snapshots:[],creates:[]};
}
for(const model of ['mediapipe','selfie-segmenter']) for(const input of ['VideoFrame','Canvas','HTMLVideoElement']) {
 const variants={
  'successful return without processing':{edit:p=>{p.processed=[];},expected:[true,true,false,false]},
  'processing without adoption':{edit:p=>{p.adoptions=[];},expected:[true,true,false,false]},
  'adoption without copy':{edit:p=>{p.draws=[];},expected:[true,true,true,false]},
  'complete linked stages':{edit:()=>{},expected:[true,true,true,true]},
  'all attempts fail':{edit:p=>{p.sdkSuccesses=0;p.raw.forEach(r=>{r.sdkSucceeded=false;r.error='CHECKER ONLY';});},expected:[true,false,false,false]},
  'all outcomes unknown':{edit:p=>{delete p.sdkSuccesses;p.raw.forEach(r=>{delete r.sdkSucceeded;});},expected:[true,false,false,false]},
  'stale ready/adopt/draw':{edit:p=>{p.processed.forEach(r=>{r.requestId++;});},expected:[true,true,false,false]},
  'duplicate raw IDs':{edit:p=>{p.raw.at(-1).id=1;},expected:[false,false,false,false]},
  'counter-only zero retained evidence':{edit:p=>{p.raw=[];p.calls=1;p.sdkSuccesses=1;p.rawDropped=0;},expected:[false,false,false,false]},
  'counter smaller than raw':{edit:p=>{p.calls=0;},expected:[false,false,false,false]},
 };
 for(const [name,{edit,expected}] of Object.entries(variants)) test(`independent F03 ${model}/${input}: ${name}`,async()=>{const p=summaryChain(model,input);edit(p);const r=await summarize(p);assert.deepEqual([r.inferenceAttempted,r.inferenceExecuted,r.resultAdopted,r.resultRendered],expected);if(!r.inferenceExecuted)assert.deepEqual(r.populations,[]);});
}
test('independent F03 identical original zero-call fixture: old true, current false',async()=>{
 const fixture={raw:[],creates:[{model:'selfie-segmenter',error:'initialization failed'}],snapshots:[],rawDropped:0};
 const old=await import(pathToFileURL(path.join(process.env.PR07_REVIEW_OLD ?? '/tmp/pr07-review-current','scripts/real-sdk/summarize.mjs')));
 const dir=await mkdtemp(path.join(os.tmpdir(),'pr07-identical-original-summary-'));
 try{await mkdir(path.join(dir,'browser/case'),{recursive:true});await writeFile(path.join(dir,'browser/case/evidence.json'),JSON.stringify({runtime:{},evidence:{probe:fixture}}));await old.summarizeRun(dir);const before=JSON.parse(await readFile(path.join(dir,'measurements.json'),'utf8'));await summarizeRun(dir);const after=JSON.parse(await readFile(path.join(dir,'measurements.json'),'utf8'));assert.equal(before.inferenceExecuted,true);assert.equal(after.inferenceExecuted,false);assert.equal(after.inferenceAttempted,false);assert.deepEqual(after.populations,[]);console.log(JSON.stringify({scope:'CHECKER ONLY identical zero-call fixture',before,after}));}finally{await rm(dir,{recursive:true,force:true});}
});
