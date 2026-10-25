// Independent CHECKER ONLY: no SDK/model execution, no model bytes/pins.
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
const target=path.resolve(process.argv[2]??'.');
assert.equal(execFileSync('git',['rev-parse','HEAD'],{cwd:target,encoding:'utf8'}).trim(),'5ee8d9f659bd08fac462671a829cb0dcbf537c49');
const {summarizeRun}=await import(pathToFileURL(path.join(target,'scripts/real-sdk/summarize.mjs')));
const root=path.resolve(process.argv[3]??'/tmp/pr23-independent-eligibility');const rows=[];
function probe(mode){const stamp={scene:'coffee.png',revision:1,stream:1};return {creates:[{model:'mediapipe',delegate:mode}],calls:1,sdkSuccesses:1,rawDropped:0,raw:[{id:1,model:'mediapipe',input:'Canvas',requestId:1,stamp,sdkSucceeded:true,ms:2}],processed:[{id:2,rawId:1,model:'mediapipe',input:'Canvas',requestId:1,stamp,status:'ready',objects:[]}],adoptions:[{id:2}],draws:[{id:2,width:512,height:512,alphaPixels:0}]};}
async function run(name,mode,edit=()=>{},expected='qualified'){
 const p=probe(mode);edit(p);const dir=root+'/'+name;await mkdir(dir+'/browser',{recursive:true});if(mode!==undefined)await writeFile(dir+'/environment.json',JSON.stringify({requestedDelegate:mode}));
 const raw=JSON.stringify({runtime:{kind:'CHECKER ONLY'},evidence:{probe:p}});const file=dir+'/browser/evidence.json';await writeFile(file,raw);await summarizeRun(dir);assert.equal(await readFile(file,'utf8'),raw);
 const r=JSON.parse(await readFile(dir+'/measurements.json','utf8'));const x=r.execution[0];assert.equal(x.attempts,p.calls);assert.equal(x.successfulSdkCalls,p.sdkSuccesses);assert.equal(x.retainedAttempts,p.raw.length);
 if(expected==='suppressed'){for(const key of ['inferenceAttempted','inferenceExecuted','resultAdopted','resultRendered'])assert.equal(r[key],false,`${name}/${key}`);assert.equal(r.populations.length,0);assert(x.evidenceErrors.length);assert.equal(x.modeEvidence.status,'unverified');}
 if(expected==='qualified'){for(const key of ['inferenceAttempted','inferenceExecuted','resultAdopted','resultRendered'])assert.equal(r[key],true,`${name}/${key}`);assert.equal(x.modeEvidence.status,'verified-for-retained-models');}
 if(expected==='evicted-success'){assert.equal(r.inferenceAttempted,true);assert.equal(r.inferenceExecuted,false);assert.equal(r.resultRendered,false);assert.equal(r.populations.length,0);assert.equal(x.modeEvidence.eligibleRetainedSuccesses,0);}
 if(expected==='legacy'){assert.equal(r.requestedDelegate,null);assert.equal(r.inferenceExecuted,true);assert.equal(x.modeEvidence.status,'not-declared');}
 rows.push({name,requested:r.requestedDelegate,expected,errors:x.evidenceErrors,modeEvidence:x.modeEvidence,rawUnchanged:true,attempts:x.attempts,successfulSdkCalls:x.successfulSdkCalls,retainedAttempts:x.retainedAttempts,inferenceAttempted:r.inferenceAttempted,inferenceExecuted:r.inferenceExecuted,resultAdopted:r.resultAdopted,resultRendered:r.resultRendered,populations:r.populations.length});
}
for(const mode of ['CPU','GPU']){
 await run(mode+'-positive',mode);
 for(const [name,edit] of [['missing',p=>{delete p.creates;}],['empty',p=>p.creates=[]],['wrong-model',p=>p.creates[0].model='selfie-segmenter'],['wrong-delegate',p=>p.creates[0].delegate=mode==='CPU'?'GPU':'CPU'],['failed-create',p=>p.creates[0].error='LOCAL_FAILURE'],['missing-delegate',p=>delete p.creates[0].delegate]])await run(mode+'-'+name,mode,edit,'suppressed');
 await run(mode+'-evicted-only',mode,p=>{p.raw=[];p.rawDropped=1;p.processed=[];p.adoptions=[];p.draws=[];},'suppressed');
 await run(mode+'-evicted-success',mode,p=>{p.calls=2;p.rawDropped=1;p.raw[0].id=2;p.raw[0].sdkSucceeded=false;p.raw[0].error='LOCAL_FAILURE';p.processed=[];p.adoptions=[];p.draws=[];},'evicted-success');
}
await run('invalid-mode','AUTO',()=>{},'suppressed');await run('legacy',undefined,p=>p.creates=[],'legacy');
await writeFile(path.join(root,'observations.json'),JSON.stringify(rows,null,2)+'\n');console.log(JSON.stringify({cases:rows.length,assertions:'PASS',summary:rows.map(x=>({name:x.name,mode:x.modeEvidence.status,executed:x.inferenceExecuted,rawUnchanged:x.rawUnchanged}))},null,2));
