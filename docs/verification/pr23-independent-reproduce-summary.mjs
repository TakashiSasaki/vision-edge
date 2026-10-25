// MODEL-FREE CHECKER DATA ONLY. Does not import/run SDK, obtain assets or change pins.
// Node24: node reproduce-summary.mjs /absolute/frozen-pr23-checkout [output-directory]
// Exit0 means observations completed, not that the validation requirement passed.
import {mkdir,readFile,writeFile,mkdtemp} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
const root=path.resolve(process.argv[2]??'.');
assert.equal(execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),'a21e7509636c364650418ba236a7e330a813f0c2');
const {summarizeRun}=await import(pathToFileURL(path.join(root,'scripts/real-sdk/summarize.mjs')));
const base=process.argv[3]?path.resolve(process.argv[3]):await mkdtemp(path.join(tmpdir(),'pr23-summary-'));const rows=[];
for(const kind of ['cpu-observed','gpu-observed','missing-create','missing-delegate','wrong-model-only']){
 const dir=path.join(base,kind);await mkdir(path.join(dir,'browser'),{recursive:true});
 await writeFile(path.join(dir,'environment.json'),JSON.stringify({requestedDelegate:'CPU'}));
 const creates=kind==='missing-create'?[]:[{model:kind==='wrong-model-only'?'selfie-segmenter':'mediapipe',...(kind==='missing-delegate'?{}:{delegate:kind==='gpu-observed'?'GPU':'CPU'})}];
 const raw=[{id:1,model:'mediapipe',input:'Canvas',sdkSucceeded:true,requestId:1,stamp:{scene:'coffee.png',revision:1,stream:1},ms:1}];
 await writeFile(path.join(dir,'browser/evidence.json'),JSON.stringify({runtime:{kind:'CHECKER ONLY; NO SDK/MODEL EXECUTED'},evidence:{probe:{creates,calls:1,sdkSuccesses:1,rawDropped:0,raw,processed:[],adoptions:[],draws:[]}}}));
 await summarizeRun(dir);const r=JSON.parse(await readFile(path.join(dir,'measurements.json'),'utf8'));rows.push({kind,requested:r.requestedDelegate,observed:r.execution[0].observedCreateDelegates,errors:r.execution[0].evidenceErrors,inferenceExecuted:r.inferenceExecuted,populations:r.populations.length});
}
await writeFile(path.join(base,'observations.json'),JSON.stringify(rows,null,2)+'\n');console.log(JSON.stringify({output:base,rows},null,2));
