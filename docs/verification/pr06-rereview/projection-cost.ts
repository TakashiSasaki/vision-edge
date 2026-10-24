import {performance} from 'node:perf_hooks';
import {writeFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import {readPipelineSnapshot} from '../../../src/services/pipelineSnapshotView';
import {PipelineMetrics,PIPELINE_STAGES} from '../../../src/services/pipelineMetrics';
const m=new PipelineMetrics(()=>0);for(const s of PIPELINE_STAGES)for(const o of ['success','failed','cancelled','stale'] as const)for(let i=0;i<3000;i++)m.record({stage:s,outcome:o,at:0,ms:i%5?i:null});
const source=m.snapshot()!;const serialized=JSON.stringify(source);let accesses=0;const cache=new WeakMap<object,object>();
function bounded(value:unknown):unknown{if(typeof value!=='object'||value===null)return value;if(cache.has(value))return cache.get(value);const p=new Proxy(value,{get(t,k,r){accesses++;return bounded(Reflect.get(t,k,r));},ownKeys(){throw Error('unbounded enumeration');},set(){throw Error('unexpected source mutation');}});cache.set(value,p);return p;}
const view=readPipelineSnapshot(bounded(source))!;assert(view&&view.stages.sdk.success);assert(accesses<600);assert.equal(JSON.stringify(source),serialized);
const boundReads=accesses;const samples:number[]=[];for(let warm=0;warm<100;warm++)readPipelineSnapshot(source);for(let rep=0;rep<30;rep++){const t=performance.now();for(let i=0;i<1000;i++)assert(readPipelineSnapshot(source));samples.push((performance.now()-t)/1000);}
const sorted=samples.slice().sort((a,b)=>a-b);const mean=samples.reduce((a,b)=>a+b)/samples.length;const output={environment:{node:process.version},method:'Node projection-only fixed 40 rows; 100 warmups; 30 batches x1000 copies; excludes collector snapshot and React/browser',boundReads,sourceMutated:false,samples,summary:{median:(sorted[14]+sorted[15])/2,p95:sorted[28],mean,min:sorted[0],max:sorted.at(-1),variance:samples.reduce((a,b)=>a+(b-mean)**2,0)/samples.length}};
writeFileSync(new URL('./projection-cost.json',import.meta.url),JSON.stringify(output,null,2)+'\n');console.log(JSON.stringify({boundReads,summary:output.summary}));
