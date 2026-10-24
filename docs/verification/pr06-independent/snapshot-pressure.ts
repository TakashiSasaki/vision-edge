import {performance} from 'node:perf_hooks';
import {writeFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import {PipelineMetrics,PipelineProbe,PIPELINE_STAGES} from '../../../src/services/pipelineMetrics';
const outcomes=['success','failed','cancelled','stale'] as const;
const m=new PipelineMetrics(()=>0);
for(const stage of PIPELINE_STAGES)for(const outcome of outcomes)for(let i=0;i<2048;i++)m.record({stage,outcome,at:0,ms:(i*41)%991});
for(let i=0;i<10;i++)m.snapshot();const snapshots:number[]=[];
for(let i=0;i<100;i++){const t=performance.now();const s=m.snapshot()!;snapshots.push(performance.now()-t);assert.equal(s.stages.engine.success.count,2048);assert.equal(s.stages.engine.success.quantileCount,300);}
const samples:{enabled:boolean;pair:number;ms:number;work:number;events:number}[]=[];
for(let pair=0;pair<8;pair++)for(const enabled of pair%2?[true,false]:[false,true]){
 const collector=enabled?new PipelineMetrics(()=>performance.now()):undefined;const p=collector?new PipelineProbe(collector):undefined;let work=0;const t=performance.now();
 for(let i=0;i<10000;i++){const stamp=p?.start();work+=(i*17)%37;p?.end('comparison',stamp);}
 const elapsed=performance.now()-t;samples.push({enabled,pair,ms:elapsed,work,events:collector?.snapshot()?.stages.comparison.success.count??0});
}
function describe(v:number[]){const x=v.slice().sort((a,b)=>a-b);const mean=v.reduce((a,b)=>a+b)/v.length;return {count:v.length,median:(x[Math.floor((x.length-1)/2)]+x[Math.floor(x.length/2)])/2,p95:x[Math.ceil(.95*x.length)-1],min:x[0],max:x.at(-1),mean,variance:v.reduce((a,b)=>a+(b-mean)**2,0)/v.length};}
assert(samples.every(s=>s.work===samples[0].work));const result={method:'Node synthetic aggregation pressure, not browser/React/GPU. Full 40 rings x 2048 samples; snapshots exclude React. Probe loop is independent matched integer work, no SDK.',environment:{node:process.version},snapshot:describe(snapshots),snapshotSamples:snapshots,probeLoopSamples:samples,probeLoop:{disabled:describe(samples.filter(s=>!s.enabled).map(s=>s.ms)),enabled:describe(samples.filter(s=>s.enabled).map(s=>s.ms)),perEventPairedDelta:describe(Array.from({length:8},(_,pair)=>(samples.find(s=>s.pair===pair&&s.enabled)!.ms-samples.find(s=>s.pair===pair&&!s.enabled)!.ms)/10000))}};
writeFileSync(new URL('./snapshot-pressure.json',import.meta.url),JSON.stringify(result,null,2));console.log(JSON.stringify({snapshot:result.snapshot,probeLoop:result.probeLoop}));
