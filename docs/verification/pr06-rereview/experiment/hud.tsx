import '../../../../src/index.css';
import {createRoot} from 'react-dom/client';
import {flushSync} from 'react-dom';
import {PipelineDetails as Before} from './Before';
import {PipelineDetails as After} from '../../../../src/components/PipelineDetails';
import {readPipelineSnapshot} from '../../../../src/services/pipelineSnapshotView';
import {PipelineMetrics,PIPELINE_STAGES} from '../../../../src/services/pipelineMetrics';
const root=createRoot(document.getElementById('hud')!);
const profiles=new Map<string,PipelineMetrics>();
for(const [profile,n] of [['empty',0],['quantile300',375],['truncated',3000]] as const){const m=new PipelineMetrics(()=>0);for(const stage of PIPELINE_STAGES)for(const outcome of ['success','failed','cancelled','stale'] as const)for(let i=0;i<n;i++)m.record({stage,outcome,at:0,ms:i%5===0?null:(i*41)%997});profiles.set(profile,m);}
function setup(version:'before'|'after',open:boolean){flushSync(()=>root.render(version==='before'?<Before snapshot={null}/>:<After snapshot={null}/>));document.querySelector('details')!.open=open;}
function html(profile:string,version:'before'|'after',open:boolean){setup(version,open);const raw=profiles.get(profile)!.snapshot();flushSync(()=>root.render(version==='before'?<Before snapshot={raw}/>:<After snapshot={readPipelineSnapshot(raw)}/>));return document.getElementById('hud')!.innerHTML;}
function trial(profile:string,version:'before'|'after',open:boolean,n:number){
 setup(version,open);const samples={snapshot:[] as number[],read:[] as number[],render:[] as number[],prepare:[] as number[],total:[] as number[]};let snapshotCalls=0,readCalls=0,renderCalls=0;
 for(let i=0;i<n;i++){
  const a=performance.now();snapshotCalls++;const raw=profiles.get(profile)!.snapshot();const b=performance.now();
  const view=version==='after'?(readCalls++,readPipelineSnapshot(raw)):null;const c=performance.now();renderCalls++;
  flushSync(()=>root.render(version==='before'?<Before snapshot={raw}/>:<After snapshot={view}/>));const d=performance.now();
  samples.snapshot.push(b-a);samples.read.push(c-b);samples.render.push(d-c);samples.prepare.push(d-b);samples.total.push(d-a);
 }
 return {profile,version,open,n,snapshotCalls,readCalls,renderCalls,samples,coverage:profiles.get(profile)!.snapshot()!.stages.sdk.success};
}
window.independentHud={trial,html};
declare global{interface Window{independentHud:{trial:typeof trial;html:typeof html}}}
