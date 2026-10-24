import {expect,it} from 'vitest';
import {readPipelineSnapshot} from '../../../src/services/pipelineSnapshotView';
import {PipelineMetrics,PIPELINE_STAGES,type StageStatistics} from '../../../src/services/pipelineMetrics';
const outcomes=['success','failed','cancelled','stale'] as const;
const row=():StageStatistics=>({count:3,measured:2,missing:1,totalMs:8,mean:4,median:4,p95:8,max:8,quantileCount:2,truncated:false});
function fixture(){return {stages:Object.fromEntries(PIPELINE_STAGES.map(s=>[s,Object.fromEntries(outcomes.map(o=>[o,row()]))])),results:{hidden:2,'first-draw':1,unbound:1},resultsTruncated:false,pendingResults:0,unclocked:0,captureDutyPercent:null};}
it('preserves independently specified partial rows, all forty cells, and owns every nested output object',()=>{
 const source=fixture();const text=JSON.stringify(source);const result=readPipelineSnapshot(source)!;expect(result).toEqual(source);expect(JSON.stringify(source)).toBe(text);expect(result).not.toBe(source);expect(result.results).not.toBe(source.results);
 for(const stage of PIPELINE_STAGES){expect(result.stages[stage]).not.toBe(source.stages[stage]);for(const outcome of outcomes)expect(result.stages[stage][outcome]).not.toBe(source.stages[stage][outcome]);}
 source.stages.sdk.success.mean=99;source.results.hidden=44;expect(result.stages.sdk.success!.mean).toBe(4);expect(result.results!.hidden).toBe(2);
});
it.each(['count','measured','missing','quantileCount'] as const)('reads counter %s once and never coerces hostile second values',key=>{
 const source=fixture();let reads=0;const before=source.stages.sdk.success[key];Object.defineProperty(source.stages.sdk.success,key,{get(){reads++;return reads===1?before:{toString(){throw Error('coerced');}};}});const result=readPipelineSnapshot(source)!;expect(reads).toBe(1);expect(result.stages.sdk.success![key]).toBe(before);
});
it.each([NaN,Infinity,-Infinity,-.001,'0',0n,Symbol('time'),{toFixed(){throw Error('coercion');},valueOf(){throw Error('coercion');}}])('unavailable individual duration %s preserves good neighbors and avoids coercion',bad=>{
 const source=fixture();Object.assign(source.stages.sdk.success,{mean:bad});const view=readPipelineSnapshot(source)!;expect(view.stages.sdk.success).toEqual({...row(),mean:null});expect(view.stages.draw.success).toEqual(row());
});
it.each([{count:2},{measured:4},{missing:-1},{quantileCount:301},{quantileCount:3},{truncated:'false'}])('rejects incoherent metadata %j without inventing full coverage',change=>{
 const source=fixture();Object.assign(source.stages.sdk.success,change);const view=readPipelineSnapshot(source)!;expect(view.stages.sdk.success).toBeNull();expect(view.stages.draw.success).toEqual(row());
});
it('does not enumerate huge unknown keys, walk cycles, call unknown accessors or retain proxies',()=>{
 const source=fixture();const reads:string[]=[];const proxyCache=new WeakMap<object,object>();
 const wrap=(value:unknown):unknown=>{
  if(typeof value!=='object'||value===null)return value;const existing=proxyCache.get(value);if(existing)return existing;
  const p=new Proxy(value,{ownKeys(){throw Error('enumeration forbidden');},get(target,key,receiver){reads.push(String(key));if(key==='toJSON'||key==='toString'||key==='unknown')throw Error('unknown access');return wrap(Reflect.get(target,key,receiver));},set(){throw Error('mutation forbidden');},getOwnPropertyDescriptor(target,key){return Reflect.getOwnPropertyDescriptor(target,key);}});proxyCache.set(value,p);return p;
 };
 Object.defineProperty(source,'unknown',{get(){throw Error('unknown root field');}});Object.defineProperty(source.stages,'unknown',{get(){throw Error('unknown stage');}});Object.assign(source.results,{unknown:source});
 const result=readPipelineSnapshot(wrap(source))!;expect(result.stages.sdk.success).toEqual(row());expect(reads.length).toBeLessThan(600);expect(result.results).toEqual({hidden:2,'first-draw':1,unbound:1});expect(result.results).not.toBe(source.results);expect(reads).not.toContain('unknown');console.log('bounded projection reads',reads.length);
});
it.each(['stages','sdk','success','mean','results','hidden'] as const)('contains throwing accessor at relevant level %s',key=>{
 const source=fixture();const target=key==='stages'||key==='results'?source:key==='sdk'?source.stages:key==='success'?source.stages.sdk:key==='mean'?source.stages.sdk.success:source.results;
 Object.defineProperty(target,key,{get(){throw Error('fault at '+key);}});expect(()=>readPipelineSnapshot(source)).not.toThrow();const v=readPipelineSnapshot(source);if(key==='mean')expect(v!.stages.sdk.success).toBeNull();
});
it('revoked proxy and result-descriptor exceptions are unavailable, not application exceptions',()=>{
 const p=Proxy.revocable({},{});p.revoke();expect(readPipelineSnapshot(p.proxy)).toBeNull();const source=fixture();const result=new Proxy(source.results,{getOwnPropertyDescriptor(){throw Error('descriptor');}});expect(readPipelineSnapshot({...source,results:result})).toBeNull();
});
it('real 2048-retention/300-quantile/missing/overflow snapshots are preserved without mutating their input',()=>{
 const m=new PipelineMetrics(()=>0);for(let i=0;i<3000;i++)m.record({at:0,stage:'comparison',outcome:'success',ms:i%3?i:null});for(let i=0;i<2;i++)m.record({at:0,stage:'capture',outcome:'success',ms:Number.MAX_VALUE});const source=m.snapshot()!;const serialized=JSON.stringify(source);const view=readPipelineSnapshot(source)!;expect(view.stages).toEqual(source.stages);expect(view.stages.comparison.success).toMatchObject({count:2048,quantileCount:300,truncated:true});expect(view.stages.capture.success).toMatchObject({totalMs:null,mean:null,max:Number.MAX_VALUE});expect(view.captureDutyPercent).toBeNull();expect(JSON.stringify(source)).toBe(serialized);
});
it('preserves zero duration, no execution, inclusive expiry, and later zero recovery',()=>{
 let now=0;const m=new PipelineMetrics(()=>now);m.record({at:0,stage:'sdk',outcome:'success',ms:0});expect(readPipelineSnapshot(m.snapshot())!.stages.sdk.success).toMatchObject({count:1,mean:0,median:0,p95:0,totalMs:0});expect(readPipelineSnapshot(m.snapshot())!.stages.draw.success).toMatchObject({count:0,mean:null});now=5000;expect(readPipelineSnapshot(m.snapshot())!.stages.sdk.success!.count).toBe(1);now+=.125;expect(readPipelineSnapshot(m.snapshot())!.stages.sdk.success!.mean).toBeNull();m.record({at:now,stage:'sdk',outcome:'success',ms:0});expect(readPipelineSnapshot(m.snapshot())!.stages.sdk.success!.mean).toBe(0);
});
it('freezes a plain source recursively and validates without writes',()=>{
 const source=fixture();for(const s of Object.values(source.stages)){for(const r of Object.values(s))Object.freeze(r);Object.freeze(s);}Object.freeze(source.stages);Object.freeze(source.results);Object.freeze(source);expect(()=>readPipelineSnapshot(source)).not.toThrow();expect(readPipelineSnapshot(source)!.stages.sdk.success).toEqual(row());
});
