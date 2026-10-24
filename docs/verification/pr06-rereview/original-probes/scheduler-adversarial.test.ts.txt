import {afterEach,expect,it,vi} from 'vitest';
import {JobLock,startFrameScheduler} from '../../../src/services/frameScheduler';
import {PipelineMetrics,type PipelineObserver} from '../../../src/services/pipelineMetrics';
afterEach(()=>{vi.useRealTimers();vi.unstubAllGlobals();});
for(const fault of ['none','work','adopt','close'] as const){
 it(`scheduler ${fault} outcome matches disabled under throwing observer/clock`,()=>{
  const run=(mode:string)=>{
   vi.useFakeTimers();let id=0,calls=0,adopts=0;const cause=Error('product');const frames:{closed:number}[]=[];const errors:unknown[]=[];
   vi.stubGlobal('VideoFrame',class{timestamp=0;displayWidth=2;displayHeight=2;index=++id;closed=0;constructor(){frames.push(this);}close(){this.closed++;if(this.index===2&&fault==='close')throw cause;}});
   const video=Object.defineProperties(document.createElement('video'),{readyState:{value:2},videoWidth:{value:2},videoHeight:{value:2},requestVideoFrameCallback:{value:()=>1},cancelVideoFrameCallback:{value:()=>{}}});
   const m=new PipelineMetrics(()=>0);const o:PipelineObserver|undefined=mode==='off'?undefined:mode==='valid'?m:{now(){if(mode==='clock')throw Error('clock');return 0;},record(){throw Error('observer');}};const lock=new JobLock();
   const stop=startFrameScheduler({video,lock,current:()=>true,instrumentation:o,input(){},work(){calls++;if(fault==='work')throw cause;return 42;},adopt(n){expect(n).toBe(42);adopts++;if(fault==='adopt')throw cause;},error(e){errors.push(e);}});vi.runOnlyPendingTimers();stop();
   expect(errors.every(x=>x===cause)).toBe(true);expect(frames.map(x=>x.closed)).toEqual([1,1]);const token=lock.acquire();expect(token).not.toBeNull();lock.release(token!);
   const result={calls,adopts,errors:errors.length,closed:frames.map(x=>x.closed)};vi.useRealTimers();return result;
  };
  const expected=run('off');for(const mode of ['valid','observer','clock'])expect(run(mode)).toEqual(expected);
 });
}
it.each(['cancelled','stale'] as const)('retired async %s owner keeps image and foreign lock until completion',async outcome=>{
 vi.useFakeTimers();let stamp=0;const frames:{closed:number}[]=[];
 vi.stubGlobal('VideoFrame',class{timestamp=stamp;displayWidth=2;displayHeight=2;closed=0;constructor(){frames.push(this);}close(){this.closed++;}});
 const video=Object.defineProperties(document.createElement('video'),{readyState:{value:2},videoWidth:{value:2},videoHeight:{value:2},requestVideoFrameCallback:{value:()=>1},cancelVideoFrameCallback:{value:()=>{}}});const lock=new JobLock();const m=new PipelineMetrics(()=>stamp);let valid=true;let resolve!:(n:number)=>void;const adopt=vi.fn();
 const stop=startFrameScheduler({video,lock,current:()=>valid,instrumentation:m,input(){},work:()=>new Promise<number>(yes=>resolve=yes),adopt,error:e=>{throw e;}});vi.runOnlyPendingTimers();stamp=20;
 if(outcome==='cancelled')stop();else valid=false;expect(frames[1].closed).toBe(0);expect(lock.acquire()).toBeNull();lock.release(Symbol('foreign'));expect(lock.acquire()).toBeNull();resolve(1);await vi.runAllTimersAsync();expect(adopt).not.toHaveBeenCalled();expect(frames.every(f=>f.closed===1)).toBe(true);expect(m.snapshot()!.stages.job[outcome]).toMatchObject({count:1,mean:20});expect(m.snapshot()!.stages.job.success.count).toBe(0);expect(lock.acquire()).not.toBeNull();stop();
});
