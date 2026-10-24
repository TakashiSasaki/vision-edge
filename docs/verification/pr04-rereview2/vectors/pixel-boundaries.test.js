import {beforeEach,afterEach,it,expect,vi} from 'vitest';
import {startFrameScheduler,JobLock} from '../../../../src/services/frameScheduler';
import {PerformanceMetrics} from '../../../../src/services/performanceMetrics';
let callbacks,id,canvases,sampler,reads,fault,onCopy,comparisons;
function rememberSampler(canvas){sampler=canvas;}
beforeEach(()=>{
 vi.useFakeTimers();callbacks=new Map();id=0;canvases=[];sampler=null;reads=0;fault=null;onCopy=()=>{};comparisons=0;
 vi.stubGlobal('VideoFrame',undefined);
 vi.stubGlobal('requestAnimationFrame',fn=>{callbacks.set(++id,fn);return id;});
 vi.stubGlobal('cancelAnimationFrame',key=>callbacks.delete(key));
 const contexts=new WeakMap();
 vi.spyOn(HTMLCanvasElement.prototype,'getContext').mockImplementation(function(){
  if(!canvases.includes(this))canvases.push(this);
  if(fault==='owned-context'&&sampler&&this!==sampler)return null;
  if(contexts.has(this))return contexts.get(this);
  let bytes=new Uint8ClampedArray();
  const ctx={
   drawImage:source=>{rememberSampler(this);bytes=source.pixels.slice();},
   getImageData:(_x,_y,width,height)=>{
    if(this===sampler){reads++;if(fault==='read')throw Error('readback');}
    return{width,height,data:fault==='length'?bytes.slice(1):bytes.slice()};
   },
   putImageData:image=>{if(fault==='copy')throw Error('copy');bytes=image.data.slice();onCopy();}
  };
  contexts.set(this,ctx);return ctx;
 });
 const every=Uint8ClampedArray.prototype.every;
 vi.spyOn(Uint8ClampedArray.prototype,'every').mockImplementation(function(fn,receiver){
  return every.call(this,(value,index,array)=>{comparisons++;return fn.call(receiver,value,index,array);});
 });
});
afterEach(()=>{vi.useRealTimers();vi.unstubAllGlobals();});
function setup(native=false,options={}){
 const video={currentTime:0,videoWidth:4,videoHeight:2,readyState:2,pixels:new Uint8ClampedArray(32)};
 video.pixels[0]=255;video.pixels[3]=255;video.pixels[31]=255;
 if(native){video.requestVideoFrameCallback=fn=>{callbacks.set(++id,fn);return id;};video.cancelVideoFrameCallback=key=>callbacks.delete(key);}
 let current=true,now=10;const metrics=new PerformanceMetrics(()=>now),inputs=vi.fn(()=>metrics.input()),work=vi.fn(input=>input),adopt=vi.fn(),error=vi.fn(),lock=options.lock??new JobLock();
 const stop=startFrameScheduler({video,current:()=>current,lock,work,adopt,error,input:inputs,...options});
 function tick(time=video.currentTime,count=time){video.currentTime=time;const entry=callbacks.entries().next().value;if(!entry)throw Error('No scheduled acquisition');callbacks.delete(entry[0]);entry[1](0,{mediaTime:time,presentedFrames:count});}
 return{video,work,adopt,error,inputs,stop,tick,lock,metrics,setNow:value=>{now=value;},invalidate:()=>{current=false;}};
}
it.each([false,true])('same bytes ignore drifting clocks and fresh metadata, expire input, with bounded work/storage (%s)',native=>{
 const f=setup(native);vi.runOnlyPendingTimers();expect(f.work).toHaveBeenCalledTimes(1);
 const owned=f.work.mock.calls[0][0];expect(owned.width).toBe(0);expect(owned.height).toBe(0);
 for(let i=1;i<=1000;i++){f.setNow(i*10+10);f.tick(i/100,i);vi.runOnlyPendingTimers();}
 expect(f.work).toHaveBeenCalledTimes(1);expect(f.inputs).toHaveBeenCalledTimes(1);
 expect(f.metrics.snapshot().stalled).toBe(true);expect(vi.getTimerCount()).toBe(0);
 expect(canvases).toHaveLength(2);expect(reads).toBe(1002);expect(comparisons).toBeLessThanOrEqual(1002*32);
 f.stop();expect(canvases.every(c=>c.width===0&&c.height===0)).toBe(true);expect(callbacks.size).toBe(0);
});
it.each([false,true])('identity includes all bytes and dimensions, and execution owns the precise sampled image (%s)',native=>{
 const f=setup(native);const consumed=[];
 f.work.mockImplementation(input=>{f.video.pixels[10]^=255;const data=input.getContext('2d').getImageData(0,0,input.width,input.height).data;consumed.push({width:input.width,height:input.height,data:Array.from(data)});return input;});
 const before=Array.from(f.video.pixels);vi.runOnlyPendingTimers();expect(consumed[0].data).toEqual(before);
 const replacement=new Uint8ClampedArray(before);[replacement[0],replacement[4]]=[replacement[4],replacement[0]];
 f.video.pixels=replacement;f.tick(0,2);vi.runOnlyPendingTimers();expect(consumed[1].data).toEqual(Array.from(replacement).map((value,i)=>i===10?value^255:value));
 // Use the last consumed bytes, changing only the interpretation of dimensions.
 f.video.pixels=new Uint8ClampedArray(consumed[1].data);f.video.videoWidth=2;f.video.videoHeight=4;f.tick(0,3);vi.runOnlyPendingTimers();
 expect(consumed[2].width).toBe(2);expect(consumed[2].height).toBe(4);expect(f.work).toHaveBeenCalledTimes(3);f.stop();
});
it.each(['read','length','owned-context','copy'])('capture failure after acquisition stops explicitly and releases lock/owned canvas: %s',stage=>{
 const f=setup();fault=stage;vi.runOnlyPendingTimers();
 expect(f.work).not.toHaveBeenCalled();expect(f.error).toHaveBeenCalledTimes(1);expect(callbacks.size).toBe(0);
 expect(canvases.every(c=>c.width===0&&c.height===0)).toBe(true);const token=f.lock.acquire();expect(token).not.toBeNull();f.lock.release(token);f.stop();
});
it('copy-time generation invalidation cannot reach SDK and still disposes its owned snapshot',()=>{
 const f=setup();onCopy=f.invalidate;vi.runOnlyPendingTimers();expect(f.work).not.toHaveBeenCalled();f.stop();
 expect(canvases.every(c=>c.width===0&&c.height===0)).toBe(true);expect(f.lock.acquire()).not.toBeNull();
});
it('unchanged failed image is consumed once, changed bytes recover without replacing model ownership',()=>{
 const f=setup(true);f.work.mockImplementationOnce(()=>{throw Error('SDK');});vi.runOnlyPendingTimers();
 for(let i=1;i<20;i++){f.tick(i,i);vi.runOnlyPendingTimers();}
 expect(f.work).toHaveBeenCalledTimes(1);expect(f.error).toHaveBeenCalledTimes(1);
 f.video.pixels[17]=2;f.tick(20,20);vi.runOnlyPendingTimers();expect(f.adopt).toHaveBeenCalledTimes(1);
 f.stop();expect(canvases.every(c=>c.width===0&&c.height===0)).toBe(true);
});
it('an old Promise owns its snapshot until finish; stop/finally cannot erase replacement sampler or unlock foreign work',async()=>{
 const lock=new JobLock();const first=setup(true,{lock});let resolve;
 first.work.mockImplementationOnce(()=>new Promise(yes=>{resolve=yes;}));vi.runOnlyPendingTimers();
 const image=first.work.mock.calls[0][0],firstSampler=sampler;expect(image.width).toBe(4);first.stop();
 expect(firstSampler.width).toBe(0);expect(image.width).toBe(4);
 const next=setup(true,{lock});vi.runOnlyPendingTimers();const nextSampler=sampler;expect(next.work).not.toHaveBeenCalled();expect(nextSampler.width).toBe(4);
 lock.release(Symbol('foreign'));expect(lock.acquire()).toBeNull();resolve(1);await vi.runAllTimersAsync();
 expect(image.width).toBe(0);expect(nextSampler.width).toBe(4);expect(first.adopt).not.toHaveBeenCalled();
 next.tick(1,1);vi.runOnlyPendingTimers();expect(next.work).toHaveBeenCalledTimes(1);next.stop();expect(nextSampler.width).toBe(0);
});
it.each([undefined,null,{totalVideoFrames:0},{totalVideoFrames:-1},{totalVideoFrames:NaN},{totalVideoFrames:Infinity}])('unusable playback-quality return still selects owned pixels: %j',value=>{
 const f=setup();f.video.getVideoPlaybackQuality=()=>value;vi.runOnlyPendingTimers();
 expect(f.work.mock.calls[0][0]).toBeInstanceOf(HTMLCanvasElement);
 for(let i=1;i<5;i++){f.tick(i,i);vi.runOnlyPendingTimers();}
 expect(f.work).toHaveBeenCalledTimes(1);f.video.pixels[18]=7;f.tick(5,5);vi.runOnlyPendingTimers();expect(f.work).toHaveBeenCalledTimes(2);f.stop();
 expect(canvases.every(c=>c.width===0&&c.height===0)).toBe(true);
});
