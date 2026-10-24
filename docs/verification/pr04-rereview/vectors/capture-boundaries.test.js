import { beforeEach, afterEach, it, expect, vi } from 'vitest';
import { JobLock, startFrameScheduler } from '../../../../src/services/frameScheduler';
let callbacks, sequence, handles, imageTime, current, onCapture;
beforeEach(()=>{
  vi.useFakeTimers(); callbacks=new Map();sequence=0;handles=[];imageTime=0;current=true;onCapture=()=>{};
  vi.stubGlobal('requestAnimationFrame',fn=>{callbacks.set(++sequence,fn);return sequence;});
  vi.stubGlobal('cancelAnimationFrame',id=>callbacks.delete(id));
  vi.stubGlobal('VideoFrame',class{
    timestamp=imageTime;displayWidth=8;displayHeight=8;close=vi.fn();
    constructor(){handles.push(this);onCapture();}
  });
});
afterEach(()=>{vi.useRealTimers();vi.unstubAllGlobals();});
function setup(native,extra={}){
  const video={currentTime:0,readyState:2,videoWidth:8,videoHeight:8,getVideoPlaybackQuality:()=>({totalVideoFrames:9})};
  if(native){video.requestVideoFrameCallback=fn=>{callbacks.set(++sequence,fn);return sequence;};video.cancelVideoFrameCallback=id=>callbacks.delete(id);}
  const work=vi.fn(frame=>frame.timestamp),adopt=vi.fn(),error=vi.fn(),input=vi.fn(),lock=extra.lock??new JobLock();
  const stop=startFrameScheduler({video,work,adopt,error,input,lock,current:()=>current,...extra});
  function arrive(time,count){video.currentTime=time;const [id,fn]=callbacks.entries().next().value;callbacks.delete(id);fn(0,{mediaTime:time,presentedFrames:count});}
  return{video,work,adopt,error,input,lock,stop,arrive};
}
it.each([false,true])('immutable input remains bound across mismatched metadata, bootstrap race and subsequent progress (%s)',native=>{
  const f=setup(native);imageTime=22;f.arrive(0.01,1);vi.runOnlyPendingTimers();
  expect(f.work.mock.calls.map(([frame])=>frame.timestamp)).toEqual([22]);
  f.arrive(9,99);vi.runOnlyPendingTimers();expect(f.work).toHaveBeenCalledTimes(1);
  imageTime=23;f.arrive(9,99);vi.runOnlyPendingTimers();expect(f.work).toHaveBeenCalledTimes(2);
  f.stop();expect(handles.every(h=>h.close.mock.calls.length===1)).toBe(true);
});
it.each([false,true])('failure owns and closes image once, deduplicates failed image and recovers on changed image (%s)',native=>{
  const f=setup(native);f.work.mockImplementationOnce(()=>{throw Error('SDK');});vi.runOnlyPendingTimers();
  f.arrive(1,4);vi.runOnlyPendingTimers();expect(f.work).toHaveBeenCalledTimes(1);expect(f.error).toHaveBeenCalledTimes(1);
  imageTime=1;f.arrive(1,4);vi.runOnlyPendingTimers();expect(f.adopt).toHaveBeenCalledWith(1);
  f.stop();expect(handles.every(h=>h.close.mock.calls.length===1)).toBe(true);
});
it.each([false,true])('queued cancellation closes all temporary captures and never consumes input (%s)',native=>{
  const f=setup(native);for(let i=0;i<100;i++){imageTime++;f.arrive(i,i);}
  expect(vi.getTimerCount()).toBe(1);expect(handles.every(h=>h.close.mock.calls.length===1)).toBe(true);
  f.stop();vi.runOnlyPendingTimers();expect(f.work).not.toHaveBeenCalled();expect(callbacks.size).toBe(0);
  const token=f.lock.acquire();expect(token).not.toBeNull();f.lock.release(token);
});
it.each([false,true])('cancellation during execution capture rejects work while closing owned image (%s)',native=>{
  const f=setup(native);onCapture=()=>{current=false;};vi.runOnlyPendingTimers();f.stop();
  expect(f.work).not.toHaveBeenCalled();expect(f.adopt).not.toHaveBeenCalled();
  expect(handles.every(h=>h.close.mock.calls.length===1)).toBe(true);expect(f.lock.acquire()).not.toBeNull();
});
it.each([false,true])('stale async finish cannot adopt, close replacement resources or release foreign ownership (%s)',async native=>{
  const lock=new JobLock();let resolve;const f=setup(native,{lock});f.work.mockImplementationOnce(()=>new Promise(yes=>{resolve=yes;}));
  vi.runOnlyPendingTimers();const owned=handles.find(h=>h.close.mock.calls.length===0);expect(owned).toBeDefined();
  f.stop();expect(owned.close).not.toHaveBeenCalled();const next=setup(native,{lock});vi.runOnlyPendingTimers();expect(next.work).not.toHaveBeenCalled();
  lock.release(Symbol('foreign'));expect(lock.acquire()).toBeNull();resolve(3);await vi.runAllTimersAsync();
  expect(owned.close).toHaveBeenCalledTimes(1);expect(f.adopt).not.toHaveBeenCalled();imageTime=1;next.arrive(1,1);vi.runOnlyPendingTimers();
  expect(next.work).toHaveBeenCalledTimes(1);next.stop();expect(handles.every(h=>h.close.mock.calls.length===1)).toBe(true);
});
it('a failed execution capture stops callbacks, releases its token and lets a fresh session retry',()=>{
  const lock=new JobLock();const f=setup(true,{lock});onCapture=()=>{throw Error('capture');};vi.runOnlyPendingTimers();
  // A native constructor that throws does not hand a frame to the caller.
  expect(f.error).toHaveBeenCalledTimes(1);expect(callbacks.size).toBe(0);f.stop();
  onCapture=()=>{};const next=setup(true,{lock});vi.runOnlyPendingTimers();expect(next.work).toHaveBeenCalledTimes(1);next.stop();
});
