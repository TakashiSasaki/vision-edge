import React from 'react';
import { act, cleanup, fireEvent, render } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { VisionCanvas } from '../../../src/components/VisionCanvas';
import { VisionEngine, type VisionTaskFactory } from '../../../src/services/visionEngine';
import { speechService } from '../../../src/services/speechService';

afterEach(()=>{cleanup();vi.useRealTimers();vi.restoreAllMocks();Reflect.deleteProperty(navigator,'mediaDevices');Reflect.deleteProperty(HTMLVideoElement.prototype,'getVideoPlaybackQuality');});
function adjacent(value:number,direction:number){
  const bits=new DataView(new ArrayBuffer(8));bits.setFloat64(0,value);
  bits.setBigUint64(0,bits.getBigUint64(0)+BigInt((value>0)===(direction>0)?1:-1));return bits.getFloat64(0);
}
it.each([false,true])('accepts all rational corner hits and keeps nearest representable outside edges silent (transpose=%s)',async transpose=>{
  vi.useFakeTimers({toFake:['setTimeout','clearTimeout','setInterval','performance']});
  const callbacks=new Map<number,FrameRequestCallback>();let next=0;
  vi.spyOn(globalThis,'requestAnimationFrame').mockImplementation(cb=>{callbacks.set(++next,cb);return next;});
  vi.spyOn(globalThis,'cancelAnimationFrame').mockImplementation(id=>{callbacks.delete(id);});
  const ctx={clearRect:vi.fn(),drawImage:vi.fn(),save:vi.fn(),restore:vi.fn(),beginPath:vi.fn(),arc:vi.fn(),roundRect:vi.fn(),stroke:vi.fn(),fill:vi.fn(),fillText:vi.fn(),measureText:vi.fn(()=>({width:20}))};
  vi.spyOn(HTMLCanvasElement.prototype,'getContext').mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
  vi.spyOn(HTMLMediaElement.prototype,'play').mockResolvedValue();vi.spyOn(HTMLMediaElement.prototype,'pause').mockImplementation(()=>{});
  Object.defineProperty(HTMLVideoElement.prototype,'getVideoPlaybackQuality',{configurable:true,value:()=>({totalVideoFrames:1})});
  Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia:vi.fn(async()=>({getTracks:()=>[Object.assign(new EventTarget(),{stop:vi.fn()})]}))}});
  const width=transpose?480:640,height=transpose?640:480;
  const detector={detectForVideo:()=>({detections:[{categories:[{categoryName:'dog',score:0.9}],boundingBox:{originX:0,originY:0,width,height}}]}),close:vi.fn()};
  const factory={load:async()=>({}),detector:vi.fn(async()=>detector),segmenter:vi.fn()}as unknown as VisionTaskFactory;
  const engine=new VisionEngine(factory);await engine.initialize('mediapipe');
  const speak=vi.spyOn(speechService,'speak').mockReturnValue(true),find=vi.spyOn(engine,'findObjectAtPoint'),select=vi.fn();
  const view=render(<VisionCanvas engine={engine} activeModel="mediapipe" modelState={engine.getState()} facingMode="user" onToggleFacingMode={()=>{}} renderOptions={{showBoxes:true,showMasks:true,showLabels:true,showConfidence:true,confidenceThreshold:0.35,maskOpacity:0.35}} ttsConfig={{enabled:false,autoReadout:false,cooldownSec:3,rate:1,pitch:1,volume:1,voiceURI:''}} isCameraActive onCameraStart={()=>{}} onCameraStop={()=>{}} onSelectObject={select} onFrame={()=>{}}/>);
  await act(async()=>{});
  const video=view.container.querySelector('video')!,canvas=view.container.querySelector('canvas')!;
  Object.defineProperties(video,{readyState:{configurable:true,value:2},videoWidth:{configurable:true,value:width},videoHeight:{configurable:true,value:height}});
  const rect=transpose?{left:82.5,top:37.25,width:540.25,height:960.75}:{left:37.25,top:82.5,width:960.75,height:540.25};
  vi.spyOn(video,'getBoundingClientRect').mockReturnValue(rect as DOMRect);
  const draw=()=>act(()=>{for(const [id,callback]of [...callbacks]){callbacks.delete(id);callback(performance.now());}});
  draw();act(()=>{vi.advanceTimersByTime(0);vi.advanceTimersByTime(250);});draw();
  const dispatch=(x:number,y:number)=>act(()=>{fireEvent.click(canvas,{clientX:transpose?y:x,clientY:transpose?x:y});});
  const corners=[[3779/24,82.5,0,0],[21067/24,82.5,1,0],[3779/24,622.75,0,1],[21067/24,622.75,1,1]];
  for(const [x,y,nx,ny]of corners){
    dispatch(x,y);draw();const normalized=transpose?[ny,nx]:[nx,ny];
    expect(find).toHaveBeenLastCalledWith(...normalized,expect.any(Array));
    expect(ctx.arc).toHaveBeenLastCalledWith(normalized[0]*width,normalized[1]*height,24,0,Math.PI*2);
  }
  expect(select).toHaveBeenCalledTimes(4);expect(speak).toHaveBeenCalledTimes(4);find.mockClear();
  const arcCalls=ctx.arc.mock.calls.length;
  for(const [x,y]of [[adjacent(3779/24,-1),352.625],[adjacent(21067/24,1),352.625],[517.625,adjacent(82.5,-1)],[517.625,adjacent(622.75,1)]])dispatch(x,y);
  expect(find).not.toHaveBeenCalled();expect(select).toHaveBeenCalledTimes(4);expect(speak).toHaveBeenCalledTimes(4);expect(ctx.arc).toHaveBeenCalledTimes(arcCalls);
  expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1);expect(factory.detector).toHaveBeenCalledTimes(1);expect(detector.close).not.toHaveBeenCalled();
});
