import React from 'react';
import { act, cleanup, fireEvent, render } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { VisionCanvas } from '../../../src/components/VisionCanvas';
import { VisionEngine, type VisionTaskFactory } from '../../../src/services/visionEngine';
import { speechService } from '../../../src/services/speechService';

let raf: Map<number, FrameRequestCallback>;
let next: number;
const ctx={clearRect:vi.fn(),drawImage:vi.fn(),save:vi.fn(),restore:vi.fn(),beginPath:vi.fn(),arc:vi.fn(),roundRect:vi.fn(),stroke:vi.fn(),fill:vi.fn(),fillText:vi.fn(),moveTo:vi.fn(),lineTo:vi.fn(),closePath:vi.fn(),measureText:vi.fn(()=>({width:20}))};
beforeEach(()=>{
  vi.useFakeTimers({toFake:['setTimeout','clearTimeout','setInterval','clearInterval','performance']});
  raf=new Map();next=0;
  vi.spyOn(globalThis,'requestAnimationFrame').mockImplementation(cb=>{raf.set(++next,cb);return next;});
  vi.spyOn(globalThis,'cancelAnimationFrame').mockImplementation(id=>{raf.delete(id);});
  vi.spyOn(HTMLCanvasElement.prototype,'getContext').mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
  Object.defineProperty(HTMLVideoElement.prototype,'getVideoPlaybackQuality',{configurable:true,value:()=>({totalVideoFrames:1})});
  vi.spyOn(HTMLMediaElement.prototype,'play').mockResolvedValue();vi.spyOn(HTMLMediaElement.prototype,'pause').mockImplementation(()=>{});
  Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia:vi.fn(async()=>({getTracks:()=>[Object.assign(new EventTarget(),{stop:vi.fn()})]}))}});
});
afterEach(()=>{cleanup();vi.useRealTimers();vi.restoreAllMocks();vi.unstubAllGlobals();Reflect.deleteProperty(navigator,'mediaDevices');Reflect.deleteProperty(HTMLVideoElement.prototype,'getVideoPlaybackQuality');for(const f of Object.values(ctx))if(vi.isMockFunction(f))f.mockClear();});
const draw=()=>act(()=>{for(const [id,callback]of [...raf]){raf.delete(id);callback(performance.now());}});

for(const kind of ['bbox','pixel-mask','polygon'] as const){
  it(`independent normalized ${kind} selection, drawing, immediate intrinsic change and silent invalid input`,async()=>{
    const detector={detectForVideo:vi.fn(()=>({detections:[{categories:[{categoryName:'dog',score:0.9}],boundingBox:{originX:64,originY:48,width:128,height:96}}]})),close:vi.fn()};
    const factory={load:vi.fn(async()=>({})),detector:vi.fn(async()=>detector),segmenter:vi.fn()} as unknown as VisionTaskFactory;
    const engine=new VisionEngine(factory);await engine.initialize('mediapipe');
    const speak=vi.spyOn(speechService,'speak').mockReturnValue(true);const find=vi.spyOn(engine,'findObjectAtPoint');
    const onFrame=vi.fn(),onSelectObject=vi.fn();
    const view=render(<VisionCanvas engine={engine} activeModel="mediapipe" modelState={engine.getState()} facingMode="user" onToggleFacingMode={()=>{}} renderOptions={{showBoxes:true,showMasks:true,showLabels:true,showConfidence:true,confidenceThreshold:0.35,maskOpacity:0.35}} ttsConfig={{enabled:false,autoReadout:false,cooldownSec:3,rate:1,pitch:1,volume:1,voiceURI:''}} isCameraActive onCameraStart={()=>{}} onCameraStop={()=>{}} onSelectObject={onSelectObject} onFrame={onFrame}/>);
    await act(async()=>{});
    const video=view.container.querySelector('video')!,canvas=view.container.querySelector('canvas')!;
    Object.defineProperties(video,{readyState:{configurable:true,value:2},videoWidth:{configurable:true,value:640},videoHeight:{configurable:true,value:480}});
    const rect=vi.spyOn(video,'getBoundingClientRect').mockReturnValue({left:50,top:-100,width:960,height:540} as DOMRect);
    vi.spyOn(canvas,'getBoundingClientRect').mockReturnValue({left:50,top:-100,width:960,height:540} as DOMRect);
    draw();act(()=>{vi.advanceTimersByTime(0);vi.advanceTimersByTime(250);});draw();
    const object=onFrame.mock.calls.at(-1)![0].objects[0];
    const alpha=vi.fn(()=>({data:new Uint8ClampedArray([0,0,0,255])}));
    if(kind!=='bbox')object.bbox={x:0.8,y:0.8,width:0.1,height:0.1};
    if(kind==='pixel-mask'){
      const mask=document.createElement('canvas');mask.width=320;mask.height=240;
      Object.defineProperty(mask,'getContext',{value:()=>({getImageData:alpha})});object.maskCanvas=mask;
    }
    if(kind==='polygon')object.maskPolygon=[{x:0.2,y:0.2},{x:0.3,y:0.2},{x:0.3,y:0.3},{x:0.2,y:0.3}];
    const click=(x:number,y:number)=>act(()=>{fireEvent.click(canvas,{clientX:x,clientY:y});});
    // 4:3 image in 960x540: left=170; source 160,120 -> client350,35.
    click(169,35);click(891,35);expect(find).not.toHaveBeenCalled();expect(speak).not.toHaveBeenCalled();
    click(350,35);draw();expect(find).toHaveBeenLastCalledWith(0.25,0.25,expect.any(Array));expect(onSelectObject).toHaveBeenCalledWith(object);
    expect(ctx.arc).toHaveBeenLastCalledWith(160,120,24,0,2*Math.PI);
    if(kind==='pixel-mask'){expect(alpha).toHaveBeenCalledWith(80,60,1,1);expect(ctx.drawImage).toHaveBeenCalledWith(object.maskCanvas,0,0,640,480);}
    if(kind==='polygon')expect(ctx.moveTo).toHaveBeenCalledWith(128,96);
    const oldCallbacks=[...raf.keys()];
    Object.defineProperties(video,{videoWidth:{configurable:true,value:1280},videoHeight:{configurable:true,value:720}});
    // Native size changes before draw: equal 16:9, client290,35 -> .25,.25.
    expect(canvas.width).toBe(640);expect(canvas.height).toBe(480);click(290,35);
    expect(find).toHaveBeenLastCalledWith(0.25,0.25,expect.any(Array));expect([...raf.keys()]).toEqual(oldCallbacks);draw();
    expect([canvas.width,canvas.height]).toEqual([1280,720]);expect(ctx.arc).toHaveBeenLastCalledWith(320,180,24,0,2*Math.PI);
    rect.mockReturnValue({left:-20,top:40,width:800,height:600} as DOMRect);click(180,227.5);
    // jsdom MouseEvent preserves the fractional CSS input here.
    expect(find).toHaveBeenLastCalledWith(0.25,0.25,expect.any(Array));
    const calls=find.mock.calls.length;
    for(const [key,bad]of [['readyState',0],['videoWidth',0],['videoHeight',NaN]] as const){
      Object.defineProperty(video,key,{configurable:true,value:bad});click(180,227.5);
      Object.defineProperty(video,key,{configurable:true,value:key==='readyState'?2:key==='videoWidth'?1280:720});
    }
    expect(find).toHaveBeenCalledTimes(calls);expect(speak).toHaveBeenCalledTimes(3);
    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1);expect(factory.detector).toHaveBeenCalledTimes(1);expect(detector.close).not.toHaveBeenCalled();
  });
}
