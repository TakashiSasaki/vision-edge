import { act, cleanup, render } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { VisionCanvas } from '../../../src/components/VisionCanvas';
import { PipelineMetrics, type PipelineSnapshot } from '../../../src/services/pipelineMetrics';
import { VisionEngine, type VisionTaskFactory } from '../../../src/services/visionEngine';
let callbacks:Map<number,FrameRequestCallback>;let next:number;
beforeEach(()=>{
 vi.useFakeTimers({toFake:['setTimeout','clearTimeout','setInterval','clearInterval','performance']});callbacks=new Map();next=0;
 vi.stubGlobal('requestAnimationFrame',(fn:FrameRequestCallback)=>{callbacks.set(++next,fn);return next;});vi.stubGlobal('cancelAnimationFrame',(id:number)=>callbacks.delete(id));
 vi.stubGlobal('VideoFrame',class{timestamp=0;displayWidth=2;displayHeight=2;close(){};});
 vi.spyOn(HTMLMediaElement.prototype,'play').mockResolvedValue();vi.spyOn(HTMLMediaElement.prototype,'pause').mockImplementation(()=>{});
 vi.spyOn(HTMLMediaElement.prototype,'readyState','get').mockReturnValue(2);vi.spyOn(HTMLVideoElement.prototype,'videoWidth','get').mockReturnValue(2);vi.spyOn(HTMLVideoElement.prototype,'videoHeight','get').mockReturnValue(2);
 vi.spyOn(HTMLCanvasElement.prototype,'getContext').mockReturnValue({clearRect(){},drawImage(){}} as unknown as CanvasRenderingContext2D);
 Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia:vi.fn(async()=>({getTracks:()=>[]}))}});
 vi.spyOn(console,'error').mockImplementation(()=>{});
});
afterEach(()=>{cleanup();vi.useRealTimers();vi.unstubAllGlobals();Reflect.deleteProperty(navigator,'mediaDevices');});
async function mount(){
 const sdk=vi.fn(()=>({detections:[]}));const factory:VisionTaskFactory={load:async()=>({wasmLoaderPath:'',wasmBinaryPath:''}),detector:async()=>({detectForVideo:sdk,close(){}}),segmenter:async()=>{throw Error('unused');}};const engine=new VisionEngine(factory);await engine.initialize('mediapipe');
 let view!:ReturnType<typeof render>;await act(async()=>{view=render(<VisionCanvas engine={engine} activeModel="mediapipe" modelState={engine.getState()} facingMode="user" onToggleFacingMode={()=>{}} renderOptions={{showBoxes:false,showMasks:false,showLabels:false,showConfidence:false,maskOpacity:0,confidenceThreshold:.5}} ttsConfig={{enabled:false,autoReadout:false,cooldownSec:0,rate:1,pitch:1,volume:1,voiceURI:''}} isCameraActive onCameraStart={()=>{}} onCameraStop={()=>{}} onSelectObject={()=>{}} onFrame={()=>{}}/>);});act(()=>{vi.advanceTimersByTime(0);});expect(sdk).toHaveBeenCalledTimes(1);return {view,sdk};
}
it('throwing statistics computation leaves real camera/engine component mounted',async()=>{
 vi.spyOn(PipelineMetrics.prototype,'snapshot').mockImplementation(()=>{throw Error('invalid stats computation');});const f=await mount();expect(()=>act(()=>vi.advanceTimersByTime(250))).not.toThrow();expect(f.view.container.querySelector('video')).not.toBeNull();
});
it('invalid statistics result must not unmount the camera/engine component',async()=>{
 const original=PipelineMetrics.prototype.snapshot;vi.spyOn(PipelineMetrics.prototype,'snapshot').mockImplementation(function(this:PipelineMetrics){return {...original.call(this),stages:null} as unknown as PipelineSnapshot;});const f=await mount();let thrown:unknown;try{act(()=>vi.advanceTimersByTime(250));}catch(e){thrown=e;}console.log('PR06-F01',JSON.stringify({error:thrown instanceof Error?thrown.message:String(thrown),videoMounted:!!f.view.container.querySelector('video'),pendingRAF:callbacks.size,cameraRequests:vi.mocked(navigator.mediaDevices.getUserMedia).mock.calls.length,sdkCalls:f.sdk.mock.calls.length}));expect(thrown).toBeUndefined();expect(f.view.container.querySelector('video')).not.toBeNull();
});
