import {act,cleanup,fireEvent,render} from '@testing-library/react';
import {afterEach,beforeEach,expect,it,vi} from 'vitest';
import {VisionCanvas} from '../../../src/components/VisionCanvas';
import {PipelineMetrics,type PipelineSnapshot} from '../../../src/services/pipelineMetrics';
import {VisionEngine,type VisionTaskFactory} from '../../../src/services/visionEngine';
let raf:Map<number,FrameRequestCallback>;let seq:number;let frames:{closed:number;timestamp:number}[];let tracks:{stop:ReturnType<typeof vi.fn>}[];
beforeEach(()=>{
 vi.useFakeTimers({toFake:['setTimeout','clearTimeout','setInterval','clearInterval','performance']});raf=new Map();seq=0;frames=[];tracks=[];
 vi.stubGlobal('requestAnimationFrame',(fn:FrameRequestCallback)=>{raf.set(++seq,fn);return seq;});vi.stubGlobal('cancelAnimationFrame',(id:number)=>raf.delete(id));
 vi.stubGlobal('VideoFrame',class{timestamp:number;displayWidth=2;displayHeight=2;closed=0;constructor(video:HTMLVideoElement){this.timestamp=video.currentTime*1e6;frames.push(this);}close(){this.closed++;}});
 vi.spyOn(HTMLMediaElement.prototype,'play').mockResolvedValue();vi.spyOn(HTMLMediaElement.prototype,'pause').mockImplementation(()=>{});vi.spyOn(HTMLMediaElement.prototype,'readyState','get').mockReturnValue(2);vi.spyOn(HTMLVideoElement.prototype,'videoWidth','get').mockReturnValue(2);vi.spyOn(HTMLVideoElement.prototype,'videoHeight','get').mockReturnValue(2);
 vi.spyOn(HTMLCanvasElement.prototype,'getContext').mockReturnValue({clearRect(){},drawImage(){}} as unknown as CanvasRenderingContext2D);
 Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia:vi.fn(async()=>{const track=Object.assign(new EventTarget(),{stop:vi.fn()});tracks.push(track);return {getTracks:()=>[track]};})}});
});
afterEach(()=>{cleanup();vi.useRealTimers();vi.unstubAllGlobals();Reflect.deleteProperty(navigator,'mediaDevices');});
async function mount(){
 const taskClose=vi.fn(),sdk=vi.fn(()=>({detections:[]}));const factory:VisionTaskFactory={load:vi.fn(async()=>({wasmLoaderPath:'',wasmBinaryPath:''})),detector:vi.fn(async()=>({detectForVideo:sdk,close:taskClose})),segmenter:async()=>{throw Error('unused');}};const engine=new VisionEngine(factory);await engine.initialize('mediapipe');
 const props={engine,activeModel:'mediapipe' as const,modelState:engine.getState(),facingMode:'user' as const,onToggleFacingMode:vi.fn(),renderOptions:{showBoxes:false,showMasks:false,showLabels:false,showConfidence:false,maskOpacity:0,confidenceThreshold:.5},ttsConfig:{enabled:false,autoReadout:false,cooldownSec:0,rate:1,pitch:1,volume:1,voiceURI:''},isCameraActive:true,onCameraStart:vi.fn(),onCameraStop:vi.fn(),onSelectObject:vi.fn(),onFrame:vi.fn()};
 let view!:ReturnType<typeof render>;await act(async()=>{view=render(<VisionCanvas {...props}/>);});act(()=>vi.advanceTimersByTime(0));const video=view.container.querySelector('video')!;
 const frame=()=>act(()=>{video.currentTime+=1/30;for(const [id,fn] of [...raf]){raf.delete(id);fn(performance.now());}vi.advanceTimersByTime(0);});
 const publish=()=>act(()=>vi.advanceTimersByTime(250));const sdkMean=()=>[...view.container.querySelectorAll('table[aria-label="success interval statistics"] tbody tr')].find(x=>x.querySelector('th')?.textContent==='sdk')!.querySelectorAll('td')[1].textContent;
 return {view,video,frame,publish,sdkMean,props,sdk,engine,factory,taskClose};
}
const cases:Array<[string,(s:PipelineSnapshot)=>unknown]>=[
 ['null root',()=>null],['null stages',s=>({...s,stages:null})],['array stage',s=>({...s,stages:{...s.stages,sdk:[]}})],['missing row',s=>({...s,stages:{...s.stages,sdk:{...s.stages.sdk,success:undefined}}})],
 ['toxic duration',s=>{Object.assign(s.stages.sdk.success,{mean:{toFixed(){throw Error('must not coerce');}}});return s;}],['invalid counters',s=>({...s,results:{hidden:1n},unclocked:{},pendingResults:'0'})],
 ['throwing stage getter',s=>Object.defineProperty(s,'stages',{get(){throw Error('statistics getter');}})],['throwing result descriptor',s=>({...s,results:new Proxy({},{getOwnPropertyDescriptor(){throw Error('descriptor');}})})],
 ['revoked root',()=>{const p=Proxy.revocable({},{});p.revoke();return p.proxy;}],
];
it.each(cases)('%s: actual camera/SDK continue, valid zero recovers, lifecycle still releases exact owners',async(_label,mutate)=>{
 const original=PipelineMetrics.prototype.snapshot;let broken=true;vi.spyOn(PipelineMetrics.prototype,'snapshot').mockImplementation(function(this:PipelineMetrics){const s=original.call(this)!;return (broken?mutate(s):s) as PipelineSnapshot;});
 const f=await mount();expect(f.sdk).toHaveBeenCalledTimes(1);const request=f.engine.getState().requestId;
 for(let i=0;i<4;i++){expect(()=>f.publish()).not.toThrow();f.frame();expect(f.view.container.querySelector('video')).toBe(f.video);expect(raf.size).toBe(2);}
 expect(f.sdk).toHaveBeenCalledTimes(5);expect(f.props.onFrame.mock.calls.filter(c=>c[0]!==null)).toHaveLength(4);expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1);expect(f.factory.detector).toHaveBeenCalledTimes(1);expect(f.taskClose).not.toHaveBeenCalled();expect(tracks[0].stop).not.toHaveBeenCalled();expect(frames.every(x=>x.closed===1)).toBe(true);expect(f.engine.getState()).toMatchObject({status:'ready',requestId:request});
 broken=false;f.publish();expect(f.sdkMean()).toBe('0.00');f.frame();expect(f.sdk).toHaveBeenCalledTimes(6);expect(f.view.container.textContent).not.toContain('[object Object]');
 await act(async()=>f.view.rerender(<VisionCanvas {...f.props} isCameraActive={false}/>));expect(raf.size).toBe(0);expect(tracks[0].stop).toHaveBeenCalledTimes(1);
 await act(async()=>f.view.rerender(<VisionCanvas {...f.props}/>));act(()=>vi.advanceTimersByTime(0));expect(f.sdk).toHaveBeenCalledTimes(7);expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(2);expect(f.factory.detector).toHaveBeenCalledTimes(1);f.view.unmount();expect(tracks.every(t=>t.stop.mock.calls.length===1)).toBe(true);expect(frames.every(x=>x.closed===1)).toBe(true);f.engine.dispose();expect(f.taskClose).toHaveBeenCalledTimes(1);
});
it('same mutable raw object publishes owned values and later invalid/valid states without memo hiding recovery',async()=>{
 const original=PipelineMetrics.prototype.snapshot;let raw:PipelineSnapshot|undefined;let broken=true;vi.spyOn(PipelineMetrics.prototype,'snapshot').mockImplementation(function(this:PipelineMetrics){const s=original.call(this)!;raw??=s;Object.assign(raw,s,{stages:broken?null:s.stages});return raw;});
 const f=await mount();f.publish();expect(f.sdkMean()).toBe('未計測');broken=false;f.publish();expect(f.sdkMean()).toBe('0.00');Object.assign(raw!,{stages:null,results:{hidden:{toString(){throw Error('coercion');}}}});fireEvent.click(f.view.getByText('計測詳細 (JS wall time)'));f.view.rerender(<VisionCanvas {...f.props} ttsConfig={{...f.props.ttsConfig,rate:1.1}}/>);expect(f.sdkMean()).toBe('0.00');
 broken=true;f.publish();expect(f.sdkMean()).toBe('未計測');broken=false;f.publish();expect(f.sdkMean()).toBe('0.00');f.frame();expect(f.sdk).toHaveBeenCalledTimes(2);expect(f.view.container.querySelector('video')).toBe(f.video);f.view.unmount();f.engine.dispose();
});
