import { expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ControlPanel } from './ControlPanel';
import { speechService } from '../services/speechService';
import { drawCanvasFrame } from '../services/canvasRenderer';
import type { DetectedObject } from '../types/vision';
vi.mock('../services/speechService',()=>({speechService:{getVoices:()=>[],speak:vi.fn()}}));
it('F4 presents semantic score separately from area and speaks the score contract',()=>{
 const object:DetectedObject={id:'segformer-0',label:'wall',labelJa:'壁',category:'建築',confidence:.42,confidenceKind:'semantic-softmax-mean',areaFraction:.8,bbox:{x:0,y:0,width:1,height:1},color:'#123456'};
 render(<ControlPanel activeModel="segformer" renderOptions={{showBoxes:true,showMasks:true,showLabels:true,showConfidence:true,maskOpacity:.35,confidenceThreshold:.35}} onChangeRenderOptions={()=>{}} ttsConfig={{enabled:false,autoReadout:false,cooldownSec:2,rate:1,pitch:1,volume:1,voiceURI:''}} onChangeTTSConfig={()=>{}} detectedObjects={[object]} hasCurrentFrame onSelectObject={()=>{}} isSpeaking={false}/>);
 expect(screen.getByText(/softmaxスコアは正答率の保証ではありません/)).toBeInTheDocument();expect(screen.getByText('平均softmaxスコア 42%')).toBeInTheDocument();expect(screen.getByText('面積 80%')).toBeInTheDocument();fireEvent.click(screen.getByText('壁'));expect(speechService.speak).toHaveBeenCalledWith('壁。平均softmaxスコアは42%です。','壁',expect.anything(),'manual');
});
it('F4 forwards a full semantic mask for a ready empty-object frame and copies it to display',()=>{
 const canvas=document.createElement('canvas'),mask=document.createElement('canvas');let context=0;const display={clearRect:vi.fn(),drawImage:vi.fn()},buffer={clearRect:vi.fn(),drawImage:vi.fn()};
 vi.spyOn(HTMLCanvasElement.prototype,'getContext').mockImplementation(()=> (++context===1?display:buffer) as unknown as CanvasRenderingContext2D);
 const overlays=vi.fn();const copied=drawCanvasFrame(canvas,{videoWidth:640,videoHeight:480,readyState:2},{model:'segformer',requestId:1,objects:[],semanticMask:mask},{current:null},overlays,()=>true);
 expect(copied).toBe(true);expect(overlays).toHaveBeenCalledWith(buffer,640,480,[],mask);expect(display.drawImage).toHaveBeenCalledTimes(1);
});
