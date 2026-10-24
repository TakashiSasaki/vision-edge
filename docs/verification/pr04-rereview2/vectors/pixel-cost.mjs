import fs from 'node:fs';
import ts from 'typescript';
import {chromium} from '@playwright/test';
const code=ts.transpileModule(fs.readFileSync('src/services/frameScheduler.ts','utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true});const results=[];
try{
 for(const [width,height] of [[64,64],[640,480],[1280,720],[1920,1080]]){
  const page=await browser.newPage();await page.setContent('<video muted playsinline></video>');await page.addScriptTag({content:'window.scheduler={};(function(exports){'+code+'})(window.scheduler);'});
  results.push(await page.evaluate(async({width,height})=>{
   const source=document.createElement('canvas');source.width=width;source.height=height;const context=source.getContext('2d');context.fillStyle='red';context.fillRect(0,0,width,height);
   const video=document.querySelector('video'),stream=source.captureStream(0),native=video.requestVideoFrameCallback.bind(video);
   const first=new Promise(resolve=>native(()=>resolve()));video.srcObject=stream;await video.play();stream.getVideoTracks()[0].requestFrame();await first;
   const create=document.createElement.bind(document),canvases=[];document.createElement=(name,...args)=>{const element=create(name,...args);if(name==='canvas')canvases.push(element);return element;};
   const read=CanvasRenderingContext2D.prototype.getImageData;let readCalls=0,readBytes=0,readMs=0;
   CanvasRenderingContext2D.prototype.getImageData=function(...args){const start=performance.now();const result=read.call(this,...args);readMs+=performance.now()-start;readCalls++;readBytes+=result.data.byteLength;return result;};
   window.VideoFrame=undefined;video.getVideoPlaybackQuality=undefined;video.requestVideoFrameCallback=undefined;
   const callbacks=new Map();let id=0;window.requestAnimationFrame=fn=>{callbacks.set(++id,fn);return id;};window.cancelAnimationFrame=key=>callbacks.delete(key);
   let work=0,input=0;const stop=window.scheduler.startFrameScheduler({video,current:()=>true,lock:new window.scheduler.JobLock(),input(){input++;},error(e){throw e;},adopt(){},work(image){if(!(image instanceof HTMLCanvasElement))throw Error('Expected owned canvas');work++;}});
   await new Promise(resolve=>setTimeout(resolve,0));
   const iterations=20,start=performance.now();
   for(let i=0;i<iterations;i++){const [key,callback]=callbacks.entries().next().value;callbacks.delete(key);callback(performance.now());}
   const elapsedMs=performance.now()-start,liveBeforeStop=canvases.filter(c=>c.width>0&&c.height>0).length;
   stop();stream.getTracks().forEach(t=>t.stop());
   return{width,height,iterations,work,input,canvases:canvases.length,liveBeforeStop,allZeroAfterStop:canvases.every(c=>c.width===0&&c.height===0),readCalls,readBytes,readMs,elapsedMs,meanPollMs:elapsedMs/iterations};
  },{width,height}));await page.close();
 }
}finally{await browser.close();}
console.log(JSON.stringify({note:'Local synthetic readback/byte-comparison costs only; no model, hardware camera, GPU or universal latency guarantee.',results},null,2));
if(results.some(r=>r.work!==1||r.input!==1||r.canvases!==2||!r.allZeroAfterStop||r.readCalls!==22))process.exitCode=1;
