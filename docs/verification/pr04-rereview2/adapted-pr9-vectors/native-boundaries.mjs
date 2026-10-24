import fs from 'node:fs';
import ts from 'typescript';
import { chromium } from '@playwright/test';
const source=fs.readFileSync('src/services/frameScheduler.ts','utf8');
const code=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true});
const results=[];
try {
  for (const fallback of [false,true]) for (const compatibility of ['immutable','live-quality','live-time']) {
    const page=await browser.newPage();await page.setContent('<video muted playsinline></video>');
    await page.addScriptTag({content:'window.scheduler={};(function(exports){'+code+'})(window.scheduler);'});
    results.push(await page.evaluate(async ({fallback,compatibility})=>{
      const NativeFrame=window.VideoFrame;
      const video=document.querySelector('video'), canvas=document.createElement('canvas');canvas.width=64;canvas.height=64;
      const ctx=canvas.getContext('2d');ctx.fillStyle='red';ctx.fillRect(0,0,64,64);
      const stream=canvas.captureStream(0),track=stream.getVideoTracks()[0];
      const rvfc=video.requestVideoFrameCallback.bind(video);
      const first=new Promise(resolve=>rvfc((_now,meta)=>resolve({time:meta.mediaTime,count:meta.presentedFrames})));
      video.srcObject=stream;await video.play();track.requestFrame();const initial=await first;
      await new Promise(resolve=>setTimeout(resolve,50));
      if(fallback)video.requestVideoFrameCallback=undefined;
      if(compatibility!=='immutable')window.VideoFrame=undefined;
      if(compatibility==='live-time')video.getVideoPlaybackQuality=undefined;
      const observations=[], errors=[];
      const stop=window.scheduler.startFrameScheduler({
        video,lock:new window.scheduler.JobLock(),current:()=>true,input(){},error(e){errors.push(String(e));},adopt(){},
        work(input) {
          const image=input instanceof NativeFrame?input:new NativeFrame(input,input instanceof HTMLCanvasElement?{timestamp:0}:undefined);
          const out=document.createElement('canvas');out.width=64;out.height=64;const c=out.getContext('2d');c.drawImage(image,0,0);
          observations.push({timestamp:image.timestamp,red:c.getImageData(0,0,1,1).data[0],time:video.currentTime,quality:video.getVideoPlaybackQuality?.().totalVideoFrames,kind:input instanceof NativeFrame?'immutable':'live'});
          if(image!==input)image.close();
        }
      });
      await new Promise(resolve=>setTimeout(resolve,350));const before=observations.length;
      ctx.fillStyle='blue';ctx.fillRect(0,0,64,64);
      const next=new Promise(resolve=>rvfc((_now,meta)=>resolve({time:meta.mediaTime,count:meta.presentedFrames})));
      track.requestFrame();const second=await next;
      await new Promise(resolve=>setTimeout(resolve,350));stop();stream.getTracks().forEach(t=>t.stop());window.VideoFrame=NativeFrame;
      return {fallback,compatibility,initial,second,before,total:observations.length,observations:observations.slice(0,30),errors};
    },{fallback,compatibility}));
    await page.close();
  }
} finally {await browser.close();}
console.log(JSON.stringify(results,null,2));
if(results.some(r=>r.before!==1||r.total!==2||r.errors.length))process.exitCode=1;
