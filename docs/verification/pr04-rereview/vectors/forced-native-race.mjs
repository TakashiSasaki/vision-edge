import fs from 'node:fs';
import ts from 'typescript';
import { chromium } from '@playwright/test';
const code=ts.transpileModule(fs.readFileSync('src/services/frameScheduler.ts','utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
const clip=fs.readFileSync('.cache/review-video.webm').toString('base64');
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true});
const results=[];
try {
  for(const fallback of [false,true])for(const compatibility of ['immutable','live-quality','live-time']){
    const page=await browser.newPage();await page.setContent('<video muted playsinline></video>');
    await page.addScriptTag({content:'window.scheduler={};(function(exports){'+code+'})(window.scheduler);'});
    results.push(await page.evaluate(async({fallback,compatibility,clip})=>{
      const NativeFrame=window.VideoFrame, video=document.querySelector('video');
      video.src='data:video/webm;base64,'+clip;await video.play();
      const rvfc=video.requestVideoFrameCallback.bind(video),raf=window.requestAnimationFrame;
      const out=document.createElement('canvas');out.width=64;out.height=64;const ctx=out.getContext('2d',{willReadFrequently:true});
      const observations=[],errors=[];let reservedTimestamp=null,blocked=false,previous=null;
      const contend=()=>{
        if(blocked)return;
        blocked=true;const image=new NativeFrame(video);reservedTimestamp=image.timestamp;image.close();
        const end=performance.now()+320;while(performance.now()<end){ /* Real decoding advances before deferred work. */ }
      };
      if(fallback){video.requestVideoFrameCallback=undefined;window.requestAnimationFrame=fn=>raf(now=>{fn(now);contend();});}
      else video.requestVideoFrameCallback=fn=>rvfc((now,meta)=>{fn(now,meta);contend();});
      if(compatibility!=='immutable')window.VideoFrame=undefined;
      if(compatibility==='live-time')video.getVideoPlaybackQuality=undefined;
      const stop=window.scheduler.startFrameScheduler({
        video,bootstrap:false,current:()=>true,lock:new window.scheduler.JobLock(),input(){},error(e){errors.push(String(e));},adopt(){},
        work(input){
          const image=input instanceof NativeFrame?input:new NativeFrame(input);ctx.drawImage(image,0,0);
          const bytes=ctx.getImageData(0,0,64,64).data;
          const samePixels=previous!==null&&bytes.every((v,i)=>v===previous[i]);previous=new Uint8ClampedArray(bytes);
          observations.push({timestamp:image.timestamp,samePixels,kind:input instanceof NativeFrame?'immutable':'live',time:video.currentTime,quality:video.getVideoPlaybackQuality?.().totalVideoFrames});
          if(image!==input)image.close();video.pause();
        }
      });
      await new Promise(resolve=>setTimeout(resolve,1000));stop();window.VideoFrame=NativeFrame;window.requestAnimationFrame=raf;
      return{fallback,compatibility,blocked,reservedTimestamp,observations,errors};
    },{fallback,compatibility,clip}));await page.close();
  }
}finally{await browser.close();}
console.log(JSON.stringify(results,null,2));
if(results.some(r=>r.errors.length||!r.blocked||!r.observations.length||r.observations.some((o,i)=>i>0&&o.timestamp===r.observations[i-1].timestamp)))process.exitCode=1;
