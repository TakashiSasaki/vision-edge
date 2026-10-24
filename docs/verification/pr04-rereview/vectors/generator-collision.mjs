import fs from 'node:fs';
import ts from 'typescript';
import { chromium } from '@playwright/test';
const code=ts.transpileModule(fs.readFileSync('src/services/frameScheduler.ts','utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true});const results=[];
try{
 for(const fallback of [false,true]){
  const page=await browser.newPage();await page.route('http://127.0.0.1:4199/**',route=>route.fulfill({contentType:'text/html',body:'<video muted playsinline></video>'}));await page.goto('http://127.0.0.1:4199/');
  await page.addScriptTag({content:'window.scheduler={};(function(exports){'+code+'})(window.scheduler);'});
  results.push(await page.evaluate(async fallback=>{
   if(typeof MediaStreamTrackGenerator!=='function')return{fallback,unsupported:true};
   const generator=new MediaStreamTrackGenerator({kind:'video'}),writer=generator.writable.getWriter(),video=document.querySelector('video');
   video.srcObject=new MediaStream([generator]);const canvas=document.createElement('canvas');canvas.width=32;canvas.height=32;const c=canvas.getContext('2d');
   const native=video.requestVideoFrameCallback.bind(video);const observations=[],presented=[];
   async function write(color,timestamp){
    c.fillStyle=color;c.fillRect(0,0,32,32);
    const arrival=new Promise(resolve=>{native((_now,meta)=>resolve({time:meta.mediaTime,count:meta.presentedFrames}));setTimeout(()=>resolve(null),250);});
    const frame=new VideoFrame(canvas,{timestamp});await writer.write(frame);frame.close();const event=await arrival;
    const image=new VideoFrame(video);const out=document.createElement('canvas');out.width=32;out.height=32;const ctx=out.getContext('2d');ctx.drawImage(image,0,0);
    presented.push({suppliedTimestamp:timestamp,event,actualTimestamp:image.timestamp,pixel:Array.from(ctx.getImageData(0,0,1,1).data)});image.close();
   }
   const playing=video.play();await write('red',1000);await playing;
   if(fallback)video.requestVideoFrameCallback=undefined;
   const stop=window.scheduler.startFrameScheduler({video,current:()=>true,lock:new window.scheduler.JobLock(),input(){},error(){},adopt(){},work(frame){observations.push({timestamp:frame.timestamp});}});
   await new Promise(resolve=>setTimeout(resolve,100));await write('blue',1000);await new Promise(resolve=>setTimeout(resolve,100));await write('green',2000);await new Promise(resolve=>setTimeout(resolve,100));
   stop();await writer.close();generator.stop();return{fallback,presented,observations};
  },fallback));await page.close();
 }
}finally{await browser.close();}
console.log(JSON.stringify(results,null,2));
if(results.some(r=>!r.unsupported&&r.presented[1].event&&r.observations.length!==3))process.exitCode=1;
