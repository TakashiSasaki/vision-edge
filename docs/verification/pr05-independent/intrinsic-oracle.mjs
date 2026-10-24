import { chromium, expect } from '@playwright/test';
import { writeFileSync } from 'node:fs';
const browser=await chromium.launch({headless:true,executablePath:'/usr/bin/chromium'});
const context=await browser.newContext({viewport:{width:1400,height:1000},deviceScaleFactor:2});
const page=await context.newPage();
await context.route('**/*',r=>new URL(r.request().url()).origin==='http://127.0.0.1:4185'?r.continue():r.abort());
await context.addInitScript(()=>{
  const p=window.intrinsicProbe={requests:0,speech:[],arcs:[],hold:false,held:[],source:null,stream:null};
  const raf=requestAnimationFrame.bind(window);
  window.requestAnimationFrame=callback=>raf(time=>{if(p.hold)p.held.push(callback);else callback(time);});
  p.release=()=>{p.hold=false;p.held.splice(0).forEach(callback=>raf(callback));};
  const arc=CanvasRenderingContext2D.prototype.arc;
  CanvasRenderingContext2D.prototype.arc=function(...args){p.arcs.push(args.slice(0,3));return arc.apply(this,args);};
  speechSynthesis.speak=u=>p.speech.push(u.text);
  Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia:async()=>{
    p.requests++;const c=p.source=document.createElement('canvas');c.width=640;c.height=480;
    const ctx=c.getContext('2d');ctx.fillStyle='rgb(12,34,56)';ctx.fillRect(0,0,640,480);ctx.fillStyle='rgb(240,180,40)';ctx.fillRect(64,48,128,96);
    return p.stream=c.captureStream(0);
  }}});
});
try{
  await page.goto('http://127.0.0.1:4185');
  await page.getByRole('banner').getByRole('combobox',{name:'モデル',exact:true}).selectOption('mediapipe');
  await expect(page.getByText('(dog)')).toBeVisible();
  await page.evaluate(()=>Object.assign(document.querySelector('video').parentElement.style,{position:'fixed',left:'37.25px',top:'82.5px',width:'960px',height:'540px',maxHeight:'none',aspectRatio:'auto',border:'0px'}));
  const result=await page.evaluate(()=>new Promise((resolve,reject)=>{
    const p=window.intrinsicProbe,v=document.querySelector('video'),canvas=document.querySelector('canvas');
    const creates=window.visionSdkProbe.creates;p.hold=true;
    const deadline=setTimeout(()=>reject(new Error('Native source resize not observed')),5000);
    v.addEventListener('resize',()=>{
      clearTimeout(deadline);
      const before={video:[v.videoWidth,v.videoHeight],backing:[canvas.width,canvas.height]};
      canvas.dispatchEvent(new MouseEvent('click',{bubbles:true,clientX:277,clientY:217}));
      const after={speech:p.speech,creates:window.visionSdkProbe.creates,requests:p.requests};
      p.release();resolve({before,after,creates,expectedNormalized:[239.75/960,134.5/540],expectedArc:[239.75/960*1280,134.5/540*720],client:[277,217]});
    },{once:true});
    p.source.width=1280;p.source.height=720;
    const ctx=p.source.getContext('2d');ctx.fillStyle='rgb(12,34,56)';ctx.fillRect(0,0,1280,720);ctx.fillStyle='rgb(240,180,40)';ctx.fillRect(128,72,256,144);
    p.stream.getVideoTracks()[0].requestFrame();
  }));
  await expect.poll(()=>page.evaluate(()=>window.intrinsicProbe.arcs.length)).toBeGreaterThan(0);
  result.draw=await page.evaluate(()=>({backing:[document.querySelector('canvas').width,document.querySelector('canvas').height],arc:window.intrinsicProbe.arcs.at(-1),selected:document.querySelector('.group.border-blue-500\\/80')!==null}));
  writeFileSync(new URL('./intrinsic-results.json',import.meta.url),JSON.stringify(result,null,2)+'\n');
  expect(result.before.video).toEqual([1280,720]);expect(result.before.backing).toEqual([640,480]);
  expect(result.after.speech).toHaveLength(1);expect(result.after.requests).toBe(1);expect(result.after.creates).toBe(result.creates);
  expect(result.draw.backing).toEqual([1280,720]);expect(result.draw.selected).toBe(true);
  expect(result.draw.arc[0]).toBeCloseTo(result.expectedArc[0],10);expect(result.draw.arc[1]).toBeCloseTo(result.expectedArc[1],10);
  await page.screenshot({path:new URL('./after/native-intrinsic-change.png',import.meta.url).pathname,scale:'css'});
  console.log('PASS native intrinsic resize before click and next draw',JSON.stringify(result));
}finally{await context.close();await browser.close();}
