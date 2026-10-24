import {test,expect} from '@playwright/test';
import {createHash} from 'node:crypto';
for(const fallback of [false,true])for(const quality of ['missing','zero']){
 test('independent pixels, dimensions, aging and capture recovery '+JSON.stringify({fallback,quality}),async({context,page})=>{
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await context.route('**/*',route=>new URL(route.request().url()).origin==='http://127.0.0.1:4186'?route.continue():route.abort());
  await context.addInitScript(({fallback,quality})=>{
   const rvfc=HTMLVideoElement.prototype.requestVideoFrameCallback,play=HTMLMediaElement.prototype.play;
   window.review2ModelGate=new Promise(resolve=>{window.releaseReview2=resolve;});
   const probe=window.review2Input={requests:0,first:[],presentations:[],samplers:[],failCapture:false,source:null,stream:null};
   const draw=CanvasRenderingContext2D.prototype.drawImage,read=CanvasRenderingContext2D.prototype.getImageData;
   CanvasRenderingContext2D.prototype.drawImage=function(input,...args){
    if(input instanceof HTMLVideoElement&&this.canvas.dataset.reviewObserver!=='true'&&!probe.samplers.includes(this.canvas))probe.samplers.push(this.canvas);
    return draw.call(this,input,...args);
   };
   CanvasRenderingContext2D.prototype.getImageData=function(...args){
    if(probe.failCapture&&probe.samplers.includes(this.canvas)){probe.failCapture=false;throw Error('Independent readback fault');}
    return read.call(this,...args);
   };
   Object.defineProperty(window,'VideoFrame',{configurable:true,value:undefined});
   Object.defineProperty(HTMLVideoElement.prototype,'getVideoPlaybackQuality',{configurable:true,value:quality==='missing'?undefined:()=>({totalVideoFrames:0})});
   if(fallback)Object.defineProperty(HTMLVideoElement.prototype,'requestVideoFrameCallback',{configurable:true,value:undefined});
   HTMLMediaElement.prototype.play=function(){
    if(this.srcObject instanceof MediaStream)rvfc.call(this,(_now,m)=>{probe.first.push({time:m.mediaTime,count:m.presentedFrames});});
    return play.call(this);
   };
   Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia:async()=>{
    probe.requests++;const source=document.createElement('canvas');source.width=16;source.height=8;probe.source=source;
    const paint=(phase,width=source.width,height=source.height)=>{
     source.width=width;source.height=height;const ctx=source.getContext('2d'),image=ctx.createImageData(width,height);
     for(let i=0;i<image.data.length;i+=4){image.data[i]=255;image.data[i+3]=255;}
     const a=phase===1?(width*height-1)*4:0,b=phase===1?0:(width*height-1)*4;
     image.data[a]=0;image.data[a+2]=255;image.data[b]=0;image.data[b+1]=255;
     if(phase===2)image.data[image.data.length-3]=124;
     ctx.putImageData(image,0,0);
    };
    paint(0);const stream=source.captureStream(0);probe.stream=stream;
    probe.send=async(phase,width,height)=>{
     if(phase!==null)paint(phase,width,height);
     else{const ctx=source.getContext('2d');ctx.putImageData(ctx.getImageData(0,0,source.width,source.height),0,0);}
     const video=document.querySelector('video');
     const next=new Promise((resolve,reject)=>{const timeout=setTimeout(()=>reject(Error('Input fixture did not present requested frame')),2000);rvfc.call(video,(_now,m)=>{clearTimeout(timeout);const event={time:m.mediaTime,count:m.presentedFrames,width:video.videoWidth,height:video.videoHeight};probe.presentations.push(event);resolve(event);});});
     stream.getVideoTracks()[0].requestFrame();return next;
    };
    setTimeout(()=>stream.getVideoTracks()[0].requestFrame(),30);return stream;
   }}});
   probe.reference=()=>{
    const video=document.querySelector('video'),canvas=document.createElement('canvas');canvas.dataset.reviewObserver='true';
    canvas.width=video.videoWidth;canvas.height=video.videoHeight;const ctx=canvas.getContext('2d');ctx.drawImage(video,0,0);
    return{width:canvas.width,height:canvas.height,pixels:Array.from(ctx.getImageData(0,0,canvas.width,canvas.height).data)};
   };
  },{fallback,quality});
  await page.goto('/');await expect.poll(()=>page.evaluate(()=>window.review2Input.first.length)).toBe(1);
  await page.evaluate(()=>window.releaseReview2());await expect.poll(()=>page.evaluate(()=>window.review2Sdk.calls)).toBe(1);
  await page.evaluate(()=>window.review2Input.send(null));await page.waitForTimeout(150);
  expect(await page.evaluate(()=>window.review2Sdk.calls)).toBe(1);
  await page.waitForTimeout(1650);
  await page.evaluate(()=>window.review2Input.send(null)); // Fresh identical presentation shortly before expiry must not renew content age.
  await page.waitForTimeout(450);
  await expect(page.getByLabel('推論FPS')).toHaveText('推論: 未計測');
  await expect(page.getByLabel('フレーム処理時間')).toContainText('未計測');
  await expect(page.getByLabel('性能状態')).toHaveText('映像入力停滞');
  await expect(page.getByLabel('描画FPS')).toContainText(/\d+\.\d+ FPS/);
  const stages=[];
  async function verifyStage(expectedCalls){
   await expect.poll(()=>page.evaluate(()=>window.review2Sdk.calls)).toBe(expectedCalls);await page.waitForTimeout(100);
   const data=await page.evaluate(()=>({image:window.review2Sdk.images.at(-1),reference:window.review2Input.reference(),samplers:window.review2Input.samplers.length,ownedReleased:window.review2Sdk.owned.every(c=>c.width===0&&c.height===0)}));
   expect(data.image.width).toBe(data.reference.width);expect(data.image.height).toBe(data.reference.height);expect(data.image.pixels).toEqual(data.reference.pixels);
   expect(data.image.sourceTimestamp).toBeNull();expect(data.image.kind).toBe('canvas');expect(data.samplers).toBe(1);expect(data.ownedReleased).toBe(true);
   stages.push({calls:expectedCalls,width:data.image.width,height:data.image.height,pixelSha256:createHash('sha256').update(Buffer.from(data.image.pixels)).digest('hex')});
  }
  await verifyStage(1);
  await page.evaluate(()=>window.review2Input.send(1));await verifyStage(2); // Swapped positions, identical histogram.
  await expect(page.getByLabel('性能状態')).toHaveText('計測中');
  await page.evaluate(()=>window.review2Input.send(1,8,16));await verifyStage(3); // Same byte count, different dimensions.
  await page.evaluate(()=>{window.review2Sdk.failures=1;return window.review2Input.send(2);});
  await expect.poll(()=>page.evaluate(()=>window.review2Sdk.calls)).toBe(4);
  await expect(page.getByLabel('性能状態')).toContainText('推論エラー');
  await page.evaluate(()=>window.review2Input.send(null));await page.waitForTimeout(350);expect(await page.evaluate(()=>window.review2Sdk.calls)).toBe(4);
  await page.evaluate(()=>window.review2Input.send(0));await verifyStage(5);
  await page.evaluate(()=>{window.review2Input.failCapture=true;return window.review2Input.send(1);});
  await expect(page.getByRole('alert')).toContainText('映像フレーム取得エラー');
  const failed=await page.evaluate(()=>({calls:window.review2Sdk.calls,samplersReleased:window.review2Input.samplers.every(c=>c.width===0&&c.height===0)}));
  expect(failed.calls).toBe(5);expect(failed.samplersReleased).toBe(true);
  await page.getByRole('button',{name:'処理を再試行'}).click();await expect.poll(()=>page.evaluate(()=>window.review2Sdk.calls)).toBe(6);
  expect(await page.evaluate(()=>window.review2Input.requests)).toBe(1);expect(await page.evaluate(()=>window.review2Sdk.creates)).toBe(1);
  const summary=await page.evaluate(()=>({presentations:window.review2Input.presentations,first:window.review2Input.first,requests:window.review2Input.requests,calls:window.review2Sdk.calls,successes:window.review2Sdk.successes,samplers:window.review2Input.samplers.length}));
  await page.getByRole('banner').getByRole('button',{name:'停止',exact:true}).click();
  expect(await page.evaluate(()=>window.review2Input.samplers.every(c=>c.width===0&&c.height===0))).toBe(true);
  expect(await page.evaluate(()=>window.review2Input.stream.getTracks().every(t=>t.readyState==='ended'))).toBe(true);
  expect(errors).toEqual([]);console.log('INDEPENDENT PIXELS',JSON.stringify({fallback,quality,stages,...summary}));
 });
}
