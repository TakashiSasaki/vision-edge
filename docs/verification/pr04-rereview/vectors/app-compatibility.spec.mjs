import { test, expect } from '@playwright/test';
for(const mode of ['immutable','live-quality','live-time']){
 test('real app static source retains single inference: '+mode,async({context,page})=>{
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await context.route('**/*',route=>new URL(route.request().url()).origin==='http://127.0.0.1:4185'?route.continue():route.abort());
  await context.addInitScript(mode=>{
   window.reviewNativeFrame=window.VideoFrame;
   let release;window.reviewModelGate=new Promise(resolve=>{release=resolve;});
   window.reviewProbe={first:null,release(){release();},stream:null};
   const nativePlay=HTMLMediaElement.prototype.play,rvfc=HTMLVideoElement.prototype.requestVideoFrameCallback;
   HTMLMediaElement.prototype.play=function(){
    if(this.srcObject instanceof MediaStream)rvfc.call(this,(_now,m)=>{window.reviewProbe.first={time:m.mediaTime,count:m.presentedFrames};});
    return nativePlay.call(this);
   };
   // Exercise the actual advertised compatibility capability combination.
   Object.defineProperty(HTMLVideoElement.prototype,'requestVideoFrameCallback',{configurable:true,value:undefined});
   if(mode!=='immutable')window.VideoFrame=undefined;
   if(mode==='live-time')Object.defineProperty(HTMLVideoElement.prototype,'getVideoPlaybackQuality',{configurable:true,value:undefined});
   Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia:async()=>{
    const canvas=document.createElement('canvas');canvas.width=64;canvas.height=64;const ctx=canvas.getContext('2d');ctx.fillStyle='red';ctx.fillRect(0,0,64,64);
    const stream=canvas.captureStream(0);window.reviewProbe.stream=stream;
    setTimeout(()=>stream.getVideoTracks()[0].requestFrame(),30);return stream;
   }}});
  },mode);
  await page.goto('/');await expect.poll(()=>page.evaluate(()=>window.reviewProbe.first)).not.toBeNull();
  await expect(page.getByLabel('カメラ状態')).toHaveText('カメラ映像を取得しました');
  await page.evaluate(()=>window.reviewProbe.release());
  await expect.poll(()=>page.evaluate(()=>window.independentSdk.calls)).toBeGreaterThan(0);
  await page.waitForTimeout(mode==='live-time'?2300:650);
  const evidence=await page.evaluate(()=>({calls:window.independentSdk.calls,inputs:window.independentSdk.inputs,first:window.reviewProbe.first,
   inference:document.querySelector('[aria-label="推論FPS"]').textContent,status:document.querySelector('[aria-label="性能状態"]').textContent}));
  console.log('INDEPENDENT actual app',mode,JSON.stringify(evidence));
  await page.getByRole('banner').getByRole('button',{name:'停止',exact:true}).click();
  expect(errors).toEqual([]);
  expect(evidence.calls).toBe(1);
  expect(new Set(evidence.inputs.map(input=>input.timestamp)).size).toBe(1);
 });
}
