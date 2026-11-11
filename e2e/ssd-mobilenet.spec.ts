import {expect,test,type Page} from '@playwright/test';
async function localOnly(page:Page){await page.route('**/*',route=>/^http:\/\/127\.0\.0\.1:417[34]\//.test(route.request().url())?route.continue():route.abort('blockedbyclient'));}
const model=(page:Page)=>page.getByRole('banner').getByRole('combobox',{name:'モデル',exact:true});
const variant=(page:Page)=>page.getByRole('banner').getByRole('combobox',{name:'バリアント',exact:true});
test('production SSD is selectable but explicitly BLOCKED before weight fetch, retry remains truthful on mobile',async({page},info)=>{
  await localOnly(page);await page.setViewportSize({width:320,height:640});await page.addInitScript(()=>{Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia:()=>Promise.reject(new DOMException('controlled camera denial','NotAllowedError'))}});});
  const weights:string[]=[];page.on('request',r=>{if(r.url().includes('ssd_mobilenet_v2.tflite'))weights.push(r.url());});
  await page.goto('/');await model(page).selectOption('mediapipe');await expect(variant(page)).toHaveValue('lite0');await expect(model(page).getByRole('option',{name:'MediaPipe Object Detector',exact:true})).toHaveCount(1);await expect(variant(page).getByRole('option')).toHaveCount(3);
  await variant(page).selectOption('ssd-mobilenet-v2');await expect(page.getByRole('alert')).toContainText('BLOCKED：');await expect(page.getByRole('alert')).toContainText('SDK0.10.14');await expect(page.getByRole('alert')).toContainText('取得・配信・実推論は無効');
  await page.getByRole('button',{name:'モデルを再試行',exact:true}).click();await expect(page.getByRole('alert')).toContainText('BLOCKED：');expect(weights).toEqual([]);expect(await page.evaluate(()=>typeof window.visionSdkProbe)).toBe('undefined');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);await page.screenshot({path:info.outputPath('ssd-blocked-320.png'),fullPage:true});
});
test('CI SDK double only: Lite0 -> blocked SSD -> Lite2 preserves camera and stop/resume, without implicit detector fallback',async({page})=>{
  await localOnly(page);await page.addInitScript(()=>{
    const probes={requests:0,streams:[] as MediaStream[],timers:[] as number[],dispose(){this.timers.forEach(clearInterval);this.streams.forEach(s=>s.getTracks().forEach(t=>t.stop()));}};
    Object.assign(window,{ssdCamera:probes});Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia:async()=>{
      probes.requests++;const c=document.createElement('canvas');c.width=640;c.height=480;const ctx=c.getContext('2d')!;const draw=()=>{ctx.fillStyle='#2468ab';ctx.fillRect(0,0,640,480);};draw();const stream=c.captureStream(10);probes.streams.push(stream);probes.timers.push(window.setInterval(draw,100));return stream;
    }}});
  });
  await page.goto('http://127.0.0.1:4174');try{
    await expect(page.getByLabel('カメラ状態')).toContainText('カメラ映像を取得しました');await model(page).selectOption('mediapipe');await expect(page.getByRole('status')).toContainText('モデル準備完了');const creates=await page.evaluate(()=>window.visionSdkProbe.creates);
    await variant(page).selectOption('ssd-mobilenet-v2');await expect(page.getByRole('alert')).toContainText('BLOCKED：');expect(await page.evaluate(()=>window.visionSdkProbe.creates)).toBe(creates);await page.getByRole('button',{name:'モデルを再試行',exact:true}).click();await expect(page.getByRole('alert')).toContainText('BLOCKED：');expect(await page.evaluate(()=>window.visionSdkProbe.creates)).toBe(creates);
    await variant(page).selectOption('lite2');await expect(page.getByRole('status')).toContainText('モデル準備完了');await expect.poll(()=>page.evaluate(()=>window.visionSdkProbe.creates)).toBe(creates+1);
    expect(await page.evaluate(()=>window.ssdCamera.requests)).toBe(1);await page.getByRole('banner').getByRole('button',{name:'停止',exact:true}).click();await expect.poll(()=>page.evaluate(()=>window.ssdCamera.streams[0].getTracks().every(t=>t.readyState==='ended'))).toBe(true);const count=await page.evaluate(()=>window.visionSdkProbe.calls);await page.waitForTimeout(250);expect(await page.evaluate(()=>window.visionSdkProbe.calls)).toBe(count);
    await page.getByRole('banner').getByRole('button',{name:'カメラ開始',exact:true}).click();await expect.poll(()=>page.evaluate(()=>window.ssdCamera.requests)).toBe(2);await expect.poll(()=>page.evaluate(()=>window.visionSdkProbe.calls)).toBeGreaterThan(count);await expect(variant(page)).toHaveValue('lite2');
  }finally{await page.evaluate(()=>window.ssdCamera.dispose());}
});
test('static inventory distinguishes configured SSD policy block from cached classifier and unknown malformed states',async({page})=>{
  await localOnly(page);let mode='blocked';const row={id:'mediapipe',variantId:'ssd-mobilenet-v2',file:'ssd_mobilenet_v2.tflite',status:'unavailable',filePresent:false,size:null,sha256:null,integrity:'blocked',acquisitionEnabled:false,blockedReason:'SDK compatibility pending',error:'BLOCKED'};
  await page.route('**/api/model-cache/status',r=>{expect(r.request().method()).toBe('GET');return r.fulfill({json:{enabled:true,schemaVersion:2,observedAt:new Date().toISOString(),catalogCount:1,variants:{'mediapipe:ssd-mobilenet-v2':mode==='blocked'?row:{...row,status:'cached',filePresent:true,size:11316189,sha256:'b8ccb1a25d45455ba52e85f26531948e1cb75efeb94c7c3d456d54fd4d6fbdd2',integrity:'pinned',expectedSize:11316189,expectedSha256:'b8ccb1a25d45455ba52e85f26531948e1cb75efeb94c7c3d456d54fd4d6fbdd2',error:null}}}});});
  await page.goto('/model-cache.html');await expect(page.getByRole('status')).toContainText('保存・検証済み 0件 ／ 設定済み 1件');await expect(page.locator('#configured')).toContainText('BLOCKED（モデル未検証）');await expect(page.locator('#configured')).toContainText('未検証・取得無効');await expect(page.locator('#cached table')).toHaveCount(0);
  mode='forged';await page.getByRole('button',{name:'状態を更新'}).click();await expect(page.getByRole('status')).toContainText('一部の状態が不正');await expect(page.locator('#configured')).not.toContainText('ハッシュ一致');await expect(page.locator('#cached table')).toHaveCount(0);
});
declare global{interface Window{ssdCamera:{requests:number;streams:MediaStream[];timers:number[];dispose:()=>void}}}
