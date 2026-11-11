import { expect,test } from '@playwright/test';
test.beforeEach(async({page})=>{await page.route('**/*',route=>new URL(route.request().url()).origin==='http://127.0.0.1:4173'?route.continue():route.abort('blockedbyclient'));});
const row=(id:string,variantId:string,file:string,integrity='pinned')=>({id,variantId,file,status:'cached',size:2048,sha256:'a'.repeat(64),expectedSize:2048,expectedSha256:'a'.repeat(64),filePresent:true,integrity,error:null});
const snapshot=(variants:Record<string,unknown>)=>({enabled:true,schemaVersion:2,observedAt:'2026-10-26T05:00:00.000Z',catalogCount:Object.keys(variants).length,variants});
test('static inventory: populated variants, safe mobile layout, read-only explicit refresh and reachable app link',async({page},info)=>{
  await page.setViewportSize({width:320,height:640});let calls=0;const modelRequests:string[]=[];
  page.on('request',r=>{if(r.url().includes('/models/'))modelRequests.push(r.url());});
  await page.route('**/api/model-cache/status',async route=>{calls++;expect(route.request().method()).toBe('GET');await route.fulfill({json:snapshot({'image-classifier:lite0':row('image-classifier','lite0','efficientnet_lite0.tflite'),'image-classifier:lite2':{...row('image-classifier','lite2','efficientnet_lite2.tflite'),status:'unavailable',filePresent:false,size:null,sha256:null,integrity:'not-present'},'mediapipe:lite0':row('mediapipe','lite0','efficientdet_lite0.tflite','self-recorded')})});});
  await page.goto('/model-cache.html');await expect(page.getByRole('status')).toContainText('保存・検証済み 2件 ／ 設定済み 3件');
  await expect(page.locator('#cached')).toContainText('固定ハッシュ一致');await expect(page.locator('#cached')).toContainText('取得時ハッシュ一致');await expect(page.locator('#cached').getByRole('cell',{name:'lite2',exact:true})).toHaveCount(0);await expect(page.locator('#configured')).toContainText('lite2');
  await expect(page.locator('#cached a.model-link')).toHaveCount(2);
  await expect(page.locator('#cached a.model-link').first()).toHaveAttribute('href', /^\/api\/model-cache\/models\//);
  await expect(page.locator('#configured a.model-link')).toHaveCount(0);
  await expect(page.locator('#page-url')).toContainText('/model-cache.html');
  await expect(page.getByRole('button',{name:'ページURLをクリップボードにコピー'})).toBeVisible();
  await expect(page.locator('#observed')).toContainText('サーバー観測時刻');expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);expect(calls).toBe(1);expect(modelRequests).toEqual([]);
  await page.getByRole('button',{name:'状態を更新'}).focus();await page.keyboard.press('Enter');await expect.poll(()=>calls).toBe(2);expect(modelRequests).toEqual([]);
  await page.screenshot({path:info.outputPath('cache-inventory-320.png'),fullPage:true});
  await page.goto('/');await expect(page.getByRole('link',{name:'コンテナのモデルキャッシュ一覧'})).toHaveAttribute('href','/model-cache.html');
  await expect(page.getByRole('link',{name:'サイトマップ'})).toHaveAttribute('href','/sitemap.html');
  await expect(page.locator('link[rel="sitemap"]')).toHaveAttribute('href','/sitemap.xml');
  await expect(page.locator('meta[name="sitemap"]')).toHaveAttribute('content','/sitemap.xml');
});
test('static inventory: empty snapshot, unavailable API and partial invalid data stay distinct',async({page})=>{
  let mode='empty';await page.route('**/api/model-cache/status',route=>mode==='unavailable'?route.fulfill({status:503,body:'unavailable'}):route.fulfill({json:mode==='empty'?snapshot({}):snapshot({'image-classifier:lite0':row('image-classifier','lite0','efficientnet_lite0.tflite'),'image-classifier:lite2':{...row('image-classifier','lite2','<img src=x onerror=alert(1)>'),error:'/private/secret',integrity:'invalid'}})}));
  await page.goto('/model-cache.html');await expect(page.getByRole('status')).toContainText('保存・検証済み 0件 ／ 設定済み 0件');mode='unavailable';await page.getByRole('button',{name:'状態を更新'}).click();await expect(page.getByRole('status')).toContainText('取得不能');await expect(page.getByRole('status')).toContainText('不明');await expect(page.locator('#cached table')).toHaveCount(0);
  mode='partial';await page.getByRole('button',{name:'状態を更新'}).click();await expect(page.getByRole('status')).toContainText('一部の状態が不正');await expect(page.locator('#cached')).toContainText('lite0');await expect(page.locator('#cached').getByRole('cell',{name:'lite2',exact:true})).toHaveCount(0);expect(await page.locator('body').innerText()).not.toContain('/private');await expect(page.locator('img')).toHaveCount(0);
});
test('actual static preview without cache APIs reports unknown rather than zero cached files',async({page})=>{
  await page.goto('/model-cache.html');await expect(page.getByRole('status')).toContainText('取得不能');await expect(page.getByRole('status')).toContainText('状態は不明');
  await expect(page.getByRole('status')).not.toContainText('0件');
});
for(const integrity of ['pinned','self-recorded'])test(`C31-F1: static page rejects unavailable absent file with ${integrity} verification`,async({page},info)=>{
  let requests=0;const modelRequests:string[]=[];page.on('request',r=>{if(r.url().includes('/models/'))modelRequests.push(r.url());});
  await page.route('**/api/model-cache/status',route=>{requests++;expect(route.request().method()).toBe('GET');return route.fulfill({json:snapshot({'image-classifier:lite0':{...row('image-classifier','lite0','efficientnet_lite0.tflite',integrity),status:'unavailable',filePresent:false,size:null,sha256:null}})});});
  await page.goto('/model-cache.html');await expect(page.getByRole('status')).toContainText('一部の状態が不正');await expect(page.locator('#configured').getByRole('cell',{name:'不明',exact:true})).toHaveCount(1);await expect(page.locator('#configured')).not.toContainText('ハッシュ一致');await expect(page.locator('#cached table')).toHaveCount(0);
  expect(requests).toBe(1);expect(modelRequests).toEqual([]);await page.screenshot({path:info.outputPath(`c31-f1-${integrity}.png`),fullPage:true});
});
test('non-JS user agent can read all 11 model file URLs on /model-cache.html',async({browser,baseURL})=>{
  const context=await browser.newContext({javaScriptEnabled:false,baseURL});
  try {
    const page=await context.newPage();
    await page.goto('/model-cache.html');
    const links=page.locator('#static-model-catalog a.catalog-model-url');
    await expect(links).toHaveCount(11);
    const hrefs=await links.evaluateAll(els=>els.map(e=>e.getAttribute('href')??''));
    const texts=await links.evaluateAll(els=>els.map(e=>(e.textContent??'').trim()));
    expect(hrefs).toContain('/api/model-cache/models/efficientnet_lite0.tflite?sha256=6c7ab0a6e5dcbf38a8c33b960996a55a3b4300b36a018c4545801de3a3c8bde0');
    expect(hrefs).toContain('/api/model-cache/models/yolov8n-seg.onnx');
    for(let i=0;i<hrefs.length;i++){
      expect(hrefs[i]).toMatch(/^\/api\/model-cache\/models\//);
      expect(texts[i]).toContain(hrefs[i]);
    }
  } finally {
    await context.close();
  }
});
