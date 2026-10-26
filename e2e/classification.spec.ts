import { expect, test, type Page } from '@playwright/test';
interface SpeechCi { texts: string[]; cancels: number; utterances: SpeechSynthesisUtterance[]; finish: () => void }
declare global { interface Window { classifierSpeechCi: SpeechCi; releaseClassifierCi?: () => void; classifierCameraCi: { dispose: () => void } } }
async function setup(page: Page) {
  await page.route('**/*', route => /^http:\/\/127\.0\.0\.1:417[34]\//.test(route.request().url()) ? route.continue() : route.abort('blockedbyclient'));
  await page.addInitScript(() => {
    const streams: MediaStream[] = [], timers: number[] = [];
    window.classifierCameraCi = { dispose() { timers.forEach(clearInterval); streams.forEach(s => s.getTracks().forEach(t => t.stop())); } };
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: async () => {
      const canvas = document.createElement('canvas'); canvas.width=640; canvas.height=480; const ctx=canvas.getContext('2d')!; let n=0;
      const draw=()=>{ ctx.fillStyle=`rgb(${++n%255},100,100)`; ctx.fillRect(0,0,640,480); }; draw();
      const stream=canvas.captureStream(10); streams.push(stream); timers.push(window.setInterval(draw,100)); return stream;
    } } });
    const probe: SpeechCi = window.classifierSpeechCi = { texts: [], cancels:0, utterances:[], finish() { const u=probe.utterances.at(-1); u?.onend?.(new Event('end') as SpeechSynthesisEvent); } };
    Object.defineProperty(window,'speechSynthesis',{configurable:true,value:{ getVoices:()=>[], cancel:()=>{probe.cancels++;}, speak:(u:SpeechSynthesisUtterance)=>{probe.texts.push(u.text);probe.utterances.push(u);u.onstart?.(new Event('start') as SpeechSynthesisEvent);} }});
  });
}
const model = (page: Page) => page.getByRole('banner').getByRole('combobox',{name:'モデル',exact:true});
const variant = (page: Page) => page.getByRole('banner').getByRole('combobox',{name:'バリアント',exact:true});
const panel = (page: Page) => page.getByRole('region',{name:'画像全体の分類結果'});
test('production both classifier variants are BLOCKED, accessible on mobile and cannot request weights', async ({page},info)=>{
  await setup(page); await page.setViewportSize({width:375,height:812}); const requests:string[]=[];
  page.on('request',r=>{if(r.url().includes('/image_classifier/'))requests.push(r.url());}); await page.goto('/');
  await model(page).selectOption('image-classifier'); await expect(page.getByText(/BLOCKED：/)).toBeVisible();
  for(const id of ['lite0','lite2']) { await variant(page).selectOption(id); await expect(page.getByText(/BLOCKED：/)).toBeVisible(); await expect(panel(page).getByRole('listitem')).toHaveCount(0); }
  for(const c of await panel(page).getByRole('checkbox').all()) await expect(c).toBeDisabled();
  await expect(page.getByRole('button',{name:'分類音声は候補ボタンから手動で開始'})).toBeDisabled();
  await page.getByLabel('分類delegate要求').selectOption('CPU'); await expect(page.getByText(/BLOCKED：/)).toBeVisible(); expect(requests).toEqual([]);
  expect(await page.evaluate(()=>typeof window.classifierCi)).toBe('undefined');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
  for(const url of await page.locator('script[src]').evaluateAll(nodes=>nodes.map(n=>(n as HTMLScriptElement).src).filter(url=>new URL(url).origin===location.origin))) expect(await (await page.request.get(url)).text()).not.toContain('CLASSIFIER_CI_ONLY');
  await page.screenshot({path:info.outputPath('classifier-blocked-mobile.png')});
});
test('CI double only: both whole-image variants show ranked raw Top5, exact translation/score/ID and threshold', async ({page})=>{
  await setup(page); await page.goto('http://127.0.0.1:4174'); await model(page).selectOption('image-classifier');
  expect(await page.evaluate(()=>window.classifierCi.marker)).toBe('CLASSIFIER_CI_ONLY');
  await expect(panel(page).getByRole('listitem')).toHaveCount(5); const first=panel(page).getByRole('listitem').first();
  await expect(first).toContainText('順位 1'); await expect(first).toContainText('SDK index 7'); await expect(first).toContainText('goldfish, Carassius auratus'); await expect(first).toContainText('金魚'); await expect(first).toContainText('0.95');
  await expect(panel(page).getByRole('listitem').nth(2)).toContainText('UNKNOWN, Exact Raw Label'); await expect(panel(page).getByRole('listitem').nth(2)).toContainText('日本語訳未登録'); expect(await panel(page).innerText()).not.toContain('%');
  await variant(page).selectOption('lite2'); await expect(first).toContainText('coffee mug'); await expect(first).toContainText('コーヒーマグ');
  const threshold=page.getByLabel('分類スコアしきい値'); await threshold.focus(); await threshold.press('End'); for(let i=0;i<10;i++) await threshold.press('ArrowLeft'); await expect(panel(page).getByRole('listitem')).toHaveCount(1);
  await page.getByLabel('画像分類カメラ').click(); expect(await page.evaluate(()=>window.classifierSpeechCi.texts)).toEqual([]);
});
test('CI double only: manual speech pins current frame; stop clears and restart resumes without automatic speech', async ({page})=>{
  await setup(page); await page.goto('http://127.0.0.1:4174'); await model(page).selectOption('image-classifier'); await expect(panel(page).getByRole('listitem')).toHaveCount(5);
  expect(await page.evaluate(()=>window.classifierSpeechCi.texts)).toEqual([]); await panel(page).getByRole('button',{name:/順位1 .*読み上げ/}).click();
  await expect(page.getByText('読み上げ対象のフレームを保持中。終了後に分類を更新します。')).toBeVisible();
  const before=await page.evaluate(()=>({calls:window.classifierCi.calls,time:(document.querySelector('video') as HTMLVideoElement).currentTime}));
  await expect.poll(()=>page.evaluate(()=>(document.querySelector('video') as HTMLVideoElement).currentTime)).toBeGreaterThan(before.time);
  expect(await page.evaluate(()=>window.classifierCi.calls)).toBe(before.calls);
  await page.evaluate(()=>window.classifierSpeechCi.finish()); await expect.poll(()=>page.evaluate(()=>window.classifierCi.calls)).toBeGreaterThan(before.calls);
  await panel(page).getByRole('button',{name:/順位1 .*読み上げ/}).click(); await page.getByRole('banner').getByRole('button',{name:'停止',exact:true}).click(); await expect(panel(page).getByRole('listitem')).toHaveCount(0);
  const texts=await page.evaluate(()=>window.classifierSpeechCi.texts.length); await page.getByRole('banner').getByRole('button',{name:'カメラ開始',exact:true}).click(); await expect(panel(page).getByRole('listitem')).toHaveCount(5); expect(await page.evaluate(()=>window.classifierSpeechCi.texts.length)).toBe(texts);
});
test('CI double only: init/run failure, explicit retry and delegate request preserve truthful states', async ({page})=>{
  await setup(page); await page.goto('http://127.0.0.1:4174'); await page.evaluate(()=>{window.classifierCi.failCreates=1;}); await model(page).selectOption('image-classifier');
  await expect(page.getByRole('alert')).toContainText('分類モデルの初期化に失敗'); await page.getByRole('button',{name:'分類モデルを再試行'}).click(); await expect(panel(page).getByRole('listitem')).toHaveCount(5);
  await page.evaluate(()=>{window.classifierCi.failRuns=1;}); await expect(page.getByRole('alert')).toContainText('分類推論に失敗'); await expect(panel(page).getByRole('listitem')).toHaveCount(0); await page.getByRole('button',{name:'分類推論を再試行'}).click(); await expect(panel(page).getByRole('listitem')).toHaveCount(5);
  await page.getByLabel('分類delegate要求').selectOption('CPU'); await expect(page.getByLabel('分類バックエンド状態')).toContainText('SDK設定: CPU'); await expect(page.getByLabel('分類バックエンド状態')).toContainText('実バックエンド: 未計測'); expect(await page.evaluate(()=>window.classifierCi.creates.at(-1))).toBe('lite0:CPU');
});
test('CI double only: rapid ABA and cross-family selection close obsolete initializations', async ({page})=>{
  await setup(page); await page.goto('http://127.0.0.1:4174'); await page.evaluate(()=>{window.classifierCi.gate=new Promise<void>(resolve=>{window.releaseClassifierCi=resolve;});});
  await model(page).selectOption('image-classifier'); await expect.poll(()=>page.evaluate(()=>window.classifierCi.creates.length)).toBe(1); await page.evaluate(()=>{window.classifierCi.gate=undefined;});
  await variant(page).selectOption('lite2'); await expect(panel(page).getByRole('listitem').first()).toContainText('coffee mug'); await variant(page).selectOption('lite0'); await expect(panel(page).getByRole('listitem').first()).toContainText('goldfish'); await page.evaluate(()=>window.releaseClassifierCi?.());
  await expect.poll(()=>page.evaluate(()=>window.classifierCi.closes)).toBe(2); await expect(panel(page).getByRole('listitem').first()).toContainText('goldfish');
  await model(page).selectOption('mediapipe'); await expect(page.getByText('画像全体の分類 Top5',{exact:true})).toHaveCount(0); await model(page).selectOption('image-classifier'); await expect(panel(page).getByRole('listitem')).toHaveCount(5);
  expect(await page.evaluate(()=>window.classifierSpeechCi.texts)).toEqual([]);
});

test.describe('B-F1 usable classification results geometry', () => {
  test.use({ hasTouch: true });
  for (const viewport of [{ width:320, height:640 }, { width:375, height:667 }]) test(`B-F1 mobile ${viewport.width}x${viewport.height} full results, touch and keyboard reachability`, async ({page},info) => {
    await setup(page); await page.setViewportSize(viewport); await page.goto('http://127.0.0.1:4174'); await model(page).selectOption('image-classifier');
    await expect(panel(page).getByRole('listitem')).toHaveCount(5);
    const dimensions = await panel(page).evaluate(element => {
      const main=element.closest('main')!, video=main.querySelector('video')!;
      const read=(e:Element)=>({ height:e.getBoundingClientRect().height, width:e.getBoundingClientRect().width, clientHeight:e.clientHeight, scrollHeight:e.scrollHeight, overflowY:getComputedStyle(e).overflowY });
      return { panel:read(element), main:read(main), video:read(video), viewport:{width:innerWidth,height:innerHeight} };
    });
    await info.attach('B-F1-dimensions.json',{body:JSON.stringify(dimensions,null,2),contentType:'application/json'});
    await page.screenshot({path:info.outputPath('B-F1-before-scroll.png')});
    // Mobile results must take their natural full content height; the outer main
    // scroll owner must provide enough room to read a candidate and its action.
    expect(dimensions.panel.height).toBeGreaterThan(600);
    expect(dimensions.panel.scrollHeight).toBeLessThanOrEqual(dimensions.panel.clientHeight+2);
    expect(dimensions.main.clientHeight).toBeGreaterThanOrEqual(200);
    expect(dimensions.video.width).toBeLessThanOrEqual(viewport.width);
    expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(viewport.width);
    const reachable = async (selector: ReturnType<typeof page.locator>) => selector.evaluate(element => {
      const r=element.getBoundingClientRect(), m=element.closest('main')!.getBoundingClientRect();
      const hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
      return { fullyVisible:r.top>=m.top-1&&r.bottom<=m.bottom+1&&r.left>=0&&r.right<=innerWidth, height:r.height,width:r.width, hit:hit===element||element.contains(hit) };
    });
    for (let i=0;i<5;i++) {
      const candidate=panel(page).getByRole('listitem').nth(i), speak=candidate.getByRole('button',{name:/読み上げ/});
      await candidate.scrollIntoViewIfNeeded();
      expect((await reachable(candidate.locator('output'))).fullyVisible).toBe(true);
      const action=await reachable(speak); expect(action.fullyVisible).toBe(true); expect(action.hit).toBe(true); expect(action.height).toBeGreaterThanOrEqual(32); expect(action.width).toBeGreaterThanOrEqual(44);
      const before=await page.evaluate(()=>window.classifierSpeechCi.texts.length); await speak.tap(); await expect.poll(()=>page.evaluate(()=>window.classifierSpeechCi.texts.length)).toBe(before+1); await page.evaluate(()=>window.classifierSpeechCi.finish());
    }
    // Focus must move the fifth candidate into the usable outer viewport even
    // after returning to the preview; Enter must operate the same manual action.
    await page.locator('main').evaluate(main=>{main.scrollTop=0;});
    const last=panel(page).getByRole('listitem').last().getByRole('button',{name:/読み上げ/}); await page.getByLabel('分類スコアしきい値').focus(); for(let i=0;i<5;i++) await page.keyboard.press('Tab'); expect((await reachable(last)).fullyVisible).toBe(true); await expect(last).toBeFocused();
    const before=await page.evaluate(()=>window.classifierSpeechCi.texts.length); await page.keyboard.press('Enter'); await expect.poll(()=>page.evaluate(()=>window.classifierSpeechCi.texts.length)).toBe(before+1); await page.evaluate(()=>window.classifierSpeechCi.finish());
    await page.screenshot({path:info.outputPath('B-F1-fifth-candidate.png')});
    const final=await panel(page).evaluate(e=>({panelScrollTop:e.scrollTop,outerScrollTop:e.closest('main')!.scrollTop})); expect(final.panelScrollTop).toBe(0); expect(final.outerScrollTop).toBeGreaterThan(0);
    await info.attach('B-F1-final-scroll.json',{body:JSON.stringify(final),contentType:'application/json'});
  });
  test('B-F1 desktop keeps split columns and its independently scrollable results', async ({page},info) => {
    await setup(page); await page.setViewportSize({width:1280,height:800}); await page.goto('http://127.0.0.1:4174'); await model(page).selectOption('image-classifier'); await expect(panel(page).getByRole('listitem')).toHaveCount(5);
    const geometry=await panel(page).evaluate(e=>{const m=e.closest('main')!,p=m.firstElementChild!.getBoundingClientRect(),r=e.getBoundingClientRect();return{display:getComputedStyle(m).display,previewTop:p.top,panelTop:r.top,previewRight:p.right,panelLeft:r.left,client:e.clientHeight,scroll:e.scrollHeight,overflow:getComputedStyle(e).overflowY};});
    expect(geometry.display).toBe('grid'); expect(Math.abs(geometry.previewTop-geometry.panelTop)).toBeLessThan(2); expect(geometry.panelLeft).toBeGreaterThan(geometry.previewRight); expect(geometry.scroll).toBeGreaterThan(geometry.client); expect(geometry.overflow).toBe('auto');
    const last=panel(page).getByRole('listitem').last().getByRole('button',{name:/読み上げ/}); await last.focus(); await expect(last).toBeInViewport(); await page.keyboard.press('Enter'); await expect.poll(()=>page.evaluate(()=>window.classifierSpeechCi.texts.length)).toBe(1);
    await page.screenshot({path:info.outputPath('B-F1-desktop.png')}); await info.attach('B-F1-desktop-geometry.json',{body:JSON.stringify(geometry),contentType:'application/json'});
  });
});
