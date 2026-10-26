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
