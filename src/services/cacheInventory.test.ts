import { afterEach, expect, it, vi } from 'vitest';
import { inventorySnapshot, mountInventory } from '../../public/model-cache.mjs';
const base = () => ({ enabled:true,schemaVersion:2,catalogCount:0,observedAt:new Date().toISOString(),variants:{} });
const row = (variantId='lite0') => ({id:'image-classifier',variantId,file:`efficientnet_${variantId}.tflite`,status:'cached',size:2048,sha256:'a'.repeat(64),expectedSize:2048,expectedSha256:'a'.repeat(64),filePresent:true,integrity:'pinned',error:null});
afterEach(()=>{document.body.replaceChildren();vi.restoreAllMocks();});
it('complete empty observation differs from unavailable/legacy/partial data',()=>{
  expect(inventorySnapshot(base())).toMatchObject({rows:[],invalid:0});
  for(const override of [{enabled:false},{schemaVersion:1},{catalogCount:1},{variants:null},{observedAt:'invalid'}])expect(()=>inventorySnapshot({...base(),...override})).toThrow();
});
it('variant isolation and partial invalid integrity never claim a corrupt file as cached',()=>{
  const data={...base(),catalogCount:2,variants:{'image-classifier:lite0':row(),'image-classifier:lite2':{...row('lite2'),sha256:'b'.repeat(64)}}};
  const s=inventorySnapshot(data);expect(s.invalid).toBe(1);expect(s.rows.filter(x=>x.cached).map(x=>x.variant)).toEqual(['lite0']);expect(s.rows[1].readiness).toContain('不明');
});
it('legacy self-recorded hash is distinct from pinned bytes; files absent from disk are not cached',()=>{
  const data={...base(),catalogCount:2,variants:{'mediapipe:lite0':{...row(),id:'mediapipe',integrity:'self-recorded'},'image-classifier:lite2':{...row('lite2'),filePresent:false}}};
  const s=inventorySnapshot(data);expect(s.rows[0]).toMatchObject({cached:true,integrity:'取得時ハッシュ一致'});expect(s.rows[1].cached).toBe(false);
});
it('rendering uses text, hides paths/arbitrary errors, refreshes read-only and clears stale counts after API failure',async()=>{
  for(const id of ['refresh','notice','observed','cached','configured']){const el=document.createElement(id==='refresh'?'button':'div');el.id=id;document.body.append(el);}
  const data={...base(),catalogCount:2,variants:{'image-classifier:lite0':row(),'image-classifier:lite2':{...row('lite2'),file:'/private/token/<img src=x onerror=alert(1)>',error:'secret /home/person/token'}}};
  const fetch=vi.fn().mockResolvedValue({ok:true,json:async()=>data}); const refresh=mountInventory(document,fetch);await vi.waitFor(()=>expect(document.getElementById('notice')!.textContent).toContain('一部'));
  expect(document.querySelector('#cached')!.textContent).toContain('lite0');expect([...document.querySelectorAll('#cached tbody tr')].map(r=>r.children[1].textContent)).toEqual(['lite0']);expect(document.body.textContent).not.toContain('/private');expect(document.body.textContent).not.toContain('/home');expect(document.querySelector('img')).toBeNull();
  fetch.mockRejectedValue(Error('denied private'));await refresh();expect(document.getElementById('notice')!.textContent).toContain('不明');expect(document.querySelector('#cached')!.children).toHaveLength(0);
  expect(fetch).toHaveBeenCalledTimes(2);for(const [url,options]of fetch.mock.calls){expect(url).toBe('/api/model-cache/status');expect(options).toMatchObject({method:'GET',cache:'no-store'});}
});
