// @vitest-environment node
// Synthetic byte fixtures and test-only module pins. No actual model/network
// acquisition and no production flag for replacing the reviewed pins.
import { afterEach, expect, it, vi } from 'vitest';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { createServer as httpServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { createServer as viteServer } from 'vite';
vi.mock('./classifierAssetManifest.ts', async importOriginal => {
  const original = await importOriginal<typeof import('./classifierAssetManifest')>();
  const { createHash } = await import('node:crypto');
  return { IMAGE_CLASSIFIER_ASSETS: Object.fromEntries(Object.entries(original.IMAGE_CLASSIFIER_ASSETS).map(([id, e], i) => {
    const bytes = Buffer.alloc(2048 + i * 1024, 31 + i), hash = createHash('sha256').update(bytes).digest('hex');
    return [id, { ...e, expectedSize: bytes.length, expectedSha256: hash, localUrl: `/api/model-cache/models/${e.file}?sha256=${hash}` }];
  })) };
});
import { ContainerModelCacheManager, containerModelCachePlugin } from './containerModelCacheServer';
import { IMAGE_CLASSIFIER_ASSETS } from './classifierAssetManifest';
const variants = ['lite0','lite2'] as const;
const payload = (v: typeof variants[number]) => Buffer.alloc(v === 'lite0' ? 2048 : 3072, v === 'lite0' ? 31 : 32);
const dirs: string[] = [], close: (() => Promise<unknown>)[] = [];
afterEach(async () => { await Promise.all(close.splice(0).map(f=>f())); await Promise.all(dirs.splice(0).map(d=>rm(d,{recursive:true,force:true}))); vi.restoreAllMocks(); });
async function manager(fetchImpl = vi.fn<typeof fetch>(async url => {
  const variant = variants.find(v => IMAGE_CLASSIFIER_ASSETS[v].upstreamUrl === String(url));
  if (!variant) return new Response(Buffer.alloc(2048, 9));
  return new Response(payload(variant));
})) {
  const dir = await mkdtemp(path.join(tmpdir(), 'classifier-cache-')); dirs.push(dir);
  return { m: new ContainerModelCacheManager({ cacheDir: dir, fetchImpl }), dir, fetch: fetchImpl };
}
async function request(m: ContainerModelCacheManager, url: string, method = 'GET') {
  let status = 200, bytes = Buffer.alloc(0); const headers: Record<string,string> = {};
  const res = { get statusCode() {return status;}, set statusCode(v: number) {status=v;}, setHeader: (k: string,v: string)=>{headers[k]=String(v);}, end: (v?: Buffer|string)=>{bytes=Buffer.from(v??'');} };
  await m.createMiddleware()({url,method} as IncomingMessage, res as unknown as ServerResponse, e=>{throw e??Error('Unexpected route');});
  return {status,bytes,headers};
}
it('status and inventory observation never acquire models; all eleven identities are present', async()=>{
  const {m,fetch}=await manager(); const s=await m.getStatusResponse();
  expect(s).toMatchObject({schemaVersion:2,catalogCount:11}); expect(Object.keys(s.models)).toHaveLength(4); expect(Object.keys(s.variants!)).toHaveLength(11);
  for (const v of variants) { expect(s.variants!['image-classifier:'+v]).toMatchObject({status:'unavailable',filePresent:false,integrity:'not-present',sha256:null,size:null}); expect(s.classification![v]).toMatchObject({status:'unavailable',filePresent:false,sha256:null}); }
  const response=await request(m,'/api/model-cache/status'); expect(response.status).toBe(200); expect(fetch).not.toHaveBeenCalled(); expect(response.bytes.toString()).not.toContain(m.cacheDir);
});
it.each(variants)('%s exact fixture passes acquisition/restart/reuse/serve with the same pins',async v=>{
  const {m,fetch,dir}=await manager(), e=IMAGE_CLASSIFIER_ASSETS[v];
  expect(await m.prefetchVariant('image-classifier:'+v)).toMatchObject({status:'cached',filePresent:true,size:e.expectedSize,sha256:e.expectedSha256,integrity:'pinned'});
  const other=new ContainerModelCacheManager({cacheDir:dir,fetchImpl:fetch});
  expect((await other.getStatusResponse()).classification![v]).toMatchObject({status:'cached',size:e.expectedSize,sha256:e.expectedSha256});
  const response=await request(other,e.localUrl); expect(response.status).toBe(200); expect(response.bytes).toEqual(payload(v)); expect(response.headers['X-Model-SHA256']).toBe(e.expectedSha256); expect(response.headers['Cache-Control']).toBe('no-store');
  expect(fetch).toHaveBeenCalledTimes(1);
  expect((await request(other,e.localUrl,'HEAD')).bytes.length).toBe(0); expect(fetch).toHaveBeenCalledTimes(1);
});
for (const v of variants) it.each(['hash','short','oversized','header','interrupted'] as const)(`${v} rejects %s acquisition, cleans temp files and never retries implicitly`,async kind=>{
  const bytes=payload(v), fake=vi.fn<typeof fetch>(async()=>{
    if(kind==='interrupted')return new Response(new ReadableStream({start(c){c.enqueue(bytes.subarray(0,1024));c.error(Error('interrupted /private/token'));}}));
    const body=kind==='hash'?Buffer.alloc(bytes.length,99):kind==='short'?bytes.subarray(0,bytes.length-1):kind==='oversized'?Buffer.concat([bytes,Buffer.alloc(1)]):bytes;
    return new Response(body,kind==='header'?{headers:{'Content-Length':String(bytes.length+1)}}:undefined);
  });
  const {m,dir}=await manager(fake), result=await m.prefetchVariant('image-classifier:'+v);
  expect(result.status).toBe('unavailable'); expect(result.sha256).toBeNull(); expect(result.error).toBe({hash:'Pinned model SHA256 mismatch',short:'Pinned model size mismatch',oversized:'Model exceeds cache size limit',header:'Pinned model size mismatch',interrupted:'Cache acquisition or validation failed'}[kind]); expect(result.error).not.toContain('/private'); expect(result.size).toBeNull();
  expect(await readdir(dir)).toEqual([]); const response=await request(m,IMAGE_CLASSIFIER_ASSETS[v].localUrl); expect(response.status).toBe(503); expect(response.bytes.toString()).toContain('no upstream fallback'); expect(fake).toHaveBeenCalledTimes(1);
});
it.each(variants)('%s rejects forged self-consistent cache bytes and wrong size on reuse/serve',async v=>{
  const {m,dir,fetch}=await manager(), e=IMAGE_CLASSIFIER_ASSETS[v], key='image-classifier:'+v;
  await m.prefetchVariant(key); const file=path.join(dir,e.file), metadata=file+'.identity.json';
  const corrupt=Buffer.alloc(e.expectedSize,89); await writeFile(file,corrupt);
  const record=JSON.parse(await readFile(metadata,'utf8')); record.sha256=createHash('sha256').update(corrupt).digest('hex'); await writeFile(metadata,JSON.stringify(record));
  expect((await m.getStatusResponse()).variants![key]).toMatchObject({status:'unavailable',filePresent:true,integrity:'invalid',sha256:null,error:expect.stringContaining('SHA256')});
  expect((await request(m,e.localUrl)).status).toBe(503);
  await writeFile(file,corrupt.subarray(0,1024)); expect((await m.getStatusResponse()).variants![key]).toMatchObject({status:'unavailable',integrity:'invalid',error:expect.stringContaining('size')}); expect(fetch).toHaveBeenCalledTimes(1);
});
it('orphaned/interrupted publication and missing files are not adopted as ready',async()=>{
  const {m,dir,fetch}=await manager(); const e=IMAGE_CLASSIFIER_ASSETS.lite0;
  await writeFile(path.join(dir,e.file),payload('lite0'));
  expect((await m.getStatusResponse()).variants!['image-classifier:lite0']).toMatchObject({status:'unavailable',integrity:'invalid',filePresent:true});
  expect((await request(m,e.localUrl)).status).toBe(503); expect(fetch).not.toHaveBeenCalled();
  await rm(path.join(dir,e.file)); const s=await m.getStatusResponse(); expect(s.variants!['image-classifier:lite0']).toMatchObject({filePresent:false,integrity:'not-present'});
});
it('same-variant concurrent acquisition deduplicates while both variants remain isolated',async()=>{
  let release!:()=>void; const gate=new Promise<void>(r=>{release=r;});
  const fake=vi.fn<typeof fetch>(async url=>{await gate; const v=String(url).includes('lite0')?'lite0':'lite2';return new Response(payload(v));});
  const {m}=await manager(fake); const requests=[m.prefetchVariant('image-classifier:lite0'),m.prefetchVariant('image-classifier:lite0'),m.prefetchVariant('image-classifier:lite2')];
  await vi.waitFor(()=>expect(fake).toHaveBeenCalledTimes(2));
  const pending=await m.getStatusResponse(); for(const v of variants)expect(pending.variants!['image-classifier:'+v]).toMatchObject({status:'downloading',integrity:'pending',size:null,sha256:null});
  release(); await Promise.all(requests);
  for (const v of variants)expect((await request(m,IMAGE_CLASSIFIER_ASSETS[v].localUrl)).bytes).toEqual(payload(v)); expect(fake).toHaveBeenCalledTimes(2);
});
it('unversioned, wrong-pin, extra-query and unknown classifier URLs cannot trigger acquisition',async()=>{
  const {m,fetch}=await manager(), e=IMAGE_CLASSIFIER_ASSETS.lite0;
  for(const url of [e.localUrl.split('?')[0],e.localUrl.replace(e.expectedSha256,'0'.repeat(64)),e.localUrl+'&other=1'])expect((await request(m,url)).status).toBe(400);
  expect((await request(m,'/api/model-cache/models/efficientnet_wrong.tflite')).status).toBe(404);expect(fetch).not.toHaveBeenCalled();
});
it('actual Vite serve plugin supplies read-only status and validated model routes with injected fixture fetch only',async()=>{
  const {dir,fetch:fake}=await manager();
  const vite=await viteServer({configFile:false,root:process.cwd(),plugins:[containerModelCachePlugin({cacheDir:dir,fetchImpl:fake,autoPrefetchOnStart:false})],server:{middlewareMode:true,hmr:false,watch:null}});
  close.push(()=>vite.close()); const server=httpServer(vite.middlewares);await new Promise<void>(r=>server.listen(0,'127.0.0.1',r)); close.push(()=>new Promise<void>((r,e)=>server.close(x=>x?e(x):r())));
  const address=server.address(); if(!address||typeof address==='string')throw Error('No local port'); const base=`http://127.0.0.1:${address.port}`;
  const before=await (await fetch(base+'/api/model-cache/status')).json();expect(before.catalogCount).toBe(11);expect(fake).not.toHaveBeenCalled();
  expect((await (await fetch(base+'/model-cache.html')).text())).toContain('mountInventory');expect(fake).not.toHaveBeenCalled();
  const response=await fetch(base+IMAGE_CLASSIFIER_ASSETS.lite2.localUrl);expect(response.status).toBe(200);expect(Buffer.from(await response.arrayBuffer())).toEqual(payload('lite2'));expect(fake).toHaveBeenCalledTimes(1);
  const after=await (await fetch(base+'/api/model-cache/status')).json();expect(after.classification.lite2.status).toBe('cached');expect(after.classification.lite0.status).toBe('unavailable');expect(fake).toHaveBeenCalledTimes(1);
});
it.each(variants)('%s revalidates the exact served buffer if bytes change after status validation',async v=>{
  const {m,dir,fetch}=await manager(), e=IMAGE_CLASSIFIER_ASSETS[v], key='image-classifier:'+v;
  await m.prefetchVariant(key); const original=m.syncVariantDiskState.bind(m);
  vi.spyOn(m,'syncVariantDiskState').mockImplementationOnce(async k=>{
    const ready=await original(k);expect(ready.status).toBe('cached');
    const corrupt=Buffer.alloc(e.expectedSize,88), file=path.join(dir,e.file);
    await writeFile(file,corrupt); const record=JSON.parse(await readFile(file+'.identity.json','utf8'));record.sha256=createHash('sha256').update(corrupt).digest('hex');await writeFile(file+'.identity.json',JSON.stringify(record));
    return ready;
  });
  const response=await request(m,e.localUrl);expect(response.status).toBe(503);expect(response.bytes.toString()).toContain('no upstream fallback');expect(response.headers['Content-Type']).toContain('application/json');expect(fetch).toHaveBeenCalledTimes(1);
});
it('C31-F1 actual server lifecycle snapshots remain coherent for the static inventory',async()=>{
  const {inventorySnapshot}=await import('../../public/model-cache.mjs');
  let release!:()=>void;const gate=new Promise<void>(r=>{release=r;});
  const fake=vi.fn<typeof fetch>(async url=>{await gate;return String(url).includes('lite0')?new Response(payload('lite0')):new Response('denied',{status:403});});
  const {m,dir}=await manager(fake);
  const empty=inventorySnapshot(await m.getStatusResponse());expect(empty.invalid).toBe(0);expect(empty.rows.filter(r=>r.cached)).toHaveLength(0);expect(fake).not.toHaveBeenCalled();
  const pending=m.prefetchVariant('image-classifier:lite0');await vi.waitFor(()=>expect(fake).toHaveBeenCalledTimes(1));
  const downloading=inventorySnapshot(await m.getStatusResponse());expect(downloading.invalid).toBe(0);expect(downloading.rows.find(r=>r.variant==='lite0'&&r.family.includes('画像分類'))).toMatchObject({cached:false,integrity:'検証待ち'});
  release();await pending;const cached=inventorySnapshot(await m.getStatusResponse());expect(cached.invalid).toBe(0);expect(cached.rows.filter(r=>r.cached)).toHaveLength(1);
  await writeFile(path.join(dir,IMAGE_CLASSIFIER_ASSETS.lite0.file),Buffer.alloc(1024));
  const invalid=inventorySnapshot(await m.getStatusResponse());expect(invalid.invalid).toBe(0);expect(invalid.rows.find(r=>r.variant==='lite0'&&r.family.includes('画像分類'))).toMatchObject({cached:false,integrity:'検証失敗'});
  await rm(path.join(dir,IMAGE_CLASSIFIER_ASSETS.lite0.file));expect(inventorySnapshot(await m.getStatusResponse()).invalid).toBe(0);
  await m.prefetchVariant('image-classifier:lite2');const failure=inventorySnapshot(await m.getStatusResponse());expect(failure.invalid).toBe(0);expect(failure.rows.filter(r=>r.cached)).toHaveLength(0);
});
