// Model-free registry/cache/SDK boundary controls. No model download/inference.
import {afterEach,beforeEach,expect,it,vi} from 'vitest';
import {mkdtemp,rm,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import path from 'node:path';
import type {IncomingMessage,ServerResponse} from 'node:http';
import {FilesetResolver,ObjectDetector} from '@mediapipe/tasks-vision';
import * as assets from './modelAssetResolver';
import {ALL_MODEL_REGISTRY,MODEL_REGISTRY,modelMetadata} from './modelRegistry';
import {IMAGE_CLASSIFIER_ASSETS} from './classifierAssetManifest';
import {ContainerModelCacheManager} from './containerModelCacheServer';
import {VisionEngine,type DetectorTask,type VisionTaskFactory} from './visionEngine';
import {inventorySnapshot} from '../../public/model-cache.mjs';
const id='ssd-mobilenet-v2',key='mediapipe:'+id;
const entry=assets.MODEL_VARIANT_CATALOG[key];
const directories:string[]=[],engines:VisionEngine[]=[];
function engine(factory?:VisionTaskFactory){const e=new VisionEngine(factory);engines.push(e);return e;}
function task(label='dog',score=.432109876,index=90):DetectorTask{return {close:vi.fn(),detectForVideo:vi.fn(()=>({detections:[{categories:[{categoryName:label,displayName:'',score,index}],boundingBox:{originX:64,originY:48,width:128,height:96,angle:0},keypoints:[]}]}))};}
function factory(create:VisionTaskFactory['detector']):VisionTaskFactory{return {load:vi.fn(async()=>({wasmLoaderPath:'controlled-loader',wasmBinaryPath:'controlled-wasm'})),detector:create,segmenter:vi.fn()};}
function deferred<T>(){let resolve!:(v:T)=>void;const promise=new Promise<T>(r=>{resolve=r;});return{promise,resolve};}
function canvas(){return Object.assign(document.createElement('canvas'),{width:640,height:480});}
beforeEach(()=>{vi.spyOn(console,'warn').mockImplementation(()=>{});assets.setContainerModelCacheEnabledForTest(false);});
afterEach(async()=>{engines.splice(0).forEach(e=>e.dispose());assets.setContainerModelCacheEnabledForTest(null);vi.restoreAllMocks();await Promise.all(directories.splice(0).map(d=>rm(d,{recursive:true,force:true})));});
async function manager(){const directory=await mkdtemp(path.join(tmpdir(),'ssd-gate-'));directories.push(directory);const fetch=vi.fn<typeof globalThis.fetch>(async()=>new Response(Buffer.alloc(2048,19)));return {m:new ContainerModelCacheManager({cacheDir:directory,fetchImpl:fetch}),directory,fetch};}
async function request(m:ContainerModelCacheManager,url:string){let status=200,body='';const headers:Record<string,string>={};const response={get statusCode(){return status;},set statusCode(v:number){status=v;},setHeader:(k:string,v:string)=>{headers[k]=String(v);},end:(v?:string|Buffer)=>{body=String(v??'');}};await m.createMiddleware()({url,method:'GET'} as IncomingMessage,response as unknown as ServerResponse,e=>{throw e??Error('Unknown route');});return {status,body,headers};}
it('adds one third detector variant while retaining Lite0 default, four engines/five families and classifier pins',()=>{
  const detector=modelMetadata('mediapipe');expect(detector.name).toBe('MediaPipe Object Detector');expect(detector.defaultVariantId).toBe('lite0');expect(detector.variants.map(v=>v.id)).toEqual(['lite0','lite2',id]);
  expect(MODEL_REGISTRY).toHaveLength(4);expect(ALL_MODEL_REGISTRY).toHaveLength(5);expect(MODEL_REGISTRY.flatMap(m=>m.variants).length+Object.keys(IMAGE_CLASSIFIER_ASSETS).length).toBe(12);
  expect(entry).toMatchObject({modelId:'mediapipe',variantId:id,file:'ssd_mobilenet_v2.tflite',expectedSize:11316189,expectedSha256:'b8ccb1a25d45455ba52e85f26531948e1cb75efeb94c7c3d456d54fd4d6fbdd2',license:'Apache-2.0',blockedReason:expect.stringContaining('SDK0.10.14')});
  expect(entry.upstreamUrl).toBe('https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float32/1/ssd_mobilenet_v2.tflite');expect(entry.localUrl).not.toBe(assets.getVariantCatalogEntry('mediapipe').localUrl);expect(assets.getVariantCatalogEntry('mediapipe').variantId).toBe('lite0');
  expect(IMAGE_CLASSIFIER_ASSETS.lite0.expectedSha256).toBe('6c7ab0a6e5dcbf38a8c33b960996a55a3b4300b36a018c4545801de3a3c8bde0');expect(IMAGE_CLASSIFIER_ASSETS.lite2.expectedSha256).toBe('94b6c84a5d6de20f932558de213b50e0338dcc06ab83d8d64b12128a476b4b83');
});
it.each([false,true])('production gate blocks SSD before all WASM/SDK/fetch activity, cache=%s',async enabled=>{
  assets.setContainerModelCacheEnabledForTest(enabled);const fetch=vi.spyOn(globalThis,'fetch').mockRejectedValue(Error('UNEXPECTED_FETCH'));const load=vi.spyOn(FilesetResolver,'forVisionTasks'),create=vi.spyOn(ObjectDetector,'createFromOptions');
  const e=engine();expect(await e.initialize('mediapipe',id)).toMatchObject({status:'error',model:'mediapipe',variantId:id,stage:'task',cause:expect.any(assets.ModelAssetsBlockedError)});
  await expect(assets.resolveModelAssetUrls('mediapipe',id)).rejects.toBeInstanceOf(assets.ModelAssetsBlockedError);const before=e.getState().requestId;expect((await e.initialize('mediapipe',id)).requestId).toBeGreaterThan(before);
  expect(fetch).not.toHaveBeenCalled();expect(load).not.toHaveBeenCalled();expect(create).not.toHaveBeenCalled();expect(e.detectVideoFrame(canvas(),'mediapipe',.3).status).toBe('not-ready');
});
it('a forged cached status cannot override SSD policy or redirect it to default weights',async()=>{
  assets.setContainerModelCacheEnabledForTest(true);const fetch=vi.spyOn(globalThis,'fetch').mockResolvedValue(new Response(JSON.stringify({enabled:true,wasm:{},models:{},variants:{[key]:{...assets.getVariantCatalogEntry('mediapipe'),status:'cached'}}}),{headers:{'Content-Type':'application/json'}}));
  await expect(assets.resolveModelAssetUrls('mediapipe',id)).rejects.toThrow('BLOCKED');expect(fetch).not.toHaveBeenCalled();
});
it.each([false,true])('cache status/serving/acquisition stay blocked with disk bytes=%s',async present=>{
  const {m,directory,fetch}=await manager();if(present){await writeFile(path.join(directory,entry.file),Buffer.alloc(2048,4));await writeFile(path.join(directory,entry.file+'.identity.json'),JSON.stringify({schemaVersion:1,modelId:'mediapipe',variantId:id,file:entry.file,localUrl:entry.localUrl,upstreamUrl:entry.upstreamUrl,size:11316189,sha256:entry.expectedSha256}));}
  const s=await m.getStatusResponse();expect(Object.keys(s.variants!)).toHaveLength(12);expect(s.variants![key]).toMatchObject({status:'unavailable',integrity:'blocked',filePresent:present,size:null,sha256:null,expectedSize:11316189,expectedSha256:entry.expectedSha256,acquisitionEnabled:false});
  expect((await m.prefetchVariant(key)).status).toBe('unavailable');const response=await request(m,entry.localUrl);expect(response.status).toBe(503);expect(response.body).toContain('BLOCKED');expect(fetch).not.toHaveBeenCalled();
  const view=inventorySnapshot(s);expect(view.invalid).toBe(0);const row=view.rows.find(r=>r.variant===id)!;expect(row.cached).toBe(false);expect(row.readiness).toContain('BLOCKED');expect(row.integrity).toBe('未検証・取得無効');
});
it('startup never acquires SSD; existing eleven acquisitions remain distinct',async()=>{
  const{m,fetch}=await manager();await m.prefetchAllModels();expect(fetch).toHaveBeenCalledTimes(11);expect(fetch.mock.calls.some(([url])=>String(url)===entry.upstreamUrl)).toBe(false);expect((await m.getStatusResponse()).variants![key].integrity).toBe('blocked');
});
it('controlled default factory uses existing GPU/VIDEO ObjectDetector options and exact SSD identity when test-only boundaries are substituted',async()=>{
  vi.spyOn(assets,'assertVariantAssetAvailable').mockImplementation(()=>{});vi.spyOn(assets,'resolveModelAssetUrls').mockResolvedValue({primaryUrl:entry.localUrl,fallbackUrl:null,source:'container-cache'});vi.spyOn(FilesetResolver,'forVisionTasks').mockResolvedValue({wasmLoaderPath:'controlled-loader',wasmBinaryPath:'controlled-wasm'});
  const t=task(),create=vi.spyOn(ObjectDetector,'createFromOptions').mockResolvedValue(t as ObjectDetector);const fetch=vi.spyOn(globalThis,'fetch').mockRejectedValue(Error('UNEXPECTED_NETWORK'));const e=engine();
  expect(await e.initialize('mediapipe',id)).toMatchObject({status:'ready',variantId:id,delegate:'GPU'});expect(create).toHaveBeenCalledWith(expect.anything(),{baseOptions:{modelAssetPath:entry.localUrl,delegate:'GPU'},scoreThreshold:.25,runningMode:'VIDEO'});expect(fetch).not.toHaveBeenCalled();
});
it('Lite0 -> SSD -> Lite2 reuses detector semantics with exact names/raw scores/boxes, no masks or class-index remapping',async()=>{
  const tasks=new Map<string,DetectorTask>(),create=vi.fn<VisionTaskFactory['detector']>(async(_f,delegate,variant)=>{expect(delegate).toBe('GPU');const t=task(variant===id?'Exact RAW SSD label, index90':'dog');tasks.set(variant!,t);return t;});const e=engine(factory(create));
  for(const variant of ['lite0',id,'lite2']){expect((await e.initialize('mediapipe',variant)).status).toBe('ready');const f=e.detectVideoFrame(canvas(),'mediapipe',.3,undefined,{bboxSmoothing:0});expect(f.status).toBe('ready');if(f.status==='ready'){expect(f.objects).toHaveLength(1);expect(f.objects[0]).toMatchObject({label:variant===id?'Exact RAW SSD label, index90':'dog',confidence:.432109876,bbox:{x:.1,y:.1,width:.2,height:.2}});expect(f.objects[0].maskCanvas).toBeUndefined();}}
  expect(tasks.get('lite0')!.close).toHaveBeenCalledTimes(1);expect(tasks.get(id)!.close).toHaveBeenCalledTimes(1);expect(create.mock.calls.map(c=>c[2])).toEqual(['lite0',id,'lite2']);
});
it('pending SSD completion cannot replace ready Lite2 and its stale task closes once',async()=>{
  const gate=deferred<DetectorTask>(),old=task('old SSD'),latest=task('new Lite2');const create=vi.fn<VisionTaskFactory['detector']>(async(_f,_d,variant)=>variant===id?gate.promise:latest);const e=engine(factory(create));
  const pending=e.initialize('mediapipe',id);await vi.waitFor(()=>expect(create).toHaveBeenCalledWith(expect.anything(),'GPU',id));expect(e.detectVideoFrame(canvas(),'mediapipe',.3).status).toBe('not-ready');await e.initialize('mediapipe','lite2');gate.resolve(old);expect((await pending).status).toBe('cancelled');expect(old.close).toHaveBeenCalledTimes(1);expect(e.getState()).toMatchObject({status:'ready',variantId:'lite2'});
});
it('SSD synchronous stale frame, clear/dispose and explicit init/run errors retain failure ownership',async()=>{
  const t=task('SSD stale'),replacement=task('Lite2');const create=vi.fn<VisionTaskFactory['detector']>().mockRejectedValueOnce(Error('SSD_INIT_CONTROL')).mockResolvedValueOnce(t).mockResolvedValue(replacement);const e=engine(factory(create));
  expect(await e.initialize('mediapipe',id)).toMatchObject({status:'error',cause:expect.objectContaining({message:'SSD_INIT_CONTROL'})});await e.initialize('mediapipe',id);
  vi.mocked(t.detectForVideo).mockImplementationOnce(()=>{throw Error('SSD_RUN_CONTROL');});expect(()=>e.detectVideoFrame(canvas(),'mediapipe',.3)).toThrow('SSD_RUN_CONTROL');
  vi.mocked(t.detectForVideo).mockImplementationOnce(()=>{void e.initialize('mediapipe','lite2');return {detections:[]};});expect(e.detectVideoFrame(canvas(),'mediapipe',.3).status).toBe('not-ready');await vi.waitFor(()=>expect(e.getState()).toMatchObject({status:'ready',variantId:'lite2'}));e.clearDetections();e.dispose();expect(e.detectVideoFrame(canvas(),'mediapipe',.3).status).toBe('not-ready');expect(t.close).toHaveBeenCalledTimes(1);
});
it.each(['cached','downloading','unavailable'])('test-only eligibility substitution proves pinned SSD %s resolution has no unverified upstream fallback',async status=>{
  const original=assets.MODEL_VARIANT_CATALOG[key];assets.MODEL_VARIANT_CATALOG[key]={...original,blockedReason:undefined};
  try{
    assets.setContainerModelCacheEnabledForTest(true);const row={...entry,id:'mediapipe',blockedReason:null,acquisitionEnabled:true,status,size:status==='cached'?entry.expectedSize:null,sha256:status==='cached'?entry.expectedSha256:null,filePresent:status==='cached'?true:status==='downloading'?null:false,integrity:status==='cached'?'pinned':status==='downloading'?'pending':'not-present',error:null};
    const fetch=vi.spyOn(globalThis,'fetch').mockResolvedValue(new Response(JSON.stringify({enabled:true,wasm:{},models:{},variants:{[key]:row}}),{headers:{'Content-Type':'application/json'}}));
    expect(await assets.resolveModelAssetUrls('mediapipe',id)).toEqual({primaryUrl:entry.localUrl,fallbackUrl:null,source:'container-cache'});expect(fetch).toHaveBeenCalledTimes(1);expect(fetch.mock.calls[0][0]).toBe('/api/model-cache/status');
  }finally{assets.MODEL_VARIANT_CATALOG[key]=original;}
});
it.each([{id:'mediapipe-wrong'},{variantId:'lite0'},{localUrl:assets.getVariantCatalogEntry('mediapipe').localUrl},{sha256:'0'.repeat(64)},{expectedSha256:'0'.repeat(64)},{size:2048},{integrity:'self-recorded'},{acquisitionEnabled:false},{error:'HTTP 403'}])('test-only eligibility substitution still rejects wrong cached SSD identity/integrity %j',async override=>{
  const original=assets.MODEL_VARIANT_CATALOG[key];assets.MODEL_VARIANT_CATALOG[key]={...original,blockedReason:undefined};
  try{
    assets.setContainerModelCacheEnabledForTest(true);const row={...entry,id:'mediapipe',blockedReason:null,acquisitionEnabled:true,status:'cached',size:entry.expectedSize,sha256:entry.expectedSha256,filePresent:true,integrity:'pinned',error:null,...override};
    vi.spyOn(globalThis,'fetch').mockResolvedValue(new Response(JSON.stringify({enabled:true,wasm:{},models:{},variants:{[key]:row}}),{headers:{'Content-Type':'application/json'}}));await expect(assets.resolveModelAssetUrls('mediapipe',id)).rejects.toBeInstanceOf(assets.ModelAssetsBlockedError);
  }finally{assets.MODEL_VARIANT_CATALOG[key]=original;}
});
it.each([false,true,null])('inventory displays coherent SSD blocked policy with presence=%s without verification success',async filePresent=>{
  const {m}=await manager();const s=await m.getStatusResponse();Object.assign(s.variants![key],{filePresent});const view=inventorySnapshot(s);expect(view.invalid).toBe(0);expect(view.rows.find(r=>r.variant===id)).toMatchObject({cached:false,integrity:'未検証・取得無効'});
  Object.assign(s.variants![key],{status:'cached',filePresent:true,size:entry.expectedSize,sha256:entry.expectedSha256,integrity:'pinned',error:null});const forged=inventorySnapshot(s);expect(forged.invalid).toBe(1);expect(forged.rows.find(r=>r.variant===id)?.cached).toBe(false);
});
it.each(['size','hash'])('controlled eligibility still rejects generated wrong SSD %s against actual production pins',async kind=>{
  const original={...entry};entry.blockedReason=undefined;
  try{
    const directory=await mkdtemp(path.join(tmpdir(),'ssd-production-pin-'));directories.push(directory);const fetch=vi.fn<typeof globalThis.fetch>(async()=>new Response(Buffer.alloc(kind==='size'?2048:entry.expectedSize!,77)));const m=new ContainerModelCacheManager({cacheDir:directory,fetchImpl:fetch});
    expect(await m.prefetchVariant(key)).toMatchObject({status:'unavailable',size:null,sha256:null,error:kind==='size'?'Pinned model size mismatch':'Pinned model SHA256 mismatch'});const http=await request(m,entry.localUrl);expect(http.status).toBe(503);expect(http.body).toContain('no upstream fallback');expect(fetch).toHaveBeenCalledTimes(1);
  }finally{Object.assign(entry,original);}
});
it('controlled eligibility/synthetic pins validate SSD-specific serving and attribution without redistributing or accepting a real model',async()=>{
  const original={...entry},bytes=Buffer.alloc(2048,39),hash=createHash('sha256').update(bytes).digest('hex');Object.assign(entry,{blockedReason:undefined,expectedSize:bytes.length,expectedSha256:hash,localUrl:entry.localUrl.split('?')[0]+'?sha256='+hash});
  try{
    const directory=await mkdtemp(path.join(tmpdir(),'ssd-synthetic-serving-'));directories.push(directory);const fetch=vi.fn<typeof globalThis.fetch>(async()=>new Response(bytes));const m=new ContainerModelCacheManager({cacheDir:directory,fetchImpl:fetch});
    expect(await m.prefetchVariant(key)).toMatchObject({status:'cached',integrity:'pinned',sha256:hash,size:2048});const http=await request(m,entry.localUrl);expect(http.status).toBe(200);expect(http.headers['X-Model-Variant']).toBe(id);expect(http.headers['X-Model-SHA256']).toBe(hash);expect(http.headers.Link).toContain('/licenses/ssd-mobilenet-v2.txt');expect(http.headers.Link).not.toContain('/licenses/classifier-models.txt');expect(fetch).toHaveBeenCalledTimes(1);
  }finally{Object.assign(entry,original);}
});
