import {createServer} from 'vite';
import {chromium} from '@playwright/test';
import {writeFileSync, mkdirSync, mkdtempSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
// Optional git ref enables comparison with an immutable baseline.
const ref = process.argv[2];
const label = ref ? 'before' : 'after';
mkdirSync('.cache/stereo-diagnostics', {recursive:true});
const output = mkdtempSync('.cache/stereo-diagnostics/run-');
const moduleUrl = ref ? `/${output}/baseline.ts` : '/src/services/stereoDepthWorker.ts';
if (ref) writeFileSync(`${output}/baseline.ts`, execFileSync('git', ['show', `${ref}:src/services/stereoDepthWorker.ts`]));
writeFileSync(`${output}/blank.html`, '<!doctype html><title>Stereo diagnostic</title>');
const server=await createServer({configFile:false,server:{host:'127.0.0.1',port:4317},root:process.cwd()});
await server.listen();
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox','--enable-unsafe-webgpu','--use-angle=swiftshader']});
try {
const page=await browser.newPage();
await page.route('**/*', route => new URL(route.request().url()).hostname === '127.0.0.1' ? route.continue() : route.abort());
await page.goto(`http://127.0.0.1:4317/${output}/blank.html`);
const result=await page.evaluate(async(moduleUrl)=>{
 const m=await import(moduleUrl);
 const gpu=await m.createWebGpuStereoComputeEngine();
 const adapter=await navigator.gpu.requestAdapter();
 const rows=[]; const images=[];
 const w=320,h=180, options={maxDisparity:48,windowSize:7,confidenceThreshold:.35};
 for(const file of ['wuppertal-schwebebahn-sbs.webm','bahn-latzen-crossing-sbs.webm','segway-ice-real-sbs.webm']){
 const v=document.createElement('video');v.src='/stereo-videos/'+file;v.muted=true;
 await new Promise((res,rej)=>{v.onloadeddata=res;v.onerror=rej;});
 for(const t of [1,4,6]){
 await new Promise(res=>{v.onseeked=res;v.currentTime=t;});
 const c=document.createElement('canvas');c.width=w*2;c.height=h;const ctx=c.getContext('2d');ctx.drawImage(v,0,0,w*2,h);
 const l=ctx.getImageData(0,0,w,h).data,r=ctx.getImageData(w,0,w,h).data;
 const maps=[m.computeSadDisparity(l,r,w,h,options),await gpu.estimate(l,r,w,h,options)];
 const panel=document.createElement('canvas');panel.width=w*3;panel.height=h*2;const p=panel.getContext('2d');p.drawImage(c,0,0,w,h,0,0,w,h);
 for(let k=0;k<2;k++){
 const d=maps[k].disparity;let valid=0,jumps=0,edges=0;
 const img=new ImageData(m.renderDisparityColormapToRgba(d,w,h,0,24,'turbo'),w,h);
 for(let i=0;i<d.length;i++){if(d[i]>=0){valid++;}
 if(i%w&&d[i]>=0&&d[i-1]>=0){edges++;if(Math.abs(d[i]-d[i-1])>3)jumps++;}}
 rows.push({file,t,engine:k?'wgsl':'sad',valid:valid/d.length,jumpRate:jumps/edges});p.putImageData(img,(k+1)*w,0);
 const heat=document.createElement('canvas');heat.width=w;heat.height=h;heat.getContext('2d').putImageData(img,0,0);p.drawImage(c,0,0,w,h,(k+1)*w,h,w,h);p.globalAlpha=.55;p.drawImage(heat,(k+1)*w,h);p.globalAlpha=1;
 } images.push({name:file+'-'+t,png:panel.toDataURL().split(',')[1]});
 }
 }
 // Controls: no correspondence and periodic ambiguity must not look like reliable depth.
 for(const kind of ['flat','periodic','shift']){
 const l=new Uint8ClampedArray(w*h*4),r=new Uint8ClampedArray(l.length);
 const hash=(x,y)=>{let v=(x+1)+Math.imul(y+1,4099);v=Math.imul(v^(v>>>16),0x45d9f3b);v=Math.imul(v^(v>>>16),0x45d9f3b);return (v^(v>>>16))&255;};
 const pattern=(x,y)=>kind==='flat'?128:kind==='periodic'?((x%8)<4?40:210):hash(x,y);
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){let i=(y*w+x)*4; l.set([pattern(x,y),pattern(x,y),pattern(x,y),255],i);r.set([pattern(x+8,y),pattern(x+8,y),pattern(x+8,y),255],i);}
 for(const [engine,out] of [['sad',m.computeSadDisparity(l,r,w,h,options)],['wgsl',await gpu.estimate(l,r,w,h,options)]]){
 let valid=0,bad=0;for(let y=10;y<h-10;y++)for(let x=60;x<w-10;x++){const d=out.disparity[y*w+x];if(d>=0){valid++;if(Math.abs(d-8)>1)bad++;}}
 rows.push({control:kind,engine,valid:valid/((h-20)*(w-70)),badAmongValid:bad/valid});
 }
 }
 gpu.destroy(); return {rows,images,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description}};
},moduleUrl);
for(const im of result.images)writeFileSync(`${output}/${im.name}.png`,Buffer.from(im.png,'base64'));
delete result.images;
writeFileSync(`${output}/results.json`,JSON.stringify(result,null,2));
if(label !== 'before') for(const row of result.rows.filter(r=>r.control)) {
 if(row.control === 'shift' ? row.valid < .98 || row.badAmongValid > .01 : row.valid > .01) throw new Error('Control failed: '+JSON.stringify(row));
}
console.log(JSON.stringify({output,adapter:result.adapter,frames:18,controls:6,mode:label}));
} finally {await browser.close();await server.close();}
