import { chromium } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';
const root=import.meta.dirname;
const browser=await chromium.launch({executablePath:'/usr/bin/chromium'});
const page=await browser.newPage();
const output={environment:{source:execFileSync('git',['-C','/tmp/pr06-product','rev-parse','HEAD'],{encoding:'utf8'}).trim(),browser:browser.version(),node:process.version,method:'independent order/count/sample validator, inspected unchanged author production-compiled workload; 4 balanced pairs per config; 8 RGBA/16 VideoFrame iterations'},warmup:[],trials:[]};
try{
 await page.goto('http://127.0.0.1:4175/docs/verification/pr06/experiment/index.html');await page.waitForFunction(()=>typeof window.runPipelineTrial==='function');
 for(const [width,height] of [[640,480],[1280,720],[1920,1080]])for(const path of ['rgba','video-frame'])for(const model of ['mediapipe','selfie-segmenter']){
  const options={width,height,path,model,iterations:path==='rgba'?8:16};
  for(const enabled of [false,true])output.warmup.push(await page.evaluate(x=>window.runPipelineTrial(x),{...options,enabled}));
  for(let pair=0;pair<4;pair++)for(const enabled of pair%2?[true,false]:[false,true]){
   const trial=await page.evaluate(x=>window.runPipelineTrial(x),{...options,enabled});const c=trial.counts,n=options.iterations;
   assert.equal(c.sdk,n);assert.equal(c.processed,n);assert.equal(c.drawn,n);assert.equal(c.captured,3*n);assert.equal(c.compared,path==='rgba'?5*n-3:0);assert.equal(c.released,path==='rgba'?0:3*n);assert.equal(c.maskRead,model==='mediapipe'?0:n);assert.equal(c.maskClosed,c.maskRead);assert.equal(c.snapshot,enabled?n/8:0);if(!enabled)assert.equal(c.metricClock,0);
   for(const stage of ['engine','draw','total','hud']){assert.equal(trial.samples[stage].length,stage==='hud'?n/8:n);assert(trial.samples[stage].every(x=>Number.isFinite(x)&&x>=0));}
   output.trials.push({pair,...trial});await writeFile(`${root}/review-overhead.json`,JSON.stringify(output,null,2));
  }
  console.log(JSON.stringify({options,trials:output.trials.length}));
 }
}finally{await browser.close();}
