import {chromium} from '@playwright/test';
import {writeFileSync} from 'node:fs';
import assert from 'node:assert/strict';
const browser=await chromium.launch({executablePath:'/usr/bin/chromium'});const page=await browser.newPage();const result={environment:{browser:browser.version(),node:process.version},html:[],warmup:[],trials:[]};
const persist=()=>writeFileSync(new URL('../hud-independent.json',import.meta.url),JSON.stringify(result,null,2));
try{
 await page.goto('http://127.0.0.1:4177/docs/verification/pr06-rereview/experiment/index.html');await page.waitForFunction(()=>!!window.independentHud);
 for(const profile of ['empty','quantile300','truncated'])for(const open of [false,true]){
  const html=await page.evaluate(({profile,open})=>({before:window.independentHud.html(profile,'before',open),after:window.independentHud.html(profile,'after',open)}),{profile,open});assert.equal(html.after,html.before);result.html.push({profile,open,equal:true});
  for(const version of ['before','after'])result.warmup.push(await page.evaluate(o=>window.independentHud.trial(o.profile,o.version,o.open,16),{profile,version,open}));
  for(let pair=0;pair<6;pair++)for(const version of pair%2?['after','before']:['before','after']){
   const t=await page.evaluate(o=>window.independentHud.trial(o.profile,o.version,o.open,32),{profile,version,open});assert.equal(t.snapshotCalls,32);assert.equal(t.renderCalls,32);assert.equal(t.readCalls,version==='after'?32:0);if(profile==='quantile300'){assert.equal(t.coverage.measured,300);assert.equal(t.coverage.quantileCount,300);}if(profile==='truncated'){assert.equal(t.coverage.count,2048);assert.equal(t.coverage.truncated,true);}result.trials.push({pair,...t});persist();
  }
  console.log(JSON.stringify({profile,open,trials:result.trials.length}));
 }
}finally{persist();await browser.close();}
