import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { clientPointToVideo } from '../../../src/services/videoCoordinates.ts';

// Python Fraction creates an independent, once-rounded fixed numeric oracle.
const vectors = JSON.parse(readFileSync(new URL('./numeric-vectors.json',import.meta.url),'utf8'));
const observations = [];
const failures = [];
for (const v of vectors) {
  globalThis.devicePixelRatio=v.dpr;
  const actual=clientPointToVideo(...v.point,v.rect,...v.video);
  observations.push({...v,actual});
  const pass=v.expected===null ? actual===null : actual && Math.abs(actual.x-v.expected[0])<1e-12 && Math.abs(actual.y-v.expected[1])<1e-12;
  if(!pass)failures.push(observations.at(-1));
}
for (const bad of [0,-1,NaN,Infinity,-Infinity]) {
  for (const index of [2,3,4,5]) {
    const values=[0,0,960,540,640,480]; values[index]=bad;
    const actual=clientPointToVideo(200,200,{left:values[0],top:values[1],width:values[2],height:values[3]},values[4],values[5]);
    assert.equal(actual,null); observations.push({invalidIndex:index,bad:String(bad),actual});
  }
}
for (const bad of [NaN,Infinity,-Infinity]) {
  for (const index of [0,1,2,3]) {
    const v=[200,200,0,0]; v[index]=bad;
    assert.equal(clientPointToVideo(v[0],v[1],{left:v[2],top:v[3],width:960,height:540},640,480),null);
    observations.push({invalidPointerIndex:index,bad:String(bad),actual:null});
  }
}
writeFileSync(new URL('./numeric-results.json',import.meta.url),JSON.stringify(observations,null,2)+'\n');
writeFileSync(new URL('./numeric-failures.json',import.meta.url),JSON.stringify(failures,null,2)+'\n');
console.log(`${observations.length} independent numeric checks; ${failures.length} strict boundary mismatches`);
assert.equal(failures.length,0,JSON.stringify(failures.slice(0,2)));
