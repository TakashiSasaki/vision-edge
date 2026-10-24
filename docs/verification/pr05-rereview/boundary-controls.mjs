import { readFileSync, writeFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { clientPointToVideo } from '../../../src/services/videoCoordinates.ts';

const vectors=JSON.parse(readFileSync(new URL('./independent-vectors.json',import.meta.url),'utf8'));
const failures=[],counts={};
for(const row of vectors){
  globalThis.devicePixelRatio=row.dpr;
  const actual=clientPointToVideo(...row.point,row.rect,...row.video);
  const pass=row.expected===null?actual===null:actual&&Math.abs(actual.x-row.expected[0])<1e-12&&Math.abs(actual.y-row.expected[1])<1e-12;
  counts[row.kind]=(counts[row.kind]??0)+1;
  if(!pass)failures.push({...row,actual});
}
// Overflow, collapse, invalid coordinates and dimensions must fail closed,
// without throwing or returning normalized values outside the unit square.
const invalid=[];
for(const bad of [0,-1,NaN,Infinity,-Infinity])for(const field of ['width','height','vw','vh']){
  const values={width:960,height:540,vw:640,vh:480,[field]:bad};
  const actual=clientPointToVideo(200,200,{left:0,top:0,width:values.width,height:values.height},values.vw,values.vh);
  invalid.push({field,bad:String(bad),actual});assert.equal(actual,null);
}
for(const bad of [NaN,Infinity,-Infinity])for(const field of ['x','y','left','top']){
  const values={x:200,y:200,left:0,top:0,[field]:bad};
  const actual=clientPointToVideo(values.x,values.y,{left:values.left,top:values.top,width:960,height:540},640,480);
  invalid.push({field,bad:String(bad),actual});assert.equal(actual,null);
}
for(const [rect,video]of [
  [{left:1e308,top:1e308,width:1e308,height:1e308},[640,480]],
  [{left:1e308,top:1e308,width:1,height:1},[640,480]],
  [{left:0,top:0,width:Number.MIN_VALUE,height:Number.MIN_VALUE},[Number.MAX_VALUE,1]],
  [{left:0,top:0,width:960,height:540},[Number.MAX_VALUE,Number.MAX_VALUE]],
]){
  const actual=clientPointToVideo(rect.left,rect.top,rect,...video);
  invalid.push({rect,video,actual});assert.equal(actual,null);
}
writeFileSync(new URL('./independent-results.json',import.meta.url),JSON.stringify({count:vectors.length,counts,failures,invalid},null,2)+'\n');
console.log(`${vectors.length} representable-boundary checks; ${failures.length} mismatches; ${invalid.length} invalid/overflow/collapse checks passed; ${JSON.stringify(counts)}`);
assert.equal(failures.length,0,JSON.stringify(failures.slice(0,3)));
