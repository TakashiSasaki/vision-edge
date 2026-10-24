// Explicit independent review build; not part of normal production.
const probe=window.review2Sdk={calls:0,successes:0,failures:0,creates:0,images:[],owned:[]};
const observer=document.createElement('canvas');observer.dataset.reviewObserver='true';
function execute(input){
  const isCanvas=input instanceof HTMLCanvasElement;
  const width='videoWidth'in input?input.videoWidth:'displayWidth'in input?input.displayWidth:input.width;
  const height='videoHeight'in input?input.videoHeight:'displayHeight'in input?input.displayHeight:input.height;
  observer.width=width;observer.height=height;const ctx=observer.getContext('2d',{willReadFrequently:true});ctx.drawImage(input,0,0);
  probe.calls++;
  if(probe.images.length<16)probe.images.push({kind:isCanvas?'canvas':'timestamp'in input?'frame':'video',width,height,sourceTimestamp:'timestamp'in input?input.timestamp:null,pixels:Array.from(ctx.getImageData(0,0,width,height).data)});
  if(isCanvas&&probe.owned.length<16)probe.owned.push(input);
  if(probe.failures>0){probe.failures--;throw Error('Independent SDK fault');}
  const end=performance.now()+1;while(performance.now()<end){ /* synthetic SDK cost */ }
  probe.successes++;
}
export const FilesetResolver={forVisionTasks:async()=>{await window.review2ModelGate;return {};}};
export const ObjectDetector={createFromOptions:async()=>{probe.creates++;return{close(){},detectForVideo(input){execute(input);return{detections:[]};}};}};
export const ImageSegmenter={createFromOptions:async()=>{probe.creates++;return{close(){},segmentForVideo(input){execute(input);return{confidenceMasks:[],close(){}};}};}};
