// Explicit independent re-review build only. Never used by normal production.
const probe=window.independentSdk={calls:0,creates:0,inputs:[]};
function execute(input){
  const Frame=window.reviewNativeFrame;
  const image=input instanceof Frame?input:new Frame(input);
  const canvas=document.createElement('canvas');canvas.width=image.displayWidth;canvas.height=image.displayHeight;
  const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);
  probe.calls++;
  if(probe.inputs.length<40)probe.inputs.push({timestamp:image.timestamp,pixel:Array.from(ctx.getImageData(0,0,1,1).data),kind:input instanceof Frame?'immutable':'live'});
  if(image!==input)image.close();
  const until=performance.now()+1;while(performance.now()<until){ /* explicit synthetic SDK work */ }
}
export const FilesetResolver={forVisionTasks:async()=>{await window.reviewModelGate;return {};}};
export const ObjectDetector={createFromOptions:async()=>{probe.creates++;return{close(){},detectForVideo(input){execute(input);return{detections:[]};}};}};
export const ImageSegmenter={createFromOptions:async()=>{probe.creates++;return{close(){},segmentForVideo(input){execute(input);return{confidenceMasks:[],close(){}};}};}};
