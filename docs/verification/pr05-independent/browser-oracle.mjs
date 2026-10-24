import { chromium, expect } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const url=process.env.REVIEW_URL ?? 'http://127.0.0.1:4185';
const output=path.resolve(process.env.REVIEW_OUTPUT ?? 'docs/verification/pr05-independent/after');
const baseline=process.env.REVIEW_BASELINE==='1';
const skipMargins=process.env.REVIEW_NO_MARGIN==='1';
mkdirSync(output,{recursive:true});
const cases=[
  {name:'pillarbox',video:[640,480],box:[960,540],content:[120,0,720,540]},
  {name:'letterbox',video:[1280,720],box:[800,600],content:[0,75,800,450]},
  {name:'equal',video:[1280,720],box:[960,540],content:[0,0,960,540]},
  {name:'square-source',video:[600,600],box:[960,540],content:[210,0,540,540]},
  {name:'square-box',video:[1280,720],box:[600,600],content:[0,131.25,600,337.5]},
  {name:'fractional',video:[640,480],box:[960.75,540.25],content:[120.20833333333331,0,720.3333333333334,540.25]},
];
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true});
const results=[];
try {
  for (const c of baseline?cases.slice(0,2):cases) {
    for (const dpr of baseline?[2]:[1,1.25,2,3]) {
      const context=await browser.newContext({viewport:{width:1400,height:1000},deviceScaleFactor:dpr});
      const page=await context.newPage(); const errors=[]; page.on('pageerror',e=>errors.push(e.message));
      await context.route('**/*',r=>new URL(r.request().url()).origin===url?r.continue():r.abort());
      await context.addInitScript(({video})=>{
        const probe=window.independentProbe={speech:[],arcs:[],boxes:[],requests:0,sources:[],streams:[]};
        const arc=CanvasRenderingContext2D.prototype.arc;
        CanvasRenderingContext2D.prototype.arc=function(...args){probe.arcs.push(args.slice(0,3));return arc.apply(this,args);};
        const box=CanvasRenderingContext2D.prototype.roundRect;
        CanvasRenderingContext2D.prototype.roundRect=function(...args){probe.boxes.push(args.slice(0,4)); if(probe.boxes.length>50)probe.boxes.shift();return box.apply(this,args);};
        speechSynthesis.speak=u=>{probe.speech.push(u.text);};
        Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia:async()=>{
          probe.requests++;const source=document.createElement('canvas');[source.width,source.height]=video;
          const ctx=source.getContext('2d');ctx.fillStyle='rgb(12,34,56)';ctx.fillRect(0,0,...video);
          ctx.fillStyle='rgb(240,180,40)';ctx.fillRect(64,48,128,96);
          probe.sources.push(source);const stream=source.captureStream(0);probe.streams.push(stream);return stream;
        }}});
      },c);
      let result={name:c.name,dpr,reference:c};
      try {
        await page.goto(url);
        await page.getByRole('banner').getByRole('combobox',{name:'モデル',exact:true}).selectOption('mediapipe');
        await expect(page.getByText('(dog)')).toBeVisible();
        // Change actual layout, never getBoundingClientRect or CSS object-fit.
        // Scroll an expanded document to verify viewport coordinates at input.
        await page.evaluate(({box})=>{
          for(const e of [document.documentElement,document.body,document.querySelector('#root')])Object.assign(e.style,{overflow:'visible',height:'2000px'});
          const parent=document.querySelector('video').parentElement;
          Object.assign(parent.style,{position:'absolute',left:'37.25px',top:'482.5px',width:`${box[0]}px`,height:`${box[1]}px`,maxHeight:'none',aspectRatio:'auto',border:'0px',borderRadius:'0px'});
          window.scrollTo(0,400);
        },c);
        await page.waitForTimeout(100);
        const geometry=await page.evaluate(()=>{
          const v=document.querySelector('video'),c=document.querySelector('canvas');
          const rect=e=>{const r=e.getBoundingClientRect(),s=getComputedStyle(e);return {left:r.left,top:r.top,width:r.width,height:r.height,fit:s.objectFit,position:s.objectPosition,transform:s.transform,padding:s.padding,border:s.borderWidth};};
          const sample=document.createElement('canvas');sample.width=v.videoWidth;sample.height=v.videoHeight;const ctx=sample.getContext('2d');ctx.drawImage(v,0,0);
          return {video:rect(v),canvas:rect(c),backing:[c.width,c.height],intrinsic:[v.videoWidth,v.videoHeight],scroll:[scrollX,scrollY],dpr:devicePixelRatio,pixels:{background:[...ctx.getImageData(10,10,1,1).data],object:[...ctx.getImageData(100,100,1,1).data]},creates:window.visionSdkProbe.creates};
        });
        const [insetX,insetY,cw,ch]=c.content;
        const sx=cw/c.video[0],sy=ch/c.video[1];
        const origin=[37.25+insetX,82.5+insetY];
        const point={x:origin[0]+180*sx,y:origin[1]+130*sy};
        const margins=[];
        if(insetX>0)margins.push([origin[0]-2,origin[1]+100*sy],[origin[0]+cw+2,origin[1]+100*sy]);
        if(insetY>0)margins.push([origin[0]+100*sx,origin[1]-2],[origin[0]+100*sx,origin[1]+ch+2]);
        if(!skipMargins)for(const [x,y] of margins)await page.mouse.click(x,y);
        await page.waitForTimeout(80);
        const observation=()=>page.evaluate(()=>({speech:[...window.independentProbe.speech],arcs:[...window.independentProbe.arcs],boxes:[...window.independentProbe.boxes],requests:window.independentProbe.requests,creates:window.visionSdkProbe.creates,selected:document.querySelector('.group.border-blue-500\\/80')!==null}));
        const before=await observation();
        await page.screenshot({path:path.join(output,`${c.name}-${dpr}-before.png`),scale:'css'});
        await page.evaluate(()=>{window.independentProbe.arcs=[];window.independentProbe.speech=[];});
        await page.mouse.click(point.x,point.y);await page.waitForTimeout(80);
        const after=await observation();
        await page.screenshot({path:path.join(output,`${c.name}-${dpr}-after.png`),scale:'css'});
        result={...result,geometry,origin,scale:[sx,sy],point,margins,testedMargins:!skipMargins,before,after,errors};
        const checks={css:geometry.video.fit==='contain'&&geometry.canvas.fit==='contain'&&geometry.video.position==='50% 50%'&&geometry.canvas.position==='50% 50%'&&geometry.video.transform==='none',sameRect:JSON.stringify(geometry.video)===JSON.stringify(geometry.canvas),backing:JSON.stringify(geometry.backing)===JSON.stringify(c.video),scrolled:geometry.scroll[1]===400,margin:before.speech.length===0&&before.arcs.length===0&&!before.selected,selected:after.selected&&after.speech.length===1,ripple:after.arcs.length>0&&Math.abs(after.arcs.at(-1)[0]-180)<=1/sx&&Math.abs(after.arcs.at(-1)[1]-130)<=1/sy,box:after.boxes.some(b=>Math.abs(b[0]-64)<1e-9&&Math.abs(b[1]-48)<1e-9&&Math.abs(b[2]-128)<1e-9&&Math.abs(b[3]-96)<1e-9),lifecycle:after.requests===1&&after.creates===geometry.creates,noError:errors.length===0};
        // Immediate CSS resize and a fractional input in the same JS task.
        // MouseEvent's long-valued client coordinates are deliberately integral.
        if(!baseline){
          const immediate=await page.evaluate(()=>{
            const v=document.querySelector('video'),c=document.querySelector('canvas');
            Object.assign(v.parentElement.style,{width:'960px',height:'540px'});
            const width=v.videoWidth,height=v.videoHeight;
            // Fixed independent points for each actual native source.
            const points=width===640?[359,228]:width===1280?[172,180]:[409,199];
            window.independentProbe.speech=[];window.independentProbe.arcs=[];
            c.dispatchEvent(new MouseEvent('click',{bubbles:true,clientX:points[0],clientY:points[1]}));
            return {width,height,speech:window.independentProbe.speech.length,rect:[v.getBoundingClientRect().width,v.getBoundingClientRect().height]};
          });
          result.immediate=immediate;checks.immediate=immediate.speech===1;
          // Unready, zero and invalid geometry must be rejected before speech.
          const invalid=await page.evaluate(()=>{
            const v=document.querySelector('video'),c=document.querySelector('canvas');const observations=[];
            const click=()=>c.dispatchEvent(new MouseEvent('click',{bubbles:true,clientX:359,clientY:228}));
            for(const [property,value] of [['readyState',0],['videoWidth',0],['videoHeight',0],['videoWidth',NaN],['videoHeight',Infinity]]){
              Object.defineProperty(v,property,{configurable:true,value});window.independentProbe.speech=[];click();observations.push({property,value:String(value),speech:window.independentProbe.speech.length});delete v[property];
            }
            Object.assign(v.parentElement.style,{width:'0px',height:'0px'});window.independentProbe.speech=[];click();observations.push({property:'CSS size',value:'0',speech:window.independentProbe.speech.length});
            return observations;
          });
          result.invalid=invalid;checks.invalid=invalid.every(x=>x.speech===0);
        }
        result.checks=checks;result.pass=Object.values(checks).every(Boolean);
      }catch(error){result.error=String(error);result.pass=false;}
      finally {await context.close();}
      results.push(result);writeFileSync(path.join(output,'results.json'),JSON.stringify(results,null,2)+'\n');
      console.log(c.name,dpr,result.pass,result.checks??result.error);
    }
  }
} finally {await browser.close();}
if(results.some(r=>!r.pass))process.exitCode=1;
