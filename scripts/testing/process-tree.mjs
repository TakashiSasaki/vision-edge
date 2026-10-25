// Linux identities prevent sending a signal to a reused descendant PID.
import { readdirSync, readFileSync } from 'node:fs';
function identity(pid) {
  try { const s=readFileSync(`/proc/${pid}/stat`,'utf8');const f=s.slice(s.lastIndexOf(')')+2).split(' ');return {pid:Number(pid),parent:Number(f[1]),group:Number(f[2]),start:f[19],state:f[0]}; } catch { return null; }
}
export function processTree(root) {
  const known=new Map();
  const rootStart=identity(root)?.start;
  function scan() {
    if(process.platform!=='linux')return;
    const rows=readdirSync('/proc').filter(p=>/^\d+$/.test(p)).map(identity).filter(Boolean);
    const parents=new Set(identity(root)?.start===rootStart&&rootStart?[root]:[]);for(const [pid,start] of known)if(identity(pid)?.start===start)parents.add(pid);
    let changed=true;while(changed){changed=false;for(const row of rows)if(parents.has(row.parent)&&!parents.has(row.pid)){parents.add(row.pid);known.set(row.pid,row.start);changed=true;}}
  }
  function signal(value) {
    scan();for(const [pid,start] of known){const row=identity(pid);if(row?.start===start&&row.state!=='Z')try{process.kill(pid,value);}catch{/* exited */}}
  }
  const timer=setInterval(scan,100);timer.unref();scan();
  return {signal,ownsGroup:()=>process.platform!=='linux'||(rootStart&&identity(root)?.start===rootStart)||[...known].some(([pid,start])=>{const r=identity(pid);return r?.start===start&&r.group===root;}),stop:()=>clearInterval(timer),remaining:()=>[...known].filter(([pid,start])=>{const r=identity(pid);return r?.start===start&&r.state!=='Z';}).map(([pid])=>pid)};
}
