import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFile } from 'node:fs/promises';
import { execute, newRun } from './capture.mjs';
import { tapCounts, unitCounts, browserCounts, controlCounts, unitFailures, browserFailures, tapFailures } from './results.mjs';
const root = fileURLToPath(new URL('../../', import.meta.url));
const [name, ...extra] = process.argv.slice(2);
const offline = ['typecheck','lint','unit','foundation','output','build','browser','controls'];
const names = new Set([...offline,'install','browser-install','real-sdk','assets','acquire','audit']);
if ((name !== 'offline' && !names.has(name)) || (name === 'offline' && extra.length)) { console.error(`Usage: node scripts/testing/run.mjs <${[...names,'offline'].join('|')}> [test arguments]`); process.exitCode=1; }
else {
  const runDir = await newRun(root);
  for (const phase of name === 'offline' ? offline : [name]) {
    const dir=path.join(runDir,phase); const node=(args, rest={})=>({name:phase,command:process.execPath,args,selection:extra.length?'custom':'default',...rest});
    const assetRows = async () => { const r=JSON.parse(await readFile(path.join(dir,'output.log'),'utf8')); if(!Array.isArray(r.rows)||!r.rows.length||r.rows.some(x=>!['pass','fail','blocked'].includes(x.status)))throw new Error('Invalid asset report');return r.rows; };
    const assetTask = args => node(args,{blockedCodes:[2],summarize:async()=>{const rows=await assetRows();return {pass:rows.filter(x=>x.status==='pass').length,fail:rows.filter(x=>x.status==='fail').length,blocked:rows.filter(x=>x.status==='blocked').length};},diagnose:async()=> (await assetRows()).filter(x=>x.status!=='pass').map(x=>`${x.id}: ${x.status}; ${x.reason}`).join('\n')});
    const testFiles=phase==='output'?['scripts/testing/runner.test.mjs','scripts/testing/corrections.test.mjs']:['scripts/real-sdk/assets.test.mjs','scripts/real-sdk/corrections.test.mjs'];
    const tasks={
      typecheck:()=>node(['node_modules/typescript/bin/tsc','--noEmit']),
      lint:()=>node(['node_modules/eslint/bin/eslint.js','.','--max-warnings=0']),
      unit:()=>node(['node_modules/vitest/vitest.mjs','run','--reporter=json',`--outputFile=${path.join(dir,'unit.json')}`,...extra],{summarize:unitCounts,diagnose:unitFailures}),
      foundation:()=>node(['--test','--test-reporter=tap',...extra,...testFiles],{summarize:tapCounts,diagnose:tapFailures}),
      output:()=>node(['--test','--test-reporter=tap',...extra,...testFiles],{summarize:tapCounts,diagnose:tapFailures}),
      build:()=>node(['node_modules/vite/bin/vite.js','build']),
      browser:()=>node(['node_modules/@playwright/test/cli.js','test',...extra],{env:{TEST_LOG_DIR:dir},summarize:browserCounts,diagnose:browserFailures}),
      controls:()=>node(['scripts/smoke-css-controls.mjs'],{env:{TEST_LOG_DIR:dir},summarize:controlCounts}),
      install:()=>{if(!process.env.npm_execpath)throw new Error('Run install through npm run ci:install');return node([process.env.npm_execpath,'ci']);},
      'browser-install':()=>node(['node_modules/@playwright/test/cli.js','install','--with-deps','chromium']),
      'real-sdk':()=>node(['scripts/real-sdk/run.mjs'],{blockedCodes:[2]}),
      assets:()=>assetTask(['scripts/real-sdk/assets.mjs']),
      acquire:()=>assetTask(['--use-env-proxy','scripts/real-sdk/assets.mjs','--acquire']),
      audit:()=>node(['scripts/real-sdk/audit.mjs']),
    };
    try { const result=await execute(tasks[phase](),{root,runDir}); if(result.exit!==0){process.exitCode=result.exit;break;} }
    catch(error) { console.error(`runner: FAIL ${String(error)}; artifacts: ${runDir}`);process.exitCode=1;break; }
  }
}
