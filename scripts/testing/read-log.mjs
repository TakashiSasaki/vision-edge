// Small local view: never dump the whole saved report into an agent by default.
import { readFile, stat } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';
import { excerpt, readTail } from './capture.mjs';
const [folder, task, pattern] = process.argv.slice(2);
try {
  if (!folder || !task || !/^[a-z][a-z0-9-]*$/.test(task)) throw new Error('Usage: npm run test:logs -- <run-directory> <task> [literal-search]');
  const dir=path.resolve(folder,task);const receipt=JSON.parse(await readFile(path.join(dir,'result.json'),'utf8'));
  console.log(JSON.stringify({task:receipt.task,state:receipt.state,exit:receipt.exit,counts:receipt.counts,logBytes:receipt.logBytes,logSha256:receipt.logSha256}));
  const file=path.join(dir,'output.log');
  if (!pattern) console.log(excerpt(await readTail(file),4000));
  else {
    const lines=createInterface({input:createReadStream(file),crlfDelay:Infinity});let output='';let count=0;let lineNo=0;
    for await(const line of lines){lineNo++;if(line.includes(pattern)){output+=`${lineNo}: ${line.slice(0,1500)}\n`;count++;if(count===10||Buffer.byteLength(output)>4000){lines.close();break;}}}
    console.log(excerpt(output||'No match',4500));
  }
  console.log(`full log: ${file} (${(await stat(file)).size} bytes)`);
} catch(error){console.error(String(error));process.exitCode=1;}
