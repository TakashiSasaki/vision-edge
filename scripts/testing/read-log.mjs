// Small local view: never dump the whole saved report into an agent by default.
import { readFile, stat } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';
import { excerpt, readTail } from './capture.mjs';
const [folder, second, third] = process.argv.slice(2);
try {
  if (!folder) throw new Error('Usage: npm run test:logs -- <printed-artifact-directory> [literal-search]');
  let dir=path.resolve(folder); let pattern=second; let direct=false;
  try{direct=(await stat(path.join(dir,'result.json'))).isFile();}catch(error){if(error.code!=='ENOENT')throw error;}
  if(!direct){if(!second||!/^[a-z][a-z0-9-]*$/.test(second))throw new Error('Expected a task directory or run directory plus task');dir=path.join(dir,second);pattern=third;}
  else if(third)throw new Error('Use the printed task directory and one optional literal search');
  const receipt=JSON.parse(await readFile(path.join(dir,'result.json'),'utf8'));
  console.log(JSON.stringify({task:receipt.task,state:receipt.state,exit:receipt.exit,selection:receipt.selection,counts:receipt.counts,logBytes:receipt.logBytes,logSha256:receipt.logSha256}));
  const file=path.join(dir,'output.log');
  const diagnostic=path.join(dir,'diagnostic.txt');let hasDiagnostic=false;try{hasDiagnostic=(await stat(diagnostic)).isFile();}catch(error){if(error.code!=='ENOENT')throw error;}
  if (!pattern) console.log(excerpt(await readTail(hasDiagnostic?diagnostic:file),4000));
  else {
    let output='';let count=0;
    for(const selected of hasDiagnostic?[diagnostic,file]:[file]){
      const lines=createInterface({input:createReadStream(selected),crlfDelay:Infinity});let lineNo=0;
      for await(const line of lines){lineNo++;if(line.includes(pattern)){output+=`${path.basename(selected)}:${lineNo}: ${line.slice(0,1500)}\n`;count++;if(count===10||Buffer.byteLength(output)>4000){lines.close();break;}}}
      if(count===10||Buffer.byteLength(output)>4000)break;
    }
    console.log(excerpt(output||'No match',4500));
  }
  console.log(`full log: ${file} (${(await stat(file)).size} bytes)`);
} catch(error){console.error(String(error));process.exitCode=1;}
