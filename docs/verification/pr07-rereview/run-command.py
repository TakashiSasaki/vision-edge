import sys, os, json, subprocess, datetime, pathlib, hashlib
out=pathlib.Path(__file__).parent
name,cwd,*argv=sys.argv[1:]
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (out/'logs'/f'{name}.txt').open('wb') as f:
 try:
  p=subprocess.run(argv,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,timeout=780)
  code=p.returncode
 except subprocess.TimeoutExpired:
  code=124
record={'name':name,'cwd':cwd,'argv':argv,'start':start,'end':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit':code,'log':f'logs/{name}.txt','logSha256':hashlib.sha256((out/'logs'/f'{name}.txt').read_bytes()).hexdigest(),'browserOverride':os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH')}
with (out/'commands.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
print(json.dumps(record))
