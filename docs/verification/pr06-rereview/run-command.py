import subprocess,sys,json,datetime,pathlib
root=pathlib.Path(__file__).resolve().parent
name=sys.argv[1]; cmd=sys.argv[2:]; start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (root/'logs'/f'{name}.txt').open('w') as f:
 p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
row={'name':name,'argv':cmd,'cwd':str(pathlib.Path.cwd()),'start':start,'end':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit':p.returncode}
with (root/'commands.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row)); print((root/'logs'/f'{name}.txt').read_text()[-3500:]);sys.exit(p.returncode)
