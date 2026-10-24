import subprocess,sys,json,datetime,pathlib
root=pathlib.Path(__file__).resolve().parent
name=sys.argv[1];args=sys.argv[2:];start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (root/'logs'/f'{name}.txt').open('w') as log:
 p=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
row=dict(name=name,argv=args,cwd=str(pathlib.Path.cwd()),start=start,end=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit=p.returncode)
with (root/'commands.jsonl').open('a') as ledger:ledger.write(json.dumps(row)+'\n')
print(json.dumps(row));print((root/'logs'/f'{name}.txt').read_text()[-1600:]);sys.exit(p.returncode)
