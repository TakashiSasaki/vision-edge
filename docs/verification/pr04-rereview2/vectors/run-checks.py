import json,os,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parents[4];logs=root/'docs/verification/pr04-rereview2/logs'
env=dict(os.environ,npm_config_cache='/workspace/pr04-rereview2-npm-cache',PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH='/usr/bin/chromium')
commands=[
 ['node','--version'],['npm','--version'],['npm','ci'],['npm','run','typecheck'],['npm','run','lint'],['npm','test'],['npm','run','build'],
 ['npm','run','test:e2e'],['npm','run','test:e2e:controls'],
 ['git','diff','--check','f1a0d0bbe343b666c75eae053340ceefc6730d5d','d92b7b40c7d2dc11572147d052f69f11657540e2'],
 ['git','diff','--check','f81ce6102ec4f459ef7a0c3fb7e55d126b157be5','d92b7b40c7d2dc11572147d052f69f11657540e2'],
 ['npm','--prefix','/workspace/pr04-baseline','test']]
records=[]
for i,command in enumerate(commands):
 start=time.time();log=logs/f'check-{i:02}.txt'
 with log.open('w') as output:r=subprocess.run(command,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT)
 record=dict(command=command,exit=r.returncode,seconds=time.time()-start,log=log.name);records.append(record)
 (logs/'checks.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(record),flush=True)
