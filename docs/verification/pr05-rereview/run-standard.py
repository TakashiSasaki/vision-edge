import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent/'logs'
env=dict(os.environ,npm_config_cache='/workspace/npm-cache',PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH='/usr/bin/chromium')
results=[]
for i,cmd in enumerate(['node --version','npm --version','chromium --version','npm run typecheck','npm run lint','npm test','npm run build','npm run test:e2e','npm run test:e2e:controls','git diff --check','git diff --check origin/main...HEAD']):
    started=time.time();name=f'standard-{i:02}.txt'
    with (OUT/name).open('w')as output:r=subprocess.run(cmd.split(),cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
    results.append(dict(command=cmd,exit=r.returncode,seconds=time.time()-started,output=name))
    (OUT/'standard.json').write_text(json.dumps(results,indent=2)+'\n');print(cmd,r.returncode,flush=True)
