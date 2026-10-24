import json, os, pathlib, subprocess, time
root = pathlib.Path(__file__).resolve().parents[4]
logs = root / 'docs/verification/pr04-rereview/logs'
env = dict(os.environ, npm_config_cache='/workspace/pr04-rereview-npm-cache', PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH='/usr/bin/chromium')
commands = [
 ['node','--version'], ['npm','--version'], ['npm','ci'],
 ['npm','run','typecheck'], ['npm','run','lint'], ['npm','test'], ['npm','run','build'],
 ['npm','run','test:e2e'], ['npm','run','test:e2e:controls'],
 ['git','diff','--check','f1a0d0bbe343b666c75eae053340ceefc6730d5d','f81ce6102ec4f459ef7a0c3fb7e55d126b157be5'],
 ['git','diff','--check','2f4cdea1bc3e6706f49e37116ede8b1881472f7f','f81ce6102ec4f459ef7a0c3fb7e55d126b157be5']
]
receipts = []
for i, command in enumerate(commands):
    log=logs/f'check-{i:02}.txt'; started=time.time()
    with log.open('w') as output:
        result=subprocess.run(command,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT)
    receipt=dict(command=command,exit=result.returncode,seconds=time.time()-started,log=log.name)
    receipts.append(receipt); (logs/'checks.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
