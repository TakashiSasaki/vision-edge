"""Run original scripts unchanged; preserve output and exit codes."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / 'logs'
env = dict(os.environ, npm_config_cache='/workspace/npm-cache', PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH='/usr/bin/chromium')
commands = ['node --version', 'npm --version', 'chromium --version', 'npm run typecheck', 'npm run lint', 'npm test', 'npm run build', 'npm run test:e2e', 'npm run test:e2e:controls', 'git diff --check', 'git diff --check origin/main...HEAD']
results = []
for i, command in enumerate(commands):
    start = time.time()
    filename = f'standard-{i:02}.txt'
    with (OUT / filename).open('w') as output:
        result = subprocess.run(command.split(), cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    results.append(dict(command=command, exit=result.returncode, seconds=time.time()-start, output=filename))
    (OUT / 'standard.json').write_text(json.dumps(results, indent=2)+'\n')
    print(command, result.returncode, flush=True)
