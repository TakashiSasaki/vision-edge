import json, os, pathlib, subprocess, time
root = pathlib.Path(__file__).resolve().parents[4]
evidence = root / 'docs/verification/pr04-independent'
env = dict(os.environ, npm_config_cache='/workspace/pr04-npm-cache', PLAYWRIGHT_BROWSERS_PATH='/workspace/pr04-browsers')
commands = [
 ['node', '--version'], ['npm', '--version'], ['npm', 'ci'],
 ['npm', 'run', 'typecheck'], ['npm', 'run', 'lint'], ['npm', 'test'], ['npm', 'run', 'build'],
 ['npx', '--no-install', 'playwright', 'install', 'chromium'],
 ['npm', 'run', 'test:e2e'], ['npm', 'run', 'test:e2e:controls'],
 ['git', 'diff', '--check', 'f1a0d0bbe343b666c75eae053340ceefc6730d5d', '2f4cdea1bc3e6706f49e37116ede8b1881472f7f'],
]
receipts = []
for i, command in enumerate(commands):
    log = evidence / 'logs' / f'check-{i:02}.txt'
    started = time.time()
    with log.open('w') as output:
        result = subprocess.run(command, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT)
    receipts.append(dict(command=command, exit=result.returncode, seconds=time.time()-started, log=str(log.relative_to(root))))
    (evidence / 'logs/checks.json').write_text(json.dumps(receipts, indent=2)+'\n')
    print(json.dumps(receipts[-1]), flush=True)
