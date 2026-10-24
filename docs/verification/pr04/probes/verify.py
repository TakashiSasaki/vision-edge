"""Required local checks; stop at the first failure. Receipts bind source hashes."""
import hashlib
import json
import os
import pathlib
import subprocess

root = pathlib.Path(__file__).resolve().parents[4]
logs = root / 'docs/verification/pr04/logs'
env = dict(os.environ, npm_config_cache='/workspace/.npm-cache', PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH='/usr/bin/chromium')
commands = [
    ['node', '--version'], ['npm', '--version'], ['npm', 'ci'],
    ['npm', 'run', 'typecheck'], ['npm', 'run', 'lint'], ['npm', 'test'],
    ['npm', 'run', 'build'], ['npm', 'run', 'test:e2e'],
    ['npm', 'run', 'test:e2e:controls'], ['git', 'diff', '--check'],
    ['git', 'diff', 'origin/main', '--check'],
]
records = []
for i, cmd in enumerate(commands):
    path = logs / f'final-{i:02d}.txt'
    with path.open('w') as out:
        result = subprocess.run(cmd, cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT)
    captured = path.read_bytes()
    if captured:
        path.write_bytes(captured.rstrip(b'\n') + b'\n')
    records.append({'command': cmd, 'exit': result.returncode, 'log': path.name})
    (logs / 'checks.json').write_text(json.dumps(records, indent=2) + '\n')
    print(f'{cmd}: exit {result.returncode}', flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=root, text=True).splitlines()
hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths if not p.startswith('docs/verification/')}
(logs / 'verified-source.json').write_text(json.dumps(hashes, indent=2) + '\n')
