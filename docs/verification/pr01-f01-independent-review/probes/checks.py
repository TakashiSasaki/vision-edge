#!/usr/bin/env python3
"""Run exact-source checks; preserve each actual exit without pipeline masking."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(os.environ.get('PR01_SOURCE_ROOT', Path(__file__).resolve().parents[4]))
OUT = Path(__file__).resolve().parents[1] / 'logs'
ENV = dict(os.environ, npm_config_cache='/tmp/vision-edge-review-npm-cache',
           npm_config_update_notifier='false', PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH='/usr/bin/chromium')
commands = [
    ['node', '--version'], ['npm', '--version'], ['chromium', '--version'],
    ['npx', '--no-install', 'playwright', '--version'], ['npm', 'ls', '--all'],
    ['npm', 'run', 'typecheck'], ['npm', 'run', 'lint'], ['npm', 'test'],
    ['npm', 'run', 'build'], ['npm', 'run', 'test:e2e'], ['npm', 'run', 'test:e2e:controls'],
    ['npm', 'run', 'test:e2e'], ['git', 'diff', '--check'],
]
results = []
for index, command in enumerate(commands):
    result = subprocess.run(command, cwd=ROOT, env=ENV, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    filename = f'check-{index:02d}.txt'
    (OUT / filename).write_text(f'Command: {command!r}\nExit: {result.returncode}\n\n{result.stdout}')
    results.append(dict(command=command, exit=result.returncode, log=filename))
    (OUT / 'checks.json').write_text(json.dumps(results, indent=2) + '\n')
    print(command, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
