"""Ordinary checks with recorded exits; local browser override is explicit."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / 'logs'
ENV = dict(os.environ, npm_config_cache='/tmp/vision-edge-review-npm-cache', npm_config_update_notifier='false', PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH='/usr/bin/chromium')
BASE = '538d25157a5d87feac2da4496e354cbf8b9bdf5f'
commands = [
    ['node', '--version'], ['npm', '--version'], ['npm', 'ci'],
    ['npx', '--no-install', 'playwright', '--version'], ['chromium', '--version'],
    ['npm', 'run', 'typecheck'], ['npm', 'run', 'lint'], ['npm', 'test'],
    ['npm', 'run', 'build'], ['npm', 'run', 'test:e2e'], ['git', 'diff', '--check'],
    ['git', 'diff', '--exit-code', BASE, '--', 'package.json', 'package-lock.json', '.nvmrc',
     '.github/workflows/ci.yml', 'playwright.config.ts', 'vitest.config.ts', 'eslint.config.js',
     'src/services/modelRegistry.ts', 'src/types/vision.ts', 'src/App.test.tsx',
     'src/services/visionEngine.test.ts', 'src/services/dictionary.ts', 'src/services/speechService.ts',
     'e2e/smoke.spec.ts', 'docs/verification/pr02.md', 'docs/verification/pr02'],
]
results = []
for index, command in enumerate(commands):
    result = subprocess.run(command, cwd=ROOT, env=ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    filename = f'final-{index:02d}.txt'
    log = f'Command: {command!r}\nExit: {result.returncode}\n\n{result.stdout}'
    (OUT / filename).write_text('\n'.join(line.rstrip() for line in log.splitlines()).rstrip() + '\n')
    results.append(dict(command=command, exit=result.returncode, log=filename))
    (OUT / 'checks.json').write_text(json.dumps(results, indent=2) + '\n')
    print(command, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
