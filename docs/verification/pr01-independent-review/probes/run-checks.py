#!/usr/bin/env python3
"""Run read-only foundation checks; write commands, exit codes and output."""
import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / 'logs'
OUT.mkdir(exist_ok=True)
ENV = dict(os.environ, npm_config_cache='/tmp/pr01-review-npm-cache', NO_COLOR='1')
COMMANDS = [
    ('versions', ['bash', '-c', 'node --version && npm --version && uname -srm && chromium --version']),
    ('install', ['npm', 'ci']),
    ('dependency-tree', ['npm', 'ls', '--all']),
    ('typecheck', ['npm', 'run', 'typecheck']),
    ('lint', ['npm', 'run', 'lint']),
    ('unit', ['npm', 'test']),
    ('build', ['npm', 'run', 'build']),
    ('diff-check', ['git', 'diff', '--check']),
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--prefix', default='')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    ROOT = args.source_root.resolve()
    if args.smoke:
        ENV['PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH'] = '/usr/bin/chromium'
        COMMANDS.append(('smoke', ['npm', 'run', 'test:e2e']))
    results = []
    lock_before = hashlib.sha256((ROOT / 'package-lock.json').read_bytes()).hexdigest()
    for name, command in COMMANDS:
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result = subprocess.run(command, cwd=ROOT, env=ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        (OUT / (args.prefix + name + '.txt')).write_text(f'UTC: {started}\nCommand: {command!r}\nExit: {result.returncode}\n\n{result.stdout}')
        results.append({'name': name, 'command': command, 'exit': result.returncode, 'started_utc': started})
        print(name, result.returncode, flush=True)
    results.append({'lock_sha256_before': lock_before, 'lock_sha256_after': hashlib.sha256((ROOT / 'package-lock.json').read_bytes()).hexdigest()})
    results.append({'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True), 'head_and_tree': subprocess.check_output(['git', 'show', '-s', '--format=%H %T', 'HEAD'], cwd=ROOT, text=True).strip()})
    (OUT / (args.prefix + 'checks.json')).write_text(json.dumps(results, indent=2) + '\n')
