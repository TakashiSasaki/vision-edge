#!/usr/bin/env python3
"""Baseline resolver, lock/ignore/secret heuristics, dev and clean checks."""
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / 'logs'
ENV = dict(os.environ, npm_config_cache='/tmp/pr01-review-npm-cache', NO_COLOR='1')

if __name__ == '__main__':
    results = {}
    with tempfile.TemporaryDirectory(prefix='pr01-review-baseline-') as directory:
        baseline = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git', 'archive', '90323c00a6cd9a8157bf3eec853dbe5a857f07a1'], cwd=ROOT))) as tar:
            tar.extractall(baseline, filter='data')
        command = ['npm', 'install', '--package-lock-only', '--ignore-scripts']
        result = subprocess.run(command, cwd=baseline, env=ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (OUT / 'independent-baseline.txt').write_text(f'Command: {command!r}\nExit: {result.returncode}\n\n{result.stdout}')
        results['baseline_resolver_exit'] = result.returncode
        results['baseline_ERESOLVE_observed'] = 'ERESOLVE' in result.stdout
    results['npm_settings'] = {key: subprocess.check_output(['npm', 'config', 'get', key], cwd=ROOT, env=ENV, text=True).strip() for key in ['force', 'legacy-peer-deps', 'ignore-scripts']}
    lock = json.loads((ROOT / 'package-lock.json').read_text())
    entries = lock['packages']
    results['lock'] = {'version': lock['lockfileVersion'], 'entries': len(entries),
                       'non_public_resolved_paths': [p for p, v in entries.items() if v.get('resolved') and not v['resolved'].startswith('https://registry.npmjs.org/')],
                       'missing_integrity_paths': [p for p, v in entries.items() if p and 'integrity' not in v]}
    results['installed_metadata'] = {}
    for name in ['vite', 'tsx', 'esbuild', 'typescript', 'typescript-eslint', '@typescript-eslint/typescript-estree', 'eslint', 'eslint-plugin-react-hooks', '@vitejs/plugin-react', 'vitest', 'jsdom', '@testing-library/react', '@mediapipe/tasks-vision']:
        data = json.loads((ROOT / 'node_modules' / name / 'package.json').read_text())
        results['installed_metadata'][name] = {k: data[k] for k in ['version', 'engines', 'peerDependencies'] if k in data}
    results['ignore'] = {}
    for path in ['node_modules/x', 'dist/x', 'server.js', 'coverage/x', 'test-results/x', 'playwright-report/x', 'blob-report/x', '.env', '.env.local', '.env.production', 'private.local', '.npm/x', '.cache/x', '.aistudio/x', '.vscode/x', '.env.example']:
        result = subprocess.run(['git', 'check-ignore', '--no-index', path], cwd=ROOT, text=True, stdout=subprocess.PIPE)
        results['ignore'][path] = result.returncode == 0
    # Output paths/counts only, never credential contents. This is heuristic, not certification.
    patterns = [rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', rb'gh[pousr]_[A-Za-z0-9]{20,}', rb'github_pat_[A-Za-z0-9_]{20,}', rb'AKIA[A-Z0-9]{16}', rb'AIza[0-9A-Za-z_-]{30,}']
    matches = []
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for path in filter(None, paths):
        data = (ROOT / path).read_bytes()
        if b'\0' in data:
            continue
        for index, pattern in enumerate(patterns):
            if re.search(pattern, data):
                matches.append({'path': path, 'pattern_index': index})
    results['credential_heuristic_matches'] = matches
    results['tracked_ignored_inherited_paths'] = subprocess.check_output(['git', 'ls-files', '-ci', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
    server = subprocess.Popen(['npm', 'run', 'dev', '--', '--host', '127.0.0.1', '--strictPort'], cwd=ROOT, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen('http://127.0.0.1:3000', timeout=1) as response:
                    html = response.read().decode()
                    results['dev'] = {'status': response.status, 'source_entrypoint': '/src/main.tsx' in html}
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError('Dev server did not start')
    finally:
        import signal
        os.killpg(server.pid, signal.SIGTERM)
        output, _ = server.communicate(timeout=10)
        (OUT / 'dev.txt').write_text(output)
    # Build artifacts may be removed here; run this after smoke/negative controls.
    (ROOT / 'server.js').write_text('// disposable clean probe\n')
    results['clean'] = []
    for _ in range(2):
        result = subprocess.run(['npm', 'run', 'clean'], cwd=ROOT, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        results['clean'].append({'exit': result.returncode, 'dist_exists': (ROOT / 'dist').exists(), 'server_js_exists': (ROOT / 'server.js').exists()})
    (OUT / 'supplemental.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))
