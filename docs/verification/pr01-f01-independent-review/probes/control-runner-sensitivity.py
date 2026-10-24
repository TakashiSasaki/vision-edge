#!/usr/bin/env python3
"""Check Author runner rejects an insensitive smoke and wrong-failure diagnosis."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(os.environ.get('PR01_SOURCE_ROOT', Path(__file__).resolve().parents[4]))
OUT = Path(__file__).resolve().parents[1] / 'logs'
ENV = dict(os.environ, PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH='/usr/bin/chromium', npm_config_update_notifier='false')

def manifest():
    return {str(p.relative_to(ROOT / 'dist')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'dist').rglob('*') if p.is_file()}

before = manifest()
results = []
for mode, diagnostic in [('css-checks-removed', 'expected smoke assertion failure'), ('wrong-diagnostic', 'missing diagnostic required stylesheet MIME')]:
    with tempfile.TemporaryDirectory(prefix='vision-css-runner-review-') as directory:
        probe = Path(directory)
        for file in ['dist', 'e2e', 'scripts', 'package.json', 'vite.config.ts', 'playwright.config.ts']:
            if (ROOT / file).is_dir():
                shutil.copytree(ROOT / file, probe / file)
            else:
                shutil.copy2(ROOT / file, probe / file)
        (probe / 'node_modules').symlink_to(ROOT / 'node_modules', target_is_directory=True)
        smoke = probe / 'e2e/smoke.spec.ts'
        text = smoke.read_text()
        if mode == 'css-checks-removed':
            start = text.index('  // Discover the build')
            end = text.index('  // Observe the real loader', start)
            text = text[:start] + text[end:]
        else:
            assert 'required stylesheet MIME:' in text
            text = text.replace('required stylesheet MIME:', 'independent wrong diagnostic:')
        smoke.write_text(text)
        result = subprocess.run(['npm', 'run', 'test:e2e:controls'], cwd=probe, env=ENV, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (OUT / ('runner-' + mode + '.txt')).write_text(f'Command: npm run test:e2e:controls\nExit: {result.returncode}\nExpected: 1\n\n{result.stdout}')
        assert result.returncode == 1 and diagnostic in result.stdout, result.stdout
        assert 'intact: exit=0; passed=1; failed=0' in result.stdout
        assert not list(Path('/tmp').glob('vision-edge-css-controls-*')), 'Author temp directory leaked'
        results.append(dict(mutation=mode,exit=result.returncode,diagnostic=diagnostic,passed=True))
        print(results[-1], flush=True)
assert manifest() == before
(OUT / 'runner-sensitivity.json').write_text(json.dumps(dict(results=results,build_unchanged=True), indent=2) + '\n')
