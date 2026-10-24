#!/usr/bin/env python3
"""Mutate disposable archived copies only; original source and tests stay intact."""
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / 'logs'
SHA = 'f20474be156522ee6141cde9c2b3a958db8d2457'
ENV = dict(os.environ, npm_config_cache='/tmp/pr01-review-npm-cache', NO_COLOR='1',
           PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH='/usr/bin/chromium')

def run(name, command, cwd, expected, stdin=None, contains=None):
    result = subprocess.run(command, cwd=cwd, env=ENV, input=stdin, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    (OUT / (name + '.txt')).write_text(f'Command: {command!r}\nExit: {result.returncode}\nExpected: {expected}\n\n{result.stdout}')
    ok = (result.returncode == 0 if expected == 'success' else result.returncode != 0)
    ok = ok and (contains is None or contains in result.stdout)
    print(name, result.returncode, 'as expected' if ok else 'UNEXPECTED', flush=True)
    return {'name': name, 'command': command, 'exit': result.returncode, 'expected': expected, 'control_passed': ok}

if __name__ == '__main__':
    results = []
    with tempfile.TemporaryDirectory(prefix='pr01-review-control-') as directory:
        probe = Path(directory)
        archive = subprocess.check_output(['git', 'archive', SHA], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(probe, filter='data')
        (probe / 'node_modules').symlink_to(ROOT / 'node_modules', target_is_directory=True)
        eslint = str(ROOT / 'node_modules/.bin/eslint')
        results.append(run('negative-eslint-typescript', [eslint, '--stdin', '--stdin-filename', 'src/review-negative.ts'], probe, 'failure',
                           'const unused: any = 1;\n', '@typescript-eslint/no-explicit-any'))
        results.append(run('negative-eslint-hooks', [eslint, '--stdin', '--stdin-filename', 'src/review-negative.tsx'], probe, 'failure',
                           "import { useEffect } from 'react';\nexport function Probe({value, flag}: {value: string; flag: boolean}) {\nif (flag) useEffect(() => { console.log(value); }, []);\nreturn null;\n}\n", 'react-hooks/rules-of-hooks'))
        canvas = (probe / 'src/components/VisionCanvas.tsx').read_text()
        results.append(run('exception-without-suppression', [eslint, '--stdin', '--stdin-filename', 'src/components/VisionCanvas.tsx'], probe, 'failure',
                           canvas.replace('    // eslint-disable-next-line react-hooks/exhaustive-deps\n', ''), "'drawOverlays'"))
        # Add a new omission to the exempt effect: document this line's blind spot.
        results.append(run('exception-new-omission', [eslint, '--stdin', '--stdin-filename', 'src/components/VisionCanvas.tsx'], probe, 'success',
                           canvas.replace('      if (!isRunning) return;', '      console.log(facingMode);\n      if (!isRunning) return;', 1)))
        bad_type = probe / 'src/review-negative.ts'
        bad_type.write_text('export const reviewNumber: number = "wrong";\n')
        results.append(run('negative-typecheck', ['npm', 'run', 'typecheck'], probe, 'failure', contains='TS2322'))
        bad_type.unlink()
        mutations = [
            ('negative-dictionary-known', 'src/services/dictionary.ts', "dog: { ja: '犬'", "dog: { ja: 'BROKEN'", 'src/services/dictionary.test.ts'),
            ('negative-dictionary-unknown', 'src/services/dictionary.ts', 'ja: fallbackJa,', "ja: 'BROKEN',", 'src/services/dictionary.test.ts'),
            ('negative-color', 'src/services/dictionary.ts', 'return COLOR_PALETTE[index];', "return label + index;", 'src/services/dictionary.test.ts'),
            ('negative-tts', 'src/services/speechService.ts', 'if (!this.synth || !config.enabled) return false;', 'if (!this.synth || !config.enabled) return true;', 'src/services/speechService.test.ts'),
            ('negative-app-ui', 'src/components/Header.tsx', 'VisionStudio AI', 'BROKEN HEADER', 'src/App.test.tsx'),
        ]
        for name, path, old, new, test in mutations:
            target = probe / path
            original = target.read_text()
            assert old in original, name
            target.write_text(original.replace(old, new, 1))
            results.append(run(name, ['npm', 'test', '--', test], probe, 'failure', contains='AssertionError' if name != 'negative-app-ui' else 'Unable to find'))
            target.write_text(original)
        shutil.copytree(Path(os.environ.get('PR01_REVIEW_DIST', ROOT / 'dist')), probe / 'dist')
        html = probe / 'dist/index.html'
        original = html.read_text()
        # Real built modules still run: only add an uncaught exception to the page.
        html.write_text(original.replace('</head>', '<script>setTimeout(() => { throw new Error("review fatal negative control"); }, 0);</script></head>'))
        results.append(run('negative-browser-fatal', ['npm', 'run', 'test:e2e'], probe, 'failure', contains='review fatal negative control'))
        # Replace the app's actual stylesheet with a missing URL. Vite's SPA
        # fallback returns HTML 200; the existing smoke should reject lost CSS.
        broken_css, changed = re.subn(r'href="/assets/[^" ]+\.css"', 'href="/assets/review-missing.css"', original)
        assert changed == 1
        html.write_text(broken_css)
        results.append(run('negative-browser-asset', ['npm', 'run', 'test:e2e'], probe, 'failure'))
    (OUT / 'negative-controls.json').write_text(json.dumps(results, indent=2) + '\n')
    if not all(result['control_passed'] for result in results):
        raise SystemExit('One or more controls did not produce the expected diagnostic')
