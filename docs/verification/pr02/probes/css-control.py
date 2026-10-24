"""Author sensitivity control: real production smoke, temporary missing CSS only."""
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / 'logs'

class Stylesheets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'link' and 'stylesheet' in attrs.get('rel', '').split() and attrs.get('href', '').startswith('/'):
            self.hrefs.append(attrs['href'])

def manifest():
    return {str(p.relative_to(ROOT / 'dist')): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'dist').rglob('*') if p.is_file()}

before = manifest()
with tempfile.TemporaryDirectory(prefix='pr02-css-control-') as directory:
    probe = Path(directory)
    for name in ['dist', 'e2e', 'package.json', 'playwright.config.ts', 'vite.config.ts']:
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, probe / name)
        else:
            shutil.copy2(source, probe / name)
    (probe / 'node_modules').symlink_to(ROOT / 'node_modules', target_is_directory=True)
    html_path = probe / 'dist/index.html'
    html = html_path.read_text()
    parser = Stylesheets()
    parser.feed(html)
    assert parser.hrefs
    for index, href in enumerate(parser.hrefs):
        html = html.replace(href, f'/assets/pr02-missing-{index}.css')
    html_path.write_text(html)
    env = dict(os.environ, PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH='/usr/bin/chromium', npm_config_update_notifier='false')
    result = subprocess.run(['npm', 'run', 'test:e2e', '--', '--reporter=json'], cwd=probe, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    report = json.loads(result.stdout[result.stdout.index('{'):])
    errors = [error.get('message', '') for suite in report['suites'] for spec in suite['specs'] for test in spec['tests'] for run in test['results'] for error in run['errors']]
    text = f'Exit: {result.returncode}\nExpected: 1 / required stylesheet MIME\n\n{result.stdout}'
    (OUT / 'missing-css.txt').write_text('\n'.join(line.rstrip() for line in text.splitlines()).rstrip() + '\n')
    assert result.returncode == 1 and report['stats']['unexpected'] == 1
    assert any('required stylesheet MIME' in error for error in errors)
assert before == manifest()
(OUT / 'css-control.json').write_text(json.dumps(dict(exit=1, expected_exit=1, diagnostic='required stylesheet MIME', passed=True, original_build_unchanged=True), indent=2) + '\n')
print('Missing CSS: expected exit 1 at CSS MIME; control passed; production build unchanged')
