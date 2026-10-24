#!/usr/bin/env python3
"""Independent build mutations + direct browser observations + frozen smoke exits.

No Author control implementation is imported. All mutations stay in temporary
archives/copies. The browser observer separately measures HTTP/CSSOM/computed
styles and real app execution before the unchanged smoke runs in a fresh browser.
"""
import hashlib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request

ROOT = Path(os.environ.get('PR01_SOURCE_ROOT', Path(__file__).resolve().parents[4]))
OUT = Path(__file__).resolve().parents[1] / 'logs'
ORIGINAL = 'f20474be156522ee6141cde9c2b3a958db8d2457'
FIX = 'accfb1e895f16b4aec32db5dd5025a1bf2782898'
ENV = dict(os.environ, npm_config_cache='/tmp/vision-edge-review-npm-cache',
           npm_config_update_notifier='false', PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH='/usr/bin/chromium', NO_COLOR='1')

OBSERVER = r'''
import { chromium } from '@playwright/test';
const browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
try {
  const context = await browser.newContext({ serviceWorkers: 'block' });
  const page = await context.newPage();
  const responses = [], external = [], errors = [];
  page.on('response', r => responses.push({url:r.url(), status:r.status(), mime:r.headers()['content-type'], resource:r.request().resourceType()}));
  page.on('pageerror', e => errors.push(e.message));
  await context.route('**/*', route => {
    const url = new URL(route.request().url());
    if (url.origin === 'http://127.0.0.1:4173') return route.continue();
    external.push(url.href);
    return route.abort('blockedbyclient');
  });
  await context.addInitScript(() => Object.defineProperty(navigator, 'mediaDevices', {configurable:true,value:{getUserMedia:()=>Promise.reject(new DOMException('Independent observer denied camera','NotAllowedError'))}}));
  await page.goto('http://127.0.0.1:4173/');
  await page.getByRole('heading', {name:'カメラエラー'}).waitFor();
  await page.waitForFunction(() => document.querySelector('header select'));
  const measured = await page.evaluate(() => ({
    title:document.title, header:document.querySelector('header').innerText,
    model:document.querySelector('header select').value,
    styles:[...document.querySelectorAll('link[rel~=stylesheet]')].filter(link=>new URL(link.href).origin===location.origin).map(link=>({href:link.href,enabled:!!link.sheet && !link.sheet.disabled,rules:link.sheet?.cssRules.length??0})),
    margin:getComputedStyle(document.body).margin,
    overflow:getComputedStyle(document.getElementById('root')).overflow,
    display:getComputedStyle(document.querySelector('header')).display,
  }));
  console.log(JSON.stringify({...measured,responses,external,errors},null,2));
} finally { await browser.close(); }
'''

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.css = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'link' and 'stylesheet' in attrs.get('rel', '').split():
            href = attrs.get('href', '')
            if href.startswith('/'):
                self.css.append(href)

def manifest(dist):
    return {str(p.relative_to(dist)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(dist.rglob('*')) if p.is_file()}

def run(command, cwd, name, timeout=120):
    result = subprocess.run(command, cwd=cwd, env=ENV, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout)
    (OUT / (name + '.txt')).write_text(f'Command: {command!r}\nExit: {result.returncode}\n\n{result.stdout}')
    return result

def observe(probe, name):
    (probe / 'observer.mjs').write_text(OBSERVER)
    with (OUT / (name + '-server.txt')).open('w') as server_log:
        server = subprocess.Popen(['node', str(ROOT / 'node_modules/vite/bin/vite.js'), 'preview', '--host', '127.0.0.1', '--port', '4173', '--strictPort'], cwd=probe, env=ENV, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            for _ in range(150):
                if server.poll() is not None:
                    raise RuntimeError('Independent preview exited unexpectedly')
                try:
                    urllib.request.urlopen('http://127.0.0.1:4173/', timeout=1).close()
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                raise RuntimeError('Independent preview did not start')
            result = run(['node', 'observer.mjs'], probe, name + '-observer')
            assert result.returncode == 0, result.stdout
            measured = json.loads(result.stdout)
            (OUT / (name + '-observation.json')).write_text(json.dumps(measured, indent=2) + '\n')
            return measured
        finally:
            server.terminate()
            server.wait(timeout=10)
            (probe / 'observer.mjs').unlink()

def assert_measurement(kind, data):
    assert data['title'] == 'Webcam Vision & Segmentation Studio'
    assert 'VisionStudio AI' in data['header']
    assert data['model'] == 'selfie-segmenter'
    assert any(r['resource'] == 'script' and '/assets/' in r['url'] and r['status'] == 200 for r in data['responses'])
    assert any('cdn.jsdelivr.net' in url for url in data['external'])
    css = [r for r in data['responses'] if r['resource'] == 'stylesheet' and r['url'].startswith('http://127.0.0.1:4173/')]
    if kind == 'healthy':
        assert css and all(r['status'] == 200 and r['mime'].startswith('text/css') for r in css)
        assert data['styles'] and all(s['rules'] > 0 and s['enabled'] for s in data['styles'])
        assert (data['margin'],data['overflow'],data['display']) == ('0px','hidden','flex')
    elif kind in ('missing-href', 'deleted-asset'):
        assert css and css[0]['status'] == 200 and css[0]['mime'].startswith('text/html')
        assert data['styles'][0]['rules'] == 0 and data['margin'] == '8px'
    elif kind == 'no-links':
        assert not css and not data['styles'] and data['margin'] == '8px'
    elif kind in ('empty', 'invalid'):
        assert css[0]['mime'].startswith('text/css') and css[0]['status'] == 200
        assert data['styles'][0]['rules'] == 0 and data['margin'] == '8px'
    elif kind == 'wrong-mime':
        assert css[0]['mime'].startswith('text/plain') and css[0]['status'] == 200
    elif kind == 'unrelated':
        assert css[0]['mime'].startswith('text/css') and data['styles'][0]['rules'] > 0 and data['margin'] == '8px'
    elif kind == 'body':
        assert data['margin'] == '13px' and data['overflow'] == 'hidden' and data['display'] == 'flex'
    elif kind == 'root':
        assert data['overflow'] == 'visible' and data['margin'] == '0px' and data['display'] == 'flex'
    elif kind == 'header':
        assert data['display'] == 'block' and data['margin'] == '0px' and data['overflow'] == 'hidden'
    elif kind == 'fatal':
        assert 'Independent fatal sentinel' in data['errors']
    if kind != 'fatal':
        assert data['errors'] == []

def trial(source, kind, expected_exit, diagnostic, prefix):
    name = prefix + '-' + kind
    with tempfile.TemporaryDirectory(prefix='vision-css-independent-') as directory:
        probe = Path(directory)
        for file in ['package.json', 'playwright.config.ts', 'vite.config.ts', 'e2e', 'dist']:
            source_file = source / file
            if source_file.is_dir():
                shutil.copytree(source_file, probe / file)
            else:
                shutil.copy2(source_file, probe / file)
        (probe / 'node_modules').symlink_to(ROOT / 'node_modules', target_is_directory=True)
        html_path = probe / 'dist/index.html'
        html = html_path.read_text()
        parser = Links()
        parser.feed(html)
        assert len(parser.css) == 1, 'Current production contract has one stylesheet'
        href = parser.css[0]
        css_path = probe / 'dist' / href.lstrip('/')
        if kind == 'missing-href':
            # Exact mutation from original PR #2; generated href discovered, never hardcoded.
            html = html.replace(href, '/assets/review-missing.css')
        elif kind == 'deleted-asset':
            css_path.unlink()
        elif kind == 'no-links':
            html = re.sub(r'<link\b[^>]*>', lambda match: '' if f'href="{href}"' in match[0] else match[0], html)
            assert href not in html
        elif kind in ('empty', 'invalid', 'unrelated'):
            css_path.write_text({'empty':'', 'invalid':'this is invalid CSS ???', 'unrelated':'.independent-never-used { color: red; }'}[kind])
        elif kind == 'wrong-mime':
            new_href = '/assets/independent-styles.txt'
            shutil.copyfile(css_path, probe / 'dist' / new_href.lstrip('/'))
            html = html.replace(href, new_href)
        elif kind in ('body', 'root', 'header'):
            rules = {'body':'body { margin: 13px !important; }', 'root':'#root { overflow: visible !important; }', 'header':'header { display: block !important; }'}
            css_path.write_text(css_path.read_text() + '\n' + rules[kind])
        elif kind == 'fatal':
            html = html.replace('</head>', '<script>setTimeout(() => {throw new Error("Independent fatal sentinel");}, 0)</script></head>')
        html_path.write_text(html)
        data = observe(probe, name)
        assert_measurement(kind, data)
        result = run(['npm', 'run', 'test:e2e', '--', '--reporter=json'], probe, name + '-smoke')
        assert result.returncode == expected_exit, f'{name}: actual {result.returncode}, wanted {expected_exit}'
        report = json.loads(result.stdout[result.stdout.index('{'):])
        errors = [error.get('message', '') for suite in report['suites'] for spec in suite['specs'] for test in spec['tests'] for result in test['results'] for error in result['errors']]
        if expected_exit:
            assert report['stats']['unexpected'] == 1 and report['stats']['expected'] == 0
            assert any(diagnostic in error for error in errors), errors
        else:
            assert report['stats']['expected'] == 1 and report['stats']['unexpected'] == 0
        row = dict(name=name,source=ORIGINAL if prefix=='original' else FIX,mutation=kind,exit=result.returncode,expected_exit=expected_exit,diagnostic=diagnostic,passed=True)
        print(json.dumps(row), flush=True)
        return row

def main():
    before = manifest(ROOT / 'dist')
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix='vision-css-original-') as directory:
            original = Path(directory)
            archive = subprocess.check_output(['git', 'archive', ORIGINAL], cwd=ROOT)
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(original, filter='data')
            (original / 'node_modules').symlink_to(ROOT / 'node_modules', target_is_directory=True)
            built = run(['npm', 'run', 'build'], original, 'original-build')
            assert built.returncode == 0
            original_manifest = manifest(original / 'dist')
            (OUT / 'original-build-sha256.json').write_text(json.dumps(original_manifest, indent=2) + '\n')
            # Tailwind scans repository text; evidence can add unused utilities.
            # Record output differences rather than assuming source equality means
            # byte-identical output across source trees and archive locations.
            for suffix in ['.css', '.js']:
                old_files = list((original / 'dist/assets').glob('*' + suffix))
                new_files = list((ROOT / 'dist/assets').glob('*' + suffix))
                if len(old_files) == len(new_files) == 1:
                    old = old_files[0].read_text()
                    new = new_files[0].read_text()
                    import difflib
                    difference = '\n'.join(difflib.unified_diff(old.replace('}', '}\n').splitlines(), new.replace('}', '}\n').splitlines(), fromfile='original'+suffix, tofile='fixed'+suffix))
                    (OUT / ('original-vs-fixed' + suffix + '.diff')).write_text(difference + '\n')
            results.append(trial(original, 'healthy', 0, None, 'original'))
            results.append(trial(original, 'missing-href', 0, 'Original F01 reproduced: unstyled app passed', 'original'))
        cases = [
            ('healthy', 0, None),
            ('missing-href', 1, 'required stylesheet MIME'),
            ('deleted-asset', 1, 'required stylesheet MIME'),
            ('no-links', 1, 'required local stylesheet links'),
            ('empty', 1, 'required stylesheet parsed rules'),
            ('invalid', 1, 'required stylesheet parsed rules'),
            ('wrong-mime', 1, 'required stylesheet MIME'),
            ('unrelated', 1, 'required stylesheet applied body reset'),
            ('body', 1, 'required stylesheet applied body reset'),
            ('root', 1, 'required stylesheet applied overflow'),
            ('header', 1, 'required stylesheet applied header layout'),
            ('fatal', 1, 'Independent fatal sentinel'),
            ('healthy', 0, None),
        ]
        for index, (kind, code, diagnostic) in enumerate(cases):
            results.append(trial(ROOT, kind, code, diagnostic, f'fixed-{index:02d}'))
    finally:
        unchanged = before == manifest(ROOT / 'dist')
        (OUT / 'independent-css-results.json').write_text(json.dumps(dict(results=results,build_unchanged=unchanged,build_sha256=before), indent=2) + '\n')
        assert unchanged, 'Root production build changed'

if __name__ == '__main__':
    main()
