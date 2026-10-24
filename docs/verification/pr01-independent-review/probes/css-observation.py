#!/usr/bin/env python3
"""Observe real production CSS loading, then a missing required CSS URL."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / 'logs'
JAVASCRIPT = """
import { chromium } from 'playwright';
const browser = await chromium.launch({executablePath:'/usr/bin/chromium'});
const context = await browser.newContext();
await context.route('**/*', route => new URL(route.request().url()).origin === 'http://127.0.0.1:4173' ? route.continue() : route.abort());
await context.addInitScript(() => Object.defineProperty(navigator, 'mediaDevices', {configurable:true,value:{getUserMedia:() => Promise.reject(new DOMException('Review denied','NotAllowedError'))}}));
const page = await context.newPage();
const cssResponses = [], errors = [], pageErrors = [];
page.on('response', r => { if(r.url().endsWith('.css')) cssResponses.push({url:r.url(), status:r.status(), contentType:r.headers()['content-type']}); });
page.on('console', msg => { if(msg.type() === 'error') errors.push(msg.text()); });
page.on('pageerror', e => pageErrors.push(e.message));
await page.goto('http://127.0.0.1:4173/');
await page.getByRole('banner').getByText('VisionStudio AI').waitFor();
await page.getByRole('heading', {name:'カメラエラー'}).waitFor();
console.log(JSON.stringify({cssResponses, errors, pageErrors, appearance:await page.evaluate(() => ({bodyBackground:getComputedStyle(document.body).backgroundColor, bodyMargin:getComputedStyle(document.body).margin, cssRuleCounts:[...document.styleSheets].map(s => {try{return s.cssRules.length}catch{return null}})}))}));
await browser.close();
"""

if __name__ == '__main__':
    results = {}
    with tempfile.TemporaryDirectory(prefix='pr01-review-css-') as directory:
        probe = Path(directory)
        for filename in ['package.json', 'vite.config.ts']:
            shutil.copy(ROOT / filename, probe / filename)
        (probe / 'node_modules').symlink_to(ROOT / 'node_modules', target_is_directory=True)
        shutil.copytree(Path(os.environ.get('PR01_REVIEW_DIST', ROOT / 'dist')), probe / 'dist')
        html = probe / 'dist/index.html'
        original = html.read_text()
        for label in ['original', 'missing_required_css']:
            if label != 'original':
                broken, count = re.subn(r'href="/assets/[^" ]+\.css"', 'href="/assets/review-missing.css"', original)
                assert count == 1
                html.write_text(broken)
            server = subprocess.Popen(['npm', 'run', 'preview', '--', '--host', '127.0.0.1', '--port', '4173', '--strictPort'], cwd=probe, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            try:
                for _ in range(100):
                    try:
                        urllib.request.urlopen('http://127.0.0.1:4173/', timeout=1).close()
                        break
                    except OSError:
                        time.sleep(0.1)
                output = subprocess.check_output(['node', '--input-type=module'], input=JAVASCRIPT, cwd=probe, text=True, timeout=30)
                results[label] = json.loads(output)
            finally:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=10)
    (OUT / 'css-observation.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))
