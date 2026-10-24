import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { cpSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const probe = mkdtempSync(path.join(tmpdir(), 'vision-edge-css-controls-'));
const html = readFileSync(path.join(root, 'dist/index.html'), 'utf8');
const links = (html.match(/<link\b[^>]*\brel="stylesheet"[^>]*>/g) ?? [])
  .filter(link => link.match(/href="([^"]+)"/)?.[1].startsWith('/'));
assert(links.length > 0, 'Build must contain stylesheet links before running controls');

// Each child executes the unchanged production smoke against a disposable build.
// Require its specific assertion failure, so a server/browser crash cannot pass
// a negative control. The intact build must pass first and again after mutations.
const controls = [
  { name: 'intact' },
  { name: 'missing-css-html-fallback', failure: 'required stylesheet MIME' },
  { name: 'absent-stylesheet-link', failure: 'required local stylesheet links' },
  { name: 'empty-css', failure: 'required stylesheet parsed rules' },
  { name: 'irrelevant-css', failure: 'required stylesheet applied body reset' },
  { name: 'fatal-javascript', failure: 'F01 fatal JavaScript control' },
  { name: 'intact-after-controls' },
];

try {
  for (const file of ['package.json', 'playwright.config.ts', 'vite.config.ts', 'e2e', 'index.html', 'src']) {
    cpSync(path.join(root, file), path.join(probe, file), { recursive: true });
  }
  symlinkSync(path.join(root, 'node_modules'), path.join(probe, 'node_modules'), 'junction');
  for (const control of controls) {
    const dist = path.join(probe, 'dist');
    rmSync(dist, { recursive: true, force: true });
    cpSync(path.join(root, 'dist'), dist, { recursive: true });
    let mutated = html;
    for (const [index, link] of links.entries()) {
      if (control.name === 'missing-css-html-fallback') {
        mutated = mutated.replace(link, link.replace(/href="[^"]+"/, `href="/assets/f01-missing-${index}.css"`));
      } else if (control.name === 'absent-stylesheet-link') {
        mutated = mutated.replace(link, '');
      } else if (control.name === 'empty-css' || control.name === 'irrelevant-css') {
        const href = link.match(/href="([^"]+)"/)?.[1];
        assert(href?.startsWith('/'), 'Expected a local production stylesheet URL');
        const file = path.resolve(dist, `.${href}`);
        assert(file.startsWith(`${dist}${path.sep}`), 'Stylesheet must stay inside disposable dist');
        writeFileSync(file, control.name === 'empty-css' ? '' : ':root { --f01-control: 1; }');
      }
    }
    if (control.name === 'fatal-javascript') {
      mutated = mutated.replace('</head>', '<script>setTimeout(() => { throw new Error("F01 fatal JavaScript control"); }, 0);</script></head>');
    }
    writeFileSync(path.join(dist, 'index.html'), mutated);
    // Controls exercise exactly the production smoke, even when other e2e files
    // are added. Ordinary CI still runs the complete browser test suite.
    const result = spawnSync(process.execPath, [path.join(root, 'node_modules/@playwright/test/cli.js'), 'test', 'e2e/smoke.spec.ts', '--retries=0', '--reporter=json'], {
      cwd: probe,
      env: process.env,
      encoding: 'utf8',
      timeout: 90_000,
    });
    assert.ifError(result.error);
    const report = JSON.parse(result.stdout);
    const errors = report.suites.flatMap(suite => suite.specs.flatMap(spec =>
      spec.tests.flatMap(test => test.results.flatMap(result => result.errors.map(error => error.message ?? '')))));
    console.log(`${control.name}: exit=${result.status}; passed=${report.stats.expected}; failed=${report.stats.unexpected}`);
    if (control.failure) {
      assert.equal(result.status, 1, `${control.name}: expected smoke assertion failure`);
      assert.equal(report.stats.unexpected, 1, `${control.name}: expected exactly one failed smoke`);
      assert(errors.some(message => message.includes(control.failure)), `${control.name}: missing diagnostic ${control.failure}\n${errors.join('\n')}\n${result.stderr}`);
      console.log(`  Expected failure confirmed: ${control.failure}`);
    } else {
      assert.equal(result.status, 0, `${control.name}: intact build must pass\n${errors.join('\n')}\n${result.stderr}`);
      assert.equal(report.stats.expected, 1, `${control.name}: expected one passed smoke`);
    }
  }
} finally {
  rmSync(probe, { recursive: true, force: true });
}
