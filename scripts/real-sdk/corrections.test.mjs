// All records and temporary bytes below are CHECKER-ONLY counterexamples.
// No model is acquired, no SDK runs, and no records reach App/VisionEngine.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, cp, symlink, rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { root, manifest, validateManifest } from './assets.mjs';
import { summarizeRun } from './summarize.mjs';
import { checkNegativeScene } from '../../e2e-real-sdk/acceptance.ts';

const edits = {
  empty: m => { m.assets = []; },
  'unsupported schema': m => { m.schema = 999; },
  'missing assets array': m => { delete m.assets; },
  'SDK only': m => { m.assets = m.assets.filter(a => a.packageFile); },
  duplicate: m => { m.assets.push(m.assets[0]); },
  'unknown identity': m => { m.assets[0].id = 'unknown'; },
  'duplicate URL': m => { m.assets[0].url = m.assets[1].url; },
  'duplicate file': m => { m.assets[0].file = m.assets[1].file; },
  'alternate source URL': m => { m.assets[0].url = 'https://example.invalid/alternate'; },
  'different version': m => { m.assets[0].version = '999'; },
  'file traversal': m => { m.assets[0].file = '../outside'; },
  'package traversal': m => { m.assets.at(-1).packageFile = '../outside'; },
  'required false': m => { m.assets[0].required = false; },
  'half pin': m => { m.assets[0].size = null; },
  'bad hash': m => { m.assets[0].sha256 = 'x'.repeat(64); },
  'size string': m => { m.assets[0].size = '4'; },
  'size zero': m => { m.assets[0].size = 0; },
  'oversized pin': m => { m.assets[0].size = 15_000_001; },
  'unversioned nonmodel': m => { m.assets[0].size = null; m.assets[0].sha256 = null; m.assets[0].acquisitionStatus = 'blocked: control'; },
  'unexplained null model': m => { delete m.assets.find(a => a.id.endsWith('.tflite')).acquisitionStatus; },
};
for (const id of manifest.assets.map(a => a.id)) edits[`missing ${id}`] = m => { m.assets = m.assets.filter(a => a.id !== id); };
for (const [name, edit] of Object.entries(edits)) test(`F01 CLI rejects ${name} before byte/network work`, async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), 'pr07-f01-'));
  try {
    await mkdir(path.join(temporary, 'scripts/real-sdk'), { recursive: true }); await mkdir(path.join(temporary, 'e2e-real-sdk'));
    await cp(path.join(root, 'scripts/real-sdk/assets.mjs'), path.join(temporary, 'scripts/real-sdk/assets.mjs'));
    const changed = structuredClone(manifest); edit(changed);
    await writeFile(path.join(temporary, 'e2e-real-sdk/assets.lock.json'), JSON.stringify(changed));
    const result = spawnSync(process.execPath, ['scripts/real-sdk/assets.mjs'], { cwd: temporary, encoding: 'utf8', timeout: 5000 });
    assert.equal(result.status, 1); assert.match(result.stderr, /INVALID_ASSET_MANIFEST:/); assert.equal(result.stdout, '');
  } finally { await rm(temporary, { recursive: true, force: true }); }
});
test('F01 valid inventory with unavailable/explicitly null models exits2, not0/1', async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), 'pr07-f01-valid-'));
  try {
    await mkdir(path.join(temporary, 'scripts/real-sdk'), { recursive: true }); await mkdir(path.join(temporary, 'e2e-real-sdk'));
    await cp(path.join(root, 'scripts/real-sdk/assets.mjs'), path.join(temporary, 'scripts/real-sdk/assets.mjs'));
    await symlink(path.join(root, 'node_modules'), path.join(temporary, 'node_modules'));
    await writeFile(path.join(temporary, 'e2e-real-sdk/assets.lock.json'), JSON.stringify(manifest));
    const result = spawnSync(process.execPath, ['scripts/real-sdk/assets.mjs'], { cwd: temporary, encoding: 'utf8', timeout: 5000 });
    assert.equal(result.status, 2); const report = JSON.parse(result.stdout); assert.equal(report.rows.length, 9);
    assert(report.rows.filter(a => a.id.endsWith('.tflite')).every(a => a.status === 'blocked'));
    assert.equal(report.rows.find(a => a.id === 'vision_bundle.mjs').status, 'pass');
    // Schema eligibility only: these pins are never stored or supplied to SDK.
    const pinned = structuredClone(manifest);
    for (const asset of pinned.assets.filter(a => a.id.endsWith('.tflite'))) { asset.size = 4; asset.sha256 = 'a'.repeat(64); }
    assert.equal(validateManifest(pinned), pinned);
  } finally { await rm(temporary, { recursive: true, force: true }); }
});

const stamp = { scene: 'empty', revision: 3, stream: 2 };
const boundary = { afterCallId: 10, afterResultId: 10, model: 'selfie-segmenter', input: 'VideoFrame', requestId: 7, stamp };
function successful() {
  const raw = Array.from({ length: 5 }, (_, index) => ({ id: index + 11, model: boundary.model, input: boundary.input, requestId: 7, stamp: { ...stamp }, sdkSucceeded: true, ms: 2 }));
  const processed = raw.map(row => ({ id: row.id + 100, rawId: row.id, model: row.model, input: row.input, requestId: row.requestId, stamp: { ...stamp }, status: 'ready', objects: [] }));
  return { calls: 15, sdkSuccesses: 5, raw, processed, adoptions: [{ id: 115 }], draws: [{ id: 115, alphaPixels: 0, width: 512, height: 512 }], creates: [], rawDropped: 10, snapshots: [] };
}
test('F02 checker-only five fresh successes plus processing/adoption/empty rendering are eligible', () => {
  const probe = successful(); probe.processed[0].objects.push({ label: 'person' }); probe.processed[1].objects.push({ label: 'person' });
  assert.equal(checkNegativeScene(probe, boundary, 'person').status, 'pass');
});
const counterexamples = {
  'published stale empty record plus ten failed attempts': p => { p.raw = Array.from({ length: 10 }, (_, i) => ({ ...p.raw[0], id: i + 11, sdkSucceeded: false, error: 'CHECKER CONTROL failure' })); p.processed = [{ ...p.processed[0], id: 1, rawId: 1 }]; },
  'no new SDK calls': p => { p.raw = []; },
  'attempts without known success': p => { p.raw.forEach(row => { delete row.sdkSucceeded; }); },
  'four successes only': p => { p.raw.pop(); },
  'wrong model': p => { p.raw.at(-1).model = 'mediapipe'; },
  'wrong input': p => { p.raw.at(-1).input = 'Canvas'; },
  'wrong generation': p => { p.raw.at(-1).requestId = 8; },
  'previous scene': p => { p.raw.forEach(row => { row.stamp.scene = 'background'; }); },
  'previous transition': p => { p.raw.forEach(row => { row.stamp.revision--; }); },
  'previous camera session': p => { p.raw.forEach(row => { row.stamp.stream--; }); },
  'unknown actual input scene': p => { p.raw.forEach(row => { row.stamp = null; }); },
  'missing raw identity': p => { p.raw[0].id = 12; },
  'duplicate processed identity': p => { p.processed.at(-1).id = p.processed[0].id; },
  'duplicate raw identity': p => { p.raw.at(-1).id = p.raw[0].id; },
  'stale processed ids': p => { p.processed.forEach(row => { row.id = 1; }); },
  'unmatched raw result': p => { p.processed.at(-1).rawId = 1; },
  'processed generation mismatch': p => { p.processed.at(-1).requestId = 8; },
  'processed stamp mismatch': p => { p.processed.at(-1).stamp.revision--; },
  'no fresh ready output': p => { p.processed.forEach(row => { row.status = 'not-ready'; }); },
  'unexpected label after retention': p => { p.processed.at(-1).objects.push({ label: 'person' }); },
  'no adoption': p => { p.adoptions = []; },
  'stale adoption': p => { p.adoptions[0].id = 1; },
  'no rendering': p => { p.draws = []; },
  'stale rendering': p => { p.draws[0].id = 1; },
  'failed later attempt after five successes': p => { p.raw.push({ ...p.raw[0], id: 16, sdkSucceeded: false, error: 'CHECKER CONTROL failure' }); },
};
for (const [name, edit] of Object.entries(counterexamples)) test(`F02 cannot accept ${name}`, () => { const probe = successful(); edit(probe); assert.notEqual(checkNegativeScene(probe, boundary, 'person').status, 'pass'); });

for (const scenario of ['zero-call startup failure', 'failed attempts only', 'unknown SDK outcome', 'SDK return without processing', 'processed without adoption', 'adopted without draw', 'complete chain', 'stale ready/adopt/draw', 'invalid counters', 'counter-only invented success']) test(`F03 summary separates stages: ${scenario}`, async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), 'pr07-f03-'));
  const p = successful();
  if (scenario === 'zero-call startup failure') Object.assign(p, { calls: 0, sdkSuccesses: 0, raw: [], processed: [], adoptions: [], draws: [], rawDropped: 0, creates: [{ model: 'selfie-segmenter', error: 'CHECKER startup failure' }] });
  if (scenario === 'failed attempts only') { p.sdkSuccesses = 0; p.raw.forEach(row => { row.sdkSucceeded = false; row.error = 'CHECKER failed call'; }); }
  if (scenario === 'unknown SDK outcome') { delete p.sdkSuccesses; p.raw.forEach(row => { delete row.sdkSucceeded; }); }
  if (scenario === 'SDK return without processing') p.processed = [];
  if (scenario === 'processed without adoption') p.adoptions = [];
  if (scenario === 'adopted without draw') p.draws = [];
  if (scenario === 'stale ready/adopt/draw') p.processed.forEach(row => { row.rawId = 1; });
  if (scenario === 'invalid counters') p.calls = 0;
  if (scenario === 'counter-only invented success') Object.assign(p, { calls: 1, sdkSuccesses: 1, raw: [], rawDropped: 0 });
  try {
    const dir = path.join(temporary, 'browser/control'); await mkdir(dir, { recursive: true });
    await writeFile(path.join(dir, 'evidence.json'), JSON.stringify({ runtime: { kind: 'CHECKER ONLY; NO SDK/MODEL' }, evidence: { probe: p } }));
    await summarizeRun(temporary); const result = JSON.parse(await readFile(path.join(temporary, 'measurements.json'), 'utf8'));
    const executed = !['zero-call startup failure', 'failed attempts only', 'unknown SDK outcome', 'invalid counters', 'counter-only invented success'].includes(scenario);
    assert.equal(result.inferenceExecuted, executed);
    assert.equal(result.inferenceAttempted, !['zero-call startup failure', 'invalid counters', 'counter-only invented success'].includes(scenario));
    assert.equal(result.resultAdopted, ['adopted without draw', 'complete chain'].includes(scenario));
    assert.equal(result.resultRendered, scenario === 'complete chain');
    if (!executed) assert.equal(result.populations.length, 0);
  } finally { await rm(temporary, { recursive: true, force: true }); }
});
