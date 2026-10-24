// Acceptance-logic counterexamples only. NO SDK/model call, input or inference result.
import assert from 'node:assert/strict';
import { readFile, mkdir, writeFile, mkdtemp, rm } from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL } from 'node:url';
const source = process.argv[2] ?? '/tmp/pr07-review-current';
const { default: ts } = await import(pathToFileURL(path.join(source, 'node_modules/typescript/lib/typescript.js')));
const { expect } = await import(pathToFileURL(path.join(source, 'node_modules/@playwright/test/index.mjs')));
const spec = await readFile(path.join(source, 'e2e-real-sdk/integration.spec.ts'), 'utf8');
const start = spec.indexOf("for (const scene of ['background', 'empty'])");
const end = spec.indexOf('const other =', start);
assert(start > 0 && end > start, 'Reinspect changed negative-control block');
const snippet = spec.slice(start, end);
const js = ts.transpileModule(snippet, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
const run = new (Object.getPrototypeOf(async function () {}).constructor)('page', 'expect', 'expected', js);
// A stale previous empty processed record; all subsequent attempted calls fail.
// This intentionally fabricated CHECKER state is never fed into a VisionEngine or SDK.
const probe = { calls: 10, raw: [], processed: [{ id: 1, status: 'ready', objects: [] }] };
globalThis.window = { __realCamera: { scene: 'previous' }, __realSdkProbe: probe };
const steps = [];
const page = { evaluate: async (fn, arg) => {
  if (String(fn).includes('window.__realSdkProbe.calls - before')) {
    for (let n = 0; n < 5; n++) { probe.calls++; probe.raw.push({ id: probe.calls, scene: window.__realCamera.scene, error: 'CHECKER CONTROL: failed attempted call; no SDK executed' }); }
  }
  const result = fn(arg); steps.push({ scene: window.__realCamera.scene, function: String(fn), result }); return result;
} };
await run(page, expect, 'person');
assert.equal(probe.raw.length, 10); assert.equal(probe.processed.length, 1);
console.log(JSON.stringify({ control: 'negative-scene acceptance predicate', extractedSource: snippet, checkerAccepted: true, successfulCallsInEitherNewScene: 0, freshProcessedResults: 0, steps, limitation: 'Counterexample to checker logic only, NOT execution of the full six tests, NOT real SDK/model success or product failure.' }, null, 2));
delete globalThis.window;
const { summarizeRun } = await import(pathToFileURL(path.join(source, 'scripts/real-sdk/summarize.mjs')));
const temporary = await mkdtemp(path.join(tmpdir(), 'pr07-empty-summary-'));
try {
  await mkdir(path.join(temporary, 'browser/startup-failed'), { recursive: true });
  await writeFile(path.join(temporary, 'browser/startup-failed/evidence.json'), JSON.stringify({ runtime: { control: 'zero SDK calls after initialization failure' }, evidence: { probe: { raw: [], creates: [{model:'selfie-segmenter',error:'initialization failed'}], snapshots: [], rawDropped: 0 } } }));
  await summarizeRun(temporary);
  const actual = JSON.parse(await readFile(path.join(temporary, 'measurements.json'), 'utf8'));
  assert.equal(actual.inferenceExecuted, true); assert(actual.populations.every(p => p.retainedSdkCalls === 0));
  console.log(JSON.stringify({ control: 'zero-call summary', actual, expectedInferenceExecuted: false, limitation: 'Summary logic fixture only; NO model bytes or inference executed.' }, null, 2));
} finally { await rm(temporary, { recursive: true, force: true }); }
